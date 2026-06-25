//! Power & energy management (see `docs/architecture/power-and-energy-management.md`).
//!
//! Energy is a first-class, capability-accounted resource — essential for the
//! mobile target where battery is the budget. This module models:
//!
//! * **System power states** (Active → Idle → Sleep → DeepSleep) with legal
//!   transitions, so the kernel can wind the machine down and bring it back.
//! * **Per-domain energy accounting**: every domain draws against a budget; a
//!   domain that exceeds its cap is throttled rather than allowed to flatten the
//!   battery. This is the energy analogue of the capability firewall.
//! * A **battery model** and a **governor** that selects a power state from the
//!   current load and battery level.
//!
//! Pure, safe `no_std`; on a desktop the battery is simply "mains" (always full).

use alloc::vec::Vec;

/// Energy unit — abstract "micro-joules" the OS accounts in.
pub type Energy = u64;

/// System-wide power state, deepest sleep last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerState {
    Active,
    Idle,
    Sleep,
    DeepSleep,
}

impl PowerState {
    /// Relative power draw multiplier (×1000) for this state.
    fn draw_milli(self) -> Energy {
        match self {
            PowerState::Active => 1000,
            PowerState::Idle => 250,
            PowerState::Sleep => 30,
            PowerState::DeepSleep => 2,
        }
    }

    /// Legal transitions — you may step one level at a time, or wake fully.
    pub fn can_transition_to(self, to: PowerState) -> bool {
        use PowerState::*;
        match (self, to) {
            (a, b) if a == b => false,
            // Wake to Active is always allowed (e.g. an interrupt).
            (_, Active) => true,
            // Descend one step at a time.
            (Active, Idle) | (Idle, Sleep) | (Sleep, DeepSleep) => true,
            // Ascend one step at a time (waking fully to Active is handled above).
            (DeepSleep, Sleep) | (Sleep, Idle) => true,
            _ => false,
        }
    }
}

/// Per-domain energy budget and running consumption.
#[derive(Clone, Debug)]
struct DomainBudget {
    id: u64,
    cap: Energy,
    used: Energy,
    throttled: bool,
}

/// A simple battery: current charge out of capacity. Mains power is modelled as a
/// battery that never depletes.
#[derive(Clone, Debug)]
pub struct Battery {
    pub capacity: Energy,
    pub charge: Energy,
    pub on_mains: bool,
}

impl Battery {
    pub fn new(capacity: Energy) -> Battery {
        Battery { capacity, charge: capacity, on_mains: false }
    }

    pub fn mains() -> Battery {
        Battery { capacity: Energy::MAX, charge: Energy::MAX, on_mains: true }
    }

    /// Fraction of charge remaining, 0..=1000 (per-mille, to stay integer).
    pub fn level_milli(&self) -> Energy {
        if self.capacity == 0 {
            return 0;
        }
        self.charge.saturating_mul(1000) / self.capacity
    }

    fn drain(&mut self, e: Energy) {
        if !self.on_mains {
            self.charge = self.charge.saturating_sub(e);
        }
    }

    pub fn recharge(&mut self, e: Energy) {
        self.charge = (self.charge + e).min(self.capacity);
    }
}

/// The power manager: tracks state, per-domain budgets, and the battery.
pub struct PowerManager {
    state: PowerState,
    domains: Vec<DomainBudget>,
    pub battery: Battery,
    total_used: Energy,
}

impl PowerManager {
    pub fn new(battery: Battery) -> PowerManager {
        PowerManager {
            state: PowerState::Active,
            domains: Vec::new(),
            battery,
            total_used: 0,
        }
    }

    pub fn state(&self) -> PowerState {
        self.state
    }

    /// Request a power-state transition; rejected if illegal.
    pub fn transition(&mut self, to: PowerState) -> bool {
        if self.state.can_transition_to(to) {
            self.state = to;
            true
        } else {
            false
        }
    }

    /// Give a domain an energy budget (idempotent: re-registering resets the cap).
    pub fn set_budget(&mut self, id: u64, cap: Energy) {
        if let Some(d) = self.domains.iter_mut().find(|d| d.id == id) {
            d.cap = cap;
        } else {
            self.domains.push(DomainBudget { id, cap, used: 0, throttled: false });
        }
    }

    /// Charge `e` energy to a domain's account, scaled by the current power state.
    /// Returns `false` if the domain is over budget and must be throttled — the
    /// energy is still recorded so accounting stays accurate.
    pub fn charge(&mut self, id: u64, e: Energy) -> bool {
        let scaled = e.saturating_mul(self.state.draw_milli()) / 1000;
        self.battery.drain(scaled);
        self.total_used = self.total_used.saturating_add(scaled);
        if let Some(d) = self.domains.iter_mut().find(|d| d.id == id) {
            d.used = d.used.saturating_add(scaled);
            if d.used > d.cap {
                d.throttled = true;
                return false;
            }
            true
        } else {
            true // un-budgeted domains are not throttled
        }
    }

    pub fn is_throttled(&self, id: u64) -> bool {
        self.domains.iter().find(|d| d.id == id).map(|d| d.throttled).unwrap_or(false)
    }

    /// Reset a domain's accounting window (e.g. each scheduling epoch).
    pub fn reset_window(&mut self, id: u64) {
        if let Some(d) = self.domains.iter_mut().find(|d| d.id == id) {
            d.used = 0;
            d.throttled = false;
        }
    }

    pub fn total_used(&self) -> Energy {
        self.total_used
    }

    /// Governor: choose a power state from the count of runnable domains and the
    /// battery level. No runnable work + low battery ⇒ deeper sleep.
    pub fn govern(&self, runnable: usize) -> PowerState {
        if runnable > 0 {
            return PowerState::Active;
        }
        let level = self.battery.level_milli();
        if self.battery.on_mains {
            PowerState::Idle
        } else if level < 50 {
            PowerState::DeepSleep
        } else if level < 200 {
            PowerState::Sleep
        } else {
            PowerState::Idle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_state_transitions_are_legal_only() {
        use PowerState::*;
        assert!(Active.can_transition_to(Idle));
        assert!(Idle.can_transition_to(Sleep));
        assert!(DeepSleep.can_transition_to(Active)); // wake
        assert!(!Active.can_transition_to(DeepSleep)); // cannot skip levels down
        assert!(!Active.can_transition_to(Active)); // no-op rejected
    }

    #[test]
    fn manager_enforces_legal_transitions() {
        let mut pm = PowerManager::new(Battery::mains());
        assert!(pm.transition(PowerState::Idle));
        assert_eq!(pm.state(), PowerState::Idle);
        assert!(!pm.transition(PowerState::DeepSleep)); // illegal skip
        assert!(pm.transition(PowerState::Sleep));
        assert!(pm.transition(PowerState::Active)); // wake
    }

    #[test]
    fn domain_over_budget_is_throttled() {
        let mut pm = PowerManager::new(Battery::new(1_000_000));
        pm.set_budget(1, 100);
        assert!(pm.charge(1, 60)); // within budget
        assert!(!pm.is_throttled(1));
        assert!(!pm.charge(1, 60)); // 120 > 100 → throttled
        assert!(pm.is_throttled(1));
        // Resetting the window clears the throttle.
        pm.reset_window(1);
        assert!(!pm.is_throttled(1));
        assert!(pm.charge(1, 50));
    }

    #[test]
    fn battery_drains_on_battery_not_on_mains() {
        let mut on_batt = PowerManager::new(Battery::new(10_000));
        on_batt.charge(0, 1_000);
        assert!(on_batt.battery.charge < 10_000);
        let mut on_mains = PowerManager::new(Battery::mains());
        let before = on_mains.battery.charge;
        on_mains.charge(0, 1_000);
        assert_eq!(on_mains.battery.charge, before); // mains never drains
    }

    #[test]
    fn deeper_state_draws_less_energy() {
        let mut a = PowerManager::new(Battery::new(1_000_000));
        a.charge(0, 1000); // Active draw
        let active_used = a.total_used();
        let mut s = PowerManager::new(Battery::new(1_000_000));
        s.transition(PowerState::Idle);
        s.transition(PowerState::Sleep);
        s.charge(0, 1000); // Sleep draw
        assert!(s.total_used() < active_used);
    }

    #[test]
    fn governor_picks_state_from_load_and_battery() {
        let mut pm = PowerManager::new(Battery::new(1000));
        assert_eq!(pm.govern(3), PowerState::Active); // work pending
        assert_eq!(pm.govern(0), PowerState::Idle); // idle, full battery
        // Drain to a low level → deeper sleep when idle.
        pm.battery.charge = 30; // 3% of 1000
        assert_eq!(pm.govern(0), PowerState::DeepSleep);
        pm.battery.charge = 150; // 15%
        assert_eq!(pm.govern(0), PowerState::Sleep);
    }
}

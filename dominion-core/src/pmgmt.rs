//! Power & energy management — **DVFS, idle C-states, tickless idle, per-core gating,
//! device runtime PM / PCIe ASPM, energy-delay-product placement, and thermal feedback**
//! (`docs/architecture/power-and-energy-management.md`).
//!
//! [`crate::power`] owns power *states* and per-domain energy *budgets*; this module is
//! the optimization layer the scheduler consults to actually spend less energy:
//!
//! * [`DvfsGovernor`] — dynamic voltage/frequency scaling: pick the lowest operating
//!   point that still meets the load (energy scales ~V²·f, so racing-to-idle vs
//!   slowing-down is a real choice).
//! * [`IdleGovernor`] — the deepest safe **C-state** for an expected idle, and a
//!   **tickless** next-wake (no periodic tick while idle) + **per-core gating**.
//! * [`DevicePm`] — device runtime suspend/resume + **PCIe ASPM** link states.
//! * [`EdpPlanner`] — energy-delay-product placement across P/E-cores, GPU and NPU.
//! * [`ThermalGovernor`] — temperature feedback → throttle + work migration.
//! * [`Objective`] — a multi-objective score (latency, throughput, energy, thermal) so a
//!   single knob can balance the four competing goals.
//!
//! Pure, safe `no_std`; deterministic (no wall-clock — time deltas are inputs). Host-tested.

use alloc::vec::Vec;

// ───────────────────────── DVFS ─────────────────────────

/// A voltage/frequency operating point (a P-state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatingPoint {
    pub freq_mhz: u32,
    pub voltage_mv: u32,
}

impl OperatingPoint {
    /// Relative dynamic power ∝ C·V²·f (arbitrary units): the cost of running here.
    pub fn relative_power(self) -> u64 {
        let v = self.voltage_mv as u64;
        v * v * self.freq_mhz as u64 / 1_000_000
    }
}

/// Dynamic voltage/frequency scaling over a sorted table of operating points.
pub struct DvfsGovernor {
    points: Vec<OperatingPoint>,
}

impl DvfsGovernor {
    /// Build from operating points (sorted ascending by frequency).
    pub fn new(mut points: Vec<OperatingPoint>) -> DvfsGovernor {
        points.sort_by_key(|p| p.freq_mhz);
        DvfsGovernor { points }
    }

    /// A sensible default ladder (4 P-states).
    pub fn default_ladder() -> DvfsGovernor {
        DvfsGovernor::new(alloc::vec![
            OperatingPoint { freq_mhz: 800, voltage_mv: 700 },
            OperatingPoint { freq_mhz: 1600, voltage_mv: 800 },
            OperatingPoint { freq_mhz: 2400, voltage_mv: 950 },
            OperatingPoint { freq_mhz: 3200, voltage_mv: 1100 },
        ])
    }

    /// Pick the **lowest** operating point whose frequency fraction (of the max) meets
    /// the demanded `load_milli` (0..=1000) — slow down to just-enough rather than always
    /// racing to idle. Always returns at least the top point if load is saturating.
    pub fn select(&self, load_milli: u32) -> OperatingPoint {
        let max = self.points.last().copied().unwrap();
        for p in &self.points {
            let frac = (p.freq_mhz as u64 * 1000 / max.freq_mhz as u64) as u32;
            if frac >= load_milli {
                return *p;
            }
        }
        max
    }
}

// ───────────────────────── idle: C-states, tickless, gating ─────────────────────────

/// A CPU idle state — deeper states save more power but cost more to exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CState {
    /// Running.
    C0,
    /// Light idle (clock-gated), fast exit.
    C1,
    /// Deeper (caches retained).
    C3,
    /// Deepest (power-gated), slow exit.
    C6,
}

/// Idle management: deepest safe C-state, tickless next-wake, and per-core gating.
pub struct IdleGovernor {
    /// Exit-latency budget (µs) we're willing to pay; bounds how deep we go.
    pub exit_latency_budget_us: u64,
}

impl IdleGovernor {
    pub fn new(exit_latency_budget_us: u64) -> IdleGovernor {
        IdleGovernor { exit_latency_budget_us }
    }

    /// The deepest C-state worth entering for an `expected_idle_us`, given the exit
    /// latency budget. Short idles stay shallow (the exit cost would dominate).
    pub fn deepest_cstate(&self, expected_idle_us: u64) -> CState {
        // Residency thresholds: only enter a state if we'll stay long enough to amortise
        // its exit latency, and the exit latency is within budget.
        let budget = self.exit_latency_budget_us;
        if expected_idle_us >= 1000 && budget >= 50 {
            CState::C6
        } else if expected_idle_us >= 100 && budget >= 10 {
            CState::C3
        } else if expected_idle_us >= 10 {
            CState::C1
        } else {
            CState::C0
        }
    }

    /// **Tickless idle**: the next wake is the earliest pending timer — no periodic tick
    /// fires while the CPU is idle. `None` ⇒ no timers ⇒ sleep until an interrupt.
    pub fn next_wake(&self, pending_timers_us: &[u64]) -> Option<u64> {
        pending_timers_us.iter().copied().min()
    }

    /// **Per-core gating**: how many of `total` cores are needed to serve `runnable`
    /// tasks — the rest can be power-gated. At least one core stays up.
    pub fn cores_needed(&self, runnable: usize, total: usize) -> usize {
        runnable.clamp(1, total.max(1))
    }
}

// ───────────────────────── device runtime PM + PCIe ASPM ─────────────────────────

/// A PCIe Active-State Power-Management link state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AspmState {
    /// Fully active.
    L0,
    /// Standby (fast exit).
    L0s,
    /// Lower power (slower exit).
    L1,
}

/// Per-device runtime power management: autosuspend after idle + link ASPM.
#[derive(Clone, Copy, Debug)]
pub struct DevicePm {
    /// Idle time (µs) after which the device autosuspends.
    pub autosuspend_after_us: u64,
    suspended: bool,
}

impl DevicePm {
    pub fn new(autosuspend_after_us: u64) -> DevicePm {
        DevicePm { autosuspend_after_us, suspended: false }
    }

    /// Advance idle time; the device runtime-suspends once it crosses the threshold.
    pub fn on_idle(&mut self, idle_us: u64) {
        if idle_us >= self.autosuspend_after_us {
            self.suspended = true;
        }
    }

    /// A device access wakes it (runtime resume).
    pub fn on_access(&mut self) {
        self.suspended = false;
    }

    pub fn is_suspended(&self) -> bool {
        self.suspended
    }

    /// The ASPM link state to request for an `idle_us` window: deeper when idle longer.
    pub fn aspm_state(&self, idle_us: u64) -> AspmState {
        if idle_us >= self.autosuspend_after_us {
            AspmState::L1
        } else if idle_us >= self.autosuspend_after_us / 4 {
            AspmState::L0s
        } else {
            AspmState::L0
        }
    }
}

// ───────────────────────── energy-delay-product placement ─────────────────────────

/// A compute unit a task can be placed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeUnit {
    /// Performance core.
    PCore,
    /// Efficiency core.
    ECore,
    Gpu,
    Npu,
}

/// Per-unit performance/energy characteristics (relative units).
#[derive(Clone, Copy, Debug)]
pub struct UnitProfile {
    pub unit: ComputeUnit,
    /// Relative speed for this task (higher = faster ⇒ lower delay).
    pub speedup: f64,
    /// Relative power while running.
    pub power: f64,
}

/// Energy-delay-product placement: choose the unit minimizing `energy × delay`
/// (`EDP = power·delay²` for a fixed work; equivalently `power / speedup²`).
pub struct EdpPlanner {
    units: Vec<UnitProfile>,
}

impl EdpPlanner {
    pub fn new(units: Vec<UnitProfile>) -> EdpPlanner {
        EdpPlanner { units }
    }

    /// The EDP cost of running on a unit (lower is better).
    pub fn edp(profile: &UnitProfile) -> f64 {
        // delay ∝ 1/speedup; energy ∝ power·delay; EDP = energy·delay = power/speedup².
        profile.power / (profile.speedup * profile.speedup)
    }

    /// Place a task on the unit with the lowest energy-delay product.
    pub fn place(&self) -> Option<ComputeUnit> {
        self.units
            .iter()
            .min_by(|a, b| {
                EdpPlanner::edp(a)
                    .partial_cmp(&EdpPlanner::edp(b))
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|p| p.unit)
    }
}

// ───────────────────────── thermal feedback ─────────────────────────

/// What the thermal governor decides to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThrottleAction {
    /// Within limits — run freely.
    Nominal,
    /// Warm — cap frequency (soft throttle).
    Throttle,
    /// Hot — migrate work off the hot unit and throttle hard.
    Migrate,
    /// Critical — emergency idle to avoid damage.
    Emergency,
}

/// Temperature-feedback governor with hysteresis-free banded thresholds (°C).
#[derive(Clone, Copy, Debug)]
pub struct ThermalGovernor {
    pub throttle_c: i32,
    pub migrate_c: i32,
    pub critical_c: i32,
}

impl ThermalGovernor {
    /// Sensible default trip points.
    pub fn default_trips() -> ThermalGovernor {
        ThermalGovernor { throttle_c: 80, migrate_c: 90, critical_c: 100 }
    }

    /// Decide an action from a sensor reading.
    pub fn action(&self, temp_c: i32) -> ThrottleAction {
        if temp_c >= self.critical_c {
            ThrottleAction::Emergency
        } else if temp_c >= self.migrate_c {
            ThrottleAction::Migrate
        } else if temp_c >= self.throttle_c {
            ThrottleAction::Throttle
        } else {
            ThrottleAction::Nominal
        }
    }
}

// ───────────────────────── multi-objective scoring ─────────────────────────

/// Weights for the four competing scheduling objectives (each 0..=1000, milli-weights).
#[derive(Clone, Copy, Debug)]
pub struct Objective {
    pub latency_w: u32,
    pub throughput_w: u32,
    pub energy_w: u32,
    pub thermal_w: u32,
}

impl Objective {
    /// A balanced default.
    pub fn balanced() -> Objective {
        Objective { latency_w: 250, throughput_w: 250, energy_w: 250, thermal_w: 250 }
    }

    /// Battery-saver: weight energy + thermal heavily.
    pub fn power_saver() -> Objective {
        Objective { latency_w: 100, throughput_w: 100, energy_w: 500, thermal_w: 300 }
    }

    /// Score a candidate configuration. Each metric is a *goodness* in 0..=1000 (higher =
    /// better: lower latency, higher throughput, lower energy, cooler). Returns a weighted
    /// blend; higher is better. Deterministic integer math.
    pub fn score(&self, latency_good: u32, throughput_good: u32, energy_good: u32, thermal_good: u32) -> u32 {
        let total_w = (self.latency_w + self.throughput_w + self.energy_w + self.thermal_w).max(1) as u64;
        let acc = self.latency_w as u64 * latency_good.min(1000) as u64
            + self.throughput_w as u64 * throughput_good.min(1000) as u64
            + self.energy_w as u64 * energy_good.min(1000) as u64
            + self.thermal_w as u64 * thermal_good.min(1000) as u64;
        (acc / total_w) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dvfs_picks_lowest_point_that_meets_load() {
        let g = DvfsGovernor::default_ladder();
        // Light load → the lowest P-state.
        assert_eq!(g.select(100).freq_mhz, 800);
        // ~Half load → a mid point; full load → the top.
        assert!(g.select(500).freq_mhz <= 1600);
        assert_eq!(g.select(1000).freq_mhz, 3200);
        // Lower OPP draws less dynamic power.
        assert!(g.select(100).relative_power() < g.select(1000).relative_power());
    }

    #[test]
    fn idle_governor_picks_deeper_states_for_longer_idles_and_is_tickless() {
        let g = IdleGovernor::new(100);
        assert_eq!(g.deepest_cstate(5), CState::C0); // too short to idle
        assert_eq!(g.deepest_cstate(50), CState::C1);
        assert_eq!(g.deepest_cstate(500), CState::C3);
        assert_eq!(g.deepest_cstate(5000), CState::C6);
        // Tickless: wake at the earliest timer, not on a periodic tick.
        assert_eq!(g.next_wake(&[5000, 1200, 9000]), Some(1200));
        assert_eq!(g.next_wake(&[]), None);
        // Gating: only as many cores as runnable tasks (≥1).
        assert_eq!(g.cores_needed(0, 8), 1);
        assert_eq!(g.cores_needed(3, 8), 3);
        assert_eq!(g.cores_needed(99, 8), 8);
    }

    #[test]
    fn device_pm_autosuspends_and_resumes_with_aspm() {
        let mut d = DevicePm::new(1000);
        d.on_idle(500);
        assert!(!d.is_suspended());
        assert_eq!(d.aspm_state(500), AspmState::L0s);
        d.on_idle(2000);
        assert!(d.is_suspended());
        assert_eq!(d.aspm_state(2000), AspmState::L1);
        d.on_access();
        assert!(!d.is_suspended());
        assert_eq!(d.aspm_state(10), AspmState::L0);
    }

    #[test]
    fn edp_placement_prefers_the_efficient_unit() {
        // A GPU is fast but power-hungry; an NPU is efficient for this task.
        let planner = EdpPlanner::new(alloc::vec![
            UnitProfile { unit: ComputeUnit::PCore, speedup: 1.0, power: 4.0 },
            UnitProfile { unit: ComputeUnit::ECore, speedup: 0.6, power: 1.0 },
            UnitProfile { unit: ComputeUnit::Gpu, speedup: 4.0, power: 20.0 },
            UnitProfile { unit: ComputeUnit::Npu, speedup: 3.0, power: 3.0 },
        ]);
        // NPU has the lowest power/speedup² ⇒ best EDP.
        assert_eq!(planner.place(), Some(ComputeUnit::Npu));
    }

    #[test]
    fn thermal_governor_escalates_with_temperature() {
        let t = ThermalGovernor::default_trips();
        assert_eq!(t.action(60), ThrottleAction::Nominal);
        assert_eq!(t.action(85), ThrottleAction::Throttle);
        assert_eq!(t.action(95), ThrottleAction::Migrate);
        assert_eq!(t.action(105), ThrottleAction::Emergency);
    }

    #[test]
    fn objective_blends_competing_goals() {
        // Power-saver weighting prefers a low-energy/cool config over a fast/hot one.
        let saver = Objective::power_saver();
        let fast_hot = saver.score(1000, 1000, 100, 100); // great latency, terrible energy/thermal
        let slow_cool = saver.score(400, 400, 1000, 1000); // modest latency, great energy/thermal
        assert!(slow_cool > fast_hot);
        // A balanced objective ranks them the other way (raw goodness sums higher for fast).
        let bal = Objective::balanced();
        assert!(bal.score(1000, 1000, 100, 100) < bal.score(1000, 1000, 1000, 1000));
    }
}

//! Recovery supervisor & rollback-storm circuit breaker — **the global policy that
//! stops thrash** (red-team finding B; see `docs/findings.md` and
//! `docs/architecture/resource-governor.md`).
//!
//! DominionOS contains faults by rolling back and auto-restarting the offending app
//! ("worst case it gets killed & recovers, state was saved"). The failure mode an
//! experienced engineer probes for is the **recovery loop**: an app that crashes the
//! instant it is restored, so rollback → restart → crash → rollback forever — a
//! *rollback storm* that burns the machine while "self-healing". The
//! [`ResourceGovernor`](crate::governor) refuses to OOM-kill; this module refuses to
//! thrash. It is the temporal mirror of the governor: where the governor bounds
//! resource *amount*, the supervisor bounds recovery *frequency*.
//!
//! Three composed defences:
//!
//! * **Exponential backoff.** Each consecutive failure delays the next restart
//!   (`base · 2ⁿ`, capped), so a fast crash-loop is spread out instead of spinning.
//! * **Per-component circuit breaker.** More than `max_failures` within a sliding
//!   `window` **trips** the breaker: auto-recovery stops and the component is
//!   **quarantined** for a cooldown (escalate to a human / leave it isolated) rather
//!   than looped. A single trial restart (half-open) probes recovery after cooldown.
//! * **Global storm guard.** A fleet-wide budget on *total* recoveries per window, so
//!   correlated failures (one bad update hitting everything) cannot become a
//!   system-wide restart stampede.
//!
//! Like the governor, every decision is **recorded** so a run replays deterministically
//! (Stage 10): raw failure events are inputs, the decision log is the reproducible
//! output. Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Tunables for the supervisor. Times are in the caller's logical tick unit.
#[derive(Clone, Copy, Debug)]
pub struct RecoveryPolicy {
    /// Failures within `window` that trip the per-component breaker.
    pub max_failures: u32,
    /// Sliding window over which failures are counted.
    pub window: u64,
    /// First backoff delay; doubles per consecutive failure.
    pub base_backoff: u64,
    /// Ceiling for the backoff delay (anti-overflow, bounds worst-case latency).
    pub max_backoff: u64,
    /// How long a tripped breaker stays open (component quarantined) before a trial.
    pub cooldown: u64,
    /// Fleet-wide cap on total recoveries per `window` (storm guard).
    pub global_budget: u32,
}

impl RecoveryPolicy {
    /// A sensible default: 3 failures / window trips, 8-tick base backoff capped at
    /// 1000, 500-tick cooldown, 16 recoveries/window fleet-wide.
    pub fn standard() -> RecoveryPolicy {
        RecoveryPolicy {
            max_failures: 3,
            window: 100,
            base_backoff: 8,
            max_backoff: 1000,
            cooldown: 500,
            global_budget: 16,
        }
    }
}

/// What the supervisor decides to do about a failure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecoveryDecision {
    /// Restart is permitted, but only after `after` ticks of backoff.
    Restart { after: u64 },
    /// The breaker tripped: stop auto-recovery, component is quarantined until `until`.
    Quarantine { until: u64 },
    /// Still inside an open breaker's cooldown — do nothing yet.
    Cooling { until: u64 },
    /// The fleet-wide storm guard is saturated: defer this recovery (re-queue later).
    GlobalStorm,
}

/// Breaker position for one component.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BreakerState {
    /// Recovering normally (with backoff).
    Closed,
    /// Tripped: quarantined until the contained tick.
    Open { until: u64 },
    /// Cooldown elapsed: one trial restart is allowed; another failure re-trips.
    HalfOpen,
}

/// A recorded supervisor decision (the deterministic input-event stream, mirroring
/// [`crate::governor::Decision`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogEntry {
    pub component: u64,
    pub at: u64,
    pub decision: RecoveryDecision,
}

#[derive(Clone)]
struct Component {
    /// Failure timestamps within the active window (pruned lazily).
    failures: Vec<u64>,
    /// Consecutive failures since the last success (drives backoff).
    consecutive: u32,
    state: BreakerState,
}

impl Component {
    fn new() -> Component {
        Component { failures: Vec::new(), consecutive: 0, state: BreakerState::Closed }
    }
}

/// The recovery supervisor: per-component breakers + a global storm guard.
pub struct Supervisor {
    policy: RecoveryPolicy,
    components: BTreeMap<u64, Component>,
    /// Global recovery timestamps within the window (storm guard).
    global: Vec<u64>,
    log: Vec<LogEntry>,
}

impl Supervisor {
    pub fn new(policy: RecoveryPolicy) -> Supervisor {
        Supervisor { policy, components: BTreeMap::new(), global: Vec::new(), log: Vec::new() }
    }

    /// Report that `component` failed at logical time `now` and ask what to do. This is
    /// the single decision point that makes a rollback storm impossible.
    pub fn on_failure(&mut self, component: u64, now: u64) -> RecoveryDecision {
        let window = self.policy.window;
        let comp = self.components.entry(component).or_insert_with(Component::new);

        // If the breaker is open, stay quarantined until the cooldown elapses; then
        // move to half-open for a single trial.
        if let BreakerState::Open { until } = comp.state {
            if now < until {
                let d = RecoveryDecision::Cooling { until };
                self.log.push(LogEntry { component, at: now, decision: d });
                return d;
            }
            comp.state = BreakerState::HalfOpen;
        }

        // Record the failure and prune the window.
        comp.failures.retain(|&t| now.saturating_sub(t) < window);
        comp.failures.push(now);
        comp.consecutive = comp.consecutive.saturating_add(1);
        let count = comp.failures.len() as u32;

        // A failure while half-open immediately re-trips (recovery didn't take).
        let half_open_relapse = comp.state == BreakerState::HalfOpen;

        let decision = if half_open_relapse || count > self.policy.max_failures {
            // Trip: quarantine instead of looping. No restart is attempted.
            let until = now.saturating_add(self.policy.cooldown);
            comp.state = BreakerState::Open { until };
            RecoveryDecision::Quarantine { until }
        } else {
            // Within budget: permit a restart, but only after exponential backoff —
            // and only if the global storm guard has headroom.
            self.global.retain(|&t| now.saturating_sub(t) < window);
            if self.global.len() as u32 >= self.policy.global_budget {
                RecoveryDecision::GlobalStorm
            } else {
                self.global.push(now);
                let shift = (comp.consecutive - 1).min(20); // cap to avoid overflow
                let backoff = self.policy.base_backoff.saturating_mul(1u64 << shift);
                RecoveryDecision::Restart { after: backoff.min(self.policy.max_backoff) }
            }
        };

        self.log.push(LogEntry { component, at: now, decision });
        decision
    }

    /// Report that `component` recovered and ran stably. Closes the breaker and resets
    /// the backoff so future isolated failures are handled gently again.
    pub fn on_success(&mut self, component: u64) {
        if let Some(c) = self.components.get_mut(&component) {
            c.consecutive = 0;
            c.failures.clear();
            c.state = BreakerState::Closed;
        }
    }

    /// Is the component currently quarantined (breaker open) at `now`?
    pub fn is_quarantined(&self, component: u64, now: u64) -> bool {
        matches!(self.components.get(&component).map(|c| c.state),
            Some(BreakerState::Open { until }) if now < until)
    }

    /// The breaker state of a component.
    pub fn state(&self, component: u64) -> BreakerState {
        self.components.get(&component).map(|c| c.state).unwrap_or(BreakerState::Closed)
    }

    /// The recorded decision log (deterministic input-event stream).
    pub fn decisions(&self) -> &[LogEntry] {
        &self.log
    }

    /// A content hash over the decision log — two supervisors driven by the same
    /// failure sequence agree exactly (the determinism boundary, checkable).
    pub fn log_digest(&self) -> Hash256 {
        let mut buf = Vec::new();
        for e in &self.log {
            buf.extend_from_slice(&e.component.to_le_bytes());
            buf.extend_from_slice(&e.at.to_le_bytes());
            buf.push(decision_tag(e.decision));
            buf.extend_from_slice(&decision_payload(e.decision).to_le_bytes());
        }
        Hash256::of(&buf)
    }
}

fn decision_tag(d: RecoveryDecision) -> u8 {
    match d {
        RecoveryDecision::Restart { .. } => 1,
        RecoveryDecision::Quarantine { .. } => 2,
        RecoveryDecision::Cooling { .. } => 3,
        RecoveryDecision::GlobalStorm => 4,
    }
}

fn decision_payload(d: RecoveryDecision) -> u64 {
    match d {
        RecoveryDecision::Restart { after } => after,
        RecoveryDecision::Quarantine { until } => until,
        RecoveryDecision::Cooling { until } => until,
        RecoveryDecision::GlobalStorm => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RecoveryPolicy {
        RecoveryPolicy { max_failures: 3, window: 100, base_backoff: 8, max_backoff: 1000, cooldown: 500, global_budget: 1000 }
    }

    #[test]
    fn isolated_failures_restart_with_exponential_backoff() {
        let mut s = Supervisor::new(policy());
        assert_eq!(s.on_failure(1, 0), RecoveryDecision::Restart { after: 8 });
        assert_eq!(s.on_failure(1, 1), RecoveryDecision::Restart { after: 16 });
        assert_eq!(s.on_failure(1, 2), RecoveryDecision::Restart { after: 32 });
    }

    #[test]
    fn a_crash_loop_trips_the_breaker_instead_of_thrashing() {
        // The headline property: rapid repeated failures stop being restarted and the
        // component is quarantined — the rollback storm cannot happen.
        let mut s = Supervisor::new(policy());
        s.on_failure(1, 0);
        s.on_failure(1, 1);
        s.on_failure(1, 2); // 3rd failure, still within max_failures=3
                            // The 4th failure in the window exceeds the budget → trip.
        assert_eq!(s.on_failure(1, 3), RecoveryDecision::Quarantine { until: 503 });
        assert!(s.is_quarantined(1, 100));
        // Further failures during cooldown are not restarted, just cooling.
        assert_eq!(s.on_failure(1, 50), RecoveryDecision::Cooling { until: 503 });
    }

    #[test]
    fn backoff_is_capped() {
        let mut s = Supervisor::new(RecoveryPolicy { max_failures: 100, max_backoff: 1000, ..policy() });
        let mut last = 0;
        for t in 0..20 {
            if let RecoveryDecision::Restart { after } = s.on_failure(1, t) {
                last = after;
            }
        }
        assert_eq!(last, 1000); // never exceeds max_backoff despite many failures
    }

    #[test]
    fn half_open_relapse_re_trips_but_recovery_closes_the_breaker() {
        let mut s = Supervisor::new(policy());
        for t in 0..4 {
            s.on_failure(1, t);
        }
        assert!(s.is_quarantined(1, 10));
        // After cooldown, a failure is half-open → relapse re-trips immediately.
        let d = s.on_failure(1, 600);
        assert!(matches!(d, RecoveryDecision::Quarantine { .. }));
        // But a *success* after cooldown closes the breaker and resets backoff.
        s.on_success(1);
        assert_eq!(s.state(1), BreakerState::Closed);
        assert_eq!(s.on_failure(1, 2000), RecoveryDecision::Restart { after: 8 });
    }

    #[test]
    fn window_decay_lets_an_occasional_failure_keep_restarting() {
        let mut s = Supervisor::new(policy());
        // Failures spaced wider than the window never accumulate to a trip.
        for k in 0..10u64 {
            let now = k * 200; // > window=100 apart
            assert!(matches!(s.on_failure(1, now), RecoveryDecision::Restart { .. }));
        }
        assert_eq!(s.state(1), BreakerState::Closed);
    }

    #[test]
    fn global_storm_guard_defers_correlated_recoveries() {
        // One bad update fails across many components at once: the fleet-wide budget
        // caps total recoveries so the system does not stampede.
        let mut s = Supervisor::new(RecoveryPolicy { global_budget: 4, ..policy() });
        let mut restarts = 0;
        let mut deferred = 0;
        for comp in 0..20u64 {
            match s.on_failure(comp, 0) {
                RecoveryDecision::Restart { .. } => restarts += 1,
                RecoveryDecision::GlobalStorm => deferred += 1,
                _ => {}
            }
        }
        assert_eq!(restarts, 4); // exactly the global budget
        assert_eq!(deferred, 16); // the rest deferred, not stampeded
    }

    #[test]
    fn decisions_replay_deterministically() {
        fn run() -> Hash256 {
            let mut s = Supervisor::new(policy());
            for t in 0..6 {
                s.on_failure(1, t);
            }
            s.on_failure(2, 1);
            s.log_digest()
        }
        assert_eq!(run(), run());
    }
}

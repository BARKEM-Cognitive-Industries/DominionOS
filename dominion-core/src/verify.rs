//! Microkernel verification harness & WCET accounting (see
//! `docs/architecture/03-stage-02-cheri-microkernel.md`).
//!
//! A verified microkernel (the seL4 lineage) ships two guarantees: an **invariant
//! suite** that machine-checking proves holds across every kernel transition, and a
//! **worst-case execution-time (WCET)** bound for every kernel entry so the system
//! is real-time-safe. We cannot run Isabelle/HOL here, but we can *encode and
//! enforce the same shape of guarantee at runtime*:
//!
//! * [`Verifier`] checks a suite of **invariants** against a state and reports every
//!   violation — the executable analogue of "the invariant holds" — plus a
//!   **refinement** check that an implementation's observable behaviour matches the
//!   abstract specification.
//! * [`WcetTable`] sums per-operation cycle costs along a path and proves it stays
//!   under a deadline, flagging the worst-case path when it does not.
//!
//! Pure, safe `no_std`, host-tested. This is the floor a real proof would sit above.

use alloc::string::String;
use alloc::vec::Vec;

/// An abstract kernel state the invariants range over (a small, checkable model).
#[derive(Clone, Debug, Default)]
pub struct KernelState {
    /// Number of live capabilities (must never exceed the table capacity).
    pub live_caps: usize,
    pub cap_capacity: usize,
    /// Sum of memory granted to domains (must not exceed total memory).
    pub granted_mem: u64,
    pub total_mem: u64,
    /// Current privilege depth (0 = user; the kernel must always return to 0).
    pub priv_depth: u32,
    /// Whether every live capability has valid provenance.
    pub provenance_ok: bool,
}

/// A named invariant: a predicate that must hold of every reachable state.
pub struct Invariant {
    pub name: &'static str,
    pub holds: fn(&KernelState) -> bool,
}

/// The standard kernel invariant suite.
pub fn standard_invariants() -> Vec<Invariant> {
    alloc::vec![
        Invariant { name: "cap-table-not-overflowed", holds: |s| s.live_caps <= s.cap_capacity },
        Invariant { name: "memory-not-oversubscribed", holds: |s| s.granted_mem <= s.total_mem },
        Invariant { name: "privilege-returns-to-user", holds: |s| s.priv_depth == 0 },
        Invariant { name: "all-caps-have-provenance", holds: |s| s.provenance_ok },
    ]
}

/// Runs an invariant suite over states.
pub struct Verifier {
    invariants: Vec<Invariant>,
}

impl Verifier {
    pub fn new(invariants: Vec<Invariant>) -> Verifier {
        Verifier { invariants }
    }

    pub fn standard() -> Verifier {
        Verifier::new(standard_invariants())
    }

    /// Names of every invariant violated by `state` (empty ⇒ the state is safe).
    pub fn violations(&self, state: &KernelState) -> Vec<&'static str> {
        self.invariants
            .iter()
            .filter(|inv| !(inv.holds)(state))
            .map(|inv| inv.name)
            .collect()
    }

    pub fn holds(&self, state: &KernelState) -> bool {
        self.violations(state).is_empty()
    }

    /// Check that a transition preserves all invariants: legal only if both the
    /// pre- and post-state satisfy the suite (inductive invariance).
    pub fn transition_preserves(&self, pre: &KernelState, post: &KernelState) -> bool {
        self.holds(pre) && self.holds(post)
    }
}

/// Refinement: an abstract spec and a concrete implementation *refine* iff they
/// produce the same observable output for the same input. This is the executable
/// shadow of seL4's functional-correctness refinement proof.
pub fn refines<I, O: PartialEq>(spec: impl Fn(&I) -> O, impl_: impl Fn(&I) -> O, inputs: &[I]) -> bool {
    inputs.iter().all(|i| spec(i) == impl_(i))
}

// ─────────────────────────── WCET accounting ───────────────────────────

/// A worst-case execution-time table: cycle cost per named operation.
#[derive(Default)]
pub struct WcetTable {
    costs: Vec<(String, u64)>,
}

impl WcetTable {
    pub fn new() -> WcetTable {
        WcetTable { costs: Vec::new() }
    }

    pub fn set_cost(&mut self, op: &str, cycles: u64) {
        if let Some(e) = self.costs.iter_mut().find(|(n, _)| n == op) {
            e.1 = cycles;
        } else {
            self.costs.push((op.into(), cycles));
        }
    }

    fn cost(&self, op: &str) -> Option<u64> {
        self.costs.iter().find(|(n, _)| n == op).map(|(_, c)| *c)
    }

    /// Total worst-case cycles for executing `path` in order. `None` if any op is
    /// uncosted — an un-bounded operation must not be admitted to a real-time path.
    pub fn path_cost(&self, path: &[&str]) -> Option<u64> {
        let mut total = 0u64;
        for op in path {
            total = total.checked_add(self.cost(op)?)?;
        }
        Some(total)
    }

    /// Does `path` complete within `deadline` cycles?
    pub fn meets_deadline(&self, path: &[&str], deadline: u64) -> bool {
        self.path_cost(path).map(|c| c <= deadline).unwrap_or(false)
    }

    /// The single costliest operation on the path (the bottleneck to optimise).
    pub fn worst_op<'a>(&self, path: &[&'a str]) -> Option<&'a str> {
        path.iter()
            .filter_map(|&op| self.cost(op).map(|c| (op, c)))
            .max_by_key(|(_, c)| *c)
            .map(|(op, _)| op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_state() -> KernelState {
        KernelState {
            live_caps: 10,
            cap_capacity: 100,
            granted_mem: 1024,
            total_mem: 4096,
            priv_depth: 0,
            provenance_ok: true,
        }
    }

    #[test]
    fn standard_invariants_hold_for_good_state() {
        let v = Verifier::standard();
        assert!(v.holds(&good_state()));
        assert!(v.violations(&good_state()).is_empty());
    }

    #[test]
    fn each_invariant_violation_is_detected() {
        let v = Verifier::standard();
        let mut s = good_state();
        s.live_caps = 200; // overflow the cap table
        assert!(v.violations(&s).contains(&"cap-table-not-overflowed"));

        let mut s = good_state();
        s.granted_mem = 9999; // oversubscribe memory
        assert!(v.violations(&s).contains(&"memory-not-oversubscribed"));

        let mut s = good_state();
        s.priv_depth = 1; // stuck in the kernel
        assert!(v.violations(&s).contains(&"privilege-returns-to-user"));

        let mut s = good_state();
        s.provenance_ok = false;
        assert!(v.violations(&s).contains(&"all-caps-have-provenance"));
    }

    #[test]
    fn transition_must_preserve_invariants() {
        let v = Verifier::standard();
        let pre = good_state();
        let mut post = good_state();
        post.live_caps = 11; // still safe
        assert!(v.transition_preserves(&pre, &post));
        post.live_caps = 1000; // breaks an invariant
        assert!(!v.transition_preserves(&pre, &post));
    }

    #[test]
    fn refinement_holds_when_impl_matches_spec() {
        // Spec: saturating add capped at 100. Impl: same behaviour.
        let spec = |x: &u32| (*x).min(100);
        let impl_ok = |x: &u32| if *x > 100 { 100 } else { *x };
        let impl_bad = |x: &u32| *x; // diverges above 100
        let inputs: Vec<u32> = alloc::vec![0, 50, 100, 150, 9999];
        assert!(refines(spec, impl_ok, &inputs));
        assert!(!refines(spec, impl_bad, &inputs));
    }

    #[test]
    fn wcet_bounds_a_path() {
        let mut t = WcetTable::new();
        t.set_cost("ipc_send", 120);
        t.set_cost("cap_lookup", 30);
        t.set_cost("context_switch", 200);
        let path = ["cap_lookup", "ipc_send", "context_switch"];
        assert_eq!(t.path_cost(&path), Some(350));
        assert!(t.meets_deadline(&path, 500));
        assert!(!t.meets_deadline(&path, 300));
        assert_eq!(t.worst_op(&path), Some("context_switch"));
    }

    #[test]
    fn uncosted_operation_is_not_admitted() {
        let mut t = WcetTable::new();
        t.set_cost("known", 10);
        // An un-bounded op makes the whole path un-boundable → fails closed.
        assert_eq!(t.path_cost(&["known", "unbounded_loop"]), None);
        assert!(!t.meets_deadline(&["known", "unbounded_loop"], u64::MAX));
    }
}

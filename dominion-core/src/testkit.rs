//! Test & verification infrastructure — **a coverage matrix, a soak harness, and a
//! differential-testing helper** (`docs/implementation/testing-and-verification-strategy.md`).
//!
//! The DST / fuzz / property / chaos harnesses already exist ([`crate::dst`],
//! [`crate::fuzz`], [`crate::props`]); this module adds the *meta* tooling the strategy
//! calls for:
//!
//! * [`CoverageMatrix`] — a per-subsystem × per-test-kind matrix with an enforced gate, so
//!   "is every subsystem covered by unit **and** property **and** metal tests?" is a
//!   checkable release condition rather than a vibe.
//! * [`soak`] — a deterministic long-running loop that asserts an invariant holds every
//!   step **and** a monotonic resource gauge stays bounded (no leak/drift). It is CI-sized
//!   here but scales to a multi-day run by raising the iteration count.
//! * [`differential`] — run the same inputs through two implementations and assert they
//!   agree (a reference vs an optimized path), the core of differential testing.
//!
//! Pure, safe `no_std`; deterministic. Host-tested.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// ───────────────────────── coverage matrix ─────────────────────────

/// A kind of test a subsystem can be covered by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestKind {
    Unit,
    Property,
    Fuzz,
    Metal,
}

impl TestKind {
    /// All test kinds.
    pub fn all() -> [TestKind; 4] {
        [TestKind::Unit, TestKind::Property, TestKind::Fuzz, TestKind::Metal]
    }
}

/// A per-subsystem coverage record over the test kinds.
#[derive(Default)]
pub struct CoverageMatrix {
    rows: BTreeMap<&'static str, Vec<TestKind>>,
}

impl CoverageMatrix {
    pub fn new() -> CoverageMatrix {
        CoverageMatrix { rows: BTreeMap::new() }
    }

    /// Record that `subsystem` is covered by `kind`.
    pub fn cover(&mut self, subsystem: &'static str, kind: TestKind) {
        let row = self.rows.entry(subsystem).or_default();
        if !row.contains(&kind) {
            row.push(kind);
        }
    }

    /// The kinds covering a subsystem.
    pub fn kinds(&self, subsystem: &str) -> Vec<TestKind> {
        self.rows.get(subsystem).cloned().unwrap_or_default()
    }

    /// Subsystems that don't yet meet `required` (the matrix gaps).
    pub fn gaps(&self, required: &[TestKind]) -> Vec<&'static str> {
        self.rows
            .iter()
            .filter(|(_, kinds)| !required.iter().all(|r| kinds.contains(r)))
            .map(|(name, _)| *name)
            .collect()
    }

    /// True iff **every** recorded subsystem meets all `required` kinds.
    pub fn meets_gate(&self, required: &[TestKind]) -> bool {
        !self.rows.is_empty() && self.gaps(required).is_empty()
    }

    /// Number of subsystems tracked.
    pub fn subsystems(&self) -> usize {
        self.rows.len()
    }
}

// ───────────────────────── soak harness ─────────────────────────

/// The outcome of a soak run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoakResult {
    /// Iterations completed.
    pub iterations: u64,
    /// True iff the invariant held every step and the resource gauge stayed bounded.
    pub passed: bool,
    /// The peak resource-gauge value observed.
    pub peak_gauge: u64,
}

/// Run a deterministic soak: each step mutates `state` via `step`, then `invariant` must
/// hold and `gauge` (e.g. live-allocation count) must stay ≤ `gauge_limit`. Detects
/// drift/leaks that only appear over a long run. Stops early on the first failure.
pub fn soak<S, Step, Inv, Gauge>(
    mut state: S,
    iterations: u64,
    gauge_limit: u64,
    mut step: Step,
    invariant: Inv,
    gauge: Gauge,
) -> SoakResult
where
    Step: FnMut(&mut S, u64),
    Inv: Fn(&S) -> bool,
    Gauge: Fn(&S) -> u64,
{
    let mut peak = 0u64;
    for i in 0..iterations {
        step(&mut state, i);
        let g = gauge(&state);
        peak = peak.max(g);
        if !invariant(&state) || g > gauge_limit {
            return SoakResult { iterations: i + 1, passed: false, peak_gauge: peak };
        }
    }
    SoakResult { iterations, passed: true, peak_gauge: peak }
}

// ───────────────────────── differential testing ─────────────────────────

/// Run every input through two implementations and return the inputs (by index) where
/// they **disagree** — empty ⇒ the implementations are equivalent over these inputs.
pub fn differential<I, O, A, B>(inputs: &[I], reference: A, candidate: B) -> Vec<usize>
where
    O: PartialEq,
    A: Fn(&I) -> O,
    B: Fn(&I) -> O,
{
    inputs
        .iter()
        .enumerate()
        .filter(|(_, x)| reference(x) != candidate(x))
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_matrix_enforces_a_gate() {
        let mut m = CoverageMatrix::new();
        // A fully-covered subsystem and an under-covered one.
        for k in TestKind::all() {
            m.cover("capability", k);
        }
        m.cover("driver", TestKind::Unit);
        m.cover("driver", TestKind::Property);
        let required = [TestKind::Unit, TestKind::Property];
        // Both meet the (unit+property) gate.
        assert!(m.meets_gate(&required));
        // But not the full matrix (driver lacks fuzz+metal).
        let gaps = m.gaps(&TestKind::all());
        assert!(gaps.contains(&"driver"));
        assert!(!gaps.contains(&"capability"));
    }

    #[test]
    fn soak_detects_a_resource_leak() {
        // A healthy system: gauge oscillates but stays bounded.
        let healthy = soak(
            0i64,
            1000,
            10,
            |s, _| *s = (*s + 1) % 8,
            |_| true,
            |s| *s as u64,
        );
        assert!(healthy.passed);
        assert!(healthy.peak_gauge <= 10);
        // A leaky system: the gauge grows without bound → caught.
        let leaky = soak(
            0u64,
            1000,
            100,
            |s, _| *s += 1, // monotonically growing "live allocations"
            |_| true,
            |s| *s,
        );
        assert!(!leaky.passed);
        assert!(leaky.iterations < 1000); // stopped early when it crossed the limit
    }

    #[test]
    fn differential_finds_a_divergence() {
        let inputs: Vec<i64> = (-5..=5).collect();
        // Two equivalent implementations of abs agree everywhere.
        let no_diff = differential(&inputs, |x: &i64| x.abs(), |x: &i64| if *x < 0 { -x } else { *x });
        assert!(no_diff.is_empty());
        // A buggy candidate (identity instead of abs) diverges on the negatives.
        let diffs = differential(&inputs, |x: &i64| x.abs(), |x: &i64| *x);
        assert_eq!(diffs.len(), 5); // -5..-1
    }
}

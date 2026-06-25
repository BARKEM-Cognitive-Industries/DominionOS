//! Compatibility & conformance harness — **AM** (see
//! `docs/implementation/hardware-software-compatibility.md`).
//!
//! "Compatible" is not a boolean — it is a **measured pass-rate over a corpus**. This
//! harness models the conformance program: each ecosystem (Linux ABI, POSIX, web
//! platform, file formats, the native Dominion capability contract) is a [`Suite`] of
//! cases; running it yields a pass-rate; and a release is **gated** on every category
//! clearing a threshold (e.g. **90%**). The built-in suites here actually exercise
//! real subsystems ([`crate::compat`] binary/ABI detection, [`crate::codec`] format
//! round-trips, [`crate::wasm`] sandbox containment), so the numbers are real, not
//! asserted. Pure, safe, host-tested.

use alloc::string::String;
use alloc::vec::Vec;

/// The outcome of one conformance case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseResult {
    pub name: String,
    pub passed: bool,
}

/// A conformance suite for one ecosystem/category.
pub struct Suite {
    pub category: String,
    results: Vec<CaseResult>,
}

impl Suite {
    pub fn new(category: &str) -> Suite {
        Suite { category: String::from(category), results: Vec::new() }
    }

    /// Record a case outcome.
    pub fn record(&mut self, name: &str, passed: bool) {
        self.results.push(CaseResult { name: String::from(name), passed });
    }

    pub fn total(&self) -> usize {
        self.results.len()
    }

    pub fn passed(&self) -> usize {
        self.results.iter().filter(|c| c.passed).count()
    }

    /// Pass-rate in per-mille (×1000). An empty suite reports 0.
    pub fn pass_rate_milli(&self) -> u32 {
        if self.results.is_empty() {
            return 0;
        }
        (self.passed() as u32 * 1000) / self.total() as u32
    }

    /// The names of the cases that failed (the work-list).
    pub fn failures(&self) -> Vec<&str> {
        self.results.iter().filter(|c| !c.passed).map(|c| c.name.as_str()).collect()
    }
}

/// An aggregated conformance report across categories, with a release gate.
#[derive(Default)]
pub struct ConformanceReport {
    suites: Vec<Suite>,
}

impl ConformanceReport {
    pub fn new() -> ConformanceReport {
        ConformanceReport { suites: Vec::new() }
    }

    pub fn add(&mut self, suite: Suite) {
        self.suites.push(suite);
    }

    /// Categories whose pass-rate is below `threshold_milli` — these block release.
    pub fn failing(&self, threshold_milli: u32) -> Vec<&str> {
        self.suites
            .iter()
            .filter(|s| s.pass_rate_milli() < threshold_milli)
            .map(|s| s.category.as_str())
            .collect()
    }

    /// The release gate: **every** category must clear the threshold.
    pub fn meets_gate(&self, threshold_milli: u32) -> bool {
        self.failing(threshold_milli).is_empty()
    }

    /// Overall pass-rate across all cases (×1000).
    pub fn overall_milli(&self) -> u32 {
        let total: usize = self.suites.iter().map(|s| s.total()).sum();
        if total == 0 {
            return 0;
        }
        let passed: usize = self.suites.iter().map(|s| s.passed()).sum();
        (passed as u32 * 1000) / total as u32
    }

    pub fn categories(&self) -> usize {
        self.suites.len()
    }
}

/// Run the built-in conformance suites against the real subsystems. The numbers are
/// produced by actually exercising the code, not declared.
pub fn run_builtin_suites() -> ConformanceReport {
    use crate::capability::{Capability, Rights};
    use crate::codec::CodecRegistry;
    use crate::compat::{detect_format, translate_syscall, Abi, BinaryFormat, HostOp};
    use crate::wasm::{Op, Sandbox, Trap};

    let mut report = ConformanceReport::new();

    // --- Binary-format detection (Linux/Windows/macOS containers) ---
    let mut fmt = Suite::new("binary-formats");
    fmt.record("elf", detect_format(&[0x7F, b'E', b'L', b'F', 0, 0]) == BinaryFormat::Elf);
    fmt.record("pe", detect_format(&[0x4D, 0x5A, 0, 0]) == BinaryFormat::Pe);
    fmt.record("macho", detect_format(&[0xCF, 0xFA, 0xED, 0xFE]) == BinaryFormat::MachO);
    fmt.record("unknown-is-rejected", detect_format(&[0, 1, 2, 3]) == BinaryFormat::Unknown);
    report.add(fmt);

    // --- Foreign-ABI syscall translation is default-closed (security conformance) ---
    let mut abi = Suite::new("abi-default-closed");
    for (name, a) in [("linux", Abi::Linux), ("win64", Abi::Win64), ("macos", Abi::MacOsArm64)] {
        // An out-of-table syscall must map to Denied for every ABI.
        abi.record(name, translate_syscall(a, 0xFFFF_FFFF) == HostOp::Denied);
    }
    report.add(abi);

    // --- File-format codec round-trip (formats corpus) ---
    let mut formats = Suite::new("format-codecs");
    let reg = CodecRegistry::with_defaults();
    let cap = Capability::mint(0, 0x1000, Rights::READ);
    let text = b"a legacy text file";
    let round_trips = match reg.import(Some("note.txt"), text, &cap) {
        Ok(obj) => reg.export(&obj, &cap).as_deref() == Ok(text.as_ref()),
        Err(_) => false,
    };
    formats.record("text-roundtrip", round_trips);
    report.add(formats);

    // --- Native sandbox containment (the Dominion capability contract) ---
    let mut sandbox = Suite::new("sandbox-containment");
    let mut computes = Sandbox::new(
        alloc::vec![Op::Const(3), Op::Const(4), Op::Add, Op::Return],
        0,
        0,
        100,
    );
    sandbox.record("guest-computes", computes.run() == Ok(7));
    let mut escapes =
        Sandbox::new(alloc::vec![Op::Call { id: 9, argc: 0 }, Op::Return], 0, 0, 100);
    sandbox.record("ungranted-call-traps", escapes.run() == Err(Trap::UngrantedHostCall));
    report.add(sandbox);

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_computes_pass_rate() {
        let mut s = Suite::new("demo");
        s.record("a", true);
        s.record("b", true);
        s.record("c", false);
        assert_eq!(s.total(), 3);
        assert_eq!(s.passed(), 2);
        assert_eq!(s.pass_rate_milli(), 666); // 2/3
        assert_eq!(s.failures(), alloc::vec!["c"]);
    }

    #[test]
    fn release_gate_blocks_on_a_failing_category() {
        let mut report = ConformanceReport::new();
        let mut good = Suite::new("good");
        good.record("x", true);
        good.record("y", true);
        let mut bad = Suite::new("bad");
        bad.record("p", true);
        bad.record("q", false); // 50%
        report.add(good);
        report.add(bad);
        // A 90% gate fails and names the offending category.
        assert!(!report.meets_gate(900));
        assert_eq!(report.failing(900), alloc::vec!["bad"]);
        // A lenient 40% gate passes.
        assert!(report.meets_gate(400));
    }

    #[test]
    fn builtin_suites_pass_the_ninety_percent_gate() {
        let report = run_builtin_suites();
        // The built-in conformance corpus (real subsystem behaviour) clears 90%.
        assert!(report.categories() >= 4);
        assert!(report.meets_gate(900), "failing: {:?}", report.failing(900));
        assert_eq!(report.overall_milli(), 1000); // everything passes today
    }
}

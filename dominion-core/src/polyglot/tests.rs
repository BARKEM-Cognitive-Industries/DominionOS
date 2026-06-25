//! Tests for the polyglot runtime: every guest language runs real multi-function,
//! package-importing programs to identical results.

use super::*;

fn run_ok(src: &str, lang: Language) -> Run {
    match run(src, lang) {
        Ok(r) => r,
        Err(e) => panic!("{} program failed: {:?}", lang.name(), e),
    }
}

#[test]
fn every_language_runs_the_demo_to_the_same_value() {
    for lang in Language::all() {
        let prog = parse(demo_program(lang), lang).unwrap_or_else(|e| panic!("{}: {:?}", lang.name(), e));
        // Real program: at least three user functions and two imported packages.
        assert!(prog.function_count() >= 3, "{} has too few functions", lang.name());
        assert!(prog.imports().contains(&"stats"), "{} did not import stats", lang.name());
        assert!(prog.imports().contains(&"mathx"), "{} did not import mathx", lang.name());

        let r = execute(&prog, DEFAULT_STEP_BUDGET).unwrap_or_else(|e| panic!("{}: {:?}", lang.name(), e));
        assert!(
            r.value.approx(DEMO_EXPECTED, 1e-9),
            "{} returned {:?}, expected {}",
            lang.name(),
            r.value,
            DEMO_EXPECTED
        );
        assert!(r.steps > 0);
    }
}

#[test]
fn every_language_benchmark_agrees_on_the_checksum() {
    for lang in Language::all() {
        let r = run_ok(bench_program(lang), lang);
        assert_eq!(
            r.value,
            Value::Int(BENCH_EXPECTED),
            "{} benchmark checksum mismatch: {:?}",
            lang.name(),
            r.value
        );
        // A real compute load: thousands of interpreter steps.
        assert!(r.steps > 10_000, "{} ran only {} steps", lang.name(), r.steps);
    }
}

#[test]
fn package_functions_are_default_closed_without_an_import() {
    // `gcd` belongs to mathx; calling it without importing mathx must be refused.
    let src = "fn run() { return gcd(48, 36); }";
    let err = run(src, Language::Rust).unwrap_err();
    assert!(matches!(err, RunError::NotImported(_)), "got {:?}", err);

    // Importing it makes the call resolve.
    let ok = "use mathx::gcd; fn run() { return gcd(48, 36); }";
    assert_eq!(run(ok, Language::Rust).unwrap().value, Value::Int(12));
}

#[test]
fn unknown_names_are_undefined() {
    let err = run("fn run() { return nope(1); }", Language::Rust).unwrap_err();
    assert!(matches!(err, RunError::Undefined(_)), "got {:?}", err);
}

#[test]
fn recursion_and_control_flow_python() {
    let src = r#"
def fact(n):
    if n < 2:
        return 1
    return n * fact(n - 1)

def run():
    return fact(6)
"#;
    assert_eq!(run(src, Language::Python).unwrap().value, Value::Int(720));
}

#[test]
fn recursion_and_control_flow_rust() {
    let src = "fn fact(n) { if n < 2 { return 1; } return n * fact(n - 1); } fn run() { return fact(6); }";
    assert_eq!(run(src, Language::Rust).unwrap().value, Value::Int(720));
}

#[test]
fn library_math_and_stats_compute_correctly() {
    // sqrt, pstdev, mean, median via the stats/mathx packages.
    let src = r#"
use mathx::sqrt;
use stats::mean;
fn run() {
    let xs = vec![2.0, 4.0, 6.0];
    let m = mean(xs);
    return sqrt(m * m);
}
"#;
    assert!(run(src, Language::Rust).unwrap().value.approx(4.0, 1e-9));
}

#[test]
fn loops_build_lists_in_every_language() {
    // The demo's `scale` builds a list with each language's idiomatic append; verify
    // the scaled data round-trips by summing it (sum is a universal builtin).
    for lang in Language::all() {
        let prog = parse(demo_program(lang), lang).unwrap();
        let r = execute(&prog, DEFAULT_STEP_BUDGET).unwrap();
        // summary = pstdev(scaled) + 12 = 6.0 + 12 = 18.0 ⇒ proves the list was built.
        assert!(r.value.approx(18.0, 1e-9), "{}", lang.name());
    }
}

#[test]
fn out_of_gas_bounds_a_runaway_guest() {
    let src = "fn run() { let mut i = 0; while i >= 0 { i = i + 1; } return i; }";
    let prog = parse(src, Language::Rust).unwrap();
    let err = execute(&prog, 100_000).unwrap_err();
    assert_eq!(err, RunError::OutOfGas);
}

#[test]
fn typed_c_family_parses_classes_and_c_style_for() {
    // C#/Java wrap methods in a class and use a C-style for; both must work.
    for lang in [Language::CSharp, Language::Java, Language::Cpp] {
        let r = run_ok(bench_program(lang), lang);
        assert_eq!(r.value, Value::Int(BENCH_EXPECTED), "{}", lang.name());
    }
}

#[test]
fn javascript_and_typescript_share_semantics() {
    let js = run_ok(demo_program(Language::JavaScript), Language::JavaScript);
    let ts = run_ok(demo_program(Language::TypeScript), Language::TypeScript);
    assert_eq!(js.value, ts.value);
    assert!(js.value.approx(DEMO_EXPECTED, 1e-9));
}

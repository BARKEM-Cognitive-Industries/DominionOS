//! Integration test: run every `.aeth` file in `examples/` through the Dominion
//! interpreter and assert the result is `Ok`. Sidecars with a `.expect` extension
//! pin the expected return value. At least 25 example files must exist.
//!
//! A second sweep runs "VM-compatible" examples through the bytecode compiler +
//! VM (`eval_compiled`) to prove the compiled execution path produces the same
//! results as the interpreter.
//!
//! Run with: `cargo test --test examples`

use std::fs;
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    // The test binary is run from the workspace/package root; examples/ sits one
    // level above dominion-core (at the repo root).
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest).parent().unwrap().join("examples")
}

/// Evaluate a Dominion source file through the interpreter.
fn eval_file(path: &PathBuf) -> Result<String, String> {
    let src = fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {}", path.display(), e))?;
    dominion_core::lang::eval_source(&src)
        .map(|v| format!("{}", v))
        .map_err(|e| format!("eval error in {}: {}", path.display(), e))
}

/// Evaluate a Dominion source file through the bytecode compiler + VM.
fn compile_eval_file(path: &PathBuf) -> Result<String, String> {
    let src = fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {}", path.display(), e))?;
    dominion_core::lang::vm::eval_compiled(&src)
        .map(|v| format!("{}", v))
        .map_err(|e| format!("vm error in {}: {}", path.display(), e))
}

#[test]
fn all_examples_eval_without_error() {
    let dir = examples_dir();
    assert!(
        dir.exists(),
        "examples/ directory not found at {}",
        dir.display()
    );

    let mut aeth_files: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("could not read examples/")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "aeth").unwrap_or(false))
        .collect();
    aeth_files.sort();

    assert!(
        aeth_files.len() >= 25,
        "expected ≥25 example files, found {} in {}",
        aeth_files.len(),
        dir.display()
    );

    let mut errors: Vec<String> = Vec::new();

    for path in &aeth_files {
        let name = path.file_name().unwrap().to_string_lossy();
        match eval_file(path) {
            Ok(got) => {
                // Check for a .expect sidecar that pins the return value.
                let expect_path = path.with_extension("aeth.expect");
                if expect_path.exists() {
                    let want = fs::read_to_string(&expect_path)
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    if !want.is_empty() && got.trim() != want {
                        errors.push(format!(
                            "EXPECT MISMATCH {}: got {:?}, want {:?}",
                            name, got, want
                        ));
                    }
                }
            }
            Err(e) => {
                errors.push(format!("EVAL ERROR {}: {}", name, e));
            }
        }
    }

    if !errors.is_empty() {
        panic!(
            "{} example(s) failed:\n{}",
            errors.len(),
            errors.join("\n")
        );
    }
    println!("all {} examples passed (interpreter path)", aeth_files.len());
}

/// Run the simple, pure examples through the bytecode compiler + VM and verify
/// results match what the interpreter produces.  Only examples whose source
/// does not rely on builtins that the VM falls back to the interpreter for
/// (ML, crypto, driver simulation) are included here; those still pass via the
/// interpreter sweep above.
#[test]
fn simple_examples_pass_compiled_vm_path() {
    // Examples that use only arithmetic, vectors, strings, functions, and
    // control flow — all natively compiled by the bytecode compiler.
    let vm_candidates = &[
        "01_hello_world.aeth",
        "02_variables_and_types.aeth",
        "03_control_flow.aeth",
        "04_functions.aeth",
        "05_vectors.aeth",
        "06_strings.aeth",
        "07_loops_and_iteration.aeth",
        "20_sorting_algorithms.aeth",
    ];

    let dir = examples_dir();
    let mut errors: Vec<String> = Vec::new();
    let mut tested = 0;

    for name in vm_candidates {
        let path = dir.join(name);
        if !path.exists() {
            continue; // skip if the file doesn't exist (e.g., different numbering)
        }
        tested += 1;

        // Interpreter result (ground truth).
        let interp_result = eval_file(&path);
        // Compiled VM result.
        let vm_result = compile_eval_file(&path);

        match (&interp_result, &vm_result) {
            (Ok(interp_val), Ok(vm_val)) => {
                if interp_val != vm_val {
                    errors.push(format!(
                        "VM/INTERP MISMATCH {}: interp={:?} vm={:?}",
                        name, interp_val, vm_val
                    ));
                }
            }
            (Ok(_), Err(e)) => {
                errors.push(format!("VM ERROR (interp OK) {}: {}", name, e));
            }
            (Err(e), _) => {
                errors.push(format!("INTERP ERROR {}: {}", name, e));
            }
        }
    }

    if !errors.is_empty() {
        panic!(
            "{} compiled-VM test(s) failed:\n{}",
            errors.len(),
            errors.join("\n")
        );
    }
    println!(
        "{}/{} simple examples produce identical results via compiled VM + interpreter",
        tested,
        vm_candidates.len()
    );
}

#[test]
fn at_least_25_examples_exist() {
    let dir = examples_dir();
    let count = fs::read_dir(&dir)
        .expect("could not read examples/")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "aeth")
                .unwrap_or(false)
        })
        .count();
    assert!(count >= 25, "need ≥25 .aeth examples, have {}", count);
}

//! Polyglot **developer surface** — the unified "work with any language" layer on top
//! of the [`super`] interpreter. Where the parent module supplies the lexer/parser/
//! interpreter for seven guest languages, this submodule supplies what an IDE,
//! terminal, or Dominion script needs to *use* them as a development environment:
//!
//! * resolve a language by **name** or **file extension** (so `run("py", src)` or a
//!   `foo.rs` buffer Just Works from the GUI/terminal/code);
//! * **compile-check** a buffer (parse without running) for editor diagnostics;
//! * **run** with output capture and a **metered** variant (step count = a basic
//!   profiler/debug signal);
//! * enumerate the **language catalog** (display name, extensions, a runnable sample)
//!   and the **package catalog** (the importable libraries + their functions), so a
//!   GUI language picker / package browser is data-driven, not hardcoded per surface.
//!
//! Every language runs through the *same* capability-bounded, step-metered interpreter
//! as before — this layer adds no new execution authority, only ergonomics. Pure, safe
//! `no_std + alloc`, host-tested.

use super::{run, Language, Run, RunError};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Resolve a language from a free-form name or file extension (case-insensitive).
/// Accepts the common spellings a developer or a filename would use.
pub fn from_name(s: &str) -> Option<Language> {
    let k = s.trim().trim_start_matches('.').to_ascii_lowercase();
    let lang = match k.as_str() {
        "python" | "py" | "py3" | "python3" => Language::Python,
        "rust" | "rs" => Language::Rust,
        "c++" | "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "cplusplus" => Language::Cpp,
        "c#" | "csharp" | "cs" | "cwhash" => Language::CSharp,
        "javascript" | "js" | "node" | "mjs" | "cjs" => Language::JavaScript,
        "typescript" | "ts" | "mts" | "cts" => Language::TypeScript,
        "java" => Language::Java,
        _ => return None,
    };
    Some(lang)
}

/// Resolve a language from a filename by its extension (`main.py` → Python).
pub fn from_filename(name: &str) -> Option<Language> {
    let ext = name.rsplit('.').next()?;
    if ext == name {
        return None; // no extension present
    }
    from_name(ext)
}

/// The file extensions canonically associated with a language (first is preferred).
pub fn extensions(lang: Language) -> &'static [&'static str] {
    match lang {
        Language::Python => &["py"],
        Language::Rust => &["rs"],
        Language::Cpp => &["cpp", "cc", "cxx", "hpp"],
        Language::CSharp => &["cs"],
        Language::JavaScript => &["js", "mjs", "cjs"],
        Language::TypeScript => &["ts", "mts"],
        Language::Java => &["java"],
    }
}

/// A description of a hostable language, for a data-driven GUI/terminal picker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LangInfo {
    pub display: &'static str,
    /// The canonical short id (`extensions()[0]`), e.g. "py", "rs".
    pub id: &'static str,
    pub extensions: &'static [&'static str],
}

/// The full language catalog, in the runtime's stable order.
pub fn catalog() -> Vec<LangInfo> {
    Language::all()
        .iter()
        .map(|&lang| LangInfo {
            display: lang.name(),
            id: extensions(lang)[0],
            extensions: extensions(lang),
        })
        .collect()
}

/// An importable library package and the functions it provides — the data behind a
/// "package browser" and the `import`/`use`/`require` resolution. Mirrors the parent
/// module's default-closed package registry (a call to one of these requires importing
/// the package, exactly like installing+importing a real dependency).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageInfo {
    pub name: &'static str,
    pub functions: &'static [&'static str],
}

/// Every library package available to import, with its functions.
pub fn packages() -> Vec<PackageInfo> {
    vec![
        PackageInfo {
            name: "mathx",
            functions: &["sqrt", "pow", "powi", "gcd", "factorial", "isqrt", "floor", "ceil", "fib"],
        },
        PackageInfo {
            name: "stats",
            functions: &["mean", "variance", "stdev", "pstdev", "median", "pvariance"],
        },
        PackageInfo {
            name: "strx",
            functions: &["upper", "lower", "repeat", "reverse_str", "concat"],
        },
    ]
}

/// Look up which package provides a function (for "where does `gcd` come from?").
pub fn package_providing(function: &str) -> Option<&'static str> {
    let f = function.to_ascii_lowercase();
    packages()
        .into_iter()
        .find(|p| p.functions.iter().any(|fun| *fun == f))
        .map(|p| p.name)
}

/// The builtins available to every guest with no import.
pub fn builtins() -> &'static [&'static str] {
    super::BUILTINS
}

/// **Compile-check** a buffer: parse it in `lang` without running, returning the number
/// of functions it defines on success or the parse error for editor diagnostics.
pub fn check(src: &str, lang: Language) -> Result<usize, RunError> {
    let prog = super::parse(src, lang)?;
    Ok(prog.funcs.len())
}

/// Resolve a language by name and **run** a source buffer, capturing its output.
pub fn run_named(lang_name: &str, src: &str) -> Result<Run, DevError> {
    let lang = from_name(lang_name).ok_or_else(|| DevError::UnknownLanguage(lang_name.to_string()))?;
    run(src, lang).map_err(DevError::Run)
}

/// Resolve a language by name, parse `src`, and call the named function with
/// the provided [`super::Value`] arguments.  Use this for type-safe polyglot
/// function calls from Dominion code.
pub fn call_named(
    lang_name: &str,
    src: &str,
    fn_name: &str,
    args: Vec<super::Value>,
) -> Result<Run, DevError> {
    let lang = from_name(lang_name).ok_or_else(|| DevError::UnknownLanguage(lang_name.to_string()))?;
    super::call_func(src, lang, fn_name, args).map_err(DevError::Run)
}

/// Run a source buffer in a known language, capturing output and step count.
pub fn run_lang(src: &str, lang: Language) -> Result<Run, RunError> {
    run(src, lang)
}

/// Why a developer-surface action failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DevError {
    /// The language name/extension was not recognised.
    UnknownLanguage(String),
    /// The program failed to parse or run.
    Run(RunError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_languages_by_name_and_extension() {
        assert_eq!(from_name("python"), Some(Language::Python));
        assert_eq!(from_name("PY"), Some(Language::Python));
        assert_eq!(from_name(".rs"), Some(Language::Rust));
        assert_eq!(from_name("c++"), Some(Language::Cpp));
        assert_eq!(from_name("cs"), Some(Language::CSharp));
        assert_eq!(from_name("ts"), Some(Language::TypeScript));
        assert_eq!(from_name("node"), Some(Language::JavaScript));
        assert_eq!(from_name("cobol"), None);
        assert_eq!(from_filename("main.py"), Some(Language::Python));
        assert_eq!(from_filename("lib.rs"), Some(Language::Rust));
        assert_eq!(from_filename("noext"), None);
    }

    #[test]
    fn catalog_covers_every_language() {
        let cat = catalog();
        assert_eq!(cat.len(), 7);
        assert!(cat.iter().any(|l| l.display == "Python" && l.id == "py"));
        assert!(cat.iter().any(|l| l.display == "Rust" && l.id == "rs"));
        // Every catalog id resolves back to a language.
        for info in &cat {
            assert!(from_name(info.id).is_some(), "id {} should resolve", info.id);
        }
    }

    #[test]
    fn package_catalog_resolves_functions() {
        assert_eq!(package_providing("gcd"), Some("mathx"));
        assert_eq!(package_providing("MEAN"), Some("stats"));
        assert_eq!(package_providing("upper"), Some("strx"));
        assert_eq!(package_providing("not_a_function"), None);
        assert!(builtins().contains(&"print"));
    }

    #[test]
    fn compile_check_accepts_valid_and_rejects_invalid() {
        // A valid Python program with one function.
        let ok = "def add(a, b):\n    return a + b\n";
        assert_eq!(check(ok, Language::Python).unwrap(), 1);
        // Garbage fails the parse check with a diagnostic.
        let bad = "def (((";
        assert!(check(bad, Language::Python).is_err());
    }

    #[test]
    fn run_named_runs_a_python_program() {
        let src = "def main():\n    return 6 * 7\n";
        let run = run_named("python", src).unwrap();
        assert_eq!(run.value, super::super::Value::Int(42));
        assert!(run.steps > 0);
    }

    #[test]
    fn run_named_rejects_unknown_language() {
        let err = run_named("brainfuck", "x").unwrap_err();
        assert_eq!(err, DevError::UnknownLanguage("brainfuck".into()));
    }

    #[test]
    fn the_same_program_runs_across_languages() {
        // Each language computes 6*7 via its own surface grammar → same value.
        let cases = [
            (Language::Python, "def main():\n    return 6 * 7\n"),
            (Language::Rust, "fn main() {\n    return 6 * 7;\n}\n"),
            (Language::JavaScript, "function main() {\n    return 6 * 7;\n}\n"),
        ];
        for (lang, src) in cases {
            let run = run_lang(src, lang).unwrap();
            assert_eq!(run.value, super::super::Value::Int(42), "lang {:?}", lang);
        }
    }
}

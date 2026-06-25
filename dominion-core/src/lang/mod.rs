//! The **Aether** language — DominionOS's sole execution-modelling language.
//!
//! Aether is intralingual (§5): the same syntax expresses a microkernel cell and
//! a user-facing pipeline; the only difference is the capability token the
//! runtime injects. This module assembles the full pipeline:
//!
//! ```text
//!   source ──lex──▶ tokens ──parse──▶ AST ──interpret──▶ Value
//! ```

pub mod ast;
pub mod emit;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod value;
// Compilation pipeline (workstream C): bytecode + compiler + VM + JIT tier.
pub mod bytecode;
pub mod compile;
pub mod vm;
pub mod jit;

pub use ast::{Item, Program};
pub use emit::to_source;
pub use interp::{Interpreter, RuntimeError};
pub use parser::{parse_source, ParseError};
pub use value::Value;

use alloc::string::String;
use alloc::format;

/// Convenience front door: evaluate Aether source in a fresh, fully-privileged
/// interpreter and return the final value. Used by the terminal's `dominion`
/// command and the one-liner REPL.
pub fn eval_source(src: &str) -> Result<Value, String> {
    let mut it = Interpreter::new();
    it.eval_str(src).map_err(|e| format!("{}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_source_front_door() {
        assert_eq!(eval_source("2 + 2").unwrap(), Value::Int(4));
        assert!(eval_source("let = 5").is_err());
    }
}

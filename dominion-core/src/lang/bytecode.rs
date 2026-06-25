//! Aether bytecode — the instruction set the compiler emits and the VM executes.
//!
//! OWNED BY: Compiler/VM/JIT agent (workstream C).
//!
//! This is the real ISA for the Aether stack machine, not a thin wrapper over the
//! tree-walking interpreter. A whole program lowers to one [`CompiledProgram`]: a
//! top-level [`Chunk`] (the statements that form the REPL result) plus one [`Chunk`]
//! per user function. Each chunk carries its own constant pool and a flat `code`
//! vector of [`Op`]s. The VM ([`crate::lang::vm`]) decodes and runs chunks over a
//! value stack with a call-frame stack; the JIT tier ([`crate::lang::jit`])
//! pre-decodes hot chunks into a faster threaded form.
//!
//! Constructs the compiler does not lower natively (the bulk of builtins, `::`
//! paths, `=>`, `|>`, exotic numeric ops) are emitted as fall-back instructions
//! that re-enter the existing [`crate::lang::Interpreter`] so behaviour stays
//! bit-identical with the tree-walker.

use super::ast::{BinOp, Expr};
use alloc::string::String;
use alloc::vec::Vec;

/// A compile-time constant interned in a chunk's pool. We keep these as a tiny,
/// fully-`Clone`/`PartialEq` enum (rather than runtime `Value`) so the bytecode is
/// data-only and trivially comparable in tests.
#[derive(Clone, PartialEq, Debug)]
pub enum Const {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// One VM instruction. Operands are inline (indices into the constant pool, the
/// locals array, the function table, or relative jump targets).
#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    // ── stack / constants ────────────────────────────────────────────────
    /// Push constant `consts[idx]`.
    Const(u32),
    /// Push `Value::Unit`.
    Unit,
    /// Pop and discard the top of stack.
    Pop,

    // ── locals (params + `let`/`linear`/loop vars live in a flat frame array) ─
    /// Push a copy of local slot `idx`.
    LoadLocal(u32),
    /// Pop the stack into local slot `idx` (declaration or assignment — slots are
    /// pre-sized per chunk, so declare and reassign are the same store).
    StoreLocal(u32),
    /// Push a copy of global slot `idx` (top-level `let`/`linear` binding, visible
    /// to function bodies exactly as the interpreter's outer scope is).
    LoadGlobal(u32),
    /// Pop the stack into global slot `idx`.
    StoreGlobal(u32),

    // ── operators ────────────────────────────────────────────────────────
    /// Arithmetic / comparison binary op on the two top stack values.
    Binary(BinOp),
    /// Unary integer/float negation.
    Neg,
    /// Logical not (by truthiness).
    Not,
    /// Index: `stack[-2][stack[-1]]`.
    Index,
    /// Field access `obj.field` (field name `strings[idx]`).
    Field(u32),

    // ── aggregates ───────────────────────────────────────────────────────
    /// Build a `Value::Vector` from the top `n` stack values (in order).
    MakeVec(u32),
    /// Build a `Value::Object` of kind `strings[kind]` from the top `fields.len()`
    /// stack values, paired with the field names in `fields`.
    MakeObject { kind: u32, fields: Vec<u32> },

    // ── control flow (relative jumps; target = (index after this op) + offset) ─
    /// Unconditional jump by signed offset.
    Jump(i32),
    /// Pop; jump by offset if the popped value is **falsey**.
    JumpIfFalse(i32),
    /// Peek (do not pop); jump by offset if top is **falsey**. For `&&`.
    JumpIfFalsePeek(i32),
    /// Peek (do not pop); jump by offset if top is **truthy**. For `||`.
    JumpIfTruePeek(i32),

    // ── calls ────────────────────────────────────────────────────────────
    /// Call user function `funcs[idx]` with `argc` args taken from the stack.
    CallUser { func: u32, argc: u32 },
    /// Call builtin named `strings[name]` with `argc` args (interpreter fallback).
    CallBuiltin { name: u32, argc: u32 },
    /// Call a `::` path `exprs[path]` (an `Expr::Path`) with `argc` args
    /// (interpreter fallback).
    CallPath { path: u32, argc: u32 },
    /// The `=>` parallel-map (`xs => callee`) — interpreter fallback. The callee
    /// expression is carried verbatim in `exprs[idx]`; the list is on the stack.
    Map(u32),
    /// The `|>` pipe (`x |> callee`) — interpreter fallback. The callee expression
    /// is carried verbatim in `exprs[idx]`; the value is on the stack.
    Pipe(u32),

    /// Catch-all interpreter fallback: evaluate `exprs[idx]` (a whole AST
    /// expression with no compiled sub-parts) directly in the embedded
    /// interpreter and push the result. Used for any construct the compiler does
    /// not lower natively — including a future `Expr` variant a concurrent
    /// session may add — so nothing ever regresses.
    EvalExpr(u32),

    /// Return the top of stack from the current function.
    Return,
}

/// A unit of compiled code: a constant pool plus a flat instruction stream, and
/// the side tables those instructions index into.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Chunk {
    /// Numeric/string/bool constants.
    pub consts: Vec<Const>,
    /// Interned strings used as field/builtin/kind names.
    pub strings: Vec<String>,
    /// Whole AST expressions carried for interpreter-fallback ops (paths/map/pipe).
    pub exprs: Vec<Expr>,
    /// The instruction stream.
    pub code: Vec<Op>,
    /// Number of local slots this chunk needs (params + bindings).
    pub n_locals: u32,
}

impl Chunk {
    pub fn new() -> Chunk {
        Chunk::default()
    }

    /// Intern a constant, returning its pool index (deduplicated).
    pub fn add_const(&mut self, c: Const) -> u32 {
        if let Some(i) = self.consts.iter().position(|x| *x == c) {
            return i as u32;
        }
        self.consts.push(c);
        (self.consts.len() - 1) as u32
    }

    /// Intern a string, returning its index (deduplicated).
    pub fn add_string(&mut self, s: String) -> u32 {
        if let Some(i) = self.strings.iter().position(|x| *x == s) {
            return i as u32;
        }
        self.strings.push(s);
        (self.strings.len() - 1) as u32
    }

    /// Stash an AST expression for a fall-back op, returning its index.
    pub fn add_expr(&mut self, e: Expr) -> u32 {
        self.exprs.push(e);
        (self.exprs.len() - 1) as u32
    }

    /// Emit an instruction, returning its index (so jumps can be patched).
    pub fn emit(&mut self, op: Op) -> usize {
        self.code.push(op);
        self.code.len() - 1
    }
}

/// A compiled user function: a name, its parameter count, and its body chunk.
#[derive(Clone, PartialEq, Debug)]
pub struct CompiledFn {
    pub name: String,
    pub arity: u32,
    pub chunk: Chunk,
}

/// The whole program lowered to bytecode: the top-level chunk plus every user
/// function. Function calls index into `funcs` by position; the top-level chunk
/// and each function body both resolve user calls through that shared table.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct CompiledProgram {
    /// The top-level statements (their last value is the REPL result).
    pub top: Chunk,
    /// One compiled chunk per user function, in definition order.
    pub funcs: Vec<CompiledFn>,
    /// The full source [`crate::lang::Program`] retained so fall-back ops can
    /// re-enter the interpreter with the original item/function context.
    pub source: super::ast::Program,
    /// Global slot index → binding name. Allows the VM to mirror every
    /// `StoreGlobal` into the embedded interpreter's scope so fallback ops
    /// (EvalExpr, Map, Pipe, CallBuiltin, CallPath) see the same bindings.
    pub global_names: alloc::vec::Vec<alloc::string::String>,
}

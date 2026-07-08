//! The Dominion abstract syntax tree.
//!
//! A program is a sequence of items: object/cell/function definitions and
//! top-level statements (the latter make the language usable as a REPL inside
//! the safe-mode terminal).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, PartialEq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Ident(String),
    /// A `::`-separated path, e.g. `StorageManager::compress_to_latent`.
    Path(Vec<String>),
    /// Unary negation.
    Neg(Box<Expr>),
    /// Logical NOT (`!x`).
    Not(Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Indexing `xs[i]` — vector element or string character.
    Index(Box<Expr>, Box<Expr>),
    /// Function or path call: `f(a, b)` / `Codec::encode(x)`.
    Call(Box<Expr>, Vec<Expr>),
    /// The parallel-map operator: `xs => Path` applies the callee to every item.
    Map(Box<Expr>, Box<Expr>),
    /// The functional pipeline operator: `x |> f` feeds `x` as the first argument
    /// to `f` (and `x |> f(a)` ⇒ `f(x, a)`), enabling left-to-right data flow with
    /// automatic dependency ordering.
    Pipe(Box<Expr>, Box<Expr>),
    /// Vector literal `[a, b, c]`.
    Vector(Vec<Expr>),
    /// Object construction `Kind { field: expr, ... }`.
    ObjectLit(String, Vec<(String, Expr)>),
    /// Field access `obj.field`.
    Field(Box<Expr>, String),
}

#[derive(Clone, PartialEq, Debug)]
pub enum Stmt {
    Let(String, Expr),
    /// An **affine** (use-once) binding: `linear x = expr`. The value may be read
    /// exactly once (reading *moves* it); any unconsumed affine value is
    /// cryptographically invalidated at scope end — no garbage collector.
    Linear(String, Expr),
    /// Reassign an existing (non-affine) binding: `x = expr`. Unlike `let`, this
    /// updates the variable in the scope where it already lives — the mechanism
    /// loop bodies use to carry an accumulator across iterations.
    Assign(String, Expr),
    Return(Expr),
    Expr(Expr),
    If {
        cond: Expr,
        then_block: Vec<Stmt>,
        else_block: Vec<Stmt>,
    },
    /// `while cond { ... }` — repeat the body while the condition is truthy.
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    /// `for x in iterable { ... }` — bind each element of a Vector (or `range`).
    For {
        var: String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    /// `break` out of the innermost loop.
    Break,
    /// `continue` to the next iteration of the innermost loop.
    Continue,
}

/// Hardware-placement hint from a decorator (`@NPU`, `@GPU`, `@CPU`). The
/// scheduler is free to honour or override it (SRS §5.4 heterogeneous tagging).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Placement {
    Cpu,
    Gpu,
    Npu,
    Any,
}

#[derive(Clone, PartialEq, Debug)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
    pub placement: Placement,
}

#[derive(Clone, PartialEq, Debug)]
pub struct ObjectDef {
    pub name: String,
    pub fields: Vec<(String, String)>, // (field, type-name)
}

#[derive(Clone, PartialEq, Debug)]
pub struct CellDef {
    pub name: String,
    /// The capability right this cell requires to run, e.g. `StorageWrite`.
    pub required_cap: Option<String>,
    pub methods: Vec<FnDef>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Item {
    Object(ObjectDef),
    Cell(CellDef),
    Fn(FnDef),
    Stmt(Stmt),
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Program {
    pub items: Vec<Item>,
}

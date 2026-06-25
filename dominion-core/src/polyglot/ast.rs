//! Polyglot AST: the shared `Program` (functions + top-level statements) that every
//! guest language lowers to, plus its `Expr`/`Stmt`/`BinOp` building blocks. One AST
//! serves all surface languages, so semantics can never drift between them. Split out
//! of the former monolithic `polyglot.rs`; the parser builds these, the interpreter
//! consumes them. Cross-module fields are `pub(crate)` so the parser and interpreter
//! (siblings/parent of this module) can construct and read them.

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<Expr>),
    Var(String),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    /// A call by dotted/`::`-qualified path (e.g. `math.sqrt`, `stats::mean`).
    Call(Vec<String>, Vec<Expr>),
    /// A `range(a, b)` / `a..b` half-open integer range (only meaningful in `for`).
    Range(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stmt {
    Let(String, Expr),
    Assign(String, Expr),
    IndexAssign(Expr, Expr, Expr),
    Return(Option<Expr>),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    /// `for VAR in RANGE { body }`.
    ForRange(String, Expr, Vec<Stmt>),
    /// `for VAR in LIST { body }`.
    ForEach(String, Expr, Vec<Stmt>),
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Func {
    pub(crate) name: String,
    pub(crate) params: Vec<String>,
    pub(crate) body: Vec<Stmt>,
}

/// A parsed guest program: imported packages, user functions, and top-level code.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub(crate) imports: Vec<&'static str>,
    pub(crate) funcs: Vec<Func>,
    pub(crate) main: Vec<Stmt>,
    pub(crate) language: Language,
}

impl Program {
    pub fn language(&self) -> Language {
        self.language
    }
    pub fn function_count(&self) -> usize {
        self.funcs.len()
    }
    /// The canonical names of the library packages this program imported.
    pub fn imports(&self) -> &[&'static str] {
        &self.imports
    }
}

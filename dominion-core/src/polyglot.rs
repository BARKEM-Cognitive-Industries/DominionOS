//! Polyglot language runtime — running the world's languages, contained.
//!
//! DominionOS grants *native* authority to exactly one language (Dominion, see
//! [`crate::lang`]). Every other language is **untrusted** and runs through this
//! runtime: real source text in that language's own grammar is lexed, parsed into a
//! shared AST, and executed by one capability-bounded interpreter over one
//! standard-library/package registry (see `docs/language/multi-language-and-runtimes.md`).
//!
//! Seven ecosystems are supported as first-class guests — **Python, Rust, C++, C#,
//! JavaScript, TypeScript, Java** — and each runs *real programs*, not toy
//! one-liners:
//!
//! * **multi-function programs** — user functions calling each other, recursion,
//!   loops, lists, conditionals;
//! * **packages / libraries** — programs `import`/`use`/`#include`/`using`/`require`
//!   a library package in that language's idiomatic syntax and call into it; a call
//!   to a package function the program did not import is refused (**default-closed**,
//!   the same discipline as the capability sandbox), so the dependency is real, not
//!   decorative.
//!
//! The two surface grammars (brace-delimited for the C-family + Rust, and
//! indentation-delimited for Python) lower to the **same** [`Program`] AST and run
//! on the **same** interpreter, so adding a language is a front-end concern and the
//! semantics can never drift between them. Execution is metered (a step budget bounds
//! every guest), which also gives the benchmark battery an honest unit of work.
//!
//! Pure, safe `no_std + alloc`, host-tested.

use crate::datatypes::{ceil, floor, sqrt};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Default per-program step budget — bounds every guest the way the sandbox VM's
/// gas meter does, and is the unit the benchmark battery counts.
pub const DEFAULT_STEP_BUDGET: u64 = 50_000_000;

// The lexer, parser, and AST were split out of this (formerly 2,600-line) file into
// their own modules for readability; the value/error types, the interpreter, the
// standard library, and the demo programs remain here. `parse` (parser) and the
// `Program` AST are re-exported so `crate::polyglot::{parse, Program}` stay stable;
// the interpreter below uses the AST's `pub(crate)` building blocks directly.
mod ast;
mod lexer;
mod parser;
/// The unified developer surface (run/compile-check/catalog) over every guest language.
pub mod runtime;
pub use ast::Program;
pub use parser::parse;
use ast::{BinOp, Expr, Func, Stmt};

// ───────────────────────────── values ─────────────────────────────

/// A runtime value. Deliberately small and language-neutral: the same value model
/// serves every guest, so a `List` produced by Python code is identical to one
/// produced by Rust code.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<Value>),
    Unit,
}

impl Value {
    fn truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Unit => false,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }
    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    /// A human display, used for `print`-family builtins.
    pub fn display(&self) -> String {
        match self {
            Value::Int(i) => format!("{}", i),
            Value::Float(f) => format!("{}", f),
            Value::Str(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Unit => String::from("unit"),
            Value::List(l) => {
                let mut s = String::from("[");
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&v.display());
                }
                s.push(']');
                s
            }
        }
    }
    /// True if two floats are within `eps` (for tests / approximate comparison).
    pub fn approx(&self, f: f64, eps: f64) -> bool {
        self.as_f64().map(|x| local_abs(x - f) <= eps).unwrap_or(false)
    }
}

fn local_abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

// ───────────────────────────── languages ─────────────────────────────

/// A guest language the runtime can host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Python,
    Rust,
    Cpp,
    CSharp,
    JavaScript,
    TypeScript,
    Java,
}

impl Language {
    pub fn name(&self) -> &'static str {
        match self {
            Language::Python => "Python",
            Language::Rust => "Rust",
            Language::Cpp => "C++",
            Language::CSharp => "C#",
            Language::JavaScript => "JavaScript",
            Language::TypeScript => "TypeScript",
            Language::Java => "Java",
        }
    }

    /// Every supported language, in a stable order (drives the test/bench batteries).
    pub fn all() -> [Language; 7] {
        [
            Language::Python,
            Language::Rust,
            Language::Cpp,
            Language::CSharp,
            Language::JavaScript,
            Language::TypeScript,
            Language::Java,
        ]
    }

    fn dialect(&self) -> Dialect {
        match self {
            Language::Python => Dialect {
                line_comment: "#",
                fn_keyword: Some("def"),
                param_name_first: true,
                rust_for: false,
                python: true,
            },
            Language::Rust => Dialect {
                line_comment: "//",
                fn_keyword: Some("fn"),
                param_name_first: true,
                rust_for: true,
                python: false,
            },
            Language::JavaScript => Dialect {
                line_comment: "//",
                fn_keyword: Some("function"),
                param_name_first: true,
                rust_for: false,
                python: false,
            },
            Language::TypeScript => Dialect {
                line_comment: "//",
                fn_keyword: Some("function"),
                param_name_first: true,
                rust_for: false,
                python: false,
            },
            // Typed C-family: no fn keyword; a function is `Type name(params) { ... }`,
            // and a parameter's name is the *last* identifier in its declarator.
            Language::Cpp | Language::CSharp | Language::Java => Dialect {
                line_comment: "//",
                fn_keyword: None,
                param_name_first: false,
                rust_for: false,
                python: false,
            },
        }
    }
}

/// Per-language surface-syntax knobs. The grammar is otherwise shared.
#[derive(Clone, Copy)]
struct Dialect {
    line_comment: &'static str,
    fn_keyword: Option<&'static str>,
    param_name_first: bool,
    rust_for: bool,
    python: bool,
}

// ───────────────────────────── errors ─────────────────────────────

/// Why a guest program failed to parse or run. Every failure is contained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    Parse(String),
    /// A name that resolves to no user function, builtin, or imported package.
    Undefined(String),
    /// A package function used without importing its package (default-closed).
    NotImported(String),
    TypeMismatch(String),
    DivideByZero,
    IndexOutOfBounds,
    BadArity(String),
    /// The step budget was exhausted (runaway guest).
    OutOfGas,
    NoMain,
}




// ───────────────────────────── interpreter ─────────────────────────────

enum Flow {
    Normal(Value),
    Return(Value),
}

/// The outcome of running a guest program.
#[derive(Clone, Debug)]
pub struct Run {
    pub value: Value,
    /// Interpreter steps consumed — the benchmark battery's unit of work.
    pub steps: u64,
    /// Anything the program printed.
    pub output: Vec<String>,
}

struct Interp<'p> {
    prog: &'p Program,
    funcs: BTreeMap<String, &'p Func>,
    steps: u64,
    budget: u64,
    output: Vec<String>,
}

/// Parse and run a guest program with the default step budget.
pub fn run(src: &str, lang: Language) -> Result<Run, RunError> {
    let prog = parse(src, lang)?;
    execute(&prog, DEFAULT_STEP_BUDGET)
}

/// Convert a [`Value`] to its literal [`Expr`] form so it can be spliced as
/// a call argument into a synthesized call statement.
fn value_to_expr(v: Value) -> ast::Expr {
    match v {
        Value::Int(i) => ast::Expr::Int(i),
        Value::Float(f) => ast::Expr::Float(f),
        Value::Bool(b) => ast::Expr::Bool(b),
        Value::Str(s) => ast::Expr::Str(s),
        Value::List(l) => ast::Expr::List(l.into_iter().map(value_to_expr).collect()),
        Value::Unit => ast::Expr::Int(0), // unit → 0 sentinel
    }
}

/// Parse `src` in `lang` and call the named function `fn_name` with the
/// provided argument values.  Definitions in the source are loaded before
/// the call; the function must be defined at the top level.
pub fn call_func(
    src: &str,
    lang: Language,
    fn_name: &str,
    args: Vec<Value>,
) -> Result<Run, RunError> {
    let prog = parse(src, lang)?;
    let mut funcs = BTreeMap::new();
    for f in &prog.funcs {
        funcs.insert(f.name.to_lowercase(), f);
    }
    let key = fn_name.to_lowercase();
    if !funcs.contains_key(&key) {
        return Err(RunError::Undefined(format!("no function '{}' found", fn_name)));
    }
    let arg_exprs: Vec<ast::Expr> = args.into_iter().map(value_to_expr).collect();
    let call_stmt = ast::Stmt::Expr(ast::Expr::Call(vec![key], arg_exprs));
    let mut it = Interp { prog: &prog, funcs, steps: 0, budget: DEFAULT_STEP_BUDGET, output: Vec::new() };
    let mut env = BTreeMap::new();
    let value = match it.exec(&call_stmt, &mut env)? {
        Flow::Return(v) | Flow::Normal(v) => v,
    };
    Ok(Run { value, steps: it.steps, output: it.output })
}

/// Run an already-parsed program with an explicit step budget.
pub fn execute(prog: &Program, budget: u64) -> Result<Run, RunError> {
    let mut funcs = BTreeMap::new();
    for f in &prog.funcs {
        funcs.insert(f.name.to_lowercase(), f);
    }
    let mut it = Interp { prog, funcs, steps: 0, budget, output: Vec::new() };

    // Run top-level statements; the program's value is the last expression's value,
    // or the value returned by a top-level `return`/the last top-level call.
    let mut last = Value::Unit;
    let stmts: Vec<Stmt> = if prog.main.is_empty() {
        // No script body: convention is to call `main`/`run`/`Main`.
        let entry = ["main", "run", "solve"]
            .iter()
            .find(|n| it.funcs.contains_key(**n))
            .ok_or(RunError::NoMain)?;
        vec![Stmt::Expr(Expr::Call(vec![entry.to_string()], Vec::new()))]
    } else {
        prog.main.clone()
    };
    let mut env: BTreeMap<String, Value> = BTreeMap::new();
    for s in &stmts {
        match it.exec(s, &mut env)? {
            Flow::Return(v) => {
                last = v;
                break;
            }
            Flow::Normal(v) => {
                if !matches!(v, Value::Unit) {
                    last = v;
                }
            }
        }
    }
    Ok(Run { value: last, steps: it.steps, output: it.output })
}

impl<'p> Interp<'p> {
    fn tick(&mut self) -> Result<(), RunError> {
        self.steps += 1;
        if self.steps > self.budget {
            return Err(RunError::OutOfGas);
        }
        Ok(())
    }

    fn exec(&mut self, s: &Stmt, env: &mut BTreeMap<String, Value>) -> Result<Flow, RunError> {
        self.tick()?;
        match s {
            Stmt::Let(name, e) | Stmt::Assign(name, e) => {
                let v = self.eval(e, env)?;
                env.insert(name.clone(), v);
                Ok(Flow::Normal(Value::Unit))
            }
            Stmt::IndexAssign(base, idx, val) => {
                let name = match base {
                    Expr::Var(n) => n.clone(),
                    _ => return Err(RunError::TypeMismatch(String::from("index-assign needs a variable"))),
                };
                let i = self.eval(idx, env)?.as_int().ok_or(RunError::TypeMismatch(String::from("index")))?;
                let v = self.eval(val, env)?;
                let list = env.get_mut(&name).ok_or_else(|| RunError::Undefined(name.clone()))?;
                if let Value::List(items) = list {
                    let idx = i as usize;
                    if idx >= items.len() {
                        return Err(RunError::IndexOutOfBounds);
                    }
                    items[idx] = v;
                    Ok(Flow::Normal(Value::Unit))
                } else {
                    Err(RunError::TypeMismatch(String::from("not a list")))
                }
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Unit,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Expr(e) => {
                let v = self.eval(e, env)?;
                Ok(Flow::Normal(v))
            }
            Stmt::If(cond, then, els) => {
                if self.eval(cond, env)?.truthy() {
                    self.exec_block(then, env)
                } else {
                    self.exec_block(els, env)
                }
            }
            Stmt::While(cond, body) => {
                while self.eval(cond, env)?.truthy() {
                    self.tick()?;
                    if let Flow::Return(v) = self.exec_block(body, env)? {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal(Value::Unit))
            }
            Stmt::ForRange(var, range, body) => {
                let (a, b) = match range {
                    Expr::Range(a, b) => (
                        self.eval(a, env)?.as_int().ok_or(RunError::TypeMismatch(String::from("range start")))?,
                        self.eval(b, env)?.as_int().ok_or(RunError::TypeMismatch(String::from("range end")))?,
                    ),
                    _ => return Err(RunError::TypeMismatch(String::from("for-range needs a range"))),
                };
                let mut i = a;
                while i < b {
                    self.tick()?;
                    env.insert(var.clone(), Value::Int(i));
                    if let Flow::Return(v) = self.exec_block(body, env)? {
                        return Ok(Flow::Return(v));
                    }
                    i += 1;
                }
                Ok(Flow::Normal(Value::Unit))
            }
            Stmt::ForEach(var, iter, body) => {
                let list = match self.eval(iter, env)? {
                    Value::List(l) => l,
                    other => return Err(RunError::TypeMismatch(format!("for-each over non-list {:?}", other))),
                };
                for item in list {
                    self.tick()?;
                    env.insert(var.clone(), item);
                    if let Flow::Return(v) = self.exec_block(body, env)? {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal(Value::Unit))
            }
        }
    }

    fn exec_block(&mut self, body: &[Stmt], env: &mut BTreeMap<String, Value>) -> Result<Flow, RunError> {
        for s in body {
            if let Flow::Return(v) = self.exec(s, env)? {
                return Ok(Flow::Return(v));
            }
        }
        Ok(Flow::Normal(Value::Unit))
    }

    fn eval(&mut self, e: &Expr, env: &mut BTreeMap<String, Value>) -> Result<Value, RunError> {
        self.tick()?;
        match e {
            Expr::Int(i) => Ok(Value::Int(*i)),
            Expr::Float(f) => Ok(Value::Float(*f)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::List(items) => {
                let mut vals = Vec::with_capacity(items.len());
                for it in items {
                    vals.push(self.eval(it, env)?);
                }
                Ok(Value::List(vals))
            }
            Expr::Var(name) => env.get(name).cloned().ok_or_else(|| RunError::Undefined(name.clone())),
            Expr::Neg(inner) => match self.eval(inner, env)? {
                Value::Int(i) => Ok(Value::Int(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(RunError::TypeMismatch(String::from("negate non-number"))),
            },
            Expr::Not(inner) => Ok(Value::Bool(!self.eval(inner, env)?.truthy())),
            Expr::Range(a, b) => {
                // Materialize as a list (allows `for x in 0..n` and list-style use).
                let a = self.eval(a, env)?.as_int().ok_or(RunError::TypeMismatch(String::from("range")))?;
                let b = self.eval(b, env)?.as_int().ok_or(RunError::TypeMismatch(String::from("range")))?;
                let mut v = Vec::new();
                let mut i = a;
                while i < b {
                    v.push(Value::Int(i));
                    i += 1;
                }
                Ok(Value::List(v))
            }
            Expr::Index(base, idx) => {
                let b = self.eval(base, env)?;
                let i = self.eval(idx, env)?.as_int().ok_or(RunError::TypeMismatch(String::from("index")))?;
                match b {
                    Value::List(l) => l.get(i as usize).cloned().ok_or(RunError::IndexOutOfBounds),
                    Value::Str(s) => s
                        .chars()
                        .nth(i as usize)
                        .map(|c| Value::Str(c.to_string()))
                        .ok_or(RunError::IndexOutOfBounds),
                    _ => Err(RunError::TypeMismatch(String::from("index non-list"))),
                }
            }
            Expr::Bin(op, l, r) => {
                // Short-circuit boolean ops.
                if matches!(op, BinOp::And) {
                    let lv = self.eval(l, env)?;
                    if !lv.truthy() {
                        return Ok(Value::Bool(false));
                    }
                    return Ok(Value::Bool(self.eval(r, env)?.truthy()));
                }
                if matches!(op, BinOp::Or) {
                    let lv = self.eval(l, env)?;
                    if lv.truthy() {
                        return Ok(Value::Bool(true));
                    }
                    return Ok(Value::Bool(self.eval(r, env)?.truthy()));
                }
                let lv = self.eval(l, env)?;
                let rv = self.eval(r, env)?;
                binop(op, lv, rv)
            }
            Expr::Call(path, args) => self.eval_call(path, args, env),
        }
    }

    fn eval_call(
        &mut self,
        path: &[String],
        args: &[Expr],
        env: &mut BTreeMap<String, Value>,
    ) -> Result<Value, RunError> {
        // Method-style call on a variable: `recv.method(args)`.
        if path.len() >= 2 {
            if let Some(recv) = env.get(&path[0]).cloned() {
                let method = path.last().unwrap().to_lowercase();
                // In-place list mutation under each language's idiomatic name:
                // Python `append`, Rust/JS `push`, Java/C# `add`, C++ `push_back`.
                if matches!(method.as_str(), "append" | "push" | "add" | "push_back")
                    && matches!(recv, Value::List(_))
                {
                    if args.len() != 1 {
                        return Err(RunError::BadArity(method));
                    }
                    let v = self.eval(&args[0], env)?;
                    if let Some(Value::List(items)) = env.get_mut(&path[0]) {
                        items.push(v);
                    }
                    return Ok(Value::Unit);
                }
                // Value-semantics method: pass the receiver as the first arg.
                let mut argv = Vec::with_capacity(args.len() + 1);
                argv.push(recv);
                for a in args {
                    argv.push(self.eval(a, env)?);
                }
                return self.dispatch(&method, argv, env);
            }
        }

        let name = path.last().unwrap().to_lowercase();

        // Idiomatic print sinks: console.log / System.out.println / Console.WriteLine.
        if is_print_call(path) {
            let mut parts = Vec::new();
            for a in args {
                parts.push(self.eval(a, env)?.display());
            }
            self.output.push(parts.join(" "));
            return Ok(Value::Unit);
        }

        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            argv.push(self.eval(a, env)?);
        }
        self.dispatch(&name, argv, env)
    }

    /// Resolve a call by canonical (lowercased) name to a user function, a builtin,
    /// or an imported package function — enforcing the import (default-closed).
    fn dispatch(
        &mut self,
        name: &str,
        argv: Vec<Value>,
        _env: &mut BTreeMap<String, Value>,
    ) -> Result<Value, RunError> {
        // 1. User-defined function.
        if let Some(f) = self.funcs.get(name).copied() {
            return self.call_user(f, argv);
        }
        // 2. Always-available builtin — `print` emits to output before delegating.
        if name == "print" {
            let line = argv.iter().map(|v| v.display()).collect::<Vec<_>>().join(" ");
            self.output.push(line);
            return Ok(Value::Unit);
        }
        if BUILTINS.contains(&name) {
            return builtin(name, &argv);
        }
        // 3. Package (library) function — must have been imported.
        if let Some(pkg) = package_of(name) {
            if self.prog.imports.contains(&pkg) {
                return library(name, &argv);
            }
            return Err(RunError::NotImported(format!("{} (provided by package '{}')", name, pkg)));
        }
        Err(RunError::Undefined(name.to_string()))
    }

    fn call_user(&mut self, f: &'p Func, argv: Vec<Value>) -> Result<Value, RunError> {
        if argv.len() != f.params.len() {
            return Err(RunError::BadArity(f.name.clone()));
        }
        let mut local: BTreeMap<String, Value> = BTreeMap::new();
        for (p, v) in f.params.iter().zip(argv) {
            local.insert(p.clone(), v);
        }
        for s in &f.body {
            if let Flow::Return(v) = self.exec(s, &mut local)? {
                return Ok(v);
            }
        }
        Ok(Value::Unit)
    }
}

fn is_print_call(path: &[String]) -> bool {
    let last = path.last().map(|s| s.to_lowercase()).unwrap_or_default();
    let head = path.first().map(|s| s.to_lowercase()).unwrap_or_default();
    matches!(last.as_str(), "log" | "println" | "writeline" | "write" | "print")
        && matches!(head.as_str(), "console" | "system" | "std")
        || matches!(last.as_str(), "println" | "writeline") // Console.WriteLine etc.
}

fn binop(op: &BinOp, l: Value, r: Value) -> Result<Value, RunError> {
    use BinOp::*;
    // String concatenation with '+'.
    if matches!(op, Add) {
        if let (Value::Str(a), b) = (&l, &r) {
            return Ok(Value::Str(format!("{}{}", a, b.display())));
        }
        if let (a, Value::Str(b)) = (&l, &r) {
            return Ok(Value::Str(format!("{}{}", a.display(), b)));
        }
    }
    // Equality / inequality across any matching types.
    if matches!(op, Eq) {
        return Ok(Value::Bool(value_eq(&l, &r)));
    }
    if matches!(op, Ne) {
        return Ok(Value::Bool(!value_eq(&l, &r)));
    }
    // Float path if either side is a float.
    let float = matches!(l, Value::Float(_)) || matches!(r, Value::Float(_));
    if float {
        let a = l.as_f64().ok_or(RunError::TypeMismatch(String::from("arith")))?;
        let b = r.as_f64().ok_or(RunError::TypeMismatch(String::from("arith")))?;
        return Ok(match op {
            Add => Value::Float(a + b),
            Sub => Value::Float(a - b),
            Mul => Value::Float(a * b),
            Div => {
                if b == 0.0 {
                    return Err(RunError::DivideByZero);
                }
                Value::Float(a / b)
            }
            Rem => Value::Float(a - b * (a / b) as i64 as f64),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            _ => return Err(RunError::TypeMismatch(String::from("bad float op"))),
        });
    }
    let a = l.as_int().ok_or(RunError::TypeMismatch(String::from("arith")))?;
    let b = r.as_int().ok_or(RunError::TypeMismatch(String::from("arith")))?;
    Ok(match op {
        Add => Value::Int(a.wrapping_add(b)),
        Sub => Value::Int(a.wrapping_sub(b)),
        Mul => Value::Int(a.wrapping_mul(b)),
        Div => {
            if b == 0 {
                return Err(RunError::DivideByZero);
            }
            Value::Int(a.wrapping_div(b))
        }
        Rem => {
            if b == 0 {
                return Err(RunError::DivideByZero);
            }
            Value::Int(a.wrapping_rem(b))
        }
        Lt => Value::Bool(a < b),
        Le => Value::Bool(a <= b),
        Gt => Value::Bool(a > b),
        Ge => Value::Bool(a >= b),
        _ => return Err(RunError::TypeMismatch(String::from("bad int op"))),
    })
}

fn value_eq(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => *a as f64 == *b,
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::List(a), Value::List(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

// ───────────────────────── standard library / packages ─────────────────────────

/// Builtins available to every guest without an import.
const BUILTINS: &[&str] = &["print", "len", "abs", "int", "float", "str", "bool", "sum", "append", "sorted", "min", "max"];

fn builtin(name: &str, a: &[Value]) -> Result<Value, RunError> {
    match name {
        "print" => Ok(Value::Unit),
        "len" => match a.first() {
            Some(Value::List(l)) => Ok(Value::Int(l.len() as i64)),
            Some(Value::Str(s)) => Ok(Value::Int(s.chars().count() as i64)),
            _ => Err(RunError::TypeMismatch(String::from("len"))),
        },
        "abs" => match a.first() {
            Some(Value::Int(i)) => Ok(Value::Int(i.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(local_abs(*f))),
            _ => Err(RunError::TypeMismatch(String::from("abs"))),
        },
        "int" => a.first().and_then(|v| v.as_int()).map(Value::Int).ok_or(RunError::TypeMismatch(String::from("int"))),
        "float" => a.first().and_then(|v| v.as_f64()).map(Value::Float).ok_or(RunError::TypeMismatch(String::from("float"))),
        "str" => Ok(Value::Str(a.first().map(|v| v.display()).unwrap_or_default())),
        "bool" => Ok(Value::Bool(a.first().map(|v| v.truthy()).unwrap_or(false))),
        "sum" => match a.first() {
            Some(Value::List(l)) => {
                let any_float = l.iter().any(|v| matches!(v, Value::Float(_)));
                if any_float {
                    let mut s = 0.0;
                    for v in l {
                        s += v.as_f64().ok_or(RunError::TypeMismatch(String::from("sum")))?;
                    }
                    Ok(Value::Float(s))
                } else {
                    let mut s = 0i64;
                    for v in l {
                        s = s.wrapping_add(v.as_int().ok_or(RunError::TypeMismatch(String::from("sum")))?);
                    }
                    Ok(Value::Int(s))
                }
            }
            _ => Err(RunError::TypeMismatch(String::from("sum"))),
        },
        "append" => match a.first() {
            Some(Value::List(l)) => {
                let mut l = l.clone();
                if let Some(v) = a.get(1) {
                    l.push(v.clone());
                }
                Ok(Value::List(l))
            }
            _ => Err(RunError::TypeMismatch(String::from("append"))),
        },
        "sorted" => match a.first() {
            Some(Value::List(l)) => {
                let mut l = l.clone();
                l.sort_by(|x, y| {
                    let xf = x.as_f64().unwrap_or(0.0);
                    let yf = y.as_f64().unwrap_or(0.0);
                    xf.partial_cmp(&yf).unwrap_or(core::cmp::Ordering::Equal)
                });
                Ok(Value::List(l))
            }
            _ => Err(RunError::TypeMismatch(String::from("sorted"))),
        },
        "min" | "max" => {
            let want_max = name == "max";
            let pool: Vec<Value> = match a.first() {
                Some(Value::List(l)) if a.len() == 1 => l.clone(),
                _ => a.to_vec(),
            };
            let mut best: Option<f64> = None;
            let mut best_v: Option<Value> = None;
            for v in pool {
                let f = v.as_f64().ok_or(RunError::TypeMismatch(String::from("min/max")))?;
                let take = match best {
                    None => true,
                    Some(b) => {
                        if want_max {
                            f > b
                        } else {
                            f < b
                        }
                    }
                };
                if take {
                    best = Some(f);
                    best_v = Some(v);
                }
            }
            best_v.ok_or(RunError::TypeMismatch(String::from("min/max of empty")))
        }
        _ => Err(RunError::Undefined(name.to_string())),
    }
}

/// Which package provides a given (lowercased) function name, if any.
fn package_of(name: &str) -> Option<&'static str> {
    match name {
        "sqrt" | "pow" | "gcd" | "factorial" | "isqrt" | "floor" | "ceil" | "powi" | "fib" => Some("mathx"),
        "mean" | "variance" | "stdev" | "pstdev" | "median" | "pvariance" => Some("stats"),
        "upper" | "lower" | "repeat" | "reverse_str" | "concat" => Some("strx"),
        _ => None,
    }
}

/// Execute a package (library) function. Reached only after the import check passes.
fn library(name: &str, a: &[Value]) -> Result<Value, RunError> {
    match name {
        // ── mathx ──
        "sqrt" => Ok(Value::Float(sqrt(arg_f(a, 0)?))),
        "pow" | "powi" => {
            let base = arg_f(a, 0)?;
            let exp = arg_i(a, 1)?;
            let mut r = 1.0f64;
            let mut e = exp.unsigned_abs();
            let mut b = base;
            while e > 0 {
                if e & 1 == 1 {
                    r *= b;
                }
                b *= b;
                e >>= 1;
            }
            if exp < 0 {
                r = 1.0 / r;
            }
            // Integer result when inputs were integral and exp >= 0.
            if exp >= 0 && a.first().map(|v| matches!(v, Value::Int(_))).unwrap_or(false) {
                Ok(Value::Int(r as i64))
            } else {
                Ok(Value::Float(r))
            }
        }
        "gcd" => {
            let mut x = arg_i(a, 0)?.abs();
            let mut y = arg_i(a, 1)?.abs();
            while y != 0 {
                let t = y;
                y = x % y;
                x = t;
            }
            Ok(Value::Int(x))
        }
        "factorial" => {
            let n = arg_i(a, 0)?;
            if n < 0 {
                return Err(RunError::TypeMismatch(String::from("factorial of negative")));
            }
            let mut r = 1i64;
            for k in 2..=n {
                r = r.wrapping_mul(k);
            }
            Ok(Value::Int(r))
        }
        "isqrt" => {
            let n = arg_i(a, 0)?;
            if n < 0 {
                return Err(RunError::TypeMismatch(String::from("isqrt of negative")));
            }
            let mut x = n;
            let mut y = (x + 1) / 2;
            if n == 0 {
                return Ok(Value::Int(0));
            }
            while y < x {
                x = y;
                y = (x + n / x) / 2;
            }
            Ok(Value::Int(x))
        }
        "fib" => {
            let n = arg_i(a, 0)?;
            let (mut p, mut q) = (0i64, 1i64);
            for _ in 0..n {
                let t = p.wrapping_add(q);
                p = q;
                q = t;
            }
            Ok(Value::Int(p))
        }
        "floor" => Ok(Value::Int(floor(arg_f(a, 0)?) as i64)),
        "ceil" => Ok(Value::Int(ceil(arg_f(a, 0)?) as i64)),

        // ── stats ──
        "mean" => Ok(Value::Float(mean(list_f(a)?.as_slice()))),
        "variance" | "pvariance" => {
            let xs = list_f(a)?;
            let sample = name == "variance";
            Ok(Value::Float(variance(&xs, sample)))
        }
        "stdev" | "pstdev" => {
            let xs = list_f(a)?;
            let sample = name == "stdev";
            Ok(Value::Float(sqrt(variance(&xs, sample))))
        }
        "median" => {
            let mut xs = list_f(a)?;
            xs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal));
            if xs.is_empty() {
                return Err(RunError::TypeMismatch(String::from("median of empty")));
            }
            let m = xs.len() / 2;
            if xs.len() % 2 == 1 {
                Ok(Value::Float(xs[m]))
            } else {
                Ok(Value::Float((xs[m - 1] + xs[m]) / 2.0))
            }
        }

        // ── strx ──
        "upper" => Ok(Value::Str(arg_s(a, 0)?.to_uppercase())),
        "lower" => Ok(Value::Str(arg_s(a, 0)?.to_lowercase())),
        "repeat" => {
            let s = arg_s(a, 0)?;
            let n = arg_i(a, 1)?.max(0) as usize;
            let mut out = String::new();
            for _ in 0..n {
                out.push_str(&s);
            }
            Ok(Value::Str(out))
        }
        "reverse_str" => Ok(Value::Str(arg_s(a, 0)?.chars().rev().collect())),
        "concat" => Ok(Value::Str(format!("{}{}", arg_s(a, 0)?, arg_s(a, 1)?))),
        _ => Err(RunError::Undefined(name.to_string())),
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], sample: bool) -> f64 {
    let n = xs.len();
    if n == 0 || (sample && n < 2) {
        return 0.0;
    }
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|x| (x - m) * (x - m)).sum();
    let denom = if sample { (n - 1) as f64 } else { n as f64 };
    ss / denom
}

fn arg_f(a: &[Value], i: usize) -> Result<f64, RunError> {
    a.get(i).and_then(|v| v.as_f64()).ok_or(RunError::TypeMismatch(format!("arg {} must be a number", i)))
}
fn arg_i(a: &[Value], i: usize) -> Result<i64, RunError> {
    a.get(i).and_then(|v| v.as_int()).ok_or(RunError::TypeMismatch(format!("arg {} must be an int", i)))
}
fn arg_s(a: &[Value], i: usize) -> Result<String, RunError> {
    match a.get(i) {
        Some(Value::Str(s)) => Ok(s.clone()),
        Some(other) => Ok(other.display()),
        None => Err(RunError::TypeMismatch(format!("arg {} must be a string", i))),
    }
}
fn list_f(a: &[Value]) -> Result<Vec<f64>, RunError> {
    match a.first() {
        Some(Value::List(l)) => l.iter().map(|v| v.as_f64().ok_or(RunError::TypeMismatch(String::from("list of numbers")))).collect(),
        _ => Err(RunError::TypeMismatch(String::from("expected a list"))),
    }
}

/// Map an import spelling (in any language) to a canonical package, dropping any
/// that name nothing the runtime provides.
fn canon_pkgs(spellings: &[String]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for s in spellings {
        let norm = s
            .trim_matches('"')
            .trim_end_matches(".hpp")
            .trim_end_matches(".h")
            .rsplit(['/', '.'])
            .next()
            .unwrap_or("")
            .to_lowercase();
        let canon = match norm.as_str() {
            "statistics" | "stats" => Some("stats"),
            "math" | "cmath" | "numeric" | "mathx" => Some("mathx"),
            "strx" | "string" | "str" | "cstring" => Some("strx"),
            _ => None,
        };
        if let Some(c) = canon {
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    out
}

// ───────────────────────── canonical demo + benchmark programs ─────────────────────────
//
// The SAME algorithm, written in each language's real surface syntax, so the test
// and benchmark batteries (host and on-metal) prove every guest runs identical
// multi-function, package-importing programs to an identical result. The demo:
//
//   * imports the `stats` and `mathx` library packages (idiomatic per language),
//   * `scale(xs, k)` builds a new list (loop + idiomatic list append),
//   * `summary(xs)` calls `gcd` (mathx) and population `stdev` (stats), and
//   * the entry point returns `summary(scale([2,4,4,4,5,5,7,9], 3))`.
//
// pstdev([6,12,12,12,15,15,21,27]) = 6.0 and gcd(48,36) = 12, so every language
// must return exactly **18.0**.

/// The exact value every language's demo program must produce.
pub const DEMO_EXPECTED: f64 = 18.0;

/// The checksum every language's benchmark program must produce (`sum of gcd(i,36)`
/// for i in 1..2000) — a cross-language equivalence proof.
pub const BENCH_EXPECTED: i64 = 9316;

/// The canonical multi-function, package-using demo program for `lang`.
pub fn demo_program(lang: Language) -> &'static str {
    match lang {
        Language::Python => PY_DEMO,
        Language::Rust => RS_DEMO,
        Language::Cpp => CPP_DEMO,
        Language::CSharp => CS_DEMO,
        Language::JavaScript => JS_DEMO,
        Language::TypeScript => TS_DEMO,
        Language::Java => JAVA_DEMO,
    }
}

/// A compute-heavy benchmark program for `lang` (a `gcd`-folding loop over a library
/// call). All seven are algorithmically identical, so the result is a checksum.
pub fn bench_program(lang: Language) -> &'static str {
    match lang {
        Language::Python => PY_BENCH,
        Language::Rust => RS_BENCH,
        Language::Cpp => CPP_BENCH,
        Language::CSharp => CS_BENCH,
        Language::JavaScript => JS_BENCH,
        Language::TypeScript => TS_BENCH,
        Language::Java => JAVA_BENCH,
    }
}

const PY_DEMO: &str = r#"
import statistics
from mathx import gcd

def scale(xs, k):
    out = []
    for x in xs:
        out.append(x * k)
    return out

def summary(xs):
    g = gcd(48, 36)
    s = statistics.pstdev(xs)
    return s + g

def run():
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    scaled = scale(data, 3)
    return summary(scaled)
"#;

const RS_DEMO: &str = r#"
use stats::pstdev;
use mathx::gcd;

fn scale(xs, k) {
    let mut out = vec![];
    for x in xs {
        out.push(x * k);
    }
    return out;
}

fn summary(xs) {
    let g = gcd(48, 36);
    let s = pstdev(xs);
    return s + g;
}

fn run() {
    let data = vec![2, 4, 4, 4, 5, 5, 7, 9];
    let scaled = scale(data, 3);
    return summary(scaled);
}
"#;

const CPP_DEMO: &str = r#"
#include <numeric>
#include "stats.hpp"

auto scale(auto xs, int k) {
    auto out = [];
    for (int i = 0; i < len(xs); i = i + 1) {
        out.push_back(xs[i] * k);
    }
    return out;
}

double summary(auto xs) {
    int g = gcd(48, 36);
    double s = stats::pstdev(xs);
    return s + g;
}

double run() {
    auto data = [2, 4, 4, 4, 5, 5, 7, 9];
    auto scaled = scale(data, 3);
    return summary(scaled);
}
"#;

const CS_DEMO: &str = r#"
using Stats;
using MathX;

class Program {
    static var scale(var xs, int k) {
        var out = [];
        for (int i = 0; i < len(xs); i = i + 1) {
            out.Add(xs[i] * k);
        }
        return out;
    }

    static double summary(var xs) {
        int g = MathX.Gcd(48, 36);
        double s = Stats.Pstdev(xs);
        return s + g;
    }

    static double run() {
        var data = [2, 4, 4, 4, 5, 5, 7, 9];
        var scaled = scale(data, 3);
        return summary(scaled);
    }
}
"#;

const JS_DEMO: &str = r#"
const stats = require('stats');
const mathx = require('mathx');

function scale(xs, k) {
    let out = [];
    for (let i = 0; i < len(xs); i = i + 1) {
        out.push(xs[i] * k);
    }
    return out;
}

function summary(xs) {
    let g = mathx.gcd(48, 36);
    let s = stats.pstdev(xs);
    return s + g;
}

function run() {
    let data = [2, 4, 4, 4, 5, 5, 7, 9];
    let scaled = scale(data, 3);
    return summary(scaled);
}
"#;

const TS_DEMO: &str = r#"
import { pstdev } from 'stats';
import { gcd } from 'mathx';

function scale(xs: number[], k: number): number[] {
    let out: number[] = [];
    for (let i = 0; i < len(xs); i = i + 1) {
        out.push(xs[i] * k);
    }
    return out;
}

function summary(xs: number[]): number {
    let g: number = gcd(48, 36);
    let s: number = pstdev(xs);
    return s + g;
}

function run(): number {
    let data: number[] = [2, 4, 4, 4, 5, 5, 7, 9];
    let scaled: number[] = scale(data, 3);
    return summary(scaled);
}
"#;

const JAVA_DEMO: &str = r#"
import stats.*;
import mathx.*;

class Program {
    static var scale(var xs, int k) {
        var out = [];
        for (int i = 0; i < len(xs); i = i + 1) {
            out.add(xs[i] * k);
        }
        return out;
    }

    static double summary(var xs) {
        int g = gcd(48, 36);
        double s = pstdev(xs);
        return s + g;
    }

    static double run() {
        var data = [2, 4, 4, 4, 5, 5, 7, 9];
        var scaled = scale(data, 3);
        return summary(scaled);
    }
}
"#;

const PY_BENCH: &str = r#"
from mathx import gcd

def work(n):
    total = 0
    for i in range(1, n):
        total = total + gcd(i, 36)
    return total

def run():
    return work(2000)
"#;

const RS_BENCH: &str = r#"
use mathx::gcd;

fn work(n) {
    let mut total = 0;
    for i in 1..n {
        total = total + gcd(i, 36);
    }
    return total;
}

fn run() {
    return work(2000);
}
"#;

const CPP_BENCH: &str = r#"
#include <numeric>

long work(int n) {
    long total = 0;
    for (int i = 1; i < n; i = i + 1) {
        total = total + gcd(i, 36);
    }
    return total;
}

long run() {
    return work(2000);
}
"#;

const CS_BENCH: &str = r#"
using MathX;

class Program {
    static long work(int n) {
        long total = 0;
        for (int i = 1; i < n; i = i + 1) {
            total = total + MathX.Gcd(i, 36);
        }
        return total;
    }
    static long run() {
        return work(2000);
    }
}
"#;

const JS_BENCH: &str = r#"
const mathx = require('mathx');

function work(n) {
    let total = 0;
    for (let i = 1; i < n; i = i + 1) {
        total = total + mathx.gcd(i, 36);
    }
    return total;
}

function run() {
    return work(2000);
}
"#;

const TS_BENCH: &str = r#"
import { gcd } from 'mathx';

function work(n: number): number {
    let total: number = 0;
    for (let i = 1; i < n; i = i + 1) {
        total = total + gcd(i, 36);
    }
    return total;
}

function run(): number {
    return work(2000);
}
"#;

const JAVA_BENCH: &str = r#"
import mathx.*;

class Program {
    static long work(int n) {
        long total = 0;
        for (int i = 1; i < n; i = i + 1) {
            total = total + gcd(i, 36);
        }
        return total;
    }
    static long run() {
        return work(2000);
    }
}
"#;

#[cfg(test)]
mod tests;

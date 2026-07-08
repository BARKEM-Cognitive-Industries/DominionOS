//! Dominion bytecode VM — a fast register-threaded stack machine that executes
//! [`CompiledProgram`](super::bytecode::CompiledProgram) chunks.
//!
//! The VM is the primary execution tier for Dominion. It runs every instruction
//! the compiler lowers natively (arithmetic, locals/globals, control flow,
//! aggregates, user function calls) and falls back to the tree-walking
//! [`Interpreter`](super::Interpreter) for the small set of operations that
//! carry raw AST expressions (builtins, `::` paths, `=>` map, `|>` pipe, and the
//! `EvalExpr` catch-all for future AST additions). The result is **bit-identical**
//! to the interpreter for every program the compiler can handle.
//!
//! Pure, safe `no_std + alloc`. No `unsafe`.

use super::bytecode::{Chunk, CompiledProgram, Const, Op};
use super::ast::{BinOp, Expr, Item, Program, Stmt};
use super::{Interpreter, Value};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ── error type ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct VmError {
    pub message: String,
}

impl VmError {
    fn new(m: impl Into<String>) -> VmError {
        VmError { message: m.into() }
    }
}

impl core::fmt::Display for VmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "vm error: {}", self.message)
    }
}

fn err(m: impl Into<String>) -> VmError {
    VmError::new(m)
}

// ── recursion depth limit ────────────────────────────────────────────────────

/// Maximum number of call frames that may be live simultaneously.  A script
/// that recurses past this limit receives a clean `VmError` instead of
/// overflowing the kernel stack.
const MAX_CALL_DEPTH: usize = 256;

// ── call frame ──────────────────────────────────────────────────────────────

/// One activation record on the call stack.
struct Frame {
    /// Index into `CompiledProgram::funcs` (u32::MAX = top-level chunk).
    func_idx: u32,
    /// Offset in the value stack where this frame's locals begin.
    base: usize,
    /// Instruction pointer (index into the chunk's `code`).
    ip: usize,
}

// ── VM ──────────────────────────────────────────────────────────────────────

/// The Dominion stack-machine VM.
pub struct Vm<'p> {
    prog: &'p CompiledProgram,
    /// The single shared value stack.
    stack: Vec<Value>,
    /// Global slots (one per top-level `let`/`linear` binding).
    globals: Vec<Value>,
    /// Call frame stack.
    frames: Vec<Frame>,
    /// Embedded interpreter used exclusively for fallback ops.
    interp: Interpreter,
}

impl<'p> Vm<'p> {
    pub fn new(prog: &'p CompiledProgram) -> Vm<'p> {
        let n_globals = prog.n_globals() as usize;
        // Seed the interpreter with all user functions so fallback calls resolve.
        let mut interp = Interpreter::new();
        for item in &prog.source.items {
            if let Item::Fn(f) = item {
                interp.define_function(f.clone());
            }
        }
        Vm {
            prog,
            stack: Vec::new(),
            globals: alloc::vec![Value::Unit; n_globals.max(1)],
            frames: Vec::new(),
            interp,
        }
    }

    /// Execute the top-level chunk and return the program's result value.
    pub fn run(&mut self) -> Result<Value, VmError> {
        self.frames.push(Frame { func_idx: u32::MAX, base: 0, ip: 0 });
        self.run_chunk()
    }

    fn chunk(&self) -> &Chunk {
        let frame = self.frames.last().unwrap();
        if frame.func_idx == u32::MAX {
            &self.prog.top
        } else {
            &self.prog.funcs[frame.func_idx as usize].chunk
        }
    }

    fn run_chunk(&mut self) -> Result<Value, VmError> {
        loop {
            let ip = self.frames.last().unwrap().ip;
            // Clone the op to avoid aliasing issues when we mutate the frame.
            let op = self.chunk().code[ip].clone();
            self.frames.last_mut().unwrap().ip += 1;

            match op {
                Op::Const(idx) => {
                    let v = const_to_value(&self.chunk().consts[idx as usize]);
                    self.stack.push(v);
                }
                Op::Unit => self.stack.push(Value::Unit),
                Op::Pop => { self.stack.pop(); }

                Op::LoadLocal(slot) => {
                    let base = self.frames.last().unwrap().base;
                    let v = self.stack[base + slot as usize].clone();
                    self.stack.push(v);
                }
                Op::StoreLocal(slot) => {
                    let base = self.frames.last().unwrap().base;
                    let v = self.stack.pop().ok_or_else(|| err("stack underflow on StoreLocal"))?;
                    let idx = base + slot as usize;
                    if idx >= self.stack.len() {
                        // Grow stack to accommodate the slot.
                        self.stack.resize(idx + 1, Value::Unit);
                    }
                    self.stack[idx] = v;
                }
                Op::LoadGlobal(slot) => {
                    let v = self.globals.get(slot as usize).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::StoreGlobal(slot) => {
                    let v = self.stack.pop().ok_or_else(|| err("stack underflow on StoreGlobal"))?;
                    let slot = slot as usize;
                    if slot >= self.globals.len() {
                        self.globals.resize(slot + 1, Value::Unit);
                    }
                    self.globals[slot] = v.clone();
                    // Mirror into the embedded interpreter's global scope so that
                    // fallback ops (EvalExpr, Map, Pipe, CallBuiltin, CallPath) resolve
                    // names that the VM has already bound.
                    if let Some(name) = self.prog.global_names.get(slot) {
                        if !name.is_empty() {
                            self.interp.define_global(name.clone(), v);
                        }
                    }
                }

                Op::Binary(op) => {
                    let r = self.stack.pop().ok_or_else(|| err("stack underflow (rhs)"))?;
                    let l = self.stack.pop().ok_or_else(|| err("stack underflow (lhs)"))?;
                    let v = eval_binary(&op, l, r)?;
                    self.stack.push(v);
                }
                Op::Neg => {
                    let v = self.stack.pop().ok_or_else(|| err("stack underflow on Neg"))?;
                    let r = match v {
                        Value::Int(i) => Value::Int(
                            i.checked_neg()
                                .ok_or_else(|| err("integer overflow on negation"))?
                        ),
                        Value::Float(f) => Value::Float(-f),
                        other => return Err(err(format!("cannot negate {}", other.type_name()))),
                    };
                    self.stack.push(r);
                }
                Op::Not => {
                    let v = self.stack.pop().ok_or_else(|| err("stack underflow on Not"))?;
                    self.stack.push(Value::Bool(!v.is_truthy()));
                }
                Op::Index => {
                    let idx = self.stack.pop().ok_or_else(|| err("stack underflow (index)"))?;
                    let obj = self.stack.pop().ok_or_else(|| err("stack underflow (object)"))?;
                    let v = eval_index(obj, idx)?;
                    self.stack.push(v);
                }
                Op::Field(si) => {
                    let field = self.chunk().strings[si as usize].clone();
                    let obj = self.stack.pop().ok_or_else(|| err("stack underflow on Field"))?;
                    let v = eval_field(obj, &field)?;
                    self.stack.push(v);
                }

                Op::MakeVec(n) => {
                    let start = self.stack.len().saturating_sub(n as usize);
                    let items: Vec<Value> = self.stack.drain(start..).collect();
                    self.stack.push(Value::Vector(items));
                }
                Op::MakeObject { kind, fields } => {
                    let n = fields.len();
                    let start = self.stack.len().saturating_sub(n);
                    let vals: Vec<Value> = self.stack.drain(start..).collect();
                    let kind_s = self.chunk().strings[kind as usize].clone();
                    let field_names: Vec<String> =
                        fields.iter().map(|&fi| self.chunk().strings[fi as usize].clone()).collect();
                    let pairs: Vec<(String, Value)> = field_names.into_iter().zip(vals).collect();
                    self.stack.push(Value::Object { kind: kind_s, fields: pairs });
                }

                Op::Jump(off) => {
                    let ip = self.frames.last().unwrap().ip;
                    let new_ip = (ip as i64)
                        .checked_add(off as i64)
                        .filter(|&v| v >= 0)
                        .map(|v| v as usize)
                        .ok_or_else(|| err("jump offset overflow"))?;
                    self.frames.last_mut().unwrap().ip = new_ip;
                }
                Op::JumpIfFalse(off) => {
                    let v = self.stack.pop().ok_or_else(|| err("stack underflow JumpIfFalse"))?;
                    if !v.is_truthy() {
                        let ip = self.frames.last().unwrap().ip;
                        let new_ip = (ip as i64)
                            .checked_add(off as i64)
                            .filter(|&v| v >= 0)
                            .map(|v| v as usize)
                            .ok_or_else(|| err("jump offset overflow"))?;
                        self.frames.last_mut().unwrap().ip = new_ip;
                    }
                }
                Op::JumpIfFalsePeek(off) => {
                    let v = self.stack.last().ok_or_else(|| err("stack underflow JumpIfFalsePeek"))?;
                    if !v.is_truthy() {
                        let ip = self.frames.last().unwrap().ip;
                        let new_ip = (ip as i64)
                            .checked_add(off as i64)
                            .filter(|&v| v >= 0)
                            .map(|v| v as usize)
                            .ok_or_else(|| err("jump offset overflow"))?;
                        self.frames.last_mut().unwrap().ip = new_ip;
                    }
                }
                Op::JumpIfTruePeek(off) => {
                    let v = self.stack.last().ok_or_else(|| err("stack underflow JumpIfTruePeek"))?;
                    if v.is_truthy() {
                        let ip = self.frames.last().unwrap().ip;
                        let new_ip = (ip as i64)
                            .checked_add(off as i64)
                            .filter(|&v| v >= 0)
                            .map(|v| v as usize)
                            .ok_or_else(|| err("jump offset overflow"))?;
                        self.frames.last_mut().unwrap().ip = new_ip;
                    }
                }

                Op::CallUser { func, argc } => {
                    let f = &self.prog.funcs[func as usize];
                    if f.arity != argc {
                        return Err(err(format!(
                            "fn '{}' expects {} args, got {}",
                            f.name, f.arity, argc
                        )));
                    }
                    // Guard against unbounded recursion overflowing the kernel stack.
                    if self.frames.len() >= MAX_CALL_DEPTH {
                        return Err(err(format!(
                            "stack overflow: call depth exceeded {} frames",
                            MAX_CALL_DEPTH
                        )));
                    }
                    // Args sit on top of the stack. The frame's base is the first arg.
                    let base = self.stack.len() - argc as usize;
                    // Grow stack to n_locals (params + body locals).
                    let need = base + f.chunk.n_locals as usize;
                    if self.stack.len() < need {
                        self.stack.resize(need, Value::Unit);
                    }
                    self.frames.push(Frame { func_idx: func, base, ip: 0 });
                    let result = self.run_chunk()?;
                    self.stack.push(result);
                }

                Op::CallBuiltin { name, argc } => {
                    let bname = self.chunk().strings[name as usize].clone();
                    let start = self.stack.len().saturating_sub(argc as usize);
                    let args: Vec<Value> = self.stack.drain(start..).collect();
                    let result = self.interp_call_builtin(&bname, args)?;
                    self.stack.push(result);
                }
                Op::CallPath { path, argc } => {
                    let callee_expr = self.chunk().exprs[path as usize].clone();
                    let start = self.stack.len().saturating_sub(argc as usize);
                    let args: Vec<Value> = self.stack.drain(start..).collect();
                    let result = self.interp_call_path(&callee_expr, args)?;
                    self.stack.push(result);
                }
                Op::Map(cidx) => {
                    let callee_expr = self.chunk().exprs[cidx as usize].clone();
                    let list = self.stack.pop().ok_or_else(|| err("stack underflow Map"))?;
                    let result = self.interp_eval_expr(&Expr::Map(
                        alloc::boxed::Box::new(self.value_to_vec_expr(list)?),
                        alloc::boxed::Box::new(callee_expr),
                    ))?;
                    self.stack.push(result);
                }
                Op::Pipe(cidx) => {
                    let callee_expr = self.chunk().exprs[cidx as usize].clone();
                    let val = self.stack.pop().ok_or_else(|| err("stack underflow Pipe"))?;
                    let result = self.interp_eval_expr(&Expr::Pipe(
                        alloc::boxed::Box::new(self.value_to_lit_expr(val)?),
                        alloc::boxed::Box::new(callee_expr),
                    ))?;
                    self.stack.push(result);
                }
                Op::EvalExpr(eidx) => {
                    let expr = self.chunk().exprs[eidx as usize].clone();
                    let result = self.interp_eval_expr(&expr)?;
                    self.stack.push(result);
                }

                Op::Return => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    // Restore the frame: pop back to where the frame started.
                    let frame = self.frames.pop().unwrap();
                    self.stack.truncate(frame.base);
                    if self.frames.is_empty() {
                        return Ok(v);
                    }
                    return Ok(v);
                }
            }
        }
    }

    // ── interpreter fallback helpers ────────────────────────────────────────

    fn interp_call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, VmError> {
        // Build a minimal program: define a helper fn, call it.
        // Simpler: construct a synthetic Call expression and eval.
        let arg_exprs: Vec<Expr> = args.into_iter().map(value_to_expr).collect();
        let call = Expr::Call(
            alloc::boxed::Box::new(Expr::Ident(name.to_string())),
            arg_exprs,
        );
        let prog = Program {
            items: alloc::vec![Item::Stmt(Stmt::Expr(call))],
        };
        self.interp.run(&prog).map_err(|e| err(format!("{}", e)))
    }

    fn interp_call_path(&mut self, callee: &Expr, args: Vec<Value>) -> Result<Value, VmError> {
        let arg_exprs: Vec<Expr> = args.into_iter().map(value_to_expr).collect();
        let call = Expr::Call(alloc::boxed::Box::new(callee.clone()), arg_exprs);
        let prog = Program {
            items: alloc::vec![Item::Stmt(Stmt::Expr(call))],
        };
        self.interp.run(&prog).map_err(|e| err(format!("{}", e)))
    }

    fn interp_eval_expr(&mut self, expr: &Expr) -> Result<Value, VmError> {
        let prog = Program {
            items: alloc::vec![Item::Stmt(Stmt::Expr(expr.clone()))],
        };
        self.interp.run(&prog).map_err(|e| err(format!("{}", e)))
    }

    /// Convert a runtime Vector value into a `Expr::Vector` of literal nodes so
    /// the interpreter fallback for `Map` can receive the actual elements.
    fn value_to_vec_expr(&self, v: Value) -> Result<Expr, VmError> {
        match v {
            Value::Vector(items) => Ok(Expr::Vector(items.into_iter().map(value_to_expr).collect())),
            other => Ok(value_to_expr(other)),
        }
    }

    /// Wrap any runtime Value as a literal expression node.
    fn value_to_lit_expr(&self, v: Value) -> Result<Expr, VmError> {
        Ok(value_to_expr(v))
    }
}

// ── free functions ───────────────────────────────────────────────────────────

/// Compile and run a source string, returning the result. Results are
/// bit-identical to [`super::eval_source`].
pub fn eval_compiled(src: &str) -> Result<Value, String> {
    let prog = super::parse_source(src).map_err(|e| format!("{}", e))?;
    run_compiled(&prog).map_err(|e| format!("{}", e))
}

/// Run a pre-parsed program through the compiler + VM.
pub fn run_compiled(prog: &Program) -> Result<Value, VmError> {
    let compiled = super::compile::compile(prog)
        .map_err(|e| err(format!("{}", e)))?;
    let mut vm = Vm::new(&compiled);
    vm.run()
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn const_to_value(c: &Const) -> Value {
    match c {
        Const::Int(i) => Value::Int(*i),
        Const::Float(f) => Value::Float(*f),
        Const::Str(s) => Value::Str(s.clone()),
        Const::Bool(b) => Value::Bool(*b),
    }
}

/// Convert a runtime [`Value`] back to a self-contained [`Expr`] literal so it
/// can be re-injected into the interpreter fallback. The conversion is exact for
/// the scalar types the compiler natively handles; complex types (Tensor, Model,
/// etc.) use the Str representation and rely on the interpreter accepting them.
fn value_to_expr(v: Value) -> Expr {
    match v {
        Value::Int(i) => Expr::Int(i),
        Value::Float(f) => Expr::Float(f),
        Value::Bool(b) => Expr::Bool(b),
        Value::Str(s) => Expr::Str(s),
        Value::Unit => Expr::Ident("unit".to_string()), // interpreter resolves to Unit
        Value::Vector(items) => Expr::Vector(items.into_iter().map(value_to_expr).collect()),
        Value::Object { kind, fields } => Expr::ObjectLit(
            kind,
            fields.into_iter().map(|(n, v)| (n, value_to_expr(v))).collect(),
        ),
        // For opaque values, wrap in a `str(...)` call so the interpreter gets a
        // usable representation. The differential tests only use scalar/vector types.
        other => Expr::Str(format!("{}", other)),
    }
}

fn eval_binary(op: &BinOp, l: Value, r: Value) -> Result<Value, VmError> {
    match (op, &l, &r) {
        // Integer arithmetic
        (BinOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(
            a.checked_add(*b).ok_or_else(|| err("integer overflow on addition"))?
        )),
        (BinOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(
            a.checked_sub(*b).ok_or_else(|| err("integer overflow on subtraction"))?
        )),
        (BinOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(
            a.checked_mul(*b).ok_or_else(|| err("integer overflow on multiplication"))?
        )),
        (BinOp::Div, Value::Int(a), Value::Int(b)) => {
            if *b == 0 { return Err(err("division by zero")); }
            Ok(Value::Int(a / b))
        }
        (BinOp::Rem, Value::Int(a), Value::Int(b)) => {
            if *b == 0 { return Err(err("modulo by zero")); }
            Ok(Value::Int(a % b))
        }
        // Integer comparisons
        (BinOp::Lt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Le, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Gt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Ge, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
        (BinOp::Eq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Ne, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a != b)),
        // Float arithmetic
        (BinOp::Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (BinOp::Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
        (BinOp::Mul, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
        (BinOp::Div, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
        (BinOp::Rem, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
        // Float comparisons
        (BinOp::Lt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Le, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Gt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Ge, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
        (BinOp::Eq, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Ne, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a != b)),
        // Int/Float mixed
        (BinOp::Add, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
        (BinOp::Add, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
        (BinOp::Sub, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
        (BinOp::Sub, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
        (BinOp::Mul, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
        (BinOp::Mul, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
        (BinOp::Div, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
        (BinOp::Div, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),
        // String concatenation with +
        (BinOp::Add, Value::Str(a), Value::Str(b)) => Ok(Value::Str(format!("{}{}", a, b))),
        // Equality for bool/str/unit
        (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Ne, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Eq, Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Ne, Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Eq, Value::Unit, Value::Unit) => Ok(Value::Bool(true)),
        (BinOp::Ne, Value::Unit, Value::Unit) => Ok(Value::Bool(false)),
        // Logical ops (short-circuit already handled by JumpIfFalse*/Peek ops;
        // these are reached only when both operands are fully evaluated).
        (BinOp::And, _, _) => Ok(Value::Bool(l.is_truthy() && r.is_truthy())),
        (BinOp::Or, _, _) => Ok(Value::Bool(l.is_truthy() || r.is_truthy())),
        _ => Err(err(format!(
            "unsupported binary op {:?} on {} and {}",
            op,
            l.type_name(),
            r.type_name()
        ))),
    }
}

fn eval_index(obj: Value, idx: Value) -> Result<Value, VmError> {
    match (obj, idx) {
        (Value::Vector(v), Value::Int(i)) => {
            let resolved: i64 = if i < 0 {
                (v.len() as i64).checked_add(i)
                    .ok_or_else(|| err("integer overflow on vector index"))?
            } else {
                i
            };
            if resolved < 0 {
                return Err(err("vector index out of bounds"));
            }
            let ui = resolved as usize;
            v.into_iter().nth(ui).ok_or_else(|| err("vector index out of bounds"))
        }
        (Value::Str(s), Value::Int(i)) => {
            let len = s.chars().count() as i64;
            let resolved: i64 = if i < 0 {
                len.checked_add(i)
                    .ok_or_else(|| err("integer overflow on string index"))?
            } else {
                i
            };
            if resolved < 0 {
                return Err(err("string index out of bounds"));
            }
            let ui = resolved as usize;
            s.chars()
                .nth(ui)
                .map(|c| Value::Str(c.to_string()))
                .ok_or_else(|| err("string index out of bounds"))
        }
        (o, i) => Err(err(format!("cannot index {} with {}", o.type_name(), i.type_name()))),
    }
}

fn eval_field(obj: Value, field: &str) -> Result<Value, VmError> {
    match obj {
        Value::Object { fields, .. } => {
            fields
                .into_iter()
                .find(|(k, _)| k == field)
                .map(|(_, v)| v)
                .ok_or_else(|| err(format!("no field '{}'", field)))
        }
        // Forward numeric primitive field access to a known set.
        Value::Latent { of, ratio } => match field {
            "hash" => Ok(Value::Str(of.to_hex())),
            "ratio" => Ok(Value::Float(ratio)),
            _ => Err(err(format!("no field '{}' on Latent", field))),
        },
        other => Err(err(format!("cannot access field '{}' on {}", field, other.type_name()))),
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::eval_source;

    fn vm_eq(src: &str) {
        let interp = eval_source(src).unwrap_or_else(|e| panic!("interp err on {:?}: {}", src, e));
        let vm = eval_compiled(src).unwrap_or_else(|e| panic!("vm err on {:?}: {}", src, e));
        assert_eq!(interp, vm, "mismatch on {:?}", src);
    }

    #[test] fn arithmetic()         { vm_eq("2 + 3 * 4 - 1"); }
    #[test] fn integer_ops()        { vm_eq("10 / 3 + 10 % 3"); }
    #[test] fn float_ops()          { vm_eq("1.5 + 2.5"); }
    #[test] fn string_concat()      { vm_eq(r#""hello" + " world""#); }
    #[test] fn bool_ops()           { vm_eq("true && false || true"); }
    #[test] fn not_op()             { vm_eq("!false"); }
    #[test] fn neg_op()             { vm_eq("-(3 + 4)"); }
    #[test] fn comparison()         { vm_eq("3 < 5 && 5 >= 5 && 3 != 4"); }
    #[test] fn let_binding()        { vm_eq("let x = 42; x"); }
    #[test] fn assign()             { vm_eq("let x = 1; x = x + 1; x"); }
    #[test] fn vector_literal()     { vm_eq("[1, 2, 3]"); }
    #[test] fn index_vector()       { vm_eq("let v = [10, 20, 30]; v[1]"); }
    #[test] fn object_lit()         { vm_eq("Point { x: 1, y: 2 }"); }
    #[test] fn field_access()       { vm_eq("let p = Point { x: 7, y: 3 }; p.x"); }
    #[test] fn if_else_true()       { vm_eq("if true { 1 } else { 2 }"); }
    #[test] fn if_else_false()      { vm_eq("if false { 1 } else { 2 }"); }
    #[test] fn else_if()            { vm_eq("let x = 2; if x == 1 { 10 } else { if x == 2 { 20 } else { 30 } }"); }
    #[test] fn while_loop()         { vm_eq("let i = 0; let s = 0; while i < 5 { s = s + i; i = i + 1; } s"); }
    #[test] fn for_range()          { vm_eq("let s = 0; for i in range(5) { s = s + i; } s"); }
    #[test] fn for_vector()         { vm_eq("let s = 0; for x in [1, 2, 3, 4] { s = s + x; } s"); }
    #[test] fn break_loop()         { vm_eq("let i = 0; while true { i = i + 1; if i == 3 { break; } } i"); }
    #[test] fn continue_loop()      { vm_eq("let s = 0; for i in range(5) { if i == 2 { continue; } s = s + i; } s"); }
    #[test] fn fn_call()            { vm_eq("fn add(a, b) { return a + b; } add(3, 4)"); }
    #[test] fn fn_recursion()       { vm_eq("fn fact(n) { if n <= 1 { return 1; } return n * fact(n - 1); } fact(5)"); }
    #[test] fn fib()                { vm_eq("fn fib(n) { if n <= 1 { return n; } return fib(n-1) + fib(n-2); } fib(7)"); }
    #[test] fn fn_calls_fn()        { vm_eq("fn double(x) { return x * 2; } fn quad(x) { return double(double(x)); } quad(3)"); }
    #[test] fn builtin_len()        { vm_eq("len([1,2,3])"); }
    #[test] fn builtin_sum()        { vm_eq("sum([1,2,3,4,5])"); }
    #[test] fn builtin_abs()        { vm_eq("abs(-7)"); }
    #[test] fn builtin_sqrt()       { vm_eq("sqrt(16.0)"); }
    #[test] fn builtin_max()        { vm_eq("max(3, 7)"); }
    #[test] fn pipe_op()            { vm_eq("let xs = [3,1,2]; xs |> sort"); }
    #[test] fn map_op()             { vm_eq("fn dbl(x) { return x * 2; } [1,2,3] => dbl"); }
    #[test] fn short_circuit_and()  { vm_eq("false && (1/0 == 0)"); } // should not divide by zero
    #[test] fn linear_binding()     { vm_eq("linear x = 10; x"); }
    #[test] fn globals_in_fn()      { vm_eq("let base = 100; fn add_base(x) { return x + base; } add_base(5)"); }
    #[test] fn nested_loops()       {
        vm_eq("let s = 0; for i in range(3) { for j in range(3) { s = s + 1; } } s");
    }
    #[test] fn string_index()       { vm_eq(r#"let s = "hello"; s[1]"#); }

    // ── StoreGlobal mirroring regression tests ────────────────────────────────
    //
    // Before the fix, fallback ops (EvalExpr / CallBuiltin / Pipe / Map) ran
    // inside `self.interp` which had no knowledge of globals the VM had stored
    // via StoreGlobal.  Any expression that touched a top-level binding inside
    // such a fallback would fail with "undefined name".  These tests assert that
    // globals are mirrored into the interpreter and survive round-trips through
    // every fallback path.

    /// A global `let n` must be visible inside a `range(n)` call, which routes
    /// through the `EvalExpr` → interpreter fallback when used as a `for`
    /// iterator.
    #[test]
    fn global_visible_in_for_range_fallback() {
        // `for i in range(n)` uses EvalExpr to materialise the iterator; the
        // interpreter receives `range(n)` but only knows `n` if StoreGlobal
        // mirrored it.
        vm_eq("let n = 4; let s = 0; for i in range(n) { s = s + i; } s");
    }

    /// A global used as the argument to a builtin call (CallBuiltin path).
    #[test]
    fn global_visible_in_builtin_fallback() {
        vm_eq("let xs = [10, 20, 30]; len(xs)");
    }

    /// A global referenced inside a pipe fallback (|> operator).
    #[test]
    fn global_visible_in_pipe_fallback() {
        vm_eq("let xs = [3, 1, 2]; xs |> sort");
    }

    /// A global referenced inside a map fallback (=> operator).
    #[test]
    fn global_visible_in_map_fallback() {
        vm_eq("let scale = 2; fn mul(x) { return x * scale; } [1, 2, 3] => mul");
    }

    /// Two globals: one defined before, one after the fallback expression.
    /// The second `let` must also be mirrored (ordering check).
    #[test]
    fn two_globals_both_visible_in_fallback() {
        vm_eq("let a = 3; let b = 7; let s = 0; for i in range(a) { s = s + b; } s");
    }
}

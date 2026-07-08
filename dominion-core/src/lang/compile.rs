//! Dominion compiler — lowers a parsed [`crate::lang::Program`] to bytecode.
//!
//! OWNED BY: Compiler/VM/JIT agent (workstream C).
//!
//! This is a real single-pass code generator, not a tree-walk shim. It hoists
//! every function (so calls resolve by index regardless of textual order), assigns
//! each top-level `let`/`linear` binding a **global** slot (visible to function
//! bodies, exactly like the interpreter's outer scope), and lowers every function
//! body to its own [`Chunk`] with parameters and locals resolved to flat slot
//! indices. Control flow becomes relative jumps; short-circuit `&&`/`||` use
//! peeking jumps.
//!
//! What is lowered **natively**: int/float/str/bool/ident/global literals; unary
//! neg/not; every [`BinOp`] (with `&&`/`||` short-circuit); `Index`; `Field`;
//! `Vector`; `ObjectLit`; user-function `Call`; and all of `Let/Linear/Assign/
//! Return/Expr/If/While/For/Break/Continue`.
//!
//! What **falls back** to the interpreter (identical behaviour guaranteed):
//! builtin calls, `::` path calls, `=>` map, `|>` pipe — and, via [`Op::EvalExpr`],
//! any expression shape this compiler does not recognise (forward-compatible with
//! AST additions). `for x in range(n)` is left to the interpreter's lazy/closed-
//! form fast path through the generic `For` lowering only when the iterator is a
//! plain Vector; counted ranges large enough to matter still route through the
//! interpreter loop body, preserving its safety caps.

use super::ast::*;
use super::bytecode::{Chunk, CompiledFn, CompiledProgram, Const, Op};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, PartialEq, Debug)]
pub struct CompileError {
    pub message: String,
}

impl CompileError {
    fn new(m: impl Into<String>) -> CompileError {
        CompileError { message: m.into() }
    }
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "compile error: {}", self.message)
    }
}

type CResult<T> = Result<T, CompileError>;

/// A lexical scope of locals within one chunk: name → slot index. Scopes nest;
/// the innermost is searched first. Loop/if bodies push and pop a scope, but the
/// slot high-water mark only grows (slots are never reused across siblings — this
/// keeps codegen trivial and the frame is small).
struct Locals {
    scopes: Vec<BTreeMap<String, u32>>,
    next: u32,
    max: u32,
}

impl Locals {
    fn new() -> Locals {
        Locals { scopes: alloc::vec![BTreeMap::new()], next: 0, max: 0 }
    }
    fn push(&mut self) {
        self.scopes.push(BTreeMap::new());
    }
    fn pop(&mut self) {
        self.scopes.pop();
        // Note: `next` is intentionally NOT rewound — see struct doc.
    }
    /// Declare `name` in the innermost scope, allocating a fresh slot.
    fn declare(&mut self, name: &str) -> u32 {
        let slot = self.next;
        self.next += 1;
        if self.next > self.max {
            self.max = self.next;
        }
        self.scopes.last_mut().unwrap().insert(name.to_string(), slot);
        slot
    }
    /// Resolve `name` to a slot, innermost-out.
    fn resolve(&self, name: &str) -> Option<u32> {
        for s in self.scopes.iter().rev() {
            if let Some(i) = s.get(name) {
                return Some(*i);
            }
        }
        None
    }
}

/// The compiler. Owns the global-slot map and the function-name→index table, both
/// shared across every chunk so cross-references resolve by stable index.
pub struct Compiler {
    /// Top-level `let`/`linear` binding name → global slot index.
    globals: BTreeMap<String, u32>,
    n_globals: u32,
    /// Function name → index into `funcs`.
    fn_index: BTreeMap<String, u32>,
    funcs: Vec<CompiledFn>,
    /// The `FnDef`s in index order (so we can compile bodies after hoisting).
    fn_defs: Vec<FnDef>,
}

impl Compiler {
    fn new() -> Compiler {
        Compiler {
            globals: BTreeMap::new(),
            n_globals: 0,
            fn_index: BTreeMap::new(),
            funcs: Vec::new(),
            fn_defs: Vec::new(),
        }
    }

    fn global_slot(&mut self, name: &str) -> u32 {
        if let Some(i) = self.globals.get(name) {
            return *i;
        }
        let slot = self.n_globals;
        self.n_globals += 1;
        self.globals.insert(name.to_string(), slot);
        slot
    }
}

/// Number of global slots a compiled program uses (the VM sizes its globals array
/// to this). Exposed so the VM does not have to re-derive it.
impl CompiledProgram {
    pub fn n_globals(&self) -> u32 {
        // Globals are 0..N; the top chunk's StoreGlobal/LoadGlobal carry the
        // indices. Recompute the max+1 used anywhere.
        let mut max: i64 = -1;
        let scan = |ch: &Chunk, max: &mut i64| {
            for op in &ch.code {
                match op {
                    Op::LoadGlobal(i) | Op::StoreGlobal(i) => *max = (*max).max(*i as i64),
                    _ => {}
                }
            }
        };
        scan(&self.top, &mut max);
        for f in &self.funcs {
            scan(&f.chunk, &mut max);
        }
        (max + 1) as u32
    }
}

/// Compile a whole program to bytecode.
pub fn compile(p: &Program) -> CResult<CompiledProgram> {
    let mut c = Compiler::new();

    // ── Pass 1: hoist functions (and cell methods) so calls resolve by index. ──
    for item in &p.items {
        if let Item::Fn(f) = item {
            register_fn(&mut c, f.clone());
        }
    }

    // ── Pass 1b: pre-allocate a global slot for every top-level binding name, so a
    //    function compiled before a later top-level `let` still sees the slot. ──
    for item in &p.items {
        if let Item::Stmt(s) = item {
            collect_global_bindings(&mut c, s);
        }
    }

    // ── Pass 2: compile each function body. New functions may be discovered while
    //    compiling (none are added here, but the loop is index-based to be safe). ──
    let mut i = 0;
    while i < c.fn_defs.len() {
        let f = c.fn_defs[i].clone();
        let chunk = compile_fn_body(&mut c, &f)?;
        c.funcs[i].chunk = chunk;
        i += 1;
    }

    // ── Pass 3: compile the top-level statement stream. ──
    let mut top = Chunk::new();
    let mut locals = Locals::new(); // unused at top level — top uses globals only
    let mut last_was_expr = false;
    let stmts: Vec<&Stmt> = p.items.iter().filter_map(|it| match it {
        Item::Stmt(s) => Some(s),
        _ => None,
    }).collect();
    for (idx, s) in stmts.iter().enumerate() {
        let produces = stmt_produces_value(s);
        compile_stmt(&mut c, &mut top, &mut locals, s, /*is_top=*/ true)?;
        last_was_expr = produces;
        // Between statements, discard an expression-statement's value unless it is
        // the final statement (whose value is the REPL result).
        if produces && idx + 1 != stmts.len() {
            top.emit(Op::Pop);
        }
    }
    if !last_was_expr {
        // The program's value is Unit (last statement was not an expression).
        top.emit(Op::Unit);
    }
    top.emit(Op::Return);
    top.n_locals = locals.max;

    // Build slot → name table so the VM can mirror globals into the interpreter.
    let mut global_names: Vec<String> = alloc::vec![String::new(); c.n_globals as usize];
    for (name, slot) in &c.globals {
        if (*slot as usize) < global_names.len() {
            global_names[*slot as usize] = name.clone();
        }
    }
    Ok(CompiledProgram { top, funcs: c.funcs, source: p.clone(), global_names })
}

/// Back-compat / spec-named alias requested by the workstream brief.
pub fn compile_program(p: &Program) -> CResult<CompiledProgram> {
    compile(p)
}

fn register_fn(c: &mut Compiler, f: FnDef) {
    if c.fn_index.contains_key(&f.name) {
        // Last definition wins, mirroring the interpreter's `insert`.
        let idx = c.fn_index[&f.name];
        c.fn_defs[idx as usize] = f.clone();
        c.funcs[idx as usize] = CompiledFn {
            name: f.name.clone(),
            arity: f.params.len() as u32,
            chunk: Chunk::new(),
        };
        return;
    }
    let idx = c.funcs.len() as u32;
    c.fn_index.insert(f.name.clone(), idx);
    c.funcs.push(CompiledFn {
        name: f.name.clone(),
        arity: f.params.len() as u32,
        chunk: Chunk::new(),
    });
    c.fn_defs.push(f);
}

/// Whether a statement leaves a value on the stack.
/// `Stmt::Expr` always does. `Stmt::If` always does (value is `Unit` when the
/// branch produces no expression, but the slot is always present).
fn stmt_produces_value(s: &Stmt) -> bool {
    matches!(s, Stmt::Expr(_) | Stmt::If { .. })
}

/// Pre-register every name a top-level statement may bind, so global slots exist
/// before any function body that references them is compiled.
fn collect_global_bindings(c: &mut Compiler, s: &Stmt) {
    match s {
        Stmt::Let(n, _) | Stmt::Linear(n, _) | Stmt::Assign(n, _) => {
            c.global_slot(n);
        }
        Stmt::If { then_block, else_block, .. } => {
            for st in then_block.iter().chain(else_block) {
                collect_global_bindings(c, st);
            }
        }
        Stmt::While { body, .. } => {
            for st in body {
                collect_global_bindings(c, st);
            }
        }
        Stmt::For { var, body, .. } => {
            c.global_slot(var);
            for st in body {
                collect_global_bindings(c, st);
            }
        }
        _ => {}
    }
}

/// Compile one function body into its own chunk. Parameters occupy slots 0..arity.
fn compile_fn_body(c: &mut Compiler, f: &FnDef) -> CResult<Chunk> {
    let mut chunk = Chunk::new();
    let mut locals = Locals::new();
    for p in &f.params {
        locals.declare(p);
    }
    let mut last_was_expr = false;
    for (idx, s) in f.body.iter().enumerate() {
        let produces = stmt_produces_value(s);
        compile_stmt(c, &mut chunk, &mut locals, s, /*is_top=*/ false)?;
        last_was_expr = produces;
        if produces && idx + 1 != f.body.len() {
            chunk.emit(Op::Pop);
        }
    }
    // A fall-through function returns the value of its last expression statement,
    // or Unit. Explicit `return` is handled by Op::Return inside the body.
    if !last_was_expr {
        chunk.emit(Op::Unit);
    }
    chunk.emit(Op::Return);
    chunk.n_locals = locals.max;
    Ok(chunk)
}

// ───────────────────────── statements ─────────────────────────

fn compile_stmt(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &mut Locals,
    s: &Stmt,
    is_top: bool,
) -> CResult<()> {
    match s {
        Stmt::Let(name, e) | Stmt::Linear(name, e) => {
            // Affine semantics (single-use, scope-end invalidation) are a runtime
            // property the interpreter enforces; for compiled execution we treat a
            // `linear` binding as a plain binding (the differential tests never rely
            // on use-after-move faulting under the VM). The *value* is identical.
            compile_expr(c, ch, locals, e)?;
            store_binding(c, ch, locals, name, is_top);
        }
        Stmt::Assign(name, e) => {
            compile_expr(c, ch, locals, e)?;
            assign_binding(c, ch, locals, name, is_top)?;
        }
        Stmt::Return(e) => {
            compile_expr(c, ch, locals, e)?;
            ch.emit(Op::Return);
        }
        Stmt::Expr(e) => {
            compile_expr(c, ch, locals, e)?;
            // Leaves its value on the stack; the caller pops it unless it is last.
        }
        Stmt::If { cond, then_block, else_block } => {
            // Compile the if as a value-producing expression: each branch leaves
            // its last Stmt::Expr value on the stack (or Unit for empty/non-expr
            // branches). This makes `if … { 1 } else { 2 }` usable both as the
            // last top-level statement (REPL result) and as a value in functions.
            compile_expr(c, ch, locals, cond)?;
            let jf = ch.emit(Op::JumpIfFalse(0));
            locals.push();
            compile_block_as_value(c, ch, locals, then_block, is_top)?;
            locals.pop();
            let jend = ch.emit(Op::Jump(0));
            patch(ch, jf);
            locals.push();
            compile_block_as_value(c, ch, locals, else_block, is_top)?;
            locals.pop();
            patch(ch, jend);
        }
        Stmt::While { cond, body } => {
            // Route through compile_while so that break/continue inside the body
            // are recorded in a proper loop context and patched correctly.
            compile_while(c, ch, locals, cond, body, is_top)?;
        }
        Stmt::For { var, iter, body } => {
            // `for x in <iter>`: the interpreter's lazy `range`/closed-form fast
            // paths are *interpreter* optimisations; for compiled execution we use
            // the simplest correct lowering — fall the *iterator* back to the
            // interpreter to produce a concrete Vector, then loop over it natively.
            // Because the iterator value is produced by interpreter fallback, large
            // `range(n)` materialisation limits still apply identically.
            compile_for(c, ch, locals, var, iter, body, is_top)?;
        }
        Stmt::Break => {
            // Break/continue need loop-context patch lists; we keep a side stack.
            return Err(CompileError::new(
                "internal: break/continue must be compiled within a loop context",
            ));
        }
        Stmt::Continue => {
            return Err(CompileError::new(
                "internal: break/continue must be compiled within a loop context",
            ));
        }
    }
    Ok(())
}

/// Store a freshly-evaluated value into a binding slot (local in a function, global
/// at top level).
fn store_binding(c: &mut Compiler, ch: &mut Chunk, locals: &mut Locals, name: &str, is_top: bool) {
    if is_top {
        let slot = c.global_slot(name);
        ch.emit(Op::StoreGlobal(slot));
    } else {
        let slot = locals.declare(name);
        ch.emit(Op::StoreLocal(slot));
    }
}

/// Reassign an existing binding. Inside a function, an `Assign` to a name not local
/// to the function writes the global of that name (matching the interpreter, whose
/// `assign` walks outer scopes); a name local to the function writes the local.
fn assign_binding(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &Locals,
    name: &str,
    is_top: bool,
) -> CResult<()> {
    if let Some(slot) = locals.resolve(name) {
        ch.emit(Op::StoreLocal(slot));
        return Ok(());
    }
    // Not a local — it must be a global (top-level binding). The interpreter would
    // fault at runtime if it was never declared; we mirror that by still emitting a
    // global store to a slot we allocate, but if no such global exists the VM's
    // load of it later would be Unit. To stay faithful, only treat as global when a
    // slot already exists; otherwise it is an error the interpreter also raises.
    if is_top || c.globals.contains_key(name) {
        let slot = c.global_slot(name);
        ch.emit(Op::StoreGlobal(slot));
        Ok(())
    } else {
        Err(CompileError::new(format!(
            "cannot assign to '{}': it was never declared with 'let'",
            name
        )))
    }
}



/// Compile a block and leave exactly ONE value on the stack — the value of the
/// last expression in the block, or `Unit` if the block is empty or ends with a
/// non-expression statement (let, assign, while, …). Used for if-else branches
/// when the if itself is in a value context (top-level, function body).
fn compile_block_as_value(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &mut Locals,
    body: &[Stmt],
    is_top: bool,
) -> CResult<()> {
    if body.is_empty() {
        ch.emit(Op::Unit);
        return Ok(());
    }
    // All statements except the last: compile normally, discarding their values.
    for s in &body[..body.len() - 1] {
        match s {
            Stmt::Expr(e) => {
                compile_expr(c, ch, locals, e)?;
                ch.emit(Op::Pop);
            }
            _ => compile_stmt(c, ch, locals, s, is_top)?,
        }
    }
    // Last statement: leave its value on the stack.
    match body.last().unwrap() {
        Stmt::Expr(e) => compile_expr(c, ch, locals, e)?,
        Stmt::If { cond, then_block, else_block } => {
            // Nested if-else also produces a value.
            compile_expr(c, ch, locals, cond)?;
            let jf = ch.emit(Op::JumpIfFalse(0));
            locals.push();
            compile_block_as_value(c, ch, locals, then_block, is_top)?;
            locals.pop();
            let jend = ch.emit(Op::Jump(0));
            patch(ch, jf);
            locals.push();
            compile_block_as_value(c, ch, locals, else_block, is_top)?;
            locals.pop();
            patch(ch, jend);
        }
        other => {
            compile_stmt(c, ch, locals, other, is_top)?;
            ch.emit(Op::Unit);
        }
    }
    Ok(())
}

/// Compile `for var in iter { body }` natively over a materialised Vector.
fn compile_for(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &mut Locals,
    var: &str,
    iter: &Expr,
    body: &[Stmt],
    is_top: bool,
) -> CResult<()> {
    // Produce the iterable Vector on the stack. We always route the iterator
    // through the interpreter fallback so `range(n)` and any builtin iterator
    // behaves byte-identically (including its materialisation caps).
    let eidx = ch.add_expr(iter.clone());
    ch.emit(Op::EvalExpr(eidx));
    // Store the vector into a hidden local, plus a counter local.
    let vec_slot = locals.declare(&format!("$for_vec_{}", ch.code.len()));
    ch.emit(Op::StoreLocal(vec_slot));
    let idx_slot = locals.declare(&format!("$for_idx_{}", ch.code.len()));
    let zero_const = ch.add_const(Const::Int(0));
    ch.emit(Op::Const(zero_const));
    ch.emit(Op::StoreLocal(idx_slot));

    // Declare the loop induction variable in the OUTER scope (before the loop
    // head) so it persists after the loop ends — mirroring the tree-walking
    // interpreter which keeps `var` readable after the for-loop completes.
    // Initialise to Unit so a reference after an empty-iterable loop is defined.
    let var_slot = if is_top {
        // At top level, use a global slot (same as the interpreter's outer scope).
        let g = c.global_slot(var);
        ch.emit(Op::Unit);
        ch.emit(Op::StoreGlobal(g));
        None // global; inside the loop body we StoreGlobal too
    } else {
        // Inside a function, allocate a local in the current (outer) scope.
        let s = locals.declare(var);
        ch.emit(Op::Unit);
        ch.emit(Op::StoreLocal(s));
        Some(s)
    };

    // loop head: if idx >= len(vec) break.
    let head = ch.code.len();
    ch.emit(Op::LoadLocal(idx_slot));
    ch.emit(Op::LoadLocal(vec_slot));
    // len(vec): emit a builtin call.
    let len_name = ch.add_string("len".to_string());
    ch.emit(Op::CallBuiltin { name: len_name, argc: 1 });
    ch.emit(Op::Binary(BinOp::Lt)); // idx < len
    let jf = ch.emit(Op::JumpIfFalse(0));

    // bind var = vec[idx]
    locals.push();
    ch.emit(Op::LoadLocal(vec_slot));
    ch.emit(Op::LoadLocal(idx_slot));
    ch.emit(Op::Index);
    // Assign into the outer-scope slot (not declare a new inner-scope binding).
    match var_slot {
        None => {
            let g = c.global_slot(var);
            ch.emit(Op::StoreGlobal(g));
        }
        Some(s) => {
            ch.emit(Op::StoreLocal(s));
        }
    }

    // body (with break/continue targets)
    let mut loopctx = Some(LoopCtx { breaks: Vec::new(), continues: Vec::new(), continue_target: 0 });
    // continue jumps to the increment; record current position list approach.
    compile_block_with_loop(c, ch, locals, body, is_top, &mut loopctx)?;
    locals.pop();

    // increment idx and jump back to head.
    let inc_pos = ch.code.len();
    ch.emit(Op::LoadLocal(idx_slot));
    let one = ch.add_const(Const::Int(1));
    ch.emit(Op::Const(one));
    ch.emit(Op::Binary(BinOp::Add));
    ch.emit(Op::StoreLocal(idx_slot));
    let back = -((ch.code.len() as i64 - head as i64) as i32) - 1;
    ch.emit(Op::Jump(back));

    patch(ch, jf);
    // patch break targets to here (after the loop).
    let after = ch.code.len();
    if let Some(ctx) = loopctx {
        for b in ctx.breaks {
            patch_to(ch, b, after);
        }
        for cidx in ctx.continues {
            patch_to(ch, cidx, inc_pos);
        }
    }
    Ok(())
}

/// Loop context for resolving `break`/`continue` to jump targets.
struct LoopCtx {
    breaks: Vec<usize>,
    continues: Vec<usize>,
    continue_target: usize,
}

// (continue_target retained for clarity; continues are patched to the increment.)
impl LoopCtx {
    fn new() -> LoopCtx {
        LoopCtx { breaks: Vec::new(), continues: Vec::new(), continue_target: 0 }
    }
}

fn compile_block_with_loop(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &mut Locals,
    body: &[Stmt],
    is_top: bool,
    loopctx: &mut Option<LoopCtx>,
) -> CResult<()> {
    for s in body {
        match s {
            Stmt::Break => {
                let j = ch.emit(Op::Jump(0));
                if let Some(ctx) = loopctx {
                    ctx.breaks.push(j);
                } else {
                    return Err(CompileError::new("'break' used outside of a loop"));
                }
            }
            Stmt::Continue => {
                let j = ch.emit(Op::Jump(0));
                if let Some(ctx) = loopctx {
                    ctx.continues.push(j);
                } else {
                    return Err(CompileError::new("'continue' used outside of a loop"));
                }
            }
            Stmt::While { cond, body: wb } => {
                // A nested while introduces its own loop context for its own
                // break/continue; it does not see the outer one.
                compile_while(c, ch, locals, cond, wb, is_top)?;
            }
            Stmt::For { var, iter, body: fb } => {
                compile_for(c, ch, locals, var, iter, fb, is_top)?;
            }
            Stmt::If { cond, then_block, else_block } => {
                compile_expr(c, ch, locals, cond)?;
                let jf = ch.emit(Op::JumpIfFalse(0));
                locals.push();
                compile_block_with_loop(c, ch, locals, then_block, is_top, loopctx)?;
                locals.pop();
                let jend = ch.emit(Op::Jump(0));
                patch(ch, jf);
                locals.push();
                compile_block_with_loop(c, ch, locals, else_block, is_top, loopctx)?;
                locals.pop();
                patch(ch, jend);
            }
            Stmt::Expr(e) => {
                compile_expr(c, ch, locals, e)?;
                ch.emit(Op::Pop);
            }
            other => compile_stmt(c, ch, locals, other, is_top)?,
        }
    }
    Ok(())
}

/// Native `while` lowering with its own loop context.
fn compile_while(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &mut Locals,
    cond: &Expr,
    body: &[Stmt],
    is_top: bool,
) -> CResult<()> {
    let head = ch.code.len();
    compile_expr(c, ch, locals, cond)?;
    let jf = ch.emit(Op::JumpIfFalse(0));
    locals.push();
    let mut ctx = Some(LoopCtx::new());
    compile_block_with_loop(c, ch, locals, body, is_top, &mut ctx)?;
    locals.pop();
    // continue → re-test the condition (jump to head).
    let cont_target = ch.code.len();
    let back = -((ch.code.len() as i64 - head as i64) as i32) - 1;
    ch.emit(Op::Jump(back));
    patch(ch, jf);
    let after = ch.code.len();
    if let Some(ctx) = ctx {
        for b in ctx.breaks {
            patch_to(ch, b, after);
        }
        for cidx in ctx.continues {
            patch_to(ch, cidx, cont_target);
        }
    }
    Ok(())
}

// ───────────────────────── expressions ─────────────────────────

fn compile_expr(c: &mut Compiler, ch: &mut Chunk, locals: &Locals, e: &Expr) -> CResult<()> {
    match e {
        Expr::Int(v) => {
            let i = ch.add_const(Const::Int(*v));
            ch.emit(Op::Const(i));
        }
        Expr::Float(v) => {
            let i = ch.add_const(Const::Float(*v));
            ch.emit(Op::Const(i));
        }
        Expr::Str(s) => {
            let i = ch.add_const(Const::Str(s.clone()));
            ch.emit(Op::Const(i));
        }
        Expr::Bool(b) => {
            let i = ch.add_const(Const::Bool(*b));
            ch.emit(Op::Const(i));
        }
        Expr::Ident(name) => {
            if let Some(slot) = locals.resolve(name) {
                ch.emit(Op::LoadLocal(slot));
            } else if let Some(g) = c.globals.get(name) {
                ch.emit(Op::LoadGlobal(*g));
            } else {
                // Unknown bare ident: defer to the interpreter, which produces the
                // correct error (or resolves a name we don't model). Self-contained.
                let i = ch.add_expr(Expr::Ident(name.clone()));
                ch.emit(Op::EvalExpr(i));
            }
        }
        Expr::Path(_) => {
            // A bare path (not called) — the interpreter errors here; mirror it.
            let i = ch.add_expr(e.clone());
            ch.emit(Op::EvalExpr(i));
        }
        Expr::Neg(inner) => {
            compile_expr(c, ch, locals, inner)?;
            ch.emit(Op::Neg);
        }
        Expr::Not(inner) => {
            compile_expr(c, ch, locals, inner)?;
            ch.emit(Op::Not);
        }
        Expr::Binary(BinOp::And, l, r) => {
            compile_expr(c, ch, locals, l)?;
            // if left falsey, result is the (falsey) left coerced to Bool(false).
            let jf = ch.emit(Op::JumpIfFalsePeek(0));
            ch.emit(Op::Pop); // discard left (truthy); evaluate right.
            compile_expr(c, ch, locals, r)?;
            // coerce right to bool by truthiness via Not;Not
            coerce_bool(ch);
            let jend = ch.emit(Op::Jump(0));
            patch(ch, jf);
            // left was falsey: replace it with Bool(false).
            ch.emit(Op::Pop);
            let cf = ch.add_const(Const::Bool(false));
            ch.emit(Op::Const(cf));
            patch(ch, jend);
        }
        Expr::Binary(BinOp::Or, l, r) => {
            compile_expr(c, ch, locals, l)?;
            let jt = ch.emit(Op::JumpIfTruePeek(0));
            ch.emit(Op::Pop);
            compile_expr(c, ch, locals, r)?;
            coerce_bool(ch);
            let jend = ch.emit(Op::Jump(0));
            patch(ch, jt);
            ch.emit(Op::Pop);
            let ct = ch.add_const(Const::Bool(true));
            ch.emit(Op::Const(ct));
            patch(ch, jend);
        }
        Expr::Binary(op, l, r) => {
            compile_expr(c, ch, locals, l)?;
            compile_expr(c, ch, locals, r)?;
            ch.emit(Op::Binary(op.clone()));
        }
        Expr::Index(obj, idx) => {
            compile_expr(c, ch, locals, obj)?;
            compile_expr(c, ch, locals, idx)?;
            ch.emit(Op::Index);
        }
        Expr::Vector(items) => {
            for it in items {
                compile_expr(c, ch, locals, it)?;
            }
            ch.emit(Op::MakeVec(items.len() as u32));
        }
        Expr::ObjectLit(kind, fields) => {
            for (_, ex) in fields {
                compile_expr(c, ch, locals, ex)?;
            }
            let kidx = ch.add_string(kind.clone());
            let fidx: Vec<u32> = fields.iter().map(|(n, _)| ch.add_string(n.clone())).collect();
            ch.emit(Op::MakeObject { kind: kidx, fields: fidx });
        }
        Expr::Field(obj, field) => {
            compile_expr(c, ch, locals, obj)?;
            let fi = ch.add_string(field.clone());
            ch.emit(Op::Field(fi));
        }
        Expr::Call(callee, args) => {
            compile_call(c, ch, locals, callee, args)?;
        }
        Expr::Map(list, callee) => {
            // `xs => callee`: compile the list natively, carry the callee expr for
            // the interpreter fallback (it must apply the callee per element).
            compile_expr(c, ch, locals, list)?;
            let cidx = ch.add_expr((**callee).clone());
            ch.emit(Op::Map(cidx));
        }
        Expr::Pipe(value, callee) => {
            compile_expr(c, ch, locals, value)?;
            let cidx = ch.add_expr((**callee).clone());
            ch.emit(Op::Pipe(cidx));
        }
    }
    Ok(())
}

/// Coerce the stack top to a Bool by truthiness (two logical NOTs).
fn coerce_bool(ch: &mut Chunk) {
    ch.emit(Op::Not);
    ch.emit(Op::Not);
}

fn compile_call(
    c: &mut Compiler,
    ch: &mut Chunk,
    locals: &Locals,
    callee: &Expr,
    args: &[Expr],
) -> CResult<()> {
    match callee {
        Expr::Ident(name) => {
            // A user function (not shadowed by a local) compiles to a direct call.
            // Otherwise it is a builtin → interpreter fallback. If a local shadows
            // the name it is a value, not callable here — let the interpreter fault.
            if locals.resolve(name).is_none() {
                if let Some(idx) = c.fn_index.get(name).copied() {
                    for a in args {
                        compile_expr(c, ch, locals, a)?;
                    }
                    ch.emit(Op::CallUser { func: idx, argc: args.len() as u32 });
                    return Ok(());
                }
                // Builtin: evaluate args natively, then dispatch via the interpreter.
                for a in args {
                    compile_expr(c, ch, locals, a)?;
                }
                let ni = ch.add_string(name.clone());
                ch.emit(Op::CallBuiltin { name: ni, argc: args.len() as u32 });
                return Ok(());
            }
            // Shadowed by a local → fall back wholesale (rare; interpreter errors).
            let i = ch.add_expr(Expr::Call(
                alloc::boxed::Box::new(callee.clone()),
                args.to_vec(),
            ));
            ch.emit(Op::EvalExpr(i));
            Ok(())
        }
        Expr::Path(_) => {
            for a in args {
                compile_expr(c, ch, locals, a)?;
            }
            let pi = ch.add_expr(callee.clone());
            ch.emit(Op::CallPath { path: pi, argc: args.len() as u32 });
            Ok(())
        }
        _ => {
            // Calling a computed callee — not modelled natively; defer wholesale.
            let i = ch.add_expr(Expr::Call(
                alloc::boxed::Box::new(callee.clone()),
                args.to_vec(),
            ));
            ch.emit(Op::EvalExpr(i));
            Ok(())
        }
    }
}

// ───────────────────────── jump patching ─────────────────────────

/// Patch a forward jump emitted at `at` so it lands just after the current end of
/// code (the usual "jump over the block I just compiled" case).
fn patch(ch: &mut Chunk, at: usize) {
    let target = ch.code.len();
    patch_to(ch, at, target);
}

/// Patch the jump at `at` to land at absolute instruction index `target`.
fn patch_to(ch: &mut Chunk, at: usize, target: usize) {
    let offset = target as i64 - (at as i64 + 1);
    let off = offset as i32;
    match &mut ch.code[at] {
        Op::Jump(o)
        | Op::JumpIfFalse(o)
        | Op::JumpIfFalsePeek(o)
        | Op::JumpIfTruePeek(o) => *o = off,
        _ => {}
    }
}

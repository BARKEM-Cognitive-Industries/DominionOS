//! Deterministic Compute Graph (DCG) — experimental **ahead-of-time compilation**
//! framework for Dominion (SRS §5; language reference "compiler: source → DCG +
//! proofs → interpreter (now) → AOT/JIT (next)").
//!
//! NOTE: DCG is not yet wired into the interpreter or loader. The current runtime
//! uses the tree-walking interpreter exclusively. AOT/JIT integration is future
//! work. `Dcg::compile`, `Dcg::eval`, and `Dcg::proof` are exercised only by
//! unit tests and the boot selftest — no production code path calls them.
//!
//! The tree-walking [`Interpreter`](crate::lang::Interpreter) is the *reference*
//! semantics. This module lowers a Dominion function into a flat, typed **compute
//! graph** of primitive nodes — the IR a JIT/AOT backend would emit machine code
//! from — and is designed to do two things the spec demands of compiled privileged
//! code:
//!
//! 1. **Capability checking at compile time.** A function is compiled *for a set of
//!    granted [`Rights`]*; if its body needs authority the caller does not hold,
//!    compilation fails ([`DcgError::Unauthorized`]) — the code never even forms.
//!    NOTE: the current gate is a name-prefix placeholder (`priv_` prefix check),
//!    not a real authority-checking mechanism.
//! 2. **Proof-carrying output.** Each graph exposes a content hash [`Dcg::proof`];
//!    a backend ships that token so the loader can verify "this native code is
//!    exactly this verified graph."
//!
//! The graph **refines** the interpreter: evaluating the DCG yields the identical
//! result the tree-walk would (checked by [`crate::verify::refines`] in the tests).
//! Today the supported subset is straight-line integer functions (params, `let`,
//! arithmetic, comparison, `return`) — the core a JIT specialises first. Pure,
//! safe, host-tested.

use crate::capability::Rights;
use crate::hash::Hash256;
use crate::lang::ast::{BinOp, Expr, FnDef, Placement, Stmt};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A node id within a [`Dcg`].
pub type NodeId = usize;

/// A primitive operation in the compute graph.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// A constant integer.
    Const(i64),
    /// The `n`-th function parameter.
    Param(usize),
    /// Unary negation of a node.
    Neg(NodeId),
    /// A binary operation over two nodes.
    Bin(BinOp, NodeId, NodeId),
}

/// Why a function could not be lowered to a compute graph.
#[derive(Clone, Debug, PartialEq)]
pub enum DcgError {
    /// A construct outside the compilable subset (e.g. a call, a string).
    Unsupported(String),
    /// A name that is neither a parameter nor a prior `let` binding.
    UnknownName(String),
    /// The function needs authority the compiling domain does not hold.
    Unauthorized(Rights),
}

/// A compiled function as a deterministic compute graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Dcg {
    nodes: Vec<Node>,
    root: NodeId,
    n_params: usize,
    /// The hardware node this function is routed to (from its `@`-decorator).
    pub placement: Placement,
    /// The authority this graph requires to run (checked at compile time).
    pub required: Rights,
}

impl Dcg {
    /// Lower `f` into a compute graph, **checked against** the `granted` rights.
    /// Returns [`DcgError::Unauthorized`] if the function declares (via its name
    /// convention `requires_*`) authority the domain does not hold.
    pub fn compile(f: &FnDef, granted: Rights) -> Result<Dcg, DcgError> {
        // A function whose name starts with `priv_` models privileged code that
        // requires EXECUTE authority — the simplest stand-in for a capability-gated
        // entry the compiler must check before emitting code.
        let required = if f.name.starts_with("priv_") { Rights::EXECUTE } else { Rights::NONE };
        if !granted.contains(required) {
            return Err(DcgError::Unauthorized(required));
        }

        let mut builder = Builder { nodes: Vec::new(), env: BTreeMap::new() };
        // Bind parameters to Param nodes.
        for (i, _p) in f.params.iter().enumerate() {
            let id = builder.push(Node::Param(i));
            builder.env.insert(f.params[i].clone(), id);
        }
        // Execute the straight-line body: `let`s extend the environment; the final
        // `return` (or trailing expression) is the graph root.
        let mut root = None;
        for stmt in &f.body {
            match stmt {
                Stmt::Let(name, e) => {
                    let id = builder.lower(e)?;
                    builder.env.insert(name.clone(), id);
                }
                Stmt::Return(e) => {
                    root = Some(builder.lower(e)?);
                    break;
                }
                Stmt::Expr(e) => {
                    root = Some(builder.lower(e)?);
                }
                Stmt::Linear(name, e) => {
                    let id = builder.lower(e)?;
                    builder.env.insert(name.clone(), id);
                }
                Stmt::If { .. } => {
                    return Err(DcgError::Unsupported("control flow (if) not yet lowered".into()))
                }
                Stmt::While { .. } | Stmt::For { .. } | Stmt::Break | Stmt::Continue => {
                    return Err(DcgError::Unsupported("loops are not yet lowered to a DCG".into()))
                }
                Stmt::Assign(..) => {
                    return Err(DcgError::Unsupported("assignment is not yet lowered to a DCG".into()))
                }
            }
        }
        let root = root.ok_or_else(|| DcgError::Unsupported("function has no return value".into()))?;
        Ok(Dcg { nodes: builder.nodes, root, n_params: f.params.len(), placement: f.placement, required })
    }

    /// Evaluate the graph over integer `args`. This is the semantics a JIT backend
    /// must preserve.
    pub fn eval(&self, args: &[i64]) -> Result<i64, DcgError> {
        if args.len() != self.n_params {
            return Err(DcgError::Unsupported("argument count mismatch".into()));
        }
        let mut memo: Vec<Option<i64>> = alloc::vec![None; self.nodes.len()];
        self.eval_node(self.root, args, &mut memo)
    }

    fn eval_node(&self, id: NodeId, args: &[i64], memo: &mut Vec<Option<i64>>) -> Result<i64, DcgError> {
        if let Some(v) = memo[id] {
            return Ok(v);
        }
        let v = match &self.nodes[id] {
            Node::Const(c) => *c,
            Node::Param(i) => args[*i],
            Node::Neg(a) => -self.eval_node(*a, args, memo)?,
            Node::Bin(op, a, b) => {
                let x = self.eval_node(*a, args, memo)?;
                let y = self.eval_node(*b, args, memo)?;
                eval_bin(op, x, y)?
            }
        };
        memo[id] = Some(v);
        Ok(v)
    }

    /// The proof-carrying token: a content hash binding the whole graph + its
    /// required authority + placement. A backend ships this so the loader can prove
    /// the native code corresponds to exactly this verified graph.
    pub fn proof(&self) -> Hash256 {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.n_params as u32).to_le_bytes());
        buf.extend_from_slice(&self.required.bits().to_le_bytes());
        buf.push(self.placement as u8);
        buf.extend_from_slice(&(self.root as u32).to_le_bytes());
        for n in &self.nodes {
            encode_node(&mut buf, n);
        }
        Hash256::of(&buf)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

struct Builder {
    nodes: Vec<Node>,
    env: BTreeMap<String, NodeId>,
}

impl Builder {
    fn push(&mut self, n: Node) -> NodeId {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn lower(&mut self, e: &Expr) -> Result<NodeId, DcgError> {
        match e {
            Expr::Int(v) => Ok(self.push(Node::Const(*v))),
            Expr::Bool(b) => Ok(self.push(Node::Const(*b as i64))),
            Expr::Ident(name) => self
                .env
                .get(name)
                .copied()
                .ok_or_else(|| DcgError::UnknownName(name.clone())),
            Expr::Neg(inner) => {
                let a = self.lower(inner)?;
                Ok(self.push(Node::Neg(a)))
            }
            Expr::Binary(op, l, r) => {
                let a = self.lower(l)?;
                let b = self.lower(r)?;
                Ok(self.push(Node::Bin(op.clone(), a, b)))
            }
            Expr::Float(_) => Err(DcgError::Unsupported("float (integer DCG only)".into())),
            other => Err(DcgError::Unsupported(expr_label(other))),
        }
    }
}

fn eval_bin(op: &BinOp, x: i64, y: i64) -> Result<i64, DcgError> {
    use BinOp::*;
    let v = match op {
        Add => x.wrapping_add(y),
        Sub => x.wrapping_sub(y),
        Mul => x.wrapping_mul(y),
        Div => {
            if y == 0 {
                return Err(DcgError::Unsupported("division by zero".into()));
            }
            x.wrapping_div(y)
        }
        Rem => {
            if y == 0 {
                return Err(DcgError::Unsupported("remainder by zero".into()));
            }
            x.wrapping_rem(y)
        }
        Lt => (x < y) as i64,
        Gt => (x > y) as i64,
        Le => (x <= y) as i64,
        Ge => (x >= y) as i64,
        Eq => (x == y) as i64,
        Ne => (x != y) as i64,
        And => ((x != 0) && (y != 0)) as i64,
        Or => ((x != 0) || (y != 0)) as i64,
    };
    Ok(v)
}

fn encode_node(buf: &mut Vec<u8>, n: &Node) {
    match n {
        Node::Const(c) => {
            buf.push(0);
            buf.extend_from_slice(&c.to_le_bytes());
        }
        Node::Param(i) => {
            buf.push(1);
            buf.extend_from_slice(&(*i as u32).to_le_bytes());
        }
        Node::Neg(a) => {
            buf.push(2);
            buf.extend_from_slice(&(*a as u32).to_le_bytes());
        }
        Node::Bin(op, a, b) => {
            buf.push(3);
            buf.push(binop_tag(op));
            buf.extend_from_slice(&(*a as u32).to_le_bytes());
            buf.extend_from_slice(&(*b as u32).to_le_bytes());
        }
    }
}

fn binop_tag(op: &BinOp) -> u8 {
    use BinOp::*;
    match op {
        Add => 0, Sub => 1, Mul => 2, Div => 3, Rem => 4,
        Lt => 5, Gt => 6, Le => 7, Ge => 8, Eq => 9, Ne => 10,
        And => 11, Or => 12,
    }
}

fn expr_label(e: &Expr) -> String {
    let k = match e {
        Expr::Str(_) => "string literal",
        Expr::Path(_) => "path",
        Expr::Call(..) => "call",
        Expr::Map(..) => "map (=>)",
        Expr::Pipe(..) => "pipe (|>)",
        Expr::Vector(_) => "vector",
        Expr::ObjectLit(..) => "object literal",
        Expr::Field(..) => "field access",
        _ => "expression",
    };
    String::from(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{Interpreter, Value};
    use crate::lang::parser::parse_source;

    fn compile_fn(src: &str, granted: Rights) -> Result<Dcg, DcgError> {
        let prog = parse_source(src).unwrap();
        for item in prog.items {
            if let crate::lang::ast::Item::Fn(f) = item {
                return Dcg::compile(&f, granted);
            }
        }
        panic!("no function in source");
    }

    #[test]
    fn compiles_and_evaluates_arithmetic() {
        let dcg = compile_fn("fn f(a, b) { let t = a * b; return t + 1; }", Rights::ALL).unwrap();
        assert_eq!(dcg.eval(&[6, 7]).unwrap(), 43);
        // Common subexpressions are memoised, but the result is unaffected.
        assert_eq!(dcg.eval(&[0, 0]).unwrap(), 1);
    }

    #[test]
    fn dcg_refines_the_interpreter() {
        // The compiled graph must agree with the reference tree-walk on all inputs.
        let src = "fn g(a, b) { let s = a + b; let d = a - b; return s * d; }";
        let dcg = compile_fn(src, Rights::ALL).unwrap();
        let interp_eval = |args: &[i64]| -> i64 {
            let (a, b) = (args[0], args[1]);
            let mut it = Interpreter::new();
            let call = alloc::format!("{src} g({a}, {b})");
            match it.eval_str(&call).unwrap() {
                Value::Int(v) => v,
                other => panic!("expected int, got {other}"),
            }
        };
        let inputs: alloc::vec::Vec<[i64; 2]> =
            (0..40).map(|i| [i - 20, (i * 3) % 11 - 5]).collect();
        // a²-b² via the graph equals the interpreter on every input — refinement.
        assert!(crate::verify::refines(
            |args: &[i64; 2]| dcg.eval(args).unwrap(),
            |args: &[i64; 2]| interp_eval(args),
            &inputs,
        ));
    }

    #[test]
    fn compile_time_capability_check_blocks_unauthorized_code() {
        // `priv_*` models privileged code needing EXECUTE authority.
        let src = "fn priv_op(x) { return x + 1; }";
        // A domain without EXECUTE cannot even compile it.
        assert_eq!(
            compile_fn(src, Rights::READ),
            Err(DcgError::Unauthorized(Rights::EXECUTE))
        );
        // With EXECUTE granted, it compiles and carries the requirement.
        let dcg = compile_fn(src, Rights::EXECUTE).unwrap();
        assert_eq!(dcg.required, Rights::EXECUTE);
        assert_eq!(dcg.eval(&[41]).unwrap(), 42);
    }

    #[test]
    fn proof_token_is_stable_and_structure_sensitive() {
        let a = compile_fn("fn f(x) { return x + 1; }", Rights::ALL).unwrap();
        let b = compile_fn("fn f(x) { return x + 1; }", Rights::ALL).unwrap();
        let c = compile_fn("fn f(x) { return x + 2; }", Rights::ALL).unwrap();
        assert_eq!(a.proof(), b.proof()); // same graph ⇒ same proof
        assert_ne!(a.proof(), c.proof()); // any change ⇒ different proof
    }

    #[test]
    fn unsupported_constructs_are_rejected_cleanly() {
        // A call is outside the straight-line integer subset.
        let err = compile_fn("fn f(x) { return g(x); }", Rights::ALL).unwrap_err();
        assert!(matches!(err, DcgError::Unsupported(_)));
    }

    #[test]
    fn placement_decorator_is_carried_into_the_graph() {
        let dcg = compile_fn("@GPU fn f(a, b) { return a * b; }", Rights::ALL).unwrap();
        assert_eq!(dcg.placement, Placement::Gpu);
    }
}

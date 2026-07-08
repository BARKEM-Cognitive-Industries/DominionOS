//! The Dominion **pretty-printer**: [`Program`] AST → source text.
//!
//! This is the inverse of [`super::parser::parse_source`] and the linchpin of the
//! IDE's bidirectional graph⇄code sync: the visual node graph and the source buffer are
//! two views of the same AST, so editing either side re-emits the other. Emission is
//! **precedence-aware** — it inserts exactly the parentheses needed to round-trip, no
//! more — so `parse(to_source(p))` reconstructs an equal tree.
//!
//! Pure, safe `no_std`.

use super::ast::*;
use alloc::format;
use alloc::string::{String, ToString};

/// Render a whole program as Dominion source. Top-level items are separated by a blank
/// line; definitions span multiple indented lines; bare statements are one line each.
pub fn to_source(p: &Program) -> String {
    let mut out = String::new();
    for (i, item) in p.items.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        emit_item(&mut out, item, 0);
    }
    out
}

/// Render a single item (used by the IDE to re-emit one node's source fragment).
pub fn item_to_source(item: &Item) -> String {
    let mut out = String::new();
    emit_item(&mut out, item, 0);
    out
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("    ");
    }
}

fn emit_item(out: &mut String, item: &Item, depth: usize) {
    match item {
        Item::Object(o) => {
            indent(out, depth);
            out.push_str("object ");
            out.push_str(&o.name);
            out.push_str(" {\n");
            for (field, ty) in &o.fields {
                indent(out, depth + 1);
                out.push_str(field);
                out.push_str(": ");
                out.push_str(ty);
                out.push_str(",\n");
            }
            indent(out, depth);
            out.push('}');
            out.push('\n');
        }
        Item::Cell(c) => {
            indent(out, depth);
            out.push_str("cell ");
            out.push_str(&c.name);
            if let Some(cap) = &c.required_cap {
                out.push_str(" [cap: Capability<");
                out.push_str(cap);
                out.push_str(">]");
            }
            out.push_str(" {\n");
            for (i, m) in c.methods.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                emit_fn(out, m, depth + 1);
            }
            indent(out, depth);
            out.push('}');
            out.push('\n');
        }
        Item::Fn(f) => emit_fn(out, f, depth),
        Item::Stmt(s) => emit_stmt(out, s, depth),
    }
}

fn emit_fn(out: &mut String, f: &FnDef, depth: usize) {
    indent(out, depth);
    match f.placement {
        Placement::Cpu => out.push_str("@CPU "),
        Placement::Gpu => out.push_str("@GPU "),
        Placement::Npu => out.push_str("@NPU "),
        Placement::Any => {}
    }
    out.push_str("fn ");
    out.push_str(&f.name);
    out.push('(');
    out.push_str(&f.params.join(", "));
    out.push_str(") {\n");
    for s in &f.body {
        emit_stmt(out, s, depth + 1);
    }
    indent(out, depth);
    out.push('}');
    out.push('\n');
}

fn emit_stmt(out: &mut String, s: &Stmt, depth: usize) {
    indent(out, depth);
    match s {
        Stmt::Let(name, e) => {
            out.push_str("let ");
            out.push_str(name);
            out.push_str(" = ");
            out.push_str(&expr(e));
            out.push(';');
        }
        Stmt::Linear(name, e) => {
            out.push_str("linear ");
            out.push_str(name);
            out.push_str(" = ");
            out.push_str(&expr(e));
            out.push(';');
        }
        Stmt::Assign(name, e) => {
            out.push_str(name);
            out.push_str(" = ");
            out.push_str(&expr(e));
            out.push(';');
        }
        Stmt::Return(e) => {
            out.push_str("return ");
            out.push_str(&expr(e));
            out.push(';');
        }
        Stmt::Expr(e) => {
            out.push_str(&expr(e));
            out.push(';');
        }
        Stmt::If { cond, then_block, else_block } => {
            out.push_str("if ");
            out.push_str(&expr(cond));
            out.push_str(" {\n");
            for s in then_block {
                emit_stmt(out, s, depth + 1);
            }
            indent(out, depth);
            out.push('}');
            if !else_block.is_empty() {
                out.push_str(" else {\n");
                for s in else_block {
                    emit_stmt(out, s, depth + 1);
                }
                indent(out, depth);
                out.push('}');
            }
        }
        Stmt::While { cond, body } => {
            out.push_str("while ");
            out.push_str(&expr(cond));
            out.push_str(" {\n");
            for s in body {
                emit_stmt(out, s, depth + 1);
            }
            indent(out, depth);
            out.push('}');
        }
        Stmt::For { var, iter, body } => {
            out.push_str("for ");
            out.push_str(var);
            out.push_str(" in ");
            out.push_str(&expr(iter));
            out.push_str(" {\n");
            for s in body {
                emit_stmt(out, s, depth + 1);
            }
            indent(out, depth);
            out.push('}');
        }
        Stmt::Break => out.push_str("break;"),
        Stmt::Continue => out.push_str("continue;"),
    }
    out.push('\n');
}

// ── expressions (precedence-aware) ──
//
// Precedence, loosest → tightest (mirrors the parser):
//   1 Pipe · 2 Map · 3 logical-or · 4 logical-and · 5 equality · 6 comparison
//   · 7 additive · 8 multiplicative · 9 unary · 10 postfix(call/field/index)
//   · 11 primary

fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Pipe(_, _) => 1,
        Expr::Map(_, _) => 2,
        Expr::Binary(op, _, _) => op_prec(op),
        Expr::Neg(_) | Expr::Not(_) => 9,
        Expr::Call(_, _) | Expr::Field(_, _) | Expr::Index(_, _) => 10,
        _ => 11,
    }
}

fn op_prec(op: &BinOp) -> u8 {
    match op {
        BinOp::Or => 3,
        BinOp::And => 4,
        BinOp::Eq | BinOp::Ne => 5,
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 6,
        BinOp::Add | BinOp::Sub => 7,
        BinOp::Mul | BinOp::Div | BinOp::Rem => 8,
    }
}

fn op_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

/// Top-level expression (no enclosing operator).
pub fn expr(e: &Expr) -> String {
    expr_p(e, 0)
}

/// Emit `e`, wrapping it in parentheses if its precedence is looser than `min`.
fn expr_p(e: &Expr, min: u8) -> String {
    let s = raw(e);
    if prec(e) < min {
        format!("({})", s)
    } else {
        s
    }
}

fn raw(e: &Expr) -> String {
    match e {
        Expr::Int(v) => v.to_string(),
        Expr::Float(v) => fmt_float(*v),
        Expr::Str(s) => fmt_str(s),
        Expr::Bool(b) => if *b { "true".into() } else { "false".into() },
        Expr::Ident(n) => n.clone(),
        Expr::Path(parts) => parts.join("::"),
        Expr::Neg(x) => format!("-{}", expr_p(x, 9)),
        Expr::Not(x) => format!("!{}", expr_p(x, 9)),
        Expr::Binary(op, l, r) => {
            let p = op_prec(op);
            // Left-associative: left child shares the level, right child is one tighter.
            format!("{} {} {}", expr_p(l, p), op_str(op), expr_p(r, p + 1))
        }
        Expr::Map(l, r) => format!("{} => {}", expr_p(l, 2), expr_p(r, 3)),
        Expr::Pipe(l, r) => format!("{} |> {}", expr_p(l, 1), expr_p(r, 2)),
        Expr::Call(callee, args) => {
            let a: alloc::vec::Vec<String> = args.iter().map(expr).collect();
            format!("{}({})", expr_p(callee, 10), a.join(", "))
        }
        Expr::Field(obj, field) => format!("{}.{}", expr_p(obj, 10), field),
        Expr::Index(obj, idx) => format!("{}[{}]", expr_p(obj, 10), expr(idx)),
        Expr::Vector(items) => {
            let a: alloc::vec::Vec<String> = items.iter().map(expr).collect();
            format!("[{}]", a.join(", "))
        }
        Expr::ObjectLit(kind, fields) => {
            let f: alloc::vec::Vec<String> = fields.iter().map(|(n, v)| format!("{}: {}", n, expr(v))).collect();
            format!("{} {{ {} }}", kind, f.join(", "))
        }
    }
}

fn fmt_float(v: f64) -> String {
    // `{:?}` renders whole floats as e.g. "19.0" (so they re-lex as Float, not Int).
    let mut s = format!("{:?}", v);
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}

fn fmt_str(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::super::parser::parse_source;
    use super::*;

    /// The core property: emitting a parsed program and re-parsing reconstructs the
    /// same AST.
    fn round_trips(src: &str) {
        let a = parse_source(src).expect("parse src");
        let emitted = to_source(&a);
        let b = parse_source(&emitted).unwrap_or_else(|e| panic!("re-parse failed: {}\n--- emitted ---\n{}", e, emitted));
        assert_eq!(a, b, "AST changed across round-trip\n--- emitted ---\n{}", emitted);
    }

    #[test]
    fn round_trips_arithmetic_and_precedence() {
        round_trips("let x = 1 + 2 * 3;");
        round_trips("let y = (1 + 2) * 3;");
        round_trips("let z = 1 - 2 - 3;"); // left-assoc must be preserved
        round_trips("let w = 2 * (3 + 4) - 5 % 2;");
        round_trips("let c = a < b == c;");
    }

    #[test]
    fn round_trips_pipelines_and_maps() {
        round_trips("let r = data |> filter(p) |> A::summarise;");
        round_trips("let m = xs => A::dbl;");
        round_trips("let n = (a => b) |> c;");
        round_trips("let o = a |> b => c;");
    }

    #[test]
    fn round_trips_calls_fields_vectors_objects() {
        round_trips("let v = Codec::encode(doc, 3);");
        round_trips("let f = a.b.c;");
        round_trips("let l = [1, 2, 3];");
        round_trips("let p = Point { x: 1, y: 2 };");
        round_trips("let q = make().field;");
    }

    #[test]
    fn round_trips_definitions() {
        round_trips("object Invoice { id: Identity, amount: Money, date: Time }");
        round_trips("cell StorageManager [cap: Capability<StorageWrite>] { fn go(x) { return x; } }");
        round_trips("@NPU fn enc(x, y) { let z = x + y; return z; }");
        round_trips("fn plain(a) { if a > 0 { return a; } else { return 0 - a; } }");
    }

    #[test]
    fn round_trips_literals_and_strings() {
        round_trips("let s = \"hello\\n\\\"world\\\"\";");
        round_trips("let f = 3.5;");
        round_trips("let g = 19.0;"); // must keep the decimal point
        round_trips("let b = true; let c = false;");
    }

    #[test]
    fn round_trips_logical_index_and_loops() {
        round_trips("let a = true && false || !c;");
        round_trips("let b = x[0] + xs[i];");
        round_trips("fn f(n) { let s = 0; let i = 0; while i < n { s = s + i; i = i + 1; } return s; }");
        round_trips("fn g(xs) { for x in xs { if x > 0 { continue; } break; } }");
        round_trips("let d = !(a && b) == c;");
    }

    #[test]
    fn emits_readable_minimal_parens() {
        let a = parse_source("let x = 1 + 2 * 3;").unwrap();
        // No spurious parens around the multiply (it binds tighter than the add).
        assert_eq!(to_source(&a).trim(), "let x = 1 + 2 * 3;");
    }
}

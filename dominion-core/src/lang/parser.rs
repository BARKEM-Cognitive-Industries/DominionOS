//! Recursive-descent parser for Aether: tokens → [`Program`].
//!
//! Operator precedence, low → high:
//!   `=>` (parallel map) · `== !=` · `< > <= >=` · `+ -` · `* / %` · unary `-`
//!   · postfix `() . ::` · primaries.
//!
//! Object literals (`Kind { f: e }`) are disabled inside `if` conditions — the
//! same `no_struct` rule Rust uses — so `if x { ... }` is never misread as a
//! struct construction.

use super::ast::*;
use super::lexer::{LexError, Spanned, Tok};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, PartialEq, Debug)]
pub struct ParseError {
    pub message: String,
    pub line: u32,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error (line {}): {}", self.line, self.message)
    }
}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        ParseError {
            message: e.message,
            line: e.line,
        }
    }
}

pub fn parse(tokens: Vec<Spanned>) -> Result<Program, ParseError> {
    let mut p = Parser { toks: tokens, pos: 0 };
    p.program()
}

/// Convenience: lex + parse in one step.
pub fn parse_source(src: &str) -> Result<Program, ParseError> {
    let tokens = super::lexer::lex(src)?;
    parse(tokens)
}

struct Parser {
    toks: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn line(&self) -> u32 {
        self.toks[self.pos].line
    }

    fn advance(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn check(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.check(t) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, what: &str) -> Result<(), ParseError> {
        if self.check(t) {
            self.advance();
            Ok(())
        } else {
            Err(self.err(&format!("expected {}, found {:?}", what, self.peek())))
        }
    }

    fn err(&self, message: &str) -> ParseError {
        ParseError {
            message: message.to_string(),
            line: self.line(),
        }
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            other => Err(self.err(&format!("expected identifier, found {:?}", other))),
        }
    }

    // ---- items ----------------------------------------------------------

    fn program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        while !self.check(&Tok::Eof) {
            items.push(self.item()?);
        }
        Ok(Program { items })
    }

    fn item(&mut self) -> Result<Item, ParseError> {
        match self.peek() {
            Tok::Object => Ok(Item::Object(self.object_def()?)),
            Tok::Cell => Ok(Item::Cell(self.cell_def()?)),
            Tok::Fn | Tok::Decorator(_) => Ok(Item::Fn(self.fn_def()?)),
            _ => Ok(Item::Stmt(self.statement()?)),
        }
    }

    fn object_def(&mut self) -> Result<ObjectDef, ParseError> {
        self.expect(&Tok::Object, "'object'")?;
        let name = self.ident()?;
        self.expect(&Tok::LBrace, "'{'")?;
        let mut fields = Vec::new();
        while !self.check(&Tok::RBrace) {
            let field = self.ident()?;
            self.expect(&Tok::Colon, "':'")?;
            // type expression: an identifier possibly followed by (..) or <..>;
            // we capture just the head type name and skip any parameters.
            let ty = self.type_name()?;
            fields.push((field, ty));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RBrace, "'}'")?;
        Ok(ObjectDef { name, fields })
    }

    /// Parse a type annotation head, skipping `(...)` and `<...>` parameters.
    fn type_name(&mut self) -> Result<String, ParseError> {
        let name = self.ident()?;
        // skip generic / unit parameters: Money(USD), Vector<Invoice>, Capability<StorageWrite>
        if self.check(&Tok::LParen) {
            self.skip_balanced(&Tok::LParen, &Tok::RParen)?;
        }
        if self.check(&Tok::Lt) {
            self.skip_balanced(&Tok::Lt, &Tok::Gt)?;
        }
        Ok(name)
    }

    fn skip_balanced(&mut self, open: &Tok, close: &Tok) -> Result<(), ParseError> {
        self.expect(open, "opening bracket")?;
        let mut depth = 1;
        while depth > 0 {
            if self.check(&Tok::Eof) {
                return Err(self.err("unbalanced brackets in type"));
            }
            if self.check(open) {
                depth += 1;
            } else if self.check(close) {
                depth -= 1;
            }
            self.advance();
        }
        Ok(())
    }

    fn cell_def(&mut self) -> Result<CellDef, ParseError> {
        self.expect(&Tok::Cell, "'cell'")?;
        let name = self.ident()?;
        let mut required_cap = None;
        // optional capability requirement: [cap: Capability<StorageWrite>]
        if self.eat(&Tok::LBracket) {
            self.expect(&Tok::Cap, "'cap'")?;
            self.expect(&Tok::Colon, "':'")?;
            // Capability<RightName>
            let head = self.ident()?; // "Capability"
            if head == "Capability" && self.eat(&Tok::Lt) {
                required_cap = Some(self.ident()?);
                self.expect(&Tok::Gt, "'>'")?;
            }
            self.expect(&Tok::RBracket, "']'")?;
        }
        self.expect(&Tok::LBrace, "'{'")?;
        let mut methods = Vec::new();
        while !self.check(&Tok::RBrace) {
            methods.push(self.fn_def()?);
        }
        self.expect(&Tok::RBrace, "'}'")?;
        Ok(CellDef {
            name,
            required_cap,
            methods,
        })
    }

    fn fn_def(&mut self) -> Result<FnDef, ParseError> {
        let placement = match self.peek().clone() {
            Tok::Decorator(d) => {
                self.advance();
                match d.as_str() {
                    "CPU" => Placement::Cpu,
                    "GPU" => Placement::Gpu,
                    "NPU" => Placement::Npu,
                    _ => Placement::Any,
                }
            }
            _ => Placement::Any,
        };
        self.expect(&Tok::Fn, "'fn'")?;
        let name = self.ident()?;
        self.expect(&Tok::LParen, "'('")?;
        let mut params = Vec::new();
        while !self.check(&Tok::RParen) {
            let pname = self.ident()?;
            // optional `: Type`
            if self.eat(&Tok::Colon) {
                self.type_name()?;
            }
            params.push(pname);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen, "')'")?;
        // optional return type `-> Type`
        if self.eat(&Tok::ThinArrow) {
            self.type_name()?;
        }
        let body = self.block()?;
        Ok(FnDef {
            name,
            params,
            body,
            placement,
        })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect(&Tok::LBrace, "'{'")?;
        let mut stmts = Vec::new();
        while !self.check(&Tok::RBrace) && !self.check(&Tok::Eof) {
            stmts.push(self.statement()?);
        }
        self.expect(&Tok::RBrace, "'}'")?;
        Ok(stmts)
    }

    // ---- statements -----------------------------------------------------

    fn statement(&mut self) -> Result<Stmt, ParseError> {
        match self.peek() {
            Tok::Let => {
                self.advance();
                let name = self.ident()?;
                self.expect(&Tok::Assign, "'='")?;
                let value = self.expr()?;
                self.eat(&Tok::Semicolon);
                Ok(Stmt::Let(name, value))
            }
            Tok::Linear => {
                self.advance();
                let name = self.ident()?;
                self.expect(&Tok::Assign, "'='")?;
                let value = self.expr()?;
                self.eat(&Tok::Semicolon);
                Ok(Stmt::Linear(name, value))
            }
            Tok::Return => {
                self.advance();
                let value = self.expr()?;
                self.eat(&Tok::Semicolon);
                Ok(Stmt::Return(value))
            }
            Tok::If => self.if_stmt(),
            Tok::While => {
                self.advance();
                let cond = self.expr_no_struct()?;
                let body = self.block()?;
                Ok(Stmt::While { cond, body })
            }
            Tok::For => {
                self.advance();
                let var = self.ident()?;
                self.expect(&Tok::In, "'in'")?;
                let iter = self.expr_no_struct()?;
                let body = self.block()?;
                Ok(Stmt::For { var, iter, body })
            }
            Tok::Break => {
                self.advance();
                self.eat(&Tok::Semicolon);
                Ok(Stmt::Break)
            }
            Tok::Continue => {
                self.advance();
                self.eat(&Tok::Semicolon);
                Ok(Stmt::Continue)
            }
            _ => {
                let e = self.expr()?;
                // assignment: `ident = expr`
                if self.check(&Tok::Assign) {
                    let name = match e {
                        Expr::Ident(n) => n,
                        _ => return Err(self.err("left side of '=' must be a variable name")),
                    };
                    self.advance(); // '='
                    let value = self.expr()?;
                    self.eat(&Tok::Semicolon);
                    return Ok(Stmt::Assign(name, value));
                }
                self.eat(&Tok::Semicolon);
                Ok(Stmt::Expr(e))
            }
        }
    }

    fn if_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(&Tok::If, "'if'")?;
        let cond = self.expr_no_struct()?;
        let then_block = self.block()?;
        let else_block = if self.eat(&Tok::Else) {
            if self.check(&Tok::If) {
                // else if -> nest
                alloc::vec![self.if_stmt()?]
            } else {
                self.block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If {
            cond,
            then_block,
            else_block,
        })
    }

    // ---- expressions ----------------------------------------------------

    fn expr(&mut self) -> Result<Expr, ParseError> {
        self.pipe_expr(true)
    }

    fn expr_no_struct(&mut self) -> Result<Expr, ParseError> {
        self.pipe_expr(false)
    }

    /// `|>` binds loosest of all the binary operators, so `a + b |> f` is
    /// `(a + b) |> f` and pipelines chain left-to-right.
    fn pipe_expr(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.map_expr(structs)?;
        while self.eat(&Tok::Pipe) {
            let right = self.map_expr(structs)?;
            left = Expr::Pipe(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn map_expr(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.logic_or(structs)?;
        while self.eat(&Tok::FatArrow) {
            let right = self.logic_or(structs)?;
            left = Expr::Map(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn logic_or(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.logic_and(structs)?;
        while self.eat(&Tok::OrOr) {
            let right = self.logic_and(structs)?;
            left = Expr::Binary(BinOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn logic_and(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.equality(structs)?;
        while self.eat(&Tok::AndAnd) {
            let right = self.equality(structs)?;
            left = Expr::Binary(BinOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn equality(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.comparison(structs)?;
        loop {
            let op = match self.peek() {
                Tok::EqEq => BinOp::Eq,
                Tok::NotEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let right = self.comparison(structs)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn comparison(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.additive(structs)?;
        loop {
            let op = match self.peek() {
                Tok::Lt => BinOp::Lt,
                Tok::Gt => BinOp::Gt,
                Tok::Le => BinOp::Le,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.additive(structs)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn additive(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.multiplicative(structs)?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.multiplicative(structs)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn multiplicative(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut left = self.unary(structs)?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Rem,
                _ => break,
            };
            self.advance();
            let right = self.unary(structs)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self, structs: bool) -> Result<Expr, ParseError> {
        if self.eat(&Tok::Minus) {
            let e = self.unary(structs)?;
            return Ok(Expr::Neg(Box::new(e)));
        }
        if self.eat(&Tok::Bang) {
            let e = self.unary(structs)?;
            return Ok(Expr::Not(Box::new(e)));
        }
        self.postfix(structs)
    }

    fn postfix(&mut self, structs: bool) -> Result<Expr, ParseError> {
        let mut e = self.primary(structs)?;
        loop {
            match self.peek() {
                Tok::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.check(&Tok::RParen) {
                        args.push(self.expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "')'")?;
                    e = Expr::Call(Box::new(e), args);
                }
                Tok::LBracket => {
                    self.advance();
                    let idx = self.expr()?;
                    self.expect(&Tok::RBracket, "']'")?;
                    e = Expr::Index(Box::new(e), Box::new(idx));
                }
                Tok::Dot => {
                    self.advance();
                    let field = self.ident()?;
                    e = Expr::Field(Box::new(e), field);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn primary(&mut self, structs: bool) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Tok::Int(v) => {
                self.advance();
                Ok(Expr::Int(v))
            }
            Tok::Float(v) => {
                self.advance();
                Ok(Expr::Float(v))
            }
            Tok::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Tok::True => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Tok::False => {
                self.advance();
                Ok(Expr::Bool(false))
            }
            Tok::LParen => {
                self.advance();
                let e = self.expr()?;
                self.expect(&Tok::RParen, "')'")?;
                Ok(e)
            }
            Tok::LBracket => {
                self.advance();
                let mut items = Vec::new();
                while !self.check(&Tok::RBracket) {
                    items.push(self.expr()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBracket, "']'")?;
                Ok(Expr::Vector(items))
            }
            Tok::Ident(first) => {
                self.advance();
                // path?  A::b::c
                if self.check(&Tok::ColonColon) {
                    let mut parts = alloc::vec![first];
                    while self.eat(&Tok::ColonColon) {
                        parts.push(self.ident()?);
                    }
                    return Ok(Expr::Path(parts));
                }
                // object literal?  Kind { field: expr, ... }
                if structs && self.check(&Tok::LBrace) {
                    return self.object_literal(first);
                }
                Ok(Expr::Ident(first))
            }
            other => Err(self.err(&format!("unexpected token in expression: {:?}", other))),
        }
    }

    fn object_literal(&mut self, kind: String) -> Result<Expr, ParseError> {
        self.expect(&Tok::LBrace, "'{'")?;
        let mut fields = Vec::new();
        while !self.check(&Tok::RBrace) {
            let name = self.ident()?;
            self.expect(&Tok::Colon, "':'")?;
            let value = self.expr()?;
            fields.push((name, value));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RBrace, "'}'")?;
        Ok(Expr::ObjectLit(kind, fields))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prog(src: &str) -> Program {
        parse_source(src).unwrap()
    }

    #[test]
    fn parses_let_and_arithmetic() {
        let p = prog("let x = 1 + 2 * 3;");
        assert_eq!(p.items.len(), 1);
        match &p.items[0] {
            Item::Stmt(Stmt::Let(n, e)) => {
                assert_eq!(n, "x");
                // precedence: 1 + (2*3)
                match e {
                    Expr::Binary(BinOp::Add, _, rhs) => {
                        assert!(matches!(**rhs, Expr::Binary(BinOp::Mul, _, _)));
                    }
                    _ => panic!("bad tree"),
                }
            }
            _ => panic!("expected let"),
        }
    }

    #[test]
    fn parses_object_def() {
        let p = prog("object Invoice { id: Identity, amount: Money(USD), date: Time }");
        match &p.items[0] {
            Item::Object(o) => {
                assert_eq!(o.name, "Invoice");
                assert_eq!(o.fields.len(), 3);
                assert_eq!(o.fields[1], ("amount".to_string(), "Money".to_string()));
            }
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn parses_cell_with_capability() {
        let p = prog("cell StorageManager [cap: Capability<StorageWrite>] { fn go(x) { return x; } }");
        match &p.items[0] {
            Item::Cell(c) => {
                assert_eq!(c.name, "StorageManager");
                assert_eq!(c.required_cap.as_deref(), Some("StorageWrite"));
                assert_eq!(c.methods.len(), 1);
            }
            _ => panic!("expected cell"),
        }
    }

    #[test]
    fn parses_decorator_placement() {
        let p = prog("@NPU fn enc(x) { return x; }");
        match &p.items[0] {
            Item::Fn(f) => assert_eq!(f.placement, Placement::Npu),
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn parses_map_operator() {
        let p = prog("let r = xs => A::b;");
        match &p.items[0] {
            Item::Stmt(Stmt::Let(_, Expr::Map(l, r))) => {
                assert!(matches!(**l, Expr::Ident(_)));
                assert!(matches!(**r, Expr::Path(_)));
            }
            _ => panic!("expected map"),
        }
    }

    #[test]
    fn parses_object_literal_and_field() {
        let p = prog("let a = Point { x: 1, y: 2 }; let b = a.x;");
        assert!(matches!(&p.items[0], Item::Stmt(Stmt::Let(_, Expr::ObjectLit(_, _)))));
        assert!(matches!(&p.items[1], Item::Stmt(Stmt::Let(_, Expr::Field(_, _)))));
    }

    #[test]
    fn if_condition_is_not_a_struct_literal() {
        // `if flag { ... }` must parse flag as an identifier, not Kind{...}
        let p = prog("if flag { let x = 1; }");
        assert!(matches!(&p.items[0], Item::Stmt(Stmt::If { .. })));
    }

    #[test]
    fn parses_call_chain() {
        let p = prog("let v = Codec::encode(doc);");
        match &p.items[0] {
            Item::Stmt(Stmt::Let(_, Expr::Call(callee, args))) => {
                assert!(matches!(**callee, Expr::Path(_)));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected call"),
        }
    }

    #[test]
    fn reports_error_with_line() {
        let err = parse_source("let x = ;").unwrap_err();
        assert_eq!(err.line, 1);
    }
}

//! A real JavaScript engine for the browser: a lexer, a recursive-descent/Pratt
//! parser, and a tree-walking interpreter with closures, plus **live DOM bindings**
//! so scripts mutate the same [`dom`](crate::dom) tree the page renders from.
//!
//! It is a substantial subset, not all of ECMAScript. Supported: `var`/`let`/`const`,
//! functions + closures + arrow functions, `if`/`else`, `while`, C-style `for` and
//! `for…of`, `break`/`continue`/`return`, objects and arrays, member/index access,
//! the full operator set (`+ - * / %`, comparisons, `=== !==`, `&& ||`, `!`, ternary,
//! `++`/`--`, compound assignment), template literals, and `typeof`/`new`. Built-ins:
//! `console`, `Math`, `JSON`, `parseInt/parseFloat`, and the common String/Array
//! methods. DOM: `document.{getElementById,querySelector(All),createElement,body,
//! title}`, and element `.textContent/.innerHTML/.id/.className/.value/.style.*/
//! .getAttribute/.setAttribute/.appendChild/.addEventListener` plus inline `on*`
//! handlers. Not supported: classes, generators, async/await, regex, modules.
//!
//! Determinism is preserved: `Math.random` is a seeded xorshift, so a page renders
//! identically on replay. Pure, safe `no_std`.

use crate::dom::{self, NodeRef};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

// ════════════════════════════ lexer ════════════════════════════

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Template(String),
    Ident(String),
    Keyword(String),
    Punct(String),
    Eof,
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "var" | "let" | "const" | "function" | "return" | "if" | "else" | "while" | "for" | "of"
            | "in" | "break" | "continue" | "true" | "false" | "null" | "undefined" | "typeof"
            | "new" | "this" | "void" | "delete"
    )
}

struct Lexer<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer { s: src.as_bytes(), src, i: 0 }
    }

    fn tokens(mut self) -> Result<Vec<Tok>, String> {
        let mut out = Vec::new();
        loop {
            let t = self.next_tok()?;
            let end = t == Tok::Eof;
            out.push(t);
            if end {
                break;
            }
        }
        Ok(out)
    }

    fn next_tok(&mut self) -> Result<Tok, String> {
        self.skip_ws_comments();
        if self.i >= self.s.len() {
            return Ok(Tok::Eof);
        }
        let c = self.s[self.i] as char;
        if c.is_ascii_digit() || (c == '.' && self.peek(1).map(|d| d.is_ascii_digit()).unwrap_or(false)) {
            return Ok(self.number());
        }
        if c == '"' || c == '\'' {
            return self.string(c);
        }
        if c == '`' {
            return self.template();
        }
        if c == '_' || c == '$' || c.is_ascii_alphabetic() {
            return Ok(self.ident());
        }
        self.punct()
    }

    fn peek(&self, ahead: usize) -> Option<char> {
        self.s.get(self.i + ahead).map(|b| *b as char)
    }

    fn skip_ws_comments(&mut self) {
        loop {
            while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
                self.i += 1;
            }
            if self.i + 1 < self.s.len() && self.s[self.i] == b'/' && self.s[self.i + 1] == b'/' {
                while self.i < self.s.len() && self.s[self.i] != b'\n' {
                    self.i += 1;
                }
                continue;
            }
            if self.i + 1 < self.s.len() && self.s[self.i] == b'/' && self.s[self.i + 1] == b'*' {
                self.i += 2;
                while self.i + 1 < self.s.len() && !(self.s[self.i] == b'*' && self.s[self.i + 1] == b'/') {
                    self.i += 1;
                }
                self.i += 2;
                continue;
            }
            break;
        }
    }

    fn number(&mut self) -> Tok {
        let start = self.i;
        if self.s[self.i] == b'0' && self.peek(1).map(|c| c == 'x' || c == 'X').unwrap_or(false) {
            self.i += 2;
            let hs = self.i;
            while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_hexdigit() {
                self.i += 1;
            }
            let v = i64::from_str_radix(&self.src[hs..self.i], 16).unwrap_or(0);
            return Tok::Num(v as f64);
        }
        while self.i < self.s.len() {
            let c = self.s[self.i] as char;
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || (c == '-' && matches!(self.s[self.i - 1] as char, 'e' | 'E')) {
                self.i += 1;
            } else {
                break;
            }
        }
        Tok::Num(self.src[start..self.i].parse::<f64>().unwrap_or(0.0))
    }

    fn string(&mut self, quote: char) -> Result<Tok, String> {
        self.i += 1;
        let mut out = String::new();
        while self.i < self.s.len() {
            let c = self.s[self.i] as char;
            if c == quote {
                self.i += 1;
                return Ok(Tok::Str(out));
            }
            if c == '\\' {
                self.i += 1;
                if self.i >= self.s.len() {
                    break;
                }
                let e = self.s[self.i] as char;
                out.push(match e {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    '`' => '`',
                    '0' => '\0',
                    other => other,
                });
                self.i += 1;
            } else {
                out.push(c);
                self.i += c.len_utf8();
            }
        }
        Err("unterminated string".to_string())
    }

    fn template(&mut self) -> Result<Tok, String> {
        self.i += 1;
        let start = self.i;
        let mut depth = 0;
        while self.i < self.s.len() {
            let c = self.s[self.i] as char;
            if c == '`' && depth == 0 {
                let inner = self.src[start..self.i].to_string();
                self.i += 1;
                return Ok(Tok::Template(inner));
            }
            if c == '{' {
                depth += 1;
            } else if c == '}' && depth > 0 {
                depth -= 1;
            }
            self.i += c.len_utf8();
        }
        Err("unterminated template literal".to_string())
    }

    fn ident(&mut self) -> Tok {
        let start = self.i;
        while self.i < self.s.len() {
            let c = self.s[self.i] as char;
            if c == '_' || c == '$' || c.is_ascii_alphanumeric() {
                self.i += 1;
            } else {
                break;
            }
        }
        let word = &self.src[start..self.i];
        if is_keyword(word) {
            Tok::Keyword(word.to_string())
        } else {
            Tok::Ident(word.to_string())
        }
    }

    fn punct(&mut self) -> Result<Tok, String> {
        const THREE: [&str; 3] = ["===", "!==", "..."];
        const TWO: [&str; 14] =
            ["==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "++", "--", "=>", "%="];
        let rest = &self.src[self.i..];
        for p in THREE {
            if rest.starts_with(p) {
                self.i += 3;
                return Ok(Tok::Punct(p.to_string()));
            }
        }
        for p in TWO {
            if rest.starts_with(p) {
                self.i += 2;
                return Ok(Tok::Punct(p.to_string()));
            }
        }
        let c = self.s[self.i] as char;
        if "(){}[];,.<>+-*/%=!?:&|".contains(c) {
            self.i += 1;
            return Ok(Tok::Punct(c.to_string()));
        }
        Err(alloc::format!("unexpected character '{}'", c))
    }
}

// ════════════════════════════ AST ════════════════════════════

#[derive(Clone, Debug)]
enum Stmt {
    VarDecl(Vec<(String, Option<Expr>)>),
    FnDecl(String, Rc<FnDef>),
    Return(Option<Expr>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
    For(Option<Box<Stmt>>, Option<Expr>, Option<Expr>, Box<Stmt>),
    ForOf(String, Expr, Box<Stmt>),
    Block(Vec<Stmt>),
    Break,
    Continue,
    Expr(Expr),
    Empty,
}

#[derive(Clone, Debug)]
struct FnDef {
    params: Vec<String>,
    body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
enum Expr {
    Num(f64),
    Str(String),
    Template(Vec<TplPart>),
    Bool(bool),
    Null,
    Undefined,
    Ident(String),
    This,
    Array(Vec<Expr>),
    Object(Vec<(String, Expr)>),
    Function(Rc<FnDef>),
    Arrow(Rc<FnDef>),
    Unary(String, Box<Expr>),
    Update(String, bool, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Logical(String, Box<Expr>, Box<Expr>),
    Assign(String, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    New(Box<Expr>, Vec<Expr>),
    Member(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
enum TplPart {
    Str(String),
    Expr(Box<Expr>),
}

// ════════════════════════════ parser ════════════════════════════

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

type PResult<T> = Result<T, String>;

impl Parser {
    fn new(toks: Vec<Tok>) -> Parser {
        Parser { toks, i: 0 }
    }

    fn peek(&self) -> &Tok {
        self.toks.get(self.i).unwrap_or(&Tok::Eof)
    }
    fn advance(&mut self) -> Tok {
        let t = self.toks.get(self.i).cloned().unwrap_or(Tok::Eof);
        self.i += 1;
        t
    }
    fn is_punct(&self, p: &str) -> bool {
        matches!(self.peek(), Tok::Punct(x) if x == p)
    }
    fn is_kw(&self, k: &str) -> bool {
        matches!(self.peek(), Tok::Keyword(x) if x == k)
    }
    fn eat_punct(&mut self, p: &str) -> bool {
        if self.is_punct(p) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_punct(&mut self, p: &str) -> PResult<()> {
        if self.eat_punct(p) {
            Ok(())
        } else {
            Err(alloc::format!("expected '{}', found {:?}", p, self.peek()))
        }
    }
    fn eat_kw(&mut self, k: &str) -> bool {
        if self.is_kw(k) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn program(&mut self) -> PResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            stmts.push(self.statement()?);
        }
        Ok(stmts)
    }

    fn statement(&mut self) -> PResult<Stmt> {
        if self.eat_punct(";") {
            return Ok(Stmt::Empty);
        }
        if self.is_punct("{") {
            return Ok(Stmt::Block(self.block()?));
        }
        if self.is_kw("var") || self.is_kw("let") || self.is_kw("const") {
            self.advance();
            let decl = self.var_decl()?;
            self.eat_punct(";");
            return Ok(decl);
        }
        if self.eat_kw("function") {
            let name = self.ident_name()?;
            let def = self.fn_rest()?;
            return Ok(Stmt::FnDecl(name, Rc::new(def)));
        }
        if self.eat_kw("return") {
            if self.is_punct(";") || self.is_punct("}") || matches!(self.peek(), Tok::Eof) {
                self.eat_punct(";");
                return Ok(Stmt::Return(None));
            }
            let e = self.expression()?;
            self.eat_punct(";");
            return Ok(Stmt::Return(Some(e)));
        }
        if self.eat_kw("if") {
            self.expect_punct("(")?;
            let cond = self.expression()?;
            self.expect_punct(")")?;
            let then = Box::new(self.statement()?);
            let els = if self.eat_kw("else") { Some(Box::new(self.statement()?)) } else { None };
            return Ok(Stmt::If(cond, then, els));
        }
        if self.eat_kw("while") {
            self.expect_punct("(")?;
            let cond = self.expression()?;
            self.expect_punct(")")?;
            let body = Box::new(self.statement()?);
            return Ok(Stmt::While(cond, body));
        }
        if self.eat_kw("for") {
            return self.for_stmt();
        }
        if self.eat_kw("break") {
            self.eat_punct(";");
            return Ok(Stmt::Break);
        }
        if self.eat_kw("continue") {
            self.eat_punct(";");
            return Ok(Stmt::Continue);
        }
        let e = self.expression()?;
        self.eat_punct(";");
        Ok(Stmt::Expr(e))
    }

    fn block(&mut self) -> PResult<Vec<Stmt>> {
        self.expect_punct("{")?;
        let mut stmts = Vec::new();
        while !self.is_punct("}") && !matches!(self.peek(), Tok::Eof) {
            stmts.push(self.statement()?);
        }
        self.expect_punct("}")?;
        Ok(stmts)
    }

    fn var_decl(&mut self) -> PResult<Stmt> {
        let mut decls = Vec::new();
        loop {
            let name = self.ident_name()?;
            let init = if self.eat_punct("=") { Some(self.assignment()?) } else { None };
            decls.push((name, init));
            if !self.eat_punct(",") {
                break;
            }
        }
        Ok(Stmt::VarDecl(decls))
    }

    fn for_stmt(&mut self) -> PResult<Stmt> {
        self.expect_punct("(")?;
        let checkpoint = self.i;
        let is_decl = self.is_kw("var") || self.is_kw("let") || self.is_kw("const");
        if is_decl {
            self.advance();
            if let Tok::Ident(name) = self.peek().clone() {
                self.advance();
                if self.eat_kw("of") || self.eat_kw("in") {
                    let iter = self.expression()?;
                    self.expect_punct(")")?;
                    let body = Box::new(self.statement()?);
                    return Ok(Stmt::ForOf(name, iter, body));
                }
            }
            self.i = checkpoint;
        }
        let init = if self.is_punct(";") {
            None
        } else if self.is_kw("var") || self.is_kw("let") || self.is_kw("const") {
            self.advance();
            Some(Box::new(self.var_decl()?))
        } else {
            Some(Box::new(Stmt::Expr(self.expression()?)))
        };
        self.expect_punct(";")?;
        let cond = if self.is_punct(";") { None } else { Some(self.expression()?) };
        self.expect_punct(";")?;
        let update = if self.is_punct(")") { None } else { Some(self.expression()?) };
        self.expect_punct(")")?;
        let body = Box::new(self.statement()?);
        Ok(Stmt::For(init, cond, update, body))
    }

    fn fn_rest(&mut self) -> PResult<FnDef> {
        self.expect_punct("(")?;
        let params = self.param_list()?;
        let body = self.block()?;
        Ok(FnDef { params, body })
    }

    fn param_list(&mut self) -> PResult<Vec<String>> {
        let mut params = Vec::new();
        while !self.is_punct(")") {
            params.push(self.ident_name()?);
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct(")")?;
        Ok(params)
    }

    fn ident_name(&mut self) -> PResult<String> {
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            Tok::Keyword(s) if matches!(s.as_str(), "of" | "in") => Ok(s),
            other => Err(alloc::format!("expected identifier, found {:?}", other)),
        }
    }

    fn expression(&mut self) -> PResult<Expr> {
        self.assignment()
    }

    fn assignment(&mut self) -> PResult<Expr> {
        let left = self.ternary()?;
        for op in ["=", "+=", "-=", "*=", "/=", "%="] {
            if self.is_punct(op) {
                self.advance();
                let right = self.assignment()?;
                return Ok(Expr::Assign(op.to_string(), Box::new(left), Box::new(right)));
            }
        }
        Ok(left)
    }

    fn ternary(&mut self) -> PResult<Expr> {
        let cond = self.logical_or()?;
        if self.eat_punct("?") {
            let then = self.assignment()?;
            self.expect_punct(":")?;
            let els = self.assignment()?;
            return Ok(Expr::Ternary(Box::new(cond), Box::new(then), Box::new(els)));
        }
        Ok(cond)
    }

    fn logical_or(&mut self) -> PResult<Expr> {
        let mut left = self.logical_and()?;
        while self.is_punct("||") {
            self.advance();
            let right = self.logical_and()?;
            left = Expr::Logical("||".to_string(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn logical_and(&mut self) -> PResult<Expr> {
        let mut left = self.equality()?;
        while self.is_punct("&&") {
            self.advance();
            let right = self.equality()?;
            left = Expr::Logical("&&".to_string(), Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn equality(&mut self) -> PResult<Expr> {
        let mut left = self.relational()?;
        loop {
            let op = match self.peek() {
                Tok::Punct(p) if matches!(p.as_str(), "==" | "!=" | "===" | "!==") => p.clone(),
                _ => break,
            };
            self.advance();
            let right = self.relational()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn relational(&mut self) -> PResult<Expr> {
        let mut left = self.additive()?;
        loop {
            let op = match self.peek() {
                Tok::Punct(p) if matches!(p.as_str(), "<" | ">" | "<=" | ">=") => p.clone(),
                Tok::Keyword(k) if k == "in" => k.clone(),
                _ => break,
            };
            self.advance();
            let right = self.additive()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn additive(&mut self) -> PResult<Expr> {
        let mut left = self.multiplicative()?;
        loop {
            let op = match self.peek() {
                Tok::Punct(p) if matches!(p.as_str(), "+" | "-") => p.clone(),
                _ => break,
            };
            self.advance();
            let right = self.multiplicative()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn multiplicative(&mut self) -> PResult<Expr> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Punct(p) if matches!(p.as_str(), "*" | "/" | "%") => p.clone(),
                _ => break,
            };
            self.advance();
            let right = self.unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> PResult<Expr> {
        if self.is_punct("!") || self.is_punct("-") || self.is_punct("+") {
            let op = if let Tok::Punct(p) = self.advance() { p } else { unreachable!() };
            let e = self.unary()?;
            return Ok(Expr::Unary(op, Box::new(e)));
        }
        if self.is_kw("typeof") || self.is_kw("void") || self.is_kw("delete") {
            let op = if let Tok::Keyword(k) = self.advance() { k } else { unreachable!() };
            let e = self.unary()?;
            return Ok(Expr::Unary(op, Box::new(e)));
        }
        if self.is_punct("++") || self.is_punct("--") {
            let op = if let Tok::Punct(p) = self.advance() { p } else { unreachable!() };
            let e = self.unary()?;
            return Ok(Expr::Update(op, true, Box::new(e)));
        }
        if self.eat_kw("new") {
            let callee = self.postfix()?;
            if let Expr::Call(f, args) = callee {
                return Ok(Expr::New(f, args));
            }
            return Ok(Expr::New(Box::new(callee), Vec::new()));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            if self.eat_punct(".") {
                let name = self.member_name()?;
                e = Expr::Member(Box::new(e), name);
            } else if self.eat_punct("[") {
                let idx = self.expression()?;
                self.expect_punct("]")?;
                e = Expr::Index(Box::new(e), Box::new(idx));
            } else if self.is_punct("(") {
                let args = self.args()?;
                e = Expr::Call(Box::new(e), args);
            } else if self.is_punct("++") || self.is_punct("--") {
                let op = if let Tok::Punct(p) = self.advance() { p } else { unreachable!() };
                e = Expr::Update(op, false, Box::new(e));
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn member_name(&mut self) -> PResult<String> {
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            Tok::Keyword(s) => Ok(s),
            other => Err(alloc::format!("expected property name, found {:?}", other)),
        }
    }

    fn args(&mut self) -> PResult<Vec<Expr>> {
        self.expect_punct("(")?;
        let mut args = Vec::new();
        while !self.is_punct(")") {
            args.push(self.assignment()?);
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct(")")?;
        Ok(args)
    }

    fn primary(&mut self) -> PResult<Expr> {
        if let Some(arrow) = self.try_arrow()? {
            return Ok(arrow);
        }
        match self.advance() {
            Tok::Num(n) => Ok(Expr::Num(n)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::Template(raw) => self.parse_template(&raw),
            Tok::Ident(name) => Ok(Expr::Ident(name)),
            Tok::Keyword(k) => match k.as_str() {
                "true" => Ok(Expr::Bool(true)),
                "false" => Ok(Expr::Bool(false)),
                "null" => Ok(Expr::Null),
                "undefined" => Ok(Expr::Undefined),
                "this" => Ok(Expr::This),
                "function" => {
                    let def = self.fn_rest()?;
                    Ok(Expr::Function(Rc::new(def)))
                }
                other => Err(alloc::format!("unexpected keyword '{}'", other)),
            },
            Tok::Punct(p) if p == "(" => {
                let e = self.expression()?;
                self.expect_punct(")")?;
                Ok(e)
            }
            Tok::Punct(p) if p == "[" => {
                let mut items = Vec::new();
                while !self.is_punct("]") {
                    items.push(self.assignment()?);
                    if !self.eat_punct(",") {
                        break;
                    }
                }
                self.expect_punct("]")?;
                Ok(Expr::Array(items))
            }
            Tok::Punct(p) if p == "{" => self.object_literal(),
            other => Err(alloc::format!("unexpected token {:?}", other)),
        }
    }

    fn object_literal(&mut self) -> PResult<Expr> {
        let mut props = Vec::new();
        while !self.is_punct("}") {
            let key = match self.advance() {
                Tok::Ident(s) | Tok::Keyword(s) => s,
                Tok::Str(s) => s,
                Tok::Num(n) => fmt_num(n),
                other => return Err(alloc::format!("bad object key {:?}", other)),
            };
            let val = if self.eat_punct(":") {
                self.assignment()?
            } else {
                Expr::Ident(key.clone())
            };
            props.push((key, val));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct("}")?;
        Ok(Expr::Object(props))
    }

    fn try_arrow(&mut self) -> PResult<Option<Expr>> {
        let start = self.i;
        if let Tok::Ident(name) = self.peek().clone() {
            if matches!(self.toks.get(self.i + 1), Some(Tok::Punct(p)) if p == "=>") {
                self.i += 2;
                let body = self.arrow_body()?;
                return Ok(Some(Expr::Arrow(Rc::new(FnDef { params: alloc::vec![name], body }))));
            }
        }
        if self.is_punct("(") && self.scan_arrow_params() {
            self.expect_punct("(")?;
            let plist = self.param_list()?;
            self.expect_punct("=>")?;
            let body = self.arrow_body()?;
            return Ok(Some(Expr::Arrow(Rc::new(FnDef { params: plist, body }))));
        }
        self.i = start;
        Ok(None)
    }

    fn scan_arrow_params(&self) -> bool {
        let mut j = self.i;
        if !matches!(self.toks.get(j), Some(Tok::Punct(p)) if p == "(") {
            return false;
        }
        let mut depth = 0;
        while let Some(t) = self.toks.get(j) {
            match t {
                Tok::Punct(p) if p == "(" => depth += 1,
                Tok::Punct(p) if p == ")" => {
                    depth -= 1;
                    if depth == 0 {
                        return matches!(self.toks.get(j + 1), Some(Tok::Punct(p)) if p == "=>");
                    }
                }
                Tok::Eof => return false,
                _ => {}
            }
            j += 1;
        }
        false
    }

    fn arrow_body(&mut self) -> PResult<Vec<Stmt>> {
        if self.is_punct("{") {
            self.block()
        } else {
            let e = self.assignment()?;
            Ok(alloc::vec![Stmt::Return(Some(e))])
        }
    }

    fn parse_template(&mut self, raw: &str) -> PResult<Expr> {
        let mut parts = Vec::new();
        let bytes = raw.as_bytes();
        let mut i = 0;
        let mut lit = String::new();
        while i < bytes.len() {
            if i + 1 < bytes.len() && bytes[i] == b'$' && bytes[i + 1] == b'{' {
                if !lit.is_empty() {
                    parts.push(TplPart::Str(core::mem::take(&mut lit)));
                }
                let mut depth = 1;
                let mut j = i + 2;
                while j < bytes.len() && depth > 0 {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        _ => {}
                    }
                    if depth == 0 {
                        break;
                    }
                    j += 1;
                }
                let expr_src = &raw[i + 2..j];
                let toks = Lexer::new(expr_src).tokens()?;
                let mut p = Parser::new(toks);
                let e = p.expression()?;
                parts.push(TplPart::Expr(Box::new(e)));
                i = j + 1;
            } else {
                let ch = raw[i..].chars().next().unwrap();
                lit.push(ch);
                i += ch.len_utf8();
            }
        }
        if !lit.is_empty() {
            parts.push(TplPart::Str(lit));
        }
        Ok(Expr::Template(parts))
    }
}

// ════════════════════════════ values ════════════════════════════

type Obj = Rc<RefCell<BTreeMap<String, Value>>>;
type Arr = Rc<RefCell<Vec<Value>>>;

#[derive(Clone)]
enum Value {
    Undefined,
    Null,
    Bool(bool),
    Num(f64),
    Str(Rc<String>),
    Array(Arr),
    Object(Obj),
    Function(Rc<Closure>),
    Native(Rc<NativeFn>),
    Node(NodeRef),
    Style(NodeRef),
    Bound(Box<Value>, String),
}

struct Closure {
    def: Rc<FnDef>,
    env: Env,
    this: RefCell<Value>,
    is_arrow: bool,
}

struct NativeFn {
    name: String,
}

impl Value {
    fn str(s: impl Into<String>) -> Value {
        Value::Str(Rc::new(s.into()))
    }

    fn truthy(&self) -> bool {
        match self {
            Value::Undefined | Value::Null => false,
            Value::Bool(b) => *b,
            Value::Num(n) => *n != 0.0 && !n.is_nan(),
            Value::Str(s) => !s.is_empty(),
            _ => true,
        }
    }

    fn to_number(&self) -> f64 {
        match self {
            Value::Num(n) => *n,
            Value::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Value::Str(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            Value::Null => 0.0,
            _ => f64::NAN,
        }
    }

    fn type_of(&self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            Value::Null => "object",
            Value::Bool(_) => "boolean",
            Value::Num(_) => "number",
            Value::Str(_) => "string",
            Value::Function(_) | Value::Native(_) | Value::Bound(_, _) => "function",
            _ => "object",
        }
    }

    fn to_display(&self) -> String {
        match self {
            Value::Undefined => "undefined".to_string(),
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Num(n) => fmt_num(*n),
            Value::Str(s) => (**s).clone(),
            Value::Array(a) => {
                let items: Vec<String> = a.borrow().iter().map(|v| v.to_display()).collect();
                items.join(",")
            }
            Value::Object(_) => "[object Object]".to_string(),
            Value::Function(_) | Value::Native(_) | Value::Bound(_, _) => "function".to_string(),
            Value::Node(_) | Value::Style(_) => "[object HTMLElement]".to_string(),
        }
    }
}

fn fmt_num(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n.is_infinite() {
        return if n < 0.0 { "-Infinity".to_string() } else { "Infinity".to_string() };
    }
    if n == (n as i64) as f64 && n.abs() < 1e15 {
        return (n as i64).to_string();
    }
    let mut s = alloc::format!("{}", n);
    if s.contains('e') {
        return s;
    }
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

// ════════════════════════════ environment ════════════════════════════

type Env = Rc<RefCell<Scope>>;

struct Scope {
    vars: BTreeMap<String, Value>,
    parent: Option<Env>,
}

impl Scope {
    fn child(parent: &Env) -> Env {
        Rc::new(RefCell::new(Scope { vars: BTreeMap::new(), parent: Some(parent.clone()) }))
    }
    fn root() -> Env {
        Rc::new(RefCell::new(Scope { vars: BTreeMap::new(), parent: None }))
    }
}

fn env_get(env: &Env, name: &str) -> Option<Value> {
    let s = env.borrow();
    if let Some(v) = s.vars.get(name) {
        return Some(v.clone());
    }
    if let Some(p) = &s.parent {
        return env_get(p, name);
    }
    None
}

fn env_set(env: &Env, name: &str, val: Value) -> bool {
    {
        let mut s = env.borrow_mut();
        if s.vars.contains_key(name) {
            s.vars.insert(name.to_string(), val);
            return true;
        }
    }
    let parent = env.borrow().parent.clone();
    if let Some(p) = parent {
        return env_set(&p, name, val);
    }
    false
}

fn env_define(env: &Env, name: &str, val: Value) {
    env.borrow_mut().vars.insert(name.to_string(), val);
}

// ════════════════════════════ interpreter ════════════════════════════

enum Flow {
    // Statement completed normally; JS statement-completion values are unobservable
    // here, so the variant carries no payload.
    Normal,
    Return(Value),
    Break,
    Continue,
}

type EvalResult = Result<Value, String>;

/// The JS engine bound to a document.
pub struct Js {
    global: Env,
    document: NodeRef,
    console: Vec<String>,
    handlers: Vec<(NodeRef, String, Value)>,
    rng: u64,
    steps: u64,
    /// Current JS call-stack depth. Checked in [`Js::call_closure`] against
    /// [`MAX_CALL_DEPTH`] to prevent kernel stack overflows from deeply
    /// recursive or mutually-recursive JS functions.
    call_depth: usize,
}

const STEP_LIMIT: u64 = 5_000_000;

/// Maximum JS call-stack depth. Exceeding this returns a `JsError::StackOverflow`
/// rather than blowing the kernel stack via unbounded Rust recursion.
const MAX_CALL_DEPTH: usize = 512;

impl Js {
    pub fn new(document: NodeRef) -> Js {
        let global = Scope::root();
        let mut js = Js { global, document, console: Vec::new(), handlers: Vec::new(), rng: 0x2545F4914F6CDD1D, steps: 0, call_depth: 0 };
        js.install_globals();
        js
    }

    pub fn console(&self) -> &[String] {
        &self.console
    }

    /// Whether `node` has a click handler — for cursor/hit-testing.
    pub fn has_click_handler(&self, node: &NodeRef) -> bool {
        self.handlers.iter().any(|(n, t, _)| t == "click" && Rc::ptr_eq(n, node))
            || dom::get_attr(node, "onclick").is_some()
    }

    pub fn run(&mut self, src: &str) -> Result<(), String> {
        let r = self.run_inner(src);
        if let Err(e) = &r {
            self.console.push(alloc::format!("Uncaught {}", e));
        }
        r
    }

    fn run_inner(&mut self, src: &str) -> Result<(), String> {
        let toks = Lexer::new(src).tokens()?;
        let prog = Parser::new(toks).program()?;
        let env = self.global.clone();
        for s in &prog {
            if let Stmt::FnDecl(name, def) = s {
                let f = Value::Function(Rc::new(Closure {
                    def: def.clone(),
                    env: env.clone(),
                    this: RefCell::new(Value::Undefined),
                    is_arrow: false,
                }));
                env_define(&env, name, f);
            }
        }
        for s in &prog {
            if matches!(s, Stmt::FnDecl(_, _)) {
                continue;
            }
            match self.exec(s, &env)? {
                Flow::Normal => {}
                _ => break,
            }
        }
        Ok(())
    }

    /// Dispatch a DOM event: runs registered listeners and any inline `on<type>`
    /// attribute. Returns whether a handler ran (so the browser re-renders).
    pub fn fire_event(&mut self, node: &NodeRef, event: &str) -> bool {
        let mut ran = false;
        let matching: Vec<Value> = self
            .handlers
            .iter()
            .filter(|(n, t, _)| t == event && Rc::ptr_eq(n, node))
            .map(|(_, _, f)| f.clone())
            .collect();
        for f in matching {
            let _ = self.call_value(f, alloc::vec![Value::Node(node.clone())], Value::Node(node.clone()));
            ran = true;
        }
        let attr = alloc::format!("on{}", event);
        if let Some(src) = dom::get_attr(node, &attr) {
            let _ = self.run(&src);
            ran = true;
        }
        ran
    }

    fn tick(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > STEP_LIMIT {
            Err("script exceeded execution step limit".to_string())
        } else {
            Ok(())
        }
    }

    fn exec(&mut self, s: &Stmt, env: &Env) -> Result<Flow, String> {
        self.tick()?;
        match s {
            Stmt::Empty => Ok(Flow::Normal),
            Stmt::VarDecl(decls) => {
                for (name, init) in decls {
                    let v = match init {
                        Some(e) => self.eval(e, env)?,
                        None => Value::Undefined,
                    };
                    env_define(env, name, v);
                }
                Ok(Flow::Normal)
            }
            Stmt::FnDecl(name, def) => {
                let f = Value::Function(Rc::new(Closure {
                    def: def.clone(),
                    env: env.clone(),
                    this: RefCell::new(Value::Undefined),
                    is_arrow: false,
                }));
                env_define(env, name, f);
                Ok(Flow::Normal)
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Undefined,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Expr(e) => {
                // Evaluate for side effects; the completion value is discarded.
                self.eval(e, env)?;
                Ok(Flow::Normal)
            }
            Stmt::Block(stmts) => {
                let scope = Scope::child(env);
                self.exec_block(stmts, &scope)
            }
            Stmt::If(cond, then, els) => {
                if self.eval(cond, env)?.truthy() {
                    self.exec(then, env)
                } else if let Some(e) = els {
                    self.exec(e, env)
                } else {
                    Ok(Flow::Normal)
                }
            }
            Stmt::While(cond, body) => {
                while self.eval(cond, env)?.truthy() {
                    self.tick()?;
                    match self.exec(body, env)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::For(init, cond, update, body) => {
                let scope = Scope::child(env);
                if let Some(i) = init {
                    self.exec(i, &scope)?;
                }
                loop {
                    self.tick()?;
                    if let Some(c) = cond {
                        if !self.eval(c, &scope)?.truthy() {
                            break;
                        }
                    }
                    match self.exec(body, &scope)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                    if let Some(u) = update {
                        self.eval(u, &scope)?;
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::ForOf(name, iter, body) => {
                let src = self.eval(iter, env)?;
                let items = self.iterable(src)?;
                for item in items {
                    let scope = Scope::child(env);
                    env_define(&scope, name, item);
                    match self.exec(body, &scope)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
        }
    }

    fn exec_block(&mut self, stmts: &[Stmt], env: &Env) -> Result<Flow, String> {
        for s in stmts {
            if let Stmt::FnDecl(name, def) = s {
                let f = Value::Function(Rc::new(Closure {
                    def: def.clone(),
                    env: env.clone(),
                    this: RefCell::new(Value::Undefined),
                    is_arrow: false,
                }));
                env_define(env, name, f);
            }
        }
        for s in stmts {
            if matches!(s, Stmt::FnDecl(_, _)) {
                continue;
            }
            match self.exec(s, env)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    fn iterable(&self, v: Value) -> Result<Vec<Value>, String> {
        match v {
            Value::Array(a) => Ok(a.borrow().clone()),
            Value::Str(s) => Ok(s.chars().map(|c| Value::str(c.to_string())).collect()),
            other => Err(alloc::format!("{} is not iterable", other.type_of())),
        }
    }

    fn eval(&mut self, e: &Expr, env: &Env) -> EvalResult {
        self.tick()?;
        match e {
            Expr::Num(n) => Ok(Value::Num(*n)),
            Expr::Str(s) => Ok(Value::str(s.clone())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Null => Ok(Value::Null),
            Expr::Undefined => Ok(Value::Undefined),
            Expr::This => Ok(env_get(env, "this").unwrap_or(Value::Undefined)),
            Expr::Ident(name) => env_get(env, name).ok_or_else(|| alloc::format!("{} is not defined", name)),
            Expr::Template(parts) => {
                let mut out = String::new();
                for p in parts {
                    match p {
                        TplPart::Str(s) => out.push_str(s),
                        TplPart::Expr(e) => out.push_str(&self.eval(e, env)?.to_display()),
                    }
                }
                Ok(Value::str(out))
            }
            Expr::Array(items) => {
                let mut v = Vec::new();
                for it in items {
                    v.push(self.eval(it, env)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            Expr::Object(props) => {
                let mut map = BTreeMap::new();
                for (k, ve) in props {
                    let val = self.eval(ve, env)?;
                    map.insert(k.clone(), val);
                }
                Ok(Value::Object(Rc::new(RefCell::new(map))))
            }
            Expr::Function(def) => Ok(Value::Function(Rc::new(Closure {
                def: def.clone(),
                env: env.clone(),
                this: RefCell::new(Value::Undefined),
                is_arrow: false,
            }))),
            Expr::Arrow(def) => Ok(Value::Function(Rc::new(Closure {
                def: def.clone(),
                env: env.clone(),
                this: RefCell::new(env_get(env, "this").unwrap_or(Value::Undefined)),
                is_arrow: true,
            }))),
            Expr::Unary(op, e) => self.eval_unary(op, e, env),
            Expr::Update(op, prefix, target) => self.eval_update(op, *prefix, target, env),
            Expr::Binary(op, l, r) => {
                let lv = self.eval(l, env)?;
                let rv = self.eval(r, env)?;
                eval_binary(op, lv, rv)
            }
            Expr::Logical(op, l, r) => {
                let lv = self.eval(l, env)?;
                match op.as_str() {
                    "&&" => {
                        if lv.truthy() {
                            self.eval(r, env)
                        } else {
                            Ok(lv)
                        }
                    }
                    _ => {
                        if lv.truthy() {
                            Ok(lv)
                        } else {
                            self.eval(r, env)
                        }
                    }
                }
            }
            Expr::Ternary(c, t, f) => {
                if self.eval(c, env)?.truthy() {
                    self.eval(t, env)
                } else {
                    self.eval(f, env)
                }
            }
            Expr::Assign(op, target, value) => self.eval_assign(op, target, value, env),
            Expr::Member(obj, name) => {
                let o = self.eval(obj, env)?;
                self.get_member(&o, name)
            }
            Expr::Index(obj, idx) => {
                let o = self.eval(obj, env)?;
                let i = self.eval(idx, env)?;
                self.get_index(&o, &i)
            }
            Expr::Call(callee, args) => self.eval_call(callee, args, env),
            Expr::New(callee, args) => self.eval_new(callee, args, env),
        }
    }

    fn eval_unary(&mut self, op: &str, e: &Expr, env: &Env) -> EvalResult {
        if op == "typeof" {
            if let Expr::Ident(name) = e {
                if env_get(env, name).is_none() {
                    return Ok(Value::str("undefined"));
                }
            }
            let v = self.eval(e, env)?;
            return Ok(Value::str(v.type_of()));
        }
        let v = self.eval(e, env)?;
        Ok(match op {
            "!" => Value::Bool(!v.truthy()),
            "-" => Value::Num(-v.to_number()),
            "+" => Value::Num(v.to_number()),
            "void" => Value::Undefined,
            "delete" => Value::Bool(true),
            _ => Value::Undefined,
        })
    }

    fn eval_update(&mut self, op: &str, prefix: bool, target: &Expr, env: &Env) -> EvalResult {
        let old = self.eval(target, env)?.to_number();
        let new = if op == "++" { old + 1.0 } else { old - 1.0 };
        self.assign_to(target, Value::Num(new), env)?;
        Ok(Value::Num(if prefix { new } else { old }))
    }

    fn eval_assign(&mut self, op: &str, target: &Expr, value: &Expr, env: &Env) -> EvalResult {
        let rhs = self.eval(value, env)?;
        let final_val = if op == "=" {
            rhs
        } else {
            let cur = self.eval(target, env)?;
            let bop = &op[..1];
            eval_binary(bop, cur, rhs)?
        };
        self.assign_to(target, final_val.clone(), env)?;
        Ok(final_val)
    }

    fn assign_to(&mut self, target: &Expr, val: Value, env: &Env) -> Result<(), String> {
        match target {
            Expr::Ident(name) => {
                if !env_set(env, name, val.clone()) {
                    env_define(&self.global, name, val);
                }
                Ok(())
            }
            Expr::Member(obj, name) => {
                let o = self.eval(obj, env)?;
                self.set_member(&o, name, val)
            }
            Expr::Index(obj, idx) => {
                let o = self.eval(obj, env)?;
                let i = self.eval(idx, env)?;
                self.set_index(&o, &i, val)
            }
            _ => Err("invalid assignment target".to_string()),
        }
    }

    fn get_member(&mut self, o: &Value, name: &str) -> EvalResult {
        match o {
            Value::Object(map) => Ok(map.borrow().get(name).cloned().unwrap_or(Value::Undefined)),
            Value::Array(a) => {
                if name == "length" {
                    Ok(Value::Num(a.borrow().len() as f64))
                } else {
                    Ok(Value::Bound(Box::new(o.clone()), name.to_string()))
                }
            }
            Value::Str(s) => {
                if name == "length" {
                    Ok(Value::Num(s.chars().count() as f64))
                } else {
                    Ok(Value::Bound(Box::new(o.clone()), name.to_string()))
                }
            }
            Value::Node(n) => self.node_get(n, name),
            Value::Style(n) => Ok(Value::str(style_get(n, name))),
            Value::Undefined | Value::Null => Err(alloc::format!("cannot read '{}' of {}", name, o.type_of())),
            _ => Ok(Value::Bound(Box::new(o.clone()), name.to_string())),
        }
    }

    fn get_index(&mut self, o: &Value, idx: &Value) -> EvalResult {
        match o {
            Value::Array(a) => {
                let i = idx.to_number();
                if i.is_finite() && i >= 0.0 {
                    Ok(a.borrow().get(i as usize).cloned().unwrap_or(Value::Undefined))
                } else {
                    Ok(Value::Undefined)
                }
            }
            Value::Object(map) => Ok(map.borrow().get(&idx.to_display()).cloned().unwrap_or(Value::Undefined)),
            Value::Str(s) => {
                let i = idx.to_number();
                if i.is_finite() && i >= 0.0 {
                    Ok(s.chars().nth(i as usize).map(|c| Value::str(c.to_string())).unwrap_or(Value::Undefined))
                } else {
                    Ok(Value::Undefined)
                }
            }
            _ => self.get_member(o, &idx.to_display()),
        }
    }

    fn set_member(&mut self, o: &Value, name: &str, val: Value) -> Result<(), String> {
        match o {
            Value::Object(map) => {
                map.borrow_mut().insert(name.to_string(), val);
                Ok(())
            }
            Value::Node(n) => self.node_set(n, name, val),
            Value::Style(n) => {
                style_set(n, name, &val.to_display());
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn set_index(&mut self, o: &Value, idx: &Value, val: Value) -> Result<(), String> {
        match o {
            Value::Array(a) => {
                let i = idx.to_number();
                if i.is_finite() && i >= 0.0 {
                    let i = i as usize;
                    let mut v = a.borrow_mut();
                    if i >= v.len() {
                        v.resize(i + 1, Value::Undefined);
                    }
                    v[i] = val;
                }
                Ok(())
            }
            Value::Object(map) => {
                map.borrow_mut().insert(idx.to_display(), val);
                Ok(())
            }
            _ => self.set_member(o, &idx.to_display(), val),
        }
    }

    fn eval_call(&mut self, callee: &Expr, args: &[Expr], env: &Env) -> EvalResult {
        let argv = self.eval_args(args, env)?;
        match callee {
            Expr::Member(obj, name) => {
                let o = self.eval(obj, env)?;
                self.call_method(o, name, argv)
            }
            Expr::Index(obj, key) => {
                let o = self.eval(obj, env)?;
                let k = self.eval(key, env)?.to_display();
                self.call_method(o, &k, argv)
            }
            _ => {
                let f = self.eval(callee, env)?;
                self.call_value(f, argv, Value::Undefined)
            }
        }
    }

    fn eval_args(&mut self, args: &[Expr], env: &Env) -> Result<Vec<Value>, String> {
        let mut v = Vec::with_capacity(args.len());
        for a in args {
            v.push(self.eval(a, env)?);
        }
        Ok(v)
    }

    fn eval_new(&mut self, callee: &Expr, args: &[Expr], env: &Env) -> EvalResult {
        let argv = self.eval_args(args, env)?;
        if let Expr::Ident(name) = callee {
            match name.as_str() {
                "Array" => return Ok(Value::Array(Rc::new(RefCell::new(argv)))),
                "Object" => return Ok(Value::Object(Rc::new(RefCell::new(BTreeMap::new())))),
                _ => {}
            }
        }
        let f = self.eval(callee, env)?;
        let this = Value::Object(Rc::new(RefCell::new(BTreeMap::new())));
        let r = self.call_value(f, argv, this.clone())?;
        Ok(match r {
            Value::Object(_) | Value::Array(_) | Value::Node(_) => r,
            _ => this,
        })
    }

    fn call_value(&mut self, f: Value, args: Vec<Value>, this: Value) -> EvalResult {
        match f {
            Value::Function(clo) => self.call_closure(&clo, args, this),
            Value::Native(nf) => self.call_native(&nf.name, args),
            Value::Bound(recv, method) => self.call_method(*recv, &method, args),
            other => Err(alloc::format!("{} is not a function", other.type_of())),
        }
    }

    fn call_closure(&mut self, clo: &Closure, args: Vec<Value>, this: Value) -> EvalResult {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err("Maximum call stack size exceeded".to_string());
        }
        self.tick()?;
        self.call_depth += 1;
        let scope = Scope::child(&clo.env);
        for (i, p) in clo.def.params.iter().enumerate() {
            env_define(&scope, p, args.get(i).cloned().unwrap_or(Value::Undefined));
        }
        env_define(&scope, "arguments", Value::Array(Rc::new(RefCell::new(args))));
        let this_val = if clo.is_arrow { clo.this.borrow().clone() } else { this };
        env_define(&scope, "this", this_val);
        let result = self.exec_block(&clo.def.body, &scope);
        self.call_depth -= 1;
        match result? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Value::Undefined),
        }
    }

    fn call_method(&mut self, recv: Value, method: &str, args: Vec<Value>) -> EvalResult {
        match &recv {
            Value::Str(s) => return string_method(s, method, &args),
            Value::Array(a) => return self.array_method(a.clone(), method, args),
            Value::Node(n) => return self.node_method(n.clone(), method, args),
            Value::Object(map) => {
                let prop = map.borrow().get(method).cloned();
                if let Some(f @ (Value::Function(_) | Value::Native(_))) = prop {
                    return self.call_value(f, args, recv.clone());
                }
            }
            _ => {}
        }
        let full = match &recv {
            Value::Native(nf) => alloc::format!("{}.{}", nf.name, method),
            _ => method.to_string(),
        };
        self.call_native(&full, args)
    }

    fn install_globals(&mut self) {
        let g = self.global.clone();
        let native = |name: &str| Value::Native(Rc::new(NativeFn { name: name.to_string() }));

        let console = BTreeMap::from([
            ("log".to_string(), native("console.log")),
            ("error".to_string(), native("console.error")),
            ("warn".to_string(), native("console.warn")),
            ("info".to_string(), native("console.log")),
        ]);
        env_define(&g, "console", Value::Object(Rc::new(RefCell::new(console))));

        let math = BTreeMap::from([
            ("floor".to_string(), native("Math.floor")),
            ("ceil".to_string(), native("Math.ceil")),
            ("round".to_string(), native("Math.round")),
            ("abs".to_string(), native("Math.abs")),
            ("max".to_string(), native("Math.max")),
            ("min".to_string(), native("Math.min")),
            ("sqrt".to_string(), native("Math.sqrt")),
            ("pow".to_string(), native("Math.pow")),
            ("random".to_string(), native("Math.random")),
            ("PI".to_string(), Value::Num(core::f64::consts::PI)),
            ("E".to_string(), Value::Num(core::f64::consts::E)),
        ]);
        env_define(&g, "Math", Value::Object(Rc::new(RefCell::new(math))));

        let json = BTreeMap::from([
            ("stringify".to_string(), native("JSON.stringify")),
            ("parse".to_string(), native("JSON.parse")),
        ]);
        env_define(&g, "JSON", Value::Object(Rc::new(RefCell::new(json))));

        let doc = BTreeMap::from([
            ("getElementById".to_string(), native("document.getElementById")),
            ("querySelector".to_string(), native("document.querySelector")),
            ("querySelectorAll".to_string(), native("document.querySelectorAll")),
            ("getElementsByTagName".to_string(), native("document.getElementsByTagName")),
            ("getElementsByClassName".to_string(), native("document.getElementsByClassName")),
            ("createElement".to_string(), native("document.createElement")),
            ("createTextNode".to_string(), native("document.createTextNode")),
        ]);
        env_define(&g, "document", Value::Object(Rc::new(RefCell::new(doc))));

        for f in ["parseInt", "parseFloat", "isNaN", "String", "Number", "Boolean", "alert", "Array", "Object"] {
            env_define(&g, f, native(f));
        }
    }

    fn call_native(&mut self, name: &str, args: Vec<Value>) -> EvalResult {
        let arg = |i: usize| args.get(i).cloned().unwrap_or(Value::Undefined);
        match name {
            "console.log" | "console.error" | "console.warn" => {
                let line: Vec<String> = args.iter().map(|v| v.to_display()).collect();
                self.console.push(line.join(" "));
                Ok(Value::Undefined)
            }
            "Math.floor" => Ok(Value::Num(crate::datatypes::floor(arg(0).to_number()))),
            "Math.ceil" => Ok(Value::Num(crate::datatypes::ceil(arg(0).to_number()))),
            "Math.round" => Ok(Value::Num(crate::datatypes::floor(arg(0).to_number() + 0.5))),
            "Math.abs" => Ok(Value::Num(arg(0).to_number().abs())),
            "Math.sqrt" => Ok(Value::Num(crate::datatypes::sqrt(arg(0).to_number()))),
            "Math.pow" => Ok(Value::Num(powf(arg(0).to_number(), arg(1).to_number()))),
            "Math.max" => Ok(Value::Num(args.iter().map(|v| v.to_number()).fold(f64::NEG_INFINITY, f64::max))),
            "Math.min" => Ok(Value::Num(args.iter().map(|v| v.to_number()).fold(f64::INFINITY, f64::min))),
            "Math.random" => Ok(Value::Num(self.random())),
            "JSON.stringify" => Ok(Value::str(json_stringify(&arg(0)))),
            "JSON.parse" => json_parse(&arg(0).to_display()),
            "parseInt" => {
                let s = arg(0).to_display();
                let n = s.trim().split('.').next().unwrap_or("").trim().parse::<i64>().ok();
                Ok(n.map(|v| Value::Num(v as f64)).unwrap_or(Value::Num(f64::NAN)))
            }
            "parseFloat" => Ok(Value::Num(arg(0).to_display().trim().parse::<f64>().unwrap_or(f64::NAN))),
            "isNaN" => Ok(Value::Bool(arg(0).to_number().is_nan())),
            "String" => Ok(Value::str(arg(0).to_display())),
            "Number" => Ok(Value::Num(arg(0).to_number())),
            "Boolean" => Ok(Value::Bool(arg(0).truthy())),
            "alert" => {
                self.console.push(alloc::format!("[alert] {}", arg(0).to_display()));
                Ok(Value::Undefined)
            }
            "Array" => Ok(Value::Array(Rc::new(RefCell::new(args)))),
            "Object" => Ok(Value::Object(Rc::new(RefCell::new(BTreeMap::new())))),
            "document.getElementById" => {
                Ok(dom::get_element_by_id(&self.document, &arg(0).to_display()).map(Value::Node).unwrap_or(Value::Null))
            }
            "document.querySelector" => {
                Ok(dom::query_selector(&self.document, &arg(0).to_display()).map(Value::Node).unwrap_or(Value::Null))
            }
            "document.querySelectorAll" => Ok(node_list(dom::query_selector_all(&self.document, &arg(0).to_display()))),
            "document.getElementsByTagName" => Ok(node_list(dom::get_elements_by_tag(&self.document, &arg(0).to_display()))),
            "document.getElementsByClassName" => Ok(node_list(dom::get_elements_by_class(&self.document, &arg(0).to_display()))),
            "document.createElement" => Ok(Value::Node(dom::Node::element(&arg(0).to_display()))),
            "document.createTextNode" => Ok(Value::Node(dom::Node::text(&arg(0).to_display()))),
            other => Err(alloc::format!("{} is not a function", other)),
        }
    }

    fn random(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }

    fn array_method(&mut self, a: Arr, method: &str, args: Vec<Value>) -> EvalResult {
        let arg = |i: usize| args.get(i).cloned().unwrap_or(Value::Undefined);
        match method {
            "push" => {
                for v in args {
                    a.borrow_mut().push(v);
                }
                Ok(Value::Num(a.borrow().len() as f64))
            }
            "pop" => Ok(a.borrow_mut().pop().unwrap_or(Value::Undefined)),
            "shift" => {
                let mut b = a.borrow_mut();
                if b.is_empty() {
                    Ok(Value::Undefined)
                } else {
                    Ok(b.remove(0))
                }
            }
            "unshift" => {
                let mut b = a.borrow_mut();
                for (i, v) in args.into_iter().enumerate() {
                    b.insert(i, v);
                }
                Ok(Value::Num(b.len() as f64))
            }
            "join" => {
                let sep = if args.is_empty() { ",".to_string() } else { arg(0).to_display() };
                let parts: Vec<String> = a.borrow().iter().map(|v| v.to_display()).collect();
                Ok(Value::str(parts.join(&sep)))
            }
            "indexOf" => {
                let target = arg(0);
                let pos = a.borrow().iter().position(|v| values_equal(v, &target));
                Ok(Value::Num(pos.map(|p| p as f64).unwrap_or(-1.0)))
            }
            "includes" => {
                let target = arg(0);
                Ok(Value::Bool(a.borrow().iter().any(|v| values_equal(v, &target))))
            }
            "slice" => {
                let b = a.borrow();
                let len = b.len() as i64;
                let start = norm_index(arg(0), len, 0);
                let end = if args.len() > 1 { norm_index(arg(1), len, len) } else { len };
                let mut out = Vec::new();
                let mut i = start;
                while i < end {
                    out.push(b[i as usize].clone());
                    i += 1;
                }
                Ok(Value::Array(Rc::new(RefCell::new(out))))
            }
            "map" => {
                let f = arg(0);
                let items = a.borrow().clone();
                let mut out = Vec::with_capacity(items.len());
                for (i, it) in items.into_iter().enumerate() {
                    out.push(self.call_value(f.clone(), alloc::vec![it, Value::Num(i as f64)], Value::Undefined)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(out))))
            }
            "filter" => {
                let f = arg(0);
                let items = a.borrow().clone();
                let mut out = Vec::new();
                for (i, it) in items.into_iter().enumerate() {
                    if self.call_value(f.clone(), alloc::vec![it.clone(), Value::Num(i as f64)], Value::Undefined)?.truthy() {
                        out.push(it);
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(out))))
            }
            "forEach" => {
                let f = arg(0);
                let items = a.borrow().clone();
                for (i, it) in items.into_iter().enumerate() {
                    self.call_value(f.clone(), alloc::vec![it, Value::Num(i as f64)], Value::Undefined)?;
                }
                Ok(Value::Undefined)
            }
            "reduce" => {
                let f = arg(0);
                let items = a.borrow().clone();
                let mut acc;
                let mut start = 0;
                if args.len() > 1 {
                    acc = arg(1);
                } else if !items.is_empty() {
                    acc = items[0].clone();
                    start = 1;
                } else {
                    return Err("reduce of empty array with no initial value".to_string());
                }
                for it in items.into_iter().skip(start) {
                    acc = self.call_value(f.clone(), alloc::vec![acc, it], Value::Undefined)?;
                }
                Ok(acc)
            }
            "find" => {
                let f = arg(0);
                let items = a.borrow().clone();
                for it in items {
                    if self.call_value(f.clone(), alloc::vec![it.clone()], Value::Undefined)?.truthy() {
                        return Ok(it);
                    }
                }
                Ok(Value::Undefined)
            }
            "some" => {
                let f = arg(0);
                let items = a.borrow().clone();
                for it in items {
                    if self.call_value(f.clone(), alloc::vec![it], Value::Undefined)?.truthy() {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            "every" => {
                let f = arg(0);
                let items = a.borrow().clone();
                for it in items {
                    if !self.call_value(f.clone(), alloc::vec![it], Value::Undefined)?.truthy() {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            "reverse" => {
                a.borrow_mut().reverse();
                Ok(Value::Array(a))
            }
            "concat" => {
                let mut out = a.borrow().clone();
                for v in args {
                    match v {
                        Value::Array(b) => out.extend(b.borrow().iter().cloned()),
                        other => out.push(other),
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(out))))
            }
            _ => Err(alloc::format!("array has no method '{}'", method)),
        }
    }

    fn node_get(&mut self, n: &NodeRef, name: &str) -> EvalResult {
        Ok(match name {
            "textContent" | "innerText" => Value::str(dom::text_content(n)),
            "innerHTML" => Value::str(dom::inner_html(n)),
            "id" => Value::str(dom::id(n).unwrap_or_default()),
            "className" => Value::str(dom::get_attr(n, "class").unwrap_or_default()),
            "tagName" => Value::str(dom::tag(n).unwrap_or_default().to_ascii_uppercase()),
            "value" => Value::str(dom::get_attr(n, "value").unwrap_or_default()),
            "href" => Value::str(dom::get_attr(n, "href").unwrap_or_default()),
            "style" => Value::Style(n.clone()),
            "children" | "childNodes" => {
                let kids = n.borrow().as_element().map(|e| e.children.clone()).unwrap_or_default();
                node_list(kids.into_iter().filter(|c| c.borrow().is_element()).collect())
            }
            "parentNode" | "parentElement" => dom::parent(n).map(Value::Node).unwrap_or(Value::Null),
            "checked" => Value::Bool(dom::get_attr(n, "checked").is_some()),
            "getAttribute" | "setAttribute" | "appendChild" | "removeChild" | "addEventListener"
            | "removeAttribute" | "querySelector" | "querySelectorAll" | "hasAttribute"
            | "getElementsByTagName" | "getElementsByClassName" | "remove" | "click" => {
                Value::Bound(Box::new(Value::Node(n.clone())), name.to_string())
            }
            _ => Value::Undefined,
        })
    }

    fn node_set(&mut self, n: &NodeRef, name: &str, val: Value) -> Result<(), String> {
        match name {
            "textContent" | "innerText" => dom::set_text_content(n, &val.to_display()),
            "innerHTML" => dom::set_inner_html(n, &val.to_display()),
            "id" => dom::set_attr(n, "id", &val.to_display()),
            "className" => dom::set_attr(n, "class", &val.to_display()),
            "value" => dom::set_attr(n, "value", &val.to_display()),
            "href" => dom::set_attr(n, "href", &val.to_display()),
            _ if name.starts_with("on") && matches!(val, Value::Function(_) | Value::Native(_)) => {
                let event = name[2..].to_string();
                register_listener(n, &event);
                self.handlers.push((n.clone(), event, val));
            }
            _ => dom::set_attr(n, name, &val.to_display()),
        }
        Ok(())
    }

    fn node_method(&mut self, n: NodeRef, method: &str, args: Vec<Value>) -> EvalResult {
        let arg = |i: usize| args.get(i).cloned().unwrap_or(Value::Undefined);
        match method {
            "getAttribute" => Ok(dom::get_attr(&n, &arg(0).to_display()).map(Value::str).unwrap_or(Value::Null)),
            "setAttribute" => {
                dom::set_attr(&n, &arg(0).to_display(), &arg(1).to_display());
                Ok(Value::Undefined)
            }
            "hasAttribute" => Ok(Value::Bool(dom::get_attr(&n, &arg(0).to_display()).is_some())),
            "removeAttribute" => {
                if let Some(e) = n.borrow_mut().as_element_mut() {
                    let key = arg(0).to_display().to_ascii_lowercase();
                    e.attrs.retain(|(k, _)| *k != key);
                }
                Ok(Value::Undefined)
            }
            "appendChild" => {
                if let Value::Node(child) = arg(0) {
                    dom::append_child(&n, &child);
                    Ok(Value::Node(child))
                } else {
                    Ok(Value::Undefined)
                }
            }
            "addEventListener" => {
                let event = arg(0).to_display();
                let handler = arg(1);
                register_listener(&n, &event);
                self.handlers.push((n.clone(), event, handler));
                Ok(Value::Undefined)
            }
            "querySelector" => Ok(dom::query_selector(&n, &arg(0).to_display()).map(Value::Node).unwrap_or(Value::Null)),
            "querySelectorAll" => Ok(node_list(dom::query_selector_all(&n, &arg(0).to_display()))),
            "getElementsByTagName" => Ok(node_list(dom::get_elements_by_tag(&n, &arg(0).to_display()))),
            "getElementsByClassName" => Ok(node_list(dom::get_elements_by_class(&n, &arg(0).to_display()))),
            "click" => {
                self.fire_event(&n, "click");
                Ok(Value::Undefined)
            }
            _ => Err(alloc::format!("element has no method '{}'", method)),
        }
    }
}

fn register_listener(n: &NodeRef, event: &str) {
    if let Some(e) = n.borrow_mut().as_element_mut() {
        if !e.listeners.iter().any(|l| l == event) {
            e.listeners.push(event.to_string());
        }
    }
}

fn node_list(nodes: Vec<NodeRef>) -> Value {
    Value::Array(Rc::new(RefCell::new(nodes.into_iter().map(Value::Node).collect())))
}

fn style_get(n: &NodeRef, prop: &str) -> String {
    let css_prop = camel_to_kebab(prop);
    if let Some(style) = dom::get_attr(n, "style") {
        for decl in style.split(';') {
            if let Some(colon) = decl.find(':') {
                if decl[..colon].trim().eq_ignore_ascii_case(&css_prop) {
                    return decl[colon + 1..].trim().to_string();
                }
            }
        }
    }
    String::new()
}

fn style_set(n: &NodeRef, prop: &str, value: &str) {
    let css_prop = camel_to_kebab(prop);
    let mut decls: Vec<(String, String)> = Vec::new();
    if let Some(style) = dom::get_attr(n, "style") {
        for decl in style.split(';') {
            if let Some(colon) = decl.find(':') {
                decls.push((decl[..colon].trim().to_string(), decl[colon + 1..].trim().to_string()));
            }
        }
    }
    if let Some(slot) = decls.iter_mut().find(|(k, _)| k.eq_ignore_ascii_case(&css_prop)) {
        slot.1 = value.to_string();
    } else {
        decls.push((css_prop, value.to_string()));
    }
    let mut out = String::new();
    for (k, v) in decls {
        out.push_str(&k);
        out.push(':');
        out.push_str(&v);
        out.push(';');
    }
    dom::set_attr(n, "style", &out);
}

fn camel_to_kebab(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn string_method(s: &str, method: &str, args: &[Value]) -> EvalResult {
    let arg = |i: usize| args.get(i).cloned().unwrap_or(Value::Undefined);
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len() as i64;
    Ok(match method {
        "toUpperCase" => Value::str(s.to_uppercase()),
        "toLowerCase" => Value::str(s.to_lowercase()),
        "trim" => Value::str(s.trim().to_string()),
        "charAt" => {
            let i = arg(0).to_number() as i64;
            Value::str(if i >= 0 && i < len { chars[i as usize].to_string() } else { String::new() })
        }
        "charCodeAt" => {
            let i = arg(0).to_number() as i64;
            if i >= 0 && i < len {
                Value::Num(chars[i as usize] as u32 as f64)
            } else {
                Value::Num(f64::NAN)
            }
        }
        "indexOf" => {
            let needle = arg(0).to_display();
            Value::Num(s.find(&needle).map(|b| s[..b].chars().count() as f64).unwrap_or(-1.0))
        }
        "includes" => Value::Bool(s.contains(&arg(0).to_display())),
        "startsWith" => Value::Bool(s.starts_with(&arg(0).to_display())),
        "endsWith" => Value::Bool(s.ends_with(&arg(0).to_display())),
        "slice" | "substring" => {
            let start = norm_index(arg(0), len, 0);
            let end = if args.len() > 1 { norm_index(arg(1), len, len) } else { len };
            let (a, b) = if method == "substring" && start > end { (end, start) } else { (start, end) };
            Value::str(chars[a.max(0) as usize..b.clamp(0, len) as usize].iter().collect::<String>())
        }
        "split" => {
            let sep = arg(0);
            let parts: Vec<Value> = match sep {
                Value::Undefined => alloc::vec![Value::str(s.to_string())],
                _ => {
                    let sep = sep.to_display();
                    if sep.is_empty() {
                        s.chars().map(|c| Value::str(c.to_string())).collect()
                    } else {
                        s.split(&sep as &str).map(|p| Value::str(p.to_string())).collect()
                    }
                }
            };
            Value::Array(Rc::new(RefCell::new(parts)))
        }
        "replace" => Value::str(s.replacen(&arg(0).to_display(), &arg(1).to_display(), 1)),
        "replaceAll" => Value::str(s.replace(&arg(0).to_display(), &arg(1).to_display())),
        "repeat" => Value::str(s.repeat(arg(0).to_number().max(0.0) as usize)),
        "trimStart" => Value::str(s.trim_start().to_string()),
        "trimEnd" => Value::str(s.trim_end().to_string()),
        "toString" => Value::str(s.to_string()),
        _ => return Err(alloc::format!("string has no method '{}'", method)),
    })
}

fn norm_index(v: Value, len: i64, default: i64) -> i64 {
    match v {
        Value::Undefined => default,
        other => {
            let n = other.to_number();
            if n.is_nan() {
                default
            } else {
                let i = n as i64;
                if i < 0 {
                    (len + i).max(0)
                } else {
                    i.min(len)
                }
            }
        }
    }
}

fn eval_binary(op: &str, l: Value, r: Value) -> EvalResult {
    Ok(match op {
        "+" => {
            if matches!(l, Value::Str(_)) || matches!(r, Value::Str(_)) {
                Value::str(alloc::format!("{}{}", l.to_display(), r.to_display()))
            } else {
                Value::Num(l.to_number() + r.to_number())
            }
        }
        "-" => Value::Num(l.to_number() - r.to_number()),
        "*" => Value::Num(l.to_number() * r.to_number()),
        "/" => Value::Num(l.to_number() / r.to_number()),
        "%" => {
            let a = l.to_number();
            let b = r.to_number();
            Value::Num(a - b * crate::datatypes::floor(a / b))
        }
        "<" => cmp(&l, &r, |o| o.is_lt()),
        ">" => cmp(&l, &r, |o| o.is_gt()),
        "<=" => cmp(&l, &r, |o| o.is_le()),
        ">=" => cmp(&l, &r, |o| o.is_ge()),
        "==" => Value::Bool(loose_equal(&l, &r)),
        "!=" => Value::Bool(!loose_equal(&l, &r)),
        "===" => Value::Bool(strict_equal(&l, &r)),
        "!==" => Value::Bool(!strict_equal(&l, &r)),
        _ => Value::Undefined,
    })
}

fn cmp(l: &Value, r: &Value, f: impl Fn(core::cmp::Ordering) -> bool) -> Value {
    if let (Value::Str(a), Value::Str(b)) = (l, r) {
        return Value::Bool(f(a.as_str().cmp(b.as_str())));
    }
    let (a, b) = (l.to_number(), r.to_number());
    match a.partial_cmp(&b) {
        Some(o) => Value::Bool(f(o)),
        None => Value::Bool(false),
    }
}

fn strict_equal(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Undefined, Value::Undefined) => true,
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Num(a), Value::Num(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => Rc::ptr_eq(a, b),
        (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
        (Value::Node(a), Value::Node(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}

fn loose_equal(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Null | Value::Undefined, Value::Null | Value::Undefined) => true,
        (Value::Num(_), Value::Num(_)) | (Value::Str(_), Value::Str(_)) | (Value::Bool(_), Value::Bool(_)) => {
            strict_equal(l, r)
        }
        (Value::Null | Value::Undefined, _) | (_, Value::Null | Value::Undefined) => false,
        _ => {
            if matches!(l, Value::Object(_) | Value::Array(_) | Value::Node(_))
                || matches!(r, Value::Object(_) | Value::Array(_) | Value::Node(_))
            {
                strict_equal(l, r)
            } else {
                l.to_number() == r.to_number()
            }
        }
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    strict_equal(a, b)
}

fn powf(base: f64, exp: f64) -> f64 {
    if exp == 0.0 {
        return 1.0;
    }
    if exp == 0.5 {
        return crate::datatypes::sqrt(base);
    }
    let n = exp as i64;
    if n as f64 == exp {
        let mut r = 1.0;
        let mut b = base;
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                r *= b;
            }
            b *= b;
            e >>= 1;
        }
        if n < 0 {
            1.0 / r
        } else {
            r
        }
    } else {
        f64::NAN
    }
}

fn json_stringify(v: &Value) -> String {
    match v {
        Value::Undefined | Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Num(n) => fmt_num(*n),
        Value::Str(s) => alloc::format!("\"{}\"", json_escape(s)),
        Value::Array(a) => {
            let items: Vec<String> = a.borrow().iter().map(json_stringify).collect();
            alloc::format!("[{}]", items.join(","))
        }
        Value::Object(map) => {
            let items: Vec<String> = map
                .borrow()
                .iter()
                .map(|(k, v)| alloc::format!("\"{}\":{}", json_escape(k), json_stringify(v)))
                .collect();
            alloc::format!("{{{}}}", items.join(","))
        }
        _ => "null".to_string(),
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn json_parse(s: &str) -> EvalResult {
    let toks = Lexer::new(s).tokens().map_err(|e| alloc::format!("JSON parse: {}", e))?;
    let mut p = Parser::new(toks);
    let e = p.expression().map_err(|e| alloc::format!("JSON parse: {}", e))?;
    json_eval(&e)
}

fn json_eval(e: &Expr) -> EvalResult {
    Ok(match e {
        Expr::Num(n) => Value::Num(*n),
        Expr::Str(s) => Value::str(s.clone()),
        Expr::Bool(b) => Value::Bool(*b),
        Expr::Null => Value::Null,
        Expr::Unary(op, inner) if op == "-" => Value::Num(-json_eval(inner)?.to_number()),
        Expr::Array(items) => {
            let mut v = Vec::new();
            for it in items {
                v.push(json_eval(it)?);
            }
            Value::Array(Rc::new(RefCell::new(v)))
        }
        Expr::Object(props) => {
            let mut map = BTreeMap::new();
            for (k, ve) in props {
                map.insert(k.clone(), json_eval(ve)?);
            }
            Value::Object(Rc::new(RefCell::new(map)))
        }
        _ => return Err("invalid JSON".to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom;

    fn run(src: &str) -> Js {
        let doc = dom::parse_document("<html><body></body></html>");
        let mut js = Js::new(doc);
        js.run(src).unwrap();
        js
    }

    fn run_on(html: &str, src: &str) -> (NodeRef, Js) {
        let doc = dom::parse_document(html);
        let mut js = Js::new(doc.clone());
        js.run(src).unwrap();
        (doc, js)
    }

    #[test]
    fn arithmetic_and_console() {
        let js = run("console.log(1 + 2 * 3); console.log('a' + 'b');");
        assert_eq!(js.console(), ["7", "ab"]);
    }

    #[test]
    fn variables_and_functions_with_closures() {
        let js = run(
            "function adder(n){ return function(x){ return x + n; }; }
             var add5 = adder(5);
             console.log(add5(10));",
        );
        assert_eq!(js.console(), ["15"]);
    }

    #[test]
    fn control_flow_for_while_if() {
        let js = run(
            "var total = 0;
             for (var i = 1; i <= 5; i++) { if (i % 2 === 0) continue; total += i; }
             var n = 3; while (n > 0) { total += n; n--; }
             console.log(total);",
        );
        assert_eq!(js.console(), ["15"]);
    }

    #[test]
    fn arrays_and_higher_order() {
        let js = run(
            "var xs = [1,2,3,4];
             var doubled = xs.map(function(x){ return x*2; });
             var evens = xs.filter(function(x){ return x % 2 === 0; });
             console.log(doubled.join(','));
             console.log(evens.length);
             console.log(xs.reduce(function(a,b){return a+b;}, 0));",
        );
        assert_eq!(js.console(), ["2,4,6,8", "2", "10"]);
    }

    #[test]
    fn arrow_functions_and_templates() {
        let js = run(
            "var sq = x => x * x;
             var name = 'world';
             console.log(`sq(4)=${sq(4)} hi ${name}`);",
        );
        assert_eq!(js.console(), ["sq(4)=16 hi world"]);
    }

    #[test]
    fn objects_and_methods() {
        let js = run(
            "var o = { a: 1, b: 2, sum: function(){ return this.a + this.b; } };
             console.log(o.sum());
             o.c = 3;
             console.log(o.c);",
        );
        assert_eq!(js.console(), ["3", "3"]);
    }

    #[test]
    fn string_methods() {
        let js = run(
            "var s = 'Hello, World';
             console.log(s.toUpperCase());
             console.log(s.split(', ').join('-'));
             console.log(s.slice(0, 5));",
        );
        assert_eq!(js.console(), ["HELLO, WORLD", "Hello-World", "Hello"]);
    }

    #[test]
    fn dom_read_and_write_text() {
        let (doc, _js) = run_on(
            "<div id='out'>old</div>",
            "document.getElementById('out').textContent = 'new value';",
        );
        let out = dom::get_element_by_id(&doc, "out").unwrap();
        assert_eq!(dom::text_content(&out), "new value");
    }

    #[test]
    fn dom_inner_html_and_create_append() {
        let (doc, _js) = run_on(
            "<ul id='list'></ul>",
            "var ul = document.getElementById('list');
             for (var i = 0; i < 3; i++) {
               var li = document.createElement('li');
               li.textContent = 'item ' + i;
               ul.appendChild(li);
             }",
        );
        let list = dom::get_element_by_id(&doc, "list").unwrap();
        assert_eq!(dom::query_selector_all(&list, "li").len(), 3);
        assert!(dom::text_content(&list).contains("item 2"));
    }

    #[test]
    fn dom_style_and_class_mutation() {
        let (doc, _js) = run_on(
            "<p id='p'>x</p>",
            "var p = document.getElementById('p');
             p.style.color = 'red';
             p.className = 'highlight active';",
        );
        let p = dom::get_element_by_id(&doc, "p").unwrap();
        assert!(dom::get_attr(&p, "style").unwrap().contains("color:red"));
        assert_eq!(dom::classes(&p), ["highlight", "active"]);
    }

    #[test]
    fn event_listener_fires_and_mutates() {
        let (doc, mut js) = run_on(
            "<button id='b'>0</button>",
            "var b = document.getElementById('b');
             var count = 0;
             b.addEventListener('click', function(){ count++; b.textContent = count; });",
        );
        let b = dom::get_element_by_id(&doc, "b").unwrap();
        assert!(js.has_click_handler(&b));
        js.fire_event(&b, "click");
        js.fire_event(&b, "click");
        assert_eq!(dom::text_content(&b), "2");
    }

    #[test]
    fn json_round_trip() {
        let js = run(
            "var o = JSON.parse('{\"a\":1,\"b\":[2,3]}');
             console.log(o.a);
             console.log(o.b[1]);
             console.log(JSON.stringify({x: 5, y: 'hi'}));",
        );
        assert_eq!(js.console(), ["1", "3", "{\"x\":5,\"y\":\"hi\"}"]);
    }

    #[test]
    fn math_and_typeof() {
        let js = run(
            "console.log(Math.max(3, 7, 2));
             console.log(Math.floor(3.7));
             console.log(typeof 'x');
             console.log(typeof undefinedVar);",
        );
        assert_eq!(js.console(), ["7", "3", "string", "undefined"]);
    }

    #[test]
    fn recursion() {
        let js = run(
            "function fib(n){ return n < 2 ? n : fib(n-1) + fib(n-2); }
             console.log(fib(10));",
        );
        assert_eq!(js.console(), ["55"]);
    }

    #[test]
    fn bad_script_is_caught_not_panicked() {
        let doc = dom::parse_document("<body></body>");
        let mut js = Js::new(doc);
        assert!(js.run("this is not ( valid").is_err());
        js.run("console.log('still alive');").unwrap();
        assert_eq!(js.console().last().unwrap(), "still alive");
    }
}


//! Polyglot parser: a `Tok` stream → the shared `Program` AST (Pratt-style
//! expression parsing plus statement/function forms). One grammar drives every
//! supported surface language; per-language quirks ride in via the `Dialect`.
//! Split out of the former monolithic `polyglot.rs`.

use super::ast::{BinOp, Expr, Func, Program, Stmt};
use super::lexer::{lex, Tok};
use super::*;

/// Hard cap on recursive-descent nesting. Guest source is untrusted input, and
/// the bare-metal kernel stack is small with no guard page, so unbounded
/// recursion (e.g. `((((...))))` or deep `if`/block nesting) would overflow the
/// stack and corrupt memory. Well past any realistic legitimate program.
const MAX_PARSE_DEPTH: usize = 128;

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    d: Dialect,
    lang: Language,
    _src: &'a str,
    /// Current recursion depth of the descent, bounded by `MAX_PARSE_DEPTH`.
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn peek_at(&self, k: usize) -> Option<&Tok> {
        self.toks.get(self.pos + k)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn is_punct(&self, p: &str) -> bool {
        matches!(self.peek(), Some(Tok::Punct(x)) if x == p)
    }
    fn is_ident(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(x)) if x == s)
    }
    fn eat_punct(&mut self, p: &str) -> bool {
        if self.is_punct(p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect_punct(&mut self, p: &str) -> Result<(), RunError> {
        if self.eat_punct(p) {
            Ok(())
        } else {
            Err(RunError::Parse(format!("expected '{}', got {:?}", p, self.peek())))
        }
    }
    fn eat_ident(&mut self, s: &str) -> bool {
        if self.is_ident(s) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn ident(&mut self) -> Result<String, RunError> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(RunError::Parse(format!("expected identifier, got {:?}", other))),
        }
    }
    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Tok::Newline)) {
            self.pos += 1;
        }
    }
}

/// Parse a complete guest program written in `lang`.
pub fn parse(src: &str, lang: Language) -> Result<Program, RunError> {
    let d = lang.dialect();
    let toks = lex(src, &d)?;
    let mut p = Parser { toks, pos: 0, d, lang, _src: src, depth: 0 };
    let mut imports: Vec<&'static str> = Vec::new();
    let mut funcs: Vec<Func> = Vec::new();
    let mut main: Vec<Stmt> = Vec::new();

    loop {
        p.skip_newlines();
        if p.peek().is_none() {
            break;
        }
        // Import line?
        if let Some(pkgs) = p.try_parse_import()? {
            for pk in pkgs {
                if !imports.contains(&pk) {
                    imports.push(pk);
                }
            }
            continue;
        }
        // Class / namespace wrapper (C#, Java, C++): pull methods out into top-level
        // functions. We support a single nesting level, which is all the guests need.
        if p.try_open_container()? {
            continue;
        }
        if p.is_punct("}") {
            // closing a container we flattened
            p.pos += 1;
            continue;
        }
        // Function definition?
        if p.looks_like_fn() {
            funcs.push(p.parse_fn()?);
            continue;
        }
        // Otherwise: a top-level statement (the "main" body / script form).
        let s = p.parse_stmt()?;
        main.push(s);
    }

    Ok(Program { imports, funcs, main, language: lang })
}

impl<'a> Parser<'a> {
    /// Recognize and consume an import/use/include/using/require line, returning the
    /// canonical package name(s) it brings in (empty vec ⇒ a recognized-but-ignored
    /// import like C#'s `using System;`). Returns `None` if this is not an import.
    fn try_parse_import(&mut self) -> Result<Option<Vec<&'static str>>, RunError> {
        match self.lang {
            Language::Python => {
                if self.eat_ident("import") {
                    let name = self.ident()?;
                    // optional `as alias`
                    if self.eat_ident("as") {
                        let _ = self.ident()?;
                    }
                    self.end_simple();
                    return Ok(Some(canon_pkgs(&[name])));
                }
                if self.eat_ident("from") {
                    let module = self.ident()?;
                    let _ = self.eat_ident("import");
                    // consume the imported-symbol list to end of line
                    while !matches!(self.peek(), Some(Tok::Newline) | None) {
                        self.pos += 1;
                    }
                    self.end_simple();
                    return Ok(Some(canon_pkgs(&[module])));
                }
                Ok(None)
            }
            Language::Rust => {
                if self.eat_ident("use") {
                    // first path segment is the crate/package
                    let krate = self.ident()?;
                    while !self.is_punct(";") && self.peek().is_some() {
                        self.pos += 1;
                    }
                    self.eat_punct(";");
                    return Ok(Some(canon_pkgs(&[krate])));
                }
                Ok(None)
            }
            Language::Cpp => {
                if self.is_punct("#") {
                    // #include <name> or #include "name"
                    self.pos += 1;
                    let _ = self.eat_ident("include");
                    let mut name = String::new();
                    if self.eat_punct("<") {
                        while !self.is_punct(">") && self.peek().is_some() {
                            if let Some(Tok::Ident(s)) = self.peek() {
                                name.push_str(s);
                            }
                            self.pos += 1;
                        }
                        self.eat_punct(">");
                    } else if let Some(Tok::Str(s)) = self.peek().cloned() {
                        name = s;
                        self.pos += 1;
                    }
                    return Ok(Some(canon_pkgs(&[name])));
                }
                // `using namespace std;` — recognized, grants nothing.
                if self.eat_ident("using") {
                    while !self.is_punct(";") && self.peek().is_some() {
                        self.pos += 1;
                    }
                    self.eat_punct(";");
                    return Ok(Some(Vec::new()));
                }
                Ok(None)
            }
            Language::CSharp => {
                if self.eat_ident("using") {
                    let mut last = String::new();
                    while !self.is_punct(";") && self.peek().is_some() {
                        if let Some(Tok::Ident(s)) = self.peek() {
                            last = s.clone();
                        }
                        self.pos += 1;
                    }
                    self.eat_punct(";");
                    return Ok(Some(canon_pkgs(&[last])));
                }
                if self.eat_ident("namespace") {
                    let _ = self.ident();
                    self.eat_punct("{");
                    return Ok(Some(Vec::new()));
                }
                Ok(None)
            }
            Language::Java => {
                if self.eat_ident("package") {
                    while !self.is_punct(";") && self.peek().is_some() {
                        self.pos += 1;
                    }
                    self.eat_punct(";");
                    return Ok(Some(Vec::new()));
                }
                if self.eat_ident("import") {
                    // import a.b.c.*;  — first segment names the package
                    let first = self.ident()?;
                    let mut last = first.clone();
                    while !self.is_punct(";") && self.peek().is_some() {
                        if let Some(Tok::Ident(s)) = self.peek() {
                            last = s.clone();
                        }
                        self.pos += 1;
                    }
                    self.eat_punct(";");
                    return Ok(Some(canon_pkgs(&[first, last])));
                }
                Ok(None)
            }
            Language::JavaScript | Language::TypeScript => {
                // const x = require('pkg');
                if (self.is_ident("const") || self.is_ident("let") || self.is_ident("var"))
                    && self.import_is_require()
                {
                    // skip `const NAME =`
                    self.pos += 1;
                    let _ = self.ident()?;
                    self.expect_punct("=")?;
                    let _ = self.eat_ident("require");
                    self.expect_punct("(")?;
                    let name = match self.next() {
                        Some(Tok::Str(s)) => s,
                        other => return Err(RunError::Parse(format!("require() wants a string, got {:?}", other))),
                    };
                    self.expect_punct(")")?;
                    self.eat_punct(";");
                    return Ok(Some(canon_pkgs(&[name])));
                }
                // import {a,b} from 'pkg';  /  import * as x from 'pkg';
                if self.is_ident("import") {
                    self.pos += 1;
                    while !self.is_ident("from") && self.peek().is_some() {
                        self.pos += 1;
                    }
                    let _ = self.eat_ident("from");
                    let name = match self.next() {
                        Some(Tok::Str(s)) => s,
                        other => return Err(RunError::Parse(format!("import ... from wants a string, got {:?}", other))),
                    };
                    self.eat_punct(";");
                    return Ok(Some(canon_pkgs(&[name])));
                }
                Ok(None)
            }
        }
    }

    /// Lookahead: is this `const/let/var NAME = require(...)`?
    fn import_is_require(&self) -> bool {
        // pattern: kw ident = require (
        matches!(self.peek_at(1), Some(Tok::Ident(_)))
            && matches!(self.peek_at(2), Some(Tok::Punct(p)) if p == "=")
            && matches!(self.peek_at(3), Some(Tok::Ident(s)) if s == "require")
    }

    /// Consume an optional statement terminator (`;` or a Python newline).
    fn end_simple(&mut self) {
        if self.d.python {
            while matches!(self.peek(), Some(Tok::Newline)) {
                self.pos += 1;
            }
        } else {
            self.eat_punct(";");
        }
    }

    /// Open a `class`/`struct`/`namespace { ... }` wrapper, flattening its methods to
    /// top level. Modifiers (`public`, `static`, …) are skipped. Returns true if one
    /// was opened (its body is then parsed by the main loop until the matching `}`).
    fn try_open_container(&mut self) -> Result<bool, RunError> {
        if self.d.python {
            return Ok(false);
        }
        let mut k = 0;
        // skip leading modifiers
        while matches!(self.peek_at(k), Some(Tok::Ident(s))
            if matches!(s.as_str(), "public" | "private" | "protected" | "static" | "final" | "abstract" | "sealed" | "internal" | "export"))
        {
            k += 1;
        }
        if matches!(self.peek_at(k), Some(Tok::Ident(s)) if matches!(s.as_str(), "class" | "struct" | "namespace"))
        {
            // advance past modifiers + keyword + name, then `{`
            self.pos += k + 1;
            let _ = self.ident(); // name
            // skip an optional `: Base` / `extends X` / `implements Y`
            while !self.is_punct("{") && self.peek().is_some() {
                self.pos += 1;
            }
            self.expect_punct("{")?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Heuristic: does the upcoming token sequence begin a function definition?
    fn looks_like_fn(&self) -> bool {
        if let Some(kw) = self.d.fn_keyword {
            return self.is_ident(kw);
        }
        // Typed C-family: skip modifiers, then we need `Type ... name (`.
        let mut k = 0;
        while matches!(self.peek_at(k), Some(Tok::Ident(s))
            if matches!(s.as_str(), "public" | "private" | "protected" | "static" | "final" | "abstract" | "virtual" | "override" | "inline" | "const"))
        {
            k += 1;
        }
        // Now expect a type (>=1 ident, possibly with <...>, [], *, &, ::), then a
        // name ident, then '('. Scan idents/type-puncts until we hit an ident that is
        // immediately followed by '('.
        let mut saw_type = false;
        loop {
            match self.peek_at(k) {
                Some(Tok::Ident(_)) => {
                    // is this ident the function name? (next non-type token is '(')
                    if matches!(self.peek_at(k + 1), Some(Tok::Punct(p)) if p == "(") && saw_type {
                        return true;
                    }
                    saw_type = true;
                    k += 1;
                }
                Some(Tok::Punct(p)) if matches!(p.as_str(), "<" | ">" | "[" | "]" | "*" | "&" | "::" | ",") => {
                    k += 1;
                }
                _ => return false,
            }
            if k > 32 {
                return false;
            }
        }
    }

    fn parse_fn(&mut self) -> Result<Func, RunError> {
        // Skip modifiers.
        while matches!(self.peek(), Some(Tok::Ident(s))
            if matches!(s.as_str(), "public" | "private" | "protected" | "static" | "final" | "abstract" | "virtual" | "override" | "inline"))
        {
            self.pos += 1;
        }
        if let Some(kw) = self.d.fn_keyword {
            let _ = self.eat_ident(kw);
            // (Python `def`, Rust `fn`, JS/TS `function`)
        } else {
            // Typed: consume the return type (everything up to the name ident that is
            // followed by '(').
            loop {
                match self.peek() {
                    Some(Tok::Ident(_)) => {
                        if matches!(self.peek_at(1), Some(Tok::Punct(p)) if p == "(") {
                            break; // this ident is the name
                        }
                        self.pos += 1;
                    }
                    Some(Tok::Punct(p)) if matches!(p.as_str(), "<" | ">" | "[" | "]" | "*" | "&" | "::" | ",") => {
                        self.pos += 1;
                    }
                    other => return Err(RunError::Parse(format!("malformed function signature near {:?}", other))),
                }
            }
        }
        let name = self.ident()?;
        self.expect_punct("(")?;
        let params = self.parse_params()?;
        self.expect_punct(")")?;

        // Optional return-type annotations: Rust `-> T`, TS `: T`, C/Java already
        // consumed. Skip until the block opener.
        if self.d.python {
            // skip `-> T` then `:`
            while !self.is_punct(":") && self.peek().is_some() {
                self.pos += 1;
            }
            self.expect_punct(":")?;
            let body = self.parse_block_python()?;
            return Ok(Func { name, params, body });
        }
        while !self.is_punct("{") && self.peek().is_some() {
            self.pos += 1;
        }
        self.expect_punct("{")?;
        let body = self.parse_block_brace()?;
        Ok(Func { name, params, body })
    }

    fn parse_params(&mut self) -> Result<Vec<String>, RunError> {
        let mut params = Vec::new();
        if self.is_punct(")") {
            return Ok(params);
        }
        loop {
            // Collect one parameter chunk (until ',' or ')'), tracking idents.
            let mut idents: Vec<String> = Vec::new();
            let mut depth = 0i32;
            loop {
                match self.peek() {
                    Some(Tok::Punct(p)) if p == "(" || p == "<" || p == "[" => {
                        depth += 1;
                        self.pos += 1;
                    }
                    Some(Tok::Punct(p)) if p == ")" || p == ">" || p == "]" => {
                        if p == ")" && depth == 0 {
                            break;
                        }
                        depth -= 1;
                        self.pos += 1;
                    }
                    Some(Tok::Punct(p)) if p == "," && depth == 0 => break,
                    Some(Tok::Ident(s)) => {
                        idents.push(s.clone());
                        self.pos += 1;
                    }
                    Some(_) => {
                        self.pos += 1;
                    }
                    None => break,
                }
            }
            if !idents.is_empty() {
                // Drop trailing type-ish keywords if name-first dialect with `name: T`.
                let name = if self.d.param_name_first {
                    idents.first().cloned().unwrap()
                } else {
                    idents.last().cloned().unwrap()
                };
                params.push(name);
            }
            if self.eat_punct(",") {
                continue;
            }
            break;
        }
        Ok(params)
    }

    // ── blocks ──

    fn parse_block_brace(&mut self) -> Result<Vec<Stmt>, RunError> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.is_punct("}") {
                self.pos += 1;
                break;
            }
            if self.peek().is_none() {
                return Err(RunError::Parse(String::from("unexpected EOF in block")));
            }
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_block_python(&mut self) -> Result<Vec<Stmt>, RunError> {
        // Expect: Newline Indent stmts Dedent
        while matches!(self.peek(), Some(Tok::Newline)) {
            self.pos += 1;
        }
        if !matches!(self.peek(), Some(Tok::Indent)) {
            return Err(RunError::Parse(String::from("expected an indented block")));
        }
        self.pos += 1;
        let mut stmts = Vec::new();
        loop {
            while matches!(self.peek(), Some(Tok::Newline)) {
                self.pos += 1;
            }
            if matches!(self.peek(), Some(Tok::Dedent)) {
                self.pos += 1;
                break;
            }
            if self.peek().is_none() {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, RunError> {
        if self.d.python {
            self.expect_punct(":")?;
            self.parse_block_python()
        } else {
            self.expect_punct("{")?;
            self.parse_block_brace()
        }
    }

    // ── statements ──

    fn parse_stmt(&mut self) -> Result<Stmt, RunError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(RunError::Parse(String::from("parser nesting too deep")));
        }
        let r = self.parse_stmt_body();
        self.depth -= 1;
        r
    }

    fn parse_stmt_body(&mut self) -> Result<Stmt, RunError> {
        // control flow
        if self.is_ident("if") {
            return self.parse_if();
        }
        if self.is_ident("while") {
            self.pos += 1;
            let cond = self.parse_paren_or_expr_cond()?;
            let body = self.parse_block()?;
            return Ok(Stmt::While(cond, body));
        }
        if self.is_ident("for") || self.is_ident("foreach") {
            return self.parse_for();
        }
        if self.is_ident("return") {
            self.pos += 1;
            if self.stmt_at_end() {
                self.end_simple();
                return Ok(Stmt::Return(None));
            }
            let e = self.parse_expr()?;
            self.end_simple();
            return Ok(Stmt::Return(Some(e)));
        }

        // declaration vs assignment vs expression: collect the simple-statement span.
        self.parse_simple_stmt()
    }

    fn stmt_at_end(&self) -> bool {
        matches!(self.peek(), Some(Tok::Newline) | None) || self.is_punct(";")
    }

    fn parse_if(&mut self) -> Result<Stmt, RunError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(RunError::Parse(String::from("parser nesting too deep")));
        }
        self.pos += 1; // if
        let cond = self.parse_paren_or_expr_cond()?;
        let then = self.parse_block()?;
        let mut els: Vec<Stmt> = Vec::new();
        self.skip_newlines();
        if self.is_ident("elif") {
            // Python elif → nested if in else branch
            els.push(self.parse_if_as_elif()?);
        } else if self.eat_ident("else") {
            if self.is_ident("if") {
                els.push(self.parse_if()?);
            } else {
                els = self.parse_block()?;
            }
        }
        self.depth -= 1;
        Ok(Stmt::If(cond, then, els))
    }

    fn parse_if_as_elif(&mut self) -> Result<Stmt, RunError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(RunError::Parse(String::from("parser nesting too deep")));
        }
        self.pos += 1; // elif
        let cond = self.parse_paren_or_expr_cond()?;
        let then = self.parse_block()?;
        let mut els: Vec<Stmt> = Vec::new();
        self.skip_newlines();
        if self.is_ident("elif") {
            els.push(self.parse_if_as_elif()?);
        } else if self.eat_ident("else") {
            els = self.parse_block()?;
        }
        self.depth -= 1;
        Ok(Stmt::If(cond, then, els))
    }

    /// A condition is either `( expr )` (C-family) or a bare `expr` (Rust/Python).
    fn parse_paren_or_expr_cond(&mut self) -> Result<Expr, RunError> {
        if !self.d.python && !self.d.rust_for && self.is_punct("(") {
            // C-family wraps conditions in parens; but Rust/Python do not. We accept
            // a leading '(' generically and let the expression parser consume it.
        }
        self.parse_expr()
    }

    fn parse_for(&mut self) -> Result<Stmt, RunError> {
        let _ = self.eat_ident("for") || self.eat_ident("foreach");
        if self.d.rust_for {
            // for VAR in (RANGE | LIST) { body }
            let var = self.ident()?;
            let _ = self.eat_ident("in");
            return self.finish_for_in(var);
        }
        if self.d.python {
            // for VAR in (range(a,b) | LIST):
            let var = self.ident()?;
            let _ = self.eat_ident("in");
            return self.finish_for_in(var);
        }
        // C-family: for ( ... ) body   — either C-style or range-for.
        self.expect_punct("(")?;
        // Detect C-style by a top-level ';' inside the parens.
        let save = self.pos;
        let mut depth = 0i32;
        let mut c_style = false;
        let mut j = self.pos;
        while let Some(t) = self.toks.get(j) {
            match t {
                Tok::Punct(p) if p == "(" => depth += 1,
                Tok::Punct(p) if p == ")" => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                Tok::Punct(p) if p == ";" && depth == 0 => {
                    c_style = true;
                    break;
                }
                _ => {}
            }
            j += 1;
        }
        let _ = save;
        if c_style {
            // init ; cond ; step
            let init = Box::new(self.parse_simple_stmt_no_term()?);
            self.expect_punct(";")?;
            let cond = self.parse_expr()?;
            self.expect_punct(";")?;
            let step = Box::new(self.parse_simple_stmt_no_term()?);
            self.expect_punct(")")?;
            let mut body = self.parse_block()?;
            // desugar to while: { init; while cond { body; step } }
            body.push(*step);
            let while_stmt = Stmt::While(cond, body);
            // We can't return two statements; wrap init+while in a synthetic block via
            // ForRange? Instead emit a While preceded by init using a small trick: a
            // block isn't a Stmt. So return a While and prepend init by returning a
            // grouping. We model grouping as ForEach over a one-shot — simpler: encode
            // as nested via a `Stmt::If(true, [init, while], [])`.
            return Ok(Stmt::If(Expr::Bool(true), vec![*init, while_stmt], Vec::new()));
        }
        // range-for: `Type VAR : EXPR` (C++/Java) or `var VAR in EXPR` (C#)
        // collect ident before the separator (':' or 'in')
        let mut idents: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::Ident(s)) if s == "in" => {
                    self.pos += 1;
                    break;
                }
                Some(Tok::Punct(p)) if p == ":" => {
                    self.pos += 1;
                    break;
                }
                Some(Tok::Ident(s)) => {
                    idents.push(s.clone());
                    self.pos += 1;
                }
                Some(Tok::Punct(p)) if matches!(p.as_str(), "<" | ">" | "[" | "]" | "*" | "&" | "::") => {
                    self.pos += 1;
                }
                other => return Err(RunError::Parse(format!("malformed for-each header near {:?}", other))),
            }
        }
        let var = idents.last().cloned().ok_or_else(|| RunError::Parse(String::from("for-each needs a variable")))?;
        let iter = self.parse_expr()?;
        self.expect_punct(")")?;
        let body = self.parse_block()?;
        Ok(Stmt::ForEach(var, iter, body))
    }

    /// After `for VAR in`, parse the range/list and the body (Rust/Python form).
    fn finish_for_in(&mut self, var: String) -> Result<Stmt, RunError> {
        // range(a, b) — or range(n) ⇒ 0..n.
        if self.is_ident("range") {
            self.pos += 1;
            self.expect_punct("(")?;
            let first = self.parse_expr()?;
            let (start, end) = if self.eat_punct(",") {
                (first, self.parse_expr()?)
            } else {
                (Expr::Int(0), first)
            };
            self.expect_punct(")")?;
            let body = self.parse_block()?;
            return Ok(Stmt::ForRange(var, Expr::Range(Box::new(start), Box::new(end)), body));
        }
        // Otherwise parse an expression; it may be `a..b` (Rust range) or a list.
        let e = self.parse_expr()?;
        let body = self.parse_block()?;
        match e {
            Expr::Range(_, _) => Ok(Stmt::ForRange(var, e, body)),
            other => Ok(Stmt::ForEach(var, other, body)),
        }
    }

    /// Parse a simple statement and consume its terminator.
    fn parse_simple_stmt(&mut self) -> Result<Stmt, RunError> {
        let s = self.parse_simple_stmt_no_term()?;
        self.end_simple();
        Ok(s)
    }

    /// Parse a declaration / assignment / expression statement without the terminator.
    fn parse_simple_stmt_no_term(&mut self) -> Result<Stmt, RunError> {
        // Declaration keyword?
        let mut is_decl = false;
        if matches!(self.peek(), Some(Tok::Ident(s)) if matches!(s.as_str(), "let" | "const" | "var" | "auto")) {
            self.pos += 1;
            is_decl = true;
            let _ = self.eat_ident("mut"); // Rust `let mut`
        }

        // Collect the left-hand side up to a top-level assignment op or end.
        let lhs_start = self.pos;
        let mut depth = 0i32;
        let mut assign_op: Option<String> = None;
        let mut scan = self.pos;
        while let Some(t) = self.toks.get(scan) {
            match t {
                Tok::Punct(p) if p == "(" || p == "[" => depth += 1,
                Tok::Punct(p) if p == ")" || p == "]" => depth -= 1,
                Tok::Punct(p) if depth == 0 && matches!(p.as_str(), "=" | "+=" | "-=" | "*=" | "/=") => {
                    assign_op = Some(p.clone());
                    break;
                }
                Tok::Punct(p) if depth == 0 && p == ";" => break,
                Tok::Newline => break,
                _ => {}
            }
            scan += 1;
        }

        if let Some(op) = assign_op {
            // Tokens [lhs_start, scan) are the LHS; classify it.
            let lhs: Vec<Tok> = self.toks[lhs_start..scan].to_vec();
            self.pos = scan + 1; // consume the assignment op
            let rhs = self.parse_expr()?;

            if let Some(Tok::Ident(name)) = lhs.first() {
                let ident_count = lhs.iter().filter(|t| matches!(t, Tok::Ident(_))).count();
                // 1. Explicit declaration keyword (`let`/`var`/`const`/`auto`): the
                //    variable name is the first identifier; any brackets belong to a
                //    type annotation (e.g. TS `let out: number[] = []`), not an index.
                if is_decl {
                    let rhs = desugar_compound(&op, name, rhs);
                    return Ok(Stmt::Let(name.clone(), rhs));
                }
                // 2. Index assignment `arr[i] = v`: the LHS ends with a `]`.
                let ends_with_bracket = matches!(lhs.last(), Some(Tok::Punct(p)) if p == "]");
                if ends_with_bracket {
                    let mut sub = Parser {
                        toks: lhs.clone(),
                        pos: 0,
                        d: self.d,
                        lang: self.lang,
                        _src: self._src,
                        depth: self.depth,
                    };
                    let target = sub.parse_expr()?;
                    if let Expr::Index(base, idx) = target {
                        // Apply the compound operator against the element read,
                        // matching the plain-var/decl branches (`arr[i] += v`).
                        let rhs = match compound_binop(&op) {
                            Some(b) => Expr::Bin(
                                b,
                                Box::new(Expr::Index(base.clone(), idx.clone())),
                                Box::new(rhs),
                            ),
                            None => rhs,
                        };
                        return Ok(Stmt::IndexAssign(*base, *idx, rhs));
                    }
                    return Err(RunError::Parse(String::from("bad index assignment")));
                }
                // 3. Typed declaration `T name = ..` (C/C#/Java): the name is the last
                //    identifier; the rest is the type.
                if ident_count > 1 {
                    return Ok(Stmt::Let(name_from_lhs(&lhs, self.d.param_name_first), rhs));
                }
                // 4. Plain assignment to an existing variable.
                let rhs = desugar_compound(&op, name, rhs);
                return Ok(Stmt::Assign(name.clone(), rhs));
            }
            return Err(RunError::Parse(String::from("unsupported assignment target")));
        }

        // No assignment: it's an expression statement (e.g. a call). If we consumed a
        // decl keyword but found no '=', treat the remainder as a declaration with a
        // default — but that's unusual; parse as expression for robustness.
        let _ = is_decl;
        let e = self.parse_expr()?;
        Ok(Stmt::Expr(e))
    }

    // ── expressions (Pratt) ──

    fn parse_expr(&mut self) -> Result<Expr, RunError> {
        self.parse_bin(0)
    }

    fn parse_bin(&mut self, min_bp: u8) -> Result<Expr, RunError> {
        let mut lhs = self.parse_unary()?;
        while let Some(Tok::Punct(p)) = self.peek() {
            let (op, bp) = match bin_for(p) {
                Some(x) => x,
                None => {
                    // Rust range `a..b`
                    if p == ".." {
                        self.pos += 1;
                        let rhs = self.parse_bin(3)?;
                        lhs = Expr::Range(Box::new(lhs), Box::new(rhs));
                        continue;
                    }
                    break;
                }
            };
            if bp < min_bp {
                break;
            }
            self.pos += 1;
            let rhs = self.parse_bin(bp + 1)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, RunError> {
        // Every level of expression nesting — parens, unary chains, casts/`new`,
        // and nested indexing — passes through here exactly once, so a single
        // depth guard bounds all recursive-descent stack growth for expressions.
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(RunError::Parse(String::from("parser nesting too deep")));
        }
        let r = self.parse_unary_body();
        self.depth -= 1;
        r
    }

    fn parse_unary_body(&mut self) -> Result<Expr, RunError> {
        if self.is_punct("-") {
            self.pos += 1;
            return Ok(Expr::Neg(Box::new(self.parse_unary()?)));
        }
        if self.is_punct("!") {
            self.pos += 1;
            return Ok(Expr::Not(Box::new(self.parse_unary()?)));
        }
        if self.is_punct("+") {
            self.pos += 1;
            return self.parse_unary();
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, RunError> {
        let mut e = self.parse_primary()?;
        loop {
            if self.is_punct("[") {
                self.pos += 1;
                let idx = self.parse_expr()?;
                self.expect_punct("]")?;
                e = Expr::Index(Box::new(e), Box::new(idx));
                continue;
            }
            break;
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> Result<Expr, RunError> {
        match self.peek().cloned() {
            Some(Tok::Int(i)) => {
                self.pos += 1;
                Ok(Expr::Int(i))
            }
            Some(Tok::Float(f)) => {
                self.pos += 1;
                Ok(Expr::Float(f))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Expr::Str(s))
            }
            Some(Tok::Punct(p)) if p == "(" => {
                self.pos += 1;
                // Could be a parenthesized expr or a C-style cast `(double) x` — skip a
                // lone type-cast: `( ident )` directly followed by a primary.
                if self.is_cast() {
                    self.pos += 1; // type ident
                    self.expect_punct(")")?;
                    return self.parse_unary();
                }
                let e = self.parse_expr()?;
                self.expect_punct(")")?;
                Ok(e)
            }
            Some(Tok::Punct(p)) if p == "[" => {
                // list literal
                self.pos += 1;
                let mut items = Vec::new();
                if !self.is_punct("]") {
                    loop {
                        items.push(self.parse_expr()?);
                        if self.eat_punct(",") {
                            if self.is_punct("]") {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                }
                self.expect_punct("]")?;
                Ok(Expr::List(items))
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                // keyword literals
                match name.as_str() {
                    "true" | "True" => return Ok(Expr::Bool(true)),
                    "false" | "False" => return Ok(Expr::Bool(false)),
                    "new" => {
                        // C#/Java `new Type[]{...}` / `new Type(...)` — skip `new` and a
                        // type, fall through to whatever literal/call follows.
                        return self.parse_unary();
                    }
                    _ => {}
                }
                // Rust macro `vec![...]` and `name!(...)`
                if self.is_punct("!") {
                    self.pos += 1;
                    if name == "vec" && self.is_punct("[") {
                        self.pos += 1;
                        let mut items = Vec::new();
                        if !self.is_punct("]") {
                            loop {
                                items.push(self.parse_expr()?);
                                if self.eat_punct(",") {
                                    if self.is_punct("]") {
                                        break;
                                    }
                                    continue;
                                }
                                break;
                            }
                        }
                        self.expect_punct("]")?;
                        return Ok(Expr::List(items));
                    }
                    // println!(...) etc → treat as a call to `name`
                }
                // Build a dotted/`::` path.
                let mut path = vec![name];
                while self.is_punct(".") || self.is_punct("::") {
                    self.pos += 1;
                    // allow method-ish names
                    let seg = self.ident()?;
                    path.push(seg);
                }
                // Call?
                if self.is_punct("(") {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if !self.is_punct(")") {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.eat_punct(",") {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect_punct(")")?;
                    return Ok(Expr::Call(path, args));
                }
                if path.len() == 1 {
                    Ok(Expr::Var(path.into_iter().next().unwrap()))
                } else {
                    // member access without a call → treat as a qualified variable
                    Ok(Expr::Var(path.join(".")))
                }
            }
            other => Err(RunError::Parse(format!("unexpected token {:?}", other))),
        }
    }

    /// Lookahead for a C-style cast `( TypeIdent )` not followed by an operator.
    fn is_cast(&self) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if matches!(s.as_str(),
            "int" | "long" | "double" | "float" | "i64" | "f64" | "i32" | "usize" | "size_t"))
            && matches!(self.peek_at(1), Some(Tok::Punct(p)) if p == ")")
            && matches!(self.peek_at(2), Some(Tok::Ident(_)) | Some(Tok::Int(_)) | Some(Tok::Float(_)) | Some(Tok::Punct(_)))
    }
}

fn name_from_lhs(lhs: &[Tok], name_first: bool) -> String {
    let idents: Vec<&String> = lhs
        .iter()
        .filter_map(|t| if let Tok::Ident(s) = t { Some(s) } else { None })
        .collect();
    let pick = if name_first { idents.first() } else { idents.last() };
    pick.map(|s| (*s).clone()).unwrap_or_default()
}

/// Map a compound-assignment operator (`+=`, `-=`, `*=`, `/=`) to its `BinOp`.
/// Returns `None` for a plain `=`.
fn compound_binop(op: &str) -> Option<BinOp> {
    Some(match op {
        "+=" => BinOp::Add,
        "-=" => BinOp::Sub,
        "*=" => BinOp::Mul,
        "/=" => BinOp::Div,
        _ => return None,
    })
}

fn desugar_compound(op: &str, name: &str, rhs: Expr) -> Expr {
    match compound_binop(op) {
        Some(b) => Expr::Bin(b, Box::new(Expr::Var(name.to_string())), Box::new(rhs)),
        None => rhs,
    }
}

fn bin_for(p: &str) -> Option<(BinOp, u8)> {
    Some(match p {
        "||" => (BinOp::Or, 1),
        "&&" => (BinOp::And, 2),
        "==" => (BinOp::Eq, 4),
        "!=" => (BinOp::Ne, 4),
        "<" => (BinOp::Lt, 5),
        "<=" => (BinOp::Le, 5),
        ">" => (BinOp::Gt, 5),
        ">=" => (BinOp::Ge, 5),
        "+" => (BinOp::Add, 6),
        "-" => (BinOp::Sub, 6),
        "*" => (BinOp::Mul, 7),
        "/" => (BinOp::Div, 7),
        "%" => (BinOp::Rem, 7),
        _ => return None,
    })
}

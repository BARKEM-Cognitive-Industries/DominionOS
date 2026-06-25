//! Polyglot lexer: source text → a flat `Tok` stream, parameterised by the active
//! language `Dialect` (comment markers, string rules, multi-char operators). Split
//! out of the former monolithic `polyglot.rs` for readability; the AST, parser,
//! and interpreter live in their own modules under `polyglot/`.

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    Punct(String),
    Newline,
    Indent,
    Dedent,
}

/// Multi-character operators, longest first.
const MULTI: &[&str] = &[
    "==", "!=", "<=", ">=", "&&", "||", "->", "=>", "::", "+=", "-=", "*=", "/=", "..",
];

pub(crate) fn lex(src: &str, d: &Dialect) -> Result<Vec<Tok>, RunError> {
    let bytes: Vec<char> = src.chars().collect();
    let n = bytes.len();
    let mut i = 0usize;
    let mut toks: Vec<Tok> = Vec::new();
    // Python indentation tracking.
    let mut indent_stack: Vec<usize> = vec![0];
    let mut at_line_start = d.python;

    while i < n {
        // Handle Python line-start indentation.
        if d.python && at_line_start {
            let mut col = 0usize;
            while i < n && (bytes[i] == ' ' || bytes[i] == '\t') {
                col += if bytes[i] == '\t' { 4 } else { 1 };
                i += 1;
            }
            // Blank line or comment-only line: skip without emitting indent tokens.
            if i >= n || bytes[i] == '\n' || bytes[i] == '\r' || is_comment_start(&bytes, i, d) {
                // consume to end of line
                while i < n && bytes[i] != '\n' {
                    i += 1;
                }
                if i < n {
                    i += 1;
                }
                continue;
            }
            let cur = *indent_stack.last().unwrap();
            if col > cur {
                indent_stack.push(col);
                toks.push(Tok::Indent);
            } else {
                while col < *indent_stack.last().unwrap() {
                    indent_stack.pop();
                    toks.push(Tok::Dedent);
                }
            }
            at_line_start = false;
            continue;
        }

        let c = bytes[i];

        // Whitespace.
        if c == ' ' || c == '\t' || c == '\r' {
            i += 1;
            continue;
        }
        if c == '\n' {
            if d.python {
                toks.push(Tok::Newline);
                at_line_start = true;
            }
            i += 1;
            continue;
        }

        // Comments.
        if is_comment_start(&bytes, i, d) {
            while i < n && bytes[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // Block comment /* ... */ (all brace languages).
        if c == '/' && i + 1 < n && bytes[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(bytes[i] == '*' && bytes[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        // C preprocessor / Python `from` directives are tokenized normally; the
        // parser recognizes import lines. But `#include` in C++ starts with `#`,
        // which is the Python comment char only when d.python — here d is C++ so
        // `#` is a punct we keep.

        // String literal.
        if c == '"' || c == '\'' {
            let quote = c;
            i += 1;
            let mut s = String::new();
            while i < n && bytes[i] != quote {
                if bytes[i] == '\\' && i + 1 < n {
                    i += 1;
                    s.push(match bytes[i] {
                        'n' => '\n',
                        't' => '\t',
                        '\\' => '\\',
                        '"' => '"',
                        '\'' => '\'',
                        other => other,
                    });
                } else {
                    s.push(bytes[i]);
                }
                i += 1;
            }
            i += 1; // closing quote
            toks.push(Tok::Str(s));
            continue;
        }

        // Number.
        if c.is_ascii_digit() {
            let start = i;
            while i < n && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // Float? a '.' followed by a digit (not '..' range, not member access).
            if i + 1 < n && bytes[i] == '.' && bytes[i + 1].is_ascii_digit() {
                i += 1;
                while i < n && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let text: String = bytes[start..i].iter().collect();
                let f: f64 = parse_f64(&text)?;
                toks.push(Tok::Float(f));
            } else {
                let text: String = bytes[start..i].iter().collect();
                let v: i64 = text.parse().map_err(|_| RunError::Parse(format!("bad int {}", text)))?;
                toks.push(Tok::Int(v));
            }
            continue;
        }

        // Identifier / keyword.
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < n && (bytes[i].is_alphanumeric() || bytes[i] == '_') {
                i += 1;
            }
            let text: String = bytes[start..i].iter().collect();
            toks.push(Tok::Ident(text));
            continue;
        }

        // Multi-char punctuation.
        let mut matched = false;
        for m in MULTI {
            let mlen = m.chars().count();
            if i + mlen <= n {
                let slice: String = bytes[i..i + mlen].iter().collect();
                if &slice == m {
                    toks.push(Tok::Punct((*m).to_string()));
                    i += mlen;
                    matched = true;
                    break;
                }
            }
        }
        if matched {
            continue;
        }

        // Single-char punctuation.
        toks.push(Tok::Punct(c.to_string()));
        i += 1;
    }

    if d.python {
        // Close out any pending line and indentation.
        toks.push(Tok::Newline);
        while indent_stack.len() > 1 {
            indent_stack.pop();
            toks.push(Tok::Dedent);
        }
    }
    Ok(toks)
}

fn is_comment_start(bytes: &[char], i: usize, d: &Dialect) -> bool {
    let lc = d.line_comment;
    let lcc: Vec<char> = lc.chars().collect();
    if i + lcc.len() > bytes.len() {
        return false;
    }
    bytes[i..i + lcc.len()] == lcc[..]
}

/// A tiny `no_std` float parser good enough for literals like `12.5`.
pub(crate) fn parse_f64(s: &str) -> Result<f64, RunError> {
    let mut parts = s.split('.');
    let int_part = parts.next().unwrap_or("0");
    let frac_part = parts.next().unwrap_or("");
    let i: i64 = int_part.parse().map_err(|_| RunError::Parse(format!("bad float {}", s)))?;
    let mut frac = 0.0f64;
    let mut scale = 0.1f64;
    for ch in frac_part.chars() {
        let dig = ch.to_digit(10).ok_or_else(|| RunError::Parse(format!("bad float {}", s)))?;
        frac += dig as f64 * scale;
        scale *= 0.1;
    }
    Ok(i as f64 + frac)
}

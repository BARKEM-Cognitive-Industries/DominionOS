//! The Aether lexer: source text → a flat token stream.
//!
//! Aether's surface syntax (SRS §5.5) is declarative and data-flow oriented.
//! The lexer recognises the keywords, the `=>` parallel-map operator, the `::`
//! path separator, `@`-decorators (hardware hints like `@NPU`), and the usual
//! literals and operators.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, PartialEq, Debug)]
pub enum Tok {
    // literals
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    Decorator(String), // @NPU, @GPU, @CPU
    // keywords
    Object,
    Cell,
    Fn,
    Let,
    Return,
    If,
    Else,
    True,
    False,
    Cap,    // the `cap` keyword inside a cell header
    Linear, // affine (use-once) binding keyword
    While,
    For,
    In,
    Break,
    Continue,
    // punctuation / operators
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Colon,
    ColonColon, // ::
    Dot,
    Assign,    // =
    FatArrow,  // =>
    ThinArrow, // ->
    Pipe,      // |>
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    NotEq,
    AndAnd, // &&
    OrOr,   // ||
    Bang,   // !
    Eof,
}

/// A token with its 1-based line for error reporting.
#[derive(Clone, PartialEq, Debug)]
pub struct Spanned {
    pub tok: Tok,
    pub line: u32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct LexError {
    pub message: String,
    pub line: u32,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error (line {}): {}", self.line, self.message)
    }
}

pub fn lex(src: &str) -> Result<Vec<Spanned>, LexError> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let mut line = 1u32;
    let mut out = Vec::new();

    let push = |out: &mut Vec<Spanned>, tok: Tok, line: u32| out.push(Spanned { tok, line });

    while i < chars.len() {
        let c = chars[i];
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            c if c.is_whitespace() => {
                i += 1;
            }
            '/' if i + 1 < chars.len() && chars[i + 1] == '*' => {
                // block comment /* ... */
                i += 2; // consume '/' and '*'
                loop {
                    if i >= chars.len() {
                        return Err(LexError {
                            message: "unterminated block comment".to_string(),
                            line,
                        });
                    }
                    if chars[i] == '\n' {
                        line += 1;
                        i += 1;
                    } else if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '/' {
                        i += 2; // consume '*' and '/'
                        break;
                    } else {
                        i += 1;
                    }
                }
            }
            '/' if i + 1 < chars.len() && chars[i + 1] == '/' => {
                // line comment
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                loop {
                    if i >= chars.len() {
                        return Err(LexError {
                            message: "unterminated string".to_string(),
                            line,
                        });
                    }
                    let ch = chars[i];
                    if ch == '"' {
                        i += 1;
                        break;
                    }
                    if ch == '\\' && i + 1 < chars.len() {
                        i += 1;
                        let esc = chars[i];
                        s.push(match esc {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '\\' => '\\',
                            '"' => '"',
                            '0' => '\0',
                            other => other,
                        });
                        i += 1;
                    } else {
                        if ch == '\n' {
                            line += 1;
                        }
                        s.push(ch);
                        i += 1;
                    }
                }
                push(&mut out, Tok::Str(s), line);
            }
            '@' => {
                i += 1;
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                if start == i {
                    return Err(LexError {
                        message: "empty decorator after '@'".to_string(),
                        line,
                    });
                }
                let name: String = chars[start..i].iter().collect();
                push(&mut out, Tok::Decorator(name), line);
            }
            c if c.is_ascii_digit() => {
                let start = i;
                let mut is_float = false;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                    is_float = true;
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                if is_float {
                    let v: f64 = text.parse().map_err(|_| LexError {
                        message: "invalid float literal".to_string(),
                        line,
                    })?;
                    push(&mut out, Tok::Float(v), line);
                } else {
                    let v: i64 = text.parse().map_err(|_| LexError {
                        message: "integer literal out of range".to_string(),
                        line,
                    })?;
                    push(&mut out, Tok::Int(v), line);
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let tok = match word.as_str() {
                    "object" => Tok::Object,
                    "cell" => Tok::Cell,
                    "fn" => Tok::Fn,
                    "let" => Tok::Let,
                    "return" => Tok::Return,
                    "if" => Tok::If,
                    "else" => Tok::Else,
                    "true" => Tok::True,
                    "false" => Tok::False,
                    "cap" => Tok::Cap,
                    "linear" => Tok::Linear,
                    "while" => Tok::While,
                    "for" => Tok::For,
                    "in" => Tok::In,
                    "break" => Tok::Break,
                    "continue" => Tok::Continue,
                    _ => Tok::Ident(word),
                };
                push(&mut out, tok, line);
            }
            _ => {
                // multi-char operators first
                let two: Option<Tok> = if i + 1 < chars.len() {
                    match (c, chars[i + 1]) {
                        ('=', '>') => Some(Tok::FatArrow),
                        ('-', '>') => Some(Tok::ThinArrow),
                        ('|', '>') => Some(Tok::Pipe),
                        (':', ':') => Some(Tok::ColonColon),
                        ('=', '=') => Some(Tok::EqEq),
                        ('!', '=') => Some(Tok::NotEq),
                        ('<', '=') => Some(Tok::Le),
                        ('>', '=') => Some(Tok::Ge),
                        ('&', '&') => Some(Tok::AndAnd),
                        ('|', '|') => Some(Tok::OrOr),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(t) = two {
                    push(&mut out, t, line);
                    i += 2;
                    continue;
                }
                let one = match c {
                    '{' => Tok::LBrace,
                    '}' => Tok::RBrace,
                    '(' => Tok::LParen,
                    ')' => Tok::RParen,
                    '[' => Tok::LBracket,
                    ']' => Tok::RBracket,
                    ',' => Tok::Comma,
                    ';' => Tok::Semicolon,
                    ':' => Tok::Colon,
                    '.' => Tok::Dot,
                    '=' => Tok::Assign,
                    '+' => Tok::Plus,
                    '-' => Tok::Minus,
                    '*' => Tok::Star,
                    '/' => Tok::Slash,
                    '%' => Tok::Percent,
                    '<' => Tok::Lt,
                    '>' => Tok::Gt,
                    '!' => Tok::Bang,
                    other => {
                        return Err(LexError {
                            message: {
                                let mut m = String::from("unexpected character '");
                                m.push(other);
                                m.push('\'');
                                m
                            },
                            line,
                        })
                    }
                };
                push(&mut out, one, line);
                i += 1;
            }
        }
    }

    push(&mut out, Tok::Eof, line);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|s| s.tok).collect()
    }

    #[test]
    fn lexes_keywords_and_idents() {
        assert_eq!(
            toks("let x"),
            vec![Tok::Let, Tok::Ident("x".into()), Tok::Eof]
        );
    }

    #[test]
    fn lexes_numbers() {
        assert_eq!(toks("42 3.5"), vec![Tok::Int(42), Tok::Float(3.5), Tok::Eof]);
    }

    #[test]
    fn lexes_fat_arrow_and_path() {
        assert_eq!(
            toks("xs => A::b"),
            vec![
                Tok::Ident("xs".into()),
                Tok::FatArrow,
                Tok::Ident("A".into()),
                Tok::ColonColon,
                Tok::Ident("b".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn lexes_decorator() {
        assert_eq!(
            toks("@NPU fn"),
            vec![Tok::Decorator("NPU".into()), Tok::Fn, Tok::Eof]
        );
    }

    #[test]
    fn lexes_string_with_escapes() {
        assert_eq!(toks(r#""a\nb""#), vec![Tok::Str("a\nb".into()), Tok::Eof]);
    }

    #[test]
    fn skips_line_comments() {
        assert_eq!(toks("1 // hi\n2"), vec![Tok::Int(1), Tok::Int(2), Tok::Eof]);
    }

    #[test]
    fn tracks_lines() {
        let spans = lex("1\n2\n3").unwrap();
        assert_eq!(spans[0].line, 1);
        assert_eq!(spans[1].line, 2);
        assert_eq!(spans[2].line, 3);
    }

    #[test]
    fn distinguishes_le_from_lt_assign() {
        assert_eq!(toks("<= < == ="), vec![Tok::Le, Tok::Lt, Tok::EqEq, Tok::Assign, Tok::Eof]);
    }

    #[test]
    fn unterminated_string_errors() {
        assert!(lex("\"oops").is_err());
    }

    #[test]
    fn skips_block_comments() {
        assert_eq!(toks("1 /* hi */ 2"), vec![Tok::Int(1), Tok::Int(2), Tok::Eof]);
    }

    #[test]
    fn block_comment_counts_newlines() {
        // Token after a two-newline block comment must be on line 3.
        let spans = lex("/* line one\nline two\n*/ 42").unwrap();
        assert_eq!(spans[0].tok, Tok::Int(42));
        assert_eq!(spans[0].line, 3);
    }

    #[test]
    fn unterminated_block_comment_errors() {
        assert!(lex("/* oops").is_err());
    }

    #[test]
    fn block_comment_inline_between_tokens() {
        assert_eq!(
            toks("let /* ignored */ x"),
            vec![Tok::Let, Tok::Ident("x".into()), Tok::Eof]
        );
    }
}

//! Syntax highlighting for the Dominion language (`docs/ui/ide.md`).
//!
//! A self-contained, allocation-light scanner that classifies source text into colored
//! [`HlSpan`]s — keywords, identifiers, numbers, strings, comments, decorators and
//! operators — so the IDE / editor can render Dominion with highlighting. It scans the raw
//! source (not the lossy lexer span stream) so every byte maps to exactly one span,
//! which is what a renderer needs. Pure, safe `no_std`; host-tested.

use alloc::vec::Vec;

/// A highlight category (the renderer maps each to a theme color).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HlKind {
    Keyword,
    Ident,
    Number,
    Str,
    Comment,
    Decorator,
    Operator,
    Punct,
    Plain,
}

/// A highlighted run: `[start, start+len)` characters of one [`HlKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HlSpan {
    pub start: usize,
    pub len: usize,
    pub kind: HlKind,
}

/// The Dominion keywords (matching `lang/lexer.rs`).
const KEYWORDS: &[&str] = &[
    "object", "cell", "fn", "let", "return", "if", "else", "true", "false", "cap", "linear",
];

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}
fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Classify `src` into a sequence of highlight spans, in source order. Whitespace is left
/// as `Plain` runs so positions stay exact.
pub fn highlight(src: &str) -> Vec<HlSpan> {
    let chars: Vec<char> = src.chars().collect();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let start = i;
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            // line comment to end of line
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            spans.push(HlSpan { start, len: i - start, kind: HlKind::Comment });
        } else if c == '"' {
            // string literal (to the closing quote, honoring a simple backslash escape)
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1;
                }
                i += 1;
            }
            if i < chars.len() {
                i += 1; // consume closing quote
            }
            spans.push(HlSpan { start, len: i - start, kind: HlKind::Str });
        } else if c == '@' && i + 1 < chars.len() && is_ident_start(chars[i + 1]) {
            // decorator (@NPU / @GPU / @CPU / …)
            i += 1;
            while i < chars.len() && is_ident_continue(chars[i]) {
                i += 1;
            }
            spans.push(HlSpan { start, len: i - start, kind: HlKind::Decorator });
        } else if c.is_ascii_digit() {
            // number (int or float)
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            spans.push(HlSpan { start, len: i - start, kind: HlKind::Number });
        } else if is_ident_start(c) {
            // identifier or keyword
            while i < chars.len() && is_ident_continue(chars[i]) {
                i += 1;
            }
            let word: alloc::string::String = chars[start..i].iter().collect();
            let kind = if KEYWORDS.contains(&word.as_str()) { HlKind::Keyword } else { HlKind::Ident };
            spans.push(HlSpan { start, len: i - start, kind });
        } else if is_operator_char(c) {
            while i < chars.len() && is_operator_char(chars[i]) {
                i += 1;
            }
            spans.push(HlSpan { start, len: i - start, kind: HlKind::Operator });
        } else if "{}()[],;:.".contains(c) {
            i += 1;
            spans.push(HlSpan { start, len: 1, kind: HlKind::Punct });
        } else {
            // whitespace / anything else → a plain run
            while i < chars.len()
                && !is_ident_start(chars[i])
                && !chars[i].is_ascii_digit()
                && !is_operator_char(chars[i])
                && !"{}()[],;:.\"@".contains(chars[i])
                && !(chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '/')
            {
                i += 1;
            }
            if i == start {
                i += 1; // guarantee progress
            }
            spans.push(HlSpan { start, len: i - start, kind: HlKind::Plain });
        }
    }
    spans
}

fn is_operator_char(c: char) -> bool {
    "=+-*/%<>|&!".contains(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds_of(src: &str) -> Vec<(HlKind, alloc::string::String)> {
        let chars: Vec<char> = src.chars().collect();
        highlight(src)
            .into_iter()
            .map(|s| (s.kind, chars[s.start..s.start + s.len].iter().collect()))
            .collect()
    }

    #[test]
    fn spans_cover_the_whole_source_exactly() {
        let src = "let x = 42 // note\n";
        let spans = highlight(src);
        // Spans are contiguous and cover every character.
        let mut at = 0;
        for s in &spans {
            assert_eq!(s.start, at);
            at += s.len;
        }
        assert_eq!(at, src.chars().count());
    }

    #[test]
    fn classifies_keywords_idents_numbers_strings_comments_decorators() {
        let got = kinds_of("let foo = 3.14");
        assert_eq!(got[0], (HlKind::Keyword, "let".into()));
        assert!(got.iter().any(|(k, t)| *k == HlKind::Ident && t == "foo"));
        assert!(got.iter().any(|(k, t)| *k == HlKind::Number && t == "3.14"));
        assert!(got.iter().any(|(k, _)| *k == HlKind::Operator));

        let s = kinds_of("\"hello\"");
        assert_eq!(s[0], (HlKind::Str, "\"hello\"".into()));

        let c = kinds_of("// a comment");
        assert_eq!(c[0].0, HlKind::Comment);

        let d = kinds_of("@NPU");
        assert_eq!(d[0], (HlKind::Decorator, "@NPU".into()));
    }

    #[test]
    fn keyword_vs_ident_boundary() {
        // "letter" is an identifier, not the `let` keyword.
        let got = kinds_of("letter");
        assert_eq!(got[0], (HlKind::Ident, "letter".into()));
    }
}

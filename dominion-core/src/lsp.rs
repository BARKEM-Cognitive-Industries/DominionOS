//! Dominion **Language Server** core — the one missing tool in the AR toolchain line
//! (`docs/language/dominion-language-reference.md`; formatter `lang::emit`, DST time-travel
//! debugger `dst`/`state`, test framework `testkit`, package manager `packaging` already ship).
//!
//! This is the editor-agnostic logic behind hover / completion / signature-help /
//! diagnostics / document-symbols — pure functions over source text, so an IDE
//! (the shell's `ide.rs`) or an external LSP transport can drive it. It depends only on
//! the **stable** language surface ([`crate::lang::parse_source`] + [`crate::lang::ParseError`]),
//! a lightweight lexical scan, and the cross-system [`crate::discovery`] catalog (packages,
//! libraries, drivers, programs, sub-nodes), so it is resilient to the evolving grammar.
//! Pure, safe `no_std`, host-tested.

use crate::lang::parse_source;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Diagnostic severity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
    Hint,
}

/// One diagnostic over a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub line: u32,
    pub severity: Severity,
    pub message: String,
}

/// The Dominion keywords offered in completion + recognised by hover.
pub const KEYWORDS: &[&str] = &[
    "let", "fn", "cell", "object", "if", "else", "while", "for", "break", "continue", "linear",
    "return", "cap", "in", "true", "false",
];

/// Built-in functions and the extended-type constructors — the COMPLETE current set, kept
/// in lock-step with `lang::interp::try_builtin` (verify against that match when builtins land).
/// Sorted so completion ordering is deterministic regardless of source churn.
pub const BUILTINS: &[&str] = &[
    "abs", "bigint", "bind", "cabs", "ceil", "chars", "clamp", "complex", "concat", "conj",
    "contains", "count_matches",
    "dconst", "dec_div", "dec_round", "dec_sqrt", "decimal", "dpow", "dsqrt", "dual", "dvar",
    "ends_with", "enumerate", "first", "flatten", "float", "floor",
    "get", "hash", "hypervector", "icontains", "ihull",
    "int", "interval", "join", "keys", "last", "len", "lerp", "lines", "load", "lower",
    "matmul", "max", "min", "mlp",
    "nn_loss", "pad_left", "pad_right", "pow", "predict", "print", "product", "push",
    "qnorm", "qnormalize", "quat",
    "range", "rational", "repeat_str", "replace", "reverse", "round", "route",
    "sign", "slice", "sort", "split", "sqrt", "starts_with", "str", "sum", "sum_f",
    "summarise", "tensor", "to_decimal", "train_xor", "trim", "unique", "upper", "values",
    "zip",
];

/// Namespaced path roots and the members each exposes (the `Ns::member` builtins in
/// `interp::call_path`). Drives `::`-context completion.
pub const PATH_MEMBERS: &[(&str, &[&str])] = &[
    (
        "Driver",
        &[
            "list", "inspect", "ops", "wellformed", "set_base", "set_irq", "set_reg", "invoke",
        ],
    ),
    ("NeuralCodec", &["encode"]),
    ("SystemGraph", &["commit"]),
];

/// A completion candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub label: String,
    pub kind: CompletionKind,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompletionKind {
    Keyword,
    Builtin,
    /// An identifier already defined in the document (top-level def, local, or param).
    Symbol,
    /// A `Ns::member` path member (namespaced builtin).
    PathMember,
    /// A cross-system catalog item surfaced from [`crate::discovery`] (package / library /
    /// driver / program / sub-node). The kind is carried in the label, e.g. `"AHCI — driver"`.
    Catalog,
}

/// A function signature, for signature-help / parameter hints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub name: String,
    pub params: Vec<String>,
    pub doc: String,
}

/// A document symbol (top-level definition), found by a resilient lexical scan rather than an
/// AST walk (so it survives grammar churn).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub line: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SymbolKind {
    Function,
    Cell,
    Binding,
}

/// The language server over Dominion source.
pub struct Lsp;

impl Lsp {
    /// Diagnostics for a document: a parse failure becomes an error at its reported line, plus
    /// a few cheap, deterministic lints that never fire on a valid program. An empty result
    /// means the document parses cleanly with nothing to flag.
    pub fn diagnostics(src: &str) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        if let Err(e) = parse_source(src) {
            out.push(Diagnostic { line: e.line, severity: Severity::Error, message: e.message });
        }
        for (i, line) in src.lines().enumerate() {
            let ln = i as u32 + 1;
            // Lint: trailing-whitespace hint (cheap, deterministic).
            if line.len() != line.trim_end().len() {
                out.push(Diagnostic {
                    line: ln,
                    severity: Severity::Hint,
                    message: String::from("trailing whitespace"),
                });
            }
            // Lint: tab indentation — Dominion sources indent with spaces; a leading tab is a
            // style smell, never a syntax error (the lexer treats it as whitespace).
            if line.starts_with('\t') {
                out.push(Diagnostic {
                    line: ln,
                    severity: Severity::Hint,
                    message: String::from("tab indentation (prefer spaces)"),
                });
            }
            // Lint: an empty block `{}` reads as dead code. Match only the bare, balanced
            // `{}` token (optionally spaced) so we never false-positive on a real block.
            let t = line.trim();
            if t.ends_with("{}") || t.ends_with("{ }") {
                out.push(Diagnostic {
                    line: ln,
                    severity: Severity::Hint,
                    message: String::from("empty block"),
                });
            }
        }
        out
    }

    /// Whether a document parses cleanly (no error-severity diagnostics).
    pub fn is_valid(src: &str) -> bool {
        parse_source(src).is_ok()
    }

    /// Completions for `prefix`: keywords, builtins, document-defined symbols, and
    /// cross-system catalog items whose names start with the prefix. Backwards-compatible
    /// entry point — case-sensitive prefix match, deterministic order, de-duplicated.
    /// Prefer [`Lsp::complete_at`] for context-aware (scope / `::`) completion.
    pub fn complete(src: &str, prefix: &str) -> Vec<Completion> {
        let mut out = Vec::new();
        Self::push_global(&mut out, prefix);
        for sym in Self::document_symbols(src) {
            if sym.name.starts_with(prefix) {
                out.push(Completion { label: sym.name, kind: CompletionKind::Symbol });
            }
        }
        Self::push_catalog(&mut out, prefix);
        dedup_completions(&mut out);
        out
    }

    /// Context-aware completion at a caret position (1-based `line`, 0-based `col`).
    ///
    /// - Extracts the identifier prefix immediately left of the caret.
    /// - After a `Ns::` it offers that namespace's path members.
    /// - Otherwise (expression position) it offers locals/params in scope (lexical scan of the
    ///   enclosing `fn` plus top-level `let`/`linear` bindings), then keywords, then builtins,
    ///   then cross-system catalog items.
    /// - Ranks exact-prefix (case-sensitive) matches first, then case-insensitive, in a
    ///   stable deterministic order, de-duplicated by label.
    /// - Returns empty when the caret is inside a string literal.
    pub fn complete_at(src: &str, line: u32, col: u32) -> Vec<Completion> {
        let lines: Vec<&str> = src.lines().collect();
        let row = line.saturating_sub(1) as usize;
        let cur = lines.get(row).copied().unwrap_or("");
        let col = col as usize;

        // Do not offer completions inside a string literal.
        if is_in_string(cur, col) {
            return Vec::new();
        }

        // `Ns::` path context — offer the namespace's members.
        if let Some((ns, member_prefix)) = path_context(cur, col) {
            let mut out = Vec::new();
            for &(name, members) in PATH_MEMBERS {
                if name == ns {
                    for &m in members {
                        if prefix_matches(m, &member_prefix) {
                            out.push(Completion {
                                label: m.to_string(),
                                kind: CompletionKind::PathMember,
                            });
                        }
                    }
                }
            }
            rank_completions(&mut out, &member_prefix);
            dedup_completions(&mut out);
            return out;
        }

        let prefix = Self::word_prefix(cur, col);
        let mut out = Vec::new();

        // Locals / params in scope first — what the author most likely means.
        for local in locals_in_scope(src, row) {
            if prefix_matches(&local, &prefix) {
                out.push(Completion { label: local, kind: CompletionKind::Symbol });
            }
        }
        // Top-level document symbols (functions, cells, top-level bindings).
        for sym in Self::document_symbols(src) {
            if prefix_matches(&sym.name, &prefix) {
                out.push(Completion { label: sym.name, kind: CompletionKind::Symbol });
            }
        }
        // Keywords + builtins.
        for &k in KEYWORDS {
            if prefix_matches(k, &prefix) {
                out.push(Completion { label: k.to_string(), kind: CompletionKind::Keyword });
            }
        }
        for &b in BUILTINS {
            if prefix_matches(b, &prefix) {
                out.push(Completion { label: b.to_string(), kind: CompletionKind::Builtin });
            }
        }
        // Namespace roots themselves (so typing `Dri` offers `Driver`).
        for &(ns, _) in PATH_MEMBERS {
            if prefix_matches(ns, &prefix) {
                out.push(Completion { label: ns.to_string(), kind: CompletionKind::PathMember });
            }
        }
        // Cross-system discovery catalog.
        Self::push_catalog(&mut out, &prefix);

        rank_completions(&mut out, &prefix);
        dedup_completions(&mut out);
        out
    }

    /// Push the global keyword + builtin completions for `prefix` (case-sensitive prefix).
    fn push_global(out: &mut Vec<Completion>, prefix: &str) {
        for &k in KEYWORDS {
            if k.starts_with(prefix) {
                out.push(Completion { label: k.to_string(), kind: CompletionKind::Keyword });
            }
        }
        for &b in BUILTINS {
            if b.starts_with(prefix) {
                out.push(Completion { label: b.to_string(), kind: CompletionKind::Builtin });
            }
        }
    }

    /// Fold the cross-system discovery catalog into completion: each [`crate::discovery::CatalogItem`]
    /// becomes a `Catalog` completion whose label carries the item kind (e.g. `"numpy — library"`,
    /// `"AHCI — driver"`), so packages / drivers / libraries / programs / sub-nodes are all
    /// reachable from autocomplete. Resilient to the catalog being transiently empty.
    fn push_catalog(out: &mut Vec<Completion>, prefix: &str) {
        for item in crate::discovery::search(prefix) {
            out.push(Completion { label: catalog_label(&item), kind: CompletionKind::Catalog });
        }
    }

    /// The identifier prefix immediately *before* a caret column on a line — the word
    /// being typed, used to drive completion. Walks left from `col` over `[A-Za-z0-9_]`
    /// runs. Returns `""` when the caret isn't after an identifier char.
    pub fn word_prefix(line: &str, col: usize) -> String {
        let chars: Vec<char> = line.chars().collect();
        let end = col.min(chars.len());
        let mut start = end;
        while start > 0 {
            let c = chars[start - 1];
            if c.is_ascii_alphanumeric() || c == '_' {
                start -= 1;
            } else {
                break;
            }
        }
        chars[start..end].iter().collect()
    }

    /// Signature help for a builtin / namespaced builtin: its parameter names + a one-line doc.
    /// Arities mirror `lang::interp` (e.g. `range` accepts `range(n)` or `range(start, end)`;
    /// the canonical form reported here is `range(n)`). Returns `None` for unknown names.
    pub fn signature(name: &str) -> Option<Signature> {
        let (params, doc): (&[&str], &str) = match name {
            // core / collections
            "print" => (&["..values"], "print values separated by spaces"),
            "len" => (&["xs"], "length of a Vector or Str"),
            "push" => (&["xs", "item"], "append `item`, returning a new Vector"),
            "sum" => (&["xs"], "sum of a Vector of Int"),
            "product" => (&["xs"], "product of a Vector of Int"),
            "range" => (&["n"], "0..n as a Vector (or use directly as a `for` iterator)"),
            "get" => (&["xs", "i"], "element at index `i` (negative indexes from the end)"),
            "first" => (&["xs"], "first element of a non-empty Vector"),
            "last" => (&["xs"], "last element of a non-empty Vector"),
            "reverse" => (&["xs"], "reverse a Vector or Str"),
            "concat" => (&["a", "b"], "concatenate two Vectors or two Strs"),
            "slice" => (&["xs", "start", "end"], "sub-Vector [start, end) (clamped)"),
            "sort" => (&["xs"], "sorted Vector (numeric, else by display)"),
            "contains" => (&["xs", "item"], "membership in a Vector / substring in a Str"),
            // data / ML
            "load" => (&["name"], "open a named dataset, returning a numeric series"),
            "summarise" => (&["series"], "reduce a numeric Vector to count/sum/mean/min/max"),
            "hash" => (&["value"], "short content hash of any value"),
            "tensor" => (&["rows", "cols", "data"], "an N-D tensor value (GPU-routed)"),
            "matmul" => (&["a", "b"], "matrix multiply two Tensors (accelerator-routed)"),
            "hypervector" => (&["dim", "seed"], "a random hyperdimensional vector (NPU-routed)"),
            "bind" => (&["a", "b"], "bind two HyperVectors (HDC)"),
            "mlp" => (&["sizes"], "build an MLP from a Vector of layer sizes, e.g. [2, 8, 1]"),
            "predict" => (&["model", "x"], "run inference; returns the output Tensor"),
            "train_xor" => (&["model", "epochs"], "train a model on XOR for `epochs` steps"),
            "nn_loss" => (&["model"], "the model's current mean-squared error on XOR"),
            "route" => (&["value"], "record the type-directed hardware placement decision"),
            // math
            "abs" => (&["x"], "absolute value of a number"),
            "min" => (&["..xs"], "minimum of the arguments (or of a single Vector)"),
            "max" => (&["..xs"], "maximum of the arguments (or of a single Vector)"),
            "floor" => (&["x"], "round down to an integer-valued number"),
            "ceil" => (&["x"], "round up to an integer-valued number"),
            "round" => (&["x"], "round to the nearest integer-valued number"),
            "sqrt" => (&["x"], "square root (as a Float)"),
            "pow" => (&["base", "exp"], "base raised to an integer exponent"),
            // conversions
            "str" => (&["value"], "format any value as a Str"),
            "int" => (&["value"], "convert a number / bool / string to Int"),
            "float" => (&["value"], "convert a number / string to Float"),
            // strings
            "upper" => (&["s"], "uppercase a Str"),
            "lower" => (&["s"], "lowercase a Str"),
            "trim" => (&["s"], "trim surrounding whitespace from a Str"),
            "split" => (&["s", "sep"], "split a Str by a separator into a Vector"),
            "join" => (&["xs", "sep"], "join a Vector into a Str with a separator"),
            "chars" => (&["s"], "the characters of a Str as a Vector of single-char Strs"),
            "starts_with" => (&["s", "prefix"], "whether a Str starts with `prefix`"),
            "ends_with" => (&["s", "suffix"], "whether a Str ends with `suffix`"),
            "replace" => (&["s", "from", "to"], "replace all `from` with `to` in a Str"),
            // decimal / bignum
            "decimal" => (&["value"], "a high-precision Decimal from a string/int/float"),
            "dec_div" => (&["a", "b", "precision"], "Decimal division to `precision` digits"),
            "dec_sqrt" => (&["a", "precision"], "Decimal square root to `precision` digits"),
            "dec_round" => (&["a", "digits"], "round a Decimal to `digits` places"),
            "bigint" => (&["value"], "an arbitrary-precision integer from a string/int"),
            "rational" => (&["num", "den"], "an exact rational num/den"),
            "to_decimal" => (&["r", "precision"], "a Rational as a Decimal to `precision`"),
            // complex / dual / interval / quaternion
            "complex" => (&["re", "im"], "a complex number re + im·i"),
            "conj" => (&["x"], "complex / quaternion conjugate"),
            "cabs" => (&["c"], "modulus (magnitude) of a Complex"),
            "dual" => (&["value", "deriv"], "a dual number for forward-mode autodiff"),
            "dvar" => (&["x"], "a dual variable (derivative 1)"),
            "dconst" => (&["x"], "a dual constant (derivative 0)"),
            "dsqrt" => (&["d"], "square root of a Dual"),
            "dpow" => (&["d", "exp"], "a Dual raised to an integer exponent"),
            "interval" => (&["lo", "hi"], "an interval [lo, hi] for interval arithmetic"),
            "ihull" => (&["a", "b"], "the convex hull of two Intervals"),
            "icontains" => (&["iv", "x"], "whether an Interval contains a number"),
            "quat" => (&["w", "x", "y", "z"], "a quaternion w + xi + yj + zk"),
            "qnorm" => (&["q"], "the norm (magnitude) of a Quaternion"),
            "qnormalize" => (&["q"], "the unit quaternion in the same direction"),
            // string extras
            "pad_left" => (&["s", "width", "pad_char"], "left-pad s to width with pad_char"),
            "pad_right" => (&["s", "width", "pad_char"], "right-pad s to width with pad_char"),
            "repeat_str" => (&["s", "n"], "repeat Str s n times"),
            "lines" => (&["s"], "split a Str on newlines into a Vector"),
            // collection extras
            "flatten" => (&["xs"], "flatten one level of nesting in a Vector"),
            "zip" => (&["a", "b"], "zip two Vectors into [[a0,b0],[a1,b1],...] (min length)"),
            "enumerate" => (&["xs"], "pairs each element with its index [[0,v0],[1,v1],...]"),
            "unique" => (&["xs"], "remove duplicates, preserving insertion order"),
            "sum_f" => (&["xs"], "sum of a Vector of numbers as a Float"),
            "count_matches" => (&["xs", "value"], "count occurrences of value in Vector"),
            "keys" => (&["obj"], "field names of an Object as a Vector of Str"),
            "values" => (&["obj"], "field values of an Object as a Vector"),
            // numeric extras
            "clamp" => (&["val", "lo", "hi"], "clamp val to [lo, hi]"),
            "lerp" => (&["a", "b", "t"], "linear interpolation: a + t*(b-a)"),
            "sign" => (&["x"], "sign of a number: -1, 0, or 1"),
            // namespaced builtins
            // polyglot
            "Lang::call" => (&["lang", "src", "fn_name", "..args"], "call a named function in another language with typed Dominion args (Execute)"),
            // namespaced builtins
            "Driver::list" => (&[], "the registered driver names (Read)"),
            "Driver::inspect" => (&["name"], "an editable Object view of a driver spec (Read)"),
            "Driver::ops" => (&["name"], "the operations a driver implements (Read)"),
            "Driver::wellformed" => (&["name"], "whether a driver spec would bind (Read)"),
            "Driver::set_base" => (&["name", "addr"], "relocate the MMIO window, re-validated (Write)"),
            "Driver::set_irq" => (&["name", "irq"], "change the device IRQ line (Write)"),
            "Driver::set_reg" => (&["name", "reg", "offset"], "move a register in the window (Write)"),
            "Driver::invoke" => (&["name", "op"], "bind + run an operation (Execute)"),
            "NeuralCodec::encode" => (&["x"], "encode a value to a Latent (content hash + ratio)"),
            "SystemGraph::commit" => (&["records"], "persist records, returning the new state root"),
            _ => return None,
        };
        Some(Signature {
            name: name.to_string(),
            params: params.iter().map(|p| p.to_string()).collect(),
            doc: doc.to_string(),
        })
    }

    /// Hover documentation for a word (keyword, builtin, or `Ns::member`), or `None`.
    /// Every builtin is documented from its [`Lsp::signature`]; keywords have hand-written docs.
    pub fn hover(word: &str) -> Option<String> {
        if let Some(kw) = keyword_doc(word) {
            return Some(kw.to_string());
        }
        let sig = Self::signature(word)?;
        let params = sig.params.join(", ");
        Some(alloc::format!("`{}({})` — {}", sig.name, params, sig.doc))
    }

    /// Top-level document symbols, found by a lexical scan for `fn` / `cell` / `let` / `linear`
    /// openings (so it survives grammar churn).
    pub fn document_symbols(src: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (i, line) in src.lines().enumerate() {
            let t = line.trim_start();
            let (kw, kind) = if let Some(r) = t.strip_prefix("fn ") {
                (r, SymbolKind::Function)
            } else if let Some(r) = t.strip_prefix("cell ") {
                (r, SymbolKind::Cell)
            } else if let Some(r) = t.strip_prefix("let ") {
                (r, SymbolKind::Binding)
            } else if let Some(r) = t.strip_prefix("linear ") {
                (r, SymbolKind::Binding)
            } else {
                continue;
            };
            if let Some(name) = ident_at_start(kw) {
                out.push(Symbol { name, kind, line: i as u32 + 1 });
            }
        }
        out
    }
}

/// Hand-written hover docs for keywords (and the namespace roots).
fn keyword_doc(word: &str) -> Option<&'static str> {
    Some(match word {
        "let" => "`let x = e` — bind a value (immutable).",
        "linear" => "`linear x = e` — an affine binding: moved on use, cryptographically invalidated at scope end (no GC).",
        "fn" => "`fn name(args) { … }` — define a function (lowers to a capability-checked DCG).",
        "cell" => "`cell name { … }` — a capability-gated unit of modularity; hot-swappable, restartable.",
        "object" => "`object Kind { … }` — define a structured object type.",
        "cap" => "`cap` — declare a capability requirement on a cell / function.",
        "if" => "`if cond { … } else { … }` — conditional.",
        "else" => "`else { … }` / `else if cond { … }` — the alternative branch of an `if`.",
        "while" => "`while cond { … }` — loop while the condition holds.",
        "for" => "`for x in iter { … }` — iterate.",
        "in" => "`for x in iter` — the iterator keyword inside a `for`.",
        "break" => "`break` — exit the innermost loop.",
        "continue" => "`continue` — skip to the next iteration of the innermost loop.",
        "return" => "`return e` — return a value from a function.",
        "true" => "`true` — the boolean literal.",
        "false" => "`false` — the boolean literal.",
        "Driver" => "`Driver::*` — the capability-gated driver registry (list/inspect/edit/invoke).",
        "NeuralCodec" => "`NeuralCodec::encode(x)` — encode a value into a neural Latent.",
        "SystemGraph" => "`SystemGraph::commit(records)` — persist records into the semantic graph.",
        _ => return None,
    })
}

/// Build a catalog completion label that carries the item kind, e.g. `"AHCI — driver"`.
fn catalog_label(item: &crate::discovery::CatalogItem) -> String {
    let kind = match item.kind {
        crate::discovery::ItemKind::Package => "package",
        crate::discovery::ItemKind::Library => "library",
        crate::discovery::ItemKind::Driver => "driver",
        crate::discovery::ItemKind::Program => "program",
        crate::discovery::ItemKind::SubNode => "sub-node",
        crate::discovery::ItemKind::Builtin => "builtin",
    };
    alloc::format!("{} — {}", item.name, kind)
}

/// Whether `candidate` is a completion of `prefix` — exact prefix or case-insensitive prefix.
/// An empty prefix matches everything (offer the full menu).
fn prefix_matches(candidate: &str, prefix: &str) -> bool {
    if prefix.is_empty() {
        return true;
    }
    if candidate.starts_with(prefix) {
        return true;
    }
    candidate.to_ascii_lowercase().starts_with(&prefix.to_ascii_lowercase())
}

/// Rank: exact-prefix (case-sensitive) matches first, then the rest (case-insensitive), each
/// group kept in its existing deterministic insertion order (a stable sort by a 0/1 key).
fn rank_completions(out: &mut [Completion], prefix: &str) {
    if prefix.is_empty() {
        return;
    }
    out.sort_by_key(|c| if c.label.starts_with(prefix) { 0u8 } else { 1u8 });
}

/// Whether the caret at `col` (character index) on `line` is inside a string literal.
/// Counts unescaped `"` characters left of the caret; an odd count means inside a string.
fn is_in_string(line: &str, col: usize) -> bool {
    let chars: Vec<char> = line.chars().take(col).collect();
    let mut in_str = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && in_str {
            i += 2; // skip escaped char
            continue;
        }
        if chars[i] == '"' {
            in_str = !in_str;
        }
        i += 1;
    }
    in_str
}

/// Remove duplicate completions by label, keeping the first (highest-ranked) occurrence.
/// Linear scan against an accumulator — input sizes are small (menu length), and it preserves
/// order, which a sort-based dedup would not.
fn dedup_completions(out: &mut Vec<Completion>) {
    let mut seen: Vec<String> = Vec::new();
    out.retain(|c| {
        if seen.iter().any(|s| s == &c.label) {
            false
        } else {
            seen.push(c.label.clone());
            true
        }
    });
}

/// If the caret sits right after a `Ns::` (optionally followed by a partial member name),
/// return `(namespace, member_prefix)`. Otherwise `None`.
fn path_context(line: &str, col: usize) -> Option<(String, String)> {
    let chars: Vec<char> = line.chars().collect();
    let end = col.min(chars.len());
    // Walk left over the member prefix (identifier chars).
    let mut m = end;
    while m > 0 {
        let c = chars[m - 1];
        if c.is_ascii_alphanumeric() || c == '_' {
            m -= 1;
        } else {
            break;
        }
    }
    // Require `::` immediately to the left of the member prefix.
    if m < 2 || chars[m - 1] != ':' || chars[m - 2] != ':' {
        return None;
    }
    // Walk left over the namespace identifier.
    let mut n = m - 2;
    let ns_end = n;
    while n > 0 {
        let c = chars[n - 1];
        if c.is_ascii_alphanumeric() || c == '_' {
            n -= 1;
        } else {
            break;
        }
    }
    if n == ns_end {
        return None;
    }
    let ns: String = chars[n..ns_end].iter().collect();
    let member_prefix: String = chars[m..end].iter().collect();
    Some((ns, member_prefix))
}

/// Locals + params visible at `row` (0-based line index): the parameters of the enclosing `fn`
/// (if any) plus every `let`/`linear` binding declared on a line at or before `row`. A
/// brace-depth scan finds the enclosing function so we don't leak one function's params into
/// another. Deterministic order: enclosing-fn params first, then bindings in source order.
fn locals_in_scope(src: &str, row: usize) -> Vec<String> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<String> = Vec::new();

    // Find the nearest enclosing `fn` header line by scanning upward and tracking brace depth:
    // a `fn ... {` whose body still encloses `row` is the one whose `{` opened before `row` and
    // whose matching `}` is at/after `row`. A simple, churn-resilient heuristic: the last `fn`
    // header on or before `row` whose opening brace's block has not closed before `row`.
    let mut depth: i32 = 0;
    let mut enclosing_fn: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        if i > row {
            break;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("fn ") && depth == 0 {
            enclosing_fn = Some(i);
        }
        for c in line.chars() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth <= 0 {
                        depth = 0;
                        // A top-level block closed; any fn we were in has ended (unless this
                        // line is still before its own header, which `depth==0` guards above).
                        enclosing_fn = enclosing_fn.filter(|&f| f >= i);
                    }
                }
                _ => {}
            }
        }
    }

    // Enclosing-fn parameters.
    if let Some(f) = enclosing_fn {
        for p in fn_params(lines[f]) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }

    // `let` / `linear` bindings declared at or before `row`.
    for line in lines.iter().take(row + 1) {
        let t = line.trim_start();
        let rest = t
            .strip_prefix("let ")
            .or_else(|| t.strip_prefix("linear "));
        if let Some(rest) = rest {
            if let Some(name) = ident_at_start(rest) {
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
    }
    out
}

/// Extract parameter names from a `fn name(a, b, c) { … }` header line.
fn fn_params(header: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (open, close) = match (header.find('('), header.find(')')) {
        (Some(o), Some(c)) if c > o => (o, c),
        _ => return out,
    };
    for part in header[open + 1..close].split(',') {
        if let Some(name) = ident_at_start(part.trim_start()) {
            out.push(name);
        }
    }
    out
}

/// Read a leading identifier (`[A-Za-z_][A-Za-z0-9_]*`) from the start of `s`.
fn ident_at_start(s: &str) -> Option<String> {
    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    let mut end = first.len_utf8();
    for (i, c) in s.char_indices().skip(1) {
        if c.is_ascii_alphanumeric() || c == '_' {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    Some(String::from(&s[..end]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_source_has_no_error_diagnostics() {
        let diags = Lsp::diagnostics("let x = 2 + 2");
        assert!(!diags.iter().any(|d| d.severity == Severity::Error));
        assert!(Lsp::is_valid("let x = 2 + 2"));
    }

    #[test]
    fn parse_error_becomes_a_diagnostic() {
        let diags = Lsp::diagnostics("let = 5");
        assert!(diags.iter().any(|d| d.severity == Severity::Error));
        assert!(!Lsp::is_valid("let = 5"));
    }

    #[test]
    fn trailing_whitespace_is_a_hint() {
        let diags = Lsp::diagnostics("let x = 1   \nlet y = 2");
        assert!(diags.iter().any(|d| d.severity == Severity::Hint && d.line == 1));
    }

    #[test]
    fn completion_offers_keywords_builtins_and_symbols() {
        // "te" → builtin "tensor".
        let c = Lsp::complete("", "te");
        assert!(c.iter().any(|x| x.label == "tensor" && x.kind == CompletionKind::Builtin));
        // "l" → keyword "let"/"linear".
        let c2 = Lsp::complete("", "l");
        assert!(c2.iter().any(|x| x.label == "let" && x.kind == CompletionKind::Keyword));
    }

    #[test]
    fn completion_includes_document_symbols() {
        let src = "fn helper() { 1 }\nlet total = 5";
        let c = Lsp::complete(src, "hel");
        assert!(c.iter().any(|x| x.label == "helper" && x.kind == CompletionKind::Symbol));
    }

    #[test]
    fn word_prefix_extracts_the_token_before_the_caret() {
        assert_eq!(Lsp::word_prefix("matmul(ten", 10), "ten");
        assert_eq!(Lsp::word_prefix("a + ", 4), "");
        assert_eq!(Lsp::word_prefix("hello", 5), "hello");
        assert_eq!(Lsp::word_prefix("hello", 3), "hel");
    }

    #[test]
    fn hover_documents_keywords_and_builtins() {
        assert!(Lsp::hover("linear").unwrap().contains("affine"));
        assert!(Lsp::hover("tensor").unwrap().contains("tensor"));
        assert!(Lsp::hover("sqrt").unwrap().contains("square root"));
        assert!(Lsp::hover("not-a-word").is_none());
    }

    #[test]
    fn document_symbols_finds_definitions() {
        let src = "fn add(a, b) { a + b }\ncell worker { 0 }\nlet k = 3";
        let syms = Lsp::document_symbols(src);
        assert_eq!(syms.len(), 3);
        assert_eq!(syms[0].name, "add");
        assert_eq!(syms[0].kind, SymbolKind::Function);
        assert_eq!(syms[1].kind, SymbolKind::Cell);
        assert_eq!(syms[2].name, "k");
        assert_eq!(syms[2].line, 3);
    }

    // ---- new coverage ----

    #[test]
    fn every_builtin_is_offered_for_its_own_prefix() {
        for &b in BUILTINS {
            let c = Lsp::complete("", b);
            assert!(
                c.iter().any(|x| x.label == b && x.kind == CompletionKind::Builtin),
                "builtin `{}` was not offered by completion",
                b
            );
        }
    }

    #[test]
    fn every_keyword_is_offered_for_its_own_prefix() {
        for &k in KEYWORDS {
            let c = Lsp::complete("", k);
            assert!(
                c.iter().any(|x| x.label == k && x.kind == CompletionKind::Keyword),
                "keyword `{}` was not offered by completion",
                k
            );
        }
    }

    #[test]
    fn signature_reports_arity() {
        let s = Lsp::signature("matmul").unwrap();
        assert_eq!(s.params.len(), 2);
        assert_eq!(s.name, "matmul");
        assert_eq!(Lsp::signature("slice").unwrap().params.len(), 3);
        assert_eq!(Lsp::signature("sqrt").unwrap().params.len(), 1);
        assert_eq!(Lsp::signature("quat").unwrap().params.len(), 4);
        assert!(Lsp::signature("not-a-builtin").is_none());
    }

    #[test]
    fn every_builtin_has_a_signature() {
        for &b in BUILTINS {
            assert!(Lsp::signature(b).is_some(), "builtin `{}` has no signature", b);
        }
    }

    #[test]
    fn complete_at_offers_locals_in_scope() {
        let src = "fn f(width, height) {\n  let area = width\n  ar\n}";
        // Caret after `ar` on line 3 (1-based), col 4.
        let c = Lsp::complete_at(src, 3, 4);
        assert!(c.iter().any(|x| x.label == "area" && x.kind == CompletionKind::Symbol));
        // Params are also in scope.
        let c2 = Lsp::complete_at("fn f(width) {\n  wi\n}", 2, 4);
        assert!(c2.iter().any(|x| x.label == "width" && x.kind == CompletionKind::Symbol));
    }

    #[test]
    fn complete_at_offers_path_members_after_colon_colon() {
        let src = "Driver::";
        let c = Lsp::complete_at(src, 1, 8);
        assert!(c.iter().any(|x| x.label == "list" && x.kind == CompletionKind::PathMember));
        assert!(c.iter().any(|x| x.label == "invoke" && x.kind == CompletionKind::PathMember));
        // Partial member prefix filters.
        let c2 = Lsp::complete_at("Driver::set_", 1, 12);
        assert!(c2.iter().all(|x| x.label.starts_with("set_")));
        assert!(c2.iter().any(|x| x.label == "set_base"));
    }

    #[test]
    fn path_namespaced_builtins_have_signatures() {
        assert_eq!(Lsp::signature("Driver::invoke").unwrap().params.len(), 2);
        assert!(Lsp::hover("Driver::list").unwrap().contains("driver"));
    }

    #[test]
    fn discovery_results_are_mapped_into_completion() {
        // The live catalog may be transiently empty (it is filled by another agent), so we do
        // NOT require non-empty results here — we assert the *mapping* shape on whatever the
        // catalog returns, and assert the label builder directly on a synthetic item.
        for c in Lsp::complete("", "") {
            if c.kind == CompletionKind::Catalog {
                assert!(c.label.contains(" — "), "catalog label `{}` lacks a kind suffix", c.label);
            }
        }
        let item = crate::discovery::CatalogItem {
            name: String::from("numpy"),
            kind: crate::discovery::ItemKind::Library,
            summary: String::from("numerical arrays"),
        };
        assert_eq!(catalog_label(&item), "numpy — library");
        let drv = crate::discovery::CatalogItem {
            name: String::from("AHCI"),
            kind: crate::discovery::ItemKind::Driver,
            summary: String::new(),
        };
        assert_eq!(catalog_label(&drv), "AHCI — driver");
    }

    #[test]
    fn complete_at_ranks_exact_prefix_first_and_dedups() {
        // With prefix "s", builtins starting with "s" should all be present, exact-prefix first,
        // and no label should appear twice.
        let c = Lsp::complete_at("s", 1, 1);
        let labels: Vec<&str> = c.iter().map(|x| x.label.as_str()).collect();
        // de-dup: unique labels.
        for (i, l) in labels.iter().enumerate() {
            assert!(!labels[i + 1..].contains(l), "duplicate completion label `{}`", l);
        }
        // exact-prefix grouping: every leading entry that starts with "s" precedes any that
        // doesn't (there should be no non-"s" entry before an "s" entry).
        let mut seen_non_s = false;
        for l in &labels {
            if l.starts_with('s') {
                assert!(!seen_non_s, "exact-prefix `{}` appeared after a non-prefix match", l);
            } else {
                seen_non_s = true;
            }
        }
    }

    #[test]
    fn lints_flag_tab_indent_and_empty_block_without_false_positives() {
        let diags = Lsp::diagnostics("\tlet x = 1");
        assert!(diags.iter().any(|d| d.message.contains("tab")));
        let empty = Lsp::diagnostics("fn f() {}");
        assert!(empty.iter().any(|d| d.message.contains("empty block")));
        // A valid program with a real block must NOT be flagged for an empty block.
        let ok = Lsp::diagnostics("fn f() { 1 }");
        assert!(!ok.iter().any(|d| d.message.contains("empty block")));
    }

    #[test]
    fn complete_is_deterministic() {
        let a = Lsp::complete("let foo = 1", "f");
        let b = Lsp::complete("let foo = 1", "f");
        assert_eq!(a, b);
    }

    #[test]
    fn complete_at_returns_empty_inside_string_literal() {
        // Caret is after `"hel` — inside a string; no completions should be offered.
        // Line: `let x = "hel`  (col 13 = after the 'l' inside the string)
        let src = r#"let x = "hel"#;
        let c = Lsp::complete_at(src, 1, 13);
        assert!(c.is_empty(), "expected no completions inside a string, got {:?}", c);

        // Caret outside the string should still get completions.
        // Line: `pr` with no quotes — prefix "pr" → print etc.
        let c2 = Lsp::complete_at("pr", 1, 2);
        assert!(!c2.is_empty(), "expected completions outside a string");

        // A closed string followed by an identifier: `"foo" + pr`
        // col 10 = after "pr" (outside the string).
        let src2 = r#""foo" + pr"#;
        let c3 = Lsp::complete_at(src2, 1, 10);
        assert!(!c3.is_empty(), "expected completions after a closed string");

        // Escaped quote inside string must not terminate the string context.
        // Line: `"say \"hel`  (caret at col 10, still inside string)
        let src3 = r#""say \"hel"#;
        let c4 = Lsp::complete_at(src3, 1, 10);
        assert!(c4.is_empty(), "expected no completions after escaped quote inside string");
    }

    #[test]
    fn is_in_string_detects_string_context() {
        // Basic: odd number of quotes → inside.
        assert!(is_in_string(r#""hello"#, 6));
        // Even number of quotes → outside.
        assert!(!is_in_string(r#""hello" + "#, 10));
        // Escaped quote does not close the string.
        assert!(is_in_string(r#""say \"hi"#, 9));
        // No quotes at all → outside.
        assert!(!is_in_string("let x = 5", 9));
    }
}

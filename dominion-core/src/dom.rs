//! The Document Object Model — the central, mutable tree the whole browser shares.
//!
//! HTML is tokenised and built into a tree of [`NodeRef`]s (`Rc<RefCell<Node>>`), CSS
//! ([`crate::css`]) computes styles over it, JavaScript ([`crate::js`]) mutates it
//! live, and the layout engine ([`crate::html`]) renders whatever the tree currently
//! says. Because every consumer operates on the *same* nodes, a script that sets
//! `element.textContent` or `innerHTML` is reflected on the next repaint — a real
//! DOM, not a one-shot parse.
//!
//! The tokenizer lives here (rather than in the layout module) because both initial
//! parsing and `innerHTML` assignment need it. `script`/`style`/`title`/`textarea`
//! bodies are captured as raw text so the CSS and JS engines can read them.
//!
//! Pure, safe `no_std`. Tree edges are `Rc` (children) + `Weak` (parent) so there are
//! no cycles to leak.

use alloc::rc::{Rc, Weak};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

// ───────────────────────────── nodes ─────────────────────────────

/// A shared, mutable handle to a DOM node.
pub type NodeRef = Rc<RefCell<Node>>;
type WeakNode = Weak<RefCell<Node>>;

/// A DOM node: an element (tag + attributes + children) or a text run.
pub enum Node {
    Element(Element),
    Text(String),
}

/// An element node.
pub struct Element {
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<NodeRef>,
    parent: WeakNode,
    /// Registered event handlers: (event-name, JS source/handler key). The JS engine
    /// stores handler closures keyed elsewhere; here we keep the event names present
    /// so layout/hit-testing knows an element is interactive.
    pub listeners: Vec<String>,
}

impl Node {
    pub fn element(tag: &str) -> NodeRef {
        Rc::new(RefCell::new(Node::Element(Element {
            tag: tag.to_ascii_lowercase(),
            attrs: Vec::new(),
            children: Vec::new(),
            parent: Weak::new(),
            listeners: Vec::new(),
        })))
    }

    pub fn text(s: &str) -> NodeRef {
        Rc::new(RefCell::new(Node::Text(s.to_string())))
    }

    pub fn is_element(&self) -> bool {
        matches!(self, Node::Element(_))
    }

    pub fn as_element(&self) -> Option<&Element> {
        match self {
            Node::Element(e) => Some(e),
            _ => None,
        }
    }
    pub fn as_element_mut(&mut self) -> Option<&mut Element> {
        match self {
            Node::Element(e) => Some(e),
            _ => None,
        }
    }
}

// ───────────────────────── accessors (on NodeRef) ─────────────────────────

/// The lowercased tag name, if `n` is an element.
pub fn tag(n: &NodeRef) -> Option<String> {
    n.borrow().as_element().map(|e| e.tag.clone())
}

/// Read an attribute (case-insensitive name).
pub fn get_attr(n: &NodeRef, name: &str) -> Option<String> {
    let b = n.borrow();
    let e = b.as_element()?;
    e.attrs.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone())
}

/// Set (or replace) an attribute.
pub fn set_attr(n: &NodeRef, name: &str, value: &str) {
    let mut b = n.borrow_mut();
    if let Some(e) = b.as_element_mut() {
        let lname = name.to_ascii_lowercase();
        if let Some(slot) = e.attrs.iter_mut().find(|(k, _)| k.eq_ignore_ascii_case(&lname)) {
            slot.1 = value.to_string();
        } else {
            e.attrs.push((lname, value.to_string()));
        }
    }
}

/// The element id, if any.
pub fn id(n: &NodeRef) -> Option<String> {
    get_attr(n, "id")
}

/// The parent element, if still alive (used by the CSS descendant matcher).
pub fn parent(n: &NodeRef) -> Option<NodeRef> {
    match &*n.borrow() {
        Node::Element(e) => e.parent.upgrade(),
        _ => None,
    }
}

/// The element's class list.
pub fn classes(n: &NodeRef) -> Vec<String> {
    match get_attr(n, "class") {
        Some(c) => c.split_whitespace().map(|s| s.to_string()).collect(),
        None => Vec::new(),
    }
}

/// Append `child` to `parent`, wiring the parent link.
pub fn append_child(parent: &NodeRef, child: &NodeRef) {
    child.borrow_mut().set_parent(parent);
    if let Some(e) = parent.borrow_mut().as_element_mut() {
        e.children.push(child.clone());
    }
}

impl Node {
    fn set_parent(&mut self, parent: &NodeRef) {
        if let Node::Element(e) = self {
            e.parent = Rc::downgrade(parent);
        }
    }
}

/// The concatenated text of a node and all its descendants.
pub fn text_content(n: &NodeRef) -> String {
    let mut out = String::new();
    collect_text(n, &mut out);
    out
}

fn collect_text(n: &NodeRef, out: &mut String) {
    match &*n.borrow() {
        Node::Text(t) => out.push_str(t),
        Node::Element(e) => {
            for c in &e.children {
                collect_text(c, out);
            }
        }
    }
}

/// Replace a node's children with a single text node (the `textContent` setter).
pub fn set_text_content(n: &NodeRef, s: &str) {
    if let Some(e) = n.borrow_mut().as_element_mut() {
        e.children.clear();
        e.children.push(Node::text(s));
    }
}

/// Serialise a node's children back to HTML (the `innerHTML` getter).
pub fn inner_html(n: &NodeRef) -> String {
    let mut out = String::new();
    if let Some(e) = n.borrow().as_element() {
        for c in &e.children {
            serialize(c, &mut out);
        }
    }
    out
}

/// Parse `html` and replace `n`'s children with the result (the `innerHTML` setter).
pub fn set_inner_html(n: &NodeRef, html: &str) {
    let kids = build_fragment(html);
    if let Some(e) = n.borrow_mut().as_element_mut() {
        e.children.clear();
    }
    for k in kids {
        append_child(n, &k);
    }
}

fn serialize(n: &NodeRef, out: &mut String) {
    match &*n.borrow() {
        Node::Text(t) => out.push_str(t),
        Node::Element(e) => {
            out.push('<');
            out.push_str(&e.tag);
            for (k, v) in &e.attrs {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                out.push_str(v);
                out.push('"');
            }
            out.push('>');
            for c in &e.children {
                serialize(c, out);
            }
            if !is_void(&e.tag) {
                out.push_str("</");
                out.push_str(&e.tag);
                out.push('>');
            }
        }
    }
}

// ───────────────────────── queries ─────────────────────────

/// Depth-first: the first element with `id`.
pub fn get_element_by_id(root: &NodeRef, target: &str) -> Option<NodeRef> {
    find(root, &mut |n| id(n).as_deref() == Some(target))
}

/// All elements with the given tag name (lowercased).
pub fn get_elements_by_tag(root: &NodeRef, name: &str) -> Vec<NodeRef> {
    let lname = name.to_ascii_lowercase();
    let mut out = Vec::new();
    find_all(root, &mut |n| tag(n).as_deref() == Some(lname.as_str()), &mut out);
    out
}

/// All elements carrying `class`.
pub fn get_elements_by_class(root: &NodeRef, class: &str) -> Vec<NodeRef> {
    let mut out = Vec::new();
    find_all(root, &mut |n| classes(n).iter().any(|c| c == class), &mut out);
    out
}

/// A simple `querySelector`: supports `#id`, `.class`, `tag`, and a compound like
/// `tag.class` / `tag#id` (no combinators — that lives in the CSS matcher).
pub fn query_selector(root: &NodeRef, sel: &str) -> Option<NodeRef> {
    find(root, &mut |n| simple_matches(n, sel))
}

pub fn query_selector_all(root: &NodeRef, sel: &str) -> Vec<NodeRef> {
    let mut out = Vec::new();
    find_all(root, &mut |n| simple_matches(n, sel), &mut out);
    out
}

fn simple_matches(n: &NodeRef, sel: &str) -> bool {
    let sel = sel.trim();
    if !n.borrow().is_element() {
        return false;
    }
    // Split compound selector into its pieces (tag, #id, .class).
    let chars = sel.char_indices().peekable();
    let mut cuts = Vec::new();
    for (i, c) in chars {
        if (c == '.' || c == '#') && i != 0 {
            cuts.push(i);
        }
    }
    let mut parts = Vec::new();
    let mut last = 0;
    for &c in &cuts {
        parts.push(&sel[last..c]);
        last = c;
    }
    parts.push(&sel[last..]);
    for p in parts {
        if p.is_empty() {
            continue;
        }
        let ok = if let Some(cl) = p.strip_prefix('.') {
            classes(n).iter().any(|c| c == cl)
        } else if let Some(i) = p.strip_prefix('#') {
            id(n).as_deref() == Some(i)
        } else if p == "*" {
            true
        } else {
            tag(n).as_deref() == Some(p.to_ascii_lowercase().as_str())
        };
        if !ok {
            return false;
        }
    }
    true
}

fn find(root: &NodeRef, pred: &mut impl FnMut(&NodeRef) -> bool) -> Option<NodeRef> {
    if pred(root) {
        return Some(root.clone());
    }
    let children: Vec<NodeRef> = match root.borrow().as_element() {
        Some(e) => e.children.clone(),
        None => Vec::new(),
    };
    for c in &children {
        if let Some(found) = find(c, pred) {
            return Some(found);
        }
    }
    None
}

fn find_all(root: &NodeRef, pred: &mut impl FnMut(&NodeRef) -> bool, out: &mut Vec<NodeRef>) {
    if pred(root) {
        out.push(root.clone());
    }
    let children: Vec<NodeRef> = match root.borrow().as_element() {
        Some(e) => e.children.clone(),
        None => Vec::new(),
    };
    for c in &children {
        find_all(c, pred, out);
    }
}

/// Visit every element under `root` in document order.
pub fn walk_elements(root: &NodeRef, f: &mut impl FnMut(&NodeRef)) {
    if root.borrow().is_element() {
        f(root);
    }
    let children: Vec<NodeRef> = match root.borrow().as_element() {
        Some(e) => e.children.clone(),
        None => Vec::new(),
    };
    for c in &children {
        walk_elements(c, f);
    }
}

// ───────────────────────── tokenizer ─────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Start { name: String, attrs: Vec<(String, String)>, self_closing: bool },
    End { name: String },
    Text(String),
}

/// Tags whose content is raw text (no nested markup): captured verbatim.
fn is_raw_text(tag: &str) -> bool {
    matches!(tag, "script" | "style" | "textarea" | "title")
}

/// Void elements that never have children or a close tag.
pub fn is_void(tag: &str) -> bool {
    matches!(
        tag,
        "area" | "base" | "br" | "col" | "embed" | "hr" | "img" | "input" | "link" | "meta" | "param" | "source" | "track" | "wbr"
    )
}

/// Tokenise HTML. Comments and the doctype are dropped; raw-text element bodies are
/// emitted as a single (un-decoded for script/style) text token.
pub fn tokenize(input: &str) -> Vec<Token> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let n = bytes.len();
    let mut text = String::new();

    while i < n {
        if bytes[i] == b'<' {
            if !text.is_empty() {
                tokens.push(Token::Text(decode_entities(&text)));
                text.clear();
            }
            if input[i..].starts_with("<!--") {
                if let Some(end) = input[i + 4..].find("-->") {
                    i = i + 4 + end + 3;
                } else {
                    i = n;
                }
                continue;
            }
            if i + 1 < n && (bytes[i + 1] == b'!' || bytes[i + 1] == b'?') {
                if let Some(end) = input[i..].find('>') {
                    i += end + 1;
                } else {
                    i = n;
                }
                continue;
            }
            let close = match input[i..].find('>') {
                Some(c) => i + c,
                None => {
                    text.push_str(&input[i..]);
                    break;
                }
            };
            let raw = &input[i + 1..close];
            i = close + 1;
            if let Some(tok) = parse_tag(raw) {
                let name = tok_name(&tok);
                let is_start = matches!(tok, Token::Start { .. });
                tokens.push(tok);
                if is_start && is_raw_text(&name) {
                    // Capture the raw body up to the matching close tag.
                    let close_seq = ["</", &name].concat();
                    let (body, next) = match find_ci(&input[i..], &close_seq) {
                        Some(end) => {
                            let body = input[i..i + end].to_string();
                            let after = i + end;
                            let next = input[after..].find('>').map(|g| after + g + 1).unwrap_or(n);
                            (body, next)
                        }
                        None => (input[i..].to_string(), n),
                    };
                    let body = if name == "script" || name == "style" { body } else { decode_entities(&body) };
                    if !body.is_empty() {
                        tokens.push(Token::Text(body));
                    }
                    tokens.push(Token::End { name });
                    i = next;
                }
            }
        } else {
            let next = input[i..].find('<').map(|p| i + p).unwrap_or(n);
            text.push_str(&input[i..next]);
            i = next;
        }
    }
    if !text.is_empty() {
        tokens.push(Token::Text(decode_entities(&text)));
    }
    tokens
}

fn tok_name(t: &Token) -> String {
    match t {
        Token::Start { name, .. } | Token::End { name } => name.clone(),
        Token::Text(_) => String::new(),
    }
}

fn parse_tag(raw: &str) -> Option<Token> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(rest) = raw.strip_prefix('/') {
        let name = rest.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
        if name.is_empty() {
            return None;
        }
        return Some(Token::End { name });
    }
    let self_closing = raw.ends_with('/');
    let raw = raw.trim_end_matches('/').trim();
    let mut name_end = raw.len();
    for (idx, c) in raw.char_indices() {
        if c.is_whitespace() {
            name_end = idx;
            break;
        }
    }
    let name = raw[..name_end].to_ascii_lowercase();
    if name.is_empty() {
        return None;
    }
    let attrs = parse_attrs(&raw[name_end..]);
    Some(Token::Start { name, attrs, self_closing })
}

fn parse_attrs(s: &str) -> Vec<(String, String)> {
    let mut attrs = Vec::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let n = b.len();
    while i < n {
        while i < n && b[i].is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        while i < n && !b[i].is_whitespace() && b[i] != '=' {
            i += 1;
        }
        let key: String = b[start..i].iter().collect::<String>().to_ascii_lowercase();
        if key.is_empty() {
            i += 1;
            continue;
        }
        while i < n && b[i].is_whitespace() {
            i += 1;
        }
        let mut value = String::new();
        if i < n && b[i] == '=' {
            i += 1;
            while i < n && b[i].is_whitespace() {
                i += 1;
            }
            if i < n && (b[i] == '"' || b[i] == '\'') {
                let quote = b[i];
                i += 1;
                let vstart = i;
                while i < n && b[i] != quote {
                    i += 1;
                }
                value = b[vstart..i].iter().collect();
                i += 1;
            } else {
                let vstart = i;
                while i < n && !b[i].is_whitespace() {
                    i += 1;
                }
                value = b[vstart..i].iter().collect();
            }
        }
        attrs.push((key, decode_entities(&value)));
    }
    attrs
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_ascii_lowercase().find(&needle.to_ascii_lowercase())
}

// ───────────────────────── tree builder ─────────────────────────

/// Build a full document tree from HTML, wrapped in a synthetic `#document` root that
/// always contains exactly the top-level nodes.
pub fn parse_document(input: &str) -> NodeRef {
    let root = Node::element("#document");
    let kids = build_fragment(input);
    for k in kids {
        append_child(&root, &k);
    }
    root
}

/// Build a list of top-level nodes from an HTML fragment (used for `innerHTML`).
pub fn build_fragment(input: &str) -> Vec<NodeRef> {
    // Cap tree depth so deeply-nested untrusted input cannot build a tree deep
    // enough to overflow the fixed kernel stack during a later recursive traversal
    // (find/find_all/walk_elements/collect_text/serialize all recurse by tree depth).
    // Beyond the cap, elements simply flatten in as siblings of the deepest open
    // element rather than nesting further. 512 is far past any legitimate document.
    const MAX_TREE_DEPTH: usize = 512;

    let tokens = tokenize(input);
    let mut roots: Vec<NodeRef> = Vec::new();
    let mut stack: Vec<NodeRef> = Vec::new();

    for tok in tokens {
        match tok {
            Token::Start { name, attrs, self_closing } => {
                // Implicit close: a new block-ish element closes an open <p>/<li> etc.
                implicit_close(&mut stack, &name);
                let el = Node::element(&name);
                if let Some(e) = el.borrow_mut().as_element_mut() {
                    e.attrs = attrs;
                }
                attach(&mut roots, &stack, &el);
                if !self_closing && !is_void(&name) && stack.len() < MAX_TREE_DEPTH {
                    stack.push(el);
                }
            }
            Token::End { name } => {
                // Pop to the matching open tag (tolerating mismatches).
                if let Some(pos) = stack.iter().rposition(|n| tag(n).as_deref() == Some(name.as_str())) {
                    stack.truncate(pos);
                }
            }
            Token::Text(t) => {
                let node = Node::text(&t);
                attach(&mut roots, &stack, &node);
            }
        }
    }
    roots
}

/// Attach `node` to the current open element, or to the top level if none.
fn attach(roots: &mut Vec<NodeRef>, stack: &[NodeRef], node: &NodeRef) {
    if let Some(parent) = stack.last() {
        append_child(parent, node);
    } else {
        roots.push(node.clone());
    }
}

/// Minimal implicit-close rules so common malformed markup nests sensibly.
fn implicit_close(stack: &mut Vec<NodeRef>, opening: &str) {
    let closes_p = matches!(
        opening,
        "p" | "div" | "ul" | "ol" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "section" | "article"
            | "header" | "footer" | "blockquote" | "pre" | "table" | "form"
    );
    if let Some(top) = stack.last() {
        let t = tag(top).unwrap_or_default();
        if (t == "p" && closes_p) || (t == "li" && opening == "li") || (t == "option" && opening == "option") {
            stack.pop();
        }
    }
}

// ───────────────────────── entities ─────────────────────────

/// Decode HTML entities (named + numeric). Unknown entities are left verbatim.
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let n = s.len();
    while i < n {
        if s.as_bytes()[i] == b'&' {
            if let Some(semi) = s[i + 1..].find(';') {
                let ent = &s[i + 1..i + 1 + semi];
                if let Some(ch) = decode_one(ent) {
                    out.push(ch);
                    i = i + 1 + semi + 1;
                    continue;
                }
            }
            out.push('&');
            i += 1;
        } else {
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn decode_one(ent: &str) -> Option<char> {
    if let Some(num) = ent.strip_prefix('#') {
        let code = if let Some(hex) = num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
            u32::from_str_radix(hex, 16).ok()?
        } else {
            num.parse::<u32>().ok()?
        };
        return char::from_u32(code);
    }
    Some(match ent {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "copy" => '\u{00A9}',
        "reg" => '\u{00AE}',
        "trade" => '\u{2122}',
        "mdash" => '\u{2014}',
        "ndash" => '\u{2013}',
        "hellip" => '\u{2026}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201C}',
        "rdquo" => '\u{201D}',
        "middot" => '\u{00B7}',
        "bull" => '\u{2022}',
        "deg" => '\u{00B0}',
        "euro" => '\u{20AC}',
        "pound" => '\u{00A3}',
        "cent" => '\u{00A2}',
        "sect" => '\u{00A7}',
        "para" => '\u{00B6}',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_tree_with_nesting() {
        let doc = parse_document("<div id=\"main\"><p>Hello <b>world</b></p></div>");
        let main = get_element_by_id(&doc, "main").unwrap();
        assert_eq!(tag(&main).as_deref(), Some("div"));
        assert_eq!(text_content(&main), "Hello world");
    }

    #[test]
    fn captures_script_and_style_raw_text() {
        let doc = parse_document("<style>.a{color:red}</style><script>var x = 1 < 2;</script>");
        let style = get_elements_by_tag(&doc, "style");
        assert_eq!(text_content(&style[0]), ".a{color:red}");
        let script = get_elements_by_tag(&doc, "script");
        assert_eq!(text_content(&script[0]), "var x = 1 < 2;");
    }

    #[test]
    fn attributes_and_classes_parse() {
        let doc = parse_document(r#"<a href="/x" class="btn big" id="go">link</a>"#);
        let a = get_element_by_id(&doc, "go").unwrap();
        assert_eq!(get_attr(&a, "href").as_deref(), Some("/x"));
        assert_eq!(classes(&a), ["btn", "big"]);
    }

    #[test]
    fn query_selector_compound() {
        let doc = parse_document("<p class='note'>a</p><p class='note key'>b</p>");
        assert!(query_selector(&doc, "p.key").is_some());
        assert_eq!(query_selector_all(&doc, "p.note").len(), 2);
        assert_eq!(query_selector_all(&doc, ".key").len(), 1);
    }

    #[test]
    fn text_content_setter_replaces_children() {
        let doc = parse_document("<p id='x'>old <b>stuff</b></p>");
        let p = get_element_by_id(&doc, "x").unwrap();
        set_text_content(&p, "new text");
        assert_eq!(text_content(&p), "new text");
    }

    #[test]
    fn inner_html_round_trips_and_sets() {
        let doc = parse_document("<div id='c'><p>one</p></div>");
        let c = get_element_by_id(&doc, "c").unwrap();
        assert!(inner_html(&c).contains("<p>one</p>"));
        set_inner_html(&c, "<span>two</span><span>three</span>");
        assert_eq!(query_selector_all(&c, "span").len(), 2);
        assert_eq!(text_content(&c), "twothree");
    }

    #[test]
    fn void_elements_dont_capture_siblings() {
        let doc = parse_document("<p>before<br>after</p>");
        let p = query_selector(&doc, "p").unwrap();
        // br is a void child; "after" stays inside <p>, not inside <br>.
        assert_eq!(text_content(&p), "beforeafter");
        let br = get_elements_by_tag(&doc, "br");
        assert_eq!(text_content(&br[0]), "");
    }

    #[test]
    fn implicit_paragraph_close() {
        let doc = parse_document("<p>one<p>two");
        assert_eq!(get_elements_by_tag(&doc, "p").len(), 2);
    }

    #[test]
    fn set_attr_updates_in_place() {
        let doc = parse_document("<div id='d' class='a'></div>");
        let d = get_element_by_id(&doc, "d").unwrap();
        set_attr(&d, "class", "b c");
        assert_eq!(classes(&d), ["b", "c"]);
    }
}


//! A real CSS engine: a parser, a selector matcher with specificity, the cascade,
//! and computed-style resolution over the [`dom`](crate::dom) tree.
//!
//! It is a meaningful subset, not the full spec: selectors cover type/`*`/`.class`/
//! `#id`/compound/descendant and selector lists; the cascade orders by origin (UA <
//! author < inline) then specificity then source order; inheritance flows the
//! inherited properties (color, font, alignment, text-transform) down the tree. The
//! supported properties map onto what the monospace layout engine can actually
//! express: `color`, `background-color`, `font-size`, `font-weight`, `font-style`,
//! `text-decoration`, `text-align`, `text-transform`, `display`, and box margins.
//! Colours parse named / `#rgb` / `#rrggbb` / `rgb()` / `rgba()`.
//!
//! Pure, safe, host-tested.

use crate::dom::{self, NodeRef};
use crate::toolkit::Color;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ───────────────────────────── computed style ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Display {
    Inline,
    Block,
    InlineBlock,
    ListItem,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextTransform {
    None,
    Upper,
    Lower,
    Capitalize,
}

/// A fully resolved style for one element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Computed {
    /// Explicit foreground colour, or `None` to use the theme default / link colour.
    pub color: Option<Color>,
    pub background: Option<Color>,
    pub font_size: i32,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    pub align: Align,
    pub display: Display,
    pub transform: TextTransform,
    pub white_space_pre: bool,
    pub margin_top: i32,
    pub margin_bottom: i32,
    pub margin_left: i32,
    /// Whether this element shows a list marker.
    pub list_item: bool,
}

impl Computed {
    /// The root default style (body context).
    pub fn root(base_font: i32) -> Computed {
        Computed {
            color: None,
            background: None,
            font_size: base_font,
            bold: false,
            italic: false,
            underline: false,
            strike: false,
            align: Align::Left,
            display: Display::Block,
            transform: TextTransform::None,
            white_space_pre: false,
            margin_top: 0,
            margin_bottom: 0,
            margin_left: 0,
            list_item: false,
        }
    }

    /// Derive a child's *starting* style from its parent: inherited properties carry
    /// down; box/display properties reset.
    fn inherit_from(parent: &Computed) -> Computed {
        Computed {
            color: parent.color,
            background: None,
            font_size: parent.font_size,
            bold: parent.bold,
            italic: parent.italic,
            underline: false,
            strike: false,
            align: parent.align,
            display: Display::Inline,
            transform: parent.transform,
            white_space_pre: parent.white_space_pre,
            margin_top: 0,
            margin_bottom: 0,
            margin_left: 0,
            list_item: false,
        }
    }
}

// ───────────────────────────── selectors ─────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct Compound {
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    universal: bool,
}

/// A selector is a descendant chain of compounds (rightmost is the subject).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Selector {
    chain: Vec<Compound>,
}

impl Selector {
    /// Specificity as (ids, classes, types), packed for ordering.
    fn specificity(&self) -> u32 {
        let mut ids = 0u32;
        let mut classes = 0u32;
        let mut types = 0u32;
        for c in &self.chain {
            if c.id.is_some() {
                ids += 1;
            }
            classes += c.classes.len() as u32;
            if c.tag.is_some() {
                types += 1;
            }
        }
        (ids << 16) | (classes << 8) | types
    }

    /// Does this selector match `node`? The rightmost compound must match the node,
    /// and each earlier compound must match some ancestor, in order.
    fn matches(&self, node: &NodeRef) -> bool {
        if self.chain.is_empty() {
            return false;
        }
        let subject = self.chain.last().unwrap();
        if !compound_matches(subject, node) {
            return false;
        }
        // Walk ancestors satisfying the remaining compounds right-to-left.
        let mut idx = self.chain.len() as i32 - 2;
        let mut cur = dom::parent(node);
        while idx >= 0 {
            let comp = &self.chain[idx as usize];
            let mut matched = false;
            while let Some(anc) = cur.clone() {
                cur = dom::parent(&anc);
                if compound_matches(comp, &anc) {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return false;
            }
            idx -= 1;
        }
        true
    }
}

fn compound_matches(c: &Compound, node: &NodeRef) -> bool {
    if c.universal && c.tag.is_none() && c.id.is_none() && c.classes.is_empty() {
        return node.borrow().is_element();
    }
    if let Some(t) = &c.tag {
        if dom::tag(node).as_deref() != Some(t.as_str()) {
            return false;
        }
    }
    if let Some(i) = &c.id {
        if dom::id(node).as_deref() != Some(i.as_str()) {
            return false;
        }
    }
    if !c.classes.is_empty() {
        let cl = dom::classes(node);
        if !c.classes.iter().all(|want| cl.iter().any(|h| h == want)) {
            return false;
        }
    }
    true
}

fn parse_selector(text: &str) -> Option<Selector> {
    let mut chain = Vec::new();
    for part in text.split_whitespace() {
        if part == ">" || part == "+" || part == "~" {
            // Combinators other than descendant are treated as descendant.
            continue;
        }
        chain.push(parse_compound(part)?);
    }
    if chain.is_empty() {
        None
    } else {
        Some(Selector { chain })
    }
}

fn parse_compound(text: &str) -> Option<Compound> {
    let mut c = Compound::default();
    let mut chars = text.char_indices().peekable();
    // Find boundaries at '.' and '#'.
    let bytes = text;
    let mut cuts = Vec::new();
    for (i, ch) in chars.by_ref() {
        if (ch == '.' || ch == '#') && i != 0 {
            cuts.push(i);
        }
    }
    let mut parts = Vec::new();
    let mut last = 0;
    for &cut in &cuts {
        parts.push(&bytes[last..cut]);
        last = cut;
    }
    parts.push(&bytes[last..]);
    for p in parts {
        if p.is_empty() {
            continue;
        }
        if let Some(cl) = p.strip_prefix('.') {
            c.classes.push(cl.to_string());
        } else if let Some(i) = p.strip_prefix('#') {
            c.id = Some(i.to_string());
        } else if p == "*" {
            c.universal = true;
        } else {
            c.tag = Some(p.to_ascii_lowercase());
        }
    }
    Some(c)
}

// ───────────────────────────── rules + stylesheet ─────────────────────────────

#[derive(Clone, Debug)]
struct Rule {
    selectors: Vec<Selector>,
    decls: Vec<(String, String)>,
    /// Source order, for tie-breaking the cascade.
    order: u32,
    /// Origin weight (0 = UA, 1 = author).
    origin: u8,
}

/// A parsed CSS stylesheet (the merged UA + author rules).
#[derive(Clone, Debug, Default)]
pub struct Stylesheet {
    rules: Vec<Rule>,
    next_order: u32,
}

impl Stylesheet {
    pub fn new() -> Stylesheet {
        Stylesheet::default()
    }

    /// The user-agent default stylesheet — what an unstyled HTML document looks like.
    pub fn user_agent() -> Stylesheet {
        let mut s = Stylesheet::new();
        s.add_css(UA_CSS, 0);
        s
    }

    /// Parse and add author CSS (origin 1, higher priority than UA).
    pub fn add_author(&mut self, css: &str) {
        self.add_css(css, 1);
    }

    fn add_css(&mut self, css: &str, origin: u8) {
        for rule in parse_rules(css) {
            let order = self.next_order;
            self.next_order += 1;
            self.rules.push(Rule { selectors: rule.0, decls: rule.1, order, origin });
        }
    }

    /// Compute the style of `node` given its parent's computed style.
    pub fn computed(&self, node: &NodeRef, parent: &Computed) -> Computed {
        let mut style = Computed::inherit_from(parent);

        // Gather matching rules, sorted by (origin, specificity, order).
        let mut hits: Vec<(&Rule, u32)> = Vec::new();
        for rule in &self.rules {
            let mut best = None;
            for sel in &rule.selectors {
                if sel.matches(node) {
                    let sp = sel.specificity();
                    best = Some(best.map_or(sp, |b: u32| b.max(sp)));
                }
            }
            if let Some(sp) = best {
                hits.push((rule, sp));
            }
        }
        hits.sort_by(|a, b| {
            a.0.origin
                .cmp(&b.0.origin)
                .then(a.1.cmp(&b.1))
                .then(a.0.order.cmp(&b.0.order))
        });
        for (rule, _) in hits {
            for (prop, val) in &rule.decls {
                apply_decl(&mut style, prop, val);
            }
        }

        // Inline style attribute wins over everything.
        if let Some(inline) = dom::get_attr(node, "style") {
            for (prop, val) in parse_decls(&inline) {
                apply_decl(&mut style, &prop, &val);
            }
        }
        style
    }
}

/// Apply a single declaration onto a computed style.
fn apply_decl(style: &mut Computed, prop: &str, value: &str) {
    let prop = prop.trim().to_ascii_lowercase();
    let value = value.trim();
    match prop.as_str() {
        "color" => {
            if let Some(c) = parse_color(value) {
                style.color = Some(c);
            }
        }
        "background" | "background-color" => {
            if let Some(c) = parse_color(value) {
                style.background = Some(c);
            }
        }
        "font-size" => {
            if let Some(px) = parse_font_size(value, style.font_size) {
                style.font_size = px;
            }
        }
        "font-weight" => {
            style.bold = matches!(value, "bold" | "bolder" | "600" | "700" | "800" | "900");
        }
        "font-style" => {
            style.italic = value == "italic" || value == "oblique";
        }
        "font" => {
            // Shorthand: pull a px size and a bold keyword if present.
            for tok in value.split_whitespace() {
                if tok == "bold" {
                    style.bold = true;
                } else if let Some(px) = parse_font_size(tok, style.font_size) {
                    style.font_size = px;
                }
            }
        }
        "text-decoration" | "text-decoration-line" => {
            style.underline = value.contains("underline");
            style.strike = value.contains("line-through");
            if value == "none" {
                style.underline = false;
                style.strike = false;
            }
        }
        "text-align" => {
            style.align = match value {
                "center" => Align::Center,
                "right" | "end" => Align::Right,
                _ => Align::Left,
            };
        }
        "text-transform" => {
            style.transform = match value {
                "uppercase" => TextTransform::Upper,
                "lowercase" => TextTransform::Lower,
                "capitalize" => TextTransform::Capitalize,
                _ => TextTransform::None,
            };
        }
        "display" => {
            style.display = match value {
                "none" => Display::None,
                "block" => Display::Block,
                "inline-block" => Display::InlineBlock,
                "list-item" => Display::ListItem,
                _ => Display::Inline,
            };
            style.list_item = matches!(style.display, Display::ListItem);
        }
        "white-space" => {
            style.white_space_pre = value == "pre" || value == "pre-wrap" || value == "pre-line";
        }
        "margin" => {
            // 1–4 values; take top/bottom from the first/(third or first).
            let parts: Vec<i32> = value.split_whitespace().filter_map(parse_px).collect();
            match parts.len() {
                1 => {
                    style.margin_top = parts[0];
                    style.margin_bottom = parts[0];
                    style.margin_left = parts[0];
                }
                2 => {
                    style.margin_top = parts[0];
                    style.margin_bottom = parts[0];
                }
                3 => {
                    style.margin_top = parts[0];
                    style.margin_bottom = parts[2];
                }
                4 => {
                    style.margin_top = parts[0];
                    style.margin_bottom = parts[2];
                    style.margin_left = parts[3];
                }
                _ => {}
            }
        }
        "margin-top" => {
            if let Some(px) = parse_px(value) {
                style.margin_top = px;
            }
        }
        "margin-bottom" => {
            if let Some(px) = parse_px(value) {
                style.margin_bottom = px;
            }
        }
        "margin-left" | "padding-left" => {
            if let Some(px) = parse_px(value) {
                style.margin_left = px;
            }
        }
        _ => {}
    }
}

// ───────────────────────────── value parsing ─────────────────────────────

fn parse_px(v: &str) -> Option<i32> {
    let v = v.trim();
    let num = v.strip_suffix("px").unwrap_or(v);
    num.trim().parse::<f32>().ok().map(|f| f as i32)
}

fn parse_font_size(v: &str, current: i32) -> Option<i32> {
    let v = v.trim();
    match v {
        "xx-small" => return Some(9),
        "x-small" => return Some(11),
        "small" => return Some(13),
        "medium" => return Some(15),
        "large" => return Some(18),
        "x-large" => return Some(22),
        "xx-large" => return Some(26),
        "smaller" => return Some((current - 2).max(8)),
        "larger" => return Some(current + 2),
        _ => {}
    }
    if let Some(em) = v.strip_suffix("em") {
        return em.trim().parse::<f32>().ok().map(|f| (f * current as f32) as i32);
    }
    if let Some(pct) = v.strip_suffix('%') {
        return pct.trim().parse::<f32>().ok().map(|f| (f / 100.0 * current as f32) as i32);
    }
    if let Some(pt) = v.strip_suffix("pt") {
        return pt.trim().parse::<f32>().ok().map(|f| (f * 4.0 / 3.0) as i32);
    }
    if let Some(rem) = v.strip_suffix("rem") {
        return rem.trim().parse::<f32>().ok().map(|f| (f * 15.0) as i32);
    }
    parse_px(v)
}

/// Parse a CSS colour: named, `#rgb`, `#rrggbb`, `rgb()`, `rgba()`.
pub fn parse_color(v: &str) -> Option<Color> {
    let v = v.trim();
    if let Some(hex) = v.strip_prefix('#') {
        return parse_hex_color(hex);
    }
    if let Some(inner) = v.strip_prefix("rgb(").and_then(|s| s.strip_suffix(')')) {
        return parse_rgb(inner);
    }
    if let Some(inner) = v.strip_prefix("rgba(").and_then(|s| s.strip_suffix(')')) {
        return parse_rgb(inner);
    }
    named_color(&v.to_ascii_lowercase())
}

fn parse_hex_color(hex: &str) -> Option<Color> {
    let h = hex.trim();
    match h.len() {
        3 => {
            let r = hex_nib(h.as_bytes()[0])?;
            let g = hex_nib(h.as_bytes()[1])?;
            let b = hex_nib(h.as_bytes()[2])?;
            Some(Color::rgb(r * 17, g * 17, b * 17))
        }
        6 => {
            let r = hex_byte(&h[0..2])?;
            let g = hex_byte(&h[2..4])?;
            let b = hex_byte(&h[4..6])?;
            Some(Color::rgb(r, g, b))
        }
        8 => {
            let r = hex_byte(&h[0..2])?;
            let g = hex_byte(&h[2..4])?;
            let b = hex_byte(&h[4..6])?;
            let a = hex_byte(&h[6..8])?;
            Some(Color::rgba(r, g, b, a))
        }
        _ => None,
    }
}

fn parse_rgb(inner: &str) -> Option<Color> {
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let r = parts[0].trim().parse::<u32>().ok()?.min(255) as u8;
    let g = parts[1].trim().parse::<u32>().ok()?.min(255) as u8;
    let b = parts[2].trim().parse::<u32>().ok()?.min(255) as u8;
    let a = if parts.len() >= 4 {
        (parts[3].trim().parse::<f32>().ok()?.clamp(0.0, 1.0) * 255.0) as u8
    } else {
        255
    };
    Some(Color::rgba(r, g, b, a))
}

fn hex_nib(b: u8) -> Option<u8> {
    (b as char).to_digit(16).map(|d| d as u8)
}
fn hex_byte(s: &str) -> Option<u8> {
    u8::from_str_radix(s, 16).ok()
}

fn named_color(name: &str) -> Option<Color> {
    Some(match name {
        "black" => Color::rgb(0, 0, 0),
        "white" => Color::rgb(255, 255, 255),
        "red" => Color::rgb(255, 0, 0),
        "green" => Color::rgb(0, 128, 0),
        "lime" => Color::rgb(0, 255, 0),
        "blue" => Color::rgb(0, 0, 255),
        "yellow" => Color::rgb(255, 255, 0),
        "cyan" | "aqua" => Color::rgb(0, 255, 255),
        "magenta" | "fuchsia" => Color::rgb(255, 0, 255),
        "gray" | "grey" => Color::rgb(128, 128, 128),
        "silver" => Color::rgb(192, 192, 192),
        "maroon" => Color::rgb(128, 0, 0),
        "olive" => Color::rgb(128, 128, 0),
        "navy" => Color::rgb(0, 0, 128),
        "teal" => Color::rgb(0, 128, 128),
        "purple" => Color::rgb(128, 0, 128),
        "orange" => Color::rgb(255, 165, 0),
        "pink" => Color::rgb(255, 192, 203),
        "brown" => Color::rgb(165, 42, 42),
        "gold" => Color::rgb(255, 215, 0),
        "indigo" => Color::rgb(75, 0, 130),
        "violet" => Color::rgb(238, 130, 238),
        "tomato" => Color::rgb(255, 99, 71),
        "crimson" => Color::rgb(220, 20, 60),
        "steelblue" => Color::rgb(70, 130, 180),
        "slategray" | "slategrey" => Color::rgb(112, 128, 144),
        "lightgray" | "lightgrey" => Color::rgb(211, 211, 211),
        "darkgray" | "darkgrey" => Color::rgb(169, 169, 169),
        "dodgerblue" => Color::rgb(30, 144, 255),
        "transparent" => Color::rgba(0, 0, 0, 0),
        _ => return None,
    })
}

// ───────────────────────────── rule parsing ─────────────────────────────

/// A single `name: value` declaration.
type Declaration = (String, String);
/// A raw parsed rule: the selectors it applies to and the declarations it sets,
/// before cascade ordering/origin are attached (see [`Rule`]).
type ParsedRule = (Vec<Selector>, Vec<Declaration>);

/// Parse a CSS string into (selectors, declarations) rule pairs. At-rules are
/// skipped; nested `@media{...}` blocks have their *inner* rules hoisted (applied
/// unconditionally — a pragmatic approximation).
fn parse_rules(css: &str) -> Vec<ParsedRule> {
    let css = strip_comments(css);
    let mut out = Vec::new();
    let bytes: Vec<char> = css.chars().collect();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // Read up to the next '{' (the selector / at-rule prelude).
        let prelude_start = i;
        while i < n && bytes[i] != '{' && bytes[i] != '}' {
            i += 1;
        }
        if i >= n {
            break;
        }
        if bytes[i] == '}' {
            i += 1;
            continue;
        }
        let prelude: String = bytes[prelude_start..i].iter().collect();
        i += 1; // skip '{'
        let prelude = prelude.trim();

        if prelude.starts_with('@') {
            // At-rule. For @media/@supports, hoist the inner rules; otherwise skip body.
            let inner_start = i;
            let mut depth = 1;
            while i < n && depth > 0 {
                match bytes[i] {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            let inner: String = bytes[inner_start..i.saturating_sub(1)].iter().collect();
            if prelude.starts_with("@media") || prelude.starts_with("@supports") {
                out.extend(parse_rules(&inner));
            }
            continue;
        }

        // Read the declaration block up to the matching '}'.
        let body_start = i;
        while i < n && bytes[i] != '}' {
            i += 1;
        }
        let body: String = bytes[body_start..i].iter().collect();
        if i < n {
            i += 1; // skip '}'
        }

        let selectors: Vec<Selector> = prelude.split(',').filter_map(parse_selector).collect();
        if selectors.is_empty() {
            continue;
        }
        let decls = parse_decls(&body);
        out.push((selectors, decls));
    }
    out
}

/// Parse a `prop: value; prop: value` declaration block.
fn parse_decls(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for decl in body.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        if let Some(colon) = decl.find(':') {
            let prop = decl[..colon].trim().to_string();
            let mut val = decl[colon + 1..].trim().to_string();
            // Strip !important (we treat it as a normal declaration).
            if let Some(stripped) = val.strip_suffix("!important") {
                val = stripped.trim().to_string();
            }
            if !prop.is_empty() && !val.is_empty() {
                out.push((prop, val));
            }
        }
    }
    out
}

fn strip_comments(css: &str) -> String {
    let mut out = String::with_capacity(css.len());
    let mut rest = css;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start + 2..].find("*/") {
            rest = &rest[start + 2 + end + 2..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

/// The built-in user-agent stylesheet.
const UA_CSS: &str = "
html,body{display:block}
div,section,article,header,footer,main,nav,aside,figure,figcaption,form,fieldset,address,dl,dt,dd,table,thead,tbody,tr{display:block}
p{display:block;margin-top:10px;margin-bottom:10px}
h1{display:block;font-size:24px;font-weight:bold;margin-top:14px;margin-bottom:10px}
h2{display:block;font-size:20px;font-weight:bold;margin-top:12px;margin-bottom:8px}
h3{display:block;font-size:18px;font-weight:bold;margin-top:10px;margin-bottom:6px}
h4,h5,h6{display:block;font-size:16px;font-weight:bold;margin-top:8px;margin-bottom:6px}
ul,ol{display:block;margin-top:8px;margin-bottom:8px;margin-left:28px}
li{display:list-item}
blockquote{display:block;margin-top:8px;margin-bottom:8px;margin-left:28px}
pre{display:block;white-space:pre;margin-top:8px;margin-bottom:8px}
b,strong{font-weight:bold}
i,em,cite,var,dfn{font-style:italic}
u,ins{text-decoration:underline}
s,strike,del{text-decoration:line-through}
a{text-decoration:underline}
center{display:block;text-align:center}
hr{display:block;margin-top:8px;margin-bottom:8px}
head,title,meta,link,script,style{display:none}
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom;

    fn style_of(html: &str, sel: &str, author: &str) -> Computed {
        let doc = dom::parse_document(html);
        let mut sheet = Stylesheet::user_agent();
        sheet.add_author(author);
        let node = dom::query_selector(&doc, sel).unwrap();
        // Resolve the ancestor chain's styles so inheritance is correct.
        resolve(&doc, &sheet, &Computed::root(15));
        compute_with_ancestors(&doc, &node, &sheet)
    }

    // Helpers that walk from the root computing inherited styles down to `node`.
    fn resolve(_root: &NodeRef, _sheet: &Stylesheet, _parent: &Computed) {}
    fn compute_with_ancestors(root: &NodeRef, node: &NodeRef, sheet: &Stylesheet) -> Computed {
        // Build the ancestor path root→node.
        let mut path = alloc::vec::Vec::new();
        let mut cur = Some(node.clone());
        while let Some(c) = cur {
            path.push(c.clone());
            cur = dom::parent(&c);
        }
        path.reverse();
        let mut style = Computed::root(15);
        let _ = root;
        for n in path {
            if n.borrow().is_element() {
                style = sheet.computed(&n, &style);
            }
        }
        style
    }

    #[test]
    fn ua_makes_h1_big_and_bold() {
        let s = style_of("<h1>Hi</h1>", "h1", "");
        assert_eq!(s.font_size, 24);
        assert!(s.bold);
        assert_eq!(s.display, Display::Block);
    }

    #[test]
    fn author_class_overrides_color_and_size() {
        let s = style_of(
            "<p class='lead'>x</p>",
            "p.lead",
            ".lead{color:#ff0000;font-size:20px}",
        );
        assert_eq!(s.color, Some(Color::rgb(255, 0, 0)));
        assert_eq!(s.font_size, 20);
    }

    #[test]
    fn id_beats_class_by_specificity() {
        let s = style_of(
            "<p id='hero' class='c'>x</p>",
            "#hero",
            ".c{color:blue} #hero{color:green}",
        );
        assert_eq!(s.color, Some(Color::rgb(0, 128, 0)));
    }

    #[test]
    fn inline_style_wins() {
        let s = style_of(
            "<p class='c' style='color:orange'>x</p>",
            "p.c",
            ".c{color:blue}",
        );
        assert_eq!(s.color, Some(Color::rgb(255, 165, 0)));
    }

    #[test]
    fn color_inherits_to_children() {
        let s = style_of(
            "<div class='box'><span id='t'>x</span></div>",
            "#t",
            ".box{color:#00ff00}",
        );
        assert_eq!(s.color, Some(Color::rgb(0, 255, 0)));
    }

    #[test]
    fn descendant_selector_matches() {
        let s = style_of(
            "<article><p>x</p></article>",
            "p",
            "article p{color:red}",
        );
        assert_eq!(s.color, Some(Color::rgb(255, 0, 0)));
    }

    #[test]
    fn parses_rgb_and_hex_forms() {
        assert_eq!(parse_color("#f00"), Some(Color::rgb(255, 0, 0)));
        assert_eq!(parse_color("#00ff00"), Some(Color::rgb(0, 255, 0)));
        assert_eq!(parse_color("rgb(10, 20, 30)"), Some(Color::rgb(10, 20, 30)));
        assert_eq!(parse_color("dodgerblue"), Some(Color::rgb(30, 144, 255)));
    }

    #[test]
    fn comments_and_media_queries_are_handled() {
        let s = style_of(
            "<p class='c'>x</p>",
            "p.c",
            "/* a comment */ @media screen { .c { color: purple } }",
        );
        assert_eq!(s.color, Some(Color::rgb(128, 0, 128)));
    }

    #[test]
    fn text_align_and_transform_parse() {
        let s = style_of("<p class='c'>x</p>", "p.c", ".c{text-align:center;text-transform:uppercase}");
        assert_eq!(s.align, Align::Center);
        assert_eq!(s.transform, TextTransform::Upper);
    }
}


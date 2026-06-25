//! The HTML layout engine: it parses HTML into the [`dom`](crate::dom) tree, resolves
//! [`css`](crate::css) computed styles over it, and runs a word-wrapping layout pass
//! that emits [`toolkit`](crate::toolkit) draw commands — now fully style-driven
//! (colours, font sizes, weight/italic/underline, text-align, margins, backgrounds,
//! `display:none`, list markers). Each glyph remembers the DOM element it came from,
//! so the browser can fire JavaScript click handlers on whatever the user clicks.
//!
//! Parsing, styling and scripting are separated: this module owns *layout*; the DOM,
//! the cascade, and the JS engine live in their own modules and all share one tree.
//! A [`Document`] therefore exposes its DOM and the scripts found in it, so the
//! engine can run them (mutating the tree) before — and after each event — laying out.
//!
//! Pure, safe, host-tested. Layout is in *content coordinates* (y from 0) so the
//! browser scrolls and hit-tests by simple translation.

use crate::css::{Align, Computed, Display, Stylesheet, TextTransform};
use crate::dom::{self, Node, NodeRef};
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A parsed, styleable HTML document sharing one DOM tree.
#[derive(Clone)]
pub struct Document {
    pub title: String,
    dom: NodeRef,
    sheet: Stylesheet,
    /// Inline `<script>` bodies, in document order.
    pub scripts: Vec<String>,
    /// `src` URLs of external scripts (the engine may fetch and run these).
    pub script_srcs: Vec<String>,
}

impl Default for Document {
    fn default() -> Document {
        Document {
            title: String::new(),
            dom: dom::parse_document(""),
            sheet: Stylesheet::user_agent(),
            scripts: Vec::new(),
            script_srcs: Vec::new(),
        }
    }
}

/// Parse HTML into a [`Document`]: builds the DOM, collects the title, the author CSS
/// from every `<style>`, and the inline/external scripts.
pub fn parse(input: &str) -> Document {
    let dom = dom::parse_document(input);

    let title = dom::get_elements_by_tag(&dom, "title")
        .first()
        .map(dom::text_content)
        .unwrap_or_default()
        .trim()
        .to_string();

    let mut sheet = Stylesheet::user_agent();
    for style_el in dom::get_elements_by_tag(&dom, "style") {
        sheet.add_author(&dom::text_content(&style_el));
    }

    let mut scripts = Vec::new();
    let mut script_srcs = Vec::new();
    for script_el in dom::get_elements_by_tag(&dom, "script") {
        if let Some(src) = dom::get_attr(&script_el, "src") {
            script_srcs.push(src);
        } else {
            let body = dom::text_content(&script_el);
            if !body.trim().is_empty() {
                scripts.push(body);
            }
        }
    }

    Document { title, dom, sheet, scripts, script_srcs }
}

impl Document {
    /// The shared DOM root (`#document`).
    pub fn dom(&self) -> &NodeRef {
        &self.dom
    }

    /// The stylesheet (UA + author), for re-styling after JS mutations.
    pub fn stylesheet(&self) -> &Stylesheet {
        &self.sheet
    }

    /// Every `<a href>` target in document order.
    pub fn links(&self) -> Vec<String> {
        let mut out = Vec::new();
        for a in dom::get_elements_by_tag(&self.dom, "a") {
            if let Some(href) = dom::get_attr(&a, "href") {
                out.push(href);
            }
        }
        out
    }

    /// The document's visible text (for accessibility / tests).
    pub fn text(&self) -> String {
        let root = self.render_root();
        dom::text_content(&root)
    }

    /// The element to lay out from: `<body>` if present, else the document root.
    fn render_root(&self) -> NodeRef {
        dom::get_elements_by_tag(&self.dom, "body").into_iter().next().unwrap_or_else(|| self.dom.clone())
    }

    /// Lay the document out to `width` content pixels at `base` body font size.
    pub fn layout(&self, width: i32, base: i32) -> Layout {
        let mut flow = Flow::new(&self.sheet, width, base);
        let root = self.render_root();
        let root_style = Computed::root(base);
        flow.walk_children(&root, &root_style, None, false);
        flow.finish()
    }
}

// ───────────────────────────── flow / layout builder ─────────────────────────────

const PAD_TOP: i32 = 12;
const PAD_LEFT: i32 = 16;
const LINE_GAP: i32 = 6;

#[derive(Clone)]
struct Glyph {
    x: i32,
    y: i32,
    size: i32,
    text: String,
    color: Option<Color>,
    is_link: bool,
    bold: bool,
    underline: bool,
    strike: bool,
    node: Option<NodeRef>,
}

/// A laid-out document in content coordinates.
#[derive(Clone, Default)]
pub struct Layout {
    pub width: i32,
    pub height: i32,
    glyphs: Vec<Glyph>,
    /// Horizontal rules: (y, x0, x1).
    rules: Vec<(i32, i32, i32)>,
    /// Block background bands: (y0, y1, color).
    backgrounds: Vec<(i32, i32, Color)>,
    /// Clickable link boxes (content coords) → href.
    pub links: Vec<(Rect, String)>,
    /// Element hot-regions (content coords) → DOM node, for JS event dispatch.
    hots: Vec<(Rect, NodeRef)>,
}

impl Layout {
    /// Render into the absolute viewport `area`, scrolled down by `scroll_y`.
    pub fn draw(&self, theme: &Theme, area: Rect, scroll_y: i32) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        let top = area.y;
        let bottom = area.y + area.h;
        // Backgrounds first.
        for (y0, y1, color) in &self.backgrounds {
            let sy = top + y0 - scroll_y;
            let h = y1 - y0;
            if sy + h < top || sy > bottom {
                continue;
            }
            s.push(DrawCmd::Rect { rect: Rect::new(area.x, sy, area.w, h), color: *color, radius: 0 });
        }
        for g in &self.glyphs {
            let sy = top + g.y - scroll_y;
            let gh = toolkit::glyph_height(g.size);
            if sy + gh < top || sy > bottom {
                continue;
            }
            let color = g.color.unwrap_or(if g.is_link { theme.primary } else { theme.text });
            let rect = Rect::new(area.x + g.x, sy, area.w, gh + 2);
            s.push(DrawCmd::Text { rect, text: g.text.clone(), color, size: g.size });
            if g.bold {
                let r2 = Rect::new(area.x + g.x + 1, sy, area.w, gh + 2);
                s.push(DrawCmd::Text { rect: r2, text: g.text.clone(), color, size: g.size });
            }
            let w = g.text.chars().count() as i32 * toolkit::mono_advance(g.size);
            if g.underline {
                let uy = sy + gh;
                s.push(DrawCmd::Line { x0: area.x + g.x, y0: uy, x1: area.x + g.x + w, y1: uy, color, width: 1 });
            }
            if g.strike {
                let my = sy + gh / 2;
                s.push(DrawCmd::Line { x0: area.x + g.x, y0: my, x1: area.x + g.x + w, y1: my, color, width: 1 });
            }
        }
        for (ry, x0, x1) in &self.rules {
            let sy = top + ry - scroll_y;
            if sy < top || sy > bottom {
                continue;
            }
            s.push(DrawCmd::Line { x0: area.x + x0, y0: sy, x1: area.x + x1, y1: sy, color: theme.muted, width: 1 });
        }
        s
    }

    /// Hit-test link boxes; returns the href if a link is under the pointer.
    pub fn link_at(&self, area: Rect, scroll_y: i32, px: i32, py: i32) -> Option<&str> {
        let cx = px - area.x;
        let cy = py - area.y + scroll_y;
        self.links.iter().find(|(r, _)| r.contains(cx, cy)).map(|(_, h)| h.as_str())
    }

    /// Hit-test element regions; returns the DOM node under the pointer (for events).
    /// Prefers a glyph's owning element (inline precision); falls back to block
    /// hot-regions (so clicking a styled block's padding still dispatches).
    pub fn node_at(&self, area: Rect, scroll_y: i32, px: i32, py: i32) -> Option<NodeRef> {
        let cx = px - area.x;
        let cy = py - area.y + scroll_y;
        for g in &self.glyphs {
            let w = g.text.chars().count() as i32 * toolkit::mono_advance(g.size);
            let h = toolkit::glyph_height(g.size);
            if cx >= g.x && cx < g.x + w && cy >= g.y && cy < g.y + h {
                if let Some(n) = &g.node {
                    return Some(n.clone());
                }
            }
        }
        self.hots.iter().rev().find(|(r, _)| r.contains(cx, cy)).map(|(_, n)| n.clone())
    }
}

/// The inline/block layout walker.
struct Flow<'a> {
    sheet: &'a Stylesheet,
    width: i32,
    base: i32,
    y: i32,
    x: i32,
    line_used: bool,
    line_start_glyph: usize,
    line_left: i32,
    line_align: Align,
    cur_size: i32,
    lay: Layout,
    /// List nesting: (ordered, next-ordinal).
    lists: Vec<(bool, u32)>,
}

impl<'a> Flow<'a> {
    fn new(sheet: &'a Stylesheet, width: i32, base: i32) -> Flow<'a> {
        Flow {
            sheet,
            width,
            base,
            y: PAD_TOP,
            x: PAD_LEFT,
            line_used: false,
            line_start_glyph: 0,
            line_left: PAD_LEFT,
            line_align: Align::Left,
            cur_size: base,
            lay: Layout { width, ..Default::default() },
            lists: Vec::new(),
        }
    }

    fn finish(mut self) -> Layout {
        self.break_line();
        self.lay.height = self.y + PAD_TOP;
        self.lay
    }

    /// Walk an element's children, laying out inline content and recursing into blocks.
    fn walk_children(&mut self, parent: &NodeRef, parent_style: &Computed, link: Option<&str>, pre: bool) {
        let children: Vec<NodeRef> = match parent.borrow().as_element() {
            Some(e) => e.children.clone(),
            None => return,
        };
        for child in &children {
            match &*child.borrow() {
                Node::Text(t) => {
                    self.add_text(t, parent_style, link, parent, pre);
                }
                Node::Element(_) => {
                    self.layout_element(child, parent_style, link, pre);
                }
            }
        }
    }

    fn layout_element(&mut self, el: &NodeRef, parent_style: &Computed, link: Option<&str>, pre: bool) {
        let style = self.sheet.computed(el, parent_style);
        if style.display == Display::None {
            return;
        }
        let tag = dom::tag(el).unwrap_or_default();
        match tag.as_str() {
            "br" => {
                self.break_line();
                return;
            }
            "img" => {
                let alt = dom::get_attr(el, "alt").unwrap_or_default();
                let label = if alt.trim().is_empty() {
                    "[image]".to_string()
                } else {
                    let mut l = String::from("[");
                    l.push_str(alt.trim());
                    l.push(']');
                    l
                };
                let mut s2 = style;
                s2.italic = true;
                self.add_text(&label, &s2, link, el, pre);
                return;
            }
            "script" | "style" | "head" | "title" | "meta" | "link" => return,
            _ => {}
        }

        let block = !matches!(style.display, Display::Inline);
        let is_link_el = tag == "a" && dom::get_attr(el, "href").is_some();
        let child_link = if is_link_el { dom::get_attr(el, "href") } else { link.map(|s| s.to_string()) };
        let child_pre = pre || style.white_space_pre || tag == "pre";

        if !block {
            // Inline element: keep flowing in the current line.
            self.walk_children(el, &style, child_link.as_deref(), child_pre);
            return;
        }

        // Block element: start on a fresh line with its margins.
        self.break_line();
        self.y += style.margin_top;

        if tag == "ul" || tag == "ol" {
            self.lists.push((tag == "ol", 1));
            self.walk_children(el, &style, child_link.as_deref(), child_pre);
            self.lists.pop();
            self.y += style.margin_bottom;
            return;
        }

        if tag == "hr" {
            self.lay.rules.push((self.y, PAD_LEFT, self.width - PAD_LEFT));
            self.y += style.margin_bottom.max(8);
            return;
        }

        let block_start_y = self.y;
        let indent = PAD_LEFT + style.margin_left;

        // List marker.
        if matches!(style.display, Display::ListItem) || tag == "li" {
            let depth = self.lists.len().max(1) as i32;
            let extra_indent = (depth - 1) * 28;
            let marker = match self.lists.last_mut() {
                Some((true, n)) => {
                    let mut m = String::new();
                    push_u32(&mut m, *n);
                    *n += 1;
                    m.push('.');
                    m.push(' ');
                    m
                }
                _ => "\u{2022} ".to_string(),
            };
            let mx = indent + extra_indent;
            self.emit_glyph(mx, self.y, self.base, marker, None, false, false, false, false, Some(el.clone()));
            self.x = mx + 28;
            self.line_left = mx + 28;
            self.line_used = true;
            self.line_start_glyph = self.lay.glyphs.len();
            self.line_align = style.align;
            self.cur_size = self.base;
            self.walk_children(el, &style, child_link.as_deref(), child_pre);
            self.break_line();
            self.y += style.margin_bottom;
            return;
        }

        // Generic block (p, div, h1-6, blockquote, pre, …).
        self.x = indent;
        self.line_left = indent;
        self.line_used = false;
        self.line_start_glyph = self.lay.glyphs.len();
        self.line_align = style.align;
        self.walk_children(el, &style, child_link.as_deref(), child_pre);
        self.break_line();

        // Background band for the block, if any.
        if let Some(bg) = style.background {
            self.lay.backgrounds.push((block_start_y - style.margin_top / 2, self.y, bg));
        }
        // A hot-region covering the block, if it (or an ancestor) is interactive.
        if dom::get_attr(el, "onclick").is_some() || el.borrow().as_element().map(|e| !e.listeners.is_empty()).unwrap_or(false) {
            self.lay.hots.push((Rect::new(indent, block_start_y, self.width - indent, (self.y - block_start_y).max(4)), el.clone()));
        }

        self.y += style.margin_bottom;
    }

    /// Add a text run, wrapping words to the content width.
    fn add_text(&mut self, text: &str, style: &Computed, link: Option<&str>, owner: &NodeRef, pre: bool) {
        let size = style.font_size;
        self.cur_size = size;
        let transformed = apply_transform(text, style.transform);
        if pre {
            // Preserve whitespace; split on newlines into hard lines.
            for (i, line) in transformed.split('\n').enumerate() {
                if i > 0 {
                    self.break_line();
                }
                if !line.is_empty() {
                    let w = line.chars().count() as i32 * toolkit::mono_advance(size);
                    self.emit_run(self.x, line, style, link, owner);
                    self.x += w;
                    self.line_used = true;
                }
            }
            return;
        }
        let collapsed = collapse_ws(&transformed);
        if collapsed.is_empty() {
            return;
        }
        let adv = toolkit::mono_advance(size);
        let right = self.width - PAD_LEFT;
        let lead_space = collapsed.starts_with(' ');
        let words: Vec<&str> = collapsed.split(' ').filter(|w| !w.is_empty()).collect();
        for (wi, word) in words.iter().enumerate() {
            let wlen = word.chars().count() as i32 * adv;
            let space = if self.x > self.line_left || (wi == 0 && lead_space && self.line_used) { adv } else { 0 };
            if self.line_used && self.x + space + wlen > right {
                self.break_line();
            }
            let gx = if self.line_used { self.x + space } else { self.x };
            self.emit_run(gx, word, style, link, owner);
            self.x = gx + wlen;
            self.line_used = true;
        }
    }

    fn emit_run(&mut self, x: i32, word: &str, style: &Computed, link: Option<&str>, owner: &NodeRef) {
        let size = style.font_size;
        let is_link = link.is_some();
        self.emit_glyph(
            x,
            self.y,
            size,
            word.to_string(),
            style.color,
            is_link,
            style.bold,
            style.underline || is_link,
            style.strike,
            Some(owner.clone()),
        );
        if let Some(href) = link {
            let w = word.chars().count() as i32 * toolkit::mono_advance(size);
            self.lay.links.push((Rect::new(x, self.y, w, toolkit::glyph_height(size)), href.to_string()));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_glyph(
        &mut self,
        x: i32,
        y: i32,
        size: i32,
        text: String,
        color: Option<Color>,
        is_link: bool,
        bold: bool,
        underline: bool,
        strike: bool,
        node: Option<NodeRef>,
    ) {
        self.lay.glyphs.push(Glyph { x, y, size, text, color, is_link, bold, underline, strike, node });
    }

    /// End the current line: apply alignment to it, then advance `y`.
    fn break_line(&mut self) {
        if self.line_used {
            self.align_current_line();
            self.y += toolkit::glyph_height(self.cur_size) + LINE_GAP;
        }
        self.x = PAD_LEFT;
        self.line_left = PAD_LEFT;
        self.line_used = false;
        self.line_start_glyph = self.lay.glyphs.len();
    }

    /// Shift the glyphs of the just-finished line for center/right alignment.
    fn align_current_line(&mut self) {
        if self.line_align == Align::Left {
            return;
        }
        let from = self.line_start_glyph;
        let n = self.lay.glyphs.len();
        if from >= n {
            return;
        }
        let line_right = self
            .lay
            .glyphs
            .get(n - 1)
            .map(|g| g.x + g.text.chars().count() as i32 * toolkit::mono_advance(g.size))
            .unwrap_or(self.line_left);
        let used = line_right - self.line_left;
        let avail = (self.width - PAD_LEFT) - self.line_left;
        let shift = match self.line_align {
            Align::Center => (avail - used) / 2,
            Align::Right => avail - used,
            Align::Left => 0,
        };
        if shift > 0 {
            for g in &mut self.lay.glyphs[from..] {
                g.x += shift;
            }
            for (r, _) in self.lay.links.iter_mut().rev() {
                // Only shift link boxes on this line (those at/after the line's y top).
                if r.y >= self.y {
                    r.x += shift;
                }
            }
        }
    }
}

fn apply_transform(s: &str, t: TextTransform) -> String {
    match t {
        TextTransform::None => s.to_string(),
        TextTransform::Upper => s.to_uppercase(),
        TextTransform::Lower => s.to_lowercase(),
        TextTransform::Capitalize => {
            let mut out = String::with_capacity(s.len());
            let mut start = true;
            for c in s.chars() {
                if c.is_whitespace() {
                    start = true;
                    out.push(c);
                } else if start {
                    out.extend(c.to_uppercase());
                    start = false;
                } else {
                    out.push(c);
                }
            }
            out
        }
    }
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out
}

fn push_u32(s: &mut String, n: u32) {
    if n >= 10 {
        push_u32(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

/// Decode HTML entities (re-exported from the DOM module for callers that build HTML
/// strings and want consistent decoding).
pub fn decode_entities(s: &str) -> String {
    dom::decode_entities(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_title_headings_and_paragraphs() {
        let doc = parse("<html><head><title>Hi</title></head><body><h1>Header</h1><p>Some text.</p></body></html>");
        assert_eq!(doc.title, "Hi");
        assert!(doc.text().contains("Header"));
        assert!(doc.text().contains("Some text."));
    }

    #[test]
    fn drops_script_and_style_from_render_but_collects_them() {
        let doc = parse("<p>before</p><script>var x = 1 < 2;</script><style>.a{color:red}</style><p>after</p>");
        let lay = doc.layout(400, 15);
        let texts: Vec<String> = lay.glyphs.iter().map(|g| g.text.clone()).collect();
        let joined = texts.join(" ");
        assert!(joined.contains("before"));
        assert!(joined.contains("after"));
        assert!(!joined.contains("var"));
        assert!(!joined.contains("color:red"));
        assert_eq!(doc.scripts.len(), 1);
    }

    #[test]
    fn extracts_links() {
        let doc = parse(r#"<p>see <a href="https://example.com/x">this link</a> now</p>"#);
        assert_eq!(doc.links(), ["https://example.com/x"]);
        let lay = doc.layout(400, 15);
        assert!(lay.links.iter().any(|(_, h)| h == "https://example.com/x"));
    }

    #[test]
    fn css_color_applies_to_glyphs() {
        let doc = parse("<style>p{color:#ff0000}</style><p>red text</p>");
        let lay = doc.layout(400, 15);
        let g = lay.glyphs.iter().find(|g| g.text == "red").unwrap();
        assert_eq!(g.color, Some(Color::rgb(255, 0, 0)));
    }

    #[test]
    fn headings_are_bold_and_larger() {
        let doc = parse("<h1>Big</h1><p>small</p>");
        let lay = doc.layout(400, 15);
        let h = lay.glyphs.iter().find(|g| g.text == "Big").unwrap();
        assert!(h.bold);
        assert_eq!(h.size, 24);
        let p = lay.glyphs.iter().find(|g| g.text == "small").unwrap();
        assert_eq!(p.size, 15);
    }

    #[test]
    fn display_none_is_not_laid_out() {
        let doc = parse("<style>.hidden{display:none}</style><p class='hidden'>secret</p><p>shown</p>");
        let lay = doc.layout(400, 15);
        let texts: Vec<String> = lay.glyphs.iter().map(|g| g.text.clone()).collect();
        assert!(!texts.iter().any(|t| t == "secret"));
        assert!(texts.iter().any(|t| t == "shown"));
    }

    #[test]
    fn inline_bold_and_italic() {
        let doc = parse("<p>a <b>bold</b> <i>ital</i></p>");
        let lay = doc.layout(400, 15);
        assert!(lay.glyphs.iter().any(|g| g.text == "bold" && g.bold));
    }

    #[test]
    fn lists_get_markers() {
        let doc = parse("<ul><li>one</li><li>two</li></ul><ol><li>a</li><li>b</li></ol>");
        let lay = doc.layout(400, 15);
        let texts: Vec<String> = lay.glyphs.iter().map(|g| g.text.clone()).collect();
        assert!(texts.iter().any(|t| t.contains('\u{2022}')));
        assert!(texts.iter().any(|t| t.trim() == "1."));
        assert!(texts.iter().any(|t| t.trim() == "2."));
    }

    #[test]
    fn inline_style_color() {
        let doc = parse("<p style='color:#00ff00'>green</p>");
        let lay = doc.layout(400, 15);
        let g = lay.glyphs.iter().find(|g| g.text == "green").unwrap();
        assert_eq!(g.color, Some(Color::rgb(0, 255, 0)));
    }

    #[test]
    fn wrapping_produces_multiple_lines() {
        let doc = parse("<p>one two three four five six seven eight nine ten eleven twelve</p>");
        let lay = doc.layout(120, 15);
        let ys: Vec<i32> = {
            let mut v: Vec<i32> = lay.glyphs.iter().map(|g| g.y).collect();
            v.dedup();
            v
        };
        assert!(ys.len() > 1, "text should wrap to multiple lines");
    }

    #[test]
    fn hot_region_recorded_for_onclick() {
        let doc = parse("<div onclick='x()'>click me</div>");
        let lay = doc.layout(400, 15);
        let area = Rect::new(0, 0, 400, 400);
        // Click on the text region returns the div node.
        let g = lay.glyphs.iter().find(|g| g.text == "click").unwrap();
        let n = lay.node_at(area, 0, g.x + 2, g.y + 2);
        assert!(n.is_some());
        assert_eq!(dom::tag(&n.unwrap()).as_deref(), Some("div"));
    }

    #[test]
    fn renders_into_draw_commands() {
        let doc = parse("<h1>Title</h1><p>body text here</p>");
        let lay = doc.layout(400, 15);
        let cmds = lay.draw(&Theme::dark(), Rect::new(0, 0, 400, 400), 0);
        assert!(cmds.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Title")));
        assert!(cmds.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "body")));
    }
}


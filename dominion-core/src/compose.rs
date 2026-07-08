//! **Composable UI** — movable / resizable / removable widgets on any page, an
//! edit **lock/unlock**, a **widget picker**, and a content-addressed **global widget
//! library** you can upload to and download from (see `docs/ui/composable-ui.md`).
//!
//! Built-in pages (the dashboard, the node viewer, the explorer) are no longer fixed
//! layouts: each is a [`Board`] of [`Panel`]s. While the board is **unlocked** (edit
//! mode) every panel grows a title bar you can **drag**, a corner handle you can
//! **resize** from, and an **× remove** button; a **+ picker** adds new widgets from
//! the palette. **Lock** the board and the chrome vanishes — it's a clean, static UI
//! again. A whole board serialises to a compact byte pack (`upload`) keyed by its
//! content hash, and a [`Library`] stores named packs so layouts can be shared,
//! published, and installed — the "global widgets library with upload/download".
//!
//! Pure, safe `no_std`, host-tested. Rendering is a [`crate::toolkit`] scene.

use crate::hash::Hash256;
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Snap granularity (px) so dragged/resized panels stay on a tidy grid.
pub const GRID: i32 = 8;
const TITLE_H: i32 = 22;
const HANDLE: i32 = 14;
const MIN_W: i32 = 64;
const MIN_H: i32 = 48;

/// The kinds of widget a board can host. Each renders a small representative scene;
/// real content is wired by the host page. The set is the **widget palette**.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidgetKind {
    Note,
    Clock,
    Chart,
    Metric,
    Terminal,
    Label,
    Image,
}

impl WidgetKind {
    /// Every kind, in palette order (what the picker lists).
    pub fn all() -> &'static [WidgetKind] {
        use WidgetKind::*;
        &[Note, Clock, Chart, Metric, Terminal, Label, Image]
    }
    pub fn label(self) -> &'static str {
        match self {
            WidgetKind::Note => "Note",
            WidgetKind::Clock => "Clock",
            WidgetKind::Chart => "Chart",
            WidgetKind::Metric => "Metric",
            WidgetKind::Terminal => "Terminal",
            WidgetKind::Label => "Label",
            WidgetKind::Image => "Image",
        }
    }
    fn to_u8(self) -> u8 {
        match self {
            WidgetKind::Note => 0,
            WidgetKind::Clock => 1,
            WidgetKind::Chart => 2,
            WidgetKind::Metric => 3,
            WidgetKind::Terminal => 4,
            WidgetKind::Label => 5,
            WidgetKind::Image => 6,
        }
    }
    fn from_u8(b: u8) -> Option<WidgetKind> {
        Some(match b {
            0 => WidgetKind::Note,
            1 => WidgetKind::Clock,
            2 => WidgetKind::Chart,
            3 => WidgetKind::Metric,
            4 => WidgetKind::Terminal,
            5 => WidgetKind::Label,
            6 => WidgetKind::Image,
            _ => return None,
        })
    }
}

/// A placed widget instance on a board.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Panel {
    pub id: u32,
    pub kind: WidgetKind,
    pub rect: Rect,
    pub title: String,
    /// Whether the user may remove it (system panels can be pinned).
    pub removable: bool,
    /// Live content bound to this panel (a note's text, a metric's value, a label…).
    /// When `Some`, it is rendered instead of the representative placeholder.
    pub content: Option<String>,
    /// Live numeric series for data widgets (a Chart's points / a Metric's sparkline).
    /// Empty ⇒ the representative placeholder series is drawn instead.
    pub data: Vec<i64>,
}

/// What the pointer is currently manipulating.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Drag {
    None,
    /// Moving panel `id`, with the grab offset from its top-left.
    Move { id: u32, dx: i32, dy: i32 },
    /// Resizing panel `id` from its bottom-right corner.
    Resize { id: u32 },
}

/// A board: a set of panels with an edit lock, a widget picker, and drag/resize
/// interaction. Built-in pages own one of these.
pub struct Board {
    panels: Vec<Panel>,
    area: Rect,
    locked: bool,
    picker_open: bool,
    drag: Drag,
    next_id: u32,
    last_left: bool,
    damage: Option<Rect>,
}

impl Board {
    pub fn new() -> Board {
        Board {
            panels: Vec::new(),
            area: Rect::new(0, 0, 1280, 720),
            locked: true,
            picker_open: false,
            drag: Drag::None,
            next_id: 1,
            last_left: false,
            damage: Some(Rect::new(0, 0, 1280, 720)),
        }
    }

    // ── state ──

    pub fn panels(&self) -> &[Panel] {
        &self.panels
    }
    pub fn is_locked(&self) -> bool {
        self.locked
    }
    pub fn picker_open(&self) -> bool {
        self.picker_open
    }
    pub fn set_area(&mut self, area: Rect) {
        self.area = area;
        self.dmg_all();
    }
    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    fn dmg_all(&mut self) {
        self.damage = Some(self.area);
    }
    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => toolkit::union(d, r),
            None => r,
        });
    }

    /// Toggle edit mode. Locked = clean static UI; unlocked = move/resize/remove.
    pub fn toggle_lock(&mut self) {
        self.locked = !self.locked;
        if self.locked {
            self.picker_open = false;
            self.drag = Drag::None;
        }
        self.dmg_all();
    }
    pub fn set_locked(&mut self, locked: bool) {
        if self.locked != locked {
            self.toggle_lock();
        }
    }

    /// Add a widget of `kind`, auto-placed and snapped to the grid. Returns its id.
    pub fn add(&mut self, kind: WidgetKind) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        // Cascade new panels so they don't stack exactly — and clear of the
        // top-left picker button so a fresh panel's title bar is grabbable.
        let n = self.panels.len() as i32;
        let x = snap(self.area.x + 48 + (n % 5) * 24);
        let y = snap(self.area.y + 48 + (n % 5) * 24);
        let rect = clamp_rect(Rect::new(x, y, 220, 150), self.area);
        self.panels.push(Panel { id, kind, rect, title: kind.label().to_string(), removable: true, content: None, data: Vec::new() });
        self.dmg_all();
        id
    }

    /// Add a panel with an explicit rect/title (used when installing a saved layout).
    pub fn add_panel(&mut self, kind: WidgetKind, rect: Rect, title: &str, removable: bool) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.panels.push(Panel { id, kind, rect, title: title.to_string(), removable, content: None, data: Vec::new() });
        self.dmg_all();
        id
    }

    /// **Bind live content** to a panel (e.g. a note's text or a metric's value), so it
    /// renders real data instead of the representative placeholder. Returns whether the
    /// panel exists.
    pub fn bind(&mut self, id: u32, content: &str) -> bool {
        if let Some(p) = self.panels.iter_mut().find(|p| p.id == id) {
            // Idempotent re-binds (a host re-pushing the same value) must not force a
            // recomposite: only damage when the content actually changed, and *union*
            // with any pending damage rather than clobbering it (clobbering would drop
            // another panel's queued region and miss its redraw).
            if p.content.as_deref() != Some(content) {
                p.content = Some(content.to_string());
                let r = p.rect;
                self.dmg(r);
            }
            true
        } else {
            false
        }
    }

    /// The live content bound to a panel, if any.
    pub fn content_of(&self, id: u32) -> Option<&str> {
        self.panels.iter().find(|p| p.id == id).and_then(|p| p.content.as_deref())
    }

    /// **Feed live system data** into the data-bound widget kinds: every Clock shows the
    /// `clock` string, every Metric the `metric` string, every Chart the `series`. Only
    /// panels whose value actually changed are damaged, so an idle desktop stays quiet.
    /// Note/Label/Image/Terminal panels keep whatever the host bound to them.
    pub fn feed_live(&mut self, clock: &str, metric: &str, series: &[i64]) {
        let mut changed: Option<Rect> = None;
        for p in &mut self.panels {
            let updated = match p.kind {
                WidgetKind::Clock if p.content.as_deref() != Some(clock) => {
                    p.content = Some(clock.to_string());
                    true
                }
                WidgetKind::Metric if p.content.as_deref() != Some(metric) => {
                    p.content = Some(metric.to_string());
                    true
                }
                WidgetKind::Chart if p.data != series => {
                    p.data = series.to_vec();
                    true
                }
                _ => false,
            };
            if updated {
                changed = Some(match changed {
                    Some(d) => toolkit::union(d, p.rect),
                    None => p.rect,
                });
            }
        }
        if let Some(r) = changed {
            self.dmg(r);
        }
    }

    /// **Drag a widget from the library onto the page** at `(x, y)`: install the named
    /// pack and drop its first panel here, preserving its kind + bound content. Returns
    /// the new panel id, or `None` if the name isn't in the library.
    pub fn drop_from_library(&mut self, lib: &Library, name: &str, x: i32, y: i32) -> Option<u32> {
        let board = lib.install(name)?;
        let src = board.panels().first()?;
        let id = self.next_id;
        self.next_id += 1;
        let rect = clamp_rect(Rect::new(snap(x), snap(y), src.rect.w, src.rect.h), self.area);
        self.panels.push(Panel {
            id,
            kind: src.kind,
            rect,
            title: src.title.clone(),
            removable: true,
            content: src.content.clone(),
            data: src.data.clone(),
        });
        self.dmg_all();
        Some(id)
    }

    /// Remove the panel with `id` (if removable).
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(i) = self.panels.iter().position(|p| p.id == id && p.removable) {
            self.panels.remove(i);
            self.dmg_all();
            true
        } else {
            false
        }
    }

    pub fn toggle_picker(&mut self) {
        if !self.locked {
            self.picker_open = !self.picker_open;
            self.dmg_all();
        }
    }

    // ── layout rects for chrome ──

    fn title_rect(p: &Panel) -> Rect {
        Rect::new(p.rect.x, p.rect.y, p.rect.w, TITLE_H)
    }
    fn close_rect(p: &Panel) -> Rect {
        Rect::new(p.rect.x + p.rect.w - TITLE_H, p.rect.y, TITLE_H, TITLE_H)
    }
    fn handle_rect(p: &Panel) -> Rect {
        Rect::new(p.rect.x + p.rect.w - HANDLE, p.rect.y + p.rect.h - HANDLE, HANDLE, HANDLE)
    }
    /// The +picker toggle button (top-left of the board, edit mode only).
    fn picker_btn(&self) -> Rect {
        Rect::new(self.area.x + 8, self.area.y + 8, 28, 28)
    }
    fn picker_item_rect(&self, i: usize) -> Rect {
        let b = self.picker_btn();
        Rect::new(b.x, b.y + 34 + i as i32 * 30, 132, 28)
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        self.last_left = left;

        // A live drag/resize tracks the pointer regardless of where it is.
        match self.drag {
            Drag::Move { id, dx, dy } if left => {
                if let Some(p) = self.panels.iter_mut().find(|p| p.id == id) {
                    let old = p.rect;
                    let nx = snap(px - dx);
                    let ny = snap(py - dy);
                    p.rect = clamp_rect(Rect::new(nx, ny, p.rect.w, p.rect.h), self.area);
                    let moved = p.rect;
                    self.dmg(toolkit::union(toolkit::inflate(old, 4), toolkit::inflate(moved, 4)));
                }
                return;
            }
            Drag::Resize { id } if left => {
                if let Some(p) = self.panels.iter_mut().find(|p| p.id == id) {
                    let old = p.rect;
                    let w = snap((px - p.rect.x).max(MIN_W));
                    let h = snap((py - p.rect.y).max(MIN_H));
                    p.rect = clamp_rect(Rect::new(p.rect.x, p.rect.y, w, h), self.area);
                    let now = p.rect;
                    self.dmg(toolkit::union(toolkit::inflate(old, 4), toolkit::inflate(now, 4)));
                }
                return;
            }
            _ => {}
        }
        if released {
            self.drag = Drag::None;
        }
        if self.locked || !pressed {
            return;
        }

        // ── edit-mode press routing ──
        // 1. Picker toggle + open picker items.
        if self.picker_btn().contains(px, py) {
            self.toggle_picker();
            return;
        }
        if self.picker_open {
            for (i, kind) in WidgetKind::all().iter().enumerate() {
                if self.picker_item_rect(i).contains(px, py) {
                    self.add(*kind);
                    self.picker_open = false;
                    self.dmg_all();
                    return;
                }
            }
            // Click elsewhere closes the picker.
            self.picker_open = false;
            self.dmg_all();
            return;
        }

        // 2. Panel chrome — topmost first (last drawn = last in vec).
        let ids: Vec<u32> = self.panels.iter().rev().map(|p| p.id).collect();
        for id in ids {
            let p = self.panels.iter().find(|p| p.id == id).unwrap().clone();
            if Self::close_rect(&p).contains(px, py) && p.removable {
                self.remove(id);
                return;
            }
            if Self::handle_rect(&p).contains(px, py) {
                self.raise(id);
                self.drag = Drag::Resize { id };
                return;
            }
            if Self::title_rect(&p).contains(px, py) {
                self.raise(id);
                self.drag = Drag::Move { id, dx: px - p.rect.x, dy: py - p.rect.y };
                return;
            }
        }
    }

    /// Bring panel `id` to the front (drawn last → on top).
    fn raise(&mut self, id: u32) {
        if let Some(i) = self.panels.iter().position(|p| p.id == id) {
            let p = self.panels.remove(i);
            self.panels.push(p);
            self.dmg_all();
        }
    }

    // ── rendering ──

    pub fn view(&self, theme: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        for p in &self.panels {
            self.paint_panel(&mut s, theme, p);
        }
        if !self.locked {
            self.paint_edit_chrome(&mut s, theme);
        }
        s
    }

    fn paint_panel(&self, s: &mut Vec<DrawCmd>, t: &Theme, p: &Panel) {
        // Card: a soft drop shadow, a 1px ring, then the surface — so widgets read as
        // tactile cards floating on the desktop rather than flat rectangles.
        s.push(DrawCmd::Rect { rect: Rect::new(p.rect.x + 2, p.rect.y + 3, p.rect.w, p.rect.h), color: Color::rgba(0, 0, 0, 50), radius: t.radius });
        s.push(DrawCmd::Rect { rect: toolkit::inflate(p.rect, 1), color: Color::rgba(t.primary.r, t.primary.g, t.primary.b, 90), radius: t.radius + 1 });
        s.push(DrawCmd::Rect { rect: p.rect, color: t.surface, radius: t.radius });

        // Header strip: an accent pip + the widget title, with the live content area below.
        const HEADER: i32 = 22;
        let header_only = matches!(p.kind, WidgetKind::Note | WidgetKind::Label);
        if !self.locked {
            s.push(toolkit::disc(p.rect.x + 13, p.rect.y + 13, 3, t.accent));
            s.push(DrawCmd::Text {
                rect: Rect::new(p.rect.x + 22, p.rect.y + 6, p.rect.w - 30, 14),
                text: p.title.clone(),
                color: t.muted,
                size: 11,
            });
        }
        let top = if header_only && self.locked { 6 } else { HEADER };
        let inner = Rect::new(p.rect.x + 10, p.rect.y + top, (p.rect.w - 20).max(0), (p.rect.h - top - 8).max(0));
        paint_kind(s, t, p, inner);

        // Edit chrome (unlocked): a title bar + remove × + resize handle over the card.
        if !self.locked {
            let title = Self::title_rect(p);
            s.push(DrawCmd::Rect { rect: title, color: t.primary, radius: t.radius });
            s.push(DrawCmd::Text {
                rect: Rect::new(title.x + 8, title.y + 3, title.w - TITLE_H - 8, TITLE_H - 4),
                text: p.title.clone(),
                color: t.on_primary,
                size: 13,
            });
            if p.removable {
                let c = Self::close_rect(p);
                s.push(DrawCmd::Text { rect: Rect::new(c.x + 6, c.y + 3, c.w, c.h), text: "×".into(), color: t.on_primary, size: 16 });
            }
            // Resize handle (two corner ticks).
            let h = Self::handle_rect(p);
            s.push(toolkit::line(h.x + 3, h.y + h.h - 3, h.x + h.w - 3, h.y + h.h - 3, t.accent, 2));
            s.push(toolkit::line(h.x + h.w - 3, h.y + 3, h.x + h.w - 3, h.y + h.h - 3, t.accent, 2));
        }
    }

    fn paint_edit_chrome(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let b = self.picker_btn();
        s.push(DrawCmd::Rect { rect: b, color: t.accent, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(b.x + 8, b.y + 4, b.w, b.h), text: "+".into(), color: t.on_primary, size: 18 });
        if self.picker_open {
            for (i, kind) in WidgetKind::all().iter().enumerate() {
                let r = self.picker_item_rect(i);
                s.push(DrawCmd::Rect { rect: r, color: t.surface, radius: t.radius });
                s.push(DrawCmd::Text { rect: Rect::new(r.x + 8, r.y + 5, r.w - 8, 18), text: kind.label().into(), color: t.text, size: 13 });
            }
        }
    }

    // ── the global widget library: upload / download ──

    /// Serialise the whole board layout into a portable byte **pack** — the unit you
    /// upload to / download from the library. Compact, self-describing, versioned.
    pub fn upload(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"AWB2"); // magic + version (AWB2 carries bound content + data)
        put_u16(&mut out, self.panels.len() as u16);
        for p in &self.panels {
            out.push(p.kind.to_u8());
            put_i16(&mut out, p.rect.x as i16);
            put_i16(&mut out, p.rect.y as i16);
            put_i16(&mut out, p.rect.w as i16);
            put_i16(&mut out, p.rect.h as i16);
            let tb = p.title.as_bytes();
            put_u16(&mut out, tb.len() as u16);
            out.extend_from_slice(tb);
            out.push(p.removable as u8);
            // Bound content: presence byte, then length-prefixed UTF-8 when set.
            match &p.content {
                Some(s) => {
                    out.push(1);
                    let cb = s.as_bytes();
                    put_u16(&mut out, cb.len() as u16);
                    out.extend_from_slice(cb);
                }
                None => out.push(0),
            }
            // Live data: count-prefixed big-endian i64s.
            put_u16(&mut out, p.data.len() as u16);
            for &d in &p.data {
                out.extend_from_slice(&d.to_be_bytes());
            }
        }
        out
    }

    /// The content hash of this layout — its stable, self-certifying name in the
    /// library (same bytes ⇒ same name, like the rest of the object graph).
    pub fn content_id(&self) -> Hash256 {
        Hash256::of(&self.upload())
    }

    /// Reconstruct a board from a pack produced by [`Board::upload`]. Returns `None`
    /// on a malformed/foreign pack.
    pub fn download(bytes: &[u8]) -> Option<Board> {
        let mut c = Cursor::new(bytes);
        if c.take(4)? != b"AWB2" {
            return None;
        }
        let count = c.u16()? as usize;
        let mut board = Board::new();
        for _ in 0..count {
            let kind = WidgetKind::from_u8(c.u8()?)?;
            let x = c.i16()? as i32;
            let y = c.i16()? as i32;
            let w = c.i16()? as i32;
            let h = c.i16()? as i32;
            let tlen = c.u16()? as usize;
            let title = core::str::from_utf8(c.take(tlen)?).ok()?.to_string();
            let removable = c.u8()? != 0;
            let id = board.add_panel(kind, Rect::new(x, y, w, h), &title, removable);
            // Restore bound content + live data (round-trips through publish/install).
            let content = match c.u8()? {
                0 => None,
                1 => {
                    let clen = c.u16()? as usize;
                    Some(core::str::from_utf8(c.take(clen)?).ok()?.to_string())
                }
                _ => return None,
            };
            let dcount = c.u16()? as usize;
            let mut data = Vec::with_capacity(dcount);
            for _ in 0..dcount {
                data.push(c.i64()?);
            }
            if let Some(p) = board.panels.iter_mut().find(|p| p.id == id) {
                p.content = content;
                p.data = data;
            }
        }
        Some(board)
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

/// The **global widget library** — named, content-addressed layout packs that can be
/// published (uploaded) and installed (downloaded) across pages and devices.
#[derive(Default)]
pub struct Library {
    entries: Vec<(String, Vec<u8>)>,
}

impl Library {
    pub fn new() -> Library {
        Library { entries: Vec::new() }
    }
    /// Publish a board under `name` (overwrites an existing entry of that name).
    /// Returns its content id so callers can verify integrity.
    pub fn publish(&mut self, name: &str, board: &Board) -> Hash256 {
        let pack = board.upload();
        let id = Hash256::of(&pack);
        if let Some(e) = self.entries.iter_mut().find(|(n, _)| n == name) {
            e.1 = pack;
        } else {
            self.entries.push((name.to_string(), pack));
        }
        id
    }
    /// Install (download) a published board by name.
    pub fn install(&self, name: &str) -> Option<Board> {
        let pack = &self.entries.iter().find(|(n, _)| n == name)?.1;
        Board::download(pack)
    }
    /// The published names (what a library browser lists).
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(n, _)| n.as_str()).collect()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── per-kind content: live data when the host has fed it, else a representative scene ──

/// Push a text run centred both horizontally and vertically inside `area`.
fn centered_text(s: &mut Vec<DrawCmd>, area: Rect, text: &str, color: Color, size: i32) {
    let tw = text.chars().count() as i32 * toolkit::mono_advance(size);
    let x = area.x + (area.w - tw).max(0) / 2;
    s.push(DrawCmd::Text { rect: Rect::new(x, area.y, tw.max(1), area.h.max(size)), text: text.into(), color, size });
}

/// Parse a leading integer percentage out of a string like `"37%"` (for the bar).
fn leading_pct(text: &str) -> i32 {
    let mut n = 0i32;
    let mut any = false;
    for c in text.chars() {
        if let Some(d) = c.to_digit(10) {
            n = n * 10 + d as i32;
            any = true;
        } else {
            break;
        }
    }
    if any { n.clamp(0, 100) } else { 0 }
}

fn paint_kind(s: &mut Vec<DrawCmd>, t: &Theme, p: &Panel, r: Rect) {
    if r.w <= 0 || r.h <= 0 {
        return;
    }
    match p.kind {
        WidgetKind::Note => {
            let text = p.content.as_deref().unwrap_or("Double-click to edit…");
            s.push(DrawCmd::Text { rect: Rect::new(r.x, r.y + 2, r.w, r.h.min(20)), text: toolkit::ellipsize_px(text, r.w, 13), color: t.text, size: 13 });
        }
        WidgetKind::Label => {
            let text = p.content.as_deref().unwrap_or("Label");
            centered_text(s, r, &toolkit::ellipsize_px(text, r.w, 16), t.text, 16);
        }
        WidgetKind::Clock => {
            // Live digital clock: a big centred time with a quiet caption underneath.
            let time = p.content.as_deref().unwrap_or("0:00:00");
            let size = (r.h / 3).clamp(18, 34);
            let big = Rect::new(r.x, r.y, r.w, r.h - 16);
            centered_text(s, big, time, t.text, size);
            centered_text(s, Rect::new(r.x, r.y + r.h - 16, r.w, 14), "elapsed", t.muted, 11);
        }
        WidgetKind::Chart => {
            // A live area chart of the fed series (e.g. CPU history). Representative wave
            // until the host feeds real data.
            let fallback = alloc::vec![2, 4, 3, 6, 5, 8, 6, 9, 7, 10];
            let series: &[i64] = if p.data.is_empty() { &fallback } else { &p.data };
            let max = series.iter().copied().max().unwrap_or(1).max(1) as i32;
            let n = series.len() as i32;
            let baseline = r.y + r.h - 2;
            let fill = Color::rgba(t.accent.r, t.accent.g, t.accent.b, 45);
            let mut pts: Vec<(i32, i32)> = Vec::with_capacity(series.len());
            for (i, v) in series.iter().enumerate() {
                let x = if n > 1 { r.x + i as i32 * (r.w - 1) / (n - 1) } else { r.x };
                let y = baseline - (*v as i32 * (r.h - 6) / max).clamp(0, r.h - 6);
                s.push(toolkit::line(x, baseline, x, y, fill, 3)); // area fill
                pts.push((x, y));
            }
            if pts.len() >= 2 {
                s.push(toolkit::polyline(pts, t.accent, 2));
            }
            // Current value, top-left.
            if let Some(last) = series.last() {
                let mut v = String::new();
                push_int(&mut v, *last);
                s.push(DrawCmd::Text { rect: Rect::new(r.x, r.y, r.w, 14), text: v, color: t.muted, size: 11 });
            }
        }
        WidgetKind::Metric => {
            // A big live number with a thin progress bar below.
            let value = p.content.as_deref().unwrap_or("—");
            let size = (r.h / 2).clamp(20, 40);
            centered_text(s, Rect::new(r.x, r.y, r.w, r.h - 14), value, t.text, size);
            let pct = leading_pct(value);
            let track = Rect::new(r.x, r.y + r.h - 10, r.w, 6);
            s.push(DrawCmd::Rect { rect: track, color: Color::rgba(t.muted.r, t.muted.g, t.muted.b, 90), radius: 3 });
            if pct > 0 {
                s.push(DrawCmd::Rect { rect: Rect::new(track.x, track.y, track.w * pct / 100, track.h), color: t.accent, radius: 3 });
            }
        }
        WidgetKind::Terminal => {
            s.push(DrawCmd::Rect { rect: r, color: Color::rgb(0x0c, 0x0e, 0x12), radius: 4 });
            let line = p.content.as_deref().unwrap_or("› _");
            s.push(DrawCmd::Text { rect: Rect::new(r.x + 6, r.y + 8, r.w - 8, 16), text: toolkit::ellipsize_px(line, r.w - 12, 13), color: t.accent, size: 13 });
        }
        WidgetKind::Image => {
            s.push(DrawCmd::Rect { rect: r, color: Color::rgba(t.accent.r, t.accent.g, t.accent.b, 50), radius: 4 });
            // A simple framed-picture motif: horizon + sun.
            s.push(toolkit::disc(r.x + r.w / 3, r.y + r.h / 2, (r.h / 6).max(2), t.accent));
            s.push(toolkit::line(r.x + 6, r.y + r.h * 2 / 3, r.x + r.w - 6, r.y + r.h * 2 / 3, Color::rgba(t.accent.r, t.accent.g, t.accent.b, 120), 2));
        }
    }
}

// ── helpers ──

fn snap(v: i32) -> i32 {
    (v + GRID / 2) / GRID * GRID
}

fn push_int(s: &mut String, mut n: i64) {
    if n < 0 {
        s.push('-');
        n = -n;
    }
    if n >= 10 {
        push_int(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

fn clamp_rect(r: Rect, area: Rect) -> Rect {
    let w = r.w.clamp(MIN_W, area.w.max(MIN_W));
    let h = r.h.clamp(MIN_H, area.h.max(MIN_H));
    let x = r.x.clamp(area.x, area.x + area.w - w);
    let y = r.y.clamp(area.y, area.y + area.h - h);
    Rect::new(x, y, w, h)
}

fn put_u16(o: &mut Vec<u8>, v: u16) {
    o.extend_from_slice(&v.to_be_bytes());
}
fn put_i16(o: &mut Vec<u8>, v: i16) {
    o.extend_from_slice(&v.to_be_bytes());
}

/// A tiny bounds-checked byte reader for [`Board::download`].
struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Cursor<'a> {
        Cursor { b, i: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.i..self.i + n)?;
        self.i += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.take(2)?;
        Some(u16::from_be_bytes([s[0], s[1]]))
    }
    fn i16(&mut self) -> Option<i16> {
        let s = self.take(2)?;
        Some(i16::from_be_bytes([s[0], s[1]]))
    }
    fn i64(&mut self) -> Option<i64> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Some(i64::from_be_bytes(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> Board {
        let mut b = Board::new();
        b.set_area(Rect::new(0, 0, 1000, 700));
        let _ = b.take_damage();
        b
    }

    #[test]
    fn locked_by_default_no_chrome() {
        let b = board();
        assert!(b.is_locked());
        // A locked board with no panels paints nothing structural.
        assert!(b.view(&Theme::dark()).is_empty());
    }

    #[test]
    fn unlock_add_move_resize_remove() {
        let mut b = board();
        b.toggle_lock();
        assert!(!b.is_locked());
        let id = b.add(WidgetKind::Chart);
        assert_eq!(b.panels().len(), 1);
        let start = b.panels()[0].rect;

        // Drag the title bar: press inside it, move, release.
        let title = Board::title_rect(&b.panels()[0]);
        b.on_pointer(title.x + 5, title.y + 5, true);
        b.on_pointer(title.x + 5 + 80, title.y + 5 + 48, true);
        b.on_pointer(title.x + 5 + 80, title.y + 5 + 48, false);
        assert_ne!(b.panels()[0].rect.x, start.x);

        // Resize from the corner handle.
        let h = Board::handle_rect(&b.panels()[0]);
        let before = b.panels()[0].rect;
        b.on_pointer(h.x + 2, h.y + 2, true);
        b.on_pointer(h.x + 60, h.y + 40, true);
        b.on_pointer(h.x + 60, h.y + 40, false);
        assert!(b.panels()[0].rect.w > before.w);

        // Remove via the × button.
        let c = Board::close_rect(&b.panels()[0]);
        b.on_pointer(c.x + 4, c.y + 4, true);
        b.on_pointer(c.x + 4, c.y + 4, false);
        assert!(b.panels().is_empty());
        let _ = id;
    }

    #[test]
    fn picker_adds_widgets_only_when_unlocked() {
        let mut b = board();
        // Locked: clicking where the picker would be does nothing.
        let pb = b.picker_btn();
        b.on_pointer(pb.x + 4, pb.y + 4, true);
        b.on_pointer(pb.x + 4, pb.y + 4, false);
        assert!(!b.picker_open());
        assert!(b.panels().is_empty());

        b.toggle_lock();
        b.on_pointer(pb.x + 4, pb.y + 4, true); // open picker
        b.on_pointer(pb.x + 4, pb.y + 4, false);
        assert!(b.picker_open());
        // Click the first palette item → a Note is added.
        let item = b.picker_item_rect(0);
        b.on_pointer(item.x + 4, item.y + 4, true);
        b.on_pointer(item.x + 4, item.y + 4, false);
        assert_eq!(b.panels().len(), 1);
        assert_eq!(b.panels()[0].kind, WidgetKind::Note);
    }

    #[test]
    fn lock_hides_edit_chrome() {
        let mut b = board();
        b.toggle_lock();
        b.add(WidgetKind::Metric);
        let unlocked = b.view(&Theme::dark());
        // Edit chrome includes the accent + picker button.
        let edit_rects = unlocked.iter().filter(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == Theme::dark().accent)).count();
        assert!(edit_rects >= 1);
        b.toggle_lock(); // lock again
        let locked = b.view(&Theme::dark());
        // The title bar (primary) and picker (accent) chrome are gone.
        assert!(!locked.iter().any(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == Theme::dark().primary)));
    }

    #[test]
    fn upload_download_round_trips_a_layout() {
        let mut b = board();
        b.toggle_lock();
        b.add(WidgetKind::Chart);
        b.add(WidgetKind::Terminal);
        b.panels(); // two panels
        let pack = b.upload();
        let restored = Board::download(&pack).expect("valid pack");
        assert_eq!(restored.panels().len(), 2);
        assert_eq!(restored.panels()[0].kind, WidgetKind::Chart);
        assert_eq!(restored.panels()[1].kind, WidgetKind::Terminal);
        // Same bytes ⇒ same content id.
        assert_eq!(b.content_id(), restored.content_id());
    }

    #[test]
    fn download_rejects_foreign_bytes() {
        assert!(Board::download(b"not a pack").is_none());
        assert!(Board::download(b"AWB1\xff").is_none()); // truncated
    }

    #[test]
    fn library_publishes_and_installs_named_packs() {
        let mut lib = Library::new();
        let mut b = board();
        b.toggle_lock();
        b.add(WidgetKind::Clock);
        b.add(WidgetKind::Metric);
        let id = lib.publish("dash-default", &b);
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.names(), alloc::vec!["dash-default"]);
        let installed = lib.install("dash-default").expect("installs");
        assert_eq!(installed.panels().len(), 2);
        assert_eq!(installed.content_id(), id);
        assert!(lib.install("missing").is_none());
    }

    #[test]
    fn panels_bind_live_content_and_render_it() {
        let mut b = board();
        let id = b.add(WidgetKind::Note);
        // Before binding, no content.
        assert!(b.content_of(id).is_none());
        // Bind live content; it is stored and rendered as text.
        assert!(b.bind(id, "buy milk"));
        assert_eq!(b.content_of(id), Some("buy milk"));
        let scene = b.view(&Theme::dark());
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "buy milk")));
        // Binding an unknown panel fails.
        assert!(!b.bind(9999, "x"));
    }

    #[test]
    fn feed_live_updates_data_widgets_by_kind() {
        let mut b = board();
        let clock = b.add(WidgetKind::Clock);
        let metric = b.add(WidgetKind::Metric);
        let chart = b.add(WidgetKind::Chart);
        let note = b.add(WidgetKind::Note);
        b.feed_live("1:02:03", "47%", &[3, 6, 9]);
        assert_eq!(b.content_of(clock), Some("1:02:03"));
        assert_eq!(b.content_of(metric), Some("47%"));
        // The chart's series is fed; the note is untouched by live data.
        assert_eq!(b.panels().iter().find(|p| p.id == chart).unwrap().data, alloc::vec![3, 6, 9]);
        assert!(b.content_of(note).is_none());
        // It renders the live time, not a placeholder.
        let scene = b.view(&Theme::dark());
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "1:02:03")));
    }

    #[test]
    fn drag_from_library_drops_a_widget_onto_the_page() {
        // Publish a board with one panel into the library.
        let mut source = board();
        source.add(WidgetKind::Chart);
        let mut lib = Library::new();
        lib.publish("my-chart", &source);
        // Drop it onto a fresh board at a location.
        let mut target = board();
        let before = target.panels().len();
        let dropped = target.drop_from_library(&lib, "my-chart", 100, 120).unwrap();
        assert_eq!(target.panels().len(), before + 1);
        let p = target.panels().iter().find(|p| p.id == dropped).unwrap();
        assert_eq!(p.kind, WidgetKind::Chart);
        // An unknown library entry can't be dropped.
        assert!(target.drop_from_library(&lib, "missing", 0, 0).is_none());
    }

    #[test]
    fn panels_stay_inside_the_board_area() {
        let mut b = board();
        b.toggle_lock();
        let id = b.add(WidgetKind::Note);
        // Try to drag it way off the right/bottom; it clamps inside the area.
        let title = Board::title_rect(&b.panels()[0]);
        b.on_pointer(title.x + 4, title.y + 4, true);
        b.on_pointer(5000, 5000, true);
        b.on_pointer(5000, 5000, false);
        let r = b.panels().iter().find(|p| p.id == id).unwrap().rect;
        assert!(r.x + r.w <= b.area.x + b.area.w);
        assert!(r.y + r.h <= b.area.y + b.area.h);
    }
}

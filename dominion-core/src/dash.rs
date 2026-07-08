//! The live dashboard — the whole DominionOS home screen, from the concept.
//!
//! One composed, fully-interactive surface (no fake windows): a **node-graph editor**
//! of applications/data wired together in the centre (pannable, draggable, rewirable);
//! an **identity / capability / NDN-namespace / network** panel on the left; a
//! **compute & system-health monitor + real-time logging** panel on the right;
//! **draggable application/detail windows**; and a status bar. The side panels can be
//! **resized and collapsed**. Clicking a node, capability, or NDN item opens a window
//! with that item's own data and controls. Everything is driven by **real system
//! metrics** the kernel feeds in each frame.
//!
//! The dashboard also tracks a **damage region** (the bounding box of what changed
//! since the last frame) so the kernel can repaint incrementally instead of redrawing
//! the whole screen. Pure, safe `no_std`.

use crate::nodes::{NodeGraph, NodeKind, Press};
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::widgets::{self, Slider};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const TITLE_H: i32 = 54;
const STATUS_H: i32 = 28;
const COLLAPSED_W: i32 = 22;
const RESIZE_GRIP: i32 = 6;
/// Margin around a window when computing its damage rect — covers the 1px border ring
/// (and any future shadow) that is drawn just outside the window's own rectangle.
const WIN_DECOR: i32 = 3;

/// Live system metrics the kernel measures and feeds in each frame.
#[derive(Clone, Default)]
pub struct Metrics {
    pub cpu_milli: u32,
    pub mem_milli: u32,
    pub gpu_milli: u32,
    pub npu_milli: u32,
    pub entropy_milli: u32,
    pub fps: u32,
    pub uptime_secs: u64,
    pub mem_used_kb: u64,
    pub mem_total_kb: u64,
    pub net_present: bool,
    pub disk_present: bool,
    pub disk_read_bps: u64,
    pub disk_write_bps: u64,
    pub net_rx_bps: u64,
    pub net_tx_bps: u64,
    pub det_hash: String,
    pub cpu_history: Vec<i64>,
    pub compute_bars: Vec<i64>,
}

/// A capability shown in the identity panel.
#[derive(Clone)]
pub struct CapRow {
    pub label: String,
    pub rights: String,
}

/// A node in the NDN namespace tree.
#[derive(Clone)]
pub struct NdnNode {
    pub label: String,
    pub children: Vec<NdnNode>,
    pub expanded: bool,
}

impl NdnNode {
    pub fn leaf(label: &str) -> NdnNode {
        NdnNode { label: label.into(), children: Vec::new(), expanded: false }
    }
    pub fn branch(label: &str, expanded: bool, children: Vec<NdnNode>) -> NdnNode {
        NdnNode { label: label.into(), children, expanded }
    }
}

struct NdnVisRow {
    depth: i32,
    label: String,
    branch: bool,
    expanded: bool,
    path: Vec<usize>,
}

/// What a floating window shows.
#[derive(Clone)]
enum WinKind {
    /// An application window for a graph node (content varies per node).
    Node(u32),
    /// Capability details + controls.
    Cap(usize),
    /// NDN namespace item details + controls.
    Ndn(String),
}

/// A draggable floating window.
struct Win {
    id: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    title: String,
    kind: WinKind,
}

/// Which side panel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// What the pointer is currently dragging.
enum Drag {
    None,
    Window(u32, i32, i32),
    Resize(Side),
    Slider,
    Graph,
}

struct Regions {
    title: Rect,
    left: Rect,
    center: Rect,
    right: Rect,
    status: Rect,
    left_collapsed: bool,
    right_collapsed: bool,
}

/// The whole dashboard.
pub struct Dash {
    graph: NodeGraph,
    metrics: Metrics,
    logs: Vec<String>,
    log_scroll: usize,
    caps: Vec<CapRow>,
    cap_selected: Option<usize>,
    ndn: Vec<NdnNode>,
    windows: Vec<Win>,
    next_win_id: u32,
    drag: Drag,
    theme_dark: bool,
    accent: Slider,
    left_w: i32,
    right_w: i32,
    left_collapsed: bool,
    right_collapsed: bool,
    last_left: bool,
    press_x: i32,
    press_y: i32,
    w: i32,
    h: i32,
    damage: Option<Rect>,
}

impl Dash {
    pub fn new() -> Dash {
        let mut graph = NodeGraph::new();
        graph.add(1, "Project Alpha Sales", "Data Object", NodeKind::Data, 60, 60);
        graph.add(2, "System Logs", "Log Stream", NodeKind::Log, 60, 190);
        graph.add(3, "Q3 Performance", "Report", NodeKind::Report, 300, 60);
        graph.add(4, "Neural Audio: Speech", "Audio Object", NodeKind::Audio, 300, 190);
        graph.add(5, "Document", "Application", NodeKind::App, 180, 320);
        graph.wire(1, 3);
        graph.wire(1, 5);
        graph.wire(2, 3);
        graph.wire(4, 5);

        Dash {
            graph,
            metrics: Metrics::default(),
            logs: Vec::new(),
            log_scroll: 0,
            caps: default_caps(),
            cap_selected: None,
            ndn: default_ndn(),
            windows: Vec::new(),
            next_win_id: 1000,
            drag: Drag::None,
            theme_dark: true,
            accent: Slider::new(620),
            left_w: 300,
            right_w: 340,
            left_collapsed: false,
            right_collapsed: false,
            last_left: false,
            press_x: 0,
            press_y: 0,
            w: 1280,
            h: 720,
            damage: Some(Rect::new(0, 0, 1280, 720)),
        }
    }

    // ── kernel-facing data feed ──

    pub fn set_metrics(&mut self, m: Metrics) {
        self.metrics = m;
        let r = self.regions(self.w, self.h);
        self.dmg(r.right); // gauges + bar chart live in the right panel
        self.dmg(r.status); // the status bar shows the live fps + det-hash too
    }

    pub fn push_log(&mut self, line: &str) {
        self.logs.push(line.to_string());
        if self.logs.len() > 500 {
            let drop = self.logs.len() - 500;
            self.logs.drain(0..drop);
        }
        let r = self.regions(self.w, self.h);
        self.dmg(r.right);
    }

    pub fn set_caps(&mut self, caps: Vec<CapRow>) {
        self.caps = caps;
        self.dmg_all();
    }

    /// Tell the dashboard the framebuffer size, so its damage rectangles line up with
    /// what [`view`](Self::view) will be asked to render. Forces a full repaint on a
    /// real change.
    pub fn set_size(&mut self, w: i32, h: i32) {
        if w != self.w || h != self.h {
            self.w = w;
            self.h = h;
            self.dmg_all();
        }
    }

    pub fn theme(&self) -> Theme {
        let mut t = if self.theme_dark { Theme::dark() } else { Theme::light() };
        let vivid = Color::rgb(0x5a, 0xc8, 0xff);
        t.primary = widgets::lerp(t.primary, vivid, self.accent.value_milli);
        t
    }

    /// The bounding box of everything that changed since the last `take_damage`.
    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }

    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => union(d, r),
            None => r,
        });
    }
    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.w, self.h));
    }

    pub fn open_windows(&self) -> usize {
        self.windows.len()
    }

    // ── layout ──

    fn regions(&self, w: i32, h: i32) -> Regions {
        let title = Rect::new(0, 0, w, TITLE_H);
        let status = Rect::new(0, h - STATUS_H, w, STATUS_H);
        let body_y = TITLE_H;
        let body_h = h - TITLE_H - STATUS_H;
        // `.max(lower)` keeps the clamp upper bound >= the lower bound: i32::clamp
        // panics if min > max, which happens for a narrow framebuffer (w/2 below
        // the 160/180 minimums).
        let lw = if self.left_collapsed { COLLAPSED_W } else { self.left_w.clamp(160, (w / 2).max(160)) };
        let rw = if self.right_collapsed { COLLAPSED_W } else { self.right_w.clamp(180, (w / 2).max(180)) };
        let left = Rect::new(0, body_y, lw, body_h);
        let right = Rect::new(w - rw, body_y, rw, body_h);
        let center = Rect::new(lw, body_y, (w - lw - rw).max(0), body_h);
        Regions {
            title,
            left,
            center,
            right,
            status,
            left_collapsed: self.left_collapsed,
            right_collapsed: self.right_collapsed,
        }
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;

        if pressed {
            self.press_x = px;
            self.press_y = py;
            self.handle_press(px, py);
        } else if left {
            self.handle_drag(px, py);
        }
        if released {
            self.handle_release(px, py);
        }
        self.last_left = left;
    }

    fn handle_press(&mut self, px: i32, py: i32) {
        let r = self.regions(self.w, self.h);
        // 1. Floating windows (topmost first): close, title-drag, body.
        let hit_win = self.windows.iter().rposition(|win| {
            Rect::new(win.x, win.y, win.w, win.h).contains(px, py)
        });
        if let Some(idx) = hit_win {
            let win = &self.windows[idx];
            let (wx, wy, ww) = (win.x, win.y, win.w);
            let id = win.id;
            // Bring to front.
            let win = self.windows.remove(idx);
            self.windows.push(win);
            self.dmg_all();
            // Close button (top-right).
            if Rect::new(wx + ww - 28, wy + 6, 22, 22).contains(px, py) {
                self.windows.retain(|w| w.id != id);
                return;
            }
            // Title bar → drag.
            if py < wy + 28 {
                self.drag = Drag::Window(id, px - wx, py - wy);
            }
            return;
        }
        // 2. Panel resize grips + collapse buttons.
        if !r.left_collapsed {
            let grip = Rect::new(r.left.x + r.left.w - RESIZE_GRIP, r.left.y, RESIZE_GRIP * 2, r.left.h);
            if grip.contains(px, py) {
                self.drag = Drag::Resize(Side::Left);
                return;
            }
        }
        if !r.right_collapsed {
            let grip = Rect::new(r.right.x - RESIZE_GRIP, r.right.y, RESIZE_GRIP * 2, r.right.h);
            if grip.contains(px, py) {
                self.drag = Drag::Resize(Side::Right);
                return;
            }
        }
        // Collapse toggles (a chevron at the top-inner corner of each panel header).
        if Rect::new(r.left.x + r.left.w - 22, r.left.y + 6, 18, 18).contains(px, py) {
            self.left_collapsed = !self.left_collapsed;
            self.dmg_all();
            return;
        }
        if Rect::new(r.right.x + 4, r.right.y + 6, 18, 18).contains(px, py) {
            self.right_collapsed = !self.right_collapsed;
            self.dmg_all();
            return;
        }
        // 3. Left panel: caps, NDN, settings.
        if r.left.contains(px, py) {
            if !r.left_collapsed {
                self.press_left(px, py, &r);
            }
            return;
        }
        // 4. Right panel: log scroll.
        if r.right.contains(px, py) && !r.right_collapsed {
            let log = self.log_rect(&r);
            if log.contains(px, py) {
                if py < log.y + log.h / 2 {
                    self.log_scroll = (self.log_scroll + 3).min(self.logs.len());
                } else {
                    self.log_scroll = self.log_scroll.saturating_sub(3);
                }
                self.dmg(r.right);
            }
            return;
        }
        // 5. Centre: the node graph (canvas coords are centre-relative).
        if r.center.contains(px, py) {
            match self.graph.on_press(px - r.center.x, py - r.center.y) {
                Press::Node(_) | Press::Empty | Press::Port => self.drag = Drag::Graph,
            }
            self.dmg(r.center);
        }
    }

    fn press_left(&mut self, px: i32, py: i32, r: &Regions) {
        // Settings: theme toggle + accent slider.
        let theme_btn = Rect::new(r.left.x + 12, r.left.y + r.left.h - 70, 110, 26);
        if theme_btn.contains(px, py) {
            self.theme_dark = !self.theme_dark;
            self.dmg_all();
            return;
        }
        if self.accent_track(r).contains(self.press_x, self.press_y) {
            self.drag = Drag::Slider;
            self.accent.drag_to(self.accent_track(r), px);
            self.dmg(r.left);
            return;
        }
        // Capability rows → open a detail window.
        let rows = self.cap_rows(r);
        for (i, rr) in rows.iter().enumerate() {
            if rr.contains(px, py) {
                self.cap_selected = Some(i);
                self.open_cap(i);
                self.dmg(r.left);
                return;
            }
        }
        // NDN tree: branch toggles expand; a leaf opens a detail window.
        let (tree_area, vis) = self.ndn_layout(r);
        if tree_area.contains(px, py) {
            let idx = ((py - tree_area.y) / 20) as usize;
            if idx < vis.len() {
                if vis[idx].branch {
                    let path = vis[idx].path.clone();
                    toggle_ndn(&mut self.ndn, &path);
                } else {
                    self.open_ndn(vis[idx].label.clone());
                }
                self.dmg_all();
            }
        }
    }

    fn handle_drag(&mut self, px: i32, py: i32) {
        match self.drag {
            Drag::Window(id, ox, oy) => {
                if let Some(win) = self.windows.iter_mut().find(|w| w.id == id) {
                    // Inflate by the window decoration so the 1px primary border ring
                    // (drawn one pixel *outside* the window rect) is always repainted —
                    // otherwise its old position leaves a trailing ghost outline.
                    let old = inflate(Rect::new(win.x, win.y, win.w, win.h), WIN_DECOR);
                    win.x = px - ox;
                    win.y = py - oy;
                    let new = inflate(Rect::new(win.x, win.y, win.w, win.h), WIN_DECOR);
                    self.damage = Some(match self.damage {
                        Some(d) => union(union(d, old), new),
                        None => union(old, new),
                    });
                }
            }
            Drag::Resize(side) => {
                match side {
                    Side::Left => self.left_w = px.clamp(160, (self.w / 2).max(160)),
                    Side::Right => self.right_w = (self.w - px).clamp(180, (self.w / 2).max(180)),
                }
                self.dmg_all();
            }
            Drag::Slider => {
                let r = self.regions(self.w, self.h);
                self.accent.drag_to(self.accent_track(&r), px);
                self.dmg_all(); // accent re-tints everything
            }
            Drag::Graph => {
                let r = self.regions(self.w, self.h);
                self.graph.on_drag(px - r.center.x, py - r.center.y);
                self.dmg(r.center);
            }
            Drag::None => {}
        }
    }

    fn handle_release(&mut self, px: i32, py: i32) {
        let r = self.regions(self.w, self.h);
        if let Drag::Graph = self.drag {
            let made = self.graph.on_release(px - r.center.x, py - r.center.y);
            // A click on a node that didn't move (and wasn't a wire) opens it.
            let moved = (px - self.press_x).abs() + (py - self.press_y).abs();
            if !made && moved < 5 {
                if let Some(id) = self.graph.selected() {
                    if r.center.contains(px, py) {
                        self.open_node(id);
                    }
                }
            }
            self.dmg(r.center);
        }
        self.drag = Drag::None;
    }

    pub fn on_key(&mut self, ch: char) {
        match ch {
            't' => {
                self.theme_dark = !self.theme_dark;
                self.dmg_all();
            }
            '\x11' => {
                self.log_scroll = (self.log_scroll + 5).min(self.logs.len());
                self.dmg_all();
            }
            '\x12' => {
                self.log_scroll = self.log_scroll.saturating_sub(5);
                self.dmg_all();
            }
            'w' => {
                self.windows.pop();
                self.dmg_all();
            }
            _ => {}
        }
    }

    // ── window management ──

    fn open_node(&mut self, id: u32) {
        self.focus_or_open(WinKind::Node(id), self.node_title(id));
    }
    fn open_cap(&mut self, i: usize) {
        let t = self.caps.get(i).map(|c| c.label.clone()).unwrap_or_default();
        self.focus_or_open(WinKind::Cap(i), t);
    }
    fn open_ndn(&mut self, label: String) {
        self.focus_or_open(WinKind::Ndn(label.clone()), label);
    }

    fn focus_or_open(&mut self, kind: WinKind, title: String) {
        // If a window of the same kind is already open, bring it to front.
        let same = self.windows.iter().position(|w| win_same(&w.kind, &kind));
        if let Some(idx) = same {
            let win = self.windows.remove(idx);
            self.windows.push(win);
        } else {
            let n = self.windows.len() as i32;
            let cx = self.regions(self.w, self.h).center;
            let win = Win {
                id: self.next_win_id,
                x: cx.x + 40 + n * 28,
                y: cx.y + 40 + n * 28,
                w: 420,
                h: 300,
                title,
                kind,
            };
            self.next_win_id += 1;
            self.windows.push(win);
        }
        self.dmg_all();
    }

    fn node_title(&self, id: u32) -> String {
        self.graph.nodes().iter().find(|n| n.id == id).map(|n| n.title.clone()).unwrap_or_default()
    }

    // ── geometry helpers ──

    fn accent_track(&self, r: &Regions) -> Rect {
        Rect::new(r.left.x + 12, r.left.y + r.left.h - 32, r.left.w - 24, 16)
    }
    fn log_rect(&self, r: &Regions) -> Rect {
        let h = r.right.h * 45 / 100;
        Rect::new(r.right.x + 10, r.right.y + r.right.h - h - 8, r.right.w - 20, h)
    }
    fn cap_rows(&self, r: &Regions) -> Vec<Rect> {
        let area = Rect::new(r.left.x + 12, r.left.y + 44, r.left.w - 24, 150);
        let row_h = 22;
        let n = (area.h / row_h).min(self.caps.len() as i32);
        (0..n).map(|i| Rect::new(area.x, area.y + i * row_h, area.w, row_h - 2)).collect()
    }
    fn ndn_layout(&self, r: &Regions) -> (Rect, Vec<NdnVisRow>) {
        let area = Rect::new(r.left.x + 12, r.left.y + 230, r.left.w - 24, r.left.h - 320);
        let mut rows = Vec::new();
        let mut path = Vec::new();
        flatten_ndn(&self.ndn, 0, &mut path, &mut rows);
        (area, rows)
    }

    // ── rendering ──

    pub fn view(&self, w: i32, h: i32) -> Vec<DrawCmd> {
        let theme = self.theme();
        let r = self.regions(w, h);
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, w, h), color: theme.bg, radius: 0 });

        // Node graph in the centre (clipped by the opaque side panels drawn after).
        let mut g = self.graph.view(&theme);
        offset_scene(&mut g, r.center.x, r.center.y);
        s.append(&mut g);

        // Floating windows (bottom-to-top).
        for win in &self.windows {
            self.draw_window(&mut s, win, &theme);
        }

        self.draw_left(&mut s, &r, &theme);
        self.draw_right(&mut s, &r, &theme);
        self.draw_title(&mut s, &r, &theme);
        self.draw_status(&mut s, &r, &theme);
        s
    }

    fn draw_title(&self, s: &mut Vec<DrawCmd>, r: &Regions, t: &Theme) {
        s.push(DrawCmd::Rect { rect: r.title, color: t.surface, radius: 0 });
        s.push(DrawCmd::Text { rect: Rect::new(20, 10, 600, 22), text: "DominionOS".into(), color: t.text, size: 20 });
        s.push(DrawCmd::Text {
            rect: Rect::new(20, 32, 700, 16),
            text: "A Verified, Heterogeneous, and Semantic Operating System".into(),
            color: t.muted,
            size: 13,
        });
    }

    fn draw_status(&self, s: &mut Vec<DrawCmd>, r: &Regions, t: &Theme) {
        s.push(DrawCmd::Rect { rect: r.status, color: t.surface, radius: 0 });
        let mut x = 16;
        let cy = r.status.y + r.status.h / 2;
        let chip = |s: &mut Vec<DrawCmd>, label: &str, x: &mut i32| {
            s.push(toolkit::disc(*x + 4, cy, 4, Color::rgb(0x3f, 0xc9, 0xb0)));
            let wpx = label.len() as i32 * 9 + 14;
            s.push(DrawCmd::Text { rect: Rect::new(*x + 12, r.status.y + 7, wpx, 16), text: label.into(), color: t.muted, size: 13 });
            *x += wpx + 16;
        };
        let mut det = String::from("Deterministic Exec (");
        det.push_str(if self.metrics.det_hash.is_empty() { "AAB3" } else { &self.metrics.det_hash });
        det.push(')');
        chip(s, &det, &mut x);
        chip(s, "Verified Boot", &mut x);
        chip(s, "SASOS Active", &mut x);
        chip(s, "JSCM Audio: EDF", &mut x);
        let mut fps = String::new();
        push_int(&mut fps, self.metrics.fps as i64);
        fps.push_str(" fps");
        s.push(DrawCmd::Text { rect: Rect::new(r.status.w - 90, r.status.y + 7, 80, 16), text: fps, color: t.muted, size: 13 });
    }

    fn draw_left(&self, s: &mut Vec<DrawCmd>, r: &Regions, t: &Theme) {
        s.push(DrawCmd::Rect { rect: r.left, color: t.surface, radius: 0 });
        if r.left_collapsed {
            // Thin strip with an expand chevron.
            s.push(DrawCmd::Text { rect: Rect::new(r.left.x + 4, r.left.y + 6, 16, 18), text: ">".into(), color: t.muted, size: 15 });
            return;
        }
        // Collapse chevron (inner corner).
        s.push(DrawCmd::Text { rect: Rect::new(r.left.x + r.left.w - 20, r.left.y + 6, 16, 18), text: "<".into(), color: t.muted, size: 14 });
        s.push(DrawCmd::Text { rect: Rect::new(r.left.x + 12, r.left.y + 12, r.left.w - 36, 18), text: "Capabilities & Identity".into(), color: t.text, size: 15 });

        let rows = self.cap_rows(r);
        for (i, rr) in rows.iter().enumerate() {
            let sel = self.cap_selected == Some(i);
            let cap = &self.caps[i];
            if sel {
                s.push(DrawCmd::Rect { rect: *rr, color: t.primary, radius: 6 });
            }
            s.push(toolkit::disc(rr.x + 8, rr.y + rr.h / 2, 3, t.accent));
            s.push(DrawCmd::Text { rect: Rect::new(rr.x + 18, rr.y + 3, rr.w - 60, 16), text: cap.label.clone(), color: if sel { t.on_primary } else { t.text }, size: 12 });
            s.push(DrawCmd::Text { rect: Rect::new(rr.x + rr.w - 56, rr.y + 3, 52, 16), text: cap.rights.clone(), color: if sel { t.on_primary } else { t.muted }, size: 12 });
        }
        s.push(DrawCmd::Text { rect: Rect::new(r.left.x + 12, r.left.y + 206, r.left.w - 20, 16), text: "NDN Namespace & Network".into(), color: t.text, size: 14 });
        let (area, vis) = self.ndn_layout(r);
        for (i, row) in vis.iter().enumerate() {
            let y = area.y + i as i32 * 20;
            if y > area.y + area.h - 20 {
                break;
            }
            let x = area.x + row.depth * 14;
            if row.branch {
                s.push(DrawCmd::Text { rect: Rect::new(x, y + 2, 12, 16), text: if row.expanded { "-".into() } else { "+".into() }, color: t.muted, size: 12 });
            } else {
                s.push(toolkit::disc(x + 4, y + 10, 2, t.accent));
            }
            s.push(DrawCmd::Text { rect: Rect::new(x + 16, y + 2, area.w - row.depth * 14 - 16, 16), text: row.label.clone(), color: t.text, size: 12 });
        }
        // Settings strip.
        let theme_btn = Rect::new(r.left.x + 12, r.left.y + r.left.h - 70, 110, 26);
        s.push(DrawCmd::Rect { rect: theme_btn, color: t.primary, radius: 6 });
        s.push(DrawCmd::Text { rect: Rect::new(theme_btn.x + 10, theme_btn.y + 6, 100, 16), text: if self.theme_dark { "Theme: Dark".into() } else { "Theme: Light".into() }, color: t.on_primary, size: 12 });
        s.push(DrawCmd::Text { rect: Rect::new(r.left.x + 132, r.left.y + r.left.h - 66, 60, 16), text: "Accent".into(), color: t.muted, size: 12 });
        s.append(&mut self.accent.view(self.accent_track(r), t.primary, t));
    }

    fn draw_right(&self, s: &mut Vec<DrawCmd>, r: &Regions, t: &Theme) {
        s.push(DrawCmd::Rect { rect: r.right, color: t.surface, radius: 0 });
        if r.right_collapsed {
            s.push(DrawCmd::Text { rect: Rect::new(r.right.x + 4, r.right.y + 6, 16, 18), text: "<".into(), color: t.muted, size: 15 });
            return;
        }
        s.push(DrawCmd::Text { rect: Rect::new(r.right.x + 4, r.right.y + 6, 16, 18), text: ">".into(), color: t.muted, size: 14 });
        s.push(DrawCmd::Text { rect: Rect::new(r.right.x + 24, r.right.y + 12, r.right.w - 30, 18), text: "Compute & System Health".into(), color: t.text, size: 15 });
        let bars = if self.metrics.compute_bars.is_empty() { alloc::vec![3, 6, 2, 8, 5, 7, 4, 9] } else { self.metrics.compute_bars.clone() };
        let chart = Rect::new(r.right.x + 12, r.right.y + 38, r.right.w - 24, 90);
        s.append(&mut widgets::bar_chart(chart, &bars, 10, t.primary, t));
        let gx = r.right.x + 12;
        let gw = r.right.w - 24;
        let mut gy = r.right.y + 140;
        for (label, v) in [
            ("CPU", self.metrics.cpu_milli),
            ("Memory", self.metrics.mem_milli),
            ("GPU", self.metrics.gpu_milli),
            ("NPU", self.metrics.npu_milli),
            ("Entropy health", self.metrics.entropy_milli),
        ] {
            s.append(&mut widgets::gauge(Rect::new(gx, gy, gw, 26), label, v, t.primary, t));
            gy += 34;
        }
        s.push(DrawCmd::Text { rect: Rect::new(r.right.x + 12, gy + 4, r.right.w - 20, 16), text: "Real-time Logging".into(), color: t.text, size: 14 });
        s.append(&mut widgets::log_view(self.log_rect(r), &self.logs, self.log_scroll, t));
    }

    /// Draw a floating window: chrome + close + per-kind content.
    fn draw_window(&self, s: &mut Vec<DrawCmd>, win: &Win, t: &Theme) {
        let app = Rect::new(win.x, win.y, win.w, win.h);
        s.push(DrawCmd::Rect { rect: Rect::new(app.x - 1, app.y - 1, app.w + 2, app.h + 2), color: t.primary, radius: t.radius + 1 });
        s.push(DrawCmd::Rect { rect: app, color: t.surface, radius: t.radius });
        s.push(DrawCmd::Rect { rect: Rect::new(app.x, app.y, app.w, 28), color: t.bg, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(app.x + 12, app.y + 6, app.w - 50, 18), text: win.title.clone(), color: t.text, size: 13 });
        let close = Rect::new(app.x + app.w - 28, app.y + 6, 22, 22);
        s.push(DrawCmd::Rect { rect: close, color: t.danger, radius: 6 });
        s.push(DrawCmd::Text { rect: Rect::new(close.x + 7, close.y + 3, 16, 16), text: "x".into(), color: t.on_primary, size: 14 });

        let body = Rect::new(app.x + 12, app.y + 36, app.w - 24, app.h - 46);
        match &win.kind {
            WinKind::Node(id) => self.draw_node_content(s, body, *id, t),
            WinKind::Cap(i) => self.draw_cap_content(s, body, *i, t),
            WinKind::Ndn(label) => self.draw_ndn_content(s, body, label, t),
        }
    }

    /// Per-node application content — each node shows *its own* data, not a shared one.
    fn draw_node_content(&self, s: &mut Vec<DrawCmd>, body: Rect, id: u32, t: &Theme) {
        let kind = self.graph.nodes().iter().find(|n| n.id == id).map(|n| n.kind);
        match kind {
            Some(NodeKind::Data) => {
                s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 16), text: "Semantic Table View".into(), color: t.muted, size: 12 });
                let table = [("Month", "Revenue"), ("Jan", "$42k"), ("Feb", "$55k"), ("Mar", "$61k"), ("Apr", "$78k")];
                for (i, (a, b)) in table.iter().enumerate() {
                    let ry = body.y + 22 + i as i32 * 22;
                    if i == 0 {
                        s.push(DrawCmd::Rect { rect: Rect::new(body.x, ry - 2, body.w, 20), color: t.bg, radius: 4 });
                    }
                    s.push(DrawCmd::Text { rect: Rect::new(body.x + 6, ry, body.w / 2, 16), text: (*a).into(), color: t.text, size: 12 });
                    s.push(DrawCmd::Text { rect: Rect::new(body.x + body.w / 2, ry, body.w / 2, 16), text: (*b).into(), color: t.text, size: 12 });
                }
            }
            Some(NodeKind::Report) => {
                s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 16), text: "Neural Graph View".into(), color: t.muted, size: 12 });
                let data = if self.metrics.cpu_history.len() >= 2 { self.metrics.cpu_history.clone() } else { alloc::vec![42, 55, 61, 78, 70, 88, 95, 120, 110] };
                s.append(&mut widgets::line_chart(Rect::new(body.x, body.y + 20, body.w, body.h - 24), &data, t.accent, t));
            }
            Some(NodeKind::Audio) => {
                s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 16), text: "Neural Audio Object".into(), color: t.muted, size: 12 });
                // A little waveform.
                let wave: Vec<i64> = (0..40).map(|i| (((i * 37) % 19) - 9) as i64).collect();
                s.append(&mut widgets::sparkline(Rect::new(body.x, body.y + 24, body.w, 60), &wave, Color::rgb(0x3f, 0xc9, 0xb0)));
                s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y + 96, body.w, 16), text: "transcript: \"summarise Q3 sales\"".into(), color: t.muted, size: 11 });
            }
            Some(NodeKind::Log) => {
                s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 16), text: "System Log Stream".into(), color: t.muted, size: 12 });
                s.append(&mut widgets::log_view(Rect::new(body.x, body.y + 20, body.w, body.h - 24), &self.logs, 0, t));
            }
            _ => {
                // App / Document: the orb + a caption.
                s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 16), text: "Natural Language".into(), color: t.muted, size: 12 });
                let cx = body.x + body.w / 2;
                let cy = body.y + body.h / 2;
                s.push(toolkit::disc(cx, cy, 34, Color::rgba(t.accent.r, t.accent.g, t.accent.b, 90)));
                s.push(toolkit::disc(cx, cy, 22, Color::rgba(0x3f, 0xc9, 0xb0, 120)));
                s.push(toolkit::disc(cx, cy, 12, Color::rgba(t.primary.r, t.primary.g, t.primary.b, 200)));
            }
        }
    }

    fn draw_cap_content(&self, s: &mut Vec<DrawCmd>, body: Rect, i: usize, t: &Theme) {
        let cap = match self.caps.get(i) {
            Some(c) => c,
            None => return,
        };
        s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 18), text: cap.label.clone(), color: t.text, size: 14 });
        s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y + 24, body.w, 16), text: "CHERI capability — unforgeable token".into(), color: t.muted, size: 12 });
        // Rights breakdown.
        for (i, (bit, name)) in [('r', "Read"), ('w', "Write"), ('x', "Execute")].iter().enumerate() {
            let on = cap.rights.contains(*bit);
            let y = body.y + 52 + i as i32 * 22;
            s.push(toolkit::disc(body.x + 6, y + 7, 4, if on { t.primary } else { t.muted }));
            s.push(DrawCmd::Text { rect: Rect::new(body.x + 18, y, body.w - 20, 16), text: name.to_string(), color: if on { t.text } else { t.muted }, size: 12 });
        }
        // Controls.
        let revoke = Rect::new(body.x, body.y + body.h - 30, 90, 24);
        s.push(DrawCmd::Rect { rect: revoke, color: t.danger, radius: 6 });
        s.push(DrawCmd::Text { rect: Rect::new(revoke.x + 16, revoke.y + 4, 80, 16), text: "Revoke".into(), color: t.on_primary, size: 12 });
        let derive = Rect::new(body.x + 100, body.y + body.h - 30, 90, 24);
        s.push(DrawCmd::Rect { rect: derive, color: t.primary, radius: 6 });
        s.push(DrawCmd::Text { rect: Rect::new(derive.x + 14, derive.y + 4, 80, 16), text: "Derive".into(), color: t.on_primary, size: 12 });
    }

    fn draw_ndn_content(&self, s: &mut Vec<DrawCmd>, body: Rect, label: &str, t: &Theme) {
        s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y, body.w, 18), text: label.into(), color: t.text, size: 14 });
        let mut name = String::from("/dominion/");
        name.push_str(label);
        s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y + 24, body.w, 16), text: name, color: t.accent, size: 12 });
        s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y + 46, body.w, 16), text: "self-certifying NDN name (H(pubkey))".into(), color: t.muted, size: 12 });
        s.push(DrawCmd::Text { rect: Rect::new(body.x, body.y + 70, body.w, 16), text: "reachable · cached · verified".into(), color: t.muted, size: 12 });
        let open = Rect::new(body.x, body.y + body.h - 30, 110, 24);
        s.push(DrawCmd::Rect { rect: open, color: t.primary, radius: 6 });
        s.push(DrawCmd::Text { rect: Rect::new(open.x + 12, open.y + 4, 100, 16), text: "Fetch / Subscribe".into(), color: t.on_primary, size: 12 });
    }
}

impl Default for Dash {
    fn default() -> Self {
        Self::new()
    }
}

// ── helpers ──

fn win_same(a: &WinKind, b: &WinKind) -> bool {
    matches!((a, b),
        (WinKind::Node(x), WinKind::Node(y)) if x == y)
        || matches!((a, b), (WinKind::Cap(x), WinKind::Cap(y)) if x == y)
        || matches!((a, b), (WinKind::Ndn(x), WinKind::Ndn(y)) if x == y)
}

/// Grow a rect by `m` pixels on every side (clamped at the buffer is the caller's job).
fn inflate(r: Rect, m: i32) -> Rect {
    Rect::new(r.x - m, r.y - m, r.w + 2 * m, r.h + 2 * m)
}

fn union(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

fn offset_scene(scene: &mut [DrawCmd], dx: i32, dy: i32) {
    for cmd in scene.iter_mut() {
        match cmd {
            DrawCmd::Rect { rect, .. } | DrawCmd::Text { rect, .. } => {
                rect.x += dx;
                rect.y += dy;
            }
            DrawCmd::Line { x0, y0, x1, y1, .. } => {
                *x0 += dx;
                *y0 += dy;
                *x1 += dx;
                *y1 += dy;
            }
            DrawCmd::Bezier { p0, c0, c1, p1, .. } => {
                for p in [p0, c0, c1, p1] {
                    p.0 += dx;
                    p.1 += dy;
                }
            }
            DrawCmd::Disc { cx, cy, .. } => {
                *cx += dx;
                *cy += dy;
            }
            DrawCmd::Polyline { points, .. } => {
                for p in points.iter_mut() {
                    p.0 += dx;
                    p.1 += dy;
                }
            }
            // 3D / GPU commands carry no 2D coordinates to translate.
            DrawCmd::Mesh3D { .. }
            | DrawCmd::VectorPath { .. }
            | DrawCmd::GpuText { .. }
            | DrawCmd::MediaBuffer { .. }
            | DrawCmd::Scene3D { .. }
            | DrawCmd::Particles { .. }
            | DrawCmd::SdfShadow { .. } => {}
        }
    }
}

fn flatten_ndn(nodes: &[NdnNode], depth: i32, path: &mut Vec<usize>, out: &mut Vec<NdnVisRow>) {
    for (i, n) in nodes.iter().enumerate() {
        path.push(i);
        out.push(NdnVisRow { depth, label: n.label.clone(), branch: !n.children.is_empty(), expanded: n.expanded, path: path.clone() });
        if n.expanded {
            flatten_ndn(&n.children, depth + 1, path, out);
        }
        path.pop();
    }
}

fn toggle_ndn(nodes: &mut [NdnNode], path: &[usize]) {
    if path.is_empty() {
        return;
    }
    if let Some(n) = nodes.get_mut(path[0]) {
        if path.len() == 1 {
            n.expanded = !n.expanded;
        } else {
            toggle_ndn(&mut n.children, &path[1..]);
        }
    }
}

fn default_caps() -> Vec<CapRow> {
    [
        ("user.identity", "rwx"),
        ("npu.invoke", "r-x"),
        ("gpu.compute", "r-x"),
        ("storage.vault", "rw-"),
        ("net.dominionlink", "r--"),
        ("audio.render", "r-x"),
    ]
    .iter()
    .map(|(l, r)| CapRow { label: (*l).into(), rights: (*r).into() })
    .collect()
}

fn default_ndn() -> Vec<NdnNode> {
    alloc::vec![NdnNode::branch(
        "dominion",
        true,
        alloc::vec![
            NdnNode::branch("alpha", false, alloc::vec![NdnNode::leaf("sales.v1"), NdnNode::leaf("report.v3")]),
            NdnNode::leaf("npu"),
            NdnNode::leaf("gpu"),
            NdnNode::leaf("cpu"),
            NdnNode::leaf("syslog"),
            NdnNode::branch("users", true, alloc::vec![NdnNode::leaf("jayden")]),
        ],
    )]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn click(d: &mut Dash, x: i32, y: i32) {
        d.on_pointer(x, y, true);
        d.on_pointer(x, y, false);
    }

    #[test]
    fn dashboard_renders_all_regions() {
        let d = Dash::new();
        let scene = d.view(1280, 720);
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "DominionOS")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Capabilities & Identity")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Compute & System Health")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Bezier { .. })));
    }

    #[test]
    fn clicking_a_node_opens_a_window_with_its_own_content() {
        let mut d = Dash::new();
        assert_eq!(d.open_windows(), 0);
        let r = d.regions(1280, 720);
        // Node 1 (Data) at graph (60,60) → screen (center.x+60, center.y+60).
        click(&mut d, r.center.x + 70, r.center.y + 80);
        assert_eq!(d.open_windows(), 1);
        let scene = d.view(1280, 720);
        // Data node → a table view.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Semantic Table View")));
        // Drag the window well below the nodes so it doesn't cover the next one.
        let (wx, wy) = (d.windows[0].x, d.windows[0].y);
        d.on_pointer(wx + 40, wy + 12, true);
        d.on_pointer(r.center.x + 4, r.center.y + 400, true);
        d.on_pointer(r.center.x + 4, r.center.y + 400, false);
        // Open node 3 (Report) → a *different* content (graph view), two windows.
        click(&mut d, r.center.x + 310, r.center.y + 80);
        assert_eq!(d.open_windows(), 2);
        let scene = d.view(1280, 720);
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Neural Graph View")));
    }

    #[test]
    fn windows_are_draggable_by_their_title_bar() {
        let mut d = Dash::new();
        let r = d.regions(1280, 720);
        click(&mut d, r.center.x + 70, r.center.y + 80); // open node 1
        // The window opened near the centre; grab its title bar and drag it.
        let wx = d.windows[0].x;
        let wy = d.windows[0].y;
        d.on_pointer(wx + 40, wy + 12, true); // press title bar
        let _ = d.take_damage();
        d.on_pointer(wx + 140, wy + 92, true); // drag by (100,80)
        assert_eq!((d.windows[0].x, d.windows[0].y), (wx + 100, wy + 80));
        // The damage rect must cover BOTH the old and new window position, inflated
        // past the 1px border ring — otherwise the old border leaves a ghost outline.
        let dmg = d.take_damage().unwrap();
        assert!(dmg.x <= wx - WIN_DECOR, "damage must include the old window's left border");
        assert!(dmg.y <= wy - WIN_DECOR, "damage must include the old window's top border");
        let new_right = wx + 100 + d.windows[0].w;
        let new_bot = wy + 80 + d.windows[0].h;
        assert!(dmg.x + dmg.w >= new_right + WIN_DECOR, "damage must include the new window's right border");
        assert!(dmg.y + dmg.h >= new_bot + WIN_DECOR, "damage must include the new window's bottom border");
        d.on_pointer(wx + 140, wy + 92, false);
    }

    #[test]
    fn sidebars_collapse_and_resize() {
        let mut d = Dash::new();
        let r = d.regions(1280, 720);
        // Collapse the left panel via its chevron.
        click(&mut d, r.left.x + r.left.w - 13, r.left.y + 14);
        assert!(d.left_collapsed);
        let r2 = d.regions(1280, 720);
        assert_eq!(r2.left.w, COLLAPSED_W);
        // Expand again.
        click(&mut d, r2.left.x + 8, r2.left.y + 14);
        assert!(!d.left_collapsed);
        // Resize the left panel by dragging its grip.
        let r3 = d.regions(1280, 720);
        let grip_x = r3.left.x + r3.left.w;
        d.on_pointer(grip_x, r3.left.y + 100, true);
        d.on_pointer(grip_x + 80, r3.left.y + 100, true);
        d.on_pointer(grip_x + 80, r3.left.y + 100, false);
        assert!(d.regions(1280, 720).left.w >= grip_x + 80 - 4);
    }

    #[test]
    fn clicking_a_capability_opens_a_detail_window() {
        let mut d = Dash::new();
        let r = d.regions(1280, 720);
        let rows = d.cap_rows(&r);
        click(&mut d, rows[0].x + 30, rows[0].y + 8);
        assert_eq!(d.cap_selected, Some(0));
        assert_eq!(d.open_windows(), 1);
        let scene = d.view(1280, 720);
        // The detail window shows the rights breakdown + a Revoke control.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Revoke")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Read")));
    }

    #[test]
    fn clicking_an_ndn_leaf_opens_details() {
        let mut d = Dash::new();
        let r = d.regions(1280, 720);
        let (area, vis) = d.ndn_layout(&r);
        // "jayden" (a user leaf) is visible because users is expanded by default.
        let idx = vis.iter().position(|row| row.label == "jayden").unwrap();
        let y = area.y + idx as i32 * 20 + 4;
        click(&mut d, area.x + 30, y);
        assert_eq!(d.open_windows(), 1);
        let scene = d.view(1280, 720);
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Fetch / Subscribe")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("jayden"))));
    }

    #[test]
    fn theme_toggle_and_log_scroll_via_keys() {
        let mut d = Dash::new();
        let dark = d.theme().bg;
        d.on_key('t');
        assert_ne!(dark, d.theme().bg);
        for i in 0..40 {
            let mut l = String::from("e");
            push_int(&mut l, i);
            d.push_log(&l);
        }
        d.on_key('\x11');
        assert_eq!(d.log_scroll, 5);
    }

    #[test]
    fn metrics_feed_into_gauges_and_damage_the_right_panel_and_status_bar() {
        let mut d = Dash::new();
        let _ = d.take_damage();
        let m = Metrics { cpu_milli: 730, fps: 30, ..Default::default() };
        d.set_metrics(m);
        // Damage covers the right panel (gauges) AND the status bar (live fps text) —
        // but is still incremental, not the whole screen.
        let dmg = d.take_damage().unwrap();
        let r = d.regions(1280, 720);
        // Right panel is within the damage.
        assert!(dmg.x <= r.right.x && dmg.x + dmg.w >= r.right.x + r.right.w);
        // Status bar (bottom) is within the damage — otherwise the fps read-out freezes.
        assert!(dmg.y + dmg.h >= r.status.y + r.status.h);
        // Not a full repaint: the title bar at the very top is untouched.
        assert!(dmg.y > r.title.y, "metric tick must not repaint the whole screen");
        let scene = d.view(1280, 720);
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "73%")));
    }

    #[test]
    fn node_graph_pans_when_dragging_empty_centre() {
        let mut d = Dash::new();
        let r = d.regions(1280, 720);
        // Drag empty centre space (well away from any node).
        let ex = r.center.x + r.center.w - 30;
        let ey = r.center.y + 20;
        d.on_pointer(ex, ey, true);
        d.on_pointer(ex - 60, ey + 40, true);
        d.on_pointer(ex - 60, ey + 40, false);
        assert_eq!(d.graph.pan(), (-60, 40));
    }
}

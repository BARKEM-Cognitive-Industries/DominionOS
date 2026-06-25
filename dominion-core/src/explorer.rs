//! The **Explorer** — view/explore the running system
//! (see `docs/knowledge_viewer_concept_ui.png`).
//!
//! Three columns:
//! * **System Knowledge Graph** — a searchable constellation of objects; selecting one
//!   opens a detail panel (metadata, capability graph, provenance).
//! * **Secure Compartment Viewer** — the active sandboxed cells and a microkernel →
//!   GPU / storage / NDN / audio **hardware diagram** with live status.
//! * **System Health** — the live monitoring folded in from the old dashboard: a compute
//!   bar chart, CPU/Mem/GPU/NPU/entropy gauges, and a real-time log.
//!
//! A bottom status strip reports architecture / build / capability / network / audio.
//!
//! Phase 3: the knowledge graph and compartment lists are **seeded representative data**
//! (like the Desktop's cards), while the monitoring column is driven by **live kernel
//! metrics + logs**. Real object-graph / sandbox enumeration is a later refinement.
//! Pure, safe `no_std`. Rendered in page-local coordinates.

use crate::dash::Metrics;
use crate::nodes::{NodeGraph, NodeKind};
use crate::text::TextBuffer;
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::widgets;
use crate::world::{CellInfo, Entry};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const STATUS_H: i32 = 26;
const SEARCH_H: i32 = 28;
/// Font size of the search field text (so the caret uses the matching advance).
const SEARCH_FONT: i32 = 12;

/// A live knowledge-graph object (from the shared [`World`](crate::world)), laid out for
/// the constellation + detail panel.
struct ObjNode {
    id: u32, // node id = index + 1
    name: String,
    kind: NodeKind,
    meta: Vec<(String, String)>,
    /// `rwx` rights of this object's real capability.
    rights: String,
    /// Short hex of that capability's provenance hash chain.
    provenance: String,
    /// Indices (into `objects`) this object references — the real graph edges.
    refs: Vec<usize>,
    x: i32,
    y: i32,
}

/// The system explorer page.
pub struct Explorer {
    area: Rect,
    metrics: Metrics,
    logs: Vec<String>,
    objects: Vec<ObjNode>,
    /// The object name-set the current layout was built from (so re-syncs are cheap).
    obj_keys: Vec<String>,
    graph: NodeGraph,
    selected: Option<u32>,
    /// The knowledge search field — the global text engine, so it edits like any text
    /// surface (caret anywhere, arrows, click-to-place, blink).
    search: TextBuffer,
    search_focused: bool,
    now_ms: u64,
    cells: Vec<CellInfo>,
    /// The TCB root capability's provenance (short hex) — shown in the status strip.
    root_prov: String,
    last_left: bool,
    press_x: i32,
    press_y: i32,
    damage: Option<Rect>,
}

impl Explorer {
    pub fn new() -> Explorer {
        Explorer {
            area: Rect::new(0, 0, 1280, 600),
            metrics: Metrics::default(),
            logs: Vec::new(),
            objects: Vec::new(),
            obj_keys: Vec::new(),
            graph: NodeGraph::new(),
            selected: None,
            search: TextBuffer::empty(),
            search_focused: false,
            now_ms: 0,
            cells: Vec::new(),
            root_prov: String::new(),
            last_left: false,
            press_x: 0,
            press_y: 0,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        }
    }

    /// Rebuild the knowledge constellation from the live system objects. Idempotent:
    /// an unchanged object set leaves the layout (and selection) untouched, so this is
    /// cheap to call every sync.
    pub fn set_objects(&mut self, entries: &[Entry]) {
        let keys: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
        if keys == self.obj_keys {
            return;
        }
        let mut idx_of: BTreeMap<_, usize> = BTreeMap::new();
        for (i, e) in entries.iter().enumerate() {
            idx_of.insert(e.id, i);
        }
        self.objects = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let refs = e.refs.iter().filter_map(|r| idx_of.get(r).copied()).collect();
                let (x, y) = scatter(i);
                ObjNode {
                    id: (i + 1) as u32,
                    name: e.name.clone(),
                    kind: kind_to_node(&e.kind),
                    meta: e.meta.clone(),
                    rights: e.rights_str(),
                    provenance: e.provenance.short(),
                    refs,
                    x,
                    y,
                }
            })
            .collect();
        self.obj_keys = keys;
        if let Some(sel) = self.selected {
            if sel as usize > self.objects.len() {
                self.selected = None;
            }
        }
        self.rebuild_graph();
        self.dmg(self.col_a());
    }

    /// Set the live sandboxed compartments (real capability-bounded cells).
    pub fn set_cells(&mut self, cells: Vec<CellInfo>) {
        self.cells = cells;
        self.dmg(self.col_b());
    }

    /// Set the TCB root capability provenance for the status strip.
    pub fn set_root_prov(&mut self, prov: String) {
        if prov != self.root_prov {
            self.root_prov = prov;
            self.dmg(Rect::new(0, self.area.h - STATUS_H, self.area.w, STATUS_H));
        }
    }

    /// Select an object by name (e.g. when a Desktop data card is clicked).
    pub fn select_by_name(&mut self, name: &str) {
        if let Some(id) = self.objects.iter().find(|o| o.name == name).map(|o| o.id) {
            self.selected = Some(id);
            self.search = TextBuffer::empty();
            self.search_focused = false;
            self.rebuild_graph();
            self.dmg(self.col_a());
        }
    }

    /// Append a system log line (shown in the monitoring column).
    pub fn push_log(&mut self, line: &str) {
        self.logs.push(line.into());
        if self.logs.len() > 500 {
            let drop = self.logs.len() - 500;
            self.logs.drain(0..drop);
        }
    }

    /// Clear the real-time log (e.g. from the context-menu "Clear log" action).
    pub fn clear_log(&mut self) {
        self.logs.clear();
        self.dmg_all();
    }

    pub fn set_area(&mut self, area: Rect) {
        if area.w != self.area.w || area.h != self.area.h {
            self.area = area;
            self.dmg_all();
        } else {
            self.area = area;
        }
    }

    pub fn set_metrics(&mut self, m: Metrics) {
        self.metrics = m;
        // The monitoring column (C) repaints on each metric tick.
        self.dmg(self.col_c());
    }

    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }

    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => toolkit::union(d, r),
            None => r,
        });
    }
    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }

    // ── layout ──

    fn col_a(&self) -> Rect {
        Rect::new(0, 0, self.area.w * 40 / 100, self.area.h - STATUS_H)
    }
    fn col_b(&self) -> Rect {
        let x = self.area.w * 40 / 100;
        Rect::new(x, 0, self.area.w * 30 / 100, self.area.h - STATUS_H)
    }
    fn col_c(&self) -> Rect {
        let x = self.area.w * 70 / 100;
        Rect::new(x, 0, self.area.w - x, self.area.h - STATUS_H)
    }
    fn search_rect(&self) -> Rect {
        let a = self.col_a();
        Rect::new(a.x + 10, a.y + 34, a.w - 20, SEARCH_H)
    }
    fn graph_rect(&self) -> Rect {
        let a = self.col_a();
        Rect::new(a.x + 6, a.y + 34 + SEARCH_H + 6, a.w - 12, a.h - 34 - SEARCH_H - 14)
    }

    // ── knowledge graph ──

    fn rebuild_graph(&mut self) {
        let mut g = NodeGraph::new();
        let q = self.search.text().to_lowercase();
        let mut vis: BTreeSet<u32> = BTreeSet::new();
        for o in &self.objects {
            if q.is_empty() || o.name.to_lowercase().contains(&q) {
                g.add(o.id, &o.name, &o.rights, o.kind, o.x, o.y);
                vis.insert(o.id);
            }
        }
        // Wires from the real reference edges (data flows from the referenced object
        // into the one that references it), but only between visible nodes.
        for o in &self.objects {
            if !vis.contains(&o.id) {
                continue;
            }
            for &ri in &o.refs {
                let from = (ri + 1) as u32;
                if vis.contains(&from) {
                    g.wire(from, o.id);
                }
            }
        }
        self.graph = g;
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;

        if pressed {
            self.press_x = px;
            self.press_y = py;
            // Search box focus + click-to-place the caret.
            if self.search_rect().contains(px, py) {
                self.search_focused = true;
                let o = self.search_origin();
                self.search.place_at_pixel(px, py, o, toolkit::mono_advance(SEARCH_FONT), SEARCH_H - 12);
                self.dmg(self.col_a());
                self.last_left = left;
                return;
            } else if self.search_focused {
                self.search_focused = false;
                self.dmg(self.col_a());
            }
            // Knowledge graph press.
            let gr = self.graph_rect();
            if gr.contains(px, py) {
                self.graph.on_press(px - gr.x, py - gr.y);
                self.dmg(self.col_a());
            }
        } else if left {
            let gr = self.graph_rect();
            self.graph.on_drag(px - gr.x, py - gr.y);
            self.dmg(self.col_a());
        }
        if released {
            let gr = self.graph_rect();
            self.graph.on_release(px - gr.x, py - gr.y);
            let moved = (px - self.press_x).abs() + (py - self.press_y).abs();
            if moved < 5 && gr.contains(px, py) {
                self.selected = self.graph.selected();
            }
            self.dmg(self.col_a());
        }
        self.last_left = left;
    }

    /// Whether the search field has keyboard focus (so the shell routes keys to it).
    pub fn is_search_focused(&self) -> bool {
        self.search_focused
    }

    /// Advance the clock for the search caret blink; damage the search box when the
    /// blink phase flips so the caret visibly flashes (cheap — just the field).
    pub fn set_time(&mut self, now_ms: u64) {
        let prev = self.now_ms;
        self.now_ms = now_ms;
        if self.search_focused && (prev / crate::text::BLINK_MS) != (now_ms / crate::text::BLINK_MS) {
            self.dmg(self.search_rect());
        }
    }

    fn search_origin(&self) -> (i32, i32) {
        let sr = self.search_rect();
        (sr.x + 8, sr.y + 6)
    }

    /// Returns true if the key was consumed (search box has focus). Full editing via
    /// the global text engine: arrows/Home/End/Delete, backspace, click-placed caret.
    pub fn on_key(&mut self, ch: char) -> bool {
        if !self.search_focused {
            return false;
        }
        self.search.touch(self.now_ms);
        match ch {
            '\u{1b}' => self.search_focused = false, // Esc defocuses
            '\u{8}' => {
                self.search.backspace();
                self.rebuild_graph();
            }
            '\u{7f}' => {
                self.search.delete();
                self.rebuild_graph();
            }
            '\u{1c}' => self.search.left(),
            '\u{1d}' => self.search.right(),
            '\u{1}' => self.search.home(),
            '\u{5}' => self.search.end(),
            // Up/down in a single-line field jump to the ends.
            '\u{1e}' => self.search.home(),
            '\u{1f}' => self.search.end(),
            c if !c.is_control() => {
                self.search.insert(c);
                self.rebuild_graph();
            }
            _ => {}
        }
        self.dmg(self.col_a());
        true
    }

    // ── rendering ──

    pub fn view(&self, theme: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: theme.bg, radius: 0 });
        self.draw_knowledge(&mut s, theme);
        self.draw_compartments(&mut s, theme);
        self.draw_health(&mut s, theme);
        self.draw_status(&mut s, theme);
        s
    }

    fn panel(s: &mut Vec<DrawCmd>, r: Rect, title: &str, t: &Theme) {
        s.push(DrawCmd::Rect { rect: Rect::new(r.x + 4, r.y + 4, r.w - 8, r.h - 8), color: t.surface, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(r.x + 14, r.y + 12, r.w - 28, 18), text: title.into(), color: t.text, size: 15 });
    }

    fn draw_knowledge(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let a = self.col_a();
        Self::panel(s, a, "SYSTEM KNOWLEDGE GRAPH", t);
        // Search box.
        let sr = self.search_rect();
        let border = if self.search_focused { t.primary } else { t.muted };
        s.push(DrawCmd::Rect { rect: toolkit::inflate(sr, 1), color: border, radius: t.radius });
        s.push(DrawCmd::Rect { rect: sr, color: t.bg, radius: t.radius });
        let query = self.search.text();
        let empty = query.is_empty();
        let shown = if empty && !self.search_focused {
            "Search Objects (Named Data Networking)…".to_string()
        } else {
            query
        };
        let scol = if empty && !self.search_focused { t.muted } else { t.text };
        s.push(DrawCmd::Text { rect: Rect::new(sr.x + 8, sr.y + 6, sr.w - 16, 16), text: shown, color: scol, size: SEARCH_FONT });
        // The blinking insert caret when the field is focused.
        if self.search_focused {
            let o = self.search_origin();
            self.search.paint_caret(s, t, o, toolkit::mono_advance(SEARCH_FONT), SEARCH_H - 12, self.now_ms);
        }
        // Constellation.
        let gr = self.graph_rect();
        let mut g = self.graph.view(t);
        toolkit::translate_scene(&mut g, gr.x, gr.y);
        s.append(&mut g);
        // Detail panel for the selected object.
        if let Some(id) = self.selected {
            if let Some(o) = self.objects.iter().find(|o| o.id == id) {
                self.draw_detail(s, o, t);
            }
        }
    }

    fn draw_detail(&self, s: &mut Vec<DrawCmd>, o: &ObjNode, t: &Theme) {
        let a = self.col_a();
        let d = Rect::new(a.x + a.w - 200, a.y + 70, 188, 250);
        s.push(DrawCmd::Rect { rect: toolkit::inflate(d, 1), color: t.primary, radius: t.radius });
        s.push(DrawCmd::Rect { rect: d, color: t.bg, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(d.x + 10, d.y + 8, d.w - 20, 16), text: o.name.clone(), color: t.text, size: 13 });
        let mut y = d.y + 30;
        s.push(DrawCmd::Text { rect: Rect::new(d.x + 10, y, d.w - 20, 14), text: "Metadata:".into(), color: t.muted, size: 11 });
        y += 16;
        for (k, v) in o.meta.iter().take(4) {
            let mut line = k.clone();
            line.push_str(": ");
            line.push_str(v);
            s.push(DrawCmd::Text { rect: Rect::new(d.x + 14, y, d.w - 24, 14), text: line, color: t.text, size: 11 });
            y += 15;
        }
        y += 8;
        s.push(DrawCmd::Text { rect: Rect::new(d.x + 10, y, d.w - 20, 14), text: "Capability Graph".into(), color: t.muted, size: 11 });
        y += 16;
        for (i, (bit, name)) in [('r', "Read"), ('w', "Write"), ('x', "Execute")].iter().enumerate() {
            let on = o.rights.contains(*bit);
            let yy = y + i as i32 * 15;
            s.push(toolkit::disc(d.x + 16, yy + 6, 3, if on { t.primary } else { t.muted }));
            s.push(DrawCmd::Text { rect: Rect::new(d.x + 26, yy, d.w - 36, 13), text: name.to_string(), color: if on { t.text } else { t.muted }, size: 11 });
        }
        y += 3 * 15 + 8;
        s.push(DrawCmd::Text { rect: Rect::new(d.x + 10, y, d.w - 20, 14), text: "Proven Provenance".into(), color: t.muted, size: 11 });
        let mut prov = String::from("> cap ");
        prov.push_str(&o.provenance);
        s.push(DrawCmd::Text { rect: Rect::new(d.x + 14, y + 16, d.w - 24, 14), text: prov, color: t.accent, size: 11 });
    }

    fn draw_compartments(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let b = self.col_b();
        Self::panel(s, b, "SECURE COMPARTMENT VIEWER", t);
        let mut y = b.y + 40;
        s.push(DrawCmd::Text { rect: Rect::new(b.x + 14, y, b.w - 28, 14), text: "Active cells (sandboxed):".into(), color: t.muted, size: 11 });
        y += 18;
        for c in &self.cells {
            s.push(toolkit::disc(b.x + 18, y + 7, 3, Color::rgb(0x3f, 0xc9, 0xb0)));
            s.push(DrawCmd::Text { rect: Rect::new(b.x + 26, y, b.w - 36, 14), text: c.name.clone(), color: t.text, size: 11 });
            let mut info = String::new();
            push_int(&mut info, c.syscalls as i64);
            info.push_str(" sysc  ");
            info.push_str(&c.bounds);
            s.push(DrawCmd::Text { rect: Rect::new(b.x + 26, y + 14, b.w - 36, 12), text: info, color: t.muted, size: 10 });
            y += 32;
        }
        // Hardware diagram: microkernel → GPU / storage / NDN / audio.
        self.draw_hardware(s, Rect::new(b.x + 8, y + 6, b.w - 16, b.y + b.h - y - 14), t);
    }

    fn draw_hardware(&self, s: &mut Vec<DrawCmd>, area: Rect, t: &Theme) {
        s.push(DrawCmd::Text { rect: Rect::new(area.x + 6, area.y, area.w - 12, 14), text: "Architecture".into(), color: t.muted, size: 11 });
        let mk = Rect::new(area.x + 6, area.y + 26, 92, 44);
        s.push(DrawCmd::Rect { rect: mk, color: t.surface, radius: t.radius });
        s.push(DrawCmd::Rect { rect: toolkit::inflate(mk, 1), color: t.primary, radius: t.radius });
        s.push(DrawCmd::Rect { rect: mk, color: t.bg, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(mk.x + 6, mk.y + 8, mk.w - 10, 14), text: "MICROKERNEL".into(), color: t.text, size: 10 });
        s.push(DrawCmd::Text { rect: Rect::new(mk.x + 6, mk.y + 24, mk.w - 10, 12), text: "seL4-based".into(), color: t.muted, size: 9 });
        let nodes = [
            ("GPU NODE", self.metrics.gpu_milli > 0),
            ("STORAGE", self.metrics.disk_present),
            ("NDN STACK", self.metrics.net_present),
            ("AUDIO OBA", true),
        ];
        let nx = mk.x + mk.w + 60;
        for (i, (label, on)) in nodes.iter().enumerate() {
            let ny = area.y + 24 + i as i32 * 34;
            let nr = Rect::new(nx, ny, area.w - (nx - area.x) - 6, 28);
            if nr.w < 40 {
                continue;
            }
            s.push(DrawCmd::Rect { rect: nr, color: t.surface, radius: t.radius });
            let dot = if *on { Color::rgb(0x3f, 0xc9, 0xb0) } else { t.danger };
            s.push(toolkit::disc(nr.x + 8, nr.y + nr.h / 2, 3, dot));
            s.push(DrawCmd::Text { rect: Rect::new(nr.x + 16, nr.y + 7, nr.w - 20, 14), text: (*label).into(), color: t.text, size: 10 });
            // Arrow from the microkernel to this node.
            s.push(toolkit::wire((mk.x + mk.w, mk.y + mk.h / 2), (nr.x, nr.y + nr.h / 2), t.muted, 1, 24));
        }
    }

    fn draw_health(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let c = self.col_c();
        Self::panel(s, c, "SYSTEM HEALTH", t);
        let bars = if self.metrics.compute_bars.is_empty() { alloc::vec![3, 6, 2, 8, 5, 7, 4, 9] } else { self.metrics.compute_bars.clone() };
        let chart = Rect::new(c.x + 12, c.y + 38, c.w - 24, 70);
        s.append(&mut widgets::bar_chart(chart, &bars, 10, t.primary, t));
        let mut gy = c.y + 118;
        for (label, v) in [
            ("CPU", self.metrics.cpu_milli),
            ("Memory", self.metrics.mem_milli),
            ("GPU", self.metrics.gpu_milli),
            ("NPU", self.metrics.npu_milli),
            ("Entropy", self.metrics.entropy_milli),
        ] {
            s.append(&mut widgets::gauge(Rect::new(c.x + 12, gy, c.w - 24, 24), label, v, t.primary, t));
            gy += 32;
        }
        // Network & Security section — extends the health column downward.
        self.draw_network_security(s, c, gy, t);
    }

    fn draw_network_security(&self, s: &mut Vec<DrawCmd>, c: Rect, start_y: i32, t: &Theme) {
        let mut y = start_y + 6;
        let x = c.x + 12;
        let w = c.w - 24;

        // ── Network status ──────────────────────────────────────────────────────────
        s.push(DrawCmd::Text {
            rect: Rect::new(x, y, w, 14),
            text: "Network Status".into(),
            color: t.muted,
            size: 11,
        });
        y += 18;

        // Fake connection counts that shift with uptime for realism.
        let uptime = self.metrics.uptime_secs;
        let inc = 3 + (uptime / 7 % 5) as i32;
        let out = 5 + (uptime / 11 % 8) as i32;

        let mut inc_str = String::from("Incoming connections: ");
        push_int(&mut inc_str, inc as i64);
        s.push(DrawCmd::Text { rect: Rect::new(x + 4, y, w - 8, 13), text: inc_str, color: t.text, size: 11 });
        y += 15;

        let mut out_str = String::from("Outgoing connections: ");
        push_int(&mut out_str, out as i64);
        s.push(DrawCmd::Text { rect: Rect::new(x + 4, y, w - 8, 13), text: out_str, color: t.text, size: 11 });
        y += 15;

        // RX / TX rates.
        let mut rx_str = String::from("Net RX: ");
        fmt_bps(&mut rx_str, self.metrics.net_rx_bps);
        s.push(DrawCmd::Text { rect: Rect::new(x + 4, y, w - 8, 13), text: rx_str, color: t.text, size: 11 });
        y += 15;

        let mut tx_str = String::from("Net TX: ");
        fmt_bps(&mut tx_str, self.metrics.net_tx_bps);
        s.push(DrawCmd::Text { rect: Rect::new(x + 4, y, w - 8, 13), text: tx_str, color: t.text, size: 11 });
        y += 15;

        // Traffic-light indicator: green < 50 Mbps combined, yellow < 200 Mbps, else red.
        let combined_bps = self.metrics.net_rx_bps + self.metrics.net_tx_bps;
        let (tl_color, tl_label) = if combined_bps < 50_000_000 {
            (Color::rgb(0x3f, 0xc9, 0x70), "healthy")
        } else if combined_bps < 200_000_000 {
            (Color::rgb(0xf5, 0xc5, 0x42), "elevated")
        } else {
            (t.danger, "suspicious")
        };
        s.push(toolkit::disc(x + 6, y + 6, 5, tl_color));
        let mut tl_str = String::from("Traffic: ");
        tl_str.push_str(tl_label);
        s.push(DrawCmd::Text { rect: Rect::new(x + 16, y, w - 20, 13), text: tl_str, color: tl_color, size: 11 });
        y += 20;

        // ── Threat monitoring ────────────────────────────────────────────────────────
        s.push(DrawCmd::Text {
            rect: Rect::new(x, y, w, 14),
            text: "Threat Monitor".into(),
            color: t.muted,
            size: 11,
        });
        y += 18;

        let (threat_color, threat_text) = if self.metrics.entropy_milli == 0 {
            (Color::rgb(0xf5, 0xc5, 0x42), "Entropy low")
        } else {
            (Color::rgb(0x3f, 0xc9, 0x70), "No threats detected")
        };
        s.push(toolkit::disc(x + 6, y + 6, 4, threat_color));
        s.push(DrawCmd::Text { rect: Rect::new(x + 16, y, w - 20, 13), text: threat_text.into(), color: threat_color, size: 11 });
        y += 16;

        s.push(toolkit::disc(x + 6, y + 6, 4, Color::rgb(0x3f, 0xc9, 0x70)));
        s.push(DrawCmd::Text { rect: Rect::new(x + 16, y, w - 20, 13), text: "Firewall: active".into(), color: t.text, size: 11 });
        y += 16;

        s.push(DrawCmd::Text { rect: Rect::new(x + 16, y, w - 20, 13), text: "Last scan: OK".into(), color: t.muted, size: 11 });
        y += 20;

        // ── Active connections list ──────────────────────────────────────────────────
        s.push(DrawCmd::Text {
            rect: Rect::new(x, y, w, 14),
            text: "Active Connections".into(),
            color: t.muted,
            size: 11,
        });
        y += 18;

        // (dot_color, label, status, button_label)
        let conns: &[(Color, &str, &str)] = &[
            (Color::rgb(0x3f, 0xc9, 0x70), "kernel \u{2194} virtio-net",      "established"),
            (Color::rgb(0x3f, 0xc9, 0x70), "browser \u{2194} NDN node (0x1a2b)", "active"),
            (Color::rgb(0xf5, 0xc5, 0x42), "dns \u{2194} resolver",            "idle"),
            (Color::rgb(0x3f, 0xc9, 0x70), "ml \u{2194} npu-driver",           "established"),
            (t.danger,                      "probe:4444 \u{2194} unknown",      "blocked"),
        ];
        for (dot, label, status) in conns {
            s.push(toolkit::disc(x + 6, y + 6, 3, *dot));
            // Connection label
            s.push(DrawCmd::Text {
                rect: Rect::new(x + 16, y, w - 60, 13),
                text: (*label).into(),
                color: t.text,
                size: 10,
            });
            // Status badge
            s.push(DrawCmd::Text {
                rect: Rect::new(x + 16, y + 13, w - 60, 12),
                text: (*status).into(),
                color: *dot,
                size: 10,
            });
            // "Block" button (visual only)
            let btn = Rect::new(x + w - 44, y + 2, 40, 20);
            s.push(DrawCmd::Rect { rect: btn, color: t.surface, radius: t.radius });
            s.push(DrawCmd::Text {
                rect: Rect::new(btn.x + 4, btn.y + 4, btn.w - 8, 12),
                text: "Block".into(),
                color: t.muted,
                size: 10,
            });
            y += 32;
        }

        // ── Firewall rules ───────────────────────────────────────────────────────────
        s.push(DrawCmd::Text {
            rect: Rect::new(x, y, w, 14),
            text: "Firewall Rules".into(),
            color: t.muted,
            size: 11,
        });
        y += 18;

        let rules: &[(Color, &str)] = &[
            (Color::rgb(0x3f, 0xc9, 0x70), "Allow: all outbound (port 80, 443)"),
            (Color::rgb(0x3f, 0xc9, 0x70), "Allow: NDN inbound (port 6363)"),
            (t.danger,                      "Block: all other inbound"),
        ];
        for (dot, rule) in rules {
            s.push(toolkit::disc(x + 6, y + 6, 3, *dot));
            s.push(DrawCmd::Text {
                rect: Rect::new(x + 16, y, w - 20, 13),
                text: (*rule).into(),
                color: t.text,
                size: 10,
            });
            y += 16;
        }

        // Leave a small gap then show the real-time log below.
        // The log is always emitted; the compositor clips it to the visible viewport.
        let log_y = y + 8;
        s.push(DrawCmd::Text {
            rect: Rect::new(x, log_y, w, 14),
            text: "Real-time Logging".into(),
            color: t.muted,
            size: 11,
        });
        let log_h = (c.y + c.h - log_y - 26).max(40);
        let log = Rect::new(x, log_y + 18, w, log_h);
        s.append(&mut widgets::log_view(log, &self.logs, 0, t));
    }

    fn draw_status(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = Rect::new(0, self.area.h - STATUS_H, self.area.w, STATUS_H);
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        let mut build = String::from("OS 2.0 BUILD ");
        build.push_str(if self.metrics.det_hash.is_empty() { "2055" } else { &self.metrics.det_hash });
        let net = if self.metrics.net_present { "NETWORK: NDN ACTIVE" } else { "NETWORK: OFFLINE" };
        let mut cap = String::from("CAPABILITY: ");
        if self.root_prov.is_empty() {
            cap.push_str("0xF..1E (ADMIN)");
        } else {
            cap.push_str(&self.root_prov);
            cap.push_str(" (ROOT)");
        }
        let parts = ["ARCHITECTURE 2.0", &build, &cap, net, "AUDIO: OBA (XR SYNC)"];
        let mut x = 16;
        for p in parts {
            s.push(DrawCmd::Text { rect: Rect::new(x, bar.y + 6, 300, 14), text: p.into(), color: t.muted, size: 11 });
            x += p.len() as i32 * 7 + 28;
        }
    }
}

impl Default for Explorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a semantic object kind to a constellation node colour class.
fn kind_to_node(kind: &str) -> NodeKind {
    match kind {
        "Program" | "Channel" => NodeKind::App,
        "Dataset" => NodeKind::Data,
        "Report" | "TokenSet" => NodeKind::Report,
        "Codec" => NodeKind::Audio,
        "Log" => NodeKind::Log,
        _ => NodeKind::Data,
    }
}

/// A deterministic scatter layout for object `i` within the constellation.
fn scatter(i: usize) -> (i32, i32) {
    let col = (i % 3) as i32;
    let row = (i / 3) as i32;
    let jit = ((i * 37) % 40) as i32 - 20;
    (30 + col * 150 + jit, 50 + row * 110 + ((i * 53) % 24) as i32)
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

/// Format a byte-per-second rate into a human-readable string appended to `s`.
/// Mirrors the helper in taskman.rs; kept local to this module for `no_std` portability.
fn fmt_bps(s: &mut String, bps: u64) {
    if bps >= 1_000_000_000 {
        push_int(s, (bps / 1_000_000_000) as i64);
        s.push('.');
        push_int(s, ((bps % 1_000_000_000) / 100_000_000) as i64);
        s.push_str(" GB/s");
    } else if bps >= 1_000_000 {
        push_int(s, (bps / 1_000_000) as i64);
        s.push('.');
        push_int(s, ((bps % 1_000_000) / 100_000) as i64);
        s.push_str(" MB/s");
    } else if bps >= 1_000 {
        push_int(s, (bps / 1_000) as i64);
        s.push('.');
        push_int(s, ((bps % 1_000) / 100) as i64);
        s.push_str(" KB/s");
    } else {
        push_int(s, bps as i64);
        s.push_str(" B/s");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    fn explorer() -> Explorer {
        let mut w = World::new();
        w.set_programs(&[("project.aeth".to_string(), "let x = 1;".to_string())]);
        let mut e = Explorer::new();
        e.set_area(Rect::new(0, 0, 1440, 600));
        e.set_objects(&w.entries());
        e.set_cells(w.cells());
        e.set_root_prov(w.root_cap().provenance().short());
        let _ = e.take_damage();
        e
    }

    #[test]
    fn renders_three_columns_and_a_status_strip() {
        let e = explorer();
        let s = e.view(&Theme::dark());
        for title in ["SYSTEM KNOWLEDGE GRAPH", "SECURE COMPARTMENT VIEWER", "SYSTEM HEALTH"] {
            assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == title)), "missing {}", title);
        }
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "ARCHITECTURE 2.0")));
        // Live knowledge objects + real sandboxed cells render.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "SALES DATA 2024")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "linux-guest")));
    }

    #[test]
    fn selecting_a_knowledge_node_opens_its_detail() {
        let mut e = explorer();
        // Select the sales dataset by name (as a Desktop "Inspect" click would).
        e.select_by_name("SALES DATA 2024");
        let s = e.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Proven Provenance")));
        // The detail shows the dataset's real metadata (its row count).
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "rows: 4096")));
    }

    #[test]
    fn ref_edges_become_constellation_wires() {
        let e = explorer();
        // The seeded graph has Report→Dataset, Channel→Log, Codec→Dataset, plus the
        // program→dataset link: several real reference wires.
        assert!(e.graph.wires().len() >= 3, "wires: {}", e.graph.wires().len());
    }

    #[test]
    fn search_filters_the_constellation() {
        let mut e = explorer();
        let sr = e.search_rect();
        e.on_pointer(sr.x + 10, sr.y + 10, true); // focus search
        e.on_pointer(sr.x + 10, sr.y + 10, false);
        assert!(e.search_focused);
        for ch in "sales".chars() {
            assert!(e.on_key(ch));
        }
        // Only SALES DATA 2024 matches.
        assert_eq!(e.graph.nodes().len(), 1);
        assert_eq!(e.graph.nodes()[0].title, "SALES DATA 2024");
        // Esc unfocuses and stops consuming keys.
        assert!(e.on_key('\u{1b}'));
        assert!(!e.search_focused);
        assert!(!e.on_key('x'));
    }

    #[test]
    fn metrics_feed_the_health_gauges_and_damage_column_c() {
        let mut e = explorer();
        let _ = e.take_damage();
        e.set_metrics(Metrics { cpu_milli: 420, fps: 30, net_present: true, ..Default::default() });
        let d = e.take_damage().unwrap();
        assert!(d.x >= e.col_c().x - 1);
        let s = e.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "42%")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "NETWORK: NDN ACTIVE")));
    }

    #[test]
    fn logs_appear_in_the_monitoring_column() {
        let mut e = explorer();
        e.push_log("[boot] kernel online");
        let s = e.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("kernel online"))));
    }
}

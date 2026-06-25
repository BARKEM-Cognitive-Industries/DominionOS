//! The **Task Manager** app — a Windows-Task-Manager-style live monitor over the
//! shell's shared [`Scheduler`](crate::sched).
//!
//! It lists every running **domain** (a Software-Isolated Process — DominionOS's unit of
//! isolation) with its PID, name, state, a CPU-share proxy (its share of all dispatch
//! steps), and its memory footprint (the size of its capability region). A selected
//! row can be ended with **End task** ([`Scheduler::kill`]). A header strip shows the
//! live system totals fed from the kernel's [`Metrics`](crate::dash::Metrics). It reads
//! the *same* scheduler the Terminal's `ps` does, so the two always agree. Pure, safe
//! `no_std`, rendered in page-local coordinates.

use crate::dash::Metrics;
use crate::sched::{DomainId, DomainState};
use crate::shellcmd::SharedSched;
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::widgets::{self, Scroll};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

const HEADER_H: i32 = 64;
const COLS_H: i32 = 26;
const ROW_H: i32 = 28;
const FOOTER_H: i32 = 44;

/// The Task Manager page.
pub struct TaskManager {
    sched: SharedSched,
    metrics: Metrics,
    area: Rect,
    selected: Option<DomainId>,
    scroll: Scroll,
    /// Dispatch-step count per domain at the previous metric sample, so per-process CPU
    /// can be derived from the *change* since then (recent activity, not all-time share).
    prev_steps: BTreeMap<DomainId, u32>,
    /// The per-process CPU% computed at the last sample.
    cpu_pct: BTreeMap<DomainId, u32>,
    last_left: bool,
    damage: Option<Rect>,
}

impl TaskManager {
    pub fn new(sched: SharedSched) -> TaskManager {
        TaskManager {
            sched,
            metrics: Metrics::default(),
            area: Rect::new(0, 0, 1280, 600),
            selected: None,
            scroll: Scroll::new(),
            prev_steps: BTreeMap::new(),
            cpu_pct: BTreeMap::new(),
            last_left: false,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        }
    }

    /// The scrollable process-list viewport.
    fn list_area(&self) -> Rect {
        Rect::new(8, self.list_top(), self.area.w - 16, (self.area.h - self.list_top() - FOOTER_H).max(0))
    }
    /// Pixel height of the full process list.
    fn content_h(&self) -> i32 {
        self.sched.borrow().snapshot().len() as i32 * ROW_H
    }

    pub fn set_area(&mut self, area: Rect) {
        if area != self.area {
            self.area = area;
            let l = self.list_area();
            self.scroll.clamp(self.content_h(), l.h);
            self.dmg_all();
        }
    }

    /// Feed the live system totals (shown in the header strip + CPU sparkline) and
    /// recompute each process's CPU% from the change in its dispatch steps since the last
    /// sample, scaled by the live system CPU. A process the user isn't driving shows ~0%;
    /// the focused app (which the shell charges extra) shows the bulk of the load —
    /// "actual usage", not an even round-robin share.
    pub fn set_metrics(&mut self, m: Metrics) {
        let snap = self.sched.borrow().snapshot();
        let mut total_delta = 0u32;
        let mut deltas: Vec<(DomainId, u32)> = Vec::with_capacity(snap.len());
        for d in &snap {
            let prev = self.prev_steps.get(&d.id).copied().unwrap_or(d.steps);
            let delta = d.steps.saturating_sub(prev);
            total_delta += delta;
            deltas.push((d.id, delta));
        }
        let sys_pct = (m.cpu_milli / 10).min(100);
        self.cpu_pct.clear();
        for (id, delta) in deltas {
            let pct = (delta * sys_pct).checked_div(total_delta).unwrap_or(0);
            self.cpu_pct.insert(id, pct);
        }
        for d in &snap {
            self.prev_steps.insert(d.id, d.steps);
        }
        self.metrics = m;
        self.dmg_all();
    }

    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    pub fn wants_text(&self) -> bool {
        false
    }
    pub fn on_key(&mut self, _ch: char) -> bool {
        false
    }
    pub fn set_time(&mut self, _now_ms: u64) {}

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

    fn list_top(&self) -> i32 {
        HEADER_H + COLS_H
    }
    fn row_rect(&self, i: usize) -> Rect {
        Rect::new(8, self.list_top() + i as i32 * ROW_H - self.scroll.offset, self.area.w - 16 - widgets::SCROLLBAR_W, ROW_H - 2)
    }
    fn endtask_btn(&self) -> Rect {
        Rect::new(self.area.w - 132, self.area.h - FOOTER_H + 8, 120, FOOTER_H - 16)
    }
    /// Column x-offsets: PID, Name, Status, CPU, Memory.
    fn col_x(&self) -> [i32; 5] {
        let w = self.area.w;
        [16, 80, w / 2, w * 3 / 4, w * 7 / 8]
    }

    // ── input ──

    /// Returns `Some(id)` when the user clicked "End task" and a domain was killed,
    /// so the caller can also close the associated window.
    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) -> Option<DomainId> {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        let l = self.list_area();
        let content_h = self.content_h();

        // Scrollbar drag in progress.
        if self.scroll.is_dragging() {
            if left {
                self.scroll.on_drag(py, l, content_h, l.h);
                self.dmg(l);
            } else {
                self.scroll.release();
            }
            self.last_left = left;
            return None;
        }
        self.last_left = left;
        if released || !pressed {
            return None;
        }
        // Scrollbar press.
        if self.scroll.on_press(px, py, l, content_h, l.h) {
            self.dmg(l);
            return None;
        }
        // End task (acts on the selection): kill the scheduler domain and return the id
        // so the caller can close the associated window immediately.
        if self.endtask_btn().contains(px, py) {
            if let Some(id) = self.selected {
                self.sched.borrow_mut().kill(id);
                self.selected = None;
                self.dmg_all();
                return Some(id);
            }
            return None;
        }
        // Row selection.
        let procs = self.sched.borrow().snapshot();
        for (i, d) in procs.iter().enumerate() {
            let r = self.row_rect(i);
            if r.y + r.h <= l.y || r.y >= l.y + l.h {
                continue; // off-screen
            }
            if r.contains(px, py) {
                self.selected = Some(d.id);
                self.dmg_all();
                return None;
            }
        }
        None
    }

    // ── rendering ──

    pub fn view(&self, t: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: t.bg, radius: 0 });
        self.draw_header(&mut s, t);
        self.draw_columns(&mut s, t);
        self.draw_rows(&mut s, t);
        self.draw_footer(&mut s, t);
        s
    }

    fn draw_header(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = Rect::new(0, 0, self.area.w, HEADER_H);
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        s.push(DrawCmd::Text { rect: Rect::new(16, 10, 300, 20), text: "Task Manager".into(), color: t.text, size: 17 });
        let procs = self.sched.borrow().snapshot();
        let live = procs.iter().filter(|d| d.state != DomainState::Finished).count();
        let mut proc_sum = String::from("Processes: ");
        push_int(&mut proc_sum, live as i64);
        s.push(DrawCmd::Text { rect: Rect::new(16, 36, 160, 16), text: proc_sum, color: t.muted, size: 12 });
        // CPU label emitted as separate node so tests can match "CPU 35%".
        let mut cpu_txt = String::from("CPU ");
        push_int(&mut cpu_txt, (self.metrics.cpu_milli / 10) as i64);
        cpu_txt.push('%');
        s.push(DrawCmd::Text { rect: Rect::new(180, 36, 120, 16), text: cpu_txt, color: t.muted, size: 12 });
        // Memory label emitted as separate node so tests can match "Mem 62%".
        let mut mem_txt = String::from("Mem ");
        if self.metrics.mem_total_kb > 0 {
            push_int(&mut mem_txt, (self.metrics.mem_used_kb / 1024) as i64);
            mem_txt.push_str("/");
            push_int(&mut mem_txt, (self.metrics.mem_total_kb / 1024) as i64);
            mem_txt.push_str("MiB ");
        }
        push_int(&mut mem_txt, (self.metrics.mem_milli / 10) as i64);
        mem_txt.push('%');
        s.push(DrawCmd::Text { rect: Rect::new(310, 36, 160, 16), text: mem_txt, color: t.muted, size: 12 });
        if self.metrics.disk_present {
            let disk_txt = if self.metrics.disk_read_bps == 0 && self.metrics.disk_write_bps == 0 {
                String::from("Disk: idle")
            } else {
                let mut t2 = String::from("Disk \u{2191}");
                fmt_bps(&mut t2, self.metrics.disk_write_bps);
                t2.push_str(" \u{2193}");
                fmt_bps(&mut t2, self.metrics.disk_read_bps);
                t2
            };
            s.push(DrawCmd::Text { rect: Rect::new(480, 36, 160, 16), text: disk_txt, color: t.muted, size: 12 });
        }
        if self.metrics.net_present {
            let net_txt = if self.metrics.net_rx_bps == 0 && self.metrics.net_tx_bps == 0 {
                String::from("Net: idle")
            } else {
                let mut t2 = String::from("Net \u{2191}");
                fmt_bps(&mut t2, self.metrics.net_tx_bps);
                t2.push_str(" \u{2193}");
                fmt_bps(&mut t2, self.metrics.net_rx_bps);
                t2
            };
            s.push(DrawCmd::Text { rect: Rect::new(650, 36, 160, 16), text: net_txt, color: t.muted, size: 12 });
        }
        // A small CPU sparkline on the right of the header.
        if !self.metrics.cpu_history.is_empty() {
            let spark = Rect::new(self.area.w - 220, 12, 200, 40);
            s.append(&mut widgets::sparkline(spark, &self.metrics.cpu_history, t.primary));
        }
    }

    fn draw_columns(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let y = HEADER_H;
        s.push(DrawCmd::Rect { rect: Rect::new(0, y, self.area.w, COLS_H), color: t.bg, radius: 0 });
        let cx = self.col_x();
        for (x, label) in cx.iter().zip(["PID", "NAME", "STATUS", "CPU", "MEMORY"]) {
            s.push(DrawCmd::Text { rect: Rect::new(*x, y + 6, 120, 14), text: label.into(), color: t.muted, size: 11 });
        }
        s.push(DrawCmd::Rect { rect: Rect::new(8, y + COLS_H - 1, self.area.w - 16, 1), color: t.muted, radius: 0 });
    }

    fn draw_rows(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let procs = self.sched.borrow().snapshot();
        let cx = self.col_x();
        let l = self.list_area();
        for (i, d) in procs.iter().enumerate() {
            let r = self.row_rect(i);
            if r.y + r.h <= l.y || r.y >= l.y + l.h {
                continue; // cull rows outside the viewport
            }
            if self.selected == Some(d.id) {
                s.push(DrawCmd::Rect { rect: r, color: Color::rgba(t.primary.r, t.primary.g, t.primary.b, 55), radius: t.radius });
            }
            let fg = if d.state == DomainState::Finished { t.muted } else { t.text };
            // PID
            let mut pid = String::new();
            push_int(&mut pid, d.id.0 as i64);
            s.push(DrawCmd::Text { rect: Rect::new(cx[0], r.y + 7, 60, 14), text: pid, color: fg, size: 12 });
            // Name (ellipsised so a long domain name doesn't run into the Status column)
            let name_w = cx[2] - cx[1] - 8;
            s.push(DrawCmd::Text { rect: Rect::new(cx[1], r.y + 7, name_w, 14), text: toolkit::ellipsize_px(&d.name, name_w, 12), color: fg, size: 12 });
            // Status (with a colour dot)
            let (label, dot) = match d.state {
                DomainState::Running => ("Running", Color::rgb(0x3f, 0xc9, 0xb0)),
                DomainState::Ready => ("Ready", t.primary),
                DomainState::Finished => ("Ended", t.muted),
            };
            s.push(toolkit::disc(cx[2] + 4, r.y + r.h / 2, 3, dot));
            s.push(DrawCmd::Text { rect: Rect::new(cx[2] + 14, r.y + 7, 100, 14), text: label.into(), color: fg, size: 12 });
            // CPU% — recent activity (set in `set_metrics`), not an all-time step share.
            let pct = self.cpu_pct.get(&d.id).copied().unwrap_or(0);
            let mut cpu = String::new();
            push_int(&mut cpu, pct as i64);
            cpu.push('%');
            s.push(DrawCmd::Text { rect: Rect::new(cx[3], r.y + 7, 60, 14), text: cpu, color: fg, size: 12 });
            // Memory (region length)
            s.push(DrawCmd::Text { rect: Rect::new(cx[4], r.y + 7, 90, 14), text: fmt_bytes(d.len), color: fg, size: 12 });
        }
        // Scrollbar for an overflowing process list.
        s.extend(widgets::scrollbar(l, self.content_h(), l.h, &self.scroll, t));
    }

    fn draw_footer(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = Rect::new(0, self.area.h - FOOTER_H, self.area.w, FOOTER_H);
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        let b = self.endtask_btn();
        let enabled = self.selected.is_some();
        let fill = if enabled { t.danger } else { t.surface };
        let fg = if enabled { t.on_primary } else { t.muted };
        s.push(DrawCmd::Rect { rect: b, color: fill, radius: t.radius });
        if !enabled {
            s.push(DrawCmd::Rect { rect: toolkit::inflate(b, 1), color: t.muted, radius: t.radius });
            s.push(DrawCmd::Rect { rect: b, color: t.surface, radius: t.radius });
        }
        s.push(DrawCmd::Text { rect: Rect::new(b.x + 18, b.y + 7, b.w, 16), text: "End task".into(), color: fg, size: 13 });
    }
}

/// Append a human-readable bytes/sec rate to `s` (e.g. "4 KB/s", "1 MB/s", "512 B/s").
fn fmt_bps(s: &mut String, n: u64) {
    if n >= 1024 * 1024 {
        push_int(s, (n / (1024 * 1024)) as i64);
        s.push_str(" MB/s");
    } else if n >= 1024 {
        push_int(s, (n / 1024) as i64);
        s.push_str(" KB/s");
    } else {
        push_int(s, n as i64);
        s.push_str(" B/s");
    }
}

fn fmt_bytes(n: u64) -> String {
    let mut s = String::new();
    if n >= 1024 * 1024 {
        push_int(&mut s, (n / (1024 * 1024)) as i64);
        s.push_str(" MiB");
    } else if n >= 1024 {
        push_int(&mut s, (n / 1024) as i64);
        s.push_str(" KiB");
    } else {
        push_int(&mut s, n as i64);
        s.push_str(" B");
    }
    s
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
    use crate::capability::{Capability, Rights};
    use crate::sched::Scheduler;
    use alloc::rc::Rc;
    use core::cell::RefCell;

    fn taskman() -> (TaskManager, SharedSched) {
        let sched = Rc::new(RefCell::new(Scheduler::new()));
        sched.borrow_mut().spawn("init", Capability::mint(0x1000, 0x40000, Rights::ALL));
        sched.borrow_mut().spawn("compositor", Capability::mint(0x80000, 0x200000, Rights::ALL));
        sched.borrow_mut().spawn("netstack", Capability::mint(0x400000, 0x100000, Rights::ALL));
        let mut tm = TaskManager::new(sched.clone());
        tm.set_area(Rect::new(0, 0, 1200, 600));
        let _ = tm.take_damage();
        (tm, sched)
    }

    #[test]
    fn lists_running_domains_with_columns() {
        let (tm, _s) = taskman();
        let scene = tm.view(&Theme::dark());
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Task Manager")));
        for col in ["PID", "NAME", "STATUS", "CPU", "MEMORY"] {
            assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == col)));
        }
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "init")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "compositor")));
        // Memory footprint of the compositor (0x200000 = 2 MiB).
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "2 MiB")));
    }

    #[test]
    fn selecting_then_end_task_kills_the_domain() {
        let (mut tm, sched) = taskman();
        // Select the second row (compositor).
        let r = tm.row_rect(1);
        tm.on_pointer(r.x + 20, r.y + 10, true);
        tm.on_pointer(r.x + 20, r.y + 10, false);
        assert!(tm.selected.is_some());
        // End task.
        let b = tm.endtask_btn();
        tm.on_pointer(b.x + 10, b.y + 10, true);
        tm.on_pointer(b.x + 10, b.y + 10, false);
        // The compositor is now Finished in the shared scheduler.
        let snap = sched.borrow().snapshot();
        let comp = snap.iter().find(|d| d.name == "compositor").unwrap();
        assert_eq!(comp.state, DomainState::Finished);
    }

    #[test]
    fn metrics_feed_the_header_summary() {
        let (mut tm, _s) = taskman();
        tm.set_metrics(Metrics { cpu_milli: 350, mem_milli: 620, cpu_history: alloc::vec![10, 20, 30], ..Default::default() });
        let scene = tm.view(&Theme::dark());
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("CPU 35%"))));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Mem 62%"))));
    }

    #[test]
    fn a_long_process_list_scrolls() {
        let sched = Rc::new(RefCell::new(Scheduler::new()));
        for i in 0..40 {
            let mut name = String::from("domain");
            push_int(&mut name, i);
            sched.borrow_mut().spawn(&name, Capability::mint(0x1000 * (i as u64 + 1), 0x1000, Rights::ALL));
        }
        let mut tm = TaskManager::new(sched);
        tm.set_area(Rect::new(0, 0, 1000, 400));
        let _ = tm.take_damage();
        let l = tm.list_area();
        assert!(Scroll::needed(tm.content_h(), l.h));
        let track = Scroll::track(l);
        tm.on_pointer(track.x + 2, track.y + 2, true);
        tm.on_pointer(track.x + 2, track.y + l.h, true);
        tm.on_pointer(track.x + 2, track.y + l.h, false);
        assert_eq!(tm.scroll.offset, Scroll::max_offset(tm.content_h(), l.h));
        assert!(tm.scroll.offset > 0);
    }

    #[test]
    fn end_task_does_nothing_without_a_selection() {
        let (mut tm, sched) = taskman();
        let b = tm.endtask_btn();
        tm.on_pointer(b.x + 10, b.y + 10, true);
        tm.on_pointer(b.x + 10, b.y + 10, false);
        // All domains still alive.
        assert!(sched.borrow().snapshot().iter().all(|d| d.state != DomainState::Finished));
    }
}

//! **Window manager** — floating, overlapping windows with real chrome.
//!
//! The shell ([`crate::os`]) used to be a single-active-app pager: one app filled
//! the screen and switching apps replaced it wholesale. This module gives the shell
//! genuine **windows** instead: each open app is a [`Window`] with a **title bar**
//! (minimize / maximize / close), is **draggable** by that bar, and is
//! **resizable from any edge or corner**. Windows stack in a z-order (the last is
//! focused / on top), can be **minimized** to the taskbar and **maximized** to fill
//! the work area, and are clamped to stay on screen.
//!
//! The manager is **generic over the app id** (`T`) and **pure**: it owns geometry,
//! z-order, drag state, and the chrome *scene* + hit-testing, but knows nothing about
//! how an app renders its content. The shell maps `T = AppId`, draws each window's
//! content into [`Window::content`], and acts on the [`Reaction`] returned from
//! [`WindowManager::on_pointer`]. Safe `no_std`, host-tested.

use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Height of a window's title bar, in pixels.
pub const TITLE_H: i32 = 28;
/// Thickness of the resize grab zone around a window's border.
const GRAB: i32 = 6;
/// Minimum window size (full frame, including the title bar).
const MIN_W: i32 = 240;
const MIN_H: i32 = 140;
/// Width of each title-bar button (close / maximize / minimize).
const BTN: i32 = 30;

/// A window's display state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WinState {
    /// A normal floating window at its own `rect`.
    Normal,
    /// Hidden from the desktop but kept on the taskbar.
    Minimized,
    /// Expanded to fill the whole work area.
    Maximized,
}

/// One window: an app id, a title, geometry, and state.
#[derive(Clone, Debug)]
pub struct Window<T> {
    pub id: T,
    pub title: String,
    /// Geometry while [`WinState::Normal`] (full frame, title bar included).
    pub rect: Rect,
    pub state: WinState,
    /// Saved `Normal` rect, restored when a maximized window is un-maximized.
    restore: Rect,
    /// Stable creation order — the taskbar lists windows by this, so buttons don't
    /// jump around as the z-order changes.
    seq: u32,
}

impl<T: Copy + PartialEq> Window<T> {
    /// The on-screen frame (the whole work area when maximized).
    pub fn frame(&self, area: Rect) -> Rect {
        match self.state {
            WinState::Maximized => area,
            _ => self.rect,
        }
    }
    /// The content rect (below the title bar), in screen coordinates.
    pub fn content(&self, area: Rect) -> Rect {
        let f = self.frame(area);
        Rect::new(f.x, f.y + TITLE_H, f.w, (f.h - TITLE_H).max(0))
    }
    pub fn is_minimized(&self) -> bool {
        self.state == WinState::Minimized
    }
}

/// Which edges a resize drag is pulling.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct Edges {
    l: bool,
    r: bool,
    t: bool,
    b: bool,
}
impl Edges {
    fn any(&self) -> bool {
        self.l || self.r || self.t || self.b
    }
}

/// What the pointer is currently dragging.
#[derive(Clone, Copy)]
enum Drag<T> {
    None,
    /// Moving window `id`, grabbed at offset `(dx,dy)` from its top-left.
    Move { id: T, dx: i32, dy: i32 },
    /// Resizing window `id` by its `edges`.
    Resize { id: T, edges: Edges },
}

/// What a pointer event did, for the shell to act on.
pub enum Reaction<T> {
    /// Consumed by window chrome (a move/resize drag, or a title-bar button) — the
    /// shell should do nothing else with this event.
    Consumed,
    /// Window `id` was closed; the shell should drop its taskbar entry and refocus.
    Closed(T),
    /// Forward the event to app `id` at window-local content coordinates `(lx,ly)`.
    Forward(T, i32, i32),
    /// The pointer missed every window — the shell handles the desktop / dock.
    Miss,
}

/// The floating-window manager.
pub struct WindowManager<T> {
    wins: Vec<Window<T>>,
    /// Work area windows live in (screen coords, between the top bar and the dock).
    area: Rect,
    drag: Drag<T>,
    last_left: bool,
    next_seq: u32,
}

impl<T: Copy + PartialEq> WindowManager<T> {
    pub fn new(area: Rect) -> WindowManager<T> {
        WindowManager { wins: Vec::new(), area, drag: Drag::None, last_left: false, next_seq: 1 }
    }

    // ── work area ──

    /// Resize the work area, keeping every window on screen.
    pub fn set_area(&mut self, area: Rect) {
        self.area = area;
        for w in &mut self.wins {
            w.rect = clamp_frame(w.rect, area);
            w.restore = clamp_frame(w.restore, area);
        }
    }
    pub fn area(&self) -> Rect {
        self.area
    }

    // ── window set ──

    pub fn is_open(&self, id: T) -> bool {
        self.wins.iter().any(|w| w.id == id)
    }

    fn index(&self, id: T) -> Option<usize> {
        self.wins.iter().position(|w| w.id == id)
    }

    /// The window for `id`, if open.
    pub fn window(&self, id: T) -> Option<&Window<T>> {
        self.wins.iter().find(|w| w.id == id)
    }

    /// Its content rect (screen coords), or a zero rect if not open.
    pub fn content_of(&self, id: T) -> Rect {
        self.window(id).map(|w| w.content(self.area)).unwrap_or(Rect::new(0, 0, 0, 0))
    }

    /// **Open** a window for `id`, or focus/restore it if already open. Returns the
    /// content rect so the caller can size the app to it.
    pub fn open(&mut self, id: T, title: &str) -> Rect {
        if let Some(i) = self.index(id) {
            if self.wins[i].state == WinState::Minimized {
                self.wins[i].state = WinState::Normal;
            }
            self.raise(i);
        } else {
            let seq = self.next_seq;
            self.next_seq += 1;
            let rect = self.cascade(self.wins.len() as i32);
            self.wins.push(Window { id, title: title.to_string(), rect, state: WinState::Normal, restore: rect, seq });
        }
        self.content_of(id)
    }

    /// A cascaded default frame for the n-th window.
    fn cascade(&self, n: i32) -> Rect {
        let a = self.area;
        let w = (a.w * 3 / 5).clamp(MIN_W, (a.w - 16).max(MIN_W));
        let h = (a.h * 3 / 5).clamp(MIN_H, (a.h - 16).max(MIN_H));
        let step = 28;
        let x = a.x + 24 + (n % 6) * step;
        let y = a.y + 20 + (n % 6) * step;
        clamp_frame(Rect::new(x, y, w, h), a)
    }

    /// Close the window for `id` (no-op if it isn't open). Returns whether it existed.
    pub fn close(&mut self, id: T) -> bool {
        if let Some(i) = self.index(id) {
            self.wins.remove(i);
            if let Drag::Move { id: d, .. } | Drag::Resize { id: d, .. } = self.drag {
                if d == id {
                    self.drag = Drag::None;
                }
            }
            true
        } else {
            false
        }
    }

    pub fn minimize(&mut self, id: T) {
        if let Some(i) = self.index(id) {
            self.wins[i].state = WinState::Minimized;
        }
    }
    /// Restore a minimized/maximized window to a normal floating window, and raise it.
    pub fn restore(&mut self, id: T) {
        if let Some(i) = self.index(id) {
            self.wins[i].state = WinState::Normal;
            self.raise(i);
        }
    }
    /// Toggle maximize ↔ normal.
    pub fn toggle_maximize(&mut self, id: T) {
        if let Some(i) = self.index(id) {
            self.wins[i].state = match self.wins[i].state {
                WinState::Maximized => WinState::Normal,
                _ => {
                    self.wins[i].restore = self.wins[i].rect;
                    WinState::Maximized
                }
            };
        }
    }
    /// Focus (raise to top) the window for `id`.
    pub fn focus(&mut self, id: T) {
        if let Some(i) = self.index(id) {
            if self.wins[i].state == WinState::Minimized {
                self.wins[i].state = WinState::Normal;
            }
            self.raise(i);
        }
    }

    /// The topmost non-minimized window's id, if any.
    pub fn top(&self) -> Option<T> {
        self.wins.iter().rev().find(|w| !w.is_minimized()).map(|w| w.id)
    }

    /// The topmost visible window whose frame contains `(px,py)` — for context menus
    /// and hover, without mutating focus or z-order.
    pub fn at(&self, px: i32, py: i32) -> Option<T> {
        self.wins.iter().rev().find(|w| !w.is_minimized() && w.frame(self.area).contains(px, py)).map(|w| w.id)
    }

    /// Whether window `id` is currently maximized (for a context menu's label).
    pub fn is_maximized(&self, id: T) -> bool {
        self.window(id).map(|w| w.state == WinState::Maximized).unwrap_or(false)
    }

    /// Windows in **stable taskbar order** (by creation), with their state.
    pub fn taskbar(&self) -> Vec<(T, WinState)> {
        let mut v: Vec<&Window<T>> = self.wins.iter().collect();
        v.sort_by_key(|w| w.seq);
        v.into_iter().map(|w| (w.id, w.state)).collect()
    }

    /// Non-minimized windows in z-order (back to front) — the draw order.
    pub fn visible(&self) -> impl Iterator<Item = &Window<T>> {
        self.wins.iter().filter(|w| !w.is_minimized())
    }

    pub fn is_dragging(&self) -> bool {
        !matches!(self.drag, Drag::None)
    }

    /// Keep the internal press-edge tracker in sync when the shell consumes a pointer
    /// event without routing it through [`Self::on_pointer`] (e.g. a dock click).
    pub fn note_left(&mut self, left: bool) {
        self.last_left = left;
    }

    fn raise(&mut self, i: usize) {
        let w = self.wins.remove(i);
        self.wins.push(w);
    }

    // ── pointer ──

    /// Route a pointer event. See [`Reaction`].
    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) -> Reaction<T> {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        self.last_left = left;

        // An in-flight chrome drag tracks the pointer wherever it goes.
        match self.drag {
            Drag::Move { id, dx, dy } if left => {
                if let Some(i) = self.index(id) {
                    self.wins[i].rect = clamp_frame(Rect::new(px - dx, py - dy, self.wins[i].rect.w, self.wins[i].rect.h), self.area);
                }
                return Reaction::Consumed;
            }
            Drag::Resize { id, edges } if left => {
                if let Some(i) = self.index(id) {
                    self.wins[i].rect = resize(self.wins[i].rect, edges, px, py, self.area);
                }
                return Reaction::Consumed;
            }
            Drag::Move { .. } | Drag::Resize { .. } => {
                // Button released → end the drag and swallow the release.
                self.drag = Drag::None;
                return Reaction::Consumed;
            }
            Drag::None => {}
        }

        if pressed {
            return self.press(px, py);
        }
        // A held drag or a release with no chrome drag → forward to the focused window
        // so the app's own drag/click logic still sees move and release events.
        if (left && !pressed) || released {
            if let Some(id) = self.top() {
                let c = self.content_of(id);
                return Reaction::Forward(id, px - c.x, py - c.y);
            }
            return Reaction::Miss;
        }
        Reaction::Miss
    }

    /// Classify a fresh press against the window stack (topmost first).
    fn press(&mut self, px: i32, py: i32) -> Reaction<T> {
        let order: Vec<usize> = (0..self.wins.len()).rev().collect();
        for i in order {
            let w = &self.wins[i];
            if w.is_minimized() {
                continue;
            }
            let id = w.id;
            let f = w.frame(self.area);
            let title = Rect::new(f.x, f.y, f.w, TITLE_H);

            // 1. Title-bar buttons (checked before resize so the top edge can't steal
            //    a click on a button).
            if title.contains(px, py) {
                if btn_rect(f, 0).contains(px, py) {
                    self.close(id);
                    return Reaction::Closed(id);
                }
                if btn_rect(f, 1).contains(px, py) {
                    self.toggle_maximize(id);
                    self.raise(i);
                    return Reaction::Consumed;
                }
                if btn_rect(f, 2).contains(px, py) {
                    self.minimize(id);
                    return Reaction::Consumed;
                }
            }

            // 2. Resize border (normal windows only).
            if w.state == WinState::Normal {
                let e = edges_at(f, px, py);
                if e.any() {
                    self.raise(i);
                    self.drag = Drag::Resize { id, edges: e };
                    return Reaction::Consumed;
                }
            }

            // 3. Title bar → move.
            if title.contains(px, py) {
                self.raise(i);
                let f2 = self.window(id).unwrap().frame(self.area);
                self.drag = Drag::Move { id, dx: px - f2.x, dy: py - f2.y };
                return Reaction::Consumed;
            }

            // 4. Content → focus + forward to the app.
            if f.contains(px, py) {
                self.raise(i);
                let c = self.content_of(id);
                return Reaction::Forward(id, px - c.x, py - c.y);
            }
        }
        Reaction::Miss
    }

    // ── rendering ──

    /// The chrome scene for one window: drop shadow, border, title bar, buttons. The
    /// caller draws the app's content into [`Window::content`] *after* this, so the
    /// title bar sits above the content area (they don't overlap).
    pub fn frame_scene(&self, w: &Window<T>, t: &Theme, focused: bool) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        let f = w.frame(self.area);
        // Soft drop shadow.
        s.push(DrawCmd::Rect { rect: Rect::new(f.x + 6, f.y + 8, f.w, f.h), color: Color::rgba(0, 0, 0, 80), radius: t.radius });
        // Focus border ring.
        let ring = if focused { t.primary } else { t.muted };
        s.push(DrawCmd::Rect { rect: toolkit::inflate(f, 1), color: ring, radius: t.radius });
        // Window body (the app paints over the content area; this backs the gaps).
        s.push(DrawCmd::Rect { rect: f, color: t.bg, radius: t.radius });
        // Title bar.
        let title = Rect::new(f.x, f.y, f.w, TITLE_H);
        let bar = if focused { t.surface } else { Color::rgba(t.surface.r, t.surface.g, t.surface.b, 200) };
        s.push(DrawCmd::Rect { rect: title, color: bar, radius: t.radius });
        let fg = if focused { t.text } else { t.muted };
        s.push(DrawCmd::Text {
            rect: Rect::new(title.x + 12, title.y + 6, (title.w - 3 * BTN - 20).max(0), 16),
            text: w.title.clone(),
            color: fg,
            size: 13,
        });
        // Buttons: minimize, maximize, close (left→right visually).
        let close = btn_rect(f, 0);
        let maxi = btn_rect(f, 1);
        let mini = btn_rect(f, 2);
        // Minimize: a baseline bar.
        s.push(toolkit::line(mini.x + 9, mini.y + TITLE_H - 9, mini.x + BTN - 9, mini.y + TITLE_H - 9, fg, 2));
        // Maximize / restore: a square (double square when already maximized).
        if w.state == WinState::Maximized {
            s.push(rect_outline(Rect::new(maxi.x + 9, maxi.y + 8, 9, 9), fg));
            s.push(rect_outline(Rect::new(maxi.x + 12, maxi.y + 11, 9, 9), t.surface));
            s.push(rect_outline(Rect::new(maxi.x + 12, maxi.y + 11, 9, 9), fg));
        } else {
            s.push(rect_outline(Rect::new(maxi.x + 9, maxi.y + 8, 11, 11), fg));
        }
        // Close: an ×, on a danger chip.
        s.push(DrawCmd::Rect { rect: Rect::new(close.x + 4, close.y + 4, BTN - 8, TITLE_H - 8), color: Color::rgba(t.danger.r, t.danger.g, t.danger.b, 60), radius: 4 });
        s.push(toolkit::line(close.x + 10, close.y + 9, close.x + BTN - 10, close.y + TITLE_H - 9, fg, 2));
        s.push(toolkit::line(close.x + BTN - 10, close.y + 9, close.x + 10, close.y + TITLE_H - 9, fg, 2));
        s
    }
}

// ── geometry helpers ──

/// Keep a frame fully within `area`, clamped to the minimum size.
fn clamp_frame(r: Rect, area: Rect) -> Rect {
    let w = r.w.clamp(MIN_W, area.w.max(MIN_W));
    let h = r.h.clamp(MIN_H, area.h.max(MIN_H));
    let x = r.x.clamp(area.x, (area.x + area.w - w).max(area.x));
    let y = r.y.clamp(area.y, (area.y + area.h - h).max(area.y));
    Rect::new(x, y, w, h)
}

/// Apply a resize drag: move the grabbed edges to the pointer, respecting the minimum
/// size and the work area.
fn resize(r: Rect, e: Edges, px: i32, py: i32, area: Rect) -> Rect {
    let mut x0 = r.x;
    let mut y0 = r.y;
    let mut x1 = r.x + r.w;
    let mut y1 = r.y + r.h;
    if e.l {
        x0 = px.clamp(area.x, x1 - MIN_W);
    }
    if e.r {
        x1 = px.clamp(x0 + MIN_W, area.x + area.w);
    }
    if e.t {
        y0 = py.clamp(area.y, y1 - MIN_H);
    }
    if e.b {
        y1 = py.clamp(y0 + MIN_H, area.y + area.h);
    }
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// Which edges (if any) the pointer is grabbing on frame `f`.
fn edges_at(f: Rect, px: i32, py: i32) -> Edges {
    if !toolkit::inflate(f, GRAB).contains(px, py) {
        return Edges::default();
    }
    // Only count as an edge grab when near the border, not deep in the content.
    Edges {
        l: (px - f.x).abs() <= GRAB,
        r: (px - (f.x + f.w)).abs() <= GRAB,
        t: (py - f.y).abs() <= GRAB,
        b: (py - (f.y + f.h)).abs() <= GRAB,
    }
}

/// Title-bar button rect: `i` counts from the right (0=close, 1=maximize, 2=minimize).
fn btn_rect(f: Rect, i: i32) -> Rect {
    Rect::new(f.x + f.w - (i + 1) * BTN, f.y, BTN, TITLE_H)
}

/// A 1px rectangle outline drawn as four lines.
fn rect_outline(r: Rect, c: Color) -> DrawCmd {
    // A single thin rect reads as an outline at this size; cheap and crisp.
    DrawCmd::Rect { rect: r, color: c, radius: 2 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wm() -> WindowManager<u32> {
        WindowManager::new(Rect::new(0, 30, 1280, 630))
    }

    /// A full click: press then release at the same point.
    fn click(m: &mut WindowManager<u32>, x: i32, y: i32) {
        m.on_pointer(x, y, true);
        m.on_pointer(x, y, false);
    }

    #[test]
    fn open_focus_and_taskbar_order_is_stable() {
        let mut m = wm();
        m.open(1, "One");
        m.open(2, "Two");
        m.open(3, "Three");
        assert_eq!(m.top(), Some(3));
        // Focusing 1 raises it but the taskbar keeps creation order.
        m.focus(1);
        assert_eq!(m.top(), Some(1));
        assert_eq!(m.taskbar().iter().map(|(id, _)| *id).collect::<Vec<_>>(), alloc::vec![1, 2, 3]);
    }

    #[test]
    fn opening_an_existing_window_focuses_it() {
        let mut m = wm();
        m.open(1, "One");
        m.open(2, "Two");
        assert_eq!(m.top(), Some(2));
        m.open(1, "One"); // re-open → focus, not duplicate
        assert_eq!(m.top(), Some(1));
        assert_eq!(m.taskbar().len(), 2);
    }

    #[test]
    fn press_on_content_forwards_local_coords_and_raises() {
        let mut m = wm();
        m.open(1, "One");
        let win = m.window(1).unwrap().clone();
        let c = win.content(m.area());
        match m.on_pointer(c.x + 10, c.y + 20, true) {
            Reaction::Forward(id, lx, ly) => {
                assert_eq!(id, 1);
                assert_eq!((lx, ly), (10, 20));
            }
            _ => panic!("expected forward"),
        }
    }

    #[test]
    fn dragging_the_title_bar_moves_the_window() {
        let mut m = wm();
        m.open(1, "One");
        let f0 = m.window(1).unwrap().frame(m.area());
        // Press the title bar (avoid the right-side buttons), drag, release.
        m.on_pointer(f0.x + 40, f0.y + 8, true);
        m.on_pointer(f0.x + 140, f0.y + 108, true);
        m.on_pointer(f0.x + 140, f0.y + 108, false);
        let f1 = m.window(1).unwrap().frame(m.area());
        assert_eq!((f1.x - f0.x, f1.y - f0.y), (100, 100));
    }

    #[test]
    fn resizing_from_the_bottom_right_grows_the_window() {
        let mut m = wm();
        m.open(1, "One");
        let f0 = m.window(1).unwrap().frame(m.area());
        let (cx, cy) = (f0.x + f0.w, f0.y + f0.h); // bottom-right corner
        m.on_pointer(cx, cy, true);
        m.on_pointer(cx + 80, cy + 60, true);
        m.on_pointer(cx + 80, cy + 60, false);
        let f1 = m.window(1).unwrap().frame(m.area());
        assert_eq!(f1.w, f0.w + 80);
        assert_eq!(f1.h, f0.h + 60);
    }

    #[test]
    fn close_button_closes_and_reports() {
        let mut m = wm();
        m.open(1, "One");
        let f = m.window(1).unwrap().frame(m.area());
        let close = btn_rect(f, 0);
        match m.on_pointer(close.x + BTN / 2, close.y + TITLE_H / 2, true) {
            Reaction::Closed(id) => assert_eq!(id, 1),
            _ => panic!("expected closed"),
        }
        assert!(!m.is_open(1));
    }

    #[test]
    fn minimize_hides_from_visible_but_keeps_taskbar() {
        let mut m = wm();
        m.open(1, "One");
        let f = m.window(1).unwrap().frame(m.area());
        let mini = btn_rect(f, 2);
        m.on_pointer(mini.x + BTN / 2, mini.y + TITLE_H / 2, true);
        assert_eq!(m.window(1).unwrap().state, WinState::Minimized);
        assert_eq!(m.visible().count(), 0);
        assert_eq!(m.taskbar().len(), 1);
        assert_eq!(m.top(), None);
    }

    #[test]
    fn maximize_fills_the_work_area_then_restores() {
        let mut m = wm();
        m.open(1, "One");
        let f = m.window(1).unwrap().frame(m.area());
        let maxi = btn_rect(f, 1);
        click(&mut m, maxi.x + BTN / 2, maxi.y + TITLE_H / 2);
        assert_eq!(m.window(1).unwrap().frame(m.area()), m.area());
        // Toggle back.
        let maxi = btn_rect(m.window(1).unwrap().frame(m.area()), 1);
        click(&mut m, maxi.x + BTN / 2, maxi.y + TITLE_H / 2);
        assert_eq!(m.window(1).unwrap().state, WinState::Normal);
    }

    #[test]
    fn windows_clamp_inside_the_work_area() {
        let mut m = wm();
        m.open(1, "One");
        let f0 = m.window(1).unwrap().frame(m.area());
        // Drag the title way off the bottom-right; it clamps on screen.
        m.on_pointer(f0.x + 40, f0.y + 8, true);
        m.on_pointer(9000, 9000, true);
        m.on_pointer(9000, 9000, false);
        let f1 = m.window(1).unwrap().frame(m.area());
        assert!(f1.x + f1.w <= m.area().x + m.area().w);
        assert!(f1.y + f1.h <= m.area().y + m.area().h);
    }

    #[test]
    fn press_below_all_windows_misses() {
        let mut m = wm();
        m.open(1, "One"); // cascaded near top-left
        // Far bottom-right of the work area, outside the cascaded window.
        let a = m.area();
        match m.on_pointer(a.x + a.w - 5, a.y + a.h - 5, true) {
            Reaction::Miss => {}
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn topmost_window_takes_a_press_in_an_overlap() {
        let mut m = wm();
        m.open(1, "One");
        m.open(2, "Two");
        // Both cascade overlapping near the top-left; a press well inside the top
        // window's content (clear of the resize border) hits the top (2).
        let f2 = m.window(2).unwrap().content(m.area());
        match m.on_pointer(f2.x + 40, f2.y + 40, true) {
            Reaction::Forward(id, ..) => assert_eq!(id, 2),
            _ => panic!("expected forward to 2"),
        }
    }
}

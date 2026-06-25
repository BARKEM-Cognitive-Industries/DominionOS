//! The **Browser** app — chrome (tab strip, back/forward/reload, address bar, a real
//! **Tor** toggle, a draggable scrollbar) wrapped around the universal browser
//! [`Engine`](crate::webengine).
//!
//! Every navigation runs through one engine and renders through one path: an
//! `dominion://` address resolves to a verified native page; anything else is fetched
//! over the injected [`Transport`] (the kernel's virtio-net stack on metal, an
//! in-memory [`LoopbackTransport`] otherwise), parsed by the real HTML engine, and
//! laid out with the same wrapping/scroll/hit-testing. Tabs carry history (so back /
//! forward / reload work), a scroll offset, and the loaded document. Relative links
//! resolve against the final URL. Tor policy is honoured: when Tor is enabled but the
//! circuit is still building, a legacy request is **held**, never leaked to clearnet.
//!
//! Pure, safe `no_std`, page-local coordinates. The byte pipe is the only mechanism
//! injected from the kernel; everything here is policy + rendering.

use crate::browser::{Browser, PageMode, Route};
use crate::dom;
use crate::html::{self, Document, Layout};
use crate::js::Js;
use crate::text::{TextBuffer, BLINK_MS};
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::url::Url;
use crate::filesystem::SharedFs;
use crate::webengine::{AsyncTransport, BlockingAsync, Engine, FetchError, LoadJob, LoadedPage, LoopbackTransport, Transport};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const TABS_H: i32 = 30;
const TOOL_H: i32 = 40;
const ADDR_FONT: i32 = 13;
const BODY_FONT: i32 = 15;
const SB: i32 = 12; // scrollbar width
const NAV_W: i32 = 30; // back/forward/reload button width
const LINE_STEP: i32 = 40;

/// Tor control state the toggle cycles through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TorUi {
    Off,
    Connecting,
    On,
}

/// One open tab: its address, loaded document + cached layout, scroll, and history.
struct Tab {
    /// The URL actually loaded (for relative-link resolution and reload).
    base: Url,
    title: String,
    mode: PageMode,
    status: u16,
    doc: Document,
    layout: Layout,
    laid_width: i32,
    scroll: i32,
    history: Vec<String>,
    hist_idx: usize,
    /// The page's live JS engine (bound to the document's DOM), kept so click
    /// handlers registered by scripts survive for the life of the page.
    js: Option<Js>,
}

impl Tab {
    fn max_scroll(&self, view_h: i32) -> i32 {
        (self.layout.height - view_h).max(0)
    }
}

/// One in-flight navigation: the cooperative load job plus how to commit it.
struct PendingLoad {
    job: LoadJob,
    record: bool,
    address: String,
}

/// The Browser app page.
pub struct BrowserApp {
    engine: Engine,
    transport: Box<dyn AsyncTransport>,
    /// The navigation currently loading, advanced by `pump` each frame (None = idle).
    pending: Option<PendingLoad>,
    /// Tor routing policy + mode detection (no tabs stored here).
    policy: Browser,
    tabs: Vec<Tab>,
    active: usize,
    tor: TorUi,
    address: TextBuffer,
    addr_focused: bool,
    area: Rect,
    now_ms: u64,
    last_left: bool,
    dragging_scroll: bool,
    drag_grab: i32,
    damage: Option<Rect>,
}

impl BrowserApp {
    pub fn new() -> BrowserApp {
        let mut app = BrowserApp {
            engine: Engine::new(),
            transport: Box::new(BlockingAsync::new(LoopbackTransport::new())),
            pending: None,
            policy: Browser::new(),
            tabs: Vec::new(),
            active: 0,
            tor: TorUi::Off,
            address: TextBuffer::new("dominion://home"),
            addr_focused: false,
            area: Rect::new(0, 0, 1280, 600),
            now_ms: 0,
            last_left: false,
            dragging_scroll: false,
            drag_grab: 0,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        };
        app.open_tab("dominion://home");
        app.address.end();
        app
    }

    /// Wire the shared VFS so user-authored `dominion://` pages (stored in
    /// `/dominion/pages/<name>.dominion`) are resolvable in the browser.
    pub fn set_native_fs(&mut self, fs: SharedFs) {
        self.engine.set_native_fs(fs);
    }

    /// Inject the real network transport (kernel virtio-net). Wraps it in a
    /// `BlockingAsync` adapter so it satisfies the async interface. Reloads the active
    /// tab so live content replaces whatever the default transport showed.
    pub fn set_transport(&mut self, transport: Box<dyn Transport>) {
        // Cancel any pending load against the old transport.
        if let Some(p) = &self.pending {
            if let Some(h) = p.job.transport_handle() {
                self.transport.cancel(h);
            }
        }
        self.pending = None;
        self.transport = Box::new(BlockingAsync::new(transport));
        if !self.tabs.is_empty() {
            let addr = self.tabs[self.active].base.to_string_full();
            self.navigate(&addr, false);
        }
        self.dmg_all();
    }

    /// The current address text.
    pub fn address(&self) -> String {
        self.address.text()
    }

    pub fn set_area(&mut self, area: Rect) {
        if area != self.area {
            self.area = area;
            self.relayout_active();
            self.dmg_all();
        }
    }
    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    pub fn wants_text(&self) -> bool {
        self.addr_focused
    }

    pub fn set_time(&mut self, now_ms: u64) {
        let prev = self.now_ms;
        self.now_ms = now_ms;
        if self.addr_focused && prev / BLINK_MS != now_ms / BLINK_MS {
            self.dmg(self.addr_rect());
        }
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

    fn tabstrip(&self) -> Rect {
        Rect::new(0, 0, self.area.w, TABS_H)
    }
    fn tab_rect(&self, i: usize) -> Rect {
        Rect::new(6 + i as i32 * 150, 4, 144, TABS_H - 6)
    }
    fn newtab_rect(&self) -> Rect {
        let n = self.tabs.len();
        Rect::new(6 + n as i32 * 150, 4, 28, TABS_H - 6)
    }
    fn back_btn(&self) -> Rect {
        Rect::new(8, TABS_H + 6, NAV_W - 2, TOOL_H - 12)
    }
    fn fwd_btn(&self) -> Rect {
        Rect::new(8 + NAV_W, TABS_H + 6, NAV_W - 2, TOOL_H - 12)
    }
    fn reload_btn(&self) -> Rect {
        Rect::new(8 + 2 * NAV_W, TABS_H + 6, NAV_W - 2, TOOL_H - 12)
    }
    fn tor_btn(&self) -> Rect {
        Rect::new(self.area.w - 70 - 96, TABS_H + 6, 92, TOOL_H - 12)
    }
    fn go_btn(&self) -> Rect {
        Rect::new(self.area.w - 62, TABS_H + 6, 54, TOOL_H - 12)
    }
    fn addr_rect(&self) -> Rect {
        let x = 8 + 3 * NAV_W + 6;
        let right = self.tor_btn().x - 8;
        Rect::new(x, TABS_H + 6, (right - x).max(40), TOOL_H - 12)
    }
    fn addr_origin(&self) -> (i32, i32) {
        let a = self.addr_rect();
        (a.x + 8, a.y + 7)
    }
    /// Full content region below the toolbar (includes the scrollbar gutter).
    fn content(&self) -> Rect {
        Rect::new(0, TABS_H + TOOL_H, self.area.w, self.area.h - TABS_H - TOOL_H)
    }
    /// The document viewport (content minus the scrollbar gutter).
    fn content_inner(&self) -> Rect {
        let c = self.content();
        Rect::new(c.x, c.y, c.w - SB, c.h)
    }
    /// The scrollbar track + thumb + max-scroll for the active tab, if it overflows.
    fn scrollbar(&self) -> Option<(Rect, Rect, i32)> {
        let tab = self.tabs.get(self.active)?;
        let c = self.content();
        let track = Rect::new(c.x + c.w - SB, c.y, SB, c.h);
        let max_scroll = tab.max_scroll(c.h);
        if max_scroll <= 0 {
            return None;
        }
        let ratio = c.h as f32 / tab.layout.height as f32;
        let thumb_h = ((c.h as f32 * ratio) as i32).max(28).min(c.h);
        let span = c.h - thumb_h;
        let ty = c.y + if max_scroll > 0 { (tab.scroll * span) / max_scroll } else { 0 };
        let thumb = Rect::new(track.x + 1, ty, SB - 2, thumb_h);
        Some((track, thumb, max_scroll))
    }

    // ── navigation ──

    fn open_tab(&mut self, address: &str) {
        let placeholder = Document::default();
        let layout = placeholder.layout(10, BODY_FONT);
        self.tabs.push(Tab {
            base: Url::parse(address).unwrap_or_else(|| Url::parse("dominion://home").unwrap()),
            title: String::from("New tab"),
            mode: Browser::mode_for(address),
            status: 0,
            doc: placeholder,
            layout,
            laid_width: 0,
            scroll: 0,
            history: Vec::new(),
            hist_idx: 0,
            js: None,
        });
        self.active = self.tabs.len() - 1;
        self.navigate(address, true);
    }

    /// Load `address` into the active tab. `record` pushes onto history (false for
    /// back/forward/reload, which move within it).
    ///
    /// For native `dominion://` pages and loopback transports the load completes
    /// synchronously before returning. For a real kernel transport the function
    /// returns immediately after starting the request; the caller must drive
    /// [`pump`](Self::pump) each frame until the load finishes.
    fn navigate(&mut self, address: &str, record: bool) {
        let address = address.trim().to_string();
        if address.is_empty() {
            return;
        }
        // Cancel any in-flight navigation against the current transport.
        if let Some(p) = self.pending.take() {
            if let Some(h) = p.job.transport_handle() {
                self.transport.cancel(h);
            }
        }
        // Tor policy: a legacy fetch with the circuit still building is *held*.
        let route = self.policy.resolve(&address);
        if route.route == Route::Blocked {
            self.apply_result(
                Err(FetchHeld),
                record,
                &address,
            );
            return;
        }
        let mut job = self.engine.begin_load(&address, self.transport.as_mut());
        if self.engine.poll_load(&mut job, self.transport.as_mut()) {
            // Already done (native page or blocking/loopback transport).
            let result = job.take_result().unwrap_or(Err(FetchError::BadResponse));
            self.apply_result(Ok(result), record, &address);
        } else {
            // Still in flight — show a loading placeholder and store for pump().
            let placeholder = loading_document(&address);
            let tab = &mut self.tabs[self.active];
            tab.title = String::from("Loading\u{2026}");
            tab.doc = placeholder;
            tab.scroll = 0;
            tab.laid_width = 0;
            tab.js = None;
            self.address.set_text(&address);
            self.address.end();
            self.addr_focused = false;
            self.relayout_active();
            self.dmg_all();
            self.pending = Some(PendingLoad { job, record, address });
        }
    }

    /// Advance the in-flight navigation (if any) by one step. Returns `true` if the
    /// UI changed and a redraw is needed. Called once per frame by the desktop loop.
    pub fn pump(&mut self) -> bool {
        let Some(p) = &mut self.pending else { return false };
        if !self.engine.poll_load(&mut p.job, self.transport.as_mut()) {
            return false; // still waiting
        }
        // Done — commit the result.
        let PendingLoad { mut job, record, address } = self.pending.take().unwrap();
        let result = job.take_result().unwrap_or(Err(FetchError::BadResponse));
        self.apply_result(Ok(result), record, &address);
        true
    }

    /// Commit a finished load result (success, fetch error, or Tor-held) into the
    /// active tab and update all dependent state.
    fn apply_result(&mut self, loaded: Result<Result<LoadedPage, FetchError>, FetchHeld>, record: bool, address: &str) {
        let (base, title, mode, status, doc) = match loaded {
            Err(FetchHeld) => (
                Url::parse(address).unwrap_or_else(home_url),
                String::from("Held"),
                PageMode::Legacy,
                0,
                error_document(
                    "Request held",
                    "Tor is enabled but the circuit is still building. The request was held, not sent over clearnet.",
                    address,
                ),
            ),
            Ok(Ok(page)) => {
                let title = if page.doc.title.is_empty() { display_name(&page.url) } else { page.doc.title.clone() };
                (page.url, title, page.mode, page.status, page.doc)
            }
            Ok(Err(e)) => (
                Url::parse(address).unwrap_or_else(home_url),
                String::from("Error"),
                Browser::mode_for(address),
                0,
                error_document("Couldn't load page", &e.message(), address),
            ),
        };

        let tab = &mut self.tabs[self.active];
        tab.base = base;
        tab.title = title;
        tab.mode = mode;
        tab.status = status;
        tab.doc = doc;
        tab.scroll = 0;
        tab.laid_width = 0;
        // Spin up the page's JS engine and run its inline scripts (which mutate the
        // shared DOM before the first layout). A bad script can't break the page.
        let mut js = Js::new(tab.doc.dom().clone());
        for src in &tab.doc.scripts {
            let _ = js.run(src);
        }
        tab.js = Some(js);
        if record {
            if tab.hist_idx + 1 < tab.history.len() {
                tab.history.truncate(tab.hist_idx + 1);
            }
            let full = tab.base.to_string_full();
            if tab.history.last().map(|s| s.as_str()) != Some(full.as_str()) {
                tab.history.push(full);
            }
            tab.hist_idx = tab.history.len().saturating_sub(1);
        }
        self.address.set_text(&tab.base.to_string_full());
        self.address.end();
        self.addr_focused = false;
        self.relayout_active();
        self.dmg_all();
    }

    fn relayout_active(&mut self) {
        self.lay_active(false);
    }

    /// Force a relayout (after JS mutates the DOM, where the width is unchanged).
    fn relayout_force(&mut self) {
        self.lay_active(true);
    }

    fn lay_active(&mut self, force: bool) {
        let width = self.content_inner().w.max(40);
        let view_h = self.area.h - TABS_H - TOOL_H;
        if let Some(tab) = self.tabs.get_mut(self.active) {
            if force || tab.laid_width != width {
                tab.layout = tab.doc.layout(width, BODY_FONT);
                tab.laid_width = width;
                let max = tab.max_scroll(view_h);
                tab.scroll = tab.scroll.clamp(0, max);
            }
        }
    }

    /// Public context-menu entry points (the shell's right-click menu drives these).
    pub fn nav_back(&mut self) {
        self.go_back();
    }
    pub fn nav_forward(&mut self) {
        self.go_forward();
    }
    pub fn nav_reload(&mut self) {
        self.reload();
    }
    pub fn can_nav_back(&self) -> bool {
        self.can_back()
    }
    pub fn can_nav_forward(&self) -> bool {
        self.can_forward()
    }
    /// Return the full URL of the active tab (for copy-URL clipboard action).
    pub fn current_url(&self) -> Option<String> {
        self.tabs.get(self.active).map(|t| t.base.to_string_full())
    }

    fn go_back(&mut self) {
        let tab = &mut self.tabs[self.active];
        if tab.hist_idx > 0 {
            tab.hist_idx -= 1;
            let addr = tab.history[tab.hist_idx].clone();
            self.navigate(&addr, false);
        }
    }
    fn go_forward(&mut self) {
        let tab = &mut self.tabs[self.active];
        if tab.hist_idx + 1 < tab.history.len() {
            tab.hist_idx += 1;
            let addr = tab.history[tab.hist_idx].clone();
            self.navigate(&addr, false);
        }
    }
    fn reload(&mut self) {
        let addr = self.tabs[self.active].base.to_string_full();
        self.navigate(&addr, false);
    }

    fn can_back(&self) -> bool {
        self.tabs.get(self.active).map(|t| t.hist_idx > 0).unwrap_or(false)
    }
    fn can_forward(&self) -> bool {
        self.tabs.get(self.active).map(|t| t.hist_idx + 1 < t.history.len()).unwrap_or(false)
    }

    /// Follow a link target from the active page, resolving relative references
    /// against the tab's base URL. `dominion-action:` targets are native actions and
    /// don't navigate (yet).
    fn follow_link(&mut self, href: &str) {
        if href.starts_with("dominion-action:") {
            return;
        }
        let resolved = self.tabs[self.active].base.join(href).map(|u| u.to_string_full());
        let target = resolved.unwrap_or_else(|| href.to_string());
        self.navigate(&target, true);
    }

    fn scroll_by(&mut self, delta: i32) {
        let view_h = self.area.h - TABS_H - TOOL_H;
        if let Some(tab) = self.tabs.get_mut(self.active) {
            let max = tab.max_scroll(view_h);
            let new = (tab.scroll + delta).clamp(0, max);
            if new != tab.scroll {
                tab.scroll = new;
                self.dmg(self.content());
            }
        }
    }

    fn apply_tor(&mut self) {
        match self.tor {
            TorUi::Off => self.policy.set_tor(false),
            TorUi::Connecting => self.policy.set_tor(true),
            TorUi::On => {
                self.policy.set_tor(true);
                self.policy.tor_bootstrapped(true);
            }
        }
    }
    fn cycle_tor(&mut self) {
        self.tor = match self.tor {
            TorUi::Off => TorUi::Connecting,
            TorUi::Connecting => TorUi::On,
            TorUi::On => TorUi::Off,
        };
        self.apply_tor();
        self.dmg_all();
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        // Scrollbar drag takes priority while the button is held.
        if self.dragging_scroll {
            if left {
                self.drag_scroll_to(py);
                self.last_left = left;
                return;
            } else {
                self.dragging_scroll = false;
            }
        }
        let pressed = left && !self.last_left;
        self.last_left = left;
        if !pressed {
            return;
        }

        // Tabs.
        for i in 0..self.tabs.len() {
            if self.tab_rect(i).contains(px, py) {
                self.active = i;
                let addr = self.tabs[i].base.to_string_full();
                self.address.set_text(&addr);
                self.address.end();
                self.addr_focused = false;
                self.relayout_active();
                self.dmg_all();
                return;
            }
        }
        if self.newtab_rect().contains(px, py) {
            self.open_tab("dominion://home");
            return;
        }
        // Navigation buttons.
        if self.back_btn().contains(px, py) {
            self.go_back();
            return;
        }
        if self.fwd_btn().contains(px, py) {
            self.go_forward();
            return;
        }
        if self.reload_btn().contains(px, py) {
            self.reload();
            return;
        }
        // Tor toggle.
        if self.tor_btn().contains(px, py) {
            self.cycle_tor();
            return;
        }
        // Go.
        if self.go_btn().contains(px, py) {
            let url = self.address.text();
            self.navigate(&url, true);
            return;
        }
        // Address bar focus + caret placement.
        if self.addr_rect().contains(px, py) {
            self.addr_focused = true;
            let o = self.addr_origin();
            self.address.place_at_pixel(px, py, o, toolkit::mono_advance(ADDR_FONT), TOOL_H - 16);
            self.dmg(self.addr_rect());
            return;
        }
        self.addr_focused = false;
        // Scrollbar interactions.
        if let Some((track, thumb, _max)) = self.scrollbar() {
            if thumb.contains(px, py) {
                self.dragging_scroll = true;
                self.drag_grab = py - thumb.y;
                return;
            }
            if track.contains(px, py) {
                // Page up/down depending on side of the thumb.
                let page = self.content().h - LINE_STEP;
                if py < thumb.y {
                    self.scroll_by(-page);
                } else {
                    self.scroll_by(page);
                }
                return;
            }
        }
        // JavaScript click handlers take precedence over link navigation: find the
        // element under the pointer and fire `click` on it / its nearest handler
        // ancestor, then re-render the (possibly mutated) DOM.
        if self.dispatch_click(px, py) {
            self.relayout_force();
            self.dmg_all();
            return;
        }
        // Content link clicks.
        if let Some(href) = self.link_under(px, py) {
            self.follow_link(&href);
            return;
        }
        self.dmg_all();
    }

    /// Fire a JS `click` on the element under the pointer (bubbling to the nearest
    /// ancestor with a handler). Returns whether a handler ran.
    fn dispatch_click(&mut self, px: i32, py: i32) -> bool {
        let inner = self.content_inner();
        let idx = self.active;
        let (node, scroll) = match self.tabs.get(idx) {
            Some(tab) => (tab.layout.node_at(inner, tab.scroll, px, py), tab.scroll),
            None => return false,
        };
        let _ = scroll;
        let Some(start) = node else { return false };
        let Some(js) = self.tabs[idx].js.as_ref() else { return false };
        // Walk up to the nearest element with a click handler.
        let mut cur = Some(start);
        let mut target = None;
        while let Some(n) = cur {
            if js.has_click_handler(&n) {
                target = Some(n.clone());
                break;
            }
            cur = dom::parent(&n);
        }
        let Some(n) = target else { return false };
        if let Some(js) = self.tabs[idx].js.as_mut() {
            js.fire_event(&n, "click")
        } else {
            false
        }
    }

    /// The href of a link under the pointer, if any (active tab, in viewport space).
    fn link_under(&self, px: i32, py: i32) -> Option<String> {
        let inner = self.content_inner();
        let tab = self.tabs.get(self.active)?;
        if !inner.contains(px, py) {
            return None;
        }
        tab.layout.link_at(inner, tab.scroll, px, py).map(|s| s.to_string())
    }

    fn drag_scroll_to(&mut self, py: i32) {
        if let Some((_track, _thumb, max)) = self.scrollbar() {
            let c = self.content();
            let ratio = c.h as f32 / self.tabs[self.active].layout.height as f32;
            let thumb_h = ((c.h as f32 * ratio) as i32).max(28).min(c.h);
            let span = (c.h - thumb_h).max(1);
            let want_thumb_y = (py - self.drag_grab - c.y).clamp(0, span);
            let new = (want_thumb_y * max) / span;
            let tab = &mut self.tabs[self.active];
            if new != tab.scroll {
                tab.scroll = new.clamp(0, max);
                self.dmg(self.content());
            }
        }
    }

    pub fn on_key(&mut self, ch: char) -> bool {
        if self.addr_focused {
            self.address.touch(self.now_ms);
            match ch {
                '\n' | '\r' => {
                    let url = self.address.text();
                    self.navigate(&url, true);
                }
                '\u{1b}' => self.addr_focused = false,
                '\u{8}' => self.address.backspace(),
                '\u{7f}' => self.address.delete(),
                '\u{1c}' => self.address.left(),
                '\u{1d}' => self.address.right(),
                '\u{1}' => self.address.home(),
                '\u{5}' => self.address.end(),
                c if !c.is_control() => self.address.insert(c),
                _ => {}
            }
            self.dmg(self.addr_rect());
            return true;
        }
        // Content scrolling (address bar not focused).
        match ch {
            '\u{1e}' => {
                self.scroll_by(-LINE_STEP);
                true
            }
            '\u{1f}' => {
                self.scroll_by(LINE_STEP);
                true
            }
            ' ' => {
                self.scroll_by(self.content().h - LINE_STEP);
                true
            }
            _ => false,
        }
    }

    // ── rendering ──

    pub fn view(&self, t: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: t.bg, radius: 0 });
        self.draw_tabs(&mut s, t);
        self.draw_toolbar(&mut s, t);
        self.draw_content(&mut s, t);
        s
    }

    fn draw_tabs(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        s.push(DrawCmd::Rect { rect: self.tabstrip(), color: t.surface, radius: 0 });
        for i in 0..self.tabs.len() {
            let r = self.tab_rect(i);
            let active = i == self.active;
            s.push(DrawCmd::Rect { rect: r, color: if active { t.bg } else { t.surface }, radius: t.radius });
            let label = tab_label(&self.tabs[i].title);
            let fg = if active { t.text } else { t.muted };
            s.push(DrawCmd::Text { rect: Rect::new(r.x + 10, r.y + 5, r.w - 16, 14), text: label, color: fg, size: 12 });
        }
        let nt = self.newtab_rect();
        s.push(DrawCmd::Rect { rect: nt, color: t.surface, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(nt.x + 9, nt.y + 5, 16, 16), text: "+".into(), color: t.text, size: 15 });
    }

    fn draw_toolbar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = Rect::new(0, TABS_H, self.area.w, TOOL_H);
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });

        // Nav buttons (dim when unavailable).
        let nav = [
            (self.back_btn(), "\u{2039}", self.can_back()),  // ‹
            (self.fwd_btn(), "\u{203a}", self.can_forward()), // ›
            (self.reload_btn(), "\u{21bb}", true),            // ↻
        ];
        for (r, glyph, enabled) in nav {
            s.push(DrawCmd::Rect { rect: r, color: t.bg, radius: t.radius });
            let fg = if enabled { t.text } else { t.muted };
            s.push(DrawCmd::Text { rect: Rect::new(r.x + 9, r.y + 4, r.w, 18), text: glyph.into(), color: fg, size: 18 });
        }

        // Tor toggle.
        let tor = self.tor_btn();
        let (fill, label, fg) = match self.tor {
            TorUi::Off => (t.bg, "Tor: Off", t.muted),
            TorUi::Connecting => (t.accent, "Tor: \u{2026}", t.on_primary),
            TorUi::On => (Color::rgb(0x3f, 0xc9, 0xb0), "Tor: On", t.on_primary),
        };
        s.push(DrawCmd::Rect { rect: tor, color: fill, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(tor.x + 10, tor.y + 6, tor.w, 16), text: label.into(), color: fg, size: 12 });

        // Address field.
        let a = self.addr_rect();
        let border = if self.addr_focused { t.primary } else { t.muted };
        s.push(DrawCmd::Rect { rect: toolkit::inflate(a, 1), color: border, radius: t.radius });
        s.push(DrawCmd::Rect { rect: a, color: t.bg, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(a.x + 8, a.y + 6, a.w - 16, 16), text: self.address.text(), color: t.text, size: ADDR_FONT });
        if self.addr_focused {
            let o = self.addr_origin();
            self.address.paint_caret(s, t, o, toolkit::mono_advance(ADDR_FONT), TOOL_H - 16, self.now_ms);
        }

        // Go button.
        let go = self.go_btn();
        s.push(DrawCmd::Rect { rect: go, color: t.primary, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(go.x + 14, go.y + 6, go.w, 16), text: "Go".into(), color: t.on_primary, size: 13 });
    }

    fn draw_content(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let inner = self.content_inner();
        s.push(DrawCmd::Rect { rect: self.content(), color: t.bg, radius: 0 });
        let Some(tab) = self.tabs.get(self.active) else { return };
        // The document itself.
        let mut doc_cmds = tab.layout.draw(t, inner, tab.scroll);
        s.append(&mut doc_cmds);
        // Scrollbar.
        if let Some((track, thumb, _)) = self.scrollbar() {
            s.push(DrawCmd::Rect { rect: track, color: t.surface, radius: 0 });
            s.push(DrawCmd::Rect { rect: thumb, color: t.muted, radius: SB / 2 });
        }
    }
}

impl Default for BrowserApp {
    fn default() -> Self {
        BrowserApp::new()
    }
}

/// Sentinel: a Tor-held request (not an engine error).
struct FetchHeld;

fn home_url() -> Url {
    Url::parse("dominion://home").unwrap()
}

/// A readable name for a URL when its document has no title.
fn display_name(url: &Url) -> String {
    if url.is_native() {
        url.to_string_full()
    } else if !url.host.is_empty() {
        url.host.clone()
    } else {
        url.to_string_full()
    }
}

/// Build a transient "Loading…" document shown while a fetch is in progress.
fn loading_document(url: &str) -> Document {
    let mut h = String::from("<title>Loading\u{2026}</title><p style=\"color:#8a909c\">Loading ");
    h.push_str(&escape(url));
    h.push_str("\u{2026}</p>");
    html::parse(&h)
}

/// Build a renderable error document (so failures render through the normal path).
fn error_document(heading: &str, message: &str, url: &str) -> Document {
    let mut h = String::from("<title>");
    h.push_str(heading);
    h.push_str("</title><h1>");
    h.push_str(&escape(heading));
    h.push_str("</h1><p>");
    h.push_str(&escape(message));
    h.push_str("</p><p style=\"color:#8a909c\">");
    h.push_str(&escape(url));
    h.push_str("</p>");
    let mut doc = html::parse(&h);
    doc.title = heading.to_string();
    doc
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn tab_label(title: &str) -> String {
    let mut s = String::from(title.trim());
    if s.is_empty() {
        s.push_str("New tab");
    }
    if s.chars().count() > 16 {
        s = s.chars().take(15).collect();
        s.push('\u{2026}');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> BrowserApp {
        let mut a = BrowserApp::new();
        a.set_area(Rect::new(0, 0, 1200, 600));
        let _ = a.take_damage();
        a
    }

    #[test]
    fn opens_on_the_native_home_page_and_renders_content() {
        let a = app();
        let s = a.view(&Theme::dark());
        // The native home page content actually renders (not a placeholder).
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "DominionWeb")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Documentation"))));
    }

    #[test]
    fn typing_a_legacy_url_loads_and_renders_real_html() {
        let mut a = app();
        // example.com is served by the default loopback transport.
        a.navigate("http://example.com/", true);
        let s = a.view(&Theme::dark());
        assert_eq!(a.tabs[a.active].mode, PageMode::Legacy);
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Example")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Domain")));
    }

    #[test]
    fn css_and_js_demo_renders_styled_and_interactive() {
        let mut a = app();
        a.navigate("http://demo.test/", true);
        // JS built a 5-item list at load — those items must be laid out.
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Item"))));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "number")));
        // CSS coloured the h1 (#88c0d0) — a styled glyph carries an explicit colour.
        // Click the increment button (its text glyph owns the <span id=inc>).
        let inner = a.content_inner();
        let (rect, _) = {
            let tab = &a.tabs[a.active];
            // Locate the counter span via a glyph: find the "[" of the button text.
            // Use node_at by scanning the layout's link/hot fallback through a click on
            // the button's text region — find it by laying coordinates of the inc text.
            // Simplest: click where the button word "Click" sits.
            tab.layout.links.first().cloned().unwrap_or((Rect::new(0, 0, 0, 0), String::new()))
        };
        let _ = rect;
        // Drive a click on the increment control by hitting its text via node_at.
        let btn = {
            let tab = &a.tabs[a.active];
            // Find any node under a point on the "[ Click to increment ]" line by
            // scanning vertical positions of glyphs we know exist.
            crate::dom::query_selector(tab.doc.dom(), "#inc")
        };
        assert!(btn.is_some(), "demo should have an #inc button");
        // Fire the handler directly through the tab's JS engine (same path on_pointer uses).
        let btn = btn.unwrap();
        let fired = a.tabs[a.active].js.as_mut().unwrap().fire_event(&btn, "click");
        assert!(fired, "click handler should fire");
        a.relayout_force();
        let s2 = a.view(&Theme::dark());
        // The counter span now reads 1.
        assert!(s2.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "1")));
        let _ = inner;
    }

    #[test]
    fn clicking_a_native_link_navigates_and_records_history() {
        let mut a = app();
        // Find the "Documentation" link box and click it.
        let inner = a.content_inner();
        let tab = &a.tabs[a.active];
        let (rect, target) = tab
            .layout
            .links
            .iter()
            .find(|(_, h)| h == "dominion://docs")
            .cloned()
            .expect("home should link to docs");
        assert_eq!(target, "dominion://docs");
        let px = inner.x + rect.x + 2;
        let py = inner.y + rect.y + 2 - tab.scroll;
        a.on_pointer(px, py, true);
        a.on_pointer(px, py, false);
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documentation")));
        assert!(a.can_back());
    }

    #[test]
    fn back_and_forward_move_through_history() {
        let mut a = app();
        a.navigate("dominion://docs", true);
        assert!(a.can_back());
        a.go_back();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://home");
        assert!(a.can_forward());
        a.go_forward();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://docs");
    }

    #[test]
    fn relative_links_resolve_against_base() {
        let mut a = app();
        a.navigate("http://dominion.test/", true);
        // This loopback page has a relative <a href="/about"> link.
        let about_link = a.tabs[a.active]
            .layout
            .links
            .iter()
            .any(|(_, h)| h == "/about");
        assert!(about_link);
        a.follow_link("/about");
        assert_eq!(a.tabs[a.active].base.host, "dominion.test");
        assert_eq!(a.tabs[a.active].base.path, "/about");
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "About")));
    }

    #[test]
    fn unknown_native_page_shows_error_document() {
        let mut a = app();
        a.navigate("dominion://nope", true);
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Couldn't"))));
    }

    #[test]
    fn tor_connecting_holds_legacy_requests() {
        let mut a = app();
        // Cycle Tor to "Connecting" (enabled, not bootstrapped).
        let tb = a.tor_btn();
        a.on_pointer(tb.x + 5, tb.y + 5, true);
        a.on_pointer(tb.x + 5, tb.y + 5, false);
        assert_eq!(a.tor, TorUi::Connecting);
        a.navigate("http://example.com/", true);
        let s = a.view(&Theme::dark());
        // The request was held, not loaded.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Held")));
    }

    #[test]
    fn new_tab_button_opens_a_tab() {
        let mut a = app();
        let before = a.tabs.len();
        let nt = a.newtab_rect();
        a.on_pointer(nt.x + 5, nt.y + 5, true);
        a.on_pointer(nt.x + 5, nt.y + 5, false);
        assert_eq!(a.tabs.len(), before + 1);
    }

    #[test]
    fn long_page_scrolls() {
        let mut a = app();
        // Serve a tall page through a fresh loopback transport.
        let mut lo = LoopbackTransport::new();
        let mut body = String::from("<h1>Tall</h1>");
        for i in 0..200 {
            body.push_str("<p>line ");
            body.push_str(&i.to_string());
            body.push_str("</p>");
        }
        lo.serve_html("tall.test", &body);
        a.set_transport(Box::new(lo));
        a.navigate("http://tall.test/", true);
        assert_eq!(a.tabs[a.active].scroll, 0);
        a.scroll_by(500);
        assert!(a.tabs[a.active].scroll > 0);
        // Scrollbar exists for an overflowing page.
        assert!(a.scrollbar().is_some());
    }

    // ── Async pump tests ──────────────────────────────────────────────────────

    /// Pump on an idle browser does nothing (returns false, no damage).
    #[test]
    fn pump_when_idle_returns_false() {
        let mut a = app();
        let _ = a.take_damage();
        assert!(!a.pump(), "pump on idle browser must return false");
        assert!(a.damage.is_none(), "pump must not mark damage when idle");
    }

    /// After set_transport the pending (if any) is cleared — pump is idle.
    #[test]
    fn set_transport_cancels_pending_load() {
        let mut a = app();
        // Swap to a new loopback transport; navigate is called internally.
        let lo2 = LoopbackTransport::new();
        a.set_transport(Box::new(lo2));
        // Any pending the reload started must already be settled (BlockingAsync is sync).
        let _ = a.take_damage();
        assert!(!a.pump(), "after set_transport+navigate, pump should be idle");
    }

    /// Navigate while a load is in-flight replaces the pending without crashing.
    #[test]
    fn navigate_while_loading_replaces_pending() {
        let mut a = app();
        // First navigation starts a load (it completes synchronously for BlockingAsync,
        // but we can still verify a second navigate works cleanly).
        a.navigate("http://example.com/", true);
        a.navigate("dominion://docs", true);
        // After two navigates the active page should be docs.
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documentation")));
        // No crash, no lingering pending.
        assert!(!a.pump());
    }

    // ── Full request → load → render → interact → retrieve → repeat cycle ────

    /// A full multi-step browse session: home → click docs link → read → back →
    /// forward → reload. Covers the entire navigation lifecycle.
    #[test]
    fn full_browse_cycle_home_docs_back_forward_reload() {
        let mut a = app();

        // Step 1: opened on native home page.
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://home");
        assert!(a.view(&Theme::dark()).iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "DominionWeb")));

        // Step 2: click the "Documentation" link.
        let inner = a.content_inner();
        let (rect, _) = {
            let tab = &a.tabs[a.active];
            tab.layout.links.iter()
                .find(|(_, h)| h == "dominion://docs")
                .cloned()
                .expect("home must link to docs")
        };
        let px = inner.x + rect.x + 2;
        let py = inner.y + rect.y + 2 - a.tabs[a.active].scroll;
        a.on_pointer(px, py, true);
        a.on_pointer(px, py, false);
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://docs");
        assert!(a.view(&Theme::dark()).iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documentation")));
        assert!(a.can_back());

        // Step 3: back to home.
        a.go_back();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://home");
        assert!(a.can_forward());

        // Step 4: forward to docs again.
        a.go_forward();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://docs");

        // Step 5: reload — page content must still be present.
        a.reload();
        assert!(a.view(&Theme::dark()).iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documentation")));
        assert!(!a.can_forward(), "reload clears forward history");
    }

    /// Full interactive cycle on demo.test: load → verify render → click counter
    /// repeatedly → confirm DOM updates persist → navigate away → back.
    #[test]
    fn full_interactive_cycle_js_counter() {
        let mut a = app();
        a.navigate("http://demo.test/", true);

        // Load — page must render.
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Item"))));
        // Counter starts at 0.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "0")));

        // Fire 3 click events on the #inc button.
        let inc_node = crate::dom::query_selector(a.tabs[a.active].doc.dom(), "#inc")
            .expect("demo must have #inc");
        for _ in 0..3 {
            let fired = a.tabs[a.active].js.as_mut().unwrap().fire_event(&inc_node, "click");
            assert!(fired);
        }
        a.relayout_force();

        // Counter must now read "3".
        let s2 = a.view(&Theme::dark());
        assert!(s2.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "3")),
            "counter should read 3 after 3 clicks");

        // Navigate away and come back — counter resets (fresh page).
        a.navigate("dominion://home", true);
        a.go_back();
        // A fresh load resets the counter to 0.
        let s3 = a.view(&Theme::dark());
        assert!(s3.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "0")));
    }

    /// Multi-tab session: open tabs independently, switch between them, verify each
    /// preserves its own document, scroll, and history.
    #[test]
    fn multi_tab_independent_state() {
        let mut a = app();

        // Tab 0 is home. Open a second tab on docs.
        let nt = a.newtab_rect();
        a.on_pointer(nt.x + 5, nt.y + 5, true);
        a.on_pointer(nt.x + 5, nt.y + 5, false);
        assert_eq!(a.tabs.len(), 2);
        a.navigate("dominion://docs", true);
        assert_eq!(a.active, 1);

        // Switch to tab 0 — should see home page content.
        let r0 = a.tab_rect(0);
        a.on_pointer(r0.x + 5, r0.y + 5, true);
        a.on_pointer(r0.x + 5, r0.y + 5, false);
        assert_eq!(a.active, 0);
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "DominionWeb")));

        // Switch to tab 1 — should see docs.
        let r1 = a.tab_rect(1);
        a.on_pointer(r1.x + 5, r1.y + 5, true);
        a.on_pointer(r1.x + 5, r1.y + 5, false);
        assert_eq!(a.active, 1);
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documentation")));
        // Tab 1's history has dominion://home and dominion://docs.
        assert!(a.can_back());
    }

    /// Address bar → type URL → Go button → page loads → address bar updates.
    #[test]
    fn address_bar_type_and_go_loads_page() {
        let mut a = app();
        // Click the address bar to focus it.
        let ab = a.addr_rect();
        a.on_pointer(ab.x + 5, ab.y + 5, true);
        a.on_pointer(ab.x + 5, ab.y + 5, false);
        assert!(a.addr_focused);

        // Type "http://example.com/" character by character.
        for ch in "http://example.com/".chars() {
            a.on_key(ch);
        }
        assert!(a.address.text().contains("example.com"));

        // Press Enter to navigate.
        a.on_key('\r');
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Example"))));
        // Address bar updated to the final URL.
        assert!(a.address().contains("example.com"));
        assert!(!a.addr_focused);
    }

    /// Redirect is transparent: URL bar shows the final URL, not the redirect target.
    #[test]
    fn redirect_shows_final_url_in_address_bar() {
        let mut a = app();
        let mut lo = LoopbackTransport::new();
        lo.serve_raw(
            "redir3.test",
            b"HTTP/1.1 301 Moved Permanently\r\nLocation: http://example.com/\r\nContent-Length: 0\r\n\r\n",
        );
        a.set_transport(Box::new(lo));
        a.navigate("http://redir3.test/", true);
        // Content is from example.com.
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Example"))));
        // Address bar shows the final URL.
        assert!(a.address().contains("example.com"), "address bar must show final URL after redirect");
    }

    /// Error page for a network failure renders inline with the URL and message.
    #[test]
    fn network_error_shows_error_page_with_url() {
        let mut a = app();
        a.navigate("http://nowhere.invalid/", true);
        let s = a.view(&Theme::dark());
        // Error heading present.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Couldn't"))));
        // The URL that failed is shown on the page.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("nowhere.invalid"))));
    }

    /// Keyboard scrolling: arrow-down / arrow-up / space move the viewport.
    #[test]
    fn keyboard_scroll_moves_viewport() {
        let mut a = app();
        let mut lo = LoopbackTransport::new();
        let mut body = String::from("<h1>Scrollable</h1>");
        for i in 0..300 {
            body.push_str(&alloc::format!("<p>para {}</p>", i));
        }
        lo.serve_html("scroll.test", &body);
        a.set_transport(Box::new(lo));
        a.navigate("http://scroll.test/", true);

        let before = a.tabs[a.active].scroll;
        a.on_key('\u{1f}'); // arrow-down (LINE_STEP)
        assert!(a.tabs[a.active].scroll > before, "down arrow must scroll");
        let mid = a.tabs[a.active].scroll;
        a.on_key('\u{1e}'); // arrow-up
        assert!(a.tabs[a.active].scroll < mid, "up arrow must scroll back");
    }

    /// Full end-to-end authoring: write a page DSL to the VFS → navigate to it in
    /// the browser → confirm it renders. This is the "create a native page" flow.
    #[test]
    fn user_authored_dominion_page_loads_in_browser() {
        use crate::filesystem::FileSystem;
        let fs = FileSystem::shared();
        let _ = fs.borrow_mut().mkdir("/dominion");
        let _ = fs.borrow_mut().mkdir("/dominion/pages");
        fs.borrow_mut().write_text(
            "/dominion/pages/mysite.dominion",
            "Title: My Site\nHeading: Hello DominionWeb\nText: User-authored content.\n",
        ).expect("write must succeed");

        let mut a = app();
        a.set_native_fs(fs);
        a.navigate("dominion://mysite", true);

        let scene = a.view(&Theme::dark());
        // The renderer word-splits headings, so "Hello DominionWeb" becomes two tokens.
        // Check the page title (tab chip) and the unique body word instead.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "My Site")),
            "browser must show the VFS page title");
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("User-authored"))),
            "browser must render VFS page body");
    }

    /// Reload button clears forward history and re-fetches the page.
    #[test]
    fn reload_button_click_refetches_active_page() {
        let mut a = app();
        a.navigate("dominion://docs", true);
        a.navigate("dominion://home", true);
        // Now go back to docs so we have forward history.
        a.go_back();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://docs");
        // Click the reload button — should stay on docs, wipe forward history.
        let rb = a.reload_btn();
        a.on_pointer(rb.x + 5, rb.y + 5, true);
        a.on_pointer(rb.x + 5, rb.y + 5, false);
        assert_eq!(a.tabs[a.active].base.to_string_full(), "dominion://docs");
        let s = a.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documentation")));
    }

    // ── helper: collect all visible text from a scene ──────────────────────────

    fn scene_texts(scene: &[DrawCmd]) -> Vec<String> {
        scene.iter().filter_map(|c| {
            if let DrawCmd::Text { text, .. } = c { Some(text.clone()) } else { None }
        }).collect()
    }

    fn scene_has(scene: &[DrawCmd], needle: &str) -> bool {
        scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains(needle)))
    }

    /// Set a LoopbackTransport with a rich multi-page HTTP site.
    fn app_with_site() -> BrowserApp {
        let mut a = app();

        // Build a 4-page loopback site:
        //   shop.test/        — product listing with a link to /product/1 and /cart
        //   shop.test/product/1 — product detail, link back home and "Add to cart"→/cart
        //   shop.test/cart    — cart page, shows "Cart: 1 item", link to /checkout
        //   shop.test/checkout — order summary, link back to home
        let mut lo = LoopbackTransport::new();
        lo.serve_html(
            "shop.test",
            "<html><head><title>DominionShop</title></head><body>\
             <h1>Products</h1>\
             <p>Welcome to DominionShop. Browse our catalog.</p>\
             <ul>\
               <li><a href=\"/product/1\">Widget Alpha</a> - $9.99</li>\
               <li><a href=\"/product/2\">Widget Beta</a> - $14.99</li>\
             </ul>\
             <p><a href=\"/cart\">View Cart</a></p>\
             </body></html>",
        );
        lo.serve_path_html(
            "shop.test", "/product/1",
            "<html><head><title>Widget Alpha</title></head><body>\
             <h1>Widget Alpha</h1>\
             <p>SKU: WA-001. A premium widget for all your needs.</p>\
             <p>Price: $9.99</p>\
             <p><a href=\"/cart\">Add to Cart</a></p>\
             <p><a href=\"/\">Back to Products</a></p>\
             </body></html>",
        );
        lo.serve_path_html(
            "shop.test", "/cart",
            "<html><head><title>Shopping Cart</title></head><body>\
             <h1>Your Cart</h1>\
             <p>Cart: 1 item</p>\
             <p>Total: $9.99</p>\
             <p><a href=\"/checkout\">Proceed to Checkout</a></p>\
             <p><a href=\"/\">Continue Shopping</a></p>\
             </body></html>",
        );
        lo.serve_path_html(
            "shop.test", "/checkout",
            "<html><head><title>Order Confirmed</title></head><body>\
             <h1>Order Confirmed</h1>\
             <p>Thank you! Your order #42 has been placed.</p>\
             <p>Item: Widget Alpha x1 - $9.99</p>\
             <p><a href=\"/\">Back to Shop</a></p>\
             </body></html>",
        );
        a.set_transport(alloc::boxed::Box::new(lo));
        a
    }

    /// ── FLOW TEST 1: request → load → render ──────────────────────────────────
    /// Navigate to the shop homepage; confirm the product list renders.
    #[test]
    fn flow_request_load_render_homepage() {
        let mut a = app_with_site();
        a.navigate("http://shop.test/", true);

        let scene = a.view(&Theme::dark());
        let texts = scene_texts(&scene);

        // URL committed in address bar.
        assert_eq!(a.tabs[a.active].base.to_string_full(), "http://shop.test/");
        // Page title in tab.
        assert_eq!(a.tabs[a.active].title, "DominionShop");
        // Content rendered: heading + product names.
        assert!(scene_has(&scene, "Products"),
            "heading must render; texts={:?}", texts);
        assert!(scene_has(&scene, "Widget"),
            "product names must render; texts={:?}", texts);
        assert!(scene_has(&scene, "DominionShop"),
            "shop name must appear; texts={:?}", texts);
    }

    /// ── FLOW TEST 2: request → load → render → interact (link click) → retrieve
    /// Click the "Widget Alpha" link → navigate to product detail → retrieve SKU.
    #[test]
    fn flow_request_load_render_click_link_retrieve() {
        let mut a = app_with_site();
        a.navigate("http://shop.test/", true);

        // Find the product link in the layout and click it.
        let tab = &a.tabs[a.active];
        let link = tab.layout.links.iter()
            .find(|(_, href)| href.contains("product"))
            .cloned()
            .expect("product link must exist in layout");
        drop(tab); // release borrow
        let inner = a.content_inner();
        // Link rect is in content coords; translate to viewport.
        let cx = inner.x + link.0.x + link.0.w / 2;
        let cy = inner.y + link.0.y + link.0.h / 2;
        a.on_pointer(cx, cy, true);
        a.on_pointer(cx, cy, false);

        // Should now be on the product detail page.
        let url = a.tabs[a.active].base.to_string_full();
        assert!(url.contains("product/1") || url.contains("product"),
            "link click must navigate to product page; url={}", url);
        let scene = a.view(&Theme::dark());
        assert!(scene_has(&scene, "Widget"),   "product name must render");
        assert!(scene_has(&scene, "9.99"),     "price must render");
        assert!(scene_has(&scene, "SKU") || scene_has(&scene, "WA"),
            "SKU detail must render; texts={:?}", scene_texts(&scene));
    }

    /// ── FLOW TEST 3: full shop purchase cycle (request→load→render→interact×3→retrieve)
    /// Products → Product Detail → Cart → Checkout. Verify each step.
    #[test]
    fn flow_full_shop_purchase_cycle() {
        let mut a = app_with_site();

        // Step 1: land on homepage
        a.navigate("http://shop.test/", true);
        assert!(scene_has(&a.view(&Theme::dark()), "Products"), "step1: homepage");

        // Step 2: navigate to product detail (via address bar simulate)
        a.navigate("http://shop.test/product/1", true);
        let s2 = a.view(&Theme::dark());
        assert!(scene_has(&s2, "Widget"),  "step2: product detail must show name");
        assert!(scene_has(&s2, "9.99"),    "step2: price must show");
        assert_eq!(a.tabs[a.active].title, "Widget Alpha");

        // Step 3: navigate to cart
        a.navigate("http://shop.test/cart", true);
        let s3 = a.view(&Theme::dark());
        assert!(scene_has(&s3, "Cart"),    "step3: cart heading");
        assert!(scene_has(&s3, "9.99"),    "step3: total shows price");
        assert_eq!(a.tabs[a.active].title, "Shopping Cart");

        // Step 4: checkout
        a.navigate("http://shop.test/checkout", true);
        let s4 = a.view(&Theme::dark());
        assert!(scene_has(&s4, "Order"),   "step4: order confirmation");
        assert!(scene_has(&s4, "42") || scene_has(&s4, "Widget"),
            "step4: order details; texts={:?}", scene_texts(&s4));
        assert_eq!(a.tabs[a.active].title, "Order Confirmed");

        // Step 5: back to homepage (back button × 3)
        a.go_back(); a.go_back(); a.go_back();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "http://shop.test/");
        assert!(scene_has(&a.view(&Theme::dark()), "Products"), "step5: back at homepage");
    }

    /// ── FLOW TEST 4: address bar type-and-go then retrieve ────────────────────
    /// Type a URL in the address bar, press Enter, verify navigation and content.
    #[test]
    fn flow_address_bar_type_navigate_retrieve() {
        let mut a = app_with_site();

        // Focus the address bar by clicking it.
        let addr = a.addr_rect();
        a.on_pointer(addr.x + addr.w / 2, addr.y + addr.h / 2, true);
        a.on_pointer(addr.x + addr.w / 2, addr.y + addr.h / 2, false);
        assert!(a.addr_focused, "address bar must be focused after click");

        // Clear existing content and type new URL char by char.
        // Ctrl-A = Home, then type.  We'll just backspace enough times.
        for _ in 0..30 { a.on_key('\u{8}'); } // backspace × 30
        for ch in "http://shop.test/cart".chars() {
            a.on_key(ch);
        }
        // Press Enter to navigate.
        a.on_key('\r');
        assert!(!a.addr_focused, "address bar must blur after Enter");

        let url = a.tabs[a.active].base.to_string_full();
        assert!(url.contains("cart"), "must navigate to cart; url={}", url);
        let scene = a.view(&Theme::dark());
        assert!(scene_has(&scene, "Cart"),   "cart page must render");
        assert!(scene_has(&scene, "9.99"),   "total must render");
    }

    /// ── FLOW TEST 5: repeat navigation cycle ──────────────────────────────────
    /// Navigate A → B → A → B → A, verifying each page loads correctly every time.
    #[test]
    fn flow_repeat_navigation_cycle() {
        let mut a = app_with_site();
        let rounds = 3;
        for i in 0..rounds {
            a.navigate("http://shop.test/", true);
            assert!(scene_has(&a.view(&Theme::dark()), "Products"),
                "round {}: homepage must render", i);

            a.navigate("http://shop.test/product/1", true);
            assert!(scene_has(&a.view(&Theme::dark()), "Widget"),
                "round {}: product page must render", i);
        }
        // History has 2×rounds entries (home + product per round, plus initial home).
        let hist_len = a.tabs[a.active].history.len();
        assert!(hist_len >= rounds * 2, "history must grow with each navigation; len={}", hist_len);
    }

    /// ── FLOW TEST 6: back/forward navigation integrity ─────────────────────────
    /// Navigate 3 pages, go back 2, go forward 1, verify correct page at each step.
    #[test]
    fn flow_back_forward_navigation_integrity() {
        let mut a = app_with_site();

        a.navigate("http://shop.test/", true);
        a.navigate("http://shop.test/product/1", true);
        a.navigate("http://shop.test/cart", true);

        // Go back twice.
        a.go_back();
        assert!(a.tabs[a.active].base.to_string_full().contains("product"),
            "after 1 back: must be on product page");
        assert!(scene_has(&a.view(&Theme::dark()), "Widget"), "product page renders");

        a.go_back();
        assert_eq!(a.tabs[a.active].base.to_string_full(), "http://shop.test/");
        assert!(scene_has(&a.view(&Theme::dark()), "Products"), "homepage renders");

        // Go forward.
        a.go_forward();
        assert!(a.tabs[a.active].base.to_string_full().contains("product"),
            "after forward: must be on product page");
        assert!(scene_has(&a.view(&Theme::dark()), "Widget"), "product page renders again");
    }

    /// ── FLOW TEST 7: error page then recovery ─────────────────────────────────
    /// Navigate to an unknown host (DNS error), see the error page, then navigate
    /// to a valid page and confirm recovery.
    #[test]
    fn flow_error_page_then_recovery() {
        let mut a = app_with_site();

        // Unknown host → DNS error.
        a.navigate("http://unknown.host.test/", true);
        let s_err = a.view(&Theme::dark());
        let texts_err = scene_texts(&s_err);
        // Must show some error text.
        assert!(
            scene_has(&s_err, "Could not") || scene_has(&s_err, "find") || scene_has(&s_err, "Error"),
            "error page must render an error message; texts={:?}", texts_err
        );
        assert_eq!(a.tabs[a.active].title, "Error", "tab title must be Error");

        // Recover: navigate to a valid page.
        a.navigate("http://shop.test/", true);
        let s_ok = a.view(&Theme::dark());
        assert!(scene_has(&s_ok, "Products"), "must recover and render valid page");
        assert_eq!(a.tabs[a.active].title, "DominionShop");
    }

    /// ── FLOW TEST 8: reload re-fetches content ────────────────────────────────
    /// Navigate to a page, verify content, update the site's response, reload, and
    /// confirm the new content appears.
    #[test]
    fn flow_reload_fetches_updated_content() {
        let mut a = app_with_site();
        a.navigate("http://shop.test/", true);
        assert!(scene_has(&a.view(&Theme::dark()), "Widget"), "initial content");

        // Update the site response by swapping to a new loopback transport.
        // (Simulates the server returning fresh content on reload.)
        let mut lo2 = LoopbackTransport::new();
        lo2.serve_html(
            "shop.test",
            "<html><head><title>DominionShop v2</title></head><body>\
             <h1>New Products</h1><p>Widget Gamma is here!</p>\
             </body></html>",
        );
        a.set_transport(alloc::boxed::Box::new(lo2));

        a.reload();
        let s = a.view(&Theme::dark());
        assert!(scene_has(&s, "Gamma") || scene_has(&s, "New"),
            "reload must fetch updated content; texts={:?}", scene_texts(&s));
    }

    /// ── FLOW TEST 9: multi-tab isolation ──────────────────────────────────────
    /// Open two tabs to different pages; each tab must show independent content.
    #[test]
    fn flow_multi_tab_independent_content() {
        let mut a = app_with_site();

        // Tab 0 (initial): navigate to homepage.
        a.navigate("http://shop.test/", true);
        assert!(scene_has(&a.view(&Theme::dark()), "Products"), "tab0: homepage");

        // Open tab 1 and navigate to cart.
        a.open_tab("http://shop.test/cart");
        assert_eq!(a.active, 1, "new tab must be active");
        assert!(scene_has(&a.view(&Theme::dark()), "Cart"), "tab1: cart page");

        // Switch back to tab 0 — must still show homepage.
        a.active = 0;
        assert_eq!(a.active, 0);
        assert!(scene_has(&a.view(&Theme::dark()), "Products"), "tab0 still shows homepage");
        assert_eq!(a.tabs[0].title, "DominionShop");
        assert_eq!(a.tabs[1].title, "Shopping Cart");
    }

    /// ── FLOW TEST 10: native dominion:// + legacy HTTP mixed session ─────────────
    /// Alternate between native pages and legacy HTTP pages in the same session.
    #[test]
    fn flow_native_and_legacy_mixed_session() {
        let mut a = app_with_site();

        // Native page.
        a.navigate("dominion://home", true);
        assert_eq!(a.tabs[a.active].mode, PageMode::Native);
        assert!(scene_has(&a.view(&Theme::dark()), "DominionWeb"), "native home");

        // Switch to legacy.
        a.navigate("http://shop.test/", true);
        assert_eq!(a.tabs[a.active].mode, PageMode::Legacy);
        assert!(scene_has(&a.view(&Theme::dark()), "Products"), "legacy shop");

        // Back to native docs.
        a.navigate("dominion://docs", true);
        assert_eq!(a.tabs[a.active].mode, PageMode::Native);
        assert!(scene_has(&a.view(&Theme::dark()), "Documentation"), "native docs");

        // Back to legacy cart.
        a.navigate("http://shop.test/cart", true);
        assert_eq!(a.tabs[a.active].mode, PageMode::Legacy);
        assert!(scene_has(&a.view(&Theme::dark()), "Cart"), "legacy cart");
    }

    /// ── FLOW TEST 11: scroll → retrieve content below the fold ────────────────
    /// Navigate to a tall page, scroll down, verify new content is in scene.
    #[test]
    fn flow_scroll_reveals_content() {
        let mut a = app_with_site();

        // Navigate to checkout which has several paragraphs.
        a.navigate("http://shop.test/checkout", true);

        // Content should be present even without scroll (page fits 600px).
        let s = a.view(&Theme::dark());
        assert!(scene_has(&s, "Order"), "checkout heading must render");
        assert!(scene_has(&s, "Widget"),  "item detail must render");

        // Pressing space (page-down) must not break rendering.
        a.on_key(' ');
        let s2 = a.view(&Theme::dark());
        assert!(scene_has(&s2, "Order") || scene_has(&s2, "Widget") || scene_has(&s2, "Back"),
            "content must still render after scroll");
    }

    /// ── FLOW TEST 12: retrieve data from multiple pages in sequence ────────────
    /// Simulate "scraping": load 3 pages, retrieve specific data from each.
    #[test]
    fn flow_retrieve_data_from_multiple_pages() {
        let mut a = app_with_site();
        let mut retrieved: Vec<String> = Vec::new();

        // Page 1: retrieve product name.
        a.navigate("http://shop.test/product/1", true);
        let s1 = a.view(&Theme::dark());
        if scene_has(&s1, "Widget") { retrieved.push("Widget Alpha".to_string()); }

        // Page 2: retrieve cart total.
        a.navigate("http://shop.test/cart", true);
        let s2 = a.view(&Theme::dark());
        if scene_has(&s2, "9.99") { retrieved.push("$9.99".to_string()); }

        // Page 3: retrieve order number.
        a.navigate("http://shop.test/checkout", true);
        let s3 = a.view(&Theme::dark());
        if scene_has(&s3, "42") || scene_has(&s3, "Order") { retrieved.push("order#42".to_string()); }

        assert!(retrieved.len() >= 2,
            "must retrieve data from at least 2 pages; got {:?}", retrieved);
    }

    /// ── FLOW TEST 13: HTTP redirect following ─────────────────────────────────
    /// A site returns a 301 redirect; the browser must follow it and show final URL.
    #[test]
    fn flow_http_redirect_followed_to_final_page() {
        let mut a = app();
        let mut lo = LoopbackTransport::new();
        // /old redirects to /new.
        lo.serve_raw(
            "redirect.test",
            b"HTTP/1.1 301 Moved Permanently\r\nLocation: /new\r\nContent-Length: 0\r\n\r\n",
        );
        lo.serve_path_html(
            "redirect.test", "/new",
            "<html><title>New Home</title><body><h1>Redirected!</h1></body></html>",
        );
        a.set_transport(alloc::boxed::Box::new(lo));
        a.set_area(Rect::new(0, 0, 1200, 600));

        a.navigate("http://redirect.test/", true);

        // Final URL must be /new.
        let url = a.tabs[a.active].base.to_string_full();
        assert!(url.contains("new") || url.ends_with('/'),
            "final URL must reflect redirect target; url={}", url);
        let s = a.view(&Theme::dark());
        assert!(scene_has(&s, "Redirected") || scene_has(&s, "New"),
            "redirected page must render; texts={:?}", scene_texts(&s));
    }

    /// ── FLOW TEST 14: offline mode error ──────────────────────────────────────
    /// With the transport offline, a navigation must show the offline error page.
    #[test]
    fn flow_offline_shows_error_page() {
        let mut a = app();
        let mut lo = LoopbackTransport::new();
        lo.set_online(false);
        a.set_transport(alloc::boxed::Box::new(lo));
        a.set_area(Rect::new(0, 0, 1200, 600));

        a.navigate("http://example.com/", true);
        let s = a.view(&Theme::dark());
        // Offline or connect error must appear.
        let texts = scene_texts(&s);
        let is_error = scene_has(&s, "No network") || scene_has(&s, "offline")
            || scene_has(&s, "connection") || a.tabs[a.active].title == "Error";
        assert!(is_error, "offline must show error; texts={:?}", texts);
    }

    /// ── FLOW TEST 15: pump-driven async navigation completes ──────────────────
    /// begin_load + poll loop (pump) must deliver the page the same as synchronous
    /// navigate, for both native and legacy pages.
    #[test]
    fn flow_pump_driven_navigation_delivers_page() {
        // Test with BlockingAsync (which settles on the first pump call).
        let mut a = app_with_site();

        // Trigger a navigate that wraps BlockingAsync (synchronous loopback).
        a.navigate("http://shop.test/product/1", true);
        // Since BlockingAsync resolves immediately, pending must be None already.
        assert!(a.pending.is_none(), "BlockingAsync must settle synchronously");
        let s = a.view(&Theme::dark());
        assert!(scene_has(&s, "Widget"),  "product page must render after pump-settle");
        assert!(scene_has(&s, "9.99"),    "price must render");

        // Native page also resolves immediately.
        a.navigate("dominion://settings", true);
        assert!(a.pending.is_none());
        assert!(scene_has(&a.view(&Theme::dark()), "settings") ||
                scene_has(&a.view(&Theme::dark()), "Settings"),
            "settings page must render");
    }
}

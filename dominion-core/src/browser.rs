//! The universal browser — native + legacy, with a **real-Tor** toggle on the legacy
//! side (see `docs/ui/universal-browser.md`).
//!
//! A browser tab is a *page object in a web view*. Two render paths:
//!
//! * **Native** — an Dominion-native semantic page (`dominionweb`/NDN) rendered directly
//!   to a [`crate::toolkit`] scene.
//! * **Legacy** — a today's-web page (HTML/JS) run **contained** in the sandbox VM
//!   with only network + surface capabilities.
//!
//! The legacy browser can route its traffic through the **actual Tor network** via a
//! toggle. This is the *control plane*: enable/disable, the SOCKS proxy endpoint of a
//! local Tor daemon, and bootstrap gating. When Tor is **enabled and bootstrapped**,
//! a legacy fetch is dialled through the Tor SOCKS proxy; when enabled but **not yet
//! bootstrapped**, the fetch is **held** rather than silently leaking over clearnet
//! (a real Tor browser blocks until the circuit is up); when **disabled**, fetches go
//! direct. (Onion routing over DominionLink is intentionally out of scope here.)
//!
//! Pure, safe `no_std`. The byte-level SOCKS5 connection lives in the kernel network
//! layer; this module decides *how every request is routed* and proves the toggle.

use crate::dominionweb::Page;
use crate::toolkit::{self, DrawCmd, Rect, Widget};
use crate::wasm::{Op, Sandbox as WasmSandbox, Trap};
use alloc::string::String;
use alloc::vec::Vec;

// ── WASM execution constants ──────────────────────────────────────────────────

/// Default memory cells allocated to a WASM guest running in the browser sandbox.
/// Each cell is an i64 (8 bytes), so this is 64 KiB — enough for typical web content
/// helpers without giving a guest an unbounded heap.
const WASM_GUEST_MEM_CELLS: usize = 8_192;

/// Default locals (registers) available to a WASM guest.
const WASM_GUEST_LOCALS: usize = 64;

/// Gas budget per WASM invocation. Chosen to permit real computation (loops, parsing,
/// simple codecs) while bounding runaway guests to a finite number of instructions.
/// TODO: expose as a per-tab policy knob once the Settings card lands.
const WASM_GAS_LIMIT: u64 = 1_000_000;

// ── WASM / content-type helpers ───────────────────────────────────────────────

/// Returns `true` if the URL or MIME type identifies a WebAssembly resource.
///
/// Routing decision: any resource that is either `application/wasm` or whose path
/// ends with `.wasm` is run through the [`WasmSandbox`] rather than the legacy JS
/// engine.  Everything else stays on the normal legacy HTML/JS path.
pub fn is_wasm_resource(url: &str, mime: Option<&str>) -> bool {
    if let Some(m) = mime {
        if m.eq_ignore_ascii_case("application/wasm") {
            return true;
        }
    }
    // Strip query-string / fragment before checking the extension.
    let path = url.split('?').next().unwrap_or(url);
    let path = path.split('#').next().unwrap_or(path);
    path.ends_with(".wasm")
}

/// The outcome of executing a WASM module inside the browser sandbox.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WasmOutcome {
    /// The module completed and left `result` on the top of its stack.
    Ok(i64),
    /// The sandbox contained a misbehaving guest — never propagated to the host.
    Trapped(Trap),
    /// The resource was identified as WASM but no compiled bytecode was provided
    /// (e.g. the fetch is still in-flight, or the compiler hasn't run yet).
    ///
    /// NOTE: AetherOS's WASM sandbox (`wasm::Sandbox`) operates on its own typed
    /// [`Op`] instruction set, not raw binary `.wasm` bytes.  A real pipeline would
    /// feed fetched bytes through a decoder that converts binary WASM sections into
    /// `Vec<Op>`; that decoder is not yet implemented.  Until it exists, fetched
    /// `.wasm` bytes produce `WasmOutcome::NotReady` rather than executing silently.
    NotReady,
}

/// Which render path a page uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageMode {
    /// Dominion-native semantic page (content-addressed, no DOM).
    Native,
    /// Legacy HTML/JS, contained in the sandbox VM.
    Legacy,
}

/// How a request leaves the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Straight out (clearnet).
    Direct,
    /// Through the local Tor daemon's SOCKS proxy.
    Tor,
    /// Tor is requested but the circuit isn't up yet — held, not leaked to clearnet.
    Blocked,
}

/// Configuration for routing the **legacy** browser through the actual Tor network.
#[derive(Clone, Debug)]
pub struct TorConfig {
    enabled: bool,
    bootstrapped: bool,
    host: String,
    port: u16,
}

impl TorConfig {
    /// Tor off by default, pointing at the conventional local Tor SOCKS endpoint.
    pub fn new() -> TorConfig {
        TorConfig { enabled: false, bootstrapped: false, host: String::from("127.0.0.1"), port: 9050 }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn bootstrapped(&self) -> bool {
        self.bootstrapped
    }

    /// The SOCKS5 endpoint the kernel net layer dials when routing through Tor.
    pub fn socks_endpoint(&self) -> (&str, u16) {
        (&self.host, self.port)
    }

    /// Point at a non-default Tor SOCKS proxy.
    pub fn set_endpoint(&mut self, host: &str, port: u16) {
        self.host = host.into();
        self.port = port;
    }
}

impl Default for TorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A browser tab.
pub struct BrowserTab {
    pub url: String,
    pub mode: PageMode,
}

/// The legacy page sandbox profile — a contained page may hold *only* these
/// capabilities (network + its own surface), and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegacyCap {
    Net,
    Surface,
}

/// The resolved request the network layer will perform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFetch {
    pub url: String,
    pub mode: PageMode,
    pub route: Route,
    /// For a Tor-routed fetch, the SOCKS endpoint to dial.
    pub via: Option<(String, u16)>,
}

/// The universal browser: tabs + the Tor control plane.
pub struct Browser {
    tabs: Vec<BrowserTab>,
    active: usize,
    tor: TorConfig,
}

impl Browser {
    pub fn new() -> Browser {
        Browser { tabs: Vec::new(), active: 0, tor: TorConfig::new() }
    }

    /// Decide the render mode for a URL: native for Dominion names/ids, legacy for
    /// `http(s)`/everything else.
    pub fn mode_for(url: &str) -> PageMode {
        if url.starts_with("dominion://") || url.starts_with("dominion:") || url.starts_with("ndn:") {
            PageMode::Native
        } else {
            PageMode::Legacy
        }
    }

    /// Open a URL in a new tab; returns the tab index.
    pub fn open(&mut self, url: &str) -> usize {
        let mode = Self::mode_for(url);
        self.tabs.push(BrowserTab { url: url.into(), mode });
        self.active = self.tabs.len() - 1;
        self.active
    }

    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }
    pub fn active_index(&self) -> usize {
        self.active
    }
    pub fn active(&self) -> Option<&BrowserTab> {
        self.tabs.get(self.active)
    }

    /// Make tab `i` active (no-op if out of range). Returns whether it changed.
    pub fn activate(&mut self, i: usize) -> bool {
        if i < self.tabs.len() {
            self.active = i;
            true
        } else {
            false
        }
    }

    /// The url of tab `i`, if it exists.
    pub fn tab_url(&self, i: usize) -> Option<&str> {
        self.tabs.get(i).map(|t| t.url.as_str())
    }

    // ── Tor control plane ──

    /// Enable or disable routing the **legacy** browser through the actual Tor
    /// network. Disabling also clears the bootstrap state.
    pub fn set_tor(&mut self, enabled: bool) {
        self.tor.enabled = enabled;
        if !enabled {
            self.tor.bootstrapped = false;
        }
    }

    /// Mark the local Tor daemon's circuit as established (bootstrap complete). The
    /// kernel net layer calls this once the SOCKS proxy answers and a circuit builds.
    pub fn tor_bootstrapped(&mut self, up: bool) {
        self.tor.bootstrapped = up;
    }

    pub fn tor_enabled(&self) -> bool {
        self.tor.enabled
    }
    pub fn tor(&self) -> &TorConfig {
        &self.tor
    }
    pub fn tor_mut(&mut self) -> &mut TorConfig {
        &mut self.tor
    }

    /// Decide how a fetch for `url` is routed. **Native pages never use Tor here**
    /// (Tor is a legacy-browser feature). For legacy pages: enabled+bootstrapped →
    /// Tor; enabled but not bootstrapped → Blocked (no clearnet leak); disabled →
    /// Direct.
    pub fn resolve(&self, url: &str) -> ResolvedFetch {
        let mode = Self::mode_for(url);
        let route = match mode {
            PageMode::Native => Route::Direct,
            PageMode::Legacy => {
                if !self.tor.enabled {
                    Route::Direct
                } else if self.tor.bootstrapped {
                    Route::Tor
                } else {
                    Route::Blocked
                }
            }
        };
        let via = if route == Route::Tor {
            let (h, p) = self.tor.socks_endpoint();
            Some((String::from(h), p))
        } else {
            None
        };
        ResolvedFetch { url: url.into(), mode, route, via }
    }

    /// The capabilities a legacy page is confined to — net + its own surface, and
    /// nothing else (the containment from `docs/ui/universal-browser.md`).
    pub fn legacy_caps() -> [LegacyCap; 2] {
        [LegacyCap::Net, LegacyCap::Surface]
    }

    /// Is `cap` something a contained legacy page may hold? (Default-closed: only
    /// Net and Surface; never filesystem, devices, other tabs, or the kernel.)
    pub fn legacy_may_hold(cap: LegacyCap) -> bool {
        Self::legacy_caps().contains(&cap)
    }

    // ── WASM execution ────────────────────────────────────────────────────────

    /// Execute a pre-compiled WASM module inside the browser sandbox.
    ///
    /// Isolation contract — mirrors [`legacy_caps`]:
    /// * The guest runs in its own linear memory (`WASM_GUEST_MEM_CELLS` cells).
    /// * No host imports are granted by default, so the guest cannot call out to the
    ///   kernel, other tabs, the filesystem, or the network.  The browser's
    ///   capability model is **default-closed**: authority must be explicitly handed
    ///   to the sandbox via [`wasm::Sandbox::grant`] before the guest can exercise it.
    /// * Execution is gas-bounded (`WASM_GAS_LIMIT`) so a runaway or malicious module
    ///   cannot monopolise the CPU.
    /// * Stack depth is bounded by [`wasm::MAX_STACK_DEPTH`] regardless of gas, so
    ///   the host heap cannot be exhausted through recursive pushes alone.
    ///
    /// If you need to expose a host capability (e.g. a surface-paint call), obtain a
    /// `Sandbox` via [`Browser::wasm_sandbox`], call [`wasm::Sandbox::grant`] with
    /// the appropriate closure, then call [`wasm::Sandbox::run`] directly.
    ///
    /// `args` are written into the first locals before execution begins (calling
    /// convention: local[0] = args[0], local[1] = args[1], …).
    pub fn execute_wasm(ops: Vec<Op>, args: &[i64]) -> WasmOutcome {
        let mut sb = Self::wasm_sandbox(ops);
        for (i, &v) in args.iter().enumerate() {
            if !sb.set_local(i, v) {
                break; // more args than locals — silently ignore extras
            }
        }
        match sb.run() {
            Ok(result) => WasmOutcome::Ok(result),
            Err(trap) => WasmOutcome::Trapped(trap),
        }
    }

    /// Build a fresh, **ungrated** WASM sandbox for the given bytecode.
    ///
    /// Callers that need to expose specific host functions (e.g. surface rendering or
    /// network I/O) should obtain the sandbox here, call [`wasm::Sandbox::grant`] for
    /// each capability they choose to expose, then call [`wasm::Sandbox::run`].
    /// This keeps the grant list explicit and auditable — the guest's whole world is
    /// exactly what the caller hands it.
    pub fn wasm_sandbox(ops: Vec<Op>) -> WasmSandbox {
        WasmSandbox::new(ops, WASM_GUEST_MEM_CELLS, WASM_GUEST_LOCALS, WASM_GAS_LIMIT)
    }

    /// Determine how a fetched resource should be executed.
    ///
    /// Returns `Some(WasmOutcome::NotReady)` when the URL/MIME identifies a WASM
    /// resource but no compiled bytecode (`ops`) is available yet (fetch in-flight or
    /// decoder not yet implemented).  Returns `Some(WasmOutcome)` when `ops` is
    /// supplied and the module has been run.  Returns `None` for non-WASM resources
    /// — the caller should continue with the normal HTML/JS pipeline.
    pub fn dispatch_resource(
        &self,
        url: &str,
        mime: Option<&str>,
        ops: Option<Vec<Op>>,
        args: &[i64],
    ) -> Option<WasmOutcome> {
        if !is_wasm_resource(url, mime) {
            return None; // not WASM — route through legacy JS engine as normal
        }
        // WASM resource: execute in the sandbox if bytecode is available.
        match ops {
            Some(code) => Some(Self::execute_wasm(code, args)),
            None => Some(WasmOutcome::NotReady),
        }
    }

    // ── rendering ──

    /// Render a native page to a toolkit scene: the title plus each text line as a
    /// label and each link as a button.
    pub fn render_native(page: &Page, theme: &toolkit::Theme, area: Rect) -> Vec<DrawCmd> {
        let mut children: Vec<Widget> = Vec::new();
        let mut id = 1u32;
        for line in page.render_text().split('\n') {
            if line.is_empty() {
                continue;
            }
            children.push(Widget::Label {
                id,
                text: String::from(line),
                size: toolkit::Size::Fixed(theme.font_size + 8),
            });
            id += 1;
        }
        if children.is_empty() {
            children.push(toolkit::label(1, "(empty page)"));
        }
        let col = Widget::Container {
            id: 0,
            axis: toolkit::Axis::Column,
            padding: theme.space,
            size: toolkit::Size::Flex(1),
            children,
        };
        toolkit::build_scene(&col, theme, area)
    }
}

impl Default for Browser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_detection_native_vs_legacy() {
        assert_eq!(Browser::mode_for("dominion://home"), PageMode::Native);
        assert_eq!(Browser::mode_for("ndn:/jayden/page"), PageMode::Native);
        assert_eq!(Browser::mode_for("https://example.com"), PageMode::Legacy);
        assert_eq!(Browser::mode_for("example.com"), PageMode::Legacy);
    }

    #[test]
    fn tor_disabled_routes_legacy_direct() {
        let b = Browser::new();
        assert!(!b.tor_enabled());
        let f = b.resolve("https://example.com");
        assert_eq!(f.route, Route::Direct);
        assert!(f.via.is_none());
    }

    #[test]
    fn enabling_tor_and_bootstrapping_routes_legacy_through_tor() {
        let mut b = Browser::new();
        b.set_tor(true);
        b.tor_bootstrapped(true);
        let f = b.resolve("https://check.torproject.org");
        assert_eq!(f.route, Route::Tor);
        // The fetch carries the SOCKS endpoint the net layer dials.
        assert_eq!(f.via, Some((String::from("127.0.0.1"), 9050)));
    }

    #[test]
    fn tor_enabled_but_not_bootstrapped_blocks_instead_of_leaking() {
        let mut b = Browser::new();
        b.set_tor(true); // circuit not up yet
        let f = b.resolve("https://example.com");
        // It must NOT fall back to clearnet — held until the circuit is up.
        assert_eq!(f.route, Route::Blocked);
        assert!(f.via.is_none());
    }

    #[test]
    fn native_pages_never_route_through_tor() {
        let mut b = Browser::new();
        b.set_tor(true);
        b.tor_bootstrapped(true);
        // Tor is a *legacy-browser* feature; native fetches stay direct.
        assert_eq!(b.resolve("dominion://home").route, Route::Direct);
    }

    #[test]
    fn disabling_tor_returns_to_direct_and_clears_bootstrap() {
        let mut b = Browser::new();
        b.set_tor(true);
        b.tor_bootstrapped(true);
        assert_eq!(b.resolve("https://x.com").route, Route::Tor);
        b.set_tor(false);
        assert!(!b.tor().bootstrapped());
        assert_eq!(b.resolve("https://x.com").route, Route::Direct);
    }

    #[test]
    fn tor_socks_endpoint_is_configurable() {
        let mut b = Browser::new();
        b.tor_mut().set_endpoint("10.0.0.5", 9150); // e.g. the Tor Browser bundle port
        b.set_tor(true);
        b.tor_bootstrapped(true);
        assert_eq!(b.resolve("https://x.com").via, Some((String::from("10.0.0.5"), 9150)));
    }

    #[test]
    fn legacy_pages_are_capability_confined() {
        // A contained legacy page may hold net + surface, and nothing else.
        assert!(Browser::legacy_may_hold(LegacyCap::Net));
        assert!(Browser::legacy_may_hold(LegacyCap::Surface));
        assert_eq!(Browser::legacy_caps().len(), 2);
    }

    // ── WASM integration tests ────────────────────────────────────────────────

    #[test]
    fn is_wasm_resource_detects_mime_and_extension() {
        assert!(is_wasm_resource("https://example.com/mod.wasm", None));
        assert!(is_wasm_resource("https://example.com/mod", Some("application/wasm")));
        assert!(is_wasm_resource("https://example.com/mod", Some("Application/Wasm"))); // case-insensitive
        assert!(!is_wasm_resource("https://example.com/script.js", None));
        assert!(!is_wasm_resource("https://example.com/script.js", Some("text/javascript")));
        // Query string must not confuse the extension check.
        assert!(is_wasm_resource("https://example.com/mod.wasm?v=2", None));
        assert!(!is_wasm_resource("https://example.com/mod.js?src=mod.wasm", None));
    }

    #[test]
    fn execute_wasm_runs_module_in_isolation() {
        use crate::wasm::Op;
        // Simple computation: (local0 + local1) * 3
        let ops = alloc::vec![
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::Add,
            Op::Const(3),
            Op::Mul,
            Op::Return,
        ];
        // args[0]=4, args[1]=6 → (4+6)*3 = 30
        assert_eq!(Browser::execute_wasm(ops, &[4, 6]), WasmOutcome::Ok(30));
    }

    #[test]
    fn execute_wasm_traps_are_contained_not_propagated() {
        use crate::wasm::{Op, Trap};
        // Divide by zero — must not panic the host.
        let ops = alloc::vec![Op::Const(1), Op::Const(0), Op::Div, Op::Return];
        assert_eq!(Browser::execute_wasm(ops, &[]), WasmOutcome::Trapped(Trap::DivideByZero));
    }

    #[test]
    fn execute_wasm_gas_bounded() {
        use crate::wasm::{Op, Trap};
        // Infinite loop — must terminate via gas exhaustion, never hang.
        let ops = alloc::vec![Op::Jump(0)];
        assert_eq!(Browser::execute_wasm(ops, &[]), WasmOutcome::Trapped(Trap::OutOfGas));
    }

    #[test]
    fn execute_wasm_no_host_imports_granted_by_default() {
        use crate::wasm::{Op, Trap};
        // A guest that attempts a host call with no grants must trap, not escape.
        let ops = alloc::vec![Op::Const(1), Op::Call { id: 0, argc: 1 }, Op::Return];
        assert_eq!(
            Browser::execute_wasm(ops, &[]),
            WasmOutcome::Trapped(Trap::UngrantedHostCall)
        );
    }

    #[test]
    fn dispatch_resource_routes_wasm_url_to_sandbox() {
        use crate::wasm::Op;
        let b = Browser::new();
        let ops = alloc::vec![Op::Const(42), Op::Return];
        let outcome = b.dispatch_resource("https://cdn.example.com/lib.wasm", None, Some(ops), &[]);
        assert_eq!(outcome, Some(WasmOutcome::Ok(42)));
    }

    #[test]
    fn dispatch_resource_returns_not_ready_when_no_bytecode() {
        let b = Browser::new();
        let outcome = b.dispatch_resource("https://cdn.example.com/lib.wasm", None, None, &[]);
        assert_eq!(outcome, Some(WasmOutcome::NotReady));
    }

    #[test]
    fn dispatch_resource_returns_none_for_non_wasm() {
        let b = Browser::new();
        let outcome = b.dispatch_resource("https://example.com/script.js", Some("text/javascript"), None, &[]);
        assert!(outcome.is_none());
    }

    #[test]
    fn open_tabs_and_render_native_page() {
        let mut b = Browser::new();
        b.open("dominion://home");
        b.open("https://example.com");
        assert_eq!(b.tab_count(), 2);
        assert_eq!(b.active().unwrap().mode, PageMode::Legacy);
        let page = Page::new("Welcome").heading("Hello").text("native semantic page");
        let scene = Browser::render_native(&page, &toolkit::Theme::dark(), Rect::new(0, 0, 400, 300));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("native semantic page"))));
    }
}

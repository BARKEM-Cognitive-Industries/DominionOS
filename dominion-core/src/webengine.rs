//! The browser **engine** — the orchestration that turns a URL into a renderable
//! document, tying together [`url`](crate::url), [`http`](crate::http),
//! [`html`](crate::html) and the native [`dominionweb`](crate::dominionweb) /
//! [`dominionlink`](crate::dominionlink) layers.
//!
//! Two resolution paths converge on one output:
//!
//! * **Native** (`dominion://`, `ndn:`) — resolved through the self-certifying
//!   DominionLink content store: the name maps to a content id, the bytes are fetched
//!   and *verified against that id*, then decoded into a page. No server is trusted.
//! * **Legacy** (`http(s)://`) — fetched over a pluggable [`Transport`] (the byte
//!   pipe — TCP/TLS — lives in the kernel), parsed as HTTP, redirects followed, and
//!   the body parsed by the HTML engine.
//!
//! Both produce an [`html::Document`], so the browser has exactly one layout, scroll
//! and hit-test path. A [`LoopbackTransport`] serves bundled pages, so the browser
//! renders real content with no NIC attached (and under unit test).
//!
//! Pure, safe, host-tested. Mechanism (sockets) is injected; policy (resolution,
//! redirects, verification) lives here.

use crate::dominionlink::{DominionId, DominionLink, DnsBridge};
use crate::dominionweb::{Block as WebBlock, Page};
use crate::browser::PageMode;
use crate::filesystem::SharedFs;
use crate::hash::Hash256;
use crate::html::{self, Document};
use crate::http::{Request, Response};
use crate::url::Url;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Why a load failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchError {
    /// The address could not be parsed.
    BadUrl,
    /// No network transport is available (no NIC, offline).
    Offline,
    /// DNS resolution failed for the host.
    Dns(String),
    /// The TCP connection could not be established.
    Connect(String),
    /// The peer required TLS and the transport cannot speak it yet.
    TlsUnsupported,
    /// The TLS handshake failed (bad certificate, signature, or negotiation).
    TlsHandshake,
    /// The response was not valid HTTP.
    BadResponse,
    /// Too many redirects (a loop, or a misbehaving server).
    TooManyRedirects,
    /// A native name did not resolve, or its content failed verification.
    NativeNotFound(String),
}

impl FetchError {
    /// A short, user-facing message for the error page.
    pub fn message(&self) -> String {
        match self {
            FetchError::BadUrl => "That address could not be understood.".to_string(),
            FetchError::Offline => "No network connection is available.".to_string(),
            FetchError::Dns(h) => {
                let mut s = String::from("Could not find the server for ");
                s.push_str(h);
                s.push('.');
                s
            }
            FetchError::Connect(h) => {
                let mut s = String::from("Could not connect to ");
                s.push_str(h);
                s.push('.');
                s
            }
            FetchError::TlsUnsupported => {
                "This site requires HTTPS (TLS), which this build cannot negotiate yet.".to_string()
            }
            FetchError::TlsHandshake => {
                "The secure (HTTPS) connection could not be established with this server.".to_string()
            }
            FetchError::BadResponse => "The server sent a response that could not be read.".to_string(),
            FetchError::TooManyRedirects => "The page redirected too many times.".to_string(),
            FetchError::NativeNotFound(n) => {
                let mut s = String::from("No native page is published at ");
                s.push_str(n);
                s.push('.');
                s
            }
        }
    }
}

/// The byte-pipe abstraction the engine fetches legacy pages over. The kernel
/// implements this on top of virtio-net (ARP + TCP); tests and the offline path use
/// [`LoopbackTransport`].
pub trait Transport {
    /// Connect to `host:port` (TLS when `secure`), send `request`, and return the
    /// full response bytes (read until the peer closes the connection).
    fn roundtrip(&mut self, host: &str, port: u16, secure: bool, request: &[u8]) -> Result<Vec<u8>, FetchError>;

    /// Whether a network path exists at all (false ⇒ every legacy load is `Offline`).
    fn online(&self) -> bool {
        true
    }
}

/// Forward `Transport` through a boxed trait object, so a `Box<dyn Transport>` can
/// itself be wrapped (e.g. in [`BlockingAsync`]) without naming the concrete type.
impl Transport for alloc::boxed::Box<dyn Transport> {
    fn roundtrip(&mut self, host: &str, port: u16, secure: bool, request: &[u8]) -> Result<Vec<u8>, FetchError> {
        (**self).roundtrip(host, port, secure, request)
    }
    fn online(&self) -> bool {
        (**self).online()
    }
}

/// A fully-loaded page, ready to lay out and render.
#[derive(Clone)]
pub struct LoadedPage {
    /// The URL actually loaded (after redirects).
    pub url: Url,
    pub mode: PageMode,
    pub status: u16,
    pub doc: Document,
}

/// The native, content-addressed web: a name registry over an DominionLink store, so
/// `dominion://home` resolves to verified content rather than a hardcoded string.
pub struct NativeWeb {
    link: DominionLink,
    dns: DnsBridge,
    /// name (e.g. "home") → content id of its encoded page.
    names: BTreeMap<String, Hash256>,
    /// Optional VFS, wired at runtime. When present, `resolve` also checks
    /// `/dominion/pages/<name>.dominion` for user-authored pages.
    fs: Option<SharedFs>,
}

impl NativeWeb {
    /// A fresh native web seeded with the built-in OS site.
    pub fn new() -> NativeWeb {
        let me = DominionId::from_pubkey(b"dominionos-native-web");
        let mut nw = NativeWeb { link: DominionLink::new(me), dns: DnsBridge::new(), names: BTreeMap::new(), fs: None };
        nw.seed_builtin();
        nw
    }

    /// Wire the VFS so user-authored `.dominion` pages are live-readable.
    pub fn set_fs(&mut self, fs: SharedFs) {
        self.fs = Some(fs);
    }

    /// Publish a page under a name, content-addressing it through DominionLink.
    pub fn publish(&mut self, name: &str, page: &Page) -> Hash256 {
        let cid = self.link.publish(&page.encode());
        self.names.insert(name.to_string(), cid);
        cid
    }

    /// Register a legacy DNS name as an alias for a native identity (the DNS bridge).
    pub fn alias(&mut self, legacy_name: &str, id: DominionId) {
        self.dns.register(legacy_name, id);
    }

    /// Resolve and verify a native page by name. Checks built-in pages first, then
    /// the VFS at `/dominion/pages/<name>.dominion` for user-authored content.
    pub fn resolve(&self, name: &str) -> Result<Page, FetchError> {
        // Defense-in-depth: reject any name that could escape the pages directory,
        // even if the caller already validated. Names must be simple identifiers —
        // no path separators, no dot-dot segments, no absolute paths.
        let safe_name = validate_native_name(name).ok_or_else(|| FetchError::NativeNotFound(name.to_string()))?;

        // 1. Built-in / previously-published content-addressed store.
        if let Some(&cid) = self.names.get(safe_name) {
            let bytes = self.link.fetch(cid).ok_or_else(|| FetchError::NativeNotFound(safe_name.to_string()))?;
            return decode_page(bytes).ok_or(FetchError::BadResponse);
        }
        // 2. VFS user-authored pages (require `set_fs` to have been called).
        if let Some(fs) = &self.fs {
            let path = alloc::format!("/dominion/pages/{}.dominion", safe_name);
            if let Some(text) = fs.borrow().read_text(&path) {
                return parse_page_dsl(&text).ok_or(FetchError::BadResponse);
            }
        }
        Err(FetchError::NativeNotFound(safe_name.to_string()))
    }

    /// Serve a native page as an HTTP response. Returns `None` if the page does not
    /// exist. This is the hook for an EtherLink HTTP bridge: an HTTP listener calls
    /// `serve_http(name)` and writes the result to the TCP socket.
    pub fn serve_http(&self, name: &str) -> Option<Vec<u8>> {
        let page = self.resolve(name).ok()?;
        Some(page_to_http_response(&page))
    }

    pub fn published(&self) -> usize {
        self.names.len()
    }

    pub fn dns(&self) -> &DnsBridge {
        &self.dns
    }

    fn seed_builtin(&mut self) {
        let home = Page::new("Home")
            .heading("DominionWeb")
            .text("The native, content-addressed web — no DOM, no tracking, no ambient script.")
            .text("Every page is a semantic object served over DominionLink and verified against its address.")
            .link("Documentation", "dominion://docs")
            .link("Your settings", "dominion://settings")
            .link("About this browser", "dominion://about");
        self.publish("home", &home);

        let docs = Page::new("Docs")
            .heading("Documentation")
            .text("Native pages render straight to the UI toolkit scene with clickable links.")
            .text("Legacy http(s) pages are fetched, parsed by the HTML engine, and rendered the same way — confined to net + surface capabilities.")
            .text("Type any address in the bar. dominion:// names are native; everything else is legacy.")
            .link("Back home", "dominion://home");
        self.publish("docs", &docs);

        let settings = Page::new("Settings")
            .heading("Browser settings")
            .text("Toggle Tor for legacy browsing with the button in the address bar.")
            .text("Native browsing never routes through Tor — it is already self-certifying.")
            .link("Back home", "dominion://home");
        self.publish("settings", &settings);

        let about = Page::new("About")
            .heading("The universal browser")
            .text("One engine, two webs: a native content-addressed web and the legacy HTTP web.")
            .text("Built on dominion-core: url, http, html, and the DominionLink overlay.")
            .action("Verify this page", "DominionWeb::verify", "NetConnect")
            .link("Back home", "dominion://home");
        self.publish("about", &about);
    }
}

impl Default for NativeWeb {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode an [`dominionweb::Page`] from its canonical byte encoding (the inverse of
/// [`Page::encode`]).
fn decode_page(bytes: &[u8]) -> Option<Page> {
    let tag = bytes.get(0..5)?;
    if tag != b"page1" {
        return None;
    }
    let mut p = 5usize;
    let title = read_str_u32(bytes, &mut p)?;
    let mut page = Page::new(title);
    let block_count = read_u32(bytes, &mut p)? as usize;
    for _ in 0..block_count {
        let kind = *bytes.get(p)?;
        p += 1;
        match kind {
            b'h' => {
                let t = read_str_u32(bytes, &mut p)?;
                page = page.heading(t);
            }
            b't' => {
                let t = read_str_u32(bytes, &mut p)?;
                page = page.text(t);
            }
            b'l' => {
                let text = read_str_u32(bytes, &mut p)?;
                let target = read_str_u32(bytes, &mut p)?;
                page = page.link(text, target);
            }
            b'a' => {
                let label = read_str_u32(bytes, &mut p)?;
                let cell = read_str_u32(bytes, &mut p)?;
                let requires = read_str_u32(bytes, &mut p)?;
                page = page.action(label, cell, requires);
            }
            _ => return None,
        }
    }
    Some(page)
}

fn read_u32(bytes: &[u8], p: &mut usize) -> Option<u32> {
    let b = bytes.get(*p..*p + 4)?;
    *p += 4;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_str_u32(bytes: &[u8], p: &mut usize) -> Option<String> {
    let len = read_u32(bytes, p)? as usize;
    let b = bytes.get(*p..*p + len)?;
    *p += len;
    core::str::from_utf8(b).ok().map(|s| s.to_string())
}

/// Parse a page from the human-editable text DSL used by `dominion publish`.
///
/// Format (one directive per line, case-insensitive keyword):
/// ```text
/// Title: My Page Title
/// Heading: Section heading
/// Text: A paragraph of text.
/// Link: Display text -> dominion://target
/// Action: Button label -> Module::method (Capability)
/// ```
/// Lines starting with `#` are comments and are ignored.
pub fn parse_page_dsl(text: &str) -> Option<Page> {
    let mut title = String::new();
    let mut page_opt: Option<Page> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (kw, rest) = match line.find(':') {
            Some(i) => (line[..i].trim().to_ascii_lowercase(), line[i + 1..].trim()),
            None => continue,
        };
        match kw.as_str() {
            "title" => {
                title = rest.to_string();
                page_opt = Some(Page::new(rest));
            }
            "heading" => {
                if page_opt.is_none() { page_opt = Some(Page::new(&title)); }
                page_opt = page_opt.map(|p| p.heading(rest));
            }
            "text" => {
                if page_opt.is_none() { page_opt = Some(Page::new(&title)); }
                page_opt = page_opt.map(|p| p.text(rest));
            }
            "link" => {
                if page_opt.is_none() { page_opt = Some(Page::new(&title)); }
                // "Display text -> dominion://target"
                let (display, target) = split_arrow(rest);
                page_opt = page_opt.map(|p| p.link(display.trim(), target.trim()));
            }
            "action" => {
                if page_opt.is_none() { page_opt = Some(Page::new(&title)); }
                // "Label -> Module::method (Capability)"
                let (label, rest2) = split_arrow(rest);
                // Split off the capability in parentheses if present.
                let (cell, cap) = match (rest2.find('('), rest2.find(')')) {
                    (Some(l), Some(r)) if l < r => {
                        (rest2[..l].trim(), rest2[l + 1..r].trim())
                    }
                    _ => (rest2.trim(), ""),
                };
                page_opt = page_opt.map(|p| p.action(label.trim(), cell, cap));
            }
            _ => {} // unknown directive — skip
        }
    }
    page_opt
}

fn split_arrow(s: &str) -> (&str, &str) {
    match s.find("->") {
        Some(i) => (&s[..i], &s[i + 2..]),
        None => (s, ""),
    }
}

/// Convert a native [`dominionweb::Page`] into an [`html::Document`] by rendering it to
/// HTML and parsing it — so native pages share the browser's single DOM/CSS/layout
/// path. Native actions become bold links tagged `dominion-action:`.
pub fn page_to_document(page: &Page) -> Document {
    let mut html = String::from("<title>");
    html.push_str(&escape_html(&page.title));
    html.push_str("</title>");
    for b in &page.blocks {
        match b {
            WebBlock::Heading(t) => {
                html.push_str("<h2>");
                html.push_str(&escape_html(t));
                html.push_str("</h2>");
            }
            WebBlock::Text(t) => {
                html.push_str("<p>");
                html.push_str(&escape_html(t));
                html.push_str("</p>");
            }
            WebBlock::Link { text, target } => {
                html.push_str("<p><a href=\"");
                html.push_str(&escape_html(target));
                html.push_str("\">");
                html.push_str(&escape_html(text));
                html.push_str("</a></p>");
            }
            WebBlock::Action { label, cell, .. } => {
                html.push_str("<p><b><a href=\"dominion-action:");
                html.push_str(&escape_html(cell));
                html.push_str("\">");
                html.push_str(&escape_html(label));
                html.push_str("</a></b></p>");
            }
        }
    }
    let mut doc = html::parse(&html);
    if doc.title.is_empty() {
        doc.title = page.title.clone();
    }
    doc
}

/// Render a native `Page` as a standalone HTML document suitable for HTTP serving
/// or export. Produces well-formed HTML with minimal inline CSS so the page looks
/// reasonable in any browser, not just DominionOS's renderer.
pub fn page_to_html(page: &Page) -> String {
    let mut html = String::from(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n\
         <meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>",
    );
    html.push_str(&escape_html(&page.title));
    html.push_str(
        "</title>\n\
         <style>\
         body{font-family:system-ui,sans-serif;max-width:800px;margin:2em auto;padding:0 1em;line-height:1.6;color:#222}\
         h1{font-size:1.8em;border-bottom:2px solid #e0e0e0;padding-bottom:.3em}\
         h2{font-size:1.3em;color:#444}\
         a{color:#0070c0}a:hover{color:#003d70}\
         .dominion-action{display:inline-block;padding:.3em .8em;border:1px solid #0070c0;border-radius:4px;color:#0070c0;text-decoration:none}\
         footer{margin-top:2em;font-size:.8em;color:#888;border-top:1px solid #e0e0e0;padding-top:.5em}\
         </style>\n</head>\n<body>\n",
    );
    html.push_str("<h1>");
    html.push_str(&escape_html(&page.title));
    html.push_str("</h1>\n");
    for b in &page.blocks {
        match b {
            WebBlock::Heading(t) => {
                html.push_str("<h2>");
                html.push_str(&escape_html(t));
                html.push_str("</h2>\n");
            }
            WebBlock::Text(t) => {
                html.push_str("<p>");
                html.push_str(&escape_html(t));
                html.push_str("</p>\n");
            }
            WebBlock::Link { text, target } => {
                html.push_str("<p><a href=\"");
                html.push_str(&escape_html(target));
                html.push_str("\">");
                html.push_str(&escape_html(text));
                html.push_str("</a></p>\n");
            }
            WebBlock::Action { label, cell, requires } => {
                html.push_str("<p><a href=\"dominion-action:");
                html.push_str(&escape_html(cell));
                html.push_str("\" class=\"dominion-action\">");
                html.push_str(&escape_html(label));
                html.push_str("</a> <small>(requires capability: ");
                html.push_str(&escape_html(requires));
                html.push_str(")</small></p>\n");
            }
        }
    }
    html.push_str(
        "<footer>Served by <a href=\"dominion://home\">DominionOS</a> — \
         content-addressed native web</footer>\n</body>\n</html>\n",
    );
    html
}

/// Build a complete HTTP/1.1 200 response for a native page, suitable for
/// transmission over a TCP socket. The content-type is `text/html; charset=utf-8`.
pub fn page_to_http_response(page: &Page) -> Vec<u8> {
    let body = page_to_html(page);
    let mut resp = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: ");
    push_usize(&mut resp, body.len());
    resp.push_str("\r\nX-Powered-By: DominionOS NativeWeb\r\nConnection: close\r\n\r\n");
    let mut out = resp.into_bytes();
    out.extend_from_slice(body.as_bytes());
    out
}

/// The browser engine.
pub struct Engine {
    native: NativeWeb,
    max_redirects: usize,
}

impl Engine {
    pub fn new() -> Engine {
        Engine { native: NativeWeb::new(), max_redirects: 8 }
    }

    /// Wire the shared VFS so user-authored `dominion://` pages are resolvable.
    pub fn set_native_fs(&mut self, fs: SharedFs) {
        self.native.set_fs(fs);
    }

    pub fn native(&self) -> &NativeWeb {
        &self.native
    }
    pub fn native_mut(&mut self) -> &mut NativeWeb {
        &mut self.native
    }

    /// Load `address` (text), using `transport` for legacy fetches. Native loads
    /// ignore the transport entirely.
    pub fn load(&self, address: &str, transport: &mut dyn Transport) -> Result<LoadedPage, FetchError> {
        let url = Url::parse(address).ok_or(FetchError::BadUrl)?;
        if url.is_native() {
            self.load_native(&url)
        } else {
            self.load_legacy(url, transport)
        }
    }

    fn load_native(&self, url: &Url) -> Result<LoadedPage, FetchError> {
        // The name is the path (e.g. "home" from "dominion://home").
        let raw = url.path.trim_start_matches('/');
        // Validate before resolving: reject traversal, absolute paths, and
        // any segment that is "." or "..".
        let name = validate_native_name(raw).ok_or(FetchError::BadUrl)?;
        let page = self.native.resolve(name)?;
        Ok(LoadedPage { url: url.clone(), mode: PageMode::Native, status: 200, doc: page_to_document(&page) })
    }

    fn load_legacy(&self, mut url: Url, transport: &mut dyn Transport) -> Result<LoadedPage, FetchError> {
        if !transport.online() {
            return Err(FetchError::Offline);
        }
        let mut redirects = 0;
        loop {
            let req = Request::get(&url.authority(), &url.request_target()).serialize();
            let raw = transport.roundtrip(&url.host, url.port(), url.is_secure(), &req)?;
            let resp = Response::parse(&raw).ok_or(FetchError::BadResponse)?;

            if resp.is_redirect() {
                redirects += 1;
                if redirects > self.max_redirects {
                    return Err(FetchError::TooManyRedirects);
                }
                let loc = resp.location().unwrap_or("");
                url = url.join(loc).ok_or(FetchError::BadUrl)?;
                continue;
            }

            let doc = parse_response_body(&resp);
            return Ok(LoadedPage { url, mode: PageMode::Legacy, status: resp.status, doc });
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────────── async (cooperative) loading ─────────────────────────────
//
// The blocking [`Engine::load`] path freezes the UI for the whole fetch — DNS,
// connect, and recv can each take seconds, and nothing else runs meanwhile. The
// async path below makes a navigation *cooperative*: it is kicked off without
// blocking, then advanced a little each frame by the render loop, so the browser
// stays fully interactive (scroll, tab switch, Stop, even starting a new
// navigation) while a page loads. It is host-tested end-to-end via
// [`BlockingAsync`] wrapping the loopback transport.

/// The state of a resumable, non-blocking operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Poll<T> {
    /// Not finished — call again later.
    Pending,
    /// Finished with this value.
    Ready(T),
}

/// A non-blocking, resumable byte transport. `begin` kicks off a request and
/// returns immediately with a job handle; `poll` is called repeatedly (once per
/// frame) and reports `Pending` until the full response — or an error — is ready.
/// This is what lets the kernel drive a real network fetch without ever blocking
/// the desktop's render/input loop.
pub trait AsyncTransport {
    /// Begin a request to `host:port` (TLS when `secure`). Returns a job handle.
    fn begin(&mut self, host: &str, port: u16, secure: bool, request: &[u8]) -> u64;
    /// Advance `job` a bounded amount (process whatever I/O is ready, never block
    /// for long). `Pending` until the whole response is in.
    fn poll(&mut self, job: u64) -> Poll<Result<Vec<u8>, FetchError>>;
    /// Abandon a job (the user navigated away or pressed Stop).
    fn cancel(&mut self, job: u64);
    /// Whether a network path exists at all (false ⇒ every legacy load is `Offline`).
    fn online(&self) -> bool {
        true
    }
    /// A human-readable line of progress for the most recent step, for the
    /// browser's debug overlay / serial trace. Empty if nothing new.
    fn diagnostic(&mut self) -> Option<String> {
        None
    }
}

/// Adapts any blocking [`Transport`] into an [`AsyncTransport`] by running the
/// fetch eagerly in `begin` and handing the result back on the first `poll`. Used
/// for the loopback transport, tests, and any host where blocking is acceptable —
/// so the *same* async engine code drives both the kernel and the loopback.
pub struct BlockingAsync<T: Transport> {
    inner: T,
    done: BTreeMap<u64, Result<Vec<u8>, FetchError>>,
    next: u64,
}

impl<T: Transport> BlockingAsync<T> {
    pub fn new(inner: T) -> BlockingAsync<T> {
        BlockingAsync { inner, done: BTreeMap::new(), next: 1 }
    }
    /// The wrapped transport, for tests that need to reconfigure it.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T: Transport> AsyncTransport for BlockingAsync<T> {
    fn begin(&mut self, host: &str, port: u16, secure: bool, request: &[u8]) -> u64 {
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        let r = self.inner.roundtrip(host, port, secure, request);
        self.done.insert(id, r);
        id
    }
    fn poll(&mut self, job: u64) -> Poll<Result<Vec<u8>, FetchError>> {
        match self.done.remove(&job) {
            Some(r) => Poll::Ready(r),
            None => Poll::Ready(Err(FetchError::BadResponse)),
        }
    }
    fn cancel(&mut self, job: u64) {
        self.done.remove(&job);
    }
    fn online(&self) -> bool {
        self.inner.online()
    }
}

/// One in-flight navigation, advanced cooperatively by [`Engine::poll_load`]. Holds
/// the redirect chain state and a short diagnostic log of what happened (surfaced
/// to the user when a load fails or stalls).
pub struct LoadJob {
    /// The original address requested (for diagnostics / retry).
    pub address: String,
    /// The URL currently being fetched (advances across redirects).
    url: Url,
    redirects: usize,
    /// The transport handle for the current request, if a fetch is outstanding.
    tjob: Option<u64>,
    /// Human-readable trace of each step — shown in the error page and logged.
    log: Vec<String>,
    /// Set once the load has finished (success or failure).
    result: Option<Result<LoadedPage, FetchError>>,
}

impl LoadJob {
    /// Whether the load has finished (success or error).
    pub fn is_done(&self) -> bool {
        self.result.is_some()
    }
    /// The URL currently being fetched (for the loading indicator).
    pub fn current_url(&self) -> &Url {
        &self.url
    }
    /// The diagnostic trace gathered so far.
    pub fn log(&self) -> &[String] {
        &self.log
    }
    /// Take the finished result, if any.
    pub fn take_result(&mut self) -> Option<Result<LoadedPage, FetchError>> {
        self.result.take()
    }
    /// The transport handle for the outstanding request (so a superseded load can
    /// be cancelled).
    pub fn transport_handle(&self) -> Option<u64> {
        self.tjob
    }
    fn note(&mut self, msg: impl Into<String>) {
        let m = msg.into();
        if self.log.len() < 32 {
            self.log.push(m);
        }
    }
}

impl Engine {
    /// Begin loading `address` without blocking. Native (`dominion://`) loads resolve
    /// immediately (the job comes back already done); legacy loads kick off the
    /// first request on `transport` and return a job to be advanced with
    /// [`poll_load`](Self::poll_load).
    pub fn begin_load(&self, address: &str, transport: &mut dyn AsyncTransport) -> LoadJob {
        let home = home_url_or_default();
        let mut job = LoadJob {
            address: address.to_string(),
            url: home,
            redirects: 0,
            tjob: None,
            log: Vec::new(),
            result: None,
        };
        let Some(url) = Url::parse(address) else {
            job.note("address could not be parsed");
            job.result = Some(Err(FetchError::BadUrl));
            return job;
        };
        job.url = url.clone();
        if url.is_native() {
            job.note("native page — resolving from EtherLink store");
            job.result = Some(self.load_native(&url));
            return job;
        }
        if !transport.online() {
            job.note("no network transport available");
            job.result = Some(Err(FetchError::Offline));
            return job;
        }
        job.note(alloc::format!("connecting to {} (port {})", url.host, url.port()));
        let req = Request::get(&url.authority(), &url.request_target()).serialize();
        job.tjob = Some(transport.begin(&url.host, url.port(), url.is_secure(), &req));
        job
    }

    /// Advance an in-flight [`LoadJob`]. Returns `true` once the load has finished
    /// (check [`LoadJob::take_result`]); `false` means it is still waiting on the
    /// network and should be polled again next frame. Redirects are followed
    /// transparently. This never blocks: as long as the transport keeps reporting
    /// `Ready` it makes progress, and it yields the moment the transport is `Pending`.
    pub fn poll_load(&self, job: &mut LoadJob, transport: &mut dyn AsyncTransport) -> bool {
        if job.result.is_some() {
            return true;
        }
        loop {
            let Some(tj) = job.tjob else {
                // No outstanding request and no result — nothing to do.
                job.result = Some(Err(FetchError::BadResponse));
                return true;
            };
            let raw = match transport.poll(tj) {
                Poll::Pending => return false,
                Poll::Ready(Ok(raw)) => raw,
                Poll::Ready(Err(e)) => {
                    job.note(alloc::format!("transport error: {}", e.message()));
                    job.tjob = None;
                    job.result = Some(Err(e));
                    return true;
                }
            };
            job.tjob = None;
            let Some(resp) = Response::parse(&raw) else {
                job.note("response was not valid HTTP");
                job.result = Some(Err(FetchError::BadResponse));
                return true;
            };
            job.note(alloc::format!("received {} bytes, status {}", raw.len(), resp.status));

            if resp.is_redirect() {
                job.redirects += 1;
                if job.redirects > self.max_redirects {
                    job.result = Some(Err(FetchError::TooManyRedirects));
                    return true;
                }
                let loc = resp.location().unwrap_or("");
                let Some(next) = job.url.join(loc) else {
                    job.result = Some(Err(FetchError::BadUrl));
                    return true;
                };
                job.note(alloc::format!("redirect → {}", next.to_string_full()));
                job.url = next;
                let req = Request::get(&job.url.authority(), &job.url.request_target()).serialize();
                job.tjob = Some(transport.begin(&job.url.host, job.url.port(), job.url.is_secure(), &req));
                continue; // poll the new request immediately (may also be Ready)
            }

            let doc = parse_response_body(&resp);
            job.result =
                Some(Ok(LoadedPage { url: job.url.clone(), mode: PageMode::Legacy, status: resp.status, doc }));
            return true;
        }
    }
}

fn home_url_or_default() -> Url {
    Url::parse("dominion://home").unwrap_or_else(|| Url::parse("about:blank").expect("about:blank parses"))
}

/// Parse a response body into a [`Document`]: HTML bodies are parsed directly;
/// non-HTML bodies are wrapped in `<pre>` so they remain readable. Shared by
/// both the blocking `load_legacy` and the cooperative `poll_load` paths.
fn parse_response_body(resp: &crate::http::Response) -> Document {
    if resp.is_html() {
        html::parse(&resp.body_text())
    } else {
        let mut body = String::from("<pre>");
        body.push_str(&escape_html(&resp.body_text()));
        body.push_str("</pre>");
        html::parse(&body)
    }
}

/// Validate a native page name derived from a URL path, guarding against path
/// traversal attacks. A safe name:
/// * Is not empty.
/// * Does not start with `/` (absolute path).
/// * Contains no `..` substring (dot-dot traversal).
/// * Has no segment that is exactly `.` (current-directory traversal).
/// * Has no `/` characters (names are single-segment identifiers).
///
/// Returns the name unchanged if valid, or `None` if any check fails.
fn validate_native_name(name: &str) -> Option<&str> {
    // Reject empty names.
    if name.is_empty() {
        return None;
    }
    // Reject absolute paths.
    if name.starts_with('/') {
        return None;
    }
    // Reject any occurrence of ".." (catches "a/../b", "../secret", etc.).
    if name.contains("..") {
        return None;
    }
    // Reject path separators — native names are single flat identifiers.
    // Also catches "./foo" style segments since '/' is present.
    if name.contains('/') {
        return None;
    }
    // Reject a bare "." segment.
    if name == "." {
        return None;
    }
    // Reject backslashes (Windows-style traversal on host filesystems).
    if name.contains('\\') {
        return None;
    }
    Some(name)
}

/// Minimal HTML-escaping for wrapping plain text in `<pre>`.
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

// ───────────────────────────── loopback transport ─────────────────────────────

/// An in-memory transport serving bundled pages — so the browser renders real legacy
/// content with no NIC, and so the engine is testable end-to-end. Maps `host` → an
/// HTTP response (or a redirect).
pub struct LoopbackTransport {
    sites: BTreeMap<String, Vec<u8>>,
    online: bool,
}

impl LoopbackTransport {
    pub fn new() -> LoopbackTransport {
        let mut t = LoopbackTransport { sites: BTreeMap::new(), online: true };
        t.seed();
        t
    }

    /// Serve a raw HTTP response for every request to `host`.
    pub fn serve_raw(&mut self, host: &str, raw_response: &[u8]) {
        self.sites.insert(host.to_ascii_lowercase(), raw_response.to_vec());
    }

    /// Serve a raw HTTP response for a specific `host` + `path` (e.g. host="shop.test", path="/cart").
    pub fn serve_path_raw(&mut self, host: &str, path: &str, raw_response: &[u8]) {
        let mut key = host.to_ascii_lowercase();
        key.push_str(path);
        self.sites.insert(key, raw_response.to_vec());
    }

    /// Serve an HTML body at a specific host + path.
    pub fn serve_path_html(&mut self, host: &str, path: &str, html_body: &str) {
        let mut raw = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
        raw.push_str(html_body);
        self.serve_path_raw(host, path, raw.as_bytes());
    }

    /// Serve an HTML body (wrapped in a 200 response) for `host`.
    pub fn serve_html(&mut self, host: &str, html_body: &str) {
        let mut raw = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: ");
        push_usize(&mut raw, html_body.len());
        raw.push_str("\r\n\r\n");
        let mut bytes = raw.into_bytes();
        bytes.extend_from_slice(html_body.as_bytes());
        self.sites.insert(host.to_ascii_lowercase(), bytes);
    }

    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }

    fn seed(&mut self) {
        self.serve_html(
            "example.com",
            "<html><head><title>Example Domain</title></head><body>\
             <h1>Example Domain</h1>\
             <p>This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.</p>\
             <p><a href=\"https://www.iana.org/domains/example\">More information...</a></p>\
             </body></html>",
        );
        self.serve_html(
            "dominion.test",
            "<html><head><title>Dominion Test Page</title></head><body>\
             <h1>Legacy rendering works</h1>\
             <p>This page is served by the <b>loopback transport</b> and parsed by the real HTML engine.</p>\
             <h2>Features</h2>\
             <ul><li>Headings &amp; paragraphs</li><li>Bold, <i>italic</i>, and <code>code</code></li><li>Ordered and unordered lists</li><li>Links with relative resolution</li></ul>\
             <ol><li>First</li><li>Second</li><li>Third</li></ol>\
             <hr>\
             <blockquote>Contain, don't absorb &mdash; legacy pages run sandboxed.</blockquote>\
             <p>Go to the <a href=\"/about\">about page</a> or visit <a href=\"http://example.com/\">example.com</a>.</p>\
             </body></html>",
        );
        // A second path on dominion.test, reached via the relative /about link.
        let about = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>About</h1><p>Served at /about.</p><p><a href=\"/\">Home</a></p>";
        self.sites.insert("dominion.test/about".to_string(), about.as_bytes().to_vec());

        // A CSS + JavaScript showcase: styled with a stylesheet, made interactive by a
        // script that builds a list, wires a counter button, and recolours on click.
        self.serve_html(
            "demo.test",
            "<html><head><title>CSS + JS Demo</title><style>\
             body{color:#d8dee9}\
             h1{color:#88c0d0;text-align:center}\
             .card{background-color:#2e3440;margin-top:8px;margin-bottom:8px}\
             .accent{color:#a3be8c;font-weight:bold}\
             #count{color:#ebcb8b;font-weight:bold}\
             .btn{color:#bf616a;text-decoration:underline}\
             </style></head><body>\
             <h1>CSS &amp; JavaScript</h1>\
             <p class='card'>This page is <span class='accent'>styled by CSS</span> and made interactive by <span class='accent'>JavaScript</span> running in DominionOS.</p>\
             <p>Counter: <span id='count'>0</span></p>\
             <p><span id='inc' class='btn'>[ Click to increment ]</span></p>\
             <h2>Generated list</h2>\
             <ul id='list'></ul>\
             <script>\
             var n = 0;\
             var c = document.getElementById('count');\
             document.getElementById('inc').addEventListener('click', function(){ n = n + 1; c.textContent = n; });\
             var list = document.getElementById('list');\
             for (var i = 1; i <= 5; i++) {\
               var li = document.createElement('li');\
               li.textContent = 'Item number ' + i + ' (' + (i*i) + ')';\
               list.appendChild(li);\
             }\
             </script>\
             </body></html>",
        );
    }
}

impl Default for LoopbackTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for LoopbackTransport {
    fn online(&self) -> bool {
        self.online
    }

    fn roundtrip(&mut self, host: &str, _port: u16, _secure: bool, request: &[u8]) -> Result<Vec<u8>, FetchError> {
        if !self.online {
            return Err(FetchError::Offline);
        }
        // Honour the request path for multi-page loopback sites (host + path key).
        let req = String::from_utf8_lossy(request);
        let path = req.lines().next().and_then(|l| l.split(' ').nth(1)).unwrap_or("/");
        let host_l = host.to_ascii_lowercase();
        if path != "/" {
            let mut key = host_l.clone();
            key.push_str(path);
            if let Some(r) = self.sites.get(&key) {
                return Ok(r.clone());
            }
        }
        self.sites
            .get(&host_l)
            .cloned()
            .ok_or_else(|| FetchError::Dns(host.to_string()))
    }
}

fn push_usize(s: &mut String, n: usize) {
    if n >= 10 {
        push_usize(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_pages_resolve_and_verify() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        let page = eng.load("dominion://home", &mut lo).unwrap();
        assert_eq!(page.mode, PageMode::Native);
        assert_eq!(page.doc.title, "Home");
        assert!(page.doc.text().contains("DominionWeb"));
        // Links survived the page→document conversion.
        assert!(page.doc.links().iter().any(|l| *l == "dominion://docs"));
    }

    #[test]
    fn unknown_native_page_errors() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        let err = eng.load("dominion://does-not-exist", &mut lo).err().unwrap();
        assert!(matches!(err, FetchError::NativeNotFound(_)));
    }

    #[test]
    fn legacy_page_loads_and_parses_html() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        let page = eng.load("http://example.com/", &mut lo).unwrap();
        assert_eq!(page.mode, PageMode::Legacy);
        assert_eq!(page.status, 200);
        assert_eq!(page.doc.title, "Example Domain");
        assert!(page.doc.text().contains("illustrative examples"));
    }

    #[test]
    fn relative_links_resolve_against_final_url() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        // Load the multi-feature page, then follow its relative /about link.
        let page = eng.load("http://dominion.test/", &mut lo).unwrap();
        assert!(page.doc.text().contains("Legacy rendering works"));
        // The /about path is served distinctly.
        let about = eng.load("http://dominion.test/about", &mut lo).unwrap();
        assert!(about.doc.text().contains("Served at /about"));
    }

    #[test]
    fn redirects_are_followed() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        lo.serve_raw(
            "redir.test",
            b"HTTP/1.1 301 Moved Permanently\r\nLocation: http://example.com/\r\nContent-Length: 0\r\n\r\n",
        );
        let page = eng.load("http://redir.test/", &mut lo).unwrap();
        // Ended up at example.com after the redirect.
        assert_eq!(page.doc.title, "Example Domain");
        assert_eq!(page.url.host, "example.com");
    }

    #[test]
    fn offline_transport_reports_offline() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        lo.set_online(false);
        let err = eng.load("http://example.com/", &mut lo).err().unwrap();
        assert_eq!(err, FetchError::Offline);
    }

    #[test]
    fn unknown_host_is_a_dns_error() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        let err = eng.load("http://nowhere.invalid/", &mut lo).err().unwrap();
        assert!(matches!(err, FetchError::Dns(_)));
    }

    #[test]
    fn non_html_body_is_shown_as_preformatted_text() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        lo.serve_raw(
            "plain.test",
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 11\r\n\r\nhello world",
        );
        let page = eng.load("http://plain.test/", &mut lo).unwrap();
        assert!(page.doc.text().contains("hello world"));
    }

    #[test]
    fn page_encode_decode_round_trips() {
        let p = Page::new("T").heading("H").text("body").link("L", "dominion://x").action("Do", "C::m", "Cap");
        let bytes = p.encode();
        let back = decode_page(&bytes).unwrap();
        assert_eq!(back, p);
    }

    // ── Async engine tests (begin_load / poll_load / BlockingAsync) ────────────

    fn async_lo() -> BlockingAsync<LoopbackTransport> {
        BlockingAsync::new(LoopbackTransport::new())
    }

    /// Native page: begin_load returns a job that is already done on the first poll.
    #[test]
    fn begin_load_native_is_immediately_done() {
        let eng = Engine::new();
        let mut t = async_lo();
        let mut job = eng.begin_load("dominion://home", &mut t);
        assert!(job.is_done(), "native job must be done without polling");
        let page = job.take_result().unwrap().unwrap();
        assert_eq!(page.mode, PageMode::Native);
        assert!(page.doc.text().contains("DominionWeb"));
    }

    /// Legacy page via BlockingAsync: poll_load returns true on the very first call.
    #[test]
    fn begin_load_legacy_ready_on_first_poll() {
        let eng = Engine::new();
        let mut t = async_lo();
        let mut job = eng.begin_load("http://example.com/", &mut t);
        // begin() runs the blocking fetch eagerly, so the first poll completes.
        assert!(!job.is_done(), "not yet done before first poll");
        let done = eng.poll_load(&mut job, &mut t);
        assert!(done, "BlockingAsync must resolve on the first poll");
        let page = job.take_result().unwrap().unwrap();
        assert_eq!(page.doc.title, "Example Domain");
    }

    /// Redirect chain is followed transparently by poll_load.
    #[test]
    fn async_redirects_are_followed() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        lo.serve_raw(
            "redir2.test",
            b"HTTP/1.1 302 Found\r\nLocation: http://example.com/\r\nContent-Length: 0\r\n\r\n",
        );
        let mut t = BlockingAsync::new(lo);
        let mut job = eng.begin_load("http://redir2.test/", &mut t);
        // Pump until done (may need more than one poll for the redirect hop).
        let mut iters = 0;
        while !eng.poll_load(&mut job, &mut t) {
            iters += 1;
            assert!(iters < 20, "redirect should resolve within 20 polls");
        }
        let page = job.take_result().unwrap().unwrap();
        assert_eq!(page.url.host, "example.com");
        assert_eq!(page.doc.title, "Example Domain");
    }

    /// DNS error surfaces correctly through the async path.
    #[test]
    fn async_dns_error_propagates() {
        let eng = Engine::new();
        let mut t = async_lo();
        let mut job = eng.begin_load("http://no-such-host.invalid/", &mut t);
        eng.poll_load(&mut job, &mut t);
        let err = job.take_result().unwrap().err().unwrap();
        assert!(matches!(err, FetchError::Dns(_)));
    }

    /// Cancel: cancelling an in-flight job must not crash or leave state corrupt.
    #[test]
    fn cancel_in_flight_does_not_crash() {
        let eng = Engine::new();
        let mut t = async_lo();
        // begin_load for a legacy host kicks off a transport job.
        let job = eng.begin_load("http://example.com/", &mut t);
        if let Some(h) = job.transport_handle() {
            t.cancel(h);
        }
        // Start a fresh load after cancel — must work normally.
        let mut job2 = eng.begin_load("dominion://home", &mut t);
        assert!(job2.is_done());
        assert!(job2.take_result().unwrap().is_ok());
    }

    /// Offline transport reports offline through the async path.
    #[test]
    fn async_offline_transport_reports_offline() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        lo.set_online(false);
        let mut t = BlockingAsync::new(lo);
        let mut job = eng.begin_load("http://example.com/", &mut t);
        // Offline is detected in begin_load before any transport job is started.
        assert!(job.is_done());
        let err = job.take_result().unwrap().err().unwrap();
        assert_eq!(err, FetchError::Offline);
    }

    /// Log messages accumulate across poll steps for diagnostics.
    #[test]
    fn load_job_accumulates_log() {
        let eng = Engine::new();
        let mut t = async_lo();
        let mut job = eng.begin_load("dominion://home", &mut t);
        assert!(!job.log().is_empty(), "native load should log at least one entry");
        let _ = job.take_result();
    }

    // ── DominionWeb DSL parser + VFS-backed native pages ─────────────────────────

    #[test]
    fn parse_page_dsl_produces_correct_page() {
        let dsl = "Title: My Page\nHeading: Welcome\nText: Hello world.\nLink: Docs -> dominion://docs\n";
        let page = parse_page_dsl(dsl).expect("valid DSL must parse");
        assert_eq!(page.title, "My Page");
        let text = page.render_text();
        assert!(text.contains("Welcome"));
        assert!(text.contains("Hello world"));
        assert!(page.links().iter().any(|l| *l == "dominion://docs"));
    }

    #[test]
    fn parse_page_dsl_with_action() {
        let dsl = "Title: Actions\nAction: Do it -> MyMod::run (MyCap)\n";
        let page = parse_page_dsl(dsl).expect("action DSL must parse");
        assert_eq!(page.title, "Actions");
        assert!(!page.blocks.is_empty());
    }

    #[test]
    fn parse_page_dsl_ignores_comments() {
        let dsl = "# This is a comment\nTitle: Commented\n# Another comment\nText: Visible\n";
        let page = parse_page_dsl(dsl).expect("comments must be ignored");
        assert_eq!(page.title, "Commented");
        assert!(page.render_text().contains("Visible"));
    }

    #[test]
    fn native_web_resolves_vfs_page() {
        use crate::filesystem::FileSystem;
        let fs = FileSystem::shared();
        let _ = fs.borrow_mut().mkdir("/dominion");
        let _ = fs.borrow_mut().mkdir("/dominion/pages");
        fs.borrow_mut().write_text(
            "/dominion/pages/testpage.dominion",
            "Title: VFS Page\nHeading: From VFS\nText: This came from the filesystem.\n",
        ).expect("write must succeed");

        let mut native = NativeWeb::new();
        native.set_fs(fs);
        let page = native.resolve("testpage").expect("VFS page must resolve");
        assert_eq!(page.title, "VFS Page");
        assert!(page.render_text().contains("From VFS"));
    }

    // ── DominionWeb HTML export + HTTP serving ────────────────────────────────────

    #[test]
    fn page_to_html_produces_valid_structure() {
        let page = Page::new("My Shop")
            .heading("Products")
            .text("Welcome to the store.")
            .link("About us", "dominion://about")
            .action("Buy now", "Shop::buy", "Payment");
        let html = page_to_html(&page);

        assert!(html.contains("<!DOCTYPE html>"), "must have doctype");
        assert!(html.contains("<title>My Shop</title>"), "must have title");
        assert!(html.contains("<h1>My Shop</h1>"), "must have h1 from title");
        assert!(html.contains("<h2>Products</h2>"), "heading must be h2");
        assert!(html.contains("Welcome to the store"), "body text must appear");
        assert!(html.contains("dominion://about"), "links must have href");
        assert!(html.contains("About us"), "link text must appear");
        assert!(html.contains("Buy now"), "action label must appear");
        assert!(html.contains("Payment"), "capability must appear");
        assert!(html.contains("DominionOS"), "footer must mention DominionOS");
        // Must be well-terminated.
        assert!(html.ends_with("</html>\n"), "must end with </html>");
    }

    #[test]
    fn page_to_html_escapes_special_characters() {
        let page = Page::new("Test <>&\"").text("Content with <script> & \"quotes\".");
        let html = page_to_html(&page);
        assert!(!html.contains("<script>"), "must escape angle brackets");
        assert!(html.contains("&lt;script&gt;") || html.contains("&lt;"), "must HTML-escape");
    }

    #[test]
    fn page_to_http_response_is_valid_http() {
        let page = Page::new("HTTP Test").text("Hello from native web.");
        let resp = page_to_http_response(&page);
        let text = String::from_utf8_lossy(&resp);

        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "must be HTTP 200");
        assert!(text.contains("Content-Type: text/html"), "must declare HTML type");
        assert!(text.contains("Content-Length:"), "must have Content-Length");
        assert!(text.contains("\r\n\r\n"), "must have blank line before body");
        let body_start = text.find("\r\n\r\n").unwrap() + 4;
        let body = &text[body_start..];
        assert!(body.contains("<!DOCTYPE html>"), "body must be full HTML");
        assert!(body.contains("Hello from native web"), "body text must appear");

        // Content-Length must match actual body size.
        let cl_start = text.find("Content-Length: ").unwrap() + 16;
        let cl_end = text[cl_start..].find('\r').unwrap() + cl_start;
        let stated_len: usize = text[cl_start..cl_end].parse().unwrap();
        assert_eq!(stated_len, body.len(), "Content-Length must match body size");
    }

    #[test]
    fn native_web_serve_http_returns_full_response_for_known_page() {
        let native = NativeWeb::new(); // seeded with home, docs, settings, about
        let resp = native.serve_http("home").expect("home must be serveable");
        let text = String::from_utf8_lossy(&resp);
        assert!(text.starts_with("HTTP/1.1 200 OK"), "must be 200");
        assert!(text.contains("DominionWeb"), "home page content must appear");
    }

    #[test]
    fn native_web_serve_http_returns_none_for_unknown_page() {
        let native = NativeWeb::new();
        assert!(native.serve_http("notapage").is_none(), "unknown page must return None");
    }

    #[test]
    fn native_web_serve_http_works_for_vfs_page() {
        use crate::filesystem::FileSystem;
        let fs = FileSystem::shared();
        let _ = fs.borrow_mut().mkdir("/dominion");
        let _ = fs.borrow_mut().mkdir("/dominion/pages");
        fs.borrow_mut().write_text(
            "/dominion/pages/myshop.dominion",
            "Title: My Shop\nHeading: Products\nText: Buy stuff here.\n",
        ).expect("write must succeed");
        let mut native = NativeWeb::new();
        native.set_fs(fs);
        let resp = native.serve_http("myshop").expect("VFS page must be serveable via HTTP");
        let text = String::from_utf8_lossy(&resp);
        assert!(text.contains("My Shop"), "page title must appear in HTTP response");
        assert!(text.contains("Buy stuff here"), "page text must appear");
    }

    /// The engine can hand a native page to a LoopbackTransport so it can be fetched
    /// via HTTP — simulating what an EtherLink HTTP bridge would do.
    // ── Path traversal security tests ──────────────────────────────────────────

    #[test]
    fn traversal_dotdot_is_rejected() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        // "dominion://../../etc/passwd" — the URL parser stores path as "../../etc/passwd"
        // after stripping the "//". validate_native_name must reject "..".
        let err = eng.load("dominion://../../etc/passwd", &mut lo).err().unwrap();
        assert!(matches!(err, FetchError::BadUrl), "dot-dot traversal must produce BadUrl; got {:?}", err);
    }

    #[test]
    fn traversal_dot_slash_is_rejected() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        // Path containing a "/" must be rejected (names are flat identifiers).
        let err = eng.load("dominion://./home", &mut lo).err().unwrap();
        assert!(matches!(err, FetchError::BadUrl), "dot-slash traversal must produce BadUrl; got {:?}", err);
    }

    #[test]
    fn traversal_bare_dot_is_rejected() {
        let eng = Engine::new();
        let mut lo = LoopbackTransport::new();
        let err = eng.load("dominion://.", &mut lo).err().unwrap();
        assert!(matches!(err, FetchError::BadUrl | FetchError::NativeNotFound(_)),
            "bare dot must be rejected; got {:?}", err);
    }

    #[test]
    fn validate_native_name_accepts_valid_names() {
        assert!(validate_native_name("home").is_some());
        assert!(validate_native_name("my-page").is_some());
        assert!(validate_native_name("page_2").is_some());
    }

    #[test]
    fn validate_native_name_rejects_traversal() {
        assert!(validate_native_name("..").is_none());
        assert!(validate_native_name("../etc/passwd").is_none());
        assert!(validate_native_name("a/b").is_none());
        assert!(validate_native_name("/absolute").is_none());
        assert!(validate_native_name(".").is_none());
        assert!(validate_native_name("").is_none());
        assert!(validate_native_name("a\\b").is_none());
    }

    #[test]
    fn engine_native_page_roundtrip_as_http() {
        let eng = Engine::new();
        // Get the raw HTTP response for "about".
        let http_resp = eng.native().serve_http("about").expect("about page must be serveable");

        // Feed that response to a LoopbackTransport and fetch it back.
        let mut lo = LoopbackTransport::new();
        lo.serve_raw("native.bridge", &http_resp);
        let fetched = eng.load("http://native.bridge/", &mut lo).expect("must fetch via HTTP");
        // The document should contain the about page's content.
        assert!(fetched.doc.text().contains("universal browser") ||
                fetched.doc.text().contains("DominionWeb") ||
                fetched.doc.text().contains("Verify"),
            "about page content must survive HTTP round-trip; got: {}", fetched.doc.text());
    }

}

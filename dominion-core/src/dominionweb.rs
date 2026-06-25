//! Dominion-native web — roadmap feature 6 (gate: networking + compositor).
//!
//! A native site is not HTML+JS; it is a graph of **declarative semantic page
//! objects** addressed by identity+hash and fetched over DominionLink (integration
//! strategy §8). A page is content — headings, text, links, and capability-gated
//! actions — with *no ambient script*. The OS renders the same page object into
//! whatever view the context calls for (here, a terminal view; a graphical view
//! composes through [`surface`](crate::surface)). Philosophically: Gemini's
//! simplicity × IPFS content-addressing × the capability model.
//!
//! Pure, safe, host-tested. Page, file, and network object are the *same*
//! verifiable, cacheable graph entry.

use crate::hash::Hash256;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One declarative element of a page.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Block {
    Heading(String),
    Text(String),
    /// A hyperlink to another page (by DominionLink address or name).
    Link { text: String, target: String },
    /// A capability-gated interactive action — the *only* form of interactivity,
    /// naming the cell that runs and the capability it requires. No ambient JS.
    Action { label: String, cell: String, requires: String },
}

/// A declarative, content-addressed page.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Page {
    pub title: String,
    pub blocks: Vec<Block>,
}

impl Page {
    pub fn new(title: impl Into<String>) -> Page {
        Page { title: title.into(), blocks: Vec::new() }
    }

    pub fn heading(mut self, text: impl Into<String>) -> Page {
        self.blocks.push(Block::Heading(text.into()));
        self
    }
    pub fn text(mut self, text: impl Into<String>) -> Page {
        self.blocks.push(Block::Text(text.into()));
        self
    }
    pub fn link(mut self, text: impl Into<String>, target: impl Into<String>) -> Page {
        self.blocks.push(Block::Link { text: text.into(), target: target.into() });
        self
    }
    pub fn action(mut self, label: impl Into<String>, cell: impl Into<String>, requires: impl Into<String>) -> Page {
        self.blocks.push(Block::Action { label: label.into(), cell: cell.into(), requires: requires.into() });
        self
    }

    /// Canonical byte encoding for content addressing.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"page1");
        out.extend_from_slice(&(self.title.len() as u32).to_le_bytes());
        out.extend_from_slice(self.title.as_bytes());
        out.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for b in &self.blocks {
            match b {
                Block::Heading(t) => {
                    out.push(b'h');
                    push_str(&mut out, t);
                }
                Block::Text(t) => {
                    out.push(b't');
                    push_str(&mut out, t);
                }
                Block::Link { text, target } => {
                    out.push(b'l');
                    push_str(&mut out, text);
                    push_str(&mut out, target);
                }
                Block::Action { label, cell, requires } => {
                    out.push(b'a');
                    push_str(&mut out, label);
                    push_str(&mut out, cell);
                    push_str(&mut out, requires);
                }
            }
        }
        out
    }

    /// The page's DominionLink content address.
    pub fn content_id(&self) -> Hash256 {
        Hash256::of(&self.encode())
    }

    /// Every link target on the page (for crawling / prefetch).
    pub fn links(&self) -> Vec<&str> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Link { target, .. } => Some(target.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Render the page to a terminal view (Stage 9's "dynamically invoke the
    /// rendering capability for the context").
    pub fn render_text(&self) -> String {
        let mut s = String::new();
        s.push_str("== ");
        s.push_str(&self.title);
        s.push_str(" ==\n");
        for b in &self.blocks {
            match b {
                Block::Heading(t) => {
                    s.push_str("# ");
                    s.push_str(t);
                    s.push('\n');
                }
                Block::Text(t) => {
                    s.push_str(t);
                    s.push('\n');
                }
                Block::Link { text, target } => {
                    s.push('[');
                    s.push_str(text);
                    s.push_str(" -> ");
                    s.push_str(target);
                    s.push_str("]\n");
                }
                Block::Action { label, requires, .. } => {
                    s.push_str("<button ");
                    s.push_str(label);
                    s.push_str(" (needs ");
                    s.push_str(requires);
                    s.push_str(")>\n");
                }
            }
        }
        s
    }
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A site: a collection of pages addressable by content id (a content-addressed
/// graph, exactly like the object store and the network).
#[derive(Default)]
pub struct Site {
    pages: alloc::collections::BTreeMap<Hash256, Page>,
}

impl Site {
    pub fn new() -> Site {
        Site { pages: alloc::collections::BTreeMap::new() }
    }
    /// Publish a page, returning its content address.
    pub fn publish(&mut self, page: Page) -> Hash256 {
        let id = page.content_id();
        self.pages.insert(id, page);
        id
    }
    pub fn fetch(&self, id: Hash256) -> Option<&Page> {
        self.pages.get(&id)
    }
    pub fn len(&self) -> usize {
        self.pages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

// ── NDN Interest / Data wire format ──────────────────────────────────────────
//
// A minimal Named-Data Networking wire format for DominionWeb pages over EtherLink.
// Two packet types:
//   Interest — "I want the content at this address"
//   Data     — "Here is the verified content for that address"
//
// Wire layout (no external encoding deps; all big-endian u32 lengths):
//   Interest: MAGIC_INT (2) + name (32) + nonce (4) + hop_limit (1) = 39 bytes
//   Data:     MAGIC_DAT (2) + name (32) + payload_len (4) + payload (N) bytes

const MAGIC_INT: [u8; 2] = [0xAE, 0x49]; // 'I'
const MAGIC_DAT: [u8; 2] = [0xAE, 0x44]; // 'D'

/// A named interest packet: "please send me the content at `name`."
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Interest {
    /// Content address being requested.
    pub name: Hash256,
    /// Random nonce for loop suppression / dedup.
    pub nonce: u32,
    /// Hop limit (TTL); decremented at each forwarding node.
    pub hop_limit: u8,
}

impl Interest {
    pub fn new(name: Hash256, nonce: u32) -> Interest {
        Interest { name, nonce, hop_limit: 64 }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(39);
        out.extend_from_slice(&MAGIC_INT);
        out.extend_from_slice(&self.name.0);
        out.extend_from_slice(&self.nonce.to_be_bytes());
        out.push(self.hop_limit);
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Interest> {
        if bytes.get(0..2)? != MAGIC_INT { return None; }
        let arr: [u8; 32] = bytes.get(2..34)?.try_into().ok()?;
        let name = Hash256(arr);
        let nonce_bytes: [u8; 4] = bytes.get(34..38)?.try_into().ok()?;
        let nonce = u32::from_be_bytes(nonce_bytes);
        let hop_limit = *bytes.get(38)?;
        Some(Interest { name, nonce, hop_limit })
    }
}

/// A data packet: the verified content that satisfies an Interest.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Data {
    /// The content address (= SHA-256 of payload — self-certifying).
    pub name: Hash256,
    /// Raw payload bytes (the output of [`Page::encode`]).
    pub payload: Vec<u8>,
}

impl Data {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(38 + self.payload.len());
        out.extend_from_slice(&MAGIC_DAT);
        out.extend_from_slice(&self.name.0);
        let len = self.payload.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Data> {
        if bytes.get(0..2)? != MAGIC_DAT { return None; }
        let arr: [u8; 32] = bytes.get(2..34)?.try_into().ok()?;
        let name = Hash256(arr);
        let len_bytes: [u8; 4] = bytes.get(34..38)?.try_into().ok()?;
        let len = u32::from_be_bytes(len_bytes) as usize;
        let payload = bytes.get(38..38 + len)?.to_vec();
        // Verify content integrity before returning.
        if Hash256::of(&payload) != name { return None; }
        Some(Data { name, payload })
    }

    /// Verify and decode the payload as a Page.
    pub fn into_page(self) -> Option<Page> {
        decode_dominionweb_page(&self.payload)
    }
}

/// Publish a page as a Data packet ready to transmit over EtherLink.
pub fn page_to_data(page: &Page) -> Data {
    let payload = page.encode();
    let name = Hash256::of(&payload);
    Data { name, payload }
}

/// Given raw Interest bytes and a Site, return raw Data bytes if the page is known.
/// This is the responder side: a peer calls this for every incoming Interest packet.
pub fn serve_interest(interest_bytes: &[u8], site: &Site) -> Option<Vec<u8>> {
    let interest = Interest::decode(interest_bytes)?;
    if interest.hop_limit == 0 { return None; }
    let page = site.fetch(interest.name)?;
    Some(page_to_data(page).encode())
}

/// Decode raw Data bytes back into a verified Page.
/// This is the requester side: call with bytes received from the network.
pub fn receive_data(data_bytes: &[u8]) -> Option<Page> {
    Data::decode(data_bytes)?.into_page()
}

fn decode_dominionweb_page(bytes: &[u8]) -> Option<Page> {
    // Identical to Page::encode layout (tag "page1" + u32-prefixed fields).
    if bytes.get(0..5)? != b"page1" { return None; }
    let mut p = 5usize;
    let title = read_str_le(&bytes, &mut p)?;
    let mut page = Page::new(title);
    let block_count = read_u32_le(&bytes, &mut p)? as usize;
    for _ in 0..block_count {
        let kind = *bytes.get(p)?; p += 1;
        match kind {
            b'h' => { let t = read_str_le(&bytes, &mut p)?; page = page.heading(t); }
            b't' => { let t = read_str_le(&bytes, &mut p)?; page = page.text(t); }
            b'l' => {
                let text   = read_str_le(&bytes, &mut p)?;
                let target = read_str_le(&bytes, &mut p)?;
                page = page.link(text, target);
            }
            b'a' => {
                let label    = read_str_le(&bytes, &mut p)?;
                let cell     = read_str_le(&bytes, &mut p)?;
                let requires = read_str_le(&bytes, &mut p)?;
                page = page.action(label, cell, requires);
            }
            _ => return None,
        }
    }
    Some(page)
}

fn read_u32_le(bytes: &[u8], p: &mut usize) -> Option<u32> {
    let b = bytes.get(*p..*p + 4)?;
    *p += 4;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_str_le(bytes: &[u8], p: &mut usize) -> Option<String> {
    let len = read_u32_le(bytes, p)? as usize;
    let b = bytes.get(*p..*p + len)?;
    *p += len;
    core::str::from_utf8(b).ok().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> Page {
        Page::new("DominionOS Home")
            .heading("Welcome")
            .text("A native, capability-secured web.")
            .link("About", "dominion://about")
            .action("Subscribe", "Mailer::subscribe", "NetConnect")
    }

    #[test]
    fn content_addressing_is_stable_and_sensitive() {
        let a = home().content_id();
        let b = home().content_id();
        assert_eq!(a, b);
        let c = home().text("extra").content_id();
        assert_ne!(a, c);
    }

    #[test]
    fn render_text_includes_structure() {
        let r = home().render_text();
        assert!(r.contains("DominionOS Home"));
        assert!(r.contains("# Welcome"));
        assert!(r.contains("About -> dominion://about"));
        assert!(r.contains("needs NetConnect"));
    }

    #[test]
    fn links_are_extracted() {
        assert_eq!(home().links(), ["dominion://about"]);
    }

    #[test]
    fn site_publishes_and_fetches_by_address() {
        let mut site = Site::new();
        let id = site.publish(home());
        assert_eq!(site.fetch(id).unwrap().title, "DominionOS Home");
        assert_eq!(site.len(), 1);
        // Fetching the same page content dedups to one entry.
        site.publish(home());
        assert_eq!(site.len(), 1);
    }

    // ── NDN Interest / Data tests ──────────────────────────────────────────────

    #[test]
    fn interest_roundtrips_through_wire_encoding() {
        let name = Hash256::of(b"example content");
        let orig = Interest::new(name, 0xDEAD_BEEF);
        let wire = orig.encode();
        assert_eq!(wire.len(), 39);
        let decoded = Interest::decode(&wire).expect("must decode");
        assert_eq!(decoded.name, orig.name);
        assert_eq!(decoded.nonce, orig.nonce);
        assert_eq!(decoded.hop_limit, 64);
    }

    #[test]
    fn data_roundtrips_and_verifies_payload() {
        let page = home();
        let data = page_to_data(&page);
        let wire = data.encode();
        // Wire must start with the Data magic.
        assert_eq!(&wire[0..2], &MAGIC_DAT);
        let decoded = Data::decode(&wire).expect("must decode");
        assert_eq!(decoded.name, data.name);
        assert_eq!(decoded.payload, data.payload);
        let recovered = decoded.into_page().expect("page must decode");
        assert_eq!(recovered.title, page.title);
        assert_eq!(recovered.blocks, page.blocks);
    }

    #[test]
    fn data_decode_rejects_tampered_payload() {
        let data = page_to_data(&home());
        let mut wire = data.encode();
        // Flip a byte in the payload section.
        let last = wire.len() - 1;
        wire[last] ^= 0xFF;
        assert!(Data::decode(&wire).is_none(), "tampered data must be rejected");
    }

    #[test]
    fn interest_with_zero_hop_limit_is_not_served() {
        let page = home();
        let mut site = Site::new();
        site.publish(page.clone());
        let data = page_to_data(&page);
        let mut interest = Interest::new(data.name, 1);
        interest.hop_limit = 0; // expired Interest — must not be served
        let wire = interest.encode();
        assert!(serve_interest(&wire, &site).is_none(), "expired Interest must not be served");
    }

    #[test]
    fn serve_interest_returns_data_for_known_page() {
        let page = home();
        let mut site = Site::new();
        site.publish(page.clone());
        let data = page_to_data(&page);

        // Build and wire-encode the Interest.
        let interest = Interest::new(data.name, 42);
        let interest_wire = interest.encode();

        // Serve it — the site must respond with Data.
        let response = serve_interest(&interest_wire, &site).expect("must respond");
        let recovered = receive_data(&response).expect("must decode page");
        assert_eq!(recovered.title, page.title);
        assert_eq!(recovered.blocks.len(), page.blocks.len());
    }

    #[test]
    fn serve_interest_returns_none_for_unknown_cid() {
        let site = Site::new(); // empty
        let interest = Interest::new(Hash256::of(b"nobody published this"), 99);
        let wire = interest.encode();
        assert!(serve_interest(&wire, &site).is_none(), "unknown CID must yield None");
    }

    #[test]
    fn full_etherlink_page_exchange_cycle() {
        // Publisher side: build a site and encode pages as Data packets.
        let mut site = Site::new();
        let page_a = Page::new("DominionShop")
            .heading("Products")
            .text("Browse our catalog.")
            .link("Widget Alpha", "dominion://widget-alpha");
        let data_a = page_to_data(&page_a);
        site.publish(page_a.clone());

        let page_b = Page::new("Widget Alpha")
            .heading("Widget Alpha")
            .text("Price: $9.99")
            .link("Back", "dominion://shop");
        site.publish(page_b.clone());

        // Requester side: send Interest for page A, verify response, follow link to B.
        let interest_a = Interest::new(data_a.name, 1);
        let wire_a = interest_a.encode();
        let resp_a = serve_interest(&wire_a, &site).expect("site must respond to interest A");
        let got_a = receive_data(&resp_a).expect("must decode page A");
        assert_eq!(got_a.title, "DominionShop");

        // Follow the first link — get the CID for page B and send a second Interest.
        let data_b = page_to_data(&page_b);
        let interest_b = Interest::new(data_b.name, 2);
        let wire_b = interest_b.encode();
        let resp_b = serve_interest(&wire_b, &site).expect("site must respond to interest B");
        let got_b = receive_data(&resp_b).expect("must decode page B");
        assert_eq!(got_b.title, "Widget Alpha");
        assert!(got_b.blocks.iter().any(|b| matches!(b, Block::Text(t) if t.contains("9.99"))));
    }
}

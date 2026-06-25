//! DominionLink — the OS's *native* network model (roadmap feature 5; SRS Stage 7,
//! Identity-Based / Named-Data Networking).
//!
//! Legacy TCP/IP secures the *channel* between two host addresses. DominionLink
//! secures the *data*: an address is a self-certifying **identity** (the hash of a
//! public key, like a Tor onion address or an IPFS PeerID) plus a **content hash**.
//! Responses are content-addressed objects, so any consumer can verify a reply
//! against its address without trusting whoever delivered it — caching, dedup, and
//! integrity fall out for free. Resolution uses a Kademlia-style DHT (XOR metric),
//! and a DNS bridge maps legacy names to native identities so the two internets
//! interoperate.
//!
//! This is pure, safe, host-tested logic. The kernel runs it over the §1 legacy
//! stack as an overlay (the `unsafe` transport is `dominion-kernel`'s job).

use crate::capability::{Capability, Rights};
use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A self-certifying network identity: the hash of a public key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct DominionId(pub Hash256);

impl DominionId {
    /// Derive the identity from a public key. Because the identity *is* the hash,
    /// nobody can claim an identity without the matching key material.
    pub fn from_pubkey(pubkey: &[u8]) -> DominionId {
        DominionId(Hash256::of(pubkey))
    }

    /// Verify that `pubkey` actually backs this identity (self-certification).
    pub fn certifies(&self, pubkey: &[u8]) -> bool {
        Hash256::of(pubkey) == self.0
    }

    /// XOR distance to another identity (the Kademlia metric), as 32 raw bytes
    /// interpreted big-endian.
    pub fn distance(&self, other: &DominionId) -> [u8; 32] {
        let mut d = [0u8; 32];
        for (slot, (a, b)) in d.iter_mut().zip(self.0 .0.iter().zip(other.0 .0.iter())) {
            *slot = a ^ b;
        }
        d
    }

    pub fn short(&self) -> alloc::string::String {
        self.0.short()
    }
}

/// An endpoint is a *capability*, not an open port: holding it authorises talking
/// to the identity, and its rights bound what you may do.
#[derive(Clone, Copy, Debug)]
pub struct Endpoint {
    pub id: DominionId,
    pub capability: Capability,
}

impl Endpoint {
    /// May this endpoint be used for a request needing `rights`? (A torn or
    /// under-privileged capability is refused, mirroring the hardware check.)
    pub fn authorises(&self, rights: Rights) -> bool {
        self.capability.is_valid() && self.capability.rights().contains(rights)
    }
}

/// A node serving and requesting content-addressed objects over DominionLink.
pub struct DominionLink {
    pub me: DominionId,
    store: BTreeMap<Hash256, Vec<u8>>,
}

impl DominionLink {
    pub fn new(me: DominionId) -> DominionLink {
        DominionLink { me, store: BTreeMap::new() }
    }

    /// Publish content, returning its address (content id).
    pub fn publish(&mut self, data: &[u8]) -> Hash256 {
        let cid = Hash256::of(data);
        self.store.entry(cid).or_insert_with(|| data.to_vec());
        cid
    }

    /// Fetch content by address. The bytes are verified against the address before
    /// being returned — a corrupted store entry is rejected, not served.
    pub fn fetch(&self, cid: Hash256) -> Option<&[u8]> {
        let data = self.store.get(&cid)?;
        if Self::verify(cid, data) {
            Some(data)
        } else {
            None
        }
    }

    /// Anyone can independently verify received bytes against an address — the
    /// heart of self-certifying networking.
    pub fn verify(cid: Hash256, data: &[u8]) -> bool {
        Hash256::of(data) == cid
    }

    pub fn stored(&self) -> usize {
        self.store.len()
    }
}

/// A Kademlia-style distributed hash table over Dominion identities.
pub struct Dht {
    me: DominionId,
    nodes: Vec<DominionId>,
}

impl Dht {
    pub fn new(me: DominionId) -> Dht {
        Dht { me, nodes: Vec::new() }
    }

    pub fn insert(&mut self, id: DominionId) {
        if id != self.me && !self.nodes.contains(&id) {
            self.nodes.push(id);
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The `k` known nodes closest to `target` by XOR distance — the core DHT
    /// lookup step.
    pub fn closest(&self, target: &DominionId, k: usize) -> Vec<DominionId> {
        let mut scored: Vec<(DominionId, [u8; 32])> =
            self.nodes.iter().map(|n| (*n, n.distance(target))).collect();
        scored.sort_by_key(|x| x.1);
        scored.into_iter().take(k).map(|(n, _)| n).collect()
    }
}

/// Bridges legacy DNS names to native Dominion identities, so the legacy and native
/// internets can name each other.
#[derive(Default)]
pub struct DnsBridge {
    forward: BTreeMap<String, DominionId>,
}

impl DnsBridge {
    pub fn new() -> DnsBridge {
        DnsBridge { forward: BTreeMap::new() }
    }

    pub fn register(&mut self, name: impl Into<String>, id: DominionId) {
        self.forward.insert(name.into(), id);
    }

    pub fn resolve(&self, name: &str) -> Option<DominionId> {
        self.forward.get(name).copied()
    }

    /// Reverse-lookup the legacy name(s) bound to an identity.
    pub fn name_of(&self, id: DominionId) -> Option<&str> {
        self.forward.iter().find(|(_, v)| **v == id).map(|(k, _)| k.as_str())
    }

    pub fn len(&self) -> usize {
        self.forward.len()
    }

    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_self_certifying() {
        let id = DominionId::from_pubkey(b"jayden-public-key");
        assert!(id.certifies(b"jayden-public-key"));
        assert!(!id.certifies(b"someone-elses-key"));
    }

    #[test]
    fn fetch_verifies_content_against_address() {
        let mut link = DominionLink::new(DominionId::from_pubkey(b"node"));
        let cid = link.publish(b"hello dominion");
        assert_eq!(link.fetch(cid).unwrap(), b"hello dominion");
        // An address that nothing matches yields nothing.
        assert!(link.fetch(Hash256::of(b"unknown")).is_none());
    }

    #[test]
    fn anyone_can_verify_received_bytes() {
        let cid = Hash256::of(b"signed payload");
        assert!(DominionLink::verify(cid, b"signed payload"));
        assert!(!DominionLink::verify(cid, b"tampered payload"));
    }

    #[test]
    fn endpoint_is_a_capability() {
        let id = DominionId::from_pubkey(b"peer");
        let cap = Capability::mint(0, 0x1000, Rights::READ);
        let ep = Endpoint { id, capability: cap };
        assert!(ep.authorises(Rights::READ));
        assert!(!ep.authorises(Rights::WRITE));
        // A tampered capability authorises nothing.
        let bad = Endpoint { id, capability: cap.tamper() };
        assert!(!bad.authorises(Rights::READ));
    }

    #[test]
    fn xor_distance_is_zero_to_self_and_symmetric() {
        let a = DominionId::from_pubkey(b"a");
        let b = DominionId::from_pubkey(b"b");
        assert_eq!(a.distance(&a), [0u8; 32]);
        assert_eq!(a.distance(&b), b.distance(&a));
    }

    #[test]
    fn dht_returns_closest_nodes_by_xor() {
        let me = DominionId::from_pubkey(b"me");
        let mut dht = Dht::new(me);
        let target = DominionId::from_pubkey(b"target");
        // Insert several nodes including the target itself.
        for k in 0..16u8 {
            dht.insert(DominionId::from_pubkey(&[k]));
        }
        dht.insert(target);
        let closest = dht.closest(&target, 3);
        assert_eq!(closest.len(), 3);
        // The nearest node to `target` must be `target` (distance 0).
        assert_eq!(closest[0], target);
        // Results are sorted by increasing distance.
        let d0 = closest[0].distance(&target);
        let d1 = closest[1].distance(&target);
        assert!(d0 <= d1);
    }

    #[test]
    fn dns_bridge_maps_legacy_names() {
        let mut dns = DnsBridge::new();
        let id = DominionId::from_pubkey(b"example.com-key");
        dns.register("example.com", id);
        assert_eq!(dns.resolve("example.com"), Some(id));
        assert_eq!(dns.resolve("unknown.com"), None);
        assert_eq!(dns.name_of(id), Some("example.com"));
    }
}

//! Named-Data Networking — the full Interest/Data forwarding plane (SRS Stage 7;
//! see `docs/architecture/networking-and-dominionlink.md`).
//!
//! [`crate::dominionlink`] gives self-certifying identities and content-addressed
//! fetch. This module adds the **NDN forwarding plane** on top: hierarchical
//! **Names**, **Interest** and **Data** packets, and the three tables that make
//! NDN work:
//!
//! * **Content Store (CS)** — caches Data by name, so a repeated request is served
//!   locally with no upstream traffic.
//! * **Pending Interest Table (PIT)** — records who asked for what; a second
//!   request for an in-flight name is **aggregated** (not re-forwarded), and Data
//!   flows back down the reverse path to *everyone* who asked.
//! * **FIB** — longest-prefix routing of names to next-hop faces.
//!
//! Data is self-verifying (its name is bound to a content digest), so any cache may
//! serve it and the consumer still checks integrity. Pure, safe `no_std`,
//! host-tested.

use crate::crypto::CryptoLayer;
use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A hierarchical content name, e.g. `/dominion/docs/readme`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Name {
    components: Vec<String>,
}

impl Name {
    /// Parse a `/`-separated name. Empty components are ignored.
    pub fn parse(s: &str) -> Name {
        Name {
            components: s.split('/').filter(|c| !c.is_empty()).map(String::from).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.components.len()
    }
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Does this name fall under `prefix`?
    pub fn has_prefix(&self, prefix: &Name) -> bool {
        prefix.components.len() <= self.components.len()
            && self.components[..prefix.components.len()] == prefix.components[..]
    }

    /// **HIBC** (Hash-of-Identity-Based Content) name: the authority component
    /// *is* the hash of the producer's public key. There is no pre-agreed trust
    /// anchor — the name itself certifies who may sign Data under it. The leading
    /// component is `H(pubkey)` in hex; `suffix` names the content beneath it.
    pub fn hibc(pubkey: &[u8], suffix: &str) -> Name {
        let mut components = alloc::vec![Hash256::of(pubkey).to_hex()];
        components.extend(suffix.split('/').filter(|c| !c.is_empty()).map(String::from));
        Name { components }
    }

    /// The authority (first) component, if any — for HIBC names this is `H(pubkey)`.
    pub fn authority(&self) -> Option<&str> {
        self.components.first().map(|s| s.as_str())
    }

    /// Does this name's authority component bind to `pubkey`? (Self-certifying: the
    /// authority equals `H(pubkey)`.)
    pub fn certifies(&self, pubkey: &[u8]) -> bool {
        self.authority() == Some(Hash256::of(pubkey).to_hex().as_str())
    }

    /// Canonical bytes of the name, for signing.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for c in &self.components {
            out.extend_from_slice(c.as_bytes());
            out.push(b'/');
        }
        out
    }
}

/// A face is a logical link/port id towards a neighbour or a local app.
pub type Face = u64;

/// A Data packet: a name, its payload, and a digest binding the two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Data {
    pub name: Name,
    pub content: Vec<u8>,
    pub digest: Hash256,
}

impl Data {
    pub fn new(name: Name, content: &[u8]) -> Data {
        Data { name, content: content.to_vec(), digest: Hash256::of(content) }
    }

    /// Self-verification: the payload still matches the bound digest.
    pub fn verify(&self) -> bool {
        Hash256::of(&self.content) == self.digest
    }
}

/// A **producer-signed** Data packet for **HIBC** namespaces (Stage 13 wiring).
///
/// The packet carries the producer's public key and a post-quantum signature
/// (via the [`CryptoLayer`]) over the name + content digest. A consumer accepts it
/// only if: the payload matches its digest, the **name's authority binds to the
/// producer key** (`name.certifies(pubkey)`), and the signature verifies. No
/// pre-shared trust anchor is needed — authority flows from the name itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedData {
    pub data: Data,
    pub algo: String,
    pub producer_pk: Vec<u8>,
    pub signature: Vec<u8>,
}

impl SignedData {
    /// What gets signed: the canonical name bytes followed by the content digest.
    fn signing_input(data: &Data) -> Vec<u8> {
        let mut m = data.name.encode();
        m.extend_from_slice(&data.digest.0);
        m
    }

    /// Produce a signed Data packet under a HIBC name derived from `producer_seed`.
    /// The producer keypair is generated through the CAL so the algorithm is
    /// agile/post-quantum.
    pub fn produce(
        cal: &CryptoLayer,
        algo: &str,
        producer_seed: &[u8],
        suffix: &str,
        content: &[u8],
    ) -> Option<SignedData> {
        let (sk, pk) = cal.keygen(algo, producer_seed)?;
        let name = Name::hibc(&pk, suffix);
        let data = Data::new(name, content);
        let sig = cal.sign(algo, &sk, &Self::signing_input(&data))?;
        Some(SignedData { data, algo: String::from(algo), producer_pk: pk, signature: sig })
    }

    /// Full verification: digest binding + HIBC name authority + PQ signature.
    pub fn verify(&self, cal: &CryptoLayer) -> bool {
        self.data.verify()
            && self.data.name.certifies(&self.producer_pk)
            && cal.verify(&self.algo, &self.producer_pk, &Self::signing_input(&self.data), &self.signature)
    }
}

/// What the forwarder decided to do with an Interest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterestOutcome {
    /// Cache hit — serve this Data back down the requesting face immediately.
    FromCache(Data),
    /// Forward upstream out of these faces (first time this name was seen).
    Forward(Vec<Face>),
    /// An identical Interest is already pending — aggregated, not re-forwarded.
    Aggregated,
    /// No route and nothing cached — drop.
    Drop,
}

/// The NDN forwarder: CS + PIT + FIB.
#[derive(Default)]
pub struct Forwarder {
    cs: BTreeMap<Name, Data>,
    /// name → set of downstream faces awaiting Data.
    pit: BTreeMap<Name, Vec<Face>>,
    /// (prefix, face) routing entries; longest match wins.
    fib: Vec<(Name, Face)>,
}

impl Forwarder {
    pub fn new() -> Forwarder {
        Forwarder { cs: BTreeMap::new(), pit: BTreeMap::new(), fib: Vec::new() }
    }

    /// Install a route: names under `prefix` forward out `face`.
    pub fn register_route(&mut self, prefix: Name, face: Face) {
        self.fib.push((prefix, face));
    }

    /// Longest-prefix-matching next-hops for a name.
    fn next_hops(&self, name: &Name) -> Vec<Face> {
        let mut best_len = 0;
        let mut faces = Vec::new();
        for (prefix, face) in &self.fib {
            if name.has_prefix(prefix) {
                let l = prefix.len();
                if l > best_len {
                    best_len = l;
                    faces = alloc::vec![*face];
                } else if l == best_len && !faces.contains(face) {
                    faces.push(*face);
                }
            }
        }
        faces
    }

    /// Process an incoming Interest for `name` arriving on `in_face`.
    pub fn recv_interest(&mut self, in_face: Face, name: &Name) -> InterestOutcome {
        // 1. Content Store.
        if let Some(data) = self.cs.get(name) {
            return InterestOutcome::FromCache(data.clone());
        }
        // 2. PIT aggregation.
        if let Some(faces) = self.pit.get_mut(name) {
            if !faces.contains(&in_face) {
                faces.push(in_face);
            }
            return InterestOutcome::Aggregated;
        }
        // 3. FIB forward.
        let hops = self.next_hops(name);
        if hops.is_empty() {
            return InterestOutcome::Drop;
        }
        self.pit.insert(name.clone(), alloc::vec![in_face]);
        InterestOutcome::Forward(hops)
    }

    /// Process incoming Data: cache it and return the downstream faces to deliver
    /// it to (the reverse path recorded in the PIT). Unsolicited Data (no PIT
    /// entry) is dropped — returns an empty list.
    pub fn recv_data(&mut self, data: Data) -> Vec<Face> {
        if !data.verify() {
            return Vec::new(); // corrupt Data is never cached or forwarded
        }
        let faces = self.pit.remove(&data.name).unwrap_or_default();
        if !faces.is_empty() {
            self.cs.insert(data.name.clone(), data);
        }
        faces
    }

    pub fn cache_len(&self) -> usize {
        self.cs.len()
    }
    pub fn pending(&self, name: &Name) -> usize {
        self.pit.get(name).map(|f| f.len()).unwrap_or(0)
    }

    /// **Push path** (the reactive-plane complement to pull). A producer announces
    /// that a new version exists under `prefix`. This returns the de-duplicated set
    /// of downstream faces with a pending Interest whose name falls under `prefix`
    /// — i.e. the notify rides the **same PIT reverse-path fan-out** as Interest/Data,
    /// so one announce wakes every subscriber under the prefix without N point-to-point
    /// sessions. The notification itself is tiny (a name + a version hash); subscribers
    /// then *fetch* the object by hash and hit the nearest cache.
    ///
    /// This is read-only over the PIT: it does not consume the entries (a long-lived
    /// subscription keeps its Interest standing). See `docs/architecture/reactive-subscription-plane.md`.
    pub fn notify(&self, prefix: &Name) -> Vec<Face> {
        let mut faces = Vec::new();
        for (name, pit_faces) in &self.pit {
            if name.has_prefix(prefix) {
                for f in pit_faces {
                    if !faces.contains(f) {
                        faces.push(*f);
                    }
                }
            }
        }
        faces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_prefix_matching() {
        let n = Name::parse("/dominion/docs/readme");
        assert!(n.has_prefix(&Name::parse("/dominion")));
        assert!(n.has_prefix(&Name::parse("/dominion/docs")));
        assert!(!n.has_prefix(&Name::parse("/dominion/code")));
        assert_eq!(n.len(), 3);
    }

    #[test]
    fn interest_with_no_route_is_dropped() {
        let mut fw = Forwarder::new();
        let outcome = fw.recv_interest(1, &Name::parse("/unknown"));
        assert_eq!(outcome, InterestOutcome::Drop);
    }

    #[test]
    fn first_interest_forwards_via_longest_prefix() {
        let mut fw = Forwarder::new();
        fw.register_route(Name::parse("/dominion"), 10);
        fw.register_route(Name::parse("/dominion/docs"), 20); // more specific
        let outcome = fw.recv_interest(1, &Name::parse("/dominion/docs/readme"));
        // Longest-prefix match selects face 20, not 10.
        assert_eq!(outcome, InterestOutcome::Forward(alloc::vec![20]));
        assert_eq!(fw.pending(&Name::parse("/dominion/docs/readme")), 1);
    }

    #[test]
    fn second_interest_is_aggregated_not_reforwarded() {
        let mut fw = Forwarder::new();
        fw.register_route(Name::parse("/v"), 9);
        let name = Name::parse("/v/clip");
        assert!(matches!(fw.recv_interest(1, &name), InterestOutcome::Forward(_)));
        // A different consumer asks for the same in-flight name.
        assert_eq!(fw.recv_interest(2, &name), InterestOutcome::Aggregated);
        // Both faces are now pending on the single upstream Interest.
        assert_eq!(fw.pending(&name), 2);
    }

    #[test]
    fn data_satisfies_all_pending_faces_and_caches() {
        let mut fw = Forwarder::new();
        fw.register_route(Name::parse("/v"), 9);
        let name = Name::parse("/v/clip");
        fw.recv_interest(1, &name);
        fw.recv_interest(2, &name);
        let data = Data::new(name.clone(), b"video bytes");
        let downstream = fw.recv_data(data);
        // Delivered to both requesters via the reverse path.
        assert_eq!(downstream, alloc::vec![1, 2]);
        // Now cached; PIT cleared.
        assert_eq!(fw.cache_len(), 1);
        assert_eq!(fw.pending(&name), 0);
    }

    #[test]
    fn cached_interest_is_served_locally() {
        let mut fw = Forwarder::new();
        fw.register_route(Name::parse("/v"), 9);
        let name = Name::parse("/v/clip");
        fw.recv_interest(1, &name);
        fw.recv_data(Data::new(name.clone(), b"payload"));
        // A later Interest hits the Content Store — no upstream forward.
        match fw.recv_interest(3, &name) {
            InterestOutcome::FromCache(d) => {
                assert_eq!(d.content, b"payload");
                assert!(d.verify());
            }
            other => panic!("expected cache hit, got {other:?}"),
        }
    }

    #[test]
    fn hibc_name_is_the_producer_key() {
        let pk = b"producer-public-key";
        let name = Name::hibc(pk, "videos/clip1");
        // The authority component binds to the producer key, with no anchor.
        assert!(name.certifies(pk));
        assert!(!name.certifies(b"someone-else"));
        assert_eq!(name.len(), 3); // H(pk) / videos / clip1
    }

    #[test]
    fn signed_data_verifies_under_hibc() {
        let cal = CryptoLayer::with_defaults();
        let sd = SignedData::produce(&cal, "lamport-pq", b"producer-seed", "doc/readme", b"hello")
            .unwrap();
        // Name authority binds to the producer key the packet carries.
        assert!(sd.data.name.certifies(&sd.producer_pk));
        assert!(sd.verify(&cal));
    }

    #[test]
    fn signed_data_rejects_content_tamper() {
        let cal = CryptoLayer::with_defaults();
        let mut sd = SignedData::produce(&cal, "lamport-pq", b"seed", "x", b"original").unwrap();
        // Tamper the payload (and its digest) — signature no longer matches.
        sd.data = Data::new(sd.data.name.clone(), b"tampered");
        assert!(!sd.verify(&cal));
    }

    #[test]
    fn signed_data_rejects_namespace_hijack() {
        let cal = CryptoLayer::with_defaults();
        let real = SignedData::produce(&cal, "lamport-pq", b"real", "a", b"data").unwrap();
        // Attacker keeps the victim's name but swaps in their own key + signature.
        let attacker = SignedData::produce(&cal, "lamport-pq", b"attacker", "a", b"data").unwrap();
        let hijack = SignedData {
            data: real.data.clone(),
            algo: attacker.algo.clone(),
            producer_pk: attacker.producer_pk.clone(),
            signature: attacker.signature.clone(),
        };
        // The name's authority does not bind to the attacker's key.
        assert!(!hijack.verify(&cal));
    }

    #[test]
    fn notify_wakes_pit_holders_under_a_prefix() {
        let mut fw = Forwarder::new();
        fw.register_route(Name::parse("/jayden"), 9);
        // Three consumers express standing Interest under the topic prefix.
        fw.recv_interest(1, &Name::parse("/jayden/inbox/msg1"));
        fw.recv_interest(2, &Name::parse("/jayden/inbox/msg2"));
        fw.recv_interest(3, &Name::parse("/jayden/calendar/evt"));
        // A producer announces a new version anywhere under /jayden/inbox.
        let mut woken = fw.notify(&Name::parse("/jayden/inbox"));
        woken.sort_unstable();
        assert_eq!(woken, alloc::vec![1, 2]); // face 3 (calendar) is not under inbox
        // A broader announce wakes everyone under /jayden — deduplicated, one pass.
        // (Face order follows the PIT's name ordering; the woken set is what matters.)
        let mut all = fw.notify(&Name::parse("/jayden"));
        all.sort_unstable();
        assert_eq!(all, alloc::vec![1, 2, 3]);
        // Notify does not consume the PIT: subscriptions stay standing.
        assert_eq!(fw.pending(&Name::parse("/jayden/inbox/msg1")), 1);
    }

    #[test]
    fn corrupt_or_unsolicited_data_is_dropped() {
        let mut fw = Forwarder::new();
        // Unsolicited Data (no PIT entry) → delivered nowhere, not cached.
        let downstream = fw.recv_data(Data::new(Name::parse("/x"), b"hi"));
        assert!(downstream.is_empty());
        assert_eq!(fw.cache_len(), 0);
        // Corrupt Data (digest mismatch) → dropped even if solicited.
        fw.register_route(Name::parse("/y"), 1);
        let name = Name::parse("/y/z");
        fw.recv_interest(5, &name);
        let mut bad = Data::new(name, b"good");
        bad.content = b"tampered".to_vec(); // digest no longer matches
        assert!(fw.recv_data(bad).is_empty());
        assert_eq!(fw.cache_len(), 0);
    }
}

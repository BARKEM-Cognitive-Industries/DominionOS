//! The semantic object graph (SRS Stage 5 & 7, and Stage 9's object-centric UI).
//!
//! "The operating system eradicates the traditional filesystem entirely.
//! Instead, data is managed as a content-addressed, immutable, and deduplicated
//! semantic graph, functioning conceptually like a system-wide Git repository.
//! Every object is hashed, allowing for instantaneous system rollbacks and
//! perfect versioning."
//!
//! This module implements exactly that, in-memory:
//!
//! * Objects are *semantic* — a named kind plus typed fields — not opaque files.
//! * Inserting an object returns its content hash ([`ObjectId`]); identical
//!   objects collapse to one entry (deduplication).
//! * Objects are immutable; "editing" creates a new object with a new id.
//! * The graph keeps a linear history of [`Commit`] roots so the whole system
//!   can roll back to any prior state instantly (perfect versioning).

use crate::bytes::Cursor;
use crate::content_store::ContentStore;
use crate::hash::Hash256;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// The content-address of an object: the SHA-256 of its canonical encoding.
pub type ObjectId = Hash256;

/// A typed field value inside a semantic object. `Ref` makes the store a real
/// graph: objects point at other objects by content hash.
#[derive(Clone, PartialEq, Debug)]
pub enum Datum {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Ref(ObjectId),
    /// Raw, uninterpreted bytes — the storage form for a `Blob` (a legacy file
    /// kept verbatim). Distinct from `Text` so non-UTF-8 content round-trips
    /// losslessly and content addressing stays exact.
    Bytes(Vec<u8>),
}

impl Datum {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Datum::Int(v) => {
                out.push(b'i');
                out.extend_from_slice(&v.to_le_bytes());
            }
            Datum::Float(v) => {
                out.push(b'f');
                // Canonicalize -0.0 to 0.0 and all NaNs to one bit pattern so
                // content addressing stays deterministic.
                let bits = if v.is_nan() {
                    0x7ff8_0000_0000_0000u64
                } else if *v == 0.0 {
                    0u64
                } else {
                    v.to_bits()
                };
                out.extend_from_slice(&bits.to_le_bytes());
            }
            Datum::Bool(v) => {
                out.push(b'b');
                out.push(*v as u8);
            }
            Datum::Text(s) => {
                out.push(b't');
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            Datum::Ref(id) => {
                out.push(b'r');
                out.extend_from_slice(&id.0);
            }
            Datum::Bytes(b) => {
                out.push(b'y');
                out.extend_from_slice(&(b.len() as u64).to_le_bytes());
                out.extend_from_slice(b);
            }
        }
    }
}

/// A semantic object: a `kind` (e.g. "Invoice", "Conversation") and a set of
/// named fields. Fields are stored sorted so encoding — and therefore the
/// content hash — is canonical regardless of insertion order.
#[derive(Clone, PartialEq, Debug)]
pub struct Object {
    pub kind: String,
    pub fields: Vec<(String, Datum)>,
}

impl Object {
    pub fn new(kind: impl Into<String>) -> Object {
        Object {
            kind: kind.into(),
            fields: Vec::new(),
        }
    }

    /// Builder-style field setter; re-setting a field replaces it.
    pub fn with(mut self, name: impl Into<String>, value: Datum) -> Object {
        self.set(name, value);
        self
    }

    pub fn set(&mut self, name: impl Into<String>, value: Datum) {
        let name = name.into();
        match self.fields.iter_mut().find(|(k, _)| *k == name) {
            Some(slot) => slot.1 = value,
            None => self.fields.push((name, value)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Datum> {
        self.fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// Canonical byte encoding used for content addressing.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"obj1");
        out.extend_from_slice(&(self.kind.len() as u64).to_le_bytes());
        out.extend_from_slice(self.kind.as_bytes());

        let mut sorted: Vec<&(String, Datum)> = self.fields.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        out.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
        for (k, v) in sorted {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            v.encode(&mut out);
        }
        out
    }

    /// This object's content id.
    pub fn id(&self) -> ObjectId {
        Hash256::of(&self.encode())
    }
}

/// A snapshot of the live object set — a point we can roll back to.
#[derive(Clone, Debug)]
pub struct Commit {
    pub root: Hash256,
    pub parent: Hash256,
    pub live: Vec<ObjectId>,
    pub message: String,
}

/// The content-addressed object store plus its commit history.
pub struct ObjectGraph {
    store: ContentStore<Object>,
    /// The currently live object set (the working "head").
    head: Vec<ObjectId>,
    history: Vec<Commit>,
}

impl ObjectGraph {
    pub fn new() -> ObjectGraph {
        ObjectGraph {
            store: ContentStore::new(),
            head: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Insert an object, returning its content id. Identical objects dedup to a
    /// single stored entry; the id is added to the live head set if new.
    pub fn put(&mut self, obj: Object) -> ObjectId {
        let id = self.store.publish(obj);
        if !self.head.contains(&id) {
            self.head.push(id);
        }
        id
    }

    pub fn get(&self, id: &ObjectId) -> Option<&Object> {
        self.store.fetch(id)
    }

    pub fn contains(&self, id: &ObjectId) -> bool {
        self.store.contains(id)
    }

    /// Number of distinct objects physically stored (post-dedup).
    pub fn stored_count(&self) -> usize {
        self.store.len()
    }

    /// Number of objects currently live in the head set.
    pub fn live_count(&self) -> usize {
        self.head.len()
    }

    pub fn live_ids(&self) -> &[ObjectId] {
        &self.head
    }

    /// The Merkle root over the *sorted* live id set — a single hash naming the
    /// entire current system state (Stage 10's "the entire machine state is a
    /// hashable object").
    pub fn root_hash(&self) -> Hash256 {
        let mut ids = self.head.clone();
        ids.sort();
        let mut acc = Hash256::ZERO;
        for id in ids {
            acc = acc.combine(&id);
        }
        acc
    }

    /// Commit the current head, recording a restorable snapshot.
    pub fn commit(&mut self, message: impl Into<String>) -> Hash256 {
        let parent = self.history.last().map(|c| c.root).unwrap_or(Hash256::ZERO);
        let root = self.root_hash().combine(&parent);
        self.history.push(Commit {
            root,
            parent,
            live: self.head.clone(),
            message: message.into(),
        });
        root
    }

    pub fn history(&self) -> &[Commit] {
        &self.history
    }

    /// Roll the live head back to a previously committed root. Because objects
    /// are immutable and never deleted from the store, this is instantaneous
    /// and lossless.
    pub fn rollback(&mut self, root: Hash256) -> Result<(), String> {
        let commit = self
            .history
            .iter()
            .find(|c| c.root == root)
            .ok_or_else(|| "no such commit root".to_string())?;
        self.head = commit.live.clone();
        Ok(())
    }

    /// Find live objects of a given kind — the semantic query the object-centric
    /// UI of Stage 9 is built on ("show me all Conversations").
    pub fn query_kind<'a>(&'a self, kind: &str) -> Vec<(&'a ObjectId, &'a Object)> {
        self.head
            .iter()
            .filter_map(|id| self.store.fetch(id).map(|o| (id, o)))
            .filter(|(_, o)| o.kind == kind)
            .collect()
    }

    /// Iterate every physically stored object (post-dedup), id-sorted. The incremental
    /// on-disk store ([`crate::objstore`]) walks this to decide which objects are not
    /// yet on disk and therefore need flushing.
    pub fn stored_objects(&self) -> impl Iterator<Item = (&ObjectId, &Object)> {
        self.store.iter()
    }


    /// The live commit history (each a restorable snapshot root).
    pub fn commits(&self) -> &[Commit] {
        &self.history
    }

    /// Rebuild a graph directly from its constituent parts — the inverse of walking
    /// [`stored_objects`](Self::stored_objects) / [`live_ids`](Self::live_ids) /
    /// [`commits`](Self::commits). Object ids are recomputed from content, so a tampered
    /// object would land under a different id than its caller expects; the incremental
    /// store relies on that for content-addressed integrity checking.
    pub fn restore(objects: Vec<Object>, head: Vec<ObjectId>, history: Vec<Commit>) -> ObjectGraph {
        let mut store = ContentStore::new();
        for obj in objects {
            store.publish(obj);
        }
        ObjectGraph { store, head, history }
    }
}

// ---------------------------------------------------------------------------
// Serialization — the byte form persisted to disk (M1). Kept next to the data
// structure so it can reach the private store/head/history. Format is versioned,
// deterministic, and self-describing.
// ---------------------------------------------------------------------------

const GRAPH_MAGIC: &[u8; 8] = b"AEGRPH01";

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn encode_datum(out: &mut Vec<u8>, d: &Datum) {
    // Reuses the same tag bytes as the canonical content encoding.
    d.encode(out);
}

fn encode_object_full(out: &mut Vec<u8>, obj: &Object) {
    put_u32(out, obj.kind.len() as u32);
    out.extend_from_slice(obj.kind.as_bytes());
    put_u32(out, obj.fields.len() as u32);
    for (k, v) in &obj.fields {
        put_u32(out, k.len() as u32);
        out.extend_from_slice(k.as_bytes());
        encode_datum(out, v);
    }
}

// Parsing uses `crate::bytes::Cursor` (aliased as `Reader` here for local
// readability) — the implementation lives in bytes.rs.
type Reader<'a> = Cursor<'a>;

// Extension methods on Cursor that are specific to the object-graph format.
trait ObjectCursorExt<'a> {
    fn string(&mut self) -> Result<String, &'static str>;
    fn datum(&mut self) -> Result<Datum, &'static str>;
    fn object_record(&mut self) -> Result<Object, &'static str>;
}

impl<'a> ObjectCursorExt<'a> for Reader<'a> {
    fn string(&mut self) -> Result<String, &'static str> {
        let len = self.read_u32_le_or("unexpected end of graph data")? as usize;
        let b = self.take_or(len, "unexpected end of graph data")?;
        core::str::from_utf8(b)
            .map(|s| s.to_string())
            .map_err(|_| "invalid utf-8 in graph data")
    }

    fn datum(&mut self) -> Result<Datum, &'static str> {
        let tag = self.read_u8_or("unexpected end of graph data")?;
        match tag {
            b'i' => {
                let b = self.take_or(8, "unexpected end of graph data")?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                Ok(Datum::Int(i64::from_le_bytes(a)))
            }
            b'f' => {
                let b = self.take_or(8, "unexpected end of graph data")?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                Ok(Datum::Float(f64::from_bits(u64::from_le_bytes(a))))
            }
            b'b' => Ok(Datum::Bool(self.read_u8_or("unexpected end of graph data")? != 0)),
            b't' => {
                // Text length is stored as u64 by Datum::encode.
                let len = self.read_u64_le_or("unexpected end of graph data")? as usize;
                let b = self.take_or(len, "unexpected end of graph data")?;
                core::str::from_utf8(b)
                    .map(|s| Datum::Text(s.to_string()))
                    .map_err(|_| "invalid utf-8 in text datum")
            }
            b'r' => Ok(Datum::Ref(self.read_hash_or("unexpected end of graph data")?)),
            b'y' => {
                let len = self.read_u64_le_or("unexpected end of graph data")? as usize;
                Ok(Datum::Bytes(self.take_or(len, "unexpected end of graph data")?.to_vec()))
            }
            _ => Err("unknown datum tag"),
        }
    }

    fn object_record(&mut self) -> Result<Object, &'static str> {
        let kind = self.string()?;
        let field_count = self.read_u32_le_or("unexpected end of graph data")? as usize;
        let mut obj = Object::new(kind);
        for _ in 0..field_count {
            let key_len = self.read_u32_le_or("unexpected end of graph data")? as usize;
            let key = core::str::from_utf8(self.take_or(key_len, "unexpected end of graph data")?)
                .map_err(|_| "invalid utf-8 in field key")?
                .to_string();
            let value = self.datum()?;
            obj.fields.push((key, value));
        }
        Ok(obj)
    }
}

impl Object {
    /// Parse the canonical content encoding produced by [`encode`](Self::encode).
    ///
    /// These bytes are *self-verifying*: `Object::decode(b)?.id()` reproduces the
    /// original id only when `b` is exactly an [`encode`](Self::encode) output, so the
    /// content-addressed store can confirm every object it reads back is intact (an
    /// integrity check the monolithic format never had).
    pub fn decode(bytes: &[u8]) -> Result<Object, &'static str> {
        let mut r = Reader::new(bytes);
        if r.take_or(4, "unexpected end of graph data")? != b"obj1" {
            return Err("bad object magic");
        }
        let kind_len = r.read_u64_le_or("unexpected end of graph data")? as usize;
        let kind = core::str::from_utf8(r.take_or(kind_len, "unexpected end of graph data")?)
            .map_err(|_| "invalid utf-8 in object kind")?
            .to_string();
        let field_count = r.read_u64_le_or("unexpected end of graph data")? as usize;
        let mut obj = Object::new(kind);
        for _ in 0..field_count {
            let key_len = r.read_u64_le_or("unexpected end of graph data")? as usize;
            let key = core::str::from_utf8(r.take_or(key_len, "unexpected end of graph data")?)
                .map_err(|_| "invalid utf-8 in field key")?
                .to_string();
            let value = r.datum()?;
            obj.fields.push((key, value));
        }
        Ok(obj)
    }
}

impl ObjectGraph {
    /// Serialise the whole graph — every stored object, the live head, and the
    /// full commit history — into a deterministic, versioned byte stream.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(GRAPH_MAGIC);

        put_u32(&mut out, self.store.len() as u32);
        for (_, obj) in self.store.iter() {
            encode_object_full(&mut out, obj);
        }

        put_u32(&mut out, self.head.len() as u32);
        for id in &self.head {
            out.extend_from_slice(&id.0);
        }

        put_u32(&mut out, self.history.len() as u32);
        for c in &self.history {
            out.extend_from_slice(&c.root.0);
            out.extend_from_slice(&c.parent.0);
            put_u32(&mut out, c.live.len() as u32);
            for id in &c.live {
                out.extend_from_slice(&id.0);
            }
            put_u32(&mut out, c.message.len() as u32);
            out.extend_from_slice(c.message.as_bytes());
        }
        out
    }

    /// Reconstruct a graph from [`serialize`](Self::serialize) output. Object ids
    /// are recomputed from content (content addressing), so the head/history id
    /// references remain valid.
    pub fn deserialize(bytes: &[u8]) -> Result<ObjectGraph, &'static str> {
        let mut r = Reader::new(bytes);
        if r.take_or(8, "unexpected end of graph data")? != GRAPH_MAGIC {
            return Err("bad graph magic");
        }
        let mut graph = ObjectGraph::new();

        let obj_count = r.read_u32_le_or("unexpected end of graph data")? as usize;
        for _ in 0..obj_count {
            let obj = r.object_record()?;
            graph.store.publish(obj);
        }

        let head_len = r.read_u32_le_or("unexpected end of graph data")? as usize;
        graph.head = Vec::with_capacity(head_len);
        for _ in 0..head_len {
            graph.head.push(r.read_hash_or("unexpected end of graph data")?);
        }

        let hist_len = r.read_u32_le_or("unexpected end of graph data")? as usize;
        graph.history = Vec::with_capacity(hist_len);
        for _ in 0..hist_len {
            let root = r.read_hash_or("unexpected end of graph data")?;
            let parent = r.read_hash_or("unexpected end of graph data")?;
            let live_len = r.read_u32_le_or("unexpected end of graph data")? as usize;
            let mut live = Vec::with_capacity(live_len);
            for _ in 0..live_len {
                live.push(r.read_hash_or("unexpected end of graph data")?);
            }
            let message = r.string()?;
            graph.history.push(Commit { root, parent, live, message });
        }
        Ok(graph)
    }
}

impl Default for ObjectGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ObjectGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObjectGraph(stored={}, live={}, commits={}, root={})",
            self.stored_count(),
            self.live_count(),
            self.history.len(),
            self.root_hash().short()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invoice(amount: i64) -> Object {
        Object::new("Invoice")
            .with("amount", Datum::Int(amount))
            .with("client", Datum::Text("Acme".into()))
    }

    #[test]
    fn identical_objects_have_identical_ids() {
        assert_eq!(invoice(100).id(), invoice(100).id());
        assert_ne!(invoice(100).id(), invoice(200).id());
    }

    #[test]
    fn field_order_does_not_affect_id() {
        let a = Object::new("P").with("x", Datum::Int(1)).with("y", Datum::Int(2));
        let b = Object::new("P").with("y", Datum::Int(2)).with("x", Datum::Int(1));
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn put_deduplicates() {
        let mut g = ObjectGraph::new();
        g.put(invoice(100));
        g.put(invoice(100));
        g.put(invoice(100));
        assert_eq!(g.stored_count(), 1);
        assert_eq!(g.live_count(), 1);
    }

    #[test]
    fn get_round_trips() {
        let mut g = ObjectGraph::new();
        let id = g.put(invoice(42));
        let got = g.get(&id).unwrap();
        assert_eq!(got.kind, "Invoice");
        assert_eq!(got.get("amount"), Some(&Datum::Int(42)));
    }

    #[test]
    fn references_form_a_graph() {
        let mut g = ObjectGraph::new();
        let child = g.put(Object::new("Client").with("name", Datum::Text("Acme".into())));
        let parent = g.put(Object::new("Invoice").with("client", Datum::Ref(child)));
        let p = g.get(&parent).unwrap();
        match p.get("client") {
            Some(Datum::Ref(r)) => assert!(g.contains(r)),
            _ => panic!("expected a ref"),
        }
    }

    #[test]
    fn commit_and_rollback_restore_state() {
        let mut g = ObjectGraph::new();
        g.put(invoice(1));
        let snap = g.commit("one invoice");
        assert_eq!(g.live_count(), 1);

        g.put(invoice(2));
        g.put(invoice(3));
        assert_eq!(g.live_count(), 3);

        g.rollback(snap).unwrap();
        assert_eq!(g.live_count(), 1);
        // Objects were never lost — they are still in the store.
        assert_eq!(g.stored_count(), 3);
    }

    #[test]
    fn rollback_to_unknown_root_errors() {
        let mut g = ObjectGraph::new();
        assert!(g.rollback(Hash256::of(b"nope")).is_err());
    }

    #[test]
    fn root_hash_reflects_live_set() {
        let mut g = ObjectGraph::new();
        let empty = g.root_hash();
        g.put(invoice(1));
        let one = g.root_hash();
        assert_ne!(empty, one);
    }

    #[test]
    fn bytes_round_trip_and_address() {
        let mut g = ObjectGraph::new();
        let raw = alloc::vec![0u8, 1, 2, 255, 254, 0, 128];
        let id = g.put(Object::new("Blob").with("data", Datum::Bytes(raw.clone())));
        match g.get(&id).unwrap().get("data") {
            Some(Datum::Bytes(b)) => assert_eq!(b, &raw),
            other => panic!("expected bytes, got {:?}", other),
        }
        // Identical bytes content-address identically; one differing byte does not.
        let id2 = g.put(Object::new("Blob").with("data", Datum::Bytes(raw)));
        assert_eq!(id, id2);
        let mut diff = alloc::vec![0u8, 1, 2, 255, 254, 0, 129];
        let id3 = g.put(Object::new("Blob").with("data", Datum::Bytes(core::mem::take(&mut diff))));
        assert_ne!(id, id3);
    }

    #[test]
    fn bytes_not_confused_with_text() {
        // "ab" as text vs the bytes [0x61,0x62] must hash differently (typed encoding).
        let t = Object::new("X").with("v", Datum::Text("ab".into())).id();
        let b = Object::new("X").with("v", Datum::Bytes(alloc::vec![0x61, 0x62])).id();
        assert_ne!(t, b);
    }

    #[test]
    fn query_kind_filters() {
        let mut g = ObjectGraph::new();
        g.put(invoice(1));
        g.put(Object::new("Client").with("name", Datum::Text("Bob".into())));
        assert_eq!(g.query_kind("Invoice").len(), 1);
        assert_eq!(g.query_kind("Client").len(), 1);
        assert_eq!(g.query_kind("Ghost").len(), 0);
    }
}

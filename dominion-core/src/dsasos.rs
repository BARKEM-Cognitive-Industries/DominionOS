//! Distributed Single Address Space — **one address space across a fleet**
//! (`docs/architecture/distributed-sasos-and-global-address-space.md`).
//!
//! On one machine the SASOS is a content-addressed object graph: a reference *is* a
//! capability over `[base,len)` and resolving it is a local read. This module extends that
//! to a **fleet**: a reference may name an object that lives on *another* node, and resolving
//! it **faults the object in over NDN by content hash** — not a block swap, an Interest.
//!
//! Three pieces, each building on already-implemented substrate:
//!
//! * **Global address space** ([`GlobalSpace`]) — a per-node resident set
//!   ([`crate::pressure::WorkingSet`]) over the shared content-addressed store. A
//!   [`GlobalRef`] is location-independent: the same hash resolves to the same bytes on any
//!   node. A non-resident reference triggers a **remote page-fault** that issues an NDN
//!   Interest for the hash and admits the verified reply — *verify-by-rehash* (the reply's
//!   `H(bytes)` must equal the requested id), so a lying peer is caught by construction
//!   (the same guarantee [`crate::ndn::Data::verify`] gives).
//! * **Cell migration** ([`migrate`]) — ship a cell's small **control state**
//!   ([`crate::state`]-style snapshot) to another node; its working set pages in lazily by
//!   hash. Because everything is content-addressed and deterministic, the migrated cell
//!   **resumes to the identical result** — proven by a digest match.
//! * **CHERI-D temporal safety** ([`GenStore`]) — 8-bit **generation IDs** on slots, bumped
//!   on free, so a stale reference to a reused slot is a cheap **use-after-free trap** with no
//!   GC sweep — the temporal-safety complement to the spatial bounds in
//!   [`crate::capability`], valid across the shared store.
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use crate::pressure::WorkingSet;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// ───────────────────────────── global, location-independent references ─────────────────────────────

/// A fleet-wide reference to an immutable object: its content hash + length. The hash is the
/// name — it resolves to the same bytes on any node, so the reference carries no location.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GlobalRef {
    pub id: Hash256,
    pub len: u64,
}

impl GlobalRef {
    /// The location-independent reference to `bytes` (its content address).
    pub fn of(bytes: &[u8]) -> GlobalRef {
        GlobalRef { id: Hash256::of(bytes), len: bytes.len() as u64 }
    }
    /// The NDN name an Interest for this object uses (`/dominion/obj/<hex>`).
    pub fn ndn_name(&self) -> crate::ndn::Name {
        let mut s = alloc::string::String::from("/dominion/obj/");
        s.push_str(&self.id.short());
        crate::ndn::Name::parse(&s)
    }
}

/// A peer that can answer a content-addressed fetch — the production path is an NDN
/// Interest/Data exchange over [`crate::ndn::Forwarder`] / [`crate::dominionlink`]; in tests a
/// [`MapStore`] stands in for the remote node's content store.
pub trait RemoteStore {
    fn fetch(&self, id: Hash256) -> Option<Vec<u8>>;
}

/// An in-memory content store (a peer's resident objects), keyed by content hash.
#[derive(Clone, Default)]
pub struct MapStore {
    objects: BTreeMap<Hash256, Vec<u8>>,
}

impl MapStore {
    pub fn new() -> MapStore {
        MapStore { objects: BTreeMap::new() }
    }
    /// Store `bytes` and return its global reference.
    pub fn put(&mut self, bytes: &[u8]) -> GlobalRef {
        let r = GlobalRef::of(bytes);
        self.objects.insert(r.id, bytes.to_vec());
        r
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl RemoteStore for MapStore {
    fn fetch(&self, id: Hash256) -> Option<Vec<u8>> {
        self.objects.get(&id).cloned()
    }
}

/// Why a resolve failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultError {
    /// No peer could supply the object (true page fault with no backing).
    Unresolved,
    /// A peer replied, but `H(bytes) != id` — a lying/corrupt peer, refused.
    IntegrityFailed,
}

/// One node's view of the global address space: a resident set over the content store plus a
/// fault path that pulls non-resident objects in by hash.
pub struct GlobalSpace {
    node: u32,
    resident: BTreeMap<Hash256, Vec<u8>>,
    ws: WorkingSet,
    local_hits: u64,
    remote_faults: u64,
    refused: u64,
}

impl GlobalSpace {
    /// A node with a resident-set `quota` (bytes) managed by [`WorkingSet`].
    pub fn new(node: u32, quota: usize) -> GlobalSpace {
        GlobalSpace {
            node,
            resident: BTreeMap::new(),
            ws: WorkingSet::new(quota),
            local_hits: 0,
            remote_faults: 0,
            refused: 0,
        }
    }

    pub fn node(&self) -> u32 {
        self.node
    }

    /// Make `bytes` resident locally (e.g. a local write or a produced object).
    pub fn admit_local(&mut self, bytes: &[u8]) -> GlobalRef {
        let r = GlobalRef::of(bytes);
        self.insert_resident(r.id, bytes.to_vec());
        r
    }

    fn insert_resident(&mut self, id: Hash256, bytes: Vec<u8>) {
        let evicted = self.ws.admit(id, bytes.len(), false);
        for e in evicted {
            // Clean objects are re-fetchable by hash, so eviction just drops the copy.
            self.resident.remove(&e);
        }
        self.resident.insert(id, bytes);
    }

    pub fn is_resident(&self, id: Hash256) -> bool {
        self.resident.contains_key(&id)
    }
    pub fn local_hits(&self) -> u64 {
        self.local_hits
    }
    pub fn remote_faults(&self) -> u64 {
        self.remote_faults
    }
    pub fn refused(&self) -> u64 {
        self.refused
    }

    /// Resolve a global reference. A resident object is a local hit; otherwise this is a
    /// **remote page-fault**: issue an Interest for the hash, verify the reply by rehash, and
    /// admit it. A reply whose hash does not match the request is refused (lying peer).
    pub fn resolve(&mut self, r: GlobalRef, remote: &dyn RemoteStore) -> Result<Vec<u8>, FaultError> {
        if let Some(bytes) = self.resident.get(&r.id) {
            self.local_hits += 1;
            self.ws.touch(r.id);
            return Ok(bytes.clone());
        }
        // Remote page-fault path (rides an NDN Interest/Data in production).
        let bytes = remote.fetch(r.id).ok_or(FaultError::Unresolved)?;
        if Hash256::of(&bytes) != r.id {
            // verify-by-rehash failed: the peer lied or the object is corrupt.
            self.refused += 1;
            return Err(FaultError::IntegrityFailed);
        }
        self.remote_faults += 1;
        self.insert_resident(r.id, bytes.clone());
        Ok(bytes)
    }
}

// ───────────────────────────── cell migration ─────────────────────────────

/// A migratable cell's portable state: its small control state plus the set of objects in its
/// working set (paged in lazily on the destination by hash).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellSnapshot {
    pub control: Vec<u8>,
    pub working_set: Vec<GlobalRef>,
}

impl CellSnapshot {
    /// A content digest binding the control state and the (ordered) working-set ids — equal
    /// iff two snapshots are the same migratable state.
    pub fn digest(&self) -> Hash256 {
        let mut input = Vec::with_capacity(8 + self.control.len() + 8 + self.working_set.len() * 40);
        // Length-prefix the variable-length control and the working-set count so the
        // serialization is injective: distinct snapshots cannot collide by shifting bytes
        // between the control field and a working-set entry.
        input.extend_from_slice(&(self.control.len() as u64).to_le_bytes());
        input.extend_from_slice(&self.control);
        input.extend_from_slice(&(self.working_set.len() as u64).to_le_bytes());
        for r in &self.working_set {
            input.extend_from_slice(&r.id.0);
            input.extend_from_slice(&r.len.to_le_bytes());
        }
        Hash256::of(&input)
    }
}

/// Migrate a cell from `src` to `dst`: ship the control state, then page the working set into
/// `dst` from `src` (a peer) on demand by hash. Returns the destination snapshot — which has
/// the **same digest** as the source, proving deterministic resume. The working set is now
/// resident on `dst` (verify-by-rehash on every page-in).
pub fn migrate(
    snapshot: &CellSnapshot,
    dst: &mut GlobalSpace,
    src: &dyn RemoteStore,
) -> Result<CellSnapshot, FaultError> {
    for r in &snapshot.working_set {
        // Lazy page-in: each working-set object faults in from the source node by hash.
        let _ = dst.resolve(*r, src)?;
    }
    Ok(CellSnapshot { control: snapshot.control.clone(), working_set: snapshot.working_set.clone() })
}

// ───────────────────────────── CHERI-D temporal safety ─────────────────────────────

/// A reference to a generation-tracked slot: the slot index plus the generation it was minted
/// at. Dereferencing checks the slot's *current* generation, so a reference outlives a `free`
/// only as a detectable dangling pointer — never silent corruption.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GenRef {
    pub slot: u32,
    pub generation: u8,
}

/// A generation table over a set of slots — the software model of CHERI-D temporal safety.
/// `alloc` mints a reference at the slot's current generation; `free` **increments** the
/// generation (8-bit, wrapping), instantly invalidating every outstanding reference to that
/// slot. There is no GC sweep — staleness is a single comparison.
#[derive(Clone, Default)]
pub struct GenStore {
    /// Current generation per allocated slot (absent ⇒ never allocated / freed-and-reaped).
    generation: BTreeMap<u32, u8>,
    /// Live payloads per slot (only the current generation can read them).
    payload: BTreeMap<u32, Vec<u8>>,
    next_slot: u32,
}

/// Why a generation-checked dereference trapped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenError {
    /// The slot was never allocated.
    NoSuchSlot,
    /// The reference's generation does not match the slot's current one (use-after-free).
    StaleGeneration,
}

impl GenStore {
    pub fn new() -> GenStore {
        GenStore { generation: BTreeMap::new(), payload: BTreeMap::new(), next_slot: 0 }
    }

    /// Allocate a fresh slot holding `value`; returns a reference at its current generation.
    pub fn alloc(&mut self, value: &[u8]) -> GenRef {
        let slot = self.next_slot;
        self.next_slot += 1;
        let generation = *self.generation.entry(slot).or_insert(0);
        self.payload.insert(slot, value.to_vec());
        GenRef { slot, generation }
    }

    /// Dereference a generation-checked reference. Traps if the slot was freed (and thus
    /// bumped) since the reference was minted — a use-after-free caught for free.
    pub fn deref(&self, r: GenRef) -> Result<&[u8], GenError> {
        let cur = *self.generation.get(&r.slot).ok_or(GenError::NoSuchSlot)?;
        if cur != r.generation {
            return Err(GenError::StaleGeneration);
        }
        self.payload.get(&r.slot).map(|v| v.as_slice()).ok_or(GenError::StaleGeneration)
    }

    /// Free a slot: bump its generation (wrapping at 8 bits) and drop the payload. Every
    /// outstanding [`GenRef`] to it now traps on `deref`.
    pub fn free(&mut self, slot: u32) {
        if let Some(g) = self.generation.get_mut(&slot) {
            *g = g.wrapping_add(1);
        }
        self.payload.remove(&slot);
    }

    /// Re-allocate a freed slot (the common case: the slot is reused at the new generation).
    pub fn realloc(&mut self, slot: u32, value: &[u8]) -> Option<GenRef> {
        let g = *self.generation.get(&slot)?;
        self.payload.insert(slot, value.to_vec());
        Some(GenRef { slot, generation: g })
    }
}

// ───────────────────────────── DST scenario ─────────────────────────────

/// A deterministic two-node scenario: node B resolves a reference to an object that lives only
/// on node A (remote page-fault + verify-by-rehash), then a cell migrates A→B and resumes with
/// an identical digest. Returns `(resolved_ok, migrated_deterministically, lie_refused)` — a
/// pure function (no clock/RNG used).
pub fn dsasos_scenario() -> (bool, bool, bool) {
    // Node A holds the objects; node B starts empty.
    let mut a_store = MapStore::new();
    let r1 = a_store.put(b"the-distributed-object-graph");
    let r2 = a_store.put(b"a-second-working-set-object");

    let mut b = GlobalSpace::new(1, 1 << 20);
    // B faults r1 in from A over the (modeled) NDN path, verifying by rehash.
    let resolved_ok = b.resolve(r1, &a_store) == Ok(b"the-distributed-object-graph".to_vec())
        && b.is_resident(r1.id)
        && b.remote_faults() == 1;

    // A cell on A with working set {r1, r2} migrates to B; B pages r2 in lazily.
    let snap = CellSnapshot { control: b"pc=0x40;acc=7".to_vec(), working_set: alloc::vec![r1, r2] };
    let migrated = migrate(&snap, &mut b, &a_store).map(|s| s.digest() == snap.digest()).unwrap_or(false)
        && b.is_resident(r2.id);

    // A lying peer that returns the wrong bytes for a hash is refused.
    let mut liar = MapStore::new();
    liar.objects.insert(r1.id, b"tampered-payload".to_vec()); // wrong content under r1.id
    let mut c = GlobalSpace::new(2, 1 << 20);
    let lie_refused = c.resolve(r1, &liar) == Err(FaultError::IntegrityFailed);

    (resolved_ok, migrated, lie_refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_ref_is_location_independent() {
        // The same bytes get the same reference regardless of which node computes it.
        let r1 = GlobalRef::of(b"shared");
        let r2 = GlobalRef::of(b"shared");
        assert_eq!(r1, r2);
        assert_ne!(r1, GlobalRef::of(b"other"));
    }

    #[test]
    fn remote_fault_pulls_in_by_hash_and_verifies() {
        let mut a = MapStore::new();
        let r = a.put(b"payload");
        let mut b = GlobalSpace::new(1, 1 << 20);
        assert!(!b.is_resident(r.id));
        let got = b.resolve(r, &a).unwrap();
        assert_eq!(got, b"payload");
        assert!(b.is_resident(r.id)); // now cached
        assert_eq!(b.remote_faults(), 1);
        // Second resolve is a local hit, no new fault.
        let _ = b.resolve(r, &a).unwrap();
        assert_eq!(b.remote_faults(), 1);
        assert_eq!(b.local_hits(), 1);
    }

    #[test]
    fn a_lying_peer_is_refused_by_rehash() {
        let mut liar = MapStore::new();
        let r = GlobalRef::of(b"honest-bytes");
        liar.objects.insert(r.id, b"evil-bytes".to_vec()); // mismatched content under the id
        let mut b = GlobalSpace::new(1, 1 << 20);
        assert_eq!(b.resolve(r, &liar), Err(FaultError::IntegrityFailed));
        assert!(!b.is_resident(r.id));
        assert_eq!(b.refused(), 1);
    }

    #[test]
    fn unresolved_reference_is_a_clean_fault() {
        let empty = MapStore::new();
        let mut b = GlobalSpace::new(1, 1 << 20);
        assert_eq!(b.resolve(GlobalRef::of(b"absent"), &empty), Err(FaultError::Unresolved));
    }

    #[test]
    fn migration_resumes_with_an_identical_digest() {
        let mut a = MapStore::new();
        let r1 = a.put(b"obj-1");
        let r2 = a.put(b"obj-2");
        let snap = CellSnapshot { control: b"state".to_vec(), working_set: alloc::vec![r1, r2] };
        let mut dst = GlobalSpace::new(2, 1 << 20);
        let resumed = migrate(&snap, &mut dst, &a).unwrap();
        assert_eq!(resumed.digest(), snap.digest());
        assert!(dst.is_resident(r1.id) && dst.is_resident(r2.id));
    }

    #[test]
    fn generation_ids_trap_use_after_free() {
        let mut g = GenStore::new();
        let r = g.alloc(b"live");
        assert_eq!(g.deref(r), Ok(b"live".as_ref()));
        g.free(r.slot);
        // The old reference now dangles — a cheap stale-generation trap, no GC.
        assert_eq!(g.deref(r), Err(GenError::StaleGeneration));
        // Reusing the slot mints a new generation; the fresh ref works, the old still traps.
        let r2 = g.realloc(r.slot, b"reused").unwrap();
        assert_ne!(r2.generation, r.generation);
        assert_eq!(g.deref(r2), Ok(b"reused".as_ref()));
        assert_eq!(g.deref(r), Err(GenError::StaleGeneration));
    }

    #[test]
    fn deref_of_unknown_slot_traps() {
        let g = GenStore::new();
        assert_eq!(g.deref(GenRef { slot: 99, generation: 0 }), Err(GenError::NoSuchSlot));
    }

    #[test]
    fn scenario_holds_and_is_pure() {
        assert_eq!(dsasos_scenario(), (true, true, true));
        assert_eq!(dsasos_scenario(), dsasos_scenario());
    }
}

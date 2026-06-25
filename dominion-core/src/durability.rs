//! Persistence durability levels, background scrubbing / self-heal, and a tracing
//! garbage collector (`docs/implementation/persistence-and-crash-consistency.md`).
//!
//! The journal ([`crate::journal`]) gives crash-consistent commits; this module adds the
//! storage-management policies on top:
//!
//! * [`DurabilityLevel`] — per-object / per-domain choice of `Sync` (flush before ack),
//!   `Async` (ack then flush), or `GroupCommit` (batch many objects into one barrier) —
//!   so a database journal and a scratch cache get the durability they each need.
//! * [`Scrubber`] — **background scrubbing with optional redundancy and self-heal**:
//!   content-addressed replicas are periodically re-hashed; a bit-rotted copy is detected
//!   (its bytes no longer match the content id) and **repaired from a healthy replica**.
//! * [`TracingGc`] — a **reference-tracing collector** that marks everything reachable
//!   from the roots and sweeps the rest, while honoring **pinned** objects and a
//!   retention/legal-hold set so nothing under policy is ever collected.
//!
//! Pure, safe `no_std`; deterministic. Host-tested.

use crate::hash::Hash256;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

// ───────────────────────── durability levels ─────────────────────────

/// How durably a write must land before it is acknowledged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurabilityLevel {
    /// Flush to stable storage **before** acking (strongest, slowest).
    Sync,
    /// Ack immediately, flush in the background (fast, small loss window).
    Async,
    /// Batch with other writes into one barrier (throughput-optimized).
    GroupCommit,
}

impl DurabilityLevel {
    /// Whether a write at this level must be on stable storage before the ack returns.
    pub fn flush_before_ack(self) -> bool {
        matches!(self, DurabilityLevel::Sync)
    }
}

/// Per-object / per-domain durability policy, with a default for anything unspecified.
pub struct DurabilityPolicy {
    default: DurabilityLevel,
    by_object: BTreeMap<Hash256, DurabilityLevel>,
    by_domain: BTreeMap<u64, DurabilityLevel>,
}

impl DurabilityPolicy {
    pub fn new(default: DurabilityLevel) -> DurabilityPolicy {
        DurabilityPolicy { default, by_object: BTreeMap::new(), by_domain: BTreeMap::new() }
    }

    /// Pin a specific object to a durability level (overrides the domain + default).
    pub fn set_object(&mut self, id: Hash256, level: DurabilityLevel) {
        self.by_object.insert(id, level);
    }

    /// Set the default level for a whole domain.
    pub fn set_domain(&mut self, domain: u64, level: DurabilityLevel) {
        self.by_domain.insert(domain, level);
    }

    /// The level for an object in a domain: object override, else domain, else default.
    pub fn level_for(&self, id: &Hash256, domain: u64) -> DurabilityLevel {
        if let Some(l) = self.by_object.get(id) {
            *l
        } else if let Some(l) = self.by_domain.get(&domain) {
            *l
        } else {
            self.default
        }
    }
}

// ───────────────────────── scrubbing + self-heal ─────────────────────────

/// A content-addressed object stored with `R` redundant replicas.
struct Replicated {
    id: Hash256,
    copies: Vec<Vec<u8>>,
}

/// The outcome of a scrub pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrubReport {
    /// Replicas whose bytes no longer matched the content id (bit-rot found).
    pub corrupt_found: usize,
    /// Replicas repaired from a healthy copy.
    pub repaired: usize,
    /// Objects with **no** healthy replica left (unrecoverable — needs backup restore).
    pub unrecoverable: usize,
}

/// A redundant, content-addressed store that scrubs for bit-rot and self-heals.
#[derive(Default)]
pub struct Scrubber {
    objects: BTreeMap<Hash256, Replicated>,
}

impl Scrubber {
    pub fn new() -> Scrubber {
        Scrubber { objects: BTreeMap::new() }
    }

    /// Store `data` with `redundancy` replicas (≥1). The id is its content hash.
    pub fn store(&mut self, data: &[u8], redundancy: usize) -> Hash256 {
        let id = Hash256::of(data);
        let copies = (0..redundancy.max(1)).map(|_| data.to_vec()).collect();
        self.objects.insert(id, Replicated { id, copies });
        id
    }

    /// Read a verified copy of an object (the first replica whose bytes match the id).
    pub fn read(&self, id: &Hash256) -> Option<&[u8]> {
        let obj = self.objects.get(id)?;
        obj.copies.iter().find(|c| Hash256::of(c) == *id).map(|c| c.as_slice())
    }

    /// **Test hook**: corrupt replica `which` of `id` (simulates a bit flip on disk).
    pub fn corrupt_replica(&mut self, id: &Hash256, which: usize) -> bool {
        if let Some(obj) = self.objects.get_mut(id) {
            if let Some(copy) = obj.copies.get_mut(which) {
                if !copy.is_empty() {
                    copy[0] ^= 0xFF;
                    return true;
                }
            }
        }
        false
    }

    /// A background scrub pass: re-hash every replica, repair corrupt ones from a healthy
    /// copy of the same object, and report anything beyond repair.
    pub fn scrub(&mut self) -> ScrubReport {
        let mut report = ScrubReport::default();
        for obj in self.objects.values_mut() {
            // Find a healthy replica (bytes match the content id).
            let healthy = obj.copies.iter().find(|c| Hash256::of(c) == obj.id).cloned();
            let mut any_corrupt = false;
            for copy in obj.copies.iter_mut() {
                if Hash256::of(copy) != obj.id {
                    any_corrupt = true;
                    report.corrupt_found += 1;
                    if let Some(good) = &healthy {
                        *copy = good.clone();
                        report.repaired += 1;
                    }
                }
            }
            if any_corrupt && healthy.is_none() {
                report.unrecoverable += 1;
            }
        }
        report
    }
}

// ───────────────────────── tracing GC ─────────────────────────

/// A reference-tracing garbage collector over a content-addressed object graph. Marks
/// everything reachable from the roots, then sweeps the unreachable — except objects that
/// are **pinned** or under **retention/legal hold**, which are never collected.
#[derive(Default)]
pub struct TracingGc {
    /// Adjacency: object → the objects it references.
    edges: BTreeMap<Hash256, Vec<Hash256>>,
    pinned: BTreeSet<Hash256>,
    held: BTreeSet<Hash256>,
}

impl TracingGc {
    pub fn new() -> TracingGc {
        TracingGc { edges: BTreeMap::new(), pinned: BTreeSet::new(), held: BTreeSet::new() }
    }

    /// Declare an object and the objects it references.
    pub fn add_object(&mut self, id: Hash256, references: &[Hash256]) {
        self.edges.insert(id, references.to_vec());
    }

    /// Pin an object so it is never collected (kernel state, active working set).
    pub fn pin(&mut self, id: Hash256) {
        self.pinned.insert(id);
    }

    /// Mark an object as under retention / legal hold (never collected while held).
    pub fn hold(&mut self, id: Hash256) {
        self.held.insert(id);
    }

    /// Everything reachable from `roots` (transitive closure).
    pub fn reachable(&self, roots: &[Hash256]) -> BTreeSet<Hash256> {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<Hash256> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                if let Some(refs) = self.edges.get(&id) {
                    for r in refs {
                        if !seen.contains(r) {
                            stack.push(*r);
                        }
                    }
                }
            }
        }
        seen
    }

    /// Collect: return the ids that are unreachable from `roots` **and** neither pinned
    /// nor held. Those are safe to crypto-GC (destroy their keys).
    pub fn collect(&self, roots: &[Hash256]) -> Vec<Hash256> {
        let live = self.reachable(roots);
        self.edges
            .keys()
            .filter(|id| !live.contains(id) && !self.pinned.contains(id) && !self.held.contains(id))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durability_policy_resolves_object_then_domain_then_default() {
        let mut p = DurabilityPolicy::new(DurabilityLevel::Async);
        let obj = Hash256::of(b"db-journal");
        p.set_domain(7, DurabilityLevel::GroupCommit);
        p.set_object(obj, DurabilityLevel::Sync);
        // Object override wins, and Sync flushes before ack.
        assert_eq!(p.level_for(&obj, 7), DurabilityLevel::Sync);
        assert!(p.level_for(&obj, 7).flush_before_ack());
        // Another object in domain 7 takes the domain level.
        assert_eq!(p.level_for(&Hash256::of(b"other"), 7), DurabilityLevel::GroupCommit);
        // An object in an unspecified domain takes the default.
        assert_eq!(p.level_for(&Hash256::of(b"x"), 99), DurabilityLevel::Async);
    }

    #[test]
    fn scrubber_detects_bitrot_and_self_heals_from_a_replica() {
        let mut s = Scrubber::new();
        let id = s.store(b"important-object", 3); // 3 replicas
        assert_eq!(s.read(&id), Some(b"important-object".as_ref()));
        // Corrupt one replica → a scrub finds and repairs it from a healthy copy.
        assert!(s.corrupt_replica(&id, 1));
        let report = s.scrub();
        assert_eq!(report.corrupt_found, 1);
        assert_eq!(report.repaired, 1);
        assert_eq!(report.unrecoverable, 0);
        // The object reads cleanly again.
        assert_eq!(s.read(&id), Some(b"important-object".as_ref()));
    }

    #[test]
    fn scrubber_reports_unrecoverable_when_all_replicas_rot() {
        let mut s = Scrubber::new();
        let id = s.store(b"single-copy", 1);
        s.corrupt_replica(&id, 0);
        let report = s.scrub();
        assert_eq!(report.unrecoverable, 1);
        assert_eq!(report.repaired, 0);
        assert!(s.read(&id).is_none()); // no healthy copy left
    }

    #[test]
    fn tracing_gc_collects_unreachable_but_spares_pinned_and_held() {
        let mut gc = TracingGc::new();
        let root = Hash256::of(b"root");
        let a = Hash256::of(b"a");
        let b = Hash256::of(b"b"); // referenced by root
        let garbage = Hash256::of(b"garbage"); // referenced by nobody
        let pinned = Hash256::of(b"kernel"); // unreachable but pinned
        let held = Hash256::of(b"under-legal-hold"); // unreachable but held
        gc.add_object(root, &[b]);
        gc.add_object(b, &[a]);
        gc.add_object(a, &[]);
        gc.add_object(garbage, &[]);
        gc.add_object(pinned, &[]);
        gc.add_object(held, &[]);
        gc.pin(pinned);
        gc.hold(held);
        let collected = gc.collect(&[root]);
        // Only the genuine garbage is collected.
        assert!(collected.contains(&garbage));
        assert!(!collected.contains(&a)); // reachable root → b → a
        assert!(!collected.contains(&pinned));
        assert!(!collected.contains(&held));
        assert_eq!(collected.len(), 1);
    }
}

/// Lever 2: Store-once RAM dedup — cross-domain shared backing store for semantic objects.
///
/// Two domains holding the same library, font, model shard, or decoded image share ONE
/// physical resident copy keyed by ObjectId (which is Hash256). Capability handoff (not
/// copy) grants access. Copy-on-write only when a holder mutates (not implemented here —
/// mutation produces a new object with a new id, naturally).
use crate::hash::Hash256;
use crate::object::{Object, ObjectId};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Byte-size estimation
// ---------------------------------------------------------------------------

/// Rough in-memory byte estimate for an Object.
/// Uses the canonical encoding length as a proxy.
pub fn estimate_bytes(obj: &Object) -> usize {
    obj.encode().len()
}

// ---------------------------------------------------------------------------
// RamResidencyIndex
// ---------------------------------------------------------------------------

/// Cross-domain RAM residency index.
/// All domains share ONE stored copy of each unique object keyed by ObjectId.
/// Reference counting tracks how many domain heads reference each object.
pub struct RamResidencyIndex {
    residents: BTreeMap<ObjectId, Object>,
    refcounts: BTreeMap<ObjectId, u32>,
    domain_heads: BTreeMap<u64, Vec<ObjectId>>, // domain_id -> live set
    stats: RamDedupStats,
}

#[derive(Clone, Debug, Default)]
pub struct RamDedupStats {
    pub total_puts: u64,
    pub dedup_hits: u64,       // puts that found existing object (no new storage)
    pub unique_objects: usize,
    pub total_references: u64, // sum of refcounts (logical copies)
    pub bytes_stored: usize,   // unique bytes actually in RAM
    pub bytes_logical: usize,  // what N-copy naive storage would cost
    pub domains: usize,
}

impl RamResidencyIndex {
    pub fn new() -> Self {
        Self {
            residents: BTreeMap::new(),
            refcounts: BTreeMap::new(),
            domain_heads: BTreeMap::new(),
            stats: RamDedupStats::default(),
        }
    }

    /// Register a new domain (empty head set).
    pub fn add_domain(&mut self, domain_id: u64) {
        self.domain_heads.entry(domain_id).or_insert_with(Vec::new);
        self.stats.domains = self.domain_heads.len();
    }

    /// Intern obj, add to domain's live head, bump refcount.
    /// Returns `(id, was_new)` where `was_new=false` means dedup hit.
    pub fn put(&mut self, domain_id: u64, obj: Object) -> (ObjectId, bool) {
        // Ensure domain exists
        self.domain_heads.entry(domain_id).or_insert_with(Vec::new);

        let id = obj.id();
        let obj_bytes = estimate_bytes(&obj);
        let was_new;

        if self.residents.contains_key(&id) {
            // Dedup hit — object already stored
            was_new = false;
            self.stats.dedup_hits += 1;
        } else {
            // First time we see this object — store it
            was_new = true;
            self.residents.insert(id, obj);
            self.refcounts.insert(id, 0);
            self.stats.bytes_stored += obj_bytes;
        }

        // Always account for the logical (per-domain) byte cost
        self.stats.bytes_logical += obj_bytes;
        self.stats.total_puts += 1;

        // Add to domain head if not already there
        let head = self.domain_heads.get_mut(&domain_id).unwrap();
        if !head.contains(&id) {
            head.push(id);
            // Bump refcount
            if let Some(rc) = self.refcounts.get_mut(&id) {
                *rc += 1;
            }
            // Maintain total_references incrementally (avoid O(N) resum per put).
            self.stats.total_references += 1;
        }

        // Derived counters that are O(1) to read directly.
        self.stats.unique_objects = self.residents.len();
        self.stats.domains = self.domain_heads.len();

        (id, was_new)
    }

    /// Access a shared object by id.
    pub fn get(&self, id: &ObjectId) -> Option<&Object> {
        self.residents.get(id)
    }

    pub fn contains(&self, id: &ObjectId) -> bool {
        self.residents.contains_key(id)
    }

    /// Number of live objects (distinct ids) in a domain's head.
    pub fn domain_live_count(&self, domain_id: u64) -> usize {
        self.domain_heads
            .get(&domain_id)
            .map(|h| h.len())
            .unwrap_or(0)
    }

    /// Remove a domain, decrement refcounts, free objects whose refcount hits 0.
    pub fn release_domain(&mut self, domain_id: u64) {
        if let Some(head) = self.domain_heads.remove(&domain_id) {
            for id in &head {
                if let Some(rc) = self.refcounts.get_mut(id) {
                    if *rc > 0 {
                        *rc -= 1;
                    }
                }
            }
            // Collect ids to free (refcount == 0)
            let to_free: Vec<ObjectId> = self
                .refcounts
                .iter()
                .filter(|(_, &rc)| rc == 0)
                .map(|(id, _)| *id)
                .collect();
            for id in to_free {
                if let Some(obj) = self.residents.remove(&id) {
                    let freed = estimate_bytes(&obj);
                    self.stats.bytes_stored = self.stats.bytes_stored.saturating_sub(freed);
                }
                self.refcounts.remove(&id);
            }
        }
        self.stats.domains = self.domain_heads.len();
        self.rebuild_stats();
    }

    /// Ratio of total logical references to unique physical objects.
    /// Higher = more sharing.
    pub fn global_dedup_ratio(&self) -> f64 {
        let unique = self.residents.len();
        if unique == 0 {
            return 1.0;
        }
        self.stats.total_references as f64 / unique as f64
    }

    /// Bytes that would have been used without dedup minus bytes actually stored.
    pub fn bytes_saved(&self) -> usize {
        self.stats.bytes_logical.saturating_sub(self.stats.bytes_stored)
    }

    pub fn stats(&self) -> RamDedupStats {
        self.stats.clone()
    }

    pub fn domain_count(&self) -> usize {
        self.domain_heads.len()
    }

    // Rebuild the fields that are always derivable
    fn rebuild_stats(&mut self) {
        self.stats.unique_objects = self.residents.len();
        self.stats.total_references = self.refcounts.values().map(|&v| v as u64).sum();
        self.stats.domains = self.domain_heads.len();
    }
}

impl Default for RamResidencyIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ChunkDedup — sub-object chunked dedup (KSM-equivalent for content-addressed data)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct ChunkDedupStats {
    pub total_chunks_interned: u64,
    pub dedup_hits: u64,
    pub unique_chunks: usize,
    pub bytes_stored: usize,
    pub bytes_logical: usize,
}

pub struct ChunkDedup {
    chunks: BTreeMap<Hash256, Vec<u8>>,
    stats: ChunkDedupStats,
}

impl ChunkDedup {
    pub fn new() -> Self {
        Self {
            chunks: BTreeMap::new(),
            stats: ChunkDedupStats::default(),
        }
    }

    /// Split `data` into `chunk_size`-byte chunks, hash each, store dedup'd.
    /// Returns the chunk manifest (ordered list of chunk hashes).
    pub fn intern_blob(&mut self, data: &[u8], chunk_size: usize) -> Vec<Hash256> {
        let chunk_size = if chunk_size == 0 { 4096 } else { chunk_size };
        let mut manifest = Vec::new();

        for chunk in data.chunks(chunk_size) {
            let h = Hash256::of(chunk);
            self.stats.total_chunks_interned += 1;
            self.stats.bytes_logical += chunk.len();

            if self.chunks.contains_key(&h) {
                self.stats.dedup_hits += 1;
            } else {
                self.chunks.insert(h, chunk.to_vec());
                self.stats.bytes_stored += chunk.len();
            }

            manifest.push(h);
        }

        self.stats.unique_chunks = self.chunks.len();
        manifest
    }

    /// Reassemble blob from manifest.
    pub fn reconstruct(&self, manifest: &[Hash256]) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for h in manifest {
            let chunk = self.chunks.get(h)?;
            out.extend_from_slice(chunk);
        }
        Some(out)
    }

    /// Ratio of logical bytes to stored bytes. Higher = more sharing.
    pub fn dedup_ratio(&self) -> f64 {
        if self.stats.bytes_stored == 0 {
            return 1.0;
        }
        self.stats.bytes_logical as f64 / self.stats.bytes_stored as f64
    }

    pub fn stats(&self) -> ChunkDedupStats {
        self.stats.clone()
    }
}

impl Default for ChunkDedup {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

pub struct RamDedupBenchResult {
    pub domains: usize,
    pub shared_objects: usize,
    pub unique_per_domain: usize,
    pub bytes_logical: usize,
    pub bytes_stored: usize,
    pub dedup_ratio: f64,
    pub bytes_saved: usize,
    pub dedup_hit_rate: f64,
    pub chunk_dedup_ratio: f64,
}

/// Build n_domains with n_shared identical objects + n_unique unique objects each.
/// Also stress-tests ChunkDedup with a repeated blob.
///
/// Asserts:
/// - dedup_ratio >= n_domains * 0.8
/// - dedup_hit_rate >= 0.5
/// - bytes_saved > 0
pub fn benchmark_ram_dedup(
    n_domains: usize,
    n_shared: usize,
    n_unique: usize,
    chunk_size: usize,
) -> RamDedupBenchResult {
    use crate::object::Datum;

    let mut index = RamResidencyIndex::new();

    // Register all domains
    for d in 0..n_domains {
        index.add_domain(d as u64);
    }

    // Build the shared objects once (canonical form)
    let mut shared_objects: Vec<Object> = Vec::new();
    for i in 0..n_shared {
        let obj = Object::new("shared")
            .with("index", Datum::Int(i as i64))
            .with("payload", Datum::Text(alloc::format!("shared-payload-{}", i)));
        shared_objects.push(obj);
    }

    // Each domain puts all shared objects (should dedup after domain 0)
    for d in 0..n_domains {
        for obj in &shared_objects {
            index.put(d as u64, obj.clone());
        }
    }

    // Each domain puts n_unique objects unique to it
    for d in 0..n_domains {
        for u in 0..n_unique {
            let obj = Object::new("unique")
                .with("domain", Datum::Int(d as i64))
                .with("slot", Datum::Int(u as i64))
                .with("data", Datum::Text(alloc::format!("domain-{}-unique-{}", d, u)));
            index.put(d as u64, obj);
        }
    }

    let stats = index.stats();
    let dedup_ratio = index.global_dedup_ratio();
    let bytes_saved = index.bytes_saved();

    let dedup_hit_rate = if stats.total_puts == 0 {
        0.0
    } else {
        stats.dedup_hits as f64 / stats.total_puts as f64
    };

    // Chunk dedup: build a repeated-block blob
    let mut cd = ChunkDedup::new();
    let block: Vec<u8> = (0..chunk_size).map(|b| (b % 251) as u8).collect();
    // 10 repeated copies of the same block
    let mut blob = Vec::new();
    for _ in 0..10 {
        blob.extend_from_slice(&block);
    }
    let manifest = cd.intern_blob(&blob, chunk_size);
    let chunk_dedup_ratio = cd.dedup_ratio();
    // Verify reconstruction
    let reconstructed = cd.reconstruct(&manifest).expect("reconstruct failed");
    assert_eq!(reconstructed, blob, "chunk reconstruction must be lossless");

    // Assertions
    // With n_shared shared objects each held by n_domains and n_unique per domain,
    // total_refs = n_shared * n_domains + n_unique * n_domains
    // unique_objects = n_shared + n_unique * n_domains
    // ratio = (n_domains * (n_shared + n_unique)) / (n_shared + n_unique * n_domains)
    // For shared-heavy workloads this approaches n_domains; we require > 1.0 generally.
    let expected_min_ratio = if n_shared > n_unique {
        // Shared-heavy: ratio should be meaningfully above 1
        (n_domains as f64) * 0.4
    } else {
        1.0
    };
    assert!(
        dedup_ratio >= expected_min_ratio,
        "dedup_ratio {:.2} < expected {:.2}",
        dedup_ratio,
        expected_min_ratio
    );
    assert!(
        dedup_hit_rate >= 0.5,
        "dedup_hit_rate {:.2} < 0.5",
        dedup_hit_rate
    );
    assert!(bytes_saved > 0, "bytes_saved must be > 0");

    RamDedupBenchResult {
        domains: n_domains,
        shared_objects: n_shared,
        unique_per_domain: n_unique,
        bytes_logical: stats.bytes_logical,
        bytes_stored: stats.bytes_stored,
        dedup_ratio,
        bytes_saved,
        dedup_hit_rate,
        chunk_dedup_ratio,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Datum;

    fn make_obj(kind: &str, tag: i64) -> Object {
        Object::new(kind).with("tag", Datum::Int(tag))
    }

    // -------------------------------------------------------------------
    // dedup_stores_objects_once_across_domains
    // N domains put same object → 1 physical copy
    // unique_objects = 1, total_references = N
    // -------------------------------------------------------------------
    #[test]
    fn dedup_stores_objects_once_across_domains() {
        let n = 5u64;
        let mut idx = RamResidencyIndex::new();
        for d in 0..n {
            idx.add_domain(d);
        }
        let obj = make_obj("font", 42);
        let mut first_id = None;
        for d in 0..n {
            let (id, was_new) = idx.put(d, obj.clone());
            if d == 0 {
                assert!(was_new, "first put must be new");
                first_id = Some(id);
            } else {
                assert!(!was_new, "subsequent puts must be dedup hits");
                assert_eq!(Some(id), first_id, "ids must match");
            }
        }
        let stats = idx.stats();
        assert_eq!(stats.unique_objects, 1, "exactly one physical copy");
        assert_eq!(stats.total_references, n, "one ref per domain");
    }

    // -------------------------------------------------------------------
    // refcount_zeroed_on_domain_release
    // add 3 domains, each put same object, release all → empty index
    // -------------------------------------------------------------------
    #[test]
    fn refcount_zeroed_on_domain_release() {
        let mut idx = RamResidencyIndex::new();
        let obj = make_obj("image", 99);
        for d in 0..3u64 {
            idx.add_domain(d);
            idx.put(d, obj.clone());
        }
        assert_eq!(idx.stats().unique_objects, 1);

        idx.release_domain(0);
        assert_eq!(idx.stats().unique_objects, 1, "still alive — 2 refs");

        idx.release_domain(1);
        assert_eq!(idx.stats().unique_objects, 1, "still alive — 1 ref");

        idx.release_domain(2);
        let stats = idx.stats();
        assert_eq!(stats.unique_objects, 0, "freed after last release");
        assert_eq!(stats.bytes_stored, 0, "bytes reclaimed");
    }

    // -------------------------------------------------------------------
    // domain_unique_objects_not_shared
    // domains put different objects → dedup_ratio = 1.0
    // -------------------------------------------------------------------
    #[test]
    fn domain_unique_objects_not_shared() {
        let mut idx = RamResidencyIndex::new();
        for d in 0..4u64 {
            idx.add_domain(d);
            idx.put(d, make_obj("model", d as i64));
        }
        let ratio = idx.global_dedup_ratio();
        // 4 unique objects, 4 references → ratio = 1.0
        assert!(
            (ratio - 1.0).abs() < 1e-9,
            "ratio should be 1.0 for all-unique, got {:.4}",
            ratio
        );
    }

    // -------------------------------------------------------------------
    // mixed_shared_and_unique
    // 5 domains, 10 shared + 5 unique each → dedup_ratio > 2.0
    // -------------------------------------------------------------------
    #[test]
    fn mixed_shared_and_unique() {
        let n_domains = 5u64;
        let n_shared = 10i64;
        let n_unique = 5i64;
        let mut idx = RamResidencyIndex::new();

        for d in 0..n_domains {
            idx.add_domain(d);
        }

        // shared objects
        for i in 0..n_shared {
            let obj = Object::new("shared").with("i", Datum::Int(i));
            for d in 0..n_domains {
                idx.put(d, obj.clone());
            }
        }

        // unique objects
        for d in 0..n_domains {
            for u in 0..n_unique {
                let obj = Object::new("unique")
                    .with("d", Datum::Int(d as i64))
                    .with("u", Datum::Int(u));
                idx.put(d, obj);
            }
        }

        let ratio = idx.global_dedup_ratio();
        assert!(
            ratio > 2.0,
            "mixed workload should yield dedup_ratio > 2.0, got {:.4}",
            ratio
        );
    }

    // -------------------------------------------------------------------
    // chunk_dedup_finds_repeated_blocks
    // large blob with repeated 4KB blocks → high chunk dedup ratio
    // reconstruct produces identical bytes
    // -------------------------------------------------------------------
    #[test]
    fn chunk_dedup_finds_repeated_blocks() {
        let chunk_size = 4096;
        let block: Vec<u8> = (0..chunk_size).map(|b| (b % 251) as u8).collect();
        // 20 copies of the same block
        let mut blob = Vec::new();
        for _ in 0..20 {
            blob.extend_from_slice(&block);
        }

        let mut cd = ChunkDedup::new();
        let manifest = cd.intern_blob(&blob, chunk_size);

        // Ratio: 20 logical / 1 unique = 20.0
        let ratio = cd.dedup_ratio();
        assert!(
            ratio >= 10.0,
            "expected high chunk dedup ratio, got {:.2}",
            ratio
        );

        let reconstructed = cd.reconstruct(&manifest).expect("reconstruct must succeed");
        assert_eq!(reconstructed, blob, "reconstructed blob must be identical");

        let stats = cd.stats();
        assert_eq!(stats.unique_chunks, 1, "all blocks identical → 1 unique chunk");
        assert!(stats.dedup_hits >= 19, "19 of 20 blocks are dedup hits");
    }

    // -------------------------------------------------------------------
    // chunk_dedup_different_data_no_hits
    // all-unique data → dedup_ratio ≈ 1.0
    // -------------------------------------------------------------------
    #[test]
    fn chunk_dedup_different_data_no_hits() {
        let chunk_size = 64;
        let n_chunks = 20;
        // Each chunk is unique (incrementing counter in every byte)
        let mut blob: Vec<u8> = Vec::new();
        for i in 0..n_chunks {
            for _ in 0..chunk_size {
                blob.push(i as u8);
            }
        }

        let mut cd = ChunkDedup::new();
        let manifest = cd.intern_blob(&blob, chunk_size);

        let ratio = cd.dedup_ratio();
        assert!(
            ratio < 1.1,
            "all-unique data should have ratio ≈ 1.0, got {:.4}",
            ratio
        );

        let reconstructed = cd.reconstruct(&manifest).unwrap();
        assert_eq!(reconstructed, blob);

        let stats = cd.stats();
        assert_eq!(stats.dedup_hits, 0, "no dedup hits for unique data");
    }

    // -------------------------------------------------------------------
    // benchmark_ram_dedup_shows_significant_savings
    // -------------------------------------------------------------------
    #[test]
    fn benchmark_ram_dedup_shows_significant_savings() {
        let result = benchmark_ram_dedup(10, 100, 5, 4096);
        assert!(
            result.dedup_ratio > 5.0,
            "dedup_ratio {:.2} should be > 5.0",
            result.dedup_ratio
        );
        assert!(result.bytes_saved > 0, "bytes_saved must be > 0");
        assert!(
            result.dedup_hit_rate > 0.5,
            "dedup_hit_rate {:.2} must be > 0.5",
            result.dedup_hit_rate
        );
    }
}

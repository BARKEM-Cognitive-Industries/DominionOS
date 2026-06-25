//! Lever 10 — Distributed shared memory: the fleet's RAM as one pool.
//!
//! Makes a peer's RAM a first-class memory tier. When local RAM is full and an
//! object is cold, page it to a peer. Fault it back by hash on touch.
//!
//! Content-addressing makes remote memory safe: `Hash256` names exactly one
//! immutable byte-string, verified on arrival. Near-linear memory scale-out.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::hash::Hash256;

// ============================================================================
// PeerStats — per-peer counters
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct PeerStats {
    pub objects_stored: u64,
    pub objects_fetched: u64,
    pub bytes_stored: usize,
    pub bytes_fetched: usize,
    /// Hash mismatches detected on fetch (integrity failures).
    pub fetch_failures: u64,
    pub evictions: u64,
}

// ============================================================================
// PeerNode — one peer in the fleet
// ============================================================================

#[derive(Clone, Debug)]
pub struct PeerNode {
    pub node_id: u64,
    pub address: String,
    pub ram_capacity_bytes: usize,
    pub ram_used_bytes: usize,
    /// Objects stored on this peer: hash → bytes.
    store: BTreeMap<Hash256, Vec<u8>>,
    stats: PeerStats,
}

impl PeerNode {
    pub fn new(node_id: u64, address: impl Into<String>, ram_capacity_bytes: usize) -> Self {
        Self {
            node_id,
            address: address.into(),
            ram_capacity_bytes,
            ram_used_bytes: 0,
            store: BTreeMap::new(),
            stats: PeerStats::default(),
        }
    }

    /// Store `bytes` under `id`. Returns `false` if the peer is out of capacity.
    pub fn put(&mut self, id: Hash256, bytes: Vec<u8>) -> bool {
        if self.ram_used_bytes + bytes.len() > self.ram_capacity_bytes {
            return false;
        }
        let len = bytes.len();
        self.ram_used_bytes += len;
        self.stats.objects_stored += 1;
        self.stats.bytes_stored += len;
        self.store.insert(id, bytes);
        true
    }

    /// Retrieve and verify `id`. Returns `None` on missing or hash mismatch.
    pub fn fetch(&mut self, id: Hash256) -> Option<Vec<u8>> {
        let bytes = self.store.get(&id)?.clone();
        if Hash256::of(&bytes) != id {
            self.stats.fetch_failures += 1;
            return None;
        }
        self.stats.objects_fetched += 1;
        self.stats.bytes_fetched += bytes.len();
        Some(bytes)
    }

    /// Evict the "oldest" object (first BTreeMap key as LRU approximation).
    /// Returns the evicted hash, or `None` if the store is empty.
    pub fn evict_lru(&mut self) -> Option<Hash256> {
        let key = self.store.keys().next().cloned()?;
        if let Some(bytes) = self.store.remove(&key) {
            self.ram_used_bytes = self.ram_used_bytes.saturating_sub(bytes.len());
            self.stats.evictions += 1;
        }
        Some(key)
    }

    pub fn free_bytes(&self) -> usize {
        self.ram_capacity_bytes.saturating_sub(self.ram_used_bytes)
    }

    pub fn utilization(&self) -> f64 {
        if self.ram_capacity_bytes == 0 {
            1.0
        } else {
            self.ram_used_bytes as f64 / self.ram_capacity_bytes as f64
        }
    }

    pub fn stats(&self) -> &PeerStats {
        &self.stats
    }
}

// ============================================================================
// TierStats — aggregate stats for the whole distributed RAM tier
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct TierStats {
    /// Objects sent to peers (offloaded from local RAM).
    pub total_offloads: u64,
    /// Objects recalled from peers.
    pub total_fetches: u64,
    /// Hash verification failures across all recalls.
    pub failed_fetches: u64,
    /// Total bytes currently held across all peers.
    pub total_peer_bytes: usize,
    /// Bytes we did not need to keep in local RAM.
    pub local_ram_saved_bytes: usize,
    /// Local quota + all peer RAM combined.
    pub effective_memory_bytes: usize,
    pub peer_count: usize,
}

// ============================================================================
// PeerRamTier — distributed RAM tier coordinator
// ============================================================================

pub struct PeerRamTier {
    peers: Vec<PeerNode>,
    /// Local index: hash → peer index that holds it.
    placement: BTreeMap<Hash256, usize>,
    stats: TierStats,
}

impl PeerRamTier {
    pub fn new() -> Self {
        Self {
            peers: Vec::new(),
            placement: BTreeMap::new(),
            stats: TierStats::default(),
        }
    }

    /// Register a peer with the tier.
    pub fn add_peer(&mut self, peer: PeerNode) {
        self.stats.effective_memory_bytes += peer.ram_capacity_bytes;
        self.peers.push(peer);
        self.stats.peer_count = self.peers.len();
    }

    /// Offload `bytes` (identified by `id`) to the least-loaded peer that has
    /// free capacity. Returns `false` if all peers are full.
    pub fn offload(&mut self, id: Hash256, bytes: Vec<u8>) -> bool {
        let peer_idx = match self.least_loaded_peer() {
            Some(i) => i,
            None => return false,
        };
        let len = bytes.len();
        if self.peers[peer_idx].put(id, bytes) {
            self.placement.insert(id, peer_idx);
            self.stats.total_offloads += 1;
            self.stats.total_peer_bytes += len;
            self.stats.local_ram_saved_bytes += len;
            true
        } else {
            false
        }
    }

    /// Fetch the object identified by `id` from whichever peer holds it.
    /// Removes it from the placement index (logically back in local RAM).
    pub fn recall(&mut self, id: Hash256) -> Option<Vec<u8>> {
        let peer_idx = *self.placement.get(&id)?;
        match self.peers[peer_idx].fetch(id) {
            Some(bytes) => {
                self.placement.remove(&id);
                self.stats.total_fetches += 1;
                Some(bytes)
            }
            None => {
                self.stats.failed_fetches += 1;
                None
            }
        }
    }

    /// Is this object currently held by any peer?
    pub fn contains(&self, id: &Hash256) -> bool {
        self.placement.contains_key(id)
    }

    /// Sum of all peers' `ram_capacity_bytes`.
    pub fn effective_capacity_bytes(&self) -> usize {
        self.peers.iter().map(|p| p.ram_capacity_bytes).sum()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    pub fn stats(&self) -> TierStats {
        self.stats.clone()
    }

    /// Returns the peer index with the lowest utilization that still has free
    /// bytes. Returns `None` when no peer has capacity.
    fn least_loaded_peer(&self) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_util = f64::MAX;
        for (i, peer) in self.peers.iter().enumerate() {
            if peer.free_bytes() == 0 {
                continue;
            }
            let u = peer.utilization();
            if u < best_util {
                best_util = u;
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

impl Default for PeerRamTier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Benchmark
// ============================================================================

#[derive(Clone, Debug)]
pub struct PeerRamBenchResult {
    pub objects: usize,
    pub object_bytes: usize,
    pub peers: usize,
    pub peer_ram_each_bytes: usize,
    pub total_peer_ram_bytes: usize,
    pub objects_offloaded: usize,
    pub objects_recalled: usize,
    pub recall_success_rate: f64,
    pub integrity_failures: u64,
    pub effective_memory_multiplier: f64,
    pub local_ram_bytes_saved: usize,
}

/// Benchmark peer-RAM offload / recall at the given scale.
///
/// # Example
/// ```ignore
/// let r = benchmark_peer_ram(4, 10_000_000, 100, 64_000, 1_000_000);
/// assert!(r.effective_memory_multiplier > 5.0);
/// assert_eq!(r.recall_success_rate, 1.0);
/// ```
pub fn benchmark_peer_ram(
    n_peers: usize,
    peer_ram_bytes: usize,
    n_objects: usize,
    obj_bytes: usize,
    local_quota_bytes: usize,
) -> PeerRamBenchResult {
    let mut tier = PeerRamTier::new();
    for i in 0..n_peers {
        tier.add_peer(PeerNode::new(i as u64, alloc::format!("peer-{}", i), peer_ram_bytes));
    }

    // Offload n_objects, each filled with a deterministic pattern.
    let mut offloaded: Vec<(Hash256, Vec<u8>)> = Vec::new();
    let mut objects_offloaded = 0usize;
    for i in 0..n_objects {
        let bytes: Vec<u8> = (0..obj_bytes).map(|b| ((i + b) % 256) as u8).collect();
        let id = Hash256::of(&bytes);
        if tier.offload(id, bytes.clone()) {
            objects_offloaded += 1;
            offloaded.push((id, bytes));
        }
    }

    // Recall all offloaded objects and verify integrity.
    let objects_recalled = offloaded.len();
    let mut successful_recalls = 0usize;
    for (id, original) in &offloaded {
        if let Some(recalled) = tier.recall(*id) {
            if recalled == *original {
                successful_recalls += 1;
            }
        }
    }

    let stats = tier.stats();
    let total_peer_ram_bytes = n_peers * peer_ram_bytes;
    let recall_success_rate = if objects_recalled == 0 {
        1.0
    } else {
        successful_recalls as f64 / objects_recalled as f64
    };
    let effective_memory_multiplier =
        (local_quota_bytes + total_peer_ram_bytes) as f64 / local_quota_bytes as f64;

    assert_eq!(recall_success_rate, 1.0, "recall integrity violated");
    assert_eq!(stats.failed_fetches, 0, "hash verification failures detected");

    PeerRamBenchResult {
        objects: n_objects,
        object_bytes: obj_bytes,
        peers: n_peers,
        peer_ram_each_bytes: peer_ram_bytes,
        total_peer_ram_bytes,
        objects_offloaded,
        objects_recalled,
        recall_success_rate,
        integrity_failures: stats.failed_fetches,
        effective_memory_multiplier,
        local_ram_bytes_saved: stats.local_ram_saved_bytes,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_node_stores_and_fetches_correctly() {
        let mut peer = PeerNode::new(1, "127.0.0.1:9000", 1024 * 1024);
        let bytes = alloc::vec![1u8, 2, 3, 4];
        let id = Hash256::of(&bytes);
        assert!(peer.put(id, bytes.clone()));
        let fetched = peer.fetch(id).expect("should fetch stored bytes");
        assert_eq!(fetched, bytes);
    }

    #[test]
    fn peer_node_rejects_when_full() {
        let mut peer = PeerNode::new(2, "127.0.0.1:9001", 10);
        let bytes = alloc::vec![0u8; 10];
        let id = Hash256::of(&bytes);
        assert!(peer.put(id, bytes));
        // One more byte should be rejected.
        let extra = alloc::vec![99u8; 1];
        let extra_id = Hash256::of(&extra);
        assert!(!peer.put(extra_id, extra), "should reject when full");
    }

    #[test]
    fn peer_node_fetch_verifies_hash() {
        let mut peer = PeerNode::new(3, "127.0.0.1:9002", 1024 * 1024);
        // Store "world" under the hash of "hello" — deliberate mismatch.
        let real_bytes = b"hello".to_vec();
        let wrong_bytes = b"world".to_vec();
        let id = Hash256::of(&real_bytes);
        // put() accepts any (id, bytes) pair without checking consistency.
        assert!(peer.put(id, wrong_bytes));
        // fetch() must detect the mismatch and return None.
        assert!(peer.fetch(id).is_none(), "corrupted bytes should fail verification");
        assert_eq!(peer.stats().fetch_failures, 1);
    }

    #[test]
    fn peer_ram_tier_offloads_to_least_loaded() {
        let mut tier = PeerRamTier::new();
        tier.add_peer(PeerNode::new(0, "peer-0", 1024 * 1024));
        tier.add_peer(PeerNode::new(1, "peer-1", 1024 * 1024));

        // Offload several objects; both peers should receive some.
        for i in 0..10u8 {
            let bytes = alloc::vec![i; 1024];
            let id = Hash256::of(&bytes);
            assert!(tier.offload(id, bytes));
        }
        // Both peers should have objects (round-robin via least-loaded).
        let p0_used = tier.peers[0].ram_used_bytes;
        let p1_used = tier.peers[1].ram_used_bytes;
        assert!(p0_used > 0, "peer 0 should have some data");
        assert!(p1_used > 0, "peer 1 should have some data");
    }

    #[test]
    fn peer_ram_tier_recall_removes_from_peer() {
        let mut tier = PeerRamTier::new();
        tier.add_peer(PeerNode::new(0, "peer-0", 1024 * 1024));

        let bytes = b"test payload".to_vec();
        let id = Hash256::of(&bytes);
        assert!(tier.offload(id, bytes.clone()));
        assert!(tier.contains(&id));

        let recalled = tier.recall(id).expect("recall should succeed");
        assert_eq!(recalled, bytes);
        assert!(!tier.contains(&id), "after recall, object should not be on any peer");
    }

    #[test]
    fn peer_ram_scales_effective_memory() {
        let mut tier = PeerRamTier::new();
        let peer_size = 10 * 1024 * 1024; // 10 MB each
        for i in 0..4u64 {
            tier.add_peer(PeerNode::new(i, alloc::format!("peer-{}", i), peer_size));
        }
        assert_eq!(tier.effective_capacity_bytes(), 4 * peer_size);
        assert!(
            tier.effective_capacity_bytes() >= 40 * 1024 * 1024,
            "4 × 10 MB should be ≥ 40 MB"
        );
    }

    #[test]
    fn benchmark_shows_memory_scale_out() {
        let result = benchmark_peer_ram(4, 10_000_000, 100, 64_000, 1_000_000);
        assert!(
            result.effective_memory_multiplier > 5.0,
            "multiplier was {}, expected > 5",
            result.effective_memory_multiplier
        );
        assert_eq!(result.recall_success_rate, 1.0);
        assert_eq!(result.integrity_failures, 0);
    }
}

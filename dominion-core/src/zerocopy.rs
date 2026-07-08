//! Zero-copy buffer pool and I/O pipeline (L3 memory-acceleration roadmap).
//!
//! Data is interned once by content hash. All consumers share the same
//! physical bytes via a [`ZeroCopyHandle`]. Handle passing costs 32 bytes;
//! byte copying costs O(data_size). Together with [`ZcpPipeline`] this
//! eliminates the "copy tax" on I/O-bound and IPC-bound paths.

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// ── Handle ────────────────────────────────────────────────────────────────────

/// An opaque, copy-cheap token representing bytes stored in a [`ZeroCopyPool`].
///
/// Sending this 32-byte value to another consumer is equivalent to sending the
/// underlying data without copying a single byte.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ZeroCopyHandle(pub Hash256);

impl From<Hash256> for ZeroCopyHandle {
    fn from(h: Hash256) -> Self {
        ZeroCopyHandle(h)
    }
}

impl From<ZeroCopyHandle> for Hash256 {
    fn from(h: ZeroCopyHandle) -> Self {
        h.0
    }
}

// ── Stats ─────────────────────────────────────────────────────────────────────

/// Accumulated statistics for a [`ZeroCopyPool`].
#[derive(Clone, Debug, Default)]
pub struct ZcpStats {
    /// Total number of intern calls (including dedup hits).
    pub total_interns: u64,
    /// Interns that found identical bytes already stored (no allocation).
    pub dedup_hits: u64,
    /// Number of times a handle was passed to an additional consumer.
    pub total_handle_passes: u64,
    /// Physical bytes currently held in the pool.
    pub bytes_stored: usize,
    /// Logical bytes all references would occupy if each were a private copy.
    pub bytes_logical: usize,
    /// Handle passes that replaced a would-be byte copy.
    pub copies_eliminated: u64,
}

// ── Pool ─────────────────────────────────────────────────────────────────────

/// A content-addressed zero-copy buffer pool.
///
/// Data is interned once; thereafter every consumer receives a
/// [`ZeroCopyHandle`] rather than a byte copy. Reference-counting ensures
/// buffers are freed when no live handle remains.
pub struct ZeroCopyPool {
    buffers: BTreeMap<Hash256, Vec<u8>>,
    refcounts: BTreeMap<Hash256, u32>,
    stats: ZcpStats,
}

impl ZeroCopyPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            refcounts: BTreeMap::new(),
            stats: ZcpStats::default(),
        }
    }

    /// Store `data` by content hash and return a handle.
    ///
    /// If identical bytes are already present the existing buffer is reused
    /// (dedup hit) and no allocation occurs.
    pub fn intern(&mut self, data: Vec<u8>) -> ZeroCopyHandle {
        let hash = Hash256::of(&data);
        self.stats.total_interns += 1;
        self.stats.bytes_logical += data.len();
        if self.buffers.contains_key(&hash) {
            self.stats.dedup_hits += 1;
            *self.refcounts.entry(hash).or_insert(0) += 1;
        } else {
            self.stats.bytes_stored += data.len();
            self.buffers.insert(hash, data);
            self.refcounts.insert(hash, 1);
        }
        ZeroCopyHandle(hash)
    }

    /// Convenience wrapper — intern from a slice (one allocation).
    pub fn intern_ref(&mut self, data: &[u8]) -> ZeroCopyHandle {
        self.intern(data.to_vec())
    }

    /// Zero-copy read: return a shared reference to the stored bytes.
    pub fn get(&self, handle: ZeroCopyHandle) -> Option<&[u8]> {
        self.buffers.get(&handle.0).map(|v| v.as_slice())
    }

    /// Model an IPC handle-pass: record that another consumer received this
    /// handle (not a byte copy) and return the cloned handle.
    ///
    /// The new consumer now holds a live reference, so the refcount is bumped
    /// (as [`retain`](Self::retain) would): the buffer is only freed once every
    /// consumer has released its handle.
    ///
    /// Returns `None` if the handle does not exist in the pool.
    pub fn pass_handle(&mut self, handle: ZeroCopyHandle) -> Option<ZeroCopyHandle> {
        if self.buffers.contains_key(&handle.0) {
            if let Some(rc) = self.refcounts.get_mut(&handle.0) {
                *rc += 1;
            }
            self.stats.total_handle_passes += 1;
            self.stats.copies_eliminated += 1;
            Some(handle)
        } else {
            None
        }
    }

    /// Increment the reference count for `handle` (new owner acquiring a copy).
    pub fn retain(&mut self, handle: ZeroCopyHandle) {
        if let Some(rc) = self.refcounts.get_mut(&handle.0) {
            *rc += 1;
        }
    }

    /// Decrement the reference count; free the buffer when it reaches zero.
    pub fn release(&mut self, handle: ZeroCopyHandle) {
        if let Some(rc) = self.refcounts.get_mut(&handle.0) {
            if *rc <= 1 {
                self.refcounts.remove(&handle.0);
                if let Some(data) = self.buffers.remove(&handle.0) {
                    self.stats.bytes_stored =
                        self.stats.bytes_stored.saturating_sub(data.len());
                }
            } else {
                *rc -= 1;
            }
        }
    }

    /// Snapshot of current statistics.
    pub fn stats(&self) -> ZcpStats {
        self.stats.clone()
    }

    /// How many byte-copy operations were avoided by handle-passing.
    pub fn copies_eliminated_vs_naive(&self) -> u64 {
        self.stats.copies_eliminated
    }
}

impl Default for ZeroCopyPool {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Accumulated statistics for a [`ZcpPipeline`].
#[derive(Clone, Debug, Default)]
pub struct PipelineStats {
    /// Total messages sent through the pipeline.
    pub messages_sent: u64,
    /// Bytes × consumers — what a naive copy-based design would allocate.
    pub total_bytes_if_copied: usize,
    /// Bytes actually stored (unique payloads only, dedup-collapsed).
    pub total_bytes_actually_stored: usize,
    /// `total_bytes_if_copied / total_bytes_actually_stored`.
    pub copy_reduction_factor: f64,
}

/// A zero-copy I/O pipeline: one producer interns data once; `consumer_count`
/// consumers each receive a [`ZeroCopyHandle`] rather than a byte copy.
pub struct ZcpPipeline {
    pool: ZeroCopyPool,
    consumer_count: usize,
    stats: PipelineStats,
}

impl ZcpPipeline {
    /// Create a pipeline with `consumer_count` downstream receivers.
    pub fn new(consumer_count: usize) -> Self {
        Self {
            pool: ZeroCopyPool::new(),
            consumer_count,
            stats: PipelineStats::default(),
        }
    }

    /// Intern `data` once, hand each consumer a handle.
    ///
    /// If the payload is identical to one already stored (dedup hit) no new
    /// allocation occurs. Handles are still issued to all consumers.
    pub fn send(&mut self, data: Vec<u8>) -> Vec<ZeroCopyHandle> {
        let len = data.len();
        self.stats.messages_sent += 1;
        self.stats.total_bytes_if_copied += len * self.consumer_count;

        let handle = self.pool.intern(data);

        // bytes_actually_stored reflects what is physically in the pool now.
        self.stats.total_bytes_actually_stored = self.pool.stats().bytes_stored;

        if self.stats.total_bytes_actually_stored > 0 {
            self.stats.copy_reduction_factor = self.stats.total_bytes_if_copied as f64
                / self.stats.total_bytes_actually_stored as f64;
        }

        let mut handles = Vec::with_capacity(self.consumer_count);
        for _ in 0..self.consumer_count {
            // pass_handle records the avoided copy; fall back to raw handle on
            // the (impossible-in-practice) None branch.
            handles.push(self.pool.pass_handle(handle).unwrap_or(handle));
        }
        handles
    }

    /// Zero-copy read by a consumer — no bytes are duplicated.
    pub fn receive(&self, handle: ZeroCopyHandle) -> Option<&[u8]> {
        self.pool.get(handle)
    }

    /// Snapshot of pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        self.stats.clone()
    }
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

/// Results from [`benchmark_zero_copy`].
pub struct ZcoBenchResult {
    pub producers: usize,
    pub consumers: usize,
    pub messages: usize,
    pub payload_bytes: usize,
    /// What a fully copying design would have allocated.
    pub bytes_if_all_copied: usize,
    /// What the zero-copy pool actually stores.
    pub bytes_actually_stored: usize,
    /// `bytes_if_all_copied / bytes_actually_stored`.
    pub copy_reduction_factor: f64,
    /// Total handle-passes recorded across all pool operations.
    pub handles_passed: u64,
    /// Byte copies eliminated (same as `handles_passed` for this benchmark).
    pub copies_eliminated: u64,
}

/// Simulate `producers` × `messages` sends to `consumers` receivers with only
/// `unique_payloads` distinct byte patterns (the rest repeat and are dedup-ed).
///
/// Returns a [`ZcoBenchResult`] quantifying how much copying was avoided.
pub fn benchmark_zero_copy(
    producers: usize,
    consumers: usize,
    messages: usize,
    payload_bytes: usize,
    unique_payloads: usize,
) -> ZcoBenchResult {
    let mut pipeline = ZcpPipeline::new(consumers);

    // Guard the divisor: a caller passing 0 distinct payloads would otherwise
    // divide by zero below. At least one unique payload is always assumed.
    let unique_payloads = unique_payloads.max(1);

    for _p in 0..producers {
        for m in 0..messages {
            let payload_idx = m % unique_payloads;
            let data = alloc::vec![payload_idx as u8; payload_bytes];
            pipeline.send(data);
        }
    }

    let pool_stats = pipeline.pool.stats();
    let bytes_if_all_copied = producers * consumers * messages * payload_bytes;
    let bytes_actually_stored = pool_stats.bytes_stored;
    let copy_reduction_factor = if bytes_actually_stored > 0 {
        bytes_if_all_copied as f64 / bytes_actually_stored as f64
    } else {
        0.0
    };

    ZcoBenchResult {
        producers,
        consumers,
        messages,
        payload_bytes,
        bytes_if_all_copied,
        bytes_actually_stored,
        copy_reduction_factor,
        handles_passed: pool_stats.total_handle_passes,
        copies_eliminated: pool_stats.copies_eliminated,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_stores_once_dedup_on_repeat() {
        let mut pool = ZeroCopyPool::new();
        let data = alloc::vec![1u8, 2, 3, 4];
        let h1 = pool.intern(data.clone());
        let h2 = pool.intern(data);
        assert_eq!(h1, h2);
        let stats = pool.stats();
        assert_eq!(stats.total_interns, 2);
        assert_eq!(stats.dedup_hits, 1);
        // Only one distinct buffer should be stored.
        assert_eq!(pool.buffers.len(), 1);
    }

    #[test]
    fn get_returns_same_bytes() {
        let mut pool = ZeroCopyPool::new();
        let data = alloc::vec![10u8, 20, 30];
        let handle = pool.intern(data.clone());
        assert_eq!(pool.get(handle), Some(data.as_slice()));
    }

    #[test]
    fn handle_pass_eliminates_copies() {
        let mut pool = ZeroCopyPool::new();
        let data = alloc::vec![0u8; 1024];
        let handle = pool.intern(data);
        // Pass the handle to 5 consumers — no bytes are copied.
        for _ in 0..5 {
            pool.pass_handle(handle);
        }
        assert_eq!(pool.copies_eliminated_vs_naive(), 5);
    }

    #[test]
    fn pipeline_single_copy_many_consumers() {
        let mut pipeline = ZcpPipeline::new(8);
        // 100 distinct payloads → no dedup, but 8 consumers share each.
        for i in 0u8..100 {
            let data = alloc::vec![i; 512];
            pipeline.send(data);
        }
        let stats = pipeline.stats();
        // bytes_if_copied = 100 × 512 × 8 = 409 600
        // bytes_stored    = 100 × 512     = 51 200
        // ratio = 8.0
        assert!(
            stats.copy_reduction_factor >= 7.0,
            "expected >= 7.0, got {}",
            stats.copy_reduction_factor
        );
    }

    #[test]
    fn release_frees_memory() {
        let mut pool = ZeroCopyPool::new();
        let data = alloc::vec![42u8; 64];
        let handle = pool.intern(data); // refcount = 1
        pool.retain(handle); // refcount = 2
        pool.retain(handle); // refcount = 3
        pool.release(handle); // refcount = 2
        pool.release(handle); // refcount = 1
        pool.release(handle); // refcount = 0 → freed
        assert!(pool.get(handle).is_none());
    }

    #[test]
    fn benchmark_shows_significant_reduction() {
        let result = benchmark_zero_copy(4, 8, 100, 4096, 10);
        assert!(
            result.copy_reduction_factor > 3.0,
            "expected > 3.0, got {}",
            result.copy_reduction_factor
        );
        assert!(result.copies_eliminated > 0);
    }
}

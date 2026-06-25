//! Content-addressed tensor memoization — Lever 3 of the ML acceleration roadmap.
//!
//! The insight: DominionOS already stores everything content-addressed. ML tensors
//! should join that substrate. Once they do:
//!
//! * **KV-cache for free** — past keys/values are content-addressed objects; an
//!   autoregressive step reuses them by hash instead of recomputing.
//! * **Cross-batch / cross-epoch CSE** — identical `(weights × input)` subgraphs
//!   dedup automatically.
//! * **Incremental inference** — change one layer and only that layer recomputes;
//!   unchanged prefix/suffix layers return cached outputs instantly.
//!
//! This module is `no_std + alloc`, pure-safe, and zero external dependencies.
//! The cache is a `BTreeMap<(u64, u64), Tensor>` keyed by two 64-bit FNV-1a hashes
//! (layer/model hash × input hash). Lookups are O(log n) — fast for the small
//! per-model cache sizes typical in inference loops.

use alloc::collections::BTreeMap;
use crate::datatypes::Tensor;

/// A content-addressed memo table for tensor computations.
///
/// Keys are `(producer_hash, input_hash)` pairs where:
/// - `producer_hash` identifies the computation (layer weights, model, or any
///   deterministic function of the inputs that produced the output).
/// - `input_hash` is [`Tensor::content_hash`] of the computation's input.
///
/// Two calls with identical keys return the identical cached output — no
/// recomputation. Bit-exact determinism is required for cache correctness; the
/// default (non-FMA) path guarantees this.
pub struct TensorMemo {
    entries: BTreeMap<(u64, u64), Tensor>,
    hits:   u64,
    misses: u64,
}

impl Default for TensorMemo {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorMemo {
    pub fn new() -> Self {
        TensorMemo { entries: BTreeMap::new(), hits: 0, misses: 0 }
    }

    /// Look up `key`. Returns a reference to the cached output on hit; increments
    /// the hit counter. Returns `None` on miss; increments the miss counter.
    pub fn get(&mut self, key: (u64, u64)) -> Option<&Tensor> {
        if self.entries.contains_key(&key) {
            self.hits += 1;
            self.entries.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store `output` under `key`. Overwrites any previous entry.
    pub fn insert(&mut self, key: (u64, u64), output: Tensor) {
        self.entries.insert(key, output);
    }

    /// Insert with a memory-bounded capacity. When at or over `max_entries`, evicts
    /// one entry (pseudo-random: the lexicographically smallest key in the BTreeMap)
    /// before inserting. Vastly better than flush-on-overflow: the cache stays warm
    /// and evicts single entries rather than losing all cached work at once.
    pub fn insert_bounded(&mut self, key: (u64, u64), output: Tensor, max_entries: usize) {
        if max_entries == 0 { return; }
        if self.entries.len() >= max_entries {
            if let Some(&k) = self.entries.keys().next() {
                self.entries.remove(&k);
            }
        }
        self.entries.insert(key, output);
    }

    /// Number of cache hits since construction.
    pub fn hits(&self) -> u64 { self.hits }
    /// Number of cache misses since construction.
    pub fn misses(&self) -> u64 { self.misses }
    /// Number of distinct entries currently cached.
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    /// Evict all cached entries (weights changed, etc.).
    pub fn clear(&mut self) { self.entries.clear(); }

    /// Hit rate as a percentage (0–100), or 0 if no calls yet.
    pub fn hit_rate_pct(&self) -> u64 {
        let total = self.hits + self.misses;
        if total == 0 { 0 } else { self.hits * 100 / total }
    }
}

/// A capacity-bounded, LRU-evicting tensor memo cache.
///
/// Fixes the cache-pollution problem in unbounded caches: write-once "fresh"
/// inputs that never repeat are evicted first (they're the coldest entries)
/// rather than randomly evicting hot repeating entries.
///
/// Implementation: two BTreeMaps for O(log n) eviction.
/// - `entries`: `key → (Tensor, gen)` — the cached payload keyed by hash pair.
/// - `by_age`:  `(gen, key) → ()` — lets us find the LRU entry in O(log n).
///
/// `gen` is a monotonically-increasing logical clock; the smallest `gen` in
/// `by_age` is the entry that was accessed least recently → the LRU victim.
pub struct BoundedTensorMemo {
    entries:     BTreeMap<(u64, u64), (Tensor, u64)>,
    by_age:      BTreeMap<(u64, (u64, u64)), ()>,
    gen:         u64,
    max_entries: usize,
    hits:        u64,
    misses:      u64,
}

impl BoundedTensorMemo {
    pub fn new(max_entries: usize) -> Self {
        BoundedTensorMemo {
            entries: BTreeMap::new(),
            by_age: BTreeMap::new(),
            gen: 0,
            max_entries,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up `key`. On hit, promotes the entry to MRU and returns a reference.
    /// On miss, increments miss counter and returns `None`.
    pub fn get(&mut self, key: (u64, u64)) -> Option<&Tensor> {
        let old_gen = if let Some((_, g)) = self.entries.get(&key) {
            *g
        } else {
            self.misses += 1;
            return None;
        };
        // Promote to MRU.
        self.by_age.remove(&(old_gen, key));
        self.gen += 1;
        let new_gen = self.gen;
        if let Some(e) = self.entries.get_mut(&key) { e.1 = new_gen; }
        self.by_age.insert((new_gen, key), ());
        self.hits += 1;
        self.entries.get(&key).map(|(t, _)| t)
    }

    /// Insert `output` under `key`, evicting the LRU entry when at capacity.
    pub fn insert(&mut self, key: (u64, u64), output: Tensor) {
        if self.max_entries == 0 { return; }
        if self.entries.len() >= self.max_entries {
            // Evict the entry with the smallest gen (least recently used).
            if let Some(&(evict_gen, evict_key)) = self.by_age.keys().next() {
                self.by_age.remove(&(evict_gen, evict_key));
                self.entries.remove(&evict_key);
            }
        }
        self.gen += 1;
        let g = self.gen;
        self.by_age.insert((g, key), ());
        self.entries.insert(key, (output, g));
    }

    pub fn hits(&self)        -> u64  { self.hits }
    pub fn misses(&self)      -> u64  { self.misses }
    pub fn len(&self)         -> usize { self.entries.len() }
    pub fn is_empty(&self)    -> bool  { self.entries.is_empty() }

    /// Hit rate as a percentage (0–100), or 0 if no calls yet.
    pub fn hit_rate_pct(&self) -> u64 {
        let total = self.hits + self.misses;
        if total == 0 { 0 } else { self.hits * 100 / total }
    }
}

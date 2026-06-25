//! Attention mechanisms for transformer inference — KV-cache.
//!
//! All code is `no_std + alloc`, `#![forbid(unsafe_code)]`, zero external deps, and
//! bit-exact deterministic.

#![forbid(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;

use crate::math::exp;

// ────────────────────────────────────────────────────────────────────────────
// KV Cache
// ────────────────────────────────────────────────────────────────────────────

/// Incremental key/value cache for transformer decode loops.
///
/// Stores keys and values in flat row-major order:
/// `k_cache[pos * n_kv_heads * d_head + head * d_head + dim]`.
///
/// Call [`KvCache::append`] each decode step; call [`KvCache::k_slice`] /
/// [`KvCache::v_slice`] to expose the filled prefix to the attention kernel.
#[derive(Clone, Debug)]
pub struct KvCache {
    pub k_cache: Vec<f64>, // [max_seq, n_kv_heads, d_head]
    pub v_cache: Vec<f64>, // [max_seq, n_kv_heads, d_head]
    pub seq_len: usize,
    pub max_seq: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
}

impl KvCache {
    /// Allocate a cache that holds at most `max_seq` positions.
    pub fn new(max_seq: usize, n_kv_heads: usize, d_head: usize) -> Self {
        let cap = max_seq * n_kv_heads * d_head;
        KvCache {
            k_cache: vec![0.0f64; cap],
            v_cache: vec![0.0f64; cap],
            seq_len: 0,
            max_seq,
            n_kv_heads,
            d_head,
        }
    }

    /// Append one or more positions from flat slices `k` and `v`, each of length
    /// `n_new_positions * n_kv_heads * d_head`.  Returns `false` (and makes no
    /// change) if the cache would overflow or the slice lengths are inconsistent.
    pub fn append(&mut self, k: &[f64], v: &[f64]) -> bool {
        let stride = self.n_kv_heads * self.d_head;
        if stride == 0 {
            return false;
        }
        if k.len() != v.len() || k.len() % stride != 0 {
            return false;
        }
        let new_pos = k.len() / stride;
        if self.seq_len + new_pos > self.max_seq {
            return false;
        }
        let base = self.seq_len * stride;
        self.k_cache[base..base + k.len()].copy_from_slice(k);
        self.v_cache[base..base + v.len()].copy_from_slice(v);
        self.seq_len += new_pos;
        true
    }

    /// Reset the cache to empty (capacity is kept).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Live key data: `k_cache[..seq_len * n_kv_heads * d_head]`.
    pub fn k_slice(&self) -> &[f64] {
        let end = self.seq_len * self.n_kv_heads * self.d_head;
        &self.k_cache[..end]
    }

    /// Live value data: `v_cache[..seq_len * n_kv_heads * d_head]`.
    pub fn v_slice(&self) -> &[f64] {
        let end = self.seq_len * self.n_kv_heads * self.d_head;
        &self.v_cache[..end]
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FlashAttention (online-softmax, blocked) — used by the transformer block
// ────────────────────────────────────────────────────────────────────────────

/// Online softmax recurrence (per query row `i`, head `h`):
/// - `m` = running max, `ℓ` = running denominator, `O` = running numerator
/// - For each block of keys: update `m_new`, rescale `ℓ` and `O`, accumulate.
/// - Final output: `O / ℓ`.
///
/// Returns `None` on slice-length mismatches.  `block_size = 0` is treated as 1.
pub fn flash_attention_cpu(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    d_head: usize,
    scale: f64,
    causal: bool,
    block_size: usize,
) -> Option<Vec<f64>> {
    if q.len() != seq_q * n_heads * d_head {
        return None;
    }
    if k.len() != seq_k * n_heads * d_head {
        return None;
    }
    if v.len() != seq_k * n_heads * d_head {
        return None;
    }

    let bs = if block_size == 0 { 1 } else { block_size };
    let mut out = vec![0.0f64; seq_q * n_heads * d_head];

    for h in 0..n_heads {
        for i in 0..seq_q {
            let q_off = i * n_heads * d_head + h * d_head;
            let q_row = &q[q_off..q_off + d_head];

            // Running online-softmax state.
            let mut m = f64::NEG_INFINITY;
            let mut l = 0.0f64;
            let mut acc = vec![0.0f64; d_head];

            let mut j_start = 0usize;
            while j_start < seq_k {
                let j_end = (j_start + bs).min(seq_k);

                // Collect valid scores for this block.
                let mut block_s: Vec<f64> = Vec::with_capacity(j_end - j_start);
                for j in j_start..j_end {
                    if causal && j > i {
                        break; // masked; stop early for causal
                    }
                    let k_off = j * n_heads * d_head + h * d_head;
                    let k_row = &k[k_off..k_off + d_head];
                    let mut dot = 0.0f64;
                    for d in 0..d_head {
                        dot += q_row[d] * k_row[d];
                    }
                    block_s.push(dot * scale);
                }

                if block_s.is_empty() {
                    j_start = j_end;
                    continue;
                }

                // m_new = max(m, max(block_s))
                let mut block_max = f64::NEG_INFINITY;
                for &s in &block_s {
                    if s > block_max {
                        block_max = s;
                    }
                }
                let m_new = if block_max > m { block_max } else { m };

                // Rescale running state.
                let alpha = exp(m - m_new); // factor for old running values
                l *= alpha;
                for a in acc.iter_mut() {
                    *a *= alpha;
                }

                // Accumulate block.
                for (idx, &s) in block_s.iter().enumerate() {
                    let j = j_start + idx;
                    let beta = exp(s - m_new);
                    l += beta;
                    let v_off = j * n_heads * d_head + h * d_head;
                    let v_row = &v[v_off..v_off + d_head];
                    for d in 0..d_head {
                        acc[d] += beta * v_row[d];
                    }
                }

                m = m_new;
                j_start = j_end;
            }

            // Write output.
            let o_off = i * n_heads * d_head + h * d_head;
            if l > 0.0 {
                for d in 0..d_head {
                    out[o_off + d] = acc[d] / l;
                }
            }
        }
    }

    Some(out)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        let diff = if a > b { a - b } else { b - a };
        diff <= tol
    }

    #[test]
    fn kv_cache_append_and_retrieve() {
        let max_seq = 8;
        let n_kv_heads = 2;
        let d_head = 4;
        let mut cache = KvCache::new(max_seq, n_kv_heads, d_head);
        assert_eq!(cache.seq_len, 0);
        assert!(cache.k_slice().is_empty());

        let stride = n_kv_heads * d_head; // 8
        let k3: Vec<f64> = (0..3 * stride).map(|i| i as f64).collect();
        let v3: Vec<f64> = (0..3 * stride).map(|i| (i as f64) * 2.0).collect();
        assert!(cache.append(&k3, &v3));
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.k_slice().len(), 3 * stride);
        assert_eq!(cache.v_slice().len(), 3 * stride);

        for (i, (&got, &expected)) in cache.k_slice().iter().zip(k3.iter()).enumerate() {
            assert!(close(got, expected, 0.0), "k mismatch at {i}");
        }
        for (i, (&got, &expected)) in cache.v_slice().iter().zip(v3.iter()).enumerate() {
            assert!(close(got, expected, 0.0), "v mismatch at {i}");
        }

        let k5: Vec<f64> = (0..5 * stride).map(|i| (i as f64) * 10.0).collect();
        let v5: Vec<f64> = (0..5 * stride).map(|_| 1.0).collect();
        assert!(cache.append(&k5, &v5));
        assert_eq!(cache.seq_len, 8);

        let k1: Vec<f64> = vec![0.0; stride];
        let v1: Vec<f64> = vec![0.0; stride];
        assert!(!cache.append(&k1, &v1), "append should fail when full");

        cache.clear();
        assert_eq!(cache.seq_len, 0);
        assert!(cache.k_slice().is_empty());
        assert!(cache.append(&k1, &v1));
        assert_eq!(cache.seq_len, 1);
    }

    #[test]
    fn kv_cache_rejects_mismatched_slices() {
        let mut cache = KvCache::new(4, 2, 4);
        assert!(!cache.append(&[0.0; 8], &[0.0; 16]));
        assert!(!cache.append(&[0.0; 5], &[0.0; 5]));
        assert_eq!(cache.seq_len, 0);
    }
}

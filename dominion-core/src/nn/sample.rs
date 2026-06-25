//! Sampling strategies for language model decoding.
//!
//! Implements a composable pipeline: repetition penalty → grid snap →
//! temperature scaling → top-k filter → top-p (nucleus) filter → softmax →
//! sample (or greedy when temperature ≈ 0).
//!
//! All functions are pure `no_std + alloc`, deterministic, zero-unsafe.

#![allow(dead_code)]

use crate::ml::{exp, Rng};
use alloc::vec::Vec;

// ─────────────────────── internal SplitMix64 ─────────────────────────────────
//
// `Rng::next_signed` is private to `crate::ml`; we cannot call it from here.
// For `sample_from_probs` we derive a deterministic uniform value by hashing
// (rng pointer address ⊕ probs content) through one SplitMix64 finalisation
// step.  Because the probability vector changes at every decode step, successive
// calls produce distinct samples while remaining fully reproducible.

/// Advance a SplitMix64 state and return a uniform `f64` in `[0, 1)`.
fn splitmix_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 53 bits → [0, 1)
    (z >> 11) as f64 / (1u64 << 53) as f64
}

// ─────────────────────────── primitive ops ───────────────────────────────────

/// Quantise every logit to one of `levels` uniformly-spaced values in
/// `[min, max]`.  With `levels = 65536` this is a 16-bit grid snap that
/// crushes floating-point noise while keeping probability mass on the same
/// peaks.  A no-op when `levels < 2` or the range is zero.
pub fn grid_snap(logits: &mut [f64], levels: u32) {
    if levels < 2 || logits.is_empty() {
        return;
    }
    let min = logits.iter().cloned().fold(f64::MAX, f64::min);
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let range = max - min;
    if range == 0.0 {
        return;
    }
    let step = range / (levels as f64 - 1.0);
    for v in logits.iter_mut() {
        let bucket = ((*v - min) / step + 0.5) as u32;
        let bucket = bucket.min(levels - 1);
        *v = min + bucket as f64 * step;
    }
}

/// Return the index of the maximum logit.  Ties are broken by lowest index.
pub fn greedy(logits: &[f64]) -> usize {
    let mut best = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Divide every logit by `temperature`.  Values ≤ 0 or ≥ 1e9 are left
/// unchanged.
pub fn apply_temperature(logits: &mut [f64], temperature: f64) {
    if temperature <= 0.0 || temperature >= 1e9 {
        return;
    }
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

/// Zero-out all logits except the top-`k` by value.  A `k` of 0 is a no-op.
pub fn top_k_filter(logits: &mut [f64], k: usize) {
    if k == 0 || logits.len() <= k {
        return;
    }
    // Collect the k largest values via a min-heap of size k (simple O(n·k)).
    let mut top: Vec<f64> = Vec::with_capacity(k);
    for &v in logits.iter() {
        if top.len() < k {
            // Insert in ascending order so top[last] is the smallest of the top.
            let pos = top.partition_point(|&x| x < v);
            top.insert(pos, v);
        } else if v > top[0] {
            top[0] = v;
            // Bubble up to maintain ascending order.
            let mut i = 0;
            while i + 1 < top.len() && top[i] > top[i + 1] {
                top.swap(i, i + 1);
                i += 1;
            }
        }
    }
    let threshold = top.first().copied().unwrap_or(f64::NEG_INFINITY);
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f64::NEG_INFINITY;
        }
    }
}

/// Nucleus (top-p) filter: keep only the smallest prefix of tokens (sorted by
/// descending probability) whose cumulative probability ≥ `p`.  A `p` ≥ 1.0
/// is a no-op.
pub fn top_p_filter(logits: &mut [f64], p: f64) {
    if p >= 1.0 || logits.is_empty() {
        return;
    }
    // Sort (value, original index) descending.
    let mut pairs: Vec<(f64, usize)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

    // Softmax probabilities over the sorted slice (numerically stable).
    let max = pairs[0].0;
    let mut denom = 0.0_f64;
    let exps: Vec<f64> = pairs
        .iter()
        .map(|&(v, _)| {
            let e = if v == f64::NEG_INFINITY {
                0.0
            } else {
                exp(v - max)
            };
            denom += e;
            e
        })
        .collect();

    // Walk until cumulative prob ≥ p.
    let mut cumsum = 0.0_f64;
    let mut keep_count = pairs.len();
    for (idx, &e) in exps.iter().enumerate() {
        cumsum += if denom > 0.0 { e / denom } else { 0.0 };
        if cumsum >= p {
            keep_count = idx + 1;
            break;
        }
    }

    // Mask everything outside the nucleus.
    let mut keep = alloc::vec![false; logits.len()];
    for &(_, orig) in &pairs[..keep_count] {
        keep[orig] = true;
    }
    for (v, &k) in logits.iter_mut().zip(keep.iter()) {
        if !k {
            *v = f64::NEG_INFINITY;
        }
    }
}

/// Repetition penalty (Keskar et al. 2019): for each token that appears in
/// `generated`, its logit is divided by `penalty` if positive or multiplied by
/// `penalty` if negative.  A `penalty` of 1.0 is a no-op.
pub fn apply_rep_penalty(logits: &mut [f64], generated: &[usize], penalty: f64) {
    if penalty == 1.0 || generated.is_empty() {
        return;
    }
    for &tok in generated {
        if let Some(v) = logits.get_mut(tok) {
            if *v >= 0.0 {
                *v /= penalty;
            } else {
                *v *= penalty;
            }
        }
    }
}

/// Numerically stable softmax in-place.  `-inf` logits produce probability 0.
pub fn softmax(logits: &mut [f64]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for v in logits.iter_mut() {
        *v = if *v == f64::NEG_INFINITY {
            0.0
        } else {
            let e = exp(*v - max);
            sum += e;
            e
        };
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Draw a token index from a probability distribution.
///
/// Advances `rng` by one step to get a fresh 64-bit value, then mixes it with
/// a FNV-1a hash of the probability vector through one SplitMix64 finalisation
/// step.  Advancing the RNG state ensures that consecutive calls — even with
/// identical `probs` — produce independent samples.
pub fn sample_from_probs(probs: &[f64], rng: &mut Rng) -> usize {
    // Advance the RNG so each call gets a fresh, independent seed value.
    let rng_step = rng.next_u64();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ rng_step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for &p in probs {
        h ^= p.to_bits();
        h = h.wrapping_mul(0x517c_c1b7_2722_0a95);
        h ^= h >> 32;
    }
    // One SplitMix64 finalisation step.
    let u = splitmix_f64(&mut h);

    let mut cumsum = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= u {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ─────────────────────────── SamplerConfig ───────────────────────────────────

/// All knobs for the token-sampling pipeline.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Softmax temperature.  0.0 → greedy; >0 → stochastic.
    pub temperature: f64,
    /// Keep only the top-k logits before sampling.  0 = disabled.
    pub top_k: usize,
    /// Nucleus probability mass.  1.0 = disabled.
    pub top_p: f64,
    /// Repetition penalty (≥1.0; 1.0 = none).
    pub rep_penalty: f64,
    /// Quantise logits to a uniform grid before sampling.
    pub grid_snap: bool,
    /// Number of grid levels for `grid_snap`.
    pub grid_snap_levels: u32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            rep_penalty: 1.0,
            grid_snap: true,
            grid_snap_levels: 65536,
        }
    }
}

impl SamplerConfig {
    /// Pure greedy decoding — always picks the highest-probability token.
    pub fn greedy() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            rep_penalty: 1.0,
            grid_snap: false,
            grid_snap_levels: 65536,
        }
    }

    /// Stochastic sampling with the given temperature and nucleus cutoff.
    pub fn sample(temperature: f64, top_p: f64) -> Self {
        SamplerConfig {
            temperature,
            top_p,
            ..SamplerConfig::default()
        }
    }
}

// ─────────────────────────── run_sampler ─────────────────────────────────────

/// Run the full sampling pipeline and return the chosen token index.
///
/// Pipeline:
/// 1. Repetition penalty
/// 2. Grid snap (if enabled)
/// 3. Temperature scaling (if `0 < temperature < 1e9`)
/// 4. Top-k filter
/// 5. Top-p filter
/// 6. Softmax
/// 7. Sample (or greedy if `temperature ≈ 0`)
pub fn run_sampler(
    logits: &mut [f64],
    cfg: &SamplerConfig,
    generated: &[usize],
    rng: &mut Rng,
) -> usize {
    apply_rep_penalty(logits, generated, cfg.rep_penalty);
    if cfg.grid_snap {
        grid_snap(logits, cfg.grid_snap_levels);
    }
    let is_greedy = cfg.temperature <= 1e-9;
    if !is_greedy {
        apply_temperature(logits, cfg.temperature);
    }
    top_k_filter(logits, cfg.top_k);
    top_p_filter(logits, cfg.top_p);
    if is_greedy {
        greedy(logits)
    } else {
        softmax(logits);
        sample_from_probs(logits, rng)
    }
}

// ─────────────────────────── tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        assert_eq!(greedy(&[0.1_f64, 5.0, 0.3, 2.0]), 1);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = alloc::vec![1.0_f64, 2.0, 3.0, 4.0];
        softmax(&mut logits);
        let sum: f64 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
        assert!(logits.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn temperature_scales_logits() {
        let mut logits = alloc::vec![2.0_f64, 4.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-12);
        assert!((logits[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn top_k_leaves_k_finite_entries() {
        let mut logits = alloc::vec![1.0_f64, 5.0, 3.0, 2.0, 4.0];
        top_k_filter(&mut logits, 2);
        let finite = logits.iter().filter(|&&v| v != f64::NEG_INFINITY).count();
        assert_eq!(finite, 2);
    }

    #[test]
    fn rep_penalty_reduces_positive_seen_token() {
        let mut logits = alloc::vec![0.0_f64, 2.0, 0.0];
        apply_rep_penalty(&mut logits, &[1], 2.0);
        assert!((logits[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn run_sampler_greedy_config() {
        let mut logits = alloc::vec![0.1_f64, 9.0, 0.2];
        let mut rng = Rng::new(42);
        let tok = run_sampler(&mut logits, &SamplerConfig::greedy(), &[], &mut rng);
        assert_eq!(tok, 1);
    }

    #[test]
    fn grid_snap_idempotent_on_uniform() {
        // All equal → range = 0 → no-op.
        let mut logits = alloc::vec![1.0_f64; 4];
        grid_snap(&mut logits, 16);
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }
}

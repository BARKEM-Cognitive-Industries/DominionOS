//! Neural codecs, generative compression & RL storage (SRS Stage 5.1–5.3; see
//! `docs/architecture/07-stage-05-generative-storage.md`).
//!
//! Generative storage compresses by **predicting** and storing only what the model
//! got wrong, and it places data on the tier a **learned policy** thinks is cheapest
//! for the observed access pattern. Where an NPU exists it runs the model; where it
//! does not, the same predictor runs on the CPU — so this is hardware-*accelerated*,
//! never hardware-*required*.
//!
//! * [`compress`] / [`decompress`] — a predictive (delta) model + run-length coding
//!   of the residual the model could not predict. **Always lossless**: it verifies
//!   against a content hash and falls back to verbatim storage when a block is
//!   incompressible, so output is never larger than input + 1 byte.
//! * [`LearnedTierPolicy`] — a reinforcement-learning (Q-learning) bandit that
//!   chooses a storage tier from rewards (negative cost), converging on the best
//!   tier for the workload.
//!
//! Pure, safe `no_std`, host-tested.

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const MODE_RAW: u8 = 0;
const MODE_DELTA_RLE: u8 = 1;

/// Delta-transform: residual[i] = in[i] − in[i−1]. Reversible; small for data the
/// model predicts well (gradients, counters, smooth signals).
fn delta_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut prev = 0u8;
    for &b in data {
        out.push(b.wrapping_sub(prev));
        prev = b;
    }
    out
}

fn delta_decode(res: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(res.len());
    let mut prev = 0u8;
    for &r in res {
        let b = r.wrapping_add(prev);
        out.push(b);
        prev = b;
    }
    out
}

/// Run-length encode as `(count, value)` pairs, counts 1..=255.
fn rle_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let v = data[i];
        let mut run = 1usize;
        while i + run < data.len() && data[i + run] == v && run < 255 {
            run += 1;
        }
        out.push(run as u8);
        out.push(v);
        i += run;
    }
    out
}

fn rle_decode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < data.len() {
        let run = data[i];
        let v = data[i + 1];
        for _ in 0..run {
            out.push(v);
        }
        i += 2;
    }
    out
}

/// Compress `data` losslessly. Output is `[mode][payload]`; mode 1 is the predictive
/// (delta) + RLE pipeline, mode 0 is verbatim fallback for incompressible blocks.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let coded = rle_encode(&delta_encode(data));
    if coded.len() < data.len() {
        let mut out = Vec::with_capacity(coded.len() + 1);
        out.push(MODE_DELTA_RLE);
        out.extend_from_slice(&coded);
        out
    } else {
        let mut out = Vec::with_capacity(data.len() + 1);
        out.push(MODE_RAW);
        out.extend_from_slice(data);
        out
    }
}

/// Decompress a block produced by [`compress`].
pub fn decompress(block: &[u8]) -> Vec<u8> {
    match block.split_first() {
        Some((&MODE_DELTA_RLE, payload)) => delta_decode(&rle_decode(payload)),
        Some((&MODE_RAW, payload)) => payload.to_vec(),
        _ => Vec::new(),
    }
}

/// A content-addressed compressed blob: the compressed bytes plus the hash of the
/// *original*, so decompression can be **verified** to reconstruct the source
/// exactly (the lossless guarantee is checkable, not just asserted).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerativeBlob {
    pub original_hash: Hash256,
    pub original_len: usize,
    pub compressed: Vec<u8>,
}

impl GenerativeBlob {
    pub fn encode(data: &[u8]) -> GenerativeBlob {
        GenerativeBlob {
            original_hash: Hash256::of(data),
            original_len: data.len(),
            compressed: compress(data),
        }
    }

    /// Decode and verify against the stored content hash. `None` if (impossibly)
    /// the reconstruction does not match — a corrupted blob is rejected, not served.
    pub fn decode(&self) -> Option<Vec<u8>> {
        let out = decompress(&self.compressed);
        if Hash256::of(&out) == self.original_hash {
            Some(out)
        } else {
            None
        }
    }

    /// Compression ratio ×1000 (original / compressed). >1000 means it shrank.
    pub fn ratio_milli(&self) -> u64 {
        if self.compressed.is_empty() {
            return 1000;
        }
        (self.original_len as u64 * 1000) / self.compressed.len() as u64
    }
}

// ─────────────────── Grid Snap (logit quantization) ───────────────────

/// **Grid Snap** (SRS §7.1): quantize a value to a fixed grid so that two values
/// differing only by floating-point **non-associativity drift** (`(a+b)+c` vs
/// `a+(b+c)`) collapse to the *same* grid point. This makes neural-codec logits
/// bit-reproducible across hardware/orderings — the precondition for using a
/// generative model deterministically. Rounding is done by integer cast (no `std`).
pub fn grid_snap(x: f64, step: f64) -> f64 {
    if step <= 0.0 {
        return x;
    }
    let q = x / step;
    // Round-half-away-from-zero via truncating cast (core-only, no libm).
    let n = if q >= 0.0 { (q + 0.5) as i64 } else { (q - 0.5) as i64 };
    n as f64 * step
}

/// Snap a whole vector of logits to the grid.
pub fn grid_snap_all(xs: &[f64], step: f64) -> Vec<f64> {
    xs.iter().map(|&x| grid_snap(x, step)).collect()
}

// ─────────────────── Two-Pass / Band-Pass routing ───────────────────

/// Where the codec should run a block, chosen by its compressibility band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Outside the neural band — a classical CPU pass is cheapest.
    Cpu,
    /// In the neural band — worth the NPU/GPU accelerator.
    Accelerator,
}

/// The result of the cheap classical **Scout** pre-pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoutResult {
    /// Estimated compression ratio ×1000.
    pub ratio_milli: u64,
    pub route: Route,
}

/// Route by ratio band (SRS §7.2): `R < 1.05` (barely compressible — overhead not
/// worth it) and `R > 3.0` (trivially compressible — a classical pass already
/// wins) both go to the **CPU**; the mid-band, where a learned model pays off,
/// routes to the **NPU/GPU accelerator**.
pub fn route_for_ratio(ratio_milli: u64) -> Route {
    if (1050..=3000).contains(&ratio_milli) {
        Route::Accelerator
    } else {
        Route::Cpu
    }
}

/// The **Two-Pass / Band-Pass** router: a cheap classical Scout pre-pass measures
/// the achievable ratio, then routes the (expensive) main pass by band.
pub fn scout(data: &[u8]) -> ScoutResult {
    let ratio_milli = GenerativeBlob::encode(data).ratio_milli();
    ScoutResult { ratio_milli, route: route_for_ratio(ratio_milli) }
}

// ─────────────────── CNN-LSTM block-access prediction ───────────────────

/// A spatiotemporal **block-access predictor** (a recency + dominant-stride model
/// standing in for the CNN-LSTM the spec calls for). It watches the sequence of
/// accessed block numbers, learns the prevailing stride, and predicts the next
/// block to **prefetch** — the input the kernel-level RL cache uses.
#[derive(Default)]
pub struct BlockAccessPredictor {
    last: Option<u64>,
    stride_votes: BTreeMap<i64, u32>,
}

impl BlockAccessPredictor {
    pub fn new() -> BlockAccessPredictor {
        BlockAccessPredictor { last: None, stride_votes: BTreeMap::new() }
    }

    /// Observe an access to `block`, updating the learned stride distribution.
    pub fn observe(&mut self, block: u64) {
        if let Some(prev) = self.last {
            let stride = block as i64 - prev as i64;
            *self.stride_votes.entry(stride).or_insert(0) += 1;
        }
        self.last = Some(block);
    }

    /// The most-observed stride so far, if any.
    pub fn dominant_stride(&self) -> Option<i64> {
        self.stride_votes.iter().max_by_key(|(_, &c)| c).map(|(&s, _)| s)
    }

    /// Predict the next block (last + dominant stride), for prefetch.
    pub fn predict_next(&self) -> Option<u64> {
        let last = self.last?;
        let stride = self.dominant_stride()?;
        let next = last as i64 + stride;
        if next >= 0 {
            Some(next as u64)
        } else {
            None
        }
    }
}

// ─────────────────── RL storage-tier policy ───────────────────

/// Storage tiers, fastest/most-expensive first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Hot = 0,
    Warm = 1,
    Cold = 2,
}

const TIERS: [Tier; 3] = [Tier::Hot, Tier::Warm, Tier::Cold];

/// A Q-learning bandit that learns which tier minimises cost for a workload.
/// Rewards are higher for cheaper outcomes; the policy converges on the best tier.
pub struct LearnedTierPolicy {
    q: [f64; 3],
    alpha: f64,
    epsilon_milli: u32,
}

impl LearnedTierPolicy {
    pub fn new() -> LearnedTierPolicy {
        LearnedTierPolicy { q: [0.0; 3], alpha: 0.3, epsilon_milli: 100 }
    }

    /// Choose a tier ε-greedily. `sample` (0..1000, from the DRNG) decides explore
    /// vs exploit, and `explore_pick` selects which tier to try when exploring.
    pub fn choose(&self, sample: u32, explore_pick: usize) -> Tier {
        if sample < self.epsilon_milli {
            TIERS[explore_pick % TIERS.len()]
        } else {
            self.best()
        }
    }

    /// The current greedy best tier (highest Q).
    pub fn best(&self) -> Tier {
        let mut best = 0;
        for i in 1..3 {
            if self.q[i] > self.q[best] {
                best = i;
            }
        }
        TIERS[best]
    }

    /// Update the chosen tier's value toward an observed `reward`.
    pub fn reward(&mut self, tier: Tier, reward: f64) {
        let i = tier as usize;
        self.q[i] += self.alpha * (reward - self.q[i]);
    }

    pub fn value(&self, tier: Tier) -> f64 {
        self.q[tier as usize]
    }
}

impl Default for LearnedTierPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_round_trips_losslessly() {
        let cases: [&[u8]; 4] = [
            b"",
            b"aaaaaaaaaaaaaaaaaaaaaaaa", // RLE-friendly
            b"0123456789:;<=>?@ABCDEFG", // delta-friendly (gradient)
            b"the quick brown fox jumps over the lazy dog",
        ];
        for c in cases {
            assert_eq!(decompress(&compress(c)), c, "round-trip failed for {c:?}");
        }
    }

    #[test]
    fn structured_data_compresses() {
        // A smooth ramp: delta → constant → one RLE run.
        let ramp: Vec<u8> = (0..200u32).map(|i| (i % 256) as u8).collect();
        let blob = GenerativeBlob::encode(&ramp);
        assert!(blob.ratio_milli() > 1000, "expected shrink, ratio={}", blob.ratio_milli());
        assert_eq!(blob.decode().unwrap(), ramp);
    }

    #[test]
    fn incompressible_data_falls_back_and_never_grows() {
        // Pseudo-random bytes from the hash → no exploitable structure.
        let mut data = Vec::new();
        for i in 0..16u8 {
            data.extend_from_slice(&Hash256::of(&[i]).0);
        }
        let comp = compress(&data);
        // Output is at most input + 1 byte (the mode header).
        assert!(comp.len() <= data.len() + 1);
        assert_eq!(decompress(&comp), data);
    }

    #[test]
    fn generative_blob_verifies_against_content_hash() {
        let data = b"semantic content addressed and regenerated";
        let mut blob = GenerativeBlob::encode(data);
        assert_eq!(blob.decode().unwrap(), data);
        // Corrupt the compressed payload → verification fails, nothing is served.
        let last = blob.compressed.len() - 1;
        blob.compressed[last] ^= 0xFF;
        assert!(blob.decode().is_none());
    }

    #[test]
    fn rl_policy_converges_on_best_tier() {
        let mut pol = LearnedTierPolicy::new();
        // Workload where Warm is cheapest (highest reward), Hot/Cold worse.
        for _ in 0..50 {
            pol.reward(Tier::Hot, 0.2);
            pol.reward(Tier::Warm, 1.0);
            pol.reward(Tier::Cold, 0.5);
        }
        assert_eq!(pol.best(), Tier::Warm);
        assert!(pol.value(Tier::Warm) > pol.value(Tier::Hot));
        assert!(pol.value(Tier::Warm) > pol.value(Tier::Cold));
    }

    #[test]
    fn grid_snap_defeats_fp_nonassociativity() {
        // Two orderings of the same sum differ by a rounding epsilon …
        let a = (0.1_f64 + 0.2) + 0.3;
        let b = 0.1 + (0.2 + 0.3);
        assert_ne!(a, b); // classic FP non-associativity
        // … but after Grid Snap to a coarse grid they agree exactly.
        let step = 1e-6;
        assert_eq!(grid_snap(a, step), grid_snap(b, step));
        // Snapping is value-preserving to within the grid step.
        assert!((grid_snap(1.2345678, 0.001) - 1.235).abs() < 1e-9);
    }

    #[test]
    fn band_pass_router_picks_cpu_outside_the_band() {
        // Barely compressible and trivially compressible → CPU.
        assert_eq!(route_for_ratio(1000), Route::Cpu); // R≈1.0
        assert_eq!(route_for_ratio(5000), Route::Cpu); // R=5.0
        // Mid-band → accelerator.
        assert_eq!(route_for_ratio(2000), Route::Accelerator); // R=2.0
    }

    #[test]
    fn scout_routes_incompressible_data_to_cpu() {
        // Pseudo-random bytes ≈ ratio 1.0 → CPU pass (don't waste the accelerator).
        let mut data = Vec::new();
        for i in 0..16u8 {
            data.extend_from_slice(&Hash256::of(&[i]).0);
        }
        assert_eq!(scout(&data).route, Route::Cpu);
    }

    #[test]
    fn block_predictor_learns_stride_and_prefetches() {
        let mut p = BlockAccessPredictor::new();
        for b in [0u64, 4, 8, 12, 16] {
            p.observe(b);
        }
        // Dominant stride is +4; next prefetch is 20.
        assert_eq!(p.dominant_stride(), Some(4));
        assert_eq!(p.predict_next(), Some(20));
    }

    #[test]
    fn epsilon_greedy_explores_then_exploits() {
        let mut pol = LearnedTierPolicy::new();
        pol.reward(Tier::Cold, 1.0); // make Cold the greedy best
        // Exploit (sample above epsilon) → best tier.
        assert_eq!(pol.choose(500, 0), Tier::Cold);
        // Explore (sample below epsilon) → the explore pick.
        assert_eq!(pol.choose(10, 0), Tier::Hot);
        assert_eq!(pol.choose(10, 1), Tier::Warm);
    }
}

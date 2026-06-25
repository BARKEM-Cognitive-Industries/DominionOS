//! Embedding operators: token lookup, sinusoidal PE, RoPE frequency tables, ALiBi.
//!
//! * [`Embedding`] — learnable token-lookup table with SGD update.
//! * [`sinusoidal_pe`] — deterministic sin/cos positional encodings.
//! * [`rope_freqs`] — pre-computed RoPE cos/sin frequency table.
//! * [`alibi_slopes`] — ALiBi per-head slopes.
//!
//! Pure `no_std + alloc`, `#![forbid(unsafe_code)]`, bit-exact deterministic.

use crate::datatypes::Tensor;
use alloc::vec;
use alloc::vec::Vec;

// ─────────────────────────── no_std trig / math helpers ──────────────────────────
//
// All implementations live in `crate::math`; aliased here for local use.
use crate::math::{sin, cos, ln, exp as math_exp};

const PI: f64 = core::f64::consts::PI;

/// Alias so existing call-sites in this file compile unchanged.
#[inline] fn sin_f64(x: f64) -> f64 { sin(x) }
/// Alias so existing call-sites in this file compile unchanged.
#[inline] fn cos_f64(x: f64) -> f64 { cos(x) }

/// `x^p` via `exp(p · ln(x))` for `x > 0`.
fn pow_f64(x: f64, p: f64) -> f64 {
    if p == 0.0 { return 1.0; }
    math_exp(p * ln(x))
}

// ──────────────────────────── SplitMix64 PRNG ─────────────────────────────
//
// Local implementation so we can call `next_signed` without relying on the
// visibility of `crate::ml::Rng`'s private methods.

struct Rng64 { state: u64 }

impl Rng64 {
    fn new(seed: u64) -> Self {
        Rng64 { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[-1.0, 1.0)`.
    fn next_signed(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let unit = bits as f64 / (1u64 << 53) as f64;
        unit * 2.0 - 1.0
    }
}

// ─────────────────────────── Embedding ────────────────────────────────────

/// Learnable token-embedding table: maps integer token IDs to dense vectors.
///
/// The weight matrix has shape `[vocab, dim]`, stored row-major in `weight`.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Flat weight table, row-major `[vocab, dim]`.
    pub weight: Vec<f64>,
    /// Vocabulary size.
    pub vocab: usize,
    /// Embedding dimension.
    pub dim: usize,
}

impl Embedding {
    /// Create a new embedding table with weights sampled uniformly from `[-scale, scale)`
    /// where `scale = 1/sqrt(dim)` (Xavier-style).
    pub fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        let mut rng = Rng64::new(seed);
        let n = vocab * dim;
        let scale = if dim > 0 { 1.0 / crate::datatypes::sqrt(dim as f64) } else { 1.0 };
        let mut weight = vec![0.0f64; n];
        for w in weight.iter_mut() {
            *w = rng.next_signed() * scale;
        }
        Embedding { weight, vocab, dim }
    }

    /// Look up `token_ids` and return a `[seq_len, dim]` tensor.
    ///
    /// Returns `None` if any token ID is out of range or `dim == 0`.
    pub fn forward(&self, token_ids: &[usize]) -> Option<Tensor> {
        if self.dim == 0 || self.vocab == 0 {
            return None;
        }
        let seq = token_ids.len();
        let mut out = vec![0.0f64; seq * self.dim];
        for (i, &id) in token_ids.iter().enumerate() {
            if id >= self.vocab {
                return None;
            }
            let src = &self.weight[id * self.dim..(id + 1) * self.dim];
            out[i * self.dim..(i + 1) * self.dim].copy_from_slice(src);
        }
        Tensor::new(vec![seq, self.dim], out)
    }

    /// Apply a single SGD gradient step to embedding row `id`.
    ///
    /// `w_new[id] -= lr * grad`. `grad` must have length `dim`.
    /// Returns `false` on any mismatch.
    pub fn update_row(&mut self, id: usize, grad: &[f64], lr: f64) -> bool {
        if id >= self.vocab || grad.len() != self.dim {
            return false;
        }
        let row = &mut self.weight[id * self.dim..(id + 1) * self.dim];
        for (w, &g) in row.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
        true
    }
}

// ──────────────────────── sinusoidal_pe ───────────────────────────────────

/// Standard sinusoidal positional encodings ("Attention Is All You Need").
///
/// `PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))`
/// `PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))`
///
/// Returns a `Tensor` of shape `[seq_len, d_model]`, or `None` if either
/// dimension is zero.
pub fn sinusoidal_pe(seq_len: usize, d_model: usize) -> Option<Tensor> {
    if d_model == 0 || seq_len == 0 {
        return None;
    }
    let mut data = vec![0.0f64; seq_len * d_model];
    for pos in 0..seq_len {
        let base = pos * d_model;
        let pairs = d_model / 2;
        for i in 0..pairs {
            let exponent = (2 * i) as f64 / d_model as f64;
            let div = pow_f64(10000.0, exponent);
            let angle = pos as f64 / div;
            data[base + 2 * i]     = sin_f64(angle);
            data[base + 2 * i + 1] = cos_f64(angle);
        }
        // If d_model is odd, fill the trailing slot with the next sin.
        if d_model % 2 == 1 {
            let i = d_model / 2;
            let exponent = (2 * i) as f64 / d_model as f64;
            let div = pow_f64(10000.0, exponent);
            data[base + 2 * i] = sin_f64(pos as f64 / div);
        }
    }
    Tensor::new(vec![seq_len, d_model], data)
}

// ──────────────────────────── rope_freqs ──────────────────────────────────

/// Pre-compute RoPE cosine/sine frequency table.
///
/// Returns a flat `Vec<f64>` of length `seq_len * d_head` storing interleaved
/// `(cos, sin)` pairs indexed by `[pos, freq_index]`:
///
/// * `theta[i] = 1 / base^(2i / d_head)`
/// * Layout `[pos * d_head + 2*i]`   = `cos(pos * theta[i])`
/// * Layout `[pos * d_head + 2*i+1]` = `sin(pos * theta[i])`
///
/// for `i` in `0 .. d_head/2`.  Total length = `seq_len * d_head`.
///
/// Returns an empty `Vec` if `d_head == 0` or `d_head` is odd.
pub fn rope_freqs(d_head: usize, base: f64, seq_len: usize) -> Vec<f64> {
    if d_head == 0 || d_head % 2 != 0 {
        return Vec::new();
    }
    let half = d_head / 2;
    let mut out = vec![0.0f64; seq_len * d_head];
    for pos in 0..seq_len {
        for i in 0..half {
            // theta[i] = base^(-2i/d_head)
            let exponent = (2 * i) as f64 / d_head as f64;
            let theta = 1.0 / pow_f64(base, exponent);
            let angle = pos as f64 * theta;
            out[pos * d_head + 2 * i]     = cos_f64(angle);
            out[pos * d_head + 2 * i + 1] = sin_f64(angle);
        }
    }
    out
}

// ──────────────────────────── alibi_slopes ────────────────────────────────

/// ALiBi per-head slopes (Press et al., 2022).
///
/// `slope_h = 2^(-8 * h / n_heads)` for `h` in `1..=n_heads`.
///
/// Returns a `Vec<f64>` of length `n_heads`, or an empty `Vec` if `n_heads == 0`.
pub fn alibi_slopes(n_heads: usize) -> Vec<f64> {
    if n_heads == 0 {
        return Vec::new();
    }
    (1..=n_heads)
        .map(|h| {
            // 2^(-8*h/n_heads) = exp(-8*h/n_heads * ln2)
            let exp_arg = -8.0 * h as f64 / n_heads as f64 * core::f64::consts::LN_2;
            crate::ml::exp(exp_arg)
        })
        .collect()
}

// ──────────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    // ── trig sanity ─────────────────────────────────────────────────────────

    #[test]
    fn trig_known_values() {
        // sin(0) = 0, cos(0) = 1.
        assert!(close(sin_f64(0.0), 0.0, 1e-14));
        assert!(close(cos_f64(0.0), 1.0, 1e-14));
        // sin(π/2) = 1, cos(π/2) = 0.
        assert!(close(sin_f64(PI / 2.0), 1.0, 1e-12));
        assert!(close(cos_f64(PI / 2.0), 0.0, 1e-12));
        // sin(π) ≈ 0, cos(π) = -1.
        assert!(close(sin_f64(PI), 0.0, 1e-12));
        assert!(close(cos_f64(PI), -1.0, 1e-12));
    }

    #[test]
    fn trig_pythagorean_identity() {
        // sin²(x) + cos²(x) = 1 for several values.
        for &a in &[0.0_f64, 0.5, 1.0, 2.0, 3.14, -1.5, 6.28, 100.0] {
            let s = sin_f64(a);
            let c = cos_f64(a);
            assert!(close(s * s + c * c, 1.0, 1e-11));
        }
    }

    // ── Embedding ───────────────────────────────────────────────────────────

    #[test]
    fn embedding_forward_correct_shape() {
        let emb = Embedding::new(100, 8, 42);
        let ids = [0usize, 5, 99];
        let t = emb.forward(&ids).unwrap();
        assert_eq!(t.shape(), &[3, 8]);
    }

    #[test]
    fn embedding_out_of_range_returns_none() {
        let emb = Embedding::new(10, 4, 1);
        assert!(emb.forward(&[10usize]).is_none());
    }

    #[test]
    fn embedding_update_row_sgd() {
        let mut emb = Embedding::new(5, 3, 7);
        let before: Vec<f64> = emb.weight[0..3].to_vec();
        let grad = vec![1.0, 1.0, 1.0];
        assert!(emb.update_row(0, &grad, 0.1));
        for i in 0..3 {
            assert!(close(emb.weight[i], before[i] - 0.1, 1e-14));
        }
    }

    // ── sinusoidal_pe ───────────────────────────────────────────────────────

    #[test]
    fn sinusoidal_pe_shape_and_pos0() {
        let pe = sinusoidal_pe(4, 8).unwrap();
        assert_eq!(pe.shape(), &[4, 8]);
        // PE[0, 0] = sin(0) = 0.
        assert!(close(pe.data()[0], 0.0, 1e-14));
        // PE[0, 1] = cos(0) = 1.
        assert!(close(pe.data()[1], 1.0, 1e-14));
    }

    #[test]
    fn sinusoidal_pe_none_on_zero() {
        assert!(sinusoidal_pe(4, 0).is_none());
        assert!(sinusoidal_pe(0, 8).is_none());
    }

    // ── rope_freqs ──────────────────────────────────────────────────────────

    #[test]
    fn rope_freqs_pos0_is_identity_rotation() {
        // At position 0 all angles are 0: cos=1, sin=0.
        let freqs = rope_freqs(4, 10000.0, 3);
        assert_eq!(freqs.len(), 3 * 4);
        // pos=0 occupies indices 0..4: (cos,sin, cos,sin).
        assert!(close(freqs[0], 1.0, 1e-14)); // cos
        assert!(close(freqs[1], 0.0, 1e-14)); // sin
        assert!(close(freqs[2], 1.0, 1e-14)); // cos
        assert!(close(freqs[3], 0.0, 1e-14)); // sin
    }

    #[test]
    fn rope_freqs_pythagorean_per_entry() {
        // For every (pos, i): cos²+sin²=1.
        let freqs = rope_freqs(8, 10000.0, 4);
        let half = 4 / 2;
        for pos in 0..8 {
            for i in 0..half {
                let c = freqs[pos * 4 + 2 * i];
                let s = freqs[pos * 4 + 2 * i + 1];
                assert!(close(c * c + s * s, 1.0, 1e-12));
            }
        }
    }

    #[test]
    fn rope_freqs_empty_on_odd_d_head() {
        assert!(rope_freqs(3, 10000.0, 5).is_empty());
    }

    // ── alibi_slopes ────────────────────────────────────────────────────────

    #[test]
    fn alibi_slopes_length_and_decreasing() {
        let slopes = alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1]);
            assert!(slopes[i] > 0.0);
        }
    }

    #[test]
    fn alibi_slopes_n1_equals_2_pow_neg8() {
        // slope_1 = 2^(-8*1/1) = 2^-8 = 1/256.
        let slopes = alibi_slopes(1);
        assert!(close(slopes[0], 1.0 / 256.0, 1e-12));
    }

    #[test]
    fn alibi_slopes_empty_for_zero_heads() {
        assert!(alibi_slopes(0).is_empty());
    }
}

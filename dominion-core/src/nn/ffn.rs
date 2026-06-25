//! Feed-forward network variants: standard FFN and gated FFN (SwiGLU/GeGLU).
//!
//! * [`Ffn`]      — `output = W2 · act(W1 · x)` (standard two-layer FFN).
//! * [`GatedFfn`] — `output = W_down · (gate(W_gate · x) ⊙ (W_up · x))`
//!                  (SwiGLU/GeGLU gated variant used in Llama/Gemma/Qwen).
//!
//! Both structs use a **cache-blocked dense matmul** tiled in rows of 32 for
//! L1 friendliness, with no external dependencies.
//!
//! Activation dispatch is via the crate-level [`crate::nn::FfnAct`] enum so
//! callers share a single vocabulary for all FFN flavors.
//!
//! Pure `no_std + alloc`, `#![forbid(unsafe_code)]`, bit-exact deterministic.

use crate::datatypes::Tensor;
use crate::ml::{sigmoid, tanh};
use super::FfnAct;
use alloc::vec;
use alloc::vec::Vec;

// ─────────────────────────── activation helpers ───────────────────────────

/// Approximate GELU: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
#[inline]
fn gelu(x: f64) -> f64 {
    // √(2/π) ≈ 0.7978845608028654.
    const C: f64 = 0.797_884_560_802_865_4_f64;
    let inner = C * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + tanh(inner))
}

/// SiLU / Swish: `x · σ(x)`.
#[inline]
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// ReLU: `max(0, x)`.
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Dispatch to the right scalar activation for a given [`FfnAct`].
///
/// For gated variants (`Swiglu`, `Geglu`, `Reglu`) this is the **gate** branch
/// activation (the `up` branch is always linear).
#[inline]
fn apply_act(v: f64, act: FfnAct) -> f64 {
    match act {
        FfnAct::Gelu   => gelu(v),
        FfnAct::Silu   => silu(v),
        FfnAct::Swiglu => silu(v),   // gate uses SiLU
        FfnAct::Geglu  => gelu(v),   // gate uses GELU
        FfnAct::Relu   => relu(v),
        FfnAct::Reglu  => relu(v),   // gate uses ReLU
        FfnAct::Linear => v,
    }
}

// ─────────────────────────── cache-blocked matmul ─────────────────────────

/// Dense matrix multiply: `out[batch, rows] = w[rows, k] × x[batch, k]`.
///
/// Weight `w` is `[rows, k]` (row-major), input `x` is `[batch, k]` (row-major).
/// Output is `[batch, rows]` (row-major).
///
/// Tiled over `rows` in blocks of `TILE` for L1 cache reuse.
fn matmul_dense(w: &[f64], x_data: &[f64], rows: usize, k: usize, batch: usize) -> Vec<f64> {
    const TILE: usize = 32;
    let mut out = vec![0.0f64; batch * rows];
    // Tile over output rows.
    let mut row_start = 0;
    while row_start < rows {
        let row_end = (row_start + TILE).min(rows);
        for b in 0..batch {
            let xrow = &x_data[b * k..(b + 1) * k];
            for r in row_start..row_end {
                let wrow = &w[r * k..(r + 1) * k];
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += wrow[p] * xrow[p];
                }
                out[b * rows + r] = acc;
            }
        }
        row_start += TILE;
    }
    out
}

// ──────────────────────────── SplitMix64 PRNG ─────────────────────────────

struct Rng64 {
    state: u64,
}

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

    fn next_signed(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let unit = bits as f64 / (1u64 << 53) as f64;
        unit * 2.0 - 1.0
    }
}

/// Kaiming-uniform-style init: scale = sqrt(2/fan_in).
fn init_weights(rng: &mut Rng64, n: usize, fan_in: usize) -> Vec<f64> {
    let scale = crate::datatypes::sqrt(2.0_f64 / fan_in.max(1) as f64);
    (0..n).map(|_| rng.next_signed() * scale).collect()
}

// ──────────────────────────────── Ffn ─────────────────────────────────────

/// Standard two-layer feed-forward network.
///
/// `output = W2 · act(W1 · x)`
///
/// * `w1`: `[d_ff, d_model]` — expands the representation.
/// * `w2`: `[d_model, d_ff]` — projects back.
/// * Input `x`: `[batch·seq, d_model]`; output: `[batch·seq, d_model]`.
#[derive(Clone, Debug)]
pub struct Ffn {
    /// First projection weight, shape `[d_ff, d_model]`.
    pub w1: Vec<f64>,
    /// Second projection weight, shape `[d_model, d_ff]`.
    pub w2: Vec<f64>,
    /// Model dimension.
    pub d_model: usize,
    /// Hidden (expanded) dimension.
    pub d_ff: usize,
}

impl Ffn {
    /// Create a new `Ffn` with Kaiming-uniform weight initialisation.
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = Rng64::new(seed);
        let w1 = init_weights(&mut rng, d_ff * d_model, d_model);
        let w2 = init_weights(&mut rng, d_model * d_ff, d_ff);
        Ffn { w1, w2, d_model, d_ff }
    }

    /// Forward pass: `x` must be `[batch, d_model]` (or `[batch*seq, d_model]`).
    ///
    /// Returns `[batch, d_model]`, or `None` on a shape mismatch.
    pub fn forward(&self, x: &Tensor, act: FfnAct) -> Option<Tensor> {
        if x.shape().len() != 2 || x.shape()[1] != self.d_model {
            return None;
        }
        if self.d_ff == 0 || self.d_model == 0 {
            return None;
        }
        let batch = x.shape()[0];
        // h = act(W1 · x):  [batch, d_ff]
        let h_raw = matmul_dense(&self.w1, x.data(), self.d_ff, self.d_model, batch);
        let mut h = vec![0.0f64; batch * self.d_ff];
        for i in 0..h_raw.len() {
            h[i] = apply_act(h_raw[i], act);
        }
        // out = W2 · h:  [batch, d_model]
        let out = matmul_dense(&self.w2, &h, self.d_model, self.d_ff, batch);
        Tensor::new(vec![batch, self.d_model], out)
    }
}

// ──────────────────────────── GatedFfn ────────────────────────────────────

/// Gated feed-forward network (SwiGLU / GeGLU / ReGLU).
///
/// `output = W_down · (gate(W_gate · x) ⊙ (W_up · x))`
///
/// * `w_gate`: `[d_ff, d_model]` — gate projection (goes through activation).
/// * `w_up`:   `[d_ff, d_model]` — up projection (linear).
/// * `w_down`: `[d_model, d_ff]` — down projection.
/// * Input `x`: `[batch·seq, d_model]`; output: `[batch·seq, d_model]`.
#[derive(Clone, Debug)]
pub struct GatedFfn {
    /// Gate projection weight, shape `[d_ff, d_model]`.
    pub w_gate: Vec<f64>,
    /// Up projection weight, shape `[d_ff, d_model]`.
    pub w_up: Vec<f64>,
    /// Down projection weight, shape `[d_model, d_ff]`.
    pub w_down: Vec<f64>,
    /// Model dimension.
    pub d_model: usize,
    /// Hidden (expanded) dimension.
    pub d_ff: usize,
}

impl GatedFfn {
    /// Create a new `GatedFfn` with Kaiming-uniform weight initialisation.
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = Rng64::new(seed);
        let w_gate = init_weights(&mut rng, d_ff * d_model, d_model);
        let w_up   = init_weights(&mut rng, d_ff * d_model, d_model);
        let w_down = init_weights(&mut rng, d_model * d_ff, d_ff);
        GatedFfn { w_gate, w_up, w_down, d_model, d_ff }
    }

    /// Forward pass: `x` must be `[batch, d_model]` (or `[batch*seq, d_model]`).
    ///
    /// The gate branch uses `apply_act(act)`; the up branch is always linear.
    /// For `Gelu`/`Silu`/`Relu`/`Linear` the gate activation is the one that
    /// matches; for `Swiglu`/`Geglu`/`Reglu` the act-name itself names the
    /// variant and the gate uses the embedded activation.
    ///
    /// Returns `[batch, d_model]`, or `None` on a shape mismatch.
    pub fn forward(&self, x: &Tensor, act: FfnAct) -> Option<Tensor> {
        if x.shape().len() != 2 || x.shape()[1] != self.d_model {
            return None;
        }
        if self.d_ff == 0 || self.d_model == 0 {
            return None;
        }
        let batch = x.shape()[0];
        // gate_raw = W_gate · x:  [batch, d_ff]
        let gate_raw = matmul_dense(&self.w_gate, x.data(), self.d_ff, self.d_model, batch);
        // up_raw   = W_up · x:   [batch, d_ff]
        let up_raw   = matmul_dense(&self.w_up,   x.data(), self.d_ff, self.d_model, batch);
        // h = act(gate) ⊙ up
        let mut h = vec![0.0f64; batch * self.d_ff];
        for i in 0..h.len() {
            h[i] = apply_act(gate_raw[i], act) * up_raw[i];
        }
        // out = W_down · h:  [batch, d_model]
        let out = matmul_dense(&self.w_down, &h, self.d_model, self.d_ff, batch);
        Tensor::new(vec![batch, self.d_model], out)
    }
}

// ─────── backwards-compat free functions (kept for existing callers) ───────

/// FFN activation used by the original `swiglu_ffn` free function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnActivation {
    /// SiLU/swish: `x·σ(x)` (Llama, Qwen).
    SiLU,
    /// tanh-approx GELU (Gemma's `gelu_pytorch_tanh`).
    GeluTanh,
}

impl FfnActivation {
    #[inline]
    pub(crate) fn apply(self, x: f64) -> f64 {
        match self {
            FfnActivation::SiLU => silu(x),
            FfnActivation::GeluTanh => gelu(x),
        }
    }
}

/// SwiGLU/GeGLU feed-forward (free-function form, kept for existing callers).
///
/// `x`: `[seq, d]`, `w_gate`/`w_up`: `[d, ff]`, `w_down`: `[ff, d]`.
/// Returns `[seq, d]`, or `None` on a shape mismatch.
pub fn swiglu_ffn(
    x: &Tensor,
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
    act: FfnActivation,
) -> Option<Tensor> {
    if x.shape().len() != 2 {
        return None;
    }
    let gate = x.matmul(w_gate)?; // [seq, ff]
    let up   = x.matmul(w_up)?;   // [seq, ff]
    if gate.shape() != up.shape() {
        return None;
    }
    let gd = gate.data();
    let ud = up.data();
    let mut h = vec![0.0f64; gd.len()];
    for i in 0..gd.len() {
        h[i] = act.apply(gd[i]) * ud[i];
    }
    let hid = Tensor::new(gate.shape().to_vec(), h)?;
    hid.matmul(w_down)
}

#[doc(hidden)]
pub fn _exp_link(x: f64) -> f64 {
    crate::ml::exp(x)
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

    // ── activation helpers ──────────────────────────────────────────────────

    #[test]
    fn silu_zero_and_known() {
        assert!(close(silu(0.0), 0.0, 1e-12));
        // silu(1) = 1·σ(1) ≈ 0.731058...
        assert!(close(silu(1.0), 0.731_058_578_630_005, 1e-9));
    }

    #[test]
    fn gelu_zero_and_monotone() {
        assert!(close(gelu(0.0), 0.0, 1e-12));
        // gelu(1) ≈ 0.8412 (tanh approx).
        assert!(close(gelu(1.0), 0.841_191_990_607, 1e-6));
        // Large positive ≈ identity, large negative ≈ 0.
        assert!(gelu(6.0) > 5.9);
        assert!(gelu(-6.0).abs() < 1e-3);
    }

    #[test]
    fn relu_nonnegative() {
        assert!(close(relu(0.0), 0.0, 1e-15));
        assert!(close(relu(3.5), 3.5, 1e-15));
        assert!(close(relu(-2.0), 0.0, 1e-15));
    }

    // ── matmul_dense ────────────────────────────────────────────────────────

    #[test]
    fn matmul_dense_identity() {
        // rows=2, k=2, batch=1. Identity weight.
        let w = vec![1.0, 0.0, 0.0, 1.0_f64]; // [2,2]
        let x = vec![3.0, 7.0_f64];             // [1,2]
        let out = matmul_dense(&w, &x, 2, 2, 1);
        assert!(close(out[0], 3.0, 1e-14));
        assert!(close(out[1], 7.0, 1e-14));
    }

    // ── Ffn ─────────────────────────────────────────────────────────────────

    #[test]
    fn ffn_output_shape() {
        let ffn = Ffn::new(4, 8, 0);
        let x = Tensor::new(vec![2, 4], vec![0.1_f64; 8]).unwrap();
        let y = ffn.forward(&x, FfnAct::Silu).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
    }

    #[test]
    fn ffn_rejects_wrong_dim() {
        let ffn = Ffn::new(4, 8, 0);
        let x = Tensor::new(vec![1, 3], vec![0.0; 3]).unwrap();
        assert!(ffn.forward(&x, FfnAct::Gelu).is_none());
    }

    #[test]
    fn ffn_relu_nonneg_output_on_positive_input() {
        // With all-positive weights and positive input, ReLU won't clip.
        let mut ffn = Ffn::new(2, 4, 99);
        // Force weights positive so the first-layer output is positive.
        for w in ffn.w1.iter_mut() { *w = w.abs() + 0.1; }
        for w in ffn.w2.iter_mut() { *w = w.abs() + 0.1; }
        let x = Tensor::new(vec![1, 2], vec![1.0, 1.0]).unwrap();
        let y = ffn.forward(&x, FfnAct::Relu).unwrap();
        // All outputs should be non-negative.
        for &v in y.data() {
            assert!(v >= 0.0);
        }
    }

    // ── GatedFfn ─────────────────────────────────────────────────────────────

    #[test]
    fn gated_ffn_output_shape() {
        let gffn = GatedFfn::new(4, 8, 1);
        let x = Tensor::new(vec![3, 4], vec![0.5_f64; 12]).unwrap();
        let y = gffn.forward(&x, FfnAct::Swiglu).unwrap();
        assert_eq!(y.shape(), &[3, 4]);
    }

    #[test]
    fn gated_ffn_rejects_wrong_dim() {
        let gffn = GatedFfn::new(4, 8, 2);
        let x = Tensor::new(vec![1, 5], vec![0.0; 5]).unwrap();
        assert!(gffn.forward(&x, FfnAct::Geglu).is_none());
    }

    #[test]
    fn gated_ffn_zero_gate_gives_zero_output() {
        // If w_gate is all-zero, gate output = act(0) = 0 for SiLU/GELU/ReLU,
        // so the hidden layer is zero, and w_down·0 = 0.
        let mut gffn = GatedFfn::new(2, 4, 5);
        for w in gffn.w_gate.iter_mut() { *w = 0.0; }
        let x = Tensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        let y = gffn.forward(&x, FfnAct::Silu).unwrap();
        for &v in y.data() {
            assert!(close(v, 0.0, 1e-14));
        }
    }

    // ── swiglu_ffn free function ─────────────────────────────────────────────

    #[test]
    fn swiglu_ffn_shapes_and_silu_identity_down() {
        // d=2, ff=2. w_down=identity so out = h = silu(gate)⊙up.
        let x    = Tensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let id   = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let out  = swiglu_ffn(&x, &id, &id, &id, FfnActivation::SiLU).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        // gate=up=[1,0]; h=[silu(1)*1, silu(0)*0]=[0.731..,0]; out=h·I=h.
        assert!(close(out.data()[0], 0.731_058_578_630_005, 1e-9));
        assert!(close(out.data()[1], 0.0, 1e-12));
    }

    #[test]
    fn swiglu_ffn_rejects_mismatch() {
        let x   = Tensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        let bad = Tensor::new(vec![3, 2], vec![0.0; 6]).unwrap();
        assert!(swiglu_ffn(&x, &bad, &bad, &bad, FfnActivation::SiLU).is_none());
    }
}

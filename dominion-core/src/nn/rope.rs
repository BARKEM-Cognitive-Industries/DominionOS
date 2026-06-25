//! Rotary Position Embeddings (RoPE), HuggingFace "rotate-half" convention — the one
//! Llama, Gemma and Qwen weights are trained with, so converted weights apply directly.
//!
//! For head dimension `hd`, the first and second halves of each head vector are treated
//! as the real/imaginary parts of `hd/2` complex numbers, each rotated by an angle
//! `pos · θ^(−2i/hd)`:
//!
//! ```text
//! out[i]        = x[i]·cos_i − x[i+hd/2]·sin_i
//! out[i+hd/2]   = x[i+hd/2]·cos_i + x[i]·sin_i        for i in 0..hd/2
//! ```

use crate::datatypes::Tensor;
use alloc::vec;
use alloc::vec::Vec;

// Trig and log primitives come from the single shared implementation in `crate::math`.
use crate::math::{sin, cos, ln};

/// Precompute `(cos, sin)` tables of shape `[seq, hd/2]` for positions
/// `pos_offset .. pos_offset+seq`. `theta` is the RoPE base (10000 for Gemma/Llama,
/// 1_000_000 for Qwen2.5 long-context). Reusable across all layers of a forward pass.
pub fn rope_tables(seq: usize, head_dim: usize, pos_offset: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let half = head_dim / 2;
    let mut cs = vec![0.0f64; seq * half];
    let mut sn = vec![0.0f64; seq * half];
    // inv_freq[i] = theta^(-2i/hd) = 1 / theta^(2i/hd).
    for i in 0..half {
        let exponent = (2 * i) as f64 / head_dim as f64;
        let inv_freq = 1.0 / pow_f64(theta, exponent);
        for s in 0..seq {
            let pos = (pos_offset + s) as f64;
            let angle = pos * inv_freq;
            cs[s * half + i] = cos(angle);
            sn[s * half + i] = sin(angle);
        }
    }
    (cs, sn)
}

/// Apply RoPE in place to a `[seq, n_heads · head_dim]` tensor using precomputed
/// tables (see [`rope_tables`]). Each row `s` is at position `pos_offset+s` (encoded in
/// the tables). Returns `false` on a shape mismatch (tensor left untouched).
/// Apply RoPE and return a new tensor. Returns None on shape mismatch.
pub fn apply_rope(
    x: &Tensor,
    n_heads: usize,
    head_dim: usize,
    cos_tab: &[f64],
    sin_tab: &[f64],
) -> Option<Tensor> {
    if x.shape().len() != 2 {
        return None;
    }
    let (seq, width) = (x.shape()[0], x.shape()[1]);
    let half = head_dim / 2;
    if width != n_heads * head_dim || head_dim % 2 != 0 || cos_tab.len() != seq * half {
        return None;
    }
    let src = x.data();
    let mut out = src.to_vec();
    for s in 0..seq {
        let crow = &cos_tab[s * half..(s + 1) * half];
        let srow = &sin_tab[s * half..(s + 1) * half];
        for h in 0..n_heads {
            let base = s * width + h * head_dim;
            for i in 0..half {
                let a = src[base + i];
                let b = src[base + half + i];
                let c = crow[i];
                let sx = srow[i];
                out[base + i]        = a * c - b * sx;
                out[base + half + i] = b * c + a * sx;
            }
        }
    }
    Tensor::new(x.shape().to_vec(), out)
}

/// `x^p` for real `p` via `exp(p · ln x)` (x > 0) — delegates to `crate::math`.
fn pow_f64(x: f64, p: f64) -> f64 {
    if p == 0.0 { return 1.0; }
    crate::math::exp(p * ln(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn pos_zero_is_identity() {
        // At position 0 every angle is 0 ⇒ cos 1, sin 0 ⇒ no change.
        let (cs, sn) = rope_tables(1, 4, 0, 10000.0);
        let x = Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = apply_rope(&x, 1, 4, &cs, &sn).unwrap();
        for (got, exp) in y.data().iter().zip([1.0f64, 2.0, 3.0, 4.0].iter()) {
            assert!(close(*got, *exp, 1e-14), "expected {exp}, got {got}");
        }
    }

    #[test]
    fn pos_one_head_dim_two() {
        // hd=2 ⇒ half=1, inv_freq[0]=theta^0=1, angle=pos·1.
        // pos=1: cos(1)=0.5403, sin(1)=0.8415. x=[1,0] → [cos, sin].
        let (cs, sn) = rope_tables(1, 2, 1, 10000.0);
        let x = Tensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let y = apply_rope(&x, 1, 2, &cs, &sn).unwrap();
        assert!(close(y.data()[0], 0.540_302_305_868_14, 1e-10));
        assert!(close(y.data()[1], 0.841_470_984_807_90, 1e-10));
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation is orthogonal ⇒ per-pair L2 norm is preserved.
        let (cs, sn) = rope_tables(3, 8, 5, 10000.0);
        let orig = vec![0.3, -0.7, 1.2, 0.5, -0.1, 0.9, -1.4, 0.2];
        let x = Tensor::new(vec![3, 8], orig.iter().cycle().take(24).cloned().collect()).unwrap();
        let before: f64 = x.data().iter().map(|v| v * v).sum();
        let y = apply_rope(&x, 1, 8, &cs, &sn).unwrap();
        let after: f64 = y.data().iter().map(|v| v * v).sum();
        assert!(close(before, after, 1e-9));
    }

    #[test]
    fn multi_head_independent() {
        // Two heads of dim 2; both should rotate identically at the same position.
        let (cs, sn) = rope_tables(1, 2, 1, 10000.0);
        let x = Tensor::new(vec![1, 4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let y = apply_rope(&x, 2, 2, &cs, &sn).unwrap();
        assert!(close(y.data()[0], y.data()[2], 1e-12));
        assert!(close(y.data()[1], y.data()[3], 1e-12));
    }

    #[test]
    fn ln_and_pow_sane() {
        assert!(close(ln(1.0), 0.0, 1e-12));
        assert!(close(ln(core::f64::consts::E), 1.0, 1e-10));
        assert!(close(pow_f64(2.0, 10.0), 1024.0, 1e-6));
        assert!(close(pow_f64(10000.0, 0.0), 1.0, 1e-12));
    }
}

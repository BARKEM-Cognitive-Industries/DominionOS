//! `nn::norm` — canonical normalisation kernels shared across the `nn` sub-crates.
//!
//! Both functions operate on flat `[n_rows, d]` slices and return a freshly
//! allocated `Vec<f64>` so callers never mutate shared weight tensors.

use alloc::vec::Vec;

/// RMS-norm: out[row, i] = x[row, i] / rms(x[row]) * w[i]
/// where rms = sqrt(mean(x²) + eps).
pub fn rms_norm(x: &[f64], w: &[f64], eps: f64, d: usize) -> Vec<f64> {
    let n = x.len() / d;
    let mut out = alloc::vec![0.0f64; x.len()];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let ms: f64 = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let rms = crate::datatypes::sqrt(ms + eps);
        for j in 0..d {
            out[i * d + j] = row[j] / rms * w[j];
        }
    }
    out
}

/// Layer-norm: out[row, i] = (x[row, i] − mean) / std * w[i] + b[i]
/// where std = sqrt(var + eps).
pub fn layer_norm(x: &[f64], w: &[f64], b: &[f64], eps: f64, d: usize) -> Vec<f64> {
    let n = x.len() / d;
    let mut out = alloc::vec![0.0f64; x.len()];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mean: f64 = row.iter().sum::<f64>() / d as f64;
        let var: f64 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        let std = crate::datatypes::sqrt(var + eps);
        for j in 0..d {
            out[i * d + j] = (row[j] - mean) / std * w[j] + b[j];
        }
    }
    out
}

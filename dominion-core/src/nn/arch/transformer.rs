use alloc::vec::Vec;
use crate::datatypes::Tensor;
use crate::ml::{exp, Rng};
use crate::nn::{NnConfig, NormKind, FfnAct};
use crate::nn::norm::{rms_norm, layer_norm};
use crate::nn::rope::apply_rope;

// ---------------------------------------------------------------------------
// Local math helpers
// ---------------------------------------------------------------------------

/// Dense matrix multiply: out[b, j] = sum_i x[b, i] * w[j, i]   (w is row-major [out_d, in_d])
fn linear(x: &[f64], w: &[f64], batch: usize, in_d: usize, out_d: usize) -> Vec<f64> {
    let mut out = alloc::vec![0.0f64; batch * out_d];
    for b in 0..batch {
        for j in 0..out_d {
            let mut acc = 0.0f64;
            for i in 0..in_d {
                acc += x[b * in_d + i] * w[j * in_d + i];
            }
            out[b * out_d + j] = acc;
        }
    }
    out
}

/// Apply scalar activation, used for non-gated FFN paths.
fn apply_activation(v: f64, act: FfnAct) -> f64 {
    match act {
        FfnAct::Relu | FfnAct::Reglu  => if v > 0.0 { v } else { 0.0 },
        FfnAct::Gelu | FfnAct::Geglu  => {
            // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            let c = 0.797_884_560_802_865_f64; // sqrt(2/pi)
            let inner = c * (v + 0.044_715 * v * v * v);
            0.5 * v * (1.0 + crate::ml::tanh(inner))
        }
        FfnAct::Silu | FfnAct::Swiglu => v * crate::ml::sigmoid(v),
        FfnAct::Linear => v,
    }
}

/// Standard O(N²) attention.
/// q: [seq_q, n_heads, d_head], k/v: [seq_k, n_heads, d_head] flat row-major.
/// `seq_q` is the number of query positions (current tokens being processed).
/// `seq_k` is the total key/value length (past KV-cache tokens + current tokens).
fn attention_standard_inline(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    d_head: usize,
    scale: f64,
    causal: bool,
) -> Vec<f64> {
    let mut out = alloc::vec![0.0f64; seq_q * n_heads * d_head];
    // past_offset: how many KV positions precede the first query position.
    // Under causal masking a query at position qi (within the current chunk)
    // maps to absolute position (seq_k - seq_q + qi) in the full sequence.
    let past_offset = seq_k - seq_q;
    for h in 0..n_heads {
        for qi in 0..seq_q {
            let abs_qi = past_offset + qi; // absolute position of this query token
            // Compute scores against all seq_k keys
            let mut scores = alloc::vec![0.0f64; seq_k];
            for ki in 0..seq_k {
                if causal && ki > abs_qi {
                    scores[ki] = f64::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f64;
                for d in 0..d_head {
                    dot += q[qi * n_heads * d_head + h * d_head + d]
                         * k[ki * n_heads * d_head + h * d_head + d];
                }
                scores[ki] = dot * scale;
            }
            // Numerically stable softmax
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for s in scores.iter_mut() {
                if *s == f64::NEG_INFINITY {
                    *s = 0.0;
                } else {
                    *s = exp(*s - max_s);
                    sum += *s;
                }
            }
            if sum > 0.0 {
                for s in scores.iter_mut() { *s /= sum; }
            }
            // Weighted sum of values
            for ki in 0..seq_k {
                for d in 0..d_head {
                    out[qi * n_heads * d_head + h * d_head + d] +=
                        scores[ki] * v[ki * n_heads * d_head + h * d_head + d];
                }
            }
        }
    }
    out
}


/// Add two flat slices element-wise (a += b).
fn add_inplace(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

/// One transformer block (pre-norm style by default):
///   x = x + attention(norm1(x))
///   x = x + ffn(norm2(x))
pub struct TransformerBlock {
    /// norm1 weights [d_model]
    pub norm1_w: Vec<f64>,
    /// norm1 bias [d_model] — only used for LayerNorm
    pub norm1_b: Vec<f64>,
    /// Attention projection matrices — each [d_model × d_model] (wk/wv may be [kv_dim × d_model])
    pub wq: Vec<f64>,
    pub wk: Vec<f64>,
    pub wv: Vec<f64>,
    pub wo: Vec<f64>,
    /// norm2 weights [d_model]
    pub norm2_w: Vec<f64>,
    /// norm2 bias [d_model]
    pub norm2_b: Vec<f64>,
    /// FFN weights
    pub w1: Vec<f64>, // gate or standard w1  [d_ff × d_model]
    pub w2: Vec<f64>, // down projection      [d_model × d_ff]
    pub w3: Vec<f64>, // up proj (SwiGLU/GeGLU); empty otherwise  [d_ff × d_model]
    /// Architecture dims
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub cfg: NnConfig,
}

impl TransformerBlock {
    /// Construct a new block with Kaiming-uniform weight init.
    /// Returns `None` if `d_model % n_heads != 0`.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_ff: usize,
        cfg: NnConfig,
        seed: u64,
    ) -> Option<Self> {
        if d_model % n_heads != 0 {
            return None;
        }
        let kv_dim = n_kv_heads * (d_model / n_heads);
        let mut rng = Rng::new(seed);
        let kaiming = |rows: usize, cols: usize, rng: &mut Rng| -> Vec<f64> {
            let std = crate::datatypes::sqrt(2.0 / cols as f64);
            (0..rows * cols).map(|_| (rng.next_signed() * 2.0 - 1.0) * std).collect()
        };
        let gated = matches!(cfg.ffn_act, FfnAct::Swiglu | FfnAct::Geglu | FfnAct::Reglu);
        Some(Self {
            norm1_w: alloc::vec![1.0f64; d_model],
            norm1_b: alloc::vec![0.0f64; d_model],
            wq: kaiming(d_model, d_model, &mut rng),
            wk: kaiming(kv_dim, d_model, &mut rng),
            wv: kaiming(kv_dim, d_model, &mut rng),
            wo: kaiming(d_model, d_model, &mut rng),
            norm2_w: alloc::vec![1.0f64; d_model],
            norm2_b: alloc::vec![0.0f64; d_model],
            w1: kaiming(d_ff, d_model, &mut rng),
            w2: kaiming(d_model, d_ff, &mut rng),
            w3: if gated { kaiming(d_ff, d_model, &mut rng) } else { Vec::new() },
            d_model,
            n_heads,
            n_kv_heads,
            d_ff,
            cfg,
        })
    }

    /// Forward pass.
    /// `x`: `[seq_len, d_model]` Tensor.
    /// Returns `[seq_len, d_model]` Tensor.
    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut crate::nn::attention::KvCache>,
        cos_sin: Option<&[f64]>,
    ) -> Option<Tensor> {
        let shape = x.shape();
        if shape.len() != 2 || shape[1] != self.d_model {
            return None;
        }
        let seq = shape[0];
        let d = self.d_model;
        let eps = self.cfg.norm_eps;

        let xd = x.data();

        // ----- helper: normalize a [seq, d] slice -----
        let norm = |data: &[f64], w: &[f64], b: &[f64]| -> Vec<f64> {
            match self.cfg.norm_kind {
                NormKind::Layer => layer_norm(data, w, b, eps, d),
                _ /* Rms | None | Group | Batch */ => rms_norm(data, w, eps, d),
            }
        };

        // ----- PRE-NORM or POST-NORM -----
        // pre_norm=true (default): norm before sublayer, add residual after
        // pre_norm=false: add residual first, norm after (post-norm)

        // ===== Attention sub-layer =====
        let (attn_in, mut residual1): (Vec<f64>, Vec<f64>) = if self.cfg.pre_norm {
            (norm(xd, &self.norm1_w, &self.norm1_b), xd.to_vec())
        } else {
            (xd.to_vec(), xd.to_vec())
        };

        // Project Q, K, V
        let d_head = d / self.n_heads;
        let kv_dim = self.n_kv_heads * d_head;

        let mut q = linear(&attn_in, &self.wq, seq, d, d);
        let mut k = linear(&attn_in, &self.wk, seq, d, kv_dim);
        let v = linear(&attn_in, &self.wv, seq, d, kv_dim);

        // Reshape to [seq, n_heads, d_head] / [seq, n_kv_heads, d_head] for RoPE
        // q is already in correct layout if linear output is [seq, d_model] and
        // we treat it as [seq, n_heads, d_head].
        // Apply RoPE if cos_sin provided.
        // The incoming `cos_sin` slice is interleaved [cos0, sin0, cos1, sin1, ...]
        // per position × per half-dim pair.  `nn::rope::apply_rope` expects separate
        // cos_tab / sin_tab of shape [seq, half].  Split once, then delegate.
        if let Some(cs) = cos_sin {
            let half = d_head / 2;
            let mut cos_tab = alloc::vec![0.0f64; seq * half];
            let mut sin_tab = alloc::vec![0.0f64; seq * half];
            for s in 0..seq {
                for i in 0..half {
                    cos_tab[s * half + i] = cs[s * half * 2 + i * 2];
                    sin_tab[s * half + i] = cs[s * half * 2 + i * 2 + 1];
                }
            }
            let qt_in = Tensor::new(alloc::vec![seq, d], q).unwrap();
            q = match apply_rope(&qt_in, self.n_heads, d_head, &cos_tab, &sin_tab) {
                Some(qt) => qt.data().to_vec(),
                None => qt_in.data().to_vec(),
            };
            let kt_in = Tensor::new(alloc::vec![seq, kv_dim], k).unwrap();
            k = match apply_rope(&kt_in, self.n_kv_heads, d_head, &cos_tab, &sin_tab) {
                Some(kt) => kt.data().to_vec(),
                None => kt_in.data().to_vec(),
            };
        }

        // Append to KV-cache if provided, then use cached K/V
        let (k_for_attn, v_for_attn, seq_k): (Vec<f64>, Vec<f64>, usize) =
            if let Some(cache) = kv_cache {
                if !cache.append(&k, &v) {
                    return None;
                }
                let seq_k = cache.seq_len;
                (cache.k_slice().to_vec(), cache.v_slice().to_vec(), seq_k)
            } else {
                let sk = seq;
                (k, v, sk)
            };

        // GQA: expand k/v from n_kv_heads to n_heads
        let (k_exp, v_exp): (Vec<f64>, Vec<f64>) = if self.n_kv_heads == self.n_heads {
            (k_for_attn, v_for_attn)
        } else {
            let ratio = self.n_heads / self.n_kv_heads;
            let mut ke = alloc::vec![0.0f64; seq_k * self.n_heads * d_head];
            let mut ve = alloc::vec![0.0f64; seq_k * self.n_heads * d_head];
            for s in 0..seq_k {
                for kh in 0..self.n_kv_heads {
                    for r in 0..ratio {
                        let h = kh * ratio + r;
                        for dd in 0..d_head {
                            ke[s * self.n_heads * d_head + h * d_head + dd] =
                                k_for_attn[s * kv_dim + kh * d_head + dd];
                            ve[s * self.n_heads * d_head + h * d_head + dd] =
                                v_for_attn[s * kv_dim + kh * d_head + dd];
                        }
                    }
                }
            }
            (ke, ve)
        };

        let scale = if self.cfg.attn_scale > 0.0 {
            self.cfg.attn_scale
        } else {
            1.0 / crate::datatypes::sqrt(d_head as f64)
        };

        let attn_out = if self.cfg.flash_block > 0 {
            crate::nn::attention::flash_attention_cpu(
                &q, &k_exp, &v_exp,
                seq, seq_k, self.n_heads, d_head,
                scale, self.cfg.causal, self.cfg.flash_block,
            ).unwrap_or_else(|| alloc::vec![0.0f64; seq * self.n_heads * d_head])
        } else {
            attention_standard_inline(
                &q, &k_exp, &v_exp,
                seq, seq_k, self.n_heads, d_head,
                scale, self.cfg.causal,
            )
        };

        // Output projection: [seq, d_model] → [seq, d_model]
        let attn_proj = linear(&attn_out, &self.wo, seq, d, d);

        // Add residual
        add_inplace(&mut residual1, &attn_proj);

        // Post-norm for post-norm style
        let attn_res: Vec<f64> = if !self.cfg.pre_norm {
            norm(&residual1, &self.norm1_w, &self.norm1_b)
        } else {
            residual1
        };

        // ===== FFN sub-layer =====
        let ffn_in: Vec<f64> = if self.cfg.pre_norm {
            norm(&attn_res, &self.norm2_w, &self.norm2_b)
        } else {
            attn_res.clone()
        };
        let mut residual2 = attn_res;

        let ffn_out = self.ffn_forward(&ffn_in, seq);

        add_inplace(&mut residual2, &ffn_out);

        let result: Vec<f64> = if !self.cfg.pre_norm {
            norm(&residual2, &self.norm2_w, &self.norm2_b)
        } else {
            residual2
        };

        Tensor::new(alloc::vec![seq, d], result)
    }

    fn ffn_forward(&self, x: &[f64], seq: usize) -> Vec<f64> {
        let d = self.d_model;
        let df = self.d_ff;
        let act = self.cfg.ffn_act;
        let gated = matches!(act, FfnAct::Swiglu | FfnAct::Geglu | FfnAct::Reglu);

        if gated && !self.w3.is_empty() {
            // gate = act(x @ w1^T),  up = x @ w3^T,  out = (gate * up) @ w2^T
            let gate_pre = linear(x, &self.w1, seq, d, df);
            let up       = linear(x, &self.w3, seq, d, df);
            let mut hidden = alloc::vec![0.0f64; seq * df];
            for i in 0..seq * df {
                hidden[i] = apply_activation(gate_pre[i], act) * up[i];
            }
            linear(&hidden, &self.w2, seq, df, d)
        } else {
            // standard: out = act(x @ w1^T) @ w2^T
            let h_pre = linear(x, &self.w1, seq, d, df);
            let mut hidden = alloc::vec![0.0f64; seq * df];
            for i in 0..seq * df {
                hidden[i] = apply_activation(h_pre[i], act);
            }
            linear(&hidden, &self.w2, seq, df, d)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{NnConfig, NormKind, FfnAct};

    #[test]
    fn test_transformer_block_shape() {
        let cfg = NnConfig::best();
        let block = TransformerBlock::new(16, 4, 4, 32, cfg, 42).unwrap();
        let x = Tensor::zeros(alloc::vec![3, 16]);
        let y = block.forward(&x, None, None).unwrap();
        assert_eq!(y.shape(), &[3, 16]);
    }

    #[test]
    fn test_transformer_swiglu() {
        let mut cfg = NnConfig::best();
        cfg.ffn_act = FfnAct::Swiglu;
        let block = TransformerBlock::new(8, 2, 2, 16, cfg, 7).unwrap();
        let x = Tensor::zeros(alloc::vec![2, 8]);
        let y = block.forward(&x, None, None).unwrap();
        assert_eq!(y.shape(), &[2, 8]);
    }

    #[test]
    fn test_transformer_layernorm() {
        let mut cfg = NnConfig::best();
        cfg.norm_kind = NormKind::Layer;
        let block = TransformerBlock::new(8, 2, 2, 16, cfg, 3).unwrap();
        let data: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let x = Tensor::new(alloc::vec![2, 8], data).unwrap();
        let y = block.forward(&x, None, None).unwrap();
        assert_eq!(y.shape(), &[2, 8]);
    }

    #[test]
    fn test_transformer_invalid_heads() {
        let cfg = NnConfig::best();
        assert!(TransformerBlock::new(10, 3, 3, 20, cfg, 0).is_none());
    }
}

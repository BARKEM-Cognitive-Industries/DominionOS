use alloc::vec::Vec;
use crate::datatypes::Tensor;
use crate::ml::{exp, Rng};

// ---------------------------------------------------------------------------
// Internal math helpers
// ---------------------------------------------------------------------------

/// Dense matmul: out[b, j] = sum_i x[b, i] * w[j, i]   (w row-major [out, in])
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

/// In-place softmax over last dimension, shape [batch, dim].
fn softmax_rows(logits: &mut [f64], batch: usize, dim: usize) {
    for b in 0..batch {
        let row = &mut logits[b * dim..(b + 1) * dim];
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for v in row.iter_mut() {
            *v = exp(*v - max);
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f64) -> f64 {
    x / (1.0 + exp(-x))
}

// ---------------------------------------------------------------------------
// Expert
// ---------------------------------------------------------------------------

/// A standard two-layer FFN expert:  output = W2 · SiLU(W1 · x)
pub struct Expert {
    /// [d_ff, d_model] row-major
    pub w1: Vec<f64>,
    /// [d_model, d_ff] row-major
    pub w2: Vec<f64>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl Expert {
    /// Kaiming-uniform init.
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let std1 = crate::datatypes::sqrt(2.0 / d_model as f64);
        let std2 = crate::datatypes::sqrt(2.0 / d_ff as f64);
        let w1 = (0..d_ff * d_model)
            .map(|_| (rng.next_signed() * 2.0 - 1.0) * std1)
            .collect();
        let w2 = (0..d_model * d_ff)
            .map(|_| (rng.next_signed() * 2.0 - 1.0) * std2)
            .collect();
        Self { w1, w2, d_model, d_ff }
    }

    /// Forward: x is `[d_model]` → `[d_model]`.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // hidden = SiLU(W1 · x)  [d_ff]
        let mut hidden = alloc::vec![0.0f64; self.d_ff];
        for j in 0..self.d_ff {
            let mut acc = 0.0f64;
            for i in 0..self.d_model {
                acc += x[i] * self.w1[j * self.d_model + i];
            }
            hidden[j] = silu(acc);
        }
        // out = W2 · hidden  [d_model]
        let mut out = alloc::vec![0.0f64; self.d_model];
        for j in 0..self.d_model {
            let mut acc = 0.0f64;
            for i in 0..self.d_ff {
                acc += hidden[i] * self.w2[j * self.d_ff + i];
            }
            out[j] = acc;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MoeLayer
// ---------------------------------------------------------------------------

/// Mixture-of-Experts layer: sparse top-K routing.
pub struct MoeLayer {
    /// Router weight matrix [num_experts, d_model]
    pub router: Vec<f64>,
    pub experts: Vec<Expert>,
    pub num_experts: usize,
    pub top_k: usize,
    pub d_model: usize,
    /// Load-balancing auxiliary loss coefficient.
    pub aux_loss_coef: f64,
}

impl MoeLayer {
    /// Construct with random-init experts and router.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
        aux_loss_coef: f64,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        // Router: small uniform init
        let router_std = crate::datatypes::sqrt(1.0 / d_model as f64);
        let router = (0..num_experts * d_model)
            .map(|_| (rng.next_signed() * 2.0 - 1.0) * router_std)
            .collect();
        // Each expert gets a derived seed
        let experts = (0..num_experts)
            .map(|e| Expert::new(d_model, d_ff, seed.wrapping_add(e as u64 + 1)))
            .collect();
        Self { router, experts, num_experts, top_k, d_model, aux_loss_coef }
    }

    /// Forward: `x` is `[n, d_model]`; returns `([n, d_model], aux_loss)`.
    ///
    /// Aux loss formula (Switch Transformer style):
    ///   `aux_loss = aux_loss_coef * num_experts * sum_e(f_e * P_e)`
    /// where `f_e` = fraction of tokens assigned to expert `e`,
    ///       `P_e` = mean router probability for expert `e`.
    pub fn forward(&self, x: &Tensor) -> Option<(Tensor, f64)> {
        let shape = x.shape();
        if shape.len() != 2 || shape[1] != self.d_model {
            return None;
        }
        let n = shape[0];
        let xd = x.data();

        // 1. Router logits → softmax probabilities  [n, num_experts]
        let mut probs = self.route(xd, n);
        softmax_rows(&mut probs, n, self.num_experts);

        // 2. Top-K selection per token
        let (indices, weights) = self.top_k_experts(&probs, n);

        // 3. Compute load-balancing aux loss
        // f_e = (tokens assigned to e) / (n * top_k)
        let mut token_count = alloc::vec![0usize; self.num_experts];
        for i in 0..n {
            for &e in &indices[i] {
                token_count[e] += 1;
            }
        }
        let total_dispatches = (n * self.top_k) as f64;
        // P_e = mean of probs[:, e]
        let mut p_e = alloc::vec![0.0f64; self.num_experts];
        for i in 0..n {
            for e in 0..self.num_experts {
                p_e[e] += probs[i * self.num_experts + e];
            }
        }
        let mut aux_loss = 0.0f64;
        for e in 0..self.num_experts {
            let f_e = token_count[e] as f64 / total_dispatches;
            let p_mean = p_e[e] / n as f64;
            aux_loss += f_e * p_mean;
        }
        aux_loss *= self.aux_loss_coef * self.num_experts as f64;

        // 4. Dispatch tokens to experts and accumulate weighted outputs
        let mut out_data = alloc::vec![0.0f64; n * self.d_model];
        for i in 0..n {
            let x_tok = &xd[i * self.d_model..(i + 1) * self.d_model];
            for (k_idx, &expert_id) in indices[i].iter().enumerate() {
                let w = weights[i][k_idx];
                let expert_out = self.experts[expert_id].forward(x_tok);
                for d in 0..self.d_model {
                    out_data[i * self.d_model + d] += w * expert_out[d];
                }
            }
        }

        let out_tensor = Tensor::new(alloc::vec![n, self.d_model], out_data)?;
        Some((out_tensor, aux_loss))
    }

    /// Compute raw router logits: `x @ router^T` → `[n, num_experts]`
    fn route(&self, x: &[f64], n: usize) -> Vec<f64> {
        linear(x, &self.router, n, self.d_model, self.num_experts)
    }

    /// For each token, pick the top-K experts by softmax probability.
    /// Returns (indices per token, normalised weights per token).
    fn top_k_experts(&self, probs: &[f64], n: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let k = self.top_k.min(self.num_experts);
        let mut all_indices = Vec::with_capacity(n);
        let mut all_weights = Vec::with_capacity(n);

        for i in 0..n {
            let row = &probs[i * self.num_experts..(i + 1) * self.num_experts];

            // Selection sort for top-K (K is typically small: 1-4)
            let mut selected: Vec<usize> = Vec::with_capacity(k);
            let mut used = alloc::vec![false; self.num_experts];
            for _ in 0..k {
                let mut best_e = 0usize;
                let mut best_v = f64::NEG_INFINITY;
                for e in 0..self.num_experts {
                    if !used[e] && row[e] > best_v {
                        best_v = row[e];
                        best_e = e;
                    }
                }
                used[best_e] = true;
                selected.push(best_e);
            }

            // Gather raw weights for selected experts
            let raw_weights: Vec<f64> = selected.iter().map(|&e| row[e]).collect();
            // Normalize to sum=1
            let sum: f64 = raw_weights.iter().sum();
            let norm_weights: Vec<f64> = if sum > 0.0 {
                raw_weights.iter().map(|&w| w / sum).collect()
            } else {
                alloc::vec![1.0 / k as f64; k]
            };

            all_indices.push(selected);
            all_weights.push(norm_weights);
        }
        (all_indices, all_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_shape() {
        let expert = Expert::new(8, 16, 42);
        let x = alloc::vec![0.1f64; 8];
        let out = expert.forward(&x);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_moe_forward_shape() {
        let moe = MoeLayer::new(8, 16, 4, 2, 0.01, 7);
        let data: Vec<f64> = (0..24).map(|i| i as f64 * 0.01).collect();
        let x = Tensor::new(alloc::vec![3, 8], data).unwrap();
        let (out, aux) = moe.forward(&x).unwrap();
        assert_eq!(out.shape(), &[3, 8]);
        assert!(aux >= 0.0);
    }

    #[test]
    fn test_moe_top1() {
        let moe = MoeLayer::new(4, 8, 3, 1, 0.0, 1);
        let x = Tensor::zeros(alloc::vec![2, 4]);
        let (out, _) = moe.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn test_moe_invalid_input() {
        let moe = MoeLayer::new(4, 8, 3, 2, 0.01, 0);
        // wrong d_model
        let x = Tensor::zeros(alloc::vec![2, 5]);
        assert!(moe.forward(&x).is_none());
    }
}

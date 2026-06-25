use alloc::vec::Vec;
use crate::datatypes::Tensor;
use crate::ml::Rng;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Kaiming-uniform bound: sqrt(6 / fan_in)
#[inline]
fn kaiming_bound(fan_in: usize) -> f64 {
    crate::datatypes::sqrt(6.0 / fan_in as f64)
}

/// ReLU
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Element-wise tensor add: a + b → new Vec (both must have same length).
fn tensor_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Add `main` feature-map to `skip` element-wise, apply ReLU, and wrap in a
/// Tensor with the same shape as `main`. Returns `None` if the lengths differ.
/// Shared by [`BottleneckBlock`] and [`BasicBlock`].
fn residual_relu(main: Tensor, skip: &[f64]) -> Option<Tensor> {
    let h_data = main.data();
    if h_data.len() != skip.len() {
        return None;
    }
    let summed: Vec<f64> = tensor_add(h_data, skip)
        .into_iter()
        .map(relu)
        .collect();
    Tensor::new(main.shape().to_vec(), summed)
}

// ---------------------------------------------------------------------------
// ConvLayer
// ---------------------------------------------------------------------------

/// 2-D convolutional layer.
///
/// Weight layout: `[out_ch, in_ch, kH, kW]` row-major.
pub struct ConvLayer {
    /// `[out_ch, in_ch, kH, kW]`
    pub weight: Vec<f64>,
    /// `[out_ch]`
    pub bias: Vec<f64>,
    pub in_ch: usize,
    pub out_ch: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
}

impl ConvLayer {
    /// Construct with Kaiming-uniform weight init.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Self {
        let fan_in = in_ch * kh * kw;
        let bound = kaiming_bound(fan_in);
        let mut rng = Rng::new(seed);
        let weight = (0..out_ch * in_ch * kh * kw)
            .map(|_| (rng.next_signed() * 2.0 - 1.0) * bound)
            .collect();
        let bias = alloc::vec![0.0f64; out_ch];
        Self { weight, bias, in_ch, out_ch, kh, kw, stride, padding }
    }

    /// Output spatial dimension for one axis.
    #[inline]
    fn out_dim(in_size: usize, k: usize, stride: usize, pad: usize) -> usize {
        (in_size + 2 * pad).saturating_sub(k) / stride + 1
    }

    /// im2col: gather input patches into `[batch * H_out * W_out, in_ch * kH * kW]`.
    fn im2col(
        x: &[f64],
        batch: usize,
        in_ch: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
        h_out: usize,
        w_out: usize,
    ) -> Vec<f64> {
        let col_rows = batch * h_out * w_out;
        let col_cols = in_ch * kh * kw;
        let mut col = alloc::vec![0.0f64; col_rows * col_cols];
        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = b * h_out * w_out + oh * w_out + ow;
                    let mut col_j = 0usize;
                    for c in 0..in_ch {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride + khi;
                                let iw = ow * stride + kwi;
                                // With padding, check bounds
                                let val = if ih < pad || iw < pad {
                                    0.0
                                } else {
                                    let ih = ih - pad;
                                    let iw = iw - pad;
                                    if ih < h && iw < w {
                                        x[b * in_ch * h * w + c * h * w + ih * w + iw]
                                    } else {
                                        0.0
                                    }
                                };
                                col[row_idx * col_cols + col_j] = val;
                                col_j += 1;
                            }
                        }
                    }
                }
            }
        }
        col
    }

    /// im2col + weight matmul + bias, optional ReLU fused in.
    fn conv_impl(&self, x: &Tensor, fused_relu: bool) -> Option<Tensor> {
        let shape = x.shape();
        if shape.len() != 4 || shape[1] != self.in_ch {
            return None;
        }
        let (batch, _ic, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = Self::out_dim(h, self.kh, self.stride, self.padding);
        let w_out = Self::out_dim(w, self.kw, self.stride, self.padding);

        let col = Self::im2col(
            x.data(), batch, self.in_ch, h, w,
            self.kh, self.kw, self.stride, self.padding,
            h_out, w_out,
        );

        // col: [batch*h_out*w_out, in_ch*kh*kw]
        // weight: [out_ch, in_ch*kh*kw]
        // result: [batch*h_out*w_out, out_ch]
        let n_rows = batch * h_out * w_out;
        let k = self.in_ch * self.kh * self.kw;
        let mut out = alloc::vec![0.0f64; n_rows * self.out_ch];
        for r in 0..n_rows {
            for oc in 0..self.out_ch {
                let mut acc = self.bias[oc];
                for ki in 0..k {
                    acc += col[r * k + ki] * self.weight[oc * k + ki];
                }
                out[r * self.out_ch + oc] = if fused_relu { relu(acc) } else { acc };
            }
        }

        // Reshape from [batch*h_out*w_out, out_ch] to [batch, out_ch, h_out, w_out]
        let mut reordered = alloc::vec![0.0f64; batch * self.out_ch * h_out * w_out];
        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let r = b * h_out * w_out + oh * w_out + ow;
                    for oc in 0..self.out_ch {
                        reordered[b * self.out_ch * h_out * w_out
                            + oc * h_out * w_out
                            + oh * w_out + ow] = out[r * self.out_ch + oc];
                    }
                }
            }
        }

        Tensor::new(alloc::vec![batch, self.out_ch, h_out, w_out], reordered)
    }

    /// Standard convolution forward: `[batch, in_ch, H, W] → [batch, out_ch, H_out, W_out]`.
    pub fn forward(&self, x: &Tensor) -> Option<Tensor> {
        self.conv_impl(x, false)
    }

    /// Fused conv + ReLU forward.
    pub fn forward_relu(&self, x: &Tensor) -> Option<Tensor> {
        self.conv_impl(x, true)
    }

    /// Fuse batch-norm parameters into conv weights (offline, once before inference).
    ///
    /// `w_fused = w * (gamma / sqrt(running_var + eps))`
    /// `b_fused = (b - running_mean) * gamma / sqrt(running_var + eps) + beta`
    pub fn fuse_batchnorm(
        &mut self,
        running_mean: &[f64],
        running_var: &[f64],
        gamma: &[f64],
        beta: &[f64],
        eps: f64,
    ) -> bool {
        if running_mean.len() != self.out_ch
            || running_var.len() != self.out_ch
            || gamma.len() != self.out_ch
            || beta.len() != self.out_ch
        {
            return false;
        }
        let k = self.in_ch * self.kh * self.kw;
        for oc in 0..self.out_ch {
            let scale = gamma[oc] / crate::datatypes::sqrt(running_var[oc] + eps);
            // Update bias
            self.bias[oc] = (self.bias[oc] - running_mean[oc]) * scale + beta[oc];
            // Scale all weights for this output channel
            for ki in 0..k {
                self.weight[oc * k + ki] *= scale;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// BottleneckBlock
// ---------------------------------------------------------------------------

/// ResNet bottleneck block: 1×1 → 3×3 → 1×1, with optional skip projection.
pub struct BottleneckBlock {
    /// 1×1 reduce
    pub conv1: ConvLayer,
    /// 3×3 spatial
    pub conv2: ConvLayer,
    /// 1×1 expand
    pub conv3: ConvLayer,
    /// 1×1 projection skip (when dims or stride change)
    pub skip: Option<ConvLayer>,
}

impl BottleneckBlock {
    /// `in_ch` → bottleneck of `mid_ch` → `out_ch`.
    /// `stride` is applied to `conv2` (3×3).
    pub fn new(in_ch: usize, mid_ch: usize, out_ch: usize, stride: usize, seed: u64) -> Self {
        let conv1 = ConvLayer::new(in_ch, mid_ch, 1, 1, 1, 0, seed);
        let conv2 = ConvLayer::new(mid_ch, mid_ch, 3, 3, stride, 1, seed.wrapping_add(1));
        let conv3 = ConvLayer::new(mid_ch, out_ch, 1, 1, 1, 0, seed.wrapping_add(2));
        let skip = if in_ch != out_ch || stride != 1 {
            Some(ConvLayer::new(in_ch, out_ch, 1, 1, stride, 0, seed.wrapping_add(3)))
        } else {
            None
        };
        Self { conv1, conv2, conv3, skip }
    }

    /// Forward: `[batch, in_ch, H, W] → [batch, out_ch, H_out, W_out]`.
    pub fn forward(&self, x: &Tensor) -> Option<Tensor> {
        // Compute skip path
        let skip_out: Vec<f64> = match &self.skip {
            Some(s) => s.forward(x)?.into_raw_data(),
            None    => x.data().to_vec(),
        };

        // Main path: conv1 → ReLU → conv2 → ReLU → conv3
        let h = self.conv1.forward_relu(x)?;
        let h = self.conv2.forward_relu(&h)?;
        let h = self.conv3.forward(&h)?;

        // Add residual then ReLU
        residual_relu(h, &skip_out)
    }
}

// ---------------------------------------------------------------------------
// BasicBlock
// ---------------------------------------------------------------------------

/// ResNet basic block: 2 conv layers with skip connection.
pub struct BasicBlock {
    pub conv1: ConvLayer,
    pub conv2: ConvLayer,
    /// Skip projection when dims/stride change.
    pub skip: Option<ConvLayer>,
}

impl BasicBlock {
    pub fn new(in_ch: usize, out_ch: usize, stride: usize, seed: u64) -> Self {
        let conv1 = ConvLayer::new(in_ch, out_ch, 3, 3, stride, 1, seed);
        let conv2 = ConvLayer::new(out_ch, out_ch, 3, 3, 1, 1, seed.wrapping_add(1));
        let skip = if in_ch != out_ch || stride != 1 {
            Some(ConvLayer::new(in_ch, out_ch, 1, 1, stride, 0, seed.wrapping_add(2)))
        } else {
            None
        };
        Self { conv1, conv2, skip }
    }

    /// Forward: `[batch, in_ch, H, W] → [batch, out_ch, H_out, W_out]`.
    pub fn forward(&self, x: &Tensor) -> Option<Tensor> {
        // Skip path
        let skip_out: Vec<f64> = match &self.skip {
            Some(s) => s.forward(x)?.into_raw_data(),
            None    => x.data().to_vec(),
        };

        // Main path: conv1 → ReLU → conv2
        let h = self.conv1.forward_relu(x)?;
        let h = self.conv2.forward(&h)?;

        // Add residual then ReLU
        residual_relu(h, &skip_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: make a [batch, ch, h, w] tensor filled with a constant.
    fn make_tensor(batch: usize, ch: usize, h: usize, w: usize, val: f64) -> Tensor {
        Tensor::new(
            alloc::vec![batch, ch, h, w],
            alloc::vec![val; batch * ch * h * w],
        )
        .unwrap()
    }

    #[test]
    fn test_conv_output_shape_same_padding() {
        // 3×3 kernel, padding=1, stride=1 → same spatial
        let conv = ConvLayer::new(3, 8, 3, 3, 1, 1, 0);
        let x = make_tensor(2, 3, 16, 16, 0.5);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 16, 16]);
    }

    #[test]
    fn test_conv_output_shape_stride2() {
        // 3×3, stride=2, pad=1 → halved spatial
        let conv = ConvLayer::new(3, 16, 3, 3, 2, 1, 1);
        let x = make_tensor(1, 3, 8, 8, 1.0);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 16, 4, 4]);
    }

    #[test]
    fn test_conv_relu_nonneg() {
        let conv = ConvLayer::new(1, 4, 3, 3, 1, 1, 99);
        let x = make_tensor(1, 1, 6, 6, -2.0);
        let y = conv.forward_relu(&x).unwrap();
        for &v in y.data() {
            assert!(v >= 0.0, "ReLU output must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_conv_invalid_channels() {
        let conv = ConvLayer::new(3, 8, 3, 3, 1, 1, 0);
        let x = make_tensor(1, 5, 8, 8, 0.0); // wrong in_ch
        assert!(conv.forward(&x).is_none());
    }

    #[test]
    fn test_bottleneck_same_dims() {
        // in_ch == out_ch, stride == 1 → identity skip
        let block = BottleneckBlock::new(16, 8, 16, 1, 42);
        assert!(block.skip.is_none());
        let x = make_tensor(2, 16, 8, 8, 0.1);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 16, 8, 8]);
    }

    #[test]
    fn test_bottleneck_dim_change() {
        // stride=2 → needs projection skip
        let block = BottleneckBlock::new(16, 8, 32, 2, 7);
        assert!(block.skip.is_some());
        let x = make_tensor(1, 16, 8, 8, 0.5);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 32, 4, 4]);
    }

    #[test]
    fn test_bottleneck_output_nonneg() {
        // Final ReLU must ensure no negatives
        let block = BottleneckBlock::new(4, 2, 4, 1, 13);
        let x = make_tensor(1, 4, 6, 6, -1.0);
        let y = block.forward(&x).unwrap();
        for &v in y.data() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_basic_block_identity_skip() {
        let block = BasicBlock::new(8, 8, 1, 3);
        assert!(block.skip.is_none());
        let x = make_tensor(1, 8, 6, 6, 0.2);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 6, 6]);
    }

    #[test]
    fn test_basic_block_projection_skip() {
        let block = BasicBlock::new(4, 8, 2, 5);
        assert!(block.skip.is_some());
        let x = make_tensor(2, 4, 8, 8, 0.3);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 4, 4]);
    }

    #[test]
    fn test_fuse_batchnorm() {
        let mut conv = ConvLayer::new(2, 4, 3, 3, 1, 1, 0);
        let running_mean = alloc::vec![0.1f64; 4];
        let running_var  = alloc::vec![1.0f64; 4];
        let gamma        = alloc::vec![1.0f64; 4];
        let beta         = alloc::vec![0.0f64; 4];
        let ok = conv.fuse_batchnorm(&running_mean, &running_var, &gamma, &beta, 1e-5);
        assert!(ok);
    }
}

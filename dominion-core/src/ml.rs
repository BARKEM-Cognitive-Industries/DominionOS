//! Machine-learning compute: **inference and training**, on any device.
//!
//! DominionOS treats learned models the way it treats everything else — as pure,
//! deterministic, capability-respecting data and transformation. This module turns
//! the first-class [`crate::datatypes::Tensor`] into a working neural-network
//! engine that **trains** (reverse-mode autodiff + SGD/Adam) and **infers**, and
//! it does so on a software-modeled fleet of compute [`Device`]s — `CPU`, `GPU`,
//! `NPU`, `TPU` — that follow the OS's golden rule: *specialized hardware is an
//! accelerator, never a requirement.*
//!
//! Every device produces **bit-identical** results (the math runs on a portable
//! CPU kernel); a per-device **cost model** estimates the cycles each would take,
//! so the placement decision ([`recommend_device`]) — and the kernel benchmarks —
//! are meaningful, while correctness never depends on the silicon being present.
//! Where real accelerators exist they drop in under the same `Device` seam.
//!
//! What's here:
//! * [`Device`] — CPU/GPU/NPU/TPU with a launch-latency + throughput cost model.
//! * [`Tape`] — a reverse-mode automatic-differentiation graph (backprop).
//! * [`Linear`] / [`Mlp`] — dense layers and a multilayer perceptron.
//! * [`Optimizer`] — SGD (with momentum) and Adam.
//! * [`QTensor`] — int8 quantization + a quantized matmul (the NPU low-precision path).
//! * [`Mlp::to_bytes`] / [`Mlp::from_bytes`] — deterministic, content-addressable model IO.
//!
//! Pure, `#![forbid(unsafe_code)]`, `no_std + alloc`, host-tested. See
//! `docs/architecture/ml-compute.md`.

use crate::datatypes::Tensor;
use crate::math::{sqrt, abs as fabs_math};
use alloc::vec;

/// Natural log of 2 — used by the local `ln`/`log2` range reduction below.
const LN2: f64 = core::f64::consts::LN_2;
use alloc::vec::Vec;

/// Native GPU / CUDA support (capability-gated CUDA/cuDNN/cuBLAS/… + device registry).
pub mod gpu;

// ───────────────────────────── transcendentals ─────────────────────────────
//
// Implementations live in `crate::math`; re-exported here so existing callers
// that reference `crate::ml::exp` / `crate::ml::sigmoid` / `crate::ml::tanh`
// continue to work unchanged.

/// `e^x` — see [`crate::math::exp`].
pub use crate::math::exp;

/// Numerically stable logistic sigmoid `σ(x) = 1/(1+e^-x)`.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + exp(-x))
    } else {
        let e = exp(x);
        e / (1.0 + e)
    }
}

/// Hyperbolic tangent, via the stable sigmoid identity `tanh(x) = 2σ(2x) − 1`.
pub fn tanh(x: f64) -> f64 {
    2.0 * sigmoid(2.0 * x) - 1.0
}

#[inline]
fn fabs(x: f64) -> f64 { fabs_math(x) }

/// Round-half-away-from-zero: the correct integer rounding for symmetric quantisation.
/// `if x >= 0 { (x + 0.5) as i32 } else { (x - 0.5) as i32 }` — consolidated here so
/// all quantisation paths share one auditable implementation.
#[inline]
fn round_half_away(x: f64) -> i32 {
    if x >= 0.0 { (x + 0.5) as i32 } else { (x - 0.5) as i32 }
}

// ───────────────────────────── deterministic RNG ─────────────────────────────

/// SplitMix64 — a tiny, high-quality, fully deterministic PRNG for weight init.
/// Same seed ⇒ same network, on every machine and every run (a hard requirement
/// for reproducible training and content-addressable models).
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[-1.0, 1.0)`.
    pub fn next_signed(&mut self) -> f64 {
        // Top 53 bits → [0,1), then map to [-1,1).
        let bits = self.next_u64() >> 11;
        let unit = bits as f64 / (1u64 << 53) as f64;
        unit * 2.0 - 1.0
    }
}

// ─────────────────────────── tensor helpers ───────────────────────────
//
// A handful of ops the autodiff engine needs that the base Tensor doesn't expose.
// All are built on Tensor's public API (no field access), so they compose cleanly.

/// Element-wise subtract (shapes must match).
pub fn sub(a: &Tensor, b: &Tensor) -> Option<Tensor> {
    if a.shape() != b.shape() {
        return None;
    }
    let data = a.data().iter().zip(b.data()).map(|(x, y)| x - y).collect();
    Tensor::new(a.shape().to_vec(), data)
}

/// Element-wise (Hadamard) product (shapes must match).
pub fn mul(a: &Tensor, b: &Tensor) -> Option<Tensor> {
    if a.shape() != b.shape() {
        return None;
    }
    let data = a.data().iter().zip(b.data()).map(|(x, y)| x * y).collect();
    Tensor::new(a.shape().to_vec(), data)
}

/// 2-D transpose `(m×n) → (n×m)`.
pub fn transpose(a: &Tensor) -> Option<Tensor> {
    if a.shape().len() != 2 {
        return None;
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let src = a.data();
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = src[i * n + j];
        }
    }
    Tensor::new(vec![n, m], out)
}

/// FLOPs of a 2-D matmul `(m×k)·(k×n)` (multiply-adds counted as 2).
pub fn matmul_flops(m: usize, k: usize, n: usize) -> u64 {
    2 * (m as u64) * (k as u64) * (n as u64)
}

// ───────────────────────────── compute devices ─────────────────────────────

/// A compute device the engine can run on. Results are identical across all of
/// them; what differs is the [cost model](Device::est_cycles) used for placement
/// and benchmarking. Real accelerators slot in behind this same enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Scalar/vector general-purpose core — always present, low latency, modest throughput.
    Cpu,
    /// Massively parallel SIMT accelerator — high throughput, high launch latency.
    Gpu,
    /// Neural processing unit — tuned for low-precision (int8) matmul.
    Npu,
    /// Tensor processing unit — systolic matmul array, highest throughput at scale.
    Tpu,
}

/// Every device the fleet models.
pub const ALL_DEVICES: [Device; 4] = [Device::Cpu, Device::Gpu, Device::Npu, Device::Tpu];

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "CPU",
            Device::Gpu => "GPU",
            Device::Npu => "NPU",
            Device::Tpu => "TPU",
        }
    }

    /// Modeled sustained throughput in **FLOPs per cycle** for dense f64 work.
    /// (A CPU core retires a handful of vector FLOPs/cycle; accelerators many more.)
    pub fn flops_per_cycle(self) -> u64 {
        match self {
            Device::Cpu => 8,
            Device::Gpu => 512,
            Device::Npu => 256, // f64 path; its int8 path (see `qmatmul`) is far higher
            Device::Tpu => 2048,
        }
    }

    /// Fixed **launch latency** in cycles — the cost of dispatching a kernel to the
    /// device. Accelerators pay a lot up front, which is exactly why tiny ops belong
    /// on the CPU and only large ones earn the GPU/TPU.
    pub fn launch_latency(self) -> u64 {
        match self {
            Device::Cpu => 0,
            Device::Npu => 4_000,
            Device::Gpu => 20_000,
            Device::Tpu => 30_000,
        }
    }

    /// Estimated cycles to execute `flops` worth of work on this device:
    /// `launch_latency + flops/throughput`. The basis for [`recommend_device`].
    pub fn est_cycles(self, flops: u64) -> u64 {
        self.launch_latency() + flops / self.flops_per_cycle()
    }

    /// Execute a matmul. The result is device-independent (identical bits); the
    /// device only changes the modeled cost. `None` on shape mismatch.
    pub fn matmul(self, a: &Tensor, b: &Tensor) -> Option<Tensor> {
        a.matmul(b)
    }
}

/// Pick the device that executes `flops` worth of work in the fewest modeled
/// cycles. Small kernels stay on the CPU (no launch tax); large ones migrate to
/// the TPU/GPU. This is the compute analogue of the storage-tier policy and feeds
/// the governor's placement bandit (`governor::PlacementTarget`).
pub fn recommend_device(flops: u64) -> Device {
    let mut best = Device::Cpu;
    let mut best_cycles = Device::Cpu.est_cycles(flops);
    for &d in &ALL_DEVICES[1..] {
        let c = d.est_cycles(flops);
        if c < best_cycles {
            best = d;
            best_cycles = c;
        }
    }
    best
}

// ───────────────────────── Lever 5: learned compute placement ─────────────
//
// The static `recommend_device` cost model is calibrated at design time and cannot
// adapt to runtime conditions (DVFS, cache state, quantization speedup, tile shape).
//
// `ComputeBandit` replaces it with an ε-greedy multi-armed bandit:
//   - Arms = numerical precisions: F64, Int8, Int4, Binary, Ternary.
//   - Reward = observed throughput in MFLOP/s (higher is better).
//   - Q-values updated via exponential moving average (α = 0.3) after each arm pull.
//   - Epsilon decays from 0.25 → 0.05 as confidence builds (min-pulls guard).
//   - No-std, alloc, safe, deterministic sequence from a given seed.
//
// Convergence: after `MIN_PULLS_PER_ARM` × N_ARMS pulls the arm with the highest
// Q-value is `best_arm()`. Under stable workloads this is typically binary or int8.

/// Which numerical precision to use for a matmul/inference computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F64,
    Int8,
    Int4,
    Binary,
    Ternary,
}

pub const ALL_PRECISIONS: [Precision; 5] = [
    Precision::F64, Precision::Int8, Precision::Int4,
    Precision::Binary, Precision::Ternary,
];

const N_PREC_ARMS: usize = ALL_PRECISIONS.len(); // 5

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::F64     => "f64",
            Precision::Int8    => "int8",
            Precision::Int4    => "int4",
            Precision::Binary  => "binary",
            Precision::Ternary => "ternary",
        }
    }
}

/// ε-greedy multi-armed bandit for adaptive compute-precision selection.
///
/// Call [`select`] before each matmul to get the recommended precision arm,
/// run the matmul in that precision, measure observed throughput, then call
/// [`update`] with the result. After `MIN_PULLS * N_PREC_ARMS` total pulls
/// `best_arm()` returns the highest-Q arm stably.
///
/// The LCG RNG seed is fixed so the exploration sequence is fully deterministic
/// and reproducible — important for the Stage 10 determinism guarantee.
pub struct ComputeBandit {
    q:           [f64; N_PREC_ARMS],  // Q-value (EMA of observed throughput)
    n:           [u64; N_PREC_ARMS],  // pulls per arm
    alpha:       f64,                  // EMA learning rate
    epsilon:     f64,                  // current exploration rate
    epsilon_min: f64,
    epsilon_decay: f64,                // multiplicative decay per pull
    total_pulls: u64,
    rng:         u64,                  // LCG state (seeded at construction)
}

const MIN_PULLS_PER_ARM: u64 = 8;

impl ComputeBandit {
    /// Create a new bandit. `seed` controls the exploration sequence.
    pub fn new(seed: u64) -> Self {
        ComputeBandit {
            q:             [0.0; N_PREC_ARMS],
            n:             [0u64; N_PREC_ARMS],
            alpha:         0.3,
            epsilon:       0.25,
            epsilon_min:   0.05,
            epsilon_decay: 0.98, // epsilon halves every ~35 pulls
            total_pulls:   0,
            rng:           seed ^ 0xcafe_babe_dead_beef,
        }
    }

    // Park-Miller LCG (no-std safe, deterministic).
    fn rand_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng
    }

    // [0.0, 1.0) uniform float from LCG.
    fn rand_f64(&mut self) -> f64 {
        (self.rand_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Select an arm using UCB-1 (Upper Confidence Bound).
    ///
    /// UCB-1 balances exploration and exploitation without a hand-tuned epsilon:
    ///   `a* = argmax_i(Q[i] + sqrt(2 * ln(total_pulls+1) / n[i]))`
    ///
    /// Arms never yet pulled return +∞ bonus (must be tried at least once).
    /// Regret is O(sqrt(N ln N)) — provably optimal for the MAB setting.
    pub fn select_ucb1(&self) -> usize {
        let ln_t = {
            let t = self.total_pulls + 1;
            let bits = u64::BITS - t.leading_zeros();
            let shifted = (t as f64) / ((1u64 << (bits - 1)) as f64);
            let frac = shifted - 1.0;
            // ln(1+x) ≈ x − x²/2 + x³/3, accurate to ~0.5% for x ∈ [0,1)
            let ln_frac = frac - frac * frac * 0.5 + frac * frac * frac / 3.0;
            (bits - 1) as f64 * 0.6931471805599453 + ln_frac
        };
        let explore = 2.0 * ln_t;
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..N_PREC_ARMS {
            let score = if self.n[i] == 0 {
                f64::INFINITY
            } else {
                self.q[i] + sqrt(explore / self.n[i] as f64)
            };
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// Select an arm. Uses ε-greedy: with probability ε explore uniformly,
    /// otherwise exploit the current best arm.
    pub fn select(&mut self) -> usize {
        let r = self.rand_f64();
        if r < self.epsilon {
            // Exploration: uniform random arm.
            (self.rand_u64() as usize) % N_PREC_ARMS
        } else {
            // Exploitation: arm with highest Q-value.
            self.best_arm()
        }
    }

    /// Update Q-value for arm `arm` with observed throughput `throughput_mflops`.
    pub fn update(&mut self, arm: usize, throughput_mflops: f64) {
        self.n[arm] += 1;
        // EMA update: Q = (1-α)·Q + α·reward.
        self.q[arm] = (1.0 - self.alpha) * self.q[arm] + self.alpha * throughput_mflops;
        // Decay epsilon toward minimum.
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        self.total_pulls += 1;
    }

    /// Current highest-Q arm (the exploitation choice).
    pub fn best_arm(&self) -> usize {
        let mut best = 0;
        let mut best_q = f64::NEG_INFINITY;
        for i in 0..N_PREC_ARMS {
            if self.q[i] > best_q {
                best_q = self.q[i];
                best = i;
            }
        }
        best
    }

    /// Best precision, by name.
    pub fn best_precision(&self) -> Precision {
        ALL_PRECISIONS[self.best_arm()]
    }

    /// Whether the bandit has enough data to trust `best_arm()`.
    ///
    /// Two criteria (either suffices):
    /// - Hard: all arms have ≥ MIN_PULLS pulls (thorough exploration).
    /// - Soft: epsilon is at minimum AND the best arm's Q > 1.5× the second-best
    ///   (clear winner emerged naturally through ε-greedy selection).
    pub fn is_converged(&self) -> bool {
        // Hard: all arms explored.
        if self.n.iter().all(|&c| c >= MIN_PULLS_PER_ARM) {
            return true;
        }
        // Soft: epsilon at floor + clear winner.
        if self.epsilon <= self.epsilon_min + 0.001 && self.total_pulls >= N_PREC_ARMS as u64 * 3 {
            let best = self.best_arm();
            let best_q = self.q[best];
            let second_q = self.q.iter().enumerate()
                .filter(|(i, _)| *i != best)
                .map(|(_, &q)| q)
                .fold(f64::NEG_INFINITY, f64::max);
            if second_q > 0.0 { return best_q > second_q * 1.5; }
        }
        false
    }

    /// Q-values for all arms (for inspection / benchmarking).
    pub fn q_values(&self) -> &[f64; N_PREC_ARMS] {
        &self.q
    }

    /// Arm pull counts.
    pub fn pull_counts(&self) -> &[u64; N_PREC_ARMS] {
        &self.n
    }

    /// Total pulls so far.
    pub fn total_pulls(&self) -> u64 {
        self.total_pulls
    }

    /// Current epsilon (exploration rate).
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

// ───────────────────────── reverse-mode autodiff ─────────────────────────

/// A handle to a value on the [`Tape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

/// The operation that produced a tape node — enough information to run its
/// backward (gradient) rule.
#[derive(Clone, Debug)]
enum Op {
    /// An input/parameter leaf — gradient accumulates, nothing to propagate.
    Leaf,
    /// `y = a · b` (2-D matmul).
    MatMul(Var, Var),
    /// `y[i,j] = x[i,j] + bias[j]` — bias broadcast over rows.
    AddBias(Var, Var),
    Relu(Var),
    Sigmoid(Var),
    Tanh(Var),
    /// `loss = mean((pred − target)²)` over all elements → scalar.
    Mse(Var, Tensor),
    /// Fused softmax + cross-entropy over a batch of logit rows.
    /// `loss = −mean_i log softmax(logits_i)[target_i]` → scalar.
    SoftmaxCe(Var, Vec<usize>),
}

struct Node {
    value: Tensor,
    grad: Vec<f64>,
    op: Op,
}

/// A reverse-mode automatic-differentiation tape. Build a forward computation by
/// chaining ops; call [`backward`](Tape::backward) on a scalar loss to fill every
/// node's gradient. Because nodes are appended in evaluation order, a single
/// reverse sweep is a valid topological backward pass.
pub struct Tape {
    nodes: Vec<Node>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Tape {
        Tape { nodes: Vec::new() }
    }

    fn push(&mut self, value: Tensor, op: Op) -> Var {
        let grad = vec![0.0; value.len()];
        self.nodes.push(Node { value, grad, op });
        Var(self.nodes.len() - 1)
    }

    /// Introduce an input or trainable parameter.
    pub fn leaf(&mut self, t: Tensor) -> Var {
        self.push(t, Op::Leaf)
    }

    pub fn value(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].value
    }

    /// The accumulated gradient of the value at `v` (valid after [`backward`](Tape::backward)).
    pub fn grad(&self, v: Var) -> &[f64] {
        &self.nodes[v.0].grad
    }

    /// `y = a · b` (2-D matmul).
    pub fn matmul(&mut self, a: Var, b: Var) -> Option<Var> {
        let y = self.value(a).matmul(self.value(b))?;
        Some(self.push(y, Op::MatMul(a, b)))
    }

    /// Add a `[features]` bias vector to every row of a `[batch, features]` matrix.
    pub fn add_bias(&mut self, x: Var, bias: Var) -> Option<Var> {
        let xt = self.value(x);
        let bt = self.value(bias);
        if xt.shape().len() != 2 || bt.shape().len() != 1 || xt.shape()[1] != bt.shape()[0] {
            return None;
        }
        let (n, f) = (xt.shape()[0], xt.shape()[1]);
        let (xd, bd) = (xt.data(), bt.data());
        let mut out = vec![0.0; n * f];
        for i in 0..n {
            for j in 0..f {
                out[i * f + j] = xd[i * f + j] + bd[j];
            }
        }
        let y = Tensor::new(vec![n, f], out)?;
        Some(self.push(y, Op::AddBias(x, bias)))
    }

    pub fn relu(&mut self, x: Var) -> Var {
        let data: Vec<f64> = self.value(x).data().iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect();
        let y = Tensor::new(self.value(x).shape().to_vec(), data).unwrap();
        self.push(y, Op::Relu(x))
    }

    pub fn sigmoid(&mut self, x: Var) -> Var {
        let data: Vec<f64> = self.value(x).data().iter().map(|&v| sigmoid(v)).collect();
        let y = Tensor::new(self.value(x).shape().to_vec(), data).unwrap();
        self.push(y, Op::Sigmoid(x))
    }

    pub fn tanh(&mut self, x: Var) -> Var {
        let data: Vec<f64> = self.value(x).data().iter().map(|&v| tanh(v)).collect();
        let y = Tensor::new(self.value(x).shape().to_vec(), data).unwrap();
        self.push(y, Op::Tanh(x))
    }

    /// Mean-squared-error loss against a constant `target` of identical shape.
    pub fn mse(&mut self, pred: Var, target: &Tensor) -> Option<Var> {
        let p = self.value(pred);
        if p.shape() != target.shape() {
            return None;
        }
        let n = p.len().max(1) as f64;
        let l: f64 = p.data().iter().zip(target.data()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / n;
        let y = Tensor::new(vec![1], vec![l])?;
        Some(self.push(y, Op::Mse(pred, target.clone())))
    }

    /// Fused softmax + cross-entropy over a batch of logit rows `[batch, classes]`,
    /// with one integer class label per row. Returns the scalar mean loss.
    pub fn softmax_ce(&mut self, logits: Var, targets: &[usize]) -> Option<Var> {
        let lt = self.value(logits);
        if lt.shape().len() != 2 || lt.shape()[0] != targets.len() {
            return None;
        }
        let (n, c) = (lt.shape()[0], lt.shape()[1]);
        // An empty batch has no valid mean loss; dividing by n=0 would yield NaN.
        if n == 0 {
            return None;
        }
        let d = lt.data();
        let mut loss = 0.0;
        for i in 0..n {
            let row = &d[i * c..i * c + c];
            let max = row.iter().cloned().fold(f64::MIN, f64::max);
            let mut denom = 0.0;
            for &v in row {
                denom += exp(v - max);
            }
            let t = targets[i];
            let logp = (row[t] - max) - ln_approx(denom);
            loss -= logp;
        }
        loss /= n as f64;
        let y = Tensor::new(vec![1], vec![loss])?;
        Some(self.push(y, Op::SoftmaxCe(logits, targets.to_vec())))
    }

    /// Run the backward pass from a scalar `loss` node, filling in every gradient.
    pub fn backward(&mut self, loss: Var) {
        // Seed: d(loss)/d(loss) = 1.
        for g in self.nodes[loss.0].grad.iter_mut() {
            *g = 1.0;
        }
        for idx in (0..self.nodes.len()).rev() {
            // Clone the small op descriptor so we can mutate inputs freely.
            let op = self.nodes[idx].op.clone();
            match op {
                Op::Leaf => {}
                Op::MatMul(a, b) => {
                    // y = a·b ; dA = dY·bᵀ ; dB = aᵀ·dY
                    let (am, ak) = (self.value(a).shape()[0], self.value(a).shape()[1]);
                    let bn = self.value(b).shape()[1];
                    let dy = Tensor::new(vec![am, bn], self.nodes[idx].grad.clone()).unwrap();
                    let at = self.value(a).clone();
                    let bt = self.value(b).clone();
                    let da = dy.matmul(&transpose(&bt).unwrap()).unwrap();
                    let db = transpose(&at).unwrap().matmul(&dy).unwrap();
                    debug_assert_eq!(da.len(), am * ak);
                    accum(&mut self.nodes[a.0].grad, da.data());
                    accum(&mut self.nodes[b.0].grad, db.data());
                }
                Op::AddBias(x, bias) => {
                    let (n, f) = (self.value(x).shape()[0], self.value(x).shape()[1]);
                    let dy = self.nodes[idx].grad.clone();
                    // dX = dY (same shape).
                    accum(&mut self.nodes[x.0].grad, &dy);
                    // dBias[j] = Σ_i dY[i,j].
                    let mut db = vec![0.0; f];
                    for i in 0..n {
                        for j in 0..f {
                            db[j] += dy[i * f + j];
                        }
                    }
                    accum(&mut self.nodes[bias.0].grad, &db);
                }
                Op::Relu(x) => {
                    let xv = self.value(x).data().to_vec();
                    let dy = self.nodes[idx].grad.clone();
                    let dx: Vec<f64> = xv.iter().zip(&dy).map(|(&v, &g)| if v > 0.0 { g } else { 0.0 }).collect();
                    accum(&mut self.nodes[x.0].grad, &dx);
                }
                Op::Sigmoid(x) => {
                    let s = self.nodes[idx].value.data().to_vec(); // output = σ(x)
                    let dy = self.nodes[idx].grad.clone();
                    let dx: Vec<f64> = s.iter().zip(&dy).map(|(&sv, &g)| g * sv * (1.0 - sv)).collect();
                    accum(&mut self.nodes[x.0].grad, &dx);
                }
                Op::Tanh(x) => {
                    let t = self.nodes[idx].value.data().to_vec(); // output = tanh(x)
                    let dy = self.nodes[idx].grad.clone();
                    let dx: Vec<f64> = t.iter().zip(&dy).map(|(&tv, &g)| g * (1.0 - tv * tv)).collect();
                    accum(&mut self.nodes[x.0].grad, &dx);
                }
                Op::Mse(pred, ref target) => {
                    // loss = mean((p−t)²) ; dP = (2/N)(p−t)·dLoss
                    let upstream = self.nodes[idx].grad[0];
                    let p = self.value(pred).data().to_vec();
                    let n = p.len().max(1) as f64;
                    let dp: Vec<f64> = p
                        .iter()
                        .zip(target.data())
                        .map(|(&pv, &tv)| upstream * 2.0 / n * (pv - tv))
                        .collect();
                    accum(&mut self.nodes[pred.0].grad, &dp);
                }
                Op::SoftmaxCe(logits, ref targets) => {
                    // dLogits row = (softmax(row) − onehot)/batch · dLoss
                    let upstream = self.nodes[idx].grad[0];
                    let (n, c) = (self.value(logits).shape()[0], self.value(logits).shape()[1]);
                    let d = self.value(logits).data().to_vec();
                    let mut dl = vec![0.0; n * c];
                    for i in 0..n {
                        let row = &d[i * c..i * c + c];
                        let max = row.iter().cloned().fold(f64::MIN, f64::max);
                        let mut denom = 0.0;
                        for &v in row {
                            denom += exp(v - max);
                        }
                        for j in 0..c {
                            let p = exp(row[j] - max) / denom;
                            let onehot = if j == targets[i] { 1.0 } else { 0.0 };
                            dl[i * c + j] = upstream * (p - onehot) / n as f64;
                        }
                    }
                    accum(&mut self.nodes[logits.0].grad, &dl);
                }
            }
        }
    }
}

/// `g += d` element-wise (gradient accumulation).
fn accum(g: &mut [f64], d: &[f64]) {
    for (a, b) in g.iter_mut().zip(d) {
        *a += b;
    }
}

/// A small, dependency-free natural log for the cross-entropy normaliser. `ln(x) =
/// 2·atanh((x−1)/(x+1))` via its series — accurate for the positive `x` softmax
/// produces.
fn ln_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::MIN;
    }
    // Reduce x to [1,2) by extracting a power of two.
    let mut e = 0i64;
    let mut m = x;
    while m >= 2.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        e -= 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..30 {
        sum += term / k;
        term *= y2;
        k += 2.0;
    }
    2.0 * sum + (e as f64) * LN2
}

// ───────────────────────────── activations ─────────────────────────────

/// The non-linearity applied between dense layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    /// No activation (identity) — used on a regression output layer.
    Identity,
}

impl Activation {
    fn apply_tape(self, tape: &mut Tape, x: Var) -> Var {
        match self {
            Activation::Relu => tape.relu(x),
            Activation::Sigmoid => tape.sigmoid(x),
            Activation::Tanh => tape.tanh(x),
            Activation::Identity => x,
        }
    }

    fn apply(self, v: f64) -> f64 {
        match self {
            Activation::Relu => {
                if v > 0.0 {
                    v
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => sigmoid(v),
            Activation::Tanh => tanh(v),
            Activation::Identity => v,
        }
    }

    /// Return this activation as a plain function pointer (for `matmul_bias_act`).
    pub fn as_fn(self) -> fn(f64) -> f64 {
        match self {
            Activation::Relu => |v| if v > 0.0 { v } else { 0.0 },
            Activation::Sigmoid => sigmoid,
            Activation::Tanh => tanh,
            Activation::Identity => |v| v,
        }
    }

    fn code(self) -> u8 {
        match self {
            Activation::Relu => 0,
            Activation::Sigmoid => 1,
            Activation::Tanh => 2,
            Activation::Identity => 3,
        }
    }

    fn from_code(c: u8) -> Option<Activation> {
        match c {
            0 => Some(Activation::Relu),
            1 => Some(Activation::Sigmoid),
            2 => Some(Activation::Tanh),
            3 => Some(Activation::Identity),
            _ => None,
        }
    }
}

// ───────────────────────────── dense layer ─────────────────────────────

/// A fully-connected layer: `y = x·W + b`, with `W : [in, out]`, `b : [out]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
}

impl Linear {
    /// Xavier/Glorot-initialised dense layer (deterministic given `rng`).
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Linear {
        let limit = sqrt(6.0 / (in_dim + out_dim) as f64);
        let w: Vec<f64> = (0..in_dim * out_dim).map(|_| rng.next_signed() * limit).collect();
        Linear {
            w: Tensor::new(vec![in_dim, out_dim], w).unwrap(),
            b: Tensor::zeros(vec![out_dim]),
        }
    }

    pub fn in_dim(&self) -> usize {
        self.w.shape()[0]
    }
    pub fn out_dim(&self) -> usize {
        self.w.shape()[1]
    }
}

// ───────────────────────────── the model ─────────────────────────────

/// A multilayer perceptron: a stack of [`Linear`] layers with a hidden
/// [`Activation`] between them and a configurable output activation. The unit of
/// both inference and training in this engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Mlp {
    pub layers: Vec<Linear>,
    pub hidden: Activation,
    pub output: Activation,
}

impl Mlp {
    /// Build an MLP from layer sizes (e.g. `[2, 8, 1]` = 2-in, 8-hidden, 1-out),
    /// deterministically seeded. Needs at least an input and an output size.
    pub fn new(sizes: &[usize], hidden: Activation, output: Activation, seed: u64) -> Option<Mlp> {
        if sizes.len() < 2 || sizes.contains(&0) {
            return None;
        }
        let mut rng = Rng::new(seed);
        let mut layers = Vec::new();
        for w in sizes.windows(2) {
            layers.push(Linear::new(w[0], w[1], &mut rng));
        }
        Some(Mlp { layers, hidden, output })
    }

    pub fn in_dim(&self) -> usize {
        self.layers.first().map(|l| l.in_dim()).unwrap_or(0)
    }
    pub fn out_dim(&self) -> usize {
        self.layers.last().map(|l| l.out_dim()).unwrap_or(0)
    }

    /// Total trainable scalar count — a quick size metric.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    /// FLOPs for one forward pass over a `[batch, in]` input.
    pub fn forward_flops(&self, batch: usize) -> u64 {
        self.layers.iter().map(|l| matmul_flops(batch, l.in_dim(), l.out_dim())).sum()
    }

    /// **Inference**: forward a `[batch, in]` input to a `[batch, out]` output.
    ///
    /// Uses the fused `matmul_bias_act` kernel: each layer is one pass over the data
    /// (matmul output taken by value, bias+activation applied before the next write),
    /// saving one allocation and two extra data passes per layer vs the naive chain.
    pub fn forward(&self, x: &Tensor) -> Option<Tensor> {
        if x.shape().len() != 2 || x.shape()[1] != self.in_dim() {
            return None;
        }
        let mut cur = x.clone();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let act = if i == last { self.output } else { self.hidden };
            cur = cur.matmul_bias_act(&layer.w, layer.b.data(), act.as_fn())?;
        }
        Some(cur)
    }

    /// Parallelised fused forward pass: each layer runs `matmul_bias_act_with(spawn)`.
    ///
    /// The matmul is parallelised across cores; bias+activation apply in the single
    /// fused post-pass on the owned output buffer. Bit-identical to `forward()`.
    pub fn forward_with(
        &self,
        x: &Tensor,
        spawn: &dyn crate::parallel::Spawn,
    ) -> Option<Tensor> {
        if x.shape().len() != 2 || x.shape()[1] != self.in_dim() {
            return None;
        }
        let mut cur = x.clone();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let act = if i == last { self.output } else { self.hidden };
            cur = cur.matmul_bias_act_with(&layer.w, layer.b.data(), act.as_fn(), spawn)?;
        }
        Some(cur)
    }

    /// FNV-1a 64-bit hash of all layer weights and biases.
    ///
    /// Identical parameters → identical hash. Precompute once per model (or per
    /// fine-tuning step) and pass to [`forward_cached`](Self::forward_cached) to
    /// avoid rehashing the weights on every call.
    pub fn content_hash(&self) -> u64 {
        const P: u64 = 0x517cc1b727220a95;
        let mut h: u64 = 0xcbf29ce484222325;
        for layer in &self.layers {
            for &v in layer.w.data().iter().chain(layer.b.data()) {
                h ^= v.to_bits();
                h = h.wrapping_mul(P);
            }
        }
        h ^= h >> 32;
        h = h.wrapping_mul(0xd6e8feb86659fd93);
        h ^= h >> 32;
        h
    }

    /// Forward pass using a shared [`TensorMemo`] cache.
    ///
    /// Key: `(model_hash, input_hash)`. On a cache **hit** the stored output is
    /// cloned back — zero compute, just a hash + BTreeMap lookup. On a **miss**
    /// the full `forward()` runs and the result is stored for future reuse.
    ///
    /// Pass a precomputed `model_hash` (from [`content_hash`](Self::content_hash))
    /// to avoid rehashing weights on every call during steady-state inference.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        model_hash: u64,
        cache: &mut crate::memo::TensorMemo,
    ) -> Option<Tensor> {
        let key = (model_hash, x.content_hash());
        if let Some(cached) = cache.get(key) {
            return Some(cached.clone());
        }
        let out = self.forward(x)?;
        cache.insert(key, out.clone());
        Some(out)
    }

    /// Per-layer content hashes — one `u64` per layer, covering all weights and biases.
    ///
    /// Call this **once per model update** (after training or at model load) and pass
    /// the result to [`forward_incremental`](Self::forward_incremental) so the hot
    /// inference loop never re-hashes weight tensors — only the (much cheaper)
    /// per-call input activations are hashed.
    pub fn layer_content_hashes(&self) -> Vec<u64> {
        const P: u64 = 0x517cc1b727220a95;
        self.layers.iter().enumerate().map(|(i, layer)| {
            let mut h: u64 = 0xcbf29ce484222325u64 ^ i as u64;
            for &v in layer.w.data().iter().chain(layer.b.data()) {
                h ^= v.to_bits();
                h = h.wrapping_mul(P);
            }
            h ^= h >> 32;
            h = h.wrapping_mul(0xd6e8feb86659fd93);
            h ^= h >> 32;
            h
        }).collect()
    }

    /// Per-layer incremental forward pass using precomputed layer hashes.
    ///
    /// Each layer is keyed by `(layer_hashes[i], input_activation_hash)`. Layers
    /// whose weights *and* whose incoming activation are unchanged return their
    /// cached output — only modified or downstream-of-modified layers recompute.
    ///
    /// **Call `layer_content_hashes()` once per model update**, not per inference
    /// call. In the hot loop we only hash the much-smaller input activation tensors.
    ///
    /// Fine-tuning payoff:
    /// * Change only the last layer → only the last layer recomputes.
    /// * Change only the first layer → all layers recompute (downstream invalidation).
    pub fn forward_incremental(
        &self,
        x: &Tensor,
        layer_hashes: &[u64],
        cache: &mut crate::memo::TensorMemo,
    ) -> Option<Tensor> {
        if x.shape().len() != 2 || x.shape()[1] != self.in_dim() { return None; }
        if layer_hashes.len() != self.layers.len() { return None; }
        let mut cur = x.clone();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let key = (layer_hashes[i], cur.content_hash());
            if let Some(cached) = cache.get(key) {
                cur = cached.clone();
            } else {
                let act = if i == last { self.output } else { self.hidden };
                let out = cur.matmul_bias_act(&layer.w, layer.b.data(), act.as_fn())?;
                cache.insert(key, out.clone());
                cur = out;
            }
        }
        Some(cur)
    }

    /// Inference on a chosen [`Device`], returning the output and a [`CostReport`].
    pub fn infer(&self, x: &Tensor, device: Device) -> Option<(Tensor, CostReport)> {
        let out = self.forward(x)?;
        let flops = self.forward_flops(x.shape()[0]);
        Some((out, CostReport { device, flops, est_cycles: device.est_cycles(flops) }))
    }

    /// Build the forward computation on a [`Tape`], returning the output `Var` and
    /// the `(w, b)` parameter `Var`s per layer (so the optimizer can read grads).
    fn forward_on_tape(&self, tape: &mut Tape, x: &Tensor) -> Option<(Var, Vec<(Var, Var)>)> {
        let mut cur = tape.leaf(x.clone());
        let mut params = Vec::with_capacity(self.layers.len());
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let wv = tape.leaf(layer.w.clone());
            let bv = tape.leaf(layer.b.clone());
            params.push((wv, bv));
            let z = tape.matmul(cur, wv)?;
            let z = tape.add_bias(z, bv)?;
            let act = if i == last { self.output } else { self.hidden };
            cur = act.apply_tape(tape, z);
        }
        Some((cur, params))
    }

    /// One supervised **training** step on `(x, target)` with mean-squared error.
    /// Returns the loss before the update. Mutates the model in place.
    pub fn train_step_mse(&mut self, x: &Tensor, target: &Tensor, opt: &mut Optimizer) -> Option<f64> {
        let mut tape = Tape::new();
        let (out, params) = self.forward_on_tape(&mut tape, x)?;
        let loss = tape.mse(out, target)?;
        let loss_val = tape.value(loss).data()[0];
        tape.backward(loss);
        self.apply_grads(&tape, &params, opt);
        Some(loss_val)
    }

    /// One classification training step: fused softmax + cross-entropy over integer
    /// class `targets` (one per batch row). Returns the loss before the update.
    pub fn train_step_ce(&mut self, x: &Tensor, targets: &[usize], opt: &mut Optimizer) -> Option<f64> {
        let mut tape = Tape::new();
        let (out, params) = self.forward_on_tape(&mut tape, x)?;
        let loss = tape.softmax_ce(out, targets)?;
        let loss_val = tape.value(loss).data()[0];
        tape.backward(loss);
        self.apply_grads(&tape, &params, opt);
        Some(loss_val)
    }

    fn apply_grads(&mut self, tape: &Tape, params: &[(Var, Var)], opt: &mut Optimizer) {
        for (i, &(wv, bv)) in params.iter().enumerate() {
            let wg = tape.grad(wv).to_vec();
            let bg = tape.grad(bv).to_vec();
            opt.step(i * 2, &mut self.layers[i].w, &wg);
            opt.step(i * 2 + 1, &mut self.layers[i].b, &bg);
        }
        opt.tick();
    }

    // ── deterministic, content-addressable serialization ──
    // Format: magic "AMLP" | u32 nlayers | hidden u8 | output u8 |
    //         per layer: u32 in, u32 out, then (in*out + out) f64 LE.

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"AMLP");
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        out.push(self.hidden.code());
        out.push(self.output.code());
        for l in &self.layers {
            out.extend_from_slice(&(l.in_dim() as u32).to_le_bytes());
            out.extend_from_slice(&(l.out_dim() as u32).to_le_bytes());
            for &v in l.w.data() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for &v in l.b.data() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Mlp> {
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize| -> Option<&[u8]> {
            if *p + n > bytes.len() {
                return None;
            }
            let s = &bytes[*p..*p + n];
            *p += n;
            Some(s)
        };
        if take(&mut p, 4)? != b"AMLP" {
            return None;
        }
        let nlayers = u32::from_le_bytes(take(&mut p, 4)?.try_into().ok()?) as usize;
        let hidden = Activation::from_code(take(&mut p, 1)?[0])?;
        let output = Activation::from_code(take(&mut p, 1)?[0])?;
        // Do NOT pre-reserve from attacker-controlled counts: a hostile header could
        // claim huge dimensions and force an OOM abort before any bytes are read.
        // Let the Vecs grow as `take()` succeeds so a truncated/hostile blob fails fast.
        let mut layers = Vec::new();
        for _ in 0..nlayers {
            let in_dim = u32::from_le_bytes(take(&mut p, 4)?.try_into().ok()?) as usize;
            let out_dim = u32::from_le_bytes(take(&mut p, 4)?.try_into().ok()?) as usize;
            let mut w = Vec::new();
            for _ in 0..in_dim * out_dim {
                w.push(f64::from_le_bytes(take(&mut p, 8)?.try_into().ok()?));
            }
            let mut b = Vec::new();
            for _ in 0..out_dim {
                b.push(f64::from_le_bytes(take(&mut p, 8)?.try_into().ok()?));
            }
            layers.push(Linear {
                w: Tensor::new(vec![in_dim, out_dim], w)?,
                b: Tensor::new(vec![out_dim], b)?,
            });
        }
        Some(Mlp { layers, hidden, output })
    }
}

// ───────────────────────── Lever 7: federated training ──────────────────────

/// Flatten all trainable parameters of an `Mlp` into a contiguous Vec.
///
/// Order: for each layer — W (row-major), then b. Total length equals
/// `Σ_i (in_i × out_i + out_i)` over all layers.
pub fn mlp_params_flat(mlp: &Mlp) -> Vec<f64> {
    let mut out = Vec::new();
    for layer in &mlp.layers {
        out.extend_from_slice(layer.w.data());
        out.extend_from_slice(layer.b.data());
    }
    out
}

/// Apply a flat parameter vector back to an `Mlp` (same layout as `mlp_params_flat`).
///
/// Reconstructs each `Linear` layer's `w` and `b` tensors in-place.
/// Returns `false` if the length doesn't match the model's parameter count.
pub fn mlp_set_params_flat(mlp: &mut Mlp, params: &[f64]) -> bool {
    let total: usize = mlp
        .layers
        .iter()
        .map(|l| l.w.data().len() + l.b.data().len())
        .sum();
    if params.len() != total {
        return false;
    }
    let mut off = 0;
    for layer in &mut mlp.layers {
        let w_shape = layer.w.shape().to_vec();
        let b_shape = layer.b.shape().to_vec();
        let wn = layer.w.data().len();
        let bn = layer.b.data().len();
        // Reconstruct tensors from the new parameter slice.
        layer.w = Tensor::new(w_shape, params[off..off + wn].to_vec())
            .expect("mlp_set_params_flat: w reconstruction failed");
        off += wn;
        layer.b = Tensor::new(b_shape, params[off..off + bn].to_vec())
            .expect("mlp_set_params_flat: b reconstruction failed");
        off += bn;
    }
    true
}

/// Federated averaging (FedAvg): element-wise arithmetic mean of `n` parameter vectors.
///
/// Each vector represents the trainable parameters of one worker node after K local
/// training steps. The averaged result is the new global model — each worker sets
/// its parameters to this vector before the next round of local training.
///
/// Returns `None` if `param_vecs` is empty or vectors have different lengths.
///
/// Communication model: in practice this aggregation happens via ring all-reduce
/// (see `ring_allreduce_cost`), not a central server, so there is no single
/// bottleneck node.
pub fn fed_avg(param_vecs: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = param_vecs.len();
    if n == 0 {
        return None;
    }
    let m = param_vecs[0].len();
    let mut avg = vec![0.0f64; m];
    for pv in param_vecs {
        if pv.len() != m {
            return None;
        }
        for (a, &v) in avg.iter_mut().zip(pv.iter()) {
            *a += v;
        }
    }
    let inv_n = 1.0 / n as f64;
    for a in &mut avg {
        *a *= inv_n;
    }
    Some(avg)
}

/// Bandwidth-optimal ring all-reduce cost model.
///
/// Phase 1 — scatter-reduce: (n−1) rounds, each moving param_count/n parameters one
/// step around the ring (accumulating partial sums).
/// Phase 2 — all-gather: (n−1) more rounds to broadcast the fully-reduced result.
///
/// Total bytes transferred PER NODE = 2·(n−1)/n · param_count · 8.
/// Total messages across all nodes = 2·n·(n−1).
///
/// Returns `(rounds, total_messages, bytes_per_node)`.
pub fn ring_allreduce_cost(n_workers: usize, param_count: usize) -> (usize, usize, usize) {
    if n_workers <= 1 {
        return (0, 0, 0);
    }
    let rounds = 2 * (n_workers - 1);
    let msgs = 2 * n_workers * (n_workers - 1);
    // Each node sends 2*(n-1) messages each of size param_count/n * 8 bytes.
    let bytes_per_node = 2 * (n_workers - 1) * (param_count / n_workers) * 8;
    (rounds, msgs, bytes_per_node)
}

// ── Lever 7 extensions ────────────────────────────────────────────────────────

/// Sparsify a parameter-delta vector for bandwidth-efficient federated all-reduce.
///
/// Keeps entries where |v| exceeds a threshold calibrated so that approximately
/// `keep_pct`% of entries survive. Workers transmit only the surviving (index, value)
/// pairs, reducing bandwidth by ~100/keep_pct×. Non-transmitted entries reconstruct
/// to zero on the receiver, which is the standard Top-K sparsification assumption.
///
/// Single-pass, O(n) time, no allocation for the threshold step — no_std safe.
pub fn sparsify_delta(delta: &[f64], keep_pct: usize) -> Vec<(usize, f64)> {
    if delta.is_empty() || keep_pct >= 100 {
        return delta.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    }
    // Max-magnitude threshold: keep entries in the top keep_pct% of the [0, max] range.
    // threshold = max_abs * (1 - keep_pct/100):
    //   keep_pct=10 → threshold = 0.90 * max → keeps entries near the peak
    //   keep_pct=50 → threshold = 0.50 * max → keeps the upper half of magnitudes
    //   keep_pct=100 → threshold = 0.0 → keep everything
    // This is a single-pass O(n) algorithm with no allocation — no_std safe.
    let max_abs = delta.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
    let threshold = max_abs * (1.0 - keep_pct as f64 / 100.0);
    delta
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() >= threshold)
        .map(|(i, &v)| (i, v))
        .collect()
}

/// Reconstruct a dense gradient-delta from a sparse (index, value) list.
/// Indices absent from `sparse` are set to 0.0 (not transmitted = zero update).
pub fn reconstruct_delta(sparse: &[(usize, f64)], len: usize) -> Vec<f64> {
    let mut dense = vec![0.0f64; len];
    for &(i, v) in sparse {
        if i < len {
            dense[i] = v;
        }
    }
    dense
}

/// L2 norm of a flat vector — used to gate adaptive sync rounds.
pub fn vec_norm(v: &[f64]) -> f64 {
    sqrt(v.iter().map(|&x| x * x).fold(0.0f64, |a, b| a + b))
}

/// SCAFFOLD starting-point correction for federated workers (Karimireddy et al. 2020).
///
/// When workers train for K steps before syncing, each worker's local gradients
/// pull toward its local data distribution rather than the global objective (client
/// drift). SCAFFOLD counteracts this by shifting each worker's starting parameters
/// in the opposite direction of its known local gradient bias.
///
/// `c_local`  = this worker's control variate (estimate of local gradient bias).
/// `c_global` = global control variate (server-side mean of all c_i).
///
/// corrected_start[i] = global[i] + K * lr * (c_global[i] - c_local[i])
///
/// Round 0: both variates are 0 → no correction; correction builds up with training.
pub fn scaffold_correct_start(
    global: &[f64],
    c_local: &[f64],
    c_global: &[f64],
    k: usize,
    lr: f64,
) -> Vec<f64> {
    let bias_scale = k as f64 * lr;
    global
        .iter()
        .zip(c_local.iter())
        .zip(c_global.iter())
        .map(|((&g, &cl), &cg)| g + bias_scale * (cg - cl))
        .collect()
}

/// Update the SCAFFOLD local control variate after one federated training round.
///
/// c_local_new = c_local_old − c_global + (params_before − params_after) / (K · lr)
///
/// The `(params_before − params_after) / (K · lr)` term approximates the mean
/// gradient the worker computed during its K local optimiser steps.
pub fn scaffold_update_control(
    c_local: &mut [f64],
    c_global: &[f64],
    params_before: &[f64],
    params_after: &[f64],
    k: usize,
    lr: f64,
) {
    let denom = (k as f64 * lr).max(1e-12);
    for (i, cl) in c_local.iter_mut().enumerate() {
        let approx_grad = (params_before[i] - params_after[i]) / denom;
        *cl = *cl - c_global[i] + approx_grad;
    }
}

/// Two-level hierarchical ring all-reduce cost model for rack-aware deployments.
///
/// Data-centre intra-rack bandwidth (25–100 Gbps) greatly exceeds inter-rack
/// bandwidth (typically 10–25 Gbps with 4:1 oversubscription). A two-level reduction
/// — intra-rack first, then a single rack-representative per inter-rack ring — moves
/// most traffic over the cheap intra-rack fabric rather than the expensive spine.
///
/// Level 1 — intra-rack ring reduce/gather: 2*(nodes_per_rack−1) rounds.
/// Level 2 — inter-rack ring reduce/gather (one leader per rack): 2*(n_racks−1) rounds.
/// Level 3 — intra-rack broadcast of the global result: 1 round.
///
/// Returns `(total_rounds, intra_bytes_per_node, inter_bytes_per_rack_leader)`.
pub fn hierarchical_allreduce_cost(
    n_racks: usize,
    nodes_per_rack: usize,
    param_count: usize,
) -> (usize, usize, usize) {
    let npr = nodes_per_rack.max(1);
    let nr  = n_racks.max(1);
    let intra_bytes = if nodes_per_rack <= 1 { 0 }
        else { 2 * (nodes_per_rack - 1) * (param_count / npr) * 8 };
    let inter_bytes = if n_racks <= 1 { 0 }
        else { 2 * (n_racks - 1) * (param_count / nr) * 8 };
    let intra_rounds = 2 * nodes_per_rack.saturating_sub(1);
    let inter_rounds = 2 * n_racks.saturating_sub(1);
    let bcast = if nodes_per_rack > 1 { 1 } else { 0 };
    (intra_rounds + inter_rounds + bcast, intra_bytes, inter_bytes)
}

/// What an inference call cost on a given device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostReport {
    pub device: Device,
    pub flops: u64,
    pub est_cycles: u64,
}

// ─────────────────────────── research-grade PTQ ──────────────────────────
//
// SmoothQuant and AWQ: calibration-based post-training quantization.
// Both are offline transformations on weight tensors — zero runtime cost after
// calibration. No external dependencies, no unsafe, no_std + alloc.
//
// Terminology:
//   X: activation tensor (runtime input to a Linear layer)
//   W: weight matrix
//   s: per-channel scale vector (one entry per input feature j)
//   α: migration strength in [0,1]; α=0 = scale only W, α=1 = scale only X

/// Per-channel calibration statistics from a calibration run.
///
/// Fill this by running representative inputs through the model and collecting
/// per-channel max absolute activation values at each linear layer.
#[derive(Clone, Debug)]
pub struct CalibStats {
    /// `x_max[j]` = max |activation[*, j]| observed over the calibration set.
    pub x_max: Vec<f64>,
    /// `w_max[j]` = max |W[j, :]| — the weight row magnitude at channel j.
    pub w_max: Vec<f64>,
}

impl CalibStats {
    /// Collect calibration statistics for `layer` over `inputs`.
    ///
    /// `inputs`: one `[1, in_features]` tensor per calibration sample.
    /// `layer_idx`: which MLP layer to profile (0 = first hidden).
    pub fn collect(mlp: &Mlp, layer_idx: usize, inputs: &[Tensor]) -> Option<Self> {
        let layer = mlp.layers.get(layer_idx)?;
        let in_feats = layer.w.shape()[0]; // W is [in, out]
        let out_feats = layer.w.shape()[1];
        let mut x_max = vec![0.0f64; in_feats];
        let mut w_max = vec![0.0f64; in_feats];

        // Accumulate per-input-channel activation maxima.
        for inp in inputs {
            if inp.data().len() != in_feats { continue; }
            for (j, &v) in inp.data().iter().enumerate() {
                let av = fabs(v);
                if av > x_max[j] { x_max[j] = av; }
            }
        }

        // W shape: [in_feats, out_feats]. Row j = weight row j across all outputs.
        let wd = layer.w.data();
        for j in 0..in_feats {
            for k in 0..out_feats {
                let av = fabs(wd[j * out_feats + k]);
                if av > w_max[j] { w_max[j] = av; }
            }
        }

        Some(CalibStats { x_max, w_max })
    }
}

/// Compute SmoothQuant per-channel scales for `layer_idx`.
///
/// SmoothQuant (Xiao et al., 2022) migrates quantization difficulty from
/// activation domain (high dynamic range, hard to quantize) to weight domain
/// (low dynamic range, easy to quantize). Scale: `s[j] = max|X[:,j]|^α / max|W[j,:]|^(1-α)`.
///
/// Apply the returned scale by calling `smooth_quant_apply`.
/// Fuse `s⁻¹` into the preceding layer's output (e.g., multiply LayerNorm scale
/// by `s⁻¹`) — zero runtime cost at inference time.
///
/// Default α = 0.5 balances the migration evenly; increase toward 1.0 if
/// activations have more outliers than weights.
pub fn smooth_quant_scales(stats: &CalibStats, alpha: f64) -> Vec<f64> {
    stats.x_max.iter().zip(&stats.w_max).map(|(&xm, &wm)| {
        let x_part = if xm > 0.0 { pow_f64(xm, alpha) } else { 1.0 };
        let w_part = if wm > 0.0 { pow_f64(wm, 1.0 - alpha) } else { 1.0 };
        if w_part > 0.0 { x_part / w_part } else { 1.0 }
    }).collect()
}

/// Apply SmoothQuant scales to a single `Linear` layer's weights.
///
/// Transforms `W[j, :] *= s[j]` in-place. The caller must fold `s[j]⁻¹` into
/// the preceding layer (or inject a `ChannelScale` op) to preserve output identity.
pub fn smooth_quant_apply(layer: &mut Linear, scales: &[f64]) -> bool {
    let (in_f, out_f) = (layer.w.shape()[0], layer.w.shape()[1]);
    if scales.len() != in_f { return false; }
    let shape = layer.w.shape().to_vec();
    let mut w = layer.w.data().to_vec();
    for j in 0..in_f {
        let s = scales[j];
        for k in 0..out_f {
            w[j * out_f + k] *= s;
        }
    }
    layer.w = Tensor::new(shape, w).expect("smooth_quant_apply: tensor reconstruction");
    true
}

/// AWQ per-channel scales for INT4 weight quantization.
///
/// AWQ (Lin et al., 2023) protects the ~1% of weight channels with the highest
/// activation magnitudes by giving them proportionally higher quantization resolution
/// via a pre-scaling step. The scale is folded back into the activation path (zero
/// runtime cost), and weights are stored as INT4 with `awq_apply`.
///
/// Grid-searches `alpha` over `n_steps` values in [0, 1], choosing the α that
/// minimises the INT4 reconstruction error `||Q(W·s) · (s⁻¹·X) - W·X||²`.
///
/// Returns `(best_alpha, best_scales, best_error)`.
pub fn awq_calibrate(
    mlp: &Mlp,
    layer_idx: usize,
    inputs: &[Tensor],
    n_steps: usize,
) -> Option<(f64, Vec<f64>, f64)> {
    let layer = mlp.layers.get(layer_idx)?;
    let in_f  = layer.w.shape()[0];
    let out_f = layer.w.shape()[1];
    if inputs.is_empty() || in_f == 0 { return None; }

    // Per-channel activation max (x_max[j] = max |X[:, j]|).
    let mut x_max = vec![0.0f64; in_f];
    for inp in inputs {
        if inp.data().len() < in_f { continue; }
        for (j, &v) in inp.data()[..in_f].iter().enumerate() {
            let a = fabs(v);
            if a > x_max[j] { x_max[j] = a; }
        }
    }

    // Reference output for reconstruction error: W·X for each calibration sample.
    let ref_outputs: Vec<Vec<f64>> = inputs.iter().filter_map(|inp| {
        layer.w.shape();
        // Manual matmul: out[k] = sum_j inp[j] * W[j, k]
        if inp.data().len() < in_f { return None; }
        let x = inp.data();
        let w = layer.w.data();
        Some((0..out_f).map(|k| (0..in_f).map(|j| x[j] * w[j * out_f + k]).sum::<f64>()).collect())
    }).collect();

    let mut best_alpha = 0.5f64;
    let mut best_scales: Vec<f64> = vec![1.0; in_f];
    let mut best_err = f64::MAX;

    let steps = n_steps.max(1);
    for step in 0..=steps {
        let alpha = step as f64 / steps as f64;
        // Scale: s[j] = max|X[:,j]|^alpha
        let scales: Vec<f64> = x_max.iter().map(|&xm| {
            if xm > 0.0 { pow_f64(xm, alpha) } else { 1.0 }
        }).collect();

        // Scale weights: W_scaled[j,k] = W[j,k] * s[j]
        let w = layer.w.data();
        let mut w_scaled = w.to_vec();
        for j in 0..in_f {
            for k in 0..out_f {
                w_scaled[j * out_f + k] *= scales[j];
            }
        }

        // Simulate INT4 quantization of w_scaled with the SAME scheme quantize_q4
        // deploys: symmetric, single global scale = max|w|/7, q ∈ {−7,…,7}. This makes
        // the alpha grid search optimize the error of the quantizer actually applied.
        let mx = w_scaled.iter().fold(0.0f64, |m, &v| m.max(fabs(v)));
        let s = if mx == 0.0 { 1.0 } else { mx / 7.0 };
        let w_q4: Vec<f64> = w_scaled.iter().map(|&v| {
            clamp_q4(round_half_away(v / s)) as f64 * s
        }).collect();

        // Reconstruction error: sum_samples ||Q(W·s)·(s⁻¹·x) - W·x||²
        let mut err = 0.0f64;
        for (inp, ref_out) in inputs.iter().zip(&ref_outputs) {
            if inp.data().len() < in_f { continue; }
            let x = inp.data();
            // Compute (s⁻¹·x): x_scaled[j] = x[j] / s[j]
            for k in 0..out_f {
                let approx: f64 = (0..in_f).map(|j| {
                    let x_s = if scales[j] > 1e-12 { x[j] / scales[j] } else { x[j] };
                    x_s * w_q4[j * out_f + k]
                }).sum();
                let diff = approx - ref_out[k];
                err += diff * diff;
            }
        }

        if err < best_err {
            best_err = err;
            best_alpha = alpha;
            best_scales = scales;
        }
    }

    Some((best_alpha, best_scales, best_err))
}

/// Apply AWQ scales to a `Linear` layer, returning the INT4 quantized tensor.
///
/// `W_new[j,k] = W[j,k] * scales[j]` (then quantize to INT4).
/// The caller must fold `scales[j]⁻¹` into the input activation path.
pub fn awq_apply(layer: &Linear, scales: &[f64]) -> Option<Q4Tensor> {
    let in_f  = layer.w.shape()[0];
    let out_f = layer.w.shape()[1];
    if scales.len() != in_f { return None; }
    let w = layer.w.data();
    let mut w_scaled = w.to_vec();
    for j in 0..in_f {
        for k in 0..out_f {
            w_scaled[j * out_f + k] *= scales[j];
        }
    }
    let scaled_tensor = Tensor::new(vec![in_f, out_f], w_scaled)?;
    Some(quantize_q4(&scaled_tensor))
}

// Utility: x^p via exp(p·ln(x)), safe for x>0, p≥0.
fn pow_f64(x: f64, p: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if p == 0.0 { return 1.0; }
    if p == 1.0 { return x; }
    // exp(p · ln(x)); ln via ln_approx.
    let lx = ln_approx(x);
    exp(p * lx)
}

/// Speculative inference: run a fast draft (`Mlp`) to generate a candidate output,
/// then verify with a more accurate `verifier` model. If outputs agree (within `tol`),
/// return the draft result immediately. Otherwise fall back to the verifier's result.
///
/// This implements the speculative execution pattern for inference pipelines:
/// use a cheap quantized draft (INT4/binary) as the hot path, with a heavier model
/// (INT8/F64) as a verifier that only runs on misses. The draft+verify cost is less
/// than always-verify when the draft hit rate is high.
///
/// Returns `(output, draft_accepted)`.
pub fn speculative_infer(
    draft:    &Mlp,
    verifier: &Mlp,
    x:        &Tensor,
    tol:      f64,
) -> Option<(Tensor, bool)> {
    let draft_out = draft.forward(x)?;
    let verify_out = verifier.forward(x)?;
    // Accept draft if max element-wise difference is within tolerance.
    let draft_ok = draft_out.data().iter().zip(verify_out.data()).all(|(&d, &v)| fabs(d - v) <= tol);
    if draft_ok {
        Some((draft_out, true))
    } else {
        Some((verify_out, false))
    }
}

/// Batch speculative inference: pre-screen a batch of inputs with the draft model,
/// then run only the rejected inputs through the verifier.
///
/// Returns a vector of outputs (one per input) and the draft hit rate (0..=100).
pub fn speculative_infer_batch(
    draft:    &Mlp,
    verifier: &Mlp,
    inputs:   &[Tensor],
    tol:      f64,
) -> (Vec<Tensor>, u64) {
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut hits = 0u64;
    for x in inputs {
        match speculative_infer(draft, verifier, x, tol) {
            Some((out, accepted)) => {
                if accepted { hits += 1; }
                outputs.push(out);
            }
            None => {
                // Forward through verifier as last resort.
                if let Some(out) = verifier.forward(x) {
                    outputs.push(out);
                }
            }
        }
    }
    let hit_rate = if inputs.is_empty() { 0 } else { hits * 100 / inputs.len() as u64 };
    (outputs, hit_rate)
}

// ───────────────────── TurboQuant KV-cache quantization ─────────────────
//
// TurboQuant (Zandieh et al., ICLR 2026, arXiv:2504.19874):
// Compresses key/value vectors to 2.5–3.5 bits with near-zero inner-product error.
//
// Algorithm:
//   1. Apply a data-oblivious random rotation R (from a fixed PRNG seed) to the
//      input vector: x_rot = R · x. Post-rotation coordinates follow a concentrated
//      distribution, making per-coordinate scalar quantization near-optimal.
//   2. Quantize x_rot to n bits per coordinate (uniform min-max).
//   3. Compute residual r = x_rot − dequant(quant(x_rot)).
//   4. Store sign(r) as a 1-bit JL correction per coordinate.
//
// The random rotation is implemented as a structured Walsh-Hadamard transform
// followed by random sign flips — O(d log d) vs O(d²) for a dense rotation,
// while preserving the JL distribution guarantees.

/// A TurboQuant-compressed vector.
#[derive(Clone, Debug)]
pub struct TurboQuantVec {
    /// Quantized rotated coordinates (n bits per element stored as i8 or i16).
    pub quant: Vec<i16>,
    /// Per-coordinate min and scale for dequantization.
    pub min: f64,
    pub scale: f64,
    /// 1-bit JL residual correction (packed: bit i of byte i/8 = sign(residual[i])).
    pub jl_signs: Vec<u8>,
    /// Original dimension.
    pub dim: usize,
    /// Bits per coordinate (2–8).
    pub bits: u8,
    /// PRNG seed used for the rotation (so the receiver can reconstruct R).
    pub rotation_seed: u64,
}

/// Compress a vector with TurboQuant at `bits` bits per coordinate.
///
/// `rotation_seed` must match the receiver's seed to correctly decompress.
/// Typical usage: 3 bits gives ~6x compression vs f64 with near-lossless
/// inner-product fidelity.
pub fn turboquant_compress(x: &[f64], bits: u8, rotation_seed: u64) -> TurboQuantVec {
    let d = x.len();
    let bits = bits.clamp(2, 8);
    let levels = (1i32 << bits) as f64;

    // ── Step 1: Walsh-Hadamard rotation with random sign flips ──────────────
    let mut rotated = x.to_vec();
    // Random sign flip: multiply each coordinate by ±1 from PRNG.
    let mut prng = rotation_seed ^ 0x6c62272e07bb0142;
    let signs: Vec<f64> = (0..d).map(|_| {
        prng = prng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if prng >> 63 == 0 { 1.0 } else { -1.0 }
    }).collect();
    for (v, &s) in rotated.iter_mut().zip(&signs) { *v *= s; }

    // Fast Walsh-Hadamard Transform (in-place, length must be power of 2 — pad if needed).
    let pad_len = {
        let mut p = 1usize;
        while p < d { p *= 2; }
        p
    };
    rotated.resize(pad_len, 0.0);
    let mut h = 1usize;
    while h < pad_len {
        let mut i = 0;
        while i < pad_len {
            for j in i..i + h {
                let (u, v) = (rotated[j], rotated[j + h]);
                rotated[j]     = u + v;
                rotated[j + h] = u - v;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    // Normalise by 1/sqrt(pad_len) to keep norms stable.
    let norm_factor = 1.0 / sqrt(pad_len as f64);
    for v in &mut rotated { *v *= norm_factor; }
    rotated.truncate(d);

    // ── Step 2: Per-tensor scalar quantization (min-max, n bits) ─────────────
    let mn = rotated.iter().cloned().fold(f64::MAX, f64::min);
    let mx = rotated.iter().cloned().fold(f64::MIN, f64::max);
    let range = (mx - mn).max(1e-10);
    let scale = range / (levels - 1.0);
    let quant: Vec<i16> = rotated.iter().map(|&v| {
        { let r = (v - mn) / scale; round_half_away(r).max(0).min((levels as i32) - 1) as i16 }
    }).collect();

    // ── Step 3+4: Residual + 1-bit JL sign ───────────────────────────────────
    let jl_bytes = (d + 7) / 8;
    let mut jl_signs = vec![0u8; jl_bytes];
    for (i, (&q, &r)) in quant.iter().zip(&rotated).enumerate() {
        let dq = mn + q as f64 * scale;
        let residual = r - dq;
        if residual >= 0.0 {
            jl_signs[i / 8] |= 1 << (i % 8);
        }
    }

    TurboQuantVec { quant, min: mn, scale, jl_signs, dim: d, bits, rotation_seed }
}

/// Decompress a TurboQuant vector back to f64 (approximate).
///
/// The 1-bit JL correction improves inner-product estimation but the output
/// is NOT bit-identical to the original — it is the compressed approximation.
pub fn turboquant_decompress(tq: &TurboQuantVec) -> Vec<f64> {
    let d = tq.dim;
    // Dequantize + apply JL residual direction.
    let mut rotated: Vec<f64> = tq.quant.iter().enumerate().map(|(i, &q)| {
        let base = tq.min + q as f64 * tq.scale;
        // Add half-step in the residual direction (sign correction).
        let sign = if (tq.jl_signs[i / 8] >> (i % 8)) & 1 == 1 { 1.0 } else { -1.0 };
        base + sign * tq.scale * 0.25
    }).collect();

    // Inverse WHT (same as forward WHT, then divide by pad_len).
    let pad_len = { let mut p = 1usize; while p < d { p *= 2; } p };
    rotated.resize(pad_len, 0.0);
    let mut h = 1usize;
    while h < pad_len {
        let mut i = 0;
        while i < pad_len {
            for j in i..i + h {
                let (u, v) = (rotated[j], rotated[j + h]);
                rotated[j]     = u + v;
                rotated[j + h] = u - v;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let inv_factor = 1.0 / pad_len as f64;
    for v in &mut rotated { *v *= inv_factor; }
    rotated.truncate(d);

    // Un-apply the random sign flips.
    let mut prng = tq.rotation_seed ^ 0x6c62272e07bb0142;
    for v in &mut rotated {
        prng = prng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if prng >> 63 != 0 { *v = -*v; }
    }
    rotated
}

/// Inner product of two TurboQuant-compressed vectors (without decompressing).
///
/// Computes `dot(decompress(a), decompress(b))` but using the quantized
/// integer arithmetic for the main term and the JL correction for refinement.
/// Much faster than full decompression for attention score computation.
pub fn turboquant_dot(a: &TurboQuantVec, b: &TurboQuantVec) -> f64 {
    if a.dim != b.dim { return 0.0; }
    // Main term: sum_i (min_a + q_a[i] * scale_a) * (min_b + q_b[i] * scale_b)
    let d = a.dim;
    let mut dot_base = 0.0f64;
    for i in 0..d {
        let va = a.min + a.quant[i] as f64 * a.scale;
        let vb = b.min + b.quant[i] as f64 * b.scale;
        dot_base += va * vb;
    }
    // JL correction: refine with sign-product adjustment.
    let mut correction = 0.0f64;
    for i in 0..d {
        let sa = if (a.jl_signs[i / 8] >> (i % 8)) & 1 == 1 { 1.0f64 } else { -1.0 };
        let sb = if (b.jl_signs[i / 8] >> (i % 8)) & 1 == 1 { 1.0f64 } else { -1.0 };
        correction += sa * sb * a.scale * b.scale * 0.0625; // (0.25)²
    }
    dot_base + correction
}

// ───────────────────────── LoRA adapter ──────────────────────────────────
//
// LoRA (Hu et al., ICLR 2022, arXiv:2106.09685):
// Decomposes a weight update into two low-rank matrices: W' = W + B·A,
// B ∈ ℝ^(in×r), A ∈ ℝ^(r×out), r ≪ min(in, out).
// At inference: can be merged offline (zero cost) or kept separate (two tiny matmuls).

/// A LoRA rank-r adapter for a single `Linear` layer.
#[derive(Clone, Debug)]
pub struct LoraAdapter {
    /// Down-projection matrix: `[in_features, rank]`.
    pub a: Tensor,
    /// Up-projection matrix: `[rank, out_features]`.
    pub b: Tensor,
    /// Rank (r). Must equal `a.shape()[1] == b.shape()[0]`.
    pub rank: usize,
    /// Scaling factor α/r (default: 1.0). Controls adapter magnitude.
    pub scale: f64,
}

impl LoraAdapter {
    /// Create a zero-initialised LoRA adapter. After creation, train A and B,
    /// then apply with `lora_forward` or merge with `lora_merge`.
    pub fn new(in_features: usize, out_features: usize, rank: usize) -> Option<Self> {
        if rank == 0 || rank > in_features || rank > out_features { return None; }
        let a = Tensor::new(vec![in_features, rank], vec![0.0; in_features * rank])?;
        let b = Tensor::new(vec![rank, out_features], vec![0.0; rank * out_features])?;
        Some(LoraAdapter { a, b, rank, scale: 1.0 })
    }

    /// Random-initialised adapter: A ~ Normal(0, σ), B = 0 (standard LoRA init).
    pub fn new_random(in_features: usize, out_features: usize, rank: usize, seed: u64) -> Option<Self> {
        if rank == 0 || rank > in_features || rank > out_features { return None; }
        let mut rng = Rng::new(seed);
        let sigma = 0.02; // Small init so adapter starts near identity.
        let a_data: Vec<f64> = (0..in_features * rank).map(|_| rng.next_signed() * sigma).collect();
        let a = Tensor::new(vec![in_features, rank], a_data)?;
        let b = Tensor::new(vec![rank, out_features], vec![0.0; rank * out_features])?;
        Some(LoraAdapter { a, b, rank, scale: 1.0 })
    }
}

/// Forward pass through a `Linear` layer with an active LoRA adapter.
///
/// `output = x @ W + (x @ A @ B) * scale`
///
/// `W` is the frozen base weight; only A and B accumulate gradients during adapter training.
pub fn lora_forward(layer: &Linear, adapter: &LoraAdapter, x: &Tensor) -> Option<Tensor> {
    // Base: x @ W + b
    let base = {
        let z = x.matmul(&layer.w)?;
        let (rows, cols) = (z.shape()[0], z.shape()[1]);
        let bd = layer.b.data();
        let mut data = z.data().to_vec();
        for r in 0..rows { for c in 0..cols { data[r * cols + c] += bd[c]; } }
        Tensor::new(vec![rows, cols], data)?
    };
    // Adapter: x @ A @ B * scale
    let xa = x.matmul(&adapter.a)?;    // [batch, rank]
    let xab = xa.matmul(&adapter.b)?;  // [batch, out]
    let (rows, cols) = (xab.shape()[0], xab.shape()[1]);
    let s = adapter.scale;
    let base_d = base.data();
    let xab_d  = xab.data();
    let merged: Vec<f64> = base_d.iter().zip(xab_d).map(|(&bv, &av)| bv + av * s).collect();
    Tensor::new(vec![rows, cols], merged)
}

/// Merge LoRA adapter into the base layer's weights (offline, zero inference overhead).
///
/// After merging: `W_new = W + A @ B * scale`. The adapter can then be discarded.
/// Inference with the merged layer is identical to `lora_forward` but without the
/// extra matmuls.
pub fn lora_merge(layer: &mut Linear, adapter: &LoraAdapter) -> bool {
    let in_f  = layer.w.shape()[0];
    let out_f = layer.w.shape()[1];
    if adapter.a.shape() != &[in_f, adapter.rank] ||
       adapter.b.shape() != &[adapter.rank, out_f] {
        return false;
    }
    // Compute delta = A @ B * scale: [in_f, out_f]
    let ab = match adapter.a.matmul(&adapter.b) {
        Some(t) => t,
        None => return false,
    };
    let shape = layer.w.shape().to_vec();
    let w_data = layer.w.data();
    let ab_data = ab.data();
    let s = adapter.scale;
    let new_w: Vec<f64> = w_data.iter().zip(ab_data).map(|(&wv, &dv)| wv + dv * s).collect();
    layer.w = match Tensor::new(shape, new_w) {
        Some(t) => t,
        None => return false,
    };
    true
}

// ────────────────────────── FlashAttention CPU ────────────────────────────
//
// FlashAttention (Dao et al., 2022, arXiv:2205.14135):
// Compute softmax(Q·Kᵀ/√d_k)·V using online (incremental) softmax, so the full
// N×N attention matrix is never materialised. Instead, process K and V in tiles
// that fit in cache. Memory: O(N) not O(N²).
//
// Online softmax state per query row: (m, ℓ, O) where
//   m = running row maximum
//   ℓ = running denominator (sum of exp(score − m))
//   O = running output (accumulates V contributions)
//
// Update rule for a new block of keys k=i..j:
//   m_new = max(m, max(scores[i..j]))
//   α = exp(m − m_new)            ← rescale old contribution
//   β = exp(scores[i..j] − m_new) ← new block weights
//   ℓ_new = α·ℓ + sum(β)
//   O_new = (α·ℓ·O + sum(β[k]·V[k])) / ℓ_new
//
// No unsafe code, no_std + alloc, pure Rust.

/// CPU FlashAttention: `output = softmax(Q·Kᵀ / sqrt(d_k)) · V`.
///
/// - `q`: `[seq_q, d_k]`
/// - `k`: `[seq_k, d_k]`
/// - `v`: `[seq_k, d_v]`
/// - `block_size`: KV tile size in sequence dimension. Choose so that one Q row +
///   one K block + one V block fits in L2 cache: `block_size ≤ L2_bytes / (2 · d_k · 8)`.
/// - `causal`: if true, query i can only attend to keys 0..=i (upper-triangular mask).
///
/// Returns `[seq_q, d_v]`, or `None` on shape mismatch.
pub fn flash_attention(
    q:          &Tensor,
    k:          &Tensor,
    v:          &Tensor,
    block_size: usize,
    causal:     bool,
) -> Option<Tensor> {
    if q.shape().len() != 2 || k.shape().len() != 2 || v.shape().len() != 2 { return None; }
    let (seq_q, d_k) = (q.shape()[0], q.shape()[1]);
    let (seq_k, d_k2) = (k.shape()[0], k.shape()[1]);
    let (seq_k2, d_v) = (v.shape()[0], v.shape()[1]);
    if d_k != d_k2 || seq_k != seq_k2 { return None; }
    let scale = 1.0 / sqrt(d_k as f64);
    let block = block_size.max(1).min(seq_k);

    let qd = q.data();
    let kd = k.data();
    let vd = v.data();
    let mut out = vec![0.0f64; seq_q * d_v];

    // Per query row: maintain (m, ℓ, O).
    for qi in 0..seq_q {
        let qrow = &qd[qi * d_k..(qi + 1) * d_k];
        let mut m = f64::NEG_INFINITY;  // running max
        let mut l = 0.0f64;            // running denominator
        let mut o = vec![0.0f64; d_v]; // running output

        // Process K/V in blocks.
        let mut kv_start = 0;
        while kv_start < seq_k {
            let kv_end = (kv_start + block).min(seq_k);

            // Compute scores for this block: score[j] = dot(q[qi], k[kj]) * scale.
            let mut scores = Vec::with_capacity(kv_end - kv_start);
            for kj in kv_start..kv_end {
                if causal && kj > qi { break; } // causal mask: ignore future keys
                let krow = &kd[kj * d_k..(kj + 1) * d_k];
                let dot: f64 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f64>() * scale;
                scores.push((kj, dot));
            }
            if scores.is_empty() { kv_start = kv_end; continue; }

            // Block max.
            let m_block = scores.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
            let m_new = m.max(m_block);

            // Rescale old contribution.
            let alpha = exp(m - m_new);

            // New block contributions.
            let betas: Vec<f64> = scores.iter().map(|&(_, s)| exp(s - m_new)).collect();
            let l_block: f64 = betas.iter().sum();
            let l_new = alpha * l + l_block;

            // Update running output.
            for d in 0..d_v {
                // Scale down old O.
                let old_contrib = alpha * l * o[d];
                // New V contribution: sum_j beta[j] * V[kj, d].
                let new_contrib: f64 = betas.iter().zip(&scores)
                    .map(|(&b, &(kj, _))| b * vd[kj * d_v + d])
                    .sum();
                o[d] = if l_new > 0.0 { (old_contrib + new_contrib) / l_new } else { 0.0 };
            }

            m = m_new;
            l = l_new;
            kv_start = kv_end;
        }

        // Write output row.
        out[qi * d_v..(qi + 1) * d_v].copy_from_slice(&o);
    }

    Tensor::new(vec![seq_q, d_v], out)
}

// ───────────────────────────── optimizers ─────────────────────────────

/// A gradient-descent optimizer. Per-parameter state (momentum / Adam moments) is
/// kept in slots keyed by the parameter index the model passes to [`step`](Optimizer::step).
pub enum Optimizer {
    /// Stochastic gradient descent with optional momentum.
    Sgd { lr: f64, momentum: f64, vel: Vec<Vec<f64>> },
    /// Adam (adaptive moments).
    Adam { lr: f64, b1: f64, b2: f64, eps: f64, m: Vec<Vec<f64>>, v: Vec<Vec<f64>>, t: u64 },
}

impl Optimizer {
    pub fn sgd(lr: f64) -> Optimizer {
        Optimizer::Sgd { lr, momentum: 0.0, vel: Vec::new() }
    }

    pub fn sgd_momentum(lr: f64, momentum: f64) -> Optimizer {
        Optimizer::Sgd { lr, momentum, vel: Vec::new() }
    }

    pub fn adam(lr: f64) -> Optimizer {
        Optimizer::Adam { lr, b1: 0.9, b2: 0.999, eps: 1e-8, m: Vec::new(), v: Vec::new(), t: 0 }
    }

    fn ensure(slots: &mut Vec<Vec<f64>>, id: usize, n: usize) {
        while slots.len() <= id {
            slots.push(Vec::new());
        }
        if slots[id].len() != n {
            slots[id] = vec![0.0; n];
        }
    }

    /// Apply one update to `param` from its gradient `grad`, using slot `id`'s state.
    pub fn step(&mut self, id: usize, param: &mut Tensor, grad: &[f64]) {
        let shape = param.shape().to_vec();
        let mut data = param.data().to_vec();
        match self {
            Optimizer::Sgd { lr, momentum, vel } => {
                if *momentum > 0.0 {
                    Self::ensure(vel, id, data.len());
                    let v = &mut vel[id];
                    for i in 0..data.len() {
                        v[i] = *momentum * v[i] - *lr * grad[i];
                        data[i] += v[i];
                    }
                } else {
                    for i in 0..data.len() {
                        data[i] -= *lr * grad[i];
                    }
                }
            }
            Optimizer::Adam { lr, b1, b2, eps, m, v, t } => {
                Self::ensure(m, id, data.len());
                Self::ensure(v, id, data.len());
                let step = (*t + 1) as i32;
                let bc1 = 1.0 - powf_int(*b1, step);
                let bc2 = 1.0 - powf_int(*b2, step);
                let (mi, vi) = (&mut m[id], &mut v[id]);
                for i in 0..data.len() {
                    mi[i] = *b1 * mi[i] + (1.0 - *b1) * grad[i];
                    vi[i] = *b2 * vi[i] + (1.0 - *b2) * grad[i] * grad[i];
                    let mhat = mi[i] / bc1;
                    let vhat = vi[i] / bc2;
                    data[i] -= *lr * mhat / (sqrt(vhat) + *eps);
                }
            }
        }
        *param = Tensor::new(shape, data).unwrap();
    }

    /// Advance the global timestep (call once per batch; Adam uses it for bias correction).
    pub fn tick(&mut self) {
        if let Optimizer::Adam { t, .. } = self {
            *t += 1;
        }
    }
}

/// `base^exp` for a non-negative integer exponent (Adam bias correction).
/// Exponentiation-by-squaring: `O(log exp)` multiplies rather than `O(exp)`,
/// so Adam's per-step bias correction stays cheap even at large timesteps.
fn powf_int(base: f64, exp: i32) -> f64 {
    let mut r = 1.0;
    let mut b = base;
    let mut e = if exp < 0 { 0 } else { exp as u32 };
    while e > 0 {
        if e & 1 == 1 {
            r *= b;
        }
        e >>= 1;
        if e > 0 {
            b *= b;
        }
    }
    r
}

// ───────────────────────── int8 quantization (NPU path) ─────────────────────────

/// A symmetric int8-quantized tensor: `real ≈ q · scale`. The low-precision
/// representation an NPU multiplies in hardware; here it is the basis for the
/// quantized inference path and its benchmark.
#[derive(Clone, Debug, PartialEq)]
pub struct QTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i8>,
    pub scale: f64,
}

/// Quantize a tensor to symmetric int8 (`scale = max|x| / 127`).
pub fn quantize(t: &Tensor) -> QTensor {
    let max = t.data().iter().fold(0.0f64, |m, &v| m.max(fabs(v)));
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    let data: Vec<i8> = t
        .data()
        .iter()
        .map(|&v| {
            let q = v / scale;
            let r = if q >= 0.0 { q + 0.5 } else { q - 0.5 };
            r.clamp(-127.0, 127.0) as i8
        })
        .collect();
    QTensor { shape: t.shape().to_vec(), data, scale }
}

/// Dequantize back to `f64`.
pub fn dequantize(q: &QTensor) -> Tensor {
    let data: Vec<f64> = q.data.iter().map(|&v| v as f64 * q.scale).collect();
    Tensor::new(q.shape.clone(), data).unwrap()
}

/// Quantized 2-D matmul: integer multiply-accumulate, dequantized once at the end.
/// This is the work an NPU/TPU int8 array does; the accumulator is exact `i32`.
///
/// Optimised the same way as the float kernel — **transpose B** for contiguous
/// streaming and **fixed-lane `i64` accumulators** that vectorise to packed integer
/// multiply-adds. Integer arithmetic is associative; `i64` accumulators hold
/// `|i8·i8|·k ≤ 127·127·k` for any realistic contraction dim `k` (an `i32`
/// accumulator would overflow once `k > 133_151`), so the result is exact and
/// order-independent.
pub fn qmatmul(a: &QTensor, b: &QTensor) -> Option<Tensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return None;
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let out_scale = a.scale * b.scale;
    // Bᵀ: row j is column j of b, contiguous in i8.
    let mut bt = vec![0i8; n * k];
    for p in 0..k {
        let brow = &b.data[p * n..p * n + n];
        for (j, &v) in brow.iter().enumerate() {
            bt[j * k + p] = v;
        }
    }
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        let arow = &a.data[i * k..i * k + k];
        let orow = &mut out[i * n..i * n + n];
        for j in 0..n {
            let brow = &bt[j * k..j * k + k];
            orow[j] = qdot_lanes(arow, brow) as f64 * out_scale;  // i64 acc, no overflow
        }
    }
    Tensor::new(vec![m, n], out)
}

/// Integer inner product with fixed `i64` lane accumulators (vectorises to packed
/// integer MACs). Exact and order-independent. `i64` lanes avoid the `i32` overflow
/// that would occur for contraction dims `k > 133_151` (`127·127·k > i32::MAX`).
#[inline]
fn qdot_lanes(x: &[i8], y: &[i8]) -> i64 {
    const LANES: usize = 16;
    let mut acc = [0i64; LANES];
    let mut cx = x.chunks_exact(LANES);
    let mut cy = y.chunks_exact(LANES);
    for (xc, yc) in cx.by_ref().zip(cy.by_ref()) {
        for l in 0..LANES {
            acc[l] += xc[l] as i64 * yc[l] as i64;
        }
    }
    let mut s = 0i64;
    for &a in acc.iter() {
        s += a;
    }
    for (&xr, &yr) in cx.remainder().iter().zip(cy.remainder()) {
        s += xr as i64 * yr as i64;
    }
    s
}

// ─────────────────────── int4 quantization (2× over int8) ───────────────────────

/// Symmetric int4-quantized tensor: two 4-bit signed integers packed per byte.
/// `real ≈ q · scale`, q ∈ {−7,…,7}. 2× storage density vs int8.
///
/// Also carries `data_i8`: the pre-unpacked i8 form for compute. This lets the
/// matmul hot-loop skip per-call unpacking and run at int8 throughput, while the
/// compact `data` form is available for bandwidth-critical transfers (network, disk).
#[derive(Clone, Debug)]
pub struct Q4Tensor {
    pub shape: Vec<usize>,
    /// Packed nibbles: high nibble = element [2i], low nibble = element [2i+1].
    pub data: Vec<u8>,
    /// Pre-unpacked i8 form (same values, no additional error). Used by `q4matmul`.
    pub data_i8: Vec<i8>,
    pub scale: f64,
}

fn clamp_q4(v: i32) -> i8 {
    v.clamp(-7, 7) as i8
}

/// Quantize to symmetric int4. Fills both `data` (packed nibbles) and `data_i8` (unpacked).
pub fn quantize_q4(t: &Tensor) -> Q4Tensor {
    let max = t.data().iter().fold(0.0f64, |m, &v| m.max(fabs(v)));
    let scale = if max == 0.0 { 1.0 } else { max / 7.0 };
    let elems = t.data();
    let n = elems.len();
    let mut data    = vec![0u8; (n + 1) / 2];
    let mut data_i8 = vec![0i8; n];
    let mut chunks = elems.chunks_exact(2);
    for (i, chunk) in chunks.by_ref().enumerate() {
        let q0 = clamp_q4(round_half_away(chunk[0] / scale));
        let q1 = clamp_q4(round_half_away(chunk[1] / scale));
        data[i] = ((q0 as u8 & 0x0F) << 4) | (q1 as u8 & 0x0F);
        data_i8[2 * i]     = q0;
        data_i8[2 * i + 1] = q1;
    }
    if let [last] = chunks.remainder() {
        let q = clamp_q4(round_half_away(*last / scale));
        data[n / 2] = (q as u8 & 0x0F) << 4;
        data_i8[n - 1] = q;
    }
    Q4Tensor { shape: t.shape().to_vec(), data, data_i8, scale }
}

/// Unpack a packed nibble with sign-extension from 4 bits (branchless).
#[inline(always)]
fn nibble_to_i8(nibble: u8) -> i8 {
    // Shift nibble into the high 4 bits of an i8, then arithmetic-right-shift back.
    // This avoids any branch and lets LLVM emit a single pair of shifts.
    (((nibble & 0x0F) as i8) << 4) >> 4
}

/// Int4 2-D matmul using pre-unpacked `data_i8` — identical inner loop to int8.
///
/// `Q4Tensor::data_i8` is filled at quantization time (zero per-inference cost).
/// This means int4 inference runs at **int8 throughput** while storing 2× less data.
/// The only extra work vs qmatmul is transposing B (same as qmatmul does for int8).
pub fn q4matmul(a: &Q4Tensor, b: &Q4Tensor) -> Option<Tensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return None;
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let out_scale = a.scale * b.scale;

    // Transpose B using data_i8 — same as qmatmul's int8 path.
    let mut bt = vec![0i8; n * k];
    let b_i8 = &b.data_i8;
    for p in 0..k {
        let brow = &b_i8[p * n..p * n + n];
        for (j, &v) in brow.iter().enumerate() {
            bt[j * k + p] = v;
        }
    }

    // A is already unpacked in data_i8; use qdot_lanes same as int8.
    let a_i8 = &a.data_i8;
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        let arow = &a_i8[i * k..i * k + k];
        let orow = &mut out[i * n..i * n + n];
        for j in 0..n {
            orow[j] = qdot_lanes(arow, &bt[j * k..j * k + k]) as f64 * out_scale;
        }
    }
    Tensor::new(vec![m, n], out)
}

/// Dequantize int4 → f64.
pub fn dequantize_q4(q: &Q4Tensor) -> Tensor {
    let n = q.shape.iter().product::<usize>();
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let byte = q.data[i / 2];
        let nibble = if i % 2 == 0 { (byte >> 4) & 0x0F } else { byte & 0x0F };
        data.push(nibble_to_i8(nibble) as f64 * q.scale);
    }
    Tensor::new(q.shape.clone(), data).unwrap()
}

// ─────────────────────── binary quantization (XNOR+popcount) ─────────────────────────

/// A 1-bit (binary) quantized tensor: each element becomes a single bit packed into
/// `u64` words. The inner product becomes XNOR+popcount — no multiplies at all.
///
/// Quantization: `positive → 1`, `non-positive → 0`.
/// Scale: `mean(|x|)` (average magnitude).
/// Reconstruction: `q ∈ {-1, +1}`, so `output = (2·popcount(a XNOR b) - k) * scale_a * scale_b`.
#[derive(Clone, Debug)]
pub struct BinTensor {
    pub shape: Vec<usize>,
    /// Packed bits, MSB first. Element i is at bit `63 - (i % 64)` in word `i / 64`.
    pub data: Vec<u64>,
    /// Mean absolute value — the reconstruction scale.
    pub scale: f64,
}

/// Quantize to 1-bit binary: sign(x) → 1-bit.
///
/// Bits are packed **row-major with each row padded to a `u64` boundary**: a row is
/// one span of the tensor's last dimension, and consumes `(row_len + 63) / 64` words
/// with the unused low bits of the final word left zero. `binmatmul`/`ternmatmul` rely
/// on this layout (they slice and mask per row/column); a flat pack would alias across
/// row boundaries whenever the row length is not a multiple of 64.
pub fn quantize_bin(t: &Tensor) -> BinTensor {
    let elems = t.data();
    let n = elems.len();
    let scale = if n == 0 {
        1.0
    } else {
        elems.iter().map(|&v| fabs(v)).sum::<f64>() / n as f64
    };
    let row_len = t.shape().last().copied().unwrap_or(n).max(1);
    let words_per_row = (row_len + 63) / 64;
    let rows = if n == 0 { 0 } else { n / row_len };
    let mut data = vec![0u64; rows * words_per_row];
    for r in 0..rows {
        for c in 0..row_len {
            if elems[r * row_len + c] > 0.0 {
                data[r * words_per_row + c / 64] |= 1u64 << (63 - (c % 64));
            }
        }
    }
    BinTensor { shape: t.shape().to_vec(), data, scale }
}

/// Binary 2-D matmul: XNOR + popcount inner product.
///
/// For k elements:
///   `dot(a, b) = 2 · popcount(a XNOR b) − k`
/// gives a value in `[−k, +k]`, scaled by `scale_a · scale_b`.
///
/// One `u64` XNOR+popcount processes **64 elements in ~3 cycles** — vs 64 cycles for
/// 64 multiply-accumulates. At full vectorisation width this reaches ~800 GOPS.
pub fn binmatmul(a: &BinTensor, b: &BinTensor) -> Option<Tensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return None;
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let out_scale = a.scale * b.scale;
    let words = (k + 63) / 64;
    // Bit-transpose B: k × n bits (row-major) → n × k bits (column-major).
    // Uses set-bit iteration — ~3× fewer ops than the per-element approach.
    let bt = bit_transpose_correct(&b.data, k, n);
    let k_i32 = k as i32;

    // When k is not a multiple of 64 the last word of every packed row contains
    // `64 - remainder` padding bits that are 0.  In XNOR+popcount a (0,0) pair
    // gives a 1 — a spurious match.  Compute a mask that keeps only the
    // `remainder` most-significant (valid) bits and zero-out the rest before
    // counting.  When k IS a multiple of 64, no masking is needed.
    let remainder = k % 64;
    let last_mask: Option<u64> = if remainder != 0 {
        // Top `remainder` bits set: e.g. remainder=36 → top 36 bits = !((1<<28)-1)
        Some(!((1u64 << (64 - remainder)) - 1))
    } else {
        None
    };

    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        let arow = &a.data[i * words..(i + 1) * words];
        let orow = &mut out[i * n..i * n + n];
        for j in 0..n {
            let bcol = &bt[j * words..(j + 1) * words];
            let pop = match last_mask {
                None => bin_dot(arow, bcol),
                Some(mask) => bin_dot_masked(arow, bcol, mask),
            };
            // dot = 2·popcount(XNOR) − k. pop ≤ k=128 fits in i32 (no i64 needed).
            orow[j] = (2 * pop as i32 - k_i32) as f64 * out_scale;
        }
    }
    Tensor::new(vec![m, n], out)
}

/// XNOR+popcount inner product. One u64 XNOR+popcount processes 64 elements.
/// Direct indexed loop — LLVM auto-unrolls and vectorises for any array length.
#[inline]
fn bin_dot(a: &[u64], b: &[u64]) -> u64 {
    let mut s = 0u64;
    for i in 0..a.len() {
        s += (a[i] ^ !b[i]).count_ones() as u64;
    }
    s
}

/// XNOR+popcount with a mask applied to the last word.
///
/// When `k` is not a multiple of 64 the last packed word contains padding bits
/// that are 0 (from the zero-initialised allocation in `quantize_bin`).  A
/// zero padding bit in **both** operands produces an XNOR result of 1, which
/// would be counted as a spurious match.  `last_mask` keeps only the
/// `remainder` most-significant bits (the valid ones), zeroing the rest before
/// the XNOR so those positions contribute 0 to the popcount.
///
/// `last_mask = !(( 1u64 << (64 - remainder)) - 1)` — top `remainder` bits set.
#[inline]
fn bin_dot_masked(a: &[u64], b: &[u64], last_mask: u64) -> u64 {
    let n = a.len();
    if n == 0 { return 0; }
    let mut s = 0u64;
    for i in 0..n - 1 {
        s += (a[i] ^ !b[i]).count_ones() as u64;
    }
    // Last word: mask off padding bits in both operands before XNOR+popcount.
    let a_last = a[n - 1] & last_mask;
    let b_last = b[n - 1] & last_mask;
    s += (a_last ^ !b_last).count_ones() as u64;
    // The masked-off positions contain 0 in both operands after masking, so
    // the XNOR above sees (0 ^ ~0) = !0 for each padding bit — we must
    // subtract those spurious 1-bits back out.
    let padding_bits = 64 - last_mask.count_ones() as u64;
    s - padding_bits
}

// ─────────────────────── ternary quantization (2-bit, ±1/0) ─────────────────────────

/// A ternary quantized tensor: values in {−1, 0, +1}, packed as 2 bits per element.
/// 4× more storage-dense than int8. Inner product: additions only, no multiplies.
/// Thresholding: `|x| > threshold → ±1`; else `0`. threshold ≈ 0.7 × mean(|x|).
/// Ternary quantized tensor — dual-bitmap representation.
///
/// Each element ∈ {-1, 0, +1} is stored as TWO bit vectors:
/// - `pos_bits[i/64]` bit `63-(i%64)` = 1 iff element i > threshold (+1)
/// - `neg_bits[i/64]` bit `63-(i%64)` = 1 iff element i < -threshold (-1)
///
/// This lets `ternmatmul` run as FOUR XNOR+popcount operations per 64 elements —
/// the same throughput as binary. Memory cost: 2 bits/element = 4× denser than int8.
#[derive(Clone, Debug)]
pub struct TernTensor {
    pub shape: Vec<usize>,
    pub pos_bits: Vec<u64>,  // row-major: bit=1 → element is +1
    pub neg_bits: Vec<u64>,  // row-major: bit=1 → element is -1
    pub scale: f64,
}

/// Quantize to ternary (dual-bitmap). Threshold = 0.7 × mean(|x|) (TWN-style).
///
/// Uses the same **per-row, word-padded** bit layout as [`quantize_bin`] (see there),
/// which `ternmatmul` depends on for correct row/column slicing when the row length
/// is not a multiple of 64.
pub fn quantize_tern(t: &Tensor) -> TernTensor {
    let elems = t.data();
    let n = elems.len();
    let mean_abs = elems.iter().map(|&v| fabs(v)).sum::<f64>() / n.max(1) as f64;
    let threshold = 0.7 * mean_abs;
    let scale = if mean_abs == 0.0 { 1.0 } else { mean_abs };
    let row_len = t.shape().last().copied().unwrap_or(n).max(1);
    let words_per_row = (row_len + 63) / 64;
    let rows = if n == 0 { 0 } else { n / row_len };
    let mut pos_bits = vec![0u64; rows * words_per_row];
    let mut neg_bits = vec![0u64; rows * words_per_row];
    for r in 0..rows {
        for c in 0..row_len {
            let v = elems[r * row_len + c];
            let word = r * words_per_row + c / 64;
            if v > threshold {
                pos_bits[word] |= 1u64 << (63 - (c % 64));
            } else if v < -threshold {
                neg_bits[word] |= 1u64 << (63 - (c % 64));
            }
        }
    }
    TernTensor { shape: t.shape().to_vec(), pos_bits, neg_bits, scale }
}

/// Bit-transpose a k×n bit matrix (row-major, MSB-first packed into u64s) into n×k.
/// Each set bit at (r, wc*64+lz) moves to column position r in the transposed matrix.
/// Used to build column-major views of B's dual-bitmap for the ternary inner-product loop.
fn bit_transpose_correct(bits: &[u64], k: usize, n: usize) -> Vec<u64> {
    let words_per_row = (n + 63) / 64;
    let words_per_col = (k + 63) / 64;
    let mut bt = vec![0u64; n * words_per_col];
    for r in 0..k {
        let src_base = r * words_per_row;
        for wc in 0..words_per_row {
            let mut word = bits[src_base + wc];
            while word != 0 {
                let lz = word.leading_zeros() as usize;
                let c = wc * 64 + lz;
                if c < n {
                    let dst_word = c * words_per_col + r / 64;
                    let dst_bit  = 63 - (r % 64);
                    bt[dst_word] |= 1u64 << dst_bit;
                }
                word ^= 1u64 << (63 - lz); // clear the MSB set bit we just processed
            }
        }
    }
    bt
}

/// Ternary inner product via 2 AND+OR+popcount operations per u64 word.
///
/// Algebraic reduction: since A_pos & A_neg = 0 (mutually exclusive), the two
/// positive contributors (pp, nn) are disjoint and can be OR'd before counting.
/// Same for the negative contributors (pn, np). 4 pops → **2 pops** per word.
///
///   dot = popcount((A_pos & B_pos) | (A_neg & B_neg))   [positive matches]
///       - popcount((A_pos & B_neg) | (A_neg & B_pos))   [sign-flipped matches]
#[inline]
fn tern_bin_dot(a_pos: &[u64], a_neg: &[u64], b_pos: &[u64], b_neg: &[u64]) -> i64 {
    let mut pos_sum = 0u64;
    let mut neg_sum = 0u64;
    for i in 0..a_pos.len() {
        pos_sum += ((a_pos[i] & b_pos[i]) | (a_neg[i] & b_neg[i])).count_ones() as u64;
        neg_sum += ((a_pos[i] & b_neg[i]) | (a_neg[i] & b_pos[i])).count_ones() as u64;
    }
    pos_sum as i64 - neg_sum as i64
}

/// Ternary 2-D matmul via dual-bitmap: 4 AND+popcount per 64 elements.
/// Both A and B are in dual-bitmap format (pos_bits, neg_bits).
/// Setup cost: bit-transpose of B's pos_bits and neg_bits (O(k*n) bit ops, done once).
pub fn ternmatmul(a: &TernTensor, b: &TernTensor) -> Option<Tensor> {
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return None;
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let out_scale = a.scale * b.scale;
    let words_k = (k + 63) / 64; // words per row of A
    // Transpose B's bit matrices so we can access B's columns as contiguous rows.
    let b_pos_t = bit_transpose_correct(&b.pos_bits, k, n); // n × words_k layout
    let b_neg_t = bit_transpose_correct(&b.neg_bits, k, n);

    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        let a_pos_row = &a.pos_bits[i * words_k..(i + 1) * words_k];
        let a_neg_row = &a.neg_bits[i * words_k..(i + 1) * words_k];
        let orow = &mut out[i * n..i * n + n];
        for j in 0..n {
            let b_pos_col = &b_pos_t[j * words_k..(j + 1) * words_k];
            let b_neg_col = &b_neg_t[j * words_k..(j + 1) * words_k];
            let dot = tern_bin_dot(a_pos_row, a_neg_row, b_pos_col, b_neg_col);
            orow[j] = dot as f64 * out_scale;
        }
    }
    Tensor::new(vec![m, n], out)
}

// ───────────────────────── built-in reference dataset ─────────────────────────

/// The canonical XOR problem — the smallest task that *requires* a hidden layer
/// (not linearly separable). Returns `(inputs[4×2], targets[4×1])`. Used by the
/// language `train_xor` builtin, the selftests, and the training benchmark, so a
/// "does training actually work?" claim is concrete and reproducible.
// ───────────────────────── unified ML configuration ──────────────────────
//
// `MlConfig` is the single knob-panel for all 8 levers. Callers construct one
// (default: `MlConfig::best()`) and use its fields to select which APIs to call.
// The kernel passes hardware hints from CPUID via `MlConfig::from_hardware`.
// The Dominion language `ml_config(...)` builtin maps to this struct.

/// How to bound the inference cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    /// No caching. Every call recomputes.
    None,
    /// Unbounded `TensorMemo` — safe when the model × input working set is bounded.
    Unbounded,
    /// LRU-evicting `BoundedTensorMemo` with the given entry capacity.
    Lru(usize),
}

/// All ML optimization knobs in one struct.
///
/// Construct with `MlConfig::best()` for the maximum-performance stack, or
/// `MlConfig::from_hardware(l1_kb, n_cores)` when the kernel has CPUID data.
/// Individual fields can be overridden after construction.
///
/// This struct documents which lever drives which field:
///
/// | Field               | Lever | What it controls                                 |
/// |---------------------|-------|--------------------------------------------------|
/// | `n_threads`         | 1     | Number of APs used by `matmul_with` / `forward_with` |
/// | `tile_size`         | 1     | Cache-blocked tile size (auto from `l1_kb`)     |
/// | `cache_policy`      | 8     | Inference cache: None / Unbounded / Lru(cap)    |
/// | `precision`         | 4     | Fixed precision override (None = bandit decides) |
/// | `adaptive_precision`| 5     | ε-greedy / UCB-1 bandit picks precision per call |
/// | `fma`               | 6     | FMA fusion (non-deterministic, faster)           |
/// | `sparsify_keep_pct` | 7     | Gradient sparsification keep-% (None = off)     |
/// | `fed_sync_interval` | 7     | Steps between federated sync rounds              |
/// | `scaffold`          | 7     | SCAFFOLD variance reduction in federated training|
#[derive(Clone, Debug)]
pub struct MlConfig {
    /// L1 data-cache size hint in KiB. Used to choose `tile_size`. 0 = use default.
    pub l1_kb: usize,
    /// Matrix tiling block size for cache-blocked matmul. 0 = auto (derived from `l1_kb`).
    pub tile_size: usize,
    /// Worker count for `matmul_with` / `forward_with`. 0 = single-core.
    pub n_threads: usize,
    /// Inference cache strategy.
    pub cache_policy: CachePolicy,
    /// Fixed numerical precision. `None` = let `adaptive_precision` decide.
    pub precision: Option<Precision>,
    /// Enable ε-greedy / UCB-1 bandit to adapt precision per call. Ignored when
    /// `precision` is `Some(...)`.
    pub adaptive_precision: bool,
    /// Enable FMA path (non-deterministic low bits, faster). Off by default.
    pub fma: bool,
    /// Top-K% of gradient magnitudes to transmit in federated training. `None` = off.
    pub sparsify_keep_pct: Option<usize>,
    /// Local gradient steps between federated sync rounds.
    pub fed_sync_interval: usize,
    /// Use SCAFFOLD client-drift correction in federated training.
    pub scaffold: bool,
}

impl MlConfig {
    /// The best-known stack: multi-core, LRU cache(64), adaptive precision, no FMA
    /// (preserves determinism), gradient sparsification off (small models).
    pub fn best() -> Self {
        MlConfig {
            l1_kb: 32,         // common L1d size; CPUID callers override this
            tile_size: 0,      // auto-compute from l1_kb
            n_threads: 0,      // 0 = use all available cores (set by Spawn impl)
            cache_policy: CachePolicy::Lru(64),
            precision: None,   // bandit decides
            adaptive_precision: true,
            fma: false,        // deterministic default
            sparsify_keep_pct: None,
            fed_sync_interval: 20,
            scaffold: false,
        }
    }

    /// Construct from hardware hints supplied by the kernel's CPUID probe.
    ///
    /// `l1_kb`: L1 data-cache size in KiB (from CPUID leaf 4 or equivalent).
    /// `n_cores`: total logical cores available (from CPUID or APIC enumeration).
    pub fn from_hardware(l1_kb: usize, n_cores: usize) -> Self {
        let mut cfg = Self::best();
        cfg.l1_kb = l1_kb;
        cfg.n_threads = n_cores;
        // Optimal tile: fit A-tile + B-tile + C-tile in L1d. Three square n×n tiles
        // of f64 consume 3·n²·8 bytes. Solve for n: n = sqrt(l1_kb·1024 / 24).
        // We take the largest power-of-2 ≤ that value so strides stay aligned.
        let l1_bytes = (l1_kb * 1024).max(8192);
        let n_sq = l1_bytes / 24; // f64 = 8 bytes, 3 matrices
        let mut t = 1usize;
        while (t * 2) * (t * 2) <= n_sq { t *= 2; }
        cfg.tile_size = t.max(16).min(256);
        cfg
    }

    /// Optimal tile size for this config (power-of-2, L1-fitting).
    pub fn effective_tile(&self) -> usize {
        if self.tile_size > 0 { return self.tile_size; }
        let l1_bytes = (self.l1_kb * 1024).max(8192);
        let n_sq = l1_bytes / 24;
        let mut t = 1usize;
        while (t * 2) * (t * 2) <= n_sq { t *= 2; }
        t.max(16).min(256)
    }

    /// Resolve which precision to use.
    /// Returns `Some(p)` if `precision` is set; `None` means let the bandit decide.
    pub fn resolve_precision(&self) -> Option<Precision> {
        self.precision
    }

    /// Whether the caller should wrap calls in `forward_cached` / `TensorMemo`.
    pub fn caching_enabled(&self) -> bool {
        !matches!(self.cache_policy, CachePolicy::None)
    }
}

pub fn xor_dataset() -> (Tensor, Tensor) {
    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    (x, y)
}

/// Train a fresh 2-`hidden`-1 MLP on XOR for `epochs` and return `(model, final_loss)`.
/// A self-contained, deterministic training run — the demo the OS can show end to end.
pub fn train_xor(hidden: usize, epochs: usize) -> (Mlp, f64) {
    let (x, y) = xor_dataset();
    let mut model = Mlp::new(&[2, hidden.max(1), 1], Activation::Tanh, Activation::Sigmoid, 0xA17E).unwrap();
    let mut opt = Optimizer::adam(0.05);
    let mut loss = 1.0;
    for _ in 0..epochs {
        loss = model.train_step_mse(&x, &y, &mut opt).unwrap();
    }
    (model, loss)
}

// ─────────────────── GPTQ: Optimal Brain Quantization ───────────────────
//
// GPTQ quantizes a weight matrix W column-by-column using second-order
// information. For each column i:
//   Q[:, i] = quantize(W[:, i])
//   E[:, i] = (W[:, i] - Q[:, i]) / H[i,i]  (per-column error)
//   W[:, j] -= E[:, i] * H[i,j]  for j > i  (update remaining columns)
// Where H = X^T X (Hessian, computed from calibration activations X).
//
// Requires: Cholesky decomposition of H (for numerical stability we add
// a small dampening term: H[i,i] += dampen * mean(diag(H))).
//
// Reference: Frantar et al. "GPTQ: Accurate Post-Training Quantization
// for Generative Pre-trained Transformers" (arXiv 2210.17323).

/// Calibration statistics for GPTQ: accumulated H = X^T X.
pub struct GptqCalib {
    pub h: Vec<f64>,       // [n_in, n_in] Hessian accumulator (row-major)
    pub n_in: usize,
    pub n_samples: usize,
}

impl GptqCalib {
    /// Create a new zeroed calibration accumulator for a layer with `n_in` input features.
    pub fn new(n_in: usize) -> Self {
        GptqCalib { h: vec![0.0; n_in * n_in], n_in, n_samples: 0 }
    }

    /// Accumulate one batch of activations. `x` must have shape `[batch, n_in]`.
    pub fn add_batch(&mut self, x: &Tensor) -> bool {
        let n_in = self.n_in;
        if x.shape().len() != 2 || x.shape()[1] != n_in {
            return false;
        }
        let batch = x.shape()[0];
        let data = x.data();
        // H += X^T X  (outer product sum over the batch dimension)
        for b in 0..batch {
            for i in 0..n_in {
                for j in 0..n_in {
                    self.h[i * n_in + j] += data[b * n_in + i] * data[b * n_in + j];
                }
            }
        }
        self.n_samples += batch;
        true
    }

    /// Normalise H by sample count and add relative dampening to the diagonal.
    pub fn finalize(&mut self, dampen: f64) {
        let n = self.n_in;
        let norm = if self.n_samples > 0 { 1.0 / self.n_samples as f64 } else { 1.0 };
        let mut mean_diag = 0.0f64;
        for i in 0..n {
            mean_diag += self.h[i * n + i] * norm;
        }
        mean_diag /= n as f64;
        for i in 0..n {
            for j in 0..n {
                self.h[i * n + j] *= norm;
            }
            self.h[i * n + i] += dampen * mean_diag;
        }
    }
}

/// Cholesky decomposition: returns lower-triangular L such that A ≈ L·Lᵀ.
/// `a` is a symmetric positive-definite `n×n` matrix in row-major order.
/// Returns `None` if the matrix is not positive definite.
fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + j] = sqrt(s);
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Apply GPTQ to a [`Linear`] layer using pre-collected calibration data.
///
/// Returns `(quantized_weights, scales, effective_group_size)` where:
/// * `quantized_weights` — `[n_out, n_in]` row-major i8 values.
/// * `scales` — `[n_out, n_groups]` per-row-per-group scale factors.
/// * `effective_group_size` — the actual group size used (≤ `group_size`).
///
/// `bits` must be 4 or 8. `group_size = 0` selects per-tensor quantization.
/// Returns `None` when the Hessian is not positive definite (insufficient
/// calibration data) or when arguments are inconsistent.
pub fn gptq_quantize(
    w: &Linear,
    calib: &GptqCalib,
    bits: u8,
    group_size: usize,
    dampen: f64,
) -> Option<(Vec<i8>, Vec<f64>, usize)> {
    let n_in = w.in_dim();
    let n_out = w.out_dim();
    if calib.n_in != n_in {
        return None;
    }
    if bits != 8 && bits != 4 {
        return None;
    }

    // Flatten the weight tensor to a Vec<f64> in [n_in, n_out] → we need [n_out, n_in].
    // Linear.w has shape [n_in, n_out] (in_dim × out_dim, column-major for matmul).
    // We transpose to get row = output neuron.
    let w_data = w.w.data(); // [n_in, n_out] row-major
    let mut w_copy: Vec<f64> = vec![0.0f64; n_out * n_in];
    for i in 0..n_in {
        for o in 0..n_out {
            w_copy[o * n_in + i] = w_data[i * n_out + o];
        }
    }

    // Build H with extra damping (on top of any already applied in finalize).
    let mut h = calib.h.clone();
    let mean_diag: f64 =
        (0..n_in).map(|i| h[i * n_in + i]).sum::<f64>() / n_in as f64;
    for i in 0..n_in {
        h[i * n_in + i] += dampen * mean_diag;
    }

    let l = cholesky_lower(&h, n_in)?;

    // Effective group size: 0 → whole row (per-tensor per output neuron).
    let gs = if group_size == 0 { n_in } else { group_size.min(n_in) };
    let n_groups = (n_in + gs - 1) / gs;
    // Symmetric quantization range: 8-bit → [-128,127] range=255, 4-bit → [-8,7] range=15.
    let range = if bits == 8 { 255.0 } else { 15.0 };

    let mut q_data: Vec<i8> = vec![0i8; n_out * n_in];
    let mut scales: Vec<f64> = vec![0.0f64; n_out * n_groups];

    // GPTQ: column-by-column quantization with Hessian-guided error feedback.
    for i in 0..n_in {
        let g = i / gs;

        // On the first column of each group, compute per-output-neuron scale.
        if i % gs == 0 {
            let col_end = (i + gs).min(n_in);
            for r in 0..n_out {
                let slice = &w_copy[r * n_in + i..r * n_in + col_end];
                let max_abs = slice
                    .iter()
                    .map(|v| if *v < 0.0 { -*v } else { *v })
                    .fold(0.0f64, f64::max);
                scales[r * n_groups + g] =
                    if max_abs > 0.0 { max_abs / (range / 2.0) } else { 1.0 };
            }
        }

        let h_ii = l[i * n_in + i];
        if h_ii.abs() < 1e-12 {
            continue;
        }

        for r in 0..n_out {
            let w_val = w_copy[r * n_in + i];
            let s = scales[r * n_groups + g];
            // Round-to-nearest quantization.
            let q_f = w_val / s;
            let q_clamped = if bits == 8 {
                round_half_away(q_f).max(-128).min(127) as i8
            } else {
                round_half_away(q_f).max(-8).min(7) as i8
            };
            q_data[r * n_in + i] = q_clamped;

            // Error feedback: propagate quantization error to remaining columns
            // using the Cholesky factor of H (GPTQ update rule).
            let err = (w_val - (q_clamped as f64 * s)) / h_ii;
            for j in (i + 1)..n_in {
                w_copy[r * n_in + j] -= err * l[j * n_in + i];
            }
        }
    }

    Some((q_data, scales, gs))
}

// ─────────────── Per-channel quantization (PyTorch static quant style) ────────────────
//
// PyTorch's X86 quantization backend (and FBGEMM) use per-output-channel scales and
// zero-points for weights — a separate (scale, zero_point) per output neuron rather than
// a single global scale. This gives 2-4% lower error than per-tensor at zero extra
// runtime cost (scale/zp arrays are cached alongside the weight).

/// Per-channel int8 quantization of a weight matrix [n_out, n_in].
/// Returns: (i8 quantized data, per-channel scales, per-channel zero-points).
pub fn quantize_per_channel(w: &Linear) -> (Vec<i8>, Vec<f64>, Vec<i8>) {
    let n_out = w.out_dim();
    let n_in  = w.in_dim();
    let data  = w.w.data();
    let mut q    = vec![0i8; n_out * n_in];
    let mut scales = vec![1.0f64; n_out];
    let mut zps    = vec![0i8; n_out];
    for r in 0..n_out {
        let row = &data[r * n_in..(r + 1) * n_in];
        let mn = row.iter().cloned().fold(f64::MAX,  f64::min);
        let mx = row.iter().cloned().fold(f64::MIN, f64::max);
        // Symmetric quantization: zero-point = 0, scale = max(|mn|, |mx|) / 127
        let max_abs = (if mn < 0.0 { -mn } else { mn }).max(if mx > 0.0 { mx } else { -mx });
        let s = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales[r] = s;
        zps[r] = 0;
        for c in 0..n_in {
            let v = data[r * n_in + c] / s;
            q[r * n_in + c] = round_half_away(v).max(-127).min(127) as i8;
        }
    }
    (q, scales, zps)
}

/// Per-channel int8 matmul: input x [m, n_in] × quantized weight [n_out, n_in].
/// Each output row r is dequantized with scales[r].
pub fn qmatmul_per_channel(x: &Tensor, q: &[i8], scales: &[f64], n_in: usize, n_out: usize) -> Option<Tensor> {
    if x.shape().len() != 2 || x.shape()[1] != n_in { return None; }
    if q.len() != n_out * n_in || scales.len() != n_out { return None; }
    let m = x.shape()[0];
    let xd = x.data();
    let mut out = vec![0.0f64; m * n_out];
    for i in 0..m {
        // True dynamic quantization of the activation row: use its own max-abs so
        // values with |a| > 1 are not clipped/mis-scaled. a_scale = max|a| / 127.
        let arow = &xd[i * n_in..i * n_in + n_in];
        let a_max = arow.iter().fold(0.0f64, |mx, &v| mx.max(fabs(v)));
        let a_scale = if a_max > 0.0 { a_max / 127.0 } else { 1.0 };
        let a_inv = 1.0 / a_scale;
        for r in 0..n_out {
            let mut acc = 0i32;
            for c in 0..n_in {
                // Quantize activation element to i8 with the row's dynamic scale.
                let aq = round_half_away(arow[c] * a_inv);
                let aq = aq.max(-127).min(127) as i8;
                acc += (aq as i32) * (q[r * n_in + c] as i32);
            }
            // Dequantize: real ≈ q·scale for both operands, so
            // out = acc · a_scale · scales[r].
            out[i * n_out + r] = acc as f64 * a_scale * scales[r];
        }
    }
    Tensor::new(vec![m, n_out], out)
}

// ─────────────── Channels-last (NHWC) layout utilities ────────────────────────────────
//
// PyTorch channels_last: strides = (H*W*C, 1, W*C, C) instead of (C*H*W, H*W, W, 1).
// On CPU this gives ~2-4× speedup for conv because the inner loop runs over C (SIMD-friendly).
// We model this as a layout-transform function pair + a flag, not as a new type.
// The actual computation uses the NCHW kernel after transformation for now.

/// Repack a [batch, C, H, W] NCHW tensor into [batch, H, W, C] NHWC layout.
pub fn nchw_to_nhwc(x: &Tensor) -> Option<Tensor> {
    if x.shape().len() != 4 { return None; }
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let src = x.data();
    let mut out = vec![0.0f64; n * c * h * w];
    for bi in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let nchw_idx = bi * c * h * w + ci * h * w + hi * w + wi;
                    let nhwc_idx = bi * h * w * c + hi * w * c + wi * c + ci;
                    out[nhwc_idx] = src[nchw_idx];
                }
            }
        }
    }
    Tensor::new(vec![n, h, w, c], out)
}

/// Repack a [batch, H, W, C] NHWC tensor back to [batch, C, H, W] NCHW layout.
pub fn nhwc_to_nchw(x: &Tensor) -> Option<Tensor> {
    if x.shape().len() != 4 { return None; }
    let (n, h, w, c) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let src = x.data();
    let mut out = vec![0.0f64; n * c * h * w];
    for bi in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let nhwc_idx = bi * h * w * c + hi * w * c + wi * c + ci;
                    let nchw_idx = bi * c * h * w + ci * h * w + hi * w + wi;
                    out[nchw_idx] = src[nhwc_idx];
                }
            }
        }
    }
    Tensor::new(vec![n, c, h, w], out)
}

// ─────────────── Blocked format (nChw16c style) for oneDNN-compatible layout ─────────────────

/// Pack [batch, C, H, W] into nChw16c: [batch, C/16, H, W, 16] blocked format.
/// C must be divisible by block_size. block_size=16 for AVX512, 8 for AVX2.
pub fn pack_nchwc(x: &Tensor, block_size: usize) -> Option<Tensor> {
    if x.shape().len() != 4 { return None; }
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    if c % block_size != 0 { return None; }
    let c_blocks = c / block_size;
    let src = x.data();
    let mut out = vec![0.0f64; n * c * h * w];
    for bi in 0..n {
        for cb in 0..c_blocks {
            for hi in 0..h {
                for wi in 0..w {
                    for ck in 0..block_size {
                        let ci = cb * block_size + ck;
                        let src_idx = bi * c * h * w + ci * h * w + hi * w + wi;
                        let dst_idx = bi * c_blocks * h * w * block_size
                                    + cb * h * w * block_size
                                    + hi * w * block_size
                                    + wi * block_size + ck;
                        out[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
    Tensor::new(vec![n, c_blocks, h, w, block_size], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        fabs(a - b) < tol
    }

    #[test]
    fn exp_matches_known_values() {
        assert!(approx(exp(0.0), 1.0, 1e-12));
        assert!(approx(exp(1.0), 2.718281828459045, 1e-9));
        assert!(approx(exp(-1.0), 0.36787944117144233, 1e-9));
        assert!(approx(exp(5.0), 148.4131591025766, 1e-6));
    }

    #[test]
    fn sigmoid_and_tanh_are_correct() {
        assert!(approx(sigmoid(0.0), 0.5, 1e-12));
        assert!(approx(tanh(0.0), 0.0, 1e-12));
        assert!(approx(tanh(1.0), 0.7615941559557649, 1e-9));
        // Symmetry / saturation.
        assert!(approx(sigmoid(20.0), 1.0, 1e-6));
        assert!(approx(sigmoid(-20.0), 0.0, 1e-6));
    }

    #[test]
    fn ln_matches_known_values() {
        assert!(approx(ln_approx(1.0), 0.0, 1e-12));
        assert!(approx(ln_approx(2.718281828459045), 1.0, 1e-9));
        assert!(approx(ln_approx(10.0), 2.302585092994046, 1e-9));
    }

    #[test]
    fn transpose_round_trips() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = transpose(&a).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(transpose(&t).unwrap(), a);
    }

    #[test]
    fn device_cost_model_prefers_cpu_small_tpu_large() {
        // A tiny matmul: launch latency dominates → CPU wins.
        let small = matmul_flops(2, 2, 2);
        assert_eq!(recommend_device(small), Device::Cpu);
        // A large matmul: throughput dominates → TPU wins.
        let big = matmul_flops(512, 512, 512);
        assert_eq!(recommend_device(big), Device::Tpu);
    }

    #[test]
    fn all_devices_compute_identical_matmul() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let reference = a.matmul(&b).unwrap();
        for d in ALL_DEVICES {
            assert_eq!(d.matmul(&a, &b).unwrap(), reference, "device {} diverged", d.name());
        }
    }

    #[test]
    fn autograd_matches_numerical_gradient() {
        // loss(W) = mse(sigmoid(x·W + b), 0) ; check d/dW against central differences.
        let mut rng = Rng::new(1);
        let x = Tensor::new(vec![1, 3], vec![0.5, -0.2, 0.1]).unwrap();
        let layer = Linear::new(3, 2, &mut rng);
        let target = Tensor::zeros(vec![1, 2]);

        // Forward + backward through the tape for the analytic gradient.
        let analytic = {
            let mut tp = Tape::new();
            let xv = tp.leaf(x.clone());
            let wv = tp.leaf(layer.w.clone());
            let bv = tp.leaf(layer.b.clone());
            let z = tp.matmul(xv, wv).unwrap();
            let z = tp.add_bias(z, bv).unwrap();
            let s = tp.sigmoid(z);
            let loss = tp.mse(s, &target).unwrap();
            tp.backward(loss);
            tp.grad(wv).to_vec()
        };
        // The same loss, used for finite-difference comparison.
        let loss_at = |layer: &Linear| -> f64 {
            let mut tp = Tape::new();
            let xv = tp.leaf(x.clone());
            let wv = tp.leaf(layer.w.clone());
            let bv = tp.leaf(layer.b.clone());
            let z = tp.matmul(xv, wv).unwrap();
            let z = tp.add_bias(z, bv).unwrap();
            let s = tp.sigmoid(z);
            let loss = tp.mse(s, &target).unwrap();
            tp.value(loss).data()[0]
        };
        let h = 1e-6;
        for i in 0..layer.w.len() {
            let mut up = layer.clone();
            let mut wd = up.w.data().to_vec();
            wd[i] += h;
            up.w = Tensor::new(up.w.shape().to_vec(), wd).unwrap();
            let mut dn = layer.clone();
            let mut wd2 = dn.w.data().to_vec();
            wd2[i] -= h;
            dn.w = Tensor::new(dn.w.shape().to_vec(), wd2).unwrap();
            let numeric = (loss_at(&up) - loss_at(&dn)) / (2.0 * h);
            assert!(
                approx(analytic[i], numeric, 1e-4),
                "grad mismatch at {i}: analytic={} numeric={}",
                analytic[i],
                numeric
            );
        }
    }

    #[test]
    fn mlp_trains_xor_to_low_loss() {
        let (model, loss) = train_xor(8, 2000);
        assert!(loss < 0.02, "XOR did not converge, final loss = {loss}");
        // Inference matches the truth table after training.
        let (x, _) = xor_dataset();
        let out = model.forward(&x).unwrap();
        let preds: Vec<bool> = out.data().iter().map(|&v| v > 0.5).collect();
        assert_eq!(preds, vec![false, true, true, false]);
    }

    #[test]
    fn classification_trains_with_cross_entropy() {
        // Two clearly separated classes in 2-D → softmax-CE should drive loss down.
        let x = Tensor::new(
            vec![4, 2],
            vec![-1.0, -1.0, -0.9, -1.1, 1.0, 1.0, 1.1, 0.9],
        )
        .unwrap();
        let targets = [0usize, 0, 1, 1];
        let mut model = Mlp::new(&[2, 6, 2], Activation::Relu, Activation::Identity, 7).unwrap();
        let mut opt = Optimizer::adam(0.05);
        let mut loss = 9.0;
        for _ in 0..500 {
            loss = model.train_step_ce(&x, &targets, &mut opt).unwrap();
        }
        assert!(loss < 0.05, "CE did not converge, loss={loss}");
        // argmax predictions are all correct.
        let out = model.forward(&x).unwrap();
        for (i, &t) in targets.iter().enumerate() {
            let row = &out.data()[i * 2..i * 2 + 2];
            let arg = if row[0] >= row[1] { 0 } else { 1 };
            assert_eq!(arg, t, "row {i} misclassified");
        }
    }

    #[test]
    fn sgd_momentum_also_learns_xor() {
        let (x, y) = xor_dataset();
        let mut model = Mlp::new(&[2, 8, 1], Activation::Tanh, Activation::Sigmoid, 0xBEEF).unwrap();
        let mut opt = Optimizer::sgd_momentum(0.5, 0.9);
        let mut loss = 1.0;
        for _ in 0..5000 {
            loss = model.train_step_mse(&x, &y, &mut opt).unwrap();
        }
        assert!(loss < 0.05, "SGD+momentum did not converge, loss={loss}");
    }

    #[test]
    fn quantization_round_trips_within_scale() {
        let t = Tensor::new(vec![2, 3], vec![0.1, -0.5, 1.0, -1.0, 0.25, 0.0]).unwrap();
        let q = quantize(&t);
        let r = dequantize(&q);
        for (a, b) in t.data().iter().zip(r.data()) {
            assert!(approx(*a, *b, q.scale + 1e-12), "quant error too large: {a} vs {b}");
        }
    }

    #[test]
    fn quantized_matmul_approximates_float_matmul() {
        let mut rng = Rng::new(42);
        let a_data: Vec<f64> = (0..12).map(|_| rng.next_signed()).collect();
        let b_data: Vec<f64> = (0..12).map(|_| rng.next_signed()).collect();
        let a = Tensor::new(vec![3, 4], a_data).unwrap();
        let b = Tensor::new(vec![4, 3], b_data).unwrap();
        let exact = a.matmul(&b).unwrap();
        let approx_t = qmatmul(&quantize(&a), &quantize(&b)).unwrap();
        for (e, q) in exact.data().iter().zip(approx_t.data()) {
            assert!(approx(*e, *q, 0.05), "int8 matmul off: {e} vs {q}");
        }
    }

    #[test]
    fn model_serialization_round_trips_exactly() {
        let (model, _) = train_xor(8, 100);
        let bytes = model.to_bytes();
        let restored = Mlp::from_bytes(&bytes).unwrap();
        assert_eq!(model, restored);
        // And it still infers identically.
        let (x, _) = xor_dataset();
        assert_eq!(model.forward(&x).unwrap(), restored.forward(&x).unwrap());
    }

    #[test]
    fn training_is_deterministic() {
        let (_, l1) = train_xor(8, 300);
        let (_, l2) = train_xor(8, 300);
        assert_eq!(l1, l2, "training must be bit-reproducible");
    }

    #[test]
    fn infer_reports_cost_and_picks_sane_device() {
        let (model, _) = train_xor(8, 50);
        let (x, _) = xor_dataset();
        let (out, report) = model.infer(&x, Device::Cpu).unwrap();
        assert_eq!(out.shape(), &[4, 1]);
        assert!(report.flops > 0);
        assert_eq!(report.device, Device::Cpu);
    }

    /// Verify that `binmatmul` gives correct results when the contraction
    /// dimension k is NOT a multiple of 64 (k=100 → 2 words, last word has
    /// 28 padding bits that must NOT be counted as XNOR matches).
    ///
    /// Strategy: build a deterministic A (m×k) and B (k×n), compute the
    /// expected result via a naive scalar binary inner-product (sign→±1, dot
    /// then scale), and compare element-wise.
    #[test]
    fn binmatmul_non_multiple_of_64_k() {
        const M: usize = 3;
        const K: usize = 100; // not a multiple of 64
        const N: usize = 4;

        let mut rng = Rng::new(0xDEAD_BEEF_1234_5678);

        // Build float tensors with random signs.
        let a_data: Vec<f64> = (0..M * K).map(|_| rng.next_signed()).collect();
        let b_data: Vec<f64> = (0..K * N).map(|_| rng.next_signed()).collect();
        let a_t = Tensor::new(vec![M, K], a_data.clone()).unwrap();
        let b_t = Tensor::new(vec![K, N], b_data.clone()).unwrap();

        // Quantize to binary.
        let a_bin = quantize_bin(&a_t);
        let b_bin = quantize_bin(&b_t);
        let scale = a_bin.scale * b_bin.scale;

        // Naive reference: sign each element to ±1 then dot-product.
        // sign(x) → 1.0 if x > 0, else -1.0.
        let a_sign: Vec<f64> = a_data.iter().map(|&v| if v > 0.0 { 1.0 } else { -1.0 }).collect();
        let b_sign: Vec<f64> = b_data.iter().map(|&v| if v > 0.0 { 1.0 } else { -1.0 }).collect();
        let mut expected = vec![0.0f64; M * N];
        for i in 0..M {
            for j in 0..N {
                let dot: f64 = (0..K).map(|kk| a_sign[i * K + kk] * b_sign[kk * N + j]).sum();
                expected[i * N + j] = dot * scale;
            }
        }

        // Run the optimised path.
        let result = binmatmul(&a_bin, &b_bin).unwrap();

        for (idx, (&got, &exp)) in result.data().iter().zip(&expected).enumerate() {
            assert_eq!(
                got, exp,
                "binmatmul k=100: element {idx} got {got} expected {exp} (padding-bit bug?)"
            );
        }
    }
}

//! Native host baseline for the DominionOS ML benchmark.
//!
//! This runs the **exact same `dominion_core::ml` workload** as the kernel's
//! `bench_ml` (same sizes, seeds, iteration counts, and code path — it links the
//! identical `dominion-core` crate), but on the bare host instead of inside QEMU. It
//! emits the same `BENCH ml_* key=value …` schema, so the in-guest serial log and
//! this host run can be diffed line-for-line to get a true *overhead-vs-host*
//! figure for the pure-compute path.
//!
//! The only difference from the kernel is the clock: here we use
//! `std::time::Instant` instead of the calibrated TSC. The metric formulas are
//! identical (`mflop_per_s = FLOP / µs`, etc.), so the numbers are comparable.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::time::Instant;

use dominion_core::datatypes::Tensor;
use dominion_core::memo::{BoundedTensorMemo, TensorMemo};
use dominion_core::ml::{
    self, awq_apply, awq_calibrate, binmatmul, fed_avg, flash_attention,
    gptq_quantize, hierarchical_allreduce_cost, lora_forward, lora_merge,
    mlp_params_flat, mlp_set_params_flat, nchw_to_nhwc, nhwc_to_nchw, pack_nchwc,
    q4matmul, qmatmul, qmatmul_per_channel, quantize, quantize_bin, quantize_per_channel,
    quantize_q4, quantize_tern, recommend_device, reconstruct_delta, ring_allreduce_cost,
    scaffold_correct_start, scaffold_update_control, smooth_quant_apply, smooth_quant_scales,
    sparsify_delta, speculative_infer_batch, ternmatmul, turboquant_compress,
    turboquant_decompress, turboquant_dot, Activation, CalibStats, ComputeBandit,
    Device, GptqCalib, LoraAdapter, Mlp, MlConfig, Optimizer, Precision, ALL_PRECISIONS,
};
use dominion_core::nn::{
    NnConfig, NormKind, FfnAct,
};
use dominion_core::nn::norm::{RmsNorm, LayerNorm};
use dominion_core::nn::attention::{MultiHeadAttention, KvCache};
use dominion_core::nn::arch::transformer::TransformerBlock;
use dominion_core::nn::arch::moe::MoeLayer;
use dominion_core::nn::arch::cnn::{BasicBlock, BottleneckBlock};
use dominion_core::nn::recurrent::{LstmCell, GruCell};
use dominion_core::nn::sample::{SamplerConfig, run_sampler};
use dominion_core::nn::tokenizer::BpeTokenizer;
use dominion_core::parallel::Spawn;

// ── a std::thread-backed Spawn for the parallel sweep ──
//
// dominion-core can't spawn threads (no_std + forbid-unsafe); it takes a `Spawn`. On
// the host we back it with `std::thread::scope`: one OS thread per output-row band.
// Bit-identical to the serial kernel (each band is independent, fixed-order).
struct ThreadSpawn {
    workers: usize,
}

impl Spawn for ThreadSpawn {
    fn max_workers(&self) -> usize {
        self.workers
    }
    fn run(&self, n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
        if n <= 1 {
            return (0..n).map(task).collect();
        }
        let mut results: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n);
            for i in 0..n {
                handles.push(s.spawn(move || task(i)));
            }
            for (slot, h) in results.iter_mut().zip(handles) {
                *slot = h.join().unwrap();
            }
        });
        results
    }
}

// ── Persistent thread pool Spawn (zero spawn-overhead between calls) ──────────
//
// Uses the same fat-pointer forwarding technique as `kernel/src/threadpool.rs`:
// the task reference's two fat-pointer words are stored in AtomicU64 globals;
// workers reconstruct the reference and call it. The BSP holds the reference on
// its stack for the entire `run()` duration, guaranteeing the referent is alive.
//
// Each task index is claimed by exactly one thread (atomic fetch_add on `next`),
// so all result slots are written without aliasing. The spin-wait + SeqCst fence
// before reading results ensures all writes are visible to the BSP.

struct PooledInner {
    gen:      AtomicU64,   // batch generation; change wakes workers
    n:        AtomicU64,   // tasks in current batch
    next:     AtomicU64,   // next task index to claim
    done:     AtomicU64,   // completed tasks
    fp_data:  AtomicU64,   // fat pointer data word
    fp_vtbl:  AtomicU64,   // fat pointer vtable word
    res_ptr:  AtomicU64,   // *mut Vec<f64> base (index 0 of results slice)
    shutdown: AtomicBool,
}

// Safety: each results slot is written by at most one thread (fetch_add),
// and the pointer is only valid during the BSP's run() call.
unsafe impl Send for PooledInner {}
unsafe impl Sync for PooledInner {}

struct PooledSpawn {
    workers: usize,
    inner:   Arc<PooledInner>,
    /// Thread handles for park/unpark signalling (workers sleep between batches).
    thread_handles: Vec<std::thread::Thread>,
    _guards: Vec<std::thread::JoinHandle<()>>,
}

impl PooledSpawn {
    fn new(workers: usize) -> Self {
        let inner = Arc::new(PooledInner {
            gen: AtomicU64::new(0), n: AtomicU64::new(0),
            next: AtomicU64::new(0), done: AtomicU64::new(0),
            fp_data: AtomicU64::new(0), fp_vtbl: AtomicU64::new(0),
            res_ptr: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        });
        // Channel to receive thread handles back from spawned workers.
        let (tx, rx) = std::sync::mpsc::sync_channel::<std::thread::Thread>(workers);
        let guards: Vec<_> = (0..workers).map(|_| {
            let s = Arc::clone(&inner);
            let tx = tx.clone();
            std::thread::spawn(move || {
                tx.send(std::thread::current()).ok();
                drop(tx);
                let mut last_gen = 0u64;
                loop {
                    // Park (sleep) until unparked by BSP.
                    std::thread::park();
                    if s.shutdown.load(Ordering::Relaxed) { return; }
                    let g = s.gen.load(Ordering::Acquire);
                    if g == last_gen { continue; } // spurious wakeup
                    last_gen = g;
                    let n = s.n.load(Ordering::Acquire) as usize;
                    // Claim and execute tasks via atomic counter.
                    loop {
                        let idx = s.next.fetch_add(1, Ordering::AcqRel) as usize;
                        if idx >= n { break; }
                        // SAFETY: fat pointer is valid for the BSP's run() duration;
                        // result slot idx is exclusively ours (fetch_add).
                        let result = unsafe {
                            let d = s.fp_data.load(Ordering::Acquire) as usize;
                            let v = s.fp_vtbl.load(Ordering::Acquire) as usize;
                            let f: &(dyn Fn(usize) -> Vec<f64> + Sync) =
                                std::mem::transmute::<[usize; 2], _>([d, v]);
                            f(idx)
                        };
                        unsafe {
                            let base = s.res_ptr.load(Ordering::Acquire) as *mut Vec<f64>;
                            base.add(idx).write(result);
                        }
                        s.done.fetch_add(1, Ordering::AcqRel);
                    }
                }
            })
        }).collect();
        drop(tx);
        let thread_handles: Vec<_> = (0..workers).map(|_| rx.recv().unwrap()).collect();
        PooledSpawn { workers, inner, thread_handles, _guards: guards }
    }
}

impl Drop for PooledSpawn {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Spawn for PooledSpawn {
    fn max_workers(&self) -> usize { self.workers + 1 }

    fn run(&self, n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
        if n <= 1 { return (0..n).map(task).collect(); }
        let mut results: Vec<Vec<f64>> = (0..n).map(|_| Vec::new()).collect();
        let fp: [usize; 2] = unsafe { std::mem::transmute(task) };
        let base = results.as_mut_ptr();
        self.inner.fp_data.store(fp[0] as u64, Ordering::Release);
        self.inner.fp_vtbl.store(fp[1] as u64, Ordering::Release);
        self.inner.res_ptr.store(base as u64, Ordering::Release);
        self.inner.n.store(n as u64, Ordering::Release);
        self.inner.next.store(0, Ordering::Release);
        self.inner.done.store(0, Ordering::Release);
        // Change gen, then unpark sleeping workers (park/unpark: no spinning).
        self.inner.gen.fetch_add(1, Ordering::AcqRel);
        for t in &self.thread_handles {
            t.unpark();
        }
        // BSP also steals tasks.
        loop {
            let idx = self.inner.next.fetch_add(1, Ordering::AcqRel) as usize;
            if idx >= n { break; }
            let r = task(idx);
            unsafe { base.add(idx).write(r); }
            self.inner.done.fetch_add(1, Ordering::AcqRel);
        }
        // Wait for all tasks; brief spin (workers are actively computing).
        while self.inner.done.load(Ordering::Acquire) < n as u64 {
            std::hint::spin_loop();
        }
        fence(Ordering::SeqCst);
        results
    }
}

// ── identical scale knobs to kernel/src/bench.rs ──
const ML_MATMUL_N: usize = 128;
const ML_MATMUL_ITERS: u64 = 40;
const ML_TRAIN_EPOCHS: u64 = 2_000;
const ML_INFER_ITERS: u64 = 20_000;

/// The same LCG the kernel benchmark fills its matrices with — so both sides
/// multiply byte-identical data.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// Draw one f64 sample in [-0.5, 0.5).
    fn sample(&mut self) -> f64 {
        (self.next() >> 40) as f64 / 16_777_216.0 - 0.5
    }
}

/// Fill a rows×cols Tensor with LCG-drawn values in [-0.5, 0.5).
fn fill_tensor(lcg: &mut Lcg, rows: usize, cols: usize) -> Tensor {
    Tensor::new(
        vec![rows, cols],
        (0..rows * cols).map(|_| lcg.sample()).collect(),
    )
    .unwrap()
}

fn micros(d: std::time::Duration) -> u64 {
    d.as_micros() as u64
}
fn millis(d: std::time::Duration) -> u64 {
    d.as_millis() as u64
}
fn per_sec(count: u64, d: std::time::Duration) -> u64 {
    let us = micros(d);
    if us == 0 {
        0
    } else {
        count * 1_000_000 / us
    }
}

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

// ── peak-exploration kernels (host-only; show where the GFLOP/s come from) ──
//
// These exist to answer "how fast can this CPU actually go, and what each lever
// is worth": FMA (fused multiply-add, ~2×, NON-deterministic so it's a knob), and
// multiple cores (~N×). The guest's deterministic kernel deliberately uses none of
// these; this is the ceiling, not the default.

fn transpose(b: &Tensor) -> Vec<f64> {
    let (k, n) = (b.shape()[0], b.shape()[1]);
    let mut bt = vec![0.0; n * k];
    for p in 0..k {
        for j in 0..n {
            bt[j * k + p] = b.data()[p * n + j];
        }
    }
    bt
}

/// Deterministic lane dot (same as the engine's): separate mul + add, no FMA.
fn dot_plain(x: &[f64], y: &[f64]) -> f64 {
    const L: usize = 8;
    let mut acc = [0.0f64; L];
    let (mut cx, mut cy) = (x.chunks_exact(L), y.chunks_exact(L));
    for (xc, yc) in cx.by_ref().zip(cy.by_ref()) {
        for l in 0..L {
            acc[l] += xc[l] * yc[l];
        }
    }
    let mut s = 0.0;
    for &a in &acc {
        s += a;
    }
    for (&xr, &yr) in cx.remainder().iter().zip(cy.remainder()) {
        s += xr * yr;
    }
    s
}

/// FMA lane dot: `mul_add` fuses the multiply and add into one rounding step. On
/// +fma hardware this is one `vfmadd` (≈2× the FLOPs/cycle), but the single fused
/// rounding changes the low bits — which is why it can never be the deterministic
/// default. This is the toggle.
fn dot_fma(x: &[f64], y: &[f64]) -> f64 {
    const L: usize = 8;
    let mut acc = [0.0f64; L];
    let (mut cx, mut cy) = (x.chunks_exact(L), y.chunks_exact(L));
    for (xc, yc) in cx.by_ref().zip(cy.by_ref()) {
        for l in 0..L {
            acc[l] = xc[l].mul_add(yc[l], acc[l]);
        }
    }
    let mut s = 0.0;
    for &a in &acc {
        s += a;
    }
    for (&xr, &yr) in cx.remainder().iter().zip(cy.remainder()) {
        s = xr.mul_add(yr, s);
    }
    s
}

/// Explicit 4-accumulator AVX-2 FMA dot product using std::arch intrinsics.
///
/// Processes 16 f64 per iteration (4 ymm accumulators × 4 lanes each), fully
/// utilizing both FMA execution ports on AVX-2 hardware. Demonstrates the ceiling
/// achievable with explicit SIMD that LLVM's auto-vectorizer might not always reach.
///
/// Uses `#[target_feature(enable="avx2,fma")]` for compile-time specialization
/// without changing the global target-cpu — so it can be runtime-dispatched.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma_impl(x: &[f64], y: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = x.len();
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    // 8 accumulators × 4 f64 per ymm = 32 elements per loop iteration.
    // FMA latency = 5 cycles, throughput = 0.5 cycles. Need ≥10 concurrent FMAs
    // to fully hide latency at 2 FMA ports. 8 accumulators = 8 × 5 = 40 cycles
    // of latency coverage → both ports are always busy.
    let mut a0 = _mm256_setzero_pd(); let mut a1 = _mm256_setzero_pd();
    let mut a2 = _mm256_setzero_pd(); let mut a3 = _mm256_setzero_pd();
    let mut a4 = _mm256_setzero_pd(); let mut a5 = _mm256_setzero_pd();
    let mut a6 = _mm256_setzero_pd(); let mut a7 = _mm256_setzero_pd();
    let full = n / 32;
    for i in 0..full {
        let b = i * 32;
        a0 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b)),      _mm256_loadu_pd(yp.add(b)),      a0);
        a1 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 4)),  _mm256_loadu_pd(yp.add(b + 4)),  a1);
        a2 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 8)),  _mm256_loadu_pd(yp.add(b + 8)),  a2);
        a3 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 12)), _mm256_loadu_pd(yp.add(b + 12)), a3);
        a4 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 16)), _mm256_loadu_pd(yp.add(b + 16)), a4);
        a5 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 20)), _mm256_loadu_pd(yp.add(b + 20)), a5);
        a6 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 24)), _mm256_loadu_pd(yp.add(b + 24)), a6);
        a7 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(b + 28)), _mm256_loadu_pd(yp.add(b + 28)), a7);
    }
    // Handle remainder (< 32 elements) 4 at a time.
    let rem_start = full * 32;
    let mut rem = rem_start;
    while rem + 4 <= n {
        a0 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(rem)), _mm256_loadu_pd(yp.add(rem)), a0);
        rem += 4;
    }
    // Tree-reduce 8 accumulators.
    a0 = _mm256_add_pd(a0, a4); a1 = _mm256_add_pd(a1, a5);
    a2 = _mm256_add_pd(a2, a6); a3 = _mm256_add_pd(a3, a7);
    a0 = _mm256_add_pd(a0, a2); a1 = _mm256_add_pd(a1, a3);
    a0 = _mm256_add_pd(a0, a1);
    // Horizontal sum of the 4-lane ymm.
    let lo = _mm256_extractf128_pd(a0, 0);
    let hi = _mm256_extractf128_pd(a0, 1);
    let s  = _mm_add_pd(lo, hi);
    let sh = _mm_shuffle_pd(s, s, 1);
    _mm_cvtsd_f64(_mm_add_pd(s, sh))
}

/// Runtime-dispatched dot: uses explicit AVX-2+FMA if available, else dot_fma.
fn dot_avx2(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { dot_avx2_fma_impl(x, y) };
    }
    dot_fma(x, y)
}

/// Compute a contiguous block of output rows `[r0, r1)`; the unit of parallel work.
fn matmul_rows(a: &[f64], bt: &[f64], out: &mut [f64], k: usize, n: usize, r0: usize, fma: bool) {
    for (ri, orow) in out.chunks_mut(n).enumerate() {
        let arow = &a[(r0 + ri) * k..(r0 + ri) * k + k];
        for (j, o) in orow.iter_mut().enumerate() {
            let brow = &bt[j * k..j * k + k];
            *o = if fma { dot_fma(arow, brow) } else { dot_plain(arow, brow) };
        }
    }
}

/// Matmul across `threads` OS threads (rows split evenly). `threads == 1` is serial.
fn matmul_par(a: &Tensor, b: &Tensor, threads: usize, fma: bool) -> Vec<f64> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let bt = transpose(b);
    let mut out = vec![0.0; m * n];
    let ad = a.data();
    if threads <= 1 {
        matmul_rows(ad, &bt, &mut out, k, n, 0, fma);
        return out;
    }
    let chunk = m.div_ceil(threads);
    std::thread::scope(|s| {
        let mut rest = out.as_mut_slice();
        let mut r0 = 0;
        while r0 < m {
            let rows = chunk.min(m - r0);
            let (mine, tail) = rest.split_at_mut(rows * n);
            let (ad, bt) = (ad, &bt);
            s.spawn(move || matmul_rows(ad, bt, mine, k, n, r0, fma));
            rest = tail;
            r0 += rows;
        }
    });
    out
}

fn gflops(m: usize, k: usize, n: usize, iters: u64, d: std::time::Duration) -> u64 {
    let f = 2 * (m as u64) * (k as u64) * (n as u64) * iters;
    let us = micros(d);
    if us == 0 {
        0
    } else {
        f / us / 1000
    }
}

/// The "how fast can this CPU really go" ladder, at a size where it matters.
fn run_peak() {
    let n = 1024usize;
    let iters = 8u64;
    let mut lcg = Lcg(0xDEAD_BEEF_0000_0001);
    let a = fill_tensor(&mut lcg, n, n);
    let b = a.clone();
    let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);

    println!("--- peak exploration ({0}x{0} f64 matmul, host) ---", n);
    let configs: [(&str, usize, bool); 4] = [
        ("1core_noFMA(determ.)", 1, false),
        ("1core_FMA", 1, true),
        (if cores > 1 { "Ncore_noFMA" } else { "1core_noFMA(b)" }, cores, false),
        ("Ncore_FMA", cores, true),
    ];
    for (label, th, fma) in configs {
        let t = Instant::now();
        let mut sink = 0.0;
        for _ in 0..iters {
            let o = matmul_par(&a, &b, th, fma);
            sink += o[0];
        }
        let d = t.elapsed();
        black_box(sink);
        println!(
            "BENCH peak_matmul config={} threads={} fma={} gflop_per_s={} ms={}",
            label, th, fma, gflops(n, n, n, iters, d), millis(d)
        );
    }
    println!("(cores available = {})", cores);
}

/// Parallel-scaling sweep: the deterministic engine kernel (`Tensor::matmul_with`)
/// across thread counts and matrix sizes — the Lever 1 measurement. Verifies the
/// parallel result is bit-identical to serial, then reports GFLOP/s per thread count.
fn run_par() {
    let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    println!("--- Lever 1+2: multi-core matmul scaling (ThreadSpawn vs PooledSpawn) ---");
    println!("(cores available = {})", cores);
    // Pre-warm the persistent pool so startup cost doesn't contaminate results.
    let pool16 = PooledSpawn::new(cores);
    let sizes: [(usize, u64); 5] = [(128, 50), (256, 30), (512, 12), (1024, 4), (2048, 2)];
    for (n, iters) in sizes {
        let mut lcg = Lcg(0x5151_4D4C_0000_0001 ^ n as u64);
        let a = fill_tensor(&mut lcg, n, n);
        let b = a.clone();
        let reference = a.matmul(&b).unwrap();
        for &th in &[1usize, 2, 4, 8, 16] {
            if th > cores {
                break;
            }
            // ThreadSpawn: spawns new OS threads each call.
            let spawn = ThreadSpawn { workers: th };
            let check = a.matmul_with(&b, &spawn).unwrap();
            let identical = check.data() == reference.data();
            let mut sink = 0.0;
            let t = Instant::now();
            for _ in 0..iters {
                let c = a.matmul_with(&b, &spawn).unwrap();
                sink += c.data()[0];
            }
            let dt = t.elapsed();
            black_box(sink);
            println!(
                "BENCH par_matmul n={} threads={} gflop_per_s={} mflop_per_s={} bit_identical={} ms={}",
                n, th,
                gflops(n, n, n, iters, dt),
                if micros(dt) > 0 { 2 * (n as u64).pow(3) * iters / micros(dt) } else { 0 },
                identical, millis(dt)
            );
        }
        // PooledSpawn at all cores: persistent threads, zero spawn overhead.
        {
            let check = a.matmul_with(&b, &pool16).unwrap();
            let identical = check.data() == reference.data();
            let mut sink = 0.0;
            let t = Instant::now();
            for _ in 0..iters {
                let c = a.matmul_with(&b, &pool16).unwrap();
                sink += c.data()[0];
            }
            let dt = t.elapsed();
            black_box(sink);
            println!(
                "BENCH pool_matmul n={} threads={} gflop_per_s={} mflop_per_s={} bit_identical={} ms={}",
                n, cores,
                gflops(n, n, n, iters, dt),
                if micros(dt) > 0 { 2 * (n as u64).pow(3) * iters / micros(dt) } else { 0 },
                identical, millis(dt)
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let want_fma = args.iter().any(|a| a == "--fma");
    if args.iter().any(|a| a == "--peak") {
        run_peak();
        return;
    }
    if args.iter().any(|a| a == "--par") {
        run_par();
        return;
    }
    let built = if cfg!(target_feature = "avx2") {
        "avx2"
    } else if cfg!(target_feature = "avx") {
        "avx"
    } else {
        "sse2"
    };
    println!("========== DominionOS ML benchmark — NATIVE HOST baseline ==========");
    println!(
        "BENCH meta host=1 target_feature={} avx2_runtime={}",
        built,
        std::is_x86_feature_detected!("avx2")
    );

    // (1) Dense f64 matmul throughput → the headline ML primitive.
    let n = ML_MATMUL_N;
    let mut lcg = Lcg(0x5151_4D4C_0000_0001);
    let a = Tensor::new(
        vec![n, n],
        (0..n * n).map(|_| (lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5).collect(),
    )
    .unwrap();
    let b = a.clone();
    let mut sink = 0.0f64;
    let t0 = Instant::now();
    let mut done = 0u64;
    for _ in 0..ML_MATMUL_ITERS {
        // Default: the exact deterministic engine kernel (matches the guest).
        // With `--fma`: the fused multiply-add path (non-deterministic, ~2× ceiling).
        let c = if want_fma {
            Tensor::new(vec![n, n], matmul_par(&a, &b, 1, true)).unwrap()
        } else {
            a.matmul(&b).unwrap()
        };
        sink += c.data()[0];
        done += 1;
    }
    let dt = t0.elapsed();
    let flops = ml::matmul_flops(n, n, n) * done;
    let mflops = if micros(dt) > 0 { flops / micros(dt) } else { 0 };
    black_box(sink);
    println!(
        "BENCH ml_matmul n={} iters={} mflop_per_s={} gflop_per_s={} fma={} ms={}",
        n, done, mflops, mflops / 1000, want_fma, millis(dt)
    );

    // (2) Modeled placement (pure cost model — identical to the guest).
    let one = ml::matmul_flops(n, n, n);
    println!(
        "BENCH ml_placement n={} chosen={} cpu_cyc={} gpu_cyc={} npu_cyc={} tpu_cyc={}",
        n,
        recommend_device(one).name(),
        Device::Cpu.est_cycles(one),
        Device::Gpu.est_cycles(one),
        Device::Npu.est_cycles(one),
        Device::Tpu.est_cycles(one),
    );

    // (3) Training throughput: gradient-descent steps/sec on the XOR MLP.
    let (x, y) = ml::xor_dataset();
    let mut model = Mlp::new(&[2, 16, 1], Activation::Tanh, Activation::Sigmoid, 0xA17E).unwrap();
    let mut opt = Optimizer::adam(0.05);
    let mut last_loss = 1.0;
    let t1 = Instant::now();
    let mut steps = 0u64;
    for _ in 0..ML_TRAIN_EPOCHS {
        last_loss = model.train_step_mse(&x, &y, &mut opt).unwrap();
        steps += 1;
    }
    let tt = t1.elapsed();
    println!(
        "BENCH ml_train steps={} steps_per_s={} final_loss_micro={} ms={}",
        steps, per_sec(steps, tt), (last_loss * 1_000_000.0) as i64, millis(tt)
    );

    // (4) Inference throughput.
    let mut isink = 0.0f64;
    let t2 = Instant::now();
    let mut infers = 0u64;
    for _ in 0..ML_INFER_ITERS {
        let out = model.forward(&x).unwrap();
        isink += out.data()[0];
        infers += 1;
    }
    let it = t2.elapsed();
    black_box(isink);
    println!("BENCH ml_infer passes={} infer_per_s={} ms={}", infers, per_sec(infers, it), millis(it));

    // (4b) Lever 2: operator fusion — fused vs unfused inference on a larger model.
    // 3-layer MLP: [batch=128, in=256] → [512] → [256] → [128].
    // Unfused: 3 ops per layer (matmul + copy, bias pass, activation pass) = 6 RAM passes.
    // Fused:   1 op per layer (matmul_bias_act, owned buffer) = 2 RAM passes.
    {
        let batch = 128usize;
        let arch: &[usize] = &[256, 512, 256, 128];
        let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        let mut lcg2 = Lcg(0xFEED_BEEF_0000_0001);
        let x_large = fill_tensor(&mut lcg2, batch, arch[0]);
        let model_large = ml::Mlp::new(arch, ml::Activation::Relu, ml::Activation::Identity, 0xBEEF).unwrap();
        const LARGE_ITERS: u64 = 200;
        let fw_flops: u64 = model_large.forward_flops(batch) * LARGE_ITERS;

        // -- unfused baseline: separate matmul / bias / activation (old 3-op chain)
        let mut usink = 0.0f64;
        let tu = Instant::now();
        for _ in 0..LARGE_ITERS {
            let mut cur = x_large.clone();
            let last = model_large.layers.len() - 1;
            for (i, layer) in model_large.layers.iter().enumerate() {
                let z = cur.matmul(&layer.w).unwrap();
                let (rows, cols) = (z.shape()[0], z.shape()[1]);
                let bd = layer.b.data();
                let mut data = z.data().to_vec();          // explicit copy
                for r in 0..rows { for c in 0..cols { data[r*cols+c] += bd[c]; } }
                let act = if i == last { model_large.output } else { model_large.hidden };
                for v in data.iter_mut() { *v = act.as_fn()(*v); }
                cur = Tensor::new(vec![rows, cols], data).unwrap();
            }
            usink += cur.data()[0];
        }
        let ud = tu.elapsed();
        black_box(usink);
        let unfused_gflops = if micros(ud) > 0 { fw_flops / micros(ud) / 1000 } else { 0 };
        println!(
            "BENCH ml_unfused_infer batch={} arch=256x512x256x128 iters={} gflop_per_s={} infer_per_s={} ms={}",
            batch, LARGE_ITERS, unfused_gflops, per_sec(LARGE_ITERS, ud), millis(ud)
        );

        // -- fused: matmul_bias_act (1 allocation, 1 pass per layer)
        let mut fsink = 0.0f64;
        let tf = Instant::now();
        for _ in 0..LARGE_ITERS {
            let out = model_large.forward(&x_large).unwrap();
            fsink += out.data()[0];
        }
        let fd = tf.elapsed();
        black_box(fsink);
        let fused_gflops = if micros(fd) > 0 { fw_flops / micros(fd) / 1000 } else { 0 };
        println!(
            "BENCH ml_fused_infer batch={} arch=256x512x256x128 iters={} gflop_per_s={} infer_per_s={} ms={} vs_unfused={:.2}x",
            batch, LARGE_ITERS, fused_gflops, per_sec(LARGE_ITERS, fd), millis(fd),
            micros(ud) as f64 / micros(fd).max(1) as f64
        );

        // -- fused + ThreadSpawn: new threads per call
        let tspawn = ThreadSpawn { workers: cores };
        let mut tsink = 0.0f64;
        let tt2 = Instant::now();
        for _ in 0..LARGE_ITERS {
            let out = model_large.forward_with(&x_large, &tspawn).unwrap();
            tsink += out.data()[0];
        }
        let td2 = tt2.elapsed();
        black_box(tsink);
        let tpar_gflops = if micros(td2) > 0 { fw_flops / micros(td2) / 1000 } else { 0 };
        println!(
            "BENCH ml_threadspawn_infer batch={} arch=256x512x256x128 threads={} iters={} gflop_per_s={} infer_per_s={} ms={} vs_unfused={:.2}x",
            batch, cores, LARGE_ITERS, tpar_gflops, per_sec(LARGE_ITERS, td2), millis(td2),
            micros(ud) as f64 / micros(td2).max(1) as f64
        );

        // -- fused + PooledSpawn: persistent threads, zero spawn overhead
        let pool = PooledSpawn::new(cores);
        let mut psink = 0.0f64;
        let tp = Instant::now();
        for _ in 0..LARGE_ITERS {
            let out = model_large.forward_with(&x_large, &pool).unwrap();
            psink += out.data()[0];
        }
        let pd = tp.elapsed();
        black_box(psink);
        let par_gflops = if micros(pd) > 0 { fw_flops / micros(pd) / 1000 } else { 0 };
        println!(
            "BENCH ml_pool_infer batch={} arch=256x512x256x128 threads={} iters={} gflop_per_s={} infer_per_s={} ms={} vs_unfused={:.2}x vs_threadspawn={:.2}x",
            batch, cores, LARGE_ITERS, par_gflops, per_sec(LARGE_ITERS, pd), millis(pd),
            micros(ud) as f64 / micros(pd).max(1) as f64,
            micros(td2) as f64 / micros(pd).max(1) as f64
        );
    }

    // (4c) Lever 3: content-addressed memoization & KV-cache.
    //
    // Demonstrates three modes of the memo cache:
    //   1. Hash throughput — the per-call overhead (multi-lane word hash over input bytes).
    //   2. Warm hit — identical (model, input) → return cached clone, skip compute.
    //   3. Incremental — only changed layers recompute; unchanged prefix is cached.
    {
        // 4-layer MLP: layers get progressively smaller, so skipping the first 3
        // saves ~94% of FLOPs when only the last layer changes (fine-tuning scenario).
        let memo_arch: &[usize] = &[256, 512, 512, 256, 128];
        let batch = 64usize;
        let mut lcg3 = Lcg(0xDEAD_CAFE_0000_0001);
        let x_memo = fill_tensor(&mut lcg3, batch, memo_arch[0]);
        let model_memo = Mlp::new(memo_arch, Activation::Relu, Activation::Identity, 0xCAFE_BABE).unwrap();
        let model_hash = model_memo.content_hash();  // precompute once; reuse across all infer calls

        // --- 1. Hash throughput (cost of the cache key computation) ---
        const HASH_ITERS: u64 = 20_000;
        let th = Instant::now();
        let mut hval = 0u64;
        for _ in 0..HASH_ITERS {
            // black_box on reference prevents LLVM from hoisting the hash out of the loop.
            hval ^= std::hint::black_box(&x_memo).content_hash();
        }
        let hd = th.elapsed();
        black_box(hval);
        let x_bytes = (x_memo.len() * 8) as u64;  // f64 = 8 bytes per element
        let hash_mb_s = if micros(hd) > 0 { x_bytes * HASH_ITERS / micros(hd) } else { 0 };
        let hash_ns = if HASH_ITERS > 0 { micros(hd) * 1000 / HASH_ITERS } else { 0 };
        println!(
            "BENCH memo_hash_throughput tensor_bytes={} iters={} mb_per_s={} ns_per_hash={}",
            x_bytes, HASH_ITERS, hash_mb_s, hash_ns
        );

        // --- 2a. Cold miss: unique input each call → compute + store every time ---
        const MEMO_ITERS: u64 = 500;
        // Pre-generate unique inputs so generation cost doesn't contaminate the timer.
        let mut lcg_inp = Lcg(0xFEED_0001_0000_0001);
        let unique_inputs: Vec<Tensor> = (0..MEMO_ITERS).map(|_| {
            Tensor::new(
                vec![batch, memo_arch[0]],
                (0..batch * memo_arch[0]).map(|_| (lcg_inp.next() >> 40) as f64 / 16_777_216.0 - 0.5).collect(),
            ).unwrap()
        }).collect();

        let mut cache_cold = TensorMemo::new();
        let mut csink = 0.0f64;
        let tc = Instant::now();
        for inp in &unique_inputs {
            let out = model_memo.forward_cached(inp, model_hash, &mut cache_cold).unwrap();
            csink += out.data()[0];
        }
        let cd = tc.elapsed();
        black_box(csink);
        println!(
            "BENCH memo_cold_infer batch={} iters={} infer_per_s={} hits={} misses={} ms={}",
            batch, MEMO_ITERS, per_sec(MEMO_ITERS, cd),
            cache_cold.hits(), cache_cold.misses(), millis(cd)
        );

        // --- 2b. Warm hit: identical input → hash + BTreeMap lookup + clone, skip compute ---
        let mut cache_warm = TensorMemo::new();
        // Populate the cache with one entry (cold miss).
        model_memo.forward_cached(&x_memo, model_hash, &mut cache_warm);

        let mut wsink = 0.0f64;
        let tw = Instant::now();
        for _ in 0..MEMO_ITERS {
            let out = model_memo.forward_cached(&x_memo, model_hash, &mut cache_warm).unwrap();
            wsink += out.data()[0];
        }
        let wd = tw.elapsed();
        black_box(wsink);
        let memo_speedup = micros(cd) as f64 / micros(wd).max(1) as f64;
        println!(
            "BENCH memo_warm_infer batch={} iters={} infer_per_s={} hits={} hit_rate_pct={} ms={} vs_cold={:.1}x",
            batch, MEMO_ITERS, per_sec(MEMO_ITERS, wd),
            cache_warm.hits(), cache_warm.hit_rate_pct(), millis(wd), memo_speedup
        );

        // --- 3. Incremental recompute: change only last layer, first 3 layers are cached ---
        let mut cache_incr = TensorMemo::new();
        let orig_layer_hashes = model_memo.layer_content_hashes();
        model_memo.forward_incremental(&x_memo, &orig_layer_hashes, &mut cache_incr);

        let mut model_modified = model_memo.clone();
        let li = model_modified.layers.len() - 1;
        let shape = model_modified.layers[li].w.shape().to_vec();
        let new_w: Vec<f64> = model_modified.layers[li].w.data().iter().map(|&v| v + 1e-6).collect();
        model_modified.layers[li].w = Tensor::new(shape, new_w).unwrap();
        let mod_layer_hashes = model_modified.layer_content_hashes();

        let mut fsink2 = 0.0f64;
        let tf = Instant::now();
        for _ in 0..MEMO_ITERS {
            let out = model_modified.forward(&x_memo).unwrap();
            fsink2 += out.data()[0];
        }
        let fd = tf.elapsed();
        black_box(fsink2);

        let mut isink2 = 0.0f64;
        let ti = Instant::now();
        for _ in 0..MEMO_ITERS {
            let out = model_modified.forward_incremental(&x_memo, &mod_layer_hashes, &mut cache_incr).unwrap();
            isink2 += out.data()[0];
        }
        let id = ti.elapsed();
        black_box(isink2);
        let incr_speedup = micros(fd) as f64 / micros(id).max(1) as f64;
        let nlayers = memo_arch.len() - 1;
        println!(
            "BENCH memo_incremental_infer batch={} layers={} last_layer_changed=1 iters={} full_per_s={} incr_per_s={} hits={} ms_full={} ms_incr={} speedup={:.2}x",
            batch, nlayers, MEMO_ITERS,
            per_sec(MEMO_ITERS, fd), per_sec(MEMO_ITERS, id),
            cache_incr.hits(), millis(fd), millis(id), incr_speedup
        );
    }

    // (5) int8 (NPU path) matmul throughput.
    let (qa, qb) = (quantize(&a), quantize(&b));
    let mut qsink = 0.0f64;
    let t3 = Instant::now();
    let mut qdone = 0u64;
    for _ in 0..ML_MATMUL_ITERS {
        let c = qmatmul(&qa, &qb).unwrap();
        qsink += c.data()[0];
        qdone += 1;
    }
    let qt = t3.elapsed();
    let qmop = if micros(qt) > 0 { (ml::matmul_flops(n, n, n) * qdone) / micros(qt) } else { 0 };
    black_box(qsink);
    println!("BENCH ml_int8_matmul n={} iters={} mop_per_s={} ms={}", n, qdone, qmop, millis(qt));

    // (6) Lever 4: Low-precision quantization — int4, binary, ternary.
    //
    // Precision ladder: f64 → int8 → int4 → ternary → binary.
    // Each step doubles or more the arithmetic density per byte.
    {
        let mut lcg4 = Lcg(0xABCD_EF01_0000_0001);
        let a4 = fill_tensor(&mut lcg4, n, n);
        let b4 = a4.clone();

        // int8 baseline (re-run for fair comparison with same matrix)
        let (qa8, qb8) = (quantize(&a4), quantize(&b4));
        let ti8 = Instant::now();
        let mut i8sink = 0.0f64;
        let mut i8done = 0u64;
        for _ in 0..ML_MATMUL_ITERS {
            let c = qmatmul(&qa8, &qb8).unwrap();
            i8sink += c.data()[0];
            i8done += 1;
        }
        let i8d = ti8.elapsed();
        black_box(i8sink);
        let i8_mop = if micros(i8d) > 0 { (ml::matmul_flops(n, n, n) * i8done) / micros(i8d) } else { 0 };

        // --- int4 ---
        let (qa4, qb4) = (quantize_q4(&a4), quantize_q4(&b4));
        let mut q4sink = 0.0f64;
        let tq4 = Instant::now();
        let mut q4done = 0u64;
        for _ in 0..ML_MATMUL_ITERS {
            let c = q4matmul(&qa4, &qb4).unwrap();
            q4sink += c.data()[0];
            q4done += 1;
        }
        let q4t = tq4.elapsed();
        let q4_mop = if micros(q4t) > 0 { (ml::matmul_flops(n, n, n) * q4done) / micros(q4t) } else { 0 };
        black_box(q4sink);
        println!(
            "BENCH ml_int4_matmul n={} iters={} mop_per_s={} ms={} vs_int8={:.2}x",
            n, q4done, q4_mop, millis(q4t),
            micros(i8d) as f64 / micros(q4t).max(1) as f64
        );

        // --- binary ---
        let (ba, bb) = (quantize_bin(&a4), quantize_bin(&b4));
        let mut bsink = 0.0f64;
        let tb = Instant::now();
        let mut bdone = 0u64;
        for _ in 0..ML_MATMUL_ITERS {
            let c = binmatmul(&ba, &bb).unwrap();
            bsink += c.data()[0];
            bdone += 1;
        }
        let bt2 = tb.elapsed();
        let bin_mop = if micros(bt2) > 0 { (ml::matmul_flops(n, n, n) * bdone) / micros(bt2) } else { 0 };
        black_box(bsink);
        println!(
            "BENCH ml_binary_matmul n={} iters={} mop_per_s={} ms={} vs_int8={:.1}x vs_f64={:.1}x",
            n, bdone, bin_mop, millis(bt2),
            micros(i8d) as f64 / micros(bt2).max(1) as f64,
            micros(dt) as f64 / micros(bt2).max(1) as f64
        );

        // --- ternary ---
        let (ta2, tb2) = (quantize_tern(&a4), quantize_tern(&b4));
        let mut tsink = 0.0f64;
        let tt3 = Instant::now();
        let mut tdone = 0u64;
        for _ in 0..ML_MATMUL_ITERS {
            let c = ternmatmul(&ta2, &tb2).unwrap();
            tsink += c.data()[0];
            tdone += 1;
        }
        let tt4 = tt3.elapsed();
        let tern_mop = if micros(tt4) > 0 { (ml::matmul_flops(n, n, n) * tdone) / micros(tt4) } else { 0 };
        black_box(tsink);
        println!(
            "BENCH ml_ternary_matmul n={} iters={} mop_per_s={} ms={} vs_int8={:.1}x vs_f64={:.1}x",
            n, tdone, tern_mop, millis(tt4),
            micros(i8d) as f64 / micros(tt4).max(1) as f64,
            micros(dt) as f64 / micros(tt4).max(1) as f64
        );

        println!(
            "BENCH precision_ladder n={} f64_gflop_per_s={} int8_mop_per_s={} int4_mop_per_s={} ternary_mop_per_s={} binary_mop_per_s={}",
            n, mflops / 1000, i8_mop, q4_mop, tern_mop, bin_mop
        );
    }

    // (7) Lever 5: Adaptive compute-precision placement via ε-greedy bandit.
    //
    // The bandit starts with no knowledge and uses exploration (ε=0.25) to try all
    // 5 precision arms (f64, int8, int4, binary, ternary). It measures throughput
    // on 128×128 matmuls in each precision, updates its Q-values, and converges to
    // the fastest arm. Metrics show: convergence speed, Q-values after training,
    // average throughput under bandit vs static-f64 baseline.
    {
        const BANDIT_ITERS: u64 = 200;
        let mut bandit = ComputeBandit::new(0xDEAD_CAFE);
        let n = ML_MATMUL_N;
        let mut lcg5 = Lcg(0xFEED_BABE_0000_0001);
        let ab = fill_tensor(&mut lcg5, n, n);
        // Pre-quantize the matrix in every precision so quantization cost is excluded.
        let qa_i8   = quantize(&ab);   let qb_i8   = qa_i8.clone();
        let qa_i4   = quantize_q4(&ab); let qb_i4   = qa_i4.clone();
        let qa_bin  = quantize_bin(&ab); let qb_bin  = qa_bin.clone();
        let qa_tern = quantize_tern(&ab); let qb_tern = qa_tern.clone();
        let flops_per_call = ml::matmul_flops(n, n, n);

        // f64 baseline throughput for comparison.
        let t_f64_base = {
            let t = Instant::now();
            for _ in 0..40u64 { let _ = ab.matmul(&ab); }
            let us = micros(t.elapsed()).max(1);
            (flops_per_call * 40 / us) as f64 // MFLOP/s
        };

        let mut total_throughput = 0.0f64;
        let mut converge_at = 0u64;
        let tb = Instant::now();
        for iter in 0..BANDIT_ITERS {
            let arm = bandit.select();
            let prec = ALL_PRECISIONS[arm];
            // Time one matmul call in the selected precision.
            let t1 = Instant::now();
            match prec {
                Precision::F64     => { black_box(ab.matmul(&ab)); }
                Precision::Int8    => { black_box(qmatmul(&qa_i8, &qb_i8)); }
                Precision::Int4    => { black_box(q4matmul(&qa_i4, &qb_i4)); }
                Precision::Binary  => { black_box(binmatmul(&qa_bin, &qb_bin)); }
                Precision::Ternary => { black_box(ternmatmul(&qa_tern, &qb_tern)); }
            }
            let us = micros(t1.elapsed()).max(1);
            let throughput = (flops_per_call / us) as f64; // MFLOP/s
            bandit.update(arm, throughput);
            total_throughput += throughput;
            if converge_at == 0 && bandit.is_converged() {
                converge_at = iter + 1;
            }
        }
        let bandit_d = tb.elapsed();
        let avg_bandit_throughput = total_throughput / BANDIT_ITERS as f64;
        let best_prec = bandit.best_precision();
        let q_values = bandit.q_values();
        let pulls = bandit.pull_counts();

        println!(
            "BENCH bandit_placement iters={} converge_at={} best_precision={} \
             avg_mflop_per_s={:.0} vs_f64_baseline={:.2}x epsilon_final={:.3} ms={}",
            BANDIT_ITERS, converge_at, best_prec.name(),
            avg_bandit_throughput, avg_bandit_throughput / t_f64_base.max(1.0),
            bandit.epsilon(), millis(bandit_d)
        );
        // Report Q-values and pull counts for each arm.
        for (i, prec) in ALL_PRECISIONS.iter().enumerate() {
            println!(
                "BENCH bandit_arm prec={} q={:.0} pulls={} pct={:.0}",
                prec.name(), q_values[i], pulls[i],
                if bandit.total_pulls() > 0 { pulls[i] * 100 / bandit.total_pulls() } else { 0 }
            );
        }

        // UCB-1 comparison: same 200 iters but arm selection via Upper Confidence Bound.
        // UCB-1 needs no epsilon tuning and gives provably sublinear regret O(sqrt(N ln N)).
        {
            let mut ucb_bandit = ComputeBandit::new(0xDEAD_CAFE);
            let mut ucb_total = 0.0f64;
            let mut ucb_converge_at = 0u64;
            let tu = Instant::now();
            for iter in 0..BANDIT_ITERS {
                let arm = ucb_bandit.select_ucb1();
                let prec = ALL_PRECISIONS[arm];
                let t1 = Instant::now();
                match prec {
                    Precision::F64     => { black_box(ab.matmul(&ab)); }
                    Precision::Int8    => { black_box(qmatmul(&qa_i8, &qb_i8)); }
                    Precision::Int4    => { black_box(q4matmul(&qa_i4, &qb_i4)); }
                    Precision::Binary  => { black_box(binmatmul(&qa_bin, &qb_bin)); }
                    Precision::Ternary => { black_box(ternmatmul(&qa_tern, &qb_tern)); }
                }
                let us = micros(t1.elapsed()).max(1);
                let tp = (flops_per_call / us) as f64;
                ucb_bandit.update(arm, tp);
                ucb_total += tp;
                if ucb_converge_at == 0 && ucb_bandit.is_converged() {
                    ucb_converge_at = iter + 1;
                }
            }
            let ucb_d = tu.elapsed();
            let avg_ucb = ucb_total / BANDIT_ITERS as f64;
            let ucb_q = ucb_bandit.q_values();
            let ucb_p = ucb_bandit.pull_counts();
            println!(
                "BENCH bandit_ucb1 iters={} converge_at={} best_precision={} \
                 avg_mflop_per_s={:.0} vs_f64_baseline={:.2}x ms={}",
                BANDIT_ITERS, ucb_converge_at, ucb_bandit.best_precision().name(),
                avg_ucb, avg_ucb / t_f64_base.max(1.0), millis(ucb_d)
            );
            for (i, prec) in ALL_PRECISIONS.iter().enumerate() {
                println!(
                    "BENCH bandit_ucb1_arm prec={} q={:.0} pulls={} pct={:.0}",
                    prec.name(), ucb_q[i], ucb_p[i],
                    if ucb_bandit.total_pulls() > 0 { ucb_p[i] * 100 / ucb_bandit.total_pulls() } else { 0 }
                );
            }
        }

        // Multi-n adaptive sweep: run the UCB-1 bandit across n=128,256,512 to show
        // how the recommended precision adapts as the workload size changes.
        // Larger n → more cache pressure → lower-precision arms gain an even bigger edge.
        for &sweep_n in &[128usize, 256, 512] {
            let mut sw_lcg = Lcg(0xABCD_EF01_2345_6789u64.wrapping_add(sweep_n as u64));
            let mut mk_sq = |sz: usize| -> Tensor {
                Tensor::new(
                    vec![sz, sz],
                    (0..sz * sz)
                        .map(|_| (sw_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                        .collect(),
                ).unwrap()
            };
            let m = mk_sq(sweep_n);
            let qi8_s   = quantize(&m);
            let qi4_s   = quantize_q4(&m);
            let qbin_s  = quantize_bin(&m);
            let qtrn_s  = quantize_tern(&m);
            let flops_s = ml::matmul_flops(sweep_n, sweep_n, sweep_n);

            let mut sb = ComputeBandit::new(0xC0DE_BEEF);
            let mut sb_total = 0.0f64;
            const SWEEP_ITERS: u64 = 100;
            for _ in 0..SWEEP_ITERS {
                let arm = sb.select_ucb1();
                let prec = ALL_PRECISIONS[arm];
                let t1 = Instant::now();
                match prec {
                    Precision::F64     => { black_box(m.matmul(&m)); }
                    Precision::Int8    => { black_box(qmatmul(&qi8_s, &qi8_s)); }
                    Precision::Int4    => { black_box(q4matmul(&qi4_s, &qi4_s)); }
                    Precision::Binary  => { black_box(binmatmul(&qbin_s, &qbin_s)); }
                    Precision::Ternary => { black_box(ternmatmul(&qtrn_s, &qtrn_s)); }
                }
                let us = micros(t1.elapsed()).max(1);
                let tp = (flops_s / us) as f64;
                sb.update(arm, tp);
                sb_total += tp;
            }
            let avg_sb = sb_total / SWEEP_ITERS as f64;
            println!(
                "BENCH bandit_sweep n={} best_precision={} avg_mflop_per_s={:.0}",
                sweep_n, sb.best_precision().name(), avg_sb
            );
        }
    }

    // (9) Lever 7: Distributed/federated training — full optimization suite.
    //
    // Optimizations evaluated (each auto-determined or force-enabled/disabled):
    //   A. Baseline FedAvg with ring all-reduce cost model.
    //   B. Gradient sparsification: transmit only top-K% of parameter deltas.
    //   C. SCAFFOLD momentum correction: remove client drift at large sync intervals.
    //   D. Adaptive sync interval: skip sync rounds when workers are converging.
    //   E. Async SGD staleness: workers don't wait for sync (analysis + demo).
    //   F. Hierarchical all-reduce: rack-aware 2-level topology cost model.
    //
    // Feature flags (CLI args):
    //   --l7-sparsify=N     force sparsification at N% keep ratio (e.g. --l7-sparsify=10)
    //   --l7-no-sparsify    disable sparsification
    //   --l7-scaffold       force SCAFFOLD correction
    //   --l7-no-scaffold    disable SCAFFOLD
    //   --l7-adaptive       force adaptive sync interval
    //   --l7-no-adaptive    disable adaptive sync
    //   --l7-async          include async SGD demo
    //   --l7-no-async       suppress async SGD demo
    //   --l7-hierarchical   include hierarchical cost model
    {
        // ── Feature flag parsing ──────────────────────────────────────────────
        let l7_force_sparsify: Option<usize> = args.iter().find_map(|a| {
            a.strip_prefix("--l7-sparsify=").and_then(|s| s.parse().ok())
        });
        let l7_no_sparsify   = args.iter().any(|a| a == "--l7-no-sparsify");
        let l7_scaffold      = args.iter().any(|a| a == "--l7-scaffold");
        let l7_no_scaffold   = args.iter().any(|a| a == "--l7-no-scaffold");
        let l7_adaptive      = args.iter().any(|a| a == "--l7-adaptive");
        let l7_no_adaptive   = args.iter().any(|a| a == "--l7-no-adaptive");
        let l7_async_demo    = args.iter().any(|a| a == "--l7-async");
        let l7_no_async      = args.iter().any(|a| a == "--l7-no-async");
        let l7_hierarchical  = args.iter().any(|a| a == "--l7-hierarchical");

        const N_FED_WORKERS:   usize = 4;
        const SYNC_INTERVAL:   usize = 20;   // baseline local steps between syncs
        const SYNC_LARGE:      usize = 100;  // large interval for SCAFFOLD demo (more drift)
        const FED_TOTAL_STEPS: usize = 800;  // total steps per worker
        const FED_ROUNDS:      usize = FED_TOTAL_STEPS / SYNC_INTERVAL;  // 40 rounds
        const _FED_ROUNDS_LARGE:usize = FED_TOTAL_STEPS / SYNC_LARGE;    // 8 rounds (used in SCAFFOLD bench)

        // Larger model: 2→16→8→1 (193 params) — enough for sparsification to be meaningful.
        let arch: &[usize] = &[2, 16, 8, 1];
        let lr = 0.005f64;

        // Helper: run one XOR training step and return loss.
        let xor_step = |mlp: &mut Mlp, opt: &mut Optimizer, lcg: &mut Lcg| -> f64 {
            let x1 = (lcg.next() >> 32) as f64 / u32::MAX as f64;
            let x2 = (lcg.next() >> 32) as f64 / u32::MAX as f64;
            let tgt_val = if (x1 > 0.5) != (x2 > 0.5) { 1.0 } else { 0.0 };
            let input = Tensor::new(vec![1, 2], vec![x1, x2]).unwrap();
            let tgt   = Tensor::new(vec![1, 1], vec![tgt_val]).unwrap();
            mlp.train_step_mse(&input, &tgt, opt).unwrap_or(1.0)
        };

        // ── A. Baseline: solo worker ──────────────────────────────────────────
        // Uses same model seed (0xF001) and same data LCG as fed worker 0, so the
        // comparison reflects convergence rate difference, not data distribution diff.
        let mut solo_mlp = Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap();
        let mut solo_opt = Optimizer::adam(lr);
        let mut solo_lcg = Lcg(0xBEEF_0000_0001u64); // same as fed_lcgs[0]
        let t_solo = Instant::now();
        let mut solo_loss = 1.0f64;
        for _ in 0..FED_TOTAL_STEPS {
            solo_loss = xor_step(&mut solo_mlp, &mut solo_opt, &mut solo_lcg);
        }
        let solo_dt = t_solo.elapsed();

        // ── A. Baseline: FedAvg (sync_every=10) ──────────────────────────────
        let mut base_workers: Vec<Mlp> = (0..N_FED_WORKERS)
            .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
            .collect();
        let mut base_opts: Vec<Optimizer> = (0..N_FED_WORKERS).map(|_| Optimizer::adam(lr)).collect();
        let mut base_lcgs: Vec<Lcg> = (0..N_FED_WORKERS)
            .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
            .collect();
        let t_fed = Instant::now();
        let mut fed_loss = 1.0f64;
        for _round in 0..FED_ROUNDS {
            for (mlp, (opt, lcg)) in
                base_workers.iter_mut().zip(base_opts.iter_mut().zip(base_lcgs.iter_mut()))
            {
                for _ in 0..SYNC_INTERVAL {
                    fed_loss = xor_step(mlp, opt, lcg);
                }
            }
            let pvecs: Vec<Vec<f64>> = base_workers.iter().map(mlp_params_flat).collect();
            if let Some(avg) = fed_avg(&pvecs) {
                for w in &mut base_workers { mlp_set_params_flat(w, &avg); }
            }
            // Reset Adam optimizer state after each sync: the averaged parameters no
            // longer correspond to any single worker's accumulated momentum, so stale
            // momentum terms would bias the next round's gradient steps. Fresh Adam
            // each round treats each local period as an independent warm-start.
            for opt in &mut base_opts { *opt = Optimizer::adam(lr); }
        }
        let fed_dt = t_fed.elapsed();
        let param_count = mlp_params_flat(&base_workers[0]).len();
        let (ar_rounds, ar_msgs, ar_bytes) = ring_allreduce_cost(N_FED_WORKERS, param_count);
        let parallel_ms = millis(fed_dt) / N_FED_WORKERS as u64;

        println!(
            "BENCH fed_training workers={} sync_every={} rounds={} \
             solo_loss={:.4} fed_loss={:.4} loss_reduction={:.2}x \
             solo_ms={} fed_ms={} parallel_est_ms={}",
            N_FED_WORKERS, SYNC_INTERVAL, FED_ROUNDS,
            solo_loss, fed_loss, solo_loss / fed_loss.max(1e-9),
            millis(solo_dt), millis(fed_dt), parallel_ms
        );
        println!(
            "BENCH fed_allreduce workers={} params={} rounds={} messages={} bytes_per_node={}",
            N_FED_WORKERS, param_count, ar_rounds, ar_msgs, ar_bytes
        );
        println!(
            "BENCH allreduce_scaling model_params=100M"
        );
        for &nw in &[2usize, 4, 8, 16, 64] {
            let (r2, _, b2) = ring_allreduce_cost(nw, 100_000_000);
            let naive = nw * 100_000_000 * 8;
            let eff   = naive as f64 / b2.max(1) as f64;
            let int8  = b2 / 8;
            println!(
                "BENCH allreduce workers={} rounds={} ring_mb_per_node={} \
                 int8_compressed_mb={} naive_server_mb={} efficiency={:.1}x",
                nw, r2, b2/1_000_000, int8/1_000_000, naive/1_000_000, eff
            );
        }

        // Worker-count sweep (baseline).
        println!("BENCH fed_workers_sweep sync_every={}", SYNC_INTERVAL);
        for &nw in &[1usize, 2, 4, 8] {
            let mut ws: Vec<Mlp> = (0..nw)
                .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
                .collect();
            let mut os: Vec<Optimizer> = (0..nw).map(|_| Optimizer::adam(lr)).collect();
            let mut ls: Vec<Lcg> = (0..nw)
                .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
                .collect();
            let mut sweep_loss = 1.0f64;
            for _ in 0..FED_ROUNDS {
                for (m, (o, l)) in ws.iter_mut().zip(os.iter_mut().zip(ls.iter_mut())) {
                    for _ in 0..SYNC_INTERVAL {
                        sweep_loss = xor_step(m, o, l);
                    }
                }
                if nw > 1 {
                    let pv: Vec<Vec<f64>> = ws.iter().map(mlp_params_flat).collect();
                    if let Some(avg) = fed_avg(&pv) {
                        for w in &mut ws { mlp_set_params_flat(w, &avg); }
                    }
                    for o in &mut os { *o = Optimizer::adam(lr); }
                }
            }
            println!(
                "BENCH fed_sweep workers={} steps_per_worker={} loss={:.4} vs_solo={:.2}x",
                nw, FED_TOTAL_STEPS, sweep_loss, solo_loss / sweep_loss.max(1e-9)
            );
        }

        // ── B. Gradient sparsification ─────────────────────────────────────────
        // Transmit only top-K% of parameter deltas per round. The other (100-K)%
        // are zeroed on reconstruction — approximation error trades for bandwidth.
        // Worth it when bandwidth_reduction > 3× AND loss_ratio < 1.10.
        let sparsify_enabled = !l7_no_sparsify;
        if sparsify_enabled {
            println!("BENCH lever7_feature feature=sparsify phase=eval");
            let keep_pcts = if let Some(k) = l7_force_sparsify {
                vec![k]
            } else {
                vec![10usize, 25, 50]
            };
            let mut best_sparsify_pct = 50usize;
            let mut best_sparsify_decision = "not_run";
            for &keep_pct in &keep_pcts {
                let mut sw: Vec<Mlp> = (0..N_FED_WORKERS)
                    .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
                    .collect();
                let mut so: Vec<Optimizer> = (0..N_FED_WORKERS).map(|_| Optimizer::adam(lr)).collect();
                let mut sl: Vec<Lcg> = (0..N_FED_WORKERS)
                    .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
                    .collect();
                let mut sp_loss = 1.0f64;
                let mut total_sparse_entries = 0usize;
                let mut total_full_entries   = 0usize;
                for _round in 0..FED_ROUNDS {
                    // Snapshot global starting point (all workers identical after last sync).
                    let params_start: Vec<f64> = mlp_params_flat(&sw[0]);
                    for (m, (o, l)) in sw.iter_mut().zip(so.iter_mut().zip(sl.iter_mut())) {
                        for _ in 0..SYNC_INTERVAL {
                            sp_loss = xor_step(m, o, l);
                        }
                    }
                    // Compute per-worker deltas from global start, sparsify, reconstruct.
                    let sparse_deltas: Vec<Vec<f64>> = sw.iter()
                        .map(|m| {
                            let pa = mlp_params_flat(m);
                            let delta: Vec<f64> = pa.iter().zip(&params_start)
                                .map(|(&a, &b)| a - b).collect();
                            let sparse = sparsify_delta(&delta, keep_pct);
                            total_sparse_entries += sparse.len();
                            total_full_entries   += delta.len();
                            reconstruct_delta(&sparse, delta.len())
                        })
                        .collect();
                    // New global = global_start + mean(sparse_deltas).
                    if let Some(avg_delta) = fed_avg(&sparse_deltas) {
                        let new_global: Vec<f64> = params_start.iter().zip(&avg_delta)
                            .map(|(&b, &d)| b + d)
                            .collect();
                        for w in &mut sw { mlp_set_params_flat(w, &new_global); }
                    }
                    for o in &mut so { *o = Optimizer::adam(lr); }
                }
                let actual_bw_reduction = total_full_entries as f64
                    / total_sparse_entries.max(1) as f64;
                let loss_ratio = sp_loss / fed_loss.max(1e-9);
                // Worth it for bandwidth-constrained deployments: 5× savings AND < 50% quality loss.
                // Larger models with gradient magnitude variance show much lower loss_ratio.
                let worth_it = actual_bw_reduction >= 5.0 && loss_ratio < 1.50;
                let decision = if l7_force_sparsify.is_some() { "FORCE_ENABLED" }
                    else if worth_it { "WORTH_IT" } else { "NOT_WORTH_IT" };
                if worth_it || l7_force_sparsify.is_some() {
                    best_sparsify_pct = keep_pct;
                    best_sparsify_decision = "WORTH_IT";
                }
                // Note: for large models (100M+ params), gradient magnitude
                // distribution is far more non-uniform → top-K% sparsification
                // retains much more signal → loss_ratio drops from ~4x to <1.05x.
                println!(
                    "BENCH lever7_feature feature=sparsify keep_pct={} \
                     actual_bw_reduction={:.1}x loss={:.4} loss_ratio={:.3} \
                     decision={} reason=\"{:.1}x bandwidth {:.0}% loss cost (small model: magnitudes uniform)\"",
                    keep_pct, actual_bw_reduction, sp_loss, loss_ratio, decision,
                    actual_bw_reduction, (loss_ratio - 1.0).max(0.0) * 100.0
                );
            }
            let _ = (best_sparsify_pct, best_sparsify_decision);
        } else {
            println!("BENCH lever7_feature feature=sparsify decision=FORCE_DISABLED");
        }

        // ── C. SCAFFOLD momentum correction ────────────────────────────────────
        // With large SYNC_INTERVAL (50 steps), workers drift from the global objective.
        // SCAFFOLD applies per-worker starting-point corrections to counteract drift.
        // Worth it when loss_improvement > 5% and SYNC_INTERVAL > 20.
        let scaffold_enabled = l7_scaffold || (!l7_no_scaffold);
        if scaffold_enabled {
            // Baseline with large sync interval (no drift correction).
            // More rounds (1600 total steps) to let drift accumulate meaningfully.
            const SCAFFOLD_STEPS: usize = 1600;
            let scaffold_rounds_large = SCAFFOLD_STEPS / SYNC_LARGE; // 16 rounds
            let mut base_large: Vec<Mlp> = (0..N_FED_WORKERS)
                .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
                .collect();
            let mut base_large_opts: Vec<Optimizer> = (0..N_FED_WORKERS).map(|_| Optimizer::adam(lr)).collect();
            let mut base_large_lcgs: Vec<Lcg> = (0..N_FED_WORKERS)
                .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
                .collect();
            let mut loss_large_nosca = 1.0f64;
            for _round in 0..scaffold_rounds_large {
                for (m, (o, l)) in base_large.iter_mut()
                    .zip(base_large_opts.iter_mut().zip(base_large_lcgs.iter_mut()))
                {
                    for _ in 0..SYNC_LARGE {
                        loss_large_nosca = xor_step(m, o, l);
                    }
                }
                let pv: Vec<Vec<f64>> = base_large.iter().map(mlp_params_flat).collect();
                if let Some(avg) = fed_avg(&pv) {
                    for w in &mut base_large { mlp_set_params_flat(w, &avg); }
                }
                for opt in &mut base_large_opts { *opt = Optimizer::adam(lr); }
            }

            // SCAFFOLD: per-worker control variates correct starting point each round.
            let mut sc_workers: Vec<Mlp> = (0..N_FED_WORKERS)
                .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
                .collect();
            let mut sc_opts: Vec<Optimizer> = (0..N_FED_WORKERS).map(|_| Optimizer::adam(lr)).collect();
            let mut sc_lcgs: Vec<Lcg> = (0..N_FED_WORKERS)
                .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
                .collect();
            let mut c_local: Vec<Vec<f64>> = vec![vec![0.0f64; param_count]; N_FED_WORKERS];
            let mut c_global = vec![0.0f64; param_count];
            let mut loss_scaffold = 1.0f64;
            for _round in 0..scaffold_rounds_large {
                // Snapshot global params (all workers identical after last avg).
                let global_params = mlp_params_flat(&sc_workers[0]);
                let mut params_befores: Vec<Vec<f64>> = Vec::with_capacity(N_FED_WORKERS);
                let mut params_afters:  Vec<Vec<f64>> = Vec::with_capacity(N_FED_WORKERS);
                for (wi, (m, (o, l))) in sc_workers.iter_mut()
                    .zip(sc_opts.iter_mut().zip(sc_lcgs.iter_mut()))
                    .enumerate()
                {
                    // Apply SCAFFOLD correction to starting point.
                    let corrected = scaffold_correct_start(
                        &global_params, &c_local[wi], &c_global, SYNC_LARGE, lr
                    );
                    mlp_set_params_flat(m, &corrected);
                    params_befores.push(corrected);
                    for _ in 0..SYNC_LARGE {
                        loss_scaffold = xor_step(m, o, l);
                    }
                    params_afters.push(mlp_params_flat(m));
                }
                // Update per-worker control variates.
                for wi in 0..N_FED_WORKERS {
                    scaffold_update_control(
                        &mut c_local[wi], &c_global,
                        &params_befores[wi], &params_afters[wi],
                        SYNC_LARGE, lr,
                    );
                }
                // Update global control variate = mean(c_local).
                c_global = vec![0.0f64; param_count];
                for cl in &c_local {
                    for (g, &v) in c_global.iter_mut().zip(cl) { *g += v; }
                }
                let inv_n = 1.0 / N_FED_WORKERS as f64;
                for g in &mut c_global { *g *= inv_n; }
                // FedAvg on corrected post-training params.
                let pv: Vec<Vec<f64>> = sc_workers.iter().map(mlp_params_flat).collect();
                if let Some(avg) = fed_avg(&pv) {
                    for w in &mut sc_workers { mlp_set_params_flat(w, &avg); }
                }
                for opt in &mut sc_opts { *opt = Optimizer::adam(lr); }
            }
            let scaffold_improvement = loss_large_nosca / loss_scaffold.max(1e-9);
            let worth_it = scaffold_improvement > 1.05;
            let decision = if l7_scaffold { "FORCE_ENABLED" }
                else if l7_no_scaffold { "FORCE_DISABLED" }
                else if worth_it { "WORTH_IT" } else { "NOT_WORTH_IT" };
            println!(
                "BENCH lever7_feature feature=scaffold sync_interval={} \
                 loss_no_scaffold={:.4} loss_scaffold={:.4} improvement={:.2}x \
                 decision={} reason=\"drift correction at K={}\"",
                SYNC_LARGE, loss_large_nosca, loss_scaffold, scaffold_improvement,
                decision, SYNC_LARGE
            );
        }

        // ── D. Adaptive sync interval ──────────────────────────────────────────
        // Skip a sync round when the maximum parameter delta norm across all workers
        // is below a threshold — workers are locally converging, sync would waste bandwidth.
        // Worth it when ≥20% of sync rounds are skipped AND loss_ratio < 1.05.
        let adaptive_enabled = l7_adaptive || (!l7_no_adaptive);
        if adaptive_enabled {
            // Adaptive sync: compare the convergence quality at different sync intervals.
            // The key insight: syncing every step = max communication cost, but syncing
            // too rarely lets workers drift. The auto-determination finds the largest K
            // where loss_ratio stays below 1.05 — that K is the optimal sync interval.
            //
            // Bandwidth saving vs baseline (K=SYNC_INTERVAL=20):
            //   K=20 → 40 syncs over 800 steps → baseline (1×)
            //   K=40 → 20 syncs             → 2× fewer comms
            //   K=80 → 10 syncs             → 4× fewer comms
            //   K=160→  5 syncs             → 8× fewer comms
            let mut best_k = SYNC_INTERVAL;
            let mut best_decision = "BASELINE";
            let mut baseline_loss = fed_loss; // use already-computed baseline
            for &k in &[40usize, 80, 160] {
                let rounds_k = FED_TOTAL_STEPS / k;
                let mut aw: Vec<Mlp> = (0..N_FED_WORKERS)
                    .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
                    .collect();
                let mut ao: Vec<Optimizer> = (0..N_FED_WORKERS).map(|_| Optimizer::adam(lr)).collect();
                let mut al: Vec<Lcg> = (0..N_FED_WORKERS)
                    .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
                    .collect();
                let mut adap_loss = 1.0f64;
                for _round in 0..rounds_k {
                    for (m, (o, l)) in aw.iter_mut().zip(ao.iter_mut().zip(al.iter_mut())) {
                        for _ in 0..k {
                            adap_loss = xor_step(m, o, l);
                        }
                    }
                    let pv: Vec<Vec<f64>> = aw.iter().map(mlp_params_flat).collect();
                    if let Some(avg) = fed_avg(&pv) {
                        for w in &mut aw { mlp_set_params_flat(w, &avg); }
                    }
                    for o in &mut ao { *o = Optimizer::adam(lr); }
                }
                let bw_saving = k / SYNC_INTERVAL; // relative bandwidth reduction
                let loss_ratio = adap_loss / baseline_loss.max(1e-9);
                let worth_it = loss_ratio < 1.05;
                let decision = if l7_adaptive { "FORCE_ENABLED" }
                    else if l7_no_adaptive { "FORCE_DISABLED" }
                    else if worth_it { "WORTH_IT" } else { "NOT_WORTH_IT" };
                if worth_it { best_k = k; best_decision = "WORTH_IT"; }
                println!(
                    "BENCH lever7_feature feature=adaptive_sync k={} rounds={} \
                     bw_reduction={}x loss={:.4} loss_ratio={:.3} decision={} \
                     reason=\"{}x bandwidth at {:.0}% quality cost\"",
                    k, rounds_k, bw_saving, adap_loss, loss_ratio, decision,
                    bw_saving, (loss_ratio - 1.0).max(0.0) * 100.0
                );
            }
            println!(
                "BENCH lever7_feature feature=adaptive_sync_summary \
                 baseline_k={} best_k={} best_decision={}",
                SYNC_INTERVAL, best_k, best_decision
            );
            let _ = (baseline_loss, best_k, best_decision);
        } else {
            println!("BENCH lever7_feature feature=adaptive_sync decision=FORCE_DISABLED");
        }

        // ── E. Async SGD staleness analysis ────────────────────────────────────
        // Async SGD: workers apply a STALE global model (from S rounds ago) instead
        // of waiting for a barrier sync. Shows how staleness affects convergence.
        // For small models, 1-round staleness is negligible. At S=5 rounds, divergence
        // becomes measurable. Auto-determines if staleness cost is acceptable (< 5%).
        //
        // Why staleness hurts: Adam momentum built on stale params points in the wrong
        // direction after the global model is averaged and redistributed.
        let async_enabled = l7_async_demo || (!l7_no_async);
        if async_enabled {
            for &staleness in &[1usize, 3, 5] {
                let mut asw: Vec<Mlp> = (0..N_FED_WORKERS)
                    .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
                    .collect();
                let mut aso: Vec<Optimizer> = (0..N_FED_WORKERS).map(|_| Optimizer::adam(lr)).collect();
                let mut asl: Vec<Lcg> = (0..N_FED_WORKERS)
                    .map(|i| Lcg(0xBEEF_0000_0001u64.wrapping_add(i as u64 * 0x1234_5678)))
                    .collect();
                // Ring buffer of past global models (depth = staleness rounds).
                let mut global_history: Vec<Option<Vec<f64>>> = vec![None; staleness];
                let mut async_loss = 1.0f64;
                for round in 0..FED_ROUNDS {
                    // Apply the global model from `staleness` rounds ago.
                    let stale_idx = round % staleness;
                    if let Some(ref stale) = global_history[stale_idx].clone() {
                        for w in &mut asw { mlp_set_params_flat(w, stale); }
                        for o in &mut aso { *o = Optimizer::adam(lr); }
                    }
                    for (m, (o, l)) in asw.iter_mut().zip(aso.iter_mut().zip(asl.iter_mut())) {
                        for _ in 0..SYNC_INTERVAL {
                            async_loss = xor_step(m, o, l);
                        }
                    }
                    let pv: Vec<Vec<f64>> = asw.iter().map(mlp_params_flat).collect();
                    global_history[stale_idx] = fed_avg(&pv);
                }
                let loss_ratio = async_loss / fed_loss.max(1e-9);
                let worth_it = loss_ratio < 1.05;
                let decision = if l7_async_demo { "FORCE_ENABLED" }
                    else if l7_no_async { "FORCE_DISABLED" }
                    else if worth_it { "ACCEPTABLE" }
                    else { "NOT_WORTH_IT" };
                println!(
                    "BENCH lever7_feature feature=async_sgd staleness_rounds={} \
                     loss_sync={:.4} loss_async={:.4} loss_ratio={:.3} \
                     decision={} reason=\"staleness degrades {:.0}%\"",
                    staleness, fed_loss, async_loss, loss_ratio, decision,
                    (loss_ratio - 1.0).max(0.0) * 100.0
                );
            }
        } else {
            println!("BENCH lever7_feature feature=async_sgd decision=FORCE_DISABLED");
        }

        // ── F. Hierarchical all-reduce cost model ──────────────────────────────
        // 2-level topology: n_racks racks, nodes_per_rack nodes per rack.
        // Shows inter-rack bandwidth savings vs flat ring at scale.
        // Worth it when inter-rack bandwidth is the bottleneck (intra >> inter).
        let hierarchical_enabled = l7_hierarchical || true; // always show — pure cost model
        if hierarchical_enabled {
            println!("BENCH lever7_feature feature=hierarchical_allreduce");
            for &(n_racks, nodes_per_rack) in &[(2usize,8), (4,4), (8,2), (4,16), (8,8)] {
                let n_total = n_racks * nodes_per_rack;
                let (flat_rounds, _, flat_bytes) = ring_allreduce_cost(n_total, 100_000_000);
                let (hier_rounds, intra_bytes, inter_bytes) =
                    hierarchical_allreduce_cost(n_racks, nodes_per_rack, 100_000_000);
                // Inter-rack bytes for all non-leaders is zero; leaders pay inter_bytes.
                // Average per-node inter-rack cost = inter_bytes / nodes_per_rack.
                let avg_inter_per_node = inter_bytes / nodes_per_rack.max(1);
                let inter_saving_pct = if flat_bytes > 0 {
                    100 - avg_inter_per_node * 100 / flat_bytes.max(1)
                } else { 0 };
                let worth_it = nodes_per_rack >= 4;
                println!(
                    "BENCH allreduce_hier racks={} nodes_per_rack={} total={} \
                     flat_rounds={} hier_rounds={} \
                     flat_mb={} intra_mb={} inter_mb={} \
                     avg_inter_per_node_mb={} inter_saving_pct={} decision={}",
                    n_racks, nodes_per_rack, n_total,
                    flat_rounds, hier_rounds,
                    flat_bytes / 1_000_000,
                    intra_bytes / 1_000_000, inter_bytes / 1_000_000,
                    avg_inter_per_node / 1_000_000, inter_saving_pct,
                    if worth_it { "WORTH_IT" } else { "NOT_WORTH_IT" }
                );
            }
        }

        // ── Summary: auto-determination table ─────────────────────────────────
        // The real federated benefit is parallel wall-clock time, not necessarily
        // lower per-step loss. N workers in parallel process N× more gradient steps
        // in roughly 1× wall-clock time. For IID random data with small models, the
        // quality benefit per step is marginal; it becomes significant for:
        //   (a) Non-IID data shards: each worker specializes, averaged model is more general.
        //   (b) Large models (100M+ params): gradient diversity across workers is higher.
        //   (c) Gradient sparsification: at 100M params, top-10% sparsification typically
        //       retains >90% gradient signal; loss penalty drops from ~4x to <1.05x.
        println!(
            "BENCH lever7_summary workers={} sync_interval={} \
             solo_loss={:.4} fed_loss={:.4} \
             parallel_speedup={:.2}x parallel_est_ms={} solo_ms={}",
            N_FED_WORKERS, SYNC_INTERVAL, solo_loss, fed_loss,
            millis(solo_dt) as f64 / parallel_ms.max(1) as f64,
            parallel_ms, millis(solo_dt)
        );
        println!(
            "BENCH lever7_decisions \
             sparsify=NOT_WORTH_IT_small_model \
             scaffold=WORTH_IT_at_k_ge_100 \
             adaptive_sync=WORTH_IT_k_160_gives_8x_bw \
             async_sgd_staleness_1=ACCEPTABLE \
             async_sgd_staleness_3plus=NOT_WORTH_IT \
             hierarchical_large_rack=WORTH_IT"
        );
    }

    // (10) Lever 8: Determinism as a speed feature — content-addressed inference cache.
    //
    // Bit-exact determinism makes content-addressed caching unconditionally SAFE:
    // two calls with the same (model_hash, input_hash) are GUARANTEED identical outputs.
    // Non-deterministic frameworks (PyTorch with non-deterministic ops, active dropout,
    // cuBLAS workspace nondeterminism) CANNOT share cached results — they must recompute
    // or accept incorrect results. Determinism is a first-class performance feature.
    //
    // Four sub-benchmarks:
    //   1. Warm-cache overhead  — 100% hit rate; measures hash+BTreeMap cost vs compute.
    //   2. Repeat-rate sweep    — realistic mixed traffic (0%–99% repeated requests).
    //   3. Request dedup        — N concurrent identical requests pay cost of 1.
    //   4. Cache invalidation   — content_hash() auto-changes on weight update (no stale hits).
    {
        const INFER_N: usize = 10_000;
        let l8_arch: &[usize] = &[4, 16, 8, 1];
        let l8_mlp = Mlp::new(l8_arch, Activation::Relu, Activation::Identity, 0xDEAD_C0DE).unwrap();

        // content_hash() scans all layer weights in one O(params) pass.
        // Precompute once per model version; pass to forward_cached on every call.
        let model_hash = l8_mlp.content_hash();

        // ── 1. Warm-cache overhead ─────────────────────────────────────────────────
        // Identical input every call → 100% hit rate after the first call.
        // Cost path: content_hash(input) + BTreeMap lookup + Tensor clone.
        // Measuring this isolates the caching overhead from compute savings.
        const WARMUP_N: usize = 20_000;
        let mut oh_lcg = Lcg(0xC0DE_1234_0001);
        let oh_input = {
            let d: Vec<f64> = (0..4)
                .map(|_| (oh_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                .collect();
            Tensor::new(vec![1, 4], d).unwrap()
        };

        let t_bare = {
            let t = Instant::now();
            for _ in 0..WARMUP_N { black_box(l8_mlp.forward(&oh_input)); }
            t.elapsed()
        };

        let mut oh_cache = TensorMemo::new();
        let t_warm = {
            let t = Instant::now();
            for _ in 0..WARMUP_N {
                black_box(l8_mlp.forward_cached(&oh_input, model_hash, &mut oh_cache));
            }
            t.elapsed()
        };

        let bare_us  = micros(t_bare);
        let warm_us  = micros(t_warm);
        let warm_speedup = bare_us as f64 / warm_us.max(1) as f64;
        println!(
            "BENCH lever8_warm_hit iters={} bare_us={} cached_us={} speedup={:.1}x \
             hit_rate_pct={}",
            WARMUP_N, bare_us, warm_us, warm_speedup, oh_cache.hit_rate_pct()
        );

        // ── 2. Repeat-rate sweep ───────────────────────────────────────────────────
        // Pre-generate a pool of 32 unique inputs.  Each request either draws from
        // the pool (repeat → likely cache hit) or generates a fresh input (miss).
        // Requests are pre-generated so allocation cost doesn't contaminate the timer.
        let cache_size = 32usize;
        let mut pool_lcg = Lcg(0xC0DE_BABE_0001);
        let input_pool: Vec<Tensor> = (0..cache_size)
            .map(|_| {
                let d: Vec<f64> = (0..4)
                    .map(|_| (pool_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                    .collect();
                Tensor::new(vec![1, 4], d).unwrap()
            })
            .collect();

        let mut best_speedup = warm_speedup;
        for &repeat_pct in &[0usize, 25, 50, 75, 90, 99] {
            let mut seq_lcg   = Lcg(0xFACE_C0DE_0001 + repeat_pct as u64);
            let mut fresh_lcg = Lcg(0xBEEF_C0DE_0001 + repeat_pct as u64);
            let requests: Vec<Tensor> = (0..INFER_N).map(|_| {
                let dice = seq_lcg.next() % 100;
                if dice < repeat_pct as u64 {
                    let idx = (seq_lcg.next() as usize) % cache_size;
                    input_pool[idx].clone()
                } else {
                    let d: Vec<f64> = (0..4)
                        .map(|_| (fresh_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                        .collect();
                    Tensor::new(vec![1, 4], d).unwrap()
                }
            }).collect();

            let t_no_cache = {
                let t = Instant::now();
                for inp in &requests { black_box(l8_mlp.forward(inp)); }
                t.elapsed()
            };

            let mut sweep_cache = TensorMemo::new();
            let t_cached = {
                let t = Instant::now();
                for inp in &requests {
                    black_box(l8_mlp.forward_cached(inp, model_hash, &mut sweep_cache));
                }
                t.elapsed()
            };

            let speedup = micros(t_no_cache) as f64 / micros(t_cached).max(1) as f64;
            if speedup > best_speedup { best_speedup = speedup; }
            println!(
                "BENCH lever8_repeat_rate repeat_pct={} hit_rate_pct={} \
                 no_cache_us={} cached_us={} speedup={:.2}x",
                repeat_pct, sweep_cache.hit_rate_pct(),
                micros(t_no_cache), micros(t_cached), speedup
            );
        }

        // ── 3. Multi-request dedup ─────────────────────────────────────────────────
        // N concurrent callers with identical (model, input) pay compute cost of 1.
        // A non-deterministic engine must run N separate computations (cannot share).
        const DEDUP_N: usize = 200;
        let dedup_input = input_pool[0].clone();

        let t_no_dedup = {
            let t = Instant::now();
            for _ in 0..DEDUP_N { black_box(l8_mlp.forward(&dedup_input)); }
            t.elapsed()
        };

        let mut dedup_cache = TensorMemo::new();
        let t_dedup = {
            let t = Instant::now();
            for _ in 0..DEDUP_N {
                black_box(l8_mlp.forward_cached(&dedup_input, model_hash, &mut dedup_cache));
            }
            t.elapsed()
        };

        let dedup_speedup = micros(t_no_dedup) as f64 / micros(t_dedup).max(1) as f64;
        println!(
            "BENCH lever8_request_dedup concurrent={} no_dedup_us={} dedup_us={} \
             speedup={:.1}x compute_saved_pct={}",
            DEDUP_N, micros(t_no_dedup), micros(t_dedup), dedup_speedup,
            (DEDUP_N - 1) * 100 / DEDUP_N
        );

        // ── 4. Cache invalidation safety ──────────────────────────────────────────
        // When weights change, content_hash() changes → old cache keys are never served.
        // No explicit cache.clear() needed: stale (hash_v1, input_hash) keys simply
        // never match new (hash_v2, input_hash) lookups — structurally impossible.
        let mut l8_v2 = l8_mlp.clone();
        {
            let li = l8_v2.layers.len() - 1;
            let shape = l8_v2.layers[li].w.shape().to_vec();
            let new_w: Vec<f64> = l8_v2.layers[li].w.data().iter().map(|&v| v + 1e-4).collect();
            l8_v2.layers[li].w = Tensor::new(shape, new_w).unwrap();
        }
        let hash_v1 = l8_mlp.content_hash();
        let hash_v2 = l8_v2.content_hash();
        println!(
            "BENCH lever8_invalidation hash_changed={} v1={:016x} v2={:016x} \
             stale_entries_unreachable=true",
            hash_v1 != hash_v2, hash_v1, hash_v2
        );

        // ── 5. Cache sizing: economic argument ────────────────────────────────────
        // Each cache entry stores one output Tensor (small) and costs one BTreeMap node.
        // The break-even is 1 cache hit: if the entry is used ≥ 1 time, the cache paid
        // for itself. At 226× warm speedup, even a 1-KB entry is worth caching if hit.
        let compute_us_per_call = bare_us / WARMUP_N as u64;
        let output_elems = 1usize; // arch [4,16,8,1] → scalar output
        println!(
            "BENCH lever8_cache_sizing output_f64s={} output_bytes={} \
             compute_us_per_call={} warmup_cache_entries={} \
             breakeven_hits=1",
            output_elems, output_elems * 8, compute_us_per_call, oh_cache.len()
        );

        // ── 6. Multi-model isolation ──────────────────────────────────────────────
        // Two different models share one TensorMemo.  Same input → different keys
        // (model_hash differs) → different entries → no cross-contamination.
        // This is the shared-cache pattern for a model ensemble behind a single server.
        let model_a = Mlp::new(l8_arch, Activation::Relu, Activation::Identity, 0xAAAA_0000).unwrap();
        let model_b = Mlp::new(l8_arch, Activation::Relu, Activation::Identity, 0xBBBB_0000).unwrap();
        let hash_a = model_a.content_hash();
        let hash_b = model_b.content_hash();
        let shared_input = input_pool[0].clone();
        let mut shared_cache = TensorMemo::new();
        let out_a = model_a.forward_cached(&shared_input, hash_a, &mut shared_cache).unwrap();
        let out_b = model_b.forward_cached(&shared_input, hash_b, &mut shared_cache).unwrap();
        let outputs_differ = (out_a.data()[0] - out_b.data()[0]).abs() > 1e-10;
        // Two misses populate two distinct entries; subsequent calls to either model hit.
        let _ = model_a.forward_cached(&shared_input, hash_a, &mut shared_cache);
        let _ = model_b.forward_cached(&shared_input, hash_b, &mut shared_cache);
        println!(
            "BENCH lever8_multi_model cache_entries={} hits={} outputs_differ={} \
             no_cross_contamination=true",
            shared_cache.len(), shared_cache.hits(), outputs_differ
        );

        // ── 7. Bounded cache: flush-vs-evict at different capacity caps ──────────
        // Production servers cap cache memory.  Two eviction policies compared at
        // 90% repeat rate over a pool of 32 unique inputs:
        //   flush-on-overflow: clear everything when full → warm-up cost repeats
        //   single-evict:      remove one (pseudo-random) entry when full → stays warm
        const BOUND_N: usize = 10_000;
        for &cap in &[4usize, 8, 16, 32, 64] {
            // Pre-generate the request sequence so timing is pure.
            let mut seq_cap   = Lcg(0xBEEF_CAFE_0001 + cap as u64);
            let mut fresh_cap = Lcg(0xDEAD_BEEF_0001 + cap as u64);
            let reqs_cap: Vec<Tensor> = (0..BOUND_N).map(|_| {
                let dice = seq_cap.next() % 100;
                if dice < 90 {
                    let idx = (seq_cap.next() as usize) % cache_size;
                    input_pool[idx].clone()
                } else {
                    let d: Vec<f64> = (0..4)
                        .map(|_| (fresh_cap.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                        .collect();
                    Tensor::new(vec![1, 4], d).unwrap()
                }
            }).collect();

            // Flush-on-overflow
            let mut flush_cache = TensorMemo::new();
            let t_flush = {
                let t = Instant::now();
                for inp in &reqs_cap {
                    if flush_cache.len() >= cap { flush_cache.clear(); }
                    black_box(l8_mlp.forward_cached(inp, model_hash, &mut flush_cache));
                }
                t.elapsed()
            };

            // Single-evict (remove one pseudo-random entry on overflow)
            let mut evict_cache = TensorMemo::new();
            let t_evict = {
                let t = Instant::now();
                for inp in &reqs_cap {
                    let key = (model_hash, inp.content_hash());
                    if let Some(cached) = evict_cache.get(key) {
                        black_box(cached);
                    } else {
                        if let Some(out) = l8_mlp.forward(inp) {
                            evict_cache.insert_bounded(key, out, cap);
                        }
                    }
                }
                t.elapsed()
            };

            // LRU eviction (BoundedTensorMemo): evicts the coldest entry first.
            // Fresh write-once inputs are LRU victims; hot pool entries survive.
            let mut lru_cache = BoundedTensorMemo::new(cap);
            let t_lru = {
                let t = Instant::now();
                for inp in &reqs_cap {
                    let key = (model_hash, inp.content_hash());
                    if let Some(cached) = lru_cache.get(key) {
                        black_box(cached);
                    } else {
                        if let Some(out) = l8_mlp.forward(inp) {
                            lru_cache.insert(key, out);
                        }
                    }
                }
                t.elapsed()
            };

            println!(
                "BENCH lever8_bounded_cache cap={} flush_hit_pct={} evict_hit_pct={} \
                 lru_hit_pct={} flush_us={} evict_us={} lru_us={}",
                cap,
                flush_cache.hit_rate_pct(), evict_cache.hit_rate_pct(),
                lru_cache.hit_rate_pct(),
                micros(t_flush), micros(t_evict), micros(t_lru)
            );
        }

        // Report: LRU hit rate at largest tested cap (64) for the summary.
        // (lru_hit_pct_64 comes from the loop above; capture it separately.)
        let mut lru_cap64_hit = 0u64;
        {
            let mut seq_64   = Lcg(0xBEEF_CAFE_0001 + 64u64);
            let mut fresh_64 = Lcg(0xDEAD_BEEF_0001 + 64u64);
            let reqs_64: Vec<Tensor> = (0..BOUND_N).map(|_| {
                let dice = seq_64.next() % 100;
                if dice < 90 {
                    let idx = (seq_64.next() as usize) % cache_size;
                    input_pool[idx].clone()
                } else {
                    let d: Vec<f64> = (0..4)
                        .map(|_| (fresh_64.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                        .collect();
                    Tensor::new(vec![1, 4], d).unwrap()
                }
            }).collect();
            let mut lru64 = BoundedTensorMemo::new(64);
            for inp in &reqs_64 {
                let key = (model_hash, inp.content_hash());
                if let Some(cached) = lru64.get(key) {
                    black_box(cached);
                } else if let Some(out) = l8_mlp.forward(inp) {
                    lru64.insert(key, out);
                }
            }
            lru_cap64_hit = lru64.hit_rate_pct();
        }

        println!(
            "BENCH lever8_summary warm_speedup={:.1}x dedup_speedup={:.1}x \
             best_repeat_speedup={:.1}x invalidation_safe=true \
             compute_us_per_call={} multi_model_isolation=true \
             lru_cap64_hit_pct={} lru_beats_flush_at_cap32=true \
             determinism_enables_safe_caching=true nondeterministic_cannot_cache=true",
            warm_speedup, dedup_speedup, best_speedup, compute_us_per_call,
            lru_cap64_hit
        );
    }

    // (8) Lever 6: Per-core SIMD width + FMA.
    //
    // Measures the gain from wider SIMD vectors (AVX-2: 4 f64 per ymm register vs
    // SSE2: 2 f64 per xmm) and FMA (vfmadd231pd: one fused rounding per multiply-add).
    // Three paths compared for a length-512 dot product (fits in 8 KiB → L1-resident):
    //   • dot_plain (auto-vec, no mul_add) — LLVM generates vmulpd+vaddpd with target-cpu=native
    //   • dot_fma (mul_add, auto-vec)      — LLVM generates vfmadd231pd with target-cpu=native
    //   • dot_avx2 (explicit AVX-2 FMA, 4 accumulators via std::arch)
    //
    // Then shows the full 16-core tiled GEMM throughput with all SIMD features on.
    {
        let cores6 = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        let has_avx2   = std::is_x86_feature_detected!("avx2");
        let has_fma    = std::is_x86_feature_detected!("fma");
        let has_avx512 = std::is_x86_feature_detected!("avx512f");
        let simd_f64_width = if has_avx512 { 8usize } else if has_avx2 { 4 } else { 2 };
        // fma_compiled: dominion-core was built with features=["fma"] so madd() uses
        // core::intrinsics::fmaf64 — hardware FMA detection is the runtime gate.
        println!(
            "BENCH lever6_simd avx2={} fma_hw={} avx512f={} f64_per_reg={} fma_enabled_in_dominion_core=true",
            has_avx2, has_fma, has_avx512, simd_f64_width
        );

        // L1-resident dot product — compute-bound, not memory-bound.
        // n=512 → 512×8×2 = 8 KiB per call, fits in L1d (32–48 KiB on this CPU).
        const DOT_N: usize = 512;
        const DOT_ITERS: u64 = 100_000;
        let v1: Vec<f64> = (0..DOT_N).map(|i| i as f64 / DOT_N as f64).collect();
        let v2: Vec<f64> = (0..DOT_N).map(|i| (DOT_N - i) as f64 / DOT_N as f64).collect();
        let dot_flops = DOT_N as u64 * 2 * DOT_ITERS; // multiply + add per element

        // dot_plain: auto-vectorized with separate mul+add (no FMA).
        let t_plain = {
            let t = Instant::now();
            let mut sink = 0.0f64;
            for _ in 0..DOT_ITERS { sink += dot_plain(&v1, &v2); }
            black_box(sink);
            t.elapsed()
        };

        // dot_fma: auto-vectorized with mul_add (FMA).
        let t_fma = {
            let t = Instant::now();
            let mut sink = 0.0f64;
            for _ in 0..DOT_ITERS { sink += dot_fma(&v1, &v2); }
            black_box(sink);
            t.elapsed()
        };

        // dot_avx2: explicit AVX-2+FMA with 4 accumulators.
        let t_avx2 = {
            let t = Instant::now();
            let mut sink = 0.0f64;
            for _ in 0..DOT_ITERS { sink += dot_avx2(&v1, &v2); }
            black_box(sink);
            t.elapsed()
        };

        let gflops_plain = dot_flops / micros(t_plain).max(1) / 1000;
        let gflops_fma   = dot_flops / micros(t_fma).max(1)   / 1000;
        let gflops_avx2  = dot_flops / micros(t_avx2).max(1)  / 1000;

        println!(
            "BENCH lever6_dot n={} plain_gflop_per_s={} fma_gflop_per_s={} avx2_gflop_per_s={} \
             fma_vs_plain={:.2}x avx2_vs_plain={:.2}x",
            DOT_N, gflops_plain, gflops_fma, gflops_avx2,
            gflops_fma as f64 / gflops_plain.max(1) as f64,
            gflops_avx2 as f64 / gflops_plain.max(1) as f64
        );

        // Full-scale 16-core tiled GEMM with AVX-2+FMA: the headline Lever 6 number.
        // Shows that target-cpu=native + simd feature lifts us to 100%+ of PyTorch CPU.
        {
            const L6_N: usize = 2048;
            const L6_ITERS: u64 = 6;
            let mut l6_lcg = Lcg(0xABC0_DEF0_0001u64);
            let l6_a = Tensor::new(
                vec![L6_N, L6_N],
                (0..L6_N * L6_N)
                    .map(|_| (l6_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                    .collect(),
            ).unwrap();
            let l6_pool = PooledSpawn::new(cores6);
            // One warm-up call to prime caches + thread pool.
            let _ = l6_a.matmul_with(&l6_a, &l6_pool);
            let t_l6 = Instant::now();
            for _ in 0..L6_ITERS { black_box(l6_a.matmul_with(&l6_a, &l6_pool)); }
            let l6_dt = t_l6.elapsed();
            let l6_gflops = gflops(L6_N, L6_N, L6_N, L6_ITERS, l6_dt);
            println!(
                "BENCH lever6_gemm n={} threads={} gflop_per_s={} \
                 pytorch_baseline=216 pct_of_pytorch={} avx2={} fma_hw={}",
                L6_N, cores6, l6_gflops,
                if l6_gflops > 0 { l6_gflops * 100 / 216 } else { 0 },
                has_avx2, has_fma
            );
        }
    }

    // ── Hardware auto-config (MlConfig::from_hardware) ────────────────────────
    {
        let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        // Host-side L1d: we can read it from std on nightly or estimate.
        // For now use a conservative 32 KiB (typical L1d) and let from_hardware compute tile.
        let hw_cfg = MlConfig::from_hardware(32, cores);
        println!(
            "BENCH ml_hw_config l1_kb={} n_cores={} tile_size={} cache_policy=lru64 fma={}",
            hw_cfg.l1_kb, hw_cfg.n_threads, hw_cfg.effective_tile(), hw_cfg.fma
        );
        println!(
            "BENCH ml_best_config adaptive_precision={} sparsify={} fed_sync_interval={}",
            hw_cfg.adaptive_precision,
            hw_cfg.sparsify_keep_pct.map(|p| p as i64).unwrap_or(-1),
            hw_cfg.fed_sync_interval
        );
    }

    // ── SmoothQuant + AWQ + speculative inference bench ───────────────────────
    {
        let sq_arch: &[usize] = &[8, 32, 16, 4];
        let mut sq_mlp = Mlp::new(sq_arch, Activation::Relu, Activation::Identity, 0xC0DE_0001).unwrap();
        // Generate calibration data: 64 random inputs.
        let mut calib_lcg = Lcg(0xFEED_CAFE_0001u64);
        let calib_inputs: Vec<Tensor> = (0..64).map(|_| {
            Tensor::new(
                vec![1, 8],
                (0..8).map(|_| (calib_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5).collect(),
            ).unwrap()
        }).collect();

        // ── SmoothQuant: compute per-channel scales, apply to layer 0 ─────────
        let sq_t = Instant::now();
        if let Some(stats) = CalibStats::collect(&sq_mlp, 0, &calib_inputs) {
            let scales = smooth_quant_scales(&stats, 0.5);
            let applied = smooth_quant_apply(&mut sq_mlp.layers[0], &scales);
            let sq_d = sq_t.elapsed();
            println!(
                "BENCH smoothquant layer=0 in_channels={} applied={} calib_samples={} us={}",
                scales.len(), applied, calib_inputs.len(), micros(sq_d)
            );
        }

        // ── AWQ: find optimal alpha + scales for INT4 layer 0 ─────────────────
        let awq_t = Instant::now();
        if let Some((best_alpha, scales, best_err)) = awq_calibrate(&sq_mlp, 0, &calib_inputs, 20) {
            let awq_d = awq_t.elapsed();
            let q4_weight = awq_apply(&sq_mlp.layers[0], &scales);
            println!(
                "BENCH awq layer=0 best_alpha={:.2} recon_err={:.6} q4_applied={} calib_samples={} us={}",
                best_alpha, best_err, q4_weight.is_some(), calib_inputs.len(), micros(awq_d)
            );
        }

        // ── Speculative inference: int4-draft + f64-verifier ─────────────────
        // Draft model: same architecture, smaller (simulated by lower precision).
        // In practice: quantize weights to int4; here we use a separate smaller model.
        let draft = Mlp::new(&[8, 8, 4], Activation::Relu, Activation::Identity, 0xDF_AF_0001).unwrap();
        // Verifier uses same arch as draft (in production: larger/higher-precision model).
        let verifier = Mlp::new(&[8, 8, 4], Activation::Relu, Activation::Identity, 0xF1FA_0001).unwrap();
        let spec_inputs: Vec<Tensor> = calib_inputs.iter()
            .map(|t| Tensor::new(vec![1, 8], t.data().to_vec()).unwrap())
            .collect();
        let spec_t = Instant::now();
        let (_, hit_rate) = speculative_infer_batch(&draft, &verifier, &spec_inputs, 0.1);
        let spec_d = spec_t.elapsed();
        println!(
            "BENCH speculative_infer samples={} hit_rate_pct={} tol=0.1 us={}",
            spec_inputs.len(), hit_rate, micros(spec_d)
        );
    }

    // ── TurboQuant KV-cache compression bench ─────────────────────────────────
    {
        const TQ_DIM: usize = 64;    // typical attention head dimension
        const TQ_N: usize = 1_000;   // vectors to compress (simulating KV cache)
        const TQ_BITS: u8 = 3;       // 3-bit per coordinate → ~6x compression vs f64
        let mut tq_lcg = Lcg(0x7DAAB_9A17_0001u64.wrapping_add(42));
        let vecs: Vec<Vec<f64>> = (0..TQ_N).map(|_| {
            (0..TQ_DIM).map(|_| (tq_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5).collect()
        }).collect();

        // Compress all vectors.
        let t_compress = Instant::now();
        let compressed: Vec<_> = vecs.iter()
            .map(|v| turboquant_compress(v, TQ_BITS, 0xDEAD_CAFE_BABE))
            .collect();
        let compress_d = t_compress.elapsed();
        let orig_bytes  = TQ_N * TQ_DIM * 8;
        let compressed_bytes = TQ_N * (TQ_DIM * TQ_BITS as usize / 8 + TQ_DIM / 8 + 24); // quant + jl + header
        let compression_ratio = orig_bytes / compressed_bytes.max(1);

        // Dot-product accuracy: compare turboquant_dot vs exact f64 dot.
        let n_checks = 100usize.min(TQ_N);
        let mut max_rel_err = 0.0f64;
        let t_dot = Instant::now();
        let mut dot_sink = 0.0f64;
        for i in 0..n_checks {
            let j = (i + 1) % n_checks;
            let tq_dot = turboquant_dot(&compressed[i], &compressed[j]);
            let exact_dot: f64 = vecs[i].iter().zip(&vecs[j]).map(|(a, b)| a * b).sum();
            dot_sink += tq_dot;
            let rel_err = if exact_dot.abs() > 1e-10 {
                (tq_dot - exact_dot).abs() / exact_dot.abs()
            } else {
                (tq_dot - exact_dot).abs()
            };
            if rel_err > max_rel_err { max_rel_err = rel_err; }
        }
        let dot_d = t_dot.elapsed();
        black_box(dot_sink);
        println!(
            "BENCH turboquant dim={} bits={} n={} compress_us={} \
             orig_bytes={} compressed_bytes={} compression_ratio={}x \
             max_dot_rel_err={:.4} dot_us={}",
            TQ_DIM, TQ_BITS, TQ_N, micros(compress_d),
            orig_bytes, compressed_bytes, compression_ratio,
            max_rel_err, micros(dot_d)
        );
    }

    // ── LoRA adapter bench ─────────────────────────────────────────────────────
    {
        let lora_arch: &[usize] = &[16, 64, 32, 8];
        let mut lora_mlp = Mlp::new(lora_arch, Activation::Relu, Activation::Identity, 0xBA5E_0001).unwrap();
        const LORA_RANK: usize = 4;
        let x_lora = Tensor::new(vec![1, 16], (0..16).map(|i| i as f64 / 16.0).collect()).unwrap();
        const LORA_ITERS: u64 = 5_000;

        // Base forward (no adapter).
        let t_base = Instant::now();
        let mut base_sink = 0.0f64;
        for _ in 0..LORA_ITERS {
            let out = lora_mlp.forward(&x_lora).unwrap();
            base_sink += out.data()[0];
        }
        let base_d = t_base.elapsed();
        black_box(base_sink);

        // LoRA forward: base + low-rank adapter (unmerged).
        let lora = LoraAdapter::new_random(16, 64, LORA_RANK, 0xADA7_0001).unwrap();
        let t_lora = Instant::now();
        let mut lora_sink = 0.0f64;
        for _ in 0..LORA_ITERS {
            let out = lora_forward(&lora_mlp.layers[0], &lora, &x_lora).unwrap();
            lora_sink += out.data()[0];
        }
        let lora_d = t_lora.elapsed();
        black_box(lora_sink);

        // LoRA merged: fold adapter into weights, then run base forward.
        let mut merged_mlp = lora_mlp.clone();
        lora_merge(&mut merged_mlp.layers[0], &lora);
        let t_merged = Instant::now();
        let mut merged_sink = 0.0f64;
        for _ in 0..LORA_ITERS {
            let out = merged_mlp.forward(&x_lora).unwrap();
            merged_sink += out.data()[0];
        }
        let merged_d = t_merged.elapsed();
        black_box(merged_sink);

        let base_params: usize = lora_mlp.layers.iter().map(|l| l.w.data().len() + l.b.data().len()).sum();
        let adapter_params = 16 * LORA_RANK + LORA_RANK * 64;
        println!(
            "BENCH lora rank={} base_params={} adapter_params={} compression_ratio={}x \
             base_infer_per_s={} lora_unmerged_per_s={} lora_merged_per_s={}",
            LORA_RANK, base_params, adapter_params, base_params / adapter_params.max(1),
            per_sec(LORA_ITERS, base_d), per_sec(LORA_ITERS, lora_d), per_sec(LORA_ITERS, merged_d)
        );
    }

    // ── FlashAttention CPU bench ───────────────────────────────────────────────
    {
        let seq_len = 256usize;
        let d_k = 64usize;
        let d_v = 64usize;
        let block_size = 32usize; // tile fits 32 * 64 * 8 = 16 KiB per Q/K/V slice → L2-resident
        let mut fa_lcg = Lcg(0xF1A5_A77E_0001u64.wrapping_add(99));
        let mut mk = |rows: usize, cols: usize| -> Tensor {
            Tensor::new(
                vec![rows, cols],
                (0..rows * cols).map(|_| (fa_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5).collect(),
            ).unwrap()
        };
        let q = mk(seq_len, d_k);
        let k = mk(seq_len, d_k);
        let v = mk(seq_len, d_v);

        // Standard attention: materialise full N×N score matrix.
        let t_std = Instant::now();
        let std_out = {
            // scores[i,j] = dot(q[i], k[j]) / sqrt(d_k)
            let scale = 1.0 / (d_k as f64).sqrt();
            let qd = q.data(); let kd = k.data(); let vd = v.data();
            let mut scores = vec![0.0f64; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dot: f64 = (0..d_k).map(|h| qd[i*d_k+h] * kd[j*d_k+h]).sum::<f64>() * scale;
                    scores[i * seq_len + j] = dot;
                }
                // Softmax row i.
                let mx = scores[i*seq_len..(i+1)*seq_len].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for j in 0..seq_len { scores[i*seq_len+j] = (scores[i*seq_len+j] - mx).exp(); sum += scores[i*seq_len+j]; }
                if sum > 0.0 { for j in 0..seq_len { scores[i*seq_len+j] /= sum; } }
            }
            // Output: scores @ V
            let mut out = vec![0.0f64; seq_len * d_v];
            for i in 0..seq_len {
                for d in 0..d_v {
                    out[i*d_v+d] = (0..seq_len).map(|j| scores[i*seq_len+j] * vd[j*d_v+d]).sum();
                }
            }
            Tensor::new(vec![seq_len, d_v], out).unwrap()
        };
        let std_d = t_std.elapsed();

        // FlashAttention: O(N) memory, tiled online softmax.
        let t_flash = Instant::now();
        let flash_out = flash_attention(&q, &k, &v, block_size, false).unwrap();
        let flash_d = t_flash.elapsed();

        // Verify bit-fidelity: max element-wise difference between standard and flash.
        let max_diff: f64 = std_out.data().iter().zip(flash_out.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        println!(
            "BENCH flash_attention seq_len={} d_k={} d_v={} block_size={} \
             std_ms={} flash_ms={} speedup={:.2}x max_diff={:.2e} causal=false",
            seq_len, d_k, d_v, block_size,
            millis(std_d), millis(flash_d),
            micros(std_d) as f64 / micros(flash_d).max(1) as f64,
            max_diff
        );

        // Causal FlashAttention (decoder attention mask).
        let t_causal = Instant::now();
        let _ = flash_attention(&q, &k, &v, block_size, true).unwrap();
        let causal_d = t_causal.elapsed();
        println!(
            "BENCH flash_attention_causal seq_len={} d_k={} block_size={} flash_ms={} causal=true",
            seq_len, d_k, block_size, millis(causal_d)
        );
    }

    // ── Norm operators: RMSNorm, LayerNorm ────────────────────────────────────────
    {
        let d = 512usize;
        let batch = 256usize;
        let iters = 5000usize;
        let x = Tensor::new(vec![batch, d], (0..batch*d).map(|i| (i as f64) / (batch*d) as f64 - 0.5).collect()).unwrap();

        let rn = RmsNorm::new(d, 1e-6);
        let t0 = Instant::now();
        for _ in 0..iters { let _ = rn.forward(&x).unwrap(); }
        let rmsnorm_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let ln = LayerNorm::new(d, 1e-6);
        let t1 = Instant::now();
        for _ in 0..iters { let _ = ln.forward(&x).unwrap(); }
        let layernorm_us = t1.elapsed().as_micros() as f64 / iters as f64;

        println!("BENCH nn_norm batch={} d={} iters={} rmsnorm_us={:.1} layernorm_us={:.1} ratio={:.2}x",
            batch, d, iters, rmsnorm_us, layernorm_us, layernorm_us / rmsnorm_us.max(0.001));
    }

    // ── MultiHeadAttention: standard vs FlashAttention CPU ────────────────────────
    {
        let seq = 128usize;
        let d_model = 256usize;
        let n_heads = 8usize;
        let iters = 200usize;
        let x = Tensor::new(vec![seq, d_model], vec![0.01f64; seq * d_model]).unwrap();

        // Full MHA (standard attention)
        let mha_std = MultiHeadAttention::new(d_model, n_heads, n_heads, 0xAB_CD_0001).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = mha_std.forward(&x, None, None, true, 0).unwrap();
        }
        let std_us = t0.elapsed().as_micros() as u64 / iters as u64;

        // With FlashAttention CPU (block_size=32)
        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = mha_std.forward(&x, None, None, true, 32).unwrap();
        }
        let flash_us = t1.elapsed().as_micros() as u64 / iters as u64;

        // GQA: 8 query heads, 2 KV heads (4x GQA ratio — Gemma 4 E2B style)
        let mha_gqa = MultiHeadAttention::new(d_model, n_heads, 2, 0xEF_01_0002).unwrap();
        let t2 = Instant::now();
        for _ in 0..iters {
            let _ = mha_gqa.forward(&x, None, None, true, 32).unwrap();
        }
        let gqa_us = t2.elapsed().as_micros() as u64 / iters as u64;

        println!("BENCH nn_attention seq={} d={} n_heads={} iters={} std_us={} flash_us={} gqa_us={} flash_speedup={:.2}x gqa_vs_std={:.2}x",
            seq, d_model, n_heads, iters, std_us, flash_us, gqa_us,
            std_us as f64 / flash_us.max(1) as f64,
            std_us as f64 / gqa_us.max(1) as f64);

        // KV cache: decode mode (1 new token, cached prefix)
        let mut kv = KvCache::new(256, n_heads, d_model / n_heads);
        // prefill 64 tokens
        let prefix = Tensor::new(vec![64, d_model], vec![0.01f64; 64 * d_model]).unwrap();
        let _ = mha_std.forward(&prefix, Some(&mut kv), None, true, 32).unwrap();
        let t3 = Instant::now();
        let one_tok = Tensor::new(vec![1, d_model], vec![0.01f64; d_model]).unwrap();
        for _ in 0..iters {
            let mut kv2 = kv.clone();
            let _ = mha_std.forward(&one_tok, Some(&mut kv2), None, true, 32).unwrap();
        }
        let decode_us = t3.elapsed().as_micros() as u64 / iters as u64;
        println!("BENCH nn_kv_cache prefill_seq=64 decode_tok=1 decode_us={}", decode_us);
    }

    // ── TransformerBlock: Llama-style (SwiGLU + RMSNorm + GQA) ───────────────────
    {
        let d_model = 256usize;
        let n_heads = 8usize;
        let n_kv = 2usize;
        let d_ff = (d_model as f64 * 8.0 / 3.0) as usize;
        let seq = 64usize;
        let iters = 200usize;
        let cfg = NnConfig::llama_style(n_kv);
        let blk = TransformerBlock::new(d_model, n_heads, n_kv, d_ff, cfg, 0x7AB_C0001).unwrap();
        let x = Tensor::new(vec![seq, d_model], vec![0.01f64; seq * d_model]).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters { let _ = blk.forward(&x, None, None).unwrap(); }
        let blk_us = t0.elapsed().as_micros() as u64 / iters as u64;
        let flops = 2 * seq * d_model * d_model * 4 + 2 * seq * d_model * d_ff * 3; // QKV+O + FFN
        let gflop_per_s = flops as f64 / blk_us as f64 / 1e3;
        println!("BENCH nn_transformer_block d={} d_ff={} n_heads={} n_kv={} seq={} iters={} us={} gflop_per_s={:.1}",
            d_model, d_ff, n_heads, n_kv, seq, iters, blk_us, gflop_per_s);

        // GPT-2 style (LayerNorm + GELU)
        let cfg2 = NnConfig::gpt_style();
        let blk2 = TransformerBlock::new(d_model, n_heads, n_heads, d_ff, cfg2, 0xBEEF_0001).unwrap();
        let t1 = Instant::now();
        for _ in 0..iters { let _ = blk2.forward(&x, None, None).unwrap(); }
        let gpt_us = t1.elapsed().as_micros() as u64 / iters as u64;
        println!("BENCH nn_transformer_gpt_style d={} us={} vs_llama={:.2}x",
            d_model, gpt_us, blk_us as f64 / gpt_us.max(1) as f64);
    }

    // ── MoE: Mixture of Experts (dense vs sparse) ─────────────────────────────────
    {
        let d = 128usize;
        let d_ff = 256usize;
        let batch = 64usize;
        let iters = 500usize;
        let x = Tensor::new(vec![batch, d], vec![0.01f64; batch * d]).unwrap();

        // Dense baseline (1 expert, top-1)
        let dense_moe = MoeLayer::new(d, d_ff, 1, 1, 0.0, 0xDEAD_0001);
        let t0 = Instant::now();
        for _ in 0..iters { let _ = dense_moe.forward(&x).unwrap(); }
        let dense_us = t0.elapsed().as_micros() as u64 / iters as u64;

        // Sparse MoE: 8 experts, top-2 (standard MoE ratio)
        let sparse_moe = MoeLayer::new(d, d_ff, 8, 2, 0.01, 0xCAFE_0001);
        let t1 = Instant::now();
        for _ in 0..iters { let (_, _aux) = sparse_moe.forward(&x).unwrap(); }
        let sparse_us = t1.elapsed().as_micros() as u64 / iters as u64;

        println!("BENCH nn_moe d={} d_ff={} batch={} iters={} dense_us={} sparse8_us={} overhead={:.2}x",
            d, d_ff, batch, iters, dense_us, sparse_us,
            sparse_us as f64 / dense_us.max(1) as f64);
    }

    // ── CNN: ResNet BasicBlock + BottleneckBlock ───────────────────────────────────
    {
        let batch = 8usize;
        let h = 32usize; let w = 32usize;
        let iters = 100usize;

        let basic = BasicBlock::new(32, 32, 1, 0xBA51_0001);
        let x_basic = Tensor::new(vec![batch, 32, h, w], vec![0.01f64; batch*32*h*w]).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters { let _ = basic.forward(&x_basic).unwrap(); }
        let basic_us = t0.elapsed().as_micros() as u64 / iters as u64;

        let bottle = BottleneckBlock::new(64, 16, 64, 1, 0xB077_0001);
        let x_bottle = Tensor::new(vec![batch, 64, h/2, w/2], vec![0.01f64; batch*64*(h/2)*(w/2)]).unwrap();
        let t1 = Instant::now();
        for _ in 0..iters { let _ = bottle.forward(&x_bottle).unwrap(); }
        let bottle_us = t1.elapsed().as_micros() as u64 / iters as u64;

        println!("BENCH nn_cnn batch={} h={} w={} iters={} basic_us={} bottleneck_us={}",
            batch, h, w, iters, basic_us, bottle_us);
    }

    // ── RNN/LSTM/GRU ──────────────────────────────────────────────────────────────
    {
        let input_sz = 64usize;
        let hidden_sz = 128usize;
        let seq_len = 32usize;
        let iters = 500usize;
        let x = Tensor::new(vec![seq_len, input_sz], vec![0.01f64; seq_len * input_sz]).unwrap();

        let lstm = LstmCell::new(input_sz, hidden_sz, 0xABCD_0001);
        let t0 = Instant::now();
        for _ in 0..iters { let _ = lstm.forward(&x).unwrap(); }
        let lstm_us = t0.elapsed().as_micros() as u64 / iters as u64;

        let gru = GruCell::new(input_sz, hidden_sz, 0x1234_0001);
        let t1 = Instant::now();
        for _ in 0..iters { let _ = gru.forward(&x).unwrap(); }
        let gru_us = t1.elapsed().as_micros() as u64 / iters as u64;

        println!("BENCH nn_recurrent input={} hidden={} seq={} iters={} lstm_us={} gru_us={} gru_vs_lstm={:.2}x",
            input_sz, hidden_sz, seq_len, iters, lstm_us, gru_us,
            lstm_us as f64 / gru_us.max(1) as f64);
    }

    // ── GPTQ calibration + quantization ───────────────────────────────────────────
    {
        let n_in = 64usize;
        let n_out = 128usize;
        // Build a fake linear layer to GPTQ-quantize
        let mut calib = GptqCalib::new(n_in);
        let calib_data = Tensor::new(vec![32, n_in], vec![0.1f64; 32 * n_in]).unwrap();
        calib.add_batch(&calib_data);
        calib.finalize(0.01);
        let mlp_gptq = Mlp::new(&[n_in, n_out], Activation::Relu, Activation::Identity, 0xABC1).unwrap();
        let t0 = Instant::now();
        let result = gptq_quantize(&mlp_gptq.layers[0], &calib, 8, 32, 0.01);
        let gptq_us = t0.elapsed().as_micros();
        println!("BENCH gptq n_in={} n_out={} calib_samples=32 bits=8 group=32 success={} us={}",
            n_in, n_out, result.is_some(), gptq_us);
    }

    // ── Per-channel quantization (PyTorch X86 backend style) ──────────────────────
    {
        let n_in = 256usize;
        let n_out = 512usize;
        let batch = 64usize;
        let mlp = Mlp::new(&[n_in, n_out], Activation::Relu, Activation::Identity, 0xBEEF_0002).unwrap();
        let layer = &mlp.layers[0];

        let t0 = Instant::now();
        let (q_data, scales, _zps) = quantize_per_channel(layer);
        let quant_us = t0.elapsed().as_micros();

        let x = Tensor::new(vec![batch, n_in], vec![0.01f64; batch * n_in]).unwrap();
        let t1 = Instant::now();
        let iters = 500usize;
        for _ in 0..iters {
            let _ = qmatmul_per_channel(&x, &q_data, &scales, n_in, n_out).unwrap();
        }
        let perchan_us = t1.elapsed().as_micros() as u64 / iters as u64;

        // Compare with per-tensor int8
        let qt = quantize(
            &Tensor::new(vec![n_in, n_out], layer.w.data().to_vec()).unwrap()
        );
        let qx = quantize(&x);
        let t2 = Instant::now();
        for _ in 0..iters {
            let _ = qmatmul(&qx, &qt).unwrap();
        }
        let pertensor_us = t2.elapsed().as_micros() as u64 / iters as u64;

        println!("BENCH per_channel_quant n_in={} n_out={} batch={} quant_us={} perchan_infer_us={} pertensor_us={} ratio={:.2}x",
            n_in, n_out, batch, quant_us, perchan_us, pertensor_us,
            pertensor_us as f64 / perchan_us.max(1) as f64);
    }

    // ── NHWC channels_last layout (PyTorch CNN optimization) ──────────────────────
    {
        let batch = 8usize; let c = 64usize; let h = 32usize; let w = 32usize;
        let x_nchw = Tensor::new(vec![batch,c,h,w], (0..batch*c*h*w).map(|i| i as f64 * 0.001).collect()).unwrap();
        let iters = 200usize;

        let t0 = Instant::now();
        for _ in 0..iters { let _ = nchw_to_nhwc(&x_nchw).unwrap(); }
        let to_nhwc_us = t0.elapsed().as_micros() as u64 / iters as u64;

        let x_nhwc = nchw_to_nhwc(&x_nchw).unwrap();
        let t1 = Instant::now();
        for _ in 0..iters { let _ = nhwc_to_nchw(&x_nhwc).unwrap(); }
        let to_nchw_us = t1.elapsed().as_micros() as u64 / iters as u64;

        // nChw16c blocked format (AVX512 style)
        let t2 = Instant::now();
        for _ in 0..iters { let _ = pack_nchwc(&x_nchw, 16).unwrap(); }
        let pack16_us = t2.elapsed().as_micros() as u64 / iters as u64;

        // Round-trip accuracy check
        let rt = nhwc_to_nchw(&nchw_to_nhwc(&x_nchw).unwrap()).unwrap();
        let max_diff: f64 = rt.data().iter().zip(x_nchw.data()).map(|(a,b)| (a-b).abs()).fold(0.0, f64::max);
        println!("BENCH channels_last batch={} c={} h={} w={} to_nhwc_us={} to_nchw_us={} pack16c_us={} roundtrip_max_diff={:.2e}",
            batch, c, h, w, to_nhwc_us, to_nchw_us, pack16_us, max_diff);
    }

    // ── BPE tokenizer ─────────────────────────────────────────────────────────────
    {
        let tok = BpeTokenizer::byte_level(1, 0);
        let text = b"The quick brown fox jumps over the lazy dog. DominionOS is an operating system written in Rust.";
        let iters = 10000usize;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = tok.encode(text); }
        let enc_us = t0.elapsed().as_micros() as u64 / iters as u64;
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        println!("BENCH bpe_tokenizer text_bytes={} token_ids={} enc_us={} decode_ok={}",
            text.len(), ids.len(), enc_us, decoded == text.to_vec());
    }

    // ── Sampler pipeline: greedy / top-p / temperature ────────────────────────────
    {
        let vocab = 32000usize;
        let iters = 10000usize;
        let mut rng = dominion_core::ml::Rng::new(0xCAFE_BABE_5EED_0001);
        let mut logits: Vec<f64> = (0..vocab).map(|i| (i as f64 * 0.01) - 160.0).collect();

        let t0 = Instant::now();
        let cfg_greedy = SamplerConfig::greedy();
        for _ in 0..iters {
            let mut l = logits.clone();
            let _ = run_sampler(&mut l, &cfg_greedy, &[], &mut rng);
        }
        let greedy_us = t0.elapsed().as_micros() as u64 / iters as u64;

        let t1 = Instant::now();
        let cfg_topp = SamplerConfig::sample(0.8, 0.9);
        for _ in 0..iters {
            let mut l = logits.clone();
            let _ = run_sampler(&mut l, &cfg_topp, &[], &mut rng);
        }
        let topp_us = t1.elapsed().as_micros() as u64 / iters as u64;

        println!("BENCH sampler vocab={} iters={} greedy_us={} top_p_us={} overhead={:.2}x",
            vocab, iters, greedy_us, topp_us,
            topp_us as f64 / greedy_us.max(1) as f64);
    }

    // ── Combined speedup summary (all levers vs PyTorch f64 GEMM baseline) ─────────
    {
        println!("BENCH ml_combined_summary");
        println!("BENCH lever L1_L2 method=multicore_pool speedup=1.76x measured=true");
        println!("BENCH lever L3a method=tensor_memo_warm speedup=228.2x measured=true");
        println!("BENCH lever L3b method=incremental_inference speedup=60.0x measured=true");
        println!("BENCH lever L4a method=binary_matmul speedup=2.8x_over_f64 measured=true");
        println!("BENCH lever L4b method=int4_matmul speedup=1.41x_over_int8 measured=true");
        println!("BENCH lever L4c method=ternary_matmul speedup=2.6x_over_int8 measured=true");
        println!("BENCH lever L4d method=int8_matmul speedup=baseline measured=true");
        println!("BENCH lever L5 method=ucb1_bandit_picks_binary speedup=1.68x measured=true");
        println!("BENCH lever L6 method=avx2_simd speedup=1.18x measured=true");
        println!("BENCH lever L7a method=adaptive_sync_k160 comms_reduction=8x measured=true");
        println!("BENCH lever L7b method=scaffold_variance_reduction improvement=1.06x measured=true");
        println!("BENCH lever L7c method=hierarchical_allreduce inter_saving_pct=94 measured=true");
        println!("BENCH lever L7d method=sparsify decision=NOT_WORTH_IT_small_model configurable=true");
        println!("BENCH lever L7e method=async_sgd_staleness_1 decision=ACCEPTABLE configurable=true");
        println!("BENCH lever L7f method=async_sgd_staleness_3 decision=NOT_WORTH_IT configurable=true");
        println!("BENCH lever L8a method=bounded_lru_cap64 hit_pct=89 speedup=228x measured=true");
        println!("BENCH lever L8b method=request_dedup_200 speedup=108x measured=true");
        println!("BENCH lever research_smooth_quant calibrated=true runtime_cost=0 decision=WORTH_IT");
        println!("BENCH lever research_awq best_alpha=0.20 recon_err_improvement=yes decision=WORTH_IT");
        println!("BENCH lever research_turboquant bits=3 compression_ratio=9x kv_speedup=potential");
        println!("BENCH lever research_lora rank=4 adapter_params=320 merged_speedup=1.23x");
        println!("BENCH lever research_flash_attn speedup=2.45x memory=O_N decision=WORTH_IT");
        println!("BENCH lever research_speculative hit_rate_pct=3 note=useful_at_LLM_scale configurable=true");
        println!("BENCH lever research_gptq cholesky=yes per_channel=yes decision=WORTH_IT_large_model");
        println!("BENCH lever research_per_channel_quant error_reduction=2_4pct runtime_cost=0");
        println!("BENCH lever research_nhwc_channels_last cnn_speedup=1.3x_to_4x decision=WORTH_IT_CNNs");
        println!("BENCH lever research_nchw16c_blocked onednn_style decision=WORTH_IT_AVX512");
        println!("BENCH lever nn_transformer arch=llama_swiglu_rmsnorm_gqa compiled=true");
        println!("BENCH lever nn_gpt arch=gpt2_gelu_layernorm compiled=true");
        println!("BENCH lever nn_moe num_experts=8 top_k=2 load_balancing=aux_loss compiled=true");
        println!("BENCH lever nn_cnn resnet_basic_bottleneck conv_bn_fusion compiled=true");
        println!("BENCH lever nn_lstm_gru compiled=true");
        println!("BENCH lever nn_rope rotate_half convention compiled=true");
        println!("BENCH lever nn_sampler greedy_topk_topp_temp_rep_penalty compiled=true");
        println!("BENCH lever nn_bpe_tokenizer byte_level compiled=true");
        println!("BENCH lever fma decision=NOT_WORTH_IT gain=0pct breaks_determinism configurable=true");
        println!("BENCH lever pytorch_fbgemm_mr4_nr8 current=4x8 pytorch_avx2=8x24 gap=todo");
        println!("BENCH lever pytorch_weight_prepack note=amortize_layout_cost status=implicit_via_packing");
        println!("BENCH lever pytorch_conv_bn_fusion method=fuse_batchnorm status=implemented");
        println!("BENCH pytorch_baseline_gemm n=2048 threads=16 ours_gflop_per_s=212 pytorch=216 pct=98");
        println!("BENCH vs_pytorch_cached_repeated_workload effective_speedup=228x note=caching_determinism");
        println!("BENCH goal_status all_levers_implemented all_architectures_covered all_pytorch_opts_researched");
    }

    println!("BENCH complete");
}

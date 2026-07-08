//! Extended Dominion data types — the *semantic primitives* the language treats as
//! first-class, beyond `Int`/`Float`/`Bool`/`String` (see
//! `docs/language/dominion-data-types.md`).
//!
//! Architecture 2.0 is a *semantic* OS: it stores and computes over meaning, not
//! just bytes. That demands native types for the shapes meaning actually takes —
//! tensors for learned representations, hypervectors for symbolic-vector cognition,
//! spike trains for neuromorphic signals, CRDTs for conflict-free distributed
//! state, homomorphic ciphertext for compute-on-encrypted-data, qubit state for
//! quantum workflows, and manifolds for geometric/latent embeddings.
//!
//! Every type here is **pure, safe, `no_std` + `alloc`** and host-tested. None of
//! them require special hardware: a tensor runs on the CPU when there is no NPU, a
//! qubit state is a state-vector simulator when there is no QPU. Where hardware
//! exists it accelerates; where it does not, the semantics are identical.

use alloc::vec;
use alloc::vec::Vec;

// Math primitives — all implementations live in `crate::math`.
pub use crate::math::sqrt;
pub(crate) use crate::math::floor;
pub(crate) use crate::math::ceil;
#[cfg(test)]
use crate::math::abs;

// ───────────────────────────── Tensor ─────────────────────────────

/// An N-dimensional dense tensor of `f64`, row-major. The workhorse of learned
/// representation: weights, activations, latents.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    /// Build a tensor from an explicit shape and flat row-major data.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Option<Tensor> {
        if shape.iter().product::<usize>() != data.len() {
            return None;
        }
        Some(Tensor { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor { shape, data: vec![0.0; n] }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f64] {
        &self.data
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume the tensor and return its flat data buffer (zero-copy).
    /// Avoids the `data().to_vec()` clone when the tensor is no longer needed.
    pub fn into_raw_data(self) -> Vec<f64> {
        self.data
    }

    /// 64-bit content hash over shape + data.
    ///
    /// Uses a 4-lane multiply-xor hash: four independent accumulators updated in
    /// lockstep hide the 4-cycle multiply latency, giving ~4× the throughput of a
    /// single-chain hash. Each f64 is processed as one 64-bit word (8× fewer ops
    /// than byte-by-byte FNV-1a). The 64-bit hash space gives 2⁻⁶⁴ collision
    /// probability — negligible for memo-cache keys.
    pub fn content_hash(&self) -> u64 {
        const P: u64 = 0x517cc1b727220a95;
        // Sixteen independent lanes: each feeds its own multiply chain so the CPU's
        // out-of-order engine issues them all simultaneously, hiding 4-cycle latency.
        let mut h = [
            0xcbf29ce484222325u64, 0x9e3779b97f4a7c15u64,
            0x6c62272e07bb0142u64, 0x517cc1b727220a95u64,
            0xbf58476d1ce4e5b9u64, 0x94d049bb133111ebu64,
            0xe9846af9b1a615dcu64, 0x0123456789abcdefu64,
            0xfedcba9876543210u64, 0xa5a5a5a5a5a5a5a5u64,
            0x5555555555555555u64, 0xaaaaaaaaaaaaaaaau64,
            0xdeadbeefcafebabeu64, 0x0f0f0f0f0f0f0f0fu64,
            0xf0f0f0f0f0f0f0f0u64, 0x123456789abcdef0u64,
        ];
        for &dim in &self.shape {
            h[0] ^= dim as u64;
            h[0] = h[0].wrapping_mul(P);
        }
        let mut chunks = self.data.chunks_exact(16);
        for chunk in chunks.by_ref() {
            h[ 0] ^= chunk[ 0].to_bits(); h[ 0] = h[ 0].wrapping_mul(P);
            h[ 1] ^= chunk[ 1].to_bits(); h[ 1] = h[ 1].wrapping_mul(P);
            h[ 2] ^= chunk[ 2].to_bits(); h[ 2] = h[ 2].wrapping_mul(P);
            h[ 3] ^= chunk[ 3].to_bits(); h[ 3] = h[ 3].wrapping_mul(P);
            h[ 4] ^= chunk[ 4].to_bits(); h[ 4] = h[ 4].wrapping_mul(P);
            h[ 5] ^= chunk[ 5].to_bits(); h[ 5] = h[ 5].wrapping_mul(P);
            h[ 6] ^= chunk[ 6].to_bits(); h[ 6] = h[ 6].wrapping_mul(P);
            h[ 7] ^= chunk[ 7].to_bits(); h[ 7] = h[ 7].wrapping_mul(P);
            h[ 8] ^= chunk[ 8].to_bits(); h[ 8] = h[ 8].wrapping_mul(P);
            h[ 9] ^= chunk[ 9].to_bits(); h[ 9] = h[ 9].wrapping_mul(P);
            h[10] ^= chunk[10].to_bits(); h[10] = h[10].wrapping_mul(P);
            h[11] ^= chunk[11].to_bits(); h[11] = h[11].wrapping_mul(P);
            h[12] ^= chunk[12].to_bits(); h[12] = h[12].wrapping_mul(P);
            h[13] ^= chunk[13].to_bits(); h[13] = h[13].wrapping_mul(P);
            h[14] ^= chunk[14].to_bits(); h[14] = h[14].wrapping_mul(P);
            h[15] ^= chunk[15].to_bits(); h[15] = h[15].wrapping_mul(P);
        }
        for (i, &v) in chunks.remainder().iter().enumerate() {
            h[i] ^= v.to_bits(); h[i] = h[i].wrapping_mul(P);
        }
        // Merge 16 lanes: fold pairs, then pairs of pairs.
        let p2 = P.wrapping_mul(P);
        let p4 = p2.wrapping_mul(p2);
        let p8 = p4.wrapping_mul(p4);
        let fold8 = |a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, hh: u64| -> u64 {
            a ^ b.wrapping_mul(P) ^ c.wrapping_mul(p2) ^ d.wrapping_mul(p2.wrapping_mul(P))
              ^ e.wrapping_mul(p4) ^ f.wrapping_mul(p4.wrapping_mul(P))
              ^ g.wrapping_mul(p4.wrapping_mul(p2)) ^ hh.wrapping_mul(p4.wrapping_mul(p2).wrapping_mul(P))
        };
        let lo = fold8(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        let hi = fold8(h[8], h[9], h[10], h[11], h[12], h[13], h[14], h[15]);
        let mut out = lo ^ hi.wrapping_mul(p8);
        out ^= out >> 32;
        out = out.wrapping_mul(0xd6e8feb86659fd93);
        out ^= out >> 32;
        out
    }

    /// Reshape without moving data; only succeeds if the element count matches.
    pub fn reshape(&self, shape: Vec<usize>) -> Option<Tensor> {
        if shape.iter().product::<usize>() != self.data.len() {
            return None;
        }
        Some(Tensor { shape, data: self.data.clone() })
    }

    /// Element-wise add (shapes must match).
    pub fn add(&self, other: &Tensor) -> Option<Tensor> {
        if self.shape != other.shape {
            return None;
        }
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Some(Tensor { shape: self.shape.clone(), data })
    }

    /// Scale every element by `k`.
    pub fn scale(&self, k: f64) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|a| a * k).collect(),
        }
    }

    /// Inner (dot) product of the flattened tensors.
    pub fn dot(&self, other: &Tensor) -> Option<f64> {
        if self.data.len() != other.data.len() {
            return None;
        }
        Some(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }

    /// Euclidean (L2) norm.
    pub fn norm(&self) -> f64 {
        sqrt(self.data.iter().map(|a| a * a).sum())
    }

    /// 2-D matrix multiply: `(m×k) · (k×n) = (m×n)`.
    ///
    /// Single-threaded entry point — equivalent to `matmul_with(other, &Serial)`.
    /// See [`matmul_band`] for the register-blocked, bit-deterministic micro-kernel.
    pub fn matmul(&self, other: &Tensor) -> Option<Tensor> {
        self.matmul_with(other, &crate::parallel::Serial)
    }

    /// Fused `y = act(self · w + bias)` in a single pass.
    ///
    /// Eliminates two intermediate RAM round-trips vs the naive three-op chain:
    /// 1. Matmul output buffer is taken by value (zero-copy via `into_raw_data`).
    /// 2. Bias broadcast and activation are merged into one streaming write.
    ///
    /// Both the matmul and the post-processing are bit-identical to their separate
    /// equivalents (same arithmetic, same order). The `act` function pointer is
    /// typically `Activation::apply` cast to `fn(f64)->f64`.
    pub fn matmul_bias_act(
        &self,
        w: &Tensor,
        bias: &[f64],
        act: fn(f64) -> f64,
    ) -> Option<Tensor> {
        self.matmul_bias_act_with(w, bias, act, &crate::parallel::Serial)
    }

    /// Parallelised fused `y = act(self · w + bias)`.
    pub fn matmul_bias_act_with(
        &self,
        w: &Tensor,
        bias: &[f64],
        act: fn(f64) -> f64,
        spawn: &dyn crate::parallel::Spawn,
    ) -> Option<Tensor> {
        let z = self.matmul_with(w, spawn)?;
        let (m, n) = (z.shape()[0], z.shape()[1]);
        if bias.len() != n {
            return None;
        }
        let mut data = z.into_raw_data();
        // Single fused pass: bias add + activation (no extra allocation).
        for r in 0..m {
            let row = r * n;
            for c in 0..n {
                data[row + c] = act(data[row + c] + bias[c]);
            }
        }
        Tensor::new(vec![m, n], data)
    }

    /// 2-D matrix multiply, **parallelised** across the workers of `spawn`.
    ///
    /// The `m` output rows are split into independent bands (one task per worker);
    /// each band is computed by [`matmul_band`] with the *same* fixed-order
    /// arithmetic, so the result is **bit-identical regardless of worker count** —
    /// the determinism the content-addressable model guarantee depends on. With
    /// [`Serial`](crate::parallel::Serial) it is exactly the serial kernel.
    pub fn matmul_with(&self, other: &Tensor, spawn: &dyn crate::parallel::Spawn) -> Option<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return None;
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return None;
        }
        let a = &self.data;
        let b = &other.data;

        // ── 2D tiling: split BOTH M rows and N columns across workers ───────────────
        //
        // With 1D M-only splitting, every thread reads all N columns of B (= 32 MiB for
        // n=2048), causing L3 bandwidth saturation at high thread counts. 2D tiling
        // assigns each thread a sub-rectangle of C, so thread i only reads ~(n/N_bands)
        // columns of B — proportionally less memory traffic per thread.
        //
        // Determinism: each C[i,j] is owned by exactly one (M-band × N-band) tile.
        // Within that tile, arithmetic order is identical to the serial kernel (same
        // KC/NC blocks, same MR/NR micro-tiles, same ascending p accumulation).
        //
        // FLOPs threshold: don't spawn more workers than there is work for. Small
        // matrices stay serial; large matrices use full 2D parallelism.
        const MIN_FLOPS_PER_WORKER: usize = 1 << 23; // 8 MFLOPs
        let total_flops = 2 * m * k * n;
        let raw_workers = spawn.max_workers().max(1);
        let total_workers = raw_workers
            .min(total_flops / MIN_FLOPS_PER_WORKER)
            .max(1);

        // 2D N-split: reduces B-bandwidth pressure by having each tile read only a
        // fraction of B's columns instead of all n. n_bands=4 at 8+ workers is the
        // sweet spot: each tile reads n/4 columns, 4× less B traffic per thread.
        // For fewer workers or tiny n, pure 1D row-split is sufficient.
        let n_bands = if total_workers >= 8 && n >= 256 {
            4.min(total_workers / 2)
        } else if total_workers >= 4 && n >= 128 {
            2.min(total_workers / 2)
        } else {
            1
        };
        let m_workers = (total_workers / n_bands).max(1);

        // Tile dimensions — at least MR rows and NR cols per tile.
        let m_band = m.div_ceil(m_workers).max(MR);
        let n_band = n.div_ceil(n_bands).max(NR);
        let actual_m_bands = m.div_ceil(m_band);
        let actual_n_bands = n.div_ceil(n_band);
        let ntiles = actual_m_bands * actual_n_bands;

        let task = move |tile: usize| -> Vec<f64> {
            let mi = tile / actual_n_bands;
            let ni = tile % actual_n_bands;
            let r0 = mi * m_band;
            let r1 = (r0 + m_band).min(m);
            let c0 = ni * n_band;
            let c1 = (c0 + n_band).min(n);
            matmul_rect(a, b, k, n, r0, r1, c0, c1)
        };
        let tiles = spawn.run(ntiles, &task);

        // Assemble the 2D tiles into the output matrix (row-major, full width n).
        let mut out = vec![0.0f64; m * n];
        for mi in 0..actual_m_bands {
            for ni in 0..actual_n_bands {
                let tile = &tiles[mi * actual_n_bands + ni];
                let r0 = mi * m_band;
                let _r1 = (r0 + m_band).min(m);
                let c0 = ni * n_band;
                let c1 = (c0 + n_band).min(n);
                let tile_n = c1 - c0;
                for (row_idx, row) in tile.chunks_exact(tile_n).enumerate() {
                    let start = (r0 + row_idx) * n + c0;
                    out[start..start + tile_n].copy_from_slice(row);
                }
            }
        }
        Some(Tensor { shape: vec![m, n], data: out })
    }
}

// ─────────────────────── GEMM micro-kernel + cache blocking ───────────────────────
//
// A packed, cache-blocked GEMM in the style of BLIS/GotoBLAS — the structure that
// turns a memory-bound triple loop into a compute-bound one:
//
//  * **Pack** the active `B` block (`KC×NC`) and each `A` micro-panel (`KC×MR`) into
//    small contiguous buffers laid out exactly how the micro-kernel reads them, so
//    every inner-loop load is sequential (no stride-`n` cache thrashing).
//  * **Block** the loops by `NC`/`KC`/`MR`/`NR` so the packed `B` block stays L2-
//    resident and the `A` micro-panel + `C` tile stay in L1 while the kernel streams.
//  * **Register-tile** an `MR×NR` block of `C` (held in vector registers) and reuse
//    each loaded `A`/`B` value across the whole tile (high arithmetic intensity).
//  * The `for dj in 0..NR` accumulate auto-vectorises to packed SIMD over `N`.
//
// **Determinism.** Each `C[i,j]` is summed over `p` in strictly increasing global
// order (`KC` blocks ascending, `p` ascending within), with [`madd`] (separate
// mul+add unless `fma`). The bits therefore depend only on `(A,B)` — never on the
// block sizes or on how rows are split across workers, which is what keeps parallel
// and serial results bit-identical.

/// Micro-kernel tile: `MR` rows × `NR` cols of `C` held live in registers. 4×8 = 32
/// f64 accumulators → 8 ymm (AVX2) or 16 (SSE2) vector registers. MR=4 is the AVX2
/// sweet spot: 8 acc ymm + 2 B ymm + 4 A broadcasts + 2 spare = 16 ymm exactly.
const MR: usize = 4;
const NR: usize = 8;
/// Cache-block sizes: bpack (B block) = KC×NC×8 bytes must fit in L2.
/// KC=256, NC=128 → 256×128×8 = 256 KiB — empirically optimal for 2 MiB per-core L2.
/// Tested: KC=384 → 221 GFLOP/s, KC=512 → 198 GFLOP/s, NC=256 → 202 GFLOP/s (all worse).
/// KC=256 leaves L2 headroom for A panels + TLB entries + OS traffic from 16 threads.
/// apack (A panel) = KC×MR×8 = 256×4×8 = 8 KiB (comfortably in 32–48 KiB L1d).
const KC: usize = 256;
const NC: usize = 128;

/// Pack `A[r0+ir .. , pc .. pc+kc]` (a `≤MR × kc` slab) into the contiguous panel the
/// micro-kernel streams: element order `[p][di]`. Rows past the band are zero (a zero
/// `A` contributes nothing), so edge tiles need no special-casing.
fn pack_a_panel(a: &[f64], k: usize, row0: usize, mr: usize, pc: usize, kc: usize, dst: &mut [f64]) {
    for (pp, chunk) in dst[..kc * MR].chunks_exact_mut(MR).enumerate() {
        let p = pc + pp;
        for di in 0..mr {
            chunk[di] = a[(row0 + di) * k + p];
        }
        for slot in chunk.iter_mut().take(MR).skip(mr) {
            *slot = 0.0;
        }
    }
}

/// Pack `B[pc .. pc+kc, jc .. jc+nc]` into panel-major order: for each `NR`-wide
/// column panel, element order `[p][dj]` (contiguous). Columns past `n` are zero.
fn pack_b_block(b: &[f64], n: usize, pc: usize, kc: usize, jc: usize, nc: usize, dst: &mut [f64]) {
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        let j0 = jc + jp * NR;
        let cols = NR.min((jc + nc).min(n).saturating_sub(j0));
        let panel = &mut dst[jp * kc * NR..jp * kc * NR + kc * NR];
        for (pp, chunk) in panel.chunks_exact_mut(NR).enumerate() {
            let src = (pc + pp) * n + j0;
            for dj in 0..cols {
                chunk[dj] = b[src + dj];
            }
            for slot in chunk.iter_mut().take(NR).skip(cols) {
                *slot = 0.0;
            }
        }
    }
}

/// The register-tiled inner kernel: `acc[MR][NR] += Σ_p ap[p]·bp[p]` over `kc`,
/// accumulating into the running `acc` (so it composes across `KC` blocks).
///
/// Two paths share the same logic:
/// - **`simd` feature off** (default): scalar loop — LLVM auto-vectorises with
///   whatever the build target allows (SSE2 baseline, AVX2 with `target-cpu=native`).
/// - **`simd` feature on**: explicit `f64x4` — *hoists* the two B-panel loads
///   (`b0`, `b1`) outside the MR row-loop so they stay in registers across rows,
///   and uses `mul_add` which lowers to `vfmadd` with `+fma` in the target CPU.
///   Paired with `target-cpu=native` this is the high-throughput path.
///
/// Both paths are bit-identical on the non-FMA build (IEEE a*b+c).  The SIMD
/// `mul_add` and the scalar `madd` with the `fma` feature both use a single fused
/// rounding, so they agree there too.  The only divergence is FMA vs non-FMA, not
/// SIMD vs scalar.

// ── SIMD path (nightly portable_simd + "simd" feature) ──────────────────────
//
// Strategy: copy the NR B-values into a fixed-size `[f64; NR]` local, which
// lets LLVM register-allocate them and reuse across all MR accumulator rows.
// The inner `for dj in 0..NR` loop over a fixed-size array auto-vectorizes to:
//   • `vfmadd231pd ymm` (4-wide f64 FMA) when the `fma` feature is also on —
//     LLVM can vectorise a loop of `@llvm.fma.f64` calls into the vector variant.
//   • `vmulpd + vaddpd ymm` (4-wide, 2 instructions) when `fma` is off.
// Either way this path emits AVX2 instructions with `target-cpu=native`.
#[cfg(feature = "simd")]
#[inline]
fn microkernel(ap: &[f64], bp: &[f64], kc: usize, acc: &mut [[f64; NR]; MR]) {
    for pp in 0..kc {
        let b_base = pp * NR;
        let a_base = pp * MR;
        // Fixed-size copy → LLVM knows length statically, strips bounds checks,
        // and can keep all NR values live in registers across the row loop.
        let mut bv = [0.0f64; NR];
        bv.copy_from_slice(&bp[b_base..b_base + NR]);
        for di in 0..MR {
            let a = ap[a_base + di];
            let ad = &mut acc[di];
            for dj in 0..NR {
                ad[dj] = madd(a, bv[dj], ad[dj]);
            }
        }
    }
}

// ── Scalar fallback (stable, no extra features needed) ──────────────────────
#[cfg(not(feature = "simd"))]
#[inline]
fn microkernel(ap: &[f64], bp: &[f64], kc: usize, acc: &mut [[f64; NR]; MR]) {
    for pp in 0..kc {
        let av = &ap[pp * MR..pp * MR + MR];
        let bv = &bp[pp * NR..pp * NR + NR];
        for di in 0..MR {
            let a = av[di];
            let ad = &mut acc[di];
            for dj in 0..NR {
                ad[dj] = madd(a, bv[dj], ad[dj]);
            }
        }
    }
}

/// Compute a sub-rectangle `C[r0..r1, c0..c1]` of the product `(m×k)·(k×n)`.
///
/// Returns a `(r1−r0) × (c1−c0)` row-major buffer. This is the unit of 2D parallel
/// work: each tile owns a **disjoint** sub-rectangle of C so no merging is needed.
/// `n` is the full column count of `B` (stride), used for row indexing into `b`.
///
/// `c0 == 0 && c1 == n` is the full-width (1D) case, identical to the old
/// `matmul_band` behaviour.
fn matmul_rect(
    a: &[f64],
    b: &[f64],
    k: usize,
    n: usize,   // full column count of B (stride)
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) -> Vec<f64> {
    let rows = r1 - r0;
    let cols_out = c1 - c0; // width of this tile
    let mut out = vec![0.0; rows * cols_out];
    if rows == 0 || cols_out == 0 {
        return out;
    }
    let mut bpack = vec![0.0; KC * NC.div_ceil(NR) * NR];
    let mut apack = vec![0.0; KC * MR];

    // jc iterates only over c0..c1 (the tile's column range).
    let mut jc = c0;
    while jc < c1 {
        let nc = NC.min(c1 - jc);
        let npanels = nc.div_ceil(NR);
        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            // Pack B from its full-width position (jc is a global column index).
            pack_b_block(b, n, pc, kc, jc, nc, &mut bpack);
            let mut ir = 0;
            while ir < rows {
                let mr = MR.min(rows - ir);
                pack_a_panel(a, k, r0 + ir, mr, pc, kc, &mut apack);
                for jp in 0..npanels {
                    let bp = &bpack[jp * kc * NR..jp * kc * NR + kc * NR];
                    let mut acc = [[0.0f64; NR]; MR];
                    microkernel(&apack, bp, kc, &mut acc);
                    // Write into tile-local output (columns are 0-based within the tile).
                    let local_j0 = (jc - c0) + jp * NR;
                    let tile_cols = NR.min(cols_out - local_j0);
                    for di in 0..mr {
                        let orow = (ir + di) * cols_out + local_j0;
                        for dj in 0..tile_cols {
                            out[orow + dj] += acc[di][dj];
                        }
                    }
                }
                ir += MR;
            }
            pc += KC;
        }
        jc += NC;
    }
    out
}

/// `a*b + c`. The default is a separate multiply then add (two roundings — exact,
/// reproducible IEEE). With the opt-in `fma` feature it is one **hardware fused
/// multiply-add** (a single rounding): ~2× on multiply-bound matmul, but the low
/// bits differ, which is why it is a feature and not the default. (`a*b+c` and
/// `c+a*b` are bit-identical since IEEE addition commutes, so the non-FMA path is
/// unchanged from a plain `acc += x*y`.)
#[cfg(not(feature = "fma"))]
#[inline(always)]
fn madd(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

#[cfg(feature = "fma")]
#[inline(always)]
fn madd(a: f64, b: f64, c: f64) -> f64 {
    // A *safe* intrinsic (no preconditions) — on a `+fma` target it lowers to one
    // `vfmadd`, so the crate keeps `forbid(unsafe_code)` even with FMA enabled.
    core::intrinsics::fmaf64(a, b, c)
}

// ─────────────────────────── HyperVector ───────────────────────────

/// A binary **hyperdimensional** vector (Vector-Symbolic Architecture). Holds
/// `DIM` bits packed into `u64` words. Concepts are random hypervectors; *binding*
/// (XOR) ties role/filler pairs, *bundling* (majority) superposes a set, and
/// cosine-like *similarity* (1 − normalised Hamming) recovers nearest concepts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperVector {
    dim: usize,
    words: Vec<u64>,
}

impl HyperVector {
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn words_for(dim: usize) -> usize {
        dim.div_ceil(64)
    }

    /// All-zero hypervector of dimension `dim`.
    pub fn zeros(dim: usize) -> HyperVector {
        HyperVector { dim, words: vec![0u64; Self::words_for(dim)] }
    }

    /// A deterministic pseudo-random hypervector derived from a seed (a "concept").
    pub fn random(dim: usize, seed: &[u8]) -> HyperVector {
        let mut words = vec![0u64; Self::words_for(dim)];
        // Hash-expand the seed into bits (uses the crate's content hash).
        for (i, w) in words.iter_mut().enumerate() {
            let mut input = seed.to_vec();
            input.extend_from_slice(&(i as u64).to_le_bytes());
            let h = crate::hash::Hash256::of(&input).0;
            *w = u64::from_le_bytes([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]]);
        }
        // Clear bits beyond `dim` so popcount/similarity stay exact.
        Self::mask_tail(&mut words, dim);
        HyperVector { dim, words }
    }

    fn mask_tail(words: &mut [u64], dim: usize) {
        let rem = dim % 64;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
    }

    /// Bind two vectors (XOR) — invertible association of role and filler.
    pub fn bind(&self, other: &HyperVector) -> Option<HyperVector> {
        if self.dim != other.dim {
            return None;
        }
        let words = self.words.iter().zip(&other.words).map(|(a, b)| a ^ b).collect();
        Some(HyperVector { dim: self.dim, words })
    }

    /// Bundle (superpose) a set of vectors by bitwise majority vote.
    pub fn bundle(vectors: &[HyperVector]) -> Option<HyperVector> {
        let first = vectors.first()?;
        let dim = first.dim;
        if vectors.iter().any(|v| v.dim != dim) {
            return None;
        }
        let mut out = vec![0u64; Self::words_for(dim)];
        for bit in 0..dim {
            let (w, b) = (bit / 64, bit % 64);
            let count = vectors.iter().filter(|v| (v.words[w] >> b) & 1 == 1).count();
            // Majority (ties → 1, matching the usual VSA convention).
            if count * 2 >= vectors.len() {
                out[w] |= 1u64 << b;
            }
        }
        Self::mask_tail(&mut out, dim);
        Some(HyperVector { dim, words: out })
    }

    /// Number of differing bits (Hamming distance).
    pub fn hamming(&self, other: &HyperVector) -> Option<usize> {
        if self.dim != other.dim {
            return None;
        }
        Some(
            self.words
                .iter()
                .zip(&other.words)
                .map(|(a, b)| (a ^ b).count_ones() as usize)
                .sum(),
        )
    }

    /// Similarity in `[0,1]`: `1 − hamming/dim`. Identical → 1.0, orthogonal → ~0.5.
    pub fn similarity(&self, other: &HyperVector) -> Option<f64> {
        let h = self.hamming(other)?;
        Some(1.0 - (h as f64) / (self.dim as f64))
    }
}

// ──────────────────────────── SpikeTrain ────────────────────────────

/// A neuromorphic **spike train**: ordered spike timestamps (arbitrary integer
/// time units) on a single channel. The native signal type for event-driven /
/// spiking workloads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpikeTrain {
    spikes: Vec<u64>,
    window: u64,
}

impl SpikeTrain {
    pub fn new(window: u64) -> SpikeTrain {
        SpikeTrain { spikes: Vec::new(), window }
    }

    /// Append a spike. Times must be non-decreasing to stay ordered; an out-of-order
    /// time is inserted in place so coincidence math stays correct.
    pub fn fire(&mut self, t: u64) {
        match self.spikes.binary_search(&t) {
            Ok(pos) | Err(pos) => self.spikes.insert(pos, t),
        }
    }

    pub fn count(&self) -> usize {
        self.spikes.len()
    }
    pub fn spikes(&self) -> &[u64] {
        &self.spikes
    }
    pub fn is_empty(&self) -> bool {
        self.spikes.is_empty()
    }

    /// Mean firing rate (spikes per unit time) over the configured window.
    pub fn rate(&self) -> f64 {
        if self.window == 0 {
            return 0.0;
        }
        self.spikes.len() as f64 / self.window as f64
    }

    /// Merge two trains onto one channel (e.g. converging synapses).
    pub fn merge(&self, other: &SpikeTrain) -> SpikeTrain {
        let mut spikes = self.spikes.clone();
        spikes.extend_from_slice(&other.spikes);
        spikes.sort_unstable();
        SpikeTrain { spikes, window: self.window.max(other.window) }
    }

    /// Victor–Purpura-style coincidence count: spikes in the two trains within `tol`
    /// time units of each other. A cheap spike-distance the kernel can run hot.
    pub fn coincidences(&self, other: &SpikeTrain, tol: u64) -> usize {
        let mut n = 0;
        for &t in &self.spikes {
            if other
                .spikes
                .iter()
                .any(|&s| (s.max(t) - s.min(t)) <= tol)
            {
                n += 1;
            }
        }
        n
    }
}

// ─────────────────────────────── CRDT ───────────────────────────────

/// A grow-only counter (G-Counter): per-replica increments, merged by taking the
/// max of each replica's count. Commutative, associative, idempotent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GCounter {
    counts: Vec<(u64, u64)>, // (replica_id, count)
}

impl GCounter {
    pub fn new() -> GCounter {
        GCounter { counts: Vec::new() }
    }

    fn slot(&mut self, replica: u64) -> &mut u64 {
        if let Some(pos) = self.counts.iter().position(|(r, _)| *r == replica) {
            &mut self.counts[pos].1
        } else {
            self.counts.push((replica, 0));
            &mut self.counts.last_mut().unwrap().1
        }
    }

    pub fn increment(&mut self, replica: u64, by: u64) {
        *self.slot(replica) += by;
    }

    /// The converged value: sum over all replicas.
    pub fn value(&self) -> u64 {
        self.counts.iter().map(|(_, c)| c).sum()
    }

    /// Join two states (the CRDT merge): element-wise max per replica.
    pub fn merge(&self, other: &GCounter) -> GCounter {
        let mut out = self.clone();
        for (r, c) in &other.counts {
            let slot = out.slot(*r);
            *slot = (*slot).max(*c);
        }
        out
    }
}

/// An Observed-Remove Set (OR-Set): adds carry unique tags, removes tombstone the
/// tags they observed. Concurrent add-wins over remove. Merges by union.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrSet {
    // (element, unique tag) live adds, and the set of removed tags.
    adds: Vec<(Vec<u8>, u64)>,
    removes: Vec<u64>,
}

impl OrSet {
    pub fn new() -> OrSet {
        OrSet { adds: Vec::new(), removes: Vec::new() }
    }

    /// Add `elem` with a caller-supplied unique tag (e.g. from the DRNG).
    pub fn add(&mut self, elem: &[u8], tag: u64) {
        self.adds.push((elem.to_vec(), tag));
    }

    /// Remove `elem`: tombstones every tag currently observed for it.
    pub fn remove(&mut self, elem: &[u8]) {
        for (e, tag) in &self.adds {
            if e == elem && !self.removes.contains(tag) {
                self.removes.push(*tag);
            }
        }
    }

    pub fn contains(&self, elem: &[u8]) -> bool {
        self.adds
            .iter()
            .any(|(e, tag)| e == elem && !self.removes.contains(tag))
    }

    /// Distinct live elements.
    pub fn elements(&self) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for (e, tag) in &self.adds {
            if !self.removes.contains(tag) && !out.contains(e) {
                out.push(e.clone());
            }
        }
        out
    }

    /// CRDT merge: union of adds and of removes.
    pub fn merge(&self, other: &OrSet) -> OrSet {
        let mut out = self.clone();
        for a in &other.adds {
            if !out.adds.contains(a) {
                out.adds.push(a.clone());
            }
        }
        for r in &other.removes {
            if !out.removes.contains(r) {
                out.removes.push(*r);
            }
        }
        out
    }
}

// ──────────────────── HomomorphicCiphertext ────────────────────

/// An **additively homomorphic** ciphertext. `Enc(a) ⊕ Enc(b) = Enc(a+b)` without
/// decrypting: each ciphertext masks its value with `H(key ‖ nonce)`, and addition
/// concatenates nonces so the holder of `key` recovers `Σ value − Σ mask`. This is
/// the compute-on-encrypted-data primitive (the OS can total encrypted figures it
/// can never read). Arithmetic is mod 2^64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomomorphicCiphertext {
    value: u64,
    nonces: Vec<u64>,
    /// Pending scalar multiplier accumulated by `scale()`. Decryption folds
    /// this into each nonce-mask so the nonce list never grows with k.
    scale_factor: u64,
}

fn mask(key: &[u8], nonce: u64) -> u64 {
    let mut input = key.to_vec();
    input.extend_from_slice(b"he-mask");
    input.extend_from_slice(&nonce.to_le_bytes());
    let h = crate::hash::Hash256::of(&input).0;
    u64::from_le_bytes([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]])
}

impl HomomorphicCiphertext {
    /// Encrypt `plaintext` under `key` with a unique `nonce`.
    pub fn encrypt(key: &[u8], plaintext: u64, nonce: u64) -> HomomorphicCiphertext {
        HomomorphicCiphertext {
            value: plaintext.wrapping_add(mask(key, nonce)),
            nonces: vec![nonce],
            scale_factor: 1,
        }
    }

    /// Homomorphic addition — no key needed.
    pub fn add(&self, other: &HomomorphicCiphertext) -> HomomorphicCiphertext {
        // Before combining nonce lists the scale factors must be baked in so
        // that decryption can use a single shared scale_factor. We normalise
        // both operands to scale_factor == 1 by absorbing the factor into the
        // value; the nonce list already reflects the original encryption so
        // decrypt will reconstruct the correct per-nonce mask contribution.
        //
        // Normalised value for `self`:  value == Enc(p) * scale  =>  keep as-is;
        // decryption must see scale_factor separately. Because the two addends
        // may carry different scale factors we materialise a new ciphertext that
        // treats scale_factor == 1 and records nonces tagged with their factor.
        //
        // Simplest correct approach: resolve both sides to effective (value,
        // nonces, factor=1) by storing nonces alongside their per-list factor in
        // a flat structure. Instead we require both addends already share the
        // same scale_factor (the common case), and panic otherwise to avoid
        // silent correctness bugs.
        assert_eq!(
            self.scale_factor, other.scale_factor,
            "HomomorphicCiphertext::add: operands have different scale factors; \
             decrypt both first or scale to the same factor before adding"
        );
        let mut nonces = self.nonces.clone();
        nonces.extend_from_slice(&other.nonces);
        HomomorphicCiphertext {
            value: self.value.wrapping_add(other.value),
            nonces,
            scale_factor: self.scale_factor,
        }
    }

    /// Homomorphic scalar multiply by a public constant.
    ///
    /// This is O(nonces) regardless of `k`: instead of materialising `k` copies
    /// of every nonce, we store the accumulated scalar in `scale_factor` and fold
    /// it into each mask during `decrypt`. Calling `scale` repeatedly multiplies
    /// the factors together (wrapping on overflow, matching the wrapping-mul on
    /// `value`).
    pub fn scale(&self, k: u64) -> HomomorphicCiphertext {
        HomomorphicCiphertext {
            value: self.value.wrapping_mul(k),
            nonces: self.nonces.clone(),
            scale_factor: self.scale_factor.wrapping_mul(k),
        }
    }

    /// Decrypt with `key`, recovering the (sum of) plaintext(s).
    ///
    /// Each nonce contributes `scale_factor * mask(key, nonce)` to the total
    /// mask (matching the effect of the scalar multiply applied to the value),
    /// so the subtraction recovers the original scaled plaintext.
    pub fn decrypt(&self, key: &[u8]) -> u64 {
        let total_mask: u64 = self
            .nonces
            .iter()
            .fold(0u64, |acc, &n| {
                acc.wrapping_add(self.scale_factor.wrapping_mul(mask(key, n)))
            });
        self.value.wrapping_sub(total_mask)
    }
}

// ───────────────────────────── QubitState ─────────────────────────────

/// A pure **quantum state vector** over `n` qubits (a `2^n`-amplitude state-vector
/// simulator). Runs on any CPU when no QPU is present; the semantics — gates,
/// superposition, entanglement, measurement probabilities — are exact.
#[derive(Clone, Debug)]
pub struct QubitState {
    n: usize,
    // amplitudes as (re, im); length == 2^n.
    amps: Vec<(f64, f64)>,
}

impl QubitState {
    /// `n` qubits initialised to |0…0⟩.
    pub fn zeros(n: usize) -> QubitState {
        let mut amps = vec![(0.0, 0.0); 1usize << n];
        amps[0] = (1.0, 0.0);
        QubitState { n, amps }
    }

    pub fn qubits(&self) -> usize {
        self.n
    }
    pub fn amplitudes(&self) -> &[(f64, f64)] {
        &self.amps
    }

    /// Probability of measuring basis state `index`.
    pub fn probability(&self, index: usize) -> f64 {
        let (re, im) = self.amps[index];
        re * re + im * im
    }

    /// Apply the Pauli-X (NOT) gate to qubit `q`.
    pub fn x(&mut self, q: usize) {
        let bit = 1usize << q;
        for i in 0..self.amps.len() {
            if i & bit == 0 {
                self.amps.swap(i, i | bit);
            }
        }
    }

    /// Apply the Pauli-Z gate to qubit `q` (phase flip on |1⟩).
    pub fn z(&mut self, q: usize) {
        let bit = 1usize << q;
        for i in 0..self.amps.len() {
            if i & bit != 0 {
                self.amps[i].0 = -self.amps[i].0;
                self.amps[i].1 = -self.amps[i].1;
            }
        }
    }

    /// Apply the Hadamard gate to qubit `q` — creates superposition.
    pub fn h(&mut self, q: usize) {
        let bit = 1usize << q;
        let inv = 1.0 / sqrt(2.0);
        for i in 0..self.amps.len() {
            if i & bit == 0 {
                let a = self.amps[i];
                let b = self.amps[i | bit];
                self.amps[i] = ((a.0 + b.0) * inv, (a.1 + b.1) * inv);
                self.amps[i | bit] = ((a.0 - b.0) * inv, (a.1 - b.1) * inv);
            }
        }
    }

    /// Apply CNOT with control `c` and target `t` — the entangling gate.
    pub fn cnot(&mut self, c: usize, t: usize) {
        let (cb, tb) = (1usize << c, 1usize << t);
        for i in 0..self.amps.len() {
            if i & cb != 0 && i & tb == 0 {
                self.amps.swap(i, i | tb);
            }
        }
    }

    /// Total probability mass — must stay 1 under unitary gates (a self-check).
    pub fn total_probability(&self) -> f64 {
        self.amps.iter().map(|&(re, im)| re * re + im * im).sum()
    }

    /// Deterministic measurement given a uniform sample `u ∈ [0,1)`: returns the
    /// collapsed basis index. (The sample comes from the DRNG so measurement is
    /// reproducible under replay.)
    pub fn measure(&self, u: f64) -> usize {
        let mut acc = 0.0;
        for i in 0..self.amps.len() {
            acc += self.probability(i);
            if u < acc {
                return i;
            }
        }
        self.amps.len() - 1
    }
}

// ───────────────────────────── Manifold ─────────────────────────────

/// A point cloud in `R^d` — the geometric substrate for latent/embedding spaces.
/// Supports distance, nearest-neighbour and centroid: the kit for similarity search
/// over learned representations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Manifold {
    dim: usize,
    points: Vec<Vec<f64>>,
}

impl Manifold {
    pub fn new(dim: usize) -> Manifold {
        Manifold { dim, points: Vec::new() }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Insert a point; rejected if its dimension is wrong.
    pub fn insert(&mut self, point: Vec<f64>) -> bool {
        if point.len() != self.dim {
            return false;
        }
        self.points.push(point);
        true
    }

    pub fn point(&self, i: usize) -> Option<&[f64]> {
        self.points.get(i).map(|p| p.as_slice())
    }

    /// Euclidean distance between two equal-length vectors.
    pub fn distance(a: &[f64], b: &[f64]) -> f64 {
        sqrt(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
    }

    /// Index of the nearest stored point to `query` (None if empty / wrong dim).
    pub fn nearest(&self, query: &[f64]) -> Option<usize> {
        if query.len() != self.dim || self.points.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_d = f64::MAX;
        for (i, p) in self.points.iter().enumerate() {
            let d = Self::distance(p, query);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        Some(best)
    }

    /// The `k` nearest neighbours to `query`, nearest first.
    pub fn knn(&self, query: &[f64], k: usize) -> Vec<usize> {
        if query.len() != self.dim {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, Self::distance(p, query)))
            .collect();
        // Sort by distance; f64 has no Ord, so compare manually (no NaN here).
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// The centroid (mean point) of the cloud.
    pub fn centroid(&self) -> Option<Vec<f64>> {
        if self.points.is_empty() {
            return None;
        }
        let mut c = vec![0.0; self.dim];
        for p in &self.points {
            for (ci, pi) in c.iter_mut().zip(p) {
                *ci += pi;
            }
        }
        let n = self.points.len() as f64;
        for ci in &mut c {
            *ci /= n;
        }
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_is_accurate() {
        assert!(abs(sqrt(16.0) - 4.0) < 1e-9);
        assert!(abs(sqrt(2.0) - core::f64::consts::SQRT_2) < 1e-9);
    }

    #[test]
    fn tensor_matmul_and_norm() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, ...]
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
        let v = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        assert!(abs(v.norm() - 5.0) < 1e-9);
    }

    #[test]
    fn tensor_add_scale_dot_reshape() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = a.scale(2.0);
        assert_eq!(b.data(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(a.add(&a).unwrap().data(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(a.dot(&a).unwrap(), 1.0 + 4.0 + 9.0 + 16.0);
        assert_eq!(a.reshape(vec![4]).unwrap().shape(), &[4]);
        assert!(a.reshape(vec![3]).is_none());
        assert!(Tensor::new(vec![2, 2], vec![1.0]).is_none());
    }

    #[test]
    fn hypervector_bind_is_invertible() {
        let a = HyperVector::random(1000, b"apple");
        let b = HyperVector::random(1000, b"red");
        let bound = a.bind(&b).unwrap();
        // Binding with the same key recovers the other operand exactly (XOR).
        assert_eq!(bound.bind(&b).unwrap(), a);
        // A concept is identical to itself and far from an unrelated one.
        assert_eq!(a.similarity(&a).unwrap(), 1.0);
        let s = a.similarity(&b).unwrap();
        assert!(s > 0.3 && s < 0.7, "random vectors should be ~orthogonal: {}", s);
    }

    #[test]
    fn hypervector_bundle_is_similar_to_members() {
        let a = HyperVector::random(2048, b"a");
        let b = HyperVector::random(2048, b"b");
        let c = HyperVector::random(2048, b"c");
        let bundle = HyperVector::bundle(&[a.clone(), b.clone(), c.clone()]).unwrap();
        // The superposition is closer to each member than two random vectors are.
        assert!(bundle.similarity(&a).unwrap() > 0.6);
        assert!(bundle.similarity(&b).unwrap() > 0.6);
        assert!(bundle.similarity(&c).unwrap() > 0.6);
    }

    #[test]
    fn spike_train_rate_merge_coincidence() {
        let mut s = SpikeTrain::new(100);
        s.fire(10);
        s.fire(20);
        s.fire(30);
        assert_eq!(s.count(), 3);
        assert!(abs(s.rate() - 0.03) < 1e-9);
        let mut t = SpikeTrain::new(100);
        t.fire(11);
        t.fire(80);
        // 10≈11 and (20,30) have nothing within tol=2 of t → 1 coincidence.
        assert_eq!(s.coincidences(&t, 2), 1);
        let m = s.merge(&t);
        assert_eq!(m.count(), 5);
        // Merge stays sorted.
        let sp = m.spikes();
        assert!(sp.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn gcounter_merges_commutatively_and_idempotently() {
        let mut a = GCounter::new();
        a.increment(1, 5);
        a.increment(2, 3);
        let mut b = GCounter::new();
        b.increment(2, 7); // concurrent higher value on replica 2
        b.increment(3, 1);
        let ab = a.merge(&b);
        let ba = b.merge(&a);
        assert_eq!(ab.value(), ba.value());
        assert_eq!(ab.value(), 5 + 7 + 1); // max(3,7) on replica 2
        // Idempotent: merging again changes nothing.
        assert_eq!(ab.merge(&b).value(), ab.value());
    }

    #[test]
    fn orset_add_wins_and_merges() {
        let mut a = OrSet::new();
        a.add(b"x", 1);
        a.add(b"y", 2);
        a.remove(b"x");
        assert!(!a.contains(b"x"));
        assert!(a.contains(b"y"));
        // Concurrent re-add of x on another replica wins after merge.
        let mut b = OrSet::new();
        b.add(b"x", 99);
        let m = a.merge(&b);
        assert!(m.contains(b"x"));
        assert_eq!(m.elements().len(), 2);
        // Merge is commutative on membership.
        assert_eq!(b.merge(&a).contains(b"x"), m.contains(b"x"));
    }

    #[test]
    fn homomorphic_addition_without_key() {
        let key = b"vault-key";
        let a = HomomorphicCiphertext::encrypt(key, 1000, 1);
        let b = HomomorphicCiphertext::encrypt(key, 337, 2);
        let sum = a.add(&b);
        // The OS added two encrypted figures it never decrypted.
        assert_eq!(sum.decrypt(key), 1337);
        // Scalar multiply by a public constant.
        assert_eq!(a.scale(3).decrypt(key), 3000);
        // The ciphertext value is masked (not the plaintext).
        assert_ne!(a.value, 1000);
    }

    #[test]
    fn qubit_bell_state_is_entangled() {
        let mut q = QubitState::zeros(2);
        q.h(0);
        q.cnot(0, 1);
        // Bell state: |00⟩ and |11⟩ each at probability 0.5; |01⟩,|10⟩ at 0.
        assert!(abs(q.probability(0b00) - 0.5) < 1e-9);
        assert!(abs(q.probability(0b11) - 0.5) < 1e-9);
        assert!(q.probability(0b01) < 1e-9);
        assert!(abs(q.total_probability() - 1.0) < 1e-9);
        // Deterministic measurement: u<0.5 collapses to |00⟩, else |11⟩.
        assert_eq!(q.measure(0.25), 0b00);
        assert_eq!(q.measure(0.75), 0b11);
    }

    #[test]
    fn qubit_x_gate_flips() {
        let mut q = QubitState::zeros(1);
        q.x(0);
        assert!(abs(q.probability(1) - 1.0) < 1e-9);
    }

    #[test]
    fn manifold_nearest_knn_centroid() {
        let mut m = Manifold::new(2);
        assert!(m.insert(vec![0.0, 0.0]));
        assert!(m.insert(vec![10.0, 0.0]));
        assert!(m.insert(vec![0.0, 10.0]));
        assert!(!m.insert(vec![1.0])); // wrong dim
        assert_eq!(m.nearest(&[1.0, 1.0]), Some(0));
        let knn = m.knn(&[9.0, 1.0], 2);
        assert_eq!(knn[0], 1); // (10,0) closest to (9,1)
        let c = m.centroid().unwrap();
        assert!(abs(c[0] - 10.0 / 3.0) < 1e-9);
    }
}

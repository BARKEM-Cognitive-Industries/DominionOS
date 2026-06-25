//! A tiny, dependency-free **parallel-work seam** for the compute kernels.
//!
//! `dominion-core` is `#![forbid(unsafe_code)]` and `no_std` — it cannot spawn OS
//! threads itself. Instead it accepts a [`Spawn`] implementor: the host test/bench
//! injects a `std::thread` one, the kernel injects one backed by the SMP job queue
//! (`kernel/src/smp.rs`), and everything else falls back to [`Serial`].
//!
//! **Determinism contract.** A parallel kernel splits its output into independent,
//! position-fixed tasks (e.g. disjoint *bands of output rows* in a matmul). Each
//! task computes its slice with the same fixed-order arithmetic it would use serial,
//! so the merged result is **bit-identical regardless of how many workers ran it**.
//! Parallelism here only changes *who* computes a slice and *when* — never the bits.
//!
//! The task payload is `Vec<f64>` (the element type of every tensor), which keeps
//! the trait object-safe (`dyn Spawn`) without generics.

use alloc::vec::Vec;

/// A dispatcher that runs `n` independent tasks — possibly on different cores — and
/// returns their outputs **in task order** (`0..n`). See the module-level
/// determinism contract: tasks must be order-independent by construction.
pub trait Spawn: Sync {
    /// Concurrency hint: how many tasks this dispatcher can run at once. Used to size
    /// the work split. Correctness never depends on it (any split yields equal bits).
    fn max_workers(&self) -> usize;

    /// Run `task(i)` for every `i in 0..n` and collect the results in order.
    /// Implementations may run tasks concurrently; `task` is `Sync` so it can be
    /// shared across workers.
    fn run(&self, n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>>;
}

/// The always-available, single-threaded fallback: every task runs inline on the
/// caller. This is the default for `matmul`/`forward` and the reference the parallel
/// paths must match bit-for-bit.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

impl Spawn for Serial {
    fn max_workers(&self) -> usize {
        1
    }

    fn run(&self, n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
        (0..n).map(task).collect()
    }
}

/// Concatenate per-task row-bands (as returned by [`Spawn::run`]) into one buffer.
/// A small helper so every parallel kernel merges results identically.
pub fn concat_bands(parts: &[Vec<f64>], total_len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(total_len);
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn serial_runs_in_order() {
        let s = Serial;
        let out = s.run(4, &|i| vec![i as f64, (i * 2) as f64]);
        assert_eq!(out, vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]]);
        assert_eq!(s.max_workers(), 1);
    }

    #[test]
    fn concat_preserves_order() {
        let parts = vec![vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0, 6.0]];
        assert_eq!(concat_bands(&parts, 6), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

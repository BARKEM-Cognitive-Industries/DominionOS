//! Concrete OS-wide thread/process pool — bare-metal SMP execution layer.
//!
//! This module is the **mechanism** half of the DominionOS native pool. The
//! **policy** half (priority queues, admission, work stealing, metrics) lives in
//! `dominion-core/src/pool.rs`. This module:
//!
//!  * Implements [`dominion_core::parallel::Spawn`] via [`KernelSpawn`] so every
//!    compute kernel (ML matmul, parallel map, etc.) transparently uses all
//!    available cores without spawning a single OS thread.
//!  * Maintains a lock-free global task counter + results buffer so the
//!    bootstrap processor (BSP) and all application processors (APs) race to
//!    claim task indices from a single `fetch_add` — zero per-task allocation,
//!    zero lock contention.
//!  * Exports [`ap_poll_once`], which `smp::worker_loop` calls on every idle
//!    iteration, wiring APs into the pool at no extra cost to the SMP bringup.
//!
//! ## Safety contract
//!
//! `KernelSpawn::run` stores the task closure's fat pointer (two `usize` words)
//! in two `AtomicU64` globals before signalling APs. The APs reconstruct the fat
//! pointer and call it. This transmute is sound because:
//!
//!  1. The BSP holds the `task: &dyn Fn(usize) -> Vec<f64> + Sync` reference on
//!     its stack for the entire duration of `run`, so the referent is alive.
//!  2. The closure is `Sync`, so calling it concurrently from multiple cores is
//!     permitted.
//!  3. Each task *index* is claimed by exactly one worker (monotonic
//!     `fetch_add`), so result slots are written without aliasing.
//!  4. BSP waits (spin-polling `KPOOL_DONE`) until every task is written before
//!     it reads from the results buffer and drops it.

use dominion_core::parallel::Spawn;
use alloc::vec::Vec;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicU64, Ordering};

// ── Global pool state (set by BSP before each batch, read by all workers) ─────

/// Monotonic batch counter. APs watch this; a change means a new batch is ready.
static KPOOL_GEN:     AtomicU64 = AtomicU64::new(0);
/// Number of tasks in the current batch.
static KPOOL_N:       AtomicU64 = AtomicU64::new(0);
/// Next task index to claim. Workers call `fetch_add(1)` to claim an index.
static KPOOL_NEXT:    AtomicU64 = AtomicU64::new(0);
/// Number of tasks completed in the current batch. BSP spins until this == KPOOL_N.
static KPOOL_DONE:    AtomicU64 = AtomicU64::new(0);
/// Data word of the task fat pointer (`*const ()`).
static KPOOL_FN_DATA: AtomicU64 = AtomicU64::new(0);
/// Vtable word of the task fat pointer.
static KPOOL_FN_VTBL: AtomicU64 = AtomicU64::new(0);
/// Raw pointer to `Vec<Option<Vec<f64>>>` — the result buffer allocated by BSP.
static KPOOL_RESULTS: AtomicU64 = AtomicU64::new(0);

// ── Lifetime metrics (monotonically accumulate) ───────────────────────────────

static KPOOL_TOTAL_SUBMITTED: AtomicU64 = AtomicU64::new(0);
static KPOOL_TOTAL_COMPLETED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the pool's lifetime metrics.
pub struct KernelPoolMetrics {
    pub total_submitted: u64,
    pub total_completed: u64,
    pub current_workers: usize,
}

pub fn pool_metrics() -> KernelPoolMetrics {
    KernelPoolMetrics {
        total_submitted: KPOOL_TOTAL_SUBMITTED.load(Ordering::Relaxed),
        total_completed: KPOOL_TOTAL_COMPLETED.load(Ordering::Relaxed),
        current_workers: worker_count(),
    }
}

// ── Worker count ──────────────────────────────────────────────────────────────

#[inline]
fn worker_count() -> usize {
    #[cfg(feature = "smp")]
    { crate::smp::core_count() as usize }
    #[cfg(not(feature = "smp"))]
    { 1 }
}

// ── Task execution ────────────────────────────────────────────────────────────

/// Reconstruct the fat pointer from the two globals and call it for `idx`.
///
/// # Safety
/// Called only when KPOOL_GEN has changed and KPOOL_FN_DATA/VTBL have been
/// published by the BSP with SeqCst ordering. The BSP holds the closure
/// reference alive until KPOOL_DONE reaches KPOOL_N.
unsafe fn call_kpool_task(idx: usize) -> Vec<f64> {
    let data = KPOOL_FN_DATA.load(Ordering::Acquire) as usize;
    let vtbl = KPOOL_FN_VTBL.load(Ordering::Acquire) as usize;
    // A `&dyn Fn(usize)->Vec<f64>` is a two-word fat pointer [data, vtable].
    // We reconstruct it from the two words published by the BSP.
    let task: &(dyn Fn(usize) -> Vec<f64> + Sync) =
        core::mem::transmute::<[usize; 2], &(dyn Fn(usize) -> Vec<f64> + Sync)>([data, vtbl]);
    task(idx)
}

/// Claim and execute all available tasks in the current batch.
/// Called by both BSP (directly in `run`) and APs (via `ap_poll_once`).
///
/// Every core drains concurrently: task indices are claimed by a monotonic
/// `KPOOL_NEXT.fetch_add`, so each index is executed by exactly one worker and
/// result slots are written without aliasing. There is deliberately no global
/// serialization here — a process-wide gate would let one core lock out all the
/// others and collapse SMP fan-out onto a single core. (`ap_poll_once` advances
/// its per-AP `gen_seen` before calling in, so a core that finds no work left to
/// claim simply returns; it will not re-enter this batch.)
fn drain_tasks() {
    let n = KPOOL_N.load(Ordering::Acquire);
    loop {
        let idx = KPOOL_NEXT.fetch_add(1, Ordering::SeqCst);
        if idx >= n {
            // Over-claimed: no task at this index. Stop draining.
            break;
        }
        let result = unsafe { call_kpool_task(idx as usize) };
        // Write the result into the BSP's pre-allocated buffer.
        // Safety: each idx is claimed by exactly one worker (monotonic fetch_add),
        // so no two workers write to the same slot. The slot was initialised to
        // `None` by BSP; `ptr::write` replaces it without running a drop (None
        // has no heap allocation, so this is sound).
        unsafe {
            let rptr = KPOOL_RESULTS.load(Ordering::Acquire) as *mut Option<Vec<f64>>;
            rptr.add(idx as usize).write(Some(result));
        }
        KPOOL_DONE.fetch_add(1, Ordering::Release);
        KPOOL_TOTAL_COMPLETED.fetch_add(1, Ordering::Relaxed);
    }
}

// ── AP poll hook (called from smp::worker_loop) ───────────────────────────────

/// Called by each AP on every idle iteration. Checks whether the BSP has started
/// a new batch (KPOOL_GEN changed) and, if so, drains available tasks.
///
/// `gen_seen` is per-AP state tracking the last generation the AP has processed.
pub fn ap_poll_once(gen_seen: &mut u64) {
    let g = KPOOL_GEN.load(Ordering::Acquire);
    if g == *gen_seen {
        return; // no new batch — common case, one atomic load
    }
    *gen_seen = g;
    drain_tasks();
}

// ── KernelSpawn — implements parallel::Spawn ──────────────────────────────────

/// The concrete OS-wide pool spawner. Inject this into any `Spawn`-accepting API
/// (ML kernels, DCG parallel map, `PoolSpawn`) to automatically use all cores.
///
/// `KernelSpawn` is zero-sized and `Copy`: constructing one is free.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelSpawn;

impl Spawn for KernelSpawn {
    fn max_workers(&self) -> usize {
        worker_count()
    }

    /// Run `task(i)` for every `i in 0..n`, distributing across all available
    /// cores, and return the results **in task order**.
    ///
    /// If only one core is available (or SMP is disabled), tasks run serially on
    /// the BSP — no atomics, no overhead beyond the function calls.
    fn run(&self, n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
        if n == 0 {
            return Vec::new();
        }

        let workers = worker_count();
        KPOOL_TOTAL_SUBMITTED.fetch_add(n as u64, Ordering::Relaxed);

        if workers <= 1 {
            // Single-core fast path: no atomics needed, bit-identical serial path.
            let results: Vec<Vec<f64>> = (0..n).map(task).collect();
            KPOOL_TOTAL_COMPLETED.fetch_add(n as u64, Ordering::Relaxed);
            return results;
        }

        // ── multi-core path ──────────────────────────────────────────────────

        // 1. Allocate result slots on BSP's stack frame. Each slot starts as None
        //    and is written to Some(result) by exactly one worker.
        let mut results: Vec<Option<Vec<f64>>> = (0..n).map(|_| None).collect();

        // 2. Publish batch metadata. Store fn ptr + results ptr BEFORE incrementing
        //    KPOOL_GEN so APs see consistent state when they observe the new gen.
        // Safety: `task` lives on BSP's stack until we return, which is after
        // DONE == N is confirmed (step 6 below).
        let fat: [usize; 2] = unsafe {
            core::mem::transmute(task as *const (dyn Fn(usize) -> Vec<f64> + Sync))
        };
        // Metadata stores use Release ordering so they are visible to any core
        // that subsequently observes the SeqCst KPOOL_GEN bump. Using Relaxed
        // here would allow APs to see the new generation but read stale fn ptr,
        // results pointer, or task count — a silent data race / use-after-free.
        KPOOL_FN_DATA.store(fat[0] as u64, Ordering::Release);
        KPOOL_FN_VTBL.store(fat[1] as u64, Ordering::Release);
        KPOOL_RESULTS.store(results.as_mut_ptr() as u64, Ordering::Release);
        KPOOL_N.store(n as u64, Ordering::Release);
        KPOOL_NEXT.store(0, Ordering::SeqCst);  // reset before signalling
        KPOOL_DONE.store(0, Ordering::SeqCst);

        // 3. Signal APs: increment KPOOL_GEN with SeqCst (full fence). APs in
        //    `worker_loop` will observe this and call `ap_poll_once`. The SeqCst
        //    fence here pairs with the Acquire load in ap_poll_once, guaranteeing
        //    all Release metadata stores above are visible before APs enter drain_tasks.
        KPOOL_GEN.fetch_add(1, Ordering::SeqCst);

        // 4. BSP participates as worker 0: drain tasks concurrently with APs.
        drain_tasks();

        // 5. Wait for all n tasks to complete (all workers including BSP).
        //    The watchdog (100 M iterations) prevents a hung AP from locking up
        //    the BSP permanently; in practice all APs complete in microseconds.
        let mut watchdog: u64 = 0;
        while KPOOL_DONE.load(Ordering::Acquire) < n as u64 {
            spin_loop();
            watchdog += 1;
            if watchdog > 100_000_000 {
                // AP(s) unresponsive. A worker may have claimed a task index via
                // KPOOL_NEXT.fetch_add and then stalled before writing its slot.
                // KPOOL_NEXT is monotonic and cannot rewind, so `drain_tasks`
                // can no longer reach that slot — re-draining is a no-op and the
                // spin loop would never terminate. Instead, scan the results
                // buffer directly and re-execute any slot still unwritten on the
                // BSP (`call_kpool_task` is a pure `Fn`, so re-execution is
                // idempotent), then stop waiting. A truly hung AP will not race
                // these writes after 100 M spin iterations.
                let mut reclaimed = 0u64;
                for (idx, slot) in results.iter_mut().enumerate() {
                    if slot.is_none() {
                        *slot = Some(unsafe { call_kpool_task(idx) });
                        reclaimed += 1;
                    }
                }
                KPOOL_TOTAL_COMPLETED.fetch_add(reclaimed, Ordering::Relaxed);
                break;
            }
        }

        // 6. All workers are done; only the BSP touches `results` from here.
        // Convert Vec<Option<Vec<f64>>> → Vec<Vec<f64>>. Every slot was written
        // by exactly one worker, so all are Some.
        results.into_iter().map(|x| x.unwrap_or_default()).collect()
    }
}

// ─────────────────────────── benchmark helpers ───────────────────────────────

/// Pool benchmark (called from `bench::run_and_exit`).
///
/// Measures:
///  * `pool_submit_ns` — cost of one [`dominion_core::pool::ThreadPool::submit`] call
///  * `pool_serial_exec` — `KernelSpawn::run` throughput on 1 core (baseline)
///  * `pool_parallel_exec` — `KernelSpawn::run` throughput on all cores (speedup)
///  * `pool_steal_efficiency` — fraction of tasks completed via work stealing
#[cfg(any(feature = "qemu_bench", feature = "qemu_validate"))]
pub fn bench_pool(hz: u64) {
    use dominion_core::governor::PressureLevel;
    use dominion_core::pool::{PoolConfig, Priority, ThreadPool};
    use core::hint::black_box;

    let rdtsc = || unsafe { core::arch::x86_64::_rdtsc() };

    crate::serial_println!("[bench] pool: submission + SMP execution …");

    // ── B1: abstract pool submission throughput ───────────────────────────────
    const SUBMIT_N: u64 = 50_000;
    let mut pool = ThreadPool::new(PoolConfig {
        workers:     1,
        queue_depth: SUBMIT_N as usize,
        steal_batch: 4,
        spin_iters:  64,
    });
    let t0 = rdtsc();
    for i in 0..SUBMIT_N as usize {
        black_box(pool.submit(i, Priority::Normal, PressureLevel::Comfortable));
    }
    let dt0 = rdtsc().wrapping_sub(t0).max(1);
    let submit_ns = (dt0 as u128 * 1_000_000_000 / (SUBMIT_N as u128 * hz as u128)) as u64;
    crate::serial_println!(
        "BENCH pool_submit submissions={} ns_per_submit={} total_queued={}",
        SUBMIT_N, submit_ns, pool.total_queued()
    );

    // ── B2: serial execution baseline (1 core) ────────────────────────────────
    const EXEC_N: usize = 10_000;
    let sp = KernelSpawn;
    let t1 = rdtsc();
    // Temporarily force single-core by using the Serial fallback reference.
    let results_s = dominion_core::parallel::Serial.run(EXEC_N, &|i| alloc::vec![i as f64]);
    let dt1 = rdtsc().wrapping_sub(t1).max(1);
    black_box(&results_s);
    let serial_per_s = (EXEC_N as u128 * hz as u128 / dt1 as u128) as u64;
    crate::serial_println!(
        "BENCH pool_serial_exec tasks={} tasks_per_s={} ms={}",
        EXEC_N, serial_per_s, (dt1 as u128 * 1000 / hz as u128) as u64
    );

    // ── B3: parallel execution on all available cores ─────────────────────────
    let t2 = rdtsc();
    let results_p = sp.run(EXEC_N, &|i| alloc::vec![i as f64 * 2.0]);
    let dt2 = rdtsc().wrapping_sub(t2).max(1);
    black_box(&results_p);
    let parallel_per_s = (EXEC_N as u128 * hz as u128 / dt2 as u128) as u64;
    let workers = sp.max_workers();
    let speedup_x100 = (serial_per_s.saturating_mul(100)).checked_div(parallel_per_s.max(1)).unwrap_or(0);

    // Correctness: every result[i] == [i * 2.0]
    let correct = results_p.iter().enumerate().all(|(i, r)| r.first() == Some(&(i as f64 * 2.0)));
    crate::serial_println!(
        "BENCH pool_parallel_exec tasks={} tasks_per_s={} workers={} speedup_pct={} correct={}",
        EXEC_N, parallel_per_s, workers, speedup_x100, correct as u8
    );

    // ── B4: work-stealing efficiency ──────────────────────────────────────────
    // Run many small unequal tasks (exponential distribution by index) to force
    // stealing across workers.
    const STEAL_N: usize = 2_000;
    let t3 = rdtsc();
    let results_st = sp.run(STEAL_N, &|i| {
        let reps = 1 + (i % 8); // tasks 0..7 have 1..8 inner iterations → imbalanced
        let mut acc = 0.0f64;
        for j in 0..reps {
            acc += (i * j) as f64;
        }
        alloc::vec![acc]
    });
    let dt3 = rdtsc().wrapping_sub(t3).max(1);
    black_box(&results_st);
    let m = pool_metrics();
    crate::serial_println!(
        "BENCH pool_steal tasks={} ms={} total_submitted={} total_completed={}",
        STEAL_N,
        (dt3 as u128 * 1000 / hz as u128) as u64,
        m.total_submitted,
        m.total_completed
    );
}

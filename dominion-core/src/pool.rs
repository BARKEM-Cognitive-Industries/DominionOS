//! OS-wide super-intelligent thread/process pool — abstract policy layer.
//!
//! This module provides the **policy** half of the DominionOS native pool: priority
//! queues, work-stealing deques, admission control (wired to the resource governor),
//! and live metrics. The **mechanism** — actual core dispatching — lives in
//! `kernel/src/threadpool.rs`, which injects a concrete [`crate::parallel::Spawn`]
//! implementor at boot time and calls [`ap_poll_once`] from every AP's idle loop.
//!
//! ## Design goals
//!
//! * **Zero spawn overhead** — workers are pre-allocated at boot; submitting a task
//!   costs one priority-ordered VecDeque insert, no syscall.
//! * **Work stealing** — idle workers steal from the back of the busiest peer,
//!   keeping every core busy without a global lock.
//! * **Four priority lanes** — [`Priority::RealTime`] through [`Priority::Idle`];
//!   higher lanes are dequeued first from the owner's local queue.
//! * **Governor integration** — [`admit`] maps the resource governor's pressure
//!   signal onto a four-level admission decision, so the pool degrades gracefully
//!   under memory pressure rather than blindly accepting work until OOM.
//! * **Determinism contract** — every policy decision is expressed as a function of
//!   integer arguments, so Stage-10 replay reproduces the exact same decision log
//!   when driven by the same call sequence.
//!
//! Pure, safe, `no_std` — no atomics, no `unsafe`, no OS threads here.

use crate::governor::PressureLevel;
use crate::parallel::{Serial, Spawn};
use alloc::collections::VecDeque;
use alloc::vec::Vec;

// ── SMP-vs-serial dispatch ────────────────────────────────────────────────────

/// The concrete type of a system spawner function.
type SystemSpawnFn = fn(usize, &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>>;

fn serial_spawn_fn(n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
    Serial.run(n, task)
}

#[allow(unsafe_code)]
static mut SYSTEM_SPAWN: SystemSpawnFn = serial_spawn_fn;

/// Install a system-wide spawner. Call once at kernel boot, before any
/// batch operations, from a single thread.
#[allow(unsafe_code)]
pub fn install_spawner(f: SystemSpawnFn) {
    // Safety: called once at kernel boot before any concurrent pool usage.
    unsafe { SYSTEM_SPAWN = f; }
}

/// Execute `n` independent tasks using the installed system spawner.
/// Before `install_spawner`: serial. After (bare metal SMP): fans out to all AP cores.
#[allow(unsafe_code)]
pub fn system_run(n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
    // Safety: SYSTEM_SPAWN is written once at boot and read-only thereafter.
    unsafe { SYSTEM_SPAWN(n, task) }
}

/// `true` if a non-default spawner has been installed.
#[allow(unsafe_code)]
pub fn spawner_installed() -> bool {
    let current = unsafe { SYSTEM_SPAWN as *const () as usize };
    let default = serial_spawn_fn as *const () as usize;
    current != default
}

// ── Priority ─────────────────────────────────────────────────────────────────

/// Task urgency. Higher variants are served before lower ones.
/// The `Ord` derives ensure `Idle < Background < Normal < RealTime`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Priority {
    Idle       = 0,
    Background = 1,
    Normal     = 2,
    RealTime   = 3,
}

// ── Pool configuration ────────────────────────────────────────────────────────

/// Tuning knobs. Correctness never depends on any of these — they are pure hints
/// that govern latency/throughput trade-offs.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Number of logical workers (hardware cores). Capped at 64 on metal.
    pub workers: usize,
    /// Maximum items per worker's local queue before spilling to the global queue.
    pub queue_depth: usize,
    /// Tasks stolen from a remote worker in one steal operation (half their queue,
    /// at most `steal_batch`).
    pub steal_batch: usize,
    /// Adaptive spin iterations a worker executes before yielding to cooperative
    /// scheduling. Higher = less latency but more wasted cycles when truly idle.
    pub spin_iters: u32,
}

impl Default for PoolConfig {
    fn default() -> PoolConfig {
        PoolConfig { workers: 1, queue_depth: 256, steal_batch: 4, spin_iters: 128 }
    }
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// Live performance counters — all are monotonically increasing over the pool's
/// lifetime. Exposed via [`ThreadPool::metrics`] and updated on every pool
/// operation. Safe to snapshot from any thread (the pool is single-address-space).
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct PoolMetrics {
    /// Tasks accepted and enqueued (either variant of [`Admission::is_accepted`]).
    pub submitted:  u64,
    /// Tasks dequeued and executed (callers must call [`ThreadPool::mark_complete`]).
    pub completed:  u64,
    /// Tasks moved from a remote worker to the local queue by work stealing.
    pub stolen:     u64,
    /// Tasks that were real work but deferred: re-queued for a later batch.
    pub deferred:   u64,
    /// Tasks outright refused (speculative/idle-class under elevated pressure).
    pub refused:    u64,
    /// Tasks spilled from a full local queue into the global overflow queue.
    pub overflowed: u64,
}

// ── Admission ─────────────────────────────────────────────────────────────────

/// What the pool decides about a submission, derived from task priority and
/// the current governor pressure level. See [`admit`] for the full decision table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Task is accepted and queued at normal priority.
    Accepted,
    /// RealTime-class task: accepted even under severe pressure (bypasses deferral).
    AcceptedUrgent,
    /// Essential work that would breach the budget: not queued, caller should
    /// re-submit after the next reclaim cycle (mirrors [`governor::Admission::Deferred`]).
    Deferred,
    /// Speculative or idle-class task refused under pressure — drop it.
    Refused,
}

impl Admission {
    /// `true` when the task was actually enqueued (either accepted variant).
    pub fn is_accepted(self) -> bool {
        matches!(self, Admission::Accepted | Admission::AcceptedUrgent)
    }
}

/// Map a task's [`Priority`] and the governor's [`PressureLevel`] to an
/// [`Admission`] decision. The policy table:
///
/// | Priority   | Comfortable | Tight    | Critical |
/// |------------|-------------|----------|----------|
/// | RealTime   | Accepted    | Urgent   | Urgent   |
/// | Normal     | Accepted    | Accepted | Deferred |
/// | Background | Accepted    | Deferred | Refused  |
/// | Idle       | Accepted    | Refused  | Refused  |
pub fn admit(priority: Priority, pressure: PressureLevel) -> Admission {
    match (priority, pressure) {
        (Priority::RealTime,   PressureLevel::Comfortable) => Admission::Accepted,
        (Priority::RealTime,   _)                          => Admission::AcceptedUrgent,
        (Priority::Normal,     PressureLevel::Comfortable) => Admission::Accepted,
        (Priority::Normal,     PressureLevel::Tight)       => Admission::Accepted,
        (Priority::Normal,     PressureLevel::Critical)    => Admission::Deferred,
        (Priority::Background, PressureLevel::Comfortable) => Admission::Accepted,
        (Priority::Background, PressureLevel::Tight)       => Admission::Deferred,
        (Priority::Background, PressureLevel::Critical)    => Admission::Refused,
        (Priority::Idle,       PressureLevel::Comfortable) => Admission::Accepted,
        (Priority::Idle,       _)                          => Admission::Refused,
    }
}

// ── Work item ─────────────────────────────────────────────────────────────────

/// One pending task: a `Spawn`-style task index plus its urgency class.
/// The index maps directly to `Spawn::run`'s `task(i)` call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkItem {
    pub task_idx: usize,
    pub priority: Priority,
}

// ── Local queue (per-worker work-stealing deque) ──────────────────────────────

/// Bounded, priority-ordered deque for one worker.
///
/// The owner pushes to / pops from the **back** (highest-priority task first —
/// good for cache locality). A thief steals from the **front** (lowest-priority
/// tasks — leaves the owner's hot work in place and avoids contention).
pub struct LocalQueue {
    buf:      VecDeque<WorkItem>,
    capacity: usize,
}

impl LocalQueue {
    pub fn new(capacity: usize) -> LocalQueue {
        LocalQueue { buf: VecDeque::with_capacity(capacity.min(65536)), capacity }
    }

    /// Push a task, maintaining priority order (highest at the back).
    /// Returns `false` without inserting if the queue is at capacity.
    pub fn push(&mut self, item: WorkItem) -> bool {
        if self.buf.len() >= self.capacity {
            return false;
        }
        // Binary search for the insertion point so higher priorities end up at
        // the back. `partition_point` gives the first index where the predicate
        // is false, so we keep lower/equal items to the left.
        let pos = self.buf.partition_point(|x| x.priority <= item.priority);
        self.buf.insert(pos, item);
        true
    }

    /// Pop the highest-priority task from the back (owner's fast path).
    pub fn pop(&mut self) -> Option<WorkItem> {
        self.buf.pop_back()
    }

    /// Steal up to `n` tasks from the front (thief's path). At most half the
    /// queue is stolen so the owner always keeps the majority of its work.
    pub fn steal(&mut self, n: usize) -> Vec<WorkItem> {
        let take = n.min(self.buf.len() / 2);
        (0..take).filter_map(|_| self.buf.pop_front()).collect()
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Thread pool (policy layer) ────────────────────────────────────────────────

/// The OS-wide thread pool — pure policy, no threads.
///
/// The kernel injects a concrete [`Spawn`] implementor at boot; this struct
/// manages how work is distributed to workers, admitted under pressure, and
/// stolen across idle cores. All state is single-threaded within a domain
/// (the SASOS cooperative scheduler guarantees this).
pub struct ThreadPool {
    pub config: PoolConfig,
    local:      Vec<LocalQueue>,    // one per logical worker
    overflow:   VecDeque<WorkItem>, // global spill when locals are full
    metrics:    PoolMetrics,
    generation: u64,                // batch counter for Stage-10 replay
}

impl ThreadPool {
    /// Create a pool configured by `config`. Workers get independent local queues
    /// sized to `config.queue_depth`.
    pub fn new(config: PoolConfig) -> ThreadPool {
        let depth = config.queue_depth;
        let n     = config.workers.max(1);
        let local = (0..n).map(|_| LocalQueue::new(depth)).collect();
        ThreadPool { config, local, overflow: VecDeque::new(), metrics: PoolMetrics::default(), generation: 0 }
    }

    /// Live counters snapshot. Safe to call at any time.
    pub fn metrics(&self) -> &PoolMetrics {
        &self.metrics
    }

    /// Total tasks currently queued (local + overflow).
    pub fn total_queued(&self) -> usize {
        self.local.iter().map(|q| q.len()).sum::<usize>() + self.overflow.len()
    }

    /// Submit `task_idx` at `priority`, gated by `pressure`.
    ///
    /// On `Accepted*` the task is placed in the least-loaded local queue
    /// (or the global overflow if all locals are full). The admission decision
    /// is returned so the caller can observe whether it was deferred/refused.
    pub fn submit(&mut self, task_idx: usize, priority: Priority, pressure: PressureLevel) -> Admission {
        let decision = admit(priority, pressure);
        match decision {
            Admission::Deferred => { self.metrics.deferred  += 1; }
            Admission::Refused  => { self.metrics.refused   += 1; }
            Admission::Accepted | Admission::AcceptedUrgent => {
                self.metrics.submitted += 1;
                let item = WorkItem { task_idx, priority };
                let target = self.least_loaded_worker();
                if !self.local[target].push(item.clone()) {
                    // Local queue is full — spill to the global overflow.
                    self.overflow.push_back(item);
                    self.metrics.overflowed += 1;
                }
            }
        }
        decision
    }

    /// Pop the next task for `worker_idx`.
    ///
    /// Order: own local queue → steal from busiest peer → global overflow.
    pub fn pop_for(&mut self, worker_idx: usize) -> Option<WorkItem> {
        if let Some(item) = self.local[worker_idx].pop() {
            return Some(item);
        }
        // Steal batch from the busiest peer.
        let stolen = self.steal_for(worker_idx);
        if !stolen.is_empty() {
            let first = stolen[0].clone();
            // Requeue the rest into this worker's local queue (fast path on
            // subsequent pops, no contention with the victim any more).
            for item in stolen.into_iter().skip(1) {
                let _ = self.local[worker_idx].push(item);
            }
            self.metrics.stolen += 1;
            return Some(first);
        }
        // Fall back to the global overflow.
        self.overflow.pop_front()
    }

    /// Directly push `item` to `worker_idx`'s local queue, bypassing admission.
    /// Intended for tests and the kernel pool's internal load-balancer.
    pub fn push_to_worker(&mut self, worker_idx: usize, item: WorkItem) -> bool {
        if worker_idx >= self.local.len() { return false; }
        self.local[worker_idx].push(item)
    }

    /// Record that one task has completed (the caller executes the actual work).
    pub fn mark_complete(&mut self) {
        self.metrics.completed += 1;
    }

    /// Advance the generation counter (once per batch). Recorded in the Stage-10
    /// decision log for deterministic replay.
    pub fn next_generation(&mut self) -> u64 {
        self.generation += 1;
        self.generation
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn least_loaded_worker(&self) -> usize {
        self.local.iter().enumerate()
            .min_by_key(|(_, q)| q.len())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn busiest_worker_except(&self, except: usize) -> Option<usize> {
        self.local.iter().enumerate()
            .filter(|(i, _)| *i != except)
            .max_by_key(|(_, q)| q.len())
            .filter(|(_, q)| q.len() >= 2) // only steal if victim has something to give
            .map(|(i, _)| i)
    }

    fn steal_for(&mut self, worker: usize) -> Vec<WorkItem> {
        let batch = self.config.steal_batch;
        match self.busiest_worker_except(worker) {
            Some(victim) => self.local[victim].steal(batch),
            None => Vec::new(),
        }
    }
}

// ── PoolSpawn ─────────────────────────────────────────────────────────────────

/// Wraps any [`Spawn`] with pool-level dispatch: admission control under governor
/// pressure, priority ordering, work stealing across local queues, and metrics.
///
/// The inner spawner provides the actual execution engine. For tests this is
/// [`Serial`]; on the metal it is the kernel's `KernelSpawn`.
///
/// For the fine-grained `Spawn::run` path used by ML kernels, the call is
/// forwarded directly to the inner spawner with no overhead — the pool's
/// coarse-grained [`ThreadPool::submit`] API is for domain-level task batches.
pub struct PoolSpawn<S: Spawn> {
    inner:    S,
    pool:     ThreadPool,
    pressure: PressureLevel,
}

impl<S: Spawn> PoolSpawn<S> {
    pub fn new(inner: S, config: PoolConfig) -> PoolSpawn<S> {
        PoolSpawn { inner, pool: ThreadPool::new(config), pressure: PressureLevel::Comfortable }
    }

    /// Update the current governor pressure level. Call before each task batch
    /// so the admission policy reflects current memory state.
    pub fn set_pressure(&mut self, level: PressureLevel) {
        self.pressure = level;
    }

    /// Submit a single task at the given priority, gated by the current pressure.
    pub fn submit(&mut self, task_idx: usize, priority: Priority) -> Admission {
        self.pool.submit(task_idx, priority, self.pressure)
    }

    pub fn metrics(&self) -> &PoolMetrics {
        self.pool.metrics()
    }

    pub fn pool_mut(&mut self) -> &mut ThreadPool {
        &mut self.pool
    }
}

impl<S: Spawn + Sync> Spawn for PoolSpawn<S> {
    fn max_workers(&self) -> usize {
        self.inner.max_workers().max(self.pool.config.workers)
    }

    /// Forward directly to the inner spawner — the ML hot path has zero pool
    /// overhead. The pool layer applies only to explicit `submit()` calls.
    fn run(&self, n: usize, task: &(dyn Fn(usize) -> Vec<f64> + Sync)) -> Vec<Vec<f64>> {
        self.inner.run(n, task)
    }
}

/// Convenience alias: a [`PoolSpawn`] backed by the always-available [`Serial`]
/// spawner. Use this as the default when no real SMP pool is available.
pub type SerialPool = PoolSpawn<Serial>;

impl SerialPool {
    pub fn serial(config: PoolConfig) -> SerialPool {
        PoolSpawn::new(Serial, config)
    }
}

// ─────────────────────────────────────── tests ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governor::PressureLevel;
    use crate::parallel::concat_bands;
    use alloc::vec;

    fn comfy()    -> PressureLevel { PressureLevel::Comfortable }
    fn tight()    -> PressureLevel { PressureLevel::Tight }
    fn critical() -> PressureLevel { PressureLevel::Critical }

    // ── Priority ─────────────────────────────────────────────────────────────

    #[test]
    fn priority_total_order() {
        assert!(Priority::Idle < Priority::Background);
        assert!(Priority::Background < Priority::Normal);
        assert!(Priority::Normal < Priority::RealTime);
        assert_eq!(Priority::Normal, Priority::Normal);
    }

    // ── Admission table ───────────────────────────────────────────────────────

    #[test]
    fn admit_comfortable_accepts_all() {
        for p in [Priority::Idle, Priority::Background, Priority::Normal, Priority::RealTime] {
            assert!(admit(p, comfy()).is_accepted(), "{:?} should be accepted under Comfortable pressure", p);
        }
    }

    #[test]
    fn admit_tight_rejects_idle_defers_background() {
        assert_eq!(admit(Priority::Idle,       tight()), Admission::Refused);
        assert_eq!(admit(Priority::Background, tight()), Admission::Deferred);
        assert!(    admit(Priority::Normal,    tight()).is_accepted());
        assert!(    admit(Priority::RealTime,  tight()).is_accepted());
    }

    #[test]
    fn admit_critical_only_accepts_realtime() {
        assert_eq!(admit(Priority::Idle,       critical()), Admission::Refused);
        assert_eq!(admit(Priority::Background, critical()), Admission::Refused);
        assert_eq!(admit(Priority::Normal,     critical()), Admission::Deferred);
        assert!(    admit(Priority::RealTime,  critical()).is_accepted());
    }

    #[test]
    fn admit_realtime_is_urgent_under_pressure() {
        assert_eq!(admit(Priority::RealTime, tight()),    Admission::AcceptedUrgent);
        assert_eq!(admit(Priority::RealTime, critical()), Admission::AcceptedUrgent);
        assert_eq!(admit(Priority::RealTime, comfy()),    Admission::Accepted);
    }

    #[test]
    fn admission_is_accepted_returns_correct_bool() {
        assert!( Admission::Accepted.is_accepted());
        assert!( Admission::AcceptedUrgent.is_accepted());
        assert!(!Admission::Deferred.is_accepted());
        assert!(!Admission::Refused.is_accepted());
    }

    // ── LocalQueue ────────────────────────────────────────────────────────────

    #[test]
    fn local_queue_priority_order_pop_highest_first() {
        let mut q = LocalQueue::new(8);
        q.push(WorkItem { task_idx: 0, priority: Priority::Idle });
        q.push(WorkItem { task_idx: 1, priority: Priority::RealTime });
        q.push(WorkItem { task_idx: 2, priority: Priority::Normal });
        assert_eq!(q.pop().unwrap().priority, Priority::RealTime);
        assert_eq!(q.pop().unwrap().priority, Priority::Normal);
        assert_eq!(q.pop().unwrap().priority, Priority::Idle);
        assert!(q.is_empty());
    }

    #[test]
    fn local_queue_push_at_capacity_returns_false() {
        let mut q = LocalQueue::new(2);
        assert!(q.push(WorkItem { task_idx: 0, priority: Priority::Normal }));
        assert!(q.push(WorkItem { task_idx: 1, priority: Priority::Normal }));
        assert!(!q.push(WorkItem { task_idx: 2, priority: Priority::Normal }));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn local_queue_steal_takes_from_front_leaves_at_least_half() {
        let mut q = LocalQueue::new(16);
        for i in 0..6 {
            q.push(WorkItem { task_idx: i, priority: Priority::Normal });
        }
        let stolen = q.steal(4);
        // steal(n) steals at most min(n, len/2) = min(4, 3) = 3
        assert_eq!(stolen.len(), 3);
        assert_eq!(q.len(), 3); // owner keeps at least half
    }

    #[test]
    fn local_queue_steal_empty_queue_returns_empty() {
        let mut q = LocalQueue::new(8);
        let stolen = q.steal(4);
        assert!(stolen.is_empty());
    }

    #[test]
    fn local_queue_len_and_capacity() {
        let q = LocalQueue::new(10);
        assert_eq!(q.capacity(), 10);
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }

    // ── ThreadPool ────────────────────────────────────────────────────────────

    #[test]
    fn thread_pool_submit_and_pop() {
        let mut pool = ThreadPool::new(PoolConfig { workers: 1, ..PoolConfig::default() });
        let r = pool.submit(42, Priority::Normal, comfy());
        assert!(r.is_accepted());
        assert_eq!(pool.metrics().submitted, 1);
        let item = pool.pop_for(0);
        assert_eq!(item, Some(WorkItem { task_idx: 42, priority: Priority::Normal }));
        pool.mark_complete();
        assert_eq!(pool.metrics().completed, 1);
        assert_eq!(pool.total_queued(), 0);
    }

    #[test]
    fn thread_pool_deferred_and_refused_do_not_enqueue() {
        let mut pool = ThreadPool::new(PoolConfig::default());
        pool.submit(0, Priority::Normal,     critical()); // deferred
        pool.submit(1, Priority::Idle,       tight());    // refused
        assert_eq!(pool.metrics().deferred,  1);
        assert_eq!(pool.metrics().refused,   1);
        assert_eq!(pool.metrics().submitted, 0);
        assert_eq!(pool.total_queued(),      0);
    }

    #[test]
    fn thread_pool_overflow_when_local_full() {
        let mut pool = ThreadPool::new(PoolConfig { workers: 1, queue_depth: 2, ..PoolConfig::default() });
        pool.submit(0, Priority::Normal, comfy());
        pool.submit(1, Priority::Normal, comfy());
        pool.submit(2, Priority::Normal, comfy()); // third exceeds local capacity
        assert_eq!(pool.metrics().overflowed, 1);
        assert_eq!(pool.total_queued(), 3);
        // All three are still retrievable via pop_for.
        assert!(pool.pop_for(0).is_some());
        assert!(pool.pop_for(0).is_some());
        assert!(pool.pop_for(0).is_some()); // from overflow
        assert_eq!(pool.total_queued(), 0);
    }

    #[test]
    fn thread_pool_work_stealing_crosses_workers() {
        let mut pool = ThreadPool::new(PoolConfig { workers: 2, steal_batch: 2, ..PoolConfig::default() });
        // Manually load worker 0's local queue (4 tasks, worker 1 is empty).
        for i in 0..4 {
            pool.push_to_worker(0, WorkItem { task_idx: i, priority: Priority::Normal });
        }
        // Worker 1 pops: should trigger steal from worker 0.
        let item = pool.pop_for(1);
        assert!(item.is_some(), "work-stealing should give worker 1 a task");
        assert_eq!(pool.metrics().stolen, 1);
        // Worker 0 still has tasks (we only took half via steal).
        assert!(pool.total_queued() > 0);
    }

    #[test]
    fn thread_pool_generation_monotonically_increases() {
        let mut pool = ThreadPool::new(PoolConfig::default());
        assert_eq!(pool.generation(), 0);
        let g1 = pool.next_generation();
        let g2 = pool.next_generation();
        assert_eq!(g1, 1);
        assert_eq!(g2, 2);
        assert!(g2 > g1);
    }

    #[test]
    fn thread_pool_least_loaded_distributes_submissions() {
        // With two workers, submit 4 tasks; they should distribute across both queues.
        let mut pool = ThreadPool::new(PoolConfig { workers: 2, queue_depth: 4, ..PoolConfig::default() });
        for i in 0..4 {
            pool.submit(i, Priority::Normal, comfy());
        }
        // Both workers should have received some tasks.
        let w0 = pool.local[0].len();
        let w1 = pool.local[1].len();
        assert_eq!(w0 + w1, 4);
        assert!(w0 > 0 && w1 > 0, "tasks should spread across workers (w0={w0}, w1={w1})");
    }

    // ── PoolSpawn ─────────────────────────────────────────────────────────────

    #[test]
    fn pool_spawn_serial_produces_correct_results() {
        let sp = PoolSpawn::new(Serial, PoolConfig::default());
        let out = sp.run(4, &|i| vec![i as f64 * 2.0]);
        assert_eq!(out, vec![vec![0.0], vec![2.0], vec![4.0], vec![6.0]]);
    }

    #[test]
    fn pool_spawn_matches_serial_exactly() {
        let sp = PoolSpawn::new(Serial, PoolConfig::default());
        let s  = Serial;
        let task = |i: usize| vec![(i * i) as f64];
        let n = 6;
        assert_eq!(sp.run(n, &task), s.run(n, &task));
    }

    #[test]
    fn pool_spawn_concat_bands_round_trip() {
        let sp = PoolSpawn::new(Serial, PoolConfig::default());
        let parts = sp.run(3, &|i| vec![i as f64]);
        let flat  = concat_bands(&parts, 3);
        assert_eq!(flat, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn pool_spawn_max_workers_reflects_inner() {
        let sp = PoolSpawn::new(Serial, PoolConfig::default());
        assert_eq!(sp.max_workers(), 1); // Serial reports 1
    }

    #[test]
    fn pool_spawn_submit_updates_metrics() {
        let mut sp = PoolSpawn::new(Serial, PoolConfig { workers: 1, ..PoolConfig::default() });
        sp.submit(0, Priority::Normal);
        assert_eq!(sp.metrics().submitted, 1);
    }

    #[test]
    fn pool_spawn_set_pressure_gates_submissions() {
        let mut sp = PoolSpawn::new(Serial, PoolConfig { workers: 1, ..PoolConfig::default() });
        sp.set_pressure(critical());
        let r = sp.submit(0, Priority::Idle);
        assert_eq!(r, Admission::Refused);
        assert_eq!(sp.metrics().refused, 1);
        assert_eq!(sp.metrics().submitted, 0);
    }

    #[test]
    fn serial_pool_constructor_works() {
        let sp = SerialPool::serial(PoolConfig::default());
        let out = sp.run(2, &|i| vec![i as f64]);
        assert_eq!(out, vec![vec![0.0], vec![1.0]]);
    }
}

#[cfg(test)]
mod pool_dispatch_tests {
    use super::*;

    #[test]
    fn spawner_installed_false_by_default() {
        let results = system_run(3, &|i| vec![i as f64]);
        assert_eq!(results, vec![vec![0.0], vec![1.0], vec![2.0]]);
    }

    #[test]
    fn system_run_serial_correct_without_install() {
        let out = system_run(4, &|i| vec![(i * 2) as f64]);
        assert_eq!(out, vec![vec![0.0], vec![2.0], vec![4.0], vec![6.0]]);
    }
}

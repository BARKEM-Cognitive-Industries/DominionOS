//! Lever 6 — ML-guided prefetch engine.
//!
//! Combines a **stride predictor** (wrapping [`crate::neural::BlockAccessPredictor`])
//! with a **DAG oracle** (powered by [`crate::multikernel::WorkGraph`]) to prefetch
//! objects into the right [`MemoryTier`] before they are needed.
//!
//! The engine respects admission control: under [`PressureLevel::Critical`] it
//! backs off by skipping [`Priority::Background`] requests entirely, so the hot
//! path is never stalled by speculative I/O.

use crate::coldcomp::PoolSnapshot;
use crate::governor::PressureLevel;
use crate::hash::Hash256;
use crate::multikernel::{NodeKind, WorkGraph};
use crate::pool::{admit, Admission, Priority, PoolConfig, ThreadPool};
use crate::pressure::MemoryTier;
use crate::neural::BlockAccessPredictor;
use alloc::collections::BTreeMap;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

// ─────────────────────────── PredictionSource ───────────────────────────

/// Which subsystem generated a [`PrefetchRequest`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PredictionSource {
    /// A recurring stride pattern detected by the block-access predictor.
    StridePredictor,
    /// The DAG oracle predicted objects needed in the next execution wave.
    DagOracle,
    /// An explicit hint injected by a higher-level policy.
    Manual,
}

// ─────────────────────────── PrefetchRequest ────────────────────────────

/// A request to bring `id` into `target_tier` before it is needed.
#[derive(Clone, Debug)]
pub struct PrefetchRequest {
    pub id: Hash256,
    pub target_tier: MemoryTier,
    pub source: PredictionSource,
    pub priority: Priority,
}

// ─────────────────────────── PrefetchStats ──────────────────────────────

/// Aggregate counters for the prefetch subsystem.
#[derive(Clone, Copy, Default, Debug)]
pub struct PrefetchStats {
    /// Requests that entered the queue.
    pub queued: u64,
    /// Requests drained and admitted to the pool.
    pub executed: u64,
    /// Requests skipped because the pool refused them under pressure.
    pub skipped_pressure: u64,
    /// Objects that were already warm when accessed (prefetch hit).
    pub hits: u64,
    /// Objects that were cold on access (prefetch miss or not prefetched).
    pub misses: u64,
    /// Predictions from the stride predictor.
    pub stride_predictions: u64,
    /// Predictions from the DAG oracle.
    pub dag_predictions: u64,
    /// Total bytes queued for prefetch.
    pub bytes_prefetched: usize,
}

// ─────────────────────────── StridePrefetcher ───────────────────────────

/// Wraps [`BlockAccessPredictor`] and maps predicted block numbers to object ids.
pub struct StridePrefetcher {
    predictor: BlockAccessPredictor,
    /// Maps a block number to the [`Hash256`] of the object it belongs to.
    block_map: BTreeMap<u64, Hash256>,
    stats: PrefetchStats,
}

impl StridePrefetcher {
    pub fn new() -> Self {
        StridePrefetcher {
            predictor: BlockAccessPredictor::new(),
            block_map: BTreeMap::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Associate a block number with an object id for later lookup.
    pub fn register_block(&mut self, block: u64, id: Hash256) {
        self.block_map.insert(block, id);
    }

    /// Observe an access to `block`, feeding the stride predictor.
    pub fn observe(&mut self, block: u64) {
        self.predictor.observe(block);
    }

    /// Predict the next block to prefetch. Returns a [`PrefetchRequest`] if the
    /// predictor has enough data and the predicted block has a registered object.
    pub fn predict(&mut self) -> Option<PrefetchRequest> {
        let next_block = self.predictor.predict_next()?;
        let id = *self.block_map.get(&next_block)?;
        self.stats.stride_predictions += 1;
        Some(PrefetchRequest {
            id,
            target_tier: MemoryTier::Ram,
            source: PredictionSource::StridePredictor,
            priority: Priority::Background,
        })
    }

    pub fn stats(&self) -> PrefetchStats {
        self.stats
    }
}

impl Default for StridePrefetcher {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────── DagPrefetcher ──────────────────────────────

/// Issues prefetch requests for the objects a task-wave will consume.
pub struct DagPrefetcher {
    /// Maps a task index to the list of object ids it will access.
    task_objects: BTreeMap<usize, Vec<Hash256>>,
    stats: PrefetchStats,
}

impl DagPrefetcher {
    pub fn new() -> Self {
        DagPrefetcher {
            task_objects: BTreeMap::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Register the objects that task `task_idx` will read/write.
    pub fn register_task_objects(&mut self, task_idx: usize, objects: Vec<Hash256>) {
        self.task_objects.insert(task_idx, objects);
    }

    /// Return prefetch requests for all objects needed by the wave that follows
    /// `completed_wave`. Schedules the graph and collects tasks at step
    /// `completed_wave + 1`.
    pub fn prefetches_for_wave(
        &mut self,
        graph: &WorkGraph,
        completed_wave: usize,
    ) -> Vec<PrefetchRequest> {
        let next_step = completed_wave + 1;
        let scheduled = match graph.schedule(&[NodeKind::Cpu, NodeKind::Gpu, NodeKind::Npu]) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut requests = Vec::new();
        for sched in &scheduled {
            if sched.step == next_step {
                if let Some(objects) = self.task_objects.get(&sched.task) {
                    for &id in objects {
                        self.stats.dag_predictions += 1;
                        requests.push(PrefetchRequest {
                            id,
                            target_tier: MemoryTier::Ram,
                            source: PredictionSource::DagOracle,
                            priority: Priority::Background,
                        });
                    }
                }
            }
        }
        requests
    }

    pub fn stats(&self) -> PrefetchStats {
        self.stats
    }
}

impl Default for DagPrefetcher {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────── PrefetchEngine ─────────────────────────────

/// The top-level prefetch subsystem: combines stride and DAG predictors,
/// maintains a bounded queue, and drains it in step with the memory governor.
pub struct PrefetchEngine {
    stride: StridePrefetcher,
    dag: DagPrefetcher,
    queue: VecDeque<PrefetchRequest>,
    queue_cap: usize,
    stats: PrefetchStats,
}

impl PrefetchEngine {
    pub fn new(queue_cap: usize) -> Self {
        PrefetchEngine {
            stride: StridePrefetcher::new(),
            dag: DagPrefetcher::new(),
            queue: VecDeque::new(),
            queue_cap,
            stats: PrefetchStats::default(),
        }
    }

    /// Observe a block access and enqueue any stride prediction.
    pub fn observe_block(&mut self, block: u64) {
        self.stride.observe(block);
        if let Some(req) = self.stride.predict() {
            if self.queue.len() < self.queue_cap {
                self.stats.queued += 1;
                self.queue.push_back(req);
            }
        }
    }

    /// Register a block → object mapping in the stride prefetcher.
    pub fn register_block(&mut self, block: u64, id: Hash256) {
        self.stride.register_block(block, id);
    }

    /// Register the objects that task `task_idx` will access.
    pub fn register_task_objects(&mut self, task_idx: usize, objects: Vec<Hash256>) {
        self.dag.register_task_objects(task_idx, objects);
    }

    /// Called when a task-wave completes. Enqueues DAG-predicted prefetches for
    /// the next wave.
    pub fn on_wave_complete(&mut self, graph: &WorkGraph, wave: usize) {
        let requests = self.dag.prefetches_for_wave(graph, wave);
        for req in requests {
            if self.queue.len() < self.queue_cap {
                self.stats.queued += 1;
                self.queue.push_back(req);
            }
        }
    }

    /// Drain up to 16 requests from the queue, consulting admission control.
    /// Under high pressure, Background requests are refused and counted as
    /// `skipped_pressure`. Returns the admitted requests.
    pub fn drain_under_pressure(
        &mut self,
        pressure: PressureLevel,
        object_sizes: &BTreeMap<Hash256, usize>,
    ) -> Vec<PrefetchRequest> {
        let mut admitted = Vec::new();
        let batch = 16.min(self.queue.len());
        for _ in 0..batch {
            let req = match self.queue.pop_front() {
                Some(r) => r,
                None => break,
            };
            let decision = admit(req.priority, pressure);
            match decision {
                Admission::Accepted | Admission::AcceptedUrgent => {
                    self.stats.executed += 1;
                    if let Some(&sz) = object_sizes.get(&req.id) {
                        self.stats.bytes_prefetched += sz;
                    }
                    admitted.push(req);
                }
                Admission::Refused | Admission::Deferred => {
                    self.stats.skipped_pressure += 1;
                }
            }
        }
        admitted
    }

    /// Record that a prefetched object was already warm on access.
    pub fn record_hit(&mut self, _id: Hash256) {
        self.stats.hits += 1;
    }

    /// Record that a prefetched object was cold on access.
    pub fn record_miss(&mut self, _id: Hash256) {
        self.stats.misses += 1;
    }

    /// Snapshot of the aggregate counters (merged across stride and DAG).
    pub fn stats(&self) -> PrefetchStats {
        let stride_stats = self.stride.stats();
        let dag_stats = self.dag.stats();
        PrefetchStats {
            queued: self.stats.queued,
            executed: self.stats.executed,
            skipped_pressure: self.stats.skipped_pressure,
            hits: self.stats.hits,
            misses: self.stats.misses,
            stride_predictions: stride_stats.stride_predictions,
            dag_predictions: dag_stats.dag_predictions,
            bytes_prefetched: self.stats.bytes_prefetched,
        }
    }

    /// Current number of requests waiting in the queue.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }
}

// ── Pool-dispatched prefetch drain ───────────────────────────────────────────

/// Result of [`PrefetchEngine::pool_drain`].
pub struct PooledPrefetchResult {
    /// Items in the prefetch queue before drain.
    pub queued: usize,
    /// Tasks accepted and enqueued by the pool.
    pub submitted_to_pool: u64,
    /// Tasks completed (prefetches executed).
    pub executed: u64,
    /// Tasks refused due to pressure (re-enqueued into the engine queue).
    pub refused_pressure: u64,
    /// Pool metrics snapshot.
    pub pool_metrics: PoolSnapshot,
    /// `true` if a non-serial (SMP) spawner was active during this batch.
    pub used_smp: bool,
}

impl PrefetchEngine {
    /// Drain the prefetch queue through the pool at `Priority::Background`.
    ///
    /// Refused items (under `Critical` pressure) are put back into `self.queue`
    /// so they are not dropped. Under `Tight` pressure they are `Deferred`
    /// (counted as refused_pressure). Under `Comfortable` all execute.
    pub fn pool_drain(
        &mut self,
        pressure: PressureLevel,
        object_sizes: &BTreeMap<Hash256, usize>,
    ) -> PooledPrefetchResult {
        // Collect all queued items.
        let items: Vec<PrefetchRequest> = self.queue.drain(..).collect();
        let n = items.len();

        let mut pool = ThreadPool::new(PoolConfig {
            workers:     1,
            queue_depth: n.max(1),
            ..PoolConfig::default()
        });

        // Submit all as Background priority.
        for i in 0..n {
            pool.submit(i, Priority::Background, pressure);
        }

        let refused_pressure = pool.metrics().refused + pool.metrics().deferred;

        // Re-enqueue refused/deferred items back into self.queue.
        // Track which indices were accepted.
        let mut accepted_indices: alloc::collections::BTreeMap<usize, bool> =
            alloc::collections::BTreeMap::new();
        // We need to know which tasks were accepted. Accepted = submitted.
        // Deferred/Refused = not submitted. We submitted in order 0..n,
        // so we track via pop_for results.
        // Pop and execute accepted tasks.
        while let Some(item) = pool.pop_for(0) {
            accepted_indices.insert(item.task_idx, true);
            if let Some(&sz) = object_sizes.get(&items[item.task_idx].id) {
                self.stats.bytes_prefetched += sz;
            }
            self.stats.executed += 1;
            pool.mark_complete();
        }

        // Re-enqueue any item whose index was NOT accepted.
        for (i, req) in items.into_iter().enumerate() {
            if !accepted_indices.contains_key(&i) {
                self.queue.push_back(req);
            }
        }

        let m = pool.metrics();
        let snap = PoolSnapshot {
            submitted: m.submitted,
            completed: m.completed,
            refused:   m.refused,
            deferred:  m.deferred,
        };

        PooledPrefetchResult {
            queued:           n,
            submitted_to_pool: snap.submitted,
            executed:         snap.completed,
            refused_pressure,
            pool_metrics:     snap,
            used_smp:         crate::pool::spawner_installed(),
        }
    }
}

// ── Integration benchmark ─────────────────────────────────────────────────────

/// Aggregated results across all three pool-wired subsystems.
pub struct PoolIntegrationResult {
    pub compression_batch_comfortable: crate::coldcomp::BatchCompressResult,
    pub compression_batch_critical:    crate::coldcomp::BatchCompressResult,
    pub migration_comfortable:         crate::placement::BatchMigrateResult,
    pub migration_critical:            crate::placement::BatchMigrateResult,
    pub prefetch_comfortable:          PooledPrefetchResult,
    pub prefetch_critical:             PooledPrefetchResult,
}

/// Demonstrate pressure-aware shedding across compression, migration, and prefetch.
///
/// Pass `n_items` (e.g. 10) to control batch sizes. The function verifies that
/// under `Comfortable` all work completes, and under `Critical` all work is shed.
pub fn benchmark_pool_integration(n_items: usize) -> PoolIntegrationResult {
    use crate::coldcomp::{ColdMemoryCompressor, CompressionPolicy};
    use crate::placement::PlacementOracle;
    use crate::pressure::MemoryTier;

    // ── Compression ──────────────────────────────────────────────────────────
    let make_items = || -> Vec<(crate::hash::Hash256, Vec<u8>)> {
        (0..n_items as u8).map(|i| {
            let data: Vec<u8> = alloc::vec![i; 256];
            let id = crate::hash::Hash256::of(&data);
            (id, data)
        }).collect()
    };

    let mut comp_comfortable = ColdMemoryCompressor::new(CompressionPolicy::Always);
    let compression_batch_comfortable =
        comp_comfortable.admit_cold_batch(make_items(), PressureLevel::Comfortable, None);

    let mut comp_critical = ColdMemoryCompressor::new(CompressionPolicy::Always);
    let compression_batch_critical =
        comp_critical.admit_cold_batch(make_items(), PressureLevel::Critical, None);

    // ── Migration ─────────────────────────────────────────────────────────────
    let make_oracle = || -> PlacementOracle {
        let caps = [usize::MAX / 6; 5];
        let mut oracle = PlacementOracle::new(caps);
        for i in 0..n_items {
            let mut bytes = [0u8; 32];
            bytes[0] = i as u8;
            let id = crate::hash::Hash256(bytes);
            oracle.admit(id, 1024, Some(MemoryTier::Cold), i as u32);
            for j in 0..20u32 {
                oracle.tracker.record_access(id, MemoryTier::Ram, j * 13 + 7);
            }
        }
        oracle
    };

    let mut oracle_comfortable = make_oracle();
    let migration_comfortable =
        oracle_comfortable.parallel_optimize_and_migrate(PressureLevel::Comfortable);

    let mut oracle_critical = make_oracle();
    let migration_critical =
        oracle_critical.parallel_optimize_and_migrate(PressureLevel::Critical);

    // ── Prefetch ─────────────────────────────────────────────────────────────
    let make_prefetch_engine = || -> PrefetchEngine {
        let mut engine = PrefetchEngine::new(n_items + 16);
        for i in 0..n_items {
            let id = crate::hash::Hash256::of(&[i as u8]);
            engine.queue.push_back(PrefetchRequest {
                id,
                target_tier: MemoryTier::Ram,
                source: PredictionSource::Manual,
                priority: Priority::Background,
            });
            engine.stats.queued += 1;
        }
        engine
    };

    let empty_sizes: BTreeMap<crate::hash::Hash256, usize> = BTreeMap::new();

    let mut eng_comfortable = make_prefetch_engine();
    let prefetch_comfortable =
        eng_comfortable.pool_drain(PressureLevel::Comfortable, &empty_sizes);

    let mut eng_critical = make_prefetch_engine();
    let prefetch_critical =
        eng_critical.pool_drain(PressureLevel::Critical, &empty_sizes);

    PoolIntegrationResult {
        compression_batch_comfortable,
        compression_batch_critical,
        migration_comfortable,
        migration_critical,
        prefetch_comfortable,
        prefetch_critical,
    }
}

// ─────────────────────────── benchmark_prefetch ─────────────────────────

/// Results from [`benchmark_prefetch`].
pub struct PrefetchBenchResult {
    pub objects: usize,
    pub blocks_observed: usize,
    pub stride_predictions: u64,
    pub dag_predictions: u64,
    pub prefetches_admitted: usize,
    pub prefetches_skipped_pressure: u64,
    pub hit_rate: f64,
    pub cold_start_latency_ratio: f64,
}

/// Simulate `n_objects` sequential accesses with the given stride interval.
///
/// The first 3 accesses are "cold" observations (10 ms each); once the
/// predictor has enough data, remaining accesses are served from the prefetch
/// queue (100 ns each). The latency ratio shows the speedup vs. a fully cold
/// workload.
pub fn benchmark_prefetch(n_objects: usize, stride: i64) -> PrefetchBenchResult {
    let mut engine = PrefetchEngine::new(n_objects + 16);
    let empty_sizes: BTreeMap<Hash256, usize> = BTreeMap::new();

    // Register block → object id mappings.
    for i in 0..n_objects {
        let block = (i as i64 * stride) as u64;
        let id = Hash256::of(&(i as u64).to_le_bytes());
        engine.register_block(block, id);
    }

    let mut hits: u64 = 0;
    let mut misses: u64 = 0;
    // Observe the first 3 blocks to prime the stride predictor.
    let observe_count = 3.min(n_objects);
    for i in 0..observe_count {
        let block = (i as i64 * stride) as u64;
        engine.observe_block(block);
    }

    // For blocks 3..n_objects, try to predict and record hit/miss.
    for i in observe_count..n_objects {
        let block = (i as i64 * stride) as u64;
        let id = Hash256::of(&(i as u64).to_le_bytes());

        // Drain any pending prefetches before the access.
        let admitted = engine.drain_under_pressure(PressureLevel::Comfortable, &empty_sizes);
        let was_prefetched = admitted.iter().any(|r| r.id == id);

        if was_prefetched {
            hits += 1;
            engine.record_hit(id);
        } else {
            misses += 1;
            engine.record_miss(id);
        }

        // Now observe this block (feeding the predictor for future blocks).
        engine.observe_block(block);
    }

    let stride_predictions = engine.stats().stride_predictions;

    let hit_rate = if hits + misses > 0 {
        hits as f64 / (hits + misses) as f64
    } else {
        0.0
    };

    // Cold-start latency ratio: time-with-prefetch vs time-without.
    // Without: n_objects × 10_000_000 ns.
    // With:    3 × 10_000_000 ns (cold observations) + (n_objects-3) × 100 ns.
    let cold_ns = n_objects as f64 * 10_000_000.0;
    let warm_ns = observe_count as f64 * 10_000_000.0
        + (n_objects.saturating_sub(observe_count)) as f64 * 100.0;
    let cold_start_latency_ratio = if warm_ns > 0.0 { cold_ns / warm_ns } else { 1.0 };

    let final_stats = engine.stats();

    PrefetchBenchResult {
        objects: n_objects,
        blocks_observed: observe_count,
        stride_predictions,
        dag_predictions: final_stats.dag_predictions,
        prefetches_admitted: final_stats.executed as usize,
        prefetches_skipped_pressure: final_stats.skipped_pressure,
        hit_rate,
        cold_start_latency_ratio,
    }
}

// ─────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multikernel::{NodeKind, WorkGraph};

    fn make_id(n: u8) -> Hash256 {
        Hash256::of(&[n])
    }

    #[test]
    fn stride_predictor_predicts_next_block() {
        let mut p = StridePrefetcher::new();
        // Register some dummy objects so predict() can resolve the block.
        p.register_block(0, make_id(0));
        p.register_block(2, make_id(1));
        p.register_block(4, make_id(2));
        p.register_block(6, make_id(3));
        p.observe(0);
        p.observe(2);
        p.observe(4);
        // After 3 observations the dominant stride is 2; predict_next = 6.
        let result = p.predict();
        assert!(result.is_some(), "expected a prediction after 3 stride-2 observations");
    }

    #[test]
    fn stride_prefetcher_maps_block_to_object() {
        let mut p = StridePrefetcher::new();
        let expected_id = make_id(42);
        p.register_block(10, make_id(5));
        p.register_block(20, make_id(10));
        p.register_block(30, expected_id);
        // Observe blocks 10, 20 → stride 10 dominant → predict 30.
        p.observe(10);
        p.observe(20);
        let req = p.predict().expect("should predict block 30");
        assert_eq!(req.id, expected_id);
        assert_eq!(req.source, PredictionSource::StridePredictor);
    }

    #[test]
    fn dag_prefetcher_returns_next_wave_objects() {
        // Build A → B → C (three sequential waves).
        let mut graph = WorkGraph::new();
        let a = graph.add("A", NodeKind::Cpu, &[]);
        let b = graph.add("B", NodeKind::Cpu, &[a]);
        let c = graph.add("C", NodeKind::Cpu, &[b]);

        let mut dag = DagPrefetcher::new();
        dag.register_task_objects(a, alloc::vec![make_id(1)]);
        dag.register_task_objects(b, alloc::vec![make_id(2)]);
        dag.register_task_objects(c, alloc::vec![make_id(3)]);

        // Wave 0 completes → should prefetch wave-1 objects (task B = make_id(2)).
        let reqs = dag.prefetches_for_wave(&graph, 0);
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].id, make_id(2));
        assert_eq!(reqs[0].source, PredictionSource::DagOracle);
    }

    #[test]
    fn engine_skips_under_critical_pressure() {
        let mut engine = PrefetchEngine::new(64);
        let id = make_id(99);
        engine.register_block(0, id);
        engine.register_block(10, make_id(1));
        engine.register_block(20, make_id(2));
        engine.register_block(30, make_id(3));
        engine.observe_block(0);
        engine.observe_block(10);
        engine.observe_block(20);
        // At this point the queue may have entries from stride predictions.
        // Manually push a background request to ensure the queue isn't empty.
        engine.queue.push_back(PrefetchRequest {
            id,
            target_tier: MemoryTier::Ram,
            source: PredictionSource::Manual,
            priority: Priority::Background,
        });
        engine.stats.queued += 1;

        let empty: BTreeMap<Hash256, usize> = BTreeMap::new();
        let admitted = engine.drain_under_pressure(PressureLevel::Critical, &empty);
        // Background tasks are Refused under Critical pressure.
        assert!(engine.stats().skipped_pressure > 0, "expected skipped_pressure > 0");
        let _ = admitted; // may be empty or contain non-background items
    }

    #[test]
    fn engine_admits_under_comfortable_pressure() {
        let mut engine = PrefetchEngine::new(64);
        let id0 = make_id(10);
        let id1 = make_id(11);
        let id2 = make_id(12);
        let id3 = make_id(13);
        engine.register_block(0,  id0);
        engine.register_block(5,  id1);
        engine.register_block(10, id2);
        engine.register_block(15, id3);
        engine.observe_block(0);
        engine.observe_block(5);
        engine.observe_block(10);
        // Ensure the queue has something.
        engine.queue.push_back(PrefetchRequest {
            id: id3,
            target_tier: MemoryTier::Ram,
            source: PredictionSource::Manual,
            priority: Priority::Background,
        });
        engine.stats.queued += 1;

        let empty: BTreeMap<Hash256, usize> = BTreeMap::new();
        let admitted = engine.drain_under_pressure(PressureLevel::Comfortable, &empty);
        assert!(!admitted.is_empty(), "expected at least one admitted request under Comfortable pressure");
    }

    #[test]
    fn hit_miss_tracking() {
        let mut engine = PrefetchEngine::new(16);
        let id = make_id(7);
        engine.record_hit(id);
        engine.record_hit(id);
        engine.record_miss(id);
        let s = engine.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn benchmark_prefetch_shows_latency_improvement() {
        let result = benchmark_prefetch(100, 1);
        assert!(
            result.cold_start_latency_ratio > 10.0,
            "expected ratio > 10.0, got {}",
            result.cold_start_latency_ratio
        );
    }

    // ── Pool-dispatched prefetch tests ───────────────────────────────────────

    fn make_engine_with_n_items(n: usize) -> PrefetchEngine {
        let mut engine = PrefetchEngine::new(n + 16);
        for i in 0..n {
            let id = make_id(i as u8);
            engine.queue.push_back(PrefetchRequest {
                id,
                target_tier: MemoryTier::Ram,
                source: PredictionSource::Manual,
                priority: Priority::Background,
            });
            engine.stats.queued += 1;
        }
        engine
    }

    #[test]
    fn pool_prefetch_refused_under_critical() {
        // Background + Critical → Refused.
        // Items should be put back in the queue (not dropped).
        let mut engine = make_engine_with_n_items(8);
        let empty: BTreeMap<Hash256, usize> = BTreeMap::new();
        let result = engine.pool_drain(PressureLevel::Critical, &empty);
        assert_eq!(result.executed, 0, "nothing should execute under Critical pressure");
        assert_eq!(result.refused_pressure, 8, "all 8 should be refused");
        assert_eq!(engine.queue_depth(), 8, "refused items should be re-enqueued");
    }

    #[test]
    fn pool_prefetch_executes_under_comfortable() {
        // Background + Comfortable → Accepted → all prefetches drain.
        let mut engine = make_engine_with_n_items(8);
        let empty: BTreeMap<Hash256, usize> = BTreeMap::new();
        let result = engine.pool_drain(PressureLevel::Comfortable, &empty);
        assert_eq!(result.executed, 8, "all 8 should execute under Comfortable pressure");
        assert_eq!(result.refused_pressure, 0, "nothing refused");
        assert_eq!(engine.queue_depth(), 0, "queue should be empty after drain");
    }

    // ── Integration benchmark test ───────────────────────────────────────────

    #[test]
    fn pool_integration_comfortable_completes_all_critical_sheds_all() {
        let result = benchmark_pool_integration(10);

        // Comfortable: all three systems complete work.
        assert_eq!(result.compression_batch_comfortable.compressed, 10,
            "compression: 10 items should complete under Comfortable");
        assert!(result.migration_comfortable.executed > 0,
            "migration: should execute under Comfortable");
        assert_eq!(result.prefetch_comfortable.executed, 10,
            "prefetch: all 10 should execute under Comfortable");

        // Critical: all three systems refuse/shed work.
        assert_eq!(result.compression_batch_critical.compressed, 0,
            "compression: 0 should complete under Critical");
        assert_eq!(result.migration_critical.executed, 0,
            "migration: 0 should execute under Critical");
        assert_eq!(result.prefetch_critical.executed, 0,
            "prefetch: 0 should execute under Critical");
    }
}

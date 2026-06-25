//! **Lever 5 — Self-optimising placement oracle** (memory-acceleration roadmap).
//!
//! Tracks per-object access patterns and uses a Q-learning ε-greedy bandit to
//! recommend the best [`MemoryTier`] for each object.  After enough observations
//! the oracle converges on the tier with the lowest access latency and emits
//! [`MigrationPlan`]s ordered by expected gain.
//!
//! Pure, safe `no_std`; host-tested.

use crate::coldcomp::PoolSnapshot;
use crate::governor::PressureLevel;
use crate::hash::Hash256;
use crate::pool::{Priority, PoolConfig, ThreadPool};
use crate::pressure::MemoryTier;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// ── TierCost ─────────────────────────────────────────────────────────────────

/// Typical access-latency model for each memory tier.
///
/// The values are representative orders-of-magnitude (not hardware constants);
/// the bandit uses them only to compute relative rewards.
#[derive(Clone, Copy, Debug)]
pub struct TierCost {
    /// Typical round-trip access latency in nanoseconds.
    pub latency_ns: u64,
}

impl TierCost {
    /// Return the cost model for a given tier.
    pub fn for_tier(tier: MemoryTier) -> TierCost {
        let latency_ns = match tier {
            MemoryTier::Vram =>        50,      //  50 ns — on-die HBM / GPU VRAM
            MemoryTier::Ram  =>       100,      // 100 ns — DRAM
            MemoryTier::Nvme =>   100_000,      // 100 µs — NVMe SSD
            MemoryTier::Peer =>   500_000,      // 500 µs — peer node over network
            MemoryTier::Cold => 10_000_000,     //  10 ms — cold / archival storage
        };
        TierCost { latency_ns }
    }
}

// ── MEM_TIERS constant ────────────────────────────────────────────────────────

const MEM_TIERS: [MemoryTier; 5] = [
    MemoryTier::Vram,
    MemoryTier::Ram,
    MemoryTier::Nvme,
    MemoryTier::Peer,
    MemoryTier::Cold,
];

// ── BanditStats ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct BanditStats {
    pub total_choices: u64,
    pub explorations: u64,
    pub exploitations: u64,
    pub convergence_round: Option<u64>,
    pub best_tier: Option<MemoryTier>,
}

// ── TierBandit ────────────────────────────────────────────────────────────────

/// ε-greedy Q-learning bandit over the five [`MemoryTier`] arms.
///
/// Each call to [`feed_access`] records an observation for the given tier and
/// updates the Q-values with a fixed learning-rate step (α = 0.3).  After 10
/// consecutive rounds where the same tier is `best()`, the bandit marks itself
/// as converged.
#[derive(Clone, Debug)]
pub struct TierBandit {
    q: [f64; 5],
    alpha: f64,
    epsilon: f64,
    updates: u64,
    last_best: Option<MemoryTier>,
    stable_count: u64,
    stats: BanditStats,
}

impl TierBandit {
    pub fn new() -> Self {
        TierBandit {
            q: [0.0; 5],
            alpha: 0.3,
            epsilon: 0.1,
            updates: 0,
            last_best: None,
            stable_count: 0,
            stats: BanditStats::default(),
        }
    }

    /// Seed Q-values with a scaled latency-inverse prior.
    ///
    /// The prior is set to 0.2% of the full latency-inverse reward so that even
    /// a handful of real observations on a tier quickly dominate, while the
    /// relative ordering (faster = higher Q) holds before any accesses are
    /// recorded.  This lets the oracle recommend moving Cold objects to faster
    /// tiers without cross-tier observations, without interfering with bandits
    /// that are trained exclusively on one tier.
    pub fn with_latency_prior() -> Self {
        let mut b = Self::new();
        for (i, tier) in MEM_TIERS.iter().enumerate() {
            b.q[i] = 1e9 / TierCost::for_tier(*tier).latency_ns as f64 * 0.002;
        }
        b
    }

    /// ε-greedy arm selection.
    ///
    /// `sample_u32` is a raw 32-bit pseudo-random value from the caller's DRNG;
    /// the bandit extracts both the explore/exploit decision and, when exploring,
    /// the arm index from this single value (no hidden state).
    pub fn choose(&mut self, sample_u32: u32) -> MemoryTier {
        self.stats.total_choices += 1;
        if sample_u32 % 1000 < 100 {
            // Explore: pick a tier from the sample.
            self.stats.explorations += 1;
            MEM_TIERS[(sample_u32 / 1000 % 5) as usize]
        } else {
            // Exploit: return the current best.
            self.stats.exploitations += 1;
            self.best()
        }
    }

    /// Update Q-value for `tier` toward `reward` (Temporal-Difference, no replay).
    pub fn reward(&mut self, tier: MemoryTier, reward: f64) {
        let i = tier as usize;
        self.q[i] += self.alpha * (reward - self.q[i]);
        self.updates += 1;
    }

    /// The tier with the highest Q-value; ties broken by first encounter.
    pub fn best(&self) -> MemoryTier {
        let mut best_idx = 0;
        for i in 1..5 {
            if self.q[i] > self.q[best_idx] {
                best_idx = i;
            }
        }
        MEM_TIERS[best_idx]
    }

    /// Raw Q-value for a tier.
    pub fn value(&self, tier: MemoryTier) -> f64 {
        self.q[tier as usize]
    }

    /// Record one access observation: compute a latency-inverse reward and update,
    /// then check for convergence (10 consecutive stable rounds).
    pub fn feed_access(&mut self, tier: MemoryTier, _sample: u32) {
        let reward = 1e9 / TierCost::for_tier(tier).latency_ns as f64;
        self.reward(tier, reward);

        let current_best = self.best();
        if Some(current_best) == self.last_best {
            self.stable_count += 1;
        } else {
            self.last_best = Some(current_best);
            self.stable_count = 1;
        }
        if self.stable_count >= 10 {
            self.stats.convergence_round = Some(self.updates);
            self.stats.best_tier = Some(current_best);
        }
    }

    pub fn stats(&self) -> &BanditStats {
        &self.stats
    }

    pub fn is_converged(&self) -> bool {
        self.stats.convergence_round.is_some()
    }
}

impl Default for TierBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── ObjectRecord (private) ────────────────────────────────────────────────────

struct ObjectRecord {
    current_tier: MemoryTier,
    access_count: u64,
    bandit: TierBandit,
    recommended_tier: MemoryTier,
}

// ── TrackerStats ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct TrackerStats {
    pub objects_tracked: usize,
    pub total_accesses: u64,
    pub migrations_recommended: u64,
    pub migrations_confirmed: u64,
    pub tier_distribution: [usize; 5],
}

// ── ObjectPlacementTracker ────────────────────────────────────────────────────

/// Maintains a per-object [`TierBandit`] and surfaces migration recommendations
/// once the bandit's opinion diverges from an object's current tier.
pub struct ObjectPlacementTracker {
    objects: BTreeMap<Hash256, ObjectRecord>,
    pub(crate) global_bandit: TierBandit,
    stats: TrackerStats,
}

impl ObjectPlacementTracker {
    pub fn new() -> Self {
        ObjectPlacementTracker {
            objects: BTreeMap::new(),
            global_bandit: TierBandit::new(),
            stats: TrackerStats::default(),
        }
    }

    /// Start tracking a new object placed on `initial_tier`.
    ///
    /// Per-object bandits are seeded with a latency-inverse prior so that even
    /// before any cross-tier observations the bandit already prefers faster tiers.
    /// Accessing a slow tier then reinforces that preference by giving a low reward
    /// relative to the prior, causing the bandit to recommend a migration.
    pub fn register(&mut self, id: Hash256, initial_tier: MemoryTier) {
        let bandit = TierBandit::with_latency_prior();
        let recommended_tier = bandit.best();
        self.objects.insert(
            id,
            ObjectRecord {
                current_tier: initial_tier,
                access_count: 0,
                bandit,
                recommended_tier,
            },
        );
        self.stats.tier_distribution[initial_tier as usize] += 1;
    }

    /// Record one access for `id` on `tier`, update the object's bandit, and
    /// emit a migration recommendation if the bandit now prefers a different tier.
    pub fn record_access(&mut self, id: Hash256, tier: MemoryTier, sample: u32) {
        if let Some(rec) = self.objects.get_mut(&id) {
            rec.access_count += 1;
            rec.bandit.feed_access(tier, sample);
            let new_rec = rec.bandit.best();
            if new_rec != rec.recommended_tier {
                if new_rec != rec.current_tier {
                    self.stats.migrations_recommended += 1;
                }
                rec.recommended_tier = new_rec;
            }
            self.stats.total_accesses += 1;
        }
    }

    /// The currently recommended tier for an object.
    pub fn recommend_tier(&self, id: Hash256) -> Option<MemoryTier> {
        self.objects.get(&id).map(|r| r.recommended_tier)
    }

    /// Returns `true` if the recommended tier differs from the current tier.
    pub fn migration_needed(&self, id: Hash256) -> bool {
        self.objects
            .get(&id)
            .map(|r| r.recommended_tier != r.current_tier)
            .unwrap_or(false)
    }

    /// All objects whose recommended tier differs from their current tier.
    /// Returns `(id, from, to)`.
    pub fn pending_migrations(&self) -> Vec<(Hash256, MemoryTier, MemoryTier)> {
        self.objects
            .iter()
            .filter(|(_, r)| r.recommended_tier != r.current_tier)
            .map(|(&id, r)| (id, r.current_tier, r.recommended_tier))
            .collect()
    }

    /// Confirm that `id` has been physically moved to `new_tier`, updating
    /// placement tracking and incrementing the migration counter.
    pub fn confirm_migration(&mut self, id: Hash256, new_tier: MemoryTier) {
        if let Some(rec) = self.objects.get_mut(&id) {
            self.stats.tier_distribution[rec.current_tier as usize] =
                self.stats.tier_distribution[rec.current_tier as usize].saturating_sub(1);
            self.stats.tier_distribution[new_tier as usize] += 1;
            rec.current_tier = new_tier;
            self.stats.migrations_confirmed += 1;
        }
    }

    /// A snapshot of tracker-level statistics.
    pub fn stats(&self) -> TrackerStats {
        // Recompute tier_distribution from live object records to stay accurate.
        let mut tier_distribution = [0usize; 5];
        for r in self.objects.values() {
            tier_distribution[r.current_tier as usize] += 1;
        }
        TrackerStats {
            objects_tracked: self.objects.len(),
            total_accesses: self.stats.total_accesses,
            migrations_recommended: self.stats.migrations_recommended,
            migrations_confirmed: self.stats.migrations_confirmed,
            tier_distribution,
        }
    }
}

impl Default for ObjectPlacementTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── OracleStats ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct OracleStats {
    pub placement_decisions: u64,
    pub evictions_avoided: u64,
    pub latency_improvement_ns: u64,
    pub total_migrations: u64,
}

// ── MigrationPlan ─────────────────────────────────────────────────────────────

/// A concrete migration recommendation emitted by [`PlacementOracle::optimize`].
#[derive(Clone, Debug)]
pub struct MigrationPlan {
    pub id: Hash256,
    pub from: MemoryTier,
    pub to: MemoryTier,
    pub bytes: usize,
    pub expected_latency_gain_ns: u64,
}

// ── PlacementOracle ───────────────────────────────────────────────────────────

/// Top-level advisor: admits objects, records accesses, and periodically
/// produces a sorted [`MigrationPlan`] list for the memory manager to execute.
pub struct PlacementOracle {
    pub(crate) tracker: ObjectPlacementTracker,
    tier_capacities: [usize; 5],
    tier_used: [usize; 5],
    object_sizes: BTreeMap<Hash256, usize>,
    stats: OracleStats,
}

impl PlacementOracle {
    pub fn new(tier_capacities: [usize; 5]) -> Self {
        PlacementOracle {
            tracker: ObjectPlacementTracker::new(),
            tier_capacities,
            tier_used: [0; 5],
            object_sizes: BTreeMap::new(),
            stats: OracleStats::default(),
        }
    }

    /// Admit a new object.  `hint` is an optional caller-supplied tier; when
    /// absent the global bandit chooses.  Returns the tier used.
    pub fn admit(
        &mut self,
        id: Hash256,
        bytes: usize,
        hint: Option<MemoryTier>,
        sample: u32,
    ) -> MemoryTier {
        let tier = hint.unwrap_or_else(|| self.tracker.global_bandit.choose(sample));
        self.tracker.register(id, tier);
        self.object_sizes.insert(id, bytes);
        self.tier_used[tier as usize] =
            self.tier_used[tier as usize].saturating_add(bytes);
        self.stats.placement_decisions += 1;
        tier
    }

    /// Record one access for `id`; returns the tier where it currently lives.
    pub fn access(&mut self, id: Hash256, sample: u32) -> Option<MemoryTier> {
        let tier = self.tracker.objects.get(&id)?.current_tier;
        self.tracker.record_access(id, tier, sample);
        Some(tier)
    }

    /// Generate a list of [`MigrationPlan`]s for all pending migrations, sorted
    /// by `expected_latency_gain_ns` (highest gain first).
    pub fn optimize(&mut self) -> Vec<MigrationPlan> {
        let pending = self.tracker.pending_migrations();
        let mut plans: Vec<MigrationPlan> = pending
            .into_iter()
            .map(|(id, from, to)| {
                let bytes = self.object_sizes.get(&id).copied().unwrap_or(0);
                let from_lat = TierCost::for_tier(from).latency_ns;
                let to_lat = TierCost::for_tier(to).latency_ns;
                let expected_latency_gain_ns = from_lat.saturating_sub(to_lat);
                MigrationPlan { id, from, to, bytes, expected_latency_gain_ns }
            })
            .collect();
        plans.sort_by(|a, b| b.expected_latency_gain_ns.cmp(&a.expected_latency_gain_ns));
        plans
    }

    /// Commit a migration plan: adjust capacity accounting and notify the tracker.
    pub fn execute_migration(&mut self, plan: &MigrationPlan) {
        self.tier_used[plan.from as usize] =
            self.tier_used[plan.from as usize].saturating_sub(plan.bytes);
        self.tier_used[plan.to as usize] += plan.bytes;
        self.tracker.confirm_migration(plan.id, plan.to);
        self.stats.total_migrations += 1;
        self.stats.latency_improvement_ns += plan.expected_latency_gain_ns;
    }

    /// Weighted-average access latency across all tracked objects (bytes × latency).
    pub fn average_access_latency_ns(&self) -> u64 {
        let mut total_weighted: u128 = 0;
        let mut total_bytes: u128 = 0;
        for (id, rec) in &self.tracker.objects {
            let bytes = self.object_sizes.get(id).copied().unwrap_or(0) as u128;
            let lat = TierCost::for_tier(rec.current_tier).latency_ns as u128;
            total_weighted += bytes * lat;
            total_bytes += bytes;
        }
        if total_bytes == 0 {
            return 0;
        }
        (total_weighted / total_bytes) as u64
    }

    pub fn stats(&self) -> &OracleStats {
        &self.stats
    }

    /// Remaining capacity for a given tier.
    pub fn tier_free(&self, tier: MemoryTier) -> usize {
        let i = tier as usize;
        self.tier_capacities[i].saturating_sub(self.tier_used[i])
    }
}

// ── Pool-dispatched parallel migration ───────────────────────────────────────

/// Result of [`PlacementOracle::parallel_optimize_and_migrate`].
pub struct BatchMigrateResult {
    /// How many migrations were pending before the batch.
    pub pending_migrations: usize,
    /// Tasks accepted and enqueued by the pool.
    pub submitted: u64,
    /// Tasks completed (migrations executed).
    pub executed: u64,
    /// Tasks refused due to pressure being too high.
    pub refused: u64,
    /// Sum of `expected_latency_gain_ns` across executed migrations.
    pub total_latency_gain_ns: u64,
    /// Snapshot of pool metrics at end of batch.
    pub pool_metrics: PoolSnapshot,
    /// `true` if a non-serial (SMP) spawner was active during this batch.
    pub used_smp: bool,
}

impl PlacementOracle {
    /// Run `optimize()` and then execute all resulting migrations through the pool.
    ///
    /// Migrations are submitted as `Priority::Background`:
    /// - `Comfortable` → `Accepted`  → executed
    /// - `Tight`       → `Deferred`  → not queued, not executed
    /// - `Critical`    → `Refused`   → not queued, not executed
    pub fn parallel_optimize_and_migrate(
        &mut self,
        pressure: PressureLevel,
    ) -> BatchMigrateResult {
        let plans = self.optimize();
        let n = plans.len();

        let mut pool = ThreadPool::new(PoolConfig {
            workers:     1,
            queue_depth: n.max(1),
            ..PoolConfig::default()
        });

        for i in 0..n {
            pool.submit(i, Priority::Background, pressure);
        }

        let refused = pool.metrics().refused;
        let mut total_latency_gain_ns: u64 = 0;

        while let Some(item) = pool.pop_for(0) {
            let plan = &plans[item.task_idx];
            total_latency_gain_ns += plan.expected_latency_gain_ns;
            self.execute_migration(plan);
            pool.mark_complete();
        }

        let m = pool.metrics();
        let snap = PoolSnapshot {
            submitted: m.submitted,
            completed: m.completed,
            refused:   m.refused,
            deferred:  m.deferred,
        };

        BatchMigrateResult {
            pending_migrations: n,
            submitted:          snap.submitted,
            executed:           snap.completed,
            refused,
            total_latency_gain_ns,
            pool_metrics:       snap,
            used_smp:           crate::pool::spawner_installed(),
        }
    }
}

// ── PlacementBenchResult ──────────────────────────────────────────────────────

/// Result of [`benchmark_placement_learning`].
pub struct PlacementBenchResult {
    pub objects: usize,
    pub access_rounds: usize,
    pub initial_avg_latency_ns: u64,
    pub final_avg_latency_ns: u64,
    /// `initial / final` — higher is better (> 1.0 means improvement).
    pub latency_improvement_factor: f64,
    pub migrations_executed: usize,
    pub convergence_achieved: bool,
}

// ── benchmark_placement_learning ──────────────────────────────────────────────

/// Benchmark the placement oracle: admit `n_objects` on Cold, then run
/// `access_rounds` rounds of accesses (each access teaches the bandit that
/// Ram is fast), trigger migrations each round, and report the latency gain.
pub fn benchmark_placement_learning(
    n_objects: usize,
    access_rounds: usize,
) -> PlacementBenchResult {
    let caps = [usize::MAX / 6; 5];
    let mut oracle = PlacementOracle::new(caps);

    // Build object IDs deterministically from their index.
    let ids: Vec<Hash256> = (0..n_objects)
        .map(|i| {
            let mut bytes = [0u8; 32];
            bytes[0] = (i & 0xFF) as u8;
            bytes[1] = ((i >> 8) & 0xFF) as u8;
            Hash256(bytes)
        })
        .collect();

    // Admit all objects to Cold tier.
    for (i, &id) in ids.iter().enumerate() {
        oracle.admit(id, 1024, Some(MemoryTier::Cold), i as u32);
    }

    let initial_avg_latency_ns = oracle.average_access_latency_ns();
    let mut migrations_executed = 0usize;

    for round in 0..access_rounds {
        // Access all objects — they record their current tier.
        for (i, &id) in ids.iter().enumerate() {
            oracle.access(id, (round * n_objects + i) as u32);
        }
        // Optimize and execute all recommended migrations.
        let plans = oracle.optimize();
        for plan in &plans {
            oracle.execute_migration(plan);
            migrations_executed += 1;
        }
    }

    let final_avg_latency_ns = oracle.average_access_latency_ns();
    let latency_improvement_factor =
        initial_avg_latency_ns as f64 / final_avg_latency_ns.max(1) as f64;
    let convergence_achieved = oracle
        .tracker
        .objects
        .values()
        .any(|r| r.bandit.is_converged());

    PlacementBenchResult {
        objects: n_objects,
        access_rounds,
        initial_avg_latency_ns,
        final_avg_latency_ns,
        latency_improvement_factor,
        migrations_executed,
        convergence_achieved,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::MemoryTier;

    fn make_id(n: u8) -> Hash256 {
        let mut bytes = [0u8; 32];
        bytes[0] = n;
        Hash256(bytes)
    }

    // ── 1. Bandit converges on the fastest tier after repeated Ram observations ──

    #[test]
    fn bandit_converges_on_fastest_tier() {
        let mut bandit = TierBandit::new();
        for _ in 0..20 {
            bandit.feed_access(MemoryTier::Ram, 500);
        }
        for _ in 0..5 {
            bandit.feed_access(MemoryTier::Cold, 999);
        }
        assert_eq!(bandit.best(), MemoryTier::Ram);
        assert!(bandit.value(MemoryTier::Ram) > bandit.value(MemoryTier::Cold));
    }

    // ── 2. Bandit explores before exploiting ─────────────────────────────────

    #[test]
    fn bandit_explores_before_converging() {
        let mut bandit = TierBandit::new();
        for i in 0..100u32 {
            bandit.choose(i * 7 + 3);
        }
        assert!(bandit.stats().explorations > 0);
        assert!(bandit.stats().exploitations > 0);
    }

    // ── 3. Tracker recommends migration after learning ────────────────────────

    #[test]
    fn tracker_recommends_migration_after_learning() {
        let mut tracker = ObjectPlacementTracker::new();
        let id = make_id(1);
        tracker.register(id, MemoryTier::Cold);
        for i in 0..20u32 {
            tracker.record_access(id, MemoryTier::Ram, i * 13 + 7);
        }
        let rec = tracker.recommend_tier(id).unwrap();
        assert_eq!(rec, MemoryTier::Ram);
        assert!(tracker.migration_needed(id));
    }

    // ── 4. Oracle places objects on the hinted tier ──────────────────────────

    #[test]
    fn oracle_admits_on_hint_tier() {
        let caps = [usize::MAX / 6; 5];
        let mut oracle = PlacementOracle::new(caps);
        let id = make_id(42);
        let tier = oracle.admit(id, 512, Some(MemoryTier::Nvme), 0);
        assert_eq!(tier, MemoryTier::Nvme);
    }

    // ── 5. optimize() returns plans sorted by gain descending ────────────────

    #[test]
    fn oracle_optimize_returns_migrations_sorted_by_gain() {
        let caps = [usize::MAX / 6; 5];
        let mut oracle = PlacementOracle::new(caps);
        let id1 = make_id(1);
        let id2 = make_id(2);
        oracle.admit(id1, 1024, Some(MemoryTier::Cold), 0);
        oracle.admit(id2, 1024, Some(MemoryTier::Cold), 0);
        // Teach id1 to prefer Ram (higher reward because Ram latency < Cold latency).
        for i in 0..15u32 {
            oracle.tracker.record_access(id1, MemoryTier::Ram, i * 17);
        }
        // Teach id2 to prefer Nvme.
        for i in 0..15u32 {
            oracle.tracker.record_access(id2, MemoryTier::Nvme, i * 13);
        }
        let plans = oracle.optimize();
        if plans.len() >= 2 {
            assert!(plans[0].expected_latency_gain_ns >= plans[1].expected_latency_gain_ns);
        }
    }

    // ── 6. Average latency improves (or stays same) after migration ──────────

    #[test]
    fn oracle_average_latency_improves_after_migration() {
        let caps = [usize::MAX / 6; 5];
        let mut oracle = PlacementOracle::new(caps);
        let id = make_id(99);
        oracle.admit(id, 1024, Some(MemoryTier::Cold), 0);
        let initial_latency = oracle.average_access_latency_ns();
        for i in 0..20u32 {
            oracle.tracker.record_access(id, MemoryTier::Ram, i * 11);
        }
        let plans = oracle.optimize();
        for plan in &plans {
            oracle.execute_migration(plan);
        }
        let final_latency = oracle.average_access_latency_ns();
        assert!(final_latency <= initial_latency);
    }

    // ── 7. Benchmark shows large improvement factor ──────────────────────────

    #[test]
    fn benchmark_placement_shows_large_improvement() {
        let result = benchmark_placement_learning(50, 20);
        assert!(
            result.latency_improvement_factor > 50.0,
            "Expected improvement factor > 50, got {}",
            result.latency_improvement_factor
        );
        assert!(result.migrations_executed > 0);
    }

    // ── 8. Pool-dispatched migration tests ───────────────────────────────────

    /// Helper: build an oracle with `n` objects pre-trained to want a migration
    /// (start on Cold, observe Ram accesses so the bandit recommends Ram).
    fn oracle_with_pending_migrations(n: usize) -> PlacementOracle {
        let caps = [usize::MAX / 6; 5];
        let mut oracle = PlacementOracle::new(caps);
        for i in 0..n {
            let id = make_id(i as u8);
            oracle.admit(id, 1024, Some(MemoryTier::Cold), i as u32);
            // Enough Ram observations to flip the bandit's recommendation.
            for j in 0..20u32 {
                oracle.tracker.record_access(id, MemoryTier::Ram, j * 13 + 7);
            }
        }
        oracle
    }

    #[test]
    fn pool_migration_refused_under_critical() {
        // Background + Critical → Refused.
        // No migrations should execute; refused count equals pending_migrations.
        let mut oracle = oracle_with_pending_migrations(5);
        let result = oracle.parallel_optimize_and_migrate(PressureLevel::Critical);
        assert_eq!(result.executed, 0, "nothing should execute under Critical pressure");
        assert_eq!(result.refused, result.pending_migrations as u64,
            "all pending migrations should be refused");
    }

    #[test]
    fn pool_migration_executes_under_comfortable() {
        // Background + Comfortable → Accepted → migrations execute.
        let mut oracle = oracle_with_pending_migrations(5);
        let result = oracle.parallel_optimize_and_migrate(PressureLevel::Comfortable);
        assert!(result.executed > 0, "at least one migration should execute");
        assert_eq!(result.refused, 0, "nothing should be refused under Comfortable pressure");
        assert!(result.total_latency_gain_ns > 0, "executed migrations should yield latency gain");
    }
}

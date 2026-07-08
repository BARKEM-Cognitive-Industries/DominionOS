//! The **Resource Governor** — memory pressure & self-managing allocation as a
//! capability/quota problem (Stage 3 extension).
//!
//! DominionOS already treats *authority* as explicit, bounded, and revocable. The
//! governor applies the **same lens to resources**: memory, compute, and energy are
//! capability-rate-limited, and *pressure* is what happens when demand approaches a
//! domain's quota. So this is the resource mirror of the firewall — over-budget
//! **degrades or defers**, it never machine-wide OOM-kills.
//!
//! | Security (today)                              | Resource governor (here)                    |
//! |-----------------------------------------------|---------------------------------------------|
//! | Authority is a capability (`capability.rs`)   | A budget is a capability quota              |
//! | Over-reach **traps** (CHERI bounds)           | Over-budget **degrades/defers** (no OOM-kill)|
//! | Revocation is recursive & instant             | Reclaim is class-granular & graph-driven    |
//! | Energy budgeted per domain (`power.rs`)        | Memory & compute budgeted the same way      |
//!
//! It composes the existing substrate rather than replacing it: [`crate::power`]
//! (per-domain energy budgets), [`crate::pressure`] (`WorkingSet` LRU residency +
//! the `Pressure` signal), [`crate::neural`] (`LearnedTierPolicy` RL bandit), and
//! [`crate::memcrypt`] (`SealedRegion` for encrypted dirty spill). This module adds
//! the **unified per-domain ledger**, the **degradation tiers / admission control**,
//! **reclaim-by-recomputability**, the **generalized placement bandit**, and the
//! **determinism guard-rail**.
//!
//! ## Determinism guard-rail (the hard constraint)
//!
//! Self-managing policies decide from observed pressure — which is *non-deterministic
//! input*. To preserve Stage 10 replay, every adaptive decision enters the state
//! machine as a **recorded input event** ([`ResourceGovernor::decisions`] /
//! [`ResourceGovernor::log_digest`]); the raw pressure samples stay at the boundary
//! (passed into [`govern`](ResourceGovernor::govern), never stored). Two governors
//! driven by the same call sequence produce an identical decision log.
//!
//! Pure, safe `no_std`; host- and on-metal-tested.

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A memory-pressure tier (mirrors `power.rs::PowerState`), set by a domain's
/// working-set fill ratio.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PressureLevel {
    /// Below 70% of budget — allocate freely.
    Comfortable,
    /// 70–90% — shed caches and decline *new speculative* allocations.
    Tight,
    /// ≥90% — refuse non-essential work and defer it (admission control), never thrash.
    Critical,
}

impl PressureLevel {
    /// Classify a prospective `used`/`cap` ratio into a tier.
    pub fn from_ratio(used: usize, cap: usize) -> PressureLevel {
        // Widen to u128 before scaling so large byte budgets can't overflow the
        // multiply (an unchecked `used * 10` wraps/panics above ~1.8e18 on 64-bit).
        if cap == 0 || (used as u128) * 10 < (cap as u128) * 7 {
            PressureLevel::Comfortable
        } else if (used as u128) * 10 < (cap as u128) * 9 {
            PressureLevel::Tight
        } else {
            PressureLevel::Critical
        }
    }
}

/// The outcome of an admission request — never an OOM-kill.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Admission {
    /// Charged and granted (domain is `Comfortable`).
    Granted,
    /// Charged but the domain is under pressure — caller should shed caches.
    Degraded(PressureLevel),
    /// Essential work that would breach the budget: not charged, **re-queue** it
    /// (the scheduler defers rather than thrashing).
    Deferred,
    /// Non-essential (speculative) work declined under pressure or over budget.
    Refused,
}

/// The recomputability class of an object — cheapest-to-reclaim first. Eviction
/// proceeds in declaration order, so [`Regenerable`](ReclaimClass::Regenerable) goes
/// before [`Pinned`](ReclaimClass::Pinned) is ever touched (it never is).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum ReclaimClass {
    /// Recompute / re-fetch for free: generative latents (`neural.rs`), NDN cache.
    Regenerable = 0,
    /// Already content-addressed on disk: evict, fault back by hash (`persist.rs`).
    CleanPersisted = 1,
    /// Derived but not yet persisted: write-through, then evict.
    CleanInMemory = 2,
    /// Uncommitted mutable state: commit (encrypted spill) then evict.
    Dirty = 3,
    /// Keys, capability tables, kernel state — never evicted.
    Pinned = 4,
}

impl ReclaimClass {
    /// Can an object of this class ever be reclaimed?
    pub fn is_evictable(self) -> bool {
        self != ReclaimClass::Pinned
    }
}

/// A placement target for the self-optimizing bandit — generalizing the storage-tier
/// policy to *resource placement*. Lines up with the language's `@CPU/@GPU/@NPU`
/// decorators and the Hot/Warm/Cold storage tiers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlacementTarget {
    Cpu = 0,
    Gpu = 1,
    Npu = 2,
    Hot = 3,
    Warm = 4,
    Cold = 5,
}

const TARGETS: [PlacementTarget; 6] = [
    PlacementTarget::Cpu,
    PlacementTarget::Gpu,
    PlacementTarget::Npu,
    PlacementTarget::Hot,
    PlacementTarget::Warm,
    PlacementTarget::Cold,
];

impl PlacementTarget {
    /// Map a Dominion placement decorator (`@CPU`/`@GPU`/`@NPU`) to its target — used
    /// as the *prior* the bandit then corrects under real pressure.
    pub fn from_decorator(decorator: &str) -> Option<PlacementTarget> {
        match decorator.trim_start_matches('@').to_ascii_uppercase().as_str() {
            "CPU" => Some(PlacementTarget::Cpu),
            "GPU" => Some(PlacementTarget::Gpu),
            "NPU" => Some(PlacementTarget::Npu),
            "HOT" => Some(PlacementTarget::Hot),
            "WARM" => Some(PlacementTarget::Warm),
            "COLD" => Some(PlacementTarget::Cold),
            _ => None,
        }
    }
}

/// A Q-learning bandit over [`PlacementTarget`]s — the generalization of
/// `neural.rs::LearnedTierPolicy` from storage tiers to compute/storage placement.
/// Reward is negative cost (latency + energy + pressure penalty), so it converges on
/// the placement that is cheapest **under current pressure** — work migrates off a
/// contended accelerator automatically.
#[derive(Clone)]
pub struct PlacementPolicy {
    q: [f64; 6],
    alpha: f64,
    epsilon_milli: u32,
}

impl PlacementPolicy {
    pub fn new() -> PlacementPolicy {
        PlacementPolicy { q: [0.0; 6], alpha: 0.3, epsilon_milli: 100 }
    }

    /// Seed a prior from a placement decorator (a static hint nudges the bandit but
    /// does not pin it — the correction still wins if the target is contended).
    pub fn prefer(&mut self, target: PlacementTarget, strength: f64) {
        self.q[target as usize] += strength;
    }

    /// Choose ε-greedily. `sample` (0..1000, from the DRNG) decides explore vs
    /// exploit; `explore_pick` selects which target to try when exploring.
    pub fn choose(&self, sample: u32, explore_pick: usize) -> PlacementTarget {
        if sample < self.epsilon_milli {
            TARGETS[explore_pick % TARGETS.len()]
        } else {
            self.best()
        }
    }

    /// The current greedy-best target (highest Q).
    pub fn best(&self) -> PlacementTarget {
        let mut best = 0;
        for i in 1..TARGETS.len() {
            if self.q[i] > self.q[best] {
                best = i;
            }
        }
        TARGETS[best]
    }

    /// Update a target's value toward an observed `reward` (higher = cheaper).
    pub fn reward(&mut self, target: PlacementTarget, reward: f64) {
        let i = target as usize;
        self.q[i] += self.alpha * (reward - self.q[i]);
    }

    pub fn value(&self, target: PlacementTarget) -> f64 {
        self.q[target as usize]
    }
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// One unified degradation action emitted by [`ResourceGovernor::govern`], in a
/// single coherent order across memory, energy, and compute pressure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DegradationAction {
    /// Drop regenerable caches (cheapest first response).
    ShedCaches,
    /// Migrate contended compute to a cheaper placement (the bandit picks where).
    MigrateCompute,
    /// Stop admitting new speculative work.
    DeferSpeculative,
    /// Refuse non-essential allocations outright (last resort before thrash).
    RefuseNonEssential,
}

/// An adaptive decision recorded in the deterministic state log. Replay reproduces
/// the exact trajectory; the raw pressure measurements that seeded it stay at the
/// boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    Reserve { domain: u64, bytes: usize, outcome: Admission },
    Evict { id: Hash256, class: ReclaimClass },
    Migrate { target: PlacementTarget },
    SetBudget { domain: u64, cap: usize },
}

/// The one per-domain quota ledger that memory, energy, and compute all read — so a
/// domain's footprint is a single coherent picture, not three disconnected meters.
#[derive(Clone, Copy, Default)]
struct DomainLedger {
    mem_cap: usize,
    mem_used: usize,
    energy_used: u64,
    compute_used: u64,
}

/// What the governor knows about a tracked object (for class-ordered reclaim).
#[derive(Clone, Copy)]
struct Tracked {
    domain: u64,
    bytes: usize,
    class: ReclaimClass,
}

/// The unified resource governor.
pub struct ResourceGovernor {
    domains: BTreeMap<u64, DomainLedger>,
    objects: BTreeMap<Hash256, Tracked>,
    placement: PlacementPolicy,
    log: Vec<Decision>,
}

impl Default for ResourceGovernor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceGovernor {
    pub fn new() -> ResourceGovernor {
        ResourceGovernor {
            domains: BTreeMap::new(),
            objects: BTreeMap::new(),
            placement: PlacementPolicy::new(),
            log: Vec::new(),
        }
    }

    /// Give a domain a working-set budget in bytes (idempotent — re-setting resets
    /// the cap). A domain with no budget is unlimited.
    pub fn set_mem_budget(&mut self, domain: u64, bytes: usize) {
        self.domains.entry(domain).or_default().mem_cap = bytes;
        self.log.push(Decision::SetBudget { domain, cap: bytes });
    }

    /// Current memory pressure tier for a domain.
    pub fn pressure(&self, domain: u64) -> PressureLevel {
        match self.domains.get(&domain) {
            Some(l) => PressureLevel::from_ratio(l.mem_used, l.mem_cap),
            None => PressureLevel::Comfortable,
        }
    }

    /// Bytes a domain currently has resident.
    pub fn mem_used(&self, domain: u64) -> usize {
        self.domains.get(&domain).map(|l| l.mem_used).unwrap_or(0)
    }

    /// **Admission control.** Try to reserve `bytes` for `domain`. `essential` work
    /// that would breach the budget is **deferred** (re-queued), never refused;
    /// speculative work is **refused** under pressure — but the machine is never
    /// OOM-killed. The decision is recorded for deterministic replay.
    pub fn reserve(&mut self, domain: u64, bytes: usize, essential: bool) -> Admission {
        let led = self.domains.entry(domain).or_default();
        let cap = led.mem_cap;
        // Use saturating_add so that astronomically large byte counts never wrap
        // a usize back to a small value, which would incorrectly pass the cap check
        // and grant an allocation the domain cannot actually hold.
        let prospective = led.mem_used.saturating_add(bytes);

        let outcome = if cap == 0 {
            // Un-budgeted domain: unlimited.
            led.mem_used = prospective;
            Admission::Granted
        } else if prospective > cap {
            // Over budget — never OOM-kill. Essential defers for reclaim+retry;
            // speculative is refused.
            if essential {
                Admission::Deferred
            } else {
                Admission::Refused
            }
        } else {
            let level = PressureLevel::from_ratio(prospective, cap);
            if !essential && level != PressureLevel::Comfortable {
                // Decline new speculative allocations under pressure (shed/back off).
                Admission::Refused
            } else {
                led.mem_used = prospective;
                match level {
                    PressureLevel::Comfortable => Admission::Granted,
                    other => Admission::Degraded(other),
                }
            }
        };
        self.log.push(Decision::Reserve { domain, bytes, outcome });
        outcome
    }

    /// Release `bytes` previously reserved for a domain (e.g. on object free).
    pub fn release(&mut self, domain: u64, bytes: usize) {
        if let Some(l) = self.domains.get_mut(&domain) {
            l.mem_used = l.mem_used.saturating_sub(bytes);
        }
    }

    /// Charge energy / compute to the *same* ledger memory reads — one coherent
    /// per-domain footprint.
    pub fn charge_energy(&mut self, domain: u64, energy: u64) {
        let l = self.domains.entry(domain).or_default();
        l.energy_used = l.energy_used.saturating_add(energy);
    }
    pub fn charge_compute(&mut self, domain: u64, units: u64) {
        let l = self.domains.entry(domain).or_default();
        l.compute_used = l.compute_used.saturating_add(units);
    }
    /// A domain's `(mem_used, energy_used, compute_used)` footprint from the unified ledger.
    pub fn footprint(&self, domain: u64) -> (usize, u64, u64) {
        self.domains
            .get(&domain)
            .map(|l| (l.mem_used, l.energy_used, l.compute_used))
            .unwrap_or((0, 0, 0))
    }

    // --- reclaim-by-recomputability ------------------------------------------

    /// Track an object so the governor can reclaim it by class when under pressure.
    pub fn track(&mut self, domain: u64, id: Hash256, bytes: usize, class: ReclaimClass) {
        self.objects.insert(id, Tracked { domain, bytes, class });
    }

    /// The recomputability class of a tracked object.
    pub fn classify(&self, id: Hash256) -> Option<ReclaimClass> {
        self.objects.get(&id).map(|t| t.class)
    }

    /// **Reclaim by recomputability**: free at least `target_bytes` from `domain` by
    /// evicting the cheapest-to-restore class first (Regenerable → CleanPersisted →
    /// CleanInMemory → Dirty), **never** touching Pinned state. Returns the evicted
    /// ids (still re-fetchable by hash / recomputable). Each eviction is logged.
    pub fn reclaim(&mut self, domain: u64, target_bytes: usize) -> Vec<Hash256> {
        // Candidate ids in this domain that are evictable, cheapest class first.
        let mut candidates: Vec<(Hash256, ReclaimClass, usize)> = self
            .objects
            .iter()
            .filter(|(_, t)| t.domain == domain && t.class.is_evictable())
            .map(|(id, t)| (*id, t.class, t.bytes))
            .collect();
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut freed = 0usize;
        let mut evicted = Vec::new();
        for (id, class, bytes) in candidates {
            if freed >= target_bytes {
                break;
            }
            self.objects.remove(&id);
            self.release(domain, bytes);
            freed += bytes;
            evicted.push(id);
            self.log.push(Decision::Evict { id, class });
        }
        evicted
    }

    // --- self-optimizing placement -------------------------------------------

    /// Borrow the placement bandit (to seed priors / read values).
    pub fn placement(&mut self) -> &mut PlacementPolicy {
        &mut self.placement
    }

    /// Seed the bandit's prior from a language placement decorator (`@NPU` etc.).
    pub fn prefer_decorator(&mut self, decorator: &str, strength: f64) -> Option<PlacementTarget> {
        let t = PlacementTarget::from_decorator(decorator)?;
        self.placement.prefer(t, strength);
        Some(t)
    }

    /// Choose a placement and **record the migration decision** (determinism guard-rail).
    pub fn place(&mut self, sample: u32, explore_pick: usize) -> PlacementTarget {
        let target = self.placement.choose(sample, explore_pick);
        self.log.push(Decision::Migrate { target });
        target
    }

    /// Feed back the observed cost of a placement (higher reward = cheaper).
    pub fn reward_placement(&mut self, target: PlacementTarget, reward: f64) {
        self.placement.reward(target, reward);
    }

    // --- unified govern policy -----------------------------------------------

    /// **Unified degradation order** from the three shared pressure signals. Memory,
    /// energy, and compute agree on one ordered response — cheapest/least-disruptive
    /// first — so low-memory + low-battery shed deferrable work before anything
    /// essential is touched. Advisory and pure: it records nothing and reads only the
    /// boundary-supplied samples plus the shared ledger.
    pub fn govern(
        &self,
        memory: PressureLevel,
        energy_throttled: bool,
        compute_contended: bool,
    ) -> Vec<DegradationAction> {
        let mut order = Vec::new();
        // 1. Shed regenerable caches first — the cheapest relief.
        if memory != PressureLevel::Comfortable || energy_throttled {
            order.push(DegradationAction::ShedCaches);
        }
        // 2. Migrate contended compute off the hot accelerator (the bandit picks where).
        if compute_contended {
            order.push(DegradationAction::MigrateCompute);
        }
        // 3. Under real pressure, stop admitting speculative work.
        if memory == PressureLevel::Critical || energy_throttled {
            order.push(DegradationAction::DeferSpeculative);
        }
        // 4. Last resort before thrash: refuse non-essential allocations.
        if memory == PressureLevel::Critical {
            order.push(DegradationAction::RefuseNonEssential);
        }
        order
    }

    // --- determinism guard-rail ----------------------------------------------

    /// The recorded decision log (the deterministic input-event stream).
    pub fn decisions(&self) -> &[Decision] {
        &self.log
    }

    /// A content hash over the decision log — two governors driven by the same call
    /// sequence agree exactly (the determinism boundary, checkable).
    pub fn log_digest(&self) -> Hash256 {
        let mut buf = Vec::new();
        for d in &self.log {
            buf.extend_from_slice(&decision_bytes(d));
        }
        Hash256::of(&buf)
    }
}

/// Canonical bytes for a decision (stable encoding for the determinism digest).
fn decision_bytes(d: &Decision) -> Vec<u8> {
    let mut b = Vec::new();
    match d {
        Decision::Reserve { domain, bytes, outcome } => {
            b.push(1);
            b.extend_from_slice(&domain.to_le_bytes());
            b.extend_from_slice(&(*bytes as u64).to_le_bytes());
            b.push(admission_tag(*outcome));
        }
        Decision::Evict { id, class } => {
            b.push(2);
            b.extend_from_slice(&id.0);
            b.push(*class as u8);
        }
        Decision::Migrate { target } => {
            b.push(3);
            b.push(*target as u8);
        }
        Decision::SetBudget { domain, cap } => {
            b.push(4);
            b.extend_from_slice(&domain.to_le_bytes());
            b.extend_from_slice(&(*cap as u64).to_le_bytes());
        }
    }
    b
}

fn admission_tag(a: Admission) -> u8 {
    match a {
        Admission::Granted => 0,
        Admission::Degraded(PressureLevel::Comfortable) => 1,
        Admission::Degraded(PressureLevel::Tight) => 2,
        Admission::Degraded(PressureLevel::Critical) => 3,
        Admission::Deferred => 4,
        Admission::Refused => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(seed: &[u8]) -> Hash256 {
        Hash256::of(seed)
    }

    // ---- graceful degradation tiers + admission control ----
    #[test]
    fn pressure_tiers_and_admission_never_oom_kill() {
        let mut g = ResourceGovernor::new();
        g.set_mem_budget(1, 1000);
        // Comfortable: a normal allocation is granted.
        assert_eq!(g.reserve(1, 600, true), Admission::Granted); // 60% → Comfortable
        assert_eq!(g.pressure(1), PressureLevel::Comfortable);
        // Now at 60%; an essential 200 takes it to 80% = Tight → Degraded (still charged).
        assert_eq!(g.reserve(1, 200, true), Admission::Degraded(PressureLevel::Tight));
        assert_eq!(g.pressure(1), PressureLevel::Tight);
        // A *speculative* allocation under pressure is refused (declined, not charged).
        assert_eq!(g.reserve(1, 50, false), Admission::Refused);
        assert_eq!(g.mem_used(1), 800); // unchanged by the refused speculative request
        // Essential work that would breach the cap is deferred (re-queued), never killed.
        assert_eq!(g.reserve(1, 500, true), Admission::Deferred);
        assert_eq!(g.mem_used(1), 800);
    }

    // ---- reclaim-by-recomputability: cheapest class first, never pinned ----
    #[test]
    fn reclaim_evicts_regenerable_before_dirty_and_never_pinned() {
        let mut g = ResourceGovernor::new();
        g.set_mem_budget(1, 1000);
        g.reserve(1, 400, true);
        let regen = h(b"latent");
        let clean = h(b"persisted");
        let dirty = h(b"uncommitted");
        let pinned = h(b"keys");
        g.track(1, regen, 100, ReclaimClass::Regenerable);
        g.track(1, clean, 100, ReclaimClass::CleanPersisted);
        g.track(1, dirty, 100, ReclaimClass::Dirty);
        g.track(1, pinned, 100, ReclaimClass::Pinned);

        // Need to free 150 bytes: takes the regenerable, then the clean-persisted.
        let evicted = g.reclaim(1, 150);
        assert_eq!(evicted, alloc::vec![regen, clean]);
        // Pinned state is never evicted; dirty survives because cheaper classes sufficed.
        assert_eq!(g.classify(pinned), Some(ReclaimClass::Pinned));
        assert_eq!(g.classify(dirty), Some(ReclaimClass::Dirty));
        assert_eq!(g.classify(regen), None);
    }

    #[test]
    fn reclaim_never_touches_pinned_even_under_extreme_pressure() {
        let mut g = ResourceGovernor::new();
        g.set_mem_budget(1, 1000);
        g.reserve(1, 200, true);
        g.track(1, h(b"k1"), 100, ReclaimClass::Pinned);
        g.track(1, h(b"k2"), 100, ReclaimClass::Pinned);
        // Ask for far more than exists — only evictable classes can go (none here).
        let evicted = g.reclaim(1, 10_000);
        assert!(evicted.is_empty());
    }

    // ---- self-optimizing placement: generalize the RL bandit + decorators ----
    #[test]
    fn placement_bandit_learns_cheapest_target_and_seeds_from_decorators() {
        let mut g = ResourceGovernor::new();
        // The @NPU decorator seeds a prior...
        assert_eq!(g.prefer_decorator("@NPU", 0.5), Some(PlacementTarget::Npu));
        // ...but if the NPU is contended (low reward) and the GPU is cheap, the
        // bandit corrects and migrates work off the accelerator.
        for _ in 0..20 {
            g.reward_placement(PlacementTarget::Npu, -1.0);
            g.reward_placement(PlacementTarget::Gpu, 1.0);
        }
        assert_eq!(g.placement().best(), PlacementTarget::Gpu);
        assert_eq!(PlacementTarget::from_decorator("@cpu"), Some(PlacementTarget::Cpu));
    }

    // ---- unified govern policy: one degradation order across mem/energy/compute ----
    #[test]
    fn unified_govern_emits_one_coherent_degradation_order() {
        let g = ResourceGovernor::new();
        // Comfortable + healthy: nothing to do.
        assert!(g.govern(PressureLevel::Comfortable, false, false).is_empty());
        // Tight memory: shed caches only.
        assert_eq!(
            g.govern(PressureLevel::Tight, false, false),
            alloc::vec![DegradationAction::ShedCaches]
        );
        // Critical memory + low battery + contended compute: full ordered escalation.
        assert_eq!(
            g.govern(PressureLevel::Critical, true, true),
            alloc::vec![
                DegradationAction::ShedCaches,
                DegradationAction::MigrateCompute,
                DegradationAction::DeferSpeculative,
                DegradationAction::RefuseNonEssential,
            ]
        );
    }

    #[test]
    fn unified_ledger_is_one_picture_across_resources() {
        let mut g = ResourceGovernor::new();
        g.set_mem_budget(7, 1000);
        g.reserve(7, 300, true);
        g.charge_energy(7, 42);
        g.charge_compute(7, 9);
        assert_eq!(g.footprint(7), (300, 42, 9));
    }

    // ---- determinism guard-rail: same calls => same decision log ----
    #[test]
    fn decision_log_is_deterministic_across_replays() {
        fn run() -> Hash256 {
            let mut g = ResourceGovernor::new();
            g.set_mem_budget(1, 1000);
            g.reserve(1, 600, true);
            g.reserve(1, 300, false);
            g.track(1, h(b"a"), 100, ReclaimClass::Regenerable);
            g.reclaim(1, 50);
            g.reward_placement(PlacementTarget::Gpu, 1.0);
            g.place(900, 0);
            g.log_digest()
        }
        assert_eq!(run(), run());
        // The log records decisions, not raw pressure samples.
        let mut g = ResourceGovernor::new();
        g.set_mem_budget(1, 100);
        g.reserve(1, 50, true);
        assert!(matches!(g.decisions().last(), Some(Decision::Reserve { .. })));
    }
}

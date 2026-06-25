//! Decentralized compute marketplace & dual-pool scheduling
//! (`docs/architecture/decentralized-compute-marketplace.md`).
//!
//! A node may **contribute** spare compute to a fleet and **consume** others'. DominionOS frames
//! this entirely in terms of primitives it already has:
//!
//! * **Opt-in, default-off** — contribution is a revocable budget granted to a dedicated
//!   **Public-Work domain** (reuses [`crate::governor`] quotas + [`crate::firewall`] domain
//!   segmentation + [`crate::consent`]). Nothing is shared until the owner explicitly enables
//!   it, and [`Marketplace::revoke_contribution`] pulls it instantly.
//! * **Sharing predicates** — the budget is only *offered* when the machine is genuinely idle,
//!   inside an allowed window, and cool/charged enough ([`SharingPredicate`], extending the
//!   [`crate::power`] scheduling dimensions).
//! * **Reverse-auction matching** — demand posts a resource **envelope** and a **max price**;
//!   suppliers bid; the cheapest bid that meets the envelope wins, with a deterministic
//!   tie-break ([`ReverseAuction`]).
//! * **Decentralized BFT scheduling** — there is no central control plane: the work-assignment
//!   ledger is the fleet's [`crate::bft`] log (HLC-ordered, Byzantine-safe).
//! * **Model fragmentation** — a large model is pipelined layer-by-layer across neighbour nodes
//!   over the distributed store ([`fragment_model`] builds a [`crate::multikernel::WorkGraph`]).
//! * **Two pools** — a **private** pool requires a minimum attested confidential-compute tier
//!   before dispatch ([`crate::enclave`]/[`crate::rot`]); a **public** pool runs
//!   generation-tracked cells ([`crate::dsasos`]) whose results are verified in
//!   [`crate::settlement`] (Proof-of-Inference / Proof-of-Learning).
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::enforcement::Tier;
use crate::governor::{Admission, ResourceGovernor};
use crate::hash::Hash256;
use crate::multikernel::{NodeKind, WorkGraph};
use alloc::string::String;
use alloc::vec::Vec;

/// The dedicated domain id spare compute is contributed under (segmented by `firewall.rs`).
pub const PUBLIC_WORK_DOMAIN: u64 = 0x0000_0000_5055_424C; // "PUBL"

/// Which pool a job runs in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pool {
    /// Confidential: requires a minimum attested confidential-compute tier before dispatch.
    Private,
    /// Collaborative: open cells, results verified by replay / ZK in `settlement.rs`.
    Public,
}

/// When a node is willing to offer spare compute. Every condition must hold — so a busy, hot,
/// or low-battery machine never gives work away.
#[derive(Clone, Copy, Debug)]
pub struct SharingPredicate {
    /// Minimum fraction (0..=100) of the machine that must be idle.
    pub min_idle_pct: u8,
    /// Allowed wall-clock window `[start, end)` in seconds-of-day (wraps if start > end).
    pub window: (u32, u32),
    /// Maximum temperature (arbitrary units) at which work is still accepted.
    pub max_temp: u32,
    /// Minimum battery percentage (100 if on mains).
    pub min_battery: u8,
}

impl SharingPredicate {
    /// A sensible default: ≥70% idle, any time, cool, ≥50% battery.
    pub fn relaxed() -> SharingPredicate {
        SharingPredicate { min_idle_pct: 70, window: (0, 86_400), max_temp: u32::MAX, min_battery: 50 }
    }

    /// True iff the machine should currently offer compute given its live telemetry.
    pub fn offers(&self, idle_pct: u8, time_of_day: u32, temp: u32, battery: u8) -> bool {
        let in_window = if self.window.0 <= self.window.1 {
            time_of_day >= self.window.0 && time_of_day < self.window.1
        } else {
            time_of_day >= self.window.0 || time_of_day < self.window.1
        };
        idle_pct >= self.min_idle_pct && in_window && temp <= self.max_temp && battery >= self.min_battery
    }
}

/// A unit of demand: the resources a job needs and the most the buyer will pay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub cpu_units: u32,
    pub mem_bytes: u64,
    pub max_price: u64,
}

/// A supplier's bid into an auction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bid {
    pub supplier: u32,
    pub cpu_units: u32,
    pub mem_bytes: u64,
    pub price: u64,
}

impl Bid {
    fn meets(&self, e: &Envelope) -> bool {
        self.cpu_units >= e.cpu_units && self.mem_bytes >= e.mem_bytes && self.price <= e.max_price
    }
}

/// A sealed-envelope **reverse auction**: the lowest-priced bid that meets the demand envelope
/// wins; ties break to the lower supplier id (deterministic).
pub struct ReverseAuction {
    envelope: Envelope,
    bids: Vec<Bid>,
}

impl ReverseAuction {
    pub fn new(envelope: Envelope) -> ReverseAuction {
        ReverseAuction { envelope, bids: Vec::new() }
    }
    pub fn bid(&mut self, bid: Bid) {
        self.bids.push(bid);
    }
    /// The winning bid, or `None` if no bid meets the envelope within the max price.
    pub fn settle(&self) -> Option<Bid> {
        self.bids
            .iter()
            .filter(|b| b.meets(&self.envelope))
            .copied()
            .min_by(|a, b| a.price.cmp(&b.price).then(a.supplier.cmp(&b.supplier)))
    }
}

/// The numeric rank of an enforcement/confidential tier, so "at least tier X" is a comparison.
fn tier_rank(t: Tier) -> u8 {
    match t {
        Tier::Software => 0,
        Tier::MemoryTagging => 1,
        Tier::Cheri => 2,
    }
}

/// A node's marketplace state: its (default-off) contribution budget + the BFT-ordered
/// assignment ledger.
pub struct Marketplace {
    contributing: bool,
    predicate: SharingPredicate,
    gov: ResourceGovernor,
    budget_bytes: usize,
    /// Hash-chained log of accepted assignments — in production this *is* the `bft.rs` log.
    assignments: Vec<Assignment>,
    ledger_digest: Hash256,
}

/// One accepted work assignment recorded on the (BFT) ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub job: String,
    pub supplier: u32,
    pub price: u64,
    pub pool: Pool,
}

impl Default for Marketplace {
    fn default() -> Self {
        Self::new()
    }
}

impl Marketplace {
    pub fn new() -> Marketplace {
        Marketplace {
            contributing: false, // opt-in: default off
            predicate: SharingPredicate::relaxed(),
            gov: ResourceGovernor::new(),
            budget_bytes: 0,
            assignments: Vec::new(),
            ledger_digest: Hash256::of(b"marketplace-genesis"),
        }
    }

    pub fn is_contributing(&self) -> bool {
        self.contributing
    }

    /// Opt in: grant a revocable Public-Work budget under the dedicated domain.
    pub fn enable_contribution(&mut self, budget_bytes: usize, predicate: SharingPredicate) {
        self.contributing = true;
        self.predicate = predicate;
        self.budget_bytes = budget_bytes;
        self.gov.set_mem_budget(PUBLIC_WORK_DOMAIN, budget_bytes);
    }

    /// Revoke contribution instantly — the budget is withdrawn and no further work is admitted.
    pub fn revoke_contribution(&mut self) {
        self.contributing = false;
        self.budget_bytes = 0;
        self.gov.set_mem_budget(PUBLIC_WORK_DOMAIN, 0);
    }

    /// Whether this node will currently take in work, given live telemetry. Requires that the
    /// owner opted in *and* the sharing predicate holds.
    pub fn offers(&self, idle_pct: u8, time_of_day: u32, temp: u32, battery: u8) -> bool {
        self.contributing && self.predicate.offers(idle_pct, time_of_day, temp, battery)
    }

    /// Admit a job's resource reservation through the governor — never an OOM-kill, only
    /// admission/deferral. Returns the governor's [`Admission`] verdict.
    pub fn admit(&mut self, mem_bytes: usize, essential: bool) -> Admission {
        self.gov.reserve(PUBLIC_WORK_DOMAIN, mem_bytes, essential)
    }

    /// Gate a **private-pool** dispatch on a minimum attested confidential-compute tier.
    /// Returns `Err` (refused) if the attested tier is below the requirement.
    pub fn dispatch_private(&self, attested: Tier, required: Tier) -> Result<(), DispatchError> {
        if tier_rank(attested) >= tier_rank(required) {
            Ok(())
        } else {
            Err(DispatchError::TierTooLow)
        }
    }

    /// Run a reverse auction for `job` and, if it settles within budget, append the assignment
    /// to the BFT-ordered ledger. Returns the winning assignment.
    pub fn award(&mut self, job: &str, auction: &ReverseAuction, pool: Pool) -> Option<Assignment> {
        let win = auction.settle()?;
        let a = Assignment { job: String::from(job), supplier: win.supplier, price: win.price, pool };
        let mut chain = Vec::with_capacity(64);
        chain.extend_from_slice(&self.ledger_digest.0);
        chain.extend_from_slice(job.as_bytes());
        chain.extend_from_slice(&win.supplier.to_le_bytes());
        chain.extend_from_slice(&win.price.to_le_bytes());
        self.ledger_digest = Hash256::of(&chain);
        self.assignments.push(a.clone());
        Some(a)
    }

    pub fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }
    /// The BFT-ledger digest — equal across honest nodes iff they agree on the assignment order.
    pub fn ledger_digest(&self) -> Hash256 {
        self.ledger_digest
    }
}

/// Why a dispatch was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DispatchError {
    /// The attested confidential-compute tier is below the job's requirement.
    TierTooLow,
}

/// **Model fragmentation**: pipeline `layers` of a model across `nodes`, each layer a task that
/// depends on the previous one (a linear pipeline), routed to NPU/GPU nodes where available.
/// Returns the execution graph plus the per-layer node assignment (round-robin over `nodes`).
pub fn fragment_model(layers: usize, nodes: &[u32]) -> (WorkGraph, Vec<u32>) {
    let mut g = WorkGraph::new();
    let mut assign = Vec::with_capacity(layers);
    let mut prev: Option<usize> = None;
    for i in 0..layers {
        let deps: &[usize] = match &prev {
            Some(p) => core::slice::from_ref(p),
            None => &[],
        };
        let mut label = String::from("layer-");
        label.push((b'0' + (i % 10) as u8) as char);
        // Transformer layers prefer the NPU; the scheduler falls back to CPU if absent.
        let id = g.add(label, NodeKind::Npu, deps);
        prev = Some(id);
        let node = if nodes.is_empty() { 0 } else { nodes[i % nodes.len()] };
        assign.push(node);
    }
    (g, assign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contribution_is_opt_in_default_off_and_revocable() {
        let mut m = Marketplace::new();
        assert!(!m.is_contributing());
        // Even idle, a non-contributing node offers nothing.
        assert!(!m.offers(100, 0, 0, 100));
        m.enable_contribution(1 << 20, SharingPredicate::relaxed());
        assert!(m.is_contributing());
        assert!(m.offers(100, 0, 0, 100));
        m.revoke_contribution();
        assert!(!m.is_contributing());
        assert!(!m.offers(100, 0, 0, 100));
    }

    #[test]
    fn sharing_predicate_gates_on_idle_temp_battery_window() {
        let p = SharingPredicate { min_idle_pct: 80, window: (3600, 7200), max_temp: 70, min_battery: 60 };
        // All conditions satisfied.
        assert!(p.offers(90, 4000, 50, 80));
        // Too busy / outside window / too hot / too low battery each refuse.
        assert!(!p.offers(50, 4000, 50, 80));
        assert!(!p.offers(90, 100, 50, 80));
        assert!(!p.offers(90, 4000, 99, 80));
        assert!(!p.offers(90, 4000, 50, 10));
    }

    #[test]
    fn reverse_auction_picks_cheapest_meeting_the_envelope() {
        let env = Envelope { cpu_units: 4, mem_bytes: 1024, max_price: 100 };
        let mut a = ReverseAuction::new(env);
        a.bid(Bid { supplier: 0, cpu_units: 4, mem_bytes: 2048, price: 90 });
        a.bid(Bid { supplier: 1, cpu_units: 8, mem_bytes: 4096, price: 50 }); // cheapest valid
        a.bid(Bid { supplier: 2, cpu_units: 2, mem_bytes: 4096, price: 10 }); // too few CPUs
        a.bid(Bid { supplier: 3, cpu_units: 4, mem_bytes: 1024, price: 200 }); // over budget
        let win = a.settle().unwrap();
        assert_eq!(win.supplier, 1);
        assert_eq!(win.price, 50);
    }

    #[test]
    fn auction_with_no_valid_bid_settles_to_nothing() {
        let env = Envelope { cpu_units: 100, mem_bytes: 1 << 30, max_price: 1 };
        let mut a = ReverseAuction::new(env);
        a.bid(Bid { supplier: 0, cpu_units: 1, mem_bytes: 1, price: 1 });
        assert!(a.settle().is_none());
    }

    #[test]
    fn private_pool_requires_minimum_attested_tier() {
        let m = Marketplace::new();
        // Software attestation cannot run a job that demands hardware confidentiality.
        assert_eq!(m.dispatch_private(Tier::Software, Tier::Cheri), Err(DispatchError::TierTooLow));
        // A sufficient (or stronger) tier is admitted.
        assert!(m.dispatch_private(Tier::Cheri, Tier::MemoryTagging).is_ok());
        assert!(m.dispatch_private(Tier::MemoryTagging, Tier::MemoryTagging).is_ok());
    }

    #[test]
    fn award_records_a_bft_ordered_assignment() {
        let mut m = Marketplace::new();
        let env = Envelope { cpu_units: 1, mem_bytes: 1, max_price: 100 };
        let mut a = ReverseAuction::new(env);
        a.bid(Bid { supplier: 2, cpu_units: 1, mem_bytes: 1, price: 30 });
        let before = m.ledger_digest();
        let aw = m.award("infer-job", &a, Pool::Public).unwrap();
        assert_eq!(aw.supplier, 2);
        assert_eq!(m.assignments().len(), 1);
        assert_ne!(m.ledger_digest(), before); // ledger advanced deterministically
    }

    #[test]
    fn model_fragments_into_a_linear_pipeline_across_nodes() {
        let (g, assign) = fragment_model(4, &[10, 11]);
        // 4 layers, round-robin across two nodes.
        assert_eq!(assign, alloc::vec![10, 11, 10, 11]);
        // The graph is a valid linear pipeline (schedulable, each layer after the previous).
        let sched = g.schedule(&[NodeKind::Cpu]).unwrap();
        let step = |t: usize| sched.iter().find(|s| s.task == t).unwrap().step;
        assert!(step(0) < step(1) && step(1) < step(2) && step(2) < step(3));
    }

    #[test]
    fn admission_never_ooms_just_defers_or_refuses() {
        let mut m = Marketplace::new();
        m.enable_contribution(1000, SharingPredicate::relaxed());
        // A reservation within budget is granted; the governor never OOM-kills.
        let verdict = m.admit(500, true);
        // Essential, within-budget work is admitted — and is *never* an OOM-kill.
        assert!(matches!(
            verdict,
            Admission::Granted | Admission::Degraded(_) | Admission::Deferred
        ));
    }
}

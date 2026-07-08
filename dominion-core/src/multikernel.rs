//! Multikernel & heterogeneous scheduling — **Stage 4** + the **consistency model**
//! (see `docs/architecture/05-stage-04-multikernel-scheduling.md`).
//!
//! The machine is treated as a *network of cores*, Barrelfish-style: each core runs
//! a minimal **CPU driver** (a non-preemptible event handler) over its **own**
//! OS-state replica, and cores share **nothing implicitly** — all cross-core
//! interaction is **explicit message passing**, ordered by a Hybrid Logical Clock
//! ([`crate::hlc`]). On top sits a **global scheduler** that treats a computation as
//! an **execution graph** and routes each task to the best **heterogeneous node**
//! (CPU / GPU / NPU) — accelerators are first-class compute nodes, not peripherals.
//!
//! Two consistency paths are provided (the Stage 4 §4.3 extension): a **convergent**
//! (CRDT) path for state that may reconcile, and a **linearizable** (quorum
//! consensus) path for value-bearing state. Per-state the caller declares which it
//! needs. Pure, safe, host-tested.

use crate::datatypes::GCounter;
use crate::hlc::{Hlc, Timestamp};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

// ─────────────────────────── heterogeneous scheduling ───────────────────────────

/// A heterogeneous compute node. Accelerators are first-class, not peripherals.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Cpu,
    Gpu,
    Npu,
}

/// A task in the execution graph: a preferred node kind and the tasks it depends on.
#[derive(Clone, Debug)]
pub struct Task {
    pub label: String,
    pub prefers: NodeKind,
    pub deps: Vec<usize>,
}

/// A computation as a directed acyclic **execution graph** of tasks.
#[derive(Default)]
pub struct WorkGraph {
    tasks: Vec<Task>,
}

/// One scheduled task: which node it runs on, and at which step (topological wave).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Scheduled {
    pub task: usize,
    pub node: NodeKind,
    pub step: usize,
}

/// Why a graph could not be scheduled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SchedError {
    /// The dependency graph contains a cycle (not a DAG).
    Cycle,
    /// A task depends on a non-existent task id.
    BadDependency,
}

impl WorkGraph {
    pub fn new() -> WorkGraph {
        WorkGraph { tasks: Vec::new() }
    }

    /// Add a task; returns its id. `deps` must reference already-added tasks.
    pub fn add(&mut self, label: impl Into<String>, prefers: NodeKind, deps: &[usize]) -> usize {
        let id = self.tasks.len();
        self.tasks.push(Task { label: label.into(), prefers, deps: deps.to_vec() });
        id
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Schedule the graph: a deterministic topological order (Kahn's algorithm)
    /// that respects every dependency, routing each task to a node of its preferred
    /// kind if `available`, else falling back to the CPU. The global orchestrator.
    pub fn schedule(&self, available: &[NodeKind]) -> Result<Vec<Scheduled>, SchedError> {
        let n = self.tasks.len();
        // Validate dependencies and build in-degrees.
        let mut indeg = alloc::vec![0usize; n];
        for t in &self.tasks {
            for &d in &t.deps {
                if d >= n {
                    return Err(SchedError::BadDependency);
                }
            }
        }
        for (i, t) in self.tasks.iter().enumerate() {
            indeg[i] = t.deps.len();
        }

        let mut scheduled = Vec::with_capacity(n);
        let mut done = alloc::vec![false; n];
        let mut step = 0;
        let mut remaining = n;
        while remaining > 0 {
            // All tasks whose deps are satisfied form the next wave (deterministic
            // by ascending id), so independent tasks schedule together.
            let wave: Vec<usize> = (0..n).filter(|&i| !done[i] && indeg[i] == 0).collect();
            if wave.is_empty() {
                return Err(SchedError::Cycle);
            }
            for &i in &wave {
                let node = if available.contains(&self.tasks[i].prefers) {
                    self.tasks[i].prefers
                } else {
                    NodeKind::Cpu
                };
                scheduled.push(Scheduled { task: i, node, step });
                done[i] = true;
                remaining -= 1;
            }
            // Releasing this wave lowers the in-degree of its dependents. Decrement once
            // per matching occurrence so a duplicated dep id (e.g. [0, 0], counted with
            // multiplicity in indeg above) is fully released rather than stalling.
            for &i in &wave {
                for j in 0..n {
                    if !done[j] {
                        let occurrences = self.tasks[j].deps.iter().filter(|&&d| d == i).count();
                        indeg[j] -= occurrences;
                    }
                }
            }
            step += 1;
        }
        Ok(scheduled)
    }
}

// ─────────────────────────── cores + message passing ───────────────────────────

/// A message between cores. State only ever moves explicitly, as a message.
#[derive(Clone, Debug)]
pub struct CoreMsg {
    pub from: u64,
    pub stamp: Timestamp,
    pub key: String,
    pub value: i64,
}

/// A single core: a non-preemptible event handler over its **own** OS-state replica
/// and a private inbox. No other core can touch this state except by sending a
/// message it chooses to apply.
pub struct Core {
    pub id: u64,
    clock: Hlc,
    replica: BTreeMap<String, i64>,
    inbox: Vec<CoreMsg>,
}

impl Core {
    fn new(id: u64) -> Core {
        Core { id, clock: Hlc::new(), replica: BTreeMap::new(), inbox: Vec::new() }
    }

    /// Read this core's local replica.
    pub fn get(&self, key: &str) -> Option<i64> {
        self.replica.get(key).copied()
    }
}

/// A machine modelled as a network of cores with explicit message passing.
pub struct Multikernel {
    cores: Vec<Core>,
}

impl Multikernel {
    pub fn new(n_cores: usize) -> Multikernel {
        Multikernel { cores: (0..n_cores as u64).map(Core::new).collect() }
    }

    pub fn core(&self, id: u64) -> &Core {
        &self.cores[id as usize]
    }

    pub fn core_count(&self) -> usize {
        self.cores.len()
    }

    /// A core writes to its **own** replica (a local event, HLC-stamped).
    pub fn local_write(&mut self, core: u64, key: &str, value: i64, pt: u64) {
        let c = &mut self.cores[core as usize];
        let _ = c.clock.local(pt);
        c.replica.insert(String::from(key), value);
    }

    /// `from` sends a state update to `to`. Nothing is shared implicitly — the
    /// update sits in the recipient's inbox until it chooses to deliver.
    pub fn send(&mut self, from: u64, to: u64, key: &str, value: i64, pt: u64) {
        // On the pathological `CounterExhausted` case (logical counter wrap within
        // one physical tick), fall back to the clock's last issued stamp — never
        // panics and preserves monotonicity.
        let from_clock = &mut self.cores[from as usize].clock;
        let stamp = from_clock.local(pt).unwrap_or_else(|_| from_clock.now());
        self.cores[to as usize].inbox.push(CoreMsg {
            from,
            stamp,
            key: String::from(key),
            value,
        });
    }

    /// `core` processes its inbox: messages are applied to the local replica in
    /// **HLC order**, so delivery is deterministic regardless of arrival order
    /// (last-writer-wins by hybrid-logical timestamp).
    pub fn deliver(&mut self, core: u64, pt: u64) {
        let mut inbox = core::mem::take(&mut self.cores[core as usize].inbox);
        inbox.sort_by_key(|m| (m.stamp, m.from));
        // Track the winning stamp per key so a stale message never overwrites a
        // newer one (convergence under reordering).
        let mut winner: BTreeMap<String, Timestamp> = BTreeMap::new();
        for m in inbox {
            let c = &mut self.cores[core as usize];
            let _ = c.clock.receive(pt, m.stamp);
            let take = winner.get(&m.key).map(|w| m.stamp >= *w).unwrap_or(true);
            if take {
                winner.insert(m.key.clone(), m.stamp);
                c.replica.insert(m.key, m.value);
            }
        }
    }
}

// ─────────────────────────── consistency model ───────────────────────────

/// The consistency a piece of state demands. Declared per-object (Stage 4 §4.3).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Consistency {
    /// Convergent: any replica order reconciles to the same value (CRDT path).
    Convergent,
    /// Linearizable: a single agreed value via quorum consensus, honest replicas only.
    Linearizable,
    /// Byzantine-fault-tolerant: a single agreed value that survives up to `f` of `3f+1`
    /// *malicious* replicas (the fleet-scale value-bearing tier — see [`crate::bft`]).
    Bft,
}

impl Consistency {
    /// Whether this level tolerates Byzantine (lying / equivocating) replicas, not just
    /// crash-stop ones. Only [`Consistency::Bft`] does — it routes to [`crate::bft`].
    pub fn tolerates_byzantine(self) -> bool {
        matches!(self, Consistency::Bft)
    }
}

/// A convergent counter replicated across cores — the CRDT path. Merges commute,
/// so every core converges to the same value no matter the message order.
#[derive(Clone, Default)]
pub struct ConvergentState {
    counter: GCounter,
}

impl ConvergentState {
    pub fn new() -> ConvergentState {
        ConvergentState { counter: GCounter::new() }
    }
    pub fn bump(&mut self, core: u64, by: u64) {
        self.counter.increment(core, by);
    }
    pub fn merge(&self, other: &ConvergentState) -> ConvergentState {
        ConvergentState { counter: self.counter.merge(&other.counter) }
    }
    pub fn value(&self) -> u64 {
        self.counter.value()
    }
}

/// Quorum consensus for value-bearing (linearizable) state: a value is agreed only
/// if a strict majority of `n_cores` proposed it. Returns `None` without a quorum.
pub fn quorum_agree(proposals: &[(u64, i64)], n_cores: usize) -> Option<i64> {
    let mut tally: BTreeMap<i64, usize> = BTreeMap::new();
    let mut seen: BTreeMap<u64, ()> = BTreeMap::new();
    for &(core, value) in proposals {
        // One vote per core.
        if seen.insert(core, ()).is_none() {
            *tally.entry(value).or_insert(0) += 1;
        }
    }
    let need = n_cores / 2 + 1;
    tally.into_iter().find(|&(_, count)| count >= need).map(|(v, _)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_respects_dependencies_and_routes_by_kind() {
        let mut g = WorkGraph::new();
        let load = g.add("load", NodeKind::Cpu, &[]);
        let conv = g.add("convolve", NodeKind::Gpu, &[load]);
        let infer = g.add("infer", NodeKind::Npu, &[conv]);
        let sched = g.schedule(&[NodeKind::Cpu, NodeKind::Gpu, NodeKind::Npu]).unwrap();
        // Topological: load before convolve before infer.
        let step_of = |t: usize| sched.iter().find(|s| s.task == t).unwrap().step;
        assert!(step_of(load) < step_of(conv));
        assert!(step_of(conv) < step_of(infer));
        // Routed to the preferred heterogeneous node.
        assert_eq!(sched.iter().find(|s| s.task == conv).unwrap().node, NodeKind::Gpu);
        assert_eq!(sched.iter().find(|s| s.task == infer).unwrap().node, NodeKind::Npu);
    }

    #[test]
    fn missing_accelerator_falls_back_to_cpu() {
        let mut g = WorkGraph::new();
        let t = g.add("infer", NodeKind::Npu, &[]);
        // No NPU available → CPU fallback (graceful heterogeneity).
        let sched = g.schedule(&[NodeKind::Cpu]).unwrap();
        assert_eq!(sched[0].node, NodeKind::Cpu);
        let _ = t;
    }

    #[test]
    fn independent_tasks_share_a_step() {
        let mut g = WorkGraph::new();
        let a = g.add("a", NodeKind::Cpu, &[]);
        let b = g.add("b", NodeKind::Cpu, &[]);
        let join = g.add("join", NodeKind::Cpu, &[a, b]);
        let sched = g.schedule(&[NodeKind::Cpu]).unwrap();
        let step = |t: usize| sched.iter().find(|s| s.task == t).unwrap().step;
        // a and b are independent → same wave; join waits for both.
        assert_eq!(step(a), step(b));
        assert!(step(join) > step(a));
    }

    #[test]
    fn cycles_are_rejected() {
        // Build a 2-cycle by hand (add then point back).
        let mut g = WorkGraph::new();
        g.add("x", NodeKind::Cpu, &[1]); // depends on task 1
        g.add("y", NodeKind::Cpu, &[0]); // depends on task 0 → cycle
        assert_eq!(g.schedule(&[NodeKind::Cpu]), Err(SchedError::Cycle));
    }

    #[test]
    fn cores_share_nothing_until_a_message_is_delivered() {
        let mut mk = Multikernel::new(2);
        mk.local_write(0, "x", 7, 1);
        // Core 1 has not received anything → its replica is empty (no implicit share).
        assert_eq!(mk.core(1).get("x"), None);
        mk.send(0, 1, "x", 7, 2);
        // Still nothing until core 1 chooses to process its inbox.
        assert_eq!(mk.core(1).get("x"), None);
        mk.deliver(1, 3);
        assert_eq!(mk.core(1).get("x"), Some(7));
    }

    #[test]
    fn delivery_is_deterministic_under_reordering() {
        // Two cores send updates to core 2; whatever order they arrive, the HLC
        // order decides the winner, so the replica is deterministic.
        let mut mk = Multikernel::new(3);
        mk.local_write(0, "k", 1, 10);
        mk.local_write(1, "k", 2, 20); // later physical time → newer HLC
        mk.send(0, 2, "k", 1, 11);
        mk.send(1, 2, "k", 2, 21);
        mk.deliver(2, 30);
        assert_eq!(mk.core(2).get("k"), Some(2)); // newest stamp wins
    }

    #[test]
    fn convergent_crdt_path_reconciles_regardless_of_order() {
        // Three cores increment independently; merging in any order converges.
        let mut a = ConvergentState::new();
        let mut b = ConvergentState::new();
        let mut c = ConvergentState::new();
        a.bump(0, 3);
        b.bump(1, 4);
        c.bump(2, 5);
        let abc = a.merge(&b).merge(&c);
        let cba = c.merge(&b).merge(&a);
        assert_eq!(abc.value(), 12);
        assert_eq!(abc.value(), cba.value()); // commutative ⇒ convergent
    }

    #[test]
    fn linearizable_path_needs_a_quorum() {
        // 5 cores; a value is agreed only with a strict majority (≥3).
        let agreed = quorum_agree(&[(0, 42), (1, 42), (2, 42), (3, 7), (4, 7)], 5);
        assert_eq!(agreed, Some(42));
        // A split with no majority agrees on nothing.
        let split = quorum_agree(&[(0, 1), (1, 1), (2, 2), (3, 2)], 5);
        assert_eq!(split, None);
        // Declaring the consistency level is explicit, per the model.
        assert_ne!(Consistency::Convergent, Consistency::Linearizable);
    }

    #[test]
    fn bft_is_the_only_byzantine_tolerant_level() {
        // The new value-bearing tier: only Bft survives malicious replicas; quorum_agree
        // (Linearizable) assumes honest votes.
        assert!(Consistency::Bft.tolerates_byzantine());
        assert!(!Consistency::Linearizable.tolerates_byzantine());
        assert!(!Consistency::Convergent.tolerates_byzantine());
        // Strength ordering: Convergent < Linearizable < Bft.
        assert!(Consistency::Convergent < Consistency::Linearizable);
        assert!(Consistency::Linearizable < Consistency::Bft);
    }
}

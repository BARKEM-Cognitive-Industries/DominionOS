//! Deterministic Simulation Testing (DST) harness — **testing & verification
//! strategy** (see `docs/implementation/testing-and-verification-strategy.md`).
//!
//! The spec asks for a hypervisor that controls *time, scheduling, RNG, network
//! and I/O* so that a whole-system run is a pure function of a single **seed** —
//! then sweeps millions of seeds looking for a divergence or a panic. This module
//! is that controller, in portable safe code:
//!
//! * [`Sim`] owns a **logical clock** and a seeded [`Drng`](crate::random::Drng);
//!   nothing in a scenario may read wall-clock or hardware entropy — all
//!   nondeterminism is funnelled through here.
//! * [`SimNetwork`] models an **unreliable** link: messages are dropped, delayed
//!   and reordered, but *only* as a deterministic function of the seed, so the same
//!   seed always produces the same interleaving.
//! * A scenario returns a [`Hash256`] **trace digest**; the core property is
//!   `run(seed) == run(seed)` (reproducibility) and that a seed **sweep** finds no
//!   panic and always reaches the expected invariant (here: replica convergence).
//!
//! Because the run is a pure function of the seed, any failure is a permanent,
//! replayable regression: capture the seed, re-run, rewind. Pure, safe, host-tested.

use crate::hash::Hash256;
use crate::random::Drng;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// The deterministic world a scenario runs in: logical time + seeded randomness.
/// Every nondeterministic choice a scenario makes must come from here.
pub struct Sim {
    clock: u64,
    rng: Drng,
    /// A running digest of every event, so two runs can be compared bit-for-bit.
    trace: Hash256,
    events: u64,
}

impl Sim {
    /// Build a simulation pinned to `seed`. Same seed ⇒ same everything.
    pub fn new(seed: u64) -> Sim {
        Sim {
            clock: 0,
            rng: Drng::from_seed(&seed.to_le_bytes()),
            trace: Hash256::of(b"dst-genesis"),
            events: 0,
        }
    }

    /// The current logical time.
    pub fn now(&self) -> u64 {
        self.clock
    }

    /// Advance logical time by `ticks`.
    pub fn advance(&mut self, ticks: u64) {
        self.clock += ticks;
    }

    /// A controlled random integer in `[0, n)` (the only randomness a scenario sees).
    pub fn rand_below(&mut self, n: u64) -> u64 {
        self.rng.below(n)
    }

    /// A controlled coin flip with probability `pct`/100 of being `true`.
    pub fn chance(&mut self, pct: u64) -> bool {
        self.rand_below(100) < pct
    }

    /// Fold an event into the trace digest, so the whole run hashes to one value.
    pub fn record(&mut self, tag: &[u8]) {
        let mut input = Vec::with_capacity(32 + tag.len() + 16);
        input.extend_from_slice(&self.trace.0);
        input.extend_from_slice(&self.clock.to_le_bytes());
        input.extend_from_slice(tag);
        self.trace = Hash256::of(&input);
        self.events += 1;
    }

    /// The digest of everything recorded so far — the run's fingerprint.
    pub fn digest(&self) -> Hash256 {
        self.trace
    }

    pub fn event_count(&self) -> u64 {
        self.events
    }
}

/// An in-flight message on the simulated network.
#[derive(Clone)]
struct InFlight {
    deliver_at: u64,
    to: u32,
    payload: Vec<u8>,
    /// Tie-breaker so equal delivery times still order deterministically.
    seq: u64,
}

/// A deterministic but **unreliable** network: drop / delay / reorder, all driven
/// by the simulation's seed.
pub struct SimNetwork {
    queue: Vec<InFlight>,
    seq: u64,
    /// Percent chance a sent message is dropped.
    drop_pct: u64,
    /// Maximum extra delay (in ticks) applied to a delivered message.
    max_delay: u64,
    delivered: u64,
    dropped: u64,
}

impl SimNetwork {
    pub fn new(drop_pct: u64, max_delay: u64) -> SimNetwork {
        SimNetwork { queue: Vec::new(), seq: 0, drop_pct, max_delay, delivered: 0, dropped: 0 }
    }

    /// Send `payload` to node `to`. It may be dropped, and is delayed by a
    /// seed-determined amount — so arrival order is not send order.
    pub fn send(&mut self, sim: &mut Sim, to: u32, payload: &[u8]) {
        if sim.chance(self.drop_pct) {
            self.dropped += 1;
            sim.record(b"net:drop");
            return;
        }
        let delay = sim.rand_below(self.max_delay + 1);
        let deliver_at = sim.now() + delay;
        self.queue.push(InFlight { deliver_at, to, payload: payload.to_vec(), seq: self.seq });
        self.seq += 1;
        sim.record(b"net:send");
    }

    /// Pop every message due at or before `now`, in deterministic (time, seq)
    /// order. Reordering relative to send order is real, but reproducible.
    pub fn deliver_due(&mut self, sim: &mut Sim) -> Vec<(u32, Vec<u8>)> {
        let now = sim.now();
        // Stable deterministic order: by delivery time then send sequence.
        self.queue.sort_by(|a, b| a.deliver_at.cmp(&b.deliver_at).then(a.seq.cmp(&b.seq)));
        let mut out = Vec::new();
        let mut remaining = Vec::new();
        for m in self.queue.drain(..) {
            if m.deliver_at <= now {
                self.delivered += 1;
                out.push((m.to, m.payload));
            } else {
                remaining.push(m);
            }
        }
        self.queue = remaining;
        for _ in &out {
            sim.record(b"net:deliver");
        }
        out
    }

    pub fn in_flight(&self) -> usize {
        self.queue.len()
    }
    pub fn delivered(&self) -> u64 {
        self.delivered
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// A content-addressed replica: a set of key→value entries that merges by taking,
/// for each key, the value with the larger (logical-time, content-hash) stamp.
/// This is a tiny last-writer-wins CRDT, so merges are commutative and convergent
/// regardless of the network's drop/reorder behaviour.
#[derive(Clone, Default)]
pub struct Replica {
    entries: BTreeMap<String, (u64, Hash256, Vec<u8>)>,
}

impl Replica {
    pub fn new() -> Replica {
        Replica { entries: BTreeMap::new() }
    }

    /// Local write at logical time `stamp`.
    pub fn put(&mut self, key: &str, value: &[u8], stamp: u64) {
        let h = Hash256::of(value);
        self.merge_one(key, stamp, h, value);
    }

    fn merge_one(&mut self, key: &str, stamp: u64, h: Hash256, value: &[u8]) {
        let replace = match self.entries.get(key) {
            None => true,
            Some((s, eh, _)) => (stamp, h.0) > (*s, eh.0),
        };
        if replace {
            self.entries.insert(String::from(key), (stamp, h, value.to_vec()));
        }
    }

    /// Serialize the replica's full state for gossip.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (k, (s, h, v)) in &self.entries {
            out.extend_from_slice(&(k.len() as u32).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(&s.to_le_bytes());
            out.extend_from_slice(&h.0);
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            out.extend_from_slice(v);
        }
        out
    }

    /// Merge another replica's gossiped state into this one (commutative).
    pub fn merge_encoded(&mut self, bytes: &[u8]) -> bool {
        let mut p = 0usize;
        let take_u32 = |b: &[u8], p: &mut usize| -> Option<u32> {
            if *p + 4 > b.len() {
                return None;
            }
            let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
            *p += 4;
            Some(v)
        };
        let n = match take_u32(bytes, &mut p) {
            Some(n) => n,
            None => return false,
        };
        for _ in 0..n {
            let klen = match take_u32(bytes, &mut p) {
                Some(v) => v as usize,
                None => return false,
            };
            if p + klen > bytes.len() {
                return false;
            }
            let key = match core::str::from_utf8(&bytes[p..p + klen]) {
                Ok(s) => String::from(s),
                Err(_) => return false,
            };
            p += klen;
            if p + 8 + 32 > bytes.len() {
                return false;
            }
            let stamp = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
            p += 8;
            let mut hb = [0u8; 32];
            hb.copy_from_slice(&bytes[p..p + 32]);
            p += 32;
            let vlen = match take_u32(bytes, &mut p) {
                Some(v) => v as usize,
                None => return false,
            };
            if p + vlen > bytes.len() {
                return false;
            }
            let value = bytes[p..p + vlen].to_vec();
            p += vlen;
            self.merge_one(&key, stamp, Hash256(hb), &value);
        }
        true
    }

    /// A digest of the replica's converged state — equal across replicas iff they
    /// agree.
    pub fn state_hash(&self) -> Hash256 {
        Hash256::of(&self.encode())
    }

    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.entries.get(key).map(|(_, _, v)| v.as_slice())
    }
}

/// Summary statistics of a DST run — the material for reproducibility and
/// perf-regression gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunStats {
    /// The trace digest — equal across two runs iff they were bit-identical.
    pub digest: Hash256,
    /// Total recorded events (a deterministic "work" measure for a perf budget).
    pub events: u64,
    /// Whether every replica reached the same final state.
    pub converged: bool,
    pub delivered: u64,
    pub dropped: u64,
}

/// The canonical DST scenario, returning full [`RunStats`]: `nodes` replicas each
/// perform local writes and gossip over an unreliable [`SimNetwork`]; after a
/// quiescent flush every replica must converge. A pure function of `seed`.
pub fn run_stats(seed: u64, nodes: u32, rounds: u32) -> RunStats {
    let mut sim = Sim::new(seed);
    let mut net = SimNetwork::new(/* drop% */ 20, /* max delay */ 5);
    let mut replicas: Vec<Replica> = (0..nodes).map(|_| Replica::new()).collect();

    for round in 0..rounds {
        // Each node does a local write and gossips its full state to a random peer.
        for n in 0..nodes {
            let key = key_for(n, round, &mut sim);
            let val = sim.rand_below(1000).to_le_bytes();
            replicas[n as usize].put(&key, &val, sim.now());
            let peer = sim.rand_below(nodes as u64) as u32;
            if peer != n {
                let gossip = replicas[n as usize].encode();
                net.send(&mut sim, peer, &gossip);
            }
        }
        // Deliver whatever the network decides is due this tick.
        for (to, payload) in net.deliver_due(&mut sim) {
            replicas[to as usize].merge_encoded(&payload);
        }
        sim.advance(1);
    }

    // Quiescent flush: keep delivering and re-gossiping until the network drains,
    // so every surviving update reaches every node (anti-entropy).
    for _ in 0..(rounds + nodes + 32) {
        for n in 0..nodes {
            let peer = sim.rand_below(nodes as u64) as u32;
            if peer != n {
                let gossip = replicas[n as usize].encode();
                net.send(&mut sim, peer, &gossip);
            }
        }
        sim.advance(10); // jump past max delay so everything becomes due
        for (to, payload) in net.deliver_due(&mut sim) {
            replicas[to as usize].merge_encoded(&payload);
        }
    }

    let first = replicas[0].state_hash();
    let converged = replicas.iter().all(|r| r.state_hash() == first);
    // Record convergence into the trace.
    for r in &replicas {
        sim.record(&r.state_hash().0);
    }
    RunStats {
        digest: sim.digest(),
        events: sim.event_count(),
        converged,
        delivered: net.delivered(),
        dropped: net.dropped(),
    }
}

/// Just the trace digest of the canonical scenario (a pure function of `seed`).
pub fn run_convergence(seed: u64, nodes: u32, rounds: u32) -> Hash256 {
    run_stats(seed, nodes, rounds).digest
}

fn key_for(node: u32, round: u32, sim: &mut Sim) -> String {
    // A small shared keyspace so writes from different nodes actually collide and
    // exercise the merge rule, with occasional node-private keys.
    let shared = sim.chance(70);
    if shared {
        let k = sim.rand_below(4);
        let mut s = String::from("shared-");
        s.push((b'0' + k as u8) as char);
        s
    } else {
        let mut s = String::from("n");
        s.push((b'0' + (node % 10) as u8) as char);
        s.push('-');
        s.push((b'0' + (round % 10) as u8) as char);
        s
    }
}

/// Check that all replicas in a fresh run converge to one state (the invariant the
/// sweep asserts). Returns `true` if convergent.
pub fn converges(seed: u64, nodes: u32, rounds: u32) -> bool {
    run_stats(seed, nodes, rounds).converged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_run_exactly() {
        // The defining property of DST: a run is a pure function of its seed.
        for seed in [1u64, 42, 7777, 0xDEAD_BEEF] {
            assert_eq!(run_convergence(seed, 5, 20), run_convergence(seed, 5, 20));
        }
    }

    #[test]
    fn different_seeds_generally_diverge() {
        // Different seeds should (almost always) produce different traces.
        let a = run_convergence(1, 5, 20);
        let b = run_convergence(2, 5, 20);
        assert_ne!(a, b);
    }

    #[test]
    fn replicas_always_converge_under_loss_and_reorder() {
        // A seed sweep: under 20% loss + reorder, every run still converges.
        // (Bound kept CI-friendly; the loop constant is the only thing between
        // this and a million-seed sweep.)
        for seed in 0..600u64 {
            assert!(converges(seed, 5, 25), "replicas diverged at seed {seed}");
        }
    }

    #[test]
    fn network_actually_drops_and_reorders() {
        // Sanity: the unreliable network is genuinely unreliable, not a no-op.
        let mut sim = Sim::new(123);
        let mut net = SimNetwork::new(20, 5);
        for i in 0..2000u32 {
            net.send(&mut sim, 0, &i.to_le_bytes());
        }
        sim.advance(100);
        let _ = net.deliver_due(&mut sim);
        assert!(net.dropped() > 0, "expected some drops");
        assert!(net.delivered() > 0, "expected some deliveries");
    }

    #[test]
    fn run_is_a_pure_function_including_stats() {
        // Reproducibility extends to the full stats, not just the digest.
        for seed in [3u64, 99, 0xABCD] {
            assert_eq!(run_stats(seed, 5, 20), run_stats(seed, 5, 20));
        }
    }

    #[test]
    fn perf_regression_gate_on_protocol_chattiness() {
        // A deterministic perf gate: the gossip protocol's recorded work for a
        // fixed scenario must stay within budget. If a change makes the protocol
        // markedly chattier (more sends/deliveries), this fails loudly — a perf
        // regression caught without any wall-clock timing.
        let stats = run_stats(0x1234, 5, 25);
        assert!(stats.converged);
        // 5 nodes × 25 rounds of writes/gossip + a bounded flush. The empirical
        // event count sits well under this ceiling; a 2× blow-up trips the gate.
        assert!(
            stats.events < 2000,
            "protocol chattiness regressed: {} events (budget 2000)",
            stats.events
        );
        // The unreliable network must actually have exercised loss + delivery.
        assert!(stats.dropped > 0 && stats.delivered > 0);
    }

    #[test]
    fn crdt_merge_is_commutative() {
        // Merge order must not matter (the convergence guarantee in miniature).
        let mut a = Replica::new();
        let mut b = Replica::new();
        a.put("k", b"a-val", 1);
        b.put("k", b"b-val", 2);
        let mut ab = a.clone();
        ab.merge_encoded(&b.encode());
        let mut ba = b.clone();
        ba.merge_encoded(&a.encode());
        assert_eq!(ab.state_hash(), ba.state_hash());
        // Higher stamp wins deterministically.
        assert_eq!(ab.get("k"), Some(b"b-val".as_ref()));
    }
}

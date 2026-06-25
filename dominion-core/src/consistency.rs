//! Causally-fenced rollback for distributed state — **the hard problem of safe
//! rollback under concurrency** (red-team finding A; see `docs/findings.md` and
//! `docs/architecture/11-stage-10-deterministic-state.md`).
//!
//! DominionOS rolls back state to contain a misbehaving app ("freeze the spread,
//! roll back everything around it"). In a *single* machine that is a head-pointer
//! flip ([`crate::object::ObjectGraph::rollback`]). Across a **fleet of gossiping
//! replicas** it is the classic distributed-systems trap: a plain last-writer-wins
//! merge (as in [`crate::dst::Replica`]) lets a rolled-back value be **resurrected**
//! by a peer that still holds it — the "split-timeline merge corruption" an
//! experienced kernel engineer would immediately probe for.
//!
//! This module makes rollback safe under concurrency with two joined ideas, both
//! commutative so convergence is preserved:
//!
//! * **Causal fence.** Every key carries a monotone **fence** ([`Timestamp`], a
//!   hybrid logical clock value). A write is only accepted if its stamp *dominates*
//!   the fence. A rollback **raises the fence past the bad write** and re-asserts the
//!   known-good value at a fresh, higher stamp. The fence travels in gossip and
//!   merges by `max`, so once any replica fences a timeline, the abandoned writes can
//!   never re-enter — resurrection is impossible by construction.
//! * **Pinned roots.** Identity roots and capability tables are **never rolled back**
//!   (the invariant the conversation asks to pin down explicitly). [`pin`] marks a key
//!   immutable to rollback; [`rollback_key`] refuses to touch it.
//!
//! Merges remain a join-semilattice (`fence = max`, value = the dominating
//! non-fenced write), so replicas still converge regardless of drop/reorder/partition
//! — now *including* across rollbacks. Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use crate::hlc::Timestamp;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

/// One key's converged cell: the dominating write plus the causal fence below which
/// no write (old, concurrent, or resurrected) may ever be accepted again.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Cell {
    stamp: Timestamp,
    vhash: Hash256,
    value: Vec<u8>,
    /// Writes with `stamp <= fence` are rejected forever (anti-resurrection).
    fence: Timestamp,
    /// True once a rollback fenced the timeline but no good value sits above the
    /// fence yet (the key is logically empty, but the fence must still propagate).
    tombstoned: bool,
}

/// Why a write or rollback was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FenceError {
    /// The write's stamp does not dominate the key's fence (stale / abandoned timeline).
    BelowFence,
    /// The key is pinned (identity root / capability table): never rolled back.
    Pinned,
    /// No prior committed value to roll the key back to.
    NoHistory,
}

/// A content-addressed replica with causally-fenced rollback. Drop-in stronger
/// sibling of [`crate::dst::Replica`]: same gossip/convergence guarantees, but a
/// rolled-back write can never be resurrected by a lagging peer.
#[derive(Clone, Default)]
pub struct FencedReplica {
    cells: BTreeMap<String, Cell>,
    /// Keys that may never be rolled back (identity root, capability tables, keys).
    pinned: BTreeSet<String>,
}

impl FencedReplica {
    pub fn new() -> FencedReplica {
        FencedReplica { cells: BTreeMap::new(), pinned: BTreeSet::new() }
    }

    /// Mark `key` rollback-immutable. Pinned roots are the invariant that survives
    /// every rollback — the answer to "what is *never* rolled back".
    pub fn pin(&mut self, key: &str) {
        self.pinned.insert(String::from(key));
    }

    pub fn is_pinned(&self, key: &str) -> bool {
        self.pinned.contains(key)
    }

    /// Local write at hybrid-logical `stamp`. Rejected if it does not dominate the
    /// key's fence — so a node cannot (even by accident) write into an abandoned
    /// timeline. Returns the accepted stamp on success.
    pub fn put(&mut self, key: &str, value: &[u8], stamp: Timestamp) -> Result<(), FenceError> {
        let h = Hash256::of(value);
        let fence = self.cells.get(key).map(|c| c.fence).unwrap_or_default();
        if stamp <= fence {
            return Err(FenceError::BelowFence);
        }
        self.write(key, stamp, h, value, fence);
        Ok(())
    }

    /// The committed value of a key (None if absent or tombstoned).
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.cells.get(key).filter(|c| !c.tombstoned).map(|c| c.value.as_slice())
    }

    /// The current fence for a key (Default if untouched).
    pub fn fence_of(&self, key: &str) -> Timestamp {
        self.cells.get(key).map(|c| c.fence).unwrap_or_default()
    }

    /// **Roll a key back** to `good_value`, fencing out the bad write and everything
    /// causally concurrent with it. `now` is the local HLC reading; the restored value
    /// is stamped strictly above the new fence so it dominates any in-flight bad gossip.
    /// Refused for pinned keys. This is the contained, non-destructive recovery the OS
    /// performs when an app "spazzes out".
    pub fn rollback_key(&mut self, key: &str, good_value: &[u8], now: u64) -> Result<(), FenceError> {
        if self.pinned.contains(key) {
            return Err(FenceError::Pinned);
        }
        let cell = self.cells.get(key).ok_or(FenceError::NoHistory)?;
        // Fence at the bad write's stamp: it and anything <= it can never return.
        let new_fence = cell.stamp.max(cell.fence);
        // Restore the good value at a fresh stamp strictly above the fence.
        let restored = Timestamp { wall: new_fence.wall.max(now), logical: new_fence.logical + 1 };
        let h = Hash256::of(good_value);
        self.write(key, restored, h, good_value, new_fence);
        Ok(())
    }

    /// Roll a key back to *empty* (pure containment: the bad write is erased, the key
    /// is fenced and left tombstoned until a fresh good write arrives).
    pub fn quarantine_key(&mut self, key: &str, now: u64) -> Result<(), FenceError> {
        if self.pinned.contains(key) {
            return Err(FenceError::Pinned);
        }
        let cell = self.cells.get_mut(key).ok_or(FenceError::NoHistory)?;
        let new_fence = Timestamp { wall: cell.stamp.wall.max(cell.fence.wall).max(now), logical: cell.stamp.logical + 1 };
        cell.fence = new_fence;
        cell.tombstoned = true;
        cell.value.clear();
        cell.vhash = Hash256::ZERO;
        cell.stamp = new_fence; // sits exactly at the fence: no value dominates it
        Ok(())
    }

    fn write(&mut self, key: &str, stamp: Timestamp, vhash: Hash256, value: &[u8], fence: Timestamp) {
        self.cells.insert(
            String::from(key),
            Cell { stamp, vhash, value: value.to_vec(), fence, tombstoned: false },
        );
    }

    /// Serialize for gossip. Carries the fence, so peers learn the abandoned timeline.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.cells.len() as u32).to_le_bytes());
        for (k, c) in &self.cells {
            out.extend_from_slice(&(k.len() as u32).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            put_ts(&mut out, c.stamp);
            put_ts(&mut out, c.fence);
            out.push(c.tombstoned as u8);
            out.extend_from_slice(&c.vhash.0);
            out.extend_from_slice(&(c.value.len() as u32).to_le_bytes());
            out.extend_from_slice(&c.value);
        }
        out
    }

    /// Merge gossiped state. The merge is commutative and idempotent:
    /// `fence = max(fences)`, then the dominating `(stamp, vhash)` write whose stamp is
    /// **strictly above the merged fence** wins; if none qualifies the key is left
    /// tombstoned at the fence. Pinned keys ignore inbound fences (never rolled back).
    pub fn merge_encoded(&mut self, bytes: &[u8]) -> bool {
        let mut r = Cursor { b: bytes, p: 0 };
        let n = match r.u32() {
            Some(n) => n,
            None => return false,
        };
        for _ in 0..n {
            let klen = match r.u32() {
                Some(v) => v as usize,
                None => return false,
            };
            let key = match r.bytes(klen).and_then(|s| core::str::from_utf8(s).ok()) {
                Some(s) => String::from(s),
                None => return false,
            };
            let stamp = match get_ts(&mut r) {
                Some(t) => t,
                None => return false,
            };
            let fence_in = match get_ts(&mut r) {
                Some(t) => t,
                None => return false,
            };
            let tomb_in = match r.u8() {
                Some(b) => b != 0,
                None => return false,
            };
            let vhash = match r.bytes(32) {
                Some(b) => {
                    let mut a = [0u8; 32];
                    a.copy_from_slice(b);
                    Hash256(a)
                }
                None => return false,
            };
            let vlen = match r.u32() {
                Some(v) => v as usize,
                None => return false,
            };
            let value = match r.bytes(vlen) {
                Some(b) => b.to_vec(),
                None => return false,
            };
            self.merge_one(&key, stamp, fence_in, tomb_in, vhash, value);
        }
        true
    }

    fn merge_one(&mut self, key: &str, stamp: Timestamp, fence_in: Timestamp, tomb_in: bool, vhash: Hash256, value: Vec<u8>) {
        // Pinned keys never accept an inbound fence (they are never rolled back) but
        // still take the dominating value so they converge.
        let pinned = self.pinned.contains(key);
        let local = self.cells.get(key).cloned();
        let local_fence = local.as_ref().map(|c| c.fence).unwrap_or_default();
        let fence = if pinned { local_fence } else { local_fence.max(fence_in) };

        // Gather the two candidate writes and keep the dominating one above the fence.
        let mut best: Option<(Timestamp, Hash256, Vec<u8>, bool)> = None;
        let consider = |stamp: Timestamp, vhash: Hash256, value: Vec<u8>, tomb: bool, best: &mut Option<(Timestamp, Hash256, Vec<u8>, bool)>| {
            if tomb || stamp <= fence {
                return; // fenced out or tombstoned: cannot be the live value
            }
            let key_tuple = (stamp, vhash.0);
            let dominates = match best {
                None => true,
                Some((s, h, _, _)) => key_tuple > (*s, h.0),
            };
            if dominates {
                *best = Some((stamp, vhash, value, false));
            }
        };
        if let Some(c) = &local {
            consider(c.stamp, c.vhash, c.value.clone(), c.tombstoned, &mut best);
        }
        consider(stamp, vhash, value, tomb_in, &mut best);

        let cell = match best {
            Some((s, h, v, _)) => Cell { stamp: s, vhash: h, value: v, fence, tombstoned: false },
            None => Cell { stamp: fence, vhash: Hash256::ZERO, value: Vec::new(), fence, tombstoned: true },
        };
        self.cells.insert(String::from(key), cell);
    }

    /// A digest of the converged state — equal across replicas iff they agree
    /// (includes fences, so two replicas that disagree on a fenced timeline differ).
    pub fn state_hash(&self) -> Hash256 {
        Hash256::of(&self.encode())
    }

    /// Does any live key currently hold `value`? (used to prove non-resurrection).
    pub fn holds_value(&self, value: &[u8]) -> bool {
        let h = Hash256::of(value);
        self.cells.values().any(|c| !c.tombstoned && c.vhash == h)
    }
}

fn put_ts(out: &mut Vec<u8>, t: Timestamp) {
    out.extend_from_slice(&t.wall.to_le_bytes());
    out.extend_from_slice(&t.logical.to_le_bytes());
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }
    fn u32(&mut self) -> Option<u32> {
        let s = self.bytes(4)?;
        Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.p + n > self.b.len() {
            return None;
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Some(s)
    }
}

fn get_ts(r: &mut Cursor) -> Option<Timestamp> {
    let w = r.bytes(8)?;
    let wall = u64::from_le_bytes(w.try_into().unwrap());
    let l = r.bytes(4)?;
    let logical = u32::from_le_bytes(l.try_into().unwrap());
    Some(Timestamp { wall, logical })
}

// ───────────────────────── DST scenario: rollback never resurrects ─────────────────────────

/// A deterministic, seed-driven scenario that reproduces the exact failure mode the
/// red team flagged and asserts the fix holds: `nodes` replicas converge under
/// loss/reorder/**partition**, one node rolls a poisoned write back, and the bad value
/// must never reappear on *any* replica after the network heals. Returns
/// `(converged, bad_value_resurrected)`. A pure function of `seed`.
pub fn rollback_safety_scenario(seed: u64, nodes: u32, rounds: u32) -> (bool, bool) {
    use crate::dst::{Sim, SimNetwork};

    let mut sim = Sim::new(seed);
    let mut net = SimNetwork::new(25, 6);
    let mut replicas: Vec<FencedReplica> = (0..nodes).map(|_| FencedReplica::new()).collect();
    let mut clocks: Vec<u64> = alloc::vec![0; nodes as usize];
    let bad = b"POISONED-by-spazzing-app";

    // Phase 1: node 0 writes the poisoned value and gossips it widely.
    clocks[0] += 1;
    let _ = replicas[0].put("shared", bad, Timestamp { wall: clocks[0], logical: 0 });
    for peer in 1..nodes {
        let g = replicas[0].encode();
        net.send(&mut sim, peer, &g);
    }
    for (to, payload) in net.deliver_due(&mut sim) {
        replicas[to as usize].merge_encoded(&payload);
    }
    sim.advance(1);

    // Phase 2: node 0 detects the fault and rolls the key back to a good value while
    // peers may still be holding/forwarding the poison (concurrent rollback).
    clocks[0] += 1;
    let _ = replicas[0].rollback_key("shared", b"clean-recovered-state", clocks[0]);

    // Phase 3: chaotic gossip with a transient partition (nodes >= nodes/2 isolated
    // for the first third of the rounds), then full heal + anti-entropy flush.
    for round in 0..rounds {
        let partitioned = round < rounds / 3;
        for n in 0..nodes {
            let isolated = partitioned && n >= nodes / 2;
            clocks[n as usize] += 1;
            // Each node makes a benign write to a private key (exercises the merge).
            let mut k = String::from("n");
            k.push((b'0' + (n % 10) as u8) as char);
            let _ = replicas[n as usize].put(&k, &round.to_le_bytes(), Timestamp { wall: clocks[n as usize], logical: 0 });
            if isolated {
                continue;
            }
            let peer = sim.rand_below(nodes as u64) as u32;
            if peer != n && !(partitioned && peer >= nodes / 2) {
                let g = replicas[n as usize].encode();
                net.send(&mut sim, peer, &g);
            }
        }
        for (to, payload) in net.deliver_due(&mut sim) {
            replicas[to as usize].merge_encoded(&payload);
        }
        sim.advance(1);
    }

    // Heal + anti-entropy: keep gossiping until the network drains.
    for _ in 0..(rounds + nodes + 64) {
        for n in 0..nodes {
            let peer = sim.rand_below(nodes as u64) as u32;
            if peer != n {
                let g = replicas[n as usize].encode();
                net.send(&mut sim, peer, &g);
            }
        }
        sim.advance(10);
        for (to, payload) in net.deliver_due(&mut sim) {
            replicas[to as usize].merge_encoded(&payload);
        }
    }

    let first = replicas[0].state_hash();
    let converged = replicas.iter().all(|r| r.state_hash() == first);
    let resurrected = replicas.iter().any(|r| r.holds_value(bad));
    (converged, resurrected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(w: u64, l: u32) -> Timestamp {
        Timestamp { wall: w, logical: l }
    }

    #[test]
    fn fence_rejects_writes_from_an_abandoned_timeline() {
        let mut r = FencedReplica::new();
        r.put("k", b"v1", ts(1, 0)).unwrap();
        r.rollback_key("k", b"good", 1).unwrap();
        // A late write at the *old* stamp is below the fence and refused.
        assert_eq!(r.put("k", b"resurrected", ts(1, 0)), Err(FenceError::BelowFence));
        assert_eq!(r.get("k"), Some(b"good".as_ref()));
    }

    #[test]
    fn merge_never_resurrects_a_rolled_back_value() {
        // Node A wrote a bad value, gossiped it to B, then rolled back. B re-gossips
        // the bad value back to A — it must NOT reappear.
        let mut a = FencedReplica::new();
        let mut b = FencedReplica::new();
        a.put("k", b"bad", ts(5, 0)).unwrap();
        b.merge_encoded(&a.encode()); // B now holds "bad"
        a.rollback_key("k", b"good", 5).unwrap(); // A fences past stamp (5,0)
        // B (still holding bad) gossips back to A — fence rejects it.
        a.merge_encoded(&b.encode());
        assert_eq!(a.get("k"), Some(b"good".as_ref()));
        assert!(!a.holds_value(b"bad"));
        // And once A's fenced state reaches B, B drops the bad value too (converges).
        b.merge_encoded(&a.encode());
        assert_eq!(b.get("k"), Some(b"good".as_ref()));
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn merge_is_commutative_across_a_rollback() {
        let mut base = FencedReplica::new();
        base.put("k", b"bad", ts(3, 0)).unwrap();
        let mut a = base.clone();
        a.rollback_key("k", b"good", 3).unwrap();
        let b = base.clone();
        // Merge order must not matter.
        let mut ab = a.clone();
        ab.merge_encoded(&b.encode());
        let mut ba = b.clone();
        ba.merge_encoded(&a.encode());
        assert_eq!(ab.state_hash(), ba.state_hash());
        assert!(!ab.holds_value(b"bad"));
    }

    #[test]
    fn pinned_roots_are_never_rolled_back() {
        let mut r = FencedReplica::new();
        r.pin("identity-root");
        r.put("identity-root", b"the-master-id", ts(1, 0)).unwrap();
        assert_eq!(r.rollback_key("identity-root", b"attacker", 9), Err(FenceError::Pinned));
        assert_eq!(r.quarantine_key("identity-root", 9), Err(FenceError::Pinned));
        assert_eq!(r.get("identity-root"), Some(b"the-master-id".as_ref()));
    }

    #[test]
    fn quarantine_contains_then_accepts_fresh_good_state() {
        let mut r = FencedReplica::new();
        r.put("k", b"bad", ts(2, 0)).unwrap();
        r.quarantine_key("k", 2).unwrap();
        assert_eq!(r.get("k"), None); // contained
                                      // A fresh good write above the fence is accepted.
        let f = r.fence_of("k");
        r.put("k", b"recovered", Timestamp { wall: f.wall + 1, logical: 0 }).unwrap();
        assert_eq!(r.get("k"), Some(b"recovered".as_ref()));
    }

    #[test]
    fn dst_rollback_is_safe_under_loss_reorder_and_partition() {
        // The headline property: across a seed sweep, replicas converge AND the
        // poisoned value is never resurrected on any node, despite loss/reorder and a
        // transient network partition during the rollback.
        for seed in 0..300u64 {
            let (converged, resurrected) = rollback_safety_scenario(seed, 5, 24);
            assert!(converged, "replicas diverged at seed {seed}");
            assert!(!resurrected, "rolled-back value resurrected at seed {seed}");
        }
    }

    #[test]
    fn scenario_is_a_pure_function_of_seed() {
        assert_eq!(rollback_safety_scenario(42, 5, 20), rollback_safety_scenario(42, 5, 20));
    }
}

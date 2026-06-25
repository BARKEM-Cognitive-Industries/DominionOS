//! Fleet-scale **Byzantine-fault-tolerant consensus** — the value-bearing tier the
//! distributed subsystems (`docs/architecture/distributed-sasos-and-global-address-space.md`,
//! `…/decentralized-compute-marketplace.md`, `economics/…proof-of-useful-work.md`) all assume.
//!
//! [`crate::multikernel::quorum_agree`] gives a strict-majority quorum that is safe only
//! when replicas are *honest* (crash-stop). Across an open fleet a replica may be
//! **Byzantine**: silent, lying, or **equivocating** (signing two different values for the
//! same slot). This module upgrades the value-bearing path to tolerate up to `f` such
//! faults out of `3f+1` replicas, the classic optimal bound:
//!
//! * **PBFT-style 3-phase commit** — `PrePrepare` (leader) → `Prepare` → `Commit`. A slot
//!   commits only on a **quorum certificate** of `2f+1` matching votes. Any two `2f+1`
//!   quorums of `3f+1` intersect in `≥ f+1` replicas, i.e. ≥1 honest — so two honest
//!   replicas can never commit different values for one slot (**safety**).
//! * **View-change / leader rotation** — a faulty (silent) leader is replaced: `2f+1`
//!   `ViewChange` votes advance the view; the new leader re-proposes the highest **locked**
//!   value carried in those votes, so nothing potentially-committed is lost (**liveness +
//!   cross-view safety**).
//! * **Equivocation detection** — a replica double-signing two values for one
//!   `(view, slot, phase)` is caught by [`EquivocationProof`]; both votes verify under the
//!   same published key, which is undeniable evidence — fed to [`crate::threat`].
//! * **Determinism boundary** — every committed slot is folded into a hash-chained
//!   [`Bft::log_digest`], so a fleet's agreed history replays bit-for-bit under DST.
//!
//! Signatures are an **XMSS-style** hash-based scheme (`crate::crypto::LamportSig` one-time
//! keys under a Merkle root — the same construction as [`crate::tokensig`]) so each validator
//! has one long-lived public key (its root) yet can sign many votes, and forgery needs only
//! inverting SHA-256 (post-quantum). Pure, safe `no_std`, host- and metal-tested.

use crate::crypto::{LamportSig, SignatureScheme};
use crate::hash::Hash256;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

// ───────────────────────────── XMSS-style multi-message signer ─────────────────────────────

fn merkle_node(a: &Hash256, b: &Hash256) -> Hash256 {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&a.0);
    input[32..].copy_from_slice(&b.0);
    Hash256::of(&input)
}

fn merkle_root(leaves: &[Hash256]) -> Hash256 {
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                next.push(merkle_node(&pair[0], &pair[1]));
            } else {
                next.push(merkle_node(&pair[0], &pair[0]));
            }
        }
        level = next;
    }
    level.first().copied().unwrap_or(Hash256::ZERO)
}

/// A one-time signature plus the Merkle path proving its key belongs to a validator root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OtsSig {
    index: usize,
    sig: Vec<u8>,
    ots_pub: Vec<u8>,
    path: Vec<Hash256>,
}

/// A validator's signing authority: `2^height` one-time keys under one Merkle root.
/// The root *is* the validator's long-lived public key (published in the [`ValidatorSet`]).
pub struct Signer {
    lamport: LamportSig,
    master: Vec<u8>,
    leaves: Vec<Hash256>,
    root: Hash256,
    /// Next unused OTS slot for `sign_next`. Monotonically increases; never clamped.
    next_slot: usize,
}

impl Clone for Signer {
    fn clone(&self) -> Self {
        // The Lamport scheme is stateless params; clone the (expensive) precomputed leaves.
        Signer {
            lamport: LamportSig::new("lamport-bft", "bft-pq"),
            master: self.master.clone(),
            leaves: self.leaves.clone(),
            root: self.root,
            next_slot: self.next_slot,
        }
    }
}

impl Signer {
    /// Build a signer with `1 << height` one-time slots from `master_seed`.
    pub fn new(master_seed: &[u8], height: u32) -> Signer {
        let lamport = LamportSig::new("lamport-bft", "bft-pq");
        let n = 1usize << height;
        let mut leaves = Vec::with_capacity(n);
        for i in 0..n {
            let (_sk, pk) = lamport.keygen(&Self::ots_seed(master_seed, i));
            leaves.push(Hash256::of(&pk));
        }
        let root = merkle_root(&leaves);
        Signer { lamport, master: master_seed.to_vec(), leaves, root, next_slot: 0 }
    }

    fn ots_seed(master: &[u8], index: usize) -> Vec<u8> {
        let mut s = master.to_vec();
        s.extend_from_slice(b":bft-ots:");
        s.extend_from_slice(&(index as u64).to_le_bytes());
        s
    }

    /// The validator's public key (the Merkle root).
    pub fn public_key(&self) -> Hash256 {
        self.root
    }

    pub fn capacity(&self) -> usize {
        self.leaves.len()
    }

    /// Sign `msg` with the one-time key at `index` (each index at most once).
    pub fn sign(&self, msg: &[u8], index: usize) -> Option<OtsSig> {
        if index >= self.leaves.len() {
            return None;
        }
        let (sk, pk) = self.lamport.keygen(&Self::ots_seed(&self.master, index));
        let sig = self.lamport.sign(&sk, msg);
        Some(OtsSig { index, sig, ots_pub: pk, path: self.auth_path(index) })
    }

    /// Consume the next unused one-time slot and sign `msg`.
    ///
    /// Returns `None` when the key set is exhausted — the validator MUST rotate to a fresh
    /// [`Signer`] (new Merkle key set with a different seed) before signing any further messages.
    /// Signing two different messages with the same Lamport leaf would expose enough preimage
    /// bits for a forger to construct arbitrary signatures under this validator's root.
    pub fn sign_next(&mut self, msg: &[u8]) -> Option<OtsSig> {
        if self.next_slot >= self.leaves.len() {
            // Exhausted: every one-time leaf has been used. Callers MUST construct a new
            // Signer with a fresh master seed. Re-using a Lamport leaf for two distinct
            // messages leaks enough preimage bits for a forger to sign arbitrary messages
            // under this validator's Merkle root.
            //
            // `debug_assert` fires in test/debug builds so callers notice exhaustion
            // immediately rather than silently accumulating None returns.
            debug_assert!(
                false,
                "sign_next called on an exhausted Signer (next_slot={}, capacity={}): \
                 rotate to a new Signer with a fresh seed before signing again",
                self.next_slot,
                self.leaves.len()
            );
            return None;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        let (sk, pk) = self.lamport.keygen(&Self::ots_seed(&self.master, slot));
        let sig = self.lamport.sign(&sk, msg);
        Some(OtsSig { index: slot, sig, ots_pub: pk, path: self.auth_path(slot) })
    }

    fn auth_path(&self, mut index: usize) -> Vec<Hash256> {
        let mut path = Vec::new();
        let mut level = self.leaves.clone();
        while level.len() > 1 {
            let sib = index ^ 1;
            path.push(level.get(sib).copied().unwrap_or(level[index]));
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                if pair.len() == 2 {
                    next.push(merkle_node(&pair[0], &pair[1]));
                } else {
                    next.push(merkle_node(&pair[0], &pair[0]));
                }
            }
            index /= 2;
            level = next;
        }
        path
    }
}

/// Verify `sig` over `msg` against a validator's published `root`.
pub fn verify_sig(root: Hash256, msg: &[u8], sig: &OtsSig) -> bool {
    let lamport = LamportSig::new("lamport-bft", "bft-pq");
    if !lamport.verify(&sig.ots_pub, msg, &sig.sig) {
        return false;
    }
    let mut acc = Hash256::of(&sig.ots_pub);
    let mut index = sig.index;
    for s in &sig.path {
        acc = if index & 1 == 0 { merkle_node(&acc, s) } else { merkle_node(s, &acc) };
        index /= 2;
    }
    acc == root
}

// ───────────────────────────── the protocol ─────────────────────────────

/// The set of validators participating in consensus: their published roots.
/// `n = 3f + 1`, so `f = (n - 1) / 3` and a quorum certificate is `2f + 1`.
#[derive(Clone)]
pub struct ValidatorSet {
    roots: Vec<Hash256>,
}

impl ValidatorSet {
    pub fn new(roots: Vec<Hash256>) -> ValidatorSet {
        ValidatorSet { roots }
    }
    pub fn n(&self) -> usize {
        self.roots.len()
    }
    /// The Byzantine bound: the largest `f` with `3f + 1 <= n`.
    pub fn f(&self) -> usize {
        self.n().saturating_sub(1) / 3
    }
    /// A quorum certificate size: `2f + 1`.
    pub fn quorum(&self) -> usize {
        2 * self.f() + 1
    }
    /// The leader for a view is round-robin over the validators.
    pub fn leader(&self, view: u32) -> u32 {
        (view as usize % self.n().max(1)) as u32
    }
    pub fn root(&self, id: u32) -> Option<Hash256> {
        self.roots.get(id as usize).copied()
    }
}

/// The phase of a consensus vote.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Phase {
    PrePrepare,
    Prepare,
    Commit,
    /// A view-change request (carries the sender's locks in `value`).
    ViewChange,
}

impl Phase {
    fn tag(self) -> u8 {
        match self {
            Phase::PrePrepare => 0,
            Phase::Prepare => 1,
            Phase::Commit => 2,
            Phase::ViewChange => 3,
        }
    }
    fn from_tag(t: u8) -> Option<Phase> {
        match t {
            0 => Some(Phase::PrePrepare),
            1 => Some(Phase::Prepare),
            2 => Some(Phase::Commit),
            3 => Some(Phase::ViewChange),
            _ => None,
        }
    }
}

/// A signed consensus vote. The full `value` rides on every vote so a replica can learn
/// it from any quorum member; `vhash = H(value)` is what tallies match on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vote {
    pub view: u32,
    pub slot: u64,
    pub phase: Phase,
    pub vhash: Hash256,
    pub value: Vec<u8>,
    pub signer: u32,
    pub sig: OtsSig,
}

impl Vote {
    /// The bytes a signature covers — everything that binds the vote (NOT the signer's own
    /// `value` field beyond its hash, so the hash is the authority).
    fn signed_bytes(view: u32, slot: u64, phase: Phase, vhash: &Hash256) -> Vec<u8> {
        let mut b = Vec::with_capacity(4 + 8 + 1 + 32);
        b.extend_from_slice(&view.to_le_bytes());
        b.extend_from_slice(&slot.to_le_bytes());
        b.push(phase.tag());
        b.extend_from_slice(&vhash.0);
        b
    }

    /// Verify the vote's signature against `set`.
    pub fn verify(&self, set: &ValidatorSet) -> bool {
        if self.vhash != Hash256::of(&self.value) {
            return false;
        }
        let root = match set.root(self.signer) {
            Some(r) => r,
            None => return false,
        };
        let msg = Vote::signed_bytes(self.view, self.slot, self.phase, &self.vhash);
        verify_sig(root, &msg, &self.sig)
    }

    /// The key on which equivocation is judged: same signer, view, slot, phase.
    fn equiv_key(&self) -> (u32, u32, u64, u8) {
        (self.signer, self.view, self.slot, self.phase.tag())
    }
}

/// Undeniable proof that a validator signed two different values for one slot/phase/view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EquivocationProof {
    pub a: Vote,
    pub b: Vote,
}

/// Validate an equivocation proof: both votes verify, same signer/view/slot/phase, but
/// different values. Anyone holding the [`ValidatorSet`] can check it — non-repudiable.
pub fn verify_equivocation(set: &ValidatorSet, p: &EquivocationProof) -> bool {
    p.a.equiv_key() == p.b.equiv_key()
        && p.a.vhash != p.b.vhash
        && p.a.verify(set)
        && p.b.verify(set)
}

/// What a validator emits while processing a vote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Out {
    /// A vote to broadcast to all validators.
    Broadcast(Vote),
    /// A slot reached a commit certificate — the agreed, final value.
    Committed { slot: u64, value: Vec<u8> },
    /// Detected equivocation by a validator (advisory → `threat.rs`, evidence for slashing).
    Equivocation(EquivocationProof),
    /// The view advanced (a faulty leader was replaced).
    ViewChanged(u32),
}

/// A single BFT validator. Drives the 3-phase protocol over its own view of the votes.
pub struct Validator {
    pub id: u32,
    set: ValidatorSet,
    signer: Signer,
    view: u32,
    /// Tally of signers per `(view, slot, phase, vhash)`.
    tally: BTreeMap<(u32, u64, u8, Hash256), BTreeSet<u32>>,
    /// First value each signer voted for at `(view, slot, phase)` — equivocation guard.
    seen: BTreeMap<(u32, u32, u64, u8), Vote>,
    /// Slots this validator has already sent a Prepare for, in a given view (vote once).
    prepared_sent: BTreeSet<(u32, u64)>,
    commit_sent: BTreeSet<(u32, u64)>,
    /// Lock: the highest-view value this replica prepared for a slot (carried in view-change).
    locked: BTreeMap<u64, (u32, Vec<u8>)>,
    /// Final committed log: slot → value.
    committed: BTreeMap<u64, Vec<u8>>,
    /// Hash-chained digest of the committed history (determinism boundary).
    log_digest: Hash256,
    /// View-change votes seen, by target view → set of signers.
    vc_tally: BTreeMap<u32, BTreeSet<u32>>,
    vc_sent: BTreeSet<u32>,
}

impl Validator {
    pub fn new(id: u32, master_seed: &[u8], set: ValidatorSet, height: u32) -> Validator {
        Validator::with_signer(id, Signer::new(master_seed, height), set)
    }

    /// Build a validator from an already-constructed [`Signer`] (lets a seed sweep build the
    /// expensive one-time keys once and clone them per run).
    pub fn with_signer(id: u32, signer: Signer, set: ValidatorSet) -> Validator {
        Validator {
            id,
            signer,
            set,
            view: 0,
            tally: BTreeMap::new(),
            seen: BTreeMap::new(),
            prepared_sent: BTreeSet::new(),
            commit_sent: BTreeSet::new(),
            locked: BTreeMap::new(),
            committed: BTreeMap::new(),
            log_digest: Hash256::of(b"bft-log-genesis"),
            vc_tally: BTreeMap::new(),
            vc_sent: BTreeSet::new(),
        }
    }

    pub fn public_key(&self) -> Hash256 {
        self.signer.public_key()
    }
    pub fn view(&self) -> u32 {
        self.view
    }
    pub fn committed_value(&self, slot: u64) -> Option<&[u8]> {
        self.committed.get(&slot).map(|v| v.as_slice())
    }
    pub fn committed_count(&self) -> usize {
        self.committed.len()
    }
    pub fn log_digest(&self) -> Hash256 {
        self.log_digest
    }
    pub fn is_leader(&self) -> bool {
        self.set.leader(self.view) == self.id
    }

    /// Produce a signed vote. Returns `None` when the one-time key set is exhausted.
    /// The validator MUST rotate its Merkle key set (new [`Signer`] with a fresh seed) before
    /// calling this again — re-using a Lamport leaf for two different messages leaks enough
    /// preimage bits for a forger to forge arbitrary signatures under this validator's root.
    fn make_vote(&mut self, slot: u64, phase: Phase, value: &[u8]) -> Option<Vote> {
        let vhash = Hash256::of(value);
        let msg = Vote::signed_bytes(self.view, slot, phase, &vhash);
        let sig = self.signer.sign_next(&msg)?;
        Some(Vote { view: self.view, slot, phase, vhash, value: value.to_vec(), signer: self.id, sig })
    }

    /// Leader action: propose `value` for `slot` in the current view.
    pub fn propose(&mut self, slot: u64, value: &[u8]) -> Option<Out> {
        if !self.is_leader() {
            return None;
        }
        // If this slot is locked at some prior view, the leader must re-propose the locked
        // value (cross-view safety) — never a fresh one.
        let value: Vec<u8> = match self.locked.get(&slot) {
            Some((_, v)) => v.clone(),
            None => value.to_vec(),
        };
        Some(Out::Broadcast(self.make_vote(slot, Phase::PrePrepare, &value)?))
    }

    /// Ingest a vote from the network and drive the protocol. Returns any outbound effects.
    pub fn ingest(&mut self, vote: &Vote) -> Vec<Out> {
        let mut out = Vec::new();
        if !vote.verify(&self.set) {
            return out; // forged / malformed: ignore
        }
        // Equivocation: same signer voting a different value at the same (view, slot, phase).
        let ek = (vote.signer, vote.view, vote.slot, vote.phase.tag());
        if let Some(prev) = self.seen.get(&ek) {
            if prev.vhash != vote.vhash {
                out.push(Out::Equivocation(EquivocationProof { a: prev.clone(), b: vote.clone() }));
                return out; // refuse to count an equivocator's conflicting vote
            }
        } else {
            self.seen.insert(ek, vote.clone());
        }

        match vote.phase {
            Phase::ViewChange => self.ingest_view_change(vote, &mut out),
            Phase::PrePrepare => self.ingest_preprepare(vote, &mut out),
            Phase::Prepare => self.ingest_prepare(vote, &mut out),
            Phase::Commit => self.ingest_commit(vote, &mut out),
        }
        out
    }

    fn ingest_preprepare(&mut self, vote: &Vote, out: &mut Vec<Out>) {
        // Only honour a PrePrepare from the current view's leader.
        if vote.view != self.view || vote.signer != self.set.leader(self.view) {
            return;
        }
        // If we are locked on a different value for this slot, refuse (safety).
        if let Some((_, locked)) = self.locked.get(&vote.slot) {
            if Hash256::of(locked) != vote.vhash {
                return;
            }
        }
        if self.prepared_sent.insert((self.view, vote.slot)) {
            if let Some(v) = self.make_vote(vote.slot, Phase::Prepare, &vote.value) {
                out.push(Out::Broadcast(v));
            }
        }
    }

    fn ingest_prepare(&mut self, vote: &Vote, out: &mut Vec<Out>) {
        if vote.view != self.view {
            return;
        }
        let key = (vote.view, vote.slot, Phase::Prepare.tag(), vote.vhash);
        let count = {
            let set = self.tally.entry(key).or_default();
            set.insert(vote.signer);
            set.len()
        };
        if count >= self.set.quorum() && self.commit_sent.insert((self.view, vote.slot)) {
            // Prepared certificate: lock the value at this view, then send Commit.
            self.locked.insert(vote.slot, (self.view, vote.value.clone()));
            if let Some(v) = self.make_vote(vote.slot, Phase::Commit, &vote.value) {
                out.push(Out::Broadcast(v));
            }
        }
    }

    fn ingest_commit(&mut self, vote: &Vote, out: &mut Vec<Out>) {
        let key = (vote.view, vote.slot, Phase::Commit.tag(), vote.vhash);
        let count = {
            let set = self.tally.entry(key).or_default();
            set.insert(vote.signer);
            set.len()
        };
        if count >= self.set.quorum() && !self.committed.contains_key(&vote.slot) {
            self.committed.insert(vote.slot, vote.value.clone());
            // Fold into the hash-chained determinism log.
            let mut chain = Vec::with_capacity(32 + 8 + 32);
            chain.extend_from_slice(&self.log_digest.0);
            chain.extend_from_slice(&vote.slot.to_le_bytes());
            chain.extend_from_slice(&vote.vhash.0);
            self.log_digest = Hash256::of(&chain);
            out.push(Out::Committed { slot: vote.slot, value: vote.value.clone() });
        }
    }

    // ───────── view-change (liveness when a leader is faulty) ─────────

    /// Suspect the current leader and request a move to the next view. Carries this
    /// replica's locks so the next leader can re-propose anything possibly committed.
    pub fn start_view_change(&mut self) -> Option<Out> {
        let target = self.view + 1;
        if !self.vc_sent.insert(target) {
            return None;
        }
        let payload = self.encode_locks();
        // A view-change vote is signed over (target_view, slot=MAX sentinel, ViewChange, H(locks)).
        let vhash = Hash256::of(&payload);
        let msg = Vote::signed_bytes(target, u64::MAX, Phase::ViewChange, &vhash);
        // Returns None when the one-time key set is exhausted — rotate the Merkle key set.
        let sig = self.signer.sign_next(&msg)?;
        Some(Out::Broadcast(Vote {
            view: target,
            slot: u64::MAX,
            phase: Phase::ViewChange,
            vhash,
            value: payload,
            signer: self.id,
            sig,
        }))
    }

    fn ingest_view_change(&mut self, vote: &Vote, out: &mut Vec<Out>) {
        let target = vote.view;
        if target <= self.view {
            return;
        }
        // Merge the sender's locks so a new leader has the full locked set.
        for (slot, view, value) in decode_locks(&vote.value) {
            let take = self.locked.get(&slot).map(|(v, _)| view > *v).unwrap_or(true);
            if take {
                self.locked.insert(slot, (view, value));
            }
        }
        let count = {
            let set = self.vc_tally.entry(target).or_default();
            set.insert(vote.signer);
            set.len()
        };
        if count >= self.set.quorum() && target > self.view {
            self.view = target;
            // New view: fresh per-view vote bookkeeping (locks persist).
            self.prepared_sent.retain(|(v, _)| *v >= self.view);
            self.commit_sent.retain(|(v, _)| *v >= self.view);
            out.push(Out::ViewChanged(target));
        }
    }

    fn encode_locks(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(self.locked.len() as u32).to_le_bytes());
        for (slot, (view, value)) in &self.locked {
            b.extend_from_slice(&slot.to_le_bytes());
            b.extend_from_slice(&view.to_le_bytes());
            b.extend_from_slice(&(value.len() as u32).to_le_bytes());
            b.extend_from_slice(value);
        }
        b
    }
}

fn decode_locks(b: &[u8]) -> Vec<(u64, u32, Vec<u8>)> {
    let mut out = Vec::new();
    if b.len() < 4 {
        return out;
    }
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let mut p = 4;
    for _ in 0..n {
        if p + 16 > b.len() {
            break;
        }
        let slot = u64::from_le_bytes(b[p..p + 8].try_into().unwrap());
        let view = u32::from_le_bytes(b[p + 8..p + 12].try_into().unwrap());
        let vlen = u32::from_le_bytes(b[p + 12..p + 16].try_into().unwrap()) as usize;
        p += 16;
        if p + vlen > b.len() {
            break;
        }
        out.push((slot, view, b[p..p + vlen].to_vec()));
        p += vlen;
    }
    out
}

// ───────────────────────────── wire encoding (for the DST network) ─────────────────────────────

/// Encode a vote for transmission over the simulated/real network.
pub fn encode_vote(v: &Vote) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&v.view.to_le_bytes());
    b.extend_from_slice(&v.slot.to_le_bytes());
    b.push(v.phase.tag());
    b.extend_from_slice(&v.vhash.0);
    b.extend_from_slice(&(v.value.len() as u32).to_le_bytes());
    b.extend_from_slice(&v.value);
    b.extend_from_slice(&v.signer.to_le_bytes());
    b.extend_from_slice(&(v.sig.index as u32).to_le_bytes());
    put_bytes(&mut b, &v.sig.sig);
    put_bytes(&mut b, &v.sig.ots_pub);
    b.extend_from_slice(&(v.sig.path.len() as u32).to_le_bytes());
    for h in &v.sig.path {
        b.extend_from_slice(&h.0);
    }
    b
}

/// Decode a vote produced by [`encode_vote`].
pub fn decode_vote(b: &[u8]) -> Option<Vote> {
    let mut p = 0usize;
    let view = u32::from_le_bytes(take(b, &mut p, 4)?.try_into().ok()?);
    let slot = u64::from_le_bytes(take(b, &mut p, 8)?.try_into().ok()?);
    let phase = Phase::from_tag(*take(b, &mut p, 1)?.first()?)?;
    let mut vh = [0u8; 32];
    vh.copy_from_slice(take(b, &mut p, 32)?);
    let vlen = u32::from_le_bytes(take(b, &mut p, 4)?.try_into().ok()?) as usize;
    let value = take(b, &mut p, vlen)?.to_vec();
    let signer = u32::from_le_bytes(take(b, &mut p, 4)?.try_into().ok()?);
    let index = u32::from_le_bytes(take(b, &mut p, 4)?.try_into().ok()?) as usize;
    let sig = get_bytes(b, &mut p)?;
    let ots_pub = get_bytes(b, &mut p)?;
    let plen = u32::from_le_bytes(take(b, &mut p, 4)?.try_into().ok()?) as usize;
    let mut path = Vec::with_capacity(plen);
    for _ in 0..plen {
        let mut h = [0u8; 32];
        h.copy_from_slice(take(b, &mut p, 32)?);
        path.push(Hash256(h));
    }
    Some(Vote { view, slot, phase, vhash: Hash256(vh), value, signer, sig: OtsSig { index, sig, ots_pub, path } })
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}
fn get_bytes(b: &[u8], p: &mut usize) -> Option<Vec<u8>> {
    let n = u32::from_le_bytes(take(b, p, 4)?.try_into().ok()?) as usize;
    Some(take(b, p, n)?.to_vec())
}
fn take<'a>(b: &'a [u8], p: &mut usize, n: usize) -> Option<&'a [u8]> {
    if *p + n > b.len() {
        return None;
    }
    let s = &b[*p..*p + n];
    *p += n;
    Some(s)
}

// ───────────────────────────── DST scenarios ─────────────────────────────

/// How a faulty validator misbehaves in a scenario.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    Honest,
    /// Sends nothing and ignores inbound (crash / silent leader).
    Silent,
}

/// Build `n` deterministic validator signers (the expensive one-time keys) plus the set.
/// A seed sweep builds these once and clones them per run.
pub fn build_signers(n: u32, height: u32) -> (Vec<Signer>, ValidatorSet) {
    let signers: Vec<Signer> = (0..n)
        .map(|i| {
            let mut s = alloc::vec![b'v'];
            s.extend_from_slice(&i.to_le_bytes());
            Signer::new(&s, height)
        })
        .collect();
    let set = ValidatorSet::new(signers.iter().map(|s| s.public_key()).collect());
    (signers, set)
}

/// Fresh validators for one run, cloning the pre-built signers (no keygen cost).
fn validators_from(signers: &[Signer], set: &ValidatorSet) -> Vec<Validator> {
    signers
        .iter()
        .enumerate()
        .map(|(i, s)| Validator::with_signer(i as u32, s.clone(), set.clone()))
        .collect()
}

/// **Safety + liveness under a Byzantine (silent) fault.** With `byz >= 1`, the leader of
/// view 0 (validator 0) is silent — sending nothing, ignoring inbound — which forces the
/// honest majority to view-change to a live leader. Honest replicas exchange votes over a
/// reordering/delaying [`crate::dst::SimNetwork`]; every vote a validator emits is retained
/// and periodically re-broadcast (anti-entropy), so the protocol heals delays deterministically.
/// Returns `(agreed, committed_slots)` where `agreed` = all honest replicas committed
/// identical values for every slot they decided (the core safety property).
pub fn bft_scenario(seed: u64, n: u32, byz: u32, slots: u64, rounds: u32) -> (bool, u64) {
    let (signers, set) = build_signers(n, 7);
    bft_run(seed, &signers, &set, byz, slots, rounds)
}

/// Run one BFT scenario over pre-built `signers` (so a sweep pays keygen once). See
/// [`bft_scenario`] for the semantics.
pub fn bft_run(seed: u64, signers: &[Signer], set: &ValidatorSet, byz: u32, slots: u64, rounds: u32) -> (bool, u64) {
    use crate::dst::{Sim, SimNetwork};

    let n = signers.len() as u32;
    let mut vals = validators_from(signers, set);
    let f = set.f();
    let byz = byz.min(f as u32);

    let mut fault = alloc::vec![Fault::Honest; n as usize];
    if byz >= 1 {
        fault[0] = Fault::Silent; // the view-0 leader crashes → view-change must rescue liveness
    }

    let mut sim = Sim::new(seed);
    // Lossless but reordering + up to 5 ticks delay — the Byzantine fault + view-change is the
    // challenge, not packet loss (loss/partition is covered by `consistency::FencedReplica`).
    let mut net = SimNetwork::new(0, 5);
    let mut proposed: BTreeSet<(u32, u64)> = BTreeSet::new();
    // Votes emitted this round, sent once each (the network never drops, only delays/reorders).
    let mut outbox: Vec<Vote> = Vec::new();

    for round in 0..rounds {
        // 1. Deliver everything due, ingest, fan out follow-on votes.
        for (to, payload) in net.deliver_due(&mut sim) {
            if fault[to as usize] == Fault::Silent {
                continue;
            }
            if let Some(vote) = decode_vote(&payload) {
                for o in vals[to as usize].ingest(&vote) {
                    if let Out::Broadcast(b) = o {
                        outbox.push(b);
                    }
                }
            }
        }

        // 2. Honest leaders propose any slot they haven't yet, in their current view.
        for i in 0..n {
            if fault[i as usize] == Fault::Silent || !vals[i as usize].is_leader() {
                continue;
            }
            for slot in 0..slots {
                if vals[i as usize].committed_value(slot).is_none()
                    && proposed.insert((vals[i as usize].view(), slot))
                {
                    let value = proposal_value(i, vals[i as usize].view(), slot);
                    if let Some(Out::Broadcast(b)) = vals[i as usize].propose(slot, &value) {
                        outbox.push(b);
                    }
                }
            }
        }

        // 3. If still not fully decided, honest replicas periodically request a view-change.
        let stuck = (0..n as usize)
            .filter(|i| fault[*i] != Fault::Silent)
            .any(|i| (vals[i].committed_count() as u64) < slots);
        if stuck && round % 4 == 3 {
            for i in 0..n {
                if fault[i as usize] == Fault::Silent {
                    continue;
                }
                if let Some(Out::Broadcast(b)) = vals[i as usize].start_view_change() {
                    outbox.push(b);
                }
            }
        }

        // 4. Transport: send each freshly-emitted vote once to every validator.
        for vote in outbox.drain(..) {
            let payload = encode_vote(&vote);
            for to in 0..n {
                net.send(&mut sim, to, &payload);
            }
        }
        sim.advance(2);
    }

    // Safety: across honest replicas, every decided slot must agree on one value.
    let honest: Vec<&Validator> = vals.iter().filter(|v| fault[v.id as usize] != Fault::Silent).collect();
    let mut agreed = true;
    for slot in 0..slots {
        let mut decided: Option<Hash256> = None;
        for v in &honest {
            if let Some(val) = v.committed_value(slot) {
                let h = Hash256::of(val);
                match decided {
                    None => decided = Some(h),
                    Some(d) if d != h => agreed = false,
                    _ => {}
                }
            }
        }
    }
    let min_committed = honest.iter().map(|v| v.committed_count() as u64).min().unwrap_or(0);
    (agreed, min_committed)
}

fn proposal_value(leader: u32, view: u32, slot: u64) -> Vec<u8> {
    let mut v = alloc::vec![b'V'];
    v.extend_from_slice(&leader.to_le_bytes());
    v.extend_from_slice(&view.to_le_bytes());
    v.extend_from_slice(&slot.to_le_bytes());
    v
}

impl Validator {
    /// Test/Byzantine helper: deliberately sign a *second*, conflicting value for a
    /// `(view, slot, phase)` this validator already voted on — i.e. equivocate.
    pub fn equivocate(&mut self, slot: u64, phase: Phase, value: &[u8]) -> Vote {
        self.make_vote(slot, phase, value).expect("key set exhausted in equivocate test helper")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signer_round_trips_many_messages_under_one_root() {
        let s = Signer::new(b"validator-seed", 4); // 16 one-time keys
        let root = s.public_key();
        for i in 0..16 {
            let msg = alloc::format!("vote-{i}");
            let sig = s.sign(msg.as_bytes(), i).unwrap();
            assert!(verify_sig(root, msg.as_bytes(), &sig), "slot {i}");
        }
    }

    #[test]
    fn tampered_vote_or_wrong_root_is_rejected() {
        let s = Signer::new(b"seed", 3);
        let sig = s.sign(b"hello", 0).unwrap();
        assert!(verify_sig(s.public_key(), b"hello", &sig));
        assert!(!verify_sig(s.public_key(), b"goodbye", &sig));
        let other = Signer::new(b"other", 3);
        assert!(!verify_sig(other.public_key(), b"hello", &sig));
    }

    #[test]
    fn validator_set_bounds_are_optimal() {
        let s = ValidatorSet::new(alloc::vec![Hash256::ZERO; 4]);
        assert_eq!(s.f(), 1);
        assert_eq!(s.quorum(), 3);
        let s7 = ValidatorSet::new(alloc::vec![Hash256::ZERO; 7]);
        assert_eq!(s7.f(), 2);
        assert_eq!(s7.quorum(), 5);
    }

    #[test]
    fn happy_path_commits_one_value_across_all_replicas() {
        // 4 validators, all honest, no faults: a proposal commits everywhere with the same value.
        let (signers, set) = build_signers(4, 5);
        let mut vals = validators_from(&signers, &set);
        // Leader of view 0 = validator 0 proposes slot 0.
        let value = b"agreed-state";
        let mut bus: Vec<Vote> = Vec::new();
        if let Some(Out::Broadcast(v)) = vals[0].propose(0, value) {
            bus.push(v);
        }
        // Run a synchronous broadcast loop to a fixed point.
        let mut committed = alloc::vec![None; 4];
        for _ in 0..12 {
            let mut next = Vec::new();
            for vote in &bus {
                for v in vals.iter_mut() {
                    for o in v.ingest(vote) {
                        match o {
                            Out::Broadcast(b) => next.push(b),
                            Out::Committed { slot, value } => {
                                assert_eq!(slot, 0);
                                committed[v.id as usize] = Some(value);
                            }
                            _ => {}
                        }
                    }
                }
            }
            bus = next;
            if committed.iter().all(|c| c.is_some()) {
                break;
            }
        }
        assert!(committed.iter().all(|c| c.as_deref() == Some(value.as_ref())));
        // All logs agree (determinism boundary).
        let d0 = vals[0].log_digest();
        assert!(vals.iter().all(|v| v.log_digest() == d0));
    }

    #[test]
    fn equivocation_is_detected_and_proven() {
        let (signers, set) = build_signers(4, 5);
        let mut vals = validators_from(&signers, &set);
        // Validator 0 (leader) signs TWO different values for (view 0, slot 0, PrePrepare).
        let v_a = vals[0].equivocate(0, Phase::PrePrepare, b"value-A");
        let v_b = vals[0].equivocate(0, Phase::PrePrepare, b"value-B");
        // An honest replica ingesting both raises an equivocation proof.
        let _ = vals[1].ingest(&v_a);
        let out = vals[1].ingest(&v_b);
        let proof = out.iter().find_map(|o| match o {
            Out::Equivocation(p) => Some(p.clone()),
            _ => None,
        });
        let proof = proof.expect("equivocation must be caught");
        // The proof is independently verifiable by anyone holding the validator set.
        assert!(verify_equivocation(&set, &proof));
    }

    #[test]
    fn safety_holds_under_byzantine_faults_across_a_seed_sweep() {
        // 4 validators, f=1 faulty (silent leader of view 0). Honest replicas must agree on
        // every committed slot, never diverge, across a seed sweep with loss/reorder.
        let (signers, set) = build_signers(4, 7);
        for seed in 0..16u64 {
            let (agreed, _committed) = bft_run(seed, &signers, &set, 1, 2, 40);
            assert!(agreed, "honest replicas diverged at seed {seed}");
        }
    }

    #[test]
    fn liveness_recovers_when_the_view0_leader_is_silent() {
        // With a silent view-0 leader, a view-change must let a later honest leader commit.
        let (_agreed, committed) = bft_scenario(7, 4, 1, 1, 80);
        assert!(committed >= 1, "no slot committed despite a view-change path");
    }

    #[test]
    fn scenario_is_a_pure_function_of_seed() {
        assert_eq!(bft_scenario(3, 4, 1, 2, 30), bft_scenario(3, 4, 1, 2, 30));
    }

    #[test]
    fn make_vote_returns_none_when_ots_key_set_exhausted() {
        // height=1 → capacity=2 one-time keys. The first two votes consume both slots;
        // a third must return None rather than clamping to a used leaf (which would allow
        // Lamport preimage recovery and forgery under this validator's root).
        let (signers, set) = build_signers(1, 1); // 1 validator, 2 OTS slots
        let mut val = validators_from(&signers, &set).remove(0);
        let v0 = val.make_vote(0, Phase::Prepare, b"msg-a");
        assert!(v0.is_some(), "first vote must succeed");
        let v1 = val.make_vote(1, Phase::Prepare, b"msg-b");
        assert!(v1.is_some(), "second vote must succeed");
        let v2 = val.make_vote(2, Phase::Prepare, b"msg-c");
        assert!(v2.is_none(), "third vote must return None — key set exhausted, must not reuse a leaf");
    }

    #[test]
    fn vote_wire_round_trips() {
        let s = Signer::new(b"seed", 3);
        let vhash = Hash256::of(b"val");
        let msg = Vote::signed_bytes(2, 5, Phase::Commit, &vhash);
        let sig = s.sign(&msg, 1).unwrap();
        let v = Vote { view: 2, slot: 5, phase: Phase::Commit, vhash, value: b"val".to_vec(), signer: 0, sig };
        let enc = encode_vote(&v);
        let dec = decode_vote(&enc).unwrap();
        assert_eq!(v, dec);
    }
}

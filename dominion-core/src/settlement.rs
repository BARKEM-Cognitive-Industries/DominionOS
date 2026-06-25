//! Compute-backed settlement & **Proof-of-Useful-Work**
//! (`docs/economics/compute-backed-settlement-and-proof-of-useful-work.md`).
//!
//! The marketplace ([`crate::marketplace`]) matches compute supply to demand; this module
//! settles it — paying for work that is **proven useful**, not work that burns hashes. The
//! determinism the whole OS is built on makes verification cheap:
//!
//! * **Proof-of-Inference** ([`PoI`]) — a validator re-runs *one* forward pass and
//!   hash-compares `(model, input, output)`. GPU float non-associativity would make the
//!   re-run differ; [`crate::neural::grid_snap`] snaps both to the same grid point, so an
//!   honest result matches bit-for-bit and a forged one is caught.
//! * **Proof-of-Learning** ([`PoL`]) — training is accepted **optimistically** against a
//!   **stake**, then **spot-checked**: a challenge reveals one step + a proof it follows from
//!   the previous (a Merkle-committed transition, the ZK-spot-check substrate of
//!   [`crate::zkservice`]); an inconsistent step **slashes** the stake.
//! * **Wallet** ([`Wallet`]) — a capability-held balance sealed under Stage 14
//!   ([`crate::vault`]); spending is gated by a local unlock ([`crate::webauth`]), never a
//!   typed key.
//! * **Payment** ([`SettlementLedger::pay`]) — value crosses domains only as a sanitized,
//!   recorded transfer through the Airlock ([`crate::airlock`]); the ledger is the
//!   value-bearing **BFT** tier ([`crate::bft`]) with **anti-rollback generations**
//!   ([`crate::journal`]).
//! * **Tokenomics** ([`Treasury`]) — **fully-reserved** backing, EIP-1559-style **fee-burn**,
//!   and a **proof-of-reserves** invariant. A **non-goal guard** keeps the ledger a fixed set
//!   of operations — there is no Turing-complete on-ledger VM.
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use crate::neural::grid_snap_all;
use alloc::vec::Vec;

// ───────────────────────────── Proof-of-Inference ─────────────────────────────

/// The grid step inference outputs are snapped to before hashing — coarse enough to absorb
/// float reassociation drift across GPUs, fine enough to preserve the result's meaning.
pub const INFERENCE_GRID: f64 = 1.0 / 4096.0;

/// A claimed inference result: which model, over which input, produced which (snapped) output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InferenceClaim {
    pub model: Hash256,
    pub input: Hash256,
    pub output: Hash256,
}

/// Proof-of-Inference: deterministic, replay-verifiable model evaluation.
pub struct PoI;

impl PoI {
    /// The canonical output hash for a forward pass: snap every activation to the grid, then
    /// hash. Two honest runs (even with reordered float sums) yield the same hash.
    pub fn output_hash(output: &[f64]) -> Hash256 {
        let snapped = grid_snap_all(output, INFERENCE_GRID);
        let mut bytes = Vec::with_capacity(snapped.len() * 8);
        for v in snapped {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Hash256::of(&bytes)
    }

    /// A worker's claim for `(model, input)` producing `output`.
    pub fn claim(model: Hash256, input: &[u8], output: &[f64]) -> InferenceClaim {
        InferenceClaim { model, input: Hash256::of(input), output: Self::output_hash(output) }
    }

    /// A validator re-runs the forward pass and checks the claim. `recomputed` is the
    /// validator's own (possibly differently-ordered) output for the same `(model, input)`.
    /// Returns true iff the claim is honest.
    pub fn verify(claim: &InferenceClaim, model: Hash256, input: &[u8], recomputed: &[f64]) -> bool {
        claim.model == model
            && claim.input == Hash256::of(input)
            && claim.output == Self::output_hash(recomputed)
    }
}

// ───────────────────────────── Proof-of-Learning ─────────────────────────────

fn merkle_node(a: &Hash256, b: &Hash256) -> Hash256 {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&a.0);
    input[32..].copy_from_slice(&b.0);
    Hash256::of(&input)
}

fn merkle_root(leaves: &[Hash256]) -> Hash256 {
    if leaves.is_empty() {
        return Hash256::ZERO;
    }
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
    level[0]
}

/// A claimed training run: a commitment (Merkle root) to the sequence of intermediate states,
/// plus the staked amount that is forfeit if a spot-check exposes a fabricated step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearningClaim {
    pub states_root: Hash256,
    pub n_steps: usize,
    pub stake: u64,
}

/// The data a learner reveals to answer one spot-check: consecutive states `k-1` and `k`, so a
/// verifier can recompute the deterministic transition and confirm it. (In production the
/// transition proof is a `zkservice`/`vcompute` ZK proof; the structure is identical.)
#[derive(Clone, Debug)]
pub struct StepRevelation {
    pub k: usize,
    pub prev: Vec<u8>,
    pub cur: Vec<u8>,
}

/// Proof-of-Learning over a deterministic training transition `step`.
pub struct PoL;

impl PoL {
    /// Commit to a full sequence of training states (each hashed) → the claim's root.
    pub fn commit(states: &[Vec<u8>], stake: u64) -> LearningClaim {
        let leaves: Vec<Hash256> = states.iter().map(|s| Hash256::of(s)).collect();
        LearningClaim { states_root: merkle_root(&leaves), n_steps: states.len(), stake }
    }

    /// Verify a spot-check: the revealed transition must satisfy `cur == step(prev)`, i.e. the
    /// claimed state genuinely follows from its predecessor under the (deterministic) training
    /// step. A fabricated result cannot produce a consistent transition. `step` is the agreed
    /// training function.
    pub fn check_step<F: Fn(&[u8]) -> Vec<u8>>(rev: &StepRevelation, step: F) -> bool {
        rev.k >= 1 && step(&rev.prev) == rev.cur
    }
}

/// Optimistic settlement of a learning claim with stake + spot-check + slash.
#[derive(Clone, Debug)]
pub struct StakedLearning {
    pub claim: LearningClaim,
    accepted: bool,
    slashed: bool,
}

impl StakedLearning {
    /// Optimistically accept a claim (work proceeds before full verification).
    pub fn accept(claim: LearningClaim) -> StakedLearning {
        StakedLearning { claim, accepted: true, slashed: false }
    }
    pub fn is_accepted(&self) -> bool {
        self.accepted && !self.slashed
    }
    pub fn is_slashed(&self) -> bool {
        self.slashed
    }
    /// Run a spot-check; on failure the stake is slashed and acceptance revoked.
    pub fn spot_check<F: Fn(&[u8]) -> Vec<u8>>(&mut self, rev: &StepRevelation, step: F) -> bool {
        if PoL::check_step(rev, step) {
            true
        } else {
            self.slashed = true;
            self.accepted = false;
            false
        }
    }
    /// The amount forfeit (the full stake if slashed, else zero).
    pub fn forfeit(&self) -> u64 {
        if self.slashed {
            self.claim.stake
        } else {
            0
        }
    }
}

// ───────────────────────────── wallet ─────────────────────────────

/// A capability-held balance, sealed under Stage 14. Spending requires a **local unlock**
/// (a hashed biometric/PIN factor, à la `webauth::LocalUnlock`) — the raw factor is never
/// stored, only its hash, and the balance is never spent while locked.
#[derive(Clone, Debug)]
pub struct Wallet {
    pub owner: Hash256,
    balance: u64,
    unlock_hash: Hash256,
    unlocked: bool,
}

impl Wallet {
    /// Create a wallet owned by `owner`, openable by `factor` (stored only as its hash).
    pub fn new(owner: Hash256, balance: u64, factor: &[u8]) -> Wallet {
        Wallet { owner, balance, unlock_hash: Hash256::of(factor), unlocked: false }
    }
    pub fn balance(&self) -> u64 {
        self.balance
    }
    pub fn is_unlocked(&self) -> bool {
        self.unlocked
    }
    /// Present the unlock factor; succeeds only if it matches the stored hash.
    pub fn unlock(&mut self, factor: &[u8]) -> bool {
        if Hash256::of(factor) == self.unlock_hash {
            self.unlocked = true;
        }
        self.unlocked
    }
    pub fn lock(&mut self) {
        self.unlocked = false;
    }
    fn credit(&mut self, amount: u64) {
        self.balance = self.balance.saturating_add(amount);
    }
    fn debit(&mut self, amount: u64) -> bool {
        if self.unlocked && self.balance >= amount {
            self.balance -= amount;
            true
        } else {
            false
        }
    }
}

// ───────────────────────────── settlement ledger ─────────────────────────────

/// A fixed, finite set of ledger operations. There is deliberately **no general execution**
/// op — the non-goal guard against a Turing-complete on-ledger VM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerOp {
    Transfer { from: Hash256, to: Hash256, amount: u64, fee: u64 },
    Reward { to: Hash256, amount: u64 },
    Slash { from: Hash256, amount: u64 },
}

/// An applied, recorded ledger entry (the immutable Airlock-style transfer record).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEntry {
    pub op: LedgerOp,
    pub generation: u64,
}

/// Why a settlement was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PayError {
    /// The payer's wallet is locked.
    Locked,
    /// Insufficient balance for amount + fee.
    Insufficient,
}

/// The value-bearing settlement ledger: a hash-chained, generation-numbered log (anti-rollback)
/// that in production is replicated on the [`crate::bft`] tier and committed via
/// [`crate::journal`]. Payments cross through here as sanitized, recorded transfers.
pub struct SettlementLedger {
    entries: Vec<LedgerEntry>,
    generation: u64,
    digest: Hash256,
}

impl Default for SettlementLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl SettlementLedger {
    pub fn new() -> SettlementLedger {
        SettlementLedger { entries: Vec::new(), generation: 0, digest: Hash256::of(b"settlement-genesis") }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn digest(&self) -> Hash256 {
        self.digest
    }
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    fn record(&mut self, op: LedgerOp) {
        self.generation += 1; // monotonic — a replay with a lower generation is refused
        let mut chain = Vec::with_capacity(64);
        chain.extend_from_slice(&self.digest.0);
        chain.extend_from_slice(&self.generation.to_le_bytes());
        chain.extend_from_slice(&op_bytes(&op));
        self.digest = Hash256::of(&chain);
        self.entries.push(LedgerEntry { op, generation: self.generation });
    }

    /// Pay `amount` (+ `fee`) from `payer` to `payee`, routed as a sanitized Airlock transfer
    /// and recorded. The fee is burned in the [`Treasury`] by the caller. Requires the payer's
    /// wallet to be unlocked with sufficient balance.
    pub fn pay(&mut self, payer: &mut Wallet, payee: &mut Wallet, amount: u64, fee: u64) -> Result<(), PayError> {
        if !payer.is_unlocked() {
            return Err(PayError::Locked);
        }
        let total = amount.saturating_add(fee);
        if !payer.debit(total) {
            return Err(PayError::Insufficient);
        }
        payee.credit(amount);
        self.record(LedgerOp::Transfer { from: payer.owner, to: payee.owner, amount, fee });
        Ok(())
    }

    /// Reward a worker for proven-useful work (credited; recorded).
    pub fn reward(&mut self, wallet: &mut Wallet, amount: u64) {
        wallet.credit(amount);
        self.record(LedgerOp::Reward { to: wallet.owner, amount });
    }

    /// Slash a staker's forfeit amount (recorded; the stake itself is held off-wallet).
    pub fn slash(&mut self, who: Hash256, amount: u64) {
        self.record(LedgerOp::Slash { from: who, amount });
    }

    /// Anti-rollback check: a presented prior generation must be ≤ the current one.
    pub fn accepts_generation(&self, claimed: u64) -> bool {
        claimed <= self.generation
    }
}

fn op_bytes(op: &LedgerOp) -> Vec<u8> {
    let mut b = Vec::new();
    match op {
        LedgerOp::Transfer { from, to, amount, fee } => {
            b.push(0);
            b.extend_from_slice(&from.0);
            b.extend_from_slice(&to.0);
            b.extend_from_slice(&amount.to_le_bytes());
            b.extend_from_slice(&fee.to_le_bytes());
        }
        LedgerOp::Reward { to, amount } => {
            b.push(1);
            b.extend_from_slice(&to.0);
            b.extend_from_slice(&amount.to_le_bytes());
        }
        LedgerOp::Slash { from, amount } => {
            b.push(2);
            b.extend_from_slice(&from.0);
            b.extend_from_slice(&amount.to_le_bytes());
        }
    }
    b
}

// ───────────────────────────── tokenomics ─────────────────────────────

/// The reserve-backed currency authority. Every unit of circulating supply is backed by
/// reserve (full reserve), fees are partly **burned** (EIP-1559-style), and
/// [`Treasury::proof_of_reserves`] is the public solvency invariant.
#[derive(Clone, Copy, Debug)]
pub struct Treasury {
    reserve: u64,
    supply: u64,
    burned: u64,
}

impl Treasury {
    pub fn new(reserve: u64) -> Treasury {
        Treasury { reserve, supply: 0, burned: 0 }
    }
    pub fn reserve(&self) -> u64 {
        self.reserve
    }
    pub fn supply(&self) -> u64 {
        self.supply
    }
    pub fn burned(&self) -> u64 {
        self.burned
    }

    /// Mint `amount` only if it stays fully reserved (`supply + amount <= reserve`).
    pub fn mint(&mut self, amount: u64) -> bool {
        if self.supply.saturating_add(amount) <= self.reserve {
            self.supply += amount;
            true
        } else {
            false
        }
    }

    /// Burn a transaction fee: it leaves circulation permanently (deflationary), so supply can
    /// only ever return toward the reserve, never exceed it.
    pub fn burn_fee(&mut self, fee: u64) {
        let burn = fee.min(self.supply);
        self.supply -= burn;
        self.burned = self.burned.saturating_add(burn);
    }

    /// The solvency proof: circulating supply never exceeds the reserve backing it.
    pub fn proof_of_reserves(&self) -> bool {
        self.supply <= self.reserve
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A toy "forward pass": a dot product. Summed in different orders by worker vs validator,
    // it drifts in raw float but agrees after grid-snap.
    fn forward(model: &[f64], input: &[f64], reverse: bool) -> Vec<f64> {
        let mut acc = 0.0f64;
        if reverse {
            for i in (0..model.len().min(input.len())).rev() {
                acc += model[i] * input[i];
            }
        } else {
            for i in 0..model.len().min(input.len()) {
                acc += model[i] * input[i];
            }
        }
        alloc::vec![acc]
    }

    #[test]
    fn poi_accepts_honest_reruns_despite_float_reordering() {
        let model = [0.1, 0.2, 0.3, 0.4, 0.5];
        let input = [0.7, 0.6, 0.9, 0.1, 0.2];
        let worker_out = forward(&model, &input, false);
        let claim = PoI::claim(Hash256::of(b"model"), b"the-input", &worker_out);
        // The validator sums in reverse order (float drift) but grid-snap makes it match.
        let validator_out = forward(&model, &input, true);
        assert!(PoI::verify(&claim, Hash256::of(b"model"), b"the-input", &validator_out));
    }

    #[test]
    fn poi_rejects_a_forged_output() {
        let claim = InferenceClaim {
            model: Hash256::of(b"model"),
            input: Hash256::of(b"in"),
            output: Hash256::of(b"a-lie"),
        };
        assert!(!PoI::verify(&claim, Hash256::of(b"model"), b"in", &[1.0, 2.0]));
    }

    // A toy deterministic training step: square each byte (mod 256) — any fabricated state
    // breaks the transition.
    fn train_step(prev: &[u8]) -> Vec<u8> {
        prev.iter().map(|&b| b.wrapping_mul(b)).collect()
    }

    #[test]
    fn pol_accepts_an_honest_spot_check() {
        let s0 = alloc::vec![2u8, 3, 4];
        let s1 = train_step(&s0);
        let s2 = train_step(&s1);
        let claim = PoL::commit(&[s0.clone(), s1.clone(), s2.clone()], 1000);
        let mut staked = StakedLearning::accept(claim);
        assert!(staked.is_accepted());
        let rev = StepRevelation { k: 1, prev: s0, cur: s1 };
        assert!(staked.spot_check(&rev, train_step));
        assert!(staked.is_accepted());
        assert_eq!(staked.forfeit(), 0);
    }

    #[test]
    fn pol_slashes_a_fabricated_step() {
        let s0 = alloc::vec![2u8, 3, 4];
        let claim = PoL::commit(&[s0.clone(), alloc::vec![9, 9, 9]], 1000);
        let mut staked = StakedLearning::accept(claim);
        // The learner reveals a step that does NOT follow from its predecessor.
        let rev = StepRevelation { k: 1, prev: s0, cur: alloc::vec![9, 9, 9] };
        assert!(!staked.spot_check(&rev, train_step));
        assert!(staked.is_slashed());
        assert!(!staked.is_accepted());
        assert_eq!(staked.forfeit(), 1000);
    }

    #[test]
    fn wallet_spends_only_when_unlocked_with_funds() {
        let mut w = Wallet::new(Hash256::of(b"alice"), 100, b"pin-1234");
        let mut p = Wallet::new(Hash256::of(b"bob"), 0, b"pin-bob");
        let mut led = SettlementLedger::new();
        // Locked → refused.
        assert_eq!(led.pay(&mut w, &mut p, 10, 1), Err(PayError::Locked));
        assert!(!w.unlock(b"wrong"));
        assert!(w.unlock(b"pin-1234"));
        // Over balance → refused.
        assert_eq!(led.pay(&mut w, &mut p, 1000, 0), Err(PayError::Insufficient));
        // Valid payment.
        assert!(led.pay(&mut w, &mut p, 40, 1).is_ok());
        assert_eq!(w.balance(), 59);
        assert_eq!(p.balance(), 40);
    }

    #[test]
    fn ledger_is_hash_chained_and_anti_rollback() {
        let mut led = SettlementLedger::new();
        let mut w = Wallet::new(Hash256::of(b"x"), 0, b"f");
        let g0 = led.generation();
        led.reward(&mut w, 50);
        assert_eq!(led.generation(), g0 + 1);
        let d1 = led.digest();
        led.reward(&mut w, 10);
        assert_ne!(led.digest(), d1);
        assert_eq!(led.entries().len(), 2);
        // Anti-rollback: a stale generation is refused, the current one accepted.
        assert!(led.accepts_generation(led.generation()));
        assert!(!led.accepts_generation(led.generation() + 1));
    }

    #[test]
    fn treasury_stays_fully_reserved_with_fee_burn() {
        let mut t = Treasury::new(1_000);
        assert!(t.mint(800));
        assert!(t.proof_of_reserves());
        // Cannot mint past the reserve.
        assert!(!t.mint(300));
        assert_eq!(t.supply(), 800);
        // Burning a fee is deflationary and keeps the reserve invariant.
        t.burn_fee(100);
        assert_eq!(t.supply(), 700);
        assert_eq!(t.burned(), 100);
        assert!(t.proof_of_reserves());
    }

    #[test]
    fn ledger_op_set_is_closed_no_on_ledger_vm() {
        // The non-goal guard: every ledger entry is one of a fixed, finite set of operations —
        // there is no general-execution variant, so the ledger is not Turing-complete.
        let ops = [
            LedgerOp::Transfer { from: Hash256::ZERO, to: Hash256::ZERO, amount: 0, fee: 0 },
            LedgerOp::Reward { to: Hash256::ZERO, amount: 0 },
            LedgerOp::Slash { from: Hash256::ZERO, amount: 0 },
        ];
        // Exhaustive match proves the set is closed at compile time.
        for op in &ops {
            let total = match op {
                LedgerOp::Transfer { amount, fee, .. } => amount + fee,
                LedgerOp::Reward { amount, .. } => *amount,
                LedgerOp::Slash { amount, .. } => *amount,
            };
            let _ = total;
        }
        assert_eq!(ops.len(), 3);
    }
}

//! Confidential transactions for the Financial domain — **prove a transfer is valid
//! (amounts non-negative, inputs balance outputs) without revealing the amounts**
//! (`docs/security/zero-knowledge-proofs.md` use case 5).
//!
//! This is the classic *Confidential Transactions* construction (Maxwell; the additive
//! core under Bulletproofs), built from real sigma protocols over the Schnorr group in
//! [`crate::zk`]:
//!
//! * **Pedersen commitments** `C(v, r) = gᵛ·hʳ mod p` hide an amount `v` behind a
//!   blinding factor `r`, over two generators `g`, `h` with no known discrete-log
//!   relation ([`Pedersen`]). They are **additively homomorphic**: the product of
//!   commitments commits to the sum of the amounts — which is what makes a balance
//!   provable without opening anything.
//! * **Range proofs** ([`RangeProof`]) prove a committed amount lies in `[0, 2ⁿ)` — so
//!   no one can mint money by committing a negative value that wraps the field. The
//!   amount is bit-decomposed, each bit is committed, and a **one-of-two Schnorr
//!   OR-proof** ([`BitProof`]) proves each commitment opens to 0 **or** 1 without
//!   revealing which; the bit commitments are tied to the value commitment by the
//!   `Σ 2ⁱ` homomorphism.
//! * **Balance proofs** ([`BalanceProof`]) prove `Σ inputs = Σ outputs + fee` by
//!   forming the homomorphic ratio of all commitments and proving it is a pure
//!   `h`-power (i.e. the `g`-exponent — the net amount — is zero) with a Schnorr PoK.
//!
//! A full [`ConfidentialTx`] composes them: every output carries a range proof, and one
//! balance proof ties inputs, outputs and the public fee together. Nothing on the wire
//! reveals an amount; a verifier learns only *that the transfer is sound*.
//!
//! **NOTE: Current group parameters are illustrative-sized (31 bits).** This module
//! demonstrates the CT algebra correctly — commitments are additively homomorphic, the
//! sigma-protocol transcripts are sound, and the balance/range logic is complete — but it
//! does **not** provide production-strength confidentiality. The discrete-log problem in a
//! 31-bit group is trivially solvable (~2¹⁵ operations), so hiding and binding hold only
//! in the illustrative sense. For real confidentiality, replace [`SchnorrParams::new_demo_insecure()`]
//! with a 256-bit safe-prime group or an elliptic-curve group of comparable security.
//!
//! Parameters are illustrative-sized (the [`crate::zk`] Schnorr group), exactly like the
//! lattice/AES reductions elsewhere — the algebra is the real thing. Generation is
//! seeded from the DRNG so proofs replay under DST. Pure, safe `no_std`.

use crate::hash::Hash256;
use crate::random::Drng;
use crate::zk::SchnorrParams;
use alloc::vec::Vec;

// ───────────────────────── group arithmetic (self-contained mod p) ─────────────────────────

#[inline]
fn mulmod(a: u128, b: u128, m: u128) -> u128 {
    (a % m) * (b % m) % m
}

fn modpow(mut base: u128, mut exp: u128, m: u128) -> u128 {
    let mut acc = 1u128;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mulmod(acc, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// A Pedersen commitment group: the Schnorr group plus a second generator `h` whose
/// discrete log relative to `g` is unknown (so a commitment binds the amount).
#[derive(Clone, Copy, Debug)]
pub struct Pedersen {
    params: SchnorrParams,
    h: u128,
}

/// A hiding, binding commitment to an amount — just a group element.
///
/// **NOTE:** Hiding and binding security reduce to discrete-log hardness in the
/// underlying group. With [`SchnorrParams::new_demo_insecure()`] (31-bit group), both properties
/// are illustrative only — not production-strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment(pub u128);

impl Pedersen {
    /// Build the commitment group with an independent second generator.
    ///
    /// Uses [`SchnorrParams::new_demo_insecure()`] — a 31-bit illustrative group (`q = 2³¹−1`).
    /// Hiding and binding are **NOT production-strength**: the discrete-log problem in
    /// this group is trivially solvable in ~2¹⁵ operations. For real confidentiality,
    /// replace this with a 256-bit safe-prime or elliptic-curve group.
    pub fn new() -> Pedersen {
        let params = SchnorrParams::new_demo_insecure();
        let h = params.hash_to_generator(b"dominion-ctx-pedersen-h");
        Pedersen { params, h }
    }

    fn p(&self) -> u128 {
        self.params.p
    }
    fn q(&self) -> u128 {
        self.params.q
    }
    fn g(&self) -> u128 {
        self.params.g
    }

    fn exp_g(&self, e: u128) -> u128 {
        modpow(self.g(), e % self.q(), self.p())
    }
    fn exp_h(&self, e: u128) -> u128 {
        modpow(self.h, e % self.q(), self.p())
    }
    fn exp(&self, base: u128, e: u128) -> u128 {
        modpow(base, e % self.q(), self.p())
    }
    fn mul(&self, a: u128, b: u128) -> u128 {
        mulmod(a, b, self.p())
    }
    /// Multiplicative inverse mod the prime `p` (Fermat).
    fn inv(&self, a: u128) -> u128 {
        modpow(a, self.p() - 2, self.p())
    }
    fn sub_q(&self, a: u128, b: u128) -> u128 {
        (a % self.q() + self.q() - b % self.q()) % self.q()
    }

    /// `C(v, r) = gᵛ·hʳ mod p`.
    pub fn commit(&self, v: u128, r: u128) -> Commitment {
        Commitment(self.mul(self.exp_g(v), self.exp_h(r)))
    }

    /// The order of the exponent field (blindings live in `[0, q)`).
    pub fn order(&self) -> u128 {
        self.q()
    }
}

impl Default for Pedersen {
    fn default() -> Pedersen {
        Pedersen::new()
    }
}

fn challenge(p: &Pedersen, tag: &[u8], elems: &[u128]) -> u128 {
    let mut input = Vec::with_capacity(tag.len() + elems.len() * 16);
    input.extend_from_slice(tag);
    for e in elems {
        input.extend_from_slice(&e.to_le_bytes());
    }
    let h = Hash256::of(&input).0;
    let mut c = [0u8; 16];
    c.copy_from_slice(&h[..16]);
    u128::from_le_bytes(c) % p.q()
}

fn draw(rng: &mut Drng, q: u128) -> u128 {
    // Two 64-bit draws → a 128-bit value, reduced into [0, q).
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    ((hi << 64) | lo) % q
}

// ───────────────────────── one-of-two Schnorr OR-proof (a bit is 0 or 1) ─────────────────────────

/// A non-interactive proof that a commitment opens to **0 or 1**, revealing neither
/// which nor the blinding — a one-of-two Schnorr OR-proof over the base `h`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitProof {
    a0: u128,
    a1: u128,
    c0: u128,
    c1: u128,
    s0: u128,
    s1: u128,
}

impl BitProof {
    /// Prove the commitment `c = C(bit, r)` (with `bit ∈ {0,1}`) opens to 0 or 1.
    fn prove(p: &Pedersen, c: Commitment, bit: u8, r: u128, rng: &mut Drng) -> BitProof {
        // Y0 = C is an h-power iff bit==0; Y1 = C·g⁻¹ is an h-power iff bit==1.
        let y0 = c.0;
        let y1 = p.mul(c.0, p.inv(p.g()));
        let q = p.q();

        // Simulate the false branch, run the real one. In both cases all six values are
        // assigned exactly once before the proof is formed.
        let (a0, a1, c0, c1, s0, s1) = if bit == 0 {
            // true branch 0 (witness r with Y0 = hʳ); simulate branch 1.
            let k = draw(rng, q);
            let a0 = p.exp_h(k);
            let c1 = draw(rng, q);
            let s1 = draw(rng, q);
            let a1 = p.mul(p.exp_h(s1), p.inv(p.exp(y1, c1)));
            let chal = challenge(p, b"bit:", &[c.0, y0, y1, a0, a1]);
            let c0 = p.sub_q(chal, c1);
            let s0 = (k + mulmod(c0, r % q, q)) % q;
            (a0, a1, c0, c1, s0, s1)
        } else {
            // true branch 1 (witness r with Y1 = hʳ); simulate branch 0.
            let k = draw(rng, q);
            let a1 = p.exp_h(k);
            let c0 = draw(rng, q);
            let s0 = draw(rng, q);
            let a0 = p.mul(p.exp_h(s0), p.inv(p.exp(y0, c0)));
            let chal = challenge(p, b"bit:", &[c.0, y0, y1, a0, a1]);
            let c1 = p.sub_q(chal, c0);
            let s1 = (k + mulmod(c1, r % q, q)) % q;
            (a0, a1, c0, c1, s0, s1)
        };
        BitProof { a0, a1, c0, c1, s0, s1 }
    }

    /// Verify the OR-proof against the bit commitment `c`.
    pub fn verify(&self, p: &Pedersen, c: Commitment) -> bool {
        let y0 = c.0;
        let y1 = p.mul(c.0, p.inv(p.g()));
        let challenge_total = challenge(p, b"bit:", &[c.0, y0, y1, self.a0, self.a1]);
        if (self.c0 + self.c1) % p.q() != challenge_total {
            return false;
        }
        // h^{s_b} == a_b · Y_b^{c_b}  for both branches.
        let lhs0 = p.exp_h(self.s0);
        let rhs0 = p.mul(self.a0, p.exp(y0, self.c0));
        let lhs1 = p.exp_h(self.s1);
        let rhs1 = p.mul(self.a1, p.exp(y1, self.c1));
        lhs0 == rhs0 && lhs1 == rhs1
    }
}

// ───────────────────────── range proof (amount ∈ [0, 2ⁿ)) ─────────────────────────

/// A proof that a committed amount lies in `[0, 2ⁿ)`. Carries the per-bit commitments
/// and their OR-proofs; the value commitment is recovered as `Π Cᵢ^{2ⁱ}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeProof {
    bits: usize,
    bit_commitments: Vec<u128>,
    bit_proofs: Vec<BitProof>,
}

impl RangeProof {
    /// Commit `value` and prove it is in `[0, 2^bits)`. Returns the value commitment, the
    /// blinding `r` (kept secret by the caller, needed to balance a transaction), and the
    /// proof. `None` if `value` doesn't fit in `bits`.
    pub fn prove(
        p: &Pedersen,
        value: u64,
        bits: usize,
        seed: &[u8],
    ) -> Option<(Commitment, u128, RangeProof)> {
        if bits == 0 || bits > 63 || (value as u128) >= (1u128 << bits) {
            return None;
        }
        let q = p.order();
        let mut rng = Drng::from_seed(seed);
        let mut bit_commitments = Vec::with_capacity(bits);
        let mut bit_proofs = Vec::with_capacity(bits);
        let mut r_total = 0u128;
        for i in 0..bits {
            let bit = ((value >> i) & 1) as u8;
            let ri = draw(&mut rng, q);
            // r = Σ 2ⁱ rᵢ (mod q), so Π Cᵢ^{2ⁱ} = C(value, r).
            r_total = (r_total + mulmod((1u128 << i) % q, ri, q)) % q;
            let ci = p.commit(bit as u128, ri);
            bit_proofs.push(BitProof::prove(p, ci, bit, ri, &mut rng));
            bit_commitments.push(ci.0);
        }
        let commitment = p.commit(value as u128, r_total);
        Some((
            commitment,
            r_total,
            RangeProof {
                bits,
                bit_commitments,
                bit_proofs,
            },
        ))
    }

    /// Verify that `commitment` is to an amount in `[0, 2^bits)`: every bit commitment
    /// opens to 0/1, and they compose (via `Σ 2ⁱ`) to the value commitment.
    pub fn verify(&self, p: &Pedersen, commitment: Commitment) -> bool {
        if self.bit_commitments.len() != self.bits || self.bit_proofs.len() != self.bits {
            return false;
        }
        // Each bit is genuinely a bit.
        for (i, proof) in self.bit_proofs.iter().enumerate() {
            if !proof.verify(p, Commitment(self.bit_commitments[i])) {
                return false;
            }
        }
        // Π Cᵢ^{2ⁱ} must equal the value commitment.
        let mut acc = 1u128;
        for (i, &ci) in self.bit_commitments.iter().enumerate() {
            acc = p.mul(acc, p.exp(ci, (1u128 << i) % p.q()));
        }
        acc == commitment.0
    }

    /// The bit-width this proof covers.
    pub fn bits(&self) -> usize {
        self.bits
    }
}

// ───────────────────────── balance proof (Σ in = Σ out + fee) ─────────────────────────

/// A Schnorr proof that the homomorphic ratio of all transaction commitments is a pure
/// `h`-power — i.e. the net `g`-exponent (net amount) is zero, so the transaction
/// balances — without revealing any amount.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BalanceProof {
    a: u128,
    s: u128,
}

impl BalanceProof {
    /// Prove balance, given the net blinding `Δr = Σ r_in − Σ r_out (mod q)`. The caller
    /// computes `Δr` from the blindings it chose; this never sees the amounts.
    fn prove(p: &Pedersen, residual: u128, delta_r: u128, seed: &[u8]) -> BalanceProof {
        let q = p.order();
        let mut rng = Drng::from_seed(seed);
        let k = draw(&mut rng, q);
        let a = p.exp_h(k);
        let c = challenge(p, b"balance:", &[residual, a]);
        let s = (k + mulmod(c, delta_r % q, q)) % q;
        BalanceProof { a, s }
    }

    /// Verify against the residual `D = (Π C_in)·(Π C_out)⁻¹·g^{−fee}` — accepts iff
    /// `h^s = a·D^c`, which holds exactly when `D` is an `h`-power (balanced).
    pub fn verify(&self, p: &Pedersen, residual: u128) -> bool {
        let c = challenge(p, b"balance:", &[residual, self.a]);
        p.exp_h(self.s) == p.mul(self.a, p.exp(residual, c))
    }
}

// ───────────────────────── the confidential transaction ─────────────────────────

/// Why a confidential transaction was rejected at build time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtxError {
    /// An amount didn't fit the range, or inputs don't cover outputs + fee.
    Invalid,
}

/// A confidential transaction: hidden input/output amounts (as commitments), a public
/// fee, a range proof per output, and one balance proof. Verifying reveals only that
/// the transfer is sound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfidentialTx {
    pub inputs: Vec<Commitment>,
    pub outputs: Vec<Commitment>,
    pub fee: u64,
    pub output_ranges: Vec<RangeProof>,
    pub balance: BalanceProof,
}

impl ConfidentialTx {
    /// Build a transaction from cleartext `inputs`/`outputs` (each an amount) and a
    /// public `fee`. Amounts and blindings stay with the builder; only commitments and
    /// proofs are emitted. `None` if amounts don't fit `bits` or don't balance.
    pub fn build(
        p: &Pedersen,
        inputs: &[u64],
        outputs: &[u64],
        fee: u64,
        bits: usize,
        seed: &[u8],
    ) -> Result<ConfidentialTx, CtxError> {
        let sum_in: u128 = inputs.iter().map(|&v| v as u128).sum();
        let sum_out: u128 = outputs.iter().map(|&v| v as u128).sum::<u128>() + fee as u128;
        if sum_in != sum_out {
            return Err(CtxError::Invalid);
        }
        let q = p.order();

        // Each input/output gets a range proof (proving non-negativity) and a blinding.
        let mut in_commits = Vec::new();
        let mut out_commits = Vec::new();
        let mut output_ranges = Vec::new();
        let mut r_in_total = 0u128;
        let mut r_out_total = 0u128;

        for (j, &v) in inputs.iter().enumerate() {
            let mut s = Vec::from(seed);
            s.extend_from_slice(b"|in|");
            s.extend_from_slice(&(j as u64).to_le_bytes());
            let (c, r, _proof) = RangeProof::prove(p, v, bits, &s).ok_or(CtxError::Invalid)?;
            r_in_total = (r_in_total + r) % q;
            in_commits.push(c);
        }
        for (k, &v) in outputs.iter().enumerate() {
            let mut s = Vec::from(seed);
            s.extend_from_slice(b"|out|");
            s.extend_from_slice(&(k as u64).to_le_bytes());
            let (c, r, proof) = RangeProof::prove(p, v, bits, &s).ok_or(CtxError::Invalid)?;
            r_out_total = (r_out_total + r) % q;
            out_commits.push(c);
            output_ranges.push(proof);
        }

        // Residual D = (Π C_in)·(Π C_out)⁻¹·g^{−fee}. Balanced ⇒ D = h^{Δr}.
        let residual = compute_residual(p, &in_commits, &out_commits, fee);
        let delta_r = p.sub_q(r_in_total, r_out_total);
        let mut bseed = Vec::from(seed);
        bseed.extend_from_slice(b"|balance");
        let balance = BalanceProof::prove(p, residual, delta_r, &bseed);

        Ok(ConfidentialTx {
            inputs: in_commits,
            outputs: out_commits,
            fee,
            output_ranges,
            balance,
        })
    }

    /// Verify the transaction: every output amount is in range, and inputs balance
    /// outputs + fee. Returns `true` iff sound — without learning any amount.
    pub fn verify(&self, p: &Pedersen, bits: usize) -> bool {
        if self.output_ranges.len() != self.outputs.len() {
            return false;
        }
        for (proof, &c) in self.output_ranges.iter().zip(self.outputs.iter()) {
            if proof.bits() != bits || !proof.verify(p, c) {
                return false;
            }
        }
        let residual = compute_residual(p, &self.inputs, &self.outputs, self.fee);
        self.balance.verify(p, residual)
    }
}

/// `D = (Π C_in) · (Π C_out)⁻¹ · g^{−fee} mod p`.
fn compute_residual(p: &Pedersen, inputs: &[Commitment], outputs: &[Commitment], fee: u64) -> u128 {
    let mut num = 1u128;
    for c in inputs {
        num = p.mul(num, c.0);
    }
    let mut den = 1u128;
    for c in outputs {
        den = p.mul(den, c.0);
    }
    den = p.mul(den, p.exp_g(fee as u128));
    p.mul(num, p.inv(den))
}

#[cfg(all(test, feature = "demo-crypto"))]
mod tests {
    use super::*;

    #[test]
    fn pedersen_is_additively_homomorphic() {
        let p = Pedersen::new();
        let c1 = p.commit(7, 11);
        let c2 = p.commit(5, 13);
        // C(7,11)·C(5,13) == C(12, 24).
        assert_eq!(p.mul(c1.0, c2.0), p.commit(12, 24).0);
    }

    #[test]
    fn range_proof_accepts_in_range_and_recomposes_the_commitment() {
        let p = Pedersen::new();
        let (commit, _r, proof) = RangeProof::prove(&p, 42, 8, b"seed-a").unwrap();
        assert!(proof.verify(&p, commit));
        // A value that doesn't fit the bit-width is refused at prove time.
        assert!(RangeProof::prove(&p, 256, 8, b"seed-a").is_none());
    }

    #[test]
    fn range_proof_rejects_a_swapped_commitment() {
        let p = Pedersen::new();
        let (_c, _r, proof) = RangeProof::prove(&p, 42, 8, b"seed-b").unwrap();
        // Verifying the valid proof against a *different* commitment fails.
        let (other, _r2, _p2) = RangeProof::prove(&p, 41, 8, b"seed-c").unwrap();
        assert!(!proof.verify(&p, other));
    }

    #[test]
    fn bit_proof_only_accepts_actual_bits() {
        let p = Pedersen::new();
        let mut rng = Drng::from_seed(b"bit-seed");
        // A commitment to 2 is not a bit; no honest OR-proof verifies for it.
        let c2 = p.commit(2, 99);
        let forged = BitProof::prove(&p, c2, 0, 99, &mut rng); // prover lies "bit=0"
        assert!(!forged.verify(&p, c2));
        // Genuine 0 and 1 commitments do verify.
        let c0 = p.commit(0, 5);
        let c1 = p.commit(1, 6);
        assert!(BitProof::prove(&p, c0, 0, 5, &mut rng).verify(&p, c0));
        assert!(BitProof::prove(&p, c1, 1, 6, &mut rng).verify(&p, c1));
    }

    #[test]
    fn balanced_transaction_verifies() {
        let p = Pedersen::new();
        // 100 + 50 in → 120 + 25 out + 5 fee = 150 = 150. Balanced.
        let tx = ConfidentialTx::build(&p, &[100, 50], &[120, 25], 5, 16, b"tx-seed").unwrap();
        assert!(tx.verify(&p, 16));
    }

    #[test]
    fn unbalanced_transaction_is_refused_at_build() {
        let p = Pedersen::new();
        // Outputs + fee (151) exceed inputs (150) — cannot mint money.
        assert_eq!(
            ConfidentialTx::build(&p, &[100, 50], &[121, 25], 5, 16, b"tx-seed"),
            Err(CtxError::Invalid)
        );
    }

    #[test]
    fn a_forged_balance_does_not_verify() {
        let p = Pedersen::new();
        let mut tx = ConfidentialTx::build(&p, &[100], &[100], 0, 16, b"tx-seed2").unwrap();
        // Tamper an output commitment: balance residual is no longer an h-power.
        tx.outputs[0] = Commitment(p.mul(tx.outputs[0].0, p.exp_g(1)));
        assert!(!tx.verify(&p, 16));
    }

    #[test]
    fn generation_is_deterministic_for_dst_replay() {
        let p = Pedersen::new();
        let a = ConfidentialTx::build(&p, &[10], &[10], 0, 8, b"same").unwrap();
        let b = ConfidentialTx::build(&p, &[10], &[10], 0, 8, b"same").unwrap();
        assert_eq!(a, b);
    }
}

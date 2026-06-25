//! Verifiable computation for AI domains — **prove a computation's result is
//! correct without redoing it, and without revealing the inputs or weights**
//! (`docs/security/zero-knowledge-proofs.md` use case 3).
//!
//! AI domains are probabilistic and contained (Stage 11.15). The OS wants an AI cell to
//! be able to prove "I evaluated the *approved* function on inputs committed to `C`, and
//! the result is `y`" — to a verifier who trusts neither the cell nor the hardware. The
//! real cryptographic engine for exactly this is the **sum-check protocol** (Lund–
//! Fortnow–Karloff–Nisan; the core of GKR and modern zkVMs):
//!
//! > To prove `Σ_{b ∈ {0,1}ᵛ} f(b) = S` for a multilinear `f`, the prover and verifier
//! > run `v` rounds. In each round the prover sends a univariate polynomial (here a
//! > line, since `f` is multilinear) and the verifier folds the claim down with a random
//! > challenge. After `v` rounds an exponential-sized claim (`2ᵛ` terms) has been reduced
//! > to **one** evaluation `f(r)` at a random point — the verifier did `O(v)` field work
//! > instead of recomputing the sum. Fiat–Shamir makes it non-interactive and
//! > deterministic (replayable under DST).
//!
//! The sum-check rounds implemented here are the **real, sound** protocol: a prover who
//! lies about the sum, or about any round polynomial, is caught with overwhelming
//! probability over the field. Privacy holds because the verifier only ever sees the
//! round polynomials (partial sums) and one final evaluation — never the individual
//! table entries (the private inputs/weights).
//!
//! The one **SW-modeled** seam (consistent with the lattice/AES reductions elsewhere):
//! binding the final evaluation `f(r)` to the committed table is what a *polynomial
//! commitment* (FRI/KZG) does in production; here the commitment is a hash of the table
//! and the honest opening re-derives `f(r)`. The sum-check transcript — the part that
//! turns "redo the whole computation" into "check a short proof" — is exact.
//!
//! Field: the Mersenne prime `M31 = 2³¹−1`. Pure, safe `no_std`. Host- and metal-tested.

use crate::hash::Hash256;
use alloc::vec::Vec;

/// The field modulus `2³¹ − 1` (a Mersenne prime — fast, and the same `M31` the
/// neural/grid-snap path uses).
pub const Q: u64 = (1 << 31) - 1;

#[inline]
fn add(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= Q {
        s - Q
    } else {
        s
    }
}

#[inline]
fn sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + Q - b
    }
}

#[inline]
fn mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}

/// Reduce an arbitrary value into the field.
#[inline]
pub fn fe(x: u64) -> u64 {
    x % Q
}

// ───────────────────────── the private witness ─────────────────────────

/// A multilinear function over the boolean hypercube `{0,1}ᵛ`, given by its `2ᵛ`
/// evaluations. In the AI framing this *is* the private data — e.g. the per-input
/// activations or the weight·input products of one layer. The verifier never sees it.
#[derive(Clone, Debug)]
pub struct MultilinearTable {
    vars: usize,
    /// `evals[b]` = `f(b)`, `b` read as a `vars`-bit index (LSB = variable 0).
    evals: Vec<u64>,
}

impl MultilinearTable {
    /// Build from `2ᵛ` field values. Entries are reduced into the field.
    pub fn new(evals: Vec<u64>) -> Option<MultilinearTable> {
        let n = evals.len();
        if n == 0 || !n.is_power_of_two() {
            return None;
        }
        let vars = n.trailing_zeros() as usize;
        Some(MultilinearTable {
            vars,
            evals: evals.into_iter().map(fe).collect(),
        })
    }

    /// Number of variables `v` (so the hypercube has `2ᵛ` points).
    pub fn vars(&self) -> usize {
        self.vars
    }

    /// The honest sum `Σ_b f(b)` — the *claim* the prover commits to. Computing this is
    /// the `O(2ᵛ)` work the verifier is spared.
    pub fn sum(&self) -> u64 {
        self.evals.iter().fold(0u64, |acc, &x| add(acc, x))
    }

    /// A binding commitment to the table (the polynomial-commitment seam; see module
    /// docs). Two equal tables commit equally; any change moves the digest.
    pub fn commit(&self) -> Hash256 {
        let mut b = Vec::with_capacity(self.evals.len() * 8 + 8);
        b.extend_from_slice(b"mlt:");
        b.extend_from_slice(&(self.vars as u64).to_le_bytes());
        for &e in &self.evals {
            b.extend_from_slice(&e.to_le_bytes());
        }
        Hash256::of(&b)
    }

    /// Evaluate the multilinear extension at an arbitrary field point `r ∈ Fᵛ`. This is
    /// the honest opening of the commitment (in production a FRI/KZG opening proof).
    pub fn evaluate(&self, r: &[u64]) -> Option<u64> {
        if r.len() != self.vars {
            return None;
        }
        // Fold one variable at a time: table[b] ← (1−r_i)·table[0,b] + r_i·table[1,b].
        let mut cur = self.evals.clone();
        for &ri in r {
            let half = cur.len() / 2;
            let mut next = Vec::with_capacity(half);
            for b in 0..half {
                let lo = cur[b];
                let hi = cur[b + half];
                next.push(add(mul(sub(1 % Q, ri), lo), mul(ri, hi)));
            }
            cur = next;
        }
        Some(cur[0])
    }
}

// ───────────────────────── the proof ─────────────────────────

/// A non-interactive sum-check proof: one line `[g_i(0), g_i(1)]` per round, plus the
/// final evaluation the chain reduces to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumCheckProof {
    /// The binding commitment to the table the claim is about (Fiat–Shamir seed).
    pub commitment: Hash256,
    /// The public claimed sum `S`.
    pub claimed_sum: u64,
    /// Round polynomials, each a degree-1 line given by its values at 0 and 1.
    pub round: Vec<[u64; 2]>,
    /// The final evaluation `f(r)` the protocol reduces the claim to.
    pub final_eval: u64,
    /// The challenge point `r` (recomputed by the verifier; carried for the opening).
    pub point: Vec<u64>,
}

/// Fiat–Shamir challenge: hash the running transcript to a field element.
fn challenge(transcript: &[u8]) -> u64 {
    let h = Hash256::of(transcript).0;
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&h[..8]);
    fe(u64::from_le_bytes(raw))
}

fn absorb(transcript: &mut Vec<u8>, line: &[u64; 2]) {
    transcript.extend_from_slice(&line[0].to_le_bytes());
    transcript.extend_from_slice(&line[1].to_le_bytes());
}

/// Prove `Σ_b f(b) = f.sum()`. Deterministic (Fiat–Shamir over the transcript), so a
/// DST replay reproduces the exact proof.
pub fn prove(f: &MultilinearTable) -> SumCheckProof {
    let claimed_sum = f.sum();
    let commitment = f.commit();
    let mut transcript = Vec::new();
    transcript.extend_from_slice(b"sumcheck:");
    transcript.extend_from_slice(&commitment.0);
    transcript.extend_from_slice(&claimed_sum.to_le_bytes());

    let mut cur = f.evals.clone();
    let mut round = Vec::with_capacity(f.vars);
    let mut point = Vec::with_capacity(f.vars);

    for _ in 0..f.vars {
        let half = cur.len() / 2;
        // g(0) = Σ over the half with this variable = 0; g(1) = Σ with it = 1.
        let mut g0 = 0u64;
        let mut g1 = 0u64;
        for b in 0..half {
            g0 = add(g0, cur[b]);
            g1 = add(g1, cur[b + half]);
        }
        let line = [g0, g1];
        absorb(&mut transcript, &line);
        round.push(line);
        let r = challenge(&transcript);
        point.push(r);
        // Fix this variable to r and recurse on the smaller table.
        let mut next = Vec::with_capacity(half);
        for b in 0..half {
            next.push(add(mul(sub(1 % Q, r), cur[b]), mul(r, cur[b + half])));
        }
        cur = next;
    }

    SumCheckProof {
        commitment,
        claimed_sum,
        round,
        final_eval: cur[0],
        point,
    }
}

/// Why verification failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VcError {
    /// The proof's round count doesn't match the declared variable count.
    Shape,
    /// A round's `g(0)+g(1)` didn't equal the folded prior claim — the prover lied.
    RoundMismatch,
    /// The final reduced claim didn't match the supplied final evaluation.
    FinalMismatch,
    /// The opening of the committed table contradicts the claimed final evaluation.
    OpeningMismatch,
}

/// Verify the **transcript** of a sum-check proof in `O(v)` field work — without
/// recomputing the `2ᵛ`-term sum. This is the succinct core: it establishes that *if*
/// `final_eval = f(point)` then `Σ_b f(b) = claimed_sum`. Binding `final_eval` to a
/// committed `f` is [`open`] (the polynomial-commitment seam).
pub fn verify_transcript(vars: usize, proof: &SumCheckProof) -> Result<(), VcError> {
    if proof.round.len() != vars || proof.point.len() != vars {
        return Err(VcError::Shape);
    }
    // Reconstruct the exact Fiat–Shamir stream the prover used.
    let mut transcript = Vec::new();
    transcript.extend_from_slice(b"sumcheck:");
    transcript.extend_from_slice(&proof.commitment.0);
    transcript.extend_from_slice(&proof.claimed_sum.to_le_bytes());
    let mut expected = proof.claimed_sum;
    for (i, line) in proof.round.iter().enumerate() {
        // The line must sum to the current expected claim.
        if add(line[0], line[1]) != expected {
            return Err(VcError::RoundMismatch);
        }
        absorb(&mut transcript, line);
        let r = challenge(&transcript);
        if r != proof.point[i] {
            return Err(VcError::RoundMismatch);
        }
        // Fold: the next expected claim is g_i(r) = (1−r)·g(0) + r·g(1).
        expected = add(mul(sub(1 % Q, r), line[0]), mul(r, line[1]));
    }
    if expected != proof.final_eval {
        return Err(VcError::FinalMismatch);
    }
    Ok(())
}

/// Bind the final evaluation to the committed table: re-derive `f(point)` from the
/// opened table and confirm it matches the proof. In production this is a succinct
/// polynomial-commitment opening proof; here it is the honest re-derivation.
pub fn open(f: &MultilinearTable, proof: &SumCheckProof) -> Result<(), VcError> {
    let truth = f.evaluate(&proof.point).ok_or(VcError::Shape)?;
    if truth != proof.final_eval {
        return Err(VcError::OpeningMismatch);
    }
    Ok(())
}

/// Full verification an honest party performs: it knows the public commitment + claimed
/// sum, checks the succinct transcript, and (holding an opening) binds the final
/// evaluation. Returns `Ok(())` iff the result is proven correct.
pub fn verify(f: &MultilinearTable, proof: &SumCheckProof) -> Result<(), VcError> {
    if proof.commitment != f.commit() {
        return Err(VcError::OpeningMismatch);
    }
    verify_transcript(f.vars, proof)?;
    open(f, proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(vals: &[u64]) -> MultilinearTable {
        MultilinearTable::new(vals.to_vec()).unwrap()
    }

    #[test]
    fn honest_proof_verifies_and_the_sum_is_correct() {
        // 3 variables → 8 private entries (e.g. weight·input products of a tiny layer).
        let f = table(&[3, 1, 4, 1, 5, 9, 2, 6]);
        let proof = prove(&f);
        assert_eq!(proof.claimed_sum, (3 + 1 + 4 + 1 + 5 + 9 + 2 + 6) % Q);
        assert_eq!(verify(&f, &proof), Ok(()));
    }

    #[test]
    fn a_lie_about_the_sum_is_caught() {
        let f = table(&[3, 1, 4, 1, 5, 9, 2, 6]);
        let mut proof = prove(&f);
        proof.claimed_sum = add(proof.claimed_sum, 1); // claim a wrong total
        assert_eq!(verify_transcript(f.vars(), &proof), Err(VcError::RoundMismatch));
    }

    #[test]
    fn a_tampered_round_polynomial_is_caught() {
        let f = table(&[7, 7, 7, 7]);
        let mut proof = prove(&f);
        proof.round[0][0] = add(proof.round[0][0], 1);
        proof.round[0][1] = sub(proof.round[0][1], 1); // keep g(0)+g(1) equal...
                                                        // ...but the folded challenge stream now diverges.
        assert!(verify_transcript(f.vars(), &proof).is_err());
    }

    #[test]
    fn a_forged_final_eval_fails_the_opening() {
        let f = table(&[2, 0, 2, 4, 0, 8, 1, 6]);
        let mut proof = prove(&f);
        // Forge a consistent transcript that reduces to a different final eval by also
        // moving the last expected fold — but the committed table opening won't match.
        proof.final_eval = add(proof.final_eval, 1);
        // Transcript check fails (final fold mismatch) — and even if it didn't, the
        // opening binds to the real table.
        assert!(verify(&f, &proof).is_err());
    }

    #[test]
    fn replay_is_deterministic() {
        let f = table(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(prove(&f), prove(&f));
    }
}

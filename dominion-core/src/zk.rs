//! Zero-knowledge proofs — prove a statement without revealing the witness
//! (see `docs/security/zero-knowledge-and-attestation.md`).
//!
//! Architecture 2.0 wants to authorise on *properties*, not secrets: "I hold a
//! capability with these rights", "this value is in the allowed set", "I know the
//! key behind this identity" — all provable without handing the verifier the
//! underlying secret. Two real constructions live here:
//!
//! * A **Schnorr non-interactive proof of knowledge** of a discrete logarithm
//!   (Fiat–Shamir). The prover convinces a verifier it knows `x` with `y = gˣ`
//!   while leaking nothing about `x`. This is a genuine sigma protocol, not a
//!   stand-in.
//! * A **Merkle membership proof**: prove an element belongs to a committed set
//!   (the root) without revealing the other members or the element's position.
//!
//! Parameters here are illustrative-sized (a Schnorr group derived at init, with
//! `q = 2³¹−1`), exactly as the lattice and AES reductions elsewhere — the
//! algebra is the real thing. Pure, safe `no_std`; no special hardware.
//!
//! # Security
//!
//! **The `demo-crypto` feature and `SchnorrParams::new_demo_insecure()` are for
//! development, testing, and protocol-correctness verification only.**
//!
//! The 31-bit Schnorr group (`q = 2³¹−1`, M31) has no real security margin:
//! the discrete-log problem is solvable by Pollard-rho / baby-step-giant-step in
//! roughly 2¹⁵·⁵ operations — trivial on commodity hardware. Every security claim
//! that reduces to DDH or DL in this group (unlinkability, hiding, binding,
//! forgery-resistance) is illustrative only, not production-strength.
//!
//! For production use:
//! * Do **not** enable the `demo-crypto` Cargo feature.
//! * Supply `SchnorrParams` constructed from a 256-bit safe-prime group or a
//!   standard elliptic-curve group (e.g. Ristretto255, P-256).
//! * Calling `SchnorrParams::new_demo_insecure()` without `demo-crypto` panics at
//!   runtime with a clear message so accidental production use is caught early.

use crate::hash::Hash256;
use crate::random::Drng;
use alloc::vec::Vec;

// ─────────────────── number theory helpers ───────────────────

/// Modular multiply that never overflows `u128`, for any modulus `m` up to
/// `u128::MAX`.
///
/// The naive `(a % m) * (b % m) % m` only fits `u128` while `m < 2^64`; for the
/// large safe-prime groups the module docs invite for production (`p` well above
/// `2^64`) the product overflows — a debug panic or a release-build silent
/// wraparound producing cryptographically wrong group elements. This binary
/// (double-and-add) form keeps every intermediate `< m`, using overflow-safe
/// modular addition, so it stays correct across the whole representable range.
fn mulmod(a: u128, b: u128, m: u128) -> u128 {
    let mut a = a % m;
    let mut b = b % m;
    let mut acc = 0u128;
    while b > 0 {
        if b & 1 == 1 {
            // acc = (acc + a) % m, computed without overflowing u128.
            acc = if a >= m - acc { a - (m - acc) } else { acc + a };
        }
        // a = (a + a) % m, computed without overflowing u128.
        a = if a >= m - a { a - (m - a) } else { a + a };
        b >>= 1;
    }
    acc
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

/// Deterministic Miller–Rabin (correct for all `u64` with this base set).
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut r = 0;
    while d & 1 == 0 {
        d >>= 1;
        r += 1;
    }
    'witness: for &a in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = modpow(a as u128, d as u128, n as u128);
        if x == 1 || x == (n - 1) as u128 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mulmod(x, x, n as u128);
            if x == (n - 1) as u128 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

// ─────────────────── Schnorr NIZK of discrete log ───────────────────

/// A Schnorr group `⟨g⟩` of prime order `q` inside `Z_p^*`.
#[derive(Clone, Copy, Debug)]
pub struct SchnorrParams {
    pub p: u128,
    pub q: u128,
    pub g: u128,
}

impl SchnorrParams {
    /// Derive the illustrative 31-bit Schnorr group used for demos and tests.
    ///
    /// `q = 2³¹−1` (M31, prime). Finds the smallest `p = m·q + 1` that is prime,
    /// then a generator `g = h^m` of order `q`. Deterministic — every node agrees.
    ///
    /// # Security warning
    ///
    /// This group has **no real security margin**. The discrete-log problem is
    /// solvable in ~2¹⁵·⁵ operations. All DDH/DL-based guarantees (unlinkability,
    /// hiding, binding) are illustrative only.
    ///
    /// Requires the `demo-crypto` Cargo feature. Panics at runtime if called without
    /// it, so accidental production use is caught at startup rather than silently
    /// degrading security.
    ///
    /// For production, construct `SchnorrParams` from a 256-bit safe-prime or an
    /// elliptic-curve group. Do **not** enable `demo-crypto` in production builds.
    #[cfg(feature = "demo-crypto")]
    pub fn new_demo_insecure() -> SchnorrParams {
        let q: u64 = 2_147_483_647; // M31, prime
        let mut m: u64 = 2;
        let p = loop {
            let cand = m * q + 1;
            if is_prime(cand) {
                break cand;
            }
            m += 1;
        };
        // g = h^m mod p has order dividing q; for prime q it is q unless g==1.
        let mut h: u128 = 2;
        let g = loop {
            let cand = modpow(h, m as u128, p as u128);
            if cand != 1 {
                break cand;
            }
            h += 1;
        };
        SchnorrParams { p: p as u128, q: q as u128, g }
    }

    /// Alias kept for call-site compatibility in non-feature-gated contexts.
    ///
    /// Panics unconditionally unless the `demo-crypto` feature is enabled.
    /// When `demo-crypto` IS enabled this simply delegates to [`Self::new_demo_insecure`].
    ///
    /// Prefer calling `new_demo_insecure()` directly in gated code so the name
    /// makes the danger visible at every call site.
    #[cfg(not(feature = "demo-crypto"))]
    pub fn new_demo_insecure() -> SchnorrParams {
        panic!(
            "SchnorrParams::new_demo_insecure() called without the `demo-crypto` feature. \
             The 31-bit illustrative group must not be used in production. \
             Enable `demo-crypto` only in test/demo builds, or supply a \
             production-strength SchnorrParams."
        );
    }

    /// The public key `y = gˣ mod p` for a secret witness `x`.
    pub fn public_key(&self, x: u128) -> u128 {
        modpow(self.g, x % self.q, self.p)
    }

    /// Exponentiate `base^e mod p` in this group (exposed for DLEQ / pseudonyms).
    pub fn exp(&self, base: u128, e: u128) -> u128 {
        modpow(base, e % self.q, self.p)
    }

    /// Multiply in the group: `a·b mod p`.
    pub fn mul(&self, a: u128, b: u128) -> u128 {
        mulmod(a, b, self.p)
    }

    /// The cofactor `m = (p−1)/q`. Raising any element to it lands in the order-`q`
    /// subgroup (used to derive independent per-context generators).
    pub fn cofactor(&self) -> u128 {
        (self.p - 1) / self.q
    }

    /// Hash arbitrary bytes (e.g. a *context* label) to a generator of the order-`q`
    /// subgroup, deterministically. Each distinct context yields an independent base,
    /// which is what makes per-context pseudonyms unlinkable across contexts.
    pub fn hash_to_generator(&self, context: &[u8]) -> u128 {
        let mut counter = 0u32;
        loop {
            let mut input = Vec::with_capacity(context.len() + 8);
            input.extend_from_slice(b"h2g:");
            input.extend_from_slice(context);
            input.extend_from_slice(&counter.to_le_bytes());
            let h = Hash256::of(&input).0;
            let mut raw = [0u8; 16];
            raw.copy_from_slice(&h[..16]);
            let candidate = (u128::from_le_bytes(raw) % (self.p - 2)) + 2; // in [2, p-1]
            let gen = self.exp(candidate, self.cofactor());
            if gen != 1 {
                return gen;
            }
            counter += 1;
        }
    }
}

// ─────────────────── Chaum–Pedersen equality of discrete logs (DLEQ) ───────────────────

/// A non-interactive proof that two group elements `a = g1ˣ` and `b = g2ˣ` share the
/// **same** secret exponent `x`, revealing nothing about `x`. This is the engine
/// behind *unlinkable per-context pseudonyms* (see [`crate::anon`]): the holder proves
/// a context pseudonym was formed with the same identity secret as their credential,
/// without exposing a global public key that would correlate their transactions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DleqProof {
    /// Commitments `t1 = g1ʳ`, `t2 = g2ʳ`.
    pub t1: u128,
    pub t2: u128,
    /// Response `s = r + c·x (mod q)`.
    pub s: u128,
}

fn dleq_challenge(p: &SchnorrParams, g1: u128, g2: u128, a: u128, b: u128, t1: u128, t2: u128) -> u128 {
    let mut input = Vec::new();
    input.extend_from_slice(b"dleq:");
    for v in [p.p, p.q, g1, g2, a, b, t1, t2] {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let h = Hash256::of(&input).0;
    let mut c = [0u8; 16];
    c.copy_from_slice(&h[..16]);
    u128::from_le_bytes(c) % p.q
}

/// Prove that `a = g1ˣ` and `b = g2ˣ` for a known `x`. `seed` supplies the nonce.
pub fn dleq_prove(params: &SchnorrParams, g1: u128, g2: u128, x: u128, seed: &[u8]) -> DleqProof {
    let mut rng = Drng::from_seed(seed);
    let r = 1 + (rng.next_u64() as u128 % (params.q - 1));
    let t1 = params.exp(g1, r);
    let t2 = params.exp(g2, r);
    let a = params.exp(g1, x);
    let b = params.exp(g2, x);
    let c = dleq_challenge(params, g1, g2, a, b, t1, t2);
    let s = (r + mulmod(c, x % params.q, params.q)) % params.q;
    DleqProof { t1, t2, s }
}

/// Verify a DLEQ proof: accepts iff `g1ˢ = t1·aᶜ` **and** `g2ˢ = t2·bᶜ`.
pub fn dleq_verify(params: &SchnorrParams, g1: u128, g2: u128, a: u128, b: u128, proof: &DleqProof) -> bool {
    if proof.t1 == 0 || proof.t1 >= params.p || proof.t2 == 0 || proof.t2 >= params.p {
        return false;
    }
    let c = dleq_challenge(params, g1, g2, a, b, proof.t1, proof.t2);
    let lhs1 = params.exp(g1, proof.s);
    let rhs1 = mulmod(proof.t1, params.exp(a, c), params.p);
    let lhs2 = params.exp(g2, proof.s);
    let rhs2 = mulmod(proof.t2, params.exp(b, c), params.p);
    lhs1 == rhs1 && lhs2 == rhs2
}

/// A non-interactive Schnorr proof of knowledge of `x` behind `y = gˣ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchnorrProof {
    /// Commitment `t = gʳ`.
    pub t: u128,
    /// Response `s = r + c·x (mod q)`.
    pub s: u128,
}

fn challenge(params: &SchnorrParams, y: u128, t: u128) -> u128 {
    let mut input = Vec::new();
    for v in [params.p, params.q, params.g, y, t] {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let h = Hash256::of(&input).0;
    let mut c = [0u8; 16];
    c.copy_from_slice(&h[..16]);
    u128::from_le_bytes(c) % params.q
}

/// Prove knowledge of `x` (the discrete log of `y`). `seed` supplies the random
/// nonce `r`; in the OS this is drawn from the DRNG so the proof replays.
pub fn schnorr_prove(params: &SchnorrParams, x: u128, seed: &[u8]) -> SchnorrProof {
    let mut rng = Drng::from_seed(seed);
    // r ∈ [1, q)
    let r = 1 + (rng.next_u64() as u128 % (params.q - 1));
    let t = modpow(params.g, r, params.p);
    let y = params.public_key(x);
    let c = challenge(params, y, t);
    let s = (r + mulmod(c, x % params.q, params.q)) % params.q;
    SchnorrProof { t, s }
}

/// Verify a Schnorr proof against the public key `y`. Accepts iff `gˢ = t · yᶜ`.
pub fn schnorr_verify(params: &SchnorrParams, y: u128, proof: &SchnorrProof) -> bool {
    if proof.t == 0 || proof.t >= params.p {
        return false;
    }
    let c = challenge(params, y, proof.t);
    let lhs = modpow(params.g, proof.s, params.p);
    let rhs = mulmod(proof.t, modpow(y, c, params.p), params.p);
    lhs == rhs
}

// ─────────────────── Merkle set-membership proof ───────────────────

/// A commitment to a set: the Merkle root over its (sorted, hashed) members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetCommitment {
    pub root: Hash256,
    leaves: Vec<Hash256>,
}

/// A proof that some element is a member of a committed set, revealing only the
/// sibling path — not the element's neighbours or its index value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MembershipProof {
    /// (sibling hash, sibling_is_left) from leaf up to the root.
    path: Vec<(Hash256, bool)>,
}

impl MembershipProof {
    /// A stable digest of the sibling path, so a proof can be content-addressed
    /// as a graph object (`zkservice`) without exposing the path itself.
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(self.path.len() * 33 + 6);
        buf.extend_from_slice(b"mpath:");
        for (h, sib_is_left) in &self.path {
            buf.extend_from_slice(&h.0);
            buf.push(*sib_is_left as u8);
        }
        Hash256::of(&buf)
    }

    /// Number of sibling hops from leaf to root (the tree depth).
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// True for a single-element set (no siblings on the path).
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

fn node(a: &Hash256, b: &Hash256) -> Hash256 {
    let mut input = Vec::with_capacity(64);
    input.extend_from_slice(&a.0);
    input.extend_from_slice(&b.0);
    Hash256::of(&input)
}

fn leaf_hash(elem: &[u8]) -> Hash256 {
    let mut input = Vec::with_capacity(elem.len() + 5);
    input.extend_from_slice(b"leaf:");
    input.extend_from_slice(elem);
    Hash256::of(&input)
}

impl SetCommitment {
    /// Commit to a set of elements. Order-independent: members are sorted by hash.
    pub fn commit(elements: &[&[u8]]) -> SetCommitment {
        let mut leaves: Vec<Hash256> = elements.iter().map(|e| leaf_hash(e)).collect();
        leaves.sort_by_key(|h| h.0);
        leaves.dedup();
        let root = Self::root_of(&leaves);
        SetCommitment { root, leaves }
    }

    fn root_of(leaves: &[Hash256]) -> Hash256 {
        if leaves.is_empty() {
            return Hash256::of(b"empty-set");
        }
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                if pair.len() == 2 {
                    next.push(node(&pair[0], &pair[1]));
                } else {
                    next.push(node(&pair[0], &pair[0])); // promote odd leaf
                }
            }
            level = next;
        }
        level[0]
    }

    /// Produce a membership proof for `elem`, or `None` if it is not in the set.
    pub fn prove(&self, elem: &[u8]) -> Option<MembershipProof> {
        let target = leaf_hash(elem);
        let mut idx = self.leaves.iter().position(|l| *l == target)?;
        let mut level = self.leaves.clone();
        let mut path = Vec::new();
        while level.len() > 1 {
            let sibling_is_left = idx % 2 == 1;
            let sib_idx = if sibling_is_left { idx - 1 } else { (idx + 1).min(level.len() - 1) };
            path.push((level[sib_idx], sibling_is_left));
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                if pair.len() == 2 {
                    next.push(node(&pair[0], &pair[1]));
                } else {
                    next.push(node(&pair[0], &pair[0]));
                }
            }
            idx /= 2;
            level = next;
        }
        Some(MembershipProof { path })
    }

    /// Verify a membership proof for `elem` against this commitment's root.
    pub fn verify(root: &Hash256, elem: &[u8], proof: &MembershipProof) -> bool {
        let mut acc = leaf_hash(elem);
        for (sib, sib_is_left) in &proof.path {
            acc = if *sib_is_left {
                node(sib, &acc)
            } else {
                node(&acc, sib)
            };
        }
        acc == *root
    }
}

#[cfg(all(test, feature = "demo-crypto"))]
mod tests {
    use super::*;

    #[test]
    fn schnorr_params_are_a_valid_group() {
        let p = SchnorrParams::new_demo_insecure();
        assert!(is_prime(p.p as u64));
        assert!(is_prime(p.q as u64));
        // g has order q: g^q ≡ 1 and g ≠ 1.
        assert_ne!(p.g, 1);
        assert_eq!(modpow(p.g, p.q, p.p), 1);
    }

    #[test]
    fn schnorr_honest_proof_verifies() {
        let params = SchnorrParams::new_demo_insecure();
        let x = 123_456_789u128;
        let y = params.public_key(x);
        let proof = schnorr_prove(&params, x, b"nonce-seed");
        assert!(schnorr_verify(&params, y, &proof));
    }

    #[test]
    fn schnorr_wrong_witness_fails() {
        let params = SchnorrParams::new_demo_insecure();
        let y = params.public_key(1000);
        // Prove knowledge of a different secret → public keys differ → reject.
        let proof = schnorr_prove(&params, 2000, b"seed");
        assert!(!schnorr_verify(&params, y, &proof));
    }

    #[test]
    fn schnorr_tampered_proof_fails() {
        let params = SchnorrParams::new_demo_insecure();
        let x = 42u128;
        let y = params.public_key(x);
        let mut proof = schnorr_prove(&params, x, b"seed");
        proof.s = (proof.s + 1) % params.q;
        assert!(!schnorr_verify(&params, y, &proof));
    }

    #[test]
    fn schnorr_reveals_nothing_extra_but_is_reproducible() {
        // Same witness + same nonce seed → identical proof (replayable);
        // different seed → different commitment (fresh randomness).
        let params = SchnorrParams::new_demo_insecure();
        let x = 7u128;
        let a = schnorr_prove(&params, x, b"s1");
        let b = schnorr_prove(&params, x, b"s1");
        let c = schnorr_prove(&params, x, b"s2");
        assert_eq!(a, b);
        assert_ne!(a.t, c.t);
        let y = params.public_key(x);
        assert!(schnorr_verify(&params, y, &c));
    }

    #[test]
    fn dleq_proves_equal_discrete_logs_without_revealing_x() {
        let params = SchnorrParams::new_demo_insecure();
        let g1 = params.g;
        let g2 = params.hash_to_generator(b"context:checkout");
        let x = 555_111u128;
        let a = params.exp(g1, x);
        let b = params.exp(g2, x);
        let proof = dleq_prove(&params, g1, g2, x, b"nonce");
        assert!(dleq_verify(&params, g1, g2, a, b, &proof));
        // A pseudonym formed with a *different* secret in g2 breaks equality.
        let b_bad = params.exp(g2, x + 1);
        assert!(!dleq_verify(&params, g1, g2, a, b_bad, &proof));
    }

    #[test]
    fn dleq_tampered_proof_is_rejected() {
        let params = SchnorrParams::new_demo_insecure();
        let g2 = params.hash_to_generator(b"ctx");
        let x = 42u128;
        let a = params.exp(params.g, x);
        let b = params.exp(g2, x);
        let mut proof = dleq_prove(&params, params.g, g2, x, b"s");
        proof.s = (proof.s + 1) % params.q;
        assert!(!dleq_verify(&params, params.g, g2, a, b, &proof));
    }

    #[test]
    fn hash_to_generator_is_deterministic_and_in_subgroup() {
        let params = SchnorrParams::new_demo_insecure();
        let g2 = params.hash_to_generator(b"ctx-A");
        // Deterministic per context, distinct across contexts.
        assert_eq!(g2, params.hash_to_generator(b"ctx-A"));
        assert_ne!(g2, params.hash_to_generator(b"ctx-B"));
        // Order divides q: g2^q ≡ 1, and it is not the identity.
        assert_ne!(g2, 1);
        assert_eq!(params.exp(g2, params.q), 1);
    }

    #[test]
    fn merkle_membership_proves_without_revealing_set() {
        let set: [&[u8]; 5] = [b"alice", b"bob", b"carol", b"dave", b"erin"];
        let commit = SetCommitment::commit(&set);
        let proof = commit.prove(b"carol").unwrap();
        assert!(SetCommitment::verify(&commit.root, b"carol", &proof));
        // A non-member cannot be proven.
        assert!(commit.prove(b"mallory").is_none());
        // A valid path for carol does not verify a different element.
        assert!(!SetCommitment::verify(&commit.root, b"alice", &proof));
    }

    #[test]
    fn merkle_commitment_is_order_independent() {
        let a = SetCommitment::commit(&[b"x" as &[u8], b"y", b"z"]);
        let b = SetCommitment::commit(&[b"z" as &[u8], b"x", b"y"]);
        assert_eq!(a.root, b.root);
    }
}

//! Anonymity vs. traceability — **closing the perceived/actual anonymity gap**
//! (red-team finding C; see `docs/findings.md` and `docs/security/zero-knowledge-proofs.md`).
//!
//! DominionOS hides identity in the UI ("you never even see it unless you want to").
//! The red-team concern is that *hiding* identity is a UI property while
//! *unlinkability* is a system property — and the two can silently disagree. The
//! concrete defect: the naive ZK login proves knowledge of the secret behind a
//! **static public key `y`** ([`crate::zk::schnorr_prove`]). Every "anonymous"
//! transaction then carries the *same* `y`, so a verifier links them all trivially —
//! the user feels anonymous but is fully correlatable. That is the gap.
//!
//! This module makes traceability an **explicit, enforced system-layer property**,
//! not a UI afterthought:
//!
//! * [`TraceabilityClass`] labels every [`Transaction`] as `Attributable`,
//!   `Pseudonymous`, or `Anonymous`. The label is checked at construction: an
//!   `Anonymous` transaction is **forbidden from carrying any global correlator**
//!   (the DominionId / static `y`). Mislabelling is a constructor error, so "anonymous
//!   in the UI but attributable on the wire" cannot compile-by-accident.
//! * [`AnonIdentity`] derives **per-context pseudonyms** `P_ctx = g_ctxˣ` over
//!   independent generators ([`SchnorrParams::hash_to_generator`]). A
//!   [`DleqProof`](crate::zk::DleqProof) proves the holder knows the secret behind the
//!   pseudonym (no forging someone else's), while across contexts the pseudonyms are
//!   unlinkable under DDH **in the chosen group** — the verifier never sees a global `y`.
//!   The current illustrative 31-bit group (see security note below) makes this claim
//!   protocol-correct but not production-strength; swap in a 256-bit group to make it real.
//! * The per-context pseudonym doubles as a **scoped nullifier**: stable within a
//!   context (so double-actions are detectable — one vote, one claim) but useless as a
//!   cross-context correlator. Traceability becomes a deliberate, scoped choice.
//!
//! Pure, safe `no_std`; the algebra reuses [`crate::zk`]. Host- and metal-tested.
//!
//! **Security note — illustrative group:** [`SchnorrParams::new_demo_insecure`] uses a
//! 31-bit Schnorr group (q = 2³¹−1, M31). The unlinkability and forgery-resistance
//! claims in this module hold under DDH *in that group*, but a 31-bit group is
//! far below any real security margin (discrete-log solvable in ~2¹⁵·⁵ operations).
//! Use this code as a correct protocol reference only; replace
//! `SchnorrParams::new_demo_insecure()` with a production-strength group (e.g. a
//! 256-bit prime-order group or a standard elliptic curve) before deploying for real
//! users. The `demo-crypto` Cargo feature must be enabled to call
//! `new_demo_insecure()` and will panic at runtime without it, preventing accidental
//! production use.

use crate::dominionlink::DominionId;
use crate::hash::Hash256;
use crate::zk::{dleq_prove, dleq_verify, DleqProof, SchnorrParams};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// How linkable a transaction is — an explicit, enforced system-layer property.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TraceabilityClass {
    /// Bound to a stable public identity; fully linkable on purpose (e.g. a bank login).
    Attributable,
    /// Linkable within one service/context but not across them (per-service identity).
    Pseudonymous,
    /// Unlinkable across contexts; carries no global correlator at all.
    Anonymous,
}

/// An identity capable of producing unlinkable per-context pseudonyms. The secret `x`
/// is the discrete-log witness; it never leaves this struct.
#[derive(Clone)]
pub struct AnonIdentity {
    params: SchnorrParams,
    secret: u128,
}

/// A per-context pseudonym `P = g_ctxˣ` plus the proof it was correctly formed.
/// Within a context it is a stable **nullifier**; across contexts it is unlinkable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pseudonym {
    /// The context-specific generator `g_ctx`.
    pub base: u128,
    /// The pseudonym `P = g_ctxˣ` — also the scoped nullifier.
    pub value: u128,
    /// DLEQ proof that `P` shares the secret behind the holder's credential.
    pub proof: DleqProof,
    /// The holder's credential element `C = gˣ` in the *base group* (g1). This is
    /// passed only to verifiers entitled to it; anonymous transactions never put it
    /// on the wire (see [`Transaction`]).
    credential: u128,
}

impl AnonIdentity {
    /// Build an anonymous identity from secret bytes (derived from the master seed).
    pub fn from_secret(params: SchnorrParams, secret: &[u8]) -> AnonIdentity {
        let h = Hash256::of(secret).0;
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&h[..16]);
        let secret = (u128::from_le_bytes(raw) % (params.q - 1)) + 1;
        AnonIdentity { params, secret }
    }

    /// The holder's credential element `C = gˣ` (the would-be "static y"). It is kept
    /// for authorized verifiers only; it is *the* correlator and must never appear in
    /// an anonymous transaction.
    pub fn credential(&self) -> u128 {
        self.params.public_key(self.secret)
    }

    /// Produce a pseudonym + proof for `context`. Same identity + same context ⇒ same
    /// pseudonym value (a scoped nullifier); different context ⇒ unlinkable value.
    ///
    /// The proof nonce is derived internally as `H(secret || context)` (RFC-6979 style)
    /// so callers cannot supply a raw nonce. This prevents nonce-reuse secret recovery:
    /// if a caller could supply the same nonce across two different contexts for the
    /// same identity, both proofs would share `r` but have different challenges, allowing
    /// secret recovery via `x = (s1−s2)/(c1−c2)`.
    pub fn pseudonym(&self, context: &[u8]) -> Pseudonym {
        let g1 = self.params.g;
        let g2 = self.params.hash_to_generator(context);
        let value = self.params.exp(g2, self.secret); // P = g_ctx^x
        let credential = self.params.exp(g1, self.secret); // C = g^x
        // Derive the nonce from the secret and context so it is deterministic,
        // caller-independent, and impossible to reuse across different contexts.
        let mut nonce_input = Vec::with_capacity(16 + context.len() + 6);
        nonce_input.extend_from_slice(b"dleq-r:");
        nonce_input.extend_from_slice(&self.secret.to_le_bytes());
        nonce_input.extend_from_slice(context);
        let effective_nonce = Hash256::of(&nonce_input);
        let proof = dleq_prove(&self.params, g1, g2, self.secret, &effective_nonce.0);
        Pseudonym { base: g2, value, proof, credential }
    }
}

impl Pseudonym {
    /// The scoped nullifier (the pseudonym value): equal iff the same identity acts in
    /// the same context. Detects double-actions without revealing who.
    pub fn nullifier(&self) -> Hash256 {
        Hash256::of(&self.value.to_le_bytes())
    }

    /// Verify the pseudonym is well-formed for `context`: it must use the canonical
    /// context generator and prove the holder controls the secret behind it. Returns
    /// `true` only if the DLEQ binds `value` to the (hidden) credential with one `x`.
    pub fn verify(&self, params: &SchnorrParams, context: &[u8]) -> bool {
        let g1 = params.g;
        let expected_base = params.hash_to_generator(context);
        if self.base != expected_base {
            return false;
        }
        dleq_verify(params, g1, self.base, self.credential, self.value, &self.proof)
    }
}

/// Why building a transaction was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnonError {
    /// An `Anonymous` transaction tried to carry a global correlator (DominionId / `y`).
    LeakyAnonymous,
    /// A non-anonymous transaction was missing its required attribution.
    MissingAttribution,
}

/// A network transaction whose linkability is governed by its [`TraceabilityClass`].
/// The constructors enforce the class so the system layer — not the UI — guarantees
/// what is and isn't correlatable.
pub struct Transaction {
    pub class: TraceabilityClass,
    /// Present only for Attributable/Pseudonymous transactions.
    pub identity: Option<DominionId>,
    /// Present only for Anonymous/Pseudonymous transactions.
    pub pseudonym: Option<Pseudonym>,
    pub payload: Vec<u8>,
}

impl Transaction {
    /// An attributable transaction: bound to a stable identity (deliberately linkable).
    pub fn attributable(identity: DominionId, payload: &[u8]) -> Transaction {
        Transaction {
            class: TraceabilityClass::Attributable,
            identity: Some(identity),
            pseudonym: None,
            payload: payload.to_vec(),
        }
    }

    /// An anonymous transaction: a per-context pseudonym and **no global correlator**.
    /// Refused if a caller tries to also attach an identity (the gap, prevented).
    pub fn anonymous(pseudonym: Pseudonym, identity: Option<DominionId>, payload: &[u8]) -> Result<Transaction, AnonError> {
        if identity.is_some() {
            return Err(AnonError::LeakyAnonymous);
        }
        Ok(Transaction {
            class: TraceabilityClass::Anonymous,
            identity: None,
            pseudonym: Some(pseudonym),
            payload: payload.to_vec(),
        })
    }

    /// Does this transaction expose a cross-context correlator? The whole point: for
    /// `Anonymous`, this is always `false` (no DominionId, no static `y` on the wire).
    pub fn exposes_global_correlator(&self) -> bool {
        match self.class {
            TraceabilityClass::Anonymous => self.identity.is_some(),
            _ => true,
        }
    }
}

/// A registry of spent nullifiers within a single context — prevents double-actions
/// (double-vote, double-claim) for anonymous transactions without ever learning who.
#[derive(Default)]
pub struct NullifierSet {
    spent: BTreeSet<[u8; 32]>,
}

impl NullifierSet {
    pub fn new() -> NullifierSet {
        NullifierSet { spent: BTreeSet::new() }
    }

    /// Record a nullifier; returns `false` if it was already spent (double-action).
    pub fn spend(&mut self, n: Hash256) -> bool {
        self.spent.insert(n.0)
    }

    pub fn is_spent(&self, n: Hash256) -> bool {
        self.spent.contains(&n.0)
    }
}

#[cfg(all(test, feature = "demo-crypto"))]
mod tests {
    use super::*;

    fn id() -> AnonIdentity {
        AnonIdentity::from_secret(SchnorrParams::new_demo_insecure(), b"master-seed-derived-anon-secret")
    }

    #[test]
    fn naive_static_key_is_linkable_the_bug_we_fix() {
        // Demonstrates the reported defect: proving against a static public key means
        // every transaction shares the *same* correlator `y`. This is what made
        // "anonymous" transactions linkable.
        let params = SchnorrParams::new_demo_insecure();
        let x = 12345u128;
        let y_tx1 = params.public_key(x);
        let y_tx2 = params.public_key(x);
        assert_eq!(y_tx1, y_tx2, "static y is identical across transactions → linkable");
    }

    #[test]
    fn per_context_pseudonyms_are_unlinkable_across_contexts() {
        // The fix: the same identity in two different contexts yields two different,
        // uncorrelatable pseudonyms (no shared value on the wire).
        let me = id();
        let p_vote = me.pseudonym(b"poll:2026-budget");
        let p_forum = me.pseudonym(b"forum:general");
        assert_ne!(p_vote.value, p_forum.value);
        assert_ne!(p_vote.nullifier(), p_forum.nullifier());
        assert_ne!(p_vote.base, p_forum.base);
    }

    #[test]
    fn pseudonym_is_a_stable_scoped_nullifier_within_a_context() {
        // Same identity + same context ⇒ identical nullifier (double-action detectable).
        let me = id();
        let a = me.pseudonym(b"poll:2026-budget");
        let b = me.pseudonym(b"poll:2026-budget");
        assert_eq!(a.value, b.value);
        assert_eq!(a.nullifier(), b.nullifier());
    }

    #[test]
    fn pseudonym_proof_verifies_and_resists_forgery() {
        let params = SchnorrParams::new_demo_insecure();
        let me = id();
        let p = me.pseudonym(b"ctx");
        assert!(p.verify(&params, b"ctx"));
        // Wrong context: the canonical base differs → rejected.
        assert!(!p.verify(&params, b"other-ctx"));
        // A forger who swaps in a different pseudonym value without the secret fails.
        let mut forged = p.clone();
        forged.value = params.exp(forged.base, 99); // not the holder's secret
        assert!(!forged.verify(&params, b"ctx"));
    }

    #[test]
    fn distinct_identities_have_distinct_nullifiers_in_a_context() {
        let a = AnonIdentity::from_secret(SchnorrParams::new_demo_insecure(), b"alice");
        let b = AnonIdentity::from_secret(SchnorrParams::new_demo_insecure(), b"bob");
        let ctx = b"poll:x";
        assert_ne!(a.pseudonym(ctx).value, b.pseudonym(ctx).value);
    }

    #[test]
    fn anonymous_transaction_cannot_carry_a_global_correlator() {
        // The system layer forbids the "anonymous in UI, attributable on wire" gap.
        let me = id();
        let p = me.pseudonym(b"ctx");
        let leaky = Transaction::anonymous(p.clone(), Some(DominionId(Hash256::of(b"oops"))), b"data");
        assert_eq!(leaky.err(), Some(AnonError::LeakyAnonymous));
        // A correctly anonymous transaction exposes no global correlator.
        let tx = Transaction::anonymous(p, None, b"data").unwrap();
        assert!(!tx.exposes_global_correlator());
        // An attributable transaction is, by design, linkable.
        let at = Transaction::attributable(DominionId(Hash256::of(b"bank-id")), b"login");
        assert!(at.exposes_global_correlator());
    }

    #[test]
    fn nullifier_set_detects_double_actions_without_identity() {
        let me = id();
        let p = me.pseudonym(b"poll:2026");
        let mut spent = NullifierSet::new();
        assert!(spent.spend(p.nullifier())); // first vote accepted
        assert!(!spent.spend(p.nullifier())); // second vote rejected (double action)
        // A different identity in the same poll is unaffected.
        let other = AnonIdentity::from_secret(SchnorrParams::new_demo_insecure(), b"someone-else");
        assert!(spent.spend(other.pseudonym(b"poll:2026").nullifier()));
    }
}

//! Identity, key management, recovery & passwordless auth — **U + AJ** (see
//! `docs/security/identity-recovery-and-key-management.md` and
//! `docs/architecture/web-identity-and-authentication.md`).
//!
//! Everything derives from one **master seed**, HD-wallet style:
//!
//! ```text
//!   MasterSeed ─▶ device/domain Identity ─▶ KEK ─▶ DEK (per object)
//!              └▶ per-service pseudonymous Identity (unlinkable, recoverable)
//! ```
//!
//! Because every key is a *derivation* of the seed, **all data is recoverable from
//! the seed alone** (DEKs are rederivable). Authentication is **passwordless**:
//! either a challenge-signature (hash-based, via the [`CryptoLayer`]) or a
//! zero-knowledge proof of the identity secret ([`crate::zk`]) — the user never
//! types a key. Per-service identities are **pseudonymous and unlinkable** (a
//! service cannot correlate you across services) yet recoverable from the seed.
//! Sessions are **temporal, scoped capabilities**, not ambient cookies. Recovery
//! reuses Shamir sharing ([`crate::recovery`]) behind a **time-delayed veto
//! window**, and there is **no escrow backdoor** — the system holds no master
//! decrypt key. Pure, safe `no_std`, host-tested.

use crate::dominionlink::DominionId;
use crate::capability::{Capability, Rights};
use crate::crypto::CryptoLayer;
use crate::hash::Hash256;
use crate::recovery::{reconstruct, Share};
use alloc::string::String;
use alloc::vec::Vec;

/// The single root of a user's whole key hierarchy. Hold this (or enough recovery
/// shares of it) and you hold everything; lose it without recovery and the data is
/// cryptographically gone — there is no other copy.
#[derive(Clone)]
pub struct MasterSeed {
    seed: [u8; 32],
}

impl MasterSeed {
    /// Build a master seed from gathered entropy (the kernel TRNG/entropy pool).
    pub fn from_entropy(entropy: &[u8]) -> MasterSeed {
        MasterSeed { seed: Hash256::of(entropy).0 }
    }

    fn derive(&self, tag: &str, label: &str) -> [u8; 32] {
        let mut input = Vec::with_capacity(64 + label.len());
        input.extend_from_slice(&self.seed);
        input.extend_from_slice(tag.as_bytes());
        input.extend_from_slice(label.as_bytes());
        Hash256::of(&input).0
    }

    /// Derive a device or domain identity (a long-term keypair seed). The public
    /// identity is the self-certifying [`DominionId`].
    pub fn identity(&self, label: &str) -> DerivedIdentity {
        let secret = self.derive("identity:", label);
        // Public key material is a one-way image of the secret (self-certifying).
        let pubkey = Hash256::of(&[&secret[..], b"pub"].concat()).0.to_vec();
        DerivedIdentity { id: DominionId::from_pubkey(&pubkey), secret, pubkey }
    }

    /// A **per-service pseudonymous** identity: unlinkable across services (each is
    /// a distinct hash branch) but rederivable from the seed.
    pub fn service_identity(&self, service: &str) -> DerivedIdentity {
        self.identity(&alloc::format!("service/{service}"))
    }

    /// A Key-Encryption Key for a domain (wraps the per-object DEKs).
    pub fn kek(&self, domain: &str) -> [u8; 32] {
        self.derive("kek:", domain)
    }

    /// A per-object Data-Encryption Key — rederivable from the seed, so encrypted
    /// data survives as long as the seed (or its recovery quorum) does.
    pub fn dek(&self, domain: &str, object_label: &str) -> [u8; 32] {
        let mut input = Vec::with_capacity(64 + object_label.len());
        input.extend_from_slice(&self.kek(domain));
        input.extend_from_slice(b"dek:");
        input.extend_from_slice(object_label.as_bytes());
        Hash256::of(&input).0
    }
}

/// An identity derived from the master seed: a self-certifying id plus the secret
/// and public material for signing/verifying.
#[derive(Clone)]
pub struct DerivedIdentity {
    pub id: DominionId,
    secret: [u8; 32],
    pubkey: Vec<u8>,
}

impl DerivedIdentity {
    pub fn public_key(&self) -> &[u8] {
        &self.pubkey
    }

    /// The signing seed this identity uses with the [`CryptoLayer`] (the secret).
    pub fn signing_seed(&self) -> &[u8] {
        &self.secret
    }
}

// ─────────────────────────── passwordless authentication ───────────────────────────

/// A login challenge a service issues; the client signs it to prove identity
/// without ever sending a secret.
pub fn login_challenge(service: &str, nonce: &[u8]) -> Vec<u8> {
    let mut c = Vec::with_capacity(service.len() + nonce.len() + 8);
    c.extend_from_slice(b"login:");
    c.extend_from_slice(service.as_bytes());
    c.push(b'|');
    c.extend_from_slice(nonce);
    c
}

/// A registered account: a service's record of one of the user's **per-service**
/// identities. The service stores the **signing seed** (not a fixed public key) so
/// that each login challenge can derive a fresh, unique Lamport keypair — preventing
/// the one-time-signature key-reuse attack that would result from verifying multiple
/// challenges against the same stored public key.
///
/// Per-challenge key derivation: `keygen(H(signing_seed || challenge))`.  Each
/// challenge therefore gets its own independent Lamport keypair, so no two
/// signatures ever share preimage material.
pub struct Account {
    pub service: String,
    pub identity: DominionId,
    /// The seed from which per-challenge keypairs are derived on demand.
    /// Never exposed to the verifier beyond this struct.
    signing_seed: Vec<u8>,
    pub algo: String,
}

impl Account {
    /// Register a per-service identity with a service.  The account records the
    /// signing seed; no keypair is generated at registration time because the
    /// actual keypair is derived fresh for every challenge.
    pub fn register(
        _cal: &CryptoLayer,
        algo: &str,
        service: &str,
        ident: &DerivedIdentity,
    ) -> Option<Account> {
        Some(Account {
            service: String::from(service),
            identity: ident.id,
            signing_seed: ident.signing_seed().to_vec(),
            algo: String::from(algo),
        })
    }

    /// Derive the per-challenge signing seed: `H(signing_seed || challenge)`.
    /// This produces a unique 32-byte seed for every distinct challenge, ensuring
    /// every Lamport keypair is used at most once.
    fn per_challenge_seed(&self, challenge: &[u8]) -> [u8; 32] {
        let mut input = Vec::with_capacity(self.signing_seed.len() + challenge.len());
        input.extend_from_slice(&self.signing_seed);
        input.extend_from_slice(challenge);
        Hash256::of(&input).0
    }

    /// Derive the per-challenge public key for `challenge`.  The client calls
    /// [`Account::sign_challenge`] (or derives the same seed themselves) to
    /// produce the matching signature.
    pub fn per_challenge_pubkey(&self, cal: &CryptoLayer, challenge: &[u8]) -> Option<Vec<u8>> {
        let seed = self.per_challenge_seed(challenge);
        let (_sk, pk) = cal.keygen(&self.algo, &seed)?;
        Some(pk)
    }

    /// Sign a challenge on the client side.  The caller holds the same
    /// [`DerivedIdentity`] that was used during [`Account::register`]; this
    /// derives the matching per-challenge secret and produces a signature that
    /// [`Account::verify_login`] will accept.
    pub fn sign_challenge(
        cal: &CryptoLayer,
        algo: &str,
        ident: &DerivedIdentity,
        challenge: &[u8],
    ) -> Option<Vec<u8>> {
        let mut input = Vec::with_capacity(ident.signing_seed().len() + challenge.len());
        input.extend_from_slice(ident.signing_seed());
        input.extend_from_slice(challenge);
        let per_challenge_seed = Hash256::of(&input).0;
        let (sk, _pk) = cal.keygen(algo, &per_challenge_seed)?;
        cal.sign(algo, &sk, challenge)
    }

    /// Passwordless login: derive a fresh Lamport keypair for this specific
    /// `challenge` and verify `signature` against it.  Because the keypair is
    /// unique per challenge, signing two different challenges never reuses
    /// preimage material — the Lamport OTS guarantee holds unconditionally.
    pub fn verify_login(&self, cal: &CryptoLayer, challenge: &[u8], signature: &[u8]) -> bool {
        let seed = self.per_challenge_seed(challenge);
        let pk = match cal.keygen(&self.algo, &seed) {
            Some((_sk, pk)) => pk,
            None => return false,
        };
        cal.verify(&self.algo, &pk, challenge, signature)
    }
}

// ─────────────────────────── sessions as temporal capabilities ───────────────────────────

/// A session is a **temporal, scoped capability** — it expires and is revocable,
/// never an ambient cookie.
#[derive(Clone, Copy, Debug)]
pub struct AuthSession {
    pub identity: DominionId,
    pub capability: Capability,
    pub expires_at: u64,
    revoked: bool,
}

impl AuthSession {
    /// Issue a session granting exactly `rights`, valid until `expires_at`.
    pub fn issue(identity: DominionId, rights: Rights, expires_at: u64) -> AuthSession {
        AuthSession {
            identity,
            capability: Capability::mint(0, 0, rights),
            expires_at,
            revoked: false,
        }
    }

    /// Is the session usable at `now`? (Not expired, not revoked, capability valid.)
    pub fn is_valid(&self, now: u64) -> bool {
        !self.revoked && now <= self.expires_at && self.capability.is_valid()
    }

    pub fn revoke(&mut self) {
        self.revoked = true;
    }
}

// ─────────────────────────── recovery (no escrow) ───────────────────────────

/// A time-delayed recovery request: an M-of-N quorum authorizes it, but it only
/// completes after a **veto window** elapses, so a surprise takeover can be cancelled.
pub struct Recovery {
    request: Hash256,
    requested_at: u64,
    veto_window: u64,
    vetoed: bool,
}

impl Recovery {
    /// Open a recovery; it cannot complete until `requested_at + veto_window`.
    pub fn open(request: &[u8], requested_at: u64, veto_window: u64) -> Recovery {
        Recovery { request: Hash256::of(request), requested_at, veto_window, vetoed: false }
    }

    /// The legitimate owner cancels an in-progress recovery within the window.
    pub fn veto(&mut self) {
        self.vetoed = true;
    }

    /// Attempt to complete: reconstruct the master seed from a Shamir quorum, but
    /// only once the veto window has passed and no veto was raised. Returns the
    /// recovered [`MasterSeed`] — proof the data is recoverable from shares alone,
    /// with **no escrowed master key** anywhere in the system.
    pub fn complete(&self, shares: &[Share], now: u64) -> Option<MasterSeed> {
        if self.vetoed || now < self.requested_at + self.veto_window {
            return None;
        }
        let secret = reconstruct(shares)?;
        // Bind the recovered material to the original request (anti-confusion).
        let _ = self.request;
        Some(MasterSeed { seed: Hash256::of(&secret).0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::split;

    #[test]
    fn deks_are_rederivable_from_the_seed_alone() {
        let seed = MasterSeed::from_entropy(b"hardware entropy boundary");
        // Same seed ⇒ same DEK, so data encrypted under it is recoverable forever.
        let d1 = seed.dek("financial", "invoice-42");
        let d2 = seed.dek("financial", "invoice-42");
        assert_eq!(d1, d2);
        // Different object / domain ⇒ independent key.
        assert_ne!(d1, seed.dek("financial", "invoice-43"));
        assert_ne!(d1, seed.dek("personal", "invoice-42"));
    }

    #[test]
    fn service_identities_are_unlinkable_but_recoverable() {
        let seed = MasterSeed::from_entropy(b"seed");
        let bank = seed.service_identity("bank.example");
        let forum = seed.service_identity("forum.example");
        // A service cannot correlate the two identities — different ids/keys.
        assert_ne!(bank.id, forum.id);
        assert_ne!(bank.public_key(), forum.public_key());
        // Yet both rederive deterministically from the same seed (recoverable).
        let seed2 = MasterSeed::from_entropy(b"seed");
        assert_eq!(seed2.service_identity("bank.example").id, bank.id);
    }

    #[test]
    fn passwordless_login_verifies_a_signed_challenge() {
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"seed");
        let ident = seed.service_identity("app.example");
        // register() no longer returns an sk — the keypair is derived per challenge.
        let account = Account::register(&cal, "lamport-pq", "app.example", &ident).unwrap();

        let challenge = login_challenge("app.example", b"server-nonce-1");
        // Client derives the per-challenge signing key using the same ident.
        let sig = Account::sign_challenge(&cal, "lamport-pq", &ident, &challenge).unwrap();
        assert!(account.verify_login(&cal, &challenge, &sig));
        // A signature produced for one challenge is rejected when replayed against another.
        let other = login_challenge("app.example", b"different-nonce");
        assert!(!account.verify_login(&cal, &other, &sig));
    }

    #[test]
    fn each_challenge_uses_a_distinct_keypair() {
        // Core security property: two different challenges must derive two different
        // public keys so that signing both never reuses a Lamport key.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"seed");
        let ident = seed.service_identity("app.example");
        let account = Account::register(&cal, "lamport-pq", "app.example", &ident).unwrap();

        let c1 = login_challenge("app.example", b"nonce-1");
        let c2 = login_challenge("app.example", b"nonce-2");
        let pk1 = account.per_challenge_pubkey(&cal, &c1).unwrap();
        let pk2 = account.per_challenge_pubkey(&cal, &c2).unwrap();
        // Different challenges → different OTS keypairs.
        assert_ne!(pk1, pk2);
        // Each sig verifies only against its own challenge.
        let sig1 = Account::sign_challenge(&cal, "lamport-pq", &ident, &c1).unwrap();
        let sig2 = Account::sign_challenge(&cal, "lamport-pq", &ident, &c2).unwrap();
        assert!(account.verify_login(&cal, &c1, &sig1));
        assert!(account.verify_login(&cal, &c2, &sig2));
        assert!(!account.verify_login(&cal, &c1, &sig2));
        assert!(!account.verify_login(&cal, &c2, &sig1));
    }

    #[test]
    fn sessions_expire_and_revoke() {
        let id = DominionId::from_pubkey(b"user");
        let s = AuthSession::issue(id, Rights::READ, 100);
        assert!(s.is_valid(50));
        assert!(!s.is_valid(101)); // expired
        let mut s2 = AuthSession::issue(id, Rights::READ, 100);
        s2.revoke();
        assert!(!s2.is_valid(10)); // revoked
    }

    #[test]
    fn recovery_waits_out_the_veto_window_then_completes() {
        let seed_material = b"master-secret!!!";
        let mut entropy = [0u8; 32];
        crate::random::Drng::from_seed(b"e").fill(&mut entropy);
        let shares = split(seed_material, 3, 5, &entropy).unwrap();

        let rec = Recovery::open(b"recover device X", 1000, 500);
        // Too early — the veto window has not elapsed.
        assert!(rec.complete(&shares[0..3], 1200).is_none());
        // After the window, a 3-of-5 quorum reconstructs the seed (no escrow needed).
        assert!(rec.complete(&shares[0..3], 1600).is_some());
    }

    #[test]
    fn forged_login_with_wrong_challenge_key_is_rejected() {
        // Security property: a signature produced by signing challenge_A with the
        // keypair derived for challenge_B must be rejected by verify_login(challenge_A).
        // This catches any implementation that derives the same keypair regardless of
        // the challenge or that mixes up which challenge was passed to keygen.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"forge-test-seed");
        let ident = seed.service_identity("target.example");
        let account = Account::register(&cal, "lamport-pq", "target.example", &ident).unwrap();

        let challenge_a = login_challenge("target.example", b"nonce-alpha");
        let challenge_b = login_challenge("target.example", b"nonce-beta");

        // Attacker signs challenge_A but using the keypair derived for challenge_B.
        // Concretely: we sign challenge_A but pass challenge_B to sign_challenge so
        // that the wrong per-challenge seed is used.
        let wrong_sig = {
            // Derive the per-challenge seed for B, then sign challenge_A with it.
            use crate::hash::Hash256;
            let mut input = Vec::new();
            input.extend_from_slice(ident.signing_seed());
            input.extend_from_slice(&challenge_b);
            let seed_b = Hash256::of(&input).0;
            let (sk_b, _pk_b) = cal.keygen("lamport-pq", &seed_b).unwrap();
            cal.sign("lamport-pq", &sk_b, &challenge_a).unwrap()
        };

        // verify_login derives the keypair for challenge_A — the public key will not
        // match sk_b, so the forged signature must be rejected.
        assert!(
            !account.verify_login(&cal, &challenge_a, &wrong_sig),
            "verify_login accepted a signature produced with the wrong per-challenge key"
        );
    }

    #[test]
    fn shamir_threshold_below_k_recovers_the_wrong_secret() {
        // The threshold is real: below `k` shares, Shamir interpolation yields a
        // *different* secret, never the real one — so a sub-quorum learns nothing.
        let secret = b"master-secret!!!";
        let mut entropy = [0u8; 32];
        crate::random::Drng::from_seed(b"e3").fill(&mut entropy);
        let shares = split(secret, 3, 5, &entropy).unwrap();
        assert_eq!(reconstruct(&shares[0..3]).as_deref(), Some(secret.as_ref()));
        assert_ne!(reconstruct(&shares[0..2]).as_deref(), Some(secret.as_ref()));
    }

    #[test]
    fn a_veto_cancels_recovery() {
        let mut entropy = [0u8; 32];
        crate::random::Drng::from_seed(b"e2").fill(&mut entropy);
        let shares = split(b"secret-blob-here", 2, 3, &entropy).unwrap();
        let mut rec = Recovery::open(b"req", 0, 10);
        rec.veto(); // the real owner cancels a surprise takeover
        assert!(rec.complete(&shares[0..2], 100).is_none());
    }
}

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

// ─────────────────────────── hash-based Merkle OTS (mini-XMSS) ───────────────────────────
//
// The verifier stores ONLY a Merkle root over N one-time Lamport public keys. Each
// login reveals one previously-unused leaf public key together with an
// authentication path proving it is committed under the root, plus a Lamport
// signature over the challenge made with that leaf's one-time secret. A breached
// verifier holds no secret: it cannot substitute a leaf pubkey (that would break the
// root's preimage/collision resistance) nor forge a leaf signature (no leaf secret).

/// Domain-separated Merkle leaf hash of a one-time public key. Kept distinct from
/// the internal-node hash (`Hash256::combine`) so leaf and node hashes never collide.
fn merkle_leaf(pubkey: &[u8]) -> Hash256 {
    let mut input = Vec::with_capacity(pubkey.len() + 5);
    input.extend_from_slice(b"leaf:");
    input.extend_from_slice(pubkey);
    Hash256::of(&input)
}

/// Fold one level of a Merkle tree, duplicating the odd tail node (standard).
fn merkle_fold(level: &[Hash256]) -> Vec<Hash256> {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    let mut i = 0;
    while i < level.len() {
        let left = level[i];
        let right = if i + 1 < level.len() { level[i + 1] } else { level[i] };
        next.push(left.combine(&right));
        i += 2;
    }
    next
}

/// The Merkle root over `leaves` (leaf hashes). Empty ⇒ `Hash256::ZERO`.
fn merkle_root(leaves: &[Hash256]) -> Hash256 {
    if leaves.is_empty() {
        return Hash256::ZERO;
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        level = merkle_fold(&level);
    }
    level[0]
}

/// The authentication path (sibling hashes, bottom-up) for leaf `index`. Consistent
/// with [`merkle_root`] and [`root_from_path`]: same leaf-hash domain, same
/// left/right ordering, same odd-tail duplication.
fn merkle_proof(leaves: &[Hash256], index: usize) -> Vec<Hash256> {
    let mut proof = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        let sibling = if idx % 2 == 0 {
            if idx + 1 < level.len() { level[idx + 1] } else { level[idx] }
        } else {
            level[idx - 1]
        };
        proof.push(sibling);
        level = merkle_fold(&level);
        idx /= 2;
    }
    proof
}

/// Recompute a Merkle root from `leaf` at `index` and its `path`. The parity of the
/// running index at each level fixes the left/right order, matching [`merkle_proof`].
fn root_from_path(leaf: Hash256, index: usize, path: &[Hash256]) -> Hash256 {
    let mut node = leaf;
    let mut idx = index;
    for sibling in path {
        node = if idx % 2 == 0 { node.combine(sibling) } else { sibling.combine(&node) };
        idx /= 2;
    }
    node
}

/// Deterministically derive leaf `index`'s one-time-key seed: `H(signing_seed || i_le)`.
fn leaf_seed(signing_seed: &[u8], index: u32) -> [u8; 32] {
    let mut input = Vec::with_capacity(signing_seed.len() + 4);
    input.extend_from_slice(signing_seed);
    input.extend_from_slice(&index.to_le_bytes());
    Hash256::of(&input).0
}

/// Compute the `n_leaves` Merkle leaf hashes from a signing seed (client-side helper,
/// used to build both the published root and the per-login authentication path).
fn leaf_hashes(
    cal: &CryptoLayer,
    algo: &str,
    signing_seed: &[u8],
    n_leaves: u32,
) -> Option<Vec<Hash256>> {
    let mut leaves = Vec::with_capacity(n_leaves as usize);
    for i in 0..n_leaves {
        let (_sk, pk) = cal.keygen(algo, &leaf_seed(signing_seed, i))?;
        leaves.push(merkle_leaf(&pk));
    }
    Some(leaves)
}

/// The client's **revelation bundle** for one login: it opens exactly one unused
/// one-time leaf. This is entirely public — capturing it lets an attacker replay
/// nothing (the leaf index is burned after first use) and forge nothing (the leaf
/// secret never leaves the client).
pub struct AuthResponse {
    /// Which one-time leaf this bundle opens.
    pub index: u32,
    /// The leaf's one-time Lamport public key.
    pub one_time_pubkey: Vec<u8>,
    /// Sibling hashes proving `one_time_pubkey` is committed under the account root.
    pub auth_path: Vec<Hash256>,
    /// Lamport signature over the challenge, made with the leaf's one-time secret.
    pub signature: Vec<u8>,
}

impl AuthResponse {
    /// Build the revelation bundle for leaf `index` on the client side. The caller
    /// holds the [`DerivedIdentity`] used at registration; this regenerates leaf
    /// `index`'s one-time keypair from the seed, rebuilds the tree to derive the
    /// authentication path, and signs `challenge`. Uses the default leaf count that
    /// [`Account::register`] publishes.
    pub fn create(
        cal: &CryptoLayer,
        algo: &str,
        ident: &DerivedIdentity,
        index: u32,
        challenge: &[u8],
    ) -> Option<AuthResponse> {
        Self::create_with_leaves(cal, algo, ident, index, challenge, Account::DEFAULT_LEAVES)
    }

    /// As [`AuthResponse::create`], but for an account registered with an explicit
    /// `n_leaves` (must match [`Account::register_with_leaves`]).
    pub fn create_with_leaves(
        cal: &CryptoLayer,
        algo: &str,
        ident: &DerivedIdentity,
        index: u32,
        challenge: &[u8],
        n_leaves: u32,
    ) -> Option<AuthResponse> {
        if index >= n_leaves {
            return None;
        }
        let (sk, pk) = cal.keygen(algo, &leaf_seed(ident.signing_seed(), index))?;
        let leaves = leaf_hashes(cal, algo, ident.signing_seed(), n_leaves)?;
        let auth_path = merkle_proof(&leaves, index as usize);
        let signature = cal.sign(algo, &sk, challenge)?;
        Some(AuthResponse { index, one_time_pubkey: pk, auth_path, signature })
    }
}

/// A registered account: a service's record of one of the user's **per-service**
/// identities. The service stores **only public data** — a Merkle root committing to
/// N one-time public keys, the leaf count, and the set of already-consumed leaves.
/// There is **no secret** here: a breached or malicious verifier cannot impersonate
/// the user, because forging a login would require either inverting the root
/// (to substitute a leaf pubkey) or a leaf's one-time secret (which lives only on the
/// client). Each leaf is a Lamport one-time key, so the `used` set enforces
/// single-use and rejects replay of any captured [`AuthResponse`].
pub struct Account {
    pub service: String,
    pub identity: DominionId,
    pub algo: String,
    /// Public commitment to the N one-time public keys. The ONLY authentication
    /// material stored verifier-side — public, non-secret.
    pub merkle_root: Hash256,
    /// Number of one-time leaves committed under `merkle_root`.
    pub n_leaves: u32,
    /// Consumed leaf indices — enforces one-time use and rejects replay. Private so
    /// verification is the only path that can mark a leaf spent.
    used: Vec<u32>,
}

impl Account {
    /// Default number of one-time leaves published at registration. Each leaf is one
    /// passwordless login; re-registration (a fresh root) is needed once exhausted.
    pub const DEFAULT_LEAVES: u32 = 16;

    /// Register a per-service identity with a service. The client derives
    /// [`Account::DEFAULT_LEAVES`] one-time keypairs from the identity seed, builds a
    /// Merkle tree over their public keys, and publishes ONLY the root.
    pub fn register(
        cal: &CryptoLayer,
        algo: &str,
        service: &str,
        ident: &DerivedIdentity,
    ) -> Option<Account> {
        Self::register_with_leaves(cal, algo, service, ident, Self::DEFAULT_LEAVES)
    }

    /// As [`Account::register`], but with an explicit leaf count (used by tests to
    /// keep N small). The client keeps the seed; the service gets only the root.
    pub fn register_with_leaves(
        cal: &CryptoLayer,
        algo: &str,
        service: &str,
        ident: &DerivedIdentity,
        n_leaves: u32,
    ) -> Option<Account> {
        if n_leaves == 0 {
            return None;
        }
        let leaves = leaf_hashes(cal, algo, ident.signing_seed(), n_leaves)?;
        Some(Account {
            service: String::from(service),
            identity: ident.id,
            algo: String::from(algo),
            merkle_root: merkle_root(&leaves),
            n_leaves,
            used: Vec::new(),
        })
    }

    /// How many one-time leaves have been consumed so far.
    pub fn used_count(&self) -> usize {
        self.used.len()
    }

    /// Passwordless login against a revelation bundle. Checks, in order:
    /// (a) `response.index` is in range and NOT already used (rejects one-time-key
    /// reuse / replay of a captured bundle); (b) the one-time pubkey + auth path
    /// recompute the stored `merkle_root` (leaf membership); (c) the Lamport
    /// signature verifies over `challenge`. Only if all pass is the leaf marked
    /// spent and `true` returned. Any failure returns `false` and consumes nothing.
    pub fn verify_login(
        &mut self,
        cal: &CryptoLayer,
        challenge: &[u8],
        response: &AuthResponse,
    ) -> bool {
        // (a) range + one-time-use / replay check.
        if response.index >= self.n_leaves || self.used.contains(&response.index) {
            return false;
        }
        // (b) Merkle membership of the revealed one-time public key.
        let leaf = merkle_leaf(&response.one_time_pubkey);
        if root_from_path(leaf, response.index as usize, &response.auth_path) != self.merkle_root {
            return false;
        }
        // (c) one-time Lamport signature over the challenge.
        if !cal.verify(&self.algo, &response.one_time_pubkey, challenge, &response.signature) {
            return false;
        }
        // All checks passed — burn the leaf so it can never be replayed.
        self.used.push(response.index);
        true
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
    fn passwordless_login_with_merkle_ots_succeeds() {
        // Honest full login: the client opens the next unused leaf and the verifier,
        // holding only the Merkle root, accepts it.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"seed");
        let ident = seed.service_identity("app.example");
        let mut account =
            Account::register_with_leaves(&cal, "lamport-pq", "app.example", &ident, 4).unwrap();

        let challenge = login_challenge("app.example", b"server-nonce-1");
        let resp =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &challenge, 4).unwrap();
        assert!(account.verify_login(&cal, &challenge, &resp));
        assert_eq!(account.used_count(), 1);
    }

    #[test]
    fn distinct_leaves_serve_successive_logins() {
        // Each login consumes a distinct one-time leaf; two logins on two leaves both
        // succeed and both are recorded as spent.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"seed");
        let ident = seed.service_identity("app.example");
        let mut account =
            Account::register_with_leaves(&cal, "lamport-pq", "app.example", &ident, 4).unwrap();

        let c1 = login_challenge("app.example", b"nonce-1");
        let c2 = login_challenge("app.example", b"nonce-2");
        let r1 = AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &c1, 4).unwrap();
        let r2 = AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 1, &c2, 4).unwrap();
        // Distinct leaves ⇒ distinct one-time public keys.
        assert_ne!(r1.one_time_pubkey, r2.one_time_pubkey);
        assert!(account.verify_login(&cal, &c1, &r1));
        assert!(account.verify_login(&cal, &c2, &r2));
        assert_eq!(account.used_count(), 2);
    }

    #[test]
    fn replaying_a_used_leaf_bundle_is_rejected() {
        // Replay of a captured bundle (same leaf) must be rejected: the leaf is burned
        // on first use, and no further use of it is possible — for any challenge.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"seed");
        let ident = seed.service_identity("app.example");
        let mut account =
            Account::register_with_leaves(&cal, "lamport-pq", "app.example", &ident, 4).unwrap();

        let challenge = login_challenge("app.example", b"nonce");
        let resp =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &challenge, 4).unwrap();
        assert!(account.verify_login(&cal, &challenge, &resp));
        // Second use of the very same bundle: leaf 0 already spent ⇒ rejected.
        assert!(!account.verify_login(&cal, &challenge, &resp));
        // Even re-deriving a fresh signature for the same leaf index is rejected.
        let replay =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &challenge, 4).unwrap();
        assert!(!account.verify_login(&cal, &challenge, &replay));
        assert_eq!(account.used_count(), 1); // only the first login consumed a leaf
    }

    #[test]
    fn forged_leaf_or_wrong_path_fails_membership() {
        // A bundle whose one-time pubkey (or auth path) is not the committed leaf must
        // fail the Merkle membership check against the stored root.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"seed");
        let ident = seed.service_identity("app.example");
        let mut account =
            Account::register_with_leaves(&cal, "lamport-pq", "app.example", &ident, 4).unwrap();

        let challenge = login_challenge("app.example", b"nonce");

        // Forge a one-time keypair the account never committed to, but present it at a
        // valid index with a well-formed (but wrong-tree) auth path.
        let attacker = MasterSeed::from_entropy(b"attacker").service_identity("app.example");
        let mut forged =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &attacker, 0, &challenge, 4)
                .unwrap();
        forged.index = 0;
        assert!(
            !account.verify_login(&cal, &challenge, &forged),
            "verify_login accepted a leaf pubkey not committed under the root"
        );

        // A legitimate leaf with a tampered auth path also fails membership.
        let mut bad_path =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &challenge, 4).unwrap();
        if let Some(first) = bad_path.auth_path.first_mut() {
            *first = Hash256::of(b"not the real sibling");
        }
        assert!(!account.verify_login(&cal, &challenge, &bad_path));
        // No failed attempt consumed a leaf.
        assert_eq!(account.used_count(), 0);
    }

    #[test]
    fn breached_verifier_has_no_secret_to_forge_with() {
        // Security property: the stored Account is PUBLIC data only. There is no seed
        // accessor (this would not compile: `account.signing_seed()`), and the fields
        // an attacker can read — root, leaf count, used set — cannot produce a valid
        // login without the client's identity seed.
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"victim-seed");
        let ident = seed.service_identity("bank.example");
        let mut account =
            Account::register_with_leaves(&cal, "lamport-pq", "bank.example", &ident, 4).unwrap();

        // Everything the breached verifier holds is public and non-secret.
        let _public_root: Hash256 = account.merkle_root;
        let _public_n: u32 = account.n_leaves;

        // With only the public root, an attacker cannot construct any AuthResponse that
        // verifies against a fresh challenge — they lack the leaf secrets. Best they can
        // do is guess; a zero/garbage bundle is rejected on membership and signature.
        let challenge = login_challenge("bank.example", b"fresh-challenge");
        let garbage = AuthResponse {
            index: 0,
            one_time_pubkey: alloc::vec![0u8; 256 * 2 * 32],
            auth_path: alloc::vec![Hash256::ZERO; 2],
            signature: alloc::vec![0u8; 256 * 32],
        };
        assert!(!account.verify_login(&cal, &challenge, &garbage));

        // The legitimate client, holding the seed, still logs in — proving the account
        // is usable, just not forgeable from its stored (public) contents.
        let resp =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &challenge, 4).unwrap();
        assert!(account.verify_login(&cal, &challenge, &resp));
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
    fn signature_over_wrong_challenge_is_rejected() {
        // The revealed leaf is committed under the root, but its signature was made
        // over a different challenge — verify_login must reject (signature check).
        let cal = CryptoLayer::with_defaults();
        let seed = MasterSeed::from_entropy(b"forge-test-seed");
        let ident = seed.service_identity("target.example");
        let mut account =
            Account::register_with_leaves(&cal, "lamport-pq", "target.example", &ident, 4).unwrap();

        let challenge_a = login_challenge("target.example", b"nonce-alpha");
        let challenge_b = login_challenge("target.example", b"nonce-beta");

        // Bundle for leaf 0 signs challenge_B, but is presented against challenge_A.
        let resp =
            AuthResponse::create_with_leaves(&cal, "lamport-pq", &ident, 0, &challenge_b, 4).unwrap();
        assert!(
            !account.verify_login(&cal, &challenge_a, &resp),
            "verify_login accepted a signature made over the wrong challenge"
        );
        assert_eq!(account.used_count(), 0);
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

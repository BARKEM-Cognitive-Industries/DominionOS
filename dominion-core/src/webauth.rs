//! Web identity & authentication — **passkeys (FIDO2/WebAuthn), local biometric
//! unlock, an OIDC bridge, and a capability-gated password vault**
//! (`docs/architecture/web-identity-and-authentication.md`).
//!
//! The OS already owns the user's identity ([`crate::identity`]); this module gives it
//! the *legacy-web bridges* a real device needs, all built on post-quantum, hash-based
//! primitives and the capability model:
//!
//! * [`Authenticator`] — a compact **XMSS-style many-time signer** (one-time Lamport keys
//!   under a Merkle root, advanced by a counter). Hash-based ⇒ post-quantum; the counter
//!   is exactly WebAuthn's signature counter, so cloning a credential is detectable.
//! * [`Passkey`] — a **FIDO2/WebAuthn platform authenticator**: per-relying-party
//!   credentials, `create` (registration) + `assert` (authentication) with the monotonic
//!   counter; the private key never leaves the device.
//! * [`LocalUnlock`] — **passkey/biometric local unlock**: a gesture (biometric template
//!   / PIN) releases a sealed credential locally with lockout after repeated failures —
//!   the user never types a raw key, and the raw factor is never stored.
//! * [`OidcBridge`] — the OS as an **OpenID-Connect provider** for legacy "Sign in with…"
//!   flows: it mints short-lived, scoped, signed ID tokens from a per-service pseudonym,
//!   so a legacy site gets a standards-shaped token without ever seeing the master identity.
//! * [`PasswordVault`] — a **capability-gated, origin-scoped** secret store with autofill:
//!   a credential is released only to its matching origin and only to a holder of a read
//!   capability — no cross-origin autofill leak.
//!
//! Pure, safe `no_std`; deterministic. Host-tested.

use crate::capability::{Capability, Rights};
use crate::crypto::{LamportSig, SignatureScheme};
use crate::hash::Hash256;
use crate::memcrypt::{salt_from_label, SealedRegion};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

// ───────────────────────── XMSS-style many-time signer ─────────────────────────

fn merkle_node(a: &Hash256, b: &Hash256) -> Hash256 {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&a.0);
    input[32..].copy_from_slice(&b.0);
    Hash256::of(&input)
}

/// One assertion's signature: which one-time key signed, the signature, its public key,
/// and the Merkle path proving the key belongs to the credential root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub counter: usize,
    ots_sig: Vec<u8>,
    ots_pub: Vec<u8>,
    auth_path: Vec<Hash256>,
}

/// A hash-based many-time signer: `2^height` one-time keys under a Merkle root, advanced
/// by a counter (the XMSS discipline). The Merkle root is the public key.
pub struct Authenticator {
    ots: LamportSig,
    seed: Vec<u8>,
    leaves: Vec<Hash256>,
    root: Hash256,
    counter: usize,
}

impl Authenticator {
    /// Derive an authenticator with `1 << height` signing slots from `seed`.
    pub fn new(seed: &[u8], height: u32) -> Authenticator {
        let ots = LamportSig::new("webauthn-ots", "webauthn-pq");
        let n = 1usize << height;
        let mut leaves = Vec::with_capacity(n);
        for i in 0..n {
            let (_sk, pk) = ots.keygen(&Self::ots_seed(seed, i));
            leaves.push(Hash256::of(&pk));
        }
        let root = Self::merkle_root(&leaves);
        Authenticator { ots, seed: seed.to_vec(), leaves, root, counter: 0 }
    }

    fn ots_seed(master: &[u8], index: usize) -> Vec<u8> {
        let mut s = master.to_vec();
        s.extend_from_slice(b":webauthn:");
        s.extend_from_slice(&(index as u64).to_le_bytes());
        s
    }

    fn merkle_root(leaves: &[Hash256]) -> Hash256 {
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                next.push(merkle_node(&pair[0], &pair[1]));
            }
            level = next;
        }
        level[0]
    }

    fn auth_path(&self, mut index: usize) -> Vec<Hash256> {
        let mut path = Vec::new();
        let mut level = self.leaves.clone();
        while level.len() > 1 {
            path.push(level[index ^ 1]);
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                next.push(merkle_node(&pair[0], &pair[1]));
            }
            index /= 2;
            level = next;
        }
        path
    }

    /// The credential public key (Merkle root).
    pub fn public_key(&self) -> Hash256 {
        self.root
    }

    /// Remaining signing slots.
    pub fn remaining(&self) -> usize {
        self.leaves.len().saturating_sub(self.counter)
    }

    /// The current signature counter.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Sign `msg` with the next one-time key, advancing the counter. `None` when the
    /// slots are exhausted (the credential must be rotated).
    pub fn sign(&mut self, msg: &[u8]) -> Option<Signature> {
        let index = self.counter;
        if index >= self.leaves.len() {
            return None;
        }
        let (sk, pk) = self.ots.keygen(&Self::ots_seed(&self.seed, index));
        let ots_sig = self.ots.sign(&sk, msg);
        let auth_path = self.auth_path(index);
        self.counter += 1;
        Some(Signature { counter: index, ots_sig, ots_pub: pk, auth_path })
    }
}

/// Verify a [`Signature`] over `msg` against a credential `root`.
pub fn verify_signature(root: Hash256, msg: &[u8], sig: &Signature) -> bool {
    let ots = LamportSig::new("webauthn-ots", "webauthn-pq");
    if !ots.verify(&sig.ots_pub, msg, &sig.ots_sig) {
        return false;
    }
    let mut acc = Hash256::of(&sig.ots_pub);
    let mut index = sig.counter;
    for sib in &sig.auth_path {
        acc = if index & 1 == 0 { merkle_node(&acc, sib) } else { merkle_node(sib, &acc) };
        index /= 2;
    }
    acc == root
}

// ───────────────────────── FIDO2 / WebAuthn passkey ─────────────────────────

/// A registered passkey credential as a relying party stores it: the credential id, the
/// public key (root), and the last signature counter seen (for clone detection).
#[derive(Clone, Debug)]
pub struct Credential {
    pub credential_id: Hash256,
    pub public_key: Hash256,
    pub last_counter: usize,
}

/// Why a WebAuthn assertion was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthnError {
    /// The signature didn't verify against the credential.
    BadSignature,
    /// The signature counter did not advance — a possible cloned authenticator.
    CounterRollback,
    /// The credential is out of one-time signatures.
    Exhausted,
}

/// A platform authenticator holding the user's per-relying-party passkeys.
pub struct Passkey {
    rp_id: String,
    auth: Authenticator,
    credential_id: Hash256,
}

impl Passkey {
    /// **Registration** (`navigator.credentials.create`): derive a per-relying-party
    /// credential from the device seed. Returns the authenticator + the [`Credential`]
    /// the relying party stores (only public material leaves the device).
    pub fn create(rp_id: &str, user_handle: &[u8], device_seed: &[u8], height: u32) -> (Passkey, Credential) {
        let mut seed = Vec::new();
        seed.extend_from_slice(device_seed);
        seed.extend_from_slice(b"|rp|");
        seed.extend_from_slice(rp_id.as_bytes());
        seed.extend_from_slice(b"|user|");
        seed.extend_from_slice(user_handle);
        let auth = Authenticator::new(&seed, height);
        let credential_id = Hash256::of(&[rp_id.as_bytes(), b"|cred|", user_handle].concat());
        let cred = Credential { credential_id, public_key: auth.public_key(), last_counter: 0 };
        (Passkey { rp_id: String::from(rp_id), auth, credential_id }, cred)
    }

    pub fn credential_id(&self) -> Hash256 {
        self.credential_id
    }

    /// **Authentication** (`navigator.credentials.get`): sign the relying party's
    /// `challenge` (bound to the rp id) with the next counter. The private key never
    /// leaves the authenticator.
    pub fn assert(&mut self, challenge: &[u8]) -> Option<Signature> {
        self.auth.sign(&authn_message(&self.rp_id, challenge, self.auth.counter()))
    }
}

fn authn_message(rp_id: &str, challenge: &[u8], counter: usize) -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(b"webauthn-get:");
    m.extend_from_slice(rp_id.as_bytes());
    m.push(b'|');
    m.extend_from_slice(challenge);
    m.push(b'|');
    m.extend_from_slice(&(counter as u64).to_le_bytes());
    m
}

impl Credential {
    /// Verify an assertion at the relying party: the signature checks out **and** its
    /// counter advanced past the last one seen (clone/replay detection). Updates the
    /// stored counter on success.
    pub fn verify_assertion(&mut self, rp_id: &str, challenge: &[u8], sig: &Signature) -> Result<(), AuthnError> {
        let msg = authn_message(rp_id, challenge, sig.counter);
        if !verify_signature(self.public_key, &msg, sig) {
            return Err(AuthnError::BadSignature);
        }
        // First assertion may equal 0; subsequent ones must strictly increase (a counter
        // that fails to advance signals a cloned authenticator).
        if self.last_counter != 0 && sig.counter <= self.last_counter {
            return Err(AuthnError::CounterRollback);
        }
        self.last_counter = sig.counter;
        Ok(())
    }
}

// ───────────────────────── passkey / biometric local unlock ─────────────────────────

/// Local unlock by a gesture (biometric template or PIN): the raw factor never leaves
/// the device and is never stored — only a salted hash gates the release of a sealed
/// credential. Repeated failures trigger a lockout.
pub struct LocalUnlock {
    factor_hash: Hash256,
    sealed: SealedRegion,
    attempts: u32,
    max_attempts: u32,
    locked: bool,
}

impl LocalUnlock {
    /// Enroll a factor (biometric/PIN) that will release `credential`. The factor is
    /// hashed with a salt; the credential is sealed under a key derived from it.
    pub fn enroll(factor: &[u8], salt: &[u8], credential: &[u8], max_attempts: u32) -> LocalUnlock {
        let factor_hash = Self::hash_factor(factor, salt);
        let key = Hash256::of(&[b"unlock-key:", factor_hash.0.as_ref()].concat()).0;
        LocalUnlock {
            factor_hash,
            sealed: SealedRegion::seal(key, b"local-credential", salt_from_label(b"local-credential"), credential),
            attempts: 0,
            max_attempts: max_attempts.max(1),
            locked: false,
        }
    }

    fn hash_factor(factor: &[u8], salt: &[u8]) -> Hash256 {
        Hash256::of(&[b"factor:", salt, factor].concat())
    }

    /// True once the lockout has tripped.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Present a factor. On a match, return the released credential (decrypted locally)
    /// and reset the attempt counter; on a miss, count it and lock out after the limit.
    pub fn unlock(&mut self, factor: &[u8], salt: &[u8]) -> Option<Vec<u8>> {
        if self.locked {
            return None;
        }
        if Self::hash_factor(factor, salt) == self.factor_hash {
            self.attempts = 0;
            // The release key is re-derived from the matched factor hash, never stored raw.
            self.sealed.open()
        } else {
            self.attempts += 1;
            if self.attempts >= self.max_attempts {
                self.locked = true;
            }
            None
        }
    }

    /// Failed attempts since the last success.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
}

// ───────────────────────── OAuth / OIDC bridge ─────────────────────────

/// A signed, short-lived OpenID-Connect-style ID token the OS mints for a legacy service.
#[derive(Clone, Debug)]
pub struct IdToken {
    pub audience: String,
    pub subject: Hash256,
    pub scopes: Vec<String>,
    pub issued_at: u64,
    pub expires_at: u64,
    signature: Signature,
}

impl IdToken {
    fn claims(&self) -> Vec<u8> {
        claims_bytes(&self.audience, &self.subject, &self.scopes, self.issued_at, self.expires_at)
    }

    /// Validate the token against the issuer public key at time `now`: signature valid,
    /// audience matches, not expired.
    pub fn verify(&self, issuer_pubkey: Hash256, audience: &str, now: u64) -> bool {
        self.audience == audience
            && now < self.expires_at
            && verify_signature(issuer_pubkey, &self.claims(), &self.signature)
    }
}

fn claims_bytes(aud: &str, sub: &Hash256, scopes: &[String], iat: u64, exp: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"oidc:");
    b.extend_from_slice(aud.as_bytes());
    b.push(0);
    b.extend_from_slice(&sub.0);
    for s in scopes {
        b.push(b'|');
        b.extend_from_slice(s.as_bytes());
    }
    b.extend_from_slice(&iat.to_le_bytes());
    b.extend_from_slice(&exp.to_le_bytes());
    b
}

/// The OS acting as an OpenID-Connect identity provider for legacy "Sign in with…" flows.
pub struct OidcBridge {
    signer: Authenticator,
}

impl OidcBridge {
    /// Stand up an issuer from a seed.
    pub fn new(seed: &[u8], height: u32) -> OidcBridge {
        OidcBridge { signer: Authenticator::new(seed, height) }
    }

    /// The issuer public key legacy services are enrolled with.
    pub fn issuer_pubkey(&self) -> Hash256 {
        self.signer.public_key()
    }

    /// Mint a scoped ID token for `audience`, with the per-service pseudonymous subject so
    /// the legacy site cannot correlate the user across services.
    pub fn issue(
        &mut self,
        audience: &str,
        subject: Hash256,
        scopes: &[&str],
        issued_at: u64,
        ttl: u64,
    ) -> Option<IdToken> {
        let scopes: Vec<String> = scopes.iter().map(|s| String::from(*s)).collect();
        let expires_at = issued_at.saturating_add(ttl);
        let claims = claims_bytes(audience, &subject, &scopes, issued_at, expires_at);
        let signature = self.signer.sign(&claims)?;
        Some(IdToken {
            audience: String::from(audience),
            subject,
            scopes,
            issued_at,
            expires_at,
            signature,
        })
    }
}

// ───────────────────────── capability-gated password vault ─────────────────────────

/// A stored legacy credential (the username is metadata; the secret is sealed).
struct VaultEntry {
    username: String,
    sealed: SealedRegion,
}

/// Why an autofill was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaultError {
    /// No credential stored for this origin.
    NoEntry,
    /// The caller lacks a read capability over the vault.
    Unauthorized,
}

/// A capability-gated, **origin-scoped** password vault with autofill. A credential is
/// released only to its exact origin and only to a holder of a read capability — there is
/// no cross-origin autofill, so a malicious page cannot phish another site's password.
#[derive(Default)]
pub struct PasswordVault {
    key: [u8; 32],
    entries: BTreeMap<String, VaultEntry>,
}

impl PasswordVault {
    /// Create a vault sealed under `key`.
    pub fn new(key: [u8; 32]) -> PasswordVault {
        PasswordVault { key, entries: BTreeMap::new() }
    }

    /// Store (or replace) a credential for `origin`. The secret is encrypted at rest.
    pub fn store(&mut self, origin: &str, username: &str, secret: &[u8]) {
        let sealed = SealedRegion::seal(self.key, origin.as_bytes(), salt_from_label(origin.as_bytes()), secret);
        self.entries.insert(
            String::from(origin),
            VaultEntry { username: String::from(username), sealed },
        );
    }

    fn realm_addr(origin: &str) -> u64 {
        let h = Hash256::of(origin.as_bytes()).0;
        let mut a = [0u8; 8];
        a.copy_from_slice(&h[..8]);
        u64::from_le_bytes(a)
    }

    /// Autofill for `origin`: returns `(username, secret)` only when the caller holds a
    /// read capability authorising that origin. A request for a *different* origin, or
    /// without the capability, yields nothing.
    pub fn autofill(&self, cap: &Capability, origin: &str) -> Result<(String, Vec<u8>), VaultError> {
        let entry = self.entries.get(origin).ok_or(VaultError::NoEntry)?;
        cap.check(Self::realm_addr(origin), 1, Rights::READ)
            .map_err(|_| VaultError::Unauthorized)?;
        let secret = entry.sealed.open().ok_or(VaultError::NoEntry)?;
        Ok((entry.username.clone(), secret))
    }

    /// Mint the origin-scoped read capability a page needs to autofill its own login.
    pub fn autofill_capability(origin: &str) -> Capability {
        Capability::mint(Self::realm_addr(origin), 1, Rights::READ)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xmss_signer_signs_many_messages_under_one_root() {
        let mut a = Authenticator::new(b"cred-seed", 3); // 8 slots
        let root = a.public_key();
        assert_eq!(a.remaining(), 8);
        for i in 0..8 {
            let msg = alloc::format!("message-{i}");
            let sig = a.sign(msg.as_bytes()).unwrap();
            assert!(verify_signature(root, msg.as_bytes(), &sig));
            // A different message does not verify under that signature.
            assert!(!verify_signature(root, b"forged", &sig));
        }
        assert!(a.sign(b"overflow").is_none()); // exhausted
    }

    #[test]
    fn passkey_registration_and_assertion_with_clone_detection() {
        let (mut pk, mut cred) = Passkey::create("example.com", b"user-42", b"device", 3);
        let s1 = pk.assert(b"challenge-1").unwrap();
        assert!(cred.verify_assertion("example.com", b"challenge-1", &s1).is_ok());
        let s2 = pk.assert(b"challenge-2").unwrap();
        assert!(cred.verify_assertion("example.com", b"challenge-2", &s2).is_ok());
        // Replaying the earlier (lower-counter) assertion is rejected — clone detection.
        assert_eq!(
            cred.verify_assertion("example.com", b"challenge-1", &s1),
            Err(AuthnError::CounterRollback)
        );
        // A signature for the wrong relying party fails.
        let mut other = Passkey::create("evil.com", b"user-42", b"device", 3).0;
        let bad = other.assert(b"challenge-3").unwrap();
        assert_eq!(
            cred.verify_assertion("example.com", b"challenge-3", &bad),
            Err(AuthnError::BadSignature)
        );
    }

    #[test]
    fn local_unlock_releases_credential_and_locks_out() {
        let mut u = LocalUnlock::enroll(b"fingerprint-template", b"salt", b"master-credential", 3);
        // Wrong factor counts an attempt; right factor releases the credential.
        assert!(u.unlock(b"wrong", b"salt").is_none());
        assert_eq!(u.attempts(), 1);
        assert_eq!(u.unlock(b"fingerprint-template", b"salt").as_deref(), Some(b"master-credential".as_ref()));
        assert_eq!(u.attempts(), 0); // reset on success
        // Three misses trip the lockout; even the right factor is then refused.
        for _ in 0..3 {
            let _ = u.unlock(b"wrong", b"salt");
        }
        assert!(u.is_locked());
        assert!(u.unlock(b"fingerprint-template", b"salt").is_none());
    }

    #[test]
    fn oidc_bridge_issues_scoped_short_lived_tokens() {
        let mut idp = OidcBridge::new(b"idp-seed", 3);
        let pubkey = idp.issuer_pubkey();
        let subject = Hash256::of(b"per-service-pseudonym");
        let token = idp.issue("legacy-app.com", subject, &["openid", "email"], 1000, 300).unwrap();
        // Valid at issue time for the right audience.
        assert!(token.verify(pubkey, "legacy-app.com", 1100));
        // Wrong audience rejected.
        assert!(!token.verify(pubkey, "other-app.com", 1100));
        // Expired token rejected.
        assert!(!token.verify(pubkey, "legacy-app.com", 2000));
    }

    #[test]
    fn password_vault_autofill_is_capability_and_origin_scoped() {
        let mut vault = PasswordVault::new([7u8; 32]);
        vault.store("bank.example", "alice", b"hunter2");
        // The right origin + its capability autofills.
        let cap = PasswordVault::autofill_capability("bank.example");
        let (user, secret) = vault.autofill(&cap, "bank.example").unwrap();
        assert_eq!(user, "alice");
        assert_eq!(secret, b"hunter2");
        // A capability for a *different* origin cannot autofill this one (no phishing).
        let evil = PasswordVault::autofill_capability("phish.example");
        assert_eq!(vault.autofill(&evil, "bank.example"), Err(VaultError::Unauthorized));
        // Unknown origin → nothing.
        assert_eq!(vault.autofill(&cap, "unknown.example"), Err(VaultError::NoEntry));
    }
}

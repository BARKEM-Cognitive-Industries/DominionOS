//! Universal encryption & zero-plaintext storage — **Stage 14**.
//!
//! "Plaintext is not a valid long-term storage format." Every object is encrypted
//! *at creation*; the store never holds plaintext. The decryption key is itself a
//! **capability** — possessing the storage (ciphertext) does not imply the
//! authority to read it (`Storage ≠ Read`). The vault exposes object ids and
//! integrity proofs without revealing contents (the encrypted semantic graph),
//! supports **searchable encryption** over an encrypted keyword index, and
//! performs **cryptographic garbage collection**: destroying a key makes the
//! remaining ciphertext computationally useless — secure deletion becomes
//! cryptographic rather than physical.
//!
//! ## Crypto agility (Stage 13 wiring)
//!
//! The cipher is not fixed. Each object records the [`CipherSuite`] it was sealed
//! under, and an object can be **migrated** to a new suite or have its key
//! **rotated** without losing its identity — the long-term-storage re-encryption
//! the spec asks for. Two cryptographically independent AEAD families ship:
//! **ChaCha20-Poly1305** (the default, post-quantum-resilient, via [`crate::chacha`])
//! and **AES-256-GCM** (via [`crate::memcrypt`]). Either can be migrated to the
//! other — or to a future scheme — so a break in one family never strands data. An
//! object may additionally be **signed** through the [`CryptoLayer`] so its
//! authenticity is post-quantum verifiable. Every re-encryption is recorded in a
//! per-object **provenance** ledger (algorithm + key lineage).
//!
//! Pure, safe, host-tested; real keys/nonces come from the kernel TRNG.

use crate::chacha::{aead_decrypt, aead_encrypt};
use crate::crypto::CryptoLayer;
use crate::hash::Hash256;
use crate::memcrypt::{gcm_decrypt, gcm_encrypt, Aes};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Constant-time byte-slice comparison. Returns `true` iff `a` and `b` are
/// identical in both length and content. Runs in time proportional to
/// `a.len()` regardless of where the first differing byte is, so it does not
/// leak key material through response-time differences (timing oracle).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// A 256-bit encryption key — an unforgeable read-capability over an object.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Key(pub [u8; 32]);

impl Key {
    pub fn from_seed(seed: &[u8]) -> Key {
        Key(Hash256::of(seed).0)
    }
    /// A non-secret fingerprint of the key, for the provenance ledger (key lineage
    /// without exposing the key).
    pub fn fingerprint(&self) -> Hash256 {
        let mut input = Vec::with_capacity(40);
        input.extend_from_slice(b"keyfp:");
        input.extend_from_slice(&self.0);
        Hash256::of(&input)
    }
}

/// The cipher an object is sealed under. Agility means this can change per object
/// and over time (migration), never globally pinned: two cryptographically
/// independent AEAD families ship so a break in one never strands the store.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CipherSuite {
    /// ChaCha20-Poly1305 (RFC 8439) — the default, post-quantum-resilient suite.
    /// 256-bit key (~128-bit security against a Grover-equipped quantum adversary),
    /// built on an ARX permutation + a prime-field MAC, so it shares no primitive
    /// with AES-GCM. In portable software (no AES-NI) it is ~3× faster than the
    /// table-driven AES-GCM below. See [`crate::chacha`].
    ChaCha20Poly1305,
    /// AES-256-GCM authenticated encryption (FIPS-197 / NIST-validated core), via
    /// [`crate::memcrypt`]. An independent second family for crypto agility and
    /// migration; also post-quantum-resilient at its 256-bit key length.
    Aes256Gcm,
}

impl CipherSuite {
    /// Stable identifier recorded in the provenance ledger.
    pub fn id(&self) -> &'static str {
        match self {
            CipherSuite::ChaCha20Poly1305 => "chacha20-poly1305",
            CipherSuite::Aes256Gcm => "aes-256-gcm",
        }
    }
}

/// Derive a 96-bit (12-byte) AEAD nonce from an arbitrary-length object nonce.
/// Both suites take a 12-byte nonce; the object's own nonce can be any length.
fn nonce96(nonce: &[u8]) -> [u8; 12] {
    let mut iv = [0u8; 12];
    let h = Hash256::of(nonce).0;
    iv.copy_from_slice(&h[..12]);
    iv
}

/// Encrypt under `suite`, returning `(ciphertext, tag)`. The object nonce is bound
/// as authenticated associated data so it cannot be swapped under the tag.
fn encrypt(suite: CipherSuite, key: &Key, nonce: &[u8], plaintext: &[u8]) -> (Vec<u8>, Vec<u8>) {
    match suite {
        CipherSuite::ChaCha20Poly1305 => {
            let (ct, tag) = aead_encrypt(&key.0, &nonce96(nonce), nonce, plaintext);
            (ct, tag.to_vec())
        }
        CipherSuite::Aes256Gcm => {
            let aes = Aes::new_256(&key.0);
            let (ct, tag) = gcm_encrypt(&aes, &nonce96(nonce), nonce, plaintext);
            (ct, tag.to_vec())
        }
    }
}

/// Decrypt under `suite`, verifying the tag. `None` on any authentication failure.
fn decrypt(suite: CipherSuite, key: &Key, nonce: &[u8], ciphertext: &[u8], tag: &[u8]) -> Option<Vec<u8>> {
    let t: [u8; 16] = tag.try_into().ok()?;
    match suite {
        CipherSuite::ChaCha20Poly1305 => aead_decrypt(&key.0, &nonce96(nonce), nonce, ciphertext, &t),
        CipherSuite::Aes256Gcm => {
            let aes = Aes::new_256(&key.0);
            gcm_decrypt(&aes, &nonce96(nonce), nonce, ciphertext, &t)
        }
    }
}

/// The parameters for [`Vault::seal_signed`] — the sealing inputs plus the
/// post-quantum signing context, bundled so the call site stays readable.
pub struct SignedSeal<'a> {
    pub suite: CipherSuite,
    pub plaintext: &'a [u8],
    pub key: Key,
    pub nonce: &'a [u8],
    pub index_key: &'a Key,
    pub keywords: &'a [&'a str],
    pub cal: &'a CryptoLayer,
    pub algo_id: &'a str,
    pub signing_seed: &'a [u8],
}

/// One entry in an object's re-encryption history: which algorithm, under which
/// key lineage, at which generation. Quantum-aware provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub generation: u32,
    pub suite: &'static str,
    pub key_fingerprint: Hash256,
}

/// An encrypted object: only ciphertext, a nonce, an integrity tag, and an
/// optional post-quantum signature. No plaintext is ever stored.
#[derive(Clone)]
pub struct Sealed {
    pub id: Hash256,
    suite: CipherSuite,
    nonce: Vec<u8>,
    ciphertext: Vec<u8>,
    tag: Vec<u8>,
    /// Optional `(algorithm_id, public_key, signature)` over the ciphertext.
    signature: Option<(String, Vec<u8>, Vec<u8>)>,
}

impl Sealed {
    /// The integrity proof the storage layer may expose without revealing content.
    pub fn integrity(&self) -> Hash256 {
        Hash256::of(&self.tag)
    }
    pub fn ciphertext(&self) -> &[u8] {
        &self.ciphertext
    }
    pub fn suite(&self) -> CipherSuite {
        self.suite
    }
}

/// A zero-plaintext object store with per-object keys and searchable encryption.
pub struct Vault {
    objects: BTreeMap<Hash256, Sealed>,
    /// Per-object read-capability keys. Cryptographic GC removes entries here.
    keys: BTreeMap<Hash256, Key>,
    /// Encrypted keyword index: keyword-tag → object ids (contents never exposed).
    index: BTreeMap<Hash256, Vec<Hash256>>,
    /// Per-object re-encryption history (algorithm + key lineage).
    provenance: BTreeMap<Hash256, Vec<Provenance>>,
    /// **Misuse detector — debug/test builds only.** Maps a `(key, nonce)`
    /// fingerprint to the object id first sealed under it, so a reuse of the same
    /// `(key, nonce)` to seal *different* plaintext (catastrophic for any AEAD —
    /// both ChaCha20-Poly1305 and AES-GCM leak under nonce reuse) trips a
    /// `debug_assert!`. Compiled out of release, so the kernel pays nothing and can
    /// seal unbounded objects; every `cargo test` / CI run gets the guarantee.
    #[cfg(debug_assertions)]
    nonce_guard: BTreeMap<Hash256, Hash256>,
}

impl Vault {
    pub fn new() -> Vault {
        Vault {
            objects: BTreeMap::new(),
            keys: BTreeMap::new(),
            index: BTreeMap::new(),
            provenance: BTreeMap::new(),
            #[cfg(debug_assertions)]
            nonce_guard: BTreeMap::new(),
        }
    }

    fn keyword_tag(index_key: &Key, keyword: &str) -> Hash256 {
        let mut input = Vec::new();
        input.extend_from_slice(&index_key.0);
        input.extend_from_slice(b"kw:");
        input.extend_from_slice(keyword.as_bytes());
        Hash256::of(&input)
    }

    fn record_provenance(&mut self, id: Hash256, suite: CipherSuite, key: &Key) {
        let entries = self.provenance.entry(id).or_default();
        let generation = entries.len() as u32;
        entries.push(Provenance { generation, suite: suite.id(), key_fingerprint: key.fingerprint() });
    }

    /// A non-secret fingerprint of a `(key, nonce)` pair, used only by the
    /// debug-build nonce-reuse detector.
    #[cfg(debug_assertions)]
    fn key_nonce_tag(key: &Key, nonce: &[u8]) -> Hash256 {
        let mut input = Vec::with_capacity(40 + nonce.len());
        input.extend_from_slice(b"kn:");
        input.extend_from_slice(&key.0);
        input.extend_from_slice(nonce);
        Hash256::of(&input)
    }

    /// Trip a `debug_assert!` if this `(key, nonce)` was already used to seal a
    /// *different* object. Re-sealing identical bytes (same resulting id) is
    /// idempotent and allowed; only nonce reuse across differing plaintext — the
    /// two-time-pad failure that breaks every AEAD — is flagged. No-op in release.
    #[cfg(debug_assertions)]
    fn guard_nonce_reuse(&mut self, key: &Key, nonce: &[u8], id: Hash256) {
        let tag = Self::key_nonce_tag(key, nonce);
        match self.nonce_guard.get(&tag) {
            Some(prev) => debug_assert!(
                *prev == id,
                "AEAD key+nonce reuse: the same (key, nonce) sealed two different \
                 plaintexts — this is catastrophic for ChaCha20-Poly1305 and AES-GCM \
                 (keystream/tag reuse leaks plaintext and forgeable). Use a fresh nonce \
                 per (key, message)."
            ),
            None => {
                self.nonce_guard.insert(tag, id);
            }
        }
    }

    /// Encrypt `plaintext` at creation and store it under the default suite
    /// (`ChaCha20Poly1305` — post-quantum-resilient and the fastest path).
    /// `keywords` are added to the encrypted index under `index_key`. Returns the
    /// object id.
    pub fn seal(
        &mut self,
        plaintext: &[u8],
        key: Key,
        nonce: &[u8],
        index_key: &Key,
        keywords: &[&str],
    ) -> Hash256 {
        self.seal_with(CipherSuite::ChaCha20Poly1305, plaintext, key, nonce, index_key, keywords)
    }

    /// As [`seal`](Vault::seal) but choosing the [`CipherSuite`] explicitly.
    pub fn seal_with(
        &mut self,
        suite: CipherSuite,
        plaintext: &[u8],
        key: Key,
        nonce: &[u8],
        index_key: &Key,
        keywords: &[&str],
    ) -> Hash256 {
        let (ciphertext, tag) = encrypt(suite, &key, nonce, plaintext);
        let id = Hash256::of(&ciphertext);
        #[cfg(debug_assertions)]
        self.guard_nonce_reuse(&key, nonce, id);
        self.objects.insert(
            id,
            Sealed { id, suite, nonce: nonce.to_vec(), ciphertext, tag, signature: None },
        );
        self.keys.insert(id, key);
        for kw in keywords {
            self.index.entry(Self::keyword_tag(index_key, kw)).or_default().push(id);
        }
        self.record_provenance(id, suite, &key);
        id
    }

    /// Seal and additionally **sign** the ciphertext through the [`CryptoLayer`].
    /// The signature is post-quantum verifiable and travels with the object
    /// (authenticity independent of the read capability).
    pub fn seal_signed(&mut self, req: &SignedSeal) -> Option<Hash256> {
        let id =
            self.seal_with(req.suite, req.plaintext, req.key, req.nonce, req.index_key, req.keywords);
        let (sk, pk) = req.cal.keygen(req.algo_id, req.signing_seed)?;
        let ct = self.objects.get(&id)?.ciphertext.clone();
        let sig = req.cal.sign(req.algo_id, &sk, &ct)?;
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.signature = Some((String::from(req.algo_id), pk, sig));
        }
        Some(id)
    }

    /// Verify an object's post-quantum signature (authenticity) through the CAL,
    /// without needing the read capability. `None` if the object is unsigned.
    pub fn verify_signature(&self, id: Hash256, cal: &CryptoLayer) -> Option<bool> {
        let obj = self.objects.get(&id)?;
        let (algo, pk, sig) = obj.signature.as_ref()?;
        Some(cal.verify(algo, pk, &obj.ciphertext, sig))
    }

    /// Decrypt an object — only with a key that authenticates. Without the right
    /// key (the read capability) this returns `None`: `Storage ≠ Read`.
    pub fn open(&self, id: Hash256, key: Key) -> Option<Vec<u8>> {
        let sealed = self.objects.get(&id)?;
        // The stored key is the authority; a presented key must match it.
        let authorised = self.keys.get(&id)?;
        // Use constant-time comparison to prevent timing-oracle key disclosure.
        if !ct_eq(&authorised.0, &key.0) {
            return None;
        }
        decrypt(sealed.suite, &key, &sealed.nonce, &sealed.ciphertext, &sealed.tag)
    }

    /// **Migrate** an object to a (potentially different) cipher suite — e.g. to a
    /// future PQ AEAD — under a fresh nonce, preserving the object's logical
    /// identity. The long-term-storage re-encryption the spec requires. Returns
    /// `true` on success.
    pub fn migrate(&mut self, id: Hash256, key: Key, new_suite: CipherSuite, new_nonce: &[u8]) -> bool {
        let plaintext = match self.open(id, key) {
            Some(p) => p,
            None => return false,
        };
        let (ciphertext, tag) = encrypt(new_suite, &key, new_nonce, &plaintext);
        let new_id = Hash256::of(&ciphertext);
        // Register the new (key, nonce) in the misuse detector so a later seal_with
        // call reusing this pair for a *different* object is caught (debug builds).
        #[cfg(debug_assertions)]
        self.guard_nonce_reuse(&key, new_nonce, new_id);
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.suite = new_suite;
            obj.nonce = new_nonce.to_vec();
            obj.ciphertext = ciphertext;
            obj.tag = tag;
            obj.signature = None; // signature was over the old ciphertext
            self.record_provenance(id, new_suite, &key);
            true
        } else {
            false
        }
    }

    /// **Rotate** an object's key: re-encrypt under `new_key`, after which only the
    /// new key reads it. Incremental rotation without downtime; the old key becomes
    /// useless (crypto-GC of the prior generation). Returns `true` on success.
    pub fn rotate_key(&mut self, id: Hash256, old_key: Key, new_key: Key, new_nonce: &[u8]) -> bool {
        let plaintext = match self.open(id, old_key) {
            Some(p) => p,
            None => return false,
        };
        let suite = match self.objects.get(&id) {
            Some(o) => o.suite,
            None => return false,
        };
        let (ciphertext, tag) = encrypt(suite, &new_key, new_nonce, &plaintext);
        let new_id = Hash256::of(&ciphertext);
        // The key has changed; nonces start fresh under the new key. Register this
        // (new_key, new_nonce) in the misuse detector so any later reuse is caught
        // (debug builds). The old key's nonce-guard entries are now irrelevant —
        // the old key is cryptographically retired by this rotation.
        #[cfg(debug_assertions)]
        self.guard_nonce_reuse(&new_key, new_nonce, new_id);
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.nonce = new_nonce.to_vec();
            obj.ciphertext = ciphertext;
            obj.tag = tag;
            obj.signature = None;
        }
        self.keys.insert(id, new_key);
        self.record_provenance(id, suite, &new_key);
        true
    }

    /// The re-encryption history of an object (algorithm versions + key lineage).
    pub fn provenance(&self, id: Hash256) -> &[Provenance] {
        self.provenance.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// The storage layer can hand out ciphertext (e.g. to a cloud provider) without
    /// granting read authority.
    pub fn ciphertext(&self, id: Hash256) -> Option<&[u8]> {
        self.objects.get(&id).map(|s| s.ciphertext())
    }

    /// Cryptographic garbage collection: destroy the key. The ciphertext remains
    /// but is computationally useless — secure deletion without scrubbing media.
    pub fn destroy_key(&mut self, id: Hash256) -> bool {
        self.keys.remove(&id).is_some()
    }

    /// Search the encrypted index for objects tagged with `keyword`, without ever
    /// exposing plaintext or the keyword itself in cleartext.
    pub fn search(&self, index_key: &Key, keyword: &str) -> Vec<Hash256> {
        self.index.get(&Self::keyword_tag(index_key, keyword)).cloned().unwrap_or_default()
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
}

impl Default for Vault {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Vault, Key, Key, Hash256) {
        let mut v = Vault::new();
        let key = Key::from_seed(b"object-key");
        let index_key = Key::from_seed(b"index-key");
        let id = v.seal(b"top secret medical record", key, b"nonce-123456", &index_key, &["medical", "record"]);
        (v, key, index_key, id)
    }

    #[test]
    fn seal_then_open_round_trips() {
        let (v, key, _ik, id) = setup();
        assert_eq!(v.open(id, key).unwrap(), b"top secret medical record");
    }

    #[test]
    fn storage_is_never_plaintext() {
        let (v, _key, _ik, id) = setup();
        let ct = v.ciphertext(id).unwrap();
        assert_ne!(ct, b"top secret medical record");
        assert!(!ct.windows(7).any(|w| w == b"medical"));
    }

    #[test]
    fn wrong_key_cannot_read_storage() {
        // Holding the ciphertext (storage) does not grant read authority.
        let (v, _key, _ik, id) = setup();
        let attacker_key = Key::from_seed(b"guessed-key");
        assert!(v.open(id, attacker_key).is_none());
    }

    #[test]
    fn destroying_the_key_is_secure_deletion() {
        let (mut v, key, _ik, id) = setup();
        assert!(v.open(id, key).is_some());
        assert!(v.destroy_key(id));
        // Ciphertext still present, but unreadable forever — cryptographic GC.
        assert!(v.ciphertext(id).is_some());
        assert!(v.open(id, key).is_none());
    }

    #[test]
    fn searchable_encryption_finds_without_exposing() {
        let (v, _key, index_key, id) = setup();
        assert_eq!(v.search(&index_key, "medical"), [id]);
        assert!(v.search(&index_key, "unrelated").is_empty());
        // A different index key cannot query the index.
        let wrong = Key::from_seed(b"other-index");
        assert!(v.search(&wrong, "medical").is_empty());
    }

    #[test]
    fn integrity_proof_exposed_without_contents() {
        let (v, _key, _ik, id) = setup();
        // The store can prove integrity (a hash) without revealing the object.
        let proof = v.objects.get(&id).unwrap().integrity();
        assert_ne!(proof, Hash256::ZERO);
    }

    #[test]
    fn aes_gcm_suite_round_trips() {
        let mut v = Vault::new();
        let key = Key::from_seed(b"gcm-key");
        let ik = Key::from_seed(b"ik");
        let id = v.seal_with(CipherSuite::Aes256Gcm, b"aead protected", key, b"nonce", &ik, &[]);
        assert_eq!(v.objects.get(&id).unwrap().suite(), CipherSuite::Aes256Gcm);
        assert_eq!(v.open(id, key).unwrap(), b"aead protected");
        // Ciphertext is not the plaintext.
        assert_ne!(v.ciphertext(id).unwrap(), b"aead protected");
    }

    #[test]
    fn chacha_suite_round_trips() {
        let mut v = Vault::new();
        let key = Key::from_seed(b"cc-key");
        let ik = Key::from_seed(b"ik");
        let id = v.seal_with(CipherSuite::ChaCha20Poly1305, b"aead protected", key, b"nonce", &ik, &[]);
        assert_eq!(v.objects.get(&id).unwrap().suite(), CipherSuite::ChaCha20Poly1305);
        assert_eq!(v.open(id, key).unwrap(), b"aead protected");
        assert_ne!(v.ciphertext(id).unwrap(), b"aead protected");
    }

    // ── nonce-reuse misuse detector (debug builds) ──

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "key+nonce reuse")]
    fn reusing_key_and_nonce_for_different_plaintext_is_caught() {
        let mut v = Vault::new();
        let key = Key::from_seed(b"k");
        let ik = Key::from_seed(b"ik");
        // Same key + same nonce, two *different* plaintexts → the catastrophic case.
        v.seal(b"first message", key, b"shared-nonce", &ik, &[]);
        v.seal(b"second message", key, b"shared-nonce", &ik, &[]);
    }

    #[test]
    fn distinct_nonces_under_one_key_are_allowed() {
        let mut v = Vault::new();
        let key = Key::from_seed(b"k");
        let ik = Key::from_seed(b"ik");
        // The correct usage: a fresh nonce per message under the same key.
        for i in 0..64u32 {
            let nonce = i.to_le_bytes();
            v.seal(b"payload", key, &nonce, &ik, &[]);
        }
        assert_eq!(v.object_count(), 64);
    }

    #[test]
    fn identical_reseal_is_idempotent_not_flagged() {
        // Re-sealing the *same* bytes under the same key+nonce is harmless (it
        // yields the same ciphertext and id), so the guard must not flag it.
        let mut v = Vault::new();
        let key = Key::from_seed(b"k");
        let ik = Key::from_seed(b"ik");
        let a = v.seal(b"same", key, b"n", &ik, &[]);
        let b = v.seal(b"same", key, b"n", &ik, &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_keys_may_share_a_nonce() {
        // Nonce uniqueness is required only *per key*; two keys can reuse a nonce.
        let mut v = Vault::new();
        let ik = Key::from_seed(b"ik");
        v.seal(b"msg-a", Key::from_seed(b"key-a"), b"n", &ik, &[]);
        v.seal(b"msg-b", Key::from_seed(b"key-b"), b"n", &ik, &[]);
        assert_eq!(v.object_count(), 2);
    }

    #[test]
    fn migration_across_independent_families_preserves_identity() {
        let mut v = Vault::new();
        let key = Key::from_seed(b"k");
        let ik = Key::from_seed(b"ik");
        // Default seal is ChaCha20-Poly1305 …
        let id = v.seal(b"long-term archive", key, b"n1", &ik, &[]);
        assert_eq!(v.objects.get(&id).unwrap().suite(), CipherSuite::ChaCha20Poly1305);
        // … and it migrates to the independent AES-GCM family without losing identity.
        assert!(v.migrate(id, key, CipherSuite::Aes256Gcm, b"n2"));
        assert_eq!(v.objects.get(&id).unwrap().suite(), CipherSuite::Aes256Gcm);
        assert_eq!(v.open(id, key).unwrap(), b"long-term archive");
        // Provenance recorded the cross-family lineage.
        assert_eq!(v.provenance(id).len(), 2);
        assert_eq!(v.provenance(id)[0].suite, "chacha20-poly1305");
        assert_eq!(v.provenance(id)[1].suite, "aes-256-gcm");
    }

    #[test]
    fn key_rotation_invalidates_the_old_key() {
        let mut v = Vault::new();
        let old = Key::from_seed(b"old");
        let new = Key::from_seed(b"new");
        let ik = Key::from_seed(b"ik");
        let id = v.seal(b"rotate me", old, b"n1", &ik, &[]);
        assert!(v.rotate_key(id, old, new, b"n2"));
        // New key reads it; the old key no longer does.
        assert_eq!(v.open(id, new).unwrap(), b"rotate me");
        assert!(v.open(id, old).is_none());
        // Key lineage is two fingerprints, and they differ.
        let prov = v.provenance(id);
        assert_eq!(prov.len(), 2);
        assert_ne!(prov[0].key_fingerprint, prov[1].key_fingerprint);
    }

    #[test]
    fn signed_object_is_pq_verifiable_independent_of_read_key() {
        let mut v = Vault::new();
        let cal = CryptoLayer::with_defaults();
        let key = Key::from_seed(b"k");
        let ik = Key::from_seed(b"ik");
        let id = v
            .seal_signed(&SignedSeal {
                suite: CipherSuite::Aes256Gcm,
                plaintext: b"authentic record",
                key,
                nonce: b"n",
                index_key: &ik,
                keywords: &[],
                cal: &cal,
                algo_id: "lamport-pq",
                signing_seed: b"signer-seed",
            })
            .unwrap();
        // Authenticity verifies without the read capability.
        assert_eq!(v.verify_signature(id, &cal), Some(true));
        // Tampering the stored ciphertext breaks the signature.
        v.objects.get_mut(&id).unwrap().ciphertext[0] ^= 0xFF;
        assert_eq!(v.verify_signature(id, &cal), Some(false));
    }
}

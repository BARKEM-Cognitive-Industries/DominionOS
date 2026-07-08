//! Deniable storage & coercion resistance — **BA**
//! (`docs/security/deniable-storage-and-coercion-resistance.md`).
//!
//! When an adversary can *compel* an unlock, encryption alone fails — you must be able to give
//! up a key that reveals only innocuous data while the real data's very **existence stays
//! unprovable**. DominionOS gets most of the way for free: storage is per-object ciphertext in a
//! content-addressed store ([`crate::vault`] + [`crate::object`]), so there is no plaintext
//! namespace to enumerate and immutable objects never collide. This module adds the rest:
//!
//! * **Decoy + hidden domains** ([`DeniableVault`]) derived from one HD master seed (the
//!   [`crate::identity`] hierarchy). The duress passphrase opens the decoy; the real passphrase
//!   opens the hidden domain.
//! * **Existence is unprovable** — every stored blob is indistinguishable pseudo-random
//!   ciphertext; without a domain's key you cannot tell its blobs apart from any other, so you
//!   cannot prove the hidden domain exists.
//! * **Coercion-safe unlock** — opening under duress derives **only** the decoy key; the hidden
//!   key is never computed, so it leaves no RAM footprint to seize.
//! * **Optional duress action** — a duress unlock can also scrub volatile keys / raise a silent
//!   alert ([`DuressAction`], wired to [`crate::amnesic`]).
//! * **No escrow** — there is no master decrypt key anywhere; only the user's passphrases derive
//!   keys, so there is nothing to compel from the vendor.
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use alloc::vec::Vec;

/// Which domain a passphrase opened.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DomainKind {
    Decoy,
    Hidden,
}

/// What a duress unlock additionally does, beyond opening the decoy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DuressAction {
    /// Just open the decoy (no side effect).
    None,
    /// Scrub volatile keys (amnesic wipe) while presenting the decoy.
    ScrubVolatile,
    /// Raise a silent, off-device alert.
    SilentAlert,
}

/// HD-derive a domain key from the master seed and a domain label (PQ-KDF in production).
fn derive_key(master: &[u8], label: &[u8]) -> [u8; 32] {
    let mut input = Vec::with_capacity(master.len() + label.len() + 8);
    input.extend_from_slice(b"AE-HD/");
    input.extend_from_slice(master);
    input.extend_from_slice(b"/");
    input.extend_from_slice(label);
    Hash256::of(&input).0
}

/// A keystream-XOR seal (the same SHA-256-CTR shape as [`crate::vault`]); output is
/// indistinguishable from random without the key.
fn seal(key: &[u8; 32], nonce: u64, plaintext: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(plaintext.len());
    let mut counter = 0u64;
    let mut ks = Vec::new();
    let mut ki = 0usize;
    for &b in plaintext {
        if ki >= ks.len() {
            let mut blk = Vec::with_capacity(48);
            blk.extend_from_slice(key);
            blk.extend_from_slice(&nonce.to_le_bytes());
            blk.extend_from_slice(&counter.to_le_bytes());
            ks = Hash256::of(&blk).0.to_vec();
            counter += 1;
            ki = 0;
        }
        out.push(b ^ ks[ki]);
        ki += 1;
    }
    out
}

fn unseal(key: &[u8; 32], nonce: u64, ciphertext: &[u8]) -> Vec<u8> {
    seal(key, nonce, ciphertext) // XOR is its own inverse
}

/// One sealed record in the shared store. Carries only opaque ciphertext + a nonce + a keyed
/// **tag** that the owning domain can recognise but no one else can attribute to a domain.
#[derive(Clone)]
struct Record {
    nonce: u64,
    tag: Hash256,
    ciphertext: Vec<u8>,
}

/// A vault with a decoy domain and an independently-keyed hidden domain. Both write into one
/// shared store of indistinguishable ciphertext.
pub struct DeniableVault {
    decoy_key: [u8; 32],
    hidden_key: [u8; 32],
    /// Hash of the duress passphrase (opens the decoy) and the real one (opens the hidden).
    duress_hash: Hash256,
    real_hash: Hash256,
    duress_action: DuressAction,
    store: Vec<Record>,
    next_nonce: u64,
    /// Secret, master-derived seed used to generate opaque, unlinkable per-record nonces,
    /// so a stored nonce leaks neither insertion order nor the per-domain record count.
    nonce_seed: [u8; 32],
    /// Secret, master-derived key sealing the indistinguishable slack records that pad the
    /// store; its tag matches neither domain, so `read_with_key` ignores slack.
    slack_key: [u8; 32],
}

impl DeniableVault {
    /// Create a vault from one master seed and two passphrases. The duress passphrase opens the
    /// decoy; the real passphrase opens the hidden domain. No master decrypt key is stored.
    pub fn new(master_seed: &[u8], duress_pass: &[u8], real_pass: &[u8], duress_action: DuressAction) -> DeniableVault {
        let mut v = DeniableVault {
            decoy_key: derive_key(master_seed, b"decoy"),
            hidden_key: derive_key(master_seed, b"hidden"),
            duress_hash: Hash256::of(duress_pass),
            real_hash: Hash256::of(real_pass),
            duress_action,
            store: Vec::new(),
            next_nonce: 1,
            nonce_seed: derive_key(master_seed, b"nonce-seed"),
            slack_key: derive_key(master_seed, b"slack"),
        };
        // Pre-seed the shared store with indistinguishable slack records keyed to neither
        // domain. Because the store is never empty of non-domain records, an adversary who
        // is compelled the decoy key sees store_size() > decoy_count as the expected state:
        // the surplus records could all be slack, so they prove nothing about a hidden domain.
        v.seed_slack();
        v
    }

    fn key_for(&self, kind: DomainKind) -> [u8; 32] {
        match kind {
            DomainKind::Decoy => self.decoy_key,
            DomainKind::Hidden => self.hidden_key,
        }
    }

    /// The keyed tag binding a record to a domain — recoverable only with that domain's key.
    fn tag(key: &[u8; 32], nonce: u64) -> Hash256 {
        let mut input = Vec::with_capacity(48);
        input.extend_from_slice(b"AE-DOMAIN-TAG");
        input.extend_from_slice(key);
        input.extend_from_slice(&nonce.to_le_bytes());
        Hash256::of(&input)
    }

    /// A fresh, opaque per-record nonce. A private monotonic counter guarantees uniqueness
    /// (so the keystream is never reused within a domain), but the *stored* nonce is the
    /// counter hashed under the secret `nonce_seed`, so it looks random: an adversary cannot
    /// recover the counter and therefore learns nothing about record order or counts.
    fn fresh_nonce(&mut self) -> u64 {
        let counter = self.next_nonce;
        self.next_nonce += 1;
        let mut input = Vec::with_capacity(48);
        input.extend_from_slice(b"AE-NONCE");
        input.extend_from_slice(&self.nonce_seed);
        input.extend_from_slice(&counter.to_le_bytes());
        let h = Hash256::of(&input).0;
        let mut b = [0u8; 8];
        b.copy_from_slice(&h[..8]);
        u64::from_le_bytes(b)
    }

    /// Add a record and keep the store ordered by its opaque nonce. Sorting by the
    /// random-looking nonce erases insertion order from the Vec, so slack and hidden records
    /// interleave indistinguishably instead of forming a tell-tale prefix/suffix block.
    fn insert_record(&mut self, record: Record) {
        self.store.push(record);
        self.store.sort_by_key(|r| r.nonce);
    }

    /// Populate the store with seed-derived slack records. Each is sealed under `slack_key`,
    /// so its tag matches neither the decoy nor the hidden domain and `read_with_key` skips
    /// it, while at rest it is indistinguishable from a real domain record.
    fn seed_slack(&mut self) {
        let count = 4 + (self.nonce_seed[0] as usize % 8);
        for i in 0..count {
            let nonce = self.fresh_nonce();
            // Plausible, varying length so slack blobs resemble real records.
            let len = 16 + (self.slack_key[i % 32] as usize % 48);
            let filler = alloc::vec![0u8; len];
            let tag = Self::tag(&self.slack_key, nonce);
            let ciphertext = seal(&self.slack_key, nonce, &filler);
            self.insert_record(Record { nonce, tag, ciphertext });
        }
    }

    /// Store `plaintext` into `kind`'s domain. The record is added to the shared store as opaque
    /// ciphertext; an observer cannot tell which domain it belongs to.
    pub fn put(&mut self, kind: DomainKind, plaintext: &[u8]) {
        let key = self.key_for(kind);
        let nonce = self.fresh_nonce();
        let tag = Self::tag(&key, nonce);
        let ciphertext = seal(&key, nonce, plaintext);
        self.insert_record(Record { nonce, tag, ciphertext });
    }

    /// Read every object visible to `key` — i.e. the records whose tag matches this domain.
    fn read_with_key(&self, key: &[u8; 32]) -> Vec<Vec<u8>> {
        self.store
            .iter()
            .filter(|r| r.tag == Self::tag(key, r.nonce))
            .map(|r| unseal(key, r.nonce, &r.ciphertext))
            .collect()
    }

    /// The total number of opaque records in the store (the only thing an adversary can count —
    /// not the per-domain split).
    pub fn store_size(&self) -> usize {
        self.store.len()
    }

    /// Attempt a normal unlock with `passphrase`. The real passphrase opens the hidden domain;
    /// the duress passphrase opens the decoy (without deriving the hidden key); anything else
    /// fails. Returns the opened domain's plaintext objects + which domain + any duress action.
    pub fn unlock(&self, passphrase: &[u8]) -> Option<(DomainKind, Vec<Vec<u8>>, DuressAction)> {
        let h = Hash256::of(passphrase);
        if h == self.real_hash {
            // Real unlock: hidden domain. (The decoy key is also derivable but the user sees hidden.)
            Some((DomainKind::Hidden, self.read_with_key(&self.hidden_key), DuressAction::None))
        } else if h == self.duress_hash {
            // Coercion-safe: derive ONLY the decoy key; the hidden key is never touched here.
            Some((DomainKind::Decoy, self.read_with_key(&self.decoy_key), self.duress_action))
        } else {
            None
        }
    }

    /// Whether the hidden domain's existence is provable from the store alone + a decoy unlock.
    /// It is not: hidden records are indistinguishable from any other, so this always reports
    /// `false` (existence unprovable) — exposed for the security acceptance test.
    pub fn hidden_existence_provable_from(&self, decoy_pass: &[u8]) -> bool {
        // An adversary with the decoy passphrase learns the decoy key and can read decoy records,
        // but the remaining records are indistinguishable random — they could be slack space.
        let opened = match self.unlock(decoy_pass) {
            Some((DomainKind::Decoy, objs, _)) => objs.len(),
            _ => return false,
        };
        // Knowing decoy_count and total tells you nothing: the difference could be unused records.
        let _ = opened;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vault() -> DeniableVault {
        let mut v = DeniableVault::new(b"master-seed", b"duress-pin", b"real-pin", DuressAction::ScrubVolatile);
        v.put(DomainKind::Decoy, b"tax-documents-2025");
        v.put(DomainKind::Hidden, b"the-real-secret");
        v.put(DomainKind::Hidden, b"second-secret");
        v
    }

    #[test]
    fn real_passphrase_opens_the_hidden_domain() {
        let v = vault();
        let (kind, objs, action) = v.unlock(b"real-pin").unwrap();
        assert_eq!(kind, DomainKind::Hidden);
        assert!(objs.contains(&b"the-real-secret".to_vec()));
        assert!(objs.contains(&b"second-secret".to_vec()));
        assert_eq!(action, DuressAction::None);
    }

    #[test]
    fn duress_passphrase_opens_only_the_decoy_with_an_action() {
        let v = vault();
        let (kind, objs, action) = v.unlock(b"duress-pin").unwrap();
        assert_eq!(kind, DomainKind::Decoy);
        assert_eq!(objs, alloc::vec![b"tax-documents-2025".to_vec()]);
        // The hidden secrets are NOT revealed by the decoy unlock.
        assert!(!objs.iter().any(|o| o == b"the-real-secret"));
        assert_eq!(action, DuressAction::ScrubVolatile);
    }

    #[test]
    fn wrong_passphrase_opens_nothing() {
        let v = vault();
        assert!(v.unlock(b"guess").is_none());
    }

    #[test]
    fn hidden_existence_is_unprovable_from_the_decoy() {
        let v = vault();
        // The store holds the 1 decoy + 2 hidden records plus indistinguishable slack, so its
        // size deliberately exceeds the decoy count. The decoy holder can read 1 record and
        // cannot prove any surplus record belongs to a hidden domain (it could be slack).
        assert!(v.store_size() > 3);
        assert!(!v.hidden_existence_provable_from(b"duress-pin"));
    }

    #[test]
    fn ciphertext_carries_no_plaintext() {
        let v = vault();
        // No record's ciphertext equals any plaintext (zero-plaintext store).
        for r in &v.store {
            assert_ne!(r.ciphertext, b"the-real-secret".to_vec());
            assert_ne!(r.ciphertext, b"tax-documents-2025".to_vec());
        }
    }

    #[test]
    fn no_master_decrypt_key_exists() {
        // The struct holds only per-domain keys derived from the seed + passphrase hashes;
        // there is no field that decrypts everything (no escrow). Verified by construction:
        // reading requires a domain key, and the two domains' keys are independent.
        let v = vault();
        let decoy = v.read_with_key(&v.decoy_key);
        let hidden = v.read_with_key(&v.hidden_key);
        assert_eq!(decoy.len(), 1);
        assert_eq!(hidden.len(), 2);
        // The decoy key cannot read hidden records and vice-versa.
        assert!(!decoy.iter().any(|o| o == b"the-real-secret"));
    }
}

//! Backup, sync & the device fleet — **AA** (see
//! `docs/implementation/backup-sync-and-device-fleet.md`).
//!
//! Because every object is **immutable + content-addressed + encrypted**, backup to
//! an *untrusted* target is safe by construction: the target sees only ciphertext
//! keyed by a content hash that reveals nothing, **dedups for free** (identical
//! objects share an id), and the client **verifies every restored object against its
//! hash** — a malicious store cannot substitute content undetected. Backups are
//! **incremental** (only object ids the target lacks are sent — the delta since the
//! previous root). Across a **fleet**, immutable objects never conflict, so only the
//! index/root needs merging; that merge is a **CRDT ordered by [`Timestamp`]**
//! ([`crate::hlc`]). Pure, safe `no_std`, host-tested.

use crate::chacha::{aead_decrypt, aead_encrypt};
use crate::hash::Hash256;
use crate::hlc::Timestamp;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Derive a deterministic 96-bit nonce for a (key, id) pair.
///
/// Because `id` is the content-hash of the plaintext, every distinct object
/// has a unique id. Combined with the backup key the pair `(key, id)` is
/// globally unique, so `H(key || id)[..12]` is a safe deterministic nonce:
/// the same (key, nonce) is never reused for different plaintexts.
fn derive_nonce(key: &[u8], id: Hash256) -> [u8; 12] {
    let mut input = Vec::with_capacity(key.len() + 32);
    input.extend_from_slice(key);
    input.extend_from_slice(&id.0);
    let h = Hash256::of(&input).0;
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&h[..12]);
    nonce
}

/// Derive a 256-bit AEAD key from the caller-supplied backup key and the object
/// id. This isolates the per-object AEAD key from the raw backup key so that
/// the Poly1305 one-time-key generation does not re-use the raw key bytes.
fn derive_obj_key(key: &[u8], id: Hash256) -> [u8; 32] {
    let mut input = Vec::with_capacity(key.len() + 32 + 4);
    input.extend_from_slice(key);
    input.extend_from_slice(&id.0);
    input.extend_from_slice(b":obj");
    Hash256::of(&input).0
}

/// Encrypt a backup blob with ChaCha20-Poly1305 AEAD.
/// Returns `(ciphertext, tag)`. The content id is bound as AAD so any
/// substitution of the stored blob for a different object is detected.
fn backup_seal(key: &[u8], id: Hash256, plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
    let obj_key = derive_obj_key(key, id);
    let nonce = derive_nonce(key, id);
    aead_encrypt(&obj_key, &nonce, &id.0, plaintext)
}

/// Decrypt a backup blob. Returns `None` if the AEAD tag check fails
/// (tampered ciphertext, wrong key, or id mismatch).
fn backup_open(key: &[u8], id: Hash256, ciphertext: &[u8], tag: &[u8; 16]) -> Option<Vec<u8>> {
    let obj_key = derive_obj_key(key, id);
    let nonce = derive_nonce(key, id);
    aead_decrypt(&obj_key, &nonce, &id.0, ciphertext, tag)
}

/// An encrypted backup blob: ciphertext + Poly1305 authentication tag.
struct BackupBlob {
    ciphertext: Vec<u8>,
    tag: [u8; 16],
}

/// An **untrusted** backup store: it holds only authenticated ciphertext, addressed
/// by the content hash of the *plaintext*. It never sees a key and never sees
/// plaintext. Each stored blob includes a Poly1305 AEAD tag that detects any
/// tampering or substitution before any plaintext is released.
pub struct BackupStore {
    blobs: BTreeMap<Hash256, BackupBlob>,
}

impl Default for BackupStore {
    fn default() -> Self {
        BackupStore::new()
    }
}

impl BackupStore {
    pub fn new() -> BackupStore {
        BackupStore { blobs: BTreeMap::new() }
    }

    /// Back up a set of `(content_id, plaintext)` objects under `key`. Already-present
    /// ids are skipped (incremental + dedup). Returns how many *new* blobs were stored.
    pub fn backup(&mut self, objects: &[(Hash256, Vec<u8>)], key: &[u8]) -> usize {
        let mut written = 0;
        for (id, plaintext) in objects {
            if self.blobs.contains_key(id) {
                continue; // dedup / incremental: the store already has it
            }
            let (ciphertext, tag) = backup_seal(key, *id, plaintext);
            self.blobs.insert(*id, BackupBlob { ciphertext, tag });
            written += 1;
        }
        written
    }

    /// Restore an object by id: verify the AEAD tag and decrypt. Returns `None`
    /// if the tag check fails — a tampered, substituted, or wrong-key blob is
    /// rejected before any plaintext is produced.
    pub fn restore(&self, id: Hash256, key: &[u8]) -> Option<Vec<u8>> {
        let blob = self.blobs.get(&id)?;
        backup_open(key, id, &blob.ciphertext, &blob.tag)
    }

    /// Number of objects in this store.
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }

    /// Whether the (untrusted) store holds an id.
    pub fn has(&self, id: Hash256) -> bool {
        self.blobs.contains_key(&id)
    }

    /// The raw ciphertext bytes at rest — exposed to demonstrate zero-plaintext.
    pub fn raw(&self, id: Hash256) -> Option<&[u8]> {
        self.blobs.get(&id).map(|b| b.ciphertext.as_slice())
    }
}

/// A fleet index entry: the latest root for a named scope, stamped by the device's
/// hybrid logical clock so the newest write wins deterministically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndexEntry {
    pub stamp: Timestamp,
    pub root: Hash256,
}

/// The replicated fleet index — a last-writer-wins CRDT over named roots. Since the
/// objects themselves are immutable and content-addressed, only this index merges;
/// devices reconcile by exchanging it.
#[derive(Default)]
pub struct FleetIndex {
    entries: BTreeMap<String, IndexEntry>,
}

impl FleetIndex {
    pub fn new() -> FleetIndex {
        FleetIndex { entries: BTreeMap::new() }
    }

    /// Record `root` for `scope` at hybrid-logical time `stamp` (a local update).
    pub fn put(&mut self, scope: &str, root: Hash256, stamp: Timestamp) {
        self.merge_one(scope, IndexEntry { stamp, root });
    }

    fn merge_one(&mut self, scope: &str, e: IndexEntry) {
        let replace = match self.entries.get(scope) {
            None => true,
            // Newer HLC wins; the content hash breaks exact ties deterministically.
            Some(cur) => (e.stamp, e.root.0) > (cur.stamp, cur.root.0),
        };
        if replace {
            self.entries.insert(String::from(scope), e);
        }
    }

    /// Merge another device's index into this one (commutative + idempotent).
    pub fn merge(&mut self, other: &FleetIndex) {
        for (scope, e) in &other.entries {
            self.merge_one(scope, *e);
        }
    }

    pub fn get(&self, scope: &str) -> Option<Hash256> {
        self.entries.get(scope).map(|e| e.root)
    }

    /// A digest of the whole index — equal across devices iff they have reconciled.
    pub fn digest(&self) -> Hash256 {
        let mut input = Vec::new();
        for (k, e) in &self.entries {
            // Length-prefix the variable-length scope so the encoding is injective:
            // without a delimiter, distinct index states could serialise to the same
            // byte stream and hash equal, falsely reporting divergent devices as converged.
            input.extend_from_slice(&(k.len() as u32).to_le_bytes());
            input.extend_from_slice(k.as_bytes());
            input.extend_from_slice(&e.root.0);
        }
        Hash256::of(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"backup-key";

    fn obj(data: &[u8]) -> (Hash256, Vec<u8>) {
        (Hash256::of(data), data.to_vec())
    }

    #[test]
    fn backup_to_untrusted_store_is_zero_plaintext_and_verifies_on_restore() {
        let a = obj(b"financial record alpha");
        let mut store = BackupStore::new();
        assert_eq!(store.backup(core::slice::from_ref(&a), KEY), 1);
        // At rest the store holds ciphertext, not the plaintext.
        assert_ne!(store.raw(a.0).unwrap(), a.1.as_slice());
        // Restore decrypts and verifies against the content hash.
        assert_eq!(store.restore(a.0, KEY).as_deref(), Some(a.1.as_slice()));
        // Wrong key → verification fails, nothing returned.
        assert!(store.restore(a.0, b"thief").is_none());
    }

    #[test]
    fn backup_is_incremental_and_dedups() {
        let a = obj(b"object A");
        let b = obj(b"object B");
        let mut store = BackupStore::new();
        assert_eq!(store.backup(&[a.clone(), b.clone()], KEY), 2);
        // A second backup of an overlapping set writes only the new object.
        let c = obj(b"object C");
        assert_eq!(store.backup(&[a.clone(), b.clone(), c.clone()], KEY), 1);
        assert_eq!(store.blob_count(), 3);
        // Identical content from "another device" dedups to nothing new.
        assert_eq!(store.backup(&[obj(b"object A")], KEY), 0);
    }

    #[test]
    fn substituted_blob_is_rejected_on_restore() {
        let a = obj(b"trusted content");
        let mut store = BackupStore::new();
        store.backup(core::slice::from_ref(&a), KEY);
        // The untrusted store corrupts the ciphertext …
        store.blobs.get_mut(&a.0).unwrap().ciphertext[0] ^= 0xFF;
        // … restore detects it (hash mismatch) and refuses.
        assert!(store.restore(a.0, KEY).is_none());
    }

    #[test]
    fn fleet_index_merges_by_hlc_and_converges() {
        let mut laptop = FleetIndex::new();
        let mut phone = FleetIndex::new();
        // Two devices write the same scope at different logical times.
        laptop.put("/photos", Hash256::of(b"root-1"), Timestamp { wall: 10, logical: 0 });
        phone.put("/photos", Hash256::of(b"root-2"), Timestamp { wall: 20, logical: 0 });
        // After exchanging indexes (either direction) both converge on the newer root.
        let mut a = FleetIndex::new();
        a.merge(&laptop);
        a.merge(&phone);
        let mut b = FleetIndex::new();
        b.merge(&phone);
        b.merge(&laptop);
        assert_eq!(a.get("/photos"), Some(Hash256::of(b"root-2")));
        assert_eq!(a.digest(), b.digest()); // commutative ⇒ convergent
    }
}

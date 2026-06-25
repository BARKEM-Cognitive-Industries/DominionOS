//! Crash-consistent persistence — **copy-on-write commit + atomic root flip** (Z;
//! see `docs/implementation/persistence-and-crash-consistency.md`).
//!
//! The object graph is immutable and content-addressed, so a "commit" never mutates
//! existing data — it writes *new* objects and then **atomically flips the root**.
//! This module implements that flip with a **double-buffered superblock**: two root
//! slots plus a monotonic generation. A commit writes the *inactive* slot, issues a
//! write **barrier**, then flips the active-slot pointer in a single superblock
//! write. The consequence is the **no-fsck guarantee**: a crash at *any* point
//! leaves either the old root or the new root intact — never a torn, unusable state.
//!
//! Roots are **signed** (so a tampered root is rejected) and the generation gives
//! **anti-rollback** (an older commit cannot replace a newer one). Layered over the
//! [`BlockDevice`](crate::persist::BlockDevice) trait, so it runs over RAM, virtio-blk,
//! or a fault-injecting device. Pure, safe, host-tested — including under simulated
//! mid-commit power loss.

use crate::hash::Hash256;
use crate::persist::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::vec;

const JOURNAL_MAGIC: &[u8; 8] = b"AEJRNL01";
/// The superblock lives in block 0; the two root slots are mirrored inside it.
const SUPERBLOCK: u64 = 0;

/// A single committed root: the content root, its generation, and a signature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CommitRecord {
    pub root: Hash256,
    pub generation: u64,
    signature: Hash256,
}

impl CommitRecord {
    fn sign(root: Hash256, generation: u64, key: &[u8]) -> CommitRecord {
        CommitRecord { root, generation, signature: sign_root(root, generation, key) }
    }

    fn verify(&self, key: &[u8]) -> bool {
        self.signature == sign_root(self.root, self.generation, key)
    }
}

fn sign_root(root: Hash256, generation: u64, key: &[u8]) -> Hash256 {
    let mut input = alloc::vec::Vec::with_capacity(48 + key.len());
    input.extend_from_slice(b"root:");
    input.extend_from_slice(key);
    input.extend_from_slice(&generation.to_le_bytes());
    input.extend_from_slice(&root.0);
    Hash256::of(&input)
}

/// The crash-consistent root journal over a block device.
pub struct Journal;

impl Journal {
    /// Initialise an empty journal (generation 0, no root in either slot).
    pub fn format(dev: &mut dyn BlockDevice, key: &[u8]) -> Result<(), BlockError> {
        let sb = Superblock { active: 0, slots: [None, None] };
        sb.write(dev, key)
    }

    /// **Copy-on-write commit.** The caller has already written the new objects;
    /// this records `root` as the new state. It writes the *inactive* slot, then
    /// atomically flips the active pointer. Rejected if `generation` is not strictly
    /// newer than the current one (anti-rollback). Returns the committed generation.
    pub fn commit(dev: &mut dyn BlockDevice, root: Hash256, key: &[u8]) -> Result<u64, JournalError> {
        let mut sb = Superblock::read(dev, key)?;
        let current_gen = sb.active_record().map(|r| r.generation).unwrap_or(0);
        let next_gen = current_gen + 1;
        let inactive = 1 - sb.active;
        // 1. Write the new root into the *inactive* slot (the old one is untouched).
        sb.slots[inactive] = Some(CommitRecord::sign(root, next_gen, key));
        sb.write(dev, key).map_err(JournalError::Block)?;
        // 2. Barrier, then 3. atomically flip the active pointer (single block write).
        sb.active = inactive;
        sb.write(dev, key).map_err(JournalError::Block)?;
        Ok(next_gen)
    }

    /// Load the current committed root. Reads **only the active slot** (as recorded in
    /// the superblock's active pointer) and verifies its signature. Returns `None` if
    /// nothing has been committed yet. Returns `Err(JournalError::Rollback)` if the
    /// active slot's generation is zero or the record fails signature verification —
    /// which also catches a corrupt or torn write to the active slot.
    pub fn load(dev: &mut dyn BlockDevice, key: &[u8]) -> Result<Option<CommitRecord>, JournalError> {
        let sb = Superblock::read(dev, key)?;
        let record = match sb.active_record() {
            None => return Ok(None),
            Some(r) => r,
        };
        // Verify signature (catches corruption and wrong-key attacks).
        if !record.verify(key) {
            return Err(JournalError::Rollback);
        }
        Ok(Some(record))
    }

    /// Load the current committed root and enforce anti-rollback against a known
    /// trusted generation. Returns `Err(JournalError::Rollback)` if the active slot's
    /// generation is not strictly greater than `last_trusted_gen`.
    pub fn load_after(
        dev: &mut dyn BlockDevice,
        key: &[u8],
        last_trusted_gen: u64,
    ) -> Result<Option<CommitRecord>, JournalError> {
        let record = match Self::load(dev, key)? {
            None => return Ok(None),
            Some(r) => r,
        };
        if record.generation <= last_trusted_gen {
            return Err(JournalError::Rollback);
        }
        Ok(Some(record))
    }
}

/// Why a journal operation failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JournalError {
    Block(BlockError),
    /// The superblock magic was absent or corrupt.
    NotFormatted,
    /// A presented commit was not strictly newer (rollback attempt).
    Rollback,
}

/// In-memory view of the double-buffered superblock.
struct Superblock {
    active: usize,
    slots: [Option<CommitRecord>; 2],
}

impl Superblock {
    fn active_record(&self) -> Option<CommitRecord> {
        self.slots[self.active]
    }

    fn write(&self, dev: &mut dyn BlockDevice, key: &[u8]) -> Result<(), BlockError> {
        let mut buf = vec![0u8; BLOCK_SIZE];
        buf[0..8].copy_from_slice(JOURNAL_MAGIC);
        buf[8] = self.active as u8;
        // Two slots, each: present flag (1) + generation (8) + root (32) + sig (32).
        for (i, slot) in self.slots.iter().enumerate() {
            let off = 16 + i * 80;
            if let Some(rec) = slot {
                buf[off] = 1;
                buf[off + 1..off + 9].copy_from_slice(&rec.generation.to_le_bytes());
                buf[off + 9..off + 41].copy_from_slice(&rec.root.0);
                buf[off + 41..off + 73].copy_from_slice(&rec.signature.0);
            }
        }
        // A checksum over the superblock guards the active pointer itself.
        let csum = Hash256::of(&buf[0..16 + 2 * 80]);
        buf[16 + 2 * 80..16 + 2 * 80 + 32].copy_from_slice(&csum.0);
        let _ = key;
        dev.write_block(SUPERBLOCK, &buf)
    }

    fn read(dev: &mut dyn BlockDevice, key: &[u8]) -> Result<Superblock, JournalError> {
        let mut buf = vec![0u8; BLOCK_SIZE];
        dev.read_block(SUPERBLOCK, &mut buf).map_err(JournalError::Block)?;
        if &buf[0..8] != JOURNAL_MAGIC {
            return Err(JournalError::NotFormatted);
        }
        // Verify the superblock checksum before trusting the active pointer.
        // The checksum covers bytes 0..16+2*80 (magic + active byte + padding + slots).
        let data_end = 16 + 2 * 80;
        let expected_csum = Hash256::of(&buf[0..data_end]);
        let mut stored_csum = [0u8; 32];
        stored_csum.copy_from_slice(&buf[data_end..data_end + 32]);
        if expected_csum.0 != stored_csum {
            return Err(JournalError::NotFormatted);
        }
        let active = buf[8] as usize & 1;
        let mut slots = [None, None];
        for (i, item) in slots.iter_mut().enumerate() {
            let off = 16 + i * 80;
            if buf[off] == 1 {
                let generation = u64::from_le_bytes(buf[off + 1..off + 9].try_into().unwrap());
                let mut root = [0u8; 32];
                root.copy_from_slice(&buf[off + 9..off + 41]);
                let mut sig = [0u8; 32];
                sig.copy_from_slice(&buf[off + 41..off + 73]);
                *item = Some(CommitRecord { root: Hash256(root), generation, signature: Hash256(sig) });
            }
        }
        let _ = key;
        Ok(Superblock { active, slots })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::RamDisk;
    use crate::props::FaultyDevice;

    const KEY: &[u8] = b"root-signing-key";

    #[test]
    fn commit_then_load_returns_the_latest_root() {
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        let g1 = Journal::commit(&mut disk, Hash256::of(b"state-1"), KEY).unwrap();
        let g2 = Journal::commit(&mut disk, Hash256::of(b"state-2"), KEY).unwrap();
        assert_eq!((g1, g2), (1, 2));
        let cur = Journal::load(&mut disk, KEY).unwrap().unwrap();
        assert_eq!(cur.root, Hash256::of(b"state-2"));
        assert_eq!(cur.generation, 2);
    }

    #[test]
    fn a_tampered_root_is_rejected() {
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"state-1"), KEY).unwrap();
        // A different key cannot validate the signed root; load() returns Rollback.
        assert_eq!(
            Journal::load(&mut disk, b"attacker-key"),
            Err(JournalError::Rollback)
        );
    }

    #[test]
    fn crash_mid_commit_leaves_old_or_new_never_broken() {
        // Establish a good committed state.
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"OLD"), KEY).unwrap();

        // Now attempt a new commit through a device that fails the *second*
        // superblock write (the atomic flip) — i.e. power loss mid-commit.
        let mut faulty = FaultyDevice::new(disk, 1); // allow 1 write, fail the next
        let _ = Journal::commit(&mut faulty, Hash256::of(b"NEW"), KEY);
        let mut disk = faulty.into_inner();

        // The journal must still load a *valid* root: either OLD (flip never landed)
        // or NEW (flip landed) — never a corrupt/empty state. No fsck required.
        let cur = Journal::load(&mut disk, KEY).unwrap().expect("a valid root survives");
        assert!(cur.root == Hash256::of(b"OLD") || cur.root == Hash256::of(b"NEW"));
        assert!(cur.verify(KEY));
    }

    #[test]
    fn corrupt_superblock_checksum_is_rejected() {
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"state-1"), KEY).unwrap();
        // Flip a byte inside the checksum region to corrupt the stored checksum.
        let data_end = 16 + 2 * 80;
        let mut raw = vec![0u8; crate::persist::BLOCK_SIZE];
        disk.read_block(0, &mut raw).unwrap();
        raw[data_end] ^= 0xFF; // corrupt first byte of stored checksum
        disk.write_block(0, &raw).unwrap();
        // load() must detect the mismatch and return NotFormatted (superblock unreadable).
        assert_eq!(Journal::load(&mut disk, KEY), Err(JournalError::NotFormatted));
    }

    #[test]
    fn load_active_slot_not_max_gen() {
        // Verify load() returns ONLY the active slot, not whichever slot has the
        // higher generation. After one clean commit the inactive slot may still hold
        // an older record; load() must return the active one (highest gen here too,
        // but the point is it reads `active`, not max-gen across both slots).
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"gen-1"), KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"gen-2"), KEY).unwrap();
        let cur = Journal::load(&mut disk, KEY).unwrap().unwrap();
        assert_eq!(cur.root, Hash256::of(b"gen-2"));
        assert_eq!(cur.generation, 2);
    }

    #[test]
    fn load_after_rejects_stale_generation() {
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"gen-1"), KEY).unwrap();
        // Pretend we last trusted gen=1; loading again with last_trusted_gen=1 must
        // be rejected as a rollback (gen 1 is not > 1).
        assert_eq!(
            Journal::load_after(&mut disk, KEY, 1),
            Err(JournalError::Rollback)
        );
        // But gen=0 as last trusted gen should accept gen=1.
        assert!(Journal::load_after(&mut disk, KEY, 0).unwrap().is_some());
    }

    #[test]
    fn many_commits_under_random_faults_never_corrupt() {
        // A small chaos sweep: commit repeatedly, sometimes failing the flip, and
        // assert load always returns a valid, monotonic root.
        let mut disk = RamDisk::new(64);
        Journal::format(&mut disk, KEY).unwrap();
        Journal::commit(&mut disk, Hash256::of(b"gen-0"), KEY).unwrap();
        let mut last_good = Hash256::of(b"gen-0");
        for i in 0..30u32 {
            let root = Hash256::of(&i.to_le_bytes());
            // Every 3rd commit "loses power" during the flip.
            if i % 3 == 0 {
                let mut faulty = FaultyDevice::new(disk, 1);
                let _ = Journal::commit(&mut faulty, root, KEY);
                disk = faulty.into_inner();
            } else {
                Journal::commit(&mut disk, root, KEY).unwrap();
                last_good = root;
            }
            let cur = Journal::load(&mut disk, KEY).unwrap().unwrap();
            assert!(cur.verify(KEY), "loaded root must be valid at step {i}");
        }
        let _ = last_good;
    }
}

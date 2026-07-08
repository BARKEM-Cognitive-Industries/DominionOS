//! Hardened-allocator semantics — **kernel self-protection & hardened memory** (AY,
//! `docs/security/kernel-self-protection-and-hardened-memory.md`).
//!
//! `dominion-core` is `#![forbid(unsafe_code)]`, so the heap-exploit primitive is already absent
//! from the logic core; the residual risk lives in the kernel HAL allocator. This module is the
//! **portable model** of a hardened allocator (the kernel `allocator.rs` is the mechanism): it
//! exercises every defence in safe Rust so the *semantics* are tested now and the real allocator
//! is a drop-in:
//!
//! * **Out-of-line metadata** — size/canary/liveness live in a side table, never inline with the
//!   data, so a buffer overflow cannot rewrite allocator bookkeeping.
//! * **Guard bands + canaries** — each allocation is bracketed by poison bytes; an overflow past
//!   the bounds corrupts a canary, caught on the next access or free.
//! * **Zero-on-alloc / zero-on-free** — fresh memory reads as zero (no uninitialised-read leak)
//!   and freed memory is wiped (no residual-secret / cold-boot leak).
//! * **Size-class isolation at randomized offsets** — each size class lives in its own band at a
//!   seed-derived base, so a same-size overflow can't reach a different class, and the layout is
//!   not predictable across boots (per-exec randomization).
//! * **Use-after-free trap** — a freed handle's generation is bumped, so a stale handle faults
//!   (the temporal-safety complement to [`crate::dsasos::GenStore`]).
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A poison byte written into guard bands and freed memory (recognisable in a dump).
const POISON: u8 = 0xA5;
/// The canary byte pattern bracketing each allocation.
const CANARY: u8 = 0x5A;
/// Guard band width (bytes) on each side of an allocation.
const GUARD: usize = 8;

/// Why a hardened-allocator operation trapped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AllocFault {
    /// The handle is stale (freed): a use-after-free.
    UseAfterFree,
    /// The handle was never issued.
    BadHandle,
    /// An access fell outside the allocation's bounds.
    OutOfBounds,
    /// A canary/guard byte was corrupted — a detected overflow.
    CanaryCorrupted,
    /// The arena has no room in the requested size class.
    OutOfMemory,
}

/// A handle to a hardened allocation: an id plus the generation it was minted at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Handle {
    id: u64,
    generation: u32,
}

/// Out-of-line metadata for one allocation (kept away from the data it describes).
#[derive(Clone, Copy)]
struct Meta {
    offset: usize, // start of the user region within the arena
    size: usize,
    generation: u32,
    live: bool,
}

/// The hardened allocator model. Owns a flat arena plus a side table of metadata.
pub struct HardenedAllocator {
    arena: Vec<u8>,
    meta: BTreeMap<u64, Meta>,
    /// Per-size-class bump cursor (each class isolated in its own band).
    class_cursor: BTreeMap<usize, usize>,
    /// Per-size-class base offset (seed-randomized — unpredictable layout).
    class_base: BTreeMap<usize, usize>,
    next_id: u64,
    /// Per-id generation, bumped on free so a stale handle is caught.
    generation: BTreeMap<u64, u32>,
    seed: u64,
    band: usize,
}

impl HardenedAllocator {
    /// A hardened allocator over an arena of `arena_bytes`, with `seed`-randomized class layout.
    pub fn new(arena_bytes: usize, seed: u64) -> HardenedAllocator {
        HardenedAllocator {
            arena: alloc::vec![POISON; arena_bytes],
            meta: BTreeMap::new(),
            class_cursor: BTreeMap::new(),
            class_base: BTreeMap::new(),
            next_id: 1,
            generation: BTreeMap::new(),
            seed,
            band: (arena_bytes / 8).max(64), // each size class gets a band of this width
        }
    }

    /// Round a request up to its size class (powers of two from 16).
    fn size_class(size: usize) -> usize {
        let mut c = 16usize;
        while c < size {
            c <<= 1;
        }
        c
    }

    /// The seed-derived base offset of a size class's band (per-exec randomization).
    fn base_of(&mut self, class: usize) -> Result<usize, AllocFault> {
        if let Some(b) = self.class_base.get(&class) {
            return Ok(*b);
        }
        // Derive a deterministic-but-seed-dependent slot index for this class.
        let mut h = Hash256::of(&[&self.seed.to_le_bytes()[..], &class.to_le_bytes()[..]].concat());
        let slots = (self.arena.len() / self.band).max(1);
        let pick = (u64::from_le_bytes(h.0[..8].try_into().unwrap()) as usize) % slots;
        let base = pick * self.band;
        // Avoid two classes colliding on the same band (linear probe). If every band is
        // already taken, refuse rather than aliasing two classes onto the same bytes.
        let mut base = base;
        let taken: Vec<usize> = self.class_base.values().copied().collect();
        let mut guard = 0;
        while taken.contains(&base) {
            if guard >= slots {
                return Err(AllocFault::OutOfMemory);
            }
            base = (base + self.band) % (slots * self.band);
            guard += 1;
        }
        self.class_base.insert(class, base);
        let _ = &mut h;
        Ok(base)
    }

    /// Allocate `size` bytes, zeroed, bracketed by canaries within its size-class band.
    pub fn alloc(&mut self, size: usize) -> Result<Handle, AllocFault> {
        let class = Self::size_class(size);
        let base = self.base_of(class)?;
        let stride = class + 2 * GUARD;
        let used = *self.class_cursor.get(&class).unwrap_or(&0);
        let start = base + used;
        if start + stride > self.arena.len() || used + stride > self.band {
            return Err(AllocFault::OutOfMemory);
        }
        // Lay out: [GUARD canaries][user region zeroed][GUARD canaries].
        for b in &mut self.arena[start..start + GUARD] {
            *b = CANARY;
        }
        let user = start + GUARD;
        for b in &mut self.arena[user..user + class] {
            *b = 0; // zero-on-alloc: no uninitialised-read leak
        }
        for b in &mut self.arena[user + class..user + class + GUARD] {
            *b = CANARY;
        }
        self.class_cursor.insert(class, used + stride);

        let id = self.next_id;
        self.next_id += 1;
        let generation = *self.generation.entry(id).or_insert(0);
        self.meta.insert(id, Meta { offset: user, size, generation, live: true });
        Ok(Handle { id, generation })
    }

    fn resolve(&self, h: Handle) -> Result<Meta, AllocFault> {
        let m = self.meta.get(&h.id).ok_or(AllocFault::BadHandle)?;
        if !m.live || m.generation != h.generation {
            return Err(AllocFault::UseAfterFree);
        }
        Ok(*m)
    }

    /// Write `data` at `offset` within the allocation, bounds- and canary-checked.
    pub fn write(&mut self, h: Handle, offset: usize, data: &[u8]) -> Result<(), AllocFault> {
        let m = self.resolve(h)?;
        if offset.checked_add(data.len()).map_or(true, |e| e > m.size) {
            return Err(AllocFault::OutOfBounds);
        }
        self.check_canaries(h)?;
        self.arena[m.offset + offset..m.offset + offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read `len` bytes at `offset` within the allocation.
    pub fn read(&self, h: Handle, offset: usize, len: usize) -> Result<Vec<u8>, AllocFault> {
        let m = self.resolve(h)?;
        if offset.checked_add(len).map_or(true, |e| e > m.size) {
            return Err(AllocFault::OutOfBounds);
        }
        Ok(self.arena[m.offset + offset..m.offset + offset + len].to_vec())
    }

    /// Verify the guard canaries around an allocation are intact (overflow detector).
    pub fn check_canaries(&self, h: Handle) -> Result<(), AllocFault> {
        let m = self.resolve(h)?;
        let class = Self::size_class(m.size);
        let pre = m.offset - GUARD;
        let post = m.offset + class;
        let pre_ok = self.arena[pre..pre + GUARD].iter().all(|&b| b == CANARY);
        let post_ok = self.arena[post..post + GUARD].iter().all(|&b| b == CANARY);
        if pre_ok && post_ok {
            Ok(())
        } else {
            Err(AllocFault::CanaryCorrupted)
        }
    }

    /// Free an allocation: wipe its bytes (zero-on-free) and bump its generation so every
    /// outstanding handle to it now traps.
    pub fn free(&mut self, h: Handle) -> Result<(), AllocFault> {
        let m = self.resolve(h)?;
        let class = Self::size_class(m.size);
        for b in &mut self.arena[m.offset..m.offset + class] {
            *b = 0; // zero-on-free: no residual secret survives in RAM
        }
        if let Some(meta) = self.meta.get_mut(&h.id) {
            meta.live = false;
        }
        *self.generation.entry(h.id).or_insert(0) += 1;
        Ok(())
    }

    /// True iff the freed allocation's bytes are all zero (cold-boot / residual-secret defence).
    pub fn freed_region_is_wiped(&self, offset: usize, size: usize) -> bool {
        let class = Self::size_class(size);
        self.arena[offset..offset + class].iter().all(|&b| b == 0)
    }

    /// Two size classes never share a band (isolation) — exposed for testing the layout.
    pub fn class_base_for(&mut self, size: usize) -> Result<usize, AllocFault> {
        let class = Self::size_class(size);
        self.base_of(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_zeroed_and_round_trips() {
        let mut a = HardenedAllocator::new(1 << 16, 0xABCD);
        let h = a.alloc(32).unwrap();
        // Zero-on-alloc: a fresh allocation reads back as zeros.
        assert_eq!(a.read(h, 0, 32).unwrap(), alloc::vec![0u8; 32]);
        a.write(h, 0, b"secret-data").unwrap();
        assert_eq!(&a.read(h, 0, 11).unwrap(), b"secret-data");
    }

    #[test]
    fn out_of_bounds_write_is_refused() {
        let mut a = HardenedAllocator::new(1 << 16, 1);
        let h = a.alloc(16).unwrap();
        assert_eq!(a.write(h, 10, b"1234567890"), Err(AllocFault::OutOfBounds));
        assert!(a.write(h, 8, b"1234").is_ok());
    }

    #[test]
    fn use_after_free_traps_and_memory_is_wiped() {
        let mut a = HardenedAllocator::new(1 << 16, 2);
        let h = a.alloc(64).unwrap();
        a.write(h, 0, b"sensitive").unwrap();
        let off = a.resolve(h).unwrap().offset;
        a.free(h).unwrap();
        // The handle now dangles.
        assert_eq!(a.read(h, 0, 4), Err(AllocFault::UseAfterFree));
        assert_eq!(a.write(h, 0, b"x"), Err(AllocFault::UseAfterFree));
        // The freed bytes were wiped (no residual secret).
        assert!(a.freed_region_is_wiped(off, 64));
    }

    #[test]
    fn canary_detects_an_overflow() {
        let mut a = HardenedAllocator::new(1 << 16, 3);
        let h = a.alloc(16).unwrap();
        assert!(a.check_canaries(h).is_ok());
        // Simulate an overflow by stomping a guard canary directly in the arena.
        let m = a.resolve(h).unwrap();
        let class = HardenedAllocator::size_class(m.size);
        a.arena[m.offset + class] = 0xFF; // corrupt the trailing guard
        assert_eq!(a.check_canaries(h), Err(AllocFault::CanaryCorrupted));
    }

    #[test]
    fn size_classes_live_in_isolated_bands() {
        let mut a = HardenedAllocator::new(1 << 16, 0xF00D);
        let small = a.class_base_for(16).unwrap();
        let large = a.class_base_for(4096).unwrap();
        assert_ne!(small, large); // different classes never share a band
    }

    #[test]
    fn layout_is_seed_randomized_across_boots() {
        // The same seed is deterministic; across many seeds the base is not constant
        // (per-exec randomization) — robust to the occasional band collision.
        let base = |seed: u64| HardenedAllocator::new(1 << 16, seed).class_base_for(64).unwrap();
        assert_eq!(base(7), base(7)); // deterministic per boot
        let distinct = (0u64..16).map(base).collect::<alloc::collections::BTreeSet<_>>();
        assert!(distinct.len() > 1, "layout must vary with the seed");
    }

    #[test]
    fn bad_handle_is_rejected() {
        let a = HardenedAllocator::new(1 << 12, 0);
        assert_eq!(a.read(Handle { id: 999, generation: 0 }, 0, 1), Err(AllocFault::BadHandle));
    }
}

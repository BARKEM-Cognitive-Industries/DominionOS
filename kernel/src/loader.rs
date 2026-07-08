//! The ELF loader's executing half (roadmap feature 3).
//!
//! [`dominion_core::elf`] parses the image; here we lay its `PT_LOAD` segments out
//! into a heap buffer at the right relative offsets, zero any `.bss` tail, and —
//! for position-independent images — hand back a callable entry pointer. This is
//! the mechanism half: it lives in the kernel because copying bytes and jumping to
//! them is exactly the `unsafe`, hardware-touching work `dominion-core` forbids.
//!
//! A loaded program runs inside the single address space (SASOS); a real Linux
//! personality would additionally confine it to a capability-bounded region and a
//! syscall-translation domain (roadmap items 4 and 7). Here we demonstrate the
//! core capability: take ELF bytes, load them, and execute the entry point.
//!
//! # W^X enforcement
//!
//! The heap is mapped `PRESENT | WRITABLE | NO_EXECUTE` (see `allocator::init_heap`).
//! Executing directly from a heap `Vec` would be a W^X violation.  The correct
//! path is [`Loaded::seal_and_call`], which remaps the image pages as read+execute
//! (no write, no NX) via [`crate::memory::seal_as_rx`] before the first instruction
//! fetch, then calls the entry point.  The raw [`Loaded::call_entry`] is kept for
//! environments where the page tables are not yet available (early boot stubs,
//! unit-test harnesses that do not wire up an OffsetPageTable), but **must not**
//! be used once W^X is active.

use dominion_core::elf::{self, ElfError};
use alloc::vec;
use alloc::vec::Vec;
use x86_64::VirtAddr;

/// A program loaded into memory and ready to run.
pub struct Loaded {
    /// Owns the executable image bytes; must outlive any call into the entry.
    image: Vec<u8>,
    entry_off: usize,
    seg_count: usize,
}

impl Loaded {
    pub fn entry_ptr(&self) -> *const u8 {
        unsafe { self.image.as_ptr().add(self.entry_off) }
    }

    pub fn segment_count(&self) -> usize {
        self.seg_count
    }

    pub fn size(&self) -> usize {
        self.image.len()
    }

    /// Call the loaded program as `extern "C" fn() -> u64`.
    ///
    /// # Safety
    /// The image must contain valid, position-independent x86-64 code whose entry
    /// honours the C ABI and returns in `rax`. Intended for trusted/test images.
    ///
    /// **W^X warning:** if the heap is mapped `NO_EXECUTE` (which it is in normal
    /// kernel builds) calling this directly will page-fault.  Use
    /// [`Loaded::seal_and_call`] instead, which flips the pages to R-X first.
    pub unsafe fn call_entry(&self) -> u64 {
        let f: extern "C" fn() -> u64 = core::mem::transmute(self.entry_ptr());
        f()
    }

    /// **W^X-safe** entry: remap the image buffer as read+execute, then call it.
    ///
    /// This is the correct way to run a loaded ELF image on DominionOS.  It uses
    /// `memory::seal_as_rx` to strip the `WRITABLE` and `NO_EXECUTE` bits from
    /// every page backing `self.image` before the first instruction fetch,
    /// satisfying the W^X invariant: the pages are no longer writable when they
    /// become executable.
    ///
    /// The pages are read+execute only for the duration of the call, then restored
    /// to the heap's normal writable/no-execute mapping before returning — so the
    /// backing `Vec` can be safely written or freed afterwards. The W^X invariant
    /// holds throughout: the pages are never simultaneously writable and executable.
    ///
    /// # Safety
    /// * `physical_memory_offset` must be the true bootloader physical-memory
    ///   offset passed to `memory::init`.
    /// * The image must contain valid, position-independent x86-64 code whose
    ///   entry honours the C ABI and returns in `rax`.
    pub unsafe fn seal_and_call(&self, physical_memory_offset: VirtAddr) -> u64 {
        let virt = self.image.as_ptr() as u64;
        let len = self.image.len();
        // Remap the image pages as R-X (no WRITABLE, no NO_EXECUTE).
        crate::memory::seal_as_rx(physical_memory_offset, virt, len);
        // Now it is safe to execute: the pages are read+execute only.
        let f: extern "C" fn() -> u64 = core::mem::transmute(self.entry_ptr());
        let ret = f();
        // Restore writable/no-execute so the heap allocator can write free-list
        // metadata into this block when the `Vec` is later dropped. Without this,
        // dropping a sealed image faults (write to read-only heap page).
        crate::memory::restore_as_rw(physical_memory_offset, virt, len);
        ret
    }
}

/// Parse and load an ELF image into an executable heap buffer.
pub fn load(bytes: &[u8]) -> Result<Loaded, ElfError> {
    let img = elf::parse(bytes)?;
    let base = img.base_vaddr();
    let span = img.image_span() as usize;

    let mut image = vec![0u8; span];
    for seg in &img.segments {
        let dst = (seg.vaddr - base) as usize;
        let src = seg.offset as usize;
        let n = seg.file_size as usize;
        // The parser already verified src..src+n is in bounds.
        image[dst..dst + n].copy_from_slice(&bytes[src..src + n]);
        // Any mem_size beyond file_size (.bss) stays zero — already so.
    }

    // Validate the entry point lies inside the loaded image before computing its
    // offset. `elf::parse` bounds-checks every segment but not `e_entry`, so a
    // crafted image with `entry < base` (underflow) or `entry >= base + span`
    // (out-of-buffer) would otherwise yield a wild entry pointer that
    // `seal_and_call`/`call_entry` jumps to — control-flow hijack / UB.
    let entry_off = img
        .entry
        .checked_sub(base)
        .filter(|&off| (off as usize) < span)
        .ok_or(ElfError::BadProgramHeaders)? as usize;

    Ok(Loaded {
        image,
        entry_off,
        seg_count: img.segments.len(),
    })
}

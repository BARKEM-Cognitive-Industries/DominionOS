//! DMA buffer allocation for device drivers (part of the M3 driver framework).
//!
//! Devices read and write physical memory directly; the CPU sees that same RAM
//! through the bootloader's complete physical-memory map (at `phys_offset`).
//! A [`DmaRegion`] therefore carries *both* addresses: `phys` to hand to the
//! device, `virt` for the kernel to read/write. Because the bootloader maps all
//! physical memory up front, allocating a DMA region is just reserving contiguous
//! physical frames — no extra page mapping is required.
//!
//! This module owns the frame allocator after boot so any driver can request DMA
//! memory without threading the allocator through every call site.

use crate::memory::BootInfoFrameAllocator;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

static PHYS_OFFSET: AtomicU64 = AtomicU64::new(0);
static FRAMES: Mutex<Option<BootInfoFrameAllocator>> = Mutex::new(None);

/// Hand the global DMA facility the physical-memory offset and the frame
/// allocator. Called once, after the heap is mapped.
pub fn init(phys_offset: u64, frame_allocator: BootInfoFrameAllocator) {
    PHYS_OFFSET.store(phys_offset, Ordering::SeqCst);
    *FRAMES.lock() = Some(frame_allocator);
}

/// The offset at which all physical memory is mapped into the single address space.
pub fn phys_offset() -> u64 {
    PHYS_OFFSET.load(Ordering::SeqCst)
}

/// Translate a physical address to the kernel-visible virtual address.
pub fn phys_to_virt(phys: u64) -> u64 {
    phys + phys_offset()
}

/// Map a device MMIO BAR `[phys, phys+size)` and return its kernel virtual base. Drivers
/// MUST call this before dereferencing a register BAR: unlike RAM, MMIO is not covered by
/// the bootloader's physical-memory map, so a raw [`phys_to_virt`] read faults on real
/// hardware. Falls back to the plain offset if the facility is not yet initialised.
pub fn map_mmio(phys: u64, size: usize) -> u64 {
    let mut guard = FRAMES.lock();
    match guard.as_mut() {
        Some(fa) => unsafe {
            crate::memory::map_mmio(x86_64::VirtAddr::new(phys_offset()), fa, phys, size)
        },
        None => phys + phys_offset(),
    }
}

/// A contiguous block of DMA-capable memory.
#[derive(Clone, Copy, Debug)]
pub struct DmaRegion {
    /// Physical base address — what the device is told.
    pub phys: u64,
    /// Virtual base address — what the kernel dereferences.
    pub virt: u64,
    /// Size in bytes (a whole number of 4 KiB pages).
    pub size: usize,
}

impl DmaRegion {
    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.virt as *mut T
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.virt as *const T
    }

    /// Zero the whole region.
    pub fn zero(&self) {
        unsafe { core::ptr::write_bytes(self.virt as *mut u8, 0, self.size) }
    }

    /// A mutable byte slice over the region.
    ///
    /// # Safety
    /// The caller must ensure no aliasing access (e.g. the device writing the
    /// same bytes concurrently) violates Rust's rules for the slice's lifetime.
    pub unsafe fn bytes_mut(&self) -> &'static mut [u8] {
        core::slice::from_raw_parts_mut(self.virt as *mut u8, self.size)
    }
}

/// Allocate `pages` contiguous 4 KiB pages of DMA memory, zeroed.
/// Returns `None` if the facility is uninitialised or memory is exhausted.
pub fn alloc(pages: usize) -> Option<DmaRegion> {
    let mut guard = FRAMES.lock();
    let fa = guard.as_mut()?;
    let frame = fa.allocate_contiguous(pages)?;
    let phys = frame.start_address().as_u64();
    let region = DmaRegion {
        phys,
        virt: phys + phys_offset(),
        size: pages * 4096,
    };
    region.zero();
    Some(region)
}

//! Virtual memory and the frame allocator (bootloader 0.11 edition).
//!
//! The bootloader maps all physical memory at a dynamic offset and hands us a
//! [`MemoryRegions`] map. We build an [`OffsetPageTable`] over that mapping and
//! a frame allocator that draws usable physical frames to back the kernel heap.
//! In keeping with the SASOS model (Stage 3) there is a single address space;
//! protection is intended to come from capabilities, not per-process tables.

use bootloader_api::info::{MemoryRegionKind, MemoryRegions};
use x86_64::structures::paging::{
    FrameAllocator, OffsetPageTable, PageTable, PhysFrame, Size4KiB,
};
use x86_64::{PhysAddr, VirtAddr};

/// Initialise a mapper over the complete physical address space.
///
/// # Safety
/// `physical_memory_offset` must be the true offset at which the bootloader
/// mapped physical memory, and this must be called only once.
pub unsafe fn init(physical_memory_offset: VirtAddr) -> OffsetPageTable<'static> {
    let level_4_table = active_level_4_table(physical_memory_offset);
    OffsetPageTable::new(level_4_table, physical_memory_offset)
}

unsafe fn active_level_4_table(physical_memory_offset: VirtAddr) -> &'static mut PageTable {
    use x86_64::registers::control::Cr3;

    let (level_4_table_frame, _) = Cr3::read();
    let phys = level_4_table_frame.start_address();
    let virt = physical_memory_offset + phys.as_u64();
    let page_table_ptr: *mut PageTable = virt.as_mut_ptr();
    &mut *page_table_ptr
}

/// Map a device **MMIO** region `[phys, phys+size)` into the kernel address space at the
/// usual physical-offset virtual address (`phys_offset + phys`), with caching disabled,
/// and return that virtual base.
///
/// Device register BARs (NVMe/AHCI/xHCI) live in physical MMIO holes that the
/// bootloader's RAM mapping does **not** cover, so dereferencing `phys_to_virt(bar)`
/// directly page-faults on real hardware. A driver must map its BAR first. Pages that
/// happen to be mapped already are tolerated. Caching off (NO_CACHE|WRITE_THROUGH) is
/// mandatory for device registers.
///
/// # Safety
/// `phys` must be a real device MMIO base and `phys_offset` the true bootloader offset.
pub unsafe fn map_mmio(
    phys_offset: VirtAddr,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    phys: u64,
    size: usize,
) -> u64 {
    use x86_64::structures::paging::{Mapper, Page, PageTableFlags};
    let mut mapper = init(phys_offset);
    let flags = PageTableFlags::PRESENT
        | PageTableFlags::WRITABLE
        | PageTableFlags::NO_CACHE
        | PageTableFlags::WRITE_THROUGH;
    let start = phys & !0xFFF;
    let end = (phys + size as u64 + 0xFFF) & !0xFFF;
    let mut p = start;
    while p < end {
        let frame = PhysFrame::<Size4KiB>::containing_address(PhysAddr::new(p));
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(phys_offset.as_u64() + p));
        match mapper.map_to(page, frame, flags, frame_allocator) {
            Ok(tlb) => tlb.flush(),
            // Already mapped (e.g. the bootloader covered it) — just make sure the TLB
            // reflects the entry and move on.
            Err(_) => x86_64::instructions::tlb::flush(page.start_address()),
        }
        p += 4096;
    }
    phys_offset.as_u64() + phys
}

/// Remap a virtual address range as **read + execute only** (no write, no NX bit).
///
/// Call this after writing an ELF image into a writable buffer and before
/// jumping into it.  This is the kernel's W^X enforcement hook for user-space
/// (or in-kernel trusted) code: a page must never be simultaneously writable
/// and executable.
///
/// The range `[virt, virt + size)` must already be mapped (e.g. via the heap
/// or a prior `map_mmio` call).  Pages are updated in 4 KiB steps; each TLB
/// entry is flushed immediately so the CPU sees the new permissions before the
/// first instruction fetch.
///
/// # Safety
/// * `physical_memory_offset` must be the true bootloader physical-offset.
/// * The range must be mapped and owned by the caller; no other thread may
///   write to it after this call.
/// * The bytes at `virt` must be valid x86-64 machine code before execution.
///
/// # Panics
/// Panics (debug) if `size` is zero or if any page in the range is not mapped.
pub unsafe fn seal_as_rx(physical_memory_offset: VirtAddr, virt: u64, size: usize) {
    use x86_64::structures::paging::{Mapper, Page, PageTableFlags};
    debug_assert!(size > 0, "seal_as_rx: zero-size range is a no-op, probably a bug");
    let mut mapper = init(physical_memory_offset);
    // R-X: PRESENT, no WRITABLE, no NO_EXECUTE.
    let rx_flags = PageTableFlags::PRESENT;
    let start = virt & !0xFFF;
    let end = (virt + size as u64 + 0xFFF) & !0xFFF;
    let mut p = start;
    while p < end {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(p));
        match mapper.update_flags(page, rx_flags) {
            Ok(tlb) => tlb.flush(),
            Err(e) => panic!("seal_as_rx: update_flags failed at {:#x}: {:?}", p, e),
        }
        p += 4096;
    }
}

/// Restore a range previously sealed by [`seal_as_rx`] back to the heap's normal
/// `PRESENT | WRITABLE | NO_EXECUTE` mapping.
///
/// This must be called once a sealed image is no longer executing and before the
/// backing memory can be written again — in particular before the owning `Vec` is
/// freed, since the allocator writes free-list metadata into the block on dealloc.
/// Re-marking the pages writable here keeps the W^X invariant intact: the pages
/// were read+execute only while code ran, and become writable-but-not-executable
/// again afterwards (never both at once).
///
/// # Safety
/// `physical_memory_offset` must be the true bootloader physical-memory offset, and
/// `virt..virt+size` must be a range that was previously passed to [`seal_as_rx`].
pub unsafe fn restore_as_rw(physical_memory_offset: VirtAddr, virt: u64, size: usize) {
    use x86_64::structures::paging::{Mapper, Page, PageTableFlags};
    debug_assert!(size > 0, "restore_as_rw: zero-size range is a no-op, probably a bug");
    let mut mapper = init(physical_memory_offset);
    // Heap default: present, writable, never executable.
    let rw_flags =
        PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    let start = virt & !0xFFF;
    let end = (virt + size as u64 + 0xFFF) & !0xFFF;
    let mut p = start;
    while p < end {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(p));
        match mapper.update_flags(page, rw_flags) {
            Ok(tlb) => tlb.flush(),
            Err(e) => panic!("restore_as_rw: update_flags failed at {:#x}: {:?}", p, e),
        }
        p += 4096;
    }
}

/// Physical bytes reserved at the bottom of RAM for the SMP startup trampoline.
/// Zero unless the `smp` feature is on, so non-SMP builds are unaffected.
#[cfg(feature = "smp")]
const LOW_RESERVE: u64 = 0x10000;

/// Whether a physical frame at `addr` may back the heap. With SMP the low
/// `LOW_RESERVE` bytes are withheld for the AP trampoline; without SMP no frame
/// is reserved. Splitting this out keeps the always-true non-SMP case out of an
/// `addr >= 0` comparison that clippy denies, and avoids an unused constant.
#[cfg(feature = "smp")]
#[inline]
fn keep_frame(addr: u64) -> bool {
    addr >= LOW_RESERVE
}
#[cfg(not(feature = "smp"))]
#[inline]
fn keep_frame(_addr: u64) -> bool {
    true
}

/// A frame allocator that draws from the bootloader's memory map.
///
/// It hands out frames by walking a **cursor** (region index + next physical address)
/// that only ever moves forward, so `allocate_frame` is O(1). The previous version did
/// `usable_frames().nth(self.next)` — rebuilding and re-scanning the whole memory map on
/// every single allocation, i.e. O(n²) over a boot. That was invisible for a 40 MiB
/// heap (~10k frames) but turns a larger heap (256 MiB ≈ 65k frames) into billions of
/// iterations — a multi-minute stall in `init_heap` that looks exactly like a hang.
pub struct BootInfoFrameAllocator {
    memory_map: &'static MemoryRegions,
    /// Index of the region the cursor is currently in.
    region_idx: usize,
    /// Next candidate physical frame address (4 KiB-aligned) within `region_idx`.
    next_addr: u64,
}

impl BootInfoFrameAllocator {
    /// # Safety
    /// The memory map must be valid and all `Usable` frames genuinely unused.
    pub unsafe fn init(memory_map: &'static MemoryRegions) -> Self {
        BootInfoFrameAllocator {
            memory_map,
            region_idx: 0,
            next_addr: 0,
        }
    }

    /// Advance the cursor to a usable region that still has a frame at `next_addr`
    /// (4 KiB-aligned, within bounds). Returns false once the map is exhausted. O(1)
    /// amortised: each region is entered at most once across the whole allocation run.
    fn seek_region(&mut self) -> bool {
        loop {
            if self.region_idx >= self.memory_map.len() {
                return false;
            }
            let r = self.memory_map[self.region_idx];
            if r.kind == MemoryRegionKind::Usable {
                let start = (r.start + 4095) & !4095;
                if self.next_addr < start {
                    self.next_addr = start;
                }
                if self.next_addr + 4096 <= r.end {
                    return true;
                }
            }
            // Region not usable, or exhausted — move to the next one.
            self.region_idx += 1;
            self.next_addr = 0;
        }
    }

    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame> + '_ {
        let regions = self.memory_map.iter();
        let usable = regions.filter(|r| r.kind == MemoryRegionKind::Usable);
        let addr_ranges = usable.map(|r| r.start..r.end);
        let frame_addresses = addr_ranges.flat_map(|r| r.step_by(4096));
        // With SMP we reserve the low 64 KiB so the AP startup trampoline can own a
        // fixed, page-aligned physical page below 1 MiB (the SIPI vector addresses
        // it) without the heap ever backing a virtual page with that frame. On
        // non-SMP builds nothing is reserved, so `keep_frame` admits every frame —
        // the predicate is `cfg`-gated rather than `addr >= 0`, which clippy
        // (correctly) rejects as an always-true comparison on the minimum `u64`.
        frame_addresses
            .filter(|&addr| keep_frame(addr))
            .map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }

    /// Count of usable frames — surfaced by the terminal's `mem` command.
    pub fn usable_frame_count(&self) -> usize {
        self.usable_frames().count()
    }

    /// Allocate `count` *physically contiguous* 4 KiB frames, returning the first.
    /// Needed for DMA structures (virtqueues) that the spec requires to occupy a
    /// single contiguous physical region. Frames skipped while searching for a run
    /// are simply leaked — acceptable given we only do this a handful of times at
    /// driver init and have plenty of RAM.
    pub fn allocate_contiguous(&mut self, count: usize) -> Option<PhysFrame> {
        if count == 0 {
            return None;
        }
        loop {
            if !self.seek_region() {
                return None;
            }
            let r = self.memory_map[self.region_idx];
            let base = self.next_addr;
            // `keep_frame` is monotonic (it only rejects addresses below LOW_RESERVE), so
            // if `base` is kept the whole contiguous run above it is kept too. Frames
            // within a single Usable region are physically contiguous by construction.
            if keep_frame(base) && base + (count as u64) * 4096 <= r.end {
                self.next_addr = base + (count as u64) * 4096;
                return Some(PhysFrame::containing_address(PhysAddr::new(base)));
            }
            if !keep_frame(base) {
                // Skip a reserved low frame and retry from the next one.
                self.next_addr += 4096;
            } else {
                // This region can't fit the run; abandon its remainder (leaked, as
                // before — contiguous allocs happen a handful of times at driver init).
                self.region_idx += 1;
                self.next_addr = 0;
            }
        }
    }
}

unsafe impl FrameAllocator<Size4KiB> for BootInfoFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        loop {
            if !self.seek_region() {
                return None;
            }
            let addr = self.next_addr;
            self.next_addr += 4096;
            if keep_frame(addr) {
                return Some(PhysFrame::containing_address(PhysAddr::new(addr)));
            }
            // Reserved low frame — skip and keep looking.
        }
    }
}

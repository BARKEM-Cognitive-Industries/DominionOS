//! The kernel heap.
//!
//! `dominion-core` is built on `alloc` (Strings, Vecs, the object graph), so the
//! kernel must provide a global allocator before the terminal can run. We map a
//! fixed region and hand it to a linked-list allocator. This is the bridge that
//! lets the same allocation-using code run both in host unit tests and on bare
//! metal. The region is 32 MiB: the lattice PQ KEM materialises ~256 KiB matrices
//! per keypair, and the graphical desktop keeps a full-screen RGB back-buffer
//! (~3.7 MiB at 1280×720), so the system needs real headroom.

use linked_list_allocator::LockedHeap;
use x86_64::structures::paging::{
    mapper::MapToError, FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB,
};
use x86_64::VirtAddr;

pub const HEAP_START: usize = 0x_4444_4444_0000;

// The default heap is 256 MiB. The graphical desktop keeps TWO full-screen 32-bit
// buffers (back + front), which is `width*height*8` bytes — ~16.6 MiB at 1080p but
// ~66 MiB at 4K. On bare metal the firmware hands us the panel's *native* resolution
// (often 1440p/4K), so a smaller heap OOMs inside `gfx::init` before the desktop can
// even draw — a black screen right after boot. 256 MiB covers up to ~5K with room for
// the object store and the lattice PQ KEM matrices. Any modern PC has the RAM; the
// frame allocator maps this region once at `init_heap`. The `big_heap` feature (pulled
// in by `qemu_bench`) raises it to 1 GiB for the benchmark battery and needs a matching
// QEMU `-m` (the bench launcher passes 4096 MiB).
#[cfg(not(feature = "big_heap"))]
pub const HEAP_SIZE: usize = 256 * 1024 * 1024; // 256 MiB
#[cfg(feature = "big_heap")]
pub const HEAP_SIZE: usize = 1024 * 1024 * 1024; // 1 GiB (benchmark battery)

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Map the heap pages and initialise the allocator over them.
pub fn init_heap(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), MapToError<Size4KiB>> {
    let page_range = {
        let heap_start = VirtAddr::new(HEAP_START as u64);
        let heap_end = heap_start + HEAP_SIZE as u64 - 1u64;
        let heap_start_page = Page::containing_address(heap_start);
        let heap_end_page = Page::containing_address(heap_end);
        Page::range_inclusive(heap_start_page, heap_end_page)
    };

    // Map every heap page, but DON'T `invlpg` per page. The heap occupies a fresh
    // virtual range (HEAP_START) that the bootloader never mapped, so no stale TLB
    // entry can exist for any of these addresses — there is nothing per-page to
    // invalidate. A per-page `.flush()` would emit one `invlpg` for every 4 KiB
    // page: 10,240 of them for the 40 MiB heap, 262,144 for the 1 GiB `big_heap`
    // build. Instead we `.ignore()` each mapping and do a single TLB flush after the
    // whole batch, before the allocator is initialised and the heap is first read.
    // NO_EXECUTE: heap pages are pure data — they must never be executable.
    // On x86-64, pages are executable by default unless NXE (EFER.NXE) is set and
    // NO_EXECUTE is set in the PTE.  Omitting NO_EXECUTE here would leave every
    // heap allocation simultaneously writable and executable, which is a W^X
    // violation that lets an attacker who controls heap bytes inject shell-code.
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    for page in page_range {
        let frame = frame_allocator
            .allocate_frame()
            .ok_or(MapToError::FrameAllocationFailed)?;
        unsafe {
            // SAFETY: brand-new mapping into an unmapped range; deferring the flush
            // to one `flush_all` below cannot expose a stale translation.
            mapper.map_to(page, frame, flags, frame_allocator)?.ignore();
        }
    }

    // One CR3-reload-based full flush instead of N `invlpg`s. Must happen before the
    // first heap access (the `ALLOCATOR.init` below stores into the region).
    x86_64::instructions::tlb::flush_all();

    unsafe {
        ALLOCATOR.lock().init(HEAP_START as *mut u8, HEAP_SIZE);
    }

    Ok(())
}

/// Bytes currently free in the heap — reported by the `mem` command.
pub fn free_bytes() -> usize {
    ALLOCATOR.lock().free()
}

/// Bytes currently handed out by the heap.
pub fn used_bytes() -> usize {
    ALLOCATOR.lock().used()
}

/// Total heap size in bytes.
pub fn total_bytes() -> usize {
    HEAP_SIZE
}

#[alloc_error_handler]
fn alloc_error_handler(layout: core::alloc::Layout) -> ! {
    // The kernel heap is a single fixed pre-mapped region (see HEAP_SIZE): there is
    // no larger pool to fall back to and no per-domain reaper to evict (untrusted
    // code runs inside the capability-bounded `wasm::Sandbox`, which is itself
    // memory-bounded — see dominion-core), so an allocation failure here is a genuine
    // capacity/leak bug in the kernel itself and is fatal. Emit the heap census over
    // serial first so the failure is diagnosable in headless/CI runs rather than an
    // opaque abort.
    //
    // Reading the allocator requires its lock; the alloc that failed has already
    // released it, so this is safe.
    crate::serial_println!(
        "[heap] OUT OF MEMORY requesting {} bytes (align {})",
        layout.size(),
        layout.align(),
    );
    crate::serial_println!(
        "[heap] census: used={} free={} total={} bytes",
        ALLOCATOR.lock().used(),
        ALLOCATOR.lock().free(),
        HEAP_SIZE,
    );
    panic!("kernel heap allocation error: {:?}", layout);
}

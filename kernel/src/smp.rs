//! Real symmetric multiprocessing — application-processor (AP) bring-up.
//!
//! The bootloader starts only the bootstrap processor (BSP). This module brings
//! the other cores online so work can run on multiple host cores in parallel —
//! the prerequisite for a *real* cross-core scaling curve (not a model).
//!
//! Sequence (Intel MP / ACPI):
//!   1. Parse the ACPI **MADT** (via the bootloader's RSDP) for the enabled local
//!      APIC ids and the LAPIC MMIO base.
//!   2. Map the LAPIC MMIO and software-enable the local APIC.
//!   3. Copy a real-mode→long-mode **trampoline** to a fixed low page (phys
//!      `0x8000`, reserved from the heap in `memory.rs`) and identity-map it.
//!   4. For each AP, send **INIT-SIPI-SIPI** and wait for it to report online.
//!
//! Because DominionOS is a single address space (SASOS), every AP shares the BSP's
//! page tables (CR3): the trampoline just enables paging with the existing PML4
//! and jumps to a 64-bit Rust entry point. APs then park on a lock-free job queue
//! used by the scaling benchmark.

use crate::serial_println;
use core::ptr::{read_unaligned, read_volatile, write_unaligned, write_volatile};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use x86_64::registers::control::Cr3;
use x86_64::structures::paging::mapper::MapToError;
use x86_64::structures::paging::{
    FrameAllocator, Mapper, Page, PageTableFlags, PhysFrame, Size4KiB,
};
use x86_64::{PhysAddr, VirtAddr};

/// Fixed low physical page for the AP trampoline (must be < 1 MiB and page-aligned;
/// the SIPI start-vector is `phys >> 12`). Reserved from the heap by `memory.rs`.
const TRAMPOLINE_PHYS: u64 = 0x8000;
/// Maximum cores we track.
pub const MAX_CPUS: usize = 64;
/// Per-AP bootstrap stack size.
const AP_STACK_SIZE: usize = 64 * 1024;

// The 16→32→64-bit trampoline. Copied verbatim to phys 0x8000, so every absolute
// address inside is written as `(label - tramp_start + 0x8000)` — the label
// difference is position-independent and 0x8000 is the known load base. The BSP
// patches CR3 / the 64-bit entry / the per-AP stack into the param block.
core::arch::global_asm!(
    r#"
.section .rodata
.code16
.global tramp_start
.global tramp_end
.global tramp_params
tramp_start:
    cli
    cld
    xorw    %ax, %ax
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %ss
    lgdtl   (gdt_ptr - tramp_start + 0x8000)
    movl    %cr0, %eax
    orl     $1, %eax
    movl    %eax, %cr0
    ljmpl   $0x08, $(prot - tramp_start + 0x8000)

.code32
prot:
    movw    $0x10, %ax
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %ss
    movl    $(p_cr3 - tramp_start + 0x8000), %edx
    movl    (%edx), %eax
    movl    %eax, %cr3
    movl    %cr4, %eax
    orl     $(1 << 5), %eax
    movl    %eax, %cr4
    movl    $0xC0000080, %ecx
    rdmsr
    orl     $((1 << 8) | (1 << 11)), %eax
    wrmsr
    movl    %cr0, %eax
    orl     $0x80000000, %eax
    movl    %eax, %cr0
    ljmpl   $0x18, $(long_mode - tramp_start + 0x8000)

.code64
long_mode:
    movw    $0x10, %ax
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %ss
    movl    $(p_stack - tramp_start + 0x8000), %eax
    movq    (%rax), %rsp
    movl    $(p_entry - tramp_start + 0x8000), %eax
    movq    (%rax), %rax
    callq   *%rax
1:  hlt
    jmp     1b

.align 16
gdt:
    .quad 0x0000000000000000
    .quad 0x00CF9A000000FFFF
    .quad 0x00CF92000000FFFF
    .quad 0x00AF9A000000FFFF
gdt_end:
.align 4
gdt_ptr:
    .word gdt_end - gdt - 1
    .long gdt - tramp_start + 0x8000
.align 8
tramp_params:
p_cr3:   .quad 0
p_entry: .quad 0
p_stack: .quad 0
tramp_end:
"#,
    options(att_syntax)
);

extern "C" {
    static tramp_start: u8;
    static tramp_end: u8;
    static tramp_params: u8;
}

/// Virtual address through which we touch the LAPIC MMIO.
static LAPIC_VADDR: AtomicU64 = AtomicU64::new(0);
/// Count of APs that have reported online (the BSP is not counted here).
static AP_ONLINE: AtomicU32 = AtomicU32::new(0);
/// Assigns each AP a worker index (BSP is index 0).
static CPU_INDEX_CTR: AtomicU32 = AtomicU32::new(0);

// ── lock-free job queue for the scaling benchmark ──
static JOB_GEN: AtomicU64 = AtomicU64::new(0);
static JOB_LIMIT: AtomicU64 = AtomicU64::new(0);
static CHUNK: AtomicU64 = AtomicU64::new(0); // monotonic chunk dispenser (never reset)
static DONE: AtomicU64 = AtomicU64::new(0); // monotonic completed-chunk counter
static ACTIVE: AtomicU32 = AtomicU32::new(0); // workers permitted to participate
static RESULT: AtomicU64 = AtomicU64::new(0); // accumulated work (defeats DCE)

const NCHUNKS: u64 = 2048;
const CHUNK_WORK: u64 = 200_000;

#[inline]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn delay_cycles(n: u64) {
    let t = rdtsc();
    while rdtsc().wrapping_sub(t) < n {
        core::hint::spin_loop();
    }
}

fn lapic_write(reg: u32, val: u32) {
    let base = LAPIC_VADDR.load(Ordering::Relaxed);
    unsafe { write_volatile((base + reg as u64) as *mut u32, val) }
}
fn lapic_read(reg: u32) -> u32 {
    let base = LAPIC_VADDR.load(Ordering::Relaxed);
    unsafe { read_volatile((base + reg as u64) as *const u32) }
}
fn lapic_id() -> u8 {
    (lapic_read(0x20) >> 24) as u8
}
fn lapic_wait_delivery() {
    let t = rdtsc();
    // ICR-low bit 12 = delivery pending.
    while lapic_read(0x300) & (1 << 12) != 0 {
        if rdtsc().wrapping_sub(t) > 50_000_000 {
            break;
        }
        core::hint::spin_loop();
    }
}

/// Parse the ACPI MADT for enabled APIC ids and the LAPIC base.
unsafe fn parse_madt(phys_offset: u64, rsdp_phys: u64) -> (alloc::vec::Vec<u8>, u64) {
    use alloc::vec::Vec;
    let mut ids: Vec<u8> = Vec::new();
    let mut lapic_base: u64 = 0xFEE0_0000;
    if rsdp_phys == 0 {
        return (ids, lapic_base);
    }
    let rsdp = phys_offset + rsdp_phys;
    let revision = read_unaligned((rsdp + 15) as *const u8);
    let (sdt_phys, entry_size) = if revision >= 2 {
        (read_unaligned((rsdp + 24) as *const u64), 8usize)
    } else {
        (read_unaligned((rsdp + 16) as *const u32) as u64, 4usize)
    };
    let sdt = phys_offset + sdt_phys;
    let length = read_unaligned((sdt + 4) as *const u32) as usize;
    let entries = length.saturating_sub(36) / entry_size;
    let mut madt_phys = 0u64;
    for i in 0..entries {
        let ent = sdt + 36 + (i * entry_size) as u64;
        let p = if entry_size == 8 {
            read_unaligned(ent as *const u64)
        } else {
            read_unaligned(ent as *const u32) as u64
        };
        let sig = core::slice::from_raw_parts((phys_offset + p) as *const u8, 4);
        if sig == b"APIC" {
            madt_phys = p;
            break;
        }
    }
    if madt_phys == 0 {
        return (ids, lapic_base);
    }
    let madt = phys_offset + madt_phys;
    let madt_len = read_unaligned((madt + 4) as *const u32) as usize;
    lapic_base = read_unaligned((madt + 36) as *const u32) as u64;
    let mut off = 44usize; // 36 header + 4 lapic addr + 4 flags
    while off + 2 <= madt_len {
        let etype = read_unaligned((madt + off as u64) as *const u8);
        let elen = read_unaligned((madt + off as u64 + 1) as *const u8) as usize;
        if elen == 0 {
            break;
        }
        match etype {
            0 => {
                let apic_id = read_unaligned((madt + off as u64 + 3) as *const u8);
                let flags = read_unaligned((madt + off as u64 + 4) as *const u32);
                if flags & 1 == 1 {
                    ids.push(apic_id);
                }
            }
            5 => {
                lapic_base = read_unaligned((madt + off as u64 + 4) as *const u64);
            }
            _ => {}
        }
        off += elen;
    }
    (ids, lapic_base)
}

fn map_page<M, A>(mapper: &mut M, fa: &mut A, virt: u64, phys: u64, flags: PageTableFlags)
where
    M: Mapper<Size4KiB>,
    A: FrameAllocator<Size4KiB>,
{
    let page = Page::<Size4KiB>::containing_address(VirtAddr::new(virt));
    let frame = PhysFrame::<Size4KiB>::containing_address(PhysAddr::new(phys));
    unsafe {
        match mapper.map_to(page, frame, flags, fa) {
            Ok(t) => t.flush(),
            // Already mapped (e.g. the bootloader's full-physical map covers it) —
            // the existing mapping is what we want, so this is fine.
            Err(MapToError::PageAlreadyMapped(_)) => {}
            Err(MapToError::ParentEntryHugePage) => {}
            Err(_) => serial_println!("[smp] warning: map_to failed for virt {:#x}", virt),
        }
    }
}

unsafe fn write_trampoline(phys_offset: u64, cr3: u64, entry: u64) {
    let start = core::ptr::addr_of!(tramp_start);
    let end = core::ptr::addr_of!(tramp_end);
    let len = end.offset_from(start) as usize;
    let dst = (phys_offset + TRAMPOLINE_PHYS) as *mut u8;
    core::ptr::copy_nonoverlapping(start, dst, len);

    let params_off = core::ptr::addr_of!(tramp_params).offset_from(start) as usize;
    let pbase = (phys_offset + TRAMPOLINE_PHYS + params_off as u64) as *mut u64;
    write_unaligned(pbase, cr3); // p_cr3
    write_unaligned(pbase.add(1), entry); // p_entry
}

unsafe fn set_ap_stack(phys_offset: u64, stack_top: u64) {
    let start = core::ptr::addr_of!(tramp_start);
    let params_off = core::ptr::addr_of!(tramp_params).offset_from(start) as usize;
    let pbase = (phys_offset + TRAMPOLINE_PHYS + params_off as u64) as *mut u64;
    write_unaligned(pbase.add(2), stack_top); // p_stack
}

fn alloc_stack_top() -> u64 {
    use alloc::vec;
    use alloc::vec::Vec;
    let v: Vec<u8> = vec![0; AP_STACK_SIZE];
    let base = v.as_ptr() as u64;
    core::mem::forget(v); // leak: APs live for the kernel's lifetime
    (base + AP_STACK_SIZE as u64) & !0xF
}

/// The 64-bit entry point every AP lands on (via the trampoline). Never returns.
#[no_mangle]
extern "C" fn ap_entry() -> ! {
    let idx = CPU_INDEX_CTR.fetch_add(1, Ordering::SeqCst) + 1; // BSP is 0
    AP_ONLINE.fetch_add(1, Ordering::SeqCst);
    worker_loop(idx as usize)
}

fn run_chunks() {
    loop {
        let c = CHUNK.fetch_add(1, Ordering::SeqCst);
        if c >= JOB_LIMIT.load(Ordering::SeqCst) {
            break;
        }
        RESULT.fetch_add(compute_chunk(c), Ordering::Relaxed);
        DONE.fetch_add(1, Ordering::SeqCst);
    }
}

fn worker_loop(idx: usize) -> ! {
    let mut seen = 0u64;
    let mut pool_gen = 0u64; // last pool generation this AP has processed
    loop {
        let g = JOB_GEN.load(Ordering::SeqCst);
        if g != seen {
            seen = g;
            if (idx as u32) < ACTIVE.load(Ordering::SeqCst) {
                run_chunks();
            }
        }
        // Drain any work submitted to the OS-wide thread pool before going idle.
        // This is the single hook that wires every AP into the native pool with
        // zero spawn overhead — the AP was already spinning here.
        crate::threadpool::ap_poll_once(&mut pool_gen);
        core::hint::spin_loop();
    }
}

#[inline(never)]
fn compute_chunk(c: u64) -> u64 {
    // CPU-bound, cache-light: a long dependent integer-mix chain seeded by the
    // chunk id. Scales with cores because there is no shared memory traffic.
    let mut x = c.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for _ in 0..CHUNK_WORK {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        x ^= x >> 31;
    }
    x
}

/// Bring up the application processors. Call once at boot, after the heap is up
/// and while the mapper + frame allocator are still available.
pub fn init<M, A>(mapper: &mut M, fa: &mut A, phys_offset: u64, rsdp: Option<u64>)
where
    M: Mapper<Size4KiB>,
    A: FrameAllocator<Size4KiB>,
{
    let rsdp_phys = rsdp.unwrap_or(0);
    let (ids, lapic_base) = unsafe { parse_madt(phys_offset, rsdp_phys) };
    serial_println!(
        "[smp] MADT: {} enabled CPU(s), LAPIC base {:#x}",
        ids.len(),
        lapic_base
    );
    if ids.len() <= 1 {
        serial_println!("[smp] single CPU — no APs to start");
        return;
    }

    // Map the LAPIC MMIO (uncached) and the trampoline page (identity).
    let lapic_virt = phys_offset + lapic_base;
    map_page(
        mapper,
        fa,
        lapic_virt,
        lapic_base,
        PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_CACHE,
    );
    LAPIC_VADDR.store(lapic_virt, Ordering::SeqCst);
    // W^X fix: write the trampoline code FIRST via the physical-offset alias
    // (already mapped RW by the bootloader), THEN map the identity page as
    // execute-only (PRESENT, no WRITABLE, no NO_EXECUTE).  This guarantees the
    // page is never simultaneously writable and executable.
    let cr3 = Cr3::read().0.start_address().as_u64();
    let entry: extern "C" fn() -> ! = ap_entry;
    unsafe { write_trampoline(phys_offset, cr3, entry as usize as u64) };

    // Identity-map the trampoline as read+execute only — no WRITABLE.
    // APs need to fetch code from this page; they must not be able to write it.
    map_page(
        mapper,
        fa,
        TRAMPOLINE_PHYS,
        TRAMPOLINE_PHYS,
        PageTableFlags::PRESENT,
    );

    // Software-enable the local APIC (spurious-interrupt vector register).
    lapic_write(0xF0, 0x1FF);
    let bsp = lapic_id();
    serial_println!("[smp] BSP apic id {}, trampoline at {:#x}", bsp, TRAMPOLINE_PHYS);

    let vector = (TRAMPOLINE_PHYS >> 12) as u32;
    for &id in ids.iter() {
        if id == bsp {
            continue;
        }
        let before = AP_ONLINE.load(Ordering::SeqCst);
        unsafe { set_ap_stack(phys_offset, alloc_stack_top()) };

        // INIT IPI.
        lapic_write(0x310, (id as u32) << 24);
        lapic_write(0x300, 0x4500);
        lapic_wait_delivery();
        delay_cycles(30_000_000); // ~10 ms

        // SIPI x2.
        for _ in 0..2 {
            lapic_write(0x310, (id as u32) << 24);
            lapic_write(0x300, 0x4600 | vector);
            lapic_wait_delivery();
            delay_cycles(1_000_000); // ~200-500 us
        }

        // Wait for the AP to check in.
        let t = rdtsc();
        while AP_ONLINE.load(Ordering::SeqCst) == before {
            if rdtsc().wrapping_sub(t) > 2_000_000_000 {
                serial_println!("[smp] AP apic id {} did not come online", id);
                break;
            }
            core::hint::spin_loop();
        }
        if AP_ONLINE.load(Ordering::SeqCst) > before {
            serial_println!("[smp] AP apic id {} online", id);
        }
    }
    serial_println!(
        "[smp] {} core(s) online (1 BSP + {} AP)",
        AP_ONLINE.load(Ordering::SeqCst) + 1,
        AP_ONLINE.load(Ordering::SeqCst)
    );
}

/// Number of usable cores (BSP + online APs).
pub fn core_count() -> u32 {
    AP_ONLINE.load(Ordering::SeqCst) + 1
}

/// The cross-core scaling benchmark: run a fixed CPU-bound workload across an
/// increasing number of cores and report throughput + parallel efficiency.
pub fn bench_scaling(hz: u64) {
    let cores = core_count();
    serial_println!("[smp] cross-core scaling across up to {} core(s)", cores);
    let mut t1_ms: u64 = 0;
    for &k in &[1u32, 2, 4, 8, 16, 32] {
        if k > cores {
            break;
        }
        let prev_done = DONE.load(Ordering::SeqCst);
        JOB_LIMIT.store(CHUNK.load(Ordering::SeqCst) + NCHUNKS, Ordering::SeqCst);
        ACTIVE.store(k, Ordering::SeqCst);
        let t0 = rdtsc();
        JOB_GEN.fetch_add(1, Ordering::SeqCst); // release the round
        run_chunks(); // BSP participates as worker 0
        // Wait for this round's chunks to complete (with a watchdog so a wedged AP
        // can never hang the whole battery — it just reports what finished).
        while DONE.load(Ordering::SeqCst).wrapping_sub(prev_done) < NCHUNKS {
            if rdtsc().wrapping_sub(t0) > hz.saturating_mul(30) {
                serial_println!("[smp] scaling round k={} watchdog: round did not complete", k);
                break;
            }
            core::hint::spin_loop();
        }
        let dt = rdtsc().wrapping_sub(t0).max(1);
        let ms = (dt as u128 * 1000 / hz as u128) as u64;
        let cps = (NCHUNKS as u128 * hz as u128 / dt as u128) as u64;
        if k == 1 {
            t1_ms = ms.max(1);
        }
        // Parallel efficiency = speedup / k, ×100. Ideal linear scaling = 100.
        let speedup_x100 = (t1_ms * 100).checked_div(ms).unwrap_or(0);
        let eff = speedup_x100 / k as u64;
        serial_println!(
            "BENCH scaling cores={} chunks={} ms={} chunks_per_s={} speedup_pct={} efficiency_pct={}",
            k, NCHUNKS, ms, cps, speedup_x100, eff
        );
    }
    serial_println!("[smp] scaling checksum {}", RESULT.load(Ordering::Relaxed));
}

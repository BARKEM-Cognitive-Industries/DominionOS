//! DominionOS boot entry point (bootloader 0.11).
//!
//! The bootloader hands control here with a [`BootInfo`] containing a linear
//! framebuffer and a memory map. We initialise the screen, the CPU tables and
//! interrupts, build the page mapper and frame allocator, map the kernel heap,
//! then launch the safe-mode terminal (ASH) — or, under `--features qemu_test`,
//! the headless bare-metal self-test battery. From the user's perspective the
//! machine powers on and lands directly in the recovery shell.
#![no_std]
#![no_main]

extern crate alloc;

use dominion_kernel::shell::{Shell, SystemInfo};
use dominion_kernel::{serial_println, vga_buffer};
use bootloader_api::config::Mapping;
use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use core::panic::PanicInfo;
use x86_64::VirtAddr;

/// Emit a boot-stage line to serial always, and — in `safe_mode` — also to the on-screen
/// text console, so boot progress is visible on bare metal without a serial cable. The
/// last line left on screen pinpoints any stage that hangs. Markers are printed BEFORE
/// each risky probe so a hang *inside* a driver still leaves its label visible.
macro_rules! boot_stage {
    ($($arg:tt)*) => {{
        dominion_kernel::serial_println!($($arg)*);
        #[cfg(feature = "safe_mode")]
        dominion_kernel::println!($($arg)*);
    }};
}

/// Ask the bootloader to map all physical memory at a dynamic offset so we can
/// build an `OffsetPageTable` and a heap.
const CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.mappings.physical_memory = Some(Mapping::Dynamic);
    // The Dominion interpreter is recursive (tree-walking), so give the kernel a
    // generous stack — the default is far too small for nested evaluation.
    config.kernel_stack_size = 1024 * 1024; // 1 MiB
    config
};

entry_point!(kernel_main, config = &CONFIG);

/// Enable the SSE/SSE2 hardware floating-point unit. Must run before any code that
/// touches an XMM register (i.e. any `f64` math in a hard-float build), or the first
/// SSE instruction would fault. Idempotent and side-effect-only, so it is safe as the
/// very first thing `kernel_main` does.
fn enable_sse() {
    use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};
    unsafe {
        let mut cr0 = Cr0::read();
        cr0.remove(Cr0Flags::EMULATE_COPROCESSOR); // EM = 0: use SSE, don't trap to emulation
        cr0.insert(Cr0Flags::MONITOR_COPROCESSOR); // MP = 1
        Cr0::write(cr0);
        let mut cr4 = Cr4::read();
        cr4.insert(Cr4Flags::OSFXSR); // FXSAVE/FXRSTOR + SSE enabled
        cr4.insert(Cr4Flags::OSXMMEXCPT_ENABLE); // unmasked SIMD FP exceptions go to #XM
        Cr4::write(cr4);
    }
}

/// Enable AVX (and the XSAVE state it needs) so the `fma`-feature build's VEX-encoded
/// FMA instructions don't fault. Checks CPUID first and no-ops if the CPU lacks
/// XSAVE/AVX, so it is always safe to call. Must run before any AVX/FMA code.
#[cfg(feature = "fma")]
fn enable_avx() {
    use core::arch::x86_64::__cpuid;
    use x86_64::registers::control::{Cr4, Cr4Flags};
    use x86_64::registers::xcontrol::{XCr0, XCr0Flags};
    unsafe {
        let c = __cpuid(1);
        let has_xsave = (c.ecx >> 26) & 1 == 1;
        let has_avx = (c.ecx >> 28) & 1 == 1;
        if !(has_xsave && has_avx) {
            serial_println!("[boot] AVX/FMA requested but CPU lacks XSAVE/AVX — staying on SSE2");
            return;
        }
        let mut cr4 = Cr4::read();
        cr4.insert(Cr4Flags::OSXSAVE);
        Cr4::write(cr4);
        let mut xcr0 = XCr0::read();
        xcr0.insert(XCr0Flags::X87 | XCr0Flags::SSE | XCr0Flags::AVX);
        XCr0::write(xcr0);
    }
}

// Under `qemu_test` the self-test runner diverges, making the shell launch below
// intentionally unreachable; that is by design.
#[cfg_attr(
    any(feature = "qemu_test", feature = "qemu_bench", feature = "qemu_validate"),
    allow(unreachable_code)
)]
fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // Stage 0: turn on hardware floating point (SSE/SSE2) before any f64 code runs.
    // The `x86_64-unknown-none` target defaults to soft-float; with the kernel built
    // for hard-float (`-C target-feature=-soft-float,+sse2`) this enables the XMM
    // unit so every f64 op (the dashboard, the ML engine, …) runs on real silicon
    // instead of software emulation — a ~10× speedup for floating-point work.
    enable_sse();
    #[cfg(feature = "fma")]
    enable_avx();
    serial_println!(
        "[boot] DominionOS kernel entry (FP: hard-float SSE2{})",
        if cfg!(feature = "fma") { " + AVX/FMA" } else { "" }
    );

    // Stage A: bring up the framebuffer console first, so anything after this
    // is visible on screen.
    if let Some(fb) = boot_info.framebuffer.as_mut() {
        vga_buffer::init(fb);
    }
    // First on-screen line after the bootloader hands off (the screen was just cleared).
    // If you see this but nothing below it, the hang is in the very next stage.
    boot_stage!("[boot] kernel alive - framebuffer console online");

    // Stage B: processor tables + interrupts.
    boot_stage!("[boot] setting up GDT/IDT/PIC + enabling interrupts ...");
    dominion_kernel::init();
    boot_stage!("[boot] GDT + IDT + PIC online, interrupts enabled");

    // Stage C: virtual memory + heap.
    let phys_offset = boot_info
        .physical_memory_offset
        .into_option()
        .expect("bootloader did not map physical memory");
    let phys_mem_offset = VirtAddr::new(phys_offset);
    let mut mapper = unsafe { dominion_kernel::memory::init(phys_mem_offset) };
    let mut frame_allocator =
        unsafe { dominion_kernel::memory::BootInfoFrameAllocator::init(&boot_info.memory_regions) };
    let usable_frames = frame_allocator.usable_frame_count();

    boot_stage!(
        "[boot] mapping {} MiB kernel heap ({} usable frames) ...",
        dominion_kernel::allocator::HEAP_SIZE / (1024 * 1024),
        usable_frames
    );
    dominion_kernel::allocator::init_heap(&mut mapper, &mut frame_allocator)
        .expect("heap initialization failed");
    boot_stage!("[boot] heap mapped, {} usable frames", usable_frames);

    // Stage C1b: parse the ACPI tables for the S5 power-off parameters (PM1
    // control ports + DSDT \_S5 sleep type) so `shutdown` can power real
    // hardware off, not just QEMU. Absence is not fatal — we fall back to the
    // QEMU port magic. Done now while the bootloader's RSDP pointer is in hand.
    boot_stage!("[boot] parsing ACPI tables for power management ...");
    if dominion_kernel::acpi::init(phys_offset, boot_info.rsdp_addr.into_option()) {
        boot_stage!("[boot] ACPI S5 power-off available");
    } else {
        boot_stage!("[boot] ACPI S5 unavailable; shutdown falls back to port magic");
    }

    // Stage C2: bring up the application processors (real SMP). Done here, while the
    // mapper + frame allocator are still in hand, so it can map the AP trampoline.
    #[cfg(feature = "smp")]
    boot_stage!("[boot] bringing up application processors (SMP) ...");
    #[cfg(feature = "smp")]
    dominion_kernel::smp::init(
        &mut mapper,
        &mut frame_allocator,
        phys_offset,
        boot_info.rsdp_addr.into_option(),
    );

    // Stage C3: wire SMP batch execution for the memory subsystem (compression,
    // migration, prefetch). After this call, all batch operations in dominion_core
    // automatically fan out to all cores via KernelSpawn.
    dominion_core::pool::install_spawner(|n, task| {
        use dominion_core::parallel::Spawn;
        dominion_kernel::threadpool::KernelSpawn.run(n, task)
    });
    serial_println!("[boot] SMP batch spawner installed for dominion-core memory subsystem");

    // Stage D: hand the frame allocator + physical offset to the DMA facility so
    // device drivers (virtio) can allocate physically-addressed buffers.
    dominion_kernel::dma::init(phys_offset, frame_allocator);
    boot_stage!("[boot] DMA facility online");

    // Stage E: probe the PCI bus and bring up the block device (M1 persistence).
    // Absence is not fatal — the system then runs purely in RAM. On bare metal this
    // touches real AHCI/NVMe/USB controllers, a common place to wedge, so the marker
    // is printed BEFORE the probe.
    // SAFE MODE skips the whole storage probe: it touches real AHCI/NVMe/USB
    // controllers (the most common bare-metal wedge point) and the recovery shell does
    // not need persistence — it runs in RAM. You can still enumerate every device with
    // `pci` (config-space reads only, no driver init). Per-driver markers are in block.rs.
    #[cfg(not(feature = "safe_mode"))]
    {
        boot_stage!("[boot] probing storage (virtio / AHCI / NVMe / USB) ...");
        if dominion_kernel::block::init_global() {
            boot_stage!(
                "[boot] block device online, {} sectors",
                dominion_kernel::block::capacity_sectors()
            );
        } else {
            boot_stage!("[boot] no block device; running in-memory only");
        }
        // Prefer a removable USB as the debug-log target so the log lands on a drive you
        // can pull out, even when the primary store is an internal AHCI/NVMe disk.
        boot_stage!("[boot] probing removable USB for logging ...");
        if dominion_kernel::block::init_log_device() {
            boot_stage!("[boot] removable USB found — debug log will persist to it");
        } else if dominion_kernel::block::log_is_usb() {
            boot_stage!("[boot] booted from USB — debug log will persist to it");
        }
    }
    #[cfg(feature = "safe_mode")]
    boot_stage!("[boot] SAFE MODE: storage probe skipped (run 'pci' to list devices)");

    // Stage F: bring up the virtio network interface (roadmap feature 1).
    boot_stage!("[boot] probing network ...");
    if dominion_kernel::netif::init_global() {
        let m = dominion_kernel::netif::mac().0;
        boot_stage!(
            "[boot] NIC online (virtio-net / e1000), MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            m[0], m[1], m[2], m[3], m[4], m[5]
        );
    } else {
        boot_stage!("[boot] no network device (virtio-net / e1000)");
    }

    // Stage G: seed the deterministic RNG from the hardware TRNG (RDRAND).
    boot_stage!("[boot] seeding entropy (RDRAND) ...");
    if dominion_kernel::entropy::init_global() {
        boot_stage!("[boot] TRNG online (RDRAND), DRNG seeded");
    } else {
        boot_stage!("[boot] no hardware entropy source (RDRAND); crypto fails closed");
    }

    // Stage G1.5: USB-HID input (keyboard + mouse over xHCI). On a real PC the keyboard
    // and mouse are USB, not PS/2 — without this the desktop has no pointer. Absence is
    // non-fatal (PS/2 / firmware emulation still works where present).
    boot_stage!("[boot] probing USB-HID input (keyboard / mouse) ...");
    if dominion_kernel::usbhid::init() {
        boot_stage!("[boot] USB-HID input online");
    } else {
        boot_stage!("[boot] no USB-HID input (using PS/2 if present)");
    }

    // Stage G2: enumerate the machine — CPU, memory, every PCI device (GPUs incl.
    // integrated + discrete, NVMe/AHCI/IDE/virtio storage, NICs, USB). This lands in
    // the boot-log capture so an unfamiliar bare-metal machine's hardware is recorded.
    boot_stage!("[boot] enumerating hardware ...");
    dominion_kernel::hwreport::log_report(usable_frames);
    boot_stage!("[boot] boot stages complete");

    // Stage H: surface the architecture stage control plane at boot — print the active
    // deployment profile and a live self-test tally. Every enabled stage's probe runs
    // here on the real machine, so the staged architecture is visible from boot.
    serial_println!(
        "{}",
        dominion_core::stages::boot_banner(&dominion_core::stages::StageControl::default())
    );
    // …and the ecosystem control plane (packages, discovery, fleet, remote, onion, …):
    // every enabled feature-set's probe runs here on the real machine at boot.
    serial_println!(
        "{}",
        dominion_core::ecosystem::boot_banner(&dominion_core::ecosystem::EcoControl::default())
    );

    // Headless CI mode: run the bare-metal test battery and exit QEMU.
    #[cfg(feature = "qemu_test")]
    dominion_kernel::selftest::run_and_exit(phys_offset);

    // Headless benchmark mode: run the real-world performance battery, report
    // machine-readable results over serial, and exit QEMU.
    #[cfg(feature = "qemu_bench")]
    dominion_kernel::bench::run_and_exit(usable_frames);

    // Headless validation mode: memory mountain, model-vs-hardware boundary, soak,
    // chaos/failure-injection (and the cross-core scaling curve when SMP is on).
    #[cfg(feature = "qemu_validate")]
    dominion_kernel::bench::run_validation_and_exit(usable_frames);

    // Otherwise: bring up the graphical desktop. It takes over the framebuffer,
    // is driven by the PS/2 mouse + keyboard, and returns to the ASH safe-mode
    // terminal when the user presses Esc or the power button.
    let info = SystemInfo {
        physical_memory_offset: phys_offset,
        usable_frames,
    };

    // Safe mode: skip the graphical desktop entirely and drop straight into the ASH
    // text shell on the framebuffer console. This bypasses gfx::init's full-screen
    // buffer allocation and the whole desktop render path — the recovery environment
    // for a machine where the desktop will not come up. Run `hw`, `pci`, `mem`, `disk`,
    // `net`, `selftest`, `log` to diagnose, then `reboot`/`shutdown`.
    #[cfg(feature = "safe_mode")]
    {
        boot_stage!("[boot] SAFE MODE — desktop skipped, starting ASH text shell");
        dominion_kernel::println!("");
        dominion_kernel::println!("=== DominionOS SAFE MODE (text REPL) ===");
        dominion_kernel::println!("desktop bypassed. type 'help' for commands, 'hw' for hardware,");
        dominion_kernel::println!("'selftest' to run checks, 'reboot' / 'shutdown' to exit.");
        dominion_kernel::println!("");
        let mut shell = Shell::new(info);
        shell.run();
    }

    #[cfg(not(feature = "safe_mode"))]
    {
        boot_stage!("[boot] launching graphical object desktop");
        let power_off = dominion_kernel::desktop::run(info);

        // Persist the full boot/run debug log to the data disk so it can be recovered
        // from the raw image after a bare-metal boot (host tool: read-bootlog.ps1).
        if dominion_kernel::bootlog::persist_best_effort() {
            serial_println!("[boot] debug log persisted to disk (recover with read-bootlog.ps1)");
        }

        if power_off {
            serial_println!("[boot] power off requested; shutting down");
            dominion_kernel::shutdown();
        }

        serial_println!("[boot] desktop exited; launching ASH safe-mode terminal");
        let mut shell = Shell::new(info);
        shell.run();
    }
}

/// Production panic handler: report to screen + serial, then idle.
#[cfg(not(any(feature = "qemu_test", feature = "qemu_bench")))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    use dominion_kernel::println;
    vga_buffer::set_color(vga_buffer::Color::White, vga_buffer::Color::Red);
    println!("\n*** KERNEL PANIC ***");
    println!("{}", info);
    serial_println!("*** KERNEL PANIC ***");
    serial_println!("{}", info);
    // Best-effort: flush the captured debug log (incl. this panic) to disk so the
    // cause survives a bare-metal crash and can be pulled off the image afterwards.
    let _ = dominion_kernel::bootlog::persist_best_effort();
    dominion_kernel::hlt_loop();
}

/// Headless panic handler: report over serial and signal failure to the harness.
#[cfg(any(feature = "qemu_test", feature = "qemu_bench"))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    dominion_kernel::test_panic_handler(info)
}

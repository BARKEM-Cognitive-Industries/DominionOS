//! # DominionOS kernel
//!
//! The bootable base system for DominionOS. It is a small, freestanding
//! `no_std` kernel whose job is to bring a bare-metal machine up to the point
//! where the **safe-mode terminal** (the OS's DOS-equivalent recovery shell,
//! [`shell`]) can run with low-level access to every subsystem from
//! `dominion-core`.
//!
//! The architecture honours the SRS where the hardware allows: a single global
//! address space (SASOS, Stage 3), capability-mediated access to resources
//! (Stage 2/3), and a deterministic, hashable machine model (Stage 10). True
//! CHERI tags and a verified microkernel are future work; this prototype models
//! their *semantics* in safe Rust so the system actually boots and runs.
//!
//! The bare-metal test battery lives in [`selftest`]; the `qemu_test` feature
//! boots straight into it and reports pass/fail over serial + `isa-debug-exit`.
#![no_std]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]

extern crate alloc;

/// Re-export the core library so integration-test crates can reach it as
/// `dominion_kernel::dominion_core::…` without taking a second path dependency
/// (which on Windows triggers a duplicate `core` under build-std).
pub use dominion_core;

pub mod allocator;
pub mod threadpool;
/// The real-world benchmark + validation batteries (`qemu_bench` / `qemu_validate`).
/// Compiled out of normal builds so they never bloat the shipping kernel.
#[cfg(any(feature = "qemu_bench", feature = "qemu_validate"))]
pub mod bench;
/// Real symmetric multiprocessing (application-processor bring-up).
#[cfg(feature = "smp")]
pub mod smp;
pub mod acpi;
pub mod ahci;
pub mod block;
pub mod bootlog;
pub mod desktop;
pub mod dma;
pub mod hwreport;
pub mod nvme;
pub mod entropy;
pub mod gdt;
pub mod gfx;
pub mod interrupts;
pub mod keyboard;
pub mod loader;
pub mod memory;
pub mod mouse;
pub mod netif;
pub mod pci;
pub mod rtl8139;
pub mod selftest;
pub mod rtc;
pub mod serial;
pub mod shell;
pub mod vga_buffer;
pub mod usbhid;
pub mod virtio;
pub mod webnet;
pub mod xhci;

use core::panic::PanicInfo;

/// One-time bring-up of the processor-level subsystems: descriptor tables,
/// interrupt controllers, and the hardware interrupt flag. Memory and the heap
/// are initialised separately because they need the bootloader's memory map.
pub fn init() {
    gdt::init();
    interrupts::init_idt();
    unsafe { interrupts::PICS.lock().initialize() };
    x86_64::instructions::interrupts::enable();
}

/// Halt the CPU until the next interrupt — the idle primitive the shell loops on
/// so the VM does not burn 100% of a core spinning.
pub fn hlt_loop() -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}

/// Power the machine off.
///
/// Tries a *real* ACPI S5 transition first — parsing the firmware's PM1 control
/// registers and `\_S5` sleep type out of the ACPI tables (see [`acpi`]), which
/// is what actually powers down bare metal. If those parameters were never
/// derived (no RSDP, or a firmware we could not parse), or the transition does
/// not take, we fall back to the QEMU-specific ACPI port magic, and finally to
/// halting the CPU so the machine at least comes to rest.
pub fn shutdown() -> ! {
    // 1) Proper ACPI S5 — works on real hardware. Returns only on failure.
    acpi::poweroff();

    // 2) QEMU / virtual-machine fallback: hard-wired ACPI PM1a control ports.
    use x86_64::instructions::port::Port;
    unsafe {
        let mut acpi_port: Port<u16> = Port::new(0x604);
        acpi_port.write(0x2000);
        // Older QEMU machine types use this port instead.
        let mut legacy: Port<u16> = Port::new(0xb004);
        legacy.write(0x2000);
        // Bochs / very old QEMU.
        let mut bochs: Port<u16> = Port::new(0x4004);
        bochs.write(0x3400);
    }

    // 3) Nothing powered us off — halt so we stop drawing the CPU at 100%.
    hlt_loop();
}

// ---------------------------------------------------------------------------
// QEMU exit + serial-reported test harness (custom_test_frameworks).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

/// Exit QEMU via the `isa-debug-exit` device. The configured success code maps
/// to host process exit status 33.
pub fn exit_qemu(exit_code: QemuExitCode) {
    use x86_64::instructions::port::Port;
    unsafe {
        let mut port = Port::new(0xf4);
        port.write(exit_code as u32);
    }
}

/// Shared panic reporter for the headless self-test path: report over serial,
/// then signal failure through `isa-debug-exit`.
pub fn test_panic_handler(info: &PanicInfo) -> ! {
    serial_println!("[failed]\n");
    serial_println!("Error: {}\n", info);
    exit_qemu(QemuExitCode::Failed);
    hlt_loop();
}

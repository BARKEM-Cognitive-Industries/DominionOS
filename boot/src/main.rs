//! DominionOS image builder.
//!
//! Takes a compiled kernel ELF and produces a bootable BIOS disk image (and,
//! when asked, a UEFI image too) using the `bootloader` crate. This is the host
//! tool that turns `dominion-kernel` into something QEMU — or real hardware — can
//! boot.
//!
//! Usage:
//!   dominion-boot <kernel-elf> <out-bios-img> [out-uefi-img]

use std::path::Path;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dominion-boot <kernel-elf> <out-bios-img> [out-uefi-img]");
        exit(2);
    }

    let kernel = Path::new(&args[1]);
    if !kernel.exists() {
        eprintln!("error: kernel ELF not found: {}", kernel.display());
        exit(1);
    }
    let bios_out = Path::new(&args[2]);

    // Request a larger framebuffer for the graphical dashboard via the `RESOLUTION`
    // env var (e.g. "1920x1080"). When unset, the bootloader's default mode is used
    // (the desktop adapts to whatever resolution it actually gets).
    let resolution = std::env::var("RESOLUTION").ok().and_then(|s| {
        let mut it = s.split('x');
        Some((it.next()?.parse::<u64>().ok()?, it.next()?.parse::<u64>().ok()?))
    });
    let boot_config = resolution.map(|(rw, rh)| {
        let mut c = bootloader::BootConfig::default();
        c.frame_buffer.minimum_framebuffer_width = Some(rw);
        c.frame_buffer.minimum_framebuffer_height = Some(rh);
        c
    });

    println!("building BIOS disk image from {} (fb {:?})", kernel.display(), resolution);
    let mut bios = bootloader::BiosBoot::new(kernel);
    if let Some(c) = &boot_config {
        bios.set_boot_config(c);
    }
    bios.create_disk_image(bios_out).expect("failed to create BIOS disk image");
    println!("  -> {}", bios_out.display());

    if let Some(uefi) = args.get(3) {
        let uefi_out = Path::new(uefi);
        println!("building UEFI disk image");
        let mut ub = bootloader::UefiBoot::new(kernel);
        if let Some(c) = &boot_config {
            ub.set_boot_config(c);
        }
        ub.create_disk_image(uefi_out).expect("failed to create UEFI disk image");
        println!("  -> {}", uefi_out.display());
    }

    println!("done.");
}

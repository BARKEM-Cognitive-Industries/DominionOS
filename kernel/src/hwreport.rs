//! Hardware enumeration & report — "what is this machine?" Scans the PCI bus and
//! classifies every device (GPUs incl. integrated + discrete, storage controllers
//! NVMe/AHCI/IDE/virtio, NICs, USB, bridges), reads the CPU brand + feature flags via
//! CPUID, and reports core count and memory. Logged at boot (so it lands in the
//! [`crate::bootlog`] capture) and available on demand via the `hw` shell command —
//! the first thing you want when bringing the OS up on an unfamiliar real machine.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

fn vendor_name(id: u16) -> &'static str {
    match id {
        0x8086 => "Intel",
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x1AF4 | 0x1B36 => "Red Hat/virtio",
        0x10EC => "Realtek",
        0x14E4 => "Broadcom",
        0x168C | 0x17CB => "Qualcomm Atheros",
        0x1234 => "QEMU/Bochs",
        0x1969 => "Atheros",
        0x15B3 => "Mellanox",
        _ => "unknown",
    }
}

/// Human label for a PCI (class, subclass) pair — the device families that matter for
/// "does it work on this machine".
fn class_label(class: u8, subclass: u8) -> &'static str {
    match (class, subclass) {
        (0x00, _) => "unclassified",
        (0x01, 0x01) => "IDE storage",
        (0x01, 0x05) => "ATA storage",
        (0x01, 0x06) => "SATA/AHCI storage",
        (0x01, 0x08) => "NVMe storage",
        (0x01, _) => "storage controller",
        (0x02, 0x00) => "Ethernet",
        (0x02, 0x80) => "network (wireless?)",
        (0x02, _) => "network controller",
        (0x03, _) => "display/GPU",
        (0x04, _) => "multimedia",
        (0x06, _) => "bridge",
        (0x0C, 0x03) => "USB controller",
        (0x0C, _) => "serial bus",
        _ => "device",
    }
}

fn gpu_kind(vendor: u16) -> &'static str {
    match vendor {
        0x10DE => "NVIDIA GPU (discrete)",
        0x1002 => "AMD GPU (discrete)",
        0x8086 => "Intel GPU (integrated)",
        0x1234 => "QEMU/Bochs VGA",
        0x1B36 => "virtio-gpu",
        _ => "GPU",
    }
}

fn storage_kind(subclass: u8) -> &'static str {
    match subclass {
        0x01 => "IDE",
        0x05 => "ATA",
        0x06 => "SATA/AHCI",
        0x08 => "NVMe",
        _ => "storage",
    }
}

/// Online core count — SMP gives the real number; without it, the BSP alone.
fn core_count() -> u32 {
    #[cfg(feature = "smp")]
    {
        crate::smp::core_count()
    }
    #[cfg(not(feature = "smp"))]
    {
        1
    }
}

fn cpu_brand() -> String {
    // CPUID extended leaves 0x80000002..4 carry the 48-byte brand string.
    let max_ext = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    if max_ext < 0x8000_0004 {
        return "x86-64 CPU".to_string();
    }
    let mut s = [0u8; 48];
    for (i, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004].iter().enumerate() {
        let r = core::arch::x86_64::__cpuid(*leaf);
        for (j, reg) in [r.eax, r.ebx, r.ecx, r.edx].iter().enumerate() {
            s[i * 16 + j * 4..i * 16 + j * 4 + 4].copy_from_slice(&reg.to_le_bytes());
        }
    }
    String::from_utf8_lossy(&s).trim_matches(|c: char| c == '\0' || c == ' ').to_string()
}

fn cpu_features() -> String {
    let r = core::arch::x86_64::__cpuid(1);
    let (edx, ecx) = (r.edx, r.ecx);
    let mut f: Vec<&str> = Vec::new();
    let check = |bits: u32, bit: u32| bits & (1 << bit) != 0;
    if check(edx, 25) {
        f.push("sse");
    }
    if check(edx, 26) {
        f.push("sse2");
    }
    if check(ecx, 0) {
        f.push("sse3");
    }
    if check(ecx, 19) {
        f.push("sse4.1");
    }
    if check(ecx, 20) {
        f.push("sse4.2");
    }
    if check(ecx, 28) {
        f.push("avx");
    }
    if check(ecx, 12) {
        f.push("fma");
    }
    if check(ecx, 30) {
        f.push("rdrand");
    }
    if check(ecx, 31) {
        f.push("hypervisor");
    }
    f.join(" ")
}

/// Build the full hardware report as lines (for the `hw` command and the boot log).
pub fn report(usable_frames: usize) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("== DominionOS hardware report ==".to_string());
    lines.push(format!("CPU:    {}", cpu_brand()));
    lines.push(format!("        cores online: {}", core_count()));
    lines.push(format!("        features: {}", cpu_features()));
    lines.push(format!(
        "Memory: {} usable frames (~{} MiB)",
        usable_frames,
        (usable_frames as u64 * 4096) / (1024 * 1024)
    ));

    let devs = crate::pci::enumerate();
    lines.push(format!("PCI:    {} devices", devs.len()));
    let (mut gpus, mut storage, mut nets, mut usb) = (Vec::new(), Vec::new(), Vec::new(), 0u32);
    for d in &devs {
        lines.push(format!(
            "  {:04x}:{:04x} {:<22} [{}]",
            d.vendor_id,
            d.device_id,
            class_label(d.class_code, d.subclass),
            vendor_name(d.vendor_id)
        ));
        match d.class_code {
            0x03 => gpus.push(d),
            0x01 => storage.push(d),
            0x02 => nets.push(d),
            0x0C if d.subclass == 0x03 => usb += 1,
            _ => {}
        }
    }

    lines.push(format!("GPUs:   {}", gpus.len()));
    for g in &gpus {
        lines.push(format!("  - {} {:04x}:{:04x}", gpu_kind(g.vendor_id), g.vendor_id, g.device_id));
    }
    lines.push(format!("Storage: {}", storage.len()));
    for s in &storage {
        lines.push(format!(
            "  - {} {:04x}:{:04x}",
            storage_kind(s.subclass),
            s.vendor_id,
            s.device_id
        ));
    }
    lines.push(format!("Network: {}", nets.len()));
    for n in &nets {
        let kind = if n.subclass == 0x80 { "wireless?" } else { "Ethernet" };
        lines.push(format!("  - {} {:04x}:{:04x} ({})", kind, n.vendor_id, n.device_id, vendor_name(n.vendor_id)));
    }
    lines.push(format!("USB controllers: {}", usb));
    lines
}

/// Log the hardware report to serial (and thus into the boot-log capture).
pub fn log_report(usable_frames: usize) {
    for line in report(usable_frames) {
        crate::serial_println!("{}", line);
    }
}

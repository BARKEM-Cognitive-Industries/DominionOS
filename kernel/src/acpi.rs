//! Minimal ACPI support for power management — specifically a *real* S5
//! (soft-off) power-off, the kind that works on bare metal rather than only
//! inside QEMU.
//!
//! The naive way to power a PC off — what this kernel did before — is to write
//! the magic word `0x2000` to I/O port `0x604`. That works under QEMU because
//! QEMU hard-wires that port to the ACPI PM1a control block with an `_S5`
//! sleep type of 0, so `0x2000` happens to be exactly `SLP_EN`. On a physical
//! machine the PM1a control register lives at a firmware-chosen I/O address and
//! the S5 sleep type is almost never 0, so the magic write does nothing and the
//! box just sits there with the CPU halted.
//!
//! Doing it properly means walking the ACPI tables the firmware left in memory:
//!
//! ```text
//!   RSDP  ──►  RSDT / XSDT  ──►  FADT ("FACP")  ──►  PM1a_CNT / PM1b_CNT ports
//!                                     │
//!                                     └──►  DSDT  ──(scan AML)──►  \_S5 → SLP_TYPa/b
//! ```
//!
//! The FADT gives us the I/O port(s) of the PM1 control register(s); the DSDT's
//! `\_S5_` AML package gives us the `SLP_TYP` value to program into the
//! `SLP_TYP` field (bits 10–12) of that register. Setting `SLP_EN` (bit 13)
//! then latches the transition and the machine powers off.
//!
//! We parse the tables once at boot (when the bootloader's physical-memory
//! mapping and the RSDP pointer are in hand) and stash the few values
//! [`poweroff`] needs into a global, because the power-off path itself has no
//! access to `phys_offset`.

use core::ptr::read_unaligned;
use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};

/// `SLP_EN` — bit 13 of the PM1 control register. Writing it latches whatever
/// sleep type is in bits 10–12 and performs the transition.
const SLP_EN: u16 = 1 << 13;
/// `SCI_EN` — bit 0 of the PM1 control register. Set when the firmware is in
/// ACPI mode (as opposed to legacy mode); a prerequisite for `SLP_EN` to work.
const SCI_EN: u16 = 1 << 0;

// Parsed-once S5 sleep parameters. Stored as plain atomics so [`poweroff`] is a
// lock-free leaf that can run from any context (including a panic path).
static READY: AtomicBool = AtomicBool::new(false);
static PM1A_CNT: AtomicU32 = AtomicU32::new(0);
static PM1B_CNT: AtomicU32 = AtomicU32::new(0);
static SLP_TYPA: AtomicU16 = AtomicU16::new(0);
static SLP_TYPB: AtomicU16 = AtomicU16::new(0);
static SMI_CMD: AtomicU32 = AtomicU32::new(0);
static ACPI_ENABLE: AtomicU16 = AtomicU16::new(0); // u8 value; u16 for atomic convenience

/// Parse the ACPI tables for the S5 power-off parameters and cache them.
///
/// `phys_offset` is the base at which the bootloader mapped all of physical
/// memory; `rsdp` is the physical address of the Root System Description
/// Pointer the bootloader recovered from the firmware (`None` on the rare
/// machine where it could not be found). Safe to call once, early in boot.
///
/// Returns `true` if a usable PM1a control port and S5 sleep type were found.
pub fn init(phys_offset: u64, rsdp: Option<u64>) -> bool {
    let rsdp_phys = match rsdp {
        Some(p) if p != 0 => p,
        _ => {
            crate::serial_println!("[acpi] no RSDP; ACPI power-off unavailable");
            return false;
        }
    };
    let found = unsafe { parse(phys_offset, rsdp_phys) };
    if found {
        READY.store(true, Ordering::SeqCst);
        crate::serial_println!(
            "[acpi] S5 ready: PM1a_CNT={:#06x} PM1b_CNT={:#06x} SLP_TYPa={} SLP_TYPb={}",
            PM1A_CNT.load(Ordering::Relaxed),
            PM1B_CNT.load(Ordering::Relaxed),
            SLP_TYPA.load(Ordering::Relaxed) >> 10,
            SLP_TYPB.load(Ordering::Relaxed) >> 10,
        );
    } else {
        crate::serial_println!("[acpi] could not derive S5 parameters; will fall back to port magic");
    }
    found
}

/// True once [`init`] has successfully cached S5 parameters.
pub fn available() -> bool {
    READY.load(Ordering::SeqCst)
}

/// Attempt a real ACPI S5 power-off using the cached parameters.
///
/// On success this never returns — the machine powers off. Returns `false`
/// (without halting) if the parameters are unavailable, so the caller can fall
/// back to another mechanism.
pub fn poweroff() -> bool {
    if !READY.load(Ordering::SeqCst) {
        return false;
    }
    let pm1a = PM1A_CNT.load(Ordering::Relaxed) as u16;
    let pm1b = PM1B_CNT.load(Ordering::Relaxed) as u16;
    let slp_a = SLP_TYPA.load(Ordering::Relaxed);
    let slp_b = SLP_TYPB.load(Ordering::Relaxed);
    let smi_cmd = SMI_CMD.load(Ordering::Relaxed);
    let acpi_enable = ACPI_ENABLE.load(Ordering::Relaxed) as u8;

    unsafe {
        use x86_64::instructions::port::Port;

        // Step 1: make sure the firmware is in ACPI mode. If SCI_EN is clear and
        // the FADT advertised an SMI command port + enable value, poke it and
        // wait (bounded) for the firmware to flip into ACPI mode. Many machines
        // — and QEMU — already boot in ACPI mode, in which case this is a no-op.
        let mut pm1a_port: Port<u16> = Port::new(pm1a);
        if pm1a_port.read() & SCI_EN == 0 && smi_cmd != 0 && acpi_enable != 0 {
            let mut smi: Port<u8> = Port::new(smi_cmd as u16);
            smi.write(acpi_enable);
            // Spin until SCI_EN appears, with a cap so we never wedge forever.
            let mut spins = 0u32;
            while pm1a_port.read() & SCI_EN == 0 && spins < 3_000_000 {
                core::hint::spin_loop();
                spins += 1;
            }
        }

        // Step 2: write SLP_TYP | SLP_EN to PM1a (and PM1b if present). This is
        // the transition; on real hardware the machine cuts power here.
        pm1a_port.write(slp_a | SLP_EN);
        if pm1b != 0 {
            let mut pm1b_port: Port<u16> = Port::new(pm1b);
            pm1b_port.write(slp_b | SLP_EN);
        }
    }

    // If we get here the transition did not take immediately; give the hardware
    // a moment by spinning briefly, then report failure so the caller falls back.
    for _ in 0..1_000_000 {
        core::hint::spin_loop();
    }
    false
}

/// Walk RSDP → RSDT/XSDT → FADT → DSDT and populate the cached parameters.
unsafe fn parse(phys_offset: u64, rsdp_phys: u64) -> bool {
    let read = |p: u64| -> u64 { phys_offset + p };

    // --- RSDP: pick RSDT (ACPI 1.0) or XSDT (ACPI 2.0+) by revision. ---
    let rsdp = read(rsdp_phys);
    let revision = read_unaligned((rsdp + 15) as *const u8);
    let (sdt_phys, entry_size) = if revision >= 2 {
        (read_unaligned((rsdp + 24) as *const u64), 8usize) // XSDT, 64-bit entries
    } else {
        (read_unaligned((rsdp + 16) as *const u32) as u64, 4usize) // RSDT, 32-bit entries
    };
    if sdt_phys == 0 {
        return false;
    }

    // --- System description table: scan its entries for the FADT ("FACP"). ---
    let sdt = read(sdt_phys);
    let sdt_len = read_unaligned((sdt + 4) as *const u32) as usize;
    let entries = sdt_len.saturating_sub(36) / entry_size;
    let mut fadt_phys = 0u64;
    for i in 0..entries {
        let ent = sdt + 36 + (i * entry_size) as u64;
        let p = if entry_size == 8 {
            read_unaligned(ent as *const u64)
        } else {
            read_unaligned(ent as *const u32) as u64
        };
        let sig = core::slice::from_raw_parts(read(p) as *const u8, 4);
        if sig == b"FACP" {
            fadt_phys = p;
            break;
        }
    }
    if fadt_phys == 0 {
        return false;
    }

    // --- FADT ("FACP"): pull the PM1 control ports, SMI/enable, and DSDT ptr. ---
    let fadt = read(fadt_phys);
    let fadt_len = read_unaligned((fadt + 4) as *const u32) as usize;

    let smi_cmd = read_unaligned((fadt + 48) as *const u32);
    let acpi_enable = read_unaligned((fadt + 52) as *const u8);
    let pm1a_cnt = read_unaligned((fadt + 64) as *const u32);
    let pm1b_cnt = read_unaligned((fadt + 68) as *const u32);

    // DSDT: 32-bit pointer at offset 40, or the 64-bit X_DSDT at offset 140 on
    // ACPI 2.0+ FADTs that are long enough to contain it and populate it.
    let mut dsdt_phys = read_unaligned((fadt + 40) as *const u32) as u64;
    if fadt_len >= 148 {
        let x_dsdt = read_unaligned((fadt + 140) as *const u64);
        if x_dsdt != 0 {
            dsdt_phys = x_dsdt;
        }
    }

    if pm1a_cnt == 0 || dsdt_phys == 0 {
        return false;
    }

    // --- DSDT: scan the AML byte stream for the \_S5_ package. ---
    let (slp_typa, slp_typb) = match find_s5(phys_offset, dsdt_phys) {
        Some(v) => v,
        None => return false,
    };

    PM1A_CNT.store(pm1a_cnt, Ordering::Relaxed);
    PM1B_CNT.store(pm1b_cnt, Ordering::Relaxed);
    SLP_TYPA.store(slp_typa, Ordering::Relaxed);
    SLP_TYPB.store(slp_typb, Ordering::Relaxed);
    SMI_CMD.store(smi_cmd, Ordering::Relaxed);
    ACPI_ENABLE.store(acpi_enable as u16, Ordering::Relaxed);
    true
}

/// Locate the `\_S5_` package in a DSDT's AML and return the two sleep-type
/// values already shifted into the `SLP_TYP` field position (bits 10–12), ready
/// to OR with `SLP_EN`.
///
/// The `_S5_` object in compiled AML looks like:
///
/// ```text
///   08 5F 53 35 5F  12  <PkgLength>  <NumElements>  <SLP_TYPa>  <SLP_TYPb> ...
///   ^NameOp ^"_S5_"  ^PackageOp
/// ```
///
/// Optionally the name is root-scoped, so it can be preceded by `5C` (`\`).
/// Integer elements may be inline (a raw byte) or prefixed with the BytePrefix
/// opcode `0x0A`. We validate the surrounding opcodes before trusting a hit so
/// a stray `_S5_` in a string literal cannot fool us.
unsafe fn find_s5(phys_offset: u64, dsdt_phys: u64) -> Option<(u16, u16)> {
    let base = phys_offset + dsdt_phys;
    let len = read_unaligned((base + 4) as *const u32) as usize;
    if len < 36 || len > 0x40_0000 {
        return None; // implausible DSDT length — refuse to scan into the weeds
    }
    let aml = core::slice::from_raw_parts(base as *const u8, len);

    // Scan the AML body (after the 36-byte SDT header) for the name "_S5_".
    let mut i = 36;
    while i + 5 <= len {
        if &aml[i..i + 4] == b"_S5_" {
            // Validate the NameOp that should introduce it: either `08 _S5_`
            // or root-scoped `08 5C _S5_`.
            let name_op = (i >= 1 && aml[i - 1] == 0x08)
                || (i >= 2 && aml[i - 2] == 0x08 && aml[i - 1] == 0x5C);
            // And the value must be a PackageOp (0x12) right after the name.
            if name_op && aml[i + 4] == 0x12 {
                // Skip "_S5_" (4) + PackageOp (1).
                let mut p = i + 5;
                if p >= len {
                    return None;
                }
                // PkgLength: the top two bits of the lead byte give the count of
                // *extra* length bytes (0–3). Skip the whole PkgLength field plus
                // the following NumElements byte → (extra + 1) + 1 = extra + 2.
                let extra = (aml[p] >> 6) as usize;
                p += extra + 2;

                // First element → SLP_TYPa. Inline byte, or 0x0A-prefixed byte.
                if p >= len {
                    return None;
                }
                if aml[p] == 0x0A {
                    p += 1;
                }
                if p >= len {
                    return None;
                }
                let slp_a = (aml[p] as u16) << 10;
                p += 1;

                // Second element → SLP_TYPb (defaults to 0 if truncated).
                let mut slp_b = 0u16;
                if p < len {
                    if aml[p] == 0x0A {
                        p += 1;
                    }
                    if p < len {
                        slp_b = (aml[p] as u16) << 10;
                    }
                }
                return Some((slp_a, slp_b));
            }
        }
        i += 1;
    }
    None
}

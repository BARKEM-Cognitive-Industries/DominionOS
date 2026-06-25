//! Foreign-binary driver **loader** scaffold — turning a real Windows `.sys`
//! (PE/COFF) or Linux `.ko` (ELF) into the admission descriptor that
//! [`crate::foreign`] confines, plus an explicit boundary on what *executes* vs
//! what is *modeled* (see `docs/architecture/capability-shim-and-foreign-compat.md`
//! §3.2).
//!
//! The pipeline:
//!
//! ```text
//!   bytes ─► detect PE/ELF ─► parse imported kernel symbols ─► ForeignDriver
//!                                       │
//!                                       ▼
//!              foreign::ForeignHost::load  (default-closed symbol shim +
//!                                           capability bounded to the device)
//!                                       │
//!                                       ▼
//!              lower_to_spec  ─► a DeviceSpec on the same bounded runtime
//! ```
//!
//! **Honest boundary.** What runs today: PE/ELF parsing, import resolution against
//! the [`KpiShim`](crate::foreign::KpiShim) whitelist, admission, and lowering a
//! *recognised* driver to a native [`DeviceSpec`] that drives real/modeled hardware.
//! What is **modeled, not executed**: the foreign driver's own machine code. Running
//! that in place needs an x86 interpreter/JIT inside a SIP sandbox plus the full
//! NDIS / Linux-KPI surface — a multi-year effort. [`ExecBoundary`] names exactly
//! where execution stops, so nothing here pretends to be more than it is.
//!
//! Pure, safe `no_std`, host-tested with constructed minimal binaries.

use crate::compat::{detect_format, BinaryFormat};
use crate::driver::{DeviceClass, DeviceSpec, ResourceClaim};
use crate::foreign::{rd_u16, rd_u32, rd_u64, ForeignAbi, ForeignDriver, LoadError};
use crate::netspec;
use alloc::string::String;
use alloc::vec::Vec;

/// How far DominionOS can take a given foreign binary — made explicit so a borrowed
/// driver's status is never overstated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecBoundary {
    /// Parsed, admitted, and **lowered to a native [`DeviceSpec`]** that runs on the
    /// bounded runtime — full end-to-end via the native path.
    LoweredToSpec,
    /// Parsed and admitted (imports resolved, capability minted), but the driver's
    /// own code is **not executed** — awaiting the in-sandbox x86 JIT.
    AdmittedNotExecuted,
}

/// The result of inspecting a foreign driver binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedImage {
    pub format: BinaryFormat,
    /// Kernel symbols the binary imports (what the KPI shim must provide).
    pub imports: Vec<String>,
}

// ───────────────────────────── ELF (.ko) ─────────────────────────────
// rd_u16 / rd_u32 / rd_u64 are imported from crate::foreign (shared readers).

fn cstr(b: &[u8], o: usize) -> Option<String> {
    let s = b.get(o..)?;
    let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    Some(String::from_utf8_lossy(&s[..end]).into_owned())
}

/// Parse the **undefined** symbols of an ELF64 object — exactly the kernel API a
/// Linux `.ko` needs the host to resolve.
pub fn parse_elf_imports(b: &[u8]) -> Option<Vec<String>> {
    // ELF64 little-endian only (the kernels we target).
    if b.len() < 64 || &b[0..4] != b"\x7FELF" || b[4] != 2 || b[5] != 1 {
        return None;
    }
    let e_shoff = rd_u64(b, 0x28)? as usize;
    let e_shentsize = rd_u16(b, 0x3A)? as usize;
    let e_shnum = rd_u16(b, 0x3C)? as usize;
    if e_shentsize < 64 {
        return None;
    }
    let mut imports = Vec::new();
    // Find a symbol table (SHT_SYMTAB=2 or SHT_DYNSYM=11) and its string table.
    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        let sh_type = rd_u32(b, sh + 4)?;
        if sh_type != 2 && sh_type != 11 {
            continue;
        }
        let sym_off = rd_u64(b, sh + 0x18)? as usize;
        let sym_size = rd_u64(b, sh + 0x20)? as usize;
        let strtab_idx = rd_u32(b, sh + 0x28)? as usize;
        let sym_entsize = rd_u64(b, sh + 0x38)? as usize;
        if sym_entsize < 24 || strtab_idx >= e_shnum {
            continue;
        }
        let str_sh = e_shoff + strtab_idx * e_shentsize;
        let str_off = rd_u64(b, str_sh + 0x18)? as usize;
        let count = sym_size / sym_entsize;
        for s in 0..count {
            let sym = sym_off + s * sym_entsize;
            let st_name = rd_u32(b, sym)? as usize;
            let st_shndx = rd_u16(b, sym + 6)?;
            // SHN_UNDEF (0) with a name ⇒ an imported symbol.
            if st_shndx == 0 && st_name != 0 {
                if let Some(name) = cstr(b, str_off + st_name) {
                    if !name.is_empty() && !imports.contains(&name) {
                        imports.push(name);
                    }
                }
            }
        }
    }
    Some(imports)
}

// ───────────────────────────── PE (.sys) ─────────────────────────────

/// Translate an RVA to a file offset using the section table.
fn rva_to_off(_b: &[u8], sections: &[(u32, u32, u32)], rva: u32) -> Option<usize> {
    for &(va, size, raw) in sections {
        if rva >= va && rva < va + size {
            return Some((raw + (rva - va)) as usize);
        }
    }
    None
}

/// Parse the import directory of a PE/COFF image (`.sys`/`.dll`/`.exe`), returning
/// the names of every imported function — the Windows kernel API surface the driver
/// needs the NDIS-style shim to provide.
pub fn parse_pe_imports(b: &[u8]) -> Option<Vec<String>> {
    if b.len() < 0x40 || &b[0..2] != b"MZ" {
        return None;
    }
    let e_lfanew = rd_u32(b, 0x3C)? as usize;
    if b.get(e_lfanew..e_lfanew + 4)? != b"PE\0\0" {
        return None;
    }
    let coff = e_lfanew + 4;
    let num_sections = rd_u16(b, coff + 2)? as usize;
    let opt_size = rd_u16(b, coff + 16)? as usize;
    let opt = coff + 20;
    let magic = rd_u16(b, opt)?;
    // Import-table data directory index is 1. Its position depends on PE32 vs PE32+.
    let (num_dirs_off, dirs_off) = match magic {
        0x10B => (opt + 92, opt + 96),  // PE32
        0x20B => (opt + 108, opt + 112), // PE32+
        _ => return None,
    };
    let num_dirs = rd_u32(b, num_dirs_off)?;
    if num_dirs < 2 {
        return Some(Vec::new()); // no import directory
    }
    let import_rva = rd_u32(b, dirs_off + 8)?; // directory[1].VirtualAddress
    if import_rva == 0 {
        return Some(Vec::new());
    }
    // Section table follows the optional header.
    let sec_table = opt + opt_size;
    let mut sections = Vec::new();
    for i in 0..num_sections {
        let s = sec_table + i * 40;
        let va = rd_u32(b, s + 12)?;
        let vsize = rd_u32(b, s + 8)?;
        let raw_size = rd_u32(b, s + 16)?;
        let raw_ptr = rd_u32(b, s + 20)?;
        sections.push((va, vsize.max(raw_size), raw_ptr));
    }

    let pe32_plus = magic == 0x20B;
    let mut imports = Vec::new();
    let mut desc = rva_to_off(b, &sections, import_rva)?;
    // Walk IMAGE_IMPORT_DESCRIPTORs (20 bytes) until the all-zero terminator.
    loop {
        let oft = rd_u32(b, desc)?; // OriginalFirstThunk (ILT)
        let name_rva = rd_u32(b, desc + 12)?;
        let first_thunk = rd_u32(b, desc + 16)?;
        if oft == 0 && name_rva == 0 && first_thunk == 0 {
            break;
        }
        let thunk_rva = if oft != 0 { oft } else { first_thunk };
        if let Some(mut thunk) = rva_to_off(b, &sections, thunk_rva) {
            loop {
                let (entry, by_ordinal) = if pe32_plus {
                    let v = rd_u64(b, thunk)?;
                    (v & 0x7FFF_FFFF, v & 0x8000_0000_0000_0000 != 0)
                } else {
                    let v = rd_u32(b, thunk)? as u64;
                    (v & 0x7FFF_FFFF, v & 0x8000_0000 != 0)
                };
                if entry == 0 && !by_ordinal {
                    break;
                }
                if !by_ordinal {
                    // IMAGE_IMPORT_BY_NAME: u16 hint then the name.
                    if let Some(off) = rva_to_off(b, &sections, entry as u32) {
                        if let Some(name) = cstr(b, off + 2) {
                            if !name.is_empty() && !imports.contains(&name) {
                                imports.push(name);
                            }
                        }
                    }
                }
                thunk += if pe32_plus { 8 } else { 4 };
            }
        }
        desc += 20;
    }
    Some(imports)
}

// ─────────────────────────── unified loader ───────────────────────────

/// Inspect a foreign driver binary: detect its container and extract the kernel
/// symbols it imports.
pub fn inspect(bytes: &[u8]) -> Option<LoadedImage> {
    let format = detect_format(bytes);
    let imports = match format {
        BinaryFormat::Pe => parse_pe_imports(bytes)?,
        BinaryFormat::Elf => parse_elf_imports(bytes)?,
        _ => return None,
    };
    Some(LoadedImage { format, imports })
}

/// Build the admission descriptor for a foreign driver from its **actual bytes**:
/// parse the imports and pair them with the declared ABI, class and resource claim.
/// The returned [`ForeignDriver`] is then admitted by
/// [`ForeignHost::load`](crate::foreign::ForeignHost::load) — which enforces the
/// default-closed shim and the capability bound. `BadImage` if it cannot be parsed.
pub fn prepare(
    bytes: &[u8],
    name: &str,
    abi: ForeignAbi,
    class: DeviceClass,
    claim: ResourceClaim,
) -> Result<ForeignDriver, LoadError> {
    let image = inspect(bytes).ok_or(LoadError::BadImage)?;
    let imports: Vec<&str> = image.imports.iter().map(|s| s.as_str()).collect();
    Ok(ForeignDriver::new(name, abi, class, claim).imports(&imports))
}

/// Lower a recognised borrowed driver to a native [`DeviceSpec`], so it runs through
/// the **same bounded runtime** as a native spec — the end-to-end native path. The
/// match is by name today (a real system would match PCI ids in the driver's `.inf`/
/// `modalias`); unrecognised drivers return `AdmittedNotExecuted`, the honest state.
pub fn lower_to_spec(driver: &ForeignDriver) -> (Option<DeviceSpec>, ExecBoundary) {
    let n = driver.name.to_ascii_lowercase();
    let base = driver.claim.mmio_base;
    let irq = driver.claim.irq;
    if n.contains("8139") || n.contains("rtl81") {
        (Some(netspec::rtl8139_spec(base, irq)), ExecBoundary::LoweredToSpec)
    } else if n.contains("e1000") || n.contains("e1g") || n.contains("82540") {
        (Some(netspec::e1000_spec(base, irq)), ExecBoundary::LoweredToSpec)
    } else {
        (None, ExecBoundary::AdmittedNotExecuted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheri::SoftwareTags;
    use crate::driver::{Driver, ModelDmaMem};
    use crate::foreign::{ForeignHost, KpiShim};
    use alloc::vec;

    // ── Build a minimal but real ELF64 .ko with two undefined symbols. ──
    fn build_elf_ko(undef: &[&str]) -> Vec<u8> {
        // Layout: [ehdr 64][symtab][strtab][shdrs].
        let mut strtab = vec![0u8]; // index 0 = ""
        let mut name_offs = Vec::new();
        for s in undef {
            name_offs.push(strtab.len() as u32);
            strtab.extend_from_slice(s.as_bytes());
            strtab.push(0);
        }
        // Symbol table: one null symbol + one per undefined name.
        let mut symtab = vec![0u8; 24]; // null symbol
        for &no in &name_offs {
            let mut sym = vec![0u8; 24];
            sym[0..4].copy_from_slice(&no.to_le_bytes()); // st_name
            sym[4] = 0x10; // st_info: GLOBAL
            sym[6..8].copy_from_slice(&0u16.to_le_bytes()); // st_shndx = SHN_UNDEF
            symtab.extend_from_slice(&sym);
        }
        let mut b = vec![0u8; 64];
        b[0..4].copy_from_slice(b"\x7FELF");
        b[4] = 2; // 64-bit
        b[5] = 1; // little-endian
        let symtab_off = b.len();
        b.extend_from_slice(&symtab);
        let strtab_off = b.len();
        b.extend_from_slice(&strtab);
        let shoff = b.len();
        // 4 section headers: null, .symtab(2), .strtab(3), .shstrtab(3-ish unused).
        let mk_sh = |sh_type: u32, off: usize, size: usize, link: u32, entsize: usize| {
            let mut sh = vec![0u8; 64];
            sh[4..8].copy_from_slice(&sh_type.to_le_bytes());
            sh[0x18..0x20].copy_from_slice(&(off as u64).to_le_bytes());
            sh[0x20..0x28].copy_from_slice(&(size as u64).to_le_bytes());
            sh[0x28..0x2C].copy_from_slice(&link.to_le_bytes());
            sh[0x38..0x40].copy_from_slice(&(entsize as u64).to_le_bytes());
            sh
        };
        let mut shdrs = Vec::new();
        shdrs.extend_from_slice(&mk_sh(0, 0, 0, 0, 0)); // null (index 0)
        shdrs.extend_from_slice(&mk_sh(2, symtab_off, symtab.len(), 2, 24)); // .symtab → link strtab idx 2
        shdrs.extend_from_slice(&mk_sh(3, strtab_off, strtab.len(), 0, 0)); // .strtab (index 2)
        b.extend_from_slice(&shdrs);
        // ehdr section-header fields.
        b[0x28..0x30].copy_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
        b[0x3A..0x3C].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
        b[0x3C..0x3E].copy_from_slice(&3u16.to_le_bytes()); // e_shnum
        b
    }

    #[test]
    fn parses_linux_ko_imports() {
        let ko = build_elf_ko(&["kmalloc", "netif_rx"]);
        let img = inspect(&ko).unwrap();
        assert_eq!(img.format, BinaryFormat::Elf);
        assert!(img.imports.contains(&"kmalloc".to_string()));
        assert!(img.imports.contains(&"netif_rx".to_string()));
    }

    // ── Build a minimal but real PE32+ .sys importing two functions. ──
    fn build_pe_sys(funcs: &[&str]) -> Vec<u8> {
        // One section ".idata" at RVA 0x1000 / raw 0x200 holds: import descriptors,
        // ILT, hint/name entries, and the DLL name.
        let sec_rva = 0x1000u32;
        let sec_raw = 0x200usize;
        let mut idata = Vec::new();
        // Reserve space for descriptors (one + null terminator = 40 bytes).
        let desc_off = 0usize;
        idata.resize(40, 0);
        // ILT (PE32+: u64 thunks) after descriptors.
        let ilt_off = idata.len();
        let mut thunks: Vec<u64> = Vec::new();
        // Names placed after ILT; compute offsets as we go (filled below).
        let mut name_blobs = Vec::new();
        let ilt_bytes = (funcs.len() + 1) * 8;
        let mut cursor = ilt_off + ilt_bytes;
        for f in funcs {
            let name_rva = sec_rva as usize + cursor; // hint(2) + name + nul
            thunks.push(name_rva as u64);
            let mut blob = vec![0u8, 0u8]; // hint
            blob.extend_from_slice(f.as_bytes());
            blob.push(0);
            cursor += blob.len();
            name_blobs.push(blob);
        }
        thunks.push(0); // ILT terminator
        // DLL name.
        let dll_rva = sec_rva as usize + cursor;
        let dll = b"ndis.sys\0";
        // Assemble idata: descriptors, ILT, names, dll.
        // descriptor[0]: OFT=ilt rva, Name=dll rva, FirstThunk=ilt rva.
        let ilt_rva = sec_rva as usize + ilt_off;
        idata[0..4].copy_from_slice(&(ilt_rva as u32).to_le_bytes()); // OriginalFirstThunk
        idata[12..16].copy_from_slice(&(dll_rva as u32).to_le_bytes()); // Name
        idata[16..20].copy_from_slice(&(ilt_rva as u32).to_le_bytes()); // FirstThunk
        // descriptor[1] left as zero terminator (bytes 20..40).
        for t in &thunks {
            idata.extend_from_slice(&t.to_le_bytes());
        }
        for blob in &name_blobs {
            idata.extend_from_slice(blob);
        }
        idata.extend_from_slice(dll);
        let _ = desc_off;

        // Build the file: DOS header, PE, COFF, optional (PE32+), one section.
        let e_lfanew = 0x80usize;
        let mut b = vec![0u8; e_lfanew];
        b[0..2].copy_from_slice(b"MZ");
        b[0x3C..0x40].copy_from_slice(&(e_lfanew as u32).to_le_bytes());
        b.extend_from_slice(b"PE\0\0");
        // COFF header (20 bytes).
        let opt_size = 0xF0usize; // PE32+ optional header w/ 16 data dirs
        let mut coff = vec![0u8; 20];
        coff[0..2].copy_from_slice(&0x8664u16.to_le_bytes()); // Machine x64
        coff[2..4].copy_from_slice(&1u16.to_le_bytes()); // 1 section
        coff[16..18].copy_from_slice(&(opt_size as u16).to_le_bytes());
        b.extend_from_slice(&coff);
        // Optional header (PE32+).
        let mut opt = vec![0u8; opt_size];
        opt[0..2].copy_from_slice(&0x20Bu16.to_le_bytes()); // PE32+ magic
        opt[108..112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes
        // data directory[1] (import) @ 112 + 1*8.
        opt[120..124].copy_from_slice(&sec_rva.to_le_bytes()); // import RVA
        opt[124..128].copy_from_slice(&(idata.len() as u32).to_le_bytes());
        b.extend_from_slice(&opt);
        // Section header (40 bytes) for .idata.
        let mut sh = vec![0u8; 40];
        sh[0..6].copy_from_slice(b".idata");
        sh[8..12].copy_from_slice(&(idata.len() as u32).to_le_bytes()); // VirtualSize
        sh[12..16].copy_from_slice(&sec_rva.to_le_bytes()); // VirtualAddress
        sh[16..20].copy_from_slice(&(idata.len() as u32).to_le_bytes()); // SizeOfRawData
        sh[20..24].copy_from_slice(&(sec_raw as u32).to_le_bytes()); // PointerToRawData
        b.extend_from_slice(&sh);
        // Pad to raw section start, then the section bytes.
        if b.len() < sec_raw {
            b.resize(sec_raw, 0);
        }
        b.extend_from_slice(&idata);
        b
    }

    #[test]
    fn parses_windows_sys_imports() {
        let sys = build_pe_sys(&["NdisMRegisterMiniport", "NdisAllocateMemory"]);
        let img = inspect(&sys).unwrap();
        assert_eq!(img.format, BinaryFormat::Pe);
        assert!(img.imports.contains(&"NdisMRegisterMiniport".to_string()));
        assert!(img.imports.contains(&"NdisAllocateMemory".to_string()));
    }

    #[test]
    fn end_to_end_admit_a_real_windows_nic_sys_and_lower_to_spec() {
        // A borrowed Windows rtl8139 NDIS driver, by its actual bytes.
        let sys = build_pe_sys(&["NdisMRegisterMiniport", "NdisAllocateMemory"]);
        let claim = ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 };
        let driver = prepare(&sys, "rtl8139.sys", ForeignAbi::WindowsNdis, DeviceClass::Net, claim).unwrap();

        // Admission: the default-closed NDIS shim must provide every import, and the
        // driver is bounded to exactly its device window.
        let host = ForeignHost::new(
            KpiShim::ndis(),
            ResourceClaim { mmio_base: 0xFEB0_0000, mmio_len: 0x10_0000, irq: 11 },
        );
        let tags = SoftwareTags::new([9u8; 32]);
        let contained = host.load(&driver, &tags).expect("admitted");
        assert!(contained.is_authentic(&tags));
        assert!(!contained.may_access(0xFEB0_0000, 4)); // cannot reach outside its window

        // Lower to a native spec → it runs on the same bounded runtime as any driver.
        let (spec, boundary) = lower_to_spec(&driver);
        assert_eq!(boundary, ExecBoundary::LoweredToSpec);
        let spec = spec.unwrap();
        let mut dma = ModelDmaMem::new();
        assert!(Driver::bind_dma(spec, &tags, &mut dma).is_ok());
    }

    #[test]
    fn a_driver_importing_an_unshimmed_symbol_is_refused() {
        // Default-closed: an import the shim does not provide blocks the load.
        let ko = build_elf_ko(&["kmalloc", "evil_backdoor_call"]);
        let claim = ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 };
        let driver = prepare(&ko, "rtl8139.ko", ForeignAbi::LinuxKpi, DeviceClass::Net, claim).unwrap();
        let host = ForeignHost::new(
            KpiShim::linuxkpi(),
            ResourceClaim { mmio_base: 0xFEB0_0000, mmio_len: 0x10_0000, irq: 11 },
        );
        let tags = SoftwareTags::new([9u8; 32]);
        match host.load(&driver, &tags) {
            Err(LoadError::MissingSymbol(s)) => assert_eq!(s, "evil_backdoor_call"),
            Err(other) => panic!("expected MissingSymbol, got {:?}", other),
            Ok(_) => panic!("expected the load to be refused (default-closed)"),
        }
    }

    #[test]
    fn unrecognised_driver_is_admitted_but_not_executed() {
        let ko = build_elf_ko(&["kmalloc"]);
        let claim = ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 };
        let driver = prepare(&ko, "exotic_fpga.ko", ForeignAbi::LinuxKpi, DeviceClass::Net, claim).unwrap();
        let (spec, boundary) = lower_to_spec(&driver);
        assert!(spec.is_none());
        assert_eq!(boundary, ExecBoundary::AdmittedNotExecuted);
    }
}

//! ELF64 parsing — the front half of the ELF loader (roadmap feature 3, gate M2).
//!
//! Dominion's *native* programs are Deterministic Compute Graphs, not ELF binaries
//! (SRS §1). But to absorb the legacy world (the Linux personality, ported tools)
//! the OS must understand the format the rest of computing ships in. This module
//! parses a 64-bit little-endian ELF and extracts what a loader needs: the entry
//! point and the `PT_LOAD` segments. It is pure and safe — the actual copy into a
//! capability-bounded region and the jump to the entry point happen in
//! `dominion-kernel`, which is where `unsafe` is allowed.

use alloc::vec::Vec;

/// Segment is mapped readable.
pub const PF_R: u32 = 4;
/// Segment is mapped writable.
pub const PF_W: u32 = 2;
/// Segment is mapped executable.
pub const PF_X: u32 = 1;

const PT_LOAD: u32 = 1;
const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EM_X86_64: u16 = 0x3E;

/// Hard upper bound on the number of program/section headers we will process.
/// Legitimate ELF binaries never approach this; it prevents O(n) abuse from
/// a crafted `e_phnum`/`e_shnum` field.
const MAX_PHNUM: usize = 65535;

/// Why an ELF blob was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElfError {
    TooShort,
    BadMagic,
    Not64Bit,
    NotLittleEndian,
    WrongMachine,
    BadProgramHeaders,
}

/// One loadable segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Segment {
    pub vaddr: u64,
    pub offset: u64,
    pub file_size: u64,
    pub mem_size: u64,
    pub flags: u32,
}

impl Segment {
    pub fn is_executable(&self) -> bool {
        self.flags & PF_X != 0
    }
    pub fn is_writable(&self) -> bool {
        self.flags & PF_W != 0
    }
}

/// A parsed ELF image: entry point plus its loadable segments.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ElfImage {
    pub entry: u64,
    pub segments: Vec<Segment>,
}

impl ElfImage {
    /// Lowest `vaddr` across all loadable segments (the image's base).
    pub fn base_vaddr(&self) -> u64 {
        self.segments.iter().map(|s| s.vaddr).min().unwrap_or(0)
    }

    /// Total span in bytes from the lowest vaddr to the end of the highest
    /// segment — the size of buffer a loader must reserve.
    ///
    /// Returns 0 on arithmetic overflow (segments with absurd sizes rejected
    /// at parse time, so this should not occur in practice).
    pub fn image_span(&self) -> u64 {
        let base = self.base_vaddr();
        self.segments
            .iter()
            .filter_map(|s| {
                // s.vaddr >= base by construction (base is the minimum vaddr).
                let end = s.vaddr.checked_add(s.mem_size)?;
                end.checked_sub(base)
            })
            .max()
            .unwrap_or(0)
    }
}

fn read_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn read_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

/// Parse a 64-bit little-endian x86-64 ELF. Validates the identification bytes
/// and machine type, then reads the program-header table.
pub fn parse(bytes: &[u8]) -> Result<ElfImage, ElfError> {
    if bytes.len() < 64 {
        return Err(ElfError::TooShort);
    }
    if bytes[0..4] != ELF_MAGIC {
        return Err(ElfError::BadMagic);
    }
    if bytes[4] != ELFCLASS64 {
        return Err(ElfError::Not64Bit);
    }
    if bytes[5] != ELFDATA2LSB {
        return Err(ElfError::NotLittleEndian);
    }
    if read_u16(bytes, 18) != EM_X86_64 {
        return Err(ElfError::WrongMachine);
    }

    let entry = read_u64(bytes, 24);

    // e_phoff: read as u64 then range-check before narrowing to usize so we
    // catch values that are valid u64 but would wrap on 32-bit targets.
    let phoff_raw = read_u64(bytes, 32);
    let phoff = usize::try_from(phoff_raw).map_err(|_| ElfError::BadProgramHeaders)?;

    let phentsize = read_u16(bytes, 54) as usize;
    let phnum = read_u16(bytes, 56) as usize;

    // e_phentsize must be at least 56 bytes (the size of an Elf64_Phdr) and
    // e_phoff must be non-zero and within the buffer before we touch it.
    if phentsize < 56 || phoff == 0 {
        return Err(ElfError::BadProgramHeaders);
    }

    // Cap e_phnum to prevent trivially huge loops from untrusted input.
    if phnum > MAX_PHNUM {
        return Err(ElfError::BadProgramHeaders);
    }

    // Verify the entire program-header table fits inside the buffer before
    // entering the loop.  `phentsize * phnum` is the table byte-size;
    // `phoff + table_size` is the exclusive end offset — both must not wrap.
    let ph_table_size = phentsize
        .checked_mul(phnum)
        .ok_or(ElfError::BadProgramHeaders)?;
    let ph_table_end = phoff
        .checked_add(ph_table_size)
        .ok_or(ElfError::BadProgramHeaders)?;
    if ph_table_end > bytes.len() {
        return Err(ElfError::BadProgramHeaders);
    }

    let mut segments = Vec::new();
    for i in 0..phnum {
        // Both multiplications and the addition are already proven safe above
        // (i < phnum, and the whole table fits), but use checked arithmetic
        // to be explicit and future-proof.
        let ph_base = phoff
            .checked_add(i.checked_mul(phentsize).ok_or(ElfError::BadProgramHeaders)?)
            .ok_or(ElfError::BadProgramHeaders)?;

        // Each program header entry must contain at least 56 bytes.
        // We already checked the full table fits, but re-verify the per-entry
        // window so `read_u*` calls below never panic.
        if ph_base.checked_add(56).ok_or(ElfError::BadProgramHeaders)? > bytes.len() {
            return Err(ElfError::BadProgramHeaders);
        }

        let p_type = read_u32(bytes, ph_base);
        if p_type != PT_LOAD {
            continue;
        }
        let flags     = read_u32(bytes, ph_base + 4);
        let offset    = read_u64(bytes, ph_base + 8);
        let vaddr     = read_u64(bytes, ph_base + 16);
        let file_size = read_u64(bytes, ph_base + 32);
        let mem_size  = read_u64(bytes, ph_base + 40);

        // p_offset + p_filesz must not overflow u64, and the resulting range
        // must lie entirely within the ELF buffer.  Cast via usize::try_from
        // so that absurdly large u64 values are caught on 32-bit targets too.
        let offset_usize =
            usize::try_from(offset).map_err(|_| ElfError::BadProgramHeaders)?;
        let file_size_usize =
            usize::try_from(file_size).map_err(|_| ElfError::BadProgramHeaders)?;
        let seg_end = offset_usize
            .checked_add(file_size_usize)
            .ok_or(ElfError::BadProgramHeaders)?;
        if seg_end > bytes.len() {
            return Err(ElfError::BadProgramHeaders);
        }

        // p_vaddr + p_memsz must not overflow u64 (the loader uses this
        // to determine the virtual address range it must map).
        vaddr
            .checked_add(mem_size)
            .ok_or(ElfError::BadProgramHeaders)?;

        segments.push(Segment { vaddr, offset, file_size, mem_size, flags });
    }

    if segments.is_empty() {
        return Err(ElfError::BadProgramHeaders);
    }
    Ok(ElfImage { entry, segments })
}

/// Build a minimal but valid ELF64 with one `PT_LOAD` (R+X) segment containing
/// `code`, loaded at `vaddr`, with the entry point at the segment's start.
///
/// A real toolchain emits these; we synthesise one so the loader can be exercised
/// end-to-end (parse → load → execute) without shipping an external binary.
pub fn build_exec_elf(vaddr: u64, code: &[u8]) -> Vec<u8> {
    let ehsize = 64usize;
    let phentsize = 56usize;
    let code_off = ehsize + phentsize;
    let mut b = alloc::vec![0u8; code_off + code.len()];

    // ELF header.
    b[0..4].copy_from_slice(&ELF_MAGIC);
    b[4] = ELFCLASS64;
    b[5] = ELFDATA2LSB;
    b[6] = 1; // version
    b[16..18].copy_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    b[18..20].copy_from_slice(&EM_X86_64.to_le_bytes());
    b[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
    b[24..32].copy_from_slice(&vaddr.to_le_bytes()); // e_entry = vaddr (code at seg start)
    b[32..40].copy_from_slice(&(ehsize as u64).to_le_bytes()); // e_phoff
    b[52..54].copy_from_slice(&(ehsize as u16).to_le_bytes()); // e_ehsize
    b[54..56].copy_from_slice(&(phentsize as u16).to_le_bytes());
    b[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

    // One program header (PT_LOAD, R+X).
    let ph = ehsize;
    b[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
    b[ph + 4..ph + 8].copy_from_slice(&(PF_R | PF_X).to_le_bytes());
    b[ph + 8..ph + 16].copy_from_slice(&(code_off as u64).to_le_bytes()); // p_offset
    b[ph + 16..ph + 24].copy_from_slice(&vaddr.to_le_bytes()); // p_vaddr
    b[ph + 24..ph + 32].copy_from_slice(&vaddr.to_le_bytes()); // p_paddr
    b[ph + 32..ph + 40].copy_from_slice(&(code.len() as u64).to_le_bytes()); // p_filesz
    b[ph + 40..ph + 48].copy_from_slice(&(code.len() as u64).to_le_bytes()); // p_memsz
    b[ph + 48..ph + 56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

    b[code_off..].copy_from_slice(code);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_elf(vaddr: u64, code: &[u8]) -> Vec<u8> {
        build_exec_elf(vaddr, code)
    }

    #[test]
    fn parses_a_minimal_elf() {
        // x86-64: mov eax, 42 ; ret
        let code = [0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3];
        let elf = make_elf(0x40_0000, &code);
        let img = parse(&elf).unwrap();
        assert_eq!(img.entry, 0x40_0000);
        assert_eq!(img.segments.len(), 1);
        assert!(img.segments[0].is_executable());
        assert_eq!(img.segments[0].file_size, code.len() as u64);
        assert_eq!(img.base_vaddr(), 0x40_0000);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut elf = make_elf(0x1000, &[0xC3]);
        elf[1] = b'X';
        assert_eq!(parse(&elf).unwrap_err(), ElfError::BadMagic);
    }

    #[test]
    fn rejects_32_bit() {
        let mut elf = make_elf(0x1000, &[0xC3]);
        elf[4] = 1; // ELFCLASS32
        assert_eq!(parse(&elf).unwrap_err(), ElfError::Not64Bit);
    }

    #[test]
    fn rejects_wrong_machine() {
        let mut elf = make_elf(0x1000, &[0xC3]);
        elf[18..20].copy_from_slice(&0xB7u16.to_le_bytes()); // AArch64
        assert_eq!(parse(&elf).unwrap_err(), ElfError::WrongMachine);
    }

    #[test]
    fn rejects_truncated() {
        assert_eq!(parse(&[0u8; 10]).unwrap_err(), ElfError::TooShort);
    }

    #[test]
    fn image_span_covers_segment() {
        let code = [0x90u8, 0x90, 0x90, 0xC3];
        let img = parse(&make_elf(0x8000, &code)).unwrap();
        assert_eq!(img.image_span(), code.len() as u64);
    }
}

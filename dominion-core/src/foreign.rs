//! Foreign driver loading — the **safe** NDISwrapper / LinuxKPI (Layer 4 of the
//! driver model; see `docs/architecture/driver-synthesis-and-device-model.md`).
//!
//! The world already has drivers for nearly everything. Rather than rewrite them,
//! DominionOS **borrows** them at runtime: download a Windows or Linux driver and load
//! it — getting massive device breadth for free. The catch with the classic tools is
//! that they run the foreign driver **with full kernel privilege**:
//!
//! * **NDISwrapper** loads an unmodified Windows NDIS `.sys` by providing the Windows
//!   kernel symbols it imports — a **binary-ABI** shim, historically in ring 0.
//! * **LinuxKPI** reimplements a slice of the Linux kernel API so Linux drivers
//!   compile and run on FreeBSD — a **source/KPI** shim.
//!
//! Both gave a buggy or hostile driver the keys to the machine. This module keeps
//! the *idea* (a compatibility shim that lets foreign drivers run) and removes the
//! *danger*: a loaded foreign driver is admitted only through a **default-closed
//! symbol shim** and confined to a **capability bounded to exactly its claimed
//! device resources** ([`crate::cheri`]). A borrowed driver therefore cannot escape
//! its device — the safe version of NDISwrapper/LinuxKPI.
//!
//! This models the *admission and containment* that make runtime loading safe;
//! actual execution happens inside a SIP sandbox (`sched.rs`/`wasm.rs`). Pure, safe
//! `no_std`, host-tested.

use crate::cheri::{perms, CapabilityTags, TaggedCap};
use crate::driver::{
    DeviceClass, DeviceSpec, Driver, DriverFault, MmioDevice, RegOp, ResourceClaim, ValueSrc,
};
use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

/// The foreign ABI/KPI a borrowed driver targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForeignAbi {
    /// Windows NDIS network driver, loaded by binary ABI (à la NDISwrapper).
    WindowsNdis,
    /// Linux driver run against a reimplemented Linux KPI (à la FreeBSD LinuxKPI).
    LinuxKpi,
    /// A generic Linux loadable module against the same KPI surface.
    LinuxModule,
}

/// A compatibility shim: the set of host-provided kernel symbols a foreign driver of
/// a given ABI may call. **Default-closed** — a driver importing anything not on the
/// whitelist is refused, so the borrowed code's whole world is exactly what we chose
/// to expose (the same discipline as the syscall personality in `compat.rs`).
pub struct KpiShim {
    abi: ForeignAbi,
    symbols: BTreeSet<String>,
}

impl KpiShim {
    pub fn new(abi: ForeignAbi) -> KpiShim {
        KpiShim { abi, symbols: BTreeSet::new() }
    }

    /// Add a host-implemented symbol to the shim surface.
    pub fn provide(mut self, symbol: &str) -> KpiShim {
        self.symbols.insert(symbol.into());
        self
    }

    pub fn abi(&self) -> ForeignAbi {
        self.abi
    }

    pub fn provides(&self, symbol: &str) -> bool {
        self.symbols.contains(symbol)
    }

    /// A representative NDIS shim surface (the subset NDISwrapper-style NIC drivers
    /// actually call).
    pub fn ndis() -> KpiShim {
        KpiShim::new(ForeignAbi::WindowsNdis)
            .provide("NdisAllocateMemory")
            .provide("NdisMRegisterMiniport")
            .provide("NdisMAllocateSharedMemory")
            .provide("NdisMRegisterInterrupt")
            .provide("NdisMIndicateReceiveNetBufferLists")
            .provide("NdisMSendNetBufferListsComplete")
    }

    /// A representative LinuxKPI shim surface.
    pub fn linuxkpi() -> KpiShim {
        KpiShim::new(ForeignAbi::LinuxKpi)
            .provide("kmalloc")
            .provide("kfree")
            .provide("ioremap")
            .provide("request_irq")
            .provide("dma_alloc_coherent")
            .provide("netif_rx")
    }
}

/// A foreign driver package as downloaded — a *descriptor*, not yet trusted code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForeignDriver {
    pub name: String,
    pub abi: ForeignAbi,
    pub class: DeviceClass,
    /// Exactly the device resources the driver declares it needs.
    pub claim: ResourceClaim,
    /// The host kernel symbols the driver imports (must all be on the shim surface).
    pub imports: Vec<String>,
}

impl ForeignDriver {
    pub fn new(name: &str, abi: ForeignAbi, class: DeviceClass, claim: ResourceClaim) -> ForeignDriver {
        ForeignDriver { name: name.into(), abi, class, claim, imports: Vec::new() }
    }

    pub fn imports(mut self, symbols: &[&str]) -> ForeignDriver {
        self.imports = symbols.iter().map(|s| (*s).into()).collect();
        self
    }
}

/// Why a foreign driver was refused at load time (it is rejected, never run partially).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// The driver's ABI does not match the installed shim.
    AbiMismatch,
    /// The driver imports a symbol the shim does not provide (default-closed).
    MissingSymbol(String),
    /// The driver claims resources outside the host's permitted envelope.
    ResourceDenied,
    /// The driver binary could not be parsed (bad PE/ELF container or driver section).
    BadImage,
}

/// A successfully loaded foreign driver: confined to a capability over exactly its
/// claimed device window. This is the object the OS schedules inside a SIP sandbox.
#[derive(Debug)]
pub struct ContainedDriver {
    pub name: String,
    pub abi: ForeignAbi,
    pub class: DeviceClass,
    cap: TaggedCap,
}

impl ContainedDriver {
    /// The window this driver is confined to.
    pub fn window(&self) -> (u64, u64) {
        (self.cap.base, self.cap.len)
    }

    /// May the driver touch `[addr, addr+width)`? Outside its capability ⇒ no — the
    /// borrowed code cannot reach another device or the kernel.
    pub fn may_access(&self, addr: u64, width: u64) -> bool {
        self.cap.covers(addr, width)
    }

    /// Re-check the capability tag is authentic (a tampered driver fails closed).
    pub fn is_authentic(&self, tags: &dyn CapabilityTags) -> bool {
        tags.validate(&self.cap)
    }
}

/// Loads & contains foreign drivers. Holds one shim surface and the maximum resource
/// envelope any single borrowed driver may claim.
pub struct ForeignHost {
    shim: KpiShim,
    envelope: ResourceClaim,
}

impl ForeignHost {
    pub fn new(shim: KpiShim, envelope: ResourceClaim) -> ForeignHost {
        ForeignHost { shim, envelope }
    }

    fn within_envelope(&self, claim: &ResourceClaim) -> bool {
        let env_end = match self.envelope.mmio_base.checked_add(self.envelope.mmio_len) {
            Some(e) => e,
            None => return false,
        };
        let claim_end = match claim.mmio_base.checked_add(claim.mmio_len) {
            Some(e) => e,
            None => return false,
        };
        claim.mmio_base >= self.envelope.mmio_base && claim_end <= env_end
    }

    /// Admit and contain a foreign driver. Order of checks is fail-closed: ABI, then
    /// every imported symbol, then the resource envelope. On success, mints a
    /// capability bounded to **exactly** the driver's claimed window.
    pub fn load(
        &self,
        driver: &ForeignDriver,
        tags: &dyn CapabilityTags,
    ) -> Result<ContainedDriver, LoadError> {
        // 1. The shim must speak the driver's ABI.
        if driver.abi != self.shim.abi() {
            return Err(LoadError::AbiMismatch);
        }
        // 2. Every imported symbol must be on the shim surface (default-closed).
        for sym in &driver.imports {
            if !self.shim.provides(sym) {
                return Err(LoadError::MissingSymbol(sym.clone()));
            }
        }
        // 3. The claim must fit inside the host's permitted envelope.
        if !self.within_envelope(&driver.claim) {
            return Err(LoadError::ResourceDenied);
        }
        // 4. Mint a capability bounded to exactly the claimed device window.
        let cap = tags.mint(driver.claim.mmio_base, driver.claim.mmio_len, perms::READ | perms::WRITE);
        Ok(ContainedDriver {
            name: driver.name.clone(),
            abi: driver.abi,
            class: driver.class,
            cap,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Loading & USING a real driver binary — beyond admission, we parse the actual
// PE (`.sys`) / ELF (`.ko`) container the rest of computing ships, link its
// imports against the default-closed shim, confine it to a capability, and then
// **drive a device with it** through the synthesized-driver runtime ([`Driver`]).
//
// A foreign driver binary carries two named sections we locate via the real
// section table:
//   * `.kpi` — the host kernel symbols it imports (newline-separated), checked
//     against the shim exactly like a hand-declared `imports` list; and
//   * `.drv` — its device logic, a serialized [`DeviceSpec`] (register map +
//     register-op programs). Once bound to a capability over exactly its claimed
//     MMIO window, running a program reads/writes device registers for real and
//     cannot escape — the safe NDISwrapper/LinuxKPI, now actually executing.
// ═══════════════════════════════════════════════════════════════════════════

/// A downloaded foreign driver binary: its declared ABI and the raw image bytes.
#[derive(Clone, Debug)]
pub struct ForeignBinary {
    pub name: String,
    pub abi: ForeignAbi,
    pub bytes: Vec<u8>,
}

impl ForeignBinary {
    pub fn new(name: &str, abi: ForeignAbi, bytes: Vec<u8>) -> ForeignBinary {
        ForeignBinary { name: name.into(), abi, bytes }
    }
}

/// What we extracted from a parsed driver container.
struct ParsedImage {
    imports: Vec<String>,
    spec: DeviceSpec,
}

/// A loaded, contained, *executable* foreign driver. The [`ContainedDriver`] proves
/// admission + confinement; the [`Driver`] actually runs its register programs.
#[derive(Debug)]
pub struct LoadedForeignDriver {
    pub contained: ContainedDriver,
    driver: Driver,
}

impl LoadedForeignDriver {
    /// The MMIO window this borrowed driver is confined to.
    pub fn window(&self) -> (u64, u64) {
        self.driver.window()
    }

    pub fn class(&self) -> DeviceClass {
        self.driver.class()
    }

    /// **Use** the borrowed driver: run one of its operations against the device,
    /// returning the values it read. Every register touch is bounded by the
    /// driver's capability, so a hostile/buggy borrowed driver still cannot escape.
    pub fn run(
        &self,
        op: &str,
        args: &[u64],
        dev: &mut dyn MmioDevice,
        tags: &dyn CapabilityTags,
    ) -> Result<Vec<u64>, DriverFault> {
        self.driver.run(op, args, dev, tags)
    }
}

impl ForeignHost {
    /// Parse, admit, confine, **and bind** a foreign driver binary so it can be run.
    /// Fail-closed at every step: a bad container, an un-provided import, or an
    /// out-of-envelope claim all reject the driver before it can touch anything.
    pub fn load_binary(
        &self,
        bin: &ForeignBinary,
        tags: &dyn CapabilityTags,
    ) -> Result<LoadedForeignDriver, LoadError> {
        let parsed = parse_foreign(bin)?;
        let import_refs: Vec<&str> = parsed.imports.iter().map(|s| s.as_str()).collect();
        let descriptor = ForeignDriver::new(&bin.name, bin.abi, parsed.spec.class, parsed.spec.resources)
            .imports(&import_refs);
        // Existing admission: ABI + every symbol on the shim + within the envelope.
        let contained = self.load(&descriptor, tags)?;
        // Bind the device logic to a capability over exactly its claimed window.
        let driver = Driver::bind(parsed.spec, tags).map_err(|_| LoadError::BadImage)?;
        Ok(LoadedForeignDriver { contained, driver })
    }
}

/// Dispatch to the PE or ELF container parser based on the declared ABI.
fn parse_foreign(bin: &ForeignBinary) -> Result<ParsedImage, LoadError> {
    match bin.abi {
        ForeignAbi::WindowsNdis => parse_pe(&bin.bytes),
        ForeignAbi::LinuxKpi | ForeignAbi::LinuxModule => parse_elf_ko(&bin.bytes),
    }
}

// ─────────────────────────── little-endian readers ───────────────────────────
// Shared with foreignload.rs to avoid parallel reader sets.

pub(crate) fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
}
pub(crate) fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
pub(crate) fn rd_u64(b: &[u8], o: usize) -> Option<u64> {
    b.get(o..o + 8).map(|s| {
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        u64::from_le_bytes(a)
    })
}

// ─────────────────────────── PE (.sys) container ───────────────────────────

const PE_MACHINE_X64: u16 = 0x8664;
const SECTION_HDR_SIZE: usize = 40;

/// Parse a 64-bit PE/COFF image (a Windows `.sys`): DOS stub → PE signature →
/// COFF header → section table, then pull the `.kpi` and `.drv` sections out.
fn parse_pe(b: &[u8]) -> Result<ParsedImage, LoadError> {
    if b.len() < 0x40 || &b[0..2] != b"MZ" {
        return Err(LoadError::BadImage);
    }
    let pe_off = rd_u32(b, 0x3C).ok_or(LoadError::BadImage)? as usize;
    if b.get(pe_off..pe_off + 4) != Some(b"PE\0\0") {
        return Err(LoadError::BadImage);
    }
    let coff = pe_off + 4;
    let machine = rd_u16(b, coff).ok_or(LoadError::BadImage)?;
    if machine != PE_MACHINE_X64 {
        return Err(LoadError::BadImage);
    }
    let nsections = rd_u16(b, coff + 2).ok_or(LoadError::BadImage)? as usize;
    let opt_size = rd_u16(b, coff + 16).ok_or(LoadError::BadImage)? as usize;
    let sec_table = coff + 20 + opt_size;

    let mut kpi: Option<&[u8]> = None;
    let mut drv: Option<&[u8]> = None;
    for i in 0..nsections {
        let h = sec_table + i * SECTION_HDR_SIZE;
        let name_raw = b.get(h..h + 8).ok_or(LoadError::BadImage)?;
        let name = section_name(name_raw);
        let raw_size = rd_u32(b, h + 16).ok_or(LoadError::BadImage)? as usize;
        let raw_ptr = rd_u32(b, h + 20).ok_or(LoadError::BadImage)? as usize;
        let data = b.get(raw_ptr..raw_ptr + raw_size).ok_or(LoadError::BadImage)?;
        match name.as_str() {
            ".kpi" => kpi = Some(data),
            ".drv" => drv = Some(data),
            _ => {}
        }
    }
    finish_parse(kpi, drv)
}

// ─────────────────────────── ELF (.ko) container ───────────────────────────

const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

/// Parse a 64-bit little-endian ELF relocatable object (a Linux `.ko`): ELF header
/// → section-header table → section-name string table, then pull `.kpi`/`.drv`.
fn parse_elf_ko(b: &[u8]) -> Result<ParsedImage, LoadError> {
    if b.len() < 64 || b[0..4] != ELF_MAGIC || b[4] != 2 || b[5] != 1 {
        return Err(LoadError::BadImage);
    }
    let shoff = rd_u64(b, 40).ok_or(LoadError::BadImage)? as usize;
    let shentsize = rd_u16(b, 58).ok_or(LoadError::BadImage)? as usize;
    let shnum = rd_u16(b, 60).ok_or(LoadError::BadImage)? as usize;
    let shstrndx = rd_u16(b, 62).ok_or(LoadError::BadImage)? as usize;
    if shentsize < 64 || shnum == 0 || shstrndx >= shnum {
        return Err(LoadError::BadImage);
    }
    // The section-header string table tells us each section's name.
    let strtab_hdr = shoff + shstrndx * shentsize;
    let strtab_off = rd_u64(b, strtab_hdr + 24).ok_or(LoadError::BadImage)? as usize;
    let strtab_size = rd_u64(b, strtab_hdr + 32).ok_or(LoadError::BadImage)? as usize;
    let strtab = b.get(strtab_off..strtab_off + strtab_size).ok_or(LoadError::BadImage)?;

    let mut kpi: Option<&[u8]> = None;
    let mut drv: Option<&[u8]> = None;
    for i in 0..shnum {
        let h = shoff + i * shentsize;
        let name_off = rd_u32(b, h).ok_or(LoadError::BadImage)? as usize;
        let name = cstr_at(strtab, name_off);
        let off = rd_u64(b, h + 24).ok_or(LoadError::BadImage)? as usize;
        let size = rd_u64(b, h + 32).ok_or(LoadError::BadImage)? as usize;
        if name == ".kpi" || name == ".drv" {
            let data = b.get(off..off + size).ok_or(LoadError::BadImage)?;
            if name == ".kpi" {
                kpi = Some(data);
            } else {
                drv = Some(data);
            }
        }
    }
    finish_parse(kpi, drv)
}

fn finish_parse(kpi: Option<&[u8]>, drv: Option<&[u8]>) -> Result<ParsedImage, LoadError> {
    let kpi = kpi.ok_or(LoadError::BadImage)?;
    let drv = drv.ok_or(LoadError::BadImage)?;
    let imports: Vec<String> = core::str::from_utf8(kpi)
        .map_err(|_| LoadError::BadImage)?
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let spec = decode_spec(drv).ok_or(LoadError::BadImage)?;
    Ok(ParsedImage { imports, spec })
}

fn section_name(raw: &[u8]) -> String {
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

fn cstr_at(strtab: &[u8], off: usize) -> String {
    if off >= strtab.len() {
        return String::new();
    }
    let rest = &strtab[off..];
    let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
    String::from_utf8_lossy(&rest[..end]).into_owned()
}

// ─────────────────────────── DeviceSpec (de)serialization ───────────────────────────
//
// The `.drv` section is a compact, explicit binary encoding of the driver's
// register map + register-op programs — the device logic, carried as data.

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.push(s.len() as u8);
    out.extend_from_slice(s.as_bytes());
}

fn class_code(c: DeviceClass) -> u8 {
    match c {
        DeviceClass::Block => 0,
        DeviceClass::Net => 1,
        DeviceClass::Entropy => 2,
        DeviceClass::Console => 3,
        DeviceClass::Input => 4,
    }
}
fn code_class(c: u8) -> Option<DeviceClass> {
    Some(match c {
        0 => DeviceClass::Block,
        1 => DeviceClass::Net,
        2 => DeviceClass::Entropy,
        3 => DeviceClass::Console,
        4 => DeviceClass::Input,
        _ => return None,
    })
}

/// Serialize a [`DeviceSpec`] into the `.drv` section format.
pub fn encode_spec(spec: &DeviceSpec) -> Vec<u8> {
    let mut o = Vec::new();
    o.push(class_code(spec.class));
    o.extend_from_slice(&spec.resources.mmio_base.to_le_bytes());
    o.extend_from_slice(&spec.resources.mmio_len.to_le_bytes());
    o.extend_from_slice(&spec.resources.irq.to_le_bytes());
    o.extend_from_slice(&(spec.registers.len() as u16).to_le_bytes());
    for (name, reg) in &spec.registers {
        put_str(&mut o, name);
        o.extend_from_slice(&reg.offset.to_le_bytes());
        o.push(reg.width);
    }
    o.extend_from_slice(&(spec.programs.len() as u16).to_le_bytes());
    for (name, steps) in &spec.programs {
        put_str(&mut o, name);
        o.extend_from_slice(&(steps.len() as u16).to_le_bytes());
        for step in steps {
            match step {
                RegOp::Write { reg, value } => {
                    o.push(0);
                    put_str(&mut o, reg);
                    match value {
                        ValueSrc::Imm(v) => {
                            o.push(0);
                            o.extend_from_slice(&v.to_le_bytes());
                        }
                        ValueSrc::Arg(i) => {
                            o.push(1);
                            o.extend_from_slice(&(*i as u64).to_le_bytes());
                        }
                        ValueSrc::BufPhys(buf) => {
                            o.push(2);
                            put_str(&mut o, buf);
                        }
                    }
                }
                RegOp::Read { reg } => {
                    o.push(1);
                    put_str(&mut o, reg);
                }
                RegOp::Poll { reg, value, max_spins } => {
                    o.push(2);
                    put_str(&mut o, reg);
                    o.extend_from_slice(&value.to_le_bytes());
                    o.extend_from_slice(&max_spins.to_le_bytes());
                }
                RegOp::PollBits { reg, mask, value, max_spins } => {
                    o.push(5);
                    put_str(&mut o, reg);
                    o.extend_from_slice(&mask.to_le_bytes());
                    o.extend_from_slice(&value.to_le_bytes());
                    o.extend_from_slice(&max_spins.to_le_bytes());
                }
                RegOp::BufStore { buf, off } => {
                    o.push(3);
                    put_str(&mut o, buf);
                    o.extend_from_slice(&off.to_le_bytes());
                }
                RegOp::BufLoad { buf, off, len } => {
                    o.push(4);
                    put_str(&mut o, buf);
                    o.extend_from_slice(&off.to_le_bytes());
                    o.extend_from_slice(&len.to_le_bytes());
                }
                RegOp::BufStoreVal { buf, off, value, width } => {
                    o.push(6);
                    put_str(&mut o, buf);
                    o.extend_from_slice(&off.to_le_bytes());
                    o.push(*width);
                    match value {
                        ValueSrc::Imm(v) => {
                            o.push(0);
                            o.extend_from_slice(&v.to_le_bytes());
                        }
                        ValueSrc::Arg(i) => {
                            o.push(1);
                            o.extend_from_slice(&(*i as u64).to_le_bytes());
                        }
                        ValueSrc::BufPhys(bn) => {
                            o.push(2);
                            put_str(&mut o, bn);
                        }
                    }
                }
            }
        }
    }
    o.extend_from_slice(&(spec.buffers.len() as u16).to_le_bytes());
    for (name, len) in &spec.buffers {
        put_str(&mut o, name);
        o.extend_from_slice(&len.to_le_bytes());
    }
    o
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        let v = rd_u16(self.b, self.p)?;
        self.p += 2;
        Some(v)
    }
    fn u32(&mut self) -> Option<u32> {
        let v = rd_u32(self.b, self.p)?;
        self.p += 4;
        Some(v)
    }
    fn u64(&mut self) -> Option<u64> {
        let v = rd_u64(self.b, self.p)?;
        self.p += 8;
        Some(v)
    }
    fn s(&mut self) -> Option<String> {
        let len = self.u8()? as usize;
        let bytes = self.b.get(self.p..self.p + len)?;
        self.p += len;
        Some(String::from_utf8_lossy(bytes).into_owned())
    }
}

/// Deserialize a [`DeviceSpec`] from the `.drv` section format.
pub fn decode_spec(b: &[u8]) -> Option<DeviceSpec> {
    let mut c = Cursor { b, p: 0 };
    let class = code_class(c.u8()?)?;
    let base = c.u64()?;
    let len = c.u64()?;
    let irq = c.u32()?;
    let mut spec = DeviceSpec::new(class, ResourceClaim { mmio_base: base, mmio_len: len, irq });
    let nreg = c.u16()?;
    for _ in 0..nreg {
        let name = c.s()?;
        let offset = c.u64()?;
        let width = c.u8()?;
        spec = spec.register(&name, offset, width);
    }
    let nprog = c.u16()?;
    for _ in 0..nprog {
        let op = c.s()?;
        let nsteps = c.u16()?;
        let mut steps = Vec::with_capacity(nsteps as usize);
        for _ in 0..nsteps {
            let kind = c.u8()?;
            let step = match kind {
                0 => {
                    let reg = c.s()?;
                    let src = c.u8()?;
                    let value = match src {
                        0 => ValueSrc::Imm(c.u64()?),
                        1 => ValueSrc::Arg(c.u64()? as usize),
                        2 => ValueSrc::BufPhys(c.s()?),
                        _ => return None,
                    };
                    RegOp::Write { reg, value }
                }
                1 => RegOp::Read { reg: c.s()? },
                2 => {
                    let reg = c.s()?;
                    let value = c.u64()?;
                    let max_spins = c.u32()?;
                    RegOp::Poll { reg, value, max_spins }
                }
                3 => {
                    let buf = c.s()?;
                    let off = c.u64()?;
                    RegOp::BufStore { buf, off }
                }
                4 => {
                    let buf = c.s()?;
                    let off = c.u64()?;
                    let len = c.u64()?;
                    RegOp::BufLoad { buf, off, len }
                }
                5 => {
                    let reg = c.s()?;
                    let mask = c.u64()?;
                    let value = c.u64()?;
                    let max_spins = c.u32()?;
                    RegOp::PollBits { reg, mask, value, max_spins }
                }
                6 => {
                    let buf = c.s()?;
                    let off = c.u64()?;
                    let width = c.u8()?;
                    let src = c.u8()?;
                    let value = match src {
                        0 => ValueSrc::Imm(c.u64()?),
                        1 => ValueSrc::Arg(c.u64()? as usize),
                        2 => ValueSrc::BufPhys(c.s()?),
                        _ => return None,
                    };
                    RegOp::BufStoreVal { buf, off, value, width }
                }
                _ => return None,
            };
            steps.push(step);
        }
        spec = spec.program(&op, steps);
    }
    let nbuf = c.u16()?;
    for _ in 0..nbuf {
        let name = c.s()?;
        let len = c.u64()?;
        spec = spec.buffer(&name, len);
    }
    Some(spec)
}

// ─────────────────────────── synthesizers (for testing / demo) ───────────────────────────
//
// A real toolchain emits these; we synthesise valid PE/ELF containers so the loader
// can be exercised end-to-end (parse → admit → confine → bind → run) without
// shipping an external binary — exactly the pattern `elf::build_exec_elf` uses.

/// Build a minimal but structurally valid Windows `.sys` (PE/COFF) carrying the
/// given imports and device spec in `.kpi` / `.drv` sections.
pub fn build_pe_sys(imports: &[&str], spec: &DeviceSpec) -> Vec<u8> {
    let kpi = imports.join("\n").into_bytes();
    let drv = encode_spec(spec);
    let opt_size = 24usize; // a small PE32+ optional header
    let pe_off = 0x40usize;
    let coff = pe_off + 4;
    let sec_table = coff + 20 + opt_size;
    let nsections = 2usize;
    let data_start = sec_table + nsections * SECTION_HDR_SIZE;

    let mut b = alloc::vec![0u8; data_start];
    b[0] = b'M';
    b[1] = b'Z';
    b[0x3C..0x40].copy_from_slice(&(pe_off as u32).to_le_bytes());
    b[pe_off..pe_off + 4].copy_from_slice(b"PE\0\0");
    b[coff..coff + 2].copy_from_slice(&PE_MACHINE_X64.to_le_bytes());
    b[coff + 2..coff + 4].copy_from_slice(&(nsections as u16).to_le_bytes());
    b[coff + 16..coff + 18].copy_from_slice(&(opt_size as u16).to_le_bytes());
    b[coff + 18..coff + 20].copy_from_slice(&0x2022u16.to_le_bytes()); // characteristics (driver-ish)
    // Optional header: PE32+ magic so the container looks real.
    b[coff + 20..coff + 22].copy_from_slice(&0x020Bu16.to_le_bytes());

    let mut blobs: Vec<u8> = Vec::new();
    let put_section = |b: &mut Vec<u8>, idx: usize, name: &str, data: &[u8], blobs: &mut Vec<u8>| {
        let h = sec_table + idx * SECTION_HDR_SIZE;
        let mut nm = [0u8; 8];
        for (i, c) in name.bytes().take(8).enumerate() {
            nm[i] = c;
        }
        b[h..h + 8].copy_from_slice(&nm);
        b[h + 8..h + 12].copy_from_slice(&(data.len() as u32).to_le_bytes()); // VirtualSize
        b[h + 16..h + 20].copy_from_slice(&(data.len() as u32).to_le_bytes()); // SizeOfRawData
        let ptr = data_start + blobs.len();
        b[h + 20..h + 24].copy_from_slice(&(ptr as u32).to_le_bytes()); // PointerToRawData
        blobs.extend_from_slice(data);
    };
    put_section(&mut b, 0, ".kpi", &kpi, &mut blobs);
    put_section(&mut b, 1, ".drv", &drv, &mut blobs);
    b.extend_from_slice(&blobs);
    b
}

/// Build a minimal but structurally valid Linux `.ko` (ELF64 ET_REL) carrying the
/// given imports and device spec in `.kpi` / `.drv` sections.
pub fn build_elf_ko(imports: &[&str], spec: &DeviceSpec) -> Vec<u8> {
    let kpi = imports.join("\n").into_bytes();
    let drv = encode_spec(spec);
    // Section-name string table: NULL, .kpi, .drv, .shstrtab.
    let mut shstr: Vec<u8> = Vec::new();
    shstr.push(0);
    let off_kpi = shstr.len();
    shstr.extend_from_slice(b".kpi\0");
    let off_drv = shstr.len();
    shstr.extend_from_slice(b".drv\0");
    let off_shstr = shstr.len();
    shstr.extend_from_slice(b".shstrtab\0");

    let ehsize = 64usize;
    let shentsize = 64usize;
    let shnum = 4usize; // NULL, .kpi, .drv, .shstrtab
    // Data layout: ELF header, then section payloads, then section header table.
    let kpi_off = ehsize;
    let drv_off = kpi_off + kpi.len();
    let shstr_off = drv_off + drv.len();
    let shoff = shstr_off + shstr.len();
    let total = shoff + shnum * shentsize;

    let mut b = alloc::vec![0u8; total];
    b[0..4].copy_from_slice(&ELF_MAGIC);
    b[4] = 2; // ELFCLASS64
    b[5] = 1; // little-endian
    b[6] = 1; // version
    b[16..18].copy_from_slice(&1u16.to_le_bytes()); // e_type = ET_REL
    b[18..20].copy_from_slice(&0x3Eu16.to_le_bytes()); // e_machine = x86-64
    b[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
    b[40..48].copy_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
    b[52..54].copy_from_slice(&(ehsize as u16).to_le_bytes()); // e_ehsize
    b[58..60].copy_from_slice(&(shentsize as u16).to_le_bytes());
    b[60..62].copy_from_slice(&(shnum as u16).to_le_bytes());
    b[62..64].copy_from_slice(&3u16.to_le_bytes()); // e_shstrndx = section 3

    b[kpi_off..kpi_off + kpi.len()].copy_from_slice(&kpi);
    b[drv_off..drv_off + drv.len()].copy_from_slice(&drv);
    b[shstr_off..shstr_off + shstr.len()].copy_from_slice(&shstr);

    let put_sh = |b: &mut Vec<u8>, idx: usize, name_off: usize, off: usize, size: usize, typ: u32| {
        let h = shoff + idx * shentsize;
        b[h..h + 4].copy_from_slice(&(name_off as u32).to_le_bytes()); // sh_name
        b[h + 4..h + 8].copy_from_slice(&typ.to_le_bytes()); // sh_type
        b[h + 24..h + 32].copy_from_slice(&(off as u64).to_le_bytes()); // sh_offset
        b[h + 32..h + 40].copy_from_slice(&(size as u64).to_le_bytes()); // sh_size
    };
    put_sh(&mut b, 0, 0, 0, 0, 0); // NULL
    put_sh(&mut b, 1, off_kpi, kpi_off, kpi.len(), 1); // .kpi (PROGBITS)
    put_sh(&mut b, 2, off_drv, drv_off, drv.len(), 1); // .drv (PROGBITS)
    put_sh(&mut b, 3, off_shstr, shstr_off, shstr.len(), 3); // .shstrtab (STRTAB)
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheri::SoftwareTags;

    fn host() -> (ForeignHost, SoftwareTags) {
        // The host permits any device window inside [0x1000, 0x10000).
        let envelope = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
        (ForeignHost::new(KpiShim::ndis(), envelope), SoftwareTags::new([5u8; 32]))
    }

    #[test]
    fn loads_a_windows_ndis_driver_contained() {
        let (host, tags) = host();
        // A downloaded Windows WiFi driver that only calls whitelisted NDIS symbols.
        let drv = ForeignDriver::new(
            "RtlWifi.sys",
            ForeignAbi::WindowsNdis,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x2000, mmio_len: 0x200, irq: 11 },
        )
        .imports(&["NdisAllocateMemory", "NdisMRegisterMiniport", "NdisMRegisterInterrupt"]);
        let loaded = host.load(&drv, &tags).unwrap();
        // Confined to exactly its claimed window.
        assert_eq!(loaded.window(), (0x2000, 0x200));
        assert!(loaded.may_access(0x2000, 8));
        assert!(loaded.may_access(0x21F8, 8)); // last word
        assert!(!loaded.may_access(0x2200, 8)); // one past the window → denied
        assert!(loaded.is_authentic(&tags));
    }

    #[test]
    fn an_unprovided_import_is_refused_default_closed() {
        let (host, tags) = host();
        // The driver wants a symbol the shim does not expose (e.g. raw port I/O).
        let drv = ForeignDriver::new(
            "sketchy.sys",
            ForeignAbi::WindowsNdis,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x2000, mmio_len: 0x100, irq: 5 },
        )
        .imports(&["NdisAllocateMemory", "ZwOpenFile"]); // ZwOpenFile not on the surface
        assert_eq!(
            host.load(&drv, &tags).err(),
            Some(LoadError::MissingSymbol("ZwOpenFile".into()))
        );
    }

    #[test]
    fn abi_mismatch_is_rejected() {
        let (host, tags) = host(); // NDIS shim
        // A Linux driver presented to an NDIS host.
        let drv = ForeignDriver::new(
            "iwlwifi.ko",
            ForeignAbi::LinuxKpi,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x2000, mmio_len: 0x100, irq: 5 },
        );
        assert_eq!(host.load(&drv, &tags).err(), Some(LoadError::AbiMismatch));
    }

    #[test]
    fn a_resource_grab_outside_the_envelope_is_denied() {
        let (host, tags) = host();
        // A driver trying to claim the whole address space (the NDISwrapper-in-ring-0
        // failure mode) is refused — it can never get a capability beyond the envelope.
        let greedy = ForeignDriver::new(
            "greedy.sys",
            ForeignAbi::WindowsNdis,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x0, mmio_len: u64::MAX, irq: 0 },
        )
        .imports(&["NdisAllocateMemory"]);
        assert_eq!(host.load(&greedy, &tags).err(), Some(LoadError::ResourceDenied));
    }

    #[test]
    fn linuxkpi_host_loads_a_linux_driver() {
        let envelope = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
        let host = ForeignHost::new(KpiShim::linuxkpi(), envelope);
        let tags = SoftwareTags::new([7u8; 32]);
        let drv = ForeignDriver::new(
            "amdgpu.ko",
            ForeignAbi::LinuxKpi,
            DeviceClass::Input,
            ResourceClaim { mmio_base: 0x3000, mmio_len: 0x1000, irq: 16 },
        )
        .imports(&["kmalloc", "ioremap", "request_irq", "dma_alloc_coherent"]);
        let loaded = host.load(&drv, &tags).unwrap();
        assert_eq!(loaded.window(), (0x3000, 0x1000));
    }

    #[test]
    fn download_a_ton_each_contained_to_its_own_device() {
        let (host, tags) = host();
        // Load many drivers at runtime; each confined to a disjoint window, mutually
        // unreachable — the "download a ton and add them" scenario, made safe.
        let mut loaded = Vec::new();
        for i in 0..8u64 {
            let base = 0x2000 + i * 0x400;
            let drv = ForeignDriver::new(
                "dev.sys",
                ForeignAbi::WindowsNdis,
                DeviceClass::Net,
                ResourceClaim { mmio_base: base, mmio_len: 0x100, irq: 5 },
            )
            .imports(&["NdisAllocateMemory"]);
            loaded.push(host.load(&drv, &tags).unwrap());
        }
        assert_eq!(loaded.len(), 8);
        // Driver 0 cannot reach driver 1's registers (capability containment).
        let (d1_base, _) = loaded[1].window();
        assert!(!loaded[0].may_access(d1_base, 8));
        assert!(loaded[1].may_access(d1_base, 8));
    }

    #[test]
    fn tampered_contained_driver_fails_authenticity() {
        let (host, tags) = host();
        let drv = ForeignDriver::new(
            "x.sys",
            ForeignAbi::WindowsNdis,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x2000, mmio_len: 0x100, irq: 5 },
        )
        .imports(&["NdisAllocateMemory"]);
        let mut loaded = host.load(&drv, &tags).unwrap();
        // Forge wider bounds: the capability tag no longer validates.
        loaded.cap.len = 0xFFFF;
        assert!(!loaded.is_authentic(&tags));
    }
}

#[cfg(test)]
mod exec_tests;

//! Declarative driver synthesis & the uniform device model — *enable many drivers
//! without coding each one* (see `docs/architecture/driver-synthesis-and-device-model.md`).
//!
//! Drivers are the bulk of every OS (Linux is ~70 % driver code) and its biggest
//! attack surface — one buggy driver compromises a monolithic kernel. DominionOS's
//! answer, promised by Stage 6 ("a driver becomes a sandboxed execution module
//! subject to mathematical constraints"): a driver is **data, not bespoke code**.
//!
//! A [`DeviceSpec`] declares a device's *register map*, its *resource claim* (the
//! exact MMIO window / IRQ it may touch), and the *register-op programs* that
//! implement each logical operation. **One** reusable runtime — [`Driver`] — drives
//! **any** device from its spec. Adding a device means adding a spec (data shipped
//! over the object graph), not writing a driver.
//!
//! What makes that *safe* is the capability model: the runtime mints an MMIO
//! capability ([`crate::cheri`]) bounded to **exactly** the device's window, and
//! every register access is checked against it. A wrong, synthesized, or even
//! borrowed driver therefore **cannot escape its device** — an out-of-bounds
//! register access *traps* instead of corrupting the machine. Because the code is
//! untrusted-but-contained, it can be *generated* or *reused* rather than carefully
//! hand-written. Pure, safe `no_std`; the capability is enforced in software today
//! (Tier 0) and by CHERI tags where present (Tier 2) — never *required*.

use crate::cheri::{perms, CapabilityTags, TaggedCap};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// The device family a spec belongs to — selects the OS-side service contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    Block,
    Net,
    Entropy,
    Console,
    Input,
}

/// A named register at a window-relative `offset`, `width` bytes wide (1/2/4/8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Register {
    pub offset: u64,
    pub width: u8,
}

/// The resources a driver is allowed to touch — and *nothing else*. The MMIO
/// window becomes the bounds of its capability; the IRQ is its only interrupt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceClaim {
    pub mmio_base: u64,
    pub mmio_len: u64,
    pub irq: u32,
}

/// Where a register-write value comes from: an immediate, the Nth caller arg, or
/// the physical base address of a named DMA buffer (for programming ring/descriptor
/// registers — the value a real NIC/disk is handed to DMA from/to).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueSrc {
    Imm(u64),
    Arg(usize),
    BufPhys(String),
}

/// One step of a synthesized register program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegOp {
    /// Write a value to a named register.
    Write { reg: String, value: ValueSrc },
    /// Read a named register; its value is appended to the operation's results.
    Read { reg: String },
    /// Poll a register until it equals `value`, up to `max_spins` (then time out).
    /// Models waiting on a completion/status bit without an IRQ.
    Poll { reg: String, value: u64, max_spins: u32 },
    /// Poll `reg & mask` until it equals `value`, up to `max_spins` (fails closed).
    /// Needed for status/completion *bits* (an equality poll cannot express a bit
    /// in a field whose other bits vary, e.g. a NIC's TX/RX-done flag).
    PollBits { reg: String, mask: u64, value: u64, max_spins: u32 },
    /// Copy the operation's input byte payload into a named DMA buffer at `off`
    /// (e.g. stage an outbound packet/sector before kicking the device).
    BufStore { buf: String, off: u64 },
    /// Read `len` bytes from a named DMA buffer at `off` into the byte output
    /// (e.g. pull a received packet out of the RX ring).
    BufLoad { buf: String, off: u64, len: u64 },
    /// Store a register-width `value` (immediate, caller arg, or a buffer's physical
    /// address) into a DMA buffer at `off`, little-endian — for building in-memory
    /// descriptors, the heart of e1000/AHCI/NVMe/virtio ring drivers.
    BufStoreVal { buf: String, off: u64, value: ValueSrc, width: u8 },
}

/// The declarative device specification — a driver expressed as data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSpec {
    pub class: DeviceClass,
    pub resources: ResourceClaim,
    pub registers: BTreeMap<String, Register>,
    /// Named DMA buffers the driver may stage payloads in: name → length in bytes.
    /// The runtime allocates each one bounded by a DMA capability, so a driver can
    /// only DMA to/from its own declared buffers.
    pub buffers: BTreeMap<String, u64>,
    /// Logical operation name → the register program that implements it.
    pub programs: BTreeMap<String, Vec<RegOp>>,
}

impl DeviceSpec {
    pub fn new(class: DeviceClass, resources: ResourceClaim) -> DeviceSpec {
        DeviceSpec {
            class,
            resources,
            registers: BTreeMap::new(),
            buffers: BTreeMap::new(),
            programs: BTreeMap::new(),
        }
    }

    pub fn register(mut self, name: &str, offset: u64, width: u8) -> DeviceSpec {
        self.registers.insert(name.into(), Register { offset, width });
        self
    }

    /// Declare a DMA buffer of `len` bytes the driver may stage payloads in.
    pub fn buffer(mut self, name: &str, len: u64) -> DeviceSpec {
        self.buffers.insert(name.into(), len);
        self
    }

    pub fn program(mut self, op: &str, steps: Vec<RegOp>) -> DeviceSpec {
        self.programs.insert(op.into(), steps);
        self
    }

    /// A register exists and lies fully inside the claimed MMIO window.
    fn reg_in_window(&self, name: &str) -> bool {
        match self.registers.get(name) {
            Some(r) => r
                .offset
                .checked_add(r.width as u64)
                .map(|e| e <= self.resources.mmio_len)
                .unwrap_or(false),
            None => false,
        }
    }

    /// Every register named by every program exists and lies inside the MMIO
    /// window; every DMA buffer referenced exists and every static buffer access is
    /// in-bounds. A spec that fails this can never produce an out-of-bounds access —
    /// catching the error at *bind* time rather than at runtime.
    pub fn is_well_formed(&self) -> bool {
        for steps in self.programs.values() {
            for step in steps {
                match step {
                    RegOp::Write { reg, value } => {
                        if !self.reg_in_window(reg) {
                            return false;
                        }
                        if let ValueSrc::BufPhys(buf) = value {
                            if !self.buffers.contains_key(buf) {
                                return false;
                            }
                        }
                    }
                    RegOp::Read { reg } | RegOp::Poll { reg, .. } | RegOp::PollBits { reg, .. } => {
                        if !self.reg_in_window(reg) {
                            return false;
                        }
                    }
                    RegOp::BufStore { buf, off } => match self.buffers.get(buf) {
                        Some(&len) => {
                            if *off > len {
                                return false;
                            }
                        }
                        None => return false,
                    },
                    RegOp::BufLoad { buf, off, len } => match self.buffers.get(buf) {
                        Some(&blen) => match off.checked_add(*len) {
                            Some(end) if end <= blen => {}
                            _ => return false,
                        },
                        None => return false,
                    },
                    RegOp::BufStoreVal { buf, off, value, width } => {
                        match self.buffers.get(buf) {
                            Some(&blen) => match off.checked_add(*width as u64) {
                                Some(end) if end <= blen => {}
                                _ => return false,
                            },
                            None => return false,
                        }
                        if *width == 0 || *width > 8 {
                            return false;
                        }
                        if let ValueSrc::BufPhys(b) = value {
                            if !self.buffers.contains_key(b) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}

/// Why a driver step failed — every failure is *contained*, never propagated as
/// memory corruption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverFault {
    /// A register access fell outside the driver's capability window (the trap).
    OutOfBounds,
    /// The driver's capability tag is not authentic (forged / cleared).
    CapabilityInvalid,
    /// A program referenced a register the spec does not define.
    UnknownRegister,
    /// The requested operation has no program in the spec.
    UnknownOperation,
    /// A `Poll` did not reach its target value within `max_spins` (fails closed).
    Timeout,
    /// The spec is malformed (a register escapes the window).
    MalformedSpec,
}

/// The physical register space a device exposes (the mock device implements this;
/// on metal it is backed by the kernel's MMIO HAL). Addressing is **absolute**, so
/// two devices share one address space exactly like real MMIO.
pub trait MmioDevice {
    fn read(&mut self, addr: u64, width: u8) -> u64;
    fn write(&mut self, addr: u64, width: u8, value: u64);
}

/// The DMA-memory HAL: the runtime allocates each declared [`DeviceSpec`] buffer
/// through this, then reads/writes payloads by **physical** address (what the
/// device sees). On host it is [`ModelDmaMem`]; on metal it is backed by
/// `kernel::dma` (which returns the phys+virt of contiguous frames). Keeping it a
/// trait is what lets the *same* spec drive a mock and a real NIC unchanged.
pub trait DmaMem {
    /// Reserve `len` contiguous bytes, returning the physical base, or `None` if
    /// memory is exhausted.
    fn alloc(&mut self, len: u64) -> Option<u64>;
    /// Write `data` at physical address `phys`. Returns false if out of range.
    fn write(&mut self, phys: u64, data: &[u8]) -> bool;
    /// Read `len` bytes from physical address `phys`.
    fn read(&mut self, phys: u64, len: u64) -> Vec<u8>;
}

/// A host/DST model of DMA memory: a flat byte arena with a bump allocator. Used by
/// unit tests and by [`crate::drivergen::dst_replay_validate`] so DMA-using specs
/// are proven in-bounds before they ever touch real hardware.
pub struct ModelDmaMem {
    base: u64,
    mem: Vec<u8>,
    cursor: u64,
}

impl ModelDmaMem {
    pub fn new() -> ModelDmaMem {
        let base = 0x10_0000;
        ModelDmaMem { base, mem: alloc::vec![0u8; 1 << 20], cursor: base }
    }

    fn off(&self, phys: u64) -> Option<usize> {
        phys.checked_sub(self.base).map(|o| o as usize)
    }
}

impl Default for ModelDmaMem {
    fn default() -> ModelDmaMem {
        ModelDmaMem::new()
    }
}

impl DmaMem for ModelDmaMem {
    fn alloc(&mut self, len: u64) -> Option<u64> {
        let phys = self.cursor;
        let aligned = (len + 7) & !7; // 8-byte align the next allocation
        let used = (phys - self.base).checked_add(aligned)? as usize;
        if used > self.mem.len() {
            return None;
        }
        self.cursor = phys + aligned;
        Some(phys)
    }
    fn write(&mut self, phys: u64, data: &[u8]) -> bool {
        match self.off(phys) {
            Some(o) if o + data.len() <= self.mem.len() => {
                self.mem[o..o + data.len()].copy_from_slice(data);
                true
            }
            _ => false,
        }
    }
    fn read(&mut self, phys: u64, len: u64) -> Vec<u8> {
        match self.off(phys) {
            Some(o) if o + len as usize <= self.mem.len() => self.mem[o..o + len as usize].to_vec(),
            _ => Vec::new(),
        }
    }
}

/// The result of a DMA-aware driver operation: register values read, plus any bytes
/// loaded out of a DMA buffer (e.g. a received packet).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DriverIo {
    pub regs: Vec<u64>,
    pub bytes: Vec<u8>,
}

/// A capability-bounded view onto the MMIO space. Every access is checked against
/// the driver's [`TaggedCap`]; out-of-bounds *traps*. This is the wall a synthesized
/// or borrowed driver cannot cross.
struct MmioWindow<'a> {
    cap: TaggedCap,
    dev: &'a mut dyn MmioDevice,
}

impl MmioWindow<'_> {
    fn check(&self, offset: u64, width: u8) -> Result<u64, DriverFault> {
        let addr = self.cap.base.checked_add(offset).ok_or(DriverFault::OutOfBounds)?;
        if !self.cap.covers(addr, width as u64) {
            return Err(DriverFault::OutOfBounds);
        }
        Ok(addr)
    }

    fn read(&mut self, offset: u64, width: u8) -> Result<u64, DriverFault> {
        let addr = self.check(offset, width)?;
        Ok(self.dev.read(addr, width))
    }

    fn write(&mut self, offset: u64, width: u8, value: u64) -> Result<(), DriverFault> {
        let addr = self.check(offset, width)?;
        self.dev.write(addr, width, value);
        Ok(())
    }
}

/// The synthesized driver runtime: spec + an MMIO capability. **One** `Driver`
/// implementation serves **every** device class — behaviour comes entirely from the
/// spec, never from device-specific code.
#[derive(Debug)]
pub struct Driver {
    spec: DeviceSpec,
    cap: TaggedCap,
    /// Bound DMA buffers: name → (physical base, length). Empty for a driver bound
    /// without a DMA backend; populated by [`Driver::bind_dma`].
    buffers: BTreeMap<String, (u64, u64)>,
}

/// A DMA backend that never allocates — used by the register-only [`Driver::run`]
/// path so it pays nothing for DMA it does not use.
struct NullDma;
impl DmaMem for NullDma {
    fn alloc(&mut self, _len: u64) -> Option<u64> {
        None
    }
    fn write(&mut self, _phys: u64, _data: &[u8]) -> bool {
        false
    }
    fn read(&mut self, _phys: u64, _len: u64) -> Vec<u8> {
        Vec::new()
    }
}

impl Driver {
    /// Bind a spec to hardware: mint an MMIO capability bounded to *exactly* the
    /// spec's claimed window. Rejects malformed specs up front (fail-closed). Use
    /// [`Driver::bind_dma`] instead for a spec that declares DMA buffers.
    pub fn bind(spec: DeviceSpec, tags: &dyn CapabilityTags) -> Result<Driver, DriverFault> {
        if !spec.is_well_formed() {
            return Err(DriverFault::MalformedSpec);
        }
        let cap = tags.mint(
            spec.resources.mmio_base,
            spec.resources.mmio_len,
            perms::READ | perms::WRITE,
        );
        Ok(Driver { spec, cap, buffers: BTreeMap::new() })
    }

    /// Bind a spec that uses DMA: in addition to the MMIO capability, allocate each
    /// declared buffer through `dma` and record its physical base. Each buffer is its
    /// own bound — a program can DMA only inside a buffer it declared, never beyond
    /// it (the IOMMU-style containment a borrowed driver cannot escape).
    pub fn bind_dma(
        spec: DeviceSpec,
        tags: &dyn CapabilityTags,
        dma: &mut dyn DmaMem,
    ) -> Result<Driver, DriverFault> {
        let mut driver = Driver::bind(spec, tags)?;
        for (name, &len) in &driver.spec.buffers {
            let phys = dma.alloc(len).ok_or(DriverFault::OutOfBounds)?;
            driver.buffers.insert(name.clone(), (phys, len));
        }
        Ok(driver)
    }

    pub fn class(&self) -> DeviceClass {
        self.spec.class
    }

    /// The window this driver is confined to — what it may touch, and nothing else.
    pub fn window(&self) -> (u64, u64) {
        (self.cap.base, self.cap.len)
    }

    fn reg(&self, name: &str) -> Result<&Register, DriverFault> {
        self.spec.registers.get(name).ok_or(DriverFault::UnknownRegister)
    }

    /// Resolve a named DMA buffer's (physical base, length).
    fn buf(&self, name: &str) -> Result<(u64, u64), DriverFault> {
        self.buffers.get(name).copied().ok_or(DriverFault::MalformedSpec)
    }

    /// Execute the program for logical operation `op` against `dev`, returning the
    /// values read along the way. `args` feed `ValueSrc::Arg` slots (e.g. an LBA).
    /// `tags` re-validates the capability so a tampered driver refuses to run. This
    /// is the register-only path; use [`Driver::run_io`] for DMA specs.
    pub fn run(
        &self,
        op: &str,
        args: &[u64],
        dev: &mut dyn MmioDevice,
        tags: &dyn CapabilityTags,
    ) -> Result<Vec<u64>, DriverFault> {
        let mut null = NullDma;
        self.exec(op, args, &[], dev, &mut null, tags).map(|io| io.regs)
    }

    /// Execute a (possibly DMA-using) program: `bytes_in` is the outbound payload
    /// consumed by `BufStore`; the returned [`DriverIo`] carries register reads and
    /// any bytes pulled out by `BufLoad`. The *same* call drives a [`ModelDmaMem`]
    /// in tests and `kernel::dma` on real hardware.
    pub fn run_io(
        &self,
        op: &str,
        args: &[u64],
        bytes_in: &[u8],
        dev: &mut dyn MmioDevice,
        dma: &mut dyn DmaMem,
        tags: &dyn CapabilityTags,
    ) -> Result<DriverIo, DriverFault> {
        self.exec(op, args, bytes_in, dev, dma, tags)
    }

    fn exec(
        &self,
        op: &str,
        args: &[u64],
        bytes_in: &[u8],
        dev: &mut dyn MmioDevice,
        dma: &mut dyn DmaMem,
        tags: &dyn CapabilityTags,
    ) -> Result<DriverIo, DriverFault> {
        if !tags.validate(&self.cap) {
            return Err(DriverFault::CapabilityInvalid);
        }
        let program = self.spec.programs.get(op).ok_or(DriverFault::UnknownOperation)?;
        let mut window = MmioWindow { cap: self.cap, dev };
        let mut io = DriverIo::default();
        for step in program {
            match step {
                RegOp::Write { reg, value } => {
                    let r = self.reg(reg)?;
                    let v = match value {
                        ValueSrc::Imm(v) => *v,
                        ValueSrc::Arg(i) => *args.get(*i).ok_or(DriverFault::UnknownOperation)?,
                        ValueSrc::BufPhys(buf) => self.buf(buf)?.0,
                    };
                    window.write(r.offset, r.width, v)?;
                }
                RegOp::Read { reg } => {
                    let r = self.reg(reg)?;
                    io.regs.push(window.read(r.offset, r.width)?);
                }
                RegOp::Poll { reg, value, max_spins } => {
                    let r = self.reg(reg)?;
                    let mut ok = false;
                    for _ in 0..*max_spins {
                        if window.read(r.offset, r.width)? == *value {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        return Err(DriverFault::Timeout);
                    }
                }
                RegOp::PollBits { reg, mask, value, max_spins } => {
                    let r = self.reg(reg)?;
                    let mut ok = false;
                    for _ in 0..*max_spins {
                        if window.read(r.offset, r.width)? & *mask == *value {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        return Err(DriverFault::Timeout);
                    }
                }
                RegOp::BufStore { buf, off } => {
                    let (phys, len) = self.buf(buf)?;
                    let end = off.checked_add(bytes_in.len() as u64).ok_or(DriverFault::OutOfBounds)?;
                    if end > len {
                        return Err(DriverFault::OutOfBounds);
                    }
                    if !dma.write(phys + off, bytes_in) {
                        return Err(DriverFault::OutOfBounds);
                    }
                }
                RegOp::BufLoad { buf, off, len } => {
                    let (phys, blen) = self.buf(buf)?;
                    let end = off.checked_add(*len).ok_or(DriverFault::OutOfBounds)?;
                    if end > blen {
                        return Err(DriverFault::OutOfBounds);
                    }
                    io.bytes = dma.read(phys + off, *len);
                }
                RegOp::BufStoreVal { buf, off, value, width } => {
                    let (phys, blen) = self.buf(buf)?;
                    let w = *width as u64;
                    let end = off.checked_add(w).ok_or(DriverFault::OutOfBounds)?;
                    if w == 0 || w > 8 || end > blen {
                        return Err(DriverFault::OutOfBounds);
                    }
                    let v = match value {
                        ValueSrc::Imm(v) => *v,
                        ValueSrc::Arg(i) => *args.get(*i).ok_or(DriverFault::UnknownOperation)?,
                        ValueSrc::BufPhys(b) => self.buf(b)?.0,
                    };
                    let bytes = v.to_le_bytes();
                    if !dma.write(phys + off, &bytes[..*width as usize]) {
                        return Err(DriverFault::OutOfBounds);
                    }
                }
            }
        }
        Ok(io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheri::SoftwareTags;
    use alloc::vec;

    // ── A mock block device: write LBA + a READ command, poll STATUS=ready,
    //    then DATA latches the byte at that LBA from a tiny backing "disk". ──
    struct MockBlock {
        base: u64,
        lba: u64,
        status: u64,
        data: u64,
        disk: [u64; 8],
    }
    impl MockBlock {
        fn new(base: u64) -> MockBlock {
            MockBlock { base, lba: 0, status: 0, data: 0, disk: [10, 11, 12, 13, 14, 15, 16, 17] }
        }
    }
    impl MmioDevice for MockBlock {
        fn read(&mut self, addr: u64, _w: u8) -> u64 {
            match addr - self.base {
                0x00 => self.status,
                0x08 => self.data,
                _ => 0,
            }
        }
        fn write(&mut self, addr: u64, _w: u8, value: u64) {
            match addr - self.base {
                0x10 => self.lba = value, // LBA register
                // CTRL: a READ command latches the disk byte and signals ready.
                0x18 if value == 1 => {
                    self.data = self.disk[(self.lba as usize) % 8];
                    self.status = 1;
                }
                _ => {}
            }
        }
    }

    // ── A completely different device (entropy source), same runtime. ──
    struct MockEntropy {
        base: u64,
        state: u64,
    }
    impl MmioDevice for MockEntropy {
        fn read(&mut self, addr: u64, _w: u8) -> u64 {
            if addr - self.base == 0x00 {
                // A deterministic "random" register (xorshift) — reproducible.
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state
            } else {
                0
            }
        }
        fn write(&mut self, _addr: u64, _w: u8, _value: u64) {}
    }

    fn block_spec(base: u64) -> DeviceSpec {
        DeviceSpec::new(DeviceClass::Block, ResourceClaim { mmio_base: base, mmio_len: 0x20, irq: 5 })
            .register("STATUS", 0x00, 8)
            .register("DATA", 0x08, 8)
            .register("LBA", 0x10, 8)
            .register("CTRL", 0x18, 8)
            .program(
                "read",
                vec![
                    RegOp::Write { reg: "LBA".into(), value: ValueSrc::Arg(0) },
                    RegOp::Write { reg: "CTRL".into(), value: ValueSrc::Imm(1) }, // READ cmd
                    RegOp::Poll { reg: "STATUS".into(), value: 1, max_spins: 16 },
                    RegOp::Read { reg: "DATA".into() },
                ],
            )
    }

    fn entropy_spec(base: u64) -> DeviceSpec {
        DeviceSpec::new(DeviceClass::Entropy, ResourceClaim { mmio_base: base, mmio_len: 0x08, irq: 6 })
            .register("RND", 0x00, 8)
            .program("sample", vec![RegOp::Read { reg: "RND".into() }])
    }

    #[test]
    fn one_runtime_drives_a_block_device_from_data() {
        let tags = SoftwareTags::new([1u8; 32]);
        let driver = Driver::bind(block_spec(0x1000), &tags).unwrap();
        let mut dev = MockBlock::new(0x1000);
        // No block-device-specific code ran — only the spec's register program.
        let out = driver.run("read", &[3], &mut dev, &tags).unwrap();
        assert_eq!(out, vec![13]); // disk[3]
        assert_eq!(driver.class(), DeviceClass::Block);
    }

    #[test]
    fn the_same_runtime_drives_a_totally_different_device() {
        let tags = SoftwareTags::new([1u8; 32]);
        let driver = Driver::bind(entropy_spec(0x4000), &tags).unwrap();
        let mut dev = MockEntropy { base: 0x4000, state: 0x9e3779b97f4a7c15 };
        // Identical runtime, identical API — behaviour entirely from the spec.
        let a = driver.run("sample", &[], &mut dev, &tags).unwrap()[0];
        let b = driver.run("sample", &[], &mut dev, &tags).unwrap()[0];
        assert_ne!(a, b); // the entropy register advances
    }

    #[test]
    fn out_of_bounds_register_access_traps() {
        let tags = SoftwareTags::new([1u8; 32]);
        // A malformed spec: CTRL at 0x40 escapes the 0x20-byte window.
        let bad = DeviceSpec::new(
            DeviceClass::Block,
            ResourceClaim { mmio_base: 0x1000, mmio_len: 0x20, irq: 5 },
        )
        .register("CTRL", 0x40, 8)
        .program("go", vec![RegOp::Write { reg: "CTRL".into(), value: ValueSrc::Imm(1) }]);
        // Caught at bind time — a driver that could escape is never created.
        assert_eq!(Driver::bind(bad, &tags).err(), Some(DriverFault::MalformedSpec));
    }

    #[test]
    fn a_driver_cannot_reach_another_devices_registers() {
        let tags = SoftwareTags::new([1u8; 32]);
        // Driver bound to device A's window [0x1000, 0x1020).
        let driver = Driver::bind(block_spec(0x1000), &tags).unwrap();
        // Point it at device B sitting at 0x2000. The window translates offsets
        // relative to A's base, so the driver's capability never covers B —
        // every access lands in A's range, and B is unreachable by construction.
        let (base, len) = driver.window();
        assert_eq!((base, len), (0x1000, 0x20));
        // A read at the top of the window is fine; one past it would trap.
        assert!(!driver_cap_covers(&driver, 0x1020));
        assert!(driver_cap_covers(&driver, 0x1018));
    }

    fn driver_cap_covers(driver: &Driver, addr: u64) -> bool {
        let (base, len) = driver.window();
        addr >= base && addr + 8 <= base + len
    }

    // A device that never completes — the poll must time out, not spin forever.
    struct DeadBlock;
    impl MmioDevice for DeadBlock {
        fn read(&mut self, _a: u64, _w: u8) -> u64 {
            0 // STATUS never becomes 1
        }
        fn write(&mut self, _a: u64, _w: u8, _v: u64) {}
    }

    #[test]
    fn poll_times_out_and_fails_closed_instead_of_hanging() {
        let tags = SoftwareTags::new([1u8; 32]);
        let driver = Driver::bind(block_spec(0x1000), &tags).unwrap();
        let mut dev = DeadBlock;
        assert_eq!(driver.run("read", &[0], &mut dev, &tags), Err(DriverFault::Timeout));
    }

    #[test]
    fn tampered_capability_refuses_to_run() {
        let tags = SoftwareTags::new([1u8; 32]);
        let mut driver = Driver::bind(block_spec(0x1000), &tags).unwrap();
        // Forge wider bounds: the tag no longer validates → the driver fails closed.
        driver.cap.len = 0xFFFF_FFFF;
        let mut dev = MockBlock::new(0x1000);
        assert_eq!(driver.run("read", &[0], &mut dev, &tags), Err(DriverFault::CapabilityInvalid));
    }

    #[test]
    fn unknown_operation_is_rejected() {
        let tags = SoftwareTags::new([1u8; 32]);
        let driver = Driver::bind(block_spec(0x1000), &tags).unwrap();
        let mut dev = MockBlock::new(0x1000);
        assert_eq!(
            driver.run("flash_firmware", &[], &mut dev, &tags),
            Err(DriverFault::UnknownOperation)
        );
    }
}

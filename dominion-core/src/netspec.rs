//! Concrete NIC device specs — RTL8139 and Intel e1000 — that drive **real**
//! hardware purely from data (see `docs/architecture/capability-shim-and-foreign-compat.md`
//! §3.1). These are the two NICs every desktop OS ships a driver for (Linux
//! `8139too`/`e1000`, Windows `rtl8139`/`e1net`) and the two QEMU emulates, so a
//! [`DeviceSpec`] here stands in for a whole slice of the legacy driver ecosystem
//! while running through the *same* capability-bounded [`Driver`](crate::driver)
//! runtime as every other device — no per-device kernel code.
//!
//! * **RTL8139** — a simple register+single-buffer NIC. TX: stage the frame in a DMA
//!   buffer, point `TSAD` at it, write the length to `TSD` (which starts the DMA),
//!   poll `TSD.TOK`. RX: a ring buffer at `RBSTART`.
//! * **e1000** — a descriptor-ring NIC, representative of the real world
//!   (e1000/AHCI/NVMe/virtio all use in-memory descriptor rings). TX: build a
//!   descriptor in a DMA ring (buffer address + length + command), bump the tail
//!   register, poll the head register.
//!
//! The same [`crate::driver::RegOp`] vocabulary expresses both. Pure, safe `no_std`;
//! host-tested against faithful loopback device models, and bound to the real device
//! by the kernel.

use crate::driver::{DeviceClass, DeviceSpec, RegOp, ResourceClaim, ValueSrc};
use alloc::vec;

fn imm(v: u64) -> ValueSrc {
    ValueSrc::Imm(v)
}
fn buf(name: &str) -> ValueSrc {
    ValueSrc::BufPhys(name.into())
}

// ─────────────────────────────── RTL8139 ───────────────────────────────

/// RTL8139 register offsets and status bits (from the RealTek datasheet).
pub mod rtl8139 {
    pub const IDR0: u64 = 0x00; // MAC bytes 0..3
    pub const TSD0: u64 = 0x10; // Transmit Status of Descriptor 0
    pub const TSAD0: u64 = 0x20; // Transmit Start Address of Descriptor 0
    pub const RBSTART: u64 = 0x30; // Receive Buffer Start (physical)
    pub const CMD: u64 = 0x37; // Command register
    pub const CAPR: u64 = 0x38; // Current Address of Packet Read
    pub const CBR: u64 = 0x3A; // Current Buffer Address
    pub const IMR: u64 = 0x3C; // Interrupt Mask
    pub const ISR: u64 = 0x3E; // Interrupt Status
    pub const RCR: u64 = 0x44; // Receive Configuration
    pub const CONFIG1: u64 = 0x52; // Config 1 (power)

    pub const CMD_RST: u64 = 0x10;
    pub const CMD_RE: u64 = 0x08;
    pub const CMD_TE: u64 = 0x04;
    pub const TSD_TOK: u64 = 0x8000; // transmit OK
    pub const ISR_ROK: u64 = 0x0001; // receive OK
    pub const ISR_TOK: u64 = 0x0004; // transmit OK
    /// Accept All Phys + Phys Match + Multicast + Broadcast + WRAP.
    pub const RCR_CFG: u64 = 0x0000_008F;
    /// 8 KiB ring + 16-byte header slack + 1.5 KiB overflow guard (WRAP set).
    pub const RX_RING_LEN: u64 = 8192 + 16 + 1536;
}

pub const RTL8139_TXBUF: &str = "tx0";
pub const RTL8139_RXRING: &str = "rxring";

/// The RTL8139 driver, expressed as data. `init`/`read_mac`/`tx`/`rx_ready`/`rx`.
pub fn rtl8139_spec(mmio_base: u64, irq: u32) -> DeviceSpec {
    use rtl8139::*;
    DeviceSpec::new(DeviceClass::Net, ResourceClaim { mmio_base, mmio_len: 0x100, irq })
        .register("IDR0", IDR0, 4)
        .register("IDR4", 0x04, 2)
        .register("TSD0", TSD0, 4)
        .register("TSAD0", TSAD0, 4)
        .register("RBSTART", RBSTART, 4)
        .register("CMD", CMD, 1)
        .register("CAPR", CAPR, 2)
        .register("CBR", CBR, 2)
        .register("IMR", IMR, 2)
        .register("ISR", ISR, 2)
        .register("RCR", RCR, 4)
        .register("CONFIG1", CONFIG1, 1)
        .buffer(RTL8139_TXBUF, 2048)
        .buffer(RTL8139_RXRING, RX_RING_LEN)
        // Power on, reset, point RX at our ring, accept frames, enable TX+RX.
        .program(
            "init",
            vec![
                RegOp::Write { reg: "CONFIG1".into(), value: imm(0x00) },
                RegOp::Write { reg: "CMD".into(), value: imm(CMD_RST) },
                // Wait for the reset bit (RST) to self-clear — other CMD bits may vary.
                RegOp::PollBits { reg: "CMD".into(), mask: CMD_RST, value: 0, max_spins: 1_000_000 },
                RegOp::Write { reg: "RBSTART".into(), value: buf(RTL8139_RXRING) },
                RegOp::Write { reg: "IMR".into(), value: imm(0x0000) }, // polled
                RegOp::Write { reg: "RCR".into(), value: imm(RCR_CFG) },
                RegOp::Write { reg: "CMD".into(), value: imm(CMD_TE | CMD_RE) },
            ],
        )
        .program("read_mac", vec![RegOp::Read { reg: "IDR0".into() }, RegOp::Read { reg: "IDR4".into() }])
        // Stage the frame (bytes_in), point the descriptor at it, write the length
        // to start the DMA, wait for transmit-OK.
        .program(
            "tx",
            vec![
                RegOp::BufStore { buf: RTL8139_TXBUF.into(), off: 0 },
                RegOp::Write { reg: "TSAD0".into(), value: buf(RTL8139_TXBUF) },
                RegOp::Write { reg: "TSD0".into(), value: ValueSrc::Arg(0) },
                RegOp::PollBits { reg: "TSD0".into(), mask: TSD_TOK, value: TSD_TOK, max_spins: 1_000_000 },
            ],
        )
        // Non-blocking check: is a received frame waiting? (ISR.ROK)
        .program(
            "rx_ready",
            vec![RegOp::PollBits { reg: "ISR".into(), mask: ISR_ROK, value: ISR_ROK, max_spins: 1 }],
        )
        // Pull the first packet (4-byte header + frame) out of the ring, ack ROK.
        .program(
            "rx",
            vec![
                RegOp::BufLoad { buf: RTL8139_RXRING.into(), off: 0, len: 1518 },
                RegOp::Write { reg: "ISR".into(), value: imm(ISR_ROK) },
            ],
        )
}

// ──────────────────────────────── e1000 ────────────────────────────────

/// Intel e1000 (82540EM) register offsets and bits.
pub mod e1000 {
    pub const CTRL: u64 = 0x0000;
    pub const STATUS: u64 = 0x0008;
    pub const RCTL: u64 = 0x0100;
    pub const TCTL: u64 = 0x0400;
    pub const RDBAL: u64 = 0x2800;
    pub const RDBAH: u64 = 0x2804;
    pub const RDLEN: u64 = 0x2808;
    pub const RDH: u64 = 0x2810;
    pub const RDT: u64 = 0x2818;
    pub const TDBAL: u64 = 0x3800;
    pub const TDBAH: u64 = 0x3804;
    pub const TDLEN: u64 = 0x3808;
    pub const TDH: u64 = 0x3810;
    pub const TDT: u64 = 0x3818;
    pub const RAL: u64 = 0x5400; // MAC low
    pub const RAH: u64 = 0x5404; // MAC high

    pub const TCTL_EN_PSP: u64 = 0x0000_010A; // EN | PSP
    pub const RCTL_EN_BAM: u64 = 0x0000_8002; // EN | BAM
    pub const TXD_CMD_EOP_RS: u64 = 0x09; // End-Of-Packet | Report-Status
    /// 8 descriptors × 16 bytes = 128 (TDLEN/RDLEN must be a multiple of 128).
    pub const RING_LEN: u64 = 128;
    pub const DESC_SIZE: u64 = 16;
}

pub const E1000_TDRING: &str = "tdring";
pub const E1000_TXBUF: &str = "txbuf";
pub const E1000_RDRING: &str = "rdring";
pub const E1000_RXBUF: &str = "rxbuf";

/// The e1000 driver, expressed as data — a descriptor-ring NIC.
pub fn e1000_spec(mmio_base: u64, irq: u32) -> DeviceSpec {
    use e1000::*;
    DeviceSpec::new(DeviceClass::Net, ResourceClaim { mmio_base, mmio_len: 0x6000, irq })
        .register("CTRL", CTRL, 4)
        .register("STATUS", STATUS, 4)
        .register("RCTL", RCTL, 4)
        .register("TCTL", TCTL, 4)
        .register("RDBAL", RDBAL, 4)
        .register("RDBAH", RDBAH, 4)
        .register("RDLEN", RDLEN, 4)
        .register("RDH", RDH, 4)
        .register("RDT", RDT, 4)
        .register("TDBAL", TDBAL, 4)
        .register("TDBAH", TDBAH, 4)
        .register("TDLEN", TDLEN, 4)
        .register("TDH", TDH, 4)
        .register("TDT", TDT, 4)
        .register("RAL", RAL, 4)
        .register("RAH", RAH, 4)
        .buffer(E1000_TDRING, RING_LEN)
        .buffer(E1000_TXBUF, 2048)
        .buffer(E1000_RDRING, RING_LEN)
        .buffer(E1000_RXBUF, 2048)
        // Program the TX and RX descriptor rings, seed RX descriptor 0's buffer,
        // and enable both engines.
        .program(
            "init",
            vec![
                RegOp::Write { reg: "TDBAL".into(), value: buf(E1000_TDRING) },
                RegOp::Write { reg: "TDBAH".into(), value: imm(0) },
                RegOp::Write { reg: "TDLEN".into(), value: imm(RING_LEN) },
                RegOp::Write { reg: "TDH".into(), value: imm(0) },
                RegOp::Write { reg: "TDT".into(), value: imm(0) },
                RegOp::Write { reg: "TCTL".into(), value: imm(TCTL_EN_PSP) },
                RegOp::Write { reg: "RDBAL".into(), value: buf(E1000_RDRING) },
                RegOp::Write { reg: "RDBAH".into(), value: imm(0) },
                RegOp::Write { reg: "RDLEN".into(), value: imm(RING_LEN) },
                // RX descriptor 0 → rx buffer.
                RegOp::BufStoreVal { buf: E1000_RDRING.into(), off: 0, value: buf(E1000_RXBUF), width: 8 },
                RegOp::Write { reg: "RDH".into(), value: imm(0) },
                RegOp::Write { reg: "RDT".into(), value: imm(7) },
                RegOp::Write { reg: "RCTL".into(), value: imm(RCTL_EN_BAM) },
            ],
        )
        .program("read_mac", vec![RegOp::Read { reg: "RAL".into() }, RegOp::Read { reg: "RAH".into() }])
        // Build TX descriptor 0 (addr, length, cmd), bump the tail, wait for the head
        // to catch up (descriptor consumed = transmitted).
        .program(
            "tx",
            vec![
                RegOp::BufStore { buf: E1000_TXBUF.into(), off: 0 },
                RegOp::BufStoreVal { buf: E1000_TDRING.into(), off: 0, value: buf(E1000_TXBUF), width: 8 },
                RegOp::BufStoreVal { buf: E1000_TDRING.into(), off: 8, value: ValueSrc::Arg(0), width: 2 },
                RegOp::BufStoreVal { buf: E1000_TDRING.into(), off: 11, value: imm(TXD_CMD_EOP_RS), width: 1 },
                RegOp::Write { reg: "TDT".into(), value: imm(1) },
                RegOp::Poll { reg: "TDH".into(), value: 1, max_spins: 1_000_000 },
            ],
        )
}

/// A default registry of device specs for the Dominion `Driver::*` API: the two real
/// NICs plus a few canonical class templates (which the cooperative model device can
/// fully execute). Drivers are *data*, so Dominion can list, inspect, edit and invoke
/// these — all capability-gated.
pub fn default_registry() -> alloc::collections::BTreeMap<alloc::string::String, DeviceSpec> {
    use crate::drivergen::{class_template, HwClass};
    use alloc::string::ToString;
    let mut m = alloc::collections::BTreeMap::new();
    m.insert("rtl8139".to_string(), rtl8139_spec(0xFEBC_0000, 11));
    m.insert("e1000".to_string(), e1000_spec(0xFEBF_0000, 11));
    let claim = |base| ResourceClaim { mmio_base: base, mmio_len: 0x1000, irq: 10 };
    m.insert("nvme".to_string(), class_template(HwClass::Nvme, claim(0x9000)));
    m.insert("ahci".to_string(), class_template(HwClass::Ahci, claim(0xA000)));
    m.insert("xhci".to_string(), class_template(HwClass::Xhci, claim(0xB000)));
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheri::SoftwareTags;
    use crate::driver::{DmaMem, Driver, MmioDevice};
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    // ── A shared DMA arena: both the driver runtime and the loopback device model
    //    see the same physical memory, exactly as on real hardware. ──
    struct Arena {
        base: u64,
        mem: Vec<u8>,
        cursor: u64,
    }
    #[derive(Clone)]
    struct SharedDma(Rc<RefCell<Arena>>);
    impl SharedDma {
        fn new() -> SharedDma {
            let base = 0x20_0000;
            SharedDma(Rc::new(RefCell::new(Arena { base, mem: vec![0u8; 1 << 20], cursor: base })))
        }
        fn read_at(&self, phys: u64, len: usize) -> Vec<u8> {
            let a = self.0.borrow();
            let o = (phys - a.base) as usize;
            a.mem[o..o + len].to_vec()
        }
        fn write_at(&self, phys: u64, data: &[u8]) {
            let mut a = self.0.borrow_mut();
            let o = (phys - a.base) as usize;
            a.mem[o..o + data.len()].copy_from_slice(data);
        }
    }
    impl DmaMem for SharedDma {
        fn alloc(&mut self, len: u64) -> Option<u64> {
            let mut a = self.0.borrow_mut();
            let phys = a.cursor;
            let aligned = (len + 15) & !15;
            let used = (phys - a.base + aligned) as usize;
            if used > a.mem.len() {
                return None;
            }
            a.cursor = phys + aligned;
            Some(phys)
        }
        fn write(&mut self, phys: u64, data: &[u8]) -> bool {
            self.write_at(phys, data);
            true
        }
        fn read(&mut self, phys: u64, len: u64) -> Vec<u8> {
            self.read_at(phys, len as usize)
        }
    }

    // ── A faithful RTL8139 loopback model: writing the length to TSD0 DMAs the TX
    //    buffer out (captured here); test code can `deliver` an inbound frame. ──
    struct Rtl8139Model {
        base: u64,
        dma: SharedDma,
        tsad0: u64,
        tsd0: u64,
        rbstart: u64,
        cmd: u64,
        isr: u64,
        idr: [u8; 6],
        last_tx: Option<Vec<u8>>,
    }
    impl Rtl8139Model {
        fn new(base: u64, dma: SharedDma) -> Rtl8139Model {
            Rtl8139Model {
                base,
                dma,
                tsad0: 0,
                tsd0: 0,
                rbstart: 0,
                cmd: 0,
                isr: 0,
                idr: [0x52, 0x54, 0x00, 0x12, 0x34, 0x56],
                last_tx: None,
            }
        }
        /// Simulate an inbound frame: write the rtl8139 RX header + frame into the
        /// ring and raise ISR.ROK.
        fn deliver(&mut self, frame: &[u8]) {
            let mut pkt = Vec::new();
            pkt.extend_from_slice(&1u16.to_le_bytes()); // status: ROK
            pkt.extend_from_slice(&((frame.len() + 4) as u16).to_le_bytes()); // len incl CRC
            pkt.extend_from_slice(frame);
            self.dma.write_at(self.rbstart, &pkt);
            self.isr |= rtl8139::ISR_ROK;
        }
    }
    impl MmioDevice for Rtl8139Model {
        fn read(&mut self, addr: u64, _w: u8) -> u64 {
            match addr - self.base {
                rtl8139::IDR0 => u32::from_le_bytes([self.idr[0], self.idr[1], self.idr[2], self.idr[3]]) as u64,
                rtl8139::TSD0 => self.tsd0,
                rtl8139::CMD => self.cmd,
                rtl8139::ISR => self.isr,
                _ => 0,
            }
        }
        fn write(&mut self, addr: u64, _w: u8, value: u64) {
            match addr - self.base {
                rtl8139::CONFIG1 => {}
                rtl8139::CMD => {
                    // RST auto-clears (and clears the register); else store.
                    self.cmd = if value & rtl8139::CMD_RST != 0 { 0 } else { value };
                }
                rtl8139::RBSTART => self.rbstart = value,
                rtl8139::TSAD0 => self.tsad0 = value,
                rtl8139::TSD0 => {
                    // Writing the length starts the TX DMA from TSAD0.
                    let len = (value & 0x1FFF) as usize;
                    let frame = self.dma.read_at(self.tsad0, len);
                    self.last_tx = Some(frame);
                    self.tsd0 = value | rtl8139::TSD_TOK | 0x2000; // TOK | OWN-done
                    self.isr |= rtl8139::ISR_TOK;
                }
                rtl8139::ISR => self.isr &= !value, // write-1-clear
                _ => {}
            }
        }
    }

    fn arp_frame() -> Vec<u8> {
        // A minimal broadcast Ethernet frame (dst, src, ethertype ARP) — enough to
        // prove the bytes round-trip through the spec's DMA path.
        let mut f = Vec::new();
        f.extend_from_slice(&[0xFF; 6]); // broadcast dst
        f.extend_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]); // src
        f.extend_from_slice(&[0x08, 0x06]); // ARP
        f.extend_from_slice(&[0xAB; 28]); // ARP payload (placeholder)
        f
    }

    #[test]
    fn rtl8139_spec_is_well_formed_and_binds() {
        let tags = SoftwareTags::new([3u8; 32]);
        let mut dma = crate::driver::ModelDmaMem::new();
        let spec = rtl8139_spec(0xFEBC_0000, 11);
        assert!(spec.is_well_formed());
        assert!(Driver::bind_dma(spec, &tags, &mut dma).is_ok());
    }

    #[test]
    fn rtl8139_transmits_a_real_frame_through_the_spec() {
        let tags = SoftwareTags::new([3u8; 32]);
        let mut dma = SharedDma::new();
        let driver = Driver::bind_dma(rtl8139_spec(0xFEBC_0000, 11), &tags, &mut dma).unwrap();
        let mut nic = Rtl8139Model::new(0xFEBC_0000, dma.clone());

        // Bring the NIC up and read its MAC — all from the spec.
        driver.run_io("init", &[], &[], &mut nic, &mut dma, &tags).unwrap();
        let mac = driver.run_io("read_mac", &[], &[], &mut nic, &mut dma, &tags).unwrap();
        assert_eq!(mac.regs[0] & 0xFFFF, 0x5452); // 0x52,0x54 little-endian

        // Transmit a frame; the model captures exactly the bytes we staged.
        let frame = arp_frame();
        driver
            .run_io("tx", &[frame.len() as u64], &frame, &mut nic, &mut dma, &tags)
            .unwrap();
        assert_eq!(nic.last_tx.as_deref(), Some(frame.as_slice()));
    }

    #[test]
    fn rtl8139_receives_a_real_frame_through_the_spec() {
        let tags = SoftwareTags::new([3u8; 32]);
        let mut dma = SharedDma::new();
        let driver = Driver::bind_dma(rtl8139_spec(0xFEBC_0000, 11), &tags, &mut dma).unwrap();
        let mut nic = Rtl8139Model::new(0xFEBC_0000, dma.clone());
        driver.run_io("init", &[], &[], &mut nic, &mut dma, &tags).unwrap();

        // No frame yet → rx_ready times out (fails closed).
        assert!(driver.run_io("rx_ready", &[], &[], &mut nic, &mut dma, &tags).is_err());

        // Deliver one, then the spec sees it ready and pulls it out of the ring.
        let frame = arp_frame();
        nic.deliver(&frame);
        assert!(driver.run_io("rx_ready", &[], &[], &mut nic, &mut dma, &tags).is_ok());
        let io = driver.run_io("rx", &[], &[], &mut nic, &mut dma, &tags).unwrap();
        // Skip the 4-byte rtl8139 header; the frame bytes follow.
        assert_eq!(&io.bytes[4..4 + frame.len()], frame.as_slice());
        // ROK was acked.
        assert_eq!(nic.isr & rtl8139::ISR_ROK, 0);
    }

    // ── A faithful e1000 loopback model: bumping TDT consumes TX descriptor 0
    //    (DMA the buffer out) and advances TDH. ──
    struct E1000Model {
        base: u64,
        dma: SharedDma,
        tdbal: u64,
        tdh: u64,
        ral: u32,
        rah: u32,
        last_tx: Option<Vec<u8>>,
    }
    impl E1000Model {
        fn new(base: u64, dma: SharedDma) -> E1000Model {
            E1000Model { base, dma, tdbal: 0, tdh: 0, ral: 0x12345678, rah: 0x9ABC, last_tx: None }
        }
    }
    impl MmioDevice for E1000Model {
        fn read(&mut self, addr: u64, _w: u8) -> u64 {
            match addr - self.base {
                e1000::TDH => self.tdh,
                e1000::RAL => self.ral as u64,
                e1000::RAH => self.rah as u64,
                e1000::STATUS => 0x8003, // link up, full duplex
                _ => 0,
            }
        }
        fn write(&mut self, addr: u64, _w: u8, value: u64) {
            match addr - self.base {
                e1000::TDBAL => self.tdbal = value,
                e1000::TDH => self.tdh = value,
                e1000::TDT => {
                    // Process descriptors in [head, tail). tail==head means the ring
                    // is empty (as in init's TDT=0) — do nothing, exactly like real HW.
                    if value != self.tdh {
                        let desc_phys = self.tdbal + self.tdh * e1000::DESC_SIZE;
                        let desc = self.dma.read_at(desc_phys, 16);
                        let addr = u64::from_le_bytes(desc[0..8].try_into().unwrap());
                        let len = u16::from_le_bytes(desc[8..10].try_into().unwrap()) as usize;
                        self.last_tx = Some(self.dma.read_at(addr, len));
                        self.tdh = value; // head catches up to tail → "transmitted"
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn e1000_spec_is_well_formed_and_binds() {
        let tags = SoftwareTags::new([4u8; 32]);
        let mut dma = crate::driver::ModelDmaMem::new();
        let spec = e1000_spec(0xFEBF_0000, 11);
        assert!(spec.is_well_formed());
        assert!(Driver::bind_dma(spec, &tags, &mut dma).is_ok());
    }

    #[test]
    fn e1000_builds_a_descriptor_and_transmits_through_the_spec() {
        let tags = SoftwareTags::new([4u8; 32]);
        let mut dma = SharedDma::new();
        let driver = Driver::bind_dma(e1000_spec(0xFEBF_0000, 11), &tags, &mut dma).unwrap();
        let mut nic = E1000Model::new(0xFEBF_0000, dma.clone());

        driver.run_io("init", &[], &[], &mut nic, &mut dma, &tags).unwrap();
        let frame = arp_frame();
        driver
            .run_io("tx", &[frame.len() as u64], &frame, &mut nic, &mut dma, &tags)
            .unwrap();
        // The descriptor the spec built drove a real DMA of exactly our frame.
        assert_eq!(nic.last_tx.as_deref(), Some(frame.as_slice()));
    }

    #[test]
    fn a_borrowed_nic_spec_cannot_dma_outside_its_buffer() {
        // Try to transmit a frame larger than the declared TX buffer: the runtime
        // traps instead of letting the device DMA out of bounds.
        let tags = SoftwareTags::new([3u8; 32]);
        let mut dma = SharedDma::new();
        let driver = Driver::bind_dma(rtl8139_spec(0xFEBC_0000, 11), &tags, &mut dma).unwrap();
        let mut nic = Rtl8139Model::new(0xFEBC_0000, dma.clone());
        driver.run_io("init", &[], &[], &mut nic, &mut dma, &tags).unwrap();
        let huge = vec![0u8; 4096]; // tx0 is only 2048 bytes
        let r = driver.run_io("tx", &[huge.len() as u64], &huge, &mut nic, &mut dma, &tags);
        assert_eq!(r.err(), Some(crate::driver::DriverFault::OutOfBounds));
        assert!(nic.last_tx.is_none()); // nothing was transmitted
    }
}

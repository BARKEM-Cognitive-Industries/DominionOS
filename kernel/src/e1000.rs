//! Intel Gigabit Ethernet — a *real* **e1000 / e1000e** driver on bare metal.
//!
//! This is the hardware NIC every hypervisor and a huge slice of real PCs ship:
//! QEMU's and VirtualBox's default `e1000` (Intel 82540EM), the 82574L (`e1000e`),
//! and the I21x/I217 family found on modern desktops and laptops. Unlike the
//! virtio path this is a physical MAC — we map its MMIO register file out of
//! BAR0, read the station address straight out of the Receive-Address registers
//! (falling back to a serial EEPROM read on hardware that leaves them clear),
//! reset the controller, bring the link up, and stand up DMA descriptor rings for
//! RX and TX.
//!
//! The protocol logic — Ethernet/ARP/IPv4/ICMP/UDP/DHCP — lives in the pure,
//! host-tested [`dominion_core::net`]; this driver only moves bytes between that
//! stack and the wire. It plugs into the same [`crate::netif::Nic`] abstraction as
//! virtio-net and RTL8139, so DHCP/ARP/ICMP/UDP run over it unchanged.

use crate::dma::{self, DmaRegion};
use crate::netif::Nic;
use crate::pci;
use alloc::vec;
use alloc::vec::Vec;
use core::ptr::{copy_nonoverlapping, read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};
use dominion_core::net::MacAddr;

/// Intel's PCI vendor id.
const INTEL_VENDOR: u16 = 0x8086;

/// PCI device ids for the e1000/e1000e family we drive. Kept small and explicit
/// (rather than matching the whole class) so we only claim controllers this
/// register layout actually fits.
const E1000_IDS: &[u16] = &[
    0x100E, // 82540EM      — QEMU / VirtualBox default "e1000"
    0x100F, // 82545EM      — VMware default
    0x10D3, // 82574L       — "e1000e"
    0x153A, // I217-LM      — common on modern desktops
    0x1539, // I211         — common gigabit controller
];

// ---------------------------------------------------------------------------
// MMIO register offsets (bytes from BAR0). Intel PCIe GbE Software Developer's
// Manual, §13.
// ---------------------------------------------------------------------------
const REG_CTRL: u64 = 0x0000; // Device Control
const REG_STATUS: u64 = 0x0008; // Device Status
const REG_EERD: u64 = 0x0014; // EEPROM Read
const REG_ICR: u64 = 0x00C0; // Interrupt Cause Read
const REG_IMC: u64 = 0x00D8; // Interrupt Mask Clear
const REG_RCTL: u64 = 0x0100; // Receive Control
const REG_TCTL: u64 = 0x0400; // Transmit Control
const REG_TIPG: u64 = 0x0410; // Transmit Inter-Packet Gap
const REG_RDBAL: u64 = 0x2800; // RX Descriptor Base Low
const REG_RDBAH: u64 = 0x2804; // RX Descriptor Base High
const REG_RDLEN: u64 = 0x2808; // RX Descriptor Length
const REG_RDH: u64 = 0x2810; // RX Descriptor Head
const REG_RDT: u64 = 0x2818; // RX Descriptor Tail
const REG_TDBAL: u64 = 0x3800; // TX Descriptor Base Low
const REG_TDBAH: u64 = 0x3804; // TX Descriptor Base High
const REG_TDLEN: u64 = 0x3808; // TX Descriptor Length
const REG_TDH: u64 = 0x3810; // TX Descriptor Head
const REG_TDT: u64 = 0x3818; // TX Descriptor Tail
const REG_MTA: u64 = 0x5200; // Multicast Table Array (128 dwords)
const REG_RAL0: u64 = 0x5400; // Receive Address Low (entry 0)
const REG_RAH0: u64 = 0x5404; // Receive Address High (entry 0)

// CTRL bits.
const CTRL_LRST: u32 = 1 << 3; // link reset
const CTRL_ASDE: u32 = 1 << 5; // auto-speed detection enable
const CTRL_SLU: u32 = 1 << 6; // set link up
const CTRL_RST: u32 = 1 << 26; // software reset
const CTRL_PHY_RST: u32 = 1 << 31; // PHY reset

// RCTL bits.
const RCTL_EN: u32 = 1 << 1; // receiver enable
const RCTL_BAM: u32 = 1 << 15; // broadcast accept mode
const RCTL_BSIZE_2048: u32 = 0 << 16; // buffer size 2048 (BSEX=0)
const RCTL_SECRC: u32 = 1 << 26; // strip Ethernet CRC

// TCTL bits.
const TCTL_EN: u32 = 1 << 1; // transmitter enable
const TCTL_PSP: u32 = 1 << 3; // pad short packets
const TCTL_CT_SHIFT: u32 = 4; // collision threshold
const TCTL_COLD_SHIFT: u32 = 12; // collision distance

// EEPROM Read (EERD) bits — 82540/82545/82546 "legacy" layout.
const EERD_START: u32 = 1 << 0;
const EERD_DONE: u32 = 1 << 4;

// TX descriptor command + status bits.
const TXD_CMD_EOP: u8 = 1 << 0; // end of packet
const TXD_CMD_IFCS: u8 = 1 << 1; // insert FCS/CRC
const TXD_CMD_RS: u8 = 1 << 3; // report status (sets DD on completion)
const TXD_STAT_DD: u8 = 1 << 0; // descriptor done

// RX descriptor status bits.
const RXD_STAT_DD: u8 = 1 << 0; // descriptor done

/// Descriptor-ring geometry. Counts must keep each ring a multiple of 128 bytes
/// (8 descriptors) as the hardware requires; 2048-byte buffers hold a full
/// 1522-byte Ethernet frame with room to spare.
const RX_DESCS: usize = 32;
const TX_DESCS: usize = 8;
const BUF_SIZE: usize = 2048;

/// Fail-fast bound on hardware-ready spin loops (a healthy controller asserts in
/// microseconds; each iteration is a slow uncached MMIO read). Mirrors nvme.rs.
const SPIN: u32 = 1_000_000;

/// Legacy receive descriptor (16 bytes). The device writes `length`/`status`
/// after DMA'ing a frame into `addr`.
#[repr(C)]
struct RxDesc {
    addr: u64,
    length: u16,
    checksum: u16,
    status: u8,
    errors: u8,
    special: u16,
}

/// Legacy transmit descriptor (16 bytes). We fill `addr`/`length`/`cmd`; the
/// device writes back `status` (DD) once the frame is on the wire.
#[repr(C)]
struct TxDesc {
    addr: u64,
    length: u16,
    cso: u8,
    cmd: u8,
    status: u8,
    css: u8,
    special: u16,
}

#[inline]
unsafe fn r32(a: u64) -> u32 {
    read_volatile(a as *const u32)
}
#[inline]
unsafe fn w32(a: u64, v: u32) {
    write_volatile(a as *mut u32, v);
}

/// Spin roughly `n` MMIO-read intervals — a crude delay used between the reset
/// and the register writes that follow it, since we have no fine-grained timer in
/// the driver. Reading STATUS is a real uncached bus cycle, so this genuinely
/// stalls rather than being optimised away.
fn delay(mmio: u64, n: u32) {
    for _ in 0..n {
        unsafe {
            let _ = r32(mmio + REG_STATUS);
        }
    }
}

/// The live Intel Gigabit NIC.
pub struct E1000 {
    mmio: u64,
    mac: MacAddr,
    rx_ring: DmaRegion,
    tx_ring: DmaRegion,
    rx_bufs: DmaRegion,
    tx_bufs: DmaRegion,
    /// Next RX descriptor the driver will inspect for a completed frame.
    rx_cur: usize,
    /// Next TX descriptor the driver will fill.
    tx_cur: usize,
}

/// Locate the first attached e1000-family controller on the PCI bus.
fn find_device() -> Option<pci::PciDevice> {
    pci::enumerate()
        .into_iter()
        .find(|d| d.vendor_id == INTEL_VENDOR && E1000_IDS.contains(&d.device_id))
}

/// Resolve BAR0 to a physical MMIO base, handling the 64-bit-BAR encoding (bits
/// 2:1 == 0b10 ⇒ the high dword lives in BAR1).
fn bar0_phys(dev: &pci::PciDevice) -> Option<u64> {
    let bar0 = dev.address.bar(0);
    let lo = (bar0 & 0xFFFF_FFF0) as u64;
    let phys = if bar0 & 0x6 == 0x4 {
        lo | ((dev.address.bar(1) as u64) << 32)
    } else {
        lo
    };
    if phys == 0 {
        None
    } else {
        Some(phys)
    }
}

/// Read one 16-bit EEPROM word through the EERD register (legacy 82540 layout:
/// address in bits 15:8, data in bits 31:16, DONE at bit 4). Returns 0 if the
/// device never reports DONE (e.g. no serial EEPROM present).
unsafe fn eeprom_read(mmio: u64, word_addr: u8) -> u16 {
    w32(mmio + REG_EERD, ((word_addr as u32) << 8) | EERD_START);
    let mut spins = 0;
    loop {
        let v = r32(mmio + REG_EERD);
        if v & EERD_DONE != 0 {
            return (v >> 16) as u16;
        }
        spins += 1;
        if spins >= SPIN {
            return 0;
        }
        core::hint::spin_loop();
    }
}

/// Read the station MAC. Preferred source is RAL0/RAH0 (the firmware/EEPROM has
/// already loaded them on QEMU and most hardware); if RAL0 reads back zero we
/// fall back to reading words 0..3 out of the serial EEPROM.
unsafe fn read_mac(mmio: u64) -> MacAddr {
    let ral = r32(mmio + REG_RAL0);
    let rah = r32(mmio + REG_RAH0);
    if ral != 0 {
        let mut m = [0u8; 6];
        m[0..4].copy_from_slice(&ral.to_le_bytes());
        m[4] = rah as u8;
        m[5] = (rah >> 8) as u8;
        return MacAddr(m);
    }
    let w0 = eeprom_read(mmio, 0);
    let w1 = eeprom_read(mmio, 1);
    let w2 = eeprom_read(mmio, 2);
    MacAddr([
        w0 as u8,
        (w0 >> 8) as u8,
        w1 as u8,
        (w1 >> 8) as u8,
        w2 as u8,
        (w2 >> 8) as u8,
    ])
}

impl E1000 {
    /// Probe PCI for an e1000-family controller and bring it fully up: reset,
    /// link up, MAC read, and RX/TX descriptor rings. Returns `None` if no such
    /// device is attached or DMA memory is exhausted (never panics or hangs).
    pub fn init() -> Option<E1000> {
        let dev = find_device()?;
        dev.address.enable_bus_master();
        let bar_phys = bar0_phys(&dev)?;
        // BAR0 is MMIO, not RAM, so it must be mapped before the first register
        // access or the read page-faults on real hardware. 128 KiB covers the
        // whole e1000 register file.
        let mmio = dma::map_mmio(bar_phys, 0x20000);

        unsafe {
            // Mask every interrupt — we poll the descriptor rings.
            w32(mmio + REG_IMC, 0xFFFF_FFFF);

            // Software reset, then wait for the RST bit to self-clear.
            let ctrl = r32(mmio + REG_CTRL);
            w32(mmio + REG_CTRL, ctrl | CTRL_RST);
            let mut spins = 0;
            while r32(mmio + REG_CTRL) & CTRL_RST != 0 {
                spins += 1;
                if spins >= SPIN {
                    return None;
                }
                core::hint::spin_loop();
            }
            delay(mmio, 1000);
            // Re-mask interrupts (reset re-enables some) and clear pending causes.
            w32(mmio + REG_IMC, 0xFFFF_FFFF);
            let _ = r32(mmio + REG_ICR);

            // Bring the link up: set-link-up + auto-speed detect, clear link/PHY
            // reset so the PHY negotiates.
            let ctrl = r32(mmio + REG_CTRL);
            let ctrl = (ctrl | CTRL_SLU | CTRL_ASDE) & !(CTRL_LRST | CTRL_PHY_RST);
            w32(mmio + REG_CTRL, ctrl);

            // Clear the 128-entry multicast table so no stale filter blocks RX.
            for i in 0..128u64 {
                w32(mmio + REG_MTA + i * 4, 0);
            }
        }

        let mac = unsafe { read_mac(mmio) };

        // ---- RX ring ----
        let rx_ring = dma::alloc(1)?; // 4 KiB page holds 32 × 16-byte descriptors
        let rx_bufs = dma::alloc((RX_DESCS * BUF_SIZE).div_ceil(4096))?;
        for i in 0..RX_DESCS {
            unsafe {
                let d = (rx_ring.virt as *mut RxDesc).add(i);
                (*d).addr = rx_bufs.phys + (i * BUF_SIZE) as u64;
                (*d).status = 0;
            }
        }
        unsafe {
            w32(mmio + REG_RDBAL, rx_ring.phys as u32);
            w32(mmio + REG_RDBAH, (rx_ring.phys >> 32) as u32);
            w32(mmio + REG_RDLEN, (RX_DESCS * 16) as u32);
            w32(mmio + REG_RDH, 0);
            // Tail = last descriptor owned by hardware ⇒ all but the head are free.
            w32(mmio + REG_RDT, (RX_DESCS - 1) as u32);
            w32(
                mmio + REG_RCTL,
                RCTL_EN | RCTL_BAM | RCTL_SECRC | RCTL_BSIZE_2048,
            );
        }

        // ---- TX ring ----
        let tx_ring = dma::alloc(1)?;
        let tx_bufs = dma::alloc((TX_DESCS * BUF_SIZE).div_ceil(4096))?;
        for i in 0..TX_DESCS {
            unsafe {
                let d = (tx_ring.virt as *mut TxDesc).add(i);
                (*d).addr = tx_bufs.phys + (i * BUF_SIZE) as u64;
                (*d).cmd = 0;
                // Pre-mark DD so the first transmit doesn't wait on a stale slot.
                (*d).status = TXD_STAT_DD;
            }
        }
        unsafe {
            w32(mmio + REG_TDBAL, tx_ring.phys as u32);
            w32(mmio + REG_TDBAH, (tx_ring.phys >> 32) as u32);
            w32(mmio + REG_TDLEN, (TX_DESCS * 16) as u32);
            w32(mmio + REG_TDH, 0);
            w32(mmio + REG_TDT, 0);
            w32(
                mmio + REG_TCTL,
                TCTL_EN | TCTL_PSP | (0x0F << TCTL_CT_SHIFT) | (0x40 << TCTL_COLD_SHIFT),
            );
            // Standard IEEE 802.3 inter-packet gap (IPGT=10, IPGR1=8, IPGR2=6).
            w32(mmio + REG_TIPG, 0x0060_200A);
        }

        Some(E1000 {
            mmio,
            mac,
            rx_ring,
            tx_ring,
            rx_bufs,
            tx_bufs,
            rx_cur: 0,
            tx_cur: 0,
        })
    }

    pub fn mac(&self) -> MacAddr {
        self.mac
    }

    /// Transmit one Ethernet frame: copy it into the next TX buffer, arm the
    /// descriptor with EOP|IFCS|RS, bump TDT, and wait (bounded) for the DD
    /// write-back. Returns `false` on an oversized frame or a timeout.
    pub fn transmit(&mut self, frame: &[u8]) -> bool {
        if frame.is_empty() || frame.len() > BUF_SIZE {
            return false;
        }
        let i = self.tx_cur;
        unsafe {
            let dst = (self.tx_bufs.virt + (i * BUF_SIZE) as u64) as *mut u8;
            copy_nonoverlapping(frame.as_ptr(), dst, frame.len());
            let d = (self.tx_ring.virt as *mut TxDesc).add(i);
            write_volatile(&mut (*d).length, frame.len() as u16);
            write_volatile(&mut (*d).cso, 0);
            write_volatile(&mut (*d).cmd, TXD_CMD_EOP | TXD_CMD_IFCS | TXD_CMD_RS);
            write_volatile(&mut (*d).status, 0);
        }
        self.tx_cur = (i + 1) % TX_DESCS;
        // Publish the descriptor stores before the device reads them off the tail.
        fence(Ordering::SeqCst);
        unsafe { w32(self.mmio + REG_TDT, self.tx_cur as u32) };

        let mut spins = 0;
        loop {
            fence(Ordering::SeqCst);
            let st = unsafe { read_volatile(&(*(self.tx_ring.virt as *const TxDesc).add(i)).status) };
            if st & TXD_STAT_DD != 0 {
                return true;
            }
            spins += 1;
            if spins >= SPIN {
                return false;
            }
            core::hint::spin_loop();
        }
    }

    /// Non-blocking receive: if the device has completed the descriptor at our
    /// cursor, copy the frame out, hand the buffer back (RDT), and advance.
    pub fn poll_frame(&mut self) -> Option<Vec<u8>> {
        let i = self.rx_cur;
        fence(Ordering::SeqCst);
        let (status, len) = unsafe {
            let d = (self.rx_ring.virt as *const RxDesc).add(i);
            (read_volatile(&(*d).status), read_volatile(&(*d).length))
        };
        if status & RXD_STAT_DD == 0 {
            return None;
        }
        // Clamp to the buffer capacity so a bogus length can't drive an
        // out-of-bounds copy past this 2 KiB slot into adjacent DMA memory.
        let total = (len as usize).min(BUF_SIZE);
        let mut frame = vec![0u8; total];
        unsafe {
            let src = (self.rx_bufs.virt + (i * BUF_SIZE) as u64) as *const u8;
            copy_nonoverlapping(src, frame.as_mut_ptr(), total);
            // Clear DD and hand the descriptor back to hardware.
            let d = (self.rx_ring.virt as *mut RxDesc).add(i);
            write_volatile(&mut (*d).status, 0);
        }
        // Moving the tail onto the just-consumed index re-arms it for the device.
        unsafe { w32(self.mmio + REG_RDT, i as u32) };
        self.rx_cur = (i + 1) % RX_DESCS;
        Some(frame)
    }
}

impl Nic for E1000 {
    fn mac(&self) -> MacAddr {
        self.mac
    }
    fn transmit(&mut self, frame: &[u8]) -> bool {
        E1000::transmit(self, frame)
    }
    fn poll_frame(&mut self) -> Option<Vec<u8>> {
        E1000::poll_frame(self)
    }
}

/// Non-destructive self-test result: the device id we matched and the MAC read
/// out of the Receive-Address registers.
pub struct E1000Probe {
    pub device_id: u16,
    pub mac: [u8; 6],
}

/// Presence + MAC probe for the self-test battery. Scans PCI for an e1000-family
/// controller and reads its MAC out of RAL0/RAH0 **without** resetting the device
/// or touching the descriptor rings, so it is safe to call even when this NIC is
/// already the live global interface. Returns `None` if none is attached.
pub fn probe() -> Option<E1000Probe> {
    let dev = find_device()?;
    dev.address.enable_bus_master();
    let bar_phys = bar0_phys(&dev)?;
    let mmio = dma::map_mmio(bar_phys, 0x20000);
    let mac = unsafe { read_mac(mmio) };
    Some(E1000Probe {
        device_id: dev.device_id,
        mac: mac.0,
    })
}

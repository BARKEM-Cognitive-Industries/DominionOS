//! Spec-driven **RTL8139** driver — a *real* NIC driven on QEMU entirely from
//! `dominion_core::netspec::rtl8139_spec`, with **zero device-specific control code**
//! in the kernel. The declarative `DeviceSpec` runs on the capability-bounded
//! `dominion_core::driver` runtime; this module supplies only the two HAL backends it
//! needs on metal — a port-I/O MMIO view and the kernel DMA allocator — and finds
//! the device on the PCI bus.
//!
//! This is the end-to-end proof of the data-defined driver model: the same spec the
//! host tests drive against a model NIC here drives a live RealTek 8139 (the NIC
//! Windows and Linux both ship drivers for), bringing it up, reading its MAC out of
//! real hardware registers, and transmitting a real Ethernet frame.

use crate::dma;
use crate::pci;
use dominion_core::cheri::SoftwareTags;
use dominion_core::driver::{DmaMem, Driver, MmioDevice};
use dominion_core::netspec::rtl8139_spec;
use alloc::vec;
use alloc::vec::Vec;
use x86_64::instructions::port::Port;

const RTL_VENDOR: u16 = 0x10EC;
const RTL_DEVICE: u16 = 0x8139;

/// Port-I/O backend: the RTL8139's BAR0 is an I/O window, so every register access
/// is an in/out at `io_base + offset`. The capability runtime has already bounded
/// `addr` to the device window before any access reaches here.
struct PortNic;
impl MmioDevice for PortNic {
    fn read(&mut self, addr: u64, width: u8) -> u64 {
        let port = addr as u16;
        unsafe {
            match width {
                1 => Port::<u8>::new(port).read() as u64,
                2 => Port::<u16>::new(port).read() as u64,
                _ => Port::<u32>::new(port).read() as u64,
            }
        }
    }
    fn write(&mut self, addr: u64, width: u8, value: u64) {
        let port = addr as u16;
        unsafe {
            match width {
                1 => Port::<u8>::new(port).write(value as u8),
                2 => Port::<u16>::new(port).write(value as u16),
                _ => Port::<u32>::new(port).write(value as u32),
            }
        }
    }
}

/// Kernel DMA backend: allocate from the global DMA facility and address by physical
/// address (what the device sees), dereferencing through the phys-offset map.
struct KernelDma;
impl DmaMem for KernelDma {
    fn alloc(&mut self, len: u64) -> Option<u64> {
        let pages = ((len + 4095) / 4096) as usize;
        dma::alloc(pages).map(|r| r.phys)
    }
    fn write(&mut self, phys: u64, data: &[u8]) -> bool {
        let virt = dma::phys_to_virt(phys) as *mut u8;
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), virt, data.len()) };
        true
    }
    fn read(&mut self, phys: u64, len: u64) -> Vec<u8> {
        let virt = dma::phys_to_virt(phys) as *const u8;
        let mut out = vec![0u8; len as usize];
        unsafe { core::ptr::copy_nonoverlapping(virt, out.as_mut_ptr(), len as usize) };
        out
    }
}

/// What the spec-driven bring-up observed on the real device.
pub struct Rtl8139Report {
    pub mac: [u8; 6],
    pub tx_ok: bool,
    pub rx_len: usize,
}

/// Build a minimal ARP request for 10.0.2.2 (the QEMU SLIRP gateway), padded to the
/// 60-byte Ethernet minimum.
fn arp_request(src_mac: [u8; 6]) -> Vec<u8> {
    let mut f = Vec::with_capacity(60);
    f.extend_from_slice(&[0xFF; 6]); // dst: broadcast
    f.extend_from_slice(&src_mac); // src
    f.extend_from_slice(&[0x08, 0x06]); // ethertype ARP
    f.extend_from_slice(&[0x00, 0x01]); // htype Ethernet
    f.extend_from_slice(&[0x08, 0x00]); // ptype IPv4
    f.push(6); // hlen
    f.push(4); // plen
    f.extend_from_slice(&[0x00, 0x01]); // oper request
    f.extend_from_slice(&src_mac); // sender hw
    f.extend_from_slice(&[10, 0, 2, 15]); // sender ip (SLIRP guest)
    f.extend_from_slice(&[0; 6]); // target hw
    f.extend_from_slice(&[10, 0, 2, 2]); // target ip (SLIRP gateway)
    f.resize(60, 0); // pad to minimum frame
    f
}

/// Find an RTL8139 and drive it entirely from the declarative spec: init, read the
/// MAC, transmit an ARP request, and best-effort poll for a reply. Returns `None`
/// if no RTL8139 is attached.
pub fn probe_and_demo() -> Option<Rtl8139Report> {
    let dev = match pci::enumerate()
        .into_iter()
        .find(|d| d.vendor_id == RTL_VENDOR && d.device_id == RTL_DEVICE)
    {
        Some(d) => d,
        None => {
            crate::serial_println!("[rtl8139] no 10EC:8139 device found on the PCI bus");
            return None;
        }
    };
    dev.address.enable_bus_master();
    let io_base = (dev.address.bar(0) & 0xFFFC) as u64;
    let irq = dev.address.read_u8(0x3C) as u32;
    crate::serial_println!("[rtl8139] found at io_base={:#x} irq={}", io_base, irq);
    // Direct sanity read: IDR0 (MAC low 4) + CMD, to confirm port I/O is live.
    unsafe {
        let idr0 = Port::<u32>::new(io_base as u16).read();
        let cmd = Port::<u8>::new((io_base + 0x37) as u16).read();
        crate::serial_println!("[rtl8139] direct IDR0={:#010x} CMD={:#04x}", idr0, cmd);
    }

    let tags = SoftwareTags::new([0xA5u8; 32]);
    let mut dmem = KernelDma;
    let driver = match Driver::bind_dma(rtl8139_spec(io_base, irq), &tags, &mut dmem) {
        Ok(d) => d,
        Err(e) => {
            crate::serial_println!("[rtl8139] bind_dma failed: {:?}", e);
            return None;
        }
    };
    let mut nic = PortNic;

    // Bring the NIC up purely from the spec.
    if let Err(e) = driver.run_io("init", &[], &[], &mut nic, &mut dmem, &tags) {
        crate::serial_println!("[rtl8139] init failed: {:?}", e);
        return None;
    }

    // Read the MAC out of real hardware registers (IDR0 low 4 + IDR4 high 2).
    let mac_io = match driver.run_io("read_mac", &[], &[], &mut nic, &mut dmem, &tags) {
        Ok(io) => io,
        Err(e) => {
            crate::serial_println!("[rtl8139] read_mac failed: {:?}", e);
            return None;
        }
    };
    let low = *mac_io.regs.first().unwrap_or(&0) as u32;
    let high = *mac_io.regs.get(1).unwrap_or(&0) as u16;
    let mut mac = [0u8; 6];
    mac[0..4].copy_from_slice(&low.to_le_bytes());
    mac[4..6].copy_from_slice(&high.to_le_bytes());

    // Transmit a real ARP request through the spec's DMA TX path.
    let frame = arp_request(mac);
    let tx_res = driver.run_io("tx", &[frame.len() as u64], &frame, &mut nic, &mut dmem, &tags);
    if let Err(e) = &tx_res {
        crate::serial_println!("[rtl8139] tx failed: {:?}", e);
    }
    let tx_ok = tx_res.is_ok();
    crate::serial_println!(
        "[rtl8139] mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} tx_ok={}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], tx_ok
    );

    // Best-effort: poll a bounded number of times for a received reply.
    let mut rx_len = 0;
    for _ in 0..2000 {
        if driver.run_io("rx_ready", &[], &[], &mut nic, &mut dmem, &tags).is_ok() {
            if let Ok(io) = driver.run_io("rx", &[], &[], &mut nic, &mut dmem, &tags) {
                // rtl8139 packet header: [u16 status][u16 length].
                let len = u16::from_le_bytes([io.bytes.get(2).copied().unwrap_or(0), io.bytes.get(3).copied().unwrap_or(0)]) as usize;
                rx_len = len;
            }
            break;
        }
    }

    Some(Rtl8139Report { mac, tx_ok, rx_len })
}

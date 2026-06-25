//! PCI bus enumeration (the M3 driver framework's discovery layer).
//!
//! "With hardware abstracted into uniform capability descriptors, traditional
//! device drivers vanish. A driver becomes a sandboxed execution module." Before
//! we can sandbox a driver we must *find* its device. This module performs real
//! PCI configuration-space access through the legacy I/O ports (`0xCF8` address,
//! `0xCFC` data) and brute-force enumerates every bus/device/function — exactly
//! what QEMU's i440fx chipset exposes.

use alloc::vec::Vec;
use spin::Mutex;
use x86_64::instructions::port::Port;

const CONFIG_ADDRESS: u16 = 0xCF8;
const CONFIG_DATA: u16 = 0xCFC;

/// PCI config access must be atomic (write address, then read/write data), so all
/// access is serialised through this lock.
static PCI_LOCK: Mutex<()> = Mutex::new(());

/// The well-known PCI vendor id for all virtio devices.
pub const VIRTIO_VENDOR: u16 = 0x1AF4;

/// A `(bus, device, function)` coordinate in PCI configuration space.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PciAddress {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciAddress {
    pub const fn new(bus: u8, device: u8, function: u8) -> Self {
        PciAddress { bus, device, function }
    }

    fn config_address(self, offset: u8) -> u32 {
        0x8000_0000
            | ((self.bus as u32) << 16)
            | ((self.device as u32) << 11)
            | ((self.function as u32) << 8)
            | ((offset as u32) & 0xFC)
    }

    /// Read a 32-bit config register (offset must be dword-aligned in effect).
    pub fn read_u32(self, offset: u8) -> u32 {
        let _g = PCI_LOCK.lock();
        unsafe {
            let mut addr = Port::<u32>::new(CONFIG_ADDRESS);
            let mut data = Port::<u32>::new(CONFIG_DATA);
            addr.write(self.config_address(offset));
            data.read()
        }
    }

    pub fn write_u32(self, offset: u8, value: u32) {
        let _g = PCI_LOCK.lock();
        unsafe {
            let mut addr = Port::<u32>::new(CONFIG_ADDRESS);
            let mut data = Port::<u32>::new(CONFIG_DATA);
            addr.write(self.config_address(offset));
            data.write(value);
        }
    }

    pub fn read_u16(self, offset: u8) -> u16 {
        let dword = self.read_u32(offset & 0xFC);
        (dword >> ((offset & 2) * 8)) as u16
    }

    pub fn read_u8(self, offset: u8) -> u8 {
        let dword = self.read_u32(offset & 0xFC);
        (dword >> ((offset & 3) * 8)) as u8
    }

    pub fn vendor_id(self) -> u16 {
        self.read_u16(0x00)
    }
    pub fn device_id(self) -> u16 {
        self.read_u16(0x02)
    }
    pub fn header_type(self) -> u8 {
        self.read_u8(0x0E)
    }
    pub fn class_code(self) -> u8 {
        self.read_u8(0x0B)
    }
    pub fn subclass(self) -> u8 {
        self.read_u8(0x0A)
    }
    /// Programming interface byte (offset 0x09) — distinguishes e.g. xHCI (0x30) from
    /// EHCI/OHCI/UHCI within the USB-controller subclass, or AHCI within storage.
    pub fn prog_if(self) -> u8 {
        self.read_u8(0x09)
    }

    /// Base Address Register `index` (0..=5).
    pub fn bar(self, index: u8) -> u32 {
        self.read_u32(0x10 + index * 4)
    }

    /// Enable I/O space, memory space and bus-mastering in the command register —
    /// required before a device will DMA.
    pub fn enable_bus_master(self) {
        // Command (low 16 bits) and status (high 16 bits) share dword 0x04, so a
        // single read fetches both — previously this issued two separate config
        // reads of the same register.
        let reg = self.read_u32(0x04);
        let command = (reg & 0xFFFF)
            | 0x1 /* I/O space */ | 0x2 /* memory space */ | 0x4 /* bus master */;
        // Preserve the status word in the upper 16 bits.
        let status = reg & 0xFFFF_0000;
        self.write_u32(0x04, status | (command & 0xFFFF));
    }
}

/// A discovered PCI function.
#[derive(Clone, Copy, Debug)]
pub struct PciDevice {
    pub address: PciAddress,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
}

impl PciDevice {
    /// True for any virtio device (vendor 0x1AF4).
    pub fn is_virtio(&self) -> bool {
        self.vendor_id == VIRTIO_VENDOR
    }
}

fn probe(address: PciAddress) -> Option<PciDevice> {
    // Identification, class and revision live in the first four config dwords
    // (offsets 0x00, 0x04, 0x08, 0x0C). The vendor/device pair shares dword 0x00
    // and the class/subclass share dword 0x08, so two 32-bit config reads cover
    // every field we keep — versus four separate read_u16/read_u8 calls, each of
    // which previously issued its own address-write + data-read port pair.
    let id = address.read_u32(0x00);
    let vendor_id = id as u16;
    if vendor_id == 0xFFFF {
        return None; // no device responds here
    }
    let class = address.read_u32(0x08);
    Some(PciDevice {
        address,
        vendor_id,
        device_id: (id >> 16) as u16,
        class_code: (class >> 24) as u8, // offset 0x0B
        subclass: (class >> 16) as u8,   // offset 0x0A
    })
}

/// Cached result of the first full bus scan. PCI topology is fixed for the life
/// of the machine (no hot-plug on QEMU's i440fx), so every enumeration after the
/// first is served from this snapshot instead of re-issuing thousands of
/// CONFIG_ADDRESS/CONFIG_DATA port cycles.
static PCI_CACHE: Mutex<Option<Vec<PciDevice>>> = Mutex::new(None);

/// Brute-force scan of every bus/device/function. QEMU's i440fx puts everything
/// on bus 0, but scanning the full space is simple and robust. The result is
/// cached after the first call (see [`PCI_CACHE`]).
pub fn enumerate() -> Vec<PciDevice> {
    if let Some(cached) = PCI_CACHE.lock().as_ref() {
        return cached.clone();
    }
    let found = scan_bus();
    *PCI_CACHE.lock() = Some(found.clone());
    found
}

/// The actual hardware scan (uncached). Kept separate so the cache logic stays
/// readable.
fn scan_bus() -> Vec<PciDevice> {
    let mut found = Vec::new();
    for bus in 0u8..=255 {
        for device in 0u8..32 {
            let base = PciAddress::new(bus, device, 0);
            // probe() reads dword 0x00 first; reuse it so we don't pay an extra
            // vendor_id() port pair just to test for presence.
            let Some(dev0) = probe(base) else { continue };
            // Multi-function devices set bit 7 of the header type (offset 0x0E,
            // in dword 0x0C).
            let multifunction = base.read_u8(0x0E) & 0x80 != 0;
            found.push(dev0);
            if multifunction {
                for function in 1..8 {
                    if let Some(dev) = probe(PciAddress::new(bus, device, function)) {
                        found.push(dev);
                    }
                }
            }
        }
    }
    found
}

/// All virtio devices on the machine.
pub fn virtio_devices() -> Vec<PciDevice> {
    enumerate().into_iter().filter(|d| d.is_virtio()).collect()
}

/// The first virtio device whose subsystem id matches `subsystem` (e.g. block=2,
/// net=1). Virtio's PCI *device id* for legacy/transitional devices is
/// `0x1000 + subsystem_id`; the subsystem id lives at config offset 0x2E.
pub fn find_virtio(subsystem: u16) -> Option<PciDevice> {
    enumerate().into_iter().find(|d| {
        d.is_virtio() && d.address.read_u16(0x2E) == subsystem
    })
}

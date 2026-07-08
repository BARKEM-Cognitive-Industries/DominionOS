//! USB-HID **boot-protocol** keyboard + mouse over xHCI.
//!
//! On a real PC the keyboard and mouse are USB HID devices, not PS/2 — so without this
//! the desktop has no pointer and (unless the firmware emulates PS/2) no keys. This
//! brings up the xHCI controller, enumerates *every* connected port (keyboard and mouse
//! are separate USB devices on separate slots), and for each HID device:
//!
//!   * sets it to **boot protocol** (`SET_PROTOCOL 0`) so reports have a fixed, known
//!     layout — no HID report-descriptor parsing needed;
//!   * configures its **interrupt IN** endpoint and queues a transfer for one report.
//!
//! [`poll`] (called from the shell + desktop loops, with no interrupts required) drains
//! completed interrupt transfers, decodes the boot reports, feeds decoded keys into
//! [`crate::keyboard`] and pointer motion into [`crate::mouse`], and re-arms each
//! endpoint. Polling-only and spin-bounded, exactly like the MSC driver in [`crate::xhci`].
//!
//! This is deliberately separate from `xhci.rs` (which is MSC/bulk): same controller,
//! different device class and a non-blocking poll model. The low-level ring/event
//! plumbing is duplicated rather than shared so neither driver can destabilise the other.

use crate::dma::{self, DmaRegion};
use crate::pci;
use alloc::vec::Vec;
use spin::Mutex;

const DEBUG: bool = true;
macro_rules! hlog {
    ($($arg:tt)*) => { if DEBUG { crate::serial_println!($($arg)*); } };
}

const SPIN: u32 = 1_000_000;
const RING_TRBS: usize = 256;

const OP_USBCMD: u64 = 0x00;
const OP_USBSTS: u64 = 0x04;
const OP_CRCR: u64 = 0x18;
const OP_DCBAAP: u64 = 0x30;
const OP_CONFIG: u64 = 0x38;
const OP_PORTSC_BASE: u64 = 0x400;

const USBCMD_RS: u32 = 1 << 0;
const USBCMD_HCRST: u32 = 1 << 1;
const USBSTS_HCH: u32 = 1 << 0;
const USBSTS_CNR: u32 = 1 << 11;

const PORTSC_CCS: u32 = 1 << 0;
const PORTSC_PED: u32 = 1 << 1;
const PORTSC_PR: u32 = 1 << 4;
const PORTSC_PP: u32 = 1 << 9;
const PORTSC_RW_MASK: u32 = PORTSC_PP;

const TRB_NORMAL: u32 = 1;
const TRB_SETUP: u32 = 2;
const TRB_DATA: u32 = 3;
const TRB_STATUS: u32 = 4;
const TRB_LINK: u32 = 6;
const TRB_ENABLE_SLOT: u32 = 9;
const TRB_ADDRESS_DEVICE: u32 = 11;
const TRB_CONFIGURE_ENDPOINT: u32 = 12;
const TRB_EVENT_TRANSFER: u32 = 32;
const TRB_EVENT_CMD_COMPLETE: u32 = 33;

const CC_SUCCESS: u32 = 1;
const CC_SHORT_PACKET: u32 = 13;

#[inline]
unsafe fn r32(a: u64) -> u32 {
    core::ptr::read_volatile(a as *const u32)
}
#[inline]
unsafe fn w32(a: u64, v: u32) {
    core::ptr::write_volatile(a as *mut u32, v);
}
#[inline]
unsafe fn w64(a: u64, v: u64) {
    core::ptr::write_volatile(a as *mut u64, v);
}

/// A producer ring (command / transfer) with a trailing wrapping Link TRB.
struct Ring {
    region: DmaRegion,
    enqueue: usize,
    pcs: bool,
}
impl Ring {
    fn new() -> Option<Ring> {
        let region = dma::alloc(1)?;
        let link = region.virt + ((RING_TRBS - 1) as u64) * 16;
        unsafe {
            w64(link, region.phys);
            w32(link + 8, 0);
            w32(link + 12, (TRB_LINK << 10) | (1 << 1));
        }
        Some(Ring { region, enqueue: 0, pcs: true })
    }
    fn phys_with_cycle(&self) -> u64 {
        self.region.phys | 1
    }
    fn push(&mut self, d0: u64, d2: u32, d3_no_cycle: u32) -> u64 {
        let slot = self.region.virt + (self.enqueue as u64) * 16;
        let slot_phys = self.region.phys + (self.enqueue as u64) * 16;
        unsafe {
            w64(slot, d0);
            w32(slot + 8, d2);
            w32(slot + 12, d3_no_cycle | (self.pcs as u32));
        }
        self.enqueue += 1;
        if self.enqueue == RING_TRBS - 1 {
            let link = self.region.virt + ((RING_TRBS - 1) as u64) * 16;
            unsafe { w32(link + 12, (TRB_LINK << 10) | (1 << 1) | (self.pcs as u32)) };
            self.pcs = !self.pcs;
            self.enqueue = 0;
        }
        slot_phys
    }
}

/// The controller's single event-ring segment; software polls the cycle bit.
struct EventRing {
    seg: DmaRegion,
    erst: DmaRegion,
    dequeue: usize,
    ccs: bool,
}
impl EventRing {
    fn new() -> Option<EventRing> {
        let seg = dma::alloc(1)?;
        let erst = dma::alloc(1)?;
        unsafe {
            w64(erst.virt, seg.phys);
            w32(erst.virt + 8, RING_TRBS as u32);
            w32(erst.virt + 12, 0);
        }
        Some(EventRing { seg, erst, dequeue: 0, ccs: true })
    }
    fn current_phys(&self) -> u64 {
        self.seg.phys + (self.dequeue as u64) * 16
    }
    /// Non-blocking: return the next event TRB if one is ready, else `None`.
    fn try_poll(&mut self) -> Option<(u64, u32, u32)> {
        let p = self.seg.virt + (self.dequeue as u64) * 16;
        let d3 = unsafe { r32(p + 12) };
        if (d3 & 1) == self.ccs as u32 {
            let d0 = unsafe { core::ptr::read_volatile(p as *const u64) };
            let d2 = unsafe { r32(p + 8) };
            self.dequeue += 1;
            if self.dequeue == RING_TRBS {
                self.dequeue = 0;
                self.ccs = !self.ccs;
            }
            Some((d0, d2, d3))
        } else {
            None
        }
    }
    /// Bounded blocking poll (setup-time only).
    fn poll(&mut self) -> Option<(u64, u32, u32)> {
        let mut spins = 0;
        loop {
            if let Some(e) = self.try_poll() {
                return Some(e);
            }
            spins += 1;
            if spins >= SPIN {
                return None;
            }
            core::hint::spin_loop();
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Keyboard,
    Mouse,
}

/// One configured HID device: its slot, interrupt-IN endpoint, transfer ring, and a
/// report bounce buffer plus the previous keyboard report (for key-edge detection).
struct HidDev {
    slot: u8,
    ep_dci: u32,
    in_ring: Ring,
    report: DmaRegion,
    report_len: u16,
    kind: Kind,
    last_kbd: [u8; 8],
    in_flight: bool,
}

struct Hid {
    op: u64,
    rt: u64,
    db: u64,
    ctx_size: u64,
    dcbaa: DmaRegion,
    cmd_ring: Ring,
    event: EventRing,
    devices: Vec<HidDev>,
}

static HID: Mutex<Option<Hid>> = Mutex::new(None);

impl Hid {
    fn ring_doorbell(&self, slot: u8, target: u32) {
        unsafe { w32(self.db + slot as u64 * 4, target) };
    }
    fn update_erdp(&self) {
        unsafe { w64(self.rt + 0x20 + 0x18, self.event.current_phys() | (1 << 3)) };
    }
    /// Bounded wait for an event of `want`; returns (completion_code, d3, trb_ptr).
    fn wait_event(&mut self, want: u32) -> Option<(u32, u32, u64)> {
        let mut total: u32 = 0;
        loop {
            total += 1;
            if total >= SPIN {
                return None;
            }
            let (d0, d2, d3) = match self.event.poll() {
                Some(e) => e,
                None => {
                    core::hint::spin_loop();
                    continue;
                }
            };
            self.update_erdp();
            if (d3 >> 10) & 0x3F == want {
                return Some(((d2 >> 24) & 0xFF, d3, d0));
            }
        }
    }

    fn control_in(
        &mut self,
        ep0: &mut Ring,
        slot: u8,
        bm: u8,
        req: u8,
        value: u16,
        index: u16,
        len: u16,
        data_phys: u64,
    ) -> Option<()> {
        let setup_lo = (bm as u64)
            | ((req as u64) << 8)
            | ((value as u64) << 16)
            | ((index as u64) << 32)
            | ((len as u64) << 48);
        ep0.push(setup_lo, 8, (TRB_SETUP << 10) | (1 << 6) | (3 << 16));
        if len > 0 {
            ep0.push(data_phys, len as u32, (TRB_DATA << 10) | (1 << 16));
        }
        ep0.push(0, 0, (TRB_STATUS << 10) | (1 << 5));
        self.ring_doorbell(slot, 1);
        let (cc, _, _) = self.wait_event(TRB_EVENT_TRANSFER)?;
        if cc == CC_SUCCESS || cc == CC_SHORT_PACKET {
            Some(())
        } else {
            None
        }
    }

    fn control_out(&mut self, ep0: &mut Ring, slot: u8, bm: u8, req: u8, value: u16, index: u16) -> Option<()> {
        let setup_lo =
            (bm as u64) | ((req as u64) << 8) | ((value as u64) << 16) | ((index as u64) << 32);
        ep0.push(setup_lo, 8, (TRB_SETUP << 10) | (1 << 6));
        ep0.push(0, 0, (TRB_STATUS << 10) | (1 << 16) | (1 << 5));
        self.ring_doorbell(slot, 1);
        let (cc, _, _) = self.wait_event(TRB_EVENT_TRANSFER)?;
        if cc == CC_SUCCESS {
            Some(())
        } else {
            None
        }
    }

    /// Enumerate and configure the HID device on `port` (1-based). Returns the device on
    /// success, or `None` if the port holds a non-HID device or setup fails.
    fn setup_device(&mut self, port: u64, speed: u32) -> Option<HidDev> {
        let ctx_size = self.ctx_size;
        // Enable a slot.
        self.cmd_ring.push(0, 0, TRB_ENABLE_SLOT << 10);
        self.ring_doorbell(0, 0);
        let (cc, d3, _) = self.wait_event(TRB_EVENT_CMD_COMPLETE)?;
        if cc != CC_SUCCESS {
            return None;
        }
        let slot = ((d3 >> 24) & 0xFF) as u8;
        if slot == 0 {
            return None;
        }

        let out_ctx = dma::alloc(1)?;
        let in_ctx = dma::alloc(1)?;
        unsafe { w64(self.dcbaa.virt + slot as u64 * 8, out_ctx.phys) };
        let mut ep0_ring = Ring::new()?;

        let mps0 = match speed {
            4 => 512,
            3 => 64,
            2 => 8,
            _ => 64,
        };
        // Input control context: add slot + EP0.
        unsafe {
            w32(in_ctx.virt + 0x04, 0b11);
            let slot_ctx = in_ctx.virt + ctx_size;
            w32(slot_ctx, (1u32 << 27) | (speed << 20));
            w32(slot_ctx + 4, (port as u32) << 16);
            let ep0_ctx = in_ctx.virt + ctx_size * 2;
            w32(ep0_ctx + 4, (4u32 << 3) | (3 << 1) | ((mps0 as u32) << 16));
            w64(ep0_ctx + 8, ep0_ring.phys_with_cycle());
        }
        self.cmd_ring.push(in_ctx.phys, 0, (TRB_ADDRESS_DEVICE << 10) | ((slot as u32) << 24));
        self.ring_doorbell(0, 0);
        let (cc, _, _) = self.wait_event(TRB_EVENT_CMD_COMPLETE)?;
        if cc != CC_SUCCESS {
            return None;
        }

        // Read the configuration descriptor (9 bytes, then full) to find a HID boot
        // interface and its interrupt IN endpoint.
        let desc = dma::alloc(1)?;
        self.control_in(&mut ep0_ring, slot, 0x80, 6, 0x0200, 0, 9, desc.phys)?;
        let total_len = u16::from_le_bytes(unsafe {
            [*((desc.virt + 2) as *const u8), *((desc.virt + 3) as *const u8)]
        });
        let want = (total_len as u32).min(desc.size as u32) as u16;
        self.control_in(&mut ep0_ring, slot, 0x80, 6, 0x0200, 0, want, desc.phys)?;
        let buf = unsafe { core::slice::from_raw_parts(desc.virt as *const u8, want as usize) };

        let cfg_value = buf.get(5).copied().unwrap_or(1);
        let mut kind: Option<Kind> = None;
        let mut iface_num = 0u8;
        let mut ep_addr = 0u8;
        let mut ep_mps = 8u16;
        let mut ep_interval = 8u8;
        let mut cur_is_hid = false;
        let mut i = 0usize;
        while i + 2 <= buf.len() {
            let len = buf[i] as usize;
            let dtype = buf[i + 1];
            if len == 0 {
                break;
            }
            if dtype == 0x04 && i + 9 <= buf.len() {
                // interface: class 0x03 = HID; boot subclass 1; protocol 1=kbd, 2=mouse
                let class = buf[i + 5];
                let subclass = buf[i + 6];
                let proto = buf[i + 7];
                cur_is_hid = class == 0x03;
                if cur_is_hid && subclass == 0x01 && (proto == 1 || proto == 2) && kind.is_none() {
                    kind = Some(if proto == 1 { Kind::Keyboard } else { Kind::Mouse });
                    iface_num = buf[i + 2];
                } else if cur_is_hid {
                    // a HID interface we don't drive (e.g. non-boot) — keep scanning
                } else {
                    cur_is_hid = false;
                }
            } else if dtype == 0x05 && cur_is_hid && kind.is_some() && ep_addr == 0 && i + 7 <= buf.len() {
                // endpoint: interrupt (attr 3) IN (addr bit7)
                let addr = buf[i + 2];
                let attr = buf[i + 3] & 0x3;
                if attr == 3 && addr & 0x80 != 0 {
                    ep_addr = addr & 0x0F;
                    ep_mps = u16::from_le_bytes([buf[i + 4], buf[i + 5]]);
                    ep_interval = buf[i + 6];
                }
            }
            i += len;
        }
        let kind = kind?;
        if ep_addr == 0 {
            return None;
        }
        let ep_dci = (ep_addr as u32) * 2 + 1; // IN endpoint DCI
        hlog!(
            "[usbhid] slot {} port {}: {} ep{} mps{} (boot protocol)",
            slot,
            port,
            if kind == Kind::Keyboard { "keyboard" } else { "mouse" },
            ep_addr,
            ep_mps
        );

        // SET_CONFIGURATION.
        self.control_out(&mut ep0_ring, slot, 0x00, 9, cfg_value as u16, 0)?;

        // Configure the interrupt IN endpoint (type 7).
        let in_ring = Ring::new()?;
        let interval = compute_interval(speed, ep_interval);
        unsafe {
            core::ptr::write_bytes(in_ctx.virt as *mut u8, 0, 4096);
            w32(in_ctx.virt + 0x04, (1u32 << ep_dci) | 1); // A0 (slot) + the EP
            let slot_ctx = in_ctx.virt + ctx_size;
            w32(slot_ctx, (ep_dci << 27) | (speed << 20));
            w32(slot_ctx + 4, (port as u32) << 16);
            let ep_ctx = in_ctx.virt + ctx_size * (1 + ep_dci as u64);
            // dword0: Interval in bits 16-23.
            w32(ep_ctx, interval << 16);
            // dword1: EP type 7 (interrupt IN), CErr=3, MaxPacketSize.
            w32(ep_ctx + 4, (7u32 << 3) | (3 << 1) | ((ep_mps as u32) << 16));
            w64(ep_ctx + 8, in_ring.phys_with_cycle());
        }
        self.cmd_ring.push(in_ctx.phys, 0, (TRB_CONFIGURE_ENDPOINT << 10) | ((slot as u32) << 24));
        self.ring_doorbell(0, 0);
        let (cc, _, _) = self.wait_event(TRB_EVENT_CMD_COMPLETE)?;
        if cc != CC_SUCCESS {
            hlog!("[usbhid] configure endpoint failed cc={}", cc);
            return None;
        }

        // Boot protocol + report-on-change idle. SET_PROTOCOL(boot=0); SET_IDLE(0).
        // (class request: bmRequestType 0x21 = host->dev, class, interface)
        let _ = self.control_out(&mut ep0_ring, slot, 0x21, 0x0B, 0, iface_num as u16); // SET_PROTOCOL boot
        let _ = self.control_out(&mut ep0_ring, slot, 0x21, 0x0A, 0, iface_num as u16); // SET_IDLE 0

        let report = dma::alloc(1)?;
        let report_len = ep_mps.clamp(1, 64);
        let mut d = HidDev {
            slot,
            ep_dci,
            in_ring,
            report,
            report_len,
            kind,
            last_kbd: [0; 8],
            in_flight: false,
        };
        // Arm the first interrupt transfer.
        arm(&mut d, self.db);
        Some(d)
    }
}

/// xHCI interrupt-endpoint Interval encoding (spec 6.2.3.6), approximated for boot HID.
fn compute_interval(speed: u32, b_interval: u8) -> u32 {
    let bi = (b_interval.max(1)) as u32;
    if speed >= 3 {
        // High/SuperSpeed: bInterval is already a 2^(n-1) microframe exponent.
        (bi - 1).min(15)
    } else {
        // Full/Low speed: period is `bInterval` ms; Interval = floor(log2(bInterval*8)).
        let frames = bi * 8;
        let mut n = 0u32;
        let mut v = frames;
        while v > 1 {
            v >>= 1;
            n += 1;
        }
        n.clamp(3, 10)
    }
}

/// Queue one interrupt-IN transfer for `d`'s report and ring its doorbell.
fn arm(d: &mut HidDev, db: u64) {
    d.in_ring.push(d.report.phys, d.report_len as u32, (TRB_NORMAL << 10) | (1 << 5));
    unsafe { w32(db + d.slot as u64 * 4, d.ep_dci) };
    d.in_flight = true;
}

fn bringup(mmio: u64) -> Option<Hid> {
    let caplen = (unsafe { r32(mmio) } & 0xFF) as u64;
    let hcsparams1 = unsafe { r32(mmio + 0x04) };
    let max_ports = ((hcsparams1 >> 24) & 0xFF) as u64;
    let max_slots = hcsparams1 & 0xFF;
    let hccparams1 = unsafe { r32(mmio + 0x10) };
    let ctx_size: u64 = if (hccparams1 >> 2) & 1 == 1 { 64 } else { 32 };
    let dboff = (unsafe { r32(mmio + 0x14) } & !0x3) as u64;
    let rtsoff = (unsafe { r32(mmio + 0x18) } & !0x1F) as u64;
    let op = mmio + caplen;
    let rt = mmio + rtsoff;
    let db = mmio + dboff;

    let mut spins = 0;
    while unsafe { r32(op + OP_USBSTS) } & USBSTS_CNR != 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
    }
    unsafe { w32(op + OP_USBCMD, 0) };
    spins = 0;
    while unsafe { r32(op + OP_USBSTS) } & USBSTS_HCH == 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
    }
    unsafe { w32(op + OP_USBCMD, USBCMD_HCRST) };
    spins = 0;
    while unsafe { r32(op + OP_USBCMD) } & USBCMD_HCRST != 0
        || unsafe { r32(op + OP_USBSTS) } & USBSTS_CNR != 0
    {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
    }

    unsafe { w32(op + OP_CONFIG, max_slots) };

    let dcbaa = dma::alloc(1)?;
    let hcsparams2 = unsafe { r32(mmio + 0x08) };
    let max_scratch = (((hcsparams2 >> 21) & 0x1F) << 5) | ((hcsparams2 >> 27) & 0x1F);
    if max_scratch > 0 {
        let sp_array = dma::alloc(1)?;
        for i in 0..max_scratch as u64 {
            let buf = dma::alloc(1)?;
            unsafe { w64(sp_array.virt + i * 8, buf.phys) };
        }
        unsafe { w64(dcbaa.virt, sp_array.phys) };
    }
    unsafe { w64(op + OP_DCBAAP, dcbaa.phys) };

    let cmd_ring = Ring::new()?;
    unsafe { w64(op + OP_CRCR, cmd_ring.phys_with_cycle()) };

    let event = EventRing::new()?;
    unsafe {
        w32(rt + 0x20 + 0x08, 1);
        w64(rt + 0x20 + 0x10, event.erst.phys);
        w64(rt + 0x20 + 0x18, event.seg.phys);
        // IMAN.IE left clear: we poll the event ring's cycle bit directly, so the
        // controller must NOT assert an interrupt line (there is no xHCI IRQ handler,
        // and an unacknowledged interrupt would storm on machines where IRQs work).
        w32(rt + 0x20 + 0x00, 0);
    }

    // Run WITHOUT USBCMD.INTE — pure polling (see above).
    unsafe { w32(op + OP_USBCMD, USBCMD_RS) };
    spins = 0;
    while unsafe { r32(op + OP_USBSTS) } & USBSTS_HCH != 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
    }
    hlog!("[usbhid] controller running, {} ports", max_ports);

    let mut hid = Hid { op, rt, db, ctx_size, dcbaa, cmd_ring, event, devices: Vec::new() };

    // Reset and enumerate EVERY connected port (kbd + mouse are distinct devices).
    for i in 0..max_ports {
        let psc = op + OP_PORTSC_BASE + i * 0x10;
        let v = unsafe { r32(psc) };
        if v & PORTSC_CCS == 0 {
            continue;
        }
        unsafe { w32(psc, (v & PORTSC_RW_MASK) | PORTSC_PP | PORTSC_PR) };
        let mut s = 0;
        while unsafe { r32(psc) } & PORTSC_PR != 0 && s < SPIN {
            s += 1;
        }
        let after = unsafe { r32(psc) };
        unsafe { w32(psc, after) }; // clear change bits
        if after & PORTSC_PED == 0 {
            continue;
        }
        let speed = (after >> 10) & 0xF;
        if let Some(d) = hid.setup_device(i + 1, speed) {
            hid.devices.push(d);
        }
    }
    Some(hid)
}

/// Probe + bring up USB-HID input. Returns true if at least one HID keyboard/mouse was
/// configured. Safe to call once at boot; absence or failure is non-fatal.
pub fn init() -> bool {
    if HID.lock().is_some() {
        return present();
    }
    let dev = pci::enumerate()
        .into_iter()
        .find(|d| d.class_code == 0x0C && d.subclass == 0x03 && d.address.prog_if() == 0x30);
    let dev = match dev {
        Some(d) => d,
        None => return false,
    };
    dev.address.enable_bus_master();
    let lo = (dev.address.bar(0) & 0xFFFF_FFF0) as u64;
    let hi = dev.address.bar(1) as u64;
    let mmio_phys = lo | (hi << 32);
    if mmio_phys == 0 {
        return false;
    }
    // If the USB mass-storage driver already brought up and owns this exact controller,
    // do NOT touch it: our bringup does a full HCRST, which would wipe the live MSC slot,
    // contexts and event ring and wedge every subsequent bulk transfer. HID input then
    // relies on PS/2 / firmware emulation for this controller.
    if crate::xhci::owns_controller(mmio_phys) {
        hlog!("[usbhid] xHCI {:#x} owned by USB mass-storage; skipping HID bringup", mmio_phys);
        return false;
    }
    let mmio = dma::map_mmio(mmio_phys, 0x40000);
    hlog!("[usbhid] xHCI at {:#x}", mmio_phys);
    match bringup(mmio) {
        Some(hid) => {
            let n = hid.devices.len();
            *HID.lock() = Some(hid);
            hlog!("[usbhid] {} HID device(s) ready", n);
            n > 0
        }
        None => false,
    }
}

/// Is at least one HID input device configured?
pub fn present() -> bool {
    HID.lock().as_ref().map(|h| !h.devices.is_empty()).unwrap_or(false)
}

/// Drain completed interrupt transfers, decode the boot reports into key/pointer input,
/// and re-arm each endpoint. Called from the shell + desktop loops; no interrupts needed.
pub fn poll() {
    let mut guard = HID.lock();
    let hid = match guard.as_mut() {
        Some(h) => h,
        None => return,
    };
    // Drain all ready transfer-completion events and dispatch by slot.
    while let Some((_d0, _d2, d3)) = hid.event.try_poll() {
        hid.update_erdp();
        if (d3 >> 10) & 0x3F != TRB_EVENT_TRANSFER {
            continue;
        }
        let slot = ((d3 >> 24) & 0xFF) as u8;
        if let Some(idx) = hid.devices.iter().position(|d| d.slot == slot) {
            dispatch(&mut hid.devices[idx]);
            hid.devices[idx].in_flight = false;
        }
    }
    // Re-arm any endpoint whose transfer completed.
    let db = hid.db;
    for d in hid.devices.iter_mut() {
        if !d.in_flight {
            arm(d, db);
        }
    }
}

/// Decode one device's freshly-received boot report and feed it to the input subsystems.
fn dispatch(d: &mut HidDev) {
    let rpt = unsafe { core::slice::from_raw_parts(d.report.virt as *const u8, 8) };
    match d.kind {
        Kind::Keyboard => {
            let mods = rpt[0];
            let shift = mods & 0x22 != 0;
            let ctrl = mods & 0x11 != 0;
            // Emit any keycode present now but not in the previous report (key-down edge).
            for &usage in &rpt[2..8] {
                if usage == 0 {
                    continue;
                }
                let was_down = d.last_kbd[2..8].contains(&usage);
                if !was_down {
                    if let Some(b) = hid_key(usage, shift, ctrl) {
                        crate::keyboard::inject(b);
                    }
                }
            }
            d.last_kbd[..8].copy_from_slice(&rpt[..8]);
        }
        Kind::Mouse => {
            // Boot mouse: [buttons, dx(i8), dy(i8), wheel]. HID dy is positive-down,
            // which matches screen Y, so pass it straight through.
            let buttons = rpt[0];
            let dx = rpt[1] as i8 as i32;
            let dy = rpt[2] as i8 as i32;
            crate::mouse::apply_hid(dx, dy, buttons & 1 != 0, buttons & 2 != 0);
        }
    }
}

/// Map a USB-HID boot keyboard usage code to the byte the input queue expects: printable
/// ASCII, Enter/Backspace/Tab/Esc/Space, the common punctuation, and the navigation /
/// clipboard control bytes the text engine understands ([`dominion_core::keys`]).
fn hid_key(usage: u8, shift: bool, ctrl: bool) -> Option<u8> {
    use dominion_core::keys;
    // Ctrl+letter → clipboard / select-all chords (matches the PS/2 path).
    if ctrl {
        return match usage {
            0x06 => Some(keys::COPY),       // c
            0x1B => Some(keys::CUT),        // x
            0x19 => Some(keys::PASTE),      // v
            0x04 => Some(keys::SELECT_ALL), // a
            _ => None,
        };
    }
    // Letters a-z.
    if (0x04..=0x1D).contains(&usage) {
        let c = b'a' + (usage - 0x04);
        return Some(if shift { c.to_ascii_uppercase() } else { c });
    }
    // Digits 1-9, 0.
    if (0x1E..=0x27).contains(&usage) {
        let normal = b"1234567890";
        let shifted = b"!@#$%^&*()";
        let idx = (usage - 0x1E) as usize;
        return Some(if shift { shifted[idx] } else { normal[idx] });
    }
    let pair = |n: u8, s: u8| Some(if shift { s } else { n });
    match usage {
        0x28 => Some(b'\n'),     // Enter
        0x29 => Some(0x1B),      // Esc
        0x2A => Some(0x08),      // Backspace
        0x2B => Some(b'\t'),     // Tab
        0x2C => Some(b' '),      // Space
        0x2D => pair(b'-', b'_'),
        0x2E => pair(b'=', b'+'),
        0x2F => pair(b'[', b'{'),
        0x30 => pair(b']', b'}'),
        0x31 => pair(b'\\', b'|'),
        0x33 => pair(b';', b':'),
        0x34 => pair(b'\'', b'"'),
        0x35 => pair(b'`', b'~'),
        0x36 => pair(b',', b'<'),
        0x37 => pair(b'.', b'>'),
        0x38 => pair(b'/', b'?'),
        0x4A => Some(keys::HOME),
        0x4C => Some(keys::DELETE),
        0x4D => Some(keys::END),
        0x4F => Some(if shift { keys::SEL_RIGHT } else { keys::ARROW_RIGHT }),
        0x50 => Some(if shift { keys::SEL_LEFT } else { keys::ARROW_LEFT }),
        0x51 => Some(if shift { keys::SEL_DOWN } else { keys::ARROW_DOWN }),
        0x52 => Some(if shift { keys::SEL_UP } else { keys::ARROW_UP }),
        _ => None,
    }
}

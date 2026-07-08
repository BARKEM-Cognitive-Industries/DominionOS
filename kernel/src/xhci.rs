//! xHCI + USB Mass-Storage (Bulk-Only Transport / SCSI) block driver.
//!
//! This is the deepest driver in the tree: a real USB 3.x host-controller bring-up
//! (slots, contexts, command/event/transfer rings, doorbells), USB device enumeration
//! (descriptors, set-configuration), and the Mass-Storage **Bulk-Only Transport** —
//! CBW → data → CSW wrapping SCSI READ(10)/WRITE(10)/READ CAPACITY — exposed as a
//! [`BlockDevice`]. With it, DominionOS can read and write a **USB flash drive**, i.e.
//! persist the boot/debug log to the very USB it booted from on bare metal.
//!
//! Discipline: polling only, every wait spin-bounded, every step `serial_println!`-logged
//! behind [`DEBUG`], and [`probe`] returns `None` on any failure — so a wedged or absent
//! controller never hangs boot, it just yields no USB disk.

use crate::dma::{self, DmaRegion};
use crate::pci;
use crate::usbmsc::{self, Bot, BulkTransport, Xfer};
use dominion_core::persist::{BlockDevice, BlockError, BLOCK_SIZE};

/// Verbose bring-up logging (helps post-mortem on real hardware via the boot log).
const DEBUG: bool = true;
macro_rules! xlog {
    ($($arg:tt)*) => { if DEBUG { crate::serial_println!($($arg)*); } };
}

// Fail-fast bound on hardware-ready spin loops (~0.3 s of slow MMIO reads); was
// 20_000_000, which on real USB controllers (× several reset/halt stages × ports) was
// minutes of stall that looked like a hang. Generous for a healthy controller.
const SPIN: u32 = 1_000_000;
const RING_TRBS: usize = 256; // one 4 KiB page of 16-byte TRBs

// ── operational / runtime register offsets ──
const OP_USBCMD: u64 = 0x00;
const OP_USBSTS: u64 = 0x04;
const OP_CRCR: u64 = 0x18;
const OP_DCBAAP: u64 = 0x30;
const OP_CONFIG: u64 = 0x38;
const OP_PORTSC_BASE: u64 = 0x400; // port 1 PORTSC; stride 0x10

const USBCMD_RS: u32 = 1 << 0;
const USBCMD_HCRST: u32 = 1 << 1;
const USBCMD_INTE: u32 = 1 << 2;
const USBSTS_HCH: u32 = 1 << 0;
const USBSTS_CNR: u32 = 1 << 11; // controller not ready

const PORTSC_CCS: u32 = 1 << 0; // current connect status
const PORTSC_PED: u32 = 1 << 1; // port enabled
const PORTSC_PR: u32 = 1 << 4; // port reset
const PORTSC_PP: u32 = 1 << 9; // port power
// Write-1-to-clear status change bits live in PORTSC; preserve them when writing.
const PORTSC_RW_MASK: u32 = PORTSC_PP; // bits we set; change bits handled separately

// TRB types.
const TRB_NORMAL: u32 = 1;
const TRB_SETUP: u32 = 2;
const TRB_DATA: u32 = 3;
const TRB_STATUS: u32 = 4;
const TRB_LINK: u32 = 6;
const TRB_ENABLE_SLOT: u32 = 9;
const TRB_ADDRESS_DEVICE: u32 = 11;
const TRB_CONFIGURE_ENDPOINT: u32 = 12;
const TRB_RESET_ENDPOINT: u32 = 14;
const TRB_SET_TR_DEQUEUE: u32 = 16;
const TRB_EVENT_TRANSFER: u32 = 32;
const TRB_EVENT_CMD_COMPLETE: u32 = 33;

const CC_SUCCESS: u32 = 1;
const CC_STALL: u32 = 6; // endpoint halted (STALL) — triggers BOT error recovery
const CC_SHORT_PACKET: u32 = 13;

// ── standard USB request constants used for stall recovery ──
const REQ_CLEAR_FEATURE: u8 = 1;
const FEAT_ENDPOINT_HALT: u16 = 0; // wValue for Clear-Feature(ENDPOINT_HALT)
const BM_STD_ENDPOINT_OUT: u8 = 0x02; // host->device, standard, recipient = endpoint

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

/// A producer ring (command or transfer): a page of TRBs with a trailing Link TRB that
/// wraps to the start and toggles the cycle state.
struct Ring {
    region: DmaRegion,
    enqueue: usize,
    pcs: bool, // producer cycle state
}

impl Ring {
    fn new() -> Option<Ring> {
        let region = dma::alloc(1)?;
        // Pre-write the Link TRB in the last slot: points back to base, toggle-cycle.
        let link = region.virt + ((RING_TRBS - 1) as u64) * 16;
        unsafe {
            w64(link, region.phys);
            w32(link + 8, 0);
            w32(link + 12, (TRB_LINK << 10) | (1 << 1)); // type=Link, TC=1, cycle set later
        }
        Some(Ring { region, enqueue: 0, pcs: true })
    }

    fn phys_with_cycle(&self) -> u64 {
        self.region.phys | 1 // RCS / DCS bit
    }

    /// Physical address of the next slot the producer will write — used as the Set TR
    /// Dequeue Pointer target when recovering a stalled endpoint.
    fn enqueue_phys(&self) -> u64 {
        self.region.phys + (self.enqueue as u64) * 16
    }

    /// Enqueue one TRB (the cycle bit is OR-ed in from the current PCS). Returns the
    /// physical address of the slot it was written to (for matching transfer events).
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
            // Set the Link TRB's cycle to the current PCS, then toggle and wrap.
            let link = self.region.virt + ((RING_TRBS - 1) as u64) * 16;
            unsafe {
                let ctl = (TRB_LINK << 10) | (1 << 1) | (self.pcs as u32);
                w32(link + 12, ctl);
            }
            self.pcs = !self.pcs;
            self.enqueue = 0;
        }
        slot_phys
    }
}

/// The event ring: a single segment the controller writes; software polls the cycle bit.
struct EventRing {
    seg: DmaRegion,
    erst: DmaRegion,
    dequeue: usize,
    ccs: bool, // consumer cycle state
}

impl EventRing {
    fn new() -> Option<EventRing> {
        let seg = dma::alloc(1)?;
        let erst = dma::alloc(1)?;
        unsafe {
            w64(erst.virt, seg.phys); // segment base
            w32(erst.virt + 8, RING_TRBS as u32); // segment size (TRBs)
            w32(erst.virt + 12, 0);
        }
        Some(EventRing { seg, erst, dequeue: 0, ccs: true })
    }

    fn current_phys(&self) -> u64 {
        self.seg.phys + (self.dequeue as u64) * 16
    }

    /// Poll for the next event TRB. Returns (d0, d2, d3) or None on timeout.
    fn poll(&mut self) -> Option<(u64, u32, u32)> {
        let mut spins = 0;
        loop {
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
                return Some((d0, d2, d3));
            }
            spins += 1;
            if spins >= SPIN {
                return None;
            }
            core::hint::spin_loop();
        }
    }
}

/// The wire-level xHCI transfer plumbing for one addressed device: rings, doorbell, event
/// ring, and the control/bulk primitives. The SCSI/BOT state machine in [`crate::usbmsc`]
/// drives this through the [`BulkTransport`] trait, so all storage-protocol logic lives
/// there and only the raw transfers live here.
struct Transport {
    db: u64, // doorbell array base
    rt: u64, // runtime base (for ERDP updates)
    slot: u8,
    ep_in_dci: u32,
    ep_out_dci: u32,
    cmd_ring: Ring,
    event: EventRing,
    ep0_ring: Ring,
    in_ring: Ring,
    out_ring: Ring,
    data: DmaRegion, // bulk data bounce
    cbw: DmaRegion,  // CBW/CSW scratch
}

impl Transport {
    fn ring_doorbell(&self, slot: u8, target: u32) {
        unsafe { w32(self.db + slot as u64 * 4, target) };
    }
    fn update_erdp(&self) {
        // Interrupter 0 ERDP at rt + 0x20 + 0x18, with the EHB (bit 3) write-1-to-clear.
        unsafe { w64(self.rt + 0x20 + 0x18, self.event.current_phys() | (1 << 3)) };
    }

    /// Wait for an event of `want_type`, draining unrelated events (e.g. port-status
    /// change) in between. Returns `(completion_code, data_ptr, dword3)`:
    /// - `completion_code`: completion status from bits 31-24 of dword2.
    /// - `data_ptr`: the TRB pointer field (dword0) — for transfer events this is the
    ///   physical address of the TRB that completed, used to match the pending transfer.
    /// - `dword3`: carries the slot id (bits 24-31) for command-completion events.
    fn wait_event(&mut self, want_type: u32) -> Option<(u32, u64, u32)> {
        // Absolute iteration cap so this can never hang. Resetting the controller we
        // booted from can trigger a continuous stream of port-status-change events; the
        // old unbounded loop drained them forever waiting for a completion that never
        // came — a multi-minute freeze at the USB probe on real hardware.
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
            let ttype = (d3 >> 10) & 0x3F;
            self.update_erdp();
            if ttype == want_type {
                // d0 is the data pointer (TRB address for transfer events, command TRB
                // address for command-completion events). Return it so callers can
                // verify which transfer completed rather than discarding it.
                return Some(((d2 >> 24) & 0xFF, d0, d3));
            }
        }
    }
}

/// Translate an xHCI transfer completion code into the transport-neutral [`Xfer`] the
/// BOT state machine consumes.
fn cc_to_xfer(cc: Option<u32>) -> Xfer {
    match cc {
        Some(CC_SUCCESS) => Xfer::Ok,
        Some(CC_SHORT_PACKET) => Xfer::Short,
        Some(CC_STALL) => Xfer::Stall,
        _ => Xfer::Fail,
    }
}

impl BulkTransport for Transport {
    fn bulk_out(&mut self, phys: u64, len: u32) -> Xfer {
        cc_to_xfer(self.bulk_buf(false, phys, len))
    }
    fn bulk_in(&mut self, phys: u64, len: u32) -> Xfer {
        cc_to_xfer(self.bulk_buf(true, phys, len))
    }
    fn clear_stall(&mut self, in_ep: bool) -> bool {
        self.clear_stall_ep(in_ep).is_some()
    }
    fn cbw_region(&self) -> (u64, u64) {
        (self.cbw.phys, self.cbw.virt)
    }
    fn data_region(&self) -> (u64, u64, usize) {
        (self.data.phys, self.data.virt, self.data.size as usize)
    }
}

/// A USB mass-storage device reached through xHCI: a [`BlockDevice`] over Bulk-Only
/// Transport + SCSI. Wire transfers go through [`Transport`]; the BOT/SCSI state machine
/// is [`crate::usbmsc::Bot`]. The two are separate fields so a `BlockDevice` call can
/// borrow the state machine and the transport at once without aliasing.
pub struct UsbMsc {
    xfer: Transport,
    bot: Bot,
}

impl UsbMsc {
    /// Exercise the SCSI command path for the boot self-test: issue INQUIRY and confirm
    /// the capacity learned at probe. Returns true when the device answers INQUIRY and
    /// reports a non-empty 512-byte-block medium.
    pub fn self_test(&mut self) -> bool {
        let inquiry_ok = self.bot.inquiry(&mut self.xfer).is_some();
        inquiry_ok && self.bot.sectors > 0 && self.bot.block_size as usize == BLOCK_SIZE
    }
}

impl BlockDevice for UsbMsc {
    fn block_count(&self) -> u64 {
        self.bot.sectors
    }
    fn read_block(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        self.bot.read_blocks(&mut self.xfer, lba, buf)
    }
    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        self.bot.write_blocks(&mut self.xfer, lba, buf)
    }
    fn read_blocks(&mut self, start_lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        self.bot.read_blocks(&mut self.xfer, start_lba, buf)
    }
    fn write_blocks(&mut self, start_lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        self.bot.write_blocks(&mut self.xfer, start_lba, buf)
    }
}

/// Probe PCI for an xHCI controller with a USB mass-storage device and bring it up.
pub fn probe() -> Option<UsbMsc> {
    let dev = pci::enumerate()
        .into_iter()
        .find(|d| d.class_code == 0x0C && d.subclass == 0x03 && d.address.prog_if() == 0x30)?;
    dev.address.enable_bus_master();
    let lo = (dev.address.bar(0) & 0xFFFF_FFF0) as u64;
    let hi = dev.address.bar(1) as u64;
    let mmio_phys = lo | (hi << 32);
    if mmio_phys == 0 {
        return None;
    }
    // Map the xHCI register BAR (capability + operational + runtime + doorbell arrays) —
    // MMIO, not RAM, so it must be mapped before any register access. 256 KiB covers the
    // runtime/doorbell arrays of large controllers.
    let mmio = dma::map_mmio(mmio_phys, 0x40000);
    xlog!("[xhci] controller at {:#x}", mmio_phys);

    bringup(mmio)
}

fn bringup(mmio: u64) -> Option<UsbMsc> {
    let caplen = (unsafe { r32(mmio) } & 0xFF) as u64;
    let hcsparams1 = unsafe { r32(mmio + 0x04) };
    let max_ports = ((hcsparams1 >> 24) & 0xFF) as u64;
    let max_slots = (hcsparams1 & 0xFF) as u32;
    let hccparams1 = unsafe { r32(mmio + 0x10) };
    let csz = (hccparams1 >> 2) & 1;
    let ctx_size: u64 = if csz == 1 { 64 } else { 32 };
    let dboff = (unsafe { r32(mmio + 0x14) } & !0x3) as u64;
    let rtsoff = (unsafe { r32(mmio + 0x18) } & !0x1F) as u64;
    let op = mmio + caplen;
    let rt = mmio + rtsoff;
    let db = mmio + dboff;
    xlog!("[xhci] caplen={} ports={} slots={} ctx_size={}", caplen, max_ports, max_slots, ctx_size);

    // Wait until the controller is ready, then halt + reset it.
    let mut spins = 0;
    while unsafe { r32(op + OP_USBSTS) } & USBSTS_CNR != 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
    }
    unsafe { w32(op + OP_USBCMD, 0) }; // clear R/S → halt
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
    xlog!("[xhci] reset complete");

    // MaxSlotsEn = MaxSlots.
    unsafe { w32(op + OP_CONFIG, max_slots) };

    // DCBAA (+ scratchpad buffers if the controller demands them).
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
        xlog!("[xhci] {} scratchpad buffers", max_scratch);
    }
    unsafe { w64(op + OP_DCBAAP, dcbaa.phys) };

    // Command ring.
    let cmd_ring = Ring::new()?;
    unsafe { w64(op + OP_CRCR, cmd_ring.phys_with_cycle()) };

    // Event ring (interrupter 0).
    let event = EventRing::new()?;
    unsafe {
        w32(rt + 0x20 + 0x08, 1); // ERSTSZ = 1 segment
        w64(rt + 0x20 + 0x10, event.erst.phys); // ERSTBA
        w64(rt + 0x20 + 0x18, event.seg.phys); // ERDP
        w32(rt + 0x20 + 0x00, 1 << 1); // IMAN.IE
    }

    // Run.
    unsafe { w32(op + OP_USBCMD, USBCMD_RS | USBCMD_INTE) };
    spins = 0;
    while unsafe { r32(op + OP_USBSTS) } & USBSTS_HCH != 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
    }
    xlog!("[xhci] running");

    // Find a connected port and reset it.
    let mut port = 0u64;
    let mut speed = 0u32;
    for i in 0..max_ports {
        let psc = op + OP_PORTSC_BASE + i * 0x10;
        let v = unsafe { r32(psc) };
        if v & PORTSC_CCS != 0 {
            // Power + reset the port; preserve change bits (write-1-clear) off.
            unsafe { w32(psc, (v & PORTSC_RW_MASK) | PORTSC_PP | PORTSC_PR) };
            let mut s = 0;
            while unsafe { r32(psc) } & PORTSC_PR != 0 && s < SPIN {
                s += 1;
            }
            let after = unsafe { r32(psc) };
            if after & PORTSC_PED != 0 {
                port = i + 1; // port numbers are 1-based
                speed = (after >> 10) & 0xF;
                // Clear any change bits.
                unsafe { w32(psc, after) };
                break;
            }
        }
    }
    if port == 0 {
        xlog!("[xhci] no enabled USB port with a device");
        return None;
    }
    xlog!("[xhci] device on port {} speed {}", port, speed);

    let mut xfer = Transport {
        db,
        rt,
        slot: 0,
        ep_in_dci: 0,
        ep_out_dci: 0,
        cmd_ring,
        event,
        ep0_ring: Ring::new()?,
        in_ring: Ring::new()?,
        out_ring: Ring::new()?,
        data: dma::alloc(8)?, // 32 KiB bounce
        cbw: dma::alloc(1)?,
    };

    // Enable a slot.
    xfer.cmd_ring.push(0, 0, TRB_ENABLE_SLOT << 10);
    xfer.ring_doorbell(0, 0);
    let (cc, _, d3) = xfer.wait_event(TRB_EVENT_CMD_COMPLETE)?;
    if cc != CC_SUCCESS {
        xlog!("[xhci] enable slot failed cc={}", cc);
        return None;
    }
    xfer.slot = ((d3 >> 24) & 0xFF) as u8;
    xlog!("[xhci] slot {}", xfer.slot);
    if xfer.slot == 0 {
        return None;
    }

    // Output device context + input context for Address Device.
    let out_ctx = dma::alloc(1)?;
    let in_ctx = dma::alloc(1)?;
    unsafe { w64(dcbaa.virt + xfer.slot as u64 * 8, out_ctx.phys) };

    // Input control context: add slot + EP0 (A0, A1).
    unsafe { w32(in_ctx.virt + 0x04, 0b11) };
    // Slot context (after the input control context, one ctx_size in).
    let slot_ctx = in_ctx.virt + ctx_size;
    let mps0 = match speed {
        4 => 512, // SuperSpeed
        3 => 64,  // High
        2 => 8,   // Low
        _ => 64,  // Full/default
    };
    unsafe {
        // dword0: context entries (1) in bits 27-31, speed in 20-23.
        w32(slot_ctx, (1u32 << 27) | (speed << 20));
        // dword1: root hub port number in bits 16-23.
        w32(slot_ctx + 4, (port as u32) << 16);
    }
    // EP0 context (input control + slot + ep0 = index 2).
    let ep0_ctx = in_ctx.virt + ctx_size * 2;
    unsafe {
        // dword1: EP type = Control (4) in bits 3-5, CErr=3 in bits 1-2, MaxPacket in 16-31.
        w32(ep0_ctx + 4, (4u32 << 3) | (3 << 1) | ((mps0 as u32) << 16));
        // TR dequeue pointer (dword2/3) with DCS=1.
        w64(ep0_ctx + 8, xfer.ep0_ring.phys_with_cycle());
    }
    xfer.cmd_ring.push(in_ctx.phys, 0, (TRB_ADDRESS_DEVICE << 10) | ((xfer.slot as u32) << 24));
    xfer.ring_doorbell(0, 0);
    let (cc, _, _) = xfer.wait_event(TRB_EVENT_CMD_COMPLETE)?;
    if cc != CC_SUCCESS {
        xlog!("[xhci] address device failed cc={}", cc);
        return None;
    }
    xlog!("[xhci] addressed");

    // GET_DESCRIPTOR(config, 9 bytes) then full config to find the MSC interface.
    let cfg = xfer.data;
    if xfer.control_in(0x80, 6, 0x0200, 0, 9, cfg.phys).is_none() {
        return None;
    }
    let total_len = u16::from_le_bytes(unsafe {
        [
            *((cfg.virt + 2) as *const u8),
            *((cfg.virt + 3) as *const u8),
        ]
    }) as u16;
    let want = (total_len as u32).min(cfg.size as u32);
    if xfer.control_in(0x80, 6, 0x0200, 0, want as u16, cfg.phys).is_none() {
        return None;
    }
    // Parse descriptors for a Mass-Storage (class 8, subclass 6 SCSI, proto 0x50 BOT)
    // interface and its bulk IN/OUT endpoints.
    let buf = unsafe { core::slice::from_raw_parts(cfg.virt as *const u8, want as usize) };
    let cfg_value = buf.get(5).copied().unwrap_or(1);
    let mut i = 0usize;
    let mut in_ep = 0u8;
    let mut out_ep = 0u8;
    let mut in_mps = 512u16;
    let mut out_mps = 512u16;
    let mut is_msc = false;
    while i + 2 <= buf.len() {
        let len = buf[i] as usize;
        let dtype = buf[i + 1];
        if len == 0 {
            break;
        }
        if dtype == 0x04 && i + 9 <= buf.len() {
            // interface descriptor: match class 0x08 / subclass 0x06 (SCSI) / proto 0x50 (BOT)
            is_msc = buf[i + 5] == usbmsc::MSC_CLASS
                && buf[i + 6] == usbmsc::MSC_SUBCLASS_SCSI
                && buf[i + 7] == usbmsc::MSC_PROTO_BOT;
        } else if dtype == 0x05 && is_msc && i + 7 <= buf.len() {
            // endpoint descriptor
            let addr = buf[i + 2];
            let attr = buf[i + 3] & 0x3;
            let mps = u16::from_le_bytes([buf[i + 4], buf[i + 5]]);
            if attr == 2 {
                // bulk
                if addr & 0x80 != 0 {
                    in_ep = addr & 0x0F;
                    in_mps = mps;
                } else {
                    out_ep = addr & 0x0F;
                    out_mps = mps;
                }
            }
        }
        i += len;
    }
    if in_ep == 0 || out_ep == 0 {
        xlog!("[xhci] no bulk endpoints found (msc={})", is_msc);
        return None;
    }
    xfer.ep_in_dci = (in_ep as u32 * 2) + 1;
    xfer.ep_out_dci = out_ep as u32 * 2;
    xlog!("[xhci] bulk in ep{} out ep{}", in_ep, out_ep);

    // SET_CONFIGURATION.
    if xfer.control_out(0x00, 9, cfg_value as u16, 0).is_none() {
        return None;
    }

    // Configure Endpoint: add the two bulk endpoints.
    unsafe {
        core::ptr::write_bytes(in_ctx.virt as *mut u8, 0, 4096);
        let add = (1u32 << xfer.ep_in_dci) | (1u32 << xfer.ep_out_dci);
        w32(in_ctx.virt + 0x04, add | 1); // A0 (slot) + the two EPs
        // Slot context: context entries = max DCI.
        let max_dci = xfer.ep_in_dci.max(xfer.ep_out_dci);
        w32(in_ctx.virt + ctx_size, (max_dci << 27) | (speed << 20));
        w32(in_ctx.virt + ctx_size + 4, (port as u32) << 16);
        // Bulk OUT EP context (type 2), bulk IN EP context (type 6).
        let out_ctx_ep = in_ctx.virt + ctx_size * (1 + xfer.ep_out_dci as u64);
        w32(out_ctx_ep + 4, (2u32 << 3) | (3 << 1) | ((out_mps as u32) << 16));
        w64(out_ctx_ep + 8, xfer.out_ring.phys_with_cycle());
        let in_ctx_ep = in_ctx.virt + ctx_size * (1 + xfer.ep_in_dci as u64);
        w32(in_ctx_ep + 4, (6u32 << 3) | (3 << 1) | ((in_mps as u32) << 16));
        w64(in_ctx_ep + 8, xfer.in_ring.phys_with_cycle());
    }
    xfer.cmd_ring.push(in_ctx.phys, 0, (TRB_CONFIGURE_ENDPOINT << 10) | ((xfer.slot as u32) << 24));
    xfer.ring_doorbell(0, 0);
    let (cc, _, _) = xfer.wait_event(TRB_EVENT_CMD_COMPLETE)?;
    if cc != CC_SUCCESS {
        xlog!("[xhci] configure endpoint failed cc={}", cc);
        return None;
    }
    xlog!("[xhci] endpoints configured");

    // Bind the SCSI/BOT state machine to the configured bulk endpoints.
    let mut bot = Bot::new();

    // TEST UNIT READY: some devices answer "not ready" until spun up, so poll a bounded
    // number of times. Non-fatal — READ CAPACITY below is the real gate.
    for _ in 0..16 {
        if bot.test_unit_ready(&mut xfer) {
            break;
        }
    }

    // INQUIRY — informational; logs the SCSI peripheral device type for the boot log.
    if let Some(inq) = bot.inquiry(&mut xfer) {
        xlog!("[xhci] INQUIRY ok, peripheral-device-type {:#x}", inq[0] & 0x1F);
    }

    // SCSI READ CAPACITY(10) → block count + size.
    if !bot.read_capacity(&mut xfer) {
        xlog!("[xhci] READ CAPACITY failed");
        return None;
    }
    xlog!(
        "[xhci] USB disk: {} sectors x {} bytes ({} MiB)",
        bot.sectors,
        bot.block_size,
        bot.sectors * bot.block_size as u64 / (1024 * 1024)
    );
    Some(UsbMsc { xfer, bot })
}

impl Transport {
    /// A control IN transfer (Setup/Data-IN/Status-OUT) on EP0 into `data_phys`.
    fn control_in(&mut self, bm: u8, req: u8, value: u16, index: u16, len: u16, data_phys: u64) -> Option<()> {
        let setup_lo = (bm as u64)
            | ((req as u64) << 8)
            | ((value as u64) << 16)
            | ((index as u64) << 32)
            | ((len as u64) << 48);
        // Setup: IDT=1 (bit6), TRT=3 (IN) in bits 16-17.
        self.ep0_ring.push(setup_lo, 8, (TRB_SETUP << 10) | (1 << 6) | (3 << 16));
        if len > 0 {
            // Data IN: DIR=1 (bit16).
            self.ep0_ring.push(data_phys, len as u32, (TRB_DATA << 10) | (1 << 16));
        }
        // Status OUT (DIR=0), IOC.
        self.ep0_ring.push(0, 0, (TRB_STATUS << 10) | (1 << 5));
        self.ring_doorbell(self.slot, 1);
        let (cc, _data_ptr, _) = self.wait_event(TRB_EVENT_TRANSFER)?;
        if cc == CC_SUCCESS || cc == CC_SHORT_PACKET {
            Some(())
        } else {
            None
        }
    }

    /// A control OUT transfer with no data stage (e.g. SET_CONFIGURATION).
    fn control_out(&mut self, bm: u8, req: u8, value: u16, index: u16) -> Option<()> {
        let setup_lo = (bm as u64) | ((req as u64) << 8) | ((value as u64) << 16) | ((index as u64) << 32);
        self.ep0_ring.push(setup_lo, 8, (TRB_SETUP << 10) | (1 << 6)); // TRT=0 (no data)
        self.ep0_ring.push(0, 0, (TRB_STATUS << 10) | (1 << 16) | (1 << 5)); // Status IN, IOC
        self.ring_doorbell(self.slot, 1);
        let (cc, _data_ptr, _) = self.wait_event(TRB_EVENT_TRANSFER)?;
        if cc == CC_SUCCESS {
            Some(())
        } else {
            None
        }
    }

    /// Bulk transfer of `len` bytes to/from a given physical buffer (Normal TRB + IOC).
    /// Returns the xHCI completion code, or `None` on timeout / event mismatch.
    fn bulk_buf(&mut self, dir_in: bool, phys: u64, len: u32) -> Option<u32> {
        let ctl = (TRB_NORMAL << 10) | (1 << 5);
        let trb_phys = if dir_in {
            let p = self.in_ring.push(phys, len, ctl);
            self.ring_doorbell(self.slot, self.ep_in_dci);
            p
        } else {
            let p = self.out_ring.push(phys, len, ctl);
            self.ring_doorbell(self.slot, self.ep_out_dci);
            p
        };
        // data_ptr in the transfer event TRB identifies which transfer completed.
        // Verify it matches the TRB we just submitted before accepting the result.
        let (cc, data_ptr, _) = self.wait_event(TRB_EVENT_TRANSFER)?;
        if data_ptr != trb_phys {
            xlog!(
                "[xhci] bulk event data_ptr mismatch: got {:#x} expected {:#x}",
                data_ptr,
                trb_phys
            );
            return None;
        }
        Some(cc)
    }

    /// Recover a halted bulk endpoint (the BOT stall-recovery path). Per xHCI: issue a
    /// Reset Endpoint command, send the device a Clear-Feature(ENDPOINT_HALT), then Set TR
    /// Dequeue Pointer so the controller resumes at the ring's next slot. `in_ep` selects
    /// the bulk IN endpoint, else the bulk OUT endpoint.
    fn clear_stall_ep(&mut self, in_ep: bool) -> Option<()> {
        let dci = if in_ep { self.ep_in_dci } else { self.ep_out_dci };
        // USB endpoint address for the Clear-Feature wIndex: number + direction bit.
        let ep_num = if in_ep { (self.ep_in_dci - 1) / 2 } else { self.ep_out_dci / 2 };
        let windex = if in_ep { ep_num | 0x80 } else { ep_num };

        // 1. Reset Endpoint command (clears the controller's Halted state).
        self.cmd_ring.push(0, 0, (TRB_RESET_ENDPOINT << 10) | (dci << 16) | ((self.slot as u32) << 24));
        self.ring_doorbell(0, 0);
        let (cc, _, _) = self.wait_event(TRB_EVENT_CMD_COMPLETE)?;
        if cc != CC_SUCCESS {
            xlog!("[xhci] reset endpoint (dci {}) failed cc={}", dci, cc);
            return None;
        }

        // 2. Clear-Feature(ENDPOINT_HALT) to the device, resetting its data toggle.
        self.control_out(BM_STD_ENDPOINT_OUT, REQ_CLEAR_FEATURE, FEAT_ENDPOINT_HALT, windex as u16)?;

        // 3. Set TR Dequeue Pointer to the ring's current enqueue position, with the
        //    producer cycle state as the Dequeue Cycle State.
        let (dq_phys, dcs) = if in_ep {
            (self.in_ring.enqueue_phys(), self.in_ring.pcs)
        } else {
            (self.out_ring.enqueue_phys(), self.out_ring.pcs)
        };
        let ptr = dq_phys | (dcs as u64);
        self.cmd_ring.push(ptr, 0, (TRB_SET_TR_DEQUEUE << 10) | (dci << 16) | ((self.slot as u32) << 24));
        self.ring_doorbell(0, 0);
        let (cc, _, _) = self.wait_event(TRB_EVENT_CMD_COMPLETE)?;
        if cc != CC_SUCCESS {
            xlog!("[xhci] set TR dequeue (dci {}) failed cc={}", dci, cc);
            return None;
        }
        Some(())
    }
}

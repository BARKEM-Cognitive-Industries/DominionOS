//! USB Mass Storage — **Bulk-Only Transport** (BBB) + minimal SCSI.
//!
//! This is the transport-independent half of the USB-disk driver. It owns the Bulk-Only
//! Transport state machine (CBW → data → CSW) and the SCSI command set, and drives them
//! over a [`BulkTransport`] — the small bundle of bulk IN/OUT, clear-stall, and
//! scratch-buffer operations that [`crate::xhci`] implements against its transfer rings.
//! Keeping the protocol here (mirroring how [`crate::usbhid`] is a class driver over the
//! same controller) means the storage logic is one self-contained unit no matter which
//! host controller (xHCI today, EHCI/OHCI later) carries the bytes.
//!
//! Matches USB Mass Storage class **0x08** / subclass **0x06** (SCSI transparent) /
//! protocol **0x50** (Bulk-Only). The command set is deliberately minimal — INQUIRY,
//! TEST UNIT READY, READ CAPACITY(10), READ(10), WRITE(10) — which is enough to report a
//! device's capacity and read/write 512-byte logical blocks as a first-class
//! [`BlockDevice`](dominion_core::persist::BlockDevice).

use dominion_core::persist::{BlockError, BLOCK_SIZE};

/// `dCBWSignature` — "USBC" little-endian on the wire.
const CBW_SIGNATURE: u32 = 0x4342_5355;
/// `dCSWSignature` — "USBS" little-endian on the wire.
const CSW_SIGNATURE: u32 = 0x5342_5355;
const CBW_LEN: usize = 31;
const CSW_LEN: usize = 13;
/// The CSW is read into the CBW scratch region at this offset so it never overlaps the CBW.
const CSW_OFFSET: u64 = 64;

// ── SCSI opcodes we issue over BOT ──
const SCSI_TEST_UNIT_READY: u8 = 0x00;
const SCSI_INQUIRY: u8 = 0x12;
const SCSI_READ_CAPACITY_10: u8 = 0x25;
const SCSI_READ_10: u8 = 0x28;
const SCSI_WRITE_10: u8 = 0x2A;

/// USB Mass-Storage interface identity, so the descriptor match in [`crate::xhci`] and the
/// protocol here agree on exactly which interface this driver binds to.
pub const MSC_CLASS: u8 = 0x08;
pub const MSC_SUBCLASS_SCSI: u8 = 0x06;
pub const MSC_PROTO_BOT: u8 = 0x50;

/// The normalised result of one bulk transfer, so the SCSI/BOT layer never sees
/// controller-specific completion codes: the [`BulkTransport`] implementation translates
/// its own codes into this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Xfer {
    /// The full requested length moved.
    Ok,
    /// Completed, but the device sent fewer bytes than requested — normal for INQUIRY /
    /// READ CAPACITY where the allocation length exceeds the reply.
    Short,
    /// The endpoint halted (STALL); the caller runs error recovery.
    Stall,
    /// Timed out or a hard transfer error.
    Fail,
}

/// The wire-level bulk plumbing a host controller must provide for [`Bot`] to run. The
/// whole SCSI/BOT layer is written against this trait, so it never touches controller
/// registers directly. [`crate::xhci`] implements it over its transfer rings.
pub trait BulkTransport {
    /// Bulk **OUT** `len` bytes starting at physical address `phys`.
    fn bulk_out(&mut self, phys: u64, len: u32) -> Xfer;
    /// Bulk **IN** up to `len` bytes into physical address `phys`.
    fn bulk_in(&mut self, phys: u64, len: u32) -> Xfer;
    /// Recover a halted bulk endpoint: reset it, Clear-Feature(ENDPOINT_HALT) on the
    /// device, and resync the transfer ring. `in_ep = true` targets the bulk IN endpoint,
    /// else the bulk OUT endpoint. Returns false if recovery could not complete.
    fn clear_stall(&mut self, in_ep: bool) -> bool;
    /// Physical + virtual base of a scratch region (at least `CSW_OFFSET + 13` bytes) used
    /// to stage the CBW and receive the CSW.
    fn cbw_region(&self) -> (u64, u64);
    /// Physical + virtual base and byte length of the bulk data bounce buffer. The BOT
    /// data phase transfers to/from here; callers copy user data in/out of it.
    fn data_region(&self) -> (u64, u64, usize);
}

/// The Bulk-Only Transport + SCSI state machine for one logical unit. Holds the running
/// CBW tag and the geometry learned from READ CAPACITY; every method drives a
/// [`BulkTransport`] passed by the caller so `Bot` never borrows the controller.
pub struct Bot {
    /// `dCBWTag` counter — each command uses a fresh tag that the CSW must echo.
    tag: u32,
    /// Logical block count, set by [`Bot::read_capacity`].
    pub sectors: u64,
    /// Logical block size in bytes (must be [`BLOCK_SIZE`] for this driver).
    pub block_size: u32,
}

impl Default for Bot {
    fn default() -> Self {
        Bot::new()
    }
}

impl Bot {
    pub const fn new() -> Bot {
        Bot { tag: 0, sectors: 0, block_size: BLOCK_SIZE as u32 }
    }

    /// Run one SCSI command through Bulk-Only Transport: **CBW → (data) → CSW**.
    ///
    /// `data_len` is the data-phase byte count (already staged in the transport's data
    /// buffer for writes); `write` selects the data direction. Implements the BOT error
    /// recovery: a STALL in the data phase clears the halted data endpoint and still reads
    /// the CSW; a STALL on the CSW clears the IN endpoint and retries the CSW once. Returns
    /// true only on a CSW that echoes the tag with `bCSWStatus == 0` (command passed).
    fn transact<T: BulkTransport + ?Sized>(
        &mut self,
        t: &mut T,
        cdb: &[u8],
        data_len: u32,
        write: bool,
    ) -> bool {
        self.tag = self.tag.wrapping_add(1);
        let tag = self.tag;
        let (cbw_phys, cbw_virt) = t.cbw_region();

        // ── command phase: send the 31-byte CBW ──
        build_cbw(cbw_virt, tag, data_len, write, cdb);
        match t.bulk_out(cbw_phys, CBW_LEN as u32) {
            Xfer::Ok => {}
            Xfer::Stall => {
                // The device stalled the command itself; clear the OUT endpoint so the
                // next command has a clean ring, but this command has failed.
                let _ = t.clear_stall(false);
                return false;
            }
            _ => return false,
        }

        // ── data phase (optional) ──
        if data_len > 0 {
            let (data_phys, _v, cap) = t.data_region();
            if data_len as usize > cap {
                return false; // caller must chunk to the bounce buffer
            }
            let r = if write {
                t.bulk_out(data_phys, data_len)
            } else {
                t.bulk_in(data_phys, data_len)
            };
            match r {
                Xfer::Ok | Xfer::Short => {}
                Xfer::Stall => {
                    // Endpoint halted mid-data: clear the halted (data) endpoint and press
                    // on to read the CSW, per the BOT class error-recovery procedure.
                    if !t.clear_stall(!write) {
                        return false;
                    }
                }
                Xfer::Fail => return false,
            }
        }

        // ── status phase: read the 13-byte CSW, retrying once past a stalled IN EP ──
        let csw_phys = cbw_phys + CSW_OFFSET;
        let csw_virt = cbw_virt + CSW_OFFSET;
        let mut got = t.bulk_in(csw_phys, CSW_LEN as u32);
        if got == Xfer::Stall {
            if !t.clear_stall(true) {
                return false;
            }
            got = t.bulk_in(csw_phys, CSW_LEN as u32);
        }
        if got != Xfer::Ok && got != Xfer::Short {
            return false;
        }
        check_csw(csw_virt, tag)
    }

    /// SCSI **INQUIRY**: fetch the standard 36-byte inquiry data (peripheral type, vendor,
    /// product). Returns the raw reply, or `None` if the command failed.
    pub fn inquiry<T: BulkTransport + ?Sized>(&mut self, t: &mut T) -> Option<[u8; 36]> {
        let cdb = [SCSI_INQUIRY, 0, 0, 0, 36, 0];
        if !self.transact(t, &cdb, 36, false) {
            return None;
        }
        let (_, data_virt, _) = t.data_region();
        let mut out = [0u8; 36];
        unsafe {
            core::ptr::copy_nonoverlapping(data_virt as *const u8, out.as_mut_ptr(), 36);
        }
        Some(out)
    }

    /// SCSI **TEST UNIT READY**: true when the unit is ready for media access. Devices may
    /// answer "not ready" until spun up, so callers typically poll this a few times.
    pub fn test_unit_ready<T: BulkTransport + ?Sized>(&mut self, t: &mut T) -> bool {
        let cdb = [SCSI_TEST_UNIT_READY, 0, 0, 0, 0, 0];
        self.transact(t, &cdb, 0, false)
    }

    /// SCSI **READ CAPACITY(10)**: learn the last LBA and block size, filling
    /// [`Bot::sectors`] / [`Bot::block_size`]. Returns true when the medium is non-empty
    /// and uses [`BLOCK_SIZE`]-byte logical blocks (the only geometry this driver serves).
    pub fn read_capacity<T: BulkTransport + ?Sized>(&mut self, t: &mut T) -> bool {
        let cdb = [SCSI_READ_CAPACITY_10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        if !self.transact(t, &cdb, 8, false) {
            return false;
        }
        let (_, data_virt, _) = t.data_region();
        let d = unsafe { core::slice::from_raw_parts(data_virt as *const u8, 8) };
        // READ CAPACITY(10) returns big-endian last-LBA then block size.
        let last_lba = u32::from_be_bytes([d[0], d[1], d[2], d[3]]) as u64;
        let bsize = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
        self.sectors = last_lba + 1;
        self.block_size = if bsize == 0 { BLOCK_SIZE as u32 } else { bsize };
        self.sectors != 0 && self.block_size as usize == BLOCK_SIZE
    }

    /// Read `buf.len() / 512` consecutive blocks starting at `start_lba` into `buf`,
    /// pipelining as many blocks as the data bounce buffer holds through each SCSI
    /// READ(10) so a run of sectors costs one command per bounce-full, not one per sector.
    pub fn read_blocks<T: BulkTransport + ?Sized>(
        &mut self,
        t: &mut T,
        start_lba: u64,
        buf: &mut [u8],
    ) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let total = buf.len() / BLOCK_SIZE;
        let per = self.max_blocks(t);
        let (_, data_virt, _) = t.data_region();
        let mut done = 0usize;
        while done < total {
            let count = core::cmp::min(per, total - done);
            let lba = start_lba + done as u64;
            if lba + count as u64 > self.sectors || lba > u32::MAX as u64 {
                return Err(BlockError::OutOfRange);
            }
            let cdb = rw10_cdb(SCSI_READ_10, lba, count as u16);
            if !self.transact(t, &cdb, (count * BLOCK_SIZE) as u32, false) {
                return Err(BlockError::DeviceFault);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(
                    data_virt as *const u8,
                    buf[done * BLOCK_SIZE..].as_mut_ptr(),
                    count * BLOCK_SIZE,
                );
            }
            done += count;
        }
        Ok(())
    }

    /// Write `buf.len() / 512` consecutive blocks starting at `start_lba` from `buf`,
    /// pipelined through SCSI WRITE(10) a bounce-full at a time — the write mirror of
    /// [`Bot::read_blocks`].
    pub fn write_blocks<T: BulkTransport + ?Sized>(
        &mut self,
        t: &mut T,
        start_lba: u64,
        buf: &[u8],
    ) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let total = buf.len() / BLOCK_SIZE;
        let per = self.max_blocks(t);
        let (_, data_virt, _) = t.data_region();
        let mut done = 0usize;
        while done < total {
            let count = core::cmp::min(per, total - done);
            let lba = start_lba + done as u64;
            if lba + count as u64 > self.sectors || lba > u32::MAX as u64 {
                return Err(BlockError::OutOfRange);
            }
            // Stage the outgoing sectors into the bounce buffer, then issue the command.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf[done * BLOCK_SIZE..].as_ptr(),
                    data_virt as *mut u8,
                    count * BLOCK_SIZE,
                );
            }
            let cdb = rw10_cdb(SCSI_WRITE_10, lba, count as u16);
            if !self.transact(t, &cdb, (count * BLOCK_SIZE) as u32, true) {
                return Err(BlockError::DeviceFault);
            }
            done += count;
        }
        Ok(())
    }

    /// Blocks that fit in one data-phase transfer (the bounce buffer capacity in sectors).
    fn max_blocks<T: BulkTransport + ?Sized>(&self, t: &T) -> usize {
        let (_, _, cap) = t.data_region();
        (cap / BLOCK_SIZE).max(1)
    }
}

/// Build the 31-byte Command Block Wrapper in place at `cbw_virt`.
fn build_cbw(cbw_virt: u64, tag: u32, data_len: u32, write: bool, cdb: &[u8]) {
    let cbw = unsafe { core::slice::from_raw_parts_mut(cbw_virt as *mut u8, CBW_LEN) };
    cbw.fill(0);
    cbw[0..4].copy_from_slice(&CBW_SIGNATURE.to_le_bytes()); // dCBWSignature
    cbw[4..8].copy_from_slice(&tag.to_le_bytes()); // dCBWTag
    cbw[8..12].copy_from_slice(&data_len.to_le_bytes()); // dCBWDataTransferLength
    cbw[12] = if write { 0x00 } else { 0x80 }; // bmCBWFlags: bit7 = data-in
    cbw[13] = 0; // bCBWLUN (LUN 0)
    let n = cdb.len().min(16);
    cbw[14] = n as u8; // bCBWCBLength
    cbw[15..15 + n].copy_from_slice(&cdb[..n]); // CBWCB (the SCSI CDB)
}

/// Validate the 13-byte Command Status Wrapper at `csw_virt`: signature, echoed tag, and
/// `bCSWStatus == 0` (command passed).
fn check_csw(csw_virt: u64, expect_tag: u32) -> bool {
    let csw = unsafe { core::slice::from_raw_parts(csw_virt as *const u8, CSW_LEN) };
    let sig = u32::from_le_bytes([csw[0], csw[1], csw[2], csw[3]]);
    let tag = u32::from_le_bytes([csw[4], csw[5], csw[6], csw[7]]);
    // csw[8..12] = dCSWDataResidue; csw[12] = bCSWStatus (0 = passed, 1 = failed, 2 = phase error)
    let ok = sig == CSW_SIGNATURE && tag == expect_tag && csw[12] == 0;
    if !ok {
        crate::serial_println!(
            "[usbmsc] CSW bad: sig={:#x} tag={} (want {}) status={}",
            sig,
            tag,
            expect_tag,
            csw[12]
        );
    }
    ok
}

/// Build a READ(10)/WRITE(10) 10-byte CDB for `count` blocks at `lba` (32-bit LBA).
fn rw10_cdb(op: u8, lba: u64, count: u16) -> [u8; 10] {
    let lba32 = lba as u32;
    [
        op,
        0,
        (lba32 >> 24) as u8,
        (lba32 >> 16) as u8,
        (lba32 >> 8) as u8,
        lba32 as u8,
        0,
        (count >> 8) as u8,
        count as u8,
        0,
    ]
}

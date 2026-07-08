//! NVMe block driver — real read/write to an NVMe SSD, the storage most modern
//! laptops/desktops actually ship. Polling, an admin queue + one I/O queue, single
//! namespace — enough for persistence and the boot-log save path on bare metal.
//!
//! Bring-up follows the NVMe 1.x spec: find the controller on PCI (class 01:08), map
//! BAR0 (64-bit MMIO) through the bootloader's physical map, reset+enable the
//! controller with admin SQ/CQ, IDENTIFY namespace 1 for its capacity, create one I/O
//! SQ/CQ pair, then issue NVM READ (0x02) / WRITE (0x01). Transfers are chunked to one
//! page so a single PRP1 entry suffices (no PRP list). Every wait is spin-bounded.

use crate::dma::{self, DmaRegion};
use crate::pci;
use dominion_core::persist::{BlockDevice, BlockError, BLOCK_SIZE};

const REG_CAP: u64 = 0x00; // capabilities (64-bit)
const REG_CC: u64 = 0x14; // controller configuration
const REG_CSTS: u64 = 0x1C; // controller status
const REG_AQA: u64 = 0x24; // admin queue attributes
const REG_ASQ: u64 = 0x28; // admin SQ base (64-bit)
const REG_ACQ: u64 = 0x30; // admin CQ base (64-bit)
const DOORBELL_BASE: u64 = 0x1000;

const CC_EN: u32 = 1 << 0;
const CSTS_RDY: u32 = 1 << 0;

const QD: u16 = 8; // queue depth (entries) for both admin and I/O
const SQE_BYTES: usize = 64;
const CQE_BYTES: usize = 16;
// Fail-fast bound on hardware-ready spin loops. Each iteration is a slow uncached MMIO
// read (~hundreds of ns on real hardware), so this is ~0.3 s — generous for a healthy
// controller (ready bits assert in microseconds) but it gives up quickly on metal that
// never responds, instead of stalling the whole boot. Was 5_000_000 (multi-second).
const SPIN: u32 = 1_000_000;
/// One page per transfer ⇒ a single PRP1 entry; 4096/512 = 8 sectors max per command.
const MAX_SECTORS_PER_CMD: usize = 4096 / BLOCK_SIZE;

#[inline]
unsafe fn r32(a: u64) -> u32 {
    core::ptr::read_volatile(a as *const u32)
}
#[inline]
unsafe fn w32(a: u64, v: u32) {
    core::ptr::write_volatile(a as *mut u32, v);
}
#[inline]
unsafe fn r64(a: u64) -> u64 {
    core::ptr::read_volatile(a as *const u64)
}
#[inline]
unsafe fn w64(a: u64, v: u64) {
    core::ptr::write_volatile(a as *mut u64, v);
}

struct Queue {
    sq: DmaRegion,
    cq: DmaRegion,
    sq_tail: u16,
    cq_head: u16,
    phase: bool,
}

pub struct NvmeDisk {
    bar: u64,
    stride: u64,
    admin: Queue,
    io: Queue,
    data: DmaRegion,
    cid: u16,
    sectors: u64,
    /// Set once a command times out: the CQ is then desynchronized (the stale completion
    /// was never consumed), so no further command may be issued or it would read the wrong
    /// status. The disk is treated as permanently faulted.
    faulted: bool,
}

impl NvmeDisk {
    /// Submit a 64-byte command on the admin (qid 0) or I/O (qid 1) queue and poll its
    /// completion. Returns the 11-bit status (0 = success).
    fn submit(&mut self, qid: u64, sqe: &[u32; 16]) -> u16 {
        let q = if qid == 0 { &mut self.admin } else { &mut self.io };
        // Write the command into the SQ at the tail.
        let slot = q.sq.virt + q.sq_tail as u64 * SQE_BYTES as u64;
        for (i, &dw) in sqe.iter().enumerate() {
            unsafe { w32(slot + i as u64 * 4, dw) };
        }
        q.sq_tail = (q.sq_tail + 1) % QD;
        unsafe { w32(self.bar + DOORBELL_BASE + (2 * qid) * self.stride, q.sq_tail as u32) };

        // Poll the CQ entry at the head for the expected phase bit.
        let cqe = q.cq.virt + q.cq_head as u64 * CQE_BYTES as u64;
        let mut spins = 0;
        let mut d3;
        loop {
            d3 = unsafe { r32(cqe + 12) };
            if ((d3 >> 16) & 1) == q.phase as u32 {
                break;
            }
            spins += 1;
            if spins >= SPIN {
                return 0x7FF; // synthetic "timeout" status
            }
            core::hint::spin_loop();
        }
        let status = ((d3 >> 17) & 0x7FF) as u16;
        q.cq_head = (q.cq_head + 1) % QD;
        if q.cq_head == 0 {
            q.phase = !q.phase;
        }
        unsafe { w32(self.bar + DOORBELL_BASE + (2 * qid + 1) * self.stride, q.cq_head as u32) };
        status
    }

    fn next_cid(&mut self) -> u16 {
        self.cid = self.cid.wrapping_add(1);
        self.cid
    }

    /// Build a zeroed SQE with opcode, cid, nsid and PRP1.
    fn sqe(&mut self, opcode: u8, nsid: u32, prp1: u64) -> [u32; 16] {
        let mut s = [0u32; 16];
        s[0] = opcode as u32 | ((self.next_cid() as u32) << 16);
        s[1] = nsid;
        s[6] = prp1 as u32;
        s[7] = (prp1 >> 32) as u32;
        s
    }

    fn rw(&mut self, write: bool, lba: u64, count: u16) -> Result<(), BlockError> {
        // A prior timeout left the CQ desynchronized; refuse to issue further commands.
        if self.faulted {
            return Err(BlockError::DeviceFault);
        }
        let opcode = if write { 0x01 } else { 0x02 };
        let prp1 = self.data.phys;
        let mut s = self.sqe(opcode, 1, prp1);
        s[10] = lba as u32; // SLBA low
        s[11] = (lba >> 32) as u32; // SLBA high
        s[12] = (count.saturating_sub(1)) as u32; // NLB (0-based)
        let status = self.submit(1, &s);
        if status == 0x7FF {
            // Timeout: the completion was never consumed, so the CQ head/phase are now
            // stale. Mark the disk faulted so a later stale completion can't be
            // mis-attributed to a subsequent command.
            self.faulted = true;
        }
        if status != 0 {
            return Err(BlockError::DeviceFault);
        }
        Ok(())
    }

    fn data_slice(&self) -> &'static mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.data.virt as *mut u8, self.data.size) }
    }
}

impl BlockDevice for NvmeDisk {
    fn block_count(&self) -> u64 {
        self.sectors
    }

    fn read_block(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        if lba >= self.sectors {
            return Err(BlockError::OutOfRange);
        }
        self.rw(false, lba, 1)?;
        buf.copy_from_slice(&self.data_slice()[..BLOCK_SIZE]);
        Ok(())
    }

    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        if lba >= self.sectors {
            return Err(BlockError::OutOfRange);
        }
        self.data_slice()[..BLOCK_SIZE].copy_from_slice(buf);
        self.rw(true, lba, 1)
    }

    fn read_blocks(&mut self, start_lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let total = buf.len() / BLOCK_SIZE;
        let mut done = 0;
        while done < total {
            let n = (total - done).min(MAX_SECTORS_PER_CMD);
            self.rw(false, start_lba + done as u64, n as u16)?;
            let bytes = n * BLOCK_SIZE;
            buf[done * BLOCK_SIZE..done * BLOCK_SIZE + bytes]
                .copy_from_slice(&self.data_slice()[..bytes]);
            done += n;
        }
        Ok(())
    }

    fn write_blocks(&mut self, start_lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let total = buf.len() / BLOCK_SIZE;
        let mut done = 0;
        while done < total {
            let n = (total - done).min(MAX_SECTORS_PER_CMD);
            let bytes = n * BLOCK_SIZE;
            self.data_slice()[..bytes]
                .copy_from_slice(&buf[done * BLOCK_SIZE..done * BLOCK_SIZE + bytes]);
            self.rw(true, start_lba + done as u64, n as u16)?;
            done += n;
        }
        Ok(())
    }
}

/// Probe PCI for an NVMe controller and bring it up. Returns the disk, or `None` if no
/// NVMe controller is present or bring-up fails (never panics or hangs).
pub fn probe() -> Option<NvmeDisk> {
    let dev = pci::enumerate()
        .into_iter()
        .find(|d| d.class_code == 0x01 && d.subclass == 0x08)?;
    dev.address.enable_bus_master();
    let lo = (dev.address.bar(0) & 0xFFFF_FFF0) as u64;
    let hi = dev.address.bar(1) as u64;
    let bar_phys = lo | (hi << 32);
    if bar_phys == 0 {
        return None;
    }
    // Map the NVMe register BAR (CAP/CC/CSTS + the doorbell array) — it is MMIO, not RAM,
    // so it must be mapped before the first register read or the access page-faults. 64 KiB
    // generously covers the controller registers and a reasonable doorbell stride/count.
    let bar = dma::map_mmio(bar_phys, 0x10000);

    let cap = unsafe { r64(bar + REG_CAP) };
    let dstrd = (cap >> 32) & 0xF;
    let stride = 4u64 << dstrd;

    // Disable the controller, then wait for RDY=0.
    unsafe { w32(bar + REG_CC, 0) };
    let mut spins = 0;
    while unsafe { r32(bar + REG_CSTS) } & CSTS_RDY != 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
        core::hint::spin_loop();
    }

    // Admin queues.
    let asq = dma::alloc(1)?;
    let acq = dma::alloc(1)?;
    let io_sq = dma::alloc(1)?;
    let io_cq = dma::alloc(1)?;
    let ident = dma::alloc(1)?;
    let data = dma::alloc(1)?;

    unsafe {
        w32(bar + REG_AQA, (((QD - 1) as u32) << 16) | (QD - 1) as u32);
        w64(bar + REG_ASQ, asq.phys);
        w64(bar + REG_ACQ, acq.phys);
        // CC: IOSQES=6 (64 B), IOCQES=4 (16 B), MPS=0 (4 KiB), EN=1.
        let cc = (6u32 << 16) | (4u32 << 20) | CC_EN;
        w32(bar + REG_CC, cc);
    }
    let mut spins = 0;
    while unsafe { r32(bar + REG_CSTS) } & CSTS_RDY == 0 {
        spins += 1;
        if spins >= SPIN {
            return None;
        }
        core::hint::spin_loop();
    }

    let mut disk = NvmeDisk {
        bar,
        stride,
        admin: Queue { sq: asq, cq: acq, sq_tail: 0, cq_head: 0, phase: true },
        io: Queue { sq: io_sq, cq: io_cq, sq_tail: 0, cq_head: 0, phase: true },
        data,
        cid: 0,
        sectors: 0,
        faulted: false,
    };

    // Identify Namespace 1 (CNS=0) → NSZE (total sectors) at bytes 0..8.
    let s = {
        let mut s = disk.sqe(0x06, 1, ident.phys);
        s[10] = 0; // CNS = 0 (identify namespace)
        s
    };
    if disk.submit(0, &s) != 0 {
        return None;
    }
    let id = unsafe { core::slice::from_raw_parts(ident.virt as *const u8, 512) };
    disk.sectors = u64::from_le_bytes([
        id[0], id[1], id[2], id[3], id[4], id[5], id[6], id[7],
    ]);
    if disk.sectors == 0 {
        return None;
    }

    // Create the I/O completion queue (qid 1), then the I/O submission queue (qid 1).
    let cq_phys = disk.io.cq.phys;
    let s = {
        let mut s = disk.sqe(0x05, 0, cq_phys); // Create I/O CQ
        s[10] = (((QD - 1) as u32) << 16) | 1; // qsize-1 | qid=1
        s[11] = 1; // PC=1, interrupts disabled
        s
    };
    if disk.submit(0, &s) != 0 {
        return None;
    }
    let sq_phys = disk.io.sq.phys;
    let s = {
        let mut s = disk.sqe(0x01, 0, sq_phys); // Create I/O SQ
        s[10] = (((QD - 1) as u32) << 16) | 1; // qsize-1 | qid=1
        s[11] = (1u32 << 16) | 1; // CQID=1 | PC=1
        s
    };
    if disk.submit(0, &s) != 0 {
        return None;
    }

    crate::serial_println!(
        "[nvme] controller up (bar={:#x}), namespace 1: {} sectors ({} MiB)",
        bar_phys,
        disk.sectors,
        disk.sectors * BLOCK_SIZE as u64 / (1024 * 1024)
    );
    Some(disk)
}

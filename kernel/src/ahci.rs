//! AHCI (SATA) block driver — real read/write to a SATA disk, so DominionOS persists
//! (incl. the boot/debug log) to an internal drive on bare metal, not just to a virtio
//! disk under QEMU. Polling, single command slot — simple and robust; the log-persist
//! and object-store paths don't need deep queueing.
//!
//! Bring-up follows the AHCI 1.3 spec: find the HBA on PCI (class 01:06:01), map its
//! ABAR (BAR5) through the bootloader's full physical map, enable AHCI, pick a port with
//! a SATA disk present, allocate the command list / FIS / command table / a DMA bounce
//! buffer, IDENTIFY the device for its capacity, then issue READ/WRITE DMA EXT commands.
//! Every wait is bounded by a spin budget so a wedged controller can never hang boot.

use crate::dma::{self, DmaRegion};
use crate::pci;
use dominion_core::persist::{BlockDevice, BlockError, BLOCK_SIZE};

// HBA (ABAR) global registers.
const HBA_GHC: u64 = 0x04; // global host control
const HBA_PI: u64 = 0x0C; // ports implemented
const GHC_AE: u32 = 1 << 31; // AHCI enable
const PORT_BASE: u64 = 0x100;
const PORT_STRIDE: u64 = 0x80;

// Per-port registers (offset within the port block).
const P_CLB: u64 = 0x00; // command list base (low)
const P_CLBU: u64 = 0x04;
const P_FB: u64 = 0x08; // FIS base (low)
const P_FBU: u64 = 0x0C;
const P_IS: u64 = 0x10; // interrupt status
const P_CMD: u64 = 0x18; // command and status
const P_TFD: u64 = 0x20; // task file data
const P_SIG: u64 = 0x24; // signature
const P_SSTS: u64 = 0x28; // SATA status
const P_CI: u64 = 0x38; // command issue

const CMD_ST: u32 = 1 << 0; // start
const CMD_FRE: u32 = 1 << 4; // FIS receive enable
const CMD_FR: u32 = 1 << 14; // FIS receive running
const CMD_CR: u32 = 1 << 15; // command list running

const TFD_BSY: u32 = 1 << 7;
const TFD_DRQ: u32 = 1 << 3;
const TFD_ERR: u32 = 1 << 0;
const IS_TFES: u32 = 1 << 30; // task file error

const SIG_SATA: u32 = 0x0000_0101;

const ATA_READ_DMA_EX: u8 = 0x25;
const ATA_WRITE_DMA_EX: u8 = 0x35;
const ATA_IDENTIFY: u8 = 0xEC;

// Fail-fast bound on hardware-ready spin loops (~0.3 s of slow MMIO reads); was
// 5_000_000. Generous for a healthy controller, but gives up quickly on metal that
// never responds rather than stalling the boot.
const SPIN: u32 = 1_000_000;
/// Bounce buffer size in sectors (one PRDT entry covers it comfortably).
const BOUNCE_SECTORS: usize = 64;

#[inline]
unsafe fn r32(addr: u64) -> u32 {
    core::ptr::read_volatile(addr as *const u32)
}
#[inline]
unsafe fn w32(addr: u64, v: u32) {
    core::ptr::write_volatile(addr as *mut u32, v);
}

/// A SATA disk reachable through one AHCI port.
pub struct AhciDisk {
    port: u64, // virtual base of the port register block
    cmd_list: DmaRegion,
    _fis: DmaRegion,
    ctba: DmaRegion,
    bounce: DmaRegion,
    sectors: u64,
}

impl AhciDisk {
    fn pr(&self, off: u64) -> u32 {
        unsafe { r32(self.port + off) }
    }
    fn pw(&self, off: u64, v: u32) {
        unsafe { w32(self.port + off, v) }
    }

    fn stop(&self) {
        let mut cmd = self.pr(P_CMD);
        cmd &= !CMD_ST;
        cmd &= !CMD_FRE;
        self.pw(P_CMD, cmd);
        let mut spins = 0;
        while self.pr(P_CMD) & (CMD_CR | CMD_FR) != 0 && spins < SPIN {
            spins += 1;
            core::hint::spin_loop();
        }
    }

    fn start(&self) {
        let mut spins = 0;
        while self.pr(P_CMD) & CMD_CR != 0 && spins < SPIN {
            spins += 1;
            core::hint::spin_loop();
        }
        let mut cmd = self.pr(P_CMD);
        cmd |= CMD_FRE;
        cmd |= CMD_ST;
        self.pw(P_CMD, cmd);
    }

    /// Issue one ATA command on slot 0 with `count` sectors of DMA into/out of the
    /// bounce buffer, blocking (polled) until completion. `write` selects direction.
    fn issue(&self, cmd: u8, lba: u64, count: u16, write: bool) -> Result<(), BlockError> {
        // Wait for the port to be idle.
        let mut spins = 0;
        while self.pr(P_TFD) & (TFD_BSY | TFD_DRQ) != 0 && spins < SPIN {
            spins += 1;
            core::hint::spin_loop();
        }
        if spins >= SPIN {
            return Err(BlockError::DeviceFault);
        }
        self.pw(P_IS, !0); // clear pending interrupts

        // Command header (slot 0) at the start of the command list.
        let hdr = self.cmd_list.virt as *mut u32;
        let bytes = (count as u32).max(1) * BLOCK_SIZE as u32;
        let cfl = 5u32; // H2D register FIS = 20 bytes = 5 DWORDs
        let dw0 = cfl | ((write as u32) << 6) | (1u32 << 16); // PRDTL = 1
        unsafe {
            core::ptr::write_volatile(hdr, dw0);
            core::ptr::write_volatile(hdr.add(1), 0); // PRDBC
            core::ptr::write_volatile(hdr.add(2), self.ctba.phys as u32); // CTBA low
            core::ptr::write_volatile(hdr.add(3), (self.ctba.phys >> 32) as u32); // CTBA high
        }

        // Command table: clear, build the H2D FIS, then the single PRDT entry.
        self.ctba.zero();
        let cfis = self.ctba.virt as *mut u8;
        unsafe {
            core::ptr::write_volatile(cfis.add(0), 0x27); // FIS_TYPE_REG_H2D
            core::ptr::write_volatile(cfis.add(1), 0x80); // C=1 (command)
            core::ptr::write_volatile(cfis.add(2), cmd);
            core::ptr::write_volatile(cfis.add(3), 0); // features low
            core::ptr::write_volatile(cfis.add(4), lba as u8);
            core::ptr::write_volatile(cfis.add(5), (lba >> 8) as u8);
            core::ptr::write_volatile(cfis.add(6), (lba >> 16) as u8);
            core::ptr::write_volatile(cfis.add(7), 0x40); // device: LBA mode
            core::ptr::write_volatile(cfis.add(8), (lba >> 24) as u8);
            core::ptr::write_volatile(cfis.add(9), (lba >> 32) as u8);
            core::ptr::write_volatile(cfis.add(10), (lba >> 40) as u8);
            core::ptr::write_volatile(cfis.add(11), 0); // features high
            core::ptr::write_volatile(cfis.add(12), count as u8);
            core::ptr::write_volatile(cfis.add(13), (count >> 8) as u8);
            // PRDT entry 0 at offset 0x80.
            let prdt = (self.ctba.virt + 0x80) as *mut u32;
            core::ptr::write_volatile(prdt, self.bounce.phys as u32); // DBA low
            core::ptr::write_volatile(prdt.add(1), (self.bounce.phys >> 32) as u32); // DBA high
            core::ptr::write_volatile(prdt.add(2), 0);
            core::ptr::write_volatile(prdt.add(3), bytes - 1); // byte count - 1, no interrupt
        }

        // Issue on slot 0 and poll to completion.
        self.pw(P_CI, 1);
        let mut spins = 0;
        while self.pr(P_CI) & 1 != 0 && spins < SPIN {
            if self.pr(P_IS) & IS_TFES != 0 {
                return Err(BlockError::DeviceFault);
            }
            spins += 1;
            core::hint::spin_loop();
        }
        if spins >= SPIN || self.pr(P_TFD) & TFD_ERR != 0 {
            return Err(BlockError::DeviceFault);
        }
        Ok(())
    }

    fn bounce_slice(&self) -> &'static mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.bounce.virt as *mut u8, self.bounce.size) }
    }
}

impl BlockDevice for AhciDisk {
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
        self.issue(ATA_READ_DMA_EX, lba, 1, false)?;
        buf.copy_from_slice(&self.bounce_slice()[..BLOCK_SIZE]);
        Ok(())
    }

    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        if lba >= self.sectors {
            return Err(BlockError::OutOfRange);
        }
        self.bounce_slice()[..BLOCK_SIZE].copy_from_slice(buf);
        self.issue(ATA_WRITE_DMA_EX, lba, 1, true)
    }

    fn read_blocks(&mut self, start_lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let total = buf.len() / BLOCK_SIZE;
        let mut done = 0;
        while done < total {
            let n = (total - done).min(BOUNCE_SECTORS);
            self.issue(ATA_READ_DMA_EX, start_lba + done as u64, n as u16, false)?;
            let bytes = n * BLOCK_SIZE;
            buf[done * BLOCK_SIZE..done * BLOCK_SIZE + bytes]
                .copy_from_slice(&self.bounce_slice()[..bytes]);
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
            let n = (total - done).min(BOUNCE_SECTORS);
            let bytes = n * BLOCK_SIZE;
            self.bounce_slice()[..bytes]
                .copy_from_slice(&buf[done * BLOCK_SIZE..done * BLOCK_SIZE + bytes]);
            self.issue(ATA_WRITE_DMA_EX, start_lba + done as u64, n as u16, true)?;
            done += n;
        }
        Ok(())
    }
}

/// Probe PCI for an AHCI controller with a SATA disk and bring it up. Returns the disk,
/// or `None` if no AHCI controller / no SATA disk is present (never panics or hangs).
pub fn probe() -> Option<AhciDisk> {
    let dev = pci::enumerate()
        .into_iter()
        .find(|d| d.class_code == 0x01 && d.subclass == 0x06)?;
    dev.address.enable_bus_master();
    let abar_phys = (dev.address.bar(5) & 0xFFFF_FFF0) as u64;
    if abar_phys == 0 {
        return None;
    }
    // Map the AHCI register BAR (ABAR) — MMIO, not RAM, so it must be mapped before the
    // first read. 8 KiB covers the generic regs + all 32 ports' register blocks.
    let abar = dma::map_mmio(abar_phys, 0x2000);

    // Enable AHCI mode.
    unsafe {
        w32(abar + HBA_GHC, r32(abar + HBA_GHC) | GHC_AE);
    }
    let pi = unsafe { r32(abar + HBA_PI) };

    // Find the first implemented port with a SATA disk present and active.
    let mut port_virt = 0u64;
    for i in 0..32u64 {
        if pi & (1 << i) == 0 {
            continue;
        }
        let pbase = abar + PORT_BASE + i * PORT_STRIDE;
        let ssts = unsafe { r32(pbase + P_SSTS) };
        let det = ssts & 0x0F;
        let ipm = (ssts >> 8) & 0x0F;
        let sig = unsafe { r32(pbase + P_SIG) };
        if det == 3 && ipm == 1 && sig == SIG_SATA {
            port_virt = pbase;
            break;
        }
    }
    if port_virt == 0 {
        return None;
    }

    // Allocate the per-port structures + a DMA bounce buffer.
    let cmd_list = dma::alloc(1)?; // command list (1 KiB used; page is 1 KiB-aligned)
    let fis = dma::alloc(1)?; // received-FIS area (256 B)
    let ctba = dma::alloc(1)?; // command table + PRDT
    let bounce = dma::alloc(BOUNCE_SECTORS * BLOCK_SIZE / 4096 + 1)?;

    let disk = AhciDisk { port: port_virt, cmd_list, _fis: fis, ctba, bounce, sectors: 0 };

    // Program the command-list and FIS base addresses, then (re)start the port.
    disk.stop();
    disk.pw(P_CLB, cmd_list.phys as u32);
    disk.pw(P_CLBU, (cmd_list.phys >> 32) as u32);
    disk.pw(P_FB, fis.phys as u32);
    disk.pw(P_FBU, (fis.phys >> 32) as u32);
    disk.pw(P_IS, !0);
    disk.start();

    // IDENTIFY DEVICE → capacity (LBA48 total sectors at words 100..104).
    let mut disk = disk;
    if disk.issue(ATA_IDENTIFY, 0, 1, false).is_err() {
        return None;
    }
    let id = disk.bounce_slice();
    let lba48 = u64::from_le_bytes([
        id[200], id[201], id[202], id[203], id[204], id[205], id[206], id[207],
    ]);
    let lba28 = u32::from_le_bytes([id[120], id[121], id[122], id[123]]) as u64;
    disk.sectors = if lba48 != 0 { lba48 } else { lba28 };
    if disk.sectors == 0 {
        return None;
    }
    crate::serial_println!(
        "[ahci] SATA disk on port (abar={:#x}), {} sectors ({} MiB)",
        abar_phys,
        disk.sectors,
        disk.sectors * BLOCK_SIZE as u64 / (1024 * 1024)
    );
    Some(disk)
}

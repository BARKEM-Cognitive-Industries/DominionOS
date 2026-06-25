//! The virtio-blk driver — the device half of **M1 (persistence)**.
//!
//! A virtio block device exposes a linear array of 512-byte sectors. Each request
//! is a three-part descriptor chain: a 16-byte header (operation + sector),
//! the data buffer, and a one-byte status the device writes back. On top of this
//! the [`persist`](dominion_core::persist) layer keeps the content-addressed object
//! graph on disk so commits survive reboot.

use alloc::boxed::Box;
use crate::dma::{self, DmaRegion};
use crate::pci;
use crate::virtio::{Buf, VirtQueue, VirtioTransport};
use dominion_core::persist::{BlockDevice, BlockError, RamDisk, BLOCK_SIZE};
use spin::Mutex;

/// The system's block device, probed once at boot.  `None` if no disk of any supported
/// kind is attached (the system then runs entirely in RAM).
///
/// Each concrete driver (`VirtioBlk`, `AhciDisk`, `NvmeDisk`, `UsbMsc`) implements
/// `BlockDevice` directly and is stored as a trait object, so adding a new backend
/// requires no changes here — just `impl BlockDevice for NewDrive` in its own module.
static DEVICE: Mutex<Option<Box<dyn BlockDevice + Send>>> = Mutex::new(None);

/// Set when the primary device is a USB mass-storage controller (determined at probe time
/// and never changes afterwards). Used to avoid re-probing the same xHCI controller for
/// the log device, which would corrupt it on real hardware.
static PRIMARY_IS_USB: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Probe for a writable disk and install it as the global device, in preference order:
/// virtio-blk (QEMU), then AHCI/SATA, then NVMe. Safe to call once at boot; later calls
/// are ignored if a device is already present.
pub fn init_global() -> bool {
    let mut guard = DEVICE.lock();
    if guard.is_some() {
        return true;
    }
    // Per-driver markers (serial + screen) so a wedge in any one probe is visible on
    // bare metal — the LAST one printed is the driver that hung.
    macro_rules! mark {
        ($($a:tt)*) => {{ crate::serial_println!($($a)*); crate::println!($($a)*); }};
    }
    mark!("[storage]  trying virtio-blk ...");
    if let Some(v) = VirtioBlk::init() {
        *guard = Some(Box::new(v));
    } else {
        mark!("[storage]  trying AHCI/SATA ...");
        if let Some(a) = crate::ahci::probe() {
            *guard = Some(Box::new(a));
        } else {
            mark!("[storage]  trying NVMe ...");
            if let Some(n) = crate::nvme::probe() {
                *guard = Some(Box::new(n));
            } else {
                // USB (xHCI) auto-probe is DISABLED by default. Re-initialising the very
                // controller the machine booted from wedges on real hardware (an endless
                // port-event storm during/after the controller reset), and USB-MSC
                // persistence is non-essential — the firmware already loaded the kernel
                // into RAM, so nothing here needs USB. Storage falls back to RAM when no
                // virtio/AHCI/NVMe disk is present. Opt back in with the `usb_storage`
                // feature once the xHCI bring-up is hardened.
                #[cfg(feature = "usb_storage")]
                {
                    mark!("[storage]  trying USB (xHCI) ...");
                    if let Some(u) = crate::xhci::probe() {
                        PRIMARY_IS_USB.store(true, core::sync::atomic::Ordering::Relaxed);
                        *guard = Some(Box::new(u));
                    }
                }
                #[cfg(not(feature = "usb_storage"))]
                mark!("[storage]  USB (xHCI) probe skipped (disabled by default)");
            }
        }
    }
    mark!("[storage]  probe complete");
    guard.is_some()
}

/// Run `f` with a block device: the real virtio-blk disk if present, otherwise a
/// transient RAM disk. The `bool` argument is `true` when the device is real.
/// This lets the same persistence code be exercised with or without hardware.
pub fn with_block_device<R>(f: impl FnOnce(&mut dyn BlockDevice, bool) -> R) -> R {
    let mut guard = DEVICE.lock();
    match guard.as_mut() {
        Some(dev) => f(&mut **dev, true),
        None => {
            let mut ram = RamDisk::new(4096);
            f(&mut ram, false)
        }
    }
}

/// A **dedicated removable-USB device** used as the preferred target for the debug log,
/// so the boot/run log lands on the USB you can pull out — even when the primary store is
/// an internal AHCI/NVMe disk. `None` when there is no separate USB (or the primary disk
/// already *is* the USB, in which case the log uses the primary).
static LOG_USB: Mutex<Option<crate::xhci::UsbMsc>> = Mutex::new(None);

fn primary_is_usb() -> bool {
    PRIMARY_IS_USB.load(core::sync::atomic::Ordering::Relaxed)
}

/// Probe once (at boot, after [`init_global`]) for a removable USB mass-storage device to
/// use as the preferred log target. Skipped if the primary device is already that USB
/// (probing the same xHCI controller twice would corrupt it). Returns true if a dedicated
/// USB log device was installed.
pub fn init_log_device() -> bool {
    // USB log target is gated on the same `usb_storage` opt-in as the USB block device:
    // probing the boot xHCI controller wedges on real hardware. With it off, the log
    // simply targets the primary block device (or RAM) instead.
    #[cfg(feature = "usb_storage")]
    {
        if primary_is_usb() {
            return false;
        }
        let mut lg = LOG_USB.lock();
        if lg.is_none() {
            *lg = crate::xhci::probe();
        }
        lg.is_some()
    }
    #[cfg(not(feature = "usb_storage"))]
    false
}

/// Is the debug-log target a USB device (a dedicated removable USB, or a USB primary)?
pub fn log_is_usb() -> bool {
    LOG_USB.lock().is_some() || primary_is_usb()
}

/// Run `f` against the preferred **log** device: a dedicated removable USB if one was
/// found, otherwise the primary block device (which itself may be a USB, or an internal
/// disk, or a RAM disk). This is what the bootlog persist path uses.
pub fn with_log_device<R>(f: impl FnOnce(&mut dyn BlockDevice, bool) -> R) -> R {
    let mut lg = LOG_USB.lock();
    if let Some(u) = lg.as_mut() {
        return f(u, true);
    }
    drop(lg);
    with_block_device(f)
}

/// Capacity of the installed device in 512-byte sectors, or 0 if none.
pub fn capacity_sectors() -> u64 {
    DEVICE.lock().as_ref().map(|d| d.block_count()).unwrap_or(0)
}

/// Is a real disk (virtio/AHCI/NVMe) installed?
pub fn present() -> bool {
    DEVICE.lock().is_some()
}

/// virtio block subsystem id (PCI subsystem field).
const VIRTIO_SUBSYSTEM_BLOCK: u16 = 2;

const VIRTIO_BLK_T_IN: u32 = 0; // read (device -> guest)
const VIRTIO_BLK_T_OUT: u32 = 1; // write (guest -> device)

// Layout of our single scratch DMA page for one in-flight request.
const OFF_HEADER: u64 = 0; // 16 bytes
const OFF_STATUS: u64 = 16; // 1 byte
const OFF_DATA: u64 = 512; // 512 bytes (sector-aligned for clarity)

// Batched I/O scratch: a separate DMA region carved into independent per-request
// slots so many requests can be in flight at once. Each slot mirrors the single-request
// layout (header / status / data) at a fixed stride, so every slot has stable physical
// addresses we can hand the device. Batching is what turns a save/restore from one
// notify-and-spin per 512-byte sector into one notify for a whole run of them.
const SLOT_STRIDE: u64 = 1024; // bytes per slot (512 data + header/status, padded)
const SLOT_HEADER: u64 = 0; // 16 bytes
const SLOT_STATUS: u64 = 16; // 1 byte
const SLOT_DATA: u64 = 512; // 512 bytes
const BATCH_SLOTS: usize = 64; // upper bound on requests staged per batch
const BATCH_PAGES: usize = BATCH_SLOTS * SLOT_STRIDE as usize / 4096; // = 16 pages

pub struct VirtioBlk {
    transport: VirtioTransport,
    queue: VirtQueue,
    scratch: DmaRegion,
    /// Multi-slot scratch for pipelined batch I/O; `None` if its DMA region could not
    /// be allocated (the driver then falls back to the single-request path).
    batch_scratch: Option<DmaRegion>,
    /// Maximum blocks to stage in one batch — bounded by both [`BATCH_SLOTS`] and the
    /// virtqueue depth (each request needs three descriptors).
    batch: usize,
    capacity_sectors: u64,
}

impl VirtioBlk {
    /// Probe PCI for a virtio-blk device and bring it up. Returns `None` if no
    /// such device is attached.
    pub fn init() -> Option<VirtioBlk> {
        let dev = pci::find_virtio(VIRTIO_SUBSYSTEM_BLOCK)?;
        dev.address.enable_bus_master();

        // Legacy virtio uses the I/O BAR (BAR0); mask off the low type bits.
        let bar0 = dev.address.bar(0);
        let io_base = (bar0 & 0xFFFC) as u16;
        let transport = unsafe { VirtioTransport::new(io_base) };

        // Accept no optional features — basic read/write works in pure legacy mode.
        transport.begin(0);
        let queue = VirtQueue::new(&transport, 0)?;
        transport.finish();

        let capacity_sectors = transport.config_u64(0);
        let scratch = dma::alloc(1)?;
        // Pipelined-batch scratch. Optional: if it can't be allocated the driver still
        // works via the single-request path, just without the batching speedup. The
        // batch is capped at a third of the queue depth — each request is a 3-descriptor
        // chain, and a whole batch must fit before the device drains any of it.
        let queue_batch = (queue.size as usize / 3).max(1);
        let (batch_scratch, batch) = match dma::alloc(BATCH_PAGES) {
            Some(region) => (Some(region), core::cmp::min(BATCH_SLOTS, queue_batch)),
            None => (None, 0),
        };
        Some(VirtioBlk {
            transport,
            queue,
            scratch,
            batch_scratch,
            batch,
            capacity_sectors,
        })
    }

    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    fn request(&mut self, op: u32, sector: u64, buf: &mut [u8]) -> bool {
        if buf.len() != BLOCK_SIZE {
            return false;
        }
        let base = self.scratch.virt;
        unsafe {
            // Header: type, reserved, sector.
            core::ptr::write_volatile((base + OFF_HEADER) as *mut u32, op);
            core::ptr::write_volatile((base + OFF_HEADER + 4) as *mut u32, 0);
            core::ptr::write_volatile((base + OFF_HEADER + 8) as *mut u64, sector);
            // Poison the status byte so we can tell the device wrote it.
            core::ptr::write_volatile((base + OFF_STATUS) as *mut u8, 0xFF);
        }
        let data_ptr = (base + OFF_DATA) as *mut u8;
        if op == VIRTIO_BLK_T_OUT {
            unsafe { core::ptr::copy_nonoverlapping(buf.as_ptr(), data_ptr, BLOCK_SIZE) };
        }

        let bufs = [
            Buf { phys: self.scratch.phys + OFF_HEADER, len: 16, device_writable: false },
            // The device writes the data buffer on read, reads it on write.
            Buf { phys: self.scratch.phys + OFF_DATA, len: BLOCK_SIZE as u32, device_writable: op == VIRTIO_BLK_T_IN },
            Buf { phys: self.scratch.phys + OFF_STATUS, len: 1, device_writable: true },
        ];

        if self.queue.submit_and_wait(&self.transport, &bufs).is_none() {
            return false;
        }

        let status = unsafe { core::ptr::read_volatile((base + OFF_STATUS) as *const u8) };
        if status != 0 {
            return false;
        }
        if op == VIRTIO_BLK_T_IN {
            unsafe { core::ptr::copy_nonoverlapping(data_ptr, buf.as_mut_ptr(), BLOCK_SIZE) };
        }
        true
    }

    /// Stage one request's header + poisoned status into batch slot `s` for sector `lba`.
    fn stage_header(region: &DmaRegion, s: usize, op: u32, lba: u64) {
        let slot = region.virt + s as u64 * SLOT_STRIDE;
        unsafe {
            core::ptr::write_volatile((slot + SLOT_HEADER) as *mut u32, op);
            core::ptr::write_volatile((slot + SLOT_HEADER + 4) as *mut u32, 0);
            core::ptr::write_volatile((slot + SLOT_HEADER + 8) as *mut u64, lba);
            core::ptr::write_volatile((slot + SLOT_STATUS) as *mut u8, 0xFF);
        }
    }

    /// Publish `count` already-staged slots on the available ring, notify the device
    /// **once**, wait for every request to complete, and verify each status byte.
    /// Returns false if descriptors are exhausted or any request reports a fault.
    /// `is_write` only sets the data descriptor's direction; the slots' headers/data
    /// must already be staged by the caller.
    fn run_batch(&mut self, count: usize, is_write: bool) -> bool {
        let region = match self.batch_scratch {
            Some(r) => r,
            None => return false,
        };
        for s in 0..count {
            let slot_phys = region.phys + s as u64 * SLOT_STRIDE;
            let bufs = [
                Buf { phys: slot_phys + SLOT_HEADER, len: 16, device_writable: false },
                // The device writes the data buffer on read, reads it on write.
                Buf { phys: slot_phys + SLOT_DATA, len: BLOCK_SIZE as u32, device_writable: !is_write },
                Buf { phys: slot_phys + SLOT_STATUS, len: 1, device_writable: true },
            ];
            // count <= batch <= queue_depth/3 and the previous batch fully drained,
            // so the free list always has room; treat exhaustion as a fault anyway.
            if self.queue.add(&bufs).is_none() {
                return false;
            }
        }
        // One notify for the whole batch, then drain all completions. Each drain
        // reclaims every chain the device has published so far with a single
        // used-ring read, so a burst of completions costs one read, not `count`.
        self.queue.kick(&self.transport);
        let mut completed = 0;
        while completed < count {
            let freed = self.queue.poll_drain();
            if freed == 0 {
                core::hint::spin_loop();
            } else {
                completed += freed;
            }
        }
        // All `count` requests are done now, so each status byte is final regardless of
        // the order completions arrived in.
        for s in 0..count {
            let slot = region.virt + s as u64 * SLOT_STRIDE;
            if unsafe { core::ptr::read_volatile((slot + SLOT_STATUS) as *const u8) } != 0 {
                return false;
            }
        }
        true
    }
}

impl BlockDevice for VirtioBlk {
    fn block_count(&self) -> u64 {
        self.capacity_sectors
    }

    fn read_block(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if lba >= self.capacity_sectors {
            return Err(BlockError::OutOfRange);
        }
        if self.request(VIRTIO_BLK_T_IN, lba, buf) {
            Ok(())
        } else {
            Err(BlockError::DeviceFault)
        }
    }

    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if lba >= self.capacity_sectors {
            return Err(BlockError::OutOfRange);
        }
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        // request() needs &mut for the read path; copy through a local buffer.
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        if self.request(VIRTIO_BLK_T_OUT, lba, &mut tmp) {
            Ok(())
        } else {
            Err(BlockError::DeviceFault)
        }
    }

    fn read_blocks(&mut self, start_lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let nblocks = (buf.len() / BLOCK_SIZE) as u64;
        if start_lba + nblocks > self.capacity_sectors {
            return Err(BlockError::OutOfRange);
        }
        // No batch region — fall back to one request per sector.
        let region = match self.batch_scratch {
            Some(r) if self.batch >= 1 => r,
            _ => {
                for (i, chunk) in buf.chunks_mut(BLOCK_SIZE).enumerate() {
                    self.read_block(start_lba + i as u64, chunk)?;
                }
                return Ok(());
            }
        };
        let nblocks = nblocks as usize;
        let mut done = 0usize;
        while done < nblocks {
            let count = core::cmp::min(self.batch, nblocks - done);
            for s in 0..count {
                Self::stage_header(&region, s, VIRTIO_BLK_T_IN, start_lba + (done + s) as u64);
            }
            if !self.run_batch(count, false) {
                return Err(BlockError::DeviceFault);
            }
            // Copy the device-filled data back out of each slot.
            for s in 0..count {
                let off = (done + s) * BLOCK_SIZE;
                let slot = region.virt + s as u64 * SLOT_STRIDE;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        (slot + SLOT_DATA) as *const u8,
                        buf[off..off + BLOCK_SIZE].as_mut_ptr(),
                        BLOCK_SIZE,
                    );
                }
            }
            done += count;
        }
        Ok(())
    }

    fn write_blocks(&mut self, start_lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        let nblocks = (buf.len() / BLOCK_SIZE) as u64;
        if start_lba + nblocks > self.capacity_sectors {
            return Err(BlockError::OutOfRange);
        }
        let region = match self.batch_scratch {
            Some(r) if self.batch >= 1 => r,
            _ => {
                for (i, chunk) in buf.chunks(BLOCK_SIZE).enumerate() {
                    self.write_block(start_lba + i as u64, chunk)?;
                }
                return Ok(());
            }
        };
        let nblocks = nblocks as usize;
        let mut done = 0usize;
        while done < nblocks {
            let count = core::cmp::min(self.batch, nblocks - done);
            // Stage header + outgoing data for each request in this batch.
            for s in 0..count {
                Self::stage_header(&region, s, VIRTIO_BLK_T_OUT, start_lba + (done + s) as u64);
                let off = (done + s) * BLOCK_SIZE;
                let slot = region.virt + s as u64 * SLOT_STRIDE;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        buf[off..off + BLOCK_SIZE].as_ptr(),
                        (slot + SLOT_DATA) as *mut u8,
                        BLOCK_SIZE,
                    );
                }
            }
            if !self.run_batch(count, true) {
                return Err(BlockError::DeviceFault);
            }
            done += count;
        }
        Ok(())
    }
}

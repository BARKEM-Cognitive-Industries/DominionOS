//! Virtio over legacy PCI — the transport at the centre of the M3 driver
//! framework and the M1 persistence story.
//!
//! Virtio is the paravirtual device interface every hypervisor (QEMU included)
//! speaks. We implement the *legacy* I/O-port interface and a split virtqueue:
//! the guest publishes buffer-descriptor chains on an "available" ring, kicks the
//! device, and the device reports completions on a "used" ring. This module is
//! device-agnostic — block (`block.rs`) and any future net device layer their
//! semantics on top of [`VirtioTransport`] + [`VirtQueue`].
//!
//! We poll the used ring rather than take completion interrupts: under TCG this
//! is simple, deterministic, and entirely adequate for the synchronous block I/O
//! the persistence layer needs.

use crate::dma::{self, DmaRegion};
use alloc::vec::Vec;
use core::sync::atomic::{fence, Ordering};
use x86_64::instructions::port::Port;

// Device status bits (virtio spec §2.1).
pub const STATUS_ACKNOWLEDGE: u8 = 1;
pub const STATUS_DRIVER: u8 = 2;
pub const STATUS_DRIVER_OK: u8 = 4;
pub const STATUS_FAILED: u8 = 128;

// Descriptor flags.
const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

/// Maximum Ethernet frame payload that any virtio-net descriptor may carry.
/// Jumbo frames aside, standard Ethernet MTU is 1500 B plus the virtio-net
/// header (12 B) — cap at the conventional maximum transmission unit so that
/// a rogue `len` field cannot drive arbitrarily large DMA or allocations.
const MAX_PACKET_SIZE: u32 = 65535;

/// Maximum descriptor-chain depth we will traverse before treating the chain
/// as corrupt and breaking out. Virtio queues are at most 32 768 entries deep
/// (spec §2.6); 128 is generous for any legitimate chain and tight enough to
/// prevent infinite-loop DoS from a forged NEXT cycle.
const MAX_CHAIN_DEPTH: usize = 128;

// Legacy virtio-pci I/O register offsets from the I/O BAR base (no MSI-X).
const REG_DEVICE_FEATURES: u16 = 0x00;
const REG_GUEST_FEATURES: u16 = 0x04;
const REG_QUEUE_PFN: u16 = 0x08;
const REG_QUEUE_SIZE: u16 = 0x0C;
const REG_QUEUE_SELECT: u16 = 0x0E;
const REG_QUEUE_NOTIFY: u16 = 0x10;
const REG_DEVICE_STATUS: u16 = 0x12;
const REG_ISR_STATUS: u16 = 0x13;
/// Device-specific config starts here when MSI-X is disabled.
const REG_DEVICE_CONFIG: u16 = 0x14;

#[repr(C)]
struct VirtqDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

/// The legacy virtio-pci transport: a thin wrapper over the device's I/O BAR.
#[derive(Clone, Copy)]
pub struct VirtioTransport {
    io_base: u16,
}

impl VirtioTransport {
    /// # Safety
    /// `io_base` must be the device's legacy I/O BAR base address.
    pub unsafe fn new(io_base: u16) -> Self {
        VirtioTransport { io_base }
    }

    fn read_u8(&self, reg: u16) -> u8 {
        unsafe { Port::<u8>::new(self.io_base + reg).read() }
    }
    fn write_u8(&self, reg: u16, v: u8) {
        unsafe { Port::<u8>::new(self.io_base + reg).write(v) }
    }
    fn read_u16(&self, reg: u16) -> u16 {
        unsafe { Port::<u16>::new(self.io_base + reg).read() }
    }
    fn write_u16(&self, reg: u16, v: u16) {
        unsafe { Port::<u16>::new(self.io_base + reg).write(v) }
    }
    fn read_u32(&self, reg: u16) -> u32 {
        unsafe { Port::<u32>::new(self.io_base + reg).read() }
    }
    fn write_u32(&self, reg: u16, v: u32) {
        unsafe { Port::<u32>::new(self.io_base + reg).write(v) }
    }

    pub fn status(&self) -> u8 {
        self.read_u8(REG_DEVICE_STATUS)
    }
    pub fn set_status(&self, status: u8) {
        self.write_u8(REG_DEVICE_STATUS, status);
    }
    /// OR additional bits into the status register.
    pub fn add_status(&self, bits: u8) {
        let s = self.status();
        self.set_status(s | bits);
    }
    pub fn reset(&self) {
        self.set_status(0);
    }

    pub fn device_features(&self) -> u32 {
        self.read_u32(REG_DEVICE_FEATURES)
    }
    pub fn set_guest_features(&self, features: u32) {
        self.write_u32(REG_GUEST_FEATURES, features);
    }

    pub fn select_queue(&self, queue: u16) {
        self.write_u16(REG_QUEUE_SELECT, queue);
    }
    pub fn queue_size(&self) -> u16 {
        self.read_u16(REG_QUEUE_SIZE)
    }
    pub fn set_queue_pfn(&self, pfn: u32) {
        self.write_u32(REG_QUEUE_PFN, pfn);
    }
    pub fn notify(&self, queue: u16) {
        self.write_u16(REG_QUEUE_NOTIFY, queue);
    }
    pub fn isr(&self) -> u8 {
        self.read_u8(REG_ISR_STATUS)
    }

    /// Read a byte of device-specific configuration.
    pub fn config_u8(&self, offset: u16) -> u8 {
        self.read_u8(REG_DEVICE_CONFIG + offset)
    }
    pub fn config_u32(&self, offset: u16) -> u32 {
        self.read_u32(REG_DEVICE_CONFIG + offset)
    }
    /// Read a 64-bit device-config field as two 32-bit halves (legacy config is
    /// little-endian; low dword first).
    pub fn config_u64(&self, offset: u16) -> u64 {
        let lo = self.config_u32(offset) as u64;
        let hi = self.config_u32(offset + 4) as u64;
        (hi << 32) | lo
    }

    /// Perform the reset → acknowledge → driver handshake and negotiate features
    /// (we accept the intersection of `wanted` and what the device offers).
    /// Returns the accepted feature set.
    pub fn begin(&self, wanted: u32) -> u32 {
        self.reset();
        self.add_status(STATUS_ACKNOWLEDGE);
        self.add_status(STATUS_DRIVER);
        let accepted = self.device_features() & wanted;
        self.set_guest_features(accepted);
        accepted
    }

    /// Signal the device that the driver is ready to operate.
    pub fn finish(&self) {
        self.add_status(STATUS_DRIVER_OK);
    }
}

/// A buffer to place in a descriptor chain.
pub struct Buf {
    pub phys: u64,
    pub len: u32,
    /// True if the *device* writes this buffer (e.g. a read's data / status byte).
    pub device_writable: bool,
}

/// A single split virtqueue: descriptor table + available ring + used ring in one
/// contiguous DMA region, as the legacy layout requires.
pub struct VirtQueue {
    pub index: u16,
    pub size: u16,
    region: DmaRegion,
    desc: *mut VirtqDesc,
    avail: *mut u8,
    used: *mut u8,
    free_head: u16,
    num_free: u16,
    last_used_idx: u16,
    /// Shadow of the available-ring index. The guest is the sole writer of
    /// `avail.idx`, so we keep an authoritative copy here and write it through to
    /// the ring on each publish — saving a volatile read-back of device-shared
    /// memory on every submission.
    avail_idx: u16,
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

impl VirtQueue {
    /// Set up queue `index` on `transport`: read its size, allocate the ring DMA
    /// region, initialise the descriptor free-list, and register its address with
    /// the device. Returns `None` if the device has no such queue.
    pub fn new(transport: &VirtioTransport, index: u16) -> Option<VirtQueue> {
        transport.select_queue(index);
        let size = transport.queue_size();
        if size == 0 {
            return None;
        }
        let qsz = size as usize;
        let desc_bytes = 16 * qsz;
        let avail_bytes = 6 + 2 * qsz; // flags + idx + ring[qsz] + used_event
        let used_offset = align_up(desc_bytes + avail_bytes, 4096);
        let used_bytes = 6 + 8 * qsz; // flags + idx + ring[qsz] + avail_event
        let total = used_offset + align_up(used_bytes, 4096);
        let pages = total / 4096;

        let region = dma::alloc(pages)?;
        let base = region.virt;

        let q = VirtQueue {
            index,
            size,
            region,
            desc: base as *mut VirtqDesc,
            avail: (base + desc_bytes as u64) as *mut u8,
            used: (base + used_offset as u64) as *mut u8,
            free_head: 0,
            num_free: size,
            last_used_idx: 0,
            avail_idx: 0,
        };

        // Chain every descriptor onto the free list.
        for i in 0..size {
            let next = if i + 1 < size { i + 1 } else { 0 };
            unsafe {
                let d = &mut *q.desc.add(i as usize);
                d.addr = 0;
                d.len = 0;
                d.flags = 0;
                d.next = next;
            }
        }

        // Tell the device where the queue lives (legacy: physical page number).
        transport.select_queue(index);
        transport.set_queue_pfn((q.region.phys / 4096) as u32);
        Some(q)
    }

    fn alloc_desc(&mut self) -> Option<u16> {
        if self.num_free == 0 {
            return None;
        }
        let head = self.free_head;
        let next = unsafe { (*self.desc.add(head as usize)).next };
        self.free_head = next;
        self.num_free -= 1;
        Some(head)
    }

    fn free_chain(&mut self, head: u16) {
        // Guard: `head` itself must be a valid descriptor index.
        if head >= self.size {
            return;
        }
        let mut idx = head;
        let mut depth = 0usize;
        loop {
            // Depth guard — break out before we can spin forever on a forged
            // NEXT cycle or a chain longer than any legitimate packet needs.
            if depth >= MAX_CHAIN_DEPTH {
                break;
            }
            depth += 1;

            let (flags, next) = unsafe {
                let d = &*self.desc.add(idx as usize);
                (d.flags, d.next)
            };
            unsafe {
                (*self.desc.add(idx as usize)).next = self.free_head;
            }
            self.free_head = idx;
            self.num_free += 1;
            if flags & VIRTQ_DESC_F_NEXT == 0 {
                break;
            }
            // Validate the next index before following it.
            if next >= self.size {
                break;
            }
            idx = next;
        }
    }

    // Available-ring field accessors.
    fn set_avail_idx(&self, idx: u16) {
        unsafe { core::ptr::write_volatile(self.avail.add(2) as *mut u16, idx) }
    }
    fn set_avail_ring(&self, slot: u16, desc: u16) {
        let off = 4 + (slot as usize % self.size as usize) * 2;
        unsafe { core::ptr::write_volatile(self.avail.add(off) as *mut u16, desc) }
    }

    // Used-ring field accessors.
    fn used_idx(&self) -> u16 {
        unsafe { core::ptr::read_volatile(self.used.add(2) as *const u16) }
    }
    fn used_elem(&self, slot: u16) -> (u32, u32) {
        let off = 4 + (slot as usize % self.size as usize) * 8;
        unsafe {
            let id = core::ptr::read_volatile(self.used.add(off) as *const u32);
            let len = core::ptr::read_volatile(self.used.add(off + 4) as *const u32);
            (id, len)
        }
    }

    /// Notify the device that new buffers are available on this queue.
    pub fn kick(&self, transport: &VirtioTransport) {
        transport.notify(self.index);
    }

    /// Non-blocking completion check: if the device has finished a request,
    /// return its `(head descriptor index, bytes written)` and reclaim the chain.
    pub fn poll(&mut self) -> Option<(u16, u32)> {
        fence(Ordering::SeqCst);
        if self.used_idx() == self.last_used_idx {
            return None;
        }
        let (id, len) = self.used_elem(self.last_used_idx);
        self.last_used_idx = self.last_used_idx.wrapping_add(1);
        // Bounds-check the device-supplied id before using it as a descriptor
        // index — a malicious or buggy device could supply any 32-bit value.
        let id16 = id as u16;
        if id16 >= self.size {
            return None;
        }
        // Cap the reported byte count to the maximum legal packet size so
        // callers cannot be tricked into oversized allocations or copies.
        let len = len.min(MAX_PACKET_SIZE);
        self.free_chain(id16);
        Some((id16, len))
    }

    /// Reclaim every completion the device has published since the last drain,
    /// returning how many chains were freed. One `fence` + one `used_idx` read
    /// covers the whole burst, instead of one of each per completion — the win
    /// when draining a submitted batch. Behaviour matches calling [`poll`] in a
    /// loop until it yields `None`, minus the redundant per-iteration MMIO-shaped
    /// reads of device-shared memory.
    pub fn poll_drain(&mut self) -> usize {
        fence(Ordering::SeqCst);
        let cur = self.used_idx();
        let mut freed = 0usize;
        while self.last_used_idx != cur {
            let (id, _len) = self.used_elem(self.last_used_idx);
            self.last_used_idx = self.last_used_idx.wrapping_add(1);
            // Bounds-check device-supplied id before using as a descriptor index.
            let id16 = id as u16;
            if id16 < self.size {
                self.free_chain(id16);
            }
            freed += 1;
        }
        freed
    }

    /// Publish a descriptor chain on the available ring (does not notify). Returns
    /// the head descriptor index. Public so drivers can manage RX rings that must
    /// not block.
    pub fn add(&mut self, buffers: &[Buf]) -> Option<u16> {
        self.submit(buffers)
    }

    /// Build a descriptor chain for `buffers` and publish it on the available
    /// ring. Returns the head descriptor index, or `None` if descriptors are
    /// exhausted.
    fn submit(&mut self, buffers: &[Buf]) -> Option<u16> {
        if buffers.is_empty() || (buffers.len() as u16) > self.num_free {
            return None;
        }
        let mut indices: Vec<u16> = Vec::with_capacity(buffers.len());
        for _ in buffers {
            indices.push(self.alloc_desc()?);
        }
        for (i, buf) in buffers.iter().enumerate() {
            let mut flags = 0u16;
            if buf.device_writable {
                flags |= VIRTQ_DESC_F_WRITE;
            }
            if i + 1 < buffers.len() {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            let next = if i + 1 < buffers.len() { indices[i + 1] } else { 0 };
            unsafe {
                let d = &mut *self.desc.add(indices[i] as usize);
                d.addr = buf.phys;
                // Cap the length so a caller passing an attacker-controlled Buf
                // cannot cause oversized DMA beyond the maximum packet size.
                d.len = buf.len.min(MAX_PACKET_SIZE);
                d.flags = flags;
                d.next = next;
            }
        }
        let head = indices[0];
        // Place the head in the next available slot. `avail_idx` is our shadow of
        // the ring index (we are its sole writer), so no read-back is needed.
        let slot = self.avail_idx;
        self.set_avail_ring(slot, head);
        // Order the descriptor + ring-slot stores before the index store the device
        // watches; this single fence is the publication barrier the spec requires.
        fence(Ordering::SeqCst);
        self.avail_idx = slot.wrapping_add(1);
        self.set_avail_idx(self.avail_idx);
        Some(head)
    }

    /// Submit a descriptor chain, kick the device, and busy-poll the used ring
    /// until this request completes. Returns the device-reported written length.
    pub fn submit_and_wait(&mut self, transport: &VirtioTransport, buffers: &[Buf]) -> Option<u32> {
        let _head = self.submit(buffers)?;
        transport.notify(self.index);
        loop {
            fence(Ordering::SeqCst);
            if self.used_idx() != self.last_used_idx {
                let (id, len) = self.used_elem(self.last_used_idx);
                self.last_used_idx = self.last_used_idx.wrapping_add(1);
                self.free_chain(id as u16);
                return Some(len);
            }
            core::hint::spin_loop();
        }
    }
}

// The DMA region keeps the queue memory alive for the queue's lifetime.
unsafe impl Send for VirtQueue {}

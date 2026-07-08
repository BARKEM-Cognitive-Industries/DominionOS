//! The virtio-net driver — roadmap feature 1's hardware half (gate M3).
//!
//! virtio-net presents two virtqueues: queue 0 receives frames, queue 1
//! transmits them. Every frame is prefixed on the wire by a 10-byte
//! `virtio_net_hdr` (we negotiate no offloads, so the header is all zeros on TX
//! and ignored on RX). The protocol logic — Ethernet/ARP/IPv4/ICMP — lives in
//! the pure, host-tested [`dominion_core::net`]; this driver only moves bytes
//! between that stack and the wire.

use crate::dma::{self, DmaRegion};
use crate::pci;
use crate::virtio::{Buf, VirtQueue, VirtioTransport};
use dominion_core::net::MacAddr;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use spin::Mutex;

/// The driver-agnostic contract every NIC implements. The rest of the kernel — the
/// ARP/IPv4/ICMP/UDP/DHCP stack in [`dominion_core::net`], the shell, the web
/// client — talks only to this trait, so virtio-net, the Intel e1000, or any future
/// controller are interchangeable behind [`with_nic`].
pub trait Nic: Send {
    /// The station MAC address.
    fn mac(&self) -> MacAddr;
    /// Transmit one Ethernet frame; returns `false` if it could not be queued.
    fn transmit(&mut self, frame: &[u8]) -> bool;
    /// Non-blocking receive: the next frame if one has arrived.
    fn poll_frame(&mut self) -> Option<Vec<u8>>;
}

/// The system NIC, probed once at boot. Boxed behind [`Nic`] so any driver fits.
static NIC: Mutex<Option<Box<dyn Nic>>> = Mutex::new(None);

/// Probe for a network device and install it globally. Returns true if found.
/// virtio-net is preferred (the paravirtual fast path); failing that we try a
/// real Intel e1000/e1000e — so a physical or VMware/VirtualBox machine still
/// gets networking through the same abstraction.
pub fn init_global() -> bool {
    let mut guard = NIC.lock();
    if guard.is_some() {
        return true;
    }
    if let Some(v) = VirtioNet::init() {
        *guard = Some(Box::new(v));
        return true;
    }
    if let Some(e) = crate::e1000::E1000::init() {
        *guard = Some(Box::new(e));
        return true;
    }
    false
}

/// Run `f` with the global NIC if one is present.
pub fn with_nic<R>(f: impl FnOnce(&mut dyn Nic) -> R) -> Option<R> {
    NIC.lock().as_mut().map(|n| f(&mut **n))
}

/// The NIC's MAC address, or all-zero if none is attached.
pub fn mac() -> MacAddr {
    NIC.lock().as_ref().map(|n| n.mac()).unwrap_or(MacAddr::ZERO)
}

pub fn present() -> bool {
    NIC.lock().is_some()
}

const VIRTIO_SUBSYSTEM_NET: u16 = 1;
const VIRTIO_NET_F_MAC: u32 = 1 << 5;
const VIRTIO_NET_HDR_LEN: usize = 10;

const RX_COUNT: usize = 16;
const RX_BUF_SIZE: usize = 2048; // virtio_net_hdr + a full 1514-byte frame, padded

pub struct VirtioNet {
    transport: VirtioTransport,
    rx: VirtQueue,
    tx: VirtQueue,
    mac: MacAddr,
    rx_pool: DmaRegion,
    tx_buf: DmaRegion,
    /// Maps an RX descriptor head back to the pool buffer index it carries.
    head_to_buf: Vec<usize>,
}

impl VirtioNet {
    /// Probe PCI for a virtio-net device and bring both queues up. Returns `None`
    /// if no NIC is attached.
    pub fn init() -> Option<VirtioNet> {
        let dev = pci::find_virtio(VIRTIO_SUBSYSTEM_NET)?;
        dev.address.enable_bus_master();
        let bar0 = dev.address.bar(0);
        let io_base = (bar0 & 0xFFFC) as u16;
        let transport = unsafe { VirtioTransport::new(io_base) };

        // Negotiate only VIRTIO_NET_F_MAC so the config-space MAC is valid and the
        // RX header stays the 10-byte legacy form (no MRG_RXBUF).
        transport.begin(VIRTIO_NET_F_MAC);

        let mut mac = [0u8; 6];
        for (i, b) in mac.iter_mut().enumerate() {
            *b = transport.config_u8(i as u16);
        }

        let rx = VirtQueue::new(&transport, 0)?;
        let tx = VirtQueue::new(&transport, 1)?;
        transport.finish();

        let rx_pool = dma::alloc((RX_COUNT * RX_BUF_SIZE).div_ceil(4096))?;
        let tx_buf = dma::alloc(1)?;
        let head_to_buf = vec![0usize; rx.size as usize];

        let mut net = VirtioNet {
            transport,
            rx,
            tx,
            mac: MacAddr(mac),
            rx_pool,
            tx_buf,
            head_to_buf,
        };
        // Hand all RX buffers to the device, then kick it once.
        for i in 0..RX_COUNT {
            net.post_rx(i);
        }
        net.rx.kick(&net.transport);
        Some(net)
    }

    pub fn mac(&self) -> MacAddr {
        self.mac
    }

    fn post_rx(&mut self, buf_index: usize) {
        let phys = self.rx_pool.phys + (buf_index * RX_BUF_SIZE) as u64;
        let bufs = [Buf { phys, len: RX_BUF_SIZE as u32, device_writable: true }];
        if let Some(head) = self.rx.add(&bufs) {
            self.head_to_buf[head as usize] = buf_index;
        }
    }

    /// Transmit one Ethernet frame. Prepends the virtio_net_hdr and waits for the
    /// device to consume it. Returns `false` on descriptor exhaustion.
    pub fn transmit(&mut self, frame: &[u8]) -> bool {
        // Reject oversized frames before the memcpy: the header + frame must fit in
        // the fixed TX DMA buffer, or we'd overflow it and corrupt adjacent memory.
        if VIRTIO_NET_HDR_LEN + frame.len() > self.tx_buf.size {
            return false;
        }
        let base = self.tx_buf.virt;
        unsafe {
            // Zeroed 10-byte virtio_net_hdr.
            core::ptr::write_bytes(base as *mut u8, 0, VIRTIO_NET_HDR_LEN);
            core::ptr::copy_nonoverlapping(
                frame.as_ptr(),
                (base + VIRTIO_NET_HDR_LEN as u64) as *mut u8,
                frame.len(),
            );
        }
        let bufs = [Buf {
            phys: self.tx_buf.phys,
            len: (VIRTIO_NET_HDR_LEN + frame.len()) as u32,
            device_writable: false,
        }];
        self.tx.submit_and_wait(&self.transport, &bufs).is_some()
    }

    /// Non-blocking receive: returns the next Ethernet frame (header stripped) if
    /// one has arrived, and re-posts its buffer to the device.
    pub fn poll_frame(&mut self) -> Option<Vec<u8>> {
        let (head, len) = self.rx.poll()?;
        let buf_index = self.head_to_buf[head as usize];
        // The device reports the used length; clamp it to the actual RX buffer
        // capacity so a buggy/malicious device can't drive an out-of-bounds read
        // past this 2 KiB slot into adjacent DMA/kernel memory.
        let total = (len as usize).min(RX_BUF_SIZE);
        let frame = if total > VIRTIO_NET_HDR_LEN {
            let start = self.rx_pool.virt + (buf_index * RX_BUF_SIZE + VIRTIO_NET_HDR_LEN) as u64;
            let frame_len = total - VIRTIO_NET_HDR_LEN;
            let mut v = vec![0u8; frame_len];
            unsafe {
                core::ptr::copy_nonoverlapping(start as *const u8, v.as_mut_ptr(), frame_len);
            }
            Some(v)
        } else {
            None
        };
        // Recycle the buffer.
        self.post_rx(buf_index);
        self.rx.kick(&self.transport);
        frame
    }
}

impl Nic for VirtioNet {
    fn mac(&self) -> MacAddr {
        self.mac
    }
    fn transmit(&mut self, frame: &[u8]) -> bool {
        VirtioNet::transmit(self, frame)
    }
    fn poll_frame(&mut self) -> Option<Vec<u8>> {
        VirtioNet::poll_frame(self)
    }
}

// Raw pointers in the queues/regions; the driver is only touched single-threaded.
unsafe impl Send for VirtioNet {}

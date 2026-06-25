//! Zero-copy ring buffer for IPC paths (L3 memory-acceleration roadmap).
//!
//! A [`ZcpRing`] carries [`ZeroCopyHandle`] tokens (32 bytes each) rather
//! than raw byte payloads. Consumers dereference handles through the shared
//! [`ZeroCopyPool`] — zero byte copies occur in flight.

use crate::zerocopy::ZeroCopyHandle;
use alloc::collections::VecDeque;

/// Statistics tracked by a [`ZcpRing`].
#[derive(Clone, Debug, Default)]
pub struct RingStats {
    /// Total handles successfully enqueued.
    pub enqueued: u64,
    /// Total handles dequeued by receivers.
    pub dequeued: u64,
    /// Handles dropped because the ring was at capacity (oldest evicted).
    pub dropped: u64,
    /// Logical bytes transferred (sum of `byte_len` args to successful sends).
    pub bytes_via_handles: usize,
    /// Bytes that would have been copied in a naive implementation.
    pub bytes_if_copied: usize,
}

/// A ring-based zero-copy IPC channel.
///
/// Senders post [`ZeroCopyHandle`] entries (32 bytes each).
/// Receivers read handles and dereference via the shared pool.
/// No byte copies occur in flight — only handle metadata moves.
///
/// When the ring is full, the **oldest** entry is evicted (head-drop
/// backpressure) and [`RingStats::dropped`] is incremented.
pub struct ZcpRing {
    queue: VecDeque<ZeroCopyHandle>,
    capacity: usize,
    stats: RingStats,
}

impl ZcpRing {
    /// Create a ring with a maximum of `capacity` in-flight handles.
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            capacity,
            stats: RingStats::default(),
        }
    }

    /// Enqueue `handle` representing `byte_len` bytes of logical payload.
    ///
    /// If the ring is at capacity the oldest entry is evicted first.
    /// Returns `true` if the handle was enqueued without eviction.
    pub fn send(&mut self, handle: ZeroCopyHandle, byte_len: usize) -> bool {
        // Always charge bytes_if_copied — even on overflow the caller "sent" them.
        self.stats.bytes_if_copied += byte_len;

        let had_room = self.queue.len() < self.capacity;
        if !had_room {
            self.queue.pop_front();
            self.stats.dropped += 1;
        }

        self.queue.push_back(handle);
        self.stats.enqueued += 1;
        self.stats.bytes_via_handles += byte_len;

        had_room
    }

    /// Dequeue the next handle (FIFO order). Returns `None` when empty.
    pub fn recv(&mut self) -> Option<ZeroCopyHandle> {
        let h = self.queue.pop_front()?;
        self.stats.dequeued += 1;
        Some(h)
    }

    /// `true` when there are no pending handles.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Number of handles currently queued.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Snapshot of ring statistics.
    pub fn stats(&self) -> RingStats {
        self.stats.clone()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zerocopy::ZeroCopyPool;

    fn make_handle(pool: &mut ZeroCopyPool, val: u8) -> ZeroCopyHandle {
        pool.intern(alloc::vec![val; 64])
    }

    #[test]
    fn ring_fifo_order() {
        let mut pool = ZeroCopyPool::new();
        let mut ring = ZcpRing::new(10);

        let h1 = make_handle(&mut pool, 1);
        let h2 = make_handle(&mut pool, 2);
        let h3 = make_handle(&mut pool, 3);

        ring.send(h1, 64);
        ring.send(h2, 64);
        ring.send(h3, 64);

        assert_eq!(ring.recv(), Some(h1));
        assert_eq!(ring.recv(), Some(h2));
        assert_eq!(ring.recv(), Some(h3));
        assert_eq!(ring.recv(), None);
    }

    #[test]
    fn ring_capacity_drops_oldest_when_full() {
        let mut pool = ZeroCopyPool::new();
        let mut ring = ZcpRing::new(3);

        let h1 = make_handle(&mut pool, 1);
        let h2 = make_handle(&mut pool, 2);
        let h3 = make_handle(&mut pool, 3);
        let h4 = make_handle(&mut pool, 4);

        ring.send(h1, 64); // queue: [h1]
        ring.send(h2, 64); // queue: [h1, h2]
        ring.send(h3, 64); // queue: [h1, h2, h3]  — at capacity
        ring.send(h4, 64); // drops h1 → queue: [h2, h3, h4]

        let stats = ring.stats();
        assert_eq!(stats.dropped, 1);

        assert_eq!(ring.recv(), Some(h2));
        assert_eq!(ring.recv(), Some(h3));
        assert_eq!(ring.recv(), Some(h4));
        assert!(ring.is_empty());
    }

    #[test]
    fn ring_stats_track_bytes() {
        let mut pool = ZeroCopyPool::new();
        let mut ring = ZcpRing::new(10);

        let h = make_handle(&mut pool, 7);
        ring.send(h, 1024);
        ring.send(h, 2048);

        let stats = ring.stats();
        assert_eq!(stats.bytes_if_copied, 3072);
        assert_eq!(stats.bytes_via_handles, 3072);
        assert_eq!(stats.enqueued, 2);
    }
}

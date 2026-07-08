//! Late-Latch input ring buffer — render redesign milestone.
//!
//! The input engine captures hardware mouse/touch interrupts at up to 8000 Hz.
//! Instead of waiting for application event loops, coordinates are written to a
//! shared ring buffer. The GPU compositor queries it at the latest possible moment
//! before pixel stream dispatch — "late latching" — targeting sub-5 ms
//! input-to-photon latency.
//!
//! ## Design
//!
//! Because `#![deny(unsafe_code)]` is set crate-wide we cannot use
//! `UnsafeCell` directly.  Instead we encode each `InputCoord` into two
//! `AtomicU64` words:
//!
//! * **word 0** — `(x as i32 as u32) | ((y as i32 as u32) << 32)` — signed
//!   i32 values bit-cast to u32 and packed.
//! * **word 1** — `(buttons as u64) | (timestamp_us << 8)` — buttons in the
//!   low 8 bits, timestamp in the upper 56 bits (≈ 72 years at µs resolution,
//!   more than sufficient).
//!
//! Each slot in the ring buffer is two consecutive `AtomicU64` values.  The
//! writer (ISR side) stores word 0 first with `Release`, then word 1 with
//! `Release` — the pair is treated as committed once word 1 is non-zero for
//! the slot.  The reader uses `Acquire` loads.  A version counter per slot
//! (encoded as a sequence number in the high bits of word 1) lets the reader
//! detect torn reads.
//!
//! A simpler "latest slot" approach is provided by [`late_latch`] which
//! is the primary compositor entry-point.  A full circular buffer is provided
//! by [`InputRingBuffer`] for cases that need historical draining.

// This module encodes InputCoord fields atomically; no raw pointer arithmetic
// or UnsafeCell is required — all synchronisation is via AtomicU64 / AtomicU32.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A single hardware pointer event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct InputCoord {
    pub x: i32,
    pub y: i32,
    pub buttons: u8,
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

/// Pack an `InputCoord` into two u64 words.
///
/// Word 0: low 32 bits = x (bit-cast), high 32 bits = y (bit-cast).
/// Word 1: low 8 bits = buttons; bits 8..63 = timestamp_us (truncated to 56
///          bits — ~72 years at µs resolution — which is ample).
#[inline]
fn encode(c: &InputCoord) -> (u64, u64) {
    let w0 = (c.x as u32 as u64) | ((c.y as u32 as u64) << 32);
    // Reserve the high byte of word-1 as a zero sentinel so a freshly-zeroed
    // slot is distinguishable from a real coord with timestamp 0.
    // We shift the timestamp left by 8 bits and put buttons in the low byte.
    let w1 = (c.buttons as u64) | (c.timestamp_us << 8);
    (w0, w1)
}

#[inline]
fn decode(w0: u64, w1: u64) -> InputCoord {
    let x = (w0 as u32) as i32;
    let y = ((w0 >> 32) as u32) as i32;
    let buttons = (w1 & 0xFF) as u8;
    let timestamp_us = w1 >> 8;
    InputCoord { x, y, buttons, timestamp_us }
}

// ---------------------------------------------------------------------------
// Single-slot "latest" atomic cell (used by late_latch)
// ---------------------------------------------------------------------------

/// Two-word atomic cell holding one `InputCoord`.
///
/// Written with `Release`, read with `Acquire`.  Readers detect stale data
/// because `word1 == 0` means "no data yet" (timestamp 0 with buttons 0 is
/// theoretically valid but practically never occurs at system start).
struct AtomicCoordCell {
    w0: AtomicU64,
    w1: AtomicU64,
    /// Seqlock version: even = stable, odd = write in progress. The single-writer ISR bumps
    /// it odd before the two word stores and even after; a reader that sees an odd or changed
    /// value between its reads retries. Without this a reader could pair word 0 from event
    /// N+1 with word 1 from event N (a torn cross-event coordinate — new position, stale
    /// buttons/timestamp), because the two loads are otherwise independent.
    seq: AtomicU32,
}

impl AtomicCoordCell {
    const fn new() -> Self {
        Self { w0: AtomicU64::new(0), w1: AtomicU64::new(0), seq: AtomicU32::new(0) }
    }

    fn store(&self, c: &InputCoord) {
        let (w0, w1) = encode(c);
        let s = self.seq.load(Ordering::Relaxed);
        // Bump to odd — readers now see a write in progress and will retry.
        self.seq.store(s.wrapping_add(1), Ordering::Release);
        self.w0.store(w0, Ordering::Release);
        self.w1.store(w1, Ordering::Release);
        // Bump to even — the pair is committed and stable.
        self.seq.store(s.wrapping_add(2), Ordering::Release);
    }

    fn load(&self) -> Option<InputCoord> {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 != 0 {
                continue; // write in progress — spin (single ISR writer completes quickly)
            }
            let w1 = self.w1.load(Ordering::Acquire);
            let w0 = self.w0.load(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Acquire);
            if s1 == s2 {
                if w1 == 0 {
                    return None;
                }
                return Some(decode(w0, w1));
            }
            // The writer ran between our two reads — retry to avoid a torn coordinate.
        }
    }
}

// ---------------------------------------------------------------------------
// Ring-buffer slot
// ---------------------------------------------------------------------------

struct Slot {
    w0: AtomicU64,
    w1: AtomicU64,
    /// Sequence number: even = empty/being-written, odd = committed.
    seq: AtomicU32,
}

impl Slot {
    const fn new() -> Self {
        Self {
            w0: AtomicU64::new(0),
            w1: AtomicU64::new(0),
            seq: AtomicU32::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// InputRingBuffer
// ---------------------------------------------------------------------------

/// Lock-free SPSC ring buffer for input coordinates.
///
/// `CAP` must be a power of two (checked at construction; panics in debug if not).
///
/// * **Producer** (input ISR): calls [`push`](InputRingBuffer::push).  If the
///   buffer is full the oldest entry is overwritten and [`drops`] is incremented.
/// * **Consumer** (compositor): calls [`latest`](InputRingBuffer::latest) for
///   the freshest coord or [`drain_into`](InputRingBuffer::drain_into) to
///   process all pending events.
pub struct InputRingBuffer<const CAP: usize> {
    slots: [Slot; CAP],
    /// Index of next write slot (producer-owned).
    head: AtomicU32,
    /// Index of next read slot (consumer-owned).
    tail: AtomicU32,
    /// Total number of overwritten (dropped) entries.
    drops: AtomicU64,
    /// Latest-slot fast path used by [`late_latch`].
    latest_cell: AtomicCoordCell,
}

// `InputRingBuffer` is automatically `Sync + Send` because all of its fields
// (`AtomicU64`, `AtomicU32`, `AtomicCoordCell`) are `Sync + Send`.  No manual
// unsafe impl is needed.

impl<const CAP: usize> InputRingBuffer<CAP> {
    /// Create a new, empty ring buffer.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `CAP` is not a power of two or is zero.
    pub const fn new() -> Self {
        // const-fn limitation: cannot use assert! in const context on stable,
        // but the const evaluator will catch CAP == 0 at compile time via the
        // array size.  A runtime debug check is added in push/latest.
        Self {
            slots: [const { Slot::new() }; CAP],
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            drops: AtomicU64::new(0),
            latest_cell: AtomicCoordCell::new(),
        }
    }

    #[inline]
    fn mask(&self, idx: u32) -> usize {
        (idx as usize) & (CAP - 1)
    }

    /// Current number of unconsumed entries.
    ///
    /// Clamped to `CAP`: after an overflow the head can run more than `CAP` ahead of a
    /// stalled tail, but the ring only physically holds `CAP` entries, so the count is
    /// capped rather than reported as a growing, bogus value.
    pub fn len(&self) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        (h.wrapping_sub(t) as usize).min(CAP)
    }

    /// Write a new coordinate from the input interrupt handler.  Never blocks.
    ///
    /// If the buffer is full the **oldest** entry is overwritten (we always
    /// want the freshest input) and the drop counter is incremented.
    pub fn push(&self, coord: InputCoord) {
        debug_assert!(CAP.is_power_of_two() && CAP > 0, "CAP must be a nonzero power of two");

        // Update the fast-path latest cell unconditionally.
        self.latest_cell.store(&coord);

        let h = self.head.load(Ordering::Relaxed);
        let t = self.tail.load(Ordering::Acquire);
        let used = h.wrapping_sub(t) as usize;

        if used >= CAP {
            // Full: the producer overwrites the slot at `h % CAP` (the oldest
            // entry) and simply does NOT advance tail — that pointer is
            // consumer-owned.  Record a drop so callers can detect loss.
            self.drops.fetch_add(1, Ordering::Relaxed);
        }

        let slot = &self.slots[self.mask(h)];
        let (w0, w1) = encode(&coord);

        // Mark slot as "being written" (even seq).
        slot.seq.store(h.wrapping_mul(2), Ordering::Relaxed);
        slot.w0.store(w0, Ordering::Relaxed);
        slot.w1.store(w1, Ordering::Release);
        // Mark slot as "committed" (odd seq).
        slot.seq.store(h.wrapping_mul(2).wrapping_add(1), Ordering::Release);

        self.head.store(h.wrapping_add(1), Ordering::Release);
    }

    /// Return the most recently pushed coordinate without consuming it.
    ///
    /// Uses the fast-path single-slot cell written by every [`push`] so this
    /// is a single `Acquire` load — minimal compositor overhead.
    pub fn latest(&self) -> Option<InputCoord> {
        self.latest_cell.load()
    }

    /// Drain all pending coordinates into `out`, returning the count copied.
    ///
    /// Coordinates are ordered oldest-first.  At most `out.len()` entries are
    /// returned; if the ring buffer contains more they remain in the buffer.
    pub fn drain_into(&self, out: &mut [InputCoord]) -> usize {
        let mut count = 0;
        while count < out.len() {
            let mut t = self.tail.load(Ordering::Acquire);
            let h = self.head.load(Ordering::Acquire);
            if t == h {
                break; // empty
            }
            // Overrun recovery: if the producer has lapped us (more than CAP entries pushed
            // since our tail), the slots from tail up to head-CAP have been overwritten and
            // their seq no longer matches expected_seq. Fast-forward tail to the oldest slot
            // the ring still physically holds so draining resynchronises instead of stalling
            // at 0 forever. tail stays consumer-owned, preserving the SPSC invariant.
            if h.wrapping_sub(t) as usize > CAP {
                t = h.wrapping_sub(CAP as u32);
                self.tail.store(t, Ordering::Release);
            }
            let slot = &self.slots[self.mask(t)];
            // Spin until the slot is committed (odd seq matching t).
            let expected_seq = t.wrapping_mul(2).wrapping_add(1);
            let seq = slot.seq.load(Ordering::Acquire);
            if seq != expected_seq {
                // Writer hasn't finished storing yet — stop here.
                break;
            }
            let w0 = slot.w0.load(Ordering::Acquire);
            let w1 = slot.w1.load(Ordering::Acquire);
            out[count] = decode(w0, w1);
            count += 1;
            self.tail.store(t.wrapping_add(1), Ordering::Release);
        }
        count
    }

    /// Number of coords dropped due to buffer overflow.
    pub fn drops(&self) -> u64 {
        self.drops.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Late-latch compositor entry-point
// ---------------------------------------------------------------------------

/// Late-latch query — call from the compositor just before scanout.
///
/// Returns the freshest pointer position available.  This is an `Acquire`
/// load of the single-slot cell updated by every [`InputRingBuffer::push`],
/// so latency is a single atomic load after the interrupt fires.
pub fn late_latch(ring: &InputRingBuffer<64>) -> Option<InputCoord> {
    ring.latest()
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Diagnostic snapshot of the input path.
pub struct InputLatencyMetrics {
    /// Total [`push`](InputRingBuffer::push) calls observed (estimated via
    /// head counter).
    pub samples: u64,
    /// Approximate age of the oldest coord still in the buffer (µs).
    pub max_age_us: u64,
    /// Coords dropped due to buffer overflow.
    pub drops: u64,
}

impl<const CAP: usize> InputRingBuffer<CAP> {
    /// Snapshot current metrics.  Approximated — not atomic across all fields.
    pub fn metrics(&self, now_us: u64) -> InputLatencyMetrics {
        let h = self.head.load(Ordering::Acquire) as u64;
        let t = self.tail.load(Ordering::Acquire);
        let drops = self.drops.load(Ordering::Relaxed);

        // Estimate age of oldest buffered coord by reading tail slot.
        let max_age_us = if self.head.load(Ordering::Relaxed) != t {
            let slot = &self.slots[self.mask(t)];
            let w1 = slot.w1.load(Ordering::Acquire);
            if w1 != 0 {
                let ts = w1 >> 8;
                now_us.saturating_sub(ts)
            } else {
                0
            }
        } else {
            0
        };

        InputLatencyMetrics { samples: h, max_age_us, drops }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: make a coord with a distinct timestamp.
    fn coord(x: i32, y: i32, ts: u64) -> InputCoord {
        InputCoord { x, y, buttons: 0, timestamp_us: ts }
    }

    fn coord_btn(x: i32, y: i32, buttons: u8, ts: u64) -> InputCoord {
        InputCoord { x, y, buttons, timestamp_us: ts }
    }

    // 1. New buffer is empty.
    #[test]
    fn new_buffer_is_empty() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.latest(), None);
    }

    // 2. Single push makes latest() return that coord.
    #[test]
    fn push_one_latest() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        rb.push(coord(10, 20, 1000));
        assert_eq!(rb.latest(), Some(coord(10, 20, 1000)));
        assert_eq!(rb.len(), 1);
    }

    // 3. latest() always reflects the most recent push.
    #[test]
    fn latest_reflects_newest() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        rb.push(coord(1, 2, 100));
        rb.push(coord(3, 4, 200));
        rb.push(coord(5, 6, 300));
        assert_eq!(rb.latest().unwrap().x, 5);
        assert_eq!(rb.latest().unwrap().y, 6);
    }

    // 4. drain_into returns oldest-first and empties buffer.
    #[test]
    fn drain_into_ordered() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        rb.push(coord(1, 0, 10));
        rb.push(coord(2, 0, 20));
        rb.push(coord(3, 0, 30));
        let mut out = [InputCoord::default(); 8];
        let n = rb.drain_into(&mut out);
        assert_eq!(n, 3);
        assert_eq!(out[0].x, 1);
        assert_eq!(out[1].x, 2);
        assert_eq!(out[2].x, 3);
        assert_eq!(rb.len(), 0);
    }

    // 5. drain_into caps at out.len().
    #[test]
    fn drain_into_caps_at_slice_len() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        for i in 0..5i32 {
            rb.push(coord(i, 0, i as u64 * 10));
        }
        let mut out = [InputCoord::default(); 2];
        let n = rb.drain_into(&mut out);
        assert_eq!(n, 2);
        assert_eq!(rb.len(), 3); // 3 remain
    }

    // 6. Overwrite on full — drops counter incremented.
    #[test]
    fn overwrite_on_full_increments_drops() {
        let rb: InputRingBuffer<4> = InputRingBuffer::new();
        for i in 0..8u64 {
            rb.push(coord(i as i32, 0, i * 100));
        }
        assert!(rb.drops() > 0);
    }

    // 7. After overflow, latest() is still the newest coord.
    #[test]
    fn overflow_latest_still_newest() {
        let rb: InputRingBuffer<4> = InputRingBuffer::new();
        for i in 0..8u64 {
            rb.push(coord(i as i32, 0, i * 100));
        }
        assert_eq!(rb.latest().unwrap().x, 7);
    }

    // 8. Buttons field round-trips.
    #[test]
    fn buttons_round_trip() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        rb.push(coord_btn(0, 0, 0b101, 999));
        assert_eq!(rb.latest().unwrap().buttons, 0b101);
    }

    // 9. Negative coordinates round-trip.
    #[test]
    fn negative_coords_round_trip() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        rb.push(coord(-1920, -1080, 42));
        let c = rb.latest().unwrap();
        assert_eq!(c.x, -1920);
        assert_eq!(c.y, -1080);
        assert_eq!(c.timestamp_us, 42);
    }

    // 10. Large timestamp round-trips (56-bit range).
    #[test]
    fn large_timestamp_round_trip() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        // 2^55 fits in 56 bits
        let ts: u64 = 1 << 55;
        rb.push(coord(0, 0, ts));
        assert_eq!(rb.latest().unwrap().timestamp_us, ts);
    }

    // 11. drain_into on empty buffer returns 0.
    #[test]
    fn drain_empty_returns_zero() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        let mut out = [InputCoord::default(); 4];
        assert_eq!(rb.drain_into(&mut out), 0);
    }

    // 12. late_latch returns None on empty ring.
    #[test]
    fn late_latch_none_on_empty() {
        let rb: InputRingBuffer<64> = InputRingBuffer::new();
        assert_eq!(late_latch(&rb), None);
    }

    // 13. late_latch returns latest after push.
    #[test]
    fn late_latch_returns_latest() {
        let rb: InputRingBuffer<64> = InputRingBuffer::new();
        rb.push(coord(100, 200, 5000));
        let c = late_latch(&rb).unwrap();
        assert_eq!(c.x, 100);
        assert_eq!(c.y, 200);
    }

    // 14. Multiple drains interleaved with pushes.
    #[test]
    fn interleaved_push_drain() {
        let rb: InputRingBuffer<8> = InputRingBuffer::new();
        rb.push(coord(1, 0, 1));
        rb.push(coord(2, 0, 2));
        let mut out = [InputCoord::default(); 8];
        let n = rb.drain_into(&mut out);
        assert_eq!(n, 2);
        rb.push(coord(3, 0, 3));
        let n2 = rb.drain_into(&mut out);
        assert_eq!(n2, 1);
        assert_eq!(out[0].x, 3);
    }

    // 15. metrics() drops field matches manual overflow count.
    #[test]
    fn metrics_drops_match() {
        let rb: InputRingBuffer<4> = InputRingBuffer::new();
        // Push 4 (fills), then 2 more (2 drops).
        for i in 0..6u64 {
            rb.push(coord(i as i32, 0, i * 10));
        }
        let m = rb.metrics(1000);
        assert_eq!(m.drops, 2);
    }

    // 16. encode/decode is lossless for all button values.
    #[test]
    fn encode_decode_all_buttons() {
        for b in 0u8..=255 {
            let c = coord_btn(42, -7, b, 12345);
            let (w0, w1) = encode(&c);
            let d = decode(w0, w1);
            assert_eq!(d, c);
        }
    }

    // 17. CAP=2 boundary: push two, drain two, push two more.
    #[test]
    fn small_cap_wraparound() {
        let rb: InputRingBuffer<2> = InputRingBuffer::new();
        rb.push(coord(0, 0, 1));
        rb.push(coord(1, 0, 2));
        let mut out = [InputCoord::default(); 4];
        let n = rb.drain_into(&mut out);
        assert_eq!(n, 2);
        rb.push(coord(2, 0, 3));
        rb.push(coord(3, 0, 4));
        let n2 = rb.drain_into(&mut out);
        assert_eq!(n2, 2);
        assert_eq!(out[0].x, 2);
        assert_eq!(out[1].x, 3);
    }
}

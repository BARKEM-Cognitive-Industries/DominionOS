//! Shared byte-cursor primitives for binary serialisation and deserialisation.
//!
//! Both the object-graph format ([`crate::object`]) and the incremental
//! on-disk store ([`crate::objstore`]) need to walk a byte buffer with checked
//! reads, and to build up an output buffer with typed writes. Keeping one copy
//! of this logic here eliminates two independent inline implementations.

use crate::hash::Hash256;
use alloc::vec::Vec;

// ── Read cursor ──────────────────────────────────────────────────────────────

/// A checked read cursor over a borrowed byte slice.
///
/// Every method returns `None` (or `Err`) when the buffer is too short rather
/// than panicking, so callers can propagate parse failures without `unwrap`.
pub struct Cursor<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    /// Advance by `n` bytes and return the slice, or `None` if there are fewer
    /// than `n` bytes remaining.
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }

    /// Infallible variant used by `object.rs` serialisation — returns
    /// `Err("unexpected end of graph data")` instead of `None`.
    pub fn take_or<'e>(&mut self, n: usize, err: &'e str) -> Result<&'a [u8], &'e str> {
        self.take(n).ok_or(err)
    }

    pub fn read_u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    pub fn read_u8_or(&mut self, err: &'static str) -> Result<u8, &'static str> {
        self.take_or(1, err).map(|b| b[0])
    }

    pub fn read_u32_le(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    pub fn read_u32_le_or(&mut self, err: &'static str) -> Result<u32, &'static str> {
        let b = self.take_or(4, err)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64_le(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Some(u64::from_le_bytes(a))
    }

    pub fn read_u64_le_or(&mut self, err: &'static str) -> Result<u64, &'static str> {
        let b = self.take_or(8, err)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    pub fn read_hash(&mut self) -> Option<Hash256> {
        let mut a = [0u8; 32];
        a.copy_from_slice(self.take(32)?);
        Some(Hash256(a))
    }

    pub fn read_hash_or(&mut self, err: &'static str) -> Result<Hash256, &'static str> {
        let b = self.take_or(32, err)?;
        let mut a = [0u8; 32];
        a.copy_from_slice(b);
        Ok(Hash256(a))
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
}

// ── Write cursor ─────────────────────────────────────────────────────────────

/// An owned, append-only byte buffer with typed write helpers.
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer { buf: Vec::new() }
    }

    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn write_u32_le(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_u64_le(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    /// Consume the writer and return the accumulated buffer.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

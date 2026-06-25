//! Persistence — milestone **M1**: backing the object graph with a block device
//! so commits survive reboot.
//!
//! This is the *logic* half (pure, safe, host-testable). The *mechanism* half —
//! the actual virtio-blk driver — lives in `dominion-kernel` and implements the
//! [`BlockDevice`] trait defined here. The kernel passes a real disk; host tests
//! pass a RAM-backed disk. Both exercise the identical save/load code.
//!
//! Format: block 0 is a superblock naming the payload length; the graph's
//! serialised bytes ([`ObjectGraph::serialize`]) follow in subsequent blocks.
//! "The whole system can roll back to any prior state" — now across power cycles.

use crate::object::ObjectGraph;
use alloc::vec::Vec;

/// Sector size of the underlying block device.
pub const BLOCK_SIZE: usize = 512;

const SUPER_MAGIC: &[u8; 8] = b"AEPERS01";

/// Why a block operation failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockError {
    /// The requested LBA is beyond the device.
    OutOfRange,
    /// A buffer was not exactly [`BLOCK_SIZE`] bytes.
    BadLength,
    /// The device reported an I/O failure.
    DeviceFault,
    /// The on-disk image is too small for the data being saved.
    Full,
}

/// A linear array of fixed-size (512-byte) sectors.
pub trait BlockDevice {
    /// Number of addressable blocks.
    fn block_count(&self) -> u64;
    /// Read one block into a [`BLOCK_SIZE`]-byte buffer.
    fn read_block(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    /// Write one [`BLOCK_SIZE`]-byte block.
    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;

    /// Read a run of consecutive blocks starting at `start_lba` into `buf`, which must
    /// be a whole number of [`BLOCK_SIZE`]-byte blocks.
    ///
    /// The default walks the run one block at a time. Real hardware drivers override
    /// this to *pipeline* the whole run through the device with a single notify, which
    /// turns N per-block round-trips (each a VM-exit + busy-poll) into one — the
    /// difference between a sluggish and a snappy boot/restore. This is a pure data-path
    /// optimisation: it changes how bytes move, never what is stored.
    fn read_blocks(&mut self, start_lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        for (i, chunk) in buf.chunks_mut(BLOCK_SIZE).enumerate() {
            self.read_block(start_lba + i as u64, chunk)?;
        }
        Ok(())
    }

    /// Write a run of consecutive blocks starting at `start_lba` from `buf`, which must
    /// be a whole number of [`BLOCK_SIZE`]-byte blocks. See [`read_blocks`](Self::read_blocks)
    /// for why drivers override the default one-block-at-a-time loop.
    fn write_blocks(&mut self, start_lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if !buf.len().is_multiple_of(BLOCK_SIZE) {
            return Err(BlockError::BadLength);
        }
        for (i, chunk) in buf.chunks(BLOCK_SIZE).enumerate() {
            self.write_block(start_lba + i as u64, chunk)?;
        }
        Ok(())
    }
}

/// Save/load the object graph to/from a block device.
pub struct Persistence;

impl Persistence {
    /// Write `graph` to `dev`: a superblock then the serialised payload.
    pub fn save(dev: &mut dyn BlockDevice, graph: &ObjectGraph) -> Result<(), BlockError> {
        Self::write_image(dev, 0, SUPER_MAGIC, &graph.serialize())
    }

    /// Load a graph from `dev`. Returns `Ok(None)` if the device is unformatted
    /// (no valid superblock) or the payload is corrupt — a fresh boot, not an error.
    pub fn load(dev: &mut dyn BlockDevice) -> Result<Option<ObjectGraph>, BlockError> {
        match Self::read_image(dev, 0, SUPER_MAGIC)? {
            Some(payload) => Ok(ObjectGraph::deserialize(&payload).ok()),
            None => Ok(None),
        }
    }

    /// Write an arbitrary `payload` to `dev` starting at `start_lba`, prefixed by a
    /// one-block superblock carrying `magic` + the payload length. This is the generic
    /// form of [`save`] for blobs that are not an object graph (e.g. the shell's VFS
    /// image), so several independent images can coexist on one disk at different LBAs.
    pub fn save_blob(
        dev: &mut dyn BlockDevice,
        start_lba: u64,
        magic: &[u8; 8],
        payload: &[u8],
    ) -> Result<(), BlockError> {
        Self::write_image(dev, start_lba, magic, payload)
    }

    /// Read a blob written by [`save_blob`] at `start_lba`. Returns `Ok(None)` if the
    /// superblock's magic does not match (no image there yet) — a fresh boot, not an error.
    pub fn load_blob(
        dev: &mut dyn BlockDevice,
        start_lba: u64,
        magic: &[u8; 8],
    ) -> Result<Option<Vec<u8>>, BlockError> {
        Self::read_image(dev, start_lba, magic)
    }

    /// Lay out `magic` + length superblock and `payload` into one contiguous,
    /// block-aligned buffer and hand the whole run to the device in a single
    /// [`write_blocks`](BlockDevice::write_blocks) call. Batching the run lets the
    /// driver pipeline it through the disk with one notify instead of one per sector.
    fn write_image(
        dev: &mut dyn BlockDevice,
        start_lba: u64,
        magic: &[u8; 8],
        payload: &[u8],
    ) -> Result<(), BlockError> {
        let blocks_needed = 1 + payload.len().div_ceil(BLOCK_SIZE) as u64;
        if start_lba + blocks_needed > dev.block_count() {
            return Err(BlockError::Full);
        }
        // One zeroed buffer covering the superblock + payload, rounded up to whole
        // blocks (the tail block's slack stays zero).
        let mut image = alloc::vec![0u8; blocks_needed as usize * BLOCK_SIZE];
        image[0..8].copy_from_slice(magic);
        image[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        image[BLOCK_SIZE..BLOCK_SIZE + payload.len()].copy_from_slice(payload);
        dev.write_blocks(start_lba, &image)
    }

    /// Inverse of [`write_image`](Self::write_image): read the superblock, then pull the
    /// whole payload run in a single [`read_blocks`](BlockDevice::read_blocks) call.
    /// Returns `Ok(None)` if `magic` does not match — an empty slot, not an error.
    fn read_image(
        dev: &mut dyn BlockDevice,
        start_lba: u64,
        magic: &[u8; 8],
    ) -> Result<Option<Vec<u8>>, BlockError> {
        let mut sb = [0u8; BLOCK_SIZE];
        dev.read_block(start_lba, &mut sb)?;
        if &sb[0..8] != magic {
            return Ok(None);
        }
        let len = u64::from_le_bytes(sb[8..16].try_into().unwrap()) as usize;
        let nblocks = len.div_ceil(BLOCK_SIZE);
        let mut payload = alloc::vec![0u8; nblocks * BLOCK_SIZE];
        if nblocks > 0 {
            dev.read_blocks(start_lba + 1, &mut payload)?;
        }
        payload.truncate(len);
        Ok(Some(payload))
    }

    /// Has this device been formatted with a graph image?
    pub fn is_formatted(dev: &mut dyn BlockDevice) -> Result<bool, BlockError> {
        let mut sb = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut sb)?;
        Ok(&sb[0..8] == SUPER_MAGIC)
    }
}

/// A RAM-backed block device — used by host tests and as a fallback when no real
/// disk is attached.
pub struct RamDisk {
    blocks: Vec<[u8; BLOCK_SIZE]>,
}

impl RamDisk {
    pub fn new(block_count: usize) -> RamDisk {
        RamDisk {
            blocks: alloc::vec![[0u8; BLOCK_SIZE]; block_count],
        }
    }
}

impl BlockDevice for RamDisk {
    fn block_count(&self) -> u64 {
        self.blocks.len() as u64
    }
    fn read_block(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        let b = self.blocks.get(lba as usize).ok_or(BlockError::OutOfRange)?;
        buf.copy_from_slice(b);
        Ok(())
    }
    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadLength);
        }
        let b = self.blocks.get_mut(lba as usize).ok_or(BlockError::OutOfRange)?;
        b.copy_from_slice(buf);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Datum, Object};

    fn sample_graph() -> ObjectGraph {
        let mut g = ObjectGraph::new();
        g.put(Object::new("Invoice").with("amount", Datum::Int(100)).with("client", Datum::Text("Acme".into())));
        g.put(Object::new("Note").with("body", Datum::Text("hello".into())));
        g.put(Object::new("Blob").with("data", Datum::Bytes(alloc::vec![1, 2, 3, 250, 0, 99])));
        g.commit("first");
        g.put(Object::new("Invoice").with("amount", Datum::Int(250)));
        g.commit("second");
        g
    }

    #[test]
    fn serialize_round_trips_in_memory() {
        let g = sample_graph();
        let bytes = g.serialize();
        let g2 = ObjectGraph::deserialize(&bytes).unwrap();
        assert_eq!(g.root_hash(), g2.root_hash());
        assert_eq!(g.stored_count(), g2.stored_count());
        assert_eq!(g.live_count(), g2.live_count());
        assert_eq!(g.history().len(), g2.history().len());
    }

    #[test]
    fn save_then_load_survives_round_trip() {
        let g = sample_graph();
        let mut disk = RamDisk::new(64);
        Persistence::save(&mut disk, &g).unwrap();
        let loaded = Persistence::load(&mut disk).unwrap().expect("should load");
        assert_eq!(g.root_hash(), loaded.root_hash());
        assert_eq!(g.stored_count(), loaded.stored_count());
        assert_eq!(g.history().len(), loaded.history().len());
    }

    #[test]
    fn rollback_target_survives_persistence() {
        // A commit root taken before saving must still be a valid rollback target
        // after a load — i.e. history is faithfully preserved across the disk.
        let g = sample_graph();
        let snap = g.history()[0].root;
        let mut disk = RamDisk::new(64);
        Persistence::save(&mut disk, &g).unwrap();
        let mut loaded = Persistence::load(&mut disk).unwrap().unwrap();
        loaded.rollback(snap).unwrap();
        assert_eq!(loaded.live_count(), g.history()[0].live.len());
    }

    #[test]
    fn unformatted_disk_loads_as_none() {
        let mut disk = RamDisk::new(8);
        assert!(Persistence::load(&mut disk).unwrap().is_none());
        assert!(!Persistence::is_formatted(&mut disk).unwrap());
    }

    #[test]
    fn formatted_flag_set_after_save() {
        let mut disk = RamDisk::new(64);
        Persistence::save(&mut disk, &ObjectGraph::new()).unwrap();
        assert!(Persistence::is_formatted(&mut disk).unwrap());
    }

    #[test]
    fn too_small_disk_errors() {
        let g = sample_graph();
        let mut tiny = RamDisk::new(1); // only room for the superblock
        assert_eq!(Persistence::save(&mut tiny, &g).unwrap_err(), BlockError::Full);
    }

    #[test]
    fn blob_round_trips_at_an_offset_lba() {
        let mut disk = RamDisk::new(64);
        let payload = alloc::vec![7u8, 8, 9, 250, 0, 42, 255, 1];
        Persistence::save_blob(&mut disk, 16, b"AEVFS001", &payload).unwrap();
        // The image at LBA 16 reads back exactly.
        let got = Persistence::load_blob(&mut disk, 16, b"AEVFS001").unwrap().unwrap();
        assert_eq!(got, payload);
        // A different magic / empty LBA reads as None, not an error.
        assert!(Persistence::load_blob(&mut disk, 16, b"OTHERMAG").unwrap().is_none());
        assert!(Persistence::load_blob(&mut disk, 40, b"AEVFS001").unwrap().is_none());
    }

    #[test]
    fn blob_and_graph_images_coexist_on_one_disk() {
        let g = sample_graph();
        let mut disk = RamDisk::new(128);
        Persistence::save(&mut disk, &g).unwrap(); // graph at LBA 0
        Persistence::save_blob(&mut disk, 64, b"AEVFS001", b"vfs-image").unwrap();
        // Both survive independently.
        assert_eq!(Persistence::load(&mut disk).unwrap().unwrap().root_hash(), g.root_hash());
        assert_eq!(Persistence::load_blob(&mut disk, 64, b"AEVFS001").unwrap().unwrap(), b"vfs-image");
    }

    #[test]
    fn batched_blocks_round_trip_and_validate_length() {
        let mut disk = RamDisk::new(16);
        // A three-block run written and read back in one call each.
        let mut written = alloc::vec![0u8; 3 * BLOCK_SIZE];
        for (i, b) in written.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        disk.write_blocks(2, &written).unwrap();
        let mut read = alloc::vec![0u8; 3 * BLOCK_SIZE];
        disk.read_blocks(2, &mut read).unwrap();
        assert_eq!(read, written);
        // It landed at the right LBA: block 1 is still untouched (all zero).
        let mut before = [0u8; BLOCK_SIZE];
        disk.read_block(1, &mut before).unwrap();
        assert!(before.iter().all(|&b| b == 0));
        // A non-block-multiple buffer is rejected, not silently truncated.
        assert_eq!(disk.write_blocks(2, &[0u8; 10]).unwrap_err(), BlockError::BadLength);
        assert_eq!(disk.read_blocks(2, &mut [0u8; 10]).unwrap_err(), BlockError::BadLength);
    }

    #[test]
    fn empty_graph_round_trips() {
        let g = ObjectGraph::new();
        let mut disk = RamDisk::new(8);
        Persistence::save(&mut disk, &g).unwrap();
        let loaded = Persistence::load(&mut disk).unwrap().unwrap();
        assert_eq!(loaded.stored_count(), 0);
        assert_eq!(loaded.live_count(), 0);
    }
}

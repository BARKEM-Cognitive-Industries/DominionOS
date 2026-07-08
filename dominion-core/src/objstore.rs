//! Incremental, content-addressed on-disk store for the object graph (M1, evolved).
//!
//! The monolithic [`Persistence`](crate::persist::Persistence) image rewrites *every*
//! byte of the graph on every save. But the graph is immutable and content-addressed:
//! an object, once written, never changes. So this store treats the disk as an
//! **append-only object log** — each distinct object is written exactly once at a
//! stable LBA — plus a small **manifest** (the id→location index, the live head, and
//! the commit history) that is rewritten each save, and a **double-buffered root
//! record** that names the current manifest. A shutdown therefore flushes only the
//! objects created *this session*, not the entire history.
//!
//! Layout from a `base_lba`:
//! ```text
//!   base_lba + 0 : root record, slot A   (1 block)
//!   base_lba + 1 : root record, slot B   (1 block)   ← double-buffered for atomicity
//!   base_lba + 2 : arena  (object log + manifest, append-only)
//! ```
//! The two root slots are written alternately; the loader picks the slot with the
//! highest generation whose integrity hash checks out. A torn write of the new slot
//! leaves the previous slot intact and authoritative — so a crash mid-save can never
//! corrupt what was already committed (the security/durability invariant is kept; this
//! only changes *how* bytes are laid down, never what is stored or who may read it).
//!
//! Pure and `no_std`: it drives a [`BlockDevice`] and is exercised on host with a
//! `RamDisk`, exactly like the layer it extends.

use crate::bytes::Cursor;
use crate::hash::Hash256;
use crate::object::{Commit, Object, ObjectGraph, ObjectId};
use crate::persist::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::vec::Vec;

const ROOT_MAGIC: &[u8; 8] = b"AEOBJL01";
const MANIFEST_MAGIC: &[u8; 8] = b"AEMAN001";

// Root-record field offsets (within its single 512-byte block).
const R_MAGIC: usize = 0; // 8
const R_GEN: usize = 8; // u64
const R_VFS_ROOT: usize = 16; // 32
const R_MANIFEST_LBA: usize = 48; // u64
const R_MANIFEST_BLOCKS: usize = 56; // u64
const R_LOG_END: usize = 64; // u64
const R_DATA_START: usize = 72; // u64
const R_HASH: usize = 80; // 32 (integrity over [0..80])

/// Reclaim dead space (old manifests buried in the arena by prior saves) with a full
/// rewrite once it exceeds this many blocks — keeps incremental saves from growing the
/// arena without bound.
const COMPACT_DEAD_BLOCKS: u64 = 2048; // 1 MiB

fn blocks_for(len: usize) -> u64 {
    len.div_ceil(BLOCK_SIZE) as u64
}

/// The in-memory record of what is already on disk, threaded from a [`load`](ObjStore::load)
/// through to the next [`save`](ObjStore::save) so saves can append only what is new.
#[derive(Clone, Debug)]
pub struct Manifest {
    base_lba: u64,
    /// Next free LBA in the arena (everything below this is written and immutable).
    log_end: u64,
    /// Content id → (LBA, exact byte length) for every object physically on disk.
    index: BTreeMap<ObjectId, (u64, u32)>,
    /// Generation of the root record this manifest was committed under.
    generation: u64,
}

impl Manifest {
    /// How many objects are recorded on disk.
    pub fn object_count(&self) -> usize {
        self.index.len()
    }
}

/// The incremental object-graph store.
pub struct ObjStore;

impl ObjStore {
    /// Persist `graph` (with namespace root `vfs_root`) at `base_lba`. If `prior` names a
    /// manifest for this `base_lba`, only objects absent from it are appended (the fast
    /// path); otherwise — or when dead space has accumulated, or an append would not fit
    /// — the whole arena is rewritten (compaction). On success `prior` is updated to the
    /// newly committed manifest, ready to thread into the next save.
    pub fn save(
        dev: &mut dyn BlockDevice,
        base_lba: u64,
        graph: &ObjectGraph,
        vfs_root: ObjectId,
        prior: &mut Option<Manifest>,
    ) -> Result<(), BlockError> {
        let data_start = base_lba + 2;
        let cap = dev.block_count();
        if data_start >= cap {
            return Err(BlockError::Full);
        }

        let head = graph.live_ids();
        let history = graph.commits();

        // Decide between an incremental append and a full compacting rewrite.
        let reuse = match prior.as_ref() {
            Some(m) if m.base_lba == base_lba => Some(m),
            _ => None,
        };
        // Derive the next generation so a fresh save always supersedes what is
        // already committed. With no in-memory prior, consult the on-disk root(s)
        // via `read_root` (a lost/absent manifest must not reset to gen 1 and be
        // shadowed by a surviving higher-generation root).
        let next_gen = match reuse {
            Some(m) => m.generation + 1,
            None => read_root(dev, base_lba)?.map(|r| r.generation).unwrap_or(0) + 1,
        };

        let new_manifest = if let Some(m) = reuse {
            // Fast path: every object already on disk keeps its immutable LBA, so its
            // canonical bytes are never re-used here. Encode (and re-hash) ONLY the
            // objects absent from the prior manifest — re-serialising the whole history
            // on every save was pure wasted work that scaled with total object count.
            // Walk in id-sorted order (the store iterates sorted) so the appended run is
            // deterministic without a second sort.
            let total_count = graph.stored_objects().count();
            let mut fresh: Vec<(ObjectId, Vec<u8>)> = graph
                .stored_objects()
                .filter(|(id, _)| !m.index.contains_key(*id))
                .map(|(id, o)| (*id, o.encode()))
                .collect();
            fresh.sort_by(|a, b| a.0.cmp(&b.0));

            let new_obj_blocks: u64 = fresh.iter().map(|(_, b)| blocks_for(b.len())).sum();
            let live_blocks: u64 = m.index.values().map(|(_, len)| blocks_for(*len as usize)).sum::<u64>()
                + new_obj_blocks;
            let dead_blocks = (m.log_end - data_start).saturating_sub(live_blocks);

            // Worst-case end if we append: new objects, then a manifest sized for the
            // full (reused + new) index.
            let est_manifest_blocks = blocks_for(manifest_size(total_count));
            let would_end = m.log_end + new_obj_blocks + est_manifest_blocks;

            if dead_blocks > COMPACT_DEAD_BLOCKS || would_end > cap {
                // Compaction needs every object's bytes; encode the rest now (only the
                // already-on-disk ones — `fresh` is reused as-is).
                let mut encoded = fresh;
                encoded.extend(
                    graph
                        .stored_objects()
                        .filter(|(id, _)| m.index.contains_key(*id))
                        .map(|(id, o)| (*id, o.encode())),
                );
                Self::write_full(dev, base_lba, &encoded, head, history, vfs_root, next_gen, cap)?
            } else {
                let fresh_refs: Vec<&(ObjectId, Vec<u8>)> = fresh.iter().collect();
                Self::write_append(dev, base_lba, m, &fresh_refs, head, history, vfs_root, next_gen, cap)?
            }
        } else {
            // No prior manifest: a full rewrite needs every object encoded once.
            let encoded: Vec<(ObjectId, Vec<u8>)> =
                graph.stored_objects().map(|(id, o)| (*id, o.encode())).collect();
            Self::write_full(dev, base_lba, &encoded, head, history, vfs_root, next_gen, cap)?
        };

        *prior = Some(new_manifest);
        Ok(())
    }

    /// Append `fresh` objects after the prior arena, then write a new manifest and flip
    /// the root. Reused objects keep their existing (immutable) LBAs.
    #[allow(clippy::too_many_arguments)]
    fn write_append(
        dev: &mut dyn BlockDevice,
        base_lba: u64,
        prior: &Manifest,
        fresh: &[&(ObjectId, Vec<u8>)],
        head: &[ObjectId],
        history: &[Commit],
        vfs_root: ObjectId,
        gen: u64,
        cap: u64,
    ) -> Result<Manifest, BlockError> {
        let data_start = base_lba + 2;
        let mut index = prior.index.clone();

        // Pack the new objects into one contiguous, block-aligned buffer and assign each
        // a stable LBA — then a single batched write lands them all at once.
        let mut buf = Vec::new();
        let mut lba = prior.log_end;
        for (id, bytes) in fresh {
            index.insert(*id, (lba, bytes.len() as u32));
            append_block_aligned(&mut buf, bytes);
            lba += blocks_for(bytes.len());
        }
        if lba > cap {
            return Err(BlockError::Full);
        }
        if !buf.is_empty() {
            dev.write_blocks(prior.log_end, &buf)?;
        }

        Self::write_manifest_and_root(dev, base_lba, data_start, lba, &index, head, history, vfs_root, gen, cap)
    }

    /// Rewrite the whole arena from `data_start`: every object laid out fresh and
    /// contiguous (id-sorted), reclaiming any dead space, then a new manifest and root.
    #[allow(clippy::too_many_arguments)]
    fn write_full(
        dev: &mut dyn BlockDevice,
        base_lba: u64,
        all: &[(ObjectId, Vec<u8>)],
        head: &[ObjectId],
        history: &[Commit],
        vfs_root: ObjectId,
        gen: u64,
        cap: u64,
    ) -> Result<Manifest, BlockError> {
        let data_start = base_lba + 2;
        let mut sorted: Vec<&(ObjectId, Vec<u8>)> = all.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut index = BTreeMap::new();
        let mut buf = Vec::new();
        let mut lba = data_start;
        for (id, bytes) in sorted {
            index.insert(*id, (lba, bytes.len() as u32));
            append_block_aligned(&mut buf, bytes);
            lba += blocks_for(bytes.len());
        }
        if lba > cap {
            return Err(BlockError::Full);
        }
        if !buf.is_empty() {
            dev.write_blocks(data_start, &buf)?;
        }

        Self::write_manifest_and_root(dev, base_lba, data_start, lba, &index, head, history, vfs_root, gen, cap)
    }

    /// Common tail: serialise + write the manifest at `manifest_lba`, then commit by
    /// writing the root record (with integrity hash) into the alternate slot.
    #[allow(clippy::too_many_arguments)]
    fn write_manifest_and_root(
        dev: &mut dyn BlockDevice,
        base_lba: u64,
        data_start: u64,
        manifest_lba: u64,
        index: &BTreeMap<ObjectId, (u64, u32)>,
        head: &[ObjectId],
        history: &[Commit],
        vfs_root: ObjectId,
        gen: u64,
        cap: u64,
    ) -> Result<Manifest, BlockError> {
        let manifest = encode_manifest(index, head, history);
        let manifest_blocks = blocks_for(manifest.len());
        let log_end = manifest_lba + manifest_blocks;
        if log_end > cap {
            return Err(BlockError::Full);
        }
        let mut mbuf = Vec::new();
        append_block_aligned(&mut mbuf, &manifest);
        dev.write_blocks(manifest_lba, &mbuf)?;

        let slot = base_lba + (gen % 2);
        let mut rec = [0u8; BLOCK_SIZE];
        rec[R_MAGIC..R_MAGIC + 8].copy_from_slice(ROOT_MAGIC);
        rec[R_GEN..R_GEN + 8].copy_from_slice(&gen.to_le_bytes());
        rec[R_VFS_ROOT..R_VFS_ROOT + 32].copy_from_slice(&vfs_root.0);
        rec[R_MANIFEST_LBA..R_MANIFEST_LBA + 8].copy_from_slice(&manifest_lba.to_le_bytes());
        rec[R_MANIFEST_BLOCKS..R_MANIFEST_BLOCKS + 8].copy_from_slice(&manifest_blocks.to_le_bytes());
        rec[R_LOG_END..R_LOG_END + 8].copy_from_slice(&log_end.to_le_bytes());
        rec[R_DATA_START..R_DATA_START + 8].copy_from_slice(&data_start.to_le_bytes());
        let h = Hash256::of(&rec[..R_HASH]);
        rec[R_HASH..R_HASH + 32].copy_from_slice(&h.0);
        dev.write_block(slot, &rec)?;

        Ok(Manifest { base_lba, log_end, index: index.clone(), generation: gen })
    }

    /// Load the graph + namespace root + manifest from `base_lba`, or `Ok(None)` if no
    /// valid root record is present (a fresh disk, not an error). A content-addressed
    /// integrity check rejects any object whose bytes do not hash to its indexed id.
    pub fn load(
        dev: &mut dyn BlockDevice,
        base_lba: u64,
    ) -> Result<Option<(ObjectGraph, ObjectId, Manifest)>, BlockError> {
        let data_start = base_lba + 2;
        let Some(root) = read_root(dev, base_lba)? else {
            return Ok(None);
        };

        // One batched read covers the whole arena; the manifest and every object are
        // sliced out of it by offset.
        let arena_blocks = root.log_end.saturating_sub(data_start);
        let mut arena = alloc::vec![0u8; arena_blocks as usize * BLOCK_SIZE];
        if arena_blocks > 0 {
            dev.read_blocks(data_start, &mut arena)?;
        }
        let slice = |lba: u64, len: usize| -> Option<&[u8]> {
            let off = lba.checked_sub(data_start)? as usize * BLOCK_SIZE;
            arena.get(off..off + len)
        };

        // Parse the manifest.
        let m_off = (root.manifest_lba.checked_sub(data_start).ok_or(BlockError::DeviceFault)?) as usize
            * BLOCK_SIZE;
        let m_len = root.manifest_blocks as usize * BLOCK_SIZE;
        let mbytes = arena.get(m_off..m_off + m_len).ok_or(BlockError::DeviceFault)?;
        let Some((index, head, history)) = decode_manifest(mbytes) else {
            return Ok(None);
        };

        // Materialise every object, verifying its content hash against its indexed id.
        let mut objects = Vec::with_capacity(index.len());
        for (id, (lba, len)) in &index {
            let bytes = slice(*lba, *len as usize).ok_or(BlockError::DeviceFault)?;
            let Ok(obj) = Object::decode(bytes) else {
                return Ok(None);
            };
            if obj.id() != *id {
                return Ok(None); // tampered or torn — refuse rather than load bad state
            }
            objects.push(obj);
        }

        let graph = ObjectGraph::restore(objects, head, history);
        let manifest = Manifest { base_lba, log_end: root.log_end, index, generation: root.generation };
        Ok(Some((graph, root.vfs_root, manifest)))
    }
}

// ── manifest codec ──
//
// The head + history change every commit but are small relative to object bodies, so
// the manifest (id→location index + head + history) is rewritten in full each save and
// the writer is handed head/history directly from the live graph.

fn manifest_size(object_count: usize) -> usize {
    // A generous upper bound for sizing decisions: magic + count + per-object index
    // entry (32 + 8 + 4) + slack for head/history.
    8 + 4 + object_count * (32 + 8 + 4) + 4096
}

fn encode_manifest(
    index: &BTreeMap<ObjectId, (u64, u32)>,
    head: &[ObjectId],
    history: &[Commit],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MANIFEST_MAGIC);
    out.extend_from_slice(&(index.len() as u32).to_le_bytes());
    for (id, (lba, len)) in index {
        out.extend_from_slice(&id.0);
        out.extend_from_slice(&lba.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }
    out.extend_from_slice(&(head.len() as u32).to_le_bytes());
    for id in head {
        out.extend_from_slice(&id.0);
    }
    out.extend_from_slice(&(history.len() as u32).to_le_bytes());
    for c in history {
        out.extend_from_slice(&c.root.0);
        out.extend_from_slice(&c.parent.0);
        out.extend_from_slice(&(c.live.len() as u32).to_le_bytes());
        for id in &c.live {
            out.extend_from_slice(&id.0);
        }
        out.extend_from_slice(&(c.message.len() as u32).to_le_bytes());
        out.extend_from_slice(c.message.as_bytes());
    }
    out
}

type ManifestParts = (BTreeMap<ObjectId, (u64, u32)>, Vec<ObjectId>, Vec<Commit>);

fn decode_manifest(bytes: &[u8]) -> Option<ManifestParts> {
    let mut r = Cursor::new(bytes);
    if r.take(8)? != MANIFEST_MAGIC {
        return None;
    }
    let count = r.read_u32_le()? as usize;
    let mut index = BTreeMap::new();
    for _ in 0..count {
        let id = r.read_hash()?;
        let lba = r.read_u64_le()?;
        let len = r.read_u32_le()?;
        index.insert(id, (lba, len));
    }
    // The manifest is NOT integrity-protected (the root hash covers only the root
    // record). Never pre-size a Vec from these untrusted counts — a single flipped
    // length byte would request a ~4-billion-element allocation and abort the
    // allocator. Push while reading instead; the `?`-checked cursor reads bound the
    // loop and turn corruption into a graceful `None`.
    let head_len = r.read_u32_le()? as usize;
    let mut head = Vec::new();
    for _ in 0..head_len {
        head.push(r.read_hash()?);
    }
    let hist_len = r.read_u32_le()? as usize;
    let mut history = Vec::new();
    for _ in 0..hist_len {
        let root = r.read_hash()?;
        let parent = r.read_hash()?;
        let live_len = r.read_u32_le()? as usize;
        let mut live = Vec::new();
        for _ in 0..live_len {
            live.push(r.read_hash()?);
        }
        let msg_len = r.read_u32_le()? as usize;
        let message = core::str::from_utf8(r.take(msg_len)?).ok()?.to_string();
        history.push(Commit { root, parent, live, message });
    }
    Some((index, head, history))
}

struct RootRecord {
    generation: u64,
    vfs_root: ObjectId,
    manifest_lba: u64,
    manifest_blocks: u64,
    log_end: u64,
}

/// Read both root slots and return the newest one whose magic and integrity hash check
/// out — so a half-written new slot is ignored in favour of the last good commit.
fn read_root(dev: &mut dyn BlockDevice, base_lba: u64) -> Result<Option<RootRecord>, BlockError> {
    let mut best: Option<RootRecord> = None;
    for slot in 0..2u64 {
        let mut rec = [0u8; BLOCK_SIZE];
        dev.read_block(base_lba + slot, &mut rec)?;
        if &rec[R_MAGIC..R_MAGIC + 8] != ROOT_MAGIC {
            continue;
        }
        let h = Hash256::of(&rec[..R_HASH]);
        if rec[R_HASH..R_HASH + 32] != h.0 {
            continue;
        }
        let generation = u64::from_le_bytes(rec[R_GEN..R_GEN + 8].try_into().unwrap());
        let mut vfs = [0u8; 32];
        vfs.copy_from_slice(&rec[R_VFS_ROOT..R_VFS_ROOT + 32]);
        let candidate = RootRecord {
            generation,
            vfs_root: Hash256(vfs),
            manifest_lba: u64::from_le_bytes(rec[R_MANIFEST_LBA..R_MANIFEST_LBA + 8].try_into().unwrap()),
            manifest_blocks: u64::from_le_bytes(rec[R_MANIFEST_BLOCKS..R_MANIFEST_BLOCKS + 8].try_into().unwrap()),
            log_end: u64::from_le_bytes(rec[R_LOG_END..R_LOG_END + 8].try_into().unwrap()),
        };
        if best.as_ref().map(|b| candidate.generation > b.generation).unwrap_or(true) {
            best = Some(candidate);
        }
    }
    Ok(best)
}

/// Append `bytes` to `buf` and zero-pad up to the next block boundary.
fn append_block_aligned(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(bytes);
    let pad = (BLOCK_SIZE - bytes.len() % BLOCK_SIZE) % BLOCK_SIZE;
    buf.resize(buf.len() + pad, 0);
}

// Manifest decoding uses `crate::bytes::Cursor` — the shared implementation
// imported at the top of this file.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Datum;
    use crate::persist::RamDisk;

    const BASE: u64 = 4;

    fn note(kind: &str, body: &str) -> Object {
        Object::new(kind).with("body", Datum::Text(body.to_string()))
    }

    #[test]
    fn save_then_load_round_trips_graph_and_root() {
        let mut g = ObjectGraph::new();
        g.put(note("Doc", "hello"));
        g.put(note("Doc", "world"));
        g.commit("first");
        let vfs_root = Hash256::of(b"namespace-root");

        let mut disk = RamDisk::new(256);
        let mut prior = None;
        ObjStore::save(&mut disk, BASE, &g, vfs_root, &mut prior).unwrap();

        let (g2, root2, _m) = ObjStore::load(&mut disk, BASE).unwrap().unwrap();
        assert_eq!(root2, vfs_root);
        assert_eq!(g2.stored_count(), g.stored_count());
        assert_eq!(g2.root_hash(), g.root_hash());
        assert_eq!(g2.commits().len(), 1);
    }

    #[test]
    fn second_save_only_appends_the_new_object() {
        let mut disk = RamDisk::new(512);
        let mut g = ObjectGraph::new();
        g.put(note("A", "one"));
        let mut prior = None;
        ObjStore::save(&mut disk, BASE, &g, Hash256::ZERO, &mut prior).unwrap();
        let m1 = prior.clone().unwrap();

        // A second, larger save: the original object must keep its on-disk location
        // (immutable, content-addressed) — only the new object + manifest are appended.
        g.put(note("B", "two"));
        ObjStore::save(&mut disk, BASE, &g, Hash256::ZERO, &mut prior).unwrap();
        let m2 = prior.clone().unwrap();

        let id_a = note("A", "one").id();
        assert_eq!(m1.index.get(&id_a), m2.index.get(&id_a), "A did not move");
        assert!(m2.log_end > m1.log_end, "arena grew by the appended object");
        assert_eq!(m2.index.len(), 2);
        assert_eq!(m2.generation, m1.generation + 1);

        let (g2, _r, _m) = ObjStore::load(&mut disk, BASE).unwrap().unwrap();
        assert_eq!(g2.stored_count(), 2);
    }

    #[test]
    fn a_torn_new_root_falls_back_to_the_previous_commit() {
        let mut disk = RamDisk::new(512);
        let mut g = ObjectGraph::new();
        g.put(note("A", "kept"));
        let mut prior = None;
        ObjStore::save(&mut disk, BASE, &g, Hash256::ZERO, &mut prior).unwrap();
        let gen1 = prior.clone().unwrap().generation;

        g.put(note("B", "lost-if-torn"));
        ObjStore::save(&mut disk, BASE, &g, Hash256::ZERO, &mut prior).unwrap();
        let gen2 = prior.clone().unwrap().generation;

        // Simulate a crash that left the just-written root slot half-baked.
        let slot = BASE + (gen2 % 2);
        let mut blk = [0u8; BLOCK_SIZE];
        disk.read_block(slot, &mut blk).unwrap();
        blk[R_HASH] ^= 0xFF;
        disk.write_block(slot, &blk).unwrap();

        // Boot rolls back to the last fully-committed generation, intact.
        let (g2, _r, m) = ObjStore::load(&mut disk, BASE).unwrap().unwrap();
        assert_eq!(m.generation, gen1);
        assert_eq!(g2.stored_count(), 1);
    }

    #[test]
    fn a_tampered_object_is_refused() {
        let mut disk = RamDisk::new(256);
        let mut g = ObjectGraph::new();
        g.put(note("Secret", "classified"));
        let mut prior = None;
        ObjStore::save(&mut disk, BASE, &g, Hash256::ZERO, &mut prior).unwrap();

        // Flip a content byte: its hash no longer matches the indexed id.
        let (lba, len) = *prior.unwrap().index.values().next().unwrap();
        let mut blk = [0u8; BLOCK_SIZE];
        disk.read_block(lba, &mut blk).unwrap();
        blk[len as usize - 1] ^= 0xFF;
        disk.write_block(lba, &blk).unwrap();

        // Content addressing catches it; the store refuses to load bad state.
        assert!(ObjStore::load(&mut disk, BASE).unwrap().is_none());
    }

    #[test]
    fn many_incremental_saves_stay_consistent() {
        let mut disk = RamDisk::new(2048);
        let mut g = ObjectGraph::new();
        let mut prior = None;
        let mut last_end = BASE + 2;
        for i in 0..20 {
            g.put(note("Item", &alloc::format!("body-{i}")));
            ObjStore::save(&mut disk, BASE, &g, Hash256::ZERO, &mut prior).unwrap();
            let end = prior.clone().unwrap().log_end;
            assert!(end >= last_end);
            last_end = end;
        }
        let (g2, _r, m) = ObjStore::load(&mut disk, BASE).unwrap().unwrap();
        assert_eq!(g2.stored_count(), 20);
        assert_eq!(m.object_count(), 20);
    }

    #[test]
    fn no_root_record_loads_as_none() {
        let mut disk = RamDisk::new(64);
        assert!(ObjStore::load(&mut disk, BASE).unwrap().is_none());
    }
}

//! Boot/install/runtime **debug log capture** — a always-on, allocation-free ring
//! buffer that every `serial_println!` is teed into, plus a best-effort persist of the
//! captured log to a writable disk so it can be recovered after a bare-metal boot.
//!
//! Why a `const`-initialised static array (not a `Vec`): the very first serial output
//! happens during early boot **before the heap allocator is up**, so the ring must live
//! in BSS and never allocate on the capture path. `snapshot`/`persist` (called late,
//! once the heap exists) are the only heap users.
//!
//! Recovery: [`persist_best_effort`] writes the captured text to the **tail** of the
//! data disk (away from the object store, which grows from the front) as a plain-text
//! blob behind a tiny `AELOG001` superblock. The host tool `read-bootlog.ps1` reads it
//! straight out of the raw image — so after booting on real hardware you can pull the
//! full boot/run log off the disk and hand it over.
//!
//! Honest scope: persisting to the *boot USB itself* at runtime needs a USB
//! mass-storage stack (xHCI + USB-MSC + FAT) — not built. This persists to whatever
//! writable block device the kernel has (virtio data disk under QEMU; an internal
//! AHCI/NVMe disk once those drivers land). The capture + on-serial mirror always work.

use spin::Mutex;

/// Capacity of the in-memory log ring (BSS, zero-initialised). 128 KiB covers a full
/// boot + selftest + a good window of runtime; older bytes wrap.
const CAP: usize = 128 * 1024;

/// Superblock magic for the persisted plain-text log blob.
pub const LOG_MAGIC: &[u8; 8] = b"AELOG001";

/// Sectors reserved at the tail of the disk for the log (512 KiB) — the log is written
/// at `block_count - LOG_RESERVE_SECTORS` so it never collides with the object store.
const LOG_RESERVE_SECTORS: u64 = 1024;

struct Ring {
    buf: [u8; CAP],
    pos: usize,
    wrapped: bool,
}

impl Ring {
    const fn new() -> Ring {
        Ring { buf: [0u8; CAP], pos: 0, wrapped: false }
    }

    fn append(&mut self, data: &[u8]) {
        for &b in data {
            self.buf[self.pos] = b;
            self.pos += 1;
            if self.pos == CAP {
                self.pos = 0;
                self.wrapped = true;
            }
        }
    }

    fn total(&self) -> usize {
        if self.wrapped {
            CAP
        } else {
            self.pos
        }
    }
}

static RING: Mutex<Ring> = Mutex::new(Ring::new());

/// Tee raw bytes into the log ring. Allocation-free and lock-cheap — safe to call from
/// the serial hot path during the earliest boot, before the heap exists.
pub fn append(data: &[u8]) {
    RING.lock().append(data);
}

/// Total bytes currently captured (saturating at the ring capacity).
pub fn captured_len() -> usize {
    RING.lock().total()
}

/// A chronological copy of the captured log (oldest→newest). Allocates, so only call
/// once the heap is up (shell, persist, end of boot).
pub fn snapshot() -> alloc::vec::Vec<u8> {
    let r = RING.lock();
    if r.wrapped {
        let mut out = alloc::vec::Vec::with_capacity(CAP);
        out.extend_from_slice(&r.buf[r.pos..]);
        out.extend_from_slice(&r.buf[..r.pos]);
        out
    } else {
        r.buf[..r.pos].to_vec()
    }
}

/// The last `n` bytes of the captured log as a string (for a `log` shell command).
pub fn tail_string(n: usize) -> alloc::string::String {
    let snap = snapshot();
    let start = snap.len().saturating_sub(n);
    alloc::string::String::from_utf8_lossy(&snap[start..]).into_owned()
}

/// The LBA the log lives at — the tail of the device, away from the object store.
fn tail_lba(dev: &dyn dominion_core::persist::BlockDevice) -> Option<u64> {
    let cap = dev.block_count();
    if cap <= LOG_RESERVE_SECTORS + 2 {
        None
    } else {
        Some(cap - LOG_RESERVE_SECTORS)
    }
}

/// Is the log region unused — i.e. all-zero or already one of our logs? Used to make the
/// *automatic* persist safe: it never overwrites real data on a user's disk, only an
/// empty tail or a prior DominionOS log. (The explicit `log save` bypasses this.)
fn tail_is_safe(dev: &mut dyn dominion_core::persist::BlockDevice, lba: u64) -> bool {
    let mut sb = [0u8; 512];
    if dev.read_block(lba, &mut sb).is_err() {
        return false;
    }
    sb[..8] == *LOG_MAGIC || sb[..8].iter().all(|&b| b == 0)
}

/// Persist the captured log to the **tail** of the given block device as a plain-text
/// blob behind an `AELOG001` superblock. Returns the LBA it was written to on success.
/// Unconditional — used by the explicit `log save`, where the user has opted in.
pub fn persist_to(dev: &mut dyn dominion_core::persist::BlockDevice) -> Option<u64> {
    let snap = snapshot();
    if snap.is_empty() {
        return None;
    }
    let lba = tail_lba(dev)?;
    match dominion_core::persist::Persistence::save_blob(dev, lba, LOG_MAGIC, &snap) {
        Ok(()) => Some(lba),
        Err(_) => None,
    }
}

/// Like [`persist_to`] but **only** if the tail region is unused (empty or a prior log),
/// so an automatic save can never clobber real user data on a disk it doesn't own.
pub fn persist_to_if_safe(dev: &mut dyn dominion_core::persist::BlockDevice) -> Option<u64> {
    let lba = tail_lba(dev)?;
    if tail_is_safe(dev, lba) {
        persist_to(dev)
    } else {
        None
    }
}

/// Read back a previously [`persist_to`]ed log from the tail of the device (the host
/// extractor does the same off the raw image). `None` if no log blob is present.
pub fn read_back(dev: &mut dyn dominion_core::persist::BlockDevice) -> Option<alloc::vec::Vec<u8>> {
    let lba = tail_lba(dev)?;
    dominion_core::persist::Persistence::load_blob(dev, lba, LOG_MAGIC).ok().flatten()
}

/// **Automatic** best-effort persist (end-of-boot, shutdown, panic). Guarded: it only
/// writes when the disk's tail is empty or already holds an DominionOS log, so it can
/// never overwrite real user data on a disk DominionOS doesn't own. No-ops with no disk.
pub fn persist_best_effort() -> bool {
    crate::block::with_log_device(|dev, _is_real| persist_to_if_safe(dev).is_some())
}

/// **Explicit** persist (the `log save` command) — the user has opted in, so this
/// writes unconditionally to the log device tail. Returns true on success. Targets the
/// preferred removable USB when present (see [`crate::block::with_log_device`]).
pub fn persist_force() -> bool {
    crate::block::with_log_device(|dev, _is_real| persist_to(dev).is_some())
}

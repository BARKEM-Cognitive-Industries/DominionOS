//! Backup, sync & fleet replication policies
//! (`docs/implementation/backup-sync-and-device-fleet.md`).
//!
//! [`crate::backup`] gives zero-plaintext, content-addressed backup + a CRDT fleet index;
//! this module adds the *policies* a real fleet needs:
//!
//! * [`BackupPolicy`] — the **3-2-1 rule** (3 copies, 2 media types, ≥1 offsite) as a
//!   checkable invariant, plus a periodic **DST restore-test** schedule.
//! * [`sync_plan`] — **sync over NDN/DominionLink** by content id: a set-difference yields
//!   exactly what to push and pull, deduplicated by hash (immutable objects never conflict).
//! * [`OfflineQueue`] — **offline-first**: writes queue while disconnected and replay in
//!   order on reconnect (the index merge is already commutative, so this is safe).
//! * [`SelectiveSync`] — **capability-scoped selective sync**: only objects whose scope a
//!   device holds a capability for are synced to it.
//!
//! Pure, safe `no_std`. Host-tested.

use crate::hash::Hash256;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

// ───────────────────────── 3-2-1 backup policy ─────────────────────────

/// A backup configuration, checked against the 3-2-1 rule.
#[derive(Clone, Copy, Debug)]
pub struct BackupPolicy {
    /// Total number of copies kept.
    pub copies: u32,
    /// Distinct storage media types (e.g. local disk + cloud + cold storage).
    pub media_types: u32,
    /// Copies held off-site.
    pub offsite: u32,
    /// How often (µs) a restore-test should run.
    pub restore_test_interval_us: u64,
}

impl BackupPolicy {
    /// The canonical 3-2-1 policy with a weekly restore test.
    pub fn three_two_one() -> BackupPolicy {
        BackupPolicy { copies: 3, media_types: 2, offsite: 1, restore_test_interval_us: 7 * 24 * 3600 * 1_000_000 }
    }

    /// True iff the configuration satisfies 3-2-1 (≥3 copies, ≥2 media, ≥1 offsite).
    pub fn satisfies_321(&self) -> bool {
        self.copies >= 3 && self.media_types >= 2 && self.offsite >= 1
    }

    /// Whether a periodic restore-test is due, given the last test time and now.
    pub fn restore_test_due(&self, last_test_us: u64, now_us: u64) -> bool {
        now_us.saturating_sub(last_test_us) >= self.restore_test_interval_us
    }
}

// ───────────────────────── sync over NDN (by content id) ─────────────────────────

/// What a sync needs to do, computed from two content-id sets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyncPlan {
    /// Ids the local side has that the remote lacks (push these).
    pub to_push: Vec<Hash256>,
    /// Ids the remote has that the local lacks (pull these).
    pub to_pull: Vec<Hash256>,
}

/// Compute the sync plan between a local and a remote object set. Immutable,
/// content-addressed objects never conflict — only the set difference moves, and a hash
/// already present on either side is skipped (free dedup).
pub fn sync_plan(local: &BTreeSet<Hash256>, remote: &BTreeSet<Hash256>) -> SyncPlan {
    let to_push: Vec<Hash256> = local.difference(remote).copied().collect();
    let to_pull: Vec<Hash256> = remote.difference(local).copied().collect();
    SyncPlan { to_push, to_pull }
}

// ───────────────────────── offline-first write queue ─────────────────────────

/// A write buffered while offline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueuedWrite {
    pub id: Hash256,
    pub bytes: Vec<u8>,
}

/// An offline-first write queue: writes accumulate while disconnected and replay in order
/// on reconnect. Because the fleet index merge is commutative (CRDT), replay is safe.
#[derive(Default)]
pub struct OfflineQueue {
    online: bool,
    pending: Vec<QueuedWrite>,
}

impl OfflineQueue {
    pub fn new(online: bool) -> OfflineQueue {
        OfflineQueue { online, pending: Vec::new() }
    }

    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }

    pub fn is_online(&self) -> bool {
        self.online
    }

    /// Record a write. While offline it queues; while online it is ready to apply now.
    /// Returns whether it was applied immediately (online) vs queued (offline).
    pub fn write(&mut self, id: Hash256, bytes: &[u8]) -> bool {
        if self.online {
            true
        } else {
            self.pending.push(QueuedWrite { id, bytes: bytes.to_vec() });
            false
        }
    }

    /// Number of queued writes.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Reconnect and drain the queue in order for application. Marks online.
    pub fn reconcile(&mut self) -> Vec<QueuedWrite> {
        self.online = true;
        core::mem::take(&mut self.pending)
    }
}

// ───────────────────────── capability-scoped selective sync ─────────────────────────

/// A scope a device is allowed to sync (e.g. a domain / folder id).
pub type Scope = u64;

/// Capability-scoped selective sync: a device syncs **only** objects in scopes it has been
/// granted, so a phone needn't replicate the whole fleet's data.
#[derive(Default)]
pub struct SelectiveSync {
    granted: BTreeSet<Scope>,
    object_scope: BTreeMap<Hash256, Scope>,
}

impl SelectiveSync {
    pub fn new() -> SelectiveSync {
        SelectiveSync { granted: BTreeSet::new(), object_scope: BTreeMap::new() }
    }

    /// Grant this device the right to sync `scope`.
    pub fn grant_scope(&mut self, scope: Scope) {
        self.granted.insert(scope);
    }

    /// Tag an object with its scope.
    pub fn tag(&mut self, id: Hash256, scope: Scope) {
        self.object_scope.insert(id, scope);
    }

    /// Should this object sync to this device? Only if its scope was granted.
    pub fn should_sync(&self, id: &Hash256) -> bool {
        self.object_scope.get(id).map(|s| self.granted.contains(s)).unwrap_or(false)
    }

    /// Filter a candidate id set down to the syncable subset.
    pub fn filter(&self, ids: &BTreeSet<Hash256>) -> BTreeSet<Hash256> {
        ids.iter().filter(|id| self.should_sync(id)).copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(b: &[u8]) -> Hash256 {
        Hash256::of(b)
    }

    #[test]
    fn three_two_one_policy_and_restore_test_schedule() {
        let p = BackupPolicy::three_two_one();
        assert!(p.satisfies_321());
        // A degenerate single-copy config fails 3-2-1.
        let bad = BackupPolicy { copies: 1, media_types: 1, offsite: 0, restore_test_interval_us: 0 };
        assert!(!bad.satisfies_321());
        // Restore test becomes due after the interval.
        assert!(!p.restore_test_due(1000, 1000));
        assert!(p.restore_test_due(0, p.restore_test_interval_us + 1));
    }

    #[test]
    fn sync_plan_is_content_addressed_set_difference() {
        let local: BTreeSet<Hash256> = [h(b"a"), h(b"b"), h(b"c")].into_iter().collect();
        let remote: BTreeSet<Hash256> = [h(b"b"), h(b"c"), h(b"d")].into_iter().collect();
        let plan = sync_plan(&local, &remote);
        assert_eq!(plan.to_push, alloc::vec![h(b"a")]); // only 'a' is local-only
        assert_eq!(plan.to_pull, alloc::vec![h(b"d")]); // only 'd' is remote-only
        // Shared objects (b, c) move in neither direction (dedup).
        assert!(!plan.to_push.contains(&h(b"b")));
        assert!(!plan.to_pull.contains(&h(b"c")));
    }

    #[test]
    fn offline_queue_buffers_then_replays_in_order() {
        let mut q = OfflineQueue::new(false);
        assert!(!q.write(h(b"1"), b"one")); // queued
        assert!(!q.write(h(b"2"), b"two"));
        assert_eq!(q.pending(), 2);
        let drained = q.reconcile();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].id, h(b"1")); // order preserved
        assert!(q.is_online());
        // Now online, writes apply immediately.
        assert!(q.write(h(b"3"), b"three"));
        assert_eq!(q.pending(), 0);
    }

    #[test]
    fn selective_sync_only_replicates_granted_scopes() {
        let mut s = SelectiveSync::new();
        s.grant_scope(1);
        s.tag(h(b"personal"), 1);
        s.tag(h(b"work"), 2); // not granted to this device
        assert!(s.should_sync(&h(b"personal")));
        assert!(!s.should_sync(&h(b"work")));
        let all: BTreeSet<Hash256> = [h(b"personal"), h(b"work")].into_iter().collect();
        let syncable = s.filter(&all);
        assert!(syncable.contains(&h(b"personal")));
        assert!(!syncable.contains(&h(b"work")));
    }
}

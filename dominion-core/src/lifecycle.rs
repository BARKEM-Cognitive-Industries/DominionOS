//! Data lifecycle, deletion & compliance (see
//! `docs/security/data-lifecycle-and-compliance.md`).
//!
//! "Delete" on an encrypted, content-addressed store cannot mean "overwrite the
//! bytes" — copies and caches may exist. It means **cryptographic erasure**:
//! destroy the key and the ciphertext is permanently unreadable, everywhere, at
//! once (the same crypto-GC the vault uses in [`crate::vault`]). This module is the
//! policy engine on top of that primitive:
//!
//! * Each datum has a **class**, a creation time and a **retention** period.
//! * A **legal hold** suspends deletion for litigation/compliance regardless of
//!   retention.
//! * A **sweep** at the current time cryptographically erases everything past
//!   retention that is not on hold.
//! * Every lifecycle event is appended to an **audit trail** (a hash-chained log)
//!   so deletion is itself provable and tamper-evident.
//!
//! Pure, safe `no_std`, host-tested.

use crate::hash::Hash256;
use crate::time::Micros;
use alloc::vec::Vec;

/// Data classification, which drives default policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataClass {
    Ephemeral,
    UserContent,
    Sensitive,
    SystemLog,
}

/// Lifecycle state of one datum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Live,
    OnHold,
    Erased,
}

#[derive(Clone, Debug)]
struct Item {
    id: u64,
    class: DataClass,
    created: Micros,
    retention: Micros,
    /// Per-datum key handle; erasing it renders the ciphertext unrecoverable.
    key_alive: bool,
    holds: u32,
    state: State,
}

/// An entry in the tamper-evident audit log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditEntry {
    pub event: AuditEvent,
    pub item: u64,
    pub at: Micros,
    /// Hash chaining this entry to all prior entries.
    pub chain: Hash256,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditEvent {
    Created,
    HoldPlaced,
    HoldReleased,
    Erased,
    EraseBlockedByHold,
}

/// Why a datum was erased — recorded on its [`Tombstone`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TombstoneReason {
    /// Manual deletion (a subject-access erase request, an operator action).
    Manual,
    /// Retention window elapsed and the policy sweep crypto-erased it.
    RetentionExpired,
    /// Consent for the datum's purpose was withdrawn.
    ConsentWithdrawn,
}

/// A deletion **tombstone**: a durable record that a datum was erased, retaining only
/// the *fact, time and authority* of the deletion — **never the content**, which is
/// cryptographically gone. Provenance keeps tombstones so deletion is itself auditable
/// (you can prove *that* and *when* something was deleted, and *by whom*, without being
/// able to recover *what*).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tombstone {
    /// The id of the now-erased datum.
    pub id: u64,
    /// The datum's classification (metadata only — not content).
    pub class: DataClass,
    /// When the erasure happened.
    pub erased_at: Micros,
    /// A fingerprint of the authority that ordered the erasure.
    pub authority: Hash256,
    /// Why it was erased.
    pub reason: TombstoneReason,
}

/// The lifecycle manager: items, policy, and the audit trail.
#[derive(Default)]
pub struct DataLifecycle {
    items: Vec<Item>,
    audit: Vec<AuditEntry>,
    tombstones: Vec<Tombstone>,
}

impl DataLifecycle {
    pub fn new() -> DataLifecycle {
        DataLifecycle { items: Vec::new(), audit: Vec::new(), tombstones: Vec::new() }
    }

    fn log(&mut self, event: AuditEvent, item: u64, at: Micros) {
        let prev = self.audit.last().map(|e| e.chain).unwrap_or(Hash256::of(b"audit-genesis"));
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(&prev.0);
        input.extend_from_slice(&item.to_le_bytes());
        input.extend_from_slice(&at.to_le_bytes());
        input.push(event as u8);
        let chain = Hash256::of(&input);
        self.audit.push(AuditEntry { event, item, at, chain });
    }

    /// Register a datum with its class, creation time and retention window.
    pub fn register(&mut self, id: u64, class: DataClass, created: Micros, retention: Micros) {
        self.items.push(Item {
            id,
            class,
            created,
            retention,
            key_alive: true,
            holds: 0,
            state: State::Live,
        });
        self.log(AuditEvent::Created, id, created);
    }

    fn item_mut(&mut self, id: u64) -> Option<&mut Item> {
        self.items.iter_mut().find(|i| i.id == id)
    }

    pub fn state(&self, id: u64) -> Option<State> {
        self.items.iter().find(|i| i.id == id).map(|i| i.state)
    }

    /// The datum's classification (drives default retention/policy decisions).
    pub fn class(&self, id: u64) -> Option<DataClass> {
        self.items.iter().find(|i| i.id == id).map(|i| i.class)
    }

    /// Is the datum's key still alive (i.e. is it still readable)?
    pub fn readable(&self, id: u64) -> bool {
        self.items.iter().find(|i| i.id == id).map(|i| i.key_alive).unwrap_or(false)
    }

    /// Place a legal hold (refcounted: several matters can hold the same datum).
    pub fn place_hold(&mut self, id: u64, at: Micros) -> bool {
        if let Some(it) = self.item_mut(id) {
            if it.state == State::Erased {
                return false;
            }
            it.holds += 1;
            it.state = State::OnHold;
            self.log(AuditEvent::HoldPlaced, id, at);
            true
        } else {
            false
        }
    }

    /// Release one legal hold; the datum returns to Live when the last is lifted.
    pub fn release_hold(&mut self, id: u64, at: Micros) -> bool {
        if let Some(it) = self.item_mut(id) {
            if it.holds == 0 {
                return false;
            }
            it.holds -= 1;
            if it.holds == 0 && it.state == State::OnHold {
                it.state = State::Live;
            }
            self.log(AuditEvent::HoldReleased, id, at);
            true
        } else {
            false
        }
    }

    /// Cryptographically erase a datum *now* (manual deletion). Refused if on hold.
    pub fn erase(&mut self, id: u64, at: Micros) -> bool {
        self.erase_by(id, at, Hash256::of(b"policy-engine"), TombstoneReason::Manual)
    }

    /// Crypto-erase a datum recording the ordering `authority` and `reason` on the
    /// tombstone. Refused if on hold. The content is destroyed; a [`Tombstone`] retains
    /// only the fact/time/authority of deletion.
    pub fn erase_by(
        &mut self,
        id: u64,
        at: Micros,
        authority: Hash256,
        reason: TombstoneReason,
    ) -> bool {
        let on_hold = self.items.iter().find(|i| i.id == id).map(|i| i.holds > 0).unwrap_or(true);
        if on_hold {
            self.log(AuditEvent::EraseBlockedByHold, id, at);
            return false;
        }
        let class = match self.items.iter().find(|i| i.id == id) {
            Some(it) if it.state != State::Erased => it.class,
            _ => return false,
        };
        if let Some(it) = self.item_mut(id) {
            it.key_alive = false; // crypto-GC: destroy the key
            it.state = State::Erased;
        }
        self.tombstones.push(Tombstone { id, class, erased_at: at, authority, reason });
        self.log(AuditEvent::Erased, id, at);
        true
    }

    /// The deletion tombstone for `id`, if it was erased.
    pub fn tombstone(&self, id: u64) -> Option<&Tombstone> {
        self.tombstones.iter().find(|t| t.id == id)
    }

    /// All deletion tombstones (the provenance record of every erasure).
    pub fn tombstones(&self) -> &[Tombstone] {
        &self.tombstones
    }

    /// Sweep at time `now`: cryptographically erase every Live datum past its
    /// retention that is not on hold. Returns the ids erased.
    pub fn sweep(&mut self, now: Micros) -> Vec<u64> {
        let expired: Vec<u64> = self
            .items
            .iter()
            .filter(|i| {
                i.state == State::Live
                    && i.holds == 0
                    && now >= i.created.saturating_add(i.retention)
            })
            .map(|i| i.id)
            .collect();
        for id in &expired {
            self.erase_by(*id, now, Hash256::of(b"retention-sweep"), TombstoneReason::RetentionExpired);
        }
        expired
    }

    /// Verify the audit chain is internally consistent (tamper-evident).
    pub fn audit_intact(&self) -> bool {
        let mut prev = Hash256::of(b"audit-genesis");
        for e in &self.audit {
            let mut input = Vec::with_capacity(64);
            input.extend_from_slice(&prev.0);
            input.extend_from_slice(&e.item.to_le_bytes());
            input.extend_from_slice(&e.at.to_le_bytes());
            input.push(e.event as u8);
            if Hash256::of(&input) != e.chain {
                return false;
            }
            prev = e.chain;
        }
        true
    }

    pub fn audit(&self) -> &[AuditEntry] {
        &self.audit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erasure_leaves_a_tombstone_with_fact_time_authority_not_content() {
        let mut lc = DataLifecycle::new();
        lc.register(1, DataClass::Ephemeral, 0, 100);
        lc.register(2, DataClass::UserContent, 0, 10_000);
        // Retention sweep erases #1 and tombstones it (reason: expired).
        lc.sweep(150);
        let t1 = lc.tombstone(1).unwrap();
        assert_eq!(t1.reason, TombstoneReason::RetentionExpired);
        assert_eq!(t1.class, DataClass::Ephemeral);
        assert_eq!(t1.erased_at, 150);
        assert_eq!(t1.authority, Hash256::of(b"retention-sweep"));
        // The content is gone (key destroyed) — the tombstone is metadata only.
        assert!(!lc.readable(1));
        // Manual erase records authority + Manual reason.
        let who = Hash256::of(b"subject-access-request");
        assert!(lc.erase_by(2, 200, who, TombstoneReason::Manual));
        assert_eq!(lc.tombstone(2).unwrap().authority, who);
        assert_eq!(lc.tombstones().len(), 2);
        // Erasing an already-erased datum is a no-op (no duplicate tombstone).
        assert!(!lc.erase_by(2, 300, who, TombstoneReason::Manual));
        assert_eq!(lc.tombstones().len(), 2);
        assert!(lc.audit_intact());
    }

    #[test]
    fn retention_sweep_erases_expired_only() {
        let mut lc = DataLifecycle::new();
        lc.register(1, DataClass::Ephemeral, 0, 100); // expires at 100
        lc.register(2, DataClass::UserContent, 0, 10_000); // long retention
        assert_eq!(lc.class(2), Some(DataClass::UserContent));
        // At t=150, only item 1 is past retention.
        let erased = lc.sweep(150);
        assert_eq!(erased, alloc::vec![1]);
        assert_eq!(lc.state(1), Some(State::Erased));
        assert!(!lc.readable(1)); // key destroyed
        assert_eq!(lc.state(2), Some(State::Live));
        assert!(lc.readable(2));
    }

    #[test]
    fn legal_hold_blocks_deletion() {
        let mut lc = DataLifecycle::new();
        lc.register(1, DataClass::Sensitive, 0, 100);
        assert!(lc.place_hold(1, 50));
        // Past retention, but the hold suspends erasure.
        assert!(lc.sweep(500).is_empty());
        assert_eq!(lc.state(1), Some(State::OnHold));
        assert!(lc.readable(1));
        // Manual erase is also refused while held.
        assert!(!lc.erase(1, 500));
        // Release the hold → now it sweeps.
        assert!(lc.release_hold(1, 600));
        assert_eq!(lc.sweep(700), alloc::vec![1]);
        assert!(!lc.readable(1));
    }

    #[test]
    fn refcounted_holds_require_all_released() {
        let mut lc = DataLifecycle::new();
        lc.register(1, DataClass::Sensitive, 0, 0);
        lc.place_hold(1, 1); // matter A
        lc.place_hold(1, 2); // matter B
        lc.release_hold(1, 3); // A done
        // Still held by B.
        assert!(lc.sweep(100).is_empty());
        lc.release_hold(1, 4); // B done
        assert_eq!(lc.sweep(100), alloc::vec![1]);
    }

    #[test]
    fn manual_crypto_erase_destroys_key() {
        let mut lc = DataLifecycle::new();
        lc.register(7, DataClass::UserContent, 0, 1_000_000);
        assert!(lc.readable(7));
        assert!(lc.erase(7, 10));
        assert!(!lc.readable(7));
        assert_eq!(lc.state(7), Some(State::Erased));
    }

    #[test]
    fn audit_trail_is_tamper_evident() {
        let mut lc = DataLifecycle::new();
        lc.register(1, DataClass::SystemLog, 0, 100);
        lc.place_hold(1, 10);
        lc.release_hold(1, 20);
        lc.sweep(200);
        assert!(lc.audit_intact());
        // Events were logged in order, ending with an erase.
        let last = lc.audit().last().unwrap();
        assert_eq!(last.event, AuditEvent::Erased);
        assert!(lc.audit().iter().any(|e| e.event == AuditEvent::HoldPlaced));
    }

    #[test]
    fn erase_blocked_event_is_recorded() {
        let mut lc = DataLifecycle::new();
        lc.register(1, DataClass::Sensitive, 0, 0);
        lc.place_hold(1, 1);
        assert!(!lc.erase(1, 2));
        assert!(lc
            .audit()
            .iter()
            .any(|e| e.event == AuditEvent::EraseBlockedByHold));
        assert!(lc.audit_intact());
    }
}

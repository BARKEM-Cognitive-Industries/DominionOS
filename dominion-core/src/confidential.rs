//! System-domain confidentiality (see
//! `docs/security/system-domain-and-internal-confidentiality.md`).
//!
//! The kernel, the internals, and the Dominion runtime run under a distinct **System
//! identity**, and their state is labelled `SystemPrivate`. No user subject can
//! read it — not its contents, and not even its *existence* (enumeration hides what
//! you may not see, so the OS internals are invisible, not merely unreadable). This
//! is a reference monitor with a lattice of classifications plus a hard rule:
//! `SystemPrivate` is reachable only from the `System` domain.
//!
//! The model is "no read up, no cross-domain read" (a Bell–LaPadula confidentiality
//! lattice) specialised so the system domain is sealed off from users. Pure, safe
//! `no_std`, host-tested.

use alloc::vec::Vec;

/// Security domain a subject or object belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// The kernel / internals / Dominion runtime.
    System,
    /// A user identity (per-user isolation by id).
    User(u64),
}

/// Confidentiality classification, ordered least → most sensitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Classification {
    Public = 0,
    Internal = 1,
    Secret = 2,
    /// Only ever readable from [`Domain::System`].
    SystemPrivate = 3,
}

/// An acting subject: its domain and clearance level.
#[derive(Clone, Copy, Debug)]
pub struct Subject {
    pub domain: Domain,
    pub clearance: Classification,
}

impl Subject {
    pub fn system() -> Subject {
        Subject { domain: Domain::System, clearance: Classification::SystemPrivate }
    }
    pub fn user(id: u64, clearance: Classification) -> Subject {
        Subject { domain: Domain::User(id), clearance }
    }
}

/// A labelled object in the store.
#[derive(Clone, Debug)]
struct Labeled {
    id: u64,
    owner: Domain,
    class: Classification,
    bytes: Vec<u8>,
}

/// The reference monitor: it owns labelled objects and gates every read.
#[derive(Default)]
pub struct Confidentiality {
    objects: Vec<Labeled>,
}

/// The decision of a read attempt — `Denied` is uniform, leaking nothing about
/// whether the object exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadResult {
    Granted(Vec<u8>),
    Denied,
}

impl Confidentiality {
    pub fn new() -> Confidentiality {
        Confidentiality { objects: Vec::new() }
    }

    /// Store a labelled object (the labeller must itself be authorised in a real
    /// system; here registration is a trusted setup step).
    pub fn put(&mut self, id: u64, owner: Domain, class: Classification, bytes: &[u8]) {
        self.objects.push(Labeled { id, owner, class, bytes: bytes.to_vec() });
    }

    /// The core access rule.
    fn may_read(subject: &Subject, obj: &Labeled) -> bool {
        // Hard seal: SystemPrivate is reachable only from the System domain.
        if obj.class == Classification::SystemPrivate {
            return subject.domain == Domain::System;
        }
        // System may read anything below SystemPrivate.
        if subject.domain == Domain::System {
            return true;
        }
        // No read up: clearance must dominate the object's classification.
        if subject.clearance < obj.class {
            return false;
        }
        // No cross-user read: a user may read its own objects, or Public ones.
        match obj.owner {
            Domain::System => false, // system-owned but sub-SystemPrivate stays internal to system
            Domain::User(uid) => {
                subject.domain == Domain::User(uid) || obj.class == Classification::Public
            }
        }
    }

    /// Attempt to read object `id` as `subject`. Returns `Denied` (uniformly) when
    /// not permitted *or* when the object does not exist — existence is not leaked.
    pub fn read(&self, subject: &Subject, id: u64) -> ReadResult {
        match self.objects.iter().find(|o| o.id == id) {
            Some(obj) if Self::may_read(subject, obj) => ReadResult::Granted(obj.bytes.clone()),
            _ => ReadResult::Denied,
        }
    }

    /// List the ids `subject` is allowed to see. Internals never appear here for a
    /// user — they are invisible, not just unreadable.
    pub fn enumerate(&self, subject: &Subject) -> Vec<u64> {
        self.objects
            .iter()
            .filter(|o| Self::may_read(subject, o))
            .map(|o| o.id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Confidentiality {
        let mut c = Confidentiality::new();
        c.put(1, Domain::System, Classification::SystemPrivate, b"kernel page tables");
        c.put(2, Domain::System, Classification::Internal, b"system metric");
        c.put(3, Domain::User(100), Classification::Secret, b"alice diary");
        c.put(4, Domain::User(100), Classification::Public, b"alice public note");
        c
    }

    #[test]
    fn user_cannot_read_system_internals() {
        let c = store();
        let alice = Subject::user(100, Classification::SystemPrivate); // even max clearance
        // SystemPrivate is sealed to the System domain regardless of clearance.
        assert_eq!(c.read(&alice, 1), ReadResult::Denied);
        // And system-owned Internal objects are not user-visible either.
        assert_eq!(c.read(&alice, 2), ReadResult::Denied);
    }

    #[test]
    fn system_can_read_its_own_internals() {
        let c = store();
        let sys = Subject::system();
        assert_eq!(c.read(&sys, 1), ReadResult::Granted(b"kernel page tables".to_vec()));
        assert_eq!(c.read(&sys, 2), ReadResult::Granted(b"system metric".to_vec()));
    }

    #[test]
    fn user_reads_own_secret_but_not_another_users() {
        let c = store();
        let alice = Subject::user(100, Classification::Secret);
        let bob = Subject::user(200, Classification::SystemPrivate);
        // Alice reads her own Secret.
        assert_eq!(c.read(&alice, 3), ReadResult::Granted(b"alice diary".to_vec()));
        // Bob, despite high clearance, cannot read Alice's diary (cross-user).
        assert_eq!(c.read(&bob, 3), ReadResult::Denied);
        // But a Public object is readable by anyone cleared.
        assert_eq!(c.read(&bob, 4), ReadResult::Granted(b"alice public note".to_vec()));
    }

    #[test]
    fn no_read_up_below_clearance() {
        let c = store();
        // A low-clearance user cannot read a Secret even if it were their own class.
        let low = Subject::user(100, Classification::Internal);
        assert_eq!(c.read(&low, 3), ReadResult::Denied); // Secret > Internal clearance
    }

    #[test]
    fn enumeration_hides_invisible_objects() {
        let c = store();
        let alice = Subject::user(100, Classification::Secret);
        let visible = c.enumerate(&alice);
        // Alice sees her own diary and the public note — never the system internals.
        assert!(visible.contains(&3));
        assert!(visible.contains(&4));
        assert!(!visible.contains(&1));
        assert!(!visible.contains(&2));
        // The System domain sees everything.
        assert_eq!(c.enumerate(&Subject::system()).len(), 4);
    }

    #[test]
    fn nonexistent_and_denied_are_indistinguishable() {
        let c = store();
        let alice = Subject::user(100, Classification::Secret);
        // Reading a forbidden object and a nonexistent one both return Denied.
        assert_eq!(c.read(&alice, 1), ReadResult::Denied); // exists, forbidden
        assert_eq!(c.read(&alice, 9999), ReadResult::Denied); // does not exist
    }
}

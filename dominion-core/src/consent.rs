//! Capability-consent UX & blast-radius limiting — **the social-engineering defense layer**
//! (`docs/security/threat-model-and-security-posture.md`: social engineering).
//!
//! Phishing works by tricking a user into granting *more authority than they realise*. The
//! capability model already removes ambient authority; this module adds the **consent
//! discipline** on top so a grant is always scoped, visible, and minimized:
//!
//! * [`blast_radius`] quantifies how much a capability exposes (rights × extent), so the
//!   consent UI can *show* what is really being asked for — no invisible authority.
//! * [`minimize`] derives the **least-authority** capability that still satisfies the
//!   stated need, so the system grants the minimum rather than what was asked.
//! * [`ConsentRequest`] / [`ConsentLedger`] record an explicit, auditable user decision
//!   per grant — there is no silent or ambient consent.
//!
//! The result: even a perfectly convincing phish can only obtain a **scoped, minimized,
//! logged** capability whose blast radius the user saw — not the keys to the kingdom.
//! Pure, safe `no_std`. Host-tested.

use crate::capability::{Capability, Rights};
use crate::hash::Hash256;
use alloc::string::String;
use alloc::vec::Vec;

/// A coarse "how much could this hurt" score for a capability: the count of granted
/// rights bits times the size of the region it spans. Bigger ⇒ more dangerous ⇒ the
/// consent UI must surface it more prominently. A zero-length or no-rights capability has
/// zero blast radius (it can do nothing).
pub fn blast_radius(cap: &Capability) -> u64 {
    if !cap.is_valid() {
        return 0;
    }
    let rights_bits = (cap.rights().bits().count_ones()) as u64;
    rights_bits.saturating_mul(cap.len())
}

/// Derive the **least-authority** capability that still meets `needed`: intersect the
/// requested rights with what's actually needed (never grant more than required), keeping
/// the requested bounds. Returns `None` if the source can't even cover the need.
pub fn minimize(requested: &Capability, needed: Rights) -> Option<Capability> {
    if !requested.rights().contains(needed) {
        return None; // the source itself doesn't hold what's needed
    }
    requested.restrict(needed).ok()
}

/// A request for the user to consent to a grant.
#[derive(Clone, Debug)]
pub struct ConsentRequest {
    /// Who is asking (an app / site / cell identity fingerprint).
    pub requester: Hash256,
    /// A human-readable purpose string (shown to the user).
    pub purpose: String,
    /// The rights actually required for the purpose.
    pub needed: Rights,
}

impl ConsentRequest {
    pub fn new(requester: Hash256, purpose: &str, needed: Rights) -> ConsentRequest {
        ConsentRequest { requester, purpose: String::from(purpose), needed }
    }
}

/// A recorded consent decision (auditable; there is no ambient/silent consent).
#[derive(Clone, Debug)]
pub struct ConsentRecord {
    pub requester: Hash256,
    pub purpose: String,
    pub granted_rights: u32,
    pub blast_radius: u64,
    pub approved: bool,
}

/// The consent ledger: every grant decision is logged, so authority changes are auditable
/// and a user can review (and revoke) what they've consented to.
#[derive(Default)]
pub struct ConsentLedger {
    records: Vec<ConsentRecord>,
}

impl ConsentLedger {
    pub fn new() -> ConsentLedger {
        ConsentLedger { records: Vec::new() }
    }

    /// Process a consent request against an available `source` capability and the user's
    /// `approve` decision. On approval, returns the **minimized** capability (least
    /// authority for the stated purpose); on denial, returns `None`. Either way the
    /// decision is recorded.
    pub fn decide(
        &mut self,
        req: &ConsentRequest,
        source: &Capability,
        approve: bool,
    ) -> Option<Capability> {
        let minimized = minimize(source, req.needed);
        let granted = if approve { minimized } else { None };
        let (rights, radius) = match &granted {
            Some(c) => (c.rights().bits(), blast_radius(c)),
            None => (0, 0),
        };
        self.records.push(ConsentRecord {
            requester: req.requester,
            purpose: req.purpose.clone(),
            granted_rights: rights,
            blast_radius: radius,
            approved: approve && granted.is_some(),
        });
        granted
    }

    /// All recorded consents (the audit surface the user reviews).
    pub fn records(&self) -> &[ConsentRecord] {
        &self.records
    }

    /// The total blast radius the user has consented to across all approved grants — the
    /// running "how much authority have I given out" figure.
    pub fn total_exposure(&self) -> u64 {
        self.records.iter().filter(|r| r.approved).map(|r| r.blast_radius).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blast_radius_reflects_rights_and_extent() {
        let read_small = Capability::mint(0, 100, Rights::READ); // 1 right × 100
        let rw_big = Capability::mint(0, 1000, Rights::READ.union(Rights::WRITE)); // 2 × 1000
        assert_eq!(blast_radius(&read_small), 100);
        assert_eq!(blast_radius(&rw_big), 2000);
        assert!(blast_radius(&rw_big) > blast_radius(&read_small));
        // An empty capability can do nothing.
        assert_eq!(blast_radius(&Capability::mint(0, 0, Rights::ALL)), 0);
    }

    #[test]
    fn minimize_grants_least_authority() {
        // The app asks with a broad capability, but only needs READ.
        let broad = Capability::mint(0x1000, 0x100, Rights::ALL);
        let least = minimize(&broad, Rights::READ).unwrap();
        assert!(least.rights().contains(Rights::READ));
        assert!(!least.rights().contains(Rights::WRITE));
        // Asking for a right the source doesn't hold fails (can't escalate).
        let read_only = Capability::mint(0, 16, Rights::READ);
        assert!(minimize(&read_only, Rights::WRITE).is_none());
    }

    #[test]
    fn consent_is_explicit_minimized_and_logged() {
        let mut ledger = ConsentLedger::new();
        let source = Capability::mint(0, 4096, Rights::ALL);
        let req = ConsentRequest::new(Hash256::of(b"some-website"), "read your notes", Rights::READ);
        // Approve → a minimized READ-only capability, logged.
        let granted = ledger.decide(&req, &source, true).unwrap();
        assert!(granted.rights().contains(Rights::READ));
        assert!(!granted.rights().contains(Rights::WRITE)); // phish can't get WRITE/ALL
        assert_eq!(ledger.records().len(), 1);
        assert!(ledger.records()[0].approved);
        // Denial → nothing granted, still recorded.
        let denied = ledger.decide(&req, &source, false);
        assert!(denied.is_none());
        assert!(!ledger.records()[1].approved);
        // Exposure counts only the approved grant's (bounded) blast radius.
        assert_eq!(ledger.total_exposure(), blast_radius(&granted));
    }
}

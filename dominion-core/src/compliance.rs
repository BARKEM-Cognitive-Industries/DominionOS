//! Data-lifecycle compliance — **purpose/consent tags, geo-pinned data residency, and
//! compliance reporting queries** (`docs/security/data-lifecycle-and-compliance.md`).
//!
//! [`crate::lifecycle`] does retention + crypto-erase + the audit chain; this module adds
//! the consent/residency/reporting layer on top:
//!
//! * [`ConsentScope`] — **purpose/consent tags**: data may only be used for purposes the
//!   user granted, and **withdrawing consent** for a purpose immediately revokes it (the
//!   caller then crypto-GCs the affected objects). Grants can expire.
//! * [`ResidencyPolicy`] — **geo-pinned domains**: each domain is pinned to a region and a
//!   cross-domain transfer is allowed only when residency permits, so data doesn't leave
//!   its jurisdiction (Airlock-enforced).
//! * [`ComplianceReport`] — **reporting as queries** over a provenance/audit event stream
//!   (counts by kind, within a time window) — compliance answers are derived, not curated.
//!
//! Pure, safe `no_std`. Host-tested.

use crate::firewall::Domain;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;

// ───────────────────────── purpose / consent tags ─────────────────────────

/// A granted purpose with an optional expiry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PurposeGrant {
    pub granted_at: u64,
    pub expires_at: Option<u64>,
}

/// The set of purposes an object's data may be used for. A capability scoped to a purpose
/// is only honoured while the matching consent is granted, unexpired and not withdrawn.
#[derive(Default)]
pub struct ConsentScope {
    grants: BTreeMap<String, PurposeGrant>,
    withdrawn: BTreeSet<String>,
}

impl ConsentScope {
    pub fn new() -> ConsentScope {
        ConsentScope { grants: BTreeMap::new(), withdrawn: BTreeSet::new() }
    }

    /// Grant consent to use the data for `purpose` (optionally expiring).
    pub fn grant(&mut self, purpose: &str, now: u64, expires_at: Option<u64>) {
        self.withdrawn.remove(purpose);
        self.grants.insert(String::from(purpose), PurposeGrant { granted_at: now, expires_at });
    }

    /// **Withdraw** consent for `purpose`: it is revoked immediately. Returns whether it
    /// was previously granted (the caller then crypto-GCs the affected objects).
    pub fn withdraw(&mut self, purpose: &str) -> bool {
        let had = self.grants.remove(purpose).is_some();
        self.withdrawn.insert(String::from(purpose));
        had
    }

    /// May the data be used for `purpose` at time `now`? Only if granted, unexpired and
    /// not withdrawn.
    pub fn permits(&self, purpose: &str, now: u64) -> bool {
        if self.withdrawn.contains(purpose) {
            return false;
        }
        match self.grants.get(purpose) {
            Some(g) => g.expires_at.map(|e| now < e).unwrap_or(true),
            None => false,
        }
    }
}

// ───────────────────────── geo-pinned data residency ─────────────────────────

/// A data-residency policy pinning each domain to a region. A cross-domain transfer is
/// permitted only when residency rules allow it (same region, or an explicitly allowed
/// flow), so geo-restricted data cannot leave its jurisdiction.
#[derive(Default)]
pub struct ResidencyPolicy {
    region: BTreeMap<Domain, String>,
    /// Explicitly allowed cross-region flows `(from_region → to_region)`.
    allowed: BTreeSet<(String, String)>,
}

impl ResidencyPolicy {
    pub fn new() -> ResidencyPolicy {
        ResidencyPolicy { region: BTreeMap::new(), allowed: BTreeSet::new() }
    }

    /// Pin `domain` to `region`.
    pub fn pin(&mut self, domain: Domain, region: &str) {
        self.region.insert(domain, String::from(region));
    }

    /// Explicitly allow data to flow from one region to another (e.g. an adequacy
    /// decision). Same-region flows are always allowed.
    pub fn allow_flow(&mut self, from_region: &str, to_region: &str) {
        self.allowed.insert((String::from(from_region), String::from(to_region)));
    }

    /// The region a domain is pinned to (None ⇒ unpinned, treated as unrestricted).
    pub fn region_of(&self, domain: Domain) -> Option<&str> {
        self.region.get(&domain).map(|s| s.as_str())
    }

    /// May data move from `from` to `to` under residency rules?
    pub fn may_transfer(&self, from: Domain, to: Domain) -> bool {
        match (self.region.get(&from), self.region.get(&to)) {
            (Some(fr), Some(tr)) => fr == tr || self.allowed.contains(&(fr.clone(), tr.clone())),
            // An unpinned endpoint imposes no residency restriction.
            _ => true,
        }
    }
}

// ───────────────────────── compliance reporting ─────────────────────────

/// A compliance-relevant event drawn from the provenance/audit stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComplianceEvent {
    pub kind: EventKind,
    pub at: u64,
}

/// The categories a compliance report counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventKind {
    Created,
    Accessed,
    Erased,
    HoldPlaced,
    HoldReleased,
    ConsentWithdrawn,
}

/// Compliance reporting **as a query** over an event stream — answers are derived from
/// the immutable provenance log, not hand-maintained.
pub struct ComplianceReport<'a> {
    events: &'a [ComplianceEvent],
}

impl<'a> ComplianceReport<'a> {
    pub fn over(events: &'a [ComplianceEvent]) -> ComplianceReport<'a> {
        ComplianceReport { events }
    }

    /// Count events of `kind` within `[from, to)`.
    pub fn count(&self, kind: EventKind, from: u64, to: u64) -> usize {
        self.events.iter().filter(|e| e.kind == kind && e.at >= from && e.at < to).count()
    }

    /// Total events of `kind` over all time.
    pub fn total(&self, kind: EventKind) -> usize {
        self.events.iter().filter(|e| e.kind == kind).count()
    }

    /// A summary: (erasures, holds outstanding, consent withdrawals) over all time.
    pub fn summary(&self) -> (usize, isize, usize) {
        let erased = self.total(EventKind::Erased);
        let holds = self.total(EventKind::HoldPlaced) as isize - self.total(EventKind::HoldReleased) as isize;
        let withdrawals = self.total(EventKind::ConsentWithdrawn);
        (erased, holds, withdrawals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consent_tags_gate_use_and_withdrawal_revokes() {
        let mut scope = ConsentScope::new();
        scope.grant("analytics", 0, Some(1000));
        assert!(scope.permits("analytics", 500));
        // Expired grant no longer permits.
        assert!(!scope.permits("analytics", 1500));
        // A purpose never granted is denied.
        assert!(!scope.permits("marketing", 0));
        // Withdrawal revokes immediately.
        scope.grant("marketing", 0, None);
        assert!(scope.permits("marketing", 10));
        assert!(scope.withdraw("marketing"));
        assert!(!scope.permits("marketing", 10));
    }

    #[test]
    fn geo_pinned_domains_enforce_residency() {
        let mut pol = ResidencyPolicy::new();
        pol.pin(Domain::Personal, "EU");
        pol.pin(Domain::Financial, "EU");
        pol.pin(Domain::Development, "US");
        // Same region → allowed.
        assert!(pol.may_transfer(Domain::Personal, Domain::Financial));
        // Cross-region without an allowed flow → denied (data stays in the EU).
        assert!(!pol.may_transfer(Domain::Personal, Domain::Development));
        // An explicit adequacy flow opens it.
        pol.allow_flow("EU", "US");
        assert!(pol.may_transfer(Domain::Personal, Domain::Development));
        assert_eq!(pol.region_of(Domain::Personal), Some("EU"));
    }

    #[test]
    fn compliance_report_answers_queries_over_the_event_stream() {
        let events = [
            ComplianceEvent { kind: EventKind::Created, at: 10 },
            ComplianceEvent { kind: EventKind::HoldPlaced, at: 20 },
            ComplianceEvent { kind: EventKind::Erased, at: 30 },
            ComplianceEvent { kind: EventKind::Erased, at: 110 },
            ComplianceEvent { kind: EventKind::ConsentWithdrawn, at: 120 },
        ];
        let report = ComplianceReport::over(&events);
        // Erasures in the first window vs all time.
        assert_eq!(report.count(EventKind::Erased, 0, 100), 1);
        assert_eq!(report.total(EventKind::Erased), 2);
        let (erased, holds, withdrawals) = report.summary();
        assert_eq!(erased, 2);
        assert_eq!(holds, 1); // one placed, none released
        assert_eq!(withdrawals, 1);
    }
}

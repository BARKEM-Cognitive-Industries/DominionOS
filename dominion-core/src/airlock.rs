//! Capability Airlock — **Stage 11.15** (inter-domain authority transfer).
//!
//! The Airlock is the *sole* path through which a capability may cross a trust
//! boundary. Capabilities are never transferred directly — they are **projected
//! and sanitized** to the minimum authority the destination needs. The Airlock
//! supports **one-way (data-diode) channels**, **temporal capabilities** that
//! expire, and **multi-party authorization** (separation of duty). Assumption:
//! every domain may eventually be compromised, so `Compromise ≠ Containment
//! Failure` — a breached domain still cannot push authority upstream.
//!
//! Built on the existing [`Capability`](crate::capability) algebra (reduction is
//! monotonic restriction), so the same guarantees the hardware would give apply.
//! Pure, safe, host-tested.

use crate::capability::{Capability, Rights};
use crate::firewall::Domain;
use crate::hash::Hash256;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// Why the Airlock refused a transfer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AirlockError {
    /// No policy authorises this `(from → to)` crossing.
    NoPolicy,
    /// Not enough approvals for a multi-party transfer.
    InsufficientApprovals,
    /// The source capability is invalid (tag cleared).
    InvalidCapability,
    /// A domain involved in the crossing is under emergency containment.
    Contained,
}

/// An immutable, hash-chained record of one cross-domain transfer — the Airlock's
/// transfer ledger. Records the *fact* of a crossing (who, where, what authority, when)
/// so every authority transfer is tamper-evidently auditable. Never holds the data that
/// crossed, only the metadata of the crossing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferRecord {
    pub seq: u64,
    pub from: Domain,
    pub to: Domain,
    /// The rights actually issued (after sanitization).
    pub granted_rights: u32,
    pub expires_at: Option<u64>,
    /// Provenance fingerprint of the issued capability.
    pub provenance: Hash256,
    /// Hash chaining this record to all prior ones (tamper-evidence).
    pub chain: Hash256,
}

/// A policy governing one directional crossing.
#[derive(Clone, Copy)]
pub struct TransferPolicy {
    pub from: Domain,
    pub to: Domain,
    /// Ceiling on the rights the destination may receive (sanitization mask).
    pub max_rights: Rights,
    /// Time-to-live for issued capabilities (in abstract ticks), if any.
    pub ttl: Option<u64>,
    /// Number of approvals required (separation of duty).
    pub approvals_required: u32,
}

/// A capability issued by the Airlock — possibly time-bounded.
#[derive(Clone, Copy, Debug)]
pub struct IssuedCapability {
    pub capability: Capability,
    pub expires_at: Option<u64>,
}

impl IssuedCapability {
    pub fn is_expired(&self, now: u64) -> bool {
        matches!(self.expires_at, Some(t) if now >= t)
    }
}

/// The formally-verified gateway for cross-domain authority transfer.
pub struct Airlock {
    policies: Vec<TransferPolicy>,
    ledger: Vec<TransferRecord>,
    contained: BTreeSet<Domain>,
}

impl Airlock {
    pub fn new() -> Airlock {
        Airlock { policies: Vec::new(), ledger: Vec::new(), contained: BTreeSet::new() }
    }

    pub fn add_policy(&mut self, policy: TransferPolicy) {
        self.policies.push(policy);
    }

    fn policy(&self, from: Domain, to: Domain) -> Option<&TransferPolicy> {
        self.policies.iter().find(|p| p.from == from && p.to == to)
    }

    /// True iff `domain` is under emergency containment.
    pub fn is_contained(&self, domain: Domain) -> bool {
        self.contained.contains(&domain)
    }

    /// **Emergency containment-before-forensics**: instantly seal a domain off — every
    /// crossing into or out of it is refused (`AirlockError::Contained`) *before* any
    /// analysis runs, so a suspected-compromised domain cannot exfiltrate authority while
    /// it is being investigated. The sealing is recorded on the immutable ledger; the
    /// forensic snapshot ([`forensic_snapshot`](Self::forensic_snapshot)) captures the
    /// exact ledger state at containment for later analysis.
    pub fn contain(&mut self, domain: Domain) {
        self.contained.insert(domain);
        self.append_record(domain, domain, Rights::NONE, None, Hash256::of(b"containment"));
    }

    /// Lift containment once forensics clears the domain.
    pub fn release(&mut self, domain: Domain) {
        self.contained.remove(&domain);
    }

    /// A tamper-evident snapshot for forensics: the current ledger digest. Anyone can
    /// later check the ledger still hashes to this (no record was altered or removed).
    pub fn forensic_snapshot(&self) -> Hash256 {
        self.ledger.last().map(|r| r.chain).unwrap_or(Hash256::of(b"airlock-genesis"))
    }

    fn append_record(
        &mut self,
        from: Domain,
        to: Domain,
        granted: Rights,
        expires_at: Option<u64>,
        provenance: Hash256,
    ) {
        let seq = self.ledger.len() as u64;
        let prev = self.ledger.last().map(|r| r.chain).unwrap_or(Hash256::of(b"airlock-genesis"));
        let mut input = Vec::with_capacity(96);
        input.extend_from_slice(&prev.0);
        input.extend_from_slice(&seq.to_le_bytes());
        input.push(from as u8);
        input.push(to as u8);
        input.extend_from_slice(&granted.bits().to_le_bytes());
        input.extend_from_slice(&provenance.0);
        // Include expires_at in the chain so an expiry cannot be silently
        // altered (e.g. an expired temporal grant turned never-expiring) while
        // the ledger still verifies. Tagged: presence byte + u64 when Some.
        match expires_at {
            Some(t) => {
                input.push(1);
                input.extend_from_slice(&t.to_le_bytes());
            }
            None => input.push(0),
        }
        let chain = Hash256::of(&input);
        self.ledger.push(TransferRecord {
            seq,
            from,
            to,
            granted_rights: granted.bits(),
            expires_at,
            provenance,
            chain,
        });
    }

    /// Like [`transfer`](Self::transfer), but honours emergency containment and appends
    /// an immutable, hash-chained [`TransferRecord`] to the ledger on success. This is
    /// the audited path the OS uses for every real cross-domain crossing.
    pub fn transfer_logged(
        &mut self,
        capability: Capability,
        from: Domain,
        to: Domain,
        approvals: u32,
        now: u64,
    ) -> Result<IssuedCapability, AirlockError> {
        if self.contained.contains(&from) || self.contained.contains(&to) {
            return Err(AirlockError::Contained);
        }
        let issued = self.transfer(capability, from, to, approvals, now)?;
        self.append_record(
            from,
            to,
            issued.capability.rights(),
            issued.expires_at,
            issued.capability.provenance(),
        );
        Ok(issued)
    }

    /// The immutable transfer ledger (every audited crossing, in order).
    pub fn ledger(&self) -> &[TransferRecord] {
        &self.ledger
    }

    /// Verify the ledger's hash chain is internally consistent (tamper-evident).
    pub fn ledger_intact(&self) -> bool {
        let mut prev = Hash256::of(b"airlock-genesis");
        for r in &self.ledger {
            let mut input = Vec::with_capacity(96);
            input.extend_from_slice(&prev.0);
            input.extend_from_slice(&r.seq.to_le_bytes());
            input.push(r.from as u8);
            input.push(r.to as u8);
            input.extend_from_slice(&r.granted_rights.to_le_bytes());
            input.extend_from_slice(&r.provenance.0);
            match r.expires_at {
                Some(t) => {
                    input.push(1);
                    input.extend_from_slice(&t.to_le_bytes());
                }
                None => input.push(0),
            }
            if Hash256::of(&input) != r.chain {
                return false;
            }
            prev = r.chain;
        }
        true
    }

    /// Transfer `capability` from `from` to `to`. The result is sanitized to the
    /// policy's `max_rights` (never more than the source held — monotonic), and
    /// time-bounded if the policy sets a TTL. Direct transfers with no policy, or
    /// without enough approvals, are refused. The reverse direction of a one-way
    /// channel simply has no policy and is therefore denied.
    pub fn transfer(
        &self,
        capability: Capability,
        from: Domain,
        to: Domain,
        approvals: u32,
        now: u64,
    ) -> Result<IssuedCapability, AirlockError> {
        if !capability.is_valid() {
            return Err(AirlockError::InvalidCapability);
        }
        let policy = self.policy(from, to).ok_or(AirlockError::NoPolicy)?;
        if approvals < policy.approvals_required {
            return Err(AirlockError::InsufficientApprovals);
        }
        // Sanitize: intersect the source's rights with the policy ceiling, then
        // restrict (monotonic — can only ever drop rights).
        let granted = capability.rights().intersect(policy.max_rights);
        let reduced = capability.restrict(granted).map_err(|_| AirlockError::InvalidCapability)?;
        Ok(IssuedCapability {
            capability: reduced,
            expires_at: policy.ttl.map(|t| now.saturating_add(t)),
        })
    }
}

impl Default for Airlock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapError;

    fn financial_cap() -> Capability {
        // Read+Write+Execute over an account region.
        Capability::mint(0x1000, 0x1000, Rights::READ.union(Rights::WRITE).union(Rights::EXECUTE))
    }

    fn airlock() -> Airlock {
        let mut a = Airlock::new();
        // Financial -> AiAgent: read-only, expires after 10 ticks, dual approval.
        a.add_policy(TransferPolicy {
            from: Domain::Financial,
            to: Domain::AiAgent,
            max_rights: Rights::READ,
            ttl: Some(10),
            approvals_required: 2,
        });
        a
    }

    #[test]
    fn transfer_ledger_is_immutable_and_tamper_evident() {
        let mut a = airlock();
        let i1 = a.transfer_logged(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 0).unwrap();
        let i2 = a.transfer_logged(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 5).unwrap();
        assert!(i1.capability.rights().contains(Rights::READ));
        assert_eq!(a.ledger().len(), 2);
        assert!(a.ledger_intact());
        // Records carry the issued rights + expiry; the chain links them.
        assert_eq!(a.ledger()[0].granted_rights, Rights::READ.bits());
        assert_eq!(a.ledger()[1].expires_at, Some(15));
        assert_ne!(a.ledger()[0].chain, a.ledger()[1].chain);
        let _ = i2;
    }

    #[test]
    fn emergency_containment_seals_a_domain_before_forensics() {
        let mut a = airlock();
        // A normal crossing succeeds.
        assert!(a.transfer_logged(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 0).is_ok());
        let snap = a.forensic_snapshot();
        // Contain the AI domain (suspected compromise): further crossings refused.
        a.contain(Domain::AiAgent);
        assert!(a.is_contained(Domain::AiAgent));
        assert_eq!(
            a.transfer_logged(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 1).unwrap_err(),
            AirlockError::Contained
        );
        // The forensic snapshot advanced (containment was itself recorded) and the
        // ledger remains intact for analysis.
        assert_ne!(a.forensic_snapshot(), snap);
        assert!(a.ledger_intact());
        // Once cleared, crossings resume.
        a.release(Domain::AiAgent);
        assert!(a.transfer_logged(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 2).is_ok());
    }

    #[test]
    fn transfer_sanitizes_to_minimum_authority() {
        let a = airlock();
        let issued = a.transfer(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 0).unwrap();
        // The AI domain receives read-only, never write/execute.
        assert!(issued.capability.rights().contains(Rights::READ));
        assert!(!issued.capability.rights().contains(Rights::WRITE));
        assert!(!issued.capability.rights().contains(Rights::EXECUTE));
    }

    #[test]
    fn reverse_direction_has_no_policy_and_is_denied() {
        let a = airlock();
        // AiAgent -> Financial is a one-way violation: no policy exists.
        assert_eq!(
            a.transfer(financial_cap(), Domain::AiAgent, Domain::Financial, 2, 0).unwrap_err(),
            AirlockError::NoPolicy
        );
    }

    #[test]
    fn multi_party_authorization_enforced() {
        let a = airlock();
        assert_eq!(
            a.transfer(financial_cap(), Domain::Financial, Domain::AiAgent, 1, 0).unwrap_err(),
            AirlockError::InsufficientApprovals
        );
        assert!(a.transfer(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 0).is_ok());
    }

    #[test]
    fn issued_capabilities_expire() {
        let a = airlock();
        let issued = a.transfer(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 100).unwrap();
        assert_eq!(issued.expires_at, Some(110));
        assert!(!issued.is_expired(105));
        assert!(issued.is_expired(110));
        assert!(issued.is_expired(200));
    }

    #[test]
    fn tampered_capability_is_refused() {
        let a = airlock();
        let bad = financial_cap().tamper();
        assert_eq!(
            a.transfer(bad, Domain::Financial, Domain::AiAgent, 2, 0).unwrap_err(),
            AirlockError::InvalidCapability
        );
    }

    #[test]
    fn reduced_capability_cannot_re_escalate() {
        let a = airlock();
        let issued = a.transfer(financial_cap(), Domain::Financial, Domain::AiAgent, 2, 0).unwrap();
        // Downstream cannot widen the read-only grant back to write.
        assert_eq!(
            issued.capability.restrict(Rights::READ.union(Rights::WRITE)).unwrap_err(),
            CapError::MonotonicityViolation
        );
    }
}

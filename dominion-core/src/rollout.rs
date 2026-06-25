//! Update rollout safety — **a health watchdog with auto-revert, audited capability-
//! scoped installs, and a DST canary** (`docs/implementation/update-and-upgrade-lifecycle.md`).
//!
//! [`crate::update`] does A/B slots, signed releases, anti-rollback and staged cohorts;
//! this module adds the activation-time safety net:
//!
//! * [`HealthWatchdog`] — after activating a new bank, watch health attestations; if too
//!   many fail within the window, **auto-revert to the last known-good bank**.
//! * [`AuditedInstall`] — install/update is a **scoped, audited capability**: it must be
//!   authorised for the target, and every install is appended to a tamper-evident log
//!   (cross-domain installs route through the Airlock).
//! * [`Canary`] — a **DST canary**: replay a recorded workload through the candidate and
//!   the known-good and only activate if the candidate matches (no behavioural regression).
//!
//! Pure, safe `no_std`. Host-tested.

use crate::hash::Hash256;
use alloc::vec::Vec;

// ───────────────────────── health watchdog + auto-revert ─────────────────────────

/// Watches post-activation health and decides whether to revert to the known-good bank.
#[derive(Clone, Copy, Debug)]
pub struct HealthWatchdog {
    /// The slot that was active before the update (the revert target).
    known_good: u32,
    /// The newly-activated slot.
    candidate: u32,
    /// How many failed health checks trigger an auto-revert.
    fail_threshold: u32,
    failures: u32,
    reverted: bool,
}

impl HealthWatchdog {
    /// Arm the watchdog after activating `candidate`, with `known_good` as the fallback.
    pub fn arm(known_good: u32, candidate: u32, fail_threshold: u32) -> HealthWatchdog {
        HealthWatchdog { known_good, candidate, fail_threshold: fail_threshold.max(1), failures: 0, reverted: false }
    }

    /// Report a health attestation result. A passing check resets the failure streak.
    pub fn report(&mut self, healthy: bool) {
        if self.reverted {
            return;
        }
        if healthy {
            self.failures = 0;
        } else {
            self.failures += 1;
            if self.failures >= self.fail_threshold {
                self.reverted = true;
            }
        }
    }

    /// The slot that should be running now — the candidate, or the known-good after revert.
    pub fn active_slot(&self) -> u32 {
        if self.reverted {
            self.known_good
        } else {
            self.candidate
        }
    }

    /// Whether the watchdog has tripped and reverted.
    pub fn reverted(&self) -> bool {
        self.reverted
    }
}

// ───────────────────────── audited, capability-scoped install ─────────────────────────

/// Why an install was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallDenied {
    /// The install capability doesn't authorise the target domain.
    Unauthorized,
}

/// A recorded install/update action (the audit trail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstallRecord {
    pub target_domain: u64,
    pub artifact: Hash256,
    pub chain: Hash256,
}

/// Install/update as a **scoped, audited capability**: an install is permitted only for
/// domains the capability authorises, and each one is appended to a hash-chained log.
#[derive(Default)]
pub struct AuditedInstall {
    /// Domains this install capability may target.
    authorized: Vec<u64>,
    log: Vec<InstallRecord>,
}

impl AuditedInstall {
    /// A capability authorising installs to `domains`.
    pub fn new(domains: &[u64]) -> AuditedInstall {
        AuditedInstall { authorized: domains.to_vec(), log: Vec::new() }
    }

    /// Install `artifact` to `target_domain`, if authorised; records an audit entry.
    pub fn install(&mut self, target_domain: u64, artifact: Hash256) -> Result<(), InstallDenied> {
        if !self.authorized.contains(&target_domain) {
            return Err(InstallDenied::Unauthorized);
        }
        let prev = self.log.last().map(|r| r.chain).unwrap_or(Hash256::of(b"install-genesis"));
        let mut input = Vec::with_capacity(72);
        input.extend_from_slice(&prev.0);
        input.extend_from_slice(&target_domain.to_le_bytes());
        input.extend_from_slice(&artifact.0);
        let chain = Hash256::of(&input);
        self.log.push(InstallRecord { target_domain, artifact, chain });
        Ok(())
    }

    /// The audit log of installs.
    pub fn log(&self) -> &[InstallRecord] {
        &self.log
    }

    /// Verify the audit chain is intact (tamper-evident).
    pub fn audit_intact(&self) -> bool {
        let mut prev = Hash256::of(b"install-genesis");
        for r in &self.log {
            let mut input = Vec::with_capacity(72);
            input.extend_from_slice(&prev.0);
            input.extend_from_slice(&r.target_domain.to_le_bytes());
            input.extend_from_slice(&r.artifact.0);
            if Hash256::of(&input) != r.chain {
                return false;
            }
            prev = r.chain;
        }
        true
    }
}

// ───────────────────────── DST canary ─────────────────────────

/// A DST canary: replay a recorded workload through a candidate build and the known-good
/// build; activation is gated on the candidate producing identical outputs (no regression).
pub struct Canary;

impl Canary {
    /// Run the `workload` through both implementations and report whether the candidate
    /// matches the known-good on every input. Only a clean run should gate activation.
    pub fn passes<I, O, Good, Cand>(workload: &[I], known_good: Good, candidate: Cand) -> bool
    where
        O: PartialEq,
        Good: Fn(&I) -> O,
        Cand: Fn(&I) -> O,
    {
        workload.iter().all(|input| known_good(input) == candidate(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_reverts_after_repeated_health_failures() {
        let mut wd = HealthWatchdog::arm(0, 1, 3); // slot 0 good, slot 1 candidate
        assert_eq!(wd.active_slot(), 1);
        wd.report(false);
        wd.report(true); // a pass resets the streak
        wd.report(false);
        wd.report(false);
        assert!(!wd.reverted()); // only 2 consecutive failures
        wd.report(false); // third consecutive → revert
        assert!(wd.reverted());
        assert_eq!(wd.active_slot(), 0); // back to known-good
    }

    #[test]
    fn audited_install_is_capability_scoped_and_logged() {
        let mut inst = AuditedInstall::new(&[1, 2]); // may install to domains 1 and 2
        assert!(inst.install(1, Hash256::of(b"pkg-a")).is_ok());
        assert!(inst.install(2, Hash256::of(b"pkg-b")).is_ok());
        // Domain 9 is not authorised.
        assert_eq!(inst.install(9, Hash256::of(b"pkg-c")), Err(InstallDenied::Unauthorized));
        assert_eq!(inst.log().len(), 2);
        assert!(inst.audit_intact());
    }

    #[test]
    fn dst_canary_gates_activation_on_no_regression() {
        let workload: Vec<i64> = (0..10).collect();
        // Candidate matches the known-good → canary passes (safe to activate).
        assert!(Canary::passes(&workload, |x: &i64| x * 2, |x: &i64| x + x));
        // Candidate regresses on some inputs → canary fails (block activation).
        assert!(!Canary::passes(&workload, |x: &i64| x * 2, |x: &i64| x * 2 + 1));
    }
}

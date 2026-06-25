//! Update & upgrade lifecycle (see `docs/implementation/update-and-upgrade-lifecycle.md`).
//!
//! An OS that boots from a content-addressed object graph updates the way it stores
//! everything else: atomically and verifiably. The model here is **A/B slots with
//! signed, content-addressed releases**:
//!
//! * A release is identified by the **hash of its image**; the package is
//!   authenticated (a vendor MAC here; the production path is the hybrid PQ
//!   signature of [`crate::crypto`]).
//! * Staging writes to the **inactive slot** and verifies *both* the content hash
//!   and the authenticity tag before anything switches — a corrupt or unsigned
//!   image can never become active.
//! * **Anti-downgrade**: a release older than the running version is refused.
//! * **Commit** flips the active slot atomically; **rollback** restores the prior
//!   slot, so a bad update is always recoverable.
//! * **Staged rollout**: a device decides via a stable hash whether it is in the
//!   current rollout cohort, so releases ramp instead of flag-day.
//!
//! Pure, safe `no_std`, host-tested.

use crate::hash::Hash256;
use alloc::vec::Vec;

/// Keyed authenticity tag (vendor MAC over `version ‖ image-hash`).
fn vendor_tag(key: &[u8], version: u64, image: Hash256) -> Hash256 {
    let mut input = Vec::with_capacity(key.len() + 40);
    input.extend_from_slice(key);
    input.extend_from_slice(b"release:");
    input.extend_from_slice(&version.to_le_bytes());
    input.extend_from_slice(&image.0);
    Hash256::of(&input)
}

/// A signed, content-addressed release package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleasePackage {
    pub version: u64,
    pub image_hash: Hash256,
    pub tag: Hash256,
    /// Rollout cohort threshold, 0..=1000 (per-mille of the fleet eligible).
    pub rollout_milli: u32,
}

impl ReleasePackage {
    /// Build and sign a release from its raw image bytes.
    pub fn build(key: &[u8], version: u64, image: &[u8], rollout_milli: u32) -> ReleasePackage {
        let image_hash = Hash256::of(image);
        ReleasePackage {
            version,
            image_hash,
            tag: vendor_tag(key, version, image_hash),
            rollout_milli: rollout_milli.min(1000),
        }
    }

    /// Verify authenticity *and* that `image` actually hashes to the claimed hash.
    pub fn verify(&self, key: &[u8], image: &[u8]) -> bool {
        Hash256::of(image) == self.image_hash
            && vendor_tag(key, self.version, self.image_hash) == self.tag
    }

    /// Is `device_id` in this release's rollout cohort? Stable per device+version.
    pub fn in_cohort(&self, device_id: u64) -> bool {
        let mut input = Vec::with_capacity(16);
        input.extend_from_slice(&device_id.to_le_bytes());
        input.extend_from_slice(&self.version.to_le_bytes());
        let h = Hash256::of(&input).0;
        let bucket = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) % 1000;
        bucket < self.rollout_milli
    }
}

/// Which physical slot is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    A,
    B,
}

impl Slot {
    fn other(self) -> Slot {
        match self {
            Slot::A => Slot::B,
            Slot::B => Slot::A,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateError {
    Unverified,
    Downgrade,
    NotStaged,
    NotInCohort,
}

/// Manages the two boot slots and the staged update.
pub struct UpdateManager {
    vendor_key: Vec<u8>,
    active: Slot,
    active_version: u64,
    /// The version sitting in the inactive slot, verified and ready to commit.
    staged: Option<u64>,
    prev_version: Option<u64>,
}

impl UpdateManager {
    pub fn new(vendor_key: &[u8], initial_version: u64) -> UpdateManager {
        UpdateManager {
            vendor_key: vendor_key.to_vec(),
            active: Slot::A,
            active_version: initial_version,
            staged: None,
            prev_version: None,
        }
    }

    pub fn active_slot(&self) -> Slot {
        self.active
    }
    pub fn active_version(&self) -> u64 {
        self.active_version
    }
    pub fn staged_version(&self) -> Option<u64> {
        self.staged
    }

    /// Stage a release into the inactive slot. Verifies authenticity, content hash,
    /// anti-downgrade, and rollout cohort before accepting.
    pub fn stage(&mut self, pkg: &ReleasePackage, image: &[u8], device_id: u64) -> Result<Slot, UpdateError> {
        if !pkg.verify(&self.vendor_key, image) {
            return Err(UpdateError::Unverified);
        }
        if pkg.version <= self.active_version {
            return Err(UpdateError::Downgrade);
        }
        if !pkg.in_cohort(device_id) {
            return Err(UpdateError::NotInCohort);
        }
        self.staged = Some(pkg.version);
        Ok(self.active.other())
    }

    /// Atomically switch to the staged slot. The previous version is retained for
    /// rollback.
    pub fn commit(&mut self) -> Result<Slot, UpdateError> {
        let version = self.staged.take().ok_or(UpdateError::NotStaged)?;
        self.prev_version = Some(self.active_version);
        self.active = self.active.other();
        self.active_version = version;
        Ok(self.active)
    }

    /// Revert to the previously active slot/version (a bad update recovery).
    pub fn rollback(&mut self) -> Result<u64, UpdateError> {
        let prev = self.prev_version.take().ok_or(UpdateError::NotStaged)?;
        self.active = self.active.other();
        self.active_version = prev;
        Ok(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"vendor-provisioned-key";

    #[test]
    fn build_and_verify_release() {
        let image = b"kernel image v2 bytes...";
        let pkg = ReleasePackage::build(KEY, 2, image, 1000);
        assert!(pkg.verify(KEY, image));
        // Tampered image fails the content-hash check.
        assert!(!pkg.verify(KEY, b"tampered image"));
        // Wrong key fails the authenticity check.
        assert!(!pkg.verify(b"attacker-key", image));
    }

    #[test]
    fn staging_requires_verification() {
        let mut mgr = UpdateManager::new(KEY, 1);
        let image = b"v2";
        let pkg = ReleasePackage::build(KEY, 2, image, 1000);
        // Wrong image bytes → unverified.
        assert_eq!(mgr.stage(&pkg, b"not-v2", 0), Err(UpdateError::Unverified));
        // Correct image stages into the *other* slot.
        assert_eq!(mgr.stage(&pkg, image, 0), Ok(Slot::B));
        assert_eq!(mgr.staged_version(), Some(2));
    }

    #[test]
    fn anti_downgrade_is_enforced() {
        let mut mgr = UpdateManager::new(KEY, 5);
        let old = ReleasePackage::build(KEY, 3, b"v3", 1000);
        assert_eq!(mgr.stage(&old, b"v3", 0), Err(UpdateError::Downgrade));
        let same = ReleasePackage::build(KEY, 5, b"v5", 1000);
        assert_eq!(mgr.stage(&same, b"v5", 0), Err(UpdateError::Downgrade));
    }

    #[test]
    fn commit_switches_slot_atomically_and_rollback_restores() {
        let mut mgr = UpdateManager::new(KEY, 1);
        let pkg = ReleasePackage::build(KEY, 2, b"v2", 1000);
        mgr.stage(&pkg, b"v2", 0).unwrap();
        assert_eq!(mgr.active_slot(), Slot::A);
        mgr.commit().unwrap();
        assert_eq!(mgr.active_slot(), Slot::B);
        assert_eq!(mgr.active_version(), 2);
        // A bad update: roll back to v1 on slot A.
        let restored = mgr.rollback().unwrap();
        assert_eq!(restored, 1);
        assert_eq!(mgr.active_slot(), Slot::A);
        assert_eq!(mgr.active_version(), 1);
    }

    #[test]
    fn commit_without_staging_fails() {
        let mut mgr = UpdateManager::new(KEY, 1);
        assert_eq!(mgr.commit(), Err(UpdateError::NotStaged));
    }

    #[test]
    fn staged_rollout_gates_by_cohort() {
        // A 10% rollout admits roughly a tenth of devices and is stable per device.
        let pkg = ReleasePackage::build(KEY, 9, b"img", 100); // 100/1000 = 10%
        let admitted = (0..1000u64).filter(|&d| pkg.in_cohort(d)).count();
        assert!(admitted > 30 && admitted < 200, "cohort size off: {admitted}");
        // Determinism: the same device gets the same answer.
        let d = 12345;
        assert_eq!(pkg.in_cohort(d), pkg.in_cohort(d));
        // A 100% rollout admits everyone.
        let full = ReleasePackage::build(KEY, 9, b"img", 1000);
        assert!((0..200u64).all(|d| full.in_cohort(d)));
    }

    #[test]
    fn out_of_cohort_device_cannot_stage() {
        let mut mgr = UpdateManager::new(KEY, 1);
        // 0% rollout: nobody is in cohort.
        let pkg = ReleasePackage::build(KEY, 2, b"v2", 0);
        assert_eq!(mgr.stage(&pkg, b"v2", 42), Err(UpdateError::NotInCohort));
    }
}

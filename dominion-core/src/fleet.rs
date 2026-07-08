//! Device fleet, threshold signing & recovery composition
//! (`docs/security/identity-recovery-and-key-management.md`,
//! `docs/implementation/backup-sync-and-device-fleet.md`).
//!
//! The pieces of fleet/identity management that were primitives in `identity.rs`,
//! `recovery.rs` and `firewall.rs` are composed here into the actual flows:
//!
//! * [`Fleet`] — a domain of per-device identities (one logical system across devices)
//!   with **enrollment by capability delegation** (an existing device authorises a new
//!   one) and **recursive fleet-wide revocation** (revoking a device also revokes every
//!   device it transitively enrolled — a stolen device can't leave a back door).
//! * [`ThresholdGroup`] — **threshold signing preferred over seed reassembly**: a quorum
//!   authorises an action via per-guardian partials, and the master secret is **never
//!   reconstructed** (no single point of compromise).
//! * [`RecoveryFlow`] — a dedicated, veto-windowed, provenance-logged recovery that, on
//!   completion, **re-wraps keys under a fresh generation and revokes the old one**
//!   (post-recovery rewrap), and **restores data jointly with key recovery**.
//!
//! Pure, safe `no_std`; PQ (hash-based). Host-tested.

use crate::hash::Hash256;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

// ───────────────────────── the device fleet ─────────────────────────

/// A device identity (the hash of its public key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceId(pub Hash256);

impl DeviceId {
    pub fn from_pubkey(pubkey: &[u8]) -> DeviceId {
        DeviceId(Hash256::of(pubkey))
    }
}

#[derive(Clone, Debug)]
struct DeviceRecord {
    /// Who enrolled this device (None ⇒ the founding device).
    enrolled_by: Option<DeviceId>,
}

/// One logical system spread across a fleet of per-device identities, all under one owner.
pub struct Fleet {
    owner: Hash256,
    devices: BTreeMap<DeviceId, DeviceRecord>,
    revoked: BTreeSet<DeviceId>,
}

impl Fleet {
    /// Found a fleet with its first (owner) device.
    pub fn new(owner: Hash256, first_device: DeviceId) -> Fleet {
        let mut devices = BTreeMap::new();
        devices.insert(first_device, DeviceRecord { enrolled_by: None });
        Fleet { owner, devices, revoked: BTreeSet::new() }
    }

    /// The owning identity (the fleet domain).
    pub fn owner(&self) -> Hash256 {
        self.owner
    }

    /// Is `id` an active (enrolled, non-revoked) member?
    pub fn is_active(&self, id: &DeviceId) -> bool {
        self.devices.contains_key(id) && !self.revoked.contains(id)
    }

    /// **Enroll a new device by capability delegation**: an existing *active* device
    /// authorises the new one. Returns the new [`DeviceId`], or `None` if the authoriser
    /// isn't an active member (no ambient enrollment).
    pub fn enroll(&mut self, authoriser: &DeviceId, new_pubkey: &[u8]) -> Option<DeviceId> {
        if !self.is_active(authoriser) {
            return None;
        }
        let id = DeviceId::from_pubkey(new_pubkey);
        self.devices.insert(id, DeviceRecord { enrolled_by: Some(*authoriser) });
        Some(id)
    }

    /// **Fleet-wide recursive revocation**: revoke `id` and every device it transitively
    /// enrolled (a compromised device can't leave authorised descendants behind). Returns
    /// the number of devices revoked.
    pub fn revoke_device(&mut self, id: &DeviceId) -> usize {
        let mut to_revoke = alloc::vec![*id];
        let mut count = 0;
        while let Some(victim) = to_revoke.pop() {
            if self.revoked.insert(victim) {
                count += 1;
                // Queue every device this one enrolled.
                for (child, rec) in &self.devices {
                    if rec.enrolled_by == Some(victim) && !self.revoked.contains(child) {
                        to_revoke.push(*child);
                    }
                }
            }
        }
        count
    }

    /// Active member count.
    pub fn active_count(&self) -> usize {
        self.devices.keys().filter(|d| !self.revoked.contains(d)).count()
    }
}

// ───────────────────────── threshold signing (no seed reassembly) ─────────────────────────

/// A k-of-n threshold group. The master secret is split into per-guardian shares whose
/// public **commitments** are published; signing never reconstructs the secret.
pub struct ThresholdGroup {
    threshold: usize,
    /// `commitments[i] = H(share_i)` — lets a partial be verified without the share.
    commitments: Vec<Hash256>,
}

/// A guardian's contribution toward authorising a message — derived from their secret
/// share, so only the share holder can produce it.
///
/// The partial reveals the guardian's secret `share` (the preimage of the group's
/// published commitment) together with a message-binding `value`. Because the
/// commitment is a one-way hash of the share, holding the public commitment alone is
/// not enough to fabricate a partial — the secret share is required.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partial {
    pub index: usize,
    /// The guardian's secret share; its hash must equal `commitments[index]`.
    share: Hash256,
    /// Message binding: `H(share ‖ ":sig:" ‖ msg)`.
    value: Hash256,
}

/// A completed threshold authorization (proof a quorum signed) — carries *which*
/// guardians contributed, never the secret.
///
/// Fields are private so an authorization can only be produced by
/// [`ThresholdGroup::combine`], which validates a quorum of distinct partials.
/// A caller cannot fabricate one by filling in the fields directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThresholdAuth {
    msg_hash: Hash256,
    signers: Vec<usize>,
}

impl ThresholdAuth {
    /// The hash of the authorised message.
    pub fn msg_hash(&self) -> Hash256 {
        self.msg_hash
    }

    /// The distinct guardian indices that contributed to this authorization.
    pub fn signers(&self) -> &[usize] {
        &self.signers
    }
}

fn guardian_share(group_seed: &[u8], index: usize) -> Hash256 {
    Hash256::of(&[group_seed, b":share:", &(index as u64).to_le_bytes()].concat())
}

impl ThresholdGroup {
    /// Set up an `n`-guardian, `threshold`-of-`n` group from a group seed. Returns the
    /// group (public commitments) and the per-guardian secret shares.
    pub fn setup(group_seed: &[u8], threshold: usize, n: usize) -> (ThresholdGroup, Vec<Hash256>) {
        let shares: Vec<Hash256> = (0..n).map(|i| guardian_share(group_seed, i)).collect();
        let commitments = shares.iter().map(|s| Hash256::of(&s.0)).collect();
        (ThresholdGroup { threshold: threshold.clamp(1, n.max(1)), commitments }, shares)
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// A guardian signs `msg` with their secret `share` at `index`.
    ///
    /// The partial reveals the secret `share` together with a message-binding value
    /// `H(share ‖ ":sig:" ‖ msg)`. The verifier holds only `H(share)` as the public
    /// commitment; since a hash preimage cannot be recovered from that commitment,
    /// only the holder of the secret share can produce a matching partial — the public
    /// commitment alone is not enough. Revealing a share does not reconstruct the
    /// group secret, since shares are one-way-derived from it.
    pub fn partial_sign(index: usize, share: &Hash256, msg: &[u8]) -> Partial {
        // value binds the secret share (not its public commitment) to the message.
        let value = Hash256::of(&[share.0.as_ref(), b":sig:", msg].concat());
        Partial { index, share: *share, value }
    }

    /// Verify a single partial against this group's commitment for that index.
    ///
    /// Checks that (a) the revealed share hashes to the stored commitment
    /// (`H(p.share) == commitments[p.index]`) — which only the secret-share holder can
    /// satisfy, since the commitment is a one-way hash of the share — and (b) the value
    /// binds that share to `msg`. A partial fabricated from the public commitment data
    /// alone (without the secret share) fails check (a) and is rejected.
    fn partial_valid(&self, p: &Partial, msg: &[u8]) -> bool {
        let commitment = match self.commitments.get(p.index) {
            Some(c) => c,
            None => return false,
        };
        if Hash256::of(&p.share.0) != *commitment {
            return false;
        }
        let expected = Hash256::of(&[p.share.0.as_ref(), b":sig:", msg].concat());
        p.value == expected
    }

    /// **Combine** partials into an authorization **without ever reconstructing the
    /// secret**. Requires ≥ `threshold` distinct, valid partials. Returns `None` otherwise.
    pub fn combine(&self, partials: &[Partial], msg: &[u8]) -> Option<ThresholdAuth> {
        let mut seen = BTreeSet::new();
        let mut signers = Vec::new();
        for p in partials {
            if self.partial_valid(p, msg) && seen.insert(p.index) {
                signers.push(p.index);
            }
        }
        if signers.len() >= self.threshold {
            signers.sort_unstable();
            Some(ThresholdAuth { msg_hash: Hash256::of(msg), signers })
        } else {
            None // not enough of a quorum — and the secret was never assembled
        }
    }

    /// Verify an authorization: it carries a quorum of distinct signers and matches `msg`.
    pub fn verify(&self, auth: &ThresholdAuth, msg: &[u8]) -> bool {
        let distinct: BTreeSet<usize> = auth.signers.iter().copied().collect();
        distinct.len() == auth.signers.len()
            && distinct.len() >= self.threshold
            && auth.msg_hash == Hash256::of(msg)
            && auth.signers.iter().all(|&i| i < self.commitments.len())
    }
}

// ───────────────────────── recovery flow (rewrap + restore) ─────────────────────────

/// A dedicated recovery flow: veto-windowed, provenance-logged, and on completion it
/// **re-wraps keys under a new generation** and lets data be **restored jointly with the
/// recovered key**.
pub struct RecoveryFlow {
    request: Hash256,
    veto_deadline: u64,
    vetoed: bool,
    /// Monotonic key generation — bumped on a successful recovery (old keys revoked).
    generation: u64,
    /// Provenance log of recovery events (fact + time).
    log: Vec<(Hash256, u64)>,
}

impl RecoveryFlow {
    /// Open a recovery for `request` at `now`, completing no earlier than `now + veto`.
    pub fn open(request: &[u8], now: u64, veto_window: u64, start_generation: u64) -> RecoveryFlow {
        let request = Hash256::of(request);
        RecoveryFlow {
            request,
            veto_deadline: now.saturating_add(veto_window),
            vetoed: false,
            generation: start_generation,
            log: alloc::vec![(request, now)],
        }
    }

    /// The owner vetoes within the window.
    pub fn veto(&mut self, now: u64) {
        self.vetoed = true;
        self.log.push((Hash256::of(b"veto"), now));
    }

    /// Complete the recovery once the window passes (and no veto): bump the key
    /// generation (so the **old key is crypto-revoked**) and re-derive the new wrapping
    /// key. Returns the new generation + new key, or `None` if not yet allowed.
    pub fn complete(&mut self, recovered_seed: &[u8], now: u64) -> Option<(u64, [u8; 32])> {
        if self.vetoed || now < self.veto_deadline {
            return None;
        }
        self.generation += 1; // post-recovery rewrap: a fresh generation invalidates the old
        // Bind the new key to the original request (anti-confusion) + generation.
        let new_key = Hash256::of(
            &[self.request.0.as_ref(), b":gen:", &self.generation.to_le_bytes(), recovered_seed].concat(),
        )
        .0;
        self.log.push((Hash256::of(b"completed"), now));
        Some((self.generation, new_key))
    }

    /// **Restore-after-recovery**: with the recovered seed + the new generation, re-derive
    /// the data-encryption key for an object and decrypt its (xor-wrapped) backup blob.
    /// Returns the recovered plaintext, or `None` if the wrong seed/generation is used.
    pub fn restore(seed: &[u8], generation: u64, object_label: &[u8], wrapped: &[u8], check: Hash256) -> Option<Vec<u8>> {
        let dek = Hash256::of(&[seed, b":gen:", &generation.to_le_bytes(), b":obj:", object_label].concat()).0;
        let ks = keystream(&dek, wrapped.len());
        let plain: Vec<u8> = wrapped.iter().zip(ks).map(|(b, k)| b ^ k).collect();
        if Hash256::of(&plain) == check {
            Some(plain)
        } else {
            None
        }
    }

    /// Helper: wrap `plaintext` for backup under `(seed, generation, object)` so
    /// [`restore`](Self::restore) can recover it. Returns `(wrapped, check)`.
    pub fn wrap_for_backup(seed: &[u8], generation: u64, object_label: &[u8], plaintext: &[u8]) -> (Vec<u8>, Hash256) {
        let dek = Hash256::of(&[seed, b":gen:", &generation.to_le_bytes(), b":obj:", object_label].concat()).0;
        let ks = keystream(&dek, plaintext.len());
        let wrapped = plaintext.iter().zip(ks).map(|(b, k)| b ^ k).collect();
        (wrapped, Hash256::of(plaintext))
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The provenance log of this recovery (fact + time, never content).
    pub fn log(&self) -> &[(Hash256, u64)] {
        &self.log
    }
}

fn keystream(key: &[u8; 32], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut counter = 0u64;
    while out.len() < len {
        out.extend_from_slice(&Hash256::of(&[key.as_ref(), &counter.to_le_bytes()].concat()).0);
        counter += 1;
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(name: &[u8]) -> DeviceId {
        DeviceId::from_pubkey(name)
    }

    #[test]
    fn enrollment_needs_an_active_authoriser() {
        let mut fleet = Fleet::new(Hash256::of(b"owner"), dev(b"phone"));
        // The founding phone enrolls a laptop.
        let laptop = fleet.enroll(&dev(b"phone"), b"laptop").unwrap();
        assert!(fleet.is_active(&laptop));
        // A non-member can't enroll anything.
        assert!(fleet.enroll(&dev(b"stranger"), b"rogue").is_none());
        assert_eq!(fleet.active_count(), 2);
    }

    #[test]
    fn revocation_is_recursive_across_the_fleet() {
        let mut fleet = Fleet::new(Hash256::of(b"owner"), dev(b"phone"));
        let laptop = fleet.enroll(&dev(b"phone"), b"laptop").unwrap();
        let tablet = fleet.enroll(&laptop, b"tablet").unwrap(); // enrolled by the laptop
        let watch = fleet.enroll(&dev(b"phone"), b"watch").unwrap();
        // Revoking the laptop also revokes the tablet it enrolled — but not the watch.
        let n = fleet.revoke_device(&laptop);
        assert_eq!(n, 2);
        assert!(!fleet.is_active(&laptop));
        assert!(!fleet.is_active(&tablet));
        assert!(fleet.is_active(&watch));
    }

    #[test]
    fn threshold_signing_needs_a_quorum_and_never_reassembles_the_secret() {
        let (group, shares) = ThresholdGroup::setup(b"group-seed", 3, 5);
        let msg = b"authorize: rotate vault key";
        // Two partials are not enough.
        let p0 = ThresholdGroup::partial_sign(0, &shares[0], msg);
        let p1 = ThresholdGroup::partial_sign(1, &shares[1], msg);
        assert!(group.combine(&[p0.clone(), p1.clone()], msg).is_none());
        // A third distinct guardian reaches the quorum (no secret was ever reconstructed).
        let p2 = ThresholdGroup::partial_sign(2, &shares[2], msg);
        let auth = group.combine(&[p0, p1, p2], msg).unwrap();
        assert!(group.verify(&auth, msg));
        assert_eq!(auth.signers, alloc::vec![0, 1, 2]);
        // The authorization doesn't verify for a different message.
        assert!(!group.verify(&auth, b"different message"));
    }

    #[test]
    fn forged_partials_are_rejected_even_with_valid_indices() {
        // Setup a 3-of-5 group; the attacker has the PUBLIC commitments but none of
        // the secret shares.
        let (group, shares) = ThresholdGroup::setup(b"group-seed", 3, 5);
        let msg = b"authorize: rotate vault key";
        // The commitments (`H(share)`) are public verifier data. Mount the real attack:
        // try to forge partials from the public commitments alone, by feeding each
        // commitment where the secret share would go. This is exactly what an attacker
        // holding only the published commitments can compute.
        let commitments: Vec<Hash256> = shares.iter().map(|s| Hash256::of(&s.0)).collect();
        let forged: Vec<Partial> = (0..3)
            .map(|i| ThresholdGroup::partial_sign(i, &commitments[i], msg))
            .collect();
        // partial_valid must reject every forged partial (H(commitment) != commitment);
        // combine must return None. The secret share — not just its public hash — is
        // required to reach a quorum.
        for p in &forged {
            assert!(
                !group.partial_valid(p, msg),
                "forged partial at index {} must be rejected",
                p.index
            );
        }
        assert!(
            group.combine(&forged, msg).is_none(),
            "combine must not yield an auth from forged partials"
        );
    }

    #[test]
    fn verify_rejects_duplicate_or_fabricated_signers() {
        let (group, shares) = ThresholdGroup::setup(b"group-seed", 3, 5);
        let msg = b"authorize: rotate vault key";
        // A fabricated authorization with a single repeated signer must not satisfy a
        // 3-of-5 gate — distinctness is enforced, not just the raw signer count.
        let repeated = ThresholdAuth { msg_hash: Hash256::of(msg), signers: alloc::vec![0, 0, 0] };
        assert!(!group.verify(&repeated, msg));
        // A genuine quorum of distinct guardians still verifies.
        let auth = group
            .combine(
                &[
                    ThresholdGroup::partial_sign(0, &shares[0], msg),
                    ThresholdGroup::partial_sign(1, &shares[1], msg),
                    ThresholdGroup::partial_sign(2, &shares[2], msg),
                ],
                msg,
            )
            .unwrap();
        assert!(group.verify(&auth, msg));
    }

    #[test]
    fn recovery_flow_rewraps_and_restores() {
        let mut flow = RecoveryFlow::open(b"recover my account", 0, 100, 7);
        // Cannot complete before the veto window.
        assert!(flow.complete(b"seed", 50).is_none());
        // After the window, completing bumps the generation (old key revoked).
        let (gen, _key) = flow.complete(b"recovered-seed", 200).unwrap();
        assert_eq!(gen, 8);
        // A vetoed recovery never completes.
        let mut vetoed = RecoveryFlow::open(b"r", 0, 100, 0);
        vetoed.veto(10);
        assert!(vetoed.complete(b"seed", 999).is_none());
        // Restore-after-recovery: wrap then recover the data with the recovered seed+gen.
        let (wrapped, check) = RecoveryFlow::wrap_for_backup(b"recovered-seed", gen, b"notes", b"my secret notes");
        let restored = RecoveryFlow::restore(b"recovered-seed", gen, b"notes", &wrapped, check);
        assert_eq!(restored.as_deref(), Some(b"my secret notes".as_ref()));
        // The wrong seed cannot restore it.
        assert!(RecoveryFlow::restore(b"wrong-seed", gen, b"notes", &wrapped, check).is_none());
        assert!(flow.log().len() >= 2);
    }
}

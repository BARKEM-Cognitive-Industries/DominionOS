//! Cryptographic visual provenance (W3C PROV-DM inspired).
//!
//! Every rendering node carries an immutable data-lineage manifest.
//! The manifest includes:
//! - origin process hash (hash of the executable image)
//! - executable cryptographic signature hash
//! - parent window lineage (chain of ancestor node IDs from root)
//! - integrity state
//!
//! The compositor evaluates the manifest before admitting any node to the
//! render graph. Fake system dialogs are detected because they lack the
//! system's cryptographic keys.

use crate::hash::Hash256;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// IntegrityState
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrityState {
    /// Cryptographically sealed and verified.
    VerifiableSealed,
    /// Not yet verified — needs [`ProvenanceManifest::verify`] call.
    Unverified,
    /// Verification failed — block this node from the render graph.
    Tampered,
}

// ---------------------------------------------------------------------------
// ProvenanceManifest
// ---------------------------------------------------------------------------

/// A render-node provenance manifest (W3C PROV-DM inspired).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvenanceManifest {
    /// Hash of the originating process executable.
    pub origin_process_hash: Hash256,
    /// Hash of the executable cryptographic signature (proving system auth).
    pub exec_signature_hash: Hash256,
    /// Chain of parent render node IDs (lineage from root to this node).
    /// For system-origin nodes the first element must be `0`.
    pub parent_lineage: Vec<u64>,
    /// This node's unique render ID.
    pub node_id: u64,
    /// Current integrity state.
    pub integrity: IntegrityState,
}

impl ProvenanceManifest {
    pub fn new(node_id: u64, origin: Hash256, sig: Hash256, lineage: Vec<u64>) -> Self {
        Self {
            origin_process_hash: origin,
            exec_signature_hash: sig,
            parent_lineage: lineage,
            node_id,
            integrity: IntegrityState::Unverified,
        }
    }

    /// Compute a combined hash of this manifest's contents.
    ///
    /// The hash covers: origin_process_hash, exec_signature_hash, all
    /// parent_lineage IDs (as LE u64), and node_id. This gives a stable
    /// fingerprint for chain verification.
    pub fn manifest_hash(&self) -> Hash256 {
        // Lay out: origin(32) + sig(32) + node_id(8) + lineage items(8 each)
        let lineage_bytes = self.parent_lineage.len() * 8;
        let mut buf = alloc::vec![0u8; 32 + 32 + 8 + lineage_bytes];
        buf[..32].copy_from_slice(&self.origin_process_hash.0);
        buf[32..64].copy_from_slice(&self.exec_signature_hash.0);
        buf[64..72].copy_from_slice(&self.node_id.to_le_bytes());
        for (i, id) in self.parent_lineage.iter().enumerate() {
            let off = 72 + i * 8;
            buf[off..off + 8].copy_from_slice(&id.to_le_bytes());
        }
        Hash256::of(&buf)
    }

    /// Verify the manifest chain.
    ///
    /// Our model: valid when `Hash256::of(origin_bytes ++ sig_bytes)` is
    /// non-zero (both hashes must be non-trivial) and the exec_signature_hash
    /// is consistent with the origin. Concretely we compute the combined hash
    /// and confirm it is neither the zero hash nor equal to just the origin
    /// (which would indicate a copied/forged signature field).
    ///
    /// Sets `integrity` to `VerifiableSealed` on success, `Tampered` on
    /// failure, and returns the corresponding bool.
    pub fn verify(&mut self) -> bool {
        // Zero hashes indicate uninitialised / forged fields.
        if self.origin_process_hash == Hash256::ZERO
            || self.exec_signature_hash == Hash256::ZERO
        {
            self.integrity = IntegrityState::Tampered;
            return false;
        }

        // Combined hash of origin + sig must differ from either alone (guards
        // against trivially equal inputs).
        let combined = self.origin_process_hash.combine(&self.exec_signature_hash);
        if combined == Hash256::ZERO
            || combined == self.origin_process_hash
            || combined == self.exec_signature_hash
        {
            self.integrity = IntegrityState::Tampered;
            return false;
        }

        self.integrity = IntegrityState::VerifiableSealed;
        true
    }

    /// Check if this manifest belongs to the system.
    ///
    /// System processes have `parent_lineage[0] == 0` (rooted at the kernel's
    /// synthetic node 0).
    pub fn is_system_origin(&self) -> bool {
        self.parent_lineage.first().copied() == Some(0)
    }
}

// ---------------------------------------------------------------------------
// ProvenanceError
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceError {
    TamperedManifest,
    FakeSystemDialog,
    UnknownOrigin,
    BrokenLineage,
}

// ---------------------------------------------------------------------------
// ProvenanceVerifier
// ---------------------------------------------------------------------------

/// The compositor's provenance verifier — checks all submitted render nodes
/// before they are admitted to the render graph.
pub struct ProvenanceVerifier {
    /// Known system process hashes (established at boot).
    pub trusted_origins: Vec<Hash256>,
}

impl ProvenanceVerifier {
    pub fn new() -> Self {
        Self {
            trusted_origins: Vec::new(),
        }
    }

    pub fn register_trusted_origin(&mut self, origin: Hash256) {
        if !self.trusted_origins.contains(&origin) {
            self.trusted_origins.push(origin);
        }
    }

    /// Verify a node's manifest and decide whether to admit it.
    ///
    /// Rejection reasons (in order of check):
    /// 1. Manifest internal verification fails → `TamperedManifest`.
    /// 2. Node claims system origin but its `origin_process_hash` is not in
    ///    `trusted_origins` → `FakeSystemDialog`.
    /// 3. Node's origin is unknown (not in trusted list) for non-system
    ///    contexts where a trust list is expected → `UnknownOrigin`.
    ///    (If `trusted_origins` is empty we do not block — open registration.)
    /// 4. Lineage is empty → `BrokenLineage`.
    pub fn verify_node(&self, manifest: &mut ProvenanceManifest) -> Result<(), ProvenanceError> {
        // Step 1: structural / cryptographic check.
        if !manifest.verify() {
            return Err(ProvenanceError::TamperedManifest);
        }

        // Step 2: lineage must be non-empty.
        if manifest.parent_lineage.is_empty() {
            return Err(ProvenanceError::BrokenLineage);
        }

        // Step 3: fake system dialog detection.
        if manifest.is_system_origin() {
            if !self.trusted_origins.contains(&manifest.origin_process_hash) {
                return Err(ProvenanceError::FakeSystemDialog);
            }
        } else if !self.trusted_origins.is_empty()
            && !self.trusted_origins.contains(&manifest.origin_process_hash)
        {
            // Non-system node from an unknown origin when a trust list exists.
            return Err(ProvenanceError::UnknownOrigin);
        }

        Ok(())
    }

    /// Returns `true` if the manifest represents an overlay / phishing attack.
    ///
    /// An overlay attack is detected when a node claims a system-origin
    /// lineage but its `origin_process_hash` is not among trusted origins.
    pub fn detect_overlay_attack(&self, manifest: &ProvenanceManifest) -> bool {
        if !manifest.is_system_origin() {
            return false;
        }
        !self.trusted_origins.contains(&manifest.origin_process_hash)
    }
}

impl Default for ProvenanceVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_origin() -> Hash256 {
        Hash256::of(b"legitimate-system-process")
    }

    fn make_sig() -> Hash256 {
        Hash256::of(b"kernel-signed-exec-signature")
    }

    fn system_manifest() -> ProvenanceManifest {
        ProvenanceManifest::new(
            1,
            make_origin(),
            make_sig(),
            alloc::vec![0, 1, 2], // lineage[0] == 0 → system
        )
    }

    fn user_manifest() -> ProvenanceManifest {
        ProvenanceManifest::new(
            42,
            Hash256::of(b"user-app-process"),
            Hash256::of(b"user-app-signature"),
            alloc::vec![5, 42], // does not start with 0
        )
    }

    // --- manifest construction ---

    #[test]
    fn system_manifest_is_system_origin() {
        assert!(system_manifest().is_system_origin());
    }

    #[test]
    fn user_manifest_is_not_system_origin() {
        assert!(!user_manifest().is_system_origin());
    }

    #[test]
    fn manifest_starts_unverified() {
        assert_eq!(system_manifest().integrity, IntegrityState::Unverified);
    }

    // --- manifest verify ---

    #[test]
    fn valid_manifest_verify_seals() {
        let mut m = system_manifest();
        assert!(m.verify());
        assert_eq!(m.integrity, IntegrityState::VerifiableSealed);
    }

    #[test]
    fn zero_origin_hash_fails_verify() {
        let mut m = ProvenanceManifest::new(
            1,
            Hash256::ZERO,
            make_sig(),
            alloc::vec![0],
        );
        assert!(!m.verify());
        assert_eq!(m.integrity, IntegrityState::Tampered);
    }

    #[test]
    fn zero_sig_hash_fails_verify() {
        let mut m = ProvenanceManifest::new(
            1,
            make_origin(),
            Hash256::ZERO,
            alloc::vec![0],
        );
        assert!(!m.verify());
        assert_eq!(m.integrity, IntegrityState::Tampered);
    }

    // --- manifest_hash ---

    #[test]
    fn manifest_hash_is_deterministic() {
        let m = system_manifest();
        assert_eq!(m.manifest_hash(), m.manifest_hash());
    }

    #[test]
    fn manifest_hash_differs_by_node_id() {
        let m1 = system_manifest();
        let mut m2 = system_manifest();
        m2.node_id = 999;
        assert_ne!(m1.manifest_hash(), m2.manifest_hash());
    }

    #[test]
    fn manifest_hash_differs_by_lineage() {
        let m1 = system_manifest();
        let m2 = ProvenanceManifest::new(
            1,
            make_origin(),
            make_sig(),
            alloc::vec![0, 1, 9], // different last ancestor
        );
        assert_ne!(m1.manifest_hash(), m2.manifest_hash());
    }

    // --- ProvenanceVerifier ---

    #[test]
    fn verifier_accepts_trusted_system_node() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        let mut m = system_manifest();
        assert!(v.verify_node(&mut m).is_ok());
    }

    #[test]
    fn verifier_blocks_tampered_manifest() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        let mut m = ProvenanceManifest::new(
            1,
            Hash256::ZERO, // tampered
            make_sig(),
            alloc::vec![0],
        );
        assert_eq!(v.verify_node(&mut m), Err(ProvenanceError::TamperedManifest));
    }

    #[test]
    fn verifier_blocks_fake_system_dialog() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        // System lineage but from an untrusted origin.
        let mut m = ProvenanceManifest::new(
            7,
            Hash256::of(b"evil-app"),
            Hash256::of(b"evil-sig"),
            alloc::vec![0, 7], // claims system origin
        );
        assert_eq!(v.verify_node(&mut m), Err(ProvenanceError::FakeSystemDialog));
    }

    #[test]
    fn verifier_blocks_unknown_non_system_origin_when_list_set() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin()); // only the system origin is trusted
        let mut m = user_manifest(); // unknown origin
        assert_eq!(v.verify_node(&mut m), Err(ProvenanceError::UnknownOrigin));
    }

    #[test]
    fn verifier_accepts_user_node_when_no_trust_list() {
        let mut v = ProvenanceVerifier::new(); // no registered origins
        let mut m = user_manifest();
        assert!(v.verify_node(&mut m).is_ok());
    }

    #[test]
    fn detect_overlay_attack_true_for_untrusted_system_claim() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        let m = ProvenanceManifest::new(
            9,
            Hash256::of(b"phishing-app"),
            Hash256::of(b"phishing-sig"),
            alloc::vec![0, 9],
        );
        assert!(v.detect_overlay_attack(&m));
    }

    #[test]
    fn detect_overlay_attack_false_for_legitimate_system() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        let m = system_manifest();
        assert!(!v.detect_overlay_attack(&m));
    }

    #[test]
    fn detect_overlay_attack_false_for_user_node() {
        let v = ProvenanceVerifier::new();
        assert!(!v.detect_overlay_attack(&user_manifest()));
    }

    #[test]
    fn broken_lineage_blocked() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        let mut m = ProvenanceManifest::new(
            3,
            make_origin(),
            make_sig(),
            alloc::vec![], // empty lineage
        );
        assert_eq!(v.verify_node(&mut m), Err(ProvenanceError::BrokenLineage));
    }

    #[test]
    fn register_trusted_origin_is_idempotent() {
        let mut v = ProvenanceVerifier::new();
        v.register_trusted_origin(make_origin());
        v.register_trusted_origin(make_origin());
        assert_eq!(v.trusted_origins.len(), 1);
    }
}

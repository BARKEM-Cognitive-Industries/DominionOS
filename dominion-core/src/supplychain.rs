//! Supply-chain hardening — **SBOM as content-addressed objects + reproducible-build
//! provenance** (`docs/security/threat-model-and-security-posture.md`,
//! `docs/implementation/update-and-upgrade-lifecycle.md`).
//!
//! A capability OS that ships signed updates still has to answer *"is the binary I run the
//! one built from the audited source?"*. This module makes that checkable:
//!
//! * A [`Sbom`] (software bill of materials) is a list of [`Component`]s, each pinned by
//!   the **content hash of its source** — and the SBOM itself is **content-addressed**, so
//!   it is a first-class object in the graph (cacheable, dedup'd, referenced by id).
//! * [`BuildProvenance`] links `source root → SBOM → artifact hash` and is **signed**, so a
//!   verifier confirms the published artifact was produced from exactly that source + SBOM.
//! * [`verify_reproducible`] checks the **reproducible-build** property: building the same
//!   source twice yields the identical artifact hash (a deterministic builder), which is
//!   what lets independent rebuilders attest the same provenance.
//!
//! Pure, safe `no_std`; PQ (hash-based signing). Host-tested.

use crate::crypto::{LamportSig, SignatureScheme};
use crate::hash::Hash256;
use alloc::string::String;
use alloc::vec::Vec;

/// One component of the build, pinned by the content hash of its source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub name: String,
    pub version: String,
    pub source_hash: Hash256,
}

impl Component {
    pub fn new(name: &str, version: &str, source: &[u8]) -> Component {
        Component { name: String::from(name), version: String::from(version), source_hash: Hash256::of(source) }
    }
}

/// A software bill of materials: the set of components, content-addressed as a whole.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sbom {
    components: Vec<Component>,
}

impl Sbom {
    pub fn new() -> Sbom {
        Sbom { components: Vec::new() }
    }

    pub fn add(&mut self, component: Component) {
        self.components.push(component);
    }

    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// Canonical bytes (components sorted by name+version, so the id is order-independent).
    fn canonical(&self) -> Vec<u8> {
        let mut sorted = self.components.clone();
        sorted.sort_by(|a, b| (a.name.as_str(), a.version.as_str()).cmp(&(b.name.as_str(), b.version.as_str())));
        let mut b = Vec::new();
        b.extend_from_slice(b"sbom:v1:");
        for c in &sorted {
            b.extend_from_slice(c.name.as_bytes());
            b.push(0);
            b.extend_from_slice(c.version.as_bytes());
            b.push(0);
            b.extend_from_slice(&c.source_hash.0);
        }
        b
    }

    /// The content address of this SBOM — its identity as a graph object.
    pub fn id(&self) -> Hash256 {
        Hash256::of(&self.canonical())
    }
}

/// A signed link from `(source root, SBOM)` to a produced artifact.
#[derive(Clone, Debug)]
pub struct BuildProvenance {
    pub source_root: Hash256,
    pub sbom_id: Hash256,
    pub artifact_hash: Hash256,
    public: Vec<u8>,
    signature: Vec<u8>,
}

/// Constant-time byte-slice equality — prevents timing side-channels when comparing keys.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn provenance_signer() -> LamportSig {
    LamportSig::new("build-provenance", "supply-chain-pq")
}

fn provenance_message(source_root: &Hash256, sbom_id: &Hash256, artifact: &Hash256) -> Vec<u8> {
    let mut b = Vec::with_capacity(96 + 12);
    b.extend_from_slice(b"prov:");
    b.extend_from_slice(&source_root.0);
    b.extend_from_slice(&sbom_id.0);
    b.extend_from_slice(&artifact.0);
    b
}

impl BuildProvenance {
    /// Return the public key embedded in this provenance record.
    pub fn public_key(&self) -> &[u8] {
        &self.public
    }

    /// Sign a provenance statement with the builder's key.
    pub fn sign(source_root: Hash256, sbom: &Sbom, artifact_hash: Hash256, secret_seed: &[u8]) -> BuildProvenance {
        let sbom_id = sbom.id();
        let signer = provenance_signer();
        let (secret, public) = signer.keygen(secret_seed);
        let msg = provenance_message(&source_root, &sbom_id, &artifact_hash);
        let signature = signer.sign(&secret, &msg);
        BuildProvenance { source_root, sbom_id, artifact_hash, public, signature }
    }

    /// Verify the signature binds this exact `(source root, SBOM, artifact)` AND that the
    /// provenance was signed by the key the caller already trusts out-of-band.
    ///
    /// `trusted_builder_key` must be the expected builder public key fetched from a
    /// trust store that the verifier controls — NOT from the provenance itself. This
    /// prevents an attacker from embedding their own key inside a self-consistent
    /// (but untrusted) provenance and having it pass verification.
    pub fn verify(&self, sbom: &Sbom, trusted_builder_key: &[u8]) -> bool {
        // 1. Reject any provenance whose embedded key differs from the trusted key.
        //    Constant-time byte comparison to avoid timing side-channels.
        if !constant_time_eq(self.public.as_slice(), trusted_builder_key) {
            return false;
        }
        // 2. Reject if the SBOM presented to the verifier doesn't match what was signed.
        if sbom.id() != self.sbom_id {
            return false;
        }
        // 3. Verify the cryptographic signature.
        let msg = provenance_message(&self.source_root, &self.sbom_id, &self.artifact_hash);
        provenance_signer().verify(&self.public, &msg, &self.signature)
    }

    /// The builder's public key (the producer identity).
    pub fn builder(&self) -> &[u8] {
        &self.public
    }
}

/// Verify the **reproducible-build** property of a deterministic builder: building the
/// same `source` twice yields the same artifact hash (so an independent rebuilder gets
/// the bit-identical artifact the provenance claims).
pub fn verify_reproducible<F: Fn(&[u8]) -> Vec<u8>>(source: &[u8], build: F) -> bool {
    Hash256::of(&build(source)) == Hash256::of(&build(source))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sbom() -> Sbom {
        let mut s = Sbom::new();
        s.add(Component::new("dominion-core", "0.1.0", b"core-source"));
        s.add(Component::new("dominion-kernel", "0.1.0", b"kernel-source"));
        s
    }

    #[test]
    fn sbom_is_content_addressed_and_order_independent() {
        let a = sample_sbom();
        let mut b = Sbom::new();
        b.add(Component::new("dominion-kernel", "0.1.0", b"kernel-source"));
        b.add(Component::new("dominion-core", "0.1.0", b"core-source"));
        // Same components in a different order ⇒ same content id.
        assert_eq!(a.id(), b.id());
        // A changed source hash ⇒ a different id.
        let mut c = Sbom::new();
        c.add(Component::new("dominion-core", "0.1.0", b"TAMPERED-source"));
        assert_ne!(a.id(), c.id());
    }

    #[test]
    fn provenance_binds_source_sbom_and_artifact() {
        let sbom = sample_sbom();
        let source_root = Hash256::of(b"the-whole-source-tree");
        let artifact = Hash256::of(b"the-built-binary");
        let seed = b"builder-key";
        let prov = BuildProvenance::sign(source_root, &sbom, artifact, seed);
        // Derive the trusted public key the same way sign() does, to simulate a
        // verifier fetching the key from an out-of-band trust store.
        let trusted_key = prov.builder().to_vec();
        assert!(prov.verify(&sbom, &trusted_key));
        // A different SBOM than the one signed must fail verification.
        let mut other = sbom.clone();
        other.add(Component::new("evil-dep", "6.6.6", b"malware"));
        assert!(!prov.verify(&other, &trusted_key));
    }

    #[test]
    fn reproducible_build_is_deterministic() {
        // A deterministic "builder": artifact = source with a fixed transform.
        let build = |src: &[u8]| {
            let mut out = Vec::from(b"ARTIFACT:".as_ref());
            out.extend_from_slice(src);
            out
        };
        assert!(verify_reproducible(b"source-v1", build));
        // The artifact the provenance should reference.
        let artifact = Hash256::of(&build(b"source-v1"));
        let sbom = sample_sbom();
        let prov = BuildProvenance::sign(Hash256::of(b"source-v1"), &sbom, artifact, b"k");
        let trusted_key = prov.builder().to_vec();
        assert!(prov.verify(&sbom, &trusted_key));
    }

    /// Regression test for the trust-anchor bug: an attacker who re-signs the same content
    /// with a different key must NOT pass verification against the original key.
    #[test]
    fn verify_rejects_provenance_signed_by_untrusted_key() {
        let sbom = sample_sbom();
        let source_root = Hash256::of(b"the-whole-source-tree");
        let artifact = Hash256::of(b"the-built-binary");

        // Key A — the legitimate builder.
        let prov_a = BuildProvenance::sign(source_root.clone(), &sbom, artifact.clone(), b"key-A-seed");
        let trusted_key_a = prov_a.builder().to_vec();

        // Key B — an attacker re-signs the identical content with their own seed.
        let prov_b = BuildProvenance::sign(source_root, &sbom, artifact, b"key-B-attacker-seed");

        // prov_b is internally self-consistent, but its embedded key is NOT key A.
        // verify() against key A must return false.
        assert!(!prov_b.verify(&sbom, &trusted_key_a),
            "verify() must reject a provenance signed by an untrusted key");

        // Sanity: prov_a still verifies against key A.
        assert!(prov_a.verify(&sbom, &trusted_key_a));
    }
}

//! Verified firmware & secure boot — **Stage 0** (see
//! `docs/architecture/01-stage-00-firmware-boot.md`).
//!
//! Secure boot is a **chain of trust**: control passes to the next stage only after
//! its code is **measured** (extended into the boot log) *and* its signature
//! **verifies against the key the previous stage vouched for**. Stage 0 (the
//! anchor) is rooted in hardware; each stage signs the *next* stage's verification
//! key, so a single trusted root inductively authenticates the whole chain. A
//! tampered or unsigned stage **halts the boot** — there is no "boot into a modified
//! image". Combined with **reproducible images** ([`image_matches`]), a verifier can
//! prove the *running* code equals the *auditable source*.
//!
//! Built over the post-quantum [`CryptoLayer`](crate::crypto); pure, safe, host-tested.

use crate::crypto::CryptoLayer;
use crate::hash::Hash256;
use alloc::string::String;
use alloc::vec::Vec;

/// A boot stage presented to the chain: its code, the verification key it vouches
/// for the *next* stage with, and the signature (made by the *previous* stage's key)
/// over `H(code) ‖ next_key`.
pub struct BootStage {
    pub name: String,
    pub code: Vec<u8>,
    pub next_key: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Why the boot chain refused to continue.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BootError {
    /// The stage's signature did not verify against the current trust anchor.
    Untrusted(String),
}

/// The signed message a stage's signature covers: the code measurement bound to the
/// key it endorses for the next stage.
fn stage_message(code: &[u8], next_key: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(32 + next_key.len());
    m.extend_from_slice(&Hash256::of(code).0);
    m.extend_from_slice(next_key);
    m
}

/// An in-progress measured + verified boot.
pub struct BootChain<'a> {
    cal: &'a CryptoLayer,
    algo: String,
    /// The verification key currently trusted (starts at the hardware anchor).
    trust_key: Vec<u8>,
    /// The boot measurement log (one entry per accepted stage).
    measurements: Vec<Hash256>,
}

impl<'a> BootChain<'a> {
    /// Begin a boot rooted in `anchor_key` (the hardware root of trust's public key).
    pub fn new(cal: &'a CryptoLayer, algo: &str, anchor_key: &[u8]) -> BootChain<'a> {
        BootChain {
            cal,
            algo: String::from(algo),
            trust_key: anchor_key.to_vec(),
            measurements: Vec::new(),
        }
    }

    /// Verify and load the next `stage`. Control transfers (the stage is measured and
    /// its endorsed key becomes the new trust anchor) only if its signature chains to
    /// the current anchor. A tampered/unsigned stage returns [`BootError::Untrusted`]
    /// and the chain does **not** advance.
    pub fn load(&mut self, stage: &BootStage) -> Result<(), BootError> {
        let msg = stage_message(&stage.code, &stage.next_key);
        if !self.cal.verify(&self.algo, &self.trust_key, &msg, &stage.signature) {
            return Err(BootError::Untrusted(stage.name.clone()));
        }
        self.measurements.push(Hash256::of(&stage.code));
        self.trust_key = stage.next_key.clone();
        Ok(())
    }

    /// The composite boot measurement — the fingerprint of exactly what booted.
    pub fn measurement(&self) -> Hash256 {
        let mut input = Vec::with_capacity(32 * self.measurements.len());
        for m in &self.measurements {
            input.extend_from_slice(&m.0);
        }
        Hash256::of(&input)
    }

    pub fn stages_booted(&self) -> usize {
        self.measurements.len()
    }
}

/// Reproducible-image check: the loaded `code` must hash to the independently
/// published `expected` digest — proof the running code equals the auditable source.
pub fn image_matches(code: &[u8], expected: Hash256) -> bool {
    Hash256::of(code) == expected
}

/// A signer used to *build* a trusted boot chain (the offline release process). Each
/// stage signs the next stage's key with its own secret.
pub struct StageSigner;

impl StageSigner {
    /// Produce a [`BootStage`] for `code` that endorses `next_key`, signed by
    /// `prev_secret` through the CAL.
    pub fn sign(
        cal: &CryptoLayer,
        algo: &str,
        name: &str,
        code: &[u8],
        next_key: &[u8],
        prev_secret: &[u8],
    ) -> Option<BootStage> {
        let msg = stage_message(code, next_key);
        let signature = cal.sign(algo, prev_secret, &msg)?;
        Some(BootStage {
            name: String::from(name),
            code: code.to_vec(),
            next_key: next_key.to_vec(),
            signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALGO: &str = "lamport-pq";

    /// Build a 3-stage signed chain: anchor → firmware → bootloader → kernel.
    fn build_chain(cal: &CryptoLayer) -> (Vec<u8>, [BootStage; 3]) {
        // Each stage has a keypair; a stage's code is signed by the previous key.
        let (anchor_sk, anchor_pk) = cal.keygen(ALGO, b"anchor").unwrap();
        let (fw_sk, fw_pk) = cal.keygen(ALGO, b"firmware").unwrap();
        let (bl_sk, bl_pk) = cal.keygen(ALGO, b"bootloader").unwrap();
        let (_k_sk, k_pk) = cal.keygen(ALGO, b"kernel").unwrap();

        let firmware = StageSigner::sign(cal, ALGO, "firmware", b"FIRMWARE-CODE", &fw_pk, &anchor_sk).unwrap();
        let bootloader = StageSigner::sign(cal, ALGO, "bootloader", b"BOOTLOADER-CODE", &bl_pk, &fw_sk).unwrap();
        let kernel = StageSigner::sign(cal, ALGO, "kernel", b"KERNEL-CODE", &k_pk, &bl_sk).unwrap();
        (anchor_pk, [firmware, bootloader, kernel])
    }

    #[test]
    fn a_correctly_signed_chain_boots() {
        let cal = CryptoLayer::with_defaults();
        let (anchor, stages) = build_chain(&cal);
        let mut chain = BootChain::new(&cal, ALGO, &anchor);
        for s in &stages {
            chain.load(s).expect("each signed stage must load");
        }
        assert_eq!(chain.stages_booted(), 3);
        assert_ne!(chain.measurement(), Hash256::ZERO);
    }

    #[test]
    fn a_tampered_stage_halts_the_boot() {
        let cal = CryptoLayer::with_defaults();
        let (anchor, mut stages) = build_chain(&cal);
        // An attacker swaps the bootloader code after it was signed.
        stages[1].code = b"BOOTLOADER-TROJAN".to_vec();
        let mut chain = BootChain::new(&cal, ALGO, &anchor);
        chain.load(&stages[0]).unwrap(); // firmware ok
        // The tampered bootloader fails verification — boot halts here.
        assert_eq!(chain.load(&stages[1]), Err(BootError::Untrusted("bootloader".into())));
        assert_eq!(chain.stages_booted(), 1);
    }

    #[test]
    fn an_unanchored_stage_is_rejected() {
        let cal = CryptoLayer::with_defaults();
        let (_anchor, stages) = build_chain(&cal);
        // Boot with the WRONG anchor key — even a validly-signed first stage fails.
        let (_sk, wrong_anchor) = cal.keygen(ALGO, b"attacker-root").unwrap();
        let mut chain = BootChain::new(&cal, ALGO, &wrong_anchor);
        assert!(chain.load(&stages[0]).is_err());
    }

    #[test]
    fn reproducible_image_proves_running_code_equals_source() {
        let published = Hash256::of(b"KERNEL-CODE");
        assert!(image_matches(b"KERNEL-CODE", published));
        assert!(!image_matches(b"KERNEL-CODE-modified", published));
    }
}

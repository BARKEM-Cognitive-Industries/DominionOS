//! Hardware Root of Trust & attestation — **AH** (see
//! `docs/security/hardware-root-of-trust-and-attestation.md`).
//!
//! A **Root-of-Trust Abstraction Layer (RTAL)** presents one interface over many
//! backends — TPM 2.0, firmware TPM, a secure element, DICE, or a pure-software
//! fallback — so the rest of the OS asks for *measured boot*, *sealing*, and
//! *attestation* without caring which silicon answers. The attestation always
//! reports the **enforcement tier** it actually ran at, so a peer can require a
//! minimum posture and a software fallback is honestly labelled as weaker.
//!
//! Three capabilities are modelled here in safe Rust:
//!
//! * **Measured boot → PCRs.** Each stage extends a Platform Configuration Register
//!   (`PCR ← H(PCR ‖ measurement)`), so the register set is a tamper-evident digest
//!   of exactly what booted.
//! * **Sealing to platform state.** A secret is sealed against the current PCR
//!   digest; if the machine boots differently, it **cannot unseal** — the device key
//!   and DRNG seed are bound to a known-good state.
//! * **Remote attestation.** A `quote` binds the PCR digest + tier + a verifier's
//!   **freshness nonce** under the device attestation key; a peer verifies it and
//!   checks the tier meets its policy.
//!
//! Pure, safe `no_std`, host-tested. Real backends drop in behind [`Backend`].

use crate::chacha::{aead_decrypt, aead_encrypt};
use crate::hash::Hash256;
use alloc::vec::Vec;

/// The available roots of trust, strongest first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Tpm20,
    FirmwareTpm,
    SecureElement,
    Dice,
    /// A hardware security module — high-assurance key custody for the Infrastructure
    /// domain (keys are generated in and never leave the HSM).
    Hsm,
    /// No hardware root — attested honestly as the weakest tier.
    Software,
}

/// The enforcement tier an attestation reports.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// Software-only root of trust (weakest).
    Software = 0,
    /// Firmware / DICE root.
    Firmware = 1,
    /// Discrete hardware root (TPM / secure element).
    Hardware = 2,
}

impl Backend {
    pub fn tier(self) -> Tier {
        match self {
            Backend::Tpm20 | Backend::SecureElement | Backend::Hsm => Tier::Hardware,
            Backend::FirmwareTpm | Backend::Dice => Tier::Firmware,
            Backend::Software => Tier::Software,
        }
    }
}

/// Number of Platform Configuration Registers.
pub const PCR_COUNT: usize = 8;

/// A signed attestation quote: what booted (`pcr_digest`), at what posture (`tier`),
/// bound to the verifier's `nonce`, authenticated under the device attestation key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attestation {
    pub pcr_digest: Hash256,
    pub tier: Tier,
    pub nonce: Vec<u8>,
    mac: Hash256,
}

/// A secret sealed to the platform: opaque unless the machine is in the same
/// measured state it was sealed in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedBlob {
    /// Per-seal 96-bit nonce — derived from H(attest_key || secret), unique per
    /// distinct secret so no two seals share a keystream.
    nonce: [u8; 12],
    ciphertext: Vec<u8>,
    /// Poly1305 authentication tag produced by ChaCha20-Poly1305.
    tag: [u8; 16],
}

/// The root of trust: a backend, the PCR bank, and the device attestation key.
pub struct RootOfTrust {
    backend: Backend,
    pcrs: [Hash256; PCR_COUNT],
    /// The device attestation key — in hardware this never leaves the chip. A
    /// verifier is enrolled with it out of band (symmetric attestation model).
    attest_key: [u8; 32],
    /// A hardware-backed monotonic counter that only ever increases (anti-rollback).
    monotonic: u64,
}

impl RootOfTrust {
    /// Bring up a root of trust on `backend`, with a device seed (from the chip /
    /// entropy boundary) for the attestation key.
    pub fn new(backend: Backend, device_seed: &[u8]) -> RootOfTrust {
        RootOfTrust {
            backend,
            pcrs: [Hash256::ZERO; PCR_COUNT],
            attest_key: Hash256::of(&[device_seed, b":attest"].concat()).0,
            monotonic: 0,
        }
    }

    /// The current value of the RoT monotonic counter.
    pub fn counter(&self) -> u64 {
        self.monotonic
    }

    /// Advance the monotonic counter (e.g. on each update activation or time tick) and
    /// return the new value. It can only ever increase — the anti-rollback primitive that
    /// update versioning and "no boot into the past" bind to.
    pub fn tick_counter(&mut self) -> u64 {
        self.monotonic = self.monotonic.saturating_add(1);
        self.monotonic
    }

    /// Bind an anti-rollback lower bound: accept `value` only if it does not move the
    /// counter backwards (a rollback attempt is refused). Returns whether it was accepted.
    pub fn bind_counter(&mut self, value: u64) -> bool {
        if value >= self.monotonic {
            self.monotonic = value;
            true
        } else {
            false // rollback attempt
        }
    }

    pub fn tier(&self) -> Tier {
        self.backend.tier()
    }

    /// Extend a PCR with a boot-stage measurement (`PCR ← H(PCR ‖ measurement)`).
    pub fn extend(&mut self, index: usize, measurement: &[u8]) {
        if index >= PCR_COUNT {
            return;
        }
        let mut input = Vec::with_capacity(32 + measurement.len());
        input.extend_from_slice(&self.pcrs[index].0);
        input.extend_from_slice(measurement);
        self.pcrs[index] = Hash256::of(&input);
    }

    /// The composite digest over all PCRs — the fingerprint of the boot state.
    pub fn pcr_digest(&self) -> Hash256 {
        let mut input = Vec::with_capacity(32 * PCR_COUNT);
        for p in &self.pcrs {
            input.extend_from_slice(&p.0);
        }
        Hash256::of(&input)
    }

    /// Produce a remote-attestation quote bound to the verifier's `nonce`.
    pub fn quote(&self, nonce: &[u8]) -> Attestation {
        let pcr_digest = self.pcr_digest();
        let tier = self.tier();
        let mac = self.attest_mac(&pcr_digest, tier, nonce);
        Attestation { pcr_digest, tier, nonce: nonce.to_vec(), mac }
    }

    fn attest_mac(&self, pcr_digest: &Hash256, tier: Tier, nonce: &[u8]) -> Hash256 {
        let mut input = Vec::with_capacity(64 + nonce.len());
        input.extend_from_slice(&self.attest_key);
        input.extend_from_slice(&pcr_digest.0);
        input.push(tier as u8);
        input.extend_from_slice(nonce);
        Hash256::of(&input)
    }

    /// The key sealing binds to: the device key folded with the **current** PCR
    /// digest, so a different boot state yields a different (useless) key.
    fn seal_key(&self) -> [u8; 32] {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(&self.attest_key);
        input.extend_from_slice(&self.pcr_digest().0);
        Hash256::of(&input).0
    }

    /// Derive a per-secret nonce: H(attest_key || secret)[..12].
    ///
    /// Because the nonce is bound to the secret content, two different secrets
    /// sealed under the same platform state receive distinct nonces and therefore
    /// distinct ChaCha20 keystreams — the many-time-pad vulnerability is closed.
    fn seal_nonce(&self, secret: &[u8]) -> [u8; 12] {
        let mut input = Vec::with_capacity(32 + secret.len());
        input.extend_from_slice(&self.attest_key);
        input.extend_from_slice(secret);
        let h = Hash256::of(&input).0;
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&h[..12]);
        nonce
    }

    /// Seal a secret to the current platform state using ChaCha20-Poly1305 AEAD.
    /// A per-secret nonce derived from H(attest_key || secret) ensures distinct
    /// keystreams even when the same PCR digest is active, eliminating the
    /// many-time-pad attack that the old SHA-256-CTR scheme was vulnerable to.
    pub fn seal(&self, secret: &[u8]) -> SealedBlob {
        let key = self.seal_key();
        let nonce = self.seal_nonce(secret);
        // The PCR digest is included as AAD: decryption fails if the platform
        // state changed, providing the same boot-binding as before.
        let aad = self.pcr_digest().0;
        let (ciphertext, tag) = aead_encrypt(&key, &nonce, &aad, secret);
        SealedBlob { nonce, ciphertext, tag }
    }

    /// Unseal — only succeeds if the platform is in the same measured state it was
    /// sealed in (PCRs unchanged). A tampered boot cannot recover the secret.
    pub fn unseal(&self, blob: &SealedBlob) -> Option<Vec<u8>> {
        let key = self.seal_key();
        let aad = self.pcr_digest().0;
        aead_decrypt(&key, &blob.nonce, &aad, &blob.ciphertext, &blob.tag)
    }

    /// The attestation key a verifier is enrolled with (out of band).
    pub fn enrollment_key(&self) -> [u8; 32] {
        self.attest_key
    }
}

/// Verify an attestation against an enrolled device key: the MAC must check, the
/// `nonce` must match (freshness — no replay), and the reported `tier` must meet the
/// verifier's minimum policy.
pub fn verify_attestation(
    att: &Attestation,
    enrollment_key: &[u8; 32],
    expected_nonce: &[u8],
    min_tier: Tier,
) -> bool {
    if att.nonce != expected_nonce || att.tier < min_tier {
        return false;
    }
    let mut input = Vec::with_capacity(64 + att.nonce.len());
    input.extend_from_slice(enrollment_key);
    input.extend_from_slice(&att.pcr_digest.0);
    input.push(att.tier as u8);
    input.extend_from_slice(&att.nonce);
    Hash256::of(&input) == att.mac
}

#[cfg(test)]
mod tests {
    use super::*;

    fn booted(backend: Backend) -> RootOfTrust {
        let mut rot = RootOfTrust::new(backend, b"device-seed");
        rot.extend(0, b"firmware-v1");
        rot.extend(0, b"bootloader-v1");
        rot.extend(1, b"kernel-v1");
        rot
    }

    #[test]
    fn measured_boot_is_order_sensitive() {
        let a = booted(Backend::Tpm20).pcr_digest();
        // A different boot (kernel swapped) yields a different digest.
        let mut tampered = RootOfTrust::new(Backend::Tpm20, b"device-seed");
        tampered.extend(0, b"firmware-v1");
        tampered.extend(0, b"bootloader-v1");
        tampered.extend(1, b"kernel-TROJAN");
        assert_ne!(a, tampered.pcr_digest());
    }

    #[test]
    fn hsm_is_a_hardware_tier_and_the_counter_is_anti_rollback() {
        // The HSM backend (Infrastructure-domain key custody) attests at the hardware tier.
        assert_eq!(Backend::Hsm.tier(), Tier::Hardware);
        let mut rot = RootOfTrust::new(Backend::Hsm, b"infra-seed");
        assert_eq!(rot.counter(), 0);
        assert_eq!(rot.tick_counter(), 1);
        assert_eq!(rot.tick_counter(), 2);
        // Binding a higher value advances; a lower value (rollback) is refused.
        assert!(rot.bind_counter(10));
        assert_eq!(rot.counter(), 10);
        assert!(!rot.bind_counter(5)); // anti-rollback
        assert_eq!(rot.counter(), 10);
    }

    #[test]
    fn attestation_verifies_with_tier_and_freshness() {
        let rot = booted(Backend::Tpm20);
        let key = rot.enrollment_key();
        let att = rot.quote(b"verifier-nonce-42");
        // Correct nonce + meets the hardware-tier requirement.
        assert!(verify_attestation(&att, &key, b"verifier-nonce-42", Tier::Hardware));
        // A replay with a stale nonce is rejected (freshness).
        assert!(!verify_attestation(&att, &key, b"old-nonce", Tier::Hardware));
    }

    #[test]
    fn software_root_cannot_meet_a_hardware_policy() {
        let rot = booted(Backend::Software);
        let key = rot.enrollment_key();
        let att = rot.quote(b"n");
        assert_eq!(att.tier, Tier::Software);
        // A peer demanding a hardware root refuses the software attestation …
        assert!(!verify_attestation(&att, &key, b"n", Tier::Hardware));
        // … but a software-tier policy accepts it (honestly labelled).
        assert!(verify_attestation(&att, &key, b"n", Tier::Software));
    }

    #[test]
    fn sealing_binds_a_secret_to_the_boot_state() {
        let rot = booted(Backend::Tpm20);
        let blob = rot.seal(b"device DRNG seed");
        // Same platform state → unseals.
        assert_eq!(rot.unseal(&blob).as_deref(), Some(b"device DRNG seed".as_ref()));
        // A machine that booted differently cannot unseal it.
        let other = {
            let mut r = RootOfTrust::new(Backend::Tpm20, b"device-seed");
            r.extend(0, b"firmware-v2-EVIL");
            r
        };
        assert!(other.unseal(&blob).is_none());
    }

    #[test]
    fn forged_attestation_is_rejected() {
        let rot = booted(Backend::Tpm20);
        let att = rot.quote(b"n");
        // An attacker without the enrolled device key cannot forge a valid quote.
        let wrong_key = [0xAAu8; 32];
        assert!(!verify_attestation(&att, &wrong_key, b"n", Tier::Software));
    }

    /// Two different secrets sealed under the same platform state must produce
    /// different ciphertexts — the regression guard for the old SHA-256-CTR
    /// many-time-pad vulnerability.  Each seal derives a distinct nonce from
    /// H(attest_key || secret), so the ChaCha20 keystreams never coincide.
    #[test]
    fn distinct_secrets_produce_distinct_ciphertexts() {
        let rot = booted(Backend::Tpm20);
        let blob_a = rot.seal(b"secret-alpha");
        let blob_b = rot.seal(b"secret-beta");
        // Nonces must differ — same nonce with the same key is the attack.
        assert_ne!(blob_a.nonce, blob_b.nonce);
        // Ciphertexts must differ (implied by distinct nonces, but assert explicitly).
        assert_ne!(blob_a.ciphertext, blob_b.ciphertext);
        // Both must still round-trip correctly on the same platform.
        assert_eq!(rot.unseal(&blob_a).as_deref(), Some(b"secret-alpha".as_ref()));
        assert_eq!(rot.unseal(&blob_b).as_deref(), Some(b"secret-beta".as_ref()));
    }
}

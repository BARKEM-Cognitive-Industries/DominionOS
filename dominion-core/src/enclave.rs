//! Confidential compute — **enclaves / CVMs behind a hardware-abstraction layer**
//! (Stage 14.9; `docs/security/stage-14-universal-encryption.md`,
//! `docs/security/system-domain-and-internal-confidentiality.md`).
//!
//! A confidential VM lets a cell run **with its code and data encrypted in memory**, so
//! even a compromised host (or a hostile cloud operator) sees only ciphertext, and a
//! remote party can **attest** that the expected code ran on a genuine enclave. The real
//! hardware mechanisms — Intel TDX, AMD SEV-SNP, Arm CCA, AWS Nitro — are modeled here
//! behind a [`EnclaveBackend`] HAL that **degrades gracefully** to a software backend on
//! commodity hardware (the same "accelerator, not requirement" rule the rest of the OS
//! follows). The confidentiality on a software backend is honestly weaker — and the
//! attestation says so (`Tier::Software`) — but the *semantics* are exercised end-to-end:
//!
//! * code + data live **sealed** in RAM ([`crate::memcrypt::SealedRegion`]); the host's
//!   view ([`ConfidentialVm::data_at_rest`]) is ciphertext;
//! * plaintext materialises **only transiently** inside [`ConfidentialVm::execute`], then
//!   is re-sealed — minimal plaintext lifetime;
//! * the result is **capability-gated** — only a holder of a read capability can open it;
//! * a [`ConfidentialVm::attest`] quote binds the **code measurement** + the enclave tier
//!   to a verifier nonce, so a remote party verifies *what ran* and *how strong the
//!   isolation is* without trusting the host ([`crate::rot`]).
//!
//! Complementary to ZK verifiable computation ([`crate::vcompute`]): a TEE asks you to
//! *trust the hardware kept it secret*; ZK *proves the result correct to someone who
//! trusts neither*. Pick per workload. Pure, safe `no_std`; deterministic.

use crate::capability::{Capability, Rights};
use crate::hash::Hash256;
use crate::memcrypt::{salt_from_label, SealedRegion};
use crate::rot::{verify_attestation, Attestation, Backend, RootOfTrust, Tier};
use alloc::vec::Vec;

/// The confidential-compute mechanism in use — a HAL over the real TEEs, with a software
/// fallback. Selected from attested hardware features at launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnclaveBackend {
    /// Intel TDX trust domain.
    IntelTdx,
    /// AMD SEV-SNP confidential VM.
    AmdSevSnp,
    /// Arm CCA realm.
    ArmCca,
    /// AWS Nitro enclave.
    AwsNitro,
    /// Software-isolated fallback (capability confinement + software memory sealing).
    Software,
}

/// Confidential-compute features a platform advertises (attested at boot). All false in
/// this build — exercising the software-degraded path on purpose.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConfidentialFeatures {
    pub intel_tdx: bool,
    pub amd_sev_snp: bool,
    pub arm_cca: bool,
    pub aws_nitro: bool,
}

impl EnclaveBackend {
    /// Pick the strongest available backend, degrading to [`EnclaveBackend::Software`].
    pub fn select(f: ConfidentialFeatures) -> EnclaveBackend {
        if f.intel_tdx {
            EnclaveBackend::IntelTdx
        } else if f.amd_sev_snp {
            EnclaveBackend::AmdSevSnp
        } else if f.arm_cca {
            EnclaveBackend::ArmCca
        } else if f.aws_nitro {
            EnclaveBackend::AwsNitro
        } else {
            EnclaveBackend::Software
        }
    }

    /// True for a hardware-backed TEE.
    pub fn is_hardware(self) -> bool {
        self != EnclaveBackend::Software
    }

    /// The attestation tier this backend can honestly claim.
    pub fn tier(self) -> Tier {
        if self.is_hardware() {
            Tier::Hardware
        } else {
            Tier::Software
        }
    }

    /// The honest guarantee statement (impossible vs contained).
    pub fn guarantee(self) -> &'static str {
        if self.is_hardware() {
            "memory confidentiality + integrity enforced by the CPU; host sees only ciphertext"
        } else {
            "confidentiality from software sealing + capability isolation; weaker vs a privileged host"
        }
    }

    /// The RoT backend that anchors this enclave's attestation.
    fn rot_backend(self) -> Backend {
        if self.is_hardware() {
            Backend::Tpm20
        } else {
            Backend::Software
        }
    }
}

/// A confidential VM: sealed code + sealed data + an attestation root.
pub struct ConfidentialVm {
    backend: EnclaveBackend,
    /// The public code measurement (identity) — what a verifier expects.
    measurement: Hash256,
    code: SealedRegion,
    data: SealedRegion,
    rot: RootOfTrust,
}

impl ConfidentialVm {
    /// Launch an enclave: measure the code, seal code+data in memory, and extend the RoT
    /// with the measurement so attestation reflects exactly what was loaded.
    pub fn launch(
        backend: EnclaveBackend,
        code: &[u8],
        initial_data: &[u8],
        device_seed: &[u8],
    ) -> ConfidentialVm {
        let measurement = Hash256::of(code);
        // Memory-encryption key — in a real TEE this is a per-VM key the CPU holds and
        // never exposes; here it is derived from the device seed.
        let mem_key = Hash256::of(&[device_seed, b":enclave-mem"].concat()).0;
        let mut rot = RootOfTrust::new(backend.rot_backend(), device_seed);
        rot.extend(0, &measurement.0); // PCR0 ← code identity
        ConfidentialVm {
            backend,
            measurement,
            code: SealedRegion::seal(mem_key, b"enclave-code", salt_from_label(b"enclave-code"), code),
            data: SealedRegion::seal(mem_key, b"enclave-data", salt_from_label(b"enclave-data"), initial_data),
            rot,
        }
    }

    /// The backend (TEE mechanism) this enclave runs on.
    pub fn backend(&self) -> EnclaveBackend {
        self.backend
    }

    /// The code measurement (public identity) a verifier pins.
    pub fn measurement(&self) -> Hash256 {
        self.measurement
    }

    /// What a privileged host sees of the enclave's data: ciphertext only.
    pub fn data_at_rest(&self) -> &[u8] {
        self.data.at_rest()
    }

    /// What a privileged host sees of the enclave's code: ciphertext only (code
    /// confidentiality for proprietary / hostile-host cells).
    pub fn code_at_rest(&self) -> &[u8] {
        self.code.at_rest()
    }

    /// The synthetic object address gating result reads (derived from the measurement).
    fn realm_addr(&self) -> u64 {
        let h = self.measurement.0;
        let mut a = [0u8; 8];
        a.copy_from_slice(&h[..8]);
        u64::from_le_bytes(a)
    }

    /// Run a pure computation `f` over the enclave's plaintext data **inside** the
    /// enclave: the data is decrypted only for the duration of the call, the result
    /// replaces it (re-sealed), and the transient plaintext is dropped. Models a TEE
    /// confidential computation; returns whether it ran (false iff the data couldn't
    /// be opened — a tampered region).
    pub fn execute<F: Fn(&[u8]) -> Vec<u8>>(&mut self, f: F) -> bool {
        match self.data.open() {
            Some(plaintext) => {
                let result = f(&plaintext);
                self.data.reseal(&result);
                true
            }
            None => false,
        }
    }

    /// Read the (sealed) result — **capability-gated**: only a holder of a `READ`
    /// capability over the enclave's address may decrypt it. Without it, the host gets
    /// nothing but ciphertext.
    pub fn read_result(&self, cap: &Capability) -> Option<Vec<u8>> {
        cap.check(self.realm_addr(), 1, Rights::READ).ok()?;
        self.data.open()
    }

    /// Produce a remote-attestation quote binding the code measurement + the enclave
    /// tier to `nonce`. A verifier checks it with [`verify`](Self::verify).
    pub fn attest(&self, nonce: &[u8]) -> Attestation {
        self.rot.quote(nonce)
    }

    /// The enrollment key a verifier is provisioned with out of band.
    pub fn enrollment_key(&self) -> [u8; 32] {
        self.rot.enrollment_key()
    }

    /// Verify an enclave's attestation: fresh nonce, meets the minimum tier, and the
    /// MAC checks out against the enrollment key. (Standalone so a remote party can call
    /// it without the enclave.)
    pub fn verify(att: &Attestation, enrollment_key: &[u8; 32], nonce: &[u8], min_tier: Tier) -> bool {
        verify_attestation(att, enrollment_key, nonce, min_tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degrades_to_software_then_selects_hardware_when_present() {
        assert_eq!(
            EnclaveBackend::select(ConfidentialFeatures::default()),
            EnclaveBackend::Software
        );
        assert_eq!(
            EnclaveBackend::select(ConfidentialFeatures { amd_sev_snp: true, ..Default::default() }),
            EnclaveBackend::AmdSevSnp
        );
        assert!(EnclaveBackend::IntelTdx.is_hardware());
        assert!(!EnclaveBackend::Software.is_hardware());
    }

    #[test]
    fn host_sees_only_ciphertext_and_result_is_capability_gated() {
        let mut vm = ConfidentialVm::launch(
            EnclaveBackend::Software,
            b"fn double(x) = x*2",
            b"secret-input-7",
            b"device-seed",
        );
        // The host's view of code and data is ciphertext, never the plaintext.
        assert_ne!(vm.data_at_rest(), b"secret-input-7");
        assert_ne!(vm.code_at_rest(), b"fn double(x) = x*2".as_ref());
        // Run a confidential computation: append a tag to the input.
        assert!(vm.execute(|data| {
            let mut out = data.to_vec();
            out.extend_from_slice(b"-processed");
            out
        }));
        // Without a READ capability, the result is unreadable.
        let no_cap = Capability::mint(0, u64::MAX, Rights::WRITE);
        assert!(vm.read_result(&no_cap).is_none());
        // With one, the holder decrypts the result.
        let cap = Capability::mint(0, u64::MAX, Rights::READ);
        assert_eq!(vm.read_result(&cap).unwrap(), b"secret-input-7-processed");
    }

    #[test]
    fn attestation_binds_code_measurement_and_tier() {
        let vm = ConfidentialVm::launch(EnclaveBackend::IntelTdx, b"approved-model-v3", b"x", b"seed");
        let key = vm.enrollment_key();
        let att = vm.attest(b"verifier-nonce");
        // Verifies with the right nonce and a hardware-tier requirement.
        assert!(ConfidentialVm::verify(&att, &key, b"verifier-nonce", Tier::Hardware));
        // A stale nonce (replay) is rejected.
        assert!(!ConfidentialVm::verify(&att, &key, b"old-nonce", Tier::Hardware));
        // The measurement identifies the code that ran.
        assert_eq!(vm.measurement(), Hash256::of(b"approved-model-v3"));
    }

    #[test]
    fn software_backend_cannot_claim_a_hardware_tier() {
        let vm = ConfidentialVm::launch(EnclaveBackend::Software, b"code", b"d", b"seed");
        let key = vm.enrollment_key();
        let att = vm.attest(b"n");
        // A policy demanding hardware refuses the software enclave (honest labelling).
        assert!(!ConfidentialVm::verify(&att, &key, b"n", Tier::Hardware));
        // It does meet the software-tier floor.
        assert!(ConfidentialVm::verify(&att, &key, b"n", Tier::Software));
        assert!(vm.backend().guarantee().contains("weaker"));
    }

    #[test]
    fn different_code_attests_differently() {
        let a = ConfidentialVm::launch(EnclaveBackend::IntelTdx, b"model-A", b"x", b"seed");
        let b = ConfidentialVm::launch(EnclaveBackend::IntelTdx, b"model-B", b"x", b"seed");
        // A tampered/swapped model produces a different measurement → different quote.
        assert_ne!(a.measurement(), b.measurement());
        assert_ne!(a.attest(b"n").pcr_digest, b.attest(b"n").pcr_digest);
    }
}

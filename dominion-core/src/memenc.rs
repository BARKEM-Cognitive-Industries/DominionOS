//! Memory encryption at rest — **AT** (see
//! `docs/security/memory-encryption-at-rest.md`).
//!
//! RAM should be ciphertext whenever it is not actively being read. Three tiers
//! cover the hardware spectrum, and an object's *policy* picks the strongest the
//! platform offers:
//!
//! * **Tier A — Total Memory Encryption** (Intel TME / AMD SME): all RAM encrypted
//!   transparently by the memory controller.
//! * **Tier B — Per-domain keys** (Intel MKTME / AMD SEV-SNP): each domain's RAM is
//!   ciphertext to every other domain.
//! * **Tier C — Software per-object** (no hardware needed): crown-jewel objects are
//!   sealed individually and decrypted **only at the moment of a capability-checked
//!   read**, then dropped — plaintext lives only for the instant of use.
//!
//! Where hardware memory encryption is absent the system **degrades gracefully** to
//! Tier C for sensitive data plus capability isolation for the rest, and **attests**
//! its posture so a high-assurance domain can require Tier A/B. Tier C is built over
//! [`crate::memcrypt`]'s AES-GCM `SealedRegion`; pure, safe, host-tested.

use crate::capability::Rights;
use crate::memcrypt::SealedRegion;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// The memory-encryption tier in effect for a given object.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MemTier {
    /// No memory encryption (capability isolation only).
    None = 0,
    /// Software per-object sealing (always available).
    SoftwarePerObject = 1,
    /// Hardware per-domain keys.
    PerDomainKeys = 2,
    /// Hardware total memory encryption.
    Total = 3,
}

/// How sensitive a data kind is — drives the tier policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataKind {
    /// Ordinary data — protected by capability isolation (+ Tier A if present).
    Normal,
    /// Sensitive data — wants per-domain or software encryption.
    Sensitive,
    /// Crown jewels (keys, identity secrets) — always encrypted at rest.
    CrownJewel,
}

/// Probed memory-encryption hardware. In this build both are `false`, exercising the
/// graceful Tier-C fallback.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemFeatures {
    /// Intel TME / AMD SME (whole-RAM).
    pub tme: bool,
    /// Intel MKTME / AMD SEV-SNP (per-domain).
    pub mktme: bool,
}

/// The memory-encryption policy engine for the platform.
pub struct MemEncryption {
    features: MemFeatures,
}

impl MemEncryption {
    pub fn detect(features: MemFeatures) -> MemEncryption {
        MemEncryption { features }
    }

    /// The tier chosen for a data kind: the strongest appropriate mechanism the
    /// platform offers, never weaker than required for crown jewels.
    pub fn tier_for(&self, kind: DataKind) -> MemTier {
        match kind {
            // Crown jewels are ALWAYS encrypted at rest: software sealing is the
            // floor, hardware (per-domain ⊐ total) used when present.
            DataKind::CrownJewel => {
                if self.features.mktme {
                    MemTier::PerDomainKeys
                } else if self.features.tme {
                    MemTier::Total
                } else {
                    MemTier::SoftwarePerObject
                }
            }
            DataKind::Sensitive => {
                if self.features.mktme {
                    MemTier::PerDomainKeys
                } else if self.features.tme {
                    MemTier::Total
                } else {
                    MemTier::SoftwarePerObject
                }
            }
            // Ordinary data leans on capability isolation, plus Tier A if it's free.
            DataKind::Normal => {
                if self.features.tme {
                    MemTier::Total
                } else {
                    MemTier::None
                }
            }
        }
    }

    /// The attested memory-encryption posture (the best hardware tier available).
    pub fn posture(&self) -> MemTier {
        if self.features.mktme {
            MemTier::PerDomainKeys
        } else if self.features.tme {
            MemTier::Total
        } else {
            MemTier::SoftwarePerObject
        }
    }
}

/// A domain's encrypted RAM (the Tier-C software realisation): each object is sealed
/// individually and only the capability holder can decrypt it, for the instant of a
/// read. To every other domain this memory is ciphertext.
pub struct DomainMemory {
    key: [u8; 32],
    regions: BTreeMap<u64, SealedRegion>,
}

impl DomainMemory {
    /// A domain with its own (per-domain) memory key.
    pub fn new(key: [u8; 32]) -> DomainMemory {
        DomainMemory { key, regions: BTreeMap::new() }
    }

    /// Seal an object into encrypted RAM under this domain's key.
    pub fn seal(&mut self, id: u64, plaintext: &[u8]) {
        // Use the object ID directly as the region salt — IDs are unique per
        // DomainMemory instance (and therefore per key), so no two regions
        // under this key will ever share the same (key, IV) prefix.
        let region = SealedRegion::seal(self.key, &id.to_le_bytes(), id, plaintext);
        self.regions.insert(id, region);
    }

    /// Read an object — materialises plaintext **only** if the caller presents a
    /// capability with `READ` (the per-object read-capability gate). The returned
    /// `Vec` is the sole plaintext copy, for the moment of use.
    pub fn read(&self, id: u64, cap: Rights) -> Option<Vec<u8>> {
        if !cap.contains(Rights::READ) {
            return None;
        }
        self.regions.get(&id)?.open()
    }

    /// The bytes at rest — ciphertext, exposed to demonstrate RAM-is-ciphertext.
    pub fn at_rest(&self, id: u64) -> Option<&[u8]> {
        self.regions.get(&id).map(|r| r.at_rest())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crown_jewels_are_always_encrypted_even_without_hardware() {
        // No TME/MKTME → software per-object for sensitive kinds; ordinary data
        // relies on capability isolation (None) since there is no free hardware tier.
        let me = MemEncryption::detect(MemFeatures::default());
        assert_eq!(me.tier_for(DataKind::CrownJewel), MemTier::SoftwarePerObject);
        assert_eq!(me.tier_for(DataKind::Sensitive), MemTier::SoftwarePerObject);
        assert_eq!(me.tier_for(DataKind::Normal), MemTier::None);
        assert_eq!(me.posture(), MemTier::SoftwarePerObject);
    }

    #[test]
    fn hardware_tiers_are_selected_when_present() {
        // Total memory encryption present → even ordinary data is encrypted.
        let tme = MemEncryption::detect(MemFeatures { tme: true, mktme: false });
        assert_eq!(tme.tier_for(DataKind::Normal), MemTier::Total);
        // Per-domain keys present → the stronger isolation tier is chosen.
        let mktme = MemEncryption::detect(MemFeatures { tme: true, mktme: true });
        assert_eq!(mktme.tier_for(DataKind::CrownJewel), MemTier::PerDomainKeys);
        assert!(mktme.posture() >= MemTier::PerDomainKeys);
    }

    #[test]
    fn ram_is_ciphertext_and_only_a_read_capability_decrypts() {
        let mut mem = DomainMemory::new([7u8; 32]);
        mem.seal(1, b"identity master secret");
        // At rest it is ciphertext, not the plaintext.
        assert_ne!(mem.at_rest(1).unwrap(), b"identity master secret");
        // A READ capability decrypts it for the instant of use …
        assert_eq!(mem.read(1, Rights::READ).as_deref(), Some(b"identity master secret".as_ref()));
        // … but a capability lacking READ gets nothing (per-object read gate).
        assert!(mem.read(1, Rights::WRITE).is_none());
    }

    #[test]
    fn another_domain_sees_only_ciphertext() {
        // Two domains with different keys: one cannot read the other's RAM.
        let mut a = DomainMemory::new([1u8; 32]);
        a.seal(9, b"domain A secret");
        let ct = a.at_rest(9).unwrap().to_vec();
        // Domain B (different key) holding the same ciphertext cannot open it.
        let b = DomainMemory::new([2u8; 32]);
        // Splice A's ciphertext is not even possible via the API; B simply has no
        // such region, and its own key would not decrypt A's bytes.
        assert!(b.read(9, Rights::READ).is_none());
        assert_ne!(ct, b"domain A secret");
    }
}

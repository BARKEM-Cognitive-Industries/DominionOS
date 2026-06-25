//! Capability Enforcement Layer & hardware tiers — **V** (see
//! `docs/implementation/hardware-targets-and-portability.md`).
//!
//! The capability *model* is fixed; the *mechanism* that enforces it varies by
//! hardware. This layer abstracts the mechanism behind a tier:
//!
//! * **Tier 0 — Software.** Rust/Dominion safety + software bounds + MPK/WASM sandbox.
//!   Always available, kept permanently, so commodity and old hardware run.
//! * **Tier 1 — Memory tagging.** ARM MTE + PAC, or Intel MPK — pointer/region
//!   integrity in hardware.
//! * **Tier 2 — CHERI.** Architectural 128-bit capabilities (Morello / RISC-V CHERI)
//!   — the design target.
//!
//! The backend is **selected at boot from attested hardware features** (the strongest
//! the platform actually offers), attestation **reports the tier**, and a domain or
//! the Airlock can **require a minimum tier** — so a high-assurance domain refuses to
//! run on a weaker mechanism than its policy demands, while everything else degrades
//! gracefully to Tier 0. Pure, safe, host-tested.

use crate::cheri::{CapabilityTags, HardwareTags};
use crate::firewall::Domain;
use alloc::collections::BTreeMap;

/// The enforcement mechanism in effect, strongest last.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// Software safety + bounds (always available).
    Software = 0,
    /// Hardware memory tagging (ARM MTE/PAC, Intel MPK).
    MemoryTagging = 1,
    /// Architectural CHERI capabilities.
    Cheri = 2,
}

impl Tier {
    /// The honest guarantee statement for this tier (impossible vs contained).
    pub fn guarantee(self) -> &'static str {
        match self {
            Tier::Software => "violations contained by language + software bounds",
            Tier::MemoryTagging => "spatial/temporal violations trapped by hardware tags",
            Tier::Cheri => "capability violations architecturally impossible",
        }
    }
}

/// Attested hardware features probed at boot. In this build all are `false` (no
/// emulator exposes them), exercising the Tier-0 degraded path on purpose.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardwareFeatures {
    pub cheri_tags: bool,
    pub arm_mte: bool,
    pub arm_pac: bool,
    pub intel_mpk: bool,
}

impl HardwareFeatures {
    /// What this build actually probes (none — the prototype runs Tier 0).
    pub fn probe() -> HardwareFeatures {
        HardwareFeatures::default()
    }
}

/// The selected enforcement backend.
pub struct EnforcementLayer {
    tier: Tier,
    tags: HardwareTags,
}

impl EnforcementLayer {
    /// **Boot-time selection.** Choose the strongest tier the attested `features`
    /// support; Tier 0 is the permanent floor.
    pub fn select(features: HardwareFeatures, software_key: [u8; 32]) -> EnforcementLayer {
        let tier = if features.cheri_tags {
            Tier::Cheri
        } else if features.arm_mte || features.intel_mpk || features.arm_pac {
            Tier::MemoryTagging
        } else {
            Tier::Software
        };
        // The CHERI tag HAL still backs the model in software where the silicon
        // is absent (degrades gracefully, same security contract).
        let tags = HardwareTags::detect(features.cheri_tags, software_key);
        EnforcementLayer { tier, tags }
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn backend_name(&self) -> &'static str {
        self.tags.backend_name()
    }

    /// Whether this layer meets a required minimum tier.
    pub fn meets(&self, min: Tier) -> bool {
        self.tier >= min
    }

    /// The underlying capability-tag mechanism (for minting/validating caps).
    pub fn tags(&self) -> &HardwareTags {
        &self.tags
    }
}

/// Per-domain minimum-tier policy: some domains may only run on a strong-enough
/// enforcement mechanism. The Airlock consults this before admitting a domain.
#[derive(Default)]
pub struct TierPolicy {
    minimums: BTreeMap<Domain, Tier>,
}

impl TierPolicy {
    pub fn new() -> TierPolicy {
        TierPolicy { minimums: BTreeMap::new() }
    }

    /// The canonical Architecture-2.0 per-domain minimum-tier policy (resolves the open
    /// question "which domains require CHERI Tier 2"). The high-assurance domains —
    /// System, Financial, Medical and Infrastructure — require **Tier 2 (CHERI)** where
    /// the hardware provides it; the airlock refuses to admit them on a weaker mechanism
    /// when a Tier-2 platform is mandated. Personal/Development run at **Tier 1** (memory
    /// tagging) when available, and the contained AiAgent / ExternalNetwork domains rely
    /// on capability confinement at **Tier 0**. On commodity hardware every domain still
    /// runs (Tier 0 is permanent); the policy is what a *high-assurance deployment* pins.
    pub fn architecture_2_0() -> TierPolicy {
        let mut p = TierPolicy::new();
        for d in [Domain::System, Domain::Financial, Domain::Medical, Domain::Infrastructure] {
            p.require(d, Tier::Cheri);
        }
        for d in [Domain::Personal, Domain::Development] {
            p.require(d, Tier::MemoryTagging);
        }
        // AiAgent + ExternalNetwork: no minimum tier — contained by capabilities alone.
        p
    }

    /// Require that `domain` only runs at `min` tier or stronger.
    pub fn require(&mut self, domain: Domain, min: Tier) {
        self.minimums.insert(domain, min);
    }

    /// The minimum tier required for `domain` (None ⇒ runs anywhere).
    pub fn minimum(&self, domain: Domain) -> Option<Tier> {
        self.minimums.get(&domain).copied()
    }

    /// Does the current enforcement layer admit `domain`? Domains with no minimum
    /// run anywhere; a high-assurance domain is refused on a weaker mechanism.
    pub fn admits(&self, domain: Domain, layer: &EnforcementLayer) -> bool {
        match self.minimums.get(&domain) {
            Some(min) => layer.meets(*min),
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_the_strongest_available_tier() {
        let key = [1u8; 32];
        // No features → Tier 0 software (the prototype's path).
        let sw = EnforcementLayer::select(HardwareFeatures::default(), key);
        assert_eq!(sw.tier(), Tier::Software);
        assert!(!sw.tags().hardware_backed());

        // MTE present → Tier 1.
        let mte = EnforcementLayer::select(
            HardwareFeatures { arm_mte: true, ..Default::default() },
            key,
        );
        assert_eq!(mte.tier(), Tier::MemoryTagging);

        // CHERI present → Tier 2 (and the tag backend reports hardware).
        let cheri = EnforcementLayer::select(
            HardwareFeatures { cheri_tags: true, ..Default::default() },
            key,
        );
        assert_eq!(cheri.tier(), Tier::Cheri);
        assert!(cheri.tags().hardware_backed());
    }

    #[test]
    fn canonical_policy_gates_high_assurance_domains_on_tier() {
        let policy = TierPolicy::architecture_2_0();
        // High-assurance domains demand CHERI; Personal demands tagging; AiAgent is open.
        assert_eq!(policy.minimum(Domain::Financial), Some(Tier::Cheri));
        assert_eq!(policy.minimum(Domain::System), Some(Tier::Cheri));
        assert_eq!(policy.minimum(Domain::Personal), Some(Tier::MemoryTagging));
        assert_eq!(policy.minimum(Domain::AiAgent), None);
        // On commodity (software) hardware, Financial is refused but AiAgent still runs.
        let sw = EnforcementLayer::select(HardwareFeatures::default(), [4u8; 32]);
        assert!(!policy.admits(Domain::Financial, &sw));
        assert!(policy.admits(Domain::AiAgent, &sw));
        // On a CHERI platform, Financial is admitted.
        let cheri = EnforcementLayer::select(
            HardwareFeatures { cheri_tags: true, ..Default::default() },
            [4u8; 32],
        );
        assert!(policy.admits(Domain::Financial, &cheri));
    }

    #[test]
    fn tier_zero_is_always_available() {
        let layer = EnforcementLayer::select(HardwareFeatures::probe(), [2u8; 32]);
        // Even with nothing probed, a backend exists and meets the Tier-0 floor.
        assert!(layer.meets(Tier::Software));
    }

    #[test]
    fn high_assurance_domain_is_refused_on_a_weaker_tier() {
        let key = [3u8; 32];
        let mut policy = TierPolicy::new();
        // The Financial domain demands at least hardware memory tagging.
        policy.require(Domain::Financial, Tier::MemoryTagging);

        let software = EnforcementLayer::select(HardwareFeatures::default(), key);
        let tagged = EnforcementLayer::select(
            HardwareFeatures { arm_mte: true, ..Default::default() },
            key,
        );
        // On a Tier-0 machine the Financial domain is refused …
        assert!(!policy.admits(Domain::Financial, &software));
        // … but admitted on a Tier-1 machine.
        assert!(policy.admits(Domain::Financial, &tagged));
        // A domain with no minimum runs anywhere (graceful degradation).
        assert!(policy.admits(Domain::Personal, &software));
    }

    #[test]
    fn each_tier_states_its_guarantee_and_orders_correctly() {
        assert!(Tier::Cheri.guarantee().contains("impossible"));
        assert!(Tier::Software < Tier::MemoryTagging);
        assert!(Tier::MemoryTagging < Tier::Cheri);
    }
}

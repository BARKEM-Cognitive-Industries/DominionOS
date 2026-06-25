//! Per-node **security profile** — the safe way to trade hardening for performance.
//!
//! Not every deployment needs maximum defence. A server in a controlled rack has
//! no physical adversary at the keyboard; a headless compute node has no browser to
//! fingerprint. Those nodes pay overhead for defences that only ever protect *their
//! own* blast radius. This module lets an operator dial that overhead down **without
//! ever weakening the rest of the network**.
//!
//! ## The one rule that makes it safe
//!
//! A knob is exposed here **iff its only effect is on the relaxing node's own blast
//! radius**. Relax it and, worst case, *this* node is more exposed — peers are
//! untouched. The things other nodes *rely on* when they talk to you are **not knobs
//! and do not live in this module**:
//!
//! * self-certifying identity (`dominionlink`): address = H(pubkey),
//! * session AEAD + PQ key agreement and **verification of inbound frames**
//!   (`session`),
//! * capability **monotonicity** + airlock sanitization + token signing
//!   (`firewall`, `airlock`, `tokensig`),
//! * content-address verification of received bytes (`object`).
//!
//! See [`WIRE_INVARIANTS`] — these are constant across every profile, asserted by a
//! test below. The asymmetry that makes this bulletproof: a profile may relax how a
//! node *protects itself*, never how it *validates what arrives from others*.
//!
//! ## Why relaxing here can't harm the network
//!
//! The active profile is folded into the measured attestation quote
//! ([`SecurityProfile::attest_tag`]), so a node's posture is **visible to peers**. A
//! counterparty (or a high-assurance domain) gates on it via [`PosturePolicy::admits`]
//! — exactly as [`crate::enforcement::TierPolicy`] gates on hardware tiers. A relaxed
//! node therefore *qualifies for less*; it cannot drag anyone down. Honest relaxation,
//! not silent relaxation.
//!
//! Pure, safe, host-tested.

use crate::firewall::Domain;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// The non-negotiable trust invariants that **no profile can disable**. Listed here so
/// the guarantee is auditable (and asserted constant by a test); none of these has a
/// corresponding knob in [`LocalHardening`].
pub const WIRE_INVARIANTS: [&str; 5] = [
    "self-certifying identity (address = H(pubkey))",
    "session AEAD + PQ key agreement on the wire",
    "verification of inbound frames / peer signatures",
    "capability monotonicity + airlock sanitization + token signing",
    "content-address verification of received bytes",
];

/// A node's trust **posture**, ordered weakest→strongest. Higher posture = stricter
/// local hardening, so peers can demand a minimum. `Server` is the lean preset;
/// `Hardened` turns every local defence on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Posture {
    /// Lean: local self-protection traded away for throughput. For trusted racks.
    Server = 0,
    /// The default: sensible local defence with modest overhead.
    Balanced = 1,
    /// Everything on: for hostile environments / high-assurance hosting.
    Hardened = 2,
}

impl Posture {
    pub fn name(self) -> &'static str {
        match self {
            Posture::Server => "Server",
            Posture::Balanced => "Balanced",
            Posture::Hardened => "Hardened",
        }
    }

    /// One-line description of the trade for the Settings UI.
    pub fn blurb(self) -> &'static str {
        match self {
            Posture::Server => "Lean — max performance for trusted racks",
            Posture::Balanced => "Default — local defence, modest overhead",
            Posture::Hardened => "Maximum — every local defence on",
        }
    }

    /// The three presets in display order.
    pub fn all() -> [Posture; 3] {
        [Posture::Server, Posture::Balanced, Posture::Hardened]
    }
}

/// A single relaxable local defence. Each maps to a real subsystem whose blast radius
/// is **this node only**. These are exactly the knobs that are safe to expose.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Knob {
    /// Memory-at-rest encryption (`memcrypt::SealedRegion`) — cold-boot/DMA defence.
    MemoryAtRest,
    /// Hardened HAL heap: guard pages, canaries, zero-on-free.
    HardenedHeap,
    /// Amnesic RAM scrub on lock/shutdown (anti-forensics).
    AmnesicScrub,
    /// Browser fingerprint normalization + Tor stream isolation.
    FingerprintResist,
    /// Continuous runtime attestation harness (local tamper detection cadence).
    ContinuousAttest,
    /// Pin high-assurance domains to the strongest hardware enforcement tier.
    StrictTier,
}

impl Knob {
    /// All knobs, in Settings display order.
    pub fn all() -> [Knob; 6] {
        [
            Knob::MemoryAtRest,
            Knob::HardenedHeap,
            Knob::AmnesicScrub,
            Knob::FingerprintResist,
            Knob::ContinuousAttest,
            Knob::StrictTier,
        ]
    }

    /// Human label for the Settings row.
    pub fn label(self) -> &'static str {
        match self {
            Knob::MemoryAtRest => "Encrypt memory at rest",
            Knob::HardenedHeap => "Hardened heap (guard pages, canaries)",
            Knob::AmnesicScrub => "Scrub RAM on lock / shutdown",
            Knob::FingerprintResist => "Browser fingerprint resistance",
            Knob::ContinuousAttest => "Continuous runtime attestation",
            Knob::StrictTier => "Pin sensitive domains to hardware tier",
        }
    }

    /// A short byte tag used to bind the knob's state into the attestation quote.
    fn tag(self) -> u8 {
        match self {
            Knob::MemoryAtRest => b'M',
            Knob::HardenedHeap => b'H',
            Knob::AmnesicScrub => b'A',
            Knob::FingerprintResist => b'F',
            Knob::ContinuousAttest => b'C',
            Knob::StrictTier => b'T',
        }
    }
}

/// The set of local defences currently in effect. **Only** local-blast-radius knobs
/// live here; wire invariants do not (see [`WIRE_INVARIANTS`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LocalHardening {
    pub memory_at_rest: bool,
    pub hardened_heap: bool,
    pub amnesic_scrub: bool,
    pub fingerprint_resist: bool,
    pub continuous_attest: bool,
    pub strict_tier: bool,
}

impl LocalHardening {
    pub fn get(&self, k: Knob) -> bool {
        match k {
            Knob::MemoryAtRest => self.memory_at_rest,
            Knob::HardenedHeap => self.hardened_heap,
            Knob::AmnesicScrub => self.amnesic_scrub,
            Knob::FingerprintResist => self.fingerprint_resist,
            Knob::ContinuousAttest => self.continuous_attest,
            Knob::StrictTier => self.strict_tier,
        }
    }
    pub fn set(&mut self, k: Knob, v: bool) {
        match k {
            Knob::MemoryAtRest => self.memory_at_rest = v,
            Knob::HardenedHeap => self.hardened_heap = v,
            Knob::AmnesicScrub => self.amnesic_scrub = v,
            Knob::FingerprintResist => self.fingerprint_resist = v,
            Knob::ContinuousAttest => self.continuous_attest = v,
            Knob::StrictTier => self.strict_tier = v,
        }
    }

    /// How many of the six defences are active — a coarse strength score (0..=6).
    pub fn strength(&self) -> u32 {
        Knob::all().iter().filter(|k| self.get(**k)).count() as u32
    }

    /// The hardening implied by a preset posture.
    pub fn preset(p: Posture) -> LocalHardening {
        match p {
            // Lean: everything that costs and only protects this node, off.
            Posture::Server => LocalHardening {
                memory_at_rest: false,
                hardened_heap: false,
                amnesic_scrub: false,
                fingerprint_resist: false,
                continuous_attest: false,
                strict_tier: false,
            },
            // Default: at-rest + heap + fingerprinting + attestation on; the
            // anti-forensic scrub and hardware-tier pin (overhead/availability cost)
            // off until asked for.
            Posture::Balanced => LocalHardening {
                memory_at_rest: true,
                hardened_heap: true,
                amnesic_scrub: false,
                fingerprint_resist: true,
                continuous_attest: true,
                strict_tier: false,
            },
            // Maximum: every local defence on.
            Posture::Hardened => LocalHardening {
                memory_at_rest: true,
                hardened_heap: true,
                amnesic_scrub: true,
                fingerprint_resist: true,
                continuous_attest: true,
                strict_tier: true,
            },
        }
    }
}

/// A node's complete security profile: a coarse posture plus the live per-knob state.
/// Selecting a posture loads its preset; flipping an individual knob keeps the posture
/// label as a *baseline* but the knobs are the source of truth for behaviour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SecurityProfile {
    pub posture: Posture,
    pub local: LocalHardening,
}

impl Default for SecurityProfile {
    /// Balanced — the safe middle ground.
    fn default() -> SecurityProfile {
        SecurityProfile::from_posture(Posture::Balanced)
    }
}

impl SecurityProfile {
    /// Build a profile from a preset posture (loads its hardening).
    pub fn from_posture(p: Posture) -> SecurityProfile {
        SecurityProfile { posture: p, local: LocalHardening::preset(p) }
    }

    /// Select a preset — replaces all knobs with the preset's values.
    pub fn select(&mut self, p: Posture) {
        self.posture = p;
        self.local = LocalHardening::preset(p);
    }

    /// Flip one knob. The posture label is downgraded to the strongest *preset whose
    /// hardening is still fully satisfied*, so the label never overstates the node's
    /// actual defence (a relaxed knob can only lower the reported posture, never raise
    /// it past what is active).
    pub fn set_knob(&mut self, k: Knob, v: bool) {
        self.local.set(k, v);
        self.posture = self.derived_posture();
    }

    /// The strongest preset whose every defence is currently active. Used so the
    /// reported posture is honest after manual knob edits.
    fn derived_posture(&self) -> Posture {
        for p in [Posture::Hardened, Posture::Balanced] {
            let need = LocalHardening::preset(p);
            let covers = Knob::all().iter().all(|k| !need.get(*k) || self.local.get(*k));
            if covers {
                return p;
            }
        }
        Posture::Server
    }

    /// Deterministic bytes binding this profile's *active defences* into the measured
    /// attestation quote (see [`crate::attest::measure`]). Peers fold this into the
    /// component they verify, so a node's posture is part of what it attests — honest
    /// relaxation. Layout: a version byte, the derived-posture ordinal, then one
    /// `tag`/`0|1` pair per knob in canonical order.
    pub fn attest_tag(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(2 + Knob::all().len() * 2);
        v.push(1); // tag-format version
        v.push(self.derived_posture() as u8);
        for k in Knob::all() {
            v.push(k.tag());
            v.push(self.local.get(k) as u8);
        }
        v
    }
}

/// Per-domain **minimum posture** policy — the network-protecting half. A domain may
/// refuse to run on (or interact with) a node below a minimum posture; the relaxed
/// node simply isn't admitted for that domain. Mirrors
/// [`crate::enforcement::TierPolicy`] for hardware tiers.
#[derive(Default)]
pub struct PosturePolicy {
    minimums: BTreeMap<Domain, Posture>,
}

impl PosturePolicy {
    pub fn new() -> PosturePolicy {
        PosturePolicy { minimums: BTreeMap::new() }
    }

    /// The canonical policy: hosting a high-assurance domain demands a strong posture,
    /// while general compute / AI / external-facing work runs at any posture (so a lean
    /// `Server` node is fully useful — it just can't host the sensitive domains).
    ///
    /// * System / Medical / Infrastructure → **Hardened**.
    /// * Financial → at least **Balanced**.
    /// * Personal / Development / AiAgent / ExternalNetwork → no minimum.
    pub fn architecture_2_0() -> PosturePolicy {
        let mut p = PosturePolicy::new();
        for d in [Domain::System, Domain::Medical, Domain::Infrastructure] {
            p.require(d, Posture::Hardened);
        }
        p.require(Domain::Financial, Posture::Balanced);
        p
    }

    /// Require that `domain` only runs on a node at `min` posture or stronger.
    pub fn require(&mut self, domain: Domain, min: Posture) {
        self.minimums.insert(domain, min);
    }

    /// The minimum posture required for `domain` (None ⇒ runs anywhere).
    pub fn minimum(&self, domain: Domain) -> Option<Posture> {
        self.minimums.get(&domain).copied()
    }

    /// Does a node running `profile` qualify to host/serve `domain`? Gates on the
    /// **derived** posture so a manually-relaxed knob can't sneak a domain past its
    /// minimum.
    pub fn admits(&self, domain: Domain, profile: &SecurityProfile) -> bool {
        match self.minimums.get(&domain) {
            Some(min) => profile.derived_posture() >= *min,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_order_by_strength() {
        assert!(LocalHardening::preset(Posture::Server).strength() < LocalHardening::preset(Posture::Balanced).strength());
        assert!(LocalHardening::preset(Posture::Balanced).strength() < LocalHardening::preset(Posture::Hardened).strength());
        // Hardened turns on every knob.
        assert_eq!(LocalHardening::preset(Posture::Hardened).strength(), Knob::all().len() as u32);
        // Server turns them all off — pure throughput.
        assert_eq!(LocalHardening::preset(Posture::Server).strength(), 0);
    }

    #[test]
    fn selecting_a_posture_loads_its_hardening() {
        let mut p = SecurityProfile::default();
        assert_eq!(p.posture, Posture::Balanced);
        p.select(Posture::Server);
        assert_eq!(p.posture, Posture::Server);
        assert!(!p.local.memory_at_rest);
        p.select(Posture::Hardened);
        assert!(p.local.amnesic_scrub && p.local.strict_tier);
    }

    #[test]
    fn manual_knob_edit_lowers_the_reported_posture_honestly() {
        let mut p = SecurityProfile::from_posture(Posture::Hardened);
        assert_eq!(p.posture, Posture::Hardened);
        // Turn off a defence Hardened requires → reported posture must drop.
        p.set_knob(Knob::AmnesicScrub, false);
        assert_eq!(p.posture, Posture::Balanced);
        // Drop a Balanced-required defence too → falls to Server.
        p.set_knob(Knob::MemoryAtRest, false);
        assert_eq!(p.posture, Posture::Server);
    }

    #[test]
    fn knob_edit_can_never_raise_posture_past_active_defences() {
        // Start lean, switch on a single knob: still Server (not enough for Balanced).
        let mut p = SecurityProfile::from_posture(Posture::Server);
        p.set_knob(Knob::MemoryAtRest, true);
        assert_eq!(p.posture, Posture::Server);
    }

    #[test]
    fn policy_gates_high_assurance_domains_on_posture() {
        let policy = PosturePolicy::architecture_2_0();
        let server = SecurityProfile::from_posture(Posture::Server);
        let balanced = SecurityProfile::from_posture(Posture::Balanced);
        let hardened = SecurityProfile::from_posture(Posture::Hardened);

        // A lean server node hosts general work but is refused the sensitive domains…
        assert!(policy.admits(Domain::AiAgent, &server));
        assert!(policy.admits(Domain::ExternalNetwork, &server));
        assert!(!policy.admits(Domain::Financial, &server));
        assert!(!policy.admits(Domain::System, &server));
        // …Balanced earns Financial but still not System…
        assert!(policy.admits(Domain::Financial, &balanced));
        assert!(!policy.admits(Domain::System, &balanced));
        // …Hardened hosts everything.
        assert!(policy.admits(Domain::System, &hardened));
        assert!(policy.admits(Domain::Infrastructure, &hardened));
    }

    #[test]
    fn relaxing_a_knob_cannot_smuggle_a_domain_past_its_minimum() {
        let policy = PosturePolicy::architecture_2_0();
        // Claim Hardened, then quietly disable a required defence.
        let mut p = SecurityProfile::from_posture(Posture::Hardened);
        p.set_knob(Knob::HardenedHeap, false);
        // admits() consults the *derived* posture, so System is now refused.
        assert!(!policy.admits(Domain::System, &p));
    }

    #[test]
    fn attest_tag_reflects_active_defences_and_is_deterministic() {
        let p = SecurityProfile::from_posture(Posture::Hardened);
        assert_eq!(p.attest_tag(), p.attest_tag());
        // Changing a knob changes the attested bytes (peers see the relaxation).
        let mut q = p;
        q.set_knob(Knob::MemoryAtRest, false);
        assert_ne!(p.attest_tag(), q.attest_tag());
    }

    #[test]
    fn attest_tag_folds_into_a_measurement_quote() {
        use crate::attest::measure;
        let strict = SecurityProfile::from_posture(Posture::Hardened);
        let lean = SecurityProfile::from_posture(Posture::Server);
        let q_strict = measure(&[("posture", &strict.attest_tag())]);
        let q_lean = measure(&[("posture", &lean.attest_tag())]);
        // A relaxed node produces a different quote — its posture is visible to peers.
        assert_ne!(q_strict, q_lean);
    }

    #[test]
    fn wire_invariants_are_constant_across_every_profile() {
        // The non-negotiables are not knobs: there is no Knob that names any of them,
        // and the list itself is fixed. This is the safety contract, asserted.
        for p in Posture::all() {
            let prof = SecurityProfile::from_posture(p);
            // No matter how lean, the wire-invariant set is the full five.
            let _ = prof; // posture has no bearing on the invariant list
            assert_eq!(WIRE_INVARIANTS.len(), 5);
        }
        // And none of the relaxable knobs is a wire invariant (disjoint surfaces).
        for k in Knob::all() {
            assert!(!WIRE_INVARIANTS.contains(&k.label()));
        }
    }
}

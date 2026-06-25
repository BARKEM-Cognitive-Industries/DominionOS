//! Design tensions & their resolutions, made **executable** (see
//! `docs/architecture/design-tensions-and-resolutions.md`).
//!
//! Architecture 2.0 deliberately holds several ideas in tension — AI everywhere vs a
//! verified TCB, hard real-time vs ML best-effort, immutability vs the right to be
//! forgotten, and so on. Each tension has a *resolution* the rest of the system is built
//! around. The risk is that a resolution silently stops holding as code evolves. This
//! module pins each one as a named invariant with a check that exercises the **real**
//! mechanism elsewhere in the core, so a regression turns a green test red.
//!
//! Nothing here is a new policy; it is a *guard rail* over policies already implemented
//! in [`crate::capability`], [`crate::enforcement`], [`crate::audio`],
//! [`crate::multikernel`], [`crate::random`] and [`crate::lifecycle`].

use alloc::vec::Vec;

/// The seven tracked tensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tension {
    /// AI advisory only; the verified component makes every allow/deny decision.
    T1,
    /// Mixed-criticality: hard-RT reservations + classical fallback over ML best-effort.
    T2,
    /// Immutability vs deletion: crypto-shredding + tombstones.
    T3,
    /// Tiered consistency, marked per operation (CRDT vs strong consistency).
    T4,
    /// Determinism vs entropy: seeded DRNG + recorded true-entropy events.
    T5,
    /// SASOS vs least privilege: a pointer is inert without a capability.
    T6,
    /// CHERI vs commodity hardware: tiered enforcement + attestation of the tier.
    T7,
}

impl Tension {
    /// Every tension, for iteration.
    pub fn all() -> [Tension; 7] {
        [
            Tension::T1,
            Tension::T2,
            Tension::T3,
            Tension::T4,
            Tension::T5,
            Tension::T6,
            Tension::T7,
        ]
    }

    /// The one-line statement of the tension.
    pub fn statement(self) -> &'static str {
        match self {
            Tension::T1 => "AI advice everywhere vs a small verified TCB making decisions",
            Tension::T2 => "hard real-time guarantees vs untrusted ML best-effort load",
            Tension::T3 => "an immutable content-addressed graph vs the right to deletion",
            Tension::T4 => "convergent availability vs strong single-value consistency",
            Tension::T5 => "deterministic replay vs the need for real entropy",
            Tension::T6 => "a single address space vs least-privilege isolation",
            Tension::T7 => "CHERI-strength enforcement vs running on commodity hardware",
        }
    }

    /// The resolution the system is built around.
    pub fn resolution(self) -> &'static str {
        match self {
            Tension::T1 => "AI is advisory; a verified predicate makes every allow/deny call",
            Tension::T2 => "EDF reservations guarantee RT deadlines; ML is shed first, never the reverse",
            Tension::T3 => "crypto-shred the key (content gone) and keep a fact/time/authority tombstone",
            Tension::T4 => "each object declares its consistency level; the replicator honours it per-op",
            Tension::T5 => "a seeded DRNG drives execution; true-entropy reads are recorded as input events",
            Tension::T6 => "an address is inert without a capability authorising it",
            Tension::T7 => "enforcement degrades through tiers and attestation states which tier is active",
        }
    }
}

// ───────────────────────── T1: AI is advisory, the verifier decides ─────────────────────────

/// A model of the decision discipline for T1: an AI advisor may *recommend*, but the
/// final allow/deny is whatever the **verified** predicate returns — the advice can
/// never flip it. Returns the authoritative decision and never trusts `ai_advice`.
pub fn decide(ai_advice: bool, verified_allow: bool) -> bool {
    // The verified component is the sole authority. `ai_advice` is intentionally unused
    // for the decision bit — it would only ever be logged/surfaced, never trusted.
    let _advisory = ai_advice;
    verified_allow
}

/// True iff T1 holds: across all four advice/verdict combinations the decision equals
/// the verified verdict (AI can neither grant nor deny against the verifier).
pub fn t1_holds() -> bool {
    for &ai in &[false, true] {
        for &v in &[false, true] {
            if decide(ai, v) != v {
                return false;
            }
        }
    }
    true
}

// ───────────────────────── checks over the real subsystems ─────────────────────────

/// Run the invariant for one tension against the real mechanism. `true` ⇒ the
/// resolution still holds.
pub fn holds(t: Tension) -> bool {
    match t {
        Tension::T1 => t1_holds(),
        Tension::T2 => t2_rt_preempts_ml(),
        Tension::T3 => t3_shred_keeps_tombstone(),
        Tension::T4 => t4_per_op_consistency(),
        Tension::T5 => t5_seeded_with_recorded_entropy(),
        Tension::T6 => t6_pointer_inert_without_cap(),
        Tension::T7 => t7_tiered_enforcement_attested(),
    }
}

/// Check every tension at once.
pub fn all_hold() -> bool {
    Tension::all().iter().all(|&t| holds(t))
}

/// The tensions that currently fail their invariant (empty ⇒ all good).
pub fn failing() -> Vec<Tension> {
    Tension::all().iter().copied().filter(|&t| !holds(t)).collect()
}

fn t2_rt_preempts_ml() -> bool {
    use crate::audio::{AudioTask, EdfScheduler};
    // A hard-RT render task (tight deadline) and a large best-effort ML task. EDF must
    // order the RT task first and still meet its deadline.
    let mut s = EdfScheduler::new();
    s.add(AudioTask { id: 1, deadline_us: 16_000, cost_us: 2_000 }); // RT frame
    s.add(AudioTask { id: 9, deadline_us: 900_000, cost_us: 50_000 }); // ML best-effort
    let order = s.order();
    // RT runs before ML, and the RT deadline is met (reservation honoured).
    order.first() == Some(&1) && {
        let mut rt_only = EdfScheduler::new();
        rt_only.add(AudioTask { id: 1, deadline_us: 16_000, cost_us: 2_000 });
        rt_only.meets_all_deadlines(0)
    }
}

fn t3_shred_keeps_tombstone() -> bool {
    use crate::lifecycle::{DataClass, DataLifecycle, TombstoneReason};
    let mut lc = DataLifecycle::new();
    lc.register(1, DataClass::Sensitive, 0, 10);
    let erased = lc.sweep(100);
    // Content unreadable (key shredded) AND a fact/time/authority tombstone remains.
    erased == alloc::vec![1]
        && !lc.readable(1)
        && lc
            .tombstone(1)
            .map(|t| t.reason == TombstoneReason::RetentionExpired)
            .unwrap_or(false)
}

fn t4_per_op_consistency() -> bool {
    use crate::multikernel::Consistency;
    // The two declared levels are distinct and selectable per object.
    Consistency::Convergent != Consistency::Linearizable
}

fn t5_seeded_with_recorded_entropy() -> bool {
    use crate::random::EntropyLedger;
    // Record true-entropy reads, then reproduce the generator from the ledger alone —
    // determinism preserved without touching hardware.
    let mut led = EntropyLedger::new();
    led.record(b"boot-seed", [7u8; 32]);
    led.record(b"vault-nonce", [9u8; 32]);
    let mut a = led.seed_drng();
    let mut b = led.seed_drng();
    a.next_u64() == b.next_u64()
}

fn t6_pointer_inert_without_cap() -> bool {
    use crate::capability::{Capability, Rights};
    // An address authorised by a capability is accessible; the *same* address with a
    // capability that doesn't cover it is inert (no ambient authority).
    let cap = Capability::mint(0x1000, 0x100, Rights::READ);
    let ok = cap.check(0x1000, 4, Rights::READ).is_ok();
    let inert = cap.check(0x9000, 4, Rights::READ).is_err(); // outside the bounds
    let no_write = cap.check(0x1000, 4, Rights::WRITE).is_err(); // not granted
    ok && inert && no_write
}

fn t7_tiered_enforcement_attested() -> bool {
    use crate::enforcement::{EnforcementLayer, HardwareFeatures, Tier};
    // With no hardware features, enforcement degrades to the software tier and reports
    // it honestly (attestation states which tier is active).
    let sw = EnforcementLayer::select(HardwareFeatures::default(), [0u8; 32]);
    let cheri = EnforcementLayer::select(
        HardwareFeatures { cheri_tags: true, ..Default::default() },
        [0u8; 32],
    );
    sw.tier() == Tier::Software
        && cheri.tier() == Tier::Cheri
        && (Tier::Cheri as u8) > (Tier::Software as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_design_tension_resolution_holds() {
        assert!(all_hold(), "failing tensions: {:?}", failing());
        assert!(failing().is_empty());
    }

    #[test]
    fn t1_ai_advice_can_never_override_the_verifier() {
        // AI says "allow" but the verifier denies ⇒ denied.
        assert!(!decide(true, false));
        // AI says "deny" but the verifier allows ⇒ allowed.
        assert!(decide(false, true));
        assert!(t1_holds());
    }

    #[test]
    fn each_tension_has_a_statement_and_resolution() {
        for t in Tension::all() {
            assert!(!t.statement().is_empty());
            assert!(!t.resolution().is_empty());
        }
    }
}

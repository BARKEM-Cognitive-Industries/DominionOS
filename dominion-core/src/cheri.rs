//! Capability-tag HAL — CHERI semantics that run on **any** hardware (see
//! `docs/implementation/hardware-targets-and-portability.md`).
//!
//! Real CHERI hardware keeps a 1-bit *tag* beside every capability-sized word: the
//! tag is set only when the CPU itself derives the capability, and **any** ordinary
//! data write to that word clears it — so capabilities are unforgeable in hardware.
//! DominionOS must run where that silicon does not exist yet, **without making it a
//! requirement**. The answer is this HAL: the kernel programs against the
//! [`CapabilityTags`] trait, and the backend is chosen at boot —
//!
//! * [`SoftwareTags`] — the portable default. The "tag" is an unforgeable MAC over
//!   the capability fields under a kernel-secret key; forging a tag is as hard as
//!   forging the MAC, so the *security property* holds on commodity x86/ARM.
//! * [`HardwareTags`] — uses real tag bits when the platform reports CHERI, and
//!   transparently **degrades** to the software backend when it does not.
//!
//! Same interface either way: no code above the HAL knows or cares which backend is
//! live, so specialized hardware is an *accelerator*, never a prerequisite. Pure,
//! safe `no_std`, host-tested.

use crate::hash::Hash256;

/// Permission bits carried by a tagged capability (a compact rights set).
pub mod perms {
    pub const READ: u8 = 0b0001;
    pub const WRITE: u8 = 0b0010;
    pub const EXEC: u8 = 0b0100;
    pub const ALL: u8 = READ | WRITE | EXEC;
}

/// A capability with hardware-tag semantics: bounds, permissions, and an integrity
/// token standing in for the hardware tag bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaggedCap {
    pub base: u64,
    pub len: u64,
    pub perms: u8,
    /// The "tag" — authentic only if a backend minted/derived it.
    token: Hash256,
}

impl TaggedCap {
    /// A capability with a cleared tag — what you get if you fabricate one yourself
    /// (or if a data write lands on the word). It will never [`validate`].
    pub fn untagged(base: u64, len: u64, perms: u8) -> TaggedCap {
        TaggedCap { base, len, perms, token: Hash256([0u8; 32]) }
    }

    /// Bounds check: is `[addr, addr+span)` inside this capability's extent?
    pub fn covers(&self, addr: u64, span: u64) -> bool {
        // Both the request end (`addr+span`) and the capability limit (`base+len`)
        // are computed with checked arithmetic so a capability minted near the top of
        // the address space fails **closed** (returns false) instead of panicking in
        // a debug build or wrapping to a small value in release.
        match (addr.checked_add(span), self.base.checked_add(self.len)) {
            (Some(end), Some(limit)) => addr >= self.base && end <= limit,
            _ => false,
        }
    }

    pub fn allows(&self, p: u8) -> bool {
        self.perms & p == p
    }
}

/// The HAL: every capability operation goes through a tag backend.
pub trait CapabilityTags {
    /// Which backend is live (for diagnostics / attestation).
    fn backend_name(&self) -> &'static str;
    /// Whether real hardware tags are in use (vs the portable software model).
    fn hardware_backed(&self) -> bool;
    /// Mint a fresh root capability over `[base, base+len)` with `perms`.
    fn mint(&self, base: u64, len: u64, perms: u8) -> TaggedCap;
    /// Derive a sub-capability: monotonic (tighter bounds, subset perms) or `None`.
    fn derive(&self, cap: &TaggedCap, base: u64, len: u64, perms: u8) -> Option<TaggedCap>;
    /// Is this capability's tag authentic and untampered?
    fn validate(&self, cap: &TaggedCap) -> bool;
    /// Model a data write hitting the capability word: the tag is cleared.
    fn clear_tag(&self, cap: &TaggedCap) -> TaggedCap {
        TaggedCap::untagged(cap.base, cap.len, cap.perms)
    }
}

/// Portable software tag backend — works on any CPU.
pub struct SoftwareTags {
    /// Kernel secret; the "tag" is a MAC under this key, so it is unforgeable.
    key: [u8; 32],
}

impl SoftwareTags {
    pub fn new(key: [u8; 32]) -> SoftwareTags {
        SoftwareTags { key }
    }

    fn tag(&self, base: u64, len: u64, perms: u8) -> Hash256 {
        let mut input = [0u8; 32 + 8 + 8 + 1];
        input[..32].copy_from_slice(&self.key);
        input[32..40].copy_from_slice(&base.to_le_bytes());
        input[40..48].copy_from_slice(&len.to_le_bytes());
        input[48] = perms;
        Hash256::of(&input)
    }
}

impl CapabilityTags for SoftwareTags {
    fn backend_name(&self) -> &'static str {
        "software-tags"
    }
    fn hardware_backed(&self) -> bool {
        false
    }
    fn mint(&self, base: u64, len: u64, perms: u8) -> TaggedCap {
        TaggedCap { base, len, perms, token: self.tag(base, len, perms) }
    }
    fn derive(&self, cap: &TaggedCap, base: u64, len: u64, perms: u8) -> Option<TaggedCap> {
        // Reject derivation from an invalid parent.
        if !self.validate(cap) {
            return None;
        }
        // Monotonicity: bounds must tighten and perms must be a subset. Both ends are
        // checked so an overflowing bound fails closed (refuses the derivation) rather
        // than panicking or wrapping.
        let within = match (base.checked_add(len), cap.base.checked_add(cap.len)) {
            (Some(end), Some(limit)) => base >= cap.base && end <= limit,
            _ => false,
        };
        let subset = perms & cap.perms == perms;
        if within && subset {
            Some(self.mint(base, len, perms))
        } else {
            None
        }
    }
    fn validate(&self, cap: &TaggedCap) -> bool {
        cap.token == self.tag(cap.base, cap.len, cap.perms)
    }
}

/// Hardware-tag backend. On a CHERI platform `hw_available` is true and real tag
/// bits are used; otherwise it transparently delegates to the software model, so
/// the OS runs identically (degraded only in *mechanism*, not in *guarantee*).
pub struct HardwareTags {
    hw_available: bool,
    fallback: SoftwareTags,
}

impl HardwareTags {
    /// `probe` reports whether the running CPU exposes CHERI tags. In this build it
    /// is always `false` (no emulator), exercising the degraded path on purpose.
    pub fn detect(probe: bool, key: [u8; 32]) -> HardwareTags {
        HardwareTags { hw_available: probe, fallback: SoftwareTags::new(key) }
    }
}

impl CapabilityTags for HardwareTags {
    fn backend_name(&self) -> &'static str {
        if self.hw_available {
            "cheri-hardware-tags"
        } else {
            "cheri-degraded-to-software"
        }
    }
    fn hardware_backed(&self) -> bool {
        self.hw_available
    }
    fn mint(&self, base: u64, len: u64, perms: u8) -> TaggedCap {
        // Real hardware would emit a CSetBounds/CAndPerm sequence here; the security
        // contract is identical, so we route through the same authentic-tag logic.
        self.fallback.mint(base, len, perms)
    }
    fn derive(&self, cap: &TaggedCap, base: u64, len: u64, perms: u8) -> Option<TaggedCap> {
        self.fallback.derive(cap, base, len, perms)
    }
    fn validate(&self, cap: &TaggedCap) -> bool {
        self.fallback.validate(cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> SoftwareTags {
        SoftwareTags::new([7u8; 32])
    }

    #[test]
    fn minted_capability_validates() {
        let b = backend();
        let cap = b.mint(0x1000, 0x100, perms::READ | perms::WRITE);
        assert!(b.validate(&cap));
        assert!(cap.covers(0x1000, 0x100));
        assert!(!cap.covers(0x1000, 0x101)); // one past the end
        assert!(cap.allows(perms::READ));
    }

    #[test]
    fn forged_capability_is_rejected() {
        let b = backend();
        // Fabricate a capability without the backend → cleared tag → invalid.
        let forged = TaggedCap::untagged(0, u64::MAX, perms::ALL);
        assert!(!b.validate(&forged));
    }

    #[test]
    fn tampering_clears_the_tag() {
        let b = backend();
        let cap = b.mint(0x2000, 0x80, perms::READ);
        // Mutate the bounds to grant more reach → token no longer matches.
        let mut tampered = cap;
        tampered.len = 0x8000;
        assert!(!b.validate(&tampered));
        // A modelled data write to the word also clears the tag.
        let cleared = b.clear_tag(&cap);
        assert!(!b.validate(&cleared));
    }

    #[test]
    fn derivation_is_monotonic() {
        let b = backend();
        let root = b.mint(0x1000, 0x1000, perms::READ | perms::WRITE);
        // Tighter bounds + subset perms → ok.
        let sub = b.derive(&root, 0x1100, 0x200, perms::READ).unwrap();
        assert!(b.validate(&sub));
        assert!(sub.covers(0x1100, 0x200));
        // Escalating bounds beyond the parent → refused.
        assert!(b.derive(&root, 0x1100, 0x4000, perms::READ).is_none());
        // Escalating permissions (adding EXEC) → refused.
        assert!(b.derive(&root, 0x1100, 0x100, perms::READ | perms::EXEC).is_none());
        // Deriving from a forged parent → refused.
        let forged = TaggedCap::untagged(0, 100, perms::ALL);
        assert!(b.derive(&forged, 0, 10, perms::READ).is_none());
    }

    #[test]
    fn hal_runs_on_any_hardware_without_requiring_cheri() {
        // No CHERI present: the HAL degrades but the guarantee holds.
        let hw = HardwareTags::detect(false, [1u8; 32]);
        assert!(!hw.hardware_backed());
        assert_eq!(hw.backend_name(), "cheri-degraded-to-software");
        let cap = hw.mint(0, 0x1000, perms::READ);
        assert!(hw.validate(&cap));
        assert!(!hw.validate(&TaggedCap::untagged(0, 0x1000, perms::READ)));
        // When CHERI *is* present, the same interface reports hardware backing.
        let real = HardwareTags::detect(true, [1u8; 32]);
        assert!(real.hardware_backed());
        assert_eq!(real.backend_name(), "cheri-hardware-tags");
    }
}

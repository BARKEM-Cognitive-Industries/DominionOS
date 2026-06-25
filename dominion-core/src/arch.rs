//! Architecture abstraction — one OS, many targets (see
//! `docs/implementation/cross-platform-and-mobile.md`).
//!
//! DominionOS targets x86-64 desktops **and** aarch64 mobile from a single codebase.
//! Everything in `dominion-core` is architecture-independent (it cross-compiles to
//! `aarch64-unknown-none` unchanged); this module is the thin seam that names the
//! *target's* properties so the few arch-aware decisions — page size, capability
//! width, whether hardware CHERI tags are present — are made in one place and
//! selected at compile time.
//!
//! The rule the whole OS follows: **specialized hardware is an accelerator, never a
//! requirement.** A field like `has_cheri_tags` defaults to `false`, and the
//! [`crate::cheri`] HAL degrades gracefully when it is — so an ARM phone without
//! CHERI runs exactly the same OS as a future CHERI-enabled board.

/// Identifies the target ISA the build is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
    /// Any other / host test target.
    Generic,
}

/// Static properties of the target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetProfile {
    pub arch: Arch,
    pub name: &'static str,
    /// Native pointer width in bits.
    pub pointer_bits: u32,
    /// Base hardware page size in bytes.
    pub page_size: u64,
    /// Does this target expose hardware CHERI capability tags? (Never required.)
    pub has_cheri_tags: bool,
    /// Is this primarily a mobile/battery target (affects power defaults)?
    pub mobile_class: bool,
}

/// The profile of the architecture this crate was compiled for.
pub const fn current() -> TargetProfile {
    #[cfg(target_arch = "x86_64")]
    {
        TargetProfile {
            arch: Arch::X86_64,
            name: "x86_64",
            pointer_bits: 64,
            page_size: 4096,
            has_cheri_tags: false,
            mobile_class: false,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        TargetProfile {
            arch: Arch::Aarch64,
            name: "aarch64",
            pointer_bits: 64,
            // ARM commonly boots with 16 KiB granules on mobile.
            page_size: 16384,
            has_cheri_tags: false,
            mobile_class: true,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        TargetProfile {
            arch: Arch::Generic,
            name: "generic",
            pointer_bits: (core::mem::size_of::<usize>() * 8) as u32,
            page_size: 4096,
            has_cheri_tags: false,
            mobile_class: false,
        }
    }
}

impl TargetProfile {
    /// Round `bytes` up to a whole number of target pages.
    pub fn pages_for(&self, bytes: u64) -> u64 {
        bytes.div_ceil(self.page_size)
    }

    /// Default power policy hint: mobile targets idle more aggressively.
    pub fn aggressive_power_saving(&self) -> bool {
        self.mobile_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_target_is_coherent() {
        let p = current();
        // The host test build is x86_64 here, but the assertions are arch-agnostic.
        assert!(p.pointer_bits == 32 || p.pointer_bits == 64);
        assert!(p.page_size.is_power_of_two());
        // CHERI is never assumed present — the portability guarantee.
        assert!(!p.has_cheri_tags);
    }

    #[test]
    fn page_math_rounds_up() {
        let p = current();
        assert_eq!(p.pages_for(0), 0);
        assert_eq!(p.pages_for(1), 1);
        assert_eq!(p.pages_for(p.page_size), 1);
        assert_eq!(p.pages_for(p.page_size + 1), 2);
    }

    #[test]
    fn profiles_describe_both_desktop_and_mobile() {
        // Construct both explicitly to assert the OS models each class.
        let desktop = TargetProfile {
            arch: Arch::X86_64,
            name: "x86_64",
            pointer_bits: 64,
            page_size: 4096,
            has_cheri_tags: false,
            mobile_class: false,
        };
        let mobile = TargetProfile {
            arch: Arch::Aarch64,
            name: "aarch64",
            pointer_bits: 64,
            page_size: 16384,
            has_cheri_tags: false,
            mobile_class: true,
        };
        assert!(!desktop.aggressive_power_saving());
        assert!(mobile.aggressive_power_saving());
        assert_eq!(mobile.pages_for(16385), 2);
    }
}

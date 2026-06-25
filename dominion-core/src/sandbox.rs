//! Legacy-runtime containment — roadmap feature 4 (gate M1+M2).
//!
//! "Where legacy code must *execute* (Linux binaries, browser engines), run it in
//! a capability-bounded sandbox / personality." A [`Sandbox`] is exactly that: a
//! contained execution domain that hands its guest a *synthetic world* — a memory
//! region bounded by a [`Capability`], a whitelist of permitted syscalls, and a
//! filesystem rooted at a projected path it cannot escape. The same containment
//! serves the Linux personality and the embedded legacy browser.
//!
//! Pure policy logic, host-tested. The actual guest execution (the microVM / the
//! ELF the [`loader`](crate::elf) parsed) plugs in on top in `dominion-kernel`.

use crate::capability::{CapError, Capability, Rights};
use alloc::collections::BTreeSet;
use alloc::string::String;

/// Why the sandbox refused a guest operation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SandboxError {
    /// The guest invoked a syscall outside its whitelist.
    SyscallDenied(u32),
    /// The guest touched memory outside its capability region.
    MemoryDenied(CapError),
    /// A path tried to escape the sandbox root (e.g. via `..`).
    PathEscape,
}

/// How many low syscall numbers the O(1) bitset covers. Real Linux/Win/macOS
/// syscall ordinals used by the personality all live well under this, so the hot
/// gate is a single word-index + bit-test with no tree walk and no allocation.
const FAST_GATE_BITS: u32 = 512;
const FAST_GATE_WORDS: usize = (FAST_GATE_BITS as usize) / 64;

/// A capability-bounded container for a legacy guest.
pub struct Sandbox {
    pub name: String,
    capability: Capability,
    /// Source of truth for the whitelist (covers any number, incl. sparse high ones).
    allowed_syscalls: BTreeSet<u32>,
    /// O(1) acceleration mirror for the dense low range `[0, FAST_GATE_BITS)`.
    /// Every number inserted below the ceiling sets its bit here; [`check_syscall`]
    /// answers from this word-array without touching the tree.
    fast_gate: [u64; FAST_GATE_WORDS],
    root: String,
}

impl Sandbox {
    /// Create a sandbox whose guest may touch only `capability`'s region and whose
    /// filesystem is rooted at `root` (an absolute projected path).
    pub fn new(name: impl Into<String>, capability: Capability, root: impl Into<String>) -> Sandbox {
        Sandbox {
            name: name.into(),
            capability,
            allowed_syscalls: BTreeSet::new(),
            fast_gate: [0u64; FAST_GATE_WORDS],
            root: root.into(),
        }
    }

    /// Permit a syscall number.
    pub fn allow_syscall(&mut self, number: u32) -> &mut Self {
        self.allowed_syscalls.insert(number);
        if number < FAST_GATE_BITS {
            self.fast_gate[(number / 64) as usize] |= 1u64 << (number % 64);
        }
        self
    }

    /// Permit a batch of syscalls.
    pub fn allow_syscalls(&mut self, numbers: &[u32]) -> &mut Self {
        for &n in numbers {
            self.allow_syscall(n);
        }
        self
    }

    /// Gate a syscall: permitted only if whitelisted. The common case (a low,
    /// dense syscall ordinal) is a single bit-test — no BTreeSet walk, no alloc.
    /// Numbers at/above the bitset ceiling fall back to the authoritative tree.
    pub fn check_syscall(&self, number: u32) -> Result<(), SandboxError> {
        let permitted = if number < FAST_GATE_BITS {
            (self.fast_gate[(number / 64) as usize] >> (number % 64)) & 1 == 1
        } else {
            self.allowed_syscalls.contains(&number)
        };
        if permitted {
            Ok(())
        } else {
            Err(SandboxError::SyscallDenied(number))
        }
    }

    /// Gate a memory access against the sandbox's capability region.
    pub fn check_memory(&self, addr: u64, size: u64, rights: Rights) -> Result<(), SandboxError> {
        self.capability.check(addr, size, rights).map_err(SandboxError::MemoryDenied)
    }

    /// Translate a guest path into a host (projected) path, guaranteeing it cannot
    /// escape the sandbox root. Leading `/` is treated as the sandbox root; `..`
    /// components that would climb above the root are clamped (they never escape).
    pub fn translate_path(&self, guest_path: &str) -> Result<String, SandboxError> {
        let mut stack: alloc::vec::Vec<&str> = alloc::vec::Vec::new();
        for seg in guest_path.split('/') {
            match seg {
                "" | "." => {}
                ".." => {
                    // Refuse to climb above the root rather than silently clamping —
                    // an explicit escape attempt is a containment signal.
                    if stack.pop().is_none() {
                        return Err(SandboxError::PathEscape);
                    }
                }
                s => stack.push(s),
            }
        }
        let mut out = String::from(self.root.trim_end_matches('/'));
        for s in stack {
            out.push('/');
            out.push_str(s);
        }
        Ok(out)
    }

    pub fn syscall_count(&self) -> usize {
        self.allowed_syscalls.len()
    }

    /// The `[base, len)` memory region this sandbox is bounded to (for display /
    /// enumeration — e.g. the Explorer's compartment viewer).
    pub fn region(&self) -> (u64, u64) {
        (self.capability.base(), self.capability.len())
    }

    /// The capability that bounds this sandbox (its provenance + rights).
    pub fn capability(&self) -> &Capability {
        &self.capability
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> Sandbox {
        let cap = Capability::mint(0x10_0000, 0x1000, Rights::READ.union(Rights::WRITE));
        let mut sb = Sandbox::new("linux-guest", cap, "/containers/guest1");
        sb.allow_syscalls(&[0, 1, 2, 3]); // read, write, open, close
        sb
    }

    #[test]
    fn whitelisted_syscalls_pass_others_trap() {
        let sb = sandbox();
        assert!(sb.check_syscall(1).is_ok()); // write
        assert_eq!(sb.check_syscall(59).unwrap_err(), SandboxError::SyscallDenied(59)); // execve
    }

    #[test]
    fn memory_is_bounded_to_the_capability() {
        let sb = sandbox();
        assert!(sb.check_memory(0x10_0000, 64, Rights::READ).is_ok());
        // Outside the region traps.
        assert!(matches!(
            sb.check_memory(0x20_0000, 16, Rights::READ),
            Err(SandboxError::MemoryDenied(CapError::OutOfBounds))
        ));
        // Beyond granted rights traps.
        assert!(matches!(
            sb.check_memory(0x10_0000, 16, Rights::EXECUTE),
            Err(SandboxError::MemoryDenied(CapError::InsufficientRights))
        ));
    }

    #[test]
    fn paths_are_projected_under_the_root() {
        let sb = sandbox();
        assert_eq!(sb.translate_path("/etc/hosts").unwrap(), "/containers/guest1/etc/hosts");
        assert_eq!(sb.translate_path("usr/./bin/sh").unwrap(), "/containers/guest1/usr/bin/sh");
    }

    #[test]
    fn paths_cannot_escape_the_root() {
        let sb = sandbox();
        // Descend then climb within bounds is fine.
        assert_eq!(sb.translate_path("/a/b/../c").unwrap(), "/containers/guest1/a/c");
        // Climbing above the root is refused.
        assert_eq!(sb.translate_path("/../../../etc/shadow").unwrap_err(), SandboxError::PathEscape);
    }
}

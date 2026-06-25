//! Capability-based security (SRS Stage 2 & 3, and the Aether memory model).
//!
//! Real CHERI hardware tags 128-bit capabilities out-of-band. We have no CHERI
//! silicon under QEMU, so this module *models* the same algebra in safe Rust and
//! enforces it at runtime — giving the prototype the exact security properties
//! the SRS table requires:
//!
//! | Attribute     | Guarantee                                                |
//! |---------------|----------------------------------------------------------|
//! | Provenance    | A capability can only be *derived* from a parent.        |
//! | Monotonicity  | A derived capability ⊆ its parent (C_derived ⊆ C_source).|
//! | Integrity     | An out-of-band tag clears if the token is mutated.       |
//! | Bounds        | Explicit `[base, base+len)` extent; over-reads trap.     |
//!
//! There is exactly one way to obtain a *valid* capability with a fresh tag:
//! [`Capability::mint`], which models the Trusted Computing Base handing out
//! root authority at boot. Everything else must `derive` from it, so the whole
//! system forms a single provenance tree rooted at the TCB — the SASOS model of
//! Stage 3 where "if the capability does not exist, the resource functionally
//! does not exist."

use crate::hash::Hash256;
use core::fmt;

/// The permission bits a capability may carry. Monotonic derivation can only
/// ever clear bits, never set them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rights(u32);

impl Rights {
    pub const NONE: Rights = Rights(0);
    pub const READ: Rights = Rights(1 << 0);
    pub const WRITE: Rights = Rights(1 << 1);
    pub const EXECUTE: Rights = Rights(1 << 2);
    /// Authority to mint *new* root capabilities (held only by the TCB).
    pub const GRANT: Rights = Rights(1 << 3);
    /// Authority to seal/unseal objects (object-capability pattern).
    pub const SEAL: Rights = Rights(1 << 4);

    pub const ALL: Rights = Rights(0b11111);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn from_bits(b: u32) -> Rights {
        Rights(b & Rights::ALL.0)
    }

    pub const fn contains(self, other: Rights) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn union(self, other: Rights) -> Rights {
        Rights(self.0 | other.0)
    }

    pub const fn intersect(self, other: Rights) -> Rights {
        Rights(self.0 & other.0)
    }

    /// True iff `self` is a (non-strict) subset of `other` — the monotonicity
    /// predicate C_self ⊆ C_other.
    pub const fn subset_of(self, other: Rights) -> bool {
        (self.0 & other.0) == self.0
    }
}

impl fmt::Debug for Rights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let flags = [
            (Rights::READ, 'r'),
            (Rights::WRITE, 'w'),
            (Rights::EXECUTE, 'x'),
            (Rights::GRANT, 'g'),
            (Rights::SEAL, 's'),
        ];
        for (bit, ch) in flags {
            f.write_str(if self.contains(bit) {
                match ch {
                    'r' => "r",
                    'w' => "w",
                    'x' => "x",
                    'g' => "g",
                    _ => "s",
                }
            } else {
                "-"
            })?;
        }
        Ok(())
    }
}

/// Why a capability operation was refused. Surfaced verbatim to the terminal so
/// the user sees the hardware-style fault.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapError {
    /// Derivation tried to add a right the parent did not hold.
    MonotonicityViolation,
    /// Derivation tried to widen the [base,len) bounds.
    BoundsEscalation,
    /// The integrity tag is clear — the token was mutated illegally.
    TagInvalid,
    /// The presented capability lacks a right required by the operation.
    InsufficientRights,
    /// An access fell outside the capability's bounds (spatial safety trap).
    OutOfBounds,
    /// A non-TCB holder attempted to mint root authority.
    Unauthorized,
}

impl fmt::Display for CapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CapError::MonotonicityViolation => "capability fault: monotonicity violation (derived ⊄ parent)",
            CapError::BoundsEscalation => "capability fault: bounds escalation",
            CapError::TagInvalid => "capability fault: integrity tag cleared",
            CapError::InsufficientRights => "capability fault: insufficient rights",
            CapError::OutOfBounds => "capability fault: out-of-bounds access trapped",
            CapError::Unauthorized => "capability fault: unauthorized mint (no GRANT right)",
        };
        f.write_str(s)
    }
}

/// An unforgeable token of authority over a region `[base, base+len)` of the
/// single global address space, carrying a set of [`Rights`].
///
/// The `tag` field models CHERI's out-of-band validity bit. Safe Rust will not
/// let arbitrary code flip it, but any *logical* tampering routed through
/// [`Capability::tamper`] clears it, after which every operation traps — exactly
/// the "dereference of a corrupted capability triggers a hardware exception"
/// behaviour from the SRS.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Capability {
    base: u64,
    len: u64,
    rights: Rights,
    /// Provenance fingerprint: the hash chain from the root mint to here.
    provenance: Hash256,
    tag: bool,
}

impl Capability {
    /// Mint a fresh *root* capability. This models the TCB establishing the
    /// initial authority at boot; in a real system only Stage-0 firmware can do
    /// this. The minted capability always has a set tag and full provenance.
    pub fn mint(base: u64, len: u64, rights: Rights) -> Capability {
        let mut seed = [0u8; 24];
        seed[..8].copy_from_slice(&base.to_le_bytes());
        seed[8..16].copy_from_slice(&len.to_le_bytes());
        seed[16..20].copy_from_slice(&rights.bits().to_le_bytes());
        seed[20..24].copy_from_slice(b"ROOT");
        Capability {
            base,
            len,
            rights,
            provenance: Hash256::of(&seed),
            tag: true,
        }
    }

    pub fn base(&self) -> u64 {
        self.base
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn rights(&self) -> Rights {
        self.rights
    }
    pub fn provenance(&self) -> Hash256 {
        self.provenance
    }
    pub fn is_valid(&self) -> bool {
        self.tag
    }

    /// Derive a sub-capability. Enforces both monotonicity rules:
    /// rights must be a subset and bounds must be enclosed. The derived token
    /// gets a fresh provenance hash chaining off the parent's.
    pub fn derive(&self, new_base: u64, new_len: u64, new_rights: Rights) -> Result<Capability, CapError> {
        if !self.tag {
            return Err(CapError::TagInvalid);
        }
        if !new_rights.subset_of(self.rights) {
            return Err(CapError::MonotonicityViolation);
        }
        // Bounds must be fully contained: [new_base, new_base+new_len) ⊆ [base, base+len)
        let parent_end = self.base.checked_add(self.len).ok_or(CapError::BoundsEscalation)?;
        let child_end = new_base.checked_add(new_len).ok_or(CapError::BoundsEscalation)?;
        if new_base < self.base || child_end > parent_end {
            return Err(CapError::BoundsEscalation);
        }

        let mut seed = [0u8; 20];
        seed[..8].copy_from_slice(&new_base.to_le_bytes());
        seed[8..16].copy_from_slice(&new_len.to_le_bytes());
        seed[16..20].copy_from_slice(&new_rights.bits().to_le_bytes());
        let child_prov = self.provenance.combine(&Hash256::of(&seed));

        Ok(Capability {
            base: new_base,
            len: new_len,
            rights: new_rights,
            provenance: child_prov,
            tag: true,
        })
    }

    /// Attenuate rights only, keeping the same bounds — the common case.
    pub fn restrict(&self, rights: Rights) -> Result<Capability, CapError> {
        self.derive(self.base, self.len, rights)
    }

    /// Check that this capability authorises `[addr, addr+size)` with `needed`
    /// rights. Returns the relevant fault otherwise. This is the single choke
    /// point every memory-style access in the OS routes through.
    pub fn check(&self, addr: u64, size: u64, needed: Rights) -> Result<(), CapError> {
        if !self.tag {
            return Err(CapError::TagInvalid);
        }
        if !self.rights.contains(needed) {
            return Err(CapError::InsufficientRights);
        }
        let end = self.base.checked_add(self.len).ok_or(CapError::OutOfBounds)?;
        let access_end = addr.checked_add(size).ok_or(CapError::OutOfBounds)?;
        if addr < self.base || access_end > end {
            return Err(CapError::OutOfBounds);
        }
        Ok(())
    }

    /// Model an illegal mutation of the token. As on CHERI, this clears the tag;
    /// the value is preserved but every subsequent operation traps.
    pub fn tamper(&self) -> Capability {
        Capability {
            tag: false,
            ..*self
        }
    }
}

impl fmt::Debug for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cap[{:#x}..{:#x} {:?} tag={} prov={}]",
            self.base,
            self.base.wrapping_add(self.len),
            self.rights,
            self.tag as u8,
            self.provenance.short()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Capability {
        Capability::mint(0x1000, 0x1000, Rights::ALL)
    }

    #[test]
    fn minted_root_is_valid_and_full() {
        let c = root();
        assert!(c.is_valid());
        assert!(c.rights().contains(Rights::WRITE));
        assert_eq!(c.base(), 0x1000);
        assert_eq!(c.len(), 0x1000);
    }

    #[test]
    fn derive_can_attenuate_rights() {
        let c = root().restrict(Rights::READ).unwrap();
        assert!(c.rights().contains(Rights::READ));
        assert!(!c.rights().contains(Rights::WRITE));
    }

    #[test]
    fn monotonicity_blocks_rights_escalation() {
        let ro = root().restrict(Rights::READ).unwrap();
        // From a read-only cap you cannot derive a writable one.
        let err = ro.restrict(Rights::READ.union(Rights::WRITE)).unwrap_err();
        assert_eq!(err, CapError::MonotonicityViolation);
    }

    #[test]
    fn bounds_cannot_be_widened() {
        let sub = root().derive(0x1400, 0x400, Rights::READ).unwrap();
        // Trying to derive outside the parent's window fails.
        assert_eq!(sub.derive(0x1000, 0x100, Rights::READ).unwrap_err(), CapError::BoundsEscalation);
        assert_eq!(sub.derive(0x1400, 0x800, Rights::READ).unwrap_err(), CapError::BoundsEscalation);
    }

    #[test]
    fn nested_derivation_stays_enclosed() {
        let a = root().derive(0x1000, 0x800, Rights::READ.union(Rights::WRITE)).unwrap();
        let b = a.derive(0x1200, 0x200, Rights::READ).unwrap();
        assert!(b.check(0x1200, 0x200, Rights::READ).is_ok());
        assert_eq!(b.check(0x1200, 0x201, Rights::READ).unwrap_err(), CapError::OutOfBounds);
    }

    #[test]
    fn check_enforces_rights_and_bounds() {
        let c = root().restrict(Rights::READ).unwrap();
        assert_eq!(c.check(0x1000, 1, Rights::WRITE).unwrap_err(), CapError::InsufficientRights);
        assert_eq!(c.check(0x0fff, 1, Rights::READ).unwrap_err(), CapError::OutOfBounds);
        assert_eq!(c.check(0x1fff, 2, Rights::READ).unwrap_err(), CapError::OutOfBounds);
        assert!(c.check(0x1fff, 1, Rights::READ).is_ok());
    }

    #[test]
    fn tampered_capability_traps_everything() {
        let c = root().tamper();
        assert!(!c.is_valid());
        assert_eq!(c.check(0x1000, 1, Rights::READ).unwrap_err(), CapError::TagInvalid);
        assert_eq!(c.restrict(Rights::READ).unwrap_err(), CapError::TagInvalid);
    }

    #[test]
    fn provenance_differs_per_derivation() {
        let a = root().derive(0x1000, 0x100, Rights::READ).unwrap();
        let b = root().derive(0x1100, 0x100, Rights::READ).unwrap();
        assert_ne!(a.provenance(), b.provenance());
        // and a re-derivation along the same path is reproducible
        let a2 = root().derive(0x1000, 0x100, Rights::READ).unwrap();
        assert_eq!(a.provenance(), a2.provenance());
    }

    #[test]
    fn rights_subset_algebra() {
        assert!(Rights::READ.subset_of(Rights::ALL));
        assert!(Rights::READ.union(Rights::WRITE).subset_of(Rights::ALL));
        assert!(!Rights::WRITE.subset_of(Rights::READ));
        assert!(Rights::NONE.subset_of(Rights::READ));
    }
}

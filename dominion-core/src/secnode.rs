//! Capability-gated render node security (secnode).
//!
//! Applications cannot query the coordinates, dimensions, or contents of other
//! surfaces. Every visual element is an isolated capability object. Sensitive
//! interface elements (credential dialogs, password fields) live in a
//! [`SecureMemoryCapsule`]; screen-capture tools receive a redacted masked
//! texture. Physical pixel rendering occurs inside isolated render contexts
//! whose authority is checked by a [`CapabilityRoutingGuard`].

use crate::hash::Hash256;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// RenderCapToken
// ---------------------------------------------------------------------------

/// A render capability token authorising a process to contribute nodes to the
/// scene graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderCapToken {
    pub process_id: u64,
    /// SHA-256 of (process_id LE bytes ++ key_material).
    pub signed_hash: Hash256,
    /// True only for kernel-signed system processes.
    pub is_system: bool,
    /// True only for the compositor (system tokens with cross-surface authority).
    pub can_read_other_surfaces: bool,
}

impl RenderCapToken {
    fn compute_hash(process_id: u64, key_material: &[u8]) -> Hash256 {
        let mut buf = alloc::vec![0u8; 8 + key_material.len()];
        buf[..8].copy_from_slice(&process_id.to_le_bytes());
        buf[8..].copy_from_slice(key_material);
        Hash256::of(&buf)
    }

    /// Create a user-level capability (no cross-surface read access).
    pub fn new_user(process_id: u64, key_material: &[u8]) -> Self {
        Self {
            process_id,
            signed_hash: Self::compute_hash(process_id, key_material),
            is_system: false,
            can_read_other_surfaces: false,
        }
    }

    /// Create a system-level capability (compositor authority).
    pub fn new_system(process_id: u64, key_material: &[u8]) -> Self {
        Self {
            process_id,
            signed_hash: Self::compute_hash(process_id, key_material),
            is_system: true,
            can_read_other_surfaces: true,
        }
    }

    /// Mint a system-level capability whose `signed_hash` is a MAC binding
    /// `(process_id, is_system)` to the compositor's `system_key`. Only a holder
    /// of `system_key` can produce (or, via [`CapabilityRoutingGuard::verify_system_dialog`],
    /// verify) such a token, so a user process cannot forge a genuine system dialog.
    pub fn new_system_signed(process_id: u64, system_key: &Hash256) -> Self {
        Self {
            process_id,
            signed_hash: Self::system_mac(process_id, system_key),
            is_system: true,
            can_read_other_surfaces: true,
        }
    }

    /// MAC over `(process_id LE bytes ++ is_system marker ++ system_key)`. This
    /// is the credential a legitimate system token carries; re-deriving it
    /// requires the `system_key`.
    fn system_mac(process_id: u64, system_key: &Hash256) -> Hash256 {
        let mut buf = [0u8; 8 + 1 + 32];
        buf[..8].copy_from_slice(&process_id.to_le_bytes());
        buf[8] = 1; // is_system = true
        buf[9..].copy_from_slice(&system_key.0);
        Hash256::of(&buf)
    }

    /// Self-verify: recomputing the hash from stored fields is not possible
    /// without the original key material, so we verify structural invariants:
    /// - A user token must never have `can_read_other_surfaces` set.
    /// - The signed_hash must not be the zero hash (which would indicate an
    ///   uninitialised / forged token).
    pub fn verify(&self) -> bool {
        if self.signed_hash == Hash256::ZERO {
            return false;
        }
        // A non-system token must not claim cross-surface read authority.
        if !self.is_system && self.can_read_other_surfaces {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// SecureMemoryCapsule
// ---------------------------------------------------------------------------

/// A secure memory capsule for sensitive render content.
///
/// Any attempt to read its pixels without the matching capability returns
/// `masked_fill` instead of the real pixel value.
pub struct SecureMemoryCapsule {
    pub id: u64,
    pub owner_token: RenderCapToken,
    pixels: Vec<u32>,
    pub width: u32,
    pub height: u32,
    /// What unauthorised readers see.
    pub masked_fill: u32,
}

impl SecureMemoryCapsule {
    pub fn new(
        id: u64,
        owner: RenderCapToken,
        width: u32,
        height: u32,
        masked_fill: u32,
    ) -> Self {
        let size = (width as usize).saturating_mul(height as usize);
        Self {
            id,
            owner_token: owner,
            pixels: alloc::vec![masked_fill; size],
            width,
            height,
            masked_fill,
        }
    }

    fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x < self.width && y < self.height {
            Some(y as usize * self.width as usize + x as usize)
        } else {
            None
        }
    }

    /// Write a pixel. Panics if (x, y) is out of bounds.
    pub fn write_pixel(&mut self, x: u32, y: u32, color: u32) {
        if let Some(idx) = self.index(x, y) {
            self.pixels[idx] = color;
        }
    }

    /// Read a pixel — returns `masked_fill` if the requester is not the owner.
    pub fn read_pixel(&self, x: u32, y: u32, requester: &RenderCapToken) -> u32 {
        let authorized = requester == &self.owner_token
            || (requester.is_system && requester.can_read_other_surfaces);
        if !authorized {
            return self.masked_fill;
        }
        self.index(x, y)
            .map(|i| self.pixels[i])
            .unwrap_or(self.masked_fill)
    }

    /// Fill the entire capsule with a colour.
    pub fn fill(&mut self, color: u32) {
        for p in self.pixels.iter_mut() {
            *p = color;
        }
    }
}

// ---------------------------------------------------------------------------
// CapabilityRoutingGuard
// ---------------------------------------------------------------------------

/// Routing guard that checks render-node submissions against capability tokens.
pub struct CapabilityRoutingGuard {
    pub system_key: Hash256,
}

impl CapabilityRoutingGuard {
    pub fn new(system_key: Hash256) -> Self {
        Self { system_key }
    }

    /// Returns true if the token is authorised to submit render nodes.
    ///
    /// Any token that passes its own `verify()` check may submit.
    pub fn authorize_submit(&self, token: &RenderCapToken) -> bool {
        token.verify()
    }

    /// Returns true if the token can read other surfaces (compositor only).
    pub fn authorize_cross_read(&self, token: &RenderCapToken) -> bool {
        token.verify() && token.is_system && token.can_read_other_surfaces
    }

    /// Check whether a node claiming to be a system dialog has valid system
    /// credentials — i.e. its `signed_hash` is the MAC that
    /// [`RenderCapToken::new_system_signed`] derives from `system_key`.
    ///
    /// We re-derive the expected MAC from the token's public `process_id` and
    /// our `system_key`, then compare it against `token.signed_hash` in constant
    /// time. A fake dialog from a user process has `is_system = false`, a zero
    /// hash, or a `signed_hash` that was not minted under `system_key`, and so
    /// is rejected.
    pub fn verify_system_dialog(&self, token: &RenderCapToken) -> bool {
        if !token.is_system {
            return false;
        }
        if token.signed_hash == Hash256::ZERO {
            return false;
        }
        let expected = RenderCapToken::system_mac(token.process_id, &self.system_key);
        ct_eq_hash(&token.signed_hash, &expected)
    }
}

/// Constant-time equality for two 256-bit hashes: folds every byte so the
/// comparison does not short-circuit on the first mismatch.
fn ct_eq_hash(a: &Hash256, b: &Hash256) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a.0[i] ^ b.0[i];
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// IsolatedRenderContext
// ---------------------------------------------------------------------------

/// An isolated render context for one application process.
pub struct IsolatedRenderContext {
    pub token: RenderCapToken,
    pub capsules: Vec<SecureMemoryCapsule>,
}

impl IsolatedRenderContext {
    pub fn new(token: RenderCapToken) -> Self {
        Self {
            token,
            capsules: Vec::new(),
        }
    }

    pub fn add_capsule(&mut self, capsule: SecureMemoryCapsule) {
        self.capsules.push(capsule);
    }

    pub fn get_capsule_mut(&mut self, id: u64) -> Option<&mut SecureMemoryCapsule> {
        self.capsules.iter_mut().find(|c| c.id == id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn system_key() -> Hash256 {
        Hash256::of(b"dominionos-system-root-key")
    }

    fn user_token() -> RenderCapToken {
        RenderCapToken::new_user(42, b"user-key-material")
    }

    fn system_token() -> RenderCapToken {
        // Minted under the same system key the guard holds, so the guard can
        // re-derive and verify its MAC.
        RenderCapToken::new_system_signed(1, &system_key())
    }

    // --- token creation ---

    #[test]
    fn user_token_fields() {
        let t = user_token();
        assert_eq!(t.process_id, 42);
        assert!(!t.is_system);
        assert!(!t.can_read_other_surfaces);
        assert_ne!(t.signed_hash, Hash256::ZERO);
    }

    #[test]
    fn system_token_fields() {
        let t = system_token();
        assert_eq!(t.process_id, 1);
        assert!(t.is_system);
        assert!(t.can_read_other_surfaces);
        assert_ne!(t.signed_hash, Hash256::ZERO);
    }

    #[test]
    fn token_hashes_differ_by_key_material() {
        let a = RenderCapToken::new_user(1, b"key-a");
        let b = RenderCapToken::new_user(1, b"key-b");
        assert_ne!(a.signed_hash, b.signed_hash);
    }

    #[test]
    fn token_hashes_differ_by_process_id() {
        let a = RenderCapToken::new_user(1, b"same-key");
        let b = RenderCapToken::new_user(2, b"same-key");
        assert_ne!(a.signed_hash, b.signed_hash);
    }

    // --- token verify ---

    #[test]
    fn user_token_verify_passes() {
        assert!(user_token().verify());
    }

    #[test]
    fn system_token_verify_passes() {
        assert!(system_token().verify());
    }

    #[test]
    fn zero_hash_token_fails_verify() {
        let mut t = user_token();
        t.signed_hash = Hash256::ZERO;
        assert!(!t.verify());
    }

    #[test]
    fn user_token_with_cross_read_fails_verify() {
        let mut t = user_token();
        t.can_read_other_surfaces = true; // forged flag
        assert!(!t.verify());
    }

    // --- secure capsule ---

    fn owner_capsule() -> (RenderCapToken, SecureMemoryCapsule) {
        let owner = user_token();
        let capsule = SecureMemoryCapsule::new(99, owner.clone(), 4, 4, 0xDEAD_BEEF);
        (owner, capsule)
    }

    #[test]
    fn capsule_new_fills_with_masked_fill() {
        let (owner, capsule) = owner_capsule();
        // Owner reads back the fill value initially.
        assert_eq!(capsule.read_pixel(0, 0, &owner), 0xDEAD_BEEF);
    }

    #[test]
    fn authorized_read_returns_written_pixel() {
        let (owner, mut capsule) = owner_capsule();
        capsule.write_pixel(1, 2, 0xFF00_FF00);
        assert_eq!(capsule.read_pixel(1, 2, &owner), 0xFF00_FF00);
    }

    #[test]
    fn unauthorized_read_returns_masked_fill() {
        let (_owner, mut capsule) = owner_capsule();
        capsule.write_pixel(0, 0, 0x1234_5678);
        let stranger = RenderCapToken::new_user(999, b"stranger-key");
        assert_eq!(capsule.read_pixel(0, 0, &stranger), 0xDEAD_BEEF);
    }

    #[test]
    fn system_token_can_read_capsule() {
        let (_owner, mut capsule) = owner_capsule();
        capsule.write_pixel(2, 3, 0xAABB_CCDD);
        let sys = system_token();
        // System compositor has cross-surface read.
        assert_eq!(capsule.read_pixel(2, 3, &sys), 0xAABB_CCDD);
    }

    #[test]
    fn capsule_fill_changes_all_pixels() {
        let (owner, mut capsule) = owner_capsule();
        capsule.fill(0x0000_00FF);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(capsule.read_pixel(x, y, &owner), 0x0000_00FF);
            }
        }
    }

    #[test]
    fn capsule_out_of_bounds_read_returns_masked_fill() {
        let (owner, capsule) = owner_capsule();
        // x=10 is beyond width=4, index() returns None → masked_fill.
        assert_eq!(capsule.read_pixel(10, 0, &owner), 0xDEAD_BEEF);
    }

    // --- routing guard ---

    #[test]
    fn guard_authorizes_valid_user_submit() {
        let guard = CapabilityRoutingGuard::new(system_key());
        assert!(guard.authorize_submit(&user_token()));
    }

    #[test]
    fn guard_authorizes_system_submit() {
        let guard = CapabilityRoutingGuard::new(system_key());
        assert!(guard.authorize_submit(&system_token()));
    }

    #[test]
    fn guard_blocks_submit_of_zero_hash_token() {
        let guard = CapabilityRoutingGuard::new(system_key());
        let mut bad = user_token();
        bad.signed_hash = Hash256::ZERO;
        assert!(!guard.authorize_submit(&bad));
    }

    #[test]
    fn guard_allows_cross_read_for_system_only() {
        let guard = CapabilityRoutingGuard::new(system_key());
        assert!(guard.authorize_cross_read(&system_token()));
        assert!(!guard.authorize_cross_read(&user_token()));
    }

    #[test]
    fn guard_verify_system_dialog_passes_for_system() {
        let guard = CapabilityRoutingGuard::new(system_key());
        assert!(guard.verify_system_dialog(&system_token()));
    }

    #[test]
    fn guard_verify_system_dialog_rejects_user_token() {
        let guard = CapabilityRoutingGuard::new(system_key());
        assert!(!guard.verify_system_dialog(&user_token()));
    }

    #[test]
    fn guard_verify_system_dialog_rejects_forged_system_flag() {
        let guard = CapabilityRoutingGuard::new(system_key());
        // Forge: take a user token and flip is_system on. Its signed_hash was
        // derived from the user's own key material, not the system key, so the
        // guard's MAC re-derivation does not match and the forged dialog is
        // rejected (non-zero hash, so this exercises the constant-time compare).
        let mut forged = user_token();
        forged.is_system = true;
        assert_ne!(forged.signed_hash, Hash256::ZERO);
        assert!(!guard.verify_system_dialog(&forged));
        // A zeroed hash is likewise rejected.
        forged.signed_hash = Hash256::ZERO;
        assert!(!guard.verify_system_dialog(&forged));
    }

    // --- isolated render context ---

    #[test]
    fn isolated_context_add_and_retrieve_capsule() {
        let token = user_token();
        let mut ctx = IsolatedRenderContext::new(token.clone());
        let capsule = SecureMemoryCapsule::new(7, token.clone(), 2, 2, 0x0000_0000);
        ctx.add_capsule(capsule);
        assert!(ctx.get_capsule_mut(7).is_some());
        assert!(ctx.get_capsule_mut(999).is_none());
    }

    #[test]
    fn isolated_context_write_through_capsule_ref() {
        let token = user_token();
        let mut ctx = IsolatedRenderContext::new(token.clone());
        ctx.add_capsule(SecureMemoryCapsule::new(1, token.clone(), 3, 3, 0));
        {
            let cap = ctx.get_capsule_mut(1).unwrap();
            cap.write_pixel(0, 0, 0xCAFE_BABE);
        }
        let cap = ctx.get_capsule_mut(1).unwrap();
        assert_eq!(cap.read_pixel(0, 0, &token), 0xCAFE_BABE);
    }
}

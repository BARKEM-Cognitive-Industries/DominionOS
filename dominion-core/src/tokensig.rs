//! Quantum-secure capability tokens — PQ-signed authority over a [`Capability`]
//! (SRS Stage 13; closes the "PQ-signing of tokens not wired" gap).
//!
//! A bare [`Capability`] is unforgeable *within* this machine (its tag is a kernel
//! secret). To carry authority **between** machines it must be signed by something
//! quantum-resistant. Lamport one-time signatures ([`crate::crypto`]) are
//! hash-based and PQ — but each key may sign only once. To sign *many* tokens
//! safely we build the standard fix: an **XMSS-style** scheme where many one-time
//! keys are authenticated by a single **Merkle root**, and that root *is* the
//! authority's long-lived public key. Forgery requires inverting SHA-256, so it
//! stands against quantum attack; security needs only a hash function — **no
//! special hardware**.
//!
//! Pure, safe `no_std`, host-tested.

use crate::capability::Capability;
use crate::crypto::{LamportSig, SignatureScheme};
use crate::hash::Hash256;
use alloc::vec::Vec;

/// Canonical byte encoding of a capability, so signer and verifier agree on exactly
/// what was authorised (fields + provenance + validity).
pub fn canonical(cap: &Capability) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + 8 + 4 + 32 + 1);
    v.extend_from_slice(&cap.base().to_le_bytes());
    v.extend_from_slice(&cap.len().to_le_bytes());
    v.extend_from_slice(&cap.rights().bits().to_le_bytes());
    v.extend_from_slice(&cap.provenance().0);
    v.push(cap.is_valid() as u8);
    v
}

fn merkle_node(a: &Hash256, b: &Hash256) -> Hash256 {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&a.0);
    input[32..].copy_from_slice(&b.0);
    Hash256::of(&input)
}

/// An issuing authority holding `2^height` one-time keys under one Merkle root.
pub struct TokenAuthority {
    lamport: LamportSig,
    master_seed: Vec<u8>,
    leaves: Vec<Hash256>, // hashed one-time public keys
    root: Hash256,
}

/// A PQ token: the capability, which one-time key signed it, the one-time
/// signature, and the Merkle path proving that key belongs to the authority.
#[derive(Clone, Debug)]
pub struct SignedToken {
    pub cap: Capability,
    index: usize,
    ots_sig: Vec<u8>,
    ots_pub: Vec<u8>,
    auth_path: Vec<Hash256>,
}

impl TokenAuthority {
    /// Build an authority with `1 << height` one-time slots from a master seed.
    pub fn new(master_seed: &[u8], height: u32) -> TokenAuthority {
        let lamport = LamportSig::new("lamport-token", "token-pq");
        let n = 1usize << height;
        let mut leaves = Vec::with_capacity(n);
        for i in 0..n {
            let seed = Self::ots_seed(master_seed, i);
            let (_sk, pk) = lamport.keygen(&seed);
            leaves.push(Hash256::of(&pk));
        }
        let root = Self::merkle_root(&leaves);
        TokenAuthority { lamport, master_seed: master_seed.to_vec(), leaves, root }
    }

    fn ots_seed(master: &[u8], index: usize) -> Vec<u8> {
        let mut s = master.to_vec();
        s.extend_from_slice(b":ots:");
        s.extend_from_slice(&(index as u64).to_le_bytes());
        s
    }

    fn merkle_root(leaves: &[Hash256]) -> Hash256 {
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                next.push(merkle_node(&pair[0], &pair[1]));
            }
            level = next;
        }
        level[0]
    }

    /// The authority's public key (the Merkle root).
    pub fn public_key(&self) -> Hash256 {
        self.root
    }

    pub fn capacity(&self) -> usize {
        self.leaves.len()
    }

    /// Sign a capability with the one-time key at `index`. Each index must be used
    /// at most once (the one-time discipline); callers advance the index per token.
    pub fn sign(&self, cap: &Capability, index: usize) -> Option<SignedToken> {
        if index >= self.leaves.len() {
            return None;
        }
        let seed = Self::ots_seed(&self.master_seed, index);
        let (sk, pk) = self.lamport.keygen(&seed);
        let msg = canonical(cap);
        let ots_sig = self.lamport.sign(&sk, &msg);
        let auth_path = self.auth_path(index);
        Some(SignedToken { cap: *cap, index, ots_sig, ots_pub: pk, auth_path })
    }

    fn auth_path(&self, mut index: usize) -> Vec<Hash256> {
        let mut path = Vec::new();
        let mut level = self.leaves.clone();
        while level.len() > 1 {
            let sib = index ^ 1;
            path.push(level[sib]);
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                next.push(merkle_node(&pair[0], &pair[1]));
            }
            index /= 2;
            level = next;
        }
        path
    }
}

/// Verify a signed token against an authority's public key (the Merkle root).
/// Checks the one-time signature *and* that its key is authentically the
/// authority's — both must hold.
pub fn verify(root: Hash256, token: &SignedToken) -> bool {
    let lamport = LamportSig::new("lamport-token", "token-pq");
    // 1. The one-time signature must verify the capability under the claimed key.
    let msg = canonical(&token.cap);
    if !lamport.verify(&token.ots_pub, &msg, &token.ots_sig) {
        return false;
    }
    // 2. The claimed one-time key must chain to the authority root.
    let mut acc = Hash256::of(&token.ots_pub);
    let mut index = token.index;
    for sib in &token.auth_path {
        acc = if index & 1 == 0 {
            merkle_node(&acc, sib)
        } else {
            merkle_node(sib, &acc)
        };
        index /= 2;
    }
    acc == root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Rights;

    #[test]
    fn signed_token_verifies_against_authority_root() {
        let authority = TokenAuthority::new(b"issuer-master-seed", 3); // 8 slots
        assert_eq!(authority.capacity(), 8);
        let cap = Capability::mint(0x1000, 0x200, Rights::READ.union(Rights::WRITE));
        let token = authority.sign(&cap, 0).unwrap();
        assert!(verify(authority.public_key(), &token));
    }

    #[test]
    fn different_one_time_keys_all_chain_to_the_root() {
        let authority = TokenAuthority::new(b"seed", 3);
        // Each slot signs a distinct capability; all verify under one root.
        for i in 0..8 {
            let cap = Capability::mint(i as u64 * 0x1000, 0x100, Rights::READ);
            let token = authority.sign(&cap, i).unwrap();
            assert!(verify(authority.public_key(), &token), "slot {i} failed");
        }
    }

    #[test]
    fn tampered_capability_fails_verification() {
        let authority = TokenAuthority::new(b"seed", 2);
        let cap = Capability::mint(0x1000, 0x100, Rights::READ);
        let mut token = authority.sign(&cap, 1).unwrap();
        // Swap in a capability with wider authority — the signature no longer covers it.
        token.cap = Capability::mint(0, u64::MAX, Rights::ALL);
        assert!(!verify(authority.public_key(), &token));
    }

    #[test]
    fn wrong_authority_root_is_rejected() {
        let real = TokenAuthority::new(b"real-issuer", 2);
        let attacker = TokenAuthority::new(b"attacker", 2);
        let cap = Capability::mint(0x1000, 0x100, Rights::READ);
        let token = real.sign(&cap, 0).unwrap();
        // Verifying against the attacker's root must fail.
        assert!(!verify(attacker.public_key(), &token));
        assert!(verify(real.public_key(), &token));
    }

    #[test]
    fn forged_auth_path_is_rejected() {
        let authority = TokenAuthority::new(b"seed", 2);
        let cap = Capability::mint(0x1000, 0x100, Rights::READ);
        let mut token = authority.sign(&cap, 0).unwrap();
        // Corrupt the Merkle authentication path.
        if let Some(first) = token.auth_path.first_mut() {
            *first = Hash256::of(b"garbage");
        }
        assert!(!verify(authority.public_key(), &token));
    }

    #[test]
    fn out_of_range_index_cannot_be_signed() {
        let authority = TokenAuthority::new(b"seed", 1); // 2 slots
        let cap = Capability::mint(0, 1, Rights::READ);
        assert!(authority.sign(&cap, 2).is_none());
    }
}

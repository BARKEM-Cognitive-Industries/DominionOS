//! Cryptographic Abstraction Layer + post-quantum signatures — **Stage 13**.
//!
//! "No cryptographic algorithm is permanently trusted." Every signature flows
//! through a [`CryptoLayer`]: identity/capability code calls it by *algorithm id*,
//! and the algorithm behind that id can be swapped without touching callers —
//! cryptographic *agility* rather than permanence.
//!
//! The concrete scheme here is a **hash-based one-time signature** (Lamport): its
//! security rests only on the preimage resistance of SHA-256, so it is
//! quantum-resistant — no factoring, discrete-log, or ECC. A [`Hybrid`] wrapper
//! combines two independent schemes so an attacker must break *both* at once
//! (the Harvest-Now-Decrypt-Later defence). In production the pair would be a
//! classical EC scheme + a lattice PQ scheme; here both members are hash-based
//! and domain-separated, which exercises the identical combinator logic.
//!
//! Pure and host-tested. Real key/nonce entropy comes from the kernel TRNG.

use crate::hash::Hash256;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A pluggable signature algorithm.
pub trait SignatureScheme {
    /// Stable algorithm identifier used by the abstraction layer.
    fn id(&self) -> &'static str;
    /// Deterministically derive a `(secret, public)` keypair from `seed`.
    fn keygen(&self, seed: &[u8]) -> (Vec<u8>, Vec<u8>);
    /// Sign `msg` with `secret`.
    fn sign(&self, secret: &[u8], msg: &[u8]) -> Vec<u8>;
    /// Verify `sig` over `msg` against `public`.
    fn verify(&self, public: &[u8], msg: &[u8], sig: &[u8]) -> bool;
}

/// A hash-based one-time signature (Lamport) over the 256-bit message hash.
/// Quantum-resistant: forgery requires inverting SHA-256.
pub struct LamportSig {
    /// Domain-separation tag, so two instances are independent algorithms.
    pub domain: &'static str,
    id: &'static str,
}

impl LamportSig {
    pub fn new(id: &'static str, domain: &'static str) -> LamportSig {
        LamportSig { id, domain }
    }

    fn secret_value(&self, seed: &[u8], index: usize, bit: u8) -> [u8; 32] {
        let mut input = Vec::with_capacity(seed.len() + self.domain.len() + 8);
        input.extend_from_slice(seed);
        input.extend_from_slice(self.domain.as_bytes());
        input.extend_from_slice(b"sk");
        input.extend_from_slice(&(index as u16).to_le_bytes());
        input.push(bit);
        Hash256::of(&input).0
    }

    fn msg_bit(hash: &[u8; 32], i: usize) -> u8 {
        (hash[i / 8] >> (i % 8)) & 1
    }
}

impl SignatureScheme for LamportSig {
    fn id(&self) -> &'static str {
        self.id
    }

    fn keygen(&self, seed: &[u8]) -> (Vec<u8>, Vec<u8>) {
        // Secret key = the seed (secret values are regenerated from it on demand).
        // Public key = the 512 hashes of the secret values.
        let mut public = Vec::with_capacity(256 * 2 * 32);
        for i in 0..256 {
            for bit in 0..2u8 {
                let sv = self.secret_value(seed, i, bit);
                public.extend_from_slice(&Hash256::of(&sv).0);
            }
        }
        (seed.to_vec(), public)
    }

    fn sign(&self, secret: &[u8], msg: &[u8]) -> Vec<u8> {
        let h = Hash256::of(msg).0;
        let mut sig = Vec::with_capacity(256 * 32);
        for i in 0..256 {
            let bit = Self::msg_bit(&h, i);
            sig.extend_from_slice(&self.secret_value(secret, i, bit));
        }
        sig
    }

    fn verify(&self, public: &[u8], msg: &[u8], sig: &[u8]) -> bool {
        if public.len() != 256 * 2 * 32 || sig.len() != 256 * 32 {
            return false;
        }
        let h = Hash256::of(msg).0;
        for i in 0..256 {
            let bit = Self::msg_bit(&h, i) as usize;
            let revealed = &sig[i * 32..i * 32 + 32];
            let computed = Hash256::of(revealed).0;
            let expected = &public[(i * 2 + bit) * 32..(i * 2 + bit) * 32 + 32];
            if computed != expected {
                return false;
            }
        }
        true
    }
}

/// Combines two schemes; a hybrid signature is valid only if *both* members
/// verify. Breaking it requires breaking both algorithm families.
pub struct Hybrid {
    pub classical: Box<dyn SignatureScheme>,
    pub post_quantum: Box<dyn SignatureScheme>,
}

fn put_chunk(out: &mut Vec<u8>, chunk: &[u8]) {
    out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(chunk);
}

fn take_chunk(data: &[u8], pos: &mut usize) -> Option<Vec<u8>> {
    if *pos + 4 > data.len() {
        return None;
    }
    let len = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]) as usize;
    *pos += 4;
    if *pos + len > data.len() {
        return None;
    }
    let out = data[*pos..*pos + len].to_vec();
    *pos += len;
    Some(out)
}

impl Hybrid {
    pub fn keygen(&self, seed: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut cseed = seed.to_vec();
        cseed.extend_from_slice(b":classical");
        let mut qseed = seed.to_vec();
        qseed.extend_from_slice(b":pq");
        let (csk, cpk) = self.classical.keygen(&cseed);
        let (qsk, qpk) = self.post_quantum.keygen(&qseed);
        let mut secret = Vec::new();
        put_chunk(&mut secret, &csk);
        put_chunk(&mut secret, &qsk);
        let mut public = Vec::new();
        put_chunk(&mut public, &cpk);
        put_chunk(&mut public, &qpk);
        (secret, public)
    }

    pub fn sign(&self, secret: &[u8], msg: &[u8]) -> Vec<u8> {
        let mut pos = 0;
        let csk = take_chunk(secret, &mut pos).unwrap_or_default();
        let qsk = take_chunk(secret, &mut pos).unwrap_or_default();
        let mut sig = Vec::new();
        put_chunk(&mut sig, &self.classical.sign(&csk, msg));
        put_chunk(&mut sig, &self.post_quantum.sign(&qsk, msg));
        sig
    }

    pub fn verify(&self, public: &[u8], msg: &[u8], sig: &[u8]) -> bool {
        let mut pp = 0;
        let cpk = match take_chunk(public, &mut pp) {
            Some(v) => v,
            None => return false,
        };
        let qpk = match take_chunk(public, &mut pp) {
            Some(v) => v,
            None => return false,
        };
        let mut sp = 0;
        let csig = match take_chunk(sig, &mut sp) {
            Some(v) => v,
            None => return false,
        };
        let qsig = match take_chunk(sig, &mut sp) {
            Some(v) => v,
            None => return false,
        };
        // BOTH must verify.
        self.classical.verify(&cpk, msg, &csig) && self.post_quantum.verify(&qpk, msg, &qsig)
    }
}

/// The Cryptographic Abstraction Layer: a registry of named schemes through which
/// all signing flows, enabling algorithm agility.
pub struct CryptoLayer {
    schemes: Vec<Box<dyn SignatureScheme>>,
}

impl CryptoLayer {
    pub fn new() -> CryptoLayer {
        CryptoLayer { schemes: Vec::new() }
    }

    /// Preloaded with the built-in schemes.
    pub fn with_defaults() -> CryptoLayer {
        let mut c = CryptoLayer::new();
        c.register(Box::new(LamportSig::new("lamport-classical", "classical")));
        c.register(Box::new(LamportSig::new("lamport-pq", "post-quantum")));
        c
    }

    pub fn register(&mut self, scheme: Box<dyn SignatureScheme>) {
        self.schemes.push(scheme);
    }

    fn by_id(&self, id: &str) -> Option<&dyn SignatureScheme> {
        self.schemes.iter().find(|s| s.id() == id).map(|s| s.as_ref())
    }

    pub fn keygen(&self, id: &str, seed: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        self.by_id(id).map(|s| s.keygen(seed))
    }
    pub fn sign(&self, id: &str, secret: &[u8], msg: &[u8]) -> Option<Vec<u8>> {
        self.by_id(id).map(|s| s.sign(secret, msg))
    }
    pub fn verify(&self, id: &str, public: &[u8], msg: &[u8], sig: &[u8]) -> bool {
        self.by_id(id).map(|s| s.verify(public, msg, sig)).unwrap_or(false)
    }
    pub fn algorithms(&self) -> Vec<&'static str> {
        self.schemes.iter().map(|s| s.id()).collect()
    }
}

impl Default for CryptoLayer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lamport_sign_verify_round_trip() {
        let s = LamportSig::new("lamport-pq", "pq");
        let (sk, pk) = s.keygen(b"seed-material");
        let sig = s.sign(&sk, b"authorize transfer");
        assert!(s.verify(&pk, b"authorize transfer", &sig));
    }

    #[test]
    fn tampered_message_fails() {
        let s = LamportSig::new("lamport-pq", "pq");
        let (sk, pk) = s.keygen(b"seed");
        let sig = s.sign(&sk, b"pay 100");
        assert!(!s.verify(&pk, b"pay 9000", &sig));
    }

    #[test]
    fn forged_signature_fails() {
        let s = LamportSig::new("lamport-pq", "pq");
        let (_sk, pk) = s.keygen(b"seed");
        let forged = alloc::vec![0u8; 256 * 32];
        assert!(!s.verify(&pk, b"msg", &forged));
    }

    #[test]
    fn keygen_is_deterministic() {
        let s = LamportSig::new("x", "d");
        assert_eq!(s.keygen(b"abc").1, s.keygen(b"abc").1);
    }

    #[test]
    fn hybrid_requires_both_schemes() {
        let h = Hybrid {
            classical: Box::new(LamportSig::new("c", "classical")),
            post_quantum: Box::new(LamportSig::new("q", "post-quantum")),
        };
        let (sk, pk) = h.keygen(b"identity-seed");
        let sig = h.sign(&sk, b"capability token");
        assert!(h.verify(&pk, b"capability token", &sig));

        // Corrupt only the PQ half of the signature: the hybrid must reject.
        let mut bad = sig.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        assert!(!h.verify(&pk, b"capability token", &bad));
    }

    #[test]
    fn crypto_layer_agility_swaps_algorithms() {
        let cal = CryptoLayer::with_defaults();
        assert!(cal.algorithms().contains(&"lamport-pq"));
        let (sk, pk) = cal.keygen("lamport-pq", b"k").unwrap();
        let sig = cal.sign("lamport-pq", &sk, b"hello").unwrap();
        assert!(cal.verify("lamport-pq", &pk, b"hello", &sig));
        // A different registered algorithm is independently usable.
        let (sk2, pk2) = cal.keygen("lamport-classical", b"k").unwrap();
        let sig2 = cal.sign("lamport-classical", &sk2, b"hello").unwrap();
        assert!(cal.verify("lamport-classical", &pk2, b"hello", &sig2));
        // Unknown algorithm id verifies nothing.
        assert!(!cal.verify("nonexistent", &pk, b"hello", &sig));
    }
}

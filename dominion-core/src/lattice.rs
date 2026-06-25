//! Lattice post-quantum KEM — a software-modeled **LWE** key-encapsulation
//! mechanism (see `docs/security/post-quantum-cryptography.md`).
//!
//! Stage 13's hash-based signatures ([`crate::crypto`]) cover *signing*; secure
//! *key agreement* needs a KEM whose hardness survives quantum attack. The
//! standardised answer is lattice cryptography (ML-KEM / Kyber). This module
//! implements a faithful **Regev LWE** PKE used as a KEM: hardness rests on the
//! Learning-With-Errors problem, for which no efficient quantum algorithm is known.
//!
//! It is parameter-reduced for clarity and test speed, not production parameters,
//! but the construction is the real thing — `A·s + e`, bit encryption at `⌊q/2⌋`,
//! noisy decryption with a rounding margin — and it round-trips, rejects tampering,
//! and is deterministic under a seed (so it replays). Pure safe `no_std`; **no
//! special hardware** — it is integer arithmetic that runs on any CPU.

use crate::random::Drng;
use alloc::vec;
use alloc::vec::Vec;

/// LWE modulus (a prime large enough that summed noise stays under `q/4`).
const Q: i64 = 7681;
/// Secret dimension.
const N: usize = 128;
/// Number of LWE samples (rows of `A`).
const M: usize = 256;
/// Bits of shared secret encapsulated.
const KEY_BITS: usize = 256;

fn modq(x: i64) -> i64 {
    let r = x % Q;
    if r < 0 {
        r + Q
    } else {
        r
    }
}

/// Centered small noise in `{-1,0,1}` from the DRNG (a tight binomial).
fn noise(rng: &mut Drng) -> i64 {
    // Two coin flips minus two: difference of two Bernoulli(1/2) → {-1,0,1}.
    let a = (rng.next_u64() & 1) as i64;
    let b = (rng.next_u64() & 1) as i64;
    a - b
}

/// An LWE public key: the matrix `A` (m×n) and `b = A·s + e`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey {
    a: Vec<i64>, // row-major m×n
    b: Vec<i64>, // length m
}

impl PublicKey {
    /// Serialize the public key to transportable bytes (length-prefixed `a` then
    /// `b`, each element little-endian `i64`). Used to publish a KEM public key and
    /// to bind it to a self-certifying identity.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + (self.a.len() + self.b.len()) * 8);
        out.extend_from_slice(&(self.a.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.b.len() as u32).to_le_bytes());
        for &x in &self.a {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &self.b {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// The 256-bit fingerprint of the public key — the material a self-certifying
    /// identity (`DominionId`) is the hash of.
    pub fn fingerprint(&self) -> crate::hash::Hash256 {
        crate::hash::Hash256::of(&self.encode())
    }
}

/// An LWE secret key: the vector `s`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretKey {
    s: Vec<i64>, // length n
}

/// A ciphertext encapsulating one shared-secret bit-vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ciphertext {
    u: Vec<i64>, // KEY_BITS × n   (one u-vector per bit)
    v: Vec<i64>, // KEY_BITS
}

/// The lattice KEM. Deterministic given the seeds supplied to each step, so the
/// whole exchange replays bit-for-bit under the deterministic state machine.
pub struct LatticeKem;

impl LatticeKem {
    pub const ALGORITHM: &'static str = "lwe-kem-regev";

    /// Generate a keypair from `seed`. `A` is derived from the seed (a public
    /// "matrix from a hash"), so only `b` and `s` carry secret-dependent data.
    pub fn keygen(seed: &[u8]) -> (PublicKey, SecretKey) {
        let mut pub_rng = Drng::from_seed(seed).derive_stream(b"lattice:A");
        let mut sec_rng = Drng::from_seed(seed).derive_stream(b"lattice:s");
        let mut err_rng = Drng::from_seed(seed).derive_stream(b"lattice:e");

        let s: Vec<i64> = (0..N).map(|_| modq(sec_rng.next_u64() as i64)).collect();
        let mut a = vec![0i64; M * N];
        for slot in a.iter_mut() {
            *slot = modq(pub_rng.next_u64() as i64);
        }
        let mut b = vec![0i64; M];
        for (i, bi) in b.iter_mut().enumerate() {
            let mut acc = 0i64;
            for j in 0..N {
                acc += a[i * N + j] * s[j];
            }
            *bi = modq(acc + noise(&mut err_rng));
        }
        (PublicKey { a, b }, SecretKey { s })
    }

    /// Encapsulate: produce a ciphertext and the shared secret (32 bytes). `seed`
    /// supplies the ephemeral randomness `r` and the secret bits.
    pub fn encapsulate(pk: &PublicKey, seed: &[u8]) -> (Ciphertext, [u8; 32]) {
        let mut r_rng = Drng::from_seed(seed).derive_stream(b"lattice:r");
        let mut k_rng = Drng::from_seed(seed).derive_stream(b"lattice:k");

        // The shared-secret bits.
        let bits: Vec<u8> = (0..KEY_BITS).map(|_| (k_rng.next_u64() & 1) as u8).collect();

        let mut u = vec![0i64; KEY_BITS * N];
        let mut v = vec![0i64; KEY_BITS];
        let half = Q / 2;

        for (bit_idx, &mu) in bits.iter().enumerate() {
            // Fresh r ∈ {0,1}^m for this bit.
            let r: Vec<i64> = (0..M).map(|_| (r_rng.next_u64() & 1) as i64).collect();
            // u = A^T r  (length n)
            for j in 0..N {
                let mut acc = 0i64;
                for (i, &ri) in r.iter().enumerate() {
                    if ri != 0 {
                        acc += pk.a[i * N + j];
                    }
                }
                u[bit_idx * N + j] = modq(acc);
            }
            // v = b^T r + mu·⌊q/2⌋
            let mut vv = 0i64;
            for (i, &ri) in r.iter().enumerate() {
                if ri != 0 {
                    vv += pk.b[i];
                }
            }
            v[bit_idx] = modq(vv + (mu as i64) * half);
        }

        let shared = derive_key(&bits);
        (Ciphertext { u, v }, shared)
    }

    /// Decapsulate: recover the shared secret from a ciphertext using `sk`.
    pub fn decapsulate(sk: &SecretKey, ct: &Ciphertext) -> [u8; 32] {
        let half = Q / 2;
        let quarter = Q / 4;
        let mut bits = vec![0u8; KEY_BITS];
        for (bit_idx, bit) in bits.iter_mut().enumerate() {
            // d = v - s^T u  (mod q); close to q/2 ⇒ bit 1.
            let mut su = 0i64;
            for j in 0..N {
                su += sk.s[j] * ct.u[bit_idx * N + j];
            }
            let d = modq(ct.v[bit_idx] - su);
            // Distance from 0 vs from q/2 (with wraparound) decides the bit.
            let dist_to_half = (d - half).abs();
            *bit = if dist_to_half < quarter { 1 } else { 0 };
        }
        derive_key(&bits)
    }
}

/// Fold the recovered bit-vector into a 256-bit symmetric key (KEM convention:
/// hash the raw secret so any structure is destroyed).
fn derive_key(bits: &[u8]) -> [u8; 32] {
    let mut packed = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        if b != 0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    crate::hash::Hash256::of(&packed).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kem_round_trips() {
        let (pk, sk) = LatticeKem::keygen(b"alice-identity-seed");
        let (ct, shared_enc) = LatticeKem::encapsulate(&pk, b"ephemeral-seed");
        let shared_dec = LatticeKem::decapsulate(&sk, &ct);
        // Both parties derive the identical shared secret.
        assert_eq!(shared_enc, shared_dec);
        // And it is not trivially zero.
        assert_ne!(shared_enc, [0u8; 32]);
    }

    #[test]
    fn keygen_is_deterministic() {
        let (pk1, sk1) = LatticeKem::keygen(b"seed");
        let (pk2, sk2) = LatticeKem::keygen(b"seed");
        assert_eq!(pk1, pk2);
        assert_eq!(sk1, sk2);
    }

    #[test]
    fn different_seeds_give_different_keys() {
        let (pk1, _) = LatticeKem::keygen(b"seed-a");
        let (pk2, _) = LatticeKem::keygen(b"seed-b");
        assert_ne!(pk1, pk2);
    }

    #[test]
    fn wrong_secret_key_fails_to_recover() {
        let (pk, _sk) = LatticeKem::keygen(b"real");
        let (_, wrong_sk) = LatticeKem::keygen(b"attacker");
        let (ct, shared) = LatticeKem::encapsulate(&pk, b"eph");
        let recovered = LatticeKem::decapsulate(&wrong_sk, &ct);
        // An attacker's key does not reconstruct the shared secret.
        assert_ne!(shared, recovered);
    }

    #[test]
    fn tampered_ciphertext_changes_secret() {
        let (pk, sk) = LatticeKem::keygen(b"id");
        let (mut ct, shared) = LatticeKem::encapsulate(&pk, b"eph");
        // Flip the first v term well past the rounding margin.
        ct.v[0] = modq(ct.v[0] + Q / 2);
        let recovered = LatticeKem::decapsulate(&sk, &ct);
        assert_ne!(shared, recovered);
    }

    #[test]
    fn many_exchanges_decode_cleanly() {
        // Exercise the noise margin across several independent exchanges.
        for i in 0..8u8 {
            let (pk, sk) = LatticeKem::keygen(&[i, 0xAA]);
            let (ct, enc) = LatticeKem::encapsulate(&pk, &[i, 0x55]);
            assert_eq!(enc, LatticeKem::decapsulate(&sk, &ct), "exchange {i} mismatch");
        }
    }
}

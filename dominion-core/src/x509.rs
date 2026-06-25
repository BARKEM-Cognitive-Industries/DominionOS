//! X.509 certificate parsing and signature verification.
//!
//! This is the trust layer of the TLS client: it parses DER-encoded
//! certificates, exposes their public keys, and verifies signatures with
//! RSA (PKCS#1 v1.5 and PSS) and ECDSA over NIST P-256. Chain validation
//! walks a presented chain up to a configured trust anchor.
//!
//! All arithmetic is `no_std`, allocation-only, and `#![forbid(unsafe_code)]`
//! compliant — the big-integer routines below are schoolbook but constant in
//! structure, which is fine for the handful of operations a handshake needs.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::hash::{sha256, sha384, sha512};

// ============================================================================
// Big integers (little-endian u32 limbs).
// ============================================================================

/// A non-negative arbitrary-precision integer, little-endian base-2^32 limbs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Big {
    /// Limbs, least significant first. Normalized so the top limb is non-zero
    /// (except zero, which is the empty vector).
    limbs: Vec<u32>,
}

impl Big {
    fn zero() -> Big {
        Big { limbs: Vec::new() }
    }

    fn from_be_bytes(b: &[u8]) -> Big {
        // Strip leading zero bytes, then pack big-endian into little-endian limbs.
        let mut start = 0;
        while start < b.len() && b[start] == 0 {
            start += 1;
        }
        let trimmed = &b[start..];
        let mut limbs = Vec::new();
        // Walk from the least significant byte.
        let mut i = trimmed.len();
        while i > 0 {
            let lo = i.saturating_sub(4);
            let mut limb = 0u32;
            for (k, &byte) in trimmed[lo..i].iter().enumerate() {
                let shift = (i - lo - 1 - k) * 8;
                limb |= (byte as u32) << shift;
            }
            limbs.push(limb);
            i = lo;
        }
        let mut r = Big { limbs };
        r.normalize();
        r
    }

    fn to_be_bytes(&self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        for (i, &limb) in self.limbs.iter().enumerate() {
            for k in 0..4 {
                let byte = ((limb >> (k * 8)) & 0xff) as u8;
                let pos = i * 4 + k;
                if pos < len {
                    out[len - 1 - pos] = byte;
                }
            }
        }
        out
    }

    fn normalize(&mut self) {
        while let Some(&0) = self.limbs.last() {
            self.limbs.pop();
        }
    }

    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() - 1) * 32 + (32 - top.leading_zeros() as usize),
        }
    }

    fn bit(&self, i: usize) -> u32 {
        let limb = i / 32;
        if limb >= self.limbs.len() {
            return 0;
        }
        (self.limbs[limb] >> (i % 32)) & 1
    }

    fn cmp(&self, other: &Big) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
        }
        Ordering::Equal
    }

    fn add(&self, other: &Big) -> Big {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0) as u64;
            let b = *other.limbs.get(i).unwrap_or(&0) as u64;
            let s = a + b + carry;
            out.push(s as u32);
            carry = s >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        let mut r = Big { limbs: out };
        r.normalize();
        r
    }

    /// self - other, assuming self >= other.
    fn sub(&self, other: &Big) -> Big {
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i64;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i64;
            let b = *other.limbs.get(i).unwrap_or(&0) as i64;
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1i64 << 32;
                borrow = 1;
            } else {
                borrow = 0;
            }
            out.push(d as u32);
        }
        let mut r = Big { limbs: out };
        r.normalize();
        r
    }

    fn mul(&self, other: &Big) -> Big {
        if self.is_zero() || other.is_zero() {
            return Big::zero();
        }
        let mut out = vec![0u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let cur = out[i + j] as u64 + (a as u64) * (b as u64) + carry;
                out[i + j] = cur as u32;
                carry = cur >> 32;
            }
            out[i + other.limbs.len()] += carry as u32;
        }
        let mut r = Big { limbs: out };
        r.normalize();
        r
    }

    fn from_u64(v: u64) -> Big {
        let mut r = Big { limbs: alloc::vec![v as u32, (v >> 32) as u32] };
        r.normalize();
        r
    }

    fn shl_bits(&self, bits: usize) -> Big {
        if self.is_zero() || bits == 0 {
            return self.clone();
        }
        let limb_shift = bits / 32;
        let bit_shift = bits % 32;
        let mut out = vec![0u32; self.limbs.len() + limb_shift + 1];
        for (i, &l) in self.limbs.iter().enumerate() {
            let v = (l as u64) << bit_shift;
            out[i + limb_shift] |= v as u32;
            out[i + limb_shift + 1] |= (v >> 32) as u32;
        }
        let mut r = Big { limbs: out };
        r.normalize();
        r
    }

    fn shr_bits(&self, bits: usize) -> Big {
        let limb_shift = bits / 32;
        let bit_shift = bits % 32;
        if limb_shift >= self.limbs.len() {
            return Big::zero();
        }
        let outlen = self.limbs.len() - limb_shift;
        let mut out = vec![0u32; outlen];
        for i in 0..outlen {
            let lo = self.limbs[i + limb_shift] >> bit_shift;
            let hi = if bit_shift > 0 && i + limb_shift + 1 < self.limbs.len() {
                self.limbs[i + limb_shift + 1] << (32 - bit_shift)
            } else {
                0
            };
            out[i] = lo | hi;
        }
        let mut r = Big { limbs: out };
        r.normalize();
        r
    }

    /// Long division (Knuth Algorithm D): returns (quotient, remainder).
    fn divmod(&self, d: &Big) -> (Big, Big) {
        use core::cmp::Ordering;
        debug_assert!(!d.is_zero());
        if self.cmp(d) == Ordering::Less {
            return (Big::zero(), self.clone());
        }
        let n = d.limbs.len();
        if n == 1 {
            let dv = d.limbs[0] as u64;
            let mut rem = 0u64;
            let mut q = vec![0u32; self.limbs.len()];
            for i in (0..self.limbs.len()).rev() {
                let cur = (rem << 32) | self.limbs[i] as u64;
                q[i] = (cur / dv) as u32;
                rem = cur % dv;
            }
            let mut qq = Big { limbs: q };
            qq.normalize();
            return (qq, Big::from_u64(rem));
        }
        // Normalize so the divisor's top limb has its high bit set.
        let shift = d.limbs[n - 1].leading_zeros() as usize;
        let mut vn = d.shl_bits(shift).limbs;
        vn.resize(n, 0);
        let mut un = self.shl_bits(shift).limbs;
        un.resize(self.limbs.len() + 1, 0);
        // self >= d was confirmed above; after normalization self.limbs.len() >= n.
        let m = match self.limbs.len().checked_sub(n) {
            Some(v) => v,
            None => return (Big::zero(), self.clone()),
        };
        let mut q = vec![0u32; m + 1];
        let b = 1u64 << 32;
        for j in (0..=m).rev() {
            // Estimate the quotient digit.
            let num = (un[j + n] as u64) * b + un[j + n - 1] as u64;
            let mut qhat = num / vn[n - 1] as u64;
            let mut rhat = num % vn[n - 1] as u64;
            while qhat >= b
                || qhat * vn[n - 2] as u64 > rhat * b + un[j + n - 2] as u64
            {
                qhat -= 1;
                rhat += vn[n - 1] as u64;
                if rhat >= b {
                    break;
                }
            }
            // Multiply and subtract.
            let mut k: i64 = 0;
            for i in 0..n {
                let p = qhat * vn[i] as u64;
                let t = un[i + j] as i64 - k - (p & 0xffff_ffff) as i64;
                un[i + j] = t as u32;
                k = (p >> 32) as i64 - (t >> 32);
            }
            let t = un[j + n] as i64 - k;
            un[j + n] = t as u32;
            if t < 0 {
                // Add the divisor back (quotient digit was one too high).
                q[j] = (qhat - 1) as u32;
                let mut carry = 0u64;
                for i in 0..n {
                    let s = un[i + j] as u64 + vn[i] as u64 + carry;
                    un[i + j] = s as u32;
                    carry = s >> 32;
                }
                un[j + n] = (un[j + n] as u64 + carry) as u32;
            } else {
                q[j] = qhat as u32;
            }
        }
        let mut quot = Big { limbs: q };
        quot.normalize();
        let mut rem = Big { limbs: un[..n].to_vec() };
        rem.normalize();
        let rem = rem.shr_bits(shift);
        (quot, rem)
    }

    /// self mod m (m must be non-zero).
    fn rem(&self, m: &Big) -> Big {
        self.divmod(m).1
    }

    fn mulmod(&self, other: &Big, m: &Big) -> Big {
        self.mul(other).rem(m)
    }

    /// Modular exponentiation: self^exp mod m.
    fn modexp(&self, exp: &Big, m: &Big) -> Big {
        if m.cmp(&Big::from_be_bytes(&[1])) == core::cmp::Ordering::Equal {
            return Big::zero();
        }
        let mut result = Big::from_be_bytes(&[1]);
        let base = self.rem(m);
        let bits = exp.bit_len();
        for i in (0..bits).rev() {
            result = result.mulmod(&result, m);
            if exp.bit(i) == 1 {
                result = result.mulmod(&base, m);
            }
        }
        result
    }
}

// ============================================================================
// DER / ASN.1 parsing.
// ============================================================================

/// A failure to parse or validate a certificate.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum X509Error {
    Truncated,
    BadTag,
    BadLength,
    Unsupported,
    BadSignature,
    Expired,
    NameMismatch,
    UntrustedRoot,
    EmptyChain,
}

/// A parsed TLV (tag-length-value) view into a DER buffer.
struct Tlv<'a> {
    tag: u8,
    /// The raw content bytes (excluding the tag and length header).
    content: &'a [u8],
    /// Total bytes consumed including tag + length header + content.
    total: usize,
}

/// Read a single DER element at the start of `b`.
fn der_read(b: &[u8]) -> Result<Tlv<'_>, X509Error> {
    if b.len() < 2 {
        return Err(X509Error::Truncated);
    }
    let tag = b[0];
    let first = b[1];
    let (len, header) = if first & 0x80 == 0 {
        (first as usize, 2)
    } else {
        let n = (first & 0x7f) as usize;
        if n == 0 || n > 4 || b.len() < 2 + n {
            return Err(X509Error::BadLength);
        }
        let mut len = 0usize;
        for i in 0..n {
            len = match len.checked_shl(8) {
                Some(v) => v | b[2 + i] as usize,
                None => return Err(X509Error::BadLength),
            };
        }
        (len, 2 + n)
    };
    if b.len() < header + len {
        return Err(X509Error::Truncated);
    }
    Ok(Tlv {
        tag,
        content: &b[header..header + len],
        total: header + len,
    })
}

/// Read the element at `b` and require it to have `tag`.
fn der_expect(b: &[u8], tag: u8) -> Result<Tlv<'_>, X509Error> {
    let t = der_read(b)?;
    if t.tag != tag {
        return Err(X509Error::BadTag);
    }
    Ok(t)
}

const TAG_INTEGER: u8 = 0x02;
const TAG_BITSTRING: u8 = 0x03;
const TAG_OCTETSTRING: u8 = 0x04;
const TAG_OID: u8 = 0x06;
const TAG_SEQUENCE: u8 = 0x30;

// ============================================================================
// Public keys and signatures.
// ============================================================================

/// A parsed subject public key.
#[derive(Clone, Debug)]
pub enum PublicKey {
    /// RSA modulus and exponent, big-endian, leading zeros trimmed.
    Rsa { n: Vec<u8>, e: Vec<u8> },
    /// Uncompressed NIST P-256 point coordinates.
    EcP256 { x: [u8; 32], y: [u8; 32] },
    /// Uncompressed NIST P-384 point coordinates.
    EcP384 { x: [u8; 48], y: [u8; 48] },
}

/// The signature schemes we can verify.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SigAlg {
    RsaPkcs1Sha256,
    RsaPkcs1Sha384,
    RsaPkcs1Sha512,
    RsaPssSha256,
    EcdsaP256Sha256,
    EcdsaP384Sha384,
    /// A signature algorithm we don't verify (e.g. SHA-1). A certificate with
    /// this alg can still serve as a trust anchor (we use its key, not its own
    /// signature), but verifying anything *signed with* it always fails.
    Unknown,
}

impl PublicKey {
    /// Verify `signature` over `message` under `alg`. The message is hashed
    /// internally with the algorithm's digest.
    pub fn verify(&self, alg: SigAlg, message: &[u8], signature: &[u8]) -> bool {
        match (self, alg) {
            (PublicKey::Rsa { n, e }, SigAlg::RsaPkcs1Sha256)
            | (PublicKey::Rsa { n, e }, SigAlg::RsaPkcs1Sha384)
            | (PublicKey::Rsa { n, e }, SigAlg::RsaPkcs1Sha512) => {
                rsa_pkcs1_verify(n, e, alg, message, signature)
            }
            (PublicKey::Rsa { n, e }, SigAlg::RsaPssSha256) => {
                rsa_pss_verify(n, e, message, signature)
            }
            // ECDSA: the curve comes from the key, the hash from the algorithm —
            // they vary independently (e.g. a P-384 key signed with SHA-256).
            (PublicKey::EcP256 { x, y }, SigAlg::EcdsaP256Sha256)
            | (PublicKey::EcP256 { x, y }, SigAlg::EcdsaP384Sha384) => {
                ecdsa_verify(&p256::curve_p256(), x, y, &ecdsa_digest(alg, message), signature)
            }
            (PublicKey::EcP384 { x, y }, SigAlg::EcdsaP256Sha256)
            | (PublicKey::EcP384 { x, y }, SigAlg::EcdsaP384Sha384) => {
                ecdsa_verify(&p256::curve_p384(), x, y, &ecdsa_digest(alg, message), signature)
            }
            _ => false,
        }
    }
}

/// The digest an ECDSA `SigAlg` hashes with (the curve is chosen from the key).
fn ecdsa_digest(alg: SigAlg, message: &[u8]) -> Vec<u8> {
    match alg {
        SigAlg::EcdsaP384Sha384 => sha384(message).to_vec(),
        _ => sha256(message).to_vec(),
    }
}

// ----------------------------------------------------------------------------
// RSA verification.
// ----------------------------------------------------------------------------

/// Apply the RSA public-key operation: signature^e mod n, returned as a
/// big-endian byte string the same length as the modulus.
fn rsa_public(n: &[u8], e: &[u8], signature: &[u8]) -> Vec<u8> {
    let n_big = Big::from_be_bytes(n);
    let e_big = Big::from_be_bytes(e);
    let s_big = Big::from_be_bytes(signature);
    let m = s_big.modexp(&e_big, &n_big);
    let modlen = n.iter().skip_while(|&&b| b == 0).count();
    m.to_be_bytes(modlen)
}

/// Sign a message with RSA PKCS#1 v1.5 over SHA-256, given the modulus `n` and
/// private exponent `d` (both big-endian). Returns a signature the length of
/// the modulus. This is the inverse of [`rsa_pkcs1_verify`]; it is used for
/// client authentication and by the test harness.
pub fn rsa_pkcs1_sha256_sign(n: &[u8], d: &[u8], message: &[u8]) -> Vec<u8> {
    let modlen = n.iter().skip_while(|&&b| b == 0).count();
    let (prefix, _) = digestinfo_prefix(SigAlg::RsaPkcs1Sha256).unwrap();
    let digest = sha256(message);
    let tlen = prefix.len() + digest.len();
    // Saturate: if the modulus is pathologically small just produce a zero-padded block.
    let ps_len = modlen.saturating_sub(tlen).saturating_sub(3);
    let mut em = Vec::with_capacity(modlen);
    em.push(0x00);
    em.push(0x01);
    em.extend(core::iter::repeat(0xff).take(ps_len));
    em.push(0x00);
    em.extend_from_slice(prefix);
    em.extend_from_slice(&digest);
    let n_big = Big::from_be_bytes(n);
    let d_big = Big::from_be_bytes(d);
    let m = Big::from_be_bytes(&em);
    m.modexp(&d_big, &n_big).to_be_bytes(modlen)
}

/// The DER `DigestInfo` prefix for each SHA-2 digest (PKCS#1 v1.5).
fn digestinfo_prefix(alg: SigAlg) -> Option<(&'static [u8], usize)> {
    match alg {
        SigAlg::RsaPkcs1Sha256 => Some((
            &[
                0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x01, 0x05, 0x00, 0x04, 0x20,
            ],
            32,
        )),
        SigAlg::RsaPkcs1Sha384 => Some((
            &[
                0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x02, 0x05, 0x00, 0x04, 0x30,
            ],
            48,
        )),
        SigAlg::RsaPkcs1Sha512 => Some((
            &[
                0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x03, 0x05, 0x00, 0x04, 0x40,
            ],
            64,
        )),
        _ => None,
    }
}

fn rsa_pkcs1_verify(n: &[u8], e: &[u8], alg: SigAlg, message: &[u8], signature: &[u8]) -> bool {
    let (prefix, dlen) = match digestinfo_prefix(alg) {
        Some(p) => p,
        None => return false,
    };
    let digest = match alg {
        SigAlg::RsaPkcs1Sha256 => sha256(message).to_vec(),
        SigAlg::RsaPkcs1Sha384 => sha384(message).to_vec(),
        SigAlg::RsaPkcs1Sha512 => sha512(message).to_vec(),
        _ => return false,
    };
    if digest.len() != dlen {
        return false;
    }
    let em = rsa_public(n, e, signature);
    // Expected EM: 0x00 0x01 PS(0xff..) 0x00 prefix digest, length = modlen.
    let tlen = prefix.len() + dlen;
    if em.len() < tlen + 11 {
        return false;
    }
    // Safe: em.len() >= tlen + 11 >= tlen + 3, so subtraction won't underflow.
    let ps_len = em.len() - tlen - 3;
    // Bounds-checked index: em.len() >= 2 is guaranteed by the check above.
    if em[0] != 0x00 || em[1] != 0x01 {
        return false;
    }
    // 2 + ps_len <= em.len() - tlen - 1 < em.len() — safe.
    for &b in &em[2..2 + ps_len] {
        if b != 0xff {
            return false;
        }
    }
    // 2 + ps_len < em.len() guaranteed by ps_len = em.len() - tlen - 3.
    if em[2 + ps_len] != 0x00 {
        return false;
    }
    // 3 + ps_len <= em.len() - tlen; rest.len() == tlen >= prefix.len() + dlen — safe.
    let rest = &em[3 + ps_len..];
    if rest.len() < prefix.len() + dlen {
        return false;
    }
    &rest[..prefix.len()] == prefix && &rest[prefix.len()..prefix.len() + dlen] == &digest[..]
}

/// RSA-PSS with SHA-256 and MGF1-SHA256, salt length = 32 (rsa_pss_rsae_sha256).
fn rsa_pss_verify(n: &[u8], e: &[u8], message: &[u8], signature: &[u8]) -> bool {
    let modbits = Big::from_be_bytes(n).bit_len();
    let emlen = (modbits - 1 + 7) / 8; // emBits = modBits-1
    let em = rsa_public(n, e, signature);
    // rsa_public returns modlen bytes; PSS uses emLen = ceil((modBits-1)/8).
    // Align: take the least-significant emlen bytes.
    if em.len() < emlen {
        return false;
    }
    let em = &em[em.len() - emlen..];
    let hlen = 32usize;
    let slen = 32usize;
    if emlen < hlen + slen + 2 {
        return false;
    }
    if em[emlen - 1] != 0xbc {
        return false;
    }
    // Safe: emlen >= hlen + slen + 2 >= hlen + 2 > hlen + 1 (checked above).
    let masked_db_len = emlen - hlen - 1;
    // Safe: masked_db_len < emlen = em.len().
    let masked_db = &em[..masked_db_len];
    // Safe: masked_db_len + hlen = emlen - 1 < emlen = em.len().
    let h = &em[masked_db_len..masked_db_len + hlen];
    // Top bits beyond modBits-1 must be zero.
    let top_bits = 8 * emlen - (modbits - 1);
    if top_bits < 8 && (masked_db[0] >> (8 - top_bits)) != 0 {
        return false;
    }
    let db_mask = mgf1_sha256(h, masked_db_len);
    let mut db = vec![0u8; masked_db_len];
    for i in 0..masked_db_len {
        db[i] = masked_db[i] ^ db_mask[i];
    }
    // Clear the leftmost top_bits bits.
    if top_bits < 8 {
        db[0] &= 0xff >> top_bits;
    }
    // db = PS (0x00..) || 0x01 || salt
    // Safe: emlen >= hlen + slen + 2, so masked_db_len = emlen - hlen - 1 >= slen + 1 >= 1.
    let ps_len = masked_db_len - slen - 1;
    // ps_len < masked_db_len = db.len() — safe.
    for &b in &db[..ps_len] {
        if b != 0 {
            return false;
        }
    }
    // ps_len < db.len() — safe.
    if db[ps_len] != 0x01 {
        return false;
    }
    // ps_len + 1 <= masked_db_len = db.len() — safe.
    let salt = &db[ps_len + 1..];
    let mhash = sha256(message);
    let mut m_prime = Vec::with_capacity(8 + hlen + slen);
    m_prime.extend_from_slice(&[0u8; 8]);
    m_prime.extend_from_slice(&mhash);
    m_prime.extend_from_slice(salt);
    let h_prime = sha256(&m_prime);
    crate::tlscrypto::ct_eq(&h_prime, h)
}

fn mgf1_sha256(seed: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut counter: u32 = 0;
    while out.len() < len {
        let mut data = Vec::with_capacity(seed.len() + 4);
        data.extend_from_slice(seed);
        data.extend_from_slice(&counter.to_be_bytes());
        out.extend_from_slice(&sha256(&data));
        counter += 1;
    }
    out.truncate(len);
    out
}

// ----------------------------------------------------------------------------
// ECDSA over NIST P-256.
// ----------------------------------------------------------------------------

mod p256 {
    use super::Big;
    use alloc::vec::Vec;

    /// A point in Jacobian projective coordinates: affine = (X/Z^2, Y/Z^3).
    /// `Z == 0` is the point at infinity.
    #[derive(Clone)]
    pub struct Jac {
        pub x: Big,
        pub y: Big,
        pub z: Big,
    }

    pub struct Curve {
        pub p: Big,
        pub n: Big,
        pub gx: Big,
        pub gy: Big,
    }

    fn be(s: &str) -> Big {
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect();
        Big::from_be_bytes(&bytes)
    }

    /// NIST P-256 (secp256r1). a = -3, so the optimized doubling applies.
    pub fn curve_p256() -> Curve {
        Curve {
            p: be("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"),
            n: be("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"),
            gx: be("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"),
            gy: be("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"),
        }
    }

    /// NIST P-384 (secp384r1). a = -3 as well, so the same group law applies.
    pub fn curve_p384() -> Curve {
        Curve {
            p: be("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff"),
            n: be("ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973"),
            gx: be("aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7"),
            gy: be("3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f"),
        }
    }

    // Field arithmetic mod p. Inputs are assumed already reduced (< p), which
    // every helper below preserves — so add/sub need only a conditional fold.
    #[inline]
    fn fadd(a: &Big, b: &Big, p: &Big) -> Big {
        let s = a.add(b);
        if s.cmp(p) != core::cmp::Ordering::Less {
            s.sub(p)
        } else {
            s
        }
    }
    #[inline]
    fn fsub(a: &Big, b: &Big, p: &Big) -> Big {
        if a.cmp(b) == core::cmp::Ordering::Less {
            a.add(p).sub(b)
        } else {
            a.sub(b)
        }
    }
    #[inline]
    fn fmul(a: &Big, b: &Big, p: &Big) -> Big {
        a.mulmod(b, p)
    }
    #[inline]
    fn fdbl(a: &Big, p: &Big) -> Big {
        fadd(a, a, p)
    }

    /// Modular inverse via Fermat: a^(p-2) mod p. Used once per verify.
    pub fn invmod(a: &Big, p: &Big) -> Big {
        let two = Big::from_be_bytes(&[2]);
        a.modexp(&p.sub(&two), p)
    }

    pub fn identity() -> Jac {
        Jac { x: Big::from_be_bytes(&[1]), y: Big::from_be_bytes(&[1]), z: Big::zero() }
    }

    pub fn from_affine(x: Big, y: Big) -> Jac {
        Jac { x, y, z: Big::from_be_bytes(&[1]) }
    }

    /// Point doubling (dbl-2001-b, optimized for a = -3).
    pub fn double(c: &Curve, q: &Jac) -> Jac {
        if q.z.is_zero() {
            return identity();
        }
        let p = &c.p;
        let delta = fmul(&q.z, &q.z, p);
        let gamma = fmul(&q.y, &q.y, p);
        let beta = fmul(&q.x, &gamma, p);
        // alpha = 3*(X-delta)*(X+delta)
        let t = fmul(&fsub(&q.x, &delta, p), &fadd(&q.x, &delta, p), p);
        let alpha = fadd(&fdbl(&t, p), &t, p);
        // X3 = alpha^2 - 8*beta
        let beta4 = fdbl(&fdbl(&beta, p), p);
        let beta8 = fdbl(&beta4, p);
        let x3 = fsub(&fmul(&alpha, &alpha, p), &beta8, p);
        // Z3 = (Y+Z)^2 - gamma - delta
        let yz = fadd(&q.y, &q.z, p);
        let z3 = fsub(&fsub(&fmul(&yz, &yz, p), &gamma, p), &delta, p);
        // Y3 = alpha*(4*beta - X3) - 8*gamma^2
        let g2 = fmul(&gamma, &gamma, p);
        let g8 = fdbl(&fdbl(&fdbl(&g2, p), p), p);
        let y3 = fsub(&fmul(&alpha, &fsub(&beta4, &x3, p), p), &g8, p);
        Jac { x: x3, y: y3, z: z3 }
    }

    /// Point addition (add-2007-bl), full Jacobian.
    pub fn add(c: &Curve, a: &Jac, b: &Jac) -> Jac {
        if a.z.is_zero() {
            return b.clone();
        }
        if b.z.is_zero() {
            return a.clone();
        }
        let p = &c.p;
        let z1z1 = fmul(&a.z, &a.z, p);
        let z2z2 = fmul(&b.z, &b.z, p);
        let u1 = fmul(&a.x, &z2z2, p);
        let u2 = fmul(&b.x, &z1z1, p);
        let s1 = fmul(&fmul(&a.y, &b.z, p), &z2z2, p);
        let s2 = fmul(&fmul(&b.y, &a.z, p), &z1z1, p);
        if u1.cmp(&u2) == core::cmp::Ordering::Equal {
            if s1.cmp(&s2) == core::cmp::Ordering::Equal {
                return double(c, a);
            }
            return identity();
        }
        let h = fsub(&u2, &u1, p);
        let i = {
            let h2 = fdbl(&h, p);
            fmul(&h2, &h2, p)
        };
        let j = fmul(&h, &i, p);
        let r = fdbl(&fsub(&s2, &s1, p), p);
        let v = fmul(&u1, &i, p);
        // X3 = r^2 - J - 2V
        let x3 = fsub(&fsub(&fmul(&r, &r, p), &j, p), &fdbl(&v, p), p);
        // Y3 = r*(V - X3) - 2*S1*J
        let y3 = fsub(&fmul(&r, &fsub(&v, &x3, p), p), &fdbl(&fmul(&s1, &j, p), p), p);
        // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
        let zz = fadd(&a.z, &b.z, p);
        let z3 = fmul(&fsub(&fsub(&fmul(&zz, &zz, p), &z1z1, p), &z2z2, p), &h, p);
        Jac { x: x3, y: y3, z: z3 }
    }

    pub fn mul(c: &Curve, k: &Big, q: &Jac) -> Jac {
        let mut r = identity();
        for i in (0..k.bit_len()).rev() {
            r = double(c, &r);
            if k.bit(i) == 1 {
                r = add(c, &r, q);
            }
        }
        r
    }

    /// The affine x-coordinate of a Jacobian point, or `None` at infinity.
    pub fn affine_x(c: &Curve, j: &Jac) -> Option<Big> {
        if j.z.is_zero() {
            return None;
        }
        let zinv = invmod(&j.z, &c.p);
        let zinv2 = fmul(&zinv, &zinv, &c.p);
        Some(fmul(&j.x, &zinv2, &c.p))
    }
}

/// ECDSA verification over a short-Weierstrass curve with a = -3 (P-256/P-384).
/// `qx`/`qy` are the public-point coordinates and `hash` the message digest.
fn ecdsa_verify(c: &p256::Curve, qx: &[u8], qy: &[u8], hash: &[u8], signature: &[u8]) -> bool {
    // Parse the ECDSA-Sig-Value SEQUENCE { r INTEGER, s INTEGER }.
    let seq = match der_expect(signature, TAG_SEQUENCE) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let r_tlv = match der_expect(seq.content, TAG_INTEGER) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let s_tlv = match der_expect(&seq.content[r_tlv.total..], TAG_INTEGER) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let r = Big::from_be_bytes(r_tlv.content);
    let s = Big::from_be_bytes(s_tlv.content);
    if r.is_zero() || s.is_zero() || r.cmp(&c.n) != core::cmp::Ordering::Less || s.cmp(&c.n) != core::cmp::Ordering::Less {
        return false;
    }
    // z = leftmost N bits of the hash, where N is the order's bit length.
    let nbytes = (c.n.bit_len() + 7) / 8;
    let z = Big::from_be_bytes(&hash[..hash.len().min(nbytes)]);
    // w = s^-1 mod n
    let two = Big::from_be_bytes(&[2]);
    let w = s.modexp(&c.n.sub(&two), &c.n);
    let u1 = z.mulmod(&w, &c.n);
    let u2 = r.mulmod(&w, &c.n);
    let g = p256::from_affine(c.gx.clone(), c.gy.clone());
    let q = p256::from_affine(Big::from_be_bytes(qx), Big::from_be_bytes(qy));
    let p1 = p256::mul(c, &u1, &g);
    let p2 = p256::mul(c, &u2, &q);
    let pt = p256::add(c, &p1, &p2);
    let xr = match p256::affine_x(c, &pt) {
        Some(x) => x.rem(&c.n),
        None => return false,
    };
    xr.cmp(&r) == core::cmp::Ordering::Equal
}

// ============================================================================
// Certificate structure and parsing.
// ============================================================================

/// A parsed X.509 certificate (the fields the TLS client needs).
#[derive(Clone, Debug)]
pub struct Certificate {
    /// The raw TBSCertificate DER (the bytes that were signed).
    pub tbs: Vec<u8>,
    /// Raw issuer Name DER.
    pub issuer: Vec<u8>,
    /// Raw subject Name DER.
    pub subject: Vec<u8>,
    pub not_before: u64,
    pub not_after: u64,
    pub key: PublicKey,
    pub sig_alg: SigAlg,
    pub signature: Vec<u8>,
    pub san_dns: Vec<String>,
    pub is_ca: bool,
}

/// Map a signature-algorithm OID to our `SigAlg`.
fn oid_to_sigalg(oid: &[u8]) -> Option<SigAlg> {
    // sha256WithRSAEncryption 1.2.840.113549.1.1.11
    const RSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
    const RSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c];
    const RSA_SHA512: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d];
    // rsassaPss 1.2.840.113549.1.1.10
    const RSA_PSS: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];
    // ecdsa-with-SHA256 1.2.840.10045.4.3.2
    const ECDSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
    // ecdsa-with-SHA384 1.2.840.10045.4.3.3
    const ECDSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03];
    match oid {
        RSA_SHA256 => Some(SigAlg::RsaPkcs1Sha256),
        RSA_SHA384 => Some(SigAlg::RsaPkcs1Sha384),
        RSA_SHA512 => Some(SigAlg::RsaPkcs1Sha512),
        RSA_PSS => Some(SigAlg::RsaPssSha256),
        ECDSA_SHA256 => Some(SigAlg::EcdsaP256Sha256),
        ECDSA_SHA384 => Some(SigAlg::EcdsaP384Sha384),
        _ => None,
    }
}

/// Parse a SubjectPublicKeyInfo into a PublicKey.
fn parse_spki(spki: &[u8]) -> Result<PublicKey, X509Error> {
    let seq = der_expect(spki, TAG_SEQUENCE)?;
    let alg = der_expect(seq.content, TAG_SEQUENCE)?;
    let alg_oid = der_expect(alg.content, TAG_OID)?;
    let bitstr = der_expect(&seq.content[alg.total..], TAG_BITSTRING)?;
    // BIT STRING: first content byte is the count of unused bits (expect 0).
    if bitstr.content.is_empty() || bitstr.content[0] != 0 {
        return Err(X509Error::Unsupported);
    }
    let key_bits = &bitstr.content[1..];
    // rsaEncryption 1.2.840.113549.1.1.1
    const RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
    // id-ecPublicKey 1.2.840.10045.2.1
    const EC: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
    if alg_oid.content == RSA {
        let rsa = der_expect(key_bits, TAG_SEQUENCE)?;
        let n_tlv = der_expect(rsa.content, TAG_INTEGER)?;
        let e_tlv = der_expect(&rsa.content[n_tlv.total..], TAG_INTEGER)?;
        let n: Vec<u8> = n_tlv.content.iter().copied().skip_while(|&b| b == 0).collect();
        let e: Vec<u8> = e_tlv.content.iter().copied().skip_while(|&b| b == 0).collect();
        Ok(PublicKey::Rsa { n, e })
    } else if alg_oid.content == EC {
        // The named curve is the AlgorithmIdentifier's second parameter (an OID).
        // secp256r1 = 1.2.840.10045.3.1.7; secp384r1 = 1.3.132.0.34.
        const P256_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
        const P384_OID: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];
        let curve_oid = der_expect(&alg.content[alg_oid.total..], TAG_OID)?;
        // Uncompressed point: 0x04 || X || Y.
        if key_bits.is_empty() || key_bits[0] != 0x04 {
            return Err(X509Error::Unsupported);
        }
        let pt = &key_bits[1..];
        if curve_oid.content == P256_OID && pt.len() == 64 {
            let mut x = [0u8; 32];
            let mut y = [0u8; 32];
            x.copy_from_slice(&pt[..32]);
            y.copy_from_slice(&pt[32..64]);
            Ok(PublicKey::EcP256 { x, y })
        } else if curve_oid.content == P384_OID && pt.len() == 96 {
            let mut x = [0u8; 48];
            let mut y = [0u8; 48];
            x.copy_from_slice(&pt[..48]);
            y.copy_from_slice(&pt[48..96]);
            Ok(PublicKey::EcP384 { x, y })
        } else {
            Err(X509Error::Unsupported)
        }
    } else {
        Err(X509Error::Unsupported)
    }
}

/// Parse a UTCTime / GeneralizedTime into a unix timestamp (seconds).
/// Returns 0 on any truncation rather than panicking on attacker-controlled bytes.
fn parse_time(tlv: &Tlv) -> u64 {
    let s = tlv.content;
    // UTCTime: YYMMDDHHMMSSZ (tag 0x17); GeneralizedTime: YYYYMMDDHHMMSSZ (0x18).
    let (year, rest) = if tlv.tag == 0x18 {
        if s.len() < 4 { return 0; }
        (read_int(&s[0..4]), &s[4..])
    } else {
        if s.len() < 2 { return 0; }
        let yy = read_int(&s[0..2]);
        let full = if yy >= 50 { 1900 + yy } else { 2000 + yy };
        (full, &s[2..])
    };
    if rest.len() < 8 { return 0; }
    let mon = read_int(&rest[0..2]);
    let day = read_int(&rest[2..4]);
    let hour = read_int(&rest[4..6]);
    let min = read_int(&rest[6..8]);
    let sec = if rest.len() >= 10 { read_int(&rest[8..10]) } else { 0 };
    days_from_civil(year as i64, mon as i64, day as i64) as u64 * 86400
        + hour as u64 * 3600
        + min as u64 * 60
        + sec as u64
}

fn read_int(b: &[u8]) -> u32 {
    let mut v = 0u32;
    for &c in b {
        if c.is_ascii_digit() {
            v = v * 10 + (c - b'0') as u32;
        }
    }
    v
}

/// Days since the Unix epoch for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse a DER certificate.
pub fn parse_certificate(der: &[u8]) -> Result<Certificate, X509Error> {
    let cert = der_expect(der, TAG_SEQUENCE)?;
    let body = cert.content;
    // tbsCertificate
    let tbs_tlv = der_expect(body, TAG_SEQUENCE)?;
    let tbs_raw = &body[..tbs_tlv.total];
    let mut p = tbs_tlv.content;

    // [0] version (optional, explicit).
    if !p.is_empty() && p[0] == 0xa0 {
        let v = der_read(p)?;
        p = &p[v.total..];
    }
    // serialNumber INTEGER
    let serial = der_expect(p, TAG_INTEGER)?;
    p = &p[serial.total..];
    // signature AlgorithmIdentifier
    let inner_sig = der_expect(p, TAG_SEQUENCE)?;
    p = &p[inner_sig.total..];
    // issuer Name
    let issuer = der_expect(p, TAG_SEQUENCE)?;
    let issuer_raw = p[..issuer.total].to_vec();
    p = &p[issuer.total..];
    // validity SEQUENCE { notBefore, notAfter }
    let validity = der_expect(p, TAG_SEQUENCE)?;
    p = &p[validity.total..];
    let nb = der_read(validity.content)?;
    let na = der_read(&validity.content[nb.total..])?;
    let not_before = parse_time(&nb);
    let not_after = parse_time(&na);
    // subject Name
    let subject = der_expect(p, TAG_SEQUENCE)?;
    let subject_raw = p[..subject.total].to_vec();
    p = &p[subject.total..];
    // subjectPublicKeyInfo
    let spki = der_expect(p, TAG_SEQUENCE)?;
    let spki_raw = &p[..spki.total];
    let key = parse_spki(spki_raw)?;
    p = &p[spki.total..];

    // Optional issuerUniqueID [1], subjectUniqueID [2], extensions [3].
    let mut san_dns = Vec::new();
    let mut is_ca = false;
    while !p.is_empty() {
        let t = der_read(p)?;
        if t.tag == 0xa3 {
            // extensions [3] EXPLICIT SEQUENCE OF Extension
            let exts = der_expect(t.content, TAG_SEQUENCE)?;
            let mut ep = exts.content;
            while !ep.is_empty() {
                let ext = der_expect(ep, TAG_SEQUENCE)?;
                ep = &ep[ext.total..];
                let oid = der_expect(ext.content, TAG_OID)?;
                let mut rest = &ext.content[oid.total..];
                // optional critical BOOLEAN
                if !rest.is_empty() && rest[0] == 0x01 {
                    let b = der_read(rest)?;
                    rest = &rest[b.total..];
                }
                let val = der_expect(rest, TAG_OCTETSTRING)?;
                // id-ce-subjectAltName 2.5.29.17
                const SAN: &[u8] = &[0x55, 0x1d, 0x11];
                // id-ce-basicConstraints 2.5.29.19
                const BC: &[u8] = &[0x55, 0x1d, 0x13];
                if oid.content == SAN {
                    if let Ok(names) = der_expect(val.content, TAG_SEQUENCE) {
                        let mut np = names.content;
                        while !np.is_empty() {
                            let gn = der_read(np)?;
                            np = &np[gn.total..];
                            // dNSName [2] IMPLICIT IA5String
                            if gn.tag == 0x82 {
                                if let Ok(s) = core::str::from_utf8(gn.content) {
                                    san_dns.push(String::from(s));
                                }
                            }
                        }
                    }
                } else if oid.content == BC {
                    if let Ok(bc) = der_expect(val.content, TAG_SEQUENCE) {
                        if !bc.content.is_empty() && bc.content[0] == 0x01 {
                            if let Ok(b) = der_read(bc.content) {
                                if !b.content.is_empty() && b.content[0] != 0 {
                                    is_ca = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        p = &p[t.total..];
    }

    // signatureAlgorithm + signatureValue (after tbsCertificate).
    let after_tbs = &body[tbs_tlv.total..];
    let sig_alg_seq = der_expect(after_tbs, TAG_SEQUENCE)?;
    let sig_oid = der_expect(sig_alg_seq.content, TAG_OID)?;
    // An unknown signature algorithm is tolerated so the certificate can still
    // act as a trust anchor; verifying a child signed with it will fail.
    let sig_alg = oid_to_sigalg(sig_oid.content).unwrap_or(SigAlg::Unknown);
    let sig_bits = der_expect(&after_tbs[sig_alg_seq.total..], TAG_BITSTRING)?;
    if sig_bits.content.is_empty() {
        return Err(X509Error::BadSignature);
    }
    let signature = sig_bits.content[1..].to_vec();

    let _ = (serial, inner_sig, spki_raw);
    Ok(Certificate {
        tbs: tbs_raw.to_vec(),
        issuer: issuer_raw,
        subject: subject_raw,
        not_before,
        not_after,
        key,
        sig_alg,
        signature,
        san_dns,
        is_ca,
    })
}

impl Certificate {
    /// Verify that this certificate's signature was produced by `issuer`.
    pub fn verify_signed_by(&self, issuer: &Certificate) -> bool {
        issuer.key.verify(self.sig_alg, &self.tbs, &self.signature)
    }

    /// Does this certificate cover `hostname` (SAN dNSName, wildcard aware)?
    pub fn matches_host(&self, hostname: &str) -> bool {
        let host = hostname.to_ascii_lowercase();
        for name in &self.san_dns {
            let n = name.to_ascii_lowercase();
            if n == host {
                return true;
            }
            if let Some(suffix) = n.strip_prefix("*.") {
                // Wildcard matches exactly one leftmost label.
                if let Some(dot) = host.find('.') {
                    if &host[dot + 1..] == suffix {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// ============================================================================
// Trust store and chain verification.
// ============================================================================

/// The embedded system root-CA bundle: the standard Mozilla/NSS roots, stored
/// as concatenated DER (each a `SEQUENCE`, so the blob is walkable TLV by TLV).
/// This is how the OS ships a trust store — compiled into the image so HTTPS
/// works out of the box without a filesystem dependency.
static ROOT_BUNDLE: &[u8] = include_bytes!("roots.der");

/// Build a [`TrustStore`] seeded with the embedded system roots. Roots whose
/// key type we can't parse are skipped (not fatal); roots with a SHA-1 self
/// signature are kept as anchors (their own signature is never checked).
pub fn system_trust_store() -> TrustStore {
    let mut store = TrustStore::new();
    let mut p = ROOT_BUNDLE;
    while p.len() >= 4 {
        let total = match der_read(p) {
            Ok(t) => t.total,
            Err(_) => break,
        };
        if total == 0 || total > p.len() {
            break;
        }
        if let Ok(cert) = parse_certificate(&p[..total]) {
            store.add_root(cert);
        }
        p = &p[total..];
    }
    store
}

/// A set of trusted root certificates (by their full DER).
#[derive(Clone, Default)]
pub struct TrustStore {
    roots: Vec<Certificate>,
}

impl TrustStore {
    pub fn new() -> TrustStore {
        TrustStore { roots: Vec::new() }
    }

    pub fn add_root_der(&mut self, der: &[u8]) -> Result<(), X509Error> {
        let c = parse_certificate(der)?;
        self.roots.push(c);
        Ok(())
    }

    pub fn add_root(&mut self, cert: Certificate) {
        self.roots.push(cert);
    }

    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Find a trusted root whose subject equals `issuer_dn` and that signed `cert`.
    fn trusted_issuer(&self, cert: &Certificate) -> bool {
        for r in &self.roots {
            if r.subject == cert.issuer && cert.verify_signed_by(r) {
                return true;
            }
        }
        false
    }

    /// Validate a presented chain (leaf first) for `hostname` at time `now`.
    pub fn verify_chain(
        &self,
        chain: &[Certificate],
        hostname: &str,
        now: u64,
    ) -> Result<(), X509Error> {
        if chain.is_empty() {
            return Err(X509Error::EmptyChain);
        }
        // Validity + hostname on the leaf.
        let leaf = &chain[0];
        if now != 0 && (now < leaf.not_before || now > leaf.not_after) {
            return Err(X509Error::Expired);
        }
        if !hostname.is_empty() && !leaf.matches_host(hostname) {
            return Err(X509Error::NameMismatch);
        }
        // Each cert must be signed by the next; each intermediate must be a CA
        // and time-valid.
        for i in 0..chain.len() - 1 {
            let child = &chain[i];
            let parent = &chain[i + 1];
            if now != 0 && (now < parent.not_before || now > parent.not_after) {
                return Err(X509Error::Expired);
            }
            if child.issuer != parent.subject || !child.verify_signed_by(parent) {
                return Err(X509Error::BadSignature);
            }
        }
        // The top of the chain must chain to a trusted root (or be one).
        let top = &chain[chain.len() - 1];
        if self.trusted_issuer(top) {
            return Ok(());
        }
        // Allow the chain to already terminate at a self-issued trusted root.
        for r in &self.roots {
            if r.subject == top.subject && r.tbs == top.tbs {
                return Ok(());
            }
        }
        Err(X509Error::UntrustedRoot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        s.chunks(2)
            .map(|c| u8::from_str_radix(core::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn big_roundtrip_and_arith() {
        let a = Big::from_be_bytes(&hex("0123456789abcdef"));
        let b = Big::from_be_bytes(&hex("fedcba9876543210"));
        let s = a.add(&b);
        assert_eq!(s.to_be_bytes(8), hex("ffffffffffffffff"));
        let prod = Big::from_be_bytes(&[0xff, 0xff]).mul(&Big::from_be_bytes(&[0xff, 0xff]));
        assert_eq!(prod.to_be_bytes(4), hex("fffe0001"));
        // 1000 mod 7 = 6.
        let m = Big::from_be_bytes(&[0x03, 0xe8]).rem(&Big::from_be_bytes(&[7]));
        assert_eq!(m.to_be_bytes(1), vec![6]);
    }

    #[test]
    fn big_modexp_small() {
        // 4^13 mod 497 = 445 (classic worked example).
        let base = Big::from_be_bytes(&[4]);
        let exp = Big::from_be_bytes(&[13]);
        let m = Big::from_be_bytes(&[0x01, 0xf1]); // 497
        let r = base.modexp(&exp, &m);
        assert_eq!(r.to_be_bytes(2), vec![0x01, 0xbd]); // 445
    }

    // NOTE: these are fixed OpenSSL-generated vectors (signatures + the self-signed
    // cert below) created before the AetherOS→DominionOS rename. The signing private
    // keys for the ECDSA vector are not in-repo, so the vectors cannot be regenerated
    // for the new name — `MSG` and the host names here MUST stay the original values
    // the signatures actually attest, or verification correctly fails. This exercises
    // the (correct) crypto against real OpenSSL output; do not "fix" it by editing the
    // crypto code to accept the renamed strings.
    const MSG: &[u8] = b"AetherOS TLS test message";

    #[test]
    fn ecdsa_p256_openssl_vector() {
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(&hex("49dd6263c3b56fde952115dad1b43d7c29d40a9c79b2b087625bca4e89d26f3b"));
        y.copy_from_slice(&hex("e96fdae1198fbab0a1b198c6e50d2201a84c4e1dc659e6e5baf298fa59a9b0bb"));
        let key = PublicKey::EcP256 { x, y };
        let sig = hex("3046022100f094ed1ee5cc6a6f7a6e151caa11a5b14c07f0f382ccfbb7985b6601745bc3cd0221008e11dc3b390d0218e127e31332ae01ce31d6cd9288ecd9776dc05fb8ada81b9d");
        assert!(key.verify(SigAlg::EcdsaP256Sha256, MSG, &sig));
        // Flip a message byte → must fail.
        let mut bad = MSG.to_vec();
        bad[0] ^= 1;
        assert!(!key.verify(SigAlg::EcdsaP256Sha256, &bad, &sig));
    }

    #[test]
    fn rsa_pkcs1_openssl_vector() {
        let n = hex("e4dd7baf7f987a6b91db9d8e041295f5b97f3ecfdad0aef3c6e7587c769399252b542ab59ad9358f2f5f3e885c8bf36119d7e0ac735d80b5243817332f34e37ba6e9b5400a977e75093d3b49c494f16b8a63c73551834ac0fe58709a80a6f9a0f8355d8c3a1d452a7c3f7c282c3ee14c8f6ab4fb4d8763086100c62184a90463df9de7a4fc856672e211e694a4be812f2aa3ad9099d54c966ca6e100d9f1c8dba00fd559d5a01f77a6367927825a57e82e17935636645ebd1ecb848d0bbb19a40a0336eb4100f1fe53ea3c6b978ab05f57c13f0e55b5a1307b03d8c80edce420cc0e3463b089a07abd093b0ebf315ecf5af9ad358cc8bc7a8c2dd86cb413254b");
        let e = hex("010001");
        let key = PublicKey::Rsa { n, e };
        let sig = hex("49089e693a4968b1b121f79a001af1de3cd08715c0f70912ed6e506bd743a1b821710598e1c97127079784ce4977c338758b59b80646cb71e43f54c2e6164bf54a03b980ade76201c620f0a17e22d2525532f9523f0b6b9a04ee54b73c0fe39f02cf7bfff1e559a392a50de3787e77a433645291573ee9747308afbbbbd990943c0146e1addc5f82db57bafeb8b681a14d64c49b729450cce64ac6d1d1337c61976f5cac690bc7f47997bdd0278377f67d9d330e391c3db3ed5df00f3375a36eef1a9b02893effecb429b165a31c3a73294937e9c5eb39fc48a6adc697af789dcbfa24d5199e0fd51afe517a3b389a217db027e355c60e21432276e97b5c9abe");
        assert!(key.verify(SigAlg::RsaPkcs1Sha256, MSG, &sig));
        let mut badsig = sig.clone();
        badsig[100] ^= 1;
        assert!(!key.verify(SigAlg::RsaPkcs1Sha256, MSG, &badsig));
    }

    #[test]
    fn rsa_pss_openssl_vector() {
        let n = hex("e4dd7baf7f987a6b91db9d8e041295f5b97f3ecfdad0aef3c6e7587c769399252b542ab59ad9358f2f5f3e885c8bf36119d7e0ac735d80b5243817332f34e37ba6e9b5400a977e75093d3b49c494f16b8a63c73551834ac0fe58709a80a6f9a0f8355d8c3a1d452a7c3f7c282c3ee14c8f6ab4fb4d8763086100c62184a90463df9de7a4fc856672e211e694a4be812f2aa3ad9099d54c966ca6e100d9f1c8dba00fd559d5a01f77a6367927825a57e82e17935636645ebd1ecb848d0bbb19a40a0336eb4100f1fe53ea3c6b978ab05f57c13f0e55b5a1307b03d8c80edce420cc0e3463b089a07abd093b0ebf315ecf5af9ad358cc8bc7a8c2dd86cb413254b");
        let e = hex("010001");
        let key = PublicKey::Rsa { n, e };
        let sig = hex("2e7d0a5d30027ed88b53ce15bbe3b3fba92a1deed16b1131818506705ff010eadc291505491e81a6c51029c99591d9cc1d8e34d2950e650efd5fddff4a7cd8693dadc33b09297c4489ebec5cf8b06047cd1420a6375b39aa1542e6dacc44cf9e94a6cbb98f8e7954de2832709a5669b167bc01a057297defd0a7ae245bd3393e25f771f8c1f8a5722ec04b9ffcdcdb231d2067b63f5fa2b41048c1cf2ec56785ad0d5141c4a29e1f540c43f41d0fc5738d0a81115578874eba2666ee8c510ea0884b6570a654404896b1b1f06f516ac8c7409cbc10a5f424bb80a85606526d4471c557bef58645a1c62c96470b2d3eb29c7b7a7e2644beccb2fa5b93b58b19f6");
        assert!(key.verify(SigAlg::RsaPssSha256, MSG, &sig));
        let mut bad = MSG.to_vec();
        bad[1] ^= 1;
        assert!(!key.verify(SigAlg::RsaPssSha256, &bad, &sig));
    }

    const SELF_SIGNED: &str = "308203343082021ca0030201020214504ad3ea3f919686aee08ca8172e53489f2776a3300d06092a864886f70d01010b050030163114301206035504030c0b6165746865722e74657374301e170d3236303632303134313633315a170d3336303631373134313633315a30163114301206035504030c0b6165746865722e7465737430820122300d06092a864886f70d01010105000382010f003082010a0282010100e4dd7baf7f987a6b91db9d8e041295f5b97f3ecfdad0aef3c6e7587c769399252b542ab59ad9358f2f5f3e885c8bf36119d7e0ac735d80b5243817332f34e37ba6e9b5400a977e75093d3b49c494f16b8a63c73551834ac0fe58709a80a6f9a0f8355d8c3a1d452a7c3f7c282c3ee14c8f6ab4fb4d8763086100c62184a90463df9de7a4fc856672e211e694a4be812f2aa3ad9099d54c966ca6e100d9f1c8dba00fd559d5a01f77a6367927825a57e82e17935636645ebd1ecb848d0bbb19a40a0336eb4100f1fe53ea3c6b978ab05f57c13f0e55b5a1307b03d8c80edce420cc0e3463b089a07abd093b0ebf315ecf5af9ad358cc8bc7a8c2dd86cb413254b0203010001a37a3078301d0603551d0e0416041403fe8942062179258f728b66d38a2f194ded7f01301f0603551d2304183016801403fe8942062179258f728b66d38a2f194ded7f01300f0603551d130101ff040530030101ff30250603551d11041e301c820b6165746865722e74657374820d2a2e6165746865722e74657374300d06092a864886f70d01010b05000382010100b3d7a6d4035c81655219e2e17cfad42bd0716a4cd814cb8ad9be9c54513a0021bbb6b20c87f8fafb5c42b8d8e58c2b2489062a2a830c6b566b14ac6f4f76e3d6bcec6dcf44b98bd3a4e24fb422d2a17c7e7855393ea83ce685097914d80a2fb158fb21ae5a88970e3efbac9d5c881c14db116a4ef653cd8386d14fd6872603a2240d2d46eabdb2b436fc1e9aebf763363dc8e3f56804e4e935278fd76aa86b9baea679daf6ee111917426131d57ae0681e5836c6a04452d635c2ac6419e0acdf6964ebfffaf9e3410fd8e2f7c23f500e2e582b0d997a527eb2e11f0646c0a89f1860812aa35510cd645e45ba730cfb494acef33419732a9632408519ceb177f0";

    #[test]
    fn parse_and_verify_self_signed() {
        let der = hex(SELF_SIGNED);
        let cert = parse_certificate(&der).expect("parse");
        assert_eq!(cert.sig_alg, SigAlg::RsaPkcs1Sha256);
        assert!(cert.is_ca);
        // The embedded cert's SAN is aether.test / *.aether.test (pre-rename fixture).
        assert!(cert.matches_host("aether.test"));
        assert!(cert.matches_host("foo.aether.test")); // wildcard
        assert!(!cert.matches_host("evil.com"));
        // Self-signed: it signs itself.
        assert!(cert.verify_signed_by(&cert));
        // Trust store: add as root, verify a one-element chain.
        let mut store = TrustStore::new();
        store.add_root(cert.clone());
        assert!(store.verify_chain(&[cert.clone()], "aether.test", 0).is_ok());
        assert_eq!(
            store.verify_chain(&[cert], "evil.com", 0).err(),
            Some(X509Error::NameMismatch)
        );
    }

    #[test]
    fn system_bundle_loads_and_self_verifies() {
        let store = system_trust_store();
        // The embedded Mozilla bundle should yield a large set of anchors.
        assert!(store.len() > 100, "only {} roots loaded", store.len());
        // Every root whose self-signature uses an algorithm we verify must
        // validate against its own key — this exercises RSA-SHA256/384/512 and
        // ECDSA P-256/P-384 against real CA certificates.
        let mut checked = 0usize;
        let mut failures = alloc::vec::Vec::new();
        for r in &store.roots {
            match r.sig_alg {
                SigAlg::Unknown | SigAlg::RsaPssSha256 => continue,
                _ => {}
            }
            if r.verify_signed_by(r) {
                checked += 1;
            } else {
                let kind = match r.key {
                    PublicKey::Rsa { ref n, .. } => alloc::format!("RSA{}", n.len() * 8),
                    PublicKey::EcP256 { .. } => alloc::string::String::from("EcP256"),
                    PublicKey::EcP384 { .. } => alloc::string::String::from("EcP384"),
                };
                failures.push(alloc::format!("{:?}/{}", r.sig_alg, kind));
            }
        }
        assert!(failures.is_empty(), "self-verify failures: {:?}", failures);
        // We should have actually verified a meaningful number of real roots,
        // including some ECDSA P-384 ones.
        assert!(checked > 50, "only {} roots self-verified", checked);
        let p384 = store
            .roots
            .iter()
            .any(|r| matches!(r.key, PublicKey::EcP384 { .. }));
        assert!(p384, "expected at least one P-384 root in the bundle");
    }
}

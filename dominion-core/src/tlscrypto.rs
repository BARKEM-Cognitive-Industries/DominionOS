//! The cryptographic primitives TLS 1.3 needs, in pure, safe `no_std` Rust, built on
//! the crate's NIST-tested [`sha256`](crate::hash::sha256):
//!
//! * **HMAC-SHA256** and **HKDF** (RFC 5869) + the TLS 1.3 `HKDF-Expand-Label` /
//!   `Derive-Secret` key-schedule helpers (RFC 8446 §7.1).
//! * **ChaCha20-Poly1305** AEAD (RFC 8439) and **AES-128-GCM** (RFC 5116/NIST) — the
//!   two TLS 1.3 cipher suites we offer.
//! * **X25519** ECDH (RFC 7748), ported from the public-domain TweetNaCl ladder.
//!
//! Every primitive is checked against its RFC/NIST test vectors below. These are the
//! building blocks; the handshake state machine lives in [`crate::tls`].

use crate::hash::sha256;
use alloc::vec::Vec;

// ════════════════════════════ HMAC-SHA256 / HKDF ════════════════════════════

const SHA256_BLOCK: usize = 64;
pub const SHA256_LEN: usize = 32;

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; SHA256_BLOCK];
    if key.len() > SHA256_BLOCK {
        let h = sha256(key);
        k[..32].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; SHA256_BLOCK];
    let mut opad = [0x5cu8; SHA256_BLOCK];
    for i in 0..SHA256_BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Vec::with_capacity(SHA256_BLOCK + msg.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(msg);
    let inner_hash = sha256(&inner);
    let mut outer = Vec::with_capacity(SHA256_BLOCK + 32);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

/// HKDF-Extract (RFC 5869).
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    hmac_sha256(salt, ikm)
}

/// HKDF-Expand (RFC 5869).
pub fn hkdf_expand(prk: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    // RFC 5869 requires L <= 255*HashLen. Beyond that the single-byte T(n)
    // counter below wraps 255 -> 0 and repeats, producing non-conformant,
    // internally-inconsistent key material. Guard the invariant.
    debug_assert!(len <= 255 * SHA256_LEN, "HKDF-Expand: len must be <= 255*HashLen");
    let mut out = Vec::with_capacity(len);
    let mut t: Vec<u8> = Vec::new();
    let mut counter: u8 = 1;
    while out.len() < len {
        let mut data = Vec::with_capacity(t.len() + info.len() + 1);
        data.extend_from_slice(&t);
        data.extend_from_slice(info);
        data.push(counter);
        let block = hmac_sha256(prk, &data);
        t = block.to_vec();
        out.extend_from_slice(&block);
        counter = counter.wrapping_add(1);
    }
    out.truncate(len);
    out
}

/// TLS 1.3 `HKDF-Expand-Label` (RFC 8446 §7.1).
pub fn hkdf_expand_label(secret: &[u8], label: &str, context: &[u8], len: usize) -> Vec<u8> {
    let mut full_label = Vec::from(&b"tls13 "[..]);
    full_label.extend_from_slice(label.as_bytes());
    let mut info = Vec::new();
    info.extend_from_slice(&(len as u16).to_be_bytes());
    info.push(full_label.len() as u8);
    info.extend_from_slice(&full_label);
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    hkdf_expand(secret, &info, len)
}

/// TLS 1.3 `Derive-Secret(secret, label, transcript)` (RFC 8446 §7.1).
pub fn derive_secret(secret: &[u8], label: &str, transcript_hash: &[u8]) -> [u8; 32] {
    let v = hkdf_expand_label(secret, label, transcript_hash, 32);
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    out
}

// ════════════════════════════ ChaCha20-Poly1305 ════════════════════════════

fn rotl32(x: u32, n: u32) -> u32 {
    x.rotate_left(n)
}

fn chacha_quarter(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = rotl32(s[d] ^ s[a], 16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = rotl32(s[b] ^ s[c], 12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = rotl32(s[d] ^ s[a], 8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = rotl32(s[b] ^ s[c], 7);
}

fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut state = [0u32; 16];
    state[0] = 0x6170_7865;
    state[1] = 0x3320_646e;
    state[2] = 0x7962_2d32;
    state[3] = 0x6b20_6574;
    for i in 0..8 {
        state[4 + i] = u32::from_le_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }
    state[12] = counter;
    state[13] = u32::from_le_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]);
    state[14] = u32::from_le_bytes([nonce[4], nonce[5], nonce[6], nonce[7]]);
    state[15] = u32::from_le_bytes([nonce[8], nonce[9], nonce[10], nonce[11]]);

    let mut working = state;
    for _ in 0..10 {
        chacha_quarter(&mut working, 0, 4, 8, 12);
        chacha_quarter(&mut working, 1, 5, 9, 13);
        chacha_quarter(&mut working, 2, 6, 10, 14);
        chacha_quarter(&mut working, 3, 7, 11, 15);
        chacha_quarter(&mut working, 0, 5, 10, 15);
        chacha_quarter(&mut working, 1, 6, 11, 12);
        chacha_quarter(&mut working, 2, 7, 8, 13);
        chacha_quarter(&mut working, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        let v = working[i].wrapping_add(state[i]);
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// ChaCha20 stream cipher (RFC 8439 §2.4), starting at block `counter`.
pub fn chacha20_xor(key: &[u8; 32], counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let mut block_counter = counter;
    let mut offset = 0;
    while offset < data.len() {
        let ks = chacha20_block(key, block_counter, nonce);
        let n = core::cmp::min(64, data.len() - offset);
        for i in 0..n {
            data[offset + i] ^= ks[i];
        }
        offset += 64;
        block_counter = block_counter.wrapping_add(1);
    }
}

/// Poly1305 one-time MAC (RFC 8439 §2.5), 130-bit arithmetic over 26-bit limbs.
fn poly1305(key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    let r0 = u32::from_le_bytes([key[0], key[1], key[2], key[3]]) & 0x3ff_ffff;
    let r1 = (u32::from_le_bytes([key[3], key[4], key[5], key[6]]) >> 2) & 0x3ff_ff03;
    let r2 = (u32::from_le_bytes([key[6], key[7], key[8], key[9]]) >> 4) & 0x3ff_c0ff;
    let r3 = (u32::from_le_bytes([key[9], key[10], key[11], key[12]]) >> 6) & 0x3f0_3fff;
    let r4 = (u32::from_le_bytes([key[12], key[13], key[14], key[15]]) >> 8) & 0x00f_ffff;

    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

    let (mut h0, mut h1, mut h2, mut h3, mut h4) = (0u32, 0u32, 0u32, 0u32, 0u32);

    for chunk in msg.chunks(16) {
        let mut block = [0u8; 17];
        block[..chunk.len()].copy_from_slice(chunk);
        block[chunk.len()] = 1; // the high bit (padding)

        h0 += u32::from_le_bytes([block[0], block[1], block[2], block[3]]) & 0x3ff_ffff;
        h1 += (u32::from_le_bytes([block[3], block[4], block[5], block[6]]) >> 2) & 0x3ff_ffff;
        h2 += (u32::from_le_bytes([block[6], block[7], block[8], block[9]]) >> 4) & 0x3ff_ffff;
        h3 += (u32::from_le_bytes([block[9], block[10], block[11], block[12]]) >> 6) & 0x3ff_ffff;
        h4 += (u32::from_le_bytes([block[12], block[13], block[14], block[15]]) >> 8) | ((block[16] as u32) << 24);

        let d0 = h0 as u64 * r0 as u64 + h1 as u64 * s4 as u64 + h2 as u64 * s3 as u64 + h3 as u64 * s2 as u64 + h4 as u64 * s1 as u64;
        let d1 = h0 as u64 * r1 as u64 + h1 as u64 * r0 as u64 + h2 as u64 * s4 as u64 + h3 as u64 * s3 as u64 + h4 as u64 * s2 as u64;
        let d2 = h0 as u64 * r2 as u64 + h1 as u64 * r1 as u64 + h2 as u64 * r0 as u64 + h3 as u64 * s4 as u64 + h4 as u64 * s3 as u64;
        let d3 = h0 as u64 * r3 as u64 + h1 as u64 * r2 as u64 + h2 as u64 * r1 as u64 + h3 as u64 * r0 as u64 + h4 as u64 * s4 as u64;
        let d4 = h0 as u64 * r4 as u64 + h1 as u64 * r3 as u64 + h2 as u64 * r2 as u64 + h3 as u64 * r1 as u64 + h4 as u64 * r0 as u64;

        let mut c;
        h0 = (d0 & 0x3ff_ffff) as u32;
        c = d0 >> 26;
        let d1 = d1 + c;
        h1 = (d1 & 0x3ff_ffff) as u32;
        c = d1 >> 26;
        let d2 = d2 + c;
        h2 = (d2 & 0x3ff_ffff) as u32;
        c = d2 >> 26;
        let d3 = d3 + c;
        h3 = (d3 & 0x3ff_ffff) as u32;
        c = d3 >> 26;
        let d4 = d4 + c;
        h4 = (d4 & 0x3ff_ffff) as u32;
        c = d4 >> 26;
        h0 += (c as u32) * 5;
        h1 += h0 >> 26;
        h0 &= 0x3ff_ffff;
    }

    // Final reduction.
    let mut c = h1 >> 26;
    h1 &= 0x3ff_ffff;
    h2 += c;
    c = h2 >> 26;
    h2 &= 0x3ff_ffff;
    h3 += c;
    c = h3 >> 26;
    h3 &= 0x3ff_ffff;
    h4 += c;
    c = h4 >> 26;
    h4 &= 0x3ff_ffff;
    h0 += c * 5;
    c = h0 >> 26;
    h0 &= 0x3ff_ffff;
    h1 += c;

    // Compute h + -p (i.e. h - p) and select.
    let mut g0 = h0.wrapping_add(5);
    c = g0 >> 26;
    g0 &= 0x3ff_ffff;
    let mut g1 = h1.wrapping_add(c);
    c = g1 >> 26;
    g1 &= 0x3ff_ffff;
    let mut g2 = h2.wrapping_add(c);
    c = g2 >> 26;
    g2 &= 0x3ff_ffff;
    let mut g3 = h3.wrapping_add(c);
    c = g3 >> 26;
    g3 &= 0x3ff_ffff;
    let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    let mask = (g4 >> 31).wrapping_sub(1); // 0xffffffff if g4 >= 0 (no borrow)
    let nmask = !mask;
    h0 = (h0 & nmask) | (g0 & mask);
    h1 = (h1 & nmask) | (g1 & mask);
    h2 = (h2 & nmask) | (g2 & mask);
    h3 = (h3 & nmask) | (g3 & mask);
    h4 = (h4 & nmask) | (g4 & mask);

    // Serialize h to 128-bit little-endian, then add s = key[16..32].
    let f0 = (h0 | (h1 << 26)) as u64;
    let f1 = ((h1 >> 6) | (h2 << 20)) as u64;
    let f2 = ((h2 >> 12) | (h3 << 14)) as u64;
    let f3 = ((h3 >> 18) | (h4 << 8)) as u64;

    let s0 = u32::from_le_bytes([key[16], key[17], key[18], key[19]]) as u64;
    let s1v = u32::from_le_bytes([key[20], key[21], key[22], key[23]]) as u64;
    let s2v = u32::from_le_bytes([key[24], key[25], key[26], key[27]]) as u64;
    let s3v = u32::from_le_bytes([key[28], key[29], key[30], key[31]]) as u64;

    let mut acc = f0 + s0;
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&((acc & 0xffff_ffff) as u32).to_le_bytes());
    acc = f1 + s1v + (acc >> 32);
    out[4..8].copy_from_slice(&((acc & 0xffff_ffff) as u32).to_le_bytes());
    acc = f2 + s2v + (acc >> 32);
    out[8..12].copy_from_slice(&((acc & 0xffff_ffff) as u32).to_le_bytes());
    acc = f3 + s3v + (acc >> 32);
    out[12..16].copy_from_slice(&((acc & 0xffff_ffff) as u32).to_le_bytes());
    out
}

fn poly1305_key_gen(key: &[u8; 32], nonce: &[u8; 12]) -> [u8; 32] {
    let block = chacha20_block(key, 0, nonce);
    let mut out = [0u8; 32];
    out.copy_from_slice(&block[..32]);
    out
}

fn pad16(v: &mut Vec<u8>, len: usize) {
    let rem = len % 16;
    if rem != 0 {
        for _ in 0..(16 - rem) {
            v.push(0);
        }
    }
}

/// AEAD encrypt (RFC 8439 §2.8): returns ciphertext || 16-byte tag.
pub fn chacha20poly1305_seal(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
    let otk = poly1305_key_gen(key, nonce);
    let mut ct = plaintext.to_vec();
    chacha20_xor(key, 1, nonce, &mut ct);

    let mut mac_data = Vec::new();
    mac_data.extend_from_slice(aad);
    pad16(&mut mac_data, aad.len());
    mac_data.extend_from_slice(&ct);
    pad16(&mut mac_data, ct.len());
    mac_data.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_data.extend_from_slice(&(ct.len() as u64).to_le_bytes());
    let tag = poly1305(&otk, &mac_data);

    ct.extend_from_slice(&tag);
    ct
}

/// AEAD decrypt: verifies the tag (constant-time) and returns the plaintext, or
/// `None` on authentication failure.
pub fn chacha20poly1305_open(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], ciphertext: &[u8]) -> Option<Vec<u8>> {
    if ciphertext.len() < 16 {
        return None;
    }
    let (ct, tag) = ciphertext.split_at(ciphertext.len() - 16);
    let otk = poly1305_key_gen(key, nonce);

    let mut mac_data = Vec::new();
    mac_data.extend_from_slice(aad);
    pad16(&mut mac_data, aad.len());
    mac_data.extend_from_slice(ct);
    pad16(&mut mac_data, ct.len());
    mac_data.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_data.extend_from_slice(&(ct.len() as u64).to_le_bytes());
    let expected = poly1305(&otk, &mac_data);

    if !ct_eq(&expected, tag) {
        return None;
    }
    let mut pt = ct.to_vec();
    chacha20_xor(key, 1, nonce, &mut pt);
    Some(pt)
}

/// Constant-time byte-slice equality.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ════════════════════════════ X25519 (TweetNaCl ladder) ════════════════════════════

type Gf = [i64; 16];

const _121665: Gf = [0xDB41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

fn gf0() -> Gf {
    [0; 16]
}

fn car25519(o: &mut Gf) {
    for i in 0..16 {
        o[i] += 1 << 16;
        let c = o[i] >> 16;
        o[(i + 1) * ((i < 15) as usize)] += c - 1 + 37 * (c - 1) * ((i == 15) as i64);
        o[i] -= c << 16;
    }
}

fn sel25519(p: &mut Gf, q: &mut Gf, b: i64) {
    let c = !(b - 1);
    for i in 0..16 {
        let t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

fn pack25519(o: &mut [u8; 32], n: &Gf) {
    let mut m = gf0();
    let mut t = *n;
    car25519(&mut t);
    car25519(&mut t);
    car25519(&mut t);
    for _ in 0..2 {
        m[0] = t[0] - 0xffed;
        for i in 1..15 {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        let b = (m[15] >> 16) & 1;
        m[14] &= 0xffff;
        sel25519(&mut t, &mut m, 1 - b);
    }
    for i in 0..16 {
        o[2 * i] = (t[i] & 0xff) as u8;
        o[2 * i + 1] = (t[i] >> 8) as u8;
    }
}

fn unpack25519(o: &mut Gf, n: &[u8; 32]) {
    for i in 0..16 {
        o[i] = n[2 * i] as i64 + ((n[2 * i + 1] as i64) << 8);
    }
    o[15] &= 0x7fff;
}

fn add(o: &mut Gf, a: &Gf, b: &Gf) {
    for i in 0..16 {
        o[i] = a[i] + b[i];
    }
}
fn sub(o: &mut Gf, a: &Gf, b: &Gf) {
    for i in 0..16 {
        o[i] = a[i] - b[i];
    }
}
fn mul(o: &mut Gf, a: &Gf, b: &Gf) {
    let mut t = [0i64; 31];
    for i in 0..16 {
        for j in 0..16 {
            t[i + j] += a[i] * b[j];
        }
    }
    for i in 0..15 {
        t[i] += 38 * t[i + 16];
    }
    let mut r = [0i64; 16];
    r.copy_from_slice(&t[..16]);
    car25519(&mut r);
    car25519(&mut r);
    *o = r;
}
fn sqr(o: &mut Gf, a: &Gf) {
    let a2 = *a;
    mul(o, a, &a2);
}

fn inv25519(o: &mut Gf, i: &Gf) {
    let mut c = *i;
    for a in (0..=253).rev() {
        let c2 = c;
        sqr(&mut c, &c2);
        if a != 2 && a != 4 {
            let c3 = c;
            mul(&mut c, &c3, i);
        }
    }
    *o = c;
}

/// X25519 scalar multiplication (RFC 7748): `scalar · point`.
pub fn x25519(scalar: &[u8; 32], point: &[u8; 32]) -> [u8; 32] {
    let mut z = *scalar;
    z[0] &= 248;
    z[31] &= 127;
    z[31] |= 64;

    let mut x = gf0();
    unpack25519(&mut x, point);
    let mut a = gf0();
    let mut b = x;
    let mut c = gf0();
    let mut d = gf0();
    let mut e = gf0();
    let mut f = gf0();
    a[0] = 1;
    d[0] = 1;

    for i in (0..=254).rev() {
        let bit = ((z[i >> 3] >> (i & 7)) & 1) as i64;
        sel25519(&mut a, &mut b, bit);
        sel25519(&mut c, &mut d, bit);
        add(&mut e, &a, &c);
        let a2 = a;
        sub(&mut a, &a2, &c);
        add(&mut c, &b, &d);
        let b2 = b;
        sub(&mut b, &b2, &d);
        sqr(&mut d, &e);
        sqr(&mut f, &a);
        let a3 = a;
        mul(&mut a, &c, &a3);
        mul(&mut c, &b, &e);
        add(&mut e, &a, &c);
        let a4 = a;
        sub(&mut a, &a4, &c);
        sqr(&mut b, &a);
        sub(&mut c, &d, &f);
        mul(&mut a, &c, &_121665);
        let a5 = a;
        add(&mut a, &a5, &d);
        let c2 = c;
        mul(&mut c, &c2, &a);
        mul(&mut a, &d, &f);
        let d2 = d;
        mul(&mut d, &b, &x);
        sqr(&mut b, &e);
        sel25519(&mut a, &mut b, bit);
        sel25519(&mut c, &mut d, bit);
        let _ = (d2, b2, a3);
    }

    let x16 = [gf0(); 1];
    let _ = x16;
    // out = a * c^-1
    let mut cc = gf0();
    inv25519(&mut cc, &c);
    let mut out_fe = gf0();
    mul(&mut out_fe, &a, &cc);
    let mut out = [0u8; 32];
    pack25519(&mut out, &out_fe);
    out
}

/// The X25519 base point (u = 9).
pub const X25519_BASE: [u8; 32] = [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Derive the public key for a private scalar (`scalar · basepoint`).
pub fn x25519_base(scalar: &[u8; 32]) -> [u8; 32] {
    x25519(scalar, &X25519_BASE)
}

/// X25519 with the RFC 7748 §6.1 low-order-point guard.
///
/// Computes `scalar · point` and returns `None` if the resulting shared secret
/// is all zeros. A peer supplying a low-order public key (the canonical case is
/// the all-zero point) can otherwise collapse the shared secret to a known
/// constant, so callers performing key agreement MUST abort in that case rather
/// than proceeding with an attacker-controlled secret. The zero check is
/// constant-time (it never branches on individual secret bytes).
pub fn x25519_checked(scalar: &[u8; 32], point: &[u8; 32]) -> Option<[u8; 32]> {
    let out = x25519(scalar, point);
    if ct_eq(&out, &[0u8; 32]) {
        None
    } else {
        Some(out)
    }
}

// ============================================================================
// AES-128 + GCM — the mandatory TLS 1.3 cipher (TLS_AES_128_GCM_SHA256).
// ============================================================================

/// The AES S-box (Rijndael substitution table).
///
/// # Timing side-channel note
/// This software AES implementation uses a 256-byte lookup table (`AES_SBOX`).
/// On platforms with data caches, the index into this table is derived from key
/// material, so cache-hit vs cache-miss timing can leak key bytes (a classic
/// cache-timing attack, cf. Bernstein 2005, Bonneau & Mironov 2006).
/// A full mitigation requires either bit-sliced AES (complex in no_std Rust) or
/// hardware AES instructions (AES-NI on x86, ARMv8 crypto extensions).
/// **The AES-128-GCM path in this crate is only safe when the caller can guarantee
/// that hardware AES acceleration is available and is actually used by the
/// compiler/CPU.** For all other deployments, prefer ChaCha20-Poly1305 (the
/// default cipher suite), whose software implementation is inherently
/// constant-time.
static AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

const AES_RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// Expanded AES-128 key schedule: 11 round keys × 16 bytes = 176 bytes.
pub fn aes128_key_schedule(key: &[u8; 16]) -> [u8; 176] {
    let mut w = [0u8; 176];
    w[..16].copy_from_slice(key);
    let mut i = 16;
    let mut rcon_idx = 0;
    while i < 176 {
        let mut temp = [w[i - 4], w[i - 3], w[i - 2], w[i - 1]];
        if i % 16 == 0 {
            // RotWord + SubWord + Rcon
            let t0 = temp[0];
            temp[0] = AES_SBOX[temp[1] as usize] ^ AES_RCON[rcon_idx];
            temp[1] = AES_SBOX[temp[2] as usize];
            temp[2] = AES_SBOX[temp[3] as usize];
            temp[3] = AES_SBOX[t0 as usize];
            rcon_idx += 1;
        }
        for j in 0..4 {
            w[i + j] = w[i - 16 + j] ^ temp[j];
        }
        i += 4;
    }
    w
}

#[inline]
fn xtime(x: u8) -> u8 {
    // Timing side-channel fix: the original branch `if hi != 0 { r ^= 0x1b }`
    // is data-dependent on the high bit of an AES state byte (which is derived
    // from key material), leaking timing information via branch prediction.
    // Replace with a branchless mask: the mask is all-ones when the high bit is
    // set and all-zeros otherwise, so the XOR is applied unconditionally but
    // only takes effect when the high bit was set.
    let mask = ((x as i8) >> 7) as u8; // 0xFF if high bit set, 0x00 otherwise
    (x << 1) ^ (0x1b & mask)
}

/// Encrypt a single 16-byte block in place with an expanded AES-128 schedule.
pub fn aes128_encrypt_block(rk: &[u8; 176], block: &mut [u8; 16]) {
    // AddRoundKey (round 0)
    for i in 0..16 {
        block[i] ^= rk[i];
    }
    for round in 1..10 {
        // SubBytes
        for i in 0..16 {
            block[i] = AES_SBOX[block[i] as usize];
        }
        // ShiftRows (column-major state: state[r + 4c])
        shift_rows(block);
        // MixColumns
        mix_columns(block);
        // AddRoundKey
        for i in 0..16 {
            block[i] ^= rk[round * 16 + i];
        }
    }
    // Final round (no MixColumns)
    for i in 0..16 {
        block[i] = AES_SBOX[block[i] as usize];
    }
    shift_rows(block);
    for i in 0..16 {
        block[i] ^= rk[160 + i];
    }
}

fn shift_rows(s: &mut [u8; 16]) {
    // State bytes are laid out column-major: s[r + 4*c], r=row, c=col.
    let t = *s;
    // row 1: shift left by 1
    s[1] = t[5];
    s[5] = t[9];
    s[9] = t[13];
    s[13] = t[1];
    // row 2: shift left by 2
    s[2] = t[10];
    s[6] = t[14];
    s[10] = t[2];
    s[14] = t[6];
    // row 3: shift left by 3
    s[3] = t[15];
    s[7] = t[3];
    s[11] = t[7];
    s[15] = t[11];
}

fn mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let i = c * 4;
        let a0 = s[i];
        let a1 = s[i + 1];
        let a2 = s[i + 2];
        let a3 = s[i + 3];
        s[i] = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
        s[i + 1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
        s[i + 2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
        s[i + 3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
    }
}

/// GHASH multiplication in GF(2^128) (per SP 800-38D, bit-reflected).
///
/// # Timing side-channel fix
/// The original implementation branched on individual bits of `x` and on the
/// LSB of `v` (both derived from secret key material — `x` is the GHASH
/// accumulator and `v` is the hash subkey H encrypted under the AES key).
/// Branches on secret bits create timing side channels exploitable by
/// local attackers or timing oracles.
///
/// This version uses branchless mask arithmetic throughout:
/// - `bit_mask` is 0xFF when the current bit of `xi` is 1, else 0x00.
/// - `lsb_mask` is 0xFF when the LSB of `v` is 1, else 0x00.
/// Both XOR operations are applied unconditionally, with the mask selecting
/// whether the effect is zero or nonzero, eliminating all data-dependent branches.
fn ghash_mul(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    let mut z = [0u8; 16];
    let mut v = *y;
    for &xi in x.iter() {
        for bit in 0..8 {
            // Timing side-channel fix: replace `if xi & mask != 0` branch with
            // a branchless mask. bit_mask is 0xFF when the bit is set, 0x00 otherwise.
            let bit_mask = (((xi >> (7 - bit)) & 1) as u8).wrapping_neg();
            for k in 0..16 {
                z[k] ^= v[k] & bit_mask;
            }
            // v = v >> 1 (big-endian shift), then conditionally reduce.
            // Timing side-channel fix: replace `if lsb != 0` branch with
            // a branchless mask derived from the LSB.
            let lsb_mask = (v[15] & 1).wrapping_neg(); // 0xFF if LSB set, 0x00 otherwise
            let mut carry = 0u8;
            for vk in v.iter_mut() {
                let new_carry = *vk & 1;
                *vk = (*vk >> 1) | (carry << 7);
                carry = new_carry;
            }
            v[0] ^= 0xe1 & lsb_mask;
        }
    }
    z
}

/// GHASH over AAD || ciphertext with hash subkey H.
fn ghash(h: &[u8; 16], aad: &[u8], ct: &[u8]) -> [u8; 16] {
    let mut y = [0u8; 16];
    fn feed(data: &[u8], y: &mut [u8; 16], h: &[u8; 16]) {
        let mut off = 0;
        while off < data.len() {
            let mut block = [0u8; 16];
            let n = core::cmp::min(16, data.len() - off);
            block[..n].copy_from_slice(&data[off..off + n]);
            for k in 0..16 {
                y[k] ^= block[k];
            }
            *y = ghash_mul(y, h);
            off += 16;
        }
    }
    feed(aad, &mut y, h);
    feed(ct, &mut y, h);
    // length block: [aad_bits (64) || ct_bits (64)] big-endian
    let mut lenblk = [0u8; 16];
    let aad_bits = (aad.len() as u64) * 8;
    let ct_bits = (ct.len() as u64) * 8;
    lenblk[..8].copy_from_slice(&aad_bits.to_be_bytes());
    lenblk[8..].copy_from_slice(&ct_bits.to_be_bytes());
    for k in 0..16 {
        y[k] ^= lenblk[k];
    }
    ghash_mul(&y, h)
}

#[inline]
fn gctr_block(rk: &[u8; 176], counter: &[u8; 16]) -> [u8; 16] {
    let mut b = *counter;
    aes128_encrypt_block(rk, &mut b);
    b
}

#[inline]
fn inc32(ctr: &mut [u8; 16]) {
    let mut c = u32::from_be_bytes([ctr[12], ctr[13], ctr[14], ctr[15]]);
    c = c.wrapping_add(1);
    ctr[12..].copy_from_slice(&c.to_be_bytes());
}

/// AES-128-GCM seal. `nonce` is 12 bytes (TLS uses a 96-bit IV). Returns
/// `ciphertext || 16-byte tag`.
pub fn aes128_gcm_seal(key: &[u8; 16], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
    let rk = aes128_key_schedule(key);
    let mut h = [0u8; 16];
    aes128_encrypt_block(&rk, &mut h);

    // J0 = nonce || 0x00000001 (for 96-bit IV)
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    // Encrypt plaintext with CTR starting at inc32(J0)
    let mut ct = Vec::with_capacity(plaintext.len());
    let mut ctr = j0;
    inc32(&mut ctr);
    let mut off = 0;
    while off < plaintext.len() {
        let ks = gctr_block(&rk, &ctr);
        let n = core::cmp::min(16, plaintext.len() - off);
        for i in 0..n {
            ct.push(plaintext[off + i] ^ ks[i]);
        }
        inc32(&mut ctr);
        off += 16;
    }

    // Tag = GHASH(H, aad, ct) XOR E(K, J0)
    let s = ghash(&h, aad, &ct);
    let ej0 = gctr_block(&rk, &j0);
    let mut out = ct;
    for i in 0..16 {
        out.push(s[i] ^ ej0[i]);
    }
    out
}

/// AES-128-GCM open. Input is `ciphertext || 16-byte tag`. Returns the
/// plaintext if the tag verifies.
pub fn aes128_gcm_open(key: &[u8; 16], nonce: &[u8; 12], aad: &[u8], input: &[u8]) -> Option<Vec<u8>> {
    if input.len() < 16 {
        return None;
    }
    let (ct, tag) = input.split_at(input.len() - 16);
    let rk = aes128_key_schedule(key);
    let mut h = [0u8; 16];
    aes128_encrypt_block(&rk, &mut h);

    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let s = ghash(&h, aad, ct);
    let ej0 = gctr_block(&rk, &j0);
    let mut expected = [0u8; 16];
    for i in 0..16 {
        expected[i] = s[i] ^ ej0[i];
    }
    if !ct_eq(&expected, tag) {
        return None;
    }

    // Decrypt
    let mut pt = Vec::with_capacity(ct.len());
    let mut ctr = j0;
    inc32(&mut ctr);
    let mut off = 0;
    while off < ct.len() {
        let ks = gctr_block(&rk, &ctr);
        let n = core::cmp::min(16, ct.len() - off);
        for i in 0..n {
            pt.push(ct[off + i] ^ ks[i]);
        }
        inc32(&mut ctr);
        off += 16;
    }
    Some(pt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        s.chunks(2).map(|c| u8::from_str_radix(core::str::from_utf8(c).unwrap(), 16).unwrap()).collect()
    }

    #[test]
    fn hmac_sha256_rfc4231_case2() {
        // RFC 4231 test case 2: key "Jefe", data "what do ya want for nothing?"
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            mac.to_vec(),
            hex("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
        );
    }

    #[test]
    fn hkdf_rfc5869_case1() {
        let ikm = hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let salt = hex("000102030405060708090a0b0c");
        let info = hex("f0f1f2f3f4f5f6f7f8f9");
        let prk = hkdf_extract(&salt, &ikm);
        assert_eq!(prk.to_vec(), hex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"));
        let okm = hkdf_expand(&prk, &info, 42);
        assert_eq!(
            okm,
            hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")
        );
    }

    #[test]
    fn chacha20_block_rfc8439() {
        // RFC 8439 §2.4.2 keystream for the worked example (counter = 1).
        let key: [u8; 32] = {
            let mut k = [0u8; 32];
            for (i, slot) in k.iter_mut().enumerate() {
                *slot = i as u8;
            }
            k
        };
        let nonce: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0x4a, 0, 0, 0, 0];
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let mut data = plaintext.to_vec();
        chacha20_xor(&key, 1, &nonce, &mut data);
        // First 16 bytes of the RFC ciphertext.
        assert_eq!(&data[..16], &hex("6e2e359a2568f98041ba0728dd0d6981")[..]);
    }

    #[test]
    fn chacha20poly1305_rfc8439_aead() {
        let key: [u8; 32] = {
            let mut k = [0u8; 32];
            k.copy_from_slice(&hex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"));
            k
        };
        let nonce: [u8; 12] = {
            let mut n = [0u8; 12];
            n.copy_from_slice(&hex("070000004041424344454647"));
            n
        };
        let aad = hex("50515253c0c1c2c3c4c5c6c7");
        let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let sealed = chacha20poly1305_seal(&key, &nonce, &aad, pt);
        // RFC tag.
        assert_eq!(&sealed[sealed.len() - 16..], &hex("1ae10b594f09e26a7e902ecbd0600691")[..]);
        // Round-trip.
        let opened = chacha20poly1305_open(&key, &nonce, &aad, &sealed).unwrap();
        assert_eq!(opened, pt.to_vec());
        // Tamper detection.
        let mut bad = sealed.clone();
        bad[0] ^= 1;
        assert!(chacha20poly1305_open(&key, &nonce, &aad, &bad).is_none());
    }

    #[test]
    fn x25519_rfc7748_vector() {
        let scalar: [u8; 32] = {
            let mut s = [0u8; 32];
            s.copy_from_slice(&hex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"));
            s
        };
        let point: [u8; 32] = {
            let mut p = [0u8; 32];
            p.copy_from_slice(&hex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"));
            p
        };
        let out = x25519(&scalar, &point);
        assert_eq!(out.to_vec(), hex("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"));
    }

    #[test]
    fn x25519_diffie_hellman_agrees() {
        let a_priv = [1u8; 32];
        let b_priv = [2u8; 32];
        let a_pub = x25519_base(&a_priv);
        let b_pub = x25519_base(&b_priv);
        let ab = x25519(&a_priv, &b_pub);
        let ba = x25519(&b_priv, &a_pub);
        assert_eq!(ab, ba);
    }

    #[test]
    fn hkdf_expand_label_shape() {
        // Smoke test: derive a 32-byte secret deterministically.
        let secret = [0x11u8; 32];
        let a = hkdf_expand_label(&secret, "derived", &[], 32);
        let b = hkdf_expand_label(&secret, "derived", &[], 32);
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn aes128_block_fips197() {
        // FIPS-197 Appendix B worked example.
        let key: [u8; 16] = {
            let mut k = [0u8; 16];
            k.copy_from_slice(&hex("2b7e151628aed2a6abf7158809cf4f3c"));
            k
        };
        let mut block: [u8; 16] = {
            let mut b = [0u8; 16];
            b.copy_from_slice(&hex("3243f6a8885a308d313198a2e0370734"));
            b
        };
        let rk = aes128_key_schedule(&key);
        aes128_encrypt_block(&rk, &mut block);
        assert_eq!(block.to_vec(), hex("3925841d02dc09fbdc118597196a0b32"));
    }

    #[test]
    fn aes128_gcm_mcgrew_case3() {
        // McGrew-Viega / NIST GCM Test Case 3 (64-byte plaintext, no AAD).
        let key: [u8; 16] = {
            let mut k = [0u8; 16];
            k.copy_from_slice(&hex("feffe9928665731c6d6a8f9467308308"));
            k
        };
        let nonce: [u8; 12] = {
            let mut n = [0u8; 12];
            n.copy_from_slice(&hex("cafebabefacedbaddecaf888"));
            n
        };
        let pt = hex(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
        );
        let sealed = aes128_gcm_seal(&key, &nonce, &[], &pt);
        let ct_len = sealed.len() - 16;
        assert_eq!(
            sealed[..ct_len].to_vec(),
            hex("42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985"),
        );
        assert_eq!(sealed[ct_len..].to_vec(), hex("4d5c2af327cd64a62cf35abd2ba6fab4"));
        // Round-trip.
        let opened = aes128_gcm_open(&key, &nonce, &[], &sealed).unwrap();
        assert_eq!(opened, pt);
        // Tamper detection.
        let mut bad = sealed.clone();
        bad[0] ^= 1;
        assert!(aes128_gcm_open(&key, &nonce, &[], &bad).is_none());
    }

    #[test]
    fn aes128_gcm_mcgrew_case4_with_aad() {
        // NIST GCM Test Case 4 (with AAD, truncated plaintext).
        let key: [u8; 16] = {
            let mut k = [0u8; 16];
            k.copy_from_slice(&hex("feffe9928665731c6d6a8f9467308308"));
            k
        };
        let nonce: [u8; 12] = {
            let mut n = [0u8; 12];
            n.copy_from_slice(&hex("cafebabefacedbaddecaf888"));
            n
        };
        let aad = hex("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let pt = hex(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        );
        let sealed = aes128_gcm_seal(&key, &nonce, &aad, &pt);
        assert_eq!(&sealed[sealed.len() - 16..], &hex("5bc94fbc3221a5db94fae95ae7121a47")[..]);
        let opened = aes128_gcm_open(&key, &nonce, &aad, &sealed).unwrap();
        assert_eq!(opened, pt);
    }
}


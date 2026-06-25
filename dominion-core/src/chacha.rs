//! ChaCha20-Poly1305 AEAD (RFC 8439) — the post-quantum-resilient cipher suite.
//!
//! This is a second, *cryptographically independent* AEAD alongside the AES-256-GCM
//! in [`crate::memcrypt`]. Independence is the point: AES-GCM rests on the AES
//! S-box and GHASH over GF(2¹²⁸); ChaCha20-Poly1305 rests on an ARX permutation and
//! a prime-field (2¹³⁰−5) MAC. A break in one family does not touch the other, so
//! the vault can stay encrypted under whichever is trusted — true crypto agility.
//!
//! On the post-quantum question: AEAD is *symmetric*, so the only quantum threat is
//! Grover's algorithm, which merely halves the effective key length. A 256-bit key
//! therefore retains ~128-bit security against a quantum adversary. Both this suite
//! and AES-256-GCM clear that bar; ChaCha20-Poly1305 adds family diversity and, in
//! portable software with no AES-NI, runs faster than table-driven AES-GCM.
//!
//! From-scratch, safe `no_std + alloc`, validated against the RFC 8439 test vectors.

use alloc::vec::Vec;

// ───────────────────────────── ChaCha20 ─────────────────────────────

/// "expand 32-byte k" — the ChaCha20 sigma constants.
const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

#[inline(always)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(7);
}

/// The ChaCha20 block function: 64 bytes of keystream for `(key, counter, nonce)`.
fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut state = [0u32; 16];
    state[..4].copy_from_slice(&SIGMA);
    for i in 0..8 {
        state[4 + i] =
            u32::from_le_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }
    state[12] = counter;
    for i in 0..3 {
        state[13 + i] =
            u32::from_le_bytes([nonce[4 * i], nonce[4 * i + 1], nonce[4 * i + 2], nonce[4 * i + 3]]);
    }

    let mut w = state;
    for _ in 0..10 {
        // Column rounds.
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        // Diagonal rounds.
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }

    let mut out = [0u8; 64];
    for i in 0..16 {
        out[4 * i..4 * i + 4].copy_from_slice(&w[i].wrapping_add(state[i]).to_le_bytes());
    }
    out
}

/// XOR `data` with the ChaCha20 keystream starting at `counter` (its own inverse).
fn chacha20_xor(key: &[u8; 32], counter: u32, nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for (i, chunk) in out.chunks_mut(64).enumerate() {
        let ks = chacha20_block(key, counter.wrapping_add(i as u32), nonce);
        for (b, k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= *k;
        }
    }
    out
}

// ───────────────────────────── Poly1305 ─────────────────────────────

/// A streaming Poly1305 one-time authenticator (130-bit accumulator held as five
/// 26-bit limbs, products in 64-bit — the "donna" layout). Fed incrementally so
/// the AEAD can authenticate ciphertext as it is produced, without buffering the
/// whole MAC input.
struct Poly1305 {
    r: [u32; 5],
    rs: [u32; 4], // r[1..5] each pre-multiplied by 5 (the mod-(2¹³⁰−5) folding term)
    h: [u32; 5],
    pad: [u32; 4], // the "s" half of the one-time key, added at finalisation
    buffer: [u8; 16],
    leftover: usize,
}

impl Poly1305 {
    fn new(key: &[u8; 32]) -> Poly1305 {
        // Clamp r per RFC 8439 while splitting it into 26-bit limbs.
        let r0 = u32::from_le_bytes([key[0], key[1], key[2], key[3]]) & 0x3ff_ffff;
        let r1 = (u32::from_le_bytes([key[3], key[4], key[5], key[6]]) >> 2) & 0x3ff_ff03;
        let r2 = (u32::from_le_bytes([key[6], key[7], key[8], key[9]]) >> 4) & 0x3ff_c0ff;
        let r3 = (u32::from_le_bytes([key[9], key[10], key[11], key[12]]) >> 6) & 0x3f0_3fff;
        let r4 = (u32::from_le_bytes([key[12], key[13], key[14], key[15]]) >> 8) & 0x00f_ffff;
        let pad = [
            u32::from_le_bytes([key[16], key[17], key[18], key[19]]),
            u32::from_le_bytes([key[20], key[21], key[22], key[23]]),
            u32::from_le_bytes([key[24], key[25], key[26], key[27]]),
            u32::from_le_bytes([key[28], key[29], key[30], key[31]]),
        ];
        Poly1305 {
            r: [r0, r1, r2, r3, r4],
            rs: [
                r1.wrapping_mul(5),
                r2.wrapping_mul(5),
                r3.wrapping_mul(5),
                r4.wrapping_mul(5),
            ],
            h: [0; 5],
            pad,
            buffer: [0; 16],
            leftover: 0,
        }
    }

    /// Absorb one 16-byte block. `hibit` is `1<<24` for a full block (the implicit
    /// 2¹²⁸ term) and `0` for a zero-padded final block.
    fn block(&mut self, block: &[u8; 16], hibit: u32) {
        let [r0, r1, r2, r3, r4] = self.r;
        let [s1, s2, s3, s4] = self.rs;

        let t0 = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let t1 = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let t2 = u32::from_le_bytes([block[8], block[9], block[10], block[11]]);
        let t3 = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);

        let mut h0 = self.h[0].wrapping_add(t0 & 0x3ff_ffff);
        let mut h1 = self.h[1].wrapping_add(((t0 >> 26) | (t1 << 6)) & 0x3ff_ffff);
        let mut h2 = self.h[2].wrapping_add(((t1 >> 20) | (t2 << 12)) & 0x3ff_ffff);
        let mut h3 = self.h[3].wrapping_add(((t2 >> 14) | (t3 << 18)) & 0x3ff_ffff);
        let mut h4 = self.h[4].wrapping_add((t3 >> 8) | hibit);

        let m = |a: u32, b: u32| (a as u64) * (b as u64);
        let d0 = m(h0, r0) + m(h1, s4) + m(h2, s3) + m(h3, s2) + m(h4, s1);
        let mut d1 = m(h0, r1) + m(h1, r0) + m(h2, s4) + m(h3, s3) + m(h4, s2);
        let mut d2 = m(h0, r2) + m(h1, r1) + m(h2, r0) + m(h3, s4) + m(h4, s3);
        let mut d3 = m(h0, r3) + m(h1, r2) + m(h2, r1) + m(h3, r0) + m(h4, s4);
        let mut d4 = m(h0, r4) + m(h1, r3) + m(h2, r2) + m(h3, r1) + m(h4, r0);

        let mut c = (d0 >> 26) as u32;
        h0 = (d0 as u32) & 0x3ff_ffff;
        d1 += c as u64; c = (d1 >> 26) as u32; h1 = (d1 as u32) & 0x3ff_ffff;
        d2 += c as u64; c = (d2 >> 26) as u32; h2 = (d2 as u32) & 0x3ff_ffff;
        d3 += c as u64; c = (d3 >> 26) as u32; h3 = (d3 as u32) & 0x3ff_ffff;
        d4 += c as u64; c = (d4 >> 26) as u32; h4 = (d4 as u32) & 0x3ff_ffff;
        h0 = h0.wrapping_add(c.wrapping_mul(5));
        c = h0 >> 26; h0 &= 0x3ff_ffff; h1 = h1.wrapping_add(c);

        self.h = [h0, h1, h2, h3, h4];
    }

    fn update(&mut self, mut data: &[u8]) {
        if self.leftover > 0 {
            let take = core::cmp::min(16 - self.leftover, data.len());
            self.buffer[self.leftover..self.leftover + take].copy_from_slice(&data[..take]);
            self.leftover += take;
            data = &data[take..];
            if self.leftover < 16 {
                return;
            }
            let blk = self.buffer;
            self.block(&blk, 1 << 24);
            self.leftover = 0;
        }
        let mut chunks = data.chunks_exact(16);
        for chunk in chunks.by_ref() {
            self.block(chunk.try_into().unwrap(), 1 << 24);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            self.buffer[..rem.len()].copy_from_slice(rem);
            self.leftover = rem.len();
        }
    }

    fn finalize(mut self) -> [u8; 16] {
        if self.leftover > 0 {
            let mut blk = [0u8; 16];
            blk[..self.leftover].copy_from_slice(&self.buffer[..self.leftover]);
            blk[self.leftover] = 1;
            self.block(&blk, 0);
        }

        let [mut h0, mut h1, mut h2, mut h3, mut h4] = self.h;

        // Fully carry h.
        let mut c;
        c = h1 >> 26; h1 &= 0x3ff_ffff; h2 = h2.wrapping_add(c);
        c = h2 >> 26; h2 &= 0x3ff_ffff; h3 = h3.wrapping_add(c);
        c = h3 >> 26; h3 &= 0x3ff_ffff; h4 = h4.wrapping_add(c);
        c = h4 >> 26; h4 &= 0x3ff_ffff; h0 = h0.wrapping_add(c.wrapping_mul(5));
        c = h0 >> 26; h0 &= 0x3ff_ffff; h1 = h1.wrapping_add(c);

        // g = h - p (p = 2¹³⁰ − 5), computed as h + 5 with a borrow out of bit 130.
        let mut g0 = h0.wrapping_add(5); c = g0 >> 26; g0 &= 0x3ff_ffff;
        let mut g1 = h1.wrapping_add(c); c = g1 >> 26; g1 &= 0x3ff_ffff;
        let mut g2 = h2.wrapping_add(c); c = g2 >> 26; g2 &= 0x3ff_ffff;
        let mut g3 = h3.wrapping_add(c); c = g3 >> 26; g3 &= 0x3ff_ffff;
        let mut g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

        // mask = 0 if h < p (a borrow occurred → keep h), all-ones otherwise (use g).
        let mask = (g4 >> 31).wrapping_sub(1);
        g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
        let nmask = !mask;
        h0 = (h0 & nmask) | g0;
        h1 = (h1 & nmask) | g1;
        h2 = (h2 & nmask) | g2;
        h3 = (h3 & nmask) | g3;
        h4 = (h4 & nmask) | g4;

        // Repack the five 26-bit limbs into four 32-bit words.
        let mut w0 = h0 | (h1 << 26);
        let mut w1 = (h1 >> 6) | (h2 << 20);
        let mut w2 = (h2 >> 12) | (h3 << 14);
        let mut w3 = (h3 >> 18) | (h4 << 8);

        // tag = (h + pad) mod 2¹²⁸.
        let mut f = w0 as u64 + self.pad[0] as u64; w0 = f as u32;
        f = w1 as u64 + self.pad[1] as u64 + (f >> 32); w1 = f as u32;
        f = w2 as u64 + self.pad[2] as u64 + (f >> 32); w2 = f as u32;
        f = w3 as u64 + self.pad[3] as u64 + (f >> 32); w3 = f as u32;

        let mut tag = [0u8; 16];
        tag[0..4].copy_from_slice(&w0.to_le_bytes());
        tag[4..8].copy_from_slice(&w1.to_le_bytes());
        tag[8..12].copy_from_slice(&w2.to_le_bytes());
        tag[12..16].copy_from_slice(&w3.to_le_bytes());
        tag
    }
}

/// One-shot Poly1305 (used by tests and any non-AEAD callers).
pub fn poly1305_mac(msg: &[u8], key: &[u8; 32]) -> [u8; 16] {
    let mut p = Poly1305::new(key);
    p.update(msg);
    p.finalize()
}

// ─────────────────────── ChaCha20-Poly1305 AEAD ───────────────────────

const ZERO_PAD: [u8; 16] = [0u8; 16];

/// Bytes needed to round `n` up to a 16-byte boundary.
fn pad16(n: usize) -> usize {
    (16 - (n % 16)) % 16
}

/// Constant-time 16-byte comparison.
fn ct_eq16(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut diff = 0u8;
    for i in 0..16 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Absorb the ciphertext zero-padding and the AAD/ciphertext length block, then
/// produce the tag.
fn finish_mac(mut poly: Poly1305, aad_len: usize, ct_len: usize) -> [u8; 16] {
    poly.update(&ZERO_PAD[..pad16(ct_len)]);
    let mut lens = [0u8; 16];
    lens[..8].copy_from_slice(&(aad_len as u64).to_le_bytes());
    lens[8..].copy_from_slice(&(ct_len as u64).to_le_bytes());
    poly.update(&lens);
    poly.finalize()
}

/// ChaCha20-Poly1305 AEAD encryption (RFC 8439) with a 96-bit nonce. Returns
/// `(ciphertext, tag)`. Encryption and authentication are fused into one pass:
/// each ciphertext block is absorbed into Poly1305 as it is produced.
pub fn aead_encrypt(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
    // The one-time Poly1305 key is the first 32 bytes of keystream block 0.
    let block0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&block0[..32]);

    let mut poly = Poly1305::new(&otk);
    poly.update(aad);
    poly.update(&ZERO_PAD[..pad16(aad.len())]);

    let mut out = plaintext.to_vec();
    for (i, chunk) in out.chunks_mut(64).enumerate() {
        let ks = chacha20_block(key, 1 + i as u32, nonce);
        for (b, k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= *k;
        }
        poly.update(chunk);
    }

    let ct_len = out.len();
    let tag = finish_mac(poly, aad.len(), ct_len);
    (out, tag)
}

/// ChaCha20-Poly1305 AEAD decryption. Verifies the tag before releasing any
/// plaintext; returns `None` on any authentication failure.
pub fn aead_decrypt(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Option<Vec<u8>> {
    let block0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&block0[..32]);

    let mut poly = Poly1305::new(&otk);
    poly.update(aad);
    poly.update(&ZERO_PAD[..pad16(aad.len())]);
    poly.update(ciphertext);
    let expected = finish_mac(poly, aad.len(), ciphertext.len());
    if !ct_eq16(&expected, tag) {
        return None;
    }
    Some(chacha20_xor(key, 1, nonce, ciphertext))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        (0..s.len())
            .step_by(2)
            .map(|i| {
                let hi = (s[i] as char).to_digit(16).unwrap();
                let lo = (s[i + 1] as char).to_digit(16).unwrap();
                (hi * 16 + lo) as u8
            })
            .collect()
    }

    #[test]
    fn chacha20_block_rfc8439_2_3_2() {
        let key: [u8; 32] = (0u8..32).collect::<Vec<_>>().try_into().unwrap();
        let nonce: [u8; 12] = hex("000000090000004a00000000").try_into().unwrap();
        let block = chacha20_block(&key, 1, &nonce);
        let expected = hex(
            "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c0680304\
             22aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b512\
             9cd1de164eb9cbd083e8a2503c4e",
        );
        assert_eq!(block.to_vec(), expected);
    }

    #[test]
    fn poly1305_rfc8439_2_5_2() {
        let key: [u8; 32] = hex(
            "85d6be7857556d337f4452fe42d506a8\
             0103808afb0db2fd4abff6af4149f51b",
        )
        .try_into()
        .unwrap();
        let msg = b"Cryptographic Forum Research Group";
        assert_eq!(
            poly1305_mac(msg, &key).to_vec(),
            hex("a8061dc1305136c6c22b8baf0c0127a9")
        );
    }

    #[test]
    fn aead_rfc8439_2_8_2() {
        let key: [u8; 32] = (0x80u8..0xa0).collect::<Vec<_>>().try_into().unwrap();
        let nonce: [u8; 12] = hex("070000004041424344454647").try_into().unwrap();
        let aad = hex("50515253c0c1c2c3c4c5c6c7");
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could \
            offer you only one tip for the future, sunscreen would be it.";
        let (ct, tag) = aead_encrypt(&key, &nonce, &aad, plaintext);
        let expected_ct = hex(
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6\
             3dbea45e8ca967128 2fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36\
             92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc\
             3ff4def08e4b7a9de576d26586cec64b6116",
        );
        assert_eq!(ct, expected_ct);
        assert_eq!(tag.to_vec(), hex("1ae10b594f09e26a7e902ecbd0600691"));
        // Round-trip and tamper detection.
        assert_eq!(aead_decrypt(&key, &nonce, &aad, &ct, &tag).as_deref(), Some(&plaintext[..]));
        let mut bad = ct.clone();
        bad[0] ^= 1;
        assert!(aead_decrypt(&key, &nonce, &aad, &bad, &tag).is_none());
        assert!(aead_decrypt(&key, &nonce, b"wrong-aad", &ct, &tag).is_none());
    }

    #[test]
    fn nonce_reuse_leaks_the_xor_of_plaintexts() {
        // This documents *why* the vault carries a key+nonce-reuse guard. Reusing a
        // (key, nonce) means the same keystream encrypts two messages, so the
        // ciphertexts' XOR equals the plaintexts' XOR — a classic two-time pad that
        // hands an eavesdropper the relationship between the messages for free.
        let key = [0x24u8; 32];
        let nonce = [0x42u8; 12];
        let p1 = b"attack at dawn!!";
        let p2 = b"retreat at noon.";
        let (c1, _) = aead_encrypt(&key, &nonce, &[], p1);
        let (c2, _) = aead_encrypt(&key, &nonce, &[], p2);

        let ct_xor: Vec<u8> = c1.iter().zip(c2.iter()).map(|(a, b)| a ^ b).collect();
        let pt_xor: Vec<u8> = p1.iter().zip(p2.iter()).map(|(a, b)| a ^ b).collect();
        assert_eq!(ct_xor, pt_xor, "keystream reuse leaks plaintext XOR");

        // With distinct nonces the leak vanishes (keystreams differ).
        let (c3, _) = aead_encrypt(&key, &[0x43u8; 12], &[], p2);
        let ct_xor2: Vec<u8> = c1.iter().zip(c3.iter()).map(|(a, b)| a ^ b).collect();
        assert_ne!(ct_xor2, pt_xor);
    }

    #[test]
    fn round_trips_across_lengths() {
        let key = [7u8; 32];
        let nonce = [9u8; 12];
        for len in [0usize, 1, 15, 16, 17, 63, 64, 65, 200] {
            let pt: Vec<u8> = (0..len).map(|i| (i * 7 + 1) as u8).collect();
            let (ct, tag) = aead_encrypt(&key, &nonce, b"hdr", &pt);
            assert_eq!(aead_decrypt(&key, &nonce, b"hdr", &ct, &tag).as_deref(), Some(pt.as_slice()));
        }
    }
}

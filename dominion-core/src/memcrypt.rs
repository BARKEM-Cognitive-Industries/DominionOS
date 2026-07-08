//! Authenticated encryption + **memory-at-rest** sealing (see
//! `docs/security/universal-encryption.md` and the memory-encryption design note).
//!
//! Two things live here:
//!
//! 1. A from-scratch, safe-Rust **AES-128/256** block cipher and **AES-GCM** AEAD,
//!    validated against the FIPS-197 and NIST GCM known-answer vectors. This is the
//!    real algorithm (S-box, key schedule, GHASH over GF(2¹²⁸)), not a model — it
//!    just runs in portable integer code, so it needs **no AES-NI and no
//!    confidential-compute hardware**. Where the CPU has acceleration it can be
//!    swapped in behind the same interface; where it does not, this is the floor.
//!
//! 2. A [`SealedRegion`]: data that stays **encrypted while resident** and is only
//!    decrypted *at read time*, into a value handed to the capability holder, then
//!    dropped. This realises "all memory at rest is encrypted; plaintext exists
//!    only for the instant a holder is reading it."
//!
//! Pure, safe, `no_std + alloc`, host-tested against standard vectors.

use alloc::vec;
use alloc::vec::Vec;

// ───────────────────────────── AES core ─────────────────────────────

#[rustfmt::skip]
const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

const RCON: [u8; 11] = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// An AES key schedule (128- or 256-bit). Forward (encryption) direction only —
/// all our modes (CTR, GCM) use the cipher in the forward direction. Round keys
/// are stored as four little-endian-packed column words so the T-table round can
/// XOR them directly.
pub struct Aes {
    round_keys: Vec<[u32; 4]>,
    rounds: usize,
}

const fn xtime(x: u8) -> u8 {
    let h = x >> 7;
    (x << 1) ^ (h * 0x1b)
}

/// The AES "T-tables": each entry folds SubBytes and one column's MixColumns
/// contribution into a single word, turning a round into four table lookups and
/// XORs per column instead of per-byte S-box + GF arithmetic. The four tables are
/// byte rotations of each other (`TE{r}[x] = TE0[x].rotate_left(8*r)`); keeping
/// all four materialised (4 KiB) lets the round avoid a rotate on every lookup,
/// which measured ~7–8% faster than rotating a single table.
///
/// Note: this is a table-lookup core, so it is not constant-time against a
/// cache-timing adversary. It is the portable software floor; on hardware with
/// AES-NI the accelerated path would be selected instead.
const fn build_te0() -> [u32; 256] {
    let mut te = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let s = SBOX[i] as u32;
        let s2 = xtime(SBOX[i]) as u32;
        let s3 = s2 ^ s;
        // Packed little-endian as (2·s, s, s, 3·s) = MixColumns of (s,0,0,0).
        te[i] = s2 | (s << 8) | (s << 16) | (s3 << 24);
        i += 1;
    }
    te
}

const fn rotl_table(src: &[u32; 256], n: u32) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = src[i].rotate_left(n);
        i += 1;
    }
    t
}

const TE0: [u32; 256] = build_te0();
const TE1: [u32; 256] = rotl_table(&TE0, 8);
const TE2: [u32; 256] = rotl_table(&TE0, 16);
const TE3: [u32; 256] = rotl_table(&TE0, 24);

impl Aes {
    /// AES-128 from a 16-byte key.
    pub fn new_128(key: &[u8; 16]) -> Aes {
        Aes::expand(key, 4, 10)
    }

    /// AES-256 from a 32-byte key.
    pub fn new_256(key: &[u8; 32]) -> Aes {
        Aes::expand(key, 8, 14)
    }

    fn expand(key: &[u8], nk: usize, rounds: usize) -> Aes {
        let total_words = 4 * (rounds + 1);
        let mut w = vec![[0u8; 4]; total_words];
        for i in 0..nk {
            w[i] = [key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]];
        }
        for i in nk..total_words {
            let mut temp = w[i - 1];
            if i % nk == 0 {
                // RotWord + SubWord + Rcon
                temp = [temp[1], temp[2], temp[3], temp[0]];
                for t in &mut temp {
                    *t = SBOX[*t as usize];
                }
                temp[0] ^= RCON[i / nk];
            } else if nk > 6 && i % nk == 4 {
                for t in &mut temp {
                    *t = SBOX[*t as usize];
                }
            }
            for j in 0..4 {
                w[i][j] = w[i - nk][j] ^ temp[j];
            }
        }
        let round_keys = (0..=rounds)
            .map(|r| {
                let mut rk = [0u32; 4];
                for c in 0..4 {
                    rk[c] = u32::from_le_bytes(w[4 * r + c]);
                }
                rk
            })
            .collect();
        Aes { round_keys, rounds }
    }

    /// Encrypt a single 16-byte block via the T-table fast path. State columns are
    /// held as little-endian words (byte `n` of a word = row `n` of that column),
    /// matching the column-major AES state and the packed round keys.
    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let rk = &self.round_keys;
        // Load columns and apply the initial round key.
        let mut s = [
            u32::from_le_bytes([block[0], block[1], block[2], block[3]]) ^ rk[0][0],
            u32::from_le_bytes([block[4], block[5], block[6], block[7]]) ^ rk[0][1],
            u32::from_le_bytes([block[8], block[9], block[10], block[11]]) ^ rk[0][2],
            u32::from_le_bytes([block[12], block[13], block[14], block[15]]) ^ rk[0][3],
        ];

        let mut t = [0u32; 4];
        for rkr in &rk[1..self.rounds] {
            for c in 0..4 {
                // ShiftRows picks row n of column (c+n) mod 4.
                let b0 = (s[c] & 0xff) as usize;
                let b1 = ((s[(c + 1) & 3] >> 8) & 0xff) as usize;
                let b2 = ((s[(c + 2) & 3] >> 16) & 0xff) as usize;
                let b3 = ((s[(c + 3) & 3] >> 24) & 0xff) as usize;
                t[c] = TE0[b0] ^ TE1[b1] ^ TE2[b2] ^ TE3[b3] ^ rkr[c];
            }
            s = t;
        }

        // Final round: SubBytes + ShiftRows + AddRoundKey, no MixColumns.
        for c in 0..4 {
            let b0 = (s[c] & 0xff) as usize;
            let b1 = ((s[(c + 1) & 3] >> 8) & 0xff) as usize;
            let b2 = ((s[(c + 2) & 3] >> 16) & 0xff) as usize;
            let b3 = ((s[(c + 3) & 3] >> 24) & 0xff) as usize;
            t[c] = (SBOX[b0] as u32)
                | ((SBOX[b1] as u32) << 8)
                | ((SBOX[b2] as u32) << 16)
                | ((SBOX[b3] as u32) << 24);
            t[c] ^= rk[self.rounds][c];
        }

        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&t[0].to_le_bytes());
        out[4..8].copy_from_slice(&t[1].to_le_bytes());
        out[8..12].copy_from_slice(&t[2].to_le_bytes());
        out[12..16].copy_from_slice(&t[3].to_le_bytes());
        out
    }
}

// ───────────────────────────── GCM ─────────────────────────────

/// Reduction values for the trailing nibble in the 4-bit windowed multiply.
const LAST4: [u64; 16] = [
    0x0000, 0x1c20, 0x3840, 0x2460, 0x7080, 0x6ca0, 0x48c0, 0x54e0, 0xe100, 0xfd20, 0xd940, 0xc560,
    0x9180, 0x8da0, 0xa9c0, 0xb5e0,
];

/// Precomputed multiples of the hash subkey `H`, indexed by a 4-bit value, split
/// into high/low 64-bit halves (Shoup's method). Built once per GCM operation;
/// it turns each GF(2¹²⁸) multiply from 128 bit-serial steps into 32 table-driven
/// nibble steps while producing bit-identical GHASH output.
struct GhashTable {
    hh: [u64; 16],
    hl: [u64; 16],
}

fn ghash_table(h: u128) -> GhashTable {
    let hb = h.to_be_bytes();
    let mut hi = u64::from_be_bytes(hb[0..8].try_into().unwrap());
    let mut lo = u64::from_be_bytes(hb[8..16].try_into().unwrap());
    let mut hh = [0u64; 16];
    let mut hl = [0u64; 16];
    // Index 8 = H; halve down to fill 4, 2, 1; then fill composites by XOR.
    hh[8] = hi;
    hl[8] = lo;
    let mut i = 4;
    while i > 0 {
        let t = ((lo & 1) as u32) * 0xe100_0000;
        let vl = (hi << 63) | (lo >> 1);
        let vh = (hi >> 1) ^ ((t as u64) << 32);
        hl[i] = vl;
        hh[i] = vh;
        lo = vl;
        hi = vh;
        i >>= 1;
    }
    let mut i = 2;
    while i <= 8 {
        let (base_l, base_h) = (hl[i], hh[i]);
        for j in 1..i {
            hh[i + j] = base_h ^ hh[j];
            hl[i + j] = base_l ^ hl[j];
        }
        i *= 2;
    }
    GhashTable { hh, hl }
}

fn ghash_mul(t: &GhashTable, x: u128) -> u128 {
    let xb = x.to_be_bytes();
    let lo0 = (xb[15] & 0xf) as usize;
    let mut zh = t.hh[lo0];
    let mut zl = t.hl[lo0];
    for i in (0..16).rev() {
        let lo = (xb[i] & 0xf) as usize;
        let hi = ((xb[i] >> 4) & 0xf) as usize;
        if i != 15 {
            let rem = (zl & 0xf) as usize;
            zl = (zh << 60) | (zl >> 4);
            zh >>= 4;
            zh ^= LAST4[rem] << 48;
            zh ^= t.hh[lo];
            zl ^= t.hl[lo];
        }
        let rem = (zl & 0xf) as usize;
        zl = (zh << 60) | (zl >> 4);
        zh >>= 4;
        zh ^= LAST4[rem] << 48;
        zh ^= t.hh[hi];
        zl ^= t.hl[hi];
    }
    ((zh as u128) << 64) | (zl as u128)
}

/// Absorb every 16-byte block of `data` (zero-padded tail) into the accumulator.
fn ghash_feed(t: &GhashTable, y: &mut u128, data: &[u8]) {
    for chunk in data.chunks(16) {
        let mut block = [0u8; 16];
        block[..chunk.len()].copy_from_slice(chunk);
        *y = ghash_mul(t, *y ^ u128::from_be_bytes(block));
    }
}

fn ghash(h: u128, aad: &[u8], ct: &[u8]) -> u128 {
    let t = ghash_table(h);
    let mut y = 0u128;
    ghash_feed(&t, &mut y, aad);
    ghash_feed(&t, &mut y, ct);
    // Length block: 64-bit AAD bitlen || 64-bit ciphertext bitlen.
    let len_block = ((aad.len() as u128 * 8) << 64) | (ct.len() as u128 * 8);
    ghash_mul(&t, y ^ len_block)
}

fn inc32(block: &mut [u8; 16]) {
    let mut ctr = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);
    ctr = ctr.wrapping_add(1);
    block[12..16].copy_from_slice(&ctr.to_be_bytes());
}

fn gctr(aes: &Aes, icb: [u8; 16], data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let mut cb = icb;
    for chunk in out.chunks_mut(16) {
        let ks = aes.encrypt_block(&cb);
        for (b, k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= *k;
        }
        inc32(&mut cb);
    }
    out
}

/// AES-GCM encryption with a 96-bit (12-byte) IV. Returns `(ciphertext, tag)`.
///
/// Encryption and authentication are fused into a single pass: each ciphertext
/// block is GHASHed the moment it is produced, while it is still hot in cache, so
/// the data is traversed once instead of twice (CTR then a separate GHASH read).
pub fn gcm_encrypt(aes: &Aes, iv: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
    let h = u128::from_be_bytes(aes.encrypt_block(&[0u8; 16]));
    let table = ghash_table(h);
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(iv);
    j0[15] = 1;

    let mut y = 0u128;
    ghash_feed(&table, &mut y, aad);

    let mut out = plaintext.to_vec();
    let mut cb = j0;
    inc32(&mut cb);
    for chunk in out.chunks_mut(16) {
        let ks = aes.encrypt_block(&cb);
        let mut block = [0u8; 16];
        for (i, (b, k)) in chunk.iter_mut().zip(ks.iter()).enumerate() {
            *b ^= *k;
            block[i] = *b;
        }
        y = ghash_mul(&table, y ^ u128::from_be_bytes(block));
        inc32(&mut cb);
    }

    let len_block = ((aad.len() as u128 * 8) << 64) | (out.len() as u128 * 8);
    y = ghash_mul(&table, y ^ len_block);
    let ej0 = u128::from_be_bytes(aes.encrypt_block(&j0));
    let tag = (y ^ ej0).to_be_bytes();
    (out, tag)
}

/// AES-GCM decryption. Returns `None` if the authentication tag does not verify
/// (the ciphertext was tampered with, or the wrong key/AAD was used).
pub fn gcm_decrypt(
    aes: &Aes,
    iv: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Option<Vec<u8>> {
    let h = u128::from_be_bytes(aes.encrypt_block(&[0u8; 16]));
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(iv);
    j0[15] = 1;
    let s = ghash(h, aad, ciphertext);
    let ej0 = u128::from_be_bytes(aes.encrypt_block(&j0));
    let expected = (s ^ ej0).to_be_bytes();
    // Constant-time tag comparison.
    if !ct_eq(&expected, tag) {
        return None;
    }
    let mut icb = j0;
    inc32(&mut icb);
    Some(gctr(aes, icb, ciphertext))
}

fn ct_eq(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut diff = 0u8;
    for i in 0..16 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ──────────────────── Memory-at-rest sealing ────────────────────

/// Derive a `u64` region salt from a label that is unique per key.
///
/// Use this when the label itself is already unique among all `SealedRegion`
/// instances that share the same encryption key — for example when each region
/// is identified by a distinct label string.  The fold is a simple FNV-1a–
/// style mix; it is not a cryptographic hash, but it is sufficient as a nonce
/// differentiator because AES-GCM only requires IV *uniqueness*, not
/// unpredictability.
pub fn salt_from_label(label: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &b in label {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    h
}

/// A region of memory kept **encrypted at rest**. The plaintext is materialised
/// only inside [`open`](SealedRegion::open) — handed to the reader and dropped —
/// so it never lingers in cleartext between accesses. Each reseal uses a fresh
/// nonce, so identical plaintext does not yield identical ciphertext.
///
/// # Nonce construction
///
/// The 96-bit AES-GCM IV is split as:
///   bytes  0..4  — big-endian per-reseal counter (starts at 1, increments on every
///                  reseal; up to 2^32 reseals per region before it wraps)
///   bytes  4..12 — the full 64-bit `region_salt` (big-endian)
///
/// Two `SealedRegion` instances that share the same encryption key **must** be
/// constructed with distinct `region_salt` values; otherwise they would both start
/// at counter=1 and produce the same (key, IV) pair on first seal — an AES-GCM
/// catastrophic nonce reuse. Callers must derive this value from something that is
/// unique per region lifetime: an object ID, a label hash, or a monotonic counter.
pub struct SealedRegion {
    key: [u8; 32],
    nonce_ctr: u64,
    /// Per-region salt baked into the low 8 bytes of the IV.  Must be distinct
    /// across all `SealedRegion` instances that share the same `key`.
    region_salt: u64,
    iv: [u8; 12],
    ciphertext: Vec<u8>,
    tag: [u8; 16],
    aad: Vec<u8>,
}

impl SealedRegion {
    /// Seal `plaintext` under a 256-bit key.
    ///
    /// `label` is bound as authenticated associated data (it cannot be swapped
    /// without invalidating the tag).
    ///
    /// `region_salt` must be **unique per key**: every `SealedRegion` created
    /// under the same `key` must receive a different salt so their nonce
    /// sequences cannot collide.  Suitable values include an object/region ID,
    /// the low 64 bits of a monotonic allocation counter, or a hash of the
    /// label when the label itself is already unique per key.
    pub fn seal(key: [u8; 32], label: &[u8], region_salt: u64, plaintext: &[u8]) -> SealedRegion {
        // salt=0 is the zero-value / forgotten-salt sentinel; reject it so callers
        // notice immediately in dev/test rather than silently sharing a nonce sequence
        // with any other region that also forgot to set a salt.
        debug_assert!(
            region_salt != 0,
            "SealedRegion::seal: region_salt must be non-zero; \
             a salt of 0 is the default/unset value and will cause nonce reuse if \
             two regions are created under the same key without an explicit salt"
        );
        let mut region = SealedRegion {
            key,
            nonce_ctr: 0,
            region_salt,
            iv: [0u8; 12],
            ciphertext: Vec::new(),
            tag: [0u8; 16],
            aad: label.to_vec(),
        };
        region.reseal(plaintext);
        region
    }

    fn next_iv(&mut self) -> [u8; 12] {
        self.nonce_ctr += 1;
        // A 12-byte IV cannot hold both a full 64-bit counter and a full 64-bit
        // salt, so we spend 4 bytes on the counter and the remaining 8 on the
        // *entire* salt. Binding all 64 bits of the salt is what prevents the
        // catastrophic nonce reuse that a low-32-bit-only salt allowed: two salts
        // colliding only above bit 32 no longer produce an identical (key, IV).
        // The cost is a 2^32-reseal ceiling per region before the counter wraps.
        debug_assert!(
            self.nonce_ctr <= u32::MAX as u64,
            "SealedRegion: exceeded 2^32 reseals for one region; nonce counter wrapped"
        );
        let mut iv = [0u8; 12];
        // Bytes 0..4: big-endian per-reseal counter (low 32 bits) — unique within
        // this region across up to 2^32 reseals.
        iv[..4].copy_from_slice(&(self.nonce_ctr as u32).to_be_bytes());
        // Bytes 4..12: the full 64-bit per-region salt — unique across regions that
        // share the same key, preventing cross-region nonce collisions.
        iv[4..12].copy_from_slice(&self.region_salt.to_be_bytes());
        iv
    }

    /// Re-encrypt new contents with a fresh nonce (e.g. after a write).
    pub fn reseal(&mut self, plaintext: &[u8]) {
        let aes = Aes::new_256(&self.key);
        self.iv = self.next_iv();
        let (ct, tag) = gcm_encrypt(&aes, &self.iv, &self.aad, plaintext);
        self.ciphertext = ct;
        self.tag = tag;
    }

    /// Decrypt at read time. Returns `None` if the region was tampered with.
    /// The returned `Vec` is the only plaintext copy and is the caller's to drop.
    pub fn open(&self) -> Option<Vec<u8>> {
        let aes = Aes::new_256(&self.key);
        gcm_decrypt(&aes, &self.iv, &self.aad, &self.ciphertext, &self.tag)
    }

    /// The bytes as they sit in memory — ciphertext, never plaintext.
    pub fn at_rest(&self) -> &[u8] {
        &self.ciphertext
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn aes128_fips197_vector() {
        // FIPS-197 Appendix B / C.1.
        let key: [u8; 16] = hex("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
        let pt: [u8; 16] = hex("00112233445566778899aabbccddeeff").try_into().unwrap();
        let aes = Aes::new_128(&key);
        let ct = aes.encrypt_block(&pt);
        assert_eq!(ct.to_vec(), hex("69c4e0d86a7b0430d8cdb78070b4c55a"));
    }

    #[test]
    fn aes256_fips197_vector() {
        let key: [u8; 32] =
            hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
                .try_into()
                .unwrap();
        let pt: [u8; 16] = hex("00112233445566778899aabbccddeeff").try_into().unwrap();
        let aes = Aes::new_256(&key);
        let ct = aes.encrypt_block(&pt);
        assert_eq!(ct.to_vec(), hex("8ea2b7ca516745bfeafc49904b496089"));
    }

    #[test]
    fn gcm_nist_empty_vector() {
        // NIST GCM test case 1: zero key, zero IV, empty PT/AAD.
        let aes = Aes::new_128(&[0u8; 16]);
        let (ct, tag) = gcm_encrypt(&aes, &[0u8; 12], &[], &[]);
        assert!(ct.is_empty());
        assert_eq!(tag.to_vec(), hex("58e2fccefa7e3061367f1d57a4e7455a"));
    }

    #[test]
    fn gcm_nist_one_block_vector() {
        // NIST GCM test case 2: zero key/IV, one zero PT block, empty AAD.
        let aes = Aes::new_128(&[0u8; 16]);
        let (ct, tag) = gcm_encrypt(&aes, &[0u8; 12], &[], &[0u8; 16]);
        assert_eq!(ct, hex("0388dace60b6a392f328c2b971b2fe78"));
        assert_eq!(tag.to_vec(), hex("ab6e47d42cec13bdf53a67b21257bddf"));
    }

    #[test]
    fn gcm_round_trip_and_auth() {
        let aes = Aes::new_256(&[7u8; 32]);
        let iv = [9u8; 12];
        let aad = b"region:vault/secret";
        let pt = b"the answer is forty two, kept secret";
        let (ct, tag) = gcm_encrypt(&aes, &iv, aad, pt);
        assert_eq!(gcm_decrypt(&aes, &iv, aad, &ct, &tag).unwrap(), pt);
        // Tamper the ciphertext → authentication fails.
        let mut bad = ct.clone();
        bad[0] ^= 1;
        assert!(gcm_decrypt(&aes, &iv, aad, &bad, &tag).is_none());
        // Tamper the AAD → authentication fails.
        assert!(gcm_decrypt(&aes, &iv, b"region:other", &ct, &tag).is_none());
    }

    #[test]
    fn sealed_region_only_exposes_plaintext_on_open() {
        let key = [42u8; 32];
        let region = SealedRegion::seal(key, b"label", 0x1234_5678_9abc_def0, b"private memory contents");
        // At rest, the bytes are ciphertext — not the plaintext.
        assert_ne!(region.at_rest(), b"private memory contents");
        assert!(!region
            .at_rest()
            .windows(7)
            .any(|w| w == b"private"));
        // Read time materialises the plaintext for the holder.
        assert_eq!(region.open().unwrap(), b"private memory contents");
    }

    #[test]
    fn reseal_uses_fresh_nonce() {
        let mut region = SealedRegion::seal([1u8; 32], b"l", 0xdead_beef_cafe_0001, b"same");
        let first = region.at_rest().to_vec();
        region.reseal(b"same");
        // Identical plaintext, different ciphertext (nonce advanced).
        assert_ne!(region.at_rest(), first.as_slice());
        assert_eq!(region.open().unwrap(), b"same");
    }

    #[test]
    fn distinct_region_salts_prevent_nonce_collision() {
        // Two SealedRegion instances created under the same key with *identical*
        // plaintext must produce different ciphertexts.  Equal ciphertexts would
        // mean the same (key, IV) was used — an AES-GCM total break.
        let key = [0x55u8; 32];
        // Use *identical* plaintext so that any ciphertext difference can only
        // come from a different nonce/IV, not from different data.
        let plaintext = b"the same secret message";

        let region_a = SealedRegion::seal(key, b"label", 0x0000_0000_0000_0001, plaintext);
        let region_b = SealedRegion::seal(key, b"label", 0x0000_0000_0000_0002, plaintext);

        // Different salts → different IVs → different ciphertexts even for
        // identical (key, plaintext, label).
        assert_ne!(
            region_a.at_rest(),
            region_b.at_rest(),
            "same-key regions with different salts must produce distinct ciphertexts"
        );

        // Both must still decrypt to the original plaintext.
        assert_eq!(region_a.open().unwrap(), plaintext);
        assert_eq!(region_b.open().unwrap(), plaintext);
    }

    /// Demonstrates the invariant callers MUST uphold: two `SealedRegion` instances
    /// created under the **same key and the same salt** start at the same counter
    /// sequence (counter=1, salt_bytes=same), so their first seal uses an identical
    /// (key, IV) pair.  For AES-GCM this is catastrophic: XOR of the two
    /// ciphertexts equals XOR of the two plaintexts, exposing both.
    ///
    /// This test is intentionally *not* a panic/failure test — `debug_assert` in
    /// `seal()` catches the zero-salt case in debug builds, but the general same-
    /// (non-zero)-salt misuse is a caller contract, not something we can detect
    /// without a global registry.  The test records the observable consequence so
    /// reviewers understand what the nonce-salt design is protecting against.
    #[test]
    fn same_salt_nonce_reuse_is_catastrophic() {
        let key = [0xAAu8; 32];
        let salt = 0xDEAD_BEEF_0000_0001u64; // same non-zero salt — deliberate misuse
        let plaintext_a = b"secret message ALPHA";
        let plaintext_b = b"secret message BETA!";

        let region_a = SealedRegion::seal(key, b"region-a", salt, plaintext_a);
        let region_b = SealedRegion::seal(key, b"region-b", salt, plaintext_b);

        // Same (key, IV) on first seal: XOR of ciphertexts == XOR of plaintexts.
        // This directly reveals the plaintexts to a passive observer — the nonce-
        // reuse catastrophe.  Verify the observable consequence:
        let ct_a = region_a.at_rest();
        let ct_b = region_b.at_rest();
        let len = plaintext_a.len().min(plaintext_b.len()).min(ct_a.len()).min(ct_b.len());
        let xor_ct: Vec<u8> = ct_a[..len].iter().zip(ct_b[..len].iter()).map(|(a, b)| a ^ b).collect();
        let xor_pt: Vec<u8> = plaintext_a[..len].iter().zip(plaintext_b[..len].iter()).map(|(a, b)| a ^ b).collect();
        assert_eq!(
            xor_ct, xor_pt,
            "XOR of ciphertexts must equal XOR of plaintexts under same (key, IV) — \
             this proves nonce reuse leaks both plaintexts; \
             callers MUST always provide distinct region_salt values per key"
        );
    }
}

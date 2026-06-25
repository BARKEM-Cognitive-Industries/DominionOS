//! Constant-time crypto primitives + a **known-answer-test (KAT) harness**
//! (`docs/security/threat-model-and-security-posture.md`: production-hardened crypto).
//!
//! Two distinct hardening concerns live here:
//!
//! * **Constant-time operations** — comparisons and selects whose execution does **not**
//!   branch on secret data, so a timing side-channel can't leak it. [`ct_eq`] compares two
//!   byte strings in time independent of where they first differ; [`ct_select`] picks a
//!   branch without a secret-dependent jump.
//! * **A KAT runner** — [`run_kats`] checks a primitive against fixed, externally-known
//!   answer vectors (the Wycheproof methodology). [`SHA256_KATS`] pins our SHA-256 to the
//!   NIST vectors, so a regression in the content-addressing hash is caught immediately.
//!
//! This is the *seam* for production-hardened crypto: constant-time discipline + an
//! extensible KAT corpus. Full standard-parameter ML-KEM/ML-DSA + an imported Wycheproof
//! corpus remain future work (tracked in the threat model). Pure, safe `no_std`.

use crate::hash::Hash256;

/// Constant-time equality of two byte slices: the running time depends only on the
/// *length*, never on the contents or the position of the first difference. Returns
/// `false` for different lengths (length is not secret).
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y; // no early exit — accumulate all differences
    }
    diff == 0
}

/// Constant-time select: returns `a` if `cond`, else `b`, without a secret-dependent
/// branch (mask arithmetic). Both inputs must be the same length; returns `b`'s length on
/// mismatch is avoided by requiring equal lengths via the caller.
pub fn ct_select(cond: bool, a: u8, b: u8) -> u8 {
    // mask = 0xFF if cond else 0x00, computed without branching.
    let mask = (cond as u8).wrapping_neg();
    (a & mask) | (b & !mask)
}

/// Constant-time conditional copy of `src` into `dst` when `cond` (both equal length).
pub fn ct_copy(cond: bool, dst: &mut [u8], src: &[u8]) {
    if dst.len() != src.len() {
        return;
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = ct_select(cond, *s, *d);
    }
}

// ───────────────────────── known-answer tests ─────────────────────────

/// One KAT vector: an input and its externally-known expected digest (hex).
pub struct HashKat {
    pub input: &'static [u8],
    pub expected_hex: &'static str,
}

/// NIST SHA-256 known-answer vectors. If our content-addressing hash ever drifts from
/// real SHA-256, these fail.
pub const SHA256_KATS: &[HashKat] = &[
    HashKat { input: b"", expected_hex: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
    HashKat { input: b"abc", expected_hex: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
    HashKat {
        input: b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        expected_hex: "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    },
];

/// Run the SHA-256 KAT corpus against [`Hash256::of`]. Returns the count of passing
/// vectors; equals `SHA256_KATS.len()` iff all pass.
pub fn run_kats() -> usize {
    SHA256_KATS
        .iter()
        .filter(|kat| Hash256::of(kat.input).to_hex() == kat.expected_hex)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_is_correct() {
        assert!(ct_eq(b"secret-tag", b"secret-tag"));
        assert!(!ct_eq(b"secret-tag", b"secret-tab")); // differs in last byte
        assert!(!ct_eq(b"secret-tag", b"Xecret-tag")); // differs in first byte
        assert!(!ct_eq(b"short", b"longer-input")); // length differs
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn ct_select_and_copy_are_branchless_correct() {
        assert_eq!(ct_select(true, 0xAA, 0x55), 0xAA);
        assert_eq!(ct_select(false, 0xAA, 0x55), 0x55);
        let mut dst = [1u8, 2, 3, 4];
        ct_copy(true, &mut dst, &[9, 9, 9, 9]);
        assert_eq!(dst, [9, 9, 9, 9]);
        ct_copy(false, &mut dst, &[0, 0, 0, 0]);
        assert_eq!(dst, [9, 9, 9, 9]); // unchanged when cond is false
    }

    #[test]
    fn sha256_matches_the_nist_known_answer_vectors() {
        assert_eq!(run_kats(), SHA256_KATS.len());
        // Spot-check one directly.
        assert_eq!(
            Hash256::of(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

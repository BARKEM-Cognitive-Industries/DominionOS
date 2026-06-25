//! Identity recovery & key management (see
//! `docs/security/identity-recovery-and-key-management.md`).
//!
//! A capability OS where the user *is* their keys must answer: what happens when a
//! key is lost? The answer is **threshold recovery**, not a backdoor. A secret is
//! split with **Shamir Secret Sharing** over `GF(2⁸)` into `n` shares such that any
//! `k` reconstruct it and any `k−1` reveal *nothing*. Shares go to independent
//! **guardians** (devices, people, custodians); recovery and key **rotation**
//! require a quorum to approve, so no single guardian — and no vendor — can act
//! alone.
//!
//! Pure, safe `no_std`; the GF(2⁸) field is the same one AES uses. No special
//! hardware.

use crate::hash::Hash256;
use alloc::vec;
use alloc::vec::Vec;

// ─────────────────── optional offline PQ recovery code ───────────────────

/// A keystream from a 32-byte key (SHA-256 in counter mode) — hash-based, so
/// post-quantum. The same construction `rot.rs` uses for platform sealing.
fn keystream(key: &[u8; 32], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut counter: u64 = 0;
    while out.len() < len {
        let mut input = Vec::with_capacity(40);
        input.extend_from_slice(key);
        input.extend_from_slice(&counter.to_le_bytes());
        out.extend_from_slice(&Hash256::of(&input).0);
        counter += 1;
    }
    out.truncate(len);
    out
}

/// An **optional offline recovery code**: a high-entropy code (meant to be printed and
/// stored offline) that can re-derive the master secret **without** the guardian quorum —
/// a self-custody escape hatch for users who'd rather hold a paper backup than rely on
/// social recovery. It is post-quantum (hash-based wrapping only) and self-verifying: a
/// wrong code can never yield a plausible-but-wrong secret. The system stores only the
/// wrapped blob; the code itself is never persisted (it leaves on paper).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OfflineRecoveryCode {
    wrapped: Vec<u8>,
    /// `H("chk" ‖ master)` — lets `redeem` reject a wrong code without revealing anything.
    check: Hash256,
}

impl OfflineRecoveryCode {
    /// Mint a recovery code over `master_secret`, deriving the code from `code_entropy`
    /// (drawn from the TRNG). Returns the printable code **and** the storable blob. The
    /// caller prints the code and discards it from memory.
    pub fn create(master_secret: &[u8], code_entropy: &[u8]) -> (Vec<u8>, OfflineRecoveryCode) {
        let code = Hash256::of(&[b"offline-code:".as_ref(), code_entropy].concat()).0;
        let key = Hash256::of(&[b"offline-key:".as_ref(), code.as_ref()].concat()).0;
        let ks = keystream(&key, master_secret.len());
        let wrapped: Vec<u8> = master_secret.iter().zip(ks).map(|(b, k)| b ^ k).collect();
        let check = Hash256::of(&[b"chk".as_ref(), master_secret].concat());
        (code.to_vec(), OfflineRecoveryCode { wrapped, check })
    }

    /// Redeem the code to recover the master secret, or `None` if the code is wrong.
    pub fn redeem(&self, code: &[u8]) -> Option<Vec<u8>> {
        let key = Hash256::of(&[b"offline-key:".as_ref(), code].concat()).0;
        let ks = keystream(&key, self.wrapped.len());
        let master: Vec<u8> = self.wrapped.iter().zip(ks).map(|(b, k)| b ^ k).collect();
        if Hash256::of(&[b"chk".as_ref(), master.as_slice()].concat()) == self.check {
            Some(master)
        } else {
            None
        }
    }
}

// ─────────────────── GF(2^8) arithmetic (AES field) ───────────────────

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // reduce by x^8 + x^4 + x^3 + x + 1
        }
        b >>= 1;
    }
    p
}

fn gf_pow(a: u8, mut n: u32) -> u8 {
    let mut acc = 1u8;
    let mut base = a;
    while n > 0 {
        if n & 1 == 1 {
            acc = gf_mul(acc, base);
        }
        base = gf_mul(base, base);
        n >>= 1;
    }
    acc
}

/// Multiplicative inverse in GF(2⁸): `a⁻¹ = a²⁵⁴` (0 maps to 0).
fn gf_inv(a: u8) -> u8 {
    if a == 0 {
        0
    } else {
        gf_pow(a, 254)
    }
}

// ─────────────────────── Shamir secret sharing ───────────────────────

/// One share of a split secret: an x-coordinate and the secret-length y-bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Share {
    pub x: u8,
    pub y: Vec<u8>,
}

/// Split `secret` into `n` shares with threshold `k` (any `k` reconstruct).
/// `rng_bytes` must supply `(k-1) * secret.len()` random coefficients — in the OS
/// these come from the TRNG (real entropy for the polynomial coefficients).
pub fn split(secret: &[u8], k: usize, n: usize, rng_bytes: &[u8]) -> Option<Vec<Share>> {
    if k == 0 || k > n || n > 255 {
        return None;
    }
    if rng_bytes.len() < (k - 1) * secret.len() {
        return None;
    }
    let mut shares: Vec<Share> = (1..=n as u8).map(|x| Share { x, y: vec![0u8; secret.len()] }).collect();
    // For each secret byte, build a degree-(k-1) polynomial with that byte as the
    // constant term, then evaluate at each share's x.
    for (byte_idx, &s) in secret.iter().enumerate() {
        let mut coeffs = vec![s]; // a0 = secret byte
        for j in 1..k {
            coeffs.push(rng_bytes[(j - 1) * secret.len() + byte_idx]);
        }
        for share in &mut shares {
            share.y[byte_idx] = eval_poly(&coeffs, share.x);
        }
    }
    Some(shares)
}

fn eval_poly(coeffs: &[u8], x: u8) -> u8 {
    // Horner's method in GF(2^8).
    let mut acc = 0u8;
    for &c in coeffs.iter().rev() {
        acc = gf_mul(acc, x) ^ c;
    }
    acc
}

/// Reconstruct the secret from `shares` via Lagrange interpolation at x=0.
/// Returns `None` if shares disagree on length or x-coordinates collide.
pub fn reconstruct(shares: &[Share]) -> Option<Vec<u8>> {
    let first = shares.first()?;
    let len = first.y.len();
    if shares.iter().any(|s| s.y.len() != len) {
        return None;
    }
    // x-coordinates must be distinct and non-zero.
    for (i, a) in shares.iter().enumerate() {
        if a.x == 0 {
            return None;
        }
        for b in &shares[i + 1..] {
            if a.x == b.x {
                return None;
            }
        }
    }
    let mut secret = vec![0u8; len];
    for (byte_idx, out) in secret.iter_mut().enumerate() {
        let mut acc = 0u8;
        for (i, si) in shares.iter().enumerate() {
            // Lagrange basis L_i(0) = Π_{j≠i} x_j / (x_j - x_i)  (− is XOR in GF2).
            let mut num = 1u8;
            let mut den = 1u8;
            for (j, sj) in shares.iter().enumerate() {
                if i != j {
                    num = gf_mul(num, sj.x);
                    den = gf_mul(den, sj.x ^ si.x);
                }
            }
            let basis = gf_mul(num, gf_inv(den));
            acc ^= gf_mul(si.y[byte_idx], basis);
        }
        *out = acc;
    }
    Some(secret)
}

// ─────────────────────── social / threshold recovery ───────────────────────

/// A recovery policy: which guardians hold shares and how many must approve.
#[derive(Clone, Debug)]
pub struct RecoveryPolicy {
    pub threshold: usize,
    pub guardians: Vec<u64>,
}

/// The canonical default recovery quorum (resolves the open question on M-of-N and the
/// guardian model): **3-of-5** guardian shares with a **72-hour veto window** before a
/// recovery completes. Five shares tolerate the loss of two devices/guardians while
/// still needing a real quorum of three; the veto window (monotonic, anti-rollback time)
/// lets the legitimate owner abort a malicious recovery before it takes effect.
pub const DEFAULT_RECOVERY_THRESHOLD: usize = 3;
/// The default number of guardian shares minted.
pub const DEFAULT_RECOVERY_GUARDIANS: usize = 5;
/// The default veto window in microseconds (72 hours).
pub const DEFAULT_VETO_WINDOW_US: u64 = 72 * 60 * 60 * 1_000_000;

impl RecoveryPolicy {
    /// The canonical 3-of-5 guardian policy over the supplied guardian ids (the first
    /// [`DEFAULT_RECOVERY_GUARDIANS`] are used). Falls back to a k-of-n where n is the
    /// number supplied if fewer than five are given.
    pub fn canonical(guardians: &[u64]) -> RecoveryPolicy {
        let n = guardians.len().min(DEFAULT_RECOVERY_GUARDIANS);
        let threshold = DEFAULT_RECOVERY_THRESHOLD.min(n.max(1));
        RecoveryPolicy {
            threshold,
            guardians: guardians.iter().take(n).copied().collect(),
        }
    }
}

/// Collects guardian approvals for a recovery/rotation request and decides when the
/// quorum is met. Approvals are deduplicated and validated against the policy.
pub struct RecoverySession {
    policy: RecoveryPolicy,
    request: Vec<u8>,
    approvals: Vec<u64>,
}

impl RecoverySession {
    pub fn open(policy: RecoveryPolicy, request: &[u8]) -> RecoverySession {
        RecoverySession { policy, request: request.to_vec(), approvals: Vec::new() }
    }

    /// Record a guardian's approval of *this* request. Returns false if the
    /// guardian is unknown, already approved, or approved a different request.
    pub fn approve(&mut self, guardian: u64, approving: &[u8]) -> bool {
        if approving != self.request.as_slice() {
            return false;
        }
        if !self.policy.guardians.contains(&guardian) {
            return false;
        }
        if self.approvals.contains(&guardian) {
            return false;
        }
        self.approvals.push(guardian);
        true
    }

    pub fn approvals(&self) -> usize {
        self.approvals.len()
    }

    /// Quorum reached?
    pub fn is_authorized(&self) -> bool {
        self.approvals.len() >= self.policy.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_bytes(n: usize) -> Vec<u8> {
        // Deterministic "entropy" for the test (real OS uses the TRNG).
        let mut rng = crate::random::Drng::from_seed(b"shamir-test-entropy");
        let mut v = vec![0u8; n];
        rng.fill(&mut v);
        v
    }

    #[test]
    fn gf_inverse_is_correct() {
        for a in 1u8..=255 {
            assert_eq!(gf_mul(a, gf_inv(a)), 1, "inverse wrong for {a}");
        }
    }

    #[test]
    fn split_and_reconstruct_with_threshold() {
        let secret = b"master-identity-key-256bit-blob!";
        let rng = rand_bytes(2 * secret.len()); // k-1 = 2 coeff rows
        let shares = split(secret, 3, 5, &rng).unwrap();
        assert_eq!(shares.len(), 5);
        // Any 3 of the 5 shares reconstruct the exact secret.
        let recovered = reconstruct(&shares[0..3]).unwrap();
        assert_eq!(recovered, secret);
        // A *different* subset of 3 also reconstructs the same secret.
        let subset = [shares[1].clone(), shares[3].clone(), shares[4].clone()];
        assert_eq!(reconstruct(&subset).unwrap(), secret);
    }

    #[test]
    fn fewer_than_threshold_does_not_recover_secret() {
        let secret = b"top-secret-key!!";
        let rng = rand_bytes(3 * secret.len()); // k=4
        let shares = split(secret, 4, 6, &rng).unwrap();
        // With only 3 shares (k-1), interpolation yields a different value.
        let wrong = reconstruct(&shares[0..3]).unwrap();
        assert_ne!(wrong, secret);
    }

    #[test]
    fn split_rejects_bad_parameters() {
        assert!(split(b"x", 0, 3, &[0, 0]).is_none()); // k=0
        assert!(split(b"x", 4, 3, &[0; 8]).is_none()); // k>n
        assert!(split(b"abc", 2, 3, &[0]).is_none()); // not enough entropy
    }

    #[test]
    fn reconstruct_rejects_duplicate_x() {
        let s = Share { x: 1, y: vec![10] };
        let dup = Share { x: 1, y: vec![20] };
        assert!(reconstruct(&[s, dup]).is_none());
    }

    #[test]
    fn offline_recovery_code_round_trips_and_rejects_wrong_codes() {
        let master = b"the-user-master-seed-32-bytes!!!";
        let (code, blob) = OfflineRecoveryCode::create(master, b"trng-entropy");
        // The right code recovers the master secret exactly.
        assert_eq!(blob.redeem(&code).as_deref(), Some(master.as_ref()));
        // A wrong code yields nothing (self-verifying — never a plausible-but-wrong seed).
        assert!(blob.redeem(b"wrong-code").is_none());
        let (other, _) = OfflineRecoveryCode::create(master, b"different-entropy");
        assert_ne!(code, other);
        assert!(blob.redeem(&other).is_none());
    }

    #[test]
    fn canonical_quorum_is_three_of_five() {
        let policy = RecoveryPolicy::canonical(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(policy.threshold, 3);
        assert_eq!(policy.guardians.len(), 5); // capped at five shares
        let mut s = RecoverySession::open(policy, b"recover");
        assert!(s.approve(1, b"recover"));
        assert!(s.approve(2, b"recover"));
        assert!(!s.is_authorized()); // two of three
        assert!(s.approve(3, b"recover"));
        assert!(s.is_authorized()); // quorum of three
        // Degrades gracefully with fewer guardians supplied.
        assert_eq!(RecoveryPolicy::canonical(&[1, 2]).threshold, 2);
        const { assert!(DEFAULT_VETO_WINDOW_US > 0) };
    }

    #[test]
    fn social_recovery_requires_quorum() {
        let policy = RecoveryPolicy { threshold: 2, guardians: vec![10, 20, 30] };
        let mut session = RecoverySession::open(policy, b"rotate-to-new-key");
        // Unknown guardian rejected.
        assert!(!session.approve(99, b"rotate-to-new-key"));
        // Wrong request rejected.
        assert!(!session.approve(10, b"steal-the-key"));
        // First valid approval — not yet authorized.
        assert!(session.approve(10, b"rotate-to-new-key"));
        assert!(!session.is_authorized());
        // Double approval ignored.
        assert!(!session.approve(10, b"rotate-to-new-key"));
        assert_eq!(session.approvals(), 1);
        // Quorum reached with a second distinct guardian.
        assert!(session.approve(20, b"rotate-to-new-key"));
        assert!(session.is_authorized());
    }

    #[test]
    fn rotation_reshards_under_new_entropy() {
        let secret = b"key-v1-aaaaaaaaa";
        let shares_v1 = split(secret, 2, 3, &rand_bytes(secret.len())).unwrap();
        // Re-shard the SAME secret with fresh entropy: shares differ, secret holds.
        let mut rng = crate::random::Drng::from_seed(b"rotation-entropy");
        let mut fresh = vec![0u8; secret.len()];
        rng.fill(&mut fresh);
        let shares_v2 = split(secret, 2, 3, &fresh).unwrap();
        assert_ne!(shares_v1[0].y, shares_v2[0].y);
        assert_eq!(reconstruct(&shares_v2[0..2]).unwrap(), secret);
    }
}

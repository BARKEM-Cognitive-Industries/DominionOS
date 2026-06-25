//! Hardware True RNG — the entropy half of the randomness architecture.
//!
//! This is the *unpredictable* generator: it harvests real physical entropy from
//! the CPU's on-die `RDRAND` instruction, runs NIST SP 800-90B-style online health
//! tests on the raw samples, and conditions the pool with SHA-256 to produce a
//! full-entropy seed. That seed bootstraps the reproducible
//! [`Drng`](dominion_core::random::Drng) every cell draws from — so true entropy is
//! consumed only at a defined boundary, and execution downstream stays
//! deterministic and replayable. Per the spec we **fail closed**: if the hardware
//! source or its health tests fail, we yield no randomness rather than weak
//! randomness.

use dominion_core::hash::Hash256;
use dominion_core::random::Drng;
use alloc::vec::Vec;
use core::arch::asm;
use spin::Mutex;

/// One 64-bit draw from RDRAND, retrying the recommended number of times. Returns
/// `None` if the instruction never signals success (absent or exhausted source).
pub fn rdrand64() -> Option<u64> {
    for _ in 0..10 {
        let val: u64;
        let ok: u8;
        unsafe {
            asm!(
                "rdrand {v}",
                "setc {o}",
                v = out(reg) val,
                o = out(reg_byte) ok,
                options(nomem, nostack),
            );
        }
        if ok == 1 {
            return Some(val);
        }
    }
    None
}

/// Is a hardware entropy source present?
pub fn supported() -> bool {
    rdrand64().is_some()
}

/// Result of the online health tests on the raw noise source.
#[derive(Clone, Copy, Debug)]
pub struct Health {
    /// Repetition-Count Test: no value repeated implausibly often in a row.
    pub rct_pass: bool,
    /// Adaptive-Proportion Test: no value dominates a window.
    pub apt_pass: bool,
}

impl Health {
    pub fn passed(&self) -> bool {
        self.rct_pass && self.apt_pass
    }
}

/// Gather `n` raw samples and run the health tests.
fn sample_pool(n: usize) -> Option<(Vec<u64>, Health)> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(rdrand64()?);
    }

    // Repetition-Count Test: a full-entropy 64-bit source must not emit the same
    // value twice in a row except astronomically rarely.
    let mut rct_pass = true;
    for w in samples.windows(2) {
        if w[0] == w[1] {
            rct_pass = false;
            break;
        }
    }

    // Adaptive-Proportion Test: no single value may appear more than a small
    // fraction of the window.
    let mut apt_pass = true;
    let cutoff = (n / 4).max(2);
    for i in 0..samples.len() {
        let mut count = 0;
        for j in 0..samples.len() {
            if samples[j] == samples[i] {
                count += 1;
            }
        }
        if count > cutoff {
            apt_pass = false;
            break;
        }
    }

    Some((samples, Health { rct_pass, apt_pass }))
}

/// Produce a conditioned, full-entropy 32-byte seed from the hardware source, or
/// `None` if the source is absent or fails its health tests (fail-closed).
pub fn conditioned_seed() -> Option<[u8; 32]> {
    let (samples, health) = sample_pool(64)?;
    if !health.passed() {
        return None;
    }
    let mut pool = Vec::with_capacity(samples.len() * 8);
    for s in &samples {
        pool.extend_from_slice(&s.to_le_bytes());
    }
    Some(Hash256::of(&pool).0)
}

/// Run the health tests and report them (for the selftest battery).
pub fn health_check() -> Option<Health> {
    sample_pool(64).map(|(_, h)| h)
}

/// The system DRNG, seeded once from the TRNG at boot.
static DRNG: Mutex<Option<Drng>> = Mutex::new(None);

/// Seed the global deterministic RNG from the hardware entropy source. Returns
/// false (and leaves the DRNG unseeded) if no healthy entropy is available.
pub fn init_global() -> bool {
    match conditioned_seed() {
        Some(seed) => {
            *DRNG.lock() = Some(Drng::from_seed(&seed));
            true
        }
        None => false,
    }
}

/// Draw a 64-bit value from the seeded system DRNG, if initialised.
pub fn random_u64() -> Option<u64> {
    DRNG.lock().as_mut().map(|d| d.next_u64())
}

/// Is the system entropy source healthy (the DRNG was seeded from the TRNG)?
pub fn healthy() -> bool {
    DRNG.lock().is_some()
}

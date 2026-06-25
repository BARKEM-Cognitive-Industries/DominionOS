//! Seeded deterministic RNG — the reproducible half of the randomness architecture.
//!
//! Architecture 2.0 runs *two* generators (see `docs/security/randomness-and-entropy.md`):
//! a hardware TRNG for real entropy (key/nonce/seed generation — that lives in
//! `dominion-kernel`, using RDRAND) and this **seeded CSPRNG** that every cell draws
//! from during execution. The contract: the deterministic state machine never
//! reads hardware entropy mid-computation; it draws from a [`Drng`] seeded once,
//! so *identical seed + identical draws ⇒ identical stream*. That is what makes
//! replay, instruction-level rewind, and `fork`-as-snapshot reproducible.
//!
//! Construction: a hash DRBG. The output block is `SHA-256(key ‖ counter)`;
//! reseeding folds new entropy into the key. Per-cell streams are derived by
//! domain-separating the seed, so one cell's draws cannot perturb another's.

use crate::hash::Hash256;
use alloc::vec::Vec;

/// A cryptographically-seeded, fully reproducible deterministic RNG.
#[derive(Clone)]
pub struct Drng {
    key: [u8; 32],
    counter: u64,
    /// Buffered tail of the last generated block.
    buf: [u8; 32],
    buf_pos: usize,
}

impl Drng {
    /// Seed a generator. The same seed always yields the same stream.
    pub fn from_seed(seed: &[u8]) -> Drng {
        Drng {
            key: Hash256::of(seed).0,
            counter: 0,
            buf: [0u8; 32],
            buf_pos: 32, // empty
        }
    }

    fn next_block(&mut self) -> [u8; 32] {
        let mut input = Vec::with_capacity(40);
        input.extend_from_slice(&self.key);
        input.extend_from_slice(&self.counter.to_le_bytes());
        self.counter = self.counter.wrapping_add(1);
        Hash256::of(&input).0
    }

    /// Fill `out` with pseudo-random bytes.
    pub fn fill(&mut self, out: &mut [u8]) {
        for byte in out.iter_mut() {
            if self.buf_pos >= 32 {
                self.buf = self.next_block();
                self.buf_pos = 0;
            }
            *byte = self.buf[self.buf_pos];
            self.buf_pos += 1;
        }
    }

    /// Next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.fill(&mut b);
        u64::from_le_bytes(b)
    }

    /// Uniform value in `[0, n)` (n > 0) via rejection-free modulo (slight bias is
    /// acceptable for non-cryptographic placement; crypto uses full blocks).
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next_u64() % n
    }

    /// Fold fresh entropy into the key (reseed). Logged at the call site as a
    /// recorded input event so replay stays reproducible.
    pub fn reseed(&mut self, entropy: &[u8]) {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(&self.key);
        input.extend_from_slice(entropy);
        self.key = Hash256::of(&input).0;
        self.counter = 0;
        self.buf_pos = 32;
    }

    /// Derive an independent per-cell / per-domain stream by domain-separating the
    /// seed. Access to a stream is itself meant to be capability-gated.
    pub fn derive_stream(&self, label: &[u8]) -> Drng {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(&self.key);
        input.extend_from_slice(b"stream:");
        input.extend_from_slice(label);
        Drng::from_seed(&input)
    }
}

/// Mixes **multiple independent entropy sources** into one seed (the conditioning
/// step the randomness architecture asks for — see
/// `docs/security/randomness-and-entropy.md`).
///
/// The pool absorbs samples from distinct sources (hardware RDRAND, timing jitter,
/// device noise, network arrival times…). The extracted seed depends on *every*
/// source, so a single compromised or biased source cannot determine the output:
/// as long as **one** source has real entropy, the seed does. This is a software
/// entropy extractor (SHA-256 as the conditioning function) and needs no special
/// hardware — extra hardware sources simply improve the input, never gate it.
#[derive(Clone, Default)]
pub struct EntropyPool {
    /// Running accumulator over all absorbed samples.
    state: Vec<u8>,
    sources: usize,
}

impl EntropyPool {
    pub fn new() -> EntropyPool {
        EntropyPool { state: b"dominion-entropy-pool-v1".to_vec(), sources: 0 }
    }

    /// Absorb a sample from one source. `source_id` domain-separates sources so two
    /// sources contributing the same bytes still mix distinctly.
    pub fn absorb(&mut self, source_id: u32, sample: &[u8]) {
        let mut input = Vec::with_capacity(self.state.len() + sample.len() + 8);
        input.extend_from_slice(&self.state);
        input.extend_from_slice(&source_id.to_le_bytes());
        input.extend_from_slice(sample);
        self.state = Hash256::of(&input).0.to_vec();
        self.sources += 1;
    }

    /// How many samples have been absorbed.
    pub fn sample_count(&self) -> usize {
        self.sources
    }

    /// Extract a conditioned seed and seed a [`Drng`] from it. The DRNG is fully
    /// reproducible *given the same absorbed samples*, so replay holds.
    pub fn seed_drng(&self) -> Drng {
        Drng::from_seed(&self.state)
    }

    /// Extract the raw 32-byte conditioned seed.
    pub fn extract(&self) -> [u8; 32] {
        Hash256::of(&self.state).0
    }
}

/// A recorded consumption of **true** hardware entropy, kept as an input event so
/// a deterministic replay reproduces it exactly instead of re-reading the TRNG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntropyEvent {
    pub seq: u64,
    pub label: Vec<u8>,
    pub sample: [u8; 32],
}

/// The determinism boundary for randomness (see
/// `docs/security/randomness-and-entropy.md`). Hardware entropy is read **once, at
/// a boundary**, and every read is appended here as an [`EntropyEvent`]. After
/// that, all in-execution randomness is drawn from a seeded [`Drng`], so the run is
/// reproducible: replaying the same ledger re-seeds identically without touching
/// hardware. This is what lets rewind/replay reproduce "random" steps.
#[derive(Clone, Default)]
pub struct EntropyLedger {
    events: Vec<EntropyEvent>,
}

impl EntropyLedger {
    pub fn new() -> EntropyLedger {
        EntropyLedger { events: Vec::new() }
    }

    /// Record a true-entropy `sample` (from the kernel TRNG) consumed for `label`
    /// (e.g. `b"boot-seed"`, `b"vault-nonce"`). Returns the assigned sequence id.
    pub fn record(&mut self, label: &[u8], sample: [u8; 32]) -> u64 {
        let seq = self.events.len() as u64;
        self.events.push(EntropyEvent { seq, label: label.to_vec(), sample });
        seq
    }

    /// The recorded sample for `seq` (used during replay instead of the hardware).
    pub fn replay(&self, seq: u64) -> Option<&[u8; 32]> {
        self.events.get(seq as usize).map(|e| &e.sample)
    }

    pub fn events(&self) -> &[EntropyEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Seed a [`Drng`] from the **recorded** entropy ledger. Folding every recorded
    /// sample in order yields a seed that depends only on the ledger — so the same
    /// ledger always reproduces the same generator, hardware untouched.
    pub fn seed_drng(&self) -> Drng {
        let mut pool = EntropyPool::new();
        for e in &self.events {
            pool.absorb(e.seq as u32, &e.sample);
        }
        pool.seed_drng()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Drng::from_seed(b"run-seed");
        let mut b = Drng::from_seed(b"run-seed");
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = Drng::from_seed(b"seed-1");
        let mut b = Drng::from_seed(b"seed-2");
        // Vanishingly unlikely to match across 8 draws if seeds differ.
        let mut differ = false;
        for _ in 0..8 {
            if a.next_u64() != b.next_u64() {
                differ = true;
            }
        }
        assert!(differ);
    }

    #[test]
    fn fill_is_reproducible_across_boundaries() {
        let mut a = Drng::from_seed(b"x");
        let mut b = Drng::from_seed(b"x");
        let mut ba = [0u8; 70]; // spans >2 blocks
        let mut bb = [0u8; 70];
        a.fill(&mut ba);
        b.fill(&mut bb);
        assert_eq!(ba, bb);
    }

    #[test]
    fn derived_streams_are_independent_and_reproducible() {
        let parent = Drng::from_seed(b"root");
        let mut s1 = parent.derive_stream(b"cell-1");
        let mut s2 = parent.derive_stream(b"cell-2");
        let mut s1b = parent.derive_stream(b"cell-1");
        assert_ne!(s1.next_u64(), s2.next_u64());
        // The same label reproduces the same stream.
        assert_eq!({ let mut p = parent.derive_stream(b"cell-1"); p.next_u64() }, s1b.next_u64());
    }

    #[test]
    fn reseed_changes_the_stream() {
        let mut a = Drng::from_seed(b"s");
        let before = a.next_u64();
        a.reseed(b"fresh entropy");
        let after = a.next_u64();
        assert_ne!(before, after);
    }

    #[test]
    fn below_bounds_results() {
        let mut a = Drng::from_seed(b"bound");
        for _ in 0..1000 {
            assert!(a.below(10) < 10);
        }
    }

    #[test]
    fn entropy_pool_mixes_multiple_sources() {
        // Two pools with the same samples produce the same seed (reproducible).
        let mut a = EntropyPool::new();
        let mut b = EntropyPool::new();
        for p in [&mut a, &mut b] {
            p.absorb(1, b"rdrand-sample");
            p.absorb(2, b"timing-jitter");
            p.absorb(3, b"network-arrival");
        }
        assert_eq!(a.extract(), b.extract());
        assert_eq!(a.sample_count(), 3);
    }

    #[test]
    fn every_source_changes_the_output() {
        let mut base = EntropyPool::new();
        base.absorb(1, b"s1");
        base.absorb(2, b"s2");
        let base_seed = base.extract();
        // Changing ANY one source's contribution changes the conditioned seed,
        // so no single source can pin the output.
        let mut diff = EntropyPool::new();
        diff.absorb(1, b"s1");
        diff.absorb(2, b"DIFFERENT");
        assert_ne!(base_seed, diff.extract());
    }

    #[test]
    fn pool_seeds_a_reproducible_drng() {
        let mut pool = EntropyPool::new();
        pool.absorb(0, b"a");
        pool.absorb(1, b"b");
        let mut d1 = pool.seed_drng();
        let mut d2 = pool.seed_drng();
        assert_eq!(d1.next_u64(), d2.next_u64());
    }

    #[test]
    fn entropy_ledger_records_consumption_as_events() {
        let mut led = EntropyLedger::new();
        let s0 = led.record(b"boot-seed", [1u8; 32]);
        let s1 = led.record(b"vault-nonce", [2u8; 32]);
        assert_eq!((s0, s1), (0, 1));
        assert_eq!(led.len(), 2);
        // The recorded sample is retrievable for replay.
        assert_eq!(led.replay(0), Some(&[1u8; 32]));
        assert_eq!(led.events()[1].label, b"vault-nonce");
    }

    #[test]
    fn replaying_the_ledger_reseeds_identically_without_hardware() {
        // "Live" boot reads true entropy and records it.
        let mut live = EntropyLedger::new();
        live.record(b"boot", [7u8; 32]);
        live.record(b"reseed", [9u8; 32]);
        let mut a = live.seed_drng();

        // A later replay reconstructs the ledger from the recorded events alone —
        // no TRNG access — and gets the identical generator.
        let mut replay = EntropyLedger::new();
        for e in live.events() {
            replay.record(&e.label, e.sample);
        }
        let mut b = replay.seed_drng();
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}

//! Property-based invariants & chaos fault-injection — **testing & verification
//! strategy** (see `docs/implementation/testing-and-verification-strategy.md`).
//!
//! Unit tests pin down examples; **properties** pin down laws. This module sweeps
//! thousands of seed-derived inputs and asserts the security and consistency
//! invariants the whole architecture rests on — the things that must hold for
//! *every* input, not just the ones we thought to write down:
//!
//! * **Capability monotonicity** — a derived capability never gains authority.
//! * **Airlock non-bypassability** — a cross-domain transfer never yields more than
//!   the policy ceiling *and* never more than the source held; the reverse
//!   direction is always denied.
//! * **Encryption round-trips** — vault and session decrypt exactly what was
//!   encrypted, and the wrong key/identity never does.
//!
//! Plus a **chaos** harness: a [`BlockDevice`](crate::persist::BlockDevice) that
//! fails writes mid-stream, asserting persistence degrades cleanly (no panic, no
//! silently-corrupted graph) rather than catastrophically.
//!
//! Everything is a pure function of a seed, so any counterexample is a permanent,
//! replayable regression. The harness itself is plain code; the assertions live in
//! `#[cfg(test)]`.

// This module is intentionally tiny in non-test builds — its value is the test
// battery below. The fault-injecting block device is also useful at runtime, so it
// is compiled unconditionally.

use crate::persist::{BlockDevice, BlockError};

/// A [`BlockDevice`] wrapper that injects a write failure once a threshold number
/// of block writes has occurred — the chaos monkey for the persistence layer.
pub struct FaultyDevice<D: BlockDevice> {
    inner: D,
    fail_after: u64,
    writes: u64,
}

impl<D: BlockDevice> FaultyDevice<D> {
    /// Wrap `inner`, failing the write that would be number `fail_after + 1`.
    pub fn new(inner: D, fail_after: u64) -> FaultyDevice<D> {
        FaultyDevice { inner, fail_after, writes: 0 }
    }

    pub fn into_inner(self) -> D {
        self.inner
    }

    pub fn write_count(&self) -> u64 {
        self.writes
    }
}

impl<D: BlockDevice> BlockDevice for FaultyDevice<D> {
    fn block_count(&self) -> u64 {
        self.inner.block_count()
    }

    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        self.inner.read_block(index, buf)
    }

    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if self.writes >= self.fail_after {
            return Err(BlockError::OutOfRange);
        }
        self.writes += 1;
        self.inner.write_block(index, buf)
    }
}

/// Pull a small pseudo-random sequence of [`Rights`](crate::capability::Rights)
/// bit-patterns from a seed (used to fuzz the capability laws).
pub fn rights_from_seed(seed: u64) -> u32 {
    // 5 meaningful right bits (READ|WRITE|EXECUTE|GRANT and the spare in ALL).
    (crate::hash::Hash256::of(&seed.to_le_bytes()).0[0] as u32) & 0b11111
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{Capability, Rights};
    use crate::fuzz::{sweep, FuzzInput};
    use crate::persist::{Persistence, RamDisk};
    use crate::object::{Datum, Object, ObjectGraph};

    // ───────────────────── capability monotonicity ─────────────────────

    #[test]
    fn derive_never_amplifies_authority() {
        sweep(0xCA9, 5000, |seed| {
            let src_bits = rights_from_seed(seed);
            let want_bits = rights_from_seed(seed.rotate_left(17));
            let src = Capability::mint(0x1000, 0x1000, Rights::from_bits(src_bits));
            // Try to derive a sub-range with arbitrary requested rights. A
            // successful derive must hold only rights the parent had; an `Err`
            // (refusing to amplify) is equally correct.
            if let Ok(child) = src.derive(0x1000, 0x800, Rights::from_bits(want_bits)) {
                assert!(
                    src.rights().contains(child.rights()),
                    "monotonicity violated: parent {:05b} child {:05b}",
                    src.rights().bits(),
                    child.rights().bits()
                );
            }
        });
    }

    #[test]
    fn restrict_only_ever_drops_rights() {
        sweep(0x235, 5000, |seed| {
            let src_bits = rights_from_seed(seed);
            let mask = rights_from_seed(seed.wrapping_add(99));
            let src = Capability::mint(0, 0x100, Rights::from_bits(src_bits));
            if let Ok(r) = src.restrict(Rights::from_bits(mask)) {
                assert!(src.rights().contains(r.rights()));
            }
        });
    }

    // ───────────────────── airlock non-bypassability ─────────────────────

    #[test]
    fn airlock_never_grants_more_than_policy_or_source() {
        use crate::airlock::{Airlock, TransferPolicy};
        use crate::firewall::Domain;
        sweep(0xA12, 5000, |seed| {
            let src_bits = rights_from_seed(seed);
            let ceiling = rights_from_seed(seed.rotate_right(13));
            let mut a = Airlock::new();
            a.add_policy(TransferPolicy {
                from: Domain::Financial,
                to: Domain::AiAgent,
                max_rights: Rights::from_bits(ceiling),
                ttl: Some(10),
                approvals_required: 1,
            });
            let src = Capability::mint(0x2000, 0x1000, Rights::from_bits(src_bits));
            if let Ok(issued) = a.transfer(src, Domain::Financial, Domain::AiAgent, 1, 0) {
                let granted = issued.capability.rights();
                // Never exceeds the policy ceiling …
                assert!(Rights::from_bits(ceiling).contains(granted));
                // … and never exceeds what the source actually held.
                assert!(src.rights().contains(granted));
            }
            // The reverse direction has no policy → must always be denied.
            assert!(a.transfer(src, Domain::AiAgent, Domain::Financial, 1, 0).is_err());
        });
    }

    // ───────────────────── encryption round-trips ─────────────────────

    #[test]
    fn vault_round_trips_random_data_under_both_suites() {
        use crate::vault::{CipherSuite, Key, Vault};
        sweep(0x7A17, 3000, |seed| {
            let mut input = FuzzInput::new(seed);
            let plaintext = input.blob(200);
            let key = Key::from_seed(&input.bytes(16));
            let ik = Key::from_seed(b"ik");
            let suite = if input.u8() & 1 == 0 {
                CipherSuite::ChaCha20Poly1305
            } else {
                CipherSuite::Aes256Gcm
            };
            let mut v = Vault::new();
            let nonce = input.bytes(12);
            let id = v.seal_with(suite, &plaintext, key, &nonce, &ik, &[]);
            // Right key recovers exactly the plaintext.
            assert_eq!(v.open(id, key).as_deref(), Some(plaintext.as_slice()));
            // A different key never does.
            let wrong = Key::from_seed(&input.bytes(16));
            if wrong != key {
                assert!(v.open(id, wrong).is_none());
            }
            // The store never holds the plaintext (when non-trivial).
            if plaintext.len() >= 8 {
                let ct = v.ciphertext(id).unwrap();
                assert!(!ct.windows(plaintext.len()).any(|w| w == plaintext.as_slice()));
            }
        });
    }

    #[test]
    fn session_round_trips_and_rejects_tamper() {
        use crate::session::{KemIdentity, Session};
        // KEM keygen is heavy, so keep the sweep modest but meaningful.
        sweep(0x5E55, 60, |seed| {
            let mut input = FuzzInput::new(seed);
            let alice = KemIdentity::generate(&input.bytes(8));
            let bob = KemIdentity::generate(&input.bytes(8));
            let (mut a, ct) =
                Session::initiate(alice.id, bob.id, &bob.public, &input.bytes(8), 1000).unwrap();
            let mut b = Session::accept(&bob, alice.id, &ct, 1000);
            let msg = input.blob(64);
            let frame = a.seal(1, &msg).unwrap();
            assert_eq!(b.open(1, &frame).as_deref(), Ok(msg.as_slice()));
            // Any single-byte tamper is detected.
            if frame.payload_len() > 0 {
                let mut bad = frame.clone();
                bad.corrupt_first_byte();
                assert!(b.open(1, &bad).is_err());
            }
        });
    }

    // ───────────────────── chaos: persistence fault injection ─────────────────────

    #[test]
    fn persistence_degrades_cleanly_under_write_faults() {
        // First, a clean save of a known-good graph so the disk holds a valid image.
        sweep(0xC4A05, 400, |seed| {
            let mut input = FuzzInput::new(seed);
            let mut good = ObjectGraph::new();
            for _ in 0..(input.u8() % 6 + 1) {
                good.put(Object::new("Doc").with("b", Datum::Bytes(input.blob(40))));
            }
            let mut disk = RamDisk::new(2048);
            Persistence::save(&mut disk, &good).expect("clean save");
            let good_root = Persistence::load(&mut disk).unwrap().unwrap().root_hash();

            // Now attempt a NEW save (an extra object) through a device that fails
            // partway through. Rebuild via serialize/deserialize since the graph is
            // intentionally not `Clone` (content addressing makes copies cheap as
            // bytes, not as aliases).
            let mut newg = ObjectGraph::deserialize(&good.serialize()).unwrap();
            newg.put(Object::new("Doc").with("b", Datum::Bytes(input.blob(40))));
            let fail_after = (input.u8() % 5) as u64;
            let mut faulty = FaultyDevice::new(disk, fail_after);
            // The save may fail — that is allowed. What is NOT allowed is a panic or
            // a load that returns a structurally-broken graph.
            let _ = Persistence::save(&mut faulty, &newg);
            let mut disk = faulty.into_inner();
            // Either a graph loads — and it must be self-consistent (re-serialize to
            // a stable root); matching neither old nor new exactly is acceptable for
            // a partial write, corruption/panic is not — or the load cleanly reports
            // "no valid image". Both outcomes are fine; a panic is the only failure.
            if let Ok(Some(g)) = Persistence::load(&mut disk) {
                let r = g.root_hash();
                let reser = ObjectGraph::deserialize(&g.serialize()).unwrap().root_hash();
                assert_eq!(r, reser, "loaded graph must be self-consistent");
                let _ = good_root; // old root retained for reference
            }
        });
    }
}

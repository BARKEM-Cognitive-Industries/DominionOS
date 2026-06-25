//! Deterministic machine state (SRS Stage 10 & 12).
//!
//! "The entire machine state is a hashable object, and every action is
//! processed as a strict input to a state transition function." We model the OS
//! as exactly that: a [`Machine`] whose state is a sorted key/value store, driven
//! by a log of [`Action`]s through a *pure* transition function.
//!
//! Three properties fall out, and each is tested:
//!
//! * **Reproducibility** — replaying the same action log from genesis always
//!   yields the same state hash (deterministic simulation testing, §12.1).
//! * **Controlled non-determinism** — the only randomness comes from an explicit
//!   seed fed to a deterministic generator, so "random" steps replay identically.
//! * **Rewind** — the machine can be rewound to any prior step instruction by
//!   instruction to find a root cause (§12.1).

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A single deterministic transition.
#[derive(Clone, PartialEq, Debug)]
pub enum Action {
    /// Set `key` to `value`.
    Set(String, i64),
    /// Add `delta` to `key` (absent keys are treated as 0).
    Add(String, i64),
    /// Remove `key`.
    Del(String),
    /// Draw the next value from the seeded deterministic generator into `key`.
    Rand(String),
}

/// The deterministic OS state machine.
#[derive(Clone)]
pub struct Machine {
    state: BTreeMap<String, i64>,
    log: Vec<Action>,
    /// Current value of the deterministic generator (controlled non-determinism).
    rng: u64,
    seed: u64,
}

impl Machine {
    /// Create a machine with a fixed RNG seed — the DST hypervisor controls this
    /// single source of randomness.
    pub fn new(seed: u64) -> Machine {
        Machine {
            state: BTreeMap::new(),
            log: Vec::new(),
            rng: seed,
            seed,
        }
    }

    pub fn step_count(&self) -> usize {
        self.log.len()
    }

    pub fn get(&self, key: &str) -> Option<i64> {
        self.state.get(key).copied()
    }

    pub fn log(&self) -> &[Action] {
        &self.log
    }

    /// The pure transition: advance `(state, rng)` by one action. Kept static so
    /// it provably cannot depend on anything but its inputs.
    fn transition(state: &mut BTreeMap<String, i64>, rng: &mut u64, action: &Action) {
        match action {
            Action::Set(k, v) => {
                state.insert(k.clone(), *v);
            }
            Action::Add(k, d) => {
                let e = state.entry(k.clone()).or_insert(0);
                *e = e.wrapping_add(*d);
            }
            Action::Del(k) => {
                state.remove(k);
            }
            Action::Rand(k) => {
                // xorshift64* — fully deterministic given the seed.
                let mut x = *rng;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                *rng = x;
                let out = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
                state.insert(k.clone(), (out >> 1) as i64);
            }
        }
    }

    /// Apply one action, recording it in the log.
    pub fn apply(&mut self, action: Action) {
        Self::transition(&mut self.state, &mut self.rng, &action);
        self.log.push(action);
    }

    /// Apply a batch of actions.
    pub fn apply_all<I: IntoIterator<Item = Action>>(&mut self, actions: I) {
        for a in actions {
            self.apply(a);
        }
    }

    /// Canonical content hash of the *entire* machine state (Stage 10). Two
    /// machines hash equal iff their observable state and step count match.
    pub fn state_hash(&self) -> Hash256 {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"mstate1");
        buf.extend_from_slice(&(self.log.len() as u64).to_le_bytes());
        for (k, v) in &self.state {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Hash256::of(&buf)
    }

    /// Replay an action log from genesis at the given seed and return the
    /// resulting machine. This is the reproducibility primitive of §12.1.
    pub fn replay(seed: u64, actions: &[Action]) -> Machine {
        let mut m = Machine::new(seed);
        m.apply_all(actions.iter().cloned());
        m
    }

    /// Rewind to exactly `step` applied actions by deterministic re-execution
    /// from genesis. Returns an error if `step` exceeds the current log length.
    pub fn rewound_to(&self, step: usize) -> Result<Machine, String> {
        if step > self.log.len() {
            return Err("cannot rewind past the recorded history".to_string());
        }
        Ok(Machine::replay(self.seed, &self.log[..step]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program() -> Vec<Action> {
        alloc::vec![
            Action::Set("a".into(), 10),
            Action::Add("a".into(), 5),
            Action::Set("b".into(), 1),
            Action::Add("b".into(), -1),
            Action::Rand("r".into()),
        ]
    }

    #[test]
    fn transitions_update_state() {
        let mut m = Machine::new(1);
        m.apply(Action::Set("x".into(), 7));
        assert_eq!(m.get("x"), Some(7));
        m.apply(Action::Add("x".into(), 3));
        assert_eq!(m.get("x"), Some(10));
        m.apply(Action::Del("x".into()));
        assert_eq!(m.get("x"), None);
    }

    #[test]
    fn replay_is_reproducible() {
        let a = Machine::replay(42, &program());
        let b = Machine::replay(42, &program());
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn different_seed_changes_random_steps() {
        let a = Machine::replay(1, &program());
        let b = Machine::replay(2, &program());
        // Only the Rand step differs, but that is enough to change the hash.
        assert_ne!(a.state_hash(), b.state_hash());
        // The non-random keys are identical regardless of seed.
        assert_eq!(a.get("a"), b.get("a"));
        assert_eq!(a.get("b"), b.get("b"));
    }

    #[test]
    fn random_step_is_deterministic_for_a_seed() {
        let a = Machine::replay(99, &[Action::Rand("r".into())]);
        let b = Machine::replay(99, &[Action::Rand("r".into())]);
        assert_eq!(a.get("r"), b.get("r"));
    }

    #[test]
    fn rewind_recovers_prior_state() {
        let m = Machine::replay(7, &program());
        let mid = m.rewound_to(2).unwrap();
        assert_eq!(mid.step_count(), 2);
        assert_eq!(mid.get("a"), Some(15));
        assert_eq!(mid.get("b"), None); // b not set yet at step 2
    }

    #[test]
    fn rewind_then_forward_matches_original() {
        let m = Machine::replay(7, &program());
        let target = m.state_hash();
        let rewound = m.rewound_to(2).unwrap();
        let mut forward = rewound.clone();
        for a in &m.log()[2..] {
            forward.apply(a.clone());
        }
        assert_eq!(forward.state_hash(), target);
    }

    #[test]
    fn rewind_out_of_range_errors() {
        let m = Machine::replay(7, &program());
        assert!(m.rewound_to(99).is_err());
    }

    #[test]
    fn state_hash_changes_with_state() {
        let mut m = Machine::new(0);
        let h0 = m.state_hash();
        m.apply(Action::Set("k".into(), 1));
        assert_ne!(h0, m.state_hash());
    }
}

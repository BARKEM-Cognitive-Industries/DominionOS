//! Continuous runtime attestation — **Stage 11.5 / 11.8**.
//!
//! The system continuously verifies that what is running is what *should* be
//! running: the kernel state hash, each cell's executable hash, and the capability
//! graph's integrity. A measurement is a deterministic hash chain over named
//! components; any unexpected mutation changes the quote and trips containment
//! (revocation, quarantine, checkpoint capture). Because execution is
//! deterministic (Stage 10), a detected incident is exactly replayable.
//!
//! Pure, safe, host-tested.

use crate::hash::Hash256;
use alloc::vec::Vec;

/// Compute a measurement quote: an order-sensitive hash chain over named
/// components `(name, bytes)`. Folding the name in binds each measurement to its
/// slot, so swapping two components changes the quote.
pub fn measure(components: &[(&str, &[u8])]) -> Hash256 {
    let mut acc = Hash256::ZERO;
    for (name, bytes) in components {
        let mut block = Vec::with_capacity(name.len() + bytes.len());
        block.extend_from_slice(name.as_bytes());
        block.push(0);
        block.extend_from_slice(bytes);
        acc = acc.combine(&Hash256::of(&block));
    }
    acc
}

/// Holds a trusted baseline quote and checks live measurements against it.
pub struct Attestor {
    baseline: Hash256,
}

impl Attestor {
    /// Trust a known-good quote.
    pub fn new(baseline: Hash256) -> Attestor {
        Attestor { baseline }
    }

    /// Capture the current component set as the trusted baseline.
    pub fn from_components(components: &[(&str, &[u8])]) -> Attestor {
        Attestor { baseline: measure(components) }
    }

    pub fn baseline(&self) -> Hash256 {
        self.baseline
    }

    /// Verify a freshly computed quote against the baseline.
    pub fn verify(&self, current: Hash256) -> bool {
        current == self.baseline
    }

    /// Re-measure the live components and report whether the system is unmodified.
    pub fn attest(&self, components: &[(&str, &[u8])]) -> bool {
        self.verify(measure(components))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn components(cell: &[u8]) -> [(&str, &[u8]); 3] {
        [
            ("kernel", b"kernel-state-v1".as_ref()),
            ("cell:shell", cell),
            ("cap-graph", b"graph-root-aaaa".as_ref()),
        ]
    }

    #[test]
    fn measure_is_deterministic() {
        assert_eq!(measure(&components(b"shell-v1")), measure(&components(b"shell-v1")));
    }

    #[test]
    fn attestation_passes_for_unmodified_system() {
        let att = Attestor::from_components(&components(b"shell-v1"));
        assert!(att.attest(&components(b"shell-v1")));
    }

    #[test]
    fn tampered_cell_is_detected() {
        let att = Attestor::from_components(&components(b"shell-v1"));
        // The shell cell's executable changed (e.g. code injection).
        assert!(!att.attest(&components(b"shell-v1-trojaned")));
    }

    #[test]
    fn order_sensitivity_binds_components_to_slots() {
        let a = measure(&[("a", b"1"), ("b", b"2")]);
        let b = measure(&[("b", b"2"), ("a", b"1")]);
        assert_ne!(a, b);
    }
}

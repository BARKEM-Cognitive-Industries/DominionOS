//! Hybrid Logical Clocks (HLC) — **secure & verifiable time** (causal half) and
//! the **consistency model**'s ordering primitive (see `docs/security/secure-time.md`
//! and the Stage 4 consistency extension).
//!
//! A multikernel is a *network of cores* with no shared clock; a fleet is a network
//! of devices. Ordering events across them needs a clock that (a) tracks **causality**
//! (if A caused B then `hlc(A) < hlc(B)`) and (b) stays close to physical time. A
//! Hybrid Logical Clock does both: it is a `(wall, logical)` pair that advances with
//! the physical clock but bumps a logical counter to preserve happens-before even
//! when physical clocks are coarse or skewed.
//!
//! The standard HLC update rules are implemented here. Pure, safe, host-tested.

/// Errors that can arise from HLC operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HlcError {
    /// The logical counter has reached `u32::MAX` and cannot be incremented without
    /// wrapping — which would violate monotonicity. The caller should either advance
    /// the physical clock or reject the event.
    CounterExhausted,
}

/// A hybrid logical timestamp: a physical-time component plus a logical tie-breaker.
/// Ordered lexicographically, so it is a total order consistent with causality.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Timestamp {
    pub wall: u64,
    pub logical: u32,
}

/// A per-core / per-device hybrid logical clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hlc {
    last: Timestamp,
}

impl Hlc {
    pub fn new() -> Hlc {
        Hlc { last: Timestamp::default() }
    }

    /// The last timestamp this clock issued.
    pub fn now(&self) -> Timestamp {
        self.last
    }

    /// Stamp a **local** event, given the current physical-clock reading `pt`.
    /// Advances to physical time when it moves forward; otherwise bumps the logical
    /// counter so two local events never collide.
    ///
    /// Returns `Err(HlcError::CounterExhausted)` if the logical counter would wrap
    /// (overflow), which would violate monotonicity.
    pub fn local(&mut self, pt: u64) -> Result<Timestamp, HlcError> {
        let prev = self.last;
        let wall = prev.wall.max(pt);
        let logical = if wall == prev.wall {
            prev.logical.checked_add(1).ok_or(HlcError::CounterExhausted)?
        } else {
            0
        };
        self.last = Timestamp { wall, logical };
        Ok(self.last)
    }

    /// Stamp the **receipt** of a message carrying timestamp `msg`, given local
    /// physical time `pt`. The merge rule guarantees the result strictly dominates
    /// both the previous local time and the message's time — so causality holds
    /// across cores/devices.
    ///
    /// Returns `Err(HlcError::CounterExhausted)` if the logical counter would wrap
    /// (overflow), which would violate monotonicity.
    pub fn receive(&mut self, pt: u64, msg: Timestamp) -> Result<Timestamp, HlcError> {
        let prev = self.last;
        let wall = prev.wall.max(msg.wall).max(pt);
        let logical = if wall == prev.wall && wall == msg.wall {
            prev.logical.max(msg.logical).checked_add(1).ok_or(HlcError::CounterExhausted)?
        } else if wall == prev.wall {
            prev.logical.checked_add(1).ok_or(HlcError::CounterExhausted)?
        } else if wall == msg.wall {
            msg.logical.checked_add(1).ok_or(HlcError::CounterExhausted)?
        } else {
            0
        };
        self.last = Timestamp { wall, logical };
        Ok(self.last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_events_are_strictly_increasing() {
        let mut c = Hlc::new();
        let a = c.local(10).unwrap();
        let b = c.local(10).unwrap(); // same physical reading → logical bumps
        let d = c.local(12).unwrap(); // physical advances → logical resets
        assert!(a < b);
        assert!(b < d);
        assert_eq!(b.logical, 1);
        assert_eq!(d.logical, 0);
    }

    #[test]
    fn receive_dominates_both_clocks() {
        // Core A stamps a send; core B (behind in physical time) receives it.
        let mut a = Hlc::new();
        let mut b = Hlc::new();
        let _ = b.local(100).unwrap(); // B is ahead physically
        let sent = a.local(5).unwrap(); // A is behind
        let recvd = b.receive(50, sent).unwrap();
        // The receive timestamp strictly dominates both the message and B's prior.
        assert!(recvd > sent);
        assert!(recvd > b_prior(100));
    }

    fn b_prior(wall: u64) -> Timestamp {
        Timestamp { wall, logical: 0 }
    }

    #[test]
    fn causality_holds_across_a_chain_of_messages() {
        // A → B → C: the final event must causally dominate the first.
        let (mut a, mut b, mut c) = (Hlc::new(), Hlc::new(), Hlc::new());
        let e1 = a.local(1).unwrap();
        let e2 = b.receive(1, e1).unwrap();
        let e3 = c.receive(1, e2).unwrap();
        assert!(e1 < e2 && e2 < e3);
    }

    #[test]
    fn clock_never_runs_backwards_under_skew() {
        // A message from the future must not let a later local event regress.
        let mut c = Hlc::new();
        let future = Timestamp { wall: 1_000, logical: 0 };
        let r = c.receive(10, future).unwrap();
        let nxt = c.local(20).unwrap(); // physical 20 ≪ 1000, but HLC must not go back
        assert!(nxt > r);
        assert!(nxt.wall >= 1_000);
    }

    #[test]
    fn counter_exhaustion_returns_err_not_wrap() {
        let mut c = Hlc::new();
        // Manually set last to a timestamp with logical at u32::MAX.
        c.last = Timestamp { wall: 1, logical: u32::MAX };
        // A local event at the same wall time must not wrap — it must error.
        assert_eq!(c.local(1).err(), Some(HlcError::CounterExhausted));
        // A receive with the same wall time also must not wrap.
        let msg = Timestamp { wall: 1, logical: 0 };
        assert_eq!(c.receive(1, msg).err(), Some(HlcError::CounterExhausted));
    }
}

//! Secure & verifiable time (see `docs/security/secure-time.md`).
//!
//! Time is a security primitive: capability TTLs, attestation freshness, airlock
//! expiry and the update lifecycle all depend on a clock an attacker cannot roll
//! back or forge. The OS therefore never trusts a single wall clock. Three
//! mechanisms here:
//!
//! * A **monotonic logical clock** that can be merged with observed timestamps
//!   (Lamport semantics) and is guaranteed never to move backwards.
//! * **Attested timestamps**: a time authority binds `(time, nonce)` with a keyed
//!   MAC, so a holder can prove *when* and prove the response is **fresh** (the
//!   nonce defeats replay).
//! * **Roughtime-style agreement**: collect attestations from several independent
//!   authorities, discard any that fail verification or echo the wrong nonce, and
//!   take the median — a single lying or compromised source cannot move time.
//!
//! Pure, safe `no_std`; no special hardware (a hardware secure clock, where
//! present, simply becomes one more high-trust source).

use crate::hash::Hash256;
use alloc::vec::Vec;

/// Microseconds since an arbitrary epoch (monotone, unsigned).
pub type Micros = u64;

/// A logical clock that is monotone non-decreasing. `observe` folds in a timestamp
/// seen from elsewhere (Lamport merge) without ever regressing local time.
#[derive(Clone, Debug, Default)]
pub struct MonotonicClock {
    now: Micros,
}

impl MonotonicClock {
    pub fn new(start: Micros) -> MonotonicClock {
        MonotonicClock { now: start }
    }

    /// Advance by at least `delta` (a tick from a hardware counter).
    pub fn advance(&mut self, delta: Micros) -> Micros {
        self.now = self.now.saturating_add(delta.max(1));
        self.now
    }

    pub fn now(&self) -> Micros {
        self.now
    }

    /// Merge an observed timestamp: jump forward to it if it is ahead, but never
    /// move backwards (a rollback attempt is ignored, and reported via the bool).
    pub fn observe(&mut self, observed: Micros) -> bool {
        if observed > self.now {
            self.now = observed;
            true
        } else {
            false // observed time was not ahead (possibly a rollback attempt)
        }
    }
}

/// Keyed MAC over a message (`H(key ‖ "mac" ‖ msg)` — a PRF for this model).
fn mac(key: &[u8], msg: &[u8]) -> Hash256 {
    let mut input = Vec::with_capacity(key.len() + msg.len() + 4);
    input.extend_from_slice(key);
    input.extend_from_slice(b"mac:");
    input.extend_from_slice(msg);
    Hash256::of(&input)
}

/// A timestamp attested by a named authority, bound to a freshness nonce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedTimestamp {
    pub authority: u64,
    pub time: Micros,
    pub nonce: u64,
    pub tag: Hash256,
}

/// A time authority holding a secret MAC key.
pub struct TimeAuthority {
    id: u64,
    key: Vec<u8>,
    /// The authority's own (possibly skewed) clock.
    clock: MonotonicClock,
}

impl TimeAuthority {
    pub fn new(id: u64, key: &[u8], start: Micros) -> TimeAuthority {
        TimeAuthority { id, key: key.to_vec(), clock: MonotonicClock::new(start) }
    }

    pub fn advance(&mut self, delta: Micros) {
        self.clock.advance(delta);
    }

    fn payload(id: u64, time: Micros, nonce: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&id.to_le_bytes());
        v.extend_from_slice(&time.to_le_bytes());
        v.extend_from_slice(&nonce.to_le_bytes());
        v
    }

    /// Answer a time request carrying `nonce`. The response binds the current time
    /// and the nonce under the authority's key.
    pub fn attest(&self, nonce: u64) -> SignedTimestamp {
        let time = self.clock.now();
        let tag = mac(&self.key, &Self::payload(self.id, time, nonce));
        SignedTimestamp { authority: self.id, time, nonce, tag }
    }
}

/// Verify an attestation against a known authority key and the nonce the client
/// actually sent (defeats replay: a stale response carries the wrong nonce).
pub fn verify_timestamp(key: &[u8], expected_nonce: u64, ts: &SignedTimestamp) -> bool {
    if ts.nonce != expected_nonce {
        return false;
    }
    let expected = mac(key, &TimeAuthority::payload(ts.authority, ts.time, ts.nonce));
    expected == ts.tag
}

/// The outcome of agreeing time across several authorities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeAgreement {
    /// The agreed time (median of verified sources).
    pub time: Micros,
    /// How many sources verified and contributed.
    pub quorum: usize,
    /// Spread between the earliest and latest verified source (skew bound).
    pub spread: Micros,
}

/// Roughtime-style agreement. Each `(key, attestation)` is verified against
/// `nonce`; failures are dropped. Returns `None` if fewer than `min_quorum`
/// sources verified. The median is robust to a minority of liars.
pub fn agree_time(
    sources: &[(&[u8], SignedTimestamp)],
    nonce: u64,
    min_quorum: usize,
) -> Option<TimeAgreement> {
    let mut times: Vec<Micros> = sources
        .iter()
        .filter(|(key, ts)| verify_timestamp(key, nonce, ts))
        .map(|(_, ts)| ts.time)
        .collect();
    if times.len() < min_quorum || times.is_empty() {
        return None;
    }
    times.sort_unstable();
    let median = times[times.len() / 2];
    let spread = times[times.len() - 1] - times[0];
    Some(TimeAgreement { time: median, quorum: times.len(), spread })
}

// ───────────────────────── explicit time-kind rules ─────────────────────────

/// The three distinct notions of time, kept **explicitly separate** so code never
/// confuses them (the spec's rule). Each has a defined use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeKind {
    /// Causal ordering across cores/devices — use the HLC / logical clock, never wall time.
    Logical,
    /// Human-facing display + external correlation — wall-clock (attested, may jump).
    WallClock,
    /// Expiry / freshness / anti-rollback — a monotonic counter that cannot run backward.
    AntiRollback,
}

impl TimeKind {
    /// The correct time kind for a purpose — the explicit rule set the spec asks for.
    pub fn for_purpose(purpose: TimePurpose) -> TimeKind {
        match purpose {
            TimePurpose::Ordering => TimeKind::Logical,
            TimePurpose::Display | TimePurpose::ExternalProof => TimeKind::WallClock,
            TimePurpose::Expiry | TimePurpose::Freshness => TimeKind::AntiRollback,
        }
    }
}

/// What time is being used for (drives [`TimeKind::for_purpose`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimePurpose {
    Ordering,
    Display,
    ExternalProof,
    Expiry,
    Freshness,
}

/// Roughtime-style agreement with **explicit outlier rejection**: verify every source,
/// take a provisional median, **discard sources more than `max_deviation` from it**, and
/// re-aggregate the survivors. A minority of liars (or a badly-skewed clock) is dropped,
/// not just out-voted. Returns `None` if fewer than `min_quorum` survive.
pub fn agree_time_robust(
    sources: &[(&[u8], SignedTimestamp)],
    nonce: u64,
    min_quorum: usize,
    max_deviation: Micros,
) -> Option<TimeAgreement> {
    let mut times: Vec<Micros> = sources
        .iter()
        .filter(|(key, ts)| verify_timestamp(key, nonce, ts))
        .map(|(_, ts)| ts.time)
        .collect();
    if times.is_empty() {
        return None;
    }
    times.sort_unstable();
    let provisional = times[times.len() / 2];
    // Drop outliers beyond max_deviation from the provisional median.
    let mut survivors: Vec<Micros> = times
        .into_iter()
        .filter(|t| t.abs_diff(provisional) <= max_deviation)
        .collect();
    if survivors.len() < min_quorum {
        return None;
    }
    survivors.sort_unstable();
    let median = survivors[survivors.len() / 2];
    let spread = survivors[survivors.len() - 1] - survivors[0];
    Some(TimeAgreement { time: median, quorum: survivors.len(), spread })
}

// ───────────────────────── uncertainty intervals + commit-wait ─────────────────────────

/// A `Time` value as the spec wants it: a timestamp **plus the uncertainty** about it.
/// True time is guaranteed to lie within `[ts − err, ts + err]` (a TrueTime-style
/// interval). Cross-node ordering reasons about the interval, never the point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UncertainTime {
    /// The best point estimate.
    pub ts: Micros,
    /// The ± uncertainty (half-width of the interval).
    pub err: Micros,
}

impl UncertainTime {
    pub fn new(ts: Micros, err: Micros) -> UncertainTime {
        UncertainTime { ts, err }
    }

    /// The earliest true time could be.
    pub fn earliest(&self) -> Micros {
        self.ts.saturating_sub(self.err)
    }

    /// The latest true time could be.
    pub fn latest(&self) -> Micros {
        self.ts.saturating_add(self.err)
    }

    /// True iff this interval is wholly before `other` — i.e. they do **not** overlap and
    /// our latest is before their earliest. Only then is ordering certain.
    pub fn definitely_before(&self, other: &UncertainTime) -> bool {
        self.latest() < other.earliest()
    }

    /// **Commit-wait** (Spanner-style): to guarantee a timestamp `ts` taken *now* has
    /// surely passed on every node before we externalize it, wait out the uncertainty.
    /// Returns the absolute time to wait until (`ts + err`), so any other node's clock —
    /// itself within ±err — has certainly moved past `ts`.
    pub fn commit_wait_until(&self) -> Micros {
        self.latest()
    }
}

/// A recorded external-time read (Roughtime/NTS response), so time, like entropy, enters
/// deterministic execution as an **input event** that DST replay reproduces without the
/// network — mirroring [`crate::random::EntropyLedger`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeEvent {
    pub seq: u64,
    pub source: u64,
    pub reading: UncertainTime,
}

/// An append-only log of external-time reads.
#[derive(Clone, Debug, Default)]
pub struct TimeLedger {
    events: Vec<TimeEvent>,
}

impl TimeLedger {
    pub fn new() -> TimeLedger {
        TimeLedger { events: Vec::new() }
    }

    /// Record an external time reading from `source`; returns its sequence id.
    pub fn record(&mut self, source: u64, reading: UncertainTime) -> u64 {
        let seq = self.events.len() as u64;
        self.events.push(TimeEvent { seq, source, reading });
        seq
    }

    /// Replay a recorded reading by sequence (used instead of the network on replay).
    pub fn replay(&self, seq: u64) -> Option<UncertainTime> {
        self.events.get(seq as usize).map(|e| e.reading)
    }

    pub fn events(&self) -> &[TimeEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_kind_rules_are_explicit() {
        assert_eq!(TimeKind::for_purpose(TimePurpose::Ordering), TimeKind::Logical);
        assert_eq!(TimeKind::for_purpose(TimePurpose::Display), TimeKind::WallClock);
        assert_eq!(TimeKind::for_purpose(TimePurpose::Expiry), TimeKind::AntiRollback);
        assert_eq!(TimeKind::for_purpose(TimePurpose::Freshness), TimeKind::AntiRollback);
    }

    #[test]
    fn robust_agreement_rejects_an_outlier_source() {
        let k1 = b"src1";
        let k2 = b"src2";
        let k3 = b"src3";
        let k4 = b"liar";
        let mk = |key: &[u8], t: Micros| {
            let a = TimeAuthority::new(0, key, t);
            a.attest(7)
        };
        // Three honest sources near 1000, one wild liar at 9_000_000.
        let sources: [(&[u8], SignedTimestamp); 4] = [
            (k1, mk(k1, 1000)),
            (k2, mk(k2, 1010)),
            (k3, mk(k3, 990)),
            (k4, mk(k4, 9_000_000)),
        ];
        let agreed = agree_time_robust(&sources, 7, 3, 5000).unwrap();
        // The liar is discarded; the median is among the honest cluster.
        assert!(agreed.time >= 990 && agreed.time <= 1010);
        assert_eq!(agreed.quorum, 3);
    }

    #[test]
    fn uncertainty_intervals_and_commit_wait() {
        let a = UncertainTime::new(1000, 50); // [950, 1050]
        let b = UncertainTime::new(1200, 30); // [1170, 1230]
        // Non-overlapping ⇒ order is certain.
        assert!(a.definitely_before(&b));
        // Overlapping intervals are not definitely ordered.
        let c = UncertainTime::new(1040, 50); // [990, 1090] overlaps a
        assert!(!a.definitely_before(&c));
        // Commit-wait waits out the uncertainty: wait until ts+err.
        assert_eq!(a.commit_wait_until(), 1050);
    }

    #[test]
    fn time_ledger_replays_external_reads_deterministically() {
        let mut led = TimeLedger::new();
        led.record(1, UncertainTime::new(5000, 10));
        led.record(2, UncertainTime::new(5001, 12));
        assert_eq!(led.len(), 2);
        assert_eq!(led.replay(0), Some(UncertainTime::new(5000, 10)));
        assert_eq!(led.replay(1).unwrap().ts, 5001);
        assert!(led.replay(9).is_none());
    }

    #[test]
    fn monotonic_clock_never_regresses() {
        let mut c = MonotonicClock::new(1000);
        assert_eq!(c.advance(50), 1050);
        // Observing an earlier time is rejected and does not move the clock.
        assert!(!c.observe(500));
        assert_eq!(c.now(), 1050);
        // Observing a later time jumps forward.
        assert!(c.observe(2000));
        assert_eq!(c.now(), 2000);
        // advance always moves by at least 1 (no stalling).
        assert_eq!(c.advance(0), 2001);
    }

    #[test]
    fn attestation_verifies_and_binds_nonce() {
        let auth = TimeAuthority::new(1, b"authority-key", 5_000);
        let ts = auth.attest(0xABCD);
        assert!(verify_timestamp(b"authority-key", 0xABCD, &ts));
        // Wrong nonce (replay) → rejected.
        assert!(!verify_timestamp(b"authority-key", 0x0000, &ts));
        // Wrong key → rejected.
        assert!(!verify_timestamp(b"wrong-key", 0xABCD, &ts));
    }

    #[test]
    fn tampered_time_fails_verification() {
        let auth = TimeAuthority::new(7, b"k", 100);
        let mut ts = auth.attest(1);
        ts.time += 10_000; // forge a later time
        assert!(!verify_timestamp(b"k", 1, &ts));
    }

    #[test]
    fn agreement_takes_median_and_ignores_liar() {
        let nonce = 42;
        let a = TimeAuthority::new(1, b"ka", 1_000).attest(nonce);
        let b = TimeAuthority::new(2, b"kb", 1_010).attest(nonce);
        let c = TimeAuthority::new(3, b"kc", 1_020).attest(nonce);
        // A liar with a wildly wrong time but a *valid* signature is still bounded
        // by the median (it is just one of four; median ignores the extreme).
        let liar = TimeAuthority::new(4, b"kd", 9_000_000).attest(nonce);
        let sources: [(&[u8], SignedTimestamp); 4] = [
            (b"ka", a),
            (b"kb", b),
            (b"kc", c),
            (b"kd", liar),
        ];
        let agreed = agree_time(&sources, nonce, 3).unwrap();
        assert_eq!(agreed.quorum, 4);
        // Median is one of the honest cluster, not the liar's 9_000_000.
        assert!(agreed.time >= 1_000 && agreed.time <= 1_020);
    }

    #[test]
    fn agreement_requires_quorum_of_verified_sources() {
        let nonce = 5;
        let good = TimeAuthority::new(1, b"k1", 100).attest(nonce);
        // A forged source with the wrong key fails verification.
        let mut forged = TimeAuthority::new(2, b"real", 200).attest(nonce);
        forged.tag = Hash256::of(b"garbage");
        let sources: [(&[u8], SignedTimestamp); 2] =
            [(b"k1", good), (b"real", forged)];
        // Only one source verifies; quorum of 2 not met.
        assert!(agree_time(&sources, nonce, 2).is_none());
        // Quorum of 1 succeeds with the single honest source.
        assert_eq!(agree_time(&sources, nonce, 1).unwrap().time, 100);
    }
}

//! Local threat detection — **on-device anomaly analysis with no cloud dependency**
//! (Stage 11.5/11.8; `docs/security/stage-11-kernel-hardening.md`).
//!
//! Architecture 2.0 assumes compromise and refuses to phone home: threat detection must
//! run *locally*, deterministically, over signals the OS already produces — capability
//! authority diffusion, denied cross-domain attempts, escalation faults, and per-domain
//! energy draw. This module is a small, dependency-free **online anomaly detector**: it
//! learns a baseline for each signal from the live stream (a Welford running mean +
//! variance — no training set, no network), then flags statistically surprising values
//! and, for severe ones, **recommends emergency containment** ([`crate::airlock`]).
//!
//! It is deliberately *not* an "AI advisory that decides" — per design tension T1 it is
//! advisory: it raises [`ThreatSignal`]s, but a verified component (the firewall/airlock)
//! makes the actual allow/deny call. Pure, safe `no_std`; fully deterministic (no RNG, no
//! wall-clock) so every alert reproduces under DST replay.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A signal the monitor watches. Each gets its own learned baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignalKind {
    /// How far authority spreads from a node (`firewall.rs::diffusion`).
    AuthorityDiffusion,
    /// Rate of cross-domain transfers denied by default.
    CrossDomainDenied,
    /// Capability escalation / tag-tamper faults.
    EscalationAttempt,
    /// Per-domain energy draw (an energy anomaly can betray hidden compute).
    EnergyDraw,
}

/// How surprising an observation is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Within normal variation.
    Normal,
    /// Beyond ~2σ — worth noting.
    Elevated,
    /// Beyond ~3σ — worth acting on.
    Suspicious,
    /// Beyond ~4σ — recommend immediate containment.
    Critical,
}

/// An online baseline: running count, mean, and sum of squared deviations (Welford).
/// Stores no history, so it is O(1) memory per signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct Baseline {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Baseline {
    pub fn new() -> Baseline {
        Baseline::default()
    }

    /// Fold a new observation into the running statistics.
    pub fn observe(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (0 until two samples exist).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Classify `x` against the current baseline, without sqrt: compare the squared
    /// deviation to k²·variance for k ∈ {2,3,4}. Returns `Normal` until `warmup` samples
    /// have been seen (so the detector doesn't cry wolf while still learning).
    pub fn classify(&self, x: f64, warmup: u64) -> Severity {
        if self.count < warmup {
            return Severity::Normal;
        }
        let var = self.variance();
        let dev2 = {
            let d = x - self.mean;
            d * d
        };
        // A zero-variance baseline: any departure from the constant mean is suspicious.
        if var <= f64::EPSILON {
            return if dev2 <= f64::EPSILON { Severity::Normal } else { Severity::Critical };
        }
        if dev2 > 16.0 * var {
            Severity::Critical
        } else if dev2 > 9.0 * var {
            Severity::Suspicious
        } else if dev2 > 4.0 * var {
            Severity::Elevated
        } else {
            Severity::Normal
        }
    }
}

/// An alert raised by the monitor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreatSignal {
    pub kind: SignalKind,
    pub severity: Severity,
    /// The observed value that triggered the alert.
    pub value: f64,
    /// The learned mean at the time of the alert (for explainability).
    pub baseline_mean: f64,
}

impl ThreatSignal {
    /// Whether this signal warrants **emergency containment** (advisory — the verified
    /// airlock/firewall acts on it). Only the most severe band recommends it.
    pub fn recommends_containment(&self) -> bool {
        self.severity == Severity::Critical
    }
}

/// The local threat monitor: one learned baseline per signal kind.
pub struct ThreatMonitor {
    baselines: BTreeMap<SignalKind, Baseline>,
    warmup: u64,
}

impl ThreatMonitor {
    /// Build a monitor that warms up over `warmup` samples before flagging anomalies.
    pub fn new(warmup: u64) -> ThreatMonitor {
        ThreatMonitor { baselines: BTreeMap::new(), warmup: warmup.max(2) }
    }

    /// Observe a signal value. Returns a [`ThreatSignal`] iff it is anomalous (severity
    /// above `Normal`). The value is folded into the baseline *after* classification, so
    /// an attack spike doesn't pollute the very baseline it is judged against.
    pub fn observe(&mut self, kind: SignalKind, value: f64) -> Option<ThreatSignal> {
        let base = self.baselines.entry(kind).or_default();
        let severity = base.classify(value, self.warmup);
        let baseline_mean = base.mean();
        base.observe(value);
        if severity == Severity::Normal {
            None
        } else {
            Some(ThreatSignal { kind, severity, value, baseline_mean })
        }
    }

    /// Feed a fleet of normal observations (e.g. to seed a baseline at boot).
    pub fn warm(&mut self, kind: SignalKind, values: &[f64]) {
        let base = self.baselines.entry(kind).or_default();
        for &v in values {
            base.observe(v);
        }
    }

    /// The current baseline for a signal kind.
    pub fn baseline(&self, kind: SignalKind) -> Option<&Baseline> {
        self.baselines.get(&kind)
    }
}

/// Scan a batch of observations and return every anomaly found, in order. A small helper
/// for replaying a recorded window of telemetry deterministically (DST regression).
pub fn scan(monitor: &mut ThreatMonitor, batch: &[(SignalKind, f64)]) -> Vec<ThreatSignal> {
    batch.iter().filter_map(|&(k, v)| monitor.observe(k, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_learns_mean_and_variance() {
        let mut b = Baseline::new();
        for x in [10.0, 12.0, 8.0, 11.0, 9.0] {
            b.observe(x);
        }
        assert_eq!(b.count(), 5);
        assert!((b.mean() - 10.0).abs() < 1e-9);
        assert!(b.variance() > 0.0);
    }

    #[test]
    fn a_diffusion_spike_is_flagged_critical_and_recommends_containment() {
        let mut m = ThreatMonitor::new(8);
        // Normal authority diffusion hovers around ~3.
        m.warm(SignalKind::AuthorityDiffusion, &[3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0, 3.0]);
        // A sudden blast radius of 40 — far outside the baseline.
        let sig = m.observe(SignalKind::AuthorityDiffusion, 40.0).unwrap();
        assert_eq!(sig.severity, Severity::Critical);
        assert!(sig.recommends_containment());
    }

    #[test]
    fn normal_values_do_not_alert() {
        let mut m = ThreatMonitor::new(8);
        m.warm(SignalKind::EnergyDraw, &[100.0, 105.0, 98.0, 102.0, 99.0, 101.0, 100.0, 103.0]);
        // A value within normal variation produces no alert.
        assert!(m.observe(SignalKind::EnergyDraw, 104.0).is_none());
    }

    #[test]
    fn detector_stays_quiet_during_warmup() {
        let mut m = ThreatMonitor::new(10);
        // Even a wild value is not flagged before the baseline has warmed up.
        assert!(m.observe(SignalKind::EscalationAttempt, 999.0).is_none());
    }

    #[test]
    fn energy_anomaly_becomes_a_threat_signal() {
        let mut m = ThreatMonitor::new(6);
        m.warm(SignalKind::EnergyDraw, &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        // Constant baseline → any real departure is treated as suspicious.
        let sig = m.observe(SignalKind::EnergyDraw, 80.0).unwrap();
        assert_eq!(sig.kind, SignalKind::EnergyDraw);
        assert!(sig.severity >= Severity::Suspicious);
    }

    #[test]
    fn scan_replays_a_telemetry_window_deterministically() {
        let warm: Vec<(SignalKind, f64)> =
            (0..8).map(|_| (SignalKind::CrossDomainDenied, 1.0)).collect();
        let mut a = ThreatMonitor::new(8);
        let mut b = ThreatMonitor::new(8);
        let mut batch = warm.clone();
        batch.push((SignalKind::CrossDomainDenied, 50.0)); // a denial storm
        let ra = scan(&mut a, &batch);
        let rb = scan(&mut b, &batch);
        assert_eq!(ra, rb); // deterministic
        assert_eq!(ra.len(), 1);
        assert!(ra[0].recommends_containment());
    }
}

//! **DominionLink overlay transport** — the L2 hard problems QUIC solved, made
//! explicit (Stage 7 / integration strategy §5).
//!
//! DominionLink runs as an overlay over UDP (like QUIC/WireGuard), so it inherits
//! the same four engineering pieces every UDP transport needs. This module
//! implements them as pure, deterministic, host- and on-metal-testable logic:
//!
//! 1. **Congestion control** ([`Cubic`], [`Bbr`]) — a window/rate controller at the
//!    overlay, exactly as QUIC carries CUBIC/BBR in application space.
//! 2. **NAT traversal** ([`IceAgent`]) — STUN/ICE candidate gathering + prioritized
//!    pairing + connectivity checks, with **TURN-like relay fallback** when every
//!    direct path is blocked (most peers are behind NAT).
//! 3. **Mobility / roaming** ([`Connection`] + [`LocatorDirectory`]) — addressing is
//!    *identity-based, not IP-based*, so a device that changes network just
//!    **re-announces its locator**; the live connection survives the IP change via
//!    path validation (QUIC connection-migration model).
//! 4. **Offline-first** ([`OfflineReplica`]) — NDN in-network caching + the device's
//!    local graph replica serve many reads with no connectivity; writes queue and
//!    **reconcile by content hash** on reconnect (same hash = same object, so the
//!    merge is naturally de-duplicated).
//!
//! Nothing here replaces IP at L3 — this is the overlay, exactly like QUIC over UDP.
//! Pure, safe `no_std`; integer/float math only (a local `cbrt`, no `libm`).

use crate::dominionlink::DominionId;
use crate::hash::Hash256;
use crate::net::Ipv4Addr;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A concrete overlay locator: where a peer can currently be reached on the
/// underlay (an `IP:port`). Peers are addressed by [`DominionId`]; a locator is the
/// *mutable* mapping the DHT resolves an identity to.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Locator {
    pub ip: Ipv4Addr,
    pub port: u16,
}

impl Locator {
    pub const fn new(ip: Ipv4Addr, port: u16) -> Locator {
        Locator { ip, port }
    }
}

// ===========================================================================
// 1. Congestion control (BBR / CUBIC)
// ===========================================================================

/// The maximum segment size (bytes) the controllers reason in.
pub const MSS: usize = 1448;

/// Integer-only cube root (Newton's method) — CUBIC needs `K = cbrt(...)` and we
/// have no `libm` in `no_std`.
fn cbrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Initial guess: halve the exponent crudely by scaling.
    let mut g = if x > 1.0 { x / 3.0 + 1.0 } else { x };
    let mut i = 0;
    while i < 40 {
        let g2 = g * g;
        if g2 == 0.0 {
            break;
        }
        let next = g - (g - x / g2) / 3.0;
        if (next - g).abs() < 1e-9 {
            g = next;
            break;
        }
        g = next;
        i += 1;
    }
    g
}

/// **CUBIC** congestion control (RFC 8312 shape): a window-based controller whose
/// growth is a cubic function of the time since the last loss — concave as it
/// approaches the prior peak (`w_max`), convex as it probes beyond it. This is the
/// default Linux/QUIC controller; here it governs the DominionLink overlay window.
#[derive(Clone, Debug)]
pub struct Cubic {
    /// Congestion window, in MSS-sized segments.
    cwnd: f64,
    ssthresh: f64,
    w_max: f64,
    /// Virtual time (sum of acked RTTs) — keeps the cubic deterministic without a clock.
    t_now: f64,
    epoch_start: f64,
    k: f64,
    in_slow_start: bool,
}

const CUBIC_C: f64 = 0.4;
const CUBIC_BETA: f64 = 0.7;

impl Cubic {
    /// A fresh controller with an IW10 initial window.
    pub fn new() -> Cubic {
        Cubic {
            cwnd: 10.0,
            ssthresh: f64::MAX,
            w_max: 0.0,
            t_now: 0.0,
            epoch_start: 0.0,
            k: 0.0,
            in_slow_start: true,
        }
    }

    /// Window in whole segments.
    pub fn cwnd_segments(&self) -> usize {
        if self.cwnd < 1.0 {
            1
        } else {
            self.cwnd as usize
        }
    }

    /// Window in bytes.
    pub fn cwnd_bytes(&self) -> usize {
        self.cwnd_segments() * MSS
    }

    pub fn in_slow_start(&self) -> bool {
        self.in_slow_start
    }

    /// An ack arrived after a round-trip of `rtt` ticks. In slow start the window
    /// grows by one segment per ack; in congestion avoidance it follows the cubic
    /// curve toward (and past) `w_max`.
    pub fn on_ack(&mut self, rtt: f64) {
        self.t_now += rtt.max(0.0);
        if self.in_slow_start {
            self.cwnd += 1.0;
            if self.cwnd >= self.ssthresh {
                self.in_slow_start = false;
                self.epoch_start = self.t_now;
            }
            return;
        }
        let t = self.t_now - self.epoch_start;
        // W_cubic(t) = C*(t - K)^3 + w_max
        let dt = t - self.k;
        let target = CUBIC_C * dt * dt * dt + self.w_max;
        // Move the window a fraction toward the cubic target (per-ack increment).
        if target > self.cwnd {
            self.cwnd += (target - self.cwnd) / self.cwnd.max(1.0);
        } else {
            self.cwnd += 0.01; // gentle probing when at/over target
        }
    }

    /// A loss was detected: multiplicatively decrease (β = 0.7), remember the peak,
    /// recompute `K`, and leave slow start.
    pub fn on_loss(&mut self) {
        self.w_max = self.cwnd;
        self.cwnd *= CUBIC_BETA;
        if self.cwnd < 1.0 {
            self.cwnd = 1.0;
        }
        self.ssthresh = self.cwnd;
        self.in_slow_start = false;
        self.epoch_start = self.t_now;
        // K = cbrt( w_max * (1 - beta) / C )
        self.k = cbrt(self.w_max * (1.0 - CUBIC_BETA) / CUBIC_C);
    }
}

impl Default for Cubic {
    fn default() -> Self {
        Self::new()
    }
}

/// **BBR**-style controller: model the path by its bottleneck bandwidth
/// (`BtlBw`, a max-filter over delivery rate) and round-trip propagation
/// (`RtProp`, a min-filter over RTT). The window targets the bandwidth-delay
/// product, and pacing matches the bottleneck — congestion is read from the path's
/// *measured* limits rather than only from loss.
#[derive(Clone, Debug)]
pub struct Bbr {
    /// Bottleneck bandwidth estimate, in **bytes per tick** (max filter).
    btlbw: f64,
    /// Round-trip propagation estimate, in **ticks** (min filter).
    rtprop: f64,
    cwnd_gain: f64,
    pacing_gain: f64,
    startup: bool,
    prev_btlbw: f64,
    plateau_rounds: u32,
}

impl Bbr {
    pub fn new() -> Bbr {
        Bbr {
            btlbw: 0.0,
            rtprop: f64::MAX,
            cwnd_gain: 2.0,
            pacing_gain: 2.885, // 2/ln(2), the BBR startup gain
            startup: true,
            prev_btlbw: 0.0,
            plateau_rounds: 0,
        }
    }

    /// A delivery sample: `bytes` acknowledged over an `rtt` of ticks. Updates the
    /// bandwidth max-filter and the RTprop min-filter, and exits Startup once
    /// bandwidth plateaus (three rounds without ~25% growth).
    pub fn on_ack(&mut self, bytes: usize, rtt: f64) {
        let rtt = rtt.max(1.0);
        let rate = bytes as f64 / rtt;
        if rate > self.btlbw {
            self.btlbw = rate;
        }
        if rtt < self.rtprop {
            self.rtprop = rtt;
        }
        if self.startup {
            if self.btlbw < self.prev_btlbw * 1.25 {
                self.plateau_rounds += 1;
                if self.plateau_rounds >= 3 {
                    self.startup = false;
                    self.cwnd_gain = 2.0;
                    self.pacing_gain = 1.0;
                }
            } else {
                self.plateau_rounds = 0;
            }
            self.prev_btlbw = self.btlbw;
        } else if self.pacing_gain < 1.0 {
            // Steady state: recover pacing after a transient post-Startup dip
            // (on_loss briefly drops it to 0.75) rather than staying clamped for
            // the rest of the connection and under-utilising the path.
            self.pacing_gain = 1.0;
        }
    }

    /// A loss is informational for BBR (it is bandwidth-led), but we cap the window
    /// at the in-flight estimate to stay polite.
    pub fn on_loss(&mut self) {
        // Bandwidth-led: nudge pacing down briefly rather than halving the window.
        if !self.startup {
            self.pacing_gain = 0.75;
        }
    }

    /// The bandwidth-delay product in bytes.
    pub fn bdp_bytes(&self) -> usize {
        if self.rtprop == f64::MAX {
            return MSS;
        }
        (self.btlbw * self.rtprop) as usize
    }

    /// Target window in bytes (BDP × cwnd gain), at least one segment.
    pub fn cwnd_bytes(&self) -> usize {
        (self.bdp_bytes() as f64 * self.cwnd_gain).max(MSS as f64) as usize
    }

    /// Pacing rate in bytes/tick (BtlBw × pacing gain).
    pub fn pacing_rate(&self) -> f64 {
        self.btlbw * self.pacing_gain
    }

    pub fn in_startup(&self) -> bool {
        self.startup
    }
}

impl Default for Bbr {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// 2. NAT traversal (STUN / ICE + relay fallback)
// ===========================================================================

/// An ICE candidate type, in descending preference: a directly-bound local
/// address, a STUN-discovered public (server-reflexive) address, or a TURN-style
/// relayed address (the always-works fallback).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CandidateType {
    Host,
    ServerReflexive,
    Relay,
}

impl CandidateType {
    /// ICE type preference (higher = more preferred): host > srflx > relay.
    pub fn type_preference(self) -> u32 {
        match self {
            CandidateType::Host => 126,
            CandidateType::ServerReflexive => 100,
            CandidateType::Relay => 0,
        }
    }
}

/// An ICE candidate: a reachable transport address with its provenance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Candidate {
    pub typ: CandidateType,
    pub addr: Locator,
    /// The base (local) address this candidate was derived from.
    pub base: Locator,
}

impl Candidate {
    /// RFC 8445 priority: `2^24 * type_pref + 2^8 * local_pref + (256 - component)`.
    pub fn priority(&self, local_pref: u32, component: u32) -> u32 {
        (1 << 24) * self.typ.type_preference() + (1 << 8) * local_pref + (256 - component.min(256))
    }
}

/// The state of a candidate pair during connectivity checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PairState {
    Frozen,
    Waiting,
    Succeeded,
    Failed,
}

/// A local↔remote candidate pair under test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CandidatePair {
    pub local: Candidate,
    pub remote: Candidate,
    pub priority: u64,
    pub state: PairState,
}

/// An ICE agent: gathers candidates, forms prioritized pairs against a peer, runs
/// connectivity checks, and nominates the best working pair — falling back to a
/// relay when every direct path is blocked by NAT.
#[derive(Default)]
pub struct IceAgent {
    candidates: Vec<Candidate>,
    /// `true` = this agent took the controlling role in the offer/answer.
    controlling: bool,
}

impl IceAgent {
    pub fn new(controlling: bool) -> IceAgent {
        IceAgent { candidates: Vec::new(), controlling }
    }

    /// Add a gathered candidate (host binding, STUN reflexive, or TURN relay).
    pub fn gather(&mut self, typ: CandidateType, addr: Locator, base: Locator) {
        self.candidates.push(Candidate { typ, addr, base });
    }

    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Form candidate pairs against the remote agent's candidates, ordered by the
    /// RFC 8445 pair priority (controlling/controlled formula), best first.
    pub fn form_pairs(&self, remote: &[Candidate]) -> Vec<CandidatePair> {
        let mut pairs = Vec::new();
        for (li, local) in self.candidates.iter().enumerate() {
            for (ri, rem) in remote.iter().enumerate() {
                let lp = local.priority(65535 - li as u32, 1) as u64;
                let rp = rem.priority(65535 - ri as u32, 1) as u64;
                let (g, d) = if self.controlling { (lp, rp) } else { (rp, lp) };
                // pair priority = 2^32 * min(G,D) + 2*max(G,D) + (G>D ? 1 : 0)
                let priority =
                    (1u64 << 32) * g.min(d) + 2 * g.max(d) + if g > d { 1 } else { 0 };
                pairs.push(CandidatePair { local: *local, remote: *rem, priority, state: PairState::Frozen });
            }
        }
        pairs.sort_by_key(|p| core::cmp::Reverse(p.priority));
        pairs
    }

    /// Run connectivity checks over the prioritized pairs and **nominate** the
    /// highest-priority pair that passes. `reachable(local, remote)` models whether
    /// a STUN binding request would succeed on the underlay (direct host paths fail
    /// behind symmetric NAT; the relayed path always works). Returns the nominated
    /// pair with its state set to `Succeeded`, or `None` if even the relay is down.
    pub fn nominate(
        &self,
        remote: &[Candidate],
        reachable: impl Fn(&Candidate, &Candidate) -> bool,
    ) -> Option<CandidatePair> {
        let pairs = self.form_pairs(remote);
        for mut pair in pairs {
            if reachable(&pair.local, &pair.remote) {
                pair.state = PairState::Succeeded;
                return Some(pair);
            }
        }
        None
    }

    /// Whether a nominated pair uses a relay (informational — direct is preferred,
    /// relay is the fallback that always works).
    pub fn is_relayed(pair: &CandidatePair) -> bool {
        pair.local.typ == CandidateType::Relay || pair.remote.typ == CandidateType::Relay
    }
}

// ===========================================================================
// 3. Mobility / roaming (identity-based connection migration)
// ===========================================================================

/// The DHT-resolved mapping from a stable identity to its *current* locator. A
/// roaming device re-announces here; everyone resolves the identity, never an IP.
#[derive(Default)]
pub struct LocatorDirectory {
    map: BTreeMap<DominionId, Locator>,
}

impl LocatorDirectory {
    pub fn new() -> LocatorDirectory {
        LocatorDirectory { map: BTreeMap::new() }
    }

    /// Re-announce (or first-announce) an identity's current locator.
    pub fn announce(&mut self, id: DominionId, locator: Locator) {
        self.map.insert(id, locator);
    }

    /// Resolve an identity to its current locator.
    pub fn resolve(&self, id: &DominionId) -> Option<Locator> {
        self.map.get(id).copied()
    }
}

/// Why a connection migration was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MigrateError {
    /// The path-validation challenge was not echoed correctly — possible off-path
    /// spoof; the connection keeps its old path.
    ValidationFailed,
}

/// A live overlay connection, keyed by the **peer identity** rather than an
/// `IP:port`. When the peer roams to a new network, the connection migrates to the
/// new path *after path validation* without tearing down — the sequence space and
/// in-flight state survive the IP change (the QUIC connection-migration model).
#[derive(Clone, Debug)]
pub struct Connection {
    peer: DominionId,
    path: Locator,
    /// Monotonic application sequence — proves the stream is preserved across migration.
    seq: u64,
    /// A path being validated (challenge issued, awaiting echo).
    pending: Option<(Locator, u64)>,
    /// Per-connection secret mixed into every PATH_CHALLENGE so an off-path
    /// attacker who knows peer_id + seq + new_path cannot forge the challenge.
    /// Derived at construction as `Hash256(random_seed || peer_id || initial_path)`;
    /// the 32-byte random seed is supplied by the caller from the kernel TRNG and
    /// is never transmitted to the peer, so even an on-path attacker who observes
    /// the initial connection setup cannot compute `migration_secret`.
    migration_secret: [u8; 32],
}

impl Connection {
    /// Create a new connection.
    ///
    /// `random_seed` **must** be 32 bytes of cryptographically secure random
    /// material from the kernel TRNG (or equivalent).  It is mixed into
    /// `migration_secret` so that the secret is unguessable even to an on-path
    /// attacker who observes `peer` and `path` during the initial handshake.
    pub fn new(peer: DominionId, path: Locator, random_seed: [u8; 32]) -> Connection {
        // migration_secret = Hash256(random_seed || peer_id || path_ip || path_port)
        // The random_seed is never sent on the wire; it is known only to this endpoint.
        let mut buf = [0u8; 70]; // 32 (seed) + 32 (peer hash) + 4 (IP) + 2 (port)
        buf[..32].copy_from_slice(&random_seed);
        buf[32..64].copy_from_slice(&peer.0 .0);
        buf[64..68].copy_from_slice(&path.ip.0);
        buf[68..70].copy_from_slice(&path.port.to_le_bytes());
        let migration_secret = Hash256::of(&buf).0;
        Connection { peer, path, seq: 0, pending: None, migration_secret }
    }

    pub fn peer(&self) -> DominionId {
        self.peer
    }
    pub fn path(&self) -> Locator {
        self.path
    }
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Advance the application stream (e.g. on each delivered frame).
    pub fn send(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Begin migration to `new_path`: issue a path-validation challenge keyed by
    /// the per-connection `migration_secret` so that only endpoints who hold the
    /// secret can produce or verify the challenge (QUIC PATH_CHALLENGE model, but
    /// with a long-lived local secret seeded from TRNG rather than a per-message
    /// random nonce).  Because `migration_secret` incorporates a TRNG-sourced seed
    /// that was never transmitted, even an on-path attacker who observed the
    /// initial handshake cannot compute the challenge.
    /// Returns the challenge the peer must echo.
    pub fn begin_migration(&mut self, new_path: Locator) -> u64 {
        // buf = migration_secret || peer_id || seq || new_path.ip || new_path.port
        // migration_secret embeds a TRNG seed never sent on the wire, so an
        // attacker — on-path or off-path — who observes peer_id + seq + new_path
        // cannot compute the correct challenge.
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.migration_secret);
        buf.extend_from_slice(&self.peer.0 .0);
        buf.extend_from_slice(&self.seq.to_le_bytes());
        buf.extend_from_slice(&new_path.ip.0);
        buf.extend_from_slice(&new_path.port.to_le_bytes());
        let challenge = u64::from_le_bytes(Hash256::of(&buf).0[..8].try_into().unwrap());
        self.pending = Some((new_path, challenge));
        challenge
    }

    /// The peer echoes the challenge from the new path. On a correct echo the
    /// connection switches to the new path **without** resetting its sequence
    /// space; an incorrect echo (off-path spoof) is refused and the old path stays.
    /// Uses a constant-time comparison to prevent timing-based forgery.
    pub fn complete_migration(&mut self, echoed: u64) -> Result<Locator, MigrateError> {
        match self.pending {
            Some((new_path, challenge)) => {
                // Constant-time compare: XOR all bytes; accept only if the result is zero.
                let expected = challenge.to_le_bytes();
                let received = echoed.to_le_bytes();
                let mut diff = 0u8;
                for i in 0..8 {
                    diff |= expected[i] ^ received[i];
                }
                self.pending = None;
                if diff == 0 {
                    self.path = new_path;
                    Ok(new_path)
                } else {
                    Err(MigrateError::ValidationFailed)
                }
            }
            None => Err(MigrateError::ValidationFailed),
        }
    }
}

// ===========================================================================
// 4. Offline-first (local replica + reconcile by hash)
// ===========================================================================

/// What a reconcile pushed/pulled when connectivity returned.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ReconcileReport {
    /// New objects pushed to the remote (were absent there).
    pub pushed: usize,
    /// Pushes skipped because the remote already had the content hash (dedup).
    pub deduped: usize,
    /// Objects pulled from the remote that were missing locally.
    pub pulled: usize,
}

/// A content-addressed store (the device's local graph replica or a remote peer's).
#[derive(Default)]
pub struct ReplicaStore {
    objects: BTreeMap<Hash256, Vec<u8>>,
}

impl ReplicaStore {
    pub fn new() -> ReplicaStore {
        ReplicaStore { objects: BTreeMap::new() }
    }
    pub fn put(&mut self, data: &[u8]) -> Hash256 {
        let id = Hash256::of(data);
        self.objects.entry(id).or_insert_with(|| data.to_vec());
        id
    }
    pub fn get(&self, id: &Hash256) -> Option<&[u8]> {
        self.objects.get(id).map(|v| v.as_slice())
    }
    pub fn has(&self, id: &Hash256) -> bool {
        self.objects.contains_key(id)
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

/// An **offline-first** replica: reads are served from the local content store with
/// no connectivity; writes are stored locally and **queued** while offline. On
/// reconnect, [`reconcile`](Self::reconcile) pushes queued writes to a peer —
/// de-duplicated by content hash (same hash = same object, so an object the peer
/// already holds is never re-sent) — and pulls anything missing locally.
pub struct OfflineReplica {
    local: ReplicaStore,
    pending: Vec<Hash256>,
    online: bool,
}

impl Default for OfflineReplica {
    fn default() -> Self {
        Self::new()
    }
}

impl OfflineReplica {
    pub fn new() -> OfflineReplica {
        OfflineReplica { local: ReplicaStore::new(), pending: Vec::new(), online: true }
    }

    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }
    pub fn is_online(&self) -> bool {
        self.online
    }

    /// Read by content id — succeeds offline whenever the object is in the local
    /// replica / NDN cache.
    pub fn read(&self, id: &Hash256) -> Option<&[u8]> {
        self.local.get(id)
    }

    /// Write `data`: store it content-addressed locally and (if offline) queue it
    /// for reconciliation. Returns the content id. Writing the same bytes twice is
    /// idempotent (one object, one pending entry).
    pub fn write(&mut self, data: &[u8]) -> Hash256 {
        let id = self.local.put(data);
        if !self.online && !self.pending.contains(&id) {
            self.pending.push(id);
        }
        id
    }

    /// Number of writes still awaiting reconciliation.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Reconnect and reconcile with `remote`: push every queued write the remote
    /// lacks (dedup the rest by hash), then pull any object ids in `want` that the
    /// remote has and we do not. Clears the pending queue and goes online.
    pub fn reconcile(&mut self, remote: &mut ReplicaStore, want: &[Hash256]) -> ReconcileReport {
        let mut report = ReconcileReport::default();
        for id in core::mem::take(&mut self.pending) {
            if remote.has(&id) {
                report.deduped += 1;
            } else if let Some(bytes) = self.local.get(&id) {
                let bytes = bytes.to_vec();
                remote.put(&bytes);
                report.pushed += 1;
            }
        }
        for id in want {
            if !self.local.has(id) {
                if let Some(bytes) = remote.get(id) {
                    let bytes = bytes.to_vec();
                    self.local.put(&bytes);
                    report.pulled += 1;
                }
            }
        }
        self.online = true;
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(d: u8, port: u16) -> Locator {
        Locator::new(Ipv4Addr::new(10, 0, 0, d), port)
    }
    fn id(seed: &[u8]) -> DominionId {
        DominionId(Hash256::of(seed))
    }

    // ---- 1. congestion control: CUBIC ----
    #[test]
    fn cubic_grows_in_slow_start_and_backs_off_on_loss() {
        let mut c = Cubic::new();
        let start = c.cwnd_segments();
        for _ in 0..5 {
            c.on_ack(1.0);
        }
        assert!(c.cwnd_segments() > start); // slow-start growth (≈ +1/ack)
        let before = c.cwnd_segments();
        c.on_loss();
        // Multiplicative decrease to ~70% and leaves slow start.
        assert!(c.cwnd_segments() < before);
        assert!(!c.in_slow_start());
        // Congestion avoidance: the cubic curve probes back upward.
        let dip = c.cwnd_segments();
        for _ in 0..50 {
            c.on_ack(1.0);
        }
        assert!(c.cwnd_segments() >= dip);
    }

    // ---- 1. congestion control: BBR ----
    #[test]
    fn bbr_tracks_bottleneck_bandwidth_and_min_rtt() {
        let mut b = Bbr::new();
        // Feed a steady 1448 bytes per 2-tick RTT → BtlBw ≈ 724 B/tick, RtProp = 2.
        for _ in 0..10 {
            b.on_ack(MSS, 2.0);
        }
        // One faster sample raises the bandwidth max-filter; one shorter RTT lowers RtProp.
        b.on_ack(MSS * 2, 1.0);
        assert!(b.bdp_bytes() >= MSS); // BDP ≈ BtlBw × RtProp
        assert!(b.cwnd_bytes() >= b.bdp_bytes()); // window ≥ BDP (gain ≥ 1)
        assert!(b.pacing_rate() > 0.0);
        assert!(!b.in_startup()); // plateaued out of Startup
    }

    // ---- 2. NAT traversal: priority ordering + relay fallback ----
    #[test]
    fn ice_prefers_host_then_srflx_and_falls_back_to_relay() {
        let mut a = IceAgent::new(true);
        a.gather(CandidateType::Host, loc(1, 5000), loc(1, 5000));
        a.gather(CandidateType::ServerReflexive, loc(99, 6000), loc(1, 5000));
        a.gather(CandidateType::Relay, loc(200, 7000), loc(1, 5000));
        let remote = [
            Candidate { typ: CandidateType::Host, addr: loc(2, 5000), base: loc(2, 5000) },
            Candidate { typ: CandidateType::Relay, addr: loc(201, 7000), base: loc(2, 5000) },
        ];

        // Symmetric NAT: only the relayed path is reachable on the underlay.
        let nominated = a
            .nominate(&remote, |l, r| {
                l.typ == CandidateType::Relay || r.typ == CandidateType::Relay
            })
            .unwrap();
        assert!(IceAgent::is_relayed(&nominated));
        assert_eq!(nominated.state, PairState::Succeeded);

        // Open path: the direct host↔host pair wins on priority over the relay.
        let nominated2 = a.nominate(&remote, |_, _| true).unwrap();
        assert_eq!(nominated2.local.typ, CandidateType::Host);
        assert_eq!(nominated2.remote.typ, CandidateType::Host);
        assert!(!IceAgent::is_relayed(&nominated2));
    }

    #[test]
    fn ice_returns_none_when_even_relay_is_down() {
        let mut a = IceAgent::new(false);
        a.gather(CandidateType::Host, loc(1, 5000), loc(1, 5000));
        let remote = [Candidate { typ: CandidateType::Host, addr: loc(2, 5000), base: loc(2, 5000) }];
        assert!(a.nominate(&remote, |_, _| false).is_none());
    }

    // ---- 3. mobility: identity-based connection migration ----
    #[test]
    fn connection_survives_ip_change_via_path_validation() {
        let peer = id(b"roamer");
        let mut dir = LocatorDirectory::new();
        dir.announce(peer, loc(5, 4433));
        let mut conn = Connection::new(peer, dir.resolve(&peer).unwrap(), [0xABu8; 32]);
        conn.send();
        conn.send();
        assert_eq!(conn.seq(), 2);

        // The peer roams to a new network and re-announces its locator.
        let new_path = loc(80, 4433);
        dir.announce(peer, new_path);
        let challenge = conn.begin_migration(new_path);

        // An off-path spoofer echoing the wrong token is refused; old path holds.
        assert_eq!(conn.complete_migration(challenge ^ 0xdead), Err(MigrateError::ValidationFailed));
        assert_eq!(conn.path(), loc(5, 4433));

        // The real peer echoes correctly: path migrates, sequence space preserved.
        let challenge2 = conn.begin_migration(new_path);
        assert_eq!(conn.complete_migration(challenge2), Ok(new_path));
        assert_eq!(conn.path(), new_path);
        assert_eq!(conn.seq(), 2); // connection NOT reset by the migration
        conn.send();
        assert_eq!(conn.seq(), 3); // and it keeps going

        // --- Forgery-resistance: an attacker who knows peer + seq + path but not
        //     the migration_secret cannot compute the correct challenge. ---
        //
        // A fresh Connection for the same peer but a different initial path gets a
        // different migration_secret. Even if the attacker supplies the same
        // peer/seq/new_path inputs, the resulting challenge will differ from the
        // one held by `conn`.
        let mut attacker_conn = Connection::new(peer, loc(9, 9999), [0xCDu8; 32]); // different seed → different secret
        attacker_conn.seq = conn.seq(); // attacker knows the current seq
        let attacker_challenge = attacker_conn.begin_migration(new_path); // same new_path

        // Issue a real challenge on the original connection so we can compare.
        let real_challenge = conn.begin_migration(new_path);
        assert_ne!(
            attacker_challenge, real_challenge,
            "attacker without the migration_secret must not be able to forge the challenge"
        );
        // Clean up the pending state so we don't leave conn in limbo.
        let _ = conn.complete_migration(real_challenge);
    }

    // ---- 4. offline-first: read offline, reconcile by hash on reconnect ----
    #[test]
    fn offline_writes_reconcile_by_hash_with_dedup() {
        let mut replica = OfflineReplica::new();
        let mut remote = ReplicaStore::new();
        // A document already shared with the remote.
        let shared = remote.put(b"shared-doc");
        replica.write(b"shared-doc"); // also present locally

        // Go offline: reads still work; writes queue.
        replica.set_online(false);
        assert_eq!(replica.read(&shared), Some(&b"shared-doc"[..])); // offline read OK
        let a = replica.write(b"offline-note-A");
        let _b = replica.write(b"offline-note-B");
        replica.write(b"offline-note-A"); // duplicate content → idempotent
        assert_eq!(replica.pending(), 2);

        // Reconnect + reconcile: A and B push; the shared doc would dedup if queued.
        let want = [shared];
        let report = replica.reconcile(&mut remote, &want);
        assert_eq!(report.pushed, 2);
        assert!(remote.has(&a));
        assert_eq!(replica.pending(), 0);
        assert!(replica.is_online());

        // Republishing the same content to a peer that already has it is deduped.
        replica.set_online(false);
        replica.write(b"offline-note-A"); // already on remote now
        let report2 = replica.reconcile(&mut remote, &[]);
        assert_eq!(report2.deduped, 1);
        assert_eq!(report2.pushed, 0);
    }
}

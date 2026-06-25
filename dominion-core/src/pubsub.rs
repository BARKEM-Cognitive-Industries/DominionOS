//! The **Reactive / Subscription plane** — native pub/sub over NDN (Stage 7).
//!
//! NDN ([`crate::ndn`]) gives the OS a **pull** substrate: consumers express
//! Interest in a *name* and signed Data flows back down the reverse path. This
//! module adds the **push** complement and the **subscription capability** that
//! together turn the object graph into a reactive event system — without a broker,
//! a socket, or an `IP:port` anywhere.
//!
//! > **Capability-gated subscriptions to named, content-addressed, identity-routed
//! > objects.**
//!
//! Every concern of a real pub/sub system maps onto a primitive that already
//! exists, so this module is mostly *wiring existing seams into a notify path*:
//!
//! | Pub/sub concern   | In DominionOS it is already…                                   |
//! |-------------------|--------------------------------------------------------------|
//! | Fan-out/delivery  | NDN PIT reverse-path + Content Store ([`crate::ndn`])        |
//! | Addressing        | Self-certifying `DominionId` ([`crate::dominionlink`]) — *who*   |
//! | Event identity    | Content-addressed object hash ([`crate::object`])            |
//! | Authorization     | A capability over the topic ([`crate::capability`]); revoke is recursive ([`crate::firewall`]) |
//! | Cross-trust       | The Airlock: one-way, sanitized, TTL transfers ([`crate::airlock`]) |
//! | Confidentiality   | Identity-bound PQ + AES-256-GCM sessions ([`crate::session`]) |
//! | Multi-writer merge| CRDTs ([`crate::datatypes`])                                 |
//! | Backpressure      | Rate-limited capability quotas ([`crate::firewall`])         |
//!
//! Because the same event delivered to 10k subscribers is *one* content-addressed
//! object — not 10k payloads — fan-out is naturally deduplicated by the substrate.
//!
//! ## Determinism boundary (hard discipline)
//!
//! Event *arrival timing* is I/O and non-deterministic, so it never enters the
//! replayable core. What is recorded is **what was delivered** (event objects, by
//! hash) at a **logical epoch** — see [`ReactivePlane::event_log`] and
//! [`ReactivePlane::log_digest`]. Two planes fed the same publish sequence produce
//! an identical log; wire-arrival timing stays at the boundary (caller-supplied
//! [`ReactivePlane::tick`]). This preserves the Stage 10 guarantee end-to-end.
//!
//! ## Non-goal guard (don't recreate a broker)
//!
//! The kernel provides *topics, subscription capabilities, the notify path, and the
//! durable object graph*. Brokered semantics — durable partitions, consumer groups,
//! offset management as a service — deliberately do **not** appear in this surface;
//! they belong in an app built on top. (See `docs/architecture/reactive-subscription-plane.md`.)
//!
//! Pure, safe `no_std`; host- and on-metal-tested.

use crate::dominionlink::DominionId;
use crate::airlock::{Airlock, AirlockError, IssuedCapability};
use crate::capability::{Capability, Rights};
use crate::datatypes::{GCounter, OrSet};
use crate::firewall::{CapabilityFirewall, Domain, FwError, NodeId};
use crate::hash::Hash256;
use crate::ndn::{Face, Forwarder, Name};
use crate::object::{Object, ObjectId};
use crate::session::{Frame, Session, SessionError};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

/// Identifier for a registered subscription.
pub type SubId = u64;
/// Identifier for a standing (predicate) query.
pub type QueryId = u64;

/// The face the plane forwards standing Interests towards (a synthetic "producer"
/// upstream). Subscriptions seat their PIT entry here so the notify path can fan out.
const TOPIC_UPSTREAM: Face = u64::MAX;

/// Declared delivery semantics for a subscription — never a hidden default.
///
/// Content-addressing makes **exactly-once dedup natural** (same hash = same
/// event), so [`Delivery::ExactlyOnce`] is the recommended mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Delivery {
    /// Deliver each version at most once; gaps are acceptable (e.g. live sensor feed).
    AtMostOnce,
    /// Deliver every version; duplicates are acceptable (consumer must be idempotent).
    /// The default — a conservative "never silently drop" policy.
    #[default]
    AtLeastOnce,
    /// Deliver every distinct event exactly once, de-duplicated by content hash.
    ExactlyOnce,
}

/// Maximum number of events kept in the in-memory event log. Once exceeded the
/// oldest half is discarded (compaction). This keeps memory bounded without
/// losing recent history.
const MAX_EVENT_LOG: usize = 10_000;

/// Why a reactive-plane operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TopicError {
    /// The capability lacks the right the operation needs (PUBLISH = `WRITE`,
    /// SUBSCRIBE = `READ`).
    Unauthorized,
    /// The capability/subscription was recursively revoked via the firewall.
    Revoked,
    /// A rate-limited capability quota was exhausted (backpressure).
    RateLimited,
    /// The subscription passed its TTL (airlock) epoch.
    Expired,
    /// A typed-topic event did not match the declared schema kind.
    SchemaMismatch,
    /// No such subscription / query id.
    NoSuchSubscription,
    /// The `since` hash supplied to [`ReactivePlane::subscribe`] was not found
    /// in the current (compacted) event log. The caller should re-subscribe
    /// without a `since` hint or use the current head hash.
    SinceHashNotFound,
}

impl From<FwError> for TopicError {
    fn from(e: FwError) -> Self {
        match e {
            FwError::RateLimited => TopicError::RateLimited,
            _ => TopicError::Unauthorized,
        }
    }
}

/// A capability minted over a topic prefix. Holding it *is* the authority to
/// publish to or subscribe under the name. It carries the firewall [`NodeId`] so
/// revocation is recursive and instant, and the publishing identity for presence +
/// encrypted delivery.
#[derive(Clone)]
pub struct TopicCap {
    /// The underlying unforgeable capability; its [`Rights`] separate PUBLISH
    /// (`WRITE`) from SUBSCRIBE (`READ`).
    pub cap: Capability,
    /// Firewall node backing this capability (target of recursive revoke / quota).
    pub node: NodeId,
    /// The topic name prefix this capability is scoped to.
    pub topic: Name,
    /// The identity behind the capability (publisher for presence, subscriber for crypto).
    pub identity: DominionId,
}

impl TopicCap {
    /// Does this capability confer publish (write) authority?
    pub fn can_publish(&self) -> bool {
        self.cap.is_valid() && self.cap.rights().contains(Rights::WRITE)
    }
    /// Does this capability confer subscribe (read + notify) authority?
    pub fn can_subscribe(&self) -> bool {
        self.cap.is_valid() && self.cap.rights().contains(Rights::READ)
    }
}

/// Options for a new subscription.
#[derive(Clone, Default)]
pub struct SubOptions {
    /// Declared delivery semantics. Defaults to [`Delivery::AtLeastOnce`].
    pub delivery: Delivery,
    /// Diff-based start point: deliver only events published *after* this hash
    /// (offline reconcile — "changes since X"). `None` = from the current head.
    pub since: Option<ObjectId>,
    /// Typed-schema filter: only deliver events whose object `kind` equals this.
    pub schema: Option<String>,
    /// Optional TTL (airlock) epoch after which the subscription expires.
    pub ttl: Option<u64>,
    /// The subscriber identity (enables presence + identity-bound encrypted delivery).
    pub subscriber: Option<DominionId>,
}

/// A live subscription handle returned to the caller.
#[derive(Clone)]
pub struct Subscription {
    pub id: SubId,
    pub topic: Name,
    pub node: NodeId,
    pub face: Face,
    pub delivery: Delivery,
}

/// Internal per-subscription state.
struct SubState {
    topic: Name,
    node: NodeId,
    face: Face,
    delivery: Delivery,
    schema: Option<String>,
    ttl: Option<u64>,
    /// Index into `events` of the next event to consider for delivery.
    cursor: usize,
    /// Hashes already delivered (for `ExactlyOnce` dedup).
    seen: BTreeSet<ObjectId>,
}

/// One delivered event, recorded in the deterministic log: *what* (by hash), under
/// *which* name, of *which* kind, by *whom*, at *which logical epoch*. No wall-clock
/// time — arrival timing stays at the boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub topic: Name,
    pub object: ObjectId,
    pub kind: String,
    pub publisher: DominionId,
    pub epoch: u64,
}

/// The result of a publish: the event's content id and the faces it woke.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishReceipt {
    /// Content-addressed id of the published object (the event identity).
    pub object: ObjectId,
    /// The de-duplicated downstream faces woken by the notify (PIT fan-out).
    pub notified: Vec<Face>,
}

/// Liveness of a publishing identity — presence as a first-class OS query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Liveness {
    /// The logical epoch of this identity's most recent publish.
    pub last_published_epoch: u64,
    /// How many events this identity has published.
    pub publish_count: u64,
}

impl Liveness {
    /// Was this identity active within `window` epochs of `now`?
    pub fn alive_within(&self, now: u64, window: u64) -> bool {
        now.saturating_sub(self.last_published_epoch) <= window
    }
}

/// A standing predicate query — a *materialized view* the OS maintains and pushes
/// deltas for, instead of the app polling a database.
struct QueryState {
    kind: String,
    predicate: fn(&Object) -> bool,
    /// Objects already matched (the materialized result set).
    matched: BTreeSet<ObjectId>,
    /// Newly matched objects since the last `poll_query` (the pushed delta).
    delta: Vec<ObjectId>,
}

/// The reactive plane: the durable object graph + the NDN notify path + the
/// capability firewall, tying publish/subscribe/standing-query/presence together.
pub struct ReactivePlane {
    fw: Forwarder,
    firewall: CapabilityFirewall,
    subs: BTreeMap<SubId, SubState>,
    queries: BTreeMap<QueryId, QueryState>,
    /// CRDT-backed multi-writer topics (counters and observed-remove sets).
    crdt_counters: BTreeMap<Name, GCounter>,
    crdt_sets: BTreeMap<Name, OrSet>,
    /// The deterministic, replayable event log (ordered, by hash + logical epoch).
    events: Vec<Event>,
    presence: BTreeMap<DominionId, Liveness>,
    next_sub: u64,
    next_node: u64,
    next_query: u64,
    epoch: u64,
}

impl Default for ReactivePlane {
    fn default() -> Self {
        Self::new()
    }
}

impl ReactivePlane {
    pub fn new() -> ReactivePlane {
        ReactivePlane {
            fw: Forwarder::new(),
            firewall: CapabilityFirewall::new(),
            subs: BTreeMap::new(),
            queries: BTreeMap::new(),
            crdt_counters: BTreeMap::new(),
            crdt_sets: BTreeMap::new(),
            events: Vec::new(),
            presence: BTreeMap::new(),
            next_sub: 1,
            next_node: 1,
            next_query: 1,
            epoch: 0,
        }
    }

    /// Advance the logical clock by one epoch and return it. Arrival timing enters
    /// the system here, at the boundary, as an explicit logical event — the
    /// replayable core never reads a wall clock.
    pub fn tick(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    /// The current logical epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    fn fresh_node(&mut self, domain: Domain) -> NodeId {
        let node = self.next_node;
        self.next_node += 1;
        self.firewall.register(node, domain);
        node
    }

    /// Mint a **PUBLISH** capability (write authority) over `topic` for `identity`
    /// in `domain`.
    pub fn mint_publish(&mut self, topic: Name, domain: Domain, identity: DominionId) -> TopicCap {
        let node = self.fresh_node(domain);
        TopicCap { cap: Capability::mint(node, 1, Rights::WRITE), node, topic, identity }
    }

    /// Mint a **SUBSCRIBE** capability (read + notify authority) over `topic` for
    /// `identity` in `domain`.
    pub fn mint_subscribe(&mut self, topic: Name, domain: Domain, identity: DominionId) -> TopicCap {
        let node = self.fresh_node(domain);
        TopicCap { cap: Capability::mint(node, 1, Rights::READ), node, topic, identity }
    }

    /// Attach a rate-limit quota to a capability/subscription node (backpressure):
    /// a slow subscriber can't be overwhelmed and a chatty producer can't DoS.
    pub fn set_quota(&mut self, node: NodeId, ops: u32) {
        self.firewall.set_quota(node, ops);
    }

    /// Recursively and instantly revoke a capability/subscription and everything
    /// derived from it (firewall reachability + revoke).
    pub fn revoke(&mut self, node: NodeId) {
        self.firewall.revoke(node);
    }

    /// Whether a node has been revoked.
    pub fn is_revoked(&self, node: NodeId) -> bool {
        self.firewall.is_revoked(node)
    }

    /// Register a subscription: seat a standing Interest under the topic prefix (so
    /// the notify path fans out to it) and record its delivery policy. Requires the
    /// capability to confer SUBSCRIBE (`READ`).
    pub fn subscribe(
        &mut self,
        cap: &TopicCap,
        face: Face,
        opts: SubOptions,
    ) -> Result<Subscription, TopicError> {
        if !cap.can_subscribe() {
            return Err(TopicError::Unauthorized);
        }
        if self.firewall.is_revoked(cap.node) {
            return Err(TopicError::Revoked);
        }
        // Seat the standing Interest so `notify(prefix)` wakes this face. We route
        // the prefix upstream once, then express Interest from `face`.
        self.fw.register_route(cap.topic.clone(), TOPIC_UPSTREAM);
        let _ = self.fw.recv_interest(face, &cap.topic);

        // Diff-based start: skip everything up to and including the `since` hash.
        // If the caller supplies a hash that is not present in the (possibly
        // compacted) log we refuse the subscription rather than silently
        // replaying the entire history — that would be both a correctness
        // hazard and a DoS vector when the hash is stale.
        let cursor = match opts.since {
            Some(h) => {
                match self.events.iter().position(|e| e.object == h) {
                    Some(i) => i + 1,
                    None => return Err(TopicError::SinceHashNotFound),
                }
            }
            None => self.events.len(),
        };

        let id = self.next_sub;
        self.next_sub += 1;
        let sub = SubState {
            topic: cap.topic.clone(),
            node: cap.node,
            face,
            delivery: opts.delivery,
            schema: opts.schema,
            ttl: opts.ttl,
            cursor,
            seen: BTreeSet::new(),
        };
        self.subs.insert(id, sub);
        Ok(Subscription { id, topic: cap.topic.clone(), node: cap.node, face, delivery: opts.delivery })
    }

    /// Publish `object` under the capability's topic: commit it as a new immutable
    /// version (the event), record it in the deterministic log, update presence and
    /// standing queries, then notify the PIT holders under the prefix. Requires the
    /// capability to confer PUBLISH (`WRITE`); consumes one quota unit (backpressure).
    pub fn publish(&mut self, cap: &TopicCap, object: Object) -> Result<PublishReceipt, TopicError> {
        self.publish_inner(cap, object, None)
    }

    /// Publish to a **typed topic**: the object `kind` must equal `expected_kind` or
    /// the event is rejected at the boundary ([`TopicError::SchemaMismatch`]) — no
    /// hand-written validation in every consumer.
    pub fn publish_typed(
        &mut self,
        cap: &TopicCap,
        expected_kind: &str,
        object: Object,
    ) -> Result<PublishReceipt, TopicError> {
        self.publish_inner(cap, object, Some(expected_kind))
    }

    fn publish_inner(
        &mut self,
        cap: &TopicCap,
        object: Object,
        expected_kind: Option<&str>,
    ) -> Result<PublishReceipt, TopicError> {
        if !cap.can_publish() {
            return Err(TopicError::Unauthorized);
        }
        if self.firewall.is_revoked(cap.node) {
            return Err(TopicError::Revoked);
        }
        if let Some(k) = expected_kind {
            if object.kind != k {
                return Err(TopicError::SchemaMismatch);
            }
        }
        // Backpressure: a chatty producer is bounded by its quota.
        self.firewall.consume(cap.node)?;

        let kind = object.kind.clone();
        let oid = object.id();

        // Record the delivered event in the deterministic log (by hash + epoch).
        self.events.push(Event {
            topic: cap.topic.clone(),
            object: oid,
            kind: kind.clone(),
            publisher: cap.identity,
            epoch: self.epoch,
        });

        // Bounded log: if we exceed MAX_EVENT_LOG, compact by draining the
        // oldest half. Subscription cursors are rebased so live subscribers
        // continue from the right position after compaction.
        if self.events.len() > MAX_EVENT_LOG {
            let keep_from = self.events.len() - MAX_EVENT_LOG / 2;
            self.events.drain(0..keep_from);
            // Rebase all subscription cursors: a cursor that pointed into the
            // drained region is clamped to 0 (start of the surviving log);
            // a cursor beyond keep_from is shifted down by keep_from.
            for sub in self.subs.values_mut() {
                sub.cursor = sub.cursor.saturating_sub(keep_from);
            }
        }

        // Presence: this identity is alive as of this epoch.
        let entry = self
            .presence
            .entry(cap.identity)
            .or_insert(Liveness { last_published_epoch: self.epoch, publish_count: 0 });
        entry.last_published_epoch = self.epoch;
        entry.publish_count += 1;

        // Standing (predicate) queries: push deltas for materialized views.
        for q in self.queries.values_mut() {
            if q.kind == kind && (q.predicate)(&object) && q.matched.insert(oid) {
                q.delta.push(oid);
            }
        }

        // Fan-out rides the PIT reverse-path: one announce wakes every subscriber
        // under the prefix, deduplicated — not N point-to-point sessions.
        let notified = self.fw.notify(&cap.topic);
        Ok(PublishReceipt { object: oid, notified })
    }

    /// Poll a subscription for new event object ids since its last poll. Honours the
    /// declared delivery semantics, the typed-schema filter, the TTL, and revocation.
    /// Consumes one quota unit (backpressure on the read side too).
    pub fn poll(&mut self, sub_id: SubId) -> Result<Vec<ObjectId>, TopicError> {
        let epoch = self.epoch;
        let sub = self.subs.get_mut(&sub_id).ok_or(TopicError::NoSuchSubscription)?;
        if let Some(ttl) = sub.ttl {
            if epoch > ttl {
                return Err(TopicError::Expired);
            }
        }
        if self.firewall.is_revoked(sub.node) {
            return Err(TopicError::Revoked);
        }
        self.firewall.consume(sub.node)?;

        let mut out = Vec::new();
        for ev in &self.events[sub.cursor..] {
            if !ev.topic.has_prefix(&sub.topic) {
                continue;
            }
            if let Some(schema) = &sub.schema {
                if &ev.kind != schema {
                    continue;
                }
            }
            match sub.delivery {
                Delivery::ExactlyOnce => {
                    if sub.seen.insert(ev.object) {
                        out.push(ev.object);
                    }
                }
                Delivery::AtLeastOnce | Delivery::AtMostOnce => out.push(ev.object),
            }
        }
        sub.cursor = self.events.len();
        Ok(out)
    }

    /// Register a **standing predicate query** — e.g. "all `Invoice` objects where
    /// `amount > 1000`". The OS maintains the result set and pushes deltas via
    /// [`poll_query`](Self::poll_query). Requires SUBSCRIBE (`READ`) authority.
    pub fn standing_query(
        &mut self,
        cap: &TopicCap,
        kind: &str,
        predicate: fn(&Object) -> bool,
    ) -> Result<QueryId, TopicError> {
        if !cap.can_subscribe() {
            return Err(TopicError::Unauthorized);
        }
        let id = self.next_query;
        self.next_query += 1;
        self.queries.insert(
            id,
            QueryState {
                kind: String::from(kind),
                predicate,
                matched: BTreeSet::new(),
                delta: Vec::new(),
            },
        );
        Ok(id)
    }

    /// Drain the pushed delta for a standing query (objects newly matching since the
    /// last call). The full materialized result set is available via
    /// [`query_result`](Self::query_result).
    pub fn poll_query(&mut self, qid: QueryId) -> Result<Vec<ObjectId>, TopicError> {
        let q = self.queries.get_mut(&qid).ok_or(TopicError::NoSuchSubscription)?;
        Ok(core::mem::take(&mut q.delta))
    }

    /// The full materialized result set of a standing query.
    pub fn query_result(&self, qid: QueryId) -> Option<Vec<ObjectId>> {
        self.queries.get(&qid).map(|q| q.matched.iter().copied().collect())
    }

    /// The NDN face a subscription delivers to (its seat in the PIT fan-out tree).
    pub fn subscription_face(&self, sub_id: SubId) -> Option<Face> {
        self.subs.get(&sub_id).map(|s| s.face)
    }

    /// Presence / liveness query for a publishing identity.
    pub fn presence(&self, identity: &DominionId) -> Option<Liveness> {
        self.presence.get(identity).copied()
    }

    /// The deterministic, replayable event log (ordered, by hash + logical epoch).
    pub fn event_log(&self) -> &[Event] {
        &self.events
    }

    /// A content hash over the ordered event log — two planes fed the same publish
    /// sequence produce the same digest (the determinism boundary, checkable).
    pub fn log_digest(&self) -> Hash256 {
        let mut buf = Vec::new();
        for ev in &self.events {
            buf.extend_from_slice(&ev.object.0);
            buf.extend_from_slice(&ev.epoch.to_le_bytes());
        }
        Hash256::of(&buf)
    }

    // --- CRDT-backed multi-writer topics -------------------------------------

    /// Increment a CRDT counter topic from a given replica (multi-writer, no
    /// coordinating server). Converges under [`merge_counter`](Self::merge_counter).
    pub fn crdt_increment(&mut self, topic: &Name, replica: u64, by: u64) {
        self.crdt_counters.entry(topic.clone()).or_default().increment(replica, by);
    }

    /// The converged value of a CRDT counter topic.
    pub fn crdt_counter_value(&self, topic: &Name) -> u64 {
        self.crdt_counters.get(topic).map(|c| c.value()).unwrap_or(0)
    }

    /// Merge another replica's view of a CRDT counter topic into ours (commutative,
    /// idempotent, associative — order-independent convergence).
    pub fn merge_counter(&mut self, topic: &Name, other: &GCounter) {
        let merged = match self.crdt_counters.get(topic) {
            Some(local) => local.merge(other),
            None => other.merge(&GCounter::new()),
        };
        self.crdt_counters.insert(topic.clone(), merged);
    }

    /// Add an element to a CRDT observed-remove set topic (multi-writer).
    pub fn crdt_set_add(&mut self, topic: &Name, elem: &[u8], tag: u64) {
        self.crdt_sets.entry(topic.clone()).or_default().add(elem, tag);
    }

    /// Whether a CRDT set topic currently contains an element.
    pub fn crdt_set_contains(&self, topic: &Name, elem: &[u8]) -> bool {
        self.crdt_sets.get(topic).map(|s| s.contains(elem)).unwrap_or(false)
    }

    /// Merge another replica's observed-remove set into ours.
    pub fn merge_set(&mut self, topic: &Name, other: &OrSet) {
        let merged = match self.crdt_sets.get(topic) {
            Some(local) => local.merge(other),
            None => other.merge(&OrSet::new()),
        };
        self.crdt_sets.insert(topic.clone(), merged);
    }

    // --- helpers for cross-domain + encrypted delivery -----------------------

    /// How many faces a notify under `prefix` would currently wake (PIT fan-out
    /// width) — useful for tests and for the resource governor.
    pub fn fanout(&self, prefix: &Name) -> usize {
        self.fw.notify(prefix).len()
    }
}

/// **Cross-domain issuance via the Airlock.** Sanitize a topic capability down to
/// the policy's `max_rights` (e.g. a publish cap → subscribe-only), enforce the
/// approval quorum, and stamp a TTL — exactly what "publish-only", "subscribe-only",
/// and "expires in 1h" topics need (sensor feeds, audit streams, untrusted
/// publishers). The transfer is one-way and monotonic: the result can only *drop*
/// rights, never gain them.
pub fn issue_cross_domain(
    airlock: &Airlock,
    cap: &TopicCap,
    from: Domain,
    to: Domain,
    approvals: u32,
    now: u64,
) -> Result<IssuedCapability, AirlockError> {
    airlock.transfer(cap.cap, from, to, approvals, now)
}

/// **Encrypted delivery (item 5).** Seal a notification (the 32-byte event hash)
/// into an identity-bound, PQ-keyed, AES-256-GCM [`Frame`]. No plaintext crosses
/// the wire; the subscriber identity + epoch are bound in as associated data.
pub fn seal_notification(
    session: &mut Session,
    now: u64,
    object: ObjectId,
) -> Result<Frame, SessionError> {
    session.seal(now, &object.0)
}

/// Open a sealed notification back into the event hash. Returns
/// [`SessionError::AuthFailed`] on any tamper, wrong key, or wrong AAD.
pub fn open_notification(
    session: &Session,
    now: u64,
    frame: &Frame,
) -> Result<ObjectId, SessionError> {
    let pt = session.open(now, frame)?;
    if pt.len() != 32 {
        return Err(SessionError::AuthFailed);
    }
    let mut h = [0u8; 32];
    h.copy_from_slice(&pt);
    Ok(Hash256(h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::airlock::TransferPolicy;
    use crate::object::Datum;
    use crate::session::{KemIdentity, Session};

    fn id(seed: &[u8]) -> DominionId {
        DominionId(Hash256::of(seed))
    }

    fn invoice(amount: i64) -> Object {
        Object::new("Invoice").with("amount", Datum::Int(amount))
    }

    // ---- item 1 & 13: push/notify path + PIT fan-out ----
    #[test]
    fn publish_notifies_subscribers_under_prefix_via_pit_fanout() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/jayden/inbox");
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"producer"));
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"sub"));

        // Three subscribers seat standing Interest under the prefix.
        p.subscribe(&subcap, 1, SubOptions::default()).unwrap();
        p.subscribe(&subcap, 2, SubOptions::default()).unwrap();
        p.subscribe(&subcap, 3, SubOptions::default()).unwrap();

        let receipt = p.publish(&pubcap, Object::new("Msg").with("body", Datum::Text("hi".into()))).unwrap();
        // One publish wakes all three faces, deduplicated.
        assert_eq!(receipt.notified, alloc::vec![1, 2, 3]);
        assert_eq!(p.fanout(&topic), 3);
    }

    // ---- item 2: subscription is a capability (PUBLISH vs SUBSCRIBE) ----
    #[test]
    fn subscribe_cap_cannot_publish_and_publish_cap_cannot_subscribe() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/t");
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"a"));
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"b"));
        assert!(subcap.can_subscribe() && !subcap.can_publish());
        assert!(pubcap.can_publish() && !pubcap.can_subscribe());
        assert_eq!(p.publish(&subcap, invoice(1)).err(), Some(TopicError::Unauthorized));
        assert_eq!(p.subscribe(&pubcap, 1, SubOptions::default()).err(), Some(TopicError::Unauthorized));
    }

    // ---- item 3a: recursive/instant revocation via firewall ----
    #[test]
    fn revocation_is_instant_and_blocks_delivery() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/t");
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"s"));
        let sub = p.subscribe(&subcap, 1, SubOptions::default()).unwrap();
        p.revoke(sub.node);
        assert!(p.is_revoked(sub.node));
        assert_eq!(p.poll(sub.id).err(), Some(TopicError::Revoked));
    }

    // ---- item 3b: cross-domain issuance via the Airlock (one-way / TTL) ----
    #[test]
    fn airlock_sanitizes_publish_cap_to_subscribe_only_with_ttl() {
        let topic = Name::parse("/audit");
        // A topic *owner* holds both publish (WRITE) and subscribe (READ) authority.
        let owner = TopicCap {
            cap: Capability::mint(0, 1, Rights::READ.union(Rights::WRITE)),
            node: 0,
            topic,
            identity: id(b"auditor"),
        };
        let mut airlock = Airlock::new();
        // One-way Financial → AiAgent: max SUBSCRIBE (read), expires after 5 ticks.
        airlock.add_policy(TransferPolicy {
            from: Domain::Financial,
            to: Domain::AiAgent,
            max_rights: Rights::READ,
            ttl: Some(5),
            approvals_required: 0,
        });
        let issued = issue_cross_domain(&airlock, &owner, Domain::Financial, Domain::AiAgent, 0, 0).unwrap();
        // Sanitized: write authority stripped, only read survives (subscribe-only).
        assert!(issued.capability.rights().contains(Rights::READ));
        assert!(!issued.capability.rights().contains(Rights::WRITE));
        assert_eq!(issued.expires_at, Some(5));
        // The reverse direction has no policy — one-way by construction.
        assert!(issue_cross_domain(&airlock, &owner, Domain::AiAgent, Domain::Financial, 0, 0).is_err());
    }

    // ---- item 4: backpressure via rate-limited quotas ----
    #[test]
    fn quota_bounds_a_chatty_publisher() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/feed");
        let pubcap = p.mint_publish(topic, Domain::Personal, id(b"sensor"));
        p.set_quota(pubcap.node, 2); // only two publishes allowed
        assert!(p.publish(&pubcap, invoice(1)).is_ok());
        assert!(p.publish(&pubcap, invoice(2)).is_ok());
        assert_eq!(p.publish(&pubcap, invoice(3)).err(), Some(TopicError::RateLimited));
    }

    // ---- item 5: per-subscription encryption over identity-bound sessions ----
    #[test]
    fn notification_is_encrypted_end_to_end_and_rejects_tamper() {
        let alice = KemIdentity::generate(b"alice-seed");
        let bob = KemIdentity::generate(b"bob-seed");
        let (mut tx, ct) = Session::initiate(alice.id, bob.id, &bob.public, b"eph", 100).unwrap();
        let rx = Session::accept(&bob, alice.id, &ct, 100);
        let oid = Hash256::of(b"event-object");
        let frame = seal_notification(&mut tx, 1, oid).unwrap();
        assert_eq!(open_notification(&rx, 1, &frame).unwrap(), oid);
        // Tamper → authentication failure, no plaintext leaks.
        let mut bad = frame.clone();
        bad.corrupt_first_byte();
        assert_eq!(open_notification(&rx, 1, &bad).err(), Some(SessionError::AuthFailed));
    }

    // ---- item 6: declared delivery semantics; exactly-once dedup ----
    #[test]
    fn exactly_once_dedups_by_content_hash_at_least_once_does_not() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/t");
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"p"));
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"s"));

        let once = p
            .subscribe(&subcap, 1, SubOptions { delivery: Delivery::ExactlyOnce, ..Default::default() })
            .unwrap();
        let many = p
            .subscribe(&subcap, 2, SubOptions { delivery: Delivery::AtLeastOnce, ..Default::default() })
            .unwrap();

        // Publish the *same* object twice (same content hash = same event).
        p.publish(&pubcap, invoice(42)).unwrap();
        p.publish(&pubcap, invoice(42)).unwrap();

        assert_eq!(p.poll(once.id).unwrap().len(), 1); // de-duplicated
        assert_eq!(p.poll(many.id).unwrap().len(), 2); // both delivered
    }

    // ---- item 7: versioned / diff-based delivery ("since hash X") ----
    #[test]
    fn diff_based_delivery_reconciles_from_a_known_hash() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/log");
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"p"));
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"s"));

        let h1 = p.publish(&pubcap, invoice(1)).unwrap().object;
        let _h2 = p.publish(&pubcap, invoice(2)).unwrap().object;
        let h3 = p.publish(&pubcap, invoice(3)).unwrap().object;

        // A subscriber that last saw h1 reconciles only the delta after it.
        let sub = p
            .subscribe(&subcap, 1, SubOptions { since: Some(h1), ..Default::default() })
            .unwrap();
        let got = p.poll(sub.id).unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&h3));
        assert!(!got.contains(&h1));
    }

    // ---- item 8: standing (predicate) queries with pushed deltas ----
    #[test]
    fn standing_query_pushes_deltas_for_matching_objects() {
        fn big(o: &Object) -> bool {
            matches!(o.get("amount"), Some(Datum::Int(a)) if *a > 1000)
        }
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/invoices");
        let pubcap = p.mint_publish(topic.clone(), Domain::Financial, id(b"p"));
        let subcap = p.mint_subscribe(topic.clone(), Domain::Financial, id(b"s"));

        let q = p.standing_query(&subcap, "Invoice", big).unwrap();
        p.publish(&pubcap, invoice(500)).unwrap(); // no match
        p.publish(&pubcap, invoice(5000)).unwrap(); // match
        let delta = p.poll_query(q).unwrap();
        assert_eq!(delta.len(), 1);
        // The materialized result set holds the match; the delta drains.
        assert_eq!(p.query_result(q).unwrap().len(), 1);
        assert!(p.poll_query(q).unwrap().is_empty());
    }

    // ---- item 9: CRDT-backed multi-writer topics ----
    #[test]
    fn crdt_counter_topic_converges_across_replicas() {
        let mut a = ReactivePlane::new();
        let mut b = ReactivePlane::new();
        let topic = Name::parse("/likes");
        a.crdt_increment(&topic, 1, 3); // replica 1 on plane A
        b.crdt_increment(&topic, 2, 4); // replica 2 on plane B
        // Exchange and merge — order-independent convergence to the same value.
        let a_view = a.crdt_counters.get(&topic).unwrap().clone();
        let b_view = b.crdt_counters.get(&topic).unwrap().clone();
        a.merge_counter(&topic, &b_view);
        b.merge_counter(&topic, &a_view);
        assert_eq!(a.crdt_counter_value(&topic), 7);
        assert_eq!(b.crdt_counter_value(&topic), 7);

        // Observed-remove set topic too.
        let set_topic = Name::parse("/members");
        a.crdt_set_add(&set_topic, b"alice", 1);
        b.crdt_set_add(&set_topic, b"bob", 2);
        let bset = b.crdt_sets.get(&set_topic).unwrap().clone();
        a.merge_set(&set_topic, &bset);
        assert!(a.crdt_set_contains(&set_topic, b"alice"));
        assert!(a.crdt_set_contains(&set_topic, b"bob"));
    }

    // ---- item 10: typed event schemas validated at the boundary ----
    #[test]
    fn typed_topic_rejects_wrong_kind_and_filters_on_subscribe() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/typed");
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"p"));
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"s"));

        // Publish-side boundary: a non-Invoice object is rejected.
        assert_eq!(
            p.publish_typed(&pubcap, "Invoice", Object::new("Receipt")).err(),
            Some(TopicError::SchemaMismatch)
        );
        assert!(p.publish_typed(&pubcap, "Invoice", invoice(10)).is_ok());

        // Subscribe-side filter: an Invoice-typed subscription ignores other kinds.
        // (Its cursor starts at the current head, so only events after it count.)
        let sub = p
            .subscribe(&subcap, 1, SubOptions { schema: Some("Invoice".into()), ..Default::default() })
            .unwrap();
        p.publish(&pubcap, Object::new("Receipt")).unwrap(); // filtered out by schema
        p.publish(&pubcap, invoice(20)).unwrap(); // delivered
        let got = p.poll(sub.id).unwrap();
        // Only the Invoice event passes the typed-schema filter.
        assert_eq!(got.len(), 1);
    }

    // ---- item 11: presence / liveness as an OS query ----
    #[test]
    fn presence_tracks_last_publish_per_identity() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/t");
        let who = id(b"publisher");
        let pubcap = p.mint_publish(topic, Domain::Personal, who);
        assert!(p.presence(&who).is_none());
        p.tick();
        p.tick(); // epoch = 2
        p.publish(&pubcap, invoice(1)).unwrap();
        p.publish(&pubcap, invoice(2)).unwrap();
        let live = p.presence(&who).unwrap();
        assert_eq!(live.publish_count, 2);
        assert_eq!(live.last_published_epoch, 2);
        assert!(live.alive_within(3, 2));
        assert!(!live.alive_within(10, 2));
    }

    // ---- item 12: determinism boundary — same publishes => same log ----
    #[test]
    fn event_log_is_deterministic_across_replays() {
        fn replay() -> Hash256 {
            let mut p = ReactivePlane::new();
            let topic = Name::parse("/t");
            let pubcap = p.mint_publish(topic, Domain::Personal, id(b"p"));
            for k in 0..5 {
                p.tick();
                p.publish(&pubcap, invoice(k)).unwrap();
            }
            p.log_digest()
        }
        // Two independent runs of the same publish sequence agree exactly.
        assert_eq!(replay(), replay());
        // The log records hashes + logical epoch only (no wall-clock).
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/t");
        let pubcap = p.mint_publish(topic, Domain::Personal, id(b"p"));
        p.tick();
        let h = p.publish(&pubcap, invoice(7)).unwrap().object;
        assert_eq!(p.event_log()[0].object, h);
        assert_eq!(p.event_log()[0].epoch, 1);
    }

    // ---- item 14: non-goal guard — no broker primitives leak into the surface ----
    #[test]
    fn surface_stays_minimal_no_broker_state() {
        // A fresh plane holds no partitions, consumer groups, or offset tables —
        // only topics + the durable log. This test documents the guard: brokered
        // semantics are an app concern, not a kernel one.
        let p = ReactivePlane::new();
        assert!(p.event_log().is_empty());
        assert_eq!(p.epoch(), 0);
    }

    // ---- bug-fix 1: event log must stay bounded after MAX_EVENT_LOG + N pushes ----
    #[test]
    fn event_log_stays_bounded_after_many_publishes() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/flood");
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"flood-producer"));

        // Push MAX_EVENT_LOG + 100 events — well past the compaction threshold.
        for k in 0..(MAX_EVENT_LOG + 100) as i64 {
            p.publish(&pubcap, invoice(k)).unwrap();
        }

        // The log must never exceed MAX_EVENT_LOG entries.
        assert!(
            p.event_log().len() <= MAX_EVENT_LOG,
            "log grew to {} entries (limit {})",
            p.event_log().len(),
            MAX_EVENT_LOG,
        );
    }

    // ---- bug-fix 2: subscribe with an unknown since-hash must NOT replay full history ----
    #[test]
    fn subscribe_with_unknown_since_hash_returns_error_not_full_replay() {
        let mut p = ReactivePlane::new();
        let topic = Name::parse("/history");
        let pubcap = p.mint_publish(topic.clone(), Domain::Personal, id(b"hist-pub"));
        let subcap = p.mint_subscribe(topic.clone(), Domain::Personal, id(b"hist-sub"));

        // Publish several events so the log is non-empty.
        for k in 0..10i64 {
            p.publish(&pubcap, invoice(k)).unwrap();
        }

        // A hash that was never published — unknown / stale.
        let ghost_hash = Hash256::of(b"this-hash-does-not-exist-in-the-log");

        let result = p.subscribe(
            &subcap,
            42,
            SubOptions { since: Some(ghost_hash), ..Default::default() },
        );

        // Must be refused — not a silent full-history replay.
        assert_eq!(
            result.err(),
            Some(TopicError::SinceHashNotFound),
            "expected SinceHashNotFound but subscribe succeeded (would have replayed full history)",
        );
    }
}

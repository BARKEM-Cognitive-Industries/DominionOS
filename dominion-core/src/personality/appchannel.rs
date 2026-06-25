//! Capability-gated **application communication channel** — the secure wrapper that
//! lets a sandboxed app (see [`super::applaunch`]) talk to other code, processes and
//! services *only* through a capability, never via ambient sockets/IPC.
//!
//! [`crate::pubsub`] supplies the capability-minted topic plane (publish = `WRITE`,
//! subscribe = `READ`, recursive revoke, quotas). This module supplies the missing
//! message-body wrapper an application actually programs against: a [`CapChannel`]
//! built from a [`TopicCap`] with dead-simple `send`/`recv` over a shared [`CapBus`],
//! where **authority is the capability** —
//!
//! * no capability ⇒ no communication (default-closed);
//! * a subscribe-only cap cannot `send`, a publish-only cap cannot `recv`;
//! * topic confinement: a channel only ever receives messages on its own topic prefix;
//! * **instant revocation** — [`CapChannel::revoke`] tampers the held capability, so
//!   every subsequent `send`/`recv` traps (the capability-native "cut this app off").
//!
//! This is how a Linux/Windows app or a polyglot program reaches the rest of the
//! system securely: the launcher hands it a channel cap scoped to exactly the topics
//! it may use, and that cap *is* the wiretap-proof, revocable connection. Pure, safe
//! `no_std`, host-tested.

use crate::ndn::Name;
use crate::pubsub::TopicCap;
use alloc::vec::Vec;

/// Why a channel operation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChanError {
    /// The capability does not confer the needed authority (publish for `send`,
    /// subscribe for `recv`), or it has been revoked (default-closed).
    Unauthorized,
}

/// A shared, capability-gated message bus. Messages are addressed by topic [`Name`];
/// a [`CapChannel`] can only put messages it has publish authority for and only take
/// messages under the topic prefix it has subscribe authority for. In the kernel this
/// rides [`crate::pubsub::ReactivePlane`]'s NDN notify path; here it is an in-memory
/// log with identical capability semantics, so the wrapper is testable end to end.
#[derive(Default)]
pub struct CapBus {
    log: Vec<(Name, Vec<u8>)>,
}

impl CapBus {
    pub fn new() -> CapBus {
        CapBus { log: Vec::new() }
    }

    /// Total messages on the bus (across all topics).
    pub fn len(&self) -> usize {
        self.log.len()
    }

    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }

    fn put(&mut self, topic: Name, body: Vec<u8>) {
        self.log.push((topic, body));
    }
}

/// An application's capability-scoped connection to the bus. Holds a [`TopicCap`] and a
/// per-channel cursor so each `recv` returns only newly-delivered messages.
pub struct CapChannel {
    cap: TopicCap,
    cursor: usize,
}

impl CapChannel {
    /// Open a channel over a topic capability.
    pub fn open(cap: TopicCap) -> CapChannel {
        CapChannel { cap, cursor: 0 }
    }

    /// The topic prefix this channel is scoped to.
    pub fn topic(&self) -> &Name {
        &self.cap.topic
    }

    /// Send a message body. Requires the capability to confer publish (`WRITE`) and be
    /// valid; otherwise the call is refused (default-closed / revoked).
    pub fn send(&self, bus: &mut CapBus, body: &[u8]) -> Result<(), ChanError> {
        if !self.cap.can_publish() {
            return Err(ChanError::Unauthorized);
        }
        bus.put(self.cap.topic.clone(), body.to_vec());
        Ok(())
    }

    /// Receive every message on the channel's topic prefix delivered since the last
    /// `recv`. Requires the capability to confer subscribe (`READ`) and be valid.
    pub fn recv(&mut self, bus: &CapBus) -> Result<Vec<Vec<u8>>, ChanError> {
        if !self.cap.can_subscribe() {
            return Err(ChanError::Unauthorized);
        }
        let mut out = Vec::new();
        for (topic, body) in &bus.log[self.cursor..] {
            if topic.has_prefix(&self.cap.topic) {
                out.push(body.clone());
            }
        }
        self.cursor = bus.log.len();
        Ok(out)
    }

    /// **Cut this app off.** Tamper the held capability so every subsequent `send`/
    /// `recv` traps — instant, race-free revocation of the app's connection.
    pub fn revoke(&mut self) {
        self.cap.cap = self.cap.cap.tamper();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dominionlink::DominionId;
    use crate::firewall::Domain;
    use crate::hash::Hash256;
    use crate::ndn::Name;
    use crate::pubsub::ReactivePlane;

    fn ident(seed: &[u8]) -> DominionId {
        DominionId(Hash256::of(seed))
    }

    // A topic owner that holds both publish and subscribe authority is convenient for
    // building send+recv channels in one test.
    fn pub_cap(plane: &mut ReactivePlane, topic: &str) -> TopicCap {
        plane.mint_publish(Name::parse(topic), Domain::Personal, ident(b"pub"))
    }
    fn sub_cap(plane: &mut ReactivePlane, topic: &str) -> TopicCap {
        plane.mint_subscribe(Name::parse(topic), Domain::Personal, ident(b"sub"))
    }

    #[test]
    fn an_app_sends_and_another_receives_over_its_topic() {
        let mut plane = ReactivePlane::new();
        let mut bus = CapBus::new();
        let sender = CapChannel::open(pub_cap(&mut plane, "/svc/log"));
        let mut receiver = CapChannel::open(sub_cap(&mut plane, "/svc/log"));

        sender.send(&mut bus, b"hello service").unwrap();
        sender.send(&mut bus, b"second message").unwrap();
        let got = receiver.recv(&bus).unwrap();
        assert_eq!(got, alloc::vec![b"hello service".to_vec(), b"second message".to_vec()]);
        // A second recv with no new traffic yields nothing (cursor advanced).
        assert!(receiver.recv(&bus).unwrap().is_empty());
    }

    #[test]
    fn a_subscribe_only_channel_cannot_send() {
        let mut plane = ReactivePlane::new();
        let mut bus = CapBus::new();
        let listener = CapChannel::open(sub_cap(&mut plane, "/t"));
        // No publish authority ⇒ send refused (default-closed).
        assert_eq!(listener.send(&mut bus, b"x"), Err(ChanError::Unauthorized));
    }

    #[test]
    fn a_publish_only_channel_cannot_receive() {
        let mut plane = ReactivePlane::new();
        let bus = CapBus::new();
        let mut talker = CapChannel::open(pub_cap(&mut plane, "/t"));
        assert_eq!(talker.recv(&bus), Err(ChanError::Unauthorized));
    }

    #[test]
    fn topic_confinement_isolates_channels() {
        let mut plane = ReactivePlane::new();
        let mut bus = CapBus::new();
        let a = CapChannel::open(pub_cap(&mut plane, "/app/a"));
        let b = CapChannel::open(pub_cap(&mut plane, "/app/b"));
        let mut listen_a = CapChannel::open(sub_cap(&mut plane, "/app/a"));

        a.send(&mut bus, b"for-a").unwrap();
        b.send(&mut bus, b"for-b").unwrap();
        // The /app/a listener only sees /app/a traffic — never /app/b.
        let got = listen_a.recv(&bus).unwrap();
        assert_eq!(got, alloc::vec![b"for-a".to_vec()]);
    }

    #[test]
    fn revocation_cuts_the_channel_off_instantly() {
        let mut plane = ReactivePlane::new();
        let mut bus = CapBus::new();
        let mut sender = CapChannel::open(pub_cap(&mut plane, "/t"));
        sender.send(&mut bus, b"before").unwrap();
        sender.revoke();
        // After revocation the capability is invalid → every op traps.
        assert_eq!(sender.send(&mut bus, b"after"), Err(ChanError::Unauthorized));
    }
}

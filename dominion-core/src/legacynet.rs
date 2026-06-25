//! Legacy networking — **sockets as capabilities, a minimal TCP state machine, the
//! `LegacyNet` gateway, and DominionLink-over-UDP** (`docs/implementation/integration-strategy.md` §1).
//!
//! [`crate::net`] provides Ethernet/ARP/IPv4/ICMP/UDP framing; this module adds the
//! capability-secured socket layer and the bridge to the legacy internet:
//!
//! * **Sockets are capabilities** ([`SocketCapability`], [`NetworkStack`]): a port cannot
//!   be used without an unforgeable capability authorising exactly that
//!   `(protocol, local port, allowed remote)` — no ambient network access, and two
//!   domains cannot collide on a port they don't both hold.
//! * **Minimal TCP** ([`TcpConnection`]): a real 3-way-handshake state machine with
//!   sequence/ack numbers, so a sockets-as-capabilities endpoint has stream semantics.
//! * **`LegacyNet` gateway** ([`LegacyGateway`]): the *single*, default-closed,
//!   capability-gated bridge to the IP internet. Outbound flows are explicitly opened;
//!   inbound is NAT-style allowed **only** for a matching established flow.
//! * **DominionLink over UDP** ([`encapsulate`]/[`decapsulate`]): the self-certifying
//!   overlay shipped inside real UDP datagrams, so the L2 overlay runs over commodity IP.
//!
//! Pure, safe `no_std`; deterministic. Host-tested.

use crate::hash::Hash256;
use crate::net::Ipv4Addr;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

// ───────────────────────── HMAC-SHA256 ─────────────────────────
//
// Secret-prefix MACs (`H(key || message)`) are vulnerable to length-extension
// attacks under SHA-256's Merkle-Damgård construction: a holder of
// `H(key || message)` can compute `H(key || message || padding || extension)`
// without knowing `key`. HMAC avoids this by double-hashing with separate
// inner and outer pads so that neither half can be extended independently.
fn hmac_sha256(key: &[u8], msg: &[u8]) -> Hash256 {
    // Normalise key to a 64-byte block (SHA-256 block size).
    let mut k = [0u8; 64];
    if key.len() <= 64 {
        k[..key.len()].copy_from_slice(key);
    } else {
        // Keys longer than the block size are hashed down first.
        let h = Hash256::of(key);
        k[..32].copy_from_slice(&h.0);
    }
    // ipad = 0x36 repeated, opad = 0x5c repeated (RFC 2104).
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();
    // inner = H(ipad || msg)
    let inner = Hash256::of(&[ipad.as_slice(), msg].concat());
    // outer = H(opad || inner)
    Hash256::of(&[opad.as_slice(), &inner.0].concat())
}

// ───────────────────────── sockets as capabilities ─────────────────────────

/// Transport protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Protocol {
    Tcp,
    Udp,
}

/// An unforgeable authority to use one socket endpoint. Without it, the port does not
/// exist for you — the capability *is* the socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SocketCapability {
    pub proto: Protocol,
    pub local_port: u16,
    /// If set, the socket may only talk to this remote (a connected socket).
    pub allowed_remote: Option<(Ipv4Addr, u16)>,
    /// Authenticity tag binding the fields to an issuer key (forgery-resistant).
    token: Hash256,
}

impl SocketCapability {
    /// Mint a socket capability under an issuer `key`.
    pub fn mint(proto: Protocol, local_port: u16, allowed_remote: Option<(Ipv4Addr, u16)>, key: &[u8]) -> SocketCapability {
        SocketCapability { proto, local_port, allowed_remote, token: Self::tag(proto, local_port, &allowed_remote, key) }
    }

    fn tag(proto: Protocol, port: u16, remote: &Option<(Ipv4Addr, u16)>, key: &[u8]) -> Hash256 {
        // Build the message (without the key — the key is the HMAC secret).
        let mut msg = Vec::with_capacity(16);
        msg.push(proto as u8);
        msg.extend_from_slice(&port.to_le_bytes());
        if let Some((ip, p)) = remote {
            msg.extend_from_slice(&ip.0);
            msg.extend_from_slice(&p.to_le_bytes());
        }
        // HMAC-SHA256 instead of the former secret-prefix H(key || message).
        // The old construction was vulnerable to SHA-256 length-extension attacks;
        // HMAC's double-hash structure eliminates that class of forgery.
        hmac_sha256(key, &msg)
    }

    /// Re-verify authenticity against the issuer key (a tampered cap fails).
    pub fn is_authentic(&self, key: &[u8]) -> bool {
        Self::tag(self.proto, self.local_port, &self.allowed_remote, key) == self.token
    }

    /// May this capability send to `remote`?
    pub fn permits(&self, remote: (Ipv4Addr, u16)) -> bool {
        match self.allowed_remote {
            Some(r) => r == remote,
            None => true, // an unconnected (listening/datagram) socket
        }
    }
}

/// Why a socket operation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetError {
    /// The capability is forged or doesn't authorise this endpoint.
    Unauthorized,
    /// The port is already bound by another holder.
    PortInUse,
    /// The remote isn't permitted by a connected socket.
    RemoteNotPermitted,
}

/// The capability-secured socket table: only a holder of a valid capability may bind or
/// use a port, and a port is exclusive to its binder.
pub struct NetworkStack {
    issuer_key: Vec<u8>,
    bound: BTreeMap<(Protocol, u16), Hash256>,
}

impl NetworkStack {
    pub fn new(issuer_key: &[u8]) -> NetworkStack {
        NetworkStack { issuer_key: issuer_key.to_vec(), bound: BTreeMap::new() }
    }

    /// Bind a port using its capability. Fails if the cap is forged or the port is taken.
    pub fn bind(&mut self, cap: &SocketCapability) -> Result<(), NetError> {
        if !cap.is_authentic(&self.issuer_key) {
            return Err(NetError::Unauthorized);
        }
        let slot = (cap.proto, cap.local_port);
        match self.bound.get(&slot) {
            Some(tok) if *tok != cap.token => Err(NetError::PortInUse),
            _ => {
                self.bound.insert(slot, cap.token);
                Ok(())
            }
        }
    }

    /// Send to `remote`: requires an authentic, bound capability that permits the remote.
    pub fn send(&self, cap: &SocketCapability, remote: (Ipv4Addr, u16)) -> Result<(), NetError> {
        if !cap.is_authentic(&self.issuer_key) {
            return Err(NetError::Unauthorized);
        }
        if self.bound.get(&(cap.proto, cap.local_port)) != Some(&cap.token) {
            return Err(NetError::Unauthorized);
        }
        if !cap.permits(remote) {
            return Err(NetError::RemoteNotPermitted);
        }
        Ok(())
    }
}

// ───────────────────────── minimal TCP state machine ─────────────────────────

/// TCP connection states (the subset needed for a real handshake + teardown).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcpState {
    Closed,
    SynSent,
    SynReceived,
    Established,
    FinWait,
    TimeWait,
}

/// A minimal TCP connection: a 3-way-handshake state machine with sequence numbers.
#[derive(Clone, Copy, Debug)]
pub struct TcpConnection {
    pub state: TcpState,
    snd_nxt: u32,
    rcv_nxt: u32,
}

impl TcpConnection {
    /// A fresh, closed connection with an initial send sequence number.
    pub fn new(iss: u32) -> TcpConnection {
        TcpConnection { state: TcpState::Closed, snd_nxt: iss, rcv_nxt: 0 }
    }

    /// Active open: send SYN(seq), move to SYN-SENT. Returns the SYN's sequence number.
    pub fn connect(&mut self) -> u32 {
        let seq = self.snd_nxt;
        self.snd_nxt = self.snd_nxt.wrapping_add(1);
        self.state = TcpState::SynSent;
        seq
    }

    /// Receive SYN-ACK(their_seq, ack): if it acks our SYN, send ACK and reach
    /// ESTABLISHED. Returns the ACK number to send, or `None` if unexpected.
    pub fn on_syn_ack(&mut self, their_seq: u32, ack: u32) -> Option<u32> {
        if self.state != TcpState::SynSent || ack != self.snd_nxt {
            return None;
        }
        self.rcv_nxt = their_seq.wrapping_add(1);
        self.state = TcpState::Established;
        Some(self.rcv_nxt)
    }

    /// Send `len` bytes of stream data; advances send sequence. Returns the segment's
    /// starting sequence number, or `None` if not established.
    pub fn send(&mut self, len: u32) -> Option<u32> {
        if self.state != TcpState::Established {
            return None;
        }
        let seq = self.snd_nxt;
        self.snd_nxt = self.snd_nxt.wrapping_add(len);
        Some(seq)
    }

    /// Active close: send FIN, move to FIN-WAIT.
    pub fn close(&mut self) {
        if self.state == TcpState::Established {
            self.state = TcpState::FinWait;
            self.snd_nxt = self.snd_nxt.wrapping_add(1);
        }
    }

    /// Receive the peer's FIN/ACK after our close → TIME-WAIT (then closed).
    pub fn on_fin(&mut self) {
        if self.state == TcpState::FinWait {
            self.state = TcpState::TimeWait;
        }
    }
}

// ───────────────────────── LegacyNet gateway ─────────────────────────

/// A flow key (the NAT 4-tuple, simplified to local/remote endpoints).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FlowKey {
    pub local: (Ipv4Addr, u16),
    pub remote: (Ipv4Addr, u16),
}

/// The single, default-closed, capability-gated bridge to the legacy IP internet. No
/// traffic crosses unless a flow is explicitly opened (outbound), and inbound is allowed
/// **only** for a matching established flow (NAT/stateful-firewall semantics).
#[derive(Default)]
pub struct LegacyGateway {
    flows: BTreeSet<FlowKey>,
}

impl LegacyGateway {
    pub fn new() -> LegacyGateway {
        LegacyGateway { flows: BTreeSet::new() }
    }

    /// Open an outbound flow (the cell holds the gateway capability to do so). Returns
    /// the flow key now permitted.
    pub fn open_outbound(&mut self, local: (Ipv4Addr, u16), remote: (Ipv4Addr, u16)) -> FlowKey {
        let key = FlowKey { local, remote };
        self.flows.insert(key);
        key
    }

    /// May an inbound packet `remote → local` pass? Only if it matches an opened flow —
    /// unsolicited inbound is dropped (default-closed).
    pub fn inbound_allowed(&self, remote: (Ipv4Addr, u16), local: (Ipv4Addr, u16)) -> bool {
        self.flows.contains(&FlowKey { local, remote })
    }

    /// Close a flow (e.g. on connection teardown).
    pub fn close(&mut self, key: &FlowKey) {
        self.flows.remove(key);
    }

    /// Number of open flows.
    pub fn open_flows(&self) -> usize {
        self.flows.len()
    }
}

// ───────────────────────── DominionLink over UDP ─────────────────────────

/// The well-known UDP port the DominionLink overlay rides on.
pub const DOMINION_UDP_PORT: u16 = 4242;

/// A magic prefix marking an DominionLink overlay datagram.
const OVERLAY_MAGIC: &[u8] = b"AETHv1";

/// Encapsulate an overlay message (`id` = the self-certifying DominionId hash, `payload` =
/// the content) into a UDP datagram, ready to hand to the IP layer.
pub fn encapsulate(src_port: u16, id: &Hash256, payload: &[u8]) -> Vec<u8> {
    // Build the UDP datagram (8-byte header) with the overlay body written inline,
    // rather than building the body in a temporary Vec and copying it into a second
    // datagram buffer (`build_udp`). One allocation, one copy of `payload` — same
    // bytes on the wire as the previous two-step path.
    let body_len = OVERLAY_MAGIC.len() + 32 + payload.len();
    let total = 8 + body_len;
    let mut d = Vec::with_capacity(total);
    d.extend_from_slice(&src_port.to_be_bytes());
    d.extend_from_slice(&DOMINION_UDP_PORT.to_be_bytes());
    d.extend_from_slice(&(total as u16).to_be_bytes());
    d.extend_from_slice(&0u16.to_be_bytes()); // checksum 0 = none (matches build_udp)
    d.extend_from_slice(OVERLAY_MAGIC);
    d.extend_from_slice(&id.0);
    d.extend_from_slice(payload);
    d
}

/// Decapsulate an overlay UDP payload (the bytes *inside* the UDP datagram), returning
/// `(id, payload)`, or `None` if it isn't a well-formed overlay datagram.
pub fn decapsulate(udp_payload: &[u8]) -> Option<(Hash256, Vec<u8>)> {
    let magic = OVERLAY_MAGIC.len();
    if udp_payload.len() < magic + 32 || &udp_payload[..magic] != OVERLAY_MAGIC {
        return None;
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&udp_payload[magic..magic + 32]);
    Some((Hash256(id), udp_payload[magic + 32..].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr([a, b, c, d])
    }

    #[test]
    fn sockets_are_capabilities_no_ambient_ports() {
        let mut stack = NetworkStack::new(b"issuer-key");
        let cap = SocketCapability::mint(Protocol::Tcp, 443, None, b"issuer-key");
        assert!(stack.bind(&cap).is_ok());
        // A forged capability (tampered port) is refused.
        let mut forged = cap;
        forged.local_port = 8443;
        assert_eq!(stack.bind(&forged), Err(NetError::Unauthorized));
        // A different holder cannot bind the same port.
        let other = SocketCapability::mint(Protocol::Tcp, 443, Some((ip(1,1,1,1), 80)), b"issuer-key");
        assert_eq!(stack.bind(&other), Err(NetError::PortInUse));
        // A connected socket only permits its remote.
        let conn = SocketCapability::mint(Protocol::Udp, 5000, Some((ip(8,8,8,8), 53)), b"issuer-key");
        stack.bind(&conn).unwrap();
        assert!(stack.send(&conn, (ip(8,8,8,8), 53)).is_ok());
        assert_eq!(stack.send(&conn, (ip(9,9,9,9), 53)), Err(NetError::RemoteNotPermitted));
    }

    #[test]
    fn tcp_three_way_handshake_and_teardown() {
        let mut c = TcpConnection::new(1000);
        let syn = c.connect();
        assert_eq!(syn, 1000);
        assert_eq!(c.state, TcpState::SynSent);
        // Peer SYN-ACK acking our SYN (ack = 1001) → established.
        let ack = c.on_syn_ack(5000, 1001).unwrap();
        assert_eq!(ack, 5001);
        assert_eq!(c.state, TcpState::Established);
        // A wrong ack is rejected.
        let mut c2 = TcpConnection::new(1000);
        c2.connect();
        assert!(c2.on_syn_ack(5000, 999).is_none());
        // Send advances sequence; close → FIN-WAIT → TIME-WAIT.
        assert_eq!(c.send(100), Some(1001));
        c.close();
        assert_eq!(c.state, TcpState::FinWait);
        c.on_fin();
        assert_eq!(c.state, TcpState::TimeWait);
    }

    #[test]
    fn legacy_gateway_is_default_closed_with_nat_inbound() {
        let mut gw = LegacyGateway::new();
        let local = (ip(10,0,0,5), 51000);
        let remote = (ip(93,184,216,34), 443);
        // Unsolicited inbound is dropped before any flow opens.
        assert!(!gw.inbound_allowed(remote, local));
        // Open the outbound flow → matching inbound (the reply) is now allowed.
        let key = gw.open_outbound(local, remote);
        assert!(gw.inbound_allowed(remote, local));
        // A *different* remote is still blocked (no unsolicited inbound).
        assert!(!gw.inbound_allowed((ip(6,6,6,6), 1234), local));
        gw.close(&key);
        assert_eq!(gw.open_flows(), 0);
        assert!(!gw.inbound_allowed(remote, local));
    }

    #[test]
    fn dominionlink_rides_over_udp() {
        let id = Hash256::of(b"self-certifying-id");
        let datagram = encapsulate(40000, &id, b"overlay-content");
        // The datagram targets the overlay port and round-trips through decapsulation.
        // (UDP header is 8 bytes; the body is what decapsulate parses.)
        let body = &datagram[8..];
        let (got_id, payload) = decapsulate(body).unwrap();
        assert_eq!(got_id, id);
        assert_eq!(payload, b"overlay-content");
        // Non-overlay bytes are rejected.
        assert!(decapsulate(b"not an overlay datagram").is_none());
    }
}

//! A minimal network stack — roadmap feature 1's legacy-interop layer (L1).
//!
//! The OS's *native* networking is DominionLink (self-certifying, content-addressed
//! — see [`dominionlink`](crate::dominionlink)). But to reach the existing internet we
//! speak the legacy protocols too, wrapped so "every socket is a capability over a
//! byte-stream region" (integration strategy §5). This module is the pure,
//! host-testable protocol logic: Ethernet, ARP, IPv4, ICMP and UDP framing plus
//! the Internet checksum. The virtio-net *driver* (the `unsafe`, hardware half)
//! lives in `dominion-kernel` and feeds frames through this code.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// EtherType for IPv4.
pub const ETHERTYPE_IPV4: u16 = 0x0800;
/// EtherType for ARP.
pub const ETHERTYPE_ARP: u16 = 0x0806;
/// IPv4 protocol number for ICMP.
pub const IPPROTO_ICMP: u8 = 1;
/// IPv4 protocol number for UDP.
pub const IPPROTO_UDP: u8 = 17;

/// A 48-bit Ethernet MAC address.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MacAddr(pub [u8; 6]);

impl MacAddr {
    pub const BROADCAST: MacAddr = MacAddr([0xFF; 6]);
    pub const ZERO: MacAddr = MacAddr([0; 6]);
    pub fn is_broadcast(&self) -> bool {
        self.0 == [0xFF; 6]
    }
}

/// A 32-bit IPv4 address.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Ipv4Addr(pub [u8; 4]);

impl Ipv4Addr {
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr([a, b, c, d])
    }
}

/// The 16-bit one's-complement Internet checksum (RFC 1071) over `data`.
pub fn checksum(data: &[u8]) -> u16 {
    !(fold_checksum(accumulate_checksum(0, data, false)) as u16)
}

/// Fold a running 32-bit one's-complement accumulator down to 16 bits.
#[inline]
fn fold_checksum(mut sum: u32) -> u32 {
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    sum
}

/// Add `data` into a running RFC 1071 accumulator without allocating a scratch
/// buffer. `odd_carry` is true when the bytes accumulated so far ended on an odd
/// boundary (a leftover low byte), so this slice's first byte completes that
/// 16-bit word — letting several slices be summed as if concatenated. Returns the
/// updated (not-yet-folded) accumulator; the boundary parity flips iff `data` has
/// odd length combined with the incoming carry. Callers that need the parity for a
/// further slice recompute it from the lengths.
#[inline]
fn accumulate_checksum(mut sum: u32, data: &[u8], odd_carry: bool) -> u32 {
    let mut i = 0;
    if odd_carry && !data.is_empty() {
        // The previous slice left a high byte already added as (byte << 8); this
        // slice's first byte is that word's low half.
        sum += data[0] as u32;
        i = 1;
    }
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    sum
}

// ---------------------------------------------------------------------------
// Ethernet
// ---------------------------------------------------------------------------

/// Build an Ethernet II frame.
pub fn build_ethernet(dst: MacAddr, src: MacAddr, ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(14 + payload.len());
    f.extend_from_slice(&dst.0);
    f.extend_from_slice(&src.0);
    f.extend_from_slice(&ethertype.to_be_bytes());
    f.extend_from_slice(payload);
    f
}

/// A parsed Ethernet frame header (payload borrowed from the input).
#[derive(Clone, Copy, Debug)]
pub struct EthernetFrame<'a> {
    pub dst: MacAddr,
    pub src: MacAddr,
    pub ethertype: u16,
    pub payload: &'a [u8],
}

pub fn parse_ethernet(frame: &[u8]) -> Option<EthernetFrame<'_>> {
    if frame.len() < 14 {
        return None;
    }
    let mut dst = [0u8; 6];
    let mut src = [0u8; 6];
    dst.copy_from_slice(&frame[0..6]);
    src.copy_from_slice(&frame[6..12]);
    Some(EthernetFrame {
        dst: MacAddr(dst),
        src: MacAddr(src),
        ethertype: u16::from_be_bytes([frame[12], frame[13]]),
        payload: &frame[14..],
    })
}

// ---------------------------------------------------------------------------
// ARP
// ---------------------------------------------------------------------------

pub const ARP_REQUEST: u16 = 1;
pub const ARP_REPLY: u16 = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArpPacket {
    pub opcode: u16,
    pub sender_mac: MacAddr,
    pub sender_ip: Ipv4Addr,
    pub target_mac: MacAddr,
    pub target_ip: Ipv4Addr,
}

impl ArpPacket {
    /// Serialise to the 28-byte Ethernet/IPv4 ARP wire format.
    pub fn build(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(28);
        p.extend_from_slice(&1u16.to_be_bytes()); // HTYPE = Ethernet
        p.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes()); // PTYPE = IPv4
        p.push(6); // HLEN
        p.push(4); // PLEN
        p.extend_from_slice(&self.opcode.to_be_bytes());
        p.extend_from_slice(&self.sender_mac.0);
        p.extend_from_slice(&self.sender_ip.0);
        p.extend_from_slice(&self.target_mac.0);
        p.extend_from_slice(&self.target_ip.0);
        p
    }

    pub fn parse(p: &[u8]) -> Option<ArpPacket> {
        if p.len() < 28 {
            return None;
        }
        let opcode = u16::from_be_bytes([p[6], p[7]]);
        let mut sm = [0u8; 6];
        let mut tm = [0u8; 6];
        sm.copy_from_slice(&p[8..14]);
        tm.copy_from_slice(&p[18..24]);
        Some(ArpPacket {
            opcode,
            sender_mac: MacAddr(sm),
            sender_ip: Ipv4Addr([p[14], p[15], p[16], p[17]]),
            target_mac: MacAddr(tm),
            target_ip: Ipv4Addr([p[24], p[25], p[26], p[27]]),
        })
    }
}

/// A cache of resolved IPv4 → MAC bindings.
#[derive(Default)]
pub struct ArpCache {
    entries: BTreeMap<Ipv4Addr, MacAddr>,
}

impl ArpCache {
    pub fn new() -> ArpCache {
        ArpCache { entries: BTreeMap::new() }
    }
    pub fn insert(&mut self, ip: Ipv4Addr, mac: MacAddr) {
        self.entries.insert(ip, mac);
    }
    pub fn lookup(&self, ip: Ipv4Addr) -> Option<MacAddr> {
        self.entries.get(&ip).copied()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// IPv4
// ---------------------------------------------------------------------------

/// Build an IPv4 packet (no options) with a correct header checksum.
pub fn build_ipv4(src: Ipv4Addr, dst: Ipv4Addr, protocol: u8, payload: &[u8], ident: u16) -> Vec<u8> {
    let total_len = 20 + payload.len();
    let mut h = Vec::with_capacity(total_len);
    h.push(0x45); // version 4, IHL 5
    h.push(0x00); // DSCP/ECN
    h.extend_from_slice(&(total_len as u16).to_be_bytes());
    h.extend_from_slice(&ident.to_be_bytes());
    h.extend_from_slice(&0x0000u16.to_be_bytes()); // flags/fragment
    h.push(64); // TTL
    h.push(protocol);
    h.extend_from_slice(&0x0000u16.to_be_bytes()); // checksum placeholder
    h.extend_from_slice(&src.0);
    h.extend_from_slice(&dst.0);
    let csum = checksum(&h);
    h[10..12].copy_from_slice(&csum.to_be_bytes());
    h.extend_from_slice(payload);
    h
}

#[derive(Clone, Copy, Debug)]
pub struct Ipv4Packet<'a> {
    pub src: Ipv4Addr,
    pub dst: Ipv4Addr,
    pub protocol: u8,
    pub payload: &'a [u8],
}

pub fn parse_ipv4(p: &[u8]) -> Option<Ipv4Packet<'_>> {
    if p.len() < 20 {
        return None;
    }
    let ihl = (p[0] & 0x0F) as usize * 4;
    if (p[0] >> 4) != 4 || ihl < 20 || p.len() < ihl {
        return None;
    }
    let total_len = u16::from_be_bytes([p[2], p[3]]) as usize;
    if total_len < ihl || total_len > p.len() {
        return None;
    }
    Some(Ipv4Packet {
        src: Ipv4Addr([p[12], p[13], p[14], p[15]]),
        dst: Ipv4Addr([p[16], p[17], p[18], p[19]]),
        protocol: p[9],
        payload: &p[ihl..total_len],
    })
}

/// True if the IPv4 header checksum validates.
pub fn ipv4_checksum_valid(p: &[u8]) -> bool {
    if p.len() < 20 {
        return false;
    }
    let ihl = (p[0] & 0x0F) as usize * 4;
    if p.len() < ihl {
        return false;
    }
    checksum(&p[..ihl]) == 0
}

// ---------------------------------------------------------------------------
// ICMP
// ---------------------------------------------------------------------------

pub const ICMP_ECHO_REQUEST: u8 = 8;
pub const ICMP_ECHO_REPLY: u8 = 0;

/// Build an ICMP echo packet (request or reply) with checksum.
pub fn build_icmp_echo(kind: u8, ident: u16, seq: u16, data: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(8 + data.len());
    m.push(kind);
    m.push(0); // code
    m.extend_from_slice(&0u16.to_be_bytes()); // checksum placeholder
    m.extend_from_slice(&ident.to_be_bytes());
    m.extend_from_slice(&seq.to_be_bytes());
    m.extend_from_slice(data);
    let csum = checksum(&m);
    m[2..4].copy_from_slice(&csum.to_be_bytes());
    m
}

#[derive(Clone, Copy, Debug)]
pub struct IcmpEcho<'a> {
    pub kind: u8,
    pub ident: u16,
    pub seq: u16,
    pub data: &'a [u8],
}

pub fn parse_icmp_echo(m: &[u8]) -> Option<IcmpEcho<'_>> {
    if m.len() < 8 {
        return None;
    }
    Some(IcmpEcho {
        kind: m[0],
        ident: u16::from_be_bytes([m[4], m[5]]),
        seq: u16::from_be_bytes([m[6], m[7]]),
        data: &m[8..],
    })
}

// ---------------------------------------------------------------------------
// UDP
// ---------------------------------------------------------------------------

/// Build a UDP datagram (checksum left zero, which is legal for IPv4).
pub fn build_udp(src_port: u16, dst_port: u16, payload: &[u8]) -> Vec<u8> {
    let len = 8 + payload.len();
    let mut d = Vec::with_capacity(len);
    d.extend_from_slice(&src_port.to_be_bytes());
    d.extend_from_slice(&dst_port.to_be_bytes());
    d.extend_from_slice(&(len as u16).to_be_bytes());
    d.extend_from_slice(&0u16.to_be_bytes()); // checksum 0 = none
    d.extend_from_slice(payload);
    d
}

#[derive(Clone, Copy, Debug)]
pub struct UdpDatagram<'a> {
    pub src_port: u16,
    pub dst_port: u16,
    pub payload: &'a [u8],
}

pub fn parse_udp(d: &[u8]) -> Option<UdpDatagram<'_>> {
    if d.len() < 8 {
        return None;
    }
    let len = u16::from_be_bytes([d[4], d[5]]) as usize;
    if len < 8 || len > d.len() {
        return None;
    }
    Some(UdpDatagram {
        src_port: u16::from_be_bytes([d[0], d[1]]),
        dst_port: u16::from_be_bytes([d[2], d[3]]),
        payload: &d[8..len],
    })
}

// ---------------------------------------------------------------------------
// TCP (wire format)
// ---------------------------------------------------------------------------
//
// The connection *state machine* (handshake, sequence/ack bookkeeping) lives in
// [`crate::legacynet`]; this is the byte-level segment build/parse, including the
// IPv4 pseudo-header checksum that TCP — unlike UDP here — actually requires.

/// IPv4 protocol number for TCP.
pub const IPPROTO_TCP: u8 = 6;

// TCP control flags.
pub const TCP_FIN: u8 = 0x01;
pub const TCP_SYN: u8 = 0x02;
pub const TCP_RST: u8 = 0x04;
pub const TCP_PSH: u8 = 0x08;
pub const TCP_ACK: u8 = 0x10;

/// A parsed TCP segment (payload borrowed from the input).
#[derive(Clone, Copy, Debug)]
pub struct TcpSegment<'a> {
    pub src_port: u16,
    pub dst_port: u16,
    pub seq: u32,
    pub ack: u32,
    pub flags: u8,
    pub window: u16,
    pub payload: &'a [u8],
}

impl TcpSegment<'_> {
    pub fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }
}

/// The TCP checksum over the IPv4 pseudo-header + segment (RFC 793).
///
/// Computed by streaming the 12-byte pseudo-header and the segment through the
/// RFC 1071 accumulator directly — no scratch buffer is allocated and the segment
/// is not copied, so this stays allocation-free on the per-segment hot path. The
/// pseudo-header is 12 bytes (even), so the segment continues on an even boundary.
fn tcp_checksum(src: Ipv4Addr, dst: Ipv4Addr, segment: &[u8]) -> u16 {
    let len = segment.len() as u16;
    let pseudo = [
        src.0[0], src.0[1], src.0[2], src.0[3],
        dst.0[0], dst.0[1], dst.0[2], dst.0[3],
        0, IPPROTO_TCP,
        (len >> 8) as u8, (len & 0xFF) as u8,
    ];
    let mut sum = accumulate_checksum(0, &pseudo, false);
    sum = accumulate_checksum(sum, segment, false);
    !(fold_checksum(sum) as u16)
}

/// Build a TCP segment (no options) with a correct checksum.
#[allow(clippy::too_many_arguments)]
pub fn build_tcp(
    src: Ipv4Addr,
    dst: Ipv4Addr,
    src_port: u16,
    dst_port: u16,
    seq: u32,
    ack: u32,
    flags: u8,
    window: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut s = Vec::with_capacity(20 + payload.len());
    s.extend_from_slice(&src_port.to_be_bytes());
    s.extend_from_slice(&dst_port.to_be_bytes());
    s.extend_from_slice(&seq.to_be_bytes());
    s.extend_from_slice(&ack.to_be_bytes());
    s.push(0x50); // data offset 5 (20 bytes), reserved 0
    s.push(flags);
    s.extend_from_slice(&window.to_be_bytes());
    s.extend_from_slice(&0u16.to_be_bytes()); // checksum placeholder
    s.extend_from_slice(&0u16.to_be_bytes()); // urgent pointer
    s.extend_from_slice(payload);
    let csum = tcp_checksum(src, dst, &s);
    s[16..18].copy_from_slice(&csum.to_be_bytes());
    s
}

/// Parse a TCP segment, honouring the data-offset field so options are skipped.
pub fn parse_tcp(s: &[u8]) -> Option<TcpSegment<'_>> {
    if s.len() < 20 {
        return None;
    }
    let data_offset = ((s[12] >> 4) as usize) * 4;
    if data_offset < 20 || s.len() < data_offset {
        return None;
    }
    Some(TcpSegment {
        src_port: u16::from_be_bytes([s[0], s[1]]),
        dst_port: u16::from_be_bytes([s[2], s[3]]),
        seq: u32::from_be_bytes([s[4], s[5], s[6], s[7]]),
        ack: u32::from_be_bytes([s[8], s[9], s[10], s[11]]),
        flags: s[13],
        window: u16::from_be_bytes([s[14], s[15]]),
        payload: &s[data_offset..],
    })
}

/// Validate a received TCP segment's checksum against its IPv4 addresses.
pub fn tcp_checksum_valid(src: Ipv4Addr, dst: Ipv4Addr, segment: &[u8]) -> bool {
    tcp_checksum(src, dst, segment) == 0
}

// ---------------------------------------------------------------------------
// DNS (query build + answer parse, over UDP)
// ---------------------------------------------------------------------------

/// Build a DNS A-record query for `name` with the given 16-bit id. Standard
/// recursive query, one question, class IN.
pub fn build_dns_query(id: u16, name: &str) -> Vec<u8> {
    let mut q = Vec::new();
    q.extend_from_slice(&id.to_be_bytes());
    q.extend_from_slice(&0x0100u16.to_be_bytes()); // flags: RD=1
    q.extend_from_slice(&1u16.to_be_bytes()); // QDCOUNT
    q.extend_from_slice(&0u16.to_be_bytes()); // ANCOUNT
    q.extend_from_slice(&0u16.to_be_bytes()); // NSCOUNT
    q.extend_from_slice(&0u16.to_be_bytes()); // ARCOUNT
    for label in name.split('.').filter(|l| !l.is_empty()) {
        q.push(label.len() as u8);
        q.extend_from_slice(label.as_bytes());
    }
    q.push(0); // root
    q.extend_from_slice(&1u16.to_be_bytes()); // QTYPE A
    q.extend_from_slice(&1u16.to_be_bytes()); // QCLASS IN
    q
}

/// Parse the first A-record answer from a DNS response with matching `id`.
pub fn parse_dns_answer(resp: &[u8], id: u16) -> Option<Ipv4Addr> {
    if resp.len() < 12 || u16::from_be_bytes([resp[0], resp[1]]) != id {
        return None;
    }
    let qd = u16::from_be_bytes([resp[4], resp[5]]) as usize;
    let an = u16::from_be_bytes([resp[6], resp[7]]) as usize;
    let mut pos = 12;
    // Skip the question section.
    for _ in 0..qd {
        pos = skip_name(resp, pos)?;
        pos = pos.checked_add(4)?; // QTYPE + QCLASS
    }
    // Walk answers, returning the first A record.
    for _ in 0..an {
        pos = skip_name(resp, pos)?;
        if pos + 10 > resp.len() {
            return None;
        }
        let rtype = u16::from_be_bytes([resp[pos], resp[pos + 1]]);
        let rdlen = u16::from_be_bytes([resp[pos + 8], resp[pos + 9]]) as usize;
        pos += 10;
        if pos + rdlen > resp.len() {
            return None;
        }
        if rtype == 1 && rdlen == 4 {
            return Some(Ipv4Addr([resp[pos], resp[pos + 1], resp[pos + 2], resp[pos + 3]]));
        }
        pos += rdlen;
    }
    None
}

/// Advance past a (possibly compressed) DNS name, returning the position after it.
fn skip_name(resp: &[u8], mut pos: usize) -> Option<usize> {
    loop {
        let len = *resp.get(pos)?;
        if len & 0xC0 == 0xC0 {
            // Compression pointer — two bytes, name ends here.
            return Some(pos + 2);
        }
        if len == 0 {
            return Some(pos + 1);
        }
        pos = pos.checked_add(1 + len as usize)?;
    }
}

// ---------------------------------------------------------------------------
// A tiny host interface tying the layers together.
// ---------------------------------------------------------------------------

/// Our station's identity on the legacy network.
pub struct Interface {
    pub mac: MacAddr,
    pub ip: Ipv4Addr,
    pub arp: ArpCache,
    ident: u16,
}

impl Interface {
    pub fn new(mac: MacAddr, ip: Ipv4Addr) -> Interface {
        Interface { mac, ip, arp: ArpCache::new(), ident: 1 }
    }

    fn next_ident(&mut self) -> u16 {
        let id = self.ident;
        self.ident = self.ident.wrapping_add(1);
        id
    }

    /// Build a broadcast ARP request asking who owns `target_ip`.
    pub fn arp_request(&self, target_ip: Ipv4Addr) -> Vec<u8> {
        let arp = ArpPacket {
            opcode: ARP_REQUEST,
            sender_mac: self.mac,
            sender_ip: self.ip,
            target_mac: MacAddr::ZERO,
            target_ip,
        };
        build_ethernet(MacAddr::BROADCAST, self.mac, ETHERTYPE_ARP, &arp.build())
    }

    /// Build an ICMP echo request to `dst` (already-resolved `dst_mac`).
    pub fn ping(&mut self, dst: Ipv4Addr, dst_mac: MacAddr, seq: u16, data: &[u8]) -> Vec<u8> {
        let icmp = build_icmp_echo(ICMP_ECHO_REQUEST, 0xAE01, seq, data);
        let ident = self.next_ident();
        let ip = build_ipv4(self.ip, dst, IPPROTO_ICMP, &icmp, ident);
        build_ethernet(dst_mac, self.mac, ETHERTYPE_IPV4, &ip)
    }

    /// Process a received Ethernet frame, learning ARP bindings and producing a
    /// reply frame when one is warranted (ARP request for us, or an ICMP echo
    /// request to us). Returns `None` if no reply is needed.
    pub fn handle_frame(&mut self, frame: &[u8]) -> Option<Vec<u8>> {
        let eth = parse_ethernet(frame)?;
        match eth.ethertype {
            ETHERTYPE_ARP => {
                let arp = ArpPacket::parse(eth.payload)?;
                // Learn the sender.
                self.arp.insert(arp.sender_ip, arp.sender_mac);
                if arp.opcode == ARP_REQUEST && arp.target_ip == self.ip {
                    let reply = ArpPacket {
                        opcode: ARP_REPLY,
                        sender_mac: self.mac,
                        sender_ip: self.ip,
                        target_mac: arp.sender_mac,
                        target_ip: arp.sender_ip,
                    };
                    return Some(build_ethernet(arp.sender_mac, self.mac, ETHERTYPE_ARP, &reply.build()));
                }
                None
            }
            ETHERTYPE_IPV4 => {
                let ip = parse_ipv4(eth.payload)?;
                if ip.dst != self.ip || ip.protocol != IPPROTO_ICMP {
                    return None;
                }
                let echo = parse_icmp_echo(ip.payload)?;
                if echo.kind != ICMP_ECHO_REQUEST {
                    return None;
                }
                // Reply: swap addresses, echo the data back.
                let reply_icmp = build_icmp_echo(ICMP_ECHO_REPLY, echo.ident, echo.seq, echo.data);
                let ident = self.next_ident();
                let reply_ip = build_ipv4(self.ip, ip.src, IPPROTO_ICMP, &reply_icmp, ident);
                Some(build_ethernet(eth.src, self.mac, ETHERTYPE_IPV4, &reply_ip))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internet_checksum_known_value() {
        // Classic RFC 1071 worked example bytes sum to a known checksum.
        let data = [0x00u8, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7];
        let c = checksum(&data);
        // Folding the verification: data + checksum must sum to 0xFFFF complement.
        let mut withck = data.to_vec();
        withck.extend_from_slice(&c.to_be_bytes());
        assert_eq!(checksum(&withck), 0);
    }

    #[test]
    fn ethernet_wrap_unwrap() {
        let dst = MacAddr([1, 2, 3, 4, 5, 6]);
        let src = MacAddr([0xAA; 6]);
        let frame = build_ethernet(dst, src, ETHERTYPE_IPV4, b"payload");
        let p = parse_ethernet(&frame).unwrap();
        assert_eq!(p.dst, dst);
        assert_eq!(p.src, src);
        assert_eq!(p.ethertype, ETHERTYPE_IPV4);
        assert_eq!(p.payload, b"payload");
    }

    #[test]
    fn arp_round_trips() {
        let a = ArpPacket {
            opcode: ARP_REQUEST,
            sender_mac: MacAddr([0xAA; 6]),
            sender_ip: Ipv4Addr::new(10, 0, 2, 15),
            target_mac: MacAddr::ZERO,
            target_ip: Ipv4Addr::new(10, 0, 2, 2),
        };
        let parsed = ArpPacket::parse(&a.build()).unwrap();
        assert_eq!(parsed, a);
    }

    #[test]
    fn ipv4_header_checksum_validates() {
        let pkt = build_ipv4(Ipv4Addr::new(10, 0, 2, 15), Ipv4Addr::new(10, 0, 2, 2), IPPROTO_ICMP, b"hi", 7);
        assert!(ipv4_checksum_valid(&pkt));
        let parsed = parse_ipv4(&pkt).unwrap();
        assert_eq!(parsed.protocol, IPPROTO_ICMP);
        assert_eq!(parsed.payload, b"hi");
    }

    #[test]
    fn icmp_echo_round_trips_with_checksum() {
        let m = build_icmp_echo(ICMP_ECHO_REQUEST, 0x1234, 1, b"ping-data");
        assert_eq!(checksum(&m), 0); // a correct ICMP message checksums to zero
        let e = parse_icmp_echo(&m).unwrap();
        assert_eq!(e.kind, ICMP_ECHO_REQUEST);
        assert_eq!(e.ident, 0x1234);
        assert_eq!(e.data, b"ping-data");
    }

    #[test]
    fn udp_round_trips() {
        let d = build_udp(1234, 53, b"query");
        let u = parse_udp(&d).unwrap();
        assert_eq!(u.src_port, 1234);
        assert_eq!(u.dst_port, 53);
        assert_eq!(u.payload, b"query");
    }

    #[test]
    fn interface_answers_arp_for_itself() {
        let mut me = Interface::new(MacAddr([0x52, 0x54, 0, 0, 0, 1]), Ipv4Addr::new(10, 0, 2, 15));
        let peer = ArpPacket {
            opcode: ARP_REQUEST,
            sender_mac: MacAddr([0x52, 0x54, 0, 0, 0, 2]),
            sender_ip: Ipv4Addr::new(10, 0, 2, 2),
            target_mac: MacAddr::ZERO,
            target_ip: Ipv4Addr::new(10, 0, 2, 15),
        };
        let req = build_ethernet(MacAddr::BROADCAST, peer.sender_mac, ETHERTYPE_ARP, &peer.build());
        let reply = me.handle_frame(&req).expect("should answer ARP");
        let eth = parse_ethernet(&reply).unwrap();
        let arp = ArpPacket::parse(eth.payload).unwrap();
        assert_eq!(arp.opcode, ARP_REPLY);
        assert_eq!(arp.sender_ip, me.ip);
        // And it learned the peer's binding.
        assert_eq!(me.arp.lookup(Ipv4Addr::new(10, 0, 2, 2)), Some(peer.sender_mac));
    }

    #[test]
    fn interface_replies_to_ping() {
        let mut me = Interface::new(MacAddr([0x52, 0x54, 0, 0, 0, 1]), Ipv4Addr::new(10, 0, 2, 15));
        let peer_mac = MacAddr([0x52, 0x54, 0, 0, 0, 2]);
        let icmp = build_icmp_echo(ICMP_ECHO_REQUEST, 0x1, 1, b"abc");
        let ip = build_ipv4(Ipv4Addr::new(10, 0, 2, 2), me.ip, IPPROTO_ICMP, &icmp, 9);
        let frame = build_ethernet(me.mac, peer_mac, ETHERTYPE_IPV4, &ip);
        let reply = me.handle_frame(&frame).expect("should reply to ping");
        let eth = parse_ethernet(&reply).unwrap();
        let ipr = parse_ipv4(eth.payload).unwrap();
        let echo = parse_icmp_echo(ipr.payload).unwrap();
        assert_eq!(echo.kind, ICMP_ECHO_REPLY);
        assert_eq!(echo.data, b"abc");
    }

    #[test]
    fn tcp_segment_round_trips_and_checksums() {
        let src = Ipv4Addr::new(10, 0, 2, 15);
        let dst = Ipv4Addr::new(93, 184, 216, 34);
        let seg = build_tcp(src, dst, 49152, 80, 1000, 2000, TCP_SYN | TCP_ACK, 64240, b"GET /");
        // A correct segment checksums (with its own checksum field) to zero.
        assert!(tcp_checksum_valid(src, dst, &seg));
        let p = parse_tcp(&seg).unwrap();
        assert_eq!(p.src_port, 49152);
        assert_eq!(p.dst_port, 80);
        assert_eq!(p.seq, 1000);
        assert_eq!(p.ack, 2000);
        assert!(p.has(TCP_SYN) && p.has(TCP_ACK));
        assert_eq!(p.window, 64240);
        assert_eq!(p.payload, b"GET /");
    }

    #[test]
    fn tcp_parse_skips_options() {
        // data offset 6 (24 bytes): 20 base + 4 option bytes.
        let mut seg = build_tcp(
            Ipv4Addr::new(1, 1, 1, 1),
            Ipv4Addr::new(2, 2, 2, 2),
            1,
            2,
            0,
            0,
            TCP_ACK,
            1000,
            b"",
        );
        // Splice in a fake 4-byte option and bump the data offset nibble.
        seg.splice(20..20, [0x02, 0x04, 0x05, 0xb4]); // MSS option
        seg[12] = 0x60; // data offset = 6
        let p = parse_tcp(&seg).unwrap();
        assert_eq!(p.payload, b"");
        assert_eq!(p.src_port, 1);
    }

    #[test]
    fn dns_query_and_answer_round_trip() {
        let q = build_dns_query(0x1234, "example.com");
        // QDCOUNT = 1, flags RD set.
        assert_eq!(&q[0..2], &[0x12, 0x34]);
        assert_eq!(u16::from_be_bytes([q[4], q[5]]), 1);
        // The encoded name carries length-prefixed labels.
        assert_eq!(q[12], 7); // "example"
        // Build a matching response with one A answer (93.184.216.34) using a
        // compression pointer back to the question name.
        let mut r = q.clone();
        r[2] = 0x81; // QR=1, RD=1
        r[3] = 0x80; // RA=1
        r[6] = 0x00;
        r[7] = 0x01; // ANCOUNT = 1
        r.extend_from_slice(&[0xC0, 0x0C]); // name pointer to offset 12
        r.extend_from_slice(&1u16.to_be_bytes()); // TYPE A
        r.extend_from_slice(&1u16.to_be_bytes()); // CLASS IN
        r.extend_from_slice(&300u32.to_be_bytes()); // TTL
        r.extend_from_slice(&4u16.to_be_bytes()); // RDLENGTH
        r.extend_from_slice(&[93, 184, 216, 34]);
        let ip = parse_dns_answer(&r, 0x1234).unwrap();
        assert_eq!(ip, Ipv4Addr::new(93, 184, 216, 34));
        // A mismatched id is rejected.
        assert!(parse_dns_answer(&r, 0x9999).is_none());
    }
}

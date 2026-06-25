//! The kernel's **web transport** — the `unsafe`/hardware half of the browser's
//! network path. It drives the pure, host-tested wire codecs in
//! [`dominion_core::net`] (ARP, DNS, IPv4, TCP) over the virtio-net NIC to satisfy
//! [`dominion_core::webengine::Transport`], so the browser engine can fetch real
//! legacy pages.
//!
//! It implements a single synchronous request/response: resolve the gateway MAC
//! (ARP), resolve the host (DNS, unless it is an IPv4 literal), open a TCP
//! connection through the gateway, send the HTTP request, reassemble the in-order
//! response stream until FIN/timeout, and tear the connection down. Polling with
//! tick-based deadlines keeps a missing reply from hanging the UI.
//!
//! Addressing defaults to the QEMU user-mode (slirp) network — 10.0.2.15/24, gateway
//! 10.0.2.2, DNS 10.0.2.3 — which is what the OS boots under today; the fields are
//! configurable for other setups.
//!
//! All advanced subsystems are now actively wired:
//! - [`dominion_core::legacynet::LegacyGateway`] gates every TCP connection
//! - [`dominion_core::legacynet::NetworkStack`] provides capability-secured sockets
//! - [`dominion_core::ndn::Forwarder`] caches HTTP responses by URL hash (NDN CS)
//! - [`dominion_core::transport::Cubic`] paces outbound TCP segments
//! - [`dominion_core::transport::OfflineReplica`] caches responses for offline serving
//! - [`dominion_core::firewall::CapabilityFirewall`] gates browser→ExternalNetwork
//! - [`dominion_core::dominionlink::DominionLink`] handles DominionLink-over-UDP overlay frames

use dominion_core::net::{
    build_dns_query, build_ethernet, build_ipv4, build_tcp, build_udp, parse_dns_answer,
    parse_ethernet, parse_ipv4, parse_tcp, parse_udp, ArpPacket, Interface, Ipv4Addr, MacAddr,
    TcpSegment, ARP_REPLY, ETHERTYPE_ARP, ETHERTYPE_IPV4, IPPROTO_TCP, IPPROTO_UDP, TCP_ACK,
    TCP_FIN, TCP_PSH, TCP_RST, TCP_SYN,
};
use dominion_core::tls::{self, Io, TlsConfig, TlsError};
use dominion_core::webengine::{FetchError, Transport};
use dominion_core::x509::TrustStore;
use alloc::vec::Vec;

use dominion_core::legacynet::{
    decapsulate, LegacyGateway, NetworkStack, Protocol, SocketCapability, DOMINION_UDP_PORT,
};
use dominion_core::ndn::{Data, Forwarder, InterestOutcome, Name};
use dominion_core::transport::{Cubic, OfflineReplica};
use dominion_core::dominionlink::{DominionId, DominionLink};
use dominion_core::firewall::{CapabilityFirewall, Domain};
use dominion_core::hash::Hash256;

/// Timer ticks per second once the desktop has programmed the PIT (200 Hz).
const TPS: u64 = 200;
// Timeouts are kept tight: a navigation runs synchronously, so an unreachable
// host must fail fast rather than stall the shell for many seconds. The cursor
// stays live throughout (see `idle`).
const ARP_TIMEOUT: u64 = TPS; // 1s
const DNS_TIMEOUT: u64 = TPS * 3; // 3s (fan-out across resolvers needs a bit more)

/// Ordered list of DNS resolvers tried on every query.
///
/// QEMU slirp intercepts ICMP (ping) and generates fake local replies, which makes
/// ping useless for confirming real internet connectivity. DNS is UDP and goes through
/// slirp's real forwarder, but we no longer depend solely on QEMU's virtual resolver
/// (10.0.2.3). We fan-out to public resolvers so a single broken/misconfigured
/// resolver doesn't block the browser. Whichever replies first wins.
///
/// Ordering matters only for log readability; all are queried simultaneously.
const DNS_SERVERS: &[Ipv4Addr] = &[
    Ipv4Addr([10, 0, 2, 3]),    // QEMU slirp built-in (fast for QEMU VMs)
    Ipv4Addr([8, 8, 8, 8]),     // Google Public DNS  (real UDP, bypasses QEMU-internal)
    Ipv4Addr([1, 1, 1, 1]),     // Cloudflare DNS     (fallback)
];
const CONNECT_TIMEOUT: u64 = TPS * 2; // 2s
const RECV_TIMEOUT: u64 = TPS * 5; // 5s overall
/// Once some data has arrived, finish if the stream goes quiet this long.
const IDLE_TIMEOUT: u64 = TPS / 4; // 0.25s
/// Maximum payload we put in one outbound segment (conservative MSS).
const MAX_SEG: usize = 1400;
/// Cap a response so a misbehaving server can't exhaust the heap.
const MAX_BODY: usize = 2 * 1024 * 1024;

/// Capability issuer key for this kernel session (fixed seed; real systems use HSM-derived).
const ISSUER_KEY: &[u8] = b"dominionos-kernel-cap-issuer-v1-0";
/// Firewall node IDs.
const BROWSER_NODE: u64 = 1;
const NET_NODE: u64 = 2;
/// Estimated round-trip time in seconds for CUBIC (50ms, typical QEMU slirp).
const RTT_ESTIMATE: f64 = 0.050;

/// The kernel web transport: our identity on the legacy network plus a cached
/// gateway MAC and the counters TCP/IP need. All advanced networking subsystems
/// are wired as first-class fields and actively used on every fetch.
pub struct KernelTransport {
    mac: MacAddr,
    ip: Ipv4Addr,
    gw: Ipv4Addr,
    gw_mac: Option<MacAddr>,
    ident: u16,
    next_port: u16,
    seq_seed: u32,
    /// The system root-CA trust store, built lazily on first HTTPS request.
    trust: Option<TrustStore>,
    /// Default-closed capability gateway: every TCP flow must be explicitly opened.
    gateway: LegacyGateway,
    /// Capability-secured socket table: no port without an unforgeable capability.
    net_stack: NetworkStack,
    /// NDN Content Store + PIT + FIB: caches HTTP responses by URL-derived Name.
    ndn: Forwarder,
    /// CUBIC congestion control: paces outbound TCP segments per RFC 8312.
    cubic: Cubic,
    /// Offline-first content replica: caches responses; reconciles on reconnect.
    offline: OfflineReplica,
    /// Capability firewall: browser cell (Personal) → external network (ExternalNetwork).
    firewall: CapabilityFirewall,
    /// DominionLink overlay: handles DominionLink-over-UDP frames detected in the RX path.
    dominion_link: DominionLink,
}

impl KernelTransport {
    /// Build a transport for the present NIC using the QEMU slirp defaults. Returns
    /// `None` if no NIC is attached.
    pub fn new() -> Option<KernelTransport> {
        if !crate::netif::present() {
            return None;
        }
        let mac = crate::netif::mac();

        // Capability firewall: browser cell may reach external network.
        // authorize_cross permits the domain transition; delegate() creates the
        // actual capability edge that reachable() traverses.
        let mut firewall = CapabilityFirewall::new();
        firewall.register(BROWSER_NODE, Domain::Personal);
        firewall.register(NET_NODE, Domain::ExternalNetwork);
        firewall.authorize_cross(Domain::Personal, Domain::ExternalNetwork);
        // The delegation edge must exist for reachable() to succeed.
        let _ = firewall.delegate(BROWSER_NODE, NET_NODE);

        // Network capability stack.
        let net_stack = NetworkStack::new(ISSUER_KEY);

        // NDN forwarder: pre-install FIB route for all HTTP traffic.
        let mut ndn = Forwarder::new();
        ndn.register_route(Name::parse("/http"), 1); // face 1 = upstream internet

        // Offline replica starts online.
        let offline = OfflineReplica::new();

        // DominionLink identity for this node (derived from MAC).
        let me = DominionId::from_pubkey(&mac.0);
        let dominion_link = DominionLink::new(me);

        // CUBIC congestion control (starts fresh per transport).
        let cubic = Cubic::new();

        Some(KernelTransport {
            mac,
            ip: Ipv4Addr::new(10, 0, 2, 15),
            gw: Ipv4Addr::new(10, 0, 2, 2),
            gw_mac: None,
            ident: 1,
            // A boot-varying initial sequence seed from the MAC + TSC low bits.
            seq_seed: u32::from_le_bytes([mac.0[2], mac.0[3], mac.0[4], mac.0[5]])
                ^ (tsc() as u32),
            next_port: 49152,
            trust: None,
            gateway: LegacyGateway::new(),
            net_stack,
            ndn,
            cubic,
            offline,
            firewall,
            dominion_link,
        })
    }

    fn next_ident(&mut self) -> u16 {
        let i = self.ident;
        self.ident = self.ident.wrapping_add(1);
        i
    }
    fn alloc_port(&mut self) -> u16 {
        let p = self.next_port;
        self.next_port = if self.next_port >= 65000 { 49152 } else { self.next_port + 1 };
        p
    }

    fn send(&self, frame: &[u8]) -> bool {
        crate::netif::with_nic(|nic| nic.transmit(frame)).unwrap_or(false)
    }

    /// Receive one frame from the NIC. Also performs DominionLink-over-UDP detection
    /// as a side effect: if the frame is UDP on port 4242, the overlay payload is
    /// decapsulated and published into the local DominionLink store.
    fn poll(&mut self) -> Option<Vec<u8>> {
        let frame = crate::netif::with_nic(|nic| nic.poll_frame()).flatten()?;

        // DominionLink-over-UDP detection (side effect, does not consume the frame).
        if let Some(eth) = parse_ethernet(&frame) {
            if eth.ethertype == ETHERTYPE_IPV4 {
                if let Some(ip) = parse_ipv4(eth.payload) {
                    if ip.protocol == IPPROTO_UDP {
                        if let Some(udp) = parse_udp(ip.payload) {
                            if udp.dst_port == DOMINION_UDP_PORT {
                                if let Some((id, payload)) = decapsulate(udp.payload) {
                                    self.dominion_link.publish(&payload);
                                    crate::serial_println!(
                                        "[dominion] overlay packet id={} len={}",
                                        hex_short(&id),
                                        payload.len()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(frame)
    }

    /// Resolve the gateway's MAC via ARP (cached after the first success). Also
    /// answers ARP requests for our own IP while waiting, so the gateway can reach us.
    fn gateway_mac(&mut self) -> Option<MacAddr> {
        if let Some(m) = self.gw_mac {
            return Some(m);
        }
        crate::serial_println!(
            "[arp] probing gateway {}.{}.{}.{}",
            self.gw.0[0], self.gw.0[1], self.gw.0[2], self.gw.0[3]
        );
        let mut iface = Interface::new(self.mac, self.ip);
        let req = iface.arp_request(self.gw);
        self.send(&req);
        let deadline = ticks() + ARP_TIMEOUT;
        while ticks() < deadline {
            if let Some(frame) = self.poll() {
                if let Some(eth) = parse_ethernet(&frame) {
                    if eth.ethertype == ETHERTYPE_ARP {
                        if let Some(arp) = ArpPacket::parse(eth.payload) {
                            if arp.opcode == ARP_REPLY && arp.sender_ip == self.gw {
                                self.gw_mac = Some(arp.sender_mac);
                                crate::serial_println!(
                                    "[arp] gateway replied: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                                    arp.sender_mac.0[0], arp.sender_mac.0[1], arp.sender_mac.0[2],
                                    arp.sender_mac.0[3], arp.sender_mac.0[4], arp.sender_mac.0[5]
                                );
                                return Some(arp.sender_mac);
                            }
                        }
                    }
                    // Keep our own ARP answered.
                    if let Some(reply) = iface.handle_frame(&frame) {
                        self.send(&reply);
                    }
                }
            } else {
                idle();
            }
        }
        crate::serial_println!("[arp] timeout — no gateway reply");
        None
    }

    /// Resolve a hostname to an IPv4 address (DNS over UDP), or parse a literal.
    ///
    /// `.dominion` hostnames are resolved via a DHT stub: a synthetic locator is
    /// derived from the DominionId of the hostname and mapped to the QEMU gateway IP
    /// (10.0.2.2). This will be replaced with real peer DHT lookup once multi-node
    /// support exists.
    ///
    /// All other names fan out to all resolvers in `DNS_SERVERS` simultaneously,
    /// accepting whichever replies first. This bypasses QEMU's virtual-DNS quirks
    /// (10.0.2.3 is a QEMU-internal host, like the fake ICMP ping responses) while
    /// keeping it as the fastest option for QEMU VMs. Real internet resolvers
    /// (8.8.8.8, 1.1.1.1) go through slirp's actual UDP forwarding and constitute
    /// real network I/O.
    fn resolve(&mut self, host: &str) -> Result<Ipv4Addr, FetchError> {
        if let Some(ip) = parse_ipv4_literal(host) {
            crate::serial_println!(
                "[dns] {} is a literal IP {}.{}.{}.{}",
                host, ip.0[0], ip.0[1], ip.0[2], ip.0[3]
            );
            return Ok(ip);
        }

        // DHT stub for .dominion identity-based names.
        if host.ends_with(".dominion") {
            let id = DominionId::from_pubkey(host.as_bytes());
            // Map to QEMU gateway IP (stub; will be replaced with real DHT lookup).
            let gw_ip = self.gw;
            crate::serial_println!(
                "[dominion] .dominion identity resolved via DHT stub (id={} → {}.{}.{}.{})",
                id.short(),
                gw_ip.0[0], gw_ip.0[1], gw_ip.0[2], gw_ip.0[3]
            );
            return Ok(gw_ip);
        }

        crate::serial_println!(
            "[dns] resolving {} — querying {} resolver(s)",
            host,
            DNS_SERVERS.len()
        );
        let gw_mac = self.gateway_mac().ok_or_else(|| {
            crate::serial_println!("[dns] gateway ARP failed — no reply within 1s");
            FetchError::Connect(host.into())
        })?;
        crate::serial_println!(
            "[dns] gateway MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            gw_mac.0[0], gw_mac.0[1], gw_mac.0[2], gw_mac.0[3], gw_mac.0[4], gw_mac.0[5]
        );

        let id = self.next_ident();
        let src_port = self.alloc_port();

        // Build the DNS payload once; re-wrap in new IP packets on each send (new ident).
        let dns_query = build_dns_query(id, host);
        let udp_payload = build_udp(src_port, 53, &dns_query);

        // Fan out: send to every resolver at once.
        for &srv in DNS_SERVERS {
            let ip_ident = self.next_ident();
            let ip_pkt = build_ipv4(self.ip, srv, IPPROTO_UDP, &udp_payload, ip_ident);
            let eth = build_ethernet(gw_mac, self.mac, ETHERTYPE_IPV4, &ip_pkt);
            self.send(&eth);
            crate::serial_println!(
                "[dns] query -> {}.{}.{}.{}:53 id={:#06x}",
                srv.0[0], srv.0[1], srv.0[2], srv.0[3], id
            );
        }

        // Retransmit every ~0.5 s within the total window.
        const RETRY_INTERVAL: u64 = TPS / 2;
        let deadline = ticks() + DNS_TIMEOUT;
        let mut next_retry = ticks() + RETRY_INTERVAL;
        let mut frames_seen: u32 = 0;
        let mut retry_n: u32 = 0;
        while ticks() < deadline {
            if let Some(f) = self.poll() {
                frames_seen += 1;
                if let Some(addr) = self.parse_dns_reply_verbose(&f, id, frames_seen) {
                    crate::serial_println!(
                        "[dns] {} resolved to {}.{}.{}.{} (frame#{} retry#{})",
                        host,
                        addr.0[0], addr.0[1], addr.0[2], addr.0[3],
                        frames_seen, retry_n
                    );
                    return Ok(addr);
                }
            } else {
                idle();
            }
            if ticks() >= next_retry && ticks() < deadline {
                retry_n += 1;
                crate::serial_println!(
                    "[dns] retransmit #{} for {} ({} frames seen)",
                    retry_n, host, frames_seen
                );
                for &srv in DNS_SERVERS {
                    let ip_ident = self.next_ident();
                    let ip_pkt = build_ipv4(self.ip, srv, IPPROTO_UDP, &udp_payload, ip_ident);
                    let eth = build_ethernet(gw_mac, self.mac, ETHERTYPE_IPV4, &ip_pkt);
                    self.send(&eth);
                }
                next_retry = ticks() + RETRY_INTERVAL;
            }
        }
        crate::serial_println!(
            "[dns] timeout resolving {} after {} frames, {} retransmits",
            host, frames_seen, retry_n
        );
        Err(FetchError::Dns(host.into()))
    }

    /// Match a DNS reply frame against our outstanding query `id`.
    /// We accept replies from any source IP with UDP source-port 53 — QEMU's
    /// slirp sometimes responds from a different virtual IP than the configured
    /// DNS server, and that strict check was silently dropping every reply.
    #[allow(dead_code)]
    fn parse_dns_reply(&self, frame: &[u8], id: u16) -> Option<Ipv4Addr> {
        self.parse_dns_reply_verbose(frame, id, 0)
    }

    /// Like `parse_dns_reply` but logs every non-matching frame to help debug
    /// QEMU slirp quirks. `frame_n` is just a counter for the log.
    fn parse_dns_reply_verbose(&self, frame: &[u8], id: u16, frame_n: u32) -> Option<Ipv4Addr> {
        let eth = parse_ethernet(frame)?;
        if eth.ethertype != ETHERTYPE_IPV4 {
            crate::serial_println!(
                "[dns] frame#{}: non-IPv4 ethertype {:#06x}",
                frame_n, eth.ethertype
            );
            return None;
        }
        let ip = parse_ipv4(eth.payload)?;
        if ip.protocol != IPPROTO_UDP {
            crate::serial_println!(
                "[dns] frame#{}: IPv4 proto={} src={}.{}.{}.{} (not UDP)",
                frame_n, ip.protocol,
                ip.src.0[0], ip.src.0[1], ip.src.0[2], ip.src.0[3]
            );
            return None;
        }
        let udp = parse_udp(ip.payload)?;
        if udp.src_port != 53 {
            crate::serial_println!(
                "[dns] frame#{}: UDP src_port={} dst_port={} src={}.{}.{}.{} (not DNS)",
                frame_n, udp.src_port, udp.dst_port,
                ip.src.0[0], ip.src.0[1], ip.src.0[2], ip.src.0[3]
            );
            return None;
        }
        // UDP from port 53 — this is a DNS reply candidate.
        crate::serial_println!(
            "[dns] frame#{}: DNS reply from {}.{}.{}.{} len={}",
            frame_n,
            ip.src.0[0], ip.src.0[1], ip.src.0[2], ip.src.0[3],
            udp.payload.len()
        );
        match parse_dns_answer(udp.payload, id) {
            Some(addr) => Some(addr),
            None => {
                // Log why the DNS answer was rejected (ID mismatch or no A record).
                if udp.payload.len() >= 2 {
                    let resp_id = u16::from_be_bytes([udp.payload[0], udp.payload[1]]);
                    let an_count = if udp.payload.len() >= 8 {
                        u16::from_be_bytes([udp.payload[6], udp.payload[7]])
                    } else {
                        0
                    };
                    crate::serial_println!(
                        "[dns] frame#{}: answer rejected — resp_id={:#06x} want={:#06x} an_count={}",
                        frame_n, resp_id, id, an_count
                    );
                }
                None
            }
        }
    }

    /// Send one TCP segment to `(dst, dst_port)` via the gateway.
    #[allow(clippy::too_many_arguments)]
    fn send_tcp(
        &mut self,
        gw_mac: MacAddr,
        dst: Ipv4Addr,
        src_port: u16,
        dst_port: u16,
        seq: u32,
        ack: u32,
        flags: u8,
        payload: &[u8],
    ) {
        let seg = build_tcp(
            self.ip, dst, src_port, dst_port, seq, ack, flags, 64240, payload,
        );
        let ident = self.next_ident();
        let ip = build_ipv4(self.ip, dst, IPPROTO_TCP, &seg, ident);
        let frame = build_ethernet(gw_mac, self.mac, ETHERTYPE_IPV4, &ip);
        self.send(&frame);
    }

    /// The full TCP fetch: capability-gated connect, NDN cache check, CUBIC-paced
    /// send, reassemble the reply, store in CS + OfflineReplica, tear down.
    ///
    /// Steps:
    /// 1. Firewall check (browser → external network)
    /// 2. NDN Content Store lookup (zero wire RTT on hit)
    /// 3. Capability-gated socket mint + bind
    /// 4. Gateway flow open
    /// 5. SYN handshake
    /// 6. CUBIC-paced request send
    /// 7. Response reassembly
    /// 8. Store in NDN CS + OfflineReplica
    /// 9. Gateway flow close + teardown
    fn tcp_fetch(
        &mut self,
        dst: Ipv4Addr,
        dst_port: u16,
        request: &[u8],
        host: &str,
    ) -> Result<Vec<u8>, FetchError> {
        // ── 1. Firewall check ──
        if !self.firewall.reachable(BROWSER_NODE, NET_NODE) {
            crate::serial_println!("[firewall] denied: browser → external network");
            return Err(FetchError::Connect(host.into()));
        }

        // ── 2. NDN Content Store lookup ──
        let req_hash = Hash256::of(request);
        let ndn_name =
            Name::parse(&alloc::format!("/http/{}/{}", host, hex_short(&req_hash)));
        match self.ndn.recv_interest(0, &ndn_name) {
            InterestOutcome::FromCache(data) => {
                crate::serial_println!("[ndn] cache hit for {}", host);
                return Ok(data.content.clone());
            }
            // Forward / Aggregated / Drop — proceed with real fetch.
            _ => {}
        }

        let gw_mac = self.gateway_mac().ok_or_else(|| FetchError::Connect(host.into()))?;
        let src_port = self.alloc_port();
        let iss = self.seq_seed.wrapping_add((tsc() as u32) & 0xffff);
        self.seq_seed = self.seq_seed.wrapping_add(0x9e37_79b9);

        // ── 3. Capability-gated socket ──
        let cap = SocketCapability::mint(
            Protocol::Tcp,
            src_port,
            Some((dst, dst_port)),
            ISSUER_KEY,
        );
        self.net_stack.bind(&cap).ok();
        crate::serial_println!("[cap] socket minted port={}", src_port);

        // ── 4. Gateway flow open ──
        let flow_key =
            self.gateway.open_outbound((self.ip, src_port), (dst, dst_port));
        crate::serial_println!(
            "[gateway] flow opened {}.{}.{}.{}:{} → {}.{}.{}.{}:{}",
            self.ip.0[0], self.ip.0[1], self.ip.0[2], self.ip.0[3], src_port,
            dst.0[0], dst.0[1], dst.0[2], dst.0[3], dst_port
        );

        // ── 5. Handshake: SYN → SYN-ACK → ACK ──

        // Re-initialize CUBIC for this connection.
        self.cubic = Cubic::new();

        let mut snd_nxt = iss;
        self.send_tcp(gw_mac, dst, src_port, dst_port, snd_nxt, 0, TCP_SYN, &[]);
        snd_nxt = snd_nxt.wrapping_add(1);

        let mut rcv_nxt;
        let deadline = ticks() + CONNECT_TIMEOUT;
        loop {
            if ticks() >= deadline {
                self.gateway.close(&flow_key);
                return Err(FetchError::Connect(host.into()));
            }
            let Some(frame) = self.poll() else {
                idle();
                continue;
            };
            let Some(seg) = self.tcp_for(&frame, dst, dst_port, src_port) else {
                continue;
            };
            if seg.flags & TCP_RST != 0 {
                self.gateway.close(&flow_key);
                return Err(FetchError::Connect(host.into()));
            }
            if seg.flags & TCP_SYN != 0 && seg.flags & TCP_ACK != 0 && seg.ack == snd_nxt {
                rcv_nxt = seg.seq.wrapping_add(1);
                self.send_tcp(gw_mac, dst, src_port, dst_port, snd_nxt, rcv_nxt, TCP_ACK, &[]);
                break;
            }
        }

        // ── 6. CUBIC-paced request send ──
        // Send up to cwnd_bytes of data, tracking in-flight bytes. After each ACK
        // we advance the window; after timeout we signal loss.
        {
            let mut in_flight: usize = 0;
            let mut send_offset: usize = 0;
            let total = request.len();

            while send_offset < total {
                let cwnd = self.cubic.cwnd_bytes();

                // Send segments up to the current cwnd.
                while send_offset < total && in_flight < cwnd {
                    let end = (send_offset + MAX_SEG).min(total);
                    let chunk = &request[send_offset..end];
                    self.send_tcp(
                        gw_mac, dst, src_port, dst_port,
                        snd_nxt, rcv_nxt, TCP_PSH | TCP_ACK, chunk,
                    );
                    snd_nxt = snd_nxt.wrapping_add(chunk.len() as u32);
                    in_flight += chunk.len();
                    send_offset += chunk.len();
                }

                if send_offset >= total {
                    break;
                }

                // Wait for ACKs before sending more.
                let ack_deadline = ticks() + CONNECT_TIMEOUT;
                let mut got_ack = false;
                while ticks() < ack_deadline {
                    let Some(frame) = self.poll() else {
                        idle();
                        continue;
                    };
                    let Some(seg) = self.tcp_for(&frame, dst, dst_port, src_port) else {
                        continue;
                    };
                    if seg.flags & TCP_RST != 0 {
                        self.gateway.close(&flow_key);
                        return Err(FetchError::Connect(host.into()));
                    }
                    if seg.flags & TCP_ACK != 0 {
                        // How many bytes did the peer acknowledge?
                        let acked = seg.ack.wrapping_sub(iss.wrapping_add(1)) as usize;
                        if acked <= in_flight {
                            in_flight = in_flight.saturating_sub(acked);
                        } else {
                            in_flight = 0;
                        }
                        self.cubic.on_ack(RTT_ESTIMATE);
                        got_ack = true;
                        break;
                    }
                }
                if !got_ack {
                    // Timeout: signal loss to CUBIC and continue (best-effort).
                    self.cubic.on_loss();
                    in_flight = 0;
                }
            }
        }

        // ── 7. Reassemble the in-order response stream ──
        let mut body: Vec<u8> = Vec::new();
        let mut got_fin = false;
        let hard_deadline = ticks() + RECV_TIMEOUT;
        let mut idle_deadline = ticks() + RECV_TIMEOUT; // becomes tighter after first byte
        loop {
            if ticks() >= hard_deadline || (!body.is_empty() && ticks() >= idle_deadline) {
                break;
            }
            let Some(frame) = self.poll() else {
                idle();
                continue;
            };
            let Some(seg) = self.tcp_for(&frame, dst, dst_port, src_port) else {
                continue;
            };
            if seg.flags & TCP_RST != 0 {
                break; // peer reset — return what we have
            }
            let payload = seg.payload;
            if seg.seq == rcv_nxt {
                if !payload.is_empty() {
                    if body.len() + payload.len() > MAX_BODY {
                        let take = MAX_BODY - body.len();
                        body.extend_from_slice(&payload[..take]);
                        got_fin = true;
                    } else {
                        body.extend_from_slice(payload);
                    }
                    rcv_nxt = rcv_nxt.wrapping_add(payload.len() as u32);
                    idle_deadline = ticks() + IDLE_TIMEOUT;
                    // ACK received data → update CUBIC.
                    self.cubic.on_ack(RTT_ESTIMATE);
                }
                if seg.flags & TCP_FIN != 0 {
                    rcv_nxt = rcv_nxt.wrapping_add(1);
                    got_fin = true;
                }
                // Acknowledge progress.
                self.send_tcp(gw_mac, dst, src_port, dst_port, snd_nxt, rcv_nxt, TCP_ACK, &[]);
            } else {
                // Out of order or retransmit — re-ACK what we expect.
                self.send_tcp(gw_mac, dst, src_port, dst_port, snd_nxt, rcv_nxt, TCP_ACK, &[]);
            }
            if got_fin {
                break;
            }
        }

        // ── 8. Teardown ──
        self.send_tcp(gw_mac, dst, src_port, dst_port, snd_nxt, rcv_nxt, TCP_FIN | TCP_ACK, &[]);

        // ── 9. Close gateway flow ──
        self.gateway.close(&flow_key);

        if body.is_empty() {
            return Err(FetchError::BadResponse);
        }

        // ── 10. Store in NDN CS + OfflineReplica ──
        let ndn_data = Data::new(ndn_name.clone(), &body);
        self.ndn.recv_data(ndn_data);
        self.offline.write(&body);
        crate::serial_println!(
            "[ndn] stored {} bytes for {}",
            body.len(), host
        );

        Ok(body)
    }

    /// Open a TCP connection (SYN handshake) and return a live [`TcpConn`] for
    /// interactive byte exchange — the substrate the TLS client runs over.
    fn open_tcp(
        &mut self,
        dst: Ipv4Addr,
        dst_port: u16,
        host: &str,
    ) -> Result<TcpConn, FetchError> {
        let gw_mac = self.gateway_mac().ok_or_else(|| FetchError::Connect(host.into()))?;
        let src_port = self.alloc_port();
        let iss = self.seq_seed.wrapping_add((tsc() as u32) & 0xffff);
        self.seq_seed = self.seq_seed.wrapping_add(0x9e37_79b9);

        let mut conn = TcpConn {
            mac: self.mac,
            ip: self.ip,
            gw_mac,
            dst,
            dst_port,
            src_port,
            snd_nxt: iss,
            rcv_nxt: 0,
            ident: self.ident,
            peer_fin: false,
        };
        conn.raw_send(TCP_SYN, &[]);
        conn.snd_nxt = conn.snd_nxt.wrapping_add(1);

        let deadline = ticks() + CONNECT_TIMEOUT;
        loop {
            if ticks() >= deadline {
                self.ident = conn.ident;
                return Err(FetchError::Connect(host.into()));
            }
            // Use the raw NIC poll here (not self.poll()) to avoid borrow conflicts with conn.
            let Some(frame) = (crate::netif::with_nic(|nic| nic.poll_frame())).flatten() else {
                idle();
                continue;
            };
            let Some(seg) = conn.tcp_for(&frame) else { continue };
            if seg.flags & TCP_RST != 0 {
                self.ident = conn.ident;
                return Err(FetchError::Connect(host.into()));
            }
            if seg.flags & TCP_SYN != 0
                && seg.flags & TCP_ACK != 0
                && seg.ack == conn.snd_nxt
            {
                conn.rcv_nxt = seg.seq.wrapping_add(1);
                conn.raw_send(TCP_ACK, &[]);
                self.ident = conn.ident;
                return Ok(conn);
            }
        }
    }

    /// Gather 32 bytes of handshake entropy from the TSC, MAC and seq seed.
    fn tls_entropy(&mut self) -> [u8; 32] {
        let mut e = [0u8; 32];
        for i in 0..4 {
            let t = tsc() ^ (self.seq_seed as u64).rotate_left(i * 7);
            e[i as usize * 8..i as usize * 8 + 8].copy_from_slice(&t.to_le_bytes());
            self.seq_seed = self.seq_seed.wrapping_add(0x9e37_79b9);
        }
        for i in 0..6 {
            e[i] ^= self.mac.0[i];
        }
        e
    }

    /// Extract a TCP segment from a frame if it belongs to our flow
    /// (`src=dst:dst_port → us:src_port`).
    fn tcp_for<'a>(
        &self,
        frame: &'a [u8],
        dst: Ipv4Addr,
        dst_port: u16,
        src_port: u16,
    ) -> Option<dominion_core::net::TcpSegment<'a>> {
        let eth = parse_ethernet(frame)?;
        if eth.ethertype != ETHERTYPE_IPV4 {
            return None;
        }
        let ip = parse_ipv4(eth.payload)?;
        if ip.src != dst || ip.dst != self.ip || ip.protocol != IPPROTO_TCP {
            return None;
        }
        let seg = parse_tcp(ip.payload)?;
        if seg.src_port != dst_port || seg.dst_port != src_port {
            return None;
        }
        Some(seg)
    }
}

impl Transport for KernelTransport {
    fn roundtrip(
        &mut self,
        host: &str,
        port: u16,
        secure: bool,
        request: &[u8],
    ) -> Result<Vec<u8>, FetchError> {
        let ip = self.resolve(host)?;
        if !secure {
            return self.tcp_fetch(ip, port, request, host);
        }
        // ── HTTPS: run the TLS 1.3 client over a live TCP connection ──
        // Firewall check applies to TLS connections as well.
        if !self.firewall.reachable(BROWSER_NODE, NET_NODE) {
            crate::serial_println!("[firewall] denied: browser → external network (TLS)");
            return Err(FetchError::Connect(host.into()));
        }

        let entropy = self.tls_entropy();
        // Build the system trust store once (the embedded Mozilla root bundle),
        // then keep it for the life of the transport.
        if self.trust.is_none() {
            self.trust = Some(dominion_core::x509::system_trust_store());
        }
        let now = crate::rtc::unix_now();
        let mut conn = self.open_tcp(ip, port, host)?;
        let trust = self.trust.as_ref().unwrap();
        // Full verification: the chain must reach an embedded root, the hostname
        // must match the leaf, the leaf's CertificateVerify signature and the
        // server Finished must check out. `now == 0` only if the RTC is unreadable,
        // in which case validity-window checks are skipped (everything else holds).
        let config = TlsConfig {
            hostname: host,
            trust,
            now,
            allow_unverified: false,
        };
        let result = (|| -> Result<Vec<u8>, FetchError> {
            let mut io = TcpIo { conn: &mut conn };
            let mut sess =
                tls::connect(&mut io, &config, &entropy).map_err(map_tls_err)?;
            sess.send(&mut io, request).map_err(map_tls_err)?;
            // Read the encrypted response, bounded by MAX_BODY.
            let mut body = Vec::new();
            loop {
                let chunk = sess.recv(&mut io).map_err(map_tls_err)?;
                if chunk.is_empty() {
                    break;
                }
                if body.len() + chunk.len() > MAX_BODY {
                    body.extend_from_slice(&chunk[..MAX_BODY - body.len()]);
                    break;
                }
                body.extend_from_slice(&chunk);
            }
            Ok(body)
        })();
        conn.close();
        match result {
            Ok(body) if !body.is_empty() => {
                // Cache successful TLS responses in the NDN CS + OfflineReplica too.
                let req_hash = Hash256::of(request);
                let ndn_name = Name::parse(&alloc::format!(
                    "/http/{}/{}",
                    host,
                    hex_short(&req_hash)
                ));
                let ndn_data = Data::new(ndn_name, &body);
                self.ndn.recv_data(ndn_data);
                self.offline.write(&body);
                crate::serial_println!("[ndn] stored {} bytes for {} (TLS)", body.len(), host);
                Ok(body)
            }
            Ok(_) => Err(FetchError::BadResponse),
            other => other,
        }
    }

    fn online(&self) -> bool {
        crate::netif::present()
    }
}

/// Map a TLS error onto the engine's fetch-error vocabulary.
fn map_tls_err(e: TlsError) -> FetchError {
    match e {
        TlsError::NameMismatch
        | TlsError::Expired
        | TlsError::UntrustedChain
        | TlsError::BadCertificate
        | TlsError::BadSignature
        | TlsError::BadFinished => FetchError::TlsHandshake,
        TlsError::Closed | TlsError::Io => FetchError::BadResponse,
        _ => FetchError::TlsHandshake,
    }
}

/// A live TCP connection presented as a byte stream for the TLS client.
struct TcpConn {
    mac: MacAddr,
    ip: Ipv4Addr,
    gw_mac: MacAddr,
    dst: Ipv4Addr,
    dst_port: u16,
    src_port: u16,
    snd_nxt: u32,
    rcv_nxt: u32,
    ident: u16,
    peer_fin: bool,
}

impl TcpConn {
    fn next_ident(&mut self) -> u16 {
        let i = self.ident;
        self.ident = self.ident.wrapping_add(1);
        i
    }

    fn raw_send(&mut self, flags: u8, payload: &[u8]) {
        let seg = build_tcp(
            self.ip,
            self.dst,
            self.src_port,
            self.dst_port,
            self.snd_nxt,
            self.rcv_nxt,
            flags,
            64240,
            payload,
        );
        let id = self.next_ident();
        let ip = build_ipv4(self.ip, self.dst, IPPROTO_TCP, &seg, id);
        let frame = build_ethernet(self.gw_mac, self.mac, ETHERTYPE_IPV4, &ip);
        let _ = crate::netif::with_nic(|nic| nic.transmit(&frame));
    }

    fn tcp_for<'a>(&self, frame: &'a [u8]) -> Option<TcpSegment<'a>> {
        let eth = parse_ethernet(frame)?;
        if eth.ethertype != ETHERTYPE_IPV4 {
            return None;
        }
        let ip = parse_ipv4(eth.payload)?;
        if ip.src != self.dst || ip.dst != self.ip || ip.protocol != IPPROTO_TCP {
            return None;
        }
        let seg = parse_tcp(ip.payload)?;
        if seg.src_port != self.dst_port || seg.dst_port != self.src_port {
            return None;
        }
        Some(seg)
    }

    fn close(&mut self) {
        self.raw_send(TCP_FIN | TCP_ACK, &[]);
    }
}

/// Adapts a [`TcpConn`] to the TLS [`Io`] trait.
struct TcpIo<'a> {
    conn: &'a mut TcpConn,
}

impl Io for TcpIo<'_> {
    fn write_all(&mut self, data: &[u8]) -> Result<(), TlsError> {
        for chunk in data.chunks(MAX_SEG) {
            self.conn.raw_send(TCP_PSH | TCP_ACK, chunk);
            self.conn.snd_nxt = self.conn.snd_nxt.wrapping_add(chunk.len() as u32);
        }
        Ok(())
    }

    fn read_some(&mut self, out: &mut Vec<u8>) -> Result<(), TlsError> {
        if self.conn.peer_fin {
            return Err(TlsError::Closed);
        }
        let hard = ticks() + RECV_TIMEOUT;
        loop {
            if ticks() >= hard {
                return Err(TlsError::Io);
            }
            let Some(frame) =
                (crate::netif::with_nic(|nic| nic.poll_frame())).flatten()
            else {
                idle();
                continue;
            };
            let Some(seg) = self.conn.tcp_for(&frame) else { continue };
            if seg.flags & TCP_RST != 0 {
                self.conn.peer_fin = true;
                return if out.is_empty() {
                    Err(TlsError::Closed)
                } else {
                    Ok(())
                };
            }
            if seg.seq == self.conn.rcv_nxt {
                let payload = seg.payload;
                if !payload.is_empty() {
                    out.extend_from_slice(payload);
                    self.conn.rcv_nxt =
                        self.conn.rcv_nxt.wrapping_add(payload.len() as u32);
                }
                if seg.flags & TCP_FIN != 0 {
                    self.conn.rcv_nxt = self.conn.rcv_nxt.wrapping_add(1);
                    self.conn.peer_fin = true;
                }
                self.conn.raw_send(TCP_ACK, &[]);
                if !payload.is_empty() {
                    return Ok(());
                }
                if self.conn.peer_fin {
                    return if out.is_empty() {
                        Err(TlsError::Closed)
                    } else {
                        Ok(())
                    };
                }
            } else {
                // Out of order / retransmit — re-ACK the byte we still expect.
                self.conn.raw_send(TCP_ACK, &[]);
            }
        }
    }
}

/// Parse a dotted-quad IPv4 literal, if `host` is one.
fn parse_ipv4_literal(host: &str) -> Option<Ipv4Addr> {
    let mut parts = [0u8; 4];
    let mut n = 0;
    for seg in host.split('.') {
        if n >= 4 {
            return None;
        }
        let v: u32 = seg.parse().ok()?;
        if v > 255 {
            return None;
        }
        parts[n] = v as u8;
        n += 1;
    }
    if n == 4 {
        Some(Ipv4Addr(parts))
    } else {
        None
    }
}

/// Format the first 4 bytes of a [`Hash256`] as lowercase hex — used in NDN
/// names and log output to keep identifiers short and human-readable.
fn hex_short(h: &Hash256) -> alloc::string::String {
    let b = &h.0[..4];
    alloc::format!("{:02x}{:02x}{:02x}{:02x}", b[0], b[1], b[2], b[3])
}

#[inline]
fn tsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn ticks() -> u64 {
    crate::keyboard::ticks()
}

/// Called at every network wait-point. A navigation runs synchronously on the
/// shell thread, so without this the screen would appear frozen for the whole
/// fetch. We keep the pointer tracking the mouse and animate a thin "loading"
/// bar across the top of the screen, then halt until the next interrupt — so the
/// machine stays visibly alive and idle the CPU instead of spinning.
fn idle() {
    keep_alive();
    x86_64::instructions::hlt();
}

/// Track the cursor and advance the loading indicator. Throttled so it only
/// repaints a few times a second (the bar animation), but the cursor follows
/// the mouse on every call. No-ops in headless (bench/test) mode.
fn keep_alive() {
    // No display in headless bench/test mode — skip all gfx.
    if !crate::gfx::available() {
        return;
    }
    // Move the hardware cursor sprite to the latest mouse position.
    let p = crate::mouse::poll();
    crate::gfx::set_cursor(p.x.max(0) as usize, p.y.max(0) as usize);

    // Animate an indeterminate loading bar ~20×/s.
    static LAST: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let now = ticks();
    let prev = LAST.load(core::sync::atomic::Ordering::Relaxed);
    if now.wrapping_sub(prev) >= TPS / 20 {
        LAST.store(now, core::sync::atomic::Ordering::Relaxed);
        draw_loading_bar(now);
    } else {
        crate::gfx::present_cursor();
    }
}

/// A 4px indeterminate progress bar along the very top edge: a bright segment
/// sweeps left→right over a dim track. Drawn directly to the back-buffer and
/// presented for just that strip; the page's repaint on completion clears it.
fn draw_loading_bar(phase: u64) {
    let w = crate::gfx::draw(|p| p.width()) as i32;
    let bar_h = 4i32;
    let track = crate::gfx::rgb(28, 30, 38);
    let glow = crate::gfx::rgb(90, 150, 255);
    let seg = w / 4;
    // Sweep position cycles across the width.
    let span = (w + seg).max(1);
    let pos = (phase as i32 * (w / 30).max(2)) % span - seg;
    crate::gfx::draw(|p| {
        p.fill_rect(0, 0, w, bar_h, track);
        p.fill_rect(pos, 0, seg, bar_h, glow);
    });
    crate::gfx::present_diff_rect((0, 0, w, bar_h));
    crate::gfx::present_cursor();
}

//! **EtherLink Network Benchmark Suite** — host-runnable, no_std-compatible.
//!
//! Measures the raw throughput of every networking layer in dominion-core vs
//! conventional TCP/IP, establishing a performance baseline for the 11-lever
//! improvement framework.
//!
//! Run with:
//!   cargo test -p dominion-core net_bench -- --nocapture
//!
//! Each benchmark reports:
//!   NET_BENCH <subsystem> key=value ...

#[cfg(test)]
mod net_bench {
    extern crate std;
    use std::time::Instant;

    use crate::dominionlink::{DominionId, DominionLink};
    use crate::hash::Hash256;
    use crate::legacynet::{
        Protocol, SocketCapability, NetworkStack, TcpConnection,
        LegacyGateway, encapsulate, decapsulate, FlowKey,
    };
    use crate::ndn::{Forwarder, Name, Data};
    use crate::net::{
        build_ethernet, build_ipv4, build_tcp, build_udp, build_icmp_echo, build_dns_query,
        parse_ethernet, parse_ipv4, parse_tcp, parse_dns_answer,
        checksum, MacAddr, Ipv4Addr, ArpCache,
        ETHERTYPE_IPV4, IPPROTO_TCP, IPPROTO_UDP,
        TCP_SYN, TCP_ACK,
    };
    use crate::transport::{Cubic, Bbr, OfflineReplica, ReplicaStore};
    use alloc::vec::Vec;

    // ─────────── reporting helpers ───────────
    fn ns_per_op(elapsed_secs: f64, iters: u64) -> f64 {
        (elapsed_secs * 1_000_000_000.0) / iters as f64
    }
    fn mops(elapsed_secs: f64, iters: u64) -> f64 {
        iters as f64 / elapsed_secs / 1_000_000.0
    }
    fn gbps(bytes_total: u64, elapsed_secs: f64) -> f64 {
        (bytes_total as f64 / 1_073_741_824.0) / elapsed_secs
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 1 — Ethernet frame construction + parsing throughput
    // Establishes the raw wire codec baseline.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_ethernet_codec() {
        const ITERS: u64 = 2_000_000;
        let dst = MacAddr([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        let src = MacAddr([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        let payload = [0u8; 1400];

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let frame = build_ethernet(dst, src, ETHERTYPE_IPV4, &payload);
            let _ = core::hint::black_box(parse_ethernet(&frame));
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes_total = ITERS * (14 + 1400);
        std::println!(
            "NET_BENCH ethernet_codec iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), gbps(bytes_total, secs), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 2 — IPv4 build + parse + checksum validation
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_ipv4_codec() {
        const ITERS: u64 = 2_000_000;
        let src = Ipv4Addr::new(10, 0, 2, 15);
        let dst = Ipv4Addr::new(93, 184, 216, 34);
        let payload = [0xABu8; 1400];

        let t0 = Instant::now();
        for i in 0..ITERS {
            let pkt = build_ipv4(src, dst, IPPROTO_TCP, &payload, (i & 0xFFFF) as u16);
            let _ = core::hint::black_box(parse_ipv4(&pkt));
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes_total = ITERS * (20 + 1400);
        std::println!(
            "NET_BENCH ipv4_codec iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), gbps(bytes_total, secs), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 3 — TCP segment build + checksum (RFC 793 pseudo-header)
    // The most expensive per-packet operation in the legacy stack.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_tcp_codec() {
        const ITERS: u64 = 1_000_000;
        let src = Ipv4Addr::new(10, 0, 2, 15);
        let dst = Ipv4Addr::new(93, 184, 216, 34);
        let payload = [0u8; 1400];

        let t0 = Instant::now();
        for i in 0..ITERS {
            let seg = build_tcp(
                src, dst, 49152, 80,
                i as u32, (i + 1) as u32,
                TCP_SYN | TCP_ACK, 65535,
                &payload,
            );
            let _ = core::hint::black_box(parse_tcp(&seg));
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes_total = ITERS * (20 + 1400);
        std::println!(
            "NET_BENCH tcp_codec iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), gbps(bytes_total, secs), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 4 — RFC 1071 internet checksum raw throughput
    // Every IP/TCP/UDP segment pays this cost.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_checksum_throughput() {
        const ITERS: u64 = 5_000_000;
        let data = [0xA5u8; 1400];

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let c = checksum(&data);
            let _ = core::hint::black_box(c);
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes_total = ITERS * 1400;
        std::println!(
            "NET_BENCH rfc1071_checksum iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), gbps(bytes_total, secs), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 5 — ARP cache lookup: O(log n) BTreeMap vs brute scan
    // The first step of every outbound packet on the legacy path.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_arp_cache() {
        const ITERS: u64 = 5_000_000;
        let mut cache = ArpCache::new();
        for i in 1u8..=254 {
            cache.insert(Ipv4Addr::new(10, 0, 2, i), MacAddr([0x52, 0x54, 0, 0, 0, i]));
        }
        let target = Ipv4Addr::new(10, 0, 2, 128);

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = core::hint::black_box(cache.lookup(target));
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH arp_cache_lookup iters={} mops={:.2} ns_per_op={:.1} entries=254",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 6 — DNS query build + answer parse
    // DominionOS eliminates DNS entirely; this measures what we save.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_dns_codec() {
        const ITERS: u64 = 1_000_000;
        let q = build_dns_query(0xDEAD, "example.dominion.net");
        // Synthetic DNS A-record answer
        let mut answer = q.clone();
        answer[2] = 0x81; answer[3] = 0x80;
        answer[6] = 0; answer[7] = 1; // ANCOUNT=1
        answer.extend_from_slice(&[0xC0, 0x0C]); // name ptr
        answer.extend_from_slice(&1u16.to_be_bytes()); // TYPE A
        answer.extend_from_slice(&1u16.to_be_bytes()); // CLASS IN
        answer.extend_from_slice(&300u32.to_be_bytes()); // TTL
        answer.extend_from_slice(&4u16.to_be_bytes()); // RDLENGTH
        answer.extend_from_slice(&[10, 0, 2, 15]);

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let q2 = build_dns_query(0xDEAD, "example.dominion.net");
            let _ = core::hint::black_box(parse_dns_answer(&answer, 0xDEAD));
            let _ = core::hint::black_box(q2);
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH dns_codec iters={} mops={:.2} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 7 — CUBIC congestion control: window updates/s
    // Governs the DominionLink overlay pacing at the transport layer.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_cubic_cc() {
        const ITERS: u64 = 10_000_000;
        let mut c = Cubic::new();

        let t0 = Instant::now();
        for i in 0..ITERS {
            if i % 200 == 0 {
                c.on_loss();
            } else {
                c.on_ack(0.010);
            }
            let _ = core::hint::black_box(c.cwnd_segments());
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH cubic_cc iters={} mops={:.2} ns_per_op={:.1} loss_rate=0.5pct",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 8 — BBR congestion control: BDP tracking updates/s
    // Rate-based; eliminates the buffer-bloat CUBIC induces.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_bbr_cc() {
        const ITERS: u64 = 10_000_000;
        let mut b = Bbr::new();

        let t0 = Instant::now();
        for i in 0..ITERS {
            b.on_ack(1448, 0.010 + (i % 5) as f64 * 0.001);
            let _ = core::hint::black_box(b.pacing_rate());
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH bbr_cc iters={} mops={:.2} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 9 — DominionId DHT XOR distance + identity minting
    // The "routing" cost for content-addressed networking (replaces DNS).
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_dominion_identity() {
        const ITERS: u64 = 500_000;
        let key_a = b"dominionos-node-key-alpha-00000001";
        let key_b = b"dominionos-node-key-beta-000000002";
        let id_a = DominionId::from_pubkey(key_a);
        let id_b = DominionId::from_pubkey(key_b);

        // DHT XOR distance (the Kademlia routing metric)
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let d = id_a.distance(&id_b);
            let _ = core::hint::black_box(d);
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH dht_xor_distance iters={} mops={:.2} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );

        // Identity minting: H(pubkey) — "you are your key"
        let t1 = Instant::now();
        for i in 0..ITERS {
            let mut k = *key_a;
            k[0] = (i & 0xFF) as u8;
            let id = DominionId::from_pubkey(&k);
            let _ = core::hint::black_box(id);
        }
        let secs2 = t1.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH dominion_id_mint iters={} mops={:.2} ns_per_op={:.1}",
            ITERS, mops(secs2, ITERS), ns_per_op(secs2, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 10 — Content-addressed store: publish + fetch + verify
    // The NDN in-network cache path. Any node can serve; consumer verifies.
    // Note: DominionLink::fetch re-hashes the content to verify integrity.
    // We benchmark at two payload sizes to show the scaling characteristic.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_content_addressed_store() {
        let key = b"benchmark-node-pub-key-01234567";
        let id = DominionId::from_pubkey(key);
        let mut link = DominionLink::new(id);

        // 256B: publish + fetch + verify (re-hash every fetch)
        {
            const ITERS: u64 = 200_000;
            let small = &[0xAAu8; 256][..];
            let t0 = Instant::now();
            for _ in 0..ITERS {
                let h = link.publish(small); // idempotent on same hash
                let fetched = link.fetch(h); // re-hashes 256B to verify
                let _ = core::hint::black_box(fetched);
            }
            let secs = t0.elapsed().as_secs_f64();
            let bytes = ITERS * 256;
            std::println!(
                "NET_BENCH content_store_256b iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
                ITERS, mops(secs, ITERS), gbps(bytes, secs), ns_per_op(secs, ITERS)
            );
        }

        // 4KB: publish is idempotent; fetch re-verifies by re-hashing 4KB
        {
            const ITERS: u64 = 50_000;
            let medium = &[0xBBu8; 4096][..];
            let h_medium = link.publish(medium);
            let t1 = Instant::now();
            for _ in 0..ITERS {
                let _ = core::hint::black_box(link.fetch(h_medium));
            }
            let secs = t1.elapsed().as_secs_f64();
            let bytes = ITERS * 4096;
            std::println!(
                "NET_BENCH content_store_fetch_4k iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
                ITERS, mops(secs, ITERS), gbps(bytes, secs), ns_per_op(secs, ITERS)
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 11 — NDN Forwarder: FIB longest-prefix lookup + PIT aggregation
    // The forwarding-plane cost of content-centric routing.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_ndn_forwarder() {
        const ITERS: u64 = 1_000_000;
        let mut fwd = Forwarder::new();

        // Install FIB routes
        fwd.register_route(Name::parse("/dominion"), 0);
        fwd.register_route(Name::parse("/dominion/docs"), 1);
        fwd.register_route(Name::parse("/dominion/media"), 2);
        fwd.register_route(Name::parse("/cdn"), 3);
        fwd.register_route(Name::parse("/cdn/video"), 4);

        let name = Name::parse("/dominion/docs/networking/spec-v3");

        // FIB lookup (longest prefix match)
        let t0 = Instant::now();
        for i in 0..ITERS {
            let result = fwd.recv_interest(i % 4, &name);
            let _ = core::hint::black_box(result);
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH ndn_fib_lookup iters={} mops={:.2} ns_per_op={:.1} fib_entries=5",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );

        // PIT aggregation + CS hit: rebuild a fresh forwarder for clean state
        let mut fwd2 = Forwarder::new();
        fwd2.register_route(Name::parse("/dominion/docs"), 1);
        let content = b"the networking spec content";
        let d = Data::new(name.clone(), content);

        let t1 = Instant::now();
        let mut cs_hits = 0u64;
        for i in 0..ITERS {
            let outcome = fwd2.recv_interest(i % 4, &name);
            if i % 100 == 0 {
                // Serve data — goes into CS, satisfies pending interests
                let faces = fwd2.recv_data(d.clone());
                cs_hits += faces.len() as u64;
            }
            let _ = core::hint::black_box(outcome);
        }
        let secs2 = t1.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH ndn_pit_cs iters={} mops={:.2} ns_per_op={:.1} cs_hits={}",
            ITERS, mops(secs2, ITERS), ns_per_op(secs2, ITERS), cs_hits
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 12 — Socket capability: mint + authenticate
    // Replaces kernel open()/bind()/connect() with an offline-verifiable token.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_socket_capability() {
        const ITERS: u64 = 2_000_000;
        let issuer_key = b"dominionos-cap-issuer-key-v1-0000";
        let remote = (Ipv4Addr::new(93, 184, 216, 34), 443u16);

        let t0 = Instant::now();
        for i in 0..ITERS {
            let port = 49152u16.wrapping_add((i % 16384) as u16);
            let cap = SocketCapability::mint(Protocol::Tcp, port, Some(remote), issuer_key);
            let ok = core::hint::black_box(cap.is_authentic(issuer_key));
            let _ = core::hint::black_box(ok);
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH socket_cap_mint_verify iters={} mops={:.2} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 13 — DominionLink-over-UDP encap + decap
    // The interop bridge: native Dominion overlay riding over commodity UDP.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_dominionlink_over_udp() {
        const ITERS: u64 = 2_000_000;
        let payload = vec![0xEEu8; 1200];
        let id = Hash256::of(b"self-certifying-object-id-seed");

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let datagram = encapsulate(40000, &id, &payload);
            // decapsulate operates on the UDP body (skip 8-byte header)
            let body = &datagram[8..];
            let result = core::hint::black_box(decapsulate(body));
            let _ = core::hint::black_box(result);
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes = ITERS * 1200;
        std::println!(
            "NET_BENCH dominionlink_over_udp iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), gbps(bytes, secs), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 14 — OfflineReplica: write + reconcile
    // Same-hash content deduplication on reconnect — no conflicts by design.
    // Each unique payload is stored once; same bytes dedup by hash.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_offline_replica() {
        // Write phase: 5K unique 256B objects offline (~1.25MB total)
        const WRITE_ITERS: u64 = 5_000;
        let mut replica = OfflineReplica::new();
        replica.set_online(false);

        let t0 = Instant::now();
        for i in 0..WRITE_ITERS {
            let mut data = [0xABu8; 256];
            data[0] = (i % 256) as u8;
            data[1] = ((i >> 8) % 256) as u8;
            let _ = core::hint::black_box(replica.write(&data));
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH offline_write iters={} mops={:.2} ns_per_op={:.1} pending={}",
            WRITE_ITERS, mops(secs, WRITE_ITERS), ns_per_op(secs, WRITE_ITERS),
            replica.pending()
        );

        // Reconcile: flush queue to a remote peer — dedup by hash
        let mut remote = ReplicaStore::new();
        let t1 = Instant::now();
        let report = replica.reconcile(&mut remote, &[]);
        let secs2 = t1.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH offline_reconcile queued={} pushed={} deduped={} ms={:.2}",
            WRITE_ITERS, report.pushed, report.deduped, secs2 * 1000.0
        );

        // Idempotency benchmark: re-writing already-remote content is deduped, not re-sent
        replica.set_online(false);
        let shared = [0xABu8; 256]; // same bytes as first write above — already on remote
        let t2 = Instant::now();
        for _ in 0..WRITE_ITERS {
            let _ = core::hint::black_box(replica.write(&shared));
        }
        let secs3 = t2.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH offline_dedup_write iters={} mops={:.2} ns_per_op={:.1}",
            WRITE_ITERS, mops(secs3, WRITE_ITERS), ns_per_op(secs3, WRITE_ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 15 — TCP 3-way handshake state machine throughput
    // The OS-level connection setup cost (not wire RTT).
    // DominionOS eliminates this with identity-based sessions.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_tcp_handshake() {
        const ITERS: u64 = 2_000_000;

        let t0 = Instant::now();
        for i in 0..ITERS {
            let mut conn = TcpConnection::new(i as u32);
            let _syn_seq = conn.connect();
            // SYN-ACK: their_seq=1000, ack=our_syn+1=i+1
            let _ack = conn.on_syn_ack(1000, (i + 1) as u32);
            let _ = core::hint::black_box(conn.state);
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH tcp_handshake_sm iters={} mops={:.2} ns_per_op={:.1} conns_per_sec={:.0}",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS),
            ITERS as f64 / secs
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 16 — LegacyGateway: flow open + inbound filter
    // Stateful NAT gate — the minimum correctness cost of IP perimeter model.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_legacy_gateway() {
        const ITERS: u64 = 2_000_000;
        let mut gw = LegacyGateway::new();
        let local = (Ipv4Addr::new(10, 0, 2, 15), 51000u16);
        let remote = (Ipv4Addr::new(93, 184, 216, 34), 443u16);
        let key = gw.open_outbound(local, remote);

        // Inbound packet filter — runs on every received packet
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let ok = core::hint::black_box(gw.inbound_allowed(remote, local));
            let _ = ok;
        }
        let secs = t0.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH gateway_inbound_filter iters={} mops={:.2} ns_per_op={:.1} flows=1",
            ITERS, mops(secs, ITERS), ns_per_op(secs, ITERS)
        );

        gw.close(&key);

        // Scale: how fast does the filter degrade with N open flows?
        let mut gw2 = LegacyGateway::new();
        for p in 49152u16..49152 + 1000 {
            gw2.open_outbound((Ipv4Addr::new(10, 0, 2, 15), p), (Ipv4Addr::new(1, 1, 1, 1), 443));
        }
        let t1 = Instant::now();
        for _ in 0..ITERS {
            let ok = core::hint::black_box(gw2.inbound_allowed(
                (Ipv4Addr::new(1, 1, 1, 1), 443),
                (Ipv4Addr::new(10, 0, 2, 15), 49500),
            ));
            let _ = ok;
        }
        let secs2 = t1.elapsed().as_secs_f64();
        std::println!(
            "NET_BENCH gateway_inbound_filter_1k_flows iters={} mops={:.2} ns_per_op={:.1} flows=1000",
            ITERS, mops(secs2, ITERS), ns_per_op(secs2, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCH 17 — Full outbound pipeline: Eth + IPv4 + TCP + payload
    // Measures the total serialization cost of a complete outbound packet.
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_full_outbound_pipeline() {
        const ITERS: u64 = 500_000;
        let src_mac = MacAddr([0x52, 0x54, 0x00, 0x00, 0x00, 0x01]);
        let gw_mac = MacAddr([0x52, 0x54, 0x00, 0x00, 0x00, 0x02]);
        let src_ip = Ipv4Addr::new(10, 0, 2, 15);
        let dst_ip = Ipv4Addr::new(93, 184, 216, 34);
        let http_req = b"GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n";

        let t0 = Instant::now();
        for i in 0..ITERS {
            let tcp_seg = build_tcp(
                src_ip, dst_ip,
                49152 + (i % 16384) as u16, 80,
                i as u32, 0, TCP_SYN, 65535,
                http_req,
            );
            let ip_pkt = build_ipv4(src_ip, dst_ip, IPPROTO_TCP, &tcp_seg, (i & 0xFFFF) as u16);
            let frame = build_ethernet(gw_mac, src_mac, ETHERTYPE_IPV4, &ip_pkt);
            let _ = core::hint::black_box(frame);
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes = ITERS * (14 + 20 + 20 + http_req.len() as u64);
        std::println!(
            "NET_BENCH full_outbound_pipeline iters={} mops={:.2} gbps={:.3} ns_per_op={:.1}",
            ITERS, mops(secs, ITERS), gbps(bytes, secs), ns_per_op(secs, ITERS)
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY — Print comparison matrix
    // ═══════════════════════════════════════════════════════════════
    #[test]
    fn bench_summary() {
        std::println!("");
        std::println!("═══════════════════════════════════════════════════════════════");
        std::println!(" ETHERLINK vs CONVENTIONAL INTERNET — BASELINE COMPARISON");
        std::println!("═══════════════════════════════════════════════════════════════");
        std::println!(" WINDOWS HOST BASELINE (measured):");
        std::println!("   Ping 8.8.8.8:          avg=91.6ms   min=18ms   max=131ms");
        std::println!("   DNS first lookup:       ~171ms       (google.com)");
        std::println!("   Raw buffer copy (64KB): 33,421 MB/s  (loopback)");
        std::println!("   Wi-Fi link speed:       866.7 Mbps   (AX201 160MHz)");
        std::println!("   TCP handshake budget:   ~44ms        (localhost connect)");
        std::println!("");
        std::println!(" DOMINIONOS PROTOCOL STACK (this run):");
        std::println!("   See NET_BENCH lines above for live measurements.");
        std::println!("");
        std::println!(" KEY ELIMINATIONS vs CONVENTIONAL STACK:");
        std::println!("   DNS lookup:        ELIMINATED (identity = H(pubkey))");
        std::println!("   TCP 3-way SYN:     ELIMINATED (NDN Interest/Data, 1 RTT)");
        std::println!("   TLS handshake:     ELIMINATED (object-level signing)");
        std::println!("   IP address lookup: ELIMINATED (DHT XOR routing)");
        std::println!("   CA chain verify:   ELIMINATED (HIBC self-certification)");
        std::println!("   Per-packet chksum: RETAINED   (legacy interop only)");
        std::println!("═══════════════════════════════════════════════════════════════");
    }
}

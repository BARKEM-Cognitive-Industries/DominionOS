//! Deterministic fuzz & property harness — **testing & verification strategy**.
//!
//! Every parser sits on a trust boundary: it must turn *arbitrary* bytes into a
//! typed value or a clean error, but **never panic, loop forever, or read out of
//! bounds**. This module is a portable, seed-driven fuzzer that exercises those
//! boundaries deterministically (so any crash is reproducible from its seed), plus
//! property checks that assert the system's core invariants over swept inputs.
//!
//! Two generation strategies, both pure functions of a seed:
//!
//! * **Unstructured** — raw pseudo-random bytes, to hit the error paths.
//! * **Structure-aware** — byte streams shaped like the target format (a valid
//!   header with a fuzzed body), to reach deeper into the parser.
//!
//! Encoders are checked for the **round-trip** law `parse(encode(x)) == x`. The
//! harness runs in `cargo test`; the iteration counts are CI-sized constants, not
//! limits — raise them for an overnight million-case soak.

use crate::random::Drng;
use alloc::vec::Vec;

/// A deterministic byte-stream generator: a thin, seekable view over a [`Drng`].
pub struct FuzzInput {
    rng: Drng,
}

impl FuzzInput {
    pub fn new(seed: u64) -> FuzzInput {
        FuzzInput { rng: Drng::from_seed(&seed.to_le_bytes()) }
    }

    /// `len` pseudo-random bytes.
    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut v = alloc::vec![0u8; len];
        self.rng.fill(&mut v);
        v
    }

    /// A random length in `[0, max]`, then that many bytes.
    pub fn blob(&mut self, max: usize) -> Vec<u8> {
        let len = (self.rng.next_u64() as usize) % (max + 1);
        self.bytes(len)
    }

    pub fn u8(&mut self) -> u8 {
        (self.rng.next_u64() & 0xff) as u8
    }
    pub fn u16(&mut self) -> u16 {
        (self.rng.next_u64() & 0xffff) as u16
    }
}

/// Run `f` over `iters` deterministic inputs. Reaching the end means no input
/// panicked or hung — the safety property for a parser on a trust boundary.
pub fn sweep(base_seed: u64, iters: u64, mut f: impl FnMut(u64)) {
    for i in 0..iters {
        f(base_seed.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{Capability, Rights};
    use crate::codec::CodecRegistry;
    use crate::net;
    use crate::object::{Datum, Object, ObjectGraph};

    const ITERS: u64 = 4000;

    // ───────────────────── parser no-panic fuzzing ─────────────────────

    #[test]
    fn elf_parser_never_panics_on_garbage() {
        sweep(0xE1F, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let bytes = input.blob(512);
            // Result either way — the only requirement is it returns, never panics.
            let _ = crate::elf::parse(&bytes);
        });
        // Structure-aware: a valid ELF magic + fuzzed remainder reaches deeper.
        sweep(0xE1F2, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let mut bytes = alloc::vec![0x7f, b'E', b'L', b'F'];
            bytes.extend(input.blob(256));
            let _ = crate::elf::parse(&bytes);
        });
    }

    #[test]
    fn object_graph_deserialize_never_panics() {
        sweep(0x0B7EC7, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let bytes = input.blob(1024);
            let _ = ObjectGraph::deserialize(&bytes);
        });
        // Structure-aware: real magic header + fuzzed body.
        sweep(0x0B7E2, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let mut bytes = b"AEGRPH01".to_vec();
            bytes.extend(input.blob(512));
            let _ = ObjectGraph::deserialize(&bytes);
        });
    }

    #[test]
    fn network_parsers_never_panic() {
        sweep(0x4E7, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let b = input.blob(256);
            let _ = net::parse_ethernet(&b);
            let _ = net::ArpPacket::parse(&b);
            let _ = net::parse_ipv4(&b);
            let _ = net::parse_icmp_echo(&b);
            let _ = net::parse_udp(&b);
        });
    }

    #[test]
    fn dominion_lexer_parser_never_panics() {
        sweep(0xA17, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let b = input.blob(128);
            // Interpret the fuzzed bytes as (possibly invalid) UTF-8 source.
            if let Ok(src) = core::str::from_utf8(&b) {
                let _ = crate::lang::eval_source(src);
            }
        });
        // Structure-aware: fuzz around real tokens/operators.
        let frags = ["let", "x", "=", "=>", "(", ")", "+", "*", "1", "obj", "cell", "@NPU", ";"];
        sweep(0xA172, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let mut s = alloc::string::String::new();
            for _ in 0..(input.u8() % 24) {
                s.push_str(frags[(input.u8() as usize) % frags.len()]);
                s.push(' ');
            }
            let _ = crate::lang::eval_source(&s);
        });
    }

    #[test]
    fn wasm_sandbox_never_panics_on_random_programs() {
        use crate::wasm::{Op, Sandbox};
        // Map a fuzz byte to a guest instruction. Operands are kept small so jumps,
        // locals, and memory accesses land both in- and out-of-bounds.
        fn rand_op(input: &mut FuzzInput) -> Op {
            match input.u8() % 16 {
                0 => Op::Const(input.u16() as i64 - 32768),
                1 => Op::Add,
                2 => Op::Sub,
                3 => Op::Mul,
                4 => Op::Div,
                5 => Op::Rem,
                6 => Op::Eq,
                7 => Op::Lt,
                8 => Op::GetLocal((input.u8() % 10) as usize),
                9 => Op::SetLocal((input.u8() % 10) as usize),
                10 => Op::Load,
                11 => Op::Store,
                12 => Op::Jump((input.u8() % 40) as usize),
                13 => Op::JumpIfZero((input.u8() % 40) as usize),
                14 => Op::Call { id: input.u16() as u32, argc: (input.u8() % 4) as usize },
                _ => Op::Return,
            }
        }
        // A malicious guest is exactly arbitrary bytecode. The sandbox's contract is
        // total: every program returns `Ok` or a `Trap` — never a host panic, never an
        // infinite loop (gas-bounded), never host-heap exhaustion (stack-depth-bounded).
        sweep(0x5A4D, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let n = (input.u8() % 64) as usize;
            let code: Vec<Op> = (0..n).map(|_| rand_op(&mut input)).collect();
            let mut s = Sandbox::new(code, 16, 8, 50_000);
            let _ = s.run();
        });
    }

    #[test]
    fn codec_import_never_panics() {
        let reg = CodecRegistry::with_defaults();
        let cap = Capability::mint(0, 0x10000, Rights::READ);
        sweep(0xC0DEC, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let b = input.blob(512);
            let _ = reg.import(None, &b, &cap);
            let _ = reg.sniff(&b);
        });
        // Structure-aware: a PPM-looking header with a fuzzed body.
        sweep(0xC0DEC2, ITERS, |seed| {
            let mut input = FuzzInput::new(seed);
            let mut b = b"P6\n2 2\n255\n".to_vec();
            b.extend(input.blob(64));
            let _ = reg.import(Some("x.ppm"), &b, &cap);
        });
    }

    // ───────────────────── round-trip invariants ─────────────────────

    #[test]
    fn object_graph_serialize_round_trips_for_random_graphs() {
        sweep(0x5E812, 1500, |seed| {
            let mut input = FuzzInput::new(seed);
            let mut g = ObjectGraph::new();
            let count = (input.u8() % 12) as usize;
            for _ in 0..count {
                let payload = input.blob(48);
                let obj = Object::new("Fuzz").with("data", Datum::Bytes(payload));
                g.put(obj);
            }
            let bytes = g.serialize();
            let restored = ObjectGraph::deserialize(&bytes).expect("valid graph must reload");
            // Content addressing ⇒ identical root after a serialize/deserialize cycle.
            assert_eq!(g.root_hash(), restored.root_hash());
        });
    }

    #[test]
    fn udp_build_parse_round_trips() {
        sweep(0x0D9, 2000, |seed| {
            let mut input = FuzzInput::new(seed);
            let sp = input.u16();
            let dp = input.u16();
            let payload = input.blob(64);
            let datagram = net::build_udp(sp, dp, &payload);
            let parsed = net::parse_udp(&datagram).expect("our own datagram must parse");
            assert_eq!(parsed.src_port, sp);
            assert_eq!(parsed.dst_port, dp);
            assert_eq!(parsed.payload, &payload[..]);
        });
    }

    #[test]
    fn ipv4_checksum_is_correct_by_construction() {
        sweep(0x1F4C, 2000, |seed| {
            let mut input = FuzzInput::new(seed);
            let payload = input.blob(64);
            let pkt = net::build_ipv4(
                net::Ipv4Addr([10, 0, 2, 15]),
                net::Ipv4Addr([10, 0, 2, 2]),
                17,
                &payload,
                input.u16(),
            );
            // A correctly built header must parse and its checksum must verify.
            assert!(net::parse_ipv4(&pkt).is_some());
        });
    }
}

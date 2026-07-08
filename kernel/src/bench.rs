//! The real-world benchmark battery (`--features qemu_bench`).
//!
//! This is the performance counterpart to [`selftest`](crate::selftest): it runs
//! *inside the booted OS* on the QEMU virtual machine, after the heap is mapped
//! and devices are up, and drives every subsystem at scale. Each benchmark prints
//! one or more machine-readable lines of the form
//!
//! ```text
//! BENCH <category> key=value key=value ...
//! ```
//!
//! so the host launcher (`run-bench.ps1`) can parse the serial log into a table
//! and a JSON file, and the Linux comparison harness (`bench/linux/`) can emit the
//! identical schema for a side-by-side comparison.
//!
//! ## What is measured vs. modelled
//!
//! Everything here is a *real* measurement of DominionOS code running on the metal —
//! not a simulation. Where the architecture imposes a ceiling (the cooperative
//! scheduler's dispatch is O(n) per step; a single QEMU vCPU; a 16 MiB data disk)
//! the benchmark measures the honest rate at a feasible size and reports the
//! headline figure as an explicit **extrapolation** (`*_projected_*`). True
//! multi-machine scaling is deferred to the host harness; here we measure the
//! distributed *substrate* (inter-core messaging + CRDT convergence).
//!
//! Timing uses the TSC, calibrated against the PIT exactly as the desktop does.

use crate::{exit_qemu, serial_println, QemuExitCode};
use dominion_core::capability::{Capability, Rights};
use dominion_core::hash::Hash256;
use dominion_core::lang::ast::Item;
use dominion_core::lang::parser::parse_source;
use dominion_core::lang::{Interpreter, Value};
use dominion_core::multikernel::{ConvergentState, Multikernel};
use dominion_core::object::{Datum, Object, ObjectGraph};
use dominion_core::persist::{BlockDevice, BlockError, Persistence, RamDisk, BLOCK_SIZE};
use dominion_core::sched::Scheduler;
use dominion_core::vault::{CipherSuite, Key, Vault};
use alloc::vec::Vec;
use core::hint::black_box;

// ─────────────────────────── scale knobs ───────────────────────────
// Sized for the 1 GiB `big_heap` the `qemu_bench` feature pulls in. The headline
// targets the user asked for (1M tasks, 1M-node DAG) run for real; the "billions"
// figures are measured at a feasible size and extrapolated linearly.

/// Lightweight tasks (domains) to spawn — the headline "1,000,000 tasks".
const TASKS: usize = 1_000_000;
/// Tasks to actually run to completion. The scheduler's dispatch is O(n) per step
/// (it scans domains by id), so run-to-completion is O(n²); we measure it at a
/// feasible size and report the per-dispatch cost.
const TASKS_COMPLETE: usize = 20_000;
/// Messages to ping-pong between two isolated domains (constant memory).
const MSGS: u64 = 5_000_000;
/// Nodes in the executed DAG — the headline "1,000,000-node graph".
const DAG_NODES: usize = 1_000_000;
/// Nodes for the built-in `WorkGraph` scheduler (O(n²), kept small on purpose).
const SCHED_NODES: usize = 2_000;
/// 1 MiB chunks for the memory-pressure walk.
const MEM_CHUNK: usize = 1024 * 1024;
/// Small objects for the metadata-heavy storage workload.
const META_OBJECTS: usize = 200_000;
/// Payloads (and their size) for the security-overhead comparison.
const SEC_PAYLOADS: usize = 4_000;
const SEC_PAYLOAD_BYTES: usize = 4096;
/// Iterations for the developer-workload proxies.
const DEV_PARSES: usize = 20_000;
const DEV_EVALS: usize = 50_000;
const DEV_COMPILES: usize = 100_000;
/// Logical cores + message rounds for the distributed-substrate benchmark.
const DIST_CORES: usize = 16;
const DIST_ROUNDS: u64 = 2_000_000;
/// ML matmul tile size for the throughput sweep (N×N · N×N).
const ML_MATMUL_N: usize = 128;
const ML_MATMUL_ITERS: u64 = 40;
/// Training/inference workload sizes.
const ML_TRAIN_EPOCHS: u64 = 2_000;
const ML_INFER_ITERS: u64 = 20_000;

// ── validation battery knobs (the `qemu_validate` feature) ──
/// Soak duration in milliseconds. Default ~8 s for a quick run; raise this (it is
/// the single knob) to run a real multi-hour soak — e.g. 3_600_000 for one hour.
/// Overridable at boot via the `DOMINION_SOAK_MS` value baked by the launcher.
const SOAK_MS: u64 = 8_000;
/// Working-set sizes (KiB) for the memory-latency mountain. Spans L1→L2→L3→DRAM on
/// real hardware (whpx); under TCG the steps flatten into emulation overhead.
const MOUNTAIN_WS_KIB: &[usize] = &[16, 48, 256, 1024, 4096, 16384, 65536, 262144];
/// Pointer-chase steps per working-set size (defeats the prefetcher via a random cycle).
const MOUNTAIN_STEPS: u64 = 40_000_000;

// ─────────────────────────── clock ───────────────────────────

#[inline]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Program the PIT (channel 0, mode 3) to `hz`, matching `desktop::set_timer_hz`.
fn set_timer_hz(hz: u32) {
    use x86_64::instructions::port::Port;
    let divisor = (1_193_182 / hz.max(1)).clamp(1, 65535) as u16;
    unsafe {
        let mut cmd: Port<u8> = Port::new(0x43);
        cmd.write(0x36);
        let mut data: Port<u8> = Port::new(0x40);
        data.write((divisor & 0xff) as u8);
        data.write((divisor >> 8) as u8);
    }
}

/// A TSC-based monotonic clock, calibrated against the PIT.
struct Clock {
    hz: u64,
}

impl Clock {
    /// Calibrate the TSC by busy-waiting a known number of PIT ticks. This also
    /// confirms the timer is actually firing (a stuck PIT would hang here, which is
    /// exactly the liveness signal we want before timing anything).
    fn calibrate() -> Clock {
        set_timer_hz(1000); // 1 kHz: each tick is 1 ms
        let t0 = rdtsc();
        let k0 = crate::keyboard::ticks();
        // Wait 500 ticks ≈ 0.5 s — long enough to swamp interrupt jitter.
        while crate::keyboard::ticks().wrapping_sub(k0) < 500 {
            x86_64::instructions::hlt();
        }
        let elapsed = rdtsc().wrapping_sub(t0).max(1);
        // 500 ticks at 1 kHz = 0.5 s, so hz = elapsed * 2.
        Clock { hz: elapsed.saturating_mul(2) }
    }

    #[inline]
    fn now(&self) -> u64 {
        rdtsc()
    }

    /// Microseconds for a cycle delta.
    fn us(&self, cycles: u64) -> u64 {
        (cycles as u128 * 1_000_000 / self.hz as u128) as u64
    }

    /// Milliseconds (×1000, i.e. returns integer ms) for a cycle delta.
    fn ms(&self, cycles: u64) -> u64 {
        (cycles as u128 * 1_000 / self.hz as u128) as u64
    }

    /// Operations per second given a count and a cycle delta.
    fn per_sec(&self, count: u64, cycles: u64) -> u64 {
        if cycles == 0 {
            return 0;
        }
        (count as u128 * self.hz as u128 / cycles as u128) as u64
    }
}

// ─────────────────────────── memory accounting ───────────────────────────

/// Bytes currently allocated on the kernel heap.
fn heap_used() -> usize {
    crate::allocator::total_bytes().saturating_sub(crate::allocator::free_bytes())
}

// A tiny deterministic LCG so the "random" access pattern is reproducible without
// touching the entropy source (and without `Math.random`-style nondeterminism).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
}

// ─────────────────────────── 0. native thread pool ───────────────────────────────

fn bench_pool(clk: &Clock) {
    crate::threadpool::bench_pool(clk.hz);
}

// ─────────────────────────── 1. process / task creation ───────────────────────────

fn bench_task_creation(clk: &Clock) {
    serial_println!("[bench] (1/9) process/task creation …");
    let base_used = heap_used();

    // ── creation: spawn TASKS lightweight domains (each owns a fresh capability) ──
    let mut sched = Scheduler::new();
    let t0 = clk.now();
    for i in 0..TASKS {
        let cap = Capability::mint((i as u64) * 0x1000, 0x1000, Rights::ALL);
        sched.spawn("", cap);
    }
    let create = clk.now().wrapping_sub(t0);
    let peak_used = heap_used();
    let per_task_bytes = (peak_used.saturating_sub(base_used)) / TASKS.max(1);

    serial_println!(
        "BENCH task_creation spawned={} create_ms={} spawn_per_s={} peak_kib={} bytes_per_task={}",
        TASKS,
        clk.ms(create),
        clk.per_sec(TASKS as u64, create),
        peak_used / 1024,
        per_task_bytes
    );

    // ── completion: run a feasible subset to completion (dispatch is O(n)) ──
    let mut s2 = Scheduler::new();
    for i in 0..TASKS_COMPLETE {
        s2.spawn("", Capability::mint((i as u64) * 0x1000, 0x1000, Rights::ALL));
    }
    let c0 = clk.now();
    let mut dispatched: u64 = 0;
    while let Some(id) = s2.next() {
        s2.finish(id); // one step then done — the minimal task lifecycle
        dispatched += 1;
    }
    let complete = clk.now().wrapping_sub(c0);
    serial_println!(
        "BENCH task_completion completed={} complete_ms={} dispatch_per_s={} note=dispatch_is_O(n)_per_step",
        dispatched,
        clk.ms(complete),
        clk.per_sec(dispatched, complete)
    );
}

// ─────────────────────────── 2. message passing ───────────────────────────

fn bench_message_passing(clk: &Clock) {
    serial_println!("[bench] (2/9) message passing …");
    let mut s = Scheduler::new();
    let a = s.spawn("a", Capability::mint(0, 0x1000, Rights::ALL));
    let b = s.spawn("b", Capability::mint(0x1000, 0x1000, Rights::ALL));
    s.open_channel(a, b).unwrap();
    let payload: Hash256 = Hash256::of(b"benchmark message payload");

    // Ping-pong so the inbox never grows: send one, receive one. This measures the
    // zero-copy IPC round-trip on the hot path with constant memory.
    let t0 = clk.now();
    let mut delivered: u64 = 0;
    for _ in 0..MSGS {
        let _ = s.send(a, b, payload);
        if s.recv(b).is_some() {
            delivered += 1;
        }
    }
    let dt = clk.now().wrapping_sub(t0);
    let thr = clk.per_sec(delivered, dt);
    // Latency per message in nanoseconds.
    let lat_ns = if delivered > 0 {
        (dt as u128 * 1_000_000_000 / (delivered as u128 * clk.hz as u128)) as u64
    } else {
        0
    };
    // Extrapolate to the "billions" figure: time to pass 1e9 messages at this rate.
    let billion_ms = (1_000_000_000u64 * 1000).checked_div(thr).unwrap_or(0);

    serial_println!(
        "BENCH message_passing delivered={} throughput_msg_per_s={} latency_ns={} projected_1e9_ms={}",
        delivered, thr, lat_ns, billion_ms
    );
}

// ─────────────────────────── 3. graph (DAG) execution ───────────────────────────

// A primitive compute node mirroring the DCG model (`dominion_core::dcg::Node`): a
// constant, or a binary op over two earlier nodes. Keeping deps to earlier indices
// makes the graph a DAG whose topological order is just ascending index, so it
// evaluates in a single O(n) pass — the linear-time execution a real DAG scheduler
// (make/bazel/Airflow) targets.
enum DagNode {
    Const(i64),
    Add(usize, usize),
    Mul(usize, usize),
}

fn bench_graph_execution(clk: &Clock) {
    serial_println!("[bench] (3/9) graph (DAG) execution …");

    // Build a DAG_NODES-node graph: two seeds, then each node combines two earlier
    // ones, giving a wide dependency fan-in.
    let mut nodes: Vec<DagNode> = Vec::with_capacity(DAG_NODES);
    nodes.push(DagNode::Const(1));
    nodes.push(DagNode::Const(2));
    let mut lcg = Lcg(0x1234_5678);
    for i in 2..DAG_NODES {
        let x = (lcg.next() as usize) % i;
        let y = (lcg.next() as usize) % i;
        if i & 1 == 0 {
            nodes.push(DagNode::Add(x, y));
        } else {
            nodes.push(DagNode::Mul(x, y));
        }
    }

    // Execute: single linear pass, memoised values (wrapping arithmetic so the
    // numbers stay bounded — we are timing the schedule+dispatch, not the result).
    let t0 = clk.now();
    let mut val: Vec<i64> = Vec::with_capacity(nodes.len());
    for n in &nodes {
        let v = match *n {
            DagNode::Const(c) => c,
            DagNode::Add(x, y) => val[x].wrapping_add(val[y]),
            DagNode::Mul(x, y) => val[x].wrapping_mul(val[y]),
        };
        val.push(v);
    }
    let dt = clk.now().wrapping_sub(t0);
    let root = *val.last().unwrap_or(&0);
    serial_println!(
        "BENCH graph_execution model=dcg_linear nodes={} exec_ms={} nodes_per_s={} root={}",
        nodes.len(),
        clk.ms(dt),
        clk.per_sec(nodes.len() as u64, dt),
        root
    );

    // Also exercise the *built-in* WorkGraph scheduler at small N. It is O(n²)
    // today (see FINDINGS), so we keep N small and report its rate to quantify the
    // gap against the linear evaluator above.
    use dominion_core::multikernel::{NodeKind, WorkGraph};
    let mut wg = WorkGraph::new();
    let mut prev = wg.add("seed", NodeKind::Cpu, &[]);
    for _ in 1..SCHED_NODES {
        prev = wg.add("op", NodeKind::Cpu, &[prev]);
    }
    let avail = [NodeKind::Cpu, NodeKind::Gpu, NodeKind::Npu];
    let s0 = clk.now();
    let scheduled = wg.schedule(&avail).map(|v| v.len()).unwrap_or(0);
    let sd = clk.now().wrapping_sub(s0);
    serial_println!(
        "BENCH graph_scheduler builtin=WorkGraph nodes={} schedule_us={} nodes_per_s={} note=O(n^2)",
        scheduled,
        clk.us(sd),
        clk.per_sec(scheduled as u64, sd)
    );
}

// ─────────────────────────── 4. memory pressure ───────────────────────────

fn bench_memory_pressure(clk: &Clock) {
    serial_println!("[bench] (4/9) memory pressure (allocate to OOM, fallibly) …");

    // Pre-size the holder with try_reserve so growing it never triggers the
    // infallible allocator (which would abort instead of letting us observe OOM).
    let cap_hint = crate::allocator::total_bytes() / MEM_CHUNK + 16;
    let mut holder: Vec<Vec<u8>> = Vec::new();
    if holder.try_reserve_exact(cap_hint).is_err() {
        serial_println!("BENCH memory_pressure error=holder_reserve_failed");
        return;
    }

    // Allocate 1 MiB at a time via the *fallible* API until the heap is exhausted.
    // Because allocation is fallible, hitting the ceiling is graceful — no panic.
    let t0 = clk.now();
    let mut chunks: u64 = 0;
    loop {
        let mut v: Vec<u8> = Vec::new();
        if v.try_reserve_exact(MEM_CHUNK).is_err() {
            break; // OOM reached, handled gracefully
        }
        v.resize(MEM_CHUNK, 0xA5); // touch it so it is really committed
        v[0] = 1;
        holder.push(v);
        chunks += 1;
    }
    let fill = clk.now().wrapping_sub(t0);
    let peak_mib = chunks; // 1 MiB chunks

    // Recovery: the workload survived OOM. Free half and prove we can allocate
    // again — i.e. work continues after pressure. Measure recovery latency.
    let survivor_check = holder.iter().all(|c| c[0] == 1);
    let r0 = clk.now();
    let keep = holder.len() / 2;
    holder.truncate(keep);
    // A fresh allocation must now succeed: the system is functional post-OOM.
    let mut recovered = false;
    let mut probe: Vec<u8> = Vec::new();
    if probe.try_reserve_exact(MEM_CHUNK).is_ok() {
        probe.resize(MEM_CHUNK, 7);
        recovered = probe[0] == 7;
    }
    let recovery = clk.now().wrapping_sub(r0);

    serial_println!(
        "BENCH memory_pressure peak_mib={} fill_ms={} oom_graceful=1 workload_survived={} recovered={} recovery_us={}",
        peak_mib,
        clk.ms(fill),
        survivor_check as u8,
        recovered as u8,
        clk.us(recovery)
    );
    // Drop everything before the next benchmark needs the heap back.
    drop(holder);
    drop(probe);
}

// ─────────────────────────── 5. storage ───────────────────────────

fn bench_storage(clk: &Clock) {
    serial_println!("[bench] (5/9) storage (sequential / random / metadata) …");

    crate::block::with_block_device(|dev: &mut dyn BlockDevice, real: bool| {
        let total = dev.block_count();
        // Leave block 0 for the superblock; cap the working set to the device.
        let blocks = core::cmp::min(total.saturating_sub(1), 20_000) as u64;
        if blocks == 0 {
            serial_println!("BENCH storage error=no_blocks real={}", real as u8);
            return;
        }
        let buf = [0xC3u8; BLOCK_SIZE];
        let mut rbuf = [0u8; BLOCK_SIZE];
        let bytes = blocks * BLOCK_SIZE as u64;

        // Sequential write.
        let t0 = clk.now();
        for lba in 1..=blocks {
            let _ = dev.write_block(lba, &buf);
        }
        let seq_w = clk.now().wrapping_sub(t0);

        // Sequential read.
        let t1 = clk.now();
        for lba in 1..=blocks {
            let _ = dev.read_block(lba, &mut rbuf);
        }
        let seq_r = clk.now().wrapping_sub(t1);

        // Random write + read (reproducible LCG-driven LBAs).
        let mut lcg = Lcg(0xDEAD_BEEF);
        let t2 = clk.now();
        for _ in 0..blocks {
            let lba = 1 + (lcg.next() % blocks);
            let _ = dev.write_block(lba, &buf);
        }
        let rnd_w = clk.now().wrapping_sub(t2);
        let t3 = clk.now();
        for _ in 0..blocks {
            let lba = 1 + (lcg.next() % blocks);
            let _ = dev.read_block(lba, &mut rbuf);
        }
        let rnd_r = clk.now().wrapping_sub(t3);

        // KiB/s (not MiB/s): the synchronous single-request virtio-blk driver does
        // ~1-2k write IOPS, which is well under 1 MiB/s and would round to 0.
        let kib_per_s = |cycles: u64| -> u64 {
            if cycles == 0 {
                0
            } else {
                (bytes as u128 * clk.hz as u128 / (cycles as u128 * 1024)) as u64
            }
        };
        let iops = |cycles: u64| clk.per_sec(blocks, cycles);

        serial_println!(
            "BENCH storage_sequential real={} blocks={} write_kib_s={} read_kib_s={} write_iops={} read_iops={}",
            real as u8, blocks, kib_per_s(seq_w), kib_per_s(seq_r), iops(seq_w), iops(seq_r)
        );
        serial_println!(
            "BENCH storage_random real={} blocks={} write_kib_s={} read_kib_s={} write_iops={} read_iops={}",
            real as u8, blocks, kib_per_s(rnd_w), kib_per_s(rnd_r), iops(rnd_w), iops(rnd_r)
        );
    });

    // Metadata-heavy: insert many small content-addressed objects (the OS's
    // filesystem replacement), then persist the whole graph to disk.
    let mut g = ObjectGraph::new();
    let t0 = clk.now();
    for i in 0..META_OBJECTS {
        g.put(Object::new("Node").with("i", Datum::Int(i as i64)));
    }
    let put_dt = clk.now().wrapping_sub(t0);
    let stored = g.stored_count();
    serial_println!(
        "BENCH storage_metadata objects={} stored={} put_per_s={} put_ms={}",
        META_OBJECTS,
        stored,
        clk.per_sec(META_OBJECTS as u64, put_dt),
        clk.ms(put_dt)
    );

    // Persist that graph (serialize + write) to whatever device is present. Size
    // the payload up front (outside the timed region) so we can skip a device that
    // is too small rather than reporting a misleading failure.
    let payload_blocks = (g.serialize().len() / BLOCK_SIZE + 2) as u64;
    crate::block::with_block_device(|dev: &mut dyn BlockDevice, real: bool| {
        if dev.block_count() < payload_blocks {
            serial_println!(
                "BENCH storage_persist real={} skipped=device_too_small need_blocks={}",
                real as u8, payload_blocks
            );
            return;
        }
        let t0 = clk.now();
        let ok = Persistence::save(dev, &g).is_ok();
        let dt = clk.now().wrapping_sub(t0);
        serial_println!(
            "BENCH storage_persist real={} ok={} save_ms={}",
            real as u8, ok as u8, clk.ms(dt)
        );
    });
}

// ─────────────────────────── 6. distributed substrate ───────────────────────────

fn bench_distributed(clk: &Clock) {
    serial_println!("[bench] (6/9) distributed substrate (inter-core msgs + CRDT) …");

    // Inter-core message passing: DominionOS's multikernel moves state only as
    // explicit messages between cores. We measure that round-trip on the hot path.
    let mut mk = Multikernel::new(DIST_CORES);
    let t0 = clk.now();
    let mut moved: u64 = 0;
    for r in 0..DIST_ROUNDS {
        let from = (r as usize % DIST_CORES) as u64;
        let to = ((r as usize + 1) % DIST_CORES) as u64;
        mk.send(from, to, "k", r as i64, r);
        mk.deliver(to, r);
        moved += 1;
    }
    let dt = clk.now().wrapping_sub(t0);
    serial_println!(
        "BENCH distributed_messaging cores={} messages={} msg_per_s={} note=single_node_substrate",
        DIST_CORES,
        moved,
        clk.per_sec(moved, dt)
    );

    // CRDT convergence: independent per-core counters merged to a consistent value
    // — the conflict-free state replication a real cluster relies on.
    let rounds: u64 = 1_000_000;
    let t1 = clk.now();
    let mut acc = ConvergentState::new();
    for r in 0..rounds {
        let mut other = ConvergentState::new();
        other.bump(r % DIST_CORES as u64, 1);
        acc = acc.merge(&other);
    }
    let mt = clk.now().wrapping_sub(t1);
    serial_println!(
        "BENCH distributed_crdt merges={} merge_per_s={} converged_value={}",
        rounds,
        clk.per_sec(rounds, mt),
        acc.value()
    );
}

// ─────────────────────────── 7. security overhead ───────────────────────────

fn bench_security(clk: &Clock) {
    serial_println!("[bench] (7/9) security overhead (encryption on vs off) …");

    // Build the payloads once (distinct so dedup doesn't collapse the work).
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(SEC_PAYLOADS);
    for i in 0..SEC_PAYLOADS {
        let mut p = alloc::vec![0u8; SEC_PAYLOAD_BYTES];
        p[0] = (i & 0xff) as u8;
        p[1] = ((i >> 8) & 0xff) as u8;
        payloads.push(p);
    }
    let mib = (SEC_PAYLOADS * SEC_PAYLOAD_BYTES) as u64;

    // ── encryption OFF: store as plaintext content-addressed bytes ──
    let mut g = ObjectGraph::new();
    let t0 = clk.now();
    for p in &payloads {
        g.put(Object::new("Blob").with("d", Datum::Bytes(p.clone())));
    }
    let plain = clk.now().wrapping_sub(t0);

    // ── encryption ON (ChaCha20-Poly1305 — the default, PQ-resilient suite) ──
    let key = Key::from_seed(b"bench-key");
    let index_key = Key::from_seed(b"bench-index");
    let nonce = [0x11u8; 16];
    let mut vault = Vault::new();
    let t1 = clk.now();
    for p in &payloads {
        vault.seal_with(CipherSuite::ChaCha20Poly1305, p, key, &nonce, &index_key, &[]);
    }
    let chacha = clk.now().wrapping_sub(t1);

    // ── encryption ON (AES-256-GCM — the independent second suite) ──
    let mut vault2 = Vault::new();
    let t2 = clk.now();
    for p in &payloads {
        vault2.seal_with(CipherSuite::Aes256Gcm, p, key, &nonce, &index_key, &[]);
    }
    let gcm = clk.now().wrapping_sub(t2);

    let mibps = |cycles: u64| -> u64 {
        if cycles == 0 {
            0
        } else {
            (mib as u128 * clk.hz as u128 / (cycles as u128 * 1024 * 1024)) as u64
        }
    };
    // Overhead as a percentage: (enc - plain) / plain * 100.
    let overhead_pct = |enc: u64| -> u64 {
        if plain == 0 {
            0
        } else {
            ((enc.saturating_sub(plain)) as u128 * 100 / plain as u128) as u64
        }
    };

    serial_println!(
        "BENCH security_overhead payloads={} payload_bytes={} plaintext_mib_s={} chacha20poly1305_mib_s={} aesgcm_mib_s={} chacha20poly1305_overhead_pct={} aesgcm_overhead_pct={}",
        SEC_PAYLOADS, SEC_PAYLOAD_BYTES,
        mibps(plain), mibps(chacha), mibps(gcm),
        overhead_pct(chacha), overhead_pct(gcm)
    );
}

// ─────────────────────────── 8. developer workloads ───────────────────────────

fn bench_developer(clk: &Clock) {
    serial_println!("[bench] (8/9) developer workloads (parse / eval / compile) …");

    // "Build a large codebase" proxy: front-end parse throughput. We parse a
    // small module repeatedly — the lexer+parser is the build front-end.
    let src = "fn g(a, b) { let s = a + b; return s * a; } \
               fn h(x) { if x < 2 { return 1; } return x * h(x - 1); } g(3, 4)";
    let t0 = clk.now();
    let mut parsed_ok: u64 = 0;
    for _ in 0..DEV_PARSES {
        if parse_source(src).is_ok() {
            parsed_ok += 1;
        }
    }
    let pt = clk.now().wrapping_sub(t0);

    // "Run a test suite" proxy: interpret a recursive function many times. Each
    // eval is an independent program run, like a test case executing.
    let mut acc: i64 = 0;
    let t1 = clk.now();
    let mut evals: u64 = 0;
    for _ in 0..DEV_EVALS {
        let mut it = Interpreter::new();
        if let Ok(Value::Int(n)) =
            it.eval_str("fn f(n){ if n<2 {return 1;} return n*f(n-1);} f(10)")
        {
            acc = acc.wrapping_add(n);
            evals += 1;
        }
    }
    let et = clk.now().wrapping_sub(t1);

    // "Dependency resolution / typecheck" proxy: compile a function to a DCG. The
    // lowering is capability-checked (the resolve+authority step) and proof-
    // carrying, exactly the work a build's resolve phase does per unit.
    let prog = parse_source("fn g(a, b) { let s = a + b; return s * a; }").unwrap();
    let mut compiled: u64 = 0;
    let mut cacc: i64 = 0;
    let t2 = clk.now();
    if let Item::Fn(f) = &prog.items[0] {
        use dominion_core::dcg::Dcg;
        for _ in 0..DEV_COMPILES {
            if let Ok(d) = Dcg::compile(f, Rights::ALL) {
                cacc = cacc.wrapping_add(d.eval(&[3, 4]).unwrap_or(0));
                compiled += 1;
            }
        }
    }
    let ct = clk.now().wrapping_sub(t2);

    serial_println!(
        "BENCH dev_build parses={} parse_per_s={} parse_ms={}",
        parsed_ok, clk.per_sec(parsed_ok, pt), clk.ms(pt)
    );
    serial_println!(
        "BENCH dev_test_suite evals={} eval_per_s={} eval_ms={} checksum={}",
        evals, clk.per_sec(evals, et), clk.ms(et), acc
    );
    serial_println!(
        "BENCH dev_depresolve compiles={} compile_per_s={} compile_ms={} checksum={}",
        compiled, clk.per_sec(compiled, ct), clk.ms(ct), cacc
    );
}

// ── CPUID: hardware hints for MlConfig auto-tuning ───────────────────────
//
// Returns (l1d_kb, n_logical_cores).  Uses CPUID leaf 4/0x8000001D for cache
// and leaf 1 for core count.  Falls back to safe defaults on any unexpected
// topology.  Unsafe because `__cpuid` is a raw hardware instruction; the only
// side-effect is reading CPU registers.

fn cpuid_ml_hints() -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::__cpuid_count;

        // ── L1d cache size ─────────────────────────────────────────────────
        // CPUID leaf 4, sub-leaf 0: "Deterministic Cache Parameters".
        // Bits [4:0] of EAX = cache type: 1=data, 2=instruction, 3=unified.
        // Bits [7:5] of EAX = cache level: 1=L1.
        // Size = (ways+1) * (partitions+1) * (line_size+1) * (sets+1)
        let l4 = __cpuid_count(4, 0);
        let cache_type = l4.eax & 0x1F;
        let cache_level = (l4.eax >> 5) & 0x7;
        let l1d_kb = if cache_type != 0 && cache_level == 1 {
            let line_size  = (l4.ebx & 0xFFF) as usize + 1;      // bits 11:0
            let partitions = ((l4.ebx >> 12) & 0x3FF) as usize + 1; // bits 21:12
            let ways       = ((l4.ebx >> 22) & 0x3FF) as usize + 1; // bits 31:22
            let sets       = l4.ecx as usize + 1;
            (line_size * partitions * ways * sets) / 1024
        } else {
            32 // safe default: 32 KiB L1d
        };

        // ── logical core count ─────────────────────────────────────────────
        // CPUID leaf 1 EBX[23:16] = max addressable logical processor IDs.
        let l1 = __cpuid_count(1, 0);
        let max_logical = ((l1.ebx >> 16) & 0xFF) as usize;
        let n_cores = if max_logical > 0 { max_logical } else { 1 };

        (l1d_kb.max(8).min(4096), n_cores.max(1).min(256))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (32, 1)
    }
}

// ── (11) machine learning: matmul throughput, training, inference, int8 ──

fn bench_ml(clk: &Clock) {
    use dominion_core::datatypes::Tensor;
    use dominion_core::memo::{BoundedTensorMemo, TensorMemo};
    use dominion_core::ml::{
        self, binmatmul, q4matmul, quantize, quantize_bin, quantize_q4, quantize_tern,
        qmatmul, recommend_device, ternmatmul, Activation, ComputeBandit, Device,
        Mlp, Optimizer, Precision, ALL_PRECISIONS,
    };
    use dominion_core::parallel::Spawn;
    use crate::threadpool::KernelSpawn;

    serial_println!("[bench] (11/11) machine learning (levers 1–8: multicore / fusion / cache / quant / bandit / simd / fed / memo) …");

    // Probe hardware and build MlConfig so downstream code knows optimal tile/thread counts.
    let (l1d_kb, hw_cores) = cpuid_ml_hints();
    {
        use dominion_core::ml::MlConfig;
        let hw_cfg = MlConfig::from_hardware(l1d_kb, hw_cores);
        serial_println!(
            "BENCH ml_hw_config l1d_kb={} hw_cores={} tile_size={} cache_policy=lru64 fma=false",
            l1d_kb, hw_cores, hw_cfg.effective_tile()
        );
    }

    let n = ML_MATMUL_N;
    let mut lcg = Lcg(0x5151_4D4C_0000_0001);
    let a = Tensor::new(
        alloc::vec![n, n],
        (0..n * n).map(|_| (lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5).collect(),
    ).unwrap();
    let b = a.clone();

    // ── Lever 1+2: single-core baseline ───────────────────────────────────────
    let ref_serial = a.matmul(&b).unwrap();
    let mut sink = 0.0f64;
    let t0 = clk.now();
    let mut done: u64 = 0;
    for _ in 0..ML_MATMUL_ITERS {
        let c = a.matmul(&b).unwrap();
        sink += c.data()[0];
        done += 1;
    }
    let dt = clk.now().wrapping_sub(t0);
    let flops = ml::matmul_flops(n, n, n) * done;
    let mflops = if clk.us(dt) > 0 { flops / clk.us(dt) } else { 1 };
    black_box(sink);
    serial_println!(
        "BENCH ml_matmul n={} iters={} mflop_per_s={} gflop_per_s={} ms={}",
        n, done, mflops, mflops / 1000, clk.ms(dt)
    );

    // ── Lever 1: multi-core matmul via KernelSpawn (all APs, bit-identical) ───
    let workers = KernelSpawn.max_workers();
    let mc_check = a.matmul_with(&b, &KernelSpawn).unwrap();
    let bit_identical = mc_check.data() == ref_serial.data();
    let mut mc_sink = 0.0f64;
    let t_mc = clk.now();
    let mut mc_done: u64 = 0;
    for _ in 0..ML_MATMUL_ITERS {
        let c = a.matmul_with(&b, &KernelSpawn).unwrap();
        mc_sink += c.data()[0];
        mc_done += 1;
    }
    let mc_dt = clk.now().wrapping_sub(t_mc);
    let mc_mflops = if clk.us(mc_dt) > 0 { (ml::matmul_flops(n, n, n) * mc_done) / clk.us(mc_dt) } else { 0 };
    black_box(mc_sink);
    // speedup_pct: 100 = same speed, 200 = 2× faster.
    let mc_speedup_pct = mc_mflops * 100 / mflops;
    serial_println!(
        "BENCH ml_multicore_matmul n={} workers={} mflop_per_s={} gflop_per_s={} bit_identical={} speedup_pct={} ms={}",
        n, workers, mc_mflops, mc_mflops / 1000, bit_identical, mc_speedup_pct, clk.ms(mc_dt)
    );

    // ── Lever 2: operator fusion — fused forward vs manual chain ──────────────
    // Fused: matmul_bias_act in one pass (one alloc, one data pass per layer).
    {
        let farch: &[usize] = &[2, 16, 8, 1];
        let fmodel = Mlp::new(farch, Activation::Relu, Activation::Identity, 0xF001).unwrap();
        let xin = Tensor::new(alloc::vec![1, 2], alloc::vec![0.1f64, 0.9]).unwrap();

        let tf = clk.now();
        let mut fsink = 0.0f64;
        for _ in 0..ML_INFER_ITERS {
            let out = fmodel.forward(&xin).unwrap();
            fsink += out.data()[0];
        }
        let fd = clk.now().wrapping_sub(tf);
        black_box(fsink);
        serial_println!(
            "BENCH ml_fused_infer arch=2x16x8x1 iters={} infer_per_s={} ms={}",
            ML_INFER_ITERS, clk.per_sec(ML_INFER_ITERS, fd), clk.ms(fd)
        );

        // Fused + KernelSpawn: each matmul layer distributes rows across APs.
        let tm = clk.now();
        let mut msink = 0.0f64;
        for _ in 0..ML_INFER_ITERS {
            let out = fmodel.forward_with(&xin, &KernelSpawn).unwrap();
            msink += out.data()[0];
        }
        let md = clk.now().wrapping_sub(tm);
        black_box(msink);
        serial_println!(
            "BENCH ml_multicore_infer arch=2x16x8x1 workers={} iters={} infer_per_s={} ms={}",
            workers, ML_INFER_ITERS, clk.per_sec(ML_INFER_ITERS, md), clk.ms(md)
        );
    }

    // ── Lever 2: device placement model ──────────────────────────────────────
    let one = ml::matmul_flops(n, n, n);
    serial_println!(
        "BENCH ml_placement n={} chosen={} cpu_cyc={} gpu_cyc={} npu_cyc={} tpu_cyc={}",
        n,
        recommend_device(one).name(),
        Device::Cpu.est_cycles(one),
        Device::Gpu.est_cycles(one),
        Device::Npu.est_cycles(one),
        Device::Tpu.est_cycles(one),
    );

    // ── Lever 3: training throughput ──────────────────────────────────────────
    let (x, y) = ml::xor_dataset();
    let mut model = Mlp::new(&[2, 16, 1], Activation::Tanh, Activation::Sigmoid, 0xA17E).unwrap();
    let mut opt = Optimizer::adam(0.05);
    let mut last_loss = 1.0;
    let t1 = clk.now();
    let mut steps: u64 = 0;
    for _ in 0..ML_TRAIN_EPOCHS {
        last_loss = model.train_step_mse(&x, &y, &mut opt).unwrap();
        steps += 1;
    }
    let tt = clk.now().wrapping_sub(t1);
    serial_println!(
        "BENCH ml_train steps={} steps_per_s={} final_loss_micro={} ms={}",
        steps, clk.per_sec(steps, tt), (last_loss * 1_000_000.0) as i64, clk.ms(tt)
    );

    // ── Lever 3: inference throughput ─────────────────────────────────────────
    let mut isink = 0.0f64;
    let t2 = clk.now();
    let mut infers: u64 = 0;
    for _ in 0..ML_INFER_ITERS {
        let out = model.forward(&x).unwrap();
        isink += out.data()[0];
        infers += 1;
    }
    let it = clk.now().wrapping_sub(t2);
    black_box(isink);
    serial_println!(
        "BENCH ml_infer passes={} infer_per_s={} ms={}",
        infers, clk.per_sec(infers, it), clk.ms(it)
    );

    // ── Lever 4: int8 quantization (NPU path) ─────────────────────────────────
    let (qa, qb) = (quantize(&a), quantize(&b));
    let mut qsink = 0.0f64;
    let t3 = clk.now();
    let mut qdone: u64 = 0;
    for _ in 0..ML_MATMUL_ITERS {
        let c = qmatmul(&qa, &qb).unwrap();
        qsink += c.data()[0];
        qdone += 1;
    }
    let qt = clk.now().wrapping_sub(t3);
    let qmflops = if clk.us(qt) > 0 { (ml::matmul_flops(n, n, n) * qdone) / clk.us(qt) } else { 0 };
    black_box(qsink);
    serial_println!(
        "BENCH ml_int8_matmul n={} iters={} mop_per_s={} ms={}",
        n, qdone, qmflops, clk.ms(qt)
    );

    // ── Lever 4: full precision ladder — int4, binary, ternary ───────────────
    let (qa4, qb4) = (quantize_q4(&a), quantize_q4(&b));
    let mut q4sink = 0.0f64;
    let tq4 = clk.now();
    let mut q4done: u64 = 0;
    for _ in 0..ML_MATMUL_ITERS {
        let c = q4matmul(&qa4, &qb4).unwrap();
        q4sink += c.data()[0];
        q4done += 1;
    }
    let q4t = clk.now().wrapping_sub(tq4);
    let q4mflops = if clk.us(q4t) > 0 { (ml::matmul_flops(n, n, n) * q4done) / clk.us(q4t) } else { 0 };
    black_box(q4sink);
    serial_println!("BENCH ml_int4_matmul n={} iters={} mop_per_s={} ms={}", n, q4done, q4mflops, clk.ms(q4t));

    let (ba, bb) = (quantize_bin(&a), quantize_bin(&b));
    let mut bsink = 0.0f64;
    let tb = clk.now();
    let mut bdone: u64 = 0;
    for _ in 0..ML_MATMUL_ITERS {
        let c = binmatmul(&ba, &bb).unwrap();
        bsink += c.data()[0];
        bdone += 1;
    }
    let bt2 = clk.now().wrapping_sub(tb);
    let bin_mflops = if clk.us(bt2) > 0 { (ml::matmul_flops(n, n, n) * bdone) / clk.us(bt2) } else { 0 };
    black_box(bsink);
    serial_println!("BENCH ml_binary_matmul n={} iters={} mop_per_s={} ms={}", n, bdone, bin_mflops, clk.ms(bt2));

    let (ta2, tb2) = (quantize_tern(&a), quantize_tern(&b));
    let mut tsink2 = 0.0f64;
    let tt3 = clk.now();
    let mut tdone: u64 = 0;
    for _ in 0..ML_MATMUL_ITERS {
        let c = ternmatmul(&ta2, &tb2).unwrap();
        tsink2 += c.data()[0];
        tdone += 1;
    }
    let tt4 = clk.now().wrapping_sub(tt3);
    let tern_mflops = if clk.us(tt4) > 0 { (ml::matmul_flops(n, n, n) * tdone) / clk.us(tt4) } else { 0 };
    black_box(tsink2);
    serial_println!("BENCH ml_ternary_matmul n={} iters={} mop_per_s={} ms={}", n, tdone, tern_mflops, clk.ms(tt4));

    serial_println!(
        "BENCH precision_ladder n={} f64_mflop={} int8_mop={} int4_mop={} ternary_mop={} binary_mop={}",
        n, mflops, qmflops, q4mflops, tern_mflops, bin_mflops
    );

    // ── Lever 5: adaptive precision via UCB-1 bandit ──────────────────────────
    {
        const BANDIT_ITERS: u64 = 100;
        let mut bandit = ComputeBandit::new(0xDEAD_CAFE);
        let qa_i8   = quantize(&a);
        let qa_i4   = quantize_q4(&a);
        let qa_bin  = quantize_bin(&a);
        let qa_tern = quantize_tern(&a);
        let flops_per_call = ml::matmul_flops(n, n, n);
        let mut converge_at = 0u64;
        for iter in 0..BANDIT_ITERS {
            let arm = bandit.select_ucb1();
            let prec = ALL_PRECISIONS[arm];
            let t1 = clk.now();
            match prec {
                Precision::F64     => { black_box(a.matmul(&a)); }
                Precision::Int8    => { black_box(qmatmul(&qa_i8, &qa_i8)); }
                Precision::Int4    => { black_box(q4matmul(&qa_i4, &qa_i4)); }
                Precision::Binary  => { black_box(binmatmul(&qa_bin, &qa_bin)); }
                Precision::Ternary => { black_box(ternmatmul(&qa_tern, &qa_tern)); }
            }
            let us = clk.us(clk.now().wrapping_sub(t1)).max(1);
            bandit.update(arm, (flops_per_call / us) as f64);
            if converge_at == 0 && bandit.is_converged() { converge_at = iter + 1; }
        }
        serial_println!(
            "BENCH ml_bandit iters={} converge_at={} best_precision={} epsilon_x1000={}",
            BANDIT_ITERS, converge_at, bandit.best_precision().name(),
            (bandit.epsilon() * 1000.0) as u64
        );
    }

    // ── Lever 7: federated training cost model (ring all-reduce) ─────────────
    {
        use dominion_core::ml::{fed_avg, mlp_params_flat, mlp_set_params_flat, ring_allreduce_cost};
        const N_WORKERS: usize = 4;
        const SYNC_EVERY: usize = 20;
        const ROUNDS: usize = 10; // keep short for in-kernel bench
        let arch: &[usize] = &[2, 16, 8, 1];
        let lr = 0.005f64;
        let mut workers_mlp: Vec<Mlp> = (0..N_WORKERS)
            .map(|_| Mlp::new(arch, Activation::Relu, Activation::Identity, 0xF001).unwrap())
            .collect();
        let mut opts: Vec<Optimizer> = (0..N_WORKERS).map(|_| Optimizer::adam(lr)).collect();
        let mut fed_lcg = Lcg(0xBEEF_0000_0001u64);
        let tf = clk.now();
        let mut fed_loss = 1.0f64;
        for _round in 0..ROUNDS {
            for (mlp, opt) in workers_mlp.iter_mut().zip(opts.iter_mut()) {
                for _ in 0..SYNC_EVERY {
                    let x1 = (fed_lcg.next() >> 40) as f64 / 16_777_216.0;
                    let x2 = (fed_lcg.next() >> 40) as f64 / 16_777_216.0;
                    let tgt_val = if (x1 > 0.5) != (x2 > 0.5) { 1.0 } else { 0.0 };
                    let inp = Tensor::new(alloc::vec![1, 2], alloc::vec![x1, x2]).unwrap();
                    let tgt = Tensor::new(alloc::vec![1, 1], alloc::vec![tgt_val]).unwrap();
                    fed_loss = mlp.train_step_mse(&inp, &tgt, opt).unwrap_or(1.0);
                }
            }
            let pvecs: Vec<alloc::vec::Vec<f64>> = workers_mlp.iter().map(mlp_params_flat).collect();
            if let Some(avg) = fed_avg(&pvecs) {
                for w in &mut workers_mlp { mlp_set_params_flat(w, &avg); }
            }
            for o in &mut opts { *o = Optimizer::adam(lr); }
        }
        let fed_dt = clk.now().wrapping_sub(tf);
        let param_count = mlp_params_flat(&workers_mlp[0]).len();
        let (ar_rounds, ar_msgs, ar_bytes) = ring_allreduce_cost(N_WORKERS, param_count);
        serial_println!(
            "BENCH ml_fed_training workers={} rounds={} loss_micro={} steps_per_s={} ms={}",
            N_WORKERS, ROUNDS, (fed_loss * 1_000_000.0) as i64,
            clk.per_sec((ROUNDS * N_WORKERS * SYNC_EVERY) as u64, fed_dt), clk.ms(fed_dt)
        );
        serial_println!(
            "BENCH ml_allreduce workers={} params={} rounds={} messages={} bytes_per_node={}",
            N_WORKERS, param_count, ar_rounds, ar_msgs, ar_bytes
        );
    }

    // ── Lever 8: inference caching (TensorMemo + BoundedTensorMemo LRU) ───────
    {
        let l8_arch: &[usize] = &[4, 16, 8, 1];
        let l8_mlp = Mlp::new(l8_arch, Activation::Relu, Activation::Identity, 0xCAFE_0001).unwrap();
        let model_hash = l8_mlp.content_hash();
        let cache_input = Tensor::new(alloc::vec![1, 4], alloc::vec![0.1f64, 0.2, 0.3, 0.4]).unwrap();
        const CACHE_ITERS: u64 = 500;

        // Bare forward (no cache) — baseline compute cost.
        let mut bare_sink = 0.0f64;
        let t_bare = clk.now();
        for _ in 0..CACHE_ITERS {
            let out = l8_mlp.forward(&cache_input).unwrap();
            bare_sink += out.data()[0];
        }
        let bare_dt = clk.now().wrapping_sub(t_bare);
        black_box(bare_sink);

        // TensorMemo warm hit (100% repeat → BTreeMap lookup + clone, skip compute).
        let mut tmemo = TensorMemo::new();
        let mut cached_sink = 0.0f64;
        let t_cached = clk.now();
        for _ in 0..CACHE_ITERS {
            let out = l8_mlp.forward_cached(&cache_input, model_hash, &mut tmemo).unwrap();
            cached_sink += out.data()[0];
        }
        let cached_dt = clk.now().wrapping_sub(t_cached);
        black_box(cached_sink);
        // speedup_pct: 100 = same, 500 = 5×.
        let cache_speedup_pct = if clk.us(cached_dt) > 0 {
            clk.us(bare_dt) * 100 / clk.us(cached_dt)
        } else { 99_900 };
        serial_println!(
            "BENCH ml_cache_warm iters={} bare_us={} cached_us={} speedup_pct={} hits={}",
            CACHE_ITERS, clk.us(bare_dt), clk.us(cached_dt), cache_speedup_pct, tmemo.hits()
        );

        // BoundedTensorMemo LRU: 90% repeat over pool of 32, cap=32.
        const LRU_CAP: usize = 32;
        const LRU_N: u64 = 1_000;
        let mut pool_lcg = Lcg(0xCAFE_BABE_0001u64);
        let input_pool: Vec<Tensor> = (0..LRU_CAP).map(|_| {
            let d: Vec<f64> = (0..4)
                .map(|_| (pool_lcg.next() >> 40) as f64 / 16_777_216.0 - 0.5)
                .collect();
            Tensor::new(alloc::vec![1, 4], d).unwrap()
        }).collect();
        let mut lru = BoundedTensorMemo::new(LRU_CAP);
        let mut seq_lcg = Lcg(0xBEEF_FACE_0001u64);
        let t_lru = clk.now();
        for _ in 0..LRU_N {
            let dice = seq_lcg.next() % 100;
            let inp = if dice < 90 {
                let idx = (seq_lcg.next() as usize) % LRU_CAP;
                &input_pool[idx]
            } else {
                &cache_input
            };
            let key = (model_hash, inp.content_hash());
            if lru.get(key).is_none() {
                if let Some(out) = l8_mlp.forward(inp) {
                    lru.insert(key, out);
                }
            }
        }
        let lru_dt = clk.now().wrapping_sub(t_lru);
        serial_println!(
            "BENCH ml_lru_cache cap={} iters={} hit_rate_pct={} us={}",
            LRU_CAP, LRU_N, lru.hit_rate_pct(), clk.us(lru_dt)
        );

        // Invalidation safety: model update → content_hash changes → stale keys unreachable.
        let mut l8_v2 = l8_mlp.clone();
        {
            let li = l8_v2.layers.len() - 1;
            let shape = l8_v2.layers[li].w.shape().to_vec();
            let new_w: Vec<f64> = l8_v2.layers[li].w.data().iter().map(|&v| v + 1e-4).collect();
            l8_v2.layers[li].w = Tensor::new(shape, new_w).unwrap();
        }
        let hash_v2 = l8_v2.content_hash();
        serial_println!(
            "BENCH ml_cache_invalidation hash_changed={} stale_entries_unreachable=true",
            model_hash != hash_v2
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION BATTERY (`--features qemu_validate`)
//
// The perf battery above measures *abstraction efficiency*. This battery answers
// the harder question — is any of this real on hardware, and does it survive
// stress and failure? Three things it pins down:
//   * the memory-latency mountain proves the cache hierarchy is REAL under whpx
//     (the guest runs natively on the host CPU via SLAT) and flattens under TCG;
//   * the model-vs-hardware boundary makes the gap between an in-memory op and a
//     hardware-mediated op (VM exit, device round-trip) explicit and measured;
//   * soak + chaos exercise sustained drift/fragmentation and deterministic
//     recovery under injected failure — the "infrastructure-grade" signal.
// (The cross-core scaling curve lives in `bench_scaling`, gated on real SMP.)
// ═══════════════════════════════════════════════════════════════════════════

// ── V1. memory-latency mountain (is the cache hierarchy real?) ──

fn bench_mem_hierarchy(clk: &Clock) {
    serial_println!("[validate] memory-latency mountain (pointer chase) …");
    serial_println!("[validate]   ws_kib  ns/access   (L1≈1ns L2≈4ns L3≈15ns DRAM≈80ns on real silicon)");
    for &ws_kib in MOUNTAIN_WS_KIB {
        let n = ws_kib * 1024 / core::mem::size_of::<usize>();
        if n < 2 {
            continue;
        }
        // Build a single random permutation cycle over [0, n): each slot points to
        // the next index to visit. A random cycle defeats the hardware prefetcher,
        // so every access is a dependent load that pays the true memory latency.
        let mut next: Vec<usize> = Vec::new();
        if next.try_reserve_exact(n).is_err() {
            serial_println!("BENCH mem_hierarchy ws_kib={} skipped=oom", ws_kib);
            continue;
        }
        next.resize(n, 0);
        // Fisher-Yates over an index list, then link consecutive picks into a cycle.
        let mut order: Vec<usize> = (0..n).collect();
        let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15 ^ (ws_kib as u64));
        for i in (1..n).rev() {
            let j = (lcg.next() as usize) % (i + 1);
            order.swap(i, j);
        }
        for i in 0..n {
            next[order[i]] = order[(i + 1) % n];
        }
        drop(order);

        let t0 = clk.now();
        let mut p = 0usize;
        for _ in 0..MOUNTAIN_STEPS {
            p = next[p];
        }
        let dt = clk.now().wrapping_sub(t0);
        black_box(p);
        let ns = (dt as u128 * 1_000_000_000 / (MOUNTAIN_STEPS as u128 * clk.hz as u128)) as u64;
        serial_println!("BENCH mem_hierarchy ws_kib={} ns_per_access={} steps={}", ws_kib, ns, MOUNTAIN_STEPS);
    }
}

// ── V2. model-vs-hardware boundary (where does abstraction end?) ──

fn bench_boundary(clk: &Clock) {
    serial_println!("[validate] model-vs-hardware boundary …");

    // In-memory ops (pure abstraction cost).
    let cap = Capability::mint(0x1000, 0x1000, Rights::ALL);
    let iters = 5_000_000u64;
    let t0 = clk.now();
    let mut acc = 0u64;
    for _ in 0..iters {
        if cap.check(0x1000, 8, Rights::READ).is_ok() {
            acc = acc.wrapping_add(1);
        }
    }
    let cap_ns = ns_per(clk, clk.now().wrapping_sub(t0), iters);
    black_box(acc);

    let mut s = Scheduler::new();
    let a = s.spawn("a", cap);
    let b = s.spawn("b", Capability::mint(0x2000, 0x1000, Rights::ALL));
    s.open_channel(a, b).unwrap();
    let pay = Hash256::of(b"x");
    let t1 = clk.now();
    for _ in 0..iters {
        let _ = s.send(a, b, pay);
        s.recv(b);
    }
    let msg_ns = ns_per(clk, clk.now().wrapping_sub(t1), iters);

    // Hardware-mediated: a port-0x80 write traps to the hypervisor (a real VM exit
    // under whpx); a virtio-blk read is a full device-queue round-trip. These cross
    // the abstraction boundary into the (virtualized) hardware.
    use x86_64::instructions::port::Port;
    let exit_iters = 200_000u64;
    let t2 = clk.now();
    unsafe {
        let mut diag: Port<u8> = Port::new(0x80);
        for _ in 0..exit_iters {
            diag.write(0);
        }
    }
    let exit_ns = ns_per(clk, clk.now().wrapping_sub(t2), exit_iters);

    let (dev_ns, dev_real) = crate::block::with_block_device(|dev: &mut dyn BlockDevice, real: bool| {
        let mut buf = [0u8; BLOCK_SIZE];
        let dev_iters = 2_000u64;
        let t = clk.now();
        for _ in 0..dev_iters {
            let _ = dev.read_block(0, &mut buf);
        }
        (ns_per(clk, clk.now().wrapping_sub(t), dev_iters), real)
    });

    serial_println!(
        "BENCH boundary_memory cap_check_ns={} msg_pass_ns={}",
        cap_ns, msg_ns
    );
    serial_println!(
        "BENCH boundary_hardware vmexit_port_ns={} blk_read_ns={} blk_real={} note=ratio_is_the_story",
        exit_ns, dev_ns, dev_real as u8
    );
}

fn ns_per(clk: &Clock, cycles: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    (cycles as u128 * 1_000_000_000 / (count as u128 * clk.hz as u128)) as u64
}

// ── V3. soak: sustained ops, latency drift, fragmentation, leak, rollback ──

fn bench_soak(clk: &Clock) {
    serial_println!("[validate] soak ({} ms): drift / fragmentation / leak / rollback …", SOAK_MS);
    let base_used = heap_used();
    let deadline = clk.now().wrapping_add((clk.hz as u128 * SOAK_MS as u128 / 1000) as u64);

    let mut g = ObjectGraph::new();
    let snap = g.commit("soak-base");
    let mut lcg = Lcg(0x0BADC0DE);
    let mut ops: u64 = 0;
    let mut rollbacks: u64 = 0;
    let mut rb_sum: u64 = 0;
    // Latency drift: compare the first window of ops to the most recent window.
    const WIN: u64 = 20_000;
    let (mut first_sum, mut first_n) = (0u64, 0u64);
    let (mut cur_sum, mut cur_n) = (0u64, 0u64);
    let (mut last_sum, mut last_n) = (0u64, 0u64);

    while clk.now() < deadline {
        let op0 = clk.now();

        // transient allocation churn (drives fragmentation)
        let sz = 256 + (lcg.next() as usize % 8192);
        let mut v: Vec<u8> = Vec::new();
        if v.try_reserve_exact(sz).is_ok() {
            v.resize(sz, 1);
            black_box(v[sz - 1]);
        }
        // bounded object churn: only 1024 distinct values, so the live set stays
        // bounded and the soak can run for hours without unbounded growth.
        g.put(Object::new("S").with("i", Datum::Int((ops & 0x3ff) as i64)));
        if ops.is_multiple_of(1000) {
            let _ = g.commit("c");
        }
        if ops.is_multiple_of(5000) && ops > 0 {
            let r0 = clk.now();
            let _ = g.rollback(snap); // truncate history → bounded; also recovery probe
            rb_sum += clk.now().wrapping_sub(r0);
            rollbacks += 1;
        }

        let lat = clk.now().wrapping_sub(op0);
        if first_n < WIN {
            first_sum += lat;
            first_n += 1;
        }
        cur_sum += lat;
        cur_n += 1;
        if cur_n >= WIN {
            last_sum = cur_sum;
            last_n = cur_n;
            cur_sum = 0;
            cur_n = 0;
        }
        ops += 1;
    }
    if last_n == 0 {
        last_sum = cur_sum;
        last_n = cur_n.max(1);
    }

    let first_avg = first_sum / first_n.max(1);
    let last_avg = last_sum / last_n.max(1);
    let drift_pct = if first_avg > 0 {
        ((last_avg as i64 - first_avg as i64) * 100) / first_avg as i64
    } else {
        0
    };

    // Fragmentation: largest contiguous allocation we can still make vs total free.
    let free = crate::allocator::free_bytes();
    let mut lo = 0usize;
    let mut hi = free;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let mut probe: Vec<u8> = Vec::new();
        if probe.try_reserve_exact(mid).is_ok() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let largest = lo;
    let frag_pct = if free > 0 { 100 - (largest as u128 * 100 / free as u128) as u64 } else { 0 };

    // Leak check: drop the graph, compare heap use to the pre-soak baseline.
    drop(g);
    let leak_kib = (heap_used().saturating_sub(base_used)) / 1024;

    serial_println!(
        "BENCH soak ms={} ops={} ops_per_s={} first_op_ns={} last_op_ns={} drift_pct={} rollbacks={} avg_rollback_us={} frag_pct={} largest_free_kib={} leak_kib={}",
        SOAK_MS,
        ops,
        clk.per_sec(ops, (clk.hz as u128 * SOAK_MS as u128 / 1000) as u64),
        ns_per(clk, first_avg, 1),
        ns_per(clk, last_avg, 1),
        drift_pct,
        rollbacks,
        clk.us(rb_sum.checked_div(rollbacks).unwrap_or(0)),
        frag_pct,
        largest / 1024,
        leak_kib
    );
}

// ── V4. chaos: failure injection + deterministic recovery ──

/// A block device that fails writes at one LBA — to test graceful I/O-fault handling.
struct FaultyDevice {
    inner: RamDisk,
    fail_lba: u64,
}
impl BlockDevice for FaultyDevice {
    fn block_count(&self) -> u64 {
        self.inner.block_count()
    }
    fn read_block(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        self.inner.read_block(lba, buf)
    }
    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        if lba == self.fail_lba {
            Err(BlockError::DeviceFault)
        } else {
            self.inner.write_block(lba, buf)
        }
    }
}

fn bench_chaos(clk: &Clock) {
    serial_println!("[validate] chaos: failure injection + deterministic recovery …");

    // C1: kill a domain mid-flight. The scheduler must keep running and the
    // survivors must still complete — no hang, no corruption of the run queue.
    let mut s = Scheduler::new();
    let mut ids = Vec::new();
    for i in 0..1000 {
        ids.push(s.spawn("", Capability::mint((i as u64) * 0x1000, 0x1000, Rights::ALL)));
    }
    let victim = ids[500];
    let mut steps = 0u64;
    let mut killed = false;
    while let Some(id) = s.next() {
        steps += 1;
        if !killed && id == victim {
            s.finish(id); // abrupt kill mid-flight
            killed = true;
        } else if s.steps_run(id) >= 2 {
            s.finish(id);
        } else {
            s.yield_back(id);
        }
    }
    let kill_ok = killed && s.live_count() == 0;
    serial_println!("BENCH chaos_kill pass={} steps={} survivors_completed={}", kill_ok as u8, steps, (s.live_count() == 0) as u8);

    // C2: corruption is detected, not silently trusted (fail-closed). A tampered
    // capability traps; a corrupted persisted image fails to load.
    let cap = Capability::mint(0x1000, 0x1000, Rights::ALL);
    let tamper_detected = cap.tamper().check(0x1000, 8, Rights::READ).is_err();
    let mut ram = RamDisk::new(64);
    let mut g = ObjectGraph::new();
    let oid = g.put(Object::new("Doc").with("v", Datum::Int(7)));
    let saved = Persistence::save(&mut ram, &g).is_ok();
    // Corrupt a byte *inside the object payload* (block 1, past the magic+count
    // header), then reload. Content-addressing must catch it: `deserialize`
    // recomputes each object's id from its bytes, so a tampered object no longer
    // hashes to its original id — the original id is simply absent (or load fails).
    let mut blk = [0u8; BLOCK_SIZE];
    ram.read_block(1, &mut blk).unwrap();
    blk[20] ^= 0xFF;
    ram.write_block(1, &blk).unwrap();
    let load = Persistence::load(&mut ram);
    let corruption_rejected = match &load {
        Err(_) => true,
        Ok(None) => true,
        Ok(Some(g2)) => !g2.contains(&oid),
    };
    serial_println!(
        "BENCH chaos_corruption pass={} cap_tamper_detected={} persist_corruption_rejected={} saved={}",
        (tamper_detected && corruption_rejected) as u8, tamper_detected as u8, corruption_rejected as u8, saved as u8
    );

    // C3: a disk write fault is surfaced as an error (graceful), not a panic, and
    // the in-memory state is untouched — the system stays usable.
    let mut faulty = FaultyDevice { inner: RamDisk::new(64), fail_lba: 2 };
    let mut g2 = ObjectGraph::new();
    for i in 0..50 {
        g2.put(Object::new("Rec").with("i", Datum::Int(i)));
    }
    let before = g2.stored_count();
    let save_res = Persistence::save(&mut faulty, &g2);
    let io_fault_handled = save_res.is_err() && g2.stored_count() == before;
    serial_println!("BENCH chaos_io_fault pass={} returned_err={} state_intact={}", io_fault_handled as u8, save_res.is_err() as u8, (g2.stored_count() == before) as u8);

    // C4: rollback under load. Repeatedly mutate then roll back to a snapshot; state
    // must return exactly and recovery latency must stay bounded.
    let mut g3 = ObjectGraph::new();
    g3.put(Object::new("Base").with("k", Datum::Int(1)));
    let snap = g3.commit("base");
    let base_live = g3.live_count();
    let rounds = 50_000u64;
    let mut ok = true;
    let (mut mn, mut mx, mut sum) = (u64::MAX, 0u64, 0u64);
    for i in 0..rounds {
        g3.put(Object::new("Churn").with("i", Datum::Int(i as i64)));
        let r0 = clk.now();
        let _ = g3.rollback(snap);
        let d = clk.now().wrapping_sub(r0);
        sum += d;
        mn = mn.min(d);
        mx = mx.max(d);
        if g3.live_count() != base_live {
            ok = false;
        }
    }
    serial_println!(
        "BENCH chaos_rollback_under_load pass={} rounds={} min_us={} avg_us={} max_us={}",
        ok as u8, rounds, clk.us(mn), clk.us(sum / rounds), clk.us(mx)
    );
}

// ── V5. cross-core scaling (requires real SMP; see smp.rs) ──
#[cfg(feature = "smp")]
fn bench_scaling(clk: &Clock) {
    crate::smp::bench_scaling(clk.hz);
}

// ─────────────────────────── driver ───────────────────────────

/// Headless entry point for the validation battery (`--features qemu_validate`).
pub fn run_validation_and_exit(usable_frames: usize) -> ! {
    serial_println!("\n========== DominionOS validation battery ==========");
    serial_println!(
        "[validate] heap {} MiB, {} MiB RAM",
        crate::allocator::total_bytes() / (1024 * 1024),
        usable_frames * 4 / 1024
    );
    let clk = Clock::calibrate();
    serial_println!("BENCH meta tsc_mhz={} ram_mib={} accel=see_host_log", clk.hz / 1_000_000, usable_frames * 4 / 1024);

    bench_mem_hierarchy(&clk);
    bench_boundary(&clk);
    #[cfg(feature = "smp")]
    bench_scaling(&clk);
    #[cfg(not(feature = "smp"))]
    serial_println!("BENCH scaling status=skipped reason=smp_feature_off");
    bench_soak(&clk);
    bench_chaos(&clk);

    serial_println!("================================================");
    serial_println!("BENCH complete");
    persist_results_to_disk("validate");
    exit_qemu(QemuExitCode::Success);
    crate::hlt_loop();
}

// ─────────────────────────── 9. polyglot language runtimes ───────────────────────────

/// Benchmark every guest language: parse throughput (the compiler front-end) and
/// execution throughput + interpreter steps for an identical compute load (a
/// `gcd`-folding loop over a library call). All seven produce the same checksum.
fn bench_polyglot(clk: &Clock) {
    use dominion_core::polyglot::{self, Language, Value as PValue, DEFAULT_STEP_BUDGET};
    serial_println!("[bench] (9/10) polyglot language runtimes (parse + execute with packages) …");
    const RUNS: u64 = 300;
    for lang in Language::all() {
        let src = polyglot::bench_program(lang);

        // Parse throughput: lex + parse the whole module RUNS times.
        let p0 = clk.now();
        let mut parsed = 0u64;
        for _ in 0..RUNS {
            if polyglot::parse(src, lang).is_ok() {
                parsed += 1;
            }
        }
        let pt = clk.now().wrapping_sub(p0);

        let prog = match polyglot::parse(src, lang) {
            Ok(p) => p,
            Err(_) => {
                serial_println!("BENCH polyglot lang={} error=parse", lang.name());
                continue;
            }
        };

        // Execution throughput + steps: run the parsed program RUNS times.
        let e0 = clk.now();
        let mut steps_total = 0u64;
        let mut checksum = 0i64;
        for _ in 0..RUNS {
            if let Ok(r) = polyglot::execute(&prog, DEFAULT_STEP_BUDGET) {
                steps_total += r.steps;
                if let PValue::Int(n) = r.value {
                    checksum = n;
                }
            }
        }
        let et = clk.now().wrapping_sub(e0);
        let steps_per_run = steps_total / RUNS.max(1);

        serial_println!(
            "BENCH polyglot lang={} parse_per_s={} runs={} exec_per_s={} steps_per_run={} steps_per_s={} checksum={}",
            lang.name(),
            clk.per_sec(parsed, pt),
            RUNS,
            clk.per_sec(RUNS, et),
            steps_per_run,
            clk.per_sec(steps_total, et),
            checksum
        );
    }
}

// ─────────────────────────── network benchmark ───────────────────────────

/// Benchmark the real EtherLink networking stack from inside the QEMU guest.
///
/// Measures wall-clock time (TSC) for:
///   1. DNS resolution (first lookup, fan-out to QEMU slirp + Google + Cloudflare)
///   2. HTTP GET to example.com (full stack: firewall→NDN CS miss→cap mint→gateway
///      open→SYN→request→reassemble→NDN store→gateway close)
///   3. NDN Content Store hit (second identical fetch, zero wire RTTs)
///   4. In-kernel protocol costs (NDN FIB lookup, DHT XOR, socket cap mint)
///
/// All measurements include QEMU virtio-net DMA and SLIRP NAT overhead — this is
/// what the full guest stack actually costs, not a simulation.
fn bench_network(clk: &Clock) {
    use dominion_core::webengine::Transport;

    // webnet.rs uses TPS=200 (the desktop's 200 Hz timer) for all tick-based
    // timeouts. The bench calibrated at 1 kHz, which makes CONNECT_TIMEOUT =
    // 400 ticks = 400 ms — too short for a SLIRP round-trip under load.
    // Reset to 200 Hz here so the timeouts behave as designed (2 s connect,
    // 3 s DNS). We restore 1 kHz afterward so the clock remains usable.
    set_timer_hz(200);

    serial_println!("[bench-net] initialising KernelTransport...");
    let mut t = match crate::webnet::KernelTransport::new() {
        Some(t) => t,
        None => {
            serial_println!("BENCH network status=no_nic skipped=true");
            set_timer_hz(1000);
            return;
        }
    };

    if !t.online() {
        serial_println!("BENCH network status=nic_offline skipped=true");
        set_timer_hz(1000);
        return;
    }

    // ── 1. DNS resolution latency ─────────────────────────────────────────
    // A plain HTTP request to a literal IP skips DNS; we force a real DNS
    // lookup by fetching "example.com" (not a literal). The transport's
    // resolve() is called internally as part of the first roundtrip; we
    // time the full first-fetch wall-clock which includes ARP + DNS + TCP.
    // To isolate DNS we time a minimal HTTP/1.0 request to a fast server.
    let http_req = b"GET / HTTP/1.0\r\nHost: example.com\r\nConnection: close\r\n\r\n";

    serial_println!("[bench-net] fetch 1: example.com (DNS + TCP + HTTP, cold)");
    let t0 = clk.now();
    let fetch1 = t.roundtrip("example.com", 80, false, http_req);
    let fetch1_cycles = clk.now().wrapping_sub(t0);
    let fetch1_ms = clk.ms(fetch1_cycles);
    let fetch1_ok = fetch1.is_ok();
    let fetch1_bytes = fetch1.as_ref().map(|b| b.len()).unwrap_or(0);

    serial_println!(
        "BENCH network_fetch_cold host=example.com port=80 ok={} bytes={} ms={} includes=arp+dns+tcp+http",
        fetch1_ok, fetch1_bytes, fetch1_ms
    );

    // ── 2. NDN Content Store hit (same request, should be cached) ─────────
    // The first fetch stored the response in the NDN CS keyed by
    // /http/example.com/<hash(request)>. The second identical request
    // should return immediately from the CS without any wire activity.
    serial_println!("[bench-net] fetch 2: example.com (NDN CS hit, no wire)");
    let t1 = clk.now();
    let fetch2 = t.roundtrip("example.com", 80, false, http_req);
    let fetch2_cycles = clk.now().wrapping_sub(t1);
    let fetch2_ms = clk.ms(fetch2_cycles);
    let fetch2_us = clk.us(fetch2_cycles);
    let fetch2_ok = fetch2.is_ok();
    let fetch2_bytes = fetch2.as_ref().map(|b| b.len()).unwrap_or(0);

    serial_println!(
        "BENCH network_fetch_cached host=example.com port=80 ok={} bytes={} ms={} us={} includes=ndn_cs_lookup_only",
        fetch2_ok, fetch2_bytes, fetch2_ms, fetch2_us
    );

    // ── 3. Second distinct host — measures DNS again, no prior ARP ────────
    // Firewall, gateway, CUBIC and OfflineReplica are all already warmed;
    // this isolates whether the ARP gateway-MAC cache is working (it should
    // be — the MAC is cached after the first gateway_mac() call).
    let http_req2 = b"GET / HTTP/1.0\r\nHost: neverssl.com\r\nConnection: close\r\n\r\n";
    serial_println!("[bench-net] fetch 3: neverssl.com (new DNS, cached ARP/GW)");
    let t2 = clk.now();
    let fetch3 = t.roundtrip("neverssl.com", 80, false, http_req2);
    let fetch3_cycles = clk.now().wrapping_sub(t2);
    let fetch3_ms = clk.ms(fetch3_cycles);
    let fetch3_ok = fetch3.is_ok();
    let fetch3_bytes = fetch3.as_ref().map(|b| b.len()).unwrap_or(0);

    serial_println!(
        "BENCH network_fetch_new_host host=neverssl.com port=80 ok={} bytes={} ms={} includes=dns+tcp+http_no_arp",
        fetch3_ok, fetch3_bytes, fetch3_ms
    );

    // ── 4. In-kernel protocol microbenchmarks (no wire I/O) ───────────────
    // These match the host-side net_bench.rs numbers but now run on the
    // actual guest CPU under QEMU, so they include virtualization overhead.
    {
        use dominion_core::ndn::{Forwarder, Name};
        use dominion_core::dominionlink::DominionId;
        use dominion_core::legacynet::{NetworkStack, Protocol, SocketCapability};
        use dominion_core::net::Ipv4Addr as CoreIpv4Addr;

        // NDN FIB lookup
        const FIB_ITERS: u64 = 200_000;
        let mut fwd = Forwarder::new();
        fwd.register_route(Name::parse("/http"), 1);
        fwd.register_route(Name::parse("/dominion"), 2);
        let query = Name::parse("/http/example.com/index");
        let t_fib = clk.now();
        for _ in 0..FIB_ITERS {
            let _ = black_box(fwd.recv_interest(0, &query));
        }
        let fib_ns = ns_per(clk, clk.now().wrapping_sub(t_fib), FIB_ITERS);

        serial_println!(
            "BENCH network_ndn_fib_guest iters={} ns_per_op={} note=includes_qemu_virt_overhead",
            FIB_ITERS, fib_ns
        );

        // DHT XOR distance
        const DHT_ITERS: u64 = 200_000;
        let id_a = DominionId::from_pubkey(b"peer-alpha-pubkey-bench");
        let id_b = DominionId::from_pubkey(b"peer-beta-pubkey-bench-");
        let t_dht = clk.now();
        for _ in 0..DHT_ITERS {
            let _ = black_box(id_a.distance(&id_b));
        }
        let dht_ns = ns_per(clk, clk.now().wrapping_sub(t_dht), DHT_ITERS);

        serial_println!(
            "BENCH network_dht_xor_guest iters={} ns_per_op={} note=includes_qemu_virt_overhead",
            DHT_ITERS, dht_ns
        );

        // Socket capability mint + verify
        const CAP_ITERS: u64 = 100_000;
        let stack = NetworkStack::new(b"bench-issuer-key-01234567890123");
        let dst = CoreIpv4Addr::new(93, 184, 216, 34);
        let t_cap = clk.now();
        for i in 0..CAP_ITERS {
            let cap = black_box(SocketCapability::mint(
                Protocol::Tcp,
                (49152 + (i as u16 % 16000)) as u16,
                Some((dst, 80)),
                b"bench-issuer-key-01234567890123",
            ));
            let _ = black_box(cap.is_authentic(b"bench-issuer-key-01234567890123"));
        }
        let cap_ns = ns_per(clk, clk.now().wrapping_sub(t_cap), CAP_ITERS);

        serial_println!(
            "BENCH network_socket_cap_guest iters={} ns_per_op={} note=mint_plus_verify",
            CAP_ITERS, cap_ns
        );

        let _ = stack; // suppress unused warning
    }

    // ── 5. Summary ────────────────────────────────────────────────────────
    let speedup = if fetch2_us > 0 && fetch1_ms > 0 {
        (fetch1_ms * 1000) / fetch2_us.max(1)
    } else {
        0
    };
    serial_println!(
        "BENCH network_summary cold_ms={} cached_us={} ndn_speedup={}x fetch1_ok={} fetch2_ok={} fetch3_ok={}",
        fetch1_ms, fetch2_us, speedup, fetch1_ok, fetch2_ok, fetch3_ok
    );

    // Restore 1 kHz so any remaining bench timing is accurate.
    set_timer_hz(1000);
}

/// Headless entry point: calibrate, run every benchmark, exit QEMU.
pub fn run_and_exit(usable_frames: usize) -> ! {
    serial_println!("\n========== DominionOS real-world benchmark battery ==========");
    serial_println!(
        "[bench] heap {} MiB, {} usable frames ({} MiB RAM)",
        crate::allocator::total_bytes() / (1024 * 1024),
        usable_frames,
        usable_frames * 4 / 1024
    );
    let clk = Clock::calibrate();
    serial_println!(
        "BENCH meta tsc_mhz={} heap_mib={} ram_mib={} accel=see_host_log",
        clk.hz / 1_000_000,
        crate::allocator::total_bytes() / (1024 * 1024),
        usable_frames * 4 / 1024
    );

    bench_pool(&clk);
    bench_task_creation(&clk);
    bench_message_passing(&clk);
    bench_graph_execution(&clk);
    bench_memory_pressure(&clk);
    bench_storage(&clk);
    bench_distributed(&clk);
    bench_security(&clk);
    bench_developer(&clk);
    bench_polyglot(&clk);
    bench_ml(&clk);
    bench_network(&clk);

    serial_println!("[bench] (12/12) done");
    serial_println!("=========================================================");
    serial_println!("BENCH complete");
    persist_results_to_disk("benchmark");
    exit_qemu(QemuExitCode::Success);
    crate::hlt_loop();
}

/// Flush the captured boot/run/benchmark serial log (which contains every `BENCH …`
/// line) to the preferred log device — a removable USB when one is present, so the
/// results land on the same stick the machine booted from. This is what makes the
/// benchmark battery useful on **bare metal**, where (unlike QEMU) there is no host
/// serial capture: after the run halts you pull the USB and extract the results with
/// `read-usb-results.ps1`. On QEMU it additionally persists to the data-disk tail,
/// which `read-bootlog.ps1` can read — harmless and occasionally handy. The user opted
/// into this run, so the write is unconditional (`persist_force`).
fn persist_results_to_disk(kind: &str) {
    if crate::bootlog::persist_force() {
        let where_ = if crate::block::log_is_usb() { "USB" } else { "disk" };
        serial_println!(
            "[{}] results + full log persisted to {} (extract with read-usb-results.ps1)",
            kind,
            where_
        );
    } else {
        serial_println!(
            "[{}] no writable disk to persist results to; capture them from serial instead",
            kind
        );
    }
}

# AetherOS — Developer Research Preview

> ⚠️ **Experimental. In Development. Use at Your Own Risk.**
> 
> This is a research OS built in ~4 days by [Cognitive Industries](https://cognitive-industries.org). We've poured 99% of the effort into the backend architecture and it works. The frontend and user experience are sparse—they'll be the main focus if this gains traction. This is not Windows, macOS, or Linux. It will not behave like them. Expect missing features, incomplete hardware support, and breaking changes between versions. Use in QEMU or on expendable hardware only. **Possible hardware damage on bare metal. You've been warned.**

---

## What Is AetherOS?

We built a ground-up operating system around three core ideas:

1. **Capabilities as first-class authority.** No ambient permission model. Every action requires an unforgeable token of authority. Memory corruption can't escalate privilege.

2. **Content-addressed storage.** Everything is immutable and deduplicated. Data survives reboots. The entire OS state is hashable and reproducible, like a system-wide Git.

3. **Deterministic execution.** The whole machine is a state machine. Every action is an input. We can rewind the OS instruction-by-instruction and replay exactly what happened.

It boots on bare metal (or QEMU). It has a working terminal. We can run the Aether language (our safe interpreter) inside the OS. The capability security, storage layer, and process model all function.

What we *haven't* finished: a polished user interface, full hardware support, formal verification, and distributed networking. Those are next if there's momentum.

---

## What Actually Works

### Core System
- ✅ **Boots on real hardware and VMs** (QEMU tested)
- ✅ **Capability security** enforced in software (CHERI-style)
- ✅ **Immutable object graph** with content addressing (system-wide Git)
- ✅ **Deterministic state machine** with instruction-level replay
- ✅ **Safe-mode terminal** for low-level access
- ✅ **Process isolation** via capability-bounded IPC
- ✅ **Real device drivers** (PCI, virtio-blk, virtio-net confirmed working)
- ✅ **Persistence** to disk; survives reboot
- ✅ **The Aether language** running inside the OS

### Security & Cryptography
- ✅ **Unforgeable capabilities** with provenance tracking
- ✅ **Post-quantum cryptography** (Lamport XMSS, hybrid signatures)
- ✅ **ChaCha20-Poly1305 + AES-256-GCM** encryption
- ✅ **Zero-plaintext vault** with crypto-agility
- ✅ **Capability firewall + airlock** for inter-domain authority
- ✅ **Deniable storage** with coercion resistance (hidden domains)
- ✅ **Runtime attestation** with hash-chain provenance

### ML & Compute
- ✅ **Neural network inference** on-device
- ✅ **Reverse-mode autodiff** for training
- ✅ **Heterogeneous execution** (@CPU/@GPU/@NPU placement hints)
- ✅ **Quantized inference** (int8) for edge devices
- ✅ **Learned XOR on the booted machine** (verified end-to-end)

### Networking & Storage
- ✅ **NDN (Named Data Networking)** forwarding
- ✅ **DominionLink** (self-certifying IDs + DHT)
- ✅ **ARP + ICMP** over virtio network
- ✅ **Content-addressed object store** with versioning
- ✅ **POSIX-projection VFS** for file access

---

## What Doesn't Work (Yet)

### Frontend / UX
- ✅ **Desktop GUI with 9 apps** (Desktop, Files, Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings)
- ✅ **Floating window manager** with z-order, draggable title bars, resizable windows
- ✅ **Taskbar, system tray, keyboard shortcuts**
- ✅ **Terminal is fully functional** — limited command set, expanding over time
- ⚠️ **Browser** renders HTML/CSS/JS but integration gaps remain
- ⚠️ **Shell/workspace architecture** has some fragmentation (multiple shell implementations)
- ⚠️ **Composable UI** implemented on Desktop but not extended to all app pages yet

### Hardware Support
- ⚠️ **Keyboard/mouse/input devices** limited (PS/2 works, USB HID incomplete)
- ⚠️ **GPU drivers** not implemented (software rendering only)
- ⚠️ **Network drivers** limited to virtio (no Intel/Broadcom/others yet)
- ⚠️ **Audio** specified but not implemented
- ⚠️ **Wireless networking** not implemented

### Advanced Features
- ❌ **Formal verification proofs** for microkernel (code is memory-safe; proofs deferred)
- ❌ **Measured boot chain** (TPM integration incomplete)
- ❌ **Generative compression** for storage (neural codecs specified but not coded)
- ❌ **Distributed multi-node** deployment (single-node focus)
- ❌ **User-space driver isolation** (drivers still in kernel)
- ❌ **ML-based threat detection** (statistical baselines only)

---

## Quick Start

### Build It

**Requirements:**
- Rust 1.70+
- `cargo-bootimage` for disk images
- QEMU (for testing)

**Compile kernel:**
```powershell
cd kernel
cargo build --release
```

**Create bootable disk image:**
```powershell
cd ..\boot
cargo run --release -- ..\kernel\target\x86_64-dominion\release\dominion-kernel ..\aetheros.img
```

This produces `aetheros.img` (raw BIOS disk image, bootable in QEMU or on bare metal).

### Run It

**In QEMU (recommended for now):**
```bash
# Windows
.\run.ps1

# Linux/macOS
cargo run --release
```

**On bare metal:**
```bash
# Create bootable USB (Windows)
.\make-bootable-usb.ps1 -ImagePath aetheros.iso -USBDrive "F:"

# Boot from USB and hope for the best
```

**In a VM:**
- VirtualBox: Mount `aetheros.iso`, boot
- VMware: Same
- QEMU: `qemu-system-x86_64 -drive file=aetheros.img,format=raw`

### Try the Terminal

Once booted, you land in a safe-mode shell (ASH). Try:
```
help
list
world
time
ml
```

The `ml` command runs a neural network benchmark. The system is working if it completes.

---

## Features & Limitations

### What You Get
- **A working desktop OS.** Nine apps, window manager, taskbar. It boots and runs. Some features have wiring gaps but it's usable.
- **True capability security.** Your programs can't escape their authority boundaries. Ever. The kernel enforces it.
- **Reproducible execution.** Crash a program, rewind it, replay the exact same sequence of instructions. No non-determinism.
- **On-device AI.** Inference and training. No cloud. No telemetry.
- **Immutable snapshots.** Commit your system state. Roll back anytime.
- **Hardware isolation.** Drivers can't DMA outside their granted memory. Capabilities make this enforced, not advisory.

### What You Don't Get
- **Plug-and-play compatibility.** Your mouse might not work. Your network card might not work. Expect to write drivers.
- **Legacy support.** No POSIX, no Win32 (mostly). Your Linux commands won't run.
- **Performance tuning.** The OS prioritizes correctness and security over speed. It's not slow, but it's not optimized.
- **Production readiness.** We've tested this in QEMU on a handful of configurations. Real hardware is a gamble.

---

## Hardware

### Confirmed Working
**Test Platform:**
- CPU: Intel Core i7-12650H (10 cores / 16 threads, ~4.0 GHz boost)
- RAM: 16 GB DDR5
- Storage: 1 TB Micron NVMe SSD (accessed via virtio-blk in QEMU)
- GPU: NVIDIA GeForce RTX 4060 Laptop (not used — software rendering only)
- Host: Windows 11, QEMU with WHPX hardware acceleration

**In QEMU (tested):**
- x86-64, BIOS boot
- 8 cores, 4GB RAM
- virtio-blk (disk), virtio-net (network)
- framebuffer console (SVGA)

### Known Limitations
- **Keyboard:** PS/2 only (no USB HID)
- **Mouse:** PS/2 mouse works; trackpads don't
- **Network:** virtio only (no Intel 82540EM, no Broadcom)
- **Storage:** virtio-blk only (no NVMe direct, no SATA)
- **Graphics:** Software framebuffer (no GPU acceleration)

**Translation:** Don't try this on your gaming PC yet. QEMU is your safest bet.

---

## Performance Benchmarks

These are **real numbers** from actual `run-bench.ps1` runs on an i7-12650H with WHPX acceleration (near-native speed). All results are from inside the booted AetherOS kernel — no fabrication, no extrapolation except where explicitly marked as `projected_*`.

**Benchmark configuration:** 8 vCPUs, 4096 MiB RAM, WHPX accel, TSC at ~3973 MHz.

### Process & Task Throughput

| Benchmark | Result | Notes |
|---|---|---|
| Task spawn rate | **1,430,747 tasks/s** | 1M tasks in 698ms |
| Task dispatch | **51,520 tasks/s** | O(n) per step — known bottleneck |
| Thread pool submit | **23 ns/submit** | 50,000 submissions |
| Thread pool serial exec | **17,356,411 tasks/s** | Pure in-kernel |
| Thread pool parallel (8 workers) | **15,233,411 tasks/s** | 8-way work-stealing |

### Message Passing (IPC)

| Benchmark | Result |
|---|---|
| Throughput | **133,266,744 msgs/s** |
| Latency | **7 ns/message** |
| Delivered (single run) | 5,000,000 messages |

### Graph Execution

| Benchmark | Result | Notes |
|---|---|---|
| DCG linear eval | **222,070,090 nodes/s** | 1M nodes in 4ms |
| WorkGraph scheduler | **294,465 nodes/s** | O(n²) — known gap at 2000 nodes |

### Storage I/O (virtio-blk, real in-guest, 20K blocks)

| Metric | Sequential | Random |
|---|---|---|
| Read IOPS | **39,070** | **35,585** |
| Write IOPS | **1,694** | **1,709** |
| Read KiB/s | **19,535** | **17,792** |
| Write KiB/s | **847** | **854** |
| Object store (puts/s) | **14,252** | — |

Write IOPS are bounded by virtio-blk emulation overhead under WHPX. Read is fast because the object graph is content-addressed (hash-match skips re-read).

### Cryptography (4096-byte payloads, 4000 iterations)

| Mode | Throughput | Overhead vs plaintext |
|---|---|---|
| Plaintext copy | 197 MiB/s | baseline |
| ChaCha20-Poly1305 | **123 MiB/s** | 59% overhead |
| AES-256-GCM | 72 MiB/s | 173% overhead |

ChaCha20-Poly1305 is the default. No AES-NI used yet — hardware acceleration would close this gap substantially.

### Memory

| Metric | Result |
|---|---|
| Peak heap under pressure | 1023 MiB |
| OOM recovery time | **53 µs** |
| Graceful recovery | ✅ yes |

### Distributed Substrate (single-node multi-core)

| Benchmark | Result | Notes |
|---|---|---|
| Inter-core messages | **3,672,069 msgs/s** | 16 cores, shared node |
| CRDT merges | **11,391,218 merges/s** | 1M merges, converges to 16 |

### ML Compute

| Benchmark | Result |
|---|---|
| Matrix multiply (128×128, SSE2) | **11.8 GFLOP/s** |
| Multi-core matmul (8 workers, bit-identical) | 9.6 GFLOP/s (81% speedup) |
| Fused neural inference (2×16×8×1 MLP) | **60,357 inferences/s** |

ML runs on CPU only — no GPU driver. No AVX/FMA in this run (deterministic SSE2 build). Enable with `.\run-bench.ps1 -Fma` for higher throughput at the cost of bit-for-bit reproducibility.

### Language Interpreter (Aether polyglot bench)

All variants run the same AST — just parsed from different source syntax:

| Lang | Parse rate | Exec rate | Steps/s |
|---|---|---|---|
| Python | 49,721/s | 604/s | 8,460,168 |
| Rust | 31,890/s | 621/s | 8,705,849 |
| Java | 33,314/s | 652/s | 18,265,485 |
| TypeScript | 34,670/s | 635/s | 17,790,190 |
| C++ | 25,537/s | 603/s | 16,899,453 |

**Linux/Windows comparisons are deferred** — running the Linux harness (`bench/linux/run-linux-bench.sh`) requires a bare-metal Linux machine or WSL. The numbers above are AetherOS-on-WHPX and are a fair baseline for AetherOS itself. We'll publish the Linux comparison once the harness is complete.

**Bottom line:** AetherOS is fast where it matters — single-digit nanosecond IPC, 133M msgs/s, fast OOM recovery, deterministic bit-identical ML. It's not yet tuned for peak throughput (no AES-NI, no AVX by default, cooperative scheduler). Those are future work.

---

## Architecture Overview

You don't need to memorize this, but here's how the pieces fit:

### Three Primitives
1. **Capabilities.** Unforgeable tokens of authority. Every operation requires one. Can't be forged or escalated.
2. **Content-addressed storage.** SHA-256 hashes. Immutable. Deduplicated. The whole OS state is hashable.
3. **Deterministic execution.** Every action is an input event. No hidden state. Reproducible down to the clock cycle.

### Layers (bottom to top)
- **Bootloader** (~50KB) — BIOS → 64-bit mode → hand off to kernel
- **Kernel** (~100KB) — Scheduling, memory management, IPC, capability enforcement
- **Core services** (~400KB) — Filesystem, networking, crypto, device drivers
- **Dominion runtime** — Safe interpreter inside the OS
- **Applications** — Shell, terminals, tools (minimal for now)

### Security Model
- **Firewall** — Directed capability graph, cross-domain denial by default
- **Airlock** — Inter-domain authority transfer with approval gates
- **Attestation** — Cryptographic hash chains for provenance
- **Amnesic domains** — Volatile memory that gets wiped on lock
- **Deniable storage** — Hidden domains indistinguishable from noise

---

## Contributing

We want contributors. Here are the rules:

1. **Read the spec first.** Check `docs/architecture.md` and the subsystem guides.
2. **We use Claude Opus 4.8 for AI-assisted development.** If you're using AI, it must be Opus 4.8. (Other models produce lower quality output for systems code.)
3. **Propose before coding.** For major features: open an issue, explain the change, get feedback. For minor fixes: PR is fine.
4. **Small, isolated changes.** One feature per PR. If it touches the kernel and the shell, split it.
5. **Tests and benchmarks required.** New code must pass all existing tests and not regress benchmarks.
6. **Compile and run it.** We test on QEMU. Make sure it boots and your feature works.

See `CONTRIBUTING.md` for full details.

---

## License & Commercial Use

### Open Source
AetherOS is free to use and modify for research, education, and non-commercial projects. Full source code, full rights to study and fork.

**License:** Dual-licensed under the **GNU Affero General Public License v3.0** (AGPLv3) for non-commercial use and the **Prosperity Public License** for commercial use.

- **Non-commercial?** Use AGPLv3. Attribution required. Share improvements.
- **Commercial (company/corporation)?** You need a license. Contact us.
- **Developer using it for a side project?** That's non-commercial. AGPLv3 applies.

See `LICENSE.md` for full terms.

### Attribution
If you use AetherOS or its code, please credit us:

> AetherOS, developed by Cognitive Industries (https://cognitive-industries.org)

---

## Getting Help

- **Architecture & design questions:** See `docs/architecture.md`
- **How to build/run:** See `DEVELOPMENT.md`
- **Contributing:** See `CONTRIBUTING.md`
- **Specific subsystem info:** Check `docs/subsystem-manifest.json`
- **Licensing & commercial use:** contact@cognitive-industries.org
- **General inquiries:** [cognitive-industries.org](https://cognitive-industries.org)

---

## Development Status

We built this in about 4 days as a proof of concept. The backend works. The frontend doesn't. Here's what that means:

- **Backend (99% done):** Capability system, storage, crypto, process isolation, network stack. All tested. All works.
- **Frontend (20% done):** Shell, terminal, GUI. Code exists. Wiring incomplete. Next priority if we continue.
- **Features (60% done):** ML inference works. Distributed networking sketched. Formal proofs deferred.
- **Hardware support (30% done):** QEMU tested. Real hardware is a research question.

We're not abandoning this. But we're also not prioritizing it unless there's genuine interest or a specific use case that makes sense.

**Frequency of updates:** Sparse, unless something changes. We'll respond to issues and PRs, but don't expect weekly releases.

---

## What's Next (If We Continue)

1. **Frontend.** Wire up the shell, terminal, and GUI properly.
2. **Hardware support.** Real drivers for Intel NICs, GPUs, etc.
3. **Formal verification.** Prove the capability system is correct.
4. **Distributed networking.** Multi-node capability enforcement.
5. **Optimization.** Hardware-accelerated crypto, SIMD, etc.

All of these are doable. None are show-stoppers. Just requires focus.

---

## FAQ

**Q: Will this replace Linux?**
A: No. We're not trying to. We're exploring a different security model. Linux is mature, battle-tested, and has massive hardware support. We're a research project.

**Q: Can I run Rust/Python/Go on this?**
A: The Aether language runs inside the OS and includes a polyglot interpreter that parses Python, Rust, JavaScript, TypeScript, Java, and C++ syntax. It's not a full runtime for those languages — it's a toy-level demo of the interpreter. Full language runtimes are future work.

**Q: The GUI — is it real?**
A: Yes. There's a full desktop with 9 apps (Desktop, Files, Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings), floating window manager, taskbar, and system tray. It boots and runs. Some wiring is incomplete — not everything is hooked up yet. The GUI is the main focus going forward.

**Q: Is this a microkernel OS?**
A: Sort of. We have a small kernel (~100KB), but drivers are still inside it. True microkernel design (all drivers in userspace) is phase 2.

**Q: Can I use this in production?**
A: No. It's experimental. Expect crashes, missing features, and breaking changes.

**Q: What's the point if it's not production-ready?**
A: Research. We're testing whether capability-based security can be both practical and performant. The answer so far is yes. That matters.

---

## Acknowledgments

- Built with [Rust](https://www.rust-lang.org)
- Inspired by seL4, the Capability Maturity Model, and CHERI
- Thanks to the [Bootloader](https://github.com/rust-osdev/bootloader) project
- Benchmarked on QEMU and real hardware

---

**AetherOS is a Cognitive Industries research project.**

Questions, feedback, or licensing inquiries? **contact@cognitive-industries.org**

Visit us: [cognitive-industries.org](https://cognitive-industries.org)

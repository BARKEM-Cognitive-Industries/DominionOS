# DominionOS

> **Experimental. In development. Use at your own risk.**
> Real hardware can be bricked. QEMU or expendable bare metal only until this stabilises.

A capability-secured operating system written from scratch in Rust. The backend (capability enforcement, storage, crypto, networking, ML) is solid. The frontend and hardware coverage are still early. It's a research OS, released because people asked to poke at it.

![Benchmark overview](docs/assets/bench-overview.svg)

---

## What this actually is

DominionOS is a research OS. Not a replacement for Linux or Windows. Not production ready. A proof of concept that answers one question: can you build a practical OS where capability-based security, deterministic execution, and content-addressed storage are first-class primitives rather than bolted-on afterthoughts?

The answer so far: yes.

The backend works. Capability enforcement, storage, cryptography, IPC, ML inference, networking, all functional. The part you'd interact with as a user (shell, desktop apps, broad hardware support) is further behind. That's the main focus going forward. This wasn't released because it's polished. It was released because the architecture is interesting and people wanted to experiment with it.

**Don't expect Windows. Don't expect Linux.** Expect a research OS that boots, has real security primitives, and is missing a lot of the things you'd consider basic.

---

## The core ideas

**Capabilities.** Every operation requires an unforgeable token of authority. There's no ambient permission model: you can't escalate privilege through memory corruption because capabilities are kernel-enforced and can't be forged. This is CHERI-style capability security implemented in software.

**Content-addressed storage.** The entire system state is a SHA-256 hash tree. Everything is immutable and deduplicated. Snapshots are instant. You can roll back the whole OS like reverting a Git commit.

**Deterministic execution.** The machine is a state machine. Every action is an explicit input event. No hidden state, no timing side channels in the core model. Crash a process, rewind it, replay exactly what happened.

These aren't marketing claims. They're in the code, enforced at the kernel level, with tests that run both on the host and on the booted machine.

---

## What works

**Core system**
- Boots on x86-64, BIOS and UEFI (QEMU tested, bare metal works with caveats)
- Capability security, enforced in the kernel, with provenance tracking
- Content-addressed immutable object graph (system-wide Git)
- SMP: 8 cores tested, near-linear scaling
- Deterministic state machine with replay
- Safe-mode terminal (ASH) for low-level access

**The Dominion language** runs inside the OS: lexer, parser, interpreter, capability-gated cells, parallel placement hints (`@CPU`/`@GPU`/`@NPU`), semantic primitives. The polyglot frontend accepts Python, Rust, JavaScript, TypeScript, Java, and C++ syntax and compiles it to the same AST. It's useful for demo-level code, not a full runtime for any of those languages.

**Security and cryptography**
- ChaCha20-Poly1305 (default vault cipher)
- AES-256-GCM
- Lamport / hash-based (XMSS-style) post-quantum signatures
- TLS 1.3 with X.509 chain validation
- Zero-plaintext encrypted vault with crypto-agility
- Capability firewall and airlock (intra/inter-domain authority)
- Deniable storage (hidden domains, coercion resistant)
- Runtime attestation with hash-chain provenance

**Storage and persistence**
- Immutable content-addressed object graph
- Instant snapshot and rollback
- VFS projection for POSIX-style file access
- Block drivers: virtio-blk, AHCI (SATA), NVMe (PCIe), and USB mass storage (Bulk-Only Transport + SCSI over xHCI)
- Persistence to disk with crash recovery

**Networking**
- Named Data Networking (NDN) forwarding
- DominionLink: self-certifying IDs and Kademlia DHT
- NIC drivers: virtio-net, RTL8139, and Intel e1000/e1000e, sharing one interface abstraction
- ARP, ICMP, UDP, DHCP
- DNS bridge (DominionLink names to DNS)
- HTTPS/TLS

**ML and compute**
- Neural network inference and training (reverse-mode autodiff)
- int8 quantization
- 11.8 GFLOP/s matrix multiply (SSE2, no AVX, no external libraries)
- 60,357 inferences/second on a 2x16x8x1 MLP
- Bit-identical results across cores by default, determinism enforced

**Desktop**
- 9 built-in apps: Desktop, Files, Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings
- Floating window manager, taskbar, system tray
- Unified 2D/3D rendering stack with software rasterizer
- Browser with HTML5 parsing, CSS cascade, JavaScript interpreter

---

## What doesn't work yet

**Frontend (a continuing focus)**
- Terminal: functional (both the ASH recovery shell and the desktop terminal), with a limited command set
- Browser: renders HTML/CSS/JS, not all features wired
- Shell consolidation, the recovery shell and desktop terminal share a command backend but the surfaces could be unified further
- Composable UI panels exist on the desktop, not yet extended to all app pages

**Hardware support (growing, still partial)**
- Input: PS/2 keyboard/mouse and USB HID keyboard/mouse both work; touchpads/touchscreens don't
- Network: virtio-net, RTL8139, Intel e1000/e1000e, validated primarily in QEMU; broad real-NIC coverage is thin
- Storage: virtio-blk, AHCI, NVMe, USB mass storage, validated primarily in QEMU
- Graphics: software framebuffer, no GPU acceleration
- Audio: not implemented
- Wireless (WiFi/Bluetooth): not implemented

**Advanced features (roadmap)**
- Formal verification proofs: deferred
- Measured / secure boot to firmware/TPM: deferred
- Multi-user: single-user for now
- Preemptive scheduling: currently cooperative
- Distributed multi-node deployment: single-node focus
- ARM64 / RISC-V: x86-64 only for now

---

## Benchmarks

These numbers come from `run-bench.ps1` running DominionOS inside QEMU with WHPX (Windows Hypervisor Platform, near-native speed). Test machine: Intel i7-12650H, 16 GB DDR5, Windows 11 host, 8 vCPUs allocated to QEMU, 4 GiB RAM.

Linux and Windows benchmarks on the same machine aren't run yet. Those comparisons will come when the Linux bench harness (`bench/linux/run-linux-bench.sh`) is complete.

![IPC throughput](docs/assets/bench-ipc.svg)

![Storage IOPS](docs/assets/bench-storage.svg)

![Crypto throughput](docs/assets/bench-crypto.svg)

![ML compute](docs/assets/bench-ml.svg)

### Numbers at a glance

| Subsystem | Metric | Result |
|---|---|---|
| IPC | Throughput | 133,266,744 msgs/s |
| IPC | Latency | 7 ns per message |
| Tasks | Spawn rate | 1,430,747 tasks/s |
| Tasks | Dispatch (O(n)) | 51,520 tasks/s |
| Graph eval | DCG linear | 222,070,090 nodes/s |
| Storage | Sequential read | 39,070 IOPS |
| Storage | Sequential write | 1,694 IOPS |
| Storage | Object puts | 14,252 obj/s |
| Crypto | ChaCha20-Poly1305 | 123 MiB/s |
| Crypto | AES-256-GCM | 72 MiB/s |
| ML | 128x128 matmul (1 core) | 11.8 GFLOP/s |
| ML | Inference (MLP) | 60,357 infer/s |
| ML | Multi-core scaling (8w) | 81% (9.6 GFLOP/s) |
| Memory | OOM recovery | 53 µs |
| CRDT | Merge rate | 11,391,218 merges/s |

Storage write IOPS are bounded by virtio-blk emulation overhead, not the object graph. Read is fast because content-addressed hashes let the system skip re-reads of unchanged data.

Crypto is software-only, no AES-NI, no hardware offload. AES-NI support would close the gap substantially. ChaCha20-Poly1305 is the default because it's faster in software and post-quantum resilient.

ML runs on CPU only, SSE2, bit-identical across cores. Enable AVX/FMA with `.\run-bench.ps1 -Fma` for higher throughput at the cost of bit-for-bit reproducibility.

---

## Hardware

### Tested configuration

- CPU: Intel Core i7-12650H (10 cores / 16 threads)
- RAM: 16 GB DDR5
- Storage: 1 TB Micron NVMe SSD (accessed via virtio-blk in QEMU)
- GPU: NVIDIA RTX 4060 Laptop + Intel UHD Graphics (neither used, software rendering only)
- Host: Windows 11, QEMU 8.x with WHPX acceleration

### In QEMU (what gets tested most)

- x86-64, BIOS and UEFI boot
- virtio-blk / AHCI / NVMe (disk), virtio-net / e1000 (network)
- PS/2 and USB HID keyboard and mouse
- Software framebuffer (SVGA)

### Known limitations

- Input: PS/2 and USB HID keyboard/mouse work; touchpads and touchscreens don't.
- Network: virtio-net, RTL8139, and Intel e1000/e1000e are implemented; other real NICs aren't covered yet.
- Storage: virtio-blk, AHCI, NVMe, and USB mass storage are implemented; validated mostly in QEMU.
- Graphics: software framebuffer. No GPU acceleration.
- Wireless: none. Audio: none.
- Real hardware: boots and runs on x86-64 bare metal, but coverage outside the drivers above is thin. Treat it as experimental.

**Bottom line for hardware:** QEMU is the safe path. A mainstream x86-64 box with SATA/NVMe storage, an Intel or RTL NIC, and PS/2 or USB HID input has a real chance of booting to a usable state. Anything needing GPU, audio, or wireless will boot without those.

---

## Build and run

**Requirements**
- Rust nightly (the kernel builds against the custom `x86_64-dominion` bare-metal target)
- QEMU for testing

**Everything at once (recommended)**
```powershell
.\run.ps1
```
This builds the kernel, wraps it into a bootable image, and launches it in QEMU.

**Manual build**
```powershell
# Kernel (freestanding x86_64)
cd kernel
cargo build --release

# Wrap into a bootable disk image
cd ..\boot
cargo run --release -- ..\kernel\target\x86_64-dominion\release\dominion-kernel ..\dominionos.img
```

**Run benchmarks**
```powershell
.\run-bench.ps1
# Results written to bench-results.json
```

**Run with AVX/FMA (non-deterministic, faster ML)**
```powershell
.\run-bench.ps1 -Fma
```

**Safe mode (text-only, no desktop)**

Build with `--features safe_mode`. Good for bare metal where the desktop doesn't come up.

---

## The terminal

Both the ASH recovery shell and the desktop terminal are functional: type, press Enter, things happen. The command set is deliberately small, it's not bash, there's no package manager, and most Unix commands don't exist. What's there: `help`, `ver`, `mem`, `ticks`, `hw`, `log`, `caps`, `obj`, `vfs`, `pci`, `net`, `link`, `disk`, `ml`, `llm`, `state`, `run`, `dominion` (a live language REPL), `selftest`, `reboot`, `shutdown`, and more. Type `help` for the full list.

---

## Feature matrix

Full list in `FEATURES.md`. Short version:

| Area | Status |
|---|---|
| Capability security | Implemented |
| Content-addressed storage | Implemented |
| Deterministic execution | Implemented |
| Dominion language | Implemented |
| ML inference + training | Implemented |
| NDN networking | Implemented |
| DominionLink DHT | Implemented |
| Storage drivers (virtio/AHCI/NVMe/USB) | Implemented |
| NIC drivers (virtio/RTL8139/e1000) | Implemented |
| USB (xHCI + HID + mass storage) | Implemented |
| Desktop (9 apps) | Partial |
| Browser | Partial |
| GPU acceleration | Roadmap |
| Audio / wireless | Roadmap |
| Formal verification | Roadmap |
| Multi-user | Roadmap |
| ARM64 / RISC-V | Roadmap |

---

## Architecture

The architecture documentation lives in `docs/architecture.md` (17 subsystems, machine-readable manifest in `docs/subsystem-manifest.json`). Short version:

- **Bootloader**: BIOS/UEFI to 64-bit mode, hands off to the kernel
- **Kernel**: freestanding x86_64 with SMP bring-up, paging, GDT/IDT, ACPI/PIC, IPC, capability enforcement, and device drivers
- **dominion-core**: filesystem, networking, crypto, ML, browser, desktop (safe Rust, host-testable)
- **Dominion runtime**: interpreter, DCG compiler, polyglot frontend

The kernel is small; the core library is large. That's intentional: most OS logic lives in the core, where it can be tested on the host without booting.

---

## Infrastructure we need to set up

DominionOS has the code for a distributed network (DominionLink), a package repository, and a compute pool. None of it is deployed yet. The full setup guide is in `INFRASTRUCTURE.md`. Short version of what's needed:

**DominionLink bootstrap nodes** (3-5 for redundancy): 2 vCPU, 4 GB RAM, 100 GB SSD each. Listen on UDP 5000 (DHT). Geographic spread helps.

**Package repository**: 1 server, 1 TB storage, PostgreSQL for metadata, HTTP API. TCP port 6000.

**Compute pool coordinator**: lightweight, TCP port 7000.

**Estimated monthly cost**: $50-100 for Phase 1. Domain registration (dominion.link or similar) on top.

See `INFRASTRUCTURE.md` for exact setup steps, config file formats, and security considerations.

---

## Contributing

Contributors welcome. Rules are in `CONTRIBUTING.md`. Key points:

- One thing per PR. If it touches the kernel and the shell, split it.
- Must compile, pass all existing tests, no benchmark regressions.
- Read the spec before touching the code, `docs/architecture.md` and the subsystem docs are the source of truth.
- For major features: open an issue first. For small fixes: PR is fine.

See `AI_DEVELOPMENT_GUIDELINES.md` for the spec-first workflow.

---

## Spec-first development

Full guide in `docs/AI_AGENT_MANUAL.md`. One-paragraph version:

Before any major feature addition, research the subsystem in `docs/architecture.md`, then update or add the relevant spec in `docs/`, then write the code against the spec with tests and benchmarks. For minor changes, update the spec and add a test. Specs are ground truth. Code serves the spec, not the other way around. If the spec is wrong, fix the spec first, then fix the code.

---

## License

Dual licensed. Details in `LICENSE.md`.

- Non-commercial (individuals, research, education): AGPLv3. Free, full source, share improvements.
- Commercial (corporations): paid license required. Contact us.

Attribution is always required:
```
DominionOS, developed by Cognitive Industries (https://cognitive-industries.org)
```

---

## Development status and roadmap

This is an early-stage research prototype. Updates may be sparse unless there's real interest or something specific needs attention, not abandoned, just not a full-time project unless that changes.

**What's coming (in rough priority order)**
1. Wire up the frontend, terminal command set, shell consolidation, GUI wiring
2. Broader hardware coverage, more NICs, real-metal validation of the existing storage/NIC drivers
3. Formal verification proofs for the security model
4. Public DominionLink bootstrap nodes
5. Package repository
6. Preemptive scheduling
7. Multi-user support
8. GPU acceleration (far out)
9. ARM64 / RISC-V (far out)

---

## Contact and links

- Website: [cognitive-industries.org](https://cognitive-industries.org)
- Email: [contact@cognitive-industries.org](mailto:contact@cognitive-industries.org) (licensing, custom development, support)
- Architecture: `docs/architecture.md`
- Subsystem map: `docs/subsystem-manifest.json`
- Feature matrix: `FEATURES.md`
- Build guide: `DEVELOPMENT.md`
- Hardware compatibility: `HARDWARE.md`
- Infrastructure setup: `INFRASTRUCTURE.md`
- Contributing: `CONTRIBUTING.md`

---

*DominionOS is a Cognitive Industries research project. Real security model, capable backend, early frontend. Released because people asked.*

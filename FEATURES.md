# AetherOS Features & Roadmap

Quick reference for what's implemented, what's partial, and what's planned.

---

## ✅ Fully Implemented

### Core System
- ✅ Boots on QEMU and real hardware
- ✅ Safe-mode terminal (ASH shell)
- ✅ Capability-based security model
- ✅ Content-addressed immutable storage
- ✅ Deterministic state machine (reproducible execution)
- ✅ SMP (multi-core) support (up to 16 cores tested)
- ✅ Virtual memory with W^X enforcement
- ✅ Cooperative round-robin scheduler

### Security & Cryptography
- ✅ Unforgeable capabilities with provenance tracking
- ✅ Capability firewall (intra-domain authority enforcement)
- ✅ Capability airlock (inter-domain transfer gateway)
- ✅ Post-quantum cryptography (Lamport XMSS + hybrid)
- ✅ ChaCha20-Poly1305 AEAD encryption
- ✅ AES-256-GCM encryption
- ✅ Vault with crypto-agility and key rotation
- ✅ Zero-plaintext encrypted storage
- ✅ Deniable storage (hidden domains, coercion resistance)
- ✅ Amnesic (volatile) domains with zero-on-free
- ✅ Runtime attestation with hash-chain provenance
- ✅ Secure time (monotonic, signed, multi-source)
- ✅ Random number generation (seeded DRNG, RDRAND fallback)

### Storage & Persistence
- ✅ Content-addressed object graph (Git-like)
- ✅ Immutable, deduplicated storage
- ✅ Instant snapshots (commit/rollback)
- ✅ Versioning with commit history
- ✅ Virtualized filesystem (VFS) as projection of object graph
- ✅ Persistence to virtio-blk disk
- ✅ Crash recovery via integrity hashes
- ✅ Shamir secret sharing for key recovery

### Networking
- ✅ Named Data Networking (NDN) forwarding
- ✅ DominionLink (self-certifying IDs + Kademlia DHT)
- ✅ ARP (address resolution)
- ✅ ICMP (ping)
- ✅ UDP datagram transport
- ✅ DNS bridge (AetherLink names ↔ DNS)
- ✅ Hybrid Logical Clocks (HLC) for causal ordering
- ✅ PQ-signed session keys

### Language & Runtime
- ✅ Aether language (lexer, parser, interpreter)
- ✅ Object-centric data model
- ✅ Capability-gated cells
- ✅ Parallel execution (@CPU/@GPU/@NPU decorators)
- ✅ Semantic primitives (Identity, Latent, Tensor)
- ✅ Foreign function interface (POSIX, Win32)
- ✅ WASM sandbox (64 KiB cells, 1M gas limit)
- ✅ LSP (Language Server Protocol) support
- ✅ IDE integration

### Machine Learning
- ✅ Neural network inference
- ✅ On-device training (reverse-mode autodiff)
- ✅ Tensor operations (dense, sparse)
- ✅ Dense layers, MLPs, ReLU/Sigmoid/Tanh activations
- ✅ SGD + momentum, Adam optimizers
- ✅ int8 quantization for edge devices
- ✅ Model serialization (deterministic, content-addressable)
- ✅ Learned XOR on the booted machine (verified)
- ✅ Performance benchmarks (GFLOP/s, steps/s, inference/s)

### Desktop & UI
- ✅ Desktop environment (Windows-like)
- ✅ 9 built-in applications (Desktop, Files, Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings)
- ✅ Floating window manager (z-order, draggable, resizable)
- ✅ Taskbar with app launcher and system tray
- ✅ Composable UI panels (movable, resizable, removable)
- ✅ Unified 2D/3D rendering stack
- ✅ Nanite virtual geometry
- ✅ Adaptive Texture Warping (ATW)
- ✅ HDR (High Dynamic Range) rendering
- ✅ PSR2 (Priority Rotation) for latency
- ✅ IDAG (immediate DAG) rendering
- ✅ Damage-aggregated repaints (incremental)
- ✅ Text engine with font rasterization
- ✅ Vector paths and animations
- ✅ WCAG accessibility support
- ✅ i18n (internationalization)

### Browser
- ✅ Universal browser (native + legacy web)
- ✅ HTML5 parsing and rendering
- ✅ CSS cascade, inheritance, computed styles
- ✅ JavaScript interpreter (no classes/async/regex)
- ✅ DOM operations (createElement, appendChild, innerHTML)
- ✅ Event binding and propagation
- ✅ Content-addressed AetherWeb pages
- ✅ Tor integration with bootstrap gating
- ✅ HTTPS/TLS (basic)

### Devices & Drivers
- ✅ PCI bus enumeration
- ✅ Driver synthesis from declarative specs
- ✅ MMIO capability bounding (drivers can't escape device window)
- ✅ virtio-pci transport
- ✅ virtio-blk (block device driver)
- ✅ virtio-net (network driver)
- ✅ PS/2 keyboard
- ✅ PS/2 mouse

### System Services
- ✅ Task management (process scheduling)
- ✅ Session lifecycle (login, logout, lock)
- ✅ Settings persistence
- ✅ Identity management
- ✅ Compliance (GDPR, data lifecycle, consent)
- ✅ Backup/restore
- ✅ Fleet synchronization (multiple devices)
- ✅ Update/upgrade lifecycle

### Testing & Benchmarking
- ✅ Unit tests (700+)
- ✅ Integration tests (100+)
- ✅ Property-based testing
- ✅ Fuzzing harness
- ✅ Performance benchmarks (boot, I/O, crypto, ML, process isolation)
- ✅ Determinism verification

---

## ⚠️ Partially Implemented

### Frontend / UX
- ⚠️ **Terminal input routing** - Output works; input buffering incomplete
- ⚠️ **Shell architecture** - Multiple shell implementations need consolidation
- ⚠️ **Composable UI** - Implemented on Desktop; not extended to all pages
- ⚠️ **Browser integration** - Renders HTML/JS but some features incomplete
- ⚠️ **IDE features** - LSP works; debugging incomplete

### Hardware Support
- ⚠️ **Storage drivers** - virtio works; NVMe, SATA need implementation
- ⚠️ **Network drivers** - virtio works; real NICs need drivers (Broadcom experimental)
- ⚠️ **Input devices** - PS/2 works; USB HID not implemented
- ⚠️ **Graphics** - Software framebuffer only; no GPU acceleration

### Networking & Distribution
- ⚠️ **AetherLink network** - Protocol implemented; public bootstrap nodes not deployed
- ⚠️ **Package repository** - Versioning works; central repo not deployed
- ⚠️ **Compute pool** - Framework done; coordinator not deployed
- ⚠️ **Multi-node firewall** - Single-node only; distributed validation deferred

### Security (Production Hardening)
- ⚠️ **Formal verification** - Security model sound; proofs not completed
- ⚠️ **Measured boot** - Architecture defined; TPM integration incomplete
- ⚠️ **Hardware crypto** - Software ChaCha20 only; AES-NI not used
- ⚠️ **ML threat detection** - Statistical baselines only; ML models pending

### Machine Learning
- ⚠️ **Generative compression** - Neural codecs specified; not coded
- ⚠️ **RL-based optimization** - Prefetch/cache tuning pending
- ⚠️ **Multi-model inference** - Single model at a time; ensemble pending

---

## ❌ Not Yet Implemented

### Hardware
- ❌ **NVMe drivers** (PCIe storage)
- ❌ **USB support** (devices, HID, mass storage)
- ❌ **WiFi drivers** (802.11)
- ❌ **Bluetooth**
- ❌ **Audio drivers** (specified but not coded)
- ❌ **GPU drivers** (NVIDIA, AMD, Intel)
- ❌ **Touchscreen support**
- ❌ **Trackpad support** (PS/2 mice work; trackpads don't)

### Advanced Networking
- ❌ **Public AetherLink bootstrap nodes** (needs deployment)
- ❌ **Public package repository** (needs deployment)
- ❌ **Compute pool coordinator** (needs deployment)
- ❌ **Worker node registration** (needs web UI)
- ❌ **Incentive model** (rewards/payments)
- ❌ **Community governance**

### Formal Verification
- ❌ **Microkernel proofs** (Coq/Lean formal verification)
- ❌ **Firewall non-forgeability proof**
- ❌ **Airlock non-bypassability proof**
- ❌ **WCET (worst-case execution time) analysis**

### System Features
- ❌ **Preemptive scheduling** (currently cooperative)
- ❌ **Hard real-time support**
- ❌ **Multi-user support** (currently single user)
- ❌ **User accounts** (identity system exists; user management incomplete)

### Advanced Storage
- ❌ **Distributed storage** (multi-node replication)
- ❌ **RAID support**
- ❌ **Erasure coding**
- ❌ **Transparent compression** (specified; not implemented)
- ❌ **Incremental backups** (full backups work)

### Language & Runtime
- ❌ **Classes in Aether** (objects exist; class syntax not in interpreter)
- ❌ **Async/await in Aether**
- ❌ **Regular expressions**
- ❌ **Module system** (packages exist; imports incomplete)
- ❌ **Exceptions** (panics handled; exception model not designed)

### Security Features
- ❌ **Sandboxed device drivers** (currently in kernel)
- ❌ **Container support** (specified; not coded)
- ❌ **Virtual machines** (emulation mode exists; full VM support deferred)
- ❌ **Encrypted swap** (RAM scrub works; swap not yet)
- ❌ **Hardware key attestation** (TPM integration incomplete)

---

## 🗺️ Roadmap

### Phase 1 (Current - Release 1.0) — June 2026
**Focus:** Stable core system
- ✅ Desktop environment functional
- ✅ Basic hardware support (virtio, PS/2)
- ✅ Security model enforced
- ✅ Single-user demo
- ✅ ML inference working
- 🔄 Terminal input wiring (in progress)
- 🔄 Hardware detection (in progress)

### Phase 2 (1.1 - 1.5) — Q3-Q4 2026
**Focus:** Polish and hardening
- 🔄 Formal verification proofs
- 🔄 Measured boot chain (TPM)
- 🔄 Distributed firewall (multi-node)
- 🔄 Public AetherLink network deployment
- 🔄 Package repository deployment
- 🔄 RL-based storage optimization
- 🔄 Real hardware drivers (Intel NICs, storage)

### Phase 3 (2.0) — 2027
**Focus:** Scale and ecosystem
- 🔄 Preemptive scheduling + hard real-time
- 🔄 Multi-user support
- 🔄 Container/VM support
- 🔄 User-space device drivers
- 🔄 Distributed compute pool
- 🔄 Community package ecosystem
- 🔄 GPU acceleration

### Phase 4 (3.0+) — 2027+
**Focus:** Production readiness
- 🔄 ARM64 port (Apple Silicon, Raspberry Pi)
- 🔄 RISC-V port
- 🔄 Formal verification completion
- 🔄 Production security audit
- 🔄 Zero-knowledge proofs for privacy
- 🔄 Federated systems support

---

## Feature Matrix: AetherOS vs Linux vs Windows

| Feature | AetherOS | Linux | Windows 11 |
|---------|----------|-------|-----------|
| **Boot time** | measured (see bench) | ~5-10s | ~15-30s |
| **Memory usage (idle)** | minimal (kernel-only) | ~200-300MB | ~500MB+ |
| **Capability security** | ✅ Native | ⚠️ Partial (SELinux) | ⚠️ Partial (SandboxToken) |
| **Immutable snapshots** | ✅ Yes | ⚠️ LVM/BTRFS | ⚠️ VSS |
| **Deterministic execution** | ✅ Yes | ❌ No | ❌ No |
| **On-device ML** | ✅ Full | ⚠️ Libraries | ⚠️ Libraries |
| **Desktop GUI** | ✅ Yes (basic) | ✅ Yes (mature) | ✅ Yes (mature) |
| **POSIX compatibility** | ⚠️ Partial | ✅ Full | ⚠️ Partial |
| **Hardware support** | ⚠️ Limited | ✅ Extensive | ✅ Extensive |
| **Multi-user** | ❌ No | ✅ Yes | ✅ Yes |
| **Network stack** | ✅ NDN | ✅ TCP/IP | ✅ TCP/IP |
| **Wireless** | ❌ No | ✅ Yes | ✅ Yes |
| **GPU support** | ❌ No | ✅ Yes | ✅ Yes |
| **Production ready** | ⚠️ Research | ✅ Yes | ✅ Yes |

---

## Known Limitations

1. **Single-user only** — Multi-user support deferred to Phase 3
2. **Limited hardware** — virtio and PS/2 only (real drivers coming)
3. **No preemption** — Cooperative scheduling (hard real-time coming Phase 2)
4. **No POSIX** — Aether language only (POSIX layer Phase 2)
5. **No container/VM** — Single namespace (containers Phase 3)
6. **Software rendering only** — No GPU acceleration (Phase 3+)
7. **4 days of work** — This is a prototype. Expect gaps.

---

## How to Request Features

1. **Check the roadmap** — Is it already planned?
2. **Check the spec** — Is there a design doc?
3. **Open an issue** — Describe what you need and why
4. **Consider contributing** — We welcome PRs. See CONTRIBUTING.md

---

## Tracking Progress

For real-time updates:
- **GitHub issues:** Feature requests and bug reports
- **GitHub discussions:** Design discussions and ideas
- **Architecture docs:** `docs/architecture.md` and subsystem specs

---

**Questions?** contact@cognitive-industries.org

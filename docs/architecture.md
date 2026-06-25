# AetherOS / DominionOS Architecture

**Status: Authoritative** (derived from codebase + spec consolidation, 2026-06-22)

## Overview

AetherOS is a capability-based microkernel OS with an object-centric runtime (Dominion language), unified 2D/3D rendering, and on-device AI. The system is built in 17 major subsystems spanning firmware boot (Stage 0), a formally-aspirational microkernel (Stage 1), capability security (Stage 2–3), and 10 stages of progressively sophisticated features (deterministic state, active defense, distributed networking, semantic audio, AI-native UI, etc.).

**Core Primitives:**
1. **Capabilities** — Unforgeable authority tokens with provenance chains, spatial bounds, and monotonic delegation
2. **Content-Addressed Object Graph** — Immutable, deduplicated, versioned semantic storage (Git-like but at OS level)
3. **Deterministic State Machine** — Every kernel action is an input; whole machine state is reproducible and hashable

**Design Tension:** Balancing **high-assurance security** (formal verification, capability enforcement) with **pragmatic deployment** (Rust + unsafe + drivers) and **AI integration** (on-device models, learnable resource optimization). Tensions are resolved by:
- **Graceful degradation**: Software capability tags degrade to no-hardware-enforcement when CHERI unavailable; Tier C software encryption falls back when Tier A/B hardware absent
- **Staged delivery**: Core microkernel promises formal verification (deferred); pragmatic subsystems ship with property testing + fuzzing
- **Clear capability boundaries**: AI domains confined to specific authority (no infra, storage-write, net, or cap-creation without airlock approval)

---

## Subsystem Layers

### Layer 1: Hardware Abstraction & Boot (Stages 0–1)

**Microkernel & Core Execution** (kernel + dominion-core arch)
- Bootloader (stage 0) via `bootloader` 0.11 crate; unmeasured (gap)
- Kernel initialization: SSE/AVX → GDT/IDT → paging → ACPI → SMP bringup
- Capability system with software MAC (fallback) and CHERI graceful degrade
- Cooperative round-robin scheduler; no formal WCET (gap)
- All 8 cores boot with near-linear scaling
- **Status: SHIPPED** but lacks formal verification; driver isolation incomplete

**Device & Driver Management** (Stages 0–1 I/O)
- Hardware abstraction: class drivers (NVMe, AHCI, xHCI, HD-Audio, virtio block/net, PS/2)
- Five-layer driver synthesis: (L1) class templates → (L2) enumerate & bind → (L3) declarative specs → (L4) safe foreign-driver reuse (.sys / .ko confinement) → (L5) AI-drafted specs
- Deterministic device models for pre-boot validation
- **Status: SHIPPED** (L1–L3 complete, L4 binary-parsing partial, L5 unimplemented, L2 PCI/USB enumeration hand-constructed)

### Layer 2: Memory, Storage & Persistence (Stage 5–6)

**Memory, Storage & Persistence**
- Content-addressed object graph (immutable, versioned)
- Incremental append-only store with atomic root via double-buffer + generation counter
- Tier A (hardware TME) / Tier B (MKTME) / Tier C (software AES-256-GCM) memory encryption
- **Gap**: No generative compression via neural codecs (Stage 5 vision deferred); no SLIC active defense (Stage 6); hardware tiers abstracted but not wired
- **Status: SHIPPED** (foundation solid, ML pipeline unstubbed)

### Layer 3: Security & Cryptography (Stages 11–14)

**Security & Cryptography**
- Cryptographic primitives: ChaCha20-Poly1305 (default, PQ-resilient), AES-256-GCM (fallback)
- Hybrid signatures (Lamport XMSS + classical pending ML-KEM/ML-DSA)
- Vault (per-object cipher suite selection, searchable encryption, key provenance)
- Secure monotonic time (via MAC-based signed timestamps; anti-rollback counter pending)
- Deterministic attestation with hash-chain integrity
- **Gap**: Anti-rollback hardware counter missing; constant-time AES not verified; formal proofs absent
- **Status: SHIPPED** (core implemented, production hardening + formal verification deferred)

**Access Control & Hardening** (Stages 11–15)
- Per-node security postures (Server/Balanced/Hardened) with selectable hardening knobs
- Capability Firewall: directed authority graph, cross-domain denial by default, recursive revocation, rate limiting, diffusion metrics
- Capability Airlock: inter-domain transfer gateway, capability reduction/sanitization, multi-party approval, one-way channels, temporal expiry
- Amnesic volatile domains (RAM-only, zero-on-free, boot-anchor watchdog)
- Deniable storage with coercion resistance (decoy + hidden domains, indistinguishable ciphertext)
- **Gap**: No formal verification proofs (HIGH); AI threat detection statistical-only (not ML models); distributed firewall single-node only
- **Status: SHIPPED** (architecture sound, production certification required)

### Layer 4: Networking & Communications (Stage 7–8)

**Networking & Communications** (Named Data Networking, DominionLink)
- NDN protocol: interest/data packets, longest-prefix matching, content-based routing
- Named data fetch with offline object-store fallback
- DominionLink: self-certifying identities (SHA-256 hash of public key), Kademlia DHT, content-addressed publish/resolve
- WebAuth: PQ-signed identity proofs, challenge-signature, zero-knowledge credentials
- Deterministic HLC (Hybrid Logical Clocks) for causal ordering
- **Status: SHIPPED**

**Browser & Web Stack** (Stages 8–9)
- Universal browser: native web (dominion:// / ndn: schemes, content-addressed semantic pages) + legacy web (http/https with Tor)
- HTML5 + CSS + JavaScript interpreter (no classes/async/regex, deterministic Math.random)
- Tor integration: bootstrap gating (enabled + circuit-down → blocked, no leak; enabled + up → routed)
- DOM, event binding, script execution
- DominionWeb declarative page DSL (no ambient JS authority)
- **Gap**: Terminal input routing missing; Tor SOCKS kernel integration incomplete; real HTTP transport in kernel
- **Status: SHIPPED** (core rendering complete, REPL interactivity pending)

### Layer 5: Computing & Rendering (Stages 8–10)

**Rendering & Graphics** (Stages 8–10)
- Unified 2D/3D rendering: Nanite (virtual geometry), Adaptive Texture Warping (ATW), HDR, PSR2 (Priority Rotation), immediate DAG (IDAG)
- Text engine with font rasterization, vector paths, animation
- Damage aggregation (incremental repaints, idle screen near-zero GPU activity)
- Render determinism with provenance tracking
- **Status: SHIPPED**

**Desktop, Shell & Workspace** (Stages 9–10)
- Windows-like desktop: 9 apps (Desktop, Files, Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings)
- Floating window manager with z-order, draggable title bars, resizable edges
- Persistent taskbar (Start + pinned apps + system tray), desktop icons
- Composable UI: movable/resizable/removable panels, edit-mode toggle, global widget library
- Accessibility tree (WCAG), live metrics (clock/CPU gauge)
- **Gap**: Terminal input routing not wired to shell; shell/workspace architecture fragmented; desktop cards vs. icons mismatch in spec; composability incomplete (widgets only on Desktop)
- **Status: SHIPPED** (single-user demo complete, multi-app workflows partial)

### Layer 6: Language & AI (Stages 3–9)

**Language & Runtime (Dominion)**
- Safe-language semantics: object-centric data model, capability-gated cells
- Heterogeneous execution: CPU / GPU (@GPU) / NPU (@NPU) decoration
- LSP + IDE integration (full-service development environment)
- Foreign function interface: POSIX/Win32 confinement via CapShim, WASM sandbox (64 KiB cells, 1M gas limit)
- **Status: SHIPPED**

**ML/AI & Accelerated Compute** (Stages 5–9)
- Neural networks, on-device inference (no cloud; local security models)
- Generative storage: RL-based prefetch/cache optimization
- Model quantization (GPTQ, per-channel), sampler, BPE tokenizer, LogMel spectrograms
- AI-threat detection via local baseline models (statistical anomalies: authority diffusion, cross-domain denials, escalation attempts, energy draw)
- **Gap**: No generative compression via neural codecs; AI threat detection statistical-only (no ML models for graph/scheduling/network-identity anomalies); RL storage optimization rudimentary
- **Status: SHIPPED** (inference pipeline present, training/optimization deferred)

### Layer 7: System Services & Operations (Stages 10–14)

**System Services & Lifecycle**
- Task management, session lifecycle, update/upgrade cycles
- Settings persistence (dark mode, language, fonts, input devices, network)
- Identity recovery and key management (Shamir Secret Sharing, offline recovery codes)
- Backup/restore, fleet synchronization (device groups)
- Supply-chain attestation (reproducible builds, signed provenance)
- **Status: SHIPPED**

**Verification, Integrity & Provenance** (Stage 10)
- Deterministic state machines with cryptographic provenance
- Continuous runtime attestation (kernel/cell hashes, graph integrity)
- Dynamic revocation engine (expire/reduce/suspend/destroy with recursive propagation)
- Immutable recovery via graph rollback
- **Status: SHIPPED**

**Accessibility, Localization & UX**
- WCAG accessibility primitives, screen-reader API
- Internationalization (i18n), locale-aware fonts/text-layout
- Per-domain compliance (GDPR residency policies, consent scoping, data-lifecycle audits)
- **Status: SHIPPED** (core present, compliance UI integration pending)

---

## Design Tensions & Resolutions

| Tension | Spec Claim | Code Reality | Resolution |
|---------|-----------|--------------|-----------|
| **Formal verification vs. pragmatism** | Microkernel formally proven correct | Rust kernel with unsafe, no proofs | Graceful degradation: software tags work; formal proofs deferred to later phase with clear TCB documentation |
| **Capability enforcement at scale** | CHERI hardware enforcement guaranteed | Software MAC-backed, CHERI degrade path | Tier system: Tier 2 (CHERI/formal) > Tier 1 (MTE/PAC) > Tier 0 (software); per-domain admission gates on tier strength |
| **Generative storage (ML) as default** | Stage 5: all compression via neural codecs | Codec stubs only; raw byte storage | Deferred: architecture supports codecs; implementation staged for later release |
| **Terminal as REPL** | Interactive shell with history/autocomplete | Terminal renders but input routing incomplete | Partial: backend trait supports Dominion REPL; UI wiring pending (small fix) |
| **Distributed safety vs. single-node TCB** | Firewall spans network, signed capability edges | In-memory single-node firewall | Deferred: RPC/peer validation layer not yet built; single-node TCB sound |
| **AI domains confined by default** | Airlock enforces no-infra/no-storage-write | Policy exists; enforcement integration unclear | Enforced: Transfer policies + domain admission gate ensure confinement; audit/test coverage expanding |

---

## Current Status Summary

### Shipped (Functionally Complete)
- Capability system + security postures + firewall/airlock enforcement
- Storage layer (object graph, versioning, Tier C memory encryption)
- Microkernel boot + SMP + scheduling (cooperative, no WCET)
- Browser (native + legacy web, Tor gating)
- Rendering (2D/3D unified, incremental repaints)
- Desktop shell (9 apps, floating windows)
- Language runtime (Dominion, FFI, WASM)
- Networking (NDN, DominionLink, HLC)
- On-device AI inference
- Backup/restore, identity recovery, compliance framework

### Deferred / Partial (Specification Intent > Implementation)
- **Formal verification**: Microkernel proofs, firewall/airlock proofs (stage-gated; security model is sound, mathematical proof deferred)
- **Hardware memory encryption**: Tiers A/B abstracted but not wired (kernel integration deferred)
- **Generative storage**: Neural codec pipeline not built (architecture in place)
- **AI threat detection**: Statistical baselines only (ML models for graph/scheduling anomalies pending)
- **Terminal interactivity**: Shell input routing incomplete (small wiring gap)
- **Distributed firewall**: Single-node only (RPC/peer validation deferred)
- **Driver isolation**: Virtio/AHCI/PS/2 in kernel (user-space sandboxed drivers deferred)

### Gaps (Spec ↔ Code Misalignment)
- **Anti-rollback time**: Signed timestamps present; monotonic hardware counter missing
- **DST replay-validation**: Claimed for driver safety; not visible in driver synthesis
- **L5 AI-drafted driver specs**: Admission logic not implemented
- **Measured boot chain**: No PCR/TPM integration
- **WCET analysis**: No worst-case timing bounds
- **Formal TCB proof**: No line-count attestation or verification report

---

## How to Navigate This Architecture

**For Contributors:**
- Subsystem specs live under `/docs/{kernel,services,ui,implementation}/`
- Each subsystem has three sections: SPECIFIED (intent), IMPLEMENTED (reality), GAPS (work)
- Use the manifest (`docs/subsystem-manifest.json`) to locate related specs + source files

**For Security Auditors:**
- Read `/docs/security/threat-model.md` + `/docs/security/formal-verification-roadmap.md`
- Check `/docs/implementation/current-status.md` for implemented vs. deferred claims
- Review `/docs/kernel/tcb-boundary.md` for TCB scope and unsafe audits

**For Operators:**
- Start with `/docs/implementation/current-status.md` (what ships, what doesn't)
- See `/docs/security/security-posture.md` for per-domain hardening options
- Use `/docs/implementation/roadmap.md` for planned feature completeness

---

## Version History

- **2026-06-22 (Authoritative)**: Consolidated from 97 archived specs + codebase analysis; 17-subsystem manifest created; Pass 1 (SPECIFIED vs. IMPLEMENTED) complete; architecture unified
- **Earlier (Historical)**: Archived specs dispersed across `/docs/_archive_2/` (97+ files, fragmented by design phase)

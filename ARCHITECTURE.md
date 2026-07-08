# DominionOS v2 Architecture

## System Overview

**DominionOS** is a capability-first operating system targeting x86_64 & aarch64 with a unified, self-verifying trust model. The kernel (a freestanding x86_64 kernel with SMP bring-up, paging, GDT/IDT, ACPI/PIC, and device drivers) provides process isolation and capability enforcement; **dominion-core** (safe Rust, `#![forbid(unsafe)]` in the core) implements a semantic compute substrate—a Dominion language interpreter, named-data networking, browser/shell, rendering pipeline, ML inference, and persistent object graph. The system executes as a **single application per boot** with nine integrated apps (Desktop, Files, Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings) sharing one filesystem, scheduler, and world graph.

**Key architectural tension:** The spec promises Stage 3-11 vision (formal microkernel verification, heterogeneous GPU/NPU scheduling, generative storage, capability-enforced driver synthesis, AI-assisted security, deniable storage) but the implementation delivers **Stage 2-4 pragmatic substrate** (working memory safety, content-addressed objects, cooperative scheduling, capability model, browser/shell/rendering). Most subsystems are **60-70% specified and 40-60% implemented**; critical gaps exist in formal verification, hardware integration (GPU/CHERI), and distributed multi-agent orchestration.

---

## Design Patterns & Invariants

**Determinism:** Every execution path is seeded via `Drng` (hash-DRBG) and recorded in the deterministic state log (`state.rs`). Replay of identical inputs yields identical system behavior (for attestation, forensics, and reproducible tests). Timestamps are recorded as input events; wall-clock reads are not.

**Capability monotonicity:** Derived capabilities ⊆ source capability. Capability checks use bitmask rights enforcement (`READ`, `WRITE`, `EXECUTE`, `DELEGATE`, `REVOKE`). Revocation is recursive: invalidating a capability invalidates all derived tokens. No ambient authority; all execution is gated on an explicit capability.

**Content addressing:** Objects are identified by SHA-256 hash of canonical bytes. Immutable; once written, bytes never change. Enables deduplication, caching, and trustless verification. Filesystem projects the object graph as a tree of paths.

**Zero-copy messaging:** Inter-domain communication via immutable object references; data flows as `ObjectId` tokens over IPC, not bytes.

**Convergent + linearizable consistency:** State is replicated per-core via HLC-ordered delivery (convergent CRDT path); value-bearing state (money, invariants) uses quorum consensus (linearizable). BFT reserved for untrusted fleet.

---

## Subsystem Layers

### 1. **Microkernel & Core Execution**
**What:** A freestanding x86_64 kernel (bootloader + SMP bring-up, page tables, GDT/IDT, ACPI/PIC, and device drivers for storage, network, USB, and PS/2). Not formally verified; pragmatic stage 2 prototype.

**Provides:** Process isolation (none—single app), IPC (none—monolithic), interrupt handlers, memory management (OffsetPageTable + frame allocator, W^X enforced).

**Critical gaps:** (1) No formal verification (spec promises seL4-grade; actual kernel is Rust + unsafe drivers). (2) No WCET analysis (cooperative scheduler, unbounded delays possible). (3) Drivers in kernel (virtio/AHCI/PS/2) rather than sandboxed userspace. (4) No measured boot or TPM integration (PCR chain missing).

**Key contract:** Kernel handles CPU/memory/interrupts only; all policy (IPC, scheduling, capability) delegated to dominion-core. MMIO capability bounding happens at driver invocation, not kernel fence.

---

### 2. **Capability System & Access Control**
**What:** `capability.rs` (provenance chains, monotonicity), `cheri.rs` (soft/hard tags), `firewall.rs` (intra-domain authority graph), `airlock.rs` (inter-domain transfer gate), `secprofile.rs` (per-node hardening postures).

**Provides:** Unforgeable tokens, recursive revocation, capability rate limiting, domain isolation, cross-domain policy enforcement, amnesic/deniable storage.

**Critical gaps:** (1) No formal proofs of non-forgeability/monotonicity/non-bypassability (required for TCB). (2) CHERI tags always false (no hardware CHERI integration; software MAC-based only). (3) Anti-rollback monotonic counter missing (threat model requires it). (4) AI threat detection only statistical baselines (no ML models for anomaly correlation).

**Key contract:** Every action requires a capability; every capability is monotonically bounded. Firewall enforces reachability; airlock enforces cross-domain policy. No authority can be escalated once granted.

---

### 3. **Scheduling & Multikernel**
**What:** `smp.rs` (ACPI MADT, INIT-SIPI-SIPI, 8-core bring-up), `pool.rs` (priority-ordered work stealing), `multikernel.rs` (per-core replicas, HLC-ordered delivery, convergent state model).

**Provides:** SMP task dispatch, cooperative round-robin scheduling, CRDT-backed replicated state (GCounter, OrSet), quorum consensus for linearizable paths, work-stealing load balancing.

**Critical gaps:** (1) Pragmatic SMP (shared page tables, global queue) not per-core CPU drivers (stage 4 spec calls for federated cores). (2) No preemptive priorities (cooperative only). (3) Heterogeneous scheduling (GPU/NPU placement hints parsed but not routed). (4) Multikernel model layers not wired to real APs.

**Key contract:** Tasks are scheduled via topological sort (DAG); cores claim work atomically. State consistency guaranteed by HLC + quorum agreement. Placement hints (`@GPU`, `@NPU`) advisory to governor (not executor).

---

### 4. **Memory, Storage & Persistence**
**What:** `persist.rs` + `objstore.rs` (content-addressed incremental append-only store), `vfs.rs` (filesystem as object-graph projection), `memenc.rs` + `memcrypt.rs` (Tier C software per-object encryption, Tiers A/B hardware abstracted but not wired), `durability.rs` + `recovery.rs` (Shamir secret sharing, offline recovery codes, scrubbing GC).

**Provides:** Immutable object versioning, instant rollback to any commit, capability-gated filesystem operations, per-object encryption with Tier-A/B hardware degradation, deterministic scrubbing.

**Critical gaps:** (1) No generative compression (Stage 5 spec requires neural codecs; Stage 6 SLIC watermarking). (2) Tier A/B hardware (TME/SME/MKTME) not integrated (code probes but always falls back to Tier C). (3) No replication/erasure coding across nodes. (4) Deduplication is logical (content-addressed) but not block-level.

**Key contract:** Objects are immutable once committed; revisions form a hash chain. Every write creates a new object; paths re-alias to new root. Encryption is transparent: high-sensitivity objects use Tier C; others degrade to hardware if available.

---

### 5. **Security & Cryptography**
**What:** `crypto.rs` (algorithm agility, hybrid Lamport signatures), `chacha.rs` + `memcrypt.rs` (AEAD: ChaCha20-Poly1305 + AES-256-GCM), `vault.rs` (crypto-agile sealing), `tokensig.rs` (PQ XMSS token signing), `time.rs` (monotonic logical clock, signed timestamps), `attest.rs` (deterministic measurement chains).

**Provides:** PQ-resistant signatures, dual-family AEAD, searchable encryption, capability tokens with cryptographic provenance, attestable reproducibility.

**Critical gaps:** (1) Production crypto parameters not finalized (Lamport illustrative, not production-grade; ML-KEM/ML-DSA stubs only). (2) Constant-time AES-GCM not verified (table-lookup core explicitly not constant-time). (3) Anti-rollback counter not implemented (time.rs has soft MAC; no hardware monotonic RTC binding). (4) Tier A/B hardware encryption not wired.

**Key contract:** Crypto layer is pluggable by id; both AEAD families must verify for dual-family capability to succeed. Key material lives encrypted at rest and is never decrypted except at the moment of use.

---

### 6. **Networking & Communications**
**What:** `ndn.rs` (name-based routing: Interest/Data model, PIT, FIB, CS), `dominionlink.rs` (content-addressed object transport, Kademlia DHT), `pubsub.rs` (reactive subscriptions, standing queries, CRDT topics, presence), `identity.rs` + `webauth.rs` (MasterSeed HD wallet, per-service pseudonymous identities, passkey auth), `legacynet.rs` (TCP/IP overlay, UDP encapsulation, NAT traversal).

**Provides:** Identity-based networking (no pre-agreed trust anchors), push/pull duality (NDN Interest + reactive pub/sub), encrypted delivery (PQ KEM + AES-256-GCM), offline-first caching, mobility via content re-announce.

**Critical gaps:** (1) Account lifecycle API missing (create/revoke/export as instant operations not exposed). (2) ZK selective disclosure not implemented (no credential issuer or ZK proof layer). (3) Device attestation not integrated into auth (step-up auth requires posture check). (4) SOCKS5 Tor routing declared in browser but kernel-level glue missing.

**Key contract:** Data identity is producer-signed; no pre-distributed PKI. Subscriptions are capabilities; access revocation is recursive. Sessions are temporal tokens scoped to rights.

---

### 7. **Language & Runtime (Dominion)**
**What:** `lang/` subsystem (lexer, parser, AST, interpreter), `datatypes.rs` (Tensor, HyperVector, CRDT, HomomorphicCiphertext, QubitState, Manifold, BigInt, Decimal, Rational, Complex, etc.), `wasm.rs` (polyglot sandbox VM with gas metering + host-call capability gates).

**Provides:** Type-safe, memory-safe language with semantic primitives (Identity, Time, Resource, Latent), hardware hints (`@CPU/@GPU/@NPU`), affine/linear types, parallel map operator (`=>`), deterministic execution.

**Critical gaps:** (1) No Deterministic Compute Graph (DCG) IR; tree-walking interpreter only (spec mandates compilation to IR + proofs). (2) Capability monotonicity not compile-time verified. (3) Affine invalidation is semantic not cryptographic (no hardware capability zeroization). (4) No heterogeneous multikernel routing (placement hints parsed but not dispatched). (5) Cells are syntactic scopes, not architectural isolation units.

**Key contract:** All values are deterministic (seeded RNG, no FPU env dependency). Capability checks are enforced at runtime; violations trap with `CapabilityFault`. Parallel sections (`=>`) are implicit parallelism over collections (data flow is safe but not automatically inferred).

---

### 8. **Browser & Web Stack**
**What:** `browser.rs` (tab management, Tor control plane, route resolution: Direct/Tor/Blocked), `webengine.rs` (HTTP roundtrip abstraction, NativeWeb publish/resolve over DominionLink), `html.rs` + `css.rs` + `js.rs` + `dom.rs` (full HTML/CSS/JS parsing, layout, deterministic JS engine), `dominionweb.rs` (native semantic pages: no DOM, no JS, capability-gated actions), `toolkit.rs` (flex layout, widget tree, theme tokens, GPU + framebuffer renderer backends).

**Provides:** Dual web paths (native: capability-gated semantic pages; legacy: full HTML/CSS/JS in sandboxed VM), Tor routing, content-addressed page caching, deterministic rendering (replay-identical).

**Critical gaps:** (1) **No authentication layer** (zero account/login/identity integration; legacy vault autofill unimplemented; FIDO2 missing). (2) SOCKS5 kernel transport layer missing (Tor UI works, circuit doesn't). (3) Async/cooperative transport loop incomplete (kernel-only). (4) GPU backend stubbed (toolkit trait defined, shaders missing). (5) WASM binary decoder missing (detect works, execute doesn't).

**Key contract:** Native pages are declarative objects (no JS, no CSS cascade); rendered directly from capability-scoped DominionLink. Legacy pages execute in confined WASM VM with default-closed capabilities (must grant Net/Surface explicitly). Same layout/scroll/hit-test pipeline for both.

---

### 9. **Rendering & Graphics**
**What:** `render3d.rs` + `nanite.rs` + `scene3d.rs` + `raster3d.rs` + `rdg.rs` (RDG compiler: declare → compile/alias → execute; Nanite LOD; retained scene graph), `atw.rs` + `hdr.rs` + `psr2.rs` (ATW motion reprojection, 16-bit HDR, damage-rect incremental updates), `idag.rs` (Instruction DAG semantic GPU scheduling), `fontgpu.rs` + `vectorpath.rs` + `sdf_shadow.rs` (Bézier glyph engine, vector tiling, SDF shadows), `compositor_svc.rs` (retained compositor, zero-copy media).

**Provides:** Unified 3D/2D render stack, virtual geometry (billions of triangles via LOD), semantic GPU scheduling by priority, HDR compositing, incremental frame updates, deterministic glyph rasterization.

**Critical gaps:** (1) **Compositor service not instantiated in production** (framework exists, never wired into os.rs render loop). (2) GPU backend missing (CPU rasterizer works; shader compilation/dispatch unimplemented). (3) Damage tracking (PSR2) declared but not called. (4) IDAG hardware integration missing (semantic scheduling logic present; no kernel GPU slice management).

**Key contract:** Scene is a retained graph of transforms/meshes/text/particles. RDG compiles to memory-aliased, barrier-aware command stream. GPU prioritizes by IDAG slice (InputLatency > UIRender > MediaDecode > BackgroundCompute) but execution is CPU-only.

---

### 10. **Desktop, Shell & Workspace**
**What:** `os.rs` (unified shell: 9 apps, window manager, accessibility tree, taskbar/topbar, Start menu, persistence), `window.rs` (generic floating window manager), `desktop_page.rs` (infinite canvas with draggable icons), `terminal.rs` (scrollback + input line + pluggable backend), `browser.rs` / `files.rs` / `editor.rs` / `ide.rs` / `explorer.rs` / `taskman.rs` / `settings.rs` (app pages).

**Provides:** Unified app host with shared filesystem/scheduler/world, floating window management, desktop launchers, terminal REPL, multi-pane layout, live metrics, persistent state.

**Critical gaps:** (1) Shell/workspace modularity fragmented (`shell.rs` unused, `workspace.rs` unintegrated time-travel undo). (2) Composable UI (`Board`) only on Desktop (spec says all pages). (3) Live kernel enumeration missing in Explorer (seeded objects, not live). (4) Account identity hardcoded ("Jayden", not from identity system). *(Terminal input routing is wired: the ASH shell and GUI desktop terminal are both interactive and test-covered.)*

**Key contract:** All 9 apps share one `FileSystem`, `Scheduler`, `World`, and `Board`. Persistence uses double-buffered root + manifest. Window stacking is z-ordered; focus routing is hit-test → text field → app → shell hotkeys.

---

### 11. **ML/AI & Accelerated Compute**
**What:** `ml.rs` (matmul + autodiff + QTensor int8 + device cost model), `nn/` (attention, RoPE, RMSNorm, SwiGLU, embeddings, samplers, tokenizers; model.rs `.aem` loader), `agent.rs` (framework: AgentBus, AgentSnapshot, NodeState, ActionKind, agent_dispatch gate), `neural.rs` + `coldcomp.rs` (predictive delta+RLE codec, verified decompression, grid_snap determinism).

**Provides:** Bit-exact deterministic matmul, quantized inference (int8), weight streaming with KvCache, device-agnostic cost model, on-device language models, agent action dispatch.

**Critical gaps:** (1) **Agent harness not wired to os.rs** (framework exists, integration deferred). (2) ModelCodec not registered (models parse but don't load via codec). (3) Model architectures incomplete (Gemma/Whisper/Kokoro/Segmind operators partially done). (4) No FFT for Whisper (log-mel feature extraction missing). (5) Offline converter pipeline empty (no Python quantizer). (6) Cargo feature matrix missing (no `model-gemma`, `agent` gates).

**Key contract:** All inference is deterministic (grid_snap + seeded RNG + fixed-lane accumulation). Models are capability-gated at load; execution is routed to preferred device (CPU/GPU/NPU) via bandit. Agent loop runs percieve → plan → act → observe → respond with rollback-safe state transitions.

---

### 12. **Device & Driver Management**
**What:** `driver.rs` (L3 proof of concept: DeviceSpec + register-op programs + MMIO capability bounding), `drivergen.rs` (L1 class drivers, L2 enumerate-and-bind, DMA capability model), `foreign.rs` (L4 safe reuse: Windows NDIS + Linux KPI shim), `discovery.rs` (static catalog + live merging).

**Provides:** Capability-bounded device drivers, fail-closed spec parsing, self-describing hardware binding, foreign-driver confinement.

**Critical gaps:** (1) **DST replay-validation not implemented** (core safety mechanism missing; specs bind without proof). (2) L2 enumerate-and-bind missing real PCI/USB/ACPI/DT parsers (only hand-constructed HwDescriptor). (3) L5 AI-drafted admissions not implemented (driver-spec authoring still manual). (4) Foreign binary parsing incomplete (PE/ELF `.sys`/`.ko` claimed but not shown). (5) Conformance gate (90% corpus) not visible.

**Key contract:** Drivers are untrusted; spec must be well-formed (reject at bind). Capability bounds MMIO window + DMA region + IRQ. Poll timeouts prevent hang; malformed ops trap immediately.

---

## Cross-Cutting Concerns

**Determinism:** All subsystems log inputs (clock tick, IRQ, network packet) to the state log. Outputs (display frame, NIC packet) are deterministic functions of logged state. Enables replay, attestation, and forensics.

**Capability flow:** Authority flows Microkernel (SMP primitive) → Kernel (IPC gate) → dominion-core (policy enforcer). Firewall/airlock mediate capability delegation. Revocation propagates recursively.

**Hardware abstraction:** `arch.rs` abstracts target (x86_64, aarch64, generic). Tier-0/1/2 enforcement backend pluggable but Tier 0 (software tags) always available. Boot-time `probe()` selects strongest available tier.

---

## Critical Load-Bearing Gaps

| Gap | Impact | Severity |
|---|---|---|
| **Formal verification missing** | TCB claims unmeasurable; no proof of microkernel correctness, firewall monotonicity, or driver containment | 🔴 CRITICAL |
| **Compositor not instantiated** | Unified 3D render stack never runs; system uses legacy flat compositing | 🔴 CRITICAL |
| **Agent harness unintegrated** | On-device AI loop (percieve/plan/act) is framework only; no live dispatch | 🔴 CRITICAL |
| **DST replay-validation missing** | Drivers bind without safety proof; spec containment unverified before hardware touch | 🔴 CRITICAL |
| **Heterogeneous scheduling unimplemented** | GPU/NPU hints parsed but not routed; all compute on CPU thread | 🟠 HIGH |
| **No generative compression** | Storage bloat; data-size exponential growth unaddressed (Stage 5 entirely deferred) | 🟠 HIGH |
| **Authentication layer absent** | Native web has no accounts; legacy browser no passwordless/FIDO2; privacy model incomplete | 🟠 HIGH |

---

## Deployment Readiness

**Prototype maturity:** System boots to interactive 9-app shell with file persistence, browser, interactive terminal, and editor on x86_64 (QEMU + metal tested). Memory safety (forbid unsafe in core) + capability model (monotonic enforcement) + determinism (replay-exact) are solid. Rendering (CPU), scheduler (SMP), storage (content-addressed), and crypto (PQ hybrid) all functional.

**Not production-ready:** Missing formal verification, GPU acceleration, full agent orchestration, and account authentication. Spec's high-assurance claims (seL4-grade kernel, formal proofs, measured boot) remain aspirational; actual kernel is pragmatic prototype. Suitable for research/demo; requires significant hardening for production.


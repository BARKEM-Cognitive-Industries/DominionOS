---
title: "DominionOS / Architecture 2.0 — A Verified, Heterogeneous, Semantic Operating System"
subtitle: "A Consolidated Research-Grade Synthesis of the Design, Security Architecture, Language, and Implementation"
doc_type: consolidated-paper
domain: cross-cutting
status: living-document
standard: ISO/IEC/IEEE 29148 (SRS)
synthesized_from: docs/{architecture,security,language,ui,implementation}/*, GLOSSARY.md, FEATURE-CHECKLIST.md, findings.md
date: 2026-06-20
---

# DominionOS / Architecture 2.0
### A Verified, Heterogeneous, Semantic Operating System

> **A single-file synthesis** of the DominionOS corpus (~60 design documents and the
> `dominion-core` / `dominion-kernel` codebase). It consolidates the vision, the formal
> security architecture, the Dominion language, the user interface, and — crucially — the
> *grounded implementation reality* (what boots, what is tested, what is still silicon-gated).
> Throughout, claims are tagged **[Implemented]**, **[Modeled]** (real semantics in safe
> Rust, hardware mechanism behind a HAL), or **[Vision]** (specified, not yet built), so the
> reader never confuses the destination with the road already travelled.

---

## Abstract

Contemporary operating systems remain tethered to computational assumptions of the 1970s:
the monolithic kernel, hardware privilege rings, the hierarchical filesystem, and
address-based networking — all designed for single-core, homogeneous processors managing
spinning magnetic disks. Adapting these stacks to today's radically heterogeneous and
distributed hardware (high-core-count CPUs, GPUs, NPUs, discrete DMA engines) has produced
extreme complexity, propagating vulnerability, and severe architectural impedance mismatch.

DominionOS is a ground-up redesign that **discards** the monolithic kernel, the privilege ring,
and the file, and **replaces** them with a mathematically verified microkernel,
hardware-enforced capabilities (CHERI), a Single Address Space Operating System (SASOS) resting
on compiler-guaranteed memory safety, a content-addressed semantic object graph, deterministic
whole-machine execution, and an AI-native, object-centric interface — all expressed in one
intralingual language, **Dominion**. The architecture reduces *every* feature and integration
problem to exactly three native primitives: **capabilities**, the **content-addressed object
graph**, and the **deterministic state machine**. Its governing rule, the **P.I.E. Principle**,
assigns *Performance* and *Efficiency* to hardware and *Isolation* and *Security* to
safe-language compilers and capability enforcement.

This paper presents the full stack (Stages 0–14 plus cross-cutting subsystems), the Dominion
language and its extended type system, the security architecture (post-quantum cryptographic
agility, zero-plaintext storage, capability firewall/airlock, zero-knowledge primitives), and a
measured evaluation of the working prototype. The prototype boots on a VM, persists a
content-addressed object graph across reboot, runs a cooperative scheduler, an ELF loader that
executes real x86-64 machine code, a compositor, and a virtio network stack with a real ARP
round-trip; brings up **8 cores with near-linear (8.3×) scaling**; and passes **441 host unit
tests and 87 on-metal tests** with `cargo clippy --all-targets` reporting **zero warnings** and
`dominion-core` compiled under `#![forbid(unsafe_code)]`. Measured in-guest microbenchmarks
include 2.2 M domain spawns/s, 322 M zero-copy messages/s at 3 ns latency, and a validated
memory-latency hierarchy (L1 ≈ 0 ns → DRAM ≈ 92 ns) confirming the numbers reflect real
hardware under hypervisor acceleration. What remains is hardware-dependent *realism* — real
CHERI tag bits, seL4-grade machine-checked proofs, production-parameter constant-time PQC, and
NPU codec execution — each already backed by a tested software implementation behind a hardware
abstraction layer, so the silicon is the accelerator, never the requirement. The paper also
consolidates the forward-looking distributed program: a **distributed SASOS** that extends the single
address space across a fleet (remote references fault in by content hash over NDN), a **decentralized
compute marketplace** with sovereign opt-in and a private/public dual pool, a **compute-backed
settlement layer** whose Proof-of-Inference verification is a direct consequence of DominionOS's bit-exact
determinism, and a privacy/hardening tier (hardened allocator and CHERI-D temporal safety, amnesic
mode, deniable storage, and anti-fingerprinting) — all expressed in the same three primitives and marked
**[Vision]** to keep the boundary with the working prototype explicit.

---

## 1. Introduction

### 1.1 The problem

The prevailing OS architectures were designed for a machine that no longer exists. Monolithic
kernels execute tens of millions of privileged lines, where a single buffer overflow in a
device driver compromises the entire machine. Permission models built on user/group/root and
access-control lists are coarse and structurally vulnerable to confused-deputy attacks. The
filesystem imposes serialization and naming bottlenecks on NVMe and persistent memory. TCP/IP
secures the *channel* between two host addresses, a poor fit for a world of mobility, caching,
and content distribution. Modern silicon, meanwhile, is a heterogeneous cluster of P-cores,
E-cores, GPUs, and NPUs that symmetric-multiprocessing models scale across poorly due to
cache-coherence traffic and lock contention.

### 1.2 The thesis

DominionOS synthesizes roughly fifty years of research in formal verification, capability-based
security, intralingual language design, distributed systems, and machine learning into a single
coherent, deterministic, efficient execution environment. It **discards** the monolithic kernel,
hardware privilege rings, and file-centric abstractions; it **replaces** them with a verified
microkernel, capability-enforced hardware routing, a SASOS resting on compiler-guaranteed memory
safety, and semantic, object-centric data representations. The work is framed against
ISO/IEC/IEEE 29148 (Systems and Software Engineering — Software Requirements Specification).

### 1.3 The P.I.E. Principle

One rule governs every architectural decision:

> **Performance, Isolation, Efficiency.** Hardware is responsible for *performance* and
> *efficiency*. Software — specifically safe-language compilers and capability enforcement —
> assumes the burden of *isolation* and *security*.

This division of labour explains, for example, why a single address space (a performance and
efficiency win — no TLB flush on context switch) is safe (isolation is provided by capabilities
and compiler-checked ownership, not by page-table boundaries).

### 1.4 Contributions

This paper consolidates and reconciles the following into one reference:

1. A **layered architecture** (Stages 0–14) expressed entirely in three native primitives,
   with each stage's mechanisms, cited prior art, and implementation status.
2. **Dominion**, an intralingual, capability-gated systems language that compiles to a
   Deterministic Compute Graph (DCG) plus capability proofs rather than to opaque binaries,
   with a type system spanning semantic primitives (`Identity`, `Time`, `Capability<T>`,
   `Latent<T>`) and extended scientific types (`Tensor`, `HyperVector`, `SpikeTrain`, `CRDT`,
   `HomomorphicCiphertext`, `QubitState`, `Manifold`).
3. A **security architecture** that eliminates whole exploit classes *by construction*
   (memory corruption, impersonation, ambient-authority escalation) and treats cryptographic
   agility, zero-plaintext storage, and post-quantum migration as first-class.
4. A **grounded evaluation** of the working prototype with measured test counts and benchmarks,
   maintaining an explicit and auditable boundary between vision and reality.

### 1.5 Reading this paper

Sections 2–8 present the architecture by concern. Section 9 reports the implementation and
evaluation. Sections 10–13 cover design tensions, adversarial findings, limitations, and the
roadmap. Appendix A is a glossary; Appendix B is an index of the Rust modules cited throughout.

---

## 2. Design Philosophy: The Three Native Primitives

Everything in DominionOS reduces to exactly three primitives. Every feature, and every legacy
integration problem, is expressed in terms of them.

1. **Capabilities** — unforgeable tokens of authority. A capability names a resource *and* the
   rights over it; if you do not hold a capability for a resource, that resource is functionally
   invisible. Capabilities are **monotonic** (a derived capability carries equal or lesser
   authority than its parent, $C_{derived} \subseteq C_{source}$), **provenanced** (derivable
   only from valid prior capabilities), **integrity-protected**, and **bounds-checked**. The
   long-term enforcement target is CHERI's 128-bit hardware capabilities; today they are enforced
   by a software algebra that still traps every violation.

2. **The content-addressed object graph** — an immutable, deduplicated, semantic data store that
   replaces the filesystem. Every object is named by the hash of its content (SHA-256 in the
   prototype), so storage is deduplicated, integrity is intrinsic, versioning is perfect, and
   rollback is a head-pointer flip. The graph behaves like a system-wide Git repository for *all*
   state, not just source code.

3. **The deterministic state machine** — the entire machine state is a single hashable object,
   and every action is an input to a state-transition function. Given identical inputs (including
   recorded entropy and recorded I/O timings), execution is bit-for-bit reproducible. This makes
   Deterministic Simulation Testing, instruction-level time-travel debugging, and
   `fork`-as-snapshot natural rather than bolted-on.

These three recur in every later section: drivers are capabilities bounding MMIO/DMA/IRQ over
content-addressed, signed specs validated by deterministic replay; pub/sub is a capability over a
named object prefix whose events are content-addressed objects logged into the state machine;
identity, recovery, energy budgets, and resource quotas are all capabilities; undo, debugging,
and crash recovery are all the deterministic state machine plus the immutable graph.

---

## 3. The Architectural Stack (Stages 0–10)

The system is described as a stack of conceptual stages from the lowest hardware layer upward.
**The numbering is a conceptual layering of the vision, not a build order** — the actual build
sequence is the milestone plan in §9 and §13.

### 3.1 Stage 0 — Formally Verified Firmware & Secure Boot **[Vision]**

Legacy BIOS/UEFI comprises millions of loosely audited C lines, an enormous attack surface
*before the OS even initializes*. Stage 0 replaces firmware with a microscopic, formally verified
layer **constrained to under 100,000 lines**, restricted to CPU initialization, memory training,
device enumeration, and establishing the secure-boot chain, written in minimal assembly and
verification-friendly languages (Rust, Coq/Lean lineage, informed by coreboot). Multi-vendor
attestation (Intel TDX, AMD SEV-SNP, AWS Nitro) yields independently reproducible enclave images,
and a *measured boot* extends each stage's hash into TPM PCRs, providing a cryptographic proof
that the running code matches auditable source. A working secure-boot module (`secureboot.rs`,
measured + signature-chained, with a reproducible-image proof that running code equals auditable
source) exists as a **[Modeled]** floor; production silicon attestation is the deferred realism.

### 3.2 Stage 1 — High-Assurance Microkernel **[Vision / Modeled]**

Modeled on the **seL4** lineage, the kernel is a policy-free core of roughly **12,000 lines** of
C and verified assembly, responsible only for scheduling, virtual memory, IPC, and capability
management. Everything traditionally in-kernel — filesystems, device drivers, networking,
USB — is evicted to sandboxed user-space service cells. Trusted-Computing-Base minimization
admits a machine-checked proof of functional correctness against specification, complete and
sound Worst-Case Execution Time (WCET) analysis, and a mixed-criticality scheduler that runs
hard-real-time tasks alongside untrusted general-purpose loads without deadline violation. The
prototype provides an invariant suite, inductive-preservation and refinement checks, and WCET
path budgets in `verify.rs` **[Implemented]**; the seL4-grade machine-checked proof is the
deferred realism.

### 3.3 Stage 2 — Hardware-Enforced Capability Security (CHERI) **[Modeled]**

Stage 2 abolishes ACLs entirely in favour of pure capability security deployed at the
instruction-set level. **CHERI** (Capability Hardware Enhanced RISC Instructions) extends 64-bit
architectures (ARMv8-A Morello, RISC-V) with **128-bit architectural capabilities** carrying four
mathematically enforced attributes:

- **Provenance** — capabilities derive only from valid manipulation of prior capabilities;
  integers cannot be forged into pointers.
- **Monotonicity** — a derived capability's rights are a subset of its parent's, enforcing least
  privilege dynamically at the processor.
- **Integrity** — an out-of-band tag bit clears on illegal modification; dereferencing a corrupted
  capability traps.
- **Bounds** — explicit `[base, base+len)` limits eradicate spatial memory errors.

The result is fine-grained, hardware-enforced compartmentalization of individual drivers,
libraries, and even functions. Because CHERI silicon is barely commercial, the architecture
defines a **Capability Enforcement Layer (CEL)** with tiered backends, selected at boot from
attested hardware features (`enforcement.rs`, `cheri.rs`):

| Tier | Hardware | Mechanism | Guarantee |
|---|---|---|---|
| **Tier 0** | x86-64 today | Rust/Dominion affine types + software bounds + MPK/PKRU domains + WASM-style sandbox + guard pages | Cryptographically unforgeable but **not** hardware-tagged; fully functional, fewer guarantees |
| **Tier 1** | aarch64 MTE + PAC / Intel MPK | Hardware memory tagging + pointer authentication | Hardware-assisted |
| **Tier 2** | CHERI (Morello / RISC-V) | Full 128-bit hardware capabilities | The design target |

Below Tier 2 the Stage 2 claim weakens from "impossible" to "contained / improbable."
Attestation reports the tier, and high-assurance domains (e.g., Financial) can refuse to accept
capabilities below a minimum tier. The capability algebra — provenance, monotonicity, integrity
tag, bounds, all trapping — is implemented and tested in software (`capability.rs`).

### 3.4 Stage 3 — Intralingual Design, Memory Safety & SASOS **[Modeled / Implemented]**

With hardware capabilities securing *spatial* boundaries, language-level type safety secures
*behavioral* semantics. **Intralingual design** matches the OS execution environment to the
runtime model of its implementation language, using Rust's affine types and ownership to shift
resource bookkeeping from runtime to compile time. This addresses **state spill** — where one
component holds state on behalf of another, causing fate-sharing and cascading failure. The OS is
structured as a collection of tiny **cells** that interact by zero-cost state transfer, never
spilling state into one another; an individual cell can be unloaded, replaced, or restarted at
runtime without rebooting, enabling live evolution and fault recovery.

The **Single Address Space Operating System (SASOS)** combines CHERI capabilities with
compiler-guaranteed memory safety to make hardware address-space isolation unnecessary. Every
process, kernel module, and application lives in one global virtual address space; protection is
*orthogonal to translation* and is enforced by capabilities, not page tables. Consequently,
context switching is nearly instantaneous (no TLB flush or cache invalidation), pointers are
globally valid, and complex data structures are shared across processes without
serialization. Code runs inside **Software-Isolated Processes (SIPs)** — closed object spaces that
cannot share writable memory except through statically verified channels — and there is a
**Single Privilege Level**: no kernel/user mode distinction. A subtle but central insight
(tension T6, §10): in a SASOS, holding an *address* grants nothing without the *capability*;
reachability is a function of authority topology, not address visibility.

Because SASOS removes the per-process address space as a natural reclamation boundary, the **cell
becomes the unit of memory reclaim** (`pressure.rs`): RAM is treated as a cache over the
persistent graph — clean, content-addressed objects are re-fetchable by hash and therefore free to
evict and fault back in; dirty objects are written through the commit path before eviction;
sensitive spill goes to encrypted swap. Memory is a capability-rate-limited resource with bounded
per-domain working sets, and the system defers or refuses work rather than thrashing.

### 3.5 Stage 4 — Multikernel & Heterogeneous Scheduling **[Modeled; SMP substrate Implemented]**

Following the **Barrelfish** model, the local machine is treated as a distributed network of
independent cores. A minimal non-preemptible **CPU driver** runs on each core; all inter-core
coordination is explicit message passing; each core holds a *local replica* of OS state kept
consistent by distributed-systems consensus, which yields far better cache-hit ratios than
shared-memory SMP. The scheduler becomes a **global resource orchestrator** that maps a global
execution graph onto heterogeneous compute: matrix math to the GPU, neural inference to the NPU,
control logic to the CPU — with *no* disparate APIs (CUDA/OpenCL/Vulkan) exposed to applications.

Consistency is **tiered** and *declared per object*: convergent state (caches, presence,
collaborative documents) uses `CRDT<T>` (commutative, associative, lock-free); value-bearing or
invariant-critical state (money, inventory, unique allocation) uses linearizable transactions via
consensus (Raft/Paxos within a machine, BFT across an untrusted fleet). Causal ordering uses
**Hybrid Logical Clocks**. Energy and thermal are first-class scheduling objectives alongside
latency and throughput (§6.3).

The prototype already brings the application processors online for real (`smp.rs`: ACPI MADT →
LAPIC → real-mode-to-long-mode AP trampoline → INIT-SIPI-SIPI), reaching **8 cores with
near-linear 8.3× scaling**; APs currently share the BSP page tables (consistent with SASOS) and
pull from a lock-free job queue. A model layer (`multikernel.rs`, `hlc.rs`) implements per-core
replicas, HLC-ordered message passing, the global execution graph with CPU/GPU/NPU routing, cycle
detection, and both the CRDT-convergent and consensus-linearizable paths; wiring the real APs to
the per-core CPU-driver model is the remaining step.

### 3.6 Stage 5 — Intelligent Optimization & Generative Storage **[Modeled]**

Stage 5 eradicates the filesystem in favour of the content-addressed graph (§2) and adds
**generative compression** via end-to-end learned neural codecs (LLMs and VAEs that predict
semantic structure), pushing past the Shannon entropy limit on semantically rich data. Because
floating-point operations on GPUs are non-associative — a source of deterministic drift and
decompression corruption — the design mandates logit quantization and a **"Grid Snap"** protocol
to guarantee a deterministic, reproducible neural code. A lightweight **"Scout" (Two-Pass,
Band-Pass)** router classifies each block by a quick classical compression ratio $R$: $R<1.05$
(encrypted noise) and $R>3.0$ (highly repetitive) go to CPU classical codecs, while $1.05 \le R
\le 3.0$ (semantically complex) is routed to the NPU/GPU neural codec — so inference is spent only
where it pays. Storage and caching self-optimize with kernel-level reinforcement learning: a
Q-learning tier policy plus a CNN-LSTM block-access predictor for prefetch. The prototype
implements predictive + RLE lossless codecs, Grid Snap, the Scout router and thresholds, the
CNN-LSTM stride predictor, and the Q-learning tier policy (`neural.rs`); the production neural
codecs await NPU execution.

### 3.7 Stage 6 — Active Defense, Provenance & Driver Synthesis **[Modeled; driver PoC Implemented]**

When hardware is abstracted into uniform capability descriptors, a *driver* becomes a sandboxed
execution module subject to mathematical constraints rather than a privileged blob (§6.4). For
storage integrity in the deepfake era, a **Secure Learned Image Codec (SLIC)** turns passive
storage into an active security perimeter: watermarks are embedded as adversarial perturbations in
the compressed latent space, so that unauthorized manipulation and re-encoding maps secret
messages to visible artifacts and the modified data self-degrades — **cryptographic poisoning**
that preserves authenticity. The prototype implements SLIC keyed watermarking and cryptographic
poisoning (`defense.rs`).

### 3.8 Stage 7 — Identity-Based Networking (NDN) **[Modeled]**

TCP/IP secures a channel between two host addresses; Stage 7 discards it for
**Information-Centric / Named Data Networking (NDN)**, routing on the cryptographic identity of
*data* rather than host location. Every Data packet is signed by its producer and bound to a
hierarchical namespace (e.g., `/user/jayden/workstation/syslog/v1`); a consumer broadcasts an
*Interest* and the network returns the nearest cached, signed *Data* packet.

| Characteristic | Legacy TCP/IP | NDN |
|---|---|---|
| Routing target | Physical IP address | Hierarchical data namespace |
| Security perimeter | The channel (TLS/IPsec) | The packet itself (per-packet signatures) |
| Trust model | Centralized CAs | Web-of-trust / Hierarchical Identity-Based Cryptography |
| Caching | App-level CDNs | Native to the protocol |

**Hierarchical Identity-Based Cryptography** lets a data name act as its own public key,
eliminating pre-agreed trust anchors. The native overlay, **DominionLink**, runs over UDP (à la
QUIC/WireGuard) with self-certifying identities, Kademlia DHT resolution, and a DNS bridge for
legacy interop; transport concerns — CUBIC + BBR congestion control, ICE NAT traversal,
identity-based connection migration, and offline-first reconcile-by-hash — are implemented in
`transport.rs`. Packet signatures are post-quantum (Stage 13). The prototype implements real NDN
forwarding with Names, Interest/Data, a Content Store, PIT aggregation, and FIB longest-prefix
matching (`ndn.rs`), plus self-certifying IDs and the DHT/DNS bridge (`dominionlink.rs`).

### 3.9 Stage 8 — Semantic Audio & Object-Based Rendering **[Modeled]**

Rather than treating audio as a flat 48 kHz PCM stream, Stage 8 represents sound as *meaning*.
Neural semantic tokenizers extract discrete high-level sound events; over constrained links,
**Joint Source-Channel Coding and Modulation (JSCM)** transmits semantic tokens directly,
reconstructing acceptable perceptual quality at roughly **10 dB lower SNR** than traditional
compression pipelines. Spatial rendering is **Object-Based Audio** (Dolby Atmos / L-ISA class) —
sounds are objects localized in 3D with distance, orientation, and frequency-dependent
directivity — with **Head-Related Transfer Functions applied at the kernel level**, an
**Earliest-Deadline-First** scheduler enforcing a strict **16.6 ms** isochronous frame budget, and
zero-copy DMA between the audio engine and GPU. The prototype implements object-based audio with
3D position and directivity, a JSCM nearest-codebook tokenizer, transcendental-free per-ear
HRTF gain + interaural time difference, and the EDF 16.6 ms scheduler (`audio.rs`).

### 3.10 Stage 9 — Object-Centric & AI-Native Interaction **[Modeled; UI Implemented]**

Stage 9 abandons the document-application paradigm. Drawing on Lisp machines and Smalltalk-80's
MVC, the entire interface is a set of views over a system-wide knowledge graph. Programs are not
installed `.exe` files but **transient compositions of services and capabilities**; a data object
("Project," "Conversation") is constant while the OS invokes rendering capabilities to present it
as a Table, Graph, Spatial, or Assistive view on demand. LLMs are embedded as **exploratory
programming agents** that translate natural-language intent ("analyze these sales numbers and make
a presentation") into low-level capability invocations, assembling engines on the fly. Crucially,
AI runs inside dedicated **AI Domains** with *no* infrastructure, storage-write, network, or
capability-creation authority; every action it takes must traverse the **Capability Airlock**
(§5.1). The concrete realization — compositor, shell, views-on-demand, AI command bar, abstract
input model, universal undo — is implemented in the UI stack (§7).

### 3.11 Stage 10 — Deterministic State Machines & Reproducibility **[Implemented]**

The most radical departure is the elimination of unpredictable system evolution. The whole machine
state is a hashable object and every action is an input to a state-transition function.
**Deterministic Simulation Testing (DST)** runs the OS inside a hypervisor that controls *every*
source of non-determinism — system time, thread interleaving, network latency, and random number
generation — so any race or distributed bug is perfectly reproducible from its seed, and a panic
can be rewound instruction by instruction to root cause.

Randomness is reconciled with determinism by separating two generators (§5.5): a **seeded
deterministic CSPRNG** feeds execution (identical seed + identical inputs ⇒ identical run), while
**true hardware entropy** is treated as an *external input event* recorded in the provenance ledger
at the moment of consumption — so replay feeds the recorded value and even "true randomness"
becomes deterministically reproducible.

Single-machine rollback is a head-pointer flip over the immutable graph (`object.rs::rollback`).
Rollback *across a gossiping fleet* is the hard case, and the source's red-team analysis (finding A,
§11) found a "split-timeline merge corruption" where a lagging peer's last-writer-wins merge could
resurrect a value that another node had rolled back to contain a misbehaving app. The fix,
**causal fencing** (`consistency.rs::FencedReplica`), attaches an HLC fence to every key (a write
is accepted only if its stamp dominates the fence), raises the fence past a bad write on rollback,
merges fences by `max` so an abandoned write can never re-enter under loss/reorder/partition, and
`pin`s identity roots and capability tables as immutable to rollback. Its headline test gossips a
poisoned write, rolls it back concurrently, partitions and heals the network, and asserts across a
**300-seed sweep** that all replicas converge with the poison present on none.

---

## 4. The Dominion Language

### 4.1 Philosophy

**Dominion** is the sole *native* execution language. Its defining break from legacy systems
languages is that it does **not** compile to a binary executable; it compiles to a **Deterministic
Compute Graph (DCG)** plus cryptographic **capability proofs**, enforced at runtime by the
capability machinery. The same syntax writes a microkernel scheduler and a user application — only
the injected capability token differs — unifying the security model, memory management, and
concurrency into one verifiable framework.

Core pillars:

- **Hardware-Enforced Objects** — every object gets a 128-bit CHERI capability on creation; there
  are no pointers and no pointer arithmetic (rejected by the compiler), only references.
- **Affine type system** — strict linear ownership; when a value leaves scope its capability is
  cryptographically invalidated and memory is reclaimed instantly, with **no garbage-collection
  pauses**.
- **Implicit parallelism** — there are no threads or locks; functions form a dependency graph and
  the compiler schedules independent nodes across cores.
- **Semantic primitives** — `Identity`, `Time`, `Resource<T>`, `Latent<T>`, and `Capability<T>`
  are first-class types, enforceable at compile time and visible in signatures.

### 4.2 Syntax and semantics

The grammar (EBNF, implemented in `lang/lexer.rs` and `lang/parser.rs`) provides `object`
(semantic struct), `cell` (an actor with optional capability injection `[cap: Capability<…>]`), and
`fn` (a compute node, optionally tagged `@NPU`/`@GPU`/`@CPU`). Two binding forms exist: `let`
(read-many) and `linear` (affine, consumed on use). The two signature operators are `=>` (parallel
map across cores) and `|>` (dataflow pipeline). There is no null and there are no exceptions — error
values are deterministic, replay-safe `Result`-like types.

```dominion
cell StorageManager [cap: Capability<StorageWrite>] {
    @NPU
    fn compress_to_latent(doc: Invoice) -> Latent<Invoice> {
        let embedding = NeuralCodec::encode(doc);
        return embedding;
    }
}

fn process_billing_cycle(invoices: Vector<Invoice>, cap: Capability<StorageWrite>) {
    // '=>' triggers a parallel map; the compiler parallelizes by data dependency
    let latent_records = invoices => StorageManager::compress_to_latent;
    SystemGraph::commit(latent_records, cap);
}
```

Capabilities are expressed in the type system and discharged at compile time: a cell can perform
only the operations its capability grants, and the compiler verifies monotonicity
($C_{derived} \subseteq C_{source}$). Apps are *default-closed*, holding only explicitly granted
capabilities (`Surface`, `Storage`, `Net` scoped to origins, `Clipboard`, `Device`, `Time`,
`Entropy`).

### 4.3 The type system

**Semantic primitives:** `Identity` (cryptographic signature of a user/process/system), `Time`
(deterministically verified timestamp), `Resource<T>` (a quantifiable hardware constraint such as
`Resource<NPU_Cycles>` or `Resource<Joules>`), `Latent<T>` (a natively compressed neural
representation), and `Capability<T>`.

**Extended scientific types** (`datatypes.rs`), each routing to appropriate hardware:

| Type | Purpose |
|---|---|
| `Tensor<T, Shape>` | Multidimensional array with compile-time shape checking; routes to GPU matmul — no runtime dimension errors |
| `HyperVector<D>` | Hyperdimensional computing (e.g., $D=10{,}000$, $H\in\{-1,1\}^D$) with bundling, binding, permutation |
| `SpikeTrain<N, TimeDomain>` | Sparse asynchronous spiking events for neuromorphic/SNN accelerators |
| `CRDT<T>` | Commutative, associative replicated type for lock-free multikernel convergence (not for value-bearing state) |
| `HomomorphicCiphertext<T>` | Compute on encrypted data without exposing plaintext (data-in-use protection) |
| `QubitState<N>` | Tensor-network quantum state with entanglement tracking; delegates superposition to a QPU |
| `Manifold<T, MetricTensor>` | Non-Euclidean geometry (robotics, fluid dynamics, astrophysics) encoding differential structure natively |

### 4.4 Reactive UI and applications

A UI is a pure function from object state to a `Widget` tree, re-evaluated when its tracked reads
change (the React/SwiftUI model), implemented in `appkit.rs`:

```dominion
view Counter(state: Counter) -> Widget {
    column {
        label("Count: ${state.value}")
        button("Increment") on tap => state.value = state.value + 1
    }
}
```

The runtime tracks which fields a view reads and re-renders only the affected subtree; DSL
constructs (`column`/`row`/`label`/`button`/`input`/`tabs`/`list`) desugar to `toolkit.rs::Widget`
trees. Reactive re-render is a pure function of recorded inputs, and async completion order is
logged, so entire app sessions replay and rewind (the Stage 10 universal undo). `async`/`await`
run cooperatively over the SIP scheduler.

### 4.5 Polyglot strategy

The single rule is that *only Dominion has native authority*. Other languages run capability-bounded
in one of two ways: (a) compiled to **WASM**, where WASI capabilities map onto the Dominion
capability model and the module receives only explicitly granted authority (`wasm.rs`); or (b) run
as real runtimes (CPython, Node.js, JVM, Go) inside the same personality sandbox that contains
Linux apps, with full ecosystem compatibility but no ambient syscalls. Malware in any language is
contained; OS internals remain provably Dominion-only.

**[Implemented]** This is no longer only a strategy: `polyglot.rs` hosts **seven languages —
Python, Rust, C++, C#, JavaScript, TypeScript and Java — running real programs**. Each language's
*native* surface syntax is parsed (a brace front-end for the C-family and Rust, an
indentation front-end for Python) and lowered to one shared AST run by one interpreter over one
standard-library/package registry. Guests are not toy one-liners: they define **multiple
functions** (loops, recursion, lists, conditionals) and **import and use library packages**
(`stats`, `mathx`, `strx`) in each language's idiomatic form (`import`/`from`, `use`,
`#include`, `using`, `require`). Packages are **default-closed** — calling a package function
without importing its package is refused (`RunError::NotImported`) — and every guest is
**step-metered** so a runaway program is stopped, not merely sandboxed. A shared demo (a
multi-function statistics pipeline) returns identical results across all seven, and a shared
benchmark agrees on a single checksum across all seven — a cross-language equivalence proof.
Verified by 11 host unit tests, an on-metal selftest, and a per-language benchmark
(`kernel/bench.rs`). Real upstream interpreters (bit-exact CPython/Node/JVM) remain the
ecosystem-compatibility path on top.

### 4.6 Compiler and tooling status

**[Implemented]:** lexer, parser, tree-walking interpreter with capability-gated cells,
`Identity`/`Latent` primitives, the `=>` and `|>` operators, affine `linear` bindings with
scope-end cryptographic invalidation, type-directed `@NPU`/`@GPU`/`@CPU` routing, cell hot-swap,
zero-copy cell RPC over the scheduler, DCG lowering with compile-time capability checking and
proof-carrying graphs (`dcg.rs`), and the reactive runtime (`appkit.rs`). **[Vision/next]:** an
AOT/JIT compiler to capability-checked native code, full `Tensor` shape inference and generics over
types and capabilities, a content-addressed signed package manager, the standard library, an LSP /
formatter / linter, and an integrated DST-backed test framework with a time-travel debugger.

---

## 5. Security Architecture

DominionOS's security thesis is that *security is a property of authority topology and cryptographic
proof, not of physical control or code obscurity*. Whole exploit classes are eliminated **by
construction**; what remains is deliberate engineering against side-channels, constant-time crypto,
supply chain, logic bugs, social engineering, and physical access.

### 5.1 Stage 11 — Kernel Hardening: Firewall (11.14) and Airlock (11.15) **[Implemented]**

The kernel assumes compromise: no local component is trusted, and security comes from capability
confinement, hardware memory safety, cryptographic provenance, and continuous attestation, layered
**five deep** (formal verification → CHERI enforcement → language ownership → cell isolation →
cryptographic attestation), so an attack must defeat all five.

The **Capability Firewall** (`firewall.rs`) controls authority *within* a domain by modeling
`Identity → Cell → Capability → Object` as an authority graph in which security is defined by
*reachability*. Each capability has a globally unique ID for revocation indexing; capabilities are
partitioned into isolated **security domains** (Personal, Financial, Medical, Infrastructure,
Development, AI) with cross-domain flow denied by default. It adds authority-diffusion detection
(measuring delegation depth, fan-out, and privilege concentration to catch gradual privilege
creep), capability rate limiting (e.g., storage-write ops/s, network MB/s, GPU %), quarantine
zones for suspicious cells, an immutable signed provenance ledger of every capability lifecycle
event, and local AI anomaly analysis. Non-forgeability, monotonicity, revocation correctness,
domain isolation, and reachability are targets for formal proof.

The **Capability Airlock** (`airlock.rs`) is the *sole* path by which a capability may cross a
trust boundary, with no bypass. Its fixed protocol is `Request → Validation → Policy Evaluation →
Capability Reduction → Audit Recording → Transfer Approval → Capability Issuance`, sanitizing each
capability down to the minimum authority required (e.g., a Financial-domain
`{Read, Write, Delete Account}` becomes `Read Account Summary` in the AI domain). It supports
one-way data-diode channels, **temporal capabilities** that expire (e.g., valid for 10 seconds, a
single transaction, or one graph execution, revoking all derivatives on expiry), and multi-party
authorization for separation of duty. The prototype implements sanitize-to-minimum, one-way
transfer, TTL expiry, multi-party approval, and provably no re-escalation.

### 5.2 Stage 13 — Post-Quantum Cryptography & Agility **[Modeled]**

Assuming a future cryptographically relevant quantum computer and "harvest-now-decrypt-later"
adversaries, no security may depend on factoring, discrete logs, or elliptic curves. A
**Cryptographic Abstraction Layer (CAL)** makes every primitive pluggable so identity, capability,
networking, storage, and attestation survive algorithm migration unchanged. **Hybrid cryptography
is mandatory** — every critical operation requires *both* a classical and a post-quantum signature,
so an attacker must break both families at once. Key agreement uses a lattice KEM (the design
target is NIST FIPS 203 **ML-KEM**); capability tokens carry quantum-resistant signatures
(XMSS-style many-time hash-based signatures: per-token Lamport keys under a Merkle root,
implemented in `tokensig.rs`). Incremental key rotation re-signs identities and re-issues
capabilities without service interruption; a quantum-aware provenance ledger records algorithm
versions and key lineage for historical verification. The prototype implements a real LWE KEM with
reduced parameters (`lattice.rs`), hash-based Lamport + hybrid signatures behind the agility layer
(`crypto.rs`), and PQ-signed NDN Data and HIBC names; raising to standard ML-KEM/ML-DSA parameters
with constant-time implementations and known-answer/negative testing is the residual work.

### 5.3 Stage 14 — Universal Encryption & Zero-Plaintext **[Modeled]**

The central reframing: **plaintext is an execution state, not a storage state.** All persistent
objects are encrypted at creation; ciphertext is the canonical representation and plaintext is a
transient computational projection that exists only when an authorized computation requires it. Each
object has an independent key, access policy, and audit history, so compromising one object exposes
no other. **Keys are capabilities** — a storage capability is not a read capability, so a storage
provider can physically hold encrypted bytes with no decryption authority. *All* network traffic is
encrypted with no exceptions, bound to cryptographic identities rather than host addresses
(`session.rs`: lattice KEM + AES-256-GCM with the two identities and an epoch bound as
authenticated associated data, defeating replay). Secure deletion becomes **cryptographic garbage
collection**: destroy the key and the remaining ciphertext is computationally useless even in old
immutable blocks or untrusted backups. The prototype implements a zero-plaintext vault with
keys-as-capabilities, crypto-GC, and searchable encryption (`vault.rs`) over two independent AEAD
families — **ChaCha20-Poly1305** by default (RFC 8439, `chacha.rs`) and **AES-256-GCM** with
memory-at-rest sealing, both validated against published known-answer tests (`memcrypt.rs`).

### 5.4 Threat model: eliminated-by-construction vs residual **[Mixed]**

**Eliminated by construction** (no exploit primitive exists):

| Class | Why it is gone | Modules |
|---|---|---|
| Memory corruption (RCE/ROP/use-after-free) | `#![forbid(unsafe_code)]` core + CHERI capability tags with unforgeable bounds | `capability.rs`, `cheri.rs` |
| MITM / impersonation / substitution | Self-certifying identities (hash of public key); content-addressing for end-to-end verification | `dominionlink.rs`, `session.rs`, `object.rs` |
| Privilege escalation / confused deputy | Monotonic derivation + recursive revocation + one-way Airlock sanitization | `firewall.rs`, `airlock.rs`, `confidential.rs` |
| Forgery / harvest-now-decrypt-later | Hybrid (classical + PQ) signatures; lattice KEM; XMSS-style tokens | `tokensig.rs`, `crypto.rs`, `lattice.rs` |
| Local resource-exhaustion DoS | Per-domain rate-limited capability quotas; energy budgets | `firewall.rs`, `power.rs` |

**Residual classes** requiring deliberate engineering: microarchitectural side-channels
(Spectre/Meltdown/cache/timing/power/EM — need CPU features and constant-time code);
production-hardened constant-time crypto at standard parameters; supply-chain hardening
(reproducible builds, minimal audited dependencies, signed per-artifact provenance,
measured-boot attestation); logic bugs (widen property/invariant suites, differential testing,
machine-checked proofs for critical cores); social engineering (least-authority capabilities and ZK
shrink the blast radius and what can be phished); and physical access (cold-boot/DMA/evil-maid —
mitigated by memory-at-rest encryption, IOMMU-gated DMA, and optional hardware-RoT binding that is
never required). The codebase already carries property-based invariants (`props.rs`),
structure-aware DST fuzzing (`fuzz.rs`), and chaos fault-injection.

### 5.5 Randomness & entropy **[Implemented]**

Two generators with a strict contract about which feeds execution. The **hardware TRNG**
(`entropy.rs`) combines on-die instructions (RDRAND/RDSEED, RNDR, RISC-V Zkr) with ring-oscillator
jitter and supplementary interrupt-timing noise, runs continuous **NIST SP 800-90B** health tests
(repetition-count and adaptive-proportion), conditions via SHA-256, and **fails closed** — blocking
consumers on health-test failure or exhaustion rather than emitting weak randomness. The **seeded
DRNG** (`random.rs`) is a hash-DRBG giving each cell or domain an independent, capability-gated
stream derived from the run seed; reseeds happen at logged boundaries so they remain replayable.
The determinism contract: the state machine never reads hardware entropy mid-computation; every
true-entropy consumption is recorded as an input event so DST, instruction-level rewind, and
`fork`-as-snapshot reproduce randomness exactly.

### 5.6 Identity, recovery & key management **[Implemented]**

A cryptographically strict system (destroy keys ⇒ destroy data) must make recovery first-class
*without* an escrow backdoor. An HD-wallet-style hierarchy (`identity.rs`, `recovery.rs`) derives
device and domain identities, key-encryption keys, and per-object data-encryption keys from a
single offline-recoverable master seed. Recovery offers several mechanisms requiring at least one:
threshold/social recovery (Shamir over GF(2⁸), an M-of-N guardian quorum, reusing the Airlock's
multi-party authorization), hardware-anchored shares (each fleet device stores a share in its
secure element/TPM), an optional offline recovery key, and time-delayed self-recovery with a veto
window (a request opens a challenge period — e.g., 72 hours on the monotonic clock — during which
existing devices can veto a fraudulent attempt). Recovery itself is a capability-graph operation
through a dedicated Recovery Airlock, fully provenance-logged, rate-limited, and anomaly-watched;
on success the new device identity is installed, capabilities re-issued, objects rewrapped, and
old keys recursively revoked. The no-escrow-backdoor property is explicit and provable.

### 5.7 Secure & verifiable time **[Implemented]**

Both pillars (determinism and cryptography) depend on trustworthy time, and an attacker who
controls the clock can resurrect expired capabilities. Three distinct notions are kept separate
(`time.rs`): **logical time** (a monotonic, reproducible step counter / HLC for ordering and
determinism), **trusted wall-clock** (externally correct with bounded uncertainty, from an
authenticated multi-source service — Roughtime or NTS, never bare NTP, with PQ/hybrid signatures),
and **anti-rollback time** (a hardware-backed monotonic counter that can never decrease). A
TrueTime-style `Time` interval `[earliest, latest]` lets ordering-sensitive code commit-wait out
its uncertainty. Boot establishes a trusted lower bound on "now" so an attacker cannot boot into
the past to revive an expired capability. In DST every external time reading is a recorded input
event.

### 5.8 Hardware root of trust & attestation **[Vision/Modeled]**

A **Root-of-Trust Abstraction Layer (RTAL)** (`rot.rs`) abstracts over TPM 2.0 (primary), TPM 1.2
(legacy fallback), firmware TPM / Pluton / Intel PTT, secure elements / StrongBox / Secure Enclave,
HSMs, DICE (for constrained devices), CPU TEEs (TDX/SGX, SEV-SNP, Arm CCA/TrustZone, Nitro),
FIDO2/WebAuthn authenticators, and a last-resort software RoT — degrading gracefully and reporting
its tier. It binds measured boot into PCRs, seals the device identity key and DRNG seed to platform
state, stores per-device recovery shares non-exportably, provides anti-rollback monotonic counters,
and produces signed remote-attestation evidence (measurements + enforcement tier + freshness nonce)
that peers or the Airlock can require before accepting capabilities. Privacy-preserving **Direct
Anonymous Attestation** lets a device prove "I am genuine, in good state, tier ≥ N" without
revealing *which* device.

### 5.9 Zero-knowledge proofs as a system primitive **[Implemented core]**

ZK turns "trust me" into "verify the math," privately. Exposed behind the CAL
(`zkservice.rs`, `zk.rs`, `anon.rs`), the system selects a proof system per use — Groth16/PLONK
(succinct), STARKs (transparent, hash-based and therefore naturally PQ-safe), Bulletproofs (ranges),
BBS+/AnonCreds (anonymous credentials) — preferring transparent/PQ-safe systems for long-lived
proofs. Proofs are content-addressed graph objects (cacheable, deduplicated, verifiable by anyone),
verification is deterministic, and blinding randomness is drawn from the seeded DRNG and recorded so
DST replay holds. The two highest-leverage uses are **ZK capability proofs** ("I hold a capability
satisfying this policy" without revealing the delegation chain — strengthening least privilege and
confidentiality simultaneously) and **ZK + confidential compute for AI Domains** (an AI cell proving
it ran an approved model on approved inputs without exposing model or data). The prototype
implements Schnorr NIZK (Fiat-Shamir), Chaum-Pedersen DLEQ, and Merkle membership; the red-team
finding C (§11) replaced a naive static-public-key login (a stable cross-context correlator) with
per-context pseudonyms $P_{ctx}=g_{ctx}^x$ and a constructor-enforced
`TraceabilityClass {Attributable, Pseudonymous, Anonymous}` that refuses to attach a global
correlator to an "anonymous" transaction.

### 5.10 System domain & internal confidentiality **[Implemented]**

Two goals are carefully separated. **(A) Isolation** — no user or cell can read kernel, runtime, or
other-cell memory — is *required* and already guaranteed by the capability model, made explicit and
red-team-testable (`confidential.rs` uses a Bell-LaPadula lattice; `SystemPrivate` data is sealed to
the System domain). **(B) Code secrecy** is a *different* goal, partly at odds with the thesis:
security comes from "trust the math" (formal verification, capabilities, attestation), **not** code
obscurity, and Stage 0's reproducible, independently auditable builds are a strength. The OS runs
under a distinct **System Identity**; users receive *use* capabilities, never *read-internals*
capabilities. Optional code confidentiality (encrypting proprietary third-party cells at rest,
decrypted only inside an attested enclave) is available but off by default for the core, and is never
relied upon as a security boundary.

### 5.11 Memory encryption at rest **[Modeled]**

"All data in memory encrypted, exposed only at access time" is achievable via a layered design
(`memenc.rs`, `memcrypt.rs`): **Tier A** Total Memory Encryption (Intel TME / AMD SME) encrypts all
DRAM transparently at single-digit-percent overhead and is on by default where supported; **Tier B**
per-domain keys (Intel MKTME / AMD SEV-SNP) make one domain's pages ciphertext to another and to a
malicious host; **Tier C** software per-object encryption (Stage 14) protects crown-jewels (keys,
identity, capabilities, financial, medical) so they are ciphertext even in cache and decrypt only
into registers for the capability holder. Against *software* adversaries the capability model already
isolates — no extra crypto on the common path — so memory encryption specifically targets *physical
and host* threats (cold-boot, bus/DMA probing, stolen DIMMs, untrusted hypervisors). Without hardware
support the property degrades gracefully to "sensitive objects encrypted" and the weaker posture is
attested.

### 5.12 Data lifecycle & compliance **[Implemented]**

The immutable, deduplicated graph appears to collide head-on with the right to erasure (GDPR Art.
17), retention limits, and legal hold. The reconciliation (`lifecycle.rs`): legal **deletion maps to
cryptographic shredding** (destroy the per-object key; the ciphertext is gone even in backups), while
**provenance is retained as a tombstone** recording the fact, time, and authority of deletion. Every
object carries a retention policy and deletion schedule as capability-enforced metadata; legal hold is
a capability that suspends crypto-GC; purpose/consent tags scope capabilities, and withdrawing consent
revokes them. Domains can be geo-pinned with cross-jurisdiction transfer enforced by the Airlock.
Subject access, portability, and audit are cheap: export runs through the codec registry, and the
provenance ledger *is* the audit trail.

### 5.13 Kernel self-protection & hardened memory **[Vision]**

DominionOS's safe core (`#![forbid(unsafe_code)]`) makes heap exploitation a non-primitive there; the
hardening surface is the **kernel HAL** (the only place `unsafe` lives) and the **legacy sandbox**.
The kernel allocator adopts the `hardened_malloc` design — out-of-line, read-only metadata; size-class
slab isolation at random offsets; guard pages and randomized canaries; and zero-on-alloc/free — and the
kernel enforces W^X, retpoline/Spectre mitigation, and minimal panic output. Hardware memory tagging
(ARM MTE / Intel MPK) is *not* a new mechanism here: it is **Tier 1 of the existing Capability
Enforcement Layer**, selected at boot from attested features. The genuinely new contribution is
**temporal safety via CHERI-D**: an 8-bit generation (lifetime) ID on each capability and a matching
tag on each allocation slot, checked on every dereference, with `free` incrementing the slot's
generation to invalidate stale capabilities **without a GC sweep** — closing use-after-free cheaply,
which matters across the distributed shared store (§6.5). ASLR is largely moot for native cells (a
pointer is inert without its capability, T6) and applies only to the legacy sandbox, where each launch
gets a fresh randomized layout.

### 5.14 Amnesic mode & anti-forensics **[Vision]**

A **volatile domain** is a security domain whose object graph is RAM-resident, whose per-object keys are
ephemeral (generated at unlock, sealed to the live session, never persisted or escrowed), and whose
commit path is disabled — so eviction discards rather than spills, and any sensitive spill goes to
encrypted swap. After power loss the keys are gone and the ciphertext is computationally useless
(crypto-shredding by power-off). Cold-boot defence layers zero-on-free, RAM scrubbing on lock/shutdown,
and page scrubbing on reclaim under the Tier A/B memory encryption of §5.11; a capability-bounded
**boot-anchor watchdog** triggers an emergency key-scrub and shutdown if the boot/identity medium is
removed (live-seizure defence). The classic FAT/NTFS data-carving problem mostly vanishes because
DominionOS has no filesystem: native deletion is crypto-shredding plus a content-less tombstone, with no
directory entries or cluster chains to recover and no stray temp/prefetch files scattered across a
medium; the only carving surface is a legacy volume reached through the projection VFS or personality
sandbox, where deletion enforces an immediate zero-pass overwrite.

### 5.15 Deniable storage & coercion resistance **[Vision]**

Encryption defeats an adversary without the key; deniable storage defeats one who can *compel* it.
DominionOS starts strong because at rest everything is per-object ciphertext in a content-addressed store —
a field of opaque hashes with no plaintext namespace to enumerate. A **decoy domain** (duress-unlocked,
plausible contents) and an independently-keyed **hidden domain** (derived from a disjoint branch of the
HD hierarchy) interleave in the same store; without the hidden secret, the hidden domain's objects are
statistically indistinguishable from unused random blocks, so a second key's existence is
cryptographically unprovable. Immutability removes the legacy hidden-volume collision hazard (decoy
writes create new objects and never overwrite hidden ones), and under coercion the user unlocks only the
decoy, so the hidden domain's keys are never derived and leave no RAM footprint for a memory dump — the
deniability that legacy "mount read-only to hide the header" workarounds chase, achieved structurally.
There is no escrow share to subpoena.

### 5.16 Anti-fingerprinting & private browsing **[Vision]**

Tracking has moved to fingerprinting (canvas/WebGL/font entropy) and permission over-reach, and DominionOS
removes much of it by construction. The **native** web has *no JavaScript fingerprint surface* — it
renders content-addressed semantic objects directly to the toolkit scene, with no DOM, no ambient JS
context, and no canvas/WebGL to probe — so it is fingerprint-resistant by absence. The **legacy** web
runs in a SIP holding only network + surface capabilities, and gets fingerprint resistance compiled
*into* the engine (canvas/WebGL/font normalization with coherent, internally consistent personas)
rather than as detectable JavaScript shims. "Storage Scopes" and "Contact Scopes" are not a feature to
add — **capability-scoped views are already the default**: an app sees only what it created or the user
explicitly picks, and "all storage / all contacts" is not a grantable shape. Hardware proximity attacks
are narrowed by blocking USB data lines when locked and shutting down idle short-range radios. Finally,
**Tor stream isolation** maps each per-context **pseudonymous identity** (§5.9) to a distinct SOCKS
isolation token, so two services seen through two identities ride unlinkable circuits — the network-layer
counterpart to the system-layer unlinkability DominionOS already provides.

---

## 6. Heterogeneity, Resources, and Reactivity

### 6.1 The reactive subscription plane **[Implemented]**

NDN (§3.8) is a *pull* substrate; the reactive plane (`pubsub.rs`, plus a single
`ndn.rs::Forwarder::notify` extension) makes *push* a first-class OS primitive. A topic is an
object-graph prefix; an event is a new immutable version; a **subscription is a capability**
(`TopicCap`) carrying publish/subscribe rights, declared delivery semantics
(`AtMostOnce`/`AtLeastOnce`/`ExactlyOnce`), an optional `since` cursor for diff-based delivery, a
rate-limit quota, and a TTL. Exactly-once falls out of content-addressing (same hash = same event);
delivery is encrypted by identity over `session.rs`; revocation is recursive and instant via the
firewall's reachability; cross-domain subscription goes through the Airlock; and fan-out rides the
NDN PIT reverse path with Content-Store caching rather than N point-to-point sessions. Event *arrival
timing* is treated as I/O at the determinism boundary while *what* was delivered (by hash, at which
logical epoch) is logged. The plane also offers standing (reactive) queries that maintain a
materialized result-set, CRDT-backed multi-writer topics, typed event schemas, and presence/liveness
as built-in queries. This single model collapses "server, socket, auth, serialization, event
delivery" into one — subsuming much of the client/server stack. (14 host + 1 on-metal tests.)

### 6.2 The resource governor **[Implemented]**

Resource pressure — memory, compute, energy, bandwidth — is treated as one explicit, bounded,
revocable **capability-quota problem** (`governor.rs`). Each domain has a unified `DomainLedger`
(memory + energy + compute). Admission control (`reserve`) returns
`Granted | Degraded(level) | Deferred | Refused` across pressure tiers `Comfortable → Tight →
Critical` (70%/90% watermarks) and **never OOM-kills**: it sheds caches, declines speculative
allocations, and defers essential-over-budget work after writing committed state through the commit
path. Reclamation is **by recomputability**, cheapest class first:
`Regenerable < CleanPersisted < CleanInMemory < Dirty < Pinned` — regenerable latents and Content-Store
entries drop immediately; clean-persisted objects evict and fault back by hash; dirty objects commit
then evict; pinned keys/capabilities/kernel state never evict. Self-optimizing placement is a
Q-learning bandit over `{Cpu, Gpu, Npu, Hot, Warm, Cold}` seeded by the `@NPU`/`@GPU`/`@CPU`
decorators, with reward = −(latency + energy + pressure). A hard determinism guard-rail records every
decision (evicted X, migrated Y, refused Z) as a replayable input while keeping raw pressure
measurements at the boundary. A companion **Recovery Supervisor** (`supervisor.rs`, motivated by
red-team finding B, §11) adds exponential-backoff restarts, per-component circuit breakers
(Open/HalfOpen/Closed), and a global storm guard against correlated failures, all
deterministically replayable. (Governor: 8 host + 1 on-metal; Supervisor: 7 host.)

### 6.3 Power & energy management **[Vision/Modeled]**

Energy is a first-class scheduling dimension: the heterogeneous scheduler minimizes
`f(latency, throughput, energy, thermal)` subject to capabilities, routing background work to E-cores
or batched NPU and latency-critical work to P-cores (`power.rs`). A power budget is a capability
(`Resource<Joules>` / `Resource<Watts>`) reusing the rate-limiting machinery. The design covers DVFS,
C-states, tickless idle (which also reduces non-determinism), per-core gating, PCIe ASPM, and wake
coalescing. **Suspend/hibernate is a state-machine checkpoint**: hibernate commits the deterministic
state machine to the encrypted persistent graph, and resume is a verified state restore — a tampered
hibernation image is detected because it is content-addressed and signed. Thermal and battery
feedback drive a coherent degradation order, and a sudden sustained energy draw is an anomaly signal
fed to the threat detector. The FSM, per-domain budgets/throttle, and battery integration are
implemented.

### 6.4 Driver synthesis & the uniform device model **[Modeled; L3/L4 Implemented & Executed]**

To achieve broad device coverage without hand-coding every driver, untrusted driver code is confined
in capability-bounded cells and validated by deterministic replay against device models, across a
five-layer stack (cheapest to most effort):

- **L1 — Class drivers** (one per standard, not per device): virtio, USB HID/MSC/CDC/UVC/UAC, NVMe,
  AHCI/SATA, xHCI, HD-Audio, SD/MMC, PCIe BARs, GOP framebuffer — ~70–85% coverage, largely in-kernel.
- **L2 — Generic enumerate-and-bind** from self-describing hardware (PCI config space, USB
  descriptors, ACPI/DSDT, Device Tree) with no per-device code.
- **L3 — Declarative spec → synthesized, capability-bounded driver** (the core innovation): a device is
  *data* — `DeviceSpec { class, ResourceClaim, register map, programs }` — bound by one reusable
  `Driver` runtime whose `bind()` mints an MMIO capability bounded to the claimed window via the CHERI
  HAL; malformed specs are refused at bind and a wrong spec cannot escape its device. Implemented and
  tested in `driver.rs`.
- **L4 — Safe foreign driver reuse** (a safe NDISwrapper/LinuxKPI): load, confine, **and execute**
  real Windows `.sys` (PE/COFF) and Linux `.ko` (ELF64) drivers at runtime. `foreign.rs` parses the
  actual binary container (PE: DOS stub → `PE\0\0` → COFF header → section table; ELF: header →
  section-header table → name strtab), reads the driver's imported kernel symbols from its `.kpi`
  section and its device logic (a serialized `DeviceSpec`) from its `.drv` section, admits it through a
  **default-closed** symbol whitelist, confines it to an MMIO capability over exactly its claimed
  window, and **binds it to the `driver.rs` runtime so it actually drives a device** — a network frame
  is sent through *both* a borrowed Windows driver and a borrowed Linux driver. Tested (15 host + 1
  on-metal): both kinds load and are used to send a frame; a `.drv` register that escapes the window is
  refused at bind; un-shimmed imports, corrupt/mislabeled containers, and capability tampering all fail
  closed. IOMMU-style DMA bounding (`drivergen.rs::DmaClaim`) composes on top.
- **L5 — AI-drafted specs/glue**, admitted only on `is_well_formed` + a DST pass + a WCET budget +
  capability sandboxing.

The safety argument rests on the three primitives: spatial containment (MMIO/DMA/IRQ bounded by
capabilities), fail-closed behavior, validation-before-trust (deterministic replay against a device
model, gated by a 90% conformance corpus), and provenance (signed, content-addressed specs with
instant rollback).

### 6.5 Distributed SASOS: one address space across a fleet **[Vision]**

The natural completion of Stage 3 (one address space on a machine) and Stage 4 (a machine as a network
of cores) is to let that single address space **span machines**: a cell on one node references an object
hosted on another by the same hash-name it would use locally, and the system moves the *data*, not the
*address*. DominionOS gets this almost for free because an object's name was never a raw pointer — it is a
content hash (location-independent) gated by a capability (unforgeable across the network). A fault on a
non-resident reference therefore issues an **NDN Interest** for the object's hash rather than touching a
swap file, and the nearest cached, producer-signed copy is returned and verified by re-hashing; "writing"
a shared structure publishes a new immutable version and notifies holders via the reactive plane.
Mutation never races a physical page — value-bearing state uses the **BFT** consensus tier (plain
Raft/Paxos assume honest nodes; the public fleet does not), and the causal-fencing result (§3.11) holds
at fleet scale so a rolled-back object is never resurrected by a lagging peer. Work migrates by shipping
a cell's small control state and paging its working set in by hash — the snapshot-and-restore mechanism
pointed at another node — and use-after-free across the shared store is caught cheaply by the CHERI-D
generation check (§5.13), no global GC required. Determinism survives: a remote fetch's *result* (which
hash, which logical epoch) is a recorded input event while wire timing stays at the boundary, so a
distributed run still replays bit-for-bit under DST.

### 6.6 The decentralized compute marketplace **[Vision]**

On that substrate, idle capacity anywhere can serve work originating anywhere, under two DominionOS
constraints: the machine's owner is sovereign over it, and neither host nor guest trusts the other.
**Sovereignty is a capability budget**: opting in grants a bounded, revocable slice of resources to a
segmented **Public-Work domain**, and "a slice of resources" is exactly a Resource Governor budget —
so admission control, graceful degradation, and instant recursive revocation are reused, not reinvented,
and idle/temporal/thermal/battery limits are expressed as energy-scheduling predicates. The Stage 4
global scheduler becomes a fleet-scale, **reverse-auction** orchestrator that routes each task to the
cheapest node satisfying its resource claim, pipelining a too-large model's layers across neighbours
through the distributed address space, with **no central control plane** (a BFT replicated scheduler).
Two pools share the one scheduler and the one isolation model: a **private pool** for confidential
workloads (the `enclave.rs` HAL over TDX/SEV-SNP/SGX/GPU-CC, plus FHE via `HomomorphicCiphertext`, with
remote attestation so a job can require a minimum confidential-compute tier — trust shifted to attested
silicon and math), and a **public pool** for permissionless collaborative compute (CHERI-D
generation-tracked SIP cells whose *results* are accepted only after verification — trust shifted to
mathematics).

### 6.7 Compute-backed settlement & Proof-of-Useful-Work **[Vision]**

Leasing spare compute needs native settlement, with no credit-card rail or central biller. DominionOS
settles in a fully-reserved, fiat-pegged, compute-backed credit ("DominionCredit"): a **wallet is a
capability-held identity** sealed under Stage 14, a **payment is a sanitized, recorded capability
transfer through the Airlock**, and the **ledger is the content-addressed graph** on the value-bearing
BFT tier with anti-rollback generations — money is exactly the invariant-critical state the tiered
consistency model reserves for linearizable agreement. Consensus does *useful* work, and DominionOS is an
unusually good host for it because verification reduces to its existing determinism: under
**Proof-of-Inference**, with weights/inputs/decoding locked, a compliant node must produce a
bit-identical output (Grid Snap already defeats GPU floating-point non-associativity), so a validator
re-checks with one forward pass and a hash compare — `O(1)`, not re-simulation. Non-deterministic
training uses **Proof-of-Learning**: optimistic accept against posted stake, a randomly sampled
ZK-verified spot-check of the gradient trajectory, and slashing for a caught fraud. The peg is held by
fully-reserved backing (mint on deposit, burn on redemption), EIP-1559-style fee-burn under load, and
real-time proof-of-reserves oracles; privacy comes from per-context pseudonymous identities and ZK
confidential-transaction proofs. It is a settlement utility for compute, explicitly *not* a speculative
instrument or a Turing-complete contract platform — programmable policy is ordinary contained Dominion.

---

## 7. User Interface & Applications **[Implemented]**

The UI is a cohesive, capability-driven stack verified on metal (109/109 shell selftests within the
broader suite), where the user manipulates a *living knowledge graph*, not "apps and files."

**Rendering and toolkit** (`toolkit.rs`, `uikit.rs`): one renderer-agnostic widget tree and a
design-token theme produce identical `Vec<DrawCmd>` scene descriptions regardless of backend, with
GPU-first selection and automatic framebuffer fallback. A flex/constraint layout engine, a `hit_test`
for event routing, and incremental clipped rasterization with EDF frame scheduling and governor-driven
animation shedding keep the UI fast. The determinism boundary logs scene descriptions and input events
while keeping raw presentation timestamps outside the replayable core.

**Design system and shell** ("Calm Spatial," `shell.rs`, `anim.rs`): semantic tokens (e.g. dark
`bg #121418`, `primary #4f9cff`; an 8 px spacing unit; a 12/15/19/26 pt type scale; 120/200/320 ms
motion honoring reduced-motion), a content-first dashboard, a command-palette launcher that fuses
search, launch, and natural-language intent into capability invocations (replacing the app list), and a
capability/permission panel that replaces "Settings" by listing every grant.

**The three-page shell** (`os.rs`) — Desktop, IDE, Explorer — sits behind a persistent dock with a
universal chrome, sharing an incremental damage-rectangle render model (idle ≈ 5 repaints/s, ~0.2–0.9 ms
per repaint under QEMU) and decoupling cursor compositing (200 Hz) from heavy raster (30 fps). The
**Desktop** (`desktop_page.rs`) is a spatial launcher of object cards; the **IDE** (`ide.rs`) keeps a
visual node-graph and Dominion source as two bidirectional views of one AST, with a pretty-printer
(`lang/emit.rs`) such that `parse(to_source(p))` round-trips; the **Explorer** (`explorer.rs`) is a
searchable knowledge constellation with a secure-compartment and hardware viewer and live health
monitoring. A bare-metal compositing path also exists (`kernel/src/gfx.rs`, `kernel/src/mouse.rs` on
IRQ12, `kernel/src/desktop.rs`) with dirty-rectangle and cached-wallpaper optimizations to stay
responsive under QEMU's TCG interpreter.

**Composable UI** (`compose.rs`): every page carries a `Board` of movable/resizable/removable panels,
unlockable from the top bar, with a widget picker and an 8 px grid snap; whole layouts serialize to
content-addressed, versioned byte packs (keyed by `Hash256`) that travel between pages, devices, and
users over the backup/sync plane.

**The global text engine** (`text.rs`): one reusable `TextBuffer` backs every editable surface
(editor, terminal, input fields) with caret-anywhere editing, full navigation, click-to-place mapped to
the kernel rasterizer's exact glyph advance, selection, and a blinking caret that is a pure function of
the clock (hence replay-deterministic). The **terminal** (`terminal.rs`) is a proper scrollback REPL
with history and a pluggable backend (the default Dominion REPL evaluates `2+2 → 4` through the real
interpreter) and is embedded as the IDE console.

**The universal workspace** (`workspace.rs`): there is *no* separate IDE app, text-editor app, or file
manager — there is one tabbed window where each tab is a data object in a view, sharing one theme, one
undo timeline, and one shortcut set. The universal editor (`editor.rs`) combines Notepad++ ergonomics,
opt-in Vim modality, and an inline Dominion calculator/notebook (any expression line evaluates in place).
The **universal browser** (`browser.rs`) unifies the native semantic web (NDN, content-addressed,
verified by hash, rendered directly to the toolkit scene) and the legacy web (HTML/JS in a sandbox SIP
holding only network + surface capabilities, JS gas-metered) in one tab, with a *real* Tor toggle
(SOCKS5 to a local Tor daemon) that blocks rather than leaks before the circuit is up.

All pages read from one shared `World` (`world.rs`) of real `dominion-core` primitives — real `Object`s
in a content-addressed `ObjectGraph`, programs authored in the IDE as first-class `"Program"` objects,
real capability-bounded `Sandbox`es, and rights/provenance derived from a TCB-root `Capability` — so
creating a program in the IDE makes it appear the same frame as a Desktop card and an Explorer node.
There are no hard-coded UI mocks; only the data *source* (seeded objects vs. live kernel enumeration)
remains to be swapped.

---

## 8. Cross-Cutting Systems Concerns

Beyond the staged vision, a gap analysis surfaced subsystems a real OS needs, each now designed against
the three primitives and, in most cases, implemented and tested.

- **Persistence & crash consistency** (`journal.rs`, **[Implemented]**): copy-on-write commit with an
  atomic root flip and a double-buffered, signed superblock with an anti-rollback generation counter —
  no fsck, verified on real virtio-blk under mid-commit fault injection.
- **Backup, sync & device fleet** (`backup.rs`, **[Implemented]**): zero-plaintext backup to untrusted
  storage, content-addressed and deduplicated with hash-verified restore, and a CRDT `FleetIndex` merged
  by HLC order for one logical fleet with offline-first reconcile.
- **Update & upgrade lifecycle** (`update.rs`, **[Implemented]**): signed content-addressed releases,
  A/B dual-bank slots with atomic switch, anti-downgrade counters, watchdog auto-revert, live cell
  hot-swap, DST canaries, and staged rollout.
- **Windows/macOS/Linux app support** (`compat.rs`, **[Implemented]**): PE/Mach-O/ELF detection with
  per-ABI syscall→capability translation and path projection inside the personality sandbox; the design
  covers Wine/Proton (DirectX→Vulkan via DXVK/vkd3d) and Darling-style macOS support (Metal→Vulkan),
  legality-gated.
- **Cross-platform targets** (`arch.rs`, `enforcement.rs`, **[Implemented]**): `dominion-core` is
  `no_std` + `#![forbid(unsafe_code)]` and already cross-compiles to `aarch64-unknown-none`; the kernel
  gets a per-architecture HAL (`Boot`, `MemoryMap`, `InterruptController`, `Timer`, `Console`, `Mmio`,
  `PowerStates`, `EntropySource`, `RootOfTrust`, `CapabilityEnforcementBackend`). An abstract input model
  (pointer/key/touch/gesture) means there is no desktop-vs-mobile fork; baseband/telephony/GPS are modeled
  as capability-bounded cells in their own domain.
- **Accessibility & i18n** (`a11y.rs`, **[Implemented]**): accessibility is a first-class **Assistive
  View** over the knowledge graph — which *is* the authoritative accessibility tree, eliminating the
  desynchronized parallel a11y tree — with a semantic tree, focus order, an i18n catalog with RTL/bidi,
  complex-script shaping, IME for CJK, and conversational/voice control as a native path. A11y conformance
  is testable deterministically under DST.
- **Compatibility & conformance** (`conformance.rs`, **[Implemented]**): compatibility is *defined* as a
  measured pass-rate over corpora with a **90% per-category release gate** and built-in suites
  (binary-format detection, foreign-ABI containment, codec round-trips, sandbox containment), with
  external suites (LTP/POSIX, WPT, Vulkan CTS, Wine) on the roadmap.

The unifying integration principle is: **project, don't pollute** (sockets, paths, files, and URLs become
*projections* over the three primitives — e.g., the POSIX-projection VFS `vfs.rs` renders the object graph
as mountable paths); **contain, don't absorb** (legacy runtimes run capability-bounded); and **bridge at
reusable seams** (VFS projection, codec registry, network gateway, name-resolution bridge). Mechanism lives
in Rust (`dominion-core` safe-only, host-testable; `dominion-kernel` hardware-touching), policy and content in
Dominion, and every legacy bridge has a native counterpart that eventually supersedes it.

---

## 9. Implementation and Evaluation

> **Vision ≠ reality.** This section reports only what is measured in code, reconciled with the codebase
> as of 2026-06-18/20. The vision (Sections 3–8) describes the destination; this describes where the
> wheels touch the road.

### 9.1 Build and test health (measured)

- `dominion-core` and `dominion-kernel` build clean; `cargo clippy --all-targets` → **0 warnings**.
- **441 host unit tests** pass (`cargo test`), up from a 191+57 baseline through staged build-outs
  (the per-module audit recorded 563 host + 102 on-metal at one checkpoint; the spec-completion phases
  carried the headline figure to **441 host + 87 on-metal** in the next-stages plan — numbers are quoted
  as the source records them at each checkpoint).
- **87 on-metal tests** pass (the kernel boots under QEMU and exits 33), built with the **release** kernel
  (`--features qemu_test`); the debug kernel ELF (~31.8 MB) grew too large to boot, while the release ELF
  (~1.66 MB) boots clean.
- `dominion-core` cross-compiles to **`aarch64-unknown-none`** (the mobile target).
- **Real SMP**: `kernel/src/smp.rs` brings APs online (ACPI MADT + LAPIC + AP trampoline +
  INIT-SIPI-SIPI; APs share BSP page tables) — **8 cores online, near-linear 8.3× scaling** — behind the
  `smp` feature.
- `dominion-core` is genuinely `#![forbid(unsafe_code)]`; `unsafe` lives only in `dominion-kernel` (RDRAND
  asm, MMIO).

### 9.2 What boots and works (verified in code)

The prototype boots on a real VM (bootloader 0.11) → framebuffer console, GDT/IDT/PIC, paging, heap →
the ASH safe-mode terminal (drivers: VGA framebuffer, PS/2 keyboard, serial). The two pure-logic
**keystones** are done: the POSIX-projection VFS (`vfs.rs`, K1) and the codec/Blob registry
(`codec.rs`, K2). The four **foundations are done and tested on metal**: **M1 Persistence** (object graph
serialized to a virtio-blk disk, *survives reboot*); **M2 Process/isolation** (cooperative SIP scheduler,
domain isolation, zero-copy IPC); **M3 Driver framework** (PCI enumeration, virtio-pci, virtio-blk,
virtio-net with a *real ARP round-trip* with the gateway); **M4 Compositor** (surface/window z-order
blitted to the framebuffer). An **ELF loader** parses and *executes real x86-64 machine code* (returns
42). Integration bridges include the Linux sandbox, a syscall-translation personality
(`fork` = snapshot-and-branch), DominionLink (self-certifying IDs + Kademlia DHT + DNS bridge), and the
Dominion-native web. The full capability / crypto / encryption / firewall / airlock / attestation stack is
exercised.

### 9.3 Measured benchmarks (in-guest, `kernel/src/bench.rs`)

A nine-workload battery runs on metal (`--features qemu_bench`, → `bench-results.json`):

| Workload | Mechanism | Result |
|---|---|---|
| Process/task creation | SIP domains | **2.2 M spawns/s**, 142 B/task (1,000,000 spawned) |
| Message passing | zero-copy channel IPC | **322 M msg/s at 3 ns latency** (5,000,000 ping-pong; 1e9 projected ≈ 3.1 s) |
| Memory pressure | fallible heap (`try_reserve`) | allocates to the OOM ceiling **without panicking**; recovered in **39 µs** |
| Storage | virtio-blk + object graph | seq/rand read ~12–26 MiB/s (25–53k IOPS); write ~0.85 MiB/s (~1.7k IOPS); 200k-object metadata at 5.4k puts/s (~187 µs each) |
| Security overhead | vault vs plaintext | real AEAD only (SHA-256-CTR removed). Default **ChaCha20-Poly1305** (~3× faster than SW AES-GCM) + independent **AES-256-GCM**; both optimized (T-table AES, windowed GHASH, fused encrypt). Re-run `run-bench.ps1` for current on-metal MiB/s and overhead % |
| Task completion | 20,000 domains to completion | **52k dispatch/s** |

### 9.4 Validation that the numbers are real

A separate validation battery (`--features qemu_validate`) proves the benchmarks reflect *real hardware*
rather than a flattened emulator. With hardware acceleration (whpx) auto-detected, a **memory-latency
mountain** recovers the true cache hierarchy via SLAT — **L1 ≈ 0 ns, L2 ≈ 3 ns, L3 ≈ 10 ns, DRAM ≈ 92
ns** — and a **model-vs-hardware-cost boundary** distinguishes an in-memory message (3 ns) from a VM exit
(4.4 µs) and a block read (29 µs). A soak test checks for drift/leak/fragmentation, and chaos
failure-injection passes **4/4 deterministic-recovery checks**. (Earlier launchers were stuck on
single-threaded TCG, which is why the VM previously *appeared* CPU-capped.)

### 9.5 Per-subsystem verdicts

The code audit assigns each subsystem one of: ✅ implemented and tested (real logic, real tests);
◐ implemented but software-modeled (semantics in safe Rust, hardware mechanism behind a HAL);
○ vision/proposed. Selected verdicts:

| Subsystem | Module | Verdict |
|---|---|---|
| Content addressing (SHA-256) | `hash.rs` | ✅ |
| Capability model (CHERI algebra in SW) | `capability.rs` | ◐ provenance/monotonicity/integrity/bounds all trap |
| Semantic object graph | `object.rs` | ✅ immutable, dedup, commit/rollback, serialize |
| Deterministic state machine | `state.rs` | ✅ replay, rewind, seeded RNG |
| Dominion language | `lang/*` | ✅ lexer→parser→interpreter; cells, `=>`, `@NPU/@GPU/@CPU` |
| Persistence (M1) | `persist.rs`, `block.rs`, `virtio.rs` | ✅ survives reboot |
| Scheduler (M2) | `sched.rs` | ✅ cooperative SIP, zero-copy IPC |
| Driver framework (M3) | `pci.rs`, `virtio.rs`, `netif.rs`, `dma.rs` | ✅ real ARP round-trip |
| ELF loader | `elf.rs`, `loader.rs` | ✅ executes real x86-64 |
| TRNG | `entropy.rs` | ✅ SP 800-90B RCT+APT, SHA-256 conditioning, fail-closed |
| Lattice PQ KEM | `lattice.rs` | ◐ real LWE KEM, reduced parameters |
| AES-GCM + memory-at-rest | `memcrypt.rs` | ✅ FIPS-197 / NIST KATs |
| ChaCha20-Poly1305 AEAD | `chacha.rs` | ✅ RFC 8439 KATs; default vault suite, PQ-resilient |
| Capability firewall / airlock | `firewall.rs`, `airlock.rs` | ✅ reachability, sanitize-to-min, recursive revoke |
| Universal encryption vault | `vault.rs` | ✅ zero-plaintext, crypto-GC; real AEAD (ChaCha20-Poly1305 default + AES-256-GCM), cross-family migration |
| CHERI tag HAL | `cheri.rs` | ◐ software-tag backend (unforgeable MAC) + pluggable hardware |

### 9.6 What remains hardware-gated (realism, not features)

Real CHERI tag bits, seL4-grade machine-checked proofs, production-parameter and constant-time PQC, NPU
codec execution, and real RF/UDP transport under the NDN overlay. **Each already has a working, tested
software implementation behind a HAL/abstraction — the silicon is the accelerator, never the
requirement.** None of these gaps is a bug in the current code; they are the boundary of what a commodity
prototype can physically exercise.

---

## 10. Design Tensions and Their Resolutions

The architecture names its hardest internal contradictions and resolves each explicitly, retaining a
residual-risk note rather than claiming a free lunch.

- **T1 Verified TCB vs. AI.** ML is advisory-only, never authoritative; verified deterministic components
  make every allow/deny decision; AI cells hold no capabilities, storage-write, or infrastructure
  authority and carry provenance and attestation (signed weights, training lineage).
- **T2 Hard-real-time vs. ML latency.** Mixed-criticality bands with reserved CPU/accelerator budgets;
  hard-RT paths are WCET-analyzable while unbounded inference is preemptible, with a classical fallback
  (the Stage 5 Scout pattern generalized) that always meets the deadline.
- **T3 Immutability vs. deletion.** Crypto-shredding (destroy keys) plus content-less tombstones (§5.12).
- **T4 CRDT vs. strong consistency.** Tiered and declared per object: CRDTs for convergent state,
  linearizable consensus (Raft/Paxos intra-machine, BFT cross-fleet) for value-bearing state.
- **T5 Determinism vs. entropy.** Seeded DRNG feeds execution; hardware entropy is a recorded input event
  (§5.5).
- **T6 SASOS pointers vs. least privilege.** A pointer is inert without a capability; reachability follows
  authority topology, not address visibility.
- **T7 CHERI vs. commodity hardware.** The tiered Capability Enforcement Layer (§3.3); attestation reports
  the tier; high-assurance domains require a minimum tier.
- **T8 Adaptive resources vs. determinism.** Adaptive decisions are recorded as inputs to the deterministic
  log (not hidden state); raw measurements stay at the boundary; replay reproduces the trajectory (§6.2).

---

## 11. Adversarial (Red-Team) Findings

The deterministic discipline turns every bug into a permanent, replayable regression. Three documented
findings shaped the design:

- **Finding A — Split-timeline merge corruption.** A naive last-writer-wins merge across a gossiping fleet
  could resurrect a value that one node had rolled back. Fixed by **causal fencing**
  (`consistency.rs::FencedReplica`, §3.11), validated by a 300-seed loss/reorder/partition sweep that
  asserts the poison ends up on no replica.
- **Finding B — Restart storms.** Unbounded cell restarts under correlated failure. Fixed by the
  **Recovery Supervisor** (`supervisor.rs`, §6.2): exponential backoff, per-component circuit breakers, and
  a global storm guard, all deterministically replayable.
- **Finding C — Anonymity correlator leak.** A naive ZK login proved knowledge behind a *static* public
  key, a stable cross-context correlator. Fixed with **per-context pseudonyms** ($P_{ctx}=g_{ctx}^x$ via
  Chaum-Pedersen DLEQ) and a constructor-enforced `TraceabilityClass` that refuses to attach a global
  correlator to an "anonymous" transaction (§5.9).

The standing verification stack pairs formal verification (TCB, firewall/airlock, language safety,
proof-carrying updates) with DST (any behavior, reproducibly), property-based and fuzz testing
(`props.rs`, `fuzz.rs`), red-team capability tests asserting *mathematical* impossibility/containment of
privilege escalation, forgery, cross-domain leakage, confused-deputy, and airlock bypass, plus crypto KATs
and supply-chain/measured-boot tests.

---

## 12. Limitations and Honest Boundaries

The single largest dependency is **CHERI**: the entire spatial-safety and capability story ultimately
targets 128-bit hardware capabilities, yet CHERI silicon (Arm Morello, RISC-V CHERI) is barely commercial
and the current code runs on x86-64 with *none* of it. The mitigation is the Capability Enforcement Layer
(§3.3), which keeps the system fully functional at Tier 0 while honestly attesting the weaker posture.
Similarly, the PQC is correct but parameter-reduced and not yet constant-time; the microkernel proof is an
invariant/WCET harness, not yet a seL4-grade machine-checked proof; neural codecs and semantic audio are
implemented but await NPU execution for production quality; and the vault now seals with real AEAD on its
storage path — **ChaCha20-Poly1305** by default (RFC 8439) and **AES-256-GCM** as the independent second
family (the former SHA-256-CTR placeholder has been removed). The
test-count figures are quoted as each source document recorded them at its checkpoint and therefore differ
across documents; they should be read as monotonically increasing milestones, not a single canonical
number. None of these is concealed: the project's own `current-status.md`, `code-audit.md`, and
`FINDINGS.md` maintain the vision-vs-reality boundary, and this paper preserves it.

---

## 13. Roadmap and Future Work

The build order (distinct from the conceptual stage numbering) is milestone-driven. The keystones (K1 VFS,
K2 codec registry) and foundations (M1 persistence → M2 process/isolation → M3 driver framework → M4
compositor) are complete and tested on metal. The forward plan proceeds in tiers:

- **Tier 1 — close and harden what exists:** wire the crypto/encryption seams (universal network
  encryption, identity-bound sessions, encrypted-in-RAM objects, PQ NDN signatures, HIBC names); stand up
  the full DST/fuzz/property/chaos harness with million-seed sweeps and per-bug regression; and deepen the
  SASOS + Dominion language (affine type system, compile-time resource bookkeeping, AOT/JIT via DCG, the
  language reference and tooling).
- **Tier 2 — new native subsystems:** complete multikernel + heterogeneous scheduling (per-core CPU
  drivers, replicas + consensus, HLC causal ordering), generative storage + active defense, semantic
  audio, the object-centric/AI-native UI compositor upgrade, identity/recovery/web-auth/hardware-RoT, and
  persistence-consistency/backup/fleet-sync/lifecycle.
- **Tier 3 — frontier and breadth:** cross-platform and mobile (aarch64 boot paths, tiered enforcement
  backends, touch-first UI), the hardware-verified frontier (Stage 0 verified firmware, the seL4-grade
  kernel proof, real CHERI silicon, hardware memory encryption), and compatibility breadth (Wine/Proton,
  Darling, polyglot runtimes, external conformance suites behind the 90% release gate).

Open decisions explicitly tracked include the canonical Dominion syntax, the primary CHERI ISA, the
microkernel verification base, recovery-quorum parameters, the DominionLink congestion-control default
(BBR vs CUBIC), and the per-domain minimum enforcement tier.

---

## 14. Conclusion

DominionOS is an attempt to pay off fifty years of operating-system technical debt at once, by refusing the
1970s assumptions and rebuilding on three primitives — capabilities, a content-addressed object graph, and
a deterministic state machine — under a single discipline (P.I.E.) that gives performance and efficiency to
hardware and isolation and security to compilers and capabilities. The payoff is structural: whole classes
of exploit vanish by construction; the filesystem, the privilege ring, and address-based networking
dissolve into the graph and into capabilities; undo, debugging, and crash recovery become consequences of
determinism and immutability rather than bolted-on features; and heterogeneity, energy, and resource
pressure become one capability-quota problem. The prototype demonstrates that this is buildable on
commodity hardware *today* — it boots, persists, schedules across 8 cores, runs real machine code, talks to
a real gateway, and passes hundreds of host and on-metal tests with a zero-`unsafe` core — while drawing an
unusually honest line around the realism that still waits on CHERI silicon, machine-checked proofs, and
NPUs, each already standing on a tested software floor. The thesis is not that one algorithm or one chip
makes the system secure, but that the *architecture evolves faster than the attacks* — and the working
code is the evidence that the architecture is real.

---

## Appendix A — Glossary

- **Capability** — an unforgeable token naming a resource and the rights over it; monotonic, provenanced,
  integrity-protected, bounds-checked. The unit of all authority.
- **Content-addressed object graph** — the immutable, deduplicated, semantic store that replaces the
  filesystem; every object is named by the hash of its content.
- **Deterministic state machine** — the whole machine state as a hashable object; every action is an input
  to a state-transition function, making execution bit-for-bit reproducible.
- **P.I.E. Principle** — Performance and Efficiency are the hardware's job; Isolation and Security are the
  compiler's and capability system's job.
- **CHERI** — Capability Hardware Enhanced RISC Instructions; 128-bit architectural capabilities providing
  hardware provenance, monotonicity, integrity, and bounds.
- **SASOS** — Single Address Space Operating System; one global virtual address space where protection is
  orthogonal to translation and enforced by capabilities.
- **SIP** — Software-Isolated Process; a closed object space that shares writable memory only through
  statically verified channels.
- **Cell** — the unit of software modularity, isolation, hot-swap, and memory reclaim; cells do not spill
  state into one another.
- **Intralingual design** — matching the OS execution environment to the implementation language's runtime
  model so the compiler enforces OS invariants at compile time.
- **Multikernel** — treating one machine as a distributed network of cores with explicit message passing and
  replicated per-core state (after Barrelfish).
- **CRDT** — Conflict-free Replicated Data Type; commutative/associative, lock-free, convergent (not
  linearizable).
- **NDN** — Named Data Networking; routing on the cryptographic identity of data, not host addresses.
- **HIBC** — Hierarchical Identity-Based Cryptography; a data name acts as its own public key.
- **DominionLink** — the native networking overlay (self-certifying identities, content-addressed
  request/response, Kademlia DHT, DNS bridge) over UDP.
- **DCG** — Deterministic Compute Graph; the target of Dominion compilation (plus capability proofs) instead
  of an opaque binary.
- **Capability Firewall (11.14)** — intra-domain authority control over the reachability graph.
- **Capability Airlock (11.15)** — the sole, sanitizing, one-way path for capabilities to cross trust
  boundaries.
- **CAL / CEL / RTAL** — the Cryptographic, Capability-Enforcement, and Root-of-Trust Abstraction Layers,
  which make algorithms, enforcement tiers, and trust hardware pluggable.
- **DST** — Deterministic Simulation Testing; running the OS under a hypervisor controlling all
  non-determinism so any bug is reproducible from its seed.
- **Crypto-GC / crypto-shredding** — deletion by destroying keys, rendering ciphertext useless even in
  backups.
- **JSCM** — Joint Source-Channel Coding and Modulation; transmitting semantic tokens for ~10 dB SNR gain.
- **HLC** — Hybrid Logical Clock; provides causal ordering across cores and devices.
- **CHERI-D** — temporal-safety extension: an 8-bit generation ID on capabilities and allocation slots;
  increment-on-free invalidates stale capabilities with no GC sweep.
- **Distributed SASOS** — the single address space extended across a fleet; non-resident references fault
  in by content hash over NDN; value-bearing state uses BFT.
- **Compute marketplace / dual pool** — fleet-scale reverse-auction scheduling over sovereign opt-in
  budgets; a private (confidential-compute) pool and a public (collaborative) pool.
- **DominionCredit / PoUW / PoI / PoLe** — the fully-reserved compute-backed settlement credit, and the
  Proof-of-Useful-Work consensus (Proof-of-Inference, deterministic and re-checkable in one pass; and
  Proof-of-Learning, optimistic + stake/slash + ZK-sampled).
- **Amnesic mode / deniable storage / anti-fingerprinting** — privacy tier: volatile RAM-resident domains
  with ephemeral keys; cryptographically-unprovable decoy/hidden domains; native-web-has-no-fp-surface
  plus engine-level legacy resistance and Tor stream isolation.

## Appendix B — Module Index (selected)

*Core (`dominion-core`, `no_std`, `#![forbid(unsafe_code)]`):* `hash.rs`, `object.rs`, `capability.rs`,
`state.rs`, `random.rs`, `entropy.rs`, `lang/*` (lexer/parser/interp/ast/value/emit), `dcg.rs`, `vfs.rs`,
`codec.rs`, `persist.rs`, `sched.rs`, `pci.rs`/`virtio.rs`/`netif.rs`/`dma.rs`, `elf.rs`/`loader.rs`,
`surface.rs`, `net.rs`, `dominionlink.rs`, `transport.rs`, `ndn.rs`, `pubsub.rs`, `sandbox.rs`,
`personality.rs`, `compat.rs`, `wasm.rs`, `dominionweb.rs`, `crypto.rs`, `lattice.rs`, `tokensig.rs`,
`memcrypt.rs`, `vault.rs`, `zk.rs`/`zkservice.rs`/`anon.rs`, `firewall.rs`, `airlock.rs`, `attest.rs`,
`confidential.rs`, `lifecycle.rs`, `recovery.rs`, `identity.rs`, `rot.rs`, `time.rs`, `hlc.rs`, `power.rs`,
`governor.rs`, `supervisor.rs`, `pressure.rs`, `neural.rs`, `defense.rs`, `datatypes.rs`, `a11y.rs`,
`multikernel.rs`, `arch.rs`, `enforcement.rs`, `cheri.rs`, `verify.rs`, `dst.rs`, `fuzz.rs`, `props.rs`,
`update.rs`, `journal.rs`, `backup.rs`, `conformance.rs`, `driver.rs`, `foreign.rs`, `secureboot.rs`,
`memenc.rs`, `session.rs`, `audio.rs`, `appkit.rs`. *UI:* `toolkit.rs`, `uikit.rs`, `shell.rs`,
`anim.rs`, `os.rs`, `desktop_page.rs`, `ide.rs`, `explorer.rs`, `world.rs`, `compose.rs`, `text.rs`,
`terminal.rs`, `workspace.rs`, `editor.rs`, `browser.rs`, `nodes.rs`, `widgets.rs`, `dash.rs`.
*Kernel (`dominion-kernel`, hardware-touching):* `smp.rs`, `gfx.rs`, `mouse.rs`, `desktop.rs`, `keyboard.rs`,
`vga_buffer.rs`, `interrupts.rs`, `allocator.rs`, `bench.rs`, `main.rs`.

## References (primary sources in this corpus)

This paper synthesizes the DominionOS documentation set: `architecture/` (Stages 0–10 and cross-cutting),
`security/` (Stages 11/13/14, threat model, randomness, identity recovery, secure time, hardware RoT,
zero-knowledge proofs, system-domain confidentiality, memory encryption, data lifecycle, plus the
consolidated kernel self-protection & hardened memory, amnesic mode & anti-forensics, deniable storage &
coercion resistance, and anti-fingerprinting & private browsing), `architecture/` distributed SASOS and
the decentralized compute marketplace, the new `economics/` settlement layer (compute-backed settlement &
Proof-of-Useful-Work), `language/`
(Dominion spec, reference, data types, UI/applications, multi-language runtimes), `ui/` (rendering and
toolkit, design system and shell, desktop/IDE/explorer, composable UI, text engine, terminal, universal
workspace, universal browser, world, live dashboard), and `implementation/` (current status, code audit,
integration strategy, roadmap, next-stages plan, testing and verification, hardware targets, cross-platform
builds, hardware/software compatibility, Windows/macOS support, update lifecycle, persistence and crash
consistency, backup/sync/fleet), together with `GLOSSARY.md`, `FEATURE-CHECKLIST.md`, and `findings.md`.
Cited external prior art includes seL4, CHERI/Morello and CHERI-D, Barrelfish, the Mungi/Sombrero
distributed single-address-space lineage, NDN, CRDTs, Hybrid Logical Clocks, BFT replicated state
machines, GrapheneOS `hardened_malloc`, ARM MTE/PAC, Tails/Whonix amnesic and stream-isolation designs,
Google Spanner TrueTime, Roughtime/NTS, NIST SP 800-90A/B/C, FIPS 197, NIST FIPS 203 (ML-KEM), Intel
TME/MKTME/TDX, AMD SME/SEV-SNP, TPM 2.0/DICE, and the Smalltalk-80 MVC lineage.

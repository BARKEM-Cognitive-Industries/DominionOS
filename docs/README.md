# AetherOS Documentation

**Updated:** 2026-06-22 (Consolidated from 97 archived specs)

Welcome! This directory contains the unified architecture and implementation status for AetherOS, a capability-based microkernel OS with an object-centric runtime (Dominion), unified rendering, and on-device AI.

## Quick Start

### I'm new — where do I start?
1. **Read:** [`architecture.md`](architecture.md) (10 min read)
   - Gives you the mental model: capabilities, content-addressed storage, deterministic state machines
   - Explains 17 subsystems and how they connect
   - Lists design tensions and resolutions

2. **Find your subsystem:** Open [`subsystem-manifest.json`](subsystem-manifest.json)
   - Machine-readable map of all subsystems
   - Each subsystem has spec files and source modules
   - Shows status (SHIPPED, PARTIAL, DEFERRED)

3. **Dive in:** Read the spec file for your subsystem
   - Understand what *should* exist (SPECIFIED section)
   - Check what actually exists in code (IMPLEMENTED)
   - See what gaps remain (GAPS section)

### I'm an auditor / security reviewer — what's the critical path?
1. **Read:** [`implementation-status.md`](implementation-status.md) → "Critical Path to High-Assurance Deployment"
2. **Risk assessment:** Check per-subsystem risk levels in implementation-status.md
3. **Formal verification roadmap:** See security/ section (TBD: formal-verification-roadmap.md)
4. **Gaps to close:** In subsystem-manifest.json, look for HIGH/CRITICAL severity gaps

### I'm a contributor — how do I pick work?
1. **Check:** [`implementation-status.md`](implementation-status.md) → "What Ships in Release 1.0" vs. "Deferred"
2. **Pick:** A subsystem you want to improve
3. **Locate:** Look up its spec in subsystem-manifest.json → spec_files
4. **Read:** SPECIFIED, IMPLEMENTED, GAPS, PRIORITY sections
5. **Code:** Read key_source_files and start contributing

### I'm a product manager — what's the status?
1. **Executive summary:** [`CONSOLIDATION_SUMMARY.md`](CONSOLIDATION_SUMMARY.md)
2. **Go/no-go checklist:** [`implementation-status.md`](implementation-status.md) → "Go/No-Go Checklist for 1.0 Release"
3. **Ship date:** Ready (with terminal input routing quick fix, ~1–2 days)
4. **Roadmap:** Phases outlined in implementation-status.md

---

## Document Guide

### Core Architecture
- **[`architecture.md`](architecture.md)** — Unified system architecture
  - High-level overview of AetherOS
  - 17 subsystems, layer-by-layer breakdown
  - Design tensions and resolutions
  - Status quo and maturity assessment

### Status & Planning
- **[`implementation-status.md`](implementation-status.md)** — What's shipped, what's deferred
  - Subsystem status matrix (15 SHIPPED, 8 PARTIAL, 6 DEFERRED, 1 PROPOSED)
  - Per-subsystem risk assessment
  - Critical path to high-assurance deployment (3 phases)
  - Go/no-go checklist for Release 1.0

- **[`CONSOLIDATION_SUMMARY.md`](CONSOLIDATION_SUMMARY.md)** — How this documentation was created
  - Multi-agent workflow methodology
  - Key findings and gaps identified
  - Next steps for Phase 2 (formal verification, etc.)

### Machine-Readable Reference
- **[`subsystem-manifest.json`](subsystem-manifest.json)** — Authoritative subsystem map
  - 17 subsystems with modules, spec files, source files, status
  - Critical gaps and spec alignment % for each
  - Use this to locate code/specs for any subsystem

### Archived Specifications (Consolidation Source)
- **[`_archive_2/`](_archive_2/)** — Original 97 spec files
  - **architecture/** (20 files) — Stages 0–10, design patterns
  - **security/** (15 files) — Threat model, crypto, hardening
  - **ui/** (12 files) — Browser, desktop, shell, rendering
  - **implementation/** (15 files) — Status, testing, roadmap
  - **language/** (5 files) — Dominion language spec
  - **ai/** (3 files) — ML/agent/benchmarks
  - **economics/** (1 file) — Compute settlement
  - **Root level:** Consolidated papers, checklists, glossaries

---

## Subsystem Taxonomy

### Layer 1: Hardware & Boot (Stages 0–1)
- **Microkernel & Core** — Capability system, boot, SMP
  - Status: SHIPPED (85% aligned with spec)
  - Gap: Formal verification, measured boot
- **Device & Driver Management** — Hardware abstraction, PCI, driver synthesis
  - Status: SHIPPED (70% aligned)
  - Gap: Real bus enumeration, DST validation, L5 AI-drafted specs

### Layer 2: Memory & Storage (Stages 5–6)
- **Memory, Storage & Persistence** — Object graph, versioning, encryption
  - Status: SHIPPED foundation (60% aligned)
  - Gap: Generative compression (Stage 5 deferred), hardware encryption wiring

### Layer 3: Security (Stages 11–14)
- **Security & Cryptography** — ChaCha20, AES-GCM, PQ, vault
  - Status: SHIPPED (85% aligned)
  - Gap: Anti-rollback counter, formal proofs
- **Access Control & Hardening** — Firewall, airlock, amnesic, deniable storage
  - Status: SHIPPED (90% aligned)
  - Gap: Formal verification proofs, ML threat detection

### Layer 4: Networking & Communications (Stages 7–8)
- **Networking & Communications** — NDN, DominionLink, HLC
  - Status: SHIPPED (95% aligned)
  - Gap: Distributed firewall (single-node only)
- **Browser & Web Stack** — Native + legacy web, Tor, DOM, JS
  - Status: SHIPPED (90% aligned)
  - Gap: Terminal input routing (small fix)

### Layer 5: Computing & Rendering (Stages 8–10)
- **Rendering & Graphics** — 2D/3D unified, Nanite, ATW, HDR, text
  - Status: SHIPPED (95% aligned)
- **Desktop, Shell & Workspace** — 9 apps, window manager, composable UI
  - Status: SHIPPED (85% aligned)
  - Gap: Terminal interactivity, architecture fragmentation

### Layer 6: Language & AI (Stages 3–9)
- **Language & Runtime (Dominion)** — Safe language, FFI, LSP
  - Status: SHIPPED (95% aligned)
- **ML/AI & Accelerated Compute** — Inference, on-device models
  - Status: SHIPPED (75% aligned)
  - Gap: Generative compression, RL optimization, ML threat detection

### Layer 7: System Services & Operations
- **System Services & Lifecycle** — Task mgmt, identity, compliance, updates
  - Status: SHIPPED (95% aligned)
- **Verification, Integrity & Provenance** — Attestation, revocation, rollback
  - Status: SHIPPED (95% aligned)
- **Accessibility, Localization & UX** — WCAG, i18n, compliance
  - Status: SHIPPED (90% aligned)
- **Utilities, Testing & Debug** — Logging, testing, benchmarking
  - Status: SHIPPED (95% aligned)

### Layer 8: Advanced
- **Distributed Compute & Economics** — BFT, DSASOS, marketplace
  - Status: SHIPPED (90% aligned)
  - Gap: Multi-node deployment
- **Scheduling & Multikernel** — SMP, cooperative scheduling, governors
  - Status: SHIPPED (95% aligned)

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ SHIPPED | Functionally complete; passing tests; ready for use |
| ⚠️ PARTIAL | Core features work; known gaps; usable with caveats |
| 🔄 DEFERRED | Architecture done; implementation staged for phase 2 |
| 📋 PROPOSED | Spec only; research phase; no implementation yet |
| 🟢 LOW RISK | Gap is post-release or non-critical |
| 🟡 MEDIUM RISK | Gap affects completeness but not correctness |
| 🟠 HIGH RISK | Gap affects core functionality; should be addressed |
| 🔴 CRITICAL | Gap blocks deployment goal; must be resolved |

---

## Key Concepts

### The Three Primitives
1. **Capabilities** — Unforgeable tokens of authority; can be delegated and revoked
2. **Content-Addressed Object Graph** — Immutable, deduplicated storage (Git-like at OS level)
3. **Deterministic State Machine** — Every action is an input; machine state is reproducible and hashable

### Design Tensions Resolved
- **Formal verification vs. pragmatism** → Graceful degradation; software tags work; proofs deferred
- **CHERI hardware may not ship** → Tier system (Tier 2 ← Tier 1 ← Tier 0)
- **Generative compression complex** → Architecture supports stubs; implementation staged
- **Distributed safety** → Single-node TCB solid; multi-node work in phase 2

### Spec Alignment
- **Shipped subsystems:** 85% average alignment with archived specs
- **Key gaps:** Formal verification, measured boot, generative compression, hardware encryption tiers
- **Clear roadmap:** 3 phases to high-assurance deployment (12–16 weeks / phase 2, 20+ weeks / phase 3)

---

## How to Use This Documentation

### For Understanding the System
1. Read `architecture.md` for the mental model
2. Check `subsystem-manifest.json` to locate your area of interest
3. Read the SPECIFIED section of the relevant archived spec
4. Read the IMPLEMENTED section to see what's actually coded
5. Scan GAPS to understand what work remains

### For Finding Code
- Look up your subsystem in `subsystem-manifest.json`
- Find source modules in the `modules` array
- Key files listed in `key_source_files`
- Source lives in `dominion-core/src/` and `kernel/src/`

### For Planning Work
1. Check `implementation-status.md` for priorities
2. Look for subsystems with status ⚠️ PARTIAL or gaps marked HIGH/CRITICAL
3. Estimate effort (short/medium/high) from the roadmap
4. Escalate formal verification work to security team

### For Making Decisions
- **Release:** See "Go/No-Go Checklist" in implementation-status.md
- **Roadmap:** See "Phase 1/2/3" in implementation-status.md
- **Architecture:** See "Design Tensions & Resolutions" in architecture.md
- **Risk:** See per-subsystem risk matrix in implementation-status.md

---

## Navigation Tips

**Jump to a subsystem:** Open `subsystem-manifest.json`, search for your area (Ctrl+F or `grep`)  
**Find a spec file:** Search `_archive_2/` by name or look in manifest under `spec_files`  
**Understand a gap:** Check `GAPS` section in subsystem analysis (in workflow output)  
**Check ship status:** Look at `status` field in manifest or see implementation-status.md  

---

## Version History

- **2026-06-22:** Consolidated from 97 archived specs using multi-agent workflow
  - 17 subsystems identified
  - 12 subsystems analyzed in depth (SPECIFIED/IMPLEMENTED/GAPS/PRIORITY)
  - Unified architecture.md created
  - Implementation status matrix built
  - This README and manifest generated

- **Earlier:** Specs created in isolation across `_archive_2/` (fragmented, ~97 files)

---

## Questions?

- **Architecture:** See `architecture.md` → "How to Navigate"
- **Status:** Check `implementation-status.md`
- **Subsystems:** Look up `subsystem-manifest.json`
- **Consolidated methodology:** See `CONSOLIDATION_SUMMARY.md`
- **Spec gaps:** Check subsystem entries in manifest under `critical_gaps`

**Next steps:** Start with `architecture.md` for a 10-minute primer, then explore your area of interest!

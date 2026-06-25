# Contributing to AetherOS

We welcome contributions. This guide explains how we work and what we expect.

---

## Before You Start

1. **Read the architecture.** Start with `docs/architecture.md` to understand how the system fits together.
2. **Check subsystem specs.** Look up your area in `docs/subsystem-manifest.json` to find relevant specs and source files.
3. **Understand the license.** See `LICENSE.md`. Non-commercial contributions use AGPLv3. If you're contributing on behalf of a company, we may need a commercial agreement.

---

## Development Guidelines

### AI-Assisted Development

If you're using an AI to help write code, **you must use Claude Opus 4.8 (or newer Claude model).**

- Opus produces higher-quality systems code than other models.
- Other LLMs (GPT-4, Gemini, Llama) have historically introduced bugs in kernel and crypto code.
- If you use a different model, we may ask you to rewrite before merging.

**Why?** We've tested this. Opus understands capability systems, Rust safety semantics, and deterministic state machines better than alternatives. It matters for correctness.

### Code Quality

- **Compiler errors:** Zero. Code must compile with no warnings on stable Rust 1.70+.
- **Tests:** New code must pass all existing tests and include tests for new functionality.
- **Benchmarks:** No performance regressions. Run the benchmark suite before submitting.
- **Safety:** No unsafe code without documented justification. Unsafe is allowed; undocumented is not.
- **Documentation:** Public APIs need doc comments. Explain the why, not just the what.

### Scope & Size

- **One feature per PR.** Don't mix kernel changes with shell improvements in the same PR.
- **Keep PRs small.** Under 500 lines of change is ideal. Over 1000 lines, we'll likely ask for a split.
- **Isolated changes.** If your change touches multiple subsystems, explain the dependency clearly.

### Testing

We test on:
- **QEMU** (x86-64, 4 cores, 4GB RAM) — required
- **Real hardware** (optional but appreciated) — Intel/AMD x86-64 only for now

**Before submitting:**
```bash
# Build
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo run --release --bin bench

# Boot in QEMU and verify your feature works
./run.ps1
```

If it boots and runs, it's ready for review.

---

## Contribution Workflow

### 1. Discuss Major Changes First

**For significant features** (new subsystem, large architectural change, new crypto):
- Open an issue describing what and why
- Wait for feedback before coding
- We'll discuss approach, scope, and fit

**For minor changes** (bug fixes, small optimizations, documentation):
- A PR is fine; no issue needed

### 2. Fork, Branch, and Code

```bash
git clone https://github.com/yourusername/aetheros.git
cd aetheros
git checkout -b feature/my-feature
```

Write your code. Test frequently. Commit with clear messages.

**Commit message format:**
```
[subsystem] Brief description

Optional longer explanation of what and why.
Reference any related issues.

- Specific change 1
- Specific change 2
```

Example:
```
[firewall] Add per-capability rate limiting

Implements token-bucket rate limiting for individual capabilities
as discussed in issue #42. Each cap can be assigned a rate
(ops/sec, bytes/sec, % device). Exceeding the rate triggers
quarantine.

- Add RateLimit struct to firewall.rs
- Implement check() with token-bucket decay
- Add tests for quota enforcement and reset
- No performance regression on unquota'd caps
```

### 3. Submit a Pull Request

- **Title:** Keep it short. `[subsystem] what you did`
- **Description:** Explain what, why, and how. Link related issues.
- **Tests:** Describe what you tested and how.
- **Benchmarks:** Report before/after if performance-relevant.

Example:
```markdown
## Description
Implements capability-based rate limiting for the firewall.
Addresses #42 (DoS mitigation via quota exhaustion).

## Changes
- Added RateLimit struct to firewall.rs
- Per-capability token-bucket enforcement
- Quarantine domain on rate exceed
- ~50 lines of code, 30 tests

## Testing
- Unit tests: all pass
- Integration: booted in QEMU, ran firewall benchmark suite
- Benchmark: no regression on unquota'd paths

## Notes
- Rate limits are per-domain, not per-process yet (phase 2)
- We're using millisecond-resolution counters (fast enough for 1M ops/sec)
```

### 4. Code Review

We'll review for:
- **Correctness.** Does it work? Does it break anything?
- **Security.** Any new vulnerabilities? Unsafe code justified?
- **Performance.** Benchmark results acceptable?
- **Style.** Does it fit the codebase?
- **Tests.** Sufficient coverage?

We may ask for revisions. That's normal. We're aiming for high quality.

### 5. Merge

Once approved, we'll merge to main. You're a contributor!

---

## Subsystems & Ownership

No single person owns a subsystem, but here's who knows what well (as of June 2026):

| Subsystem | Owner/Expert | Status |
|-----------|--------------|--------|
| Capability system | @barkem | Core, well-tested |
| Firewall/Airlock | @barkem | Core, high-assurance focus |
| Storage/Object graph | @barkem | Stable |
| Crypto/Vault | @barkem | Stable |
| ML/Neural networks | @barkem | Working, optimization ongoing |
| Rendering/Graphics | @barkem | Implemented |
| Desktop/Shell | @barkem | Partial, needs wiring |
| Networking/NDN | @barkem | Stable |
| Dominion language | @barkem | Functional |
| Device drivers | @barkem | Partial (virtio working) |

**Contributing to an area?** Consider reaching out to the expert first. They can point you to gotchas and prioritize your work.

---

## Specs Are Ground Truth

Here's how we manage specs and code:

### Major Feature Addition
1. **Read the spec.** Check if there's already a design doc (docs/subsystem-manifest.json).
2. **Research & design.** Study the spec, related code, and related work.
3. **Propose changes to spec first.** If the spec is wrong or incomplete, fix it.
4. **Write tests & benchmarks.** Decide what success looks like before coding.
5. **Implement against the spec.** Code should follow the spec, not the other way around.
6. **Test & benchmark.** Verify the implementation matches the spec.

### Minor Change (Bug Fix, Small Improvement)
1. **Check the spec.** Is there a spec for this subsystem? If yes, update it if your fix changes behavior.
2. **Add tests.** Especially for bug fixes (test that should fail before fix, pass after).
3. **Implement.** Keep it small.
4. **Test & benchmark.** No regressions.

**Why?** Specs are our source of truth. Code should implement specs, not create them. If the spec is wrong, we fix it together first.

---

## Common Mistakes (Don't Do These)

❌ **Using non-Opus LLMs for significant code.** We've seen bugs. Use Opus.

❌ **Large PRs with multiple unrelated changes.** Split them up.

❌ **No tests.** Every change needs tests. No exceptions.

❌ **Performance regression without justification.** Benchmarks must pass.

❌ **Unsafe code without comments explaining why.** We'll ask you to rewrite.

❌ **Changing specs and code simultaneously.** Do specs first, then code.

❌ **Assuming you understand the security model.** Read `docs/architecture.md` first. Seriously.

❌ **Submitting without testing in QEMU.** We'll ask for this anyway.

---

## Questions?

- **Architecture:** `docs/architecture.md`
- **Specific subsystem:** `docs/subsystem-manifest.json`
- **How to build/run:** `DEVELOPMENT.md`
- **License & commercial:** `LICENSE.md`
- **General:** contact@cognitive-industries.org

---

## Code of Conduct

We're welcoming and respectful. Treat others like they're smart and competent (they are). Disagree with code, not people. No harassment, discrimination, or trolling.

If someone's being jerky, we'll ask them to stop. If they don't, they're out.

---

## Recognition

Contributors are listed in:
- Git commit history (your name & email)
- `CONTRIBUTORS.md` (major contributors)
- Release notes (significant features)

We value your work. Thanks for being here.

---

**Thanks for contributing to AetherOS.**

Questions? contact@cognitive-industries.org

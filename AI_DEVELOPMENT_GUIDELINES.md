# AI Development Guidelines for DominionOS

This is a quick reference for AI assistants (LLMs) helping develop DominionOS. **Specs are ground truth. Code follows specs.**

---

## The Rule

**Specs First. Code Second. Tests Always.**

Before writing code:
1. **Check if a spec exists.** See `docs/subsystem-manifest.json`
2. **If it's a major feature:** Research → Spec changes → Code → Tests → Benchmarks
3. **If it's a minor change:** Spec update (if needed) → Code → Tests → Benchmarks
4. **Always test and benchmark.** No regressions. No exceptions.

---

## Workflow by Change Type

### Major Feature (New subsystem, significant architecture change, new protocol)

1. **Research**
   - Read existing specs (architecture.md, subsystem specs)
   - Review related source code
   - Check academic references or RFCs
   - Document findings and approach

2. **Modify Spec**
   - Open `docs/subsystem-manifest.json`
   - Find or create the relevant subsystem entry
   - Update `spec_files` array with new or modified spec files
   - Add SPECIFIED section (what *should* exist)
   - Add IMPLEMENTED section (what actually exists)
   - List GAPS and PRIORITY
   - Document design decisions and trade-offs

3. **Write Tests First**
   - Before touching production code, write tests
   - Define success criteria: "Test X passes" means feature works
   - Tests should verify against the spec, not the implementation

4. **Implement Against Spec**
   - Code should match the spec
   - If code diverges from spec, update the spec (don't hide it)
   - Comment any deviations and explain why

5. **Run Full Test Suite**
   - `cargo test --release`
   - All existing tests must pass
   - New tests must pass
   - No warnings

6. **Run Benchmarks**
   - `cargo run --release --bin bench`
   - Compare before/after
   - Report results in PR
   - If performance regresses, document why it's acceptable (or fix it)

7. **Submit for Review**
   - Open a PR with spec changes + code + tests
   - Explain: what, why, how, trade-offs
   - Link relevant issues

---

### Minor Change (Bug fix, small optimization, documentation update)

1. **Check Spec (2 min)**
   - Is there a spec for this subsystem? Yes → does spec need updating?
   - Update spec if behavior changes
   - If no spec, no spec update needed

2. **Write Test (5 min)**
   - Especially for bug fixes: test that fails before fix, passes after
   - For optimization: benchmark before/after

3. **Implement (15 min)**
   - Keep it focused
   - One change per PR

4. **Test (5 min)**
   - `cargo test --release`
   - `cargo run --release --bin bench`

5. **Submit**
   - Small PR, clear message, done

**Time budget for minor:** 30-45 minutes total.

---

## When Specs Conflict with Code

**Specs win.** If code does something the spec doesn't describe:

### Option A: Update the Spec
If the code is correct and the spec is incomplete:
```
Update `docs/subsystem-manifest.json` for the subsystem.
Modify IMPLEMENTED section to match code.
Document any deviations in GAPS.
Submit with PR.
```

### Option B: Fix the Code
If the spec is right and the code is wrong:
```
Modify the code to match the spec.
Add/fix tests to prevent regression.
Submit with PR.
Document the bug fix.
```

**Never:** Leave specs and code silently out of sync. Always make them consistent.

---

## Code Quality Checklist

Before submitting any PR:

- [ ] Code compiles with zero warnings: `cargo build --release 2>&1 | grep -i warn`
- [ ] All tests pass: `cargo test --release`
- [ ] Benchmarks show no regression: `cargo run --release --bin bench`
- [ ] Unsafe code is justified: comments explain why
- [ ] Public APIs have doc comments
- [ ] Specs are consistent with code
- [ ] New tests added for new functionality
- [ ] Commit messages are clear
- [ ] PR describes what, why, and how

---

## Spec Structure (template)

Use this structure when creating or updating a subsystem spec:

```markdown
# [Subsystem Name] Specification

## Specified (What should exist)
- Capability system: ... (from archived spec or RFC)
- Contracts: ... (what this subsystem promises)
- Behaviors: ... (inputs, outputs, invariants)

## Implemented (What actually exists)
- ✅ Feature X: working, tested, production-ready
- ⚠️ Feature Y: partial implementation, known limitations
- ❌ Feature Z: not implemented yet

## Gaps (Differences)
| Gap | Severity | Why | Impact |
|-----|----------|-----|--------|
| Feature Z missing | MEDIUM | Deferred to phase 2 | Users can't do X |

## Priority
1. Gap A (CRITICAL)
2. Gap B (HIGH)
3. Gap C (MEDIUM)

## Tests
- Test X: (what passes)
- Test Y: (what should pass but doesn't)

## Notes
- Design trade-offs: (why we chose A over B)
- Known limitations: (what we're accepting for now)
- Dependencies: (other subsystems this relies on)
```

---

## Testing Strategy

### Unit Tests (In-module)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_cannot_escalate() {
        // Test: derived cap has lower rights than parent
        let parent = Capability::mint(...);
        let child = parent.derive(...);
        assert!(parent.rights() > child.rights());
    }
}
```

### Integration Tests
```bash
# Run the booted OS
./run.ps1

# In the shell
test           # Run integration test suite
```

### Benchmarks
```bash
# Run performance benchmarks
cargo run --release --bin bench

# Should compare: before change, after change
# Report: latency, throughput, memory, regression %
```

---

## Documentation Requirements

### In Code
- **Public functions:** Doc comments explaining what, inputs, outputs, panics
- **Unsafe blocks:** Comment explaining why unsafe is necessary and what invariants hold
- **Complex logic:** Inline comments for non-obvious algorithms

### In Specs
- **Contracts:** What does this subsystem promise?
- **Invariants:** What must always be true?
- **Gotchas:** What surprised past developers?

### In PRs
- **What:** What does this change do?
- **Why:** Why is this necessary?
- **How:** How does it work?
- **Trade-offs:** What did we sacrifice, and why?
- **Benchmarks:** Did performance change?

---

## LLM Recommendations

### Model Choice
- **Recommended:** Claude Opus 4.8 or newer
- **Acceptable:** Opus 4.7, Claude 3 Sonnet
- **Not recommended:** GPT-4, Llama, Gemini
- **Why:** Opus understands Rust, capability systems, and deterministic semantics better

### Prompting Style
When asking an LLM to help:
1. **Provide context:** Link to relevant spec and source files
2. **Be specific:** "Implement this function to match this spec" not "improve the code"
3. **Require specs-first approach:** "First, update the spec. Then implement."
4. **Ask for tests:** "Write tests first, then implementation"
5. **Request benchmarking:** "Ensure no performance regression"

Example:
```
Read docs/architecture.md and docs/subsystem-manifest.json
Look at firewall.rs (capability enforcement)
Task: Implement rate-limiting for capabilities

First: Update the spec section in subsystem-manifest.json
- What is rate-limiting?
- What does the code need to enforce?
- What tests verify correctness?

Then: Write unit tests for RateLimit struct
Then: Implement RateLimit in firewall.rs
Then: Run benchmarks to ensure <1% regression

Use Claude Opus 4.8 for code generation.
```

---

## Handling Disagreements

If the AI suggests code that conflicts with the spec:
1. **Don't just accept it.** Question why.
2. **Update the spec first.** If the spec is wrong, fix it.
3. **Then implement.** Follow the (corrected) spec.
4. **Document the decision.** Why did we change the spec?

---

## Common Mistakes to Avoid

- ❌ Writing code without a spec. (Specs first!)
- ❌ Ignoring failing tests. (Fix them, don't skip them.)
- ❌ Performance regression with no justification. (Benchmark or revert.)
- ❌ Unsafe code without comments. (Document why.)
- ❌ PRs that mix unrelated changes. (One feature per PR.)
- ❌ Specs that drift from code. (Keep them in sync.)
- ❌ Using non-Opus LLMs for significant code. (Quality degrades.)

---

## Quick Reference

**Spec location:** `docs/subsystem-manifest.json`  
**Test command:** `cargo test --release`  
**Benchmark command:** `cargo run --release --bin bench`  
**Boot and verify:** `./run.ps1`  
**Arch reference:** `docs/architecture.md`  

---

## Questions?

- **Spec syntax:** Check existing entries in `subsystem-manifest.json`
- **Test framework:** Rust's built-in `#[test]` and `#[cfg(test)]`
- **Benchmark framework:** See `dominion-core/benches/`
- **General:** contact@cognitive-industries.org

---

**Specs are ground truth. Code implements specs. Tests verify compliance.**

Good luck!

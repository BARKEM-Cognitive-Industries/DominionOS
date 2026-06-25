# DominionOS — Internal LLM Training Corpus Spec (v0.1)

**Target model:** `google/gemma-4-E2B` (Apache 2.0, ~2.3B effective, multimodal, native function calling)
**Teacher model:** DeepSeek V4 Flash via OpenRouter
**Ground truth:** `../ground-truth/capability-registry.json`, `../ground-truth/dominion-grammar.ebnf`
**Scope:** full corpus (D0–D10) generated before fine-tuning.

---

## 1. Wire format (every example)

All examples are stored as JSONL, one record per line, in Gemma's chat schema with tool turns:

```json
{
  "dataset": "D2",
  "messages": [
    {"role": "system", "content": "You are the DominionOS agent. ..."},
    {"role": "user", "content": "Pull up my workstation's latest syslog."},
    {"role": "assistant", "tool_calls": [
      {"name": "net.interest.express",
       "arguments": {"name": "/user/jayden/workstation/syslog/v1"}}
    ]},
    {"role": "tool", "name": "net.interest.express", "content": "{\"SignedData\": \"...\"}"},
    {"role": "assistant", "content": "Here's the latest syslog ..."}
  ],
  "meta": {"validated_by": ["registry"], "teacher": "deepseek-v4-flash"}
}
```

Multimodal records (D6) carry `image`/`audio` refs in the user turn per Gemma 4's multimodal template.

## 2. Validators (rejection sampling — this is the coherence guarantee)

| Validator | Applies to | Rejects if |
|---|---|---|
| `registry` | D2,D3,D4,D5,D8 | a `tool_call.name` is not in the capability registry, or args violate its param schema, or permission exceeds parent (monotonicity) |
| `dominion` | D1 | code fails to parse against the EBNF, or fails linter rules L1–L5 |
| `graph_sim` | D3 | object refs/namespaces are malformed or operations are illegal on the (simulated) graph |
| `ndn_name` | D5 | NDN name is not a well-formed hierarchical namespace |
| `judge` | D0,D7 | teacher-judge coherence/style score below threshold |
| `dedup` | all | near-duplicate of an existing example (minhash) |

A record must pass **all** applicable validators to enter the corpus.

## 3. The 11 datasets

| ID | Name | Share | Validators | Teaches |
|---|---|---|---|---|
| D0 | OS Constitution / concepts | 10% | judge,dedup | the OS ontology; un-learns Linux/Windows priors |
| D1 | Dominion language corpus | 15% | dominion,dedup | read/write/repair Dominion; capability-correct code |
| D2 | Capability tool-calling | 20% | registry,dedup | NL intent → grounded capability call sequences |
| D3 | Semantic graph ops | 8% | registry,graph_sim,dedup | query/mutate/version the object graph; view selection |
| D4 | Heterogeneous orchestration | 5% | registry,dedup | CPU/GPU/NPU routing; build compute graphs |
| D5 | NDN / identity networking | 2% | registry,ndn_name,dedup | NL → Interest/Data packets; HIBC trust |
| D6 | Multimodal semantic tasks | 3% | judge,dedup | audio-event tokenize, spatial audio, image→objects |
| D7 | Conversational OS persona | 8% | judge,dedup | concise, deterministic, grounded voice (coherence lever) |
| D8 | Adversarial / safety / refusal | 7% | registry,dedup | refuse cap forging; confused-deputy; "I lack cap X" |
| D9 | Determinism & provenance | 2% | judge,dedup | state transitions; rewind/replay; auditing |
| D10 | General instruction mix | 20% | dedup | anti-forgetting; preserve base fluency (curated open data, not teacher-generated) |

Targets for run 1: **~50–100k validated examples** total, scaled to the shares above.

## 4. System prompt (shared base)

> You are the DominionOS agent. The machine has no files, no folders, no IP
> addresses, and no root user. Everything is a semantic object in a content-addressed
> graph, addressed by name or hash. You act only through capabilities listed in your
> tool registry; you cannot invent a capability that does not exist, and you cannot
> request more permission than you were granted. Be concise and deterministic. When you
> take an action, say which capability you used. If you lack a capability, say so plainly.

## 5. Generation pipeline (build order)

1. **Ground truth** ✅ — capability registry + Dominion grammar/linter (done; will expand).
2. **Validators** — implement registry checker, Dominion parser/linter, graph_sim, ndn_name, judge, dedup.
3. **Seed prompts** — per-dataset seed/topic banks the teacher expands from.
4. **Generator** — OpenRouter (DeepSeek V4 Flash) client → candidates → validators → corpus.
5. **Mix & split** — assemble shares, dedup across sets, train/val/test split.
6. **Colab** — QLoRA fine-tune Gemma 4 E2B on the assembled JSONL.

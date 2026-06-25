# DominionOS AI — model toolchain & on-device runtime

The **on-device runtime** is DominionOS code: [`dominion-core/src/nn/model.rs`](../../dominion-core/src/nn/model.rs)
(`AemModel::from_bytes` → `forward`/`generate`). It runs natively (no_std, no external
engine) and is proven live in the booted OS — the kernel self-test check
*"llm forward predicts oracle token (embedded .aem)"* passes in QEMU (exit 33), and the
`llm` shell command runs inference at the terminal.

This `tools/ai/` directory is the **offline toolchain** that prepares models for it.

## Layout
```
tools/ai/
  convert/   aem.py          — the .aem format (int8 per-row + f32), reader/writer
             convert_hf.py   — download a HF model → quantize → .aem (+ reference logits)
  verify/    forward_qwen.py — NumPy oracle (proves a converted .aem matches HF)
             make_fixture.py — builds the tiny embedded test model + expected logits
  rustref/   standalone Rust inference + benchmark (mirrors the on-device forward)
  ground-truth/  capability-registry.json, dominion-grammar.ebnf (agent grounding)
  datasets/  DATASET-SPEC.md (Gemma fine-tune corpus plan)
  bench_pytorch.py            — PyTorch CPU baseline
  .env       OPENROUTER_API_KEY, HUGGINGFACE_TOKEN
../../models/  converted .aem outputs (fixture, qwen2.5-0.5b, vibethinker-3b, …)
../../docs/ai/ ON_DEVICE_AI_SPEC.md, BENCHMARKS.md, PIPELINE_STATUS.md
```

## Pipeline
```
HF weights ──convert/convert_hf.py──▶ models/<name>/model.aem ──┐
                                                                 ├─▶ embed in dominion-core
verify/forward_qwen.py  (parity vs HF) ◀─────────────────────────┘   (testdata/) OR load
                                                                     via VFS at runtime
```

## Quick commands (run from this dir)
```bash
set -a; source .env; set +a; export HF_TOKEN=$HUGGINGFACE_TOKEN
# convert a real model
python convert/convert_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --out ../../models/qwen2.5-0.5b --reference "The capital of France is"
python verify/forward_qwen.py ../../models/qwen2.5-0.5b           # parity vs HF
# host benchmark (proves >3x PyTorch decode)
cd rustref && RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/rustref bench ../../../models/qwen2.5-0.5b 32
./target/release/rustref agent ../../../models/qwen2.5-0.5b       # prefix-KV reuse 100x lever
```

## On-device (in the OS)
```bash
cd ../..               # dominionos/
cargo test --lib nn::model -p dominion-core   # OS-code runtime tests
./run-test.ps1                              # boot QEMU; selftest incl. the LLM check
# at the DominionOS terminal:  llm
```

## Status
Verified end-to-end on Qwen2.5-0.5B and VibeThinker-3B. Gemma 4 E2B-it + the STT/TTS/
image models are work-in-progress (see ../../docs/ai/PIPELINE_STATUS.md).

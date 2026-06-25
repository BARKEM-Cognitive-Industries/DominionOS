"""
convert_hf.py — download a real HuggingFace model and convert it to DominionOS `.aem`.

Validated first on Qwen2.5-0.5B (same architecture family as the VibeThinker-3B
target), so the conversion + the native `nn/model.rs` forward generalize straight to
the full fleet. Also dumps reference logits (via transformers) so the Rust forward can
be parity-checked bit-for-(near)bit.

Usage:
    python convert_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --out ../models/qwen2.5-0.5b
    python convert_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --out ... --reference "Hello"

Gated models (e.g. google/gemma-4-E2B) need `--hf-token <token>` or HF_TOKEN env.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from aem import AemWriter

# Architecture-specific config extraction. Each maps an HF config.json to the compact
# hyperparameter set `nn/model.rs` needs. Add a family here to support a new model.
QWEN_LIKE = {"qwen2", "qwen3", "llama", "mistral"}  # RoPE + RMSNorm + GQA + SwiGLU
GEMMA_LIKE = {"gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text"}


def extract_llm_config(cfg: dict) -> dict:
    # Gemma wraps the LLM hyperparameters in `text_config`.
    tc = cfg.get("text_config", cfg)
    mt = cfg.get("model_type", "")
    is_gemma = mt.startswith("gemma") or tc.get("model_type", "").startswith("gemma")
    hidden = tc["hidden_size"]
    n_heads = tc["num_attention_heads"]
    head_dim = tc.get("head_dim", hidden // n_heads)
    import math
    out = {
        "family": "gemma" if is_gemma else "qwen_like",
        "model_type": mt,
        "vocab_size": tc["vocab_size"],
        "hidden_size": hidden,
        "n_layers": tc["num_hidden_layers"],
        "n_heads": n_heads,
        "n_kv_heads": tc.get("num_key_value_heads", n_heads),
        "head_dim": head_dim,
        "intermediate_size": tc["intermediate_size"],
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta": float(tc.get("rope_theta", 10000.0)),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)),
        "norm_offset": 1.0 if is_gemma else 0.0,
        "ffn_act": "gelu_tanh" if is_gemma else "silu",
    }
    if is_gemma:
        # Gemma-specific forward knobs the on-device runtime must honour.
        out.update({
            "embed_scale": math.sqrt(hidden),                  # Gemma scales input embeddings
            "sliding_window": tc.get("sliding_window", 0),
            "rope_local_base_freq": float(tc.get("rope_local_base_freq", 10000.0)),
            "query_pre_attn_scalar": tc.get("query_pre_attn_scalar", head_dim),
            "final_logit_softcapping": tc.get("final_logit_softcapping", 0.0) or 0.0,
            "attn_logit_softcapping": tc.get("attn_logit_softcapping", 0.0) or 0.0,
            "num_kv_shared_layers": tc.get("num_kv_shared_layers", 0),
            "layer_types": tc.get("layer_types", []),           # per-layer local/global
            "sliding_window_pattern": tc.get("sliding_window_pattern", 0),
        })
    return out


# Tensors kept full-precision (small, precision-sensitive): all norms + biases.
def keep_f32(name: str) -> bool:
    n = name.lower()
    return ("norm" in n) or n.endswith(".bias") or ("layernorm" in n)


def convert(model_id: str, out_dir: str, reference: str | None, hf_token: str | None) -> None:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] downloading {model_id} …", flush=True)
    local = snapshot_download(
        model_id,
        token=hf_token,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
    )
    local = Path(local)

    cfg = json.loads((local / "config.json").read_text())
    mt = cfg.get("model_type", "")
    if mt not in QWEN_LIKE and mt not in GEMMA_LIKE:
        print(f"[warn] model_type='{mt}' unrecognised; converting as qwen_like")
    mc = extract_llm_config(cfg)
    print(f"[2/5] arch: {json.dumps(mc)}", flush=True)

    # Gather all safetensors shards.
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        sys.exit("no .safetensors found")

    writer = AemWriter(arch=mt or "qwen_like", config=mc, tokenizer="tokenizer.json")
    n_q8 = n_f32 = 0
    print(f"[3/5] quantizing {len(shards)} shard(s) …", flush=True)
    import torch  # weights are commonly bf16, which NumPy cannot read directly
    for shard in shards:
        # framework="pt" handles bf16/fp16; upcast to f32 on the way out.
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                arr = f.get_tensor(name).to(torch.float32).numpy()
                if keep_f32(name) or arr.ndim == 1:
                    writer.add_f32(name, arr)
                    n_f32 += 1
                else:
                    writer.add_q8(name, arr)
                    n_q8 += 1

    aem_path = out / "model.aem"
    hdr = writer.write(aem_path)
    size_mb = aem_path.stat().st_size / 1e6
    print(f"[4/5] wrote {aem_path} ({size_mb:.1f} MB) — {n_q8} q8 + {n_f32} f32 tensors", flush=True)

    # Copy tokenizer assets.
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = local / fn
        if src.exists():
            shutil.copy(src, out / fn)

    # Optional reference logits for Rust parity.
    if reference is not None:
        print("[5/5] dumping reference logits via transformers …", flush=True)
        dump_reference(model_id, local, reference, out, hf_token)
    else:
        print("[5/5] skipped reference dump (pass --reference TEXT to enable)", flush=True)

    print(f"DONE: {model_id} -> {aem_path}")


def dump_reference(model_id: str, local: Path, text: str, out: Path, hf_token: str | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(local))
    model = AutoModelForCausalLM.from_pretrained(str(local), torch_dtype=torch.float32, token=hf_token)
    model.eval()
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0]  # [seq, vocab]
    last = logits[-1]
    topk = torch.topk(last, 10)
    ref = {
        "prompt": text,
        "input_ids": ids[0].tolist(),
        "top10_token_ids": topk.indices.tolist(),
        "top10_logits": [round(v, 5) for v in topk.values.tolist()],
        "argmax_id": int(last.argmax()),
        "argmax_decoded": tok.decode([int(last.argmax())]),
    }
    (out / "reference.json").write_text(json.dumps(ref, indent=2))
    print(f"      reference argmax next token: {ref['argmax_id']} = {ref['argmax_decoded']!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="prompt text to dump reference logits for")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()
    convert(args.model, args.out, args.reference, args.hf_token)


if __name__ == "__main__":
    main()

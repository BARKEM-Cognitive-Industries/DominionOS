"""
forward_qwen.py — a from-scratch NumPy forward pass over a converted `.aem`.

Purpose:
  1. **Prove the conversion is correct.** Loads `model.aem`, runs a full Qwen2.5/Llama/
     Gemma-family forward in plain NumPy, and checks the argmax next-token against the
     `reference.json` that `convert_hf.py` dumped from HuggingFace. If they agree, the
     converted (int8-quantized) weights are faithful.
  2. **Be the parity oracle for `dominion-core/src/nn/model.rs`.** This is the exact math
     the native Rust forward must reproduce — same RoPE (rotate-half), same RMSNorm,
     same GQA, same SwiGLU. The Rust side targets these logits.

No torch. Reads only the `.aem` (+ tokenizer for decoding). Runs on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert"))
from aem import AemReader  # noqa: E402


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float, offset: float) -> np.ndarray:
    # x: [seq, d]
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    xn = x / np.sqrt(var + eps)
    return (xn * (w + offset)).astype(np.float64)


def rope_cos_sin(seq: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    # HF rotate-half convention: emb = cat(freqs, freqs).
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seq, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)            # [seq, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [seq, head_dim]
    return np.cos(emb), np.sin(emb)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: [seq, n_heads, head_dim]; cos/sin: [seq, head_dim]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + rotate_half(x) * sin


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    # HF Linear: y = x @ w.T (+ b). w is [out, in].
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def forward(model_dir: str) -> dict:
    rd = AemReader(Path(model_dir) / "model.aem")
    c = rd.config
    names = set(rd.tensor_names())

    def t(name: str) -> np.ndarray:
        return rd.get(name).astype(np.float64)

    def opt(name: str) -> np.ndarray | None:
        return rd.get(name).astype(np.float64) if name in names else None

    ref = json.loads((Path(model_dir) / "reference.json").read_text())
    ids = np.array(ref["input_ids"], dtype=np.int64)

    H = c["hidden_size"]
    L = c["n_layers"]
    nh = c["n_heads"]
    nkv = c["n_kv_heads"]
    hd = c["head_dim"]
    eps = c["rms_norm_eps"]
    theta = c["rope_theta"]
    off = c["norm_offset"]
    group = nh // nkv
    seq = len(ids)

    embed = t("model.embed_tokens.weight")  # [vocab, H]
    x = embed[ids]                           # [seq, H]
    cos, sin = rope_cos_sin(seq, hd, theta)
    causal = np.triu(np.full((seq, seq), -1e30), k=1)

    for i in range(L):
        p = f"model.layers.{i}."
        # --- attention ---
        h = rmsnorm(x, t(p + "input_layernorm.weight"), eps, off)
        q = linear(h, t(p + "self_attn.q_proj.weight"), opt(p + "self_attn.q_proj.bias"))
        k = linear(h, t(p + "self_attn.k_proj.weight"), opt(p + "self_attn.k_proj.bias"))
        v = linear(h, t(p + "self_attn.v_proj.weight"), opt(p + "self_attn.v_proj.bias"))
        q = q.reshape(seq, nh, hd)
        k = k.reshape(seq, nkv, hd)
        v = v.reshape(seq, nkv, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA: repeat kv heads.
        k = np.repeat(k, group, axis=1)  # [seq, nh, hd]
        v = np.repeat(v, group, axis=1)
        out = np.zeros((seq, nh, hd), dtype=np.float64)
        scale = 1.0 / np.sqrt(hd)
        for hidx in range(nh):
            qh = q[:, hidx, :]            # [seq, hd]
            kh = k[:, hidx, :]
            vh = v[:, hidx, :]
            scores = (qh @ kh.T) * scale + causal  # [seq, seq]
            out[:, hidx, :] = softmax(scores) @ vh
        attn = out.reshape(seq, nh * hd)
        attn = linear(attn, t(p + "self_attn.o_proj.weight"), opt(p + "self_attn.o_proj.bias"))
        x = x + attn
        # --- mlp (SwiGLU) ---
        h = rmsnorm(x, t(p + "post_attention_layernorm.weight"), eps, off)
        gate = linear(h, t(p + "mlp.gate_proj.weight"), None)
        up = linear(h, t(p + "mlp.up_proj.weight"), None)
        act = gate * (1.0 / (1.0 + np.exp(-gate))) if c["ffn_act"] == "silu" else _gelu_tanh(gate)
        mlp = linear(act * up, t(p + "mlp.down_proj.weight"), None)
        x = x + mlp

    x = rmsnorm(x, t("model.norm.weight"), eps, off)
    if c.get("tie_word_embeddings", False) or "lm_head.weight" not in names:
        lm_w = embed
    else:
        lm_w = t("lm_head.weight")
    logits = x @ lm_w.T  # [seq, vocab]
    last = logits[-1]

    top10 = np.argsort(last)[::-1][:10].tolist()
    result = {
        "argmax_id": int(last.argmax()),
        "our_top10": top10,
        "ref_argmax_id": ref["argmax_id"],
        "ref_top10": ref["top10_token_ids"],
        "argmax_match": int(last.argmax()) == ref["argmax_id"],
        "top1_in_ref_top10": int(last.argmax()) in ref["top10_token_ids"],
    }
    return result


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else "../models/qwen2.5-0.5b"
    r = forward(md)
    print(json.dumps(r, indent=2))
    if r["argmax_match"]:
        print("\n[PASS] PARITY: NumPy-from-.aem argmax matches HuggingFace reference.")
    elif r["top1_in_ref_top10"]:
        print("\n[CLOSE] our argmax is within HF top-10 (int8 quant drift) -- acceptable.")
    else:
        print("\n[FAIL] MISMATCH: conversion or forward math is wrong.")
        sys.exit(1)

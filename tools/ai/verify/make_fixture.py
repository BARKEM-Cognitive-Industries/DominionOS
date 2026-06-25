"""
make_fixture.py — build a TINY Qwen2-family `.aem` + exact expected logits.

The real converted model (496 MB) is too big for a Rust unit test. This generates a
small (~few-hundred-KB) random-but-deterministic model with the *same architecture and
tensor names* as Qwen2.5/VibeThinker, stored in **f32** (so parity is exact, with no
int8 quant noise), plus the full last-row logits computed by the proven NumPy forward.

`dominion-core`'s native `nn/model.rs` loads `fixture.aem` and must reproduce
`fixture_expected.json`'s logits to ~1e-4. That makes on-device inference correctness a
fast, dependency-free unit test — the parity target for the Rust forward.

Output: ai/models/fixture/{fixture.aem, fixture_expected.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert"))
from aem import AemReader, AemWriter  # noqa: E402

# Tiny but structurally complete Qwen2 config (GQA: 4 q-heads, 2 kv-heads).
CFG = {
    "family": "qwen_like",
    "model_type": "qwen2",
    "vocab_size": 48,
    "hidden_size": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "norm_offset": 0.0,
    "ffn_act": "silu",
}


def build(out_dir: str) -> None:
    rng = np.random.RandomState(1234)
    H, L = CFG["hidden_size"], CFG["n_layers"]
    nh, nkv, hd = CFG["n_heads"], CFG["n_kv_heads"], CFG["head_dim"]
    inter, vocab = CFG["intermediate_size"], CFG["vocab_size"]
    qd, kvd = nh * hd, nkv * hd

    def r(*shape):
        # Small magnitudes keep logits in a sane range.
        return (rng.randn(*shape) * 0.08).astype(np.float32)

    w = AemWriter(arch="qwen2", config=CFG, tokenizer="none")
    # Embedding (tied to lm_head).
    w.add_f32("model.embed_tokens.weight", r(vocab, H))
    for i in range(L):
        p = f"model.layers.{i}."
        w.add_f32(p + "input_layernorm.weight", (rng.rand(H).astype(np.float32) * 0.5 + 0.75))
        w.add_f32(p + "self_attn.q_proj.weight", r(qd, H))
        w.add_f32(p + "self_attn.q_proj.bias", r(qd))
        w.add_f32(p + "self_attn.k_proj.weight", r(kvd, H))
        w.add_f32(p + "self_attn.k_proj.bias", r(kvd))
        w.add_f32(p + "self_attn.v_proj.weight", r(kvd, H))
        w.add_f32(p + "self_attn.v_proj.bias", r(kvd))
        w.add_f32(p + "self_attn.o_proj.weight", r(H, qd))
        w.add_f32(p + "post_attention_layernorm.weight", (rng.rand(H).astype(np.float32) * 0.5 + 0.75))
        w.add_f32(p + "mlp.gate_proj.weight", r(inter, H))
        w.add_f32(p + "mlp.up_proj.weight", r(inter, H))
        w.add_f32(p + "mlp.down_proj.weight", r(H, inter))
    w.add_f32("model.norm.weight", (rng.rand(H).astype(np.float32) * 0.5 + 0.75))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w.write(out / "fixture.aem")

    # Exact NumPy forward (f64) for a fixed token sequence → expected logits.
    ids = [5, 17, 3, 42, 9, 28]
    logits = numpy_forward(out / "fixture.aem", ids)
    expected = {
        "config": CFG,
        "input_ids": ids,
        "logits_last": [round(float(x), 6) for x in logits.tolist()],
        "argmax_id": int(logits.argmax()),
    }
    (out / "fixture_expected.json").write_text(json.dumps(expected, indent=2))
    print(f"wrote {out/'fixture.aem'} and fixture_expected.json")
    print(f"argmax next id = {expected['argmax_id']}, logit = {max(expected['logits_last']):.4f}")


def numpy_forward(aem_path, ids) -> np.ndarray:
    """Compact exact forward (f64) — same math as forward_qwen.py, returns last logits."""
    rd = AemReader(aem_path)
    c = rd.config
    g = lambda n: rd.get(n).astype(np.float64)
    names = set(rd.tensor_names())
    H, L = c["hidden_size"], c["n_layers"]
    nh, nkv, hd = c["n_heads"], c["n_kv_heads"], c["head_dim"]
    eps, theta, off = c["rms_norm_eps"], c["rope_theta"], c["norm_offset"]
    group = nh // nkv
    ids = np.array(ids)
    seq = len(ids)

    def rms(x, wname):
        wt = g(wname)
        var = np.mean(x**2, axis=-1, keepdims=True)
        return (x / np.sqrt(var + eps)) * (wt + off)

    inv = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    fr = np.outer(np.arange(seq), inv)
    emb = np.concatenate([fr, fr], -1)
    cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]

    def roth(x):
        h = x.shape[-1] // 2
        return np.concatenate([-x[..., h:], x[..., :h]], -1)

    embed = g("model.embed_tokens.weight")
    x = embed[ids]
    mask = np.triu(np.full((seq, seq), -1e30), 1)
    for i in range(L):
        p = f"model.layers.{i}."
        h = rms(x, p + "input_layernorm.weight")
        q = (h @ g(p + "self_attn.q_proj.weight").T + g(p + "self_attn.q_proj.bias")).reshape(seq, nh, hd)
        k = (h @ g(p + "self_attn.k_proj.weight").T + g(p + "self_attn.k_proj.bias")).reshape(seq, nkv, hd)
        v = (h @ g(p + "self_attn.v_proj.weight").T + g(p + "self_attn.v_proj.bias")).reshape(seq, nkv, hd)
        q = q * cos + roth(q) * sin
        k = k * cos + roth(k) * sin
        k = np.repeat(k, group, 1)
        v = np.repeat(v, group, 1)
        o = np.zeros((seq, nh, hd))
        sc = 1.0 / np.sqrt(hd)
        for hh in range(nh):
            s = (q[:, hh] @ k[:, hh].T) * sc + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o[:, hh] = s @ v[:, hh]
        x = x + o.reshape(seq, nh * hd) @ g(p + "self_attn.o_proj.weight").T
        h = rms(x, p + "post_attention_layernorm.weight")
        gate = h @ g(p + "mlp.gate_proj.weight").T
        up = h @ g(p + "mlp.up_proj.weight").T
        act = gate * (1.0 / (1.0 + np.exp(-gate)))
        x = x + (act * up) @ g(p + "mlp.down_proj.weight").T
    x = rms(x, "model.norm.weight")
    lm = embed if (c.get("tie_word_embeddings") or "lm_head.weight" not in names) else g("lm_head.weight")
    return (x @ lm.T)[-1]


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "../models/fixture")

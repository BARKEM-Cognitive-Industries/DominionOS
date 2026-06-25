"""
AEM — the DominionOS on-device model format (writer + reader).

A `.aem` file is what ships inside DominionOS and is loaded natively by
`dominion-core/src/nn/model.rs`. It is deliberately simple, self-describing, and
content-verifiable — a quantization-aware cousin of safetensors:

    bytes   field
    4       magic            b"AEM1"
    4       header_len (u32 LE)
    H       header           UTF-8 JSON (see below), H = header_len
    ...     data blob        concatenated tensor payloads, in `tensors` order

Header JSON:
    {
      "arch":   "qwen2" | "gemma3" | "whisper" | "kokoro" | "dreamlite",
      "config": { ...architecture hyperparameters... },
      "tensors": [
        { "name": "model.layers.0.self_attn.q_proj.weight",
          "dtype": "q8" | "f32",
          "shape": [out, in],
          "offset": <byte offset into data blob>,
          "nbytes": <int8/f32 payload length>,
          "scale_offset": <offset of per-row f32 scales, q8 only>,
          "scale_rows": <number of f32 scales, q8 only> },
        ...
      ],
      "tokenizer": "tokenizer.json",   # sidecar shipped next to the .aem
      "sha256": "<hex of the data blob>"
    }

Quantization (`q8`): per-output-row symmetric int8. For a weight row r,
`scale_r = max(|row|)/127`, `q_i = round(x_i / scale_r)` clamped to [-127,127];
dequant is `x_i ≈ q_i * scale_r`. This matches `dominion-core::ml::QTensor`
(int8 values + a scale), so `model.rs` reconstructs a `QTensor` directly and the
existing `qmatmul` kernel (the NPU low-precision path, ~4× lever) runs unchanged.

Pure-stdlib + numpy. No torch needed to *read* an .aem.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"AEM1"


@dataclass
class TensorEntry:
    name: str
    dtype: str            # "q8" | "f32"
    shape: list[int]
    payload: bytes        # int8 or f32 bytes
    scales: bytes | None  # f32 per-row scales (q8 only)


def quantize_q8_per_row(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-row symmetric int8. `w` is [out, in] (HF Linear layout).

    Returns (int8 values [out,in], f32 scales [out]). Rows that are all-zero get
    scale 1.0 to avoid division by zero (their quantized values are all 0).
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    absmax = np.max(np.abs(w), axis=1)
    scales = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.round(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales


@dataclass
class AemWriter:
    arch: str
    config: dict[str, Any]
    tokenizer: str = "tokenizer.json"
    _entries: list[TensorEntry] = field(default_factory=list)

    def add_f32(self, name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._entries.append(
            TensorEntry(name, "f32", list(arr.shape), arr.tobytes(), None)
        )

    def add_q8(self, name: str, arr: np.ndarray) -> None:
        """Quantize a 2-D (or 1-D) weight to per-row int8 and record it."""
        q, scales = quantize_q8_per_row(arr)
        self._entries.append(
            TensorEntry(name, "q8", list(arr.shape), q.tobytes(), scales.tobytes())
        )

    def write(self, path: str | Path) -> dict[str, Any]:
        """Serialize to `path`. Returns the header dict (for logging/tests)."""
        blob = bytearray()
        tmeta: list[dict[str, Any]] = []
        for e in self._entries:
            offset = len(blob)
            blob.extend(e.payload)
            meta = {
                "name": e.name,
                "dtype": e.dtype,
                "shape": e.shape,
                "offset": offset,
                "nbytes": len(e.payload),
            }
            if e.dtype == "q8":
                assert e.scales is not None
                meta["scale_offset"] = len(blob)
                meta["scale_rows"] = len(e.scales) // 4
                blob.extend(e.scales)
            tmeta.append(meta)

        digest = hashlib.sha256(bytes(blob)).hexdigest()
        header = {
            "arch": self.arch,
            "config": self.config,
            "tensors": tmeta,
            "tokenizer": self.tokenizer,
            "sha256": digest,
        }
        hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(hbytes)))
            f.write(hbytes)
            f.write(bytes(blob))
        return header


class AemReader:
    """Read an .aem (for round-trip tests and Python-side verification)."""

    def __init__(self, path: str | Path):
        with open(path, "rb") as f:
            raw = f.read()
        if raw[:4] != MAGIC:
            raise ValueError("not an AEM1 file")
        (hlen,) = struct.unpack("<I", raw[4:8])
        self.header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
        self._blob = raw[8 + hlen :]
        got = hashlib.sha256(self._blob).hexdigest()
        if got != self.header.get("sha256"):
            raise ValueError("data blob hash mismatch (corrupt .aem)")

    @property
    def arch(self) -> str:
        return self.header["arch"]

    @property
    def config(self) -> dict[str, Any]:
        return self.header["config"]

    def tensor_names(self) -> list[str]:
        return [t["name"] for t in self.header["tensors"]]

    def get(self, name: str) -> np.ndarray:
        """Dequantize (q8) or read (f32) a tensor back to float32."""
        meta = next(t for t in self.header["tensors"] if t["name"] == name)
        off, n = meta["offset"], meta["nbytes"]
        if meta["dtype"] == "f32":
            arr = np.frombuffer(self._blob[off : off + n], dtype=np.float32)
            return arr.reshape(meta["shape"]).copy()
        # q8
        q = np.frombuffer(self._blob[off : off + n], dtype=np.int8).reshape(meta["shape"])
        so = meta["scale_offset"]
        scales = np.frombuffer(self._blob[so : so + meta["scale_rows"] * 4], dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return (q.astype(np.float32) * scales[:, None]).reshape(meta["shape"]).copy()


if __name__ == "__main__":
    # Self-test: round-trip a tiny model and report quantization error.
    w = np.random.RandomState(0).randn(8, 16).astype(np.float32)
    wr = AemWriter(arch="test", config={"hidden": 16})
    wr.add_q8("w.q8", w)
    wr.add_f32("w.f32", w)
    hdr = wr.write("/tmp/_aem_selftest.aem")
    rd = AemReader("/tmp/_aem_selftest.aem")
    assert rd.tensor_names() == ["w.q8", "w.f32"]
    f32_back = rd.get("w.f32")
    q8_back = rd.get("w.q8")
    assert np.array_equal(f32_back, w), "f32 round-trip must be exact"
    rel = np.abs(q8_back - w).max() / (np.abs(w).max() + 1e-9)
    print(f"OK: f32 exact; q8 max rel error = {rel:.4f} (expect < 0.01)")
    assert rel < 0.01

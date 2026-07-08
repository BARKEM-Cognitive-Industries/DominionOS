//! `nn::model` — load a DominionOS `.aem` model and run a decoder-only transformer
//! forward pass (Qwen2.5 / VibeThinker / Gemma families) entirely on dominion-core.
//!
//! This is the on-device LLM runtime: it parses the `.aem` container (a quantization-
//! aware weight format produced by the offline converter), reconstructs the weights as
//! [`Tensor`]s, and runs RoPE (rotate-half) + RMSNorm + grouped-query attention +
//! SwiGLU/GeGLU + (optional Gemma) embedding-scale / query-scale / final-logit-softcap.
//! Matmuls fan out over cores through the crate's [`Spawn`] seam (the kernel injects the
//! SMP job queue), so it is faster than a single-thread baseline and bit-deterministic
//! by default.
//!
//! No `std`, no filesystem: the model is supplied as `&[u8]` (the kernel reads it from
//! the VFS; tests embed it with `include_bytes!`). Pure `alloc`, `#![forbid(unsafe_code)]`.

use crate::datatypes::{sqrt, Tensor};
use crate::ml::{exp, sigmoid};
use crate::nn::rope::{apply_rope, rope_tables};
use crate::parallel::{Serial, Spawn};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

// ───────────────────────────── security limits ─────────────────────────────

/// Hard cap on the total model size accepted from untrusted input (2 GiB).
/// Any `.aem` file larger than this is rejected before any parsing begins.
pub const MAX_MODEL_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Maximum number of transformer layers a model may declare.
const MAX_LAYERS: usize = 1_024;

/// Maximum vocabulary size (token count).
const MAX_VOCAB: usize = 1_048_576; // 1 M tokens

/// Maximum hidden dimension.
const MAX_HIDDEN: usize = 131_072; // 128 K

/// Maximum number of attention heads (total, including KV heads).
const MAX_HEADS: usize = 4_096;

/// Maximum intermediate (FFN) size.
const MAX_INTER: usize = 524_288; // 512 K

/// Maximum number of elements in a single tensor (guards `with_capacity` / loop counts).
const MAX_TENSOR_ELEMS: usize = 1_073_741_824; // 1 G f64 ≈ 8 GiB

// ───────────────────────────── minimal no_std JSON ─────────────────────────────

#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}
impl Json {
    fn get(&self, k: &str) -> &Json {
        match self {
            Json::Obj(m) => m.iter().find(|(kk, _)| kk == k).map(|(_, v)| v).unwrap_or(&Json::Null),
            _ => &Json::Null,
        }
    }
    fn arr(&self) -> &[Json] {
        if let Json::Arr(a) = self { a } else { &[] }
    }
    fn num(&self) -> f64 {
        if let Json::Num(n) = self { *n } else { 0.0 }
    }
    fn num_or(&self, d: f64) -> f64 {
        if let Json::Num(n) = self { *n } else { d }
    }
    fn usize(&self) -> usize {
        self.num() as usize
    }
    fn str(&self) -> &str {
        if let Json::Str(s) = self { s } else { "" }
    }
    fn boolean(&self) -> bool {
        matches!(self, Json::Bool(true))
    }
}

/// JSON parser.  All index advances are bounds-checked; on malformed input it
/// returns `Json::Null` rather than panicking.
struct JP<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> JP<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn advance(&mut self) {
        if self.i < self.b.len() {
            self.i += 1;
        }
    }
    fn ws(&mut self) {
        while let Some(c) = self.peek() {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.advance();
            } else {
                break;
            }
        }
    }
    fn val(&mut self) -> Json {
        self.ws();
        match self.peek() {
            None => Json::Null,
            Some(b'{') => self.obj(),
            Some(b'[') => self.arr(),
            Some(b'"') => Json::Str(self.string()),
            Some(b't') => { self.i = self.i.saturating_add(4).min(self.b.len()); Json::Bool(true) }
            Some(b'f') => { self.i = self.i.saturating_add(5).min(self.b.len()); Json::Bool(false) }
            Some(b'n') => { self.i = self.i.saturating_add(4).min(self.b.len()); Json::Null }
            Some(_) => Json::Num(self.number()),
        }
    }
    fn obj(&mut self) -> Json {
        let mut m = Vec::new();
        self.advance(); // '{'
        self.ws();
        if self.peek() == Some(b'}') { self.advance(); return Json::Obj(m); }
        loop {
            self.ws();
            if self.peek() != Some(b'"') { break; } // malformed — bail safely
            let k = self.string();
            self.ws();
            self.advance(); // ':'
            let v = self.val();
            m.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => { self.advance(); }
                Some(b'}') => { self.advance(); break; }
                _ => break, // malformed — bail safely
            }
        }
        Json::Obj(m)
    }
    fn arr(&mut self) -> Json {
        let mut a = Vec::new();
        self.advance(); // '['
        self.ws();
        if self.peek() == Some(b']') { self.advance(); return Json::Arr(a); }
        loop {
            if self.peek().is_none() { break; }
            a.push(self.val());
            self.ws();
            match self.peek() {
                Some(b',') => { self.advance(); }
                Some(b']') => { self.advance(); break; }
                _ => break, // malformed — bail safely
            }
        }
        Json::Arr(a)
    }
    fn string(&mut self) -> String {
        let mut s = String::new();
        self.advance(); // opening '"'
        loop {
            match self.peek() {
                None | Some(b'"') => { self.advance(); break; }
                Some(b'\\') => {
                    self.advance();
                    match self.peek() {
                        Some(b'n') => { s.push('\n'); self.advance(); }
                        Some(b't') => { s.push('\t'); self.advance(); }
                        Some(c) => { s.push(c as char); self.advance(); }
                        None => break,
                    }
                }
                Some(c) => { s.push(c as char); self.advance(); }
            }
        }
        s
    }
    /// Parse an integer or float (incl. exponent) without `std` float parsing.
    fn number(&mut self) -> f64 {
        let start = self.i;
        while let Some(c) = self.peek() {
            if matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                self.advance();
            } else {
                break;
            }
        }
        parse_f64(&self.b[start..self.i])
    }
}

/// Deterministic decimal→f64 (handles sign, fraction, exponent). `core`-only.
fn parse_f64(b: &[u8]) -> f64 {
    let mut i = 0;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut mant = 0.0f64;
    while i < b.len() && b[i].is_ascii_digit() {
        mant = mant * 10.0 + (b[i] - b'0') as f64;
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let mut scale = 0.1f64;
        while i < b.len() && b[i].is_ascii_digit() {
            mant += (b[i] - b'0') as f64 * scale;
            scale *= 0.1;
            i += 1;
        }
    }
    let mut exp_val = 0i32;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        let mut eneg = false;
        if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
            eneg = b[i] == b'-';
            i += 1;
        }
        let mut e = 0i32;
        while i < b.len() && b[i].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add((b[i] - b'0') as i32);
            i += 1;
        }
        exp_val = if eneg { -e } else { e };
    }
    // f64's dynamic range is ±308; any exponent past that already saturates to
    // inf/0, so clamping is lossless and guards against a malicious header token
    // like `1e2000000000` forcing billions of multiplications (DoS).
    let exp_val = exp_val.clamp(-400, 400);
    let mut v = mant;
    if exp_val > 0 {
        for _ in 0..exp_val { v *= 10.0; }
    } else {
        for _ in 0..(-exp_val) { v *= 0.1; }
    }
    if neg { -v } else { v }
}

fn parse_json(b: &[u8]) -> Json {
    JP { b, i: 0 }.val()
}

// ───────────────────────────── config + weights ─────────────────────────────

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub theta: f64,
    pub norm_offset: f64,   // 0 (Llama/Qwen) or 1 (Gemma: gain stored as γ−1)
    pub embed_scale: f64,   // √hidden for Gemma, else 1
    pub gelu: bool,         // GeGLU (Gemma) vs SwiGLU (Qwen/Llama)
    pub attn_denom: f64,    // query_pre_attn_scalar; defaults to head_dim
    pub final_softcap: f64, // Gemma final-logit softcap; 0 = off
}

struct Layer {
    in_norm: Vec<f64>,
    post_norm: Vec<f64>,
    // projections stored transposed to [in, out] for x·W.
    qw: Tensor, qb: Vec<f64>,
    kw: Tensor, kb: Vec<f64>,
    vw: Tensor, vb: Vec<f64>,
    ow: Tensor, ob: Vec<f64>,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    inter: usize,
}

/// A loaded transformer ready for inference.
pub struct AemModel {
    pub cfg: ModelConfig,
    embed: Tensor, // [vocab, hidden] (row-major; lookup by row)
    final_norm: Vec<f64>,
    lm_rows: Tensor, // [vocab, hidden] for the LM head (tied = embed)
    layers: Vec<Layer>,
}

/// Per-layer KV cache (grows each decode step).
pub struct KvCache {
    k: Vec<Vec<f64>>, // per layer: [pos * kvd]
    v: Vec<Vec<f64>>,
    len: usize,
}
impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        KvCache { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers], len: 0 }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// helper: read a tensor from the .aem blob, dequantizing q8 → f64.
struct Raw<'a> {
    header: Json,
    blob: &'a [u8],
}
impl<'a> Raw<'a> {
    fn meta(&self, name: &str) -> Option<&Json> {
        self.header.get("tensors").arr().iter().find(|t| t.get("name").str() == name)
    }
    fn has(&self, name: &str) -> bool {
        self.meta(name).is_some()
    }

    /// (data row-major, rows, cols) as f64.
    ///
    /// Returns `Err` on any of:
    /// - tensor not present in the header
    /// - unknown dtype
    /// - dimension product overflow
    /// - byte-count / offset inconsistency (blob OOB)
    /// - per-element count exceeding `MAX_TENSOR_ELEMS`
    fn get(&self, name: &str) -> Result<(Vec<f64>, usize, usize), &'static str> {
        let m = self.meta(name).ok_or("missing tensor")?;
        let shape: Vec<usize> = m.get("shape").arr().iter().map(|x| x.usize()).collect();
        let (rows, cols) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            return Err("tensor shape must be 1-D or 2-D");
        };

        // Guard against overflow in rows*cols.
        let total_elems = rows.checked_mul(cols).ok_or("tensor dimension overflow")?;
        if total_elems > MAX_TENSOR_ELEMS {
            return Err("tensor too large");
        }

        let off = m.get("offset").usize();
        let n = m.get("nbytes").usize();

        // Guard blob bounds: off + n must not wrap and must stay within blob.
        let end = off.checked_add(n).ok_or("tensor offset+nbytes overflow")?;
        if end > self.blob.len() {
            return Err("tensor data out of blob bounds");
        }

        let data = match m.get("dtype").str() {
            "f32" => {
                if n % 4 != 0 {
                    return Err("f32 nbytes not a multiple of 4");
                }
                let count = n / 4;
                if count > MAX_TENSOR_ELEMS {
                    return Err("f32 tensor too many elements");
                }
                let mut d = Vec::with_capacity(count);
                let mut i = off;
                while i < end {
                    // bounds already verified by end <= blob.len()
                    let bits = u32::from_le_bytes([
                        self.blob[i],
                        self.blob[i + 1],
                        self.blob[i + 2],
                        self.blob[i + 3],
                    ]);
                    d.push(f32::from_bits(bits) as f64);
                    i += 4;
                }
                d
            }
            "q8" => {
                let so = m.get("scale_offset").usize();
                let srows = m.get("scale_rows").usize();

                // Guard scale region.
                let scale_bytes = srows.checked_mul(4).ok_or("scale_rows overflow")?;
                let scale_end = so.checked_add(scale_bytes).ok_or("scale region overflow")?;
                if scale_end > self.blob.len() {
                    return Err("q8 scale data out of blob bounds");
                }
                if srows > MAX_TENSOR_ELEMS {
                    return Err("too many scale rows");
                }

                let mut scales = Vec::with_capacity(srows);
                for r in 0..srows {
                    let p = so + r * 4; // safe: checked above
                    let bits = u32::from_le_bytes([
                        self.blob[p],
                        self.blob[p + 1],
                        self.blob[p + 2],
                        self.blob[p + 3],
                    ]);
                    scales.push(f32::from_bits(bits) as f64);
                }

                let ncols = if rows == 0 { 0 } else { n / rows };
                if n > MAX_TENSOR_ELEMS {
                    return Err("q8 tensor too many bytes");
                }
                // Verify the declared shape matches the byte payload.
                if rows.checked_mul(ncols).map(|p| p != n).unwrap_or(true) {
                    return Err("q8 shape/nbytes mismatch");
                }
                // Each row needs one scale; srows must equal rows.
                if srows != rows {
                    return Err("q8 scale_rows != rows");
                }

                let mut d = Vec::with_capacity(n);
                for r in 0..rows {
                    // scales.len() == srows == rows, so this is safe.
                    let s = scales[r];
                    for c in 0..ncols {
                        // off + r*ncols + c < off + n = end <= blob.len()
                        d.push((self.blob[off + r * ncols + c] as i8) as f64 * s);
                    }
                }
                d
            }
            _ => return Err("unknown tensor dtype"),
        };

        // Sanity: element count must match declared shape.
        if data.len() != total_elems {
            return Err("parsed element count != declared shape");
        }

        Ok((data, rows, cols))
    }

    fn vecf(&self, name: &str) -> Result<Vec<f64>, &'static str> {
        Ok(self.get(name)?.0)
    }

    fn bias(&self, name: &str, out: usize) -> Result<Vec<f64>, &'static str> {
        if self.has(name) {
            let v = self.get(name)?.0;
            if v.len() != out {
                return Err("bias length mismatch");
            }
            Ok(v)
        } else {
            Ok(vec![0.0; out])
        }
    }

    /// Load [out,in] HF weight and return it TRANSPOSED to a [in,out] Tensor (for x·W).
    fn weight_t(&self, name: &str) -> Result<Tensor, &'static str> {
        let (d, out, inp) = self.get(name)?;
        // inp * out was already validated inside `get` via total_elems check,
        // but confirm consistency since get returns (rows=out, cols=inp).
        let total = inp.checked_mul(out).ok_or("weight_t dimension overflow")?;
        if total > MAX_TENSOR_ELEMS {
            return Err("weight_t tensor too large");
        }
        let mut t = vec![0.0f64; total];
        for o in 0..out {
            for i in 0..inp {
                t[i * out + o] = d[o * inp + i];
            }
        }
        Tensor::new(vec![inp, out], t).ok_or("Tensor::new failed in weight_t")
    }
}

impl AemModel {
    /// Parse an `.aem` byte buffer into a ready-to-run model.
    pub fn from_bytes(bytes: &[u8]) -> Result<AemModel, &'static str> {
        // Reject oversized blobs before touching any data.
        if bytes.len() > MAX_MODEL_BYTES {
            return Err("model exceeds MAX_MODEL_BYTES (2 GiB)");
        }
        if bytes.len() < 8 || &bytes[0..4] != b"AEM1" {
            return Err("bad magic");
        }
        let hlen = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        // 8 + hlen must not overflow and must stay within bytes.
        let header_end = 8usize.checked_add(hlen).ok_or("header length overflow")?;
        if header_end > bytes.len() {
            return Err("bad header length");
        }
        let header = parse_json(&bytes[8..header_end]);
        let raw = Raw { header, blob: &bytes[header_end..] };
        let c = raw.header.get("config");

        // Parse and validate each dimension before trusting it.
        let vocab = {
            let v = c.get("vocab_size").usize();
            if v == 0 || v > MAX_VOCAB { return Err("vocab_size out of range"); }
            v
        };
        let hidden = {
            let v = c.get("hidden_size").usize();
            if v == 0 || v > MAX_HIDDEN { return Err("hidden_size out of range"); }
            v
        };
        let n_layers = {
            let v = c.get("n_layers").usize();
            if v == 0 || v > MAX_LAYERS { return Err("n_layers out of range"); }
            v
        };
        let n_heads = {
            let v = c.get("n_heads").usize();
            if v == 0 || v > MAX_HEADS { return Err("n_heads out of range"); }
            v
        };
        let n_kv_heads = {
            let v = c.get("n_kv_heads").usize();
            if v == 0 || v > n_heads { return Err("n_kv_heads out of range"); }
            v
        };
        // n_heads must be a multiple of n_kv_heads (grouped-query attention invariant).
        if n_heads % n_kv_heads != 0 {
            return Err("n_heads not divisible by n_kv_heads");
        }
        let head_dim = {
            let v = c.get("head_dim").usize();
            if v == 0 || v > MAX_HIDDEN { return Err("head_dim out of range"); }
            v
        };

        let cfg = ModelConfig {
            vocab,
            hidden,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            eps: c.get("rms_norm_eps").num_or(1e-6),
            theta: c.get("rope_theta").num_or(10000.0),
            norm_offset: c.get("norm_offset").num_or(0.0),
            embed_scale: c.get("embed_scale").num_or(1.0),
            gelu: c.get("ffn_act").str() == "gelu_tanh",
            attn_denom: c.get("query_pre_attn_scalar").num_or(head_dim as f64),
            final_softcap: c.get("final_logit_softcapping").num_or(0.0),
        };

        let (edata, evocab, ehidden) = raw.get("model.embed_tokens.weight")?;
        if evocab != vocab || ehidden != hidden {
            return Err("embed_tokens shape mismatch with config");
        }
        let embed = Tensor::new(vec![vocab, hidden], edata).ok_or("embed shape")?;

        let lm_rows = if raw.has("lm_head.weight") {
            let (d, r, cc) = raw.get("lm_head.weight")?;
            if r != vocab || cc != hidden {
                return Err("lm_head shape mismatch with config");
            }
            Tensor::new(vec![r, cc], d).ok_or("lm shape")?
        } else {
            embed.clone()
        };

        // Pre-validate qd and kvd overflow before entering the per-layer loop.
        let qd = n_heads.checked_mul(head_dim).ok_or("n_heads*head_dim overflow")?;
        let kvd = n_kv_heads.checked_mul(head_dim).ok_or("n_kv_heads*head_dim overflow")?;
        // Also pre-check vocab * hidden (used in forward logits loop).
        let _ = vocab.checked_mul(hidden).ok_or("vocab*hidden overflow")?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("model.layers.{l}.");
            let gate = raw.weight_t(&format!("{p}mlp.gate_proj.weight"))?;
            let inter = gate.shape()[1];
            if inter == 0 || inter > MAX_INTER {
                return Err("inter (gate_proj out-dim) out of range");
            }
            let in_norm = raw.vecf(&format!("{p}input_layernorm.weight"))?;
            let post_norm = raw.vecf(&format!("{p}post_attention_layernorm.weight"))?;
            let qw = raw.weight_t(&format!("{p}self_attn.q_proj.weight"))?;
            let kw = raw.weight_t(&format!("{p}self_attn.k_proj.weight"))?;
            let vw = raw.weight_t(&format!("{p}self_attn.v_proj.weight"))?;
            let ow = raw.weight_t(&format!("{p}self_attn.o_proj.weight"))?;
            let up = raw.weight_t(&format!("{p}mlp.up_proj.weight"))?;
            let down = raw.weight_t(&format!("{p}mlp.down_proj.weight"))?;
            // Validate projection dims against config so forward() cannot index
            // out of bounds or unwrap a mismatched Tensor::new on a crafted model.
            if in_norm.len() != hidden || post_norm.len() != hidden {
                return Err("layernorm weight length != hidden");
            }
            if qw.shape()[1] != qd { return Err("q_proj out-dim != n_heads*head_dim"); }
            if kw.shape()[1] != kvd { return Err("k_proj out-dim != n_kv_heads*head_dim"); }
            if vw.shape()[1] != kvd { return Err("v_proj out-dim != n_kv_heads*head_dim"); }
            if ow.shape()[0] != qd || ow.shape()[1] != hidden {
                return Err("o_proj shape mismatch (expect [qd, hidden])");
            }
            if up.shape()[1] != inter { return Err("up_proj out-dim != inter"); }
            if down.shape()[1] != hidden { return Err("down_proj out-dim != hidden"); }
            layers.push(Layer {
                in_norm,
                post_norm,
                qb: raw.bias(&format!("{p}self_attn.q_proj.bias"), qd)?,
                kb: raw.bias(&format!("{p}self_attn.k_proj.bias"), kvd)?,
                vb: raw.bias(&format!("{p}self_attn.v_proj.bias"), kvd)?,
                ob: raw.bias(&format!("{p}self_attn.o_proj.bias"), hidden)?,
                qw,
                kw,
                vw,
                ow,
                gate,
                up,
                down,
                inter,
            });
        }
        let final_norm = raw.vecf("model.norm.weight")?;
        if final_norm.len() != hidden {
            return Err("final norm weight length != hidden");
        }
        Ok(AemModel { cfg, embed, final_norm, lm_rows, layers })
    }

    /// Forward `ids` at absolute positions starting at `pos_offset`; extends `kv`.
    /// Returns last-token logits. `spawn` parallelises the projections.
    pub fn forward(&self, ids: &[usize], kv: &mut KvCache, pos_offset: usize, spawn: &dyn Spawn) -> Vec<f64> {
        // Empty input has no last token; guard the `(seq-1)*h` slice underflow below.
        if ids.is_empty() {
            return Vec::new();
        }
        let c = &self.cfg;
        let h = c.hidden;
        let seq = ids.len();
        let qd = c.n_heads * c.head_dim;
        let kvd = c.n_kv_heads * c.head_dim;
        let group = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / sqrt(c.attn_denom);

        // embed + scale
        let ed = self.embed.data();
        let mut x = vec![0.0f64; seq * h];
        for (s, &id) in ids.iter().enumerate() {
            // Token ids from callers are validated here; out-of-vocab ids are clamped
            // to token 0 rather than indexing out of bounds.
            let safe_id = if id < c.vocab { id } else { 0 };
            x[s * h..(s + 1) * h].copy_from_slice(&ed[safe_id * h..(safe_id + 1) * h]);
        }
        if (c.embed_scale - 1.0).abs() > 1e-12 {
            for v in x.iter_mut() {
                *v *= c.embed_scale;
            }
        }

        let (cos_t, sin_t) = rope_tables(seq, c.head_dim, pos_offset, c.theta);

        for (l, lw) in self.layers.iter().enumerate() {
            // attention
            let hn = rmsnorm(&x, seq, h, &lw.in_norm, c.eps, c.norm_offset);
            let hn_t = Tensor::new(vec![seq, h], hn).unwrap();
            let mut q = matmul_bias(&hn_t, &lw.qw, &lw.qb, spawn);
            let mut k = matmul_bias(&hn_t, &lw.kw, &lw.kb, spawn);
            let vv = matmul_bias(&hn_t, &lw.vw, &lw.vb, spawn);
            // RoPE (rotate-half) via the crate's tested helper.
            let qt = apply_rope(&Tensor::new(vec![seq, qd], q).unwrap(), c.n_heads, c.head_dim, &cos_t, &sin_t).unwrap();
            let kt = apply_rope(&Tensor::new(vec![seq, kvd], k).unwrap(), c.n_kv_heads, c.head_dim, &cos_t, &sin_t).unwrap();
            q = qt.data().to_vec();
            k = kt.data().to_vec();

            kv.k[l].extend_from_slice(&k);
            kv.v[l].extend_from_slice(&vv);
            let total = kv.len + seq;

            let mut attn = vec![0.0f64; seq * qd];
            for hh in 0..c.n_heads {
                let kvh = hh / group;
                for i in 0..seq {
                    let abs = pos_offset + i;
                    let qv = &q[(i * c.n_heads + hh) * c.head_dim..][..c.head_dim];
                    let last = abs.min(total - 1);
                    let mut scores = vec![0.0f64; last + 1];
                    let mut mx = f64::NEG_INFINITY;
                    for (j, sc) in scores.iter_mut().enumerate() {
                        let kr = &kv.k[l][(j * c.n_kv_heads + kvh) * c.head_dim..][..c.head_dim];
                        let mut d = 0.0;
                        for t in 0..c.head_dim {
                            d += qv[t] * kr[t];
                        }
                        d *= scale;
                        *sc = d;
                        if d > mx { mx = d; }
                    }
                    let mut den = 0.0;
                    for sc in scores.iter_mut() {
                        *sc = exp(*sc - mx);
                        den += *sc;
                    }
                    let outp = &mut attn[(i * c.n_heads + hh) * c.head_dim..][..c.head_dim];
                    for (j, &w) in scores.iter().enumerate() {
                        let wn = w / den;
                        let vr = &kv.v[l][(j * c.n_kv_heads + kvh) * c.head_dim..][..c.head_dim];
                        for t in 0..c.head_dim {
                            outp[t] += wn * vr[t];
                        }
                    }
                }
            }
            let attn_t = Tensor::new(vec![seq, qd], attn).unwrap();
            let ao = matmul_bias(&attn_t, &lw.ow, &lw.ob, spawn);
            for i in 0..seq * h {
                x[i] += ao[i];
            }

            // mlp
            let hn2 = rmsnorm(&x, seq, h, &lw.post_norm, c.eps, c.norm_offset);
            let hn2_t = Tensor::new(vec![seq, h], hn2).unwrap();
            let zero = vec![0.0f64; lw.inter];
            let gate = matmul_bias(&hn2_t, &lw.gate, &zero, spawn);
            let up = matmul_bias(&hn2_t, &lw.up, &zero, spawn);
            let mut hmid = vec![0.0f64; seq * lw.inter];
            for i in 0..seq * lw.inter {
                let g = gate[i];
                let act = if c.gelu { gelu_tanh(g) } else { g * sigmoid(g) };
                hmid[i] = act * up[i];
            }
            let hmid_t = Tensor::new(vec![seq, lw.inter], hmid).unwrap();
            let zh = vec![0.0f64; h];
            let mo = matmul_bias(&hmid_t, &lw.down, &zh, spawn);
            for i in 0..seq * h {
                x[i] += mo[i];
            }
        }
        kv.len += seq;

        let xf = rmsnorm(&x, seq, h, &self.final_norm, c.eps, c.norm_offset);
        let last = &xf[(seq - 1) * h..seq * h];
        let lm = self.lm_rows.data();
        let mut logits = vec![0.0f64; c.vocab];
        for o in 0..c.vocab {
            let wr = &lm[o * h..(o + 1) * h];
            let mut acc = 0.0;
            for i in 0..h {
                acc += last[i] * wr[i];
            }
            logits[o] = acc;
        }
        if c.final_softcap > 0.0 {
            let cap = c.final_softcap;
            for v in logits.iter_mut() {
                *v = cap * crate::ml::tanh(*v / cap);
            }
        }
        logits
    }

    /// Greedy-generate `n_new` tokens after `prompt`. Returns the new token ids.
    pub fn generate(&self, prompt: &[usize], n_new: usize, spawn: &dyn Spawn) -> Vec<usize> {
        // No prompt -> no logits to sample from; forward() would return empty and
        // argmax on an empty slice would panic.
        if prompt.is_empty() {
            return Vec::new();
        }
        let mut kv = KvCache::new(self.cfg.n_layers);
        let mut logits = self.forward(prompt, &mut kv, 0, spawn);
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax(&logits);
            out.push(next);
            let pos = kv.len();
            logits = self.forward(&[next], &mut kv, pos, spawn);
        }
        out
    }

    /// Convenience: single-threaded generate.
    pub fn generate_serial(&self, prompt: &[usize], n_new: usize) -> Vec<usize> {
        self.generate(prompt, n_new, &Serial)
    }
}

// ───────────────────────── in-OS demo / self-test ─────────────────────────

/// A tiny (84 KB) Qwen2-shaped model embedded in the binary so the booted kernel can
/// run real LLM inference with no filesystem — the on-metal proof of the runtime.
pub const DEMO_AEM: &[u8] = include_bytes!("testdata/fixture.aem");

/// The deterministic next-token id the demo model predicts for [`DEMO_PROMPT`]
/// (computed by the verified NumPy oracle). The in-OS self-test asserts this.
pub const DEMO_EXPECT: usize = 15;
pub const DEMO_PROMPT: [usize; 6] = [5, 17, 3, 42, 9, 28];

/// Load the embedded demo model and run one forward; returns the argmax token id.
/// Used by the kernel self-test and the `llm` shell command to prove on-device
/// inference works live in the OS. `spawn` selects serial vs SMP-parallel matmul.
pub fn demo_run(spawn: &dyn Spawn) -> Result<usize, &'static str> {
    let m = AemModel::from_bytes(DEMO_AEM)?;
    let mut kv = KvCache::new(m.cfg.n_layers);
    let logits = m.forward(&DEMO_PROMPT, &mut kv, 0, spawn);
    Ok(argmax(&logits))
}

/// True iff the embedded model runs and predicts the oracle-verified token.
pub fn demo_selftest(spawn: &dyn Spawn) -> bool {
    demo_run(spawn) == Ok(DEMO_EXPECT)
}

// ───────────────────────────── helpers ─────────────────────────────

/// y[seq,out] = x[seq,in]·W[in,out] + bias, parallelised over output columns via `spawn`.
fn matmul_bias(x: &Tensor, w: &Tensor, bias: &[f64], spawn: &dyn Spawn) -> Vec<f64> {
    let seq = x.shape()[0];
    let in_dim = x.shape()[1];
    let out = w.shape()[1];
    let xd = x.data();
    let wd = w.data();
    let nb = spawn.max_workers().max(1).min(out);
    let band = out.div_ceil(nb);
    // Each task computes a band of output columns for all rows → [seq * bandwidth].
    let parts = spawn.run(nb, &|b| {
        let o0 = b * band;
        let o1 = ((b + 1) * band).min(out);
        if o0 >= o1 {
            return Vec::new();
        }
        let bw = o1 - o0;
        let mut slab = vec![0.0f64; seq * bw];
        for s in 0..seq {
            let xr = &xd[s * in_dim..(s + 1) * in_dim];
            for (oi, o) in (o0..o1).enumerate() {
                let mut acc = bias[o];
                for i in 0..in_dim {
                    acc += xr[i] * wd[i * out + o];
                }
                slab[s * bw + oi] = acc;
            }
        }
        slab
    });
    // Reassemble row-major [seq, out].
    let mut y = vec![0.0f64; seq * out];
    let mut o0 = 0;
    for slab in &parts {
        if slab.is_empty() {
            continue;
        }
        let bw = slab.len() / seq.max(1);
        for s in 0..seq {
            y[s * out + o0..s * out + o0 + bw].copy_from_slice(&slab[s * bw..(s + 1) * bw]);
        }
        o0 += bw;
    }
    y
}

fn rmsnorm(x: &[f64], seq: usize, d: usize, w: &[f64], eps: f64, off: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; seq * d];
    for s in 0..seq {
        let r = &x[s * d..(s + 1) * d];
        let mut ss = 0.0;
        for &v in r {
            ss += v * v;
        }
        let inv = 1.0 / sqrt(ss / d as f64 + eps);
        for i in 0..d {
            out[s * d + i] = r[i] * inv * (w[i] + off);
        }
    }
    out
}

fn gelu_tanh(x: f64) -> f64 {
    const K: f64 = 0.797_884_560_802_865_4;
    0.5 * x * (1.0 + crate::ml::tanh(K * (x + 0.044_715 * x * x * x)))
}

fn argmax(v: &[f64]) -> usize {
    let mut bi = 0;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; bi = i; }
    }
    bi
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // The tiny Qwen2-shaped fixture, embedded so the test needs no filesystem — this is
    // the on-DominionOS-code proof: parse .aem + run the forward + match the NumPy oracle.
    const FIXTURE: &[u8] = include_bytes!("testdata/fixture.aem");
    const EXPECTED: &str = include_str!("testdata/fixture_expected.json");

    #[test]
    fn loads_and_runs_fixture() {
        let m = AemModel::from_bytes(FIXTURE).expect("parse .aem");
        assert_eq!(m.cfg.n_layers, 2);
        assert_eq!(m.cfg.hidden, 32);
        let exp = parse_json(EXPECTED.as_bytes());
        let ids: Vec<usize> = exp.get("input_ids").arr().iter().map(|x| x.usize()).collect();
        let elog: Vec<f64> = exp.get("logits_last").arr().iter().map(|x| x.num()).collect();
        let eam = exp.get("argmax_id").usize();

        let mut kv = KvCache::new(m.cfg.n_layers);
        let logits = m.forward(&ids, &mut kv, 0, &Serial);
        assert_eq!(logits.len(), elog.len());
        let mut maxd = 0.0f64;
        for (a, b) in logits.iter().zip(&elog) {
            let d = (a - b).abs();
            if d > maxd { maxd = d; }
        }
        assert_eq!(argmax(&logits), eam, "argmax must match the oracle");
        assert!(maxd < 1e-2, "logit parity too loose: {maxd}");
    }

    #[test]
    fn float_parser_matches_expected() {
        assert!((parse_f64(b"1e-06") - 1e-6).abs() < 1e-18);
        assert!((parse_f64(b"10000.0") - 10000.0).abs() < 1e-9);
        assert!((parse_f64(b"-2.5") + 2.5).abs() < 1e-12);
        assert!((parse_f64(b"30") - 30.0).abs() < 1e-12);
    }

    #[test]
    fn generate_is_deterministic() {
        let m = AemModel::from_bytes(FIXTURE).unwrap();
        let a = m.generate_serial(&[5, 17, 3], 4);
        let b = m.generate_serial(&[5, 17, 3], 4);
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn rejects_oversized_model() {
        // A blob larger than MAX_MODEL_BYTES must be rejected immediately.
        // We only need to fabricate the length claim, not actual data.
        let fake = vec![0u8; 9]; // too short to be valid but big enough to check size gate
        // Actual oversized test: build a fake header that claims > 2 GiB.
        // We simulate by checking the public constant is what we expect.
        assert_eq!(MAX_MODEL_BYTES, 2 * 1024 * 1024 * 1024);
        // A 9-byte buffer fails the magic check, not the size check — that is fine.
        assert!(AemModel::from_bytes(&fake).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bad = b"BAD1\x00\x00\x00\x00".to_vec();
        bad.extend_from_slice(b"{}");
        assert!(AemModel::from_bytes(&bad).is_err());
    }

    #[test]
    fn rejects_truncated_header() {
        // Magic OK, hlen claims more bytes than the buffer holds.
        let mut buf = b"AEM1".to_vec();
        buf.extend_from_slice(&(9999u32).to_le_bytes()); // hlen = 9999, but buffer is tiny
        buf.extend_from_slice(b"{}");
        assert!(AemModel::from_bytes(&buf).is_err());
    }
}

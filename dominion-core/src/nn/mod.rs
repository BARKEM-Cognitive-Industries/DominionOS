//! Neural-network operator layer — transformer blocks, CNN ops, RNN cells,
//! samplers, tokenizers, and full architecture templates.
//!
//! Every operator is pure no_std+alloc, #![forbid(unsafe_code)], bit-exact
//! deterministic. Every tunable is a knob in NnConfig.
//!
//! Architecture support:
//! - Transformers: attention, RoPE, RMSNorm/LayerNorm, SwiGLU/GeGLU, KV cache
//! - CNNs: ConvLayer (im2col, inline in arch/cnn.rs), BatchNorm/GroupNorm
//! - RNNs: LSTM, GRU, vanilla RNN
//! - MoE: top-K sparse routing + expert FFNs
//! - Embeddings: lookup, sinusoidal, RoPE, ALiBi
//! - Samplers: greedy, top-k, top-p, temperature, repetition penalty
//! - Tokenizers: BPE (byte-level), log-mel spectrogram

pub mod attention;
pub mod arch;
pub mod embed;
pub mod ffn;
pub mod model;
pub mod norm;
pub mod rope;
pub mod sample;
pub mod tokenizer;

/// Which normalization operator to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind { Rms, Layer, Group, Batch, None }

/// FFN activation variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnAct { Gelu, Silu, Swiglu, Geglu, Relu, Reglu, Linear }

/// Convolution padding strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvPad { Valid, Same, Explicit }

/// All runtime knobs for the nn layer. Every field is configurable.
/// Default = [`NnConfig::best()`].
#[derive(Clone, Debug)]
pub struct NnConfig {
    // attention
    pub kv_heads: usize,
    pub flash_block: usize,
    pub causal: bool,
    pub attn_scale: f64,
    pub alibi: bool,
    pub rope_base: f64,
    pub sliding_window: usize,
    pub kv_cache_len: usize,
    // norm
    pub norm_eps: f64,
    pub norm_kind: NormKind,
    pub group_norm_groups: usize,
    // ffn
    pub ffn_act: FfnAct,
    pub ffn_ratio: f64,
    pub pre_norm: bool,
    pub fuse_gate_proj: bool,
    // conv
    pub conv_pad: ConvPad,
    pub depthwise_sep: bool,
    // sampling
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub rep_penalty: f64,
    pub sample_seed: u64,
    pub grid_snap: bool,
    pub grid_snap_levels: u32,
    // moe
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub moe_aux_loss: f64,
    // quantization
    pub precision: crate::ml::Precision,
    pub turboquant_kv: bool,
    pub turboquant_bits: u8,
    pub gptq: bool,
    pub awq: bool,
    pub smooth_quant_alpha: f64,
    // perf
    pub conv_tile: usize,
    pub incremental_infer: bool,
    pub speculative: bool,
    pub spec_tol: f64,
    pub lora_rank: usize,
}

impl Default for NnConfig { fn default() -> Self { NnConfig::best() } }

impl NnConfig {
    pub fn best() -> Self {
        NnConfig {
            kv_heads: 0, flash_block: 32, causal: true, attn_scale: 0.0,
            alibi: false, rope_base: 10000.0, sliding_window: 0, kv_cache_len: 2048,
            norm_eps: 1e-6, norm_kind: NormKind::Rms, group_norm_groups: 32,
            ffn_act: FfnAct::Swiglu, ffn_ratio: 8.0 / 3.0, pre_norm: true, fuse_gate_proj: true,
            conv_pad: ConvPad::Same, depthwise_sep: false,
            temperature: 1.0, top_k: 0, top_p: 1.0, rep_penalty: 1.0,
            sample_seed: 0x8B1A_2F3E_4C5D_6E7F, grid_snap: true, grid_snap_levels: 65536,
            num_experts: 1, top_k_experts: 1, moe_aux_loss: 0.01,
            precision: crate::ml::Precision::Int8,
            turboquant_kv: false, turboquant_bits: 4, gptq: false, awq: false,
            smooth_quant_alpha: 0.0,
            conv_tile: 32, incremental_infer: true, speculative: false, spec_tol: 0.05,
            lora_rank: 0,
        }
    }
    pub fn accurate() -> Self {
        let mut c = Self::best();
        c.precision = crate::ml::Precision::F64;
        c.gptq = false; c.awq = false; c.smooth_quant_alpha = 0.0;
        c.turboquant_kv = false; c.grid_snap = false; c
    }
    pub fn llama_style(kv_heads: usize) -> Self {
        let mut c = Self::best();
        c.kv_heads = kv_heads; c.norm_kind = NormKind::Rms;
        c.ffn_act = FfnAct::Swiglu; c.rope_base = 10000.0; c.causal = true; c
    }
    pub fn gpt_style() -> Self {
        let mut c = Self::best();
        c.kv_heads = 0; c.norm_kind = NormKind::Layer;
        c.ffn_act = FfnAct::Gelu; c.pre_norm = false; c
    }
    pub fn effective_kv_heads(&self, n_heads: usize) -> usize {
        if self.kv_heads == 0 { n_heads } else { self.kv_heads }
    }
    pub fn is_gqa(&self, n_heads: usize) -> bool {
        self.kv_heads > 0 && self.kv_heads < n_heads
    }
}

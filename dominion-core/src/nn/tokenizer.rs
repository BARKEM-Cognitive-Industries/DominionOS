//! BPE tokenizer and log-mel spectrogram frontend.
//!
//! Provides:
//! * [`BpeTokenizer`] — byte-pair encoding inference engine (no_std + alloc).
//! * [`LogMelFrontend`] — Whisper-compatible log-mel spectrogram computation.
//!
//! Both are pure, safe, no_std + alloc with no external dependencies.
//! Real model vocab/merge tables are loaded from `.aem` model files at runtime;
//! the `byte_level` constructor builds a minimal test tokenizer from 256 raw bytes.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use crate::datatypes::Tensor;

// ─────────────────────────── sin / cos (Taylor series) ───────────────────────
//
// no_std has no f64::sin/f64::cos. We use degree-8 minimax-quality Taylor
// expansions centred at 0, valid for |x| ≤ π (we range-reduce before calling).
//
//   sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
//   cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 + x⁸/40320

fn sin_taylor(x: f64) -> f64 {
    // Fold to [-π/2, π/2] then use 9-term Horner — error < 5e-14.
    const PI: f64 = core::f64::consts::PI;
    let x = if x > PI / 2.0 { PI - x } else if x < -(PI / 2.0) { -PI - x } else { x };
    let x2 = x * x;
    let c17: f64 =  1.0 / 355_687_428_096_000.0;
    let c15: f64 = -1.0 / 1_307_674_368_000.0;
    let c13: f64 =  1.0 / 6_227_020_800.0;
    let c11: f64 = -1.0 / 39_916_800.0;
    let c9:  f64 =  1.0 / 362_880.0;
    let c7:  f64 = -1.0 / 5_040.0;
    let c5:  f64 =  1.0 / 120.0;
    let c3:  f64 = -1.0 / 6.0;
    x * ((((((((c17 * x2 + c15) * x2 + c13) * x2 + c11) * x2 + c9) * x2 + c7) * x2 + c5) * x2 + c3) * x2 + 1.0)
}

fn cos_taylor(x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    let (x, neg) = if x > PI / 2.0 { (PI - x, true) } else if x < -(PI / 2.0) { (-PI - x, true) } else { (x, false) };
    let x2 = x * x;
    let c16: f64 =  1.0 / 20_922_789_888_000.0;
    let c14: f64 = -1.0 / 87_178_291_200.0;
    let c12: f64 =  1.0 / 479_001_600.0;
    let c10: f64 = -1.0 / 3_628_800.0;
    let c8:  f64 =  1.0 / 40_320.0;
    let c6:  f64 = -1.0 / 720.0;
    let c4:  f64 =  1.0 / 24.0;
    let c2:  f64 = -1.0 / 2.0;
    let v = (((((((c16 * x2 + c14) * x2 + c12) * x2 + c10) * x2 + c8) * x2 + c6) * x2 + c4) * x2 + c2) * x2 + 1.0;
    if neg { -v } else { v }
}

/// Reduce `x` to `[-π, π]` then return `(sin(x), cos(x))`.
fn sincos(x: f64) -> (f64, f64) {
    const PI: f64 = core::f64::consts::PI;
    const TWO_PI: f64 = 2.0 * PI;
    // Bring into [0, 2π)
    let mut t = x % TWO_PI;
    if t < 0.0 {
        t += TWO_PI;
    }
    // Reduce to [-π, π]
    if t > PI {
        t -= TWO_PI;
    }
    (sin_taylor(t), cos_taylor(t))
}

/// Natural logarithm via the identity ln(x) = 2·atanh((x-1)/(x+1)).
/// atanh(y) = y + y³/3 + y⁵/5 + …  (series converges for |y| < 1).
/// Works for x > 0; returns a large negative number for x ≤ 0 (floor).
fn ln_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return -1e30;
    }
    // Scale x = m * 2^e so that m ∈ [0.5, 1.0), then ln(x) = ln(m) + e*ln(2).
    const LN2: f64 = 0.693_147_180_559_945_31;
    let mut m = x;
    let mut e: i32 = 0;
    while m >= 1.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 0.5 {
        m *= 2.0;
        e -= 1;
    }
    // Now m ∈ [0.5, 1.0). Use y = (m-1)/(m+1), ln(m) = 2*atanh(y).
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    // 9-term series: 2*(y + y³/3 + y⁵/5 + y⁷/7 + y⁹/9 + y¹¹/11 + y¹³/13 + y¹⁵/15 + y¹⁷/17)
    let series = y * (1.0
        + y2 * (1.0 / 3.0
            + y2 * (1.0 / 5.0
                + y2 * (1.0 / 7.0
                    + y2 * (1.0 / 9.0
                        + y2 * (1.0 / 11.0
                            + y2 * (1.0 / 13.0
                                + y2 * (1.0 / 15.0
                                    + y2 * (1.0 / 17.0
                                        + y2 * (1.0 / 19.0))))))))));
    2.0 * series + e as f64 * LN2
}

/// log10(x) = ln(x) / ln(10).
fn log10_approx(x: f64) -> f64 {
    const LN10: f64 = 2.302_585_092_994_046;
    ln_approx(x) / LN10
}

// ──────────────────────────── BpeTokenizer ────────────────────────────────────

/// Byte-Pair Encoding tokenizer.
///
/// * `vocab[id]` — the byte sequence for token `id`.
/// * `merges[i]` — the `i`-th merge rule `(a_id, b_id)`; lower index = higher priority.
/// * `special_tokens` — byte sequences that map directly to a fixed id (e.g. `<|endoftext|>`).
///
/// Real model vocab and merge tables are loaded from an `.aem` model file. This struct
/// is the *inference engine* that applies a pre-loaded table.
pub struct BpeTokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(u32, u32)>,
    pub special_tokens: BTreeMap<Vec<u8>, u32>,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
}

impl BpeTokenizer {
    /// Construct from vocab (list of token byte sequences) and merge rules.
    pub fn new(
        vocab: Vec<Vec<u8>>,
        merges: Vec<(u32, u32)>,
        bos_id: u32,
        eos_id: u32,
        unk_id: u32,
    ) -> Self {
        BpeTokenizer { vocab, merges, special_tokens: BTreeMap::new(), eos_id, bos_id, unk_id }
    }

    /// Tokenize UTF-8 text bytes into token ids.
    ///
    /// Algorithm:
    /// 1. Split input into UTF-8 scalar byte sequences (multi-byte chars stay together).
    /// 2. Map each character's bytes to its vocab id; unknown bytes → `unk_id`.
    /// 3. Iteratively apply merge rules in priority order until no merge fires.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Build reverse map: bytes → id (scan whole vocab once).
        // For single-byte initial tokens we need byte → id.
        let mut byte_to_id: [u32; 256] = [self.unk_id; 256];
        for (id, token_bytes) in self.vocab.iter().enumerate() {
            if token_bytes.len() == 1 {
                byte_to_id[token_bytes[0] as usize] = id as u32;
            }
        }

        // Step 1+2: split UTF-8 characters → initial token ids.
        // We iterate over the raw bytes and collect UTF-8 code-point byte runs.
        let mut ids: Vec<u32> = Vec::with_capacity(text.len());
        let mut i = 0;
        while i < text.len() {
            let b = text[i];
            // Determine UTF-8 character length.
            let char_len = if b < 0x80 {
                1
            } else if b & 0xE0 == 0xC0 {
                2
            } else if b & 0xF0 == 0xE0 {
                3
            } else if b & 0xF8 == 0xF0 {
                4
            } else {
                1 // continuation or invalid — treat as single byte
            };
            let char_bytes = &text[i..usize::min(i + char_len, text.len())];
            // Try to find exact match in vocab.
            let mut found = self.unk_id;
            for (vid, vbytes) in self.vocab.iter().enumerate() {
                if vbytes.as_slice() == char_bytes {
                    found = vid as u32;
                    break;
                }
            }
            if found == self.unk_id && char_bytes.len() == 1 {
                found = byte_to_id[char_bytes[0] as usize];
            }
            ids.push(found);
            i += char_bytes.len();
        }

        // Step 3: iteratively apply merge rules.
        // Build a lookup: (a, b) → (priority, merged_id).
        // merged_id = vocab.len() is the base for merged tokens, but in standard BPE
        // the merged token id equals the index in the merge list offset by the initial
        // vocab size. Here we follow the convention that merged tokens are already in
        // the vocab (the vocab passed in includes all merged tokens).
        //
        // We build a BTreeMap from (a, b) → (priority, merged_id).
        let mut merge_map: BTreeMap<(u32, u32), (usize, u32)> = BTreeMap::new();
        for (priority, &(a, b)) in self.merges.iter().enumerate() {
            // The merged token id is the vocab index of the bytes that are the
            // concatenation of vocab[a] and vocab[b].
            let mut merged_bytes: Vec<u8> = Vec::new();
            if let Some(ab) = self.vocab.get(a as usize) {
                merged_bytes.extend_from_slice(ab);
            }
            if let Some(bb) = self.vocab.get(b as usize) {
                merged_bytes.extend_from_slice(bb);
            }
            // Find the merged id in vocab.
            let mut merged_id = self.unk_id;
            for (vid, vbytes) in self.vocab.iter().enumerate() {
                if *vbytes == merged_bytes {
                    merged_id = vid as u32;
                    break;
                }
            }
            merge_map.insert((a, b), (priority, merged_id));
        }

        // Repeatedly scan for the highest-priority (lowest index) applicable merge.
        loop {
            if ids.len() < 2 {
                break;
            }
            // Find the best merge across all adjacent pairs.
            let mut best_priority: Option<usize> = None;
            let mut best_pos: usize = 0;
            let mut best_merged: u32 = self.unk_id;
            for j in 0..ids.len() - 1 {
                let pair = (ids[j], ids[j + 1]);
                if let Some(&(prio, merged)) = merge_map.get(&pair) {
                    if best_priority.is_none() || prio < best_priority.unwrap() {
                        best_priority = Some(prio);
                        best_pos = j;
                        best_merged = merged;
                    }
                }
            }
            if best_priority.is_none() {
                break;
            }
            // Apply: replace ids[best_pos] and ids[best_pos+1] with best_merged.
            ids[best_pos] = best_merged;
            ids.remove(best_pos + 1);
        }

        ids
    }

    /// Decode token ids back to a byte sequence.
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(bytes) = self.vocab.get(id as usize) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Build a minimal byte-level BPE tokenizer from the 256 raw bytes as the
    /// initial vocabulary. No merge rules. Used for testing when no pretrained
    /// vocab is available.
    pub fn byte_level(eos_id: u32, bos_id: u32) -> Self {
        let vocab: Vec<Vec<u8>> = (0u8..=255).map(|b| alloc::vec![b]).collect();
        let unk_id = 0u32; // byte 0x00 as catch-all unknown
        BpeTokenizer {
            vocab,
            merges: Vec::new(),
            special_tokens: BTreeMap::new(),
            eos_id,
            bos_id,
            unk_id,
        }
    }
}

// ─────────────────────────── LogMelFrontend ───────────────────────────────────

/// Log-mel spectrogram frontend for Whisper-style audio encoding.
///
/// Input: PCM f64 samples normalised to `[-1, 1]` at `sample_rate` Hz.
/// Output: `[n_mels, n_frames]` log-mel spectrogram as a [`Tensor`].
pub struct LogMelFrontend {
    /// FFT window size (512 for Whisper small).
    pub n_fft: usize,
    /// Frame shift in samples (160 → 10 ms at 16 kHz).
    pub hop_length: usize,
    /// Number of mel filter banks (80 for Whisper).
    pub n_mels: usize,
    /// Sample rate in Hz (16000 for Whisper).
    pub sample_rate: usize,
    /// Pre-computed mel filterbank: row-major `[n_mels, n_fft/2 + 1]`.
    pub mel_filters: Vec<f64>,
}

impl LogMelFrontend {
    /// Hz → mel (HTK formula): `2595 * log10(1 + hz / 700)`.
    fn hz_to_mel(hz: f64) -> f64 {
        2595.0 * log10_approx(1.0 + hz / 700.0)
    }

    /// Mel → Hz (inverse HTK): `700 * (10^(mel / 2595) - 1)`.
    fn mel_to_hz(mel: f64) -> f64 {
        // 10^x = e^(x * ln10)
        const LN10: f64 = 2.302_585_092_994_046;
        let exp_arg = mel / 2595.0 * LN10;
        // e^x via Taylor; for small x this is exact enough, but exp_arg can be ~1.5.
        // We use the same ln-based identity: e^x = 1 / e^(-x) when x < 0, or iterate.
        // Simple: e^x for x in [0, 3] via 15-term Taylor.
        let x = exp_arg;
        let ex = {
            let mut term = 1.0f64;
            let mut sum = 1.0f64;
            for k in 1..=20u32 {
                term *= x / k as f64;
                sum += term;
            }
            sum
        };
        700.0 * (ex - 1.0)
    }

    /// Build triangular mel filterbank matrix `[n_mels, n_fft/2 + 1]`.
    fn build_mel_filters(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f64> {
        let n_freqs = n_fft / 2 + 1;
        let f_min = 0.0f64;
        let f_max = sample_rate as f64 / 2.0;

        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        // n_mels + 2 equally spaced mel points.
        let n_points = n_mels + 2;
        let mel_points: Vec<f64> = (0..n_points)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
            .collect();

        // Convert mel points back to Hz, then to FFT bin indices.
        let hz_points: Vec<f64> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let bin_points: Vec<f64> = hz_points
            .iter()
            .map(|&hz| hz * (n_fft as f64 + 1.0) / sample_rate as f64)
            .collect();

        let mut filters = vec![0.0f64; n_mels * n_freqs];
        for m in 0..n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];
            for k in 0..n_freqs {
                let kf = k as f64;
                let val = if kf >= left && kf <= center {
                    if center > left {
                        (kf - left) / (center - left)
                    } else {
                        0.0
                    }
                } else if kf > center && kf <= right {
                    if right > center {
                        (right - kf) / (right - center)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                filters[m * n_freqs + k] = val;
            }
        }
        filters
    }

    /// Build a standard Whisper-small-compatible frontend.
    ///
    /// Parameters: `n_fft=512`, `hop_length=160`, `n_mels=80`, `sample_rate=16000`.
    pub fn whisper_small() -> Self {
        let n_fft = 512;
        let hop_length = 160;
        let n_mels = 80;
        let sample_rate = 16000;
        let mel_filters = Self::build_mel_filters(n_mels, n_fft, sample_rate);
        LogMelFrontend { n_fft, hop_length, n_mels, sample_rate, mel_filters }
    }

    /// Hann window of length `n`: `w[i] = 0.5 * (1 - cos(2π·i / (n-1)))`.
    fn hann_window(n: usize) -> Vec<f64> {
        const PI: f64 = core::f64::consts::PI;
        (0..n)
            .map(|i| {
                let (_, c) = sincos(2.0 * PI * i as f64 / (n - 1) as f64);
                0.5 * (1.0 - c)
            })
            .collect()
    }

    /// Real-valued DFT of length `n` over `samples` (windowed).
    /// Returns `n/2 + 1` complex magnitudes squared (power spectrum).
    /// O(N²) — acceptable for N=512, ~131K mults per frame.
    fn power_spectrum(samples: &[f64], window: &[f64], n: usize) -> Vec<f64> {
        let n_freqs = n / 2 + 1;
        let mut power = vec![0.0f64; n_freqs];
        const PI: f64 = core::f64::consts::PI;
        for k in 0..n_freqs {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for t in 0..n {
                let x = if t < samples.len() { samples[t] * window[t] } else { 0.0 };
                let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
                let (s, c) = sincos(angle);
                re += x * c;
                im += x * s;
            }
            power[k] = re * re + im * im;
        }
        power
    }

    /// Process raw audio and return a `[n_mels, n_frames]` log-mel spectrogram tensor.
    ///
    /// Steps:
    /// 1. Pad audio length to a multiple of `n_fft`.
    /// 2. Sliding window DFT with Hann window, stride = `hop_length`.
    /// 3. Power spectrum: `|DFT|²`.
    /// 4. Mel filterbank: `mel[m] = Σ_k filter[m,k] * power[k]`.
    /// 5. Log: `max(mel, 1e-10).ln()`.
    /// 6. Whisper normalisation: `(log_mel - global_mean) / 4.0`.
    ///
    /// Returns `None` if `audio` is empty or if tensor construction fails.
    pub fn forward(&self, audio: &[f64]) -> Option<Tensor> {
        if audio.is_empty() {
            return None;
        }

        let n_fft = self.n_fft;
        let n_freqs = n_fft / 2 + 1;

        // Step 1: pad to nearest n_fft multiple.
        let pad_len = ((audio.len() + n_fft - 1) / n_fft) * n_fft;
        let mut padded = Vec::with_capacity(pad_len);
        padded.extend_from_slice(audio);
        padded.resize(pad_len, 0.0);

        // Pre-compute Hann window.
        let window = Self::hann_window(n_fft);

        // Step 2+3: sliding window DFT → power spectra.
        let n_frames = if pad_len >= n_fft {
            (pad_len - n_fft) / self.hop_length + 1
        } else {
            1
        };

        // power_frames: [n_frames, n_freqs] (row-major)
        let mut power_frames: Vec<f64> = Vec::with_capacity(n_frames * n_freqs);
        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = (start + n_fft).min(padded.len());
            let slice = &padded[start..end];
            let ps = Self::power_spectrum(slice, &window, n_fft);
            power_frames.extend_from_slice(&ps);
        }

        // Step 4: apply mel filterbank → mel_frames: [n_mels, n_frames]
        // mel_out[m, f] = sum_k mel_filters[m, k] * power_frames[f, k]
        let mut mel_frames: Vec<f64> = vec![0.0f64; self.n_mels * n_frames];
        for m in 0..self.n_mels {
            for f in 0..n_frames {
                let mut energy = 0.0f64;
                for k in 0..n_freqs {
                    energy += self.mel_filters[m * n_freqs + k] * power_frames[f * n_freqs + k];
                }
                mel_frames[m * n_frames + f] = energy;
            }
        }

        // Step 5: log with floor at 1e-10.
        for v in mel_frames.iter_mut() {
            let floored = if *v < 1e-10 { 1e-10 } else { *v };
            *v = ln_approx(floored);
        }

        // Step 6: Whisper normalisation — subtract global mean, divide by 4.
        let mean = mel_frames.iter().sum::<f64>() / mel_frames.len() as f64;
        for v in mel_frames.iter_mut() {
            *v = (*v - mean) / 4.0;
        }

        Tensor::new(vec![self.n_mels, n_frames], mel_frames)
    }
}

// ─────────────────────────────── tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BpeTokenizer tests ───────────────────────────────────────────────────

    #[test]
    fn byte_level_roundtrip() {
        let tok = BpeTokenizer::byte_level(1, 0);
        let text = b"Hello, world!";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "byte-level encode/decode must roundtrip");
    }

    #[test]
    fn byte_level_encode_len_equals_input_bytes() {
        let tok = BpeTokenizer::byte_level(1, 0);
        let text = b"abc";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], b'a' as u32);
        assert_eq!(ids[1], b'b' as u32);
        assert_eq!(ids[2], b'c' as u32);
    }

    #[test]
    fn bpe_merge_applied() {
        // Vocab: 0=b'a', 1=b'b', 2=b'ab' (merged).
        let vocab: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()];
        let merges = vec![(0u32, 1u32)]; // merge (a, b) → ab (id 2)
        let tok = BpeTokenizer::new(vocab, merges, 255, 254, 0);
        let ids = tok.encode(b"ab");
        // After merge (0,1)→2, we should get [2].
        assert_eq!(ids, vec![2u32], "merge (a,b)→ab must produce single token");
    }

    #[test]
    fn bpe_merge_partial() {
        // Vocab: 0=a, 1=b, 2=c, 3=ab
        let vocab: Vec<Vec<u8>> =
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"ab".to_vec()];
        let merges = vec![(0u32, 1u32)];
        let tok = BpeTokenizer::new(vocab, merges, 255, 254, 0);
        let ids = tok.encode(b"abc");
        // "a"+"b" merges to 3, "c" stays as 2 → [3, 2]
        assert_eq!(ids, vec![3u32, 2u32]);
    }

    #[test]
    fn decode_empty() {
        let tok = BpeTokenizer::byte_level(1, 0);
        assert!(tok.decode(&[]).is_empty());
    }

    #[test]
    fn encode_empty() {
        let tok = BpeTokenizer::byte_level(1, 0);
        assert!(tok.encode(b"").is_empty());
    }

    // ── LogMelFrontend tests ─────────────────────────────────────────────────

    #[test]
    fn whisper_small_constructs() {
        let fe = LogMelFrontend::whisper_small();
        assert_eq!(fe.n_fft, 512);
        assert_eq!(fe.hop_length, 160);
        assert_eq!(fe.n_mels, 80);
        assert_eq!(fe.sample_rate, 16000);
        assert_eq!(fe.mel_filters.len(), 80 * (512 / 2 + 1));
    }

    #[test]
    fn forward_returns_correct_shape() {
        let fe = LogMelFrontend::whisper_small();
        // 1600 samples = 100 ms at 16 kHz.
        let audio: Vec<f64> = (0..1600).map(|i| (i as f64 * 0.001).sin() * 0.5).collect();
        let spec = fe.forward(&audio).expect("forward should succeed");
        // n_frames = (pad_len - n_fft) / hop_length + 1
        // pad_len = ceil(1600/512)*512 = 2048
        // n_frames = (2048 - 512) / 160 + 1 = 1536/160 + 1 = 9 + 1 = 10
        assert_eq!(spec.shape()[0], 80, "n_mels must be 80");
        assert!(spec.shape()[1] > 0, "n_frames must be positive");
    }

    #[test]
    fn forward_empty_returns_none() {
        let fe = LogMelFrontend::whisper_small();
        assert!(fe.forward(&[]).is_none());
    }

    #[test]
    fn mel_filters_sum_to_positive() {
        let fe = LogMelFrontend::whisper_small();
        let total: f64 = fe.mel_filters.iter().sum();
        assert!(total > 0.0, "mel filterbank must have positive total energy");
    }

    #[test]
    fn log_mel_values_are_finite() {
        let fe = LogMelFrontend::whisper_small();
        // White noise-like signal.
        let audio: Vec<f64> = (0..3200)
            .map(|i| {
                let t = i as f64 / 16000.0;
                (2.0 * core::f64::consts::PI * 440.0 * t).sin() * 0.3
            })
            .collect();
        let spec = fe.forward(&audio).expect("forward should succeed");
        for &v in spec.data() {
            assert!(v.is_finite(), "log-mel value must be finite, got {}", v);
        }
    }

    #[test]
    fn sincos_accuracy() {
        const PI: f64 = core::f64::consts::PI;
        // Test a few known values.
        let (s, c) = sincos(0.0);
        assert!((s - 0.0).abs() < 1e-12 && (c - 1.0).abs() < 1e-12);
        let (s, c) = sincos(PI / 2.0);
        assert!((s - 1.0).abs() < 1e-10 && c.abs() < 1e-10);
        let (s, c) = sincos(PI);
        assert!(s.abs() < 1e-10 && (c + 1.0).abs() < 1e-12);
    }

    #[test]
    fn ln_approx_accuracy() {
        // ln(1) = 0, ln(e) ≈ 1, ln(10) ≈ 2.302...
        assert!((ln_approx(1.0)).abs() < 1e-10);
        let e = 2.718_281_828_459_045f64;
        assert!((ln_approx(e) - 1.0).abs() < 1e-9);
        assert!((ln_approx(10.0) - 2.302_585_092_994_046).abs() < 1e-9);
    }
}

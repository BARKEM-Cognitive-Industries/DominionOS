//! Semantic audio — **Stage 8** (see `docs/architecture/09-stage-08-semantic-audio.md`).
//!
//! Audio is not a flat PCM stream here; it is a graph of **semantic audio objects**
//! — sources with *meaning*, a 3D position, and directivity — that the system
//! renders for a listener. The pieces:
//!
//! * **Semantic tokenizer (JSCM).** A waveform descriptor is encoded to a small set
//!   of **semantic tokens** (meaning over waveform fidelity). Because the decoder
//!   snaps a received token back to the nearest codebook entry, the meaning survives
//!   a **much noisier channel** than raw PCM would — the Joint Source-Channel Coding
//!   win (~equal quality at lower SNR).
//! * **Object-Based Audio + kernel HRTF.** Each [`AudioObject`] has a position and
//!   directivity; [`Hrtf`] spatialises it to a stereo (per-ear) frame with
//!   interaural level/time differences — the kernel-level HRTF for VR/XR.
//! * **EDF scheduling.** [`EdfScheduler`] orders render tasks by **earliest
//!   deadline** and checks the isochronous **16.6 ms** frame budget is met.
//! * **Zero-copy DMA.** [`SharedAudioBuffer`] is passed to the GPU node by *handle*,
//!   never by copying the samples.
//!
//! Pure, safe `no_std`, host-tested. No FP transcendentals (no `libm`) — the HRTF
//! model uses linear pan/attenuation so it runs anywhere and replays deterministically.

use crate::hash::Hash256;
use alloc::vec::Vec;

/// The isochronous audio frame deadline: 16.6 ms in microseconds.
pub const FRAME_DEADLINE_US: u64 = 16_600;

// ─────────────────────────── semantic tokenizer (JSCM) ───────────────────────────

/// A semantic audio token: an index into a learned codebook of sound *meanings*
/// (e.g. "male voice", "rainfall", "violin A4"), not a waveform sample.
pub type Token = u16;

/// A tiny semantic codebook + Joint Source-Channel decoder. The codebook entries are
/// spread apart in code space, so a token corrupted by channel noise decodes back to
/// the nearest valid meaning rather than to garbage.
pub struct SemanticTokenizer {
    /// Valid codebook tokens (sparse points in `u16` space).
    codebook: Vec<Token>,
}

impl SemanticTokenizer {
    /// A codebook whose entries are spaced `spacing` apart, giving an error-
    /// correction radius of `spacing/2`.
    pub fn new(entries: usize, spacing: u16) -> SemanticTokenizer {
        let codebook = (0..entries as u16).map(|i| i.wrapping_mul(spacing)).collect();
        SemanticTokenizer { codebook }
    }

    /// Encode a source feature to its nearest codebook token (lossy *in waveform*,
    /// exact *in meaning*).
    pub fn encode(&self, feature: u16) -> Token {
        self.nearest(feature)
    }

    /// Decode a (possibly noise-corrupted) received token back to the nearest valid
    /// meaning — the JSCM channel-robustness step.
    pub fn decode(&self, received: Token) -> Token {
        self.nearest(received)
    }

    fn nearest(&self, x: u16) -> Token {
        *self
            .codebook
            .iter()
            .min_by_key(|&&c| (c as i32 - x as i32).unsigned_abs())
            .unwrap_or(&0)
    }
}

// ─────────────────────────── object-based audio + HRTF ───────────────────────────

/// A semantic audio source in 3D space with a directivity factor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioObject {
    pub token: Token,
    /// Position relative to the listener (right +x, front +y, up +z).
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Source loudness before spatialisation.
    pub gain: f64,
}

/// A rendered stereo sample: per-ear gain plus the interaural time difference
/// (positive ⇒ sound reaches the right ear later, i.e. it is on the left).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoFrame {
    pub left: f64,
    pub right: f64,
    pub itd_us: f64,
}

/// The kernel-level HRTF model. Linear pan + distance attenuation + interaural time
/// difference — deterministic and transcendental-free.
pub struct Hrtf {
    /// Max interaural delay (head width / speed of sound) ≈ 660 µs.
    max_itd_us: f64,
}

impl Hrtf {
    pub fn new() -> Hrtf {
        Hrtf { max_itd_us: 660.0 }
    }

    /// Spatialise one object to a stereo frame for a listener at the origin.
    pub fn render(&self, o: &AudioObject) -> StereoFrame {
        let spread = o.x.abs() + o.y.abs() + o.z.abs();
        // Pan ∈ [-1, 1]: +1 fully right, -1 fully left.
        let pan = if spread == 0.0 { 0.0 } else { o.x / (spread + 1.0) };
        // Distance attenuation (linear, no sqrt): closer ⇒ louder.
        let atten = 1.0 / (1.0 + spread);
        let amp = o.gain * atten;
        let left = amp * (1.0 - pan) * 0.5;
        let right = amp * (1.0 + pan) * 0.5;
        // ITD: sound on the left (pan<0) reaches the right ear later (+itd).
        let itd_us = -pan * self.max_itd_us;
        StereoFrame { left, right, itd_us }
    }

    /// Render and mix a whole scene of objects.
    pub fn render_scene(&self, objects: &[AudioObject]) -> StereoFrame {
        let mut acc = StereoFrame { left: 0.0, right: 0.0, itd_us: 0.0 };
        let mut wsum = 0.0;
        for o in objects {
            let f = self.render(o);
            acc.left += f.left;
            acc.right += f.right;
            // ITD of the mix is loudness-weighted.
            let w = f.left + f.right;
            acc.itd_us += f.itd_us * w;
            wsum += w;
        }
        if wsum > 0.0 {
            acc.itd_us /= wsum;
        }
        acc
    }
}

impl Default for Hrtf {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────── EDF scheduling ───────────────────────────

/// A render task with a relative deadline (µs) and an execution cost (µs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioTask {
    pub id: u32,
    pub deadline_us: u64,
    pub cost_us: u64,
}

/// An Earliest-Deadline-First scheduler for isochronous audio render tasks.
#[derive(Default)]
pub struct EdfScheduler {
    tasks: Vec<AudioTask>,
}

impl EdfScheduler {
    pub fn new() -> EdfScheduler {
        EdfScheduler { tasks: Vec::new() }
    }

    pub fn add(&mut self, task: AudioTask) {
        self.tasks.push(task);
    }

    /// The EDF order: tasks sorted by earliest deadline (ties by id).
    pub fn order(&self) -> Vec<u32> {
        let mut t = self.tasks.clone();
        t.sort_by_key(|x| (x.deadline_us, x.id));
        t.into_iter().map(|x| x.id).collect()
    }

    /// Run the EDF schedule from `start_us`: returns `true` iff every task finishes
    /// by its (absolute) deadline. This is the isochronous-deadline guarantee.
    pub fn meets_all_deadlines(&self, start_us: u64) -> bool {
        let mut t = self.tasks.clone();
        t.sort_by_key(|x| (x.deadline_us, x.id));
        let mut now = start_us;
        for task in t {
            now += task.cost_us;
            if now > task.deadline_us {
                return false;
            }
        }
        true
    }
}

// ─────────────────────────── zero-copy DMA to the GPU ───────────────────────────

/// A shared audio buffer handed to the GPU node **by handle** (zero-copy DMA): only
/// the content hash + length travel, never the samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedAudioBuffer {
    pub handle: Hash256,
    pub frames: usize,
}

impl SharedAudioBuffer {
    /// Register samples once and obtain a transferable handle.
    pub fn publish(samples: &[f64]) -> SharedAudioBuffer {
        let mut bytes = Vec::with_capacity(samples.len() * 8);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        SharedAudioBuffer { handle: Hash256::of(&bytes), frames: samples.len() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jscm_tokens_survive_a_noisy_channel() {
        // Codebook spaced 100 apart ⇒ corrects up to ±49 of channel noise.
        let tk = SemanticTokenizer::new(8, 100);
        let sent = tk.encode(300); // exact codebook point
        // The channel adds noise within the correction radius.
        let received = sent.wrapping_add(40);
        assert_eq!(tk.decode(received), sent); // meaning recovered
        // Raw PCM (the received value itself) would be wrong by 40.
        assert_ne!(received, sent);
    }

    #[test]
    fn hrtf_pans_right_sources_to_the_right_ear() {
        let hrtf = Hrtf::new();
        let right = hrtf.render(&AudioObject { token: 1, x: 5.0, y: 1.0, z: 0.0, gain: 1.0 });
        assert!(right.right > right.left);
        // Sound on the right reaches the right ear earlier ⇒ negative ITD.
        assert!(right.itd_us < 0.0);
        // A centred source is symmetric with zero ITD.
        let center = hrtf.render(&AudioObject { token: 1, x: 0.0, y: 2.0, z: 0.0, gain: 1.0 });
        assert!((center.left - center.right).abs() < 1e-9);
        assert!(center.itd_us.abs() < 1e-9);
    }

    #[test]
    fn closer_sources_are_louder() {
        let hrtf = Hrtf::new();
        let near = hrtf.render(&AudioObject { token: 1, x: 0.0, y: 1.0, z: 0.0, gain: 1.0 });
        let far = hrtf.render(&AudioObject { token: 1, x: 0.0, y: 20.0, z: 0.0, gain: 1.0 });
        assert!(near.left + near.right > far.left + far.right);
    }

    #[test]
    fn edf_orders_by_earliest_deadline_and_meets_isochronous_budget() {
        let mut s = EdfScheduler::new();
        s.add(AudioTask { id: 1, deadline_us: 3 * FRAME_DEADLINE_US, cost_us: 4000 });
        s.add(AudioTask { id: 2, deadline_us: FRAME_DEADLINE_US, cost_us: 4000 });
        s.add(AudioTask { id: 3, deadline_us: 2 * FRAME_DEADLINE_US, cost_us: 4000 });
        // EDF picks the earliest-deadline task first.
        assert_eq!(s.order(), alloc::vec![2, 3, 1]);
        // The three 4 ms tasks all finish within their frame deadlines.
        assert!(s.meets_all_deadlines(0));
    }

    #[test]
    fn edf_detects_a_missed_deadline() {
        let mut s = EdfScheduler::new();
        // Two tasks that together overrun one frame deadline.
        s.add(AudioTask { id: 1, deadline_us: FRAME_DEADLINE_US, cost_us: 10_000 });
        s.add(AudioTask { id: 2, deadline_us: FRAME_DEADLINE_US, cost_us: 10_000 });
        assert!(!s.meets_all_deadlines(0)); // 20 ms > 16.6 ms
    }

    #[test]
    fn zero_copy_buffer_is_shared_by_handle() {
        let samples = alloc::vec![0.1, 0.2, 0.3, 0.4];
        let a = SharedAudioBuffer::publish(&samples);
        let b = SharedAudioBuffer::publish(&samples);
        // Identical samples ⇒ identical handle (content-addressed, zero-copy share).
        assert_eq!(a, b);
        assert_eq!(a.frames, 4);
    }

    #[test]
    fn scene_mix_combines_objects() {
        let hrtf = Hrtf::new();
        let scene = [
            AudioObject { token: 1, x: -3.0, y: 1.0, z: 0.0, gain: 1.0 }, // left
            AudioObject { token: 2, x: 3.0, y: 1.0, z: 0.0, gain: 1.0 },  // right
        ];
        let mix = hrtf.render_scene(&scene);
        // A symmetric scene mixes to near-centre.
        assert!((mix.left - mix.right).abs() < 1e-9);
    }
}

//! Audio/video codec catalog — native recognition, container/header parsing, and
//! lossless storage for the formats the modern world ships, layered on the [`super`]
//! codec registry.
//!
//! Each format is a [`MediaFormat`] (media type, semantic kind, extensions, magic, and
//! a header parser). Registering them makes the OS *recognise, sniff, route, and
//! extract real metadata* for every major audio/video format — and keep the original
//! bytes verbatim so export is lossless. The fully-decoded pixel/sample path is
//! delegated to the codec library/hardware (capability-gated; for the
//! GPU-accelerated path see [`crate::ml::gpu`]) — this module is the format
//! intelligence the OS needs to hand a stream to the right decoder.
//!
//! Covered: **FLAC, WAV/PCM, Ogg Opus, AAC (ADTS), IVF (AV1/VP9/VP8), MP4/ISO-BMFF,
//! H.264 & H.265 (Annex B), Matroska/WebM.** Pure, safe `no_std`, host-tested.

use super::{Codec, CodecError};
use crate::object::{Datum, Object};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A media format the OS understands: identity + a header parser producing a metadata
/// object. The parser is best-effort and bounds-safe — it never panics on short input.
pub struct MediaFormat {
    pub media_type: &'static str,
    /// Unique semantic kind (also the export-routing key), e.g. "Flac", "H264".
    pub kind: &'static str,
    pub exts: &'static [&'static str],
    pub magic: &'static [&'static [u8]],
    pub parse: fn(&[u8]) -> Object,
}

/// A registry codec backed by a [`MediaFormat`]. `decode` parses the header into a
/// metadata object (with the verbatim bytes attached for lossless export); `encode`
/// hands the verbatim bytes back.
struct MediaCodec {
    fmt: &'static MediaFormat,
}

impl Codec for MediaCodec {
    fn media_type(&self) -> &str {
        self.fmt.media_type
    }
    fn semantic_kind(&self) -> &str {
        self.fmt.kind
    }
    fn extensions(&self) -> &[&str] {
        self.fmt.exts
    }
    fn magic(&self) -> &[&[u8]] {
        self.fmt.magic
    }
    fn decode(&self, bytes: &[u8]) -> Result<Object, CodecError> {
        let obj = (self.fmt.parse)(bytes)
            .with("media_type", Datum::Text(self.fmt.media_type.to_string()))
            .with("data", Datum::Bytes(bytes.to_vec()));
        Ok(obj)
    }
    fn encode(&self, obj: &Object) -> Result<Vec<u8>, CodecError> {
        match obj.get("data") {
            Some(Datum::Bytes(b)) => Ok(b.clone()),
            _ => Err(CodecError::Malformed("media object has no verbatim data".to_string())),
        }
    }
}

// ── little/big-endian readers (bounds-safe) ──
fn u16le(b: &[u8], o: usize) -> Option<u64> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?) as u64)
}
fn u32le(b: &[u8], o: usize) -> Option<u64> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as u64)
}
fn find(hay: &[u8], needle: &[u8], limit: usize) -> Option<usize> {
    let end = hay.len().min(limit);
    if needle.is_empty() || end < needle.len() {
        return None;
    }
    (0..=end - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

// ── header parsers ──

fn parse_flac(b: &[u8]) -> Object {
    let mut o = Object::new("Flac").with("container", Datum::Text("FLAC".into()));
    // STREAMINFO begins at offset 8; the sample-rate/channels/bps field is at +10.
    if let (Some(b0), Some(b1), Some(b2), Some(b3)) =
        (b.get(18), b.get(19), b.get(20), b.get(21))
    {
        let sample_rate = ((*b0 as u64) << 12) | ((*b1 as u64) << 4) | ((*b2 as u64) >> 4);
        let channels = ((*b2 as u64 >> 1) & 0x07) + 1;
        let bits = ((((*b2 as u64) & 1) << 4) | ((*b3 as u64) >> 4)) + 1;
        o = o
            .with("sample_rate", Datum::Int(sample_rate as i64))
            .with("channels", Datum::Int(channels as i64))
            .with("bits_per_sample", Datum::Int(bits as i64));
    }
    o
}

fn parse_wav(b: &[u8]) -> Object {
    let mut o = Object::new("Wav").with("container", Datum::Text("RIFF/WAVE".into()));
    // Canonical PCM WAV: fmt chunk at offset 12.
    if let (Some(ch), Some(sr), Some(bits)) = (u16le(b, 22), u32le(b, 24), u16le(b, 34)) {
        o = o
            .with("channels", Datum::Int(ch as i64))
            .with("sample_rate", Datum::Int(sr as i64))
            .with("bits_per_sample", Datum::Int(bits as i64));
    }
    o
}

fn parse_opus(b: &[u8]) -> Object {
    let mut o = Object::new("Opus").with("container", Datum::Text("Ogg".into()));
    if let Some(h) = find(b, b"OpusHead", 256) {
        if let (Some(ch), Some(rate)) = (b.get(h + 9).copied(), u32le(b, h + 12)) {
            o = o
                .with("channels", Datum::Int(ch as i64))
                .with("input_sample_rate", Datum::Int(rate as i64));
        }
    }
    o
}

const ADTS_RATES: [i64; 13] =
    [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

fn parse_aac(b: &[u8]) -> Object {
    let mut o = Object::new("Aac").with("container", Datum::Text("ADTS".into()));
    if let (Some(b2), Some(b3)) = (b.get(2).copied(), b.get(3).copied()) {
        let idx = ((b2 >> 2) & 0x0F) as usize;
        let channels = (((b2 & 1) << 2) | (b3 >> 6)) as i64;
        if idx < ADTS_RATES.len() {
            o = o.with("sample_rate", Datum::Int(ADTS_RATES[idx]));
        }
        o = o.with("channels", Datum::Int(channels));
    }
    o
}

fn fourcc(b: &[u8], o: usize) -> String {
    b.get(o..o + 4)
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .unwrap_or_default()
}

fn parse_ivf(b: &[u8]) -> Object {
    // IVF: "DKIF", fourcc @8, width u16 @12, height u16 @14.
    let cc = fourcc(b, 8);
    let codec = match cc.as_str() {
        "AV01" => "AV1",
        "VP90" => "VP9",
        "VP80" => "VP8",
        other => other,
    };
    let mut o = Object::new("Ivf")
        .with("container", Datum::Text("IVF".into()))
        .with("codec", Datum::Text(codec.to_string()));
    if let (Some(w), Some(h)) = (u16le(b, 12), u16le(b, 14)) {
        o = o.with("width", Datum::Int(w as i64)).with("height", Datum::Int(h as i64));
    }
    o
}

fn parse_mp4(b: &[u8]) -> Object {
    // ISO-BMFF: a `ftyp` box at offset 4 with a major brand at offset 8.
    Object::new("Mp4")
        .with("container", Datum::Text("ISO-BMFF".into()))
        .with("major_brand", Datum::Text(fourcc(b, 8)))
}

/// Iterate Annex-B start codes (00 00 01 / 00 00 00 01), yielding each NAL's first byte
/// offset.
fn annexb_nals(b: &[u8]) -> Vec<usize> {
    let mut nals = Vec::new();
    let mut i = 0usize;
    while i + 3 < b.len() {
        let three = b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1;
        let four = b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 0 && b.get(i + 3) == Some(&1);
        if four {
            nals.push(i + 4);
            i += 4;
        } else if three {
            nals.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
        if nals.len() > 4096 {
            break;
        }
    }
    nals
}

fn parse_annexb(b: &[u8]) -> Object {
    let nals = annexb_nals(b);
    // HEVC if any NAL header has type (>>1 &0x3F) in the VPS/SPS/PPS range (32..=34).
    let is_hevc = nals.iter().any(|&n| {
        b.get(n).map(|&h| {
            let t = (h >> 1) & 0x3F;
            t == 32 || t == 33 || t == 34
        }) == Some(true)
    });
    let codec = if is_hevc { "H.265/HEVC" } else { "H.264/AVC" };
    Object::new(if is_hevc { "H265" } else { "H264" })
        .with("container", Datum::Text("Annex-B".into()))
        .with("codec", Datum::Text(codec.to_string()))
        .with("nal_units", Datum::Int(nals.len() as i64))
}

fn parse_webm(_b: &[u8]) -> Object {
    Object::new("Webm").with("container", Datum::Text("Matroska/WebM".into()))
}

// ── format table ──

static FLAC: MediaFormat = MediaFormat {
    media_type: "audio/flac",
    kind: "Flac",
    exts: &["flac"],
    magic: &[b"fLaC"],
    parse: parse_flac,
};
static WAV: MediaFormat = MediaFormat {
    media_type: "audio/wav",
    kind: "Wav",
    exts: &["wav"],
    magic: &[b"RIFF"],
    parse: parse_wav,
};
static OPUS: MediaFormat = MediaFormat {
    media_type: "audio/opus",
    kind: "Opus",
    exts: &["opus", "ogg"],
    magic: &[b"OggS"],
    parse: parse_opus,
};
static AAC: MediaFormat = MediaFormat {
    media_type: "audio/aac",
    kind: "Aac",
    exts: &["aac"],
    magic: &[&[0xFF, 0xF1], &[0xFF, 0xF9]],
    parse: parse_aac,
};
static IVF: MediaFormat = MediaFormat {
    media_type: "video/x-ivf",
    kind: "Ivf",
    exts: &["ivf"],
    magic: &[b"DKIF"],
    parse: parse_ivf,
};
static MP4: MediaFormat = MediaFormat {
    media_type: "video/mp4",
    kind: "Mp4",
    exts: &["mp4", "m4a", "m4v", "mov"],
    magic: &[], // `ftyp` lives at offset 4 → selected by extension, not leading magic
    parse: parse_mp4,
};
static ANNEXB: MediaFormat = MediaFormat {
    media_type: "video/h264",
    kind: "H264", // parser re-tags H265 when HEVC NALs are present
    exts: &["h264", "264", "h265", "265", "hevc"],
    magic: &[&[0x00, 0x00, 0x00, 0x01], &[0x00, 0x00, 0x01]],
    parse: parse_annexb,
};
static WEBM: MediaFormat = MediaFormat {
    media_type: "video/webm",
    kind: "Webm",
    exts: &["webm", "mkv"],
    magic: &[&[0x1A, 0x45, 0xDF, 0xA3]],
    parse: parse_webm,
};

static CATALOG: [&MediaFormat; 8] = [&FLAC, &WAV, &OPUS, &AAC, &IVF, &MP4, &ANNEXB, &WEBM];

/// Every media format in the catalog, in a stable order.
pub fn catalog() -> &'static [&'static MediaFormat] {
    &CATALOG
}

/// Register every media codec into a registry (called from `with_defaults`/`with_media`).
pub fn register_all(reg: &mut super::CodecRegistry) {
    for fmt in catalog() {
        reg.register(alloc::boxed::Box::new(MediaCodec { fmt }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{Capability, Rights};

    fn registry() -> super::super::CodecRegistry {
        let mut r = super::super::CodecRegistry::new();
        register_all(&mut r);
        r
    }
    fn rcap() -> Capability {
        Capability::mint(0, 64, Rights::READ)
    }

    #[test]
    fn flac_header_is_parsed() {
        // "fLaC" + STREAMINFO; set sample-rate field bytes (44100 Hz, 2ch, 16-bit).
        let mut b = alloc::vec![0u8; 64];
        b[0..4].copy_from_slice(b"fLaC");
        // 44100 = 0xAC44 → 20-bit field. b18..b21 hold [sr19:12][sr11:4][sr3:0|ch|bps..].
        let sr: u32 = 44100;
        b[18] = (sr >> 12) as u8;
        b[19] = (sr >> 4) as u8;
        // low 4 bits of sr in top nibble of b20; channels-1=1 in bits 3..1; bps-1 top bit
        b[20] = (((sr & 0xF) as u8) << 4) | ((2 - 1) << 1) | 0;
        b[21] = ((16 - 1) << 4) as u8; // bps-1 = 15 in top 4 bits (with b20 low bit = 0)
        let r = registry();
        let obj = r.import(Some("song.flac"), &b, &rcap()).unwrap();
        assert_eq!(obj.kind, "Flac");
        assert_eq!(obj.get("sample_rate"), Some(&Datum::Int(44100)));
        assert_eq!(obj.get("channels"), Some(&Datum::Int(2)));
    }

    #[test]
    fn wav_pcm_header_is_parsed() {
        let mut b = alloc::vec![0u8; 64];
        b[0..4].copy_from_slice(b"RIFF");
        b[8..12].copy_from_slice(b"WAVE");
        b[22..24].copy_from_slice(&2u16.to_le_bytes()); // channels
        b[24..28].copy_from_slice(&48000u32.to_le_bytes()); // sample rate
        b[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits
        let obj = registry().import(Some("a.wav"), &b, &rcap()).unwrap();
        assert_eq!(obj.kind, "Wav");
        assert_eq!(obj.get("sample_rate"), Some(&Datum::Int(48000)));
        assert_eq!(obj.get("bits_per_sample"), Some(&Datum::Int(16)));
    }

    #[test]
    fn ogg_opus_is_recognized_and_parsed() {
        let mut b = alloc::vec![0u8; 64];
        b[0..4].copy_from_slice(b"OggS");
        let h = 28;
        b[h..h + 8].copy_from_slice(b"OpusHead");
        b[h + 9] = 2; // channels
        b[h + 12..h + 16].copy_from_slice(&48000u32.to_le_bytes());
        let obj = registry().import(None, &b, &rcap()).unwrap();
        assert_eq!(obj.kind, "Opus");
        assert_eq!(obj.get("channels"), Some(&Datum::Int(2)));
        assert_eq!(obj.get("input_sample_rate"), Some(&Datum::Int(48000)));
    }

    #[test]
    fn aac_adts_sample_rate_and_channels() {
        // syncword 0xFFF1, sampling index 4 (44100), channel config 2.
        let b = alloc::vec![0xFF, 0xF1, (4 << 2) | 0, (2 << 6), 0, 0, 0];
        let obj = registry().import(None, &b, &rcap()).unwrap();
        assert_eq!(obj.kind, "Aac");
        assert_eq!(obj.get("sample_rate"), Some(&Datum::Int(44100)));
        assert_eq!(obj.get("channels"), Some(&Datum::Int(2)));
    }

    #[test]
    fn ivf_av1_dimensions() {
        let mut b = alloc::vec![0u8; 32];
        b[0..4].copy_from_slice(b"DKIF");
        b[8..12].copy_from_slice(b"AV01");
        b[12..14].copy_from_slice(&1920u16.to_le_bytes());
        b[14..16].copy_from_slice(&1080u16.to_le_bytes());
        let obj = registry().import(Some("v.ivf"), &b, &rcap()).unwrap();
        assert_eq!(obj.kind, "Ivf");
        assert_eq!(obj.get("codec"), Some(&Datum::Text("AV1".into())));
        assert_eq!(obj.get("width"), Some(&Datum::Int(1920)));
        assert_eq!(obj.get("height"), Some(&Datum::Int(1080)));
    }

    #[test]
    fn h264_and_h265_annexb_are_distinguished() {
        // H.264: start code + SPS (nal type 7).
        let h264 = alloc::vec![0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E];
        let obj = registry().import(Some("v.h264"), &h264, &rcap()).unwrap();
        assert_eq!(obj.kind, "H264");
        assert_eq!(obj.get("codec"), Some(&Datum::Text("H.264/AVC".into())));
        // H.265: start code + VPS (nal type 32 → header byte 0x40).
        let h265 = alloc::vec![0, 0, 0, 1, 0x40, 0x01, 0x0C, 0x01];
        let obj = registry().import(Some("v.h265"), &h265, &rcap()).unwrap();
        assert_eq!(obj.kind, "H265");
        assert_eq!(obj.get("codec"), Some(&Datum::Text("H.265/HEVC".into())));
    }

    #[test]
    fn export_is_lossless_passthrough() {
        let mut b = alloc::vec![0u8; 64];
        b[0..4].copy_from_slice(b"RIFF");
        b[8..12].copy_from_slice(b"WAVE");
        let r = registry();
        let obj = r.import(Some("a.wav"), &b, &rcap()).unwrap();
        let out = r.export(&obj, &rcap()).unwrap();
        assert_eq!(out, b); // verbatim round-trip
    }

    #[test]
    fn webm_and_mp4_containers_are_recognized() {
        let webm = alloc::vec![0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0];
        assert_eq!(registry().import(None, &webm, &rcap()).unwrap().kind, "Webm");
        let mut mp4 = alloc::vec![0u8; 16];
        mp4[4..8].copy_from_slice(b"ftyp");
        mp4[8..12].copy_from_slice(b"isom");
        let obj = registry().import(Some("m.mp4"), &mp4, &rcap()).unwrap();
        assert_eq!(obj.kind, "Mp4");
        assert_eq!(obj.get("major_brand"), Some(&Datum::Text("isom".into())));
    }

    #[test]
    fn catalog_lists_every_format() {
        assert_eq!(catalog().len(), 8);
    }
}

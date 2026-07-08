//! Codec / Blob registry — keystone **K2** of the integration strategy.
//!
//! The OS has no filesystem of opaque files; it has the content-addressed
//! [`ObjectGraph`](crate::object). To absorb the legacy world we must turn its
//! byte formats into *semantic objects* and back again, losslessly. This module
//! provides exactly that seam:
//!
//! * A [`Codec`] is a `(legacy bytes) ⇄ (semantic Object)` transcoder. Each one
//!   declares the media type it speaks, the file extensions and magic bytes that
//!   identify it, and the semantic `kind` it produces (e.g. `"Image"`).
//! * A [`Blob`] is the lossless fallback: the original bytes kept *verbatim* as a
//!   `Blob` object, so a legacy app can always read the exact bytes back even if
//!   no codec understands them. This is the "two layers, lossless" rule from the
//!   strategy doc — a faithful `Blob` plus a semantic object parsed on demand.
//! * The [`CodecRegistry`] routes by extension / content sniffing and gates every
//!   import and export behind a [`Capability`]: transcoding is a *cell* that
//!   requires authority, matching the Dominion "capability-gated codec" model.
//!
//! Pure, safe, `no_std`, and host-unit-tested — it has no hardware dependency, so
//! it can be proven today and reused by the file system (#3), the Linux
//! personalities (#2), and the legacy browser (#4).

use crate::capability::{CapError, Capability, Rights};
use crate::object::{Datum, Object};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Audio/video codec catalog (FLAC/WAV/Opus/AAC/IVF/MP4/H264/H265/WebM).
pub mod media;

/// The semantic `kind` of a verbatim byte blob.
pub const BLOB_KIND: &str = "Blob";
/// Media type used when nothing more specific is known.
pub const OCTET_STREAM: &str = "application/octet-stream";

/// Why a transcode (or the capability guarding it) was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum CodecError {
    /// No registered codec claims this format, and a Blob was not requested.
    UnknownFormat,
    /// The object's `kind` does not match the codec asked to encode it.
    WrongKind,
    /// The bytes are not valid for this codec (truncated header, bad magic, …).
    Malformed(String),
    /// The presented capability did not authorise the operation.
    Capability(CapError),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::UnknownFormat => f.write_str("codec error: unknown format (no codec, no blob)"),
            CodecError::WrongKind => f.write_str("codec error: object kind does not match codec"),
            CodecError::Malformed(why) => write!(f, "codec error: malformed input ({})", why),
            CodecError::Capability(e) => write!(f, "codec error: {}", e),
        }
    }
}

impl From<CapError> for CodecError {
    fn from(e: CapError) -> Self {
        CodecError::Capability(e)
    }
}

/// Authority check shared by import (READ — pulling external data in) and export
/// (READ — reading the object out as bytes). A cleared tag or missing right traps
/// exactly as a CHERI capability would.
fn require(cap: &Capability, needed: Rights) -> Result<(), CodecError> {
    if !cap.is_valid() {
        return Err(CapError::TagInvalid.into());
    }
    if !cap.rights().contains(needed) {
        return Err(CapError::InsufficientRights.into());
    }
    Ok(())
}

/// A bidirectional transcoder between one legacy byte format and one semantic
/// object kind.
pub trait Codec {
    /// The IANA-style media type this codec speaks (e.g. `"image/x-portable-pixmap"`).
    fn media_type(&self) -> &str;
    /// The semantic object `kind` produced by [`decode`](Codec::decode).
    fn semantic_kind(&self) -> &str;
    /// File extensions (no dot, lowercase) that select this codec.
    fn extensions(&self) -> &[&str] {
        &[]
    }
    /// Leading magic-byte signatures used for content sniffing.
    fn magic(&self) -> &[&[u8]] {
        &[]
    }
    /// Does `bytes` begin with one of this codec's magic signatures?
    fn sniff(&self, bytes: &[u8]) -> bool {
        self.magic().iter().any(|m| !m.is_empty() && bytes.starts_with(m))
    }
    /// Parse legacy bytes into a semantic object.
    fn decode(&self, bytes: &[u8]) -> Result<Object, CodecError>;
    /// Serialize a semantic object back into legacy bytes.
    fn encode(&self, obj: &Object) -> Result<Vec<u8>, CodecError>;
}

/// Helpers for the verbatim-bytes fallback layer.
pub struct Blob;

impl Blob {
    /// Wrap raw bytes as a `Blob` object, tagging the media type so an exporter
    /// can hand the exact bytes back.
    pub fn to_object(media_type: &str, bytes: &[u8]) -> Object {
        Object::new(BLOB_KIND)
            .with("media_type", Datum::Text(media_type.to_string()))
            .with("data", Datum::Bytes(bytes.to_vec()))
    }

    /// Extract the verbatim bytes from a `Blob` object.
    pub fn bytes_of(obj: &Object) -> Option<&[u8]> {
        match obj.get("data") {
            Some(Datum::Bytes(b)) => Some(b),
            _ => None,
        }
    }

    /// The media type recorded on a `Blob` object, if any.
    pub fn media_type_of(obj: &Object) -> Option<&str> {
        match obj.get("media_type") {
            Some(Datum::Text(s)) => Some(s),
            _ => None,
        }
    }
}

/// The registry of known codecs plus the capability-gated import/export seam.
pub struct CodecRegistry {
    codecs: Vec<Box<dyn Codec>>,
}

impl CodecRegistry {
    /// An empty registry (Blob fallback always works regardless).
    pub fn new() -> CodecRegistry {
        CodecRegistry { codecs: Vec::new() }
    }

    /// A registry preloaded with the built-in codecs (UTF-8 text, PPM image).
    pub fn with_defaults() -> CodecRegistry {
        let mut r = CodecRegistry::new();
        r.register(Box::new(TextCodec));
        r.register(Box::new(PpmCodec));
        r
    }

    /// A registry preloaded with the built-in codecs **plus** the full audio/video
    /// media catalog (FLAC, WAV, Opus, AAC, IVF/AV1/VP9, MP4, H.264/H.265, WebM).
    pub fn with_media() -> CodecRegistry {
        let mut r = CodecRegistry::with_defaults();
        media::register_all(&mut r);
        r
    }

    pub fn register(&mut self, codec: Box<dyn Codec>) {
        self.codecs.push(codec);
    }

    pub fn len(&self) -> usize {
        self.codecs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codecs.is_empty()
    }

    /// First codec whose extension list contains `ext` (case-insensitive, dotless).
    pub fn by_extension(&self, ext: &str) -> Option<&dyn Codec> {
        let ext = ext.trim_start_matches('.');
        self.codecs
            .iter()
            .find(|c| c.extensions().iter().any(|e| e.eq_ignore_ascii_case(ext)))
            .map(|c| c.as_ref())
    }

    /// First codec advertising `media_type`.
    pub fn by_media_type(&self, media_type: &str) -> Option<&dyn Codec> {
        self.codecs
            .iter()
            .find(|c| c.media_type() == media_type)
            .map(|c| c.as_ref())
    }

    /// First codec whose magic bytes match the start of `bytes` (content sniffing).
    pub fn sniff(&self, bytes: &[u8]) -> Option<&dyn Codec> {
        self.codecs.iter().find(|c| c.sniff(bytes)).map(|c| c.as_ref())
    }

    /// First codec producing the semantic `kind` — used to pick an encoder for an
    /// object already living in the graph.
    pub fn by_kind(&self, kind: &str) -> Option<&dyn Codec> {
        self.codecs
            .iter()
            .find(|c| c.semantic_kind() == kind)
            .map(|c| c.as_ref())
    }

    /// Choose a codec for `bytes`: prefer the filename's extension, then fall back
    /// to content sniffing. `name` may be a filename or a bare extension.
    pub fn pick(&self, name: Option<&str>, bytes: &[u8]) -> Option<&dyn Codec> {
        if let Some(name) = name {
            let ext = name.rsplit('.').next().unwrap_or(name);
            if let Some(c) = self.by_extension(ext) {
                return Some(c);
            }
        }
        self.sniff(bytes)
    }

    /// Import legacy bytes into a semantic object. Picks a codec by name/sniff; if
    /// none applies the bytes are preserved verbatim as a `Blob` (never lossy).
    /// Requires `READ` authority.
    pub fn import(&self, name: Option<&str>, bytes: &[u8], cap: &Capability) -> Result<Object, CodecError> {
        require(cap, Rights::READ)?;
        match self.pick(name, bytes) {
            Some(codec) => codec.decode(bytes),
            None => Ok(Blob::to_object(OCTET_STREAM, bytes)),
        }
    }

    /// Strict import: like [`import`](Self::import) but errors instead of falling
    /// back to a Blob when no codec understands the bytes.
    pub fn import_strict(&self, name: Option<&str>, bytes: &[u8], cap: &Capability) -> Result<Object, CodecError> {
        require(cap, Rights::READ)?;
        match self.pick(name, bytes) {
            Some(codec) => codec.decode(bytes),
            None => Err(CodecError::UnknownFormat),
        }
    }

    /// Export a semantic object back to legacy bytes. A `Blob` yields its verbatim
    /// bytes; any other kind is routed to the codec that speaks it. Requires
    /// `READ` authority.
    pub fn export(&self, obj: &Object, cap: &Capability) -> Result<Vec<u8>, CodecError> {
        require(cap, Rights::READ)?;
        if obj.kind == BLOB_KIND {
            return Blob::bytes_of(obj)
                .map(|b| b.to_vec())
                .ok_or_else(|| CodecError::Malformed("blob has no data field".to_string()));
        }
        match self.by_kind(&obj.kind) {
            Some(codec) => codec.encode(obj),
            None => Err(CodecError::UnknownFormat),
        }
    }
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Built-in codecs
// ---------------------------------------------------------------------------

/// UTF-8 plain text ⇄ `Text { content }`.
pub struct TextCodec;

impl Codec for TextCodec {
    fn media_type(&self) -> &str {
        "text/plain"
    }
    fn semantic_kind(&self) -> &str {
        "Text"
    }
    fn extensions(&self) -> &[&str] {
        &["txt", "md", "log", "cfg", "conf"]
    }
    fn decode(&self, bytes: &[u8]) -> Result<Object, CodecError> {
        let s = core::str::from_utf8(bytes).map_err(|_| CodecError::Malformed("not valid UTF-8".to_string()))?;
        Ok(Object::new("Text").with("content", Datum::Text(s.to_string())))
    }
    fn encode(&self, obj: &Object) -> Result<Vec<u8>, CodecError> {
        if obj.kind != "Text" {
            return Err(CodecError::WrongKind);
        }
        match obj.get("content") {
            Some(Datum::Text(s)) => Ok(s.as_bytes().to_vec()),
            _ => Err(CodecError::Malformed("Text object missing 'content'".to_string())),
        }
    }
}

/// Binary Netpbm PPM (`P6`) ⇄ `Image { width, height, channels, pixels }`.
///
/// A small but genuine raster format: header `P6 <w> <h> <maxval>` then raw RGB
/// bytes. Demonstrates round-tripping real pixel data through the semantic layer.
pub struct PpmCodec;

impl PpmCodec {
    /// Read one ASCII unsigned integer from `bytes` starting at `*pos`, skipping
    /// leading whitespace and `#`-comments. Advances `*pos` past the number.
    fn read_uint(bytes: &[u8], pos: &mut usize) -> Result<u32, CodecError> {
        // Skip whitespace and comment lines.
        loop {
            while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            if *pos < bytes.len() && bytes[*pos] == b'#' {
                while *pos < bytes.len() && bytes[*pos] != b'\n' {
                    *pos += 1;
                }
            } else {
                break;
            }
        }
        let start = *pos;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
        if *pos == start {
            return Err(CodecError::Malformed("expected an integer in PPM header".to_string()));
        }
        let mut v: u32 = 0;
        for &b in &bytes[start..*pos] {
            v = v
                .checked_mul(10)
                .and_then(|v| v.checked_add((b - b'0') as u32))
                .ok_or_else(|| CodecError::Malformed("PPM header integer overflow".to_string()))?;
        }
        Ok(v)
    }
}

impl Codec for PpmCodec {
    fn media_type(&self) -> &str {
        "image/x-portable-pixmap"
    }
    fn semantic_kind(&self) -> &str {
        "Image"
    }
    fn extensions(&self) -> &[&str] {
        &["ppm"]
    }
    fn magic(&self) -> &[&[u8]] {
        const M: &[&[u8]] = &[b"P6"];
        M
    }
    fn decode(&self, bytes: &[u8]) -> Result<Object, CodecError> {
        if !bytes.starts_with(b"P6") {
            return Err(CodecError::Malformed("not a P6 PPM (bad magic)".to_string()));
        }
        let mut pos = 2usize;
        let width = Self::read_uint(bytes, &mut pos)?;
        let height = Self::read_uint(bytes, &mut pos)?;
        let maxval = Self::read_uint(bytes, &mut pos)?;
        if maxval == 0 || maxval > 255 {
            return Err(CodecError::Malformed("only 8-bit PPM (maxval 1..=255) supported".to_string()));
        }
        // Exactly one whitespace byte separates the header from the pixel data.
        if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
            return Err(CodecError::Malformed("missing whitespace before PPM pixel data".to_string()));
        }
        pos += 1;
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| CodecError::Malformed("PPM dimensions overflow".to_string()))?;
        let pixels = &bytes[pos..];
        if pixels.len() < need {
            return Err(CodecError::Malformed("PPM pixel data truncated".to_string()));
        }
        Ok(Object::new("Image")
            .with("width", Datum::Int(width as i64))
            .with("height", Datum::Int(height as i64))
            .with("channels", Datum::Int(3))
            .with("pixels", Datum::Bytes(pixels[..need].to_vec())))
    }
    fn encode(&self, obj: &Object) -> Result<Vec<u8>, CodecError> {
        if obj.kind != "Image" {
            return Err(CodecError::WrongKind);
        }
        let width = match obj.get("width") {
            Some(Datum::Int(w)) if *w >= 0 => *w as u32,
            _ => return Err(CodecError::Malformed("Image missing non-negative 'width'".to_string())),
        };
        let height = match obj.get("height") {
            Some(Datum::Int(h)) if *h >= 0 => *h as u32,
            _ => return Err(CodecError::Malformed("Image missing non-negative 'height'".to_string())),
        };
        let pixels = match obj.get("pixels") {
            Some(Datum::Bytes(p)) => p,
            _ => return Err(CodecError::Malformed("Image missing 'pixels'".to_string())),
        };
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| CodecError::Malformed("Image dimensions overflow".to_string()))?;
        if pixels.len() != need {
            return Err(CodecError::Malformed("Image pixel length does not match dimensions".to_string()));
        }
        // Header: "P6\n<w> <h>\n255\n" then raw RGB.
        let mut out = Vec::with_capacity(need + 32);
        out.extend_from_slice(b"P6\n");
        push_uint(&mut out, width);
        out.push(b' ');
        push_uint(&mut out, height);
        out.extend_from_slice(b"\n255\n");
        out.extend_from_slice(pixels);
        Ok(out)
    }
}

/// Append the base-10 ASCII of `v` to `out` (no_std-friendly, no `format!`).
fn push_uint(out: &mut Vec<u8>, v: u32) {
    if v == 0 {
        out.push(b'0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    let mut n = v;
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&buf[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Rights;
    use alloc::vec;

    fn read_cap() -> Capability {
        Capability::mint(0, 0x1000, Rights::READ)
    }

    fn write_only_cap() -> Capability {
        Capability::mint(0, 0x1000, Rights::WRITE)
    }

    #[test]
    fn text_round_trips() {
        let reg = CodecRegistry::with_defaults();
        let cap = read_cap();
        let obj = reg.import(Some("notes.txt"), b"hello \xe2\x98\x83 world", &cap).unwrap();
        assert_eq!(obj.kind, "Text");
        let back = reg.export(&obj, &cap).unwrap();
        assert_eq!(back, b"hello \xe2\x98\x83 world");
    }

    #[test]
    fn text_rejects_non_utf8() {
        let codec = TextCodec;
        let err = codec.decode(&[0xff, 0xfe, 0x00]).unwrap_err();
        assert!(matches!(err, CodecError::Malformed(_)));
    }

    #[test]
    fn ppm_round_trips_pixels() {
        // 2x1 image: red pixel then green pixel.
        let raw = b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let reg = CodecRegistry::with_defaults();
        let cap = read_cap();
        let img = reg.import(Some("pic.ppm"), raw, &cap).unwrap();
        assert_eq!(img.kind, "Image");
        assert_eq!(img.get("width"), Some(&Datum::Int(2)));
        assert_eq!(img.get("height"), Some(&Datum::Int(1)));
        let back = reg.export(&img, &cap).unwrap();
        // Re-decoding the exported bytes yields the identical object (lossless).
        let img2 = reg.import(Some("pic.ppm"), &back, &cap).unwrap();
        assert_eq!(img.id(), img2.id());
    }

    #[test]
    fn ppm_selected_by_sniffing_without_name() {
        let raw = b"P6 1 1 255 \x10\x20\x30";
        let reg = CodecRegistry::with_defaults();
        let codec = reg.sniff(raw).expect("should sniff P6");
        assert_eq!(codec.semantic_kind(), "Image");
    }

    #[test]
    fn ppm_rejects_truncated_pixels() {
        let raw = b"P6\n2 2\n255\n\xff\x00"; // claims 12 bytes, has 2
        let codec = PpmCodec;
        assert!(matches!(codec.decode(raw).unwrap_err(), CodecError::Malformed(_)));
    }

    #[test]
    fn ppm_handles_header_comments() {
        let raw = b"P6\n# a comment\n1 1\n255\n\x01\x02\x03";
        let codec = PpmCodec;
        let img = codec.decode(raw).unwrap();
        assert_eq!(img.get("pixels"), Some(&Datum::Bytes(vec![1, 2, 3])));
    }

    #[test]
    fn unknown_format_falls_back_to_blob() {
        let reg = CodecRegistry::with_defaults();
        let cap = read_cap();
        let weird = &[0u8, 159, 146, 150, 4, 4];
        let obj = reg.import(Some("mystery.bin"), weird, &cap).unwrap();
        assert_eq!(obj.kind, BLOB_KIND);
        // Verbatim bytes survive the round trip exactly.
        let back = reg.export(&obj, &cap).unwrap();
        assert_eq!(back, weird);
    }

    #[test]
    fn import_strict_errors_on_unknown() {
        let reg = CodecRegistry::with_defaults();
        let cap = read_cap();
        let err = reg.import_strict(Some("mystery.bin"), &[0u8, 1, 2], &cap).unwrap_err();
        assert_eq!(err, CodecError::UnknownFormat);
    }

    #[test]
    fn import_requires_read_capability() {
        let reg = CodecRegistry::with_defaults();
        let err = reg.import(Some("a.txt"), b"hi", &write_only_cap()).unwrap_err();
        assert_eq!(err, CodecError::Capability(CapError::InsufficientRights));
    }

    #[test]
    fn tampered_capability_traps() {
        let reg = CodecRegistry::with_defaults();
        let cap = read_cap().tamper();
        let err = reg.import(Some("a.txt"), b"hi", &cap).unwrap_err();
        assert_eq!(err, CodecError::Capability(CapError::TagInvalid));
    }

    #[test]
    fn export_wrong_codec_kind_errors() {
        let codec = TextCodec;
        let img = Object::new("Image");
        assert_eq!(codec.encode(&img).unwrap_err(), CodecError::WrongKind);
    }

    #[test]
    fn blob_helpers_round_trip() {
        let b = Blob::to_object("application/x-thing", &[9, 8, 7]);
        assert_eq!(Blob::bytes_of(&b), Some(&[9u8, 8, 7][..]));
        assert_eq!(Blob::media_type_of(&b), Some("application/x-thing"));
    }

    #[test]
    fn registry_lookup_paths() {
        let reg = CodecRegistry::with_defaults();
        assert!(reg.by_extension("ppm").is_some());
        assert!(reg.by_extension(".PPM").is_some()); // case + dot insensitive
        assert!(reg.by_media_type("text/plain").is_some());
        assert!(reg.by_kind("Image").is_some());
        assert!(reg.by_extension("xyz").is_none());
        assert_eq!(reg.len(), 2);
    }
}

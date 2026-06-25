//! Active defense & provenance — **Stage 6** (see
//! `docs/architecture/07-stage-06-active-defense-provenance.md`).
//!
//! Generative storage keeps media as a compressed **latent**. Stage 6 turns that
//! latent into an *active* defense: a **SLIC** watermark is woven into the latent as
//! a keyed perturbation, and the media is **cryptographically poisoned** so that any
//! **unauthorized re-compression** (a re-encode by someone without the key) maps the
//! hidden secret to a **visible artifact** and the content **self-degrades** —
//! tampered media destroys itself on re-encode rather than leaking cleanly.
//!
//! Modeled in safe Rust over the [`crate::neural`] codec: the latent is the
//! compressed blob, the "adversarial perturbation" is a SHA-256 keystream, and an
//! unauthorized re-encode is a lossy transform applied without the key. The
//! *semantics* — authorized recovery works, unauthorized re-encode breaks both the
//! watermark and the content — are exercised and tested. Pure, safe `no_std`.

use crate::hash::Hash256;
use crate::neural::{compress, decompress};
use alloc::vec::Vec;

/// A SHA-256 counter-mode keystream XOR — the keyed "adversarial perturbation"
/// applied to the latent. With the key it is reversible; without it the carrier is
/// opaque.
fn keyed_xor(data: &[u8], key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for (counter, chunk) in data.chunks(32).enumerate() {
        let mut input = Vec::with_capacity(key.len() + 8);
        input.extend_from_slice(key);
        input.extend_from_slice(&(counter as u64).to_le_bytes());
        let ks = Hash256::of(&input).0;
        for (i, &b) in chunk.iter().enumerate() {
            out.push(b ^ ks[i]);
        }
    }
    out
}

/// Media protected by the active-defense layer: an opaque carrier (the perturbed
/// latent), the embedded watermark tag, and the original content hash used to detect
/// self-degradation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtectedMedia {
    carrier: Vec<u8>,
    watermark: Hash256,
    original_hash: Hash256,
}

impl ProtectedMedia {
    /// The bytes as they sit at rest — the perturbed latent, opaque without the key.
    pub fn carrier(&self) -> &[u8] {
        &self.carrier
    }
}

/// Compute the SLIC watermark tag binding the key, the latent, and the secret
/// payload — the perturbation a recompressor cannot reproduce without the key.
fn watermark_tag(key: &[u8], latent: &[u8], payload: &[u8]) -> Hash256 {
    let mut input = Vec::with_capacity(key.len() + latent.len() + payload.len() + 8);
    input.extend_from_slice(b"slic:");
    input.extend_from_slice(key);
    input.extend_from_slice(latent);
    input.extend_from_slice(payload);
    Hash256::of(&input)
}

/// **Protect** `content`: compress it to a latent, embed the SLIC `watermark`
/// payload, and store the keyed-perturbed carrier. Only a key holder can recover the
/// content or verify the watermark.
pub fn protect(content: &[u8], key: &[u8], watermark: &[u8]) -> ProtectedMedia {
    let latent = compress(content);
    ProtectedMedia {
        carrier: keyed_xor(&latent, key),
        watermark: watermark_tag(key, &latent, watermark),
        original_hash: Hash256::of(content),
    }
}

/// **Recover** the original content — only with the key, and only if the media has
/// not self-degraded. Returns `None` if the key is wrong or the content was
/// re-encoded without authorization (the poison fired).
pub fn recover(media: &ProtectedMedia, key: &[u8]) -> Option<Vec<u8>> {
    let latent = keyed_xor(&media.carrier, key);
    let content = decompress(&latent);
    if Hash256::of(&content) == media.original_hash {
        Some(content)
    } else {
        None
    }
}

/// **Verify** that the SLIC watermark is intact for `payload` under `key`. An
/// unauthorized re-encode perturbs the latent and breaks this — tamper-evidence.
pub fn watermark_intact(media: &ProtectedMedia, key: &[u8], payload: &[u8]) -> bool {
    let latent = keyed_xor(&media.carrier, key);
    watermark_tag(key, &latent, payload) == media.watermark
}

/// Model an **unauthorized re-compression**: an attacker without the key treats the
/// opaque carrier as raw bytes and re-encodes it (a lossy transform here:
/// re-compress, then quantize away the low nibble). Because they lack the key, the
/// perturbed latent is mangled — the watermark breaks and recovery degrades. This is
/// the cryptographic-poisoning / self-degradation property: tampered media destroys
/// itself on re-encode.
pub fn unauthorized_reencode(media: &ProtectedMedia) -> ProtectedMedia {
    // A lossy re-encode of the carrier without understanding it.
    let recompressed = compress(&decompress(&compress(media.carrier())));
    let degraded: Vec<u8> = recompressed.iter().map(|b| b & 0xF0).collect();
    ProtectedMedia {
        carrier: degraded,
        watermark: media.watermark,
        original_hash: media.original_hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTENT: &[u8] = b"a frame of generatively-stored video, semantically compressed";
    const KEY: &[u8] = b"owner-capability-key";
    const PAYLOAD: &[u8] = b"owner=jayden;license=personal";

    #[test]
    fn authorized_holder_recovers_and_watermark_verifies() {
        let media = protect(CONTENT, KEY, PAYLOAD);
        // At rest the carrier is opaque (not the content).
        assert_ne!(media.carrier(), CONTENT);
        // With the key: content recovers and the watermark is intact.
        assert_eq!(recover(&media, KEY).as_deref(), Some(CONTENT));
        assert!(watermark_intact(&media, KEY, PAYLOAD));
    }

    #[test]
    fn wrong_key_recovers_nothing() {
        let media = protect(CONTENT, KEY, PAYLOAD);
        assert!(recover(&media, b"attacker-key").is_none());
        assert!(!watermark_intact(&media, b"attacker-key", PAYLOAD));
    }

    #[test]
    fn wrong_watermark_payload_is_rejected() {
        let media = protect(CONTENT, KEY, PAYLOAD);
        assert!(!watermark_intact(&media, KEY, b"owner=mallory;license=pirate"));
    }

    #[test]
    fn unauthorized_reencode_self_degrades_and_breaks_watermark() {
        let media = protect(CONTENT, KEY, PAYLOAD);
        // An attacker re-encodes without the key …
        let tampered = unauthorized_reencode(&media);
        // … the content destroys itself (recovery with the real key now fails) …
        assert!(recover(&tampered, KEY).is_none());
        // … and the SLIC watermark no longer verifies (tamper-evident).
        assert!(!watermark_intact(&tampered, KEY, PAYLOAD));
    }
}

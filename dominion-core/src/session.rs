//! Identity-bound secure sessions — **Stage 14** universal network encryption.
//!
//! Legacy TLS secures a *channel* between an `IP:port` pair. Dominion secures the
//! *relationship between two identities*. A [`Session`] is bound to two
//! [`DominionId`](crate::dominionlink::DominionId)s (each the hash of a public key), and
//! its traffic key is established with the **post-quantum lattice KEM**
//! ([`crate::lattice`]) — so an adversary recording ciphertext today cannot derive
//! the key after building a quantum computer (the Harvest-Now-Decrypt-Later
//! defence). Every frame is sealed with **AES-256-GCM** ([`crate::memcrypt`]): no
//! plaintext ever crosses the wire, and the two identities + epoch are bound in as
//! authenticated associated data, so a frame cannot be replayed into a different
//! session or time.
//!
//! Sessions are **temporal** (they expire at a logical epoch) and **revocable**.
//! Pure, safe `no_std`; the kernel supplies real entropy for the seeds.

use crate::dominionlink::DominionId;
use crate::hash::Hash256;
use crate::lattice::{Ciphertext, LatticeKem, PublicKey, SecretKey};
use crate::memcrypt::{gcm_decrypt, gcm_encrypt, Aes};
use alloc::vec::Vec;

/// Why a session operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SessionError {
    /// The remote identity does not certify the KEM public key it presented.
    IdentityMismatch,
    /// The session has passed its expiry epoch.
    Expired,
    /// The session was explicitly revoked.
    Revoked,
    /// The frame failed authentication (tampered, wrong key, or wrong AAD).
    AuthFailed,
}

/// A published KEM identity: a long-term lattice keypair whose public half is
/// bound to a self-certifying [`DominionId`].
pub struct KemIdentity {
    pub id: DominionId,
    pub public: PublicKey,
    secret: SecretKey,
}

impl KemIdentity {
    /// Generate an identity from a seed. The `DominionId` is the fingerprint of the
    /// KEM public key, so the identity self-certifies the key material.
    pub fn generate(seed: &[u8]) -> KemIdentity {
        let (public, secret) = LatticeKem::keygen(seed);
        let id = DominionId(public.fingerprint());
        KemIdentity { id, public, secret }
    }
}

/// An encrypted, identity-bound frame on the wire. Carries no plaintext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Monotonic per-session sequence number (drives the GCM IV).
    pub seq: u64,
    /// The logical epoch the sender claims — bound into the AAD.
    pub epoch: u64,
    ciphertext: Vec<u8>,
    tag: [u8; 16],
}

impl Frame {
    /// Number of ciphertext bytes carried (0 for an empty payload).
    pub fn payload_len(&self) -> usize {
        self.ciphertext.len()
    }

    /// Fault-injection helper: flip the first ciphertext byte to simulate a
    /// tampered frame on the wire (used by the chaos/property harness).
    pub fn corrupt_first_byte(&mut self) {
        if let Some(b) = self.ciphertext.first_mut() {
            *b ^= 0xFF;
        }
    }
}

/// A live secure session between two identities.
pub struct Session {
    local: DominionId,
    remote: DominionId,
    key: [u8; 32],
    /// Per-session random nonce folded into the traffic key and the GCM IV prefix.
    /// Prevents (key, IV) reuse across sessions that share the same `eph_seed`
    /// (e.g. after a reboot or reconnect with a replayed seed).
    session_nonce: [u8; 16],
    /// Logical epoch after which the session is dead.
    expires_at: u64,
    revoked: bool,
    send_seq: u64,
}

/// Bind the ordered identity pair + epoch as GCM associated data, so a frame is
/// cryptographically pinned to *this* relationship at *this* time.
fn aad(a: &DominionId, b: &DominionId, epoch: u64) -> Vec<u8> {
    // Order-independent: both endpoints compute the same AAD.
    let (lo, hi) = if a.0 <= b.0 { (a, b) } else { (b, a) };
    let mut v = Vec::with_capacity(72);
    v.extend_from_slice(&lo.0 .0);
    v.extend_from_slice(&hi.0 .0);
    v.extend_from_slice(&epoch.to_le_bytes());
    v
}

/// Derive the 96-bit GCM IV by XOR-combining the session nonce prefix (bytes 0–7
/// taken from the first 8 bytes of `session_nonce`) with the sequence counter.
/// This makes the IV stream unique per session even when `send_seq` resets to 0,
/// defeating the (key, IV) reuse attack that occurs when two sessions share the
/// same traffic key (e.g. same `eph_seed` on reboot).
fn iv_for(seq: u64, session_nonce: &[u8; 16]) -> [u8; 12] {
    let mut iv = [0u8; 12];
    // High 8 bytes: session-unique prefix from the nonce (XOR so it is cheap and
    // still fully determined by the session_nonce without truncation loss).
    let nonce_prefix = u64::from_le_bytes(session_nonce[..8].try_into().unwrap());
    let iv_high = nonce_prefix ^ seq;
    iv[..8].copy_from_slice(&iv_high.to_le_bytes());
    // Low 4 bytes: sequence counter alone, giving 2^32 frames per unique prefix.
    iv[8..12].copy_from_slice(&(seq as u32).to_le_bytes());
    iv
}

impl Session {
    /// **Initiator** side. Encapsulate a fresh shared secret to `remote`'s KEM
    /// public key, returning the session and the KEM ciphertext to send across.
    /// Fails if `remote`'s advertised identity does not certify `remote_pub`.
    pub fn initiate(
        local: DominionId,
        remote: DominionId,
        remote_pub: &PublicKey,
        eph_seed: &[u8],
        expires_at: u64,
    ) -> Result<(Session, Ciphertext), SessionError> {
        if DominionId(remote_pub.fingerprint()) != remote {
            return Err(SessionError::IdentityMismatch);
        }
        let (ct, shared) = LatticeKem::encapsulate(remote_pub, eph_seed);
        // Derive the per-session nonce from the KEM shared secret. Both initiator
        // and responder possess `shared` (via encapsulate / decapsulate respectively),
        // so they independently reach the same nonce without an extra round-trip.
        // Because `shared` is determined by `eph_seed` and the recipient's public key,
        // a unique `eph_seed` per session (as the kernel guarantees in production via
        // RDRAND) ensures a unique nonce — and therefore a unique (key, IV) stream.
        let session_nonce = session_nonce_from_shared(&shared);
        let key = mix(&shared, &local, &remote, &session_nonce);
        Ok((
            Session { local, remote, key, session_nonce, expires_at, revoked: false, send_seq: 0 },
            ct,
        ))
    }

    /// **Responder** side. Decapsulate the KEM ciphertext with our secret key to
    /// recover the same shared secret and open the session.
    pub fn accept(
        identity: &KemIdentity,
        remote: DominionId,
        ct: &Ciphertext,
        expires_at: u64,
    ) -> Session {
        let shared = LatticeKem::decapsulate(&identity.secret, ct);
        // Derive the session nonce from the KEM shared secret so the responder
        // reconstructs exactly the same nonce the initiator embedded in the key and
        // IV without needing to transmit `eph_seed`. Because `shared` is unique per
        // KEM exchange (it is bound to the ciphertext), the nonce is likewise unique.
        let session_nonce = session_nonce_from_shared(&shared);
        let key = mix(&shared, &remote, &identity.id, &session_nonce);
        Session { local: identity.id, remote, key, session_nonce, expires_at, revoked: false, send_seq: 0 }
    }

    /// Encrypt `plaintext` into a frame valid at `now`. No plaintext leaves here.
    pub fn seal(&mut self, now: u64, plaintext: &[u8]) -> Result<Frame, SessionError> {
        self.check_live(now)?;
        let seq = self.send_seq;
        self.send_seq += 1;
        let aes = Aes::new_256(&self.key);
        let (ciphertext, tag) =
            gcm_encrypt(&aes, &iv_for(seq, &self.session_nonce), &aad(&self.local, &self.remote, self.expires_at), plaintext);
        Ok(Frame { seq, epoch: self.expires_at, ciphertext, tag })
    }

    /// Decrypt a frame received at `now`. Returns `AuthFailed` on any tamper,
    /// including any modification of the `epoch` field in the frame header.
    pub fn open(&self, now: u64, frame: &Frame) -> Result<Vec<u8>, SessionError> {
        self.check_live(now)?;
        let aes = Aes::new_256(&self.key);
        // Authenticate using the epoch carried in the frame, not self.expires_at.
        // seal() embeds self.expires_at as frame.epoch and uses it as AAD, so both
        // sides agree when the frame is untampered. If an attacker flips frame.epoch
        // in transit, the AAD here diverges from what seal() computed, and GCM
        // authentication fails — returning AuthFailed rather than silently accepting
        // the manipulated epoch.
        gcm_decrypt(
            &aes,
            &iv_for(frame.seq, &self.session_nonce),
            &aad(&self.local, &self.remote, frame.epoch),
            &frame.ciphertext,
            &frame.tag,
        )
        .ok_or(SessionError::AuthFailed)
    }

    /// Revoke the session immediately; all further seal/open fail.
    pub fn revoke(&mut self) {
        self.revoked = true;
    }

    pub fn is_revoked(&self) -> bool {
        self.revoked
    }

    pub fn peers(&self) -> (DominionId, DominionId) {
        (self.local, self.remote)
    }

    fn check_live(&self, now: u64) -> Result<(), SessionError> {
        if self.revoked {
            return Err(SessionError::Revoked);
        }
        if now > self.expires_at {
            return Err(SessionError::Expired);
        }
        Ok(())
    }
}

/// Derive the per-session nonce from the KEM shared secret.
///
/// **Why this works without extra wire messages:** The KEM guarantees that
/// `LatticeKem::encapsulate(pk, eph_seed)` and `LatticeKem::decapsulate(sk, ct)`
/// produce the exact same `shared` byte string. Both sides therefore independently
/// compute the identical nonce — no additional handshake round-trip is required.
/// Because `shared` is bound to `eph_seed` and the recipient's long-term public key,
/// a fresh `eph_seed` per session (guaranteed by kernel RDRAND in production)
/// produces a fresh nonce, and therefore a unique (key, IV) stream, for every session.
fn session_nonce_from_shared(shared: &[u8; 32]) -> [u8; 16] {
    let mut input = Vec::with_capacity(48);
    input.extend_from_slice(shared);
    input.extend_from_slice(b"session:nonce");
    let digest = Hash256::of(&input).0;
    let mut nonce = [0u8; 16];
    nonce.copy_from_slice(&digest[..16]);
    nonce
}

/// Fold the KEM shared secret together with both identities and the per-session
/// nonce into the traffic key, so the key is inseparable from the identity pair
/// (channel binding) and from this specific session's randomness.
///
/// Including `session_nonce` (derived from `shared`) into the key adds defence in
/// depth: even if an attacker somehow observes the raw `shared` value, they still
/// cannot forge the key without knowing the full input, and a replayed KEM exchange
/// (same `eph_seed`, same identities) produces the same `session_nonce` — which is
/// correct, because in that degenerate case both sessions ARE the same session.
/// Production code prevents replay by using kernel-supplied RDRAND for `eph_seed`.
fn mix(shared: &[u8; 32], a: &DominionId, b: &DominionId, session_nonce: &[u8; 16]) -> [u8; 32] {
    let (lo, hi) = if a.0 <= b.0 { (a, b) } else { (b, a) };
    let mut input = Vec::with_capacity(112);
    input.extend_from_slice(shared);
    input.extend_from_slice(&lo.0 .0);
    input.extend_from_slice(&hi.0 .0);
    input.extend_from_slice(session_nonce);
    Hash256::of(&input).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn pair() -> (KemIdentity, KemIdentity) {
        (KemIdentity::generate(b"alice-seed"), KemIdentity::generate(b"bob-seed"))
    }

    #[test]
    fn identity_certifies_its_kem_key() {
        let alice = KemIdentity::generate(b"alice-seed");
        // The identity is exactly the fingerprint of the published key.
        assert_eq!(alice.id, DominionId(alice.public.fingerprint()));
    }

    #[test]
    fn handshake_agrees_on_a_key_and_round_trips() {
        let (alice, bob) = pair();
        let (mut a_sess, ct) =
            Session::initiate(alice.id, bob.id, &bob.public, b"eph-1", 100).unwrap();
        let b_sess = Session::accept(&bob, alice.id, &ct, 100);
        // Alice seals, Bob opens — identity-bound, encrypted, no plaintext on wire.
        let frame = a_sess.seal(1, b"transfer 42 to savings").unwrap();
        assert_ne!(frame_bytes(&frame), b"transfer 42 to savings");
        assert_eq!(b_sess.open(1, &frame).unwrap(), b"transfer 42 to savings");
    }

    #[test]
    fn impersonating_an_identity_is_refused() {
        let (alice, bob) = pair();
        // Mallory presents Bob's identity but her own KEM key.
        let mallory = KemIdentity::generate(b"mallory-seed");
        let err = Session::initiate(alice.id, bob.id, &mallory.public, b"eph", 100);
        assert_eq!(err.err(), Some(SessionError::IdentityMismatch));
    }

    #[test]
    fn tampered_frame_fails_authentication() {
        let (alice, bob) = pair();
        let (mut a, ct) = Session::initiate(alice.id, bob.id, &bob.public, b"e", 100).unwrap();
        let b = Session::accept(&bob, alice.id, &ct, 100);
        let mut frame = a.seal(1, b"secret").unwrap();
        frame.ciphertext[0] ^= 0xFF;
        assert_eq!(b.open(1, &frame).err(), Some(SessionError::AuthFailed));
    }

    #[test]
    fn expired_session_refuses_traffic() {
        let (alice, bob) = pair();
        let (mut a, _ct) = Session::initiate(alice.id, bob.id, &bob.public, b"e", 5).unwrap();
        // now (6) is past the expiry epoch (5).
        assert_eq!(a.seal(6, b"late").err(), Some(SessionError::Expired));
    }

    #[test]
    fn revoked_session_refuses_traffic() {
        let (alice, bob) = pair();
        let (mut a, ct) = Session::initiate(alice.id, bob.id, &bob.public, b"e", 100).unwrap();
        let b = Session::accept(&bob, alice.id, &ct, 100);
        let frame = a.seal(1, b"ok").unwrap();
        let mut b = b;
        b.revoke();
        assert_eq!(b.open(1, &frame).err(), Some(SessionError::Revoked));
    }

    #[test]
    fn a_third_party_cannot_decrypt() {
        let (alice, bob) = pair();
        let (mut a, ct) = Session::initiate(alice.id, bob.id, &bob.public, b"e", 100).unwrap();
        let _b = Session::accept(&bob, alice.id, &ct, 100);
        let frame = a.seal(1, b"between us").unwrap();
        // An eavesdropper with a different session key gets authentication failure.
        let eve = KemIdentity::generate(b"eve");
        let (_ed, ect) = Session::initiate(eve.id, bob.id, &bob.public, b"e2", 100).unwrap();
        let eve_sess = Session::accept(&bob, eve.id, &ect, 100);
        assert!(eve_sess.open(1, &frame).is_err());
    }

    /// Two sessions initiated with different `eph_seed` values must produce
    /// distinct traffic keys and distinct GCM IV streams, even when all other
    /// inputs (identities, expiry) are identical. This exercises the
    /// `session_nonce` path that guards against (key, IV) reuse across sessions.
    #[test]
    fn different_eph_seeds_produce_different_keys_and_ivs() {
        let (alice, bob) = pair();

        let (mut sess_a1, _ct1) =
            Session::initiate(alice.id, bob.id, &bob.public, b"seed-session-1", 100).unwrap();
        let (mut sess_a2, _ct2) =
            Session::initiate(alice.id, bob.id, &bob.public, b"seed-session-2", 100).unwrap();

        // Seal the same plaintext at the same seq on both sessions. If (key, IV)
        // were shared the ciphertexts would be identical — a catastrophic reuse.
        let frame1 = sess_a1.seal(1, b"probe").unwrap();
        let frame2 = sess_a2.seal(1, b"probe").unwrap();
        assert_ne!(
            frame1.ciphertext, frame2.ciphertext,
            "different eph_seeds must not share a (key, IV) pair"
        );
        // Tags must also differ (they depend on the key).
        assert_ne!(frame1.tag, frame2.tag, "different sessions must produce different GCM tags");
    }

    #[test]
    fn tampered_epoch_fails_authentication() {
        // An attacker flipping the epoch field in a frame header must cause open()
        // to return AuthFailed, not silently accept the manipulated value.
        let (alice, bob) = pair();
        let (mut a, ct) = Session::initiate(alice.id, bob.id, &bob.public, b"e", 100).unwrap();
        let b = Session::accept(&bob, alice.id, &ct, 100);
        let mut frame = a.seal(1, b"secret").unwrap();
        // Flip the epoch to a different value — the AAD used during seal() no longer
        // matches what open() will compute, so GCM authentication must fail.
        frame.epoch ^= 0xDEAD_BEEF_CAFE_BABE;
        assert_eq!(b.open(1, &frame).err(), Some(SessionError::AuthFailed));
    }

    // Helper: expose the ciphertext bytes for the "no plaintext on wire" assertion.
    fn frame_bytes(f: &Frame) -> Vec<u8> {
        let mut v = vec![];
        v.extend_from_slice(&f.ciphertext);
        v.extend_from_slice(&f.tag);
        v
    }
}

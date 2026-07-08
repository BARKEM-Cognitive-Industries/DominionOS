//! A TLS 1.3 client (RFC 8446).
//!
//! This is the secure transport for the browser: it drives the full 1-RTT
//! handshake — ClientHello, key exchange (X25519), key schedule (HKDF), record
//! protection (AES-128-GCM or ChaCha20-Poly1305), certificate-chain validation
//! ([`crate::x509`]) and Finished verification — then exposes an encrypted
//! application-data channel.
//!
//! The protocol logic is pure and byte-driven: it talks to the network only
//! through the [`Io`] trait, so the kernel supplies a raw TCP pipe and the host
//! test-suite supplies an in-memory peer. No global state, no `unsafe`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::hash::sha256;
use crate::tlscrypto as cr;
use crate::x509::{self, SigAlg, TrustStore};

/// Errors that can abort a handshake or a record operation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TlsError {
    Io,
    Closed,
    Protocol,
    UnsupportedCipher,
    UnsupportedGroup,
    BadRecord,
    DecryptFailed,
    BadCertificate,
    BadSignature,
    BadFinished,
    UntrustedChain,
    NameMismatch,
    Expired,
    /// The peer's key share is a low-order point: the X25519 shared secret came
    /// out all-zero (RFC 7748 §6.1), so the handshake is aborted rather than
    /// continuing with an attacker-forced constant secret.
    WeakPublicKey,
}

/// The raw byte transport the handshake runs over (typically a TCP socket).
pub trait Io {
    /// Write the entire buffer.
    fn write_all(&mut self, data: &[u8]) -> Result<(), TlsError>;
    /// Append at least one byte of newly received data to `out`, or return
    /// `Err(TlsError::Closed)` if the peer closed with nothing left.
    fn read_some(&mut self, out: &mut Vec<u8>) -> Result<(), TlsError>;
}

// Record content types.
const REC_CHANGE_CIPHER_SPEC: u8 = 20;
const REC_ALERT: u8 = 21;
const REC_HANDSHAKE: u8 = 22;
const REC_APPLICATION_DATA: u8 = 23;

// Handshake message types.
const HS_CLIENT_HELLO: u8 = 1;
const HS_SERVER_HELLO: u8 = 2;
const HS_ENCRYPTED_EXTENSIONS: u8 = 8;
const HS_CERTIFICATE: u8 = 11;
const HS_CERTIFICATE_VERIFY: u8 = 15;
const HS_FINISHED: u8 = 20;

// Cipher suites we support (all SHA-256, identical key schedule).
const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
const TLS_CHACHA20_POLY1305_SHA256: u16 = 0x1303;

// Named group.
const GROUP_X25519: u16 = 0x001d;

// Signature schemes (for signature_algorithms and CertificateVerify).
const SIG_ECDSA_P256_SHA256: u16 = 0x0403;
const SIG_ECDSA_P384_SHA384: u16 = 0x0503;
const SIG_RSA_PSS_RSAE_SHA256: u16 = 0x0804;
const SIG_RSA_PKCS1_SHA256: u16 = 0x0401;

/// The AEAD selected for the connection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Aead {
    Aes128Gcm,
    ChaCha20Poly1305,
}

impl Aead {
    fn key_len(self) -> usize {
        match self {
            Aead::Aes128Gcm => 16,
            Aead::ChaCha20Poly1305 => 32,
        }
    }
    fn seal(self, key: &[u8], nonce: &[u8; 12], aad: &[u8], pt: &[u8]) -> Vec<u8> {
        match self {
            Aead::Aes128Gcm => {
                let mut k = [0u8; 16];
                k.copy_from_slice(key);
                cr::aes128_gcm_seal(&k, nonce, aad, pt)
            }
            Aead::ChaCha20Poly1305 => {
                let mut k = [0u8; 32];
                k.copy_from_slice(key);
                cr::chacha20poly1305_seal(&k, nonce, aad, pt)
            }
        }
    }
    fn open(self, key: &[u8], nonce: &[u8; 12], aad: &[u8], ct: &[u8]) -> Option<Vec<u8>> {
        match self {
            Aead::Aes128Gcm => {
                let mut k = [0u8; 16];
                k.copy_from_slice(key);
                cr::aes128_gcm_open(&k, nonce, aad, ct)
            }
            Aead::ChaCha20Poly1305 => {
                let mut k = [0u8; 32];
                k.copy_from_slice(key);
                cr::chacha20poly1305_open(&k, nonce, aad, ct)
            }
        }
    }
}

// ============================================================================
// Key schedule.
// ============================================================================

/// One direction's record-protection state: traffic secret, derived key/iv,
/// and a record sequence number.
#[derive(Clone)]
struct Keys {
    aead: Aead,
    key: Vec<u8>,
    iv: [u8; 12],
    secret: [u8; 32],
    seq: u64,
}

impl Keys {
    fn derive(aead: Aead, secret: [u8; 32]) -> Keys {
        let key = cr::hkdf_expand_label(&secret, "key", &[], aead.key_len());
        let iv_v = cr::hkdf_expand_label(&secret, "iv", &[], 12);
        let mut iv = [0u8; 12];
        iv.copy_from_slice(&iv_v);
        Keys { aead, key, iv, secret, seq: 0 }
    }

    fn nonce(&self) -> [u8; 12] {
        let mut n = self.iv;
        let s = self.seq.to_be_bytes();
        for i in 0..8 {
            n[4 + i] ^= s[i];
        }
        n
    }
}

/// The finished-key for a traffic secret.
fn finished_key(secret: &[u8; 32]) -> Vec<u8> {
    cr::hkdf_expand_label(secret, "finished", &[], 32)
}

// ============================================================================
// Record layer.
// ============================================================================

fn record_header(ct: u8, len: usize) -> [u8; 5] {
    [ct, 0x03, 0x03, (len >> 8) as u8, len as u8]
}

/// Encrypt one handshake/application record. `inner_type` is the true content
/// type appended before encryption (TLS 1.3 inner plaintext).
fn seal_record(keys: &mut Keys, inner_type: u8, content: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(content.len() + 1);
    inner.extend_from_slice(content);
    inner.push(inner_type);
    let total = inner.len() + 16; // + AEAD tag
    let header = record_header(REC_APPLICATION_DATA, total);
    let nonce = keys.nonce();
    let ct = keys.aead.seal(&keys.key, &nonce, &header, &inner);
    keys.seq += 1;
    let mut out = Vec::with_capacity(5 + ct.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(&ct);
    out
}

/// Decrypt one protected record body, returning (inner_type, plaintext).
fn open_record(keys: &mut Keys, header: &[u8; 5], ct: &[u8]) -> Result<(u8, Vec<u8>), TlsError> {
    let nonce = keys.nonce();
    let mut pt = keys
        .aead
        .open(&keys.key, &nonce, header, ct)
        .ok_or(TlsError::DecryptFailed)?;
    keys.seq += 1;
    // Strip trailing zero padding; the last non-zero byte is the content type.
    while let Some(&0) = pt.last() {
        pt.pop();
    }
    let inner_type = pt.pop().ok_or(TlsError::BadRecord)?;
    Ok((inner_type, pt))
}

// ============================================================================
// A buffered record reader over an Io.
// ============================================================================

struct RecordReader {
    buf: Vec<u8>,
}

impl RecordReader {
    fn new() -> RecordReader {
        RecordReader { buf: Vec::new() }
    }

    /// Read exactly one TLS record. Returns (content_type, header, body).
    fn next<I: Io>(&mut self, io: &mut I) -> Result<(u8, [u8; 5], Vec<u8>), TlsError> {
        loop {
            if self.buf.len() >= 5 {
                let len = ((self.buf[3] as usize) << 8) | self.buf[4] as usize;
                if len > (1 << 14) + 256 {
                    return Err(TlsError::BadRecord);
                }
                if self.buf.len() >= 5 + len {
                    let mut header = [0u8; 5];
                    header.copy_from_slice(&self.buf[..5]);
                    let body = self.buf[5..5 + len].to_vec();
                    self.buf.drain(..5 + len);
                    return Ok((header[0], header, body));
                }
            }
            io.read_some(&mut self.buf)?;
        }
    }
}

// ============================================================================
// Wire-format helpers.
// ============================================================================

fn push_u16(v: &mut Vec<u8>, x: u16) {
    v.push((x >> 8) as u8);
    v.push(x as u8);
}

fn push_u24(v: &mut Vec<u8>, x: usize) {
    v.push((x >> 16) as u8);
    v.push((x >> 8) as u8);
    v.push(x as u8);
}

/// Wrap `body` as a handshake message of `msg_type` (1-byte type + u24 length).
fn handshake_msg(msg_type: u8, body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + body.len());
    v.push(msg_type);
    push_u24(&mut v, body.len());
    v.extend_from_slice(body);
    v
}

// ============================================================================
// The handshake.
// ============================================================================

/// Per-connection configuration.
pub struct TlsConfig<'a> {
    pub hostname: &'a str,
    pub trust: &'a TrustStore,
    /// Unix time (seconds) for validity checks, or 0 to skip time checks.
    pub now: u64,
    /// If true, a chain that does not reach a trust anchor still completes the
    /// handshake (the caller decides how to surface that). Off by default.
    pub allow_unverified: bool,
}

/// An established TLS 1.3 connection carrying application-data keys.
pub struct TlsConnection {
    client: Keys,
    server: Keys,
    reader: RecordReader,
    /// True once the peer's close_notify has been seen.
    closed: bool,
}

/// Derive both client and server private values from a 32-byte entropy seed.
fn expand_entropy(entropy: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a.copy_from_slice(&sha256(&[entropy.as_slice(), b"x25519"].concat()));
    b.copy_from_slice(&sha256(&[entropy.as_slice(), b"client-random"].concat()));
    (a, b)
}

/// Run the TLS 1.3 client handshake over `io`.
pub fn connect<I: Io>(
    io: &mut I,
    config: &TlsConfig,
    entropy: &[u8; 32],
) -> Result<TlsConnection, TlsError> {
    let (x_priv, client_random) = expand_entropy(entropy);
    // Clamp is applied inside x25519; the public key is scalar·basepoint.
    let client_pub = cr::x25519_base(&x_priv);

    // --- Build and send ClientHello ---
    let ch_body = build_client_hello(config.hostname, &client_random, &client_pub);
    let ch = handshake_msg(HS_CLIENT_HELLO, &ch_body);
    let mut transcript: Vec<u8> = ch.clone();
    io.write_all(&wrap_plaintext(REC_HANDSHAKE, &ch))?;

    // --- Read ServerHello (skipping any ChangeCipherSpec) ---
    let mut reader = RecordReader::new();
    let (sh_type, _sh_hdr, sh_body) = read_handshake_plaintext(io, &mut reader)?;
    if sh_type != HS_SERVER_HELLO {
        return Err(TlsError::Protocol);
    }
    let sh = handshake_msg(HS_SERVER_HELLO, &sh_body);
    transcript.extend_from_slice(&sh);
    let (suite, server_pub) = parse_server_hello(&sh_body)?;
    let aead = match suite {
        TLS_AES_128_GCM_SHA256 => Aead::Aes128Gcm,
        TLS_CHACHA20_POLY1305_SHA256 => Aead::ChaCha20Poly1305,
        _ => return Err(TlsError::UnsupportedCipher),
    };

    // --- Key schedule: handshake secrets ---
    // RFC 7748 §6.1: reject a low-order peer key that forces an all-zero secret.
    let shared = cr::x25519_checked(&x_priv, &server_pub).ok_or(TlsError::WeakPublicKey)?;
    let early = cr::hkdf_extract(&[0u8; 32], &[0u8; 32]);
    let empty_hash = sha256(b"");
    let derived = cr::derive_secret(&early, "derived", &empty_hash);
    let handshake_secret = cr::hkdf_extract(&derived, &shared);
    let th_ch_sh = sha256(&transcript);
    let c_hs = cr::derive_secret(&handshake_secret, "c hs traffic", &th_ch_sh);
    let s_hs = cr::derive_secret(&handshake_secret, "s hs traffic", &th_ch_sh);
    let mut client_hs = Keys::derive(aead, c_hs);
    let mut server_hs = Keys::derive(aead, s_hs);

    // --- Read encrypted handshake flight: EE, Certificate, CertificateVerify, Finished ---
    let mut hs_stream: Vec<u8> = Vec::new();
    let mut ee_seen = false;
    let mut cert_chain: Vec<x509::Certificate> = Vec::new();
    let mut cert_verify: Option<(SigAlg, Vec<u8>)> = None;
    let mut th_before_certverify: Option<[u8; 32]> = None;
    let server_finished: Vec<u8>;
    let mut th_through_certverify: Option<[u8; 32]> = None;

    'flight: loop {
        // Pull more decrypted handshake bytes if we can't parse a full message.
        while !has_full_handshake(&hs_stream) {
            let (ct, header, body) = reader.next(io)?;
            match ct {
                REC_CHANGE_CIPHER_SPEC => continue,
                REC_ALERT => return Err(TlsError::Protocol),
                REC_APPLICATION_DATA => {
                    let (inner, pt) = open_record(&mut server_hs, &header, &body)?;
                    if inner == REC_ALERT {
                        return Err(TlsError::Protocol);
                    }
                    if inner != REC_HANDSHAKE {
                        return Err(TlsError::Protocol);
                    }
                    hs_stream.extend_from_slice(&pt);
                }
                _ => return Err(TlsError::Protocol),
            }
        }

        // Consume each complete handshake message in the stream.
        while let Some((mtype, msg)) = take_handshake(&mut hs_stream) {
            match mtype {
                HS_ENCRYPTED_EXTENSIONS => {
                    ee_seen = true;
                    transcript.extend_from_slice(&msg);
                }
                HS_CERTIFICATE => {
                    if !ee_seen {
                        return Err(TlsError::Protocol);
                    }
                    cert_chain = parse_certificate_msg(&msg[4..])?;
                    transcript.extend_from_slice(&msg);
                    th_before_certverify = Some(sha256(&transcript));
                }
                HS_CERTIFICATE_VERIFY => {
                    cert_verify = Some(parse_certificate_verify(&msg[4..])?);
                    transcript.extend_from_slice(&msg);
                    th_through_certverify = Some(sha256(&transcript));
                }
                HS_FINISHED => {
                    server_finished = msg[4..].to_vec();
                    // Do not add to transcript yet for the Finished check itself.
                    break 'flight;
                }
                _ => return Err(TlsError::Protocol),
            }
        }
    }

    // --- Verify the server's certificate chain ---
    if cert_chain.is_empty() {
        return Err(TlsError::BadCertificate);
    }
    let leaf = cert_chain[0].clone();
    let chain_ok = config
        .trust
        .verify_chain(&cert_chain, config.hostname, config.now)
        .map(|_| true)
        .or_else(|e| match e {
            x509::X509Error::NameMismatch => Err(TlsError::NameMismatch),
            x509::X509Error::Expired => Err(TlsError::Expired),
            x509::X509Error::UntrustedRoot | x509::X509Error::EmptyChain => {
                if config.allow_unverified {
                    Ok(false)
                } else {
                    Err(TlsError::UntrustedChain)
                }
            }
            _ => Err(TlsError::BadCertificate),
        })?;
    let _ = chain_ok;

    // --- Verify CertificateVerify over the transcript ---
    let (cv_alg, cv_sig) = cert_verify.ok_or(TlsError::Protocol)?;
    let th_cert = th_before_certverify.ok_or(TlsError::Protocol)?;
    let cv_message = certificate_verify_message(&th_cert);
    if !leaf.key.verify(cv_alg, &cv_message, &cv_sig) {
        return Err(TlsError::BadSignature);
    }

    // --- Verify the server's Finished ---
    let th_cv = th_through_certverify.ok_or(TlsError::Protocol)?;
    let sfin = server_finished;
    let s_fin_key = finished_key(&server_hs.secret);
    let expected = cr::hmac_sha256(&s_fin_key, &th_cv);
    if !cr::ct_eq(&expected, &sfin) {
        return Err(TlsError::BadFinished);
    }
    // Add server Finished to the transcript now (CH..server Finished).
    transcript.extend_from_slice(&handshake_msg(HS_FINISHED, &sfin));
    let th_sfin = sha256(&transcript);

    // --- Application-data key schedule ---
    let derived2 = cr::derive_secret(&handshake_secret, "derived", &empty_hash);
    let master = cr::hkdf_extract(&derived2, &[0u8; 32]);
    let c_ap = cr::derive_secret(&master, "c ap traffic", &th_sfin);
    let s_ap = cr::derive_secret(&master, "s ap traffic", &th_sfin);
    let client_app = Keys::derive(aead, c_ap);
    let server_app = Keys::derive(aead, s_ap);

    // --- Send change_cipher_spec (compat) + client Finished (handshake-encrypted) ---
    io.write_all(&wrap_plaintext(REC_CHANGE_CIPHER_SPEC, &[0x01]))?;
    let c_fin_key = finished_key(&client_hs.secret);
    let c_verify = cr::hmac_sha256(&c_fin_key, &th_sfin);
    let fin_msg = handshake_msg(HS_FINISHED, &c_verify);
    let fin_record = seal_record(&mut client_hs, REC_HANDSHAKE, &fin_msg);
    io.write_all(&fin_record)?;

    Ok(TlsConnection {
        client: client_app,
        server: server_app,
        reader,
        closed: false,
    })
}

impl TlsConnection {
    /// Encrypt and send application data.
    pub fn send<I: Io>(&mut self, io: &mut I, data: &[u8]) -> Result<(), TlsError> {
        // Fragment into <=16KB records.
        for chunk in data.chunks(16 * 1024) {
            let rec = seal_record(&mut self.client, REC_APPLICATION_DATA, chunk);
            io.write_all(&rec)?;
        }
        Ok(())
    }

    /// Receive the next chunk of application data. Returns an empty vector when
    /// the peer has closed (close_notify).
    pub fn recv<I: Io>(&mut self, io: &mut I) -> Result<Vec<u8>, TlsError> {
        if self.closed {
            return Ok(Vec::new());
        }
        loop {
            let (ct, header, body) = match self.reader.next(io) {
                Ok(v) => v,
                Err(TlsError::Closed) => {
                    self.closed = true;
                    return Ok(Vec::new());
                }
                Err(e) => return Err(e),
            };
            match ct {
                REC_CHANGE_CIPHER_SPEC => continue,
                REC_APPLICATION_DATA => {
                    let (inner, pt) = open_record(&mut self.server, &header, &body)?;
                    match inner {
                        REC_APPLICATION_DATA => return Ok(pt),
                        REC_HANDSHAKE => continue, // session tickets / key update: ignore
                        REC_ALERT => {
                            self.closed = true;
                            return Ok(Vec::new());
                        }
                        _ => continue,
                    }
                }
                REC_ALERT => {
                    self.closed = true;
                    return Ok(Vec::new());
                }
                _ => return Err(TlsError::Protocol),
            }
        }
    }

    /// Read the entire response until the peer closes the connection.
    pub fn recv_to_end<I: Io>(&mut self, io: &mut I) -> Result<Vec<u8>, TlsError> {
        let mut out = Vec::new();
        loop {
            let chunk = self.recv(io)?;
            if chunk.is_empty() {
                break;
            }
            out.extend_from_slice(&chunk);
        }
        Ok(out)
    }
}

// ============================================================================
// ClientHello construction.
// ============================================================================

fn wrap_plaintext(ct: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(5 + payload.len());
    v.extend_from_slice(&record_header(ct, payload.len()));
    v.extend_from_slice(payload);
    v
}

fn build_client_hello(hostname: &str, random: &[u8; 32], key_share: &[u8; 32]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u16(&mut b, 0x0303); // legacy_version = TLS 1.2
    b.extend_from_slice(random);
    // legacy_session_id: 32 bytes (echoes for middlebox compatibility).
    b.push(32);
    b.extend_from_slice(random); // any 32 bytes; reuse random
    // cipher_suites
    let suites = [TLS_AES_128_GCM_SHA256, TLS_CHACHA20_POLY1305_SHA256];
    push_u16(&mut b, (suites.len() * 2) as u16);
    for s in suites {
        push_u16(&mut b, s);
    }
    // legacy_compression_methods: null only
    b.push(1);
    b.push(0);
    // extensions
    let mut ext = Vec::new();
    // server_name (0)
    {
        let mut sni = Vec::new();
        let name = hostname.as_bytes();
        push_u16(&mut sni, (name.len() + 3) as u16); // server_name_list length
        sni.push(0); // host_name
        push_u16(&mut sni, name.len() as u16);
        sni.extend_from_slice(name);
        push_ext(&mut ext, 0x0000, &sni);
    }
    // supported_versions (43): TLS 1.3
    {
        let mut sv = Vec::new();
        sv.push(2);
        push_u16(&mut sv, 0x0304);
        push_ext(&mut ext, 0x002b, &sv);
    }
    // supported_groups (10): x25519
    {
        let mut g = Vec::new();
        push_u16(&mut g, 2);
        push_u16(&mut g, GROUP_X25519);
        push_ext(&mut ext, 0x000a, &g);
    }
    // signature_algorithms (13)
    {
        let mut s = Vec::new();
        let algs = [
            SIG_ECDSA_P256_SHA256,
            SIG_ECDSA_P384_SHA384,
            SIG_RSA_PSS_RSAE_SHA256,
            SIG_RSA_PKCS1_SHA256,
        ];
        push_u16(&mut s, (algs.len() * 2) as u16);
        for a in algs {
            push_u16(&mut s, a);
        }
        push_ext(&mut ext, 0x000d, &s);
    }
    // key_share (51): x25519
    {
        let mut ks = Vec::new();
        let mut entry = Vec::new();
        push_u16(&mut entry, GROUP_X25519);
        push_u16(&mut entry, 32);
        entry.extend_from_slice(key_share);
        push_u16(&mut ks, entry.len() as u16);
        ks.extend_from_slice(&entry);
        push_ext(&mut ext, 0x0033, &ks);
    }
    push_u16(&mut b, ext.len() as u16);
    b.extend_from_slice(&ext);
    b
}

fn push_ext(ext: &mut Vec<u8>, kind: u16, body: &[u8]) {
    push_u16(ext, kind);
    push_u16(ext, body.len() as u16);
    ext.extend_from_slice(body);
}

// ============================================================================
// Parsing.
// ============================================================================

/// Read one plaintext handshake message (used for ServerHello). Skips
/// ChangeCipherSpec. Returns (msg_type, body-without-header).
fn read_handshake_plaintext<I: Io>(
    io: &mut I,
    reader: &mut RecordReader,
) -> Result<(u8, [u8; 5], Vec<u8>), TlsError> {
    loop {
        let (ct, header, body) = reader.next(io)?;
        match ct {
            REC_CHANGE_CIPHER_SPEC => continue,
            REC_ALERT => return Err(TlsError::Protocol),
            REC_HANDSHAKE => {
                if body.len() < 4 {
                    return Err(TlsError::Protocol);
                }
                let mtype = body[0];
                let len = ((body[1] as usize) << 16) | ((body[2] as usize) << 8) | body[3] as usize;
                if body.len() < 4 + len {
                    return Err(TlsError::Protocol);
                }
                return Ok((mtype, header, body[4..4 + len].to_vec()));
            }
            _ => return Err(TlsError::Protocol),
        }
    }
}

/// Is there at least one complete handshake message buffered?
fn has_full_handshake(s: &[u8]) -> bool {
    if s.len() < 4 {
        return false;
    }
    let len = ((s[1] as usize) << 16) | ((s[2] as usize) << 8) | s[3] as usize;
    s.len() >= 4 + len
}

/// Pop one complete handshake message (type + 4-byte header + body) off the
/// front of `s`. Returns (msg_type, full_message_including_header).
fn take_handshake(s: &mut Vec<u8>) -> Option<(u8, Vec<u8>)> {
    if !has_full_handshake(s) {
        return None;
    }
    let mtype = s[0];
    let len = ((s[1] as usize) << 16) | ((s[2] as usize) << 8) | s[3] as usize;
    let msg = s[..4 + len].to_vec();
    s.drain(..4 + len);
    Some((mtype, msg))
}

fn parse_server_hello(body: &[u8]) -> Result<(u16, [u8; 32]), TlsError> {
    let mut p = 0usize;
    // legacy_version (2) + random (32)
    if body.len() < 35 {
        return Err(TlsError::Protocol);
    }
    p += 2 + 32;
    // legacy_session_id_echo
    let sid_len = body[p] as usize;
    p += 1 + sid_len;
    if body.len() < p + 3 {
        return Err(TlsError::Protocol);
    }
    // cipher_suite (2)
    let suite = ((body[p] as u16) << 8) | body[p + 1] as u16;
    p += 2;
    // legacy_compression_method (1)
    p += 1;
    // extensions
    if body.len() < p + 2 {
        return Err(TlsError::Protocol);
    }
    let ext_len = ((body[p] as usize) << 8) | body[p + 1] as usize;
    p += 2;
    let ext_end = p + ext_len;
    if ext_end > body.len() {
        return Err(TlsError::Protocol);
    }
    let mut server_pub: Option<[u8; 32]> = None;
    while p + 4 <= ext_end {
        let kind = ((body[p] as u16) << 8) | body[p + 1] as u16;
        let elen = ((body[p + 2] as usize) << 8) | body[p + 3] as usize;
        p += 4;
        if p + elen > ext_end {
            return Err(TlsError::Protocol);
        }
        let edata = &body[p..p + elen];
        if kind == 0x0033 {
            // key_share: group (2) + key_exchange length (2) + key
            if edata.len() < 4 {
                return Err(TlsError::Protocol);
            }
            let group = ((edata[0] as u16) << 8) | edata[1] as u16;
            if group != GROUP_X25519 {
                return Err(TlsError::UnsupportedGroup);
            }
            let klen = ((edata[2] as usize) << 8) | edata[3] as usize;
            if klen != 32 || edata.len() < 4 + 32 {
                return Err(TlsError::Protocol);
            }
            let mut k = [0u8; 32];
            k.copy_from_slice(&edata[4..36]);
            server_pub = Some(k);
        }
        p += elen;
    }
    let sp = server_pub.ok_or(TlsError::UnsupportedGroup)?;
    Ok((suite, sp))
}

/// Parse the Certificate message body (after the 4-byte handshake header).
fn parse_certificate_msg(body: &[u8]) -> Result<Vec<x509::Certificate>, TlsError> {
    let mut p = 0usize;
    // certificate_request_context (u8 length)
    if body.is_empty() {
        return Err(TlsError::BadCertificate);
    }
    let ctx_len = body[p] as usize;
    p += 1 + ctx_len;
    if body.len() < p + 3 {
        return Err(TlsError::BadCertificate);
    }
    // certificate_list length (u24)
    let list_len = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | body[p + 2] as usize;
    p += 3;
    let end = p + list_len;
    if end > body.len() {
        return Err(TlsError::BadCertificate);
    }
    let mut chain = Vec::new();
    while p + 3 <= end {
        let clen = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | body[p + 2] as usize;
        p += 3;
        if p + clen > end {
            return Err(TlsError::BadCertificate);
        }
        let cert = x509::parse_certificate(&body[p..p + clen]).map_err(|_| TlsError::BadCertificate)?;
        chain.push(cert);
        p += clen;
        // extensions (u16) per CertificateEntry
        if p + 2 > end {
            break;
        }
        let elen = ((body[p] as usize) << 8) | body[p + 1] as usize;
        p += 2 + elen;
    }
    Ok(chain)
}

/// Parse CertificateVerify body: scheme (2) + signature (u16 length).
fn parse_certificate_verify(body: &[u8]) -> Result<(SigAlg, Vec<u8>), TlsError> {
    if body.len() < 4 {
        return Err(TlsError::Protocol);
    }
    let scheme = ((body[0] as u16) << 8) | body[1] as u16;
    let slen = ((body[2] as usize) << 8) | body[3] as usize;
    if body.len() < 4 + slen {
        return Err(TlsError::Protocol);
    }
    let sig = body[4..4 + slen].to_vec();
    let alg = match scheme {
        SIG_ECDSA_P256_SHA256 => SigAlg::EcdsaP256Sha256,
        SIG_ECDSA_P384_SHA384 => SigAlg::EcdsaP384Sha384,
        SIG_RSA_PSS_RSAE_SHA256 => SigAlg::RsaPssSha256,
        // RFC 8446 §4.4.3: rsa_pkcs1_* code points are permitted only for
        // legacy certificate signatures, never for the CertificateVerify
        // handshake signature (which must use RSASSA-PSS). Reject them here.
        SIG_RSA_PKCS1_SHA256 => return Err(TlsError::BadSignature),
        _ => return Err(TlsError::BadSignature),
    };
    Ok((alg, sig))
}

/// The message whose signature CertificateVerify carries (RFC 8446 §4.4.3).
fn certificate_verify_message(transcript_hash: &[u8; 32]) -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&[0x20; 64]);
    m.extend_from_slice(b"TLS 1.3, server CertificateVerify");
    m.push(0x00);
    m.extend_from_slice(transcript_hash);
    m
}

/// A convenience: the SNI/host string is owned here so callers can keep it.
pub fn host_string(s: &str) -> String {
    String::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        s.chunks(2)
            .map(|c| u8::from_str_radix(core::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect()
    }

    // ---- RFC 8448 "Simple 1-RTT Handshake" key-schedule vectors ----
    #[test]
    fn rfc8448_key_schedule() {
        // Early Secret = HKDF-Extract(0, 0).
        let early = cr::hkdf_extract(&[0u8; 32], &[0u8; 32]);
        assert_eq!(
            early.to_vec(),
            hex("33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a")
        );
        // derived = Derive-Secret(early, "derived", "")
        let empty = sha256(b"");
        let derived = cr::derive_secret(&early, "derived", &empty);
        assert_eq!(
            derived.to_vec(),
            hex("6f2615a108c702c5678f54fc9dbab69716c076189c48250cebeac3576c3611ba")
        );
        // Handshake Secret = HKDF-Extract(derived, ECDHE).
        let ecdhe = hex("8bd4054fb55b9d63fdfbacf9f04b9f0d35e6d63f537563efd46272900f89492d");
        let hs = cr::hkdf_extract(&derived, &ecdhe);
        assert_eq!(
            hs.to_vec(),
            hex("1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac")
        );
    }

    // ---- RFC 8448 record-key derivation (validates hkdf_expand_label key/iv) ----
    #[test]
    fn rfc8448_server_handshake_key_iv() {
        // The server handshake-traffic secret from RFC 8448.
        let s_hs = hex("b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38");
        let mut secret = [0u8; 32];
        secret.copy_from_slice(&s_hs);
        let keys = Keys::derive(Aead::Aes128Gcm, secret);
        // Expected key/iv from RFC 8448 §3 (server handshake traffic).
        assert_eq!(keys.key, hex("3fce516009c21727d0f2e4e86ee403bc"));
        assert_eq!(keys.iv.to_vec(), hex("5d313eb2671276ee13000b30"));
    }

    #[test]
    fn record_seal_open_roundtrip() {
        let secret = [0x42u8; 32];
        for aead in [Aead::Aes128Gcm, Aead::ChaCha20Poly1305] {
            let mut s = Keys::derive(aead, secret);
            let mut r = Keys::derive(aead, secret);
            let rec = seal_record(&mut s, REC_APPLICATION_DATA, b"GET / HTTP/1.1\r\n\r\n");
            let header: [u8; 5] = [rec[0], rec[1], rec[2], rec[3], rec[4]];
            let (inner, pt) = open_record(&mut r, &header, &rec[5..]).unwrap();
            assert_eq!(inner, REC_APPLICATION_DATA);
            assert_eq!(pt, b"GET / HTTP/1.1\r\n\r\n");
            // Sequence advanced: a second record uses a fresh nonce.
            let rec2 = seal_record(&mut s, REC_APPLICATION_DATA, b"second");
            let header2: [u8; 5] = [rec2[0], rec2[1], rec2[2], rec2[3], rec2[4]];
            let (_, pt2) = open_record(&mut r, &header2, &rec2[5..]).unwrap();
            assert_eq!(pt2, b"second");
        }
    }

    #[test]
    fn client_hello_is_wellformed() {
        let ch = build_client_hello("example.com", &[7u8; 32], &[9u8; 32]);
        // version + random(32) + sid(1+32) + suites(2+4) + comp(2) ...
        assert_eq!(&ch[..2], &[0x03, 0x03]);
        assert_eq!(ch[34], 32); // session id length
        // The SNI hostname appears verbatim somewhere in the extensions.
        let needle = b"example.com";
        assert!(ch.windows(needle.len()).any(|w| w == needle));
    }

    // ---- RFC 7748 §6.1: reject a low-order peer key share ----
    #[test]
    fn x25519_checked_rejects_low_order_point() {
        // The all-zero point is the canonical low-order point: any scalar
        // multiplied by it yields an all-zero shared secret.
        let scalar = [0x11u8; 32];
        assert!(cr::x25519_checked(&scalar, &[0u8; 32]).is_none());
        // A genuine peer public key still agrees to a non-zero secret.
        let peer_pub = cr::x25519_base(&[0x5au8; 32]);
        let ok = cr::x25519_checked(&scalar, &peer_pub).expect("normal agreement is Some");
        assert_ne!(ok, [0u8; 32]);
    }

    #[test]
    fn handshake_aborts_on_low_order_server_key_share() {
        let store = TrustStore::new();
        let config = TlsConfig {
            hostname: "aether.test",
            trust: &store,
            now: 0,
            allow_unverified: true,
        };

        // A ServerHello whose X25519 key_share is the all-zero low-order point.
        let sh_body = build_server_hello(&[0xa5u8; 32], &[], TLS_AES_128_GCM_SHA256, &[0u8; 32]);
        let sh = handshake_msg(HS_SERVER_HELLO, &sh_body);
        let sh_record = wrap_plaintext(REC_HANDSHAKE, &sh);

        let mut pipe = Pipe {
            to_peer: Vec::new(),
            from_peer: sh_record,
        };
        let mut client = ClientSide { p: &mut pipe };

        // The client must abort at the ECDH step rather than deriving keys from
        // an attacker-forced all-zero shared secret.
        match connect(&mut client, &config, &[0x33u8; 32]) {
            Err(TlsError::WeakPublicKey) => {}
            Err(other) => panic!("expected WeakPublicKey, got {:?}", other),
            Ok(_) => panic!("handshake must abort on a low-order key share"),
        }
    }

    // ---- A minimal in-test TLS 1.3 server to drive the full client path ----

    /// An in-memory bidirectional pipe.
    struct Pipe {
        to_peer: Vec<u8>,
        from_peer: Vec<u8>,
    }

    struct ClientSide<'a> {
        p: &'a mut Pipe,
    }
    impl Io for ClientSide<'_> {
        fn write_all(&mut self, data: &[u8]) -> Result<(), TlsError> {
            self.p.to_peer.extend_from_slice(data);
            Ok(())
        }
        fn read_some(&mut self, out: &mut Vec<u8>) -> Result<(), TlsError> {
            if self.p.from_peer.is_empty() {
                return Err(TlsError::Closed);
            }
            out.extend_from_slice(&self.p.from_peer);
            self.p.from_peer.clear();
            Ok(())
        }
    }

    // openssl-generated self-signed RSA cert (CN=aether.test, SAN aether.test/*.aether.test).
    // Pre-rename fixture: the DER below encodes aether.test and cannot be regenerated for the
    // new name without re-signing, so the handshake test connects to "aether.test" to match it.
    const CERT_DER: &str = "308203343082021ca0030201020214504ad3ea3f919686aee08ca8172e53489f2776a3300d06092a864886f70d01010b050030163114301206035504030c0b6165746865722e74657374301e170d3236303632303134313633315a170d3336303631373134313633315a30163114301206035504030c0b6165746865722e7465737430820122300d06092a864886f70d01010105000382010f003082010a0282010100e4dd7baf7f987a6b91db9d8e041295f5b97f3ecfdad0aef3c6e7587c769399252b542ab59ad9358f2f5f3e885c8bf36119d7e0ac735d80b5243817332f34e37ba6e9b5400a977e75093d3b49c494f16b8a63c73551834ac0fe58709a80a6f9a0f8355d8c3a1d452a7c3f7c282c3ee14c8f6ab4fb4d8763086100c62184a90463df9de7a4fc856672e211e694a4be812f2aa3ad9099d54c966ca6e100d9f1c8dba00fd559d5a01f77a6367927825a57e82e17935636645ebd1ecb848d0bbb19a40a0336eb4100f1fe53ea3c6b978ab05f57c13f0e55b5a1307b03d8c80edce420cc0e3463b089a07abd093b0ebf315ecf5af9ad358cc8bc7a8c2dd86cb413254b0203010001a37a3078301d0603551d0e0416041403fe8942062179258f728b66d38a2f194ded7f01301f0603551d2304183016801403fe8942062179258f728b66d38a2f194ded7f01300f0603551d130101ff040530030101ff30250603551d11041e301c820b6165746865722e74657374820d2a2e6165746865722e74657374300d06092a864886f70d01010b05000382010100b3d7a6d4035c81655219e2e17cfad42bd0716a4cd814cb8ad9be9c54513a0021bbb6b20c87f8fafb5c42b8d8e58c2b2489062a2a830c6b566b14ac6f4f76e3d6bcec6dcf44b98bd3a4e24fb422d2a17c7e7855393ea83ce685097914d80a2fb158fb21ae5a88970e3efbac9d5c881c14db116a4ef653cd8386d14fd6872603a2240d2d46eabdb2b436fc1e9aebf763363dc8e3f56804e4e935278fd76aa86b9baea679daf6ee111917426131d57ae0681e5836c6a04452d635c2ac6419e0acdf6964ebfffaf9e3410fd8e2f7c23f500e2e582b0d997a527eb2e11f0646c0a89f1860812aa35510cd645e45ba730cfb494acef33419732a9632408519ceb177f0";
    // The cert's RSA key material, kept so the CertificateVerify signature below
    // can be regenerated (see `CV_SIG_PSS`).
    #[allow(dead_code)]
    const RSA_N: &str = "e4dd7baf7f987a6b91db9d8e041295f5b97f3ecfdad0aef3c6e7587c769399252b542ab59ad9358f2f5f3e885c8bf36119d7e0ac735d80b5243817332f34e37ba6e9b5400a977e75093d3b49c494f16b8a63c73551834ac0fe58709a80a6f9a0f8355d8c3a1d452a7c3f7c282c3ee14c8f6ab4fb4d8763086100c62184a90463df9de7a4fc856672e211e694a4be812f2aa3ad9099d54c966ca6e100d9f1c8dba00fd559d5a01f77a6367927825a57e82e17935636645ebd1ecb848d0bbb19a40a0336eb4100f1fe53ea3c6b978ab05f57c13f0e55b5a1307b03d8c80edce420cc0e3463b089a07abd093b0ebf315ecf5af9ad358cc8bc7a8c2dd86cb413254b";
    #[allow(dead_code)]
    const RSA_D: &str = "14fda164a671c58ca1d5eadc9b2559e9587d975fb8af2f7f54020f9cbec2be7c3e689ba588bc724bf32980fd406bba4346265ac8c6535044e23bb36e5cde8a1cb861880a5b23ab94056869cc7ec20b1074cd68f876c4f1716e57e3355b26600fdcf95c094aa28e4a48175f30fa0f3c5a696efd316d5a1a57003a9cc6cc8edf44334ee09c7bf4c9632c78620d5fc858c15f08a6fed83894b010b5bf608e800a06f1bde985e15d95549c49bdd9d097c777103abcb5d161a34e7ca7d95de3fd8b93c3579a94b781b4482c89f9a4ea5b55652ebce5e91393dfc824cf1ea80896843da2d2cc43d39b860b77c445764429310d240c9a0485e85d4822244a1627ca9295";
    // Precomputed rsa_pss_rsae_sha256 signature over the deterministic
    // CertificateVerify message (see `mock_server_respond`). RFC 8446 requires
    // PSS here; rsa_pkcs1_* is rejected by the client.
    const CV_SIG_PSS: &str = "a02ec57dc60ecf027ec5bcc7fb7b3626c20bc65c6b6647997904d405629666b9cd82a9a460df1719c5db354a0c87686cb911f4e4672415f27a4cb86511d74fa1c89672ab17af03299f0b8da689d40357897406788dc1e56aaab921910c202c34e958d312af53e6b28dfed2026758b93b85f02d45a122dff67b89e7970949d5724a3a70ebb92d9af74e905950dc1c9f7b46c793477f5db73da26e680557caeb89aeda5792f5e3ea84b43f3f204918a4ca13035cfc56ee93e38ea995280d7a8ce475936ebd1bc422e1122f022f28e819d3ba789a2ebbafa30e9b39a2339d1db9c633e620b9e717ec13f0694e9e5f58c871113b94a091af51bd37f150508ed14262";

    /// Build the server's encrypted handshake flight and finished handling.
    /// Returns the bytes the server sends and the server-side app keys.
    #[allow(clippy::too_many_arguments)]
    fn mock_server_respond(client_hello_record: &[u8]) -> (Vec<u8>, Keys, Keys, Vec<u8>) {
        // Parse the client's ClientHello out of the plaintext handshake record.
        assert_eq!(client_hello_record[0], REC_HANDSHAKE);
        let ch_len = ((client_hello_record[3] as usize) << 8) | client_hello_record[4] as usize;
        let ch = client_hello_record[5..5 + ch_len].to_vec();
        // ch is handshake_msg(ClientHello). Extract the client key_share.
        let ch_body = &ch[4..];
        let client_pub = extract_client_keyshare(ch_body);

        // Server ephemeral key.
        let s_priv = [0x5au8; 32];
        let s_pub = cr::x25519_base(&s_priv);
        let s_random = [0xa5u8; 32];

        // Build ServerHello (echo session id from CH).
        let sid_len = ch_body[34] as usize;
        let sid = ch_body[35..35 + sid_len].to_vec();
        let sh_body = build_server_hello(&s_random, &sid, TLS_AES_128_GCM_SHA256, &s_pub);
        let sh = handshake_msg(HS_SERVER_HELLO, &sh_body);

        let mut transcript = ch.clone();
        transcript.extend_from_slice(&sh);

        // Key schedule (server perspective).
        let shared = cr::x25519_checked(&s_priv, &client_pub).expect("client key share is low-order");
        let early = cr::hkdf_extract(&[0u8; 32], &[0u8; 32]);
        let empty = sha256(b"");
        let derived = cr::derive_secret(&early, "derived", &empty);
        let handshake_secret = cr::hkdf_extract(&derived, &shared);
        let th = sha256(&transcript);
        let c_hs = cr::derive_secret(&handshake_secret, "c hs traffic", &th);
        let s_hs = cr::derive_secret(&handshake_secret, "s hs traffic", &th);
        let client_hs = Keys::derive(Aead::Aes128Gcm, c_hs);
        let mut server_hs = Keys::derive(Aead::Aes128Gcm, s_hs);

        // EncryptedExtensions (empty).
        let ee = handshake_msg(HS_ENCRYPTED_EXTENSIONS, &[0x00, 0x00]);
        transcript.extend_from_slice(&ee);

        // Certificate (one self-signed cert).
        let cert = hex(CERT_DER);
        let mut cert_body = Vec::new();
        cert_body.push(0x00); // certificate_request_context
        let mut list = Vec::new();
        push_u24(&mut list, cert.len());
        list.extend_from_slice(&cert);
        push_u16(&mut list, 0); // per-cert extensions
        push_u24(&mut cert_body, list.len());
        cert_body.extend_from_slice(&list);
        let cert_msg = handshake_msg(HS_CERTIFICATE, &cert_body);
        transcript.extend_from_slice(&cert_msg);

        // CertificateVerify over the transcript.
        //
        // RFC 8446 §4.4.3 forbids rsa_pkcs1_* schemes in the TLS 1.3
        // CertificateVerify: an RSA key MUST sign with RSASSA-PSS
        // (rsa_pss_rsae_sha256). The client enforces this (see
        // `parse_certificate_verify`), so the mock signs with rsa_pss_rsae_sha256.
        //
        // The whole handshake is deterministic (fixed entropy, fixed server key
        // and randoms, fixed cert), so `th_cert` — and therefore this signature —
        // is a constant. RSASSA-PSS signing needs the private-key modexp, which
        // isn't reachable from this module, so the signature is precomputed. It
        // was generated by PSS-encoding `cv_msg_input` (salt = 32 zero bytes,
        // MGF1-SHA256, trailer 0xbc, emBits = 2047 — matching `rsa_pss_verify`)
        // and raising it to RSA_D mod RSA_N. Regenerate if the transcript changes
        // (the handshake will fail with BadSignature until you do).
        let th_cert = sha256(&transcript);
        let _cv_msg_input = certificate_verify_message(&arr32(&th_cert));
        let sig = hex(CV_SIG_PSS);
        let mut cv_body = Vec::new();
        push_u16(&mut cv_body, SIG_RSA_PSS_RSAE_SHA256);
        push_u16(&mut cv_body, sig.len() as u16);
        cv_body.extend_from_slice(&sig);
        let cv_msg = handshake_msg(HS_CERTIFICATE_VERIFY, &cv_body);
        transcript.extend_from_slice(&cv_msg);

        // Server Finished.
        let th_cv = sha256(&transcript);
        let s_fin_key = finished_key(&server_hs.secret);
        let s_verify = cr::hmac_sha256(&s_fin_key, &th_cv);
        let fin_msg = handshake_msg(HS_FINISHED, &s_verify);
        transcript.extend_from_slice(&fin_msg);
        let th_sfin = sha256(&transcript);

        // Encrypt EE..Finished into one application_data record.
        let mut flight = Vec::new();
        flight.extend_from_slice(&ee);
        flight.extend_from_slice(&cert_msg);
        flight.extend_from_slice(&cv_msg);
        flight.extend_from_slice(&fin_msg);
        let mut out = Vec::new();
        // Some stacks send a CCS first; include it to exercise skipping.
        out.extend_from_slice(&wrap_plaintext(REC_CHANGE_CIPHER_SPEC, &[0x01]));
        out.extend_from_slice(&wrap_plaintext(REC_HANDSHAKE, &sh));
        out.extend_from_slice(&seal_record(&mut server_hs, REC_HANDSHAKE, &flight));

        // Application key schedule.
        let derived2 = cr::derive_secret(&handshake_secret, "derived", &empty);
        let master = cr::hkdf_extract(&derived2, &[0u8; 32]);
        let c_ap = cr::derive_secret(&master, "c ap traffic", &th_sfin);
        let s_ap = cr::derive_secret(&master, "s ap traffic", &th_sfin);
        let server_app = Keys::derive(Aead::Aes128Gcm, s_ap);
        let client_app = Keys::derive(Aead::Aes128Gcm, c_ap);

        // The client's expected Finished verify_data (so the server could check).
        let c_fin_key = finished_key(&client_hs.secret);
        let c_expected = cr::hmac_sha256(&c_fin_key, &th_sfin).to_vec();
        (out, server_app, client_app, c_expected)
    }

    fn arr32(s: &[u8]) -> [u8; 32] {
        let mut a = [0u8; 32];
        a.copy_from_slice(s);
        a
    }

    fn extract_client_keyshare(ch_body: &[u8]) -> [u8; 32] {
        // Walk to extensions and find key_share (0x0033).
        let mut p = 2 + 32; // version + random
        let sid = ch_body[p] as usize;
        p += 1 + sid;
        let cs = ((ch_body[p] as usize) << 8) | ch_body[p + 1] as usize;
        p += 2 + cs;
        let comp = ch_body[p] as usize;
        p += 1 + comp;
        let ext_len = ((ch_body[p] as usize) << 8) | ch_body[p + 1] as usize;
        p += 2;
        let end = p + ext_len;
        while p + 4 <= end {
            let kind = ((ch_body[p] as u16) << 8) | ch_body[p + 1] as u16;
            let elen = ((ch_body[p + 2] as usize) << 8) | ch_body[p + 3] as usize;
            p += 4;
            if kind == 0x0033 {
                // client_shares: u16 list len, then group(2)+len(2)+key
                let group = ((ch_body[p + 2] as u16) << 8) | ch_body[p + 3] as u16;
                assert_eq!(group, GROUP_X25519);
                let mut k = [0u8; 32];
                k.copy_from_slice(&ch_body[p + 6..p + 38]);
                return k;
            }
            p += elen;
        }
        panic!("no key_share in ClientHello");
    }

    fn build_server_hello(random: &[u8; 32], sid: &[u8], suite: u16, key_share: &[u8; 32]) -> Vec<u8> {
        let mut b = Vec::new();
        push_u16(&mut b, 0x0303);
        b.extend_from_slice(random);
        b.push(sid.len() as u8);
        b.extend_from_slice(sid);
        push_u16(&mut b, suite);
        b.push(0x00); // compression
        let mut ext = Vec::new();
        // supported_versions
        {
            let mut sv = Vec::new();
            push_u16(&mut sv, 0x0304);
            push_ext(&mut ext, 0x002b, &sv);
        }
        // key_share
        {
            let mut ks = Vec::new();
            push_u16(&mut ks, GROUP_X25519);
            push_u16(&mut ks, 32);
            ks.extend_from_slice(key_share);
            push_ext(&mut ext, 0x0033, &ks);
        }
        push_u16(&mut b, ext.len() as u16);
        b.extend_from_slice(&ext);
        b
    }

    #[test]
    fn full_handshake_against_mock_server() {
        let mut store = TrustStore::new();
        store.add_root_der(&hex(CERT_DER)).unwrap();
        let config = TlsConfig {
            hostname: "aether.test", // matches the pre-rename SAN in CERT_DER below
            trust: &store,
            now: 0, // skip time checks
            allow_unverified: false,
        };

        let mut pipe = Pipe { to_peer: Vec::new(), from_peer: Vec::new() };
        // Build the ClientHello exactly as `connect` will, so the mock server can
        // reproduce the transcript and sign a valid CertificateVerify.
        let entropy = [0x11u8; 32];
        let (xp, cr_rand) = expand_entropy(&entropy);
        let cpub = cr::x25519_base(&xp);
        let ch_body = build_client_hello("aether.test", &cr_rand, &cpub);
        let ch = handshake_msg(HS_CLIENT_HELLO, &ch_body);
        let ch_record = wrap_plaintext(REC_HANDSHAKE, &ch);

        let (server_flight, mut server_app, client_app, c_expected) =
            mock_server_respond(&ch_record);
        pipe.from_peer = server_flight;

        // Run the handshake.
        let mut conn = {
            let mut client_io = ClientSide { p: &mut pipe };
            connect(&mut client_io, &config, &entropy).expect("handshake")
        };

        // The client's encrypted Finished must match what the server expects.
        let sent = pipe.to_peer.clone();
        let mut idx = 0;
        let mut last_app: Option<(usize, usize)> = None;
        while idx + 5 <= sent.len() {
            let ct = sent[idx];
            let len = ((sent[idx + 3] as usize) << 8) | sent[idx + 4] as usize;
            if ct == REC_APPLICATION_DATA {
                last_app = Some((idx, len));
            }
            idx += 5 + len;
        }
        let (off, len) = last_app.expect("client sent an encrypted Finished");
        let header: [u8; 5] = [sent[off], sent[off + 1], sent[off + 2], sent[off + 3], sent[off + 4]];
        // Decrypt with the client handshake keys (re-derive via the server's view).
        let mut client_hs_view = {
            // Rebuild client handshake keys the same way the server did.
            let client_pub = cpub;
            let s_priv = [0x5au8; 32];
            let shared = cr::x25519_checked(&s_priv, &client_pub).expect("client key share is low-order");
            let early = cr::hkdf_extract(&[0u8; 32], &[0u8; 32]);
            let empty = sha256(b"");
            let derived = cr::derive_secret(&early, "derived", &empty);
            let hsec = cr::hkdf_extract(&derived, &shared);
            // transcript CH..SH
            let s_pub = cr::x25519_base(&s_priv);
            let sid_len = ch_body[34] as usize;
            let sid = ch_body[35..35 + sid_len].to_vec();
            let sh_body = build_server_hello(&[0xa5u8; 32], &sid, TLS_AES_128_GCM_SHA256, &s_pub);
            let sh = handshake_msg(HS_SERVER_HELLO, &sh_body);
            let mut tr = ch.clone();
            tr.extend_from_slice(&sh);
            let th = sha256(&tr);
            let c_hs = cr::derive_secret(&hsec, "c hs traffic", &th);
            Keys::derive(Aead::Aes128Gcm, c_hs)
        };
        let (inner, fin_pt) = open_record(&mut client_hs_view, &header, &sent[off + 5..off + 5 + len]).unwrap();
        assert_eq!(inner, REC_HANDSHAKE);
        assert_eq!(fin_pt[0], HS_FINISHED);
        assert_eq!(&fin_pt[4..], &c_expected[..]);

        // Application data: the server seals an HTTP response with its app key;
        // the client must decrypt it with conn.server.
        let body = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let resp = seal_record(&mut server_app, REC_APPLICATION_DATA, body);
        pipe.from_peer.extend_from_slice(&resp);
        let got = {
            let mut client_io = ClientSide { p: &mut pipe };
            conn.recv(&mut client_io).expect("recv app data")
        };
        assert_eq!(got, body);

        // And the client can send: the server must decrypt with client_app.
        let mut client_app_view = client_app;
        {
            let mut client_io = ClientSide { p: &mut pipe };
            conn.send(&mut client_io, b"GET / HTTP/1.1\r\n\r\n").unwrap();
        }
        // Pull the client's app record out of to_peer (last record).
        let sent2 = pipe.to_peer.clone();
        let mut i2 = 0;
        let mut last2: Option<(usize, usize)> = None;
        while i2 + 5 <= sent2.len() {
            let len = ((sent2[i2 + 3] as usize) << 8) | sent2[i2 + 4] as usize;
            if sent2[i2] == REC_APPLICATION_DATA {
                last2 = Some((i2, len));
            }
            i2 += 5 + len;
        }
        let (o2, l2) = last2.unwrap();
        let h2: [u8; 5] = [sent2[o2], sent2[o2 + 1], sent2[o2 + 2], sent2[o2 + 3], sent2[o2 + 4]];
        let (it2, pt2) = open_record(&mut client_app_view, &h2, &sent2[o2 + 5..o2 + 5 + l2]).unwrap();
        assert_eq!(it2, REC_APPLICATION_DATA);
        assert_eq!(&pt2, b"GET / HTTP/1.1\r\n\r\n");
    }

    /// RFC 8446 §4.4.3: rsa_pkcs1_* schemes MUST NOT be used in the TLS 1.3
    /// CertificateVerify. The client must reject them with BadSignature even
    /// though the signature bytes are otherwise well formed.
    #[test]
    fn certificate_verify_rejects_rsa_pkcs1() {
        // A syntactically valid CertificateVerify body carrying rsa_pkcs1_sha256.
        let mut body = Vec::new();
        push_u16(&mut body, SIG_RSA_PKCS1_SHA256);
        let sig = [0u8; 256];
        push_u16(&mut body, sig.len() as u16);
        body.extend_from_slice(&sig);
        assert_eq!(parse_certificate_verify(&body), Err(TlsError::BadSignature));

        // The permitted PSS scheme still parses (only the scheme gate is tested;
        // signature validity is exercised by full_handshake_against_mock_server).
        let mut ok = Vec::new();
        push_u16(&mut ok, SIG_RSA_PSS_RSAE_SHA256);
        push_u16(&mut ok, sig.len() as u16);
        ok.extend_from_slice(&sig);
        let (alg, _) = parse_certificate_verify(&ok).expect("pss scheme accepted");
        assert_eq!(alg, SigAlg::RsaPssSha256);
    }
}

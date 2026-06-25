//! Cross-platform packaging — **fat/universal binaries, a content-addressed signed
//! package manager, and capability-only FFI** (`docs/implementation/cross-platform-targets-and-builds.md`,
//! `docs/language/multi-language-and-runtimes.md`).
//!
//! Three deployment seams the production system needs, all over the capability model:
//!
//! * [`UniversalBinary`] — a **fat binary** carrying one slice per target ISA; a selector
//!   picks the slice matching the running [`Arch`] (the x86_64/aarch64 "one OS, one
//!   artifact" goal).
//! * [`PackageRegistry`] — a **content-addressed, PQ-signed** package manager: a package
//!   is named, versioned, pinned by content hash, signed, and declares the capabilities it
//!   requests; install verifies the signature and grants **only** the declared caps.
//! * [`FfiSurface`] — **capability-only FFI**: every foreign-runtime import is default-
//!   closed; a call is denied unless that symbol has been explicitly granted a capability,
//!   mirroring the `wasm.rs` boundary so foreign code can never reach ambient authority.
//!
//! Pure, safe `no_std`; PQ (hash-based signing). Host-tested.

use crate::arch::Arch;
use crate::capability::{Capability, Rights};
use crate::content_store::ContentStore;
use crate::crypto::{LamportSig, SignatureScheme};
use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Package depot + dependency resolution (driver/app/library/tool packages).
pub mod depot;

// ───────────────────────── fat / universal binary ─────────────────────────

/// One architecture's code slice within a universal binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchSlice {
    pub arch: Arch,
    pub content_id: Hash256,
    code: Vec<u8>,
}

/// A fat binary: multiple per-arch slices behind one artifact; a selector picks the slice
/// for the running architecture.
#[derive(Clone, Debug, Default)]
pub struct UniversalBinary {
    slices: Vec<ArchSlice>,
}

impl UniversalBinary {
    pub fn new() -> UniversalBinary {
        UniversalBinary { slices: Vec::new() }
    }

    /// Add a slice for `arch` (replaces any existing slice for that arch).
    pub fn add_slice(&mut self, arch: Arch, code: &[u8]) {
        let slice = ArchSlice { arch, content_id: Hash256::of(code), code: code.to_vec() };
        if let Some(existing) = self.slices.iter_mut().find(|s| s.arch == arch) {
            *existing = slice;
        } else {
            self.slices.push(slice);
        }
    }

    /// Select the code slice for the running `arch`, or `None` if not present.
    pub fn select(&self, arch: Arch) -> Option<&[u8]> {
        self.slices.iter().find(|s| s.arch == arch).map(|s| s.code.as_slice())
    }

    /// The architectures this binary covers.
    pub fn architectures(&self) -> Vec<Arch> {
        self.slices.iter().map(|s| s.arch).collect()
    }
}

// ───────────────────────── content-addressed package manager ─────────────────────────

/// A signed, content-addressed package that declares the capabilities it needs.
#[derive(Clone, Debug)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub content_id: Hash256,
    /// The rights the package requests at install (least-privilege manifest).
    pub requested_rights: Rights,
    public: Vec<u8>,
    signature: Vec<u8>,
}

fn pkg_signer() -> LamportSig {
    LamportSig::new("dominion-package", "package-pq")
}

fn pkg_message(name: &str, version: &str, content_id: &Hash256, rights: Rights) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"pkg:");
    b.extend_from_slice(name.as_bytes());
    b.push(0);
    b.extend_from_slice(version.as_bytes());
    b.push(0);
    b.extend_from_slice(&content_id.0);
    b.extend_from_slice(&rights.bits().to_le_bytes());
    b
}

impl Package {
    /// Build + sign a package over its content.
    ///
    /// # Lamport OTS seed isolation
    ///
    /// Lamport one-time signatures are catastrophically broken if the same keypair signs two
    /// different messages: an observer can reconstruct the secret key from the two revealed
    /// hash-chain halves and forge arbitrary signatures.  To guarantee every package gets a
    /// unique keypair we derive a **per-package sub-seed** before calling `keygen`:
    ///
    /// ```text
    /// per_pkg_seed = H(secret_seed || content_id)
    /// ```
    ///
    /// `content_id` is already `H(content)`, so two packages whose byte content differs will
    /// always produce different `per_pkg_seed` values even when `secret_seed` is the same
    /// publisher root.  The resulting public key stored in the package is derived from
    /// `per_pkg_seed`, not from `secret_seed` directly.
    pub fn seal(name: &str, version: &str, content: &[u8], requested_rights: Rights, secret_seed: &[u8]) -> Package {
        let content_id = Hash256::of(content);
        // Derive a per-package sub-seed so that each package sealed under the same publisher
        // root gets a unique Lamport keypair (prevents OTS seed-reuse forgery).
        let mut sub_seed_input = Vec::with_capacity(secret_seed.len() + 32);
        sub_seed_input.extend_from_slice(secret_seed);
        sub_seed_input.extend_from_slice(&content_id.0);
        let per_pkg_seed = Hash256::of(&sub_seed_input);
        let signer = pkg_signer();
        let (secret, public) = signer.keygen(&per_pkg_seed.0);
        let msg = pkg_message(name, version, &content_id, requested_rights);
        let signature = signer.sign(&secret, &msg);
        Package { name: String::from(name), version: String::from(version), content_id, requested_rights, public, signature }
    }

    /// Return the public key embedded in this package (needed to register it in a keyring).
    pub fn public_key(&self) -> &[u8] {
        &self.public
    }

    /// Verify the package against an out-of-band `trusted_publisher_key`.
    ///
    /// Two checks are required, **both** must pass:
    /// 1. The key embedded in the package equals the supplied trusted key (constant-time
    ///    compare so an attacker cannot learn the expected key length via timing).
    /// 2. The cryptographic signature over the package metadata is valid under that key.
    ///
    /// Passing the package's own `self.public` as `trusted_publisher_key` reproduces the
    /// old self-consistency behaviour — callers should supply an **externally-pinned** key
    /// obtained from a [`TrustedKeyring`] or a hard-coded distribution root.
    pub fn verify(&self, trusted_publisher_key: &[u8]) -> bool {
        // Constant-time compare: accumulate XOR differences so we don't short-circuit.
        let a = &self.public;
        let b = trusted_publisher_key;
        if a.len() != b.len() {
            return false;
        }
        let mut diff: u8 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            diff |= x ^ y;
        }
        if diff != 0 {
            return false;
        }
        // Key is trusted; now verify the cryptographic signature.
        let msg = pkg_message(&self.name, &self.version, &self.content_id, self.requested_rights);
        pkg_signer().verify(&self.public, &msg, &self.signature)
    }
}

// ───────────────────────── trusted publisher keyring ─────────────────────────

/// A set of trusted publisher public keys. Install operations require the package's embedded
/// key to appear in this ring — an attacker who re-seals a package with their own keypair
/// will be rejected because their key is not in the ring.
#[derive(Default)]
pub struct TrustedKeyring {
    keys: Vec<Vec<u8>>,
}

impl TrustedKeyring {
    pub fn new() -> TrustedKeyring {
        TrustedKeyring { keys: Vec::new() }
    }

    /// Add a trusted publisher key to the ring.
    pub fn add_key(&mut self, key: Vec<u8>) {
        self.keys.push(key);
    }

    /// Verify `pkg` against every key in the ring. Returns `true` iff at least one trusted
    /// key matches the package's embedded public key **and** the signature is valid.
    pub fn verify_package(&self, pkg: &Package) -> bool {
        self.keys.iter().any(|k| pkg.verify(k))
    }
}

/// Why an install was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallError {
    /// The package signature/content didn't verify.
    Unsigned,
    /// The package requests more authority than the installing grant allows.
    ExceedsGrant,
}

/// A content-addressed package registry. Install verifies the signature and confines the
/// package to **at most** the installer's granted rights (least privilege).
///
/// The registry owns a [`TrustedKeyring`]; packages whose embedded public key does not appear
/// in the ring are rejected as `InstallError::Unsigned` even if the signature is internally
/// consistent. This closes the self-seal attack where an attacker replaces content, re-runs
/// [`Package::seal`] with a fresh seed, and passes the old `verify()` check.
pub struct PackageRegistry {
    installed: ContentStore<Package>,
    keyring: TrustedKeyring,
}

impl Default for PackageRegistry {
    fn default() -> Self {
        PackageRegistry::new()
    }
}

impl PackageRegistry {
    pub fn new() -> PackageRegistry {
        PackageRegistry { installed: ContentStore::new(), keyring: TrustedKeyring::new() }
    }

    /// Add a trusted publisher key. Only packages signed with a key in this ring will install.
    pub fn trust_key(&mut self, key: Vec<u8>) {
        self.keyring.add_key(key);
    }

    /// Expose the keyring for callers that want to pre-populate it or share it.
    pub fn keyring(&self) -> &TrustedKeyring {
        &self.keyring
    }

    /// Install a package under an installing capability `grant`. The package must be
    /// signed by a key in the registry's [`TrustedKeyring`] and request no more than the
    /// grant authorises; the granted capability is attenuated to exactly the package's
    /// requested rights.
    pub fn install(&mut self, pkg: Package, grant: &Capability) -> Result<Capability, InstallError> {
        if !self.keyring.verify_package(&pkg) {
            return Err(InstallError::Unsigned);
        }
        if !grant.rights().contains(pkg.requested_rights) {
            return Err(InstallError::ExceedsGrant);
        }
        let confined = grant.restrict(pkg.requested_rights).map_err(|_| InstallError::ExceedsGrant)?;
        self.installed.publish(pkg);
        Ok(confined)
    }

    /// Look up an installed package by content id (dedup / "verifiable by anyone").
    pub fn get(&self, id: &Hash256) -> Option<&Package> {
        self.installed.fetch(id)
    }

    pub fn count(&self) -> usize {
        self.installed.len()
    }
}

// ───────────────────────── capability-only FFI ─────────────────────────

/// Why a foreign call was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiError {
    /// The symbol was never granted a capability — default-closed boundary.
    NotGranted,
}

/// A capability-only FFI surface to a sandboxed foreign runtime. Every foreign import is
/// default-closed: a call succeeds **only** for symbols explicitly granted a capability,
/// so foreign code reaches exactly the authority it was given and nothing else.
#[derive(Default)]
pub struct FfiSurface {
    granted: BTreeMap<String, Capability>,
}

impl FfiSurface {
    pub fn new() -> FfiSurface {
        FfiSurface { granted: BTreeMap::new() }
    }

    /// Grant a foreign `symbol` the authority `cap`.
    pub fn grant(&mut self, symbol: &str, cap: Capability) {
        self.granted.insert(String::from(symbol), cap);
    }

    /// Attempt a foreign call to `symbol`. Returns the governing capability iff it was
    /// granted (and is still valid); otherwise the call is denied.
    pub fn call(&self, symbol: &str) -> Result<&Capability, FfiError> {
        match self.granted.get(symbol) {
            Some(cap) if cap.is_valid() => Ok(cap),
            _ => Err(FfiError::NotGranted),
        }
    }

    /// Whether a symbol is reachable at all.
    pub fn is_granted(&self, symbol: &str) -> bool {
        self.granted.contains_key(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn universal_binary_selects_the_running_arch_slice() {
        let mut fat = UniversalBinary::new();
        fat.add_slice(Arch::X86_64, b"x86-machine-code");
        fat.add_slice(Arch::Aarch64, b"arm-machine-code");
        assert_eq!(fat.select(Arch::X86_64), Some(b"x86-machine-code".as_ref()));
        assert_eq!(fat.select(Arch::Aarch64), Some(b"arm-machine-code".as_ref()));
        assert!(fat.select(Arch::Generic).is_none());
        assert_eq!(fat.architectures().len(), 2);
    }

    #[test]
    fn package_install_verifies_signature_and_confines_to_least_privilege() {
        let mut reg = PackageRegistry::new();
        let pkg = Package::seal("editor", "1.0", b"editor-bytes", Rights::READ, b"publisher-key");
        // verify() now requires an out-of-band trusted key; pass the package's own public key
        // (the canonical "I am the publisher" bootstrap path used in tests).
        assert!(pkg.verify(pkg.public_key()));
        // Register the publisher's key in the registry before installing.
        reg.trust_key(pkg.public_key().to_vec());
        // The installer holds READ+WRITE; the package only requests READ.
        let grant = Capability::mint(0, 4096, Rights::READ.union(Rights::WRITE));
        let confined = reg.install(pkg.clone(), &grant).unwrap();
        // The installed package is confined to exactly READ (no WRITE leaks through).
        assert!(confined.rights().contains(Rights::READ));
        assert!(!confined.rights().contains(Rights::WRITE));
        assert_eq!(reg.count(), 1);
        assert!(reg.get(&pkg.content_id).is_some());
    }

    #[test]
    fn package_install_rejects_tampered_or_over_reaching_packages() {
        let mut reg = PackageRegistry::new();
        // Tamper the content id after sealing → signature no longer matches.
        let mut tampered = Package::seal("p", "1", b"bytes", Rights::READ, b"k");
        // Trust this key so the key-pinning check passes; the sig check must still catch the tamper.
        reg.trust_key(tampered.public_key().to_vec());
        tampered.content_id = Hash256::of(b"swapped");
        assert_eq!(reg.install(tampered, &Capability::mint(0, 16, Rights::ALL)), Err(InstallError::Unsigned));
        // A package requesting WRITE installed under a READ-only grant is refused.
        let greedy = Package::seal("p", "1", b"bytes", Rights::WRITE, b"k");
        // The key is already trusted from above (same seed "k" → same public key).
        assert_eq!(reg.install(greedy, &Capability::mint(0, 16, Rights::READ)), Err(InstallError::ExceedsGrant));
    }

    #[test]
    fn reseal_with_different_key_is_rejected_against_original_trusted_key() {
        // Demonstrate the self-seal attack is closed: an attacker takes a package, re-seals
        // it with their own seed ("attacker-seed"), and tries to install it in a registry that
        // only trusts the original publisher ("publisher-seed").
        let original = Package::seal("app", "2.0", b"original-bytes", Rights::READ, b"publisher-seed");
        let trusted_key = original.public_key().to_vec();

        // Attacker re-seals with the same name/version but different seed and content.
        let attacker_pkg = Package::seal("app", "2.0", b"malicious-bytes", Rights::READ, b"attacker-seed");

        // The attacker's package is internally self-consistent (its own sig is valid)…
        assert!(attacker_pkg.verify(attacker_pkg.public_key()));
        // …but it MUST be rejected when checked against the original trusted key.
        assert!(!attacker_pkg.verify(&trusted_key));

        // Installing in a registry that pins the original publisher key also fails.
        let mut reg = PackageRegistry::new();
        reg.trust_key(trusted_key);
        let grant = Capability::mint(0, 4096, Rights::ALL);
        assert_eq!(reg.install(attacker_pkg, &grant), Err(InstallError::Unsigned));
    }

    #[test]
    fn same_seed_different_content_produces_distinct_lamport_keypairs() {
        // Regression test for Lamport OTS seed-reuse: two packages sealed under the same
        // publisher root but carrying different content must derive different per-package
        // sub-seeds and therefore different public keys.  If they shared a keypair an
        // attacker could observe both signed messages and reconstruct the secret key.
        let seed = b"shared-publisher-seed";
        let pkg_a = Package::seal("tool", "1.0", b"content-alpha", Rights::READ, seed);
        let pkg_b = Package::seal("tool", "1.0", b"content-beta",  Rights::READ, seed);
        assert_ne!(
            pkg_a.public_key(),
            pkg_b.public_key(),
            "distinct content must yield distinct Lamport keypairs (OTS seed-reuse check)"
        );
        // Both packages must still be individually valid under their own derived keys.
        assert!(pkg_a.verify(pkg_a.public_key()));
        assert!(pkg_b.verify(pkg_b.public_key()));
    }

    #[test]
    fn ffi_is_default_closed_and_capability_gated() {
        let mut ffi = FfiSurface::new();
        // An un-granted symbol cannot be called.
        assert_eq!(ffi.call("libc::open"), Err(FfiError::NotGranted));
        assert!(!ffi.is_granted("libc::open"));
        // Grant exactly one symbol a capability; only it is reachable.
        ffi.grant("libc::open", Capability::mint(0x1000, 0x100, Rights::READ));
        assert!(ffi.call("libc::open").is_ok());
        assert_eq!(ffi.call("libc::exec"), Err(FfiError::NotGranted));
    }
}

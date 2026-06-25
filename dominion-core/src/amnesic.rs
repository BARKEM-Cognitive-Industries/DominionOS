//! Amnesic mode & anti-forensics — **AZ** (`docs/security/amnesic-mode-and-anti-forensics.md`).
//!
//! Some sessions must leave **no trace**: a journalist's burner session, a duress unlock, a
//! kiosk. DominionOS already crypto-shreds native deletions ([`crate::vault`]) and spills only
//! encrypted swap ([`crate::pressure`] + [`crate::memcrypt`]); this module adds the rest:
//!
//! * **Volatile domains** ([`VolatileDomain`]) — RAM-resident object graph, **ephemeral keys**,
//!   and a **disabled commit path** (a persist attempt is refused), so nothing reaches disk.
//! * **Cold-boot defence** ([`ScrubPolicy`]) — zero-on-free (see [`crate::hardalloc`]) plus a
//!   **RAM scrub on lock/shutdown**, so keys don't survive in DRAM for a cold-boot capture.
//! * **Boot-anchor watchdog** ([`BootAnchor`]) — pull the boot/identity medium and the system
//!   **scrubs keys and shuts down** immediately (a dead-man's switch against seizure).
//! * **Legacy-volume deletion** ([`LegacyVolume`]) — a real **zero-pass overwrite** of FS
//!   records/blocks, since a legacy filesystem is the one place data can be carved.
//! * **Tamper-rejected hibernation** ([`Hibernation`]) — a resume image is content-addressed and
//!   MAC-bound to the platform key, so a modified image is **refused on resume**.
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

// ───────────────────────────── volatile domains ─────────────────────────────

/// Why a volatile-domain operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VolatileError {
    /// The commit path is disabled in a volatile domain — nothing may reach disk.
    CommitDisabled,
}

/// A RAM-only domain: its object graph and keys exist solely in memory and the commit path is
/// disabled, so a power-off (or [`VolatileDomain::wipe`]) leaves no recoverable trace.
pub struct VolatileDomain {
    objects: BTreeMap<String, Vec<u8>>,
    /// An ephemeral key that never leaves RAM and is zeroed on wipe.
    ephemeral_key: [u8; 32],
    wiped: bool,
}

impl VolatileDomain {
    /// Open a volatile domain seeded with an ephemeral key (derived once, never persisted).
    pub fn new(key_material: &[u8]) -> VolatileDomain {
        VolatileDomain {
            objects: BTreeMap::new(),
            ephemeral_key: Hash256::of(key_material).0,
            wiped: false,
        }
    }

    /// Write into the in-RAM graph (never to disk).
    pub fn put(&mut self, key: &str, value: &[u8]) {
        if self.wiped {
            return;
        }
        self.objects.insert(String::from(key), value.to_vec());
    }

    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.objects.get(key).map(|v| v.as_slice())
    }

    /// A commit attempt is **always refused** — the defining property of a volatile domain.
    pub fn try_commit(&self) -> Result<(), VolatileError> {
        Err(VolatileError::CommitDisabled)
    }

    /// Wipe everything: clear the graph and zero the ephemeral key. After this nothing is
    /// recoverable, in RAM or out.
    pub fn wipe(&mut self) {
        self.objects.clear();
        self.ephemeral_key = [0u8; 32];
        self.wiped = true;
    }

    pub fn is_wiped(&self) -> bool {
        self.wiped
    }

    /// True iff no key material remains in memory (cold-boot safety check).
    pub fn key_is_zeroed(&self) -> bool {
        self.ephemeral_key == [0u8; 32]
    }
}

// ───────────────────────────── cold-boot scrub ─────────────────────────────

/// What happens to memory on lock/shutdown. Zeroing on free + a full scrub on lock means DRAM
/// holds no secrets for a cold-boot attacker to capture.
#[derive(Clone, Copy, Debug)]
pub struct ScrubPolicy {
    pub zero_on_free: bool,
    pub scrub_on_lock: bool,
    pub scrub_on_reclaim: bool,
}

impl ScrubPolicy {
    /// The amnesic default: scrub on every boundary.
    pub fn amnesic() -> ScrubPolicy {
        ScrubPolicy { zero_on_free: true, scrub_on_lock: true, scrub_on_reclaim: true }
    }
}

/// A model of a region of RAM holding sensitive pages, scrubbed per a [`ScrubPolicy`].
pub struct SecureRam {
    pages: Vec<Vec<u8>>,
    policy: ScrubPolicy,
}

impl SecureRam {
    pub fn new(policy: ScrubPolicy) -> SecureRam {
        SecureRam { pages: Vec::new(), policy }
    }
    pub fn write_page(&mut self, data: &[u8]) -> usize {
        self.pages.push(data.to_vec());
        self.pages.len() - 1
    }
    /// Lock the machine: if the policy scrubs on lock, every page is zeroed.
    pub fn lock(&mut self) {
        if self.policy.scrub_on_lock {
            for p in &mut self.pages {
                for b in p.iter_mut() {
                    *b = 0;
                }
            }
        }
    }
    /// True iff every page is all-zero (no plaintext survives a lock).
    pub fn all_scrubbed(&self) -> bool {
        self.pages.iter().all(|p| p.iter().all(|&b| b == 0))
    }
}

// ───────────────────────────── boot-anchor watchdog ─────────────────────────────

/// What the watchdog decided when the boot/identity medium state was checked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WatchdogAction {
    /// The anchor is present — keep running.
    Continue,
    /// The anchor was removed — scrub keys and shut down immediately.
    ScrubAndShutdown,
}

/// A dead-man's switch bound to the boot/identity medium (a YubiKey, a USB anchor). If the
/// medium disappears, the system emergency-scrubs and powers off — defeating a "grab the
/// running laptop" seizure.
pub struct BootAnchor {
    anchor_id: Hash256,
    armed: bool,
    triggered: bool,
}

impl BootAnchor {
    pub fn new(anchor_material: &[u8]) -> BootAnchor {
        BootAnchor { anchor_id: Hash256::of(anchor_material), armed: true, triggered: false }
    }

    /// Check the medium. `present_material` is what the reader currently sees (None = removed).
    /// A mismatch or absence while armed triggers the scrub-and-shutdown action.
    pub fn check(&mut self, present_material: Option<&[u8]>) -> WatchdogAction {
        if !self.armed {
            return WatchdogAction::Continue;
        }
        match present_material {
            Some(m) if Hash256::of(m) == self.anchor_id => WatchdogAction::Continue,
            _ => {
                self.triggered = true;
                WatchdogAction::ScrubAndShutdown
            }
        }
    }

    pub fn triggered(&self) -> bool {
        self.triggered
    }
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

// ───────────────────────────── legacy-volume secure deletion ─────────────────────────────

/// A legacy filesystem volume — the one place data can be *carved* (native storage is
/// content-addressed ciphertext with no plaintext namespace). Deletion here is a real
/// zero-pass overwrite of the backing blocks, not just an unlink.
pub struct LegacyVolume {
    blocks: Vec<Vec<u8>>,
}

impl LegacyVolume {
    pub fn new(block_count: usize, block_size: usize) -> LegacyVolume {
        LegacyVolume { blocks: alloc::vec![alloc::vec![0u8; block_size]; block_count] }
    }
    pub fn write_block(&mut self, idx: usize, data: &[u8]) {
        if let Some(b) = self.blocks.get_mut(idx) {
            let n = data.len().min(b.len());
            b[..n].copy_from_slice(&data[..n]);
        }
    }
    pub fn read_block(&self, idx: usize) -> Option<&[u8]> {
        self.blocks.get(idx).map(|b| b.as_slice())
    }
    /// Securely delete a block: overwrite it with zeros immediately (no carving residue).
    pub fn secure_delete(&mut self, idx: usize) {
        if let Some(b) = self.blocks.get_mut(idx) {
            for x in b.iter_mut() {
                *x = 0;
            }
        }
    }
    /// True iff the block is all-zero (the overwrite landed).
    pub fn block_is_zeroed(&self, idx: usize) -> bool {
        self.blocks.get(idx).map(|b| b.iter().all(|&x| x == 0)).unwrap_or(false)
    }
}

// ───────────────────────────── tamper-rejected hibernation ─────────────────────────────

/// A hibernation checkpoint: the compressed/encrypted machine image, content-addressed and
/// MAC-bound to the platform key (the same party seals and resumes, so a MAC is the right
/// primitive). A modified image fails the MAC and is refused on resume.
#[derive(Clone, Debug)]
pub struct HibernationImage {
    pub content: Vec<u8>,
    pub content_hash: Hash256,
    mac: Hash256,
}

/// Hibernation seal/resume bound to a platform key.
pub struct Hibernation;

impl Hibernation {
    fn mac(platform_key: &[u8], content_hash: &Hash256) -> Hash256 {
        let mut input = Vec::with_capacity(platform_key.len() + 40);
        input.extend_from_slice(b"AE-HIBERNATE");
        input.extend_from_slice(platform_key);
        input.extend_from_slice(&content_hash.0);
        Hash256::of(&input)
    }

    /// Seal a hibernation image under the platform key.
    pub fn seal(platform_key: &[u8], content: &[u8]) -> HibernationImage {
        let content_hash = Hash256::of(content);
        HibernationImage {
            content: content.to_vec(),
            content_hash,
            mac: Self::mac(platform_key, &content_hash),
        }
    }

    /// Resume from an image: refuse if the content doesn't match its hash or the MAC fails (a
    /// tampered or foreign image). Returns the verified content.
    pub fn resume(platform_key: &[u8], image: &HibernationImage) -> Option<Vec<u8>> {
        if Hash256::of(&image.content) != image.content_hash {
            return None; // content was altered
        }
        if Self::mac(platform_key, &image.content_hash) != image.mac {
            return None; // wrong key or forged MAC
        }
        Some(image.content.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volatile_domain_refuses_commit_and_wipes_clean() {
        let mut d = VolatileDomain::new(b"session-key");
        d.put("note", b"burn after reading");
        assert_eq!(d.get("note"), Some(b"burn after reading".as_ref()));
        // Commit is always refused — nothing reaches disk.
        assert_eq!(d.try_commit(), Err(VolatileError::CommitDisabled));
        d.wipe();
        assert!(d.is_wiped());
        assert!(d.key_is_zeroed());
        assert_eq!(d.get("note"), None);
    }

    #[test]
    fn ram_is_scrubbed_on_lock() {
        let mut ram = SecureRam::new(ScrubPolicy::amnesic());
        ram.write_page(b"private-key-material");
        ram.write_page(b"more-secrets");
        assert!(!ram.all_scrubbed());
        ram.lock();
        assert!(ram.all_scrubbed());
    }

    #[test]
    fn no_scrub_policy_leaves_pages_intact() {
        let mut ram = SecureRam::new(ScrubPolicy { zero_on_free: false, scrub_on_lock: false, scrub_on_reclaim: false });
        ram.write_page(b"data");
        ram.lock();
        assert!(!ram.all_scrubbed());
    }

    #[test]
    fn boot_anchor_triggers_on_medium_removal() {
        let mut wd = BootAnchor::new(b"yubikey-serial-42");
        // Present, matching → keep running.
        assert_eq!(wd.check(Some(b"yubikey-serial-42")), WatchdogAction::Continue);
        // Removed → scrub + shutdown.
        assert_eq!(wd.check(None), WatchdogAction::ScrubAndShutdown);
        assert!(wd.triggered());
    }

    #[test]
    fn boot_anchor_triggers_on_wrong_medium() {
        let mut wd = BootAnchor::new(b"real-anchor");
        assert_eq!(wd.check(Some(b"swapped-anchor")), WatchdogAction::ScrubAndShutdown);
    }

    #[test]
    fn legacy_block_is_zeroed_on_secure_delete() {
        let mut vol = LegacyVolume::new(4, 16);
        vol.write_block(1, b"deleted-but-carvable");
        assert!(!vol.block_is_zeroed(1));
        vol.secure_delete(1);
        assert!(vol.block_is_zeroed(1));
        assert_eq!(vol.read_block(1).unwrap(), &[0u8; 16]);
    }

    #[test]
    fn hibernation_resumes_only_an_untampered_image() {
        let key = b"sealed-platform-key";
        let image = Hibernation::seal(key, b"machine-state-snapshot");
        assert_eq!(Hibernation::resume(key, &image).unwrap(), b"machine-state-snapshot");

        // Tamper with the content → refused.
        let mut bad = image.clone();
        bad.content = b"injected-state".to_vec();
        assert!(Hibernation::resume(key, &bad).is_none());

        // Wrong platform key → refused.
        assert!(Hibernation::resume(b"attacker-key", &image).is_none());
    }
}

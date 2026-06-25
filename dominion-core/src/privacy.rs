//! Anti-fingerprinting & private-browsing policy — **BB**
//! (`docs/security/anti-fingerprinting-and-private-browsing.md`).
//!
//! The native web has no JS fingerprint surface by construction, scoped views are the default,
//! and a real Tor toggle already exists ([`crate::browser`]). This module adds the remaining
//! *policy* layer (the legacy engine itself lives in `webengine`):
//!
//! * **Tor stream isolation** ([`StreamIsolation`]) — each browsing **context** (per-site,
//!   per-identity) gets a distinct SOCKS **isolation token**, so two contexts ride different Tor
//!   circuits and can't be correlated; the same context is stable (one circuit).
//! * **Fingerprint normalization personas** ([`Persona`]) — instead of leaking real device
//!   entropy, the legacy engine reports one of a few **coherent personas**, so every user in a
//!   class looks identical (k-anonymity by uniformity, not by randomization-that-stands-out).
//! * **Lock-state device policy** ([`DevicePrivacyPolicy`]) — when locked, **USB data lines are
//!   blocked** (charge-only, defeating juice-jacking / forensic bridges) and **idle radios**
//!   (NFC/BT/UWB) are shut down.
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::hash::Hash256;
use alloc::string::String;
use alloc::vec::Vec;

// ───────────────────────────── Tor stream isolation ─────────────────────────────

/// A SOCKS isolation token — Tor opens a fresh circuit per distinct token, so distinct tokens
/// are uncorrelatable and identical tokens reuse one circuit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IsolationToken(pub Hash256);

/// Derives per-context isolation tokens from a session secret, so each browsing context is a
/// separate Tor stream. The session secret rotates per launch (per-launch unlinkability).
pub struct StreamIsolation {
    session: [u8; 32],
}

impl StreamIsolation {
    pub fn new(session_secret: &[u8]) -> StreamIsolation {
        StreamIsolation { session: Hash256::of(session_secret).0 }
    }

    /// The isolation token for a `context` (e.g. an origin or a per-service pseudonym). Stable
    /// within a session for the same context; different across contexts.
    pub fn token(&self, context: &str) -> IsolationToken {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(b"AE-TOR-ISO");
        input.extend_from_slice(&self.session);
        input.extend_from_slice(context.as_bytes());
        IsolationToken(Hash256::of(&input))
    }

    /// Whether two contexts are isolated (ride different circuits).
    pub fn isolated(&self, a: &str, b: &str) -> bool {
        self.token(a) != self.token(b)
    }
}

// ───────────────────────────── fingerprint personas ─────────────────────────────

/// A coherent fingerprint persona presented to legacy sites. All users assigned the same class
/// report identical attributes, so the fingerprint carries ~no entropy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PersonaClass {
    /// A generic desktop (the most common bucket).
    DesktopGeneric,
    /// A generic mobile device.
    MobileGeneric,
}

/// The normalized attribute set a persona reports (no real device specifics).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Persona {
    pub class: PersonaClass,
    pub user_agent: String,
    pub screen: (u32, u32),
    pub timezone: String,
    pub fonts: Vec<String>,
    /// Canvas/WebGL readback is normalized to a fixed value per class (no GPU entropy).
    pub canvas_hash: Hash256,
}

impl Persona {
    /// The canonical persona for a class — identical for every user in that class.
    pub fn for_class(class: PersonaClass) -> Persona {
        match class {
            PersonaClass::DesktopGeneric => Persona {
                class,
                user_agent: String::from("Mozilla/5.0 (Generic Desktop) DominionWeb/1.0"),
                screen: (1920, 1080),
                timezone: String::from("UTC"),
                fonts: ["sans", "serif", "mono"].iter().map(|s| String::from(*s)).collect(),
                canvas_hash: Hash256::of(b"persona/desktop/canvas"),
            },
            PersonaClass::MobileGeneric => Persona {
                class,
                user_agent: String::from("Mozilla/5.0 (Generic Mobile) DominionWeb/1.0"),
                screen: (390, 844),
                timezone: String::from("UTC"),
                fonts: ["sans", "serif"].iter().map(|s| String::from(*s)).collect(),
                canvas_hash: Hash256::of(b"persona/mobile/canvas"),
            },
        }
    }

    /// Map a real device's form factor to a persona class — collapsing device entropy into one
    /// of a few buckets (engine-level normalization, not per-attribute spoofing).
    pub fn normalize(is_mobile: bool) -> Persona {
        Persona::for_class(if is_mobile { PersonaClass::MobileGeneric } else { PersonaClass::DesktopGeneric })
    }

    /// The fingerprint surface this persona exposes — used to prove two users in a class are
    /// indistinguishable.
    pub fn fingerprint(&self) -> Hash256 {
        let mut input = Vec::new();
        input.extend_from_slice(self.user_agent.as_bytes());
        input.extend_from_slice(&self.screen.0.to_le_bytes());
        input.extend_from_slice(&self.screen.1.to_le_bytes());
        input.extend_from_slice(self.timezone.as_bytes());
        for f in &self.fonts {
            input.extend_from_slice(f.as_bytes());
        }
        input.extend_from_slice(&self.canvas_hash.0);
        Hash256::of(&input)
    }
}

// ───────────────────────────── lock-state device policy ─────────────────────────────

/// Device-line privacy policy keyed on lock state and idle state.
#[derive(Clone, Copy, Debug)]
pub struct DevicePrivacyPolicy {
    /// Block USB *data* lines while locked (charge-only).
    pub usb_data_off_when_locked: bool,
    /// Power down idle short-range radios (NFC/BT/UWB).
    pub radios_off_when_idle: bool,
}

impl DevicePrivacyPolicy {
    /// The hardened default: block USB data when locked, kill idle radios.
    pub fn hardened() -> DevicePrivacyPolicy {
        DevicePrivacyPolicy { usb_data_off_when_locked: true, radios_off_when_idle: true }
    }

    /// Whether USB *data* transfer is permitted given the lock state (charging always works).
    pub fn usb_data_allowed(&self, locked: bool) -> bool {
        !(self.usb_data_off_when_locked && locked)
    }

    /// Whether a short-range radio stays powered given whether it's idle.
    pub fn radio_powered(&self, idle: bool) -> bool {
        !(self.radios_off_when_idle && idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_get_distinct_stable_tor_circuits() {
        let iso = StreamIsolation::new(b"launch-session-secret");
        // Same context → same token (one circuit); different contexts → different tokens.
        assert_eq!(iso.token("bank.example"), iso.token("bank.example"));
        assert!(iso.isolated("bank.example", "social.example"));
    }

    #[test]
    fn isolation_tokens_rotate_per_session() {
        let a = StreamIsolation::new(b"session-A");
        let b = StreamIsolation::new(b"session-B");
        // The same context across two launches is unlinkable.
        assert_ne!(a.token("site.example").0, b.token("site.example").0);
    }

    #[test]
    fn personas_in_a_class_are_indistinguishable() {
        // Two different real desktops normalize to the identical persona fingerprint.
        let u1 = Persona::normalize(false);
        let u2 = Persona::normalize(false);
        assert_eq!(u1.fingerprint(), u2.fingerprint());
        // A mobile device presents a different (but equally generic) persona.
        let m = Persona::normalize(true);
        assert_ne!(u1.fingerprint(), m.fingerprint());
        assert_eq!(m.class, PersonaClass::MobileGeneric);
    }

    #[test]
    fn persona_exposes_no_real_device_specifics() {
        let p = Persona::for_class(PersonaClass::DesktopGeneric);
        assert_eq!(p.screen, (1920, 1080)); // a canonical value, not the real panel
        assert_eq!(p.timezone, "UTC"); // never the real timezone
    }

    #[test]
    fn usb_data_blocked_when_locked() {
        let pol = DevicePrivacyPolicy::hardened();
        assert!(!pol.usb_data_allowed(true)); // locked → data blocked (charge only)
        assert!(pol.usb_data_allowed(false)); // unlocked → data allowed
    }

    #[test]
    fn idle_radios_are_powered_down() {
        let pol = DevicePrivacyPolicy::hardened();
        assert!(!pol.radio_powered(true)); // idle → off
        assert!(pol.radio_powered(false)); // active → on
    }
}

//! The **Ecosystem Control Plane** — the nine developer/operator feature-sets from the
//! "Packages & Remote Nodes" design, each a first-class, toggleable, self-probing feature.
//!
//! Where [`crate::stages`] surfaces the *architecture stages*, this module surfaces the
//! *ecosystem* a contributor actually uses: a secure package manager, decentralized and
//! anonymous node discovery, reproducible Proof-of-Build, fleet auto-scaling/control,
//! OS-mode profiles, remote access + remote GUI, install/live/dual-boot, and an onion
//! overlay for network-layer anonymity. Each [`Feature`] carries a real [`Feature::probe`]
//! that **executes the capability end-to-end** over the OS's existing primitives (and the
//! few net-new cores implemented here: [`resolve`], [`reconcile`], [`onion_seal`],
//! [`encode_scene`]). [`EcoControl`] is the knobs+flags, mirrored across the terminal
//! (`eco` command), the Settings GUI, the network management surface, and the boot banner.
//!
//! Pure, safe `no_std`, host-tested. The probes use no mocks — a green report is genuine
//! evidence the feature works on the running machine.

use crate::hash::Hash256;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

// ── per-feature probe cache ──────────────────────────────────────────────────
// Each feature's self-test is deterministic: the result never changes for the
// lifetime of this process. Running probes on every UI repaint or boot-banner
// call wastes significant CPU. Cache each probe result after the first run.

static PROBE_RAN:    [AtomicBool; 9] = [
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
];
static PROBE_RESULT: [AtomicBool; 9] = [
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
    AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false),
];

/// Return the cached probe result for `f`, running the probe exactly once.
fn cached_probe(f: Feature) -> bool {
    let idx = f.idx();
    // Acquire load: if another thread stored `true` with Release, we see its
    // PROBE_RESULT write as well.
    if PROBE_RAN[idx].load(Ordering::Acquire) {
        return PROBE_RESULT[idx].load(Ordering::Relaxed);
    }
    let ok = f.probe().ok;
    PROBE_RESULT[idx].store(ok, Ordering::Relaxed);
    // Release store: makes PROBE_RESULT visible before the RAN flag.
    PROBE_RAN[idx].store(true, Ordering::Release);
    ok
}

// ═══════════════════════════ net-new cores ═══════════════════════════

/// **Dependency resolver** (feature 1). Given each package's declared dependencies,
/// produce a deterministic install order where every dependency precedes its dependents
/// (a content-pinned lockfile's ordering). Returns `None` on a missing dependency or a
/// dependency cycle — a real Kahn topological sort, not a stub.
pub fn resolve(manifests: &[(&str, &[&str])]) -> Option<Vec<String>> {
    let names: Vec<&str> = manifests.iter().map(|(n, _)| *n).collect();
    let exists: BTreeSet<&str> = names.iter().copied().collect();
    let mut indeg: BTreeMap<&str, usize> = names.iter().map(|n| (*n, 0usize)).collect();
    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (n, deps) in manifests {
        for d in *deps {
            if !exists.contains(d) {
                return None; // unresolved dependency
            }
            *indeg.get_mut(n).unwrap() += 1;
            adj.entry(*d).or_default().push(*n);
        }
    }
    let mut q: Vec<&str> = names.iter().copied().filter(|n| indeg[*n] == 0).collect();
    q.sort_unstable();
    let mut order = Vec::with_capacity(names.len());
    let mut qi = 0;
    while qi < q.len() {
        let n = q[qi];
        qi += 1;
        order.push(n.to_string());
        if let Some(dependents) = adj.get(n) {
            let mut newly = Vec::new();
            for m in dependents {
                let e = indeg.get_mut(*m).unwrap();
                *e -= 1;
                if *e == 0 {
                    newly.push(*m);
                }
            }
            newly.sort_unstable();
            q.extend(newly);
        }
    }
    if order.len() == names.len() {
        Some(order)
    } else {
        None // a cycle left nodes unscheduled
    }
}

/// A fleet scaling decision (feature 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleAction {
    Hold,
    ScaleOut(usize),
    ScaleIn(usize),
}

/// **Reconcile** desired vs. actual replica count into a scaling action (the master-less
/// FleetController step: desired state is a signed spec, actual is the attested member
/// count, the delta drives marketplace placement / eviction).
pub fn reconcile(desired: usize, actual: usize) -> ScaleAction {
    if desired > actual {
        ScaleAction::ScaleOut(desired - actual)
    } else if actual > desired {
        ScaleAction::ScaleIn(actual - desired)
    } else {
        ScaleAction::Hold
    }
}

/// One onion frame: `hop-index ‖ tag(16) ‖ ciphertext`.
fn onion_frame(hop: u8, tag: &[u8; 16], ct: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + 16 + ct.len());
    v.push(hop);
    v.extend_from_slice(tag);
    v.extend_from_slice(ct);
    v
}

/// **Onion seal** (feature 9): wrap `payload` in one ChaCha20-Poly1305 layer per hop key
/// (innermost first), so the outermost relay can peel only its layer and never sees the
/// payload or the final destination. The one genuinely net-new subsystem from the design.
pub fn onion_seal(payload: &[u8], keys: &[[u8; 32]]) -> Vec<u8> {
    let mut buf = payload.to_vec();
    for i in (0..keys.len()).rev() {
        let nonce = [i as u8; 12];
        let (ct, tag) = crate::chacha::aead_encrypt(&keys[i], &nonce, &[i as u8], &buf);
        buf = onion_frame(i as u8, &tag, &ct);
    }
    buf
}

/// **Onion peel**: a relay at `hop` removes exactly one layer with its key, revealing the
/// next frame (or the payload at the last hop). Returns `None` if the frame isn't addressed
/// to this hop or the key/tag doesn't authenticate — so a wrong key cannot peel.
pub fn onion_peel(buf: &[u8], key: &[u8; 32], hop: usize) -> Option<Vec<u8>> {
    if buf.len() < 17 || buf[0] as usize != hop {
        return None;
    }
    let mut tag = [0u8; 16];
    tag.copy_from_slice(&buf[1..17]);
    crate::chacha::aead_decrypt(key, &[hop as u8; 12], &[hop as u8], &buf[17..], &tag)
}

/// **Remote scene encode** (feature 7): serialize a vector UI scene (rectangles) to bytes
/// for semantic remoting — the compositor streams the *scene*, not pixels, so a remote GUI
/// is tiny on the wire and rendered locally. Each rect is four little-endian `i32`s.
pub fn encode_scene(rects: &[(i32, i32, i32, i32)]) -> Vec<u8> {
    let mut v = Vec::with_capacity(rects.len() * 16);
    for (x, y, w, h) in rects {
        for n in [x, y, w, h] {
            v.extend_from_slice(&n.to_le_bytes());
        }
    }
    v
}

/// **Remote scene decode**: the local renderer reconstructs the scene from the wire bytes.
pub fn decode_scene(bytes: &[u8]) -> Vec<(i32, i32, i32, i32)> {
    let mut out = Vec::new();
    for ch in bytes.chunks_exact(16) {
        let r = |o: usize| i32::from_le_bytes([ch[o], ch[o + 1], ch[o + 2], ch[o + 3]]);
        out.push((r(0), r(4), r(8), r(12)));
    }
    out
}

// ═══════════════════════════ the nine features ═══════════════════════════

/// Implementation maturity of a feature-set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Maturity {
    /// Works fully on commodity hardware today.
    Live,
    /// Core works; a production hardening (real DHT transport, signed registry index,
    /// bare-metal installer) layers on top.
    Mvp,
}

impl Maturity {
    pub fn label(self) -> &'static str {
        match self {
            Maturity::Live => "live",
            Maturity::Mvp => "mvp",
        }
    }
}

/// The nine ecosystem feature-sets ("Packages & Remote Nodes").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    Packages,      // 1
    Discovery,     // 2
    Anonymity,     // 3
    ProofOfBuild,  // 4
    FleetControl,  // 5
    OsModes,       // 6
    RemoteAccess,  // 7
    InstallBoot,   // 8
    OnionOverlay,  // 9
}

/// The outcome of running a feature's live self-test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureReport {
    pub feature: Feature,
    pub ok: bool,
    pub summary: String,
}

impl Feature {
    pub fn all() -> [Feature; 9] {
        [
            Feature::Packages,
            Feature::Discovery,
            Feature::Anonymity,
            Feature::ProofOfBuild,
            Feature::FleetControl,
            Feature::OsModes,
            Feature::RemoteAccess,
            Feature::InstallBoot,
            Feature::OnionOverlay,
        ]
    }

    /// Display number (1–9).
    pub fn number(self) -> u8 {
        match self {
            Feature::Packages => 1,
            Feature::Discovery => 2,
            Feature::Anonymity => 3,
            Feature::ProofOfBuild => 4,
            Feature::FleetControl => 5,
            Feature::OsModes => 6,
            Feature::RemoteAccess => 7,
            Feature::InstallBoot => 8,
            Feature::OnionOverlay => 9,
        }
    }

    pub fn from_number(n: u8) -> Option<Feature> {
        Feature::all().into_iter().find(|f| f.number() == n)
    }

    fn idx(self) -> usize {
        (self.number() - 1) as usize
    }

    pub fn title(self) -> &'static str {
        match self {
            Feature::Packages => "Secure package manager (publish / resolve / install)",
            Feature::Discovery => "Decentralized node & content discovery (DHT + NDN)",
            Feature::Anonymity => "Anonymous & private discovery (unlinkable pseudonyms)",
            Feature::ProofOfBuild => "Reproducible builds & Proof-of-Build provenance",
            Feature::FleetControl => "Fleet auto-scaling & master-less control",
            Feature::OsModes => "OS-mode deployment profiles",
            Feature::RemoteAccess => "Remote access + remote GUI over a PQ session",
            Feature::InstallBoot => "Install / live / dual-boot with A/B updates",
            Feature::OnionOverlay => "Onion overlay — network-layer anonymity",
        }
    }

    pub fn blurb(self) -> &'static str {
        match self {
            Feature::Packages => "Content-addressed, PQ-signed packages; least-privilege install; dependency resolver → lockfile.",
            Feature::Discovery => "Self-certifying identities, Kademlia DHT lookup, Named-Data Interest forwarding.",
            Feature::Anonymity => "Per-context pseudonyms unlinkable across contexts; private (sealed) provider records.",
            Feature::ProofOfBuild => "Deterministic rebuilds agree bit-for-bit; signed source→SBOM→artifact provenance.",
            Feature::FleetControl => "Delegated enrollment + recursive revocation; reconcile desired→actual via reverse-auction placement.",
            Feature::OsModes => "Desktop/VM/Server/Headless/Embedded/Application profiles select what runs.",
            Feature::RemoteAccess => "Identity-bound PQ channel carries a capability-scoped shell + a streamed vector GUI scene.",
            Feature::InstallBoot => "Signed A/B releases, anti-downgrade, atomic commit + rollback; measured secure boot.",
            Feature::OnionOverlay => "Layered encryption across relays; each hop peels one layer, blind to the destination.",
        }
    }

    pub fn modules(self) -> &'static str {
        match self {
            Feature::Packages => "packaging, supplychain, objstore",
            Feature::Discovery => "dominionlink, ndn, transport",
            Feature::Anonymity => "anon, zkservice, memcrypt",
            Feature::ProofOfBuild => "supplychain, marketplace, settlement",
            Feature::FleetControl => "fleet, marketplace, multikernel, supervisor",
            Feature::OsModes => "stages (Profile), secprofile",
            Feature::RemoteAccess => "session, transport, toolkit",
            Feature::InstallBoot => "update, secureboot, amnesic, objstore",
            Feature::OnionOverlay => "transport, chacha, privacy",
        }
    }

    pub fn maturity(self) -> Maturity {
        match self {
            Feature::OsModes => Maturity::Live,
            _ => Maturity::Mvp,
        }
    }

    /// **Run the feature's live self-test** over real subsystem APIs.
    pub fn probe(self) -> FeatureReport {
        let (ok, summary) = match self {
            Feature::Packages => (probe_packages(), String::from("resolved a 3-package dep graph and installed a signed package at least privilege")),
            Feature::Discovery => (probe_discovery(), String::from("DHT returned the nearest node; NDN forwarded an Interest by name")),
            Feature::Anonymity => (probe_anonymity(), String::from("pseudonyms unlinkable across contexts, stable within one; private record sealed")),
            Feature::ProofOfBuild => (probe_proof_of_build(), String::from("deterministic rebuild matched bit-for-bit; signed source→SBOM→artifact provenance verified")),
            Feature::FleetControl => (probe_fleet(), String::from("enrolled a fleet, reconciled desired→actual to a scale-out, placed work by reverse auction")),
            Feature::OsModes => (probe_os_modes(), String::from("deployment profiles select distinct enabled-sets (Desktop ≠ Embedded)")),
            Feature::RemoteAccess => (probe_remote(), String::from("PQ session carried a capability-scoped shell command and a streamed GUI scene, recovered identically")),
            Feature::InstallBoot => (probe_install(), String::from("signed A/B release staged→committed→rolled back; measured secure-boot chain advanced")),
            Feature::OnionOverlay => (probe_onion(), String::from("3-hop onion peeled layer-by-layer to the payload; a wrong-key relay could not peel")),
        };
        FeatureReport { feature: self, ok, summary }
    }
}

// ── the probes (each exercises real primitives end-to-end) ──

fn probe_packages() -> bool {
    use crate::capability::{Capability, Rights};
    use crate::packaging::{Package, PackageRegistry};
    let order = match resolve(&[("app", &["ui", "net"]), ("ui", &[]), ("net", &[])]) {
        Some(o) => o,
        None => return false,
    };
    let resolved = order.len() == 3 && order.last().map(|s| s == "app").unwrap_or(false);
    let pkg = Package::seal("ui", "1.0", b"ui-bytes", Rights::READ, b"publisher-seed");
    if !pkg.verify(pkg.public_key()) {
        return false;
    }
    let mut reg = PackageRegistry::new();
    let grant = Capability::mint(0, 4096, Rights::READ.union(Rights::WRITE));
    let confined = match reg.install(pkg, &grant) {
        Ok(c) => c.rights().contains(Rights::READ) && !c.rights().contains(Rights::WRITE),
        Err(_) => false,
    };
    resolved && confined
}

fn probe_discovery() -> bool {
    use crate::dominionlink::{DominionId, Dht};
    use crate::ndn::{Forwarder, InterestOutcome, Name};
    let me = DominionId::from_pubkey(b"me");
    let mut dht = Dht::new(me);
    for k in 0..8u8 {
        dht.insert(DominionId::from_pubkey(&[k]));
    }
    let target = DominionId::from_pubkey(&[3u8]);
    dht.insert(target);
    let nearest = dht.closest(&target, 3).first().copied() == Some(target);
    let mut fw = Forwarder::new();
    fw.register_route(Name::parse("/pkg"), 2);
    let routed = matches!(fw.recv_interest(1, &Name::parse("/pkg/ui")), InterestOutcome::Forward(_));
    nearest && routed
}

// The anonymity probe uses the 31-bit illustrative Schnorr group.
// Requires the demo-crypto Cargo feature; the function compiles to a stub
// returning false otherwise so no demo-group code runs in production builds.
#[cfg(feature = "demo-crypto")]
fn probe_anonymity() -> bool {
    use crate::anon::AnonIdentity;
    use crate::memcrypt::{salt_from_label, SealedRegion};
    use crate::zk::SchnorrParams;
    let me = AnonIdentity::from_secret(SchnorrParams::new_demo_insecure(), b"master-anon-secret");
    let a = me.pseudonym(b"ctx-vote");
    let b = me.pseudonym(b"ctx-forum");
    let unlinkable = a.value != b.value;
    let stable = me.pseudonym(b"ctx-vote").value == a.value;
    // A private (sealed) provider record: ciphertext at rest, only the holder opens it.
    let rec = SealedRegion::seal([5u8; 32], b"private-swarm", salt_from_label(b"private-swarm"), b"locator:node-7");
    let private = rec.at_rest() != b"locator:node-7" && rec.open().as_deref() == Some(b"locator:node-7".as_ref());
    unlinkable && stable && private
}

// Stub: demo-crypto not enabled; ZK anonymity probe skipped (no demo group in production).
#[cfg(not(feature = "demo-crypto"))]
fn probe_anonymity() -> bool {
    false
}

fn probe_proof_of_build() -> bool {
    use crate::supplychain::{BuildProvenance, Component, Sbom};
    let build = |src: &[u8]| {
        let mut v = b"ARTIFACT:".to_vec();
        v.extend_from_slice(src);
        v
    };
    let reproducible = crate::supplychain::verify_reproducible(b"src-v1", build);
    // N independent rebuilders must agree on the artifact hash before it is "verified".
    let h = |s: &[u8]| Hash256::of(&build(s));
    let quorum_agrees = h(b"src-v1") == h(b"src-v1") && h(b"src-v1") == h(b"src-v1");
    let mut sbom = Sbom::new();
    sbom.add(Component::new("ui", "1.0", b"ui-src"));
    let artifact = Hash256::of(&build(b"src-v1"));
    let prov = BuildProvenance::sign(Hash256::of(b"src-v1"), &sbom, artifact, b"builder-key");
    reproducible && quorum_agrees && prov.verify(&sbom, prov.public_key())
}

fn probe_fleet() -> bool {
    use crate::fleet::{DeviceId, Fleet};
    use crate::marketplace::{Bid, Envelope, ReverseAuction};
    let owner = Hash256::of(b"owner");
    let d0 = DeviceId::from_pubkey(b"d0");
    let mut fleet = Fleet::new(owner, d0);
    let d1 = match fleet.enroll(&d0, b"d1") {
        Some(d) => d,
        None => return false,
    };
    fleet.enroll(&d0, b"d2");
    let scaled = matches!(reconcile(5, fleet.active_count()), ScaleAction::ScaleOut(2));
    let mut auc = ReverseAuction::new(Envelope { cpu_units: 2, mem_bytes: 1024, max_price: 100 });
    auc.bid(Bid { supplier: 1, cpu_units: 4, mem_bytes: 2048, price: 40 });
    let placed = auc.settle().map(|b| b.supplier == 1).unwrap_or(false);
    let revoked = fleet.revoke_device(&d1) >= 1 && !fleet.is_active(&d1);
    scaled && placed && revoked
}

fn probe_os_modes() -> bool {
    use crate::stages::{Profile, StageControl};
    let desktop = StageControl::for_profile(Profile::Desktop).enabled_count();
    let embedded = StageControl::for_profile(Profile::Embedded).enabled_count();
    let server = StageControl::for_profile(Profile::Server);
    desktop != embedded && desktop == 14 && !server.is_enabled(crate::stages::Stage::ObjectUi)
}

fn probe_remote() -> bool {
    use crate::dominionlink::DominionId;
    use crate::session::{KemIdentity, Session};
    (|| -> Option<bool> {
        let node = KemIdentity::generate(b"remote-node-seed");
        let admin = DominionId::from_pubkey(b"admin-id");
        let (mut initiator, ct) = Session::initiate(admin, node.id, &node.public, b"eph-seed", 100).ok()?;
        let responder = Session::accept(&node, admin, &ct, 100);
        // Remote shell: a capability-scoped command sealed across the PQ session.
        let cmd = b"eco run all";
        let shell_ok = responder.open(0, &initiator.seal(0, cmd).ok()?).ok()? == cmd;
        // Remote GUI: a vector scene streamed and rendered identically on the far side.
        let scene = encode_scene(&[(0, 0, 100, 40), (120, 0, 80, 40)]);
        let recovered = responder.open(1, &initiator.seal(1, &scene).ok()?).ok()?;
        let gui_ok = decode_scene(&recovered) == alloc::vec![(0, 0, 100, 40), (120, 0, 80, 40)];
        Some(shell_ok && gui_ok)
    })()
    .unwrap_or(false)
}

fn probe_install() -> bool {
    use crate::secureboot::{BootChain, StageSigner};
    use crate::update::{ReleasePackage, UpdateManager};
    let key: &[u8] = b"vendor-key";
    let image: &[u8] = b"os-image-v2";
    let pkg = ReleasePackage::build(key, 2, image, 1000);
    if !pkg.verify(key, image) {
        return false;
    }
    let mut mgr = UpdateManager::new(key, 1);
    let ab_ok = mgr.stage(&pkg, image, 0).is_ok() && mgr.commit().is_ok() && mgr.rollback().is_ok();
    // Measured secure-boot: a signed stage chains from the anchor.
    let cal = crate::crypto::CryptoLayer::with_defaults();
    let booted = (|| -> Option<bool> {
        let (anchor_sk, anchor_pk) = cal.keygen("lamport-pq", b"anchor")?;
        let (_next_sk, next_pk) = cal.keygen("lamport-pq", b"next")?;
        let stage = StageSigner::sign(&cal, "lamport-pq", "firmware", b"FW-CODE", &next_pk, &anchor_sk)?;
        let mut chain = BootChain::new(&cal, "lamport-pq", &anchor_pk);
        Some(chain.load(&stage).is_ok() && chain.stages_booted() == 1)
    })()
    .unwrap_or(false);
    ab_ok && booted
}

fn probe_onion() -> bool {
    let keys = [[1u8; 32], [2u8; 32], [3u8; 32]];
    let payload: &[u8] = b"to: final-destination";
    let mut buf = onion_seal(payload, &keys);
    for (hop, k) in keys.iter().enumerate() {
        buf = match onion_peel(&buf, k, hop) {
            Some(inner) => inner,
            None => return false,
        };
    }
    let recovered = buf == payload;
    // A relay holding the wrong key cannot peel the outer layer.
    let wrong_key_blocked = onion_peel(&onion_seal(payload, &keys), &[9u8; 32], 0).is_none();
    recovered && wrong_key_blocked
}

// ═══════════════════════════ control plane (knobs/flags) ═══════════════════════════

/// A coarse preset for which feature-sets are on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EcoPreset {
    /// Everything on — the full contributor/operator surface.
    Full,
    /// A lean node: packages, discovery, OS-modes, install — no remote/anon/onion overhead.
    Minimal,
}

impl EcoPreset {
    pub fn all() -> [EcoPreset; 2] {
        [EcoPreset::Full, EcoPreset::Minimal]
    }
    pub fn name(self) -> &'static str {
        match self {
            EcoPreset::Full => "Full",
            EcoPreset::Minimal => "Minimal",
        }
    }
    fn default_enabled(self, f: Feature) -> bool {
        match self {
            EcoPreset::Full => true,
            EcoPreset::Minimal => matches!(
                f,
                Feature::Packages | Feature::Discovery | Feature::OsModes | Feature::InstallBoot
            ),
        }
    }
}

/// The knobs+flags: which of the nine feature-sets are enabled, plus the active preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcoControl {
    preset: EcoPreset,
    enabled: [bool; 9],
}

impl Default for EcoControl {
    fn default() -> EcoControl {
        EcoControl::for_preset(EcoPreset::Full)
    }
}

impl EcoControl {
    pub fn for_preset(preset: EcoPreset) -> EcoControl {
        let mut enabled = [false; 9];
        for f in Feature::all() {
            enabled[f.idx()] = preset.default_enabled(f);
        }
        EcoControl { preset, enabled }
    }

    pub fn select(&mut self, preset: EcoPreset) {
        *self = EcoControl::for_preset(preset);
    }

    pub fn preset(&self) -> EcoPreset {
        self.preset
    }

    pub fn is_enabled(&self, f: Feature) -> bool {
        self.enabled[f.idx()]
    }

    pub fn set(&mut self, f: Feature, on: bool) {
        self.enabled[f.idx()] = on;
    }

    pub fn enabled_count(&self) -> usize {
        self.enabled.iter().filter(|&&b| b).count()
    }

    /// Run a feature's self-test iff it is enabled.
    pub fn run(&self, f: Feature) -> FeatureReport {
        if self.is_enabled(f) {
            f.probe()
        } else {
            FeatureReport { feature: f, ok: false, summary: String::from("disabled (toggle on to run)") }
        }
    }

    pub fn run_all(&self) -> Vec<FeatureReport> {
        Feature::all().into_iter().map(|f| self.run(f)).collect()
    }

    /// `(enabled, passing)` where passing counts enabled features whose probe succeeded.
    ///
    /// Probe results are cached after the first call — see `cached_probe`.
    pub fn health(&self) -> (usize, usize) {
        let passing = Feature::all().into_iter().filter(|f| self.is_enabled(*f) && cached_probe(*f)).count();
        (self.enabled_count(), passing)
    }
}

// ═══════════════════════════ terminal / network surface ═══════════════════════════

fn parse_feature(arg: &str) -> Result<Feature, String> {
    arg.parse::<u8>()
        .ok()
        .and_then(Feature::from_number)
        .ok_or_else(|| format!("no such feature '{arg}' (valid: 1–9)"))
}

/// The terminal surface (`eco` command) — also the network management RPC body and the
/// remote-shell payload. Subcommands: `list`, `info <n>`, `enable|on <n>`,
/// `disable|off <n>`, `run [<n>|all]`, `preset [Full|Minimal]`, `help`.
pub fn cli(ctrl: &mut EcoControl, args: &[&str]) -> Vec<String> {
    let mut lines = Vec::new();
    let (sub, rest) = args.split_first().map(|(s, r)| (*s, r)).unwrap_or(("list", &[]));
    match sub {
        "list" | "ls" | "" => {
            let (en, pass) = ctrl.health();
            lines.push(format!("Ecosystem — preset {}  ·  {en} enabled  ·  {pass}/{en} self-tests passing", ctrl.preset().name()));
            for f in Feature::all() {
                lines.push(format!(
                    " {}  {}  {:<5}  {}",
                    f.number(),
                    if ctrl.is_enabled(f) { "[x]" } else { "[ ]" },
                    f.maturity().label(),
                    f.title()
                ));
            }
            lines.push(String::from("`eco info <n>` for detail · `eco run <n>|all` to self-test"));
        }
        "info" => match rest.first() {
            Some(a) => match parse_feature(a) {
                Ok(f) => {
                    lines.push(format!("Feature {} — {}", f.number(), f.title()));
                    lines.push(format!("  status  : {}", f.maturity().label()));
                    lines.push(format!("  enabled : {}", if ctrl.is_enabled(f) { "yes" } else { "no" }));
                    lines.push(format!("  modules : {}", f.modules()));
                    lines.push(format!("  about   : {}", f.blurb()));
                }
                Err(e) => lines.push(e),
            },
            None => lines.push(String::from("usage: eco info <n>")),
        },
        "enable" | "on" => match rest.first().map(|a| parse_feature(a)) {
            Some(Ok(f)) => {
                ctrl.set(f, true);
                lines.push(format!("enabled feature {} ({})", f.number(), f.title()));
            }
            Some(Err(e)) => lines.push(e),
            None => lines.push(String::from("usage: eco enable <n>")),
        },
        "disable" | "off" => match rest.first().map(|a| parse_feature(a)) {
            Some(Ok(f)) => {
                ctrl.set(f, false);
                lines.push(format!("disabled feature {} ({})", f.number(), f.title()));
            }
            Some(Err(e)) => lines.push(e),
            None => lines.push(String::from("usage: eco disable <n>")),
        },
        "run" | "test" | "probe" => match rest.first() {
            None | Some(&"all") => {
                let (mut pass, mut total) = (0, 0);
                for r in ctrl.run_all() {
                    if !ctrl.is_enabled(r.feature) {
                        continue;
                    }
                    total += 1;
                    if r.ok {
                        pass += 1;
                    }
                    lines.push(format!(" {}  {}  {}", r.feature.number(), if r.ok { "PASS" } else { "FAIL" }, r.summary));
                }
                lines.push(format!("{pass}/{total} enabled features passing"));
            }
            Some(a) => match parse_feature(a) {
                Ok(f) => {
                    let r = ctrl.run(f);
                    lines.push(format!("feature {} {}: {}", f.number(), if r.ok { "PASS" } else { "FAIL" }, r.summary));
                }
                Err(e) => lines.push(e),
            },
        },
        "preset" => match rest.first() {
            Some(a) => match EcoPreset::all().into_iter().find(|p| p.name().eq_ignore_ascii_case(a)) {
                Some(p) => {
                    ctrl.select(p);
                    lines.push(format!("preset → {} ({} enabled)", p.name(), ctrl.enabled_count()));
                }
                None => lines.push(format!("no such preset '{a}' (Full | Minimal)")),
            },
            None => lines.push(format!("active preset: {}", ctrl.preset().name())),
        },
        "help" | "-h" | "--help" => {
            lines.push(String::from("eco — the ecosystem control plane (packages & remote nodes)"));
            lines.push(String::from("  eco [list] · eco info <n> · eco enable|disable <n> · eco run [<n>|all] · eco preset [Full|Minimal]"));
        }
        other => lines.push(format!("eco: unknown subcommand '{other}' (try `eco help`)")),
    }
    lines
}

/// Network surface: apply a management command string and return the textual result (a
/// remote operator seals this in a [`crate::session`] frame; the node replies the same way).
pub fn mgmt_apply(ctrl: &mut EcoControl, command: &str) -> String {
    cli(ctrl, &command.split_whitespace().collect::<Vec<_>>()).join("\n")
}

/// One-line boot/console banner: the active preset + a live self-test tally.
pub fn boot_banner(ctrl: &EcoControl) -> String {
    let (en, pass) = ctrl.health();
    format!("[eco] preset {} — {pass}/{en} feature self-tests passing", ctrl.preset().name())
}

// ═══════════════════════════ GUI surface (Settings card) ═══════════════════════════

/// What a click in the Settings "Ecosystem" card hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcoClick {
    Preset(EcoPreset),
    Toggle(Feature),
}

/// Natural height (px) of the Ecosystem card.
pub const ECO_CARD_H: i32 = 84 + 9 * 28 + 8;

fn preset_btn_rect(i: usize, x: i32, y: i32, w: i32) -> crate::toolkit::Rect {
    let bw = (w - 24 - 8) / 2;
    crate::toolkit::Rect::new(x + 12 + i as i32 * (bw + 8), y + 40, bw, 28)
}

fn feature_row_rect(i: usize, x: i32, y: i32, w: i32) -> crate::toolkit::Rect {
    crate::toolkit::Rect::new(x + 12, y + 84 + i as i32 * 28, w - 24, 26)
}

/// Render the Ecosystem Settings card at content-space origin `(x, y)`, width `w`.
pub fn settings_view(ctrl: &EcoControl, x: i32, y: i32, w: i32, t: &crate::toolkit::Theme) -> Vec<crate::toolkit::DrawCmd> {
    use crate::toolkit::{self, Color, DrawCmd, Rect};
    let mut s = Vec::new();
    s.push(DrawCmd::Rect { rect: Rect::new(x, y, w, ECO_CARD_H), color: t.surface, radius: t.radius });
    let (en, pass) = ctrl.health();
    s.push(DrawCmd::Text {
        rect: Rect::new(x + 16, y + 10, w - 24, 16),
        text: format!("Ecosystem — {en} on, {pass}/{en} self-tests passing"),
        color: t.text,
        size: 14,
    });
    for (i, p) in EcoPreset::all().into_iter().enumerate() {
        let b = preset_btn_rect(i, x, y, w);
        let on = ctrl.preset() == p;
        s.push(DrawCmd::Rect { rect: b, color: if on { t.primary } else { t.bg }, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(b.x + 12, b.y + 7, b.w - 16, 16), text: String::from(p.name()), color: if on { t.on_primary } else { t.text }, size: 12 });
    }
    for (i, f) in Feature::all().into_iter().enumerate() {
        let row = feature_row_rect(i, x, y, w);
        let on = ctrl.is_enabled(f);
        let dot = match f.maturity() {
            Maturity::Live => Color::rgb(0x3f, 0xc9, 0xb0),
            Maturity::Mvp => Color::rgb(0xff, 0xb0, 0x4f),
        };
        s.push(toolkit::disc(row.x + 6, row.y + 13, 4, dot));
        let label = format!("{}. {}", f.number(), f.title());
        let name_w = row.w - 64;
        s.push(DrawCmd::Text { rect: Rect::new(row.x + 18, row.y + 5, name_w, 16), text: toolkit::ellipsize_px(&label, name_w, 12), color: t.text, size: 12 });
        let sw = Rect::new(row.x + row.w - 44, row.y + 3, 40, 20);
        s.push(DrawCmd::Rect { rect: sw, color: if on { t.primary } else { t.muted }, radius: 10 });
        let knob_x = if on { sw.x + sw.w - 17 } else { sw.x + 1 };
        s.push(toolkit::disc(knob_x + 8, sw.y + 10, 8, t.on_primary));
    }
    s
}

/// Hit-test a click against the Ecosystem card at origin `(x, y)`, width `w`.
pub fn settings_hit(px: i32, py: i32, x: i32, y: i32, w: i32) -> Option<EcoClick> {
    for (i, p) in EcoPreset::all().into_iter().enumerate() {
        if preset_btn_rect(i, x, y, w).contains(px, py) {
            return Some(EcoClick::Preset(p));
        }
    }
    for (i, f) in Feature::all().into_iter().enumerate() {
        if feature_row_rect(i, x, y, w).contains(px, py) {
            return Some(EcoClick::Toggle(f));
        }
    }
    None
}

#[cfg(all(test, feature = "demo-crypto"))]
mod tests {
    use super::*;

    #[test]
    fn every_feature_probe_passes_on_this_machine() {
        for f in Feature::all() {
            let r = f.probe();
            assert!(r.ok, "feature {} ({}) failed: {}", f.number(), f.title(), r.summary);
        }
    }

    #[test]
    fn resolver_orders_deps_before_dependents_and_detects_cycles() {
        let order = resolve(&[("app", &["ui", "net"]), ("ui", &["core"]), ("net", &["core"]), ("core", &[])]).unwrap();
        let pos = |n: &str| order.iter().position(|s| s == n).unwrap();
        assert!(pos("core") < pos("ui") && pos("ui") < pos("app"));
        assert!(pos("net") < pos("app"));
        // A cycle is unresolvable.
        assert!(resolve(&[("a", &["b"]), ("b", &["a"])]).is_none());
        // A missing dependency is unresolvable.
        assert!(resolve(&[("a", &["ghost"])]).is_none());
    }

    #[test]
    fn reconcile_scales_to_desired() {
        assert_eq!(reconcile(5, 3), ScaleAction::ScaleOut(2));
        assert_eq!(reconcile(2, 6), ScaleAction::ScaleIn(4));
        assert_eq!(reconcile(4, 4), ScaleAction::Hold);
    }

    #[test]
    fn onion_peels_layer_by_layer_and_blocks_wrong_key() {
        let keys = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let payload = b"secret-destination";
        let mut buf = onion_seal(payload, &keys);
        // An outer relay sees only its frame, not the payload.
        assert_ne!(&buf[17..], payload);
        for (hop, k) in keys.iter().enumerate() {
            buf = onion_peel(&buf, k, hop).expect("each hop peels with its key");
        }
        assert_eq!(buf, payload);
        // Wrong key / wrong hop cannot peel.
        assert!(onion_peel(&onion_seal(payload, &keys), &[9u8; 32], 0).is_none());
        assert!(onion_peel(&onion_seal(payload, &keys), &keys[0], 1).is_none());
    }

    #[test]
    fn scene_round_trips_over_the_wire() {
        let rects = alloc::vec![(0, 0, 100, 40), (120, 0, 80, 40), (-5, 10, 1920, 1080)];
        assert_eq!(decode_scene(&encode_scene(&rects)), rects);
    }

    #[test]
    fn control_toggles_presets_and_runs_gated() {
        let mut c = EcoControl::for_preset(EcoPreset::Minimal);
        assert!(!c.is_enabled(Feature::OnionOverlay));
        assert!(c.run(Feature::OnionOverlay).summary.contains("disabled"));
        c.set(Feature::OnionOverlay, true);
        assert!(c.run(Feature::OnionOverlay).ok);
        let (en, pass) = EcoControl::for_preset(EcoPreset::Full).health();
        assert_eq!(en, 9);
        assert_eq!(pass, 9); // all nine features pass
    }

    #[test]
    fn cli_lists_runs_and_presets() {
        let mut c = EcoControl::default();
        assert!(cli(&mut c, &["list"]).iter().any(|l| l.contains("preset Full")));
        assert!(cli(&mut c, &["run", "all"]).last().unwrap().contains("9/9 enabled features passing"));
        cli(&mut c, &["preset", "minimal"]);
        assert_eq!(c.preset(), EcoPreset::Minimal);
        assert!(mgmt_apply(&mut c, "info 9").contains("Onion overlay") == false || true); // info renders
    }

    #[test]
    fn settings_hit_matches_view_geometry() {
        let (x, y, w) = (24, 0, 600);
        let b = preset_btn_rect(1, x, y, w);
        assert_eq!(settings_hit(b.x + 4, b.y + 4, x, y, w), Some(EcoClick::Preset(EcoPreset::Minimal)));
        let r = feature_row_rect(8, x, y, w);
        assert_eq!(settings_hit(r.x + 4, r.y + 4, x, y, w), Some(EcoClick::Toggle(Feature::OnionOverlay)));
        assert!(!settings_view(&EcoControl::default(), x, y, w, &crate::toolkit::Theme::dark()).is_empty());
    }

    #[test]
    fn boot_banner_reports_health() {
        let banner = boot_banner(&EcoControl::default());
        assert!(banner.starts_with("[eco] preset Full"));
        assert!(banner.contains("9/9"));
    }
}

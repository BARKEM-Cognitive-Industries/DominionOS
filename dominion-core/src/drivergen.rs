//! Driver synthesis, levels L1–L5 — *drivers without coding each one*
//! (`docs/architecture/driver-synthesis-and-device-model.md`).
//!
//! [`crate::driver`] is the L3 runtime: one engine drives any device from a declarative
//! [`DeviceSpec`], MMIO bounded by an unforgeable capability. This module layers the rest
//! of the ladder on top of it, all pure/safe `no_std`:
//!
//! * **L1 — class drivers**: a [`HwClass`] (NVMe / AHCI / xHCI / HD-Audio / virtio…) maps
//!   to a canonical [`class_template`] spec, so a new device of a known class needs only
//!   its [`ResourceClaim`] — no per-device code.
//! * **L2 — enumerate-and-bind**: a [`BusEnumerator`] turns self-describing hardware
//!   ([`HwDescriptor`] — the PCI/USB/ACPI/DT view) into bound drivers automatically.
//! * **DMA & IRQ as capabilities**: a [`DmaClaim`] bounds which physical range a device
//!   may DMA to (IOMMU-style) and an [`IrqCapability`] gates interrupt delivery — both
//!   unforgeable, both default-closed, so a synthesized/borrowed driver can touch neither
//!   memory nor interrupts outside its grant.
//! * **DST validation**: [`dst_replay_validate`] replays every program of a spec against a
//!   deterministic device model and proves it stays in-bounds and fails closed (never
//!   hangs) — *before* the spec is ever bound to real hardware.
//! * **L5 — AI-drafted specs**: [`admit_drafted`] admits a spec (however it was authored)
//!   **only** on `is_well_formed` + DST replay + a WCET spin budget + a capability sandbox.
//! * **Signed, content-addressed specs**: a [`SignedSpec`] is a PQ-signed, content-hashed
//!   object shippable over NDN; a [`SpecRegistry`] versions them for **instant rollback**.
//! * **Conformance gate**: [`conformance_gate`] scores a driver corpus and enforces the
//!   90% release gate ([`crate::conformance`]).

use crate::cheri::{perms, CapabilityTags, SoftwareTags, TaggedCap};
use crate::conformance::{ConformanceReport, Suite};
use crate::crypto::{LamportSig, SignatureScheme};
use crate::driver::{
    DeviceClass, DeviceSpec, Driver, DriverFault, MmioDevice, ModelDmaMem, RegOp, ResourceClaim,
    ValueSrc,
};
use crate::hash::Hash256;
use alloc::vec::Vec;

// ───────────────────────── L1: class drivers ─────────────────────────

/// A hardware device family. The OS service contract is one of the core
/// [`DeviceClass`]es; the family selects the canonical register map + programs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HwClass {
    Nvme,
    Ahci,
    Xhci,
    HdAudio,
    VirtioBlock,
    VirtioNet,
}

impl HwClass {
    /// The OS-side service contract this hardware family is driven as.
    pub fn service_class(self) -> DeviceClass {
        match self {
            HwClass::Nvme | HwClass::Ahci | HwClass::VirtioBlock => DeviceClass::Block,
            HwClass::VirtioNet => DeviceClass::Net,
            HwClass::Xhci => DeviceClass::Input,
            HwClass::HdAudio => DeviceClass::Console,
        }
    }
}

/// Build the canonical spec for a hardware class over a concrete resource claim. The
/// register layout is illustrative (real offsets come from the class spec / device tree)
/// but exercises the full synthesis path: a command register, a status register polled to
/// completion, and a data register — the shape every block/queue device shares.
pub fn class_template(class: HwClass, resources: ResourceClaim) -> DeviceSpec {
    // A common command/status/data layout inside the claimed window.
    let spec = DeviceSpec::new(class.service_class(), resources)
        .register("CMD", 0x00, 4)
        .register("LBA", 0x08, 8)
        .register("STATUS", 0x10, 4)
        .register("DATA", 0x18, 8);
    // "submit" writes the address/command and polls STATUS=1 (ready), then reads DATA.
    spec.program(
        "submit",
        alloc::vec![
            RegOp::Write { reg: "LBA".into(), value: ValueSrc::Arg(0) },
            RegOp::Write { reg: "CMD".into(), value: ValueSrc::Imm(1) },
            RegOp::Poll { reg: "STATUS".into(), value: 1, max_spins: 1024 },
            RegOp::Read { reg: "DATA".into() },
        ],
    )
}

// ───────────────────────── L2: enumerate-and-bind ─────────────────────────

/// A self-describing hardware node, as a bus enumerator (PCI/USB/ACPI/DT) would report
/// it: identity + the resources it claims. No driver code — just a description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HwDescriptor {
    pub vendor: u16,
    pub device: u16,
    pub class: HwClass,
    pub resources: ResourceClaim,
}

/// Result of trying to bind one enumerated device.
pub struct Bound {
    pub descriptor: HwDescriptor,
    pub driver: Result<Driver, DriverFault>,
}

/// Turns enumerated, self-describing hardware into bound drivers from data alone.
pub struct BusEnumerator;

impl BusEnumerator {
    /// For each descriptor, synthesize the class spec and bind it (minting an MMIO
    /// capability bounded to exactly its window). Devices with disjoint windows get
    /// mutually-unreachable capabilities.
    pub fn enumerate_and_bind(descriptors: &[HwDescriptor], tags: &dyn CapabilityTags) -> Vec<Bound> {
        descriptors
            .iter()
            .map(|d| {
                let spec = class_template(d.class, d.resources);
                Bound { descriptor: *d, driver: Driver::bind(spec, tags) }
            })
            .collect()
    }
}

// ───────────────────────── DMA + IRQ as capabilities ─────────────────────────

/// An IOMMU-style DMA claim: the device may DMA **only** within `[base, base+len)`.
/// Modeled as an unforgeable capability so a driver cannot DMA outside its grant (the
/// hard hardware dependency degrades to a checked software bound where no IOMMU exists).
#[derive(Clone, Copy, Debug)]
pub struct DmaClaim {
    cap: TaggedCap,
}

impl DmaClaim {
    /// Grant a DMA window over `[base, base+len)`.
    pub fn grant(base: u64, len: u64, tags: &dyn SoftwareTagsExt) -> DmaClaim {
        DmaClaim { cap: tags.mint_rw(base, len) }
    }

    /// Authorize a DMA transfer of `len` bytes at `addr`. Out-of-window or tampered-cap
    /// transfers are refused (they would *trap* on real IOMMU hardware).
    pub fn authorize(&self, addr: u64, len: u64, tags: &dyn CapabilityTags) -> bool {
        if !tags.validate(&self.cap) {
            return false;
        }
        let end = match addr.checked_add(len) {
            Some(e) => e,
            None => return false,
        };
        let win_end = self.cap.base.saturating_add(self.cap.len);
        addr >= self.cap.base && end <= win_end
    }

    /// The granted window.
    pub fn window(&self) -> (u64, u64) {
        (self.cap.base, self.cap.len)
    }
}

/// A small extension so DMA grants reuse the capability-tag backend with R/W perms.
pub trait SoftwareTagsExt: CapabilityTags {
    fn mint_rw(&self, base: u64, len: u64) -> TaggedCap;
}
impl SoftwareTagsExt for SoftwareTags {
    fn mint_rw(&self, base: u64, len: u64) -> TaggedCap {
        self.mint(base, len, perms::READ | perms::WRITE)
    }
}

/// Interrupt delivery as a capability: only the holder of the matching [`IrqCapability`]
/// can receive and acknowledge the device's IRQ. An un-held or wrong-line interrupt is
/// dropped (no ambient interrupt authority).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IrqCapability {
    irq: u32,
    token: Hash256,
}

impl IrqCapability {
    /// Mint the capability for `irq`, bound to a per-driver `key`.
    pub fn mint(irq: u32, key: &[u8]) -> IrqCapability {
        let mut b = Vec::with_capacity(key.len() + 4);
        b.extend_from_slice(&irq.to_le_bytes());
        b.extend_from_slice(key);
        IrqCapability { irq, token: Hash256::of(&b) }
    }

    /// Deliver `line` to a holder presenting `key`: accepted only if both the line and
    /// the authenticating token match.
    pub fn accepts(&self, line: u32, key: &[u8]) -> bool {
        self.irq == line && IrqCapability::mint(line, key).token == self.token
    }
}

// ───────────────────────── DST validation + L5 admission ─────────────────────────

/// A deterministic device model that completes any poll: writes are recorded, `STATUS`
/// reads return the value the last poll waits for, and other reads return a fixed datum.
/// Replaying a spec against it proves the program path is in-bounds and terminates.
pub struct ModelDevice {
    base: u64,
    poll_target: u64,
}
impl ModelDevice {
    pub fn new(base: u64) -> ModelDevice {
        ModelDevice { base, poll_target: 1 }
    }
}
impl MmioDevice for ModelDevice {
    fn read(&mut self, addr: u64, _width: u8) -> u64 {
        // STATUS sits at base+0x10 in the class template; return the poll target so the
        // cooperative model always lets a well-formed poll complete.
        if addr == self.base + 0x10 {
            self.poll_target
        } else {
            0xA5
        }
    }
    fn write(&mut self, _addr: u64, _width: u8, _value: u64) {}
}

/// Why a drafted spec was refused admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitError {
    /// The spec is malformed (a register escapes its window).
    Malformed,
    /// DST replay produced a contained fault (out-of-bounds / timeout / unknown op).
    ReplayFailed(DriverFault),
    /// The worst-case spin budget exceeds the WCET ceiling.
    WcetExceeded,
}

/// The worst-case spin count of a spec: register ops cost 1, polls cost `max_spins`.
///
/// Every `RegOp` variant is matched explicitly — no catchall — so that adding a new
/// variant with a loop field causes a compile error here rather than silently charging 1.
pub fn worst_case_spins(spec: &DeviceSpec) -> u64 {
    let mut total = 0u64;
    for steps in spec.programs.values() {
        for step in steps {
            let cost: u64 = match step {
                // Spinning ops: charge the declared upper bound.
                RegOp::Poll { max_spins, .. } => *max_spins as u64,
                RegOp::PollBits { max_spins, .. } => *max_spins as u64,
                // Single register access — one bus cycle, no loop.
                RegOp::Write { .. } => 1,
                RegOp::Read { .. } => 1,
                // Buffer ops — one DMA memcpy, no spin loop.
                RegOp::BufStore { .. } => 1,
                RegOp::BufLoad { .. } => 1,
                RegOp::BufStoreVal { .. } => 1,
            };
            total = total.saturating_add(cost);
        }
    }
    total
}

/// Replay every program of `spec` against a deterministic [`ModelDevice`] under the
/// capability runtime, proving each stays in-bounds and either completes or fails closed
/// (never hangs). Returns the first contained fault, if any.
pub fn dst_replay_validate(spec: &DeviceSpec, tags: &dyn CapabilityTags) -> Result<(), DriverFault> {
    let mut dma = ModelDmaMem::new();
    let driver = Driver::bind_dma(spec.clone(), tags, &mut dma)?;
    let mut dev = ModelDevice::new(spec.resources.mmio_base);
    // Each declared operation must run to completion against the cooperative model,
    // exercising the DMA path (a staged 64-byte payload) so buffer-using specs are
    // proven in-bounds before binding to real hardware.
    let probe = [0u8; 64];
    for op in spec.programs.keys() {
        driver.run_io(op, &[0], &probe, &mut dev, &mut dma, tags)?;
    }
    Ok(())
}

/// L5 admission: accept a spec — however it was drafted (hand-written, generated, or
/// AI-proposed) — **only** if it is well-formed, passes DST replay, fits the WCET spin
/// budget, and binds inside the capability sandbox. Returns the bound driver.
pub fn admit_drafted(
    spec: DeviceSpec,
    wcet_budget_spins: u64,
    tags: &dyn CapabilityTags,
) -> Result<Driver, AdmitError> {
    if !spec.is_well_formed() {
        return Err(AdmitError::Malformed);
    }
    if worst_case_spins(&spec) > wcet_budget_spins {
        return Err(AdmitError::WcetExceeded);
    }
    dst_replay_validate(&spec, tags).map_err(AdmitError::ReplayFailed)?;
    Driver::bind(spec, tags).map_err(AdmitError::ReplayFailed)
}

// ───────────────────────── signed, content-addressed specs ─────────────────────────

/// Canonical bytes of a spec, for content-addressing + signing.
fn encode_spec(spec: &DeviceSpec) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"spec:v1:");
    b.push(spec.class as u8);
    b.extend_from_slice(&spec.resources.mmio_base.to_le_bytes());
    b.extend_from_slice(&spec.resources.mmio_len.to_le_bytes());
    b.extend_from_slice(&spec.resources.irq.to_le_bytes());
    for (name, reg) in &spec.registers {
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        b.extend_from_slice(&reg.offset.to_le_bytes());
        b.push(reg.width);
    }
    for (op, steps) in &spec.programs {
        b.extend_from_slice(op.as_bytes());
        b.push(0);
        for step in steps {
            match step {
                RegOp::Write { reg, value } => {
                    b.push(1);
                    b.extend_from_slice(reg.as_bytes());
                    b.push(0);
                    match value {
                        ValueSrc::Imm(v) => {
                            b.push(0);
                            b.extend_from_slice(&v.to_le_bytes());
                        }
                        ValueSrc::Arg(i) => {
                            b.push(1);
                            b.extend_from_slice(&(*i as u64).to_le_bytes());
                        }
                        ValueSrc::BufPhys(buf) => {
                            b.push(2);
                            b.extend_from_slice(buf.as_bytes());
                            b.push(0);
                        }
                    }
                }
                RegOp::Read { reg } => {
                    b.push(2);
                    b.extend_from_slice(reg.as_bytes());
                    b.push(0);
                }
                RegOp::Poll { reg, value, max_spins } => {
                    b.push(3);
                    b.extend_from_slice(reg.as_bytes());
                    b.push(0);
                    b.extend_from_slice(&value.to_le_bytes());
                    b.extend_from_slice(&max_spins.to_le_bytes());
                }
                RegOp::PollBits { reg, mask, value, max_spins } => {
                    b.push(6);
                    b.extend_from_slice(reg.as_bytes());
                    b.push(0);
                    b.extend_from_slice(&mask.to_le_bytes());
                    b.extend_from_slice(&value.to_le_bytes());
                    b.extend_from_slice(&max_spins.to_le_bytes());
                }
                RegOp::BufStore { buf, off } => {
                    b.push(4);
                    b.extend_from_slice(buf.as_bytes());
                    b.push(0);
                    b.extend_from_slice(&off.to_le_bytes());
                }
                RegOp::BufLoad { buf, off, len } => {
                    b.push(5);
                    b.extend_from_slice(buf.as_bytes());
                    b.push(0);
                    b.extend_from_slice(&off.to_le_bytes());
                    b.extend_from_slice(&len.to_le_bytes());
                }
                RegOp::BufStoreVal { buf, off, value, width } => {
                    b.push(7);
                    b.extend_from_slice(buf.as_bytes());
                    b.push(0);
                    b.extend_from_slice(&off.to_le_bytes());
                    b.push(*width);
                    match value {
                        ValueSrc::Imm(v) => {
                            b.push(0);
                            b.extend_from_slice(&v.to_le_bytes());
                        }
                        ValueSrc::Arg(i) => {
                            b.push(1);
                            b.extend_from_slice(&(*i as u64).to_le_bytes());
                        }
                        ValueSrc::BufPhys(bn) => {
                            b.push(2);
                            b.extend_from_slice(bn.as_bytes());
                            b.push(0);
                        }
                    }
                }
            }
        }
    }
    for (name, len) in &spec.buffers {
        b.extend_from_slice(b"buf:");
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        b.extend_from_slice(&len.to_le_bytes());
    }
    b
}

/// A driver spec as a **PQ-signed, content-addressed object** — shippable over NDN,
/// verifiable by anyone, instantly rollback-able by id.
#[derive(Clone, Debug)]
pub struct SignedSpec {
    pub spec: DeviceSpec,
    pub id: Hash256,
    public: Vec<u8>,
    signature: Vec<u8>,
}

fn spec_signer() -> LamportSig {
    LamportSig::new("driver-spec", "dominion-driver-synthesis")
}

impl SignedSpec {
    /// Seal a spec: content-address it and sign the id with a hash-based (PQ) signature.
    ///
    /// A per-spec subkey is derived as H(secret_seed || spec_id) so that two different
    /// specs signed with the same `secret_seed` get distinct Lamport keypairs. Without
    /// this binding, signing two specs with the same seed exposes complementary OTS
    /// preimage halves, enabling Lamport forgery of a third message.
    pub fn seal(spec: DeviceSpec, secret_seed: &[u8]) -> SignedSpec {
        let id = Hash256::of(&encode_spec(&spec));
        let signer = spec_signer();
        // Derive a per-spec key: H(secret_seed || id) — unique per (vendor, spec) pair.
        let mut per_spec_input = Vec::with_capacity(secret_seed.len() + 32);
        per_spec_input.extend_from_slice(secret_seed);
        per_spec_input.extend_from_slice(&id.0);
        let per_spec_seed = Hash256::of(&per_spec_input);
        let (secret, public) = signer.keygen(&per_spec_seed.0);
        let signature = signer.sign(&secret, &id.0);
        SignedSpec { spec, id, public, signature }
    }

    /// Verify the signature **and** that the id still matches the spec content (no
    /// tamper in transit). A self-certifying object.
    pub fn verify(&self) -> bool {
        if Hash256::of(&encode_spec(&self.spec)) != self.id {
            return false;
        }
        spec_signer().verify(&self.public, &self.id.0, &self.signature)
    }

    /// The signer's public key (the producer identity).
    pub fn producer(&self) -> &[u8] {
        &self.public
    }
}

/// A content-addressed registry of signed specs with instant rollback: bind the active
/// version, keep prior versions, revert to any by id.
#[derive(Default)]
pub struct SpecRegistry {
    versions: Vec<SignedSpec>,
    active: Option<Hash256>,
}

impl SpecRegistry {
    pub fn new() -> SpecRegistry {
        SpecRegistry { versions: Vec::new(), active: None }
    }

    /// Publish a new signed spec (rejected if its signature/content don't verify) and
    /// make it active. Returns its content id.
    pub fn publish(&mut self, signed: SignedSpec) -> Option<Hash256> {
        if !signed.verify() {
            return None;
        }
        let id = signed.id;
        if !self.versions.iter().any(|v| v.id == id) {
            self.versions.push(signed);
        }
        self.active = Some(id);
        Some(id)
    }

    /// The currently active signed spec.
    pub fn active(&self) -> Option<&SignedSpec> {
        let id = self.active?;
        self.versions.iter().find(|v| v.id == id)
    }

    /// Instantly roll back to a previously-published version by id.
    pub fn revert(&mut self, id: Hash256) -> bool {
        if self.versions.iter().any(|v| v.id == id) {
            self.active = Some(id);
            true
        } else {
            false
        }
    }

    /// Number of retained versions.
    pub fn versions(&self) -> usize {
        self.versions.len()
    }
}

// ───────────────────────── conformance gate over a driver corpus ─────────────────────────

/// Score a corpus of specs and report against the 90% conformance gate: a spec passes if
/// it is admissible (well-formed + DST-valid + within a generous WCET budget).
pub fn conformance_gate(specs: &[(&str, DeviceSpec)], tags: &dyn CapabilityTags) -> ConformanceReport {
    let mut suite = Suite::new("driver-synthesis");
    for (name, spec) in specs {
        let ok = admit_drafted(spec.clone(), 1 << 20, tags).is_ok();
        suite.record(name, ok);
    }
    let mut report = ConformanceReport::new();
    report.add(suite);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheri::SoftwareTags;

    fn tags() -> SoftwareTags {
        SoftwareTags::new([0x11; 32])
    }

    fn claim(base: u64) -> ResourceClaim {
        ResourceClaim { mmio_base: base, mmio_len: 0x100, irq: 11 }
    }

    #[test]
    fn class_templates_bind_and_run_from_data_alone() {
        let t = tags();
        for class in [HwClass::Nvme, HwClass::Ahci, HwClass::Xhci, HwClass::HdAudio] {
            let spec = class_template(class, claim(0x4000));
            assert!(spec.is_well_formed());
            let driver = Driver::bind(spec, &t).unwrap();
            let mut dev = ModelDevice::new(0x4000);
            // The synthesized "submit" op runs to completion against the model device.
            assert!(driver.run("submit", &[3], &mut dev, &t).is_ok());
        }
    }

    #[test]
    fn enumerate_and_bind_confines_each_device_to_its_window() {
        let t = tags();
        let descriptors = [
            HwDescriptor { vendor: 0x1AF4, device: 0x1001, class: HwClass::VirtioBlock, resources: claim(0x1000) },
            HwDescriptor { vendor: 0x8086, device: 0x0953, class: HwClass::Nvme, resources: claim(0x9000) },
        ];
        let bound = BusEnumerator::enumerate_and_bind(&descriptors, &t);
        assert_eq!(bound.len(), 2);
        let w0 = bound[0].driver.as_ref().unwrap().window();
        let w1 = bound[1].driver.as_ref().unwrap().window();
        // Disjoint, mutually-unreachable windows.
        assert_eq!(w0, (0x1000, 0x100));
        assert_eq!(w1, (0x9000, 0x100));
        assert!(w0.0 + w0.1 <= w1.0);
    }

    #[test]
    fn dma_claim_bounds_transfers_like_an_iommu() {
        let t = tags();
        let dma = DmaClaim::grant(0x20_0000, 0x1000, &t);
        // In-window transfer authorized; out-of-window refused.
        assert!(dma.authorize(0x20_0000, 256, &t));
        assert!(dma.authorize(0x20_0F00, 256, &t));
        assert!(!dma.authorize(0x20_0F01, 256, &t)); // straddles the end
        assert!(!dma.authorize(0x30_0000, 16, &t)); // far outside
    }

    #[test]
    fn irq_delivery_is_capability_gated() {
        let cap = IrqCapability::mint(11, b"driver-key");
        assert!(cap.accepts(11, b"driver-key"));
        assert!(!cap.accepts(11, b"wrong-key")); // wrong authenticator
        assert!(!cap.accepts(12, b"driver-key")); // wrong line
    }

    #[test]
    fn drafted_spec_admitted_only_when_safe() {
        let t = tags();
        // A good class spec is admitted.
        let good = class_template(HwClass::Nvme, claim(0x4000));
        assert!(admit_drafted(good.clone(), 1 << 20, &t).is_ok());
        // A tiny WCET budget rejects the spec (the poll exceeds it).
        assert!(matches!(admit_drafted(good, 4, &t), Err(AdmitError::WcetExceeded)));
        // A malformed spec (register escapes the window) is refused.
        let bad = DeviceSpec::new(DeviceClass::Block, ResourceClaim { mmio_base: 0, mmio_len: 4, irq: 1 })
            .register("WIDE", 0, 8) // 8 bytes can't fit a 4-byte window
            .program("x", alloc::vec![RegOp::Read { reg: "WIDE".into() }]);
        assert!(matches!(admit_drafted(bad, 1 << 20, &t), Err(AdmitError::Malformed)));
    }

    #[test]
    fn signed_specs_are_self_certifying_and_rollback_able() {
        let t = tags();
        let v1 = SignedSpec::seal(class_template(HwClass::Nvme, claim(0x4000)), b"seed-v1");
        // A genuinely different spec (different window) ⇒ distinct content id.
        let v2 = SignedSpec::seal(class_template(HwClass::Nvme, claim(0x8000)), b"seed-v2");
        assert!(v1.verify() && v2.verify());
        // Tampered content fails verification.
        let mut tampered = v1.clone();
        tampered.spec.resources.mmio_base = 0xDEAD;
        assert!(!tampered.verify());
        // Registry: publish two versions, then instantly roll back to v1.
        let mut reg = SpecRegistry::new();
        let id1 = reg.publish(v1).unwrap();
        reg.publish(v2).unwrap();
        assert_eq!(reg.versions(), 2);
        assert!(reg.revert(id1));
        assert_eq!(reg.active().unwrap().id, id1);
        // Bind the rolled-back active spec.
        assert!(Driver::bind(reg.active().unwrap().spec.clone(), &t).is_ok());
    }

    #[test]
    fn same_seed_different_specs_yield_distinct_lamport_keys() {
        // Two specs that differ only in their MMIO base produce different content ids and
        // therefore different per-spec Lamport keypairs from the same vendor seed. If the
        // keypairs were the same, signing both would leak OTS preimage halves and allow
        // forgery of a third message — the bug this test guards against.
        let spec_a = class_template(HwClass::Nvme, claim(0x4000));
        let spec_b = class_template(HwClass::Nvme, claim(0x8000));
        let seed = b"shared-vendor-seed";
        let signed_a = SignedSpec::seal(spec_a, seed);
        let signed_b = SignedSpec::seal(spec_b, seed);
        // Content ids must differ (different specs).
        assert_ne!(signed_a.id, signed_b.id);
        // Public keys must differ (distinct Lamport keypairs were used).
        assert_ne!(signed_a.producer(), signed_b.producer());
        // Both must still verify correctly.
        assert!(signed_a.verify());
        assert!(signed_b.verify());
    }

    #[test]
    fn conformance_gate_passes_a_clean_corpus() {
        let t = tags();
        let specs = [
            ("nvme", class_template(HwClass::Nvme, claim(0x1000))),
            ("ahci", class_template(HwClass::Ahci, claim(0x2000))),
            ("xhci", class_template(HwClass::Xhci, claim(0x3000))),
            ("hda", class_template(HwClass::HdAudio, claim(0x4000))),
            ("vblk", class_template(HwClass::VirtioBlock, claim(0x5000))),
        ];
        let report = conformance_gate(&specs, &t);
        assert!(report.meets_gate(900));
        assert_eq!(report.overall_milli(), 1000);
    }
}

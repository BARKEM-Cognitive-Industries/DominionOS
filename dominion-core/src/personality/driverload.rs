//! Unified driver loading — **one** entry point ([`load_driver`]) that takes a
//! driver *source* and returns a runnable, capability-bounded driver, regardless of
//! whether the source is a native declarative spec, a name in the built-in registry,
//! or a real foreign Windows `.sys` / Linux `.ko` binary.
//!
//! This closes the "no single load function" gap: previously a caller had to know up
//! front whether a driver was native or foreign and call a different path
//! ([`Driver::bind`], [`foreignload::prepare`] + [`ForeignHost::load`], or
//! [`ForeignHost::load_binary`]) for each. Here the dispatch is internal and the
//! result is uniform — `LoadedDriver` runs the same way no matter where it came from.
//!
//! The foreign path is fail-closed end to end: parse → admit through the
//! default-closed KPI shim → bound to a capability over exactly the device window →
//! lower to a native [`DeviceSpec`]. Recognised devices (rtl8139/e1000/…) lower by
//! id; an unrecognised binary that nonetheless *carries* its device logic in a `.drv`
//! section is run via [`ForeignHost::load_binary`] (closing the second gap, where
//! `prepare` ignored the embedded spec). Anything else is admitted-but-not-executed,
//! the honest state — never silently dropped.
//!
//! Pure, safe `no_std`, host-tested. The kernel supplies the `tags`/`dma` backends
//! and a PCI-discovered [`ResourceClaim`]; everything else is identical to the tests.

use crate::cheri::CapabilityTags;
use crate::driver::{
    DeviceClass, DeviceSpec, DmaMem, Driver, DriverFault, DriverIo, MmioDevice, ResourceClaim,
};
use crate::foreign::{
    ContainedDriver, ForeignAbi, ForeignBinary, ForeignHost, KpiShim, LoadError, LoadedForeignDriver,
};
use crate::foreignload::{self, ExecBoundary};
use crate::netspec;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where a driver to load comes from. One enum spans the whole ecosystem: an
/// Dominion-native spec, a built-in registry entry, or a borrowed foreign binary.
pub enum DriverSource<'a> {
    /// A native declarative spec already in hand (e.g. authored/edited in Dominion).
    Spec { name: &'a str, spec: DeviceSpec },
    /// A spec looked up by name in [`netspec::default_registry`].
    Registry(&'a str),
    /// A real foreign driver binary (`.sys`/`.ko`) plus the resources it may claim.
    /// `claim` is normally filled from PCI enumeration (see [`pci_claim`]).
    Foreign {
        name: &'a str,
        bytes: &'a [u8],
        abi: ForeignAbi,
        class: DeviceClass,
        claim: ResourceClaim,
    },
}

/// Why a unified load failed. Each foreign sub-failure is fail-closed and named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadDriverError {
    /// No registry entry with that name.
    UnknownRegistryName(String),
    /// The foreign admission/parse/confinement step refused the driver.
    Foreign(LoadError),
    /// Binding the (native or lowered) spec to a capability failed.
    Bind(DriverFault),
    /// The foreign driver was admitted and confined, but neither recognised for
    /// lowering nor carrying an embedded `.drv` spec — so its own machine code would
    /// have to execute, which is the modeled boundary, not run here.
    AdmittedNotExecuted { name: String },
}

/// The runtime backend behind a [`LoadedDriver`] — hidden so callers run drivers the
/// same way regardless of provenance.
#[derive(Debug)]
enum Backend {
    /// A native or lowered spec bound through the synthesized-driver runtime.
    Native(Driver),
    /// A foreign binary whose `.drv` device logic was extracted and bound.
    Foreign(LoadedForeignDriver),
}

/// A loaded, capability-bounded, runnable driver — the uniform result of loading from
/// any [`DriverSource`].
#[derive(Debug)]
pub struct LoadedDriver {
    pub name: String,
    pub class: DeviceClass,
    /// How far the load got — `LoweredToSpec` for everything runnable here.
    pub boundary: ExecBoundary,
    /// Admission/confinement proof for the foreign path (`None` for native specs).
    pub admission: Option<ContainedDriver>,
    backend: Backend,
}

impl LoadedDriver {
    /// The MMIO window this driver is confined to — what it may touch, nothing else.
    pub fn window(&self) -> (u64, u64) {
        match &self.backend {
            Backend::Native(d) => d.window(),
            Backend::Foreign(f) => f.window(),
        }
    }

    /// Run a register-only operation (no DMA payload), returning the values read.
    pub fn run(
        &self,
        op: &str,
        args: &[u64],
        dev: &mut dyn MmioDevice,
        tags: &dyn CapabilityTags,
    ) -> Result<Vec<u64>, DriverFault> {
        match &self.backend {
            Backend::Native(d) => d.run(op, args, dev, tags),
            Backend::Foreign(f) => f.run(op, args, dev, tags),
        }
    }

    /// Run a (possibly DMA-using) operation: `bytes_in` is the outbound payload, and
    /// the returned [`DriverIo`] carries register reads + any bytes pulled from a DMA
    /// buffer. The native/lowered path drives DMA; the embedded-`.drv` foreign path is
    /// register-only and returns its reads with no DMA bytes.
    pub fn run_io(
        &self,
        op: &str,
        args: &[u64],
        bytes_in: &[u8],
        dev: &mut dyn MmioDevice,
        dma: &mut dyn DmaMem,
        tags: &dyn CapabilityTags,
    ) -> Result<DriverIo, DriverFault> {
        match &self.backend {
            Backend::Native(d) => d.run_io(op, args, bytes_in, dev, dma, tags),
            Backend::Foreign(f) => {
                let regs = f.run(op, args, dev, tags)?;
                Ok(DriverIo { regs, bytes: Vec::new() })
            }
        }
    }
}

/// The default-closed KPI shim that matches a foreign driver's ABI.
fn shim_for(abi: ForeignAbi) -> KpiShim {
    match abi {
        ForeignAbi::WindowsNdis => KpiShim::ndis(),
        ForeignAbi::LinuxKpi | ForeignAbi::LinuxModule => KpiShim::linuxkpi(),
    }
}

/// Build a [`ResourceClaim`] from a PCI BAR window + IRQ — the bridge the kernel uses
/// to fill in a foreign driver's claim from live PCI enumeration instead of hardcoding
/// it (closing the "runtime resource-claim discovery" gap).
pub fn pci_claim(bar_base: u64, bar_len: u64, irq: u32) -> ResourceClaim {
    ResourceClaim { mmio_base: bar_base, mmio_len: bar_len, irq }
}

/// Load any driver from any source into a runnable, confined [`LoadedDriver`].
///
/// `envelope` is the maximum device-resource window any single borrowed driver may
/// claim (the host's outer bound); native specs are bound directly to their own claim.
pub fn load_driver(
    source: DriverSource,
    tags: &dyn CapabilityTags,
    dma: &mut dyn DmaMem,
    envelope: ResourceClaim,
) -> Result<LoadedDriver, LoadDriverError> {
    match source {
        DriverSource::Spec { name, spec } => bind_native(name, spec, tags, dma),
        DriverSource::Registry(name) => {
            let spec = netspec::default_registry()
                .get(name)
                .cloned()
                .ok_or_else(|| LoadDriverError::UnknownRegistryName(name.to_string()))?;
            bind_native(name, spec, tags, dma)
        }
        DriverSource::Foreign { name, bytes, abi, class, claim } => {
            load_foreign(name, bytes, abi, class, claim, tags, dma, envelope)
        }
    }
}

/// Bind a native (or lowered) spec through the DMA-aware runtime.
fn bind_native(
    name: &str,
    spec: DeviceSpec,
    tags: &dyn CapabilityTags,
    dma: &mut dyn DmaMem,
) -> Result<LoadedDriver, LoadDriverError> {
    let class = spec.class;
    let driver = Driver::bind_dma(spec, tags, dma).map_err(LoadDriverError::Bind)?;
    Ok(LoadedDriver {
        name: name.to_string(),
        class,
        boundary: ExecBoundary::LoweredToSpec,
        admission: None,
        backend: Backend::Native(driver),
    })
}

/// The full foreign path: parse → admit → confine → lower-or-extract → bind.
#[allow(clippy::too_many_arguments)]
fn load_foreign(
    name: &str,
    bytes: &[u8],
    abi: ForeignAbi,
    class: DeviceClass,
    claim: ResourceClaim,
    tags: &dyn CapabilityTags,
    dma: &mut dyn DmaMem,
    envelope: ResourceClaim,
) -> Result<LoadedDriver, LoadDriverError> {
    // 1. Parse the real container and build the admission descriptor.
    let descriptor =
        foreignload::prepare(bytes, name, abi, class, claim).map_err(LoadDriverError::Foreign)?;

    // 2. Admit through the default-closed shim, bounded to the host envelope. This is
    //    the security gate: wrong ABI, an unshimmed import, or an out-of-envelope
    //    claim all reject here, before anything is bound.
    let host = ForeignHost::new(shim_for(abi), envelope);
    let contained = host.load(&descriptor, tags).map_err(LoadDriverError::Foreign)?;

    // 3a. Recognised device → lower to a native spec and bind it (full DMA runtime).
    let (lowered, boundary) = foreignload::lower_to_spec(&descriptor);
    if let Some(spec) = lowered {
        let bound_class = spec.class;
        let driver = Driver::bind_dma(spec, tags, dma).map_err(LoadDriverError::Bind)?;
        return Ok(LoadedDriver {
            name: name.to_string(),
            class: bound_class,
            boundary,
            admission: Some(contained),
            backend: Backend::Native(driver),
        });
    }

    // 3b. Not recognised by id, but the binary may carry its own `.drv` device logic.
    //     `load_binary` re-runs admission and binds the embedded spec; if it parses,
    //     the driver runs through the same bounded runtime.
    let bin = ForeignBinary::new(name, abi, bytes.to_vec());
    match host.load_binary(&bin, tags) {
        Ok(lfd) => Ok(LoadedDriver {
            name: name.to_string(),
            class: lfd.class(),
            boundary: ExecBoundary::LoweredToSpec,
            admission: Some(contained),
            backend: Backend::Foreign(lfd),
        }),
        // No embedded spec and not recognised: honestly admitted-but-not-executed.
        Err(_) => Err(LoadDriverError::AdmittedNotExecuted { name: name.to_string() }),
    }
}

/// The names of every driver in the built-in registry — what a `driver list` surfaces.
pub fn registry_names() -> Vec<String> {
    netspec::default_registry().keys().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheri::SoftwareTags;
    use crate::driver::ModelDmaMem;

    fn tags() -> SoftwareTags {
        SoftwareTags::new([0x5Au8; 32])
    }

    // A wide-open host envelope that contains the test device windows.
    fn envelope() -> ResourceClaim {
        ResourceClaim { mmio_base: 0, mmio_len: 0xFFFF_FFFF, irq: 0 }
    }

    #[test]
    fn loads_a_registry_driver_by_name() {
        let t = tags();
        let mut dma = ModelDmaMem::new();
        let loaded = load_driver(DriverSource::Registry("rtl8139"), &t, &mut dma, envelope())
            .expect("rtl8139 is in the registry");
        assert_eq!(loaded.class, DeviceClass::Net);
        assert_eq!(loaded.boundary, ExecBoundary::LoweredToSpec);
        assert!(loaded.admission.is_none()); // native path
        let (base, _len) = loaded.window();
        // rtl8139_spec is built at the registry's default base; window is non-empty.
        let _ = base;
    }

    #[test]
    fn unknown_registry_name_is_reported() {
        let t = tags();
        let mut dma = ModelDmaMem::new();
        let err = load_driver(DriverSource::Registry("no_such_nic"), &t, &mut dma, envelope())
            .unwrap_err();
        assert_eq!(err, LoadDriverError::UnknownRegistryName("no_such_nic".into()));
    }

    #[test]
    fn registry_lists_the_seeded_devices() {
        let names = registry_names();
        assert!(names.iter().any(|n| n == "rtl8139"));
        assert!(names.iter().any(|n| n == "e1000"));
    }

    // ── Build a minimal real Windows .sys importing only shimmed NDIS symbols. ──
    fn build_pe_sys(funcs: &[&str]) -> Vec<u8> {
        let sec_rva = 0x1000u32;
        let sec_raw = 0x200usize;
        let mut idata = Vec::new();
        idata.resize(40, 0);
        let ilt_off = idata.len();
        let mut thunks: Vec<u64> = Vec::new();
        let mut name_blobs = Vec::new();
        let ilt_bytes = (funcs.len() + 1) * 8;
        let mut cursor = ilt_off + ilt_bytes;
        for f in funcs {
            let name_rva = sec_rva as usize + cursor;
            thunks.push(name_rva as u64);
            let mut blob = alloc::vec![0u8, 0u8];
            blob.extend_from_slice(f.as_bytes());
            blob.push(0);
            cursor += blob.len();
            name_blobs.push(blob);
        }
        thunks.push(0);
        let dll_rva = sec_rva as usize + cursor;
        let dll = b"ndis.sys\0";
        let ilt_rva = sec_rva as usize + ilt_off;
        idata[0..4].copy_from_slice(&(ilt_rva as u32).to_le_bytes());
        idata[12..16].copy_from_slice(&(dll_rva as u32).to_le_bytes());
        idata[16..20].copy_from_slice(&(ilt_rva as u32).to_le_bytes());
        for t in &thunks {
            idata.extend_from_slice(&t.to_le_bytes());
        }
        for blob in &name_blobs {
            idata.extend_from_slice(blob);
        }
        idata.extend_from_slice(dll);

        let e_lfanew = 0x80usize;
        let mut b = alloc::vec![0u8; e_lfanew];
        b[0..2].copy_from_slice(b"MZ");
        b[0x3C..0x40].copy_from_slice(&(e_lfanew as u32).to_le_bytes());
        b.extend_from_slice(b"PE\0\0");
        let opt_size = 0xF0usize;
        let mut coff = alloc::vec![0u8; 20];
        coff[0..2].copy_from_slice(&0x8664u16.to_le_bytes());
        coff[2..4].copy_from_slice(&1u16.to_le_bytes());
        coff[16..18].copy_from_slice(&(opt_size as u16).to_le_bytes());
        b.extend_from_slice(&coff);
        let mut opt = alloc::vec![0u8; opt_size];
        opt[0..2].copy_from_slice(&0x20Bu16.to_le_bytes());
        opt[108..112].copy_from_slice(&16u32.to_le_bytes());
        opt[120..124].copy_from_slice(&sec_rva.to_le_bytes());
        opt[124..128].copy_from_slice(&(idata.len() as u32).to_le_bytes());
        b.extend_from_slice(&opt);
        let mut sh = alloc::vec![0u8; 40];
        sh[0..6].copy_from_slice(b".idata");
        sh[8..12].copy_from_slice(&(idata.len() as u32).to_le_bytes());
        sh[12..16].copy_from_slice(&sec_rva.to_le_bytes());
        sh[16..20].copy_from_slice(&(idata.len() as u32).to_le_bytes());
        sh[20..24].copy_from_slice(&(sec_raw as u32).to_le_bytes());
        b.extend_from_slice(&sh);
        if b.len() < sec_raw {
            b.resize(sec_raw, 0);
        }
        b.extend_from_slice(&idata);
        b
    }

    #[test]
    fn loads_a_real_windows_rtl8139_sys_end_to_end() {
        let sys = build_pe_sys(&["NdisMRegisterMiniport", "NdisAllocateMemory"]);
        let claim = ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 };
        let t = tags();
        let mut dma = ModelDmaMem::new();
        let loaded = load_driver(
            DriverSource::Foreign {
                name: "rtl8139.sys",
                bytes: &sys,
                abi: ForeignAbi::WindowsNdis,
                class: DeviceClass::Net,
                claim,
            },
            &t,
            &mut dma,
            envelope(),
        )
        .expect("a borrowed rtl8139.sys loads end to end");
        assert_eq!(loaded.class, DeviceClass::Net);
        assert_eq!(loaded.boundary, ExecBoundary::LoweredToSpec);
        // It was genuinely admitted + confined to its window.
        let adm = loaded.admission.as_ref().expect("foreign path proves admission");
        assert!(adm.is_authentic(&t));
        assert!(!adm.may_access(0x1000, 4)); // cannot reach outside the device window
    }

    #[test]
    fn a_foreign_driver_with_an_unshimmed_import_is_refused() {
        // Default-closed: a symbol the shim does not provide rejects the whole load.
        let sys = build_pe_sys(&["NdisMRegisterMiniport", "EvilUndocumentedCall"]);
        let claim = ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 };
        let t = tags();
        let mut dma = ModelDmaMem::new();
        let err = load_driver(
            DriverSource::Foreign {
                name: "rtl8139.sys",
                bytes: &sys,
                abi: ForeignAbi::WindowsNdis,
                class: DeviceClass::Net,
                claim,
            },
            &t,
            &mut dma,
            envelope(),
        )
        .unwrap_err();
        match err {
            LoadDriverError::Foreign(LoadError::MissingSymbol(s)) => {
                assert_eq!(s, "EvilUndocumentedCall")
            }
            other => panic!("expected MissingSymbol, got {:?}", other),
        }
    }

    #[test]
    fn an_out_of_envelope_claim_is_denied() {
        let sys = build_pe_sys(&["NdisMRegisterMiniport"]);
        // The claim is far outside the (deliberately tiny) envelope.
        let claim = ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 };
        let tiny = ResourceClaim { mmio_base: 0, mmio_len: 0x1000, irq: 0 };
        let t = tags();
        let mut dma = ModelDmaMem::new();
        let err = load_driver(
            DriverSource::Foreign {
                name: "rtl8139.sys",
                bytes: &sys,
                abi: ForeignAbi::WindowsNdis,
                class: DeviceClass::Net,
                claim,
            },
            &t,
            &mut dma,
            tiny,
        )
        .unwrap_err();
        assert_eq!(err, LoadDriverError::Foreign(LoadError::ResourceDenied));
    }
}

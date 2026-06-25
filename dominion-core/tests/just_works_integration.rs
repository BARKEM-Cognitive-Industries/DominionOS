//! End-to-end "it just works" integration test — drives the whole cross-cutting
//! pipeline through dominion-core's **public API only**, the way the OS GUI/terminal
//! would: load a driver, launch a sandboxed foreign app, let the app communicate over
//! a capability-gated channel, run a polyglot program, and install a package with its
//! dependencies. Each step is capability-confined; this proves the pillars compose.

use dominion_core::capability::{Capability, Rights};
use dominion_core::cheri::SoftwareTags;
use dominion_core::driver::{DeviceClass, ModelDmaMem, ResourceClaim};

use dominion_core::personality::appchannel::{CapBus, CapChannel};
use dominion_core::personality::applaunch::{launch_app, Grants, SyscallStep};
use dominion_core::personality::driverload::{load_driver, DriverSource};

use dominion_core::dominionlink::DominionId;
use dominion_core::firewall::Domain;
use dominion_core::hash::Hash256;
use dominion_core::ndn::Name;
use dominion_core::foreign::ForeignAbi;
use dominion_core::packaging::depot::{default_depot, Depot, Manifest, PackageKind};
use dominion_core::packaging::PackageRegistry;
use dominion_core::polyglot::runtime;
use dominion_core::pubsub::ReactivePlane;

fn minimal_elf() -> Vec<u8> {
    let mut b = vec![0u8; 64];
    b[0..4].copy_from_slice(b"\x7FELF");
    b[4] = 2; // 64-bit
    b[5] = 1; // little-endian
    b
}

#[test]
fn the_whole_just_works_pipeline_composes() {
    // ── 1. Load a driver from the registry, confined to its MMIO window ──
    let tags = SoftwareTags::new([0x5Au8; 32]);
    let mut dma = ModelDmaMem::new();
    let envelope = ResourceClaim { mmio_base: 0, mmio_len: 0xFFFF_FFFF, irq: 0 };
    let driver = load_driver(DriverSource::Registry("rtl8139"), &tags, &mut dma, envelope)
        .expect("rtl8139 loads from the registry");
    assert_eq!(driver.class, DeviceClass::Net);
    let (_base, len) = driver.window();
    assert!(len > 0);

    // ── 2. Launch a sandboxed Linux app and run a tiny program in it ──
    let launcher = Capability::mint(0x100_000, 0x100_000, Rights::ALL);
    let mut app = launch_app(
        &minimal_elf(),
        "/sandbox/app",
        &launcher,
        Grants::sandboxed(0x1000, 0x4000),
    )
    .expect("the ELF launches confined");
    let results = app.run_program(&[
        SyscallStep::new(2).path("out.txt"),          // open(O_CREAT)
        SyscallStep::new(1).fd(3).data(b"it works!"), // write
        SyscallStep::new(60).fd(0),                   // exit(0)
    ]);
    assert!(results[0] >= 3);
    assert_eq!(results[1], 9);
    assert_eq!(app.exit_code(), Some(0));

    // ── 3. The app communicates over a capability-gated channel, then is cut off ──
    let mut plane = ReactivePlane::new();
    let topic = Name::parse("/app/ipc");
    let id = DominionId(Hash256::of(b"app"));
    let pubcap = plane.mint_publish(topic.clone(), Domain::Personal, id);
    let subcap = plane.mint_subscribe(topic.clone(), Domain::Personal, id);
    let mut bus = CapBus::new();
    let sender = CapChannel::open(pubcap);
    let mut receiver = CapChannel::open(subcap);
    sender.send(&mut bus, b"hello system").expect("publish-capable channel can send");
    assert_eq!(receiver.recv(&bus).unwrap(), vec![b"hello system".to_vec()]);

    // ── 4. Run a program in a polyglot language (Python) ──
    let run = runtime::run_named("python", "def main():\n    return 6 * 7\n")
        .expect("python runs");
    assert_eq!(run.value.display(), "42");

    // ── 5. Install a package and its dependencies, verified + confined ──
    let depot = default_depot();
    let mut reg = PackageRegistry::new();
    let grant = Capability::mint(0, 1 << 20, Rights::READ.union(Rights::WRITE));
    let order = depot
        .install_with_deps("text-editor", &mut reg, &grant)
        .expect("text-editor installs with its dependency");
    assert_eq!(order, vec!["mathx".to_string(), "text-editor".to_string()]);
    assert_eq!(reg.count(), 2);
}

// A minimal but real Windows PE/COFF `.sys` importing only shimmed NDIS symbols —
// enough for the loader to parse, admit, and lower an rtl8139 driver.
fn build_pe_sys(funcs: &[&str]) -> Vec<u8> {
    let sec_rva = 0x1000u32;
    let sec_raw = 0x200usize;
    let mut idata = vec![0u8; 40];
    let ilt_off = idata.len();
    let mut thunks: Vec<u64> = Vec::new();
    let mut name_blobs: Vec<Vec<u8>> = Vec::new();
    let ilt_bytes = (funcs.len() + 1) * 8;
    let mut cursor = ilt_off + ilt_bytes;
    for f in funcs {
        thunks.push((sec_rva as usize + cursor) as u64);
        let mut blob = vec![0u8, 0u8];
        blob.extend_from_slice(f.as_bytes());
        blob.push(0);
        cursor += blob.len();
        name_blobs.push(blob);
    }
    thunks.push(0);
    let dll_rva = sec_rva as usize + cursor;
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
    idata.extend_from_slice(b"ndis.sys\0");

    let e_lfanew = 0x80usize;
    let mut b = vec![0u8; e_lfanew];
    b[0..2].copy_from_slice(b"MZ");
    b[0x3C..0x40].copy_from_slice(&(e_lfanew as u32).to_le_bytes());
    b.extend_from_slice(b"PE\0\0");
    let opt_size = 0xF0usize;
    let mut coff = vec![0u8; 20];
    coff[0..2].copy_from_slice(&0x8664u16.to_le_bytes());
    coff[2..4].copy_from_slice(&1u16.to_le_bytes());
    coff[16..18].copy_from_slice(&(opt_size as u16).to_le_bytes());
    b.extend_from_slice(&coff);
    let mut opt = vec![0u8; opt_size];
    opt[0..2].copy_from_slice(&0x20Bu16.to_le_bytes());
    opt[108..112].copy_from_slice(&16u32.to_le_bytes());
    opt[120..124].copy_from_slice(&sec_rva.to_le_bytes());
    opt[124..128].copy_from_slice(&(idata.len() as u32).to_le_bytes());
    b.extend_from_slice(&opt);
    let mut sh = vec![0u8; 40];
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
fn download_install_then_load_a_foreign_driver_package() {
    // 1. "Download": publish a Windows rtl8139 .sys as a signed Driver package.
    let sys = build_pe_sys(&["NdisMRegisterMiniport", "NdisAllocateMemory"]);
    let mut depot = Depot::new();
    depot.publish(
        Manifest::new("rtl8139.sys", "1.0", PackageKind::Driver, Rights::READ),
        &sys,
        b"vendor-seed",
    );

    // 2. Install (signature-verified, capability-confined) and fetch the content.
    let mut reg = PackageRegistry::new();
    let grant = Capability::mint(0, 1 << 20, Rights::READ.union(Rights::WRITE));
    let fetched = depot.install_and_fetch("rtl8139.sys", &mut reg, &grant).unwrap();
    assert_eq!(fetched.len(), 1);
    let (name, kind, content) = &fetched[0];
    assert_eq!(kind, &PackageKind::Driver);

    // 3. Load the installed driver binary through the unified loader → it just runs,
    //    admitted through the default-closed shim and confined to its device window.
    let tags = SoftwareTags::new([7u8; 32]);
    let mut dma = ModelDmaMem::new();
    let envelope = ResourceClaim { mmio_base: 0, mmio_len: 0xFFFF_FFFF, irq: 0 };
    let loaded = load_driver(
        DriverSource::Foreign {
            name,
            bytes: content,
            abi: ForeignAbi::WindowsNdis,
            class: DeviceClass::Net,
            claim: ResourceClaim { mmio_base: 0xFEBC_0000, mmio_len: 0x100, irq: 11 },
        },
        &tags,
        &mut dma,
        envelope,
    )
    .expect("the downloaded+installed driver loads end to end");
    assert_eq!(loaded.class, DeviceClass::Net);
    let adm = loaded.admission.as_ref().expect("foreign load proves admission");
    assert!(adm.is_authentic(&tags));
}

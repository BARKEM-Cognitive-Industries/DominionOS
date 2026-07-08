//! The bare-metal self-test battery.
//!
//! These are the kernel's integration tests. They run *inside the booted OS* on
//! the QEMU virtual machine — after the GDT/IDT/PIC are up and the heap is
//! mapped — so they prove every subsystem works on a real machine, not just on
//! the host. The same battery is driven two ways:
//!
//! * the `selftest` shell command, for an operator at the live terminal, and
//! * [`run_and_exit`], the headless CI entry point (`--features qemu_test`),
//!   which reports over serial and signals pass/fail through `isa-debug-exit`.
//!
//! Keeping one battery means the interactive and automated paths can never drift.

use crate::{exit_qemu, serial_println, QemuExitCode};
use dominion_core::capability::{Capability, CapError, Rights};
use dominion_core::codec::{CodecError, CodecRegistry};
use dominion_core::hash::Hash256;
use dominion_core::lang::{Interpreter, Value};
use dominion_core::object::{Datum, Object, ObjectGraph};
use dominion_core::persist::Persistence;
use dominion_core::sched::{DomainState, IpcError, Scheduler};
use dominion_core::state::{Action, Machine};
use dominion_core::vfs::{Vfs, VfsError};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Snapshot `nblocks` 512-byte blocks starting at `start_lba`, so a destructive
/// self-test write can be rolled back afterwards. The battery runs against
/// whatever disk is actually attached — on bare metal that is the operator's real
/// drive — so every raw write it performs must be paired with a restore, or the
/// test would silently corrupt live data (e.g. the GPT partition-entry array at
/// LBA 2-33, or a saved filesystem image). Returns `None` if the region can't be
/// read, in which case the caller must not proceed with the destructive write.
fn snapshot_region(
    dev: &mut dyn dominion_core::persist::BlockDevice,
    start_lba: u64,
    nblocks: usize,
) -> Option<Vec<u8>> {
    let mut buf = alloc::vec![0u8; nblocks * 512];
    dev.read_blocks(start_lba, &mut buf).ok()?;
    Some(buf)
}

/// Write a snapshot taken by [`snapshot_region`] back, undoing a self-test's
/// scratch write so the disk is left byte-for-byte as it was found.
fn restore_region(dev: &mut dyn dominion_core::persist::BlockDevice, start_lba: u64, snap: &[u8]) {
    let _ = dev.write_blocks(start_lba, snap);
}

/// Run the whole battery, invoking `report(name, passed)` for each check.
/// Returns `(passed, failed)`.
pub fn run(phys_offset: u64, mut report: impl FnMut(&str, bool)) -> (u32, u32) {
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut check = |name: &str, ok: bool, report: &mut dyn FnMut(&str, bool)| {
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
        report(name, ok);
    };

    // ---- hashing / content addressing ----
    check(
        "sha256 abc known-answer",
        Hash256::of(b"abc").to_hex()
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        &mut report,
    );
    check(
        "sha256 empty known-answer",
        Hash256::of(b"").to_hex()
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        &mut report,
    );

    // ---- on-device LLM inference (nn::model runs the embedded .aem on the metal) ----
    check(
        "llm forward predicts oracle token (embedded .aem)",
        dominion_core::nn::model::demo_selftest(&dominion_core::parallel::Serial),
        &mut report,
    );

    // ---- capability security ----
    let root = Capability::mint(0x1000, 0x1000, Rights::ALL);
    let ro = root.restrict(Rights::READ).unwrap();
    check(
        "capability attenuation (derive r--)",
        ro.rights().contains(Rights::READ) && !ro.rights().contains(Rights::WRITE),
        &mut report,
    );
    check(
        "capability monotonicity blocks escalation",
        matches!(
            ro.restrict(Rights::READ.union(Rights::WRITE)),
            Err(CapError::MonotonicityViolation)
        ),
        &mut report,
    );
    check(
        "capability bounds trap out-of-bounds",
        matches!(ro.check(0x3000, 1, Rights::READ), Err(CapError::OutOfBounds)),
        &mut report,
    );
    check(
        "capability integrity trap on tamper",
        matches!(
            ro.tamper().check(0x1000, 1, Rights::READ),
            Err(CapError::TagInvalid)
        ),
        &mut report,
    );

    // ---- semantic object graph ----
    let mut g = ObjectGraph::new();
    let id1 = g.put(Object::new("Invoice").with("amount", Datum::Int(100)));
    let id2 = g.put(Object::new("Invoice").with("amount", Datum::Int(100)));
    check(
        "object graph deduplicates identical objects",
        id1 == id2 && g.stored_count() == 1,
        &mut report,
    );
    let snap = g.commit("one");
    g.put(Object::new("Invoice").with("amount", Datum::Int(200)));
    let before = g.live_count();
    g.rollback(snap).unwrap();
    check(
        "object graph commit + rollback",
        before == 2 && g.live_count() == 1 && g.stored_count() == 2,
        &mut report,
    );

    // ---- deterministic state machine ----
    let prog = [
        Action::Set(String::from("a"), 10),
        Action::Add(String::from("a"), 5),
        Action::Rand(String::from("r")),
    ];
    let m1 = Machine::replay(7, &prog);
    let m2 = Machine::replay(7, &prog);
    check(
        "state machine reproducible replay",
        m1.state_hash() == m2.state_hash() && m1.get("a") == Some(15),
        &mut report,
    );
    check(
        "state machine rewind to prior step",
        m1.rewound_to(1).unwrap().get("a") == Some(10),
        &mut report,
    );

    // ---- Dominion language ----
    check(
        "dominion arithmetic + precedence",
        matches!(Interpreter::new().eval_str("2 + 3 * 4"), Ok(Value::Int(14))),
        &mut report,
    );
    check(
        "dominion recursion (factorial 6 = 720)",
        matches!(
            Interpreter::new().eval_str("fn f(n){ if n<2 {return 1;} return n*f(n-1);} f(6)"),
            Ok(Value::Int(720))
        ),
        &mut report,
    );
    check(
        "dominion parallel-map operator =>",
        matches!(
            Interpreter::new().eval_str("fn d(x){return x*2;} sum([1,2,3,4] => d)"),
            Ok(Value::Int(20))
        ),
        &mut report,
    );
    check(
        "dominion objects + field access",
        matches!(
            Interpreter::new().eval_str("let p = Point{ x:3, y:4 }; p.x + p.y"),
            Ok(Value::Int(7))
        ),
        &mut report,
    );

    // capability-gated cell: denied without rights, allowed with them
    let denied = Interpreter::with_rights(Rights::READ)
        .eval_str("cell S [cap: Capability<StorageWrite>] { fn p(x){return x;} } S::p(1)")
        .is_err();
    let allowed = matches!(
        Interpreter::new()
            .eval_str("cell S [cap: Capability<StorageWrite>] { fn p(x){return x+1;} } S::p(41)"),
        Ok(Value::Int(42))
    );
    check(
        "dominion cell denied without capability",
        denied,
        &mut report,
    );
    check("dominion cell runs with capability", allowed, &mut report);

    // end-to-end storage pipeline → semantic graph commit
    check(
        "dominion storage pipeline commits graph",
        matches!(
            Interpreter::new().eval_str(
                "let xs=[Doc{n:1},Doc{n:2}]; SystemGraph::commit(xs => NeuralCodec::encode)"
            ),
            Ok(Value::Str(_))
        ),
        &mut report,
    );

    // ---- keystone K2: codec / blob registry ----
    let reg = CodecRegistry::with_defaults();
    let rcap = Capability::mint(0, 0x1000, Rights::READ);
    check(
        "codec text import/export round-trips",
        match reg.import(Some("a.txt"), b"hi there", &rcap) {
            Ok(obj) => {
                obj.kind == "Text" && reg.export(&obj, &rcap).as_deref() == Ok(b"hi there".as_ref())
            }
            Err(_) => false,
        },
        &mut report,
    );
    check(
        "codec PPM decode/encode is lossless",
        (|| -> Option<bool> {
            let raw = b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
            let img = reg.import(Some("p.ppm"), raw, &rcap).ok()?;
            let back = reg.export(&img, &rcap).ok()?;
            let img2 = reg.import(Some("p.ppm"), &back, &rcap).ok()?;
            Some(img.kind == "Image" && img.id() == img2.id())
        })()
        .unwrap_or(false),
        &mut report,
    );
    check(
        "codec unknown format preserved verbatim as Blob",
        {
            let weird = &[0u8, 159, 146, 150];
            match reg.import(Some("x.bin"), weird, &rcap) {
                Ok(obj) => {
                    obj.kind == "Blob" && reg.export(&obj, &rcap).as_deref() == Ok(weird.as_ref())
                }
                Err(_) => false,
            }
        },
        &mut report,
    );
    check(
        "codec import requires READ capability",
        matches!(
            reg.import(Some("a.txt"), b"x", &Capability::mint(0, 0x1000, Rights::WRITE)),
            Err(CodecError::Capability(CapError::InsufficientRights))
        ),
        &mut report,
    );

    // ---- keystone K1: POSIX-projection VFS ----
    let wcap = Capability::mint(0, 0x1000, Rights::ALL);
    let mut vg = ObjectGraph::new();
    let mut vfs = Vfs::with_fhs();
    let txt = |s: &str| Object::new("Text").with("content", Datum::Text(String::from(s)));
    check(
        "vfs write/read round-trips over the graph",
        vfs.write_object(&mut vg, "/etc/motd", txt("welcome"), &wcap).is_ok()
            && vfs
                .read_object(&vg, "/etc/motd", &wcap)
                .map(|o| o.get("content") == Some(&Datum::Text(String::from("welcome"))))
                .unwrap_or(false),
        &mut report,
    );
    check(
        "vfs edit creates new immutable object, keeps old",
        (|| -> Option<bool> {
            let id1 = vfs.write_object(&mut vg, "/f", txt("v1"), &wcap).ok()?;
            let id2 = vfs.write_object(&mut vg, "/f", txt("v2"), &wcap).ok()?;
            Some(id1 != id2 && vfs.resolve("/f") == Some(id2) && vg.contains(&id1))
        })()
        .unwrap_or(false),
        &mut report,
    );
    check(
        "vfs write requires WRITE capability",
        matches!(
            vfs.write_object(&mut vg, "/g", txt("x"), &Capability::mint(0, 0x1000, Rights::READ)),
            Err(VfsError::Capability(CapError::InsufficientRights))
        ),
        &mut report,
    );
    check(
        "vfs namespace snapshot is deterministic + change-sensitive",
        {
            let r1 = vfs.snapshot_namespace(&mut vg);
            let r1b = vfs.snapshot_namespace(&mut vg);
            let wrote = vfs.write_object(&mut vg, "/etc/new", txt("z"), &wcap).is_ok();
            let r2 = vfs.snapshot_namespace(&mut vg);
            wrote && r1 == r1b && r1 != r2
        },
        &mut report,
    );
    check(
        "vfs+codec end-to-end legacy file round-trip",
        (|| -> Option<bool> {
            let raw = b"P6\n1 1\n255\n\x10\x20\x30";
            let img = reg.import(Some("logo.ppm"), raw, &rcap).ok()?;
            vfs.write_object(&mut vg, "/usr/share/logo.ppm", img, &wcap).ok()?;
            let stored = vfs.read_object(&vg, "/usr/share/logo.ppm", &rcap).ok()?;
            let exported = reg.export(stored, &rcap).ok()?;
            let reimported = reg.import(Some("logo.ppm"), &exported, &rcap).ok()?;
            Some(stored.id() == reimported.id())
        })()
        .unwrap_or(false),
        &mut report,
    );

    // ---- M3 driver framework: PCI enumeration ----
    let pci_devices = crate::pci::enumerate();
    check(
        "PCI enumeration finds devices on the bus",
        !pci_devices.is_empty(),
        &mut report,
    );
    check(
        "PCI finds the i440fx host bridge (vendor 0x8086)",
        pci_devices.iter().any(|d| d.vendor_id == 0x8086),
        &mut report,
    );

    // ---- M1 persistence: virtio-blk + disk-backed graph ----
    // Build a small graph, save it through the block device, reload it, and
    // confirm the content-addressed root survives. Uses the real virtio-blk disk
    // when attached, else a RAM disk — the persistence code path is identical.
    let (device_real, sector_ok, persist_ok) = crate::block::with_block_device(|dev, is_real| {
        // Raw sector round-trip through the BlockDevice trait. `dev` is the real
        // disk when one is attached, so bounds-check the LBA (mirroring the AHCI/NVMe
        // helper below) and snapshot the sector first, restoring it afterwards so the
        // test never leaves the disk mutated.
        let mut pattern = [0u8; 512];
        for (i, b) in pattern.iter_mut().enumerate() {
            *b = (i as u8) ^ 0xA5;
        }
        let lba = 64u64;
        let sector_ok = dev.block_count() > lba
            && match snapshot_region(dev, lba, 1) {
                Some(saved) => {
                    let ok = dev.write_block(lba, &pattern).is_ok() && {
                        let mut readback = [0u8; 512];
                        dev.read_block(lba, &mut readback).is_ok() && readback == pattern
                    };
                    restore_region(dev, lba, &saved);
                    ok
                }
                None => false,
            };

        // Full graph persistence round-trip. `save` lays a superblock + payload at
        // LBA 0 — the disk's MBR/GPT area on real hardware — so snapshot exactly the
        // blocks it will touch and restore them once the round-trip is verified.
        let mut g = ObjectGraph::new();
        g.put(Object::new("Doc").with("n", Datum::Int(1)));
        g.put(Object::new("Doc").with("n", Datum::Int(2)));
        g.commit("persist test");
        let root_before = g.root_hash();
        let image_blocks = 1 + g.serialize().len().div_ceil(512);
        let persist_ok = match snapshot_region(dev, 0, image_blocks) {
            Some(saved) => {
                let ok = Persistence::save(dev, &g).is_ok()
                    && matches!(
                        Persistence::load(dev),
                        Ok(Some(ref loaded)) if loaded.root_hash() == root_before
                    );
                restore_region(dev, 0, &saved);
                ok
            }
            None => false,
        };
        (is_real, sector_ok, persist_ok)
    });
    let _ = device_real;
    check("block device sector write/read-back", sector_ok, &mut report);
    check("graph persists across save/load on disk", persist_ok, &mut report);

    // ---- Real storage drivers: AHCI (SATA) + NVMe sector write/read round-trips ----
    // Probe each controller directly and, if a disk is attached, prove a real DMA
    // write+read-back. Absent controller → skip (pass), so the suite runs anywhere.
    {
        let rw_roundtrip = |disk: &mut dyn dominion_core::persist::BlockDevice, key: u8| -> bool {
            let mut pat = [0u8; 512];
            for (i, b) in pat.iter_mut().enumerate() {
                *b = (i as u8) ^ key;
            }
            let lba = 8u64;
            if disk.block_count() <= lba {
                return false;
            }
            // On bare metal `disk` is the operator's real drive and LBA 8 sits inside
            // the GPT partition-entry array, so save the sector, prove the DMA
            // round-trip, then put the original contents back.
            let Some(saved) = snapshot_region(disk, lba, 1) else { return false; };
            let ok = disk.write_block(lba, &pat).is_ok() && {
                let mut rb = [0u8; 512];
                disk.read_block(lba, &mut rb).is_ok() && rb == pat
            };
            restore_region(disk, lba, &saved);
            ok
        };
        let ahci_ok = match crate::ahci::probe() {
            Some(mut disk) => rw_roundtrip(&mut disk, 0x3C),
            None => true,
        };
        check("AHCI (SATA) disk write/read round-trip", ahci_ok, &mut report);
        let nvme_ok = match crate::nvme::probe() {
            Some(mut disk) => rw_roundtrip(&mut disk, 0x5A),
            None => true,
        };
        check("NVMe SSD write/read round-trip", nvme_ok, &mut report);
        // USB is owned by the log-device probe at boot, so exercise it through that
        // handle (re-probing the same xHCI controller would corrupt it).
        let usb_ok = if crate::block::log_is_usb() {
            crate::block::with_log_device(|dev, _| rw_roundtrip(dev, 0x69))
        } else {
            true
        };
        check("USB mass-storage (xHCI/BOT) write/read round-trip", usb_ok, &mut report);

        // USB Mass Storage SCSI command path: exercise INQUIRY + the capacity learned by
        // READ CAPACITY(10) at probe, driving the Bulk-Only Transport state machine over
        // the already-owned USB handle (re-probing the boot xHCI controller would corrupt
        // it). Skip-as-pass when no USB storage is attached, like the other
        // optional-hardware probes above.
        let msc_scsi_ok = crate::block::with_log_usb(|usb| usb.self_test()).unwrap_or(true);
        check("USB mass-storage INQUIRY + READ CAPACITY (BOT/SCSI)", msc_scsi_ok, &mut report);
    }

    // ---- Debug bootlog: capture + persist-to-disk round-trip ----
    // The serial output is teed into an always-on ring; persist it to the tail of the
    // disk and read it back, confirming the boot/run log is recoverable off the image.
    check("bootlog captures boot output", crate::bootlog::captured_len() > 0, &mut report);
    // Persist + read back through the preferred LOG device (the removable USB when one is
    // present) — verifying the exact path used on bare metal to recover the log.
    let log_roundtrip = crate::block::with_log_device(|dev, _| {
        match crate::bootlog::persist_to(dev) {
            Some(_) => crate::bootlog::read_back(dev)
                .map(|b| b.windows(6).any(|w| w == b"[boot]"))
                .unwrap_or(false),
            None => false,
        }
    });
    let log_target = if crate::block::log_is_usb() { "USB" } else { "disk" };
    check("bootlog persists to disk and reads back", log_roundtrip, &mut report);
    let _ = log_target;

    // ---- Hardware enumeration: the machine self-describes (CPU/PCI/GPU/storage/net) ----
    let hw = crate::hwreport::report(0);
    check(
        "hardware report enumerates the machine",
        hw.iter().any(|l| l.starts_with("CPU:")) && hw.iter().any(|l| l.starts_with("PCI:")),
        &mut report,
    );

    // ---- Durable shell filesystem: VFS image survives a save/load on real disk ----
    // The desktop writes a `FileSystem` image to a high LBA on shutdown and restores it
    // on boot. Exercise that exact path on the booted machine: author a file, image it,
    // save_blob → load_blob → restore_from_bytes, and confirm the file comes back.
    let fs_persist_ok = crate::block::with_block_device(|dev, _| {
        let mut fs = dominion_core::filesystem::FileSystem::new();
        if fs.write_text("/home/jayden/onmetal.txt", "survived reboot").is_err() {
            return false;
        }
        let image = fs.to_bytes();
        // Snapshot the LBA-8192 blob region and restore it afterwards so the test
        // never overwrites a real filesystem image on an attached disk.
        let image_blocks = 1 + image.len().div_ceil(512);
        let Some(saved) = snapshot_region(dev, 8192, image_blocks) else { return false; };
        if Persistence::save_blob(dev, 8192, b"AEVFS001", &image).is_err() {
            restore_region(dev, 8192, &saved);
            return false;
        }
        let ok = match Persistence::load_blob(dev, 8192, b"AEVFS001") {
            Ok(Some(blob)) => {
                let mut booted = dominion_core::filesystem::FileSystem::new();
                booted.restore_from_bytes(&blob)
                    && booted.read_text("/home/jayden/onmetal.txt").as_deref() == Some("survived reboot")
            }
            _ => false,
        };
        restore_region(dev, 8192, &saved);
        ok
    });
    check("shell filesystem persists across save/load on disk", fs_persist_ok, &mut report);

    // ---- M2 process / isolation / scheduler ----
    // Run a real cooperative round-robin to completion on the booted machine.
    let mut sched = Scheduler::new();
    let worker_a = sched.spawn("worker-a", Capability::mint(0x10_0000, 0x1000, Rights::ALL));
    let worker_b = sched.spawn("worker-b", Capability::mint(0x10_1000, 0x1000, Rights::ALL));
    let mut budget: BTreeMap<dominion_core::sched::DomainId, u32> = BTreeMap::new();
    budget.insert(worker_a, 2);
    budget.insert(worker_b, 3);
    let mut dispatched = 0u32;
    let mut sched_ok = true;
    while let Some(id) = sched.next() {
        dispatched += 1;
        // A scheduler regression must record a FAIL, not panic the kernel: bail
        // cleanly if next() hands back an unknown id or over-dispatches a domain
        // whose budget is already spent (which would underflow `*left`).
        let Some(left) = budget.get_mut(&id) else {
            sched_ok = false;
            break;
        };
        if *left == 0 {
            sched_ok = false;
            break;
        }
        *left -= 1;
        if *left == 0 {
            sched.finish(id);
        } else {
            sched.yield_back(id);
        }
        if dispatched > 100 {
            break; // safety against a scheduling bug livelocking the test
        }
    }
    check(
        "scheduler runs all domains to completion (cooperative)",
        sched_ok
            && dispatched == 5
            && sched.live_count() == 0
            && sched.state(worker_a) == Some(DomainState::Finished),
        &mut report,
    );

    // SIP isolation: a domain reaching outside its own capability region traps.
    check(
        "domain isolation traps cross-region access",
        matches!(
            sched.check_access(worker_a, 0x10_1000, 16, Rights::READ),
            Err(CapError::OutOfBounds)
        ),
        &mut report,
    );

    // Zero-copy IPC: messages carry an object reference and need an open channel.
    let mut ipc = Scheduler::new();
    let p = ipc.spawn("producer", Capability::mint(0, 0x1000, Rights::ALL));
    let c = ipc.spawn("consumer", Capability::mint(0x1000, 0x1000, Rights::ALL));
    let shared = Object::new("Reading").with("v", Datum::Int(42)).id();
    let denied = ipc.send(p, c, shared) == Err(IpcError::NoChannel);
    ipc.open_channel(p, c).unwrap();
    let sent = ipc.send(p, c, shared).is_ok();
    let delivered = ipc.recv(c).map(|m| m.payload == shared && m.from == p).unwrap_or(false);
    check(
        "zero-copy IPC delivers only over an explicit channel",
        denied && sent && delivered,
        &mut report,
    );

    // ---- feature 3: ELF loader (parse + load + execute on metal) ----
    // A position-independent x86-64 program: `mov eax, 42 ; ret` (returns 42).
    let code = [0xB8u8, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    let elf_bytes = dominion_core::elf::build_exec_elf(0x40_0000, &code);
    check(
        "ELF parses: entry + one executable PT_LOAD segment",
        match dominion_core::elf::parse(&elf_bytes) {
            Ok(img) => {
                img.entry == 0x40_0000 && img.segments.len() == 1 && img.segments[0].is_executable()
            }
            Err(_) => false,
        },
        &mut report,
    );
    check(
        "ELF loads and the loaded entry executes (returns 42)",
        match crate::loader::load(&elf_bytes) {
            // W^X-safe: seal the image pages as R-X before executing.
            // The heap is mapped WRITABLE|NO_EXECUTE; seal_and_call strips
            // WRITABLE and NO_EXECUTE so the page is never simultaneously
            // writable and executable.
            Ok(prog) => unsafe {
                prog.seal_and_call(x86_64::VirtAddr::new(phys_offset)) == 42
            },
            Err(_) => false,
        },
        &mut report,
    );

    // ---- feature 1: networking (virtio-net + stack) ----
    use dominion_core::net::{Interface, Ipv4Addr, MacAddr};
    check(
        "virtio-net present with a valid MAC",
        crate::netif::present() && crate::netif::mac() != MacAddr::ZERO,
        &mut report,
    );

    // Pure stack on metal: our interface answers an ARP request for itself.
    let mut iface_logic = Interface::new(MacAddr([0x52, 0x54, 0, 0, 0, 1]), Ipv4Addr::new(10, 0, 2, 15));
    let peer = dominion_core::net::ArpPacket {
        opcode: dominion_core::net::ARP_REQUEST,
        sender_mac: MacAddr([0x52, 0x54, 0, 0, 0, 2]),
        sender_ip: Ipv4Addr::new(10, 0, 2, 2),
        target_mac: MacAddr::ZERO,
        target_ip: Ipv4Addr::new(10, 0, 2, 15),
    };
    let req_frame = dominion_core::net::build_ethernet(
        MacAddr::BROADCAST,
        peer.sender_mac,
        dominion_core::net::ETHERTYPE_ARP,
        &peer.build(),
    );
    check(
        "network stack builds an ARP reply on metal",
        iface_logic.handle_frame(&req_frame).is_some(),
        &mut report,
    );

    // Real round trip: ARP for the SLIRP gateway (10.0.2.2) over the actual NIC
    // and confirm we receive and learn its hardware address.
    let mut iface = Interface::new(crate::netif::mac(), Ipv4Addr::new(10, 0, 2, 15));
    let gateway = Ipv4Addr::new(10, 0, 2, 2);
    let learned = crate::netif::with_nic(|nic| {
        let request = iface.arp_request(gateway);
        nic.transmit(&request);
        for _ in 0..20_000_000u64 {
            if let Some(frame) = nic.poll_frame() {
                let _ = iface.handle_frame(&frame);
                if iface.arp.lookup(gateway).is_some() {
                    return true;
                }
            }
        }
        false
    })
    .unwrap_or(false);
    check(
        "virtio-net ARP round-trip with gateway (real RX/TX)",
        learned,
        &mut report,
    );

    // DNS round-trip: fan-out to all resolvers (mirrors KernelTransport::resolve).
    // QEMU's ICMP (ping) is locally spoofed at 0.1 ms and proves nothing about real
    // internet connectivity. DNS uses real UDP forwarding through slirp. We query
    // both the QEMU virtual resolver (10.0.2.3) AND public resolvers simultaneously
    // and accept whichever answers first — same fan-out strategy as production code.
    {
        use dominion_core::net::{
            build_dns_query, build_ethernet, build_ipv4, build_udp,
            parse_dns_answer, parse_ethernet, parse_ipv4, parse_udp,
            ETHERTYPE_IPV4, IPPROTO_UDP,
        };
        let my_mac = crate::netif::mac();
        let my_ip  = Ipv4Addr::new(10, 0, 2, 15);
        let gw_ip  = Ipv4Addr::new(10, 0, 2, 2);
        // Mirrors DNS_SERVERS in webnet.rs — QEMU virtual, Google, Cloudflare.
        let dns_servers: &[Ipv4Addr] = &[
            Ipv4Addr::new(10, 0, 2, 3),
            Ipv4Addr::new(8, 8, 8, 8),
            Ipv4Addr::new(1, 1, 1, 1),
        ];

        let dns_result = crate::netif::with_nic(|nic| {
            // Re-learn gateway MAC (cheap: it was just learned above).
            let mut iface2 = Interface::new(my_mac, my_ip);
            nic.transmit(&iface2.arp_request(gw_ip));
            let gw_mac = 'arp: {
                for _ in 0..20_000_000u64 {
                    if let Some(f) = nic.poll_frame() {
                        let _ = iface2.handle_frame(&f);
                        if let Some(m) = iface2.arp.lookup(gw_ip) {
                            break 'arp m;
                        }
                    }
                }
                return None; // ARP timed out
            };

            // Fan-out: send same DNS query to all resolvers simultaneously.
            let query_id: u16 = 0x4145; // "AE"
            let src_port: u16 = 53001;
            let qpayload = build_dns_query(query_id, "example.com");
            let udp = build_udp(src_port, 53, &qpayload);
            for &srv in dns_servers {
                let ip_pkt = build_ipv4(my_ip, srv, IPPROTO_UDP, &udp, 0xF001);
                let eth    = build_ethernet(gw_mac, my_mac, ETHERTYPE_IPV4, &ip_pkt);
                nic.transmit(&eth);
                crate::serial_println!(
                    "[dns-test] query -> {}.{}.{}.{}:53 id={:#06x}",
                    srv.0[0], srv.0[1], srv.0[2], srv.0[3], query_id
                );
            }

            // Busy-poll for reply; retransmit to all servers every ~20 M iters.
            for i in 0..100_000_000u64 {
                if i > 0 && i % 20_000_000 == 0 {
                    for &srv in dns_servers {
                        let ip_pkt = build_ipv4(my_ip, srv, IPPROTO_UDP, &udp, 0xF001);
                        let eth    = build_ethernet(gw_mac, my_mac, ETHERTYPE_IPV4, &ip_pkt);
                        nic.transmit(&eth);
                    }
                    crate::serial_println!("[dns-test] retransmit (iter {})", i);
                }
                let Some(f) = nic.poll_frame() else { continue };
                let Some(e) = parse_ethernet(&f) else { continue };
                if e.ethertype != ETHERTYPE_IPV4 { continue }
                let Some(ip4) = parse_ipv4(e.payload) else { continue };
                if ip4.protocol != IPPROTO_UDP { continue }
                let Some(u) = parse_udp(ip4.payload) else { continue };
                if u.src_port != 53 { continue }
                crate::serial_println!(
                    "[dns-test] DNS reply from {}.{}.{}.{} iter={}",
                    ip4.src.0[0], ip4.src.0[1], ip4.src.0[2], ip4.src.0[3], i
                );
                if let Some(addr) = parse_dns_answer(u.payload, query_id) {
                    crate::serial_println!(
                        "[dns-test] example.com resolved to {}.{}.{}.{}",
                        addr.0[0], addr.0[1], addr.0[2], addr.0[3]
                    );
                    return Some(addr);
                }
                crate::serial_println!("[dns-test] DNS frame ID/no-A mismatch, continuing");
            }
            crate::serial_println!("[dns-test] timeout — no DNS reply from any resolver");
            None
        }).flatten();

        check(
            "virtio-net DNS round-trip (10.0.2.3 + 8.8.8.8 + 1.1.1.1 fan-out)",
            dns_result.is_some(),
            &mut report,
        );
    }
    // ---- spec-driven RTL8139: a declarative DeviceSpec drives a *real* NIC ----
    // The same dominion_core::netspec::rtl8139_spec the host tests drive against a
    // model here brings up a live RealTek 8139 (vendor 0x10EC) over the PCI bus,
    // reads its MAC out of hardware registers and transmits an ARP frame — with no
    // device-specific control code in the kernel. Skipped (pass) if no RTL8139 is
    // attached, exercised when run-test.ps1 adds `-device rtl8139`.
    match crate::rtl8139::probe_and_demo() {
        Some(r) => {
            // QEMU assigns MACs from its default OUI 52:54:00 (the rtl8139 is the
            // second NIC, so its last octet differs from virtio-net's). Checking the
            // OUI proves we read a real hardware MAC without hardcoding the instance.
            let mac_ok = r.mac[0..3] == [0x52, 0x54, 0x00] && r.mac != [0u8; 6];
            check("rtl8139 spec-driven MAC read from real hardware", mac_ok, &mut report);
            check("rtl8139 spec-driven TX of a real Ethernet frame", r.tx_ok, &mut report);
        }
        None => {
            check("rtl8139 spec-driven NIC (no device attached, skipped)", true, &mut report);
        }
    }

    // ---- real Intel e1000/e1000e Gigabit NIC ----
    // A physical MAC (QEMU/VirtualBox default `-device e1000`, or a real Intel
    // controller): map BAR0, read the station address out of the RAL0/RAH0
    // registers, and confirm it is a valid non-zero hardware MAC. Non-destructive
    // (no reset, no ring teardown), so it is safe even when the e1000 is the live
    // interface. Skipped (pass) when no e1000-class device is attached.
    match crate::e1000::probe() {
        Some(p) => {
            let mac_ok = p.mac != [0u8; 6];
            crate::serial_println!(
                "[e1000] {:#06x} mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                p.device_id, p.mac[0], p.mac[1], p.mac[2], p.mac[3], p.mac[4], p.mac[5]
            );
            check("e1000 Intel Gigabit MAC read from real hardware", mac_ok, &mut report);
        }
        None => {
            check("e1000 Intel Gigabit NIC (no device attached, skipped)", true, &mut report);
        }
    }

    // ---- feature 5: DominionLink (native content-addressed networking) ----
    {
        use dominion_core::dominionlink::{DominionId, DominionLink, Dht, DnsBridge};
        let mut link = DominionLink::new(DominionId::from_pubkey(b"node-key"));
        let cid = link.publish(b"native payload");
        check(
            "dominionlink: self-certifying identity + verified fetch",
            DominionId::from_pubkey(b"node-key").certifies(b"node-key")
                && link.fetch(cid) == Some(b"native payload".as_ref())
                && !DominionLink::verify(cid, b"tampered"),
            &mut report,
        );
        let mut dht = Dht::new(DominionId::from_pubkey(b"me"));
        let target = DominionId::from_pubkey(b"target");
        for k in 0..12u8 {
            dht.insert(DominionId::from_pubkey(&[k]));
        }
        dht.insert(target);
        let closest = dht.closest(&target, 3);
        let mut dns = DnsBridge::new();
        dns.register("example.com", target);
        check(
            "dominionlink: DHT XOR lookup + DNS bridge",
            closest.first() == Some(&target) && dns.resolve("example.com") == Some(target),
            &mut report,
        );
    }

    // ---- feature 4: legacy sandbox containment ----
    {
        use dominion_core::sandbox::{Sandbox, SandboxError};
        let cap = Capability::mint(0x10_0000, 0x1000, Rights::READ.union(Rights::WRITE));
        let mut sb = Sandbox::new("guest", cap, "/containers/g1");
        sb.allow_syscalls(&[0, 1, 2, 3]);
        check(
            "sandbox: syscall whitelist + capability-bounded memory",
            sb.check_syscall(1).is_ok()
                && sb.check_syscall(59) == Err(SandboxError::SyscallDenied(59))
                && sb.check_memory(0x20_0000, 16, Rights::READ).is_err(),
            &mut report,
        );
        check(
            "sandbox: projected paths cannot escape the root",
            sb.translate_path("/etc/x").as_deref() == Ok("/containers/g1/etc/x")
                && sb.translate_path("/../../etc/shadow") == Err(SandboxError::PathEscape),
            &mut report,
        );
    }

    // ---- feature 7: Linux syscall-translation personality ----
    {
        use dominion_core::personality::{classify, LinuxPersonality, SyscallClass, O_CREAT, SYS_OPEN};
        let mut p = LinuxPersonality::new(100);
        let fd = p.open("/tmp/note", O_CREAT) as i32;
        let wrote = p.write(fd, b"hello") == 5;
        p.close(fd);
        let fd2 = p.open("/tmp/note", 0) as i32;
        let read_back = p.read(fd2, 5).map(|v| v == b"hello").unwrap_or(false);
        check(
            "personality: open/write/read translate to VFS ops",
            classify(SYS_OPEN) == SyscallClass::File && wrote && read_back,
            &mut report,
        );
        let (mut child, child_pid) = p.fork(101);
        let cfd = child.open("/tmp/note", 0) as i32;
        check(
            "personality: fork = snapshot-and-branch of the world",
            child_pid == 101 && child.read(cfd, 5).map(|v| v == b"hello").unwrap_or(false),
            &mut report,
        );
    }

    // ---- feature 6: Dominion-native web ----
    {
        use dominion_core::dominionweb::Page;
        let page = Page::new("Home").heading("Welcome").link("About", "dominion://about");
        let rendered = page.render_text();
        check(
            "dominionweb: content-addressed page + semantic render",
            page.content_id() == Page::new("Home").heading("Welcome").link("About", "dominion://about").content_id()
                && rendered.contains("Home")
                && page.links() == ["dominion://about"],
            &mut report,
        );
    }

    // ---- M4: compositor (and a real blit to the framebuffer) ----
    {
        use dominion_core::surface::{fb_at, Compositor, Surface};
        let mut comp = Compositor::new(64, 48, 0x101018);
        comp.add(Surface::solid(1, 0, 0, 64, 48, 0, 0x202060).unwrap()); // background panel
        comp.add(Surface::solid(2, 8, 8, 24, 16, 0, 0x40C040).unwrap()); // a green window on top
        let fb = comp.composite();
        let occludes = fb_at(&fb, 64, 12, 12) == 0x40C040 && fb_at(&fb, 64, 60, 44) == 0x202060;
        // Present it for real in the top-right corner of the screen (visual proof).
        crate::vga_buffer::blit_rgb(1000, 16, 64, 48, &fb);
        check("compositor: z-order occlusion + framebuffer blit", occludes, &mut report);
    }

    // ---- feature 2: codec -> VFS -> persistence, end-to-end ----
    {
        use dominion_core::codec::CodecRegistry;
        use dominion_core::vfs::Vfs;
        let reg = CodecRegistry::with_defaults();
        let rcap = Capability::mint(0, 0x1000, Rights::READ);
        let wcap = Capability::mint(0, 0x1000, Rights::ALL);
        let mut g = ObjectGraph::new();
        let mut v = Vfs::with_fhs();

        // Import a legacy file, store it at a path, snapshot the namespace.
        let original = b"a legacy file that must survive a reboot";
        let obj = reg.import(Some("readme.txt"), original, &rcap).unwrap();
        let oid = obj.id();
        v.write_object(&mut g, "/etc/readme.txt", obj, &wcap).unwrap();
        v.commit(&mut g, "snapshot", &wcap).unwrap();

        // Persist to disk, reload, and re-export the file to identical bytes.
        let ok = crate::block::with_block_device(|dev, _| {
            // Snapshot the LBA-0 image region (MBR/GPT area on real hardware) and
            // restore it once the reload round-trip has been checked.
            let image_blocks = 1 + g.serialize().len().div_ceil(512);
            let Some(saved) = snapshot_region(dev, 0, image_blocks) else { return false; };
            if Persistence::save(dev, &g).is_err() {
                restore_region(dev, 0, &saved);
                return false;
            }
            let result = match Persistence::load(dev) {
                Ok(Some(reloaded)) => match reloaded.get(&oid) {
                    Some(file) => reg.export(file, &rcap).as_deref() == Ok(original.as_ref()),
                    None => false,
                },
                _ => false,
            };
            restore_region(dev, 0, &saved);
            result
        });
        check("feature 2: codec -> VFS -> persist -> reload -> export", ok, &mut report);
    }

    // ---- randomness: hardware TRNG (RDRAND) + seeded DRNG ----
    {
        let s1 = crate::entropy::rdrand64();
        let s2 = crate::entropy::rdrand64();
        let health_ok = crate::entropy::health_check().map(|h| h.passed()).unwrap_or(false);
        check(
            "TRNG: RDRAND yields healthy, non-repeating entropy",
            s1.is_some() && s2.is_some() && s1 != s2 && health_ok && crate::entropy::conditioned_seed().is_some(),
            &mut report,
        );
        use dominion_core::random::Drng;
        let mut a = Drng::from_seed(b"seed");
        let mut b = Drng::from_seed(b"seed");
        check(
            "DRNG: seeded stream is reproducible (determinism contract)",
            a.next_u64() == b.next_u64() && a.next_u64() == b.next_u64(),
            &mut report,
        );
    }

    // ---- Stage 13: post-quantum crypto (CAL + hybrid signatures) ----
    {
        use dominion_core::crypto::{CryptoLayer, Hybrid, LamportSig, SignatureScheme};
        use alloc::boxed::Box;
        let s = LamportSig::new("pq", "post-quantum");
        let (sk, pk) = s.keygen(b"id-seed");
        let sig = s.sign(&sk, b"capability token");
        check(
            "crypto: hash-based (PQ) signature verifies; tamper rejected",
            s.verify(&pk, b"capability token", &sig) && !s.verify(&pk, b"forged token", &sig),
            &mut report,
        );
        let h = Hybrid {
            classical: Box::new(LamportSig::new("c", "classical")),
            post_quantum: Box::new(LamportSig::new("q", "post-quantum")),
        };
        let (hsk, hpk) = h.keygen(b"hybrid-seed");
        let hsig = h.sign(&hsk, b"msg");
        let mut bad = hsig.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        let cal = CryptoLayer::with_defaults();
        check(
            "crypto: hybrid needs BOTH schemes; CAL provides agility",
            h.verify(&hpk, b"msg", &hsig) && !h.verify(&hpk, b"msg", &bad) && cal.algorithms().len() >= 2,
            &mut report,
        );
    }

    // ---- Stage 14: universal encryption / zero-plaintext vault ----
    {
        use dominion_core::vault::{Key, Vault};
        let mut v = Vault::new();
        let key = Key::from_seed(b"obj-key");
        let ik = Key::from_seed(b"index-key");
        let id = v.seal(b"classified", key, b"nonce0001", &ik, &["secret"]);
        let ct_is_encrypted = v.ciphertext(id).map(|c| c != b"classified").unwrap_or(false);
        let wrong = Key::from_seed(b"attacker");
        check(
            "vault: encrypted-at-rest; Storage != Read; searchable",
            ct_is_encrypted
                && v.open(id, key).as_deref() == Some(b"classified".as_ref())
                && v.open(id, wrong).is_none()
                && v.search(&ik, "secret") == [id],
            &mut report,
        );
        check(
            "vault: destroying the key is cryptographic secure-deletion",
            v.destroy_key(id) && v.open(id, key).is_none() && v.ciphertext(id).is_some(),
            &mut report,
        );
    }

    // ---- Stage 11.14: capability firewall (authority-graph reachability) ----
    {
        use dominion_core::firewall::{CapabilityFirewall, Domain, FwError};
        let mut fw = CapabilityFirewall::new();
        for n in 1..=4 {
            fw.register(n, Domain::Financial);
        }
        fw.delegate(1, 2).unwrap();
        fw.delegate(2, 3).unwrap();
        fw.delegate(3, 4).unwrap();
        fw.register(9, Domain::AiAgent);
        let cross_denied = fw.delegate(1, 9) == Err(FwError::CrossDomainDenied);
        let reach = fw.reachable(1, 4);
        fw.revoke(2);
        check(
            "firewall: reachability + cross-domain deny + recursive revoke",
            reach && cross_denied && fw.is_revoked(4) && !fw.reachable(1, 4),
            &mut report,
        );
    }

    // ---- Stage 11.15: capability airlock (cross-domain transfer) ----
    {
        use dominion_core::airlock::{Airlock, AirlockError, TransferPolicy};
        use dominion_core::firewall::Domain;
        let mut a = Airlock::new();
        a.add_policy(TransferPolicy {
            from: Domain::Financial,
            to: Domain::AiAgent,
            max_rights: Rights::READ,
            ttl: Some(10),
            approvals_required: 2,
        });
        let src = Capability::mint(0x1000, 0x1000, Rights::READ.union(Rights::WRITE));
        let issued = a.transfer(src, Domain::Financial, Domain::AiAgent, 2, 100).unwrap();
        let sanitized = issued.capability.rights().contains(Rights::READ)
            && !issued.capability.rights().contains(Rights::WRITE);
        let reverse_denied =
            matches!(a.transfer(src, Domain::AiAgent, Domain::Financial, 2, 0), Err(AirlockError::NoPolicy));
        check(
            "airlock: sanitizes authority, one-way, temporal expiry",
            sanitized && reverse_denied && issued.is_expired(110) && !issued.is_expired(105),
            &mut report,
        );
    }

    // ---- Stage 11.5: runtime attestation ----
    {
        use dominion_core::attest::Attestor;
        let baseline: [(&str, &[u8]); 2] = [("kernel", b"state-v1"), ("shell", b"shell-v1")];
        let att = Attestor::from_components(&baseline);
        let tampered: [(&str, &[u8]); 2] = [("kernel", b"state-v1"), ("shell", b"shell-trojan")];
        check(
            "attestation: unmodified verifies, tamper detected",
            att.attest(&baseline) && !att.attest(&tampered),
            &mut report,
        );
    }

    // ---- Stage 13: lattice PQ KEM (LWE) on metal ----
    {
        use dominion_core::lattice::LatticeKem;
        let (pk, sk) = LatticeKem::keygen(b"metal-id");
        let (ct, enc) = LatticeKem::encapsulate(&pk, b"metal-eph");
        check(
            "lattice KEM: encapsulate/decapsulate agree on the shared secret",
            enc == LatticeKem::decapsulate(&sk, &ct),
            &mut report,
        );
    }

    // ---- Stage 14: real AES-256-GCM (NIST-correct) on metal ----
    {
        use dominion_core::memcrypt::{gcm_decrypt, gcm_encrypt, salt_from_label, Aes, SealedRegion};
        let aes = Aes::new_256(&[7u8; 32]);
        let (ct, tag) = gcm_encrypt(&aes, &[9u8; 12], b"aad", b"secret payload");
        let round_trips = gcm_decrypt(&aes, &[9u8; 12], b"aad", &ct, &tag).as_deref()
            == Some(b"secret payload".as_ref());
        let tamper_rejected = {
            let mut bad = ct.clone();
            bad[0] ^= 1;
            gcm_decrypt(&aes, &[9u8; 12], b"aad", &bad, &tag).is_none()
        };
        // Memory-at-rest: plaintext only materialises on open.
        let region = SealedRegion::seal([42u8; 32], b"label", salt_from_label(b"label"), b"resident secret");
        let at_rest_encrypted = region.at_rest() != b"resident secret";
        check(
            "AES-256-GCM: round-trip + tamper-reject + memory-at-rest sealing",
            round_trips && tamper_rejected && at_rest_encrypted
                && region.open().as_deref() == Some(b"resident secret".as_ref()),
            &mut report,
        );
    }

    // ---- zero-knowledge proof (Schnorr NIZK) on metal ----
    // Gated: requires the demo-crypto Cargo feature (31-bit illustrative group).
    // Enable only in test/CI builds via `--features demo-crypto`; never in production.
    #[cfg(feature = "demo-crypto")]
    {
        use dominion_core::zk::{schnorr_prove, schnorr_verify, SchnorrParams};
        let params = SchnorrParams::new_demo_insecure();
        let x = 123_456u128;
        let y = params.public_key(x);
        let proof = schnorr_prove(&params, x, b"metal-nonce");
        check(
            "ZK: Schnorr proof of knowledge verifies; wrong witness rejected",
            schnorr_verify(&params, y, &proof)
                && !schnorr_verify(&params, params.public_key(x + 1), &proof),
            &mut report,
        );
    }

    // ---- extended Dominion data types (quantum + tensor) on metal ----
    {
        use dominion_core::datatypes::{QubitState, Tensor};
        let mut q = QubitState::zeros(2);
        q.h(0);
        q.cnot(0, 1); // Bell state
        let entangled = q.probability(0b00) > 0.49 && q.probability(0b11) > 0.49;
        let a = Tensor::new(alloc::vec![2, 2], alloc::vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m = a.matmul(&a).unwrap();
        check(
            "datatypes: qubit Bell entanglement + tensor matmul",
            entangled && m.data() == [7.0, 10.0, 15.0, 22.0],
            &mut report,
        );
    }

    // ---- CHERI tag HAL: portable backend on commodity hardware ----
    {
        use dominion_core::cheri::{perms, CapabilityTags, HardwareTags, TaggedCap};
        let hal = HardwareTags::detect(false, [3u8; 32]); // no CHERI silicon here
        let cap = hal.mint(0x1000, 0x100, perms::READ);
        let forged = TaggedCap::untagged(0, u64::MAX, perms::ALL);
        check(
            "CHERI HAL: software tag backend validates real caps, rejects forgeries",
            !hal.hardware_backed() && hal.validate(&cap) && !hal.validate(&forged),
            &mut report,
        );
    }

    // ---- NDN forwarding plane on metal ----
    {
        use dominion_core::ndn::{Data, Forwarder, InterestOutcome, Name};
        let mut fw = Forwarder::new();
        fw.register_route(Name::parse("/v"), 9);
        let name = Name::parse("/v/clip");
        let forwarded = matches!(fw.recv_interest(1, &name), InterestOutcome::Forward(_));
        let aggregated = fw.recv_interest(2, &name) == InterestOutcome::Aggregated;
        let delivered = fw.recv_data(Data::new(name.clone(), b"bytes")) == alloc::vec![1, 2];
        let cached = matches!(fw.recv_interest(3, &name), InterestOutcome::FromCache(_));
        check(
            "NDN: Interest forward + PIT aggregate + Data satisfy + CS cache hit",
            forwarded && aggregated && delivered && cached,
            &mut report,
        );
    }

    // ---- reactive / subscription plane (pub/sub over NDN) on metal ----
    {
        use dominion_core::firewall::Domain;
        use dominion_core::ndn::Name;
        use dominion_core::object::{Datum, Object};
        use dominion_core::pubsub::{Delivery, ReactivePlane, SubOptions, TopicError};

        let mut plane = ReactivePlane::new();
        let topic = Name::parse("/jayden/inbox");
        let publisher = dominion_core::dominionlink::DominionId(Hash256::of(b"producer"));
        let subscriber = dominion_core::dominionlink::DominionId(Hash256::of(b"consumer"));
        let pubcap = plane.mint_publish(topic.clone(), Domain::Personal, publisher);
        let subcap = plane.mint_subscribe(topic.clone(), Domain::Personal, subscriber);

        // Two subscribers seat standing Interest; one publish fans out to both (PIT tree).
        let s1 = plane
            .subscribe(&subcap, 1, SubOptions { delivery: Delivery::ExactlyOnce, ..Default::default() })
            .unwrap();
        plane.subscribe(&subcap, 2, SubOptions::default()).unwrap();
        let receipt = plane
            .publish(&pubcap, Object::new("Msg").with("body", Datum::Text(String::from("hi"))))
            .unwrap();
        let fanned_out = receipt.notified.len() == 2;

        // Exactly-once dedup: republishing the same object delivers it once.
        plane.publish(&pubcap, Object::new("Msg").with("body", Datum::Text(String::from("hi")))).unwrap();
        let once = plane.poll(s1.id).map(|v| v.len()).unwrap_or(99) == 1;

        // A SUBSCRIBE capability cannot publish; recursive revoke blocks delivery.
        let cannot_publish =
            plane.publish(&subcap, Object::new("Msg")).err() == Some(TopicError::Unauthorized);
        plane.revoke(s1.node);
        let revoked = plane.poll(s1.id).err() == Some(TopicError::Revoked);

        check(
            "pubsub: PIT fan-out + exactly-once dedup + cap-gated publish + instant revoke",
            fanned_out && once && cannot_publish && revoked,
            &mut report,
        );
    }

    // ---- DominionLink transport: congestion + NAT traversal + mobility + offline ----
    {
        use dominion_core::net::Ipv4Addr;
        use dominion_core::transport::{
            Bbr, Candidate, CandidateType, Connection, Cubic, IceAgent, Locator, LocatorDirectory,
            MigrateError, OfflineReplica, ReplicaStore, MSS,
        };

        // Congestion: CUBIC grows then backs off on loss; BBR tracks a BDP window.
        let mut cubic = Cubic::new();
        let base = cubic.cwnd_segments();
        for _ in 0..5 {
            cubic.on_ack(1.0);
        }
        let grew = cubic.cwnd_segments() > base;
        let peak = cubic.cwnd_segments();
        cubic.on_loss();
        let backed_off = cubic.cwnd_segments() < peak && !cubic.in_slow_start();
        let mut bbr = Bbr::new();
        for _ in 0..6 {
            bbr.on_ack(MSS, 2.0);
        }
        let bbr_ok = bbr.cwnd_bytes() >= MSS && bbr.pacing_rate() > 0.0;

        // NAT traversal: relay fallback wins when direct paths are blocked.
        let mut ice = IceAgent::new(true);
        let l = Locator::new(Ipv4Addr::new(10, 0, 0, 1), 5000);
        ice.gather(CandidateType::Host, l, l);
        ice.gather(CandidateType::Relay, Locator::new(Ipv4Addr::new(10, 0, 0, 200), 7000), l);
        let remote = [Candidate {
            typ: CandidateType::Relay,
            addr: Locator::new(Ipv4Addr::new(10, 0, 0, 201), 7000),
            base: Locator::new(Ipv4Addr::new(10, 0, 0, 2), 5000),
        }];
        let nat_ok = ice
            .nominate(&remote, |a, b| {
                a.typ == CandidateType::Relay || b.typ == CandidateType::Relay
            })
            .map(|p| IceAgent::is_relayed(&p))
            .unwrap_or(false);

        // Mobility: a connection survives an IP change via path validation.
        let peer = dominion_core::dominionlink::DominionId(Hash256::of(b"roamer"));
        let mut dir = LocatorDirectory::new();
        let old_path = Locator::new(Ipv4Addr::new(10, 0, 0, 5), 4433);
        dir.announce(peer, old_path);
        let mut conn = Connection::new(peer, dir.resolve(&peer).unwrap(), [0x42u8; 32]);
        conn.send();
        conn.send();
        let new_path = Locator::new(Ipv4Addr::new(10, 0, 0, 80), 4433);
        let spoof = conn.begin_migration(new_path);
        let refused = conn.complete_migration(spoof ^ 0x1) == Err(MigrateError::ValidationFailed);
        let chal = conn.begin_migration(new_path);
        let migrated = conn.complete_migration(chal) == Ok(new_path) && conn.seq() == 2;

        // Offline-first: writes queue offline, reconcile by hash with dedup.
        let mut replica = OfflineReplica::new();
        let mut remote_store = ReplicaStore::new();
        replica.set_online(false);
        replica.write(b"note-A");
        replica.write(b"note-B");
        replica.write(b"note-A"); // duplicate content
        let rec = replica.reconcile(&mut remote_store, &[]);
        let offline_ok = rec.pushed == 2 && replica.pending() == 0 && replica.is_online();

        check(
            "transport: CUBIC/BBR congestion + ICE relay fallback + connection migration + offline reconcile",
            grew && backed_off && bbr_ok && nat_ok && refused && migrated && offline_ok,
            &mut report,
        );
    }

    // ---- native thread pool: admission, work stealing, SMP execution on metal ----
    {
        use dominion_core::governor::PressureLevel;
        use dominion_core::pool::{admit, Admission, LocalQueue, PoolConfig, Priority, ThreadPool, WorkItem};
        use crate::threadpool::KernelSpawn;
        use dominion_core::parallel::Spawn;

        // Admission table spot checks.
        let admit_rt_critical = admit(Priority::RealTime, PressureLevel::Critical);
        let admit_idle_tight  = admit(Priority::Idle,     PressureLevel::Tight);
        let admit_norm_comfy  = admit(Priority::Normal,   PressureLevel::Comfortable);
        check(
            "pool: admission table (rt/critical=urgent, idle/tight=refused, norm/comfy=accepted)",
            admit_rt_critical == Admission::AcceptedUrgent
                && !admit_idle_tight.is_accepted()
                && admit_norm_comfy == Admission::Accepted,
            &mut report,
        );

        // Priority ordering in local queue.
        let mut q = LocalQueue::new(8);
        q.push(WorkItem { task_idx: 0, priority: Priority::Background });
        q.push(WorkItem { task_idx: 1, priority: Priority::RealTime });
        q.push(WorkItem { task_idx: 2, priority: Priority::Normal });
        let first = q.pop().map(|w| w.priority);
        check(
            "pool: local queue pops highest priority first",
            first == Some(Priority::RealTime),
            &mut report,
        );

        // Work stealing: worker 1 steals from overloaded worker 0.
        let mut pool = ThreadPool::new(PoolConfig { workers: 2, steal_batch: 2, ..PoolConfig::default() });
        for i in 0..4 {
            pool.push_to_worker(0, WorkItem { task_idx: i, priority: Priority::Normal });
        }
        let stolen_item = pool.pop_for(1);
        check(
            "pool: work stealing transfers tasks from busy to idle worker",
            stolen_item.is_some() && pool.metrics().stolen > 0,
            &mut report,
        );

        // Submit + pop + complete metrics.
        let mut pool2 = ThreadPool::new(PoolConfig { workers: 1, queue_depth: 8, ..PoolConfig::default() });
        pool2.submit(10, Priority::Normal, PressureLevel::Comfortable);
        pool2.submit(11, Priority::Idle,   PressureLevel::Critical);  // refused
        let got = pool2.pop_for(0);
        if got.is_some() { pool2.mark_complete(); }
        check(
            "pool: submit/pop/complete metrics are consistent",
            pool2.metrics().submitted == 1
                && pool2.metrics().refused   == 1
                && pool2.metrics().completed == 1,
            &mut report,
        );

        // KernelSpawn: correctness — every result[i] == [i as f64].
        let sp = KernelSpawn;
        let results = sp.run(8, &|i| alloc::vec![i as f64]);
        let correct = results.iter().enumerate().all(|(i, r)| r.first() == Some(&(i as f64)));
        check(
            "pool: KernelSpawn::run produces bit-identical results regardless of worker count",
            correct && results.len() == 8,
            &mut report,
        );

        // KernelSpawn zero-task edge case.
        let empty = sp.run(0, &|i| alloc::vec![i as f64]);
        check("pool: KernelSpawn::run(0, _) returns empty vec", empty.is_empty(), &mut report);

        // max_workers reports at least 1.
        check("pool: KernelSpawn::max_workers() >= 1", sp.max_workers() >= 1, &mut report);
    }

    // ---- resource governor: degradation + reclaim-by-recomputability on metal ----
    {
        use dominion_core::governor::{Admission, PlacementTarget, PressureLevel, ReclaimClass, ResourceGovernor};

        let mut g = ResourceGovernor::new();
        g.set_mem_budget(1, 1000);
        // Admission control: essential grant, then speculative refused under pressure,
        // then over-budget essential deferred — never an OOM-kill.
        let granted = g.reserve(1, 600, true) == Admission::Granted;
        let tight = g.reserve(1, 200, true) == Admission::Degraded(PressureLevel::Tight);
        let refused = g.reserve(1, 50, false) == Admission::Refused;
        let deferred = g.reserve(1, 500, true) == Admission::Deferred;

        // Reclaim-by-recomputability: evict regenerable before dirty, never pinned.
        g.track(1, Hash256::of(b"latent"), 100, ReclaimClass::Regenerable);
        g.track(1, Hash256::of(b"keys"), 100, ReclaimClass::Pinned);
        let evicted = g.reclaim(1, 50);
        let reclaim_ok = evicted == alloc::vec![Hash256::of(b"latent")]
            && g.classify(Hash256::of(b"keys")) == Some(ReclaimClass::Pinned);

        // Placement bandit migrates off a contended accelerator.
        for _ in 0..20 {
            g.reward_placement(PlacementTarget::Npu, -1.0);
            g.reward_placement(PlacementTarget::Gpu, 1.0);
        }
        let placement_ok = g.placement().best() == PlacementTarget::Gpu;

        check(
            "governor: admission tiers + reclaim-by-recomputability + placement bandit",
            granted && tight && refused && deferred && reclaim_ok && placement_ok,
            &mut report,
        );
    }

    // ---- identity recovery: Shamir k-of-n on metal ----
    {
        use dominion_core::recovery::{reconstruct, split};
        let secret = b"master-key-blob!";
        let mut entropy = [0u8; 32];
        dominion_core::random::Drng::from_seed(b"metal-entropy").fill(&mut entropy);
        let shares = split(secret, 3, 5, &entropy).unwrap();
        let recovered = reconstruct(&shares[1..4]); // any 3 of 5
        check(
            "recovery: Shamir 3-of-5 reconstructs the secret from a quorum",
            recovered.as_deref() == Some(secret.as_ref()),
            &mut report,
        );
    }

    // ---- PQ-signed capability token (XMSS-style) on metal ----
    {
        use dominion_core::tokensig::{verify, TokenAuthority};
        let authority = TokenAuthority::new(b"metal-issuer", 2);
        let cap = Capability::mint(0x2000, 0x100, Rights::READ);
        let token = authority.sign(&cap, 0).unwrap();
        check(
            "tokensig: PQ-signed capability token verifies against authority root",
            verify(authority.public_key(), &token),
            &mut report,
        );
    }

    // ---- polyglot sandbox VM: bounded authority on metal ----
    {
        use dominion_core::wasm::{Op, Sandbox, Trap};
        // (3 + 4) * 5 = 35 with no host imports granted.
        let mut s = Sandbox::new(
            alloc::vec![Op::Const(3), Op::Const(4), Op::Add, Op::Const(5), Op::Mul, Op::Return],
            0,
            0,
            1000,
        );
        let computed = s.run() == Ok(35);
        // An un-granted host call traps (cannot escape the sandbox).
        let mut s2 = Sandbox::new(alloc::vec![Op::Call { id: 1, argc: 0 }, Op::Return], 0, 0, 100);
        let contained = s2.run() == Err(Trap::UngrantedHostCall);
        check(
            "sandbox VM: computes guest code, traps un-granted host calls",
            computed && contained,
            &mut report,
        );
    }

    // ---- multi-source entropy pool on metal ----
    {
        use dominion_core::random::EntropyPool;
        let mut pool = EntropyPool::new();
        pool.absorb(1, b"rdrand");
        pool.absorb(2, b"jitter");
        let mut mirror = EntropyPool::new();
        mirror.absorb(1, b"rdrand");
        mirror.absorb(2, b"jitter");
        check(
            "entropy pool: mixes multiple sources into a reproducible seed",
            pool.extract() == mirror.extract() && pool.sample_count() == 2,
            &mut report,
        );
    }

    // ---- Phase A: identity-bound secure session (PQ KEM + AES-GCM) on metal ----
    {
        use dominion_core::session::{KemIdentity, Session};
        let alice = KemIdentity::generate(b"metal-alice");
        let bob = KemIdentity::generate(b"metal-bob");
        let (mut a, ct) = Session::initiate(alice.id, bob.id, &bob.public, b"eph", 100).unwrap();
        let mut b = Session::accept(&bob, alice.id, &ct, 100);
        let frame = a.seal(1, b"identity-bound secret").unwrap();
        let opened = b.open(1, &frame).as_deref() == Ok(b"identity-bound secret".as_ref());
        // Impersonation (Bob's id, attacker's key) is refused at handshake.
        let mallory = KemIdentity::generate(b"metal-mallory");
        let impersonation_refused = Session::initiate(alice.id, bob.id, &mallory.public, b"e", 100).is_err();
        check(
            "session: identity-bound encrypted channel; impersonation refused",
            opened && impersonation_refused,
            &mut report,
        );
    }

    // ---- Phase A: vault crypto-agility (migrate + rotate + PQ-signed) on metal ----
    {
        use dominion_core::vault::{CipherSuite, Key, SignedSeal, Vault};
        use dominion_core::crypto::CryptoLayer;
        let mut v = Vault::new();
        let key = Key::from_seed(b"metal-obj");
        let newk = Key::from_seed(b"metal-rot");
        let ik = Key::from_seed(b"metal-ik");
        let id = v.seal(b"long-term archive", key, b"n1", &ik, &[]);
        let migrated = v.migrate(id, key, CipherSuite::Aes256Gcm, b"n2")
            && v.open(id, key).as_deref() == Some(b"long-term archive".as_ref());
        let rotated = v.rotate_key(id, key, newk, b"n3")
            && v.open(id, newk).is_some()
            && v.open(id, key).is_none();
        let cal = CryptoLayer::with_defaults();
        let sid = v
            .seal_signed(&SignedSeal {
                suite: CipherSuite::Aes256Gcm,
                plaintext: b"authentic",
                key,
                nonce: b"n",
                index_key: &ik,
                keywords: &[],
                cal: &cal,
                algo_id: "lamport-pq",
                signing_seed: b"signer",
            })
            .unwrap();
        let signed_ok = v.verify_signature(sid, &cal) == Some(true);
        check(
            "vault: cipher migration + key rotation + PQ-signed authenticity",
            migrated && rotated && signed_ok,
            &mut report,
        );
    }

    // ---- Phase A: HIBC PQ-signed NDN Data on metal ----
    {
        use dominion_core::crypto::CryptoLayer;
        use dominion_core::ndn::SignedData;
        let cal = CryptoLayer::with_defaults();
        let sd = SignedData::produce(&cal, "lamport-pq", b"metal-producer", "doc/x", b"payload").unwrap();
        let name_binds = sd.data.name.certifies(&sd.producer_pk);
        let verifies = sd.verify(&cal);
        check(
            "NDN HIBC: data name is the producer key; PQ signature verifies",
            name_binds && verifies,
            &mut report,
        );
    }

    // ---- Phase A: entropy ledger replay (determinism contract) on metal ----
    {
        use dominion_core::random::EntropyLedger;
        let mut live = EntropyLedger::new();
        // Capture two real TRNG samples at the boundary, recorded as input events.
        let mut s = [0u8; 32];
        if let Some(seed) = crate::entropy::conditioned_seed() {
            s = seed;
        }
        live.record(b"boot-seed", s);
        live.record(b"reseed", [0x5a; 32]);
        let mut a = live.seed_drng();
        // Replay reconstructs the generator from recorded events alone.
        let mut replay = EntropyLedger::new();
        for e in live.events() {
            replay.record(&e.label, e.sample);
        }
        let mut b = replay.seed_drng();
        check(
            "entropy ledger: recorded true-entropy events replay deterministically",
            a.next_u64() == b.next_u64() && live.len() == 2,
            &mut report,
        );
    }

    // ---- Phase B: deterministic simulation testing (DST) on metal ----
    {
        use dominion_core::dst::{converges, run_convergence};
        // Same seed reproduces the run bit-for-bit; replicas converge under loss.
        // (Small params: the full seed-sweep runs host-side; this is a smoke test.)
        let reproducible = run_convergence(0xBEEF, 4, 10) == run_convergence(0xBEEF, 4, 10);
        let diverges_on_diff_seed = run_convergence(1, 4, 10) != run_convergence(2, 4, 10);
        let converged = converges(0xBEEF, 4, 10) && converges(7, 3, 8);
        check(
            "DST: seeded run reproduces exactly; replicas converge under loss/reorder",
            reproducible && diverges_on_diff_seed && converged,
            &mut report,
        );
    }

    // ---- Phase B: fuzz harness no-panic on a trust boundary, on metal ----
    {
        use dominion_core::fuzz::{sweep, FuzzInput};
        // Feed garbage to the ELF + object-graph parsers; reaching the end without
        // a panic (which would abort the kernel) is the property.
        sweep(0xF422, 120, |seed| {
            let mut input = FuzzInput::new(seed);
            let bytes = input.blob(256);
            let _ = dominion_core::elf::parse(&bytes);
            let _ = dominion_core::object::ObjectGraph::deserialize(&bytes);
        });
        check("fuzz: parsers survive 120 garbage inputs without panicking", true, &mut report);
    }

    // ---- Phase C: Dominion language depth (|>, linear, type-directed routing) ----
    {
        use dominion_core::lang::Interpreter;
        let mut it = Interpreter::new();
        // Pipeline operator threads a value left-to-right.
        let piped = it.eval_str("fn add(a, b) { return a + b; } 10 |> add(5) |> add(100)").unwrap();
        // Tensor value is type-routed to the GPU node.
        let routed = it.eval_str("route(tensor(2, 2, [1, 2, 3, 4]))").unwrap();
        // Affine value invalidated at scope end (no GC).
        let mut it2 = Interpreter::new();
        it2.eval_str("fn f() { linear s = 42; return 1; } f()").unwrap();
        check(
            "dominion: |> pipeline + type-directed routing + affine scope-end invalidation",
            matches!(piped, dominion_core::lang::Value::Int(115))
                && matches!(routed, dominion_core::lang::Value::Str(ref s) if s == "GPU")
                && it2.invalidations().len() == 1,
            &mut report,
        );
    }

    // ---- Phase C: DCG compiler (capability-checked, proof-carrying, refines) ----
    {
        use dominion_core::dcg::{Dcg, DcgError};
        use dominion_core::lang::ast::Item;
        use dominion_core::lang::parser::parse_source;
        let prog = parse_source("fn g(a, b) { let s = a + b; return s * a; }").unwrap();
        let mut ok = false;
        if let Item::Fn(f) = &prog.items[0] {
            let dcg = Dcg::compile(f, Rights::ALL).unwrap();
            // Refines the interpreter on a sample input, and carries a stable proof.
            let refines = dcg.eval(&[6, 7]).unwrap() == (6 + 7) * 6;
            let proof_stable = dcg.proof() == Dcg::compile(f, Rights::ALL).unwrap().proof();
            ok = refines && proof_stable;
        }
        // Compile-time capability check refuses unauthorized privileged code.
        let priv_prog = parse_source("fn priv_op(x) { return x + 1; }").unwrap();
        let mut denied = false;
        if let Item::Fn(f) = &priv_prog.items[0] {
            denied = matches!(Dcg::compile(f, Rights::READ), Err(DcgError::Unauthorized(_)));
        }
        check(
            "DCG: compiles + refines interpreter + proof-carrying + compile-time cap check",
            ok && denied,
            &mut report,
        );
    }

    // ---- Phase D: multikernel + heterogeneous scheduling + consistency ----
    {
        use dominion_core::multikernel::{
            quorum_agree, ConvergentState, Multikernel, NodeKind, WorkGraph,
        };
        // Heterogeneous execution graph: load(CPU) -> convolve(GPU) -> infer(NPU).
        let mut g = WorkGraph::new();
        let load = g.add("load", NodeKind::Cpu, &[]);
        let conv = g.add("convolve", NodeKind::Gpu, &[load]);
        let infer = g.add("infer", NodeKind::Npu, &[conv]);
        let sched = g.schedule(&[NodeKind::Cpu, NodeKind::Gpu, NodeKind::Npu]).unwrap();
        let step_of = |t: usize| sched.iter().find(|s| s.task == t).unwrap().step;
        let routed = sched.iter().find(|s| s.task == conv).unwrap().node == NodeKind::Gpu
            && sched.iter().find(|s| s.task == infer).unwrap().node == NodeKind::Npu;
        let ordered = step_of(load) < step_of(conv) && step_of(conv) < step_of(infer);

        // Cores share nothing until an explicit message is delivered (HLC-ordered).
        let mut mk = Multikernel::new(2);
        mk.local_write(0, "x", 9, 1);
        let before = mk.core(1).get("x").is_none();
        mk.send(0, 1, "x", 9, 2);
        mk.deliver(1, 3);
        let after = mk.core(1).get("x") == Some(9);

        // CRDT convergent path + quorum consensus path.
        let mut a = ConvergentState::new();
        let mut b = ConvergentState::new();
        a.bump(0, 3);
        b.bump(1, 4);
        let converged = a.merge(&b).value() == b.merge(&a).value();
        let quorum = quorum_agree(&[(0, 42), (1, 42), (2, 42), (3, 7)], 5) == Some(42)
            && quorum_agree(&[(0, 1), (1, 2)], 5).is_none();

        check(
            "multikernel: heterogeneous DAG schedule + explicit msg-passing + CRDT/quorum",
            routed && ordered && before && after && converged && quorum,
            &mut report,
        );
    }

    // ---- Phase M: fleet-scale Byzantine-fault-tolerant consensus (bft.rs) ----
    {
        use dominion_core::bft::{
            bft_run, build_signers, verify_equivocation, EquivocationProof, Out, Phase,
            Validator,
        };
        // Small signer set (16 one-time keys/validator) so the crypto is light on metal.
        let (signers, set) = build_signers(4, 4);
        // Safety + liveness: validator 0 (the view-0 leader) is silent; a view-change must
        // still let the honest majority commit one agreed value.
        let (agreed, committed) = bft_run(0xB1, &signers, &set, 1, 1, 48);

        // Equivocation is detected and the proof is independently verifiable.
        let mut a = Validator::with_signer(0, signers[0].clone(), set.clone());
        let mut b = Validator::with_signer(1, signers[1].clone(), set.clone());
        let va = a.equivocate(0, Phase::PrePrepare, b"value-A");
        let vb = a.equivocate(0, Phase::PrePrepare, b"value-B");
        let _ = b.ingest(&va);
        let caught = b.ingest(&vb).iter().any(|o| {
            if let Out::Equivocation(p) = o {
                verify_equivocation(&set, p)
            } else {
                false
            }
        });
        // A hand-built proof also verifies (and a non-conflicting pair does not).
        let proof_ok = verify_equivocation(&set, &EquivocationProof { a: va.clone(), b: vb.clone() });

        check(
            "bft: silent-leader view-change commits + Byzantine safety + equivocation proof",
            agreed && committed >= 1 && caught && proof_ok,
            &mut report,
        );
    }

    // ---- Phase N: distributed SASOS (remote fault-in by hash + migration + CHERI-D) ----
    {
        use dominion_core::dsasos::{dsasos_scenario, GenError, GenStore};
        let (resolved, migrated, lie_refused) = dsasos_scenario();
        // CHERI-D generation IDs trap a use-after-free across the shared store.
        let mut g = GenStore::new();
        let r = g.alloc(b"obj");
        let live = g.deref(r) == Ok(b"obj".as_ref());
        g.free(r.slot);
        let uaf_trapped = g.deref(r) == Err(GenError::StaleGeneration);
        check(
            "dsasos: remote fault verify-by-rehash + cell migration + CHERI-D UAF trap",
            resolved && migrated && lie_refused && live && uaf_trapped,
            &mut report,
        );
    }

    // ---- Phase O: decentralized compute marketplace (dual-pool + reverse auction) ----
    {
        use dominion_core::enforcement::Tier;
        use dominion_core::marketplace::{Bid, Envelope, Marketplace, Pool, ReverseAuction, SharingPredicate};
        let mut m = Marketplace::new();
        let off_by_default = !m.is_contributing();
        m.enable_contribution(1 << 16, SharingPredicate::relaxed());
        let env = Envelope { cpu_units: 2, mem_bytes: 1024, max_price: 100 };
        let mut a = ReverseAuction::new(env);
        a.bid(Bid { supplier: 0, cpu_units: 4, mem_bytes: 4096, price: 80 });
        a.bid(Bid { supplier: 1, cpu_units: 2, mem_bytes: 2048, price: 40 });
        let win = a.settle().map(|b| b.supplier == 1 && b.price == 40).unwrap_or(false);
        let awarded = m.award("infer", &a, Pool::Public).is_some();
        let tier_gate = m.dispatch_private(Tier::Software, Tier::Cheri).is_err()
            && m.dispatch_private(Tier::Cheri, Tier::MemoryTagging).is_ok();
        check(
            "marketplace: opt-in default-off + reverse auction + private-pool tier gate",
            off_by_default && win && awarded && tier_gate,
            &mut report,
        );
    }

    // ---- Phase P: compute-backed settlement & PoUW (PoI/PoL + ledger + tokenomics) ----
    {
        use dominion_core::hash::Hash256;
        use dominion_core::settlement::{PoI, PoL, SettlementLedger, StakedLearning, Treasury, Wallet};
        // Proof-of-Inference: grid-snap makes an honest re-run hash-match.
        let claim = PoI::claim(Hash256::of(b"model"), b"input", &[1.0, 2.0, 3.0]);
        let poi_ok = PoI::verify(&claim, Hash256::of(b"model"), b"input", &[1.0, 2.0, 3.0]);
        // Proof-of-Learning: the states are genuinely committed (so the Merkle
        // membership binding passes), but [9] is not step([2]) = [4], so the
        // fabricated transition is slashed.
        let step = |p: &[u8]| -> alloc::vec::Vec<u8> { p.iter().map(|&b| b.wrapping_mul(b)).collect() };
        let states2 = [alloc::vec![2u8], alloc::vec![9u8]];
        let claim2 = PoL::commit(&states2, 500);
        let mut staked = StakedLearning::accept(claim2);
        let slashed = !staked.spot_check(&PoL::reveal(&states2, 1), step)
            && staked.forfeit() == 500;
        // Wallet + ledger: locked spend refused, unlocked spend recorded.
        let mut alice = Wallet::new(Hash256::of(b"a"), 100, b"pin");
        let mut bob = Wallet::new(Hash256::of(b"b"), 0, b"pinb");
        let mut led = SettlementLedger::new();
        alice.unlock(b"pin");
        let paid = led.pay(&mut alice, &mut bob, 30, 1).is_ok() && bob.balance() == 30;
        // Tokenomics: fully reserved + fee-burn keeps the invariant.
        let mut t = Treasury::new(1000);
        let reserves = t.mint(500) && {
            t.burn_fee(50);
            t.proof_of_reserves()
        };
        check(
            "settlement: PoI grid-snap + PoL slash + wallet/ledger + proof-of-reserves",
            poi_ok && slashed && paid && reserves,
            &mut report,
        );
    }

    // ---- Phase Q: hardened memory + amnesic + deniable + anti-fingerprinting ----
    {
        use dominion_core::amnesic::{BootAnchor, Hibernation, ScrubPolicy, SecureRam, VolatileDomain, WatchdogAction};
        use dominion_core::deniable::{DeniableVault, DomainKind, DuressAction};
        use dominion_core::hardalloc::{AllocFault, HardenedAllocator};
        use dominion_core::privacy::{DevicePrivacyPolicy, Persona, StreamIsolation};

        // Hardened allocator: zero-on-alloc, UAF trap, zero-on-free.
        let mut ha = HardenedAllocator::new(1 << 16, 0xBEEF);
        let h = ha.alloc(32).unwrap();
        ha.write(h, 0, b"secret").unwrap();
        let off = ha.read(h, 0, 6).map(|v| v == b"secret").unwrap_or(false);
        ha.free(h).unwrap();
        let uaf = ha.read(h, 0, 1) == Err(AllocFault::UseAfterFree);

        // Amnesic: volatile domain refuses commit + scrubs; boot-anchor watchdog; hibernation tamper-reject.
        let mut vd = VolatileDomain::new(b"k");
        vd.put("x", b"y");
        let no_commit = vd.try_commit().is_err();
        vd.wipe();
        let wiped = vd.key_is_zeroed() && vd.get("x").is_none();
        let mut ram = SecureRam::new(ScrubPolicy::amnesic());
        ram.write_page(b"key");
        ram.lock();
        let scrubbed = ram.all_scrubbed();
        let mut wd = BootAnchor::new(b"anchor");
        let watchdog = wd.check(Some(b"anchor")) == WatchdogAction::Continue
            && wd.check(None) == WatchdogAction::ScrubAndShutdown;
        let img = Hibernation::seal(b"pk", b"state");
        let hib = Hibernation::resume(b"pk", &img).is_some() && Hibernation::resume(b"evil", &img).is_none();

        // Deniable: duress opens only the decoy; hidden existence unprovable.
        let mut dv = DeniableVault::new(b"seed", b"duress", b"real", DuressAction::ScrubVolatile);
        dv.put(DomainKind::Decoy, b"decoy-doc");
        dv.put(DomainKind::Hidden, b"real-secret");
        let duress = dv.unlock(b"duress").map(|(k, o, _)| k == DomainKind::Decoy && !o.iter().any(|x| x == b"real-secret")).unwrap_or(false);
        let hidden = dv.unlock(b"real").map(|(k, o, _)| k == DomainKind::Hidden && o.iter().any(|x| x == b"real-secret")).unwrap_or(false);
        let unprovable = !dv.hidden_existence_provable_from(b"duress");

        // Anti-fingerprinting: per-context Tor isolation + persona uniformity + USB lock.
        let iso = StreamIsolation::new(b"session");
        let tor = iso.isolated("a.example", "b.example");
        let persona = Persona::normalize(false).fingerprint() == Persona::normalize(false).fingerprint();
        let usb = !DevicePrivacyPolicy::hardened().usb_data_allowed(true);

        check(
            "hardening: hardened-alloc UAF + amnesic/watchdog/hibernation + deniable + anti-fp",
            off && uaf && no_commit && wiped && scrubbed && watchdog && hib && duress && hidden && unprovable && tor && persona && usb,
            &mut report,
        );
    }

    // ---- Phase R: Dominion LSP + anonymous-credential selective disclosure ----
    {
        use dominion_core::bft::Signer;
        use dominion_core::credential::{verify_presentation, Attribute, Credential};
        use dominion_core::lsp::{Lsp, Severity};
        // LSP: clean source has no errors; a parse error surfaces as a diagnostic; symbols found.
        let clean = Lsp::is_valid("let x = 2 + 2") && !Lsp::diagnostics("let x = 2 + 2").iter().any(|d| d.severity == Severity::Error);
        let errs = Lsp::diagnostics("let = 5").iter().any(|d| d.severity == Severity::Error);
        let syms = Lsp::document_symbols("fn add(a, b) { a + b }").len() == 1;
        // Anonymous credential: disclose only the predicate; verify; forging fails.
        let mut iss = Signer::new(b"issuer", 3);
        let attrs = alloc::vec![
            Attribute::new("birthdate", b"1990-01-01"),
            Attribute::new("over_18", b"true"),
        ];
        let cred = Credential::issue(&mut iss, attrs, b"holder").unwrap();
        let pres = cred.present(&["over_18"], "site");
        let disclosed_ok = verify_presentation(iss.public_key(), &pres)
            .map(|v| v.len() == 1 && v[0].name == "over_18")
            .unwrap_or(false);
        let hides_birthdate = !pres.disclosed().iter().any(|a| a.name == "birthdate");
        check(
            "lsp+credential: diagnostics/symbols + selective-disclosure (predicate only, unlinkable)",
            clean && errs && syms && disclosed_ok && hides_birthdate,
            &mut report,
        );
    }

    // ---- Phase E: generative storage (grid-snap + band routing + prefetch) ----
    {
        use dominion_core::neural::{
            grid_snap, route_for_ratio, BlockAccessPredictor, Route,
        };
        // Grid Snap collapses FP non-associativity drift.
        let snapped = grid_snap((0.1 + 0.2) + 0.3, 1e-6) == grid_snap(0.1 + (0.2 + 0.3), 1e-6);
        // Band-pass router: mid-band → accelerator, extremes → CPU.
        let routed = route_for_ratio(2000) == Route::Accelerator
            && route_for_ratio(1000) == Route::Cpu
            && route_for_ratio(5000) == Route::Cpu;
        // CNN-LSTM-style predictor learns the stride and prefetches.
        let mut p = BlockAccessPredictor::new();
        for b in [0u64, 4, 8, 12, 16] {
            p.observe(b);
        }
        let prefetch = p.predict_next() == Some(20);
        check(
            "generative storage: grid-snap + band-pass routing + stride prefetch",
            snapped && routed && prefetch,
            &mut report,
        );
    }

    // ---- Phase E: active defense (SLIC watermark + cryptographic poisoning) ----
    {
        use dominion_core::defense::{protect, recover, unauthorized_reencode, watermark_intact};
        let content = b"semantic media frame";
        let key = b"owner-key";
        let payload = b"owner=jayden";
        let media = protect(content, key, payload);
        let authorized = recover(&media, key).as_deref() == Some(content.as_ref())
            && watermark_intact(&media, key, payload);
        // Unauthorized re-encode self-degrades the media and breaks the watermark.
        let tampered = unauthorized_reencode(&media);
        let poisoned = recover(&tampered, key).is_none() && !watermark_intact(&tampered, key, payload);
        check(
            "active defense: SLIC watermark verifies; unauthorized re-encode self-degrades",
            authorized && poisoned,
            &mut report,
        );
    }

    // ---- Phase F: semantic audio (OBA + HRTF + JSCM + EDF) ----
    {
        use dominion_core::audio::{
            AudioObject, AudioTask, EdfScheduler, Hrtf, SemanticTokenizer, FRAME_DEADLINE_US,
        };
        // JSCM: a noise-corrupted semantic token decodes back to the right meaning.
        let tk = SemanticTokenizer::new(8, 100);
        let sent = tk.encode(300);
        let jscm = tk.decode(sent.wrapping_add(40)) == sent;
        // HRTF: a right-side source is louder in the right ear with negative ITD.
        let hrtf = Hrtf::new();
        let f = hrtf.render(&AudioObject { token: 1, x: 5.0, y: 1.0, z: 0.0, gain: 1.0 });
        let spatial = f.right > f.left && f.itd_us < 0.0;
        // EDF: orders by earliest deadline and meets the isochronous 16.6 ms budget.
        let mut s = EdfScheduler::new();
        s.add(AudioTask { id: 1, deadline_us: 2 * FRAME_DEADLINE_US, cost_us: 4000 });
        s.add(AudioTask { id: 2, deadline_us: FRAME_DEADLINE_US, cost_us: 4000 });
        let edf = s.order() == alloc::vec![2u32, 1] && s.meets_all_deadlines(0);
        check(
            "semantic audio: JSCM token recovery + HRTF spatialization + EDF deadlines",
            jscm && spatial && edf,
            &mut report,
        );
    }

    // ---- Phase G: object-centric AI-native UI (views + command bar + undo) ----
    {
        use dominion_core::object::{Datum, Object};
        use dominion_core::ui::{
            interpret, invoke, normalize, permission_list, render, History, InputEvent, UiError,
            View,
        };
        let obj = Object::new("Invoice").with("amount", Datum::Int(100));
        // One object, multiple views on demand.
        let multiview = render(&obj, View::Table).contains("Invoice")
            && render(&obj, View::Assistive).contains("amount");
        // AI command bar: natural language → capability-gated action.
        let intent = interpret("open the invoice").unwrap();
        let gated = invoke(&intent, Rights::READ).is_ok()
            && invoke(&intent, Rights::NONE) == Err(UiError::Unauthorized);
        // Abstract input: mouse press and touch normalise to the same action.
        let unified = normalize(InputEvent::Pointer { x: 5, y: 6, pressed: true })
            == normalize(InputEvent::Touch { x: 5, y: 6 });
        // Universal undo over content-addressed roots.
        let mut h = History::new(Hash256::of(b"g"));
        h.commit(Hash256::of(b"a"));
        h.commit(Hash256::of(b"b"));
        let undo = h.undo().unwrap() == Hash256::of(b"a") && h.redo() == Some(Hash256::of(b"b"));
        // Visible capabilities: the one settings surface lists exactly what is held.
        let settings = permission_list(Rights::READ.union(Rights::WRITE)) == alloc::vec!["read", "write"];
        check(
            "object UI: views-on-demand + AI command bar (cap-gated) + abstract input + undo",
            multiview && gated && unified && undo && settings,
            &mut report,
        );
    }

    // ---- Phase H: identity, key hierarchy, passwordless auth, recovery ----
    {
        use dominion_core::crypto::CryptoLayer;
        use dominion_core::identity::{login_challenge, Account, AuthResponse, MasterSeed, Recovery};
        use dominion_core::recovery::split;
        let seed = MasterSeed::from_entropy(b"trng boundary");
        // DEKs rederive from the seed alone; per-service identities are unlinkable.
        let dek_stable = seed.dek("financial", "inv-1") == seed.dek("financial", "inv-1");
        let unlinkable = seed.service_identity("a").id != seed.service_identity("b").id;
        // Passwordless login: a Merkle-OTS revelation bundle verifies; the verifier
        // stores only the public Merkle root, and a replayed bundle is rejected.
        let cal = CryptoLayer::with_defaults();
        let ident = seed.service_identity("app");
        let mut account = Account::register(&cal, "lamport-pq", "app", &ident).unwrap();
        let chal = login_challenge("app", b"nonce");
        let resp = AuthResponse::create(&cal, "lamport-pq", &ident, 0, &chal).unwrap();
        let login = account.verify_login(&cal, &chal, &resp)   // honest login on leaf 0 succeeds
            && !account.verify_login(&cal, &chal, &resp);       // replay of the spent leaf is rejected
        // Recovery: M-of-N quorum after a veto window, no escrow.
        let mut ent = [0u8; 32];
        if let Some(s) = crate::entropy::conditioned_seed() {
            ent = s;
        }
        let shares = split(b"master-secret!!!", 3, 5, &ent).unwrap();
        let rec = Recovery::open(b"recover", 1000, 500);
        let recovery = rec.complete(&shares[0..3], 1200).is_none()
            && rec.complete(&shares[0..3], 1600).is_some();
        check(
            "identity: HD keys + unlinkable service ids + passwordless login + M-of-N recovery",
            dek_stable && unlinkable && login && recovery,
            &mut report,
        );
    }

    // ---- Phase H: hardware root of trust (measured boot + sealing + attestation) ----
    {
        use dominion_core::rot::{verify_attestation, Backend, RootOfTrust, Tier};
        let mut rot = RootOfTrust::new(Backend::Tpm20, b"device-seed");
        rot.extend(0, b"firmware");
        rot.extend(1, b"kernel");
        let key = rot.enrollment_key();
        // Attestation binds PCRs + tier + freshness nonce.
        let att = rot.quote(b"verifier-nonce");
        let attested = verify_attestation(&att, &key, b"verifier-nonce", Tier::Hardware)
            && !verify_attestation(&att, &key, b"stale-nonce", Tier::Hardware);
        // Sealing binds a secret to the boot state.
        let blob = rot.seal(b"drng seed");
        let sealed_ok = rot.unseal(&blob).as_deref() == Some(b"drng seed".as_ref());
        // System-domain confidentiality: internals are unreadable by a user subject.
        use dominion_core::confidential::{Classification, Confidentiality, Domain, ReadResult, Subject};
        let mut cf = Confidentiality::new();
        cf.put(1, Domain::System, Classification::SystemPrivate, b"kernel internals");
        let user = Subject::user(7, Classification::Secret);
        let hidden = cf.read(&user, 1) == ReadResult::Denied
            && matches!(cf.read(&Subject::system(), 1), ReadResult::Granted(_));
        check(
            "root of trust: measured boot + platform-sealed secret + attestation + system-domain",
            attested && sealed_ok && hidden,
            &mut report,
        );
    }

    // ---- Phase I: crash-consistent journal on the real virtio-blk disk ----
    {
        use dominion_core::hash::Hash256;
        use dominion_core::journal::Journal;
        let key = b"root-signing-key";
        let ok = crate::block::with_block_device(|dev, _| {
            if Journal::format(dev, key).is_err() {
                return false;
            }
            // CoW commits flip the root atomically; load returns the newest signed root.
            let g1 = Journal::commit(dev, Hash256::of(b"state-1"), key);
            let g2 = Journal::commit(dev, Hash256::of(b"state-2"), key);
            if !(g1 == Ok(1) && g2 == Ok(2)) {
                return false;
            }
            match Journal::load(dev, key) {
                Ok(Some(rec)) => rec.root == Hash256::of(b"state-2") && rec.generation == 2,
                _ => false,
            }
        });
        check("journal: CoW commit + atomic root flip on real virtio-blk disk", ok, &mut report);
    }

    // ---- Phase I: zero-plaintext backup + fleet sync + memory reclaim ----
    {
        use dominion_core::backup::{BackupStore, FleetIndex};
        use dominion_core::hash::Hash256;
        use dominion_core::hlc::Timestamp;
        use dominion_core::pressure::{Pressure, WorkingSet};
        // Backup to an untrusted store: ciphertext-only, dedup, verified restore.
        let key = b"backup-key";
        let pt = alloc::vec![b'A'; 64];
        let a = (Hash256::of(&pt), pt); // content id = hash of the plaintext
        let mut store = BackupStore::new();
        let wrote = store.backup(core::slice::from_ref(&a), key) == 1;
        let zero_plain = store.raw(a.0).map(|c| c != a.1.as_slice()).unwrap_or(false);
        let restored = store.restore(a.0, key).as_deref() == Some(a.1.as_slice())
            && store.restore(a.0, b"thief").is_none();
        let dedup = store.backup(core::slice::from_ref(&a), key) == 0;
        // Fleet index merges by HLC and converges.
        let mut x = FleetIndex::new();
        let mut y = FleetIndex::new();
        x.put("/d", Hash256::of(b"r1"), Timestamp { wall: 1, logical: 0 });
        y.put("/d", Hash256::of(b"r2"), Timestamp { wall: 2, logical: 0 });
        x.merge(&y);
        let converged = x.get("/d") == Some(Hash256::of(b"r2"));
        // Memory pressure: clean LRU object evicted under quota, re-fetchable by hash.
        let mut ws = WorkingSet::new(80);
        ws.admit(Hash256::of(b"1"), 40, false);
        ws.admit(Hash256::of(b"2"), 40, false);
        let evicted = ws.admit(Hash256::of(b"3"), 40, false);
        let reclaim = evicted == alloc::vec![Hash256::of(b"1")] && ws.used() <= 80;
        let signal = {
            let mut w = WorkingSet::new(100);
            w.admit(Hash256::of(b"a"), 95, false);
            w.pressure() == Pressure::High
        };
        check(
            "persistence: untrusted backup (zero-plaintext+dedup) + fleet merge + OOM reclaim",
            wrote && zero_plain && restored && dedup && converged && reclaim && signal,
            &mut report,
        );
    }

    // ---- Phase J: tiered capability enforcement + arch profile ----
    {
        use dominion_core::arch;
        use dominion_core::cheri::CapabilityTags;
        use dominion_core::enforcement::{EnforcementLayer, HardwareFeatures, Tier, TierPolicy};
        use dominion_core::firewall::Domain;
        // Boot-time backend selection: this machine has no tags → Tier 0 software.
        let layer = EnforcementLayer::select(HardwareFeatures::probe(), [9u8; 32]);
        let tier0 = layer.tier() == Tier::Software && layer.meets(Tier::Software);
        // A high-assurance domain requiring hardware tagging is refused on Tier 0,
        // admitted once a stronger backend is selected (degrades gracefully).
        let mut policy = TierPolicy::new();
        policy.require(Domain::Financial, Tier::MemoryTagging);
        let refused = !policy.admits(Domain::Financial, &layer);
        let stronger = EnforcementLayer::select(
            HardwareFeatures { cheri_tags: true, ..Default::default() },
            [9u8; 32],
        );
        let admitted = policy.admits(Domain::Financial, &stronger)
            && stronger.tier() == Tier::Cheri
            && stronger.tags().hardware_backed();
        // Arch profile reports the target this kernel was built for (x86_64 here).
        let profile = arch::current();
        let arch_ok = matches!(profile.arch, arch::Arch::X86_64) && profile.pointer_bits == 64;
        check(
            "enforcement: boot-time tier selection + per-domain min-tier policy + arch profile",
            tier0 && refused && admitted && arch_ok,
            &mut report,
        );
    }

    // ---- Phase K: verified secure boot chain + memory encryption at rest ----
    {
        use dominion_core::crypto::CryptoLayer;
        use dominion_core::memenc::{DataKind, DomainMemory, MemEncryption, MemFeatures, MemTier};
        use dominion_core::secureboot::{image_matches, BootChain, StageSigner};
        use dominion_core::hash::Hash256;
        let cal = CryptoLayer::with_defaults();
        // Build a signed boot chain: anchor → firmware → kernel.
        let (anchor_sk, anchor_pk) = cal.keygen("lamport-pq", b"anchor").unwrap();
        let (fw_sk, fw_pk) = cal.keygen("lamport-pq", b"firmware").unwrap();
        let (_k_sk, k_pk) = cal.keygen("lamport-pq", b"kernel").unwrap();
        let firmware = StageSigner::sign(&cal, "lamport-pq", "firmware", b"FW", &fw_pk, &anchor_sk).unwrap();
        let kernel = StageSigner::sign(&cal, "lamport-pq", "kernel", b"KERNEL", &k_pk, &fw_sk).unwrap();
        let mut chain = BootChain::new(&cal, "lamport-pq", &anchor_pk);
        let booted = chain.load(&firmware).is_ok() && chain.load(&kernel).is_ok();
        // A tampered kernel image is rejected (reproducible-image proof).
        let reproducible = image_matches(b"KERNEL", Hash256::of(b"KERNEL"))
            && !image_matches(b"KERNEL-trojan", Hash256::of(b"KERNEL"));
        // Memory encryption: no hardware here → crown jewels still software-sealed.
        let me = MemEncryption::detect(MemFeatures::default());
        let tiered = me.tier_for(DataKind::CrownJewel) == MemTier::SoftwarePerObject;
        let mut mem = DomainMemory::new([7u8; 32]);
        mem.seal(1, b"identity secret");
        let ram_ct = mem.at_rest(1).map(|c| c != b"identity secret").unwrap_or(false);
        let cap_gated = mem.read(1, Rights::READ).as_deref() == Some(b"identity secret".as_ref())
            && mem.read(1, Rights::WRITE).is_none();
        check(
            "secure boot: signed measured chain + reproducible image + memory-at-rest tiers",
            booted && reproducible && tiered && ram_ct && cap_gated,
            &mut report,
        );
    }

    // ---- Phase L: compatibility conformance suites + 90% release gate ----
    {
        use dominion_core::compat::{detect_format, translate_syscall, Abi, BinaryFormat, HostOp};
        use dominion_core::conformance::run_builtin_suites;
        // Win/Mac/Linux binary formats are recognised; foreign syscalls default-closed.
        let formats = detect_format(&[0x4D, 0x5A, 0, 0]) == BinaryFormat::Pe
            && detect_format(&[0xCF, 0xFA, 0xED, 0xFE]) == BinaryFormat::MachO
            && detect_format(&[0x7F, b'E', b'L', b'F', 0, 0]) == BinaryFormat::Elf;
        let default_closed = translate_syscall(Abi::Win64, 0xFFFF_FFFF) == HostOp::Denied;
        // Run the real conformance corpus and enforce the 90% release gate.
        let creport = run_builtin_suites();
        let gated = creport.categories() >= 4 && creport.meets_gate(900) && creport.overall_milli() == 1000;
        check(
            "compatibility: format detect + default-closed ABI + conformance 90% release gate",
            formats && default_closed && gated,
            &mut report,
        );
    }

    // ---- Stage 6: declarative driver synthesis on metal ----
    {
        use dominion_core::cheri::SoftwareTags;
        use dominion_core::driver::{
            DeviceClass, DeviceSpec, Driver, DriverFault, MmioDevice, RegOp, ResourceClaim, ValueSrc,
        };
        // A mock block device described entirely by data (no driver code).
        struct MockBlk {
            base: u64,
            lba: u64,
            status: u64,
            data: u64,
        }
        impl MmioDevice for MockBlk {
            fn read(&mut self, addr: u64, _w: u8) -> u64 {
                match addr - self.base {
                    0x00 => self.status,
                    0x08 => self.data,
                    _ => 0,
                }
            }
            fn write(&mut self, addr: u64, _w: u8, v: u64) {
                match addr - self.base {
                    0x10 => self.lba = v,
                    0x18 if v == 1 => {
                        self.data = 100 + self.lba;
                        self.status = 1;
                    }
                    _ => {}
                }
            }
        }
        let tags = SoftwareTags::new([9u8; 32]);
        let spec = DeviceSpec::new(
            DeviceClass::Block,
            ResourceClaim { mmio_base: 0x5000, mmio_len: 0x20, irq: 5 },
        )
        .register("STATUS", 0x00, 8)
        .register("DATA", 0x08, 8)
        .register("LBA", 0x10, 8)
        .register("CTRL", 0x18, 8)
        .program(
            "read",
            alloc::vec![
                RegOp::Write { reg: "LBA".into(), value: ValueSrc::Arg(0) },
                RegOp::Write { reg: "CTRL".into(), value: ValueSrc::Imm(1) },
                RegOp::Poll { reg: "STATUS".into(), value: 1, max_spins: 16 },
                RegOp::Read { reg: "DATA".into() },
            ],
        );
        let driver = Driver::bind(spec, &tags).unwrap();
        let mut dev = MockBlk { base: 0x5000, lba: 0, status: 0, data: 0 };
        let read_ok = driver.run("read", &[7], &mut dev, &tags) == Ok(alloc::vec![107]);
        // A spec whose register escapes its window is refused at bind (contained).
        let escaper = DeviceSpec::new(
            DeviceClass::Block,
            ResourceClaim { mmio_base: 0x5000, mmio_len: 0x20, irq: 5 },
        )
        .register("OOB", 0x40, 8)
        .program("go", alloc::vec![RegOp::Read { reg: "OOB".into() }]);
        let contained = Driver::bind(escaper, &tags).err() == Some(DriverFault::MalformedSpec);
        check(
            "driver synthesis: one runtime drives a device from a spec; out-of-window refused",
            read_ok && contained,
            &mut report,
        );
    }

    // ---- Stage 6: safe foreign-driver loading (NDISwrapper/LinuxKPI) on metal ----
    {
        use dominion_core::cheri::SoftwareTags;
        use dominion_core::driver::{DeviceClass, ResourceClaim};
        use dominion_core::foreign::{ForeignAbi, ForeignDriver, ForeignHost, KpiShim, LoadError};
        let tags = SoftwareTags::new([3u8; 32]);
        let host = ForeignHost::new(
            KpiShim::ndis(),
            ResourceClaim { mmio_base: 0x6000, mmio_len: 0x2000, irq: 0 },
        );
        // A downloaded Windows NIC driver, contained to exactly its device window.
        let good = ForeignDriver::new(
            "RtlNic.sys",
            ForeignAbi::WindowsNdis,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x6000, mmio_len: 0x200, irq: 11 },
        )
        .imports(&["NdisAllocateMemory", "NdisMRegisterMiniport"]);
        let loaded = host.load(&good, &tags).unwrap();
        let contained = loaded.window() == (0x6000, 0x200)
            && loaded.may_access(0x6100, 8)
            && !loaded.may_access(0x6200, 8); // one past its window → denied
        // A driver importing an un-shimmed symbol is refused (default-closed).
        let sketchy = ForeignDriver::new(
            "x.sys",
            ForeignAbi::WindowsNdis,
            DeviceClass::Net,
            ResourceClaim { mmio_base: 0x6000, mmio_len: 0x100, irq: 5 },
        )
        .imports(&["ZwOpenFile"]);
        let refused = matches!(host.load(&sketchy, &tags), Err(LoadError::MissingSymbol(_)));
        check(
            "foreign driver: runtime-load NDIS driver capability-contained; un-shimmed import refused",
            contained && refused,
            &mut report,
        );
    }

    // ---- Stage 9: GPU-first UI toolkit (renderer-agnostic) on metal ----
    {
        use dominion_core::toolkit::{
            build_scene, hit_test, layout, select_renderer, Axis, Backend, DrawCmd, Rect, Size, Theme,
            Widget,
        };
        // GPU-first selection with framebuffer fallback.
        let gpu = select_renderer(true).backend() == Backend::Gpu;
        let fb = select_renderer(false).backend() == Backend::Framebuffer;
        // A small dashboard: sidebar + a button bar.
        let ui = Widget::Container {
            id: 0,
            axis: Axis::Row,
            padding: 8,
            size: Size::Flex(1),
            children: alloc::vec![
                Widget::Container { id: 1, axis: Axis::Column, padding: 0, size: Size::Fixed(48), children: alloc::vec![] },
                Widget::Button { id: 2, text: "Run".into(), variant: dominion_core::toolkit::ButtonVariant::Primary, size: Size::Flex(1) },
            ],
        };
        let area = Rect::new(0, 0, 320, 200);
        let scene = build_scene(&ui, &Theme::dark(), area);
        let placements = layout(&ui, area);
        // The button paints with the theme's primary token, and hit-testing routes a
        // click on the right side to the button (id 2), not the container.
        let painted = scene.iter().any(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == Theme::dark().primary));
        let routed = hit_test(&placements, 200, 100) == Some(2);
        check(
            "UI toolkit: GPU-first/fallback select + themed scene + hit-test routing",
            gpu && fb && painted && routed,
            &mut report,
        );
    }

    // ---- Stage 9: toolkit framebuffer rasterisation on metal ----
    {
        use dominion_core::toolkit::{self, raster, ButtonVariant, Rect as TRect, Theme};
        let t = Theme::dark();
        // Build a real shell scene and rasterise it into an in-RAM canvas (the same
        // path the desktop uses against the framebuffer back-buffer).
        let (cw, ch) = (160usize, 80usize);
        let mut canvas = alloc::vec![raster::pack(t.bg); cw * ch];
        let strip = toolkit::row(
            0,
            alloc::vec![
                toolkit::button_variant(1, "OK", ButtonVariant::Primary),
                toolkit::button_variant(2, "No", ButtonVariant::Danger),
            ],
        );
        let scene = toolkit::build_scene(&strip, &t, TRect::new(0, 0, cw as i32, 30));
        let texts = raster::render(&scene, &mut canvas, cw, ch);
        let painted_primary = canvas.iter().any(|&px| px == raster::pack(t.primary));
        let painted_danger = canvas.iter().any(|&px| px == raster::pack(t.danger));
        check(
            "toolkit rasteriser: themed scene fills the framebuffer canvas + returns text",
            painted_primary && painted_danger && texts.len() == 2,
            &mut report,
        );
    }

    // ---- Stage 9: universal editor (Notepad++ ⊕ Vim ⊕ calculator) on metal ----
    {
        use dominion_core::editor::{Editor, Mode};
        let mut e = Editor::new("21 * 2\nplain prose line");
        let results = e.evaluate();
        let calc_ok = results.iter().any(|(r, v)| *r == 0 && v == "42");
        e.key('x'); // Vim Normal-mode delete-char
        let vim_ok = e.lines()[0].starts_with('1');
        e.key('i');
        let mode_ok = e.mode() == Mode::Insert;
        check(
            "universal editor: inline calculator (=42) + Vim modal editing",
            calc_ok && vim_ok && mode_ok,
            &mut report,
        );
    }

    // ---- Stage 9: tabbed Workspace + universal undo on metal ----
    {
        use dominion_core::workspace::{TabContent, Workspace};
        let mut ws = Workspace::new();
        ws.open(TabContent::Browser("dominion://home".into()), "Home");
        ws.open(TabContent::Files("~/proj".into()), "Files");
        let tabs_ok = ws.tab_count() == 3 && ws.active_index() == 2;
        let r1 = dominion_core::hash::Hash256::of(b"e1");
        ws.commit(r1);
        let undo_ok = ws.undo().is_ok() && ws.redo() == Some(r1);
        check(
            "workspace: tabs (object+view) + universal undo timeline",
            tabs_ok && undo_ok,
            &mut report,
        );
    }

    // ---- Stage 9: universal browser + real-Tor enable/disable on metal ----
    {
        use dominion_core::browser::{Browser, Route};
        let mut b = Browser::new();
        let direct = b.resolve("https://example.com").route == Route::Direct;
        b.set_tor(true);
        let blocked = b.resolve("https://example.com").route == Route::Blocked;
        b.tor_bootstrapped(true);
        let via_tor = b.resolve("https://example.com").route == Route::Tor;
        let native_direct = b.resolve("dominion://home").route == Route::Direct;
        check(
            "browser: real-Tor toggle (direct/blocked/tor) on legacy; native stays direct",
            direct && blocked && via_tor && native_direct,
            &mut report,
        );
    }

    // ---- Stage 9: reactive app framework on metal ----
    {
        use dominion_core::appkit::{App, AppCap, AppCapabilities, Store};
        use dominion_core::toolkit::{self, Rect};
        use alloc::collections::BTreeSet;
        fn view(s: &Store, reads: &mut BTreeSet<alloc::string::String>) -> toolkit::Widget {
            let c = s.read("count", reads);
            let mut t = alloc::string::String::from("Count: ");
            t.push_str(&alloc::format!("{c}"));
            toolkit::label(1, &t)
        }
        fn inc(s: &mut Store) {
            let n = s.get("count");
            s.set("count", n + 1);
        }
        let mut app = App::new(Store::new(), view);
        app.on("inc", inc);
        app.render(&toolkit::Theme::dark(), Rect::new(0, 0, 200, 40));
        app.dispatch("inc");
        let reactive_ok = app.store.get("count") == 1 && app.needs_rerender();
        let mut caps = AppCapabilities::new();
        let cap_ok = caps.require(AppCap::Net).is_err() && {
            caps.grant(AppCap::Net);
            caps.require(AppCap::Net).is_ok()
        };
        check(
            "appkit: reactive store + event dispatch + default-closed app capabilities",
            reactive_ok && cap_ok,
            &mut report,
        );
    }

    // ---- Stage 9: shell (dashboard + dock + capability panel + command routing) ----
    {
        use dominion_core::shell::{CapabilityPanel, DockItem, Shell};
        use dominion_core::toolkit::{DrawCmd, Rect, Theme};
        use dominion_core::workspace::TabContent;
        let shell = Shell::new();
        let area = Rect::new(0, 0, 640, 400);
        let scene = shell.view(&Theme::dark(), area);
        let dock_ok = scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Browser"));
        // A dock click selects a surface; a command routes to the right tab.
        let click_ok = shell.dock_hit(&Theme::dark(), area, 30, 400 - 22) == Some(DockItem::Workspace);
        let route_ok = matches!(Shell::command_to_tab("https://example.com"), TabContent::Browser(_));
        // The capability panel lists + toggles a grant.
        let mut panel = CapabilityPanel::new();
        panel.add("Net", true);
        let cap_ok = panel.toggle("Net") == Some(false);
        check(
            "shell: dashboard + dock select + command→tab routing + capability panel",
            dock_ok && click_ok && route_ok && cap_ok,
            &mut report,
        );
    }

    // ---- Stage 9: frame scheduling (EDF) + motion on metal ----
    {
        use dominion_core::anim::{FrameOutcome, FrameScheduler, Tween};
        let mut fs = FrameScheduler::new(60);
        let edf_ok = fs.submit(10_000) == FrameOutcome::Presented
            && fs.submit(30_000) == FrameOutcome::Dropped
            && FrameScheduler::shed_animation(1);
        let tween = Tween::new(0, 100, 1000);
        let motion_ok = tween.value(0) == 0 && tween.value(1000) == 100 && tween.value(500) > 50;
        let reduced_ok = Tween::new(0, 100, 1000).reduced_motion(true).value(0) == 100;
        check(
            "animation: EDF present/drop + pressure shed + ease-out tween (reduced-motion aware)",
            edf_ok && motion_ok && reduced_ok,
            &mut report,
        );
    }

    // ---- Stage 9: LIVE shell (Desktop page) render to the real framebuffer ----
    {
        // Bring up the graphics surface (the same path the desktop uses) and render
        // the actual DominionOS shell scene into it (boots on the Desktop page), then
        // read pixels back to confirm the anti-aliased framebuffer backend painted the
        // chrome, the cards, and text.
        if let Some((w, h)) = crate::gfx::init() {
            serial_println!("  [info] framebuffer resolution: {}x{}", w, h);
            let mut os = dominion_core::os::Os::new();
            os.set_size(w as i32, h as i32);
            let theme = dominion_core::toolkit::Theme::dark(); // the shell boots dark
            let pack = |c: dominion_core::toolkit::Color| ((c.r as u32) << 16) | ((c.g as u32) << 8) | c.b as u32;
            let (bg, surface, primary, accent) =
                (pack(theme.bg), pack(theme.surface), pack(theme.primary), pack(theme.accent));
            let scene = os.view(w as i32, h as i32);
            crate::gfx::raster_scene(&scene);
            // Sample the whole back-buffer and confirm the shell actually painted the
            // wallpaper, the chrome/card surfaces, accent content (dock glyphs/cards),
            // AND anti-aliased text (a colour that is none of the flat theme tokens —
            // i.e. a blended glyph edge).
            let (mut saw_bg, mut saw_surface, mut saw_content, mut saw_text) = (false, false, false, false);
            let mut y = 0;
            while y < h {
                let mut x = 0;
                while x < w {
                    if let Some(px) = crate::gfx::back_pixel(x, y) {
                        if px == bg {
                            saw_bg = true;
                        } else if px == surface {
                            saw_surface = true;
                        } else if px == primary || px == accent {
                            saw_content = true;
                        } else if px != bg && px != surface {
                            saw_text = true; // a blended (anti-aliased) pixel
                        }
                    }
                    x += 3;
                }
                y += 3;
            }
            check(
                "live shell renders to the framebuffer (wallpaper + chrome + cards + AA text)",
                saw_bg && saw_surface && saw_content && saw_text,
                &mut report,
            );

            // ---- the taskbar switches apps (Desktop → Explorer) ----
            // Opening Explorer from the Start menu launches it in a window; rendering
            // again must show the Explorer's header text that the Desktop never draws.
            // (Apps are now floating windows; the taskbar lists only *running* windows,
            // so the launcher lives in the Start menu.)
            os.on_pointer(0, 0, false); // settle pointer state
            // Start button (DOCK_H=60): x=8 w=76 → centre (46, h-30). The Start menu pops
            // up above it: rect = (8, h-344, 200, 286); item i centred at y = (h-344)+8+i*30+15.
            // Explorer is ALL_APPS index 6 (Desktop, Files, Browser, Terminal, Editor, IDE,
            // Explorer, …) → y = (h-344) + 8 + 6*30 + 15 = h-141; x = 8+20 = 28.
            let dock_y = h as i32 - 30;
            os.on_pointer(46, dock_y, true); // open Start
            os.on_pointer(46, dock_y, false);
            let menu_y = h as i32 - 141;
            os.on_pointer(28, menu_y, true); // click Explorer
            os.on_pointer(28, menu_y, false);
            let scene2 = os.view(w as i32, h as i32);
            let switched = scene2.iter().any(|c| matches!(c, dominion_core::toolkit::DrawCmd::Text { text, .. } if text == "SYSTEM KNOWLEDGE GRAPH"));
            check("shell opens Explorer in a window from the Start menu", switched, &mut report);

            // ---- damage-rectangle (incremental) raster ----
            // Re-render a *different* scene (light theme via key 'g') but clipped to a
            // small rect over empty desktop background. Pixels inside must update to the
            // new theme; a far corner must stay exactly as it was.
            let mut os2 = dominion_core::os::Os::new();
            os2.set_size(w as i32, h as i32);
            os2.on_key('g'); // → light theme
            let light_bg = pack(dominion_core::toolkit::Theme::light().bg);
            let scene3 = os2.view(w as i32, h as i32);
            // A rect over desktop background: the band just below the top bar (y≈31) and
            // above the first card row (cards start at local y=60 → screen y≈90).
            let (rx, ry, rw, rh) = (500, 48, 44, 34);
            let outside_before = crate::gfx::back_pixel(5, 5); // top bar, far outside
            crate::gfx::raster_scene_clipped(&scene3, (rx, ry, rw, rh));
            let outside_after = crate::gfx::back_pixel(5, 5);
            let mut inside_changed = false;
            let mut yy = ry;
            while yy < ry + rh {
                let mut xx = rx;
                while xx < rx + rw {
                    if crate::gfx::back_pixel(xx as usize, yy as usize) == Some(light_bg) {
                        inside_changed = true;
                    }
                    xx += 2;
                }
                yy += 2;
            }
            check(
                "damage-rect raster repaints only the clip (inside updates, outside untouched)",
                inside_changed && outside_before == outside_after && outside_before != Some(light_bg),
                &mut report,
            );
        } else {
            check("live shell renders to the framebuffer", false, &mut report);
        }
    }

    // ---- UI: global text engine (anywhere-caret + arrows + click + blink) on metal ----
    {
        use dominion_core::text::TextBuffer;
        let mut b = TextBuffer::new("helloworld");
        b.set_caret(0, 5);
        b.insert(' '); // caret placed mid-string, not just at the end
        let place_ok = b.text() == "hello world" && b.caret() == (0, 6);
        // Arrow navigation + click-to-place map to a real caret position.
        b.place_at_pixel(8 * 3 + 2, 4, (0, 0), 8, 16);
        let click_ok = b.caret() == (0, 3);
        b.left();
        let arrow_ok = b.caret() == (0, 2);
        // The caret blinks on the clock (on for [0,500), off for [500,1000)) and then
        // holds solid right after activity.
        let blink_ok = b.caret_visible(0) && !b.caret_visible(500) && {
            b.touch(600);
            b.caret_visible(700) // held solid across the off-half
        };
        check(
            "text engine: caret places anywhere + arrows/click + blinking caret",
            place_ok && click_ok && arrow_ok && blink_ok,
            &mut report,
        );
    }

    // ---- UI: embedded terminal (REPL + history + render) on metal ----
    {
        use dominion_core::terminal::{LineKind, Terminal};
        let mut t = Terminal::new();
        for c in "2 + 2".chars() {
            t.input_key(c);
        }
        t.input_key('\n'); // submit → evaluated as Dominion
        let repl_ok = t.lines().iter().any(|l| l.kind == LineKind::Output && l.text.contains('4'));
        // A driver (the IDE) can stream output in.
        t.println("→ streamed");
        let driver_ok = t.lines().iter().any(|l| l.text.contains("streamed"));
        // It renders a real scene (surface + prompt + caret).
        t.tick(0);
        let scene = t.view(&dominion_core::toolkit::Theme::dark(), dominion_core::toolkit::Rect::new(0, 0, 400, 200));
        let render_ok = scene.iter().any(|c| matches!(c, dominion_core::toolkit::DrawCmd::Text { text, .. } if text.starts_with('›')));
        check(
            "terminal: live Dominion REPL + driver output + renders prompt/caret",
            repl_ok && driver_ok && render_ok,
            &mut report,
        );
    }

    // ---- UI: composable widget board (move/resize/remove + library) on metal ----
    {
        use dominion_core::compose::{Board, Library, WidgetKind};
        use dominion_core::toolkit::Rect;
        let mut board = Board::new();
        board.set_area(Rect::new(0, 0, 1000, 700));
        board.toggle_lock(); // enter edit mode
        board.add(WidgetKind::Chart);
        board.add(WidgetKind::Terminal);
        let add_ok = board.panels().len() == 2;
        // Drag the first panel's title bar to move it.
        let p0 = board.panels()[0].clone();
        let start_x = p0.rect.x;
        board.on_pointer(p0.rect.x + 5, p0.rect.y + 5, true);
        board.on_pointer(p0.rect.x + 5 + 80, p0.rect.y + 5 + 40, true);
        board.on_pointer(p0.rect.x + 5 + 80, p0.rect.y + 5 + 40, false);
        let move_ok = board.panels()[0].rect.x != start_x;
        // Upload → library publish → install round-trips the layout.
        let mut lib = Library::new();
        lib.publish("default", &board);
        let installed = lib.install("default").map(|b| b.panels().len()).unwrap_or(0);
        let lib_ok = installed == 2;
        check(
            "composable UI: add/move widgets + content-addressed library upload/download",
            add_ok && move_ok && lib_ok,
            &mut report,
        );
    }

    // ---- Phase M: red-team hardening (rollback safety + storm guard + anonymity) ----
    // Finding A — distributed rollback never resurrects a contained write, and
    // pinned roots are never rolled back, under loss/reorder/partition.
    {
        use dominion_core::consistency::{rollback_safety_scenario, FencedReplica};
        use dominion_core::hlc::Timestamp;
        let (converged, resurrected) = rollback_safety_scenario(0xC0FFEE, 5, 16);
        let mut r = FencedReplica::new();
        r.pin("identity-root");
        r.put("identity-root", b"master", Timestamp { wall: 1, logical: 0 }).unwrap();
        let pinned_held = r.rollback_key("identity-root", b"attacker", 9).is_err()
            && r.get("identity-root") == Some(b"master".as_ref());
        check(
            "consistency: fenced rollback converges + no resurrection + pinned roots immutable",
            converged && !resurrected && pinned_held,
            &mut report,
        );
    }

    // Finding B — a crash-loop trips the breaker (quarantine) instead of thrashing,
    // and isolated failures back off exponentially.
    {
        use dominion_core::supervisor::{RecoveryDecision, RecoveryPolicy, Supervisor};
        let mut s = Supervisor::new(RecoveryPolicy::standard());
        let backoff = matches!(s.on_failure(1, 0), RecoveryDecision::Restart { after: 8 });
        let mut tripped = false;
        for t in 1..6 {
            if let RecoveryDecision::Quarantine { .. } = s.on_failure(1, t) {
                tripped = true;
            }
        }
        check(
            "supervisor: exponential backoff + crash-loop trips breaker (no rollback storm)",
            backoff && tripped && s.is_quarantined(1, 10),
            &mut report,
        );
    }

    // Finding C — anonymous per-context pseudonyms are unlinkable across contexts,
    // stable within a context (scoped nullifier), and the system layer forbids an
    // anonymous transaction from carrying a global correlator.
    // Gated: requires demo-crypto feature (31-bit illustrative group).
    #[cfg(feature = "demo-crypto")]
    {
        use dominion_core::anon::{AnonIdentity, NullifierSet, Transaction};
        use dominion_core::zk::SchnorrParams;
        let params = SchnorrParams::new_demo_insecure();
        let me = AnonIdentity::from_secret(params, b"metal-anon-secret");
        let a = me.pseudonym(b"poll:budget");
        let b = me.pseudonym(b"forum:general");
        let unlinkable = a.value != b.value && a.verify(&params, b"poll:budget");
        let scoped = me.pseudonym(b"poll:budget").nullifier() == a.nullifier();
        let mut spent = NullifierSet::new();
        let double_blocked = spent.spend(a.nullifier()) && !spent.spend(a.nullifier());
        let no_leak = Transaction::anonymous(a, None, b"x").map(|t| !t.exposes_global_correlator()).unwrap_or(false);
        check(
            "anon: unlinkable pseudonyms + scoped nullifier + double-action block + no correlator leak",
            unlinkable && scoped && double_blocked && no_leak,
            &mut report,
        );
    }

    // ---- ZK system service: capability proofs + retrievability audit, on metal ----
    // Proves the higher-level ZK applications work end-to-end on the booted machine:
    // a content-addressed, capability-gated proof that held authority satisfies a
    // policy (without revealing the rest of the authority graph), and a proof-of-
    // retrievability audit that catches a store which dropped a block.
    // Gated: ZkService::new() calls SchnorrParams::new_demo_insecure() internally.
    #[cfg(feature = "demo-crypto")]
    {
        use dominion_core::capability::{Capability, Rights};
        use dominion_core::zkservice::{CapabilityWallet, Retrievability, ZkService};
        let mut svc = ZkService::new([0x5Au8; 32]);
        let verify_cap = Capability::mint(0, u64::MAX, Rights::READ);

        // ZK capability proof: hold three caps, prove one admits WRITE.
        let caps = [
            Capability::mint(0x1000, 16, Rights::READ),
            Capability::mint(0x2000, 16, Rights::READ.union(Rights::WRITE)),
            Capability::mint(0x3000, 16, Rights::EXECUTE),
        ];
        let wallet = CapabilityWallet::commit(&caps);
        let root = wallet.root();
        let cap_proof = wallet.prove_policy(&mut svc, Rights::WRITE, 0xC0FFEE).unwrap();
        let cap_ok = cap_proof.verify(&svc, &verify_cap, &root, Rights::WRITE) == Ok(true)
            // a policy nobody satisfies cannot be proven
            && wallet.prove_policy(&mut svc, Rights::SEAL, 0xC0FFEE).is_none();

        // Proof of retrievability: honest store passes, dropped/corrupt block fails.
        let blocks: Vec<Vec<u8>> = (0..8u8).map(|i| alloc::vec![i; 64]).collect();
        let por = Retrievability::commit(blocks);
        let proot = por.root();
        let idx = por.challenge_indices(b"metal-audit", 4);
        let resp = por.answer(&mut svc, &idx);
        let honest_ok =
            Retrievability::audit(&svc, &verify_cap, &proot, &idx, &resp) == Ok(true);
        let mut tampered = resp.clone();
        tampered[0].block[0] ^= 0xFF;
        let tamper_caught =
            Retrievability::audit(&svc, &verify_cap, &proot, &idx, &tampered) == Ok(false);

        check(
            "zkservice: ZK capability proof + capability-gated verify + retrievability audit",
            cap_ok && honest_ok && tamper_caught,
            &mut report,
        );
    }

    // ---- advanced ZK: verifiable computation + confidential tx + search integrity ----
    // The three constructions that need real circuit/range-proof machinery: a sum-check
    // proof of an aggregate (verifiable computation), a confidential transfer (Pedersen
    // + range + balance), and an authenticated search result — all on the booted machine.
    // Gated: Pedersen::new() and ZkService::new() call SchnorrParams::new_demo_insecure().
    #[cfg(feature = "demo-crypto")]
    {
        use dominion_core::ctx::{ConfidentialTx, Pedersen};
        use dominion_core::vcompute::{prove as sc_prove, verify as sc_verify, MultilinearTable};
        use dominion_core::zkservice::{AuthenticatedIndex, ZkService};

        // 1) Verifiable computation: prove Σ over a private 8-entry table; tamper caught.
        let table = MultilinearTable::new(alloc::vec![3, 1, 4, 1, 5, 9, 2, 6]).unwrap();
        let proof = sc_prove(&table);
        let mut lied = proof.clone();
        lied.claimed_sum = lied.claimed_sum.wrapping_add(1);
        let vc_ok = sc_verify(&table, &proof).is_ok() && sc_verify(&table, &lied).is_err();

        // 2) Confidential transaction: balanced transfer verifies, hidden amounts.
        let ped = Pedersen::new();
        let tx = ConfidentialTx::build(&ped, &[100, 50], &[120, 25], 5, 16, b"metal-tx").unwrap();
        let tx_ok = tx.verify(&ped, 16)
            && ConfidentialTx::build(&ped, &[100], &[101], 0, 16, b"x").is_err();

        // 3) Search-result integrity: dropped hit is caught.
        let mut svc = ZkService::new([0x33u8; 32]);
        let entries = alloc::vec![(
            b"k".to_vec(),
            alloc::vec![b"d1".to_vec(), b"d2".to_vec(), b"d3".to_vec()],
        )];
        let index = AuthenticatedIndex::build(&entries);
        let iroot = index.root();
        let res = index.query(&mut svc, b"k").unwrap();
        let vcap = Capability::mint(0, u64::MAX, Rights::READ);
        let mut dropped = res.clone();
        dropped.posting.pop();
        let search_ok = res.verify(&svc, &vcap, &iroot) == Ok(true)
            && dropped.verify(&svc, &vcap, &iroot) == Ok(false);

        check(
            "advzk: sum-check verifiable compute + confidential tx (range+balance) + search integrity",
            vc_ok && tx_ok && search_ok,
            &mut report,
        );
    }

    // ---- goal build-out: tensions, drivers, auth, power, i18n, net, packaging ----
    // Smoke-tests the subsystems added to drive the checklist to completion, on metal.
    {
        use dominion_core::arch::Arch;
        use dominion_core::driver::ResourceClaim;
        use dominion_core::drivergen::{class_template, HwClass};
        use dominion_core::hash::Hash256;
        use dominion_core::i18n::{shape, LocaleFormat};
        use dominion_core::legacynet::{decapsulate, encapsulate};
        use dominion_core::packaging::{FfiSurface, UniversalBinary};
        use dominion_core::pmgmt::{DvfsGovernor, ThermalGovernor, ThrottleAction};
        use dominion_core::tensions;
        use dominion_core::webauth::Passkey;

        // Design tensions all hold; a driver class template binds.
        let tensions_ok = tensions::all_hold();
        let claim = ResourceClaim { mmio_base: 0x4000, mmio_len: 0x100, irq: 11 };
        let drv_ok = class_template(HwClass::Nvme, claim).is_well_formed();

        // WebAuthn passkey: register + assert verifies.
        let (mut pk, mut cred) = Passkey::create("example.com", b"u", b"dev", 3);
        let sig = pk.assert(b"challenge").unwrap();
        let webauthn_ok = cred.verify_assertion("example.com", b"challenge", &sig).is_ok();

        // Power: DVFS picks low OPP at light load; thermal escalates to emergency.
        let pwr_ok = DvfsGovernor::default_ladder().select(100).freq_mhz == 800
            && ThermalGovernor::default_trips().action(105) == ThrottleAction::Emergency;

        // i18n: locale currency formatting + bidi reordering produce output.
        let i18n_ok = LocaleFormat::de_de().format_currency(123456) == "1.234,56 €"
            && !shape("abc\u{05D0}\u{05D1}\u{05D2}", false).is_empty();

        // Networking: DominionLink rides over UDP and round-trips.
        let id = Hash256::of(b"id");
        let dg = encapsulate(40000, &id, b"payload");
        let net_ok = decapsulate(&dg[8..]).map(|(i, p)| i == id && p == b"payload").unwrap_or(false);

        // Packaging: fat binary selects an arch slice; FFI is default-closed.
        let mut fat = UniversalBinary::new();
        fat.add_slice(Arch::X86_64, b"code");
        let pkg_ok = fat.select(Arch::X86_64).is_some() && FfiSurface::new().call("x").is_err();

        check(
            "goal build-out: tensions + driver-synth + webauthn + power + i18n + net + packaging",
            tensions_ok && drv_ok && webauthn_ok && pwr_ok && i18n_ok && net_ok && pkg_ok,
            &mut report,
        );
    }

    // ---- polyglot runtime: real multi-function programs with packages, 7 languages ----
    {
        use dominion_core::polyglot::{self, Language, Value as PValue};
        let mut langs_ok = true;
        for lang in Language::all() {
            // The demo: imports stats+mathx, multi-function, returns exactly 18.0.
            let demo_ok = matches!(
                polyglot::run(polyglot::demo_program(lang), lang),
                Ok(ref r) if r.value.approx(polyglot::DEMO_EXPECTED, 1e-9)
            );
            // The benchmark: a gcd-folding loop over a library call; identical checksum.
            let bench_ok = matches!(
                polyglot::run(polyglot::bench_program(lang), lang),
                Ok(ref r) if r.value == PValue::Int(polyglot::BENCH_EXPECTED)
            );
            if !demo_ok || !bench_ok {
                langs_ok = false;
            }
        }
        // Default-closed: a package function used without importing its package is refused.
        let default_closed = polyglot::run("fn run() { return gcd(48, 36); }", Language::Rust).is_err();
        check(
            "polyglot: Python/Rust/C++/C#/JS/TS/Java run multi-function programs with packages to identical results",
            langs_ok && default_closed,
            &mut report,
        );
    }

    // ---- foreign drivers: load AND use a real Windows PE (.sys) and Linux ELF (.ko) ----
    {
        use dominion_core::cheri::SoftwareTags;
        use dominion_core::driver::{DeviceClass, DeviceSpec, MmioDevice, RegOp, ResourceClaim, ValueSrc};
        use dominion_core::foreign::{build_elf_ko, build_pe_sys, ForeignAbi, ForeignBinary, ForeignHost, KpiShim};

        // A tiny NIC: writing the doorbell "sends" the TX length (into the outbox)
        // and raises STATUS — an observable effect the borrowed driver produces.
        struct Nic {
            base: u64,
            tx: u64,
            outbox: u64,
            status: u64,
        }
        impl MmioDevice for Nic {
            fn read(&mut self, addr: u64, _w: u8) -> u64 {
                if addr - self.base == 0x0c {
                    self.status
                } else {
                    0
                }
            }
            fn write(&mut self, addr: u64, _w: u8, v: u64) {
                match addr - self.base {
                    0x04 => self.tx = v,
                    0x08 if v == 1 => {
                        self.outbox = self.tx;
                        self.status = 1;
                    }
                    _ => {}
                }
            }
        }

        let spec = DeviceSpec::new(DeviceClass::Net, ResourceClaim { mmio_base: 0x3000, mmio_len: 0x14, irq: 11 })
            .register("TXLEN", 0x04, 4)
            .register("TXDB", 0x08, 4)
            .register("STATUS", 0x0c, 4)
            .program(
                "send",
                alloc::vec![
                    RegOp::Write { reg: String::from("TXLEN"), value: ValueSrc::Arg(0) },
                    RegOp::Write { reg: String::from("TXDB"), value: ValueSrc::Imm(1) },
                    RegOp::Read { reg: String::from("STATUS") },
                ],
            );
        let tags = SoftwareTags::new([3u8; 32]);
        let env = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };

        // Parse + admit + confine + bind a Windows PE driver and a Linux ELF driver.
        let win = ForeignHost::new(KpiShim::ndis(), env).load_binary(
            &ForeignBinary::new(
                "nic.sys",
                ForeignAbi::WindowsNdis,
                build_pe_sys(&["NdisAllocateMemory", "NdisMRegisterMiniport"], &spec),
            ),
            &tags,
        );
        let lin = ForeignHost::new(KpiShim::linuxkpi(), env).load_binary(
            &ForeignBinary::new("nic.ko", ForeignAbi::LinuxKpi, build_elf_ko(&["kmalloc", "netif_rx"], &spec)),
            &tags,
        );

        let mut both_used = false;
        if let (Ok(w), Ok(l)) = (win, lin) {
            let mut da = Nic { base: 0x3000, tx: 0, outbox: 0, status: 0 };
            let mut db = Nic { base: 0x3000, tx: 0, outbox: 0, status: 0 };
            let ra = w.run("send", &[42], &mut da, &tags);
            let rb = l.run("send", &[42], &mut db, &tags);
            both_used = ra == Ok(alloc::vec![1]) && rb == Ok(alloc::vec![1]) && da.outbox == 42 && db.outbox == 42;
        }
        check(
            "foreign drivers: load + use a Windows PE (.sys) and a Linux ELF (.ko) NIC driver (frame sent through both)",
            both_used,
            &mut report,
        );
    }

    // ---- heap liveness ----
    // ---- machine learning: train + infer on-metal, device-agnostic ----
    {
        use dominion_core::ml::{
            self, dequantize, qmatmul, quantize, recommend_device, Device, Mlp,
        };

        // A real gradient-descent training run: an MLP learns XOR on the booted
        // machine. Convergence here proves autodiff + the optimizer work on-metal.
        let (model, loss) = ml::train_xor(8, 1500);
        check("ml: MLP trains XOR (loss < 0.02)", loss < 0.02, &mut report);

        // Inference reproduces the XOR truth table.
        let (x, _) = ml::xor_dataset();
        let preds: Vec<bool> = model.forward(&x).unwrap().data().iter().map(|&v| v > 0.5).collect();
        check(
            "ml: inference matches XOR truth table",
            preds == [false, true, true, false],
            &mut report,
        );

        // The same model serialises and reloads byte-identically (content-addressable).
        let restored = Mlp::from_bytes(&model.to_bytes()).unwrap();
        check("ml: model serialization round-trips", restored == model, &mut report);

        // Every device computes the identical result (hardware is an accelerator,
        // never a requirement); the cost model still routes small→CPU, large→TPU.
        let a = dominion_core::datatypes::Tensor::new(
            alloc::vec![2, 2],
            alloc::vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let reference = a.matmul(&a).unwrap();
        let all_same = Device::Gpu.matmul(&a, &a).unwrap() == reference
            && Device::Npu.matmul(&a, &a).unwrap() == reference
            && Device::Tpu.matmul(&a, &a).unwrap() == reference;
        check("ml: CPU/GPU/NPU/TPU agree bit-for-bit", all_same, &mut report);
        check(
            "ml: placement picks CPU (tiny) and TPU (large)",
            recommend_device(ml::matmul_flops(2, 2, 2)) == Device::Cpu
                && recommend_device(ml::matmul_flops(512, 512, 512)) == Device::Tpu,
            &mut report,
        );

        // The int8 (NPU) quantized matmul approximates the float result.
        let q = qmatmul(&quantize(&a), &quantize(&a)).unwrap();
        let close = q.data().iter().zip(reference.data()).all(|(x, y)| {
            let d = x - y;
            (if d < 0.0 { -d } else { d }) < 0.2
        });
        check("ml: int8 quantized matmul ≈ float matmul", close, &mut report);
        let _ = dequantize(&quantize(&a));
    }

    check(
        "kernel heap allocation (Vec of 5000)",
        {
            let v: Vec<u64> = (0..5000).collect();
            v.iter().sum::<u64>() == (0..5000u64).sum()
        },
        &mut report,
    );

    (pass, fail)
}

/// Headless entry point: run the battery, report over serial, and exit QEMU with
/// the success/failure code the test harness checks.
pub fn run_and_exit(phys_offset: u64) -> ! {
    serial_println!("\n========== DominionOS bare-metal selftest ==========");
    let (pass, fail) = run(phys_offset, |name, ok| {
        serial_println!("  [{}] {}", if ok { "PASS" } else { "FAIL" }, name);
    });
    serial_println!("==================================================");
    serial_println!("  result: {} passed, {} failed", pass, fail);
    if fail == 0 {
        serial_println!("  ALL BARE-METAL TESTS PASSED");
        exit_qemu(QemuExitCode::Success);
    } else {
        serial_println!("  BARE-METAL TESTS FAILED");
        exit_qemu(QemuExitCode::Failed);
    }
    crate::hlt_loop();
}

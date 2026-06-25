//! Loading AND using real PE/ELF driver binaries: parse the container, admit it
//! through the default-closed shim, confine it to a capability over exactly its
//! device window, and then actually drive a device with it.

use super::*;
use crate::cheri::SoftwareTags;
use crate::driver::{DeviceClass, DeviceSpec, DriverFault, MmioDevice, RegOp, ResourceClaim, ValueSrc};
use alloc::vec;

/// A mock NIC at `base`: a control/enable register, a TX length + doorbell, a
/// status bit, and an RX length. Ringing the doorbell "sends a frame" (records the
/// TX length in the outbox and raises STATUS) — an observable device effect.
struct MockNic {
    base: u64,
    enabled: u64,
    txlen: u64,
    status: u64,
    outbox: u64,
    rxlen: u64,
}
impl MockNic {
    fn new(base: u64) -> MockNic {
        MockNic { base, enabled: 0, txlen: 0, status: 0, outbox: 0, rxlen: 64 }
    }
}
impl MmioDevice for MockNic {
    fn read(&mut self, addr: u64, _w: u8) -> u64 {
        match addr - self.base {
            0x0c => self.status,
            0x10 => self.rxlen,
            _ => 0,
        }
    }
    fn write(&mut self, addr: u64, _w: u8, value: u64) {
        match addr - self.base {
            0x00 => self.enabled = value,
            0x04 => self.txlen = value,
            0x08 if value == 1 => {
                self.outbox = self.txlen; // a frame leaves the NIC
                self.status = 1;
            }
            _ => {}
        }
    }
}

/// The canonical NIC driver spec a borrowed `.sys`/`.ko` carries in its `.drv` section.
fn nic_spec(base: u64) -> DeviceSpec {
    DeviceSpec::new(DeviceClass::Net, ResourceClaim { mmio_base: base, mmio_len: 0x14, irq: 11 })
        .register("CTRL", 0x00, 4)
        .register("TXLEN", 0x04, 4)
        .register("TXDB", 0x08, 4)
        .register("STATUS", 0x0c, 4)
        .register("RXLEN", 0x10, 4)
        .program("init", vec![RegOp::Write { reg: "CTRL".into(), value: ValueSrc::Imm(1) }])
        .program(
            "send",
            vec![
                RegOp::Write { reg: "TXLEN".into(), value: ValueSrc::Arg(0) },
                RegOp::Write { reg: "TXDB".into(), value: ValueSrc::Imm(1) },
                RegOp::Read { reg: "STATUS".into() },
            ],
        )
        .program("recv", vec![RegOp::Read { reg: "RXLEN".into() }])
}

const NDIS_IMPORTS: &[&str] =
    &["NdisAllocateMemory", "NdisMRegisterMiniport", "NdisMRegisterInterrupt", "NdisMSendNetBufferListsComplete"];
const LINUX_IMPORTS: &[&str] = &["kmalloc", "ioremap", "request_irq", "netif_rx"];

#[test]
fn spec_round_trips_through_the_drv_section() {
    let spec = nic_spec(0x3000);
    let encoded = encode_spec(&spec);
    let decoded = decode_spec(&encoded).unwrap();
    assert_eq!(spec, decoded);
}

#[test]
fn loads_and_uses_a_windows_pe_sys_driver() {
    let envelope = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
    let host = ForeignHost::new(KpiShim::ndis(), envelope);
    let tags = SoftwareTags::new([9u8; 32]);

    // A downloaded Windows NIC driver as a real PE container.
    let bytes = build_pe_sys(NDIS_IMPORTS, &nic_spec(0x3000));
    let bin = ForeignBinary::new("RtNic.sys", ForeignAbi::WindowsNdis, bytes);
    let loaded = host.load_binary(&bin, &tags).expect("PE driver should load");

    // Confined to exactly its claimed window.
    assert_eq!(loaded.window(), (0x3000, 0x14));
    assert_eq!(loaded.class(), DeviceClass::Net);

    // USE it: bring the NIC up, send a 42-byte frame, receive.
    let mut nic = MockNic::new(0x3000);
    loaded.run("init", &[], &mut nic, &tags).unwrap();
    assert_eq!(nic.enabled, 1);
    let status = loaded.run("send", &[42], &mut nic, &tags).unwrap();
    assert_eq!(status, vec![1]); // STATUS=ready after the doorbell
    assert_eq!(nic.outbox, 42); // a frame actually left through the borrowed driver
    let rx = loaded.run("recv", &[], &mut nic, &tags).unwrap();
    assert_eq!(rx, vec![64]);
}

#[test]
fn loads_and_uses_a_linux_elf_ko_driver() {
    let envelope = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
    let host = ForeignHost::new(KpiShim::linuxkpi(), envelope);
    let tags = SoftwareTags::new([4u8; 32]);

    // A downloaded Linux NIC module as a real ELF .ko container.
    let bytes = build_elf_ko(LINUX_IMPORTS, &nic_spec(0x5000));
    let bin = ForeignBinary::new("e1000.ko", ForeignAbi::LinuxKpi, bytes);
    let loaded = host.load_binary(&bin, &tags).expect("ELF .ko driver should load");

    assert_eq!(loaded.window(), (0x5000, 0x14));

    // USE it with the identical OS-side API — behaviour comes entirely from its spec.
    let mut nic = MockNic::new(0x5000);
    loaded.run("init", &[], &mut nic, &tags).unwrap();
    let status = loaded.run("send", &[128], &mut nic, &tags).unwrap();
    assert_eq!(status, vec![1]);
    assert_eq!(nic.outbox, 128);
}

#[test]
fn both_driver_kinds_drive_the_same_device_model() {
    // The whole point: a Windows and a Linux driver, loaded from their native binary
    // formats, present the identical capability-confined device service.
    let tags = SoftwareTags::new([7u8; 32]);
    let env = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };

    let win = ForeignHost::new(KpiShim::ndis(), env)
        .load_binary(
            &ForeignBinary::new("w.sys", ForeignAbi::WindowsNdis, build_pe_sys(NDIS_IMPORTS, &nic_spec(0x3000))),
            &tags,
        )
        .unwrap();
    let lin = ForeignHost::new(KpiShim::linuxkpi(), env)
        .load_binary(
            &ForeignBinary::new("l.ko", ForeignAbi::LinuxKpi, build_elf_ko(LINUX_IMPORTS, &nic_spec(0x3000))),
            &tags,
        )
        .unwrap();

    let mut a = MockNic::new(0x3000);
    let mut b = MockNic::new(0x3000);
    win.run("send", &[7], &mut a, &tags).unwrap();
    lin.run("send", &[7], &mut b, &tags).unwrap();
    assert_eq!(a.outbox, b.outbox);
}

#[test]
fn a_borrowed_driver_cannot_escape_its_device() {
    // A malicious .drv whose register escapes the claimed window is rejected at bind
    // time — it can never be created, let alone run out of bounds.
    let bad = DeviceSpec::new(DeviceClass::Net, ResourceClaim { mmio_base: 0x3000, mmio_len: 0x14, irq: 11 })
        .register("ESCAPE", 0x40, 8) // past the 0x14 window
        .program("pwn", vec![RegOp::Write { reg: "ESCAPE".into(), value: ValueSrc::Imm(1) }]);
    let tags = SoftwareTags::new([1u8; 32]);
    let env = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
    let host = ForeignHost::new(KpiShim::ndis(), env);
    let bin = ForeignBinary::new("evil.sys", ForeignAbi::WindowsNdis, build_pe_sys(NDIS_IMPORTS, &bad));
    assert_eq!(host.load_binary(&bin, &tags).err(), Some(LoadError::BadImage));
}

#[test]
fn an_unprovided_import_in_the_binary_is_refused() {
    let tags = SoftwareTags::new([1u8; 32]);
    let env = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
    let host = ForeignHost::new(KpiShim::ndis(), env);
    // The .kpi section asks for a symbol the shim does not expose (raw file I/O).
    let imports = &["NdisAllocateMemory", "ZwOpenFile"];
    let bin = ForeignBinary::new("sneaky.sys", ForeignAbi::WindowsNdis, build_pe_sys(imports, &nic_spec(0x3000)));
    assert_eq!(host.load_binary(&bin, &tags).err(), Some(LoadError::MissingSymbol("ZwOpenFile".into())));
}

#[test]
fn a_corrupt_container_is_rejected() {
    let tags = SoftwareTags::new([1u8; 32]);
    let env = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
    let host = ForeignHost::new(KpiShim::ndis(), env);
    // Truncated bytes are not a valid PE.
    let bin = ForeignBinary::new("trunc.sys", ForeignAbi::WindowsNdis, alloc::vec![0u8; 8]);
    assert_eq!(host.load_binary(&bin, &tags).err(), Some(LoadError::BadImage));
    // An ELF presented as a PE fails the PE magic check.
    let elf = build_elf_ko(LINUX_IMPORTS, &nic_spec(0x3000));
    let bin2 = ForeignBinary::new("mislabeled.sys", ForeignAbi::WindowsNdis, elf);
    assert_eq!(host.load_binary(&bin2, &tags).err(), Some(LoadError::BadImage));
}

#[test]
fn a_tampered_borrowed_driver_fails_closed_at_run_time() {
    let tags = SoftwareTags::new([1u8; 32]);
    let env = ResourceClaim { mmio_base: 0x1000, mmio_len: 0xF000, irq: 0 };
    let host = ForeignHost::new(KpiShim::linuxkpi(), env);
    let loaded = host
        .load_binary(
            &ForeignBinary::new("x.ko", ForeignAbi::LinuxKpi, build_elf_ko(LINUX_IMPORTS, &nic_spec(0x3000))),
            &tags,
        )
        .unwrap();
    // A *different* tag authority cannot validate this driver's capability.
    let other = SoftwareTags::new([2u8; 32]);
    let mut nic = MockNic::new(0x3000);
    assert_eq!(loaded.run("init", &[], &mut nic, &other), Err(DriverFault::CapabilityInvalid));
}

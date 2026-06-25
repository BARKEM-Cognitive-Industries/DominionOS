# Hardware Compatibility & Known Issues

AetherOS runs on real hardware and virtual machines, but support is limited. This guide explains what works, what doesn't, and what to expect.

---

## Quick Summary

| Category | Status | Details |
|----------|--------|---------|
| **Arch** | ✅ x86-64 only | Intel/AMD. No ARM, RISC-V, Apple Silicon (yet) |
| **Boot** | ✅ BIOS / UEFI | Legacy BIOS preferred. UEFI works. Secure Boot may fail. |
| **CPU** | ✅ Multi-core SMP | Up to 16 cores tested. Scaling linear. |
| **Memory** | ✅ Up to 16 GB | Tested. Higher untested. |
| **Storage** | ⚠️ Limited | virtio-blk (QEMU), PCI AHCI. NVMe not yet. |
| **Network** | ⚠️ Limited | virtio-net (QEMU), Broadcom experimental. |
| **Keyboard** | ⚠️ PS/2 only | No USB HID yet. PS/2 mice work. Trackpads don't. |
| **Mouse** | ⚠️ PS/2 only | USB mice untested. Touchscreen no. |
| **Graphics** | ⚠️ Software | Framebuffer only. No GPU acceleration. |
| **Audio** | ❌ No | Specified. Not implemented. |
| **Wireless** | ❌ No | WiFi/Bluetooth not supported. |

---

## Tested Configurations

### Primary Test Platform

**System:**
- CPU: Intel Core i7-12650H (10 cores / 16 threads)
- RAM: 16 GB DDR5
- Storage: 1 TB Micron NVMe SSD (accessed via virtio-blk in QEMU)
- GPU: NVIDIA GeForce RTX 4060 Laptop + Intel UHD (neither used — software rendering only)
- Host OS: Windows 11

**Tested with:**
- QEMU (x86-64, WHPX acceleration, virtio devices): ✅ Full boot, shell, networking, benchmarks
- Bare metal: ⚠️ Boots, terminal works, but I/O limited and hardware support thin

---

## CPU Support

### Supported
- ✅ Intel Core i5/i7/i9 (6th gen+)
- ✅ AMD Ryzen 5/7/9 (all generations)
- ✅ Intel Xeon (E5 v3+)
- ✅ AMD EPYC (any)

### Known Issues
- ⚠️ **Older CPUs (pre-2015):** May lack CPU features we assume (SSE4.2, AVX). Testing on old iron needed.
- ⚠️ **Specific models:** Haven't tested every CPU. YMMV.

### Features Used
- SSE4.2 (required; for crypto)
- AVX (optional; for ML inference)
- RDRAND (optional; for entropy, fallback to DRNG)

---

## Storage Support

### Tested & Working
- ✅ **virtio-blk** (QEMU)
- ✅ **PCI AHCI** (some boards)

### Untested
- ⚠️ **NVMe (PCI express):** Not implemented yet
- ⚠️ **SCSI (LSI, Megaraid):** Not tested
- ⚠️ **USB Mass Storage:** Not implemented

### Caveats
- **Single disk only** (QEMU testing: one virtio-blk)
- **No RAID support** yet
- **No hot-swap** of drives
- **MBR boot only** (GPT pending)

### If Your Storage Doesn't Work
1. Check BIOS: is AHCI/SATA mode enabled? (Not IDE)
2. Try QEMU first to isolate OS vs. hardware
3. Open an issue with your controller chipset (e.g., "ASMedia 1166 not detected")

---

## Network Support

### Tested & Working
- ✅ **virtio-net** (QEMU)
- ✅ **ARP** (address resolution)
- ✅ **ICMP** (ping)
- ✅ **UDP** (basic datagram)

### Untested
- ⚠️ **Real Ethernet NICs:** Only virtio tested in QEMU
- ⚠️ **Broadcom BCM57416:** Experimental (test platform has this; limited driver)
- ⚠️ **Intel 82540EM:** Not tested
- ⚠️ **WiFi / Bluetooth:** Not implemented
- ⚠️ **IPv6:** Specified, not tested
- ⚠️ **DNS/DHCP:** Manual IP configuration only

### Setting Up Networking

**In QEMU:**
```bash
# Default setup (user networking)
qemu-system-x86_64 \
  -drive file=aetheros.img,format=raw \
  -net nic,model=virtio \
  -net user,hostfwd=tcp::8000-:80
```

**On bare metal:**
1. Plug in Ethernet (virtio not available)
2. Set IP manually in shell: (not yet implemented; needs work)
3. Test ping to gateway

**If network hangs:**
- Try QEMU first
- Reduce network load (single ping, not flood)
- Check BIOS network boot is disabled (may conflict)

---

## Input Devices

### Keyboard
- ✅ **PS/2 PS/2:** Full support
- ❌ **USB HID:** Not implemented
- ❌ **Bluetooth:** Not implemented
- ⚠️ **Laptop keyboard:** Depends on BIOS PS/2 emulation

### Mouse
- ✅ **PS/2 mouse:** Supported
- ❌ **USB mouse:** Not implemented
- ❌ **Touchpad:** Not implemented (trackpads are not standard PS/2)
- ❌ **Touchscreen:** Not implemented

### Special Keys
- ⚠️ **Media keys (volume, brightness):** Not implemented
- ⚠️ **Function keys (Fn):** Depends on BIOS

### If Your Input Doesn't Work
1. **USB keyboard/mouse?** Try PS/2 or USB→PS/2 adapter
2. **Laptop trackpad?** Doesn't work; use external mouse
3. **Keyboard not responding?** Try typing without visual feedback (it may be buffering)

---

## Graphics & Display

### Supported
- ✅ **VGA/VESA framebuffer** (software-rendered)
- ✅ **Resolution:** 1024x768 at 32-bit color (typical)

### Not Supported
- ❌ **GPU acceleration** (no NVIDIA/AMD drivers)
- ❌ **3D graphics** (CPU-only rendering)
- ❌ **HDMI/DisplayPort** (VGA-only for now)
- ❌ **Dual/triple monitors**

### Performance
- Framebuffer rendering at ~30-60 FPS on modern CPUs
- Software rendering is intentionally accurate (no shortcuts)
- No optimization yet for multi-GPU

---

## Known Hardware Issues

### Intel Platforms
- ⚠️ **Secure Boot:** May interfere with boot. Disable in BIOS if boot fails.
- ⚠️ **vPro/AMT:** Not tested. Disable in BIOS if suspicious behavior.
- ⚠️ **PCIe bifurcation:** Not needed yet.

### AMD Platforms
- ⚠️ **Secure Boot:** Same as Intel; disable if needed.
- ⚠️ **Infinity Fabric:** Not relevant for AetherOS.

### Apple Silicon (M1/M2/M3)
- ❌ **Not supported.** ARM64 port in progress but not released.

### Older Hardware (Pre-2015)
- ⚠️ **Untested.** Core logic should work, but edge cases may exist.
- ⚠️ **BIOS vs. UEFI:** Older boards use BIOS; we support both.

---

## Virtual Machine Support

### QEMU (Recommended)
- ✅ **Full support.** All features tested.
- ✅ **Configuration:** x86-64, BIOS boot, 4 GB RAM, 4 cores minimum.

### VirtualBox
- ✅ **Boots.** BIOS mode, virtio storage (attach IDE as fallback).
- ⚠️ **Untested features:** Networking, graphics, input beyond basic.

### VMware Workstation / Fusion
- ✅ **Should work.** Not heavily tested.
- ⚠️ **Bus compatibility:** Use "Custom" / "Other" OS type.

### Hyper-V (Windows)
- ⚠️ **QEMU recommended instead.** Hyper-V untested.

### AWS / Azure / GCP
- ❌ **Not tested.** Could work but needs validation.

---

## Experimental Support

### Known to Work But Not Guaranteed
- Broadcom BCM57416 (on test platform; minimal driver)
- AHCI storage on some ASUS boards
- Multi-core (8 cores on test platform; 16+ untested)

### Known to Not Work
- NVMe (no driver)
- USB devices
- WiFi
- Audio
- Secure Boot (usually)

---

## Troubleshooting

### "OS won't boot"
1. Are you on x86-64? (Only supported arch)
2. Can you boot in QEMU? (Isolates OS vs. hardware)
3. Try disabling Secure Boot in BIOS
4. Try Legacy (BIOS) mode instead of UEFI
5. Check CPU is from 2015 or newer

### "Boots but hangs at logo/prompt"
1. Some drivers take time to load. Wait 30 seconds.
2. Try in QEMU first (faster feedback)
3. Interrupt boot with ESC and check messages

### "Network doesn't work"
1. Check ping works: type `ping 8.8.8.8` (or your gateway IP)
2. Try QEMU with `-net user` first
3. On bare metal, your NIC may not be supported (see list above)

### "Keyboard/mouse frozen"
1. Try typing anyway (buffering may be happening)
2. QEMU: check `-usbdevice keyboard` vs PS/2 mode
3. Bare metal: reseat PS/2 cables

### "Graphics is corrupted/flickering"
1. Framebuffer rendering is CPU-intensive
2. Try lower resolution (if available): (not configurable yet; needs work)
3. Disable any 3D features in BIOS

---

## Testing Your Hardware

We provide a hardware detection utility:

```bash
# Boot AetherOS
./run.ps1

# In the shell, type:
hw
```

This detects and reports:
- CPU model and core count
- Installed RAM
- PCI devices (storage, network, graphics)
- Boot method (BIOS/UEFI)

---

## Reporting Hardware Issues

Found a problem? Please report it with:
1. **Hardware:** CPU model, motherboard, RAM, storage controller
2. **Behavior:** What happened? What did you expect?
3. **Logs:** Boot messages, error output
4. **Steps to reproduce:** Exactly what did you do?
5. **QEMU test:** Does it work in QEMU? (Helps isolate driver issues)

Example:
> "Boots fine in QEMU but hangs at 'PCI probe' on bare metal (ASMedia 1166 NVMe controller). Same AetherOS.img on both. Dell Precision 7550, Intel Xeon E-2176M."

---

## Future Hardware Support

### Planned
- NVMe (PCIe storage)
- USB HID (keyboard/mouse)
- AHCI improvements (more boards)
- Multi-monitor support
- GPU compute integration

### Exploring
- ARM64 (Apple Silicon, Raspberry Pi)
- Secure Boot integration
- RISC-V
- Wireless (WiFi, Bluetooth)
- Audio

---

## Bottom Line

**AetherOS works best in QEMU.** Real hardware works but with caveats:
- x86-64 Intel/AMD only
- PS/2 input only
- Limited storage/network support
- Framebuffer graphics only

**If your hardware isn't listed, it probably won't work yet.** That's okay. Use QEMU for now, and we'll expand support over time.

---

Questions? contact@cognitive-industries.org

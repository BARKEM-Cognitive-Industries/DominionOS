# Build the kernel in test mode, assemble a BIOS image, boot it headless in
# QEMU, and report the serial log + exit code (33 = all bare-metal tests pass).
# Resolve the repo root from this script's own location so the launcher works from
# any clone path without editing hardcoded directories.
#
# -ExtraFeatures adds cargo features on top of `qemu_test` (e.g. `usb_storage` to
# activate the USB mass-storage auto-probe against the attached usb-storage device).
param([string]$ExtraFeatures = "")
$root = $PSScriptRoot
$env:Path = (Join-Path $env:USERPROFILE '.cargo\bin') + ";C:\Program Files\qemu;" + $env:Path
. "$root\qemu-common.ps1"
# Release build: a 1.7 MiB optimised kernel boots in seconds under TCG, where the
# 31 MiB debug build takes minutes. Correctness is unchanged.
$elf  = "$root\kernel\target\x86_64-dominion\release\dominion-kernel"
$img  = "$root\dominionos-test.img"
$log  = "$root\selftest-serial.log"
$disk = "$root\dominion-data-test.img"
# A framebuffer mode for the live-dashboard render test (BIOS VBE provides 720p).
$env:RESOLUTION = "1280x720"

# A scratch virtio-blk data disk (16 MiB), separate from the boot image, so the
# persistence tests have a real block device to write to.
if (-not (Test-Path $disk)) {
  $fs = [System.IO.File]::Create($disk)
  $fs.SetLength(16MB)
  $fs.Close()
}

# Extra storage controllers so the hardware report demonstrably enumerates NVMe + AHCI
# (SATA) alongside virtio — i.e. the storage classes a real PC presents.
$nvme = "$root\dominion-nvme-test.img"
if (-not (Test-Path $nvme)) {
  $fs = [System.IO.File]::Create($nvme)
  $fs.SetLength(8MB)
  $fs.Close()
}
$sata = "$root\dominion-sata-test.img"
if (-not (Test-Path $sata)) {
  $fs = [System.IO.File]::Create($sata)
  $fs.SetLength(8MB)
  $fs.Close()
}
$usb = "$root\dominion-usb-test.img"
if (-not (Test-Path $usb)) {
  $fs = [System.IO.File]::Create($usb)
  $fs.SetLength(8MB)
  $fs.Close()
}

$features = "qemu_test"
if ($ExtraFeatures) { $features = "qemu_test $ExtraFeatures" }
Push-Location "$root\kernel"
cargo build --release --features "$features" 2>&1 | Select-Object -Last 2
Pop-Location

Push-Location "$root\boot"
cargo run --release -- $elf $img 2>&1 | Select-Object -Last 1
Pop-Location

$accel = Get-DominionAccel
$vcpus = Get-DominionVcpus
Write-Host "Running bare-metal selftest (accel=$accel, smp=$vcpus) ..."
"" | Out-File -FilePath $log -Encoding ascii
$p = Start-Process -FilePath "qemu-system-x86_64.exe" -ArgumentList @(
  "-cpu","qemu64,+rdrand", "-accel",$accel, "-smp","$vcpus", "-m","2048", "-vga","std",
  "-drive","format=raw,file=$img",
  "-drive","id=dominiondata,format=raw,if=none,file=$disk",
  "-device","virtio-blk-pci,drive=dominiondata",
  "-drive","id=nvmedisk,format=raw,if=none,file=$nvme",
  "-device","nvme,drive=nvmedisk,serial=aeth-nvme0",
  "-device","ich9-ahci,id=sata0",
  "-drive","id=satadisk,format=raw,if=none,file=$sata",
  "-device","ide-hd,drive=satadisk,bus=sata0.0",
  "-device","qemu-xhci,id=xhci",
  "-drive","id=usbdisk,format=raw,if=none,file=$usb",
  "-device","usb-storage,bus=xhci.0,drive=usbdisk",
  "-netdev","user,id=n0",
  "-device","virtio-net-pci,netdev=n0",
  "-netdev","user,id=n1",
  "-device","rtl8139,netdev=n1",
  "-netdev","user,id=n2",
  "-device","e1000,netdev=n2",
  "-device","isa-debug-exit,iobase=0xf4,iosize=0x04",
  "-serial","file:$log","-display","none","-no-reboot"
) -PassThru
$ok = $p.WaitForExit(180000)
if (-not $ok) { Stop-Process -Id $p.Id -Force; "QEMU TIMEOUT" }
"=== QEMU exit code: $($p.ExitCode)  (33 = pass, 35 = fail) ==="
Get-Content $log -ErrorAction SilentlyContinue


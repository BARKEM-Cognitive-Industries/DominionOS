<#
.SYNOPSIS
    Boot DominionOS in QEMU configured to mimic a REAL modern PC as closely as possible,
    so bugs that only show on bare metal reproduce here instead of hiding behind QEMU's
    convenient defaults.

.DESCRIPTION
    The normal launchers use the easy path (i440fx chipset, virtio disk/net, SeaBIOS,
    PS/2 input) — which is exactly why the desktop "worked in QEMU" but died on metal.
    This launcher instead uses:

      * q35           — modern PCIe chipset (like a real PC), APIC interrupt routing
      * UEFI (OVMF)   — real firmware, not legacy BIOS
      * USB boot      — the image is attached as a USB mass-storage device on an xHCI
                        controller and booted from there (the real "boot from USB" path)
      * USB HID input — usb-kbd + usb-mouse on xHCI, and NO PS/2 devices, so input must
                        go through a real USB HID stack (this is why the mouse is dead on
                        your machine — the OS has no USB-HID driver yet, only PS/2)
      * NVMe          — a real NVMe data disk to exercise the NVMe driver

    So this is the harness for developing the USB-HID driver and fixing NVMe without
    reflashing a stick every time.

.PARAMETER Mode   live | safe | bench   (which image to build/boot)
.PARAMETER Headless  Run with no window and capture serial to baremetal-serial.log.
.PARAMETER Seconds   With -Headless, how long to run before stopping (default 30).
.PARAMETER NoBuild   Boot the existing image without rebuilding.

.EXAMPLE
    .\run-baremetal.ps1              # build + boot live in a window; try the mouse
.EXAMPLE
    .\run-baremetal.ps1 -Mode safe -Headless -Seconds 20
#>
#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('live', 'safe', 'bench')][string]$Mode = 'live',
    [switch]$Headless,
    [int]$Seconds = 30,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot; if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$env:Path = (Join-Path $env:USERPROFILE '.cargo\bin') + ";C:\Program Files\qemu;" + $env:Path
$qemu = 'C:\Program Files\qemu\qemu-system-x86_64.exe'

function Invoke-Native($exe, $cmdArgs) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $exe @cmdArgs | Out-Host } finally { $ErrorActionPreference = $prev }
    return $LASTEXITCODE
}

# Feature set per mode (matches make-bootable-usb.ps1).
switch ($Mode) {
    'safe'  { $features = 'safe_mode' }
    'bench' { $features = 'qemu_bench' }
    default { $features = '' }
}
$img = Join-Path $root "dominionos-baremetal-$Mode.img"
$elf = Join-Path $root "kernel\target\x86_64-dominion\release\dominion-kernel"

if (-not $NoBuild) {
    Write-Host "Building kernel (mode=$Mode, features=[$features]) ..." -ForegroundColor Cyan
    Push-Location (Join-Path $root 'kernel')
    try {
        $a = @('build', '--release'); if ($features) { $a += @('--features', $features) }
        if ((Invoke-Native 'cargo' $a) -ne 0) { throw "kernel build failed" }
    } finally { Pop-Location }
    $env:RESOLUTION = '1920x1080'
    Write-Host "Building UEFI USB image ..." -ForegroundColor Cyan
    Push-Location (Join-Path $root 'boot')
    try {
        # arg2 = throwaway BIOS img (builder always writes it); arg3 = the UEFI image we boot.
        if ((Invoke-Native 'cargo' @('run', '--release', '--', $elf, "$img.bios", $img)) -ne 0) { throw "image build failed" }
    } finally { Pop-Location }
}
if (-not (Test-Path $img)) { throw "image not found: $img (run without -NoBuild)" }

# OVMF UEFI firmware (code-only is fine: boots the removable \EFI\BOOT\BOOTX64.EFI path).
# Copy it to a SPACE-FREE local path: QEMU's `-drive file=...` splits on the space in
# "C:\Program Files\...", which silently breaks the pflash drive.
$ovmfSrc = $null
foreach ($c in @('C:\Program Files\qemu\share\edk2-x86_64-code.fd', 'C:\Program Files\qemu\share\OVMF_CODE.fd')) {
    if (Test-Path $c) { $ovmfSrc = $c; break }
}
if (-not $ovmfSrc) { throw "OVMF UEFI firmware not found in the QEMU share dir." }
$ovmf = Join-Path $root 'ovmf-code.fd'
if (-not (Test-Path $ovmf)) { Copy-Item $ovmfSrc $ovmf }

# A persistent NVMe data disk to exercise the NVMe driver (256 MiB).
$nvmeImg = Join-Path $root "dominionos-baremetal-nvme.img"
if (-not (Test-Path $nvmeImg)) {
    $fs = [System.IO.File]::Create($nvmeImg); $fs.SetLength(256MB); $fs.Close()
}

. (Join-Path $root 'qemu-common.ps1')
$accel = Get-DominionAccel

# The bare-metal-like machine: q35 + UEFI + boot from USB-MSC on xHCI + USB HID + NVMe,
# and explicitly NO PS/2 devices, so input must traverse a real USB-HID path.
$qargs = @(
    '-machine', 'q35',
    '-cpu', 'qemu64,+rdrand',
    '-accel', $accel,
    '-m', '4096',
    '-drive', "if=pflash,format=raw,readonly=on,file=$ovmf",
    # boot disk as USB mass-storage on an xHCI controller
    '-device', 'qemu-xhci,id=xhci',
    '-drive', "if=none,id=usbboot,format=raw,file=$img",
    '-device', 'usb-storage,bus=xhci.0,drive=usbboot,bootindex=0',
    # USB HID input (no PS/2) — reproduces the dead-mouse situation on real hardware
    '-device', 'usb-kbd,bus=xhci.0',
    '-device', 'usb-mouse,bus=xhci.0',
    # a real NVMe disk to test the NVMe driver
    '-drive', "if=none,id=nvme0,format=raw,file=$nvmeImg",
    '-device', 'nvme,drive=nvme0,serial=DOMINION01',
    '-vga', 'std'
)

$log = Join-Path $root 'baremetal-serial.log'
Write-Host "Booting bare-metal-like QEMU (q35 + UEFI + USB boot + USB HID + NVMe, accel=$accel)" -ForegroundColor Green
if ($Headless) {
    "" | Out-File -FilePath $log -Encoding ascii
    $qargs += @('-serial', "file:$log", '-display', 'none', '-no-reboot')
    $p = Start-Process -FilePath $qemu -ArgumentList $qargs -PassThru
    Start-Sleep -Seconds $Seconds
    if (-not $p.HasExited) { try { Stop-Process -Id $p.Id -Force } catch {} }
    Write-Host "=== serial tail ($log) ===" -ForegroundColor Cyan
    Get-Content $log -Tail 40
}
else {
    Write-Host "(a window will open; try the mouse/keyboard. Close it to exit.)" -ForegroundColor Gray
    $qargs += @('-serial', "file:$log")
    [void](Invoke-Native $qemu $qargs)
    Write-Host "serial log: $log" -ForegroundColor Gray
}

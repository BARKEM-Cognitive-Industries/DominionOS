# Build the interactive dominionos and boot it into the safe-mode terminal in a
# QEMU window. Use the keyboard to drive ASH (try: help, caps, obj, state,
# run fib, dominion 6 * 7, selftest, shutdown).
# Resolve the repo root from this script's own location so the launcher works from
# any clone path without editing hardcoded directories.
$root = $PSScriptRoot
$env:Path = (Join-Path $env:USERPROFILE '.cargo\bin') + ";C:\Program Files\qemu;" + $env:Path
. "$root\qemu-common.ps1"
# Release build: the graphical dashboard renders 1080p at 30 fps, which needs
# optimised code (debug is far too slow for real-time rendering under TCG).
$elf  = "$root\kernel\target\x86_64-dominion\release\dominion-kernel"
$img  = "$root\dominionos.img"
$disk = "$root\dominion-data.img"

# Request a full-HD framebuffer for the dashboard (the desktop adapts to whatever
# mode the firmware actually provides).
$env:RESOLUTION = "1920x1080"

# Persistent virtio-blk data disk (16 MiB) ??? survives across boots, so anything
# saved with the `disk` command is still there next time you run.
if (-not (Test-Path $disk)) {
  $fs = [System.IO.File]::Create($disk)
  $fs.SetLength(16MB)
  $fs.Close()
}

Push-Location "$root\kernel"
cargo build --release 2>&1 | Select-Object -Last 2
Pop-Location

Push-Location "$root\boot"
cargo run --release -- $elf $img 2>&1 | Select-Object -Last 1
Pop-Location

# Hardware acceleration (whpx) if available, else multi-threaded TCG. This is what
# lets QEMU use the host CPU fully instead of being stuck on one slow TCG thread.
$accel = Get-DominionAccel
$vcpus = Get-DominionVcpus
Write-Host "Booting dominionos in QEMU (accel=$accel, smp=$vcpus; close the window to exit)..."
qemu-system-x86_64 -cpu "qemu64,+rdrand" `
  -accel $accel `
  -smp $vcpus `
  -m 2048 `
  -vga std `
  -drive "format=raw,file=$img" `
  -drive "id=dominiondata,format=raw,if=none,file=$disk" `
  -device "virtio-blk-pci,drive=dominiondata" `
  -netdev "user,id=n0" `
  -device "virtio-net-pci,netdev=n0" `
  -serial stdio


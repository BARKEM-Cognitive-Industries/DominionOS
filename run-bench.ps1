# Build DominionOS in benchmark mode, boot it headless in QEMU, and report the
# real-world performance battery. Each `BENCH ...` serial line is parsed into a
# table and written to bench-results.json for tracking / regression gates.
#
#   ./run-bench.ps1                  # auto accel, default scale (deterministic SSE2)
#   ./run-bench.ps1 -Fma             # opt-in AVX/FMA build (non-deterministic, ~FMA ceiling)
#   $env:DOMINION_ACCEL='tcg'; ./run-bench.ps1   # force software emulation
#
# The kernel is built with --features qemu_bench, which also raises the heap to
# 1 GiB (big_heap); we give QEMU 4 GiB so the frame allocator can map it.
param([switch]$Fma)
# Resolve the repo root from this script's own location so the launcher works from
# any clone path without editing hardcoded directories.
$root = $PSScriptRoot
$env:Path = (Join-Path $env:USERPROFILE '.cargo\bin') + ";C:\Program Files\qemu;" + $env:Path
. "$root\qemu-common.ps1"

# FMA knob: opt-in AVX+FMA target (non-deterministic, faster on multiply-bound math).
if ($Fma) {
  $features = "qemu_bench,fma"
  $targetArg = @("--target", "x86_64-dominion-fma.json")
  $elfTarget = "x86_64-dominion-fma"
  $cpu = "qemu64,+rdrand,+avx,+fma,+xsave,+xsaveopt"
} else {
  $features = "qemu_bench"
  $targetArg = @()
  $elfTarget = "x86_64-dominion"
  $cpu = "qemu64,+rdrand"
}
$elf  = "$root\kernel\target\$elfTarget\release\dominion-kernel"
$img  = "$root\dominionos-bench.img"
$log  = "$root\bench-serial.log"
$json = "$root\bench-results.json"
$disk = "$root\dominion-data-bench.img"
$env:RESOLUTION = "1280x720"

# A roomy scratch disk so the storage + persistence benchmarks have real space
# (256 MiB). Separate from the boot image and the interactive data disk.
if (-not (Test-Path $disk)) {
  $fs = [System.IO.File]::Create($disk)
  $fs.SetLength(256MB)
  $fs.Close()
}

Push-Location "$root\kernel"
cargo build --release --features $features @targetArg 2>&1 | Select-Object -Last 2
Pop-Location

Push-Location "$root\boot"
cargo run --release -- $elf $img 2>&1 | Select-Object -Last 1
Pop-Location

$accel = Get-DominionAccel
$hostCores = [int]$env:NUMBER_OF_PROCESSORS
$vcpus = [Math]::Max(1, [Math]::Min($hostCores, 8))
Write-Host "Running benchmark battery (accel=$accel, smp=$vcpus, ram=4096 MiB). This can take a while under TCG..."
"" | Out-File -FilePath $log -Encoding ascii
$p = Start-Process -FilePath "qemu-system-x86_64.exe" -ArgumentList @(
  "-cpu",$cpu, "-accel",$accel, "-smp","$vcpus",
  "-m","4096",
  "-vga","std",
  "-drive","format=raw,file=$img",
  "-drive","id=dominiondata,format=raw,if=none,file=$disk",
  "-device","virtio-blk-pci,drive=dominiondata",
  "-netdev","user,id=n0",
  "-device","virtio-net-pci,netdev=n0",
  "-device","isa-debug-exit,iobase=0xf4,iosize=0x04",
  "-serial","file:$log","-display","none","-no-reboot"
) -PassThru
# Benchmarks are heavy; allow up to 15 minutes (fast under whpx, slow under TCG).
$ok = $p.WaitForExit(900000)
if (-not $ok) { try { Stop-Process -Id $p.Id -Force } catch { }; Write-Host "QEMU TIMEOUT" }
"=== QEMU exit code: $($p.ExitCode)  (33 = clean finish) ==="

# ?????? parse the BENCH lines ??????
$lines = Get-Content $log -ErrorAction SilentlyContinue
$results = @()
foreach ($line in $lines) {
  if ($line -match '^BENCH\s+(\S+)\s*(.*)$') {
    $category = $Matches[1]
    $rest = $Matches[2]
    $row = [ordered]@{ category = $category }
    foreach ($kv in ($rest -split '\s+')) {
      if ($kv -match '^([^=]+)=(.*)$') { $row[$Matches[1]] = $Matches[2] }
    }
    $results += [pscustomobject]$row
  }
}

if ($results.Count -eq 0) {
  Write-Host "No BENCH lines found. Full serial log:"
  $lines | ForEach-Object { Write-Host $line }
} else {
  Write-Host ""
  Write-Host "================ DominionOS benchmark results ================"
  foreach ($r in $results) {
    Write-Host ("--- {0} ---" -f $r.category)
    foreach ($prop in $r.PSObject.Properties) {
      if ($prop.Name -ne 'category') { Write-Host ("    {0,-26} {1}" -f $prop.Name, $prop.Value) }
    }
  }
  # Stamp host-known facts the guest can't see, then persist as JSON.
  $doc = [ordered]@{
    accel     = $accel
    vcpus     = $vcpus
    ram_mib   = 4096
    exit_code = $p.ExitCode
    results   = $results
  }
  $doc | ConvertTo-Json -Depth 6 | Out-File -FilePath $json -Encoding utf8
  Write-Host ""
  Write-Host "Wrote $json"
}


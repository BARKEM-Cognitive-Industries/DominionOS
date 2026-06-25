# Build DominionOS in VALIDATION mode and boot it headless in QEMU.
#
# Where run-bench.ps1 measures abstraction efficiency, this answers the harder
# question: is any of it real on hardware, and does it survive stress + failure?
#   - memory-latency mountain  (cache hierarchy is REAL under whpx, flat under TCG)
#   - model-vs-hardware boundary (in-memory op vs VM-exit / device round-trip)
#   - soak (drift / fragmentation / leak / rollback frequency)
#   - chaos (failure injection + deterministic recovery)
#
#   ./run-validate.ps1                       # auto accel
#   ./run-validate.ps1 -Accels whpx,tcg      # run twice; contrast the cache mountain
param([string[]]$Accels = @())

# Resolve the repo root from this script's own location so the launcher works from
# any clone path without editing hardcoded directories.
$root = $PSScriptRoot
$env:Path = (Join-Path $env:USERPROFILE '.cargo\bin') + ";C:\Program Files\qemu;" + $env:Path
. "$root\qemu-common.ps1"

$elf  = "$root\kernel\target\x86_64-dominion\release\dominion-kernel"
$img  = "$root\dominionos-validate.img"
$disk = "$root\dominion-data-validate.img"
$env:RESOLUTION = "1280x720"

if (-not (Test-Path $disk)) {
  $fs = [System.IO.File]::Create($disk); $fs.SetLength(64MB); $fs.Close()
}

Push-Location "$root\kernel"
cargo build --release --features qemu_validate 2>&1 | Select-Object -Last 2
Pop-Location
Push-Location "$root\boot"
cargo run --release -- $elf $img 2>&1 | Select-Object -Last 1
Pop-Location

# The validation build brings up real SMP, so give QEMU several vCPUs (capped at 8)
# so the cross-core scaling curve has cores to scale across. whpx and tcg,thread=multi
# both run each vCPU on its own host thread ??? real parallel execution.
$hostCores = [int]$env:NUMBER_OF_PROCESSORS
$vcpus = [Math]::Max(1, [Math]::Min($hostCores, 8))
# Default: a single auto-selected accelerator. Or run each accel the caller asked
# for (e.g. whpx then tcg) to show the cache mountain is real vs emulated.
if ($Accels.Count -eq 0) { $runs = @((Get-DominionAccel)) }
else {
  $runs = @()
  foreach ($a in $Accels) { $env:DOMINION_ACCEL = $a; $runs += (Get-DominionAccel) }
  Remove-Item Env:\DOMINION_ACCEL -ErrorAction SilentlyContinue
}

$allDocs = @()
foreach ($accel in $runs) {
  $tag = ($accel -split ',')[0]
  $log  = "$root\validate-serial-$tag.log"
  $json = "$root\validate-results-$tag.json"
  Write-Host ""
  Write-Host "=== validation run: accel=$accel ==="
  "" | Out-File -FilePath $log -Encoding ascii
  $p = Start-Process -FilePath "qemu-system-x86_64.exe" -ArgumentList @(
    "-cpu","qemu64,+rdrand", "-accel",$accel, "-smp","$vcpus", "-m","4096",
    "-vga","std",
    "-drive","format=raw,file=$img",
    "-drive","id=dominiondata,format=raw,if=none,file=$disk",
    "-device","virtio-blk-pci,drive=dominiondata",
    "-device","isa-debug-exit,iobase=0xf4,iosize=0x04",
    "-serial","file:$log","-display","none","-no-reboot"
  ) -PassThru
  $ok = $p.WaitForExit(1200000)
  if (-not $ok) { try { Stop-Process -Id $p.Id -Force } catch { }; Write-Host "QEMU TIMEOUT" }
  "=== QEMU exit code: $($p.ExitCode)  (33 = clean finish) ==="

  $lines = Get-Content $log -ErrorAction SilentlyContinue
  $results = @()
  foreach ($line in $lines) {
    if ($line -match '^BENCH\s+(\S+)\s*(.*)$') {
      $row = [ordered]@{ category = $Matches[1] }
      foreach ($kv in ($Matches[2] -split '\s+')) {
        if ($kv -match '^([^=]+)=(.*)$') { $row[$Matches[1]] = $Matches[2] }
      }
      $results += [pscustomobject]$row
    }
  }
  if ($results.Count -eq 0) { Write-Host "No BENCH lines. Serial tail:"; $lines | Select-Object -Last 20 | ForEach-Object { Write-Host $_ } }
  else {
    foreach ($r in $results) {
      Write-Host ("--- {0} ---" -f $r.category)
      foreach ($prop in $r.PSObject.Properties) {
        if ($prop.Name -ne 'category') { Write-Host ("    {0,-26} {1}" -f $prop.Name, $prop.Value) }
      }
    }
    $doc = [ordered]@{ accel = $accel; vcpus = $vcpus; ram_mib = 4096; exit_code = $p.ExitCode; results = $results }
    $doc | ConvertTo-Json -Depth 6 | Out-File -FilePath $json -Encoding utf8
    Write-Host "Wrote $json"
    $allDocs += $doc
  }
}

# If we ran more than one accelerator, print the cache-mountain contrast side by side.
if ($allDocs.Count -gt 1) {
  Write-Host ""
  Write-Host "=== memory-latency mountain: ns/access by working set ==="
  $header = "ws_kib"; foreach ($d in $allDocs) { $header += ("`t" + ($d.accel -split ',')[0]) }
  Write-Host $header
  $sizes = $allDocs[0].results | Where-Object { $_.category -eq 'mem_hierarchy' } | ForEach-Object { $_.ws_kib }
  foreach ($ws in $sizes) {
    $rowtxt = "$ws"
    foreach ($d in $allDocs) {
      $cell = ($d.results | Where-Object { $_.category -eq 'mem_hierarchy' -and $_.ws_kib -eq $ws }).ns_per_access
      $rowtxt += "`t$cell"
    }
    Write-Host $rowtxt
  }
}


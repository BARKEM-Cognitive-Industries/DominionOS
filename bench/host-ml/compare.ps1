# Host-vs-guest ML benchmark comparison.
#
# Runs the native host baseline (this crate) and lines it up against the most
# recent in-QEMU guest run (../../bench-serial.log from ../../run-bench.ps1),
# printing a side-by-side table with the guest/host ratio for each metric.
# Both sides run the identical dominion-core ml workload, so the ratio is the real
# compute overhead DominionOS+QEMU adds vs the bare host.
#
# Usage: ./compare.ps1            ./compare.ps1 -Native   (also show host AVX2 ceiling)
param([switch]$Native)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$here\..\.."
$guestLog = "$root\bench-serial.log"
$exe = "$here\target\release\host-ml-bench.exe"

function Parse-Bench($lines) {
  $m = @{}
  foreach ($line in $lines) {
    if ($line -match "BENCH (ml_\w+)\s+(.*)") {
      $cat = $matches[1]; $rest = $matches[2]
      foreach ($kv in ($rest -split "\s+")) {
        if ($kv -match "^(\w+)=(-?\d+)$") { $m["$cat.$($matches[1])"] = [int64]$matches[2] }
      }
    }
  }
  return $m
}

Write-Host "Building + running host baseline (same code as guest) ..." -ForegroundColor Cyan
Push-Location $here
cargo build --release 2>&1 | Out-Null
Pop-Location
$hostOut = & $exe
$hostM = Parse-Bench $hostOut

if (-not (Test-Path $guestLog)) {
  Write-Host "No guest log yet. Run ../../run-bench.ps1 first." -ForegroundColor Yellow
  $hostOut | Select-String "BENCH ml"
  exit 0
}
$guestM = Parse-Bench (Get-Content $guestLog)

$metrics = @(
  @{k="ml_matmul.mflop_per_s";    label="f64 matmul MFLOP/s"},
  @{k="ml_int8_matmul.mop_per_s"; label="int8 matmul Mop/s"},
  @{k="ml_train.steps_per_s";     label="training steps/s"},
  @{k="ml_infer.infer_per_s";     label="inference passes/s"}
)

Write-Host ""
Write-Host ("{0,-22} {1,14} {2,14} {3,12}" -f "metric","guest_QEMU","host_native","guest_over_host")
Write-Host ("-" * 66)
foreach ($met in $metrics) {
  $g = $guestM[$met.k]; $h = $hostM[$met.k]
  if (($null -ne $g) -and ($null -ne $h) -and ($h -ne 0)) {
    $r = [math]::Round($g / $h, 2)
    Write-Host ("{0,-22} {1,14} {2,14} {3,12}" -f $met.label, $g, $h, ("{0}x" -f $r))
  }
}
Write-Host ("-" * 66)
Write-Host "ratio at or above 1.0 means DominionOS-in-QEMU matched or beat the bare host."
Write-Host "same physical CPU via WHPX; guest can win because the single-vCPU VM has no host interference."

if ($Native) {
  Write-Host ""
  Write-Host "Host full-ISA ceiling, target-cpu=native AVX2:" -ForegroundColor Cyan
  $env:RUSTFLAGS = "-C target-cpu=native"
  Push-Location $here
  & cargo run --release 2>$null | Select-String "BENCH ml_matmul|BENCH ml_int8|BENCH ml_train|BENCH ml_infer"
  Pop-Location
  Remove-Item Env:\RUSTFLAGS
}

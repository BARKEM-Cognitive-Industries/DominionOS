<#
.SYNOPSIS
    Extract the DominionOS boot/install/run/benchmark log from a USB (or raw image)
    after a bare-metal boot, and parse out the benchmark results.

.DESCRIPTION
    On bare metal the kernel persists its full captured serial log - which includes
    every "BENCH ..." line - to the TAIL of the drive it booted from, behind an
    "AELOG001" superblock (see kernel/src/bootlog.rs). This reads that blob straight
    off the raw device and writes:

        bootlog-usb.txt     - the complete boot/install/run/benchmark log (plain text)
        bench-results.json  - the parsed BENCH table (only if benchmark lines exist)

    Reading a physical drive needs Administrator. Reading a .img file does not.

.PARAMETER Disk
    Physical disk number to read (as shown by make-bootable-usb.ps1 / Get-Disk).

.PARAMETER Image
    A raw image file to read instead of a physical disk.

.PARAMETER Out
    Output path for the extracted log (default: bootlog-usb.txt).

.EXAMPLE
    .\read-usb-results.ps1 -Disk 2

.EXAMPLE
    .\read-usb-results.ps1 -Image dominionos-usb-bench-uefi.img
#>
#requires -Version 5.1
[CmdletBinding(DefaultParameterSetName = 'disk')]
param(
    [Parameter(ParameterSetName = 'disk', Mandatory = $true)]
    [int]$Disk,

    [Parameter(ParameterSetName = 'image', Mandatory = $true)]
    [string]$Image,

    [string]$Out = 'bootlog-usb.txt',

    [string]$Json = 'bench-results.json'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }

# Must match kernel/src/bootlog.rs.
$RESERVE = 1024            # sectors reserved at the tail for the log
$BS = 512                  # sector size
$MAGIC = 'AELOG001'        # superblock magic

function Test-Admin {
    $p = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Resolve the device path + total byte length.
if ($PSCmdlet.ParameterSetName -eq 'disk') {
    if (-not (Test-Admin)) { Write-Error "Reading a physical disk needs Administrator. Re-run from an elevated PowerShell."; exit 1 }
    $d = Get-Disk -Number $Disk -ErrorAction Stop
    $devPath = "\\.\PhysicalDrive$Disk"
    $length = [int64]$d.Size
    Write-Host "Reading disk #$Disk  $($d.FriendlyName)  ($([math]::Round($length/1GB,1)) GB)" -ForegroundColor Cyan
}
else {
    if (-not (Test-Path $Image)) { Write-Error "image not found: $Image"; exit 1 }
    $devPath = (Resolve-Path $Image).Path
    $length = (Get-Item $devPath).Length
    Write-Host "Reading image $devPath  ($([math]::Round($length/1MB,1)) MiB)" -ForegroundColor Cyan
}

$cap = [int64][math]::Floor($length / $BS)
if ($cap -le ($RESERVE + 2)) { Write-Error "device too small ($cap sectors) to hold a log"; exit 1 }
$lba = $cap - $RESERVE
$off = $lba * $BS

# Open the device. A physical drive must be opened with FileShare ReadWrite, and all
# reads must be sector-aligned in BOTH offset and length - so we read the whole
# reserved tail region in one aligned gulp and parse the header out of memory.
$fs = New-Object System.IO.FileStream($devPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
try {
    $fs.Seek($off, [System.IO.SeekOrigin]::Begin) | Out-Null
    $regionBytes = [int]($RESERVE * $BS)
    $region = New-Object byte[] $regionBytes
    $read = 0
    while ($read -lt $regionBytes) {
        $n = $fs.Read($region, $read, ($regionBytes - $read))
        if ($n -le 0) { break }
        $read += $n
    }

    $magic = [System.Text.Encoding]::ASCII.GetString($region, 0, 8)
    if ($magic -ne $MAGIC) {
        Write-Error "No DominionOS log at LBA $lba (found magic '$magic')."
        Write-Host "  The OS may not have persisted one. For a live boot, run 'log save' in the ASH shell or power off cleanly. For a bench/validate boot, let it finish before powering off." -ForegroundColor Yellow
        exit 1
    }
    $len = [BitConverter]::ToUInt64($region, 8)
    if ($len -le 0 -or $len -gt (($RESERVE - 1) * $BS)) { Write-Error "implausible log length: $len"; exit 1 }

    # Payload starts in the sector after the superblock (offset $BS within the region).
    $payload = New-Object byte[] $len
    [System.Array]::Copy($region, $BS, $payload, 0, [int]$len)
    $text = [System.Text.Encoding]::UTF8.GetString($payload, 0, [int]$len)

    $outPath = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $root $Out }
    Set-Content -Path $outPath -Value $text -Encoding utf8
    Write-Host "Extracted $len bytes of log -> $outPath  (from LBA $lba)" -ForegroundColor Green
}
finally { $fs.Dispose() }

# -- parse BENCH lines into JSON (same schema as run-bench.ps1) --
$lines = $text -split "`r?`n"
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

if ($results.Count -gt 0) {
    $jsonPath = if ([System.IO.Path]::IsPathRooted($Json)) { $Json } else { Join-Path $root $Json }
    $doc = [ordered]@{
        source      = $devPath
        extracted   = (Get-Date).ToString('s')
        bench_count = $results.Count
        results     = $results
    }
    $doc | ConvertTo-Json -Depth 6 | Out-File -FilePath $jsonPath -Encoding utf8
    Write-Host ""
    Write-Host "================ DominionOS benchmark results ($($results.Count) categories) ================" -ForegroundColor Cyan
    foreach ($r in $results) {
        Write-Host ("--- {0} ---" -f $r.category) -ForegroundColor White
        foreach ($prop in $r.PSObject.Properties) {
            if ($prop.Name -ne 'category') { Write-Host ("    {0,-26} {1}" -f $prop.Name, $prop.Value) }
        }
    }
    Write-Host ""
    Write-Host "Wrote $jsonPath" -ForegroundColor Green
}
else {
    Write-Host "No BENCH lines in the log (this was a live/selftest boot, not a benchmark run)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "----------------- log tail (last 30 lines) -----------------" -ForegroundColor DarkGray
($text -split "`r?`n" | Select-Object -Last 30) -join [Environment]::NewLine

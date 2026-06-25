<#
.SYNOPSIS
    Diagnostic: hex-dump the DominionOS log superblock and scan a USB/disk tail for the
    log magic and any captured boot text. Use when read-usb-results.ps1 reports an
    odd/zero length, to see what is actually on the device.

.EXAMPLE
    .\scan-usb-log.ps1 -Disk 1
#>
#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$Disk,
    # How many MiB at the end of the device to scan for the magic + log text. The
    # kernel may write at its own idea of "tail" if its USB capacity differs from
    # Windows', so scan a wide window by default.
    [int]$ScanMiB = 512
)
$ErrorActionPreference = 'Stop'
$BS = 512; $RESERVE = 1024; $MAGIC = 'AELOG001'

$p = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Run from an elevated PowerShell."; exit 1
}
$d = Get-Disk -Number $Disk
$size = [int64]$d.Size
$cap = [int64][math]::Floor($size / $BS)
$tailLba = $cap - $RESERVE
Write-Host "Disk #$Disk  $($d.FriendlyName)  size=$size bytes  ($cap sectors)" -ForegroundColor Cyan
Write-Host "Expected log superblock LBA = $tailLba (offset $($tailLba*$BS))" -ForegroundColor Cyan

$fs = New-Object System.IO.FileStream("\\.\PhysicalDrive$Disk", [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
try {
    # 1) Hex-dump the superblock sector at the expected tail.
    $fs.Seek($tailLba * $BS, 'Begin') | Out-Null
    $sb = New-Object byte[] $BS
    [void]$fs.Read($sb, 0, $BS)
    $magic = [Text.Encoding]::ASCII.GetString($sb, 0, 8)
    $len = [BitConverter]::ToUInt64($sb, 8)
    Write-Host ""
    Write-Host "=== superblock @ tail LBA $tailLba ===" -ForegroundColor White
    Write-Host ("magic bytes : {0}" -f (($sb[0..7] | ForEach-Object { $_.ToString('X2') }) -join ' '))
    Write-Host ("magic ascii : '{0}'" -f $magic)
    Write-Host ("len  bytes  : {0}" -f (($sb[8..15] | ForEach-Object { $_.ToString('X2') }) -join ' '))
    Write-Host ("len  value  : {0}" -f $len)
    Write-Host ("next 16 B   : {0}" -f (($sb[16..31] | ForEach-Object { $_.ToString('X2') }) -join ' '))

    # 2) Scan the last $ScanMiB for the magic and for log text, in 1 MiB aligned chunks.
    $scanBytes = [int64]$ScanMiB * 1MB
    $endOff = [int64]$cap * $BS
    $startOff = $endOff - $scanBytes
    if ($startOff -lt 0) { $startOff = [int64]0 }
    $startOff = $startOff - ($startOff % $BS)
    Write-Host ""
    Write-Host "=== scanning last $ScanMiB MiB (from offset $startOff) ===" -ForegroundColor White
    $fs.Seek($startOff, 'Begin') | Out-Null
    $chunk = New-Object byte[] (1MB)
    $magicBytes = [Text.Encoding]::ASCII.GetBytes($MAGIC)
    $pos = $startOff
    $foundMagic = @(); $foundText = @()
    $patterns = @('[boot]', 'BENCH ', 'PANIC', '[desktop]', 'DominionOS')
    while ($pos -lt $endOff) {
        $n = $fs.Read($chunk, 0, $chunk.Length)
        if ($n -le 0) { break }
        $s = [Text.Encoding]::ASCII.GetString($chunk, 0, $n)
        # magic
        $idx = $s.IndexOf($MAGIC)
        while ($idx -ge 0) {
            $absLba = [int64](($pos + $idx) / $BS)
            $lenAt = if (($idx + 16) -le $n) { [BitConverter]::ToUInt64($chunk, $idx + 8) } else { -1 }
            $foundMagic += "  magic at offset $($pos+$idx)  (LBA $absLba)  len=$lenAt"
            $idx = $s.IndexOf($MAGIC, $idx + 1)
        }
        # text patterns (report first hit per chunk per pattern)
        foreach ($pat in $patterns) {
            $ti = $s.IndexOf($pat)
            if ($ti -ge 0) { $foundText += "  '$pat' at offset $($pos+$ti)" }
        }
        $pos += $n
    }
    Write-Host "-- magic hits --" -ForegroundColor Yellow
    if ($foundMagic) { $foundMagic | ForEach-Object { Write-Host $_ } } else { Write-Host "  (none)" }
    Write-Host "-- log-text hits --" -ForegroundColor Yellow
    if ($foundText) { $foundText | Select-Object -Unique | ForEach-Object { Write-Host $_ } } else { Write-Host "  (none - no captured boot text on the device)" }
}
finally { $fs.Dispose() }
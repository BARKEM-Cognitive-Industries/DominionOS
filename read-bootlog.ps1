# read-bootlog.ps1 — extract the DominionOS debug boot log from a raw disk image.
#
# After booting DominionOS (in QEMU or on real bare-metal that writes to a recognised
# data disk), the kernel persists its full captured boot/install/run log as plain text
# to the TAIL of the data disk behind an `AELOG001` superblock. This script reads it
# straight out of the raw image so you can hand it over for debugging.
#
# Usage:
#   .\read-bootlog.ps1                       # reads .\dominion-data.img -> .\bootlog.txt
#   .\read-bootlog.ps1 -Image E:\           # a raw USB/disk path
#   .\read-bootlog.ps1 -Image disk.img -Out mylog.txt
param(
    [string]$Image = "dominion-data.img",
    [string]$Out   = "bootlog.txt"
)

$RESERVE = 1024            # sectors reserved at the tail (must match bootlog.rs)
$BS      = 512             # sector size
$MAGIC   = "AELOG001"      # superblock magic (must match LOG_MAGIC)

if (-not (Test-Path $Image)) { Write-Error "image not found: $Image"; exit 1 }
$path = (Resolve-Path $Image).Path
$fs = [System.IO.File]::OpenRead($path)
try {
    $cap = [int64][math]::Floor($fs.Length / $BS)
    if ($cap -le ($RESERVE + 2)) { Write-Error "image too small ($cap sectors) to hold a log"; exit 1 }
    $lba = $cap - $RESERVE
    $off = $lba * $BS

    $fs.Seek($off, [System.IO.SeekOrigin]::Begin) | Out-Null
    $hdr = New-Object byte[] 16
    [void]$fs.Read($hdr, 0, 16)
    $magic = [System.Text.Encoding]::ASCII.GetString($hdr, 0, 8)
    if ($magic -ne $MAGIC) {
        Write-Error "no boot log at LBA $lba (found magic '$magic'). The OS may not have persisted one - run the 'log save' shell command, or boot to the desktop and exit."
        exit 1
    }
    $len = [BitConverter]::ToUInt64($hdr, 8)
    if ($len -le 0 -or $len -gt ($RESERVE * $BS)) { Write-Error "implausible log length: $len"; exit 1 }

    $fs.Seek($off + $BS, [System.IO.SeekOrigin]::Begin) | Out-Null   # payload starts next block
    $buf = New-Object byte[] $len
    $read = 0
    while ($read -lt $len) {
        $n = $fs.Read($buf, $read, [int]($len - $read))
        if ($n -le 0) { break }
        $read += $n
    }
    $text = [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
    Set-Content -Path $Out -Value $text -Encoding utf8

    Write-Output "Extracted $read bytes of DominionOS boot log -> $Out  (from LBA $lba of $path)"
    Write-Output "----------------- tail (last 40 lines) -----------------"
    ($text -split '\r?\n' | Select-Object -Last 40) -join [Environment]::NewLine
}
finally {
    $fs.Close()
}

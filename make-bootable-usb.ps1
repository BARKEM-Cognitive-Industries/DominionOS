<#
.SYNOPSIS
    DominionOS bootable-USB wizard - configure, build, and flash a USB you can boot
    live (or install from) on a real PC, then pull boot/run/benchmark results back
    off the stick.

.DESCRIPTION
    Walks through every configuration option, builds the matching bootable disk
    image with the 'dominion-boot' builder (the same bootloader path run.ps1 uses),
    and - if you ask it to - flashes that image to a physical USB drive so it boots
    on bare metal.

    Boot modes it can build:
      * live      - the interactive graphical desktop + ASH shell (run live / install;
                    the object store and the boot/run log persist to the drive it
                    boots from, so it behaves like an installed system).
      * bench     - the headless real-world benchmark battery. Auto-runs at boot,
                    measures every subsystem on the actual hardware, PERSISTS the
                    results + full log to the USB tail, then halts.
      * validate  - the headless validation battery (memory-latency mountain, soak,
                    chaos/failure-injection). Also persists results to the USB.
      * selftest  - the headless CI self-test battery.

    On bare metal there is no host serial capture, so the kernel writes the full
    boot/install/run log AND every "BENCH ..." line to the TAIL of the USB behind an
    "AELOG001" superblock. After running, plug the USB back into this PC and run
    read-usb-results.ps1 (the wizard offers to do it for you) to extract:
        bootlog-usb.txt      - the complete boot/install/run/benchmark log
        bench-results.json   - the parsed benchmark table

.NOTES
    Flashing a USB requires Administrator. The build and QEMU-test steps do not.
    Run interactively (recommended) or drive it non-interactively with the params.

.EXAMPLE
    .\make-bootable-usb.ps1
        Full interactive wizard.

.EXAMPLE
    .\make-bootable-usb.ps1 -Mode bench -Firmware uefi -Action write -AssumeYes
        Build the benchmark image (UEFI) and flash the one removable USB present.
#>
#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('live', 'safe', 'bench', 'validate', 'selftest')]
    [string]$Mode,

    [ValidateSet('uefi', 'bios', 'both')]
    [string]$Firmware,

    [string]$Resolution,

    [switch]$Fma,

    [switch]$Smp,

    [ValidateSet('build', 'write', 'qemu')]
    [string]$Action,

    [int]$DiskNumber = -1,

    # Reveal fixed (internal) disks as flash targets - for "install to internal
    # disk". OFF by default so the wizard only ever offers removable USBs.
    [switch]$IncludeFixedDisks,

    # Skip the final yes/no confirmations (the UAC prompt + disk identity are still
    # shown). Use only when you are certain of the target.
    [switch]$AssumeYes,

    # Internal: re-entry after elevation so we don't loop forever.
    [switch]$Elevated
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$env:Path = (Join-Path $env:USERPROFILE '.cargo\bin') + ";C:\Program Files\qemu;" + $env:Path

# ----------------------------- pretty output -----------------------------
function Write-Section($t) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host "  $t" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}
function Write-Info($t) { Write-Host "  $t" -ForegroundColor Gray }
function Write-Ok($t) { Write-Host "  [ok] $t" -ForegroundColor Green }
function Write-Warn($t) { Write-Host "  [!]  $t" -ForegroundColor Yellow }
function Write-Err($t) { Write-Host "  [x]  $t" -ForegroundColor Red }
function Write-Step($t) { Write-Host ""; Write-Host ">> $t" -ForegroundColor White }

# Run a native exe (cargo/qemu) without letting its stderr - which carries ordinary
# compiler warnings - trip the script's Stop error preference. Returns the exit code.
function Invoke-Native($exe, $cmdArgs) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    # Out-Host so the command's own stdout streams to the screen and is NOT returned
    # as part of this function's output - the only thing we return is the exit code.
    # No 2>&1: native stderr (compiler warnings) flows straight to the console rather
    # than being wrapped as PowerShell error records.
    try { & $exe @cmdArgs | Out-Host }
    finally { $ErrorActionPreference = $prev }
    return $LASTEXITCODE
}

# ----------------------------- prompt helpers -----------------------------
# Numbered single-choice prompt. $options is an array of @{Key;Label;Detail}.
# Returns the chosen Key. Honors $default (a Key) on empty input.
function Read-Choice($title, $options, $default) {
    Write-Host ""
    Write-Host "  $title" -ForegroundColor White
    for ($i = 0; $i -lt $options.Count; $i++) {
        $o = $options[$i]
        $mark = if ($o.Key -eq $default) { "*" } else { " " }
        Write-Host ("   {0}{1}) {2}" -f $mark, ($i + 1), $o.Label) -ForegroundColor Cyan
        if ($o.Detail) { Write-Host ("        {0}" -f $o.Detail) -ForegroundColor DarkGray }
    }
    while ($true) {
        $defLabel = ($options | Where-Object { $_.Key -eq $default } | Select-Object -First 1).Label
        $ans = Read-Host ("  choose 1-{0} [default: {1}]" -f $options.Count, $defLabel)
        if ([string]::IsNullOrWhiteSpace($ans)) { return $default }
        $n = 0
        if ([int]::TryParse($ans, [ref]$n) -and $n -ge 1 -and $n -le $options.Count) {
            return $options[$n - 1].Key
        }
        $hit = $options | Where-Object { $_.Key -eq $ans } | Select-Object -First 1
        if ($hit) { return $hit.Key }
        Write-Warn "enter a number 1-$($options.Count)."
    }
}

function Read-YesNo($question, $defaultYes) {
    $suffix = if ($defaultYes) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $ans = Read-Host "  $question $suffix"
        if ([string]::IsNullOrWhiteSpace($ans)) { return [bool]$defaultYes }
        switch -regex ($ans.Trim().ToLower()) {
            '^(y|yes)$' { return $true }
            '^(n|no)$' { return $false }
            default { Write-Warn "please answer y or n." }
        }
    }
}

# ----------------------------- environment -----------------------------
function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Elevate {
    Write-Warn "Flashing a USB needs Administrator. Relaunching the wizard elevated..."
    $argList = @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Elevated')
    try {
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $argList | Out-Null
        Write-Info "An elevated window has opened - continue there. This window can be closed."
    }
    catch {
        Write-Err "Elevation was declined. Re-run this script from an elevated PowerShell to flash a USB."
    }
    exit 0
}

function Test-Prereqs {
    Write-Step "Checking the build toolchain"
    $ok = $true
    foreach ($t in @('cargo', 'rustc')) {
        $c = Get-Command $t -ErrorAction SilentlyContinue
        if ($c) { Write-Ok "$t -> $($c.Source)" }
        else { Write-Err "$t not found on PATH"; $ok = $false }
    }
    $q = Get-Command qemu-system-x86_64 -ErrorAction SilentlyContinue
    if ($q) { Write-Ok "qemu-system-x86_64 (optional, for test-boot) -> $($q.Source)" }
    else { Write-Warn "qemu-system-x86_64 not found - the 'test in QEMU first' step will be unavailable." }
    if (-not (Test-Path (Join-Path $root 'kernel\Cargo.toml'))) {
        Write-Err "kernel\Cargo.toml not found under $root - run this from the dominionos repo."
        $ok = $false
    }
    if (-not $ok) { throw "prerequisites missing; aborting." }
}

# ----------------------------- build -----------------------------
# Returns @{ Bios=<path or $null>; Uefi=<path or $null>; Elf=<path> }
function Invoke-Build($mode, $firmware, $resolution, $fma, $smp) {
    $features = @()
    switch ($mode) {
        'bench' { $features += 'qemu_bench' }
        'validate' { $features += 'qemu_validate' }
        'selftest' { $features += 'qemu_test' }
        'safe' { $features += 'safe_mode' }
        'live' { if ($smp) { $features += 'smp' } }
    }

    if ($fma -and ($mode -eq 'live' -or $mode -eq 'selftest')) {
        Write-Warn "FMA only affects the ML benchmark; ignoring it for '$mode'."
        $fma = $false
    }
    if ($fma) {
        $features += 'fma'
        $targetArg = @('--target', 'x86_64-dominion-fma.json')
        $elfTarget = 'x86_64-dominion-fma'
    }
    else {
        $targetArg = @()
        $elfTarget = 'x86_64-dominion'
    }

    $featCsv = ($features -join ',')
    Write-Step "Building the kernel"
    Write-Info ("mode={0}  features=[{1}]  target={2}" -f $mode, $featCsv, $elfTarget)

    Push-Location (Join-Path $root 'kernel')
    try {
        $buildArgs = @('build', '--release') + $targetArg
        if ($featCsv) { $buildArgs += @('--features', $featCsv) }
        $code = Invoke-Native 'cargo' $buildArgs
        if ($code -ne 0) { throw "kernel build failed (cargo exit $code)." }
    }
    finally { Pop-Location }

    $elf = Join-Path $root "kernel\target\$elfTarget\release\dominion-kernel"
    if (-not (Test-Path $elf)) { throw "kernel ELF not produced at $elf" }
    Write-Ok "kernel ELF: $elf"

    $tag = $mode
    if ($fma) { $tag += "-fma" }
    $biosImg = Join-Path $root "dominionos-usb-$tag-bios.img"
    $uefiImg = Join-Path $root "dominionos-usb-$tag-uefi.img"

    Write-Step "Wrapping the kernel into a bootable disk image"
    if ($resolution) { $env:RESOLUTION = $resolution } else { Remove-Item Env:\RESOLUTION -ErrorAction SilentlyContinue }
    $resLabel = if ($resolution) { $resolution } else { 'bootloader default' }
    Write-Info ("resolution={0}  firmware={1}" -f $resLabel, $firmware)

    # dominion-boot signature: <kernel-elf> <out-bios-img> [out-uefi-img]. It always
    # writes the BIOS image; pass a 3rd arg to also get the UEFI image.
    $bootArgs = @('run', '--release', '--', $elf, $biosImg)
    $wantUefi = ($firmware -eq 'uefi' -or $firmware -eq 'both')
    if ($wantUefi) { $bootArgs += $uefiImg }

    Push-Location (Join-Path $root 'boot')
    try {
        $code = Invoke-Native 'cargo' $bootArgs
        if ($code -ne 0) { throw "image build failed (dominion-boot exit $code)." }
    }
    finally { Pop-Location }

    $result = @{ Elf = $elf; Bios = $null; Uefi = $null }
    if (Test-Path $biosImg) {
        $result.Bios = $biosImg
        Write-Ok ("BIOS image: {0} ({1:N1} MiB)" -f $biosImg, ((Get-Item $biosImg).Length / 1MB))
    }
    if ($wantUefi -and (Test-Path $uefiImg)) {
        $result.Uefi = $uefiImg
        Write-Ok ("UEFI image: {0} ({1:N1} MiB)" -f $uefiImg, ((Get-Item $uefiImg).Length / 1MB))
    }
    return $result
}

# ----------------------------- disk discovery -----------------------------
function Get-CandidateDisks($includeFixed) {
    $disks = Get-Disk | Sort-Object Number
    $out = @()
    foreach ($d in $disks) {
        $isUsb = ($d.BusType -eq 'USB')
        $removable = $isUsb -or ($d.BusType -in @('SD', 'MMC'))
        if (-not $includeFixed -and -not $removable) { continue }
        $forbidden = ($d.IsSystem -or $d.IsBoot)
        $out += [pscustomobject]@{
            Number    = $d.Number
            Friendly  = $d.FriendlyName
            BusType   = $d.BusType
            SizeGB    = [math]::Round($d.Size / 1GB, 1)
            Removable = $removable
            Forbidden = $forbidden
        }
    }
    return $out
}

function Select-TargetDisk($includeFixed, $preset) {
    Write-Step "Choosing the target drive"
    $cands = Get-CandidateDisks $includeFixed
    if (-not $cands -or $cands.Count -eq 0) {
        $kindMsg = if ($includeFixed) { 'eligible' } else { 'removable USB' }
        Write-Err "No $kindMsg disks found. Plug in your USB and retry (or pass -IncludeFixedDisks)."
        return $null
    }
    Write-Host ""
    Write-Host ("   {0,-4} {1,-28} {2,-7} {3,8}  {4}" -f '#', 'Model', 'Bus', 'Size', 'Type') -ForegroundColor White
    foreach ($c in $cands) {
        $type = if ($c.Forbidden) { 'SYSTEM/BOOT - refused' } elseif ($c.Removable) { 'removable' } else { 'FIXED (internal)' }
        $color = if ($c.Forbidden) { 'Red' } elseif ($c.Removable) { 'Cyan' } else { 'Yellow' }
        Write-Host ("   {0,-4} {1,-28} {2,-7} {3,6} GB  {4}" -f $c.Number, $c.Friendly, $c.BusType, $c.SizeGB, $type) -ForegroundColor $color
    }
    if ($preset -ge 0) {
        $chosen = $cands | Where-Object { $_.Number -eq $preset } | Select-Object -First 1
        if (-not $chosen) { Write-Err "Disk #$preset is not an eligible target."; return $null }
    }
    else {
        while ($true) {
            $ans = Read-Host "`n  Enter the disk # to flash (or 'q' to cancel)"
            if ($ans -match '^(q|quit)$') { return $null }
            $n = 0
            if ([int]::TryParse($ans, [ref]$n)) {
                $chosen = $cands | Where-Object { $_.Number -eq $n } | Select-Object -First 1
                if ($chosen) { break }
            }
            Write-Warn "pick a # from the list."
        }
    }
    if ($chosen.Forbidden) {
        Write-Err "Disk #$($chosen.Number) is the Windows system/boot disk. Refusing - pick another."
        return $null
    }
    if (-not $chosen.Removable) {
        Write-Warn "Disk #$($chosen.Number) is an INTERNAL/FIXED disk. This is the 'install to internal disk' path."
    }
    return $chosen
}

# ----------------------------- flashing -----------------------------
function Write-ImageToDisk($diskNumber, $imagePath, $assumeYes) {
    if (-not (Test-Admin)) { throw "flashing requires Administrator." }
    $img = Get-Item $imagePath
    $disk = Get-Disk -Number $diskNumber
    if ($disk.IsSystem -or $disk.IsBoot) { throw "refusing to write to the system/boot disk." }

    Write-Section "FINAL CONFIRMATION - this ERASES the target drive"
    Write-Host ("   Target  : disk #{0}  {1}  ({2:N1} GB, {3})" -f $disk.Number, $disk.FriendlyName, ($disk.Size / 1GB), $disk.BusType) -ForegroundColor Yellow
    Write-Host ("   Writing : {0}  ({1:N1} MiB)" -f $img.Name, ($img.Length / 1MB)) -ForegroundColor Yellow
    Write-Host "   ALL existing data on this drive will be destroyed." -ForegroundColor Red
    if (-not $assumeYes) {
        $typed = Read-Host "`n  Type the disk number ($($disk.Number)) to confirm, anything else to cancel"
        if ($typed -ne "$($disk.Number)") { Write-Warn "Cancelled - nothing was written."; return $false }
    }

    Write-Step "Preparing the drive"
    # Wipe the partition table so no volume locks remain, then take the disk offline:
    # an offline disk is not held by the volume manager, so a raw write to the whole
    # device succeeds. We bring it back online in the finally block.
    try { Clear-Disk -Number $diskNumber -RemoveData -RemoveOEM -Confirm:$false -ErrorAction Stop }
    catch { Write-Info "Clear-Disk: $($_.Exception.Message) (continuing)" }
    Set-Disk -Number $diskNumber -IsOffline $true -ErrorAction SilentlyContinue
    Set-Disk -Number $diskNumber -IsReadOnly $false -ErrorAction SilentlyContinue

    $devPath = "\\.\PhysicalDrive$diskNumber"
    Write-Step "Flashing $($img.Name) -> $devPath"
    $src = $null; $dst = $null
    try {
        $src = [System.IO.File]::OpenRead($img.FullName)
        $dst = New-Object System.IO.FileStream($devPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
        $total = $src.Length
        $bufSize = 4MB
        $buf = New-Object byte[] $bufSize
        $written = [int64]0
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        while ($true) {
            $n = $src.Read($buf, 0, $bufSize)
            if ($n -le 0) { break }
            # Raw device writes must be sector-aligned. The image length is a multiple
            # of 512; pad only a short final read up to a 512 boundary.
            if ($n % 512 -ne 0) {
                $pad = 512 - ($n % 512)
                for ($i = 0; $i -lt $pad; $i++) { $buf[$n + $i] = 0 }
                $n += $pad
            }
            $dst.Write($buf, 0, $n)
            $written += $n
            $pct = [int](($written * 100) / $total)
            $mbps = if ($sw.Elapsed.TotalSeconds -gt 0) { ($written / 1MB) / $sw.Elapsed.TotalSeconds } else { 0 }
            Write-Progress -Activity "Flashing $($img.Name)" -Status ("{0:N0} / {1:N0} MiB  ({2:N1} MiB/s)" -f ($written / 1MB), ($total / 1MB), $mbps) -PercentComplete ([math]::Min($pct, 100))
        }
        $dst.Flush()
        Write-Progress -Activity "Flashing" -Completed
        Write-Ok ("Wrote {0:N1} MiB in {1:N1}s" -f ($written / 1MB), $sw.Elapsed.TotalSeconds)
    }
    finally {
        if ($dst) { $dst.Dispose() }
        if ($src) { $src.Dispose() }
        # Bring the disk back so Windows re-reads the new partition table.
        Set-Disk -Number $diskNumber -IsOffline $false -ErrorAction SilentlyContinue
    }
    return $true
}

# ----------------------------- QEMU test-boot -----------------------------
function Find-Ovmf {
    $candidates = @(
        'C:\Program Files\qemu\share\edk2-x86_64-code.fd',
        'C:\Program Files\qemu\share\OVMF_CODE.fd',
        'C:\Program Files\qemu\share\OVMF.fd'
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

function Test-BootInQemu($imagePath, $firmware) {
    if (-not (Get-Command qemu-system-x86_64 -ErrorAction SilentlyContinue)) {
        Write-Warn "QEMU not available; skipping test-boot."
        return
    }
    . (Join-Path $root 'qemu-common.ps1')
    $accel = Get-DominionAccel
    $qargs = @('-accel', $accel, '-m', '4096', '-vga', 'std',
        '-drive', "format=raw,file=$imagePath",
        '-serial', 'stdio')
    if ($firmware -eq 'uefi') {
        $ovmf = Find-Ovmf
        if (-not $ovmf) {
            Write-Warn "No OVMF UEFI firmware found in the QEMU share dir; cannot test the UEFI image in QEMU."
            Write-Info "Flash it and boot on real UEFI hardware, or test the BIOS image instead."
            return
        }
        $qargs += @('-drive', "if=pflash,format=raw,readonly=on,file=$ovmf")
    }
    Write-Step "Booting the image in QEMU (accel=$accel). Close the window to continue."
    [void](Invoke-Native 'qemu-system-x86_64' $qargs)
}

# ----------------------------- manifest -----------------------------
function Save-Manifest($cfg, $images, $flashed) {
    $doc = [ordered]@{
        created      = (Get-Date).ToString('s')
        mode         = $cfg.Mode
        firmware     = $cfg.Firmware
        resolution   = $cfg.Resolution
        fma          = [bool]$cfg.Fma
        smp          = [bool]$cfg.Smp
        bios_image   = $images.Bios
        uefi_image   = $images.Uefi
        flashed_disk = $flashed
    }
    $path = Join-Path $root 'usb-build-manifest.json'
    $doc | ConvertTo-Json -Depth 5 | Out-File -FilePath $path -Encoding utf8
    Write-Ok "Saved build manifest -> $path"
}

# ============================== main ==============================
Clear-Host
Write-Section "DominionOS - Bootable USB Wizard"
Write-Info "Build a USB you can boot live or install from, benchmark on real hardware,"
Write-Info "and pull the logs + benchmark results back off afterwards."
if ($Elevated) { Write-Ok "Running elevated (USB flashing enabled)." }

Test-Prereqs

# Interactive when either Mode or Action was not supplied as a parameter.
$interactive = -not ($Mode -and $Action)

if (-not $Mode) {
    $Mode = Read-Choice "What should this USB do when it boots?" @(
        @{Key = 'live'; Label = 'Live OS - interactive desktop + shell (run live / install)'; Detail = 'Boots the graphical desktop. Object store + logs persist to the drive.' }
        @{Key = 'safe'; Label = 'Safe mode - text REPL, no desktop (recovery / diagnostics)'; Detail = 'Skips the desktop + GPU buffers; shows boot stages on screen and drops to the ASH shell. Use if the desktop will not come up.' }
        @{Key = 'bench'; Label = 'Benchmark battery - auto-run, save results to USB, halt'; Detail = 'Measures every subsystem on the bare metal, persists results to the USB tail.' }
        @{Key = 'validate'; Label = 'Validation battery - memory mountain / soak / chaos'; Detail = 'Stress + correctness battery; also persists results to the USB.' }
        @{Key = 'selftest'; Label = 'Self-test battery - headless CI checks'; Detail = 'Quick pass/fail bring-up battery.' }
    ) 'live'
}

if (-not $Firmware) {
    $Firmware = Read-Choice "Which firmware will the target PC boot with?" @(
        @{Key = 'uefi'; Label = 'UEFI (recommended - almost all PCs since ~2012)'; Detail = 'Disable Secure Boot in firmware first: the bootloader is unsigned.' }
        @{Key = 'bios'; Label = 'Legacy BIOS / CSM (older machines)'; Detail = 'Use if the PC has no UEFI or you boot in CSM/legacy mode.' }
        @{Key = 'both'; Label = 'Build both images (decide which to flash later)'; Detail = 'Produces a BIOS and a UEFI image; you flash one.' }
    ) 'uefi'
}

if (-not $Resolution) {
    if ($Mode -eq 'live') {
        if ($interactive) {
            $resKey = Read-Choice "Framebuffer resolution for the desktop?" @(
                @{Key = '1920x1080'; Label = '1920 x 1080 (Full HD)'; Detail = '' }
                @{Key = '1280x720'; Label = '1280 x 720 (HD - safer/faster on odd panels)'; Detail = '' }
                @{Key = '2560x1440'; Label = '2560 x 1440 (QHD)'; Detail = '' }
                @{Key = 'default'; Label = 'Firmware default (let the OS adapt)'; Detail = 'Most compatible if a fixed mode misbehaves.' }
            ) '1920x1080'
            $Resolution = if ($resKey -eq 'default') { $null } else { $resKey }
        }
        else {
            $Resolution = '1920x1080'
        }
    }
    elseif ($Mode -eq 'bench' -or $Mode -eq 'validate') {
        $Resolution = '1280x720'
    }
}

if ($interactive -and ($Mode -eq 'bench' -or $Mode -eq 'validate') -and -not $Fma) {
    $Fma = Read-YesNo "Build the opt-in AVX/FMA ML kernel? (faster matmul, NON-deterministic)" $false
}
if ($interactive -and $Mode -eq 'live' -and -not $Smp) {
    $Smp = Read-YesNo "Enable real multi-core (SMP) bring-up for the live OS?" $true
}

if (-not $Action) {
    $Action = Read-Choice "What do you want to do now?" @(
        @{Key = 'write'; Label = 'Build AND flash to a USB drive (needs Admin)'; Detail = 'The end-to-end path: build, confirm the drive, write it.' }
        @{Key = 'qemu'; Label = 'Build AND test-boot in QEMU first'; Detail = 'Verify the image boots before touching a real drive.' }
        @{Key = 'build'; Label = 'Build the image only (no flashing)'; Detail = 'Produces the .img file(s); flash later.' }
    ) 'write'
}

Write-Section "Configuration"
$resShow = if ($Resolution) { $Resolution } else { 'firmware default' }
Write-Info ("Boot mode    : {0}" -f $Mode)
Write-Info ("Firmware     : {0}" -f $Firmware)
Write-Info ("Resolution   : {0}" -f $resShow)
Write-Info ("AVX/FMA      : {0}" -f $Fma)
Write-Info ("SMP (live)   : {0}" -f $Smp)
Write-Info ("Action       : {0}" -f $Action)
if ($interactive) {
    if (-not (Read-YesNo "Proceed with this configuration?" $true)) { Write-Warn "Cancelled."; exit 0 }
}

# Only the 'write' action needs admin.
if ($Action -eq 'write' -and -not (Test-Admin)) {
    if ($Elevated) { throw "expected to be elevated but am not." }
    Invoke-Elevate
}

# -- build --
$cfg = @{ Mode = $Mode; Firmware = $Firmware; Resolution = $Resolution; Fma = $Fma; Smp = $Smp }
$images = Invoke-Build $Mode $Firmware $Resolution $Fma $Smp

# Pick which image the flash/test will use.
$flashImage = $null
if ($Firmware -eq 'bios') { $flashImage = $images.Bios }
elseif ($Firmware -eq 'uefi') { $flashImage = $images.Uefi }
else {
    if ($Action -ne 'build') {
        $fw = Read-Choice "You built both - which image for this $Action?" @(
            @{Key = 'uefi'; Label = 'UEFI image' }
            @{Key = 'bios'; Label = 'BIOS image' }
        ) 'uefi'
        $flashImage = if ($fw -eq 'uefi') { $images.Uefi } else { $images.Bios }
        $Firmware = $fw
    }
}

$flashed = $null
switch ($Action) {
    'build' {
        Write-Section "Done - images built"
        if ($images.Bios) { Write-Info "BIOS: $($images.Bios)" }
        if ($images.Uefi) { Write-Info "UEFI: $($images.Uefi)" }
        Write-Info "Flash later with:  .\make-bootable-usb.ps1 -Mode $Mode -Firmware $Firmware -Action write"
    }
    'qemu' {
        Test-BootInQemu $flashImage $Firmware
        if (Read-YesNo "Flash this image to a USB now?" $false) {
            if (-not (Test-Admin)) { Invoke-Elevate }
            $disk = Select-TargetDisk $IncludeFixedDisks $DiskNumber
            if ($disk) { if (Write-ImageToDisk $disk.Number $flashImage $AssumeYes) { $flashed = $disk.Number } }
        }
    }
    'write' {
        $disk = Select-TargetDisk $IncludeFixedDisks $DiskNumber
        if ($disk) { if (Write-ImageToDisk $disk.Number $flashImage $AssumeYes) { $flashed = $disk.Number } }
    }
}

Save-Manifest $cfg $images $flashed

# -- after-flash guidance --
if ($null -ne $flashed) {
    Write-Section "Flashed - next steps"
    Write-Info "1. Plug the USB into the target PC."
    Write-Info "2. Open the boot menu (often F12/F11/F9/Esc) and pick the USB."
    if ($Firmware -eq 'uefi') { Write-Info "   UEFI: turn OFF Secure Boot first (the bootloader is unsigned)." }
    if ($Mode -eq 'live') {
        Write-Info "3. The desktop comes up. Press Esc for the ASH shell; 'log save' forces a log flush."
        Write-Info "   Power off cleanly so the boot/run log is persisted to the USB tail."
    }
    else {
        Write-Info "3. The $Mode battery runs headless and persists results to the USB, then halts."
        Write-Info "   Wait until the screen stops updating / the machine halts, then power off."
    }
    Write-Info "4. Bring the USB back here and extract everything it captured:"
    Write-Host  "       .\read-usb-results.ps1 -Disk $flashed" -ForegroundColor Green
    Write-Info  "   -> bootlog-usb.txt (full boot/install/run/benchmark log)"
    if ($Mode -eq 'bench' -or $Mode -eq 'validate') { Write-Info "   -> bench-results.json (parsed benchmark table)" }

    if (Read-YesNo "`n  Extract results from disk $flashed now? (only if you already ran it)" $false) {
        & (Join-Path $root 'read-usb-results.ps1') -Disk $flashed
    }
}
Write-Section "Wizard complete"

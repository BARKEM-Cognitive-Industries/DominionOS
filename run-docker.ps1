# DominionOS in Docker (Windows / PowerShell).
# ----------------------------------------------------------------------------
# Builds the Docker image (Dockerfile) and then, depending on the command:
#   test  (default) : run the dominion-core host unit-test suite (1000+ tests)
#   build           : build the kernel + assemble dominionos.img (BIOS) + .efi.img (UEFI)
#   boot            : build (if needed) then boot dominionos.img headless in QEMU (serial)
#   shell           : drop into an interactive shell in the toolchain container
#
# Usage:
#   .\run-docker.ps1                 # == .\run-docker.ps1 test
#   .\run-docker.ps1 build
#   .\run-docker.ps1 boot
#   .\run-docker.ps1 boot -RamMib 2048   # raise RAM (needed for big_heap builds)
#   .\run-docker.ps1 shell
#
# Notes:
#   * Produced images land in .\out (bind-mounted), so you can import them into
#     VirtualBox / VMware / Hyper-V on the host afterwards (see docs).
#   * QEMU runs headless (serial to stdout). DominionOS is a single cooperative
#     core -> one vCPU. Hardware accel inside Docker Desktop (WSL2) is generally
#     unavailable, so this uses TCG; for fast iteration use the native run.ps1.
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('test', 'build', 'boot', 'shell')]
    [string]$Command = 'test',

    # RAM for `boot`. Default 512 MiB; raise to >= 1280 for big_heap builds.
    [int]$RamMib = 512
)

$ErrorActionPreference = 'Stop'
$image = 'dominionos:dev'
$root  = $PSScriptRoot
$out   = Join-Path $root 'out'
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

Write-Host "==> Building Docker image ($image) ..."
docker build -t $image $root
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

switch ($Command) {
    'test' {
        Write-Host "==> Running dominion-core host test suite ..."
        docker run --rm $image `
            cargo test --manifest-path dominion-core/Cargo.toml --release
    }

    'build' {
        Write-Host "==> Building kernel + assembling bootable images into .\out ..."
        # Build the release kernel, then run the bootloader image builder to emit
        # both a BIOS image (dominionos.img) and a UEFI image (dominionos.efi.img).
        docker run --rm -v "${out}:/out" $image bash -c @'
set -e
cd /dominionos/kernel && cargo build --release
cd /dominionos/boot && cargo run --release -- \
  /dominionos/kernel/target/x86_64-dominion/release/dominion-kernel \
  /out/dominionos.img /out/dominionos.efi.img
echo "Built /out/dominionos.img (BIOS) and /out/dominionos.efi.img (UEFI)"
'@
    }

    'boot' {
        if (-not (Test-Path (Join-Path $out 'dominionos.img'))) {
            Write-Host "==> dominionos.img not found; building first ..."
            & $PSCommandPath build
        }
        Write-Host "==> Booting dominionos.img in QEMU (headless, $RamMib MiB, serial->stdout) ..."
        Write-Host "    (Ctrl-A then X to quit QEMU.)"
        docker run --rm -it -v "${out}:/out" $image `
            qemu-system-x86_64 `
                -cpu qemu64,+rdrand `
                -smp 1 `
                -m $RamMib `
                -drive "format=raw,file=/out/dominionos.img" `
                -serial stdio `
                -display none `
                -no-reboot
    }

    'shell' {
        docker run --rm -it -v "${out}:/out" $image bash
    }
}

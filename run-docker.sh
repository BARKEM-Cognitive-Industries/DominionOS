#!/usr/bin/env bash
# DominionOS in Docker (Linux / macOS).
# ----------------------------------------------------------------------------
# Builds the Docker image (Dockerfile) and then, depending on the command:
#   test  (default) : run the dominion-core host unit-test suite (1000+ tests)
#   build           : build the kernel + assemble dominionos.img (BIOS) + .efi.img (UEFI)
#   boot            : build (if needed) then boot dominionos.img headless in QEMU (serial)
#   shell           : drop into an interactive shell in the toolchain container
#
# Usage:
#   ./run-docker.sh                 # == ./run-docker.sh test
#   ./run-docker.sh build
#   ./run-docker.sh boot
#   ./run-docker.sh boot 2048       # boot with 2048 MiB RAM (needed if built with big_heap)
#   ./run-docker.sh shell
#
# Notes:
#   * Produced images land in ./out (bind-mounted), so you can copy them into
#     VirtualBox/VMware/Hyper-V on the host afterwards.
#   * QEMU runs headless (-display none, serial to stdout). DominionOS is a single
#     cooperative core, so we use one vCPU. KVM is used if /dev/kvm is present.
set -euo pipefail

IMAGE="dominionos:dev"
CMD="${1:-test}"
RAM_MIB="${2:-512}"          # default RAM; raise to >=1280 for big_heap builds
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/out"
mkdir -p "$OUT"

echo "==> Building Docker image ($IMAGE) ..."
docker build -t "$IMAGE" "$ROOT"

# KVM passthrough if the host exposes it (near-native speed); else QEMU uses TCG.
KVM_ARGS=()
if [ -e /dev/kvm ]; then
  KVM_ARGS=(--device /dev/kvm)
fi

case "$CMD" in
  test)
    echo "==> Running dominion-core host test suite ..."
    docker run --rm "$IMAGE" \
      cargo test --manifest-path dominion-core/Cargo.toml --release
    ;;

  build)
    echo "==> Building kernel + assembling bootable images into ./out ..."
    # Build the release kernel, then run the bootloader image builder to emit
    # both a BIOS image (dominionos.img) and a UEFI image (dominionos.efi.img).
    docker run --rm -v "$OUT:/out" "$IMAGE" bash -c '
      set -e
      cd /dominionos/kernel && cargo build --release
      cd /dominionos/boot && cargo run --release -- \
        /dominionos/kernel/target/x86_64-dominion/release/dominion-kernel \
        /out/dominionos.img /out/dominionos.efi.img
      echo "Built /out/dominionos.img (BIOS) and /out/dominionos.efi.img (UEFI)"
    '
    ;;

  boot)
    # Build if the image is missing, then boot the BIOS image headless.
    if [ ! -f "$OUT/dominionos.img" ]; then
      echo "==> dominionos.img not found; building first ..."
      "$0" build
    fi
    echo "==> Booting dominionos.img in QEMU (headless, ${RAM_MIB} MiB, serial->stdout) ..."
    echo "    (Ctrl-A then X to quit QEMU.)"
    docker run --rm -it "${KVM_ARGS[@]}" -v "$OUT:/out" "$IMAGE" \
      qemu-system-x86_64 \
        -cpu qemu64,+rdrand \
        -smp 1 \
        -m "$RAM_MIB" \
        -drive "format=raw,file=/out/dominionos.img" \
        -serial stdio \
        -display none \
        -no-reboot
    ;;

  shell)
    docker run --rm -it -v "$OUT:/out" "$IMAGE" bash
    ;;

  *)
    echo "unknown command: $CMD" >&2
    echo "usage: $0 [test|build|boot [ram_mib]|shell]" >&2
    exit 2
    ;;
esac

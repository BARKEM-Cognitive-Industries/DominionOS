# DominionOS reproducible build / test / boot environment.
# ----------------------------------------------------------------------------
# This image gives you a self-contained toolchain that can:
#   (a) run the dominion-core host unit-test suite (1000+ tests, links std),
#   (b) build the bare-metal kernel and assemble the bootable BIOS + UEFI
#       disk images via the `bootloader` 0.11 crate, and
#   (c) boot the produced image headless in QEMU (serial -> stdout).
#
# WHY a single image: the kernel needs the *nightly* toolchain plus `rust-src`
# and `llvm-tools-preview` (it builds `core`/`alloc` from source for the custom
# `x86_64-dominion.json` hard-float target — see kernel/.cargo/config.toml). The
# nightly toolchain is a strict superset of what dominion-core's host tests need,
# so one toolchain covers both build paths. We install qemu-system-x86 so the
# same image can also boot the artifact it just produced.
#
# Base: rust:1-bookworm gives a known-good Debian + rustup layout; we then add a
# PINNED nightly via rust-toolchain.toml (see below) so builds are reproducible.
# ----------------------------------------------------------------------------

# Pin the base to a Debian release + Rust major line. (Bookworm = Debian 12.)
FROM rust:1-bookworm

# --- OS packages -------------------------------------------------------------
# qemu-system-x86  : boot the produced image (BIOS via SeaBIOS, UEFI via OVMF).
# ovmf             : UEFI firmware blob, so the UEFI image can be booted too.
# python3          : optional helper (repo ships ppm2png.py for screenshots).
# The rest are standard build prerequisites for linking host test binaries.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        qemu-system-x86 \
        ovmf \
        python3 \
        ca-certificates \
        pkg-config \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Rust toolchain ----------------------------------------------------------
# We do NOT hard-code a nightly date here: the kernel's kernel/rust-toolchain.toml
# already declares `channel = "nightly"` plus the required components
# (rust-src, llvm-tools-preview). rustup honours that file automatically the
# first time a cargo command runs inside kernel/. To make the image build
# self-contained (so `docker build` fails fast if nightly is unavailable rather
# than at `docker run` time), we pre-install a nightly with the needed bits.
#
# For a *fully* pinned build, override NIGHTLY at build time, e.g.:
#   docker build --build-arg NIGHTLY=nightly-2026-01-15 -t dominionos .
ARG NIGHTLY=nightly
RUN rustup toolchain install ${NIGHTLY} \
        --component rust-src llvm-tools-preview \
    && rustup default stable

WORKDIR /dominionos

# Copy the build inputs. .dockerignore keeps target/, *.img, .git and logs out,
# so this stays small and the cache stays warm across source-only edits.
COPY . .

# Pre-fetch dependencies so an offline / repeat `docker run` does not re-download.
# (Best-effort: ignored if the network is unavailable at build time.)
RUN cargo fetch --manifest-path dominion-core/Cargo.toml || true \
    && cargo fetch --manifest-path boot/Cargo.toml || true

# Default entrypoint: run the host test suite. Override the command to build /
# boot the OS — see run-docker.sh / run-docker.ps1, which call:
#   build : cargo build --release in kernel/, then cargo run -p dominion-boot
#   boot  : qemu-system-x86_64 ... -drive format=raw,file=dominionos.img -serial stdio
CMD ["cargo", "test", "--manifest-path", "dominion-core/Cargo.toml", "--release"]

# Development Setup & Compilation

Getting DominionOS built and running on your machine.

---

## System Requirements

### Minimum
- **OS:** Windows 10+, macOS 11+, or Linux (recent distro)
- **CPU:** Intel/AMD x86-64 (2+ cores)
- **RAM:** 8 GB
- **Disk:** 20 GB free for builds and disk images

### Recommended
- **OS:** Linux (Ubuntu 22.04+) or Windows 11 with WSL2
- **CPU:** Intel i5/i7 or AMD Ryzen 5+ (4+ cores)
- **RAM:** 16+ GB (builds are parallel and cache-heavy)
- **Disk:** SSD recommended

### For Testing
- **QEMU 7.0+** (hypervisor for testing)
- **KVM** (Linux; for hardware acceleration)
- **Hyper-V** (Windows; for nested virtualization)

---

## Install Dependencies

### Windows (PowerShell)

```powershell
# Install Rust (if not already installed)
$webClient = New-Object System.Net.WebClient
$webClient.DownloadFile("https://win.rustup.rs", "$env:TEMP\rustup-init.exe")
& "$env:TEMP\rustup-init.exe" -y

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Install required tools
cargo install cargo-bootimage

# Install QEMU (optional but recommended)
# Via Chocolatey: choco install qemu
# Or download from https://www.qemu.org/download/
```

### macOS

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install tools
cargo install cargo-bootimage

# Install QEMU
brew install qemu

# Set up KVM (macOS uses Hypervisor.framework)
# No additional setup needed
```

### Linux (Ubuntu/Debian)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install build dependencies
sudo apt-get install build-essential llvm-dev lld

# Install tools
cargo install cargo-bootimage

# Install QEMU with KVM
sudo apt-get install qemu-system-x86 qemu-kvm

# Add your user to kvm group (for non-root QEMU)
sudo usermod -aG kvm $USER
```

---

## Build DominionOS

### Clone the Repository

```bash
git clone https://github.com/cognitive-industries/dominionos.git
cd dominionos
```

### Check Your Environment

```bash
# Verify Rust is installed
rustc --version
cargo --version

# Verify cargo-bootimage is available
cargo bootimage --help
```

### Build the Kernel & Core

```bash
# Build in debug mode (fast compilation, slower runtime)
cargo build

# Build in release mode (slow compilation, faster runtime)
cargo build --release
```

**First build?** Expect 3-5 minutes (debug) or 10-15 minutes (release) as dependencies are compiled.

**Subsequent builds?** 10-30 seconds (debug) or 1-2 minutes (release) depending on changes.

### Create Bootable Images

```bash
# Create BIOS boot image (for QEMU and real hardware)
cargo bootimage --release

# Output files:
#   dominionos.img    (raw disk image, ~256 MB)
#   dominionos.iso    (ISO 9660 image, for CD/USB)
```

---

## Run the OS

### In QEMU (Recommended)

```bash
# Windows (PowerShell)
.\run.ps1

# Linux/macOS
./run.ps1
# or
cargo run --release
```

**First boot?** Takes 10-20 seconds. You'll see:
1. BIOS boot sequence
2. Kernel initialization messages
3. Framebuffer console
4. Safe-mode shell prompt (ASH)

**At the prompt, try:**
```
help              # Show available commands
time              # Print system time
ml                # Run ML benchmark
list              # List objects in the world
world             # Show system state
```

### Customize QEMU Options

Edit `run.ps1` or `run.sh` to change:
- Number of CPU cores (`-smp`)
- RAM amount (`-m`)
- Graphics output (`-display`)
- Networking (`-net`)

Example:
```bash
qemu-system-x86_64 \
  -drive file=dominionos.img,format=raw \
  -smp 8 \                     # 8 cores instead of 4
  -m 8G \                       # 8 GB RAM instead of 4
  -net nic,model=virtio \
  -net user,hostfwd=tcp::8000-:80
```

### On Real Hardware (Bare Metal)

**WARNING:** Do this only on expendable hardware or in a VM. DominionOS is experimental.

#### Create Bootable USB

**Windows (PowerShell):**
```powershell
# List USB drives
Get-Disk | Where-Object {$_.BusType -eq "USB"}

# Create bootable USB (example: F: drive)
# DESTRUCTIVE: This will erase the USB drive
.\make-bootable-usb.ps1 -ImagePath dominionos.iso -USBDrive "F:"
```

**Linux/macOS:**
```bash
# List disk devices
lsblk                          # Linux
diskutil list                  # macOS

# Write image to USB (example: /dev/sdb)
# DESTRUCTIVE: This will erase the USB drive
sudo dd if=dominionos.iso of=/dev/sdb bs=4M status=progress
sudo sync
```

#### Boot from USB
1. Insert USB drive
2. Power on machine
3. Enter BIOS/UEFI boot menu (usually F12, DEL, or ESC during startup)
4. Select USB drive
5. Boot

**What to expect:**
- Boot takes 5-15 seconds (faster than QEMU)
- You reach the ASH prompt
- Hardware support varies (see `docs/HARDWARE.md`)

---

## Test Your Build

### Run the Test Suite

```bash
# Run all unit tests
cargo test --release

# Run tests for a specific subsystem
cargo test --release firewall      # Test firewall module
cargo test --release ml            # Test ML engine

# Run tests on the booted OS
cargo bootimage --release
# Boot in QEMU
./run.ps1
# In the shell: test
```

### Run Benchmarks

```bash
# Run performance benchmarks
cargo run --release --bin bench

# This tests:
# - Boot time
# - Memory I/O
# - Crypto operations (SHA-256, ChaCha20, etc.)
# - ML inference & training
# - Process isolation
# - Storage commits

# Benchmarks report time in seconds, throughput in MB/s or ops/s
```

---

## Troubleshooting

### Build fails with "cargo-bootimage not found"
```bash
cargo install cargo-bootimage --force
```

### Build fails with LLVM errors
Update your Rust toolchain:
```bash
rustup update
rustup component add rust-src
```

### QEMU doesn't boot
- Ensure `dominionos.img` exists and is ~256 MB
- Try adding `-bios /path/to/bios.bin` (platform-specific)
- On Linux, check KVM is enabled: `kvm-ok`

### QEMU is very slow
- Disable graphics: change `-display gtk` to `-display none`
- Enable KVM acceleration: add `-enable-kvm` (Linux only)
- Reduce cores/memory if resource-constrained

### "No such file or directory: ./run.ps1"
- On Linux/macOS, use `cargo run --release` instead
- Or make the script executable: `chmod +x run.ps1`

### Terminal is unresponsive in QEMU
- Click the QEMU window first to give it input focus
- Or reboot and try with fewer cores: `cargo run -- -smp 2`

---

## Development Workflow

### Typical Loop

```bash
# 1. Make changes to code
# (edit dominion-core/src/*.rs or kernel/src/*.rs)

# 2. Test locally
cargo test --release

# 3. Build bootable image
cargo bootimage --release

# 4. Run in QEMU
./run.ps1

# 5. Verify your feature works
# (type commands in the shell)

# 6. Commit and submit PR
git add .
git commit -m "[subsystem] Your change"
git push origin feature/your-feature
```

### Checking Build Variants

We support multiple build profiles:

```bash
# Debug (fast build, slow runtime)
cargo build

# Release (slow build, fast runtime)
cargo build --release

# Safe mode (additional runtime checks)
cargo build --release --features safe

# Benchmark mode (optimized for latency measurement)
cargo build --release --features bench
```

---

## Architecture Walkthrough

### File Structure

```
dominionos/
├── kernel/                 # Microkernel (bootloader → scheduler → drivers)
│   └── src/
│       ├── main.rs         # Boot entry point
│       ├── memory.rs       # Virtual memory
│       ├── interrupt.rs    # IRQ handlers
│       └── ...
├── dominion-core/          # Core library (capability system, storage, lang)
│   └── src/
│       ├── capability.rs   # Unforgeable tokens
│       ├── firewall.rs     # Authority enforcement
│       ├── storage.rs      # Object graph
│       ├── crypto.rs       # Cryptography
│       ├── ml.rs           # Neural networks
│       └── ... (160+ modules)
├── docs/                   # Specifications and guides
│   ├── architecture.md     # System design
│   ├── subsystem-manifest.json  # Module map
│   └── ...
└── Cargo.toml              # Build manifest
```

### Compiling Just the Kernel

```bash
cd kernel
cargo build --release
```

### Compiling Just dominion-core

```bash
cd dominion-core
cargo test --release
```

---

## Debugging

### Print Debug Output

```rust
// In your code
println!("Debug: value = {}", some_var);

// Output appears on the console when booted in QEMU
```

### Enable Verbose Logging

```bash
# Build with debug logs
RUST_LOG=debug cargo run --release

# Boot and check kernel output
```

### Examine Boot Logs

After QEMU closes, check:
```bash
cat bootlog.txt      # If it was generated
```

---

## Cross-Compiling

To build for different architectures:

```bash
# Build for aarch64 (ARM 64-bit)
rustup target add aarch64-unknown-none
cargo build --release --target aarch64-unknown-none

# Build for riscv64 (RISC-V 64-bit)
rustup target add riscv64imac-unknown-none-elf
cargo build --release --target riscv64imac-unknown-none-elf
```

**Note:** Bootloader and kernel need architecture-specific adjustments. The above builds dominion-core (the portable library).

---

## Performance Tips

### Faster Builds
```bash
# Use mold linker (faster on Linux)
sudo apt-get install mold
RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo build --release

# Use parallel builds
export CARGO_BUILD_JOBS=8
cargo build --release
```

### Faster Tests
```bash
# Run tests in parallel
cargo test --release -- --test-threads=8
```

### Faster Benchmarks
```bash
# Run specific benchmark
cargo run --release --bin bench -- firewall
```

---

## Next Steps

1. **Read architecture:** `docs/architecture.md`
2. **Find your area:** `docs/subsystem-manifest.json`
3. **Make a change:** Pick a subsystem and start hacking
4. **Test locally:** `cargo test --release`
5. **Boot and verify:** `./run.ps1`
6. **Submit PR:** See `CONTRIBUTING.md`

---

## Questions?

- **Build errors:** Check Rust version (`rustc --version`) and LLVM (`llvm-config --version`)
- **Runtime errors:** See `HARDWARE.md` for known hardware issues
- **General:** contact@cognitive-industries.org

**Happy hacking!**

//! Windows / macOS / Linux application compatibility (see
//! `docs/implementation/windows-mac-app-support.md`).
//!
//! Foreign apps run the way every untrusted thing runs on this OS — behind an
//! **airlock**. A guest executable is identified by its binary format, its native
//! ABI is mapped through a **personality** that translates each foreign syscall to
//! a host capability operation (or denies it), and it executes against a
//! **projected filesystem** and bounded memory. The host kernel, object graph and
//! Dominion runtime are never in the guest's reach — it sees only the capabilities
//! the personality grants.
//!
//! This module is the *policy and translation* layer (pure, safe `no_std`); actual
//! instruction execution is delegated to the [`crate::wasm`] sandbox or a JIT in
//! the kernel. No special hardware required.

use alloc::string::String;
use alloc::vec::Vec;

/// Executable container formats the loader recognises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryFormat {
    /// Windows PE/COFF (`MZ`).
    Pe,
    /// Apple Mach-O (32/64/fat).
    MachO,
    /// ELF (Linux / Dominion-native).
    Elf,
    Unknown,
}

/// Detect a binary's container format from its magic bytes.
pub fn detect_format(bytes: &[u8]) -> BinaryFormat {
    if bytes.len() < 4 {
        return BinaryFormat::Unknown;
    }
    match bytes {
        [0x4D, 0x5A, ..] => BinaryFormat::Pe, // "MZ"
        [0x7F, b'E', b'L', b'F', ..] => BinaryFormat::Elf,
        // Mach-O thin (LE/BE 32/64) and fat/universal.
        [0xCF, 0xFA, 0xED, 0xFE, ..]
        | [0xCE, 0xFA, 0xED, 0xFE, ..]
        | [0xFE, 0xED, 0xFA, 0xCF, ..]
        | [0xFE, 0xED, 0xFA, 0xCE, ..]
        | [0xCA, 0xFE, 0xBA, 0xBE, ..] => BinaryFormat::MachO,
        _ => BinaryFormat::Unknown,
    }
}

/// The native ABI a guest expects, which selects its syscall personality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    Win64,
    MacOsArm64,
    Linux,
}

/// A host capability operation a translated syscall maps to. The personality can
/// only ever produce one of these — the guest cannot invent new authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostOp {
    OpenFile,
    ReadFile,
    WriteFile,
    CloseFile,
    AllocMemory,
    FreeMemory,
    GetTime,
    Exit,
    /// Recognised but deliberately refused (e.g. raw device / ptrace / mount).
    Denied,
}

impl HostOp {
    /// A stable bit index for this op, so an allow-list can be a single-word bitmask
    /// instead of a `Vec` walked on every syscall. `Denied` deliberately has no bit —
    /// it can never be admitted, matching the default-closed gate.
    #[inline]
    fn bit(self) -> u16 {
        match self {
            HostOp::OpenFile => 1 << 0,
            HostOp::ReadFile => 1 << 1,
            HostOp::WriteFile => 1 << 2,
            HostOp::CloseFile => 1 << 3,
            HostOp::AllocMemory => 1 << 4,
            HostOp::FreeMemory => 1 << 5,
            HostOp::GetTime => 1 << 6,
            HostOp::Exit => 1 << 7,
            HostOp::Denied => 0,
        }
    }
}

/// Translate a foreign syscall number to a host operation for the given ABI.
/// Anything not in the table is `Denied` — default-closed, like the firewall.
///
/// Each ABI arm is a `match` over a small, fixed set of ordinals that the compiler
/// lowers to a jump table / branchless compare chain — O(1) per call, no lookup
/// structure to walk. Inlined so the dispatcher in [`crate::capshim`] fuses this
/// translation into its own `HostOp` match (one combined branch on the hot path).
#[inline]
pub fn translate_syscall(abi: Abi, number: u64) -> HostOp {
    match abi {
        Abi::Linux => match number {
            0 => HostOp::ReadFile,
            1 => HostOp::WriteFile,
            2 => HostOp::OpenFile,
            3 => HostOp::CloseFile,
            9 => HostOp::AllocMemory, // mmap
            11 => HostOp::FreeMemory, // munmap
            201 => HostOp::GetTime,   // time
            60 => HostOp::Exit,
            _ => HostOp::Denied,
        },
        Abi::Win64 => match number {
            // Illustrative NT-style ordinals.
            0x55 => HostOp::OpenFile,  // NtCreateFile
            0x06 => HostOp::ReadFile,  // NtReadFile
            0x08 => HostOp::WriteFile, // NtWriteFile
            0x0F => HostOp::CloseFile, // NtClose
            0x18 => HostOp::AllocMemory,
            0x1E => HostOp::FreeMemory,
            0x5A => HostOp::GetTime,
            0x29 => HostOp::Exit, // NtTerminateProcess
            _ => HostOp::Denied,
        },
        Abi::MacOsArm64 => match number {
            5 => HostOp::OpenFile,
            3 => HostOp::ReadFile,
            4 => HostOp::WriteFile,
            6 => HostOp::CloseFile,
            197 => HostOp::AllocMemory, // mmap
            73 => HostOp::FreeMemory,   // munmap
            116 => HostOp::GetTime,     // gettimeofday
            1 => HostOp::Exit,
            _ => HostOp::Denied,
        },
    }
}

/// A foreign process's confined world: which host ops it may invoke, the root its
/// file paths are projected under, and its memory ceiling.
pub struct ForeignProcess {
    pub abi: Abi,
    allowed: Vec<HostOp>,
    /// O(1) acceleration mirror of `allowed`: bit `HostOp::bit()` set iff allowed.
    /// The hot syscall gate tests one bit instead of scanning the `Vec` each call.
    allowed_mask: u16,
    fs_root: String,
    mem_limit: usize,
    mem_used: usize,
}

impl ForeignProcess {
    /// Create a process confined to `fs_root` with a memory ceiling and a default
    /// allow-list of benign ops. (Add more with [`allow`](Self::allow).)
    pub fn new(abi: Abi, fs_root: impl Into<String>, mem_limit: usize) -> ForeignProcess {
        let allowed = alloc::vec![
            HostOp::OpenFile,
            HostOp::ReadFile,
            HostOp::WriteFile,
            HostOp::CloseFile,
            HostOp::AllocMemory,
            HostOp::FreeMemory,
            HostOp::GetTime,
            HostOp::Exit,
        ];
        let allowed_mask = Self::mask_of(&allowed);
        ForeignProcess {
            abi,
            allowed,
            allowed_mask,
            fs_root: fs_root.into(),
            mem_limit,
            mem_used: 0,
        }
    }

    fn mask_of(ops: &[HostOp]) -> u16 {
        ops.iter().fold(0u16, |m, op| m | op.bit())
    }

    /// Restrict to a specific allow-list (e.g. a read-only viewer).
    pub fn with_allowed(mut self, ops: Vec<HostOp>) -> ForeignProcess {
        self.allowed_mask = Self::mask_of(&ops);
        self.allowed = ops;
        self
    }

    pub fn allow(&mut self, op: HostOp) {
        if !self.allowed.contains(&op) {
            self.allowed.push(op);
            self.allowed_mask |= op.bit();
        }
    }

    /// Project a guest path under the confinement root. Path-traversal (`..`) and
    /// absolute escapes are neutralised, so the guest can never name a host path.
    pub fn project_path(&self, guest_path: &str) -> String {
        let mut out = String::new();
        out.push_str(&self.fs_root);
        if !self.fs_root.ends_with('/') {
            out.push('/');
        }
        for seg in guest_path.split(['/', '\\']) {
            if seg.is_empty() || seg == "." || seg == ".." {
                continue; // drop traversal and roots
            }
            out.push_str(seg);
            out.push('/');
        }
        // Trim the trailing slash if we added segments.
        if out.ends_with('/') && out.len() > self.fs_root.len() + 1 {
            out.pop();
        }
        out
    }

    /// Attempt a syscall: translate it, then admit it only if it maps to an allowed,
    /// non-denied op. Returns the host op to perform, or `None` if refused.
    pub fn syscall(&self, number: u64) -> Option<HostOp> {
        let op = translate_syscall(self.abi, number);
        // `Denied` has bit 0, so a denied op is never admitted by the mask — the
        // gate is a single AND with no `Vec` scan and no separate `Denied` compare.
        if self.allowed_mask & op.bit() != 0 {
            Some(op)
        } else {
            None
        }
    }

    /// Account a memory allocation against the ceiling.
    pub fn alloc(&mut self, bytes: usize) -> bool {
        match self.mem_used.checked_add(bytes) {
            Some(t) if t <= self.mem_limit => {
                self.mem_used = t;
                true
            }
            _ => false,
        }
    }

    pub fn mem_used(&self) -> usize {
        self.mem_used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_binary_formats() {
        assert_eq!(detect_format(b"MZ\x90\x00"), BinaryFormat::Pe);
        assert_eq!(detect_format(b"\x7FELF"), BinaryFormat::Elf);
        assert_eq!(detect_format(&[0xCF, 0xFA, 0xED, 0xFE]), BinaryFormat::MachO);
        assert_eq!(detect_format(&[0xCA, 0xFE, 0xBA, 0xBE]), BinaryFormat::MachO);
        assert_eq!(detect_format(b"not a binary"), BinaryFormat::Unknown);
        assert_eq!(detect_format(b"x"), BinaryFormat::Unknown);
    }

    #[test]
    fn translates_syscalls_per_abi() {
        assert_eq!(translate_syscall(Abi::Linux, 1), HostOp::WriteFile);
        assert_eq!(translate_syscall(Abi::Win64, 0x08), HostOp::WriteFile);
        assert_eq!(translate_syscall(Abi::MacOsArm64, 4), HostOp::WriteFile);
        // Unknown numbers are default-denied across all ABIs.
        assert_eq!(translate_syscall(Abi::Linux, 9999), HostOp::Denied);
        assert_eq!(translate_syscall(Abi::Win64, 9999), HostOp::Denied);
    }

    #[test]
    fn path_projection_blocks_traversal() {
        let p = ForeignProcess::new(Abi::Win64, "/apps/photoshop/sandbox", 1 << 20);
        assert_eq!(
            p.project_path("Documents\\file.psd"),
            "/apps/photoshop/sandbox/Documents/file.psd"
        );
        // `..` and absolute escape attempts are stripped — cannot leave the root.
        let escaped = p.project_path("../../../etc/passwd");
        assert!(escaped.starts_with("/apps/photoshop/sandbox"));
        assert!(!escaped.contains(".."));
        let abs = p.project_path("/C:/Windows/System32");
        assert!(abs.starts_with("/apps/photoshop/sandbox"));
    }

    #[test]
    fn syscall_gate_admits_allowed_denies_rest() {
        let p = ForeignProcess::new(Abi::Linux, "/sandbox", 1024);
        assert_eq!(p.syscall(1), Some(HostOp::WriteFile)); // allowed
        assert_eq!(p.syscall(9999), None); // unknown → denied
        // A read-only viewer that only permits read/open/close.
        let ro = ForeignProcess::new(Abi::Linux, "/sandbox", 1024)
            .with_allowed(alloc::vec![HostOp::OpenFile, HostOp::ReadFile, HostOp::CloseFile]);
        assert_eq!(ro.syscall(0), Some(HostOp::ReadFile));
        assert_eq!(ro.syscall(1), None); // write refused for a viewer
    }

    #[test]
    fn memory_is_bounded() {
        let mut p = ForeignProcess::new(Abi::MacOsArm64, "/s", 1000);
        assert!(p.alloc(600));
        assert!(!p.alloc(600)); // 1200 > 1000 → refused
        assert_eq!(p.mem_used(), 600);
        assert!(p.alloc(400)); // exactly fits
    }
}

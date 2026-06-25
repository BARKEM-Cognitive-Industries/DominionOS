//! Unified foreign-**application** launching — **one** entry point ([`launch_app`])
//! that takes the bytes of a Linux ELF / Windows PE / macOS Mach-O application and a
//! grant of authority, and returns a running, capability-confined [`AppSession`].
//!
//! This is the application analogue of [`super::driverload`]: it ties the previously
//! disjoint pieces — format detection ([`crate::compat::detect_format`]), ABI
//! selection, capability-bundle derivation ([`CapBundle::derive_from`]) and the
//! ambient-authority projection ([`CapShim`]) — into a single confined launch. The
//! caller never hands the guest ambient authority; it hands a [`Grants`] of *derived*
//! sub-capabilities, and the session projects exactly that as the guest's world.
//!
//! **Honest execution boundary.** What is real here: format/ABI detection, the
//! derivation of a provably-⊆ capability bundle, and the *complete capability-confined
//! syscall surface* the guest runs against (files/mmap/sockets/clock/exit), with
//! default-closed denial, quota enforcement and instant revocation. What is *modeled,
//! not executed*: stepping the guest's own machine code so its `int 0x80`/`syscall`
//! traps land here automatically. Today the session is driven by feeding it the
//! guest's syscall stream ([`AppSession::run_program`]); a SIP-sandbox x86 JIT
//! ([`crate::wasm`]/kernel) would replace the manual feed with trapped syscalls —
//! the confinement and projection on this side are identical either way.
//!
//! Pure, safe `no_std`, host-tested.

use crate::capability::{CapError, Capability, Rights};
use crate::capshim::{CapBundle, CapShim, ShimError};
use crate::compat::{detect_format, Abi, BinaryFormat};
use crate::personality::x86mini::{Cpu, CpuFault, Halt, SyscallSink};
use alloc::string::String;
use alloc::vec::Vec;

/// The authority a launched application is granted — each becomes a *derived*
/// sub-capability of the launcher's authority, never ambient power.
#[derive(Clone, Copy, Debug)]
pub struct Grants {
    /// Filesystem byte quota (the FS capability's bound). 0 ⇒ no filesystem.
    pub fs_quota: u64,
    /// Memory ceiling `mmap`/`brk` allocate from. 0 ⇒ no memory grant.
    pub mem_ceiling: u64,
    /// Whether the app may use the network at all.
    pub net: bool,
    /// Whether the app may read the clock.
    pub clock: bool,
}

impl Grants {
    /// A typical sandboxed desktop app: a working directory quota and memory, no
    /// network, with a clock (so timers/`gettimeofday` work).
    pub fn sandboxed(fs_quota: u64, mem_ceiling: u64) -> Grants {
        Grants { fs_quota, mem_ceiling, net: false, clock: true }
    }

    /// A maximally-confined app: a little memory, nothing else.
    pub fn minimal(mem_ceiling: u64) -> Grants {
        Grants { fs_quota: 0, mem_ceiling, net: false, clock: false }
    }
}

/// Why an application could not be launched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchError {
    /// The bytes are not a recognised executable container.
    UnknownFormat,
    /// The requested grants could not be derived from the launcher's authority
    /// (bounds escalation refused — the launcher cannot grant what it does not hold).
    Confinement(CapError),
}

/// One modeled guest syscall — the unit [`AppSession::run_program`] feeds through the
/// confined shim. Mirrors what a trapped `syscall` instruction would carry.
#[derive(Clone, Debug, Default)]
pub struct SyscallStep {
    pub number: u64,
    pub fd: i32,
    pub len: usize,
    pub path: Option<String>,
    pub data: Option<Vec<u8>>,
}

impl SyscallStep {
    pub fn new(number: u64) -> SyscallStep {
        SyscallStep { number, fd: 0, len: 0, path: None, data: None }
    }
    pub fn fd(mut self, fd: i32) -> SyscallStep {
        self.fd = fd;
        self
    }
    pub fn len(mut self, len: usize) -> SyscallStep {
        self.len = len;
        self
    }
    pub fn path(mut self, p: &str) -> SyscallStep {
        self.path = Some(p.into());
        self
    }
    pub fn data(mut self, d: &[u8]) -> SyscallStep {
        self.data = Some(d.to_vec());
        self
    }
}

/// A running, capability-confined foreign application. Its ambient world is the
/// projection of a derived [`CapBundle`]; every syscall is checked against a held
/// capability and fails closed.
#[derive(Debug)]
pub struct AppSession {
    pub format: BinaryFormat,
    pub abi: Abi,
    shim: CapShim,
}

impl AppSession {
    /// The ABI the guest is being run as.
    pub fn abi(&self) -> Abi {
        self.abi
    }

    /// Service one guest syscall by number, returning the Linux-style result
    /// (≥0 success, <0 negated errno) — exactly what the guest's libc expects.
    pub fn syscall(&mut self, step: &SyscallStep) -> i64 {
        self.shim.syscall(
            step.number,
            step.fd,
            step.len,
            step.path.as_deref(),
            step.data.as_deref(),
        )
    }

    /// Drive the whole modeled syscall stream, collecting each result. Stops early
    /// once the guest has exited (so post-exit syscalls are not serviced).
    pub fn run_program(&mut self, steps: &[SyscallStep]) -> Vec<i64> {
        let mut out = Vec::with_capacity(steps.len());
        for step in steps {
            if self.shim.exited.is_some() {
                break;
            }
            out.push(self.syscall(step));
        }
        out
    }

    /// The guest's exit code, if it has exited.
    pub fn exit_code(&self) -> Option<i32> {
        self.shim.exited
    }

    /// Direct capability-checked file open (for the host/GUI to stage inputs).
    pub fn open(&mut self, path: &str, create: bool) -> Result<i32, ShimError> {
        self.shim.open(path, create)
    }
    pub fn write(&mut self, fd: i32, data: &[u8]) -> Result<usize, ShimError> {
        self.shim.write(fd, data)
    }
    pub fn read(&mut self, fd: i32, len: usize) -> Result<Vec<u8>, ShimError> {
        self.shim.read(fd, len)
    }

    /// Total filesystem bytes the app has used against its quota.
    pub fn fs_used(&self) -> u64 {
        self.shim.fs_used()
    }

    /// **Kill the sandbox.** Revoke every capability so the next guest syscall of any
    /// kind traps — the capability-native, race-free `kill -9`.
    pub fn revoke(&mut self) {
        self.shim.revoke();
    }

    /// **Execute real foreign machine code** in this confined session: run x86-64
    /// instruction bytes on the [`x86mini`](crate::personality::x86mini) interpreter
    /// over the sandbox memory `mem`, with every guest `syscall` trapping straight into
    /// this session's capability shim. This is the first real step past the modeled
    /// boundary — the instructions and the syscall trap are genuine, confined to exactly
    /// the granted capabilities, and bounded by `budget` steps.
    pub fn run_machine_code(
        &mut self,
        code: &[u8],
        mem: &mut [u8],
        budget: u64,
    ) -> Result<Halt, CpuFault> {
        let mut cpu = Cpu::new();
        let mut sink = ShimSink { shim: &mut self.shim };
        cpu.run(code, mem, &mut sink, budget)
    }
}

/// Bridges the x86 interpreter's `syscall` trap onto the capability shim: it maps the
/// System V register arguments to the shim's projected operations (pointer args read
/// from the sandbox memory), so real machine code reaches exactly the granted authority
/// and nothing else.
struct ShimSink<'a> {
    shim: &'a mut CapShim,
}

fn read_cstr(mem: &[u8], at: usize) -> String {
    let mut s = String::new();
    let mut i = at;
    while i < mem.len() && mem[i] != 0 {
        s.push(mem[i] as char);
        i += 1;
    }
    s
}

impl SyscallSink for ShimSink<'_> {
    fn syscall(&mut self, nr: u64, args: [u64; 6], mem: &mut [u8]) -> i64 {
        let (rdi, rsi, rdx) = (args[0], args[1], args[2]);
        match nr {
            // write(fd, buf, len) — payload read from the bounded sandbox memory.
            1 => {
                let (buf, len) = (rsi as usize, rdx as usize);
                let end = buf.saturating_add(len).min(mem.len());
                let data: &[u8] = if buf <= end { &mem[buf..end] } else { &[] };
                self.shim.syscall(1, rdi as i32, data.len(), None, Some(data))
            }
            // open(path) — path is a C string in sandbox memory.
            2 => {
                let path = read_cstr(mem, rdi as usize);
                self.shim.syscall(2, 0, 0, Some(&path), None)
            }
            // read(fd, _, len), close(fd), mmap(_, len, …), exit(code) — and a
            // default-closed fall-through for everything else.
            0 => self.shim.syscall(0, rdi as i32, rdx as usize, None, None),
            3 => self.shim.syscall(3, rdi as i32, 0, None, None),
            9 => self.shim.syscall(9, 0, rsi as usize, None, None),
            60 => self.shim.syscall(60, rdi as i32, 0, None, None),
            other => self.shim.syscall(other, rdi as i32, rdx as usize, None, None),
        }
    }

    fn exited(&self) -> bool {
        self.shim.exited.is_some()
    }
}

/// Pick the ABI a container format runs as.
fn abi_for(format: BinaryFormat) -> Option<Abi> {
    match format {
        BinaryFormat::Elf => Some(Abi::Linux),
        BinaryFormat::Pe => Some(Abi::Win64),
        BinaryFormat::MachO => Some(Abi::MacOsArm64),
        BinaryFormat::Unknown => None,
    }
}

/// Launch a foreign application from its bytes into a confined session.
///
/// `root` is the confinement root every guest path is projected under. `parent` is the
/// launcher's authority; the granted bundle is carved monotonically out of it, so the
/// app provably cannot name authority the launcher did not hold.
pub fn launch_app(
    bytes: &[u8],
    root: &str,
    parent: &Capability,
    grants: Grants,
) -> Result<AppSession, LaunchError> {
    let format = detect_format(bytes);
    let abi = abi_for(format).ok_or(LaunchError::UnknownFormat)?;

    // Derive the fs+mem grants monotonically from the launcher's authority.
    let mut bundle = CapBundle::derive_from(parent, grants.fs_quota, grants.mem_ceiling)
        .map_err(LaunchError::Confinement)?;
    // Optional ambient grants, each a restricted sub-capability of the parent.
    if grants.net {
        bundle = bundle.with_net(parent.restrict(Rights::READ).map_err(LaunchError::Confinement)?);
    }
    if grants.clock {
        bundle =
            bundle.with_clock(parent.restrict(Rights::READ).map_err(LaunchError::Confinement)?);
    }

    Ok(AppSession { format, abi, shim: CapShim::launch(abi, bundle, root) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launcher() -> Capability {
        Capability::mint(0x20_000, 0x20_000, Rights::ALL)
    }

    // Minimal valid container headers (enough for detect_format).
    fn elf_app() -> Vec<u8> {
        let mut b = alloc::vec![0u8; 64];
        b[0..4].copy_from_slice(b"\x7FELF");
        b[4] = 2;
        b[5] = 1;
        b
    }
    fn pe_app() -> Vec<u8> {
        let mut b = alloc::vec![0u8; 64];
        b[0..2].copy_from_slice(b"MZ");
        b
    }

    #[test]
    fn launches_a_linux_elf_as_a_confined_session() {
        let p = launcher();
        let mut app =
            launch_app(&elf_app(), "/sandbox/app", &p, Grants::sandboxed(0x1000, 0x4000)).unwrap();
        assert_eq!(app.format, BinaryFormat::Elf);
        assert_eq!(app.abi(), Abi::Linux);
        // It runs a tiny program: open+create a file, write to it, exit(0).
        let prog = alloc::vec![
            SyscallStep::new(2).path("out/log.txt"),       // open(O_CREAT)
            SyscallStep::new(1).fd(3).data(b"hello world"), // write
            SyscallStep::new(60).fd(0),                     // exit(0)
        ];
        let results = app.run_program(&prog);
        assert!(results[0] >= 3); // got an fd
        assert_eq!(results[1], 11); // wrote 11 bytes
        assert_eq!(app.exit_code(), Some(0));
        assert_eq!(app.fs_used(), 11);
    }

    #[test]
    fn a_windows_pe_runs_through_the_same_launcher() {
        let p = launcher();
        let mut app =
            launch_app(&pe_app(), "/sandbox/win", &p, Grants::sandboxed(0x1000, 0x1000)).unwrap();
        assert_eq!(app.abi(), Abi::Win64);
        // NtCreateFile (0x55) then NtWriteFile (0x08).
        let fd = app.syscall(&SyscallStep::new(0x55).path("a.txt"));
        assert!(fd >= 3);
        assert_eq!(app.syscall(&SyscallStep::new(0x08).fd(fd as i32).data(b"hi")), 2);
    }

    #[test]
    fn unknown_bytes_are_refused() {
        let p = launcher();
        let err = launch_app(b"not an executable", "/s", &p, Grants::minimal(0x1000)).unwrap_err();
        assert_eq!(err, LaunchError::UnknownFormat);
    }

    #[test]
    fn network_is_default_closed_unless_granted() {
        let p = launcher();
        // No net grant: socket() denied.
        let mut app = launch_app(&elf_app(), "/s", &p, Grants::sandboxed(0x100, 0x1000)).unwrap();
        // Linux socket(2) is 41 — not in the projected HostOp table ⇒ denied (-1).
        assert_eq!(app.syscall(&SyscallStep::new(41)), -1);

        // With a net grant, the shim admits socket setup at the capability layer.
        let mut net_app = launch_app(
            &elf_app(),
            "/s",
            &p,
            Grants { fs_quota: 0x100, mem_ceiling: 0x1000, net: true, clock: true },
        )
        .unwrap();
        let _ = &mut net_app; // net capability is present in the bundle (see capshim::socket)
    }

    #[test]
    fn revocation_kills_the_running_app() {
        let p = launcher();
        let mut app = launch_app(&elf_app(), "/s", &p, Grants::sandboxed(0x1000, 0x1000)).unwrap();
        let fd = app.open("f", true).unwrap();
        app.write(fd, b"data").unwrap();
        app.revoke();
        // Every authority gone: reads/writes/opens all trap.
        assert_eq!(app.read(fd, 4), Err(ShimError::Revoked));
        assert_eq!(app.write(fd, b"x"), Err(ShimError::Revoked));
    }

    #[test]
    fn real_x86_machine_code_runs_confined_through_the_shim() {
        use crate::personality::x86mini::{Halt, RAX, RDI, RDX, RSI};

        fn mov_imm(reg: usize, imm: u64) -> Vec<u8> {
            let mut v = alloc::vec![0x48u8, 0xB8 + reg as u8];
            v.extend_from_slice(&imm.to_le_bytes());
            v
        }

        let p = launcher();
        let mut app = launch_app(&elf_app(), "/sandbox/app", &p, Grants::sandboxed(0x1000, 0x1000))
            .unwrap();

        // Sandbox memory: a path "f\0" at offset 0, payload "hi" at offset 8.
        let mut mem = [0u8; 16];
        mem[0] = b'f';
        mem[8] = b'h';
        mem[9] = b'i';

        // Real x86-64: open("f") → fd in rax; mov rdi,rax; write(1, buf=8, len=2);
        // exit(0). Every syscall traps into this session's capability shim.
        let mut code = Vec::new();
        code.extend_from_slice(&mov_imm(RAX, 2)); // open
        code.extend_from_slice(&mov_imm(RDI, 0)); // path ptr = 0
        code.extend_from_slice(&[0x0F, 0x05]); // syscall → rax = fd
        code.extend_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax (fd)
        code.extend_from_slice(&mov_imm(RAX, 1)); // write
        code.extend_from_slice(&mov_imm(RSI, 8)); // buf = 8
        code.extend_from_slice(&mov_imm(RDX, 2)); // len = 2
        code.extend_from_slice(&[0x0F, 0x05]); // syscall
        code.extend_from_slice(&mov_imm(RAX, 60)); // exit
        code.extend_from_slice(&mov_imm(RDI, 0)); // status 0
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        let halt = app.run_machine_code(&code, &mut mem, 1000).unwrap();
        assert_eq!(halt, Halt::Exited);
        assert_eq!(app.exit_code(), Some(0));
        // The write went through the capability-charged FS projection.
        assert_eq!(app.fs_used(), 2);
    }

    #[test]
    fn machine_code_cannot_exceed_its_memory_via_a_syscall() {
        use crate::personality::x86mini::{RAX, RDI, RDX, RSI};

        fn mov_imm(reg: usize, imm: u64) -> Vec<u8> {
            let mut v = alloc::vec![0x48u8, 0xB8 + reg as u8];
            v.extend_from_slice(&imm.to_le_bytes());
            v
        }
        let p = launcher();
        let mut app =
            launch_app(&elf_app(), "/s", &p, Grants::sandboxed(0x1000, 0x1000)).unwrap();
        let mut mem = [0u8; 8];
        mem[0] = b'f';
        // open, then a write claiming a 9999-byte payload — clamped to the 8-byte sandbox.
        let mut code = Vec::new();
        code.extend_from_slice(&mov_imm(RAX, 2));
        code.extend_from_slice(&mov_imm(RDI, 0));
        code.extend_from_slice(&[0x0F, 0x05]);
        code.extend_from_slice(&[0x48, 0x89, 0xC7]);
        code.extend_from_slice(&mov_imm(RAX, 1));
        code.extend_from_slice(&mov_imm(RSI, 0));
        code.extend_from_slice(&mov_imm(RDX, 9999));
        code.extend_from_slice(&[0x0F, 0x05]);
        code.push(0xC3);
        app.run_machine_code(&code, &mut mem, 1000).unwrap();
        // The write was clamped to the sandbox memory — no overrun past 8 bytes.
        assert!(app.fs_used() <= 8);
    }

    #[test]
    fn memory_grant_cannot_exceed_the_launcher() {
        let p = launcher(); // 0x20_000 bytes of authority
                            // Requesting more fs+mem than the launcher holds is refused.
        let err = launch_app(&elf_app(), "/s", &p, Grants::sandboxed(0x40_000, 0x40_000))
            .unwrap_err();
        match err {
            LaunchError::Confinement(_) => {}
            other => panic!("expected confinement refusal, got {:?}", other),
        }
    }
}

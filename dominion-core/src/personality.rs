//! Linux syscall-translation personality — roadmap feature 7 (the WSL1-style
//! endgame, gate: ELF loader + sandbox).
//!
//! Rather than bake a whole Linux kernel into the OS, we *translate* the ~100
//! syscalls real programs use directly onto dominion primitives (integration
//! strategy §6). The mappings are where the architecture shines:
//!
//! * file syscalls → the projection VFS;
//! * `mmap`/`brk` → capability-bounded regions of the single address space;
//! * sockets → the network capabilities;
//! * **`fork` → snapshot-and-branch** — the deterministic state machine already
//!   expresses this, so DominionOS forks *more* naturally than Linux does.
//!
//! This module is the pure translation logic with a working file/process subset,
//! host-tested. The kernel binds it to a loaded ELF guest inside a [`Sandbox`].

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Unified driver loading (native specs, registry names, foreign `.sys`/`.ko`).
pub mod driverload;
/// Unified foreign-application launching into a capability-confined sandbox.
pub mod applaunch;
/// Capability-gated app↔system/process/network communication channels.
pub mod appchannel;
/// A minimal real x86-64 interpreter — first foothold executing foreign machine code.
pub mod x86mini;

// x86-64 Linux syscall numbers (the common subset).
pub const SYS_READ: u32 = 0;
pub const SYS_WRITE: u32 = 1;
pub const SYS_OPEN: u32 = 2;
pub const SYS_CLOSE: u32 = 3;
pub const SYS_MMAP: u32 = 9;
pub const SYS_BRK: u32 = 12;
pub const SYS_GETPID: u32 = 39;
pub const SYS_SOCKET: u32 = 41;
pub const SYS_SENDTO: u32 = 44;
pub const SYS_FORK: u32 = 57;
pub const SYS_EXIT: u32 = 60;

// Negated errno values (syscalls return -errno on failure).
pub const ENOENT: i64 = -2;
pub const EBADF: i64 = -9;
pub const ENOSYS: i64 = -38;

/// Open flags (subset).
pub const O_CREAT: u32 = 0x40;

/// Which native subsystem services a given Linux syscall — the translation table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SyscallClass {
    /// Serviced by the projection VFS.
    File,
    /// Serviced by capability-bounded memory regions.
    Memory,
    /// Serviced by the network capabilities.
    Network,
    /// Serviced by the deterministic state machine / scheduler.
    Process,
    /// Not implemented.
    Unsupported,
}

/// Classify a syscall to the dominion subsystem that handles it.
pub fn classify(number: u32) -> SyscallClass {
    match number {
        SYS_READ | SYS_WRITE | SYS_OPEN | SYS_CLOSE => SyscallClass::File,
        SYS_MMAP | SYS_BRK => SyscallClass::Memory,
        SYS_SOCKET | SYS_SENDTO => SyscallClass::Network,
        SYS_GETPID | SYS_FORK | SYS_EXIT => SyscallClass::Process,
        _ => SyscallClass::Unsupported,
    }
}

struct OpenFile {
    path: String,
    offset: usize,
    writable: bool,
}

/// A running Linux guest, its world projected onto dominion primitives.
pub struct LinuxPersonality {
    /// The projected filesystem (path → contents). In the kernel this is the
    /// real [`Vfs`](crate::vfs); here it is an in-memory stand-in with identical
    /// semantics so the translation logic is testable.
    files: BTreeMap<String, Vec<u8>>,
    fds: BTreeMap<i32, OpenFile>,
    next_fd: i32,
    pid: i32,
    pub exited: Option<i32>,
}

impl LinuxPersonality {
    pub fn new(pid: i32) -> LinuxPersonality {
        LinuxPersonality {
            files: BTreeMap::new(),
            fds: BTreeMap::new(),
            next_fd: 3, // 0/1/2 reserved for stdio
            pid,
            exited: None,
        }
    }

    pub fn pid(&self) -> i32 {
        self.pid
    }

    /// `getpid(2)`.
    pub fn getpid(&self) -> i64 {
        self.pid as i64
    }

    /// `open(2)` → resolve/create a file in the projection VFS, returning an fd.
    pub fn open(&mut self, path: &str, flags: u32) -> i64 {
        let exists = self.files.contains_key(path);
        if !exists {
            if flags & O_CREAT == 0 {
                return ENOENT;
            }
            self.files.insert(path.to_string(), Vec::new());
        }
        let fd = self.next_fd;
        self.next_fd += 1;
        self.fds.insert(fd, OpenFile { path: path.to_string(), offset: 0, writable: true });
        fd as i64
    }

    /// `write(2)` → append bytes to the file at the fd's offset.
    pub fn write(&mut self, fd: i32, data: &[u8]) -> i64 {
        let of = match self.fds.get_mut(&fd) {
            Some(of) if of.writable => of,
            _ => return EBADF,
        };
        let file = self.files.get_mut(&of.path).expect("open fd implies file");
        if of.offset > file.len() {
            file.resize(of.offset, 0);
        }
        let end = of.offset + data.len();
        if end > file.len() {
            file.resize(end, 0);
        }
        file[of.offset..end].copy_from_slice(data);
        of.offset = end;
        data.len() as i64
    }

    /// `read(2)` → read up to `len` bytes from the fd's offset.
    pub fn read(&mut self, fd: i32, len: usize) -> Result<Vec<u8>, i64> {
        let of = self.fds.get_mut(&fd).ok_or(EBADF)?;
        let file = self.files.get(&of.path).ok_or(EBADF)?;
        let start = of.offset.min(file.len());
        let end = (start + len).min(file.len());
        let out = file[start..end].to_vec();
        of.offset = end;
        Ok(out)
    }

    /// `close(2)`.
    pub fn close(&mut self, fd: i32) -> i64 {
        if self.fds.remove(&fd).is_some() {
            0
        } else {
            EBADF
        }
    }

    /// `fork(2)` → **snapshot-and-branch**. The child inherits a copy of the whole
    /// projected world; the parent receives the child's pid (Linux semantics).
    pub fn fork(&self, child_pid: i32) -> (LinuxPersonality, i64) {
        let mut child_fds = BTreeMap::new();
        for (fd, of) in &self.fds {
            child_fds.insert(*fd, OpenFile { path: of.path.clone(), offset: of.offset, writable: of.writable });
        }
        let child = LinuxPersonality {
            files: self.files.clone(),
            fds: child_fds,
            next_fd: self.next_fd,
            pid: child_pid,
            exited: None,
        };
        (child, child_pid as i64)
    }

    /// `exit(2)`.
    pub fn exit(&mut self, code: i32) -> i64 {
        self.exited = Some(code);
        0
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscalls_classify_to_the_right_subsystem() {
        assert_eq!(classify(SYS_OPEN), SyscallClass::File);
        assert_eq!(classify(SYS_MMAP), SyscallClass::Memory);
        assert_eq!(classify(SYS_SOCKET), SyscallClass::Network);
        assert_eq!(classify(SYS_FORK), SyscallClass::Process);
        assert_eq!(classify(999), SyscallClass::Unsupported);
    }

    #[test]
    fn open_write_read_round_trip_via_syscalls() {
        let mut p = LinuxPersonality::new(100);
        let fd = p.open("/tmp/note", O_CREAT) as i32;
        assert!(fd >= 3);
        assert_eq!(p.write(fd, b"hello world"), 11);
        p.close(fd);

        let fd2 = p.open("/tmp/note", 0) as i32;
        assert_eq!(p.read(fd2, 5).unwrap(), b"hello");
        assert_eq!(p.read(fd2, 100).unwrap(), b" world");
    }

    #[test]
    fn open_without_creat_on_missing_file_errors() {
        let mut p = LinuxPersonality::new(1);
        assert_eq!(p.open("/missing", 0), ENOENT);
    }

    #[test]
    fn read_write_on_bad_fd_errors() {
        let mut p = LinuxPersonality::new(1);
        assert_eq!(p.write(42, b"x"), EBADF);
        assert_eq!(p.read(42, 1).unwrap_err(), EBADF);
        assert_eq!(p.close(42), EBADF);
    }

    #[test]
    fn fork_snapshots_the_world() {
        let mut parent = LinuxPersonality::new(100);
        let fd = parent.open("/shared", O_CREAT) as i32;
        parent.write(fd, b"parent-data");
        parent.close(fd);

        let (mut child, child_pid) = parent.fork(101);
        assert_eq!(child_pid, 101);
        assert_eq!(child.pid(), 101);
        // Child sees the parent's file...
        let cfd = child.open("/shared", 0) as i32;
        assert_eq!(child.read(cfd, 100).unwrap(), b"parent-data");
        // ...but mutations are independent (snapshot, not shared memory).
        let nfd = child.open("/child-only", O_CREAT) as i32;
        child.write(nfd, b"x");
        assert_eq!(parent.file_count(), 1);
        assert_eq!(child.file_count(), 2);
    }

    #[test]
    fn exit_records_status() {
        let mut p = LinuxPersonality::new(1);
        assert_eq!(p.exit(0), 0);
        assert_eq!(p.exited, Some(0));
    }
}

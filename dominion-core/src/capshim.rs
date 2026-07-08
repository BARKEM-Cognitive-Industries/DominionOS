//! The **capability shim** — projecting a granted capability bundle as the
//! ambient authority that foreign POSIX/Win32 code expects, confined to exactly
//! the bundle (see `docs/architecture/capability-shim-and-foreign-compat.md`).
//!
//! A foreign process never holds ambient authority. At launch it is handed a
//! [`CapBundle`]: a few *derived* (monotone) [`Capability`]s — a filesystem
//! region, a memory ceiling, optionally a network endpoint and a clock. This
//! module is the connective tissue that makes that bundle *behave* like the
//! ambient world legacy code assumes (`open`/`mmap`/`socket`/`time`), while every
//! single operation is checked against a held capability. The three layers it
//! unifies are [`crate::compat`] (the per-ABI syscall → [`HostOp`] policy),
//! [`crate::personality`] (the file/fd/process mechanics) and [`crate::capability`]
//! (the unforgeable tokens).
//!
//! The properties aliasing POSIX/Win32 into the kernel could never give us fall
//! straight out of the capability algebra:
//!
//! * **No amplification** — every projected resource is a sub-capability of the
//!   bundle; a foreign process cannot name authority it was not granted.
//! * **Default-closed** — a syscall with no covering capability is *refused*, not
//!   served best-effort.
//! * **Instant revocation** — [`CapBundle::revoke`] tampers every capability, so
//!   the next syscall of any kind traps.
//! * **Monotone memory** — `mmap` derives a sub-capability of the memory grant, so
//!   total projected memory can never exceed the ceiling.
//!
//! Pure, safe `no_std`, host-tested.

use crate::capability::{CapError, Capability, Rights};
use crate::compat::{translate_syscall, Abi, HostOp};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Why a shimmed syscall was refused. Each maps to the errno/NTSTATUS the foreign
/// caller is given, so confined code fails the way it already knows how to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShimError {
    /// No capability in the bundle covers this resource/right (default-closed).
    NoCapability,
    /// The op is recognised but not on the process's allow-list (e.g. raw device).
    Denied,
    /// A memory request would exceed (escalate beyond) the memory grant.
    OutOfMemory,
    /// Operation on a file descriptor that is not open.
    BadFd,
    /// `open` without create on a path that does not exist.
    NotFound,
    /// The bundle has been revoked — every capability's tag is cleared.
    Revoked,
}

impl ShimError {
    /// The Linux errno a confined Linux guest sees (negated, syscall convention).
    pub fn errno(self) -> i64 {
        match self {
            ShimError::NoCapability | ShimError::Denied => -1, // EPERM
            ShimError::OutOfMemory => -12,                     // ENOMEM
            ShimError::BadFd => -9,                            // EBADF
            ShimError::NotFound => -2,                         // ENOENT
            ShimError::Revoked => -1,                          // EPERM (authority gone)
        }
    }
}

/// The bundle of capabilities a confined foreign process is launched with. Each is
/// `None` when the corresponding ambient authority was not granted — and a `None`
/// grant means the matching syscalls are default-closed.
///
/// Every capability here must be *derived* from the launcher's authority, so the
/// bundle is provably ⊆ what the launcher held (capability monotonicity).
#[derive(Clone, Debug)]
pub struct CapBundle {
    /// Authority over the projected filesystem. `len` is the byte quota; rights
    /// gate read vs write.
    pub fs: Option<Capability>,
    /// Authority over a region of the single address space that `mmap`/`brk`
    /// allocate from, by monotone derivation.
    pub mem: Option<Capability>,
    /// Authority to use the network. Presence (with the right) admits sockets.
    pub net: Option<Capability>,
    /// Authority to read the clock.
    pub clock: Option<Capability>,
}

impl CapBundle {
    /// An empty bundle — a maximally-confined process that can do nothing ambient.
    pub fn empty() -> CapBundle {
        CapBundle { fs: None, mem: None, net: None, clock: None }
    }

    pub fn with_fs(mut self, cap: Capability) -> CapBundle {
        self.fs = Some(cap);
        self
    }
    pub fn with_mem(mut self, cap: Capability) -> CapBundle {
        self.mem = Some(cap);
        self
    }
    pub fn with_net(mut self, cap: Capability) -> CapBundle {
        self.net = Some(cap);
        self
    }
    pub fn with_clock(mut self, cap: Capability) -> CapBundle {
        self.clock = Some(cap);
        self
    }

    /// Derive a standard bundle from a single parent authority: an FS quota and a
    /// memory ceiling carved monotonically out of `parent`. Returns `None` if the
    /// parent cannot cover the requested extents (bounds escalation refused).
    ///
    /// `parent` is laid out as `[fs_quota | mem_ceiling]` from its base, so both
    /// children are genuine sub-capabilities (provenance chains to the parent).
    pub fn derive_from(
        parent: &Capability,
        fs_quota: u64,
        mem_ceiling: u64,
    ) -> Result<CapBundle, CapError> {
        let base = parent.base();
        let fs = parent.derive(base, fs_quota, Rights::READ.union(Rights::WRITE))?;
        let mem = parent.derive(base + fs_quota, mem_ceiling, Rights::READ.union(Rights::WRITE))?;
        Ok(CapBundle { fs: Some(fs), mem: Some(mem), net: None, clock: None })
    }

    /// Revoke the whole bundle: tamper every capability so its tag clears. After
    /// this every shimmed operation traps — the capability-native answer to
    /// "kill -9 the sandbox", with no race and no scan of outstanding handles.
    pub fn revoke(&mut self) {
        for c in [&mut self.fs, &mut self.mem, &mut self.net, &mut self.clock] {
            if let Some(cap) = c {
                *cap = cap.tamper();
            }
        }
    }

    fn valid(cap: &Option<Capability>) -> bool {
        cap.as_ref().map(|c| c.is_valid()).unwrap_or(false)
    }
}

#[derive(Debug)]
struct OpenFile {
    path: String,
    offset: usize,
    writable: bool,
}

/// A confined foreign process whose ambient world is the projection of a
/// [`CapBundle`]. The same shim backs a Linux, Win64 or macOS guest — only the
/// per-ABI syscall numbers differ ([`Abi`]).
/// A precomputed, allocation-free snapshot of the bundle's *static* shape: which
/// grants are present, their rights, and the FS quota. Capability rights and bounds
/// never change after launch — only the validity tag flips on revocation — so this is
/// derived once (at launch / on bundle replacement) and the hot syscall path reads it
/// instead of re-walking the `Option`s and re-deriving rights on every call.
#[derive(Clone, Copy, Debug)]
struct BundleCache {
    fs_present: bool,
    fs_read: bool,
    fs_write: bool,
    fs_quota: u64,
    mem_present: bool,
    net_present: bool,
    clock_present: bool,
}

impl BundleCache {
    fn of(bundle: &CapBundle) -> BundleCache {
        let (fs_present, fs_read, fs_write, fs_quota) = match &bundle.fs {
            Some(c) => {
                let r = c.rights();
                (true, r.contains(Rights::READ), r.contains(Rights::WRITE), c.len())
            }
            None => (false, false, false, 0),
        };
        BundleCache {
            fs_present,
            fs_read,
            fs_write,
            fs_quota,
            mem_present: bundle.mem.is_some(),
            net_present: bundle.net.is_some(),
            clock_present: bundle.clock.is_some(),
        }
    }
}

#[derive(Debug)]
pub struct CapShim {
    abi: Abi,
    bundle: CapBundle,
    /// Static shape of `bundle`, derived once so the hot path avoids re-deriving
    /// capability rights/quota per syscall. Kept in sync whenever `bundle` changes.
    cache: BundleCache,
    /// Confinement root every guest path is projected under.
    root: String,
    /// The projected filesystem (path → bytes). In the kernel this is the real
    /// [`Vfs`](crate::vfs); here an in-memory store with identical semantics.
    files: BTreeMap<String, Vec<u8>>,
    fds: BTreeMap<i32, OpenFile>,
    next_fd: i32,
    /// FS bytes used, charged against the FS capability's `len` (the quota).
    fs_used: u64,
    /// Bump cursor for `mmap` derivations within the memory grant.
    mem_cursor: u64,
    pub exited: Option<i32>,
}

impl CapShim {
    /// Launch a confined process under `abi`, projecting `bundle` as its world,
    /// with all guest paths rooted at `root`.
    pub fn launch(abi: Abi, bundle: CapBundle, root: impl Into<String>) -> CapShim {
        let mem_cursor = bundle.mem.as_ref().map(|c| c.base()).unwrap_or(0);
        let cache = BundleCache::of(&bundle);
        CapShim {
            abi,
            bundle,
            cache,
            root: root.into(),
            files: BTreeMap::new(),
            fds: BTreeMap::new(),
            next_fd: 3, // 0/1/2 reserved for stdio
            fs_used: 0,
            mem_cursor,
            exited: None,
        }
    }

    /// Replace the active bundle, keeping the derived [`BundleCache`] in sync. The
    /// single mutation point for `bundle` after launch, so the cache can never drift.
    fn set_bundle(&mut self, bundle: CapBundle) {
        self.cache = BundleCache::of(&bundle);
        self.bundle = bundle;
    }

    /// Revoke this process's authority. Subsequent syscalls all trap. Revocation only
    /// clears each capability's validity tag (rights/quota are unchanged), so the
    /// cached static shape stays valid — the live tag check is what now fails closed.
    pub fn revoke(&mut self) {
        self.bundle.revoke();
    }

    pub fn abi(&self) -> Abi {
        self.abi
    }

    /// Project a guest path under the confinement root, neutralising traversal
    /// (`..`) and absolute escapes — the guest can never name a host path.
    pub fn project_path(&self, guest_path: &str) -> String {
        let mut out = String::new();
        out.push_str(&self.root);
        if !self.root.ends_with('/') {
            out.push('/');
        }
        let start = out.len();
        for seg in guest_path.split(['/', '\\']) {
            if seg.is_empty() || seg == "." || seg == ".." {
                continue;
            }
            out.push_str(seg);
            out.push('/');
        }
        if out.ends_with('/') && out.len() > start {
            out.pop();
        }
        out
    }

    // ── ambient-authority projection ────────────────────────────────────

    /// `open(2)` / `NtCreateFile` — gated by the FS capability. Creating or writing
    /// requires WRITE in the grant; reading requires READ.
    pub fn open(&mut self, guest_path: &str, create: bool) -> Result<i32, ShimError> {
        let cap = self.bundle.fs.as_ref().ok_or(ShimError::NoCapability)?;
        if !cap.is_valid() {
            return Err(ShimError::Revoked);
        }
        // Required right read from the precomputed cache (rights never change post-launch).
        let have = if create { self.cache.fs_write } else { self.cache.fs_read };
        if !have {
            return Err(ShimError::NoCapability);
        }
        let path = self.project_path(guest_path);
        let exists = self.files.contains_key(&path);
        if !exists {
            if !create {
                return Err(ShimError::NotFound);
            }
            self.files.insert(path.clone(), Vec::new());
        }
        let writable = self.cache.fs_write;
        let fd = self.next_fd;
        self.next_fd += 1;
        self.fds.insert(fd, OpenFile { path, offset: 0, writable });
        Ok(fd)
    }

    /// `write(2)` — appends at the fd offset, charged against the FS quota
    /// (`fs cap len`). Exceeding the quota is `ENOMEM`, never a silent overrun.
    pub fn write(&mut self, fd: i32, data: &[u8]) -> Result<usize, ShimError> {
        let cap = self.bundle.fs.as_ref().ok_or(ShimError::NoCapability)?;
        if !cap.is_valid() {
            return Err(ShimError::Revoked);
        }
        let quota = self.cache.fs_quota;
        let of = self.fds.get(&fd).ok_or(ShimError::BadFd)?;
        if !of.writable {
            return Err(ShimError::NoCapability);
        }
        // Charge only bytes that grow the file (writes within existing bytes are free).
        let path = of.path.clone();
        let offset = of.offset;
        let file_len = self.files.get(&path).map(|f| f.len()).unwrap_or(0);
        let end = offset.saturating_add(data.len());
        let growth = end.saturating_sub(file_len) as u64;
        if self.fs_used + growth > quota {
            return Err(ShimError::OutOfMemory);
        }
        let file = self.files.get_mut(&path).ok_or(ShimError::BadFd)?;
        if end > file.len() {
            file.resize(end, 0);
        }
        file[offset..end].copy_from_slice(data);
        self.fs_used += growth;
        let of = self.fds.get_mut(&fd).expect("fd checked above");
        of.offset = end;
        Ok(data.len())
    }

    /// `read(2)` — up to `len` bytes from the fd offset.
    pub fn read(&mut self, fd: i32, len: usize) -> Result<Vec<u8>, ShimError> {
        let cap = self.bundle.fs.as_ref().ok_or(ShimError::NoCapability)?;
        if !cap.is_valid() {
            return Err(ShimError::Revoked);
        }
        if !self.cache.fs_read {
            return Err(ShimError::NoCapability);
        }
        let of = self.fds.get_mut(&fd).ok_or(ShimError::BadFd)?;
        let file = self.files.get(&of.path).ok_or(ShimError::BadFd)?;
        let start = of.offset.min(file.len());
        // `len` is guest-supplied: saturate the add so a huge count can't wrap
        // (release) or overflow-panic (debug) and produce an inverted range.
        let stop = start.saturating_add(len).min(file.len());
        let out = file[start..stop].to_vec();
        of.offset = stop;
        Ok(out)
    }

    /// `close(2)`.
    pub fn close(&mut self, fd: i32) -> Result<(), ShimError> {
        self.fds.remove(&fd).map(|_| ()).ok_or(ShimError::BadFd)
    }

    /// `mmap(2)` / `brk` — a **monotone derivation** from the memory grant. Returns
    /// the sub-capability for the new region (its `base` is the mapping address).
    /// When the grant cannot cover the request the derivation fails ⇒ `ENOMEM`,
    /// so projected memory provably never exceeds the ceiling.
    pub fn mmap(&mut self, bytes: u64) -> Result<Capability, ShimError> {
        let grant = self.bundle.mem.as_ref().ok_or(ShimError::NoCapability)?;
        if !grant.is_valid() {
            return Err(ShimError::Revoked);
        }
        let at = self.mem_cursor;
        let sub = grant
            .derive(at, bytes, Rights::READ.union(Rights::WRITE))
            .map_err(|_| ShimError::OutOfMemory)?;
        self.mem_cursor = at + bytes;
        Ok(sub)
    }

    /// `socket(2)` — admitted only if a network capability was granted.
    pub fn socket(&self) -> Result<(), ShimError> {
        match &self.bundle.net {
            Some(c) if c.is_valid() => Ok(()),
            Some(_) => Err(ShimError::Revoked),
            None => Err(ShimError::NoCapability),
        }
    }

    /// `time(2)`/`gettimeofday` — admitted only if a clock capability was granted.
    pub fn gettime(&self) -> Result<(), ShimError> {
        match &self.bundle.clock {
            Some(c) if c.is_valid() => Ok(()),
            Some(_) => Err(ShimError::Revoked),
            None => Err(ShimError::NoCapability),
        }
    }

    /// `exit(2)`.
    pub fn exit(&mut self, code: i32) {
        self.exited = Some(code);
    }

    // ── the unified syscall entry point ─────────────────────────────────

    /// Translate and service one foreign syscall by number. This routes
    /// [`compat::translate_syscall`] → bundle check → projection, returning the
    /// Linux-style result (≥0 success, <0 negated errno) so an integration test
    /// can drive the whole stack the way a real guest would.
    ///
    /// `path_arg`/`data_arg` stand in for pointer arguments a real trampoline would
    /// copy from guest memory.
    pub fn syscall(
        &mut self,
        number: u64,
        fd: i32,
        len: usize,
        path_arg: Option<&str>,
        data_arg: Option<&[u8]>,
    ) -> i64 {
        // Once revoked, nothing is serviceable. The `fs_present` test is the cached
        // shape; the two `valid()` calls are the live tag checks revocation flips.
        if self.cache.fs_present
            && !CapBundle::valid(&self.bundle.fs)
            && !CapBundle::valid(&self.bundle.mem)
        {
            return ShimError::Revoked.errno();
        }
        match translate_syscall(self.abi, number) {
            HostOp::OpenFile => match path_arg {
                Some(p) => self.open(p, true).map(|fd| fd as i64).unwrap_or_else(|e| e.errno()),
                None => ShimError::NotFound.errno(),
            },
            HostOp::ReadFile => {
                self.read(fd, len).map(|b| b.len() as i64).unwrap_or_else(|e| e.errno())
            }
            HostOp::WriteFile => match data_arg {
                Some(d) => self.write(fd, d).map(|n| n as i64).unwrap_or_else(|e| e.errno()),
                None => 0,
            },
            HostOp::CloseFile => self.close(fd).map(|_| 0).unwrap_or_else(|e| e.errno()),
            HostOp::AllocMemory => {
                self.mmap(len as u64).map(|c| c.base() as i64).unwrap_or_else(|e| e.errno())
            }
            HostOp::FreeMemory => 0,
            HostOp::GetTime => self.gettime().map(|_| 0).unwrap_or_else(|e| e.errno()),
            HostOp::Exit => {
                self.exit(fd);
                0
            }
            HostOp::Denied => ShimError::Denied.errno(),
        }
    }

    /// Total FS bytes charged against the quota.
    pub fn fs_used(&self) -> u64 {
        self.fs_used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root_authority() -> Capability {
        // The launcher's authority over a slab of the single address space.
        Capability::mint(0x10_000, 0x10_000, Rights::ALL)
    }

    fn standard_shim() -> CapShim {
        let parent = root_authority();
        let bundle = CapBundle::derive_from(&parent, 0x1000, 0x4000).unwrap();
        CapShim::launch(Abi::Linux, bundle, "/sandbox/app")
    }

    #[test]
    fn bundle_is_a_genuine_sub_capability_of_the_parent() {
        let parent = root_authority();
        let bundle = CapBundle::derive_from(&parent, 0x1000, 0x4000).unwrap();
        let fs = bundle.fs.unwrap();
        let mem = bundle.mem.unwrap();
        // Both children are enclosed by the parent and carry valid tags.
        assert!(fs.is_valid() && mem.is_valid());
        assert!(fs.base() >= parent.base());
        assert!(mem.base() + mem.len() <= parent.base() + parent.len());
        // Can't carve more than the parent holds (bounds escalation refused).
        assert!(CapBundle::derive_from(&parent, 0x1_0000, 0x1_0000).is_err());
    }

    #[test]
    fn open_write_read_round_trip_through_capabilities() {
        let mut s = standard_shim();
        let fd = s.open("notes/todo.txt", true).unwrap();
        assert!(fd >= 3);
        assert_eq!(s.write(fd, b"hello shim").unwrap(), 10);
        s.close(fd).unwrap();
        let fd2 = s.open("notes/todo.txt", false).unwrap();
        assert_eq!(s.read(fd2, 5).unwrap(), b"hello");
        assert_eq!(s.read(fd2, 100).unwrap(), b" shim");
    }

    #[test]
    fn default_closed_without_an_fs_capability() {
        let mut s = CapShim::launch(Abi::Linux, CapBundle::empty(), "/sandbox");
        // No FS grant ⇒ open is refused regardless of the path.
        assert_eq!(s.open("anything", true), Err(ShimError::NoCapability));
        assert_eq!(s.syscall(2, 0, 0, Some("anything"), None), -1);
    }

    #[test]
    fn read_only_fs_grant_refuses_writes() {
        let parent = root_authority();
        let ro = parent.derive(parent.base(), 0x1000, Rights::READ).unwrap();
        let bundle = CapBundle::empty().with_fs(ro);
        let mut s = CapShim::launch(Abi::Linux, bundle, "/ro");
        // Creating requires WRITE, which the grant lacks.
        assert_eq!(s.open("x", true), Err(ShimError::NoCapability));
    }

    #[test]
    fn path_traversal_cannot_escape_the_root() {
        let s = standard_shim();
        let p = s.project_path("../../../etc/passwd");
        assert!(p.starts_with("/sandbox/app"));
        assert!(!p.contains(".."));
        let abs = s.project_path("/C:/Windows/System32");
        assert!(abs.starts_with("/sandbox/app"));
    }

    #[test]
    fn fs_quota_is_enforced_by_the_capability_len() {
        // Quota of 16 bytes.
        let parent = root_authority();
        let fs = parent.derive(parent.base(), 16, Rights::READ.union(Rights::WRITE)).unwrap();
        let mut s = CapShim::launch(Abi::Linux, CapBundle::empty().with_fs(fs), "/q");
        let fd = s.open("f", true).unwrap();
        assert_eq!(s.write(fd, b"0123456789").unwrap(), 10); // 10 ≤ 16
        assert_eq!(s.write(fd, b"abcdef"), Ok(6)); // exactly 16
        assert_eq!(s.write(fd, b"!"), Err(ShimError::OutOfMemory)); // 17 > 16
        assert_eq!(s.fs_used(), 16);
    }

    #[test]
    fn mmap_is_monotone_and_bounded() {
        let mut s = standard_shim(); // mem ceiling 0x4000
        let a = s.mmap(0x1000).unwrap();
        let b = s.mmap(0x2000).unwrap();
        // Each mapping is a sub-capability with read+write, non-overlapping.
        assert!(a.rights().contains(Rights::WRITE));
        assert_eq!(b.base(), a.base() + 0x1000);
        // The next request would exceed the ceiling ⇒ ENOMEM, never an overrun.
        assert_eq!(s.mmap(0x2000), Err(ShimError::OutOfMemory));
    }

    #[test]
    fn revocation_traps_every_subsequent_op() {
        let mut s = standard_shim();
        let fd = s.open("f", true).unwrap();
        s.write(fd, b"data").unwrap();
        s.revoke();
        // Every authority is now gone — reads, writes, opens and mmaps all trap.
        assert_eq!(s.read(fd, 4), Err(ShimError::Revoked));
        assert_eq!(s.write(fd, b"x"), Err(ShimError::Revoked));
        assert_eq!(s.open("g", true), Err(ShimError::Revoked));
        assert_eq!(s.mmap(8), Err(ShimError::Revoked));
    }

    #[test]
    fn net_and_clock_are_default_closed_until_granted() {
        let mut s = standard_shim();
        assert_eq!(s.socket(), Err(ShimError::NoCapability));
        assert_eq!(s.gettime(), Err(ShimError::NoCapability));
        // Grant them and they open up.
        let parent = root_authority();
        let regranted = s
            .bundle
            .clone()
            .with_net(parent.restrict(Rights::READ).unwrap())
            .with_clock(parent.restrict(Rights::READ).unwrap());
        s.set_bundle(regranted);
        assert_eq!(s.socket(), Ok(()));
        assert_eq!(s.gettime(), Ok(()));
    }

    #[test]
    fn win64_abi_routes_through_the_same_shim() {
        let parent = root_authority();
        let bundle = CapBundle::derive_from(&parent, 0x1000, 0x1000).unwrap();
        let mut s = CapShim::launch(Abi::Win64, bundle, "/win");
        // NtCreateFile (0x55) then NtWriteFile (0x08) via the unified entry point.
        let fd = s.syscall(0x55, 0, 0, Some("a.txt"), None);
        assert!(fd >= 3);
        assert_eq!(s.syscall(0x08, fd as i32, 0, None, Some(b"hi")), 2);
    }

    #[test]
    fn unknown_syscalls_are_denied() {
        let mut s = standard_shim();
        assert_eq!(s.syscall(9999, 0, 0, None, None), -1);
    }
}

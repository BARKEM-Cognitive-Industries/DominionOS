//! The shell's **live filesystem** — a single, shared, capability-gated handle over
//! the real [`Vfs`](crate::vfs) POSIX projection and its backing
//! [`ObjectGraph`](crate::object).
//!
//! Both the graphical **Files** app ([`crate::files`], a Windows-Explorer-style
//! browser) and the **Terminal**'s Linux-like shell ([`crate::shellcmd`]) operate on
//! *one and the same* filesystem: type `mkdir /home/jayden/x` in the terminal and the
//! new folder appears in Files; save a file in Files and `cat` reads it back. That is
//! achieved by sharing one [`FileSystem`] behind an [`Rc`]`<`[`RefCell`]`>` — both safe,
//! `no_std`, and `#![forbid(unsafe_code)]`-clean.
//!
//! Files are stored the DominionOS way — every write is a new immutable `Text`/`Blob`
//! object in the graph and the path is re-aliased — so the "filesystem" is never a
//! second source of truth, only a path projection of the object graph (keystone K1).

use crate::capability::{Capability, Rights};
use crate::hash::Hash256;
use crate::object::{Datum, Object, ObjectGraph};
use crate::objstore::{Manifest, ObjStore};
use crate::persist::{BlockDevice, BlockError};
use crate::vfs::{Vfs, VfsError};
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

/// A shared handle to the one live filesystem (cheap to clone — it is reference
/// counted). Hand a clone to every surface that needs the filesystem.
pub type SharedFs = Rc<RefCell<FileSystem>>;

/// One directory entry, ready for display: a name, whether it is a folder, and (for
/// files) the byte length of its current content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FsEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: usize,
}

/// The live, capability-gated filesystem the shell shares across apps.
pub struct FileSystem {
    vfs: Vfs,
    graph: ObjectGraph,
    /// The shell's filesystem authority (full rights over the path namespace). Real
    /// apps are handed *attenuated* capabilities; this is the shell's own root handle.
    cap: Capability,
    /// The terminal's current working directory (absolute, normalised, no trailing `/`).
    cwd: String,
    /// Set by any content mutation, cleared by a successful persist/restore. Lets a
    /// periodic checkpoint skip the device entirely when nothing has changed since the
    /// last flush — an unchanged tree never touches the disk.
    dirty: bool,
}

impl FileSystem {
    /// Build the shared filesystem, seeded with a realistic home + system tree so the
    /// machine feels populated on first boot (like a freshly-installed OS).
    pub fn shared() -> SharedFs {
        Rc::new(RefCell::new(FileSystem::new()))
    }

    /// A filesystem seeded with an FHS skeleton plus a populated `/home/jayden`.
    pub fn new() -> FileSystem {
        let mut fs = FileSystem {
            vfs: Vfs::with_fhs(),
            graph: ObjectGraph::new(),
            cap: Capability::mint(0, 0x1_0000_0000, Rights::ALL),
            cwd: "/home/jayden".to_string(),
            dirty: true,
        };
        fs.seed();
        fs
    }

    fn seed(&mut self) {
        for d in [
            "/home/jayden",
            "/home/jayden/Documents",
            "/home/jayden/Projects",
            "/home/jayden/Pictures",
            "/home/jayden/Downloads",
            "/var/log",
            "/usr/bin",
        ] {
            let _ = self.mkdir(d);
        }
        let seed_files: &[(&str, &str)] = &[
            (
                "/home/jayden/Documents/welcome.txt",
                "Welcome to DominionOS.\n\nThis is a real file in the object graph,\nprojected onto a POSIX path by the VFS (keystone K1).\nEdit it in Files, or `cat` it in the Terminal.\n",
            ),
            (
                "/home/jayden/Documents/readme.md",
                "# DominionOS\n\nGraphical like Windows, a developer terminal like Linux,\nbuilt on one capability-secured object graph.\n",
            ),
            (
                "/home/jayden/Projects/project.aeth",
                "let gross = 5000;\nlet tax = gross / 5;\nlet net = gross - tax;\nnet\n",
            ),
            ("/etc/hostname", "dominionos\n"),
            ("/etc/motd", "DominionOS — capability-secured, deterministic.\n"),
            ("/var/log/system.log", "[boot] system online\n"),
        ];
        for (path, body) in seed_files {
            let _ = self.write_text(path, body);
        }
    }

    /// The current working directory (absolute).
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// Change the working directory; returns false (leaving cwd unchanged) if the
    /// target is not an existing directory.
    pub fn set_cwd(&mut self, path: &str) -> bool {
        let abs = self.normalize(path);
        if self.vfs.is_dir(&abs) || abs == "/" {
            self.cwd = abs;
            true
        } else {
            false
        }
    }

    /// Resolve a possibly-relative path against the cwd into a normalised absolute
    /// path (collapsing `.`/`..` and redundant separators). The empty path is the cwd.
    pub fn normalize(&self, path: &str) -> String {
        let mut comps: Vec<&str> = Vec::new();
        let base = if path.starts_with('/') { "" } else { &self.cwd[..] };
        for seg in base.split('/').chain(path.split('/')) {
            match seg {
                "" | "." => {}
                ".." => {
                    comps.pop();
                }
                s => comps.push(s),
            }
        }
        if comps.is_empty() {
            "/".to_string()
        } else {
            let mut s = String::new();
            for c in &comps {
                s.push('/');
                s.push_str(c);
            }
            s
        }
    }

    pub fn is_dir(&self, path: &str) -> bool {
        let abs = self.normalize(path);
        abs == "/" || self.vfs.is_dir(&abs)
    }

    pub fn is_file(&self, path: &str) -> bool {
        self.vfs.is_file(&self.normalize(path))
    }

    pub fn exists(&self, path: &str) -> bool {
        let abs = self.normalize(path);
        abs == "/" || self.vfs.exists(&abs)
    }

    /// List a directory as display-ready entries, directories first then files, each
    /// alphabetical. Returns `None` if the path is not a directory.
    pub fn entries(&self, path: &str) -> Option<Vec<FsEntry>> {
        let abs = self.normalize(path);
        let names = self.vfs.list(&abs).ok()?;
        let mut out: Vec<FsEntry> = names
            .into_iter()
            .map(|name| {
                let child = join(&abs, &name);
                let is_dir = self.vfs.is_dir(&child);
                let size = if is_dir { 0 } else { self.read_text(&child).map(|t| t.len()).unwrap_or(0) };
                FsEntry { name, is_dir, size }
            })
            .collect();
        out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
        Some(out)
    }

    /// Read a text file's content, or `None` if it is missing / not text.
    pub fn read_text(&self, path: &str) -> Option<String> {
        let abs = self.normalize(path);
        let obj = self.vfs.read_object(&self.graph, &abs, &self.cap).ok()?;
        match obj.get("content") {
            Some(Datum::Text(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Write (create or overwrite) a text file. Every write is a new immutable object
    /// in the graph with the path re-aliased to it — versioned, never mutated in place.
    pub fn write_text(&mut self, path: &str, content: &str) -> Result<(), VfsError> {
        let abs = self.normalize(path);
        let obj = Object::new("Text").with("content", Datum::Text(content.to_string()));
        self.vfs.write_object(&mut self.graph, &abs, obj, &self.cap)?;
        self.dirty = true;
        Ok(())
    }

    /// Create a directory (and any missing parents).
    pub fn mkdir(&mut self, path: &str) -> Result<(), VfsError> {
        let abs = self.normalize(path);
        self.vfs.mkdir_p(&abs, &self.cap)?;
        self.dirty = true;
        Ok(())
    }

    /// Remove a file alias or an empty directory (the object survives in history).
    pub fn remove(&mut self, path: &str) -> Result<(), VfsError> {
        let abs = self.normalize(path);
        self.vfs.remove(&abs, &self.cap)?;
        self.dirty = true;
        Ok(())
    }

    /// Whether content has changed since the last successful persist/restore. A periodic
    /// checkpoint consults this to avoid spinning the disk on an idle, unchanged tree.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// The number of live objects backing the filesystem (for diagnostics / `df`).
    pub fn object_count(&self) -> usize {
        self.graph.live_ids().len()
    }

    /// Iterate all stored objects in the backing graph.
    /// Used by the OS to register objects with the RAM dedup index after a persist.
    pub fn stored_objects(&self) -> impl Iterator<Item = (&crate::object::ObjectId, &Object)> {
        self.graph.stored_objects()
    }

    // ── durable persistence ──

    /// Serialise the whole filesystem to a self-contained byte image: the 32-byte
    /// namespace-root object id, then the backing object graph. The kernel writes this
    /// to virtio-blk on shutdown so files survive a reboot.
    pub fn to_bytes(&mut self) -> Vec<u8> {
        let root = self.vfs.snapshot_namespace(&mut self.graph);
        let mut out = Vec::with_capacity(32 + 64);
        out.extend_from_slice(&root.0);
        out.extend_from_slice(&self.graph.serialize());
        out
    }

    /// Restore the filesystem from a [`to_bytes`](Self::to_bytes) image. Returns false
    /// (leaving the current filesystem untouched) if the image is too short, the graph
    /// fails to deserialise, or the root is not a directory — so a corrupt image can
    /// never break boot; the seeded filesystem simply stays in place.
    pub fn restore_from_bytes(&mut self, bytes: &[u8]) -> bool {
        if bytes.len() < 32 {
            return false;
        }
        let mut root = [0u8; 32];
        root.copy_from_slice(&bytes[..32]);
        let root_id = Hash256(root);
        let Ok(graph) = ObjectGraph::deserialize(&bytes[32..]) else {
            return false;
        };
        let Some(vfs) = Vfs::from_namespace(&graph, root_id) else {
            return false;
        };
        self.graph = graph;
        self.vfs = vfs;
        // Keep the cwd valid against the restored tree.
        let cwd = self.cwd.clone();
        if !self.is_dir(&cwd) {
            self.cwd = "/".to_string();
        }
        true
    }

    // ── incremental durable persistence (preferred over [`to_bytes`]) ──

    /// Persist the filesystem to `dev` at `base_lba` through the incremental, content-
    /// addressed [`ObjStore`]. `prior` is the manifest returned by the matching
    /// [`restore_from`](Self::restore_from) (or `None` on a fresh disk); on success it is
    /// updated so the *next* save appends only objects created since — turning a shutdown
    /// flush from "rewrite the whole graph" into "write this session's new versions".
    pub fn persist_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        base_lba: u64,
        prior: &mut Option<Manifest>,
    ) -> Result<(), BlockError> {
        let root = self.vfs.snapshot_namespace(&mut self.graph);
        ObjStore::save(dev, base_lba, &self.graph, root, prior)?;
        self.dirty = false;
        Ok(())
    }

    /// Restore from an [`ObjStore`] image at `base_lba`, returning the manifest to thread
    /// into the next [`persist_to`](Self::persist_to). `Ok(None)` (filesystem left as-is)
    /// when there is no valid image or it fails to reconstruct — so a corrupt or absent
    /// store can never break boot, exactly as [`restore_from_bytes`](Self::restore_from_bytes).
    pub fn restore_from(
        &mut self,
        dev: &mut dyn BlockDevice,
        base_lba: u64,
    ) -> Result<Option<Manifest>, BlockError> {
        match ObjStore::load(dev, base_lba)? {
            Some((graph, root, manifest)) => {
                let Some(vfs) = Vfs::from_namespace(&graph, root) else {
                    return Ok(None);
                };
                self.graph = graph;
                self.vfs = vfs;
                let cwd = self.cwd.clone();
                if !self.is_dir(&cwd) {
                    self.cwd = "/".to_string();
                }
                // The on-disk image now matches memory — a checkpoint would be a no-op
                // until the next edit.
                self.dirty = false;
                Ok(Some(manifest))
            }
            None => Ok(None),
        }
    }
}

impl Default for FileSystem {
    fn default() -> Self {
        FileSystem::new()
    }
}

/// Join an absolute directory path with a child name into a normalised absolute path.
pub fn join(dir: &str, name: &str) -> String {
    if dir == "/" {
        let mut s = String::from("/");
        s.push_str(name);
        s
    } else {
        let mut s = String::from(dir);
        s.push('/');
        s.push_str(name);
        s
    }
}

/// The parent directory of an absolute path (`/a/b/c` → `/a/b`, `/x` → `/`, `/` → `/`).
pub fn parent(path: &str) -> String {
    match path.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => path[..i].to_string(),
        None => "/".to_string(),
    }
}

/// The final component of a path (`/a/b/c` → `c`, `/` → `/`).
pub fn basename(path: &str) -> &str {
    if path == "/" {
        return "/";
    }
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_a_populated_home_tree() {
        let fs = FileSystem::new();
        assert_eq!(fs.cwd(), "/home/jayden");
        assert!(fs.is_dir("/home/jayden/Documents"));
        assert!(fs.is_file("/home/jayden/Documents/welcome.txt"));
        let body = fs.read_text("/home/jayden/Documents/welcome.txt").unwrap();
        assert!(body.contains("Welcome to DominionOS"));
    }

    #[test]
    fn relative_paths_resolve_against_cwd() {
        let mut fs = FileSystem::new();
        assert_eq!(fs.normalize("Documents"), "/home/jayden/Documents");
        assert_eq!(fs.normalize("./Projects/.."), "/home/jayden");
        assert_eq!(fs.normalize("/etc/hostname"), "/etc/hostname");
        assert_eq!(fs.normalize("../.."), "/");
        assert!(fs.set_cwd("Documents"));
        assert_eq!(fs.cwd(), "/home/jayden/Documents");
        assert!(!fs.set_cwd("nope"));
        assert_eq!(fs.cwd(), "/home/jayden/Documents");
    }

    #[test]
    fn entries_list_dirs_first_then_files_alphabetical() {
        let fs = FileSystem::new();
        let e = fs.entries("/home/jayden").unwrap();
        // Documents, Downloads, Pictures, Projects are dirs (first), nothing else here.
        assert!(e[0].is_dir);
        let names: Vec<&str> = e.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"Documents"));
        assert!(names.contains(&"Projects"));
        // All listed are directories and sorted.
        assert!(e.iter().all(|x| x.is_dir));
    }

    #[test]
    fn write_is_versioned_and_readable_back() {
        let mut fs = FileSystem::new();
        fs.write_text("/home/jayden/note.txt", "first").unwrap();
        assert_eq!(fs.read_text("/home/jayden/note.txt").as_deref(), Some("first"));
        fs.write_text("/home/jayden/note.txt", "second").unwrap();
        assert_eq!(fs.read_text("/home/jayden/note.txt").as_deref(), Some("second"));
        // Both object versions remain live in the graph (immutable history).
        assert!(fs.object_count() >= 2);
    }

    #[test]
    fn mkdir_and_remove_round_trip() {
        let mut fs = FileSystem::new();
        fs.mkdir("/home/jayden/Pictures/2026").unwrap();
        assert!(fs.is_dir("/home/jayden/Pictures/2026"));
        fs.remove("/home/jayden/Pictures/2026").unwrap();
        assert!(!fs.exists("/home/jayden/Pictures/2026"));
    }

    #[test]
    fn shared_handle_is_one_filesystem() {
        let a = FileSystem::shared();
        let b = a.clone();
        a.borrow_mut().write_text("/tmp/x", "hi").unwrap();
        // The clone sees the same write — one filesystem, two handles.
        assert_eq!(b.borrow().read_text("/tmp/x").as_deref(), Some("hi"));
    }

    #[test]
    fn to_bytes_then_restore_survives_a_reboot() {
        // Author some state, image it, then restore into a fresh filesystem.
        let mut fs = FileSystem::new();
        fs.write_text("/home/jayden/Documents/report.txt", "Q3 numbers").unwrap();
        fs.mkdir("/home/jayden/Projects/dominion").unwrap();
        fs.set_cwd("/home/jayden/Projects");
        let image = fs.to_bytes();

        let mut booted = FileSystem::new();
        assert!(booted.restore_from_bytes(&image));
        // The authored file + folder are present with their content.
        assert_eq!(booted.read_text("/home/jayden/Documents/report.txt").as_deref(), Some("Q3 numbers"));
        assert!(booted.is_dir("/home/jayden/Projects/dominion"));
        // The seeded files are still there too.
        assert!(booted.is_file("/etc/hostname"));
    }

    #[test]
    fn incremental_persist_then_restore_survives_a_reboot() {
        let mut disk = crate::persist::RamDisk::new(4096);
        let mut fs = FileSystem::new();
        fs.write_text("/home/jayden/Documents/report.txt", "Q3 numbers").unwrap();
        let mut prior = None;
        fs.persist_to(&mut disk, 4, &mut prior).unwrap();

        // A later edit + an incremental save (only the new version is appended).
        fs.write_text("/home/jayden/Projects/notes.md", "# plans").unwrap();
        fs.persist_to(&mut disk, 4, &mut prior).unwrap();

        // Boot fresh and restore from the store.
        let mut booted = FileSystem::new();
        let manifest = booted.restore_from(&mut disk, 4).unwrap();
        assert!(manifest.is_some());
        assert_eq!(booted.read_text("/home/jayden/Documents/report.txt").as_deref(), Some("Q3 numbers"));
        assert_eq!(booted.read_text("/home/jayden/Projects/notes.md").as_deref(), Some("# plans"));
        // Seeded files survive too.
        assert!(booted.is_file("/etc/hostname"));
    }

    #[test]
    fn dirty_flag_gates_the_periodic_checkpoint() {
        let mut disk = crate::persist::RamDisk::new(4096);
        let mut fs = FileSystem::new();
        // A freshly-seeded (never-persisted) filesystem is dirty: the first checkpoint
        // must flush it.
        assert!(fs.is_dirty());

        let mut prior = None;
        fs.persist_to(&mut disk, 4, &mut prior).unwrap();
        // After a successful flush there is nothing new — a checkpoint here would be a
        // pure no-op, so the kernel skips the disk.
        assert!(!fs.is_dirty());

        // Any content edit re-arms the checkpoint.
        fs.write_text("/home/jayden/Documents/report.txt", "Q3").unwrap();
        assert!(fs.is_dirty());
        fs.persist_to(&mut disk, 4, &mut prior).unwrap();
        assert!(!fs.is_dirty());

        // mkdir and remove count as changes too.
        fs.mkdir("/home/jayden/Pictures/2026").unwrap();
        assert!(fs.is_dirty());
        fs.persist_to(&mut disk, 4, &mut prior).unwrap();
        fs.remove("/home/jayden/Pictures/2026").unwrap();
        assert!(fs.is_dirty());

        // A restore matches disk to memory, so it also clears the flag.
        let mut booted = FileSystem::new();
        assert!(booted.is_dirty());
        booted.restore_from(&mut disk, 4).unwrap();
        assert!(!booted.is_dirty());
    }

    #[test]
    fn incremental_restore_on_a_blank_disk_is_none() {
        let mut disk = crate::persist::RamDisk::new(64);
        let mut fs = FileSystem::new();
        assert!(fs.restore_from(&mut disk, 4).unwrap().is_none());
        // The seeded filesystem is left untouched.
        assert!(fs.is_file("/home/jayden/Documents/welcome.txt"));
    }

    #[test]
    fn restore_rejects_a_corrupt_image_without_touching_state() {
        let mut fs = FileSystem::new();
        assert!(!fs.restore_from_bytes(&[1, 2, 3])); // too short
        assert!(!fs.restore_from_bytes(&[0u8; 64])); // 32-byte zero root + garbage graph
        // The original seeded filesystem is intact.
        assert!(fs.is_file("/home/jayden/Documents/welcome.txt"));
    }

    #[test]
    fn path_helpers() {
        assert_eq!(join("/", "a"), "/a");
        assert_eq!(join("/a/b", "c"), "/a/b/c");
        assert_eq!(parent("/a/b/c"), "/a/b");
        assert_eq!(parent("/x"), "/");
        assert_eq!(basename("/a/b/c"), "c");
        assert_eq!(basename("/"), "/");
    }
}

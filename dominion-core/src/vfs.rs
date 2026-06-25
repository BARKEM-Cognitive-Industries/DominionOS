//! POSIX-projection virtual filesystem — keystone **K1** of the integration
//! strategy.
//!
//! DominionOS has no filesystem; it has the content-addressed
//! [`ObjectGraph`](crate::object). But the legacy world speaks *paths*
//! (`/etc/hosts`, `/usr/lib/...`). This module renders the graph as a mountable
//! path namespace so legacy code sees familiar paths, while underneath every
//! path is just an **alias into the graph** — never a second, parallel filesystem.
//!
//! The rules, straight from the strategy doc:
//!
//! * **Project, don't pollute.** A path resolves to an [`ObjectId`]; the legacy
//!   path model never enters the graph itself.
//! * **Writes are captured as new immutable objects + commits.** "Editing" a file
//!   stores a *new* object (new content id) and re-points the alias; the previous
//!   bytes are never mutated and remain in the graph's history.
//! * **The namespace is itself content-addressable.** [`Vfs::snapshot_namespace`]
//!   materialises the directory tree into `Dir` objects, so the whole namespace
//!   has a single root hash and can later be persisted/rolled back like any other
//!   object (the bridge to milestone M1, persistence).
//!
//! Every mutation is gated by a [`Capability`]: writing needs `WRITE`, reading
//! needs `READ`. Pure, safe, `no_std`, host-unit-tested.

use crate::capability::{CapError, Capability, Rights};
use crate::codec::Blob;
use crate::hash::Hash256;
use crate::object::{Datum, Object, ObjectGraph, ObjectId};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The semantic kind used to materialise a directory node into the graph.
pub const DIR_KIND: &str = "Dir";

/// Why a VFS operation was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum VfsError {
    /// No entry exists at the path.
    NotFound,
    /// A path component that needed to be a directory was a file.
    NotADirectory,
    /// Expected a file but found a directory.
    IsADirectory,
    /// Tried to remove a non-empty directory.
    NotEmpty,
    /// The path is empty / refers to the root where a name was required.
    InvalidPath,
    /// The object an alias points at is missing from the graph.
    DanglingAlias,
    /// The presented capability did not authorise the operation.
    Capability(CapError),
}

impl From<CapError> for VfsError {
    fn from(e: CapError) -> Self {
        VfsError::Capability(e)
    }
}

impl core::fmt::Display for VfsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VfsError::NotFound => f.write_str("vfs error: no such path"),
            VfsError::NotADirectory => f.write_str("vfs error: not a directory"),
            VfsError::IsADirectory => f.write_str("vfs error: is a directory"),
            VfsError::NotEmpty => f.write_str("vfs error: directory not empty"),
            VfsError::InvalidPath => f.write_str("vfs error: invalid path"),
            VfsError::DanglingAlias => f.write_str("vfs error: alias points at a missing object"),
            VfsError::Capability(e) => write!(f, "vfs error: {}", e),
        }
    }
}

fn require(cap: &Capability, needed: Rights) -> Result<(), VfsError> {
    if !cap.is_valid() {
        return Err(CapError::TagInvalid.into());
    }
    if !cap.rights().contains(needed) {
        return Err(CapError::InsufficientRights.into());
    }
    Ok(())
}

/// A node in the path namespace: either a directory of named children or a file
/// that aliases one content-addressed object in the graph.
enum Node {
    Dir(BTreeMap<String, Node>),
    File(ObjectId),
}

/// A projection of the [`ObjectGraph`] as a hierarchical path namespace.
pub struct Vfs {
    root: Node,
}

impl Vfs {
    /// An empty namespace containing just the root directory.
    pub fn new() -> Vfs {
        Vfs {
            root: Node::Dir(BTreeMap::new()),
        }
    }

    /// A namespace pre-seeded with a minimal Filesystem-Hierarchy-Standard skeleton
    /// (`/etc`, `/usr/lib`, `/tmp`, `/home`) so legacy tools find expected paths.
    pub fn with_fhs() -> Vfs {
        let mut v = Vfs::new();
        for d in ["/etc", "/usr/lib", "/usr/bin", "/tmp", "/home", "/var"] {
            v.force_mkdir(d);
        }
        v
    }

    /// Split a path into normalised components, resolving `.` and `..`.
    fn components(path: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for seg in path.split('/') {
            match seg {
                "" | "." => {}
                ".." => {
                    out.pop();
                }
                s => out.push(s.to_string()),
            }
        }
        out
    }

    fn root_map_mut(&mut self) -> &mut BTreeMap<String, Node> {
        match &mut self.root {
            Node::Dir(m) => m,
            Node::File(_) => unreachable!("root is always a directory"),
        }
    }

    /// Navigate to (creating as needed) the directory named by `comps`, returning
    /// its child map. Fails if any component is an existing file.
    fn ensure_dir_mut(&mut self, comps: &[String]) -> Result<&mut BTreeMap<String, Node>, VfsError> {
        let mut cur = self.root_map_mut();
        for c in comps {
            let node = cur.entry(c.clone()).or_insert_with(|| Node::Dir(BTreeMap::new()));
            cur = match node {
                Node::Dir(m) => m,
                Node::File(_) => return Err(VfsError::NotADirectory),
            };
        }
        Ok(cur)
    }

    /// Resolve `comps` to a node without mutating. Empty `comps` is the root.
    fn node_at(&self, comps: &[String]) -> Option<&Node> {
        let mut node = &self.root;
        for c in comps {
            match node {
                Node::Dir(m) => node = m.get(c)?,
                Node::File(_) => return None,
            }
        }
        Some(node)
    }

    /// Internal, uncapped mkdir -p used for seeding skeletons.
    fn force_mkdir(&mut self, path: &str) {
        let comps = Self::components(path);
        let _ = self.ensure_dir_mut(&comps);
    }

    /// Create a directory (and any missing parents). Requires `WRITE`.
    pub fn mkdir_p(&mut self, path: &str, cap: &Capability) -> Result<(), VfsError> {
        require(cap, Rights::WRITE)?;
        let comps = Self::components(path);
        self.ensure_dir_mut(&comps)?;
        Ok(())
    }

    /// Store `obj` in the graph and alias `path` to it, creating parent
    /// directories as needed. Returns the new object's content id. Because the
    /// id is the content hash, re-writing different content yields a *new* id
    /// while the old object remains in the graph — immutable, versioned writes.
    /// Requires `WRITE`.
    pub fn write_object(
        &mut self,
        graph: &mut ObjectGraph,
        path: &str,
        obj: Object,
        cap: &Capability,
    ) -> Result<ObjectId, VfsError> {
        require(cap, Rights::WRITE)?;
        let comps = Self::components(path);
        let (name, dirs) = comps.split_last().ok_or(VfsError::InvalidPath)?;
        let map = self.ensure_dir_mut(dirs)?;
        if matches!(map.get(name), Some(Node::Dir(_))) {
            return Err(VfsError::IsADirectory);
        }
        let id = graph.put(obj);
        map.insert(name.clone(), Node::File(id));
        Ok(id)
    }

    /// Write raw bytes at `path` as a verbatim [`Blob`] object. Requires `WRITE`.
    pub fn write_bytes(
        &mut self,
        graph: &mut ObjectGraph,
        path: &str,
        media_type: &str,
        bytes: &[u8],
        cap: &Capability,
    ) -> Result<ObjectId, VfsError> {
        let obj = Blob::to_object(media_type, bytes);
        self.write_object(graph, path, obj, cap)
    }

    /// The content id a path currently aliases, if it is a file.
    pub fn resolve(&self, path: &str) -> Option<ObjectId> {
        match self.node_at(&Self::components(path)) {
            Some(Node::File(id)) => Some(*id),
            _ => None,
        }
    }

    /// Read the object a path aliases. Requires `READ`.
    pub fn read_object<'g>(
        &self,
        graph: &'g ObjectGraph,
        path: &str,
        cap: &Capability,
    ) -> Result<&'g Object, VfsError> {
        require(cap, Rights::READ)?;
        let id = match self.node_at(&Self::components(path)) {
            Some(Node::File(id)) => *id,
            Some(Node::Dir(_)) => return Err(VfsError::IsADirectory),
            None => return Err(VfsError::NotFound),
        };
        graph.get(&id).ok_or(VfsError::DanglingAlias)
    }

    /// Read a file's raw bytes — verbatim for a `Blob`. Requires `READ`.
    pub fn read_bytes(
        &self,
        graph: &ObjectGraph,
        path: &str,
        cap: &Capability,
    ) -> Result<Vec<u8>, VfsError> {
        let obj = self.read_object(graph, path, cap)?;
        match Blob::bytes_of(obj) {
            Some(b) => Ok(b.to_vec()),
            None => Err(VfsError::NotADirectory), // a non-blob file has no raw bytes
        }
    }

    pub fn exists(&self, path: &str) -> bool {
        self.node_at(&Self::components(path)).is_some()
    }

    pub fn is_dir(&self, path: &str) -> bool {
        matches!(self.node_at(&Self::components(path)), Some(Node::Dir(_)))
    }

    pub fn is_file(&self, path: &str) -> bool {
        matches!(self.node_at(&Self::components(path)), Some(Node::File(_)))
    }

    /// List the entry names of a directory, sorted (the map is already ordered).
    pub fn list(&self, path: &str) -> Result<Vec<String>, VfsError> {
        match self.node_at(&Self::components(path)) {
            Some(Node::Dir(m)) => Ok(m.keys().cloned().collect()),
            Some(Node::File(_)) => Err(VfsError::NotADirectory),
            None => Err(VfsError::NotFound),
        }
    }

    /// Remove a file alias or an empty directory. The underlying object is *not*
    /// deleted from the graph — it remains in history (immutable). Requires `WRITE`.
    pub fn remove(&mut self, path: &str, cap: &Capability) -> Result<(), VfsError> {
        require(cap, Rights::WRITE)?;
        let comps = Self::components(path);
        let (name, dirs) = comps.split_last().ok_or(VfsError::InvalidPath)?;
        let map = self.ensure_dir_mut(dirs)?;
        match map.get(name) {
            None => Err(VfsError::NotFound),
            Some(Node::Dir(m)) if !m.is_empty() => Err(VfsError::NotEmpty),
            Some(_) => {
                map.remove(name);
                Ok(())
            }
        }
    }

    /// Materialise the whole namespace into the graph as a tree of `Dir` objects,
    /// returning the root directory's content id. Identical trees produce the
    /// identical id (content addressing); any change anywhere changes the root.
    pub fn snapshot_namespace(&self, graph: &mut ObjectGraph) -> ObjectId {
        Self::snapshot_node(&self.root, graph)
    }

    /// Rebuild a namespace from a previously [`snapshot_namespace`](Self::snapshot_namespace)ed
    /// tree: `root` must name a `Dir` object in `graph`. The inverse of snapshotting —
    /// used to restore the filesystem from a persisted image. Returns `None` if `root`
    /// is missing or is not a directory.
    pub fn from_namespace(graph: &ObjectGraph, root: ObjectId) -> Option<Vfs> {
        let node = Self::load_node(graph, root)?;
        match node {
            Node::Dir(_) => Some(Vfs { root: node }),
            Node::File(_) => None,
        }
    }

    /// Reconstruct one namespace node from the graph: a `Dir` object becomes a
    /// directory of its referenced children; anything else is a file aliasing that id.
    fn load_node(graph: &ObjectGraph, id: ObjectId) -> Option<Node> {
        let obj = graph.get(&id)?;
        if obj.kind == DIR_KIND {
            let mut map = BTreeMap::new();
            for (name, datum) in &obj.fields {
                if let Datum::Ref(cid) = datum {
                    if let Some(child) = Self::load_node(graph, *cid) {
                        map.insert(name.clone(), child);
                    }
                }
            }
            Some(Node::Dir(map))
        } else {
            Some(Node::File(id))
        }
    }

    fn snapshot_node(node: &Node, graph: &mut ObjectGraph) -> ObjectId {
        match node {
            Node::File(id) => *id,
            Node::Dir(m) => {
                let mut obj = Object::new(DIR_KIND);
                for (name, child) in m {
                    let cid = Self::snapshot_node(child, graph);
                    obj.set(name, Datum::Ref(cid));
                }
                graph.put(obj)
            }
        }
    }

    /// Snapshot the namespace into the graph, then commit the graph — capturing
    /// the entire current filesystem state as one restorable point. Returns the
    /// commit root. Requires `WRITE` (it mutates the graph head).
    pub fn commit(
        &self,
        graph: &mut ObjectGraph,
        message: &str,
        cap: &Capability,
    ) -> Result<Hash256, VfsError> {
        require(cap, Rights::WRITE)?;
        self.snapshot_namespace(graph);
        Ok(graph.commit(message))
    }
}

impl Default for Vfs {
    fn default() -> Self {
        Self::with_fhs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;

    fn rw() -> Capability {
        Capability::mint(0, 0x1000, Rights::ALL)
    }
    fn ro() -> Capability {
        Capability::mint(0, 0x1000, Rights::READ)
    }

    fn text(s: &str) -> Object {
        Object::new("Text").with("content", Datum::Text(s.to_string()))
    }

    #[test]
    fn write_then_read_round_trips() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/etc/motd", text("welcome"), &cap).unwrap();
        let got = v.read_object(&g, "/etc/motd", &cap).unwrap();
        assert_eq!(got.get("content"), Some(&Datum::Text("welcome".to_string())));
    }

    #[test]
    fn nested_write_autocreates_parents() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        v.write_object(&mut g, "/a/b/c/d.txt", text("x"), &rw()).unwrap();
        assert!(v.is_dir("/a/b/c"));
        assert!(v.is_file("/a/b/c/d.txt"));
    }

    #[test]
    fn list_is_sorted() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/d/zebra", text("1"), &cap).unwrap();
        v.write_object(&mut g, "/d/apple", text("2"), &cap).unwrap();
        v.write_object(&mut g, "/d/mango", text("3"), &cap).unwrap();
        assert_eq!(v.list("/d").unwrap(), ["apple", "mango", "zebra"]);
    }

    #[test]
    fn editing_a_path_creates_new_immutable_object() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        let id1 = v.write_object(&mut g, "/f", text("v1"), &cap).unwrap();
        let id2 = v.write_object(&mut g, "/f", text("v2"), &cap).unwrap();
        assert_ne!(id1, id2);
        // The alias now points at v2 ...
        assert_eq!(v.resolve("/f"), Some(id2));
        // ... but the original object was never mutated and is still retrievable.
        assert_eq!(g.get(&id1).unwrap().get("content"), Some(&Datum::Text("v1".to_string())));
    }

    #[test]
    fn write_requires_write_capability() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let err = v.write_object(&mut g, "/f", text("x"), &ro()).unwrap_err();
        assert_eq!(err, VfsError::Capability(CapError::InsufficientRights));
    }

    #[test]
    fn read_requires_read_capability() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        v.write_object(&mut g, "/f", text("x"), &rw()).unwrap();
        let write_only = Capability::mint(0, 0x1000, Rights::WRITE);
        let err = v.read_object(&g, "/f", &write_only).unwrap_err();
        assert_eq!(err, VfsError::Capability(CapError::InsufficientRights));
    }

    #[test]
    fn tampered_capability_traps() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let err = v.write_object(&mut g, "/f", text("x"), &rw().tamper()).unwrap_err();
        assert_eq!(err, VfsError::Capability(CapError::TagInvalid));
    }

    #[test]
    fn remove_unlinks_alias_but_keeps_object() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        let id = v.write_object(&mut g, "/f", text("keep"), &cap).unwrap();
        v.remove("/f", &cap).unwrap();
        assert!(!v.exists("/f"));
        // Object survives in the graph (immutable history).
        assert!(g.contains(&id));
    }

    #[test]
    fn remove_nonempty_dir_errors_but_empty_ok() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/dir/inner", text("x"), &cap).unwrap();
        assert_eq!(v.remove("/dir", &cap).unwrap_err(), VfsError::NotEmpty);
        v.mkdir_p("/empty", &cap).unwrap();
        assert!(v.remove("/empty", &cap).is_ok());
    }

    #[test]
    fn writing_through_a_file_component_errors() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/a", text("file"), &cap).unwrap();
        // /a is a file; /a/b treats it as a dir.
        assert_eq!(v.write_object(&mut g, "/a/b", text("x"), &cap).unwrap_err(), VfsError::NotADirectory);
    }

    #[test]
    fn dotdot_normalisation() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/x/y/../z", text("here"), &cap).unwrap();
        assert!(v.is_file("/x/z"));
        assert!(!v.exists("/x/y"));
    }

    #[test]
    fn snapshot_is_deterministic_and_change_sensitive() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/etc/a", text("1"), &cap).unwrap();
        let r1 = v.snapshot_namespace(&mut g);
        let r1b = v.snapshot_namespace(&mut g);
        assert_eq!(r1, r1b, "same tree must snapshot to the same root");
        v.write_object(&mut g, "/etc/b", text("2"), &cap).unwrap();
        let r2 = v.snapshot_namespace(&mut g);
        assert_ne!(r1, r2, "a changed tree must change the root");
    }

    #[test]
    fn commit_records_history() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/f", text("x"), &cap).unwrap();
        assert_eq!(g.history().len(), 0);
        v.commit(&mut g, "first snapshot", &cap).unwrap();
        assert_eq!(g.history().len(), 1);
    }

    #[test]
    fn snapshot_then_from_namespace_round_trips_the_tree() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        v.write_object(&mut g, "/etc/motd", text("welcome"), &cap).unwrap();
        v.write_object(&mut g, "/home/jay/notes.txt", text("hi"), &cap).unwrap();
        v.mkdir_p("/home/jay/empty", &cap).unwrap();
        let root = v.snapshot_namespace(&mut g);
        // Rebuild a fresh Vfs purely from the persisted graph + root id.
        let restored = Vfs::from_namespace(&g, root).expect("root is a dir");
        assert!(restored.is_dir("/etc"));
        assert!(restored.is_file("/etc/motd"));
        assert!(restored.is_file("/home/jay/notes.txt"));
        assert!(restored.is_dir("/home/jay/empty"));
        // Listings match.
        assert_eq!(restored.list("/home/jay").unwrap(), ["empty", "notes.txt"]);
        // The aliased content is still readable.
        let got = restored.read_object(&g, "/etc/motd", &cap).unwrap();
        assert_eq!(got.get("content"), Some(&Datum::Text("welcome".to_string())));
    }

    #[test]
    fn from_namespace_rejects_a_non_dir_root() {
        let mut g = ObjectGraph::new();
        let file_id = g.put(text("x"));
        assert!(Vfs::from_namespace(&g, file_id).is_none());
    }

    #[test]
    fn fhs_skeleton_present() {
        let v = Vfs::with_fhs();
        assert!(v.is_dir("/etc"));
        assert!(v.is_dir("/usr/lib"));
        assert!(v.is_dir("/tmp"));
    }

    #[test]
    fn legacy_file_round_trips_through_codec_and_vfs() {
        // The full keystone story: import legacy bytes -> semantic object -> store
        // at a path -> read back -> export to identical legacy bytes.
        let reg = CodecRegistry::with_defaults();
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();

        let raw = b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let img = reg.import(Some("logo.ppm"), raw, &cap).unwrap();
        v.write_object(&mut g, "/usr/share/logo.ppm", img, &cap).unwrap();

        let stored = v.read_object(&g, "/usr/share/logo.ppm", &cap).unwrap();
        let exported = reg.export(stored, &cap).unwrap();
        // Re-importing the export reproduces the same semantic object: lossless.
        let reimported = reg.import(Some("logo.ppm"), &exported, &cap).unwrap();
        assert_eq!(stored.id(), reimported.id());
    }

    #[test]
    fn write_and_read_raw_bytes() {
        let mut g = ObjectGraph::new();
        let mut v = Vfs::new();
        let cap = rw();
        let data = &[0u8, 1, 2, 250, 251, 0];
        v.write_bytes(&mut g, "/tmp/raw.bin", "application/octet-stream", data, &cap).unwrap();
        assert_eq!(v.read_bytes(&g, "/tmp/raw.bin", &cap).unwrap(), data);
    }

    #[test]
    fn path_components_are_canonical_and_cannot_escape_root() {
        // Adversarial paths: dot/dot-dot games, empty segments, trailing slashes, and
        // traversal attempts. `components` must fully normalise every one.
        let adversarial = [
            "/", "", ".", "..", "/..", "/../..", "/../../etc/passwd",
            "////a////b", "/a/./b/../c", "a/../../../../b", "/a/b/../../..",
            "/./././.", "/a/..", "/foo/bar/../baz", "no/leading/slash",
            "/trailing/slash/", "/weird//empty///segs", "/.../..../a",
        ];
        for p in adversarial {
            let comps = Vfs::components(p);
            // No normalised component is ever empty, ".", or "..": a resolved path can
            // never reference its own parent or self, so it cannot be tricked into
            // aliasing a node outside the subtree the caller named.
            for c in &comps {
                assert!(!c.is_empty(), "empty segment from {:?}", p);
                assert_ne!(c, ".", "'.' survived in {:?}", p);
                assert_ne!(c, "..", "'..' survived in {:?}", p);
            }
            // Canonicalisation is a fixed point: re-normalising changes nothing.
            let joined = alloc::format!("/{}", comps.join("/"));
            assert_eq!(Vfs::components(&joined), comps, "not idempotent for {:?}", p);
        }
        // No quantity of leading `..` can climb above the root — extra parent refs on
        // an already-empty stack are absorbed, never producing an escaping path.
        let escaped = Vfs::components("/../../../../../../etc/shadow");
        assert_eq!(escaped.len(), 2);
        assert_eq!(escaped[0], "etc");
        assert_eq!(escaped[1], "shadow");
    }

    #[test]
    fn deeply_nested_paths_resolve_without_overflow() {
        // Build a 512-deep path and confirm normalisation handles it (no stack blow-up,
        // correct depth) — the recursive-deletion / deep-tree case from the audit.
        let mut path = String::new();
        for i in 0..512 {
            path.push('/');
            path.push_str(&alloc::format!("d{}", i));
        }
        let comps = Vfs::components(&path);
        assert_eq!(comps.len(), 512);
        assert_eq!(comps[0], "d0");
        assert_eq!(comps[511], "d511");
        // A trailing pile of `..` collapses it right back to the root.
        let mut climb = path.clone();
        for _ in 0..512 {
            climb.push_str("/..");
        }
        assert!(Vfs::components(&climb).is_empty());
    }
}

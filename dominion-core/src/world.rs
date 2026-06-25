//! The **World** — DominionOS's live system model that the shell's pages share.
//!
//! Before this, the Desktop launcher, the Explorer and the IDE each carried their
//! own hard-coded mock data. The World makes them three views of **one coherent
//! object graph** instead:
//!
//! * **Programs** authored in the IDE are real semantic [`Object`]s (kind `"Program"`).
//! * **Datasets / reports / logs / channels / codecs** are real objects in a real
//!   [`ObjectGraph`], linked by real `Ref` edges (the knowledge-graph wires).
//! * **Sandboxed cells** are real capability-bounded [`Sandbox`]es.
//! * Every object carries a **real capability** derived from the TCB root, so the
//!   rights (r/w/x) and **provenance** hash chain the Explorer shows are genuine —
//!   derived through [`Capability::derive`], not decorative strings.
//!
//! The shell ([`crate::os::Os`]) owns one `World`, syncs the IDE's programs into it,
//! and feeds the enumerated [`Entry`] list to the Desktop and Explorer. Pure, safe
//! `no_std`.

use crate::capability::{Capability, Rights};
use crate::hash::Hash256;
use crate::object::{Datum, Object, ObjectGraph, ObjectId};
use crate::sandbox::Sandbox;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The TCB root region the whole capability tree derives from (the `mint` at boot).
const ROOT_BASE: u64 = 0x4000_0000;
const ROOT_LEN: u64 = 0x1000_0000;

/// A display-ready enumerated object: identity + semantic metadata + the **real**
/// capability the system gates access to it through.
pub struct Entry {
    pub id: ObjectId,
    pub kind: String,
    pub name: String,
    /// Metadata key/value lines (already resolved — `Ref`s show the target name).
    pub meta: Vec<(String, String)>,
    pub rights: Rights,
    /// The provenance fingerprint of this object's derived capability.
    pub provenance: Hash256,
    /// Object ids this object points at (the knowledge-graph edges).
    pub refs: Vec<ObjectId>,
    /// Program source — present only for `kind == "Program"`, so the IDE can open it.
    pub source: Option<String>,
}

impl Entry {
    /// `rwx` rights string (the three user-facing capability bits).
    pub fn rights_str(&self) -> String {
        let mut s = String::new();
        s.push(if self.rights.contains(Rights::READ) { 'r' } else { '-' });
        s.push(if self.rights.contains(Rights::WRITE) { 'w' } else { '-' });
        s.push(if self.rights.contains(Rights::EXECUTE) { 'x' } else { '-' });
        s
    }
}

/// A sandboxed compartment, enumerated for display.
pub struct CellInfo {
    pub name: String,
    pub bounds: String,
    pub syscalls: u32,
}

/// The live system model.
pub struct World {
    graph: ObjectGraph,
    /// Programs authored in the IDE (kind `"Program"`), kept separate from the seeded
    /// graph so the IDE can replace the whole set without fighting content-addressing.
    programs: Vec<Object>,
    sandboxes: Vec<Sandbox>,
    root: Capability,
}

impl World {
    pub fn new() -> World {
        World {
            graph: seed_graph(),
            programs: Vec::new(),
            sandboxes: seed_sandboxes(),
            root: Capability::mint(ROOT_BASE, ROOT_LEN, Rights::ALL),
        }
    }

    /// The TCB root capability (its provenance names the whole system's authority).
    pub fn root_cap(&self) -> &Capability {
        &self.root
    }

    /// The Merkle root over the live object set — a single hash naming system state.
    pub fn root_hash(&self) -> Hash256 {
        self.graph.root_hash()
    }

    pub fn sandboxes(&self) -> &[Sandbox] {
        &self.sandboxes
    }

    /// The enumerated compartments for the Explorer's secure-compartment viewer.
    pub fn cells(&self) -> Vec<CellInfo> {
        self.sandboxes
            .iter()
            .map(|s| {
                let (base, len) = s.region();
                CellInfo {
                    name: s.name.clone(),
                    bounds: fmt_region(base, len),
                    syscalls: s.syscall_count() as u32,
                }
            })
            .collect()
    }

    /// Replace the program set from the IDE's `(name, source)` snapshot. Cheap and
    /// idempotent: identical snapshots rebuild identical (content-addressed) objects.
    pub fn set_programs(&mut self, progs: &[(String, String)]) {
        self.programs = progs
            .iter()
            .map(|(name, src)| {
                Object::new("Program")
                    .with("name", Datum::Text(name.clone()))
                    .with("source", Datum::Text(src.clone()))
            })
            .collect();
    }

    /// Enumerate every live object — programs first, then the seeded graph — enriched
    /// with its real capability rights + provenance and resolved metadata.
    pub fn entries(&self) -> Vec<Entry> {
        // Build a name index so `Ref` metadata resolves to readable target names.
        let mut names: BTreeMap<ObjectId, String> = BTreeMap::new();
        for o in &self.programs {
            names.insert(o.id(), object_name(o));
        }
        for id in self.graph.live_ids() {
            if let Some(o) = self.graph.get(id) {
                names.insert(*id, object_name(o));
            }
        }

        let mut out = Vec::new();
        for o in &self.programs {
            out.push(self.entry_for(o, &names));
        }
        for id in self.graph.live_ids() {
            if let Some(o) = self.graph.get(id) {
                out.push(self.entry_for(o, &names));
            }
        }

        // Programs consume data: link each to the first Dataset so the constellation
        // shows the dataflow (a synthesized edge — programs reference data by name in
        // their source, not by content hash).
        if let Some(ds) = out.iter().find(|e| e.kind == "Dataset").map(|e| e.id) {
            for e in out.iter_mut().filter(|e| e.kind == "Program") {
                if !e.refs.contains(&ds) {
                    e.refs.push(ds);
                }
            }
        }
        out
    }

    fn entry_for(&self, o: &Object, names: &BTreeMap<ObjectId, String>) -> Entry {
        let id = o.id();
        let rights = rights_for_kind(&o.kind);
        let cap = self.derive_for(&id, rights);
        let mut meta = Vec::new();
        let mut refs = Vec::new();
        for (k, v) in &o.fields {
            if k == "name" || k == "source" {
                continue;
            }
            match v {
                Datum::Ref(h) => {
                    refs.push(*h);
                    let target = names.get(h).cloned().unwrap_or_else(|| h.short());
                    meta.push((k.clone(), {
                        let mut s = String::from("→ ");
                        s.push_str(&target);
                        s
                    }));
                }
                other => meta.push((k.clone(), datum_str(other))),
            }
        }
        Entry {
            id,
            kind: o.kind.clone(),
            name: object_name(o),
            meta,
            rights,
            provenance: cap.provenance(),
            refs,
            source: o.get("source").and_then(|d| match d {
                Datum::Text(s) => Some(s.clone()),
                _ => None,
            }),
        }
    }

    /// Create a new named folder entry in the object graph.
    pub fn add_folder(&mut self, name: &str) {
        let obj = Object::new("Folder")
            .with("name", Datum::Text(name.into()));
        self.graph.put(obj);
    }

    /// Create a shortcut entry pointing to `target`.
    pub fn add_shortcut(&mut self, name: &str, target: &str) {
        let obj = Object::new("Shortcut")
            .with("name", Datum::Text(name.into()))
            .with("target", Datum::Text(target.into()));
        self.graph.put(obj);
    }

    /// A monotonically increasing counter for generating unique names.
    pub fn next_entry_id(&self) -> usize {
        self.graph.live_ids().len()
    }

    /// Derive an object's capability from the TCB root: a distinct sub-region keyed
    /// by content hash, attenuated to the kind's rights. Genuine provenance chain.
    fn derive_for(&self, id: &ObjectId, rights: Rights) -> Capability {
        let slot = id.0[0] as u64;
        let base = ROOT_BASE + slot * 0x1_0000;
        self.root.derive(base, 0x1_0000, rights).unwrap_or(self.root)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

/// The rights a given object kind is granted (least authority that still works).
fn rights_for_kind(kind: &str) -> Rights {
    match kind {
        "Program" | "Codec" => Rights::READ.union(Rights::EXECUTE),
        "Dataset" | "Channel" => Rights::READ.union(Rights::WRITE),
        _ => Rights::READ,
    }
}

fn object_name(o: &Object) -> String {
    match o.get("name") {
        Some(Datum::Text(s)) => s.clone(),
        _ => o.kind.clone(),
    }
}

fn datum_str(d: &Datum) -> String {
    match d {
        Datum::Int(i) => int_str(*i),
        Datum::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Datum::Text(s) => s.clone(),
        Datum::Float(_) => "…".to_string(),
        Datum::Ref(h) => {
            let mut s = String::from("→ ");
            s.push_str(&h.short());
            s
        }
        Datum::Bytes(_) => "<bytes>".to_string(),
    }
}

fn fmt_region(base: u64, len: u64) -> String {
    let mut s = String::from("[0x");
    push_hex(&mut s, base);
    s.push_str(", +");
    s.push_str(&int_str((len / (1024 * 1024)) as i64));
    s.push_str("M)");
    s
}

fn push_hex(s: &mut String, v: u64) {
    let mut started = false;
    for i in (0..16).rev() {
        let nib = ((v >> (i * 4)) & 0xf) as u8;
        if nib != 0 || started || i == 0 {
            started = true;
            s.push(if nib < 10 { (b'0' + nib) as char } else { (b'A' + nib - 10) as char });
        }
    }
}

fn int_str(n: i64) -> String {
    let mut s = String::new();
    push_int(&mut s, n);
    s
}

fn push_int(s: &mut String, mut n: i64) {
    if n < 0 {
        s.push('-');
        n = -n;
    }
    if n >= 10 {
        push_int(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

/// Seed the system object graph with representative datasets / reports / logs /
/// channels / codecs / token sets, linked by real `Ref` edges.
fn seed_graph() -> ObjectGraph {
    let mut g = ObjectGraph::new();
    let sales = g.put(
        Object::new("Dataset")
            .with("name", Datum::Text("SALES DATA 2024".into()))
            .with("rows", Datum::Int(4096))
            .with("origin", Datum::Text("ingested".into())),
    );
    g.put(
        Object::new("Report")
            .with("name", Datum::Text("Q3 PERFORMANCE REPORT".into()))
            .with("derived_from", Datum::Ref(sales))
            .with("generated", Datum::Bool(true)),
    );
    let logs = g.put(
        Object::new("Log")
            .with("name", Datum::Text("SYS LOGS /v1".into()))
            .with("append_only", Datum::Bool(true)),
    );
    g.put(
        Object::new("Channel")
            .with("name", Datum::Text("TEAM COMMS".into()))
            .with("members", Datum::Int(5))
            .with("journal", Datum::Ref(logs)),
    );
    g.put(
        Object::new("Codec")
            .with("name", Datum::Text("NEURAL CODECS".into()))
            .with("ratio", Datum::Text("12.4x".into()))
            .with("trained_on", Datum::Ref(sales)),
    );
    g.put(
        Object::new("TokenSet")
            .with("name", Datum::Text("CAPABILITY TOKENS".into()))
            .with("root", Datum::Text("minted".into())),
    );
    g
}

/// Seed the live sandboxes — real capability-bounded compartments derived from root.
fn seed_sandboxes() -> Vec<Sandbox> {
    let root = Capability::mint(ROOT_BASE, ROOT_LEN, Rights::ALL);
    let rw = Rights::READ.union(Rights::WRITE);
    let mut v = Vec::new();

    if let Ok(cap) = root.derive(0x4000_0000, 0x0400_0000, rw) {
        let mut sb = Sandbox::new("linux-guest", cap, "/containers/guest1");
        sb.allow_syscalls(&[0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 16, 21]);
        v.push(sb);
    }
    if let Ok(cap) = root.derive(0x4400_0000, 0x0200_0000, rw) {
        let mut sb = Sandbox::new("browser-sandbox", cap, "/containers/web");
        sb.allow_syscalls(&[0, 1, 2, 3, 8, 9]);
        v.push(sb);
    }
    if let Ok(cap) = root.derive(0x4600_0000, 0x0100_0000, Rights::READ.union(Rights::EXECUTE)) {
        let mut sb = Sandbox::new("npu-runtime", cap, "/containers/npu");
        sb.allow_syscalls(&[0, 1, 9]);
        v.push(sb);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_seeded_objects_with_real_capabilities() {
        let w = World::new();
        let entries = w.entries();
        // The seeded objects are all present.
        for name in ["SALES DATA 2024", "Q3 PERFORMANCE REPORT", "SYS LOGS /v1", "NEURAL CODECS"] {
            assert!(entries.iter().any(|e| e.name == name), "missing {}", name);
        }
        // Rights are real and kind-appropriate.
        let dataset = entries.iter().find(|e| e.name == "SALES DATA 2024").unwrap();
        assert_eq!(dataset.rights_str(), "rw-");
        let report = entries.iter().find(|e| e.kind == "Report").unwrap();
        assert_eq!(report.rights_str(), "r--");
        // Provenance is a genuine derived hash chain (not zero).
        assert_ne!(dataset.provenance, Hash256::ZERO);
        assert_ne!(dataset.provenance, report.provenance);
    }

    #[test]
    fn ref_edges_resolve_to_target_names() {
        let w = World::new();
        let entries = w.entries();
        let report = entries.iter().find(|e| e.kind == "Report").unwrap();
        // The report's `derived_from` ref points at the sales dataset.
        assert!(report.meta.iter().any(|(k, v)| k == "derived_from" && v == "→ SALES DATA 2024"));
        let sales_id = entries.iter().find(|e| e.name == "SALES DATA 2024").unwrap().id;
        assert!(report.refs.contains(&sales_id));
    }

    #[test]
    fn ide_programs_become_first_class_objects() {
        let mut w = World::new();
        w.set_programs(&[("payroll.aeth".to_string(), "let net = gross - tax;".to_string())]);
        let entries = w.entries();
        let prog = entries.iter().find(|e| e.name == "payroll.aeth").unwrap();
        assert_eq!(prog.kind, "Program");
        assert_eq!(prog.rights_str(), "r-x");
        assert_eq!(prog.source.as_deref(), Some("let net = gross - tax;"));
        // A program is linked to the first dataset (consumes data).
        let sales_id = entries.iter().find(|e| e.name == "SALES DATA 2024").unwrap().id;
        assert!(prog.refs.contains(&sales_id));
        // Programs are enumerated before the seeded graph.
        assert_eq!(entries[0].kind, "Program");
    }

    #[test]
    fn cells_are_real_capability_bounded_sandboxes() {
        let w = World::new();
        let cells = w.cells();
        assert_eq!(cells.len(), 3);
        let guest = cells.iter().find(|c| c.name == "linux-guest").unwrap();
        assert!(guest.syscalls > 0);
        assert!(guest.bounds.starts_with("[0x4000"), "bounds: {}", guest.bounds);
    }
}

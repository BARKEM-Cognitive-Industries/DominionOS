//! Unified discovery catalog — one searchable index over everything a Dominion
//! author can reach from the IDE: packages, libraries, drivers, programs, and
//! sub-nodes. The IDE/LSP fold [`search`] results into autocomplete so a writer
//! can find and wire any reachable building block without leaving the editor.
//!
//! OWNED BY: Discovery agent (workstream D). The public signatures below are the
//! frozen handoff to the IDE/LSP agent (workstream I) and MUST keep these exact
//! names/shapes; the agent fills the bodies with real aggregation over
//! `packaging`, `drivergen`, `polyglot`, `nodes`, and the IDE program list.
//!
//! Two layers:
//!
//! * The free [`catalog`]/[`search`] functions return a **curated, always-present
//!   set** — the standard libraries, device-class drivers, and language builtins
//!   that ship with the OS — so the IDE has something to show with no live state.
//!   The static set is derived straight from the real subsystem enums
//!   ([`crate::polyglot::Language`], [`crate::drivergen::HwClass`]) so it never
//!   drifts from what the runtime actually supports.
//! * A [`Catalog`] instance merges that static set with **live** state the IDE
//!   feeds in: installed [`crate::packaging::Package`]s, open programs, and the
//!   sub-nodes of a [`crate::nodes::NodeGraph`]. Entries are kept sorted (by kind,
//!   then name) so the index is deterministic and testable.

use crate::drivergen::HwClass;
use crate::nodes::{NodeGraph, NodeKind};
use crate::packaging::{Package, PackageRegistry};
use crate::polyglot::Language;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// What kind of building block a catalog entry refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ItemKind {
    Package,
    Library,
    Driver,
    Program,
    SubNode,
    Builtin,
}

impl ItemKind {
    /// A stable sort rank so the catalog has a deterministic kind ordering.
    fn rank(self) -> u8 {
        match self {
            ItemKind::Package => 0,
            ItemKind::Library => 1,
            ItemKind::Driver => 2,
            ItemKind::Program => 3,
            ItemKind::SubNode => 4,
            ItemKind::Builtin => 5,
        }
    }
}

/// One discoverable building block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogItem {
    pub name: String,
    pub kind: ItemKind,
    pub summary: String,
}

impl CatalogItem {
    fn new(name: &str, kind: ItemKind, summary: &str) -> CatalogItem {
        CatalogItem { name: name.to_string(), kind, summary: summary.to_string() }
    }
}

// ───────────────────────── static source tables ─────────────────────────

/// The standard library packages every guest can `import` (from `polyglot`'s
/// package registry: `mathx`, `stats`, `strx`), each surfaced as a [`Library`].
///
/// [`Library`]: ItemKind::Library
const STD_LIBRARIES: &[(&str, &str)] = &[
    ("mathx", "Standard math library: sqrt, pow, gcd, factorial, isqrt, fib, floor, ceil"),
    ("stats", "Statistics library: mean, variance, stdev, median (sample + population)"),
    ("strx", "String library: upper, lower, repeat, reverse_str, concat"),
];

/// The Dominion-language interpreter builtins (always available, no import), with a
/// one-line summary each. The IDE shows these as `Builtin` autocomplete entries.
const BUILTINS: &[(&str, &str)] = &[
    ("print", "Print values to the program output"),
    ("len", "Length of a vector or string"),
    ("push", "Append an element to a vector"),
    ("sum", "Sum the elements of a vector"),
    ("range", "Build a vector of integers over a half-open range"),
    ("load", "Load a dataset / resource by name"),
    ("summarise", "Summary statistics over a vector"),
    ("hash", "Content hash of a value"),
    ("tensor", "Construct a dense tensor"),
    ("matmul", "Matrix multiply two tensors"),
    ("hypervector", "Build a high-dimensional hypervector"),
    ("bind", "Bind two hypervectors (VSA binding)"),
    ("mlp", "Construct a multilayer perceptron"),
    ("predict", "Run a model forward pass"),
    ("train_xor", "Train the demo XOR network"),
    ("nn_loss", "Neural-network loss over predictions"),
    ("route", "Route a value through a placement graph"),
    ("abs", "Absolute value"),
    ("max", "Maximum of the arguments"),
    ("floor", "Round down to an integer"),
    ("ceil", "Round up to an integer"),
    ("round", "Round to the nearest integer"),
    ("sqrt", "Square root"),
    ("pow", "Raise a base to a power"),
    ("str", "Convert a value to a string"),
    ("int", "Convert a value to an integer"),
    ("float", "Convert a value to a float"),
    ("get", "Index into a vector or map"),
    ("first", "First element of a vector"),
    ("last", "Last element of a vector"),
    ("reverse", "Reverse a vector"),
    ("concat", "Concatenate two vectors"),
    ("slice", "Sub-range of a vector"),
    ("sort", "Sort a vector"),
    ("product", "Product of the elements of a vector"),
    ("contains", "Whether a vector/string contains a value"),
    ("upper", "Uppercase a string"),
    ("lower", "Lowercase a string"),
    ("trim", "Trim surrounding whitespace"),
    ("split", "Split a string on a separator"),
    ("join", "Join a vector of strings"),
    ("chars", "Characters of a string as a vector"),
    ("starts_with", "Whether a string starts with a prefix"),
    ("ends_with", "Whether a string ends with a suffix"),
    ("replace", "Replace occurrences within a string"),
    ("decimal", "Construct an arbitrary-precision decimal"),
    ("dec_div", "Decimal division"),
    ("dec_sqrt", "Decimal square root"),
    ("dec_round", "Round a decimal"),
    ("bigint", "Construct an arbitrary-precision integer"),
    ("rational", "Construct an exact rational number"),
    ("to_decimal", "Convert to a decimal"),
    ("complex", "Construct a complex number"),
    ("conj", "Complex conjugate"),
    ("cabs", "Magnitude of a complex number"),
    ("dual", "Construct a dual number (forward-mode AD)"),
    ("dvar", "Dual variable (seeded derivative)"),
    ("dconst", "Dual constant"),
    ("dsqrt", "Dual-number square root"),
    ("dpow", "Dual-number power"),
    ("interval", "Construct an interval"),
    ("ihull", "Interval hull of two intervals"),
    ("icontains", "Whether an interval contains a value"),
    ("quat", "Construct a quaternion"),
    ("qnorm", "Quaternion norm"),
    ("qnormalize", "Normalize a quaternion"),
    // string extras
    ("pad_left", "Left-pad a string to a given width"),
    ("pad_right", "Right-pad a string to a given width"),
    ("repeat_str", "Repeat a string n times"),
    ("lines", "Split a string on newlines into a vector"),
    // collection extras
    ("flatten", "Flatten one level of nesting in a vector"),
    ("zip", "Zip two vectors into pairs [[a0,b0],[a1,b1],...]"),
    ("enumerate", "Pair each element with its index [[0,v0],[1,v1],...]"),
    ("unique", "Remove duplicates preserving insertion order"),
    ("sum_f", "Sum a vector of numbers as a Float"),
    ("count_matches", "Count occurrences of a value in a vector"),
    ("keys", "Field names of an Object as a vector of strings"),
    ("values", "Field values of an Object as a vector"),
    // numeric extras
    ("clamp", "Clamp a number to [lo, hi]"),
    ("lerp", "Linear interpolation: a + t*(b-a)"),
    ("sign", "Sign of a number: -1, 0, or 1"),
    // numeric aliases already present but missing from earlier list
    ("min", "Minimum of the arguments or a single vector"),
];

/// One-line description for a hardware device class, used as a [`Driver`] summary.
///
/// [`Driver`]: ItemKind::Driver
fn hwclass_summary(class: HwClass) -> &'static str {
    match class {
        HwClass::Nvme => "NVMe block device — synthesized block driver (class L1)",
        HwClass::Ahci => "AHCI/SATA block device — synthesized block driver (class L1)",
        HwClass::Xhci => "xHCI USB controller — synthesized input driver (class L1)",
        HwClass::HdAudio => "HD-Audio controller — synthesized console driver (class L1)",
        HwClass::VirtioBlock => "virtio-blk paravirtual block device (class L1)",
        HwClass::VirtioNet => "virtio-net paravirtual network device (class L1)",
    }
}

/// The lowercase, stable catalog name for a hardware class.
fn hwclass_name(class: HwClass) -> &'static str {
    match class {
        HwClass::Nvme => "nvme",
        HwClass::Ahci => "ahci",
        HwClass::Xhci => "xhci",
        HwClass::HdAudio => "hdaudio",
        HwClass::VirtioBlock => "virtio-block",
        HwClass::VirtioNet => "virtio-net",
    }
}

/// Every hardware class the synthesis ladder knows how to drive, in a stable order
/// (mirrors `drivergen`'s [`HwClass`] variants — kept here so the catalog is
/// deterministic; new variants should be added alongside the enum).
const HW_CLASSES: &[HwClass] = &[
    HwClass::Nvme,
    HwClass::Ahci,
    HwClass::Xhci,
    HwClass::HdAudio,
    HwClass::VirtioBlock,
    HwClass::VirtioNet,
];

/// The summary string for a node-graph kind, when surfaced as a sub-node entry.
fn nodekind_summary(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::App => "Sub-node: running application / view",
        NodeKind::Data => "Sub-node: data object in the graph",
        NodeKind::Audio => "Sub-node: neural audio object",
        NodeKind::Report => "Sub-node: generated report",
        NodeKind::Log => "Sub-node: system log stream",
        NodeKind::Program => "Sub-node: embedded program / project reference",
    }
}

// ───────────────────────── the live catalog ─────────────────────────

/// A unified, searchable catalog of everything reachable from the IDE. Start from
/// the static [`Catalog::standard`] set and fold in live state (installed packages,
/// open programs, a node graph's sub-nodes); entries stay sorted so the index is
/// deterministic.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    items: Vec<CatalogItem>,
}

impl Catalog {
    /// An empty catalog.
    pub fn new() -> Catalog {
        Catalog { items: Vec::new() }
    }

    /// A catalog pre-loaded with the always-present built-in set (standard
    /// libraries, device-class drivers, and language builtins).
    pub fn with_builtins() -> Catalog {
        let mut c = Catalog::new();
        c.extend_libraries();
        c.extend_drivers();
        c.extend_builtins();
        c.sort();
        c
    }

    /// Alias for [`Catalog::with_builtins`] — the standard, no-live-state catalog.
    pub fn standard() -> Catalog {
        Catalog::with_builtins()
    }

    /// Add a single entry (de-duplicated on name+kind), keeping the catalog sorted.
    pub fn add(&mut self, name: &str, kind: ItemKind, summary: &str) {
        self.push(CatalogItem::new(name, kind, summary));
        self.sort();
    }

    /// Register an IDE program (an open Dominion program / view) as discoverable.
    pub fn add_program(&mut self, name: &str, summary: &str) {
        self.add(name, ItemKind::Program, summary);
    }

    /// Fold in the standard library packages as [`ItemKind::Library`] entries.
    pub fn extend_libraries(&mut self) {
        for (name, summary) in STD_LIBRARIES {
            self.push(CatalogItem::new(name, ItemKind::Library, summary));
        }
        // Each polyglot guest language is itself a discoverable library surface.
        for lang in Language::all() {
            self.push(CatalogItem::new(
                lang.name(),
                ItemKind::Library,
                "Polyglot guest language (sandboxed, capability-bounded runtime)",
            ));
        }
        self.sort();
    }

    /// Fold in every synthesizable device class as [`ItemKind::Driver`] entries,
    /// derived from `drivergen`'s real [`HwClass`] enum.
    pub fn extend_drivers(&mut self) {
        for &class in HW_CLASSES {
            self.push(CatalogItem::new(hwclass_name(class), ItemKind::Driver, hwclass_summary(class)));
        }
        self.sort();
    }

    /// Fold in the Dominion-language builtins as [`ItemKind::Builtin`] entries.
    pub fn extend_builtins(&mut self) {
        for (name, summary) in BUILTINS {
            self.push(CatalogItem::new(name, ItemKind::Builtin, summary));
        }
        self.sort();
    }

    /// Register an installed [`Package`] as an [`ItemKind::Package`] entry
    /// (name + version summary). This is the per-package path the IDE drives with
    /// live registry state.
    pub fn add_package(&mut self, pkg: &Package) {
        let summary = alloc::format!("Installed package, version {}", pkg.version);
        self.push(CatalogItem::new(&pkg.name, ItemKind::Package, &summary));
        self.sort();
    }

    /// Fold every package known to a live [`PackageRegistry`] into the catalog as
    /// [`ItemKind::Package`] entries. `PackageRegistry` is content-addressed and
    /// does not expose iteration, so the IDE passes the packages it installed via
    /// `packages` (the registry is taken to confirm each is genuinely installed);
    /// any package not present in the registry is skipped.
    pub fn extend_packages(&mut self, registry: &PackageRegistry, packages: &[Package]) {
        for pkg in packages {
            if registry.get(&pkg.content_id).is_some() {
                self.add_package(pkg);
            }
        }
    }

    /// Build a catalog from a live [`NodeGraph`]: every node becomes a discoverable
    /// sub-node entry (merged on top of the static built-in set).
    pub fn from_node_graph(graph: &NodeGraph) -> Catalog {
        let mut c = Catalog::with_builtins();
        c.add_node_graph(graph);
        c
    }

    /// Fold a node graph's nodes into this catalog as [`ItemKind::SubNode`] entries.
    pub fn add_node_graph(&mut self, graph: &NodeGraph) {
        for node in graph.nodes() {
            let summary = if node.subtitle.is_empty() {
                nodekind_summary(node.kind).to_string()
            } else {
                node.subtitle.clone()
            };
            self.push(CatalogItem::new(&node.title, ItemKind::SubNode, &summary));
        }
        self.sort();
    }

    /// Catalog entries whose name starts with `prefix` (case-insensitive).
    pub fn search(&self, prefix: &str) -> Vec<CatalogItem> {
        let p = prefix.to_ascii_lowercase();
        self.items.iter().filter(|it| it.name.to_ascii_lowercase().starts_with(&p)).cloned().collect()
    }

    /// All entries of a given kind, in catalog order.
    pub fn by_kind(&self, kind: ItemKind) -> Vec<&CatalogItem> {
        self.items.iter().filter(|it| it.kind == kind).collect()
    }

    /// Every entry, in deterministic (kind, name) order.
    pub fn items(&self) -> &[CatalogItem] {
        &self.items
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    // ── internals ──

    /// Push an entry, replacing any prior entry with the same name+kind (so live
    /// updates overwrite rather than duplicate). Does not re-sort.
    fn push(&mut self, item: CatalogItem) {
        if let Some(existing) = self.items.iter_mut().find(|e| e.name == item.name && e.kind == item.kind) {
            *existing = item;
        } else {
            self.items.push(item);
        }
    }

    /// Sort by kind rank, then name — the catalog's deterministic order.
    fn sort(&mut self) {
        self.items.sort_by(|a, b| a.kind.rank().cmp(&b.kind.rank()).then_with(|| a.name.cmp(&b.name)));
    }
}

// ───────────────────────── free convenience entry (LSP) ─────────────────────────

/// Every discoverable item across the always-present static sources — standard
/// libraries, device-class drivers, and language builtins — in deterministic order.
pub fn catalog() -> Vec<CatalogItem> {
    Catalog::with_builtins().items
}

/// Catalog entries whose name starts with `prefix` (case-insensitive).
pub fn search(prefix: &str) -> Vec<CatalogItem> {
    let p = prefix.to_ascii_lowercase();
    catalog().into_iter().filter(|it| it.name.to_ascii_lowercase().starts_with(&p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{NodeGraph, NodeKind};

    #[test]
    fn static_catalog_is_non_empty_and_sorted() {
        let items = catalog();
        assert!(!items.is_empty());
        // Deterministic (kind, name) ordering.
        for w in items.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let ord = a.kind.rank().cmp(&b.kind.rank()).then_with(|| a.name.cmp(&b.name));
            assert!(ord != core::cmp::Ordering::Greater, "catalog not sorted at {:?} / {:?}", a.name, b.name);
        }
    }

    #[test]
    fn static_catalog_ships_every_static_kind() {
        let items = catalog();
        for kind in [ItemKind::Driver, ItemKind::Library, ItemKind::Builtin] {
            assert!(items.iter().any(|it| it.kind == kind), "missing kind {:?}", kind);
        }
    }

    #[test]
    fn search_finds_the_tensor_builtin() {
        let hits = search("ten");
        assert!(hits.iter().any(|it| it.name == "tensor" && it.kind == ItemKind::Builtin));
    }

    #[test]
    fn search_is_case_insensitive() {
        let lower = search("ten");
        let upper = search("TEN");
        assert_eq!(lower, upper);
        assert!(!lower.is_empty());
    }

    #[test]
    fn drivers_are_derived_from_the_hwclass_enum() {
        let cat = Catalog::with_builtins();
        let drivers = cat.by_kind(ItemKind::Driver);
        // One driver entry per real HwClass variant.
        assert_eq!(drivers.len(), HW_CLASSES.len());
        assert!(drivers.iter().any(|d| d.name == "nvme"));
        assert!(drivers.iter().any(|d| d.name == "virtio-net"));
    }

    #[test]
    fn libraries_include_every_guest_language() {
        let cat = Catalog::with_builtins();
        let libs = cat.by_kind(ItemKind::Library);
        for lang in Language::all() {
            assert!(libs.iter().any(|l| l.name == lang.name()), "missing language {}", lang.name());
        }
        // ...plus the standard stdlib packages.
        assert!(libs.iter().any(|l| l.name == "mathx"));
    }

    #[test]
    fn instance_merges_live_programs_and_subnodes_with_static_entries() {
        let mut g = NodeGraph::new();
        g.add(1, "Project Alpha", "Data Object", NodeKind::Data, 0, 0);
        g.add(2, "Q3 Report", "", NodeKind::Report, 0, 0);

        let mut cat = Catalog::from_node_graph(&g);
        cat.add_program("editor.aeth", "The open editor program");

        // Static entries survive the merge.
        assert!(!cat.by_kind(ItemKind::Builtin).is_empty());
        assert!(!cat.by_kind(ItemKind::Driver).is_empty());
        // Live sub-nodes are present (subtitle used when set; class summary otherwise).
        let subnodes = cat.by_kind(ItemKind::SubNode);
        assert_eq!(subnodes.len(), 2);
        assert!(subnodes.iter().any(|n| n.name == "Project Alpha" && n.summary == "Data Object"));
        assert!(subnodes.iter().any(|n| n.name == "Q3 Report"));
        // Live program is present.
        assert!(cat.by_kind(ItemKind::Program).iter().any(|p| p.name == "editor.aeth"));
        // And search still spans both static + live.
        assert!(cat.search("proj").iter().any(|it| it.name == "Project Alpha"));
        assert!(cat.search("ten").iter().any(|it| it.name == "tensor"));
    }

    #[test]
    fn instance_ordering_is_deterministic() {
        let mut g = NodeGraph::new();
        g.add(1, "Zeta", "", NodeKind::Data, 0, 0);
        g.add(2, "Alpha", "", NodeKind::App, 0, 0);
        let a = Catalog::from_node_graph(&g);
        let b = Catalog::from_node_graph(&g);
        assert_eq!(a.items(), b.items());
        // SubNodes sort by name within their kind.
        let subs: Vec<&str> = a.by_kind(ItemKind::SubNode).iter().map(|n| n.name.as_str()).collect();
        assert_eq!(subs, ["Alpha", "Zeta"]);
    }

    #[test]
    fn live_packages_merge_via_registry() {
        use crate::capability::{Capability, Rights};
        use crate::packaging::{Package, PackageRegistry};
        let mut reg = PackageRegistry::new();
        let pkg = Package::seal("editor", "2.1", b"editor-bytes", Rights::READ, b"publisher-key");
        let grant = Capability::mint(0, 4096, Rights::READ.union(Rights::WRITE));
        // Trust the publisher key before installing via the strict direct API (the
        // depot install path trusts its catalog automatically; this one does not).
        reg.trust_key(pkg.public_key().to_vec());
        reg.install(pkg.clone(), &grant).unwrap();

        let mut cat = Catalog::with_builtins();
        cat.extend_packages(&reg, &[pkg]);
        let pkgs = cat.by_kind(ItemKind::Package);
        assert_eq!(pkgs.len(), 1);
        assert_eq!(pkgs[0].name, "editor");
        assert!(pkgs[0].summary.contains("2.1"));
        // Static entries still present after a live package merge.
        assert!(!cat.by_kind(ItemKind::Builtin).is_empty());
    }

    #[test]
    fn add_replaces_rather_than_duplicates() {
        let mut cat = Catalog::new();
        cat.add_program("p", "first");
        cat.add_program("p", "second");
        let progs = cat.by_kind(ItemKind::Program);
        assert_eq!(progs.len(), 1);
        assert_eq!(progs[0].summary, "second");
    }
}

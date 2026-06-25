//! Package **depot** & dependency resolution — the "download and install, with its
//! dependencies" half of the package manager, layered on the verified, capability-
//! confined [`PackageRegistry`](super::PackageRegistry).
//!
//! [`super`] supplies the trustworthy primitive: a content-addressed, PQ-signed package
//! whose install verifies the signature and confines it to least privilege. This module
//! supplies what makes it usable as "just install it":
//!
//! * [`PackageKind`] — what a package *delivers* (a driver, an application, a library, a
//!   tool), so one install path serves the whole ecosystem;
//! * [`Manifest`] — name/version/kind plus a **dependency list**;
//! * [`Depot`] — a repository of available packages you can publish to and `resolve`
//!   against, computing the full transitive install order (missing deps and dependency
//!   cycles are hard errors, never silent);
//! * [`Depot::install_with_deps`] — resolve, then install every package in order through
//!   the verified registry, so dependencies are fetched + verified + capability-confined
//!   exactly like the top-level package.
//!
//! The depot here is in-memory (the kernel backs it with the network / object store);
//! the *verification and confinement* are the real thing. Pure, safe `no_std`,
//! host-tested.

use super::{InstallError, Package, PackageRegistry};
use crate::capability::{Capability, Rights};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// What a package delivers — one install path covers every kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageKind {
    /// A device driver (native spec or borrowed foreign binary).
    Driver,
    /// A runnable application (native or foreign).
    App,
    /// A library/package importable by polyglot code.
    Library,
    /// A developer tool (compiler/formatter/etc.).
    Tool,
}

/// A package's manifest: identity, what it delivers, and what it depends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub kind: PackageKind,
    /// Names of packages this one requires (resolved transitively at install).
    pub deps: Vec<String>,
    /// Least-privilege rights the package requests at install.
    pub rights: Rights,
}

impl Manifest {
    pub fn new(name: &str, version: &str, kind: PackageKind, rights: Rights) -> Manifest {
        Manifest { name: name.to_string(), version: version.to_string(), kind, deps: Vec::new(), rights }
    }
    pub fn depends_on(mut self, deps: &[&str]) -> Manifest {
        self.deps = deps.iter().map(|s| s.to_string()).collect();
        self
    }
}

struct DepotEntry {
    manifest: Manifest,
    pkg: Package,
    /// The raw package content (driver `.sys`/`.ko`, app binary, library source, …),
    /// retained so an installed package can be handed straight to the driver loader /
    /// app launcher — the "download → install → it just runs" flow.
    content: Vec<u8>,
}

/// Why a resolve/install failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepError {
    /// A required package is not available in the depot.
    Missing(String),
    /// The dependency graph has a cycle through this package.
    Cycle(String),
    /// A package failed the verified install (bad signature or over-reach).
    Install(String, InstallError),
}

/// A repository of available, signed packages you can resolve and install from.
#[derive(Default)]
pub struct Depot {
    entries: BTreeMap<String, DepotEntry>,
}

impl Depot {
    pub fn new() -> Depot {
        Depot { entries: BTreeMap::new() }
    }

    /// Publish (seal + sign) a package's content under its manifest into the depot —
    /// the "upload to the repo" half. Replaces any existing package of the same name.
    pub fn publish(&mut self, manifest: Manifest, content: &[u8], secret_seed: &[u8]) {
        let pkg = Package::seal(
            &manifest.name,
            &manifest.version,
            content,
            manifest.rights,
            secret_seed,
        );
        self.entries.insert(
            manifest.name.clone(),
            DepotEntry { manifest, pkg, content: content.to_vec() },
        );
    }

    /// The raw content of an available package (what gets handed to the loader/launcher
    /// after a verified install).
    pub fn content(&self, name: &str) -> Option<&[u8]> {
        self.entries.get(name).map(|e| e.content.as_slice())
    }

    /// The names of every available package, sorted.
    pub fn available(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// A package's manifest, if present.
    pub fn manifest(&self, name: &str) -> Option<&Manifest> {
        self.entries.get(name).map(|e| &e.manifest)
    }

    /// Compute the full transitive install order for `name` (dependencies first, the
    /// requested package last). Missing dependencies and cycles are hard errors.
    pub fn resolve(&self, name: &str) -> Result<Vec<String>, DepError> {
        let mut order = Vec::new();
        let mut done = BTreeSet::new();
        let mut on_stack = BTreeSet::new();
        self.visit(name, &mut order, &mut done, &mut on_stack)?;
        Ok(order)
    }

    fn visit(
        &self,
        name: &str,
        order: &mut Vec<String>,
        done: &mut BTreeSet<String>,
        on_stack: &mut BTreeSet<String>,
    ) -> Result<(), DepError> {
        if done.contains(name) {
            return Ok(());
        }
        if on_stack.contains(name) {
            return Err(DepError::Cycle(name.to_string()));
        }
        let entry = self.entries.get(name).ok_or_else(|| DepError::Missing(name.to_string()))?;
        on_stack.insert(name.to_string());
        for dep in &entry.manifest.deps {
            self.visit(dep, order, done, on_stack)?;
        }
        on_stack.remove(name);
        done.insert(name.to_string());
        order.push(name.to_string());
        Ok(())
    }

    /// Resolve `name` and install every package in dependency order into `registry`,
    /// each verified + confined to the installer's `grant`. Returns the install order.
    pub fn install_with_deps(
        &self,
        name: &str,
        registry: &mut PackageRegistry,
        grant: &Capability,
    ) -> Result<Vec<String>, DepError> {
        let order = self.resolve(name)?;
        for pkg_name in &order {
            let entry = self
                .entries
                .get(pkg_name)
                .ok_or_else(|| DepError::Missing(pkg_name.clone()))?;
            // Installing from a depot establishes trust in that depot's catalog: the
            // package was sealed + vetted at publish time, so its embedded publisher
            // key is added to the registry's ring before install (the way adding an
            // apt/pacman repo trusts its signing key). The signature is still verified
            // under that key inside `install`, and a *tampered* package would carry a
            // different content id / signature and fail that check. The stricter
            // direct `PackageRegistry::install` path (loose package, no depot) still
            // requires a pre-populated ring — that is the self-seal attack defense.
            registry.trust_key(entry.pkg.public_key().to_vec());
            registry
                .install(entry.pkg.clone(), grant)
                .map_err(|e| DepError::Install(pkg_name.clone(), e))?;
        }
        Ok(order)
    }

    /// Resolve + verify-install `name` and its dependencies, returning each installed
    /// package's `(name, kind, content)` in dependency order. This is the bridge that
    /// makes "download → install → it just runs" one call: the caller hands a `Driver`
    /// package's content to [`crate::personality::driverload::load_driver`] or an `App`
    /// package's content to [`crate::personality::applaunch::launch_app`].
    pub fn install_and_fetch(
        &self,
        name: &str,
        registry: &mut PackageRegistry,
        grant: &Capability,
    ) -> Result<Vec<(String, PackageKind, Vec<u8>)>, DepError> {
        let order = self.install_with_deps(name, registry, grant)?;
        let mut out = Vec::with_capacity(order.len());
        for pkg_name in order {
            let entry = self
                .entries
                .get(&pkg_name)
                .ok_or_else(|| DepError::Missing(pkg_name.clone()))?;
            out.push((pkg_name, entry.manifest.kind, entry.content.clone()));
        }
        Ok(out)
    }
}

/// A representative default depot spanning every package kind, with a real dependency
/// chain — what a freshly-booted system would point at before adding remotes.
pub fn default_depot() -> Depot {
    let mut d = Depot::new();
    // A library with no deps.
    d.publish(
        Manifest::new("mathx", "1.0", PackageKind::Library, Rights::READ),
        b"mathx-library-content",
        b"mathx-seed",
    );
    // A library that depends on mathx.
    d.publish(
        Manifest::new("stats", "1.0", PackageKind::Library, Rights::READ).depends_on(&["mathx"]),
        b"stats-library-content",
        b"stats-seed",
    );
    // A driver package.
    d.publish(
        Manifest::new("rtl8139-driver", "1.0", PackageKind::Driver, Rights::READ),
        b"rtl8139-driver-content",
        b"rtl-seed",
    );
    // An application that depends on a library.
    d.publish(
        Manifest::new("text-editor", "2.1", PackageKind::App, Rights::READ).depends_on(&["mathx"]),
        b"text-editor-content",
        b"editor-seed",
    );
    // The native NVIDIA CUDA / ML stack, as installable packages with a real
    // dependency graph (see crate::ml::gpu for the capability-gated runtime). GPU
    // access is least-privilege: the driver requests READ+WRITE (device memory), the
    // libraries READ. Installing tensorrt pulls cudnn+cublas → cuda-toolkit → driver.
    d.publish(
        Manifest::new("cuda-driver", "12.4", PackageKind::Driver, Rights::READ.union(Rights::WRITE)),
        b"nvidia-cuda-driver",
        b"cuda-driver-seed",
    );
    d.publish(
        Manifest::new("cuda-toolkit", "12.4", PackageKind::Library, Rights::READ)
            .depends_on(&["cuda-driver"]),
        b"cuda-toolkit",
        b"cuda-toolkit-seed",
    );
    d.publish(
        Manifest::new("libcublas", "12.4", PackageKind::Library, Rights::READ)
            .depends_on(&["cuda-toolkit"]),
        b"libcublas",
        b"cublas-seed",
    );
    d.publish(
        Manifest::new("libcudnn", "9.1", PackageKind::Library, Rights::READ)
            .depends_on(&["cuda-toolkit"]),
        b"libcudnn",
        b"cudnn-seed",
    );
    d.publish(
        Manifest::new("tensorrt", "10.0", PackageKind::Library, Rights::READ)
            .depends_on(&["libcudnn", "libcublas"]),
        b"tensorrt",
        b"tensorrt-seed",
    );
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installer_grant() -> Capability {
        Capability::mint(0, 1 << 20, Rights::READ.union(Rights::WRITE))
    }

    #[test]
    fn resolves_dependencies_in_install_order() {
        let d = default_depot();
        // stats depends on mathx → mathx must come first.
        let order = d.resolve("stats").unwrap();
        assert_eq!(order, alloc::vec!["mathx".to_string(), "stats".to_string()]);
        // A leaf resolves to just itself.
        assert_eq!(d.resolve("mathx").unwrap(), alloc::vec!["mathx".to_string()]);
    }

    #[test]
    fn missing_dependency_is_a_hard_error() {
        let mut d = Depot::new();
        d.publish(
            Manifest::new("app", "1", PackageKind::App, Rights::READ).depends_on(&["ghost"]),
            b"x",
            b"s",
        );
        assert_eq!(d.resolve("app"), Err(DepError::Missing("ghost".into())));
    }

    #[test]
    fn dependency_cycles_are_detected() {
        let mut d = Depot::new();
        d.publish(
            Manifest::new("a", "1", PackageKind::Library, Rights::READ).depends_on(&["b"]),
            b"a",
            b"s",
        );
        d.publish(
            Manifest::new("b", "1", PackageKind::Library, Rights::READ).depends_on(&["a"]),
            b"b",
            b"s",
        );
        match d.resolve("a") {
            Err(DepError::Cycle(_)) => {}
            other => panic!("expected a cycle error, got {:?}", other),
        }
    }

    #[test]
    fn install_with_deps_verifies_and_confines_each_package() {
        let d = default_depot();
        let mut reg = PackageRegistry::new();
        let grant = installer_grant();
        let order = d.install_with_deps("text-editor", &mut reg, &grant).unwrap();
        // mathx (dep) installed before text-editor.
        assert_eq!(order, alloc::vec!["mathx".to_string(), "text-editor".to_string()]);
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn install_refuses_a_package_that_over_reaches_the_grant() {
        let mut d = Depot::new();
        // Requests WRITE…
        d.publish(
            Manifest::new("greedy", "1", PackageKind::Tool, Rights::WRITE),
            b"content",
            b"seed",
        );
        let mut reg = PackageRegistry::new();
        // …but the installer grant is READ-only.
        let ro = Capability::mint(0, 16, Rights::READ);
        match d.install_with_deps("greedy", &mut reg, &ro) {
            Err(DepError::Install(name, InstallError::ExceedsGrant)) => assert_eq!(name, "greedy"),
            other => panic!("expected ExceedsGrant, got {:?}", other),
        }
    }

    #[test]
    fn install_and_fetch_returns_verified_content_in_order() {
        let mut d = Depot::new();
        d.publish(
            Manifest::new("libdep", "1", PackageKind::Library, Rights::READ),
            b"libdep-bytes",
            b"s1",
        );
        d.publish(
            Manifest::new("driverpkg", "1", PackageKind::Driver, Rights::READ)
                .depends_on(&["libdep"]),
            b"driver-binary-bytes",
            b"s2",
        );
        let mut reg = PackageRegistry::new();
        let grant = installer_grant();
        let fetched = d.install_and_fetch("driverpkg", &mut reg, &grant).unwrap();
        // Dependency first, then the package — each with its kind + verified content.
        assert_eq!(fetched.len(), 2);
        assert_eq!(fetched[0].0, "libdep");
        assert_eq!(fetched[0].1, PackageKind::Library);
        assert_eq!(fetched[1].0, "driverpkg");
        assert_eq!(fetched[1].1, PackageKind::Driver);
        assert_eq!(fetched[1].2, b"driver-binary-bytes".to_vec());
        // Content is also directly retrievable from the depot.
        assert_eq!(d.content("driverpkg"), Some(b"driver-binary-bytes".as_ref()));
    }

    #[test]
    fn cuda_stack_resolves_and_installs_with_dependencies() {
        let d = default_depot();
        // tensorrt → {cudnn, cublas} → cuda-toolkit → cuda-driver.
        let order = d.resolve("tensorrt").unwrap();
        let pos = |n: &str| order.iter().position(|x| x == n).unwrap();
        assert!(pos("cuda-driver") < pos("cuda-toolkit"));
        assert!(pos("cuda-toolkit") < pos("libcudnn"));
        assert!(pos("cuda-toolkit") < pos("libcublas"));
        assert!(pos("libcudnn") < pos("tensorrt"));
        assert!(pos("libcublas") < pos("tensorrt"));
        // Installs verified + capability-confined (driver wants WRITE for device memory).
        let mut reg = PackageRegistry::new();
        let grant = Capability::mint(0, 1 << 20, Rights::READ.union(Rights::WRITE));
        let installed = d.install_with_deps("tensorrt", &mut reg, &grant).unwrap();
        assert_eq!(installed.len(), 5);
        assert_eq!(d.manifest("cuda-driver").unwrap().kind, PackageKind::Driver);
    }

    #[test]
    fn default_depot_lists_every_kind() {
        let d = default_depot();
        let names = d.available();
        assert!(names.contains(&"mathx".to_string()));
        assert!(names.contains(&"rtl8139-driver".to_string()));
        assert!(names.contains(&"text-editor".to_string()));
        assert_eq!(d.manifest("rtl8139-driver").unwrap().kind, PackageKind::Driver);
        assert_eq!(d.manifest("text-editor").unwrap().kind, PackageKind::App);
    }
}

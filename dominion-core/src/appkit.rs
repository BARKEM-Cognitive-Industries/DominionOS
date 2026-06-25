//! The reactive application framework — Dominion for apps & GUIs (see
//! `docs/language/dominion-ui-and-applications.md`).
//!
//! A UI is a **pure function from state to a view tree** ([`crate::toolkit::Widget`]),
//! re-evaluated when the state it *read* changes. This module is the runtime that
//! makes that real, plus the rest of what real applications need:
//!
//! * **Reactive [`Store`]** — state + **read-tracking**: a view records which keys it
//!   read, and the app re-renders only when one of *those* keys changes.
//! * **Events** — handlers mutate the store; the view re-runs (no manual `setState`).
//! * **App capabilities** — an explicit grant set (`Surface`/`Storage`/`Net`/…); an
//!   app can only do what it was granted, surfaced in the capability panel.
//! * **Async** — a cooperative [`Executor`] that records completion order so async
//!   still replays deterministically (Stage 10).
//! * **Modules/packages** — a content-addressed registry: "install" = resolve a hash.
//!
//! Reactive bindings can also be driven by the [`crate::pubsub`] subscription plane
//! for live remote data; this module is the local, self-contained core. Pure, safe
//! `no_std`.

use crate::hash::Hash256;
use crate::toolkit::{self, DrawCmd, Rect, Widget};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

// ───────────────────────────── reactive store ─────────────────────────────

/// Reactive key→integer state with read-tracking and a change set. (Integer state
/// keeps the core simple and testable; richer values layer on top.)
#[derive(Default)]
pub struct Store {
    values: BTreeMap<String, i64>,
    version: u64,
    dirty: BTreeSet<String>,
}

impl Store {
    pub fn new() -> Store {
        Store { values: BTreeMap::new(), version: 0, dirty: BTreeSet::new() }
    }

    /// Read a key, recording the access in `reads` so the view's dependencies are
    /// tracked. Missing keys read as 0.
    pub fn read(&self, key: &str, reads: &mut BTreeSet<String>) -> i64 {
        reads.insert(key.into());
        *self.values.get(key).unwrap_or(&0)
    }

    /// Read without tracking (for handlers / non-reactive code).
    pub fn get(&self, key: &str) -> i64 {
        *self.values.get(key).unwrap_or(&0)
    }

    /// Set a key; marks it dirty and bumps the version.
    pub fn set(&mut self, key: &str, value: i64) {
        self.values.insert(key.into(), value);
        self.dirty.insert(key.into());
        self.version += 1;
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    fn clear_dirty(&mut self) {
        self.dirty.clear();
    }
}

/// A view is a pure function from store → widget tree, tracking the keys it reads.
pub type ViewFn = fn(&Store, &mut BTreeSet<String>) -> Widget;
/// An event handler mutates the store.
pub type Handler = fn(&mut Store);

/// A reactive application: a store, a view, and event handlers.
pub struct App {
    pub store: Store,
    view: ViewFn,
    handlers: BTreeMap<String, Handler>,
    last_reads: BTreeSet<String>,
    rendered_once: bool,
}

impl App {
    pub fn new(store: Store, view: ViewFn) -> App {
        App { store, view, handlers: BTreeMap::new(), last_reads: BTreeSet::new(), rendered_once: false }
    }

    /// Register an event handler.
    pub fn on(&mut self, event: &str, handler: Handler) {
        self.handlers.insert(event.into(), handler);
    }

    /// Dispatch an event: run its handler (mutating the store). Returns false if no
    /// handler is registered.
    pub fn dispatch(&mut self, event: &str) -> bool {
        if let Some(h) = self.handlers.get(event) {
            h(&mut self.store);
            true
        } else {
            false
        }
    }

    /// Does the view need re-rendering? True on first render, or when any key it read
    /// last time has since changed.
    pub fn needs_rerender(&self) -> bool {
        if !self.rendered_once {
            return true;
        }
        self.last_reads.iter().any(|k| self.store.dirty.contains(k))
    }

    /// Render: build the view, capture its read-set, and clear the change set so the
    /// next `needs_rerender` reflects only changes *after* this render.
    pub fn render(&mut self, theme: &toolkit::Theme, area: Rect) -> Vec<DrawCmd> {
        let mut reads = BTreeSet::new();
        let widget = (self.view)(&self.store, &mut reads);
        self.last_reads = reads;
        self.rendered_once = true;
        let scene = toolkit::build_scene(&widget, theme, area);
        self.store.clear_dirty();
        scene
    }

    /// The keys the last render depended on.
    pub fn dependencies(&self) -> &BTreeSet<String> {
        &self.last_reads
    }
}

// ─────────────────────────── app capabilities ───────────────────────────

/// A capability an application may be granted. An app can do **only** what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AppCap {
    Surface,
    Storage,
    Net,
    Clipboard,
    Device,
    Time,
    Entropy,
}

/// The set of capabilities granted to an app — explicit, revocable, and shown in the
/// capability panel.
#[derive(Default)]
pub struct AppCapabilities {
    granted: BTreeSet<AppCap>,
}

impl AppCapabilities {
    pub fn new() -> AppCapabilities {
        AppCapabilities { granted: BTreeSet::new() }
    }
    pub fn grant(&mut self, cap: AppCap) {
        self.granted.insert(cap);
    }
    pub fn revoke(&mut self, cap: AppCap) {
        self.granted.remove(&cap);
    }
    pub fn holds(&self, cap: AppCap) -> bool {
        self.granted.contains(&cap)
    }
    /// Gate an operation needing `cap`; `Err` if not granted (default-closed).
    pub fn require(&self, cap: AppCap) -> Result<(), AppCap> {
        if self.holds(cap) {
            Ok(())
        } else {
            Err(cap)
        }
    }
}

// ─────────────────────────── cooperative async ───────────────────────────

/// The state of a polled task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poll {
    Ready,
    Pending,
}

type Task = Box<dyn FnMut() -> Poll>;

/// A cooperative async executor. Tasks are polled in **spawn order** each round, and
/// completion order is recorded so an async session replays deterministically.
#[derive(Default)]
pub struct Executor {
    tasks: Vec<(u64, Task)>,
    completed: Vec<u64>,
}

impl Executor {
    pub fn new() -> Executor {
        Executor { tasks: Vec::new(), completed: Vec::new() }
    }

    /// Spawn a task with a stable id.
    pub fn spawn(&mut self, id: u64, task: Task) {
        self.tasks.push((id, task));
    }

    /// Poll every live task once; completed tasks are removed and recorded in order.
    pub fn tick(&mut self) {
        let mut still = Vec::new();
        for (id, mut task) in self.tasks.drain(..) {
            match task() {
                Poll::Ready => self.completed.push(id),
                Poll::Pending => still.push((id, task)),
            }
        }
        self.tasks = still;
    }

    /// Run until all tasks complete (bounded to avoid a runaway loop).
    pub fn run(&mut self) {
        let mut guard = 0;
        while !self.tasks.is_empty() && guard < 10_000 {
            self.tick();
            guard += 1;
        }
    }

    pub fn pending(&self) -> usize {
        self.tasks.len()
    }
    /// The order in which tasks completed (deterministic for a given program).
    pub fn completed_order(&self) -> &[u64] {
        &self.completed
    }
}

// ─────────────────────────── modules / packages ───────────────────────────

/// A content-addressed package registry. A "package" is bytes addressed by hash;
/// "installing" is resolving a hash to its bytes (no global mutable package state).
#[derive(Default)]
pub struct Modules {
    store: BTreeMap<Hash256, Vec<u8>>,
}

impl Modules {
    pub fn new() -> Modules {
        Modules { store: BTreeMap::new() }
    }

    /// Publish a package; returns its content address.
    pub fn publish(&mut self, bytes: &[u8]) -> Hash256 {
        let id = Hash256::of(bytes);
        self.store.entry(id).or_insert_with(|| bytes.to_vec());
        id
    }

    /// Resolve ("install") a package by content address — verified against the hash,
    /// so a corrupted entry is rejected, not served.
    pub fn resolve(&self, id: Hash256) -> Option<&[u8]> {
        let bytes = self.store.get(&id)?;
        if Hash256::of(bytes) == id {
            Some(bytes)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny counter app: the view reads "count" and renders it; the "increment"
    // event bumps it. This is the React/Elm model, working.
    fn counter_view(store: &Store, reads: &mut BTreeSet<String>) -> Widget {
        let count = store.read("count", reads);
        let mut label = String::from("Count: ");
        label.push_str(&alloc::format!("{count}"));
        toolkit::column(
            0,
            alloc::vec![toolkit::label(1, &label), toolkit::button(2, "Increment")],
        )
    }
    fn increment(store: &mut Store) {
        let n = store.get("count");
        store.set("count", n + 1);
    }

    #[test]
    fn store_tracks_reads_and_changes() {
        let mut store = Store::new();
        store.set("a", 1);
        let mut reads = BTreeSet::new();
        assert_eq!(store.read("a", &mut reads), 1);
        assert!(reads.contains("a"));
        assert_eq!(store.read("missing", &mut reads), 0);
    }

    #[test]
    fn reactive_rerender_only_on_read_keys() {
        let mut app = App::new(Store::new(), counter_view);
        // First render is always needed; it reads "count".
        assert!(app.needs_rerender());
        app.render(&toolkit::Theme::dark(), Rect::new(0, 0, 200, 80));
        assert!(app.dependencies().contains("count"));
        assert!(!app.needs_rerender()); // nothing changed since render
        // Changing an UNread key does not trigger a re-render.
        app.store.set("unrelated", 5);
        assert!(!app.needs_rerender());
        // Changing the read key DOES.
        app.store.set("count", 1);
        assert!(app.needs_rerender());
    }

    #[test]
    fn events_mutate_state_and_drive_rerender() {
        let mut app = App::new(Store::new(), counter_view);
        app.on("increment", increment);
        app.render(&toolkit::Theme::dark(), Rect::new(0, 0, 200, 80));
        assert_eq!(app.store.get("count"), 0);
        assert!(app.dispatch("increment"));
        assert_eq!(app.store.get("count"), 1);
        assert!(app.needs_rerender());
        // An unknown event is a no-op.
        assert!(!app.dispatch("nope"));
        // Re-render shows the new value.
        let scene = app.render(&toolkit::Theme::dark(), Rect::new(0, 0, 200, 80));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Count: 1")));
    }

    #[test]
    fn app_capabilities_are_default_closed() {
        let mut caps = AppCapabilities::new();
        assert!(caps.require(AppCap::Net).is_err());
        caps.grant(AppCap::Net);
        assert!(caps.require(AppCap::Net).is_ok());
        // A different capability is still denied.
        assert_eq!(caps.require(AppCap::Device), Err(AppCap::Device));
        caps.revoke(AppCap::Net);
        assert!(!caps.holds(AppCap::Net));
    }

    #[test]
    fn async_executor_runs_to_completion_in_deterministic_order() {
        use core::cell::Cell;
        use alloc::rc::Rc;
        let mut ex = Executor::new();
        // Task 1 completes immediately.
        ex.spawn(1, Box::new(|| Poll::Ready));
        // Task 2 is pending once, then ready.
        let n = Rc::new(Cell::new(0));
        let n2 = n.clone();
        ex.spawn(
            2,
            Box::new(move || {
                if n2.get() == 0 {
                    n2.set(1);
                    Poll::Pending
                } else {
                    Poll::Ready
                }
            }),
        );
        ex.run();
        assert_eq!(ex.pending(), 0);
        // Deterministic completion order: 1 finishes in round 1, 2 in round 2.
        assert_eq!(ex.completed_order(), &[1, 2]);
    }

    #[test]
    fn module_registry_resolves_by_content_address() {
        let mut mods = Modules::new();
        let id = mods.publish(b"package: hello-world v1");
        assert_eq!(mods.resolve(id), Some(b"package: hello-world v1".as_ref()));
        // An unknown hash resolves to nothing.
        assert!(mods.resolve(Hash256::of(b"not published")).is_none());
        // Re-publishing the same bytes dedups.
        let id2 = mods.publish(b"package: hello-world v1");
        assert_eq!(id, id2);
        assert_eq!(mods.len(), 1);
    }
}

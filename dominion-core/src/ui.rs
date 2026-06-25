//! Object-centric, AI-native UI — **Stage 9** + the UX/GUI vision (see
//! `docs/architecture/10-stage-09-object-centric-ai-ui.md` and
//! `docs/architecture/user-experience-and-gui.md`).
//!
//! The UI is not application-centric. There is no app menu and no install step: the
//! user works with **objects** on a spatial canvas, each shown through one of several
//! **views on demand** (Table / Graph / Spatial / Assistive), and "programs" are
//! transient **capability compositions** assembled on the fly. Three further pieces:
//!
//! * **AI command bar** — natural language is parsed to a **capability-gated**
//!   action; without the authority, the action does not run ("no capability ⇒ the
//!   resource does not exist").
//! * **Abstract input model** — pointer / key / touch / gesture all normalise to the
//!   same `UiAction`, so one compositor drives desktop *and* mobile.
//! * **Universal undo / time-travel** — every action commits a content-addressed
//!   root, and undo rewinds deterministically (the Stage 10 rollback, surfaced).
//!
//! Pure, safe `no_std`, host-tested.

use crate::capability::Rights;
use crate::hash::Hash256;
use crate::object::{Object, ObjectId};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ─────────────────────────── views on demand ───────────────────────────

/// A way to look at one object. The same object can be shown through any of these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Table,
    Graph,
    Spatial,
    /// The accessibility view — a first-class view type, not a bolt-on.
    Assistive,
}

/// Render an object through a view as a deterministic textual model (a real GUI
/// backend would draw this; the *content* is the same knowledge-graph object).
pub fn render(obj: &Object, view: View) -> String {
    let fields = &obj.fields;
    match view {
        View::Table => {
            let mut s = alloc::format!("[{}]", obj.kind);
            for (k, v) in fields {
                s.push_str(&alloc::format!(" {}={:?}", k, v));
            }
            s
        }
        View::Graph => {
            let mut s = alloc::format!("{}(", obj.kind);
            for (i, (k, _)) in fields.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(k);
            }
            s.push(')');
            s
        }
        View::Spatial => alloc::format!("⬢ {} [{} fields]", obj.kind, fields.len()),
        View::Assistive => {
            // A screen-reader-friendly description straight from the object's typed
            // fields — the knowledge graph *is* the accessibility tree.
            let mut s = alloc::format!("{} object with {} fields:", obj.kind, fields.len());
            for (k, _) in fields {
                s.push_str(&alloc::format!(" {};", k));
            }
            s
        }
    }
}

/// The set of views available for any object (views are on demand, not per-app).
pub fn available_views() -> [View; 4] {
    [View::Table, View::Graph, View::Spatial, View::Assistive]
}

// ─────────────────────────── abstract input model ───────────────────────────

/// A raw input event from any device class.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum InputEvent {
    Pointer { x: i32, y: i32, pressed: bool },
    Key(char),
    Touch { x: i32, y: i32 },
    Gesture(GestureKind),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GestureKind {
    Tap,
    Swipe,
    Pinch,
}

/// The normalised, device-independent action the compositor acts on — so the same
/// compositor serves desktop and mobile without forking.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UiAction {
    SelectAt(i32, i32),
    Activate,
    Text(char),
    Pan,
    Zoom,
}

/// Normalise any device event to a single UI action model.
pub fn normalize(ev: InputEvent) -> UiAction {
    match ev {
        InputEvent::Pointer { x, y, pressed: true } => UiAction::SelectAt(x, y),
        InputEvent::Pointer { .. } => UiAction::Activate,
        InputEvent::Touch { x, y } => UiAction::SelectAt(x, y),
        InputEvent::Key(c) => UiAction::Text(c),
        InputEvent::Gesture(GestureKind::Tap) => UiAction::Activate,
        InputEvent::Gesture(GestureKind::Swipe) => UiAction::Pan,
        InputEvent::Gesture(GestureKind::Pinch) => UiAction::Zoom,
    }
}

// ─────────────────────────── AI command bar ───────────────────────────

/// An intent parsed from natural language: a verb on a target, plus the authority it
/// requires. The command bar maps language → capability invocation.
#[derive(Clone, PartialEq, Debug)]
pub struct Intent {
    pub verb: String,
    pub target: String,
    pub needs: Rights,
}

/// Parse a natural-language command into a capability-gated [`Intent`]. A real build
/// would use the embedded LLM agent; this is a deterministic keyword intent parser
/// with the same contract (it never grants authority — it only *names* what is
/// needed, which the capability check then enforces).
pub fn interpret(nl: &str) -> Option<Intent> {
    let lower = to_lower(nl);
    let words: Vec<&str> = lower.split_whitespace().collect();
    let (verb, needs) = words.iter().find_map(|w| match *w {
        "show" | "open" | "read" | "view" | "find" => Some(("read", Rights::READ)),
        "edit" | "write" | "rename" | "set" | "save" => Some(("write", Rights::WRITE)),
        "delete" | "remove" | "shred" => Some(("delete", Rights::WRITE.union(Rights::GRANT))),
        "run" | "execute" | "launch" => Some(("run", Rights::EXECUTE)),
        "encrypt" | "seal" | "lock" => Some(("seal", Rights::SEAL)),
        _ => None,
    })?;
    // The target is the last noun-ish word (the object name).
    let target = words.last().copied().unwrap_or("").to_string();
    Some(Intent { verb: verb.to_string(), target, needs })
}

/// Invoke an intent under the domain's `granted` rights. Returns the action that
/// would run, or refuses if the capability is absent — the command bar cannot exceed
/// the user's authority.
pub fn invoke(intent: &Intent, granted: Rights) -> Result<String, UiError> {
    if !granted.contains(intent.needs) {
        return Err(UiError::Unauthorized);
    }
    Ok(alloc::format!("{} {}", intent.verb, intent.target))
}

/// Why a UI operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UiError {
    Unauthorized,
    NothingToUndo,
}

// ─────────────────────────── object-centric workspace ───────────────────────────

/// An object placed on the spatial canvas, with its current view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Placed {
    pub object: ObjectId,
    pub x: i32,
    pub y: i32,
    pub view: View,
}

/// The object-centric shell: a spatial canvas of objects, not an application menu.
#[derive(Default)]
pub struct Workspace {
    items: Vec<Placed>,
}

impl Workspace {
    pub fn new() -> Workspace {
        Workspace { items: Vec::new() }
    }

    /// Place an object on the canvas with an initial view.
    pub fn place(&mut self, object: ObjectId, x: i32, y: i32, view: View) {
        self.items.push(Placed { object, x, y, view });
    }

    /// Switch the view of every placement of an object (views on demand).
    pub fn set_view(&mut self, object: ObjectId, view: View) {
        for it in self.items.iter_mut() {
            if it.object == object {
                it.view = view;
            }
        }
    }

    pub fn items(&self) -> &[Placed] {
        &self.items
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A "program": a transient composition of capabilities + cells. There is no
/// install/uninstall — it exists only as long as its authority is held.
#[derive(Clone, PartialEq, Debug)]
pub struct Program {
    pub cells: Vec<String>,
    pub authority: Rights,
}

impl Program {
    /// Compose a program on the fly from cells + the authority it is granted.
    pub fn compose(cells: &[&str], authority: Rights) -> Program {
        Program { cells: cells.iter().map(|s| s.to_string()).collect(), authority }
    }

    /// Whether the composition may perform an action needing `needs` — least
    /// privilege, with no ambient authority.
    pub fn may(&self, needs: Rights) -> bool {
        self.authority.contains(needs)
    }
}

// ─────────────────────────── universal undo / time-travel ───────────────────────────

/// A system-wide undo history over content-addressed state roots. Each user action
/// commits a root; undo rewinds to the previous root deterministically (the Stage 10
/// rollback surfaced as a universal Ctrl-Z over the whole machine).
#[derive(Default)]
pub struct History {
    roots: Vec<Hash256>,
    /// Index of the current root within `roots`.
    cursor: usize,
    active: bool,
}

impl History {
    pub fn new(genesis: Hash256) -> History {
        History { roots: alloc::vec![genesis], cursor: 0, active: true }
    }

    /// Record a new state root after an action (truncates any redo tail).
    pub fn commit(&mut self, root: Hash256) {
        self.roots.truncate(self.cursor + 1);
        self.roots.push(root);
        self.cursor = self.roots.len() - 1;
    }

    /// The current state root.
    pub fn current(&self) -> Hash256 {
        self.roots[self.cursor]
    }

    /// Undo to the previous root. Returns the root now current.
    pub fn undo(&mut self) -> Result<Hash256, UiError> {
        if self.cursor == 0 {
            return Err(UiError::NothingToUndo);
        }
        self.cursor -= 1;
        Ok(self.current())
    }

    /// Redo to the next root if one exists.
    pub fn redo(&mut self) -> Option<Hash256> {
        if self.cursor + 1 < self.roots.len() {
            self.cursor += 1;
            Some(self.current())
        } else {
            None
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

// ─────────────────────────── unified settings (visible capabilities) ───────────────────────────

/// The single settings/permissions surface: enumerate the capabilities the domain
/// holds. There is no hidden ambient authority — what you can do is exactly this list.
pub fn permission_list(granted: Rights) -> Vec<&'static str> {
    let mut out = Vec::new();
    if granted.contains(Rights::READ) {
        out.push("read");
    }
    if granted.contains(Rights::WRITE) {
        out.push("write");
    }
    if granted.contains(Rights::EXECUTE) {
        out.push("execute");
    }
    if granted.contains(Rights::GRANT) {
        out.push("grant");
    }
    if granted.contains(Rights::SEAL) {
        out.push("seal");
    }
    out
}

fn to_lower(s: &str) -> String {
    s.chars().map(|c| c.to_ascii_lowercase()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Datum;

    fn doc() -> Object {
        Object::new("Invoice")
            .with("amount", Datum::Int(100))
            .with("payee", Datum::Text("acme".into()))
    }

    #[test]
    fn one_object_renders_through_many_views() {
        let o = doc();
        // Every view describes the same object differently — views on demand.
        let table = render(&o, View::Table);
        let graph = render(&o, View::Graph);
        let assistive = render(&o, View::Assistive);
        assert!(table.contains("Invoice") && table.contains("amount"));
        assert!(graph.contains("Invoice(") && graph.contains("payee"));
        // The assistive view is derived from the same typed fields (a11y tree = graph).
        assert!(assistive.contains("amount") && assistive.contains("payee"));
        assert_eq!(available_views().len(), 4);
    }

    #[test]
    fn abstract_input_unifies_desktop_and_mobile() {
        // A mouse press and a touch at the same point produce the same action.
        let m = normalize(InputEvent::Pointer { x: 10, y: 20, pressed: true });
        let t = normalize(InputEvent::Touch { x: 10, y: 20 });
        assert_eq!(m, t);
        assert_eq!(m, UiAction::SelectAt(10, 20));
        // A pinch gesture and... well, it zooms.
        assert_eq!(normalize(InputEvent::Gesture(GestureKind::Pinch)), UiAction::Zoom);
        assert_eq!(normalize(InputEvent::Key('a')), UiAction::Text('a'));
    }

    #[test]
    fn ai_command_bar_maps_language_to_capability_gated_action() {
        let intent = interpret("please open the invoice").unwrap();
        assert_eq!(intent.verb, "read");
        assert_eq!(intent.target, "invoice");
        // With READ, the action runs …
        assert_eq!(invoke(&intent, Rights::READ).unwrap(), "read invoice");
        // … without it, the command bar refuses (no ambient authority).
        assert_eq!(invoke(&intent, Rights::NONE), Err(UiError::Unauthorized));
    }

    #[test]
    fn destructive_commands_demand_more_authority() {
        let del = interpret("delete the old backup").unwrap();
        assert!(del.needs.contains(Rights::WRITE));
        // A read-only domain cannot delete.
        assert_eq!(invoke(&del, Rights::READ), Err(UiError::Unauthorized));
    }

    #[test]
    fn workspace_places_objects_and_switches_views() {
        let id = doc().id();
        let mut ws = Workspace::new();
        ws.place(id, 0, 0, View::Spatial);
        ws.place(id, 100, 0, View::Table);
        ws.set_view(id, View::Graph);
        assert_eq!(ws.len(), 2);
        assert!(ws.items().iter().all(|p| p.view == View::Graph));
    }

    #[test]
    fn programs_are_capability_compositions_not_installs() {
        let editor = Program::compose(&["TextCell", "StorageCell"], Rights::READ.union(Rights::WRITE));
        assert!(editor.may(Rights::WRITE));
        assert!(!editor.may(Rights::EXECUTE)); // least privilege — nothing ambient
    }

    #[test]
    fn universal_undo_rewinds_and_redoes() {
        let mut h = History::new(Hash256::of(b"genesis"));
        h.commit(Hash256::of(b"a"));
        h.commit(Hash256::of(b"b"));
        assert_eq!(h.current(), Hash256::of(b"b"));
        assert_eq!(h.undo().unwrap(), Hash256::of(b"a"));
        assert_eq!(h.undo().unwrap(), Hash256::of(b"genesis"));
        assert_eq!(h.undo(), Err(UiError::NothingToUndo));
        // Redo walks forward again.
        assert_eq!(h.redo(), Some(Hash256::of(b"a")));
        // A new commit truncates the redo tail.
        h.commit(Hash256::of(b"c"));
        assert_eq!(h.redo(), None);
    }

    #[test]
    fn settings_surface_lists_exactly_the_held_capabilities() {
        assert_eq!(permission_list(Rights::READ.union(Rights::WRITE)), alloc::vec!["read", "write"]);
        assert!(permission_list(Rights::NONE).is_empty());
    }
}

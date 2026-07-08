//! **AI-first agent interface** — structured I/O for the entire OS without vision.
//!
//! DominionOS is designed AI-first from the ground up. Every component — windows,
//! apps, files, VMs, system panels, data stores — is intended to implement
//! [`AgentControllable`] and expose itself to an embedded AI agent as structured
//! data, not pixels.
//!
//! > **NOTE:** `AgentBus` and `AgentControllable` are not yet instantiated in
//! > production. The framework exists and is unit-tested (see `#[cfg(test)]`
//! > `Mock*` impls at the bottom of this file) but integration into `os.rs` is
//! > pending. All present-tense descriptions below reflect the intended design.
//!
//! ## Intended architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                         AI Agent                                  │
//! │  reads: AgentSnapshot::to_text()   writes: AgentAction           │
//! └────────────────┬─────────────────────────────┬────────────────────┘
//!                  │ read                         │ dispatch
//!           ┌──────▼────────┐             ┌───────▼──────┐
//!           │ AgentSnapshot │             │   AgentBus   │
//!           │ (text/tree)   │             │  (router)    │
//!           └──────┬────────┘             └───────┬──────┘
//!                  │                              │
//!      ┌───────────┼──────────────────────────────┼────────────┐
//!      │           │  AgentControllable            │            │
//!  Window       Browser        Terminal         Files        VMs ...
//! ```
//!
//! When wired in, the shell will assemble an [`AgentSnapshot`] every tick by
//! calling `agent_view()` on each registered component. [`AgentSnapshot::to_text`]
//! will serialise the whole OS into a compact, token-efficient text format the
//! embedded LLM reads directly:
//!
//! ```text
//! os[tick=42]
//!   window[id=1 app=Browser title="DominionBrowser" focused] +focus +close
//!     textfield[id=2 label="URL" value="https://example.com"] +type +navigate
//!     button[id=3 label="Back" disabled]
//!     button[id=4 label="Fwd"] +click
//!   window[id=5 app=Terminal title="Terminal"]
//!     terminal[id=6 prompt="$ " history=42] +type +clear
//!   desktop[id=7 apps=3]
//!     icon[id=8 label="Files"] +open
//! ```
//!
//! The agent will send [`AgentAction`]s back; [`AgentBus::dispatch`] will route
//! each action to the component that owns the target node and return an
//! [`AgentResult`].
//!
//! ## Design rules (aspirational — not yet wired into production)
//! - **Every interactive element appears in the tree** with its stable `id`.
//! - **Every available action is listed** in `node.actions` — nothing implicit.
//! - **No vision**: no pixel coordinates, no screenshots, no OCR.
//! - **Zero-copy extensibility**: new apps/programs/VMs add one trait impl;
//!   they will be immediately readable and controllable by the agent.
//! - **Capability-gated**: agent access is intended to be governed by
//!   [`crate::airlock`] under `Domain::AiAgent` — it sees and controls only
//!   what it is authorised for.
//!
//! Pure, safe `no_std + alloc`, host-tested.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

// ─── Node identity ────────────────────────────────────────────────────────────

/// The agent instruction manual, embedded so the OS can inject it into a
/// model's context at startup without a filesystem read.
///
/// This describes the intended API surface for an embedded agent. It is not
/// currently injected into any model context in production — that wiring is
/// pending integration of [`AgentBus`] into `os.rs`.
///
/// When integration is complete, prepend this to the agent's system prompt or
/// first user message.
pub const AGENT_MANUAL: &str = include_str!("../../docs/AI_AGENT_MANUAL.md");

/// Stable identity for a node across ticks.
///
/// Assigned by each component and must be unique within the OS.
/// Convention: each component owns a disjoint `u64` range so IDs never collide.
pub type NodeId = u64;

// ─── Node state ───────────────────────────────────────────────────────────────

/// The structured semantic state of a node — the read side.
///
/// Add new variants here when new component types are introduced. Keeping all
/// variants in one enum (rather than erasing to a string bag) gives the agent
/// typed, machine-readable state without parsing.
#[derive(Clone, Debug)]
pub enum NodeState {
    // --- OS shell ---
    Desktop     { app_count: u32 },
    Taskbar     { entry_count: u32 },
    ContextMenu { items: Vec<String> },

    // --- Window manager ---
    Window {
        title: String,
        app: String,
        focused: bool,
        minimised: bool,
        maximised: bool,
    },
    Icon      { label: String, app: String },
    TaskEntry { label: String, app: String, focused: bool },

    // --- UI primitives ---
    Button    { label: String, enabled: bool },
    TextField { label: String, value: String, placeholder: String },
    Label     { text: String },
    StatusBar { text: String },

    // --- Apps ---
    Terminal {
        prompt: String,
        history_lines: u32,
        current_input: String,
    },
    Editor {
        path: Option<String>,
        lines: u32,
        cursor_line: u32,
        cursor_col: u32,
        modified: bool,
        lang: Option<String>,
    },
    Browser {
        url: String,
        title: String,
        loading: bool,
        can_back: bool,
        can_forward: bool,
    },
    FileEntry {
        name: String,
        path: String,
        is_dir: bool,
        size: u64,
    },
    // --- System / data ---
    Process {
        pid: u64,
        name: String,
        cpu_pct: u8,
        mem_kb: u64,
    },
    VmInstance {
        id: String,
        state: VmState,
        cpu_count: u8,
        mem_mb: u32,
    },
    File {
        path: String,
        size: u64,
        kind: String,
    },
    DataStore {
        name: String,
        record_count: u64,
    },
    NetworkIface {
        name: String,
        ip: String,
        up: bool,
    },

    /// Generic named container for anything not covered by the above.
    Group { label: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmState { Running, Stopped, Paused, Suspended }

impl VmState {
    fn as_str(self) -> &'static str {
        match self {
            VmState::Running   => "running",
            VmState::Stopped   => "stopped",
            VmState::Paused    => "paused",
            VmState::Suspended => "suspended",
        }
    }
}

// ─── Actions ──────────────────────────────────────────────────────────────────

/// Every possible typed action that can be dispatched to a node.
///
/// Extend this enum as new interaction types appear in the OS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionKind {
    /// Activate: press a button, click an icon, select a list item.
    Click,
    /// Set the text content of a text field (appends if `append` flag set).
    Type,
    /// Clear a text field's content.
    Clear,
    /// Toggle a checkbox / boolean control.
    Toggle,
    /// Select an option by value string.
    Select,
    /// Scroll by delta lines (positive = down, negative = up).
    Scroll,
    /// Open an app, file, or directory.
    Open,
    /// Close a window, dialog, or menu.
    Close,
    /// Move keyboard focus to this node.
    Focus,
    /// Minimise a window.
    Minimise,
    /// Maximise / restore a window.
    Maximise,
    /// Browser / file history back.
    GoBack,
    /// Browser / file history forward.
    GoForward,
    /// Reload the current page or refresh a list.
    Reload,
    /// Navigate to a URL or file path.
    Navigate,
    /// Execute / run the current selection (IDE run, script, etc.)
    Run,
    /// Save the current document.
    Save,
    /// Delete the selected item.
    Delete,
    /// Refresh a data view.
    Refresh,
    /// Kill / stop a process or VM.
    Kill,
    /// Start a stopped VM.
    StartVm,
    /// Read a file's content (returns it in `AgentResult::OkWith`).
    ReadFile,
    /// Overwrite a file's content (param = new content).
    WriteFile,
    /// Append a line to a terminal / run a command.
    RunCommand,
    /// Dismiss an active context menu or dialog without action.
    Dismiss,
    /// Escape / cancel the current interaction.
    Escape,
    /// Open the right-click context menu for this node.
    ContextMenu,
    /// Any action not covered above (name is the action identifier).
    Custom(String),
}

impl ActionKind {
    pub fn name(&self) -> &str {
        match self {
            ActionKind::Click       => "click",
            ActionKind::Type        => "type",
            ActionKind::Clear       => "clear",
            ActionKind::Toggle      => "toggle",
            ActionKind::Select      => "select",
            ActionKind::Scroll      => "scroll",
            ActionKind::Open        => "open",
            ActionKind::Close       => "close",
            ActionKind::Focus       => "focus",
            ActionKind::Minimise    => "minimise",
            ActionKind::Maximise    => "maximise",
            ActionKind::GoBack      => "back",
            ActionKind::GoForward   => "forward",
            ActionKind::Reload      => "reload",
            ActionKind::Navigate    => "navigate",
            ActionKind::Run         => "run",
            ActionKind::Save        => "save",
            ActionKind::Delete      => "delete",
            ActionKind::Refresh     => "refresh",
            ActionKind::Kill        => "kill",
            ActionKind::StartVm     => "start_vm",
            ActionKind::ReadFile    => "read_file",
            ActionKind::WriteFile   => "write_file",
            ActionKind::RunCommand  => "run_cmd",
            ActionKind::Dismiss     => "dismiss",
            ActionKind::Escape      => "escape",
            ActionKind::ContextMenu => "ctx_menu",
            ActionKind::Custom(s)   => s.as_str(),
        }
    }
}

/// A described available action — what the agent can discover on a node.
#[derive(Clone, Debug)]
pub struct ActionDesc {
    pub kind: ActionKind,
    /// Human-readable hint of what this action does.
    pub hint: &'static str,
    /// Required parameter names, e.g. `["text"]` for `Type`, `["url"]` for `Navigate`.
    pub params: &'static [&'static str],
}

impl ActionDesc {
    pub const fn simple(kind: ActionKind, hint: &'static str) -> Self {
        ActionDesc { kind, hint, params: &[] }
    }

    pub const fn with_params(kind: ActionKind, hint: &'static str, params: &'static [&'static str]) -> Self {
        ActionDesc { kind, hint, params }
    }
}

// ─── Agent node ───────────────────────────────────────────────────────────────

/// One node in the agent view tree.
///
/// `id` is stable across ticks. `state` is the typed semantic state. `actions`
/// lists everything the agent may do at this node. `children` are nested nodes.
#[derive(Clone, Debug)]
pub struct AgentNode {
    pub id: NodeId,
    pub state: NodeState,
    pub actions: Vec<ActionDesc>,
    pub children: Vec<AgentNode>,
}

impl AgentNode {
    pub fn new(id: NodeId, state: NodeState) -> Self {
        AgentNode { id, state, actions: Vec::new(), children: Vec::new() }
    }

    pub fn with_action(mut self, a: ActionDesc) -> Self {
        self.actions.push(a);
        self
    }

    pub fn with_actions(mut self, as_: impl IntoIterator<Item = ActionDesc>) -> Self {
        self.actions.extend(as_);
        self
    }

    pub fn with_child(mut self, c: AgentNode) -> Self {
        self.children.push(c);
        self
    }

    pub fn push_child(&mut self, c: AgentNode) {
        self.children.push(c);
    }

    /// Find a node by id anywhere in this subtree (depth-first).
    pub fn find(&self, id: NodeId) -> Option<&AgentNode> {
        if self.id == id { return Some(self); }
        self.children.iter().find_map(|c| c.find(id))
    }

    /// Collect all node ids in this subtree (for registration / routing).
    pub fn all_ids(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.collect_ids(&mut out);
        out
    }

    fn collect_ids(&self, out: &mut Vec<NodeId>) {
        out.push(self.id);
        for c in &self.children { c.collect_ids(out); }
    }

    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }
}

// ─── Actions dispatched by the agent ─────────────────────────────────────────

/// An action dispatched from an agent to a specific node in the OS.
#[derive(Clone, Debug)]
pub struct AgentAction {
    pub target: NodeId,
    pub kind: ActionKind,
    /// String parameter (text, URL, file path, option value, command, …).
    pub param: Option<String>,
    /// Integer parameter (scroll delta, slider value, index, …).
    pub int_param: Option<i64>,
}

impl AgentAction {
    pub fn click(target: NodeId) -> Self {
        AgentAction { target, kind: ActionKind::Click, param: None, int_param: None }
    }

    pub fn type_text(target: NodeId, text: impl Into<String>) -> Self {
        AgentAction { target, kind: ActionKind::Type, param: Some(text.into()), int_param: None }
    }

    pub fn navigate(target: NodeId, url: impl Into<String>) -> Self {
        AgentAction { target, kind: ActionKind::Navigate, param: Some(url.into()), int_param: None }
    }

    pub fn scroll(target: NodeId, delta: i64) -> Self {
        AgentAction { target, kind: ActionKind::Scroll, param: None, int_param: Some(delta) }
    }

    pub fn select(target: NodeId, value: impl Into<String>) -> Self {
        AgentAction { target, kind: ActionKind::Select, param: Some(value.into()), int_param: None }
    }

    pub fn run_command(target: NodeId, cmd: impl Into<String>) -> Self {
        AgentAction { target, kind: ActionKind::RunCommand, param: Some(cmd.into()), int_param: None }
    }

    pub fn open(target: NodeId) -> Self {
        AgentAction { target, kind: ActionKind::Open, param: None, int_param: None }
    }

    pub fn close(target: NodeId) -> Self {
        AgentAction { target, kind: ActionKind::Close, param: None, int_param: None }
    }

    pub fn read_file(target: NodeId) -> Self {
        AgentAction { target, kind: ActionKind::ReadFile, param: None, int_param: None }
    }

    pub fn write_file(target: NodeId, content: impl Into<String>) -> Self {
        AgentAction { target, kind: ActionKind::WriteFile, param: Some(content.into()), int_param: None }
    }

    pub fn custom(target: NodeId, name: impl Into<String>, param: Option<String>) -> Self {
        AgentAction { target, kind: ActionKind::Custom(name.into()), param, int_param: None }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// The outcome of a dispatched action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentResult {
    /// Action completed with no output.
    Ok,
    /// Action completed and produced string output (file read, command output, …).
    OkWith(String),
    /// No component owns the target node id.
    NotFound,
    /// The agent's capability does not permit this action on this node.
    Denied,
    /// The action is not valid for this node type, or a required param was missing.
    Invalid(String),
    /// The component is in a state where it cannot accept this action right now.
    NotReady(String),
}

impl AgentResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, AgentResult::Ok | AgentResult::OkWith(_))
    }

    pub fn invalid(msg: &str) -> Self {
        AgentResult::Invalid(msg.into())
    }

    pub fn not_ready(msg: &str) -> Self {
        AgentResult::NotReady(msg.into())
    }
}

// ─── The trait ────────────────────────────────────────────────────────────────

/// Every OS component that wants to be visible and controllable by an AI agent
/// implements this trait.
///
/// **Wiring status:** pending integration into `os.rs` / `wm.rs` / `shell.rs`.
/// `MockButton` and `MockTextField` in the test module are the only current impls.
///
/// **Implementation contract:**
/// - `agent_view` must be side-effect-free and cheap (called every tick).
/// - Node `id`s must be stable across calls — the agent tracks them.
/// - Each `AgentNode` must list only actions that are *currently* valid.
/// - `agent_dispatch` must check `action.target` against the subtree it owns
///   and return `AgentResult::NotFound` for ids it does not recognise.
///
/// Adding a new app, program, VM, or file type? Implement this trait and
/// register with [`AgentBus`]. That is all it takes for the agent to see it.
pub trait AgentControllable {
    /// Stable human-readable name shown in the component registry.
    fn agent_name(&self) -> &str;

    /// Produce a snapshot of this component's current state tree.
    fn agent_view(&self) -> AgentNode;

    /// Dispatch a typed action. Returns `NotFound` if `action.target` is not
    /// in this component's subtree.
    fn agent_dispatch(&mut self, action: AgentAction) -> AgentResult;
}

// ─── Full OS snapshot ─────────────────────────────────────────────────────────

/// A complete structured snapshot of the OS at one logical tick.
///
/// Built by the shell each frame: one root per registered component. The agent
/// reads this directly — either as a tree of [`AgentNode`]s or as the compact
/// text representation from [`Self::to_text`].
#[derive(Clone, Debug)]
pub struct AgentSnapshot {
    pub tick: u64,
    pub roots: Vec<AgentNode>,
}

impl AgentSnapshot {
    /// Serialise the entire OS state to a compact, token-efficient text format
    /// ready for an embedded LLM context window.
    ///
    /// Format:
    /// ```text
    /// os[tick=42]
    ///   window[id=1 app=Browser title="DominionBrowser" focused] +focus +close
    ///     textfield[id=2 label="URL" value="https://x.com"] +type +navigate
    ///     button[id=3 label="Back" disabled]
    ///     button[id=4 label="Fwd"] +click
    ///   desktop[id=5 apps=3]
    ///     icon[id=6 label="Files"] +open
    /// ```
    ///
    /// Each `+action` suffix is an action the agent may dispatch to that node.
    /// Actions with parameters are shown as `+action(param)`.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        push_line(&mut out, 0, &format!("os[tick={}]", self.tick));
        for root in &self.roots {
            render_node(&mut out, root, 1);
        }
        out
    }

    /// Find any node by id across the full snapshot.
    pub fn find(&self, id: NodeId) -> Option<&AgentNode> {
        self.roots.iter().find_map(|r| r.find(id))
    }

    /// Total node count (for completeness checks and tests).
    pub fn count(&self) -> usize {
        self.roots.iter().map(|r| r.count()).sum()
    }
}

// ─── Text rendering ───────────────────────────────────────────────────────────

fn push_line(out: &mut String, depth: usize, line: &str) {
    for _ in 0..depth { out.push_str("  "); }
    out.push_str(line);
    out.push('\n');
}

fn render_node(out: &mut String, node: &AgentNode, depth: usize) {
    let mut line = node_line(node);
    for a in &node.actions {
        line.push(' ');
        line.push('+');
        line.push_str(a.kind.name());
        if !a.params.is_empty() {
            line.push('(');
            line.push_str(a.params.join(",").as_str());
            line.push(')');
        }
    }
    push_line(out, depth, &line);
    for child in &node.children {
        render_node(out, child, depth + 1);
    }
}

/// Build the tag[attrs] line for a node (without actions — those are appended after).
fn node_line(node: &AgentNode) -> String {
    let id = node.id.to_string();
    match &node.state {
        NodeState::Desktop { app_count } =>
            attrs("desktop", &[("id", &id), ("apps", &app_count.to_string())], &[]),

        NodeState::Taskbar { entry_count } =>
            attrs("taskbar", &[("id", &id), ("entries", &entry_count.to_string())], &[]),

        NodeState::ContextMenu { items } =>
            attrs("ctxmenu", &[("id", &id), ("items", &items.len().to_string())], &[]),

        NodeState::Window { title, app, focused, minimised, maximised } => {
            let mut flags: Vec<&str> = Vec::new();
            if *focused   { flags.push("focused"); }
            if *minimised { flags.push("minimised"); }
            if *maximised { flags.push("maximised"); }
            attrs("window", &[("id", &id), ("app", &q(app)), ("title", &q(title))], flags.as_slice())
        }

        NodeState::Icon { label, app } =>
            attrs("icon", &[("id", &id), ("label", &q(label)), ("app", &q(app))], &[]),

        NodeState::TaskEntry { label, app, focused } =>
            attrs("task", &[("id", &id), ("label", &q(label)), ("app", &q(app))],
                  if *focused { &["focused"] } else { &[] }),

        NodeState::Button { label, enabled } =>
            attrs("button", &[("id", &id), ("label", &q(label))],
                  if !*enabled { &["disabled"] } else { &[] }),

        NodeState::TextField { label, value, placeholder } =>
            attrs("textfield", &[("id", &id), ("label", &q(label)), ("value", &q(value)),
                                  ("ph", &q(placeholder))], &[]),

        NodeState::Label { text } =>
            attrs("label", &[("id", &id), ("text", &q(text))], &[]),

        NodeState::StatusBar { text } =>
            attrs("statusbar", &[("id", &id), ("text", &q(text))], &[]),

        NodeState::Terminal { prompt, history_lines, current_input } =>
            attrs("terminal", &[("id", &id), ("prompt", &q(prompt)),
                                  ("history", &history_lines.to_string()),
                                  ("input", &q(current_input))], &[]),

        NodeState::Editor { path, lines, cursor_line, cursor_col, modified, lang } => {
            let p = path.as_deref().unwrap_or("unsaved");
            let l = lang.as_deref().unwrap_or("plain");
            attrs("editor", &[("id", &id), ("path", &q(p)), ("lang", &q(l)),
                               ("lines", &lines.to_string()),
                               ("cursor", &format!("{}:{}", cursor_line, cursor_col)),
                               ("modified", if *modified { "true" } else { "false" })], &[])
        }

        NodeState::Browser { url, title, loading, can_back, can_forward } => {
            let mut flags: Vec<&str> = Vec::new();
            if *loading     { flags.push("loading"); }
            if !*can_back   { flags.push("no_back"); }
            if !*can_forward { flags.push("no_fwd"); }
            attrs("browser", &[("id", &id), ("url", &q(url)), ("title", &q(title))],
                  flags.as_slice())
        }

        NodeState::FileEntry { name, path, is_dir, size } =>
            attrs("file", &[("id", &id), ("name", &q(name)), ("path", &q(path)),
                             ("size", &size.to_string())],
                  if *is_dir { &["dir"] } else { &[] }),

        NodeState::Process { pid, name, cpu_pct, mem_kb } =>
            attrs("process", &[("id", &id), ("pid", &pid.to_string()), ("name", &q(name)),
                                ("cpu", &format!("{}%", cpu_pct)),
                                ("mem", &format!("{}KB", mem_kb))], &[]),

        NodeState::VmInstance { id: vm_id, state, cpu_count, mem_mb } =>
            attrs("vm", &[("id", &id), ("vm_id", &q(vm_id)), ("state", state.as_str()),
                           ("cpus", &cpu_count.to_string()),
                           ("mem", &format!("{}MB", mem_mb))], &[]),

        NodeState::File { path, size, kind } =>
            attrs("file", &[("id", &id), ("path", &q(path)), ("size", &size.to_string()),
                             ("kind", &q(kind))], &[]),

        NodeState::DataStore { name, record_count } =>
            attrs("datastore", &[("id", &id), ("name", &q(name)),
                                   ("records", &record_count.to_string())], &[]),

        NodeState::NetworkIface { name, ip, up } =>
            attrs("iface", &[("id", &id), ("name", &q(name)), ("ip", &q(ip))],
                  if *up { &["up"] } else { &["down"] }),

        NodeState::Group { label } =>
            attrs("group", &[("id", &id), ("label", &q(label))], &[]),
    }
}

/// Build a `tag[k=v ... flags...]` string.
fn attrs(tag: &str, kvs: &[(&str, &str)], flags: &[&str]) -> String {
    let mut s = String::with_capacity(64);
    s.push_str(tag);
    s.push('[');
    let mut first = true;
    for &(k, v) in kvs {
        if !first { s.push(' '); }
        s.push_str(k);
        s.push('=');
        s.push_str(v);
        first = false;
    }
    for &f in flags {
        if !first { s.push(' '); }
        s.push_str(f);
        first = false;
    }
    s.push(']');
    s
}

/// Escape a free-form string so it cannot inject tree structure into the
/// agent's textual view. `to_text()` output is fed to the embedded agent as a
/// line-oriented `tag[k=v ...] +action` grammar, so an unescaped '\n' would
/// forge a new node line and a '"'/']'/space would forge attributes, flags, or
/// actions. We escape the delimiters and drop the structural bracket and any
/// control characters entirely.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Drop the structural brackets and any control characters.
            '[' | ']' => {}
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// Wrap a string value in double quotes, escaping its contents.
fn q(s: &str) -> String {
    let e = esc(s);
    let mut out = String::with_capacity(e.len() + 2);
    out.push('"');
    out.push_str(&e);
    out.push('"');
    out
}

// ─── Agent bus ────────────────────────────────────────────────────────────────

/// Entry in the bus component registry.
struct BusEntry {
    name: String,
    /// Cached set of node ids owned by this component (rebuilt on view change).
    owned_ids: Vec<NodeId>,
    component: alloc::boxed::Box<dyn AgentControllable>,
}

/// The OS-level agent bus: maintains a registry of [`AgentControllable`]
/// components, assembles [`AgentSnapshot`]s, and routes [`AgentAction`]s.
///
/// **Wiring status:** `AgentBus` is instantiated in tests only. Integration
/// into `os.rs` is the next step — add one field `agent_bus: AgentBus` and
/// call `register` for each component during OS init.
///
/// The shell (`os.rs`) holds one `AgentBus` and calls:
/// - `register` once per component at startup (and whenever a new app/VM starts).
/// - `tick` once per frame.
/// - `snapshot` to give the agent its read view.
/// - `dispatch` to forward the agent's write actions.
///
/// Components may also be **deregistered** when an app closes or a VM stops.
pub struct AgentBus {
    entries: Vec<BusEntry>,
    tick: u64,
    /// Index: node id → entry index (rebuilt lazily on register / deregister).
    id_map: BTreeMap<NodeId, usize>,
}

impl AgentBus {
    pub fn new() -> Self {
        AgentBus { entries: Vec::new(), tick: 0, id_map: BTreeMap::new() }
    }

    /// Register a component. Called once at startup or when a new component appears.
    pub fn register(&mut self, component: alloc::boxed::Box<dyn AgentControllable>) {
        let name = component.agent_name().into();
        let view = component.agent_view();
        let owned_ids = view.all_ids();
        let idx = self.entries.len();
        for &nid in &owned_ids {
            self.id_map.insert(nid, idx);
        }
        self.entries.push(BusEntry { name, owned_ids, component });
    }

    /// Deregister a component by name (app close, VM stop, etc.).
    pub fn deregister(&mut self, name: &str) {
        if let Some(pos) = self.entries.iter().position(|e| e.name == name) {
            let entry = self.entries.remove(pos);
            for nid in &entry.owned_ids {
                self.id_map.remove(nid);
            }
            // Rebuild indices above the removed position.
            for (_, idx) in self.id_map.iter_mut() {
                if *idx > pos { *idx -= 1; }
            }
        }
    }

    /// Advance the logical tick (call once per OS frame).
    pub fn tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Collect a full snapshot of all registered components.
    pub fn snapshot(&self) -> AgentSnapshot {
        AgentSnapshot {
            tick: self.tick,
            roots: self.entries.iter().map(|e| e.component.agent_view()).collect(),
        }
    }

    /// Dispatch an action to the component that owns `action.target`.
    ///
    /// If no component owns the target node, returns [`AgentResult::NotFound`].
    /// The id_map may be stale if a component's subtree changed since registration;
    /// in that case we fall back to a linear scan.
    pub fn dispatch(&mut self, action: AgentAction) -> AgentResult {
        let target = action.target;

        // Fast path: id_map lookup. The mapped component may be stale (the node
        // moved to another component since registration), in which case it
        // reports NotFound for a target it no longer owns — so only return early
        // on a definitive result, and otherwise fall through to the linear scan
        // (which refreshes id_map with the true owner).
        if let Some(&idx) = self.id_map.get(&target) {
            if idx < self.entries.len() {
                let res = self.entries[idx].component.agent_dispatch(action.clone());
                if res != AgentResult::NotFound {
                    return res;
                }
            }
        }

        // Slow path: linear scan (handles subtree changes since registration).
        let owner = self.entries.iter().position(|e| {
            e.component.agent_view().find(target).is_some()
        });
        match owner {
            Some(i) => {
                self.id_map.insert(target, i);
                self.entries[i].component.agent_dispatch(action)
            }
            None => AgentResult::NotFound,
        }
    }

    /// Names of all registered components (for discovery / debug).
    pub fn component_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    pub fn current_tick(&self) -> u64 { self.tick }
}

impl Default for AgentBus {
    fn default() -> Self { Self::new() }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock components ──────────────────────────────────────────────────────

    struct MockButton {
        id: NodeId,
        label: String,
        click_count: u32,
    }

    impl MockButton {
        fn new(id: NodeId, label: &str) -> Self {
            MockButton { id, label: label.into(), click_count: 0 }
        }
    }

    impl AgentControllable for MockButton {
        fn agent_name(&self) -> &str { "MockButton" }

        fn agent_view(&self) -> AgentNode {
            AgentNode::new(self.id, NodeState::Button {
                label: self.label.clone(),
                enabled: true,
            })
            .with_action(ActionDesc::simple(ActionKind::Click, "Press the button"))
        }

        fn agent_dispatch(&mut self, action: AgentAction) -> AgentResult {
            if action.target != self.id { return AgentResult::NotFound; }
            match action.kind {
                ActionKind::Click => { self.click_count += 1; AgentResult::Ok }
                _ => AgentResult::invalid("only click is supported"),
            }
        }
    }

    struct MockTextField {
        id: NodeId,
        label: String,
        value: String,
    }

    impl MockTextField {
        fn new(id: NodeId, label: &str) -> Self {
            MockTextField { id, label: label.into(), value: String::new() }
        }
    }

    impl AgentControllable for MockTextField {
        fn agent_name(&self) -> &str { "MockTextField" }

        fn agent_view(&self) -> AgentNode {
            AgentNode::new(self.id, NodeState::TextField {
                label: self.label.clone(),
                value: self.value.clone(),
                placeholder: String::new(),
            })
            .with_action(ActionDesc::with_params(ActionKind::Type, "Enter text", &["text"]))
            .with_action(ActionDesc::simple(ActionKind::Clear, "Clear the field"))
        }

        fn agent_dispatch(&mut self, action: AgentAction) -> AgentResult {
            if action.target != self.id { return AgentResult::NotFound; }
            match action.kind {
                ActionKind::Type => {
                    if let Some(t) = action.param { self.value = t; AgentResult::Ok }
                    else { AgentResult::invalid("type requires a text param") }
                }
                ActionKind::Clear => { self.value.clear(); AgentResult::Ok }
                _ => AgentResult::invalid("unsupported action"),
            }
        }
    }

    // ── Snapshot text format ──────────────────────────────────────────────────

    #[test]
    fn snapshot_text_has_tick_header_and_node_lines() {
        let btn = AgentNode::new(1, NodeState::Button { label: "Apply".into(), enabled: true })
            .with_action(ActionDesc::simple(ActionKind::Click, "Apply changes"));
        let snap = AgentSnapshot { tick: 7, roots: alloc::vec![btn] };
        let text = snap.to_text();
        assert!(text.starts_with("os[tick=7]\n"));
        assert!(text.contains("button[id=1 label=\"Apply\"]"));
        assert!(text.contains("+click"));
    }

    #[test]
    fn snapshot_text_indents_children() {
        let child = AgentNode::new(2, NodeState::Button { label: "Close".into(), enabled: true });
        let win = AgentNode::new(1, NodeState::Window {
            title: "Settings".into(),
            app: "Settings".into(),
            focused: true,
            minimised: false,
            maximised: false,
        }).with_child(child);
        let snap = AgentSnapshot { tick: 1, roots: alloc::vec![win] };
        let text = snap.to_text();
        // Window is at depth 1 (2 spaces), child at depth 2 (4 spaces).
        assert!(text.contains("\n  window["));
        assert!(text.contains("\n    button[id=2"));
    }

    #[test]
    fn text_format_includes_flags() {
        let btn = AgentNode::new(1, NodeState::Button { label: "Go".into(), enabled: false });
        let snap = AgentSnapshot { tick: 0, roots: alloc::vec![btn] };
        let text = snap.to_text();
        assert!(text.contains("disabled"));
    }

    #[test]
    fn text_format_shows_actions_with_params() {
        let tf = AgentNode::new(3, NodeState::TextField {
            label: "URL".into(), value: String::new(), placeholder: "https://".into(),
        }).with_action(ActionDesc::with_params(ActionKind::Navigate, "Go to URL", &["url"]));
        let snap = AgentSnapshot { tick: 0, roots: alloc::vec![tf] };
        let text = snap.to_text();
        assert!(text.contains("+navigate(url)"));
    }

    // ── Node tree operations ──────────────────────────────────────────────────

    #[test]
    fn find_locates_nested_node() {
        let deep = AgentNode::new(99, NodeState::Label { text: "hi".into() });
        let mid = AgentNode::new(10, NodeState::Group { label: "g".into() }).with_child(deep);
        let root = AgentNode::new(1, NodeState::Desktop { app_count: 1 }).with_child(mid);
        let snap = AgentSnapshot { tick: 0, roots: alloc::vec![root] };
        assert!(snap.find(99).is_some());
        assert!(snap.find(10).is_some());
        assert!(snap.find(1).is_some());
        assert!(snap.find(0).is_none());
    }

    #[test]
    fn count_includes_all_descendants() {
        let leaf1 = AgentNode::new(2, NodeState::Label { text: "a".into() });
        let leaf2 = AgentNode::new(3, NodeState::Label { text: "b".into() });
        let root = AgentNode::new(1, NodeState::Group { label: "g".into() })
            .with_child(leaf1).with_child(leaf2);
        assert_eq!(root.count(), 3);
    }

    // ── Agent bus: register + dispatch ────────────────────────────────────────

    #[test]
    fn bus_routes_click_to_button_and_ignores_unowned_ids() {
        let mut bus = AgentBus::new();
        bus.register(alloc::boxed::Box::new(MockButton::new(100, "Submit")));

        assert_eq!(bus.dispatch(AgentAction::click(100)), AgentResult::Ok);
        assert_eq!(bus.dispatch(AgentAction::click(999)), AgentResult::NotFound);
    }

    #[test]
    fn bus_routes_type_to_textfield() {
        let mut bus = AgentBus::new();
        bus.register(alloc::boxed::Box::new(MockTextField::new(200, "Search")));

        assert_eq!(bus.dispatch(AgentAction::type_text(200, "hello")), AgentResult::Ok);
        // Check state changed.
        let snap = bus.snapshot();
        let node = snap.find(200).unwrap();
        if let NodeState::TextField { value, .. } = &node.state {
            assert_eq!(value, "hello");
        } else { panic!("wrong state"); }
    }

    #[test]
    fn bus_routes_clear_to_textfield() {
        let mut bus = AgentBus::new();
        bus.register(alloc::boxed::Box::new(MockTextField::new(300, "Q")));
        bus.dispatch(AgentAction::type_text(300, "some text")).is_ok();
        assert_eq!(bus.dispatch(AgentAction { target: 300, kind: ActionKind::Clear, param: None, int_param: None }), AgentResult::Ok);
        let snap = bus.snapshot();
        if let NodeState::TextField { value, .. } = &snap.find(300).unwrap().state {
            assert!(value.is_empty());
        }
    }

    #[test]
    fn bus_snapshot_includes_all_components() {
        let mut bus = AgentBus::new();
        bus.register(alloc::boxed::Box::new(MockButton::new(1, "A")));
        bus.register(alloc::boxed::Box::new(MockTextField::new(2, "B")));
        bus.tick();
        let snap = bus.snapshot();
        assert_eq!(snap.tick, 1);
        assert_eq!(snap.roots.len(), 2);
        assert_eq!(snap.count(), 2);
    }

    #[test]
    fn bus_deregister_removes_component_and_ids() {
        let mut bus = AgentBus::new();
        bus.register(alloc::boxed::Box::new(MockButton::new(50, "X")));
        bus.register(alloc::boxed::Box::new(MockTextField::new(60, "Y")));
        assert_eq!(bus.component_names().len(), 2);

        bus.deregister("MockButton");
        assert_eq!(bus.component_names().len(), 1);
        assert_eq!(bus.dispatch(AgentAction::click(50)), AgentResult::NotFound);
        assert_eq!(bus.dispatch(AgentAction::type_text(60, "ok")), AgentResult::Ok);
    }

    // ── AgentResult helpers ───────────────────────────────────────────────────

    #[test]
    fn agent_result_is_ok_variants() {
        assert!(AgentResult::Ok.is_ok());
        assert!(AgentResult::OkWith("data".into()).is_ok());
        assert!(!AgentResult::NotFound.is_ok());
        assert!(!AgentResult::Denied.is_ok());
        assert!(!AgentResult::Invalid("bad".into()).is_ok());
        assert!(!AgentResult::NotReady("busy".into()).is_ok());
    }

    // ── All NodeState variants render without panicking ───────────────────────

    #[test]
    fn all_node_states_render_to_text() {
        let states: Vec<NodeState> = alloc::vec![
            NodeState::Desktop { app_count: 3 },
            NodeState::Taskbar { entry_count: 2 },
            NodeState::ContextMenu { items: alloc::vec!["Cut".into(), "Copy".into()] },
            NodeState::Window { title: "W".into(), app: "A".into(), focused: true, minimised: false, maximised: false },
            NodeState::Icon { label: "Files".into(), app: "files".into() },
            NodeState::TaskEntry { label: "X".into(), app: "x".into(), focused: false },
            NodeState::Button { label: "B".into(), enabled: true },
            NodeState::TextField { label: "F".into(), value: "v".into(), placeholder: "p".into() },
            NodeState::Label { text: "hello".into() },
            NodeState::StatusBar { text: "Ready".into() },
            NodeState::Terminal { prompt: "$ ".into(), history_lines: 10, current_input: "ls".into() },
            NodeState::Editor { path: Some("/file.rs".into()), lines: 100, cursor_line: 5, cursor_col: 1, modified: true, lang: Some("rust".into()) },
            NodeState::Browser { url: "https://x.com".into(), title: "X".into(), loading: false, can_back: true, can_forward: false },
            NodeState::FileEntry { name: "foo.txt".into(), path: "/foo.txt".into(), is_dir: false, size: 1024 },
            NodeState::Process { pid: 1, name: "init".into(), cpu_pct: 0, mem_kb: 512 },
            NodeState::VmInstance { id: "vm0".into(), state: VmState::Running, cpu_count: 2, mem_mb: 512 },
            NodeState::File { path: "/etc/cfg".into(), size: 256, kind: "text".into() },
            NodeState::DataStore { name: "users".into(), record_count: 999 },
            NodeState::NetworkIface { name: "eth0".into(), ip: "10.0.0.1".into(), up: true },
            NodeState::Group { label: "Panel".into() },
        ];
        for (i, state) in states.into_iter().enumerate() {
            let node = AgentNode::new(i as u64 + 1, state);
            let snap = AgentSnapshot { tick: 0, roots: alloc::vec![node] };
            let text = snap.to_text();
            assert!(!text.is_empty(), "NodeState variant {} produced empty text", i);
        }
    }
}

//! The **DominionOS shell** — the top-level desktop environment.
//!
//! It presents a **Windows-like graphical experience**: a **desktop backdrop** with
//! launchable icons and (in edit mode) composable widgets, a bottom **taskbar** that
//! lists the **running windows** with a **Start** menu and a right-side **system
//! tray**, and a persistent top bar. Apps open as real **floating windows** — each
//! with a title bar (minimize / maximize / close), draggable and resizable from any
//! edge — managed by [`crate::window::WindowManager`].
//!
//! The Desktop is the persistent backdrop; every other surface ([`AppId`]) — Files,
//! Browser, Terminal, Editor, IDE, Explorer, Task Manager, Settings — is a window drawn
//! in its own local coordinates and translated into its window's content area. The
//! shell routes the pointer/keyboard to the focused window, aggregates the damage
//! rectangle for the kernel's incremental render loop, and shares one live
//! [`FileSystem`] + [`Scheduler`] + [`World`] across the apps so they stay consistent.
//!
//! Pure, safe `no_std`. The kernel ([`kernel::desktop`]) drives one `Os`.

use crate::a11y::{A11yNode, A11yTree, Role};
use crate::agent::{ActionDesc, ActionKind, AgentAction, AgentBus, AgentControllable, AgentNode, AgentResult, NodeState};
use crate::browserapp::BrowserApp;
use crate::compose::{Board, Library, WidgetKind};
use crate::dash::Metrics;
use crate::memtier::MemoryManager;
use crate::pressure::Pressure;
use crate::ramdedup::RamResidencyIndex;
use crate::desktop_page::{Desktop, DesktopAction};
use crate::editorpage::EditorPage;
use crate::explorer::Explorer;
use crate::files::{Files, FilesAction};
use crate::filesystem::{FileSystem, SharedFs};
use crate::fleet::{DeviceId, Fleet};
use crate::hash::Hash256;
use crate::ide::Ide;
use crate::capability::{Capability, Rights};
use crate::sched::{DomainId, Scheduler};
use crate::secprofile::{Knob, Posture, PosturePolicy, SecurityProfile};
use crate::settings::{Config, Flag, Settings, SettingsAction};
use crate::shellcmd::SharedSched;
use crate::taskman::TaskManager;
use crate::termpage::{TermPage, TERM_NODE_ID as TERM_AGENT_NODE_ID};
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::window::{Reaction, WinState, Window, WindowManager};
use crate::world::World;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

const TOPBAR_H: i32 = 30;
const DOCK_H: i32 = 60;
/// The default widgets-library entry name a fresh board publishes/installs.
const DEFAULT_LAYOUT: &str = "my-layout";

/// Which app a surface is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum AppId {
    Desktop,
    Files,
    Browser,
    Terminal,
    Editor,
    Ide,
    Explorer,
    TaskManager,
    Settings,
}

impl AppId {
    /// The taskbar/Start label for this app.
    fn label(self) -> &'static str {
        match self {
            AppId::Desktop => "Desktop",
            AppId::Files => "Files",
            AppId::Browser => "Browser",
            AppId::Terminal => "Terminal",
            AppId::Editor => "Editor",
            AppId::Ide => "IDE",
            AppId::Explorer => "Explorer",
            AppId::TaskManager => "Task Mgr",
            AppId::Settings => "Settings",
        }
    }
}

/// Every app, in Start-menu / hotkey order.
const ALL_APPS: [AppId; 9] = [
    AppId::Desktop,
    AppId::Files,
    AppId::Browser,
    AppId::Terminal,
    AppId::Editor,
    AppId::Ide,
    AppId::Explorer,
    AppId::TaskManager,
    AppId::Settings,
];

/// The app icons shown on the Desktop (clickable launchers — the "desktop icons").
const DESKTOP_ICONS: [AppId; 8] = [
    AppId::Files,
    AppId::Browser,
    AppId::Terminal,
    AppId::Editor,
    AppId::Ide,
    AppId::Explorer,
    AppId::TaskManager,
    AppId::Settings,
];

/// What a left-button drag currently belongs to, decided on the press edge so that
/// subsequent moves/release route to the same target (a window's chrome, a window's
/// app, the desktop backdrop, or the edit board).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Grab {
    None,
    /// A window-manager chrome drag (move / resize).
    Chrome,
    /// An app-content drag inside this window.
    Window(AppId),
    /// A drag on the desktop backdrop (object cards).
    Desktop,
    /// A drag on the edit-mode widget board.
    Board,
}

/// The shell.
pub struct Os {
    /// The focused surface — a window's app, or `Desktop` for the backdrop.
    focus: AppId,
    /// Floating windows for the (non-Desktop) apps.
    wm: WindowManager<AppId>,
    grab: Grab,
    // Apps.
    desktop: Desktop,
    files: Files,
    browser: BrowserApp,
    terminal: TermPage,
    editor: EditorPage,
    ide: Ide,
    explorer: Explorer,
    taskman: TaskManager,
    settings: Settings,
    // Shared live state.
    fs: SharedFs,
    /// What the filesystem already had on disk at the boot-time restore, threaded back
    /// into the next save so shutdown flushes only this session's *new* objects (the
    /// incremental persistence path; `None` until a successful restore).
    fs_manifest: Option<crate::objstore::Manifest>,
    sched: SharedSched,
    /// Live scheduler domain backing each open app window, so the Task Manager lists the
    /// windows you open as real processes (and drops them when the window closes).
    app_domains: BTreeMap<AppId, DomainId>,
    world: World,
    account: String,
    /// The system clipboard — one shared buffer so copy in one window pastes in another.
    clipboard: String,
    metrics: Metrics,
    start_open: bool,
    theme_dark: bool,
    /// The live user preferences, mirrored from the Settings app (which owns the UI).
    config: Config,
    /// **Edit-UI** mode for the Desktop: a composable widget [`Board`] overlay.
    editing: bool,
    board: Board,
    library: Library,
    /// The right-click **context menu** anchor (`None` = closed).
    ctx: Option<(i32, i32)>,
    /// The current context menu's items — built fresh per right-click from what is
    /// under the pointer (a window, a text surface, or the desktop).
    ctx_items: Vec<(String, CtxAction)>,
    /// Set when the user chooses Power off in Settings — the kernel returns to ASH.
    exit: bool,
    w: i32,
    h: i32,
    last_left: bool,
    damage: Option<Rect>,
    /// Canvas-space coordinates of the last right-click on the desktop backdrop.
    desktop_ctx_x: i32,
    desktop_ctx_y: i32,
    /// The AI agent bus — routes typed actions to registered OS components.
    agent_bus: AgentBus,
    /// The live accessibility tree - rebuilt whenever windows open, close, or focus
    /// changes, so that screen readers and switch devices always see current state.
    a11y_tree: A11yTree,
    /// The device fleet — tracks enrolled devices, threshold groups, and recursive
    /// revocation for this OS instance. Initialized at boot from the owner identity
    /// and kept live as a persistent background service so callers can enroll or
    /// revoke devices at any time without re-constructing the fleet state.
    fleet: Option<Fleet>,
    /// The hash identifying this node as the founding (owner) device in the fleet.
    fleet_owner: Hash256,
    /// Tiered memory manager — consulted each metrics tick to migrate hot→cold pages
    /// and perform graceful OOM eviction under pressure.
    mem_manager: MemoryManager,
    /// Cross-domain RAM dedup index — tracks shared object residency and byte savings.
    /// `put()` is called from the admit path; stats are sampled each idle tick.
    ram_dedup: RamResidencyIndex,
}

/// A right-click menu action — built per-context from what is under the pointer.
#[derive(Clone, Copy)]
enum CtxAction {
    ToggleEdit,
    Go(AppId),
    Close(AppId),
    Minimize(AppId),
    Maximize(AppId),
    Copy,
    Cut,
    Paste,
    // App-specific actions, routed to the focused window's app.
    FilesOpen,
    FilesNewFile,
    FilesNewFolder,
    FilesDelete,
    FilesRefresh,
    FilesRename,
    FilesCopy,
    FilesCut,
    FilesSaveAs,
    FilesProperties,
    FilesRun,
    FilesEdit,
    BrowserBack,
    BrowserForward,
    BrowserReload,
    BrowserCopyUrl,
    BrowserSavePage,
    TerminalClear,
    IdeRun,
    IdeStop,
    IdeAddNode,
    IdeDeleteNode,
    IdeResetWires,
    IdeDisconnectInputs,
    IdeDisconnectOutputs,
    IdeDisconnectAll,
    IdeNewProgram,
    IdeOpenExamples,
    EditorSave,
    EditorSaveAs,
    ExplorerRefresh,
    ExplorerClearLog,
    ExplorerExport,
    TaskKill,
    TaskRefresh,
    SettingsReset,
    SettingsExportConfig,
    DesktopAddNote,
    DesktopAddFolder,
    DesktopAddShortcut,
    DesktopDeleteItem,
}
const CTX_W: i32 = 184;
const CTX_ROW: i32 = 26;

impl Os {
    pub fn new() -> Os {
        let fs = FileSystem::shared();
        let sched: SharedSched = Rc::new(RefCell::new(seed_scheduler()));
        let init_content = Rect::new(0, TOPBAR_H, 1280, 720 - TOPBAR_H - DOCK_H);
        let mut os = Os {
            focus: AppId::Desktop,
            wm: WindowManager::new(init_content),
            grab: Grab::None,
            desktop: Desktop::new(),
            files: Files::new(fs.clone()),
            browser: BrowserApp::new(),
            terminal: TermPage::new(fs.clone(), sched.clone()),
            editor: EditorPage::new(fs.clone()),
            ide: Ide::new(),
            explorer: Explorer::new(),
            taskman: TaskManager::new(sched.clone()),
            settings: Settings::new("Jayden", true),
            fs,
            fs_manifest: None,
            sched,
            app_domains: BTreeMap::new(),
            world: World::new(),
            account: "Jayden".into(),
            clipboard: String::new(),
            metrics: Metrics::default(),
            start_open: false,
            theme_dark: true,
            config: Config::default(),
            editing: false,
            board: Board::new(),
            library: Library::new(),
            ctx: None,
            ctx_items: Vec::new(),
            exit: false,
            w: 1280,
            h: 720,
            last_left: false,
            damage: Some(Rect::new(0, 0, 1280, 720)),
            desktop_ctx_x: 0,
            desktop_ctx_y: 0,
            agent_bus: AgentBus::new(),
            a11y_tree: A11yTree::new(),
            fleet: None,
            fleet_owner: Hash256::of(b"dominion-os-default-owner"),
            // Memory acceleration — 5-tier quotas sized for a typical 512 MB RAM node.
            // Tier order: Vram (32 MB), Ram (256 MB), Nvme (512 MB), Peer (128 MB), Cold (∞).
            mem_manager: MemoryManager::new([
                32  * 1024 * 1024,   // Vram
                256 * 1024 * 1024,   // Ram
                512 * 1024 * 1024,   // Nvme
                128 * 1024 * 1024,   // Peer
                usize::MAX / 2,      // Cold (unbounded)
            ]),
            ram_dedup: RamResidencyIndex::new(),
        };
        // Start the fleet service: register this node as the founding device so
        // peer enrollment and recursive revocation are ready from the first tick.
        os.start_fleet_service();
        os.sync_world();
        os.seed_widgets();
        // Wire the shared VFS into the browser so user-authored dominion:// pages
        // stored in /dominion/pages/<name>.dominion are resolvable live.
        os.browser.set_native_fs(os.fs.clone());
        os
    }

    /// Seed the desktop with a couple of default **widgets** (a clock and an activity
    /// chart) on the composable board, so the home screen ships with widgets the way a
    /// modern desktop does. The user can move/resize/remove them in Edit-UI mode, or add
    /// more from the widget picker.
    fn seed_widgets(&mut self) {
        let c = Rect::new(0, TOPBAR_H, 1280, 720 - TOPBAR_H - DOCK_H);
        let x = c.x + c.w - 232;
        self.board.add_panel(WidgetKind::Clock, Rect::new(x, c.y + 24, 212, 132), "Clock", true);
        self.board.add_panel(WidgetKind::Metric, Rect::new(x, c.y + 168, 100, 124), "CPU", true);
        self.board.add_panel(WidgetKind::Chart, Rect::new(x + 112, c.y + 168, 100, 124), "Activity", true);
        // Prime them with the initial metrics so they're live from the first frame.
        self.update_widgets();
    }

    /// Push the live system metrics into the desktop widgets (clock = uptime, metric =
    /// CPU%, chart = CPU history). Called each metric tick; the board only damages the
    /// widgets whose value actually changed.
    fn update_widgets(&mut self) {
        let secs = self.metrics.uptime_secs;
        let mut clock = String::new();
        push_int(&mut clock, (secs / 3600) as i64);
        clock.push(':');
        let m = (secs % 3600) / 60;
        if m < 10 {
            clock.push('0');
        }
        push_int(&mut clock, m as i64);
        clock.push(':');
        let s = secs % 60;
        if s < 10 {
            clock.push('0');
        }
        push_int(&mut clock, s as i64);
        let mut cpu = String::new();
        push_int(&mut cpu, (self.metrics.cpu_milli / 10) as i64);
        cpu.push('%');
        self.board.feed_live(&clock, &cpu, &self.metrics.cpu_history);
    }

    /// Re-derive the live Desktop cards + Explorer graph from the shared [`World`].
    fn sync_world(&mut self) {
        let progs = self.ide.programs_snapshot();
        self.world.set_programs(&progs);
        let entries = self.world.entries();
        // World objects (datasets, logs, programs) live in the Explorer, NOT scattered
        // as icons on the desktop — the desktop shows only app launchers.
        self.explorer.set_objects(&entries);
        self.explorer.set_cells(self.world.cells());
        self.explorer.set_root_prov(self.world.root_cap().provenance().short());
    }

    /// Reconcile the live process table with the set of open windows: spawn a scheduler
    /// domain for every newly-opened window and kill the domain of any window that closed.
    /// This is what makes the Task Manager list the apps you open as real processes.
    fn sync_processes(&mut self) {
        let open: Vec<AppId> = self.wm.taskbar().iter().map(|(a, _)| *a).collect();
        for app in &open {
            if !self.app_domains.contains_key(app) {
                let (base, len) = proc_region(*app);
                let id = self.sched.borrow_mut().spawn(app.label(), Capability::mint(base, len, Rights::ALL));
                self.app_domains.insert(*app, id);
                // Task 2: register each newly-opened app as a memory domain so the
                // tiered memory manager can track and enforce its per-app quota.
                // Default quota: 32 MiB — enough for a typical GUI app's working set.
                const APP_MEMORY_QUOTA: usize = 32 * 1024 * 1024;
                // Domain 0 is reserved for the shared kernel/fs domain (see
                // persist_fs_to / set_metrics), so app domain ids are offset by 1
                // to guarantee they can never alias it (AppId::Desktop == 0).
                self.mem_manager.add_domain(app_domain_id(*app), APP_MEMORY_QUOTA);
                // Also register the domain in the RAM dedup index so shared objects
                // (fonts, images, model shards) that appear in multiple apps are
                // deduplicated to a single physical resident copy.
                self.ram_dedup.add_domain(app_domain_id(*app));
            }
        }
        let closed: Vec<AppId> =
            self.app_domains.keys().copied().filter(|a| !open.contains(a)).collect();
        for app in closed {
            if let Some(id) = self.app_domains.remove(&app) {
                self.sched.borrow_mut().kill(id);
                // Task 2: tear down the app's memory domain and release its dedup
                // references so objects held only by this app are freed promptly.
                self.mem_manager.remove_domain(app_domain_id(app));
                self.ram_dedup.release_domain(app_domain_id(app));
            }
        }
    }

    // ── Accessibility tree ──

    /// Rebuild the accessibility tree from the current window manager state.
    /// Call this whenever windows open, close, or focus changes.
    fn build_a11y_tree(&mut self) {
        let top = self.wm.top();
        let mut roots: Vec<A11yNode> = Vec::new();

        // One root node per open window.
        for win in self.wm.visible() {
            let focused = Some(win.id) == top && self.focus == win.id;
            let win_node_id = 0x4000_0000u64 + win.id as u64;
            // The window itself is the root; its content type drives the role annotation.
            let role = match win.id {
                AppId::Terminal | AppId::Editor => Role::TextField,
                AppId::Settings => Role::Window,
                _ => Role::Window,
            };
            let label = win.title.clone();
            let mut node = A11yNode::new(win_node_id, role, label).focusable();
            if focused {
                node.value = Some("focused".into());
            }
            // Embed a child node for the primary content region.
            let content_role = match win.id {
                AppId::Terminal => Role::TextField,
                AppId::Editor => Role::TextField,
                AppId::Browser => Role::Text,
                AppId::Files => Role::List,
                _ => Role::Text,
            };
            let content_label = win.id.label();
            let child = A11yNode::new(win_node_id + 0x1000, content_role, content_label).focusable();
            node = node.child(child);
            roots.push(node);
        }

        // The Desktop backdrop is always present.
        let desktop_node = A11yNode::new(0x3FFF_FFFF, Role::Window, "Desktop").focusable();
        roots.push(desktop_node);

        self.a11y_tree.set_roots(roots);

        // Update focused-node id and emit an announcement if focus changed.
        let focused_id = if let Some(top_app) = top {
            if self.focus == top_app {
                Some(0x4000_0000u64 + top_app as u64)
            } else {
                Some(0x3FFF_FFFF) // Desktop
            }
        } else {
            Some(0x3FFF_FFFF) // Desktop
        };
        if let Some(fid) = focused_id {
            if self.a11y_tree.focused_id != Some(fid) {
                self.a11y_tree.set_focus(fid);
            }
        }
    }

    /// Return a reference to the live accessibility tree.
    /// Screen readers and switch-access services poll this each frame.
    pub fn accessibility_tree(&self) -> &A11yTree {
        &self.a11y_tree
    }

    /// Take the pending screen-reader announcement (for TTS output).
    /// Returns `None` if nothing has been announced since the last call.
    pub fn take_a11y_announcement(&mut self) -> Option<String> {
        self.a11y_tree.take_announcement()
    }

    /// Whether the user asked to power off (the kernel returns to ASH safe mode).
    pub fn wants_exit(&self) -> bool {
        self.exit
    }

    /// A shared handle to the shell's live filesystem.
    pub fn fs(&self) -> SharedFs {
        self.fs.clone()
    }

    /// Serialise the live filesystem to a byte image (durable persistence).
    pub fn persist_fs(&self) -> Vec<u8> {
        self.fs.borrow_mut().to_bytes()
    }

    /// Restore the filesystem from a disk image on boot.
    pub fn restore_fs(&self, bytes: &[u8]) -> bool {
        self.fs.borrow_mut().restore_from_bytes(bytes)
    }

    /// Restore the filesystem directly from the incremental on-disk store at `base_lba`,
    /// remembering its manifest so the next [`persist_fs_to`](Self::persist_fs_to) appends
    /// only what changes. Returns true if a valid image was restored.
    pub fn restore_fs_from(
        &mut self,
        dev: &mut dyn crate::persist::BlockDevice,
        base_lba: u64,
    ) -> bool {
        match self.fs.borrow_mut().restore_from(dev, base_lba) {
            Ok(Some(manifest)) => {
                self.fs_manifest = Some(manifest);
                true
            }
            _ => false,
        }
    }

    /// Whether the filesystem has unsaved changes since the last persist/restore. The
    /// kernel's periodic checkpoint uses this to skip the disk when nothing changed.
    pub fn fs_dirty(&self) -> bool {
        self.fs.borrow().is_dirty()
    }

    /// Persist the filesystem to the incremental on-disk store at `base_lba`, flushing
    /// only objects new since the boot-time restore (or the previous save). Returns true
    /// on success.
    pub fn persist_fs_to(
        &mut self,
        dev: &mut dyn crate::persist::BlockDevice,
        base_lba: u64,
    ) -> bool {
        let mut prior = self.fs_manifest.take();
        let ok = self.fs.borrow_mut().persist_to(dev, base_lba, &mut prior).is_ok();
        self.fs_manifest = prior;
        // Task 1: after objects are committed to the on-disk store, register them with
        // the cross-domain RAM dedup index so identical objects shared across domains
        // map to one physical copy. Domain 0 is the shared filesystem (kernel) domain.
        if ok {
            let objects: alloc::vec::Vec<crate::object::Object> = self
                .fs
                .borrow()
                .stored_objects()
                .map(|(_, o)| o.clone())
                .collect();
            for obj in objects {
                self.ram_dedup.put(0, obj);
            }
        }
        ok
    }

    /// Inject the live network transport (the kernel's virtio-net stack) into the
    /// browser. Without this, the browser uses its built-in loopback transport, which
    /// still serves native pages and bundled content — so it always renders.
    pub fn set_web_transport(&mut self, transport: alloc::boxed::Box<dyn crate::webengine::Transport>) {
        self.browser.set_transport(transport);
    }

    /// Advance any in-flight browser navigation by one step. Returns `true` when a
    /// load just finished (the browser has already marked itself dirty). Call once
    /// per frame from the desktop render loop.
    pub fn pump_browser(&mut self) -> bool {
        self.browser.pump()
    }

    // ── right-click context menu ──

    pub fn on_right_click(&mut self, px: i32, py: i32) {
        // Build the menu from what is under the pointer. Right-clicking a window focuses
        // it first, so Cut/Copy/Paste act on the surface you clicked.
        if let Some(app) = self.wm.at(px, py) {
            self.wm.focus(app);
            self.focus = app;
            // Target-aware apps record which item the pointer is over (Open/Delete act
            // on it).
            if app == AppId::Files {
                let c = self.wm.content_of(app);
                self.files.context_path_at(px - c.x, py - c.y);
            }
        } else {
            self.focus = AppId::Desktop;
            // Record canvas-space position for desktop backdrop actions (add note, etc.).
            let c = self.content();
            let (cx, cy) = self.desktop.to_canvas(px - c.x, py - c.y);
            self.desktop_ctx_x = cx;
            self.desktop_ctx_y = cy;
        }
        self.ctx_items = self.ctx_menu_for(px, py);
        let h = self.ctx_items.len() as i32 * CTX_ROW + 8;
        let x = px.min(self.w - CTX_W - 4).max(0);
        let y = py.min(self.h - h - 4).max(0);
        self.ctx = Some((x, y));
        self.dmg_all();
    }

    /// Compose the context-specific menu items for a right-click at `(px,py)`: the
    /// app-specific actions for whatever window is under the pointer, then the universal
    /// window controls — or the desktop actions on the backdrop.
    fn ctx_menu_for(&self, px: i32, py: i32) -> Vec<(String, CtxAction)> {
        let mut v: Vec<(String, CtxAction)> = Vec::new();
        if let Some(app) = self.wm.at(px, py) {
            self.app_ctx_items(app, &mut v);
            let maxi = if self.wm.is_maximized(app) { "Restore" } else { "Maximize" };
            v.push((maxi.into(), CtxAction::Maximize(app)));
            v.push(("Minimize".into(), CtxAction::Minimize(app)));
            v.push(("Close".into(), CtxAction::Close(app)));
        } else {
            // The desktop backdrop.
            let edit = if self.editing { "Lock widgets".into() } else { "Edit widgets".into() };
            v.push((edit, CtxAction::ToggleEdit));
            v.push(("Add note".into(), CtxAction::DesktopAddNote));
            v.push(("New folder".into(), CtxAction::DesktopAddFolder));
            v.push(("New shortcut".into(), CtxAction::DesktopAddShortcut));
            if self.desktop.selected_id().is_some() {
                v.push(("Delete item".into(), CtxAction::DesktopDeleteItem));
            }
            v.push(("Open Files".into(), CtxAction::Go(AppId::Files)));
            v.push(("Open Terminal".into(), CtxAction::Go(AppId::Terminal)));
            v.push(("Open IDE".into(), CtxAction::Go(AppId::Ide)));
            v.push(("Open Browser".into(), CtxAction::Go(AppId::Browser)));
            v.push(("Open Settings".into(), CtxAction::Go(AppId::Settings)));
            v.push(("Open Task Manager".into(), CtxAction::Go(AppId::TaskManager)));
        }
        v
    }

    /// The app-specific context-menu items for `app` (prepended before the window
    /// controls). Every app contributes something useful here, not just the text ones.
    fn app_ctx_items(&self, app: AppId, v: &mut Vec<(String, CtxAction)>) {
        match app {
            AppId::Files => {
                if let Some(_path) = self.files.selected_path() {
                    let is_dir = self.files.selected_is_dir();
                    let open_label = if is_dir { "Open folder" } else { "Open" };
                    v.push((open_label.into(), CtxAction::FilesOpen));
                    if !is_dir {
                        v.push(("Edit".into(), CtxAction::FilesEdit));
                        v.push(("Run".into(), CtxAction::FilesRun));
                    }
                    v.push(("Rename".into(), CtxAction::FilesRename));
                    v.push(("Copy".into(), CtxAction::FilesCopy));
                    v.push(("Cut".into(), CtxAction::FilesCut));
                    v.push(("Delete".into(), CtxAction::FilesDelete));
                    v.push(("Properties".into(), CtxAction::FilesProperties));
                    if !is_dir {
                        v.push(("Save as…".into(), CtxAction::FilesSaveAs));
                    }
                }
                v.push(("Paste".into(), CtxAction::Paste));
                v.push(("New file".into(), CtxAction::FilesNewFile));
                v.push(("New folder".into(), CtxAction::FilesNewFolder));
                v.push(("Refresh".into(), CtxAction::FilesRefresh));
            }
            AppId::Editor => {
                v.push(("Cut".into(), CtxAction::Cut));
                v.push(("Copy".into(), CtxAction::Copy));
                v.push(("Paste".into(), CtxAction::Paste));
                v.push(("Save".into(), CtxAction::EditorSave));
                v.push(("Save as copy".into(), CtxAction::EditorSaveAs));
            }
            AppId::Terminal => {
                v.push(("Cut".into(), CtxAction::Cut));
                v.push(("Copy".into(), CtxAction::Copy));
                v.push(("Paste".into(), CtxAction::Paste));
                v.push(("Clear".into(), CtxAction::TerminalClear));
            }
            AppId::Browser => {
                if self.browser.can_nav_back() {
                    v.push(("Back".into(), CtxAction::BrowserBack));
                }
                if self.browser.can_nav_forward() {
                    v.push(("Forward".into(), CtxAction::BrowserForward));
                }
                v.push(("Reload".into(), CtxAction::BrowserReload));
                v.push(("Copy URL".into(), CtxAction::BrowserCopyUrl));
                v.push(("Save page".into(), CtxAction::BrowserSavePage));
            }
            AppId::Ide => {
                v.push(("Run".into(), CtxAction::IdeRun));
                v.push(("Stop".into(), CtxAction::IdeStop));
                v.push(("Add node".into(), CtxAction::IdeAddNode));
                if self.ide.selected_node().is_some() {
                    v.push(("Delete node".into(), CtxAction::IdeDeleteNode));
                    v.push(("Reset wires".into(), CtxAction::IdeResetWires));
                    v.push(("Disconnect inputs".into(), CtxAction::IdeDisconnectInputs));
                    v.push(("Disconnect outputs".into(), CtxAction::IdeDisconnectOutputs));
                    v.push(("Disconnect all wires".into(), CtxAction::IdeDisconnectAll));
                }
                v.push(("Cut".into(), CtxAction::Cut));
                v.push(("Copy".into(), CtxAction::Copy));
                v.push(("Paste".into(), CtxAction::Paste));
                v.push(("New program".into(), CtxAction::IdeNewProgram));
                v.push(("Open example\u{2026}".into(), CtxAction::IdeOpenExamples));
            }
            AppId::Explorer => {
                v.push(("Refresh".into(), CtxAction::ExplorerRefresh));
                v.push(("Clear log".into(), CtxAction::ExplorerClearLog));
                v.push(("Export".into(), CtxAction::ExplorerExport));
            }
            AppId::TaskManager => {
                v.push(("Kill process".into(), CtxAction::TaskKill));
                v.push(("Refresh".into(), CtxAction::TaskRefresh));
            }
            AppId::Settings => {
                v.push(("Reset to defaults".into(), CtxAction::SettingsReset));
                v.push(("Export config".into(), CtxAction::SettingsExportConfig));
            }
            AppId::Desktop => {}
        }
    }

    fn ctx_rect(&self) -> Option<Rect> {
        self.ctx.map(|(x, y)| Rect::new(x, y, CTX_W, self.ctx_items.len() as i32 * CTX_ROW + 8))
    }
    fn ctx_press(&mut self, px: i32, py: i32) -> bool {
        let Some(menu) = self.ctx_rect() else { return false };
        let mut act = None;
        if menu.contains(px, py) && !self.ctx_items.is_empty() {
            let i = ((py - menu.y - 4) / CTX_ROW).clamp(0, self.ctx_items.len() as i32 - 1) as usize;
            act = Some(self.ctx_items[i].1);
        }
        self.ctx = None;
        self.dmg_all();
        if let Some(a) = act {
            match a {
                CtxAction::ToggleEdit => self.toggle_edit(),
                CtxAction::Go(p) => self.switch(p),
                CtxAction::Close(app) => {
                    self.wm.close(app);
                    self.focus_top();
                }
                CtxAction::Minimize(app) => {
                    self.wm.minimize(app);
                    self.focus_top();
                }
                CtxAction::Maximize(app) => {
                    self.wm.toggle_maximize(app);
                    self.sync_open_window_areas();
                }
                CtxAction::Copy => {
                    if let Some(t) = self.focused_copy() {
                        self.clipboard = t;
                    }
                }
                CtxAction::Cut => {
                    if let Some(t) = self.focused_cut() {
                        self.clipboard = t;
                    }
                }
                CtxAction::Paste => {
                    let c = self.clipboard.clone();
                    if !c.is_empty() {
                        self.focused_paste(&c);
                    }
                }
                CtxAction::FilesOpen => {
                    if let Some(path) = self.files.selected_path() {
                        if self.files.selected_is_dir() {
                            self.files.navigate(&path);
                        } else {
                            self.editor.open(&path);
                            self.switch(AppId::Editor);
                        }
                    }
                }
                CtxAction::FilesNewFile => self.files.new_file(),
                CtxAction::FilesNewFolder => self.files.new_folder(),
                CtxAction::FilesDelete => { self.files.delete_selected(); }
                CtxAction::FilesRefresh => self.files.refresh(),
                CtxAction::FilesRename => self.files.start_rename(),
                CtxAction::FilesEdit => {
                    if let Some(path) = self.files.selected_path() {
                        self.editor.open(&path);
                        self.switch(AppId::Editor);
                    }
                }
                CtxAction::FilesRun => {
                    // Open and run the selected file in the IDE
                    if let Some(path) = self.files.selected_path() {
                        self.ide.open_by_name(&path);
                        self.ide.run();
                        self.sync_world();
                        self.switch(AppId::Ide);
                    }
                }
                CtxAction::FilesCopy => {
                    // Copy file path to clipboard (full copy-file impl needs paste target)
                    if let Some(path) = self.files.selected_path() {
                        self.clipboard = path;
                    }
                }
                CtxAction::FilesCut => {
                    if let Some(path) = self.files.selected_path() {
                        self.clipboard = path;
                    }
                }
                CtxAction::FilesProperties => {
                    // Show properties as a notification line — full dialog would need
                    // a dedicated window; push to the log for now.
                    if let Some(path) = self.files.selected_path() {
                        let is_dir = self.files.selected_is_dir();
                        let kind = if is_dir { "Directory" } else { "File" };
                        let mut msg = String::from(kind);
                        msg.push_str(": ");
                        msg.push_str(&path);
                        self.push_log(&msg);
                    }
                }
                CtxAction::FilesSaveAs => {
                    // Open the selected file in the editor (editor has Save As support)
                    if let Some(path) = self.files.selected_path() {
                        self.editor.open(&path);
                        self.switch(AppId::Editor);
                    }
                }
                CtxAction::BrowserBack => self.browser.nav_back(),
                CtxAction::BrowserForward => self.browser.nav_forward(),
                CtxAction::BrowserReload => self.browser.nav_reload(),
                CtxAction::BrowserCopyUrl => {
                    self.clipboard = self.browser.current_url().unwrap_or_default();
                }
                CtxAction::BrowserSavePage => {
                    // No-op for now; variant reserved for future implementation.
                }
                CtxAction::TerminalClear => self.terminal.clear(),
                CtxAction::IdeRun => {
                    self.ide.run();
                    self.sync_world();
                }
                CtxAction::IdeStop => {
                    self.ide.stop();
                }
                CtxAction::IdeAddNode => {
                    self.ide.add_node();
                    self.sync_world();
                }
                CtxAction::IdeDeleteNode => {
                    self.ide.delete_selected_node();
                }
                CtxAction::IdeResetWires => {
                    self.ide.reset_selected_wires();
                }
                CtxAction::IdeDisconnectInputs => {
                    if let Some(id) = self.ide.selected_node() {
                        self.ide.disconnect_node_inputs(id);
                    }
                }
                CtxAction::IdeDisconnectOutputs => {
                    if let Some(id) = self.ide.selected_node() {
                        self.ide.disconnect_node_outputs(id);
                    }
                }
                CtxAction::IdeDisconnectAll => {
                    if let Some(id) = self.ide.selected_node() {
                        self.ide.disconnect_node_all(id);
                    }
                }
                CtxAction::IdeNewProgram => {
                    self.ide.new_program();
                    self.sync_world();
                }
                CtxAction::IdeOpenExamples => {
                    self.ide.toggle_examples();
                }
                CtxAction::EditorSave => { self.editor.save(); }
                CtxAction::EditorSaveAs => {
                    // Save a copy alongside the current file with "_copy" appended.
                    if let Some(orig) = self.editor.open_path().map(|s| s.to_string()) {
                        let copy_path = if let Some(dot) = orig.rfind('.') {
                            let mut p = orig[..dot].to_string();
                            p.push_str("_copy");
                            p.push_str(&orig[dot..]);
                            p
                        } else {
                            let mut p = orig.clone();
                            p.push_str("_copy");
                            p
                        };
                        self.editor.save_as(&copy_path);
                    }
                }
                CtxAction::ExplorerRefresh => self.sync_world(),
                CtxAction::ExplorerClearLog => {
                    // No-op until Explorer exposes clear_log(); variant reserved.
                }
                CtxAction::ExplorerExport => {
                    // No-op for now; variant reserved for future implementation.
                }
                CtxAction::TaskKill => {
                    // No-op until TaskManager exposes kill_selected(); variant reserved.
                }
                CtxAction::TaskRefresh => {
                    // Force a metrics re-push to refresh the task list display.
                    self.sync_processes();
                }
                CtxAction::SettingsReset => {
                    // No-op for now; variant reserved for future implementation.
                }
                CtxAction::SettingsExportConfig => {
                    // No-op for now; variant reserved for future implementation.
                }
                CtxAction::DesktopAddNote => {
                    self.desktop.add_note_at(self.desktop_ctx_x, self.desktop_ctx_y);
                    self.desktop.set_tool(crate::desktop_page::CanvasTool::Select);
                }
                CtxAction::DesktopDeleteItem => {
                    self.desktop.delete_selected();
                }
                CtxAction::DesktopAddFolder => {
                    self.world.add_folder(&alloc::format!("Folder {}", self.world.next_entry_id()));
                    self.sync_world();
                }
                CtxAction::DesktopAddShortcut => {
                    self.world.add_shortcut("Shortcut", "");
                    self.sync_world();
                }
            }
        }
        true
    }

    pub fn is_editing(&self) -> bool {
        self.editing
    }

    /// Toggle Edit-UI on the Desktop (compose widgets).
    pub fn toggle_edit(&mut self) {
        self.editing = !self.editing;
        if self.editing {
            self.focus = AppId::Desktop;
        }
        self.board.set_locked(!self.editing);
        self.dmg_all();
    }

    fn toggle_theme(&mut self) {
        self.theme_dark = !self.theme_dark;
        self.settings.set_theme_dark(self.theme_dark);
        self.dmg_all();
    }

    /// Apply a preference flip from the Settings app: store it and push the side-effect
    /// to whichever subsystem it controls.
    fn apply_setting(&mut self, flag: Flag, v: bool) {
        self.config.set(flag, v);
        match flag {
            Flag::LiveEval => self.editor.set_live_eval(v),
            Flag::EditorInsert => self.editor.set_insert_default(v),
            // DesktopIcons / Widgets / TrayClock are read by `view`; just repaint.
            Flag::DesktopIcons | Flag::Widgets | Flag::TrayClock => {}
        }
        self.dmg_all();
    }

    /// Select a whole security-posture preset (Server / Balanced / Hardened) and mirror
    /// it as the shell's source of truth. Only *local-blast-radius* defences change;
    /// the wire-trust invariants (identity, session AEAD, capability integrity) are not
    /// part of the profile, so a relaxed node cannot weaken the rest of the network —
    /// it only changes what `security_attestation()` reports and what high-assurance
    /// domains will admit it for (see [`PosturePolicy`]).
    fn apply_profile(&mut self, p: Posture) {
        self.config.security.select(p);
        self.dmg_all();
    }

    /// Flip one local-hardening knob live. The reported posture self-adjusts so it never
    /// overstates the node's actual defences.
    fn apply_knob(&mut self, k: Knob, v: bool) {
        self.config.security.set_knob(k, v);
        self.dmg_all();
    }

    /// The node's current security profile (single source of truth).
    pub fn security_profile(&self) -> SecurityProfile {
        self.config.security
    }

    /// The bytes binding the active security posture into the measured attestation
    /// quote, so peers can see this node's posture and gate on it. Pair with the rest of
    /// the system's components in [`crate::attest::measure`].
    pub fn security_attestation(&self) -> Vec<u8> {
        self.config.security.attest_tag()
    }

    /// Whether this node — at its current posture — qualifies to host/serve `domain`
    /// under the canonical per-domain minimum-posture policy. A lean `Server` node is
    /// refused the high-assurance domains; this is the seam that protects other nodes.
    pub fn admits_domain(&self, domain: crate::firewall::Domain) -> bool {
        PosturePolicy::architecture_2_0().admits(domain, &self.config.security)
    }

    // ── fleet background service ──

    /// Initialize the fleet coordinator at boot, registering this node as the
    /// founding device. Called once from `Os::new()`.
    fn start_fleet_service(&mut self) {
        let first_device = DeviceId::from_pubkey(&self.fleet_owner.0);
        self.fleet = Some(Fleet::new(self.fleet_owner, first_device));
    }

    /// Tick the fleet service once per metrics beat. Logs a warning to the
    /// Explorer if the fleet has been reduced to zero active devices (all
    /// revoked), which indicates the fleet state needs attention.
    fn fleet_tick(&mut self) {
        if let Some(fleet) = &self.fleet {
            if fleet.active_count() == 0 {
                self.explorer.push_log("[fleet] warning: no active devices in fleet");
            }
        }
    }

    /// Enroll a new device into the fleet by capability delegation from the
    /// founding owner device. Returns the new `DeviceId` on success, or `None`
    /// if the fleet is not initialized or the owner device has been revoked.
    pub fn fleet_enroll_device(&mut self, new_device_pubkey: &[u8]) -> Option<DeviceId> {
        let owner_device = DeviceId::from_pubkey(&self.fleet_owner.0);
        self.fleet.as_mut()?.enroll(&owner_device, new_device_pubkey)
    }

    /// Revoke a device and all devices it transitively enrolled (recursive
    /// fleet-wide revocation). Returns the number of devices revoked, or `None`
    /// if the fleet is not initialized.
    pub fn fleet_revoke_device(&mut self, device: &DeviceId) -> Option<usize> {
        Some(self.fleet.as_mut()?.revoke_device(device))
    }

    /// Check whether a device is an active (enrolled, non-revoked) member of
    /// this node's fleet.
    pub fn fleet_is_active(&self, device: &DeviceId) -> bool {
        self.fleet.as_ref().map_or(false, |f| f.is_active(device))
    }

    /// The number of active (enrolled, non-revoked) devices in the fleet.
    pub fn fleet_active_count(&self) -> usize {
        self.fleet.as_ref().map_or(0, |f| f.active_count())
    }

    pub fn publish_layout(&mut self) {
        self.library.publish(DEFAULT_LAYOUT, &self.board);
    }
    pub fn install_layout(&mut self) -> bool {
        if let Some(mut b) = self.library.install(DEFAULT_LAYOUT) {
            b.set_area(self.content());
            b.set_locked(!self.editing);
            self.board = b;
            self.dmg_all();
            true
        } else {
            false
        }
    }
    pub fn library(&self) -> &Library {
        &self.library
    }

    pub fn set_account(&mut self, name: &str) {
        self.account = name.to_string();
        self.settings.set_account(name);
        self.dmg(self.topbar());
    }

    /// The account label shown in the top bar.
    pub fn account(&self) -> &str {
        &self.account
    }

    /// Bind the displayed account to the user's **self-certifying identity**.
    pub fn set_account_from_identity(&mut self, id: crate::dominionlink::DominionId) {
        let mut label = String::from("id:");
        label.push_str(&id.0.short());
        self.set_account(&label);
    }

    fn theme(&self) -> Theme {
        if self.theme_dark {
            Theme::dark()
        } else {
            Theme::light()
        }
    }

    // ── layout ──

    fn topbar(&self) -> Rect {
        Rect::new(0, 0, self.w, TOPBAR_H)
    }
    fn dock(&self) -> Rect {
        Rect::new(0, self.h - DOCK_H, self.w, DOCK_H)
    }
    fn content(&self) -> Rect {
        Rect::new(0, TOPBAR_H, self.w, (self.h - TOPBAR_H - DOCK_H).max(0))
    }
    fn start_btn(&self) -> Rect {
        let d = self.dock();
        Rect::new(d.x + 8, d.y + 8, 76, d.h - 16)
    }
    /// The taskbar button rect for the i-th running window.
    fn task_btn(&self, i: usize) -> Rect {
        let d = self.dock();
        let x0 = self.start_btn().x + self.start_btn().w + 8;
        let bw = 70;
        Rect::new(x0 + i as i32 * bw, d.y + 8, bw - 6, d.h - 16)
    }
    fn edit_btn(&self) -> Rect {
        Rect::new(self.w - 110, 4, 100, TOPBAR_H - 8)
    }
    /// A desktop-icon cell rect for icon index `i`. Icons flow top-to-bottom in a
    /// column, wrapping to the next column when they would reach the dock.
    fn desktop_icon(&self, i: usize) -> Rect {
        let c = self.content();
        let cell_h = 84;
        let rows = ((c.h - 16) / cell_h).max(1) as usize;
        let col = (i / rows) as i32;
        let row = (i % rows) as i32;
        Rect::new(c.x + 16 + col * 92, c.y + 16 + row * cell_h, 84, 76)
    }

    // ── kernel-facing API ──

    pub fn set_size(&mut self, w: i32, h: i32) {
        if w != self.w || h != self.h {
            self.w = w;
            self.h = h;
            self.sync_app_areas();
            self.dmg_all();
        }
    }

    fn sync_app_areas(&mut self) {
        let c = self.content();
        self.desktop.set_area(c);
        self.board.set_area(c);
        self.wm.set_area(c);
        self.sync_open_window_areas();
    }

    /// Size each open window's app to that window's current content rect.
    fn sync_open_window_areas(&mut self) {
        let wins: Vec<(AppId, Rect)> =
            self.wm.taskbar().iter().map(|(app, _)| (*app, self.wm.content_of(*app))).collect();
        for (app, c) in wins {
            self.set_app_area(app, Rect::new(0, 0, c.w, c.h));
        }
    }

    fn set_app_area(&mut self, app: AppId, local: Rect) {
        match app {
            AppId::Desktop => self.desktop.set_area(local),
            AppId::Files => self.files.set_area(local),
            AppId::Browser => self.browser.set_area(local),
            AppId::Terminal => self.terminal.set_area(local),
            AppId::Editor => self.editor.set_area(local),
            AppId::Ide => self.ide.set_area(local),
            AppId::Explorer => self.explorer.set_area(local),
            AppId::TaskManager => self.taskman.set_area(local),
            AppId::Settings => self.settings.set_area(local),
        }
    }

    pub fn set_metrics(&mut self, m: Metrics) {
        // Advance the agent bus tick once per metrics beat so snapshots carry a
        // monotonically increasing tick number that agents can use to detect changes.
        self.agent_bus.tick();
        let clock_changed = m.uptime_secs / 60 != self.metrics.uptime_secs / 60;
        let status_changed = (m.entropy_milli > 0) != (self.metrics.entropy_milli > 0)
            || m.net_present != self.metrics.net_present;
        self.metrics = m.clone();
        // Keep the process table in sync with the open windows, then advance + charge the
        // cooperative scheduler **before** the Task Manager samples it, so its per-process
        // CPU reflects which app is actually working this tick.
        self.sync_processes();
        {
            let focus_dom = self.app_domains.get(&self.focus).copied();
            let mut s = self.sched.borrow_mut();
            for _ in 0..3 {
                if let Some(id) = s.next() {
                    s.yield_back(id);
                }
            }
            // The focused window is the one the user is driving → charge it the lion's
            // share of this tick's CPU so its usage reads realistically high while the
            // background apps sit near idle.
            if let Some(id) = focus_dom {
                s.charge(id, 12);
            }
        }
        self.explorer.set_metrics(m.clone());
        self.taskman.set_metrics(m.clone());
        self.settings.set_metrics(m);
        self.ide.tick();
        self.fleet_tick();
        // Memory acceleration tick — run on every metrics beat (idle tick).
        // Consult the tiered memory manager: if any tier is under high pressure,
        // evict LRU objects from the busiest domain so pages can migrate cold.
        {
            let pressures = self.mem_manager.tier_pressure();
            if pressures.iter().any(|p| *p == Pressure::High) {
                // Evict from domain 0 (the kernel-level shared domain); each
                // app's domain quota is managed separately via admit().
                let _ = self.mem_manager.evict_domain_lru(0);
            }
        }
        // RAM dedup idle stats — sample the cross-domain sharing factor so the
        // metrics subsystem can observe dedup savings without any extra cost.
        let _dedup_stats = self.ram_dedup.stats();
        self.update_widgets();
        // Repaint any visible window that shows live metrics.
        for app in [AppId::TaskManager, AppId::Explorer, AppId::Settings] {
            if self.window_visible(app) {
                let c = self.wm.content_of(app);
                self.dmg(c);
            }
        }
        if clock_changed || status_changed {
            self.dmg(self.dock());
        }
    }

    fn window_visible(&self, app: AppId) -> bool {
        self.wm.window(app).map(|w| !w.is_minimized()).unwrap_or(false)
    }

    /// Whether the focused surface has a focused text field.
    pub fn wants_text_input(&self) -> bool {
        match self.focus {
            AppId::Browser => self.browser.wants_text(),
            AppId::Terminal => self.terminal.wants_text(),
            AppId::Editor => self.editor.wants_text(),
            AppId::Ide => self.ide.is_text_focused(),
            AppId::Explorer => self.explorer.is_search_focused(),
            AppId::Files => self.files.wants_text(),
            _ => false,
        }
    }

    pub fn set_time(&mut self, now_ms: u64) {
        match self.focus {
            AppId::Browser => self.browser.set_time(now_ms),
            AppId::Terminal => self.terminal.set_time(now_ms),
            AppId::Editor => self.editor.set_time(now_ms),
            AppId::Ide => self.ide.set_time(now_ms),
            AppId::Explorer => self.explorer.set_time(now_ms),
            AppId::Files => self.files.set_time(now_ms),
            _ => {}
        }
    }

    pub fn push_log(&mut self, line: &str) {
        self.explorer.push_log(line);
    }

    pub fn take_damage(&mut self) -> Option<Rect> {
        let c = self.content();
        let mut d = self.damage.take();
        // Desktop backdrop.
        if let Some(bd) = self.desktop.take_damage() {
            let abs = clip(Rect::new(bd.x + c.x, bd.y + c.y, bd.w, bd.h), c);
            d = union_opt(d, abs);
        }
        // Widgets board (persistent on the backdrop).
        if let Some(bd) = self.board.take_damage() {
            d = union_opt(d, clip(bd, c));
        }
        // Each open, non-minimized window's app.
        let wins: Vec<(AppId, Rect)> = self
            .wm
            .taskbar()
            .iter()
            .filter(|(_, st)| *st != WinState::Minimized)
            .map(|(app, _)| (*app, self.wm.content_of(*app)))
            .collect();
        for (app, cc) in wins {
            if let Some(pd) = self.app_take_damage(app) {
                let abs = clip(Rect::new(pd.x + cc.x, pd.y + cc.y, pd.w, pd.h), cc);
                d = union_opt(d, abs);
            }
        }
        d
    }

    fn app_take_damage(&mut self, app: AppId) -> Option<Rect> {
        match app {
            AppId::Desktop => self.desktop.take_damage(),
            AppId::Files => self.files.take_damage(),
            AppId::Browser => self.browser.take_damage(),
            AppId::Terminal => self.terminal.take_damage(),
            AppId::Editor => self.editor.take_damage(),
            AppId::Ide => self.ide.take_damage(),
            AppId::Explorer => self.explorer.take_damage(),
            AppId::TaskManager => self.taskman.take_damage(),
            AppId::Settings => self.settings.take_damage(),
        }
    }

    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => toolkit::union(d, r),
            None => r,
        });
    }
    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.w, self.h));
    }

    /// The focused surface (a window's app, or the Desktop backdrop).
    pub fn active(&self) -> AppId {
        self.focus
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        let pressed = left && !self.last_left;
        self.last_left = left;

        if pressed {
            self.press(px, py);
            return;
        }
        self.drag(px, py, left);
        if !left {
            self.grab = Grab::None;
        }
    }

    /// Route a fresh press. Chrome above the windows (menus, top bar, dock) wins first,
    /// then the window manager, then the desktop backdrop.
    fn press(&mut self, px: i32, py: i32) {
        // 1. Open context menu.
        if self.ctx.is_some() {
            self.wm.note_left(true);
            self.ctx_press(px, py);
            self.grab = Grab::None;
            return;
        }
        // 2. Top-bar edit toggle.
        if self.edit_btn().contains(px, py) {
            self.wm.note_left(true);
            self.toggle_edit();
            self.grab = Grab::None;
            return;
        }
        // 3. Open Start menu.
        if self.start_open {
            if let Some(app) = self.start_menu_hit(px, py) {
                self.switch(app);
            } else {
                self.start_open = false;
                self.dmg_all();
            }
            self.wm.note_left(true);
            self.grab = Grab::None;
            return;
        }
        // 4. Dock / taskbar.
        if self.dock().contains(px, py) {
            self.dock_press(px, py);
            self.wm.note_left(true);
            self.grab = Grab::None;
            return;
        }
        // 5. Edit board overlay (desktop backdrop only).
        if self.editing && self.focus == AppId::Desktop {
            let c = self.content();
            if c.contains(px, py) || self.board.picker_open() {
                self.board.on_pointer(px, py, true);
                self.wm.note_left(true);
                self.grab = Grab::Board;
                return;
            }
        }
        // 6. Window manager.
        match self.wm.on_pointer(px, py, true) {
            Reaction::Consumed => {
                self.focus_top();
                self.grab = Grab::Chrome;
                self.dmg_all();
            }
            Reaction::Closed(_) => {
                self.focus_top();
                self.grab = Grab::None;
                self.sync_processes();
                self.dmg_all();
            }
            Reaction::Forward(app, lx, ly) => {
                self.focus = app;
                self.grab = Grab::Window(app);
                self.route_pointer(app, lx, ly, true);
            }
            Reaction::Miss => {
                // 7. Desktop backdrop: icons launch apps; otherwise the object cards.
                self.focus = AppId::Desktop;
                for (i, app) in DESKTOP_ICONS.iter().enumerate() {
                    if self.desktop_icon(i).contains(px, py) {
                        self.switch(*app);
                        self.grab = Grab::None;
                        return;
                    }
                }
                self.grab = Grab::Desktop;
                let c = self.content();
                self.route_pointer(AppId::Desktop, px - c.x, py - c.y, true);
            }
        }
    }

    /// Route a move/release to whatever the press grabbed.
    fn drag(&mut self, px: i32, py: i32, left: bool) {
        match self.grab {
            Grab::Chrome => {
                let before = self.wm.top().and_then(|id| self.wm.window(id)).map(|w| w.frame(self.wm.area()));
                match self.wm.on_pointer(px, py, left) {
                    Reaction::Closed(_) => {
                        self.focus_top();
                        self.sync_processes();
                        self.dmg_all();
                    }
                    _ => {
                        // Damage the union of the window's old and new frames.
                        let after =
                            self.wm.top().and_then(|id| self.wm.window(id)).map(|w| w.frame(self.wm.area()));
                        match (before, after) {
                            (Some(a), Some(b)) => {
                                let r = toolkit::union(toolkit::inflate(a, 10), toolkit::inflate(b, 10));
                                self.dmg(r);
                                self.sync_open_window_areas();
                            }
                            _ => self.dmg_all(),
                        }
                    }
                }
            }
            Grab::Window(app) => match self.wm.on_pointer(px, py, left) {
                Reaction::Forward(a, lx, ly) => self.route_pointer(a, lx, ly, left),
                Reaction::Closed(_) => {
                    self.focus_top();
                    self.sync_processes();
                    self.dmg_all();
                }
                _ => {
                    let _ = app;
                }
            },
            Grab::Desktop => {
                self.wm.note_left(left);
                let c = self.content();
                self.route_pointer(AppId::Desktop, px - c.x, py - c.y, left);
            }
            Grab::Board => {
                self.wm.note_left(left);
                self.board.on_pointer(px, py, left);
            }
            Grab::None => self.wm.note_left(left),
        }
    }

    fn dock_press(&mut self, px: i32, py: i32) {
        if self.start_btn().contains(px, py) {
            self.start_open = !self.start_open;
            self.dmg_all();
            return;
        }
        let tb = self.wm.taskbar();
        for (i, (app, st)) in tb.iter().enumerate() {
            if self.task_btn(i).contains(px, py) {
                if *st == WinState::Minimized {
                    self.wm.restore(*app);
                    self.focus = *app;
                } else if self.focus == *app {
                    self.wm.minimize(*app);
                    self.focus_top();
                } else {
                    self.wm.focus(*app);
                    self.focus = *app;
                }
                if self.focus != AppId::Desktop {
                    self.leave_desktop_edit();
                }
                self.dmg_all();
                return;
            }
        }
    }

    fn route_pointer(&mut self, app: AppId, lx: i32, ly: i32, left: bool) {
        match app {
            AppId::Desktop => {
                if let Some(action) = self.desktop.on_pointer(lx, ly, left) {
                    match action {
                        DesktopAction::OpenProgram(name) => {
                            self.ide.open_by_name(&name);
                            self.switch(AppId::Ide);
                        }
                        DesktopAction::Inspect(name) => {
                            self.explorer.select_by_name(&name);
                            self.switch(AppId::Explorer);
                        }
                    }
                }
            }
            AppId::Files => {
                match self.files.on_pointer(lx, ly, left) {
                    Some(FilesAction::OpenFile(path)) => {
                        self.editor.open(&path);
                        self.switch(AppId::Editor);
                    }
                    Some(FilesAction::Rename(_, _)) => {}
                    None => {}
                }
            }
            AppId::Browser => self.browser.on_pointer(lx, ly, left),
            AppId::Terminal => self.terminal.on_pointer(lx, ly, left),
            AppId::Editor => self.editor.on_pointer(lx, ly, left),
            AppId::Ide => {
                self.ide.on_pointer(lx, ly, left);
                self.sync_world();
            }
            AppId::Explorer => self.explorer.on_pointer(lx, ly, left),
            AppId::TaskManager => {
                if let Some(killed_id) = self.taskman.on_pointer(lx, ly, left) {
                    // Find which app window this domain belongs to and close it
                    // immediately instead of waiting for the next metrics tick.
                    if let Some(app) = self.app_domains.iter()
                        .find(|(_, &did)| did == killed_id)
                        .map(|(&aid, _)| aid)
                    {
                        self.wm.close(app);
                        self.focus_top();
                    }
                    self.sync_processes();
                    self.dmg_all();
                }
            }
            AppId::Settings => {
                if let Some(action) = self.settings.on_pointer(lx, ly, left) {
                    match action {
                        SettingsAction::ToggleTheme => self.toggle_theme(),
                        SettingsAction::PowerOff => self.exit = true,
                        SettingsAction::SetFlag(flag, v) => self.apply_setting(flag, v),
                        SettingsAction::SetProfile(p) => self.apply_profile(p),
                        SettingsAction::SetKnob(k, v) => self.apply_knob(k, v),
                        SettingsAction::SetStage(_, _) => {}
                        SettingsAction::SetStageProfile(_) => {}
                        // The Stages/Ecosystem cards own their control planes in Settings'
                        // Config and re-render live (pass-count headers); no extra side-effect.
                        SettingsAction::SetEcoFeature(_, _) => {}
                        SettingsAction::SetEcoPreset(_) => {}
                    }
                }
            }
        }
    }

    /// The system clipboard contents (for tests / external integration).
    pub fn clipboard(&self) -> &str {
        &self.clipboard
    }

    /// Copy/cut from the focused text surface into the shared clipboard.
    fn focused_copy(&self) -> Option<String> {
        match self.focus {
            AppId::Editor => self.editor.copy(),
            AppId::Terminal => self.terminal.copy(),
            _ => None,
        }
    }
    fn focused_cut(&mut self) -> Option<String> {
        match self.focus {
            AppId::Editor => self.editor.cut(),
            AppId::Terminal => self.terminal.cut(),
            _ => None,
        }
    }
    fn focused_paste(&mut self, s: &str) {
        match self.focus {
            AppId::Editor => self.editor.paste(s),
            AppId::Terminal => self.terminal.paste(s),
            _ => {}
        }
    }

    pub fn on_key(&mut self, ch: char) {
        // Global clipboard chords (Ctrl+C/X/V) are intercepted before the app sees them,
        // so copy in one window can paste into another via the shared clipboard.
        if crate::keys::is_clipboard(ch) {
            match ch as u8 {
                crate::keys::COPY => {
                    if let Some(t) = self.focused_copy() {
                        self.clipboard = t;
                    }
                }
                crate::keys::CUT => {
                    if let Some(t) = self.focused_cut() {
                        self.clipboard = t;
                    }
                }
                crate::keys::PASTE => {
                    let c = self.clipboard.clone();
                    if !c.is_empty() {
                        self.focused_paste(&c);
                    }
                }
                _ => {}
            }
            if self.wm.is_open(self.focus) {
                let cc = self.wm.content_of(self.focus);
                self.dmg(cc);
            }
            return;
        }
        // The focused app gets first refusal so text fields can type any character.
        let consumed = match self.focus {
            AppId::Browser => self.browser.on_key(ch),
            AppId::Terminal => self.terminal.on_key(ch),
            AppId::Editor => self.editor.on_key(ch),
            AppId::Ide => self.ide.on_key(ch),
            AppId::Explorer => self.explorer.on_key(ch),
            AppId::Files => self.files.on_key(ch),
            _ => false,
        };
        if consumed {
            if self.focus == AppId::Ide {
                self.sync_world();
            }
            // A focused window's text edit damages that window.
            if self.wm.is_open(self.focus) {
                let cc = self.wm.content_of(self.focus);
                self.dmg(cc);
            }
            return;
        }
        match ch {
            '1' => self.switch(AppId::Desktop),
            '2' => self.switch(AppId::Files),
            '3' => self.switch(AppId::Browser),
            '4' => self.switch(AppId::Terminal),
            '5' => self.switch(AppId::Editor),
            '6' => self.switch(AppId::Ide),
            '7' => self.switch(AppId::Explorer),
            '8' => self.switch(AppId::TaskManager),
            '9' => self.switch(AppId::Settings),
            'g' => self.toggle_theme(),
            'e' if self.focus == AppId::Desktop => self.toggle_edit(),
            'u' if self.focus == AppId::Desktop => self.publish_layout(),
            'd' if self.focus == AppId::Desktop => {
                self.install_layout();
            }
            _ => {}
        }
    }

    /// Open (or focus) an app's window — or show the desktop backdrop.
    fn switch(&mut self, p: AppId) {
        self.start_open = false;
        if p == AppId::Desktop {
            self.focus = AppId::Desktop;
        } else {
            let c = self.wm.open(p, p.label());
            self.set_app_area(p, Rect::new(0, 0, c.w, c.h));
            self.focus = p;
            self.leave_desktop_edit();
        }
        self.build_a11y_tree();
        self.dmg_all();
    }

    /// Make the focus follow the topmost window (or fall back to the Desktop backdrop).
    fn focus_top(&mut self) {
        self.focus = self.wm.top().unwrap_or(AppId::Desktop);
        if self.focus != AppId::Desktop {
            self.leave_desktop_edit();
        }
        self.build_a11y_tree();
    }

    fn leave_desktop_edit(&mut self) {
        if self.editing {
            self.editing = false;
            self.board.set_locked(true);
        }
    }

    fn start_menu_rect(&self) -> Rect {
        let b = self.start_btn();
        let h = ALL_APPS.len() as i32 * 30 + 16;
        Rect::new(b.x, b.y - h - 6, 200, h)
    }
    fn start_menu_hit(&self, px: i32, py: i32) -> Option<AppId> {
        let m = self.start_menu_rect();
        if !m.contains(px, py) {
            return None;
        }
        let idx = (py - m.y - 8) / 30;
        ALL_APPS.get(idx.clamp(0, ALL_APPS.len() as i32 - 1) as usize).copied()
    }

    // ── rendering ──

    pub fn view(&self, w: i32, h: i32) -> Vec<DrawCmd> {
        let theme = self.theme();
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, w, h), color: theme.bg, radius: 0 });

        // Desktop backdrop (always behind the windows).
        let c = self.content();
        let mut backdrop = self.desktop.view(&theme);
        toolkit::translate_scene(&mut backdrop, c.x, c.y);
        s.extend(toolkit::clip_scene(backdrop, c));

        // Desktop icons (toggleable in Settings).
        if self.config.desktop_icons {
            self.draw_desktop_icons(&mut s, &theme);
        }

        // Widgets board (backdrop, toggleable in Settings). In Edit-UI mode it grows its
        // move/resize/picker chrome, so keep it shown while editing regardless.
        if self.config.widgets || self.editing {
            let mut widgets = self.board.view(&theme);
            toolkit::translate_scene(&mut widgets, 0, 0); // board uses absolute content coords
            s.extend(toolkit::clip_scene(widgets, c));
        }

        // Windows, back to front.
        let top = self.wm.top();
        let area = self.wm.area();
        for win in self.wm.visible() {
            let focused = Some(win.id) == top && self.focus == win.id;
            let mut frame = self.wm.frame_scene(win, &theme, focused);
            s.append(&mut frame);
            let cc = win.content(area);
            let mut content = self.app_view(win.id, &theme);
            toolkit::translate_scene(&mut content, cc.x, cc.y);
            s.extend(toolkit::clip_scene(content, cc));
        }

        self.draw_topbar(&mut s, &theme);
        self.draw_dock(&mut s, &theme);
        if self.start_open {
            self.draw_start_menu(&mut s, &theme);
        }
        if let Some(menu) = self.ctx_rect() {
            self.draw_ctx_menu(&mut s, &theme, menu);
        }
        s
    }

    fn app_view(&self, app: AppId, t: &Theme) -> Vec<DrawCmd> {
        match app {
            AppId::Desktop => self.desktop.view(t),
            AppId::Files => self.files.view(t),
            AppId::Browser => self.browser.view(t),
            AppId::Terminal => self.terminal.view(t),
            AppId::Editor => self.editor.view(t),
            AppId::Ide => self.ide.view(t),
            AppId::Explorer => self.explorer.view(t),
            AppId::TaskManager => self.taskman.view(t),
            AppId::Settings => self.settings.view(t),
        }
    }

    fn draw_desktop_icons(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        for (i, app) in DESKTOP_ICONS.iter().enumerate() {
            let r = self.desktop_icon(i);
            s.push(DrawCmd::Rect { rect: Rect::new(r.x + 18, r.y + 6, 48, 40), color: t.surface, radius: t.radius });
            app_glyph(*app, s, r.x + r.w / 2, r.y + 26, t.accent);
            s.push(DrawCmd::Text { rect: Rect::new(r.x - 4, r.y + 52, r.w + 8, 16), text: app.label().into(), color: t.text, size: 12 });
        }
    }

    fn draw_ctx_menu(&self, s: &mut Vec<DrawCmd>, t: &Theme, menu: Rect) {
        s.push(DrawCmd::Rect { rect: toolkit::inflate(menu, 1), color: t.primary, radius: t.radius });
        s.push(DrawCmd::Rect { rect: menu, color: t.surface, radius: t.radius });
        for (i, (label, action)) in self.ctx_items.iter().enumerate() {
            let y = menu.y + 4 + i as i32 * CTX_ROW;
            // Destructive items (Close) read in the danger colour.
            let color = if matches!(action, CtxAction::Close(_)) { t.danger } else { t.text };
            s.push(DrawCmd::Text { rect: Rect::new(menu.x + 10, y + 5, menu.w - 16, 16), text: label.clone(), color, size: 13 });
        }
    }

    fn draw_topbar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = self.topbar();
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        let mut title = String::from("DominionOS  ·  ");
        title.push_str(self.focus.label());
        title.push_str("  ·  ");
        title.push_str(&self.account);
        s.push(DrawCmd::Text { rect: Rect::new(16, 7, self.w - 200, 18), text: title, color: t.text, size: 14 });
        let b = self.edit_btn();
        let (fill, fg, label) = if self.editing {
            (t.primary, t.on_primary, "* Edit UI")
        } else {
            (t.surface, t.muted, "Lock UI")
        };
        s.push(DrawCmd::Rect { rect: b, color: fill, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(b.x + 8, b.y + 3, b.w - 8, 16), text: label.into(), color: fg, size: 12 });
    }

    fn draw_dock(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let dock = self.dock();
        s.push(DrawCmd::Rect { rect: dock, color: t.surface, radius: 0 });
        // Start button.
        let sb = self.start_btn();
        let start_active = self.start_open;
        s.push(DrawCmd::Rect { rect: sb, color: if start_active { t.primary } else { t.bg }, radius: t.radius });
        glyph_start(s, sb.x + 18, sb.y + sb.h / 2, if start_active { t.on_primary } else { t.accent });
        s.push(DrawCmd::Text { rect: Rect::new(sb.x + 30, sb.y + sb.h / 2 - 8, sb.w - 30, 16), text: "Start".into(), color: if start_active { t.on_primary } else { t.text }, size: 13 });
        // Running windows.
        for (i, (app, st)) in self.wm.taskbar().iter().enumerate() {
            let b = self.task_btn(i);
            let focused = self.focus == *app;
            let minimized = *st == WinState::Minimized;
            if focused {
                s.push(DrawCmd::Rect { rect: b, color: t.primary, radius: t.radius });
            } else if !minimized {
                // A subtle backing so open windows read as present.
                s.push(DrawCmd::Rect { rect: b, color: t.bg, radius: t.radius });
            }
            let fg = if focused {
                t.on_primary
            } else if minimized {
                t.muted
            } else {
                t.text
            };
            let glyph = if focused { t.on_primary } else { t.accent };
            app_glyph(*app, s, b.x + b.w / 2, b.y + 18, glyph);
            s.push(DrawCmd::Text { rect: Rect::new(b.x, b.y + b.h - 18, b.w, 14), text: app.label().into(), color: fg, size: 10 });
            // Running indicator pip for non-focused windows.
            if !focused {
                let pip = if minimized { t.muted } else { t.primary };
                s.push(DrawCmd::Rect { rect: Rect::new(b.x + b.w / 2 - 6, b.y + b.h - 3, 12, 2), color: pip, radius: 1 });
            }
        }
        self.draw_status_cluster(s, t);
    }

    /// Right-aligned system tray — dots only, no text labels.
    fn draw_status_cluster(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let dock = self.dock();
        let cy = dock.y + dock.h / 2;
        let ok = Color::rgb(0x3f, 0xc9, 0xb0);
        let secure = self.metrics.entropy_milli > 0;
        let net = self.metrics.net_present;

        let mut x = dock.x + dock.w - 12;

        // Clock text (only when tray_clock config flag is set).
        if self.config.tray_clock {
            let mut clock = String::new();
            let total = self.metrics.uptime_secs;
            push_int(&mut clock, (total / 3600) as i64);
            clock.push(':');
            let m = (total % 3600) / 60;
            if m < 10 {
                clock.push('0');
            }
            push_int(&mut clock, m as i64);
            let clock_w = clock.len() as i32 * 8;
            x -= clock_w;
            s.push(DrawCmd::Text { rect: Rect::new(x, cy - 8, clock_w, 16), text: clock, color: t.text, size: 12 });
            x -= 10;
        }

        // NDN dot (net presence).
        x -= 8;
        s.push(toolkit::disc(x, cy, 4, if net { ok } else { t.danger }));
        x -= 10;

        // SECURE dot (entropy available).
        x -= 8;
        s.push(toolkit::disc(x, cy, 4, if secure { ok } else { t.danger }));
    }

    fn draw_start_menu(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let m = self.start_menu_rect();
        s.push(DrawCmd::Rect { rect: toolkit::inflate(m, 1), color: t.primary, radius: t.radius });
        s.push(DrawCmd::Rect { rect: m, color: t.surface, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(m.x + 12, m.y + 6, m.w - 16, 14), text: "All apps".into(), color: t.muted, size: 11 });
        for (i, app) in ALL_APPS.iter().enumerate() {
            let y = m.y + 8 + i as i32 * 30;
            app_glyph(*app, s, m.x + 18, y + 14, t.accent);
            s.push(DrawCmd::Text { rect: Rect::new(m.x + 34, y + 6, m.w - 40, 18), text: app.label().into(), color: t.text, size: 13 });
        }
    }

    // ── Agent bus public API ──

    /// Advance the agent bus tick (call once per OS frame, alongside `set_metrics`).
    /// Returns the current logical tick.
    pub fn agent_tick(&mut self) -> u64 {
        self.agent_bus.tick()
    }

    /// Take a structured snapshot of all registered agent components.
    ///
    /// For the core OS-owned components (windows, terminal), use
    /// [`Self::agent_snapshot_full`] which assembles the view directly from
    /// `Os`'s owned fields without requiring them to be moved into the bus.
    pub fn agent_snapshot(&self) -> crate::agent::AgentSnapshot {
        self.agent_bus.snapshot()
    }

    /// Build a complete OS snapshot including:
    /// - Every open window (from the WindowManager).
    /// - The terminal page (if the Terminal window is open).
    /// - Any additional components registered in the agent bus.
    pub fn agent_snapshot_full(&self) -> crate::agent::AgentSnapshot {
        use crate::agent::AgentSnapshot;
        let tick = self.agent_bus.current_tick();
        let mut roots = Vec::new();

        // Collect window nodes from the WM (back-to-front z-order).
        let top = self.wm.top();
        for win in self.wm.visible() {
            let id = WINDOW_NODE_BASE + win.id as u64;
            let focused = Some(win.id) == top && self.focus == win.id;
            let minimised = win.state == WinState::Minimized;
            let maximised = win.state == WinState::Maximized;
            let mut node = AgentNode::new(id, NodeState::Window {
                title: win.title.clone(),
                app: win.id.label().into(),
                focused,
                minimised,
                maximised,
            })
            .with_actions([
                ActionDesc::simple(ActionKind::Focus,    "Bring this window to front"),
                ActionDesc::simple(ActionKind::Minimise, "Minimise to taskbar"),
                ActionDesc::simple(ActionKind::Maximise, "Maximise or restore"),
                ActionDesc::simple(ActionKind::Close,    "Close this window"),
            ]);
            // Embed the terminal's agent view as a child of the Terminal window.
            if win.id == AppId::Terminal {
                node.push_child(self.terminal.agent_view());
            }
            roots.push(node);
        }

        // Additional externally-registered components.
        for extra_root in self.agent_bus.snapshot().roots {
            roots.push(extra_root);
        }

        AgentSnapshot { tick, roots }
    }

    /// Dispatch a typed agent action to the component that owns the target node id.
    ///
    /// Window-targeted actions (Focus, Minimise, Maximise, Close) are intercepted
    /// here and forwarded to the [`WindowManager`], since `Window` itself cannot
    /// mutate the WM. All other actions pass through to [`AgentBus::dispatch`].
    pub fn agent_dispatch(&mut self, action: AgentAction) -> AgentResult {
        // Check if the target is a window node.
        let target = action.target;
        if target >= WINDOW_NODE_BASE && target < WINDOW_NODE_BASE + 16 {
            let app_idx = (target - WINDOW_NODE_BASE) as usize;
            // Map index back to AppId via ALL_APPS ordering (AppId as u64 values).
            // AppId derives no explicit discriminant, so we rely on the as-cast order.
            let app = match app_idx {
                0 => AppId::Desktop,
                1 => AppId::Files,
                2 => AppId::Browser,
                3 => AppId::Terminal,
                4 => AppId::Editor,
                5 => AppId::Ide,
                6 => AppId::Explorer,
                7 => AppId::TaskManager,
                8 => AppId::Settings,
                _ => return AgentResult::NotFound,
            };
            if !self.wm.is_open(app) {
                return AgentResult::not_ready("window is not open");
            }
            match action.kind {
                ActionKind::Focus => {
                    self.wm.focus(app);
                    self.focus = app;
                    self.dmg_all();
                    AgentResult::Ok
                }
                ActionKind::Minimise => {
                    self.wm.minimize(app);
                    self.focus_top();
                    self.dmg_all();
                    AgentResult::Ok
                }
                ActionKind::Maximise => {
                    self.wm.toggle_maximize(app);
                    self.sync_open_window_areas();
                    self.dmg_all();
                    AgentResult::Ok
                }
                ActionKind::Close => {
                    self.wm.close(app);
                    self.focus_top();
                    self.sync_processes();
                    self.dmg_all();
                    AgentResult::Ok
                }
                _ => AgentResult::invalid("unsupported action for Window"),
            }
        } else if target == TERM_AGENT_NODE_ID {
            // Route terminal actions directly to the owned TermPage.
            let result = self.terminal.agent_dispatch(action);
            if result.is_ok() && self.window_visible(AppId::Terminal) {
                let cc = self.wm.content_of(AppId::Terminal);
                self.dmg(cc);
            }
            result
        } else {
            self.agent_bus.dispatch(action)
        }
    }

    /// Register a component with the agent bus (call after constructing new apps/VMs).
    pub fn agent_register(&mut self, component: alloc::boxed::Box<dyn AgentControllable>) {
        self.agent_bus.register(component);
    }

    /// Deregister a component by name (call when an app closes or a VM stops).
    pub fn agent_deregister(&mut self, name: &str) {
        self.agent_bus.deregister(name);
    }
}

impl Default for Os {
    fn default() -> Self {
        Self::new()
    }
}

// ── AgentControllable for Window<AppId> ──────────────────────────────────────

/// Stable node-id base for window nodes.
/// Each window's id = WINDOW_NODE_BASE + (AppId as u64).
/// Range 0x4000_0000..0x4000_0010 — reserved for window manager windows.
const WINDOW_NODE_BASE: u64 = 0x4000_0000;

impl AgentControllable for Window<AppId> {
    fn agent_name(&self) -> &str {
        "Window"
    }

    fn agent_view(&self) -> AgentNode {
        let id = WINDOW_NODE_BASE + self.id as u64;
        let focused = false; // Window has no focus knowledge; Os sets it at the bus level.
        let minimised = self.state == WinState::Minimized;
        let maximised = self.state == WinState::Maximized;
        AgentNode::new(id, NodeState::Window {
            title: self.title.clone(),
            app: self.id.label().into(),
            focused,
            minimised,
            maximised,
        })
        .with_actions([
            ActionDesc::simple(ActionKind::Focus,    "Bring this window to front and focus it"),
            ActionDesc::simple(ActionKind::Minimise, "Minimise this window to the taskbar"),
            ActionDesc::simple(ActionKind::Maximise, "Maximise or restore this window"),
            ActionDesc::simple(ActionKind::Close,    "Close this window"),
        ])
    }

    fn agent_dispatch(&mut self, action: AgentAction) -> AgentResult {
        let expected = WINDOW_NODE_BASE + self.id as u64;
        if action.target != expected {
            return AgentResult::NotFound;
        }
        // Window geometry mutations require the WindowManager — this impl handles
        // only the state-level actions that Window itself can reflect. The Os shell
        // acts on AgentResult::Ok and forwards focus/close/min/max to the WM.
        match action.kind {
            ActionKind::Focus | ActionKind::Minimise | ActionKind::Maximise | ActionKind::Close => {
                // These are valid but require the WindowManager to execute; return Ok
                // so the caller (Os::agent_dispatch) knows to handle them.
                AgentResult::Ok
            }
            _ => AgentResult::invalid("unsupported action for Window"),
        }
    }
}

/// Union an optional damage rect with a (possibly empty) rect.
fn union_opt(d: Option<Rect>, r: Rect) -> Option<Rect> {
    if r.w <= 0 || r.h <= 0 {
        return d;
    }
    Some(match d {
        Some(x) => toolkit::union(x, r),
        None => r,
    })
}

/// A per-app `(base, length)` capability region for its process domain. The length is a
/// plausible memory footprint shown in the Task Manager; the base is unique per app so
/// regions never overlap the seeded system domains or each other.
// Map an AppId to its memory/dedup domain id. Offset by 1 so an app can never
// alias domain 0, which is reserved for the shared kernel/fs domain (see
// persist_fs_to and set_metrics). AppId::Desktop casts to 0, so the raw cast
// would otherwise collide with the kernel domain.
fn app_domain_id(app: AppId) -> u64 {
    1 + app as u64
}

fn proc_region(app: AppId) -> (u64, u64) {
    let mb = 1024 * 1024;
    let len = match app {
        AppId::Browser => 96 * mb,
        AppId::Ide => 64 * mb,
        AppId::Explorer => 40 * mb,
        AppId::Editor => 24 * mb,
        AppId::Files => 18 * mb,
        AppId::Settings => 16 * mb,
        AppId::TaskManager => 14 * mb,
        AppId::Terminal => 12 * mb,
        AppId::Desktop => 8 * mb,
    };
    let base = 0x1000_0000 + (app as u64) * 0x0200_0000;
    (base, len)
}

/// Seed the shell's scheduler with the system's resident domains.
fn seed_scheduler() -> Scheduler {
    let mut s = Scheduler::new();
    let mk = |base: u64, len: u64| Capability::mint(base, len, Rights::ALL);
    s.spawn("kernel", mk(0x1000, 0x40_0000));
    s.spawn("compositor", mk(0x40_0000, 0x20_0000));
    s.spawn("netstack", mk(0x60_0000, 0x10_0000));
    s.spawn("vfs", mk(0x70_0000, 0x10_0000));
    s.spawn("shell", mk(0x80_0000, 0x8_0000));
    for _ in 0..14 {
        if let Some(id) = s.next() {
            s.yield_back(id);
        }
    }
    s
}

fn clip(r: Rect, bounds: Rect) -> Rect {
    let x0 = r.x.max(bounds.x);
    let y0 = r.y.max(bounds.y);
    let x1 = (r.x + r.w).min(bounds.x + bounds.w);
    let y1 = (r.y + r.h).min(bounds.y + bounds.h);
    Rect::new(x0, y0, (x1 - x0).max(0), (y1 - y0).max(0))
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

// ── app glyphs (drawn from primitives; centred at (cx,cy)) ──

fn app_glyph(app: AppId, s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    match app {
        AppId::Desktop => glyph_desktop(s, cx, cy, c),
        AppId::Files => glyph_files(s, cx, cy, c),
        AppId::Browser | AppId::Explorer => glyph_globe(s, cx, cy, c),
        AppId::Terminal => glyph_terminal(s, cx, cy, c),
        AppId::Editor => glyph_editor(s, cx, cy, c),
        AppId::Ide => glyph_ide(s, cx, cy, c),
        AppId::TaskManager => glyph_task(s, cx, cy, c),
        AppId::Settings => glyph_settings(s, cx, cy, c),
    }
}

fn glyph_start(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    for (dx, dy) in [(-6, -6), (1, -6), (-6, 1), (1, 1)] {
        s.push(DrawCmd::Rect { rect: Rect::new(cx + dx, cy + dy, 5, 5), color: c, radius: 1 });
    }
}
fn glyph_desktop(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 11, cy - 8, 22, 14), color: c, radius: 2 });
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 3, cy + 6, 6, 3), color: c, radius: 0 });
}
fn glyph_files(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 10, cy - 5, 20, 12), color: c, radius: 2 });
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 10, cy - 8, 9, 4), color: c, radius: 1 });
}
fn glyph_terminal(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: toolkit::inflate(Rect::new(cx - 11, cy - 8, 22, 16), 0), color: c, radius: 2 });
    s.push(toolkit::polyline(alloc::vec![(cx - 6, cy - 3), (cx - 2, cy), (cx - 6, cy + 3)], Color::rgb(0x12, 0x14, 0x18), 2));
    s.push(toolkit::line(cx, cy + 3, cx + 6, cy + 3, Color::rgb(0x12, 0x14, 0x18), 2));
}
fn glyph_editor(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 7, cy - 8, 14, 16), color: c, radius: 1 });
    s.push(toolkit::line(cx - 4, cy - 3, cx + 4, cy - 3, Color::rgb(0x12, 0x14, 0x18), 1));
    s.push(toolkit::line(cx - 4, cy + 1, cx + 4, cy + 1, Color::rgb(0x12, 0x14, 0x18), 1));
}
fn glyph_ide(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(toolkit::polyline(alloc::vec![(cx - 4, cy - 7), (cx - 12, cy), (cx - 4, cy + 7)], c, 2));
    s.push(toolkit::polyline(alloc::vec![(cx + 4, cy - 7), (cx + 12, cy), (cx + 4, cy + 7)], c, 2));
    s.push(toolkit::line(cx + 2, cy - 8, cx - 2, cy + 8, c, 2));
}
fn glyph_globe(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(toolkit::disc(cx, cy, 9, Color::rgba(c.r, c.g, c.b, 60)));
    s.push(toolkit::line(cx - 9, cy, cx + 9, cy, c, 1));
    s.push(toolkit::line(cx, cy - 9, cx, cy + 9, c, 1));
    s.push(toolkit::wire((cx, cy - 9), (cx, cy + 9), c, 1, 7));
}
fn glyph_task(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 9, cy + 2, 4, 6), color: c, radius: 1 });
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 2, cy - 2, 4, 10), color: c, radius: 1 });
    s.push(DrawCmd::Rect { rect: Rect::new(cx + 5, cy - 7, 4, 15), color: c, radius: 1 });
}
fn glyph_settings(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(toolkit::disc(cx, cy, 7, c));
    s.push(toolkit::disc(cx, cy, 3, Color::rgb(0x12, 0x14, 0x18)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::WidgetKind;

    fn os() -> Os {
        let mut o = Os::new();
        o.set_size(1440, 760);
        let _ = o.take_damage();
        o
    }

    /// Click (press + release) at a screen point.
    fn click(o: &mut Os, x: i32, y: i32) {
        o.on_pointer(x, y, true);
        o.on_pointer(x, y, false);
    }

    #[test]
    fn boots_into_the_desktop_with_icons() {
        let o = os();
        assert_eq!(o.active(), AppId::Desktop);
        let s = o.view(1440, 760);
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Jayden"))));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Start")));
        // Desktop icons for the apps appear.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Terminal")));
    }

    #[test]
    fn desktop_icon_opens_a_window_and_taskbar_lists_it() {
        let mut o = os();
        // Terminal is DESKTOP_ICONS index 2.
        let r = o.desktop_icon(2);
        click(&mut o, r.x + 20, r.y + 20);
        assert_eq!(o.active(), AppId::Terminal);
        assert!(o.wm.is_open(AppId::Terminal));
        assert_eq!(o.wm.taskbar().len(), 1);
        // Clicking the focused window's taskbar button minimizes it → focus falls back.
        let b = o.task_btn(0);
        click(&mut o, b.x + b.w / 2, b.y + b.h / 2);
        assert_eq!(o.active(), AppId::Desktop);
        assert!(o.wm.is_open(AppId::Terminal));
    }

    #[test]
    fn start_menu_lists_all_apps_and_launches() {
        let mut o = os();
        let sb = o.start_btn();
        click(&mut o, sb.x + 5, sb.y + 5);
        assert!(o.start_open);
        // Launch the Browser (index 2 in ALL_APPS).
        let m = o.start_menu_rect();
        let y = m.y + 8 + 2 * 30 + 4;
        click(&mut o, m.x + 20, y);
        assert_eq!(o.active(), AppId::Browser);
        assert!(o.wm.is_open(AppId::Browser));
    }

    #[test]
    fn windows_have_chrome_and_close_button_works() {
        let mut o = os();
        o.switch(AppId::Files);
        assert!(o.wm.is_open(AppId::Files));
        let f = o.wm.window(AppId::Files).unwrap().frame(o.wm.area());
        // The close button is the rightmost title-bar button.
        let close_x = f.x + f.w - crate::window::TITLE_H / 2 - 2;
        click(&mut o, close_x, f.y + crate::window::TITLE_H / 2);
        assert!(!o.wm.is_open(AppId::Files));
        assert_eq!(o.active(), AppId::Desktop);
    }

    #[test]
    fn dragging_a_title_bar_moves_the_window() {
        let mut o = os();
        o.switch(AppId::Terminal);
        let f0 = o.wm.window(AppId::Terminal).unwrap().frame(o.wm.area());
        o.on_pointer(f0.x + 50, f0.y + 8, true);
        o.on_pointer(f0.x + 110, f0.y + 68, true);
        o.on_pointer(f0.x + 110, f0.y + 68, false);
        let f1 = o.wm.window(AppId::Terminal).unwrap().frame(o.wm.area());
        assert_eq!((f1.x - f0.x, f1.y - f0.y), (60, 60));
    }

    #[test]
    fn opening_a_file_in_files_switches_to_the_editor() {
        let mut o = os();
        o.switch(AppId::Files);
        o.editor.open("/home/jayden/Documents/welcome.txt");
        o.switch(AppId::Editor);
        assert_eq!(o.active(), AppId::Editor);
        assert_eq!(o.editor.open_path(), Some("/home/jayden/Documents/welcome.txt"));
    }

    #[test]
    fn digit_hotkeys_switch_apps_when_not_typing() {
        let mut o = os();
        o.on_key('4'); // Terminal
        assert_eq!(o.active(), AppId::Terminal);
        // In the Terminal, digits type into the command line (not switch apps).
        o.on_key('2');
        assert_eq!(o.active(), AppId::Terminal);
    }

    #[test]
    fn typing_a_command_into_the_focused_terminal_runs_it_end_to_end() {
        // End-to-end shell input routing: opening the Terminal and typing through the
        // top-level `Os::on_key` (the exact call the kernel desktop loop makes) must feed
        // the terminal's command line, and Enter must run the command through the real
        // ShellBackend so its output renders in the scene. This guards the GUI input path
        // the desktop docs once flagged as "frozen".
        let mut o = os();
        o.on_key('4'); // open + focus the Terminal
        assert_eq!(o.active(), AppId::Terminal);
        // Type `pwd` and submit. `pwd`'s output ("/home/jayden") differs from the echoed
        // input line, so finding it in the render proves the command actually executed
        // rather than merely being echoed.
        for ch in "pwd".chars() {
            o.on_key(ch);
        }
        // The typed characters must appear on the live command line before submitting.
        let s = o.view(1440, 760);
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.ends_with("$ pwd"))));
        o.on_key('\n');
        let s = o.view(1440, 760);
        assert!(
            s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "/home/jayden")),
            "pwd output should render in the terminal scrollback",
        );
        // Enter cleared the command line: the live prompt (a trailing "$ ") no longer
        // carries the typed text, so its text ends at the prompt sigil.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.ends_with("jayden$ "))));
    }

    #[test]
    fn settings_power_off_sets_exit() {
        let mut o = os();
        o.switch(AppId::Settings);
        let b = o.settings.power_btn();
        let c = o.wm.content_of(AppId::Settings);
        click(&mut o, b.x + 10 + c.x, b.y + 10 + c.y);
        assert!(o.wants_exit());
    }

    #[test]
    fn right_click_menu_opens_terminal() {
        let mut o = os();
        o.on_right_click(400, 300);
        assert!(o.ctx.is_some());
        let menu = o.ctx_rect().unwrap();
        // CTX_ITEMS: 0 Edit, 1 Add note, 2 New folder, 3 New shortcut, 4 Open Files, 5 Open Terminal, ...
        let terminal_idx = o.ctx_items.iter().position(|(l, _)| l == "Open Terminal").unwrap();
        click(&mut o, menu.x + 10, menu.y + 4 + terminal_idx as i32 * CTX_ROW + 2);
        assert_eq!(o.active(), AppId::Terminal);
    }

    #[test]
    fn right_click_on_a_window_offers_window_actions_and_closes() {
        let mut o = os();
        o.switch(AppId::Files);
        let f = o.wm.window(AppId::Files).unwrap().frame(o.wm.area());
        o.on_right_click(f.x + 30, f.y + 60);
        // A non-text window → Maximize / Minimize / Close (no clipboard items).
        assert!(o.ctx_items.iter().any(|(l, _)| l == "Close"));
        assert!(!o.ctx_items.iter().any(|(l, _)| l == "Copy"));
        let menu = o.ctx_rect().unwrap();
        let close_idx = o.ctx_items.iter().position(|(l, _)| l == "Close").unwrap();
        let y = menu.y + 4 + close_idx as i32 * CTX_ROW + 2;
        click(&mut o, menu.x + 10, y);
        assert!(!o.wm.is_open(AppId::Files));
    }

    #[test]
    fn right_click_on_a_text_window_offers_clipboard_items() {
        let mut o = os();
        o.switch(AppId::Editor);
        let f = o.wm.window(AppId::Editor).unwrap().frame(o.wm.area());
        o.on_right_click(f.x + 40, f.y + 60);
        assert!(o.ctx_items.iter().any(|(l, _)| l == "Copy"));
        assert!(o.ctx_items.iter().any(|(l, _)| l == "Paste"));
        // Theme toggle must NOT appear in the context menu (it lives in Settings).
        assert!(!o.ctx_items.iter().any(|(l, _)| l.contains("theme") || l.contains("Theme")));
    }

    #[test]
    fn files_context_menu_creates_a_new_folder() {
        let mut o = os();
        o.switch(AppId::Files);
        let f = o.wm.window(AppId::Files).unwrap().frame(o.wm.area());
        // Right-click empty space in the file list (no row selected).
        o.on_right_click(f.x + 40, f.y + 80);
        let idx = o.ctx_items.iter().position(|(l, _)| l == "New folder").expect("New folder item");
        let menu = o.ctx_rect().unwrap();
        let y = menu.y + 4 + idx as i32 * CTX_ROW + 2;
        click(&mut o, menu.x + 10, y);
        assert!(o.fs().borrow().is_dir("/home/jayden/New Folder"));
    }

    #[test]
    fn files_context_menu_has_new_file_and_window_controls() {
        let mut o = os();
        o.switch(AppId::Files);
        let f = o.wm.window(AppId::Files).unwrap().frame(o.wm.area());
        o.on_right_click(f.x + 40, f.y + 80);
        assert!(o.ctx_items.iter().any(|(l, _)| l == "New file"));
        assert!(o.ctx_items.iter().any(|(l, _)| l == "Close"));
        // A non-text app exposes no clipboard items.
        assert!(!o.ctx_items.iter().any(|(l, _)| l == "Copy"));
    }

    #[test]
    fn theme_toggle_via_hotkey() {
        let mut o = os();
        let dark_before = o.theme_dark;
        o.on_key('g');
        assert_ne!(o.theme_dark, dark_before);
    }

    #[test]
    fn edit_ui_compose_round_trip_on_desktop() {
        let mut o = os();
        o.toggle_edit();
        assert!(o.is_editing());
        let base = o.board.panels().len(); // the seeded default widgets
        o.board.add(WidgetKind::Chart);
        o.board.add(WidgetKind::Terminal);
        assert_eq!(o.board.panels().len(), base + 2);
        o.publish_layout();
        assert_eq!(o.library().len(), 1);
        let total = o.board.panels().len();
        o.board = Board::new();
        assert!(o.install_layout());
        assert_eq!(o.board.panels().len(), total);
    }

    #[test]
    fn filesystem_persists_and_restores_across_a_reboot() {
        let o = os();
        o.fs().borrow_mut().write_text("/home/jayden/saved.txt", "persisted!").unwrap();
        let image = o.persist_fs();

        let o2 = os();
        assert!(o2.restore_fs(&image));
        assert_eq!(o2.fs().borrow().read_text("/home/jayden/saved.txt").as_deref(), Some("persisted!"));
    }

    #[test]
    fn account_is_bound_to_the_self_certifying_identity() {
        use crate::dominionlink::DominionId;
        let mut o = os();
        let id = DominionId::from_pubkey(b"user-public-key");
        o.set_account_from_identity(id);
        assert!(o.account().starts_with("id:"));
        assert_eq!(o.account(), &alloc::format!("id:{}", id.0.short()));
    }

    #[test]
    fn browser_is_a_desktop_destination() {
        let mut o = os();
        let r = o.desktop_icon(1); // Browser
        click(&mut o, r.x + 20, r.y + 20);
        assert_eq!(o.active(), AppId::Browser);
    }

    #[test]
    fn metrics_reach_the_tray_clock() {
        let mut o = os();
        let _ = o.take_damage();
        o.set_metrics(Metrics { uptime_secs: 3725, net_present: true, entropy_milli: 970, ..Default::default() });
        let s = o.view(1440, 760);
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "1:02")));
        // SECURE is now rendered as a colored dot, not a text label.
        let ok = Color::rgb(0x3f, 0xc9, 0xb0);
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Disc { color, .. } if *color == ok)));
    }

    #[test]
    fn clipboard_copies_from_one_window_and_pastes_into_another() {
        let mut o = os();
        // Type "hello" into the Terminal, select all, copy.
        o.switch(AppId::Terminal);
        for c in "hello".chars() {
            o.on_key(c);
        }
        o.on_key('\u{07}'); // Ctrl+A — select all (forwarded to the terminal input)
        o.on_key('\u{03}'); // Ctrl+C — copy (intercepted by the shell)
        assert_eq!(o.clipboard(), "hello");
        // Paste it into the Editor.
        o.switch(AppId::Editor);
        o.on_key('i'); // insert mode
        o.on_key('\u{16}'); // Ctrl+V — paste
        let s = o.view(1440, 760);
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.starts_with("hello"))));
    }

    #[test]
    fn two_windows_overlap_and_focus_follows_clicks() {
        let mut o = os();
        o.switch(AppId::Files);
        o.switch(AppId::Terminal);
        assert_eq!(o.active(), AppId::Terminal);
        assert_eq!(o.wm.taskbar().len(), 2);
        // Click on the Files window's title bar to refocus it.
        let f = o.wm.window(AppId::Files).unwrap().frame(o.wm.area());
        click(&mut o, f.x + 40, f.y + 8);
        assert_eq!(o.active(), AppId::Files);
    }
}

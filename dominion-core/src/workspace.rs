//! The universal Workspace — one tabbed window, each tab an object in a view (see
//! `docs/ui/universal-workspace.md`).
//!
//! There are no apps: the "editor", "files", "browser" and any object are **tabs**
//! in one Workspace, each a *view over an object*. The Workspace owns the tab strip,
//! the split/tile layout, tab detach, and a **single undo timeline** (Stage 10
//! time-travel via [`crate::ui::History`]) that spans every tab — undo works
//! everywhere, across the whole window.
//!
//! Renders to a backend-agnostic [`crate::toolkit`] scene: a tab bar plus the active
//! tab's content. Pure, safe `no_std`.

use crate::editor::Editor;
use crate::hash::Hash256;
use crate::toolkit::{self, Axis, DrawCmd, Rect, Size, Widget};
use crate::ui::{History, UiError};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// What a tab shows — every kind is a *view over an object*, not a separate app.
pub enum TabContent {
    /// The universal editor (Notepad++ ⊕ Vim ⊕ calculator).
    Editor(Editor),
    /// A web view (native or legacy) addressed by name/URL.
    Browser(String),
    /// A folder object shown as a file tree.
    Files(String),
    /// An arbitrary graph object shown in its best view.
    Object(Hash256),
}

impl TabContent {
    /// The display kind of this tab ("Editor"/"Browser"/"Files"/"Object").
    pub fn kind_label(&self) -> &'static str {
        match self {
            TabContent::Editor(_) => "Editor",
            TabContent::Browser(_) => "Browser",
            TabContent::Files(_) => "Files",
            TabContent::Object(_) => "Object",
        }
    }
}

/// A single tab.
pub struct Tab {
    pub id: u32,
    pub title: String,
    pub content: TabContent,
}

/// How the Workspace tiles its visible panes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitLayout {
    /// One pane (the active tab).
    Single,
    /// Two panes side by side.
    Vertical,
    /// Two panes stacked.
    Horizontal,
}

/// The tabbed Workspace.
pub struct Workspace {
    tabs: Vec<Tab>,
    active: usize,
    next_id: u32,
    layout: SplitLayout,
    /// The universal undo timeline (spans every tab).
    history: History,
}

impl Workspace {
    /// A fresh Workspace with one empty editor tab.
    pub fn new() -> Workspace {
        let genesis = Hash256::of(b"workspace-genesis");
        let mut ws = Workspace {
            tabs: Vec::new(),
            active: 0,
            next_id: 1,
            layout: SplitLayout::Single,
            history: History::new(genesis),
        };
        ws.open(TabContent::Editor(Editor::new("")), "untitled");
        ws
    }

    /// Open a new tab and make it active; returns its id.
    pub fn open(&mut self, content: TabContent, title: &str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.tabs.push(Tab { id, title: title.into(), content });
        self.active = self.tabs.len() - 1;
        id
    }

    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }
    pub fn active_index(&self) -> usize {
        self.active
    }
    pub fn layout(&self) -> SplitLayout {
        self.layout
    }
    pub fn tabs(&self) -> &[Tab] {
        &self.tabs
    }

    /// The active tab (panics only if there are no tabs, which `new` prevents).
    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active]
    }
    pub fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active]
    }

    /// Activate a tab by id; returns false if unknown.
    pub fn activate(&mut self, id: u32) -> bool {
        if let Some(i) = self.tabs.iter().position(|t| t.id == id) {
            self.active = i;
            true
        } else {
            false
        }
    }

    /// Cycle to the next / previous tab (wrapping).
    pub fn next(&mut self) {
        if !self.tabs.is_empty() {
            self.active = (self.active + 1) % self.tabs.len();
        }
    }
    pub fn prev(&mut self) {
        if !self.tabs.is_empty() {
            self.active = (self.active + self.tabs.len() - 1) % self.tabs.len();
        }
    }

    /// Close a tab by id. The active index is kept valid. The last tab cannot be
    /// closed (an empty Workspace has no meaning); it is instead left in place.
    pub fn close(&mut self, id: u32) -> bool {
        if self.tabs.len() <= 1 {
            return false;
        }
        if let Some(i) = self.tabs.iter().position(|t| t.id == id) {
            self.tabs.remove(i);
            if self.active >= self.tabs.len() {
                self.active = self.tabs.len() - 1;
            } else if self.active > i {
                self.active -= 1;
            }
            true
        } else {
            false
        }
    }

    /// Detach a tab into its own window: remove and return it (the caller hosts it in
    /// a new Workspace). The last tab is not detachable.
    pub fn detach(&mut self, id: u32) -> Option<Tab> {
        if self.tabs.len() <= 1 {
            return None;
        }
        let i = self.tabs.iter().position(|t| t.id == id)?;
        let tab = self.tabs.remove(i);
        if self.active >= self.tabs.len() {
            self.active = self.tabs.len() - 1;
        } else if self.active > i {
            self.active -= 1;
        }
        Some(tab)
    }

    /// Set the split/tile layout.
    pub fn set_layout(&mut self, layout: SplitLayout) {
        self.layout = layout;
    }

    // ── universal undo timeline (spans all tabs) ──

    /// Record a new whole-Workspace content root (e.g. after any edit in any tab).
    pub fn commit(&mut self, root: Hash256) {
        self.history.commit(root);
    }
    /// Undo across the whole Workspace.
    pub fn undo(&mut self) -> Result<Hash256, UiError> {
        self.history.undo()
    }
    /// Redo.
    pub fn redo(&mut self) -> Option<Hash256> {
        self.history.redo()
    }
    pub fn current_root(&self) -> Hash256 {
        self.history.current()
    }

    // ── rendering ──

    /// Build the Workspace scene: the tab strip plus the active tab's content. The
    /// tab labels show `title` and the active one is highlighted by the toolkit.
    pub fn view(&self, theme: &toolkit::Theme, area: Rect) -> Vec<DrawCmd> {
        let labels: Vec<String> = self
            .tabs
            .iter()
            .map(|t| {
                let mut s = String::new();
                s.push_str(t.title.as_str());
                s
            })
            .collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        // Chrome: a column of [tab bar, content placeholder].
        let chrome = Widget::Container {
            id: 0,
            axis: Axis::Column,
            padding: 0,
            size: Size::Flex(1),
            children: vec![
                toolkit::tabs(1, 100, &label_refs, self.active),
                Widget::Container { id: 2, axis: Axis::Column, padding: 0, size: Size::Flex(1), children: vec![] },
            ],
        };
        let placements = toolkit::layout(&chrome, area);
        let mut scene = toolkit::build_scene(&chrome, theme, area);
        let content_area = placements
            .iter()
            .find(|(id, _)| *id == 2)
            .map(|(_, r)| *r)
            .unwrap_or(area);
        // Render the active tab's content into the content area.
        match &self.active_tab().content {
            TabContent::Editor(e) => scene.extend(e.view(theme, content_area)),
            other => {
                let title = match other {
                    TabContent::Browser(url) => {
                        let mut s = String::from("web: ");
                        s.push_str(url);
                        s
                    }
                    TabContent::Files(path) => {
                        let mut s = String::from("files: ");
                        s.push_str(path);
                        s
                    }
                    TabContent::Object(h) => {
                        let mut s = String::from("object ");
                        s.push_str(&h.short());
                        s
                    }
                    TabContent::Editor(_) => String::new(),
                };
                let body = toolkit::label(3, &title);
                scene.extend(toolkit::build_scene(&body, theme, content_area));
            }
        }
        scene
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_with_one_editor_tab() {
        let ws = Workspace::new();
        assert_eq!(ws.tab_count(), 1);
        assert_eq!(ws.active_tab().content.kind_label(), "Editor");
    }

    #[test]
    fn open_close_activate_cycle() {
        let mut ws = Workspace::new();
        let b = ws.open(TabContent::Browser("dominion://home".into()), "Home");
        let f = ws.open(TabContent::Files("~/project".into()), "Files");
        assert_eq!(ws.tab_count(), 3);
        assert_eq!(ws.active_index(), 2); // newly opened is active
        assert!(ws.activate(b));
        assert_eq!(ws.active_tab().content.kind_label(), "Browser");
        // next/prev wrap.
        ws.next();
        assert_eq!(ws.active_tab().content.kind_label(), "Files");
        ws.next();
        assert_eq!(ws.active_index(), 0);
        ws.prev();
        assert_eq!(ws.active_index(), 2);
        // close a tab; active stays valid.
        assert!(ws.close(f));
        assert_eq!(ws.tab_count(), 2);
    }

    #[test]
    fn last_tab_cannot_be_closed_or_detached() {
        let mut ws = Workspace::new();
        let only = ws.active_tab().id;
        assert!(!ws.close(only));
        assert!(ws.detach(only).is_none());
        assert_eq!(ws.tab_count(), 1);
    }

    #[test]
    fn detach_removes_and_returns_the_tab() {
        let mut ws = Workspace::new();
        let b = ws.open(TabContent::Browser("x".into()), "B");
        let taken = ws.detach(b).unwrap();
        assert_eq!(taken.content.kind_label(), "Browser");
        assert_eq!(ws.tab_count(), 1);
    }

    #[test]
    fn split_layout_set_and_get() {
        let mut ws = Workspace::new();
        assert_eq!(ws.layout(), SplitLayout::Single);
        ws.set_layout(SplitLayout::Vertical);
        assert_eq!(ws.layout(), SplitLayout::Vertical);
    }

    #[test]
    fn universal_undo_spans_the_workspace() {
        let mut ws = Workspace::new();
        let v1 = ws.current_root();
        let r1 = Hash256::of(b"edit-1");
        let r2 = Hash256::of(b"edit-2");
        ws.commit(r1);
        ws.commit(r2);
        assert_eq!(ws.current_root(), r2);
        // Undo steps back across the whole window's timeline.
        assert_eq!(ws.undo().unwrap(), r1);
        assert_eq!(ws.undo().unwrap(), v1);
        // Redo returns forward.
        assert_eq!(ws.redo(), Some(r1));
    }

    #[test]
    fn view_renders_tab_bar_and_active_editor_content() {
        let mut ws = Workspace::new();
        // Put an expression in the editor tab so its content shows an inline result.
        if let TabContent::Editor(e) = &mut ws.active_tab_mut().content {
            *e = Editor::new("21 * 2");
        }
        ws.open(TabContent::Browser("dominion://news".into()), "News");
        ws.activate(1); // back to the editor tab
        let scene = ws.view(&toolkit::Theme::dark(), Rect::new(0, 0, 400, 300));
        // The tab strip shows both tab titles, and the active editor renders "42".
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "untitled")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "News")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("42"))));
    }
}

//! The desktop shell — dashboard, dock, command palette and the capability panel
//! (see `docs/ui/design-system-and-shell.md`).
//!
//! Composes the *Calm Spatial* shell from [`crate::toolkit`] primitives: a top bar
//! (launcher + workspace + search/command bar + toggles), a content region, and a
//! dock of the few universal surfaces. The launcher is a **command palette** that
//! fuses search + launch + intent → a [`crate::workspace`] tab (there is no
//! installed-app list). The **capability panel** *is* the settings surface: every
//! grant a thing holds, shown and revocable. Renders to a backend-agnostic scene.
//!
//! Pure, safe `no_std`.

use crate::toolkit::{self, ButtonVariant, DrawCmd, Rect, Size, Widget};
use crate::workspace::TabContent;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// A universal surface on the dock (not an installed app).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockItem {
    Workspace,
    Editor,
    Browser,
    Files,
    Settings,
}

impl DockItem {
    pub fn label(self) -> &'static str {
        match self {
            DockItem::Workspace => "Workspace",
            DockItem::Editor => "Editor",
            DockItem::Browser => "Browser",
            DockItem::Files => "Files",
            DockItem::Settings => "Settings",
        }
    }
    fn all() -> [DockItem; 5] {
        [DockItem::Workspace, DockItem::Editor, DockItem::Browser, DockItem::Files, DockItem::Settings]
    }
}

const TOPBAR_H: i32 = 40;
const DOCK_H: i32 = 44;
/// Widget id base for dock buttons (so hit-testing can map back to a `DockItem`).
const DOCK_BASE: u32 = 200;

/// The shell: the active dock selection and the current command-bar query.
pub struct Shell {
    active: DockItem,
    query: String,
}

impl Shell {
    pub fn new() -> Shell {
        Shell { active: DockItem::Workspace, query: String::new() }
    }

    pub fn active(&self) -> DockItem {
        self.active
    }
    pub fn set_active(&mut self, item: DockItem) {
        self.active = item;
    }
    pub fn query(&self) -> &str {
        &self.query
    }
    pub fn set_query(&mut self, q: &str) {
        self.query = q.to_string();
    }

    /// Build the dashboard scene: top bar + content area + dock.
    pub fn view(&self, theme: &toolkit::Theme, area: Rect) -> Vec<DrawCmd> {
        // Top bar: launcher glyph, workspace label (flex), search input, theme toggle.
        let topbar = Widget::Container {
            id: 1,
            axis: toolkit::Axis::Row,
            padding: theme.space / 2,
            size: Size::Fixed(TOPBAR_H),
            children: vec![
                toolkit::button_variant(10, "DominionOS", ButtonVariant::Ghost),
                Widget::Label { id: 11, text: "workspace".into(), size: Size::Flex(1) },
                Widget::Input {
                    id: 12,
                    text: self.query.clone(),
                    placeholder: "ask or search…".into(),
                    size: Size::Fixed(220),
                },
                toolkit::button_variant(13, "◐", ButtonVariant::Ghost),
            ],
        };
        // Dock: the universal surfaces, the active one highlighted.
        let dock_children: Vec<Widget> = DockItem::all()
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let v = if *item == self.active { ButtonVariant::Primary } else { ButtonVariant::Ghost };
                toolkit::button_variant(DOCK_BASE + i as u32, item.label(), v)
            })
            .collect();
        let dock = Widget::Container {
            id: 2,
            axis: toolkit::Axis::Row,
            padding: theme.space / 2,
            size: Size::Fixed(DOCK_H),
            children: dock_children,
        };
        let shell = Widget::Container {
            id: 0,
            axis: toolkit::Axis::Column,
            padding: 0,
            size: Size::Flex(1),
            children: vec![
                topbar,
                Widget::Container { id: 3, axis: toolkit::Axis::Column, padding: 0, size: Size::Flex(1), children: vec![] },
                dock,
            ],
        };
        toolkit::build_scene(&shell, theme, area)
    }

    /// The content region rect (between the top bar and the dock) — where the active
    /// surface (Workspace/editor/browser/files) renders.
    pub fn content_area(&self, area: Rect) -> Rect {
        Rect::new(area.x, area.y + TOPBAR_H, area.w, (area.h - TOPBAR_H - DOCK_H).max(0))
    }

    /// Map a click to a dock item, if any (using the layout of [`view`]).
    pub fn dock_hit(&self, theme: &toolkit::Theme, area: Rect, px: i32, py: i32) -> Option<DockItem> {
        let _ = self.view(theme, area); // ensure consistent layout assumptions
        // Re-derive the dock layout the same way `view` builds it.
        let dock_children: Vec<Widget> = DockItem::all()
            .iter()
            .enumerate()
            .map(|(i, item)| toolkit::button_variant(DOCK_BASE + i as u32, item.label(), ButtonVariant::Ghost))
            .collect();
        let dock = Widget::Container {
            id: 2,
            axis: toolkit::Axis::Row,
            padding: theme.space / 2,
            size: Size::Flex(1),
            children: dock_children,
        };
        let dock_area = Rect::new(area.x, area.y + area.h - DOCK_H, area.w, DOCK_H);
        let placements = toolkit::layout(&dock, dock_area);
        let id = toolkit::hit_test(&placements, px, py)?;
        if (DOCK_BASE..DOCK_BASE + 5).contains(&id) {
            Some(DockItem::all()[(id - DOCK_BASE) as usize])
        } else {
            None
        }
    }

    /// The launcher/command palette overlay (search input + result rows).
    pub fn command_palette(&self, theme: &toolkit::Theme, area: Rect, results: &[&str]) -> Vec<DrawCmd> {
        let palette = toolkit::command_palette(50, &self.query, results);
        toolkit::build_scene(&palette, theme, area)
    }

    /// Route a command-bar query to the tab it should open — fusing search, launch and
    /// intent without an installed-app list. A URL or "browse …" opens a Browser tab;
    /// "files" a Files tab; otherwise an Editor tab seeded with the text.
    pub fn command_to_tab(query: &str) -> TabContent {
        let q = query.trim();
        let lower = to_lower(q);
        if lower.starts_with("http")
            || lower.starts_with("dominion://")
            || lower.starts_with("browse ")
            || lower.contains(".com")
            || lower.contains(".org")
        {
            let url = lower.strip_prefix("browse ").unwrap_or(&lower).to_string();
            TabContent::Browser(url)
        } else if lower.starts_with("files") || (lower.starts_with("open ") && lower.contains('/')) {
            TabContent::Files(q.to_string())
        } else {
            TabContent::Editor(crate::editor::Editor::new(q))
        }
    }
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

/// One capability shown in the panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapEntry {
    pub label: String,
    pub granted: bool,
}

/// The capability/permission panel — the settings surface. Lists every capability a
/// subject holds with a grant/revoke action; revocation is recursive at the
/// `firewall.rs` layer.
pub struct CapabilityPanel {
    entries: Vec<CapEntry>,
}

impl CapabilityPanel {
    pub fn new() -> CapabilityPanel {
        CapabilityPanel { entries: Vec::new() }
    }

    pub fn add(&mut self, label: &str, granted: bool) {
        self.entries.push(CapEntry { label: label.into(), granted });
    }

    pub fn entries(&self) -> &[CapEntry] {
        &self.entries
    }

    /// Toggle a capability's grant; returns the new state.
    pub fn toggle(&mut self, label: &str) -> Option<bool> {
        let e = self.entries.iter_mut().find(|e| e.label == label)?;
        e.granted = !e.granted;
        Some(e.granted)
    }

    /// Build the panel scene: a row per capability (label + grant/revoke button).
    pub fn view(&self, theme: &toolkit::Theme, area: Rect) -> Vec<DrawCmd> {
        let mut rows: Vec<Widget> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let action = if e.granted {
                toolkit::button_variant(1000 + i as u32, "Revoke", ButtonVariant::Danger)
            } else {
                toolkit::button_variant(1000 + i as u32, "Grant", ButtonVariant::Primary)
            };
            let row = Widget::Container {
                id: 500 + i as u32,
                axis: toolkit::Axis::Row,
                padding: 2,
                size: Size::Fixed(theme.font_size + 14),
                children: vec![
                    Widget::Label { id: 600 + i as u32, text: e.label.clone(), size: Size::Flex(1) },
                    action,
                ],
            };
            rows.push(row);
        }
        if rows.is_empty() {
            rows.push(toolkit::label(1, "(no capabilities held)"));
        }
        let col = Widget::Container {
            id: 0,
            axis: toolkit::Axis::Column,
            padding: theme.space,
            size: Size::Flex(1),
            children: rows,
        };
        toolkit::build_scene(&col, theme, area)
    }
}

impl Default for CapabilityPanel {
    fn default() -> Self {
        Self::new()
    }
}

fn to_lower(s: &str) -> String {
    s.chars().map(|c| c.to_ascii_lowercase()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_renders_topbar_and_dock() {
        let shell = Shell::new();
        let scene = shell.view(&toolkit::Theme::dark(), Rect::new(0, 0, 800, 480));
        // The dock shows the universal surfaces; the active one is primary.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Workspace")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Browser")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "ask or search…")));
    }

    #[test]
    fn content_area_excludes_bars() {
        let shell = Shell::new();
        let area = Rect::new(0, 0, 800, 480);
        let c = shell.content_area(area);
        assert_eq!(c.y, TOPBAR_H);
        assert_eq!(c.h, 480 - TOPBAR_H - DOCK_H);
    }

    #[test]
    fn dock_click_selects_a_surface() {
        let shell = Shell::new();
        let theme = toolkit::Theme::dark();
        let area = Rect::new(0, 0, 500, 480);
        // The dock is the bottom strip; click in its left-most fifth → Workspace.
        let hit = shell.dock_hit(&theme, area, 40, 480 - DOCK_H / 2);
        assert_eq!(hit, Some(DockItem::Workspace));
        // Click further right hits a later dock item.
        let hit2 = shell.dock_hit(&theme, area, 450, 480 - DOCK_H / 2);
        assert!(hit2.is_some());
        // A click in the content area is not a dock hit.
        assert_eq!(shell.dock_hit(&theme, area, 250, 200), None);
    }

    #[test]
    fn command_routing_picks_the_right_tab() {
        assert!(matches!(Shell::command_to_tab("https://example.com"), TabContent::Browser(_)));
        assert!(matches!(Shell::command_to_tab("browse dominion://news"), TabContent::Browser(_)));
        assert!(matches!(Shell::command_to_tab("files"), TabContent::Files(_)));
        assert!(matches!(Shell::command_to_tab("write a note about the toolkit"), TabContent::Editor(_)));
        // The editor tab is seeded with the query as a live notebook line.
        if let TabContent::Editor(e) = Shell::command_to_tab("21 * 2") {
            assert!(e.evaluate().iter().any(|(_, v)| v == "42"));
        } else {
            panic!("expected an editor tab");
        }
    }

    #[test]
    fn capability_panel_lists_and_toggles() {
        let mut panel = CapabilityPanel::new();
        panel.add("Net → example.com", true);
        panel.add("Storage", false);
        let scene = panel.view(&toolkit::Theme::dark(), Rect::new(0, 0, 400, 200));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Net"))));
        // A granted capability shows "Revoke"; revoking flips it.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Revoke")));
        assert_eq!(panel.toggle("Net → example.com"), Some(false));
        assert_eq!(panel.toggle("Storage"), Some(true));
        assert_eq!(panel.toggle("nonexistent"), None);
    }

    #[test]
    fn command_palette_overlay_builds() {
        let mut shell = Shell::new();
        shell.set_query("inv");
        let scene = shell.command_palette(&toolkit::Theme::dark(), Rect::new(0, 0, 400, 300), &["Open Sales", "Find invoices"]);
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("invoices"))));
    }

    #[test]
    fn dock_selection_state() {
        let mut shell = Shell::new();
        assert_eq!(shell.active(), DockItem::Workspace);
        shell.set_active(DockItem::Browser);
        assert_eq!(shell.active(), DockItem::Browser);
    }
}

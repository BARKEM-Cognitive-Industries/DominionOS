//! Accessibility & internationalisation (see
//! `docs/architecture/accessibility-and-i18n.md`).
//!
//! Accessibility is not a bolt-on: because the UI is built from *semantic* object
//! nodes (not opaque pixels), every surface already carries the structure a screen
//! reader needs. This module provides:
//!
//! * A **semantic accessibility tree** — roles, labels, values, focus order — that
//!   a screen reader or switch device walks to announce and navigate the UI.
//! * An **i18n catalog**: locale selection, message translation with fallback, and
//!   text-direction (LTR/RTL) so the same UI renders correctly in any language.
//! * **User preferences** (scale, high-contrast, reduced-motion) the compositor and
//!   apps honour.
//!
//! Pure, safe `no_std + alloc`, host-tested.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

// ───────────────────────── Accessibility tree ─────────────────────────

/// The full accessibility tree for the running UI.
/// Built by the OS shell each frame from the window manager's state and
/// exposed to screen readers, switch devices, and automated testing tools.
#[derive(Default)]
pub struct A11yTree {
    /// Top-level nodes (one per open window, plus the desktop backdrop).
    pub roots: Vec<A11yNode>,
    /// The id of the currently focused node, if any.
    pub focused_id: Option<u64>,
    /// The last text announced to the screen reader (via [`A11yTree::announce`]).
    pub last_announcement: String,
}

impl A11yTree {
    pub fn new() -> A11yTree {
        A11yTree { roots: Vec::new(), focused_id: None, last_announcement: String::new() }
    }

    /// Replace the tree's roots with a fresh set built from the window manager.
    pub fn set_roots(&mut self, roots: Vec<A11yNode>) {
        self.roots = roots;
    }

    /// Update the focused node id and emit a screen-reader announcement for it.
    pub fn set_focus(&mut self, id: u64) {
        self.focused_id = Some(id);
        if let Some(node) = self.find(id) {
            let text = node.announce();
            self.last_announcement = text;
        }
    }

    /// Announce arbitrary text to the screen reader (e.g. on focus change or
    /// status update).  The text is stored in `last_announcement` so that kernel
    /// audio / TTS drivers can poll and speak it.
    pub fn announce(&mut self, text: impl Into<String>) {
        self.last_announcement = text.into();
    }

    /// The most recent screen-reader announcement (cleared after the caller
    /// consumes it with [`A11yTree::take_announcement`]).
    pub fn last_announcement(&self) -> &str {
        &self.last_announcement
    }

    /// Take (and clear) the pending announcement — call once per TTS tick.
    pub fn take_announcement(&mut self) -> Option<String> {
        if self.last_announcement.is_empty() {
            None
        } else {
            Some(core::mem::take(&mut self.last_announcement))
        }
    }

    /// Depth-first search for a node with the given id.
    pub fn find(&self, id: u64) -> Option<&A11yNode> {
        for root in &self.roots {
            if let Some(n) = root.find(id) {
                return Some(n);
            }
        }
        None
    }

    /// Flat focus-order list across all roots (depth-first, focusable nodes only).
    pub fn focus_order(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for root in &self.roots {
            out.extend(root.focus_order());
        }
        out
    }

    /// Total node count across all roots.
    pub fn count(&self) -> usize {
        self.roots.iter().map(|r| r.count()).sum()
    }
}

/// The semantic role of a UI node — what an assistive technology announces it as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Window,
    Button,
    Text,
    TextField,
    List,
    ListItem,
    Image,
    Heading,
    Checkbox,
}

impl Role {
    pub fn spoken(self) -> &'static str {
        match self {
            Role::Window => "window",
            Role::Button => "button",
            Role::Text => "text",
            Role::TextField => "text field",
            Role::List => "list",
            Role::ListItem => "list item",
            Role::Image => "image",
            Role::Heading => "heading",
            Role::Checkbox => "checkbox",
        }
    }
}

/// One node in the accessibility tree.
#[derive(Clone, Debug)]
pub struct A11yNode {
    pub id: u64,
    pub role: Role,
    /// Accessible name (already localised by the time it lands here).
    pub label: String,
    pub value: Option<String>,
    pub focusable: bool,
    pub children: Vec<A11yNode>,
}

impl A11yNode {
    pub fn new(id: u64, role: Role, label: impl Into<String>) -> A11yNode {
        A11yNode { id, role, label: label.into(), value: None, focusable: false, children: Vec::new() }
    }

    pub fn focusable(mut self) -> A11yNode {
        self.focusable = true;
        self
    }

    pub fn with_value(mut self, v: impl Into<String>) -> A11yNode {
        self.value = Some(v.into());
        self
    }

    pub fn child(mut self, node: A11yNode) -> A11yNode {
        self.children.push(node);
        self
    }

    /// What a screen reader announces for this node.
    pub fn announce(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.label);
        s.push_str(", ");
        s.push_str(self.role.spoken());
        if let Some(v) = &self.value {
            s.push_str(", ");
            s.push_str(v);
        }
        s
    }

    /// Depth-first focus order: every focusable node, parents before children.
    pub fn focus_order(&self) -> Vec<u64> {
        let mut out = Vec::new();
        self.collect_focusable(&mut out);
        out
    }

    fn collect_focusable(&self, out: &mut Vec<u64>) {
        if self.focusable {
            out.push(self.id);
        }
        for c in &self.children {
            c.collect_focusable(out);
        }
    }

    /// Total node count (for completeness checks).
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }

    /// Depth-first search for a node by id within this subtree.
    pub fn find(&self, id: u64) -> Option<&A11yNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(n) = child.find(id) {
                return Some(n);
            }
        }
        None
    }
}

/// Text directionality of a locale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    LeftToRight,
    RightToLeft,
}

/// A translation catalog for one or more locales, with fallback to a base locale.
#[derive(Default)]
pub struct I18n {
    /// locale → (message key → translated string)
    catalogs: BTreeMap<String, BTreeMap<String, String>>,
    /// locale → direction
    directions: BTreeMap<String, Direction>,
    fallback: String,
    current: String,
}

impl I18n {
    pub fn new(fallback: impl Into<String>) -> I18n {
        let fb = fallback.into();
        I18n {
            catalogs: BTreeMap::new(),
            directions: BTreeMap::new(),
            fallback: fb.clone(),
            current: fb,
        }
    }

    pub fn add_locale(&mut self, locale: impl Into<String>, dir: Direction) {
        let l = locale.into();
        self.catalogs.entry(l.clone()).or_default();
        self.directions.insert(l, dir);
    }

    pub fn set_message(&mut self, locale: &str, key: &str, value: &str) {
        self.catalogs
            .entry(locale.into())
            .or_default()
            .insert(key.into(), value.into());
    }

    /// Select the active locale (must have been added).
    pub fn set_locale(&mut self, locale: &str) -> bool {
        if self.catalogs.contains_key(locale) {
            self.current = locale.into();
            true
        } else {
            false
        }
    }

    pub fn direction(&self) -> Direction {
        *self.directions.get(&self.current).unwrap_or(&Direction::LeftToRight)
    }

    /// Translate `key` in the current locale, falling back to the base locale, then
    /// to the key itself (so missing translations are visible, never blank).
    pub fn translate(&self, key: &str) -> String {
        if let Some(v) = self.catalogs.get(&self.current).and_then(|c| c.get(key)) {
            return v.clone();
        }
        if let Some(v) = self.catalogs.get(&self.fallback).and_then(|c| c.get(key)) {
            return v.clone();
        }
        key.into()
    }
}

/// System accessibility/display preferences the UI must honour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct A11yPrefs {
    /// UI scale in per-mille (1000 = 100%).
    pub scale_milli: u32,
    pub high_contrast: bool,
    pub reduced_motion: bool,
    pub screen_reader: bool,
}

impl Default for A11yPrefs {
    fn default() -> A11yPrefs {
        A11yPrefs { scale_milli: 1000, high_contrast: false, reduced_motion: false, screen_reader: false }
    }
}

impl A11yPrefs {
    /// Scale a pixel length by the preference (e.g. font sizes).
    pub fn scale(&self, px: u32) -> u32 {
        px.saturating_mul(self.scale_milli) / 1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> A11yNode {
        A11yNode::new(1, Role::Window, "Settings")
            .child(A11yNode::new(2, Role::Heading, "Display"))
            .child(A11yNode::new(3, Role::Checkbox, "High contrast").focusable().with_value("off"))
            .child(
                A11yNode::new(4, Role::List, "Options")
                    .child(A11yNode::new(5, Role::ListItem, "Brightness").focusable())
                    .child(A11yNode::new(6, Role::Button, "Apply").focusable()),
            )
    }

    #[test]
    fn node_announcement_includes_role_and_value() {
        let n = A11yNode::new(1, Role::Checkbox, "High contrast").with_value("off");
        assert_eq!(n.announce(), "High contrast, checkbox, off");
    }

    #[test]
    fn focus_order_is_depth_first_focusables_only() {
        let tree = sample_tree();
        // Only nodes 3, 5, 6 are focusable, in DFS order.
        assert_eq!(tree.focus_order(), alloc::vec![3, 5, 6]);
        assert_eq!(tree.count(), 6);
    }

    #[test]
    fn i18n_translates_with_fallback() {
        let mut i = I18n::new("en");
        i.add_locale("en", Direction::LeftToRight);
        i.add_locale("ar", Direction::RightToLeft);
        i.set_message("en", "greeting", "Hello");
        i.set_message("ar", "greeting", "مرحبا");
        // Default locale.
        assert_eq!(i.translate("greeting"), "Hello");
        // Switch to Arabic: translation + RTL direction.
        assert!(i.set_locale("ar"));
        assert_eq!(i.translate("greeting"), "مرحبا");
        assert_eq!(i.direction(), Direction::RightToLeft);
        // Missing key in ar falls back to en, then to the key itself.
        i.set_message("en", "save", "Save");
        assert_eq!(i.translate("save"), "Save"); // fell back to en
        assert_eq!(i.translate("nonexistent"), "nonexistent");
    }

    #[test]
    fn unknown_locale_is_rejected() {
        let mut i = I18n::new("en");
        i.add_locale("en", Direction::LeftToRight);
        assert!(!i.set_locale("zz"));
    }

    #[test]
    fn prefs_scale_and_defaults() {
        let mut p = A11yPrefs::default();
        assert_eq!(p.scale(16), 16);
        assert!(!p.high_contrast);
        p.scale_milli = 1500; // 150%
        assert_eq!(p.scale(16), 24);
    }

    #[test]
    fn a11y_tree_builds_roots_and_tracks_focus() {
        let mut tree = A11yTree::new();
        assert_eq!(tree.count(), 0);

        // Build two window roots.
        let w1 = A11yNode::new(1, Role::Window, "Files").focusable();
        let w2 = A11yNode::new(2, Role::Window, "Terminal")
            .focusable()
            .child(A11yNode::new(3, Role::TextField, "input").focusable());
        tree.set_roots(alloc::vec![w1, w2]);
        assert_eq!(tree.count(), 3); // Files + Terminal + input
        assert_eq!(tree.focus_order(), alloc::vec![1, 2, 3]);

        // Focus Terminal; announcement should name it.
        tree.set_focus(2);
        assert_eq!(tree.focused_id, Some(2));
        assert!(tree.last_announcement().contains("Terminal"));
        assert!(tree.last_announcement().contains("window"));

        // take_announcement clears the buffer.
        let text = tree.take_announcement();
        assert!(text.is_some());
        assert!(tree.take_announcement().is_none());
    }

    #[test]
    fn a11y_tree_find_descends_into_children() {
        let mut tree = A11yTree::new();
        let root = A11yNode::new(10, Role::Window, "IDE")
            .child(A11yNode::new(11, Role::TextField, "code").focusable());
        tree.set_roots(alloc::vec![root]);
        assert!(tree.find(11).is_some());
        assert!(tree.find(99).is_none());
    }
}

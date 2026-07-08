//! The **Files** app — a graphical file manager with the familiar Windows-Explorer
//! shape (a toolbar with Back/Up + breadcrumb, a Quick-Access sidebar, and a detail
//! list of folders and files), drawn over the shell's live
//! [`FileSystem`](crate::filesystem).
//!
//! It is a *view over the object graph*, not a parallel filesystem: every folder and
//! file shown is a real path alias in the [`Vfs`](crate::vfs), so anything the
//! Terminal's `mkdir`/`touch` creates appears here, and a file opened here is the same
//! object the Terminal can `cat`. Clicking a folder navigates into it; clicking a file
//! asks the shell to open it in the Editor. Pure, safe `no_std`, rendered in
//! page-local coordinates following the same page contract as the other shell pages.

use crate::filesystem::{self, SharedFs};
use crate::text::TextBuffer;
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::widgets::Scroll;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const TOOLBAR_H: i32 = 38;
const SEARCH_H: i32 = 30;
const SIDEBAR_W: i32 = 168;
const ROW_H: i32 = 30;

/// What a click in the Files app asks the shell to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilesAction {
    /// Open this file path in the Editor.
    OpenFile(String),
    /// Rename a file from old path to new name (same directory).
    Rename(String, String),
}

/// Quick-access shortcuts shown in the sidebar (label, absolute path).
const QUICK: [(&str, &str); 6] = [
    ("Home", "/home/jayden"),
    ("Documents", "/home/jayden/Documents"),
    ("Downloads", "/home/jayden/Downloads"),
    ("Pictures", "/home/jayden/Pictures"),
    ("Projects", "/home/jayden/Projects"),
    ("System (/)", "/"),
];

/// The Files app page.
pub struct Files {
    fs: SharedFs,
    area: Rect,
    /// The directory currently shown (absolute, normalised).
    path: String,
    /// The currently selected row index (for highlight), if any.
    selected: Option<usize>,
    /// Vertical scroll of the file list.
    scroll: Scroll,
    /// File search query. Only rows matching the query (case-insensitive) are shown.
    search: TextBuffer,
    search_focused: bool,
    now_ms: u64,
    /// When Some, the index of the row being renamed and the editing buffer.
    renaming: Option<(usize, TextBuffer)>,
    last_left: bool,
    damage: Option<Rect>,
}

impl Files {
    pub fn new(fs: SharedFs) -> Files {
        Files {
            fs,
            area: Rect::new(0, 0, 1280, 600),
            path: "/home/jayden".to_string(),
            selected: None,
            scroll: Scroll::new(),
            search: TextBuffer::empty(),
            search_focused: false,
            now_ms: 0,
            renaming: None,
            last_left: false,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        }
    }

    /// Pixel height of the full filtered file list (for the scroll model).
    fn content_h(&self) -> i32 {
        let q = self.search.text();
        let q = q.trim().to_lowercase();
        let entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
        let n = if q.is_empty() {
            entries.len()
        } else {
            entries.iter().filter(|e| e.name.to_lowercase().contains(q.as_str())).count()
        };
        n as i32 * ROW_H + 16
    }

    /// The directory currently shown.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Navigate to an absolute directory (no-op if it is not a directory).
    pub fn navigate(&mut self, path: &str) {
        if self.fs.borrow().is_dir(path) {
            self.path = self.fs.borrow().normalize(path);
            self.selected = None;
            self.scroll.offset = 0; // a new folder starts at the top
            self.dmg_all();
        }
    }

    pub fn set_area(&mut self, area: Rect) {
        if area != self.area {
            self.area = area;
            // Re-clamp the scroll so shrinking the window can't strand the list past its
            // (now shorter) content — the file list reflows cleanly on resize.
            let l = self.list_area();
            self.scroll.clamp(self.content_h(), l.h);
            self.dmg_all();
        }
    }

    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    /// Files wants text input when the search box or rename field is focused.
    pub fn wants_text(&self) -> bool {
        self.search_focused || self.renaming.is_some()
    }
    pub fn on_key(&mut self, ch: char) -> bool {
        // Rename mode: route to rename buffer
        if let Some((_, ref mut buf)) = self.renaming {
            match ch {
                '\n' | '\r' => {
                    // Commit handled in commit_rename below; signal with Esc too
                    let _ = self.commit_rename();
                }
                '\x1b' => { self.renaming = None; self.dmg_all(); }
                '\x08' => { buf.backspace(); self.dmg_all(); }
                c if !c.is_control() => { buf.insert(c); self.dmg_all(); }
                _ => {}
            }
            return true;
        }
        // Search box
        if self.search_focused {
            match ch {
                '\x1b' => { self.search_focused = false; self.search = TextBuffer::empty(); self.dmg_all(); }
                '\x08' => { self.search.backspace(); self.dmg_all(); }
                '\n' | '\r' => { self.search_focused = false; self.dmg_all(); }
                c if !c.is_control() => { self.search.insert(c); self.dmg_all(); }
                _ => {}
            }
            return true;
        }
        false
    }
    pub fn set_time(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
    }

    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }

    /// Begin renaming the currently selected entry.
    pub fn start_rename(&mut self) {
        if let Some(idx) = self.selected {
            let entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
            if let Some(e) = entries.get(idx) {
                let mut buf = TextBuffer::empty();
                for c in e.name.chars() { buf.insert(c); }
                self.renaming = Some((idx, buf));
                self.dmg_all();
            }
        }
    }

    /// Commit the rename in progress; returns the (old_path, new_path) on success.
    pub fn commit_rename(&mut self) -> Option<(String, String)> {
        if let Some((idx, ref buf)) = self.renaming.take() {
            let new_name = buf.text();
            let new_name = new_name.trim();
            if new_name.is_empty() { self.dmg_all(); return None; }
            let entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
            if let Some(e) = entries.get(idx) {
                let old = filesystem::join(&self.path, &e.name);
                let new = filesystem::join(&self.path, new_name);
                if old != new {
                    let ok = if e.is_dir {
                        // Recursively copy the entire subtree to the new path, then
                        // remove the old subtree.  A plain mkdir would leave all children
                        // orphaned under the old (now unlisted) path — the data-loss bug.
                        rename_dir_recursive(&self.fs, &old, &new)
                    } else {
                        // Read into an owned value first so the immutable borrow is
                        // dropped before we take the mutable borrows below (an if-let
                        // scrutinee holds its Ref for the whole block → double-borrow).
                        let content = self.fs.borrow().read_text(&old);
                        if let Some(content) = content {
                            let wrote = self.fs.borrow_mut().write_text(&new, &content).is_ok();
                            wrote && self.fs.borrow_mut().remove(&old).is_ok()
                        } else {
                            false
                        }
                    };
                    self.dmg_all();
                    if ok { return Some((old, new)); }
                }
            }
            self.dmg_all();
        }
        None
    }

    // ── layout ──

    fn toolbar(&self) -> Rect {
        Rect::new(0, 0, self.area.w, TOOLBAR_H)
    }
    fn sidebar(&self) -> Rect {
        Rect::new(0, TOOLBAR_H, SIDEBAR_W, self.area.h - TOOLBAR_H)
    }
    fn search_rect(&self) -> Rect {
        Rect::new(SIDEBAR_W + 8, TOOLBAR_H + 4, self.area.w - SIDEBAR_W - 16, SEARCH_H - 4)
    }
    fn list_area(&self) -> Rect {
        Rect::new(SIDEBAR_W, TOOLBAR_H + SEARCH_H, self.area.w - SIDEBAR_W, self.area.h - TOOLBAR_H - SEARCH_H)
    }
    fn up_btn(&self) -> Rect {
        Rect::new(8, 6, 52, TOOLBAR_H - 12)
    }
    fn newfolder_btn(&self) -> Rect {
        Rect::new(self.area.w - 116, 6, 108, TOOLBAR_H - 12)
    }
    fn quick_item(&self, i: usize) -> Rect {
        let s = self.sidebar();
        Rect::new(s.x + 8, s.y + 12 + i as i32 * 32, s.w - 16, 28)
    }
    fn row_rect(&self, i: usize) -> Rect {
        let l = self.list_area();
        Rect::new(l.x + 8, l.y + 8 + i as i32 * ROW_H - self.scroll.offset, l.w - 16 - crate::widgets::SCROLLBAR_W, ROW_H - 2)
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) -> Option<FilesAction> {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        let l = self.list_area();
        let content_h = self.content_h();

        // A scrollbar drag in progress tracks the pointer until release.
        if self.scroll.is_dragging() {
            if left {
                self.scroll.on_drag(py, l, content_h, l.h);
                self.dmg(l);
            } else {
                self.scroll.release();
            }
            self.last_left = left;
            return None;
        }
        self.last_left = left;
        if released || !pressed {
            return None;
        }
        // Scrollbar press (thumb grab or track page).
        if self.scroll.on_press(px, py, l, content_h, l.h) {
            self.dmg(l);
            return None;
        }
        // Toolbar: Up / New Folder.
        if self.up_btn().contains(px, py) {
            let parent = filesystem::parent(&self.path);
            self.navigate(&parent);
            return None;
        }
        if self.newfolder_btn().contains(px, py) {
            self.make_new_folder();
            return None;
        }
        // Sidebar quick access.
        if self.sidebar().contains(px, py) {
            for (i, (_, target)) in QUICK.iter().enumerate() {
                if self.quick_item(i).contains(px, py) {
                    self.navigate(target);
                    return None;
                }
            }
            return None;
        }
        // Search box click.
        if self.search_rect().contains(px, py) {
            self.search_focused = true;
            self.dmg(self.search_rect());
            return None;
        }
        // Clicking outside search box defocuses it.
        if self.search_focused {
            self.search_focused = false;
            self.dmg(self.search_rect());
        }

        // File/folder rows (filtered by search query).
        let q = self.search.text();
        let q = q.trim().to_lowercase();
        let all_entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
        let entries: Vec<_> = if q.is_empty() {
            all_entries.iter().cloned().collect()
        } else {
            all_entries.iter().filter(|e| e.name.to_lowercase().contains(q.as_str())).cloned().collect()
        };
        for (i, e) in entries.iter().enumerate() {
            if self.row_rect(i).contains(px, py) {
                // Map filtered index back to the original index for selection tracking
                let orig_idx = all_entries.iter().position(|x| x.name == e.name).unwrap_or(i);
                self.selected = Some(orig_idx);
                let child = filesystem::join(&self.path, &e.name);
                if e.is_dir {
                    self.navigate(&child);
                    return None;
                } else {
                    self.dmg(self.list_area());
                    return Some(FilesAction::OpenFile(child));
                }
            }
        }
        None
    }

    // ── right-click context operations (driven by the shell's context menu) ──

    /// Select whatever row sits under a page-local point (a right-click), returning the
    /// path of the item there, or `None` for empty space. Used to target Open/Delete.
    pub fn context_path_at(&mut self, lx: i32, ly: i32) -> Option<String> {
        let entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
        for (i, e) in entries.iter().enumerate() {
            if self.row_rect(i).contains(lx, ly) {
                self.selected = Some(i);
                self.dmg(self.list_area());
                return Some(filesystem::join(&self.path, &e.name));
            }
        }
        self.selected = None;
        self.dmg(self.list_area());
        None
    }

    /// The path of the currently-selected entry, if any.
    pub fn selected_path(&self) -> Option<String> {
        let entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
        self.selected.and_then(|i| entries.get(i).map(|e| filesystem::join(&self.path, &e.name)))
    }
    /// Whether the current selection is a directory.
    pub fn selected_is_dir(&self) -> bool {
        let entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
        self.selected.and_then(|i| entries.get(i).map(|e| e.is_dir)).unwrap_or(false)
    }

    /// Create a uniquely-named empty "New File.txt" in the current directory.
    pub fn new_file(&mut self) {
        let mut name = String::from("New File.txt");
        let mut n = 2;
        while self.fs.borrow().exists(&filesystem::join(&self.path, &name)) {
            name = String::from("New File ");
            push_int(&mut name, n);
            name.push_str(".txt");
            n += 1;
        }
        let target = filesystem::join(&self.path, &name);
        let _ = self.fs.borrow_mut().write_text(&target, "");
        self.dmg_all();
    }

    /// Create a new folder (public entry point for the context menu).
    pub fn new_folder(&mut self) {
        self.make_new_folder();
    }

    /// Delete the currently-selected file or folder. Returns whether it was removed.
    pub fn delete_selected(&mut self) -> bool {
        if let Some(path) = self.selected_path() {
            let ok = self.fs.borrow_mut().remove(&path).is_ok();
            self.selected = None;
            self.dmg_all();
            return ok;
        }
        false
    }

    /// Refresh the view (re-read the directory). The list is always read live from the
    /// VFS, so this just forces a repaint.
    pub fn refresh(&mut self) {
        self.dmg_all();
    }

    /// Create a uniquely-named "New Folder" in the current directory.
    fn make_new_folder(&mut self) {
        let mut name = String::from("New Folder");
        let mut n = 2;
        while self.fs.borrow().exists(&filesystem::join(&self.path, &name)) {
            name = String::from("New Folder ");
            push_int(&mut name, n);
            n += 1;
        }
        let target = filesystem::join(&self.path, &name);
        let _ = self.fs.borrow_mut().mkdir(&target);
        self.dmg_all();
    }

    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => toolkit::union(d, r),
            None => r,
        });
    }

    // ── rendering ──

    pub fn view(&self, t: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: t.bg, radius: 0 });
        self.draw_toolbar(&mut s, t);
        self.draw_sidebar(&mut s, t);
        self.draw_search(&mut s, t);
        self.draw_list(&mut s, t);
        s
    }

    fn draw_toolbar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = self.toolbar();
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        // Up button.
        let up = self.up_btn();
        s.push(DrawCmd::Rect { rect: up, color: t.primary, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(up.x + 12, up.y + 6, up.w, 16), text: "↑ Up".into(), color: t.on_primary, size: 13 });
        // Breadcrumb path (ellipsised to fit between the Up and New-Folder buttons).
        let crumb_x = up.x + up.w + 12;
        let crumb_w = (self.newfolder_btn().x - crumb_x - 12).max(0);
        let crumb = toolkit::ellipsize_px(&self.path, crumb_w, 13);
        s.push(DrawCmd::Text { rect: Rect::new(crumb_x, bar.y + 11, crumb_w, 16), text: crumb, color: t.text, size: 13 });
        // New Folder button.
        let nf = self.newfolder_btn();
        s.push(DrawCmd::Rect { rect: nf, color: t.surface, radius: t.radius });
        s.push(DrawCmd::Rect { rect: toolkit::inflate(nf, 1), color: t.accent, radius: t.radius });
        s.push(DrawCmd::Rect { rect: nf, color: t.surface, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(nf.x + 10, nf.y + 6, nf.w, 16), text: "+ New Folder".into(), color: t.text, size: 12 });
    }

    fn draw_sidebar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = self.sidebar();
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        s.push(DrawCmd::Text { rect: Rect::new(bar.x + 14, bar.y - 2, bar.w - 20, 14), text: "Quick access".into(), color: t.muted, size: 11 });
        for (i, (label, target)) in QUICK.iter().enumerate() {
            let r = self.quick_item(i);
            let active = self.path == *target;
            if active {
                s.push(DrawCmd::Rect { rect: r, color: t.primary, radius: t.radius });
            }
            let fg = if active { t.on_primary } else { t.text };
            glyph_folder(s, r.x + 10, r.y + r.h / 2, if active { t.on_primary } else { t.accent });
            s.push(DrawCmd::Text { rect: Rect::new(r.x + 26, r.y + 6, r.w - 30, 16), text: (*label).into(), color: fg, size: 12 });
        }
    }

    fn draw_search(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let r = self.search_rect();
        let border = if self.search_focused { t.primary } else { t.muted };
        s.push(DrawCmd::Rect { rect: toolkit::inflate(r, 1), color: border, radius: t.radius });
        s.push(DrawCmd::Rect { rect: r, color: t.bg, radius: t.radius });
        let query = self.search.text();
        let display = if query.trim().is_empty() && !self.search_focused {
            toolkit::ellipsize_px("🔍 Search files…", r.w - 16, 12)
        } else {
            toolkit::ellipsize_px(&query, r.w - 16, 12)
        };
        let fg = if query.trim().is_empty() && !self.search_focused { t.muted } else { t.text };
        s.push(DrawCmd::Text { rect: Rect::new(r.x + 8, r.y + 5, r.w - 16, 14), text: display, color: fg, size: 12 });
        if self.search_focused {
            let advance = toolkit::mono_advance(12);
            self.search.paint_caret(s, t, (r.x + 8, r.y + 4), advance, r.h - 8, self.now_ms);
        }
    }

    fn draw_list(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let l = self.list_area();
        let q = self.search.text();
        let q = q.trim().to_lowercase();
        // Read the directory once and build the filtered view as (orig_idx, entry)
        // pairs in a single pass, so there is no per-row position() scan.
        let all_entries = self.fs.borrow().entries(&self.path).unwrap_or_default();
        let entries: Vec<(usize, _)> = all_entries
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, e)| q.is_empty() || e.name.to_lowercase().contains(q.as_str()))
            .collect();
        if entries.is_empty() {
            let msg = if q.is_empty() { "(empty folder)" } else { "(no matches)" };
            s.push(DrawCmd::Text { rect: Rect::new(l.x + 16, l.y + 16, l.w - 24, 16), text: msg.into(), color: t.muted, size: 13 });
            return;
        }
        let content_h = entries.len() as i32 * ROW_H + 16;
        // Draw every row at its scrolled position, culling those outside the viewport.
        for (i, (orig_idx, e)) in entries.iter().enumerate() {
            let orig_idx = *orig_idx;
            let r = self.row_rect(i);
            if r.y + r.h <= l.y || r.y >= l.y + l.h {
                continue; // off-screen
            }
            if self.selected == Some(orig_idx) {
                s.push(DrawCmd::Rect { rect: r, color: Color::rgba(t.primary.r, t.primary.g, t.primary.b, 50), radius: t.radius });
            }
            // Rename overlay: show inline text field over this row
            if let Some((ri, ref buf)) = self.renaming {
                if ri == orig_idx {
                    let input_rect = Rect::new(r.x + 26, r.y + 3, r.w - 140, r.h - 6);
                    s.push(DrawCmd::Rect { rect: toolkit::inflate(input_rect, 1), color: t.primary, radius: t.radius });
                    s.push(DrawCmd::Rect { rect: input_rect, color: t.bg, radius: t.radius });
                    let txt = toolkit::ellipsize_px(&buf.text(), input_rect.w - 8, 13);
                    s.push(DrawCmd::Text { rect: Rect::new(input_rect.x + 4, input_rect.y + 4, input_rect.w - 8, 14), text: txt, color: t.text, size: 13 });
                    let advance = toolkit::mono_advance(13);
                    buf.paint_caret(s, t, (input_rect.x + 4, input_rect.y + 2), advance, input_rect.h - 4, self.now_ms);
                    if e.is_dir { glyph_folder(s, r.x + 12, r.y + r.h / 2, t.accent); } else { glyph_file(s, r.x + 12, r.y + r.h / 2, t.muted); }
                    continue;
                }
            }
            if e.is_dir {
                glyph_folder(s, r.x + 12, r.y + r.h / 2, t.accent);
            } else {
                glyph_file(s, r.x + 12, r.y + r.h / 2, t.muted);
            }
            // Name column, ellipsised so long names don't collide with the size column.
            let name_w = r.w - 30 - 116;
            let name = toolkit::ellipsize_px(&e.name, name_w, 13);
            s.push(DrawCmd::Text { rect: Rect::new(r.x + 30, r.y + 7, name_w, 16), text: name, color: t.text, size: 13 });
            // Right-aligned type / size column.
            let info = if e.is_dir {
                "Folder".to_string()
            } else {
                let mut sz = String::new();
                push_int(&mut sz, e.size as i64);
                sz.push_str(" B");
                sz
            };
            s.push(DrawCmd::Text { rect: Rect::new(r.x + r.w - 110, r.y + 7, 104, 16), text: info, color: t.muted, size: 12 });
        }
        // Scrollbar (reuse the already-computed content height).
        s.extend(crate::widgets::scrollbar(l, content_h, l.h, &self.scroll, t));
    }
}

/// Recursively copy the directory subtree at `old` to `new`, then remove the
/// old subtree.  This is the correct implementation of directory rename: because
/// children are not keyed on the parent path in the VFS, we must walk the tree
/// and recreate every node under the new name before tearing down the old one.
///
/// Returns `true` if the entire operation succeeded (new tree created AND old
/// tree fully removed).  On any error the new tree may be partially created, but
/// the old tree is left intact for everything that was not yet removed — so the
/// worst outcome is a partial duplicate, not silent data loss.
fn rename_dir_recursive(fs: &crate::filesystem::SharedFs, old: &str, new: &str) -> bool {
    // 1. Create the destination directory.
    if fs.borrow_mut().mkdir(new).is_err() {
        return false;
    }
    // 2. Copy every child into the new directory.
    let children = fs.borrow().entries(old).unwrap_or_default();
    for child in &children {
        let old_child = filesystem::join(old, &child.name);
        let new_child = filesystem::join(new, &child.name);
        if child.is_dir {
            if !rename_dir_recursive(fs, &old_child, &new_child) {
                return false;
            }
        } else {
            let content = match fs.borrow().read_text(&old_child) {
                Some(c) => c,
                None => return false,
            };
            if fs.borrow_mut().write_text(&new_child, &content).is_err() {
                return false;
            }
            if fs.borrow_mut().remove(&old_child).is_err() {
                return false;
            }
        }
    }
    // 3. Remove the now-empty old directory.
    fs.borrow_mut().remove(old).is_ok()
}

/// A little folder glyph centred at `(cx,cy)`.
fn glyph_folder(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 7, cy - 4, 14, 9), color: c, radius: 2 });
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 7, cy - 6, 6, 3), color: c, radius: 1 });
}

/// A little file/page glyph centred at `(cx,cy)`.
fn glyph_file(s: &mut Vec<DrawCmd>, cx: i32, cy: i32, c: Color) {
    s.push(DrawCmd::Rect { rect: Rect::new(cx - 5, cy - 7, 11, 14), color: c, radius: 1 });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filesystem::FileSystem;

    fn files() -> Files {
        let fs = FileSystem::shared();
        let mut f = Files::new(fs);
        f.set_area(Rect::new(0, 0, 1280, 600));
        let _ = f.take_damage();
        f
    }

    #[test]
    fn renders_toolbar_sidebar_and_seeded_entries() {
        let f = files();
        let s = f.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "↑ Up")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Quick access")));
        // The seeded home folders appear as rows.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Documents")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Projects")));
    }

    #[test]
    fn clicking_a_folder_navigates_into_it() {
        let mut f = files();
        // Find the Documents row and click it.
        let entries = f.fs.borrow().entries(f.path()).unwrap();
        let idx = entries.iter().position(|e| e.name == "Documents").unwrap();
        let r = f.row_rect(idx);
        let act = f.on_pointer(r.x + 20, r.y + 10, true);
        f.on_pointer(r.x + 20, r.y + 10, false);
        assert_eq!(act, None);
        assert_eq!(f.path(), "/home/jayden/Documents");
    }

    #[test]
    fn clicking_a_file_asks_the_shell_to_open_it() {
        let mut f = files();
        f.navigate("/home/jayden/Documents");
        let entries = f.fs.borrow().entries(f.path()).unwrap();
        let idx = entries.iter().position(|e| e.name == "welcome.txt").unwrap();
        let r = f.row_rect(idx);
        let act = f.on_pointer(r.x + 20, r.y + 10, true);
        assert_eq!(act, Some(FilesAction::OpenFile("/home/jayden/Documents/welcome.txt".to_string())));
    }

    #[test]
    fn up_button_navigates_to_parent() {
        let mut f = files();
        f.navigate("/home/jayden/Projects");
        let up = f.up_btn();
        f.on_pointer(up.x + 5, up.y + 5, true);
        f.on_pointer(up.x + 5, up.y + 5, false);
        assert_eq!(f.path(), "/home/jayden");
    }

    #[test]
    fn new_folder_button_creates_a_folder() {
        let mut f = files();
        f.navigate("/home/jayden/Downloads");
        let nf = f.newfolder_btn();
        f.on_pointer(nf.x + 5, nf.y + 5, true);
        f.on_pointer(nf.x + 5, nf.y + 5, false);
        assert!(f.fs.borrow().is_dir("/home/jayden/Downloads/New Folder"));
    }

    #[test]
    fn a_long_file_list_scrolls_via_the_scrollbar() {
        let fs = FileSystem::shared();
        let mut f = Files::new(fs.clone());
        f.set_area(Rect::new(0, 0, 1000, 400));
        fs.borrow_mut().mkdir("/home/jayden/Big").unwrap();
        for i in 0..40 {
            let mut name = String::from("/home/jayden/Big/file");
            push_int(&mut name, i);
            name.push_str(".txt");
            fs.borrow_mut().write_text(&name, "x").unwrap();
        }
        f.navigate("/home/jayden/Big");
        let _ = f.take_damage();
        let l = f.list_area();
        assert!(Scroll::needed(f.content_h(), l.h)); // overflows
        // Grab the thumb at the top and drag it to the bottom.
        let track = Scroll::track(l);
        f.on_pointer(track.x + 2, track.y + 2, true);
        f.on_pointer(track.x + 2, track.y + l.h, true);
        f.on_pointer(track.x + 2, track.y + l.h, false);
        assert_eq!(f.scroll.offset, Scroll::max_offset(f.content_h(), l.h));
        assert!(f.scroll.offset > 0);
    }

    #[test]
    fn shrinking_the_window_reclamps_the_scroll() {
        let fs = FileSystem::shared();
        let mut f = Files::new(fs.clone());
        f.set_area(Rect::new(0, 0, 1000, 800));
        fs.borrow_mut().mkdir("/home/jayden/Many").unwrap();
        for i in 0..40 {
            let mut name = String::from("/home/jayden/Many/f");
            push_int(&mut name, i);
            fs.borrow_mut().write_text(&name, "x").unwrap();
        }
        f.navigate("/home/jayden/Many");
        // Scroll to the bottom in the tall window.
        let l = f.list_area();
        f.scroll.offset = Scroll::max_offset(f.content_h(), l.h);
        let tall_max = f.scroll.offset;
        // Shrink the window: the offset must not exceed the new (smaller) maximum.
        f.set_area(Rect::new(0, 0, 1000, 300));
        let l2 = f.list_area();
        assert!(f.scroll.offset <= Scroll::max_offset(f.content_h(), l2.h));
        assert!(f.scroll.offset <= tall_max);
    }

    #[test]
    fn long_names_are_ellipsised_not_clipped() {
        let fs = FileSystem::shared();
        let mut f = Files::new(fs.clone());
        f.set_area(Rect::new(0, 0, 600, 400)); // narrow → names must shrink
        fs.borrow_mut().mkdir("/home/jayden/E").unwrap();
        fs.borrow_mut()
            .write_text("/home/jayden/E/an-extremely-long-file-name-that-overflows.txt", "x")
            .unwrap();
        f.navigate("/home/jayden/E");
        let s = f.view(&Theme::dark());
        // The rendered name is truncated with an ellipsis.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains('…'))));
    }

    #[test]
    fn quick_access_jumps_to_root() {
        let mut f = files();
        let r = f.quick_item(5); // "System (/)"
        f.on_pointer(r.x + 5, r.y + 5, true);
        f.on_pointer(r.x + 5, r.y + 5, false);
        assert_eq!(f.path(), "/");
    }

    /// Renaming a directory must move all of its children to the new path.
    ///
    /// Before the fix, `commit_rename` called `mkdir` on the new path and returned
    /// without touching the children, leaving them orphaned under the old (now
    /// absent) directory.  This test catches that data-loss bug.
    #[test]
    fn rename_dir_moves_children_not_orphans_them() {
        let fs = FileSystem::shared();
        let mut f = Files::new(fs.clone());
        f.set_area(Rect::new(0, 0, 1280, 600));

        // Create /home/jayden/Alpha with two files and a nested subdirectory.
        fs.borrow_mut().mkdir("/home/jayden/Alpha").unwrap();
        fs.borrow_mut().write_text("/home/jayden/Alpha/file_a.txt", "hello").unwrap();
        fs.borrow_mut().write_text("/home/jayden/Alpha/file_b.txt", "world").unwrap();
        fs.borrow_mut().mkdir("/home/jayden/Alpha/Sub").unwrap();
        fs.borrow_mut().write_text("/home/jayden/Alpha/Sub/nested.txt", "deep").unwrap();

        // Navigate into /home/jayden, select "Alpha", and commit a rename to "Beta".
        f.navigate("/home/jayden");
        let entries = f.fs.borrow().entries(f.path()).unwrap();
        let idx = entries.iter().position(|e| e.name == "Alpha").unwrap();
        f.selected = Some(idx);
        f.start_rename();
        // Replace the rename buffer content with "Beta".
        if let Some((_, ref mut buf)) = f.renaming {
            // Clear the existing "Alpha" content and type "Beta".
            for _ in 0.."Alpha".len() { buf.backspace(); }
            for c in "Beta".chars() { buf.insert(c); }
        }
        let result = f.commit_rename();

        // The rename must have succeeded and reported the correct paths.
        assert_eq!(result, Some(("/home/jayden/Alpha".to_string(), "/home/jayden/Beta".to_string())));

        // Beta must exist and contain all children.
        assert!(fs.borrow().is_dir("/home/jayden/Beta"),
            "Beta directory not created");
        assert_eq!(fs.borrow().read_text("/home/jayden/Beta/file_a.txt").as_deref(), Some("hello"),
            "file_a.txt missing under Beta");
        assert_eq!(fs.borrow().read_text("/home/jayden/Beta/file_b.txt").as_deref(), Some("world"),
            "file_b.txt missing under Beta");
        assert!(fs.borrow().is_dir("/home/jayden/Beta/Sub"),
            "Sub subdirectory not moved");
        assert_eq!(fs.borrow().read_text("/home/jayden/Beta/Sub/nested.txt").as_deref(), Some("deep"),
            "nested.txt missing under Beta/Sub");

        // The old Alpha directory and all its children must no longer exist.
        assert!(!fs.borrow().exists("/home/jayden/Alpha"),
            "Alpha still exists after rename");
        assert!(!fs.borrow().exists("/home/jayden/Alpha/file_a.txt"),
            "file_a.txt still accessible under old Alpha path");
        assert!(!fs.borrow().exists("/home/jayden/Alpha/Sub"),
            "Sub still accessible under old Alpha path");
    }
}

//! The **Editor** app page — wraps the universal [`Editor`](crate::editor)
//! (Notepad++ ⊕ Vim ⊕ live calculator) with a title bar, a mode indicator and a
//! **Save** button that writes back to the shared [`FileSystem`](crate::filesystem).
//!
//! Opening a file from the Files app loads it here; saving stores a new immutable
//! object in the graph and re-aliases the path (so the Terminal can immediately `cat`
//! the new content). Following the page contract, it reports `wants_text` so every key
//! reaches the buffer — including Esc, which the Vim modality uses to leave Insert
//! mode rather than quitting the desktop. Pure, safe `no_std`.

use crate::editor::{Editor, Mode};
use crate::filesystem::{basename, SharedFs};
use crate::text::BLINK_MS;
use crate::toolkit::{self, DrawCmd, Rect, Theme};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const TITLE_H: i32 = 34;

/// The Editor app page.
pub struct EditorPage {
    fs: SharedFs,
    editor: Editor,
    /// The file currently open (absolute path), or `None` for an unsaved scratch buffer.
    path: Option<String>,
    /// Whether the buffer has unsaved edits since the last save/open.
    dirty: bool,
    /// Live-notebook evaluation preference (Settings).
    live_eval: bool,
    /// Whether opening a file drops straight into insert mode (Settings).
    insert_default: bool,
    area: Rect,
    now_ms: u64,
    last_left: bool,
    /// True while a mouse-drag selection is in progress.
    dragging: bool,
    damage: Option<Rect>,
}

impl EditorPage {
    pub fn new(fs: SharedFs) -> EditorPage {
        EditorPage {
            fs,
            editor: Editor::new("Welcome to the universal editor — a live Dominion notebook.\n\nType maths or code; it evaluates as you edit. Variables carry\nbetween lines, a trailing = computes, and for-loops repeat:\n\ntest_val = 21 * 5 + 4\ntest_val + 4\n\n21 * 5 + 4 =\n\nfor 3\n    test_val + index\n"),
            path: None,
            dirty: false,
            live_eval: true,
            insert_default: false,
            area: Rect::new(0, 0, 1280, 600),
            now_ms: 0,
            last_left: false,
            dragging: false,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        }
    }

    /// Toggle live-notebook evaluation (Settings preference).
    pub fn set_live_eval(&mut self, on: bool) {
        self.live_eval = on;
        self.editor.set_live_eval(on);
        self.dmg_all();
    }
    /// Whether opening a file should drop straight into insert mode (Settings preference).
    pub fn set_insert_default(&mut self, on: bool) {
        self.insert_default = on;
    }

    /// Copy the editor's selection to give to the shell clipboard.
    pub fn copy(&self) -> Option<String> {
        self.editor.copy()
    }
    /// Cut the editor's selection.
    pub fn cut(&mut self) -> Option<String> {
        let t = self.editor.cut();
        if t.is_some() {
            self.dirty = true;
            self.dmg_all();
        }
        t
    }
    /// Paste text from the shell clipboard at the caret.
    pub fn paste(&mut self, s: &str) {
        self.editor.paste(s);
        self.dirty = true;
        self.dmg_all();
    }

    /// Open a file: load its content into the editor and remember its path.
    pub fn open(&mut self, path: &str) {
        let content = self.fs.borrow().read_text(path).unwrap_or_default();
        self.editor = Editor::new(&content);
        self.editor.set_live_eval(self.live_eval);
        if self.insert_default {
            self.editor.key('i');
        }
        self.editor.tick(self.now_ms);
        self.path = Some(path.to_string());
        self.dirty = false;
        self.dmg_all();
    }

    /// Save the buffer back to its path (no-op for an unsaved scratch buffer).
    pub fn save(&mut self) -> bool {
        if let Some(path) = self.path.clone() {
            if self.fs.borrow_mut().write_text(&path, &self.editor.text()).is_ok() {
                self.dirty = false;
                self.dmg(self.title_bar());
                return true;
            }
        }
        false
    }

    /// Save the buffer to a new path (Save As). Sets this as the new working path.
    pub fn save_as(&mut self, new_path: &str) -> bool {
        if new_path.trim().is_empty() {
            return false;
        }
        if self.fs.borrow_mut().write_text(new_path, &self.editor.text()).is_ok() {
            self.path = Some(new_path.to_string());
            self.dirty = false;
            self.dmg_all();
            return true;
        }
        false
    }

    /// The path of the open file, if any.
    pub fn open_path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    pub fn set_area(&mut self, area: Rect) {
        if area != self.area {
            self.area = area;
            self.dmg_all();
        }
    }
    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    pub fn wants_text(&self) -> bool {
        true
    }

    pub fn set_time(&mut self, now_ms: u64) {
        let prev = self.now_ms;
        self.now_ms = now_ms;
        self.editor.tick(now_ms);
        if prev / BLINK_MS != now_ms / BLINK_MS {
            self.dmg(self.editor_area());
        }
    }

    pub fn on_key(&mut self, ch: char) -> bool {
        self.editor.key(ch);
        self.dirty = true;
        self.dmg_all();
        true
    }

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        let ea = self.editor_area();
        if pressed {
            self.last_left = left;
            if self.save_btn().contains(px, py) {
                self.save();
                return;
            }
            if ea.contains(px, py) {
                // Press anchors a selection at the click; a drag extends it. A plain
                // click (no drag) leaves an empty selection → just a placed caret.
                self.editor.begin_select(px, py, ea, 15);
                self.dragging = true;
                self.dmg_all();
            }
            return;
        }
        if left && self.dragging {
            self.editor.extend_select(px, py, ea, 15);
            self.dmg_all();
        } else if released {
            self.dragging = false;
        }
        self.last_left = left;
    }

    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => toolkit::union(d, r),
            None => r,
        });
    }
    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }

    fn title_bar(&self) -> Rect {
        Rect::new(0, 0, self.area.w, TITLE_H)
    }
    fn editor_area(&self) -> Rect {
        Rect::new(0, TITLE_H, self.area.w, self.area.h - TITLE_H)
    }
    fn save_btn(&self) -> Rect {
        Rect::new(self.area.w - 84, 5, 76, TITLE_H - 10)
    }

    pub fn view(&self, t: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: t.bg, radius: 0 });
        // Title bar.
        let bar = self.title_bar();
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        let mut title = match &self.path {
            Some(p) => basename(p).to_string(),
            None => "untitled".to_string(),
        };
        if self.dirty {
            title.push_str(" •");
        }
        s.push(DrawCmd::Text { rect: Rect::new(14, 9, self.area.w - 220, 16), text: title, color: t.text, size: 14 });
        // Mode indicator (Vim modality).
        let mode = match self.editor.mode() {
            Mode::Normal => "NORMAL",
            Mode::Insert => "INSERT",
            Mode::Visual => "VISUAL",
        };
        s.push(DrawCmd::Text { rect: Rect::new(self.area.w - 170, 9, 70, 16), text: mode.into(), color: t.muted, size: 12 });
        // Save button.
        let sb = self.save_btn();
        s.push(DrawCmd::Rect { rect: sb, color: t.primary, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(sb.x + 16, sb.y + 5, sb.w, 16), text: "Save".into(), color: t.on_primary, size: 13 });
        // Editor body.
        s.extend(self.editor.view(t, self.editor_area()));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filesystem::FileSystem;

    fn page() -> (EditorPage, SharedFs) {
        let fs = FileSystem::shared();
        let mut p = EditorPage::new(fs.clone());
        p.set_area(Rect::new(0, 0, 1000, 500));
        let _ = p.take_damage();
        (p, fs)
    }

    #[test]
    fn opens_a_file_and_shows_its_name_and_content() {
        let (mut p, _fs) = page();
        p.open("/home/jayden/Documents/welcome.txt");
        assert_eq!(p.open_path(), Some("/home/jayden/Documents/welcome.txt"));
        let s = p.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "welcome.txt")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Welcome to DominionOS"))));
    }

    #[test]
    fn editing_and_saving_writes_back_to_the_filesystem() {
        let (mut p, fs) = page();
        fs.borrow_mut().write_text("/tmp/edit.txt", "old").unwrap();
        p.open("/tmp/edit.txt");
        // Enter insert mode and type.
        p.on_key('i');
        for c in "new ".chars() {
            p.on_key(c);
        }
        p.save();
        assert!(fs.borrow().read_text("/tmp/edit.txt").unwrap().starts_with("new "));
    }

    #[test]
    fn inline_calculator_still_works_in_the_page() {
        let (p, _fs) = page();
        let s = p.view(&Theme::dark());
        // The seeded notebook binds `test_val = 21 * 5 + 4` → 109, shown inline.
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("109"))));
    }

    #[test]
    fn always_wants_text_input() {
        let (p, _fs) = page();
        assert!(p.wants_text());
    }
}

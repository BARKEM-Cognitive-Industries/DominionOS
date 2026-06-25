//! The **global text engine** — the one reusable text-editing core that *every*
//! text surface in the OS shares (see `docs/ui/text-engine.md`).
//!
//! Before this module, each place that handled text (the editor, node fragments,
//! input fields) reimplemented its own ad-hoc cursor handling, and most of them
//! could only ever append/backspace at the end — you could not click to place the
//! caret, move it with the arrow keys, or even see it blink. That is the opposite
//! of "normal" text behaviour.
//!
//! [`TextBuffer`] fixes that once, for everyone: a multi-line buffer with a caret
//! that can sit **anywhere**, full keyboard navigation (left/right/up/down, home/
//! end, word hops, document ends), **click-to-place** (pixel → caret), **selection**,
//! and a **blinking insert caret** whose visibility is a pure function of the clock
//! (so it needs no per-frame state and stays replay-deterministic). It renders the
//! caret as an ordinary [`toolkit::DrawCmd::Rect`], so any backend draws it for free.
//!
//! The universal [`crate::editor::Editor`] (its Vim modality), the
//! [`crate::terminal::Terminal`] input line, and toolkit text fields all sit on top
//! of this one engine. Pure, safe `no_std`, fully host-tested.

use crate::toolkit::{Color, DrawCmd, Rect, Theme};
use alloc::string::String;
use alloc::vec::Vec;

/// How long (ms) one half of the caret blink lasts. The caret is shown for
/// `BLINK_MS`, hidden for `BLINK_MS`, repeating — the familiar ~1 Hz flash.
pub const BLINK_MS: u64 = 500;

/// A caret position: `(row, col)` in **characters** (not bytes).
pub type Pos = (usize, usize);

/// A multi-line text buffer with a navigable, blinking caret and selection — the
/// shared heart of every editable text surface.
#[derive(Clone, Debug)]
pub struct TextBuffer {
    lines: Vec<String>,
    /// Caret column (chars) and row (line index).
    cx: usize,
    cy: usize,
    /// The selection anchor, if a selection is active. The selection spans from
    /// `anchor` to the caret (either order).
    anchor: Option<Pos>,
    /// The caret is held **solid** (no blink) until this time — set on every edit
    /// or motion so it doesn't flash away the instant you move it, exactly like a
    /// real editor. `0` means "blink normally".
    solid_until_ms: u64,
    /// Additional carets for multi-cursor editing (the primary is `(cy, cx)`).
    extra: Vec<Pos>,
}

impl TextBuffer {
    /// Open a buffer from initial text (split on `\n`). Caret starts at the top.
    pub fn new(text: &str) -> TextBuffer {
        let mut lines: Vec<String> = text.split('\n').map(String::from).collect();
        if lines.is_empty() {
            lines.push(String::new());
        }
        TextBuffer { lines, cx: 0, cy: 0, anchor: None, solid_until_ms: 0, extra: Vec::new() }
    }

    /// An empty single-line buffer (e.g. a fresh input field).
    pub fn empty() -> TextBuffer {
        TextBuffer::new("")
    }

    // ── inspection ──

    pub fn lines(&self) -> &[String] {
        &self.lines
    }
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
    /// The caret as `(row, col)`.
    pub fn caret(&self) -> Pos {
        (self.cy, self.cx)
    }
    pub fn row(&self) -> usize {
        self.cy
    }
    pub fn col(&self) -> usize {
        self.cx
    }
    /// The whole buffer as a single string.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }
    /// The current line's text.
    pub fn cur_line(&self) -> &str {
        &self.lines[self.cy]
    }
    /// Character length of the current line.
    pub fn cur_len(&self) -> usize {
        self.lines[self.cy].chars().count()
    }
    fn line_len(&self, row: usize) -> usize {
        self.lines[row].chars().count()
    }

    /// Byte offset of char column `col` on line `row` (for `String` mutation).
    fn byte_at(&self, row: usize, col: usize) -> usize {
        self.lines[row].char_indices().nth(col).map(|(b, _)| b).unwrap_or(self.lines[row].len())
    }

    /// Keep the caret inside the buffer (after external line edits).
    pub fn clamp(&mut self) {
        if self.cy >= self.lines.len() {
            self.cy = self.lines.len().saturating_sub(1);
        }
        let max = self.cur_len();
        if self.cx > max {
            self.cx = max;
        }
    }

    /// Replace the whole buffer's text, keeping the caret in bounds.
    pub fn set_text(&mut self, text: &str) {
        self.lines = text.split('\n').map(String::from).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.anchor = None;
        self.clamp();
    }

    /// Move the caret to an explicit `(row, col)`, clamped into the buffer.
    pub fn set_caret(&mut self, row: usize, col: usize) {
        self.cy = row.min(self.lines.len().saturating_sub(1));
        self.cx = col.min(self.cur_len());
        self.touch(0);
    }

    // ── editing ──

    /// Insert one character at the caret. Replaces the selection first, if any.
    pub fn insert(&mut self, ch: char) {
        self.delete_selection();
        let byte = self.byte_at(self.cy, self.cx);
        self.lines[self.cy].insert(byte, ch);
        self.cx += 1;
        self.touch(0);
    }

    /// Insert a string (may contain newlines).
    pub fn insert_str(&mut self, s: &str) {
        for ch in s.chars() {
            if ch == '\n' {
                self.newline();
            } else {
                self.insert(ch);
            }
        }
    }

    /// Split the current line at the caret (Enter).
    pub fn newline(&mut self) {
        self.delete_selection();
        let byte = self.byte_at(self.cy, self.cx);
        let tail = self.lines[self.cy].split_off(byte);
        self.lines.insert(self.cy + 1, tail);
        self.cy += 1;
        self.cx = 0;
        self.touch(0);
    }

    /// Delete the character before the caret (Backspace); joins lines at column 0.
    /// If a selection is active, deletes that instead.
    pub fn backspace(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        if self.cx > 0 {
            let byte = self.byte_at(self.cy, self.cx - 1);
            self.lines[self.cy].remove(byte);
            self.cx -= 1;
        } else if self.cy > 0 {
            let cur = self.lines.remove(self.cy);
            self.cy -= 1;
            self.cx = self.cur_len();
            self.lines[self.cy].push_str(&cur);
        }
        self.touch(0);
    }

    /// Delete the character **at** the caret (Delete/`x`); joins the next line at EOL.
    pub fn delete(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        if self.cx < self.cur_len() {
            let byte = self.byte_at(self.cy, self.cx);
            self.lines[self.cy].remove(byte);
        } else if self.cy + 1 < self.lines.len() {
            let next = self.lines.remove(self.cy + 1);
            self.lines[self.cy].push_str(&next);
        }
        self.touch(0);
    }

    /// Delete the whole current line (Vim `dd`).
    pub fn delete_line(&mut self) {
        if self.lines.len() > 1 {
            self.lines.remove(self.cy);
            if self.cy >= self.lines.len() {
                self.cy = self.lines.len() - 1;
            }
        } else {
            self.lines[0].clear();
        }
        self.cx = self.cx.min(self.cur_len());
        self.touch(0);
    }

    // ── motion (the "normal" arrow-key behaviour) ──

    /// Move left one character, wrapping to the end of the previous line.
    pub fn left(&mut self) {
        self.clear_selection();
        if self.cx > 0 {
            self.cx -= 1;
        } else if self.cy > 0 {
            self.cy -= 1;
            self.cx = self.cur_len();
        }
        self.touch(0);
    }

    /// Move right one character, wrapping to the start of the next line.
    pub fn right(&mut self) {
        self.clear_selection();
        if self.cx < self.cur_len() {
            self.cx += 1;
        } else if self.cy + 1 < self.lines.len() {
            self.cy += 1;
            self.cx = 0;
        }
        self.touch(0);
    }

    /// Move up one line, keeping the column where possible.
    pub fn up(&mut self) {
        self.clear_selection();
        if self.cy > 0 {
            self.cy -= 1;
            self.cx = self.cx.min(self.cur_len());
        } else {
            self.cx = 0;
        }
        self.touch(0);
    }

    /// Move down one line, keeping the column where possible.
    pub fn down(&mut self) {
        self.clear_selection();
        if self.cy + 1 < self.lines.len() {
            self.cy += 1;
            self.cx = self.cx.min(self.cur_len());
        } else {
            self.cx = self.cur_len();
        }
        self.touch(0);
    }

    /// Caret to the start of the line (Home / `0`).
    pub fn home(&mut self) {
        self.clear_selection();
        self.cx = 0;
        self.touch(0);
    }

    /// Caret to the end of the line (End / `$`).
    pub fn end(&mut self) {
        self.clear_selection();
        self.cx = self.cur_len();
        self.touch(0);
    }

    /// Caret to the very start of the buffer.
    pub fn doc_start(&mut self) {
        self.clear_selection();
        self.cy = 0;
        self.cx = 0;
        self.touch(0);
    }

    /// Caret to the very end of the buffer.
    pub fn doc_end(&mut self) {
        self.clear_selection();
        self.cy = self.lines.len() - 1;
        self.cx = self.cur_len();
        self.touch(0);
    }

    /// Hop to the start of the next word (Vim `w` / Ctrl-Right).
    pub fn next_word(&mut self) {
        self.clear_selection();
        let chars: Vec<char> = self.lines[self.cy].chars().collect();
        let mut i = self.cx;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() && self.cy + 1 < self.lines.len() {
            // Roll onto the next line's first word.
            self.cy += 1;
            self.cx = 0;
            let next: Vec<char> = self.lines[self.cy].chars().collect();
            let mut j = 0;
            while j < next.len() && next[j].is_whitespace() {
                j += 1;
            }
            self.cx = j;
        } else {
            self.cx = i.min(chars.len());
        }
        self.touch(0);
    }

    /// Hop to the start of the previous word (Ctrl-Left).
    pub fn prev_word(&mut self) {
        self.clear_selection();
        if self.cx == 0 {
            if self.cy > 0 {
                self.cy -= 1;
                self.cx = self.cur_len();
            }
            self.touch(0);
            return;
        }
        let chars: Vec<char> = self.lines[self.cy].chars().collect();
        let mut i = self.cx;
        while i > 0 && chars[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !chars[i - 1].is_whitespace() {
            i -= 1;
        }
        self.cx = i;
        self.touch(0);
    }

    /// **Click-to-place**: move the caret to the character nearest a pixel, where
    /// the text is laid out from `origin` with a fixed `char_w` advance and `line_h`
    /// row height (the mono font the renderer uses). `px,py` are screen pixels.
    /// Rounding to the *nearest* gap means a click between two glyphs lands on the
    /// closer side, like every real editor.
    pub fn place_at_pixel(&mut self, px: i32, py: i32, origin: (i32, i32), char_w: i32, line_h: i32) {
        self.clear_selection();
        let line_h = line_h.max(1);
        let char_w = char_w.max(1);
        let row = ((py - origin.1) / line_h).max(0) as usize;
        self.cy = row.min(self.lines.len().saturating_sub(1));
        let col = ((px - origin.0) + char_w / 2) / char_w;
        self.cx = (col.max(0) as usize).min(self.cur_len());
        self.touch(0);
    }

    // ── selection ──

    fn has_selection(&self) -> bool {
        self.anchor.map(|a| a != (self.cy, self.cx)).unwrap_or(false)
    }
    /// Begin (or keep) a selection anchored at the current caret — call before a
    /// shift-motion so the moved caret extends a highlight.
    pub fn begin_selection(&mut self) {
        if self.anchor.is_none() {
            self.anchor = Some((self.cy, self.cx));
        }
    }
    pub fn clear_selection(&mut self) {
        self.anchor = None;
    }
    /// Whether a (non-empty) selection is active — exposed so surfaces can decide
    /// whether copy/cut have anything to act on.
    pub fn is_selecting(&self) -> bool {
        self.has_selection()
    }

    /// Run a caret motion while **extending** the selection (the Shift+motion gesture):
    /// anchor at the current caret if no selection is active, move, and keep the anchor
    /// so the moved caret grows a highlight. The plain motions clear the selection; these
    /// preserve it.
    fn shift_move(&mut self, motion: fn(&mut TextBuffer)) {
        self.begin_selection();
        let a = self.anchor;
        motion(self);
        self.anchor = a;
    }
    pub fn select_left(&mut self) {
        self.shift_move(TextBuffer::left);
    }
    pub fn select_right(&mut self) {
        self.shift_move(TextBuffer::right);
    }
    pub fn select_up(&mut self) {
        self.shift_move(TextBuffer::up);
    }
    pub fn select_down(&mut self) {
        self.shift_move(TextBuffer::down);
    }
    pub fn select_home(&mut self) {
        self.shift_move(TextBuffer::home);
    }
    pub fn select_end(&mut self) {
        self.shift_move(TextBuffer::end);
    }
    /// Select the whole buffer (Ctrl+A).
    pub fn select_all(&mut self) {
        self.anchor = Some((0, 0));
        self.cy = self.lines.len().saturating_sub(1);
        self.cx = self.cur_len();
        self.touch(0);
    }

    /// **Begin a mouse selection** at a pixel: place the caret there and drop the
    /// selection anchor on it, so a following [`select_to_pixel`](Self::select_to_pixel)
    /// drag grows a highlight from the click point.
    pub fn begin_select_at_pixel(&mut self, px: i32, py: i32, origin: (i32, i32), char_w: i32, line_h: i32) {
        self.place_at_pixel(px, py, origin, char_w, line_h); // moves caret, clears anchor
        self.anchor = Some((self.cy, self.cx));
    }
    /// **Extend a mouse selection** to a pixel: move the caret there, keeping the anchor
    /// from [`begin_select_at_pixel`](Self::begin_select_at_pixel) so the range tracks
    /// the drag.
    pub fn select_to_pixel(&mut self, px: i32, py: i32, origin: (i32, i32), char_w: i32, line_h: i32) {
        let a = self.anchor.unwrap_or((self.cy, self.cx));
        self.place_at_pixel(px, py, origin, char_w, line_h);
        self.anchor = Some(a);
    }
    /// The selection as an ordered `(start, end)` pair, or `None`.
    pub fn selection(&self) -> Option<(Pos, Pos)> {
        let a = self.anchor?;
        let b = (self.cy, self.cx);
        if a == b {
            return None;
        }
        Some(if a <= b { (a, b) } else { (b, a) })
    }
    /// The selected text (joined with `\n` across lines).
    pub fn selected_text(&self) -> String {
        let Some((s, e)) = self.selection() else { return String::new() };
        if s.0 == e.0 {
            self.lines[s.0].chars().skip(s.1).take(e.1 - s.1).collect()
        } else {
            let mut out = String::new();
            for row in s.0..=e.0 {
                let (from, to) = if row == s.0 {
                    (s.1, self.line_len(row))
                } else if row == e.0 {
                    (0, e.1)
                } else {
                    (0, self.line_len(row))
                };
                let frag: String = self.lines[row].chars().skip(from).take(to - from).collect();
                out.push_str(&frag);
                if row != e.0 {
                    out.push('\n');
                }
            }
            out
        }
    }
    /// Delete the active selection (if any), collapsing the caret to its start.
    /// Returns whether anything was removed.
    pub fn delete_selection(&mut self) -> bool {
        let Some((s, e)) = self.selection() else {
            self.anchor = None;
            return false;
        };
        if s.0 == e.0 {
            let b0 = self.byte_at(s.0, s.1);
            let b1 = self.byte_at(s.0, e.1);
            self.lines[s.0].replace_range(b0..b1, "");
        } else {
            let head: String = self.lines[s.0].chars().take(s.1).collect();
            let tail: String = self.lines[e.0].chars().skip(e.1).collect();
            let merged = head + &tail;
            self.lines.drain(s.0..=e.0);
            self.lines.insert(s.0, merged);
        }
        self.cy = s.0;
        self.cx = s.1;
        self.anchor = None;
        self.touch(0);
        true
    }

    // ── caret blink ──

    /// Hold the caret solid (un-blinking) for `BLINK_MS` from `now_ms`. Pass the
    /// current time on edits/motions so the caret doesn't blink off the instant you
    /// type. Surfaces with no clock can pass `0` (pure blink).
    pub fn touch(&mut self, now_ms: u64) {
        if now_ms > 0 {
            self.solid_until_ms = now_ms + BLINK_MS;
        }
    }

    /// Is the caret currently visible? A pure function of the clock: solid right
    /// after activity, otherwise flashing on a `BLINK_MS` cycle. Deterministic, so
    /// replay reproduces the same frames.
    pub fn caret_visible(&self, now_ms: u64) -> bool {
        if now_ms < self.solid_until_ms {
            return true;
        }
        (now_ms / BLINK_MS).is_multiple_of(2)
    }

    /// The caret's pixel rectangle — a thin vertical bar — for the given layout
    /// (`origin`, mono `char_w`, `line_h`). Render it as a `Rect` in the theme's
    /// `primary` token when [`caret_visible`](Self::caret_visible) is true.
    pub fn caret_rect(&self, origin: (i32, i32), char_w: i32, line_h: i32) -> Rect {
        let x = origin.0 + self.cx as i32 * char_w;
        let y = origin.1 + self.cy as i32 * line_h;
        Rect::new(x, y, 2.max(char_w / 8), line_h)
    }

    /// Append the caret (and any selection highlight) to a scene, if visible. A
    /// convenience for surfaces that lay text out monospaced from `origin`.
    pub fn paint_caret(
        &self,
        scene: &mut Vec<DrawCmd>,
        theme: &Theme,
        origin: (i32, i32),
        char_w: i32,
        line_h: i32,
        now_ms: u64,
    ) {
        // Selection highlight (under the caret), one rect per covered row.
        if let Some((s, e)) = self.selection() {
            let hl = Color::rgba(theme.primary.r, theme.primary.g, theme.primary.b, 60);
            for row in s.0..=e.0 {
                let from = if row == s.0 { s.1 } else { 0 } as i32;
                let to = if row == e.0 { e.1 } else { self.line_len(row) } as i32;
                let x = origin.0 + from * char_w;
                let w = ((to - from).max(0) * char_w).max(char_w / 2);
                let y = origin.1 + row as i32 * line_h;
                scene.push(DrawCmd::Rect { rect: Rect::new(x, y, w, line_h), color: hl, radius: 0 });
            }
        }
        if self.caret_visible(now_ms) {
            scene.push(DrawCmd::Rect {
                rect: self.caret_rect(origin, char_w, line_h),
                color: theme.primary,
                radius: 0,
            });
        }
    }
}

impl TextBuffer {
    // ── clipboard: cut / copy / paste ──

    /// Copy the current selection (returns `None` if nothing is selected).
    pub fn copy(&self) -> Option<String> {
        self.selection().map(|_| self.selected_text())
    }

    /// Cut the selection: returns its text and deletes it (caret lands at the start).
    pub fn cut(&mut self) -> Option<String> {
        let text = self.copy()?;
        self.delete_selection();
        Some(text)
    }

    /// Paste `s` at the caret, replacing any active selection first.
    pub fn paste(&mut self, s: &str) {
        if self.selection().is_some() {
            self.delete_selection();
        }
        self.insert_str(s);
    }

    // ── find / replace ──

    /// Find the first occurrence of `needle` at or after `start`, scanning forward.
    /// Returns its `(row, col)`, or `None`. An empty needle never matches.
    pub fn find_from(&self, needle: &str, start: Pos) -> Option<Pos> {
        if needle.is_empty() {
            return None;
        }
        let needle_chars: Vec<char> = needle.chars().collect();
        for row in start.0..self.lines.len() {
            let line: Vec<char> = self.lines[row].chars().collect();
            let from_col = if row == start.0 { start.1 } else { 0 };
            if line.len() >= needle_chars.len() {
                for col in from_col..=line.len().saturating_sub(needle_chars.len()) {
                    if line[col..col + needle_chars.len()] == needle_chars[..] {
                        return Some((row, col));
                    }
                }
            }
        }
        None
    }

    /// Every occurrence of `needle`, in document order.
    pub fn find_all(&self, needle: &str) -> Vec<Pos> {
        let mut out = Vec::new();
        let step = needle.chars().count().max(1);
        let mut at = (0usize, 0usize);
        while let Some(pos) = self.find_from(needle, at) {
            out.push(pos);
            at = (pos.0, pos.1 + step);
        }
        out
    }

    /// Replace the next occurrence of `needle` at/after the caret with `replacement`,
    /// moving the caret past it. Returns whether a replacement was made.
    pub fn replace_next(&mut self, needle: &str, replacement: &str) -> bool {
        let Some((row, col)) = self.find_from(needle, (self.cy, self.cx)) else {
            return false;
        };
        let n = needle.chars().count();
        let mut chars: Vec<char> = self.lines[row].chars().collect();
        chars.splice(col..col + n, replacement.chars());
        self.lines[row] = chars.into_iter().collect();
        self.set_caret(row, col + replacement.chars().count());
        true
    }

    /// Replace **all** occurrences of `needle` with `replacement`. Returns the count.
    pub fn replace_all(&mut self, needle: &str, replacement: &str) -> usize {
        if needle.is_empty() {
            return 0;
        }
        let n = needle.chars().count();
        let mut count = 0;
        for row in 0..self.lines.len() {
            let mut chars: Vec<char> = self.lines[row].chars().collect();
            let mut col = 0;
            while col + n <= chars.len() {
                if chars[col..col + n] == needle.chars().collect::<Vec<_>>()[..] {
                    let rep: Vec<char> = replacement.chars().collect();
                    chars.splice(col..col + n, rep.iter().copied());
                    col += rep.len();
                    count += 1;
                } else {
                    col += 1;
                }
            }
            self.lines[row] = chars.into_iter().collect();
        }
        self.clamp();
        count
    }

    // ── multi-cursor ──

    /// Add a secondary caret at `(row, col)` (deduplicated against existing carets).
    pub fn add_cursor(&mut self, row: usize, col: usize) {
        let p = (row, col);
        if p != (self.cy, self.cx) && !self.extra.contains(&p) {
            self.extra.push(p);
        }
    }

    /// All carets (primary + extras), sorted in document order and deduplicated.
    pub fn cursors(&self) -> Vec<Pos> {
        let mut all = self.extra.clone();
        all.push((self.cy, self.cx));
        all.sort_unstable();
        all.dedup();
        all
    }

    /// Drop all secondary carets (collapse back to a single caret).
    pub fn clear_extra_cursors(&mut self) {
        self.extra.clear();
    }

    /// Insert single-line text `s` at **every** caret at once (the core multi-cursor
    /// edit). `s` must not contain a newline. Carets shift to just after their insertion.
    pub fn insert_str_all(&mut self, s: &str) {
        if s.contains('\n') {
            return;
        }
        let len = s.chars().count();
        let mut carets = self.cursors(); // ascending
        // Apply top-to-bottom, tracking the per-row column shift so earlier inserts on a
        // row push later carets on the same row rightward.
        let mut cur_row = usize::MAX;
        let mut shift = 0usize;
        let mut updated: Vec<Pos> = Vec::with_capacity(carets.len());
        for (row, col) in carets.drain(..) {
            if row != cur_row {
                cur_row = row;
                shift = 0;
            }
            let actual = col + shift;
            self.set_caret(row, actual);
            self.insert_str(s);
            shift += len;
            updated.push((row, actual + len));
        }
        // The first caret becomes primary; the rest stay as extras.
        if let Some(&(r, c)) = updated.first() {
            self.cy = r;
            self.cx = c;
        }
        self.extra = updated.into_iter().skip(1).collect();
        self.clamp();
    }
}

impl Default for TextBuffer {
    fn default() -> Self {
        TextBuffer::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_inserts_at_caret_anywhere() {
        let mut b = TextBuffer::new("helloworld");
        // Place the caret between "hello" and "world" and insert a space.
        b.set_caret(0, 5);
        b.insert(' ');
        assert_eq!(b.text(), "hello world");
        assert_eq!(b.caret(), (0, 6));
    }

    #[test]
    fn arrow_motions_move_left_right_up_down_with_wrap() {
        let mut b = TextBuffer::new("ab\ncde");
        b.end(); // (0,2)
        assert_eq!(b.caret(), (0, 2));
        b.right(); // wraps to start of next line
        assert_eq!(b.caret(), (1, 0));
        b.left(); // wraps back to end of previous line
        assert_eq!(b.caret(), (0, 2));
        b.down();
        assert_eq!(b.caret().0, 1);
        b.up();
        assert_eq!(b.caret().0, 0);
    }

    #[test]
    fn home_end_and_document_ends() {
        let mut b = TextBuffer::new("first\nmiddle\nlast");
        b.set_caret(1, 3);
        b.home();
        assert_eq!(b.caret(), (1, 0));
        b.end();
        assert_eq!(b.caret(), (1, 6));
        b.doc_start();
        assert_eq!(b.caret(), (0, 0));
        b.doc_end();
        assert_eq!(b.caret(), (2, 4));
    }

    #[test]
    fn word_hops_forward_and_back() {
        let mut b = TextBuffer::new("foo bar baz");
        b.next_word();
        assert_eq!(b.caret(), (0, 4));
        b.next_word();
        assert_eq!(b.caret(), (0, 8));
        b.prev_word();
        assert_eq!(b.caret(), (0, 4));
    }

    #[test]
    fn click_places_caret_at_nearest_glyph() {
        let mut b = TextBuffer::new("hello\nworld");
        // char_w=8, line_h=16, origin at (0,0). Click near col 3 of row 1.
        b.place_at_pixel(8 * 3 + 2, 16 + 4, (0, 0), 8, 16);
        assert_eq!(b.caret(), (1, 3));
        // Click far right clamps to line end.
        b.place_at_pixel(9999, 4, (0, 0), 8, 16);
        assert_eq!(b.caret(), (0, 5));
    }

    #[test]
    fn backspace_and_delete_join_lines() {
        let mut b = TextBuffer::new("ab\ncd");
        b.set_caret(1, 0);
        b.backspace(); // join → "abcd", caret at (0,2)
        assert_eq!(b.text(), "abcd");
        assert_eq!(b.caret(), (0, 2));
        b.set_caret(0, 4);
        b.delete(); // nothing after → no-op at very end
        assert_eq!(b.text(), "abcd");
    }

    #[test]
    fn selection_extends_and_deletes() {
        let mut b = TextBuffer::new("hello world");
        b.set_caret(0, 0);
        b.begin_selection();
        for _ in 0..5 {
            // Shift-Right would call begin then right; emulate the extend by hand.
            let a = b.anchor;
            b.cx += 1;
            b.anchor = a; // right() clears selection, so move manually for the test
        }
        assert_eq!(b.selected_text(), "hello");
        b.delete_selection();
        assert_eq!(b.text(), " world");
        assert_eq!(b.caret(), (0, 0));
    }

    #[test]
    fn caret_blinks_on_a_clock_and_holds_solid_after_activity() {
        let mut b = TextBuffer::new("x");
        // Pure blink: on for [0,500), off for [500,1000), on again at 1000.
        assert!(b.caret_visible(0));
        assert!(b.caret_visible(499));
        assert!(!b.caret_visible(500));
        assert!(b.caret_visible(1000));
        // After typing at t=600 it is held solid through 1100 even across a blink edge.
        b.touch(600);
        assert!(b.caret_visible(700)); // would be the off-half, but held solid
        assert!(b.caret_visible(1099)); // still solid up to 600+BLINK_MS
        assert!(b.caret_visible(1100)); // solid ends; on-half of the resumed blink
        assert!(!b.caret_visible(1500)); // next off-half
    }

    #[test]
    fn caret_rect_tracks_position_and_paints_when_visible() {
        let mut b = TextBuffer::new("abc");
        b.set_caret(0, 2);
        let r = b.caret_rect((10, 4), 8, 16);
        assert_eq!((r.x, r.y, r.h), (10 + 16, 4, 16));
        let theme = Theme::dark();
        let mut scene = Vec::new();
        b.paint_caret(&mut scene, &theme, (10, 4), 8, 16, 0); // visible at t=0
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == theme.primary)));
        scene.clear();
        b.paint_caret(&mut scene, &theme, (10, 4), 8, 16, 500); // blinked off
        assert!(!scene.iter().any(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == theme.primary)));
    }

    #[test]
    fn insert_str_with_newlines_splits_lines() {
        let mut b = TextBuffer::empty();
        b.insert_str("ab\ncd");
        assert_eq!(b.line_count(), 2);
        assert_eq!(b.text(), "ab\ncd");
        assert_eq!(b.caret(), (1, 2));
    }

    #[test]
    fn shift_motions_extend_a_selection() {
        let mut b = TextBuffer::new("hello world");
        b.set_caret(0, 0);
        // Shift+Right ×5 selects "hello".
        for _ in 0..5 {
            b.select_right();
        }
        assert_eq!(b.selected_text(), "hello");
        // Shift+End extends to the end of the line.
        b.select_end();
        assert_eq!(b.selected_text(), "hello world");
        // A plain motion clears it.
        b.left();
        assert!(b.selection().is_none());
        // Select-all grabs everything.
        b.select_all();
        assert_eq!(b.selected_text(), "hello world");
    }

    #[test]
    fn mouse_drag_selects_a_range() {
        let mut b = TextBuffer::new("hello\nworld");
        // char_w=8, line_h=16, origin (0,0). Begin at row0 col0, drag to row1 col3.
        b.begin_select_at_pixel(0, 0, (0, 0), 8, 16);
        b.select_to_pixel(8 * 3 + 1, 16 + 2, (0, 0), 8, 16);
        assert_eq!(b.selection(), Some(((0, 0), (1, 3))));
        assert_eq!(b.selected_text(), "hello\nwor");
    }

    #[test]
    fn cut_copy_paste_round_trip() {
        let mut b = TextBuffer::new("hello world");
        b.set_caret(0, 0);
        b.begin_selection();
        b.set_caret(0, 5); // select "hello"
        assert_eq!(b.copy().as_deref(), Some("hello"));
        let cut = b.cut().unwrap();
        assert_eq!(cut, "hello");
        assert_eq!(b.text(), " world");
        // Paste it at the end.
        b.doc_end();
        b.paste("hello");
        assert_eq!(b.text(), " worldhello");
        // Nothing selected → copy is None.
        b.clear_selection();
        assert!(b.copy().is_none());
    }

    #[test]
    fn find_and_replace() {
        let mut b = TextBuffer::new("foo bar foo\nbaz foo");
        assert_eq!(b.find_from("foo", (0, 0)), Some((0, 0)));
        assert_eq!(b.find_from("foo", (0, 1)), Some((0, 8)));
        assert_eq!(b.find_all("foo"), alloc::vec![(0, 0), (0, 8), (1, 4)]);
        assert!(b.find_from("missing", (0, 0)).is_none());
        // Replace next from the caret.
        b.set_caret(0, 0);
        assert!(b.replace_next("foo", "X"));
        assert_eq!(b.lines()[0], "X bar foo");
        // Replace all.
        let n = b.replace_all("foo", "Q");
        assert_eq!(n, 2);
        assert_eq!(b.text(), "X bar Q\nbaz Q");
    }

    #[test]
    fn multi_cursor_inserts_at_every_caret() {
        let mut b = TextBuffer::new("aaa\nbbb\nccc");
        b.set_caret(0, 0);
        b.add_cursor(1, 0);
        b.add_cursor(2, 0);
        assert_eq!(b.cursors(), alloc::vec![(0, 0), (1, 0), (2, 0)]);
        // Type ">" at all three line starts at once.
        b.insert_str_all(">");
        assert_eq!(b.text(), ">aaa\n>bbb\n>ccc");
        // Two carets on the same line shift correctly.
        let mut c = TextBuffer::new("0123456789");
        c.set_caret(0, 2);
        c.add_cursor(0, 6);
        c.insert_str_all("XX");
        assert_eq!(c.text(), "01XX2345XX6789");
        c.clear_extra_cursors();
        assert_eq!(c.cursors().len(), 1);
    }
}

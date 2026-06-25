//! The universal editor — Notepad++ ⊕ Vim ⊕ a live calculator (see
//! `docs/ui/universal-workspace.md`).
//!
//! One text surface subsumes the usual three+ apps:
//!
//! * **Notepad++ ergonomics** — full caret navigation (click-to-place, arrows,
//!   home/end, word hops), insert/delete/newline, find. The buffer and caret are
//!   the **global text engine** ([`crate::text::TextBuffer`]) every text surface
//!   shares, so the editor behaves like a normal editor: the caret goes anywhere,
//!   moves with the arrow keys, and **blinks**.
//! * **Vim modality (opt-in)** — `Normal`/`Insert`/`Visual` modes with composable
//!   motions (`h j k l 0 $ w`) and operators (`x`, `dd`, `i a o v`). Modality is a
//!   preference over the *same* buffer, not a separate program.
//! * **Inline calculator / live notebook** — any line that is a valid Dominion
//!   expression **evaluates in place** (`2 + 2 → 4`, `let t = 19 * 1.1; t → 20.9`),
//!   because the editor runs lines through the real Dominion interpreter
//!   ([`crate::lang::eval_source`]). "Notepad", "calculator", "code editor" and
//!   "REPL" are modes of one editor.
//!
//! Renders to a backend-agnostic [`crate::toolkit`] scene. Pure, safe `no_std`.

use crate::lsp::{Completion, CompletionKind, Lsp};
use crate::text::TextBuffer;
use crate::toolkit::{self, DrawCmd, Rect};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

/// Scrollbar gutter width (px) inside the editor viewport.
const SB: i32 = 10;

/// The mono advance the renderer actually uses — the single source of truth in
/// [`crate::toolkit::mono_advance`], so the caret lands exactly on the glyphs.
fn char_w(font_size: i32) -> i32 {
    toolkit::mono_advance(font_size)
}
/// Row height used for each editor line (label height in [`Editor::view`]).
fn line_h(font_size: i32) -> i32 {
    font_size + 6
}

/// Vim-style editing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Insert,
    Visual,
}

/// The editor: the shared text buffer plus a modal state machine and a clock for
/// the blinking caret.
pub struct Editor {
    buf: TextBuffer,
    mode: Mode,
    /// A pending operator (e.g. the first `d` of `dd`).
    pending: Option<char>,
    /// Whether this editor has keyboard focus (caret is painted only when focused).
    focused: bool,
    /// Whether lines are evaluated as a live notebook (a Settings preference).
    live_eval: bool,
    /// Last-known wall-clock (ms) for caret blink; advanced via [`Editor::tick`].
    now_ms: u64,
    /// Vertical scroll offset, in pixels (0 = top). Clamped to the content height.
    scroll: i32,
    /// Live autocomplete candidates (LSP-driven). Empty when the popup is hidden.
    completions: Vec<Completion>,
    /// Whether the completion popup is showing.
    comp_active: bool,
    /// Selected index into `completions`.
    comp_sel: usize,
    /// Monotonically increasing counter, bumped on every text mutation.
    /// Used to skip redundant `evaluate()` calls between frames.
    text_gen: u64,
    /// Cached result of the last `evaluate()` call, paired with the `text_gen`
    /// at which it was computed. On frames where the text did not change this
    /// lets `view()` skip the interpreter entirely.
    eval_cache: RefCell<Option<(u64, Vec<(usize, String)>)>>,
}

impl Editor {
    /// Open a buffer from initial text. Vim convention: start in Normal mode.
    pub fn new(text: &str) -> Editor {
        Editor {
            buf: TextBuffer::new(text),
            mode: Mode::Normal,
            pending: None,
            focused: true,
            live_eval: true,
            now_ms: 0,
            scroll: 0,
            completions: Vec::new(),
            comp_active: false,
            comp_sel: 0,
            text_gen: 0,
            eval_cache: RefCell::new(None),
        }
    }

    /// Enable/disable the inline live-notebook evaluation (a Settings preference). When
    /// off the editor is a plain text editor with no ` → value` annotations.
    pub fn set_live_eval(&mut self, on: bool) {
        self.live_eval = on;
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }
    /// The caret as `(row, col)`.
    pub fn cursor(&self) -> (usize, usize) {
        self.buf.caret()
    }
    pub fn lines(&self) -> &[String] {
        self.buf.lines()
    }
    pub fn line_count(&self) -> usize {
        self.buf.line_count()
    }
    /// The whole buffer as text.
    pub fn text(&self) -> String {
        self.buf.text()
    }
    /// Access the underlying shared buffer (e.g. for selection or external edits).
    pub fn buffer(&self) -> &TextBuffer {
        &self.buf
    }
    pub fn buffer_mut(&mut self) -> &mut TextBuffer {
        &mut self.buf
    }
    pub fn set_focused(&mut self, on: bool) {
        self.focused = on;
    }
    /// Advance the editor's clock (drives caret blink). Idempotent and cheap.
    pub fn tick(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
    }

    fn cur_len(&self) -> usize {
        self.buf.cur_len()
    }

    // ── vertical scrolling ──

    /// The pixel height of all rendered lines at `font_size` (the content extent).
    pub fn content_height(&self, font_size: i32) -> i32 {
        self.buf.line_count() as i32 * line_h(font_size) + 2 * 8
    }
    /// The largest valid scroll offset for a viewport of `view_h` px at `font_size`.
    pub fn max_scroll(&self, view_h: i32, font_size: i32) -> i32 {
        (self.content_height(font_size) - view_h).max(0)
    }
    /// Current vertical scroll offset (px).
    pub fn scroll(&self) -> i32 {
        self.scroll
    }
    /// Scroll by `delta` px (positive = down), clamped to `[0, max]`. Returns whether
    /// the offset actually moved.
    pub fn scroll_by(&mut self, delta: i32, view_h: i32, font_size: i32) -> bool {
        let max = self.max_scroll(view_h, font_size);
        let new = (self.scroll + delta).clamp(0, max);
        if new != self.scroll {
            self.scroll = new;
            true
        } else {
            false
        }
    }
    /// Ensure the caret row is inside the viewport, nudging `scroll` the minimum amount.
    /// Called after edits/navigation so the caret never scrolls off-screen.
    pub fn ensure_caret_visible(&mut self, view_h: i32, font_size: i32) {
        let lh = line_h(font_size);
        let (row, _) = self.buf.caret();
        let caret_top = 8 + row as i32 * lh;
        let caret_bot = caret_top + lh;
        // The visible window is [scroll, scroll + view_h).
        if caret_top < self.scroll + 8 {
            self.scroll = (caret_top - 8).max(0);
        } else if caret_bot > self.scroll + view_h - 8 {
            self.scroll = caret_bot - view_h + 8;
        }
        self.scroll = self.scroll.clamp(0, self.max_scroll(view_h, font_size));
    }

    // ── autocomplete / suggestions ──

    /// Whether the completion popup is currently shown.
    pub fn completion_active(&self) -> bool {
        self.comp_active && !self.completions.is_empty()
    }
    /// The current completion candidates (for tests / inspection).
    pub fn completions(&self) -> &[Completion] {
        &self.completions
    }
    /// The selected candidate index.
    pub fn completion_selected(&self) -> usize {
        self.comp_sel
    }

    /// Recompute completions for the word prefix at the caret and (re)show the popup if
    /// the prefix is non-empty and yields candidates. Hides it otherwise.
    pub fn refresh_completions(&mut self) {
        let (row, col) = self.buf.caret();
        let line = self.buf.lines().get(row).cloned().unwrap_or_default();
        // Never show completions while the caret is inside a string literal.
        if is_in_string(&line, col) {
            self.dismiss_completions();
            return;
        }
        let prefix = Lsp::word_prefix(&line, col);
        if prefix.is_empty() {
            self.dismiss_completions();
            return;
        }
        let src = self.buf.text();
        let mut cands = Lsp::complete(&src, &prefix);
        // Don't offer a single exact-match candidate (nothing to complete).
        cands.retain(|c| c.label != prefix);
        if cands.is_empty() {
            self.dismiss_completions();
        } else {
            self.completions = cands;
            self.comp_active = true;
            self.comp_sel = self.comp_sel.min(self.completions.len() - 1);
        }
    }
    /// Explicitly open completions (Ctrl+Space): like [`Self::refresh_completions`] but
    /// resets the selection to the top.
    pub fn open_completions(&mut self) {
        self.comp_sel = 0;
        self.refresh_completions();
    }
    /// Hide the completion popup.
    pub fn dismiss_completions(&mut self) {
        self.comp_active = false;
        self.completions.clear();
        self.comp_sel = 0;
    }
    /// Move the completion selection by `delta` (wrapping).
    pub fn move_completion(&mut self, delta: i32) {
        if self.completions.is_empty() {
            return;
        }
        let n = self.completions.len() as i32;
        self.comp_sel = (((self.comp_sel as i32 + delta) % n + n) % n) as usize;
    }
    /// Accept the selected completion: insert the remaining suffix after the typed
    /// prefix at the caret. Returns whether anything was inserted.
    pub fn accept_completion(&mut self) -> bool {
        if !self.completion_active() {
            return false;
        }
        let label = self.completions[self.comp_sel].label.clone();
        let (row, col) = self.buf.caret();
        let line = self.buf.lines().get(row).cloned().unwrap_or_default();
        let prefix = Lsp::word_prefix(&line, col);
        self.mode = Mode::Insert;
        if let Some(suffix) = label.strip_prefix(&prefix) {
            for c in suffix.chars() {
                self.buf.insert(c);
            }
            self.text_gen = self.text_gen.wrapping_add(1);
        }
        self.buf.touch(self.now_ms);
        self.dismiss_completions();
        true
    }

    // ── editing (Insert mode / direct) ──

    /// Insert a character at the caret (Insert mode).
    pub fn insert(&mut self, ch: char) {
        self.buf.insert(ch);
        self.buf.touch(self.now_ms);
        self.text_gen = self.text_gen.wrapping_add(1);
    }
    /// Split the current line at the caret (Enter).
    pub fn newline(&mut self) {
        self.buf.newline();
        self.buf.touch(self.now_ms);
        self.text_gen = self.text_gen.wrapping_add(1);
    }
    /// Delete the character before the caret (Backspace); joins lines at column 0.
    pub fn backspace(&mut self) {
        self.buf.backspace();
        self.buf.touch(self.now_ms);
        self.text_gen = self.text_gen.wrapping_add(1);
    }

    // ── caret navigation (works in any mode; the "normal" arrow-key behaviour) ──

    pub fn left(&mut self) {
        self.buf.left();
        self.buf.touch(self.now_ms);
    }
    pub fn right(&mut self) {
        self.buf.right();
        self.buf.touch(self.now_ms);
    }
    pub fn up(&mut self) {
        self.buf.up();
        self.buf.touch(self.now_ms);
    }
    pub fn down(&mut self) {
        self.buf.down();
        self.buf.touch(self.now_ms);
    }
    pub fn home(&mut self) {
        self.buf.home();
    }
    pub fn end(&mut self) {
        self.buf.end();
    }

    /// **Click-to-place**: move the caret to the glyph nearest a pixel within the
    /// editor `area` that [`Editor::view`] last rendered into. Switches into Insert
    /// mode so the next keystroke types where you clicked, like any editor.
    pub fn place_cursor(&mut self, px: i32, py: i32, area: toolkit::Rect, font_size: i32) {
        let origin = (area.x + 8, area.y + 8); // matches view()'s padding
        self.buf.place_at_pixel(px, py, origin, char_w(font_size), line_h(font_size));
        self.buf.touch(self.now_ms);
        self.mode = Mode::Insert;
        self.focused = true;
    }

    /// **Begin a mouse selection** at a pixel (press): place the caret and anchor a
    /// selection there. A drag then extends it via [`Editor::extend_select`].
    pub fn begin_select(&mut self, px: i32, py: i32, area: toolkit::Rect, font_size: i32) {
        let origin = (area.x + 8, area.y + 8);
        self.buf.begin_select_at_pixel(px, py, origin, char_w(font_size), line_h(font_size));
        self.buf.touch(self.now_ms);
        self.mode = Mode::Insert;
        self.focused = true;
    }
    /// **Extend a mouse selection** to a pixel (drag).
    pub fn extend_select(&mut self, px: i32, py: i32, area: toolkit::Rect, font_size: i32) {
        let origin = (area.x + 8, area.y + 8);
        self.buf.select_to_pixel(px, py, origin, char_w(font_size), line_h(font_size));
        self.buf.touch(self.now_ms);
    }

    // ── clipboard (the shell routes Ctrl+C/X/V here for the focused editor) ──

    /// Copy the current selection, or `None` if nothing is selected.
    pub fn copy(&self) -> Option<String> {
        self.buf.copy()
    }
    /// Cut the current selection (returns its text and removes it).
    pub fn cut(&mut self) -> Option<String> {
        let t = self.buf.cut();
        if t.is_some() {
            self.text_gen = self.text_gen.wrapping_add(1);
        }
        self.buf.touch(self.now_ms);
        t
    }
    /// Paste text at the caret, replacing any selection; enters Insert mode.
    pub fn paste(&mut self, s: &str) {
        self.buf.paste(s);
        self.buf.touch(self.now_ms);
        self.text_gen = self.text_gen.wrapping_add(1);
        self.mode = Mode::Insert;
    }

    // ── key handling ──

    /// Feed one key. In Insert mode, printable chars are inserted, Esc (`\x1b`)
    /// returns to Normal, and the arrow/navigation control codes move the caret. In
    /// Normal mode, keys are Vim motions/operators.
    pub fn key(&mut self, ch: char) {
        // Completion popup: arrows cycle selection; Tab/Enter accept; Esc dismisses.
        if self.completion_active() {
            match ch {
                '\x1e' => { self.move_completion(-1); return; } // Up
                '\x1f' => { self.move_completion(1); return; }  // Down
                '\x09' | '\n' => { let _ = self.accept_completion(); return; }
                '\x1b' => { self.dismiss_completions(); /* fall through to Normal-mode switch */ }
                _ => {}
            }
        }
        // Navigation control codes are honoured in *every* mode so the arrow keys
        // always work (the kernel maps real arrow scancodes to these).
        match ch {
            '\x1c' => {
                self.left();
                return;
            }
            '\x1d' => {
                self.right();
                return;
            }
            '\x1e' => {
                self.up();
                return;
            }
            '\x1f' => {
                self.down();
                return;
            }
            '\x01' => {
                self.home();
                return;
            }
            '\x05' => {
                self.end();
                return;
            }
            // Shift+motion → extend the selection (see `crate::keys`).
            '\u{11}' => {
                self.buf.select_left();
                self.buf.touch(self.now_ms);
                return;
            }
            '\u{12}' => {
                self.buf.select_right();
                self.buf.touch(self.now_ms);
                return;
            }
            '\u{13}' => {
                self.buf.select_up();
                self.buf.touch(self.now_ms);
                return;
            }
            '\u{14}' => {
                self.buf.select_down();
                self.buf.touch(self.now_ms);
                return;
            }
            '\u{02}' => {
                self.buf.select_home();
                self.buf.touch(self.now_ms);
                return;
            }
            '\u{06}' => {
                self.buf.select_end();
                self.buf.touch(self.now_ms);
                return;
            }
            '\u{07}' => {
                self.buf.select_all();
                return;
            }
            _ => {}
        }
        match self.mode {
            Mode::Insert => match ch {
                '\x1b' => {
                    self.mode = Mode::Normal;
                    self.dismiss_completions();
                    let (r, c) = self.buf.caret();
                    if c > 0 {
                        self.buf.set_caret(r, c - 1);
                    }
                }
                '\n' => {
                    self.newline();
                    self.dismiss_completions();
                }
                '\x08' => {
                    self.backspace();
                    self.refresh_completions();
                }
                '\x7f' => {
                    self.buf.delete();
                    self.buf.touch(self.now_ms);
                    self.text_gen = self.text_gen.wrapping_add(1);
                    self.refresh_completions();
                }
                '\x00' => {
                    // Ctrl+Space: explicitly open the completion popup.
                    self.open_completions();
                }
                '\x09' => {
                    // Tab (no active completion): insert 4 spaces.
                    for _ in 0..4 {
                        self.buf.insert(' ');
                    }
                    self.buf.touch(self.now_ms);
                    self.text_gen = self.text_gen.wrapping_add(1);
                    self.refresh_completions();
                }
                '\u{19}' => {
                    // Shift+Tab: remove one indent level (up to 4 spaces or 1 tab) from
                    // the start of the current line.
                    let (row, _col) = self.buf.caret();
                    let remove = {
                        let lines = self.buf.lines();
                        if let Some(line) = lines.get(row) {
                            if line.starts_with('\t') {
                                1
                            } else {
                                line.chars().take(4).take_while(|&c| c == ' ').count()
                            }
                        } else {
                            0
                        }
                    };
                    if remove > 0 {
                        // Move caret to column 0 and delete `remove` chars forward.
                        self.buf.set_caret(row, 0);
                        for _ in 0..remove {
                            self.buf.delete();
                        }
                        self.buf.touch(self.now_ms);
                        self.text_gen = self.text_gen.wrapping_add(1);
                    }
                    self.refresh_completions();
                }
                c if !c.is_control() => {
                    self.insert(c);
                    self.refresh_completions();
                }
                _ => {}
            },
            Mode::Normal | Mode::Visual => self.normal_key(ch),
        }
    }

    fn normal_key(&mut self, ch: char) {
        // Complete a pending operator (e.g. `dd`).
        if let Some(op) = self.pending.take() {
            if op == 'd' && ch == 'd' {
                self.buf.delete_line();
                self.buf.touch(self.now_ms);
                self.text_gen = self.text_gen.wrapping_add(1);
            }
            return;
        }
        let (r, c) = self.buf.caret();
        match ch {
            'h' => {
                if c > 0 {
                    self.buf.set_caret(r, c - 1);
                }
            }
            'l' => {
                if c < self.cur_len() {
                    self.buf.set_caret(r, c + 1);
                }
            }
            'j' => self.buf.down(),
            'k' => self.buf.up(),
            '0' => self.buf.home(),
            '$' => self.buf.end(),
            'w' => self.buf.next_word(),
            'b' => self.buf.prev_word(),
            'x' => {
                self.buf.delete();
                self.buf.touch(self.now_ms);
                self.text_gen = self.text_gen.wrapping_add(1);
            }
            'i' => self.mode = Mode::Insert,
            'a' => {
                if c < self.cur_len() {
                    self.buf.set_caret(r, c + 1);
                }
                self.mode = Mode::Insert;
            }
            'o' => {
                self.buf.end();
                self.buf.newline();
                self.mode = Mode::Insert;
            }
            'v' => self.mode = Mode::Visual,
            'd' => self.pending = Some('d'),
            '\x1b' => self.mode = Mode::Normal,
            _ => {}
        }
        self.buf.touch(self.now_ms);
    }

    // ── Notepad++ find ──

    /// Find the first occurrence of `needle` at or after the caret, returning its
    /// `(row, col)`. Searches forward across lines.
    pub fn find(&self, needle: &str) -> Option<(usize, usize)> {
        if needle.is_empty() {
            return None;
        }
        let (cy, cx) = self.buf.caret();
        let lines = self.buf.lines();
        for (row, line) in lines.iter().enumerate().skip(cy) {
            let start = if row == cy { cx } else { 0 };
            let hay: String = line.chars().skip(start).collect();
            if let Some(b) = hay.find(needle) {
                let col = start + hay[..b].chars().count();
                return Some((row, col));
            }
        }
        None
    }

    // ── inline calculator / live notebook ──

    /// Evaluate the buffer as a **live notebook**, top-to-bottom, on one persistent
    /// interpreter so variables carry between lines. Returns `(row, result)` for lines
    /// that produce a value. Prose lines simply don't evaluate. Beyond plain expressions
    /// this understands a few researcher-friendly conveniences:
    ///
    /// * **Variables** — `test_val = 21 * 5 + 4` binds `test_val` (and shows `109`); a
    ///   later `test_val + 4` sees it and shows `113`.
    /// * **Trailing `=`** — `21 * 5 + 4 =` shows the result, calculator-style.
    /// * **`for N` blocks** — a `for 3` header repeats its indented body 3 times with
    ///   `index` bound to `0..3`; each body line shows its per-iteration values.
    pub fn evaluate(&self) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        let mut it = crate::lang::Interpreter::new();
        let lines = self.buf.lines();
        let mut row = 0;
        while row < lines.len() {
            let raw = &lines[row];
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                row += 1;
                continue;
            }
            // A `for N` notebook loop: gather the indented body and run it N times.
            // For large N we preview only the first PREVIEW iterations to avoid
            // O(N) Rust loops with N up to 10^11 crashing the OS. The Dominion
            // `for index in range(N) { expr }` form goes through the interpreter's
            // dead-loop elimination / Faulhaber path and is always O(1).
            if let Some(n) = parse_for_header(trimmed) {
                const PREVIEW: i64 = 8;
                let indent = leading_ws(raw);
                let mut body: Vec<usize> = Vec::new();
                let mut r = row + 1;
                while r < lines.len() && !lines[r].trim().is_empty() && leading_ws(&lines[r]) > indent {
                    body.push(r);
                    r += 1;
                }
                let run_count = n.min(PREVIEW);
                let mut acc: Vec<Vec<String>> = body.iter().map(|_| Vec::new()).collect();
                for i in 0..run_count {
                    if let Ok(p) = crate::lang::parse_source(&format!("let index = {};", i)) {
                        let _ = it.run(&p);
                    }
                    for (bi, br) in body.iter().enumerate() {
                        if let Some(s) = eval_unit(&mut it, lines[*br].trim()) {
                            acc[bi].push(s);
                        }
                    }
                }
                for (bi, br) in body.iter().enumerate() {
                    if !acc[bi].is_empty() {
                        let mut joined = acc[bi].join(", ");
                        if n > PREVIEW {
                            joined.push_str(&format!(", … ({} total)", n));
                        }
                        out.push((*br, joined));
                    }
                }
                row = r;
                continue;
            }
            if let Some(s) = eval_unit(&mut it, trimmed) {
                out.push((row, s));
            }
            row += 1;
        }
        out
    }

    /// Build a toolkit view: the visible slice of lines (scrolled), each with its inline
    /// result appended as ` → value`, a blinking caret when focused, a scrollbar when the
    /// content overflows, inline diagnostic markers, and a floating completion popup.
    ///
    /// Lines are positioned manually (not via the flex column) so a vertical `scroll`
    /// offset can clip the buffer to the viewport and only paint what's visible.
    pub fn view(&self, theme: &toolkit::Theme, area: toolkit::Rect) -> Vec<DrawCmd> {
        let fs = theme.font_size;
        let lh = line_h(fs);
        let pad = theme.space;
        let results = if self.live_eval {
            // Cache miss: run the interpreter and store the result.  Cache hit (text
            // unchanged since last frame): return the stored slice — O(clone) not O(eval).
            let mut cache = self.eval_cache.borrow_mut();
            if cache.as_ref().map_or(false, |(g, _)| *g == self.text_gen) {
                cache.as_ref().unwrap().1.clone()
            } else {
                let r = self.evaluate();
                *cache = Some((self.text_gen, r.clone()));
                r
            }
        } else {
            Vec::new()
        };
        // Diagnostics → a per-line set of error rows (for inline markers).
        let diags = Lsp::diagnostics(&self.buf.text());

        let mut scene: Vec<DrawCmd> = Vec::new();
        // Background panel.
        scene.push(DrawCmd::Rect { rect: area, color: theme.surface, radius: theme.radius });

        let inner = Rect::new(area.x + pad, area.y + pad, area.w - 2 * pad, area.h - 2 * pad);
        let top = area.y + pad - self.scroll;
        for (row, line) in self.buf.lines().iter().enumerate() {
            let ly = top + row as i32 * lh;
            // Cull lines fully outside the viewport (the scroll fast-path).
            if ly + lh <= area.y || ly >= area.y + area.h {
                continue;
            }
            let mut text = line.clone();
            if let Some((_, r)) = results.iter().find(|(rr, _)| *rr == row) {
                text.push_str("   → ");
                text.push_str(r);
            }
            // Inline error marker: tint the row's text danger when it has an error diag.
            let has_err = diags.iter().any(|d| d.line == row as u32 + 1 && d.severity == crate::lsp::Severity::Error);
            let color = if has_err { theme.danger } else { theme.text };
            scene.push(DrawCmd::Text { rect: Rect::new(area.x + pad, ly, inner.w, lh), text, color, size: fs });
        }

        // Blinking caret (shifted by the scroll offset).
        if self.focused {
            let origin = (area.x + pad, area.y + pad - self.scroll);
            self.buf.paint_caret(&mut scene, theme, origin, char_w(fs), lh, self.now_ms);
        }

        // Scrollbar when content overflows.
        let max = self.max_scroll(area.h, fs);
        if max > 0 {
            let track = Rect::new(area.x + area.w - SB, area.y, SB, area.h);
            let ch = self.content_height(fs).max(1);
            let thumb_h = ((area.h * area.h) / ch).clamp(24, area.h);
            let span = area.h - thumb_h;
            let ty = area.y + if max > 0 { (self.scroll * span) / max } else { 0 };
            scene.push(DrawCmd::Rect { rect: track, color: theme.bg, radius: 0 });
            scene.push(DrawCmd::Rect { rect: Rect::new(track.x + 1, ty, SB - 2, thumb_h), color: theme.muted, radius: SB / 2 });
        }

        // Completion popup, anchored just below the caret.
        if self.completion_active() {
            self.paint_completions(&mut scene, theme, area);
        }
        scene
    }

    /// Paint the floating completion list near the caret: a surface card, one row per
    /// candidate (label + kind tag), the selected row filled with the primary token.
    fn paint_completions(&self, scene: &mut Vec<DrawCmd>, theme: &toolkit::Theme, area: Rect) {
        let fs = theme.font_size;
        let lh = line_h(fs);
        let pad = theme.space;
        let (row, col) = self.buf.caret();
        let cw = char_w(fs);
        let cx = area.x + pad + col as i32 * cw;
        let cy = area.y + pad - self.scroll + (row as i32 + 1) * lh;
        let rows = self.completions.len() as i32;
        let pop_w = 200;
        let pop_h = rows * lh + 4;
        // Keep the popup on-screen horizontally/vertically within the area.
        let px = cx.min(area.x + area.w - pop_w).max(area.x);
        let py = if cy + pop_h > area.y + area.h { (cy - lh - pop_h).max(area.y) } else { cy };
        scene.push(DrawCmd::Rect { rect: toolkit::inflate(Rect::new(px, py, pop_w, pop_h), 1), color: theme.primary, radius: theme.radius });
        scene.push(DrawCmd::Rect { rect: Rect::new(px, py, pop_w, pop_h), color: theme.surface, radius: theme.radius });
        for (i, c) in self.completions.iter().enumerate() {
            let ry = py + 2 + i as i32 * lh;
            let selected = i == self.comp_sel;
            if selected {
                scene.push(DrawCmd::Rect { rect: Rect::new(px + 2, ry, pop_w - 4, lh), color: theme.primary, radius: theme.radius / 2 });
            }
            let fg = if selected { theme.on_primary } else { theme.text };
            scene.push(DrawCmd::Text { rect: Rect::new(px + 8, ry, pop_w - 60, lh), text: c.label.clone(), color: fg, size: fs });
            let tag = match c.kind {
                CompletionKind::Keyword => "kw",
                CompletionKind::Builtin => "fn",
                CompletionKind::Symbol => "id",
                CompletionKind::PathMember => "::",
                CompletionKind::Catalog => "pkg",
            };
            let tag_fg = if selected { theme.on_primary } else { theme.muted };
            scene.push(DrawCmd::Text { rect: Rect::new(px + pop_w - 30, ry, 26, lh), text: tag.into(), color: tag_fg, size: fs - 3 });
        }
    }
}

// ── string-literal detection ──

/// Whether the caret at `col` (character index) on `line` is inside a string literal.
/// Counts unescaped `"` characters left of the caret; an odd count means inside a string.
fn is_in_string(line: &str, col: usize) -> bool {
    let chars: Vec<char> = line.chars().take(col).collect();
    let mut in_str = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && in_str {
            i += 2; // skip escaped char
            continue;
        }
        if chars[i] == '"' {
            in_str = !in_str;
        }
        i += 1;
    }
    in_str
}

// ── notebook evaluation helpers ──

/// Count leading spaces/tabs (a line's indent depth).
fn leading_ws(s: &str) -> usize {
    s.chars().take_while(|c| *c == ' ' || *c == '\t').count()
}

/// Whether `s` is a bare identifier (`[A-Za-z_][A-Za-z0-9_]*`).
fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A `for N` notebook-loop header → the (non-negative) repeat count `N`.
fn parse_for_header(line: &str) -> Option<i64> {
    let rest = line.strip_prefix("for ")?.trim();
    rest.parse::<i64>().ok().filter(|n| *n >= 0)
}

/// A bare `name = expr` assignment (rejecting `==`, `<=`, `>=`, `!=` and a missing RHS).
/// Returns `(name, expr)`.
fn parse_assignment(line: &str) -> Option<(&str, &str)> {
    let eq = line.find('=')?;
    let bytes = line.as_bytes();
    if eq + 1 < bytes.len() && bytes[eq + 1] == b'=' {
        return None; // `==`
    }
    if eq > 0 && matches!(bytes[eq - 1], b'!' | b'<' | b'>' | b'=') {
        return None; // `!=` `<=` `>=`
    }
    let name = line[..eq].trim();
    let expr = line[eq + 1..].trim();
    if expr.is_empty() || !is_ident(name) {
        return None;
    }
    Some((name, expr))
}

/// Evaluate one logical notebook line on the persistent interpreter `it`, returning a
/// display string when it yields a value. Handles `name = expr` (binds and shows the
/// value), a trailing `=` (calculator-style), and plain expressions.
fn eval_unit(it: &mut crate::lang::Interpreter, line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let src = if let Some((name, expr)) = parse_assignment(line) {
        // Bind it, then echo the bound value as the line's result.
        format!("let {} = {};\n{}", name, expr, name)
    } else {
        // Drop a trailing `=` ("21 * 5 ="), then evaluate what's left.
        line.strip_suffix('=').map(str::trim).unwrap_or(line).into()
    };
    let prog = crate::lang::parse_source(&src).ok()?;
    let v = it.run(&prog).ok()?;
    let shown = format!("{}", v);
    if shown.is_empty() || shown == "()" || shown == "unit" {
        None
    } else {
        Some(shown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_and_newline_and_backspace() {
        let mut e = Editor::new("");
        e.mode = Mode::Insert;
        for c in "hello".chars() {
            e.insert(c);
        }
        assert_eq!(e.text(), "hello");
        e.newline();
        for c in "world".chars() {
            e.insert(c);
        }
        assert_eq!(e.text(), "hello\nworld");
        assert_eq!(e.line_count(), 2);
        // Backspace at col 0 joins lines.
        e.buf.set_caret(1, 0);
        e.backspace();
        assert_eq!(e.text(), "helloworld");
    }

    #[test]
    fn vim_motions_clamp_at_bounds() {
        let mut e = Editor::new("abc\ndefgh");
        e.key('h');
        assert_eq!(e.cursor(), (0, 0));
        e.key('l');
        e.key('l');
        assert_eq!(e.cursor(), (0, 2));
        e.key('$');
        assert_eq!(e.cursor(), (0, 3));
        e.key('j'); // down to the longer line, col preserved
        assert_eq!(e.cursor().0, 1);
        e.key('0');
        assert_eq!(e.cursor(), (1, 0));
        e.key('k');
        assert_eq!(e.cursor().0, 0);
    }

    #[test]
    fn vim_x_and_dd_delete() {
        let mut e = Editor::new("hello\nworld\nbye");
        e.key('x'); // delete 'h'
        assert_eq!(e.lines()[0], "ello");
        e.key('d');
        e.key('d'); // delete the line
        assert_eq!(e.text(), "world\nbye");
        assert_eq!(e.line_count(), 2);
    }

    #[test]
    fn vim_insert_append_open_transitions() {
        let mut e = Editor::new("ab");
        assert_eq!(e.mode(), Mode::Normal);
        e.key('i');
        assert_eq!(e.mode(), Mode::Insert);
        e.key('Z');
        assert_eq!(e.lines()[0], "Zab");
        e.key('\x1b'); // back to Normal
        assert_eq!(e.mode(), Mode::Normal);
        e.key('o'); // open a line below, enter insert
        assert_eq!(e.mode(), Mode::Insert);
        assert_eq!(e.line_count(), 2);
    }

    #[test]
    fn vim_word_motion() {
        let mut e = Editor::new("foo bar baz");
        e.key('w');
        assert_eq!(e.cursor(), (0, 4)); // start of "bar"
        e.key('w');
        assert_eq!(e.cursor(), (0, 8)); // start of "baz"
    }

    #[test]
    fn find_locates_text() {
        let e = Editor::new("alpha\nbeta gamma\ndelta");
        assert_eq!(e.find("gamma"), Some((1, 5)));
        assert_eq!(e.find("zzz"), None);
    }

    #[test]
    fn inline_calculator_evaluates_expression_lines() {
        let e = Editor::new("2 + 2\nlet t = 19 * 1.1; t\nthis is prose, not code\n40 + 2");
        let r = e.evaluate();
        assert!(r.iter().any(|(row, v)| *row == 0 && v == "4"));
        assert!(r.iter().any(|(row, v)| *row == 3 && v == "42"));
        assert!(!r.iter().any(|(row, _)| *row == 2));
    }

    #[test]
    fn notebook_carries_variables_trailing_equals_and_for_loops() {
        let e = Editor::new(
            "hello this is just example text\n\ntest_val = 21 * 5 + 4\ntest_val + 4\n\n21 * 5 + 4 =\n\nfor 3\n    test_val + index\n",
        );
        let r = e.evaluate();
        let at = |row: usize| r.iter().find(|(rr, _)| *rr == row).map(|(_, v)| v.as_str());
        // Prose lines don't evaluate.
        assert_eq!(at(0), None);
        // `test_val = 21 * 5 + 4` binds and shows 109.
        assert_eq!(at(2), Some("109"));
        // The variable carries to the next line: 109 + 4 = 113.
        assert_eq!(at(3), Some("113"));
        // Trailing `=` evaluates calculator-style.
        assert_eq!(at(5), Some("109"));
        // `for 3` repeats the body with index 0..3 → 109, 110, 111.
        assert_eq!(at(8), Some("109, 110, 111"));
    }

    #[test]
    fn view_builds_a_scene_with_results() {
        let e = Editor::new("21 * 2");
        let scene = e.view(&toolkit::Theme::dark(), toolkit::Rect::new(0, 0, 300, 100));
        assert!(scene.iter().any(|c| matches!(c, toolkit::DrawCmd::Text { text, .. } if text.contains("42"))));
    }

    #[test]
    fn arrow_keys_and_click_navigate_anywhere() {
        let mut e = Editor::new("hello\nworld");
        e.key('i'); // insert mode
        e.key('\x1d'); // right
        e.key('\x1d'); // right
        assert_eq!(e.cursor(), (0, 2));
        e.key('\x1f'); // down
        assert_eq!(e.cursor().0, 1);
        e.key('\x05'); // end
        assert_eq!(e.cursor(), (1, 5));
        e.key('\x01'); // home
        assert_eq!(e.cursor(), (1, 0));
        // Click to place the caret mid-word on row 0.
        e.place_cursor(8 + 3 * char_w(15) + 1, 8 + 2, toolkit::Rect::new(0, 0, 300, 100), 15);
        assert_eq!(e.cursor(), (0, 3));
        assert_eq!(e.mode(), Mode::Insert);
    }

    #[test]
    fn shift_select_and_clipboard_round_trip() {
        let mut e = Editor::new("hello world");
        e.key('i'); // insert mode, caret at (0,0)
        e.key('\u{05}'); // End → caret at line end
        e.key('\u{02}'); // Shift+Home → select the whole line
        assert_eq!(e.copy().as_deref(), Some("hello world"));
        let cut = e.cut().unwrap();
        assert_eq!(cut, "hello world");
        assert_eq!(e.text(), "");
        e.paste("hi");
        assert_eq!(e.text(), "hi");
    }

    #[test]
    fn mouse_drag_selects_in_the_editor() {
        let mut e = Editor::new("hello");
        let area = toolkit::Rect::new(0, 0, 300, 100);
        // Press at col 0, drag to col 3 on row 0.
        e.begin_select(8, 8, area, 15);
        e.extend_select(8 + 3 * char_w(15), 8, area, 15);
        assert_eq!(e.copy().as_deref(), Some("hel"));
    }

    #[test]
    fn scroll_clamps_and_keeps_caret_visible() {
        // 50 lines, a short viewport → scrollable.
        let mut text = String::new();
        for i in 0..50 {
            text.push_str(&format!("line {}\n", i));
        }
        let mut e = Editor::new(&text);
        let view_h = 100;
        let fs = 15;
        assert!(e.max_scroll(view_h, fs) > 0);
        // Scroll down, then over-scroll clamps to max.
        assert!(e.scroll_by(40, view_h, fs));
        assert_eq!(e.scroll(), 40);
        e.scroll_by(100_000, view_h, fs);
        assert_eq!(e.scroll(), e.max_scroll(view_h, fs));
        // Move caret to the top → ensure_caret_visible scrolls back up.
        e.buffer_mut().set_caret(0, 0);
        e.ensure_caret_visible(view_h, fs);
        assert_eq!(e.scroll(), 0);
        // Move caret to the bottom → it scrolls down to reveal it.
        e.buffer_mut().set_caret(49, 0);
        e.ensure_caret_visible(view_h, fs);
        assert!(e.scroll() > 0);
    }

    #[test]
    fn typing_opens_completions_and_enter_accepts_the_suffix() {
        let mut e = Editor::new("");
        e.key('i'); // insert mode
        // Type "ten" → completes to builtin "tensor".
        for c in "ten".chars() {
            e.key(c);
        }
        assert!(e.completion_active());
        assert!(e.completions().iter().any(|c| c.label == "tensor"));
        // Select "tensor" (it's first since keywords don't match "ten") and accept it.
        let sel = e.completions()[e.completion_selected()].label.clone();
        assert!(e.accept_completion());
        assert_eq!(e.text(), sel);
        assert!(!e.completion_active());
    }

    #[test]
    fn completion_navigation_and_dismiss() {
        let mut e = Editor::new("");
        e.key('i');
        for c in "l".chars() {
            e.key(c);
        }
        // "l" → let, linear keywords.
        assert!(e.completion_active());
        let first = e.completion_selected();
        e.move_completion(1);
        assert_ne!(e.completion_selected(), first);
        e.dismiss_completions();
        assert!(!e.completion_active());
    }

    #[test]
    fn no_completions_inside_string_literal() {
        let mut e = Editor::new("");
        e.key('i'); // insert mode

        // Type `"ten` — the caret is now inside a string literal.
        // Outside a string, "ten" would trigger completions (e.g. "tensor").
        for c in "\"ten".chars() {
            e.key(c);
        }
        // The editor must NOT show completions while inside the string.
        assert!(
            !e.completion_active(),
            "completions should be suppressed inside a string literal"
        );

        // Close the string and type an identifier outside it — completions should resume.
        e.key('"'); // close the string
        for c in "ten".chars() {
            e.key(c);
        }
        // Now we're outside any string: "ten" should trigger completions again.
        assert!(
            e.completion_active(),
            "completions should be active outside a string literal"
        );
    }

    #[test]
    fn focused_view_paints_a_blinking_caret() {
        let mut e = Editor::new("ab");
        e.tick(0); // caret-visible half of the blink
        let scene = e.view(&toolkit::Theme::dark(), toolkit::Rect::new(0, 0, 200, 60));
        let theme = toolkit::Theme::dark();
        assert!(scene.iter().any(|c| matches!(c, toolkit::DrawCmd::Rect { color, .. } if *color == theme.primary)));
        e.set_focused(false);
        let scene = e.view(&theme, toolkit::Rect::new(0, 0, 200, 60));
        assert!(!scene.iter().any(|c| matches!(c, toolkit::DrawCmd::Rect { color, .. } if *color == theme.primary)));
    }
}

//! A **proper terminal** — a real scrollback terminal emulator and command line,
//! reusable as a widget anywhere (see `docs/ui/terminal.md`).
//!
//! This is the GUI counterpart to the kernel's safe-mode ASH prompt: a terminal you
//! can drop into any page. It owns
//!
//! * a **scrollback** of styled lines (output / input echo / errors / info),
//! * an **input line** backed by the global text engine ([`crate::text::TextBuffer`])
//!   — so the caret navigates, clicks, and **blinks** like any text surface,
//! * **command history** (recall with up/down), and
//! * a pluggable [`Backend`] that actually runs a submitted line. The default
//!   [`DominionBackend`] runs the real Dominion interpreter plus a few builtins, so the
//!   terminal *is* a live REPL out of the box.
//!
//! The IDE embeds one of these as its **output console**: program runs stream into
//! the terminal instead of a dumb text list, so "IDE output is an embedded terminal"
//! is literally true. Pure, safe `no_std`, host-tested.

use crate::text::TextBuffer;
use crate::toolkit::{Axis, Color, DrawCmd, Rect, Size, Theme, Widget};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The role of a scrollback line — selects its colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// Normal program/command output.
    Output,
    /// The echoed command the user submitted (prompt + text).
    Input,
    /// An error / failure line.
    Error,
    /// A system/info notice.
    Info,
}

/// One line of terminal scrollback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TermLine {
    pub kind: LineKind,
    pub text: String,
}

impl TermLine {
    pub fn new(kind: LineKind, text: impl Into<String>) -> TermLine {
        TermLine { kind, text: text.into() }
    }
}

/// Runs a submitted command line and returns the lines to print. Implement this to
/// back a terminal with any command set (a shell, a debugger, a chat).
pub trait Backend {
    fn exec(&mut self, line: &str) -> Vec<TermLine>;
    /// A one-line banner shown when the terminal opens (optional).
    fn banner(&self) -> Option<String> {
        None
    }
}

/// The default backend: a live **Dominion REPL** with a few builtins (`help`, `echo`,
/// `ver`). Any other line is evaluated as a Dominion expression.
#[derive(Default)]
pub struct DominionBackend;

impl Backend for DominionBackend {
    fn banner(&self) -> Option<String> {
        Some("DominionOS terminal — type `help`, or any Dominion expression.".into())
    }
    fn exec(&mut self, line: &str) -> Vec<TermLine> {
        let line = line.trim();
        if line.is_empty() {
            return Vec::new();
        }
        let (cmd, rest) = match line.find(char::is_whitespace) {
            Some(i) => (&line[..i], line[i..].trim()),
            None => (line, ""),
        };
        match cmd {
            "help" | "?" => vec_lines(&[
                "Commands: help, echo <text>, ver, clear.",
                "Anything else is evaluated as Dominion — e.g. `2 + 2`, `let x = 21; x * 2`.",
            ]),
            "ver" | "version" | "about" => {
                alloc::vec![TermLine::new(LineKind::Info, "DominionOS — Dominion terminal v1.0")]
            }
            "echo" => alloc::vec![TermLine::new(LineKind::Output, rest.to_string())],
            _ => match crate::lang::eval_source(line) {
                Ok(v) => {
                    let shown = alloc::format!("{}", v);
                    alloc::vec![TermLine::new(LineKind::Output, alloc::format!("→ {}", shown))]
                }
                Err(e) => alloc::vec![TermLine::new(LineKind::Error, alloc::format!("! {}", e))],
            },
        }
    }
}

fn vec_lines(lines: &[&str]) -> Vec<TermLine> {
    lines.iter().map(|l| TermLine::new(LineKind::Output, *l)).collect()
}

/// A terminal: scrollback + an input line + history + a command backend.
pub struct Terminal {
    lines: Vec<TermLine>,
    input: TextBuffer,
    prompt: String,
    history: Vec<String>,
    /// Cursor into `history` while recalling (None = editing a fresh line).
    hist_idx: Option<usize>,
    backend: Box<dyn Backend>,
    /// Cap on retained scrollback lines.
    max_lines: usize,
    now_ms: u64,
    dirty: bool,
}

impl Terminal {
    /// A terminal with the default Dominion REPL backend.
    pub fn new() -> Terminal {
        Terminal::with_backend(Box::new(DominionBackend))
    }

    /// A terminal backed by a custom command runner.
    pub fn with_backend(backend: Box<dyn Backend>) -> Terminal {
        let mut t = Terminal {
            lines: Vec::new(),
            input: TextBuffer::empty(),
            prompt: "› ".into(),
            history: Vec::new(),
            hist_idx: None,
            backend,
            max_lines: 500,
            now_ms: 0,
            dirty: true,
        };
        if let Some(b) = t.backend.banner() {
            t.lines.push(TermLine::new(LineKind::Info, b));
        }
        t
    }

    pub fn set_prompt(&mut self, p: &str) {
        self.prompt = p.to_string();
    }
    pub fn tick(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
    }
    pub fn lines(&self) -> &[TermLine] {
        &self.lines
    }
    pub fn input_text(&self) -> String {
        self.input.text()
    }
    pub fn take_dirty(&mut self) -> bool {
        core::mem::replace(&mut self.dirty, false)
    }

    // ── output API (drivers like the IDE push here) ──

    /// Print one output line.
    pub fn println(&mut self, text: impl Into<String>) {
        self.push(TermLine::new(LineKind::Output, text.into()));
    }
    /// Print one error line.
    pub fn eprintln(&mut self, text: impl Into<String>) {
        self.push(TermLine::new(LineKind::Error, text.into()));
    }
    /// Print one info/notice line.
    pub fn info(&mut self, text: impl Into<String>) {
        self.push(TermLine::new(LineKind::Info, text.into()));
    }
    /// Push a fully-styled line.
    pub fn push(&mut self, line: TermLine) {
        self.lines.push(line);
        if self.lines.len() > self.max_lines {
            let drop = self.lines.len() - self.max_lines;
            self.lines.drain(0..drop);
        }
        self.dirty = true;
    }
    /// Clear the scrollback.
    pub fn clear(&mut self) {
        self.lines.clear();
        self.dirty = true;
    }

    // ── input ──

    /// Feed one key to the input line. Enter submits; Backspace/Delete edit; the
    /// arrow control codes navigate; up/down recall history. Returns the submitted
    /// command, if Enter was pressed.
    pub fn input_key(&mut self, ch: char) -> Option<String> {
        self.input.touch(self.now_ms);
        self.dirty = true;
        match ch {
            '\n' => return Some(self.submit()),
            '\x08' => self.input.backspace(),
            '\x7f' => self.input.delete(),
            '\x1c' => self.input.left(),
            '\x1d' => self.input.right(),
            '\x1e' => self.history_prev(),
            '\x1f' => self.history_next(),
            '\x01' => self.input.home(),
            '\x05' => self.input.end(),
            // Shift+motion → extend the input-line selection (single line, so the
            // vertical pair are no-ops here). See `crate::keys`.
            '\u{11}' => self.input.select_left(),
            '\u{12}' => self.input.select_right(),
            '\u{02}' => self.input.select_home(),
            '\u{06}' => self.input.select_end(),
            '\u{07}' => self.input.select_all(),
            c if !c.is_control() => self.input.insert(c),
            _ => {}
        }
        None
    }

    /// Copy the input-line selection (for the shell clipboard).
    pub fn copy_input(&self) -> Option<String> {
        self.input.copy()
    }
    /// Cut the input-line selection.
    pub fn cut_input(&mut self) -> Option<String> {
        let t = self.input.cut();
        self.dirty = true;
        t
    }
    /// Paste text into the input line at the caret.
    pub fn paste_input(&mut self, s: &str) {
        // Keep it single-line: a pasted newline would otherwise submit unexpectedly.
        let one_line: String = s.chars().filter(|c| *c != '\n' && *c != '\r').collect();
        self.input.paste(&one_line);
        self.input.touch(self.now_ms);
        self.dirty = true;
    }

    /// Geometry of the input line for a rendered `area` (origin, char advance, row
    /// height) — the single source of truth shared by [`Terminal::view`] and the
    /// click-to-place helpers, so a click lands exactly on the glyph it points at.
    fn input_layout(&self, area: Rect, font_size: i32, space: i32) -> ((i32, i32), i32, i32) {
        let row_h = font_size + 5;
        let char_w = crate::toolkit::mono_advance(font_size);
        let inner = area.inset(space);
        let rows = (inner.h / row_h).max(1) as usize;
        let body_rows = rows.saturating_sub(1);
        let py = inner.y + body_rows as i32 * row_h;
        let prompt_cols = self.prompt.chars().count() as i32;
        ((inner.x + prompt_cols * char_w, py), char_w, row_h)
    }

    /// Click-to-place the caret in the input line, given the rendered `area`.
    pub fn place_cursor_in(&mut self, px: i32, py: i32, area: Rect, font_size: i32, space: i32) {
        let (origin, cw, lh) = self.input_layout(area, font_size, space);
        self.input.place_at_pixel(px, py, origin, cw, lh);
        self.input.touch(self.now_ms);
        self.dirty = true;
    }
    /// Begin a mouse selection on the input line at a pixel (press).
    pub fn begin_select_in(&mut self, px: i32, py: i32, area: Rect, font_size: i32, space: i32) {
        let (origin, cw, lh) = self.input_layout(area, font_size, space);
        self.input.begin_select_at_pixel(px, py, origin, cw, lh);
        self.input.touch(self.now_ms);
        self.dirty = true;
    }
    /// Extend a mouse selection on the input line to a pixel (drag).
    pub fn extend_select_in(&mut self, px: i32, py: i32, area: Rect, font_size: i32, space: i32) {
        let (origin, cw, lh) = self.input_layout(area, font_size, space);
        self.input.select_to_pixel(px, py, origin, cw, lh);
        self.input.touch(self.now_ms);
        self.dirty = true;
    }

    /// Click-to-place using a raw origin (legacy helper retained for callers that
    /// already know the layout).
    pub fn place_cursor(&mut self, px: i32, py: i32, input_origin: (i32, i32), char_w: i32, line_h: i32) {
        self.input.place_at_pixel(px, py, input_origin, char_w, line_h);
        self.input.touch(self.now_ms);
        self.dirty = true;
    }

    /// Submit the current input line: echo it, run it through the backend, append the
    /// result, record history, and clear the input. Returns the submitted text.
    pub fn submit(&mut self) -> String {
        let line = self.input.text();
        self.input.set_text("");
        self.hist_idx = None;
        // Echo the command (prompt + text).
        self.push(TermLine::new(LineKind::Input, alloc::format!("{}{}", self.prompt, line)));
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            self.history.push(line.clone());
            // `clear` is handled by the terminal itself.
            if trimmed == "clear" || trimmed == "cls" {
                self.clear();
            } else {
                let out = self.backend.exec(trimmed);
                for l in out {
                    self.push(l);
                }
            }
        }
        line
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.hist_idx = Some(idx);
        self.input.set_text(&self.history[idx]);
        self.input.end();
    }
    fn history_next(&mut self) {
        match self.hist_idx {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.hist_idx = Some(i + 1);
                self.input.set_text(&self.history[i + 1]);
                self.input.end();
            }
            Some(_) => {
                // Past the newest entry → back to a blank fresh line.
                self.hist_idx = None;
                self.input.set_text("");
            }
        }
    }

    // ── rendering ──

    fn color(kind: LineKind, t: &Theme) -> Color {
        match kind {
            LineKind::Output => t.text,
            LineKind::Input => t.primary,
            LineKind::Error => t.danger,
            LineKind::Info => t.muted,
        }
    }

    /// Render the terminal into `area`: a console surface, the last N scrollback
    /// lines that fit, and a prompt line with the input text + blinking caret.
    pub fn view(&self, theme: &Theme, area: Rect) -> Vec<DrawCmd> {
        let row_h = theme.font_size + 5;
        let char_w = crate::toolkit::mono_advance(theme.font_size);
        let mut scene = alloc::vec![DrawCmd::Rect { rect: area, color: theme.surface, radius: theme.radius }];
        let inner = area.inset(theme.space);
        let rows = (inner.h / row_h).max(1) as usize;
        // Reserve the last row for the prompt; show the tail of scrollback above it.
        let body_rows = rows.saturating_sub(1);
        let start = self.lines.len().saturating_sub(body_rows);
        let mut y = inner.y;
        for line in &self.lines[start..] {
            scene.push(DrawCmd::Text {
                rect: Rect::new(inner.x, y, inner.w, row_h),
                text: line.text.clone(),
                color: Self::color(line.kind, theme),
                size: theme.font_size,
            });
            y += row_h;
        }
        // Prompt line at the bottom of the body.
        let py = inner.y + body_rows as i32 * row_h;
        let prompt_text = alloc::format!("{}{}", self.prompt, self.input.text());
        scene.push(DrawCmd::Text {
            rect: Rect::new(inner.x, py, inner.w, row_h),
            text: prompt_text,
            color: theme.primary,
            size: theme.font_size,
        });
        // Blinking caret, offset past the prompt glyphs.
        let prompt_cols = self.prompt.chars().count() as i32;
        let origin = (inner.x + prompt_cols * char_w, py);
        self.input.paint_caret(&mut scene, theme, origin, char_w, row_h, self.now_ms);
        scene
    }

    /// Build the terminal as a single flex widget (for embedding in a layout tree).
    pub fn widget(&self, id: u32) -> Widget {
        Widget::Container { id, axis: Axis::Column, padding: 0, size: Size::Flex(1), children: Vec::new() }
    }
}

impl Default for Terminal {
    fn default() -> Self {
        Terminal::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_line(t: &mut Terminal, s: &str) {
        for c in s.chars() {
            t.input_key(c);
        }
        t.input_key('\n');
    }

    #[test]
    fn evaluates_dominion_expressions_as_a_repl() {
        let mut t = Terminal::new();
        type_line(&mut t, "2 + 2");
        assert!(t.lines().iter().any(|l| l.kind == LineKind::Output && l.text.contains('4')));
        type_line(&mut t, "let x = 21; x * 2");
        assert!(t.lines().iter().any(|l| l.text.contains("42")));
    }

    #[test]
    fn builtins_help_echo_ver_and_clear() {
        let mut t = Terminal::new();
        type_line(&mut t, "echo hello there");
        assert!(t.lines().iter().any(|l| l.text == "hello there"));
        type_line(&mut t, "ver");
        assert!(t.lines().iter().any(|l| l.kind == LineKind::Info && l.text.contains("DominionOS")));
        type_line(&mut t, "clear");
        // After clear, scrollback only holds the echo of the `clear` command itself.
        assert!(t.lines().iter().all(|l| !l.text.contains("hello there")));
    }

    #[test]
    fn echoes_the_command_with_the_prompt() {
        let mut t = Terminal::new();
        type_line(&mut t, "ver");
        assert!(t.lines().iter().any(|l| l.kind == LineKind::Input && l.text.starts_with("› ver")));
    }

    #[test]
    fn errors_render_as_error_lines() {
        let mut t = Terminal::new();
        type_line(&mut t, "let x = ;"); // a parse error
        assert!(t.lines().iter().any(|l| l.kind == LineKind::Error));
    }

    #[test]
    fn history_recall_walks_previous_commands() {
        let mut t = Terminal::new();
        type_line(&mut t, "echo one");
        type_line(&mut t, "echo two");
        // Up once → most recent ("echo two"); up again → "echo one".
        t.input_key('\x1e');
        assert_eq!(t.input_text(), "echo two");
        t.input_key('\x1e');
        assert_eq!(t.input_text(), "echo one");
        // Down → forward to "echo two"; down again → blank fresh line.
        t.input_key('\x1f');
        assert_eq!(t.input_text(), "echo two");
        t.input_key('\x1f');
        assert_eq!(t.input_text(), "");
    }

    #[test]
    fn editing_caret_moves_within_the_input_line() {
        let mut t = Terminal::new();
        for c in "helloworld".chars() {
            t.input_key(c);
        }
        // Move left 5 and insert a space → "hello world".
        for _ in 0..5 {
            t.input_key('\x1c');
        }
        t.input_key(' ');
        assert_eq!(t.input_text(), "hello world");
    }

    #[test]
    fn input_selection_and_clipboard() {
        let mut t = Terminal::new();
        for c in "hello".chars() {
            t.input_key(c);
        }
        t.input_key('\u{07}'); // Ctrl+A select all
        assert_eq!(t.copy_input().as_deref(), Some("hello"));
        let cut = t.cut_input().unwrap();
        assert_eq!(cut, "hello");
        assert_eq!(t.input_text(), "");
        t.paste_input("echo hi");
        assert_eq!(t.input_text(), "echo hi");
        // A pasted newline is stripped so it can't submit unexpectedly.
        let mut t2 = Terminal::new();
        t2.paste_input("a\nb");
        assert_eq!(t2.input_text(), "ab");
    }

    #[test]
    fn driver_can_push_output_lines() {
        // The IDE embeds a terminal and streams program output into it.
        let mut t = Terminal::new();
        t.println("→ 1764");
        t.eprintln("! boom");
        assert!(t.lines().iter().any(|l| l.kind == LineKind::Output && l.text.contains("1764")));
        assert!(t.lines().iter().any(|l| l.kind == LineKind::Error && l.text.contains("boom")));
    }

    #[test]
    fn view_renders_surface_prompt_and_caret() {
        let mut t = Terminal::new();
        t.tick(0); // caret-visible half of the blink
        for c in "ab".chars() {
            t.input_key(c);
        }
        let theme = Theme::dark();
        let scene = t.view(&theme, Rect::new(0, 0, 400, 200));
        // A surface, a prompt line containing the input, and a primary caret rect.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == theme.surface)));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("ab"))));
        assert!(scene.iter().filter(|c| matches!(c, DrawCmd::Rect { color, .. } if *color == theme.primary)).count() >= 1);
    }
}

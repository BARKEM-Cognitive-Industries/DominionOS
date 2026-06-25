//! The **Terminal** app page — a full-window developer terminal that hosts a real
//! [`Terminal`](crate::terminal) driven by the Unix-like [`ShellBackend`](crate::shellcmd).
//!
//! This is the Linux half of the experience: a focused, scrollback console with a
//! `cwd $ ` prompt, command history, a blinking caret, and the full builtin set
//! (`ls`/`cd`/`cat`/`ps`/…) over the shared filesystem and scheduler. It follows the
//! same page contract as every other shell page, and reports `wants_text` so the
//! kernel routes every keystroke (including digits and Esc) into the command line
//! rather than treating them as shell hotkeys. Pure, safe `no_std`.

use crate::agent::{ActionDesc, ActionKind, AgentAction, AgentControllable, AgentNode, AgentResult, NodeState};
use crate::shellcmd::{SharedSched, ShellBackend};
use crate::filesystem::SharedFs;
use crate::terminal::Terminal;
use crate::text::BLINK_MS;
use crate::toolkit::{DrawCmd, Rect, Theme};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The Terminal app page.
pub struct TermPage {
    term: Terminal,
    fs: SharedFs,
    area: Rect,
    now_ms: u64,
    last_left: bool,
    /// True while a mouse-drag selection is in progress on the input line.
    dragging: bool,
    damage: Option<Rect>,
}

impl TermPage {
    pub fn new(fs: SharedFs, sched: SharedSched) -> TermPage {
        let backend = ShellBackend::new(fs.clone(), sched);
        let term = Terminal::with_backend(Box::new(backend));
        let mut p = TermPage {
            term,
            fs,
            area: Rect::new(0, 0, 1280, 600),
            now_ms: 0,
            last_left: false,
            dragging: false,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        };
        p.update_prompt();
        p
    }

    /// Copy the input-line selection (for the shell clipboard).
    pub fn copy(&self) -> Option<String> {
        self.term.copy_input()
    }
    /// Cut the input-line selection.
    pub fn cut(&mut self) -> Option<String> {
        let t = self.term.cut_input();
        if self.term.take_dirty() {
            self.dmg_all();
        }
        t
    }
    /// Paste clipboard text into the input line.
    pub fn paste(&mut self, s: &str) {
        self.term.paste_input(s);
        if self.term.take_dirty() {
            self.dmg_all();
        }
    }

    /// Clear the terminal scrollback (context-menu "Clear").
    pub fn clear(&mut self) {
        self.term.clear();
        self.dmg_all();
    }

    /// Refresh the prompt to the Unix-style `user@host:cwd$ ` so it tracks `cd`.
    fn update_prompt(&mut self) {
        let cwd = self.fs.borrow().cwd().to_string();
        let mut prompt = String::from("jayden@dominionos:");
        prompt.push_str(&cwd);
        prompt.push_str("$ ");
        self.term.set_prompt(&prompt);
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

    /// A terminal is always a text surface — every key is routed to the command line.
    pub fn wants_text(&self) -> bool {
        true
    }

    pub fn on_key(&mut self, ch: char) -> bool {
        // Treat carriage return as newline (some paths deliver `\r`).
        let ch = if ch == '\r' { '\n' } else { ch };
        let submitted = self.term.input_key(ch);
        if submitted.is_some() {
            self.update_prompt();
        }
        if self.term.take_dirty() {
            self.dmg_all();
        }
        true
    }

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        // Click places the input caret; a drag selects the input line. (The shell theme
        // uses font_size 15 / space 8, matching `view`'s layout.)
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        let area = Rect::new(0, 0, self.area.w, self.area.h);
        if pressed {
            self.term.begin_select_in(px, py, area, 15, 8);
            self.dragging = true;
            self.dmg_all();
        } else if left && self.dragging {
            self.term.extend_select_in(px, py, area, 15, 8);
            self.dmg_all();
        } else if released {
            self.dragging = false;
        }
        self.last_left = left;
    }

    pub fn set_time(&mut self, now_ms: u64) {
        let prev = self.now_ms;
        self.now_ms = now_ms;
        self.term.tick(now_ms);
        // Flash the caret: when the blink phase flips, repaint just the bottom band
        // where the prompt + caret live (cheap — not the whole console).
        if prev / BLINK_MS != now_ms / BLINK_MS {
            let band = Rect::new(0, (self.area.h - 28).max(0), self.area.w, 28);
            self.dmg(band);
        }
    }

    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => crate::toolkit::union(d, r),
            None => r,
        });
    }
    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }

    pub fn view(&self, theme: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: theme.bg, radius: 0 });
        s.extend(self.term.view(theme, Rect::new(0, 0, self.area.w, self.area.h)));
        s
    }
}

// ── AgentControllable ─────────────────────────────────────────────────────────

/// Stable node id for the terminal page.
/// Range 0x3000_0000..0x3000_0001 — reserved for TermPage.
pub(crate) const TERM_NODE_ID: u64 = 0x3000_0000;

impl AgentControllable for TermPage {
    fn agent_name(&self) -> &str {
        "Terminal"
    }

    fn agent_view(&self) -> AgentNode {
        let history_lines = self.term.lines().len() as u32;
        let current_input = self.term.input_text();
        let prompt = {
            // Reconstruct the display prompt from the last line in view; fall back
            // to a static string if the terminal's prompt field isn't directly
            // accessible here (we set it in update_prompt).
            let cwd = self.fs.borrow().cwd().to_string();
            let mut p = String::from("jayden@dominionos:");
            p.push_str(&cwd);
            p.push_str("$ ");
            p
        };
        AgentNode::new(TERM_NODE_ID, NodeState::Terminal {
            prompt,
            history_lines,
            current_input,
        })
        .with_actions([
            ActionDesc::with_params(ActionKind::Type, "Type text into the command line", &["text"]),
            ActionDesc::with_params(ActionKind::RunCommand, "Submit a shell command", &["cmd"]),
            ActionDesc::simple(ActionKind::Clear, "Clear terminal scrollback"),
        ])
    }

    fn agent_dispatch(&mut self, action: AgentAction) -> AgentResult {
        if action.target != TERM_NODE_ID {
            return AgentResult::NotFound;
        }
        match action.kind {
            ActionKind::Type => {
                let Some(text) = action.param else {
                    return AgentResult::invalid("Type requires a text param");
                };
                for ch in text.chars() {
                    self.on_key(ch);
                }
                AgentResult::Ok
            }
            ActionKind::RunCommand => {
                let Some(cmd) = action.param else {
                    return AgentResult::invalid("RunCommand requires a cmd param");
                };
                for ch in cmd.chars() {
                    self.on_key(ch);
                }
                self.on_key('\n');
                AgentResult::Ok
            }
            ActionKind::Clear => {
                self.clear();
                AgentResult::Ok
            }
            _ => AgentResult::invalid("unsupported action for Terminal"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{Capability, Rights};
    use crate::filesystem::FileSystem;
    use crate::sched::Scheduler;
    use alloc::rc::Rc;
    use core::cell::RefCell;

    fn page() -> TermPage {
        let fs = FileSystem::shared();
        let sched = Rc::new(RefCell::new(Scheduler::new()));
        sched.borrow_mut().spawn("init", Capability::mint(0, 0x1000, Rights::ALL));
        let mut p = TermPage::new(fs, sched);
        p.set_area(Rect::new(0, 0, 1000, 500));
        let _ = p.take_damage();
        p
    }

    fn run(p: &mut TermPage, cmd: &str) {
        for c in cmd.chars() {
            p.on_key(c);
        }
        p.on_key('\n');
    }

    #[test]
    fn renders_prompt_with_cwd() {
        let p = page();
        let s = p.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("jayden@dominionos:/home/jayden$"))));
    }

    #[test]
    fn typing_a_command_runs_it_and_damages() {
        let mut p = page();
        run(&mut p, "ls");
        // After running, the scrollback shows the home folders.
        let s = p.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Documents/"))));
        assert!(p.take_damage().is_some());
    }

    #[test]
    fn cd_updates_the_prompt() {
        let mut p = page();
        run(&mut p, "cd Projects");
        let s = p.view(&Theme::dark());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains(":/home/jayden/Projects$"))));
    }

    #[test]
    fn always_wants_text_input() {
        let p = page();
        assert!(p.wants_text());
    }

    #[test]
    fn caret_blink_damages_the_bottom_band() {
        let mut p = page();
        let _ = p.take_damage();
        p.set_time(0);
        let _ = p.take_damage();
        p.set_time(BLINK_MS + 1); // phase flip
        assert!(p.take_damage().is_some());
    }
}

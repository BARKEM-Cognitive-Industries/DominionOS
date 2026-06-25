//! **Keyboard control-byte protocol** — the single source of truth for the
//! non-printable key codes the kernel's keyboard ISR emits and the text surfaces
//! (editor, terminal, input fields) consume.
//!
//! The kernel decodes real scancodes (arrows, Home/End, Delete, and — with the Shift
//! or Ctrl modifier — selection and clipboard chords) into these single bytes, pushes
//! them into the key queue, and the shell forwards them as `char`s. Keeping the
//! mapping here means the on-metal driver and the pure, host-tested UI never drift.
//!
//! All values are below `0x20` (or `0x7f`), so they never collide with a printable
//! character — a surface can test `(ch as u32) < 0x20` to know it holds a control key.

// ── caret motion ──
/// Left / Right / Up / Down arrow.
pub const ARROW_LEFT: u8 = 0x1c;
pub const ARROW_RIGHT: u8 = 0x1d;
pub const ARROW_UP: u8 = 0x1e;
pub const ARROW_DOWN: u8 = 0x1f;
/// Home / End.
pub const HOME: u8 = 0x01;
pub const END: u8 = 0x05;
/// Backspace / forward Delete / Escape / Enter.
pub const BACKSPACE: u8 = 0x08;
pub const DELETE: u8 = 0x7f;
pub const ESC: u8 = 0x1b;
pub const ENTER: u8 = b'\n';

// ── selection (Shift + motion) ──
pub const SEL_LEFT: u8 = 0x11;
pub const SEL_RIGHT: u8 = 0x12;
pub const SEL_UP: u8 = 0x13;
pub const SEL_DOWN: u8 = 0x14;
pub const SEL_HOME: u8 = 0x02;
pub const SEL_END: u8 = 0x06;

// ── clipboard / selection chords (Ctrl + letter) ──
/// Ctrl+C — copy the selection to the system clipboard.
pub const COPY: u8 = 0x03;
/// Ctrl+X — cut the selection.
pub const CUT: u8 = 0x18;
/// Ctrl+V — paste the clipboard at the caret.
pub const PASTE: u8 = 0x16;
/// Ctrl+A — select all.
pub const SELECT_ALL: u8 = 0x07;

/// Shift+Tab — remove one indent level from the current line (0x19 = Ctrl+Y slot,
/// repurposed as Shift+Tab in the DominionOS keyboard protocol).
pub const SHIFT_TAB: u8 = 0x19;

/// Whether a char is one of the clipboard chords the shell intercepts globally.
pub fn is_clipboard(ch: char) -> bool {
    matches!(ch as u32, x if x == COPY as u32 || x == CUT as u32 || x == PASTE as u32)
}

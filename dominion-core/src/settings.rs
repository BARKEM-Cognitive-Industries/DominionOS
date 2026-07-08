//! The **Settings** app — the unified control panel. In an object-capability OS there
//! is no scattered registry of toggles: Settings shows the live **system** state, the
//! **appearance** theme, the **account**, and the **capabilities** currently granted
//! (each a real authority the system holds, shown with its status). It replaces both
//! "Control Panel" and a separate permissions manager.
//!
//! It is a thin, declarative surface: clicks emit a [`SettingsAction`] the shell acts
//! on (e.g. flip the theme), and live totals arrive via [`Metrics`](crate::dash::Metrics).
//! Pure, safe `no_std`, page-local coordinates.

use crate::dash::Metrics;
use crate::secprofile::{Knob, Posture, SecurityProfile};
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::widgets::{self, Scroll};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Natural stacked height of the settings cards (title … security) — the scroll
/// content height. The power button is a pinned footer, outside this.
const CONTENT_H: i32 = 2218;

/// Content-space y where the Hardware card begins (below Capabilities).
const HARDWARE_Y: i32 = 500;

/// Content-space y where the Preferences card begins (below Hardware).
const PREFS_Y: i32 = 730;

/// Content-space y where the Security card begins (below Preferences).
const SEC_Y: i32 = 948;

/// Content-space y where the Stages card begins (below Security).
const STAGES_Y: i32 = 1294;

/// Content-space y where the Ecosystem card begins (below Stages).
const ECO_Y: i32 = 1866;

/// A user-toggleable preference. These are deliberately the *safe* knobs — appearance
/// and ergonomics — so configuring the OS can never disable a security capability or
/// break networking. Capability grants stay read-only in their own card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flag {
    DesktopIcons,
    Widgets,
    LiveEval,
    EditorInsert,
    TrayClock,
}

/// The set of user preferences, the single source of truth the shell mirrors. The
/// ergonomic [`Flag`]s and the [`SecurityProfile`] live side by side, but the security
/// half is governed by [`crate::secprofile`]'s rules — only local-blast-radius knobs
/// are exposed; wire-trust invariants are never toggleable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub desktop_icons: bool,
    pub widgets: bool,
    pub live_eval: bool,
    pub editor_insert: bool,
    pub tray_clock: bool,
    /// The node's security posture + per-knob local hardening.
    pub security: SecurityProfile,
    /// The architecture stage control plane (which stages are enabled + the profile).
    pub stages: crate::stages::StageControl,
    /// The ecosystem control plane (packages/discovery/fleet/remote/… feature-sets).
    pub eco: crate::ecosystem::EcoControl,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            desktop_icons: true,
            widgets: true,
            live_eval: true,
            editor_insert: false,
            tray_clock: true,
            security: SecurityProfile::default(),
            stages: crate::stages::StageControl::default(),
            eco: crate::ecosystem::EcoControl::default(),
        }
    }
}

impl Config {
    pub fn get(&self, f: Flag) -> bool {
        match f {
            Flag::DesktopIcons => self.desktop_icons,
            Flag::Widgets => self.widgets,
            Flag::LiveEval => self.live_eval,
            Flag::EditorInsert => self.editor_insert,
            Flag::TrayClock => self.tray_clock,
        }
    }
    pub fn set(&mut self, f: Flag, v: bool) {
        match f {
            Flag::DesktopIcons => self.desktop_icons = v,
            Flag::Widgets => self.widgets = v,
            Flag::LiveEval => self.live_eval = v,
            Flag::EditorInsert => self.editor_insert = v,
            Flag::TrayClock => self.tray_clock = v,
        }
    }
}

/// The preference rows, top-to-bottom, in the Preferences card.
const TOGGLES: [(Flag, &str); 5] = [
    (Flag::DesktopIcons, "Show desktop icons"),
    (Flag::Widgets, "Show desktop widgets"),
    (Flag::LiveEval, "Live code evaluation in Editor"),
    (Flag::EditorInsert, "Editor opens in insert mode"),
    (Flag::TrayClock, "Show uptime clock in the tray"),
];

/// What a click in Settings asks the shell to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsAction {
    /// Toggle between the dark and light themes.
    ToggleTheme,
    /// Power off — return to the ASH safe-mode terminal.
    PowerOff,
    /// Set a preference flag to a new value (the shell applies the side-effect).
    SetFlag(Flag, bool),
    /// Select a security posture preset (loads its local-hardening defaults).
    SetProfile(Posture),
    /// Flip one local-hardening knob (the shell applies the side-effect + re-attests).
    SetKnob(Knob, bool),
    /// Enable/disable one architecture stage (the shell mirrors it to terminal + boot).
    SetStage(crate::stages::Stage, bool),
    /// Select a stage deployment profile (loads its default enabled-set).
    SetStageProfile(crate::stages::Profile),
    /// Enable/disable one ecosystem feature-set.
    SetEcoFeature(crate::ecosystem::Feature, bool),
    /// Select an ecosystem preset (Full / Minimal).
    SetEcoPreset(crate::ecosystem::EcoPreset),
}

/// The Settings page.
pub struct Settings {
    account: String,
    theme_dark: bool,
    /// The live user preferences (mirrored by the shell, which applies side-effects).
    cfg: Config,
    metrics: Metrics,
    area: Rect,
    /// Vertical scroll so the full control panel is reachable in any window size.
    scroll: Scroll,
    last_left: bool,
    damage: Option<Rect>,
}

impl Settings {
    pub fn new(account: &str, theme_dark: bool) -> Settings {
        Settings {
            account: account.to_string(),
            theme_dark,
            cfg: Config::default(),
            metrics: Metrics::default(),
            area: Rect::new(0, 0, 1280, 600),
            scroll: Scroll::new(),
            last_left: false,
            damage: Some(Rect::new(0, 0, 1280, 600)),
        }
    }

    /// The current preferences (for the shell to read on startup).
    pub fn config(&self) -> Config {
        self.cfg
    }

    /// A toggle-row rect in content space for preference index `i` (offset by scroll).
    fn toggle_row(&self, i: usize) -> Rect {
        Rect::new(self.card_x() + 12, self.sy(PREFS_Y + 44 + i as i32 * 30), self.card_w() - 24, 26)
    }

    /// A security-posture preset button (i = 0..3) in the Security card header strip.
    fn preset_btn(&self, i: usize) -> Rect {
        let bw = 112;
        Rect::new(self.card_x() + 12 + i as i32 * (bw + 8), self.sy(SEC_Y + 40), bw, 28)
    }

    /// A local-hardening knob row (i = 0..6) in the Security card.
    fn knob_row(&self, i: usize) -> Rect {
        Rect::new(self.card_x() + 12, self.sy(SEC_Y + 104 + i as i32 * 30), self.card_w() - 24, 26)
    }

    /// `y` in content space → on-screen y (accounting for scroll).
    fn sy(&self, y: i32) -> i32 {
        y - self.scroll.offset
    }

    pub fn set_theme_dark(&mut self, dark: bool) {
        if dark != self.theme_dark {
            self.theme_dark = dark;
            self.dmg_all();
        }
    }
    pub fn set_account(&mut self, name: &str) {
        self.account = name.to_string();
        self.dmg_all();
    }
    pub fn set_metrics(&mut self, m: Metrics) {
        self.metrics = m;
        self.dmg_all();
    }

    pub fn set_area(&mut self, area: Rect) {
        if area != self.area {
            self.area = area;
            self.scroll.clamp(CONTENT_H, self.viewport().h);
            self.dmg_all();
        }
    }
    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    pub fn wants_text(&self) -> bool {
        false
    }
    pub fn on_key(&mut self, _ch: char) -> bool {
        false
    }
    pub fn set_time(&mut self, _now_ms: u64) {}

    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }

    fn theme_btn(&self) -> Rect {
        Rect::new(self.card_x() + 16, self.sy(196), 160, 32)
    }
    /// The power button is a **pinned footer** at the window bottom — always visible and
    /// reachable regardless of scroll, so you can power off from any window size.
    pub(crate) fn power_btn(&self) -> Rect {
        Rect::new(self.card_x() + 16, (self.area.h - 52).max(120), 160, 36)
    }
    /// The Capabilities card (fixed natural geometry, offset by scroll).
    fn caps_rect(&self) -> Rect {
        Rect::new(self.card_x(), self.sy(334), self.card_w(), 150)
    }

    /// A hardware-feature status row in the Hardware card (i = 0..5).
    fn hardware_row(&self, i: usize) -> Rect {
        Rect::new(self.card_x() + 12, self.sy(HARDWARE_Y + 44 + i as i32 * 26), self.card_w() - 24, 22)
    }
    fn card_x(&self) -> i32 {
        24
    }
    fn card_w(&self) -> i32 {
        (self.area.w - 48).min(620)
    }
    /// The scrolling viewport for the cards (above the pinned power-button footer).
    fn viewport(&self) -> Rect {
        Rect::new(0, 0, self.area.w, (self.area.h - 60).max(40))
    }

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) -> Option<SettingsAction> {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        let vp = self.viewport();

        // Scrollbar drag.
        if self.scroll.is_dragging() {
            if left {
                self.scroll.on_drag(py, vp, CONTENT_H, vp.h);
                self.dmg_all();
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
        // The pinned footer button takes priority over the scrollable cards.
        if self.power_btn().contains(px, py) {
            return Some(SettingsAction::PowerOff);
        }
        if self.scroll.on_press(px, py, vp, CONTENT_H, vp.h) {
            self.dmg_all();
            return None;
        }
        if self.theme_btn().contains(px, py) {
            return Some(SettingsAction::ToggleTheme);
        }
        // Preference toggles — flip the local state and ask the shell to apply it.
        for (i, (flag, _)) in TOGGLES.iter().enumerate() {
            if self.toggle_row(i).contains(px, py) {
                let v = !self.cfg.get(*flag);
                self.cfg.set(*flag, v);
                self.dmg_all();
                return Some(SettingsAction::SetFlag(*flag, v));
            }
        }
        // Security posture presets — load a whole hardening set at once.
        for (i, p) in Posture::all().iter().enumerate() {
            if self.preset_btn(i).contains(px, py) {
                self.cfg.security.select(*p);
                self.dmg_all();
                return Some(SettingsAction::SetProfile(*p));
            }
        }
        // Individual local-hardening knobs — relax/tighten one defence on the fly.
        for (i, k) in Knob::all().iter().enumerate() {
            if self.knob_row(i).contains(px, py) {
                let v = !self.cfg.security.local.get(*k);
                self.cfg.security.set_knob(*k, v);
                self.dmg_all();
                return Some(SettingsAction::SetKnob(*k, v));
            }
        }
        // Stages card — toggle one stage or select a deployment profile.
        match crate::stages::settings_hit(px, py, self.card_x(), self.sy(STAGES_Y), self.card_w()) {
            Some(crate::stages::StageClick::Profile(p)) => {
                self.cfg.stages.select(p);
                self.dmg_all();
                return Some(SettingsAction::SetStageProfile(p));
            }
            Some(crate::stages::StageClick::Toggle(st)) => {
                let v = !self.cfg.stages.is_enabled(st);
                self.cfg.stages.set(st, v);
                self.dmg_all();
                return Some(SettingsAction::SetStage(st, v));
            }
            None => {}
        }
        // Ecosystem card — toggle one feature-set or select a preset.
        match crate::ecosystem::settings_hit(px, py, self.card_x(), self.sy(ECO_Y), self.card_w()) {
            Some(crate::ecosystem::EcoClick::Preset(p)) => {
                self.cfg.eco.select(p);
                self.dmg_all();
                return Some(SettingsAction::SetEcoPreset(p));
            }
            Some(crate::ecosystem::EcoClick::Toggle(f)) => {
                let v = !self.cfg.eco.is_enabled(f);
                self.cfg.eco.set(f, v);
                self.dmg_all();
                return Some(SettingsAction::SetEcoFeature(f, v));
            }
            None => {}
        }
        None
    }

    // ── rendering ──

    pub fn view(&self, t: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: t.bg, radius: 0 });
        s.push(DrawCmd::Text { rect: Rect::new(24, self.sy(16), 400, 24), text: "Settings".into(), color: t.text, size: 20 });

        let x = self.card_x();
        let w = self.card_w();
        // System card.
        let sys = Rect::new(x, self.sy(52), w, 110);
        card(&mut s, sys, "System", t);
        let mut info = String::from("DominionOS  ·  DominionOS  ·  build ");
        info.push_str(if self.metrics.det_hash.is_empty() { "—" } else { &self.metrics.det_hash });
        kv(&mut s, sys, 0, "Edition", &info, t);
        let mut up = String::new();
        push_int(&mut up, self.metrics.uptime_secs as i64);
        up.push_str(" s uptime   ·   ");
        push_int(&mut up, (self.metrics.mem_milli / 10) as i64);
        up.push_str("% memory used");
        kv(&mut s, sys, 1, "Status", &up, t);

        // Appearance card (theme toggle).
        let appearance = Rect::new(x, self.sy(172), w, 78);
        card(&mut s, appearance, "Appearance", t);
        let tb = self.theme_btn();
        s.push(DrawCmd::Rect { rect: tb, color: t.primary, radius: t.radius });
        let label = if self.theme_dark { "Theme: Dark" } else { "Theme: Light" };
        s.push(DrawCmd::Text { rect: Rect::new(tb.x + 16, tb.y + 7, tb.w, 16), text: label.into(), color: t.on_primary, size: 13 });

        // Account card.
        let acct = Rect::new(x, self.sy(260), w, 64);
        card(&mut s, acct, "Account", t);
        let mut who = String::from("Signed in as ");
        who.push_str(&self.account);
        kv(&mut s, acct, 0, "User", &who, t);

        // Capabilities card — the live authorities the system holds. Its height adapts
        // to the window so it never collides with the power button.
        let caps = self.caps_rect();
        card(&mut s, caps, "Capabilities", t);
        let granted = [
            ("Network (NDN / sockets)", self.metrics.net_present),
            ("Storage (virtio-blk)", self.metrics.disk_present),
            ("Entropy (hardware TRNG)", self.metrics.entropy_milli > 0),
            ("Surface (framebuffer)", true),
        ];
        for (i, (name, on)) in granted.iter().enumerate() {
            let y = caps.y + 36 + i as i32 * 26;
            if y + 16 > caps.y + caps.h {
                break; // don't draw rows past the (possibly shrunk) card
            }
            let dot = if *on { Color::rgb(0x3f, 0xc9, 0xb0) } else { t.danger };
            s.push(toolkit::disc(caps.x + 22, y + 7, 4, dot));
            let name_w = caps.w - 120;
            s.push(DrawCmd::Text { rect: Rect::new(caps.x + 34, y, name_w, 16), text: toolkit::ellipsize_px(name, name_w, 12), color: t.text, size: 12 });
            let status = if *on { "granted" } else { "unavailable" };
            s.push(DrawCmd::Text { rect: Rect::new(caps.x + caps.w - 110, y, 100, 16), text: status.into(), color: t.muted, size: 12 });
        }

        // Hardware card — read-only display of active hardware capabilities.
        let hw_h = 36 + 6 * 26 + 8; // header + 6 rows * 26px + padding = 206
        let hw = Rect::new(x, self.sy(HARDWARE_Y), w, hw_h);
        card(&mut s, hw, "Hardware", t);
        let hw_features: [(&str, bool); 6] = [
            ("TPM / Hardware Entropy",   self.metrics.entropy_milli > 0),
            ("Hardware Network (NDN)",   self.metrics.net_present),
            ("Block Storage (virtio-blk)", self.metrics.disk_present),
            ("GPU Accelerator",          self.metrics.gpu_milli > 0),
            ("NPU/AI Accelerator",       self.metrics.npu_milli > 0),
            ("Framebuffer Display",      true),
        ];
        for (i, (name, on)) in hw_features.iter().enumerate() {
            let row = self.hardware_row(i);
            let dot = if *on { Color::rgb(0x3f, 0xc9, 0xb0) } else { t.danger };
            s.push(toolkit::disc(row.x + 10, row.y + 11, 4, dot));
            let name_w = row.w - 120;
            s.push(DrawCmd::Text { rect: Rect::new(row.x + 22, row.y + 3, name_w, 16), text: toolkit::ellipsize_px(name, name_w, 12), color: t.text, size: 12 });
            let status = if *on { "active" } else { "not detected" };
            s.push(DrawCmd::Text { rect: Rect::new(row.x + row.w - 100, row.y + 3, 92, 16), text: status.into(), color: t.muted, size: 12 });
        }

        // Preferences card — the safe, user-configurable knobs (toggle switches).
        let prefs = Rect::new(x, self.sy(PREFS_Y), w, 44 + TOGGLES.len() as i32 * 30 + 8);
        card(&mut s, prefs, "Preferences", t);
        for (i, (flag, label)) in TOGGLES.iter().enumerate() {
            let row = self.toggle_row(i);
            let on = self.cfg.get(*flag);
            let name_w = row.w - 64;
            s.push(DrawCmd::Text { rect: Rect::new(row.x + 8, row.y + 5, name_w, 16), text: toolkit::ellipsize_px(label, name_w, 12), color: t.text, size: 12 });
            // A pill switch on the right: filled+knob-right when on, muted+knob-left off.
            let sw = Rect::new(row.x + row.w - 44, row.y + 3, 40, 20);
            s.push(DrawCmd::Rect { rect: sw, color: if on { t.primary } else { t.muted }, radius: 10 });
            let knob_x = if on { sw.x + sw.w - 17 } else { sw.x + 1 };
            s.push(toolkit::disc(knob_x + 8, sw.y + 10, 8, t.on_primary));
        }

        // Security card — the safe relaxation surface. A posture preset row, the live
        // local-hardening knobs, and a standing note that network trust is never a knob.
        let sec_h = 104 + Knob::all().len() as i32 * 30 + 30;
        let sec = Rect::new(x, self.sy(SEC_Y), w, sec_h);
        card(&mut s, sec, "Security profile", t);
        let active = self.cfg.security.posture;
        for (i, p) in Posture::all().iter().enumerate() {
            let b = self.preset_btn(i);
            let on = *p == active;
            s.push(DrawCmd::Rect { rect: b, color: if on { t.primary } else { t.bg }, radius: t.radius });
            let col = if on { t.on_primary } else { t.text };
            s.push(DrawCmd::Text { rect: Rect::new(b.x + 12, b.y + 7, b.w - 16, 16), text: p.name().into(), color: col, size: 13 });
        }
        // The active posture's one-line trade.
        s.push(DrawCmd::Text { rect: Rect::new(sec.x + 16, self.sy(SEC_Y + 76), w - 32, 14), text: active.blurb().into(), color: t.muted, size: 12 });
        // The six local-hardening knobs (same pill switches as Preferences).
        for (i, k) in Knob::all().iter().enumerate() {
            let row = self.knob_row(i);
            let on = self.cfg.security.local.get(*k);
            let name_w = row.w - 64;
            s.push(DrawCmd::Text { rect: Rect::new(row.x + 8, row.y + 5, name_w, 16), text: toolkit::ellipsize_px(k.label(), name_w, 12), color: t.text, size: 12 });
            let sw = Rect::new(row.x + row.w - 44, row.y + 3, 40, 20);
            s.push(DrawCmd::Rect { rect: sw, color: if on { t.primary } else { t.muted }, radius: 10 });
            let knob_x = if on { sw.x + sw.w - 17 } else { sw.x + 1 };
            s.push(toolkit::disc(knob_x + 8, sw.y + 10, 8, t.on_primary));
        }
        // The standing guarantee: relaxing above only lowers *this* node's self-defence.
        let note_y = SEC_Y + 104 + Knob::all().len() as i32 * 30 + 4;
        s.push(toolkit::disc(sec.x + 22, self.sy(note_y) + 7, 4, Color::rgb(0x3f, 0xc9, 0xb0)));
        s.push(DrawCmd::Text { rect: Rect::new(sec.x + 34, self.sy(note_y), w - 48, 14), text: "Network trust (identity, encryption, capabilities) stays enforced.".into(), color: t.muted, size: 11 });

        // Stages card — the architecture stage control plane (knobs/flags + profiles).
        s.extend(crate::stages::settings_view(&self.cfg.stages, x, self.sy(STAGES_Y), w, t));

        // Ecosystem card — the nine packages/remote-nodes feature-sets (knobs/flags + presets).
        s.extend(crate::ecosystem::settings_view(&self.cfg.eco, x, self.sy(ECO_Y), w, t));

        // Power button.
        let pb = self.power_btn();
        s.push(DrawCmd::Rect { rect: pb, color: t.danger, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(pb.x + 18, pb.y + 9, pb.w, 16), text: "Power off".into(), color: t.on_primary, size: 13 });
        // Scrollbar when the cards are taller than the viewport.
        let vp = self.viewport();
        s.extend(widgets::scrollbar(vp, CONTENT_H, vp.h, &self.scroll, t));
        s
    }
}

fn card(s: &mut Vec<DrawCmd>, r: Rect, title: &str, t: &Theme) {
    s.push(DrawCmd::Rect { rect: r, color: t.surface, radius: t.radius });
    s.push(DrawCmd::Text { rect: Rect::new(r.x + 16, r.y + 10, r.w - 24, 16), text: title.into(), color: t.text, size: 14 });
}

fn kv(s: &mut Vec<DrawCmd>, card: Rect, row: i32, key: &str, value: &str, t: &Theme) {
    let y = card.y + 38 + row * 22;
    s.push(DrawCmd::Text { rect: Rect::new(card.x + 16, y, 120, 14), text: key.into(), color: t.muted, size: 12 });
    let val_w = card.w - 132;
    s.push(DrawCmd::Text { rect: Rect::new(card.x + 120, y, val_w, 14), text: toolkit::ellipsize_px(value, val_w, 12), color: t.text, size: 12 });
}

fn push_int(s: &mut String, n: i64) {
    if n < 0 {
        s.push('-');
    }
    // Format the magnitude as unsigned so `i64::MIN` (whose negation overflows) is total.
    push_uint(s, n.unsigned_abs());
}

fn push_uint(s: &mut String, n: u64) {
    if n >= 10 {
        push_uint(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        let mut s = Settings::new("Jayden", true);
        s.set_area(Rect::new(0, 0, 1000, 600));
        let _ = s.take_damage();
        s
    }

    #[test]
    fn renders_the_control_panel_sections() {
        let s = settings();
        let scene = s.view(&Theme::dark());
        for section in ["Settings", "System", "Appearance", "Account", "Capabilities"] {
            assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == section)), "missing {}", section);
        }
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Jayden"))));
    }

    #[test]
    fn theme_button_emits_toggle() {
        let mut s = settings();
        let b = s.theme_btn();
        let act = s.on_pointer(b.x + 10, b.y + 10, true);
        s.on_pointer(b.x + 10, b.y + 10, false);
        assert_eq!(act, Some(SettingsAction::ToggleTheme));
    }

    #[test]
    fn power_button_emits_poweroff() {
        let mut s = settings();
        let b = s.power_btn();
        let act = s.on_pointer(b.x + 10, b.y + 10, true);
        assert_eq!(act, Some(SettingsAction::PowerOff));
    }

    #[test]
    fn power_button_stays_visible_on_a_short_window() {
        let mut s = Settings::new("Jayden", true);
        s.set_area(Rect::new(0, 0, 500, 360)); // far shorter than the cards need
        let pb = s.power_btn();
        // Pinned footer: fully inside the window, so it is always reachable.
        assert!(pb.y >= 0 && pb.y + pb.h <= 360);
        // And the cards scroll (content taller than the viewport).
        assert!(Scroll::needed(CONTENT_H, s.viewport().h));
    }

    #[test]
    fn preference_toggle_flips_and_emits_setflag() {
        let mut s = Settings::new("Jayden", true);
        s.set_area(Rect::new(0, 0, 1000, 800)); // tall enough that the cards don't scroll
        let _ = s.take_damage();
        assert!(s.config().desktop_icons);
        let r = s.toggle_row(0); // "Show desktop icons"
        let act = s.on_pointer(r.x + 10, r.y + 10, true);
        s.on_pointer(r.x + 10, r.y + 10, false);
        assert_eq!(act, Some(SettingsAction::SetFlag(Flag::DesktopIcons, false)));
        assert!(!s.config().desktop_icons);
    }

    #[test]
    fn preferences_card_renders_toggle_rows() {
        let s = settings();
        let scene = s.view(&Theme::dark());
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Preferences")));
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Show desktop icons")));
    }

    #[test]
    fn security_card_renders_presets_and_knobs() {
        let s = settings();
        let scene = s.view(&Theme::dark());
        for label in ["Security profile", "Server", "Balanced", "Hardened", "Encrypt memory at rest"] {
            assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == label)), "missing {}", label);
        }
        // The standing wire-trust guarantee is shown.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("Network trust"))));
    }

    #[test]
    fn preset_button_selects_a_posture_and_emits_setprofile() {
        let mut s = Settings::new("Jayden", true);
        s.set_area(Rect::new(0, 0, 1000, 1200)); // tall enough that nothing scrolls
        let _ = s.take_damage();
        assert_eq!(s.config().security.posture, Posture::Balanced);
        let b = s.preset_btn(0); // "Server"
        let act = s.on_pointer(b.x + 10, b.y + 10, true);
        s.on_pointer(b.x + 10, b.y + 10, false);
        assert_eq!(act, Some(SettingsAction::SetProfile(Posture::Server)));
        // Selecting Server loaded its (all-off) hardening.
        assert_eq!(s.config().security.posture, Posture::Server);
        assert!(!s.config().security.local.memory_at_rest);
    }

    #[test]
    fn knob_toggle_relaxes_one_defence_and_emits_setknob() {
        let mut s = Settings::new("Jayden", true);
        s.set_area(Rect::new(0, 0, 1000, 1200));
        let _ = s.take_damage();
        // Start Hardened so the first knob (MemoryAtRest) is on.
        s.on_pointer(s.preset_btn(2).x + 10, s.preset_btn(2).y + 10, true);
        s.on_pointer(s.preset_btn(2).x + 10, s.preset_btn(2).y + 10, false);
        assert!(s.config().security.local.memory_at_rest);
        let r = s.knob_row(0); // "Encrypt memory at rest"
        let act = s.on_pointer(r.x + 10, r.y + 10, true);
        s.on_pointer(r.x + 10, r.y + 10, false);
        assert_eq!(act, Some(SettingsAction::SetKnob(Knob::MemoryAtRest, false)));
        assert!(!s.config().security.local.memory_at_rest);
        // Honest posture: dropping a Hardened defence lowers the reported posture.
        assert_ne!(s.config().security.posture, Posture::Hardened);
    }

    #[test]
    fn capabilities_reflect_live_metrics() {
        let mut s = settings();
        s.set_metrics(Metrics { net_present: true, disk_present: true, entropy_milli: 970, ..Default::default() });
        let scene = s.view(&Theme::dark());
        // Three granted authorities show as "granted".
        let granted = scene.iter().filter(|c| matches!(c, DrawCmd::Text { text, .. } if text == "granted")).count();
        assert!(granted >= 4);
    }

    #[test]
    fn hardware_card_renders_all_rows() {
        let mut s = settings();
        s.set_metrics(Metrics {
            net_present: true,
            disk_present: true,
            entropy_milli: 800,
            gpu_milli: 500,
            npu_milli: 200,
            ..Default::default()
        });
        let scene = s.view(&Theme::dark());
        // The card header must appear.
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Hardware")));
        // All six feature names must appear.
        for name in [
            "TPM / Hardware Entropy",
            "Hardware Network (NDN)",
            "Block Storage (virtio-blk)",
            "GPU Accelerator",
            "NPU/AI Accelerator",
            "Framebuffer Display",
        ] {
            assert!(
                scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == name)),
                "missing hardware row: {}",
                name
            );
        }
        // With all features active the "active" label should appear 6 times.
        let active_count = scene.iter().filter(|c| matches!(c, DrawCmd::Text { text, .. } if text == "active")).count();
        assert_eq!(active_count, 6);
    }

    #[test]
    fn hardware_card_shows_not_detected_when_absent() {
        let s = settings();
        // Leave all optional hardware at defaults (zeroed / false) — only Framebuffer is always on.
        let scene = s.view(&Theme::dark());
        let not_detected = scene.iter().filter(|c| matches!(c, DrawCmd::Text { text, .. } if text == "not detected")).count();
        // entropy=0, net_present=false, disk_present=false, gpu=0, npu=0 → 5 not-detected; framebuffer always active.
        assert_eq!(not_detected, 5);
    }
}

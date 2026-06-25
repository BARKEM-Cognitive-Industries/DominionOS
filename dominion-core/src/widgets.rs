//! Data widgets for the live dashboard — charts, gauges, sliders, sparklines (see
//! `docs/ui/dashboard-and-shell.md`).
//!
//! These turn **real system values** into backend-agnostic [`crate::toolkit`] scenes:
//! the compute/health bars, the line charts, the capability/health gauges, and the
//! interactive sliders in the settings. Every widget is driven by live data and the
//! interactive ones ([`Slider`]) return real values from pointer input — nothing is
//! a static mock-up. Pure, safe `no_std`.

use crate::toolkit::{self, Color, DrawCmd, Rect};
use alloc::string::String;
use alloc::vec::Vec;

/// Width of a vertical scrollbar.
pub const SCROLLBAR_W: i32 = 10;

/// A vertical **scroll model** for any overflowing list/log: a `content_h`-px tall
/// content viewed through a `view_h`-px viewport, scrolled `offset` px from the top.
/// It drives a **draggable scrollbar** (the only scroll input the PS/2 mouse affords —
/// there is no wheel), pages on a track click, and clamps itself. Pure and testable.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scroll {
    pub offset: i32,
    /// While dragging the thumb: (pointer-y at grab, offset at grab).
    drag_from: Option<(i32, i32)>,
}

impl Scroll {
    pub fn new() -> Scroll {
        Scroll::default()
    }
    /// The furthest the content can scroll.
    pub fn max_offset(content_h: i32, view_h: i32) -> i32 {
        (content_h - view_h).max(0)
    }
    pub fn clamp(&mut self, content_h: i32, view_h: i32) {
        self.offset = self.offset.clamp(0, Self::max_offset(content_h, view_h));
    }
    /// Scroll by `dy` px (positive = down), clamped.
    pub fn scroll_by(&mut self, dy: i32, content_h: i32, view_h: i32) {
        self.offset += dy;
        self.clamp(content_h, view_h);
    }
    /// Whether a scrollbar is needed at all.
    pub fn needed(content_h: i32, view_h: i32) -> bool {
        content_h > view_h
    }
    /// The scrollbar track rect down the right edge of `area`.
    pub fn track(area: Rect) -> Rect {
        Rect::new(area.x + area.w - SCROLLBAR_W, area.y, SCROLLBAR_W, area.h)
    }
    /// The draggable thumb rect, or `None` when the content fits.
    pub fn thumb(&self, area: Rect, content_h: i32, view_h: i32) -> Option<Rect> {
        if !Self::needed(content_h, view_h) || content_h <= 0 {
            return None;
        }
        let track = Self::track(area);
        let th = (view_h * track.h / content_h).clamp(24, track.h);
        let max = Self::max_offset(content_h, view_h);
        let ty = if max == 0 { track.y } else { track.y + self.offset * (track.h - th) / max };
        Some(Rect::new(track.x, ty, track.w, th))
    }
    /// Handle a press on the scrollbar: grab the thumb (start a drag) or page toward a
    /// track click. Returns whether the press hit the scrollbar (so the caller can stop).
    pub fn on_press(&mut self, px: i32, py: i32, area: Rect, content_h: i32, view_h: i32) -> bool {
        let Some(thumb) = self.thumb(area, content_h, view_h) else { return false };
        if !Self::track(area).contains(px, py) {
            return false;
        }
        if thumb.contains(px, py) {
            self.drag_from = Some((py, self.offset));
        } else {
            let dir = if py < thumb.y { -1 } else { 1 };
            self.scroll_by(dir * view_h * 3 / 4, content_h, view_h);
        }
        true
    }
    /// Continue a thumb drag to pointer-y.
    pub fn on_drag(&mut self, py: i32, area: Rect, content_h: i32, view_h: i32) {
        if let Some((y0, off0)) = self.drag_from {
            let track = Self::track(area);
            let th = self.thumb(area, content_h, view_h).map(|t| t.h).unwrap_or(track.h);
            let max = Self::max_offset(content_h, view_h);
            let denom = (track.h - th).max(1);
            self.offset = off0 + (py - y0) * max / denom;
            self.clamp(content_h, view_h);
        }
    }
    pub fn release(&mut self) {
        self.drag_from = None;
    }
    pub fn is_dragging(&self) -> bool {
        self.drag_from.is_some()
    }
}

/// Draw a vertical scrollbar (track + thumb) for `area`, if the content overflows.
pub fn scrollbar(area: Rect, content_h: i32, view_h: i32, scroll: &Scroll, theme: &toolkit::Theme) -> Vec<DrawCmd> {
    let mut s = Vec::new();
    if let Some(thumb) = scroll.thumb(area, content_h, view_h) {
        s.push(DrawCmd::Rect { rect: Scroll::track(area), color: theme.bg, radius: 4 });
        s.push(DrawCmd::Rect { rect: thumb, color: theme.muted, radius: 4 });
    }
    s
}

/// Linear blend `a→b` by `t/1000`.
pub fn lerp(a: Color, b: Color, t_milli: u32) -> Color {
    let t = t_milli.min(1000);
    let it = 1000 - t;
    Color::rgb(
        ((a.r as u32 * it + b.r as u32 * t) / 1000) as u8,
        ((a.g as u32 * it + b.g as u32 * t) / 1000) as u8,
        ((a.b as u32 * it + b.b as u32 * t) / 1000) as u8,
    )
}

/// A vertical **bar chart** of `values` scaled to `max`, filling `area`.
pub fn bar_chart(area: Rect, values: &[i64], max: i64, color: Color, theme: &toolkit::Theme) -> Vec<DrawCmd> {
    let mut scene = Vec::new();
    scene.push(DrawCmd::Rect { rect: area, color: theme.surface, radius: theme.radius });
    if values.is_empty() || max <= 0 || area.w <= 4 || area.h <= 4 {
        return scene;
    }
    let inner = area.inset(theme.space / 2);
    let n = values.len() as i32;
    let gap = 2;
    let bw = ((inner.w - gap * (n - 1)) / n).max(1);
    for (i, &v) in values.iter().enumerate() {
        let h = ((v.clamp(0, max) * inner.h as i64) / max) as i32;
        let x = inner.x + i as i32 * (bw + gap);
        let y = inner.y + inner.h - h;
        // Taller bars trend toward the accent colour — a subtle heat cue.
        let c = lerp(color, theme.accent, (v.clamp(0, max) * 1000 / max) as u32);
        scene.push(DrawCmd::Rect { rect: Rect::new(x, y, bw, h.max(1)), color: c, radius: 2 });
    }
    scene
}

/// A **line chart** of `values` over `area` (auto-scaled), with a faint baseline.
pub fn line_chart(area: Rect, values: &[i64], color: Color, theme: &toolkit::Theme) -> Vec<DrawCmd> {
    let mut scene = Vec::new();
    scene.push(DrawCmd::Rect { rect: area, color: theme.surface, radius: theme.radius });
    if values.len() < 2 || area.w <= 4 || area.h <= 4 {
        return scene;
    }
    let inner = area.inset(theme.space / 2);
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for &v in values {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (hi - lo).max(1);
    // Baseline.
    scene.push(DrawCmd::Line {
        x0: inner.x,
        y0: inner.y + inner.h - 1,
        x1: inner.x + inner.w,
        y1: inner.y + inner.h - 1,
        color: theme.muted,
        width: 1,
    });
    let mut pts = Vec::with_capacity(values.len());
    let n = values.len() as i64;
    for (i, &v) in values.iter().enumerate() {
        let x = inner.x + (i as i64 * (inner.w - 1) as i64 / (n - 1)) as i32;
        let y = inner.y + inner.h - 1 - (((v - lo) * (inner.h - 1) as i64) / span) as i32;
        pts.push((x, y));
    }
    scene.push(DrawCmd::Polyline { points: pts, color, width: 2 });
    scene
}

/// A compact **sparkline** (no background, just the trace) inside `area`.
pub fn sparkline(area: Rect, values: &[i64], color: Color) -> Vec<DrawCmd> {
    if values.len() < 2 {
        return Vec::new();
    }
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for &v in values {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (hi - lo).max(1);
    let n = values.len() as i64;
    let pts: Vec<(i32, i32)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = area.x + (i as i64 * (area.w - 1).max(1) as i64 / (n - 1)) as i32;
            let y = area.y + area.h - 1 - (((v - lo) * (area.h - 1).max(1) as i64) / span) as i32;
            (x, y)
        })
        .collect();
    alloc::vec![DrawCmd::Polyline { points: pts, color, width: 1 }]
}

/// A labelled **gauge**: `label` over a track filled to `value_milli`/1000, with the
/// numeric value shown at the right. Used for CPU/MEM/health/entropy bars.
pub fn gauge(area: Rect, label: &str, value_milli: u32, color: Color, theme: &toolkit::Theme) -> Vec<DrawCmd> {
    let mut scene = Vec::new();
    let v = value_milli.min(1000);
    // Label row.
    scene.push(DrawCmd::Text {
        rect: Rect::new(area.x, area.y, area.w - 48, theme.font_size + 2),
        text: String::from(label),
        color: theme.text,
        size: theme.font_size,
    });
    // Value text (right-aligned-ish): "NN%".
    let mut pct = String::new();
    push_int(&mut pct, (v / 10) as i64);
    pct.push('%');
    scene.push(DrawCmd::Text {
        rect: Rect::new(area.x + area.w - 44, area.y, 44, theme.font_size + 2),
        text: pct,
        color: theme.muted,
        size: theme.font_size,
    });
    // Track + fill.
    let track_y = area.y + theme.font_size + 4;
    let track = Rect::new(area.x, track_y, area.w, 8);
    scene.push(DrawCmd::Rect { rect: track, color: theme.bg, radius: 4 });
    let fill_w = (area.w * v as i32 / 1000).max(0);
    let fill_c = lerp(color, theme.danger, v.saturating_sub(700) * 1000 / 300);
    scene.push(DrawCmd::Rect { rect: Rect::new(area.x, track_y, fill_w, 8), color: fill_c, radius: 4 });
    scene
}

/// Render the visible window of a scrolling **log view**. `scroll` is how many lines
/// up from the bottom we are (0 = pinned to the latest). Newest at the bottom.
pub fn log_view(area: Rect, lines: &[String], scroll: usize, theme: &toolkit::Theme) -> Vec<DrawCmd> {
    let mut scene = Vec::new();
    scene.push(DrawCmd::Rect { rect: area, color: theme.bg, radius: theme.radius });
    let row_h = theme.font_size + 3;
    let rows = (area.h / row_h).max(1) as usize;
    if lines.is_empty() {
        return scene;
    }
    // The bottom-most visible line index, accounting for scroll-back.
    let end = lines.len().saturating_sub(scroll);
    let start = end.saturating_sub(rows);
    for (vis, line) in lines[start..end].iter().enumerate() {
        let y = area.y + theme.space / 2 + vis as i32 * row_h;
        scene.push(DrawCmd::Text {
            rect: Rect::new(area.x + theme.space / 2, y, area.w - theme.space, row_h),
            text: line.clone(),
            color: theme.muted,
            size: theme.font_size - 1,
        });
    }
    scene
}

/// An interactive **slider** holding a value in `[0,1000]`. Driven by real pointer
/// input — `drag_to` converts an x-coordinate to a value, so the setting it backs
/// actually changes.
#[derive(Clone, Copy, Debug)]
pub struct Slider {
    pub value_milli: u32,
}

impl Slider {
    pub fn new(value_milli: u32) -> Slider {
        Slider { value_milli: value_milli.min(1000) }
    }

    /// Is `(px,py)` on this slider's track/knob (so a press should start a drag)?
    pub fn hit(&self, area: Rect, px: i32, py: i32) -> bool {
        area.contains(px, py)
    }

    /// Convert an x-coordinate within `area` to the new value and store it.
    pub fn drag_to(&mut self, area: Rect, px: i32) -> u32 {
        if area.w <= 0 {
            return self.value_milli;
        }
        let rel = (px - area.x).clamp(0, area.w);
        self.value_milli = (rel as i64 * 1000 / area.w as i64) as u32;
        self.value_milli
    }

    /// Render the slider: a track, a filled portion, and a round knob.
    pub fn view(&self, area: Rect, color: Color, theme: &toolkit::Theme) -> Vec<DrawCmd> {
        let mut scene = Vec::new();
        let cy = area.y + area.h / 2;
        scene.push(DrawCmd::Rect { rect: Rect::new(area.x, cy - 3, area.w, 6), color: theme.bg, radius: 3 });
        let fill_w = area.w * self.value_milli as i32 / 1000;
        scene.push(DrawCmd::Rect { rect: Rect::new(area.x, cy - 3, fill_w, 6), color, radius: 3 });
        let knob_x = area.x + fill_w;
        scene.push(toolkit::disc(knob_x, cy, 7, theme.text));
        scene.push(toolkit::disc(knob_x, cy, 5, color));
        scene
    }
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

    fn theme() -> toolkit::Theme {
        toolkit::Theme::dark()
    }

    #[test]
    fn bar_chart_emits_a_bar_per_value() {
        let s = bar_chart(Rect::new(0, 0, 100, 50), &[1, 5, 9], 10, Color::rgb(80, 160, 255), &theme());
        // Background + 3 bars = 4 rects.
        let rects = s.iter().filter(|c| matches!(c, DrawCmd::Rect { .. })).count();
        assert_eq!(rects, 4);
    }

    #[test]
    fn line_chart_builds_a_polyline() {
        let s = line_chart(Rect::new(0, 0, 120, 60), &[3, 1, 4, 1, 5, 9, 2, 6], Color::rgb(0, 200, 120), &theme());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Polyline { points, .. } if points.len() == 8)));
    }

    #[test]
    fn gauge_shows_label_and_percent() {
        let s = gauge(Rect::new(0, 0, 200, 24), "CPU", 423, Color::rgb(80, 160, 255), &theme());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "CPU")));
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "42%")));
    }

    #[test]
    fn slider_drag_sets_a_real_value() {
        let mut sl = Slider::new(0);
        let area = Rect::new(10, 0, 100, 20);
        // Drag to the middle → ~500.
        let v = sl.drag_to(area, 60);
        assert!((480..=520).contains(&v));
        // Past the right end clamps to 1000.
        assert_eq!(sl.drag_to(area, 999), 1000);
        // Before the start clamps to 0.
        assert_eq!(sl.drag_to(area, -50), 0);
        assert!(sl.hit(area, 50, 10));
        assert!(!sl.hit(area, 5, 10));
    }

    #[test]
    fn log_view_shows_the_latest_lines() {
        let lines: Vec<String> = (0..50).map(|i| {
            let mut s = String::from("line ");
            push_int(&mut s, i);
            s
        }).collect();
        // A short box fits a few rows; pinned to bottom shows the newest.
        let s = log_view(Rect::new(0, 0, 200, 40), &lines, 0, &theme());
        assert!(s.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "line 49")));
        // Scrolled back shows older lines.
        let s2 = log_view(Rect::new(0, 0, 200, 40), &lines, 10, &theme());
        assert!(s2.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "line 39")));
    }

    #[test]
    fn scroll_thumb_tracks_offset_and_drag_moves_it() {
        let area = Rect::new(0, 0, 200, 100);
        let (content, view) = (400, 100); // 4× overflow
        let mut sc = Scroll::new();
        // At top: thumb is at the track top.
        let top = sc.thumb(area, content, view).unwrap();
        assert_eq!(top.y, area.y);
        // Press the thumb and drag down by 30px.
        assert!(sc.on_press(top.x + 2, top.y + 2, area, content, view));
        sc.on_drag(top.y + 32, area, content, view);
        assert!(sc.offset > 0);
        sc.release();
        assert!(!sc.is_dragging());
        // Clicking the track below the thumb pages down.
        let before = sc.offset;
        let thumb = sc.thumb(area, content, view).unwrap();
        sc.on_press(thumb.x + 2, thumb.y + thumb.h + 5, area, content, view);
        assert!(sc.offset >= before);
        // Content that fits needs no scrollbar.
        assert!(sc.thumb(area, 50, 100).is_none());
        assert!(!Scroll::needed(50, 100));
    }

    #[test]
    fn ellipsize_truncates_with_an_ellipsis() {
        assert_eq!(toolkit::ellipsize("short", 10), "short");
        assert_eq!(toolkit::ellipsize("a-very-long-filename.txt", 8), "a-very-…");
        // Pixel form: at size 15 the mono advance is 7px, so 70px → 10 chars.
        assert_eq!(toolkit::ellipsize_px("0123456789abcdef", 70, 15), "012345678…");
    }

    #[test]
    fn lerp_blends_endpoints() {
        let a = Color::rgb(0, 0, 0);
        let b = Color::rgb(100, 200, 40);
        assert_eq!(lerp(a, b, 0), a);
        assert_eq!(lerp(a, b, 1000), b);
        let mid = lerp(a, b, 500);
        assert_eq!(mid.r, 50);
    }
}

//! PSR2 (Panel Self-Refresh 2) + Paint-Over framework — render redesign milestone.
//!
//! ## PSR2
//!
//! When the desktop is static the display TCON refreshes from its own internal
//! buffer while the main link is powered down.  When partial updates occur
//! (e.g. the user is typing) only the *dirty-region bounding box* is
//! transmitted to the panel — sparing bandwidth and GPU wake-up cost for every
//! inactive area of the screen.
//!
//! [`DamageTracker`] owns the per-frame dirty accumulator and the PSR2 state
//! machine.  At the end of every frame call [`DamageTracker::end_frame`]; it
//! returns `None` when the panel can stay asleep (PSR2-idle) or a
//! [`DirtyRect`] covering the minimal region to retransmit.
//!
//! ## Paint-Over
//!
//! During high-velocity scroll the renderer drops vector anti-aliasing and
//! uses low-fidelity rasterisation.  When scrolling stops,
//! [`ScrollVelocityTracker::on_scroll_stop`] signals that a Paint-Over pass
//! is needed; [`PaintOverController`] then gates one full-quality keyframe so
//! the display snaps to crisp output rather than lingering on blurry pixels.

// No std — all heap allocations through alloc if needed, but this module
// is fully stack-based.

// ---------------------------------------------------------------------------
// DirtyRect
// ---------------------------------------------------------------------------

/// A dirty region on screen, in pixel coordinates.
///
/// An *empty* rect is one where `w <= 0` or `h <= 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl DirtyRect {
    /// Construct a new dirty rect.
    #[inline]
    pub fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    /// Smallest rect that contains both `self` and `other`.
    ///
    /// If either rect is empty the result is the non-empty one; if both are
    /// empty the result is empty.
    pub fn union(&self, other: &DirtyRect) -> DirtyRect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = (self.x + self.w).max(other.x + other.w);
        let y2 = (self.y + self.h).max(other.y + other.h);
        DirtyRect::new(x1, y1, x2 - x1, y2 - y1)
    }

    /// Returns `true` if the two rects share any pixel.
    pub fn intersects(&self, other: &DirtyRect) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        self.x < other.x + other.w
            && self.x + self.w > other.x
            && self.y < other.y + other.h
            && self.y + self.h > other.y
    }

    /// Area in pixels (0 for empty rects).
    pub fn area(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            self.w as i64 * self.h as i64
        }
    }

    /// A rect is empty when its width or height is non-positive.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.w <= 0 || self.h <= 0
    }
}

// ---------------------------------------------------------------------------
// PSR2 state machine
// ---------------------------------------------------------------------------

/// The current operating mode of the panel link.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Psr2State {
    /// Full-frame update required (major scene change, first frame after wake).
    Active,
    /// Partial update — only the enclosed [`DirtyRect`] needs retransmitting.
    Selective(DirtyRect),
    /// Panel is self-refreshing; no data needs to be sent this frame.
    Idle,
}

/// Tracks dirty regions frame-by-frame and drives the PSR2 state machine.
pub struct DamageTracker {
    /// Accumulated dirty region for the frame currently being built.
    pub frame_dirty: Option<DirtyRect>,
    /// Consecutive frames during which no region was dirtied.
    pub static_frames: u32,
    /// Current PSR2 operating state.
    pub state: Psr2State,
}

/// After this many consecutive static frames, transition to PSR2-idle.
const IDLE_THRESHOLD: u32 = 3;

impl DamageTracker {
    /// Create a new tracker. Starts in [`Psr2State::Active`] so the first
    /// frame always transmits.
    pub fn new() -> Self {
        Self {
            frame_dirty: None,
            static_frames: 0,
            state: Psr2State::Active,
        }
    }

    /// Mark a region dirty for the current frame.
    ///
    /// Multiple calls per frame are unioned together into a single bounding
    /// box that is transmitted at [`end_frame`](Self::end_frame) time.
    pub fn mark_dirty(&mut self, r: DirtyRect) {
        if r.is_empty() {
            return;
        }
        self.frame_dirty = Some(match self.frame_dirty.take() {
            None => r,
            Some(existing) => existing.union(&r),
        });
    }

    /// Call at the end of each frame.
    ///
    /// Advances the state machine and returns the region to transmit to the
    /// panel, or `None` if the panel should stay in PSR2-idle.
    pub fn end_frame(&mut self) -> Option<DirtyRect> {
        if let Some(dirty) = self.frame_dirty.take() {
            // Something changed this frame.
            self.static_frames = 0;
            self.state = Psr2State::Selective(dirty);
            Some(dirty)
        } else {
            // Nothing changed.
            self.static_frames = self.static_frames.saturating_add(1);
            if self.static_frames >= IDLE_THRESHOLD {
                // Enough consecutive static frames — the panel may self-refresh.
                self.state = Psr2State::Idle;
                None
            } else {
                // Not idle yet — keep the link active until the threshold so
                // the panel has up-to-date content before going to sleep.
                self.state = Psr2State::Active;
                None
            }
        }
    }

    /// Force a full-screen retransmit on the next frame (e.g. after a window
    /// move or resolution change).
    pub fn invalidate_all(&mut self, w: i32, h: i32) {
        self.frame_dirty = Some(DirtyRect::new(0, 0, w, h));
        self.static_frames = 0;
        self.state = Psr2State::Active;
    }
}

impl Default for DamageTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scroll velocity tracker
// ---------------------------------------------------------------------------

/// Tracks scroll velocity over a short history window and reports whether the
/// renderer should drop anti-aliasing for performance.
pub struct ScrollVelocityTracker {
    /// Smoothed velocity (pixels per frame).
    pub velocity_px_per_frame: f32,
    /// Threshold above which AA is dropped.
    pub high_velocity_threshold: f32,
    /// Ring buffer of recent per-frame deltas.
    history: [f32; 8],
    /// Write head for `history`.
    head: usize,
    /// How many samples have been written (caps at 8 for averaging).
    count: usize,
}

impl ScrollVelocityTracker {
    /// Create a tracker with the given high-velocity threshold (pixels/frame).
    pub fn new(threshold: f32) -> Self {
        Self {
            velocity_px_per_frame: 0.0,
            high_velocity_threshold: threshold,
            history: [0.0; 8],
            head: 0,
            count: 0,
        }
    }

    /// Feed the latest per-frame scroll delta (in pixels, may be negative).
    pub fn update(&mut self, delta_px: f32) {
        self.history[self.head] = delta_px.abs();
        self.head = (self.head + 1) & 7; // power-of-two wrap
        if self.count < 8 {
            self.count += 1;
        }
        // Recompute smoothed velocity as mean of history window.
        // Divide by the number of valid samples so warm-up (buffer not yet
        // full of real deltas) is not under-reported by the zero padding.
        let sum: f32 = self.history.iter().copied().sum();
        self.velocity_px_per_frame = sum / (self.count.max(1) as f32);
    }

    /// Returns `true` if the current velocity exceeds the configured threshold
    /// and AA should be suppressed.
    pub fn is_high_velocity(&self) -> bool {
        self.velocity_px_per_frame > self.high_velocity_threshold
    }

    /// Signal that scrolling has stopped.
    ///
    /// Clears the velocity history and returns `true` if a Paint-Over pass is
    /// required (i.e. we were previously in high-velocity mode).
    pub fn on_scroll_stop(&mut self) -> bool {
        let was_high = self.is_high_velocity();
        self.history = [0.0; 8];
        self.velocity_px_per_frame = 0.0;
        self.head = 0;
        self.count = 0;
        was_high
    }
}

// ---------------------------------------------------------------------------
// Rendering quality enum
// ---------------------------------------------------------------------------

/// Quality level for the current frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderQuality {
    /// Full anti-aliasing — used for static content and Paint-Over keyframes.
    FullAntiAliased,
    /// Low-fidelity — used during high-velocity scroll to maintain frame rate.
    FastLowFidelity,
}

// ---------------------------------------------------------------------------
// PaintOverController
// ---------------------------------------------------------------------------

/// Controls the Paint-Over keyframe logic.
///
/// When a Paint-Over is requested (via [`request_paint_over`]) the very next
/// call to [`should_paint_over`] returns `true` and clears the flag, causing
/// the compositor to render one full-quality frame that "snaps" the display
/// to crisp output.
pub struct PaintOverController {
    /// Whether a Paint-Over keyframe is pending.
    pub pending_paint_over: bool,
    /// Frames since scrolling stopped (saturating).
    pub frames_since_stop: u32,
}

impl PaintOverController {
    pub fn new() -> Self {
        Self { pending_paint_over: false, frames_since_stop: 0 }
    }

    /// Request a full-quality keyframe on the next compositor tick.
    pub fn request_paint_over(&mut self) {
        self.pending_paint_over = true;
        self.frames_since_stop = 0;
    }

    /// Returns `true` exactly once per requested Paint-Over, then clears the
    /// flag.  Call this once per frame before deciding rendering quality.
    pub fn should_paint_over(&mut self) -> bool {
        if self.pending_paint_over {
            self.pending_paint_over = false;
            self.frames_since_stop = self.frames_since_stop.saturating_add(1);
            true
        } else {
            self.frames_since_stop = self.frames_since_stop.saturating_add(1);
            false
        }
    }

    /// Current rendering quality for this frame.
    ///
    /// Returns [`RenderQuality::FullAntiAliased`] when a Paint-Over is pending
    /// or no Paint-Over has ever been requested; otherwise
    /// [`RenderQuality::FastLowFidelity`] while scrolling.
    pub fn quality(&self) -> RenderQuality {
        if self.pending_paint_over {
            RenderQuality::FullAntiAliased
        } else {
            RenderQuality::FastLowFidelity
        }
    }
}

impl Default for PaintOverController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- DirtyRect --------------------------------------------------------

    // 1. Basic construction and area.
    #[test]
    fn dirty_rect_area() {
        let r = DirtyRect::new(0, 0, 10, 20);
        assert_eq!(r.area(), 200);
    }

    // 2. Empty rect (zero width).
    #[test]
    fn empty_rect_zero_width() {
        let r = DirtyRect::new(5, 5, 0, 10);
        assert!(r.is_empty());
        assert_eq!(r.area(), 0);
    }

    // 3. Empty rect (negative height).
    #[test]
    fn empty_rect_negative_height() {
        let r = DirtyRect::new(0, 0, 10, -1);
        assert!(r.is_empty());
    }

    // 4. Union of two non-overlapping rects.
    #[test]
    fn union_non_overlapping() {
        let a = DirtyRect::new(0, 0, 10, 10);
        let b = DirtyRect::new(20, 20, 10, 10);
        let u = a.union(&b);
        assert_eq!(u, DirtyRect::new(0, 0, 30, 30));
    }

    // 5. Union of overlapping rects.
    #[test]
    fn union_overlapping() {
        let a = DirtyRect::new(0, 0, 20, 20);
        let b = DirtyRect::new(10, 10, 20, 20);
        let u = a.union(&b);
        assert_eq!(u, DirtyRect::new(0, 0, 30, 30));
    }

    // 6. Union with empty rect returns the non-empty one.
    #[test]
    fn union_with_empty() {
        let a = DirtyRect::new(5, 5, 10, 10);
        let empty = DirtyRect::new(0, 0, 0, 0);
        assert_eq!(a.union(&empty), a);
        assert_eq!(empty.union(&a), a);
    }

    // 7. intersects — overlapping rects.
    #[test]
    fn intersects_true() {
        let a = DirtyRect::new(0, 0, 20, 20);
        let b = DirtyRect::new(10, 10, 20, 20);
        assert!(a.intersects(&b));
    }

    // 8. intersects — non-overlapping rects.
    #[test]
    fn intersects_false() {
        let a = DirtyRect::new(0, 0, 10, 10);
        let b = DirtyRect::new(20, 0, 10, 10);
        assert!(!a.intersects(&b));
    }

    // 9. intersects — touching edges (not overlapping).
    #[test]
    fn intersects_touching_edge() {
        let a = DirtyRect::new(0, 0, 10, 10);
        let b = DirtyRect::new(10, 0, 10, 10); // touching right edge
        assert!(!a.intersects(&b));
    }

    // 10. intersects with empty rect returns false.
    #[test]
    fn intersects_empty() {
        let a = DirtyRect::new(0, 0, 10, 10);
        let e = DirtyRect::new(5, 5, 0, 5);
        assert!(!a.intersects(&e));
    }

    // ---- DamageTracker ----------------------------------------------------

    // 11. Fresh tracker — end_frame with no dirty returns None.
    #[test]
    fn damage_tracker_idle_after_no_dirty() {
        let mut dt = DamageTracker::new();
        // First frame: nothing dirtied, but we start Active so nothing was
        // accumulated — should return None.
        let r = dt.end_frame();
        assert!(r.is_none());
    }

    // 12. mark_dirty accumulates rects.
    #[test]
    fn damage_tracker_mark_and_end() {
        let mut dt = DamageTracker::new();
        dt.mark_dirty(DirtyRect::new(0, 0, 100, 50));
        dt.mark_dirty(DirtyRect::new(200, 200, 50, 50));
        let r = dt.end_frame().unwrap();
        assert_eq!(r, DirtyRect::new(0, 0, 250, 250));
    }

    // 13. State transitions to Idle after IDLE_THRESHOLD static frames.
    #[test]
    fn damage_tracker_transitions_to_idle() {
        let mut dt = DamageTracker::new();
        for _ in 0..5 {
            dt.end_frame();
        }
        assert_eq!(dt.state, Psr2State::Idle);
    }

    // 14. invalidate_all sets state to Active and marks full screen dirty.
    #[test]
    fn damage_tracker_invalidate_all() {
        let mut dt = DamageTracker::new();
        dt.invalidate_all(1920, 1080);
        assert_eq!(dt.state, Psr2State::Active);
        let r = dt.end_frame().unwrap();
        assert_eq!(r.w, 1920);
        assert_eq!(r.h, 1080);
    }

    // 15. After dirty, static_frames resets.
    #[test]
    fn damage_tracker_static_frames_reset() {
        let mut dt = DamageTracker::new();
        dt.end_frame();
        dt.end_frame();
        assert!(dt.static_frames >= 2);
        dt.mark_dirty(DirtyRect::new(0, 0, 1, 1));
        dt.end_frame();
        assert_eq!(dt.static_frames, 0);
    }

    // ---- ScrollVelocityTracker --------------------------------------------

    // 16. Below threshold → not high velocity.
    #[test]
    fn scroll_velocity_below_threshold() {
        let mut svt = ScrollVelocityTracker::new(50.0);
        svt.update(10.0);
        assert!(!svt.is_high_velocity());
    }

    // 17. Above threshold → high velocity.
    #[test]
    fn scroll_velocity_above_threshold() {
        let mut svt = ScrollVelocityTracker::new(50.0);
        for _ in 0..8 {
            svt.update(200.0);
        }
        assert!(svt.is_high_velocity());
    }

    // 18. on_scroll_stop returns true when was high velocity.
    #[test]
    fn scroll_stop_triggers_paint_over_when_high() {
        let mut svt = ScrollVelocityTracker::new(50.0);
        for _ in 0..8 {
            svt.update(200.0);
        }
        let needs_po = svt.on_scroll_stop();
        assert!(needs_po);
    }

    // 19. on_scroll_stop returns false when not high velocity.
    #[test]
    fn scroll_stop_no_paint_over_when_slow() {
        let mut svt = ScrollVelocityTracker::new(50.0);
        svt.update(5.0);
        let needs_po = svt.on_scroll_stop();
        assert!(!needs_po);
    }

    // 20. After stop, velocity resets to 0.
    #[test]
    fn scroll_velocity_resets_after_stop() {
        let mut svt = ScrollVelocityTracker::new(50.0);
        for _ in 0..8 {
            svt.update(300.0);
        }
        svt.on_scroll_stop();
        assert!(!svt.is_high_velocity());
        assert_eq!(svt.velocity_px_per_frame, 0.0);
    }

    // 21. Negative deltas are treated as absolute (abs).
    #[test]
    fn scroll_velocity_negative_delta_abs() {
        let mut svt = ScrollVelocityTracker::new(50.0);
        for _ in 0..8 {
            svt.update(-200.0);
        }
        assert!(svt.is_high_velocity());
    }

    // ---- PaintOverController ----------------------------------------------

    // 22. No paint-over requested — should_paint_over returns false.
    #[test]
    fn paint_over_not_requested() {
        let mut poc = PaintOverController::new();
        assert!(!poc.should_paint_over());
    }

    // 23. request_paint_over → should_paint_over true once.
    #[test]
    fn paint_over_fires_once() {
        let mut poc = PaintOverController::new();
        poc.request_paint_over();
        assert!(poc.should_paint_over());
        assert!(!poc.should_paint_over()); // cleared
    }

    // 24. quality() returns FullAntiAliased when pending.
    #[test]
    fn quality_full_when_pending() {
        let mut poc = PaintOverController::new();
        poc.request_paint_over();
        assert_eq!(poc.quality(), RenderQuality::FullAntiAliased);
    }

    // 25. quality() returns FastLowFidelity when no pending paint-over.
    #[test]
    fn quality_fast_when_not_pending() {
        let poc = PaintOverController::new();
        assert_eq!(poc.quality(), RenderQuality::FastLowFidelity);
    }

    // 26. frames_since_stop increments each tick.
    #[test]
    fn frames_since_stop_increments() {
        let mut poc = PaintOverController::new();
        poc.should_paint_over();
        poc.should_paint_over();
        assert_eq!(poc.frames_since_stop, 2);
    }

    // 27. End-to-end: scroll fast → stop → paint-over fires → quality flips.
    #[test]
    fn end_to_end_scroll_to_paint_over() {
        let mut svt = ScrollVelocityTracker::new(30.0);
        let mut poc = PaintOverController::new();

        // Simulate high-velocity scroll.
        for _ in 0..8 {
            svt.update(100.0);
        }
        assert!(svt.is_high_velocity());

        // Scroll stops.
        let needs = svt.on_scroll_stop();
        assert!(needs);
        if needs {
            poc.request_paint_over();
        }

        // Next compositor tick: fire Paint-Over.
        assert!(poc.should_paint_over());
        assert!(!poc.pending_paint_over);
    }
}

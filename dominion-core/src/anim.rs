//! Frame scheduling & motion — jank-free, deadline-driven, reduced-motion aware
//! (see `docs/ui/rendering-and-toolkit.md` §"Frame scheduling").
//!
//! Two small, pure pieces:
//!
//! * [`FrameScheduler`] — an **Earliest-Deadline-First** present loop with a hard
//!   frame budget (e.g. 16.6 ms at 60 Hz). A frame that renders within budget is
//!   presented; one that overruns is **dropped** (the compositor keeps the last
//!   complete scene) rather than tearing or stalling. Under memory/compute pressure
//!   it **sheds** non-essential animation first (the resource-governor contract).
//! * [`Tween`] — interpolates a value over a duration with an ease-out curve, and
//!   honours **reduced-motion** (jumps straight to the end). Integer math, no floats.
//!
//! Pure, safe `no_std`.

/// Microseconds.
pub type Micros = u64;

/// What happened to a submitted frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameOutcome {
    /// Rendered within budget — presented this frame.
    Presented,
    /// Overran the budget — dropped (the previous complete scene stays up).
    Dropped,
}

/// An EDF present loop with a fixed per-frame budget.
#[derive(Clone, Debug)]
pub struct FrameScheduler {
    budget_us: Micros,
    presented: u64,
    dropped: u64,
}

impl FrameScheduler {
    /// A scheduler targeting `fps` (e.g. 60 → ~16 666 µs budget).
    pub fn new(fps: u32) -> FrameScheduler {
        let budget_us = if fps == 0 { 16_666 } else { 1_000_000 / fps as u64 };
        FrameScheduler { budget_us, presented: 0, dropped: 0 }
    }

    pub fn budget_us(&self) -> Micros {
        self.budget_us
    }

    /// Submit a frame that took `render_us` to produce. EDF: present if it met the
    /// deadline, otherwise drop and keep the last complete scene.
    pub fn submit(&mut self, render_us: Micros) -> FrameOutcome {
        if render_us <= self.budget_us {
            self.presented += 1;
            FrameOutcome::Presented
        } else {
            self.dropped += 1;
            FrameOutcome::Dropped
        }
    }

    pub fn presented(&self) -> u64 {
        self.presented
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Under pressure, should non-essential animation be shed this frame? The
    /// resource governor's degradation level (0 = comfortable, higher = tighter)
    /// drives this — at any pressure above comfortable, drop decorative motion.
    pub fn shed_animation(pressure_level: u8) -> bool {
        pressure_level > 0
    }
}

/// Ease-out-quadratic on a per-mille parameter `t ∈ [0,1000]` → `[0,1000]`:
/// `1 − (1 − t)²`. Fast start, gentle finish — the "motion with meaning" curve.
fn ease_out(t_milli: i64) -> i64 {
    let t = t_milli.clamp(0, 1000);
    let inv = 1000 - t;
    1000 - inv * inv / 1000
}

/// A value interpolated over a duration, with ease-out and reduced-motion support.
#[derive(Clone, Copy, Debug)]
pub struct Tween {
    start: i64,
    end: i64,
    duration_us: Micros,
    reduced: bool,
}

impl Tween {
    pub fn new(start: i64, end: i64, duration_us: Micros) -> Tween {
        Tween { start, end, duration_us, reduced: false }
    }

    /// Respect the reduced-motion accessibility preference (`a11y.rs`): the tween
    /// snaps straight to its end with no in-between motion.
    pub fn reduced_motion(mut self, reduced: bool) -> Tween {
        self.reduced = reduced;
        self
    }

    /// The value at `elapsed_us`. Clamps to `start` at 0 and `end` at/after the
    /// duration; reduced-motion returns `end` immediately.
    pub fn value(&self, elapsed_us: Micros) -> i64 {
        if self.reduced || self.duration_us == 0 || elapsed_us >= self.duration_us {
            return self.end;
        }
        let t = (elapsed_us as i64 * 1000) / self.duration_us as i64;
        let e = ease_out(t);
        self.start + (self.end - self.start) * e / 1000
    }

    /// Is the tween finished at `elapsed_us`?
    pub fn done(&self, elapsed_us: Micros) -> bool {
        self.reduced || elapsed_us >= self.duration_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_presents_in_budget_drops_overruns() {
        let mut s = FrameScheduler::new(60);
        // 60 fps ≈ 16 666 µs budget.
        assert!(s.budget_us() >= 16_000 && s.budget_us() <= 17_000);
        assert_eq!(s.submit(10_000), FrameOutcome::Presented);
        assert_eq!(s.submit(20_000), FrameOutcome::Dropped); // overran → dropped
        assert_eq!(s.submit(16_666), FrameOutcome::Presented); // exactly on deadline
        assert_eq!(s.presented(), 2);
        assert_eq!(s.dropped(), 1);
    }

    #[test]
    fn pressure_sheds_animation() {
        assert!(!FrameScheduler::shed_animation(0)); // comfortable → keep motion
        assert!(FrameScheduler::shed_animation(1)); // tight → shed
        assert!(FrameScheduler::shed_animation(2)); // critical → shed
    }

    #[test]
    fn tween_interpolates_with_ease_out() {
        let t = Tween::new(0, 100, 1000);
        assert_eq!(t.value(0), 0); // start
        assert_eq!(t.value(1000), 100); // end
        assert_eq!(t.value(2000), 100); // past the end clamps
        // Ease-out is past the linear midpoint at the half-time (fast start).
        let mid = t.value(500);
        assert!(mid > 50 && mid < 100, "ease-out half should exceed linear 50: {mid}");
        assert!(!t.done(500));
        assert!(t.done(1000));
    }

    #[test]
    fn reduced_motion_snaps_to_end() {
        let t = Tween::new(0, 100, 1000).reduced_motion(true);
        assert_eq!(t.value(0), 100); // no in-between motion
        assert!(t.done(0));
    }

    #[test]
    fn tween_handles_descending_and_zero_duration() {
        let down = Tween::new(100, 0, 1000);
        assert_eq!(down.value(0), 100);
        assert_eq!(down.value(1000), 0);
        assert!(down.value(500) < 100 && down.value(500) > 0);
        // Zero-duration tween is instantly at its end.
        let instant = Tween::new(5, 9, 0);
        assert_eq!(instant.value(0), 9);
    }
}

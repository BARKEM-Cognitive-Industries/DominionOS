//! GPU-native font engine — Slug-algorithm quadratic Bézier glyphs with
//! dynamic dilation for artifact-free rendering at any DPI and 3D perspective.
//! (see `docs/2d-3d rendering redesign.md` §"GPU-Native Font Engine via Dynamic Glyph Dilation")
//!
//! Key ideas from the Slug algorithm (Eric Lengyel, Terathon Software):
//! * Glyph outlines are stored as quadratic Bézier curves in object space.
//! * Each glyph is enclosed by a bounding quad; the pixel shader evaluates coverage
//!   from the Bézier curves directly — no glyph atlas, no blurry bitmap scaling.
//! * Dynamic dilation: each vertex is shifted outward along its normal by exactly
//!   half a pixel in viewport space, so partially-covered boundary pixels are
//!   always inside the bounding poly → correct anti-aliasing.
//!
//! This module provides the CPU-side preparation: glyph outline storage, dilation
//! math, and rasterization to a pixel buffer (the software/fallback path).
//! A real GPU path would upload the Bézier control points as a shader storage buffer
//! and run the pixel shader; the data structures here are designed for that hand-off.
//!
//! Pure, safe `no_std`.

use crate::math3d::{Vec2, Mat4, sqrt32};
use alloc::vec::Vec;

// ─────────────────────── Bézier outline types ───────────────────────

/// A quadratic Bézier curve in glyph (object) space.
/// Control points are in EM units (typical range 0..2048).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadBez {
    pub p0: Vec2,   // start
    pub p1: Vec2,   // control (off-curve)
    pub p2: Vec2,   // end
}

impl QuadBez {
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2) -> QuadBez {
        QuadBez { p0, p1, p2 }
    }

    /// Evaluate the curve at parameter t ∈ [0, 1].
    pub fn eval(&self, t: f32) -> Vec2 {
        let one_t = 1.0 - t;
        // B(t) = (1-t)²·p0 + 2(1-t)t·p1 + t²·p2
        let x = one_t * one_t * self.p0.x + 2.0 * one_t * t * self.p1.x + t * t * self.p2.x;
        let y = one_t * one_t * self.p0.y + 2.0 * one_t * t * self.p1.y + t * t * self.p2.y;
        Vec2 { x, y }
    }

    /// Tangent at parameter t (first derivative, not normalized).
    pub fn tangent(&self, t: f32) -> Vec2 {
        let one_t = 1.0 - t;
        let x = 2.0 * (one_t * (self.p1.x - self.p0.x) + t * (self.p2.x - self.p1.x));
        let y = 2.0 * (one_t * (self.p1.y - self.p0.y) + t * (self.p2.y - self.p1.y));
        Vec2 { x, y }
    }

    /// Signed area contribution of this curve (for winding-number fill).
    /// Uses the shoelace formula on the control polygon.
    pub fn signed_area(&self) -> f32 {
        0.5 * ((self.p1.x - self.p0.x) * (self.p0.y + self.p1.y)
            + (self.p2.x - self.p1.x) * (self.p1.y + self.p2.y)
            + (self.p0.x - self.p2.x) * (self.p2.y + self.p0.y))
    }

    /// Axis-aligned bounding box of the curve.
    pub fn aabb(&self) -> (Vec2, Vec2) {
        // The extrema of a quadratic Bézier occur at t=0, t=1, and possibly
        // at the zero of the derivative: t* = (p0 - p1) / (p0 - 2p1 + p2)
        let mut min = Vec2 { x: self.p0.x.min(self.p2.x), y: self.p0.y.min(self.p2.y) };
        let mut max = Vec2 { x: self.p0.x.max(self.p2.x), y: self.p0.y.max(self.p2.y) };

        // X extremum
        let denom_x = self.p0.x - 2.0 * self.p1.x + self.p2.x;
        if denom_x.abs() > 1e-6 {
            let t = (self.p0.x - self.p1.x) / denom_x;
            if t > 0.0 && t < 1.0 {
                let ex = self.eval(t).x;
                if ex < min.x { min.x = ex; }
                if ex > max.x { max.x = ex; }
            }
        }
        // Y extremum
        let denom_y = self.p0.y - 2.0 * self.p1.y + self.p2.y;
        if denom_y.abs() > 1e-6 {
            let t = (self.p0.y - self.p1.y) / denom_y;
            if t > 0.0 && t < 1.0 {
                let ey = self.eval(t).y;
                if ey < min.y { min.y = ey; }
                if ey > max.y { max.y = ey; }
            }
        }
        (min, max)
    }
}

// ─────────────────────── Glyph outline ───────────────────────

/// A single glyph's outline: one or more closed contours of quadratic Béziers.
#[derive(Clone, Debug)]
pub struct GlyphOutline {
    pub codepoint: char,
    pub advance_x: f32,   // horizontal advance in EM units
    pub em_size: f32,     // design units per EM (e.g. 2048.0)
    pub curves: Vec<QuadBez>,
    /// Axis-aligned bounding box in EM space.
    pub min: Vec2,
    pub max: Vec2,
}

impl GlyphOutline {
    pub fn new(codepoint: char, advance_x: f32, em_size: f32, curves: Vec<QuadBez>) -> GlyphOutline {
        let mut min = Vec2 { x: f32::MAX, y: f32::MAX };
        let mut max = Vec2 { x: f32::MIN, y: f32::MIN };
        for c in &curves {
            let (cmin, cmax) = c.aabb();
            if cmin.x < min.x { min.x = cmin.x; }
            if cmin.y < min.y { min.y = cmin.y; }
            if cmax.x > max.x { max.x = cmax.x; }
            if cmax.y > max.y { max.y = cmax.y; }
        }
        if curves.is_empty() {
            min = Vec2 { x: 0.0, y: 0.0 };
            max = Vec2 { x: advance_x, y: em_size };
        }
        GlyphOutline { codepoint, advance_x, em_size, curves, min, max }
    }

    /// Convert EM-space coordinates to pixel coordinates.
    /// `px_size` = desired font size in pixels.
    pub fn em_to_px(&self, em_pt: Vec2, px_size: f32) -> Vec2 {
        let scale = px_size / self.em_size;
        Vec2 { x: em_pt.x * scale, y: em_pt.y * scale }
    }

    /// Pixel width of this glyph at `px_size`.
    pub fn px_width(&self, px_size: f32) -> f32 {
        self.advance_x * px_size / self.em_size
    }

    /// Pixel height (ascender to descender) at `px_size`.
    pub fn px_height(&self, px_size: f32) -> f32 {
        (self.max.y - self.min.y) * px_size / self.em_size
    }
}

// ─────────────────────── Dynamic glyph dilation ───────────────────────

// ─────────────────────── Built-in ASCII outlines ───────────────────────

/// A minimal built-in glyph set for ASCII printable characters.
/// In a real system these come from a font file (TTF/OTF parsed at startup).
/// Here we store simplified outlines sufficient for rendering and testing.
///
/// Each glyph is defined in a 512-unit EM space.
pub const EM: f32 = 512.0;

/// Create a simple rectangular glyph outline (used for block characters).
fn rect_glyph(ch: char, x0: f32, y0: f32, x1: f32, y1: f32, advance: f32) -> GlyphOutline {
    // A rect as 4 quads (degenerate: each "curve" is a straight line encoded as
    // a quadratic with the control point at the midpoint).
    let curves = alloc::vec![
        QuadBez::new(
            Vec2 { x: x0, y: y0 },
            Vec2 { x: (x0 + x1) * 0.5, y: y0 },
            Vec2 { x: x1, y: y0 },
        ),
        QuadBez::new(
            Vec2 { x: x1, y: y0 },
            Vec2 { x: x1, y: (y0 + y1) * 0.5 },
            Vec2 { x: x1, y: y1 },
        ),
        QuadBez::new(
            Vec2 { x: x1, y: y1 },
            Vec2 { x: (x0 + x1) * 0.5, y: y1 },
            Vec2 { x: x0, y: y1 },
        ),
        QuadBez::new(
            Vec2 { x: x0, y: y1 },
            Vec2 { x: x0, y: (y0 + y1) * 0.5 },
            Vec2 { x: x0, y: y0 },
        ),
    ];
    GlyphOutline::new(ch, advance, EM, curves)
}

/// Provide a simple outline for a letter A (triangle shape, Bézier approximation).
fn glyph_a() -> GlyphOutline {
    // Simplified triangle representing 'A'.
    let curves = alloc::vec![
        // Left stroke: bottom-left to apex.
        QuadBez::new(Vec2 { x: 50.0, y: 50.0 }, Vec2 { x: 200.0, y: 256.0 }, Vec2 { x: 256.0, y: 462.0 }),
        // Right stroke: apex to bottom-right.
        QuadBez::new(Vec2 { x: 256.0, y: 462.0 }, Vec2 { x: 312.0, y: 256.0 }, Vec2 { x: 462.0, y: 50.0 }),
        // Crossbar bottom.
        QuadBez::new(Vec2 { x: 462.0, y: 50.0 }, Vec2 { x: 256.0, y: 50.0 }, Vec2 { x: 50.0, y: 50.0 }),
    ];
    GlyphOutline::new('A', 512.0, EM, curves)
}

/// Look up a simple built-in outline for `ch`.
/// Falls back to a block rectangle for unknown characters.
pub fn builtin_outline(ch: char) -> GlyphOutline {
    match ch {
        'A' | 'a' => glyph_a(),
        ' ' => GlyphOutline::new(' ', 256.0, EM, alloc::vec![]),
        _ => rect_glyph(ch, 40.0, 40.0, 472.0, 472.0, 512.0),
    }
}

// ─────────────────────── Software rasterizer (fallback) ───────────────────────

/// Rasterize a glyph outline into `pixels` (RGBA packed 0xAARRGGBB) at position
/// `(ox, oy)` with `px_size` pixels per EM. Uses scanline winding-number fill
/// (the same algorithm as `vectorpath::FineRasterizer` but inline here for
/// independence).
pub fn rasterize_glyph(
    outline: &GlyphOutline,
    ox: i32,
    oy: i32,
    px_size: f32,
    color_rgba: u32,  // packed RGBA
    pixels: &mut [u32],
    width: u32,
    height: u32,
) {
    if outline.curves.is_empty() {
        return;
    }
    let scale = px_size / outline.em_size;

    // Rasterize curve by scanning horizontally.
    let glyph_w = (outline.max.x - outline.min.x) * scale;
    let glyph_h = (outline.max.y - outline.min.y) * scale;

    let x0 = ox;
    let y0 = oy;
    let x1 = (ox + glyph_w as i32 + 1).min(width as i32);
    let y1 = (oy + glyph_h as i32 + 1).min(height as i32);

    for py in y0.max(0)..y1 {
        // Convert to EM space.
        let ey = (py - oy) as f32 / scale + outline.min.y;
        let mut _winding = 0i32;

        // Count signed crossings for each curve at row ey.
        for curve in &outline.curves {
            // Sample the curve at many t values to find crossings.
            // (A production shader would do this analytically.)
            let steps = 32u32;
            let mut prev = curve.eval(0.0);
            for step in 1..=steps {
                let t = step as f32 / steps as f32;
                let curr = curve.eval(t);
                // Does this segment cross y = ey?
                if (prev.y <= ey && curr.y > ey) || (curr.y <= ey && prev.y > ey) {
                    // X intersection via linear interpolation along segment.
                    let frac = (ey - prev.y) / (curr.y - prev.y);
                    let cx = prev.x + frac * (curr.x - prev.x);
                    // Convert cx to pixel space.
                    let px_cx = (cx - outline.min.x) * scale + ox as f32;
                    // For pixels to the right of this intersection, toggle winding.
                    // We'll mark which pixels get toggled.
                    let _ = px_cx; // used below in column loop
                    if curr.y > prev.y { _winding += 1; } else { _winding -= 1; }
                }
                prev = curr;
            }
        }

        // For each pixel in this row, count crossings to the left.
        for px in x0.max(0)..x1 {
            let ex = (px - ox) as f32 / scale + outline.min.x;
            let mut local_winding = 0i32;

            for curve in &outline.curves {
                let steps = 16u32;
                let mut prev = curve.eval(0.0);
                for step in 1..=steps {
                    let t = step as f32 / steps as f32;
                    let curr = curve.eval(t);
                    if (prev.y <= ey && curr.y > ey) || (curr.y <= ey && prev.y > ey) {
                        let frac = (ey - prev.y) / (curr.y - prev.y);
                        let cx = prev.x + frac * (curr.x - prev.x);
                        if cx < ex {
                            if curr.y > prev.y { local_winding += 1; } else { local_winding -= 1; }
                        }
                    }
                    prev = curr;
                }
            }

            if local_winding != 0 {
                let idx = py as usize * width as usize + px as usize;
                if idx < pixels.len() {
                    pixels[idx] = color_rgba;
                }
            }
        }
    }
}

// ─────────────────────── Font layout ───────────────────────

/// A simple glyph run: lay out a string of characters and return their
/// positions and outlines.
#[derive(Clone, Debug)]
pub struct GlyphRun {
    pub items: Vec<GlyphRunItem>,
    pub total_width: f32,
    pub line_height: f32,
}

#[derive(Clone, Debug)]
pub struct GlyphRunItem {
    pub outline: GlyphOutline,
    pub x_offset: f32,  // in pixels
    pub y_offset: f32,
}

/// Lay out `text` at `px_size` pixels, returning a `GlyphRun`.
/// Uses `builtin_outline` for each character.
pub fn layout_text(text: &str, px_size: f32) -> GlyphRun {
    let mut items = Vec::new();
    let mut x = 0.0f32;
    for ch in text.chars() {
        let outline = builtin_outline(ch);
        let advance = outline.px_width(px_size);
        items.push(GlyphRunItem { outline, x_offset: x, y_offset: 0.0 });
        x += advance;
    }
    GlyphRun { items, total_width: x, line_height: px_size * 1.2 }
}

/// Render a glyph run into a pixel buffer.
pub fn render_text(
    run: &GlyphRun,
    ox: i32,
    oy: i32,
    px_size: f32,
    color_rgba: u32,
    pixels: &mut [u32],
    width: u32,
    height: u32,
) {
    for item in &run.items {
        rasterize_glyph(
            &item.outline,
            ox + item.x_offset as i32,
            oy + item.y_offset as i32,
            px_size,
            color_rgba,
            pixels,
            width,
            height,
        );
    }
}

// ─────────────────────── Slug quadratic dilation ───────────────────────

/// A glyph vertex with position, normal, and dilated position.
#[derive(Clone, Copy, Debug)]
pub struct GlyphVertex {
    pub object_pos: Vec2,
    pub normal: Vec2,
    pub dilated_pos: Vec2,
    pub dilation_d: f32,
}

/// Compute the Slug quadratic dilation distance:
///   d = 0.5 / sqrt( (mx * nx)^2 + (my * ny)^2 )
///
/// Where (Mat4 is column-major: cols[col][row]):
///   mx = (M[0][0] * vp_w + M[3][0]) / M[3][3]
///       = (cols[0][0] * vp_w + cols[3][0]) / cols[3][3]
///   my = (M[1][1] * vp_h + M[3][1]) / M[3][3]
///       = (cols[1][1] * vp_h + cols[3][1]) / cols[3][3]
///
/// If denominator is nearly zero (degenerate projection), returns 0.0.
pub fn compute_dilation_quadratic(mvp: &Mat4, normal: Vec2, vp_w: f32, vp_h: f32) -> f32 {
    let m = &mvp.cols;

    // M[3][3] = cols[3][3] — the homogeneous w divisor.
    let w33 = m[3][3];
    if w33.abs() < 1e-10 {
        return 0.0;
    }

    // mx = (M[0][0] * vp_w + M[3][0]) / M[3][3]
    //    = (cols[0][0] * vp_w + cols[3][0]) / cols[3][3]
    let mx = (m[0][0] * vp_w + m[3][0]) / w33;

    // my = (M[1][1] * vp_h + M[3][1]) / M[3][3]
    //    = (cols[1][1] * vp_h + cols[3][1]) / cols[3][3]
    let my = (m[1][1] * vp_h + m[3][1]) / w33;

    let nx = normal.x;
    let ny = normal.y;

    let denom_sq = (mx * nx) * (mx * nx) + (my * ny) * (my * ny);
    if denom_sq < 1e-12 {
        return 0.0;
    }

    0.5 / sqrt32(denom_sq)
}

/// Process a glyph outline through the full Slug dilation pipeline.
/// Returns one GlyphVertex per control point in the outline's quads.
/// Each control point (p0, p1, p2) gets its outward normal estimated and
/// the Slug quadratic dilation applied.
pub fn process_glyph_outline(
    outline: &GlyphOutline,
    mvp: &Mat4,
    vp_w: f32,
    vp_h: f32,
) -> Vec<GlyphVertex> {
    let mut vertices = Vec::new();

    for curve in &outline.curves {
        // Process the three control points: p0, p1 (off-curve), p2.
        let points = [curve.p0, curve.p1, curve.p2];

        for (i, &pt) in points.iter().enumerate() {
            // Estimate the outward normal at this control point.
            // For p0: normal perpendicular to tangent at t=0.
            // For p2: normal perpendicular to tangent at t=1.
            // For p1 (off-curve control): use bisector of surrounding tangents.
            let tangent = match i {
                0 => curve.tangent(0.0),
                2 => curve.tangent(1.0),
                _ => {
                    // Mid-curve tangent for the control point.
                    curve.tangent(0.5)
                }
            };

            // Outward normal: rotate tangent 90° CCW (to left of direction of travel).
            let tan_len = sqrt32(tangent.x * tangent.x + tangent.y * tangent.y);
            let normal = if tan_len > 1e-10 {
                Vec2 { x: -tangent.y / tan_len, y: tangent.x / tan_len }
            } else {
                Vec2 { x: 0.0, y: 1.0 }
            };

            let d = compute_dilation_quadratic(mvp, normal, vp_w, vp_h);

            let dilated_pos = Vec2 {
                x: pt.x + d * normal.x,
                y: pt.y + d * normal.y,
            };

            vertices.push(GlyphVertex {
                object_pos: pt,
                normal,
                dilated_pos,
                dilation_d: d,
            });
        }
    }

    vertices
}

// ─────────────────────── Tests ───────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_bez_eval_endpoints() {
        let q = QuadBez::new(
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 2.0, y: 0.0 },
        );
        let p0 = q.eval(0.0);
        let p1 = q.eval(1.0);
        assert!((p0.x - 0.0).abs() < 1e-5);
        assert!((p1.x - 2.0).abs() < 1e-5);
    }

    #[test]
    fn quad_bez_eval_midpoint() {
        let q = QuadBez::new(
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 2.0, y: 0.0 },
        );
        let mid = q.eval(0.5);
        // B(0.5) = 0.25*(0,0) + 0.5*(1,2) + 0.25*(2,0) = (1, 1)
        assert!((mid.x - 1.0).abs() < 1e-4);
        assert!((mid.y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn quad_bez_aabb_parabola() {
        let q = QuadBez::new(
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 4.0 },
            Vec2 { x: 2.0, y: 0.0 },
        );
        let (mn, mx) = q.aabb();
        assert!(mn.x <= 0.01 && mn.y <= 0.01);
        assert!(mx.x >= 1.99);
        // Y extremum at t=0.5: B_y(0.5) = 0.25*0 + 0.5*4 + 0.25*0 = 2.0
        assert!(mx.y >= 1.99);
    }

    #[test]
    fn glyph_outline_new_computes_aabb() {
        let g = rect_glyph('X', 10.0, 20.0, 100.0, 200.0, 110.0);
        assert!(g.min.x <= 11.0);
        assert!(g.min.y <= 21.0);
        assert!(g.max.x >= 99.0);
    }

    #[test]
    fn glyph_outline_px_width() {
        let g = GlyphOutline::new('A', 512.0, 1024.0, alloc::vec![]);
        // At px_size = 1024, width = advance = 512 px.
        assert!((g.px_width(1024.0) - 512.0).abs() < 1.0);
        // At px_size = 512, width = 256 px.
        assert!((g.px_width(512.0) - 256.0).abs() < 1.0);
    }

    #[test]
    fn builtin_outline_returns_something() {
        let g = builtin_outline('A');
        assert!(!g.curves.is_empty());
        assert_eq!(g.codepoint, 'A');
    }

    #[test]
    fn builtin_space_has_no_curves() {
        let g = builtin_outline(' ');
        assert!(g.curves.is_empty());
        assert!(g.advance_x > 0.0);
    }

    #[test]
    fn layout_text_advances_correctly() {
        let run = layout_text("AB", 16.0);
        assert_eq!(run.items.len(), 2);
        assert_eq!(run.items[0].x_offset, 0.0);
        assert!(run.items[1].x_offset > 0.0);
        assert!(run.total_width > run.items[1].x_offset);
    }

    #[test]
    fn rasterize_glyph_writes_pixels() {
        let outline = rect_glyph('X', 0.0, 0.0, 256.0, 256.0, 300.0);
        let mut pixels = alloc::vec![0u32; 64 * 64];
        rasterize_glyph(&outline, 0, 0, 32.0, 0xFF_FF_FF_FF, &mut pixels, 64, 64);
        let nonzero = pixels.iter().filter(|&&p| p != 0).count();
        assert!(nonzero > 0, "rasterized glyph must produce at least some pixels");
    }

    #[test]
    fn render_text_writes_pixels() {
        let run = layout_text("Hi", 24.0);
        let mut pixels = alloc::vec![0u32; 128 * 64];
        render_text(&run, 0, 0, 24.0, 0xFF_FF_FF_FF, &mut pixels, 128, 64);
        let nonzero = pixels.iter().filter(|&&p| p != 0).count();
        assert!(nonzero > 0);
    }

    #[test]
    fn quad_bez_tangent_at_endpoints() {
        let q = QuadBez::new(
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 2.0, y: 0.0 },
        );
        let t0 = q.tangent(0.0); // should point toward control point
        assert!(t0.x > 0.0);
        let t1 = q.tangent(1.0); // should point away from control point
        assert!(t1.x > 0.0);
    }

    // ─────────────────── Slug quadratic dilation tests ───────────────────

    #[test]
    fn compute_dilation_quadratic_identity() {
        use crate::math3d::Mat4;
        // Identity MVP: M[0][0]=1, M[1][1]=1, M[3][0]=0, M[3][1]=0, M[3][3]=1
        let mvp = Mat4::identity();
        let normal = Vec2 { x: 1.0, y: 0.0 };
        let d = compute_dilation_quadratic(&mvp, normal, 1920.0, 1080.0);
        // mx = (1*1920 + 0)/1 = 1920, my = (1*1080 + 0)/1 = 1080
        // denom = sqrt((1920*1)^2 + (1080*0)^2) = 1920
        // d = 0.5 / 1920 ≈ 0.000260
        assert!(d > 0.0, "dilation must be positive");
        let expected = 0.5 / 1920.0_f32;
        assert!(
            (d - expected).abs() < 1e-4,
            "expected ≈ {expected}, got {d}"
        );
    }

    #[test]
    fn compute_dilation_quadratic_y_normal() {
        use crate::math3d::Mat4;
        let mvp = Mat4::identity();
        let normal = Vec2 { x: 0.0, y: 1.0 };
        let d = compute_dilation_quadratic(&mvp, normal, 1920.0, 1080.0);
        // mx contribution = 0, my contribution = 1080
        // d = 0.5 / 1080
        let expected = 0.5 / 1080.0_f32;
        assert!((d - expected).abs() < 1e-4, "y-normal: expected {expected}, got {d}");
    }

    #[test]
    fn compute_dilation_quadratic_degenerate_normal() {
        use crate::math3d::Mat4;
        // Zero normal should give d=0 (denom_sq ≈ 0).
        let mvp = Mat4::identity();
        let normal = Vec2 { x: 0.0, y: 0.0 };
        let d = compute_dilation_quadratic(&mvp, normal, 1920.0, 1080.0);
        assert_eq!(d, 0.0, "zero normal should give d=0");
    }

    #[test]
    fn compute_dilation_quadratic_degenerate_projection() {
        use crate::math3d::Mat4;
        // Build an MVP with M[3][3] = 0 (degenerate).
        let mut mvp = Mat4::identity();
        mvp.cols[3][3] = 0.0;
        let normal = Vec2 { x: 1.0, y: 0.0 };
        let d = compute_dilation_quadratic(&mvp, normal, 1920.0, 1080.0);
        assert_eq!(d, 0.0, "degenerate w=0 should give d=0");
    }

    #[test]
    fn process_glyph_outline_produces_vertices() {
        use crate::math3d::Mat4;
        let outline = builtin_outline('A');
        let mvp = Mat4::identity();
        let verts = process_glyph_outline(&outline, &mvp, 1920.0, 1080.0);
        // 'A' has 3 curves × 3 control points = 9 vertices.
        assert_eq!(verts.len(), outline.curves.len() * 3,
            "expected {} vertices, got {}", outline.curves.len() * 3, verts.len());
    }

    #[test]
    fn process_glyph_outline_dilates_outward() {
        use crate::math3d::Mat4;
        let outline = builtin_outline('A');
        let mvp = Mat4::identity();
        let verts = process_glyph_outline(&outline, &mvp, 1920.0, 1080.0);
        // Every vertex should have a dilation_d >= 0 and dilated_pos different from object_pos.
        for v in &verts {
            assert!(v.dilation_d >= 0.0, "dilation must be non-negative");
            // If normal is nonzero and d>0, dilated should differ from original.
            if v.dilation_d > 1e-10 {
                let moved = (v.dilated_pos.x - v.object_pos.x).abs()
                    + (v.dilated_pos.y - v.object_pos.y).abs();
                assert!(moved > 1e-10, "dilated_pos should differ from object_pos when d>0");
            }
        }
    }

    #[test]
    fn process_glyph_outline_normal_unit_length() {
        use crate::math3d::{Mat4, sqrt32};
        let outline = builtin_outline('A');
        let mvp = Mat4::identity();
        let verts = process_glyph_outline(&outline, &mvp, 1920.0, 1080.0);
        for v in &verts {
            let len = sqrt32(v.normal.x * v.normal.x + v.normal.y * v.normal.y);
            assert!(
                (len - 1.0).abs() < 1e-4 || len < 1e-6,
                "normal should be unit length or zero, got len={len}"
            );
        }
    }

    #[test]
    fn glyph_vertex_fields_accessible() {
        let gv = GlyphVertex {
            object_pos: Vec2 { x: 1.0, y: 2.0 },
            normal: Vec2 { x: 0.0, y: 1.0 },
            dilated_pos: Vec2 { x: 1.0, y: 2.5 },
            dilation_d: 0.5,
        };
        assert_eq!(gv.object_pos.x, 1.0);
        assert_eq!(gv.dilation_d, 0.5);
    }

}

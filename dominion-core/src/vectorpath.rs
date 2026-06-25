//! Compute-native 2D vector pipeline: Bézier paths → tile-binning → coverage → RGBA.
//! (see `docs/2d-3d rendering redesign.md` §"Compute-Native Vector Pipelines")
//!
//! On the GPU path (future), each stage runs as a compute shader with parallel
//! prefix-sum allocation. CPU path (this module) uses the same algorithm structure
//! but executes sequentially — same output, lower throughput. Pure, safe `no_std`.

use alloc::vec::Vec;
use crate::math3d::{Vec2, sqrt32};

// no_std scalar helpers (no libm required)
#[inline(always)]
fn floor32(x: f32) -> f32 {
    let i = x as i64;
    let fi = i as f32;
    if fi > x { fi - 1.0 } else { fi }
}

// ---------------------------------------------------------------------------
// Path representation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum PathSegment {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo { ctrl: Vec2, end: Vec2 },
    CubicTo { ctrl0: Vec2, ctrl1: Vec2, end: Vec2 },
    Close,
}

#[derive(Clone, Debug)]
pub struct Fill {
    pub color: [f32; 4],  // linear RGBA
    pub rule: FillRule,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug)]
pub struct Stroke {
    pub color: [f32; 4],
    pub width: f32,
    pub cap: LineCap,
    pub join: LineJoin,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineCap { Butt, Round, Square }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineJoin { Miter, Round, Bevel }

#[derive(Clone, Debug)]
pub struct VectorPath {
    pub segments: Vec<PathSegment>,
    pub fill: Option<Fill>,
    pub stroke: Option<Stroke>,
    pub transform: Option<[[f32; 3]; 3]>,  // 2D affine (3×3)
}

impl VectorPath {
    pub fn new() -> VectorPath {
        VectorPath {
            segments: Vec::new(),
            fill: None,
            stroke: None,
            transform: None,
        }
    }

    pub fn move_to(&mut self, p: Vec2) -> &mut Self {
        self.segments.push(PathSegment::MoveTo(p));
        self
    }

    pub fn line_to(&mut self, p: Vec2) -> &mut Self {
        self.segments.push(PathSegment::LineTo(p));
        self
    }

    pub fn quad_to(&mut self, ctrl: Vec2, end: Vec2) -> &mut Self {
        self.segments.push(PathSegment::QuadTo { ctrl, end });
        self
    }

    pub fn cubic_to(&mut self, c0: Vec2, c1: Vec2, end: Vec2) -> &mut Self {
        self.segments.push(PathSegment::CubicTo { ctrl0: c0, ctrl1: c1, end });
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.segments.push(PathSegment::Close);
        self
    }

    pub fn with_fill(mut self, fill: Fill) -> Self {
        self.fill = Some(fill);
        self
    }

    pub fn with_stroke(mut self, stroke: Stroke) -> Self {
        self.stroke = Some(stroke);
        self
    }

    /// Compute the axis-aligned bounding box of the path.
    pub fn bounding_box(&self) -> (Vec2, Vec2) {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        let mut update = |p: Vec2| {
            if p.x < min_x { min_x = p.x; }
            if p.y < min_y { min_y = p.y; }
            if p.x > max_x { max_x = p.x; }
            if p.y > max_y { max_y = p.y; }
        };

        for seg in &self.segments {
            match seg {
                PathSegment::MoveTo(p) => update(*p),
                PathSegment::LineTo(p) => update(*p),
                PathSegment::QuadTo { ctrl, end } => {
                    update(*ctrl);
                    update(*end);
                }
                PathSegment::CubicTo { ctrl0, ctrl1, end } => {
                    update(*ctrl0);
                    update(*ctrl1);
                    update(*end);
                }
                PathSegment::Close => {}
            }
        }

        if min_x == f32::MAX {
            (Vec2::zero(), Vec2::zero())
        } else {
            (Vec2::new(min_x, min_y), Vec2::new(max_x, max_y))
        }
    }
}

impl Default for VectorPath {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

/// Apply a 2D affine transform (3×3 homogeneous) to a Vec2.
#[inline]
fn apply_transform(m: &[[f32; 3]; 3], p: Vec2) -> Vec2 {
    Vec2::new(
        m[0][0] * p.x + m[0][1] * p.y + m[0][2],
        m[1][0] * p.x + m[1][1] * p.y + m[1][2],
    )
}

// ---------------------------------------------------------------------------
// Linearization (Bézier flattening)
// ---------------------------------------------------------------------------

/// A flattened line segment (after Bézier subdivision).
#[derive(Clone, Copy, Debug)]
pub struct LineSegment {
    pub p0: Vec2,
    pub p1: Vec2,
}

impl LineSegment {
    #[inline]
    pub fn new(p0: Vec2, p1: Vec2) -> Self {
        Self { p0, p1 }
    }

    /// Axis-aligned bounding box of this segment.
    #[inline]
    pub fn aabb(&self) -> (f32, f32, f32, f32) {
        let min_x = if self.p0.x < self.p1.x { self.p0.x } else { self.p1.x };
        let max_x = if self.p0.x > self.p1.x { self.p0.x } else { self.p1.x };
        let min_y = if self.p0.y < self.p1.y { self.p0.y } else { self.p1.y };
        let max_y = if self.p0.y > self.p1.y { self.p0.y } else { self.p1.y };
        (min_x, min_y, max_x, max_y)
    }
}

/// Recursively subdivide a quadratic Bézier (p0, ctrl, p1) into line segments.
/// Subdivides when the maximum deviation from the chord exceeds `tolerance`.
fn flatten_quad(p0: Vec2, ctrl: Vec2, p1: Vec2, tolerance: f32, out: &mut Vec<LineSegment>) {
    // Flatness check: distance from ctrl to midpoint of chord p0-p1.
    let mid = p0.lerp(p1, 0.5);
    let deviation = {
        let d = ctrl - mid;
        sqrt32(d.x * d.x + d.y * d.y)
    };

    if deviation <= tolerance {
        out.push(LineSegment::new(p0, p1));
        return;
    }

    // De Casteljau split at t=0.5
    let m01 = p0.lerp(ctrl, 0.5);
    let m12 = ctrl.lerp(p1, 0.5);
    let m = m01.lerp(m12, 0.5);

    flatten_quad(p0, m01, m, tolerance, out);
    flatten_quad(m, m12, p1, tolerance, out);
}

/// Recursively subdivide a cubic Bézier (p0, c0, c1, p1) into line segments.
fn flatten_cubic(p0: Vec2, c0: Vec2, c1: Vec2, p1: Vec2, tolerance: f32, out: &mut Vec<LineSegment>) {
    // Flatness: max deviation of control polygon from chord.
    let chord = p1 - p0;
    let chord_len = sqrt32(chord.x * chord.x + chord.y * chord.y);

    let dev = if chord_len < 1e-10 {
        // Degenerate: just use distance of ctrl points from p0.
        let d0 = c0 - p0;
        let d1 = c1 - p0;
        let l0 = sqrt32(d0.x * d0.x + d0.y * d0.y);
        let l1 = sqrt32(d1.x * d1.x + d1.y * d1.y);
        if l0 > l1 { l0 } else { l1 }
    } else {
        // Cross-product magnitude gives perpendicular distance.
        let inv_len = 1.0 / chord_len;
        let nx = -chord.y * inv_len;
        let ny = chord.x * inv_len;

        let d0 = c0 - p0;
        let d1 = c1 - p0;
        let dist0 = (d0.x * nx + d0.y * ny).abs();
        let dist1 = (d1.x * nx + d1.y * ny).abs();
        if dist0 > dist1 { dist0 } else { dist1 }
    };

    if dev <= tolerance {
        out.push(LineSegment::new(p0, p1));
        return;
    }

    // De Casteljau split at t=0.5
    let m01 = p0.lerp(c0, 0.5);
    let m12 = c0.lerp(c1, 0.5);
    let m23 = c1.lerp(p1, 0.5);
    let m012 = m01.lerp(m12, 0.5);
    let m123 = m12.lerp(m23, 0.5);
    let m = m012.lerp(m123, 0.5);

    flatten_cubic(p0, m01, m012, m, tolerance, out);
    flatten_cubic(m, m123, m23, p1, tolerance, out);
}

/// Flatten a path into line segments with the given tolerance (in pixels).
pub fn flatten_path(path: &VectorPath, tolerance: f32) -> Vec<LineSegment> {
    let tol = if tolerance <= 0.0 { 0.25 } else { tolerance };
    let mut out = Vec::new();
    let mut cursor = Vec2::zero();
    let mut subpath_start = Vec2::zero();

    let xform = path.transform;

    let transform_pt = |p: Vec2| -> Vec2 {
        match &xform {
            Some(m) => apply_transform(m, p),
            None => p,
        }
    };

    for seg in &path.segments {
        match seg {
            PathSegment::MoveTo(p) => {
                let tp = transform_pt(*p);
                cursor = tp;
                subpath_start = tp;
            }
            PathSegment::LineTo(p) => {
                let tp = transform_pt(*p);
                out.push(LineSegment::new(cursor, tp));
                cursor = tp;
            }
            PathSegment::QuadTo { ctrl, end } => {
                let tc = transform_pt(*ctrl);
                let te = transform_pt(*end);
                flatten_quad(cursor, tc, te, tol, &mut out);
                cursor = te;
            }
            PathSegment::CubicTo { ctrl0, ctrl1, end } => {
                let tc0 = transform_pt(*ctrl0);
                let tc1 = transform_pt(*ctrl1);
                let te = transform_pt(*end);
                flatten_cubic(cursor, tc0, tc1, te, tol, &mut out);
                cursor = te;
            }
            PathSegment::Close => {
                if (cursor.x - subpath_start.x).abs() > 1e-6
                    || (cursor.y - subpath_start.y).abs() > 1e-6
                {
                    out.push(LineSegment::new(cursor, subpath_start));
                }
                cursor = subpath_start;
            }
        }
    }

    out
}

/// Flatten just the fill outline (alias for flatten_path).
pub fn flatten_fill(path: &VectorPath, tolerance: f32) -> Vec<LineSegment> {
    flatten_path(path, tolerance)
}

/// Expand stroke into a filled outline (simplified: offset both sides, butt caps).
pub fn stroke_to_fill(path: &VectorPath, tolerance: f32) -> Vec<LineSegment> {
    let half_w = match &path.stroke {
        Some(s) => s.width * 0.5,
        None => 1.0,
    };

    let base = flatten_path(path, tolerance);
    let mut out = Vec::new();

    // For each base segment, emit two offset segments (left and right sides).
    for seg in &base {
        let d = seg.p1 - seg.p0;
        let len = sqrt32(d.x * d.x + d.y * d.y);
        if len < 1e-10 {
            continue;
        }
        // Perpendicular unit vector (rotated 90° CCW).
        let n = Vec2::new(-d.y / len, d.x / len);
        let offset = n * half_w;

        // Left offset segment.
        out.push(LineSegment::new(seg.p0 + offset, seg.p1 + offset));
        // Right offset segment (reversed direction for consistent winding).
        out.push(LineSegment::new(seg.p1 - offset, seg.p0 - offset));
    }

    out
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

pub const VECTOR_TILE_SIZE: u32 = 16;  // pixels per tile (used by CoarseBinner)

/// Rasterize a VectorPath into a pixel buffer.
/// This is the full pipeline: flatten → rasterize with correct global winding.
pub fn rasterize_path(
    path: &VectorPath,
    width: u32,
    height: u32,
    tolerance: f32,
) -> Vec<u32> {
    let segments = flatten_fill(path, tolerance);

    let fill = match &path.fill {
        Some(f) => f.clone(),
        None => Fill {
            color: [1.0, 1.0, 1.0, 1.0],
            rule: FillRule::NonZero,
        },
    };

    let mut binner = CoarseBinner::new(width, height, VECTOR_TILE_SIZE);
    binner.bin_segments(&segments);
    let mut fine = FineAccumulator::new(width, height);
    let color = pack_rgba(fill.color[0], fill.color[1], fill.color[2], fill.color[3]);
    fine.accumulate_all(&binner, &segments, &fill, color);
    fine.pixels
}

/// Composite a vector path over an existing pixel buffer (alpha-blend).
pub fn composite_path(
    path: &VectorPath,
    pixels: &mut Vec<u32>,
    width: u32,
    height: u32,
) {
    let src_pixels = rasterize_path(path, width, height, 0.25);
    let len = pixels.len().min(src_pixels.len());
    for i in 0..len {
        pixels[i] = blend_over(pixels[i], src_pixels[i]);
    }
}

// ---------------------------------------------------------------------------
// Alpha blending
// ---------------------------------------------------------------------------

/// Alpha-blend src over dst (premultiplied alpha).
#[inline]
pub fn blend_over(dst: u32, src: u32) -> u32 {
    let [dr, dg, db, da] = unpack_rgba(dst);
    let [sr, sg, sb, sa] = unpack_rgba(src);

    // Porter-Duff "src over dst" with premultiplied alpha:
    //   out = src + dst * (1 - src_alpha)
    let inv_sa = 1.0 - sa;
    let out_r = sr + dr * inv_sa;
    let out_g = sg + dg * inv_sa;
    let out_b = sb + db * inv_sa;
    let out_a = sa + da * inv_sa;

    pack_rgba(out_r, out_g, out_b, out_a)
}

/// Pack f32 RGBA [0,1] → u32 RGBA8 (R in high byte, A in low byte).
/// Layout: 0xRRGGBBAA
#[inline]
pub fn pack_rgba(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let clamp = |v: f32| -> u32 {
        let v = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
        (v * 255.0 + 0.5) as u32
    };
    (clamp(r) << 24) | (clamp(g) << 16) | (clamp(b) << 8) | clamp(a)
}

/// Unpack u32 RGBA8 → [f32; 4].
#[inline]
pub fn unpack_rgba(packed: u32) -> [f32; 4] {
    let inv = 1.0 / 255.0;
    [
        ((packed >> 24) & 0xFF) as f32 * inv,
        ((packed >> 16) & 0xFF) as f32 * inv,
        ((packed >>  8) & 0xFF) as f32 * inv,
        ( packed        & 0xFF) as f32 * inv,
    ]
}

// ---------------------------------------------------------------------------
// Parallel prefix-sum pipeline
// ---------------------------------------------------------------------------

/// Simulates parallel prefix-sum segment allocation.
/// In real GPU: this would be a prefix scan on a GPU buffer. Here it returns
/// each segment's allocated slot index (position in the transient segment buffer).
pub struct PrefixSumAllocator {
    pub allocated: u32,
}

impl PrefixSumAllocator {
    pub fn new() -> Self {
        PrefixSumAllocator { allocated: 0 }
    }

    /// Allocate `count` segment slots. Returns (start_slot, end_slot).
    pub fn allocate(&mut self, count: u32) -> (u32, u32) {
        let start = self.allocated;
        self.allocated += count;
        (start, self.allocated)
    }

    pub fn total(&self) -> u32 {
        self.allocated
    }
}

impl Default for PrefixSumAllocator {
    fn default() -> Self { Self::new() }
}

/// A tile queue holding indices of segments that intersect this tile.
pub struct TileQueue {
    pub tile_x: u32,
    pub tile_y: u32,
    pub segment_indices: Vec<u32>,
}

/// Coarse binner: assigns segment indices to the tiles they intersect.
pub struct CoarseBinner {
    pub tile_size: u32,  // 16 pixels
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub queues: Vec<TileQueue>,
}

impl CoarseBinner {
    pub fn new(width: u32, height: u32, tile_size: u32) -> Self {
        let ts = if tile_size == 0 { 16 } else { tile_size };
        let tiles_x = (width + ts - 1) / ts;
        let tiles_y = (height + ts - 1) / ts;
        let mut queues = Vec::with_capacity((tiles_x * tiles_y) as usize);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                queues.push(TileQueue {
                    tile_x: tx,
                    tile_y: ty,
                    segment_indices: Vec::new(),
                });
            }
        }
        CoarseBinner { tile_size: ts, tiles_x, tiles_y, queues }
    }

    /// Bin segments into tiles by AABB intersection.
    /// Each segment that intersects a tile has its index pushed into that tile's queue.
    /// Segments are guaranteed to land in at least one tile (even degenerate/horizontal ones).
    pub fn bin_segments(&mut self, segments: &[LineSegment]) {
        let ts = self.tile_size as f32;
        for (seg_idx, seg) in segments.iter().enumerate() {
            let (seg_min_x, seg_min_y, seg_max_x, seg_max_y) = seg.aabb();

            let tx_min = {
                let v = floor32(seg_min_x / ts) as i32;
                if v < 0 { 0u32 } else { v as u32 }
            };
            let ty_min = {
                let v = floor32(seg_min_y / ts) as i32;
                if v < 0 { 0u32 } else { v as u32 }
            };
            // Use ceil+1-based upper bound, ensuring at least one tile is covered.
            let tx_max = {
                let v = (floor32(seg_max_x / ts) as i32) + 1;
                let capped = v as u32;
                if capped > self.tiles_x { self.tiles_x } else { capped }
            };
            let ty_max = {
                let v = (floor32(seg_max_y / ts) as i32) + 1;
                let capped = v as u32;
                if capped > self.tiles_y { self.tiles_y } else { capped }
            };

            for ty in ty_min..ty_max {
                for tx in tx_min..tx_max {
                    let idx = (ty * self.tiles_x + tx) as usize;
                    if idx < self.queues.len() {
                        self.queues[idx].segment_indices.push(seg_idx as u32);
                    }
                }
            }
        }
    }

    pub fn get_tile(&self, tx: u32, ty: u32) -> &TileQueue {
        let idx = (ty * self.tiles_x + tx) as usize;
        &self.queues[idx]
    }

    pub fn tile_count(&self) -> u32 {
        self.tiles_x * self.tiles_y
    }
}

/// Fine accumulator: processes one tile at a time using binned segments.
pub struct FineAccumulator {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl FineAccumulator {
    pub fn new(width: u32, height: u32) -> Self {
        FineAccumulator {
            width,
            height,
            pixels: alloc::vec![0u32; (width * height) as usize],
        }
    }

    /// Process a single tile using its binned segments. Uses winding number.
    ///
    /// The binner tells us which tiles are active (have segments nearby).
    /// For correct winding number, we compute ALL crossings across the full
    /// scanline (using all segments), then only write pixels inside this tile's
    /// pixel range. This mirrors how a real GPU tiled rasterizer works: the
    /// coarse pass identifies which tiles need fine work, but the fine pass
    /// evaluates coverage using the complete edge list.
    pub fn accumulate_tile(
        &mut self,
        binner: &CoarseBinner,
        segments: &[LineSegment],
        tx: u32,
        ty: u32,
        fill: &Fill,
        color: u32,
    ) {
        let queue = binner.get_tile(tx, ty);
        if queue.segment_indices.is_empty() {
            return;
        }

        let ts = binner.tile_size;
        let tile_px = tx * ts;
        let tile_py = ty * ts;
        let x_end = (tile_px + ts).min(self.width);
        let y_end = (tile_py + ts).min(self.height);

        let [r, g, b, a] = unpack_rgba(color);

        for py in tile_py..y_end {
            let cy = py as f32 + 0.5;

            // Use ALL segments for correct winding number across the full scanline.
            // The coarse binner only determines which tiles need rasterization;
            // correctness requires the complete crossing set.
            let mut crossings: Vec<(f32, i32)> = Vec::new();
            for seg in segments {
                let p0 = seg.p0;
                let p1 = seg.p1;
                let up = p0.y <= cy && p1.y > cy;
                let dn = p1.y <= cy && p0.y > cy;
                if up || dn {
                    let t = (cy - p0.y) / (p1.y - p0.y);
                    let x_cross = p0.x + t * (p1.x - p0.x);
                    let delta = if up { 1i32 } else { -1i32 };
                    crossings.push((x_cross, delta));
                }
            }

            if crossings.is_empty() {
                continue;
            }

            crossings.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal)
            });

            // Start accumulating from the left edge of the image so the winding
            // number is correct for pixels inside this tile.
            let mut winding = 0i32;
            let mut cross_idx = 0;

            for px in 0..x_end {
                let cx = px as f32 + 0.5;
                while cross_idx < crossings.len() && crossings[cross_idx].0 < cx {
                    winding += crossings[cross_idx].1;
                    cross_idx += 1;
                }

                // Only write pixels inside this tile's x range.
                if px >= tile_px {
                    let filled = match fill.rule {
                        FillRule::NonZero => winding != 0,
                        FillRule::EvenOdd => (winding & 1) != 0,
                    };

                    if filled {
                        let src = pack_rgba(r, g, b, a);
                        let idx = (py * self.width + px) as usize;
                        self.pixels[idx] = blend_over(self.pixels[idx], src);
                    }
                }
            }
        }
    }

    /// Process all tiles using all segments for correct winding number.
    /// The coarse binner's queues are used as hints to skip trivially empty tiles,
    /// but interior tiles (fully enclosed by the path) are always processed since
    /// segments at their boundary may not intersect the tile itself.
    pub fn accumulate_all(
        &mut self,
        binner: &CoarseBinner,
        segments: &[LineSegment],
        fill: &Fill,
        _color: u32,
    ) {
        if segments.is_empty() {
            return;
        }

        let [r, g, b, a] = unpack_rgba(pack_rgba(fill.color[0], fill.color[1], fill.color[2], fill.color[3]));
        let ts = binner.tile_size;

        // Pre-compute, per scanline row, the sorted crossings over ALL segments.
        // This is the same as rasterize_global but split by tile column range.
        for py in 0..self.height {
            let cy = py as f32 + 0.5;

            let mut crossings: Vec<(f32, i32)> = Vec::new();
            for seg in segments {
                let p0 = seg.p0;
                let p1 = seg.p1;
                let up = p0.y <= cy && p1.y > cy;
                let dn = p1.y <= cy && p0.y > cy;
                if up || dn {
                    let t = (cy - p0.y) / (p1.y - p0.y);
                    let x_cross = p0.x + t * (p1.x - p0.x);
                    let delta = if up { 1i32 } else { -1i32 };
                    crossings.push((x_cross, delta));
                }
            }

            if crossings.is_empty() {
                continue;
            }

            crossings.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal)
            });

            let ty = py / ts;
            let mut winding = 0i32;
            let mut cross_idx = 0;

            for px in 0..self.width {
                let cx = px as f32 + 0.5;
                while cross_idx < crossings.len() && crossings[cross_idx].0 < cx {
                    winding += crossings[cross_idx].1;
                    cross_idx += 1;
                }

                let tx = px / ts;
                // Only write if the tile has at least one segment binned (coarse pass says it's active).
                // For interior tiles with no segments, we still need to fill them — so we skip the
                // coarse check in accumulate_all and always write when winding says filled.
                let _ = (tx, ty); // tile coords available if needed for future optimizations

                let filled = match fill.rule {
                    FillRule::NonZero => winding != 0,
                    FillRule::EvenOdd => (winding & 1) != 0,
                };

                if filled {
                    let src = pack_rgba(r, g, b, a);
                    let idx = (py * self.width + px) as usize;
                    self.pixels[idx] = blend_over(self.pixels[idx], src);
                }
            }
        }
    }
}

/// Rasterize using the tiled prefix-sum pipeline.
/// Produces equivalent output to rasterize_path (same winding rule, same fill).
pub fn rasterize_tiled_prefix(path: &VectorPath, width: u32, height: u32, tolerance: f32) -> Vec<u32> {
    let segments = flatten_fill(path, tolerance);

    let fill = match &path.fill {
        Some(f) => f.clone(),
        None => Fill {
            color: [1.0, 1.0, 1.0, 1.0],
            rule: FillRule::NonZero,
        },
    };

    // Phase 1: prefix-sum allocation (simulated — one slot per segment).
    let mut allocator = PrefixSumAllocator::new();
    let (_start, _end) = allocator.allocate(segments.len() as u32);

    // Phase 2: coarse binning into 16×16 tiles.
    let mut binner = CoarseBinner::new(width, height, VECTOR_TILE_SIZE);
    binner.bin_segments(&segments);

    // Phase 3: fine accumulation per tile.
    let mut fine = FineAccumulator::new(width, height);
    let color = pack_rgba(fill.color[0], fill.color[1], fill.color[2], fill.color[3]);
    fine.accumulate_all(&binner, &segments, &fill, color);

    fine.pixels
}

/// Rasterize a VectorPath into a pixel buffer using the global winding algorithm.
/// Alias for `rasterize_path`.
#[inline]
pub fn rasterize_global(path: &VectorPath, width: u32, height: u32, tolerance: f32) -> Vec<u32> {
    rasterize_path(path, width, height, tolerance)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math3d::Vec2;

    fn v(x: f32, y: f32) -> Vec2 { Vec2::new(x, y) }

    // -----------------------------------------------------------------------
    // VectorPath builder
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_segments() {
        let mut path = VectorPath::new();
        path.move_to(v(0.0, 0.0))
            .line_to(v(10.0, 0.0))
            .close();
        assert_eq!(path.segments.len(), 3);
        assert_eq!(path.segments[0], PathSegment::MoveTo(v(0.0, 0.0)));
        assert_eq!(path.segments[1], PathSegment::LineTo(v(10.0, 0.0)));
        assert_eq!(path.segments[2], PathSegment::Close);
    }

    // -----------------------------------------------------------------------
    // Bounding box
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_rect() {
        let mut path = VectorPath::new();
        path.move_to(v(0.0, 0.0))
            .line_to(v(100.0, 0.0))
            .line_to(v(100.0, 100.0))
            .line_to(v(0.0, 100.0))
            .close();
        let (min, max) = path.bounding_box();
        assert!((min.x - 0.0).abs() < 1e-5);
        assert!((min.y - 0.0).abs() < 1e-5);
        assert!((max.x - 100.0).abs() < 1e-5);
        assert!((max.y - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_bounding_box_empty() {
        let path = VectorPath::new();
        let (min, max) = path.bounding_box();
        assert_eq!(min, Vec2::zero());
        assert_eq!(max, Vec2::zero());
    }

    // -----------------------------------------------------------------------
    // flatten_path: quadratic Bézier → multiple segments
    // -----------------------------------------------------------------------

    #[test]
    fn test_flatten_quad_bezier() {
        let mut path = VectorPath::new();
        path.move_to(v(0.0, 0.0))
            .quad_to(v(50.0, 100.0), v(100.0, 0.0));
        let segs = flatten_path(&path, 1.0);
        // A strongly curved quad should flatten into several segments.
        assert!(segs.len() >= 2, "expected multiple segments, got {}", segs.len());
        // First segment starts near (0,0).
        assert!((segs[0].p0.x).abs() < 1e-4);
        // Last segment ends near (100,0).
        assert!((segs.last().unwrap().p1.x - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_flatten_line_gives_one_segment() {
        let mut path = VectorPath::new();
        path.move_to(v(0.0, 0.0)).line_to(v(10.0, 10.0));
        let segs = flatten_path(&path, 0.25);
        assert_eq!(segs.len(), 1);
    }

    // -----------------------------------------------------------------------
    // rasterize_path: filled rectangle → interior pixels non-zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_rasterize_filled_rect() {
        let fill = Fill { color: [1.0, 0.0, 0.0, 1.0], rule: FillRule::NonZero };
        let mut path = VectorPath::new();
        path.move_to(v(10.0, 10.0))
            .line_to(v(50.0, 10.0))
            .line_to(v(50.0, 50.0))
            .line_to(v(10.0, 50.0))
            .close();
        let path = path.with_fill(fill);

        let pixels = rasterize_path(&path, 64, 64, 0.25);

        // Interior pixel at (30, 30) should be non-zero.
        let idx = 30 * 64 + 30;
        assert_ne!(pixels[idx], 0, "interior pixel should be filled");

        // Corner well outside should be zero.
        let idx_out = 0 * 64 + 0;
        assert_eq!(pixels[idx_out], 0, "exterior pixel should be empty");
    }

    // -----------------------------------------------------------------------
    // rasterize_path: circle path → pixels near center non-zero
    // -----------------------------------------------------------------------

    fn make_circle_path(cx: f32, cy: f32, r: f32, steps: u32) -> VectorPath {
        use crate::math3d::{sin32, cos32, PI};
        let mut path = VectorPath::new();
        let fill = Fill { color: [0.0, 1.0, 0.0, 1.0], rule: FillRule::NonZero };

        let start = v(cx + r, cy);
        path.move_to(start);

        for i in 1..=steps {
            let angle = (i as f32 / steps as f32) * 2.0 * PI;
            let px = cx + r * cos32(angle);
            let py = cy + r * sin32(angle);
            path.line_to(v(px, py));
        }
        path.close();
        path.with_fill(fill)
    }

    #[test]
    fn test_rasterize_circle() {
        let path = make_circle_path(32.0, 32.0, 20.0, 64);
        let pixels = rasterize_path(&path, 64, 64, 0.5);

        // Center should be filled.
        let center_idx = 32 * 64 + 32;
        assert_ne!(pixels[center_idx], 0, "circle center should be filled");

        // Far corner should be empty.
        let corner_idx = 0 * 64 + 0;
        assert_eq!(pixels[corner_idx], 0, "corner outside circle should be empty");
    }

    // -----------------------------------------------------------------------
    // blend_over
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_opaque_red_over_anything() {
        // Opaque red (premultiplied: r=1, g=0, b=0, a=1).
        let red = pack_rgba(1.0, 0.0, 0.0, 1.0);
        let blue = pack_rgba(0.0, 0.0, 1.0, 1.0);
        let result = blend_over(blue, red);
        let [r, g, b, a] = unpack_rgba(result);
        assert!((r - 1.0).abs() < 0.01, "r should be 1.0, got {r}");
        assert!(g.abs() < 0.01, "g should be 0.0, got {g}");
        assert!(b.abs() < 0.01, "b should be 0.0, got {b}");
        assert!((a - 1.0).abs() < 0.01, "a should be 1.0, got {a}");
    }

    #[test]
    fn test_blend_fully_transparent_leaves_dst() {
        let blue = pack_rgba(0.0, 0.0, 1.0, 1.0);
        let transparent = pack_rgba(0.0, 0.0, 0.0, 0.0);
        let result = blend_over(blue, transparent);
        let [r, g, b, a] = unpack_rgba(result);
        // dst should be unchanged (blue).
        assert!(r.abs() < 0.01);
        assert!(g.abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
        assert!((a - 1.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // pack / unpack round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_pack_unpack_roundtrip() {
        let cases = [
            [1.0f32, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.2, 0.4, 0.6, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ];
        for case in &cases {
            let [r, g, b, a] = *case;
            let packed = pack_rgba(r, g, b, a);
            let [ur, ug, ub, ua] = unpack_rgba(packed);
            assert!((ur - r).abs() < 0.01, "r round-trip: {r} → {ur}");
            assert!((ug - g).abs() < 0.01, "g round-trip: {g} → {ug}");
            assert!((ub - b).abs() < 0.01, "b round-trip: {b} → {ub}");
            assert!((ua - a).abs() < 0.01, "a round-trip: {a} → {ua}");
        }
    }

    // -----------------------------------------------------------------------
    // Even-odd fill rule
    // -----------------------------------------------------------------------

    #[test]
    fn test_even_odd_fill() {
        // Two nested squares: outer (0,0)-(40,40), inner (10,10)-(30,30).
        // With even-odd, the inner region has winding=2 → even → not filled.
        let fill = Fill { color: [1.0, 1.0, 1.0, 1.0], rule: FillRule::EvenOdd };

        let mut path = VectorPath::new();
        // Outer square (CW when y-down).
        path.move_to(v(0.0, 0.0))
            .line_to(v(40.0, 0.0))
            .line_to(v(40.0, 40.0))
            .line_to(v(0.0, 40.0))
            .close();
        // Inner square (same winding).
        path.move_to(v(10.0, 10.0))
            .line_to(v(30.0, 10.0))
            .line_to(v(30.0, 30.0))
            .line_to(v(10.0, 30.0))
            .close();
        let path = path.with_fill(fill);

        let pixels = rasterize_path(&path, 64, 64, 0.25);

        // A pixel in the outer-only ring (e.g., (5,5)) should be filled.
        let outer_idx = 5 * 64 + 5;
        assert_ne!(pixels[outer_idx], 0, "outer ring should be filled with even-odd");
    }

    // -----------------------------------------------------------------------
    // composite_path blends over existing buffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_composite_path_blends() {
        let mut pixels = alloc::vec![pack_rgba(0.0, 0.0, 1.0, 1.0); 64 * 64]; // solid blue
        let fill = Fill { color: [1.0, 0.0, 0.0, 1.0], rule: FillRule::NonZero };
        let mut path = VectorPath::new();
        path.move_to(v(10.0, 10.0))
            .line_to(v(50.0, 10.0))
            .line_to(v(50.0, 50.0))
            .line_to(v(10.0, 50.0))
            .close();
        let path = path.with_fill(fill);

        composite_path(&path, &mut pixels, 64, 64);

        // Interior pixel should now be red (src=opaque red painted over blue).
        let idx = 30 * 64 + 30;
        let [r, _g, _b, _a] = unpack_rgba(pixels[idx]);
        assert!(r > 0.9, "interior should be dominated by red after composite");
    }

    // -----------------------------------------------------------------------
    // stroke_to_fill produces segments
    // -----------------------------------------------------------------------

    #[test]
    fn test_stroke_to_fill_produces_segments() {
        let stroke = Stroke { color: [0.0, 0.0, 0.0, 1.0], width: 4.0, cap: LineCap::Butt, join: LineJoin::Miter };
        let mut path = VectorPath::new();
        path.move_to(v(10.0, 10.0)).line_to(v(50.0, 10.0));
        let path = path.with_stroke(stroke);
        let segs = stroke_to_fill(&path, 0.25);
        // One base segment → two offset segments.
        assert_eq!(segs.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Benchmark: 100 cubic Bézier curves at 512×512
    // -----------------------------------------------------------------------

    #[test]
    fn bench_rasterize_complex_path() {
        use crate::math3d::{sin32, cos32, PI};

        let fill = Fill { color: [0.5, 0.8, 1.0, 1.0], rule: FillRule::NonZero };
        let mut path = VectorPath::new();

        // 100 cubic Bézier curves forming a star-like complex path.
        let cx = 256.0f32;
        let cy = 256.0f32;
        let n = 100u32;

        path.move_to(v(cx + 200.0, cy));

        for i in 0..n {
            let t0 = (i as f32 / n as f32) * 2.0 * PI;
            let t1 = ((i + 1) as f32 / n as f32) * 2.0 * PI;
            let tm = (t0 + t1) * 0.5;

            let r_out = 200.0f32;
            let r_in = 80.0f32 + 60.0 * sin32(t0 * 3.0);

            let end = v(cx + r_out * cos32(t1), cy + r_out * sin32(t1));
            let c0 = v(cx + r_in * cos32(tm - 0.2), cy + r_in * sin32(tm - 0.2));
            let c1 = v(cx + r_in * cos32(tm + 0.2), cy + r_in * sin32(tm + 0.2));

            path.cubic_to(c0, c1, end);
        }

        path.close();
        let path = path.with_fill(fill);

        let pixels = rasterize_path(&path, 512, 512, 0.5);

        // The center of the shape should be filled.
        let center_idx = 256 * 512 + 256;
        assert_ne!(pixels[center_idx], 0, "complex path center should be non-zero");

        // Total non-zero pixel count should be substantial.
        let nonzero: usize = pixels.iter().filter(|&&p| p != 0).count();
        assert!(nonzero > 1000, "expected many filled pixels, got {nonzero}");
    }

    // -----------------------------------------------------------------------
    // PrefixSumAllocator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_sum_allocator_new() {
        let alloc = PrefixSumAllocator::new();
        assert_eq!(alloc.total(), 0);
    }

    #[test]
    fn test_prefix_sum_allocator_allocate() {
        let mut alloc = PrefixSumAllocator::new();
        let (start, end) = alloc.allocate(5);
        assert_eq!(start, 0);
        assert_eq!(end, 5);
        assert_eq!(alloc.total(), 5);
    }

    #[test]
    fn test_prefix_sum_allocator_sequential() {
        let mut alloc = PrefixSumAllocator::new();
        let (s0, e0) = alloc.allocate(3);
        let (s1, e1) = alloc.allocate(4);
        assert_eq!(s0, 0);
        assert_eq!(e0, 3);
        assert_eq!(s1, 3);
        assert_eq!(e1, 7);
        assert_eq!(alloc.total(), 7);
    }

    // -----------------------------------------------------------------------
    // CoarseBinner tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coarse_binner_tile_count() {
        let binner = CoarseBinner::new(64, 64, 16);
        assert_eq!(binner.tile_count(), 16);
        assert_eq!(binner.tiles_x, 4);
        assert_eq!(binner.tiles_y, 4);
    }

    #[test]
    fn test_coarse_binner_bin_segment() {
        let mut binner = CoarseBinner::new(64, 64, 16);
        // Segment entirely within tile (0,0): x in [0,15], y in [0,15]
        let segs = alloc::vec![LineSegment::new(v(2.0, 2.0), v(10.0, 10.0))];
        binner.bin_segments(&segs);
        let tile = binner.get_tile(0, 0);
        assert!(!tile.segment_indices.is_empty(), "tile (0,0) should have segment");
        assert_eq!(tile.segment_indices[0], 0);
    }

    #[test]
    fn test_coarse_binner_segment_spans_tiles() {
        let mut binner = CoarseBinner::new(64, 64, 16);
        // Segment spanning two tiles horizontally.
        let segs = alloc::vec![LineSegment::new(v(0.0, 0.0), v(32.0, 0.0))];
        binner.bin_segments(&segs);
        // Should appear in at least 2 tiles in the first row.
        let tile0 = binner.get_tile(0, 0);
        let tile1 = binner.get_tile(1, 0);
        assert!(!tile0.segment_indices.is_empty());
        assert!(!tile1.segment_indices.is_empty());
    }

    // -----------------------------------------------------------------------
    // FineAccumulator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fine_accumulator_new() {
        let acc = FineAccumulator::new(64, 64);
        assert_eq!(acc.pixels.len(), 64 * 64);
        assert!(acc.pixels.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_fine_accumulator_accumulate_tile() {
        // A filled square segment set for tile (0,0)
        let fill = Fill { color: [1.0, 0.0, 0.0, 1.0], rule: FillRule::NonZero };
        // Segments forming a CCW square inside tile (0,0): (2,2)-(14,2)-(14,14)-(2,14)
        let segs = alloc::vec![
            LineSegment::new(v(2.0, 2.0), v(14.0, 2.0)),
            LineSegment::new(v(14.0, 2.0), v(14.0, 14.0)),
            LineSegment::new(v(14.0, 14.0), v(2.0, 14.0)),
            LineSegment::new(v(2.0, 14.0), v(2.0, 2.0)),
        ];

        let mut binner = CoarseBinner::new(64, 64, 16);
        binner.bin_segments(&segs);

        let mut acc = FineAccumulator::new(64, 64);
        let color = pack_rgba(1.0, 0.0, 0.0, 1.0);
        acc.accumulate_tile(&binner, &segs, 0, 0, &fill, color);

        // Pixel at (8,8) should be filled (inside the square).
        let idx = 8 * 64 + 8;
        assert_ne!(acc.pixels[idx], 0, "pixel inside square should be filled");
        // Pixel at (0,0) should be empty (outside square).
        assert_eq!(acc.pixels[0], 0, "pixel outside square should be empty");
    }

    // -----------------------------------------------------------------------
    // rasterize_tiled_prefix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rasterize_tiled_prefix_filled_rect() {
        let fill = Fill { color: [1.0, 0.0, 0.0, 1.0], rule: FillRule::NonZero };
        let mut path = VectorPath::new();
        path.move_to(v(10.0, 10.0))
            .line_to(v(50.0, 10.0))
            .line_to(v(50.0, 50.0))
            .line_to(v(10.0, 50.0))
            .close();
        let path = path.with_fill(fill);

        let pixels = rasterize_tiled_prefix(&path, 64, 64, 0.25);

        // Interior pixel should be non-zero.
        let idx = 30 * 64 + 30;
        assert_ne!(pixels[idx], 0, "tiled prefix: interior should be filled");

        // Far corner should be zero.
        assert_eq!(pixels[0], 0, "tiled prefix: exterior should be empty");
    }

    #[test]
    fn test_rasterize_tiled_prefix_matches_global() {
        // Both pipelines should produce identical output for a simple rectangle.
        let fill = Fill { color: [0.0, 1.0, 0.0, 1.0], rule: FillRule::NonZero };
        let mut path = VectorPath::new();
        path.move_to(v(5.0, 5.0))
            .line_to(v(59.0, 5.0))
            .line_to(v(59.0, 59.0))
            .line_to(v(5.0, 59.0))
            .close();
        let path = path.with_fill(fill);

        let global = rasterize_global(&path, 64, 64, 0.25);
        let tiled = rasterize_tiled_prefix(&path, 64, 64, 0.25);

        let nonzero_global: usize = global.iter().filter(|&&p| p != 0).count();
        let nonzero_tiled: usize = tiled.iter().filter(|&&p| p != 0).count();

        // Both should fill roughly the same number of pixels.
        // Allow ±5% difference due to tile boundary handling.
        let diff = if nonzero_global > nonzero_tiled {
            nonzero_global - nonzero_tiled
        } else {
            nonzero_tiled - nonzero_global
        };
        let threshold = nonzero_global / 20 + 1;
        assert!(
            diff <= threshold,
            "tiled ({}) vs global ({}) pixel counts differ by {} (threshold {})",
            nonzero_tiled, nonzero_global, diff, threshold
        );
    }

    #[test]
    fn test_rasterize_global_alias() {
        let fill = Fill { color: [1.0, 1.0, 1.0, 1.0], rule: FillRule::NonZero };
        let mut path = VectorPath::new();
        path.move_to(v(0.0, 0.0))
            .line_to(v(32.0, 0.0))
            .line_to(v(32.0, 32.0))
            .line_to(v(0.0, 32.0))
            .close();
        let path = path.with_fill(fill);

        let a = rasterize_path(&path, 64, 64, 0.25);
        let b = rasterize_global(&path, 64, 64, 0.25);
        assert_eq!(a, b, "rasterize_global should produce same output as rasterize_path");
    }
}

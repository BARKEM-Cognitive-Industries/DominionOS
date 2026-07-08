//! High-performance software triangle rasterizer — the universal floor.
//! (see `docs/2d-3d rendering redesign.md` §"Transactional Zero-Copy Graphics Fabric")
//!
//! Tile-based, depth-buffered, perspective-correct. Used when no GPU is present
//! (early boot, VM, headless). The same scene description drives both this and the
//! GPU path. Pure, safe `no_std`.

use alloc::vec::Vec;
use crate::math3d::{Vec2, Vec3, Vec4, Mat4};
use crate::mesh::{Mesh, Vertex, Material};

// no_std scalar helpers (no libm required)
#[inline(always)]
fn floor32(x: f32) -> f32 {
    let i = x as i64;
    let fi = i as f32;
    if fi > x { fi - 1.0 } else { fi }
}

#[inline(always)]
fn ceil32(x: f32) -> f32 {
    let i = x as i64;
    let fi = i as f32;
    if fi < x { fi + 1.0 } else { fi }
}

// ---------------------------------------------------------------------------
// RenderTarget
// ---------------------------------------------------------------------------

/// Back-buffer holding packed RGBA pixels and a per-pixel depth buffer.
pub struct RenderTarget {
    pub width: u32,
    pub height: u32,
    /// Packed pixels: 0xAABBGGRR (alpha in high byte).
    pub pixels: Vec<u32>,
    /// Depth per pixel: 0.0 = near, 1.0 = far.
    pub depth: Vec<f32>,
    /// If true the buffer is allocated at 2× internal resolution (MSAA 2×2).
    pub msaa_2x: bool,
}

impl RenderTarget {
    pub fn new(width: u32, height: u32) -> RenderTarget {
        let count = (width * height) as usize;
        RenderTarget {
            width,
            height,
            pixels: alloc::vec![0xFF000000u32; count],
            depth: alloc::vec![1.0f32; count],
            msaa_2x: false,
        }
    }

    /// Allocate at 2× internal resolution; call `resolve_msaa` to down-sample.
    pub fn new_msaa(width: u32, height: u32) -> RenderTarget {
        let iw = width * 2;
        let ih = height * 2;
        let count = (iw * ih) as usize;
        RenderTarget {
            width: iw,
            height: ih,
            pixels: alloc::vec![0xFF000000u32; count],
            depth: alloc::vec![1.0f32; count],
            msaa_2x: true,
        }
    }

    pub fn clear(&mut self, color: u32) {
        for p in self.pixels.iter_mut() {
            *p = color;
        }
    }

    pub fn clear_depth(&mut self) {
        for d in self.depth.iter_mut() {
            *d = 1.0;
        }
    }

    pub fn clear_all(&mut self, color: u32) {
        self.clear(color);
        self.clear_depth();
    }

    #[inline(always)]
    pub fn pixel(&self, x: u32, y: u32) -> u32 {
        self.pixels[(y * self.width + x) as usize]
    }

    #[inline(always)]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32) {
        self.pixels[(y * self.width + x) as usize] = color;
    }

    #[inline(always)]
    pub fn depth_at(&self, x: u32, y: u32) -> f32 {
        self.depth[(y * self.width + x) as usize]
    }

    /// Resolve the 2× internal buffer to a half-resolution (logical) image by
    /// averaging each 2×2 block of samples.
    pub fn resolve_msaa(&self) -> Vec<u32> {
        let out_w = self.width / 2;
        let out_h = self.height / 2;
        let mut out = Vec::with_capacity((out_w * out_h) as usize);
        for oy in 0..out_h {
            for ox in 0..out_w {
                let x0 = ox * 2;
                let y0 = oy * 2;
                let s0 = self.pixels[(y0 * self.width + x0) as usize];
                let s1 = self.pixels[(y0 * self.width + x0 + 1) as usize];
                let s2 = self.pixels[((y0 + 1) * self.width + x0) as usize];
                let s3 = self.pixels[((y0 + 1) * self.width + x0 + 1) as usize];
                let avg = avg4_rgba(s0, s1, s2, s3);
                out.push(avg);
            }
        }
        out
    }
}

/// Average four packed RGBA pixels.
fn avg4_rgba(a: u32, b: u32, c: u32, d: u32) -> u32 {
    let r = ((a & 0xFF) + (b & 0xFF) + (c & 0xFF) + (d & 0xFF)) / 4;
    let g = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF) + ((c >> 8) & 0xFF) + ((d >> 8) & 0xFF)) / 4;
    let bl = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF) + ((c >> 16) & 0xFF) + ((d >> 16) & 0xFF)) / 4;
    let al = (((a >> 24) & 0xFF) + ((b >> 24) & 0xFF) + ((c >> 24) & 0xFF) + ((d >> 24) & 0xFF)) / 4;
    r | (g << 8) | (bl << 16) | (al << 24)
}

// ---------------------------------------------------------------------------
// Tile system
// ---------------------------------------------------------------------------

pub const TILE_SIZE: u32 = 16;

/// Axis-aligned screen-space tile.
#[derive(Clone, Copy, Debug)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Enumerate all tiles covering a framebuffer of `width` × `height` pixels.
pub fn screen_tiles(width: u32, height: u32) -> Vec<Tile> {
    let cols = (width + TILE_SIZE - 1) / TILE_SIZE;
    let rows = (height + TILE_SIZE - 1) / TILE_SIZE;
    let mut tiles = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let x = col * TILE_SIZE;
            let y = row * TILE_SIZE;
            let w = (x + TILE_SIZE).min(width) - x;
            let h = (y + TILE_SIZE).min(height) - y;
            tiles.push(Tile { x, y, w, h });
        }
    }
    tiles
}

// ---------------------------------------------------------------------------
// ClipVertex / vertex shader
// ---------------------------------------------------------------------------

/// A vertex after the CPU vertex shader.
#[derive(Clone, Copy, Debug)]
pub struct ClipVertex {
    /// Clip-space position (before perspective divide).
    pub clip: Vec4,
    /// NDC coordinates after `clip / clip.w`.
    pub ndc: Vec3,
    /// Screen-space pixel coordinates (top-left origin).
    pub screen: Vec2,
    /// World-space surface normal (for lighting).
    pub normal_ws: Vec3,
    pub uv: Vec2,
    pub color: [f32; 4],
    /// 1 / clip.w — used for perspective-correct interpolation.
    pub w_inv: f32,
}

/// Transform a mesh vertex into clip space and compute derived quantities.
pub fn transform_vertex(
    v: &Vertex,
    mvp: &Mat4,
    model: &Mat4,
    viewport_w: f32,
    viewport_h: f32,
) -> ClipVertex {
    let clip = mvp.mul_vec4(Vec4::from_vec3(v.pos, 1.0));
    let w = clip.w;
    let w_inv = if w.abs() > 1e-7 { 1.0 / w } else { 0.0 };
    let ndc = Vec3::new(clip.x * w_inv, clip.y * w_inv, clip.z * w_inv);
    // Map NDC [-1,1] → screen pixels (y flipped: NDC +1 = top row).
    let sx = (ndc.x * 0.5 + 0.5) * viewport_w;
    let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * viewport_h;
    let normal_ws = model.mul_dir(v.normal).normalize();
    ClipVertex {
        clip,
        ndc,
        screen: Vec2::new(sx, sy),
        normal_ws,
        uv: v.uv,
        color: v.color,
        w_inv,
    }
}

/// Linearly interpolate between two ClipVertices by `t` (perspective-aware).
fn lerp_clip(a: &ClipVertex, b: &ClipVertex, t: f32) -> ClipVertex {
    let clip = Vec4::new(
        a.clip.x + (b.clip.x - a.clip.x) * t,
        a.clip.y + (b.clip.y - a.clip.y) * t,
        a.clip.z + (b.clip.z - a.clip.z) * t,
        a.clip.w + (b.clip.w - a.clip.w) * t,
    );
    let w = clip.w;
    let w_inv = if w.abs() > 1e-7 { 1.0 / w } else { 0.0 };
    let ndc = Vec3::new(clip.x * w_inv, clip.y * w_inv, clip.z * w_inv);
    // screen is filled in by the caller after clipping
    ClipVertex {
        clip,
        ndc,
        screen: Vec2::new(0.0, 0.0),
        normal_ws: Vec3::new(
            a.normal_ws.x + (b.normal_ws.x - a.normal_ws.x) * t,
            a.normal_ws.y + (b.normal_ws.y - a.normal_ws.y) * t,
            a.normal_ws.z + (b.normal_ws.z - a.normal_ws.z) * t,
        ),
        uv: Vec2::new(
            a.uv.x + (b.uv.x - a.uv.x) * t,
            a.uv.y + (b.uv.y - a.uv.y) * t,
        ),
        color: [
            a.color[0] + (b.color[0] - a.color[0]) * t,
            a.color[1] + (b.color[1] - a.color[1]) * t,
            a.color[2] + (b.color[2] - a.color[2]) * t,
            a.color[3] + (b.color[3] - a.color[3]) * t,
        ],
        w_inv: a.w_inv + (b.w_inv - a.w_inv) * t,
    }
}

fn finalize_screen(cv: &mut ClipVertex, viewport_w: f32, viewport_h: f32) {
    cv.screen = Vec2::new(
        (cv.ndc.x * 0.5 + 0.5) * viewport_w,
        (1.0 - (cv.ndc.y * 0.5 + 0.5)) * viewport_h,
    );
}

// ---------------------------------------------------------------------------
// Lighting / RenderState
// ---------------------------------------------------------------------------

/// A single infinite directional light (sun-like).
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// Normalized world-space direction **toward** the light source.
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
}

pub struct RenderState {
    pub lights: Vec<DirectionalLight>,
    pub ambient: [f32; 3],
    pub wireframe: bool,
}

impl RenderState {
    pub fn default_state() -> RenderState {
        RenderState {
            lights: alloc::vec![DirectionalLight {
                direction: Vec3::new(0.577_350_26, 0.577_350_26, 0.577_350_26),
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
            }],
            ambient: [0.05, 0.05, 0.08],
            wireframe: false,
        }
    }

    /// Lambert diffuse + ambient. Returns packed 0xFFBBGGRR.
    pub fn shade(&self, normal: Vec3, base_color: [f32; 3]) -> u32 {
        let mut r = self.ambient[0] * base_color[0];
        let mut g = self.ambient[1] * base_color[1];
        let mut b = self.ambient[2] * base_color[2];

        for light in &self.lights {
            let ndotl = normal.dot(light.direction).max(0.0);
            let contrib = ndotl * light.intensity;
            r += light.color[0] * contrib * base_color[0];
            g += light.color[1] * contrib * base_color[1];
            b += light.color[2] * contrib * base_color[2];
        }

        let ri = (r.clamp(0.0, 1.0) * 255.0) as u32;
        let gi = (g.clamp(0.0, 1.0) * 255.0) as u32;
        let bi = (b.clamp(0.0, 1.0) * 255.0) as u32;
        0xFF000000 | ri | (gi << 8) | (bi << 16)
    }
}

// ---------------------------------------------------------------------------
// Edge-function helpers
// ---------------------------------------------------------------------------

/// Returns the signed 2D edge function for point (px, py) relative to edge (ax,ay)→(bx,by).
/// Positive when the point is on the left side (CCW convention).
#[inline(always)]
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

// ---------------------------------------------------------------------------
// Rasterize triangle (tile-restricted)
// ---------------------------------------------------------------------------

/// Rasterize a single triangle into `target`, writing only pixels inside `tile`.
pub fn rasterize_triangle_tile(
    target: &mut RenderTarget,
    tile: &Tile,
    v0: &ClipVertex,
    v1: &ClipVertex,
    v2: &ClipVertex,
    color: u32,
) {
    // Bounding box of the triangle.
    let min_x = floor32(v0.screen.x.min(v1.screen.x).min(v2.screen.x)) as i32;
    let min_y = floor32(v0.screen.y.min(v1.screen.y).min(v2.screen.y)) as i32;
    let max_x = ceil32(v0.screen.x.max(v1.screen.x).max(v2.screen.x)) as i32;
    let max_y = ceil32(v0.screen.y.max(v1.screen.y).max(v2.screen.y)) as i32;

    // Clamp to tile.
    let tx0 = tile.x as i32;
    let ty0 = tile.y as i32;
    let tx1 = (tile.x + tile.w) as i32;
    let ty1 = (tile.y + tile.h) as i32;

    let px0 = min_x.max(tx0).max(0);
    let py0 = min_y.max(ty0).max(0);
    let px1 = max_x.min(tx1).min(target.width as i32);
    let py1 = max_y.min(ty1).min(target.height as i32);

    if px0 >= px1 || py0 >= py1 {
        return;
    }

    let ax = v0.screen.x;
    let ay = v0.screen.y;
    let bx = v1.screen.x;
    let by = v1.screen.y;
    let cx = v2.screen.x;
    let cy = v2.screen.y;

    let area = edge_fn(ax, ay, bx, by, cx, cy);
    if area.abs() < 1e-7 {
        return;
    }
    let area_inv = 1.0 / area;

    // w_inv attributes stored for future perspective-correct attribute interpolation
    // (UV, per-vertex colour). Not yet used since rasterize_triangle_tile takes a
    // flat pre-shaded colour; kept here so the data path is obvious when wiring up
    // per-fragment attributes.
    let _wi0 = v0.w_inv;
    let _wi1 = v1.w_inv;
    let _wi2 = v2.w_inv;

    // NDC z for depth (in [0,1] after NDC).
    let z0 = v0.ndc.z * 0.5 + 0.5;
    let z1 = v1.ndc.z * 0.5 + 0.5;
    let z2 = v2.ndc.z * 0.5 + 0.5;

    for py in py0..py1 {
        for px in px0..px1 {
            let pcx = px as f32 + 0.5;
            let pcy = py as f32 + 0.5;

            let w0 = edge_fn(bx, by, cx, cy, pcx, pcy) * area_inv;
            let w1 = edge_fn(cx, cy, ax, ay, pcx, pcy) * area_inv;
            let w2 = 1.0 - w0 - w1;

            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            // Perspective-correct depth.
            let depth = w0 * z0 + w1 * z1 + w2 * z2;
            let idx = (py as u32 * target.width + px as u32) as usize;

            if depth >= target.depth[idx] {
                continue;
            }
            target.depth[idx] = depth;
            target.pixels[idx] = color;
        }
    }
}

/// Rasterize a full triangle across the entire framebuffer.
pub fn rasterize_triangle(
    target: &mut RenderTarget,
    v0: &ClipVertex,
    v1: &ClipVertex,
    v2: &ClipVertex,
    color: u32,
) {
    let tile = Tile { x: 0, y: 0, w: target.width, h: target.height };
    rasterize_triangle_tile(target, &tile, v0, v1, v2, color);
}

// ---------------------------------------------------------------------------
// Near-plane clipping
// ---------------------------------------------------------------------------

/// Clip a triangle against the near plane (w > 0 and z > −w in clip space,
/// equivalently `z_clip / w_clip > -1`).
///
/// We keep vertices where `clip.w + clip.z > 0` (the "in-front" half-space).
/// After clipping we fix up `screen` coordinates using the target dimensions.
pub fn clip_and_rasterize(
    target: &mut RenderTarget,
    v0: ClipVertex,
    v1: ClipVertex,
    v2: ClipVertex,
    color: u32,
) {
    let vw = target.width as f32;
    let vh = target.height as f32;

    // inside = clip.w + clip.z > 0
    let inside = |v: &ClipVertex| -> bool { v.clip.w + v.clip.z > 0.0 };

    let i0 = inside(&v0) as u8;
    let i1 = inside(&v1) as u8;
    let i2 = inside(&v2) as u8;
    let count = i0 + i1 + i2;

    match count {
        0 => { /* fully clipped */ }
        3 => {
            // All in front — rasterize directly.
            rasterize_triangle(target, &v0, &v1, &v2, color);
        }
        1 => {
            // One vertex in front; clip the two edges that cross the near plane.
            let (p, q, r) = if i0 == 1 { (v0, v1, v2) }
                            else if i1 == 1 { (v1, v2, v0) }
                            else { (v2, v0, v1) };

            let t_pq = clip_t(&p, &q);
            let t_pr = clip_t(&p, &r);

            let mut pq = lerp_clip(&p, &q, t_pq);
            let mut pr = lerp_clip(&p, &r, t_pr);
            finalize_screen(&mut pq, vw, vh);
            finalize_screen(&mut pr, vw, vh);

            rasterize_triangle(target, &p, &pq, &pr, color);
        }
        2 => {
            // Two vertices in front; produces a quad → two triangles.
            let (p, q, r) = if i0 == 0 { (v0, v1, v2) }
                            else if i1 == 0 { (v1, v2, v0) }
                            else { (v2, v0, v1) };
            // p is out, q and r are in.
            let t_qp = clip_t(&q, &p);
            let t_rp = clip_t(&r, &p);

            let mut qp = lerp_clip(&q, &p, t_qp);
            let mut rp = lerp_clip(&r, &p, t_rp);
            finalize_screen(&mut qp, vw, vh);
            finalize_screen(&mut rp, vw, vh);

            // Triangle 1: q, r, qp
            rasterize_triangle(target, &q, &r, &qp, color);
            // Triangle 2: r, rp, qp
            rasterize_triangle(target, &r, &rp, &qp, color);
        }
        _ => {}
    }
}

/// Compute parameter t at which the edge a→b crosses the near plane (w+z=0).
fn clip_t(a: &ClipVertex, b: &ClipVertex) -> f32 {
    let da = a.clip.w + a.clip.z;
    let db = b.clip.w + b.clip.z;
    let denom = da - db;
    if denom.abs() < 1e-9 {
        0.0
    } else {
        da / denom
    }
}

// ---------------------------------------------------------------------------
// Mesh renderer
// ---------------------------------------------------------------------------

/// Draw a full mesh with MVP transform and lighting.
pub fn draw_mesh(
    target: &mut RenderTarget,
    mesh: &Mesh,
    model: &Mat4,
    proj_view: &Mat4,
    state: &RenderState,
    material: &Material,
) {
    let mvp = *proj_view * *model;
    let vw = target.width as f32;
    let vh = target.height as f32;
    let base = [material.base_color[0], material.base_color[1], material.base_color[2]];

    let verts: Vec<ClipVertex> = mesh.vertices.iter()
        .map(|v| transform_vertex(v, &mvp, model, vw, vh))
        .collect();

    let idx = &mesh.indices;
    let n = idx.len() / 3;
    for t in 0..n {
        let i0 = idx[t * 3] as usize;
        let i1 = idx[t * 3 + 1] as usize;
        let i2 = idx[t * 3 + 2] as usize;

        if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
            continue;
        }

        let cv0 = verts[i0];
        let cv1 = verts[i1];
        let cv2 = verts[i2];

        // Compute face normal for shading.
        let n_ws = ((cv0.normal_ws + cv1.normal_ws + cv2.normal_ws) * (1.0 / 3.0)).normalize();
        let color = state.shade(n_ws, base);

        clip_and_rasterize(target, cv0, cv1, cv2, color);
    }
}

/// Draw only the specified clusters of a mesh (for Nanite-style LOD integration).
pub fn draw_mesh_clusters(
    target: &mut RenderTarget,
    mesh: &Mesh,
    cluster_indices: &[u32],
    model: &Mat4,
    proj_view: &Mat4,
    state: &RenderState,
    material: &Material,
) {
    let mvp = *proj_view * *model;
    let vw = target.width as f32;
    let vh = target.height as f32;
    let base = [material.base_color[0], material.base_color[1], material.base_color[2]];

    let verts: Vec<ClipVertex> = mesh.vertices.iter()
        .map(|v| transform_vertex(v, &mvp, model, vw, vh))
        .collect();

    for &ci in cluster_indices {
        let ci = ci as usize;
        if ci >= mesh.clusters.len() {
            continue;
        }
        let cluster = &mesh.clusters[ci];
        let start = cluster.start_index as usize;
        let end = start + cluster.count as usize;
        let idx = &mesh.indices;
        let tri_end = end.min(if idx.len() >= 3 { (idx.len() / 3) * 3 } else { 0 });

        let mut t = start;
        while t + 2 < tri_end {
            let i0 = idx[t] as usize;
            let i1 = idx[t + 1] as usize;
            let i2 = idx[t + 2] as usize;
            t += 3;

            if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
                continue;
            }

            let cv0 = verts[i0];
            let cv1 = verts[i1];
            let cv2 = verts[i2];

            let n_ws = ((cv0.normal_ws + cv1.normal_ws + cv2.normal_ws) * (1.0 / 3.0)).normalize();
            let color = state.shade(n_ws, base);

            clip_and_rasterize(target, cv0, cv1, cv2, color);
        }
    }
}

/// Draw N instances of a mesh with per-instance model matrices.
pub fn draw_instanced(
    target: &mut RenderTarget,
    mesh: &Mesh,
    transforms: &[Mat4],
    proj_view: &Mat4,
    state: &RenderState,
    material: &Material,
) {
    for model in transforms {
        draw_mesh(target, mesh, model, proj_view, state, material);
    }
}

// ---------------------------------------------------------------------------
// Hierarchical Z-buffer
// ---------------------------------------------------------------------------

/// A 4-level hierarchical depth buffer for early occlusion culling.
/// Level 0 = full-resolution depth copy; each subsequent level is a 2×-downsampled
/// maximum of the level above (conservative — if the HZB says occluded, it is).
pub struct HierarchicalZBuffer {
    pub levels: Vec<Vec<f32>>,
    pub widths: Vec<u32>,
    pub heights: Vec<u32>,
}

impl HierarchicalZBuffer {
    /// Build a 4-level HZB from the current depth buffer of `target`.
    pub fn build(target: &RenderTarget) -> HierarchicalZBuffer {
        let mut levels: Vec<Vec<f32>> = Vec::with_capacity(4);
        let mut widths: Vec<u32> = Vec::with_capacity(4);
        let mut heights: Vec<u32> = Vec::with_capacity(4);

        // Level 0: copy full-resolution depth.
        levels.push(target.depth.clone());
        widths.push(target.width);
        heights.push(target.height);

        // Levels 1..3: 2× downsampled max-depth.
        for lvl in 1..4 {
            let pw = widths[lvl - 1];
            let ph = heights[lvl - 1];
            let cw = (pw + 1) / 2;
            let ch = (ph + 1) / 2;
            let prev = &levels[lvl - 1];
            let mut cur = alloc::vec![0.0f32; (cw * ch) as usize];
            for cy in 0..ch {
                for cx in 0..cw {
                    let px0 = cx * 2;
                    let py0 = cy * 2;
                    let px1 = (px0 + 1).min(pw - 1);
                    let py1 = (py0 + 1).min(ph - 1);
                    let s00 = prev[(py0 * pw + px0) as usize];
                    let s10 = prev[(py0 * pw + px1) as usize];
                    let s01 = prev[(py1 * pw + px0) as usize];
                    let s11 = prev[(py1 * pw + px1) as usize];
                    // Conservative: take maximum depth (farthest = only cull when behind all).
                    cur[(cy * cw + cx) as usize] = s00.max(s10).max(s01).max(s11);
                }
            }
            levels.push(cur);
            widths.push(cw);
            heights.push(ch);
        }

        HierarchicalZBuffer { levels, widths, heights }
    }

    /// Returns `true` if the screen-space AABB at `near_depth` is fully occluded.
    /// Uses the coarsest HZB level that covers the footprint.
    pub fn test_aabb(&self, screen_min: Vec2, screen_max: Vec2, near_depth: f32) -> bool {
        let footprint_w = (screen_max.x - screen_min.x).abs();
        let footprint_h = (screen_max.y - screen_min.y).abs();

        let full_w = self.widths[0] as f32;
        let full_h = self.heights[0] as f32;

        // Pick the coarsest level whose footprint still spans at least ~1 texel.
        // (Footprint span at level `lvl` is footprint * widths[lvl] / full_w.)
        let mut chosen = 0usize;
        for lvl in 0..self.levels.len() {
            let tw = footprint_w * self.widths[lvl] as f32 / full_w;
            let th = footprint_h * self.heights[lvl] as f32 / full_h;
            if tw < 1.0 || th < 1.0 {
                break;
            }
            chosen = lvl;
        }

        let lw = self.widths[chosen];
        let lh = self.heights[chosen];
        let ldata = &self.levels[chosen];

        // Map screen-space min/max into level texel coords.
        let tx0 = floor32((screen_min.x / full_w) * lw as f32) as i32;
        let ty0 = floor32((screen_min.y / full_h) * lh as f32) as i32;
        let tx1 = ceil32((screen_max.x / full_w) * lw as f32) as i32;
        let ty1 = ceil32((screen_max.y / full_h) * lh as f32) as i32;

        let tx0 = tx0.max(0).min(lw as i32) as u32;
        let ty0 = ty0.max(0).min(lh as i32) as u32;
        let tx1 = tx1.max(0).min(lw as i32) as u32;
        let ty1 = ty1.max(0).min(lh as i32) as u32;

        // Empty footprint (off-screen or past a level edge) → nothing to occlude against.
        if tx0 >= tx1 || ty0 >= ty1 {
            return false;
        }

        // Find the maximum (farthest) depth value stored in the HZB footprint.
        let mut hzb_max = 0.0f32;
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let d = ldata[(ty * lw + tx) as usize];
                if d > hzb_max {
                    hzb_max = d;
                }
            }
        }

        // Occluded only if the AABB is behind every stored (farthest) surface.
        near_depth > hzb_max
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math3d::{Vec2, Vec3, Vec4, Mat4};
    use crate::mesh::{Vertex, Mesh, Material, generate_cube};

    fn make_clip_vertex(sx: f32, sy: f32, depth: f32) -> ClipVertex {
        // Build a ClipVertex positioned directly at screen-space coordinates.
        // We bypass the MVP for simplicity: set ndc z and w_inv directly.
        let z_ndc = depth * 2.0 - 1.0; // depth in [0,1] → ndc z in [-1,1]
        ClipVertex {
            clip: Vec4::new(0.0, 0.0, z_ndc, 1.0),
            ndc: Vec3::new(0.0, 0.0, z_ndc),
            screen: Vec2::new(sx, sy),
            normal_ws: Vec3::new(0.0, 0.0, 1.0),
            uv: Vec2::zero(),
            color: [1.0, 1.0, 1.0, 1.0],
            w_inv: 1.0,
        }
    }

    #[test]
    fn test_clear_color() {
        let mut rt = RenderTarget::new(16, 16);
        rt.clear(0xFF112233);
        assert!(rt.pixels.iter().all(|&p| p == 0xFF112233));
    }

    #[test]
    fn test_clear_depth() {
        let mut rt = RenderTarget::new(16, 16);
        // Write something first.
        for d in rt.depth.iter_mut() { *d = 0.5; }
        rt.clear_depth();
        assert!(rt.depth.iter().all(|&d| (d - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_rasterize_triangle_center() {
        let mut rt = RenderTarget::new(32, 32);
        rt.clear_all(0xFF000000);
        let bg = 0xFF000000u32;
        let fg = 0xFFFFFFFFu32;

        // Large triangle covering the center of the buffer.
        let v0 = make_clip_vertex(16.0, 2.0, 0.5);
        let v1 = make_clip_vertex(2.0, 30.0, 0.5);
        let v2 = make_clip_vertex(30.0, 30.0, 0.5);

        rasterize_triangle(&mut rt, &v0, &v1, &v2, fg);

        // Center pixel (16,16) should be set.
        assert_eq!(rt.pixel(16, 16), fg);
        // Corner pixel (0,0) should be untouched.
        assert_eq!(rt.pixel(0, 0), bg);
    }

    #[test]
    fn test_depth_test_behind() {
        let mut rt = RenderTarget::new(32, 32);
        rt.clear_all(0xFF000000);

        let close = 0xFFFFFFFFu32;
        let far_col = 0xFF0000FFu32;

        // Draw a close triangle.
        let v0 = make_clip_vertex(16.0, 2.0, 0.2);
        let v1 = make_clip_vertex(2.0, 30.0, 0.2);
        let v2 = make_clip_vertex(30.0, 30.0, 0.2);
        rasterize_triangle(&mut rt, &v0, &v1, &v2, close);

        // Attempt to draw a far triangle over it — should NOT overwrite.
        let f0 = make_clip_vertex(16.0, 2.0, 0.8);
        let f1 = make_clip_vertex(2.0, 30.0, 0.8);
        let f2 = make_clip_vertex(30.0, 30.0, 0.8);
        rasterize_triangle(&mut rt, &f0, &f1, &f2, far_col);

        assert_eq!(rt.pixel(16, 16), close);
    }

    #[test]
    fn test_depth_test_in_front() {
        let mut rt = RenderTarget::new(32, 32);
        rt.clear_all(0xFF000000);

        let far_col = 0xFF0000FFu32;
        let close = 0xFFFFFFFFu32;

        // Draw far triangle first.
        let f0 = make_clip_vertex(16.0, 2.0, 0.8);
        let f1 = make_clip_vertex(2.0, 30.0, 0.8);
        let f2 = make_clip_vertex(30.0, 30.0, 0.8);
        rasterize_triangle(&mut rt, &f0, &f1, &f2, far_col);

        // Draw closer triangle — should overwrite.
        let v0 = make_clip_vertex(16.0, 2.0, 0.2);
        let v1 = make_clip_vertex(2.0, 30.0, 0.2);
        let v2 = make_clip_vertex(30.0, 30.0, 0.2);
        rasterize_triangle(&mut rt, &v0, &v1, &v2, close);

        assert_eq!(rt.pixel(16, 16), close);
    }

    #[test]
    fn test_transform_vertex_identity() {
        let v = Vertex::new(Vec3::zero(), Vec3::z_axis(), Vec2::zero());
        let mvp = Mat4::identity();
        let model = Mat4::identity();
        let cv = transform_vertex(&v, &mvp, &model, 64.0, 64.0);
        // NDC origin → screen center.
        assert!((cv.screen.x - 32.0).abs() < 1.0, "screen.x={}", cv.screen.x);
        assert!((cv.screen.y - 32.0).abs() < 1.0, "screen.y={}", cv.screen.y);
    }

    #[test]
    fn test_shade_front_facing() {
        let state = RenderState::default_state();
        let n = Vec3::new(0.577_350_26, 0.577_350_26, 0.577_350_26); // aligned with light
        let color = state.shade(n, [1.0, 1.0, 1.0]);
        let r = color & 0xFF;
        // Should be bright (well above ambient).
        assert!(r > 100, "r={r} expected bright");
    }

    #[test]
    fn test_shade_back_facing() {
        let state = RenderState::default_state();
        let n = Vec3::new(-0.577_350_26, -0.577_350_26, -0.577_350_26); // away from light
        let color = state.shade(n, [1.0, 1.0, 1.0]);
        let r = color & 0xFF;
        // Should be near ambient only (≈ 5%).
        assert!(r < 20, "r={r} expected near-ambient");
    }

    #[test]
    fn test_draw_mesh_cube_sets_pixels() {
        let mut rt = RenderTarget::new(128, 128);
        rt.clear_all(0xFF000000);
        let bg = 0xFF000000u32;

        let mut mesh = generate_cube(1.0);
        mesh.compute_aabb();

        // Place camera a little back from origin looking at the cube.
        use crate::math3d::{perspective, look_at};
        let proj = perspective(60.0f32.to_radians(), 1.0, 0.1, 100.0);
        let view = look_at(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::zero(),
            Vec3::y_axis(),
        );
        let proj_view = proj * view;
        let model = Mat4::identity();
        let state = RenderState::default_state();
        let mat = Material::default_white();

        draw_mesh(&mut rt, &mesh, &model, &proj_view, &state, &mat);

        let filled = rt.pixels.iter().filter(|&&p| p != bg).count();
        assert!(filled > 0, "Expected some non-background pixels after drawing cube");
    }

    #[test]
    fn test_hzb_build_cleared_target() {
        let rt = RenderTarget::new(32, 32);
        // Fresh target has depth = 1.0 everywhere.
        let hzb = HierarchicalZBuffer::build(&rt);
        assert_eq!(hzb.level_count(), 4);
        // Level 0 should match.
        assert!(hzb.levels[0].iter().all(|&d| (d - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_screen_tiles_64x64() {
        let tiles = screen_tiles(64, 64);
        // 64/16 = 4 cols × 4 rows = 16 tiles.
        assert_eq!(tiles.len(), 16);
        // All tiles should be 16×16.
        for t in &tiles {
            assert_eq!(t.w, 16);
            assert_eq!(t.h, 16);
        }
    }

    #[test]
    fn test_screen_tiles_non_multiple() {
        // 20×20 with TILE_SIZE=16 → 2×2 = 4 tiles; some are partial.
        let tiles = screen_tiles(20, 20);
        assert_eq!(tiles.len(), 4);
        let last = tiles[3];
        assert_eq!(last.w, 4);
        assert_eq!(last.h, 4);
    }

    #[test]
    fn bench_raster_1000_triangles() {
        use crate::math3d::sqrt32;
        let mut rt = RenderTarget::new(256, 256);
        rt.clear_all(0xFF000000);
        let bg = 0xFF000000u32;

        // Deterministic LCG for reproducible "random" triangles.
        let mut seed: u32 = 0xDEAD_BEEF;
        let mut rng = || -> f32 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as f32 / 65535.0
        };

        for _ in 0..1000 {
            let x0 = rng() * 256.0;
            let y0 = rng() * 256.0;
            let x1 = rng() * 256.0;
            let y1 = rng() * 256.0;
            let x2 = rng() * 256.0;
            let y2 = rng() * 256.0;
            let depth = rng();
            let color = 0xFF000000 | ((rng() * 255.0) as u32) | (((rng() * 255.0) as u32) << 8);

            let v0 = make_clip_vertex(x0, y0, depth);
            let v1 = make_clip_vertex(x1, y1, depth);
            let v2 = make_clip_vertex(x2, y2, depth);
            rasterize_triangle(&mut rt, &v0, &v1, &v2, color);
        }

        let filled = rt.pixels.iter().filter(|&&p| p != bg).count();
        assert!(filled > 0, "Expected non-background pixels after 1000 triangles, got 0");
    }
}

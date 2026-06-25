//! Software graphics for the DominionOS desktop.
//!
//! The bootloader hands us a linear framebuffer; this module turns it into a real
//! 2-D drawing surface. It keeps a **back-buffer** in RAM (one `u32` per pixel,
//! `0x00RRGGBB`), draws the whole scene there cheaply, then **presents** rectangles
//! to the (slow, MMIO) hardware framebuffer. A small **cursor sprite** is composited
//! on top at present time, so moving the mouse only repaints a few hundred pixels —
//! the desktop stays responsive even under QEMU's interpreter.
//!
//! Primitives: filled/outlined rects, rounded panels, lines, an 8×8 scalable font
//! (the same `font8x8` the text console uses), and a vertical gradient for the
//! wallpaper. Everything is integer math — no `libm`, no floats.

use dominion_core::toolkit::DrawCmd;
use alloc::vec;
use alloc::vec::Vec;
use font8x8::legacy::BASIC_LEGACY;
use noto_sans_mono_bitmap::{get_raster, get_raster_width, FontWeight, RasterHeight};
use spin::Mutex;

/// A packed `0x00RRGGBB` colour.
pub type Rgb = u32;

pub const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

/// The cursor sprite (a classic arrow): `.` transparent, `X` white, `o` black edge.
const CURSOR: [&str; 19] = [
    "X..................",
    "Xo.................",
    "Xoo................",
    "Xooo...............",
    "Xoooo..............",
    "Xooooo.............",
    "Xoooooo............",
    "Xooooooo...........",
    "Xoooooooo..........",
    "Xooooooooo.........",
    "XooooooXXXX........",
    "XooXoo.............",
    "XoX.Xoo............",
    "XX..Xoo............",
    "X....Xoo...........",
    ".....Xoo...........",
    "......Xoo..........",
    "......Xoo..........",
    ".......X...........",
];
pub const CURSOR_W: usize = 19;
pub const CURSOR_H: usize = 19;

struct Screen {
    base: usize,
    width: usize,
    height: usize,
    stride: usize,
    bpp: usize,
    bgr: bool,
    gray: bool,
    back: Vec<u32>,
    /// What is currently on the hardware framebuffer (cursor excluded). The
    /// dashboard presents by **diffing** `back` against this and writing only the
    /// pixels that changed — the key to a smooth 30 fps over slow MMIO.
    front: Vec<u32>,
    cursor_x: usize,
    cursor_y: usize,
    /// The last cursor rect drawn, so it can be erased before the next.
    last_cursor: (usize, usize),
}

// Safe: the framebuffer is a fixed MMIO region this module owns while the desktop runs.
unsafe impl Send for Screen {}

static SCREEN: Mutex<Option<Screen>> = Mutex::new(None);

/// Take over the framebuffer for graphics. Returns `(width, height)`, or `None` if
/// there is no framebuffer.
pub fn init() -> Option<(usize, usize)> {
    let fb = crate::vga_buffer::raw_framebuffer()?;
    let back = vec![0u32; fb.width * fb.height];
    // `front` starts as all-ones so the first diff-present writes every pixel.
    let front = vec![0xFFFF_FFFFu32; fb.width * fb.height];
    let dims = (fb.width, fb.height);
    *SCREEN.lock() = Some(Screen {
        base: fb.base,
        width: fb.width,
        height: fb.height,
        stride: fb.stride,
        bpp: fb.bpp,
        bgr: fb.bgr,
        gray: fb.gray,
        back,
        front,
        cursor_x: fb.width / 2,
        cursor_y: fb.height / 2,
        last_cursor: (fb.width / 2, fb.height / 2),
    });
    Some(dims)
}

/// Returns `true` if the framebuffer has been initialised (i.e. `init()` succeeded).
/// Use this to guard gfx calls in headless / benchmark mode where there is no display.
pub fn available() -> bool {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| SCREEN.lock().is_some())
}

/// Run `f` with the drawing surface (locks the screen for the duration).
pub fn draw<R>(f: impl FnOnce(&mut Painter) -> R) -> R {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        let mut guard = SCREEN.lock();
        let screen = guard.as_mut().expect("gfx not initialised");
        let mut p = Painter { s: screen };
        f(&mut p)
    })
}

/// Present the whole back-buffer to the hardware framebuffer (then the cursor).
pub fn present() {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            let (w, h) = (s.width, s.height);
            s.present_rect(0, 0, w, h);
        }
    });
}

/// Present only a sub-rectangle (used to erase the old cursor position cheaply).
pub fn present_rect(x: usize, y: usize, w: usize, h: usize) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            s.present_rect(x, y, w, h);
        }
    });
}

/// Move the cursor to `(x, y)`, repainting only the affected regions.
pub fn move_cursor(x: usize, y: usize) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            let (ox, oy) = (s.cursor_x, s.cursor_y);
            s.cursor_x = x.min(s.width.saturating_sub(1));
            s.cursor_y = y.min(s.height.saturating_sub(1));
            s.present_rect(ox, oy, CURSOR_W, CURSOR_H);
            s.present_rect(s.cursor_x, s.cursor_y, CURSOR_W, CURSOR_H);
        }
    });
}

pub fn dimensions() -> Option<(usize, usize)> {
    SCREEN.lock().as_ref().map(|s| (s.width, s.height))
}

/// Read a back-buffer pixel (`0x00RRGGBB`) — for on-metal render verification.
pub fn back_pixel(x: usize, y: usize) -> Option<u32> {
    SCREEN.lock().as_ref().and_then(|s| {
        if x < s.width && y < s.height {
            Some(s.back[y * s.width + x])
        } else {
            None
        }
    })
}

/// Set the pointer position (the next present composites the cursor here).
pub fn set_cursor(x: usize, y: usize) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            s.cursor_x = x.min(s.width.saturating_sub(1));
            s.cursor_y = y.min(s.height.saturating_sub(1));
        }
    });
}

/// Composite just the cursor (cheap) — used on frames where only the pointer moved.
pub fn present_cursor() {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            s.present_cursor();
        }
    });
}

// ───────────────────────── anti-aliased text ─────────────────────────

/// Pick the nearest available Noto raster height for a requested pixel size.
fn raster_height(size: i32) -> RasterHeight {
    if size <= 17 {
        RasterHeight::Size16
    } else if size <= 22 {
        RasterHeight::Size20
    } else {
        RasterHeight::Size24
    }
}

/// On-screen width of `text` at `size`, using the mono advance.
pub fn text_width(text: &str, size: i32) -> i32 {
    let adv = get_raster_width(FontWeight::Regular, raster_height(size)) as i32;
    text.chars().count() as i32 * adv
}

/// A clip rectangle in pixel bounds `[x0,x1) × [y0,y1)`.
#[derive(Clone, Copy)]
struct Clip {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

/// A back-buffer render target: the pixel slice, its dimensions, and the active
/// clip (damage) rectangle. Bundling these keeps the AA-text helpers down to a
/// handful of arguments instead of threading four positional parameters through
/// every call.
struct Canvas<'a> {
    back: &'a mut [u32],
    w: usize,
    h: usize,
    clip: Clip,
}

impl Canvas<'_> {
    /// Blend a coverage-weighted colour over one pixel (for AA glyph edges),
    /// rejecting anything outside the clip (damage) rectangle.
    #[inline]
    fn blend(&mut self, x: i32, y: i32, color: u32, cov: u8) {
        if x < self.clip.x0 || y < self.clip.y0 || x >= self.clip.x1 || y >= self.clip.y1 {
            return;
        }
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h || cov == 0 {
            return;
        }
        let idx = y as usize * self.w + x as usize;
        if cov == 255 {
            self.back[idx] = color;
            return;
        }
        let dst = self.back[idx];
        let a = cov as u32;
        let ia = 255 - a;
        let r = (((color >> 16) & 0xff) * a + ((dst >> 16) & 0xff) * ia) / 255;
        let g = (((color >> 8) & 0xff) * a + ((dst >> 8) & 0xff) * ia) / 255;
        let b = ((color & 0xff) * a + (dst & 0xff) * ia) / 255;
        self.back[idx] = (r << 16) | (g << 8) | b;
    }

    /// Draw anti-aliased text at `(x,y)` (top-left), clipped to the damage rectangle.
    fn draw_text(&mut self, x: i32, y: i32, text: &str, size: i32, color: u32) {
        let rh = raster_height(size);
        let adv = get_raster_width(FontWeight::Regular, rh) as i32;
        let mut cx = x;
        for ch in text.chars() {
            // Skip glyphs whose cell is entirely outside the clip rect (x-range).
            if cx + adv > self.clip.x0 && cx < self.clip.x1 {
                if let Some(g) = get_raster(ch, FontWeight::Regular, rh) {
                    for (gy, row) in g.raster().iter().enumerate() {
                        // Row-level cull: during an incremental repaint the damage band is
                        // short, so most glyph rows lie outside it — skip the whole pixel row
                        // without touching it.
                        let py = y + gy as i32;
                        if py < self.clip.y0 || py >= self.clip.y1 {
                            continue;
                        }
                        for (gx, &cov) in row.iter().enumerate() {
                            if cov == 0 {
                                continue; // transparent — skip the blend call entirely
                            }
                            self.blend(cx + gx as i32, py, color, cov);
                        }
                    }
                }
            }
            cx += adv;
        }
    }
}

/// Rasterise a whole toolkit **scene** into the back-buffer: vector primitives via
/// `dominion_core::toolkit::raster`, text via the anti-aliased Noto font (vertically
/// centred in each text rect). This is the framebuffer backend for the GPU-first
/// toolkit — the live dashboard renders entirely through here.
pub fn raster_scene(scene: &[DrawCmd]) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            let (w, h) = (s.width, s.height);
            let clip = Clip { x0: 0, y0: 0, x1: w as i32, y1: h as i32 };
            s.raster_clipped(scene, clip);
        }
    });
}

/// Rasterise the scene but only **within the damage rectangle** `(x,y,w,h)` — the
/// incremental fast path. Combined with [`present_diff_rect`], a change to one panel
/// repaints just that panel instead of the whole 1080p screen, which is the bulk of a
/// frame's cost under QEMU's interpreter.
pub fn raster_scene_clipped(scene: &[DrawCmd], rect: (i32, i32, i32, i32)) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            let (w, h) = (s.width as i32, s.height as i32);
            let clip = Clip {
                x0: rect.0.max(0),
                y0: rect.1.max(0),
                x1: (rect.0 + rect.2).min(w),
                y1: (rect.1 + rect.3).min(h),
            };
            s.raster_clipped(scene, clip);
        }
    });
}

/// Present by **diffing**: write only the pixels that changed since the last present
/// (cursor excluded), then composite the cursor. Over slow MMIO this is what makes a
/// full-screen 1080p dashboard redraw smoothly at 30 fps.
pub fn present_diff() {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            // Walk row-by-row so the (x, y) of each changed pixel comes from cheap
            // increments instead of a `%`/`/` pair per pixel — at 1080p that removed
            // ~4 million integer divisions from every full-screen present. The set of
            // pixels written to MMIO (those that differ from `front`) is unchanged.
            let (w, h) = (s.width, s.height);
            for y in 0..h {
                s.flush_changed_row(y, 0, w);
            }
            s.present_cursor();
        }
    });
}

/// Diff-present but scan **only the damage rectangle** — avoids the full-screen
/// back-vs-front scan when only a small region changed. Then composites the cursor.
pub fn present_diff_rect(rect: (i32, i32, i32, i32)) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(s) = SCREEN.lock().as_mut() {
            let (w, h) = (s.width as i32, s.height as i32);
            let x0 = rect.0.max(0) as usize;
            let y0 = rect.1.max(0) as usize;
            let x1 = ((rect.0 + rect.2).min(w).max(0)) as usize;
            let y1 = ((rect.1 + rect.3).min(h).max(0)) as usize;
            for y in y0..y1 {
                s.flush_changed_row(y, x0, x1);
            }
            s.present_cursor();
        }
    });
}

/// The drawing API, operating on the in-RAM back-buffer.
pub struct Painter<'a> {
    s: &'a mut Screen,
}

impl Painter<'_> {
    pub fn width(&self) -> usize {
        self.s.width
    }
    pub fn height(&self) -> usize {
        self.s.height
    }

    #[inline]
    pub fn pixel(&mut self, x: usize, y: usize, c: Rgb) {
        if x < self.s.width && y < self.s.height {
            self.s.back[y * self.s.width + x] = c;
        }
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, c: Rgb) {
        let x0 = x.max(0) as usize;
        let y0 = y.max(0) as usize;
        let x1 = ((x + w).max(0) as usize).min(self.s.width);
        let y1 = ((y + h).max(0) as usize).min(self.s.height);
        for yy in y0..y1 {
            let row = yy * self.s.width;
            for xx in x0..x1 {
                self.s.back[row + xx] = c;
            }
        }
    }

    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, c: Rgb) {
        self.fill_rect(x, y, w, 1, c);
        self.fill_rect(x, y + h - 1, w, 1, c);
        self.fill_rect(x, y, 1, h, c);
        self.fill_rect(x + w - 1, y, 1, h, c);
    }

    /// A filled panel with rounded corners (radius `r`).
    pub fn rounded(&mut self, x: i32, y: i32, w: i32, h: i32, r: i32, c: Rgb) {
        let r = r.max(0).min(w / 2).min(h / 2);
        self.fill_rect(x + r, y, w - 2 * r, h, c);
        self.fill_rect(x, y + r, r, h - 2 * r, c);
        self.fill_rect(x + w - r, y + r, r, h - 2 * r, c);
        // Four quarter-circle corners.
        self.corner(x + r, y + r, r, c, true, true);
        self.corner(x + w - r - 1, y + r, r, c, false, true);
        self.corner(x + r, y + h - r - 1, r, c, true, false);
        self.corner(x + w - r - 1, y + h - r - 1, r, c, false, false);
    }

    fn corner(&mut self, cx: i32, cy: i32, r: i32, c: Rgb, left: bool, top: bool) {
        for dy in 0..=r {
            for dx in 0..=r {
                if dx * dx + dy * dy <= r * r {
                    let px = if left { cx - dx } else { cx + dx };
                    let py = if top { cy - dy } else { cy + dy };
                    if px >= 0 && py >= 0 {
                        self.pixel(px as usize, py as usize, c);
                    }
                }
            }
        }
    }

    /// A vertical gradient fill from `top` to `bottom` colour.
    pub fn gradient(&mut self, x: i32, y: i32, w: i32, h: i32, top: Rgb, bottom: Rgb) {
        if h <= 0 {
            return;
        }
        let (tr, tg, tb) = ((top >> 16) & 0xff, (top >> 8) & 0xff, top & 0xff);
        let (br, bg, bb) = ((bottom >> 16) & 0xff, (bottom >> 8) & 0xff, bottom & 0xff);
        for row in 0..h {
            let t = row as i64;
            let denom = (h - 1).max(1) as i64;
            let r = (tr as i64 + (br as i64 - tr as i64) * t / denom) as u32;
            let g = (tg as i64 + (bg as i64 - tg as i64) * t / denom) as u32;
            let b = (tb as i64 + (bb as i64 - tb as i64) * t / denom) as u32;
            self.fill_rect(x, y + row, w, 1, r << 16 | g << 8 | b);
        }
    }

    /// Draw `text` at `(x, y)` in colour `fg`, scaled by `scale`, transparent
    /// background. Returns the x-advance.
    pub fn text(&mut self, x: i32, y: i32, text: &str, scale: i32, fg: Rgb) -> i32 {
        let mut cx = x;
        let s = scale.max(1);
        for ch in text.bytes() {
            let glyph = BASIC_LEGACY.get(ch as usize).copied().unwrap_or([0; 8]);
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in 0..8i32 {
                    if (bits >> gx) & 1 != 0 {
                        self.fill_rect(cx + gx * s, y + gy as i32 * s, s, s, fg);
                    }
                }
            }
            cx += 8 * s;
        }
        cx
    }

    /// The on-screen pixel width of `text` at `scale`.
    pub fn text_width(text: &str, scale: i32) -> i32 {
        text.len() as i32 * 8 * scale.max(1)
    }
}

/// Returns `true` for GPU-only commands that have no software fallback at this layer.
/// `Mesh3D` is NOT in this list — it is handled inline in `raster_clipped`.
#[inline(always)]
fn is_gpu_only_cmd(cmd: &DrawCmd) -> bool {
    matches!(
        cmd,
        DrawCmd::VectorPath { .. }
            | DrawCmd::GpuText { .. }
            | DrawCmd::MediaBuffer { .. }
            | DrawCmd::Scene3D { .. }
            | DrawCmd::Particles { .. }
            | DrawCmd::SdfShadow { .. }
    )
}

impl Screen {
    /// Rasterise a toolkit scene into the back-buffer, writing only within `clip`:
    /// vector primitives via `toolkit::raster::render_clipped`, text via the AA Noto
    /// font (also clipped). Shared by the full-screen and damage-rect paths.
    ///
    /// `Mesh3D` commands are software-rasterised inline: a `RenderTarget` is
    /// created at the viewport size, `raster3d::draw_mesh` fills it, and the
    /// result is blitted into the back-buffer with an R/B channel swap
    /// (raster3d uses 0xFFBBGGRR; the back-buffer uses 0x00RRGGBB).
    ///
    /// GPU-only commands (`VectorPath`, `GpuText`, `MediaBuffer`, `Scene3D`,
    /// `Particles`, `SdfShadow`) are still skipped here — they require a
    /// GPU compositor that is not yet active at boot time.
    fn raster_clipped(&mut self, scene: &[DrawCmd], clip: Clip) {
        let (w, h) = (self.width, self.height);
        let clip_box = (clip.x0, clip.y0, clip.x1 - clip.x0, clip.y1 - clip.y0);
        // Composite in **true z-order**: walk the scene once, rasterising each run of
        // vector/fill commands, then painting any text that immediately follows *before*
        // moving on to later fills. Drawing all text in one trailing pass (as before)
        // let labels from lower surfaces — desktop icons, windows behind — paint over a
        // window that is supposed to occlude them. Interleaving fixes that occlusion.
        //
        // 3D / GPU commands are explicitly partitioned out before the 2D sub-slice is
        // handed to `render_clipped` — belt-and-suspenders: `render_clipped` already
        // skips them, but keeping them out of the slice makes the intent crystal-clear
        // and keeps the slice contiguous and small for the common (all-2D) case.
        let mut i = 0;
        while i < scene.len() {
            // Software-rasterise Mesh3D inline: render into a temp RenderTarget
            // then blit into the back-buffer with a channel swap (0xFFBBGGRR → 0x00RRGGBB).
            if let DrawCmd::Mesh3D { mesh, material, model, proj_view, viewport } = &scene[i] {
                let vw = viewport.w.max(1) as u32;
                let vh = viewport.h.max(1) as u32;
                let mut rt = dominion_core::raster3d::RenderTarget::new(vw, vh);
                rt.clear_all(0xFF000000);
                let state = dominion_core::raster3d::RenderState::default_state();
                dominion_core::raster3d::draw_mesh(&mut rt, &mesh.0, model, proj_view, &state, material);
                let ox = viewport.x.max(0) as usize;
                let oy = viewport.y.max(0) as usize;
                for dy in 0..vh as usize {
                    for dx in 0..vw as usize {
                        let sx = ox + dx;
                        let sy = oy + dy;
                        if sx < w && sy < h {
                            let src = rt.pixels[dy * vw as usize + dx];
                            if src != 0xFF000000 {
                                // Swap R (bits 7:0) ↔ B (bits 23:16) to match back-buffer format.
                                let r = src & 0xFF;
                                let g = (src >> 8) & 0xFF;
                                let b = (src >> 16) & 0xFF;
                                self.back[sy * w + sx] = (r << 16) | (g << 8) | b;
                            }
                        }
                    }
                }
                i += 1;
                continue;
            }

            // Skip GPU-only commands with no software fallback.
            if is_gpu_only_cmd(&scene[i]) {
                i += 1;
                continue;
            }

            // Accumulate a contiguous run of 2D non-text commands.
            let start = i;
            while i < scene.len() && !matches!(scene[i], DrawCmd::Text { .. }) && !is_gpu_only_cmd(&scene[i]) {
                i += 1;
            }
            if i > start {
                let _ = dominion_core::toolkit::raster::render_clipped(&scene[start..i], &mut self.back, w, h, clip_box);
            }
            // Paint the contiguous run of text commands at this z-level.
            if i < scene.len() && matches!(scene[i], DrawCmd::Text { .. }) {
                let mut canvas = Canvas { back: &mut self.back, w, h, clip };
                while i < scene.len() {
                    let DrawCmd::Text { rect, text, color, size } = &scene[i] else { break };
                    // Skip text whose rect misses the clip entirely.
                    let on = rect.x < clip.x1 && rect.x + rect.w > clip.x0 && rect.y < clip.y1 && rect.y + rect.h > clip.y0;
                    if on {
                        let col = rgb(color.r, color.g, color.b);
                        let rhpx = raster_height(*size).val() as i32;
                        let ty = rect.y + (rect.h - rhpx).max(0) / 2;
                        canvas.draw_text(rect.x, ty, text, *size, col);
                    }
                    i += 1;
                }
            }
        }
    }

    /// Copy a back-buffer rectangle to the hardware framebuffer, compositing the
    /// cursor sprite on top wherever it intersects.
    fn present_rect(&mut self, x: usize, y: usize, w: usize, h: usize) {
        let x1 = (x + w).min(self.width);
        let y1 = (y + h).min(self.height);
        for yy in y..y1 {
            for xx in x..x1 {
                let mut c = self.back[yy * self.width + xx];
                if let Some(cc) = self.cursor_pixel(xx, yy) {
                    c = cc;
                }
                self.put_fb(xx, yy, c);
            }
        }
    }

    /// Erase the cursor at its previous position (restoring the back-buffer there),
    /// then composite it at the current position. Cheap — two ~19×19 blits.
    fn present_cursor(&mut self) {
        // Restore the region under the old cursor from the (cursor-free) back-buffer.
        let (ox, oy) = self.last_cursor;
        let ox1 = (ox + CURSOR_W).min(self.width);
        let oy1 = (oy + CURSOR_H).min(self.height);
        for yy in oy..oy1 {
            for xx in ox..ox1 {
                self.put_fb(xx, yy, self.back[yy * self.width + xx]);
            }
        }
        // Draw the cursor at the new position.
        let (cx, cy) = (self.cursor_x, self.cursor_y);
        let cx1 = (cx + CURSOR_W).min(self.width);
        let cy1 = (cy + CURSOR_H).min(self.height);
        for yy in cy..cy1 {
            for xx in cx..cx1 {
                let c = self.cursor_pixel(xx, yy).unwrap_or(self.back[yy * self.width + xx]);
                self.put_fb(xx, yy, c);
            }
        }
        self.last_cursor = (cx, cy);
    }

    /// The cursor sprite colour at an absolute pixel, if any.
    fn cursor_pixel(&self, x: usize, y: usize) -> Option<u32> {
        if x < self.cursor_x || y < self.cursor_y {
            return None;
        }
        let lx = x - self.cursor_x;
        let ly = y - self.cursor_y;
        if lx >= CURSOR_W || ly >= CURSOR_H {
            return None;
        }
        match CURSOR[ly].as_bytes()[lx] {
            b'X' => Some(0x00FF_FFFF),
            b'o' => Some(0x0000_0000),
            _ => None,
        }
    }

    /// Diff-present the pixels of row `y` in the column range `[x0, x1)`: write only
    /// those that changed since the last present, **coalescing each maximal run of
    /// adjacent changed pixels into a single [`put_fb_run`]** so a span of changed
    /// pixels is one bulk MMIO sweep with the encoding branch resolved once, instead
    /// of an independently-decoded `put_fb` per pixel. The pixels written and the
    /// bytes written per pixel are identical to the previous per-pixel diff loop.
    #[inline]
    fn flush_changed_row(&mut self, y: usize, x0: usize, x1: usize) {
        let row = y * self.width;
        let mut x = x0;
        while x < x1 {
            // Skip an unchanged span.
            if self.back[row + x] == self.front[row + x] {
                x += 1;
                continue;
            }
            // Extend a changed run as far as it goes, mirroring into `front`.
            let start = x;
            while x < x1 && self.back[row + x] != self.front[row + x] {
                self.front[row + x] = self.back[row + x];
                x += 1;
            }
            self.put_fb_run(start, y, &self.back[row + start..row + x]);
        }
    }

    #[inline]
    fn put_fb(&self, x: usize, y: usize, c: u32) {
        let (r, g, b) = ((c >> 16) as u8, (c >> 8) as u8, c as u8);
        let offset = (y * self.stride + x) * self.bpp;
        let p = (self.base + offset) as *mut u8;
        unsafe {
            if self.gray {
                let lum = ((r as u16 * 30 + g as u16 * 59 + b as u16 * 11) / 100) as u8;
                p.write_volatile(lum);
            } else if self.bgr {
                p.write_volatile(b);
                p.add(1).write_volatile(g);
                p.add(2).write_volatile(r);
            } else {
                p.write_volatile(r);
                p.add(1).write_volatile(g);
                p.add(2).write_volatile(b);
            }
        }
    }

    /// Write a horizontal **run** of `src` pixels to the hardware framebuffer
    /// starting at `(x, y)`. Equivalent to calling [`put_fb`] for each pixel in
    /// order — the exact same bytes are written (the pad byte of a 4-bpp pixel is
    /// left untouched, just as before) — but the per-pixel offset multiply and the
    /// gray/bgr/rgb branch are hoisted out of the loop, leaving a single base
    /// pointer that simply advances by `bpp`. On the diff present path most changed
    /// pixels fall in contiguous spans, so this removes a large amount of repeated
    /// address arithmetic and branching around each MMIO write.
    #[inline]
    fn put_fb_run(&self, x: usize, y: usize, src: &[u32]) {
        let offset = (y * self.stride + x) * self.bpp;
        let mut p = (self.base + offset) as *mut u8;
        let bpp = self.bpp;
        unsafe {
            if self.gray {
                for &c in src {
                    let (r, g, b) = ((c >> 16) as u8, (c >> 8) as u8, c as u8);
                    let lum = ((r as u16 * 30 + g as u16 * 59 + b as u16 * 11) / 100) as u8;
                    p.write_volatile(lum);
                    p = p.add(bpp);
                }
            } else if self.bgr {
                for &c in src {
                    let (r, g, b) = ((c >> 16) as u8, (c >> 8) as u8, c as u8);
                    p.write_volatile(b);
                    p.add(1).write_volatile(g);
                    p.add(2).write_volatile(r);
                    p = p.add(bpp);
                }
            } else {
                for &c in src {
                    let (r, g, b) = ((c >> 16) as u8, (c >> 8) as u8, c as u8);
                    p.write_volatile(r);
                    p.add(1).write_volatile(g);
                    p.add(2).write_volatile(b);
                    p = p.add(bpp);
                }
            }
        }
    }
}

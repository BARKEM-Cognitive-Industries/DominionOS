//! GPU-first UI toolkit core — the renderer-agnostic heart of the "beautiful UI
//! library" (see `docs/ui/rendering-and-toolkit.md` and `docs/ui/design-system-and-shell.md`).
//!
//! A UI here is described **once**, as a backend-agnostic *scene* (a list of draw
//! commands) produced from a widget tree, a **design-token theme**, and a
//! **constraint layout** engine. *How* that scene is rasterised is the backend's
//! job: a [`Backend::Gpu`] path when an accelerator is present (Stage 4 makes the
//! GPU a first-class compute node), falling back to [`Backend::Framebuffer`]
//! software rendering (`surface.rs`) otherwise. The same widget tree yields the
//! **identical scene** on either backend, so the UI is written once and looks the
//! same everywhere — only the speed differs.
//!
//! This module is the testable core: tokens, layout, scene-building, backend
//! selection, and hit-testing — all pure, safe `no_std`, no display required.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::math3d::Mat4;
use crate::mesh::{Material, MeshHandle};

// ───────────────────────────── geometry ─────────────────────────────

/// An axis-aligned rectangle in device-independent pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Rect {
        Rect { x, y, w, h }
    }
    /// Does the rect contain a point?
    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }
    /// Shrink on all sides by `by` (padding), clamped to non-negative size.
    pub fn inset(&self, by: i32) -> Rect {
        Rect {
            x: self.x + by,
            y: self.y + by,
            w: (self.w - 2 * by).max(0),
            h: (self.h - 2 * by).max(0),
        }
    }
}

/// An 8-bit-per-channel RGBA colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
        Color { r, g, b, a: 255 }
    }
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
        Color { r, g, b, a }
    }
    /// Perceptual luminance ×1000 (Rec. 601), for contrast checks.
    pub fn luminance_milli(&self) -> u32 {
        (299 * self.r as u32 + 587 * self.g as u32 + 114 * self.b as u32) / 255
    }
}

// ──────────────────────────── design tokens ────────────────────────────

/// The design-token theme — the single source of the OS's *look*. Semantic tokens
/// (not raw colours scattered through code) are what make a UI coherent and
/// re-skinnable; changing the theme re-skins everything at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    pub bg: Color,
    pub surface: Color,
    pub primary: Color,
    pub on_primary: Color,
    pub text: Color,
    pub muted: Color,
    pub accent: Color,
    pub danger: Color,
    /// Corner radius for surfaces/buttons.
    pub radius: i32,
    /// Base spacing unit (a 4/8px scale multiplier).
    pub space: i32,
    /// Base body font size.
    pub font_size: i32,
}

impl Theme {
    /// The default dark theme — calm, high-contrast, modern.
    pub fn dark() -> Theme {
        Theme {
            bg: Color::rgb(0x12, 0x14, 0x18),
            surface: Color::rgb(0x1c, 0x1f, 0x26),
            primary: Color::rgb(0x4f, 0x9c, 0xff),
            on_primary: Color::rgb(0x0a, 0x0c, 0x10),
            text: Color::rgb(0xe8, 0xea, 0xed),
            muted: Color::rgb(0x8a, 0x90, 0x9c),
            accent: Color::rgb(0x9d, 0x7c, 0xff),
            danger: Color::rgb(0xff, 0x5c, 0x5c),
            radius: 10,
            space: 8,
            font_size: 15,
        }
    }

    /// The light theme — same token roles, inverted surfaces.
    pub fn light() -> Theme {
        Theme {
            bg: Color::rgb(0xf6, 0xf7, 0xf9),
            surface: Color::rgb(0xff, 0xff, 0xff),
            primary: Color::rgb(0x1f, 0x6f, 0xeb),
            on_primary: Color::rgb(0xff, 0xff, 0xff),
            text: Color::rgb(0x14, 0x16, 0x1a),
            muted: Color::rgb(0x5a, 0x60, 0x6a),
            accent: Color::rgb(0x6d, 0x4a, 0xe0),
            danger: Color::rgb(0xd2, 0x2f, 0x2f),
            radius: 10,
            space: 8,
            font_size: 15,
        }
    }

    /// A spacing step on the base scale (`step` × `space`).
    pub fn space_n(&self, step: i32) -> i32 {
        self.space * step
    }
}

// ─────────────────────────── monospace font metrics ───────────────────────────
//
// The renderer draws text with the **Noto Sans Mono** bitmap font at one of three
// raster heights (16/20/24 px), and every glyph at a given height has the *same*
// advance. Any code that positions a text caret (the editor, terminal, search box)
// must use the **exact same** advance, or the caret drifts from the glyphs. These two
// functions are the single source of truth, matching the kernel's `raster_height`
// thresholds and the font's real `RASTER_WIDTH`s (regular: 16→7, 20→9, 24→11 px), so
// the pure-side layout and the on-metal renderer agree to the pixel.

/// The monospace x-advance (px per character) the renderer uses for `size`.
pub fn mono_advance(size: i32) -> i32 {
    if size <= 17 {
        7
    } else if size <= 22 {
        9
    } else {
        11
    }
}

/// The glyph cell height (px) the renderer rasterises `size` at (16/20/24).
pub fn glyph_height(size: i32) -> i32 {
    if size <= 17 {
        16
    } else if size <= 22 {
        20
    } else {
        24
    }
}

// ──────────────────────────── renderer backends ────────────────────────────

/// Which rasteriser is live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Hardware-accelerated (Stage 4 GPU compute node).
    Gpu,
    /// Software fallback to the framebuffer (`surface.rs`).
    Framebuffer,
}

/// One backend-agnostic draw command. A *scene* is a list of these — the contract
/// between the toolkit and any renderer.
///
/// **2D commands** (Rect…Polyline) work on every renderer and are the primary
/// widget-tree output.  **3D commands** (Mesh3D…Scene3D) are emitted by
/// applications that place geometry into the unified desktop scene graph; the
/// compositor ingests them into the Global Render Dependency Graph.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawCmd {
    // ── 2D / UI commands ─────────────────────────────────────────────────────
    Rect { rect: Rect, color: Color, radius: i32 },
    Text { rect: Rect, text: String, color: Color, size: i32 },
    /// A straight line of pixel `width` from `(x0,y0)` to `(x1,y1)`.
    Line { x0: i32, y0: i32, x1: i32, y1: i32, color: Color, width: i32 },
    /// A cubic Bézier (node-editor wire): endpoints `p0`,`p1`, controls `c0`,`c1`.
    Bezier { p0: (i32, i32), c0: (i32, i32), c1: (i32, i32), p1: (i32, i32), color: Color, width: i32 },
    /// A filled circle centred at `(cx,cy)`.
    Disc { cx: i32, cy: i32, r: i32, color: Color },
    /// A connected sequence of line segments (charts) of pixel `width`.
    Polyline { points: Vec<(i32, i32)>, color: Color, width: i32 },

    // ── 3D / scene-graph commands ─────────────────────────────────────────────
    /// Render a 3D mesh into `viewport` on screen.
    ///
    /// The command is self-contained: the caller owns the mesh (via a cheap
    /// `Arc` clone) and supplies the full camera matrices, so the compositor
    /// can rasterise it without an external asset registry.
    Mesh3D {
        /// Shared reference to the mesh geometry (zero-copy across clones).
        mesh: MeshHandle,
        /// PBR surface material for this draw call.
        material: Material,
        /// Object-to-world transform (column-major Mat4).
        model: Mat4,
        /// World-to-NDC view-projection transform (column-major Mat4).
        proj_view: Mat4,
        /// Screen rectangle where the 3D viewport is composited.
        viewport: Rect,
    },

    /// A compute-native vector path (GPU Bézier pipeline, see `vectorpath.rs`).
    /// The path data is an opaque index into the compositor's vector-path registry.
    VectorPath {
        path_id: u32,
        /// Bounding rect for damage tracking.
        rect: Rect,
        /// Fill color (applied as a tint over the path's own fill, if any).
        color: Color,
    },

    /// A GPU-native text run using quadratic-Bézier glyph outlines (see `fontgpu.rs`).
    /// The `run_id` is a handle registered with the font service.
    GpuText {
        run_id: u32,
        rect: Rect,
        color: Color,
        /// Font size in pixels.
        px_size: i32,
    },

    /// A zero-copy media buffer (video frame, camera feed, etc.).
    /// `handle` is a sysmem token produced by the media service.
    MediaBuffer {
        handle: u64,
        rect: Rect,
        /// z-order override for overlay scanout (0 = composited normally).
        overlay_z: i32,
    },

    /// Embed a full 3D scene reference. The compositor merges the referenced scene
    /// into its Global Render Dependency Graph (RDG).
    /// `scene_id` is a handle registered with the compositor service.
    Scene3D {
        scene_id: u32,
        /// Viewport within the 2D surface where the 3D scene is projected.
        viewport: Rect,
        /// Camera index within the referenced scene.
        camera_idx: u32,
    },

    /// Particle system node — N particles with positions/velocities updated by GPU compute.
    Particles {
        emitter_id: u32,
        rect: Rect,
        max_count: u32,
        color: Color,
    },

    /// A signed-distance-field shadow cast by a vector UI element.
    /// Used by the compositor to generate physically-accurate shadows
    /// from UI paths without texture-filtering artifacts.
    SdfShadow {
        source_path_id: u32,
        rect: Rect,
        /// Shadow blur radius in pixels.
        blur_radius: i32,
        color: Color,
    },
}

/// A renderer records a scene; the concrete backend decides how to rasterise it.
pub trait Renderer {
    fn backend(&self) -> Backend;
    fn push(&mut self, cmd: DrawCmd);
    fn commands(&self) -> &[DrawCmd];
    /// Replay a whole scene.
    fn submit(&mut self, scene: &[DrawCmd]) {
        for cmd in scene {
            self.push(cmd.clone());
        }
    }
}

/// The GPU backend (records commands; real rasterisation uploads them to the
/// accelerator). Reports [`Backend::Gpu`].
#[derive(Default)]
pub struct GpuRenderer {
    cmds: Vec<DrawCmd>,
}
impl Renderer for GpuRenderer {
    fn backend(&self) -> Backend {
        Backend::Gpu
    }
    fn push(&mut self, cmd: DrawCmd) {
        self.cmds.push(cmd);
    }
    fn commands(&self) -> &[DrawCmd] {
        &self.cmds
    }
}

/// The software framebuffer backend — the universal floor when no GPU is present.
#[derive(Default)]
pub struct FramebufferRenderer {
    cmds: Vec<DrawCmd>,
}
impl Renderer for FramebufferRenderer {
    fn backend(&self) -> Backend {
        Backend::Framebuffer
    }
    fn push(&mut self, cmd: DrawCmd) {
        self.cmds.push(cmd);
    }
    fn commands(&self) -> &[DrawCmd] {
        &self.cmds
    }
}

/// Pick the best available renderer — **GPU first, framebuffer fallback** — so the
/// OS is fast where it can be and always renders where it cannot.
pub fn select_renderer(gpu_available: bool) -> Box<dyn Renderer> {
    if gpu_available {
        Box::new(GpuRenderer::default())
    } else {
        Box::new(FramebufferRenderer::default())
    }
}

// ──────────────────────────── layout + widgets ────────────────────────────

/// Main-axis sizing of a child within its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    /// A fixed extent in pixels.
    Fixed(i32),
    /// A flexible share of the remaining space (weight).
    Flex(u32),
}

/// Layout direction of a container's children.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Row,
    Column,
}

/// The emphasis of a button — selects which theme token paints it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonVariant {
    /// Filled primary action.
    Primary,
    /// Outlined, on a surface.
    Secondary,
    /// Text-only, no fill.
    Ghost,
    /// Destructive action.
    Danger,
}

/// The widget tree. Each node carries a stable `id` (for hit-testing / event
/// routing). Containers lay their children out along an axis with padding.
#[derive(Clone, Debug)]
pub enum Widget {
    Container { id: u32, axis: Axis, padding: i32, size: Size, children: Vec<Widget> },
    Label { id: u32, text: String, size: Size },
    Button { id: u32, text: String, variant: ButtonVariant, size: Size },
    /// A text input field (shows `text`, or `placeholder` in muted when empty).
    Input { id: u32, text: String, placeholder: String, size: Size },
    /// A thin separator line; its `size` is its thickness along the parent axis.
    Divider { id: u32, size: Size },
}

impl Widget {
    pub fn id(&self) -> u32 {
        match self {
            Widget::Container { id, .. }
            | Widget::Label { id, .. }
            | Widget::Button { id, .. }
            | Widget::Input { id, .. }
            | Widget::Divider { id, .. } => *id,
        }
    }
    fn size(&self) -> Size {
        match self {
            Widget::Container { size, .. }
            | Widget::Label { size, .. }
            | Widget::Button { size, .. }
            | Widget::Input { size, .. }
            | Widget::Divider { size, .. } => *size,
        }
    }
}

/// A computed placement: a widget id and the rect it occupies.
pub type Placement = (u32, Rect);

/// Lay a widget tree into `area`, returning a placement per node (parents before
/// children, so the list reads top-down). A flex/constraint solver: fixed children
/// take their size, the remainder is split by flex weight along the axis.
pub fn layout(root: &Widget, area: Rect) -> Vec<Placement> {
    let mut out = Vec::new();
    layout_into(root, area, &mut out);
    out
}

fn layout_into(node: &Widget, area: Rect, out: &mut Vec<Placement>) {
    out.push((node.id(), area));
    if let Widget::Container { axis, padding, children, .. } = node {
        let inner = area.inset(*padding);
        if children.is_empty() {
            return;
        }
        // Main-axis extent and how much is already claimed by fixed children.
        let main_total = match axis {
            Axis::Row => inner.w,
            Axis::Column => inner.h,
        };
        let mut fixed_used = 0;
        let mut flex_sum = 0u32;
        for c in children {
            match c.size() {
                Size::Fixed(px) => fixed_used += px,
                Size::Flex(w) => flex_sum += w,
            }
        }
        let free = (main_total - fixed_used).max(0);
        // Walk children, assigning main-axis offsets.
        let mut cursor = match axis {
            Axis::Row => inner.x,
            Axis::Column => inner.y,
        };
        let mut remaining_flex = flex_sum;
        let mut remaining_free = free;
        for c in children {
            let extent = match c.size() {
                Size::Fixed(px) => px,
                Size::Flex(w) => {
                    // Distribute remaining free space proportionally; the last flex
                    // child absorbs rounding so the row/column fills exactly.
                    if remaining_flex == 0 {
                        0
                    } else if w == remaining_flex {
                        remaining_free
                    } else {
                        let share = remaining_free * w as i32 / remaining_flex as i32;
                        remaining_flex -= w;
                        remaining_free -= share;
                        share
                    }
                }
            };
            let child_area = match axis {
                Axis::Row => Rect::new(cursor, inner.y, extent, inner.h),
                Axis::Column => Rect::new(inner.x, cursor, inner.w, extent),
            };
            cursor += extent;
            layout_into(c, child_area, out);
        }
    }
}

/// Build the backend-agnostic scene for a widget tree, painting each node with the
/// appropriate **theme token**. The returned `DrawCmd` list is identical regardless
/// of which renderer will rasterise it.
pub fn build_scene(root: &Widget, theme: &Theme, area: Rect) -> Vec<DrawCmd> {
    let placements = layout(root, area);
    let mut scene = Vec::new();
    paint(root, theme, &placements, &mut scene);
    scene
}

fn rect_of(id: u32, placements: &[Placement]) -> Rect {
    placements.iter().find(|(pid, _)| *pid == id).map(|(_, r)| *r).unwrap_or(Rect::new(0, 0, 0, 0))
}

fn paint(node: &Widget, theme: &Theme, placements: &[Placement], scene: &mut Vec<DrawCmd>) {
    let rect = rect_of(node.id(), placements);
    match node {
        Widget::Container { children, .. } => {
            scene.push(DrawCmd::Rect { rect, color: theme.surface, radius: theme.radius });
            for c in children {
                paint(c, theme, placements, scene);
            }
        }
        Widget::Label { text, .. } => {
            scene.push(DrawCmd::Text { rect, text: text.clone(), color: theme.text, size: theme.font_size });
        }
        Widget::Button { text, variant, .. } => {
            let (fill, fg) = match variant {
                ButtonVariant::Primary => (theme.primary, theme.on_primary),
                ButtonVariant::Secondary => (theme.surface, theme.text),
                ButtonVariant::Ghost => (Color::rgba(0, 0, 0, 0), theme.text),
                ButtonVariant::Danger => (theme.danger, theme.on_primary),
            };
            scene.push(DrawCmd::Rect { rect, color: fill, radius: theme.radius });
            scene.push(DrawCmd::Text { rect, text: text.clone(), color: fg, size: theme.font_size });
        }
        Widget::Input { text, placeholder, .. } => {
            scene.push(DrawCmd::Rect { rect, color: theme.surface, radius: theme.radius });
            let (shown, color) = if text.is_empty() {
                (placeholder.clone(), theme.muted)
            } else {
                (text.clone(), theme.text)
            };
            scene.push(DrawCmd::Text { rect, text: shown, color, size: theme.font_size });
        }
        Widget::Divider { .. } => {
            scene.push(DrawCmd::Rect { rect, color: theme.muted, radius: 0 });
        }
    }
}

// ──────────────────────────── builder helpers ────────────────────────────
//
// Ergonomic constructors so views read declaratively. Components like tabs, lists,
// command palettes and sheets are *compositions* of the primitives above — there is
// no privileged widget, exactly as the design system intends.

/// A flexible label.
pub fn label(id: u32, text: &str) -> Widget {
    Widget::Label { id, text: text.into(), size: Size::Flex(1) }
}

/// A primary button.
pub fn button(id: u32, text: &str) -> Widget {
    Widget::Button { id, text: text.into(), variant: ButtonVariant::Primary, size: Size::Flex(1) }
}

/// A button with an explicit variant.
pub fn button_variant(id: u32, text: &str, variant: ButtonVariant) -> Widget {
    Widget::Button { id, text: text.into(), variant, size: Size::Flex(1) }
}

/// A text input field.
pub fn input(id: u32, text: &str, placeholder: &str) -> Widget {
    Widget::Input { id, text: text.into(), placeholder: placeholder.into(), size: Size::Flex(1) }
}

/// A horizontal divider of `thickness` px.
pub fn divider(id: u32, thickness: i32) -> Widget {
    Widget::Divider { id, size: Size::Fixed(thickness) }
}

/// A column container.
pub fn column(id: u32, children: Vec<Widget>) -> Widget {
    Widget::Container { id, axis: Axis::Column, padding: 0, size: Size::Flex(1), children }
}

/// A row container.
pub fn row(id: u32, children: Vec<Widget>) -> Widget {
    Widget::Container { id, axis: Axis::Row, padding: 0, size: Size::Flex(1), children }
}

/// A tab bar: a fixed-height row of buttons, the active one Primary and the rest
/// Ghost. `base_id` is the id of the first tab; tab *i* has id `base_id + i`.
pub fn tabs(container_id: u32, base_id: u32, labels: &[&str], active: usize) -> Widget {
    let children = labels
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let v = if i == active { ButtonVariant::Primary } else { ButtonVariant::Ghost };
            Widget::Button { id: base_id + i as u32, text: (*t).into(), variant: v, size: Size::Flex(1) }
        })
        .collect();
    Widget::Container { id: container_id, axis: Axis::Row, padding: 0, size: Size::Fixed(34), children }
}

/// A vertical list: each item a fixed-height ghost button (selectable row).
pub fn list(container_id: u32, base_id: u32, items: &[&str], row_h: i32) -> Widget {
    let children = items
        .iter()
        .enumerate()
        .map(|(i, t)| Widget::Button {
            id: base_id + i as u32,
            text: (*t).into(),
            variant: ButtonVariant::Ghost,
            size: Size::Fixed(row_h),
        })
        .collect();
    Widget::Container { id: container_id, axis: Axis::Column, padding: 0, size: Size::Flex(1), children }
}

/// A command palette: a search input over a list of results — the launcher/"start".
pub fn command_palette(id: u32, query: &str, results: &[&str]) -> Widget {
    let mut children = alloc::vec![Widget::Input {
        id: id + 1,
        text: query.into(),
        placeholder: "what do you want to do?".into(),
        size: Size::Fixed(40),
    }];
    children.push(divider(id + 2, 1));
    children.push(list(id + 3, id + 100, results, 32));
    Widget::Container { id, axis: Axis::Column, padding: 8, size: Size::Flex(1), children }
}

/// Hit-test a point against a layout, returning the **topmost** (deepest) widget id
/// under it — the target for a click/tap. Because `layout` lists parents before
/// children, the last match is the deepest.
pub fn hit_test(placements: &[Placement], px: i32, py: i32) -> Option<u32> {
    placements.iter().rev().find(|(_, r)| r.contains(px, py)).map(|(id, _)| *id)
}

/// Software rasteriser: paint a scene's **Rect** commands into a caller-provided
/// `0x00RRGGBB` pixel buffer. This is the framebuffer-fallback path made concrete and
/// portable — the kernel's back-buffer *is* a `&mut [u32]`, so it rasterises the UI
/// by handing that slice here. Text commands need a font, so they are **returned**
/// for the caller to render (the kernel uses its 8×8 font); everything visual and
/// structural (panels, buttons, dividers, rounded corners) is done here. Pure, safe,
/// host- and on-metal-testable.
pub mod raster {
    use super::{Color, DrawCmd};
    use alloc::vec::Vec;

    /// Pack a toolkit colour into the framebuffer's `0x00RRGGBB`.
    pub fn pack(c: Color) -> u32 {
        (c.r as u32) << 16 | (c.g as u32) << 8 | c.b as u32
    }

    /// A target buffer with its dimensions and a clip rectangle (the **damage
    /// region**): nothing is written outside `[clip_x0,clip_x1) × [clip_y0,clip_y1)`.
    /// For a full repaint the clip is the whole buffer; for an incremental repaint it
    /// is the bounding box of what changed, so only those pixels are touched.
    struct Canvas<'a> {
        buf: &'a mut [u32],
        width: usize,
        height: usize,
        clip_x0: i32,
        clip_y0: i32,
        clip_x1: i32,
        clip_y1: i32,
    }

    impl Canvas<'_> {
        /// Alpha-blend `color` over the existing pixel (`a/255` coverage). Opaque
        /// colours overwrite; translucent ones produce smooth glass panels and
        /// anti-aliased edges.
        #[inline]
        fn blend(&mut self, x: i32, y: i32, color: Color) {
            if x < self.clip_x0 || y < self.clip_y0 || x >= self.clip_x1 || y >= self.clip_y1 || color.a == 0 {
                return;
            }
            if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
                return;
            }
            let idx = y as usize * self.width + x as usize;
            if idx >= self.buf.len() {
                return;
            }
            if color.a == 255 {
                self.buf[idx] = pack(color);
                return;
            }
            let dst = self.buf[idx];
            let (dr, dg, db) = ((dst >> 16) & 0xff, (dst >> 8) & 0xff, dst & 0xff);
            let a = color.a as u32;
            let ia = 255 - a;
            let r = (color.r as u32 * a + dr * ia) / 255;
            let g = (color.g as u32 * a + dg * ia) / 255;
            let b = (color.b as u32 * a + db * ia) / 255;
            self.buf[idx] = (r << 16) | (g << 8) | b;
        }

        /// A filled circle of radius `r` centred at `(cx,cy)`, anti-aliased at the rim.
        fn disc(&mut self, cx: i32, cy: i32, r: i32, color: Color) {
            if r <= 0 {
                self.blend(cx, cy, color);
                return;
            }
            let r2 = r * r;
            let ro2 = (r + 1) * (r + 1);
            for dy in -r - 1..=r + 1 {
                for dx in -r - 1..=r + 1 {
                    let d2 = dx * dx + dy * dy;
                    if d2 <= r2 {
                        self.blend(cx + dx, cy + dy, color);
                    } else if d2 <= ro2 {
                        // One-pixel feathered rim for a smooth edge.
                        let mut c = color;
                        c.a = (color.a as u32 * 2 / 5) as u8;
                        self.blend(cx + dx, cy + dy, c);
                    }
                }
            }
        }

        /// A line of pixel `width` from `(x0,y0)` to `(x1,y1)` (Bresenham; each step
        /// stamps a small disc so thick lines have round caps/joins).
        fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, width: i32, color: Color) {
            let w = width.max(1);
            let (mut x0, mut y0) = (x0, y0);
            let dx = (x1 - x0).abs();
            let dy = -(y1 - y0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let sy = if y0 < y1 { 1 } else { -1 };
            let mut err = dx + dy;
            let half = w / 2;
            loop {
                if w <= 1 {
                    self.blend(x0, y0, color);
                } else {
                    self.disc(x0, y0, half, color);
                }
                if x0 == x1 && y0 == y1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    x0 += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        /// A cubic Bézier sampled into short line segments — the node-editor wire.
        #[allow(clippy::too_many_arguments)]
        fn bezier(&mut self, p0: (i32, i32), c0: (i32, i32), c1: (i32, i32), p1: (i32, i32), width: i32, color: Color) {
            // Adaptive-ish sample count from the bounding extent.
            let span = (p1.0 - p0.0).abs() + (p1.1 - p0.1).abs() + (c0.0 - c1.0).abs();
            let steps = (span / 8).clamp(12, 96);
            let mut prev = p0;
            for i in 1..=steps {
                let t = i as i64 * 1000 / steps as i64; // 0..1000
                let pt = cubic(p0, c0, c1, p1, t);
                self.line(prev.0, prev.1, pt.0, pt.1, width, color);
                prev = pt;
            }
        }

        fn rounded(&mut self, rect: super::Rect, radius: i32, color: Color) {
            let (rx, ry, rw, rh) = (rect.x, rect.y, rect.w, rect.h);
            if rw <= 0 || rh <= 0 {
                return;
            }
            let r = radius.min(rw / 2).min(rh / 2).max(0);
            // Fast path: an opaque, square-cornered rectangle (the wallpaper and most
            // panels) is a tight row fill with the bounds checked once — no per-pixel
            // alpha math — which is the bulk of a frame's pixels.
            if r == 0 && color.a == 255 {
                let rgb = pack(color);
                let x0 = rx.max(self.clip_x0).max(0) as usize;
                let x1 = ((rx + rw).min(self.clip_x1).max(0) as usize).min(self.width);
                let y0 = ry.max(self.clip_y0).max(0) as usize;
                let y1 = ((ry + rh).min(self.clip_y1).max(0) as usize).min(self.height);
                if x1 <= x0 || y1 <= y0 {
                    return;
                }
                for yy in y0..y1 {
                    let row = yy * self.width;
                    for px in &mut self.buf[row + x0..row + x1] {
                        *px = rgb;
                    }
                }
                return;
            }
            for py in 0..rh {
                for px in 0..rw {
                    if r > 0 {
                        let in_left = px < r;
                        let in_right = px >= rw - r;
                        let in_top = py < r;
                        let in_bot = py >= rh - r;
                        if (in_left || in_right) && (in_top || in_bot) {
                            let cx = if in_left { r - 1 } else { rw - r };
                            let cy = if in_top { r - 1 } else { rh - r };
                            let dx = px - cx;
                            let dy = py - cy;
                            if dx * dx + dy * dy > r * r {
                                continue;
                            }
                        }
                    }
                    self.blend(rx + px, ry + py, color);
                }
            }
        }
    }

    /// Cubic Bézier point at parameter `t ∈ [0,1000]` (integer math).
    fn cubic(p0: (i32, i32), c0: (i32, i32), c1: (i32, i32), p1: (i32, i32), t: i64) -> (i32, i32) {
        let u = 1000 - t;
        let w0 = u * u * u;
        let w1 = 3 * u * u * t;
        let w2 = 3 * u * t * t;
        let w3 = t * t * t;
        let den = 1_000_000_000i64;
        let x = (w0 * p0.0 as i64 + w1 * c0.0 as i64 + w2 * c1.0 as i64 + w3 * p1.0 as i64) / den;
        let y = (w0 * p0.1 as i64 + w1 * c0.1 as i64 + w2 * c1.1 as i64 + w3 * p1.1 as i64) / den;
        (x as i32, y as i32)
    }

    /// Rasterise the scene's visual primitives (rects, lines, Béziers, discs,
    /// polylines) into `buf`; return the text commands (in order) for the caller to
    /// render with its own font.
    pub fn render<'a>(scene: &'a [DrawCmd], buf: &mut [u32], width: usize, height: usize) -> Vec<&'a DrawCmd> {
        render_clipped(scene, buf, width, height, (0, 0, width as i32, height as i32))
    }

    /// Rasterise the scene but write **only within the clip rectangle** `(x,y,w,h)` —
    /// the damage-region fast path for incremental repaints. Returns the text commands
    /// whose rect intersects the clip (so the caller skips off-region text too).
    pub fn render_clipped<'a>(
        scene: &'a [DrawCmd],
        buf: &mut [u32],
        width: usize,
        height: usize,
        clip: (i32, i32, i32, i32),
    ) -> Vec<&'a DrawCmd> {
        let (cx, cy, cw, ch) = clip;
        let mut canvas = Canvas {
            buf,
            width,
            height,
            clip_x0: cx.max(0),
            clip_y0: cy.max(0),
            clip_x1: (cx + cw).min(width as i32),
            clip_y1: (cy + ch).min(height as i32),
        };
        // A command whose bounding box misses the clip rectangle does no visible work,
        // so skip it **before** running its (often long) per-pixel inner loop — the
        // incremental-repaint fast path. Output is identical; only off-damage primitives
        // are dropped. `w` is the line/stroke width, padded into the box.
        let (bx0, by0, bx1, by1) = (canvas.clip_x0, canvas.clip_y0, canvas.clip_x1, canvas.clip_y1);
        let outside = |x0: i32, y0: i32, x1: i32, y1: i32| -> bool { x1 <= bx0 || x0 >= bx1 || y1 <= by0 || y0 >= by1 };
        let mut texts = Vec::new();
        for cmd in scene {
            match cmd {
                DrawCmd::Rect { rect, color, radius } => {
                    if color.a == 0 || outside(rect.x, rect.y, rect.x + rect.w, rect.y + rect.h) {
                        continue;
                    }
                    canvas.rounded(*rect, *radius, *color);
                }
                DrawCmd::Line { x0, y0, x1, y1, color, width: w } => {
                    let pad = (*w / 2) + 1;
                    if outside(*x0.min(x1) - pad, *y0.min(y1) - pad, *x0.max(x1) + pad, *y0.max(y1) + pad) {
                        continue;
                    }
                    canvas.line(*x0, *y0, *x1, *y1, *w, *color);
                }
                DrawCmd::Bezier { p0, c0, c1, p1, color, width: w } => {
                    let pad = (*w / 2) + 1;
                    let xs = [p0.0, c0.0, c1.0, p1.0];
                    let ys = [p0.1, c0.1, c1.1, p1.1];
                    let (x0, x1) = (xs.iter().copied().min().unwrap() - pad, xs.iter().copied().max().unwrap() + pad);
                    let (y0, y1) = (ys.iter().copied().min().unwrap() - pad, ys.iter().copied().max().unwrap() + pad);
                    if outside(x0, y0, x1, y1) {
                        continue;
                    }
                    canvas.bezier(*p0, *c0, *c1, *p1, *w, *color);
                }
                DrawCmd::Disc { cx, cy, r, color } => {
                    if outside(*cx - *r - 1, *cy - *r - 1, *cx + *r + 1, *cy + *r + 1) {
                        continue;
                    }
                    canvas.disc(*cx, *cy, *r, *color);
                }
                DrawCmd::Polyline { points, color, width: w } => {
                    let pad = (*w / 2) + 1;
                    for seg in points.windows(2) {
                        let (a, b) = (seg[0], seg[1]);
                        if outside(a.0.min(b.0) - pad, a.1.min(b.1) - pad, a.0.max(b.0) + pad, a.1.max(b.1) + pad) {
                            continue;
                        }
                        canvas.line(a.0, a.1, b.0, b.1, *w, *color);
                    }
                }
                DrawCmd::Text { rect, .. } => {
                    // Skip text fully outside the damage region. A generous vertical
                    // margin keeps glyphs that are centred within a taller rect.
                    let intersects = rect.x < canvas.clip_x1
                        && rect.x + rect.w > canvas.clip_x0
                        && rect.y < canvas.clip_y1
                        && rect.y + rect.h > canvas.clip_y0;
                    if intersects {
                        texts.push(cmd);
                    }
                }
                // 3D / GPU commands are passed through to the GPU compositor; the 2D
                // software rasteriser cannot render them and skips them here.
                DrawCmd::Mesh3D { .. }
                | DrawCmd::VectorPath { .. }
                | DrawCmd::GpuText { .. }
                | DrawCmd::MediaBuffer { .. }
                | DrawCmd::Scene3D { .. }
                | DrawCmd::Particles { .. }
                | DrawCmd::SdfShadow { .. } => {}
            }
        }
        texts
    }

    /// Fill a whole buffer with one colour (e.g. the wallpaper before a frame).
    pub fn clear(buf: &mut [u32], color: Color) {
        let rgb = pack(color);
        for px in buf.iter_mut() {
            *px = rgb;
        }
    }
}

// ──────────────────────────── primitive builders ────────────────────────────

/// A line draw command.
pub fn line(x0: i32, y0: i32, x1: i32, y1: i32, color: Color, width: i32) -> DrawCmd {
    DrawCmd::Line { x0, y0, x1, y1, color, width }
}

/// A cubic-Bézier wire between two points, with horizontal control handles of
/// length `slack` (the classic node-editor S-curve).
pub fn wire(from: (i32, i32), to: (i32, i32), color: Color, width: i32, slack: i32) -> DrawCmd {
    DrawCmd::Bezier {
        p0: from,
        c0: (from.0 + slack, from.1),
        c1: (to.0 - slack, to.1),
        p1: to,
        color,
        width,
    }
}

/// A filled circle.
pub fn disc(cx: i32, cy: i32, r: i32, color: Color) -> DrawCmd {
    DrawCmd::Disc { cx, cy, r, color }
}

/// A chart polyline through `points`.
pub fn polyline(points: Vec<(i32, i32)>, color: Color, width: i32) -> DrawCmd {
    DrawCmd::Polyline { points, color, width }
}

/// The smallest rectangle covering both `a` and `b` — used to accumulate a damage region.
pub fn union(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// Grow a rect by `m` pixels on every side (e.g. to cover a window's border/shadow ring).
pub fn inflate(r: Rect, m: i32) -> Rect {
    Rect::new(r.x - m, r.y - m, r.w + 2 * m, r.h + 2 * m)
}

/// Truncate `text` to at most `max` characters, appending an ellipsis when cut. Used
/// everywhere a label, filename or value must fit a fixed-width cell instead of being
/// silently clipped mid-glyph.
pub fn ellipsize(text: &str, max: usize) -> String {
    let n = text.chars().count();
    if n <= max || max == 0 {
        return text.into();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Truncate `text` to fit `width_px` at the mono `size`, with an ellipsis. A pixel-aware
/// companion to [`ellipsize`] for cells whose width is known in pixels.
pub fn ellipsize_px(text: &str, width_px: i32, size: i32) -> String {
    let cw = mono_advance(size).max(1);
    let max = (width_px / cw).max(0) as usize;
    ellipsize(text, max)
}

/// Intersection of two rects (zero-sized if they don't overlap).
pub fn intersect(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    Rect::new(x0, y0, (x1 - x0).max(0), (y1 - y0).max(0))
}

/// **Clip a scene to `bounds`** so a window's content cannot bleed past its frame.
/// `Rect` and `Text` commands are intersected with `bounds` (and dropped when they fall
/// fully outside); vector primitives (lines, discs, béziers, polylines) — which apps
/// already draw inside their own area — are kept if their bounding point set is not
/// trivially outside. This is the cheap structural clip; the rasteriser still applies
/// the pixel-exact damage clip on top.
pub fn clip_scene(scene: Vec<DrawCmd>, bounds: Rect) -> Vec<DrawCmd> {
    let mut out = Vec::with_capacity(scene.len());
    for cmd in scene {
        match cmd {
            DrawCmd::Rect { rect, color, radius } => {
                let r = intersect(rect, bounds);
                if r.w > 0 && r.h > 0 {
                    // Preserve the original radius only when the rect wasn't trimmed,
                    // so a clipped corner doesn't round mid-edge.
                    let rad = if r == rect { radius } else { 0 };
                    out.push(DrawCmd::Rect { rect: r, color, radius: rad });
                }
            }
            DrawCmd::Text { rect, text, color, size } => {
                // Keep text whose rect overlaps bounds; the rasteriser clips the glyphs.
                if intersect(rect, bounds).w > 0 && intersect(rect, bounds).h > 0 {
                    out.push(DrawCmd::Text { rect, text, color, size });
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Translate every coordinate in a scene by `(dx, dy)`. The shell renders each page in
/// its own local coordinates (origin `0,0`) and then translates the page's scene into
/// its on-screen content area with this — so pages are position-agnostic and reusable.
pub fn translate_scene(scene: &mut [DrawCmd], dx: i32, dy: i32) {
    for cmd in scene.iter_mut() {
        match cmd {
            DrawCmd::Rect { rect, .. } | DrawCmd::Text { rect, .. } => {
                rect.x += dx;
                rect.y += dy;
            }
            DrawCmd::Line { x0, y0, x1, y1, .. } => {
                *x0 += dx;
                *y0 += dy;
                *x1 += dx;
                *y1 += dy;
            }
            DrawCmd::Bezier { p0, c0, c1, p1, .. } => {
                for p in [p0, c0, c1, p1] {
                    p.0 += dx;
                    p.1 += dy;
                }
            }
            DrawCmd::Disc { cx, cy, .. } => {
                *cx += dx;
                *cy += dy;
            }
            DrawCmd::Polyline { points, .. } => {
                for p in points.iter_mut() {
                    p.0 += dx;
                    p.1 += dy;
                }
            }
            // 3D / GPU commands carry no 2D coordinates to translate.
            DrawCmd::Mesh3D { .. }
            | DrawCmd::VectorPath { .. }
            | DrawCmd::GpuText { .. }
            | DrawCmd::MediaBuffer { .. }
            | DrawCmd::Scene3D { .. }
            | DrawCmd::Particles { .. }
            | DrawCmd::SdfShadow { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn theme_tokens_are_coherent() {
        let dark = Theme::dark();
        let light = Theme::light();
        // Dark bg is darker than light bg; text contrasts its bg in both.
        assert!(dark.bg.luminance_milli() < light.bg.luminance_milli());
        assert!(dark.text.luminance_milli() > dark.bg.luminance_milli());
        assert!(light.text.luminance_milli() < light.bg.luminance_milli());
        // Spacing scale multiplies.
        assert_eq!(dark.space_n(3), 24);
    }

    #[test]
    fn renderer_selection_prefers_gpu_then_falls_back() {
        assert_eq!(select_renderer(true).backend(), Backend::Gpu);
        assert_eq!(select_renderer(false).backend(), Backend::Framebuffer);
    }

    #[test]
    fn scene_is_identical_across_backends() {
        let theme = Theme::dark();
        let ui = Widget::Button { id: 1, text: "OK".into(), variant: ButtonVariant::Primary, size: Size::Flex(1) };
        let scene = build_scene(&ui, &theme, Rect::new(0, 0, 100, 40));
        // The UI is written once; both backends receive the same scene.
        let mut gpu = GpuRenderer::default();
        let mut fb = FramebufferRenderer::default();
        gpu.submit(&scene);
        fb.submit(&scene);
        assert_eq!(gpu.commands(), fb.commands());
        assert_ne!(gpu.backend(), fb.backend());
    }

    #[test]
    fn column_layout_distributes_flex_space() {
        // Three flex:1 children split 300px height evenly inside a 0-padding column.
        let ui = Widget::Container {
            id: 0,
            axis: Axis::Column,
            padding: 0,
            size: Size::Flex(1),
            children: vec![
                Widget::Label { id: 1, text: "a".into(), size: Size::Flex(1) },
                Widget::Label { id: 2, text: "b".into(), size: Size::Flex(1) },
                Widget::Label { id: 3, text: "c".into(), size: Size::Flex(1) },
            ],
        };
        let p = layout(&ui, Rect::new(0, 0, 200, 300));
        assert_eq!(rect_of(1, &p), Rect::new(0, 0, 200, 100));
        assert_eq!(rect_of(2, &p), Rect::new(0, 100, 200, 100));
        assert_eq!(rect_of(3, &p), Rect::new(0, 200, 200, 100));
    }

    #[test]
    fn row_layout_mixes_fixed_and_flex_with_padding() {
        // A 8px-padded row: a fixed 40px sidebar + a flex main area.
        let ui = Widget::Container {
            id: 0,
            axis: Axis::Row,
            padding: 8,
            size: Size::Flex(1),
            children: vec![
                Widget::Container { id: 1, axis: Axis::Column, padding: 0, size: Size::Fixed(40), children: vec![] },
                Widget::Container { id: 2, axis: Axis::Column, padding: 0, size: Size::Flex(1), children: vec![] },
            ],
        };
        let p = layout(&ui, Rect::new(0, 0, 200, 100));
        // Inner area after padding is (8,8, 184,84).
        assert_eq!(rect_of(1, &p), Rect::new(8, 8, 40, 84)); // fixed sidebar
        assert_eq!(rect_of(2, &p), Rect::new(48, 8, 144, 84)); // flex fills the rest
    }

    #[test]
    fn flex_weights_split_proportionally_and_fill_exactly() {
        let ui = Widget::Container {
            id: 0,
            axis: Axis::Row,
            padding: 0,
            size: Size::Flex(1),
            children: vec![
                Widget::Label { id: 1, text: "1".into(), size: Size::Flex(1) },
                Widget::Label { id: 2, text: "2".into(), size: Size::Flex(2) },
            ],
        };
        let p = layout(&ui, Rect::new(0, 0, 99, 10));
        // 1:2 split of 99 → 33 / 66, filling exactly (no lost pixel).
        assert_eq!(rect_of(1, &p).w, 33);
        assert_eq!(rect_of(2, &p).w, 66);
        assert_eq!(rect_of(1, &p).w + rect_of(2, &p).w, 99);
    }

    #[test]
    fn button_paints_with_primary_token() {
        let theme = Theme::dark();
        let ui = Widget::Button { id: 1, text: "Go".into(), variant: ButtonVariant::Primary, size: Size::Flex(1) };
        let scene = build_scene(&ui, &theme, Rect::new(0, 0, 80, 30));
        // First command is the button surface in the primary colour.
        match &scene[0] {
            DrawCmd::Rect { color, .. } => assert_eq!(*color, theme.primary),
            other => panic!("expected a rect, got {other:?}"),
        }
        // Followed by its label in on_primary.
        assert!(matches!(&scene[1], DrawCmd::Text { color, .. } if *color == theme.on_primary));
    }

    #[test]
    fn hit_test_returns_the_deepest_widget() {
        let ui = Widget::Container {
            id: 0,
            axis: Axis::Row,
            padding: 0,
            size: Size::Flex(1),
            children: vec![
                Widget::Button { id: 1, text: "L".into(), variant: ButtonVariant::Primary, size: Size::Flex(1) },
                Widget::Button { id: 2, text: "R".into(), variant: ButtonVariant::Primary, size: Size::Flex(1) },
            ],
        };
        let p = layout(&ui, Rect::new(0, 0, 200, 50));
        // Left half hits button 1, right half hits button 2 (not the container).
        assert_eq!(hit_test(&p, 50, 25), Some(1));
        assert_eq!(hit_test(&p, 150, 25), Some(2));
        // Outside everything → nothing.
        assert_eq!(hit_test(&p, 500, 500), None);
    }

    #[test]
    fn button_variants_paint_distinct_tokens() {
        let t = Theme::dark();
        let prim = build_scene(&button_variant(1, "P", ButtonVariant::Primary), &t, Rect::new(0, 0, 50, 20));
        let danger = build_scene(&button_variant(1, "D", ButtonVariant::Danger), &t, Rect::new(0, 0, 50, 20));
        let ghost = build_scene(&button_variant(1, "G", ButtonVariant::Ghost), &t, Rect::new(0, 0, 50, 20));
        match (&prim[0], &danger[0], &ghost[0]) {
            (DrawCmd::Rect { color: p, .. }, DrawCmd::Rect { color: d, .. }, DrawCmd::Rect { color: g, .. }) => {
                assert_eq!(*p, t.primary);
                assert_eq!(*d, t.danger);
                assert_eq!(g.a, 0); // ghost has a transparent fill
            }
            _ => panic!("expected rects"),
        }
    }

    #[test]
    fn input_shows_placeholder_when_empty() {
        let t = Theme::dark();
        let empty = build_scene(&input(1, "", "search…"), &t, Rect::new(0, 0, 100, 24));
        let filled = build_scene(&input(1, "hello", "search…"), &t, Rect::new(0, 0, 100, 24));
        // Empty → muted placeholder text; filled → normal text.
        assert!(matches!(&empty[1], DrawCmd::Text { text, color, .. } if text == "search…" && *color == t.muted));
        assert!(matches!(&filled[1], DrawCmd::Text { text, color, .. } if text == "hello" && *color == t.text));
    }

    #[test]
    fn tabs_highlight_the_active_tab() {
        let t = Theme::dark();
        let w = tabs(10, 20, &["Editor", "Browser", "Files"], 1);
        let scene = build_scene(&w, &t, Rect::new(0, 0, 300, 34));
        // Find the three tab-button fills: active (idx 1) is primary, others ghost.
        let fills: Vec<&Color> = scene
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Rect { color, .. } => Some(color),
                _ => None,
            })
            .collect();
        // [container surface, tab0 ghost, tab1 primary, tab2 ghost]
        assert_eq!(*fills[2], t.primary);
        assert_eq!(fills[1].a, 0);
        assert_eq!(fills[3].a, 0);
    }

    #[test]
    fn command_palette_composes_input_plus_results() {
        let t = Theme::dark();
        let w = command_palette(1, "inv", &["Open Sales", "Find invoices", "New note"]);
        let p = layout(&w, Rect::new(0, 0, 400, 300));
        // The palette lays out its input, a divider, and a result list — clicking a
        // result row routes to a real widget id.
        let _scene = build_scene(&w, &t, Rect::new(0, 0, 400, 300));
        // First result row id is base 101 (id+100) — hit-test the top of the list.
        let hit = hit_test(&p, 200, 60);
        assert!(hit.is_some());
    }

    #[test]
    fn divider_is_a_thin_muted_line() {
        let t = Theme::dark();
        let scene = build_scene(&divider(1, 2), &t, Rect::new(0, 10, 100, 2));
        assert!(matches!(&scene[0], DrawCmd::Rect { color, radius: 0, .. } if *color == t.muted));
    }

    #[test]
    fn rasteriser_fills_rects_into_a_pixel_buffer() {
        let (w, h) = (16usize, 8usize);
        let mut buf = alloc::vec![0u32; w * h];
        let red = Color::rgb(0xff, 0, 0);
        let scene = alloc::vec![DrawCmd::Rect { rect: Rect::new(2, 1, 4, 3), color: red, radius: 0 }];
        let texts = raster::render(&scene, &mut buf, w, h);
        assert!(texts.is_empty());
        // The 4×3 block is red; everything else stays background.
        assert_eq!(buf[w + 2], 0xFF0000);
        assert_eq!(buf[3 * w + 5], 0xFF0000);
        assert_eq!(buf[0], 0); // outside the rect
        assert_eq!(buf[w + 6], 0); // just past the right edge
    }

    #[test]
    fn clipped_raster_touches_only_the_damage_rect() {
        let (w, h) = (16usize, 8usize);
        let mut buf = alloc::vec![0u32; w * h];
        let red = Color::rgb(0xff, 0, 0);
        // A rect spanning the whole buffer, but clipped to a 4×3 damage window.
        let scene = alloc::vec![DrawCmd::Rect { rect: Rect::new(0, 0, 16, 8), color: red, radius: 0 }];
        let texts = raster::render_clipped(&scene, &mut buf, w, h, (2, 1, 4, 3));
        assert!(texts.is_empty());
        // Inside the clip → painted.
        assert_eq!(buf[w + 2], 0xFF0000);
        assert_eq!(buf[3 * w + 5], 0xFF0000);
        // Outside the clip → untouched, even though the rect covered it.
        assert_eq!(buf[0], 0);
        assert_eq!(buf[w + 6], 0); // just past the clip's right edge
        assert_eq!(buf[4 * w + 2], 0); // just past the clip's bottom edge
    }

    #[test]
    fn culling_keeps_visible_primitives_and_drops_off_clip_ones() {
        let (w, h) = (40usize, 40usize);
        let mut buf = alloc::vec![0u32; w * h];
        let white = Color::rgb(255, 255, 255);
        let scene = alloc::vec![
            // A horizontal line crossing the clip band at y=10 — must still draw.
            DrawCmd::Line { x0: 0, y0: 10, x1: 39, y1: 10, color: white, width: 1 },
            // A disc far below the clip — culled, leaves the buffer untouched.
            disc(20, 35, 3, Color::rgb(255, 0, 0)),
        ];
        // Clip to a band that contains the line but not the disc.
        let texts = raster::render_clipped(&scene, &mut buf, w, h, (0, 5, 40, 10));
        assert!(texts.is_empty());
        assert_eq!(buf[10 * w + 20], raster::pack(white)); // line drew inside the clip
        // Nothing red anywhere (the off-clip disc was skipped).
        assert!(buf.iter().all(|&p| p >> 16 < 200 || (p & 0xffff) != 0 || (p >> 16) != 0xff));
        // The disc's centre row is outside the clip and untouched.
        assert_eq!(buf[35 * w + 20], 0);
    }

    #[test]
    fn clipped_raster_drops_text_outside_the_damage_rect() {
        let (w, h) = (40usize, 20usize);
        let mut buf = alloc::vec![0u32; w * h];
        let scene = alloc::vec![
            DrawCmd::Text { rect: Rect::new(0, 0, 8, 8, ), text: "in".into(), color: Color::rgb(255, 255, 255), size: 8 },
            DrawCmd::Text { rect: Rect::new(30, 14, 8, 8), text: "out".into(), color: Color::rgb(255, 255, 255), size: 8 },
        ];
        let texts = raster::render_clipped(&scene, &mut buf, w, h, (0, 0, 10, 10));
        // Only the text intersecting the clip is returned for the caller to draw.
        assert_eq!(texts.len(), 1);
        assert!(matches!(texts[0], DrawCmd::Text { text, .. } if text == "in"));
    }

    #[test]
    fn rasteriser_skips_transparent_and_returns_text() {
        let (w, h) = (8usize, 8usize);
        let mut buf = alloc::vec![0u32; w * h];
        let scene = alloc::vec![
            DrawCmd::Rect { rect: Rect::new(0, 0, 8, 8), color: Color::rgba(0, 0, 0, 0), radius: 0 },
            DrawCmd::Text { rect: Rect::new(0, 0, 8, 8), text: "hi".into(), color: Color::rgb(255, 255, 255), size: 8 },
        ];
        let texts = raster::render(&scene, &mut buf, w, h);
        // Transparent fill left the buffer untouched; the text is handed back to draw.
        assert!(buf.iter().all(|&p| p == 0));
        assert_eq!(texts.len(), 1);
    }

    #[test]
    fn rasteriser_rounds_corners() {
        let (w, h) = (10usize, 10usize);
        let mut buf = alloc::vec![0u32; w * h];
        let scene = alloc::vec![DrawCmd::Rect {
            rect: Rect::new(0, 0, 10, 10),
            color: Color::rgb(1, 2, 3),
            radius: 4,
        }];
        raster::render(&scene, &mut buf, w, h);
        // The very corner pixel is clipped by the radius; the centre is filled.
        assert_eq!(buf[0], 0);
        assert_eq!(buf[5 * w + 5], raster::pack(Color::rgb(1, 2, 3)));
    }

    #[test]
    fn rasteriser_draws_lines_discs_and_wires() {
        let (w, h) = (40usize, 40usize);
        let mut buf = alloc::vec![0u32; w * h];
        let white = Color::rgb(255, 255, 255);
        let scene = alloc::vec![
            DrawCmd::Line { x0: 2, y0: 20, x1: 37, y1: 20, color: white, width: 1 },
            disc(20, 10, 5, Color::rgb(255, 0, 0)),
            wire((2, 30), (37, 35), Color::rgb(0, 255, 0), 2, 12),
        ];
        let texts = raster::render(&scene, &mut buf, w, h);
        assert!(texts.is_empty());
        // The horizontal line painted the middle row.
        assert_eq!(buf[20 * w + 20], raster::pack(white));
        // The disc filled its centre.
        assert_eq!(buf[10 * w + 20], raster::pack(Color::rgb(255, 0, 0)));
        // The wire painted *something* green in the lower band.
        assert!(buf[25 * w..].iter().any(|&p| (p >> 8) & 0xff > 100 && (p & 0xff) < 80 && (p >> 16) < 80));
    }

    #[test]
    fn polyline_chart_connects_points() {
        let (w, h) = (30usize, 30usize);
        let mut buf = alloc::vec![0u32; w * h];
        let c = Color::rgb(100, 180, 255);
        let scene = alloc::vec![polyline(alloc::vec![(2, 28), (10, 10), (20, 20), (28, 2)], c, 1)];
        raster::render(&scene, &mut buf, w, h);
        // At least the vertices are painted.
        assert_eq!(buf[10 * w + 10], raster::pack(c));
    }

    #[test]
    fn alpha_blend_produces_translucent_panels() {
        let (w, h) = (4usize, 4usize);
        let mut buf = alloc::vec![raster::pack(Color::rgb(0, 0, 0)); w * h];
        // A 50%-alpha white panel over black → mid-grey.
        let scene = alloc::vec![DrawCmd::Rect {
            rect: Rect::new(0, 0, 4, 4),
            color: Color::rgba(255, 255, 255, 128),
            radius: 0,
        }];
        raster::render(&scene, &mut buf, w, h);
        let p = buf[0];
        let r = (p >> 16) & 0xff;
        assert!((120..=135).contains(&r), "expected ~50% grey, got {r}");
    }

    #[test]
    fn full_dashboard_scene_rasterises_without_panic() {
        // A realistic shell: top bar + command palette, rasterised into a screen-sized
        // buffer — the framebuffer-fallback path end to end.
        let t = Theme::dark();
        let (w, h) = (320usize, 200usize);
        let mut buf = alloc::vec![raster::pack(t.bg); w * h];
        let shell = column(
            0,
            alloc::vec![
                tabs(1, 10, &["Editor", "Browser", "Files"], 0),
                command_palette(2, "", &["Open Sales", "New note"]),
            ],
        );
        let scene = build_scene(&shell, &t, Rect::new(0, 0, w as i32, h as i32));
        let texts = raster::render(&scene, &mut buf, w, h);
        // Some surface pixels were painted, and text was collected for the font pass.
        assert!(buf.iter().any(|&p| p == raster::pack(t.surface) || p == raster::pack(t.primary)));
        assert!(!texts.is_empty());
    }
}

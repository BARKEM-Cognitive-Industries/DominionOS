//! The **Desktop** — a Windows-style home screen: a grid of **icons** on the
//! wallpaper, with **widgets** (the composable [`Board`](crate::compose)) layered on
//! top. (It used to float big 3D "card" slabs of every object — that read as a node
//! canvas, not a desktop. Now each object is a flat, labelled **desktop icon** you can
//! drag and double-click, exactly like a real desktop.)
//!
//! Each system object becomes an icon: a click opens it (programs in the IDE, data in
//! the Explorer). Icons are draggable and remember their positions. The persistent
//! dock, app-launcher icons and widget board live in the shell ([`crate::os`]); this
//! page owns the wallpaper and the object icons.
//!
//! The desktop is an **infinite canvas**: pan by dragging empty space, zoom with
//! `zoom_in`/`zoom_out`, and snap back to origin (0, 0) with `snap_to_center`. A
//! minimap in the lower-right shows a bird's-eye view of icon positions and the
//! current viewport. A crosshair marks the canvas origin.
//!
//! Pure, safe `no_std`. Rendered in **page-local coordinates** (origin `0,0`); the
//! shell translates the scene into the on-screen content area.

use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::world::Entry;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// An icon cell's footprint (tile + label).
const ICON_W: i32 = 96;
const ICON_H: i32 = 88;
/// The icon tile (the coloured square the glyph sits on).
const TILE_W: i32 = 52;
const TILE_H: i32 = 44;
/// Left margin so object icons clear the shell's app-launcher column.
const GRID_LEFT: i32 = 128;
const GRID_TOP: i32 = 16;

/// Minimap dimensions and margin from the bottom-right corner.
const MINI_W: i32 = 120;
const MINI_H: i32 = 90;
const MINI_MARGIN_R: i32 = 10;
const MINI_MARGIN_B: i32 = 10;

/// The visual family of an object icon (selects its glyph).
#[derive(Clone, Copy, PartialEq, Eq)]
enum IconKind {
    Data,
    Chart,
    Log,
    Program,
    Generic,
}

fn icon_for(kind: &str) -> IconKind {
    match kind {
        "Dataset" => IconKind::Data,
        "Report" | "Codec" => IconKind::Chart,
        "Log" | "Channel" => IconKind::Log,
        "Program" => IconKind::Program,
        _ => IconKind::Generic,
    }
}

/// A desktop icon built from a [`World`](crate::world) [`Entry`].
struct DeskIcon {
    /// Stable identity (the object name) — preserves position across re-syncs.
    key: String,
    title: String,
    icon: IconKind,
    /// Programs open in the IDE; everything else is inspected in the Explorer.
    is_program: bool,
    /// Position in canvas space (canvas origin = 0,0).
    x: i32,
    y: i32,
}

/// A text label pinned to a canvas-space position.
pub struct CanvasLabel {
    pub x: i32,
    pub y: i32,
    pub text: String,
}

/// A click on an icon: programs open in the IDE, everything else in the Explorer.
pub enum DesktopAction {
    OpenProgram(String),
    Inspect(String),
}

/// Which authoring tool is active on the canvas.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CanvasTool {
    Select,
    AddNote,
    AddBox,
}

/// The content of a canvas item.
#[derive(Clone)]
pub enum CanvasItemKind {
    Note { text: String },
    Box  { w: i32, h: i32, label: String },
}

/// A user-placed item on the infinite canvas.
#[derive(Clone)]
pub struct CanvasItem {
    pub id: u32,
    pub x: i32,
    pub y: i32,
    pub kind: CanvasItemKind,
}

/// The desktop home page — an infinite pannable/zoomable icon canvas.
pub struct Desktop {
    icons: Vec<DeskIcon>,
    /// Remembered icon positions by key, so a re-sync doesn't reset drags.
    positions: BTreeMap<String, (i32, i32)>,
    area: Rect,
    drag: Option<(String, i32, i32)>,
    press_x: i32,
    press_y: i32,
    last_left: bool,
    damage: Option<Rect>,

    // ── infinite canvas state ──

    /// Pan offset: the canvas origin is rendered at (canvas_x, canvas_y) in screen
    /// space. Start at (0,0) so the canvas origin sits at the top-left of the area.
    pub canvas_x: i32,
    pub canvas_y: i32,

    /// Zoom level in percent (50..=200, default 100).
    pub zoom: i32,

    /// Active panning gesture.
    panning: bool,
    pan_start_x: i32,
    pan_start_y: i32,
    canvas_start_x: i32,
    canvas_start_y: i32,

    /// Text labels pinned to canvas space.
    pub canvas_labels: Vec<CanvasLabel>,

    // ── canvas authoring ──

    /// Currently active authoring tool.
    pub tool: CanvasTool,
    /// User-placed canvas items.
    pub canvas_items: Vec<CanvasItem>,
    /// The id of the currently selected canvas item, if any.
    selected_id: Option<u32>,
    /// Counter for assigning unique ids to new canvas items.
    next_id: u32,
}

impl Desktop {
    pub fn new() -> Desktop {
        Desktop {
            icons: Vec::new(),
            positions: BTreeMap::new(),
            area: Rect::new(0, 0, 1280, 600),
            drag: None,
            press_x: 0,
            press_y: 0,
            last_left: false,
            damage: Some(Rect::new(0, 0, 1280, 600)),

            canvas_x: 0,
            canvas_y: 0,
            zoom: 100,

            panning: false,
            pan_start_x: 0,
            pan_start_y: 0,
            canvas_start_x: 0,
            canvas_start_y: 0,

            canvas_labels: alloc::vec![
                CanvasLabel { x: -200, y: -200, text: "Projects".to_string() },
                CanvasLabel { x: 200,  y: 100,  text: "Data".to_string() },
            ],

            tool: CanvasTool::Select,
            canvas_items: Vec::new(),
            selected_id: None,
            next_id: 1,
        }
    }

    // ── canvas helpers ──

    /// Reset pan and zoom back to origin.
    pub fn snap_to_center(&mut self) {
        self.canvas_x = 0;
        self.canvas_y = 0;
        self.zoom = 100;
        self.dmg_all();
    }

    /// Increase zoom by 10%, clamped to 200.
    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom + 10).min(200);
        self.dmg_all();
    }

    /// Decrease zoom by 10%, clamped to 50.
    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom - 10).max(50);
        self.dmg_all();
    }

    /// Convert a canvas-space coordinate to screen space.
    fn to_screen(&self, cx: i32, cy: i32) -> (i32, i32) {
        let sx = self.canvas_x + cx * self.zoom / 100;
        let sy = self.canvas_y + cy * self.zoom / 100;
        (sx, sy)
    }

    /// Convert a screen-space coordinate to canvas space.
    pub fn to_canvas(&self, sx: i32, sy: i32) -> (i32, i32) {
        let cx = (sx - self.canvas_x) * 100 / self.zoom.max(1);
        let cy = (sy - self.canvas_y) * 100 / self.zoom.max(1);
        (cx, cy)
    }

    /// The screen-space rect for an icon at its canvas position.
    fn icon_screen_rect(&self, c: &DeskIcon) -> Rect {
        let (sx, sy) = self.to_screen(c.x, c.y);
        let sw = ICON_W * self.zoom / 100;
        let sh = ICON_H * self.zoom / 100;
        Rect::new(sx, sy, sw.max(1), sh.max(1))
    }

    /// The Home button rect (screen space, top-right of desktop area).
    fn home_btn_rect(&self) -> Rect {
        Rect::new(self.area.w - 80, 8, 70, 22)
    }

    // ── canvas authoring: public API ──

    /// Set the active authoring tool.
    pub fn set_tool(&mut self, t: CanvasTool) {
        self.tool = t;
    }

    /// Return the active authoring tool.
    pub fn tool(&self) -> CanvasTool {
        self.tool
    }

    /// Return the id of the currently selected canvas item.
    pub fn selected_id(&self) -> Option<u32> {
        self.selected_id
    }

    /// Place a sticky note at canvas-space coordinates.
    pub fn add_note_at(&mut self, cx: i32, cy: i32) {
        let id = self.next_id;
        self.next_id += 1;
        self.canvas_items.push(CanvasItem {
            id,
            x: cx - 60,
            y: cy - 30,
            kind: CanvasItemKind::Note { text: "Note".to_string() },
        });
        self.selected_id = Some(id);
        self.dmg_all();
    }

    /// Place a labelled box on the canvas.
    pub fn add_box_at(&mut self, cx: i32, cy: i32, cw: i32, ch: i32) {
        let id = self.next_id;
        self.next_id += 1;
        self.canvas_items.push(CanvasItem {
            id,
            x: cx,
            y: cy,
            kind: CanvasItemKind::Box { w: cw.max(40), h: ch.max(24), label: "Area".to_string() },
        });
        self.selected_id = Some(id);
        self.dmg_all();
    }

    /// Delete the currently selected canvas item, if any.
    pub fn delete_selected(&mut self) {
        if let Some(id) = self.selected_id {
            self.canvas_items.retain(|item| item.id != id);
            self.selected_id = None;
            self.dmg_all();
        }
    }

    // ── canvas authoring: private helpers ──

    /// Compute the screen-space rect for a canvas item.
    fn item_screen_rect(&self, item: &CanvasItem) -> Rect {
        let (sx, sy) = self.to_screen(item.x, item.y);
        match &item.kind {
            CanvasItemKind::Note { .. } => {
                let sw = 120 * self.zoom / 100;
                let sh = 60 * self.zoom / 100;
                Rect::new(sx, sy, sw.max(1), sh.max(1))
            }
            CanvasItemKind::Box { w, h, .. } => {
                let sw = w * self.zoom / 100;
                let sh = h * self.zoom / 100;
                Rect::new(sx, sy, sw.max(1), sh.max(1))
            }
        }
    }

    /// Render a single canvas item.
    fn draw_canvas_item(&self, s: &mut Vec<DrawCmd>, item: &CanvasItem, t: &Theme) {
        let rect = self.item_screen_rect(item);
        let selected = self.selected_id == Some(item.id);
        match &item.kind {
            CanvasItemKind::Note { text } => {
                // Selection border (drawn first, behind the note).
                if selected {
                    s.push(DrawCmd::Rect {
                        rect: Rect::new(rect.x - 2, rect.y - 2, rect.w + 4, rect.h + 4),
                        color: t.primary,
                        radius: 8,
                    });
                }
                // Note background — warm yellow.
                s.push(DrawCmd::Rect {
                    rect,
                    color: Color::rgba(255, 235, 150, 220),
                    radius: 6,
                });
                // Grip line at top.
                s.push(toolkit::line(
                    rect.x + 4, rect.y + 6,
                    rect.x + rect.w - 4, rect.y + 6,
                    Color::rgba(180, 160, 80, 255),
                    1,
                ));
                // Text content.
                let inner = Rect::new(rect.x + 6, rect.y + 6, rect.w - 12, rect.h - 12);
                s.push(DrawCmd::Text {
                    rect: inner,
                    text: text.clone(),
                    color: Color::rgba(40, 30, 0, 255),
                    size: 12,
                });
            }
            CanvasItemKind::Box { label, .. } => {
                let border_color = if selected { t.primary } else { t.accent };
                // Draw 4 border lines (outline only, no fill).
                s.push(toolkit::line(rect.x, rect.y, rect.x + rect.w, rect.y, border_color, 1));
                s.push(toolkit::line(rect.x, rect.y + rect.h, rect.x + rect.w, rect.y + rect.h, border_color, 1));
                s.push(toolkit::line(rect.x, rect.y, rect.x, rect.y + rect.h, border_color, 1));
                s.push(toolkit::line(rect.x + rect.w, rect.y, rect.x + rect.w, rect.y + rect.h, border_color, 1));
                // Label text at top-left.
                s.push(DrawCmd::Text {
                    rect: Rect::new(rect.x + 4, rect.y + 2, rect.w - 8, 16),
                    text: label.clone(),
                    color: border_color,
                    size: 11,
                });
            }
        }
    }

    // ── world sync ──

    /// Rebuild the desktop icons from the live system objects. Idempotent: an unchanged
    /// object set leaves the icons (and their dragged positions) untouched.
    pub fn set_entries(&mut self, entries: &[Entry]) {
        let same = entries.len() == self.icons.len()
            && entries.iter().all(|e| self.icons.iter().any(|c| c.key == e.name));
        if same {
            return;
        }
        let mut icons = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            let (x, y) = self.positions.get(&e.name).copied().unwrap_or_else(|| self.default_pos(i));
            icons.push(DeskIcon {
                key: e.name.clone(),
                title: e.name.clone(),
                icon: icon_for(&e.kind),
                is_program: e.kind == "Program",
                x,
                y,
            });
        }
        self.icons = icons;
        self.dmg_all();
    }

    /// Flow icons in a grid to the right of the app-launcher column (canvas space).
    fn default_pos(&self, i: usize) -> (i32, i32) {
        let cols = ((self.area.w - GRID_LEFT) / ICON_W).max(1);
        let col = i as i32 % cols;
        let row = i as i32 / cols;
        (GRID_LEFT + col * ICON_W, GRID_TOP + row * ICON_H)
    }

    pub fn set_area(&mut self, area: Rect) {
        if area.w != self.area.w || area.h != self.area.h {
            self.area = area;
            self.dmg_all();
        } else {
            self.area = area;
        }
    }

    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }

    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }



    // ── input (page-local / screen coordinates) ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) -> Option<DesktopAction> {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;
        let mut action = None;

        if pressed {
            self.press_x = px;
            self.press_y = py;

            // Home button?
            if self.home_btn_rect().contains(px, py) {
                self.snap_to_center();
                self.last_left = left;
                return None;
            }

            // AddNote: place a note and revert to Select.
            if self.tool == CanvasTool::AddNote {
                let (cx, cy) = self.to_canvas(px, py);
                self.add_note_at(cx, cy);
                self.tool = CanvasTool::Select;
                self.last_left = left;
                return None;
            }

            // Hit-test icons in screen space.
            if let Some(idx) = self.icons.iter().rposition(|c| self.icon_screen_rect(c).contains(px, py)) {
                let c = &self.icons[idx];
                // Store the drag offset in canvas space.
                let (cx, cy) = self.to_canvas(px, py);
                self.drag = Some((c.key.clone(), cx - c.x, cy - c.y));
                let c = self.icons.remove(idx);
                self.icons.push(c); // raise to top
                self.dmg_all();
            } else {
                // Hit-test canvas items (Select tool only).
                if self.tool == CanvasTool::Select {
                    if let Some(item) = self.canvas_items.iter().rev().find(|item| self.item_screen_rect(item).contains(px, py)) {
                        self.selected_id = Some(item.id);
                        self.dmg_all();
                        self.last_left = left;
                        return None;
                    }
                    // Nothing hit — deselect.
                    self.selected_id = None;
                }
                // Start panning on empty space.
                self.panning = true;
                self.pan_start_x = px;
                self.pan_start_y = py;
                self.canvas_start_x = self.canvas_x;
                self.canvas_start_y = self.canvas_y;
            }
        } else if left {
            if let Some((key, ox, oy)) = self.drag.clone() {
                // Drag icon in canvas space — compute canvas coords before mutable borrow.
                let (cx, cy) = self.to_canvas(px, py);
                if let Some(c) = self.icons.iter_mut().find(|c| c.key == key) {
                    c.x = cx - ox;
                    c.y = cy - oy;
                    self.positions.insert(key, (c.x, c.y));
                    self.dmg_all();
                }
            } else if self.panning {
                // Pan the canvas.
                let dx = px - self.pan_start_x;
                let dy = py - self.pan_start_y;
                self.canvas_x = self.canvas_start_x + dx;
                self.canvas_y = self.canvas_start_y + dy;
                self.dmg_all();
            }
        }

        if released {
            let moved = (px - self.press_x).abs() + (py - self.press_y).abs();
            if moved < 5 {
                // Click — check icons in screen space.
                if let Some(c) = self.icons.iter().rev().find(|c| self.icon_screen_rect(c).contains(px, py)) {
                    action = Some(if c.is_program {
                        DesktopAction::OpenProgram(c.title.clone())
                    } else {
                        DesktopAction::Inspect(c.title.clone())
                    });
                }
            }
            self.drag = None;
            self.panning = false;
        }

        self.last_left = left;
        action
    }

    // ── rendering (page-local / screen coordinates) ──

    pub fn view(&self, theme: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        let (w, h) = (self.area.w, self.area.h);

        // Wallpaper.
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, w, h), color: theme.bg, radius: 0 });

        // Canvas labels.
        for lbl in &self.canvas_labels {
            let (sx, sy) = self.to_screen(lbl.x, lbl.y);
            // Only draw if roughly on-screen.
            if sx > -200 && sx < w + 200 && sy > -40 && sy < h + 40 {
                s.push(DrawCmd::Text {
                    rect: Rect::new(sx, sy, 120, 18),
                    text: lbl.text.clone(),
                    color: theme.muted,
                    size: 13,
                });
            }
        }

        // Canvas items (user-placed notes and boxes).
        for item in &self.canvas_items {
            self.draw_canvas_item(&mut s, item, theme);
        }

        // Icons.
        for c in &self.icons {
            self.draw_icon_scaled(&mut s, c, theme);
        }

        // Origin crosshair (marks canvas 0,0 in screen space).
        self.draw_crosshair(&mut s, theme);

        // Home button (top-right).
        self.draw_home_btn(&mut s, theme);

        // Minimap (lower-right).
        self.draw_minimap(&mut s, theme);

        s
    }

    /// Draw an icon at its screen-space position, scaled by zoom.
    fn draw_icon_scaled(&self, s: &mut Vec<DrawCmd>, c: &DeskIcon, t: &Theme) {
        let r = self.icon_screen_rect(c);
        let tw = TILE_W * self.zoom / 100;
        let th = TILE_H * self.zoom / 100;
        let tile = Rect::new(r.x + (r.w - tw) / 2, r.y + r.h / 20, tw.max(1), th.max(1));
        s.push(DrawCmd::Rect { rect: tile, color: t.surface, radius: t.radius });
        let cx = tile.x + tile.w / 2;
        let cy = tile.y + tile.h / 2;
        draw_glyph(s, c.icon, cx, cy, t);
        let label = ellipsize(&c.title, 12);
        s.push(DrawCmd::Text {
            rect: Rect::new(r.x - 2, r.y + th + 8, r.w + 4, 16),
            text: label,
            color: t.text,
            size: 12,
        });
    }

    /// Draw a small `+` crosshair at the canvas origin in screen space.
    fn draw_crosshair(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let ox = self.canvas_x;
        let oy = self.canvas_y;
        // Only draw if near the viewport.
        if ox < -20 || ox > self.area.w + 20 || oy < -20 || oy > self.area.h + 20 {
            return;
        }
        let arm = 5;
        // Horizontal arm.
        s.push(toolkit::line(ox - arm, oy, ox + arm, oy, t.muted, 1));
        // Vertical arm.
        s.push(toolkit::line(ox, oy - arm, ox, oy + arm, t.muted, 1));
        // Small circle at centre.
        s.push(toolkit::disc(ox, oy, 2, t.muted));
    }

    /// Draw the "⌂ Home" snap button.
    fn draw_home_btn(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let btn = self.home_btn_rect();
        s.push(DrawCmd::Rect { rect: btn, color: t.surface, radius: 4 });
        s.push(DrawCmd::Text {
            rect: btn,
            text: "\u{2302} Home".to_string(),
            color: t.text,
            size: 12,
        });
    }

    /// Draw the minimap in the lower-right corner.
    fn draw_minimap(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let mx = self.area.w - MINI_W - MINI_MARGIN_R;
        let my = self.area.h - MINI_H - MINI_MARGIN_B;

        // Background.
        s.push(DrawCmd::Rect {
            rect: Rect::new(mx, my, MINI_W, MINI_H),
            color: Color::rgba(t.bg.r, t.bg.g, t.bg.b, 200),
            radius: 4,
        });
        // Border.
        s.push(toolkit::line(mx, my, mx + MINI_W, my, t.muted, 1));
        s.push(toolkit::line(mx, my + MINI_H, mx + MINI_W, my + MINI_H, t.muted, 1));
        s.push(toolkit::line(mx, my, mx, my + MINI_H, t.muted, 1));
        s.push(toolkit::line(mx + MINI_W, my, mx + MINI_W, my + MINI_H, t.muted, 1));

        // Determine the canvas extent that maps into the minimap.
        // We show a fixed canvas window of ±800 x ±600 around the origin.
        let canvas_view_w: i32 = 1600;
        let canvas_view_h: i32 = 1200;
        let canvas_view_ox: i32 = -800;
        let canvas_view_oy: i32 = -600;

        // Helper: canvas coord -> minimap screen coord.
        let to_mini = |cx: i32, cy: i32| -> (i32, i32) {
            let rx = (cx - canvas_view_ox) * MINI_W / canvas_view_w;
            let ry = (cy - canvas_view_oy) * MINI_H / canvas_view_h;
            (mx + rx, my + ry)
        };

        // Dot for each icon.
        for c in &self.icons {
            let (dx, dy) = to_mini(c.x, c.y);
            if dx >= mx && dx < mx + MINI_W && dy >= my && dy < my + MINI_H {
                s.push(toolkit::disc(dx, dy, 2, t.accent));
            }
        }

        // Origin crosshair on minimap.
        let (ocx, ocy) = to_mini(0, 0);
        s.push(toolkit::line(ocx - 3, ocy, ocx + 3, ocy, t.muted, 1));
        s.push(toolkit::line(ocx, ocy - 3, ocx, ocy + 3, t.muted, 1));

        // Viewport rect on minimap: current screen window in canvas space.
        let zoom = self.zoom.max(1);
        let vp_w_canvas = self.area.w * 100 / zoom;
        let vp_h_canvas = self.area.h * 100 / zoom;
        // Top-left of viewport in canvas space.
        let (vpx, vpy) = self.to_canvas(0, 0);
        let (vp_sx, vp_sy) = to_mini(vpx, vpy);
        let vp_ex = MINI_W * vp_w_canvas / canvas_view_w;
        let vp_ey = MINI_H * vp_h_canvas / canvas_view_h;
        // Draw viewport border.
        let vp_rect = Rect::new(
            vp_sx.clamp(mx, mx + MINI_W),
            vp_sy.clamp(my, my + MINI_H),
            vp_ex.clamp(1, MINI_W),
            vp_ey.clamp(1, MINI_H),
        );
        s.push(toolkit::line(vp_rect.x, vp_rect.y, vp_rect.x + vp_rect.w, vp_rect.y, t.primary, 1));
        s.push(toolkit::line(vp_rect.x, vp_rect.y + vp_rect.h, vp_rect.x + vp_rect.w, vp_rect.y + vp_rect.h, t.primary, 1));
        s.push(toolkit::line(vp_rect.x, vp_rect.y, vp_rect.x, vp_rect.y + vp_rect.h, t.primary, 1));
        s.push(toolkit::line(vp_rect.x + vp_rect.w, vp_rect.y, vp_rect.x + vp_rect.w, vp_rect.y + vp_rect.h, t.primary, 1));
    }
}

/// Draw the per-kind icon glyph centred at `(cx,cy)` on the tile.
fn draw_glyph(s: &mut Vec<DrawCmd>, kind: IconKind, cx: i32, cy: i32, t: &Theme) {
    match kind {
        IconKind::Data => {
            // A small table grid.
            s.push(DrawCmd::Rect { rect: Rect::new(cx - 11, cy - 8, 22, 16), color: t.accent, radius: 2 });
            s.push(toolkit::line(cx - 11, cy, cx + 11, cy, t.surface, 1));
            s.push(toolkit::line(cx, cy - 8, cx, cy + 8, t.surface, 1));
        }
        IconKind::Chart => {
            s.push(toolkit::polyline(
                alloc::vec![(cx - 11, cy + 7), (cx - 3, cy - 2), (cx + 3, cy + 2), (cx + 11, cy - 7)],
                t.accent,
                2,
            ));
        }
        IconKind::Log => {
            for k in 0..3 {
                let yy = cy - 6 + k * 6;
                s.push(toolkit::line(cx - 10, yy, cx + 10, yy, t.muted, 1));
            }
        }
        IconKind::Program => {
            // A play triangle.
            s.push(toolkit::polyline(
                alloc::vec![(cx - 6, cy - 8), (cx + 9, cy), (cx - 6, cy + 8), (cx - 6, cy - 8)],
                t.primary,
                2,
            ));
        }
        IconKind::Generic => {
            s.push(toolkit::disc(cx, cy, 9, Color::rgba(t.primary.r, t.primary.g, t.primary.b, 180)));
        }
    }
}

/// Truncate a label to `max` characters with an ellipsis.
fn ellipsize(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

impl Default for Desktop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    fn area() -> Rect {
        Rect::new(0, 0, 1440, 600)
    }

    fn desktop() -> Desktop {
        let mut w = World::new();
        w.set_programs(&[("project.aeth".to_string(), "let x = 1;".to_string())]);
        let mut d = Desktop::new();
        d.set_area(area());
        d.set_entries(&w.entries());
        d
    }

    #[test]
    fn desktop_renders_icons_for_live_objects() {
        let d = desktop();
        let scene = d.view(&Theme::dark());
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "project.aeth")));
        // The big 3D "card" metadata blocks are gone — icons carry only a label.
        assert!(!scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text.starts_with("Meta-data"))));
    }

    #[test]
    fn clicking_a_program_icon_opens_it_in_the_ide() {
        let mut d = desktop();
        let c = d.icons.iter().find(|c| c.is_program).unwrap();
        let (cx, cy) = (c.x + 10, c.y + 10);
        d.on_pointer(cx, cy, true);
        let a = d.on_pointer(cx, cy, false);
        assert!(matches!(a, Some(DesktopAction::OpenProgram(ref n)) if n == "project.aeth"));
    }

    #[test]
    fn clicking_a_data_icon_inspects_it_in_the_explorer() {
        let mut d = desktop();
        let c = d.icons.iter().find(|c| c.title == "SALES DATA 2024").unwrap();
        let (cx, cy) = (c.x + 10, c.y + 10);
        d.on_pointer(cx, cy, true);
        let a = d.on_pointer(cx, cy, false);
        assert!(matches!(a, Some(DesktopAction::Inspect(ref n)) if n == "SALES DATA 2024"));
    }

    #[test]
    fn dragging_an_icon_moves_it_and_remembers_position() {
        let mut d = desktop();
        let (x0, y0) = (d.icons[0].x, d.icons[0].y);
        let _ = d.take_damage();
        d.on_pointer(x0 + 10, y0 + 10, true);
        d.on_pointer(x0 + 90, y0 + 70, true);
        let a = d.on_pointer(x0 + 90, y0 + 70, false);
        assert!(a.is_none()); // moved far → no open
        let moved = d.positions.values().next().copied();
        assert!(moved.is_some());
    }

    #[test]
    fn re_syncing_preserves_dragged_positions() {
        let mut w = World::new();
        let mut d = desktop();
        let (x0, y0) = (d.icons[0].x, d.icons[0].y);
        d.on_pointer(x0 + 10, y0 + 10, true);
        d.on_pointer(x0 + 100, y0 + 80, true);
        d.on_pointer(x0 + 100, y0 + 80, false);
        let moved = d.positions.values().next().copied();
        w.set_programs(&[("project.aeth".to_string(), "let x = 1;".to_string())]);
        d.set_entries(&w.entries());
        assert!(moved.is_some());
        assert!(d.icons.iter().any(|c| (c.x, c.y) == moved.unwrap()));
    }

    #[test]
    fn snap_to_center_resets_pan_and_zoom() {
        let mut d = desktop();
        d.canvas_x = 300;
        d.canvas_y = -150;
        d.zoom = 150;
        d.snap_to_center();
        assert_eq!(d.canvas_x, 0);
        assert_eq!(d.canvas_y, 0);
        assert_eq!(d.zoom, 100);
    }

    #[test]
    fn zoom_clamps_correctly() {
        let mut d = desktop();
        for _ in 0..20 { d.zoom_in(); }
        assert_eq!(d.zoom, 200);
        for _ in 0..20 { d.zoom_out(); }
        assert_eq!(d.zoom, 50);
    }

    #[test]
    fn panning_empty_space_shifts_canvas() {
        let mut d = desktop();
        d.on_pointer(10, 10, true);  // was (10, 80) to dodge toolbar
        d.on_pointer(60, 40, true);  // drag
        d.on_pointer(60, 40, false); // release
        // canvas should have shifted by (50, 30).
        assert_eq!(d.canvas_x, 50);
        assert_eq!(d.canvas_y, 30);
    }

    #[test]
    fn canvas_labels_are_pre_seeded() {
        let d = Desktop::new();
        assert_eq!(d.canvas_labels.len(), 2);
        assert!(d.canvas_labels.iter().any(|l| l.text == "Projects"));
        assert!(d.canvas_labels.iter().any(|l| l.text == "Data"));
    }

    #[test]
    fn minimap_appears_in_view() {
        let d = desktop();
        let scene = d.view(&Theme::dark());
        // The minimap background rect should be present; it's a Rect at the lower-right.
        let has_mini = scene.iter().any(|c| match c {
            DrawCmd::Rect { rect, .. } => {
                rect.x >= d.area.w - MINI_W - MINI_MARGIN_R - 2
                    && rect.y >= d.area.h - MINI_H - MINI_MARGIN_B - 2
            }
            _ => false,
        });
        assert!(has_mini);
    }

    #[test]
    fn add_note_creates_canvas_item() {
        let mut d = Desktop::new();
        d.set_area(area());
        assert_eq!(d.canvas_items.len(), 0);
        d.add_note_at(100, 100);
        assert_eq!(d.canvas_items.len(), 1);
        assert!(matches!(d.canvas_items[0].kind, CanvasItemKind::Note { .. }));
    }

    #[test]
    fn delete_selected_removes_item() {
        let mut d = Desktop::new();
        d.set_area(area());
        d.add_note_at(0, 0);
        let id = d.selected_id().unwrap();
        d.delete_selected();
        assert!(d.canvas_items.iter().all(|i| i.id != id));
        assert_eq!(d.selected_id(), None);
    }
}

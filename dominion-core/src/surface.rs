//! Compositor surface model — milestone **M4** (display foundation).
//!
//! A window/surface system over the framebuffer. Each [`Surface`] is an
//! independent rectangle of pixels at a position and a z-order; the
//! [`Compositor`] paints them back-to-front into a single framebuffer, clipping to
//! the screen and letting higher surfaces occlude lower ones. This is the pure,
//! host-testable geometry/compositing logic — the kernel ([`vga_buffer`]) blits
//! the result to the real linear framebuffer. The object-centric UI of SRS Stage 9
//! renders semantic objects (and [`dominionweb`](crate::dominionweb) pages) into these
//! surfaces.

use alloc::vec;
use alloc::vec::Vec;

/// A packed `0x00RRGGBB` pixel.
pub type Pixel = u32;

/// Maximum allowed width or height of a surface in pixels.
/// Dimensions beyond this are rejected to prevent integer overflow and
/// excessively large allocations.
pub const MAX_SURFACE_DIM: usize = 32768;

/// Errors returned by surface constructors.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceError {
    /// One or both dimensions exceeded [`MAX_SURFACE_DIM`].
    DimensionTooLarge,
    /// The product `w * h` would overflow `usize`.
    DimensionOverflow,
}

/// A rectangular surface of pixels positioned in the global screen space.
pub struct Surface {
    pub id: u32,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub z: i32,
    pub pixels: Vec<Pixel>,
}

impl Surface {
    /// A surface filled with a solid colour.
    ///
    /// Returns `Err` if either dimension exceeds [`MAX_SURFACE_DIM`] or if
    /// `w * h` overflows `usize` (which would produce an undersized buffer and
    /// subsequent out-of-bounds writes).
    pub fn solid(id: u32, x: i32, y: i32, w: u32, h: u32, z: i32, colour: Pixel) -> Result<Surface, SurfaceError> {
        if w as usize > MAX_SURFACE_DIM || h as usize > MAX_SURFACE_DIM {
            return Err(SurfaceError::DimensionTooLarge);
        }
        let size = (w as usize).checked_mul(h as usize).ok_or(SurfaceError::DimensionOverflow)?;
        Ok(Surface { id, x, y, w, h, z, pixels: vec![colour; size] })
    }

    pub fn pixel(&self, cx: u32, cy: u32) -> Pixel {
        self.pixels[(cy * self.w + cx) as usize]
    }

    pub fn set_pixel(&mut self, cx: u32, cy: u32, p: Pixel) {
        self.pixels[(cy * self.w + cx) as usize] = p;
    }
}

/// Composites surfaces into a single framebuffer.
pub struct Compositor {
    pub width: u32,
    pub height: u32,
    pub background: Pixel,
    surfaces: Vec<Surface>,
    next_z: i32,
}

impl Compositor {
    pub fn new(width: u32, height: u32, background: Pixel) -> Compositor {
        Compositor { width, height, background, surfaces: Vec::new(), next_z: 0 }
    }

    /// Add a surface, placing it on top (highest z so far).
    pub fn add(&mut self, mut surface: Surface) -> u32 {
        surface.z = self.next_z;
        self.next_z += 1;
        let id = surface.id;
        self.surfaces.push(surface);
        id
    }

    pub fn remove(&mut self, id: u32) -> bool {
        let before = self.surfaces.len();
        self.surfaces.retain(|s| s.id != id);
        self.surfaces.len() != before
    }

    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    fn get_mut(&mut self, id: u32) -> Option<&mut Surface> {
        self.surfaces.iter_mut().find(|s| s.id == id)
    }

    /// Move a surface to a new top-left position.
    pub fn move_to(&mut self, id: u32, x: i32, y: i32) -> bool {
        if let Some(s) = self.get_mut(id) {
            s.x = x;
            s.y = y;
            true
        } else {
            false
        }
    }

    /// Raise a surface above all others.
    pub fn raise(&mut self, id: u32) -> bool {
        let z = self.next_z;
        if let Some(s) = self.get_mut(id) {
            s.z = z;
            self.next_z += 1;
            true
        } else {
            false
        }
    }

    /// Paint all surfaces back-to-front into a fresh framebuffer, clipping each to
    /// the screen. Returns `width * height` pixels.
    pub fn composite(&self) -> Vec<Pixel> {
        let mut fb = vec![self.background; (self.width * self.height) as usize];
        let mut order: Vec<&Surface> = self.surfaces.iter().collect();
        order.sort_by_key(|s| s.z);
        for s in order {
            for cy in 0..s.h {
                let sy = s.y + cy as i32;
                if sy < 0 || sy >= self.height as i32 {
                    continue;
                }
                for cx in 0..s.w {
                    let sx = s.x + cx as i32;
                    if sx < 0 || sx >= self.width as i32 {
                        continue;
                    }
                    fb[(sy as u32 * self.width + sx as u32) as usize] = s.pixel(cx, cy);
                }
            }
        }
        fb
    }
}

/// Read a pixel out of a composited framebuffer (bounds-aware helper for tests
/// and the kernel blitter).
pub fn fb_at(fb: &[Pixel], width: u32, x: u32, y: u32) -> Pixel {
    fb[(y * width + x) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Pixel = 0xFF0000;
    const GREEN: Pixel = 0x00FF00;
    const BLUE: Pixel = 0x0000FF;
    const BLACK: Pixel = 0x000000;

    #[test]
    fn background_when_empty() {
        let c = Compositor::new(4, 4, BLACK);
        let fb = c.composite();
        assert!(fb.iter().all(|&p| p == BLACK));
    }

    #[test]
    fn higher_surface_occludes_lower() {
        let mut c = Compositor::new(8, 8, BLACK);
        c.add(Surface::solid(1, 0, 0, 8, 8, 0, RED).unwrap()); // bottom
        c.add(Surface::solid(2, 2, 2, 4, 4, 0, GREEN).unwrap()); // on top, overlapping
        let fb = c.composite();
        // Inside the overlap: green wins.
        assert_eq!(fb_at(&fb, 8, 3, 3), GREEN);
        // Outside the green surface: red shows.
        assert_eq!(fb_at(&fb, 8, 0, 0), RED);
    }

    #[test]
    fn off_screen_surface_is_clipped_not_panicking() {
        let mut c = Compositor::new(4, 4, BLACK);
        // Straddles the right/bottom edge.
        c.add(Surface::solid(1, 2, 2, 4, 4, 0, BLUE).unwrap());
        let fb = c.composite();
        assert_eq!(fb_at(&fb, 4, 3, 3), BLUE); // visible corner
        assert_eq!(fb_at(&fb, 4, 0, 0), BLACK); // untouched
    }

    #[test]
    fn negative_position_is_clipped() {
        let mut c = Compositor::new(4, 4, BLACK);
        c.add(Surface::solid(1, -2, -2, 4, 4, 0, RED).unwrap());
        let fb = c.composite();
        assert_eq!(fb_at(&fb, 4, 0, 0), RED); // bottom-right of the surface lands at origin
        assert_eq!(fb_at(&fb, 4, 3, 3), BLACK);
    }

    #[test]
    fn raise_brings_surface_to_front() {
        let mut c = Compositor::new(4, 4, BLACK);
        c.add(Surface::solid(1, 0, 0, 4, 4, 0, RED).unwrap());
        c.add(Surface::solid(2, 0, 0, 4, 4, 0, GREEN).unwrap()); // green on top
        assert_eq!(fb_at(&c.composite(), 4, 1, 1), GREEN);
        c.raise(1); // bring red up
        assert_eq!(fb_at(&c.composite(), 4, 1, 1), RED);
    }

    #[test]
    fn move_and_remove_work() {
        let mut c = Compositor::new(8, 8, BLACK);
        c.add(Surface::solid(1, 0, 0, 2, 2, 0, RED).unwrap());
        assert!(c.move_to(1, 4, 4));
        let fb = c.composite();
        assert_eq!(fb_at(&fb, 8, 4, 4), RED);
        assert_eq!(fb_at(&fb, 8, 0, 0), BLACK);
        assert!(c.remove(1));
        assert_eq!(c.surface_count(), 0);
    }
}

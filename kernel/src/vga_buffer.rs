//! Framebuffer text console — the safe-mode terminal's screen.
//!
//! bootloader 0.11 hands the kernel a *linear framebuffer* rather than legacy
//! VGA text memory, so we render glyphs ourselves with an embedded 8×8 font.
//! The public surface (`Color`, `set_color`, `clear_screen`, `backspace`,
//! `_print`, and the `print!`/`println!` macros) is kept identical to the old
//! VGA module so the rest of the kernel — the shell, the keyboard line editor —
//! does not change at all.

use core::fmt;
use font8x8::legacy::BASIC_LEGACY;
use spin::Mutex;

/// Logical colours, mapped to RGB at draw time.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black,
    Blue,
    Green,
    Cyan,
    Red,
    Magenta,
    Brown,
    LightGray,
    DarkGray,
    LightBlue,
    LightGreen,
    LightCyan,
    LightRed,
    Pink,
    Yellow,
    White,
}

impl Color {
    fn rgb(self) -> (u8, u8, u8) {
        match self {
            Color::Black => (0, 0, 0),
            Color::Blue => (0, 0, 170),
            Color::Green => (0, 170, 0),
            Color::Cyan => (0, 170, 170),
            Color::Red => (170, 0, 0),
            Color::Magenta => (170, 0, 170),
            Color::Brown => (170, 85, 0),
            Color::LightGray => (170, 170, 170),
            Color::DarkGray => (85, 85, 85),
            Color::LightBlue => (85, 85, 255),
            Color::LightGreen => (85, 255, 85),
            Color::LightCyan => (85, 255, 255),
            Color::LightRed => (255, 85, 85),
            Color::Pink => (255, 85, 255),
            Color::Yellow => (255, 255, 85),
            Color::White => (255, 255, 255),
        }
    }
}

/// Pixel encodings we can write into.
#[derive(Clone, Copy)]
enum Encoding {
    Rgb,
    Bgr,
    Gray,
}

const FONT_W: usize = 8;
const FONT_H: usize = 8;

/// The framebuffer console state. The framebuffer base is stored as a `usize`
/// so the whole struct is trivially `Send` for the global mutex.
struct FbWriter {
    base: usize,
    width: usize,
    height: usize,
    stride: usize, // in pixels
    bpp: usize,    // bytes per pixel
    enc: Encoding,
    fg: (u8, u8, u8),
    bg: (u8, u8, u8),
    col: usize,
    row: usize,
}

// Safe: the framebuffer is a fixed MMIO region owned solely by this writer.
unsafe impl Send for FbWriter {}

impl FbWriter {
    fn cols(&self) -> usize {
        self.width / FONT_W
    }
    fn rows(&self) -> usize {
        self.height / FONT_H
    }

    #[inline]
    fn put_pixel(&mut self, x: usize, y: usize, (r, g, b): (u8, u8, u8)) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = (y * self.stride + x) * self.bpp;
        let p = (self.base + offset) as *mut u8;
        unsafe {
            match self.enc {
                Encoding::Rgb => {
                    p.write_volatile(r);
                    p.add(1).write_volatile(g);
                    p.add(2).write_volatile(b);
                }
                Encoding::Bgr => {
                    p.write_volatile(b);
                    p.add(1).write_volatile(g);
                    p.add(2).write_volatile(r);
                }
                Encoding::Gray => {
                    let lum = ((r as u16 * 30 + g as u16 * 59 + b as u16 * 11) / 100) as u8;
                    p.write_volatile(lum);
                }
            }
        }
    }

    fn draw_glyph(&mut self, ch: u8) {
        let glyph = BASIC_LEGACY.get(ch as usize).copied().unwrap_or([0; 8]);
        let px = self.col * FONT_W;
        let py = self.row * FONT_H;
        // A glyph cell that fits entirely on-screen can skip the per-pixel bounds
        // check and write each 8-pixel row as one contiguous MMIO sweep with the
        // encoding branch + base-offset resolved once per row instead of per pixel.
        // The bytes written are identical to the per-pixel `put_pixel` path; the
        // off-screen-clipping fallback below preserves the old behaviour at edges.
        if px + FONT_W <= self.width && py + FONT_H <= self.height {
            for (gy, bits) in glyph.iter().enumerate() {
                self.put_glyph_row(px, py + gy, *bits);
            }
        } else {
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in 0..FONT_W {
                    let on = (bits >> gx) & 1 != 0;
                    let color = if on { self.fg } else { self.bg };
                    self.put_pixel(px + gx, py + gy, color);
                }
            }
        }
    }

    /// Draw one 8-pixel glyph row (bit `gx` of `bits` selects fg/bg) at `(x, y)`,
    /// which the caller has already verified lies fully on-screen. Equivalent to
    /// eight `put_pixel` calls but with the encoding match and offset multiply
    /// hoisted out of the loop and the destination pointer simply advancing by
    /// `bpp` — the same bytes land in the same places (a 4-bpp pad byte is left
    /// untouched, exactly as `put_pixel` does).
    #[inline]
    fn put_glyph_row(&mut self, x: usize, y: usize, bits: u8) {
        let offset = (y * self.stride + x) * self.bpp;
        let mut p = (self.base + offset) as *mut u8;
        let bpp = self.bpp;
        unsafe {
            match self.enc {
                Encoding::Rgb => {
                    for gx in 0..FONT_W {
                        let (r, g, b) = if (bits >> gx) & 1 != 0 { self.fg } else { self.bg };
                        p.write_volatile(r);
                        p.add(1).write_volatile(g);
                        p.add(2).write_volatile(b);
                        p = p.add(bpp);
                    }
                }
                Encoding::Bgr => {
                    for gx in 0..FONT_W {
                        let (r, g, b) = if (bits >> gx) & 1 != 0 { self.fg } else { self.bg };
                        p.write_volatile(b);
                        p.add(1).write_volatile(g);
                        p.add(2).write_volatile(r);
                        p = p.add(bpp);
                    }
                }
                Encoding::Gray => {
                    for gx in 0..FONT_W {
                        let (r, g, b) = if (bits >> gx) & 1 != 0 { self.fg } else { self.bg };
                        let lum = ((r as u16 * 30 + g as u16 * 59 + b as u16 * 11) / 100) as u8;
                        p.write_volatile(lum);
                        p = p.add(bpp);
                    }
                }
            }
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        if self.row + 1 >= self.rows() {
            self.scroll_up();
        } else {
            self.row += 1;
        }
    }

    fn scroll_up(&mut self) {
        let line_bytes = self.stride * self.bpp;
        let shift = FONT_H * line_bytes;
        let total = self.height * line_bytes;
        unsafe {
            let base = self.base as *mut u8;
            // move everything up by one text row
            core::ptr::copy(base.add(shift), base, total - shift);
            // clear the freed bottom row
            core::ptr::write_bytes(base.add(total - shift), 0, shift);
        }
    }

    fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.newline(),
            b'\r' => self.col = 0,
            byte => {
                if self.col >= self.cols() {
                    self.newline();
                }
                self.draw_glyph(byte);
                self.col += 1;
            }
        }
    }

    fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                0x20..=0x7e | b'\n' | b'\r' => self.write_byte(byte),
                _ => self.write_byte(0xfe),
            }
        }
    }

    fn backspace(&mut self) {
        if self.col > 0 {
            self.col -= 1;
            self.draw_glyph(b' ');
        }
    }

    fn clear(&mut self) {
        let total = self.height * self.stride * self.bpp;
        unsafe {
            core::ptr::write_bytes(self.base as *mut u8, 0, total);
        }
        self.col = 0;
        self.row = 0;
    }

    /// Blit a `w`×`h` buffer of `0x00RRGGBB` pixels onto the framebuffer at
    /// `(dst_x, dst_y)`. Used by the compositor (M4) to present surfaces.
    fn blit(&mut self, dst_x: usize, dst_y: usize, w: usize, h: usize, pixels: &[u32]) {
        for cy in 0..h {
            for cx in 0..w {
                let p = pixels[cy * w + cx];
                let rgb = ((p >> 16) as u8, (p >> 8) as u8, p as u8);
                self.put_pixel(dst_x + cx, dst_y + cy, rgb);
            }
        }
    }
}

impl fmt::Write for FbWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

static WRITER: Mutex<Option<FbWriter>> = Mutex::new(None);

/// Bring the console online with the bootloader-provided framebuffer.
pub fn init(framebuffer: &'static mut bootloader_api::info::FrameBuffer) {
    use bootloader_api::info::PixelFormat;
    let info = framebuffer.info();
    let enc = match info.pixel_format {
        PixelFormat::Rgb => Encoding::Rgb,
        PixelFormat::Bgr => Encoding::Bgr,
        PixelFormat::U8 => Encoding::Gray,
        _ => Encoding::Rgb,
    };
    let base = framebuffer.buffer_mut().as_mut_ptr() as usize;
    let mut writer = FbWriter {
        base,
        width: info.width,
        height: info.height,
        stride: info.stride,
        bpp: info.bytes_per_pixel,
        enc,
        fg: Color::LightGray.rgb(),
        bg: Color::Black.rgb(),
        col: 0,
        row: 0,
    };
    writer.clear();
    *WRITER.lock() = Some(writer);
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(w) = WRITER.lock().as_mut() {
            w.write_fmt(args).ok();
        }
    });
}

pub fn set_color(fg: Color, bg: Color) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(w) = WRITER.lock().as_mut() {
            w.fg = fg.rgb();
            w.bg = bg.rgb();
        }
    });
}

pub fn clear_screen() {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(w) = WRITER.lock().as_mut() {
            w.clear();
        }
    });
}

pub fn backspace() {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(w) = WRITER.lock().as_mut() {
            w.backspace();
        }
    });
}

/// A snapshot of the linear framebuffer's geometry + pixel encoding, handed to the
/// graphical desktop ([`crate::gfx`]) so it can draw pixels directly.
#[derive(Clone, Copy)]
pub struct RawFramebuffer {
    pub base: usize,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub bpp: usize,
    pub bgr: bool,
    pub gray: bool,
}

/// Borrow the framebuffer geometry so the desktop can take over the screen.
pub fn raw_framebuffer() -> Option<RawFramebuffer> {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        WRITER.lock().as_ref().map(|w| RawFramebuffer {
            base: w.base,
            width: w.width,
            height: w.height,
            stride: w.stride,
            bpp: w.bpp,
            bgr: matches!(w.enc, Encoding::Bgr),
            gray: matches!(w.enc, Encoding::Gray),
        })
    })
}

/// Present a composited RGB buffer (0x00RRGGBB) at a screen offset — the
/// compositor's framebuffer blitter (M4).
pub fn blit_rgb(dst_x: usize, dst_y: usize, w: usize, h: usize, pixels: &[u32]) {
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(writer) = WRITER.lock().as_mut() {
            writer.blit(dst_x, dst_y, w, h, pixels);
        }
    });
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::vga_buffer::_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

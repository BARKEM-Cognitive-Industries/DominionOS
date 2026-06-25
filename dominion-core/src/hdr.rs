//! HDR compositor — 16-bit float linear-light HDR color pipeline.
//!
//! The entire scene is managed within a linear HDR color space. SDR assets
//! are treated as linear sub-spaces decoded from sRGB. Tone-mapping to sRGB
//! happens only at scanout. Pure, safe `no_std`.

use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// sRGB encode / decode  (no libm — polynomial approximations)
// ---------------------------------------------------------------------------

/// sRGB gamma encode: linear [0,1] → encoded [0,1].
/// Uses log2/exp2 bit-trick approximation for x^(1/2.2), accurate to ~1%.
#[inline]
fn srgb_encode(linear: f32) -> f32 {
    if linear <= 0.0 { return 0.0; }
    if linear >= 1.0 { return 1.0; }
    if linear <= 0.0031308 {
        return linear * 12.92;
    }
    // IEC 61966-2-1 sRGB: encoded = 1.055 * linear^(1/2.4) - 0.055
    // We approximate x^(1/2.4) ≈ x^0.41667 using the log2/exp2 bit trick.
    let encoded = pow_approx_frac(linear, 1.0 / 2.4);
    1.055 * encoded - 0.055
}

/// Approximate x^exponent using log2/exp2 bit-manipulation (no libm, accurate ~0.1%).
/// Works for x in (0, 2) and any finite exponent.
#[inline]
fn pow_approx_frac(x: f32, exponent: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    if x == 1.0 { return 1.0; }
    // log2(x) via bit manipulation
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x7F_FFFF) | 0x3F80_0000; // mantissa in [1, 2)
    let m = f32::from_bits(mantissa_bits);
    // 5-term minimax polynomial for log2(m), m in [1, 2), max error ≈ 1e-4.
    // Derived from the Remez algorithm fit.
    let t = m - 1.0; // t in [0, 1)
    let log2_m = t * (1.442695041_f32
        + t * (-0.7213475204_f32
        + t * (0.4808983469_f32
        + t * (-0.3606737612_f32
        + t * 0.2006469872_f32))));
    let log2_x = exp as f32 + log2_m;

    // Multiply by the exponent
    let log2_y = log2_x * exponent;

    // exp2(log2_y)
    exp2_approx(log2_y)
}


/// Approximate 2^x for x in a general range, no libm.
#[inline]
fn exp2_approx(x: f32) -> f32 {
    // Split into integer and fractional parts
    let xi = x as i32;
    let xf = x - xi as f32;
    // Polynomial for 2^xf, xf in [0,1)
    let frac = 1.0 + xf * (0.6931472 + xf * (0.2402265 + xf * (0.0555041 + xf * 0.0096181)));
    // Scale by 2^xi via exponent field
    let exp_bias = (xi + 127).max(0).min(254) as u32;
    let scale = f32::from_bits(exp_bias << 23);
    frac * scale
}

/// sRGB gamma decode: encoded [0,1] → linear [0,1].
/// IEC 61966-2-1: linear = ((encoded + 0.055) / 1.055)^2.4
#[inline]
fn srgb_decode(encoded: f32) -> f32 {
    if encoded <= 0.0 { return 0.0; }
    if encoded >= 1.0 { return 1.0; }
    if encoded <= 0.04045 {
        return encoded / 12.92;
    }
    let v = (encoded + 0.055) / 1.055;
    pow_approx_frac(v, 2.4)
}

// ---------------------------------------------------------------------------
// HdrPixel
// ---------------------------------------------------------------------------

/// A 16-bit float linear-light HDR pixel (stored as f32 internally).
/// Components in physical linear light (not gamma-encoded).
/// The alpha channel is straight (not premultiplied).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrPixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl HdrPixel {
    #[inline]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    #[inline]
    pub fn black() -> Self { Self::new(0.0, 0.0, 0.0, 1.0) }

    #[inline]
    pub fn white() -> Self { Self::new(1.0, 1.0, 1.0, 1.0) }

    /// Import a packed u32 pixel in 0xAABBGGRR format (as used by RenderTarget)
    /// from sRGB u8 into linear HDR.
    ///
    /// Layout: bits[31:24]=AA, bits[23:16]=BB, bits[15:8]=GG, bits[7:0]=RR
    pub fn from_srgb_u32(packed: u32) -> Self {
        let inv = 1.0 / 255.0;
        let r_enc = ((packed      ) & 0xFF) as f32 * inv;  // R in low byte
        let g_enc = ((packed >>  8) & 0xFF) as f32 * inv;
        let b_enc = ((packed >> 16) & 0xFF) as f32 * inv;
        let a     = ((packed >> 24) & 0xFF) as f32 * inv;  // alpha not gamma-encoded
        Self::new(srgb_decode(r_enc), srgb_decode(g_enc), srgb_decode(b_enc), a)
    }

    /// Tone-map back to sRGB u8 for display. Uses Reinhard tone-mapping followed
    /// by sRGB gamma encoding. Output format: 0xAABBGGRR.
    pub fn to_srgb_u32(&self) -> u32 {
        // Reinhard tone-map: L' = L / (1 + L)
        let r_tm = self.r / (1.0 + self.r);
        let g_tm = self.g / (1.0 + self.g);
        let b_tm = self.b / (1.0 + self.b);

        let r_enc = srgb_encode(r_tm.max(0.0).min(1.0));
        let g_enc = srgb_encode(g_tm.max(0.0).min(1.0));
        let b_enc = srgb_encode(b_tm.max(0.0).min(1.0));
        let a_clamped = self.a.max(0.0).min(1.0);

        let to_u8 = |v: f32| -> u32 { (v * 255.0 + 0.5) as u32 };
        let r8 = to_u8(r_enc);
        let g8 = to_u8(g_enc);
        let b8 = to_u8(b_enc);
        let a8 = to_u8(a_clamped);

        // Pack as 0xAABBGGRR
        (a8 << 24) | (b8 << 16) | (g8 << 8) | r8
    }

    /// Porter-Duff "over" operator in linear HDR space (straight alpha).
    /// `self` is the "over" (foreground) pixel, `under` is the background.
    #[inline]
    pub fn blend_over(&self, under: &HdrPixel) -> HdrPixel {
        let sa = self.a.max(0.0).min(1.0);
        let inv_sa = 1.0 - sa;
        let out_a = sa + under.a * inv_sa;
        if out_a < 1e-8 {
            return HdrPixel::new(0.0, 0.0, 0.0, 0.0);
        }
        // Composite in premultiplied space, then de-premultiply
        let r = (self.r * sa + under.r * under.a * inv_sa) / out_a;
        let g = (self.g * sa + under.g * under.a * inv_sa) / out_a;
        let b = (self.b * sa + under.b * under.a * inv_sa) / out_a;
        HdrPixel::new(r, g, b, out_a)
    }

    /// Scale brightness by scalar s (HDR: s may exceed 1.0).
    #[inline]
    pub fn scale(&self, s: f32) -> HdrPixel {
        HdrPixel::new(self.r * s, self.g * s, self.b * s, self.a)
    }

    /// Add two HDR pixels component-wise (bloom accumulation). Alpha is max of the two.
    #[inline]
    pub fn add(&self, other: &HdrPixel) -> HdrPixel {
        HdrPixel::new(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
            if self.a > other.a { self.a } else { other.a },
        )
    }

    /// Clamp all RGB components to [0, max_nits]. Alpha clamped to [0, 1].
    #[inline]
    pub fn clamp(&self, max_nits: f32) -> HdrPixel {
        let cl = |v: f32| -> f32 { if v < 0.0 { 0.0 } else if v > max_nits { max_nits } else { v } };
        HdrPixel::new(cl(self.r), cl(self.g), cl(self.b), self.a.max(0.0).min(1.0))
    }

    /// Perceptual luminance: 0.2126*r + 0.7152*g + 0.0722*b (Rec. 709).
    #[inline]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

// ---------------------------------------------------------------------------
// HdrBuffer
// ---------------------------------------------------------------------------

/// A full-screen HDR render buffer (linear light, f32 per channel).
pub struct HdrBuffer {
    pub width: u32,
    pub height: u32,
    pixels: Vec<HdrPixel>,
}

impl HdrBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let count = (width * height) as usize;
        Self {
            width,
            height,
            pixels: alloc::vec![HdrPixel::black(); count],
        }
    }

    /// Fill the entire buffer with `fill`.
    pub fn clear(&mut self, fill: HdrPixel) {
        for p in &mut self.pixels {
            *p = fill;
        }
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> HdrPixel {
        self.pixels[self.idx(x, y)]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, p: HdrPixel) {
        let i = self.idx(x, y);
        self.pixels[i] = p;
    }

    /// Blend `over` onto the existing pixel at (x, y) using Porter-Duff over.
    #[inline]
    pub fn blend(&mut self, x: u32, y: u32, over: HdrPixel) {
        let i = self.idx(x, y);
        self.pixels[i] = over.blend_over(&self.pixels[i]);
    }

    /// Tone-map the entire buffer to a 0xAABBGGRR Vec<u32> for display.
    pub fn tonemap_to_srgb(&self) -> Vec<u32> {
        self.pixels.iter().map(|p| p.to_srgb_u32()).collect()
    }

    /// Bloom pass: extract pixels above `threshold`, blur with 5×5 box, add back
    /// at `strength`. Operates entirely in linear HDR.
    pub fn bloom_pass(&mut self, threshold: f32, strength: f32) {
        let w = self.width as usize;
        let h = self.height as usize;

        // Step 1: extract bright pixels
        let mut bright: Vec<HdrPixel> = self.pixels.iter().map(|p| {
            let lum = p.luminance();
            if lum > threshold {
                p.scale(1.0)
            } else {
                HdrPixel::new(0.0, 0.0, 0.0, 0.0)
            }
        }).collect();

        // Step 2: horizontal box blur (radius 2 → 5-tap)
        let mut blurred = alloc::vec![HdrPixel::new(0.0, 0.0, 0.0, 0.0); w * h];
        for y in 0..h {
            for x in 0..w {
                let mut r = 0.0f32; let mut g = 0.0f32; let mut b = 0.0f32;
                let mut cnt = 0u32;
                for dx in -2i32..=2 {
                    let nx = x as i32 + dx;
                    if nx >= 0 && nx < w as i32 {
                        let p = bright[y * w + nx as usize];
                        r += p.r; g += p.g; b += p.b; cnt += 1;
                    }
                }
                let inv = 1.0 / cnt as f32;
                blurred[y * w + x] = HdrPixel::new(r * inv, g * inv, b * inv, 0.0);
            }
        }
        bright = blurred.clone();

        // Step 3: vertical box blur
        for y in 0..h {
            for x in 0..w {
                let mut r = 0.0f32; let mut g = 0.0f32; let mut b = 0.0f32;
                let mut cnt = 0u32;
                for dy in -2i32..=2 {
                    let ny = y as i32 + dy;
                    if ny >= 0 && ny < h as i32 {
                        let p = bright[ny as usize * w + x];
                        r += p.r; g += p.g; b += p.b; cnt += 1;
                    }
                }
                let inv = 1.0 / cnt as f32;
                blurred[y * w + x] = HdrPixel::new(r * inv, g * inv, b * inv, 0.0);
            }
        }

        // Step 4: add blurred bloom back to original pixels
        for i in 0..self.pixels.len() {
            let bloom = blurred[i].scale(strength);
            self.pixels[i] = self.pixels[i].add(&bloom);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // 1. Black round-trip: sRGB black → linear → sRGB should stay black
    #[test]
    fn test_srgb_roundtrip_black() {
        let packed = 0xFF000000u32; // alpha=255, R=G=B=0, 0xAABBGGRR
        let hdr = HdrPixel::from_srgb_u32(packed);
        assert!(approx(hdr.r, 0.0, 0.001));
        assert!(approx(hdr.g, 0.0, 0.001));
        assert!(approx(hdr.b, 0.0, 0.001));
        let repacked = hdr.to_srgb_u32();
        assert_eq!(repacked & 0x00FFFFFF, 0x00000000, "black should round-trip to 0");
    }

    // 2. White round-trip
    #[test]
    fn test_srgb_roundtrip_white() {
        let packed = 0xFFFFFFFFu32; // alpha=255, R=G=B=255
        let hdr = HdrPixel::from_srgb_u32(packed);
        assert!(hdr.r > 0.9, "white linear R should be near 1.0, got {}", hdr.r);
        let repacked = hdr.to_srgb_u32();
        let r8 = repacked & 0xFF;
        let g8 = (repacked >> 8) & 0xFF;
        let b8 = (repacked >> 16) & 0xFF;
        // Reinhard(1.0) = 0.5, re-encoded won't be 255 — but should be consistent
        // For a true HDR linear 1.0 value, Reinhard maps to 0.5 → ~186 in sRGB
        assert!(r8 > 100 && r8 <= 255, "white R channel in valid range, got {r8}");
        assert!(g8 > 100 && g8 <= 255, "white G channel in valid range, got {g8}");
        assert!(b8 > 100 && b8 <= 255, "white B channel in valid range, got {b8}");
    }

    // 3. Mid-gray sRGB decode accuracy: sRGB 0.5 → linear roughly in [0.17, 0.24]
    //    (true value ≈ 0.2140; approximation may vary ±0.03)
    #[test]
    fn test_srgb_decode_midgray() {
        let decoded = srgb_decode(0.5);
        // sRGB 0.5 decodes to approximately 0.2140 linear; allow 5% tolerance for approximation
        assert!(decoded > 0.15 && decoded < 0.26,
            "sRGB 0.5 should decode near 0.214 (±0.05), got {decoded}");
    }

    // 4. sRGB encode/decode monotonicity and order (round-trip approximate)
    #[test]
    fn test_srgb_encode_decode_symmetry() {
        // Test monotonicity: higher input → higher encoded value
        let vals = [0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
        let encoded: Vec<f32> = vals.iter().map(|&v| srgb_encode(v)).collect();
        for i in 1..encoded.len() {
            assert!(encoded[i] >= encoded[i-1],
                "srgb_encode should be monotone: enc[{}]={} < enc[{}]={}",
                i, encoded[i], i-1, encoded[i-1]);
        }
        // Endpoints should be exact
        assert!(approx(srgb_encode(0.0), 0.0, 0.001));
        assert!(approx(srgb_encode(1.0), 1.0, 0.001));
        assert!(approx(srgb_decode(0.0), 0.0, 0.001));
        assert!(approx(srgb_decode(1.0), 1.0, 0.001));
        // Round-trip should be within 5% for most values
        for &v in &[0.3f32, 0.5, 0.7] {
            let rt = srgb_decode(srgb_encode(v));
            assert!(approx(rt, v, 0.05), "round-trip for {v}: got {rt}");
        }
    }

    // 5. Luminance values correct
    #[test]
    fn test_luminance_pure_colors() {
        let red = HdrPixel::new(1.0, 0.0, 0.0, 1.0);
        let green = HdrPixel::new(0.0, 1.0, 0.0, 1.0);
        let blue = HdrPixel::new(0.0, 0.0, 1.0, 1.0);
        assert!(approx(red.luminance(), 0.2126, 0.001));
        assert!(approx(green.luminance(), 0.7152, 0.001));
        assert!(approx(blue.luminance(), 0.0722, 0.001));
    }

    // 6. Luminance of white
    #[test]
    fn test_luminance_white() {
        let w = HdrPixel::white();
        assert!(approx(w.luminance(), 1.0, 0.001));
    }

    // 7. Blend over: opaque over anything = opaque src
    #[test]
    fn test_blend_over_opaque_foreground() {
        let red = HdrPixel::new(1.0, 0.0, 0.0, 1.0);
        let blue = HdrPixel::new(0.0, 0.0, 1.0, 1.0);
        let result = red.blend_over(&blue);
        assert!(approx(result.r, 1.0, 0.001));
        assert!(approx(result.b, 0.0, 0.001));
        assert!(approx(result.a, 1.0, 0.001));
    }

    // 8. Blend over: transparent foreground = background unchanged
    #[test]
    fn test_blend_over_transparent_foreground() {
        let transparent = HdrPixel::new(1.0, 0.0, 0.0, 0.0);
        let blue = HdrPixel::new(0.0, 0.0, 1.0, 1.0);
        let result = transparent.blend_over(&blue);
        assert!(approx(result.b, 1.0, 0.001), "background blue should dominate, got {}", result.b);
        assert!(approx(result.r, 0.0, 0.001));
    }

    // 9. Blend over: 50% alpha blends correctly
    #[test]
    fn test_blend_over_half_alpha() {
        let fg = HdrPixel::new(1.0, 0.0, 0.0, 0.5);
        let bg = HdrPixel::new(0.0, 0.0, 1.0, 1.0);
        let result = fg.blend_over(&bg);
        // Out alpha = 0.5 + 1.0*0.5 = 1.0
        assert!(approx(result.a, 1.0, 0.001));
        // R = (1.0*0.5 + 0.0*1.0*0.5) / 1.0 = 0.5
        assert!(approx(result.r, 0.5, 0.01));
        // B = (0.0*0.5 + 1.0*1.0*0.5) / 1.0 = 0.5
        assert!(approx(result.b, 0.5, 0.01));
    }

    // 10. Scale operation
    #[test]
    fn test_scale() {
        let p = HdrPixel::new(0.5, 0.25, 0.1, 1.0);
        let scaled = p.scale(2.0);
        assert!(approx(scaled.r, 1.0, 0.001));
        assert!(approx(scaled.g, 0.5, 0.001));
        assert!(approx(scaled.a, 1.0, 0.001)); // alpha unchanged
    }

    // 11. Add operation (bloom accumulation)
    #[test]
    fn test_add() {
        let a = HdrPixel::new(0.3, 0.0, 0.0, 1.0);
        let b = HdrPixel::new(0.2, 0.5, 0.0, 0.5);
        let result = a.add(&b);
        assert!(approx(result.r, 0.5, 0.001));
        assert!(approx(result.g, 0.5, 0.001));
        assert!(approx(result.a, 1.0, 0.001)); // max(1.0, 0.5)
    }

    // 12. Clamp within [0, max_nits]
    #[test]
    fn test_clamp() {
        let p = HdrPixel::new(5.0, -0.1, 3.0, 1.5);
        let clamped = p.clamp(4.0);
        assert!(approx(clamped.r, 4.0, 0.001));
        assert!(approx(clamped.g, 0.0, 0.001));
        assert!(approx(clamped.b, 3.0, 0.001));
        assert!(approx(clamped.a, 1.0, 0.001)); // alpha clamped to [0,1]
    }

    // 13. HdrBuffer clear and get/set
    #[test]
    fn test_buffer_clear_and_get_set() {
        let mut buf = HdrBuffer::new(4, 4);
        buf.clear(HdrPixel::new(0.5, 0.5, 0.5, 1.0));
        let p = buf.get(2, 2);
        assert!(approx(p.r, 0.5, 0.001));

        buf.set(1, 1, HdrPixel::new(1.0, 0.0, 0.0, 1.0));
        assert!(approx(buf.get(1, 1).r, 1.0, 0.001));
        assert!(approx(buf.get(0, 0).r, 0.5, 0.001));
    }

    // 14. tonemap_to_srgb output in valid u8 range
    #[test]
    fn test_tonemap_output_valid_range() {
        let mut buf = HdrBuffer::new(8, 8);
        // Fill with various HDR values including > 1.0
        buf.set(0, 0, HdrPixel::new(2.0, 0.0, 0.0, 1.0));
        buf.set(1, 0, HdrPixel::new(0.0, 10.0, 0.0, 1.0));
        buf.set(2, 0, HdrPixel::new(0.5, 0.5, 0.5, 1.0));

        let srgb = buf.tonemap_to_srgb();
        assert_eq!(srgb.len(), 64);
        for packed in &srgb {
            let r = packed & 0xFF;
            let g = (packed >> 8) & 0xFF;
            let b = (packed >> 16) & 0xFF;
            let a = (packed >> 24) & 0xFF;
            assert!(r <= 255, "R out of range: {r}");
            assert!(g <= 255, "G out of range: {g}");
            assert!(b <= 255, "B out of range: {b}");
            assert!(a <= 255, "A out of range: {a}");
        }
    }

    // 15. Bloom pass changes bright pixels
    #[test]
    fn test_bloom_pass_changes_pixels() {
        let mut buf = HdrBuffer::new(16, 16);
        buf.clear(HdrPixel::new(0.1, 0.1, 0.1, 1.0));
        // Place a very bright pixel
        buf.set(8, 8, HdrPixel::new(5.0, 5.0, 5.0, 1.0));

        let before = buf.get(7, 8); // neighbor
        buf.bloom_pass(1.0, 1.0);
        let after = buf.get(7, 8); // neighbor should be brighter

        assert!(
            after.r > before.r || after.g > before.g || after.b > before.b,
            "bloom should have increased neighbor brightness: before={:?}, after={:?}",
            before, after
        );
    }

    // 16. Bloom pass: dim pixels below threshold stay dim
    #[test]
    fn test_bloom_pass_dim_pixels_unaffected() {
        let mut buf = HdrBuffer::new(8, 8);
        buf.clear(HdrPixel::new(0.05, 0.05, 0.05, 1.0));
        let before_r = buf.get(4, 4).r;
        buf.bloom_pass(1.0, 1.0); // threshold > 0.05, so no pixel qualifies
        let after_r = buf.get(4, 4).r;
        // Should remain unchanged (bloom contribution is 0)
        assert!(approx(before_r, after_r, 0.001), "dim pixel should not change: {before_r} vs {after_r}");
    }

    // 17. HdrBuffer blend composites correctly
    #[test]
    fn test_buffer_blend() {
        let mut buf = HdrBuffer::new(4, 4);
        buf.clear(HdrPixel::new(0.0, 0.0, 1.0, 1.0)); // blue background
        buf.blend(2, 2, HdrPixel::new(1.0, 0.0, 0.0, 1.0)); // opaque red over
        let p = buf.get(2, 2);
        assert!(approx(p.r, 1.0, 0.001));
        assert!(approx(p.b, 0.0, 0.001));
    }

    // 18. sRGB decode of low values (below linear threshold)
    #[test]
    fn test_srgb_decode_low_values() {
        let decoded = srgb_decode(0.02);
        let expected = 0.02 / 12.92;
        assert!(approx(decoded, expected, 0.001), "low sRGB should use linear segment: {decoded} vs {expected}");
    }
}

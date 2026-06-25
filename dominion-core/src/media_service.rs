//! Media Service — zero-copy video stream management.
//!
//! When an application allocates a visual asset (decoding video stream,
//! high-performance canvas), memory is allocated from device-local physical heap.
//! Bound directly to the display engine's primary scanout via DRM KMS hardware
//! overlays. Decoders bypass the GPU composition pipeline entirely during
//! full-screen media playback.
//!
//! Zero-copy token: allocate → duplicate token → share with GPU → negotiate
//! unified layout → allocate shared buffer collection → direct GPU write /
//! display scanout read.
//!
//! Pure, safe `no_std`.

extern crate alloc;

use alloc::vec::Vec;
use alloc::collections::BTreeMap;

// ── PixelFormat ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Nv12,
    Yuv420,
    Rgba8,
    Rgba16F,
}

// ── ZeroCopyToken ─────────────────────────────────────────────────────────────

/// A zero-copy allocation token (represents shared physical memory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroCopyToken {
    pub id: u64,
    pub size_bytes: u64,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// true once negotiated with GPU
    pub is_shared: bool,
    /// true once bound to display overlay
    pub is_scanout_ready: bool,
}

impl ZeroCopyToken {
    pub fn allocate(id: u64, width: u32, height: u32, format: PixelFormat) -> Self {
        let size_bytes = Self::compute_size(width, height, format);
        Self {
            id,
            size_bytes,
            width,
            height,
            format,
            is_shared: false,
            is_scanout_ready: false,
        }
    }

    /// Negotiate unified layout with GPU — marks token as shared.
    pub fn share_with_gpu(&mut self) {
        self.is_shared = true;
    }

    /// Bind token to display scanout overlay.
    pub fn bind_scanout(&mut self) {
        self.is_scanout_ready = true;
    }

    /// Bytes per frame based on format and dimensions.
    pub fn bytes_per_frame(&self) -> u64 {
        Self::compute_size(self.width, self.height, self.format)
    }

    fn compute_size(width: u32, height: u32, format: PixelFormat) -> u64 {
        let pixels = (width as u64) * (height as u64);
        match format {
            PixelFormat::Nv12   => pixels + pixels / 2,  // Y plane + interleaved UV (1.5 bpp)
            PixelFormat::Yuv420 => pixels + pixels / 2,  // Y + U/4 + V/4 (1.5 bpp)
            PixelFormat::Rgba8  => pixels * 4,
            PixelFormat::Rgba16F => pixels * 8,
        }
    }
}

// ── VideoStreamBuffer ─────────────────────────────────────────────────────────

/// A video stream buffer — holds decoded frames in zero-copy memory.
pub struct VideoStreamBuffer {
    pub stream_id: u64,
    pub token: ZeroCopyToken,
    /// Decoded frame data (in no_std simulation — raw RGBA pixels per frame).
    frames: Vec<Vec<u32>>,
    pub current_frame: usize,
    pub frame_rate: u32,
    pub is_direct_overlay: bool,
}

impl VideoStreamBuffer {
    pub fn new(stream_id: u64, width: u32, height: u32, frame_rate: u32) -> Self {
        let token = ZeroCopyToken::allocate(stream_id, width, height, PixelFormat::Rgba8);
        Self {
            stream_id,
            token,
            frames: Vec::new(),
            current_frame: 0,
            frame_rate,
            is_direct_overlay: false,
        }
    }

    /// Push a decoded frame (simulate hardware decode writing directly to token).
    pub fn push_frame(&mut self, pixels: Vec<u32>) {
        self.frames.push(pixels);
    }

    /// Get the current frame for display.
    pub fn current_pixels(&self) -> Option<&[u32]> {
        self.frames.get(self.current_frame).map(|v| v.as_slice())
    }

    /// Advance to next frame (wraps around).
    pub fn advance(&mut self) {
        if !self.frames.is_empty() {
            self.current_frame = (self.current_frame + 1) % self.frames.len();
        }
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Enable direct overlay scanout (bypass compositor).
    pub fn enable_direct_overlay(&mut self) {
        self.is_direct_overlay = true;
        self.token.share_with_gpu();
        self.token.bind_scanout();
    }
}

// ── MediaService ──────────────────────────────────────────────────────────────

/// The media service — manages all active streams.
pub struct MediaService {
    pub streams: BTreeMap<u64, VideoStreamBuffer>,
    next_id: u64,
}

impl MediaService {
    pub fn new() -> Self {
        Self {
            streams: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Open a new video stream. Returns the stream_id.
    pub fn open_stream(&mut self, width: u32, height: u32, frame_rate: u32) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let buf = VideoStreamBuffer::new(id, width, height, frame_rate);
        self.streams.insert(id, buf);
        id
    }

    /// Close a stream, releasing its zero-copy allocation. Returns true if found.
    pub fn close_stream(&mut self, id: u64) -> bool {
        self.streams.remove(&id).is_some()
    }

    /// Push a decoded frame to a stream. Returns true if the stream exists.
    pub fn push_frame(&mut self, stream_id: u64, pixels: Vec<u32>) -> bool {
        if let Some(buf) = self.streams.get_mut(&stream_id) {
            buf.push_frame(pixels);
            true
        } else {
            false
        }
    }

    /// Get direct-overlay streams (these bypass compositor).
    pub fn direct_overlay_streams(&self) -> Vec<u64> {
        self.streams
            .values()
            .filter(|s| s.is_direct_overlay)
            .map(|s| s.stream_id)
            .collect()
    }

    /// Get composited streams (non-overlay).
    pub fn composited_streams(&self) -> Vec<u64> {
        self.streams
            .values()
            .filter(|s| !s.is_direct_overlay)
            .map(|s| s.stream_id)
            .collect()
    }

    /// Advance all streams by one frame.
    pub fn tick(&mut self) {
        for buf in self.streams.values_mut() {
            buf.advance();
        }
    }

    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

impl Default for MediaService {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_frame(w: u32, h: u32, color: u32) -> Vec<u32> {
        alloc::vec![color; (w * h) as usize]
    }

    // 1. open_stream returns incrementing IDs.
    #[test]
    fn test_open_stream_ids() {
        let mut svc = MediaService::new();
        let id1 = svc.open_stream(1920, 1080, 60);
        let id2 = svc.open_stream(1280, 720, 30);
        assert_ne!(id1, id2);
        assert_eq!(svc.stream_count(), 2);
    }

    // 2. close_stream returns true and removes the stream.
    #[test]
    fn test_close_stream() {
        let mut svc = MediaService::new();
        let id = svc.open_stream(1920, 1080, 60);
        assert!(svc.close_stream(id));
        assert_eq!(svc.stream_count(), 0);
    }

    // 3. close_stream on missing ID returns false.
    #[test]
    fn test_close_stream_missing() {
        let mut svc = MediaService::new();
        assert!(!svc.close_stream(999));
    }

    // 4. push_frame to valid stream returns true and frame is stored.
    #[test]
    fn test_push_frame_valid() {
        let mut svc = MediaService::new();
        let id = svc.open_stream(2, 2, 30);
        let ok = svc.push_frame(id, rgba_frame(2, 2, 0xFF0000FF));
        assert!(ok);
        assert_eq!(svc.streams[&id].frame_count(), 1);
    }

    // 5. push_frame to missing stream returns false.
    #[test]
    fn test_push_frame_missing() {
        let mut svc = MediaService::new();
        assert!(!svc.push_frame(999, rgba_frame(2, 2, 0)));
    }

    // 6. advance wraps around after last frame.
    #[test]
    fn test_advance_wraps() {
        let mut svc = MediaService::new();
        let id = svc.open_stream(2, 2, 30);
        svc.push_frame(id, rgba_frame(2, 2, 0xAABBCCDD));
        svc.push_frame(id, rgba_frame(2, 2, 0x11223344));
        {
            let buf = svc.streams.get_mut(&id).unwrap();
            assert_eq!(buf.current_frame, 0);
            buf.advance();
            assert_eq!(buf.current_frame, 1);
            buf.advance();
            assert_eq!(buf.current_frame, 0); // wrapped
        }
    }

    // 7. current_pixels returns the right frame data.
    #[test]
    fn test_current_pixels() {
        let mut buf = VideoStreamBuffer::new(1, 2, 2, 30);
        buf.push_frame(rgba_frame(2, 2, 0xAABBCCDD));
        buf.push_frame(rgba_frame(2, 2, 0x11223344));
        let pixels = buf.current_pixels().unwrap();
        assert_eq!(pixels[0], 0xAABBCCDD);
        buf.advance();
        let pixels2 = buf.current_pixels().unwrap();
        assert_eq!(pixels2[0], 0x11223344);
    }

    // 8. enable_direct_overlay sets flags and binds token.
    #[test]
    fn test_enable_direct_overlay() {
        let mut buf = VideoStreamBuffer::new(1, 1920, 1080, 60);
        assert!(!buf.is_direct_overlay);
        assert!(!buf.token.is_shared);
        assert!(!buf.token.is_scanout_ready);
        buf.enable_direct_overlay();
        assert!(buf.is_direct_overlay);
        assert!(buf.token.is_shared);
        assert!(buf.token.is_scanout_ready);
    }

    // 9. direct_overlay_streams vs composited_streams partition correctly.
    #[test]
    fn test_overlay_vs_composited_partition() {
        let mut svc = MediaService::new();
        let id1 = svc.open_stream(1920, 1080, 60);
        let id2 = svc.open_stream(1280, 720, 30);
        let id3 = svc.open_stream(640, 480, 24);
        // Make id1 a direct overlay.
        svc.streams.get_mut(&id1).unwrap().enable_direct_overlay();

        let overlays = svc.direct_overlay_streams();
        let composited = svc.composited_streams();
        assert_eq!(overlays.len(), 1);
        assert!(overlays.contains(&id1));
        assert_eq!(composited.len(), 2);
        assert!(composited.contains(&id2));
        assert!(composited.contains(&id3));
    }

    // 10. ZeroCopyToken bytes_per_frame for RGBA8.
    #[test]
    fn test_bytes_per_frame_rgba8() {
        let token = ZeroCopyToken::allocate(1, 1920, 1080, PixelFormat::Rgba8);
        assert_eq!(token.bytes_per_frame(), 1920 * 1080 * 4);
    }

    // 11. ZeroCopyToken bytes_per_frame for NV12 (1.5 bytes/pixel).
    #[test]
    fn test_bytes_per_frame_nv12() {
        let token = ZeroCopyToken::allocate(2, 1920, 1080, PixelFormat::Nv12);
        // NV12: Y plane + interleaved UV = W*H + W*H/2 = 1.5 * W * H
        assert_eq!(token.bytes_per_frame(), 1920 * 1080 * 3 / 2);
    }

    // 12. ZeroCopyToken bytes_per_frame for Rgba16F (8 bytes/pixel).
    #[test]
    fn test_bytes_per_frame_rgba16f() {
        let token = ZeroCopyToken::allocate(3, 1920, 1080, PixelFormat::Rgba16F);
        assert_eq!(token.bytes_per_frame(), 1920 * 1080 * 8);
    }

    // 13. share_with_gpu and bind_scanout transition token state.
    #[test]
    fn test_token_negotiate_flow() {
        let mut token = ZeroCopyToken::allocate(1, 1920, 1080, PixelFormat::Rgba8);
        assert!(!token.is_shared);
        token.share_with_gpu();
        assert!(token.is_shared);
        assert!(!token.is_scanout_ready);
        token.bind_scanout();
        assert!(token.is_scanout_ready);
    }

    // 14. tick advances all streams by one frame.
    #[test]
    fn test_tick_advances_all_streams() {
        let mut svc = MediaService::new();
        let id1 = svc.open_stream(2, 2, 30);
        let id2 = svc.open_stream(2, 2, 60);
        svc.push_frame(id1, rgba_frame(2, 2, 0xAABBCCDD));
        svc.push_frame(id1, rgba_frame(2, 2, 0x11223344));
        svc.push_frame(id2, rgba_frame(2, 2, 0xFF000000));
        svc.push_frame(id2, rgba_frame(2, 2, 0x00FF0000));
        assert_eq!(svc.streams[&id1].current_frame, 0);
        svc.tick();
        assert_eq!(svc.streams[&id1].current_frame, 1);
        assert_eq!(svc.streams[&id2].current_frame, 1);
    }

    // 15. current_pixels returns None when no frames have been pushed.
    #[test]
    fn test_current_pixels_empty() {
        let buf = VideoStreamBuffer::new(1, 1920, 1080, 60);
        assert!(buf.current_pixels().is_none());
    }
}

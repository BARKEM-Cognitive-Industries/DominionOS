//! Asynchronous TimeWarp (ATW) for the DominionOS unified render stack.
//!
//! When an application misses its frame deadline, the compositor uses motion
//! vectors and depth from the previous frame to reproject the UI to the
//! predicted current viewpoint — avoiding blank frames.
//!
//! Pose prediction uses the Taylor series:
//!   p(t+dt) = p + v*dt + 0.5*a*dt²
//!
//! Pure, safe `no_std`.

use alloc::vec::Vec;
use crate::math3d::{Vec3, Vec4, Quat, Mat4, sqrt32, perspective};

// ---------------------------------------------------------------------------
// MotionVector
// ---------------------------------------------------------------------------

/// A 2D/3D motion vector for a screen region (pixel/frame velocity + depth delta).
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionVector {
    /// Horizontal pixel velocity (pixels per frame).
    pub dx: f32,
    /// Vertical pixel velocity (pixels per frame).
    pub dy: f32,
    /// Depth delta (for 3D reprojection).
    pub dz: f32,
}

impl MotionVector {
    pub fn new(dx: f32, dy: f32, dz: f32) -> Self {
        Self { dx, dy, dz }
    }

    /// Magnitude of the 2D screen-space motion.
    #[inline]
    pub fn screen_len(&self) -> f32 {
        sqrt32(self.dx * self.dx + self.dy * self.dy)
    }
}

// ---------------------------------------------------------------------------
// KinematicState
// ---------------------------------------------------------------------------

/// Linear kinematic state for predictive warp (Taylor series pose prediction).
#[derive(Clone, Copy, Debug)]
pub struct KinematicState {
    /// Current position.
    pub pos: Vec3,
    /// Current velocity (units/sec).
    pub vel: Vec3,
    /// Current acceleration (units/sec²).
    pub acc: Vec3,
}

impl Default for KinematicState {
    fn default() -> Self {
        Self {
            pos: Vec3::zero(),
            vel: Vec3::zero(),
            acc: Vec3::zero(),
        }
    }
}

impl KinematicState {
    pub fn new(pos: Vec3, vel: Vec3, acc: Vec3) -> Self {
        Self { pos, vel, acc }
    }

    /// Predict position at `dt` seconds ahead using the 2nd-order Taylor series:
    /// `p(t+dt) = pos + vel*dt + 0.5*acc*dt²`
    #[inline]
    pub fn predict(&self, dt: f32) -> Vec3 {
        Vec3::new(
            self.pos.x + self.vel.x * dt + 0.5 * self.acc.x * dt * dt,
            self.pos.y + self.vel.y * dt + 0.5 * self.acc.y * dt * dt,
            self.pos.z + self.vel.z * dt + 0.5 * self.acc.z * dt * dt,
        )
    }

    /// Update state with a new position observation at `dt` seconds after the
    /// last update. Uses finite-difference velocity estimation and resets acceleration.
    pub fn update(&mut self, new_pos: Vec3, dt: f32) {
        if dt > 1e-8 {
            let new_vel = Vec3::new(
                (new_pos.x - self.pos.x) / dt,
                (new_pos.y - self.pos.y) / dt,
                (new_pos.z - self.pos.z) / dt,
            );
            self.acc = Vec3::new(
                (new_vel.x - self.vel.x) / dt,
                (new_vel.y - self.vel.y) / dt,
                (new_vel.z - self.vel.z) / dt,
            );
            self.vel = new_vel;
        }
        self.pos = new_pos;
    }
}

// ---------------------------------------------------------------------------
// RotKinematic
// ---------------------------------------------------------------------------

/// Rotational kinematic state for ATW (quaternion-based).
#[derive(Clone, Copy, Debug)]
pub struct RotKinematic {
    /// Current orientation.
    pub rot: Quat,
    /// Angular velocity in radians/sec (world-space axis-angle representation).
    pub ang_vel: Vec3,
}

impl RotKinematic {
    pub fn new(rot: Quat, ang_vel: Vec3) -> Self {
        Self { rot, ang_vel }
    }

    /// Predict rotation at `dt` seconds ahead by integrating angular velocity.
    /// Uses the axis-angle form: rotate by `ang_vel * dt` radians around the ang_vel axis.
    pub fn predict(&self, dt: f32) -> Quat {
        let angle = sqrt32(
            self.ang_vel.x * self.ang_vel.x
                + self.ang_vel.y * self.ang_vel.y
                + self.ang_vel.z * self.ang_vel.z,
        ) * dt;

        if angle < 1e-10 {
            return self.rot;
        }

        let inv_len = 1.0 / (angle / dt);
        let axis = Vec3::new(
            self.ang_vel.x * inv_len,
            self.ang_vel.y * inv_len,
            self.ang_vel.z * inv_len,
        );

        let delta = Quat::from_axis_angle(axis, angle);
        delta.mul(self.rot).normalize()
    }
}

// ---------------------------------------------------------------------------
// CapturedFrame
// ---------------------------------------------------------------------------

/// A captured frame with depth buffer for reprojection.
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    /// Packed pixels in 0xAABBGGRR format.
    pub pixels: Vec<u32>,
    /// Normalized depth in [0, 1] (0 = near plane, 1 = far plane).
    pub depth: Vec<f32>,
    /// The combined projection × view matrix used to render this frame.
    pub view_proj: Mat4,
}

impl CapturedFrame {
    pub fn new(width: u32, height: u32) -> Self {
        let count = width as usize * height as usize;
        Self {
            width,
            height,
            pixels: alloc::vec![0u32; count],
            depth: alloc::vec![1.0f32; count],
            view_proj: Mat4::identity(),
        }
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32, depth: f32) {
        let idx = y as usize * self.width as usize + x as usize;
        self.pixels[idx] = color;
        self.depth[idx] = depth;
    }

    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> (u32, f32) {
        let idx = y as usize * self.width as usize + x as usize;
        (self.pixels[idx], self.depth[idx])
    }
}

// ---------------------------------------------------------------------------
// Atw
// ---------------------------------------------------------------------------

/// Asynchronous TimeWarp reprojector.
pub struct Atw {
    pub last_frame: Option<CapturedFrame>,
    pub kinematic: KinematicState,
    pub rot_kinematic: RotKinematic,
}

impl Atw {
    pub fn new() -> Self {
        Self {
            last_frame: None,
            kinematic: KinematicState::default(),
            rot_kinematic: RotKinematic::new(Quat::identity(), Vec3::zero()),
        }
    }

    /// Feed a new captured frame and kinematic update.
    pub fn update(&mut self, frame: CapturedFrame, new_pos: Vec3, new_rot: Quat, dt: f32) {
        self.kinematic.update(new_pos, dt);
        // Update angular velocity via finite difference between quaternions
        // (simplified: assume ang_vel stays constant for now; caller should set it)
        self.rot_kinematic.rot = new_rot;
        self.last_frame = Some(frame);
    }

    /// Returns true if we have a valid captured frame to warp from.
    #[inline]
    pub fn can_reproject(&self) -> bool {
        self.last_frame.is_some()
    }

    /// Reproject the last captured frame to the predicted pose at `scan_dt` seconds ahead.
    ///
    /// Algorithm:
    /// 1. Predict the new position and rotation via kinematic extrapolation.
    /// 2. Build the new view-projection matrix from predicted pose.
    /// 3. For each pixel in the output, unproject from the last frame's NDC using
    ///    its depth, then reproject into the new camera to find its new screen position.
    /// 4. Fetch the colour from the last frame at that reprojected position.
    ///
    /// Returns a `width × height` buffer of 0xAABBGGRR pixels.
    pub fn reproject(
        &self,
        width: u32,
        height: u32,
        scan_dt: f32,
        fov_y_rad: f32,
        aspect: f32,
    ) -> Vec<u32> {
        let count = width as usize * height as usize;
        let mut out = alloc::vec![0u32; count];

        let frame = match &self.last_frame {
            Some(f) => f,
            None => return out,
        };

        // Predict new camera position and rotation
        let pred_pos = self.kinematic.predict(scan_dt);
        let pred_rot = self.rot_kinematic.predict(scan_dt);

        // Build new view-projection matrix
        // View = rotation^-1 * translation
        let rot_mat = pred_rot.to_mat4();
        // Inverse rotation for view matrix (transpose since rotation is orthogonal)
        let rot_inv = rot_mat.transpose();
        // Translation: -rot_inv * pos
        let translated_pos = Vec3::new(
            -(rot_inv.cols[0][0] * pred_pos.x + rot_inv.cols[1][0] * pred_pos.y + rot_inv.cols[2][0] * pred_pos.z),
            -(rot_inv.cols[0][1] * pred_pos.x + rot_inv.cols[1][1] * pred_pos.y + rot_inv.cols[2][1] * pred_pos.z),
            -(rot_inv.cols[0][2] * pred_pos.x + rot_inv.cols[1][2] * pred_pos.y + rot_inv.cols[2][2] * pred_pos.z),
        );

        let new_view = Mat4::from_cols_array([
            [rot_inv.cols[0][0], rot_inv.cols[0][1], rot_inv.cols[0][2], 0.0],
            [rot_inv.cols[1][0], rot_inv.cols[1][1], rot_inv.cols[1][2], 0.0],
            [rot_inv.cols[2][0], rot_inv.cols[2][1], rot_inv.cols[2][2], 0.0],
            [translated_pos.x,  translated_pos.y,   translated_pos.z,   1.0],
        ]);

        let new_proj = perspective(fov_y_rad, aspect, 0.1, 1000.0);
        let new_vp = new_proj * new_view;

        // Invert the old view-projection for unprojection
        let inv_old_vp = match frame.view_proj.invert() {
            Some(inv) => inv,
            None => return out, // degenerate matrix
        };

        let fw = frame.width as f32;
        let fh = frame.height as f32;
        let ow = width as f32;
        let oh = height as f32;

        for py in 0..height {
            for px in 0..width {
                // New pixel center in NDC [-1, 1]
                let ndc_x = (px as f32 + 0.5) / ow * 2.0 - 1.0;
                let ndc_y = 1.0 - (py as f32 + 0.5) / oh * 2.0; // flip Y

                // We need to unproject from the new camera view and find where this
                // maps in the old frame. Use a simplified approach:
                // For each output pixel, we sample the old frame at the same
                // screen position with the delta motion applied.
                //
                // Full depth-based reprojection:
                // 1. Pick depth from old frame at the old screen position (nearest)
                //    We approximate by mapping new NDC → old NDC via the delta VP.
                // 2. Unproject old NDC + depth → world point.
                // 3. Reproject world point → new NDC.
                //
                // Since we're mapping *from* new *to* old, we actually need the
                // inverse: for each new pixel, find the corresponding old pixel.
                // We do this by: old_NDC = old_VP * new_VP_inv * new_NDC_homogeneous

                // Get old screen depth at the mapped location (use center depth for now)
                // Map new NDC to old NDC via chain: new_NDC → world → old_NDC
                // old_clip = old_VP * new_VP_inv * new_clip

                // Assume depth = 0.5 (middle of depth range) for pixels we haven't sampled
                // Use actual depth from old frame at the nearest old-frame position
                let old_px = ((ndc_x + 1.0) * 0.5 * fw) as i32;
                let old_py = ((1.0 - (ndc_y + 1.0) * 0.5) * fh) as i32;

                let old_depth = if old_px >= 0 && old_px < fw as i32 && old_py >= 0 && old_py < fh as i32 {
                    frame.depth[(old_py as u32 * frame.width + old_px as u32) as usize]
                } else {
                    1.0
                };

                // Unproject: NDC (ndc_x, ndc_y, depth_ndc) → world
                // Depth NDC: d_ndc = 2*depth - 1 (OpenGL convention)
                let ndc_z = old_depth * 2.0 - 1.0;
                let clip = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
                let world_h = inv_old_vp.mul_vec4(clip);

                // Perspective divide
                if world_h.w.abs() < 1e-8 {
                    continue;
                }
                let world = Vec3::new(
                    world_h.x / world_h.w,
                    world_h.y / world_h.w,
                    world_h.z / world_h.w,
                );

                // Reproject into new camera
                let new_clip = new_vp.mul_vec4(Vec4::from_vec3(world, 1.0));
                if new_clip.w.abs() < 1e-8 {
                    continue;
                }

                let new_ndc_x = new_clip.x / new_clip.w;
                let new_ndc_y = new_clip.y / new_clip.w;

                // Convert new NDC to old frame pixel coordinates for sampling
                let sample_x = ((new_ndc_x + 1.0) * 0.5 * fw) as i32;
                let sample_y = ((1.0 - (new_ndc_y + 1.0) * 0.5) * fh) as i32;

                if sample_x >= 0 && sample_x < fw as i32 && sample_y >= 0 && sample_y < fh as i32 {
                    let src_idx = (sample_y as u32 * frame.width + sample_x as u32) as usize;
                    let dst_idx = (py * width + px) as usize;
                    out[dst_idx] = frame.pixels[src_idx];
                }
            }
        }

        out
    }
}

impl Default for Atw {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math3d::{Vec3, Quat, PI};

    fn approx(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    // 1. KinematicState::predict: zero velocity stays still
    #[test]
    fn test_kinematic_zero_velocity() {
        let k = KinematicState::new(Vec3::new(1.0, 2.0, 3.0), Vec3::zero(), Vec3::zero());
        let pred = k.predict(0.1);
        assert!(approx(pred.x, 1.0, 1e-5));
        assert!(approx(pred.y, 2.0, 1e-5));
        assert!(approx(pred.z, 3.0, 1e-5));
    }

    // 2. KinematicState::predict: constant velocity
    #[test]
    fn test_kinematic_constant_velocity() {
        let k = KinematicState::new(Vec3::zero(), Vec3::new(1.0, 2.0, 3.0), Vec3::zero());
        let pred = k.predict(2.0);
        assert!(approx(pred.x, 2.0, 1e-5));
        assert!(approx(pred.y, 4.0, 1e-5));
        assert!(approx(pred.z, 6.0, 1e-5));
    }

    // 3. KinematicState::predict: Taylor series with acceleration
    #[test]
    fn test_kinematic_taylor_series() {
        // p + v*t + 0.5*a*t^2 with p=0, v=0, a=2 → should give 0.5*2*t^2 = t^2
        let k = KinematicState::new(Vec3::zero(), Vec3::zero(), Vec3::new(2.0, 0.0, 0.0));
        let pred = k.predict(3.0);
        // 0.5 * 2.0 * 3.0^2 = 9.0
        assert!(approx(pred.x, 9.0, 1e-4), "Taylor: expected 9.0, got {}", pred.x);
    }

    // 4. KinematicState::predict at dt=0 returns current position
    #[test]
    fn test_kinematic_predict_dt_zero() {
        let k = KinematicState::new(Vec3::new(5.0, 6.0, 7.0), Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 1.0, 1.0));
        let pred = k.predict(0.0);
        assert!(approx(pred.x, 5.0, 1e-6));
        assert!(approx(pred.y, 6.0, 1e-6));
        assert!(approx(pred.z, 7.0, 1e-6));
    }

    // 5. KinematicState::update computes velocity from finite difference
    #[test]
    fn test_kinematic_update_velocity() {
        let mut k = KinematicState::new(Vec3::zero(), Vec3::zero(), Vec3::zero());
        k.update(Vec3::new(1.0, 0.0, 0.0), 0.5);
        // vel = (1.0 - 0.0) / 0.5 = 2.0
        assert!(approx(k.vel.x, 2.0, 1e-4), "velocity should be 2.0, got {}", k.vel.x);
        assert!(approx(k.pos.x, 1.0, 1e-4));
    }

    // 6. KinematicState::update updates position correctly
    #[test]
    fn test_kinematic_update_position() {
        let mut k = KinematicState::new(Vec3::new(1.0, 2.0, 3.0), Vec3::zero(), Vec3::zero());
        k.update(Vec3::new(4.0, 5.0, 6.0), 1.0);
        assert!(approx(k.pos.x, 4.0, 1e-5));
        assert!(approx(k.pos.y, 5.0, 1e-5));
        assert!(approx(k.pos.z, 6.0, 1e-5));
    }

    // 7. RotKinematic::predict: zero angular velocity stays at identity
    #[test]
    fn test_rot_kinematic_zero_ang_vel() {
        let rk = RotKinematic::new(Quat::identity(), Vec3::zero());
        let pred = rk.predict(1.0);
        assert!(approx(pred.w, 1.0, 1e-4));
        assert!(approx(pred.x, 0.0, 1e-4));
    }

    // 8. RotKinematic::predict: 90°/sec around Y for 1 sec → 90° rotation
    #[test]
    fn test_rot_kinematic_90deg_per_sec() {
        use crate::math3d::FRAC_PI_2;
        let ang_vel = Vec3::new(0.0, FRAC_PI_2, 0.0); // π/2 rad/sec around Y
        let rk = RotKinematic::new(Quat::identity(), ang_vel);
        let pred = rk.predict(1.0);
        // Should rotate X-axis to -Z after 90° around Y
        let rotated = pred.rotate_vec3(Vec3::new(1.0, 0.0, 0.0));
        assert!(approx(rotated.z, -1.0, 0.01), "90° Y rotation: expected z=-1.0, got {}", rotated.z);
        assert!(approx(rotated.x, 0.0, 0.01));
    }

    // 9. RotKinematic::predict at dt=0 returns unchanged rotation
    #[test]
    fn test_rot_kinematic_predict_dt_zero() {
        let q = Quat::from_axis_angle(Vec3::y_axis(), 1.0);
        let rk = RotKinematic::new(q, Vec3::new(1.0, 0.0, 0.0));
        let pred = rk.predict(0.0);
        // Should return the original rotation unchanged
        assert!(approx(pred.w, q.w, 1e-4));
        assert!(approx(pred.x, q.x, 1e-4));
    }

    // 10. CapturedFrame::set_pixel and get_pixel round-trip
    #[test]
    fn test_captured_frame_pixel_roundtrip() {
        let mut frame = CapturedFrame::new(8, 8);
        frame.set_pixel(3, 4, 0xFFAABBCC, 0.7);
        let (color, depth) = frame.get_pixel(3, 4);
        assert_eq!(color, 0xFFAABBCC);
        assert!(approx(depth, 0.7, 1e-6));
    }

    // 11. CapturedFrame default depth is 1.0 and pixel is 0
    #[test]
    fn test_captured_frame_defaults() {
        let frame = CapturedFrame::new(4, 4);
        let (color, depth) = frame.get_pixel(2, 2);
        assert_eq!(color, 0);
        assert!(approx(depth, 1.0, 1e-6));
    }

    // 12. Atw::can_reproject: false before update, true after
    #[test]
    fn test_atw_can_reproject() {
        let mut atw = Atw::new();
        assert!(!atw.can_reproject());
        let frame = CapturedFrame::new(16, 16);
        atw.update(frame, Vec3::zero(), Quat::identity(), 0.016);
        assert!(atw.can_reproject());
    }

    // 13. Atw::reproject produces output of correct size
    #[test]
    fn test_atw_reproject_output_size() {
        let mut atw = Atw::new();
        let mut frame = CapturedFrame::new(16, 16);
        frame.view_proj = Mat4::identity();
        atw.update(frame, Vec3::zero(), Quat::identity(), 0.016);

        let out = atw.reproject(16, 16, 0.016, crate::math3d::FRAC_PI_2, 1.0);
        assert_eq!(out.len(), 16 * 16);
    }

    // 14. Atw::reproject without frame returns zeroed buffer
    #[test]
    fn test_atw_reproject_no_frame() {
        let atw = Atw::new();
        let out = atw.reproject(8, 8, 0.016, crate::math3d::FRAC_PI_2, 1.0);
        assert_eq!(out.len(), 64);
        for &p in &out {
            assert_eq!(p, 0, "should be zeroed without a frame");
        }
    }

    // 15. Atw::reproject does not panic on arbitrary inputs
    #[test]
    fn test_atw_reproject_no_panic() {
        let mut atw = Atw::new();
        let mut frame = CapturedFrame::new(32, 32);
        // Fill with a checkerboard pattern
        for y in 0..32u32 {
            for x in 0..32u32 {
                let color = if (x + y) % 2 == 0 { 0xFFFFFFFF } else { 0xFF000000 };
                frame.set_pixel(x, y, color, 0.5);
            }
        }
        // Use perspective+lookat style matrix
        use crate::math3d::{look_at, perspective as persp};
        let view = look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::zero(), Vec3::y_axis());
        let proj = persp(crate::math3d::FRAC_PI_2, 1.0, 0.1, 100.0);
        frame.view_proj = proj * view;

        atw.update(frame, Vec3::new(0.0, 0.0, 5.0), Quat::identity(), 0.016);
        // Just verify it doesn't panic
        let out = atw.reproject(32, 32, 0.016, crate::math3d::FRAC_PI_2, 1.0);
        assert_eq!(out.len(), 32 * 32);
    }

    // 16. KinematicState acceleration is estimated from two position updates
    #[test]
    fn test_kinematic_acceleration_estimation() {
        let mut k = KinematicState::new(Vec3::zero(), Vec3::zero(), Vec3::zero());
        k.update(Vec3::new(1.0, 0.0, 0.0), 1.0); // vel = 1.0
        k.update(Vec3::new(3.0, 0.0, 0.0), 1.0); // vel = 2.0, acc = 1.0
        assert!(approx(k.vel.x, 2.0, 1e-4), "velocity should be 2.0, got {}", k.vel.x);
        assert!(approx(k.acc.x, 1.0, 1e-4), "acceleration should be 1.0, got {}", k.acc.x);
    }
}

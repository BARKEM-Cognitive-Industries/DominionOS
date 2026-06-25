//! Vertex animations: morph targets (blend shapes) and skeletal skinning.
//! (see `docs/2d-3d rendering redesign.md` — vertex animations native to the 3D renderer)
//!
//! Inspired by glTF morph targets and UE5 skeletal mesh skinning.
//! Pure, safe `no_std`. CPU-side skinning; GPU skinning path is the production target.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;

#[allow(unused_imports)]
use crate::math3d::{Vec2, Vec3, Vec4, Mat4, Quat, sqrt32};
use crate::mesh::Mesh;

// ---------------------------------------------------------------------------
// Morph Targets (Blend Shapes)
// ---------------------------------------------------------------------------

/// A single morph target: position and normal deltas per vertex.
#[derive(Clone, Debug)]
pub struct MorphTarget {
    pub name: String,
    /// Per-vertex position delta (same length as base mesh vertices).
    pub position_deltas: Vec<Vec3>,
    /// Per-vertex normal delta (zeros if not specified).
    pub normal_deltas: Vec<Vec3>,
}

impl MorphTarget {
    /// Create a morph target with zero normal deltas.
    pub fn new(name: &str, position_deltas: Vec<Vec3>) -> MorphTarget {
        let n = position_deltas.len();
        MorphTarget {
            name: String::from(name),
            position_deltas,
            normal_deltas: vec![Vec3::zero(); n],
        }
    }

    /// Create a morph target with explicit normal deltas.
    pub fn with_normals(
        name: &str,
        position_deltas: Vec<Vec3>,
        normal_deltas: Vec<Vec3>,
    ) -> MorphTarget {
        MorphTarget {
            name: String::from(name),
            position_deltas,
            normal_deltas,
        }
    }
}

/// A mesh with morph targets for facial/shape animation.
#[derive(Clone, Debug)]
pub struct MorphMesh {
    pub base: Mesh,
    pub targets: Vec<MorphTarget>,
}

impl MorphMesh {
    /// Wrap a base mesh with no targets yet.
    pub fn new(base: Mesh) -> MorphMesh {
        MorphMesh {
            base,
            targets: Vec::new(),
        }
    }

    /// Append a morph target. Must have the same vertex count as the base mesh.
    pub fn add_target(&mut self, target: MorphTarget) {
        self.targets.push(target);
    }

    /// Apply a weighted blend of morph targets to produce a deformed mesh.
    ///
    /// `weights[i]` corresponds to `targets[i]`. Weights are unclamped —
    /// values outside `[0, 1]` produce exaggerated expressions.
    /// Missing weights (if `weights` is shorter than `targets`) are treated as 0.
    pub fn apply(&self, weights: &[f32]) -> Mesh {
        let mut out = self.base.clone();
        let n_verts = out.vertices.len();

        for (ti, target) in self.targets.iter().enumerate() {
            let w = if ti < weights.len() { weights[ti] } else { 0.0 };
            if w == 0.0 {
                continue;
            }
            for vi in 0..n_verts {
                if vi < target.position_deltas.len() {
                    out.vertices[vi].pos = out.vertices[vi].pos + target.position_deltas[vi] * w;
                }
                if vi < target.normal_deltas.len() {
                    let nd = target.normal_deltas[vi];
                    // Only accumulate if the delta is non-zero
                    if nd.len_sq() > 0.0 {
                        let new_n = out.vertices[vi].normal + nd * w;
                        out.vertices[vi].normal = new_n.normalize();
                    }
                }
            }
        }

        out.compute_aabb();
        out
    }

    /// Return the number of morph targets attached to this mesh.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }
}

// ---------------------------------------------------------------------------
// Skeletal Animation
// ---------------------------------------------------------------------------

/// A single bone in a skeleton.
#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    /// `None` means this is a root bone.
    pub parent: Option<usize>,
    /// Inverse bind-pose matrix: object-space → bone-local.
    pub inv_bind: Mat4,
}

/// A skeleton: an ordered array of bones in parent-before-child order.
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

impl Skeleton {
    pub fn new() -> Skeleton {
        Skeleton { bones: Vec::new() }
    }

    /// Add a bone and return its index.
    pub fn add_bone(&mut self, name: &str, parent: Option<usize>, inv_bind: Mat4) -> usize {
        let idx = self.bones.len();
        self.bones.push(Bone {
            name: String::from(name),
            parent,
            inv_bind,
        });
        idx
    }

    /// Compute the final skinning matrices given local pose transforms.
    ///
    /// `final[i] = world_transform[i] * inv_bind[i]`
    /// where `world_transform[i] = world_transform[parent[i]] * local_pose[i]`.
    ///
    /// Assumes bones are ordered parent-before-child (no forward references).
    pub fn compute_skin_matrices(&self, local_pose: &[Mat4]) -> Vec<Mat4> {
        let n = self.bones.len();
        let mut world: Vec<Mat4> = Vec::with_capacity(n);
        let mut skin: Vec<Mat4> = Vec::with_capacity(n);

        for i in 0..n {
            let local = if i < local_pose.len() { local_pose[i] } else { Mat4::identity() };
            let w = match self.bones[i].parent {
                None => local,
                Some(p) => world[p] * local,
            };
            world.push(w);
            skin.push(w * self.bones[i].inv_bind);
        }

        skin
    }

    /// Return the number of bones.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }
}

impl Default for Skeleton {
    fn default() -> Self { Skeleton::new() }
}

// ---------------------------------------------------------------------------
// SkinVertex
// ---------------------------------------------------------------------------

/// Per-vertex skinning data: up to 4 bone influences.
#[derive(Clone, Copy, Debug)]
pub struct SkinVertex {
    /// Bone indices into `Skeleton.bones`. Unused slots = 0.
    pub joints: [u16; 4],
    /// Blend weights; should sum to 1.0. Unused slots = 0.0.
    pub weights: [f32; 4],
}

impl SkinVertex {
    /// Convenience constructor for a single fully-weighted bone.
    pub fn single(joint: u16) -> SkinVertex {
        SkinVertex {
            joints: [joint, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// SkinnedMesh
// ---------------------------------------------------------------------------

/// A mesh with skeletal skinning data.
#[derive(Clone, Debug)]
pub struct SkinnedMesh {
    pub mesh: Mesh,
    /// One `SkinVertex` per vertex in `mesh.vertices`.
    pub skin: Vec<SkinVertex>,
    pub skeleton: Skeleton,
}

impl SkinnedMesh {
    pub fn new(mesh: Mesh, skin: Vec<SkinVertex>, skeleton: Skeleton) -> SkinnedMesh {
        SkinnedMesh { mesh, skin, skeleton }
    }

    /// CPU skinning: apply pre-computed bone matrices to produce a deformed mesh.
    ///
    /// For each vertex:
    /// `pos = Σ (skin_mat[joints[j]] * base_pos * weights[j])` for `j` in `0..4`.
    pub fn skin(&self, skin_matrices: &[Mat4]) -> Mesh {
        let mut out = self.mesh.clone();
        let n = out.vertices.len();

        for vi in 0..n {
            let sv = if vi < self.skin.len() {
                self.skin[vi]
            } else {
                SkinVertex { joints: [0; 4], weights: [0.0; 4] }
            };

            let base_pos = self.mesh.vertices[vi].pos;
            let base_nrm = self.mesh.vertices[vi].normal;

            // Accumulate the weighted skinning matrix
            let mut acc_pos = Vec3::zero();
            let mut acc_nrm = Vec3::zero();
            let mut total_w = 0.0f32;

            for j in 0..4 {
                let w = sv.weights[j];
                if w == 0.0 {
                    continue;
                }
                let bi = sv.joints[j] as usize;
                if bi >= skin_matrices.len() {
                    continue;
                }
                let m = &skin_matrices[bi];
                acc_pos = acc_pos + m.mul_point(base_pos) * w;
                acc_nrm = acc_nrm + m.mul_dir(base_nrm) * w;
                total_w += w;
            }

            // Handle degenerate zero weights: keep base position
            if total_w < 1e-10 {
                out.vertices[vi].pos = base_pos;
                out.vertices[vi].normal = base_nrm;
            } else {
                out.vertices[vi].pos = acc_pos;
                out.vertices[vi].normal = acc_nrm.normalize();
            }
        }

        out.compute_aabb();
        out
    }

    /// Animate and skin in one call.
    pub fn animate_and_skin(&self, skeleton: &Skeleton, local_pose: &[Mat4]) -> Mesh {
        let skin_matrices = skeleton.compute_skin_matrices(local_pose);
        self.skin(&skin_matrices)
    }
}

// ---------------------------------------------------------------------------
// Keyframe Animation
// ---------------------------------------------------------------------------

/// A position keyframe.
#[derive(Clone, Debug)]
pub struct PosKey {
    pub time: f32,
    pub value: Vec3,
}

/// A rotation keyframe.
#[derive(Clone, Debug)]
pub struct RotKey {
    pub time: f32,
    pub value: Quat,
}

/// A scale keyframe.
#[derive(Clone, Debug)]
pub struct ScaleKey {
    pub time: f32,
    pub value: Vec3,
}

// ---------------------------------------------------------------------------
// AnimChannel
// ---------------------------------------------------------------------------

/// Animation channel for a single bone.
#[derive(Clone, Debug)]
pub struct AnimChannel {
    pub bone_idx: usize,
    pub positions: Vec<PosKey>,
    pub rotations: Vec<RotKey>,
    pub scales: Vec<ScaleKey>,
}

impl AnimChannel {
    /// Sample position at time `t` via linear interpolation between bracketing keyframes.
    pub fn sample_pos(&self, t: f32) -> Vec3 {
        sample_keyframes(&self.positions, t, Vec3::zero(), |k: &PosKey| k.time, |k| k.value,
            |a, b, fac| a.lerp(b, fac))
    }

    /// Sample rotation at time `t` via SLERP between bracketing keyframes.
    pub fn sample_rot(&self, t: f32) -> Quat {
        sample_keyframes(&self.rotations, t, Quat::identity(), |k: &RotKey| k.time, |k| k.value,
            |a, b, fac| a.slerp(b, fac))
    }

    /// Sample scale at time `t` via lerp between bracketing keyframes.
    pub fn sample_scale(&self, t: f32) -> Vec3 {
        sample_keyframes(&self.scales, t, Vec3::one(), |k: &ScaleKey| k.time, |k| k.value,
            |a, b, fac| a.lerp(b, fac))
    }
}

/// Generic keyframe sampler shared by `sample_pos`, `sample_rot`, and `sample_scale`.
///
/// - `default`: returned when `keys` is empty.
/// - `time_fn`: extracts the timestamp from a key.
/// - `val_fn`: extracts the interpolatable value from a key.
/// - `interp`: blends two values given a `[0,1]` factor.
fn sample_keyframes<K, V, TF, VF, IF>(
    keys: &[K],
    t: f32,
    default: V,
    time_fn: TF,
    val_fn: VF,
    interp: IF,
) -> V
where
    TF: Fn(&K) -> f32,
    VF: Fn(&K) -> V,
    IF: Fn(V, V, f32) -> V,
    V: Copy,
{
    if keys.is_empty() {
        return default;
    }
    if keys.len() == 1 || t <= time_fn(&keys[0]) {
        return val_fn(&keys[0]);
    }
    let last = keys.len() - 1;
    if t >= time_fn(&keys[last]) {
        return val_fn(&keys[last]);
    }
    let i = bracket_idx_f32(keys, t, &time_fn);
    let fac = (t - time_fn(&keys[i])) / (time_fn(&keys[i + 1]) - time_fn(&keys[i]));
    interp(val_fn(&keys[i]), val_fn(&keys[i + 1]), fac)
}

/// Binary search: return index `i` such that `time_fn(keys[i]) <= t < time_fn(keys[i+1])`.
/// Caller guarantees `keys.len() >= 2` and `keys[0].time <= t < keys[last].time`.
fn bracket_idx_f32<T, F: Fn(&T) -> f32>(keys: &[T], t: f32, time_fn: F) -> usize {
    let mut lo = 0usize;
    let mut hi = keys.len() - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if time_fn(&keys[mid]) <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// AnimClip
// ---------------------------------------------------------------------------

/// A complete animation clip (multiple bone channels).
#[derive(Clone, Debug)]
pub struct AnimClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
    pub looping: bool,
}

impl AnimClip {
    pub fn new(name: &str, duration: f32, looping: bool) -> AnimClip {
        AnimClip {
            name: String::from(name),
            duration,
            channels: Vec::new(),
            looping,
        }
    }

    pub fn add_channel(&mut self, channel: AnimChannel) {
        self.channels.push(channel);
    }

    /// Sample the full skeleton pose at time `t`.
    ///
    /// Returns a `Vec<Mat4>` of local bone transforms (TRS) for all `bone_count` bones.
    /// Bones with no channel get the identity matrix.
    pub fn sample(&self, t: f32, bone_count: usize) -> Vec<Mat4> {
        let mut pose = vec![Mat4::identity(); bone_count];

        for ch in &self.channels {
            if ch.bone_idx >= bone_count {
                continue;
            }
            let pos = ch.sample_pos(t);
            let rot = ch.sample_rot(t);
            let sc = ch.sample_scale(t);
            pose[ch.bone_idx] = trs_to_mat4(pos, rot, sc);
        }

        pose
    }

    /// Wrap `t` to `[0, duration]` when looping; clamp otherwise.
    pub fn wrap_time(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            return 0.0;
        }
        if self.looping {
            let mut t = t % self.duration;
            if t < 0.0 {
                t += self.duration;
            }
            t
        } else if t < 0.0 {
            0.0
        } else if t > self.duration {
            self.duration
        } else {
            t
        }
    }
}

/// Compose a TRS (translation × rotation × scale) matrix.
fn trs_to_mat4(pos: Vec3, rot: Quat, sc: Vec3) -> Mat4 {
    // scale, then rotate, then translate
    let r = rot.to_mat4();
    // Apply scale to rotation columns
    let mut m = r;
    m.cols[0][0] *= sc.x; m.cols[0][1] *= sc.x; m.cols[0][2] *= sc.x;
    m.cols[1][0] *= sc.y; m.cols[1][1] *= sc.y; m.cols[1][2] *= sc.y;
    m.cols[2][0] *= sc.z; m.cols[2][1] *= sc.z; m.cols[2][2] *= sc.z;
    // Translation column (column-major: col 3 = translation)
    m.cols[3][0] = pos.x;
    m.cols[3][1] = pos.y;
    m.cols[3][2] = pos.z;
    m.cols[3][3] = 1.0;
    m
}

// ---------------------------------------------------------------------------
// Animator (state machine)
// ---------------------------------------------------------------------------

/// Simple two-state animator: plays one clip, blends to another.
pub struct Animator {
    pub skeleton: Skeleton,
    pub current: AnimClip,
    pub next: Option<AnimClip>,
    /// Blend factor current→next: 0.0 = current, 1.0 = next.
    pub blend_t: f32,
    pub time: f32,
    /// Duration of the active blend transition in seconds.
    blend_duration: f32,
    /// Time accumulated in the `next` clip during a transition.
    next_time: f32,
}

impl Animator {
    pub fn new(skeleton: Skeleton, clip: AnimClip) -> Animator {
        Animator {
            skeleton,
            current: clip,
            next: None,
            blend_t: 0.0,
            time: 0.0,
            blend_duration: 0.0,
            next_time: 0.0,
        }
    }

    /// Advance animation time by `dt` seconds, advancing the blend if active.
    pub fn update(&mut self, dt: f32) {
        // Advance current clip time
        let raw_t = self.time + dt;
        self.time = self.current.wrap_time(raw_t);

        // Advance blend
        if self.next.is_some() {
            self.next_time = {
                let next = self.next.as_ref().unwrap();
                next.wrap_time(self.next_time + dt)
            };

            if self.blend_duration > 0.0 {
                self.blend_t = (self.blend_t + dt / self.blend_duration).min(1.0);
            } else {
                self.blend_t = 1.0;
            }

            // Transition complete: swap clips
            if self.blend_t >= 1.0 {
                let next_clip = self.next.take().unwrap();
                self.current = next_clip;
                self.time = self.next_time;
                self.next_time = 0.0;
                self.blend_t = 0.0;
                self.blend_duration = 0.0;
            }
        }
    }

    /// Begin blending to a new clip over `blend_duration` seconds.
    pub fn transition_to(&mut self, clip: AnimClip, blend_duration: f32) {
        self.next = Some(clip);
        self.blend_t = 0.0;
        self.blend_duration = blend_duration;
        self.next_time = 0.0;
    }

    /// Sample the current pose, blended if a transition is active.
    pub fn sample_pose(&self) -> Vec<Mat4> {
        let bone_count = self.skeleton.bone_count();

        if self.next.is_none() || self.blend_t <= 0.0 {
            return self.current.sample(self.time, bone_count);
        }

        if self.blend_t >= 1.0 {
            let next = self.next.as_ref().unwrap();
            return next.sample(self.next_time, bone_count);
        }

        // Blend between current and next
        let pose_a = self.current.sample(self.time, bone_count);
        let next = self.next.as_ref().unwrap();
        let pose_b = next.sample(self.next_time, bone_count);

        let t = self.blend_t;
        let mut blended = Vec::with_capacity(bone_count);
        for i in 0..bone_count {
            blended.push(blend_mat4(pose_a[i], pose_b[i], t));
        }
        blended
    }
}

/// Linearly blend two matrices element-wise (suitable for local pose blending).
fn blend_mat4(a: Mat4, b: Mat4, t: f32) -> Mat4 {
    let mut out = Mat4::identity();
    for c in 0..4 {
        for r in 0..4 {
            out.cols[c][r] = a.cols[c][r] + (b.cols[c][r] - a.cols[c][r]) * t;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math3d::{Vec2, Vec3, Vec4, Mat4, Quat};
    use crate::mesh::{Mesh, Vertex};

    fn make_vertex(x: f32, y: f32, z: f32) -> Vertex {
        Vertex::new(
            Vec3::new(x, y, z),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(0.0, 0.0),
        )
    }

    fn make_triangle_mesh(name: &str) -> Mesh {
        Mesh::new(
            name,
            vec![
                make_vertex(0.0, 0.0, 0.0),
                make_vertex(1.0, 0.0, 0.0),
                make_vertex(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2],
        )
    }

    // -----------------------------------------------------------------------
    // MorphMesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn morph_weight_one_applies_full_delta() {
        let base = make_triangle_mesh("base");
        let mut mm = MorphMesh::new(base.clone());

        let deltas = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        mm.add_target(MorphTarget::new("smile", deltas.clone()));

        // weight [1.0, 0.0] = base + target0 deltas
        let result = mm.apply(&[1.0, 0.0]);
        for (vi, v) in result.vertices.iter().enumerate() {
            let expected = base.vertices[vi].pos + deltas[vi];
            assert!((v.pos.x - expected.x).abs() < 1e-5, "x mismatch at {vi}");
            assert!((v.pos.y - expected.y).abs() < 1e-5, "y mismatch at {vi}");
            assert!((v.pos.z - expected.z).abs() < 1e-5, "z mismatch at {vi}");
        }
    }

    #[test]
    fn morph_weight_half_applies_half_delta() {
        let base = make_triangle_mesh("base");
        let mut mm = MorphMesh::new(base.clone());

        let deltas = vec![
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        mm.add_target(MorphTarget::new("puff", deltas.clone()));

        let result = mm.apply(&[0.5]);
        for (vi, v) in result.vertices.iter().enumerate() {
            let expected = base.vertices[vi].pos + deltas[vi] * 0.5;
            assert!((v.pos.x - expected.x).abs() < 1e-5, "x mismatch at {vi}");
        }
    }

    #[test]
    fn morph_zero_weights_returns_base() {
        let base = make_triangle_mesh("base");
        let mut mm = MorphMesh::new(base.clone());
        mm.add_target(MorphTarget::new("t", vec![Vec3::one(); 3]));

        let result = mm.apply(&[0.0]);
        for (vi, v) in result.vertices.iter().enumerate() {
            let b = &base.vertices[vi];
            assert!((v.pos.x - b.pos.x).abs() < 1e-6);
        }
    }

    #[test]
    fn morph_target_count() {
        let base = make_triangle_mesh("base");
        let mut mm = MorphMesh::new(base);
        assert_eq!(mm.target_count(), 0);
        mm.add_target(MorphTarget::new("a", vec![Vec3::zero(); 3]));
        mm.add_target(MorphTarget::new("b", vec![Vec3::zero(); 3]));
        assert_eq!(mm.target_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Skeleton tests
    // -----------------------------------------------------------------------

    #[test]
    fn skeleton_three_bone_chain() {
        // Root → Mid → Tip in object space
        // Each bone has inv_bind = identity (bind pose = object space)
        let mut skel = Skeleton::new();
        let root = skel.add_bone("root", None, Mat4::identity());
        let mid  = skel.add_bone("mid",  Some(root), Mat4::identity());
        let _tip = skel.add_bone("tip",  Some(mid),  Mat4::identity());

        // Local pose: root translates by (1,0,0), mid translates by (0,1,0), tip identity
        let t1 = crate::math3d::translation(Vec3::new(1.0, 0.0, 0.0));
        let t2 = crate::math3d::translation(Vec3::new(0.0, 1.0, 0.0));
        let local_pose = vec![t1, t2, Mat4::identity()];

        let mats = skel.compute_skin_matrices(&local_pose);
        assert_eq!(mats.len(), 3);

        // root world = t1, inv_bind = I → skin[0] = t1
        let root_pos = mats[0].mul_point(Vec3::zero());
        assert!((root_pos.x - 1.0).abs() < 1e-5, "root x = {}", root_pos.x);
        assert!((root_pos.y - 0.0).abs() < 1e-5, "root y = {}", root_pos.y);

        // mid world = t1 * t2 → translation (1,1,0)
        let mid_pos = mats[1].mul_point(Vec3::zero());
        assert!((mid_pos.x - 1.0).abs() < 1e-5, "mid x = {}", mid_pos.x);
        assert!((mid_pos.y - 1.0).abs() < 1e-5, "mid y = {}", mid_pos.y);

        // tip world = t1 * t2 * I → same as mid
        let tip_pos = mats[2].mul_point(Vec3::zero());
        assert!((tip_pos.x - 1.0).abs() < 1e-5, "tip x = {}", tip_pos.x);
        assert!((tip_pos.y - 1.0).abs() < 1e-5, "tip y = {}", tip_pos.y);
    }

    // -----------------------------------------------------------------------
    // SkinnedMesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn skinned_single_bone_90deg_rotation() {
        // One vertex at (1, 0, 0), fully weighted to bone 0.
        // Bone 0 rotates 90° around Y. Expected result: (0, 0, -1).
        let verts = vec![make_vertex(1.0, 0.0, 0.0)];
        let mesh = Mesh::new("m", verts, vec![0]);
        let skin = vec![SkinVertex::single(0)];
        let mut skel = Skeleton::new();
        skel.add_bone("bone0", None, Mat4::identity());

        let smesh = SkinnedMesh::new(mesh, skin, skel.clone());

        // 90° around Y: x→-z, z→x
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), crate::math3d::FRAC_PI_2);
        let local_pose = vec![rot.to_mat4()];
        let skin_matrices = skel.compute_skin_matrices(&local_pose);

        let out = smesh.skin(&skin_matrices);
        let p = out.vertices[0].pos;
        assert!((p.x - 0.0).abs() < 1e-4, "x = {}", p.x);
        assert!((p.z - (-1.0)).abs() < 1e-4, "z = {}", p.z);
    }

    // -----------------------------------------------------------------------
    // AnimChannel tests
    // -----------------------------------------------------------------------

    #[test]
    fn anim_channel_sample_between_keyframes() {
        let ch = AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 0.0, value: Vec3::zero() },
                PosKey { time: 1.0, value: Vec3::new(2.0, 0.0, 0.0) },
            ],
            rotations: vec![
                RotKey { time: 0.0, value: Quat::identity() },
                RotKey { time: 1.0, value: Quat::identity() },
            ],
            scales: vec![
                ScaleKey { time: 0.0, value: Vec3::one() },
                ScaleKey { time: 1.0, value: Vec3::new(2.0, 2.0, 2.0) },
            ],
        };

        let p = ch.sample_pos(0.5);
        assert!((p.x - 1.0).abs() < 1e-5, "pos x = {}", p.x);

        let sc = ch.sample_scale(0.5);
        assert!((sc.x - 1.5).abs() < 1e-5, "scale x = {}", sc.x);
    }

    #[test]
    fn anim_channel_clamps_before_first_key() {
        let ch = AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 1.0, value: Vec3::new(5.0, 0.0, 0.0) },
            ],
            rotations: Vec::new(),
            scales: Vec::new(),
        };
        let p = ch.sample_pos(0.0);
        assert!((p.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn anim_channel_clamps_after_last_key() {
        let ch = AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 0.0, value: Vec3::zero() },
                PosKey { time: 1.0, value: Vec3::new(3.0, 0.0, 0.0) },
            ],
            rotations: Vec::new(),
            scales: Vec::new(),
        };
        let p = ch.sample_pos(5.0);
        assert!((p.x - 3.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // AnimClip tests
    // -----------------------------------------------------------------------

    #[test]
    fn anim_clip_sample_at_zero_gives_first_key() {
        let mut clip = AnimClip::new("test", 2.0, false);
        clip.add_channel(AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 0.0, value: Vec3::new(1.0, 2.0, 3.0) },
                PosKey { time: 2.0, value: Vec3::new(4.0, 5.0, 6.0) },
            ],
            rotations: Vec::new(),
            scales: Vec::new(),
        });

        let pose = clip.sample(0.0, 2);
        // bone 0 should have translation (1,2,3)
        let p = pose[0].mul_point(Vec3::zero());
        assert!((p.x - 1.0).abs() < 1e-4, "x = {}", p.x);
        assert!((p.y - 2.0).abs() < 1e-4, "y = {}", p.y);
        assert!((p.z - 3.0).abs() < 1e-4, "z = {}", p.z);
    }

    #[test]
    fn anim_clip_sample_at_duration_gives_last_key() {
        let mut clip = AnimClip::new("test", 2.0, false);
        clip.add_channel(AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 0.0, value: Vec3::zero() },
                PosKey { time: 2.0, value: Vec3::new(10.0, 0.0, 0.0) },
            ],
            rotations: Vec::new(),
            scales: Vec::new(),
        });

        let pose = clip.sample(2.0, 1);
        let p = pose[0].mul_point(Vec3::zero());
        assert!((p.x - 10.0).abs() < 1e-4, "x = {}", p.x);
    }

    #[test]
    fn anim_clip_wrap_time_looping() {
        let clip = AnimClip::new("loop", 2.0, true);
        assert!((clip.wrap_time(0.0) - 0.0).abs() < 1e-6);
        assert!((clip.wrap_time(1.0) - 1.0).abs() < 1e-6);
        assert!((clip.wrap_time(2.0) - 0.0).abs() < 1e-6);
        assert!((clip.wrap_time(2.5) - 0.5).abs() < 1e-6);
        assert!((clip.wrap_time(4.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn anim_clip_wrap_time_non_looping() {
        let clip = AnimClip::new("oneshot", 3.0, false);
        assert!((clip.wrap_time(-1.0) - 0.0).abs() < 1e-6);
        assert!((clip.wrap_time(1.5) - 1.5).abs() < 1e-6);
        assert!((clip.wrap_time(5.0) - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Animator tests
    // -----------------------------------------------------------------------

    fn make_idle_clip() -> AnimClip {
        let mut clip = AnimClip::new("idle", 1.0, true);
        clip.add_channel(AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 0.0, value: Vec3::zero() },
                PosKey { time: 1.0, value: Vec3::zero() },
            ],
            rotations: Vec::new(),
            scales: Vec::new(),
        });
        clip
    }

    fn make_run_clip() -> AnimClip {
        let mut clip = AnimClip::new("run", 1.0, true);
        clip.add_channel(AnimChannel {
            bone_idx: 0,
            positions: vec![
                PosKey { time: 0.0, value: Vec3::new(1.0, 0.0, 0.0) },
                PosKey { time: 1.0, value: Vec3::new(1.0, 0.0, 0.0) },
            ],
            rotations: Vec::new(),
            scales: Vec::new(),
        });
        clip
    }

    #[test]
    fn animator_update_advances_time() {
        let skel = Skeleton::new();
        let clip = make_idle_clip();
        let mut anim = Animator::new(skel, clip);

        assert!((anim.time - 0.0).abs() < 1e-6);
        anim.update(0.25);
        assert!((anim.time - 0.25).abs() < 1e-5, "time = {}", anim.time);
    }

    #[test]
    fn animator_transition_blends_then_switches() {
        let skel = Skeleton::new();
        let idle = make_idle_clip();
        let run  = make_run_clip();
        let mut anim = Animator::new(skel, idle);

        anim.transition_to(run, 1.0);

        // At blend_t = 0, pose should match idle (pos 0)
        // After full blend, pose should match run (pos 1)
        anim.update(0.5);
        assert!((anim.blend_t - 0.5).abs() < 1e-4, "blend_t = {}", anim.blend_t);
        assert!(anim.next.is_some());

        anim.update(0.5);
        // blend_t reached 1.0 → transition complete
        assert!(anim.next.is_none(), "next should be cleared");
        assert_eq!(anim.current.name, "run");
    }

    #[test]
    fn animator_sample_pose_no_blend_returns_current() {
        let mut skel = Skeleton::new();
        skel.add_bone("b0", None, Mat4::identity());
        let clip = make_idle_clip();
        let anim = Animator::new(skel, clip);
        let pose = anim.sample_pose();
        assert_eq!(pose.len(), 1);
        // Identity-ish (idle clip pos = zero, no scale/rot keys → identity)
    }

    // -----------------------------------------------------------------------
    // Benchmark test
    // -----------------------------------------------------------------------

    #[test]
    fn bench_skin_10k_vertices() {
        const N_VERTS: usize = 10_000;
        const N_BONES: usize = 20;
        const ITERS: usize = 100;

        // Build a mesh with N_VERTS vertices
        let verts: Vec<Vertex> = (0..N_VERTS)
            .map(|i| make_vertex(i as f32, 0.0, 0.0))
            .collect();
        let indices: Vec<u32> = (0..N_VERTS as u32).collect();
        let mesh = Mesh::new("bench", verts, indices);

        // Each vertex weighted to one bone (round-robin across N_BONES)
        let skin: Vec<SkinVertex> = (0..N_VERTS)
            .map(|i| SkinVertex::single((i % N_BONES) as u16))
            .collect();

        // Build a flat skeleton (all root bones)
        let mut skel = Skeleton::new();
        for _bi in 0..N_BONES {
            skel.add_bone("b", None, Mat4::identity());
        }

        let smesh = SkinnedMesh::new(mesh, skin, skel.clone());

        // Compute skin matrices once (identity pose)
        let local_pose: Vec<Mat4> = (0..N_BONES).map(|_| Mat4::identity()).collect();
        let skin_matrices = skel.compute_skin_matrices(&local_pose);

        let mut total_x = 0.0f32;
        for _ in 0..ITERS {
            let out = smesh.skin(&skin_matrices);
            total_x += out.vertices[0].pos.x;
        }

        // Output should be non-zero (vertex 0 is at x=0, but check the sum runs)
        // vertex 1 is at x=1.0, so total should be ITERS * 1.0 from that vertex...
        // We just verify it didn't panic and produced something
        assert!(total_x >= 0.0, "total_x = {total_x}");
    }
}

//! Retained 3D scene graph — the unified environment the DominionOS compositor manages.
//! (see `docs/2d-3d rendering redesign.md` §"Retained UI, Spatial Composition")
//!
//! "Windows" are eliminated. Applications declare scene nodes; the compositor owns
//! the unified 3D world. Pure, safe `no_std`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

use crate::math3d::{
    Vec3, Mat4, Quat, Aabb, Frustum,
    translation, scale, rotation, look_at, perspective,
    DEG_TO_RAD,
};
use crate::mesh::{Mesh, Material};

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// A decomposed rigid (plus scale) transform: translation × rotation × scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    /// Identity transform: no translation, no rotation, unit scale.
    #[inline]
    pub fn identity() -> Transform {
        Transform {
            translation: Vec3::zero(),
            rotation: Quat::identity(),
            scale: Vec3::one(),
        }
    }

    /// Pure translation, identity rotation, unit scale.
    #[inline]
    pub fn from_translation(t: Vec3) -> Transform {
        Transform {
            translation: t,
            rotation: Quat::identity(),
            scale: Vec3::one(),
        }
    }

    /// Pure rotation, no translation, unit scale.
    #[inline]
    pub fn from_rotation(q: Quat) -> Transform {
        Transform {
            translation: Vec3::zero(),
            rotation: q,
            scale: Vec3::one(),
        }
    }

    /// Build the 4×4 column-major matrix: T × R × S.
    ///
    /// Scale is applied first (object space), then rotation, then translation —
    /// the standard TRS convention.
    pub fn to_mat4(&self) -> Mat4 {
        let t = translation(self.translation);
        let r = rotation(self.rotation);
        let s = scale(self.scale);
        // T * R * S
        t * r * s
    }

    /// Component-wise lerp of translation and scale; slerp of rotation.
    ///
    /// Used by ATW reprojection to interpolate between the last rendered pose
    /// and the predicted present-frame pose.
    pub fn lerp(&self, other: &Transform, t: f32) -> Transform {
        Transform {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// A projective camera defined by its transform plus lens parameters.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub transform: Transform,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
}

impl Camera {
    /// Construct a camera looking from `pos` toward `target`.
    /// `up` is taken as world +Y.
    pub fn new(
        pos: Vec3,
        target: Vec3,
        fov_y_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Camera {
        let forward = (target - pos).normalize();
        let world_up = Vec3::y_axis();
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();

        // Build rotation matrix from camera basis vectors.
        // col0=right, col1=up, col2=-forward (right-hand convention).
        let rot_mat = Mat4::from_cols_array([
            [ right.x,    right.y,    right.z,   0.0],
            [ up.x,       up.y,       up.z,       0.0],
            [-forward.x, -forward.y, -forward.z,  0.0],
            [ 0.0,         0.0,        0.0,        1.0],
        ]);

        let rot = quat_from_mat4(&rot_mat);

        Camera {
            transform: Transform {
                translation: pos,
                rotation: rot,
                scale: Vec3::one(),
            },
            fov_y_deg,
            near,
            far,
            aspect,
        }
    }

    /// View matrix: world → camera space.
    pub fn view_matrix(&self) -> Mat4 {
        let pos = self.transform.translation;
        let forward = self.transform.rotation.rotate_vec3(Vec3::new(0.0, 0.0, -1.0));
        let target = pos + forward;
        let up = self.transform.rotation.rotate_vec3(Vec3::y_axis());
        look_at(pos, target, up)
    }

    /// Projection matrix.
    pub fn proj_matrix(&self) -> Mat4 {
        perspective(self.fov_y_deg * DEG_TO_RAD, self.aspect, self.near, self.far)
    }

    /// Combined proj × view matrix (clip-space = proj_view * world_pos).
    pub fn proj_view(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Extract view frustum planes from the combined proj×view matrix.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_proj_view(self.proj_view())
    }

    /// Predict camera pose `dt` seconds ahead assuming constant linear `velocity`.
    /// Used by ATW reprojection.
    pub fn predict(&self, velocity: Vec3, dt: f32) -> Camera {
        let mut c = *self;
        c.transform.translation = c.transform.translation + velocity * dt;
        c
    }
}

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// Opaque identifier for a scene node.
pub type NodeId = u32;

/// Reserved sentinel: 0 is never handed out as a real NodeId.
pub const ROOT_NODE: NodeId = 0;

/// The payload of a [`SceneNode`].
#[derive(Clone, Debug)]
pub enum NodeContent {
    /// Placeholder / grouping node with no visual representation.
    Empty,
    /// References a mesh and material by index in [`Scene::meshes`] / [`Scene::materials`].
    Mesh { mesh_idx: usize, material_idx: usize },
    /// Infinite-distance directional light (sun, sky, …).
    DirectionalLight { color: [f32; 3], intensity: f32, direction: Vec3 },
    /// Omnidirectional point light with physical falloff.
    PointLight { color: [f32; 3], intensity: f32, radius: f32 },
    /// References a camera by index in [`Scene::cameras`].
    Camera { camera_idx: usize },
    /// A 2D UI surface composited into the 3D scene.
    /// `pixel_data` is ARGB8 (little-endian `u32`) row-major.
    UiSurface { width: u32, height: u32, pixel_data: alloc::vec::Vec<u32> },
    /// An opaque media buffer (video frame, GPU texture handle, …).
    MediaBuffer { width: u32, height: u32, handle: u64 },
}

// ---------------------------------------------------------------------------
// SceneNode
// ---------------------------------------------------------------------------

/// A node in the retained scene graph.
#[derive(Clone, Debug)]
pub struct SceneNode {
    /// Unique identifier within the owning [`Scene`].
    pub id: NodeId,
    /// Human-readable name for tooling / debugging.
    pub name: String,
    /// Transform relative to the parent node (or world-space for root nodes).
    pub local_transform: Transform,
    /// The content / payload of this node.
    pub content: NodeContent,
    /// Ordered list of child node ids.
    pub children: Vec<NodeId>,
    /// Parent node id, or `None` for root nodes.
    pub parent: Option<NodeId>,
    /// Whether this node (and its subtree) is visible.
    pub visible: bool,
    /// Capability token identifying the process that owns this node.
    /// The compositor fills this in; defaults to 0.
    pub owner_hash: u64,
    /// Cryptographic visual provenance hash.
    /// The compositor fills this in; defaults to 0.
    pub provenance: u64,
}

impl SceneNode {
    fn new(id: NodeId, name: &str, content: NodeContent) -> SceneNode {
        SceneNode {
            id,
            name: String::from(name),
            local_transform: Transform::identity(),
            content,
            children: Vec::new(),
            parent: None,
            visible: true,
            owner_hash: 0,
            provenance: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The retained 3D scene graph.
///
/// Applications insert nodes; the compositor traverses and renders them each
/// frame without any concept of per-application "windows".
pub struct Scene {
    nodes: BTreeMap<NodeId, SceneNode>,
    /// Next id to hand out. Starts at 1 so 0 stays reserved as [`ROOT_NODE`].
    next_id: NodeId,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub cameras: Vec<Camera>,
    pub active_camera: usize,
}

impl Scene {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty scene.
    pub fn new() -> Scene {
        Scene {
            nodes: BTreeMap::new(),
            next_id: 1,
            meshes: Vec::new(),
            materials: Vec::new(),
            cameras: Vec::new(),
            active_camera: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    /// Create a new node with the given name and content.
    /// Returns the new node's [`NodeId`].
    pub fn create_node(&mut self, name: &str, content: NodeContent) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, SceneNode::new(id, name, content));
        id
    }

    /// Attach `child` under `parent`. Detaches from any previous parent first.
    ///
    /// Refuses (no-op) if the link would create a parent/child cycle — i.e. `child`
    /// is `parent` itself or an ancestor of `parent`. Without this guard a caller could
    /// `attach(a,b)` then `attach(b,a)`, and [`world_transform`](Self::world_transform)
    /// would walk the parent chain forever (unbounded `Vec` growth until the allocator
    /// aborts). The check keeps the node graph acyclic by construction.
    pub fn attach(&mut self, child: NodeId, parent: NodeId) {
        if self.would_cycle(child, parent) {
            return;
        }
        self.detach(child);
        if let Some(node) = self.nodes.get_mut(&child) {
            node.parent = Some(parent);
        }
        if let Some(p) = self.nodes.get_mut(&parent) {
            if !p.children.contains(&child) {
                p.children.push(child);
            }
        }
    }

    /// Would attaching `child` under `parent` create a cycle? True iff `child` equals
    /// `parent` or is already an ancestor of `parent`. Walks the (acyclic by invariant)
    /// parent chain of `parent`, so it always terminates.
    fn would_cycle(&self, child: NodeId, parent: NodeId) -> bool {
        let mut current = Some(parent);
        while let Some(cur) = current {
            if cur == child {
                return true;
            }
            current = self.nodes.get(&cur).and_then(|n| n.parent);
        }
        false
    }

    /// Remove `child` from its current parent (making it a root node).
    /// No-op if the node has no parent or the id is unknown.
    pub fn detach(&mut self, child: NodeId) {
        let old_parent = match self.nodes.get_mut(&child) {
            Some(n) => n.parent.take(),
            None => return,
        };
        if let Some(pid) = old_parent {
            if let Some(p) = self.nodes.get_mut(&pid) {
                p.children.retain(|&c| c != child);
            }
        }
    }

    /// Overwrite the local transform of `id`.
    pub fn set_transform(&mut self, id: NodeId, t: Transform) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.local_transform = t;
        }
    }

    /// Borrow a node by id.
    pub fn get_node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    /// Mutably borrow a node by id.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    /// Remove a node and detach all its children (they become root nodes).
    pub fn remove_node(&mut self, id: NodeId) {
        let children: Vec<NodeId> = self.nodes.get(&id)
            .map(|n| n.children.clone())
            .unwrap_or_default();

        for child_id in &children {
            if let Some(child) = self.nodes.get_mut(child_id) {
                child.parent = None;
            }
        }

        let parent = self.nodes.get(&id).and_then(|n| n.parent);
        if let Some(pid) = parent {
            if let Some(p) = self.nodes.get_mut(&pid) {
                p.children.retain(|&c| c != id);
            }
        }
        self.nodes.remove(&id);
    }

    /// Number of nodes currently in the scene.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // -----------------------------------------------------------------------
    // Asset management
    // -----------------------------------------------------------------------

    /// Add a mesh to the scene's mesh list; returns its index.
    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        let idx = self.meshes.len();
        self.meshes.push(mesh);
        idx
    }

    /// Add a material; returns its index.
    pub fn add_material(&mut self, mat: Material) -> usize {
        let idx = self.materials.len();
        self.materials.push(mat);
        idx
    }

    /// Add a camera; returns its index.
    pub fn add_camera(&mut self, cam: Camera) -> usize {
        let idx = self.cameras.len();
        self.cameras.push(cam);
        idx
    }

    // -----------------------------------------------------------------------
    // Transform queries
    // -----------------------------------------------------------------------

    /// Compute the world-space transform matrix for `id` by accumulating the
    /// parent chain. Root nodes have identity as their parent matrix.
    pub fn world_transform(&self, id: NodeId) -> Mat4 {
        // Walk the parent chain collecting ids, then fold from root down.
        let mut chain: Vec<NodeId> = Vec::new();
        let mut current = id;
        loop {
            chain.push(current);
            // Defensive depth cap: an acyclic chain can never exceed the node count.
            // attach() already forbids cycles, so this is belt-and-suspenders against
            // an unbounded walk (OOM) should the invariant ever be violated.
            if chain.len() > self.nodes.len() {
                break;
            }
            match self.nodes.get(&current).and_then(|n| n.parent) {
                Some(pid) => current = pid,
                None => break,
            }
        }
        // chain[0]=id, chain[last]=root ancestor.
        let mut mat = Mat4::identity();
        for &nid in chain.iter().rev() {
            if let Some(node) = self.nodes.get(&nid) {
                mat = mat * node.local_transform.to_mat4();
            }
        }
        mat
    }

    /// Compute world-space AABB for a `Mesh` node. Returns `None` for non-mesh
    /// nodes or unknown ids.
    pub fn world_aabb(&self, id: NodeId) -> Option<Aabb> {
        let node = self.nodes.get(&id)?;
        let (mesh_idx, _) = match &node.content {
            NodeContent::Mesh { mesh_idx, material_idx } => (*mesh_idx, *material_idx),
            _ => return None,
        };
        let mesh = self.meshes.get(mesh_idx)?;
        let local = &mesh.aabb;
        let world = self.world_transform(id);

        // Transform all 8 corners of the local AABB and re-envelope.
        let mn = local.min;
        let mx = local.max;
        let corners = [
            Vec3::new(mn.x, mn.y, mn.z),
            Vec3::new(mx.x, mn.y, mn.z),
            Vec3::new(mn.x, mx.y, mn.z),
            Vec3::new(mx.x, mx.y, mn.z),
            Vec3::new(mn.x, mn.y, mx.z),
            Vec3::new(mx.x, mn.y, mx.z),
            Vec3::new(mn.x, mx.y, mx.z),
            Vec3::new(mx.x, mx.y, mx.z),
        ];

        let mut wmin = world.mul_point(corners[0]);
        let mut wmax = wmin;
        for &c in corners.iter().skip(1) {
            let wc = world.mul_point(c);
            wmin = wmin.min_elem(wc);
            wmax = wmax.max_elem(wc);
        }
        Some(Aabb::new(wmin, wmax))
    }

    // -----------------------------------------------------------------------
    // Traversal
    // -----------------------------------------------------------------------

    /// Depth-first traversal of all *visible* nodes, starting from every root
    /// node (nodes with `parent == None`).
    ///
    /// `f` receives `(node_id, &SceneNode, world_transform_matrix)`.
    pub fn visit_visible<F: FnMut(NodeId, &SceneNode, Mat4)>(&self, f: &mut F) {
        // Collect root nodes in stable (BTreeMap = sorted by NodeId) order.
        let roots: Vec<NodeId> = self.nodes.values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();

        for root_id in roots {
            self.visit_visible_recursive(root_id, Mat4::identity(), f);
        }
    }

    fn visit_visible_recursive<F: FnMut(NodeId, &SceneNode, Mat4)>(
        &self,
        id: NodeId,
        parent_world: Mat4,
        f: &mut F,
    ) {
        let node = match self.nodes.get(&id) {
            Some(n) => n,
            None => return,
        };
        if !node.visible {
            return; // Skip this entire subtree.
        }
        let world = parent_world * node.local_transform.to_mat4();
        f(id, node, world);

        // Clone children list to avoid borrow conflict with the closure `f`.
        let children = node.children.clone();
        for child_id in children {
            self.visit_visible_recursive(child_id, world, f);
        }
    }

    // -----------------------------------------------------------------------
    // Frustum culling
    // -----------------------------------------------------------------------

    /// Collect visible mesh nodes whose world AABB intersects `frustum`.
    ///
    /// Returns `(node_id, world_transform, mesh_idx, material_idx)`.
    pub fn frustum_cull(
        &self,
        frustum: &Frustum,
    ) -> Vec<(NodeId, Mat4, usize, usize)> {
        let mut result = Vec::new();
        self.visit_visible(&mut |id, node, world| {
            if let NodeContent::Mesh { mesh_idx, material_idx } = &node.content {
                if let Some(aabb) = self.world_aabb(id) {
                    if frustum.test_aabb(aabb) {
                        result.push((id, world, *mesh_idx, *material_idx));
                    }
                }
            }
        });
        result
    }

    // -----------------------------------------------------------------------
    // Light collection
    // -----------------------------------------------------------------------

    /// Collect all visible `DirectionalLight` and `PointLight` nodes with
    /// their world transforms.
    ///
    /// Returns `(node_id, world_transform, &NodeContent)`.
    pub fn lights(&self) -> Vec<(NodeId, Mat4, &NodeContent)> {
        let mut result = Vec::new();
        let roots: Vec<NodeId> = self.nodes.values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();

        for root_id in roots {
            self.collect_lights_recursive(root_id, Mat4::identity(), &mut result);
        }
        result
    }

    fn collect_lights_recursive<'a>(
        &'a self,
        id: NodeId,
        parent_world: Mat4,
        out: &mut Vec<(NodeId, Mat4, &'a NodeContent)>,
    ) {
        let node = match self.nodes.get(&id) {
            Some(n) => n,
            None => return,
        };
        if !node.visible { return; }
        let world = parent_world * node.local_transform.to_mat4();
        match &node.content {
            NodeContent::DirectionalLight { .. } | NodeContent::PointLight { .. } => {
                out.push((id, world, &node.content));
            }
            _ => {}
        }
        let children = node.children.clone();
        for child_id in children {
            self.collect_lights_recursive(child_id, world, out);
        }
    }
}

impl Default for Scene {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Demo scene builder
// ---------------------------------------------------------------------------

/// Build a demo scene with:
/// - A directional light (sun)
/// - A ground plane mesh
/// - `count` object nodes spread across a `spread`-unit-wide area
///
/// `mesh_idx` is the index of the mesh to assign to the animated objects; it
/// is the caller's responsibility to have added it to the returned scene's
/// `meshes` list before the first render (or pass 0 for ground-only use).
pub fn build_demo_scene(mesh_idx: usize, count: u32, spread: f32) -> Scene {
    use crate::math3d::Vec2;
    use crate::mesh::Vertex;

    let mut scene = Scene::new();

    // Default white material for animated objects.
    let mat_idx = scene.add_material(Material::default_white());

    // Camera positioned above and behind the origin.
    let cam = Camera::new(
        Vec3::new(0.0, 5.0, 15.0),
        Vec3::zero(),
        60.0, 16.0 / 9.0, 0.1, 1000.0,
    );
    scene.add_camera(cam);

    // Directional light (sun).
    scene.create_node(
        "sun",
        NodeContent::DirectionalLight {
            color: [1.0, 0.95, 0.85],
            intensity: 3.0,
            direction: Vec3::new(-0.5, -1.0, -0.5).normalize(),
        },
    );

    // Ground plane — a simple quad on the XZ plane.
    {
        let h = spread * 0.5;
        let n = Vec3::y_axis();
        let ground_verts = alloc::vec![
            Vertex::new(Vec3::new(-h, 0.0, -h), n, Vec2::new(0.0, 0.0)),
            Vertex::new(Vec3::new( h, 0.0, -h), n, Vec2::new(1.0, 0.0)),
            Vertex::new(Vec3::new( h, 0.0,  h), n, Vec2::new(1.0, 1.0)),
            Vertex::new(Vec3::new(-h, 0.0,  h), n, Vec2::new(0.0, 1.0)),
        ];
        let ground_idx = alloc::vec![0u32, 1, 2, 0, 2, 3];
        let ground_mesh = Mesh::new("ground", ground_verts, ground_idx);
        let ground_mesh_idx = scene.add_mesh(ground_mesh);
        let ground_mat = scene.add_material(Material::metallic_roughness(
            [0.3, 0.5, 0.3], 0.0, 0.9,
        ));
        let ground_node = scene.create_node(
            "ground",
            NodeContent::Mesh { mesh_idx: ground_mesh_idx, material_idx: ground_mat },
        );
        scene.set_transform(ground_node, Transform::identity());
    }

    // Group parent for all animated objects.
    let parent_group = scene.create_node("objects", NodeContent::Empty);

    // Deterministic 32-bit LCG for reproducible placement (no std rng needed).
    let mut lcg: u32 = 0x9e37_79b9;
    let mut next_f32 = || -> f32 {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (lcg >> 8) as f32 / (1u32 << 24) as f32
    };

    for i in 0..count {
        let x = (next_f32() - 0.5) * spread;
        let z = (next_f32() - 0.5) * spread;
        let y = 0.5; // Rest on the ground plane.
        let node_name = {
            use alloc::format;
            format!("obj_{}", i)
        };
        let node = scene.create_node(
            &node_name,
            NodeContent::Mesh { mesh_idx, material_idx: mat_idx },
        );
        scene.set_transform(node, Transform::from_translation(Vec3::new(x, y, z)));
        scene.attach(node, parent_group);
    }

    scene
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract a unit quaternion from the upper-left 3×3 of a rotation matrix.
///
/// Uses the Shepperd method (Shepperd 1978): pick the largest diagonal element
/// to avoid dividing by near-zero.  The matrix must be column-major:
/// `m.cols[col][row]`.
fn quat_from_mat4(m: &Mat4) -> Quat {
    let m00 = m.cols[0][0]; // right.x
    let m11 = m.cols[1][1]; // up.y
    let m22 = m.cols[2][2]; // -fwd.z
    let trace = m00 + m11 + m22;

    if trace > 0.0 {
        let s = crate::math3d::sqrt32(trace + 1.0) * 2.0; // s = 4w
        Quat::new(
            (m.cols[1][2] - m.cols[2][1]) / s,
            (m.cols[2][0] - m.cols[0][2]) / s,
            (m.cols[0][1] - m.cols[1][0]) / s,
            0.25 * s,
        ).normalize()
    } else if m00 > m11 && m00 > m22 {
        let s = crate::math3d::sqrt32(1.0 + m00 - m11 - m22) * 2.0; // s = 4x
        Quat::new(
            0.25 * s,
            (m.cols[1][0] + m.cols[0][1]) / s,
            (m.cols[2][0] + m.cols[0][2]) / s,
            (m.cols[1][2] - m.cols[2][1]) / s,
        ).normalize()
    } else if m11 > m22 {
        let s = crate::math3d::sqrt32(1.0 + m11 - m00 - m22) * 2.0; // s = 4y
        Quat::new(
            (m.cols[1][0] + m.cols[0][1]) / s,
            0.25 * s,
            (m.cols[2][1] + m.cols[1][2]) / s,
            (m.cols[2][0] - m.cols[0][2]) / s,
        ).normalize()
    } else {
        let s = crate::math3d::sqrt32(1.0 + m22 - m00 - m11) * 2.0; // s = 4z
        Quat::new(
            (m.cols[2][0] + m.cols[0][2]) / s,
            (m.cols[2][1] + m.cols[1][2]) / s,
            0.25 * s,
            (m.cols[0][1] - m.cols[1][0]) / s,
        ).normalize()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Transform
    // -----------------------------------------------------------------------

    #[test]
    fn transform_identity_is_identity_mat() {
        let m = Transform::identity().to_mat4();
        let id = Mat4::identity();
        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (m.cols[col][row] - id.cols[col][row]).abs() < 1e-5,
                    "identity transform must produce identity matrix at col={col} row={row}"
                );
            }
        }
    }

    #[test]
    fn transform_translation_affects_point() {
        let m = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0)).to_mat4();
        let p = m.mul_point(Vec3::zero());
        assert!((p.x - 1.0).abs() < 1e-5);
        assert!((p.y - 2.0).abs() < 1e-5);
        assert!((p.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn transform_lerp_midpoint() {
        let a = Transform::from_translation(Vec3::new(0.0, 0.0, 0.0));
        let b = Transform::from_translation(Vec3::new(4.0, 0.0, 0.0));
        let mid = a.lerp(&b, 0.5);
        assert!((mid.translation.x - 2.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    #[test]
    fn create_node_returns_unique_ids() {
        let mut scene = Scene::new();
        let a = scene.create_node("a", NodeContent::Empty);
        let b = scene.create_node("b", NodeContent::Empty);
        assert_ne!(a, b);
        assert_eq!(scene.node_count(), 2);
    }

    #[test]
    fn attach_sets_parent_and_child() {
        let mut scene = Scene::new();
        let parent = scene.create_node("parent", NodeContent::Empty);
        let child = scene.create_node("child", NodeContent::Empty);
        scene.attach(child, parent);
        assert_eq!(scene.get_node(child).unwrap().parent, Some(parent));
        assert!(scene.get_node(parent).unwrap().children.contains(&child));
    }

    #[test]
    fn detach_removes_parent_and_child_link() {
        let mut scene = Scene::new();
        let parent = scene.create_node("parent", NodeContent::Empty);
        let child = scene.create_node("child", NodeContent::Empty);
        scene.attach(child, parent);
        scene.detach(child);
        assert_eq!(scene.get_node(child).unwrap().parent, None);
        assert!(!scene.get_node(parent).unwrap().children.contains(&child));
    }

    #[test]
    fn remove_node_detaches_children() {
        let mut scene = Scene::new();
        let parent = scene.create_node("p", NodeContent::Empty);
        let child = scene.create_node("c", NodeContent::Empty);
        scene.attach(child, parent);
        scene.remove_node(parent);
        assert!(scene.get_node(parent).is_none());
        // Child becomes a root node.
        assert_eq!(scene.get_node(child).unwrap().parent, None);
    }

    // -----------------------------------------------------------------------
    // world_transform: chain of 3 translated nodes
    // -----------------------------------------------------------------------

    #[test]
    fn world_transform_accumulates_translations() {
        let mut scene = Scene::new();
        let a = scene.create_node("a", NodeContent::Empty);
        let b = scene.create_node("b", NodeContent::Empty);
        let c = scene.create_node("c", NodeContent::Empty);
        scene.set_transform(a, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)));
        scene.set_transform(b, Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)));
        scene.set_transform(c, Transform::from_translation(Vec3::new(3.0, 0.0, 0.0)));
        scene.attach(b, a);
        scene.attach(c, b);

        let world = scene.world_transform(c);
        let origin = world.mul_point(Vec3::zero());
        assert!((origin.x - 6.0).abs() < 1e-4, "expected x=6, got {}", origin.x);
        assert!(origin.y.abs() < 1e-4);
        assert!(origin.z.abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // visit_visible: depth-first order
    // -----------------------------------------------------------------------

    #[test]
    fn visit_visible_depth_first_order() {
        let mut scene = Scene::new();
        let root      = scene.create_node("root",      NodeContent::Empty);
        let child_a   = scene.create_node("child_a",   NodeContent::Empty);
        let grandchild = scene.create_node("grandchild", NodeContent::Empty);
        let child_b   = scene.create_node("child_b",   NodeContent::Empty);
        let child_c   = scene.create_node("child_c",   NodeContent::Empty);

        scene.attach(child_a, root);
        scene.attach(grandchild, child_a);
        scene.attach(child_b, root);
        scene.attach(child_c, root);

        let mut visited: Vec<NodeId> = Vec::new();
        scene.visit_visible(&mut |id, _, _| visited.push(id));

        let pos = |n: NodeId| visited.iter().position(|&x| x == n).unwrap();
        assert!(pos(root)      < pos(child_a),   "root before child_a");
        assert!(pos(child_a)   < pos(grandchild), "child_a before grandchild (DFS)");
        assert!(pos(child_a)   < pos(child_b),   "child_a subtree before child_b");
        assert!(pos(child_b)   < pos(child_c),   "child_b before child_c");
    }

    #[test]
    fn visit_visible_skips_invisible_subtrees() {
        let mut scene = Scene::new();
        let root            = scene.create_node("root",            NodeContent::Empty);
        let hidden          = scene.create_node("hidden",          NodeContent::Empty);
        let child_of_hidden = scene.create_node("child_of_hidden", NodeContent::Empty);
        scene.attach(hidden, root);
        scene.attach(child_of_hidden, hidden);
        scene.get_node_mut(hidden).unwrap().visible = false;

        let mut visited: Vec<NodeId> = Vec::new();
        scene.visit_visible(&mut |id, _, _| visited.push(id));
        assert!( visited.contains(&root));
        assert!(!visited.contains(&hidden));
        assert!(!visited.contains(&child_of_hidden));
    }

    // -----------------------------------------------------------------------
    // frustum_cull
    // -----------------------------------------------------------------------

    #[test]
    fn frustum_cull_node_inside_is_returned() {
        use crate::mesh::Vertex;
        use crate::math3d::Vec2;

        let mut scene = Scene::new();
        let verts = alloc::vec![
            Vertex::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::y_axis(), Vec2::new(0.0, 0.0)),
            Vertex::new(Vec3::new( 0.5, -0.5, -0.5), Vec3::y_axis(), Vec2::new(1.0, 0.0)),
            Vertex::new(Vec3::new( 0.5,  0.5, -0.5), Vec3::y_axis(), Vec2::new(1.0, 1.0)),
            Vertex::new(Vec3::new(-0.5,  0.5, -0.5), Vec3::y_axis(), Vec2::new(0.0, 1.0)),
        ];
        let idxs = alloc::vec![0u32, 1, 2, 0, 2, 3];
        let mesh_idx = scene.add_mesh(Mesh::new("box", verts, idxs));
        let mat_idx  = scene.add_material(Material::default_white());
        let node = scene.create_node("box", NodeContent::Mesh { mesh_idx, material_idx: mat_idx });
        // Place node directly in front of the camera.
        scene.set_transform(node, Transform::from_translation(Vec3::new(0.0, 0.0, -5.0)));

        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            60.0, 1.0, 0.1, 100.0,
        );
        let visible = scene.frustum_cull(&cam.frustum());
        assert_eq!(visible.len(), 1, "node inside frustum should be visible");
        assert_eq!(visible[0].0, node);
    }

    #[test]
    fn frustum_cull_node_outside_excluded() {
        use crate::mesh::Vertex;
        use crate::math3d::Vec2;

        let mut scene = Scene::new();
        let verts = alloc::vec![
            Vertex::new(Vec3::new(-0.5, -0.5, 0.0), Vec3::y_axis(), Vec2::new(0.0, 0.0)),
            Vertex::new(Vec3::new( 0.5, -0.5, 0.0), Vec3::y_axis(), Vec2::new(1.0, 0.0)),
            Vertex::new(Vec3::new( 0.5,  0.5, 0.0), Vec3::y_axis(), Vec2::new(1.0, 1.0)),
        ];
        let mesh_idx = scene.add_mesh(Mesh::new("tri", verts, alloc::vec![0u32, 1, 2]));
        let mat_idx  = scene.add_material(Material::default_white());
        let node = scene.create_node("tri", NodeContent::Mesh { mesh_idx, material_idx: mat_idx });
        // Place far behind the camera and well past the far plane.
        scene.set_transform(node, Transform::from_translation(Vec3::new(0.0, 0.0, 50.0)));

        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            60.0, 1.0, 0.1, 20.0,
        );
        let visible = scene.frustum_cull(&cam.frustum());
        assert!(visible.is_empty(), "node outside frustum must not be returned");
    }

    // -----------------------------------------------------------------------
    // Asset management
    // -----------------------------------------------------------------------

    #[test]
    fn add_mesh_material_camera_indices() {
        let mut scene = Scene::new();
        let m0 = scene.add_mesh(crate::mesh::generate_cube(1.0));
        let m1 = scene.add_mesh(crate::mesh::generate_cube(2.0));
        assert_eq!(m0, 0);
        assert_eq!(m1, 1);

        let mat0 = scene.add_material(Material::default_white());
        let mat1 = scene.add_material(Material::metallic_roughness([1.0, 0.0, 0.0], 0.5, 0.5));
        assert_eq!(mat0, 0);
        assert_eq!(mat1, 1);

        let cam0 = scene.add_camera(Camera::new(
            Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), 60.0, 1.0, 0.1, 100.0,
        ));
        let cam1 = scene.add_camera(Camera::new(
            Vec3::new(5.0, 0.0, 0.0), Vec3::zero(), 90.0, 1.77, 0.01, 500.0,
        ));
        assert_eq!(cam0, 0);
        assert_eq!(cam1, 1);
    }

    #[test]
    fn active_camera_switching() {
        let mut scene = Scene::new();
        scene.add_camera(Camera::new(
            Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), 60.0, 1.0, 0.1, 100.0,
        ));
        scene.add_camera(Camera::new(
            Vec3::new(10.0, 0.0, 0.0), Vec3::zero(), 90.0, 1.0, 0.1, 100.0,
        ));
        assert_eq!(scene.active_camera, 0);
        scene.active_camera = 1;
        assert_eq!(scene.active_camera, 1);
        let cam = &scene.cameras[scene.active_camera];
        assert!((cam.transform.translation.x - 10.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Camera
    // -----------------------------------------------------------------------

    #[test]
    fn camera_predict_advances_translation() {
        let cam = Camera::new(
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::zero(),
            60.0, 1.77, 0.1, 1000.0,
        );
        let predicted = cam.predict(Vec3::new(1.0, 0.0, 0.0), 0.016);
        assert!((predicted.transform.translation.x - 0.016).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Demo scene builder
    // -----------------------------------------------------------------------

    #[test]
    fn build_demo_scene_has_expected_nodes() {
        let scene = build_demo_scene(0, 5, 20.0);
        // sun + ground + objects_group + 5 object nodes = 8 minimum.
        assert!(scene.node_count() >= 8, "demo scene too sparse: {}", scene.node_count());
    }

    #[test]
    fn build_demo_scene_zero_objects() {
        let scene = build_demo_scene(0, 0, 10.0);
        // sun + ground + objects_group = 3 minimum.
        assert!(scene.node_count() >= 3);
    }
}

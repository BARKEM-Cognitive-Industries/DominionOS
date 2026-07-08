//! Hierarchical GPU instancing: BVH-accelerated frustum culling + draw call batching.
//! (see `docs/2d-3d rendering redesign.md` §"Multi-GPU, Multi-Instance Federated Execution")
//!
//! Inspired by Unreal's hierarchical instanced static meshes. A BVH over world-space
//! AABBs lets us cull entire subtrees in O(log N), then batch surviving instances by
//! mesh+material for a single GPU draw call each.
//!
//! Pure, safe `no_std`.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

use crate::math3d::{Vec3, Mat4, Aabb, Frustum};

// ---------------------------------------------------------------------------
// AABB transform helper — Arvo method via all 8 corners
// ---------------------------------------------------------------------------

/// Transform an AABB by a Mat4 by transforming all 8 corners and refitting.
fn transform_aabb(aabb: Aabb, m: &Mat4) -> Aabb {
    let corners = [
        Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
    ];

    let first = m.mul_point(corners[0]);
    let mut out = Aabb::new(first, first);
    for &c in &corners[1..] {
        let p = m.mul_point(c);
        out = Aabb::new(out.min.min_elem(p), out.max.max_elem(p));
    }
    out
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

/// A single renderable instance: a mesh placed in the world with a transform.
#[derive(Clone, Debug)]
pub struct Instance {
    pub id: u32,
    pub mesh_id: u32,
    pub material_id: u32,
    pub transform: Mat4,
    /// AABB in object (local) space.
    pub local_aabb: Aabb,
    /// AABB transformed into world space.
    pub world_aabb: Aabb,
}

impl Instance {
    /// Create a new instance and immediately compute `world_aabb`.
    pub fn new(
        id: u32,
        mesh_id: u32,
        material_id: u32,
        transform: Mat4,
        local_aabb: Aabb,
    ) -> Instance {
        let world_aabb = transform_aabb(local_aabb, &transform);
        Instance {
            id,
            mesh_id,
            material_id,
            transform,
            local_aabb,
            world_aabb,
        }
    }

    /// Recompute `world_aabb` from `local_aabb` and `transform`.
    pub fn update_world_aabb(&mut self) {
        self.world_aabb = transform_aabb(self.local_aabb, &self.transform);
    }
}

// ---------------------------------------------------------------------------
// BVH
// ---------------------------------------------------------------------------

/// A node in the bounding volume hierarchy.
#[derive(Clone, Debug)]
pub enum BvhNode {
    Leaf {
        /// Index into `InstanceRenderer::instances`.
        instance_idx: usize,
        aabb: Aabb,
    },
    Inner {
        /// Union AABB of all descendants.
        aabb: Aabb,
        /// Index into `Bvh::nodes` for the left child.
        left: usize,
        /// Index into `Bvh::nodes` for the right child.
        right: usize,
        /// Which axis was chosen for this split: 0=X, 1=Y, 2=Z.
        split_axis: u8,
    },
}

impl BvhNode {
    fn aabb(&self) -> Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => *aabb,
            BvhNode::Inner { aabb, .. } => *aabb,
        }
    }
}

/// A bounding volume hierarchy built over a flat slice of instances.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    root: usize,
}

impl Bvh {
    /// Build a BVH using SAH median-split over the given instances.
    pub fn build(instances: &[Instance]) -> Bvh {
        if instances.is_empty() {
            // Degenerate: empty tree — root is a sentinel; we never query it.
            let sentinel_aabb = Aabb::new(Vec3::zero(), Vec3::zero());
            let nodes = alloc::vec![BvhNode::Leaf { instance_idx: 0, aabb: sentinel_aabb }];
            return Bvh { nodes, root: 0 };
        }

        let mut nodes: Vec<BvhNode> = Vec::new();
        // Work with index ranges into the original `instances` slice.
        let mut indices: Vec<usize> = (0..instances.len()).collect();
        let root = build_recursive(instances, &mut indices, 0, instances.len(), &mut nodes);
        Bvh { nodes, root }
    }

    /// Returns the indices of instances whose `world_aabb` intersects the frustum.
    pub fn frustum_cull<'a>(&self, instances: &'a [Instance], frustum: &Frustum) -> Vec<usize> {
        // Guard the empty-scene sentinel: `build(&[])` stores a placeholder leaf
        // (instance_idx 0) rather than an empty node list, so short-circuit here
        // to avoid returning a phantom out-of-bounds index for no instances.
        if self.nodes.is_empty() || instances.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        traverse(self, self.root, instances, frustum, &mut result);
        result
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// BVH build — recursive SAH median-split
// ---------------------------------------------------------------------------

/// Recursively build BVH nodes. Returns the index of the new node in `nodes`.
fn build_recursive(
    instances: &[Instance],
    indices: &mut Vec<usize>,
    start: usize,
    end: usize,
    nodes: &mut Vec<BvhNode>,
) -> usize {
    let count = end - start;

    // Leaf condition: 4 or fewer instances.
    if count <= 4 {
        return build_leaf_cluster(instances, indices, start, end, nodes);
    }

    // 1. Compute union AABB of all world_aabbs in [start, end).
    let union_aabb = compute_union_aabb(instances, indices, start, end);

    // 2. Choose split axis: largest extent.
    let extents = union_aabb.max - union_aabb.min;
    let split_axis: u8 = if extents.x >= extents.y && extents.x >= extents.z {
        0
    } else if extents.y >= extents.z {
        1
    } else {
        2
    };

    // 3. Sort [start, end) sub-range by center along split_axis.
    let sub = &mut indices[start..end];
    sub.sort_unstable_by(|&a, &b| {
        let ca = aabb_center_axis(&instances[a].world_aabb, split_axis);
        let cb = aabb_center_axis(&instances[b].world_aabb, split_axis);
        ca.partial_cmp(&cb).unwrap_or(core::cmp::Ordering::Equal)
    });

    // 4. Split at the median.
    let mid = start + count / 2;

    // Reserve a slot for this inner node before recursing, so children can
    // be placed after it without invalidating the parent index.
    let node_idx = nodes.len();
    // Push a placeholder; we fill it in after we know left/right.
    nodes.push(BvhNode::Leaf {
        instance_idx: 0,
        aabb: union_aabb,
    });

    let left = build_recursive(instances, indices, start, mid, nodes);
    let right = build_recursive(instances, indices, mid, end, nodes);

    // Fill in the real inner node.
    nodes[node_idx] = BvhNode::Inner {
        aabb: union_aabb,
        left,
        right,
        split_axis,
    };

    node_idx
}

/// Build one or more leaf nodes for a cluster of ≤4 instances.
/// Returns the root of this leaf cluster (a single leaf if count==1,
/// otherwise a small inner node spanning all leaves).
fn build_leaf_cluster(
    instances: &[Instance],
    indices: &[usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<BvhNode>,
) -> usize {
    let count = end - start;

    if count == 1 {
        let idx = indices[start];
        let leaf_idx = nodes.len();
        nodes.push(BvhNode::Leaf {
            instance_idx: idx,
            aabb: instances[idx].world_aabb,
        });
        return leaf_idx;
    }

    // Build individual leaves first.
    let mut leaf_indices: Vec<usize> = Vec::new();
    for i in start..end {
        let inst_idx = indices[i];
        let li = nodes.len();
        nodes.push(BvhNode::Leaf {
            instance_idx: inst_idx,
            aabb: instances[inst_idx].world_aabb,
        });
        leaf_indices.push(li);
    }

    // Chain them into a binary tree by pairing up.
    let mut current = leaf_indices;
    while current.len() > 1 {
        let mut next: Vec<usize> = Vec::new();
        let mut i = 0;
        while i < current.len() {
            if i + 1 < current.len() {
                let l = current[i];
                let r = current[i + 1];
                let aabb = nodes[l].aabb().union(nodes[r].aabb());
                let inner = nodes.len();
                nodes.push(BvhNode::Inner {
                    aabb,
                    left: l,
                    right: r,
                    split_axis: 0,
                });
                next.push(inner);
                i += 2;
            } else {
                next.push(current[i]);
                i += 1;
            }
        }
        current = next;
    }

    current[0]
}

// ---------------------------------------------------------------------------
// BVH traversal
// ---------------------------------------------------------------------------

fn traverse(
    bvh: &Bvh,
    node_idx: usize,
    instances: &[Instance],
    frustum: &Frustum,
    result: &mut Vec<usize>,
) {
    let node = &bvh.nodes[node_idx];
    match node {
        BvhNode::Leaf { instance_idx, aabb } => {
            if frustum.test_aabb(*aabb) {
                result.push(*instance_idx);
            }
        }
        BvhNode::Inner { aabb, left, right, .. } => {
            if frustum.test_aabb(*aabb) {
                traverse(bvh, *left, instances, frustum, result);
                traverse(bvh, *right, instances, frustum, result);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BVH build helpers
// ---------------------------------------------------------------------------

fn compute_union_aabb(instances: &[Instance], indices: &[usize], start: usize, end: usize) -> Aabb {
    let first = instances[indices[start]].world_aabb;
    let mut aabb = first;
    for i in (start + 1)..end {
        aabb = aabb.union(instances[indices[i]].world_aabb);
    }
    aabb
}

fn aabb_center_axis(aabb: &Aabb, axis: u8) -> f32 {
    let c = aabb.center();
    match axis {
        0 => c.x,
        1 => c.y,
        _ => c.z,
    }
}

// ---------------------------------------------------------------------------
// DrawBatch
// ---------------------------------------------------------------------------

/// A group of instances that share the same mesh and material, ready for a
/// single GPU instanced draw call.
#[derive(Clone, Debug)]
pub struct DrawBatch {
    pub mesh_id: u32,
    pub material_id: u32,
    pub transforms: Vec<Mat4>,
    pub instance_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// InstanceRenderer
// ---------------------------------------------------------------------------

/// Top-level manager: stores all instances, owns the BVH, and produces
/// per-frame draw batches after frustum culling.
pub struct InstanceRenderer {
    instances: Vec<Instance>,
    bvh: Option<Bvh>,
    dirty: bool,
    last_visible: usize,
}

impl InstanceRenderer {
    pub fn new() -> InstanceRenderer {
        InstanceRenderer {
            instances: Vec::new(),
            bvh: None,
            dirty: false,
            last_visible: 0,
        }
    }

    /// Add an instance; marks the BVH dirty.
    pub fn add(&mut self, instance: Instance) {
        self.instances.push(instance);
        self.dirty = true;
    }

    /// Remove an instance by id. Returns `true` if one was found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.instances.iter().position(|i| i.id == id) {
            self.instances.swap_remove(pos);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Rebuild the BVH if it is dirty.
    pub fn build_bvh(&mut self) {
        if self.dirty || self.bvh.is_none() {
            self.bvh = Some(Bvh::build(&self.instances));
            self.dirty = false;
        }
    }

    /// Frustum-cull all instances via the BVH, then group surviving ones by
    /// `(mesh_id, material_id)`. Returns draw batches sorted by `mesh_id`.
    pub fn cull_and_batch(&mut self, frustum: &Frustum) -> Vec<DrawBatch> {
        if self.instances.is_empty() {
            return Vec::new();
        }
        self.build_bvh();

        let visible_indices = match &self.bvh {
            Some(bvh) => bvh.frustum_cull(&self.instances, frustum),
            None => Vec::new(),
        };

        self.last_visible = visible_indices.len();

        // Group by (mesh_id, material_id).
        let mut map: BTreeMap<(u32, u32), DrawBatch> = BTreeMap::new();
        for idx in visible_indices {
            let inst = &self.instances[idx];
            let key = (inst.mesh_id, inst.material_id);
            let batch = map.entry(key).or_insert_with(|| DrawBatch {
                mesh_id: inst.mesh_id,
                material_id: inst.material_id,
                transforms: Vec::new(),
                instance_ids: Vec::new(),
            });
            batch.transforms.push(inst.transform);
            batch.instance_ids.push(inst.id);
        }

        // Collect and sort by mesh_id for deterministic output.
        let mut batches: Vec<DrawBatch> = map.into_values().collect();
        batches.sort_unstable_by_key(|b| b.mesh_id);
        batches
    }

    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    pub fn bvh_node_count(&self) -> usize {
        self.bvh.as_ref().map(|b| b.node_count()).unwrap_or(0)
    }

    /// How many instances survived the last `cull_and_batch` call.
    pub fn last_visible_count(&self) -> usize {
        self.last_visible
    }
}

impl Default for InstanceRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// InstanceGroup — hierarchical instances-of-instances
// ---------------------------------------------------------------------------

/// A named group that composes instances and sub-groups, enabling hierarchical
/// instancing (e.g. a forest = many trees, a tree = many branches).
pub struct InstanceGroup {
    pub name: String,
    pub members: Vec<Instance>,
    /// Sub-groups with their local-to-parent transform offset.
    pub subgroups: Vec<(InstanceGroup, Mat4)>,
    pub group_aabb: Aabb,
}

impl InstanceGroup {
    pub fn new(name: &str) -> InstanceGroup {
        let empty_aabb = Aabb::new(Vec3::zero(), Vec3::zero());
        InstanceGroup {
            name: String::from(name),
            members: Vec::new(),
            subgroups: Vec::new(),
            group_aabb: empty_aabb,
        }
    }

    /// Add a direct member instance and expand `group_aabb`.
    pub fn add_instance(&mut self, inst: Instance) {
        if self.members.is_empty() && self.subgroups.is_empty() {
            self.group_aabb = inst.world_aabb;
        } else {
            self.group_aabb = self.group_aabb.union(inst.world_aabb);
        }
        self.members.push(inst);
    }

    /// Add a sub-group with a local-to-parent offset transform and expand `group_aabb`.
    pub fn add_subgroup(&mut self, group: InstanceGroup, offset: Mat4) {
        let sub_world_aabb = transform_aabb(group.group_aabb, &offset);
        if self.members.is_empty() && self.subgroups.is_empty() {
            self.group_aabb = sub_world_aabb;
        } else {
            self.group_aabb = self.group_aabb.union(sub_world_aabb);
        }
        self.subgroups.push((group, offset));
    }

    /// Flatten the entire hierarchy into a list of `Instance` values, each
    /// with `transform` composed with all ancestor `parent_transform` matrices.
    pub fn flatten(&self, parent_transform: &Mat4) -> Vec<Instance> {
        let mut result = Vec::new();
        self.flatten_into(parent_transform, &mut result);
        result
    }

    fn flatten_into(&self, parent_transform: &Mat4, out: &mut Vec<Instance>) {
        // Direct members: compose parent_transform * instance.transform.
        for inst in &self.members {
            let composed = *parent_transform * inst.transform;
            let mut flat = Instance::new(
                inst.id,
                inst.mesh_id,
                inst.material_id,
                composed,
                inst.local_aabb,
            );
            flat.update_world_aabb();
            out.push(flat);
        }

        // Recurse into sub-groups.
        for (subgroup, offset) in &self.subgroups {
            let combined = *parent_transform * *offset;
            subgroup.flatten_into(&combined, out);
        }
    }

    /// Total number of instances (direct + all recursive sub-groups).
    pub fn total_instance_count(&self) -> usize {
        let direct = self.members.len();
        let sub: usize = self.subgroups.iter().map(|(g, _)| g.total_instance_count()).sum();
        direct + sub
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math3d::{perspective, look_at, translation, DEG_TO_RAD};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn unit_aabb() -> Aabb {
        Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5))
    }

    fn make_instance(id: u32, mesh_id: u32, material_id: u32, pos: Vec3) -> Instance {
        Instance::new(id, mesh_id, material_id, translation(pos), unit_aabb())
    }

    /// A frustum looking down -Z from the origin with 90° FOV.
    fn front_frustum() -> Frustum {
        let proj = perspective(DEG_TO_RAD * 90.0, 1.0, 1.0, 1000.0);
        let view = look_at(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::y_axis(),
        );
        Frustum::from_proj_view(proj * view)
    }

    // -----------------------------------------------------------------------
    // Instance
    // -----------------------------------------------------------------------

    #[test]
    fn test_instance_new_computes_world_aabb() {
        let pos = Vec3::new(5.0, 0.0, 0.0);
        let inst = make_instance(1, 1, 1, pos);
        // The world AABB should be shifted by (5, 0, 0).
        assert!(inst.world_aabb.min.x > 4.0);
        assert!(inst.world_aabb.max.x < 6.0);
    }

    #[test]
    fn test_update_world_aabb_transformed_contains_corners() {
        let mut inst = make_instance(1, 1, 1, Vec3::new(10.0, 20.0, 30.0));
        inst.update_world_aabb();
        // All 8 original corners transformed by the translate should be inside world_aabb.
        let corners = [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new( 0.5, -0.5, -0.5),
            Vec3::new(-0.5,  0.5, -0.5),
            Vec3::new( 0.5,  0.5, -0.5),
            Vec3::new(-0.5, -0.5,  0.5),
            Vec3::new( 0.5, -0.5,  0.5),
            Vec3::new(-0.5,  0.5,  0.5),
            Vec3::new( 0.5,  0.5,  0.5),
        ];
        for c in &corners {
            let world_c = inst.transform.mul_point(*c);
            assert!(
                inst.world_aabb.contains_point(world_c),
                "corner {:?} -> {:?} not inside world_aabb {:?}",
                c,
                world_c,
                inst.world_aabb
            );
        }
    }

    // -----------------------------------------------------------------------
    // BVH
    // -----------------------------------------------------------------------

    #[test]
    fn test_bvh_build_single_instance() {
        let inst = make_instance(0, 1, 1, Vec3::zero());
        let bvh = Bvh::build(&[inst.clone()]);
        // Should not panic; must have at least one node.
        assert!(bvh.node_count() >= 1);
    }

    #[test]
    fn test_bvh_build_100_instances_node_count_gt_1() {
        let instances: Vec<Instance> = (0..100)
            .map(|i| make_instance(i, 1, 1, Vec3::new(i as f32, 0.0, -10.0)))
            .collect();
        let bvh = Bvh::build(&instances);
        assert!(bvh.node_count() > 1, "expected more than 1 node, got {}", bvh.node_count());
    }

    #[test]
    fn test_frustum_cull_all_visible() {
        // 10 instances directly in front of the camera.
        let instances: Vec<Instance> = (0..10)
            .map(|i| make_instance(i, 1, 1, Vec3::new(0.0, 0.0, -(10.0 + i as f32))))
            .collect();
        let bvh = Bvh::build(&instances);
        let frustum = front_frustum();
        let visible = bvh.frustum_cull(&instances, &frustum);
        assert_eq!(visible.len(), 10, "all 10 should be visible");
    }

    #[test]
    fn test_frustum_cull_none_visible() {
        // 10 instances well behind the camera (positive Z is behind for this view).
        let instances: Vec<Instance> = (0..10)
            .map(|i| make_instance(i, 1, 1, Vec3::new(0.0, 0.0, 500.0 + i as f32)))
            .collect();
        let bvh = Bvh::build(&instances);
        let frustum = front_frustum();
        let visible = bvh.frustum_cull(&instances, &frustum);
        assert_eq!(visible.len(), 0, "none should be visible behind the camera");
    }

    #[test]
    fn test_frustum_cull_half_visible() {
        // Half in front, half behind.
        let mut instances: Vec<Instance> = Vec::new();
        for i in 0..10u32 {
            instances.push(make_instance(i, 1, 1, Vec3::new(0.0, 0.0, -(5.0 + i as f32))));
        }
        for i in 10..20u32 {
            instances.push(make_instance(i, 1, 1, Vec3::new(0.0, 0.0, 500.0 + i as f32)));
        }
        let bvh = Bvh::build(&instances);
        let frustum = front_frustum();
        let visible = bvh.frustum_cull(&instances, &frustum);
        assert_eq!(visible.len(), 10, "exactly the front 10 should be visible");
    }

    // -----------------------------------------------------------------------
    // InstanceRenderer
    // -----------------------------------------------------------------------

    #[test]
    fn test_renderer_add_and_count() {
        let mut r = InstanceRenderer::new();
        for i in 0..50u32 {
            r.add(make_instance(i, 1, 1, Vec3::new(i as f32, 0.0, -10.0)));
        }
        assert_eq!(r.instance_count(), 50);
    }

    #[test]
    fn test_renderer_remove_decreases_count() {
        let mut r = InstanceRenderer::new();
        r.add(make_instance(0, 1, 1, Vec3::zero()));
        r.add(make_instance(1, 1, 1, Vec3::zero()));
        assert!(r.remove(0));
        assert_eq!(r.instance_count(), 1);
        // Removing a non-existent id returns false.
        assert!(!r.remove(99));
        assert_eq!(r.instance_count(), 1);
    }

    #[test]
    fn test_renderer_cull_and_batch_two_mesh_ids() {
        let mut r = InstanceRenderer::new();
        // 5 instances with mesh_id=1, 5 with mesh_id=2, all in front.
        for i in 0..5u32 {
            r.add(make_instance(i, 1, 1, Vec3::new(i as f32 * 0.1, 0.0, -10.0)));
        }
        for i in 5..10u32 {
            r.add(make_instance(i, 2, 1, Vec3::new(i as f32 * 0.1, 0.0, -10.0)));
        }
        let frustum = front_frustum();
        let batches = r.cull_and_batch(&frustum);
        assert_eq!(batches.len(), 2, "expected 2 batches, got {}", batches.len());
        assert_eq!(batches[0].mesh_id, 1);
        assert_eq!(batches[1].mesh_id, 2);
        assert_eq!(batches[0].transforms.len(), 5);
        assert_eq!(batches[1].transforms.len(), 5);
    }

    #[test]
    fn test_renderer_build_bvh_then_node_count() {
        let mut r = InstanceRenderer::new();
        for i in 0..20u32 {
            r.add(make_instance(i, 1, 1, Vec3::new(i as f32, 0.0, -5.0)));
        }
        r.build_bvh();
        assert!(r.bvh_node_count() > 1);
    }

    #[test]
    fn test_renderer_last_visible_count() {
        let mut r = InstanceRenderer::new();
        for i in 0..10u32 {
            r.add(make_instance(i, 1, 1, Vec3::new(0.0, 0.0, -(5.0 + i as f32))));
        }
        let frustum = front_frustum();
        r.cull_and_batch(&frustum);
        assert_eq!(r.last_visible_count(), 10);
    }

    // -----------------------------------------------------------------------
    // InstanceGroup
    // -----------------------------------------------------------------------

    #[test]
    fn test_instance_group_flatten_members_plus_subgroup() {
        let mut group = InstanceGroup::new("forest");
        group.add_instance(make_instance(0, 1, 1, Vec3::new(0.0, 0.0, -5.0)));
        group.add_instance(make_instance(1, 1, 1, Vec3::new(1.0, 0.0, -5.0)));
        group.add_instance(make_instance(2, 1, 1, Vec3::new(2.0, 0.0, -5.0)));

        let mut sub = InstanceGroup::new("cluster");
        sub.add_instance(make_instance(3, 2, 1, Vec3::new(0.0, 0.0, -5.0)));
        sub.add_instance(make_instance(4, 2, 1, Vec3::new(1.0, 0.0, -5.0)));

        group.add_subgroup(sub, Mat4::identity());

        assert_eq!(group.total_instance_count(), 5);

        let flat = group.flatten(&Mat4::identity());
        assert_eq!(flat.len(), 5, "flatten should produce 5 instances");
    }

    #[test]
    fn test_instance_group_flatten_applies_parent_transform() {
        let mut group = InstanceGroup::new("g");
        // Instance at local origin.
        group.add_instance(make_instance(0, 1, 1, Vec3::new(0.0, 0.0, -10.0)));

        // Parent transform: translate +100 on X.
        let parent = translation(Vec3::new(100.0, 0.0, 0.0));
        let flat = group.flatten(&parent);
        assert_eq!(flat.len(), 1);
        // The center of the world AABB should be near x=100.
        let c = flat[0].world_aabb.center();
        assert!(c.x > 99.0 && c.x < 101.0, "expected x≈100, got {}", c.x);
    }

    #[test]
    fn test_instance_group_empty() {
        let group = InstanceGroup::new("empty");
        let flat = group.flatten(&Mat4::identity());
        assert_eq!(flat.len(), 0);
        assert_eq!(group.total_instance_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Benchmark-style test (functional, not timed — runs under cargo test)
    // -----------------------------------------------------------------------

    #[cfg(test)]
    fn bench_million_instance_cull() {
        // 100_000 instances on a 316×316 grid, z=-50.
        let side = 316usize;
        let mut renderer = InstanceRenderer::new();
        for y in 0..side {
            for x in 0..side {
                let id = (y * side + x) as u32;
                let pos = Vec3::new(x as f32 * 2.0 - 316.0, 0.0, -(y as f32 * 2.0 + 10.0));
                renderer.add(make_instance(id, 1, 1, pos));
            }
        }
        assert_eq!(renderer.instance_count(), side * side);

        renderer.build_bvh();
        assert!(renderer.bvh_node_count() > 1);

        // Camera looking down -Z from origin, narrow frustum → only central strip visible.
        let frustum = front_frustum();
        let batches = renderer.cull_and_batch(&frustum);

        let visible = renderer.last_visible_count();
        // With a ±50 unit wide frustum at z=-50 the BVH should cull some.
        assert!(
            visible < side * side,
            "BVH should have culled some instances; got {} visible out of {}",
            visible,
            side * side
        );

        // All visible end up in one batch (mesh_id=1).
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].transforms.len(), visible);
        assert_eq!(batches[0].instance_ids.len(), visible);
    }

    #[test]
    fn test_bench_million_instance_cull_runs() {
        bench_million_instance_cull();
    }
}

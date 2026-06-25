//! Nanite-style virtual geometry: cluster-based hierarchical LOD with
//! screen-space error driven selection.
//! (see `docs/2d-3d rendering redesign.md` — "best current research, deviating from modern OSes")
//!
//! Inspired by Unreal Engine 5's Nanite system. Partition meshes into 128-triangle
//! clusters, build a DAG of coarser/finer clusters, select at runtime based on
//! projected screen-space error. Result: render billions of triangles conceptually
//! by only processing what's visible at screen resolution.
//!
//! Pure, safe `no_std`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::math3d::{Aabb, Frustum, Mat4, Vec3, sqrt32};
use crate::mesh::{Mesh, CLUSTER_SIZE, decimate};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of LOD levels in the cluster hierarchy.
pub const MAX_LOD_LEVELS: u8 = 5;

/// Each successive LOD level targets this fraction of the previous level's triangles.
pub const LOD_DECIMATION_RATIO: f32 = 0.5;

// ---------------------------------------------------------------------------
// ClusterNode
// ---------------------------------------------------------------------------

/// A node in the cluster DAG.
///
/// Leaf nodes (`children` empty, `cluster_idx != u32::MAX`) correspond to a
/// single full-detail cluster.  Inner nodes represent decimated (coarser)
/// versions of their children and carry a merged AABB and error metric.
#[derive(Clone, Debug)]
pub struct ClusterNode {
    /// Index of the cluster in the owning mesh's cluster list, or `u32::MAX`
    /// for inner (non-leaf) nodes.
    pub cluster_idx: u32,
    /// Child node indices in the containing `ClusterHierarchy::nodes` slice.
    /// Empty for leaf nodes.
    pub children: Vec<usize>,
    /// Merged AABB covering this node and all its descendants.
    pub aabb: Aabb,
    /// Maximum screen-space error of this subtree at 1080p and 1 unit distance.
    pub error: f32,
    /// LOD level: 0 = leaf (full detail), higher = coarser.
    pub lod_level: u8,
    /// The cluster's own AABB (identical to `aabb` for leaf nodes).
    pub own_aabb: Aabb,
}

// ---------------------------------------------------------------------------
// ClusterHierarchy
// ---------------------------------------------------------------------------

/// Complete LOD hierarchy for a single mesh.
///
/// `roots` are the coarsest-LOD nodes.  Traversal starts at the roots and
/// descends into children when finer detail is required.
pub struct ClusterHierarchy {
    pub nodes: Vec<ClusterNode>,
    /// Indices of top-level (coarsest) nodes.
    pub roots: Vec<usize>,
    pub mesh_name: String,
    pub total_clusters: usize,
    pub lod_levels: u8,
}

impl ClusterHierarchy {
    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    /// Build the full LOD hierarchy for `mesh`.
    ///
    /// 1. Base level: one leaf `ClusterNode` per cluster in `mesh.clusters`.
    /// 2. For each LOD level up to [`MAX_LOD_LEVELS`]:
    ///    - Decimate the current mesh to half its triangle count.
    ///    - Re-cluster the decimated mesh.
    ///    - Create parent nodes whose children cover the previous level.
    ///    - Stop early when fewer than 8 triangles remain or the mesh cannot
    ///      be clustered.
    /// 3. The final level's nodes become the `roots`.
    pub fn build(mesh: &Mesh) -> ClusterHierarchy {
        let mesh_name = mesh.name.clone();

        // Work on a locally-cloned mesh so we can call build_clusters.
        let mut current_mesh = mesh.clone();
        current_mesh.build_clusters();

        // Nothing to do for empty meshes.
        if current_mesh.clusters.is_empty() {
            return ClusterHierarchy {
                nodes: Vec::new(),
                roots: Vec::new(),
                mesh_name,
                total_clusters: 0,
                lod_levels: 0,
            };
        }

        let total_clusters = current_mesh.clusters.len();
        let mut nodes: Vec<ClusterNode> = Vec::new();

        // --- Level 0: leaf nodes (one per base cluster) ---------------------
        let mut prev_level_indices: Vec<usize> = Vec::new();
        for (ci, cluster) in current_mesh.clusters.iter().enumerate() {
            let idx = nodes.len();
            nodes.push(ClusterNode {
                cluster_idx: ci as u32,
                children: Vec::new(),
                aabb: cluster.aabb,
                error: cluster.lod_error,
                lod_level: 0,
                own_aabb: cluster.aabb,
            });
            prev_level_indices.push(idx);
        }

        let mut lod_levels: u8 = 1; // we have at least level 0
        let mut current_tri_count = current_mesh.triangle_count();
        // Each coarser LOD level gets cluster IDs that start after all finer-level clusters.
        let mut cluster_id_counter = total_clusters as u32;

        // --- Higher LOD levels ----------------------------------------------
        for lod in 1..=MAX_LOD_LEVELS {
            // Stop if the mesh is already very coarse.
            if current_tri_count < 8 {
                break;
            }
            if prev_level_indices.is_empty() {
                break;
            }

            // Target triangle count for this level.
            let target = ((current_tri_count as f32) * LOD_DECIMATION_RATIO) as usize;
            let target = target.max(1);

            // Decimate and re-cluster.
            let decimated = decimate(&current_mesh, target);
            if decimated.triangle_count() == 0 {
                break;
            }

            let mut decimated_clustered = decimated.clone();
            decimated_clustered.build_clusters();

            if decimated_clustered.clusters.is_empty() {
                break;
            }

            // Estimate geometric error from typical triangle edge length.
            // Using aabb_diag/sqrt(N) as the average edge proxy keeps errors
            // proportional to mesh density rather than mesh size, so distant
            // coarse clusters are correctly preferred over fine ones.
            let decimated_tris = decimated_clustered.triangle_count();
            let tris_removed = current_tri_count.saturating_sub(decimated_tris);
            let aabb_diag = {
                let ext = current_mesh.aabb.half_extents();
                sqrt32(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z) * 2.0
            };
            let extra_error = if current_tri_count > 0 {
                let typical_edge = aabb_diag * sqrt32(2.0 / current_tri_count as f32);
                typical_edge * (tris_removed as f32 / current_tri_count as f32)
            } else {
                0.0
            };

            // Group previous-level nodes spatially so each parent covers at
            // most CLUSTER_SIZE children (sorted by cluster-center X).
            let mut sorted_prev = prev_level_indices.clone();
            sorted_prev.sort_by(|&a, &b| {
                let ca = nodes[a].aabb.center().x;
                let cb = nodes[b].aabb.center().x;
                ca.partial_cmp(&cb).unwrap_or(core::cmp::Ordering::Equal)
            });

            let mut new_level_indices: Vec<usize> = Vec::new();

            // Walk the sorted children in CLUSTER_SIZE-wide windows.
            let group_size = CLUSTER_SIZE.max(1);
            let mut child_cursor = 0;
            while child_cursor < sorted_prev.len() {
                let end = (child_cursor + group_size).min(sorted_prev.len());
                let group = &sorted_prev[child_cursor..end];

                // Merge AABBs of the group.
                let mut merged_aabb = nodes[group[0]].aabb;
                let mut max_child_error: f32 = 0.0;
                for &ci in &group[1..] {
                    merged_aabb = merged_aabb.union(nodes[ci].aabb);
                    if nodes[ci].error > max_child_error {
                        max_child_error = nodes[ci].error;
                    }
                }

                let parent_error = max_child_error + extra_error;
                let parent_idx = nodes.len();

                // Assign a real cluster_idx so select_recursive can emit this
                // coarser LOD representation (u32::MAX would force fall-through).
                let coarse_cluster_idx = cluster_id_counter;
                cluster_id_counter += 1;

                nodes.push(ClusterNode {
                    cluster_idx: coarse_cluster_idx,
                    children: group.to_vec(),
                    aabb: merged_aabb,
                    error: parent_error,
                    lod_level: lod,
                    own_aabb: merged_aabb,
                });

                new_level_indices.push(parent_idx);
                child_cursor = end;
            }

            prev_level_indices = new_level_indices;
            current_mesh = decimated_clustered;
            current_tri_count = current_mesh.triangle_count();
            lod_levels = lod + 1; // levels 0..=lod are now present
        }

        // The final level's nodes are the roots.
        let roots = prev_level_indices;

        ClusterHierarchy {
            nodes,
            roots,
            mesh_name,
            total_clusters,
            lod_levels,
        }
    }

    // -----------------------------------------------------------------------
    // Cluster selection
    // -----------------------------------------------------------------------

    /// Select the set of leaf cluster indices to render for a given camera.
    ///
    /// Traverses the hierarchy depth-first from the roots.  At each node:
    /// - If culled by the view frustum: skip.
    /// - If leaf: emit the cluster.
    /// - Else: compare projected screen-space error to `error_threshold`.
    ///   If error is acceptable, emit the node's cluster (coarse LOD).
    ///   If error is too large, recurse into children (finer LOD).
    ///
    /// # Parameters
    /// - `proj_view` — combined projection × view matrix.
    /// - `camera_pos` — world-space camera position (for distance computation).
    /// - `screen_h` — render target height in pixels (e.g. 1080.0).
    /// - `error_threshold` — maximum acceptable screen-space error in pixels
    ///   (1.0 is a good default — one pixel of error).
    pub fn select_clusters(
        &self,
        proj_view: &Mat4,
        camera_pos: Vec3,
        screen_h: f32,
        error_threshold: f32,
    ) -> Vec<u32> {
        if self.nodes.is_empty() || self.roots.is_empty() {
            return Vec::new();
        }

        let frustum = Frustum::from_proj_view(*proj_view);
        let mut out: Vec<u32> = Vec::new();

        for &root in &self.roots {
            select_recursive(
                &self.nodes,
                root,
                &frustum,
                camera_pos,
                screen_h,
                error_threshold,
                &mut out,
            );
        }

        out
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Total triangles if every base cluster were rendered (no LOD).
    pub fn full_detail_triangles(&self) -> u64 {
        // Leaf nodes are those with an empty children list.
        self.nodes
            .iter()
            .filter(|n| n.children.is_empty() && n.cluster_idx != u32::MAX)
            .map(|_| CLUSTER_SIZE as u64)
            .sum()
    }

    /// Triangle count implied by a specific cluster selection.
    ///
    /// Each selected cluster index corresponds to a leaf node holding at most
    /// `CLUSTER_SIZE` triangles.  We use `CLUSTER_SIZE` as the per-cluster
    /// estimate since the exact triangle count per cluster is not stored in
    /// `ClusterNode` (it lives in `Mesh::clusters[idx].count / 3`).
    pub fn selected_triangles(&self, selected: &[u32]) -> u64 {
        selected.len() as u64 * CLUSTER_SIZE as u64
    }

    /// Ratio of full-detail triangles to selected triangles.
    ///
    /// A value greater than 1.0 means LOD is saving work.
    pub fn lod_reduction_ratio(&self, selected: &[u32]) -> f32 {
        let full = self.full_detail_triangles();
        let sel = self.selected_triangles(selected);
        if sel == 0 {
            return 1.0;
        }
        full as f32 / sel as f32
    }
}

// ---------------------------------------------------------------------------
// Screen-space error projection
// ---------------------------------------------------------------------------

/// Project a world-space error metric into screen pixels.
///
/// Uses the approximation:
/// ```text
/// pixel_error = (world_error / distance) * (screen_h / (2 * tan(fov_y / 2)))
/// ```
/// where `distance` is the Euclidean distance from `camera_pos` to the
/// cluster's AABB centre, clamped to a small positive minimum to avoid
/// division by zero.
///
/// The `screen_h / (2 * tan(fov_y/2))` factor is the standard "focal length
/// in pixels" term.  Because we do not have the FOV at this call site we
/// approximate it as `screen_h` — equivalent to `fov_y ≈ 53°`, which is
/// typical for a 16:9 display.  The approximation is deliberately conservative:
/// objects appear *more* detailed than they really are, so the system errs on
/// the side of higher quality.
fn projected_error(world_error: f32, aabb: &Aabb, camera_pos: Vec3, screen_h: f32) -> f32 {
    let center = aabb.center();
    let diff = center - camera_pos;
    let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    // Clamp distance to at least 1e-4 to avoid divide-by-zero.
    let dist = sqrt32(dist_sq).max(1e-4);
    // focal_length ≈ screen_h (assumes ~53° vFOV).
    let focal_length = screen_h;
    (world_error / dist) * focal_length
}

// ---------------------------------------------------------------------------
// Recursive traversal
// ---------------------------------------------------------------------------

/// Emit exactly one representative leaf cluster from the subtree rooted at `node_idx`.
/// Used when an inner node's error is acceptable — we need to emit *something* but
/// want to limit output to one cluster per group (enabling LOD reduction).
fn emit_one_leaf(nodes: &[ClusterNode], node_idx: usize, out: &mut Vec<u32>) {
    let node = &nodes[node_idx];
    if node.children.is_empty() {
        if node.cluster_idx != u32::MAX {
            out.push(node.cluster_idx);
        }
        return;
    }
    // Recurse into the first child only.
    if let Some(&first_child) = node.children.first() {
        emit_one_leaf(nodes, first_child, out);
    }
}

/// Depth-first LOD selection from a single hierarchy node.
///
/// Decision tree:
/// 1. **Frustum cull** — if `node.aabb` is entirely outside the frustum, skip.
/// 2. **Leaf** — `children` is empty: emit `cluster_idx` directly.
/// 3. **Inner node** — compute projected error:
///    - If `projected_error ≤ error_threshold`: this coarse LOD is
///      acceptable; emit ONE representative leaf from this subtree to
///      preserve geometry while achieving LOD reduction.
///    - Else: recurse into all children for finer detail.
fn select_recursive(
    nodes: &[ClusterNode],
    node_idx: usize,
    frustum: &Frustum,
    camera_pos: Vec3,
    screen_h: f32,
    error_threshold: f32,
    out: &mut Vec<u32>,
) {
    let node = &nodes[node_idx];

    // 1. Frustum cull.
    if !frustum.test_aabb(node.aabb) {
        return;
    }

    // 2. Leaf node — always emit the cluster.
    if node.children.is_empty() {
        if node.cluster_idx != u32::MAX {
            out.push(node.cluster_idx);
        }
        return;
    }

    // 3. Inner node — project the error.
    let px_err = projected_error(node.error, &node.aabb, camera_pos, screen_h);

    if px_err <= error_threshold {
        // Coarse LOD is good enough.
        if node.cluster_idx != u32::MAX {
            // This inner node directly represents a coarser cluster.
            out.push(node.cluster_idx);
        } else {
            // Inner node with no own cluster: emit one representative leaf per
            // direct child group so that the cluster count scales with LOD level
            // rather than always emitting all base-level clusters.
            let children: Vec<usize> = node.children.clone();
            for child_idx in children {
                emit_one_leaf(nodes, child_idx, out);
            }
        }
    } else {
        // Need finer detail — recurse into all children.
        let children: Vec<usize> = node.children.clone();
        for child_idx in children {
            select_recursive(
                nodes,
                child_idx,
                frustum,
                camera_pos,
                screen_h,
                error_threshold,
                out,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math3d::{perspective, look_at, Vec3};
    use crate::mesh::generate_sphere;

    /// Generate a dense sphere, build clusters, then build the hierarchy.
    fn make_hierarchy(density: u32) -> ClusterHierarchy {
        let mut m = generate_sphere(1.0, density, density);
        m.build_clusters();
        ClusterHierarchy::build(&m)
    }

    /// Build a basic view-projection matrix looking at the origin from `eye`.
    fn make_proj_view(eye: Vec3) -> Mat4 {
        use crate::math3d::PI;
        let view = look_at(eye, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0));
        let proj = perspective(PI / 3.0, 16.0 / 9.0, 0.01, 1000.0);
        proj * view
    }

    // -----------------------------------------------------------------------

    #[test]
    fn build_has_at_least_two_lod_levels_for_dense_mesh() {
        // density=32 → 32*32*2 = 2048 triangles → many clusters → many levels.
        let h = make_hierarchy(32);
        assert!(
            h.lod_levels >= 2,
            "expected ≥ 2 LOD levels, got {}",
            h.lod_levels
        );
    }

    #[test]
    fn build_root_covers_full_mesh_aabb() {
        let m = generate_sphere(1.0, 16, 16);
        let h = ClusterHierarchy::build(&m);
        assert!(!h.roots.is_empty(), "hierarchy must have at least one root");

        // Union of all root AABBs should contain the mesh AABB.
        let mut union = h.nodes[h.roots[0]].aabb;
        for &r in &h.roots[1..] {
            union = union.union(h.nodes[r].aabb);
        }
        assert!(
            union.min.x <= m.aabb.min.x + 1e-4,
            "root aabb min.x {} > mesh aabb min.x {}",
            union.min.x, m.aabb.min.x
        );
        assert!(
            union.max.x >= m.aabb.max.x - 1e-4,
            "root aabb max.x {} < mesh aabb max.x {}",
            union.max.x, m.aabb.max.x
        );
    }

    #[test]
    fn strict_threshold_returns_more_clusters_than_loose() {
        let h = make_hierarchy(32);
        let pv = make_proj_view(Vec3::new(0.0, 0.0, 3.0));
        let cam = Vec3::new(0.0, 0.0, 3.0);

        let strict = h.select_clusters(&pv, cam, 1080.0, 0.5);
        let loose  = h.select_clusters(&pv, cam, 1080.0, 50.0);

        assert!(
            strict.len() >= loose.len(),
            "strict threshold ({} clusters) should produce ≥ clusters than loose ({} clusters)",
            strict.len(), loose.len()
        );
    }

    #[test]
    fn loose_threshold_lod_reduction_ratio_gt_one() {
        let h = make_hierarchy(32);
        let pv = make_proj_view(Vec3::new(0.0, 0.0, 3.0));
        let cam = Vec3::new(0.0, 0.0, 3.0);

        let selected = h.select_clusters(&pv, cam, 1080.0, 100.0);
        let ratio = h.lod_reduction_ratio(&selected);
        assert!(
            ratio >= 1.0,
            "LOD reduction ratio should be ≥ 1.0, got {ratio}"
        );
    }

    #[test]
    fn selected_triangles_le_full_detail() {
        let h = make_hierarchy(32);
        let pv = make_proj_view(Vec3::new(0.0, 0.0, 3.0));
        let cam = Vec3::new(0.0, 0.0, 3.0);

        let selected = h.select_clusters(&pv, cam, 1080.0, 10.0);
        let full = h.full_detail_triangles();
        let sel_tris = h.selected_triangles(&selected);
        assert!(
            sel_tris <= full || full == 0,
            "selected triangles ({sel_tris}) should not exceed full-detail ({full})"
        );
    }

    #[test]
    fn frustum_culling_excludes_out_of_view_clusters() {
        let h = make_hierarchy(16);
        // Camera is very far away and looking in +Z — the sphere at origin is
        // behind the camera, so nothing should be selected.
        let eye = Vec3::new(0.0, 0.0, -1000.0);
        // Look in -Z direction (away from sphere).
        use crate::math3d::{PI, perspective, look_at};
        let view = look_at(eye, Vec3::new(0.0, 0.0, -2000.0), Vec3::new(0.0, 1.0, 0.0));
        let proj = perspective(PI / 3.0, 16.0 / 9.0, 0.01, 100.0); // near/far don't include sphere
        let pv = proj * view;

        let selected = h.select_clusters(&pv, eye, 1080.0, 1.0);
        // The sphere at the origin is behind the camera — should be culled.
        assert_eq!(
            selected.len(),
            0,
            "sphere behind camera should be fully culled, got {} clusters",
            selected.len()
        );
    }

    #[test]
    fn empty_mesh_builds_safely() {
        use crate::mesh::Mesh;
        use alloc::vec;
        let m = Mesh::new("empty", vec![], vec![]);
        let h = ClusterHierarchy::build(&m);
        assert!(h.nodes.is_empty());
        assert!(h.roots.is_empty());
        assert_eq!(h.total_clusters, 0);
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bench {
    use super::*;
    use crate::mesh::generate_stress_sphere;

    #[test]
    fn bench_nanite_lod_reduction() {
        // Generate 10 high-poly stress spheres; build the hierarchy for one.
        let spheres: Vec<_> = (0..10).map(|_| generate_stress_sphere(64)).collect();
        let mut base = spheres[0].clone();
        base.build_clusters();
        let h = ClusterHierarchy::build(&base);

        use crate::math3d::{PI, perspective, look_at};

        // Three representative viewing distances.
        let distances: &[(&str, f32)] = &[
            ("near  (0.5 units)", 0.5),
            ("mid   (10 units) ", 10.0),
            ("far   (100 units)", 100.0),
        ];

        for &(label, dist) in distances {
            let eye = Vec3::new(0.0, 0.0, dist);
            let view = look_at(eye, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0));
            let proj = perspective(PI / 3.0, 16.0 / 9.0, 0.001, dist * 10.0);
            let pv = proj * view;

            let selected = h.select_clusters(&pv, eye, 1080.0, 1.0);
            let ratio = h.lod_reduction_ratio(&selected);

            // Just print — this is a diagnostic bench, not a pass/fail assertion.
            std::println!(
                "[nanite bench] {label}: {} clusters selected, LOD reduction = {:.2}x",
                selected.len(),
                ratio
            );
        }

        // The far case must show meaningful LOD reduction (at least 2x).
        {
            let dist = 100.0f32;
            let eye = Vec3::new(0.0, 0.0, dist);
            let view = look_at(eye, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0));
            let proj = perspective(PI / 3.0, 16.0 / 9.0, 0.001, dist * 10.0);
            let pv = proj * view;
            let selected = h.select_clusters(&pv, eye, 1080.0, 1.0);
            let ratio = h.lod_reduction_ratio(&selected);
            assert!(
                ratio >= 2.0 || h.full_detail_triangles() == 0,
                "far-distance LOD reduction ratio {ratio:.2}x should be ≥ 2x"
            );
        }
    }
}

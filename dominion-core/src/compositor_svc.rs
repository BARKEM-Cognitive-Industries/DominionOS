//! Compositor Service — retained desktop scene graph.
//!
//! Applications declare interface structures as scene nodes (vector primitives,
//! meshes, media buffers, text, particle systems). The compositor owns a single
//! unified 3D environment — the concept of a "window" is eliminated.
//!
//! NOTE: `CompositorService` is not yet wired into the production render path.
//! `os.rs` uses `compose::Board` as the active scene authority. This module is
//! a retained-mode compositor framework pending integration. It is currently
//! exercised only by `#[cfg(test)]` paths.
//!
//! Pure, safe `no_std`.

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;

use crate::math3d::{Vec3, Mat4};
use crate::rdg::{Rdg, RdgPass, PassKind, ResourceKind};

// ── NodeCapability ────────────────────────────────────────────────────────────

/// A capability tag on a scene node (identifies who owns it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeCapability {
    pub process_id: u64,
    /// true → capsule rendering (secure isolation)
    pub is_secure: bool,
}

// ── SceneNodeContent ──────────────────────────────────────────────────────────

/// The kind of content in a scene node (replaces "window").
#[derive(Clone, Debug)]
pub enum SceneNodeContent {
    VectorPrimitive {
        path_data: Vec<u8>,
        fill_color: [f32; 4],
        stroke_color: [f32; 4],
        stroke_width: f32,
    },
    MeshNode {
        mesh_id: u64,
        material_id: u64,
        transform: Mat4,
    },
    MediaBuffer {
        stream_id: u64,
        width: u32,
        height: u32,
        /// true = bypass compositor (direct overlay scanout)
        is_direct_overlay: bool,
    },
    TextStructure {
        text: String,
        font_size: f32,
        color: [f32; 4],
        transform: Mat4,
    },
    ParticleSystem {
        emitter_pos: Vec3,
        particle_count: u32,
        lifetime: f32,
        color: [f32; 4],
    },
    Container {
        /// Child node IDs
        children: Vec<u64>,
    },
}

// ── SceneNode ─────────────────────────────────────────────────────────────────

/// A node in the retained desktop scene graph.
#[derive(Clone, Debug)]
pub struct SceneNode {
    pub id: u64,
    pub capability: NodeCapability,
    pub content: SceneNodeContent,
    pub transform: Mat4,
    pub opacity: f32,
    pub z_order: i32,
    pub visible: bool,
    pub dirty: bool,
}

impl SceneNode {
    pub fn new(id: u64, cap: NodeCapability, content: SceneNodeContent) -> Self {
        Self {
            id,
            capability: cap,
            content,
            transform: Mat4::identity(),
            opacity: 1.0,
            z_order: 0,
            visible: true,
            dirty: true,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Returns true if this node is a direct media overlay (bypasses compositor).
    pub fn is_media_overlay(&self) -> bool {
        matches!(
            &self.content,
            SceneNodeContent::MediaBuffer { is_direct_overlay: true, .. }
        )
    }
}

// ── NodeSubmission ────────────────────────────────────────────────────────────

/// Submission packet from an application to the compositor.
pub struct NodeSubmission {
    pub node: SceneNode,
    pub timestamp_us: u64,
}

// ── CompositorError ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompositorError {
    UnauthorizedNode,
    InvalidContent,
    DuplicateId,
    QuotaExceeded,
}

// ── CompositorService ─────────────────────────────────────────────────────────

/// Per-process node quota (max nodes a single process may submit).
const PROCESS_NODE_QUOTA: usize = 1024;

/// The compositor service — manages the retained scene graph and compiles it to an RDG.
pub struct CompositorService {
    /// All live scene nodes, keyed by node ID.
    pub scene_graph: BTreeMap<u64, SceneNode>,
    /// Pending submissions not yet integrated.
    pub pending: Vec<NodeSubmission>,
    pub screen_width: u32,
    pub screen_height: u32,
}

impl CompositorService {
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            scene_graph: BTreeMap::new(),
            pending: Vec::new(),
            screen_width,
            screen_height,
        }
    }

    /// Submit a node from an application. Validated before insertion.
    pub fn submit_node(&mut self, submission: NodeSubmission) -> Result<u64, CompositorError> {
        let node = &submission.node;

        // Reject duplicate IDs.
        if self.scene_graph.contains_key(&node.id) {
            return Err(CompositorError::DuplicateId);
        }
        // Check pending for duplicate too.
        if self.pending.iter().any(|s| s.node.id == node.id) {
            return Err(CompositorError::DuplicateId);
        }

        // Quota check: count how many nodes this process already has.
        let process_id = node.capability.process_id;
        let existing_count = self
            .scene_graph
            .values()
            .filter(|n| n.capability.process_id == process_id)
            .count()
            + self
                .pending
                .iter()
                .filter(|s| s.node.capability.process_id == process_id)
                .count();
        if existing_count >= PROCESS_NODE_QUOTA {
            return Err(CompositorError::QuotaExceeded);
        }

        let id = node.id;
        self.pending.push(submission);
        Ok(id)
    }

    /// Remove a node (application teardown).
    pub fn remove_node(&mut self, id: u64) -> Option<SceneNode> {
        self.scene_graph.remove(&id)
    }

    /// Flush pending submissions into the scene graph.
    pub fn flush_pending(&mut self) {
        for submission in self.pending.drain(..) {
            let node = submission.node;
            self.scene_graph.insert(node.id, node);
        }
    }

    /// Collect the dirty region (union of dirty node bounds) as (x0, y0, x1, y1).
    ///
    /// For simplicity, dirty nodes contribute a nominal screen-covering region.
    /// Returns None if no dirty nodes.
    pub fn collect_dirty_region(&mut self) -> Option<(i32, i32, i32, i32)> {
        let mut any_dirty = false;
        let mut x0 = i32::MAX;
        let mut y0 = i32::MAX;
        let mut x1 = i32::MIN;
        let mut y1 = i32::MIN;

        for node in self.scene_graph.values_mut() {
            if node.dirty && node.visible {
                any_dirty = true;
                // Use z_order as a nominal offset for distinguishable bounds.
                let nx0 = 0_i32;
                let ny0 = 0_i32;
                let nx1 = self.screen_width as i32;
                let ny1 = self.screen_height as i32;
                if nx0 < x0 { x0 = nx0; }
                if ny0 < y0 { y0 = ny0; }
                if nx1 > x1 { x1 = nx1; }
                if ny1 > y1 { y1 = ny1; }
                node.dirty = false;
            }
        }

        if any_dirty {
            Some((x0, y0, x1, y1))
        } else {
            None
        }
    }

    /// Compile the scene graph into an RDG for the current frame.
    /// Only live, visible, non-overlay nodes are included.
    pub fn compile_frame_rdg(&self) -> Rdg {
        let mut rdg = Rdg::new();

        // Add a colour target for the composited output.
        let color_target = rdg.add_resource(
            ResourceKind::ColorTarget {
                w: self.screen_width,
                h: self.screen_height,
            },
            true,
            "compositor_output",
        );
        let swapchain = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swapchain");

        // For each visible, non-overlay node, add a render pass.
        let mut pass_resources: Vec<u32> = Vec::new();
        for node in self.sorted_nodes() {
            if node.is_media_overlay() {
                continue;
            }
            let kind = match &node.content {
                SceneNodeContent::VectorPrimitive { .. } => PassKind::Graphics,
                SceneNodeContent::MeshNode { .. } => PassKind::Graphics,
                SceneNodeContent::TextStructure { .. } => PassKind::Graphics,
                SceneNodeContent::ParticleSystem { .. } => PassKind::Compute,
                SceneNodeContent::Container { .. } => PassKind::Graphics,
                SceneNodeContent::MediaBuffer { .. } => PassKind::Graphics,
            };

            // Each node gets a small buffer resource representing its geometry/data.
            let node_buf = rdg.add_resource(
                ResourceKind::Buffer { bytes: 256 },
                true,
                "node_data",
            );
            pass_resources.push(node_buf);

            let reads = if pass_resources.len() > 1 {
                alloc::vec![pass_resources[pass_resources.len() - 2]]
            } else {
                alloc::vec![]
            };

            rdg.add_pass(RdgPass {
                name: "node_pass",
                reads,
                writes: alloc::vec![node_buf],
                kind,
            });
        }

        // Composite pass: reads the last node buffer, writes color_target.
        let reads = if let Some(&last) = pass_resources.last() {
            alloc::vec![last]
        } else {
            alloc::vec![]
        };
        rdg.add_pass(RdgPass {
            name: "composite",
            reads,
            writes: alloc::vec![color_target],
            kind: PassKind::Graphics,
        });

        // Present pass.
        rdg.add_pass(RdgPass {
            name: "present",
            reads: alloc::vec![color_target],
            writes: alloc::vec![swapchain],
            kind: PassKind::Present,
        });

        rdg
    }

    /// Get all visible nodes sorted by z_order (ascending).
    pub fn sorted_nodes(&self) -> Vec<&SceneNode> {
        let mut nodes: Vec<&SceneNode> = self
            .scene_graph
            .values()
            .filter(|n| n.visible)
            .collect();
        nodes.sort_by_key(|n| n.z_order);
        nodes
    }

    /// Total scene node count.
    pub fn node_count(&self) -> usize {
        self.scene_graph.len()
    }

    /// Count of direct media overlay nodes (bypass compositor).
    pub fn overlay_count(&self) -> usize {
        self.scene_graph.values().filter(|n| n.is_media_overlay()).count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cap(pid: u64) -> NodeCapability {
        NodeCapability { process_id: pid, is_secure: false }
    }

    fn vector_node(id: u64, pid: u64) -> SceneNode {
        SceneNode::new(
            id,
            make_cap(pid),
            SceneNodeContent::VectorPrimitive {
                path_data: alloc::vec![1, 2, 3],
                fill_color: [1.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
            },
        )
    }

    fn media_node(id: u64, pid: u64, overlay: bool) -> SceneNode {
        SceneNode::new(
            id,
            make_cap(pid),
            SceneNodeContent::MediaBuffer {
                stream_id: id,
                width: 1920,
                height: 1080,
                is_direct_overlay: overlay,
            },
        )
    }

    fn submission(node: SceneNode) -> NodeSubmission {
        NodeSubmission { node, timestamp_us: 1000 }
    }

    // 1. Submit a single node successfully.
    #[test]
    fn test_submit_node_success() {
        let mut svc = CompositorService::new(1920, 1080);
        let node = vector_node(1, 100);
        let result = svc.submit_node(submission(node));
        assert_eq!(result, Ok(1));
        assert_eq!(svc.pending.len(), 1);
    }

    // 2. Duplicate ID is rejected.
    #[test]
    fn test_submit_duplicate_id_rejected() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        let result = svc.submit_node(submission(vector_node(1, 100)));
        assert_eq!(result, Err(CompositorError::DuplicateId));
    }

    // 3. Duplicate ID in scene_graph is rejected.
    #[test]
    fn test_submit_duplicate_after_flush() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        svc.flush_pending();
        let result = svc.submit_node(submission(vector_node(1, 100)));
        assert_eq!(result, Err(CompositorError::DuplicateId));
    }

    // 4. flush_pending moves nodes into scene_graph.
    #[test]
    fn test_flush_pending() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        svc.submit_node(submission(vector_node(2, 100))).unwrap();
        assert_eq!(svc.pending.len(), 2);
        assert_eq!(svc.node_count(), 0);
        svc.flush_pending();
        assert_eq!(svc.pending.len(), 0);
        assert_eq!(svc.node_count(), 2);
    }

    // 5. sorted_nodes returns visible nodes by z_order ascending.
    #[test]
    fn test_sorted_nodes_by_z_order() {
        let mut svc = CompositorService::new(1920, 1080);
        let mut n1 = vector_node(1, 100);
        n1.z_order = 10;
        let mut n2 = vector_node(2, 100);
        n2.z_order = -5;
        let mut n3 = vector_node(3, 100);
        n3.z_order = 0;
        svc.submit_node(submission(n1)).unwrap();
        svc.submit_node(submission(n2)).unwrap();
        svc.submit_node(submission(n3)).unwrap();
        svc.flush_pending();
        let sorted = svc.sorted_nodes();
        assert_eq!(sorted.len(), 3);
        assert!(sorted[0].z_order <= sorted[1].z_order);
        assert!(sorted[1].z_order <= sorted[2].z_order);
        assert_eq!(sorted[0].z_order, -5);
        assert_eq!(sorted[2].z_order, 10);
    }

    // 6. invisible nodes are excluded from sorted_nodes.
    #[test]
    fn test_sorted_nodes_excludes_invisible() {
        let mut svc = CompositorService::new(1920, 1080);
        let mut n1 = vector_node(1, 100);
        n1.visible = false;
        let n2 = vector_node(2, 100);
        svc.submit_node(submission(n1)).unwrap();
        svc.submit_node(submission(n2)).unwrap();
        svc.flush_pending();
        let sorted = svc.sorted_nodes();
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].id, 2);
    }

    // 7. dirty region collection returns Some when dirty nodes exist.
    #[test]
    fn test_collect_dirty_region_some() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        svc.flush_pending();
        let region = svc.collect_dirty_region();
        assert!(region.is_some());
        let (x0, y0, x1, y1) = region.unwrap();
        assert!(x0 < x1);
        assert!(y0 < y1);
    }

    // 8. dirty region returns None after all dirty cleared.
    #[test]
    fn test_collect_dirty_region_cleared() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        svc.flush_pending();
        svc.collect_dirty_region(); // clears dirty flag
        let region = svc.collect_dirty_region();
        assert!(region.is_none());
    }

    // 9. overlay_count counts only direct overlay media nodes.
    #[test]
    fn test_overlay_count() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(media_node(1, 100, true))).unwrap();
        svc.submit_node(submission(media_node(2, 100, false))).unwrap();
        svc.submit_node(submission(vector_node(3, 100))).unwrap();
        svc.flush_pending();
        assert_eq!(svc.overlay_count(), 1);
    }

    // 10. is_media_overlay correctly identifies overlay nodes.
    #[test]
    fn test_is_media_overlay() {
        let overlay = media_node(1, 100, true);
        let composited = media_node(2, 100, false);
        let vector = vector_node(3, 100);
        assert!(overlay.is_media_overlay());
        assert!(!composited.is_media_overlay());
        assert!(!vector.is_media_overlay());
    }

    // 11. remove_node returns the node and removes it from the graph.
    #[test]
    fn test_remove_node() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        svc.flush_pending();
        assert_eq!(svc.node_count(), 1);
        let removed = svc.remove_node(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, 1);
        assert_eq!(svc.node_count(), 0);
    }

    // 12. remove_node on missing ID returns None.
    #[test]
    fn test_remove_node_missing() {
        let mut svc = CompositorService::new(1920, 1080);
        let result = svc.remove_node(999);
        assert!(result.is_none());
    }

    // 13. compile_frame_rdg produces a valid RDG with passes.
    #[test]
    fn test_compile_frame_rdg_valid() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(vector_node(1, 100))).unwrap();
        svc.submit_node(submission(vector_node(2, 100))).unwrap();
        svc.flush_pending();
        let rdg = svc.compile_frame_rdg();
        // Must have at least the composite + present pass.
        assert!(rdg.pass_count() >= 2);
        assert!(rdg.resource_count() >= 2);
    }

    // 14. compile_frame_rdg with no non-overlay nodes still produces composite+present.
    #[test]
    fn test_compile_frame_rdg_only_overlays() {
        let mut svc = CompositorService::new(1920, 1080);
        svc.submit_node(submission(media_node(1, 100, true))).unwrap();
        svc.flush_pending();
        let rdg = svc.compile_frame_rdg();
        // Should still have composite and present passes.
        assert!(rdg.pass_count() >= 2);
    }

    // 15. compile_frame_rdg compiles without panic and has a Present pass.
    #[test]
    fn test_compile_frame_rdg_compiles() {
        let mut svc = CompositorService::new(3840, 2160);
        for i in 1..=5 {
            svc.submit_node(submission(vector_node(i, 42))).unwrap();
        }
        svc.flush_pending();
        let rdg = svc.compile_frame_rdg();
        let compiled = rdg.compile();
        // Present pass must be live.
        assert!(!compiled.execution_order.is_empty());
    }

    // 16. mark_dirty sets dirty flag.
    #[test]
    fn test_mark_dirty() {
        let mut node = vector_node(1, 100);
        node.dirty = false;
        node.mark_dirty();
        assert!(node.dirty);
    }

    // 17. Multiple processes, each below quota, all succeed.
    #[test]
    fn test_multiple_processes_submit() {
        let mut svc = CompositorService::new(1920, 1080);
        for i in 0..10u64 {
            let node = vector_node(i, i); // each in its own process
            svc.submit_node(submission(node)).unwrap();
        }
        svc.flush_pending();
        assert_eq!(svc.node_count(), 10);
    }

    // 18. Container node is not an overlay.
    #[test]
    fn test_container_not_overlay() {
        let node = SceneNode::new(
            1,
            make_cap(1),
            SceneNodeContent::Container { children: alloc::vec![2, 3] },
        );
        assert!(!node.is_media_overlay());
    }
}

//! Clean public API surface for the DominionOS 3D render library.
//!
//! This module re-exports the key types from every subsystem so OS code, shell
//! pages, and applications can reach them through a single `use`:
//!
//! ```rust,ignore
//! use dominion_core::render3d::prelude::*;
//! ```
//!
//! ## Usage pattern (inside the OS / kernel)
//!
//! ```rust,ignore
//! use dominion_core::render3d::prelude::*;
//!
//! // 1. Build or load a mesh.
//! let mesh = generate_stress_sphere(32);   // ~4 k triangles
//!
//! // 2. Create an off-screen render target the same size as the viewport.
//! let mut rt = RenderTarget::new(width, height);
//! rt.clear_all(0xFF_10_10_18);
//!
//! // 3. Set up camera and lights.
//! let camera = Camera::new(eye, target, 65.0, aspect, 0.1, 1000.0);
//! let state  = RenderState::default_lit();
//! let mat    = Material::metallic_roughness([0.8, 0.4, 0.1], 0.3, 0.7);
//!
//! // 4. Draw. All transforms are Mat4 (column-major).
//! draw_mesh(&mut rt, &mesh, &Mat4::identity(), &camera.proj_view(), &state, &mat);
//!
//! // 5. Get the RRGGBB pixels and composite them via the kernel gfx layer.
//! //    (.pixels is pub Vec<u32> in 0x00RRGGBB format, ready for the back-buffer)
//! let pixels = &rt.pixels;
//!
//! // For Nanite-style LOD at scale:
//! let mut hier = ClusterHierarchy::build(&mesh);
//! let selected = hier.select_clusters(&camera.proj_view(), eye, height as f32, 1.5);
//! draw_mesh_clusters(&mut rt, &mesh, &selected, &Mat4::identity(),
//!                    &camera.proj_view(), &state, &mat);
//!
//! // For hierarchical instancing (1 M+ objects, O(log N) culling):
//! let mut ir = InstanceRenderer::new();
//! for i in 0..count { ir.add(Instance::new(i, 0, 0, transform[i], local_aabb)); }
//! ir.build_bvh();
//! let batches = ir.cull_and_batch(&camera.frustum());
//! for batch in &batches {
//!     draw_instanced(&mut rt, &mesh, &batch.transforms, &camera.proj_view(), &state, &mat);
//! }
//!
//! // For vertex animations (skeletal or morph-target):
//! let deformed = skinned_mesh.skin(&skin_matrices);
//! draw_mesh(&mut rt, &deformed, &model, &camera.proj_view(), &state, &mat);
//! ```

// ── Math & geometry ───────────────────────────────────────────────────────────
pub use crate::math3d::{
    Vec2, Vec3, Vec4, Mat4, Quat, Aabb, Frustum, Ray,
    translation, scale, perspective, look_at,
    sin32, cos32, sqrt32,
};

// ── Mesh types ────────────────────────────────────────────────────────────────
pub use crate::mesh::{
    Vertex, Material, Cluster, Mesh,
    generate_sphere, generate_cube, generate_grid,
    generate_stress_sphere, generate_scene_mesh,
    decimate,
};

// ── Scene graph ───────────────────────────────────────────────────────────────
pub use crate::scene3d::{
    Camera, Transform, Scene, SceneNode, NodeContent, NodeId,
    build_demo_scene,
};

// ── Software rasterizer ───────────────────────────────────────────────────────
pub use crate::raster3d::{
    RenderTarget, RenderState, DirectionalLight,
    draw_mesh, draw_mesh_clusters, draw_instanced,
};

// ── Nanite virtual geometry ───────────────────────────────────────────────────
pub use crate::nanite::{ClusterHierarchy, ClusterNode};

// ── Hierarchical BVH instancing ───────────────────────────────────────────────
pub use crate::instances::{Instance, InstanceRenderer, DrawBatch, Bvh};

// ── Vertex animations ─────────────────────────────────────────────────────────
pub use crate::vertanim::{
    MorphMesh, MorphTarget, Skeleton, Bone,
    SkinnedMesh, SkinVertex, AnimClip, AnimChannel,
    PosKey, RotKey, ScaleKey, Animator,
};

// ── Render dependency graph ───────────────────────────────────────────────────
pub use crate::rdg::{Rdg, RdgPass, PassKind, ResourceKind, CompiledRdg};

// ── HDR pipeline ──────────────────────────────────────────────────────────────
pub use crate::hdr::{HdrPixel, HdrBuffer};

// ── Vector path rasterizer ────────────────────────────────────────────────────
pub use crate::vectorpath::{VectorPath, FillRule, rasterize_path, rasterize_tiled_prefix};

// ── Font / glyph rendering ────────────────────────────────────────────────────
pub use crate::fontgpu::{
    GlyphOutline, GlyphRun, render_text, rasterize_glyph,
    compute_dilation_quadratic, process_glyph_outline,
};

// ── SDF shadows ───────────────────────────────────────────────────────────────
pub use crate::sdf_shadow::{SdfField, ShadowParams, ray_march_shadow, render_sdf_shadow};

// ── Asynchronous TimeWarp ─────────────────────────────────────────────────────
pub use crate::atw::{Atw, KinematicState, RotKinematic, CapturedFrame};

// ── Compositor scene graph ────────────────────────────────────────────────────
pub use crate::compositor_svc::{
    CompositorService, SceneNode as CompositorNode,
    SceneNodeContent, NodeCapability,
};

// ── Media service ─────────────────────────────────────────────────────────────
pub use crate::media_service::{MediaService, VideoStreamBuffer, ZeroCopyToken, PixelFormat};

// ── IDAG / federated GPU scheduler ───────────────────────────────────────────
pub use crate::idag::{IDAG, FedcmScheduler, GpuSlice, SlicePriority};

// ── Security & provenance ─────────────────────────────────────────────────────
pub use crate::secnode::{RenderCapToken, SecureMemoryCapsule, CapabilityRoutingGuard};
pub use crate::render_provenance::{ProvenanceManifest, ProvenanceVerifier, IntegrityState};

// ── Deterministic rendering ───────────────────────────────────────────────────
pub use crate::render_determinism::{DeterministicRenderer, RenderCommand, cleanse_registers};

// ── Input latching / PSR2 ─────────────────────────────────────────────────────
pub use crate::input_latch::{InputRingBuffer, InputCoord, late_latch};
pub use crate::psr2::{DamageTracker, DirtyRect, PaintOverController, RenderQuality};

// ── Benchmarks (available for shell/diagnostic pages) ─────────────────────────
pub use crate::render_bench::{
    BenchResult, PpmFrame,
    bench_single_mesh_render, bench_nanite_lod_reduction,
    bench_instancing, bench_vertex_animation,
    bench_full_pipeline_video, bench_rdg_scene,
};

/// A convenience prelude — `use dominion_core::render3d::prelude::*` pulls in
/// every type needed for typical 3D rendering work.
pub mod prelude {
    pub use super::*;
}

// ── RenderState convenience constructor ───────────────────────────────────────
impl RenderState {
    /// A sensible default: one warm key light and a cool fill, low ambient.
    pub fn default_lit() -> Self {
        use crate::raster3d::DirectionalLight;
        RenderState {
            lights: alloc::vec![
                DirectionalLight {
                    direction: crate::math3d::Vec3::new(0.5, -1.0, -0.5).normalize(),
                    color: [1.0, 0.95, 0.9],
                    intensity: 1.3,
                },
                DirectionalLight {
                    direction: crate::math3d::Vec3::new(-1.0, 0.5, 1.0).normalize(),
                    color: [0.3, 0.4, 0.6],
                    intensity: 0.4,
                },
            ],
            ambient: [0.04, 0.04, 0.07],
            wireframe: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn prelude_mesh_roundtrip() {
        let mesh = generate_sphere(1.0, 8, 8);
        assert!(mesh.vertices.len() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn prelude_render_produces_pixels() {
        let mesh = generate_sphere(1.0, 8, 8);
        let mut rt = RenderTarget::new(64, 64);
        rt.clear_all(0xFF_10_10_18);
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 3.0), Vec3::zero(),
            65.0, 1.0, 0.1, 100.0,
        );
        let state = RenderState::default_lit();
        let mat = Material::metallic_roughness([0.8, 0.3, 0.1], 0.2, 0.8);
        draw_mesh(&mut rt, &mesh, &Mat4::identity(), &cam.proj_view(), &state, &mat);
        let nonblack = rt.pixels.iter().filter(|&&p| p != 0xFF_10_10_18).count();
        assert!(nonblack > 0, "3D render should produce non-background pixels");
    }

    #[test]
    fn prelude_nanite_lod_accessible() {
        let mut mesh = generate_stress_sphere(16);
        mesh.build_clusters();
        let hier = ClusterHierarchy::build(&mesh);
        assert!(hier.full_detail_triangles() > 0);
    }

    #[test]
    fn prelude_instancing_accessible() {
        let mesh = generate_sphere(0.5, 4, 4);
        let aabb = mesh.aabb;
        let mut ir = InstanceRenderer::new();
        for i in 0..10u32 {
            let t = translation(Vec3::new(i as f32 * 2.0, 0.0, 0.0));
            ir.add(Instance::new(i, 0, 0, t, aabb));
        }
        ir.build_bvh();
        assert_eq!(ir.last_visible_count(), 0); // no cull called yet, so 0
    }

    #[test]
    fn prelude_default_lit_state() {
        let s = RenderState::default_lit();
        assert_eq!(s.lights.len(), 2);
        assert!(!s.wireframe);
    }

    #[test]
    fn prelude_hdr_accessible() {
        let p = HdrPixel::new(1.0, 0.5, 0.0, 1.0);
        assert!(p.luminance() > 0.0);
    }

    #[test]
    fn prelude_damage_tracker_accessible() {
        let mut dt = DamageTracker::new();
        dt.mark_dirty(DirtyRect::new(0, 0, 100, 100));
        assert!(dt.end_frame().is_some());
    }
}

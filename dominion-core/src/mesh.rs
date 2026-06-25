//! Mesh data structures, procedural generators, and cluster partitioning
//! for the DominionOS unified 2D/3D renderer.
//! (see `docs/2d-3d rendering redesign.md`)
//!
//! Pure, safe `no_std`.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::math3d::{Aabb, Vec2, Vec3, Vec4, cos32, sin32};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of triangles per Nanite-style cluster.
pub const CLUSTER_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// Vertex
// ---------------------------------------------------------------------------

/// A fully-specified vertex for 3D rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// Object-space position.
    pub pos: Vec3,
    /// Object-space normal (unit vector).
    pub normal: Vec3,
    /// Texture coordinates in [0..1].
    pub uv: Vec2,
    /// Tangent vector; w component is the bitangent sign (-1 or +1).
    pub tangent: Vec4,
    /// Per-vertex RGBA colour (linear); default white [1,1,1,1].
    pub color: [f32; 4],
}

impl Vertex {
    /// Construct a vertex with default colour (white) and zero tangent.
    #[inline]
    pub fn new(pos: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self {
            pos,
            normal,
            uv,
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// PBR-style surface material.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// Linear-space base albedo RGBA.
    pub base_color: [f32; 4],
    /// 0 = mirror, 1 = matte.
    pub roughness: f32,
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// HDR emission (may exceed 1.0).
    pub emissive: [f32; 3],
    pub double_sided: bool,
}

impl Material {
    /// Plain white diffuse material.
    pub fn default_white() -> Material {
        Material {
            name: String::from("default_white"),
            base_color: [1.0, 1.0, 1.0, 1.0],
            roughness: 0.8,
            metallic: 0.0,
            emissive: [0.0, 0.0, 0.0],
            double_sided: false,
        }
    }

    /// Self-luminous material. `intensity` scales the colour linearly.
    pub fn emissive(color: [f32; 3], intensity: f32) -> Material {
        Material {
            name: String::from("emissive"),
            base_color: [color[0], color[1], color[2], 1.0],
            roughness: 1.0,
            metallic: 0.0,
            emissive: [color[0] * intensity, color[1] * intensity, color[2] * intensity],
            double_sided: false,
        }
    }

    /// Standard metallic/roughness material with a solid base colour.
    pub fn metallic_roughness(base: [f32; 3], metallic: f32, roughness: f32) -> Material {
        Material {
            name: String::from("metallic_roughness"),
            base_color: [base[0], base[1], base[2], 1.0],
            roughness,
            metallic,
            emissive: [0.0, 0.0, 0.0],
            double_sided: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster
// ---------------------------------------------------------------------------

/// A triangle cluster — the atomic unit of Nanite-style rendering.
///
/// Each cluster holds at most [`CLUSTER_SIZE`] triangles.
#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    /// First index in the owning mesh's index buffer.
    pub start_index: u32,
    /// Number of indices covered (= triangles × 3).
    pub count: u32,
    /// Cluster's world-space bounding box (updated on upload).
    pub aabb: Aabb,
    /// Screen-space error budget for LOD selection.
    pub lod_error: f32,
    /// LOD level: 0 = full detail, higher = coarser.
    pub lod_level: u8,
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// A triangle mesh with optional cluster and LOD data.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Axis-aligned bounding box in object space.
    pub aabb: Aabb,
    /// Pre-computed clusters (empty until [`Mesh::build_clusters`] is called).
    pub clusters: Vec<Cluster>,
    /// Index into the scene's material array.
    pub material_idx: usize,
}

impl Mesh {
    /// Create a new mesh from explicit vertex and index data.
    /// Computes the AABB immediately; tangents and clusters must be requested separately.
    pub fn new(name: &str, vertices: Vec<Vertex>, indices: Vec<u32>) -> Mesh {
        let mut m = Mesh {
            name: String::from(name),
            vertices,
            indices,
            aabb: Aabb::new(Vec3::zero(), Vec3::zero()),
            clusters: Vec::new(),
            material_idx: 0,
        };
        m.compute_aabb();
        m
    }

    /// Recompute the AABB from the current vertex positions.
    pub fn compute_aabb(&mut self) {
        if self.vertices.is_empty() {
            self.aabb = Aabb::new(Vec3::zero(), Vec3::zero());
            return;
        }
        let mut mn = self.vertices[0].pos;
        let mut mx = self.vertices[0].pos;
        for v in &self.vertices[1..] {
            mn = mn.min_elem(v.pos);
            mx = mx.max_elem(v.pos);
        }
        self.aabb = Aabb::new(mn, mx);
    }

    /// Compute per-vertex tangents using a Mikktspace-style accumulation.
    ///
    /// For every triangle the UV-space tangent (T) and bitangent (B) are
    /// derived from the position/UV deltas, then accumulated into each
    /// corner vertex and normalised.  The bitangent sign is stored in
    /// `tangent.w`.
    pub fn compute_tangents(&mut self) {
        let n = self.vertices.len();
        let mut tan1: Vec<Vec3> = (0..n).map(|_| Vec3::zero()).collect();
        let mut tan2: Vec<Vec3> = (0..n).map(|_| Vec3::zero()).collect();

        let tri_count = self.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = self.indices[t * 3] as usize;
            let i1 = self.indices[t * 3 + 1] as usize;
            let i2 = self.indices[t * 3 + 2] as usize;

            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];

            let e1 = v1.pos - v0.pos;
            let e2 = v2.pos - v0.pos;

            let du1 = v1.uv.x - v0.uv.x;
            let dv1 = v1.uv.y - v0.uv.y;
            let du2 = v2.uv.x - v0.uv.x;
            let dv2 = v2.uv.y - v0.uv.y;

            let denom = du1 * dv2 - du2 * dv1;
            let r = if denom.abs() < 1e-8 { 1.0 } else { 1.0 / denom };

            let sdir = Vec3::new(
                (dv2 * e1.x - dv1 * e2.x) * r,
                (dv2 * e1.y - dv1 * e2.y) * r,
                (dv2 * e1.z - dv1 * e2.z) * r,
            );
            let tdir = Vec3::new(
                (du1 * e2.x - du2 * e1.x) * r,
                (du1 * e2.y - du2 * e1.y) * r,
                (du1 * e2.z - du2 * e1.z) * r,
            );

            tan1[i0] = tan1[i0] + sdir;
            tan1[i1] = tan1[i1] + sdir;
            tan1[i2] = tan1[i2] + sdir;

            tan2[i0] = tan2[i0] + tdir;
            tan2[i1] = tan2[i1] + tdir;
            tan2[i2] = tan2[i2] + tdir;
        }

        for i in 0..n {
            let n_v = self.vertices[i].normal;
            let t = tan1[i];

            // Gram-Schmidt orthogonalise T with respect to N.
            let t_ortho = (t - n_v * n_v.dot(t)).normalize();

            // Determine handedness: if N×T and B are in the same direction → +1, else −1.
            let sign = if n_v.cross(t).dot(tan2[i]) < 0.0 { -1.0f32 } else { 1.0f32 };

            self.vertices[i].tangent = Vec4::new(t_ortho.x, t_ortho.y, t_ortho.z, sign);
        }
    }

    /// Partition the index buffer into sequential [`CLUSTER_SIZE`]-triangle clusters.
    /// Each cluster's AABB is computed from the vertex positions it references.
    pub fn build_clusters(&mut self) {
        self.clusters.clear();

        let total_indices = self.indices.len();
        let indices_per_cluster = CLUSTER_SIZE * 3;

        let mut start = 0usize;
        while start < total_indices {
            let end = (start + indices_per_cluster).min(total_indices);
            // Round down to a triangle boundary.
            let end = end - (end - start) % 3;
            if end <= start {
                break;
            }

            // Compute cluster AABB.
            let slice = &self.indices[start..end];
            let first_pos = self.vertices[slice[0] as usize].pos;
            let mut mn = first_pos;
            let mut mx = first_pos;
            for &idx in &slice[1..] {
                let p = self.vertices[idx as usize].pos;
                mn = mn.min_elem(p);
                mx = mx.max_elem(p);
            }

            self.clusters.push(Cluster {
                start_index: start as u32,
                count: (end - start) as u32,
                aabb: Aabb::new(mn, mx),
                lod_error: 0.0,
                lod_level: 0,
            });

            start = end;
        }
    }

    /// Total number of triangles in the mesh.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of precomputed clusters.
    #[inline]
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
}

// ---------------------------------------------------------------------------
// MeshHandle — shared-ownership wrapper for DrawCmd::Mesh3D
// ---------------------------------------------------------------------------

/// A reference-counted, cloneable handle to a [`Mesh`].
///
/// `DrawCmd::Mesh3D` carries a `MeshHandle` so the same mesh data can be
/// submitted from multiple windows or draw calls without copying.  Equality
/// is tested by **pointer identity** (two handles are equal iff they point to
/// the same allocation), not by deep mesh comparison.
#[derive(Clone, Debug)]
pub struct MeshHandle(pub Arc<Mesh>);

impl MeshHandle {
    pub fn new(mesh: Mesh) -> Self { Self(Arc::new(mesh)) }
}

impl PartialEq for MeshHandle {
    fn eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }
}

// ---------------------------------------------------------------------------
// Procedural generators
// ---------------------------------------------------------------------------

/// Unit sphere with `stacks` latitude rings and `slices` longitude segments.
///
/// Vertex layout:
/// - `(stacks + 1) * (slices + 1)` vertices (poles are duplicated per slice so
///   UVs are unambiguous and tangents are valid).
/// - `stacks * slices * 6` indices.
///
/// Normals point outward; UVs are standard spherical mapping; tangents are
/// computed analytically.
pub fn generate_sphere(radius: f32, stacks: u32, slices: u32) -> Mesh {
    use crate::math3d::PI;
    let stacks = stacks.max(2);
    let slices = slices.max(3);

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Generate (stacks+1) rows × (slices+1) columns of vertices.
    for stack in 0..=stacks {
        // φ from −π/2 (south pole) to +π/2 (north pole).
        let phi = -PI * 0.5 + PI * (stack as f32) / (stacks as f32);
        let cos_phi = cos32(phi);
        let sin_phi = sin32(phi);
        let v_coord = 1.0 - (stack as f32) / (stacks as f32); // top=0, bottom=1? Convention: top UV=0.

        for slice in 0..=slices {
            // θ from 0 to 2π.
            let theta = 2.0 * PI * (slice as f32) / (slices as f32);
            let cos_theta = cos32(theta);
            let sin_theta = sin32(theta);

            // Normal / position on unit sphere.
            let nx = cos_phi * cos_theta;
            let ny = sin_phi;
            let nz = cos_phi * sin_theta;

            let pos = Vec3::new(nx * radius, ny * radius, nz * radius);
            // Normalise explicitly: sin32/cos32 introduce small polynomial errors
            // that accumulate in the composed normal, so we cannot rely on the
            // trigonometric identity cos²φ + sin²φ = 1 holding exactly.
            let normal = Vec3::new(nx, ny, nz).normalize();
            let u_coord = (slice as f32) / (slices as f32);

            // Analytic tangent: dPos/dTheta, normalised.
            // dPos/dTheta = (-sin_theta * cos_phi, 0, cos_theta * cos_phi) * radius
            let tan3 = Vec3::new(-sin_theta, 0.0, cos_theta).normalize();
            let tangent = Vec4::new(tan3.x, tan3.y, tan3.z, 1.0);

            vertices.push(Vertex {
                pos,
                normal,
                uv: Vec2::new(u_coord, v_coord),
                tangent,
                color: [1.0, 1.0, 1.0, 1.0],
            });
        }
    }

    // Build quad-strip indices.
    let ring = slices + 1;
    for stack in 0..stacks {
        for slice in 0..slices {
            let a = stack * ring + slice;
            let b = a + ring;
            let c = a + 1;
            let d = b + 1;

            // Two triangles per quad.
            indices.push(a);
            indices.push(b);
            indices.push(c);

            indices.push(c);
            indices.push(b);
            indices.push(d);
        }
    }

    Mesh::new("sphere", vertices, indices)
}

/// Axis-aligned cube with half-extent `half`.
///
/// 24 vertices (4 per face), 36 indices.  Each face has its own set of
/// vertices so normals are face-flat and tangents are well-defined.
pub fn generate_cube(half: f32) -> Mesh {
    let h = half;

    // Each face: 4 vertices, 2 triangles (6 indices).
    // Order: (pos, normal, uv_top_left, uv_top_right, uv_bottom_left, uv_bottom_right)
    // Faces: +X, -X, +Y, -Y, +Z, -Z

    struct FaceDef {
        normal: [f32; 3],
        // Four corners in CCW winding (viewed from outside): bl, br, tr, tl
        corners: [[f32; 3]; 4],
        // Tangent direction (in world space)
        tangent: [f32; 3],
    }

    let faces = [
        // +X
        FaceDef {
            normal: [1.0, 0.0, 0.0],
            corners: [
                [ h, -h,  h],
                [ h, -h, -h],
                [ h,  h, -h],
                [ h,  h,  h],
            ],
            tangent: [0.0, 0.0, -1.0],
        },
        // -X
        FaceDef {
            normal: [-1.0, 0.0, 0.0],
            corners: [
                [-h, -h, -h],
                [-h, -h,  h],
                [-h,  h,  h],
                [-h,  h, -h],
            ],
            tangent: [0.0, 0.0, 1.0],
        },
        // +Y
        FaceDef {
            normal: [0.0, 1.0, 0.0],
            corners: [
                [-h,  h,  h],
                [ h,  h,  h],
                [ h,  h, -h],
                [-h,  h, -h],
            ],
            tangent: [1.0, 0.0, 0.0],
        },
        // -Y
        FaceDef {
            normal: [0.0, -1.0, 0.0],
            corners: [
                [-h, -h, -h],
                [ h, -h, -h],
                [ h, -h,  h],
                [-h, -h,  h],
            ],
            tangent: [1.0, 0.0, 0.0],
        },
        // +Z
        FaceDef {
            normal: [0.0, 0.0, 1.0],
            corners: [
                [-h, -h,  h],
                [ h, -h,  h],
                [ h,  h,  h],
                [-h,  h,  h],
            ],
            tangent: [1.0, 0.0, 0.0],
        },
        // -Z
        FaceDef {
            normal: [0.0, 0.0, -1.0],
            corners: [
                [ h, -h, -h],
                [-h, -h, -h],
                [-h,  h, -h],
                [ h,  h, -h],
            ],
            tangent: [-1.0, 0.0, 0.0],
        },
    ];

    let uvs = [
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
    ];

    let mut vertices: Vec<Vertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    for face in &faces {
        let base = vertices.len() as u32;
        let n = Vec3::new(face.normal[0], face.normal[1], face.normal[2]);
        let t = Vec3::new(face.tangent[0], face.tangent[1], face.tangent[2]);
        let tangent = Vec4::new(t.x, t.y, t.z, 1.0);

        for (corner, uv) in face.corners.iter().zip(uvs.iter()) {
            vertices.push(Vertex {
                pos: Vec3::new(corner[0], corner[1], corner[2]),
                normal: n,
                uv: *uv,
                tangent,
                color: [1.0, 1.0, 1.0, 1.0],
            });
        }

        // Two triangles: (0,1,2) and (0,2,3)
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
        indices.push(base);
        indices.push(base + 2);
        indices.push(base + 3);
    }

    Mesh::new("cube", vertices, indices)
}

/// Flat grid in the XZ plane, `cols × rows` quads, centred at the origin.
///
/// Normals point up (+Y); UVs tile once over the whole grid.
pub fn generate_grid(cols: u32, rows: u32, cell_size: f32) -> Mesh {
    let cols = cols.max(1);
    let rows = rows.max(1);

    let width = cols as f32 * cell_size;
    let depth = rows as f32 * cell_size;

    let verts_x = cols + 1;
    let verts_z = rows + 1;

    let mut vertices: Vec<Vertex> = Vec::with_capacity((verts_x * verts_z) as usize);
    let mut indices: Vec<u32> = Vec::with_capacity((cols * rows * 6) as usize);

    for z in 0..verts_z {
        for x in 0..verts_x {
            let px = -width * 0.5 + x as f32 * cell_size;
            let pz = -depth * 0.5 + z as f32 * cell_size;
            let u = x as f32 / cols as f32;
            let v = z as f32 / rows as f32;

            vertices.push(Vertex {
                pos: Vec3::new(px, 0.0, pz),
                normal: Vec3::new(0.0, 1.0, 0.0),
                uv: Vec2::new(u, v),
                tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
                color: [1.0, 1.0, 1.0, 1.0],
            });
        }
    }

    for z in 0..rows {
        for x in 0..cols {
            let a = z * verts_x + x;
            let b = a + verts_x;

            // CCW from above (+Y looking down).
            indices.push(a);
            indices.push(b);
            indices.push(a + 1);

            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
        }
    }

    Mesh::new("grid", vertices, indices)
}

/// High-poly sphere suitable for stress testing.
///
/// - `density = 64`  → ~8 192 triangles
/// - `density = 128` → ~32 768 triangles
pub fn generate_stress_sphere(density: u32) -> Mesh {
    let d = density.max(4);
    generate_sphere(1.0, d, d)
}

/// Merge `count` offset copies of `base` into a single mesh for bulk testing.
///
/// Each copy is translated by a deterministic pseudo-random offset within a
/// `±spread/2` cube (using a simple LCG so results are reproducible and no
/// `std` rng is needed).
pub fn generate_scene_mesh(base: &Mesh, count: u32, spread: f32) -> Mesh {
    if count == 0 || base.vertices.is_empty() {
        return Mesh::new("scene_mesh", Vec::new(), Vec::new());
    }

    let total_verts = base.vertices.len() * count as usize;
    let total_idx = base.indices.len() * count as usize;
    let mut vertices: Vec<Vertex> = Vec::with_capacity(total_verts);
    let mut indices: Vec<u32> = Vec::with_capacity(total_idx);

    // Simple 32-bit LCG for deterministic offsets.
    let mut state: u32 = 0x9e37_79b9;
    let mut next_f32 = move || -> f32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to [0, 1)
        (state >> 8) as f32 / (1u32 << 24) as f32
    };

    for _ in 0..count {
        let ox = (next_f32() - 0.5) * spread;
        let oy = (next_f32() - 0.5) * spread;
        let oz = (next_f32() - 0.5) * spread;
        let offset = Vec3::new(ox, oy, oz);

        let base_vertex = vertices.len() as u32;

        for v in &base.vertices {
            vertices.push(Vertex {
                pos: v.pos + offset,
                ..*v
            });
        }

        for &idx in &base.indices {
            indices.push(base_vertex + idx);
        }
    }

    Mesh::new("scene_mesh", vertices, indices)
}

// ---------------------------------------------------------------------------
// LOD decimation — greedy edge-collapse (edge-length heuristic)
// ---------------------------------------------------------------------------

/// Decimate `mesh` to approximately `target_tri_count` triangles.
///
/// Uses a greedy edge-collapse heuristic: at each iteration the shortest
/// edge is selected, the two endpoint vertices are merged at their midpoint,
/// and the index buffer is updated.  Degenerate triangles (those whose three
/// indices are not all distinct after collapsing) are removed.
///
/// This is intentionally simple — a quadric-error metric would give higher
/// quality but the complexity is out of scope for the LOD hierarchy builder
/// used here.
pub fn decimate(mesh: &Mesh, target_tri_count: usize) -> Mesh {
    // Work on cloned data.
    let mut positions: Vec<Vec3> = mesh.vertices.iter().map(|v| v.pos).collect();
    let mut indices: Vec<u32> = mesh.indices.clone();

    // `remap[i]` follows the chain of collapses for vertex i.
    let mut remap: Vec<u32> = (0..positions.len() as u32).collect();

    /// Follow the remap chain to the canonical representative.
    fn resolve(remap: &[u32], mut v: u32) -> u32 {
        while remap[v as usize] != v {
            v = remap[v as usize];
        }
        v
    }

    let mut current_tris = indices.len() / 3;

    while current_tris > target_tri_count {
        // Build a list of unique edges (avoiding duplicates via min/max ordering).
        // We cap the candidate list to keep O(n) per step acceptable.
        let mut best_len_sq = f32::MAX;
        let mut best_a = u32::MAX;
        let mut best_b = u32::MAX;

        let tri_count = indices.len() / 3;
        for t in 0..tri_count {
            let ia = resolve(&remap, indices[t * 3]);
            let ib = resolve(&remap, indices[t * 3 + 1]);
            let ic = resolve(&remap, indices[t * 3 + 2]);

            // Skip already-degenerate triangles.
            if ia == ib || ib == ic || ia == ic {
                continue;
            }

            // Check all three edges.
            for &(ea, eb) in &[(ia, ib), (ib, ic), (ia, ic)] {
                let d = positions[ea as usize] - positions[eb as usize];
                let l2 = d.x * d.x + d.y * d.y + d.z * d.z;
                if l2 < best_len_sq {
                    best_len_sq = l2;
                    best_a = ea;
                    best_b = eb;
                }
            }
        }

        if best_a == u32::MAX {
            // No collapable edges found.
            break;
        }

        // Merge best_b → best_a; move best_a to midpoint.
        let mid = Vec3::new(
            (positions[best_a as usize].x + positions[best_b as usize].x) * 0.5,
            (positions[best_a as usize].y + positions[best_b as usize].y) * 0.5,
            (positions[best_a as usize].z + positions[best_b as usize].z) * 0.5,
        );
        positions[best_a as usize] = mid;
        remap[best_b as usize] = best_a;

        // Re-resolve the index buffer and count live triangles.
        let mut new_indices: Vec<u32> = Vec::with_capacity(indices.len());
        let tris = indices.len() / 3;
        for t in 0..tris {
            let ia = resolve(&remap, indices[t * 3]);
            let ib = resolve(&remap, indices[t * 3 + 1]);
            let ic = resolve(&remap, indices[t * 3 + 2]);

            // Drop degenerate triangles.
            if ia == ib || ib == ic || ia == ic {
                continue;
            }

            new_indices.push(ia);
            new_indices.push(ib);
            new_indices.push(ic);
        }

        current_tris = new_indices.len() / 3;
        indices = new_indices;
    }

    // Build the output mesh: re-index only the referenced vertices.
    let mut used: Vec<bool> = (0..positions.len()).map(|_| false).collect();
    for &idx in &indices {
        used[idx as usize] = true;
    }

    // Build old→new mapping.
    let mut old_to_new: Vec<u32> = (0..positions.len()).map(|_| u32::MAX).collect();
    let mut new_vertices: Vec<Vertex> = Vec::new();
    for (old_idx, &is_used) in used.iter().enumerate() {
        if is_used {
            old_to_new[old_idx] = new_vertices.len() as u32;
            // Reconstruct a vertex: reuse the original vertex data but update pos.
            let orig = &mesh.vertices[old_idx.min(mesh.vertices.len() - 1)];
            new_vertices.push(Vertex {
                pos: positions[old_idx],
                ..*orig
            });
        }
    }

    let new_indices: Vec<u32> = indices.iter().map(|&i| old_to_new[i as usize]).collect();

    let mut out = Mesh::new(&mesh.name, new_vertices, new_indices);
    out.material_idx = mesh.material_idx;
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Sphere ------------------------------------------------------------

    #[test]
    fn sphere_vertex_count() {
        let stacks = 8u32;
        let slices = 16u32;
        let m = generate_sphere(1.0, stacks, slices);
        let expected_verts = (stacks + 1) * (slices + 1);
        assert_eq!(
            m.vertices.len(),
            expected_verts as usize,
            "sphere vertex count mismatch"
        );
    }

    #[test]
    fn sphere_index_count() {
        let stacks = 8u32;
        let slices = 16u32;
        let m = generate_sphere(1.0, stacks, slices);
        let expected_idx = stacks * slices * 6;
        assert_eq!(
            m.indices.len(),
            expected_idx as usize,
            "sphere index count mismatch"
        );
    }

    #[test]
    fn sphere_normals_unit_length() {
        let m = generate_sphere(1.0, 8, 16);
        for v in &m.vertices {
            let len = v.normal.len();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "sphere normal length {len} deviates from 1.0"
            );
        }
    }

    #[test]
    fn sphere_aabb_approximately_unit() {
        let m = generate_sphere(1.0, 16, 32);
        // AABB min/max should be within ±1 in every axis.
        assert!(m.aabb.min.x >= -1.01 && m.aabb.max.x <= 1.01);
        assert!(m.aabb.min.y >= -1.01 && m.aabb.max.y <= 1.01);
        assert!(m.aabb.min.z >= -1.01 && m.aabb.max.z <= 1.01);
    }

    // ---- Cube --------------------------------------------------------------

    #[test]
    fn cube_vertex_and_index_counts() {
        let m = generate_cube(1.0);
        assert_eq!(m.vertices.len(), 24, "cube must have exactly 24 vertices");
        assert_eq!(m.indices.len(), 36, "cube must have exactly 36 indices");
    }

    #[test]
    fn cube_normals_unit_length() {
        let m = generate_cube(0.5);
        for v in &m.vertices {
            let len = v.normal.len();
            assert!((len - 1.0).abs() < 1e-5, "cube normal length {len}");
        }
    }

    // ---- Grid --------------------------------------------------------------

    #[test]
    fn grid_vertex_and_index_counts() {
        let cols = 4u32;
        let rows = 6u32;
        let m = generate_grid(cols, rows, 1.0);
        assert_eq!(m.vertices.len(), ((cols + 1) * (rows + 1)) as usize);
        assert_eq!(m.indices.len(), (cols * rows * 6) as usize);
    }

    // ---- Clusters ----------------------------------------------------------

    #[test]
    fn cluster_count_correct() {
        let m_base = generate_sphere(1.0, 16, 32);
        let mut m = m_base.clone();
        m.build_clusters();

        let tri_count = m.triangle_count();
        let expected = (tri_count + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
        assert_eq!(m.cluster_count(), expected, "wrong cluster count");
    }

    #[test]
    fn cluster_indices_cover_all_triangles() {
        let mut m = generate_sphere(1.0, 8, 16);
        m.build_clusters();

        let total: u32 = m.clusters.iter().map(|c| c.count).sum();
        assert_eq!(total as usize, m.indices.len());
    }

    // ---- Decimate ----------------------------------------------------------

    #[test]
    fn decimate_reduces_triangles() {
        let m = generate_sphere(1.0, 16, 32);
        let target = 100usize;
        let d = decimate(&m, target);
        // Allow a small tolerance.
        assert!(
            d.triangle_count() <= target + 10,
            "decimate returned {} triangles, expected ≤ {}",
            d.triangle_count(),
            target + 10
        );
    }

    #[test]
    fn decimate_already_small_mesh() {
        let m = generate_cube(1.0);
        // Target larger than mesh → mesh should be unchanged in triangle count.
        let target = 1000;
        let d = decimate(&m, target);
        assert!(d.triangle_count() <= m.triangle_count());
    }

    // ---- compute_aabb ------------------------------------------------------

    #[test]
    fn compute_aabb_correct() {
        let verts = vec![
            Vertex::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(0.0, 1.0, 0.0), Vec2::new(0.0, 0.0)),
            Vertex::new(Vec3::new( 1.0,  2.0,  3.0), Vec3::new(0.0, 1.0, 0.0), Vec2::new(1.0, 1.0)),
            Vertex::new(Vec3::new( 0.0,  0.0,  0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::new(0.5, 0.5)),
        ];
        let idx = vec![0u32, 1, 2];
        let m = Mesh::new("test", verts, idx);
        assert!((m.aabb.min.x - (-1.0)).abs() < 1e-6);
        assert!((m.aabb.max.x - 1.0).abs() < 1e-6);
        assert!((m.aabb.min.y - (-2.0)).abs() < 1e-6);
        assert!((m.aabb.max.y - 2.0).abs() < 1e-6);
        assert!((m.aabb.min.z - (-3.0)).abs() < 1e-6);
        assert!((m.aabb.max.z - 3.0).abs() < 1e-6);
    }

    // ---- Material ----------------------------------------------------------

    #[test]
    fn material_default_white() {
        let m = Material::default_white();
        assert_eq!(m.base_color, [1.0, 1.0, 1.0, 1.0]);
        assert!(!m.double_sided);
    }

    #[test]
    fn material_emissive_scales_color() {
        let m = Material::emissive([0.5, 0.0, 0.0], 4.0);
        assert!((m.emissive[0] - 2.0).abs() < 1e-6);
        assert!((m.emissive[1]).abs() < 1e-6);
    }

    #[test]
    fn material_metallic_roughness_fields() {
        let m = Material::metallic_roughness([0.8, 0.6, 0.4], 0.9, 0.1);
        assert!((m.metallic - 0.9).abs() < 1e-6);
        assert!((m.roughness - 0.1).abs() < 1e-6);
        assert!((m.base_color[0] - 0.8).abs() < 1e-6);
    }

    // ---- Compute tangents --------------------------------------------------

    #[test]
    fn compute_tangents_produces_unit_tangents() {
        let mut m = generate_sphere(1.0, 8, 16);
        m.compute_tangents();
        for v in &m.vertices {
            let t = Vec3::new(v.tangent.x, v.tangent.y, v.tangent.z);
            let len = t.len();
            // Degenerate tangents at poles are expected to be zero; skip them.
            if len > 0.1 {
                assert!(
                    (len - 1.0).abs() < 1e-3,
                    "tangent length {len} is not unit"
                );
            }
        }
    }

    // ---- Stress sphere / scene mesh ----------------------------------------

    #[test]
    fn stress_sphere_density_64() {
        let m = generate_stress_sphere(64);
        // density=64 → stacks=64, slices=64 → 64*64*2 = 8192 triangles
        assert_eq!(m.triangle_count(), 64 * 64 * 2);
    }

    #[test]
    fn scene_mesh_vertex_count() {
        let base = generate_cube(1.0);
        let scene = generate_scene_mesh(&base, 5, 10.0);
        assert_eq!(scene.vertices.len(), base.vertices.len() * 5);
        assert_eq!(scene.indices.len(), base.indices.len() * 5);
    }
}

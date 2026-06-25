//! IDAG — Instruction Directed Acyclic Graph for multi-GPU federated execution.
//!
//! FEDCM: GPU scheduler monitors resource-usage and routes tasks to the optimal
//! GPU module. Virtual Buffers / Global Address Space (GAS): buffer slices across
//! hardware units appear as a single contiguous memory pool. Intra-Context
//! Spatial Multiplexing: scheduler splits large GPU resources into parallel
//! slices for different task types (UI layout, video decode, background compile).
//!
//! Pure, safe `no_std`.

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::collections::BTreeMap;

// ── SlicePriority ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlicePriority {
    InputLatency,
    UIRender,
    MediaDecode,
    BackgroundCompute,
}

// ── GpuSlice ──────────────────────────────────────────────────────────────────

/// A GPU virtual slice (subset of GPU resources: SM blocks, memory partitions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuSlice {
    pub slice_id: u32,
    /// Streaming multiprocessors in this slice.
    pub sm_count: u32,
    /// Local memory allocation in MB.
    pub memory_mb: u32,
    pub priority: SlicePriority,
}

// ── InstructionKind ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionKind {
    VectorRender,
    MeshRender,
    ComputeShader,
    MediaDecode,
    UILayout,
    CopyBuffer,
    Present,
}

// ── IDAGInstruction ───────────────────────────────────────────────────────────

/// An instruction in the IDAG (a single dispatchable GPU task).
#[derive(Clone, Debug)]
pub struct IDAGInstruction {
    pub id: u64,
    pub label: String,
    pub kind: InstructionKind,
    /// IDs of instructions that must complete before this one runs.
    pub depends_on: Vec<u64>,
    /// Which GPU slice this is assigned to.
    pub assigned_slice: Option<u32>,
    pub estimated_cycles: u64,
}

// ── IDAG ─────────────────────────────────────────────────────────────────────

/// The Instruction DAG — translated from an RDG for multi-GPU federated execution.
pub struct IDAG {
    pub instructions: Vec<IDAGInstruction>,
    pub slices: Vec<GpuSlice>,
}

impl IDAG {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            slices: Vec::new(),
        }
    }

    pub fn add_instruction(&mut self, inst: IDAGInstruction) {
        self.instructions.push(inst);
    }

    pub fn add_slice(&mut self, slice: GpuSlice) {
        self.slices.push(slice);
    }

    /// Assign instructions to slices based on priority and kind.
    ///
    /// - InputLatency slice → VectorRender + UILayout
    /// - MediaDecode slice → MediaDecode
    /// - BackgroundCompute slice → ComputeShader
    /// - UIRender slice → MeshRender, CopyBuffer, Present (fallback for rest)
    pub fn assign_to_slices(&mut self) {
        // Build index maps: priority → slice_id.
        let mut priority_to_slice: BTreeMap<u8, u32> = BTreeMap::new();
        for slice in &self.slices {
            let key = match slice.priority {
                SlicePriority::InputLatency    => 0,
                SlicePriority::UIRender        => 1,
                SlicePriority::MediaDecode     => 2,
                SlicePriority::BackgroundCompute => 3,
            };
            priority_to_slice.entry(key).or_insert(slice.slice_id);
        }

        let input_latency_id = priority_to_slice.get(&0).copied();
        let ui_render_id     = priority_to_slice.get(&1).copied();
        let media_decode_id  = priority_to_slice.get(&2).copied();
        let bg_compute_id    = priority_to_slice.get(&3).copied();

        // Default slice to use when preferred is missing.
        let default_slice = self.slices.first().map(|s| s.slice_id);

        for inst in &mut self.instructions {
            let preferred = match inst.kind {
                InstructionKind::VectorRender => input_latency_id,
                InstructionKind::UILayout     => input_latency_id,
                InstructionKind::MediaDecode  => media_decode_id,
                InstructionKind::ComputeShader => bg_compute_id,
                InstructionKind::MeshRender   => ui_render_id,
                InstructionKind::CopyBuffer   => ui_render_id,
                InstructionKind::Present      => ui_render_id,
            };
            inst.assigned_slice = preferred.or(default_slice);
        }
    }

    /// Topological sort of instructions (Kahn's algorithm).
    ///
    /// Returns IDs in topological order (dependencies before dependents).
    pub fn toposort(&self) -> Vec<u64> {
        // Build id → index map.
        let mut id_to_idx: BTreeMap<u64, usize> = BTreeMap::new();
        for (i, inst) in self.instructions.iter().enumerate() {
            id_to_idx.insert(inst.id, i);
        }

        let n = self.instructions.len();
        let mut in_degree: Vec<usize> = alloc::vec![0; n];
        // edges: idx → list of dependent idxs
        let mut edges: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];

        for (i, inst) in self.instructions.iter().enumerate() {
            for &dep_id in &inst.depends_on {
                if let Some(&dep_idx) = id_to_idx.get(&dep_id) {
                    edges[dep_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        let mut ready: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        ready.sort_unstable(); // deterministic

        let mut order: Vec<u64> = Vec::with_capacity(n);
        while !ready.is_empty() {
            ready.sort_unstable();
            let current = ready.remove(0);
            order.push(self.instructions[current].id);
            for &next in &edges[current] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    ready.push(next);
                }
            }
        }

        order
    }

    /// Simulate execution across slices in parallel.
    ///
    /// Returns `(wall_clock_cycles, slice_utilization_map)`.
    ///
    /// Wall clock is determined by the critical path. Per-slice utilization is
    /// the sum of instruction cycles on that slice divided by wall_clock_cycles.
    pub fn simulate_execution(&self) -> (u64, BTreeMap<u32, f32>) {
        // Compute finish time for each instruction respecting dependencies.
        let mut id_to_idx: BTreeMap<u64, usize> = BTreeMap::new();
        for (i, inst) in self.instructions.iter().enumerate() {
            id_to_idx.insert(inst.id, i);
        }

        let n = self.instructions.len();
        let order = self.toposort();
        let mut finish_time: Vec<u64> = alloc::vec![0u64; n];

        for &id in &order {
            if let Some(&idx) = id_to_idx.get(&id) {
                let inst = &self.instructions[idx];
                // Earliest start = max finish time of all dependencies.
                let dep_finish: u64 = inst
                    .depends_on
                    .iter()
                    .filter_map(|dep_id| id_to_idx.get(dep_id))
                    .map(|&dep_idx| finish_time[dep_idx])
                    .max()
                    .unwrap_or(0);
                finish_time[idx] = dep_finish + inst.estimated_cycles;
            }
        }

        let wall_clock = finish_time.iter().copied().max().unwrap_or(0);

        // Per-slice total cycles.
        let mut slice_cycles: BTreeMap<u32, u64> = BTreeMap::new();
        for inst in &self.instructions {
            if let Some(sid) = inst.assigned_slice {
                *slice_cycles.entry(sid).or_insert(0) += inst.estimated_cycles;
            }
        }

        // Utilization = slice_total_cycles / wall_clock (capped at 1.0).
        let mut utilization: BTreeMap<u32, f32> = BTreeMap::new();
        if wall_clock > 0 {
            for (sid, cycles) in &slice_cycles {
                let u = ((*cycles) as f32) / (wall_clock as f32);
                utilization.insert(*sid, if u > 1.0 { 1.0 } else { u });
            }
        } else {
            for sid in slice_cycles.keys() {
                utilization.insert(*sid, 0.0);
            }
        }

        (wall_clock, utilization)
    }

    /// Critical path length (longest dependency chain in cycles).
    pub fn critical_path_cycles(&self) -> u64 {
        let mut id_to_idx: BTreeMap<u64, usize> = BTreeMap::new();
        for (i, inst) in self.instructions.iter().enumerate() {
            id_to_idx.insert(inst.id, i);
        }

        let n = self.instructions.len();
        let order = self.toposort();
        let mut path_cost: Vec<u64> = alloc::vec![0u64; n];

        for &id in &order {
            if let Some(&idx) = id_to_idx.get(&id) {
                let inst = &self.instructions[idx];
                let max_dep: u64 = inst
                    .depends_on
                    .iter()
                    .filter_map(|dep_id| id_to_idx.get(dep_id))
                    .map(|&dep_idx| path_cost[dep_idx])
                    .max()
                    .unwrap_or(0);
                path_cost[idx] = max_dep + inst.estimated_cycles;
            }
        }

        path_cost.iter().copied().max().unwrap_or(0)
    }
}

impl Default for IDAG {
    fn default() -> Self { Self::new() }
}

// ── FedcmScheduler ────────────────────────────────────────────────────────────

/// FEDCM scheduler — routes tasks to optimal GPU module.
pub struct FedcmScheduler {
    pub slices: Vec<GpuSlice>,
    /// Cache residency estimate per slice index (0.0–1.0).
    pub cache_residency: Vec<f32>,
    /// Current utilization per slice index (0.0–1.0).
    pub utilization: Vec<f32>,
}

impl FedcmScheduler {
    pub fn new(slices: Vec<GpuSlice>) -> Self {
        let n = slices.len();
        Self {
            slices,
            cache_residency: alloc::vec![0.5; n],
            utilization: alloc::vec![0.0; n],
        }
    }

    /// Pick the optimal slice for a given instruction kind and priority.
    ///
    /// Strategy:
    /// 1. Find slices whose priority matches the requested priority.
    /// 2. Among those, pick the one with lowest utilization.
    /// 3. If tie, pick the one with highest cache residency.
    /// 4. If no priority match, fall back to lowest-utilization slice overall.
    ///
    /// Returns the slice_id, or None if no slices registered.
    pub fn route(&self, kind: InstructionKind, priority: SlicePriority) -> Option<u32> {
        if self.slices.is_empty() {
            return None;
        }

        // Determine preferred priority from kind (mirrors assign_to_slices logic).
        let preferred_priority = match kind {
            InstructionKind::VectorRender  => SlicePriority::InputLatency,
            InstructionKind::UILayout      => SlicePriority::InputLatency,
            InstructionKind::MediaDecode   => SlicePriority::MediaDecode,
            InstructionKind::ComputeShader => SlicePriority::BackgroundCompute,
            InstructionKind::MeshRender    => SlicePriority::UIRender,
            InstructionKind::CopyBuffer    => SlicePriority::UIRender,
            InstructionKind::Present       => SlicePriority::UIRender,
        };
        // Caller may override with explicit priority; use the tighter of the two.
        let target_priority = if priority < preferred_priority { priority } else { preferred_priority };

        // Find candidates matching target_priority.
        let candidates: Vec<usize> = self
            .slices
            .iter()
            .enumerate()
            .filter(|(_, s)| s.priority == target_priority)
            .map(|(i, _)| i)
            .collect();

        let search = if candidates.is_empty() {
            // Fallback: all slices.
            (0..self.slices.len()).collect::<Vec<_>>()
        } else {
            candidates
        };

        // Among candidates, pick lowest utilization, then highest cache residency.
        search
            .into_iter()
            .min_by(|&a, &b| {
                let ua = self.utilization[a];
                let ub = self.utilization[b];
                if (ua - ub).abs() > 1e-6 {
                    ua.partial_cmp(&ub).unwrap_or(core::cmp::Ordering::Equal)
                } else {
                    // Tie-break: higher cache residency wins (reverse order).
                    self.cache_residency[b]
                        .partial_cmp(&self.cache_residency[a])
                        .unwrap_or(core::cmp::Ordering::Equal)
                }
            })
            .map(|idx| self.slices[idx].slice_id)
    }

    /// Update utilization after dispatching a load to a slice.
    pub fn update_utilization(&mut self, slice_id: u32, added_load: f32) {
        if let Some(idx) = self.slices.iter().position(|s| s.slice_id == slice_id) {
            let u = self.utilization[idx] + added_load;
            self.utilization[idx] = if u > 1.0 { 1.0 } else { u };
        }
    }

    /// Build a default 3-slice config:
    /// - Slice 0: InputLatency (UI layout, vector render)
    /// - Slice 1: UIRender + MediaDecode (mesh rendering, video decode)
    /// - Slice 2: BackgroundCompute (shader compilation, ML)
    pub fn default_3slice() -> Self {
        let slices = alloc::vec![
            GpuSlice { slice_id: 0, sm_count: 8,  memory_mb: 512,  priority: SlicePriority::InputLatency },
            GpuSlice { slice_id: 1, sm_count: 32, memory_mb: 2048, priority: SlicePriority::UIRender },
            GpuSlice { slice_id: 2, sm_count: 16, memory_mb: 1024, priority: SlicePriority::BackgroundCompute },
        ];
        Self::new(slices)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(id: u64, kind: InstructionKind, deps: &[u64], cycles: u64) -> IDAGInstruction {
        IDAGInstruction {
            id,
            label: alloc::format!("inst_{}", id),
            kind,
            depends_on: deps.to_vec(),
            assigned_slice: None,
            estimated_cycles: cycles,
        }
    }

    // 1. Empty IDAG toposort returns empty.
    #[test]
    fn test_toposort_empty() {
        let dag = IDAG::new();
        assert!(dag.toposort().is_empty());
    }

    // 2. Single instruction toposort.
    #[test]
    fn test_toposort_single() {
        let mut dag = IDAG::new();
        dag.add_instruction(inst(1, InstructionKind::UILayout, &[], 100));
        let order = dag.toposort();
        assert_eq!(order, alloc::vec![1]);
    }

    // 3. Linear chain toposort: 1 → 2 → 3.
    #[test]
    fn test_toposort_linear_chain() {
        let mut dag = IDAG::new();
        dag.add_instruction(inst(1, InstructionKind::UILayout,    &[],    100));
        dag.add_instruction(inst(2, InstructionKind::VectorRender, &[1], 200));
        dag.add_instruction(inst(3, InstructionKind::Present,      &[2], 50));
        let order = dag.toposort();
        assert_eq!(order.len(), 3);
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(1) < pos(2));
        assert!(pos(2) < pos(3));
    }

    // 4. Diamond DAG toposort.
    #[test]
    fn test_toposort_diamond() {
        let mut dag = IDAG::new();
        dag.add_instruction(inst(1, InstructionKind::UILayout,     &[],    100));
        dag.add_instruction(inst(2, InstructionKind::MeshRender,   &[1], 150));
        dag.add_instruction(inst(3, InstructionKind::ComputeShader,&[1], 200));
        dag.add_instruction(inst(4, InstructionKind::Present,      &[2, 3], 50));
        let order = dag.toposort();
        assert_eq!(order.len(), 4);
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(1) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(4));
        assert!(pos(3) < pos(4));
    }

    // 5. assign_to_slices routes VectorRender to InputLatency slice.
    #[test]
    fn test_assign_vector_render_to_input_latency() {
        let mut dag = IDAG::new();
        dag.add_slice(GpuSlice { slice_id: 0, sm_count: 4, memory_mb: 256, priority: SlicePriority::InputLatency });
        dag.add_slice(GpuSlice { slice_id: 1, sm_count: 16, memory_mb: 1024, priority: SlicePriority::UIRender });
        dag.add_instruction(inst(1, InstructionKind::VectorRender, &[], 100));
        dag.assign_to_slices();
        assert_eq!(dag.instructions[0].assigned_slice, Some(0));
    }

    // 6. assign_to_slices routes MediaDecode to MediaDecode slice.
    #[test]
    fn test_assign_media_decode_to_media_slice() {
        let mut dag = IDAG::new();
        dag.add_slice(GpuSlice { slice_id: 0, sm_count: 4, memory_mb: 256, priority: SlicePriority::InputLatency });
        dag.add_slice(GpuSlice { slice_id: 2, sm_count: 8, memory_mb: 512, priority: SlicePriority::MediaDecode });
        dag.add_instruction(inst(1, InstructionKind::MediaDecode, &[], 300));
        dag.assign_to_slices();
        assert_eq!(dag.instructions[0].assigned_slice, Some(2));
    }

    // 7. assign_to_slices routes ComputeShader to BackgroundCompute slice.
    #[test]
    fn test_assign_compute_to_bg_slice() {
        let mut dag = IDAG::new();
        dag.add_slice(GpuSlice { slice_id: 3, sm_count: 16, memory_mb: 1024, priority: SlicePriority::BackgroundCompute });
        dag.add_instruction(inst(1, InstructionKind::ComputeShader, &[], 500));
        dag.assign_to_slices();
        assert_eq!(dag.instructions[0].assigned_slice, Some(3));
    }

    // 8. critical_path_cycles for a linear chain.
    #[test]
    fn test_critical_path_linear() {
        let mut dag = IDAG::new();
        dag.add_instruction(inst(1, InstructionKind::UILayout,    &[],  100));
        dag.add_instruction(inst(2, InstructionKind::MeshRender,  &[1], 200));
        dag.add_instruction(inst(3, InstructionKind::Present,     &[2], 50));
        // Critical path = 100 + 200 + 50 = 350.
        assert_eq!(dag.critical_path_cycles(), 350);
    }

    // 9. critical_path_cycles for a diamond — takes the longer branch.
    #[test]
    fn test_critical_path_diamond() {
        let mut dag = IDAG::new();
        dag.add_instruction(inst(1, InstructionKind::UILayout,      &[],    100));
        dag.add_instruction(inst(2, InstructionKind::MeshRender,    &[1], 150));
        dag.add_instruction(inst(3, InstructionKind::ComputeShader, &[1], 300));
        dag.add_instruction(inst(4, InstructionKind::Present,       &[2, 3], 50));
        // Branch A: 100+150+50=300; Branch B: 100+300+50=450 — critical = 450.
        assert_eq!(dag.critical_path_cycles(), 450);
    }

    // 10. simulate_execution wall clock equals critical path for independent insts.
    #[test]
    fn test_simulate_execution_independent() {
        let mut dag = IDAG::new();
        dag.add_slice(GpuSlice { slice_id: 0, sm_count: 8, memory_mb: 512, priority: SlicePriority::InputLatency });
        dag.add_instruction(inst(1, InstructionKind::UILayout,    &[], 100));
        dag.add_instruction(inst(2, InstructionKind::VectorRender,&[], 200));
        dag.assign_to_slices();
        let (wall, util) = dag.simulate_execution();
        // Wall clock = max(100, 200) = 200.
        assert_eq!(wall, 200);
        // Utilization for slice 0 = (100 + 200) / 200 = 1.5 → capped to 1.0.
        assert!(util.contains_key(&0));
        assert!(util[&0] <= 1.0);
    }

    // 11. simulate_execution returns plausible values for a chain.
    #[test]
    fn test_simulate_execution_chain() {
        let mut dag = IDAG::new();
        dag.add_slice(GpuSlice { slice_id: 0, sm_count: 8, memory_mb: 512, priority: SlicePriority::InputLatency });
        dag.add_slice(GpuSlice { slice_id: 1, sm_count: 16, memory_mb: 1024, priority: SlicePriority::UIRender });
        dag.add_instruction(inst(1, InstructionKind::UILayout,  &[],  100));
        dag.add_instruction(inst(2, InstructionKind::MeshRender,&[1], 200));
        dag.assign_to_slices();
        let (wall, _util) = dag.simulate_execution();
        assert_eq!(wall, 300); // serial: 100 + 200
    }

    // 12. FedcmScheduler default_3slice has exactly 3 slices.
    #[test]
    fn test_default_3slice_count() {
        let sched = FedcmScheduler::default_3slice();
        assert_eq!(sched.slices.len(), 3);
    }

    // 13. FedcmScheduler routes VectorRender to InputLatency slice.
    #[test]
    fn test_fedcm_route_vector_render() {
        let sched = FedcmScheduler::default_3slice();
        let sid = sched.route(InstructionKind::VectorRender, SlicePriority::InputLatency);
        assert_eq!(sid, Some(0)); // slice 0 is InputLatency
    }

    // 14. FedcmScheduler routes ComputeShader to BackgroundCompute slice.
    #[test]
    fn test_fedcm_route_compute() {
        let sched = FedcmScheduler::default_3slice();
        let sid = sched.route(InstructionKind::ComputeShader, SlicePriority::BackgroundCompute);
        assert_eq!(sid, Some(2)); // slice 2 is BackgroundCompute
    }

    // 15. update_utilization caps at 1.0.
    #[test]
    fn test_update_utilization_capped() {
        let mut sched = FedcmScheduler::default_3slice();
        sched.update_utilization(0, 0.8);
        sched.update_utilization(0, 0.5); // would exceed 1.0
        assert!(sched.utilization[0] <= 1.0);
    }

    // 18. UILayout routes to InputLatency.
    #[test]
    fn test_assign_ui_layout_to_input_latency() {
        let mut dag = IDAG::new();
        dag.add_slice(GpuSlice { slice_id: 0, sm_count: 4, memory_mb: 256, priority: SlicePriority::InputLatency });
        dag.add_instruction(inst(1, InstructionKind::UILayout, &[], 80));
        dag.assign_to_slices();
        assert_eq!(dag.instructions[0].assigned_slice, Some(0));
    }
}

//! Global Render Dependency Graph (RDG) — the 3-phase compile pipeline.
//! (see `docs/2d-3d rendering redesign.md` §"The Global Render Dependency Graph Compiler")
//!
//! Phase 1: Declare resources + passes (no allocation).
//! Phase 2: Cull dead passes, alias transient memory, generate split-barriers.
//! Phase 3: Produce `CompiledRdg` ready for GPU command dispatch.
//!
//! Pure, safe `no_std`.

extern crate alloc;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

// ─── Resource types ──────────────────────────────────────────────────────────

type ResourceId = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    /// RGBA8 colour render target.
    ColorTarget { w: u32, h: u32 },
    /// f32 depth render target.
    DepthTarget { w: u32, h: u32 },
    /// Generic GPU buffer (e.g. vertex / uniform / storage).
    Buffer { bytes: u64 },
    /// The final display surface — never aliased.
    SwapchainTarget,
}

#[derive(Clone, Debug)]
struct RdgResource {
    pub id: ResourceId,
    pub kind: ResourceKind,
    /// `true` = lifetime is bounded within a frame; can alias with other transients.
    pub transient: bool,
    pub name: &'static str,
}

// ─── Pass types ──────────────────────────────────────────────────────────────

type PassIdx = usize;

#[derive(Clone, Debug)]
pub struct RdgPass {
    pub name: &'static str,
    /// Resources this pass reads from.
    pub reads: Vec<ResourceId>,
    /// Resources this pass writes to.
    pub writes: Vec<ResourceId>,
    pub kind: PassKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// Rasterisation pass.
    Graphics,
    /// Compute-shader pass.
    Compute,
    /// Copy / blit pass.
    Transfer,
    /// Final scanout — this is the root of the liveness tree.
    Present,
}

// ─── Barrier types ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
enum BarrierKind {
    ReadToWrite,
    WriteToRead,
    WriteToPresent,
}

/// A split-barrier inserted between two passes.
///
/// `before_pass` is the index into `CompiledRdg::execution_order`; the barrier
/// must be signalled before that pass begins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Barrier {
    /// Index *into `execution_order`* of the consuming pass.
    pub before_pass: usize,
    pub resource: ResourceId,
    pub transition: BarrierKind,
}

// ─── Memory aliasing ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
struct MemoryAlias {
    pub resource: ResourceId,
    /// The resource whose physical memory backing is reused.
    pub aliases_with: ResourceId,
    /// Byte offset inside the shared heap slab.
    pub offset: u64,
}

// ─── Bindless resource table ──────────────────────────────────────────────────

/// A bindless resource table: resource handles indexed directly by u32.
/// In a real GPU, this maps to a bindless descriptor heap (DX12) or
/// VK_EXT_descriptor_indexing arrays. No descriptor sets required.
#[derive(Clone, Debug)]
struct BindlessResourceTable {
    pub entries: Vec<(u64, ResourceKind)>,  // (resource_id, kind)
}

impl BindlessResourceTable {
    pub fn new() -> Self {
        BindlessResourceTable { entries: Vec::new() }
    }

    /// Register a resource. Returns its index in the table.
    pub fn register(&mut self, resource_id: u64, kind: ResourceKind) -> u32 {
        let idx = self.entries.len() as u32;
        self.entries.push((resource_id, kind));
        idx
    }

    /// Look up an entry by index.
    pub fn get(&self, index: u32) -> Option<&(u64, ResourceKind)> {
        self.entries.get(index as usize)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for BindlessResourceTable {
    fn default() -> Self { Self::new() }
}

// ─── Async compute queue ─────────────────────────────────────────────────────

/// Async compute queue: passes that run independently of graphics.
/// In a real GPU, these submit to the async compute queue rather than the
/// main 3D/graphics queue, enabling overlap with rasterization.
#[derive(Clone, Debug)]
struct AsyncComputeQueue {
    /// Indices into execution_order for passes placed on the async compute queue.
    pub passes: Vec<usize>,
}

impl AsyncComputeQueue {
    pub fn new() -> Self {
        AsyncComputeQueue { passes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }
}

impl Default for AsyncComputeQueue {
    fn default() -> Self { Self::new() }
}

// ─── Compiled RDG ────────────────────────────────────────────────────────────

/// The result of `Rdg::compile()`.
pub struct CompiledRdg {
    /// Topologically-sorted pass indices (execution order).
    pub execution_order: Vec<PassIdx>,
    /// Passes that are not reachable from any `Present` pass.
    pub culled: Vec<PassIdx>,
    /// Memory aliasing assignments for transient resources.
    aliases: Vec<MemoryAlias>,
    /// Split-barrier list in execution order.
    pub barriers: Vec<Barrier>,
    /// Total transient heap size needed (bytes), after aliasing.
    pub heap_bytes: u64,
    async_compute: AsyncComputeQueue,
    bindless_table: BindlessResourceTable,
}

// ─── RDG builder ─────────────────────────────────────────────────────────────

pub struct Rdg {
    resources: Vec<RdgResource>,
    passes: Vec<RdgPass>,
    next_id: ResourceId,
}

impl Rdg {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            next_id: 0,
        }
    }

    /// Register a resource; returns its stable `ResourceId`.
    pub fn add_resource(
        &mut self,
        kind: ResourceKind,
        transient: bool,
        name: &'static str,
    ) -> ResourceId {
        let id = self.next_id;
        self.next_id += 1;
        self.resources.push(RdgResource { id, kind, transient, name });
        id
    }

    /// Register a render pass. The order of calls defines declaration order.
    pub fn add_pass(&mut self, pass: RdgPass) {
        self.passes.push(pass);
    }

    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    // ─── Compiler ────────────────────────────────────────────────────────────

    /// Byte footprint of a single resource (for heap sizing and aliasing).
    fn resource_bytes(kind: &ResourceKind) -> u64 {
        match kind {
            ResourceKind::ColorTarget { w, h } => (*w as u64) * (*h as u64) * 4,
            ResourceKind::DepthTarget { w, h } => (*w as u64) * (*h as u64) * 4,
            ResourceKind::Buffer { bytes } => *bytes,
            ResourceKind::SwapchainTarget => 0, // managed by the OS / display driver
        }
    }

    /// Compile the RDG through all three phases and return a `CompiledRdg`.
    ///
    /// # Phase 1 — Liveness / reachability
    /// Walk backwards from every `Present` pass, following `read → write` edges.
    /// Any pass not reached is marked as culled.
    ///
    /// # Phase 2 — Topological sort
    /// Kahn's algorithm over the live pass set using `write → read` edges.
    ///
    /// # Phase 3 — Resource lifetimes, aliasing, barriers
    /// Compute `[first_write, last_read]` intervals in execution order, then
    /// greedily pack transient resources into heap slots.
    pub fn compile(&self) -> CompiledRdg {
        let n = self.passes.len();

        // ── Phase 1: backward reachability from Present passes ───────────────

        // Build a map: resource_id → set of passes that *write* it.
        let mut writers: BTreeMap<ResourceId, Vec<PassIdx>> = BTreeMap::new();
        for (idx, pass) in self.passes.iter().enumerate() {
            for &rid in &pass.writes {
                writers.entry(rid).or_default().push(idx);
            }
        }

        // BFS/DFS backwards: from a pass, all resources it *reads* must have been
        // written by some live pass → mark that writer live, recurse.
        let mut live = alloc::vec![false; n];
        let mut stack: Vec<PassIdx> = Vec::new();

        // Seed with all Present passes.
        for (idx, pass) in self.passes.iter().enumerate() {
            if pass.kind == PassKind::Present {
                if !live[idx] {
                    live[idx] = true;
                    stack.push(idx);
                }
            }
        }

        while let Some(idx) = stack.pop() {
            for &rid in &self.passes[idx].reads {
                if let Some(wlist) = writers.get(&rid) {
                    for &widx in wlist {
                        if !live[widx] {
                            live[widx] = true;
                            stack.push(widx);
                        }
                    }
                }
            }
        }

        let culled: Vec<PassIdx> = (0..n).filter(|&i| !live[i]).collect();
        let live_passes: Vec<PassIdx> = (0..n).filter(|&i| live[i]).collect();

        // ── Phase 2: topological sort (Kahn's) over live passes ──────────────

        // Build a map: resource_id → set of live passes that *read* it.
        let mut readers: BTreeMap<ResourceId, Vec<PassIdx>> = BTreeMap::new();
        for &idx in &live_passes {
            for &rid in &self.passes[idx].reads {
                readers.entry(rid).or_default().push(idx);
            }
        }

        // For each live pass compute in-degree = number of live predecessor passes
        // (i.e. live writers of any resource this pass reads).
        let mut in_degree: BTreeMap<PassIdx, usize> = BTreeMap::new();
        for &idx in &live_passes {
            in_degree.insert(idx, 0);
        }
        // Edge: writer → reader (write → read dependency).
        // For each live pass's writes, find readers and add edges.
        let mut edges: BTreeMap<PassIdx, Vec<PassIdx>> = BTreeMap::new(); // src → dsts
        for &widx in &live_passes {
            for &rid in &self.passes[widx].writes {
                if let Some(rlist) = readers.get(&rid) {
                    for &ridx in rlist {
                        if ridx != widx && live[ridx] {
                            edges.entry(widx).or_default().push(ridx);
                            *in_degree.entry(ridx).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Kahn's queue — deterministic: a min-heap always pops the smallest ready
        // index in O(log V), avoiding the O(V) shift of remove(0) and per-iteration
        // re-sorts.
        let mut ready: BinaryHeap<Reverse<PassIdx>> = live_passes
            .iter()
            .copied()
            .filter(|idx| in_degree[idx] == 0)
            .map(Reverse)
            .collect();

        let mut execution_order: Vec<PassIdx> = Vec::with_capacity(live_passes.len());
        while let Some(Reverse(current)) = ready.pop() {
            execution_order.push(current);

            if let Some(dsts) = edges.get(&current) {
                for &dst in dsts {
                    let deg = in_degree.entry(dst).or_insert(1);
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(Reverse(dst));
                    }
                }
            }
        }

        // ── Phase 3: resource lifetimes ──────────────────────────────────────
        //
        // "Time" here is the index into `execution_order`.

        // Map pass index → position in execution_order.
        let mut exec_pos: BTreeMap<PassIdx, usize> = BTreeMap::new();
        for (pos, &pidx) in execution_order.iter().enumerate() {
            exec_pos.insert(pidx, pos);
        }

        // For each resource: first_write_time, last_read_time, last_write_time
        // (in exec positions). last_write is tracked so a write occurring after the
        // last read still extends the resource's live interval — otherwise its slot
        // could be aliased away while a later write is still pending.
        struct Lifetime {
            first_write: usize,
            last_read: usize,
            last_write: usize,
        }

        let mut lifetimes: BTreeMap<ResourceId, Lifetime> = BTreeMap::new();

        for (pos, &pidx) in execution_order.iter().enumerate() {
            let pass = &self.passes[pidx];
            for &rid in &pass.writes {
                let lt = lifetimes.entry(rid).or_insert(Lifetime {
                    first_write: pos,
                    last_read: pos,
                    last_write: pos,
                });
                if pos < lt.first_write {
                    lt.first_write = pos;
                }
                if pos > lt.last_write {
                    lt.last_write = pos;
                }
            }
            for &rid in &pass.reads {
                let lt = lifetimes.entry(rid).or_insert(Lifetime {
                    first_write: pos,
                    last_read: pos,
                    last_write: pos,
                });
                if pos > lt.last_read {
                    lt.last_read = pos;
                }
            }
        }

        // ── Phase 3b: greedy memory aliasing ─────────────────────────────────
        //
        // Resources are scanned in declaration order (stable ResourceId order).
        // We maintain a list of "slots": (current_end_time, base_offset, slot_size).
        // A transient resource can reuse a slot if slot_end_time < resource.first_write.

        struct Slot {
            end_time: usize, // last-touch (max of last_read/last_write) of the resource assigned
            base_offset: u64,
            size: u64,
        }

        let mut slots: Vec<Slot> = Vec::new();
        let mut aliases: Vec<MemoryAlias> = Vec::new();
        // resource_id → (base_offset, canonical_resource_id)
        let mut resource_placement: BTreeMap<ResourceId, (u64, ResourceId)> = BTreeMap::new();
        let mut heap_bytes: u64 = 0;

        // Collect transient resources sorted by declaration order (ResourceId).
        let mut transient_ids: Vec<ResourceId> = self
            .resources
            .iter()
            .filter(|r| r.transient && r.kind != ResourceKind::SwapchainTarget)
            .map(|r| r.id)
            .collect();
        transient_ids.sort_unstable();

        for &rid in &transient_ids {
            let res = match self.resources.iter().find(|r| r.id == rid) {
                Some(r) => r,
                None => continue,
            };
            let lt = match lifetimes.get(&rid) {
                Some(lt) => lt,
                // Resource declared but never used — still reserve space for safety.
                None => {
                    let sz = Self::resource_bytes(&res.kind);
                    heap_bytes += sz;
                    continue;
                }
            };
            let size = Self::resource_bytes(&res.kind);

            // Find a slot whose current occupant's lifetime has ended before our first write.
            let mut found: Option<usize> = None;
            for (si, slot) in slots.iter().enumerate() {
                if slot.end_time < lt.first_write && slot.size >= size {
                    found = Some(si);
                    break;
                }
            }

            if let Some(si) = found {
                // Reuse the slot.
                let base = slots[si].base_offset;
                // Find the canonical (original) resource that opened this slot.
                let canon = resource_placement
                    .iter()
                    .find(|(_, &(off, _))| off == base)
                    .map(|(&id, _)| id)
                    .unwrap_or(rid);
                aliases.push(MemoryAlias {
                    resource: rid,
                    aliases_with: canon,
                    offset: base,
                });
                resource_placement.insert(rid, (base, canon));
                // Update slot's end time (true last touch — read or write).
                slots[si].end_time = lt.last_read.max(lt.last_write);
            } else {
                // Open a new slot.
                let base = heap_bytes;
                heap_bytes += size;
                slots.push(Slot { end_time: lt.last_read.max(lt.last_write), base_offset: base, size });
                resource_placement.insert(rid, (base, rid));
            }
        }

        // ── Phase 3c: barriers ────────────────────────────────────────────────
        //
        // For every resource, find consecutive (producer_pos, consumer_pos) pairs
        // in execution order. Insert the appropriate barrier before the consumer.

        // Collect all write events per resource: (exec_pos, pass_kind).
        let mut write_events: BTreeMap<ResourceId, Vec<(usize, PassKind)>> = BTreeMap::new();
        let mut read_events: BTreeMap<ResourceId, Vec<usize>> = BTreeMap::new();

        for (pos, &pidx) in execution_order.iter().enumerate() {
            let pass = &self.passes[pidx];
            for &rid in &pass.writes {
                write_events
                    .entry(rid)
                    .or_default()
                    .push((pos, pass.kind.clone()));
            }
            for &rid in &pass.reads {
                read_events.entry(rid).or_default().push(pos);
            }
        }

        let mut barriers: Vec<Barrier> = Vec::new();

        // For each write event, find all reads that immediately follow it
        // (i.e. the smallest read_pos > write_pos).
        for (&rid, wevts) in &write_events {
            let reads = read_events.get(&rid);
            for &(wpos, ref wkind) in wevts {
                // Determine whether the next consumer is a Present pass.
                // A Present pass must be the very last pass consuming a swapchain
                // resource, but we may also have an explicit WriteToPresent transition.
                //
                // Insert WriteToRead between any write and a subsequent read.
                // Insert WriteToPresent when the consuming pass is a Present pass.
                if let Some(rlist) = reads {
                    for &rpos in rlist {
                        if rpos > wpos {
                            // Find the consuming pass.
                            let consumer_pidx = execution_order[rpos];
                            let consumer_kind = &self.passes[consumer_pidx].kind;
                            let transition = if *consumer_kind == PassKind::Present {
                                BarrierKind::WriteToPresent
                            } else {
                                match wkind {
                                    _ => BarrierKind::WriteToRead,
                                }
                            };
                            barriers.push(Barrier {
                                before_pass: rpos,
                                resource: rid,
                                transition,
                            });
                        }
                    }
                }
            }
        }

        // Also handle read→write (e.g. a resource read by pass A then overwritten by pass B).
        for (&rid, rlist) in &read_events {
            if let Some(wevts) = write_events.get(&rid) {
                for &rpos in rlist {
                    for &(wpos, _) in wevts {
                        if wpos > rpos {
                            barriers.push(Barrier {
                                before_pass: wpos,
                                resource: rid,
                                transition: BarrierKind::ReadToWrite,
                            });
                        }
                    }
                }
            }
        }

        // Sort barriers by `before_pass` for stable execution ordering.
        barriers.sort_unstable_by_key(|b| (b.before_pass, b.resource));
        // Deduplicate (same pass + resource + transition).
        barriers.dedup_by(|a, b| {
            a.before_pass == b.before_pass
                && a.resource == b.resource
                && a.transition == b.transition
        });

        // ── Async compute: collect Compute passes with no reads ───────────────
        let mut async_compute = AsyncComputeQueue::new();
        for (exec_pos, &pidx) in execution_order.iter().enumerate() {
            let pass = &self.passes[pidx];
            if pass.kind == PassKind::Compute && pass.reads.is_empty() {
                async_compute.passes.push(exec_pos);
            }
        }

        // ── Bindless table: register all non-SwapchainTarget resources ────────
        let mut bindless_table = BindlessResourceTable::new();
        for res in &self.resources {
            if res.kind != ResourceKind::SwapchainTarget {
                bindless_table.register(res.id as u64, res.kind.clone());
            }
        }

        CompiledRdg {
            execution_order,
            culled,
            aliases,
            barriers,
            heap_bytes,
            async_compute,
            bindless_table,
        }
    }
}

impl CompiledRdg {
    /// Simulate the execution phase: returns (graphics_pass_count, async_compute_count).
    /// graphics_pass_count = passes in execution_order NOT in async_compute.
    /// async_compute_count = passes in async_compute.
    pub fn execute_stats(&self) -> (usize, usize) {
        let async_count = self.async_compute.len();
        let graphics_count = self.execution_order.len().saturating_sub(async_count);
        (graphics_count, async_count)
    }
}

impl Default for Rdg {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn color(w: u32, h: u32) -> ResourceKind {
        ResourceKind::ColorTarget { w, h }
    }
    fn depth(w: u32, h: u32) -> ResourceKind {
        ResourceKind::DepthTarget { w, h }
    }
    fn buf(bytes: u64) -> ResourceKind {
        ResourceKind::Buffer { bytes }
    }

    // ── 1. Simple linear chain A → B → C → Present ───────────────────────────

    #[test]
    fn test_linear_chain() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(color(1920, 1080), true, "gbuffer");
        let r1 = rdg.add_resource(color(1920, 1080), true, "lighting");
        let r2 = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swapchain");

        // Pass A: writes r0
        rdg.add_pass(RdgPass {
            name: "GBuffer",
            reads: Vec::new(),
            writes: alloc::vec![r0],
            kind: PassKind::Graphics,
        });
        // Pass B: reads r0, writes r1
        rdg.add_pass(RdgPass {
            name: "Lighting",
            reads: alloc::vec![r0],
            writes: alloc::vec![r1],
            kind: PassKind::Compute,
        });
        // Pass C: reads r1, writes r2
        rdg.add_pass(RdgPass {
            name: "Composite",
            reads: alloc::vec![r1],
            writes: alloc::vec![r2],
            kind: PassKind::Graphics,
        });
        // Present: reads r2
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![r2],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        assert_eq!(rdg.pass_count(), 4);
        assert_eq!(rdg.resource_count(), 3);

        let compiled = rdg.compile();

        assert!(compiled.culled.is_empty(), "no passes should be culled");
        assert_eq!(compiled.execution_order.len(), 4, "all 4 passes in order");

        // Verify that the execution order respects GBuffer < Lighting < Composite < Present.
        let pos_of = |name: &str| -> usize {
            compiled
                .execution_order
                .iter()
                .position(|&i| rdg.passes[i].name == name)
                .unwrap()
        };
        assert!(pos_of("GBuffer") < pos_of("Lighting"));
        assert!(pos_of("Lighting") < pos_of("Composite"));
        assert!(pos_of("Composite") < pos_of("Present"));
    }

    // ── 2. Diamond dependency ─────────────────────────────────────────────────

    #[test]
    fn test_diamond_dependency() {
        let mut rdg = Rdg::new();

        let base = rdg.add_resource(color(800, 600), true, "base");
        let branch_a = rdg.add_resource(color(800, 600), true, "branchA");
        let branch_b = rdg.add_resource(color(800, 600), true, "branchB");
        let merged = rdg.add_resource(color(800, 600), true, "merged");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // Pass 0: produces base
        rdg.add_pass(RdgPass {
            name: "Base",
            reads: Vec::new(),
            writes: alloc::vec![base],
            kind: PassKind::Graphics,
        });
        // Pass 1: reads base, produces branch_a
        rdg.add_pass(RdgPass {
            name: "BranchA",
            reads: alloc::vec![base],
            writes: alloc::vec![branch_a],
            kind: PassKind::Compute,
        });
        // Pass 2: reads base, produces branch_b
        rdg.add_pass(RdgPass {
            name: "BranchB",
            reads: alloc::vec![base],
            writes: alloc::vec![branch_b],
            kind: PassKind::Compute,
        });
        // Pass 3: reads both branches, produces merged
        rdg.add_pass(RdgPass {
            name: "Merge",
            reads: alloc::vec![branch_a, branch_b],
            writes: alloc::vec![merged],
            kind: PassKind::Graphics,
        });
        // Pass 4: reads merged, writes swap
        rdg.add_pass(RdgPass {
            name: "Composite",
            reads: alloc::vec![merged],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        // Pass 5: Present
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        assert!(compiled.culled.is_empty(), "diamond: no culled passes");
        assert_eq!(compiled.execution_order.len(), 6);

        let pos_of = |name: &str| -> usize {
            compiled
                .execution_order
                .iter()
                .position(|&i| rdg.passes[i].name == name)
                .unwrap()
        };

        assert!(pos_of("Base") < pos_of("BranchA"));
        assert!(pos_of("Base") < pos_of("BranchB"));
        assert!(pos_of("BranchA") < pos_of("Merge"));
        assert!(pos_of("BranchB") < pos_of("Merge"));
        assert!(pos_of("Merge") < pos_of("Composite"));
        assert!(pos_of("Composite") < pos_of("Present"));
    }

    // ── 3. Dead pass (not connected to Present) ───────────────────────────────

    #[test]
    fn test_dead_pass_culled() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(color(640, 480), true, "main");
        let dead_res = rdg.add_resource(color(64, 64), true, "dead_tex");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // Live pass: writes r0
        rdg.add_pass(RdgPass {
            name: "Main",
            reads: Vec::new(),
            writes: alloc::vec![r0],
            kind: PassKind::Graphics,
        });
        // Dead pass: writes dead_res — nothing reads it
        rdg.add_pass(RdgPass {
            name: "Dead",
            reads: Vec::new(),
            writes: alloc::vec![dead_res],
            kind: PassKind::Compute,
        });
        // Present reads r0 via an intermediate.
        rdg.add_pass(RdgPass {
            name: "ToSwap",
            reads: alloc::vec![r0],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        assert_eq!(compiled.culled.len(), 1, "exactly one pass is dead");
        let dead_idx = rdg.passes.iter().position(|p| p.name == "Dead").unwrap();
        assert!(compiled.culled.contains(&dead_idx));
        assert_eq!(compiled.execution_order.len(), 3, "3 live passes");
    }

    // ── 4. Memory aliasing ────────────────────────────────────────────────────

    #[test]
    fn test_memory_aliasing() {
        let mut rdg = Rdg::new();

        // Two transient colour targets: A is used in passes 0–1, B in passes 2–3.
        // They have disjoint lifetimes → should alias.
        let res_a = rdg.add_resource(color(1920, 1080), true, "transA");
        let res_b = rdg.add_resource(color(1920, 1080), true, "transB");
        let out = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        let a_bytes = Rdg::resource_bytes(&color(1920, 1080));
        let b_bytes = Rdg::resource_bytes(&color(1920, 1080));
        assert_eq!(a_bytes, b_bytes);

        // Pass 0: writes res_a
        rdg.add_pass(RdgPass {
            name: "WriteA",
            reads: Vec::new(),
            writes: alloc::vec![res_a],
            kind: PassKind::Graphics,
        });
        // Pass 1: reads res_a, writes something we can use to feed B's pass.
        let tmp = rdg.add_resource(buf(256), true, "tmp");
        rdg.add_pass(RdgPass {
            name: "ReadA",
            reads: alloc::vec![res_a],
            writes: alloc::vec![tmp],
            kind: PassKind::Compute,
        });
        // Pass 2: reads tmp, writes res_b (res_a lifetime is now over)
        rdg.add_pass(RdgPass {
            name: "WriteB",
            reads: alloc::vec![tmp],
            writes: alloc::vec![res_b],
            kind: PassKind::Graphics,
        });
        // Pass 3: reads res_b, writes out
        rdg.add_pass(RdgPass {
            name: "ReadB",
            reads: alloc::vec![res_b],
            writes: alloc::vec![out],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![out],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        assert!(compiled.culled.is_empty());

        // At least one alias should exist (res_b aliases res_a).
        let has_alias = compiled
            .aliases
            .iter()
            .any(|a| (a.resource == res_b && a.aliases_with == res_a)
                || (a.resource == res_a && a.aliases_with == res_b));
        assert!(has_alias, "res_a and res_b should alias; aliases: {:?}", compiled.aliases);

        // Total heap < naive sum of all transient sizes.
        let naive_sum = a_bytes + b_bytes + 256u64; // res_a + res_b + tmp
        assert!(
            compiled.heap_bytes < naive_sum,
            "heap_bytes ({}) should be < naive_sum ({})",
            compiled.heap_bytes,
            naive_sum
        );
    }

    // ── 5. Barrier generation ─────────────────────────────────────────────────

    #[test]
    fn test_barrier_write_to_read() {
        let mut rdg = Rdg::new();

        let res = rdg.add_resource(color(512, 512), true, "tex");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // Pass 0 writes res.
        rdg.add_pass(RdgPass {
            name: "Writer",
            reads: Vec::new(),
            writes: alloc::vec![res],
            kind: PassKind::Graphics,
        });
        // Pass 1 reads res, writes swap.
        rdg.add_pass(RdgPass {
            name: "Reader",
            reads: alloc::vec![res],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        // Present reads swap.
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        // Expect a WriteToRead barrier on `res` before the Reader pass.
        let reader_exec_pos = compiled
            .execution_order
            .iter()
            .position(|&i| rdg.passes[i].name == "Reader")
            .unwrap();

        let has_w2r = compiled.barriers.iter().any(|b| {
            b.resource == res
                && b.before_pass == reader_exec_pos
                && b.transition == BarrierKind::WriteToRead
        });
        assert!(has_w2r, "WriteToRead barrier missing for res before Reader; barriers: {:?}", compiled.barriers);
    }

    #[test]
    fn test_barrier_write_to_present() {
        let mut rdg = Rdg::new();

        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // A single pass writes the swapchain.
        rdg.add_pass(RdgPass {
            name: "Render",
            reads: Vec::new(),
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        // Present reads it.
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        let present_exec_pos = compiled
            .execution_order
            .iter()
            .position(|&i| rdg.passes[i].name == "Present")
            .unwrap();

        let has_w2p = compiled.barriers.iter().any(|b| {
            b.resource == swap
                && b.before_pass == present_exec_pos
                && b.transition == BarrierKind::WriteToPresent
        });
        assert!(has_w2p, "WriteToPresent barrier missing; barriers: {:?}", compiled.barriers);
    }

    // ── 6. Pass counts before/after compile ───────────────────────────────────

    #[test]
    fn test_pass_counts() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(color(256, 256), true, "rt0");
        let r1 = rdg.add_resource(buf(1024), true, "ub0");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        rdg.add_pass(RdgPass { name: "P0", reads: Vec::new(), writes: alloc::vec![r0], kind: PassKind::Graphics });
        rdg.add_pass(RdgPass { name: "P1", reads: alloc::vec![r0], writes: alloc::vec![r1], kind: PassKind::Compute });
        rdg.add_pass(RdgPass { name: "DeadA", reads: Vec::new(), writes: Vec::new(), kind: PassKind::Transfer });
        rdg.add_pass(RdgPass { name: "DeadB", reads: Vec::new(), writes: Vec::new(), kind: PassKind::Compute });
        rdg.add_pass(RdgPass {
            name: "ToSwap",
            reads: alloc::vec![r0, r1],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass { name: "Present", reads: alloc::vec![swap], writes: Vec::new(), kind: PassKind::Present });

        assert_eq!(rdg.pass_count(), 6, "6 declared passes");
        assert_eq!(rdg.resource_count(), 3, "3 declared resources");

        let compiled = rdg.compile();

        assert_eq!(compiled.execution_order.len(), 4, "4 live passes");
        assert_eq!(compiled.culled.len(), 2, "2 dead passes");

        // Verify the dead ones are DeadA and DeadB.
        let culled_names: Vec<&str> = compiled.culled.iter().map(|&i| rdg.passes[i].name).collect();
        assert!(culled_names.contains(&"DeadA"));
        assert!(culled_names.contains(&"DeadB"));
    }

    // ── 7. resource_bytes unit tests ──────────────────────────────────────────

    #[test]
    fn test_resource_bytes() {
        assert_eq!(Rdg::resource_bytes(&color(1920, 1080)), 1920 * 1080 * 4);
        assert_eq!(Rdg::resource_bytes(&depth(512, 512)), 512 * 512 * 4);
        assert_eq!(Rdg::resource_bytes(&buf(65536)), 65536);
        assert_eq!(Rdg::resource_bytes(&ResourceKind::SwapchainTarget), 0);
    }

    // ── 8. Empty graph compiles without panic ────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let rdg = Rdg::new();
        let compiled = rdg.compile();
        assert!(compiled.execution_order.is_empty());
        assert!(compiled.culled.is_empty());
        assert_eq!(compiled.heap_bytes, 0);
    }

    // ── 9. Transfer pass in the chain ────────────────────────────────────────

    #[test]
    fn test_transfer_pass_live() {
        let mut rdg = Rdg::new();

        let src = rdg.add_resource(buf(4096), false, "src_buf");
        let dst = rdg.add_resource(buf(4096), true, "dst_buf");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        rdg.add_pass(RdgPass {
            name: "Upload",
            reads: alloc::vec![src],
            writes: alloc::vec![dst],
            kind: PassKind::Transfer,
        });
        rdg.add_pass(RdgPass {
            name: "Draw",
            reads: alloc::vec![dst],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        assert!(compiled.culled.is_empty(), "transfer pass should be live");
        assert_eq!(compiled.execution_order.len(), 3);
    }

    // ── Bindless resource table tests ────────────────────────────────────────

    #[test]
    fn test_bindless_table_empty() {
        let table = BindlessResourceTable::new();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_bindless_table_register_and_get() {
        let mut table = BindlessResourceTable::new();
        let idx0 = table.register(42, ResourceKind::Buffer { bytes: 1024 });
        let idx1 = table.register(7, ResourceKind::ColorTarget { w: 1920, h: 1080 });
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(table.len(), 2);

        let e0 = table.get(0).unwrap();
        assert_eq!(e0.0, 42);

        let e1 = table.get(1).unwrap();
        assert_eq!(e1.0, 7);
        assert!(table.get(2).is_none());
    }

    #[test]
    fn test_bindless_table_in_compiled_rdg() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(color(1920, 1080), true, "color");
        let r1 = rdg.add_resource(buf(4096), true, "uniforms");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        rdg.add_pass(RdgPass {
            name: "Draw",
            reads: alloc::vec![r1],
            writes: alloc::vec![r0],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "ToSwap",
            reads: alloc::vec![r0],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        // Bindless table should contain all non-swapchain resources (r0, r1 = 2).
        assert_eq!(compiled.bindless_table.len(), 2,
            "bindless table should have 2 entries (no swapchain), got {}",
            compiled.bindless_table.len());
    }

    // ── Async compute queue tests ─────────────────────────────────────────────

    #[test]
    fn test_async_compute_queue_empty() {
        let q = AsyncComputeQueue::new();
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }

    #[test]
    fn test_async_compute_no_reads_goes_async() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(color(256, 256), true, "rt");
        let r1 = rdg.add_resource(buf(1024), true, "compute_out");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // Compute pass with no reads → should be async.
        rdg.add_pass(RdgPass {
            name: "AsyncCompute",
            reads: Vec::new(),
            writes: alloc::vec![r1],
            kind: PassKind::Compute,
        });
        rdg.add_pass(RdgPass {
            name: "Draw",
            reads: alloc::vec![r1],
            writes: alloc::vec![r0],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "ToSwap",
            reads: alloc::vec![r0],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        assert!(!compiled.async_compute.is_empty(),
            "Compute pass with no reads should be in async compute queue");
        assert_eq!(compiled.async_compute.len(), 1);
    }

    #[test]
    fn test_async_compute_with_reads_stays_graphics() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(buf(512), false, "input_buf");
        let r1 = rdg.add_resource(buf(512), true, "output_buf");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // Compute pass WITH reads → should NOT be async.
        rdg.add_pass(RdgPass {
            name: "ComputeWithRead",
            reads: alloc::vec![r0],
            writes: alloc::vec![r1],
            kind: PassKind::Compute,
        });
        rdg.add_pass(RdgPass {
            name: "Draw",
            reads: alloc::vec![r1],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();

        assert!(compiled.async_compute.is_empty(),
            "Compute pass WITH reads should NOT be async (has dependency)");
    }

    // ── execute_stats tests ───────────────────────────────────────────────────

    #[test]
    fn test_execute_stats_no_async() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(color(64, 64), true, "rt");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        rdg.add_pass(RdgPass {
            name: "Draw",
            reads: Vec::new(),
            writes: alloc::vec![r0],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "ToSwap",
            reads: alloc::vec![r0],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();
        let (gfx, async_c) = compiled.execute_stats();

        assert_eq!(async_c, 0, "no async passes expected");
        assert_eq!(gfx, compiled.execution_order.len(),
            "all passes should be graphics");
    }

    #[test]
    fn test_execute_stats_with_async() {
        let mut rdg = Rdg::new();

        let r0 = rdg.add_resource(buf(256), true, "compute_buf");
        let r1 = rdg.add_resource(color(64, 64), true, "rt");
        let swap = rdg.add_resource(ResourceKind::SwapchainTarget, false, "swap");

        // One async compute (no reads).
        rdg.add_pass(RdgPass {
            name: "AsyncWork",
            reads: Vec::new(),
            writes: alloc::vec![r0],
            kind: PassKind::Compute,
        });
        rdg.add_pass(RdgPass {
            name: "Draw",
            reads: alloc::vec![r0],
            writes: alloc::vec![r1],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "ToSwap",
            reads: alloc::vec![r1],
            writes: alloc::vec![swap],
            kind: PassKind::Graphics,
        });
        rdg.add_pass(RdgPass {
            name: "Present",
            reads: alloc::vec![swap],
            writes: Vec::new(),
            kind: PassKind::Present,
        });

        let compiled = rdg.compile();
        let (gfx, async_c) = compiled.execute_stats();

        assert_eq!(async_c, 1, "one async pass expected");
        assert_eq!(gfx + async_c, compiled.execution_order.len(),
            "gfx + async should equal total passes");
    }

    #[test]
    fn test_execute_stats_empty_graph() {
        let rdg = Rdg::new();
        let compiled = rdg.compile();
        let (gfx, async_c) = compiled.execute_stats();
        assert_eq!(gfx, 0);
        assert_eq!(async_c, 0);
    }
}

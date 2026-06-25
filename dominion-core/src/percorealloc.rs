//! Per-core hardened allocator with huge-page TLB model — Lever 8 of the
//! DominionOS memory-acceleration roadmap: "Hardened, contention-free allocation
//! with huge pages."
//!
//! ## What this module does
//!
//! ### `PerCorePool`
//! Replaces a single spin-locked global heap with one [`CoreArena`] per
//! logical core.  Because each core owns its own arena, allocations from
//! different cores never touch a shared lock — the SMP bottleneck is removed.
//!
//! ### `HugePageAllocator`
//! Models 2 MiB and 1 GiB huge-page selection for large allocations.  Mapping
//! 1 GiB with 4 KiB pages costs 262 144 TLB entries; one 1 GiB page costs 1.
//! The allocator tracks the TLB pressure saved vs. the all-4 KiB baseline.
//!
//! ### `benchmark_per_core_alloc`
//! Simulates `num_cores × allocs_per_core` allocations and returns a
//! [`PerCoreBenchResult`] quantifying both the contention reduction and the
//! TLB reduction.
//!
//! Pure, safe `no_std + alloc`, consistent with the rest of `dominion-core`.

use crate::hardalloc::{Handle, HardenedAllocator};
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// PageSize — huge-page model
// ---------------------------------------------------------------------------

/// The page granularity used to map a region of memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PageSize {
    /// 4 KiB — the universal baseline page size.
    Standard = 4_096,
    /// 2 MiB — Transparent Huge Pages (THP) or explicit madvise(MADV_HUGEPAGE).
    Huge2M = 2_097_152,
    /// 1 GiB — PDPEiG huge pages (x86-64 PDPE1G feature).
    Huge1G = 1_073_741_824,
}

impl PageSize {
    /// Size of this page in bytes.
    pub fn bytes(self) -> usize {
        self as usize
    }

    /// Choose the largest page size that evenly covers `alloc_bytes`.
    ///
    /// * `>= 1 GiB` → `Huge1G`
    /// * `>= 2 MiB` → `Huge2M`
    /// * otherwise  → `Standard`
    pub fn best_for(alloc_bytes: usize) -> PageSize {
        if alloc_bytes >= PageSize::Huge1G as usize {
            PageSize::Huge1G
        } else if alloc_bytes >= PageSize::Huge2M as usize {
            PageSize::Huge2M
        } else {
            PageSize::Standard
        }
    }

    /// Number of TLB entries required to map `total_bytes` using this page size.
    pub fn tlb_entries_for(self, total_bytes: usize) -> usize {
        (total_bytes + self.bytes() - 1) / self.bytes()
    }
}

// ---------------------------------------------------------------------------
// CoreArena — one core's private allocator shard
// ---------------------------------------------------------------------------

/// One core's private shard of the hardened heap.
///
/// Because each core owns its own [`CoreArena`], threads pinned to different
/// cores never share a lock — contention drops to zero.
pub(crate) struct CoreArena {
    /// Logical core ID this arena belongs to.
    pub core_id: usize,
    inner: HardenedAllocator,
    alloc_count: u64,
    bytes_allocated: usize,
    bytes_freed: usize,
}

impl CoreArena {
    /// Create a new arena for `core_id` backed by `size` bytes of hardened
    /// heap, seeded with `seed` for randomized size-class layout.
    pub fn new(core_id: usize, size: usize, seed: u64) -> Self {
        CoreArena {
            core_id,
            inner: HardenedAllocator::new(size, seed ^ (core_id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            alloc_count: 0,
            bytes_allocated: 0,
            bytes_freed: 0,
        }
    }

    /// Allocate `class` bytes from this core's arena.
    ///
    /// `class` is a raw byte count; the underlying [`HardenedAllocator`] will
    /// round it up to the next size class.  Returns `None` if the arena is
    /// full.
    pub fn alloc(&mut self, class: usize) -> Option<Handle> {
        // Size must be at least 1; use 16 as the minimum meaningful class.
        let size = class.max(16);
        match self.inner.alloc(size) {
            Ok(h) => {
                self.alloc_count += 1;
                self.bytes_allocated += size;
                Some(h)
            }
            Err(_) => None,
        }
    }

    /// Free a previously allocated handle back to this core's arena.
    pub fn free(&mut self, handle: Handle) {
        if self.inner.free(handle).is_ok() {
            self.bytes_freed += 16; // conservative accounting; actual size unknown here
        }
    }

    /// Total number of successful allocations on this core.
    pub fn alloc_count(&self) -> u64 {
        self.alloc_count
    }

    /// Net bytes currently in use (`bytes_allocated - bytes_freed`), clamped
    /// to zero (bytes_freed tracking is conservative).
    pub fn bytes_used(&self) -> usize {
        self.bytes_allocated.saturating_sub(self.bytes_freed)
    }
}

// ---------------------------------------------------------------------------
// PerCorePool — top-level multi-core allocator
// ---------------------------------------------------------------------------

/// Per-core pool statistics.
#[derive(Clone, Debug, Default)]
pub(crate) struct PoolStats {
    /// Total allocations across all cores.
    pub total_allocs: u64,
    /// Total frees across all cores.
    pub total_frees: u64,
    /// Total bytes allocated (sum over all arenas).
    pub total_bytes_allocated: usize,
    /// Lock contentions avoided: with a single global lock, every allocation
    /// on cores 1..N would have to wait for core 0.  Per-core arenas give
    /// each core its own lock-free path.
    pub lock_contentions_avoided: u64,
    /// Number of active cores in the pool.
    pub active_cores: usize,
}

/// A pool of per-core hardened allocator arenas.
///
/// Allocations on core *i* are served entirely by `arenas[i]`, so cores never
/// compete for the same lock.
pub(crate) struct PerCorePool {
    arenas: Vec<CoreArena>,
    stats: PoolStats,
}

impl PerCorePool {
    /// Create a pool with `num_cores` arenas, each `arena_size` bytes, seeded
    /// from `seed` (each arena gets a distinct derived seed).
    pub fn new(num_cores: usize, arena_size: usize, seed: u64) -> Self {
        let arenas = (0..num_cores)
            .map(|id| CoreArena::new(id, arena_size, seed))
            .collect();
        PerCorePool {
            arenas,
            stats: PoolStats {
                active_cores: num_cores,
                ..PoolStats::default()
            },
        }
    }

    /// Allocate `class` bytes on behalf of `core_id`.
    ///
    /// `core_id` wraps with modulo so callers may use any non-negative integer.
    pub fn alloc_on_core(&mut self, core_id: usize, class: usize) -> Option<Handle> {
        let n = self.arenas.len();
        if n == 0 {
            return None;
        }
        let idx = core_id % n;
        let result = self.arenas[idx].alloc(class);
        if result.is_some() {
            self.stats.total_allocs += 1;
            self.stats.total_bytes_allocated += class.max(16);
            // Every alloc on a non-zero core would have contended a global lock.
            if idx != 0 {
                self.stats.lock_contentions_avoided += 1;
            }
        }
        result
    }

    /// Free `handle` on behalf of `core_id`.
    pub fn free_on_core(&mut self, core_id: usize, handle: Handle) {
        let n = self.arenas.len();
        if n == 0 {
            return;
        }
        let idx = core_id % n;
        self.arenas[idx].free(handle);
        self.stats.total_frees += 1;
    }

    /// A snapshot of pool-wide statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_allocs: self.stats.total_allocs,
            total_frees: self.stats.total_frees,
            total_bytes_allocated: self.stats.total_bytes_allocated,
            lock_contentions_avoided: self.stats.lock_contentions_avoided,
            active_cores: self.arenas.len(),
        }
    }

    /// How many lock contentions per-core arenas avoided vs. a global lock.
    ///
    /// With a global lock, every allocation from cores 1..N would have to
    /// wait; with per-core arenas they proceed in parallel.
    pub fn contention_savings(&self) -> u64 {
        self.stats.lock_contentions_avoided
    }

    /// Number of cores in this pool.
    pub fn num_cores(&self) -> usize {
        self.arenas.len()
    }
}

// ---------------------------------------------------------------------------
// HugePageAllocator — huge-page-aware virtual allocator model
// ---------------------------------------------------------------------------

/// Statistics for the huge-page allocator.
#[derive(Clone, Debug, Default)]
pub(crate) struct HugeAllocStats {
    /// Total number of allocations made.
    pub alloc_count: u64,
    /// Total bytes allocated.
    pub total_bytes: usize,
    /// Allocations that used 4 KiB standard pages.
    pub standard_page_allocs: u64,
    /// Allocations that used 2 MiB huge pages.
    pub huge2m_allocs: u64,
    /// Allocations that used 1 GiB huge pages.
    pub huge1g_allocs: u64,
    /// TLB entries saved compared to an all-4 KiB mapping baseline.
    pub tlb_entries_saved: usize,
}

/// A virtual allocator model that selects the best page size for each
/// allocation, minimising TLB pressure.
pub(crate) struct HugePageAllocator {
    /// `(start_offset, size_bytes, page_size_chosen)`
    allocations: Vec<(usize, usize, PageSize)>,
    next_offset: usize,
    stats: HugeAllocStats,
}

impl HugePageAllocator {
    /// Create an empty huge-page allocator.
    pub fn new() -> Self {
        HugePageAllocator {
            allocations: Vec::new(),
            next_offset: 0,
            stats: HugeAllocStats::default(),
        }
    }

    /// Allocate `bytes`, automatically choosing the best page size.
    ///
    /// Returns the start offset within the modelled virtual address space.
    pub fn alloc(&mut self, bytes: usize) -> usize {
        let ps = PageSize::best_for(bytes);
        let offset = self.next_offset;

        // TLB cost if we used standard pages vs. the chosen huge page.
        let standard_entries = PageSize::Standard.tlb_entries_for(bytes);
        let optimal_entries = ps.tlb_entries_for(bytes);
        let saved = standard_entries.saturating_sub(optimal_entries);

        self.allocations.push((offset, bytes, ps));
        self.next_offset += bytes;

        self.stats.alloc_count += 1;
        self.stats.total_bytes += bytes;
        self.stats.tlb_entries_saved += saved;
        match ps {
            PageSize::Standard => self.stats.standard_page_allocs += 1,
            PageSize::Huge2M => self.stats.huge2m_allocs += 1,
            PageSize::Huge1G => self.stats.huge1g_allocs += 1,
        }

        offset
    }

    /// Total TLB entries saved vs. the all-4 KiB mapping baseline.
    pub fn tlb_savings(&self) -> usize {
        self.stats.tlb_entries_saved
    }

    /// `standard_tlb_entries / optimal_tlb_entries` across all allocations.
    pub fn tlb_reduction_factor(&self) -> f64 {
        let mut standard_total = 0usize;
        let mut optimal_total = 0usize;
        for &(_, bytes, ps) in &self.allocations {
            standard_total += PageSize::Standard.tlb_entries_for(bytes);
            optimal_total += ps.tlb_entries_for(bytes);
        }
        if optimal_total == 0 {
            1.0
        } else {
            standard_total as f64 / optimal_total as f64
        }
    }

    /// A snapshot of allocation statistics.
    pub fn stats(&self) -> HugeAllocStats {
        self.stats.clone()
    }
}

impl Default for HugePageAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// benchmark_per_core_alloc
// ---------------------------------------------------------------------------

/// Result of [`benchmark_per_core_alloc`].
#[derive(Clone, Debug)]
pub(crate) struct PerCoreBenchResult {
    /// Number of cores simulated.
    pub num_cores: usize,
    /// Allocations performed per core.
    pub allocs_per_core: usize,
    /// Total successful allocations across all cores.
    pub total_allocs: u64,
    /// Lock contentions avoided vs. a global-lock baseline.
    pub lock_contentions_avoided: u64,
    /// `total_allocs / allocs_on_core_0`: how many times more work proceeds
    /// in parallel than if serialised through a single lock.
    pub contention_reduction_factor: f64,
    /// `standard_tlb_entries / huge_page_entries` across the workload.
    pub tlb_reduction_factor: f64,
    /// TLB entries needed with all 4 KiB pages.
    pub tlb_entries_standard: usize,
    /// TLB entries needed with optimal huge-page selection.
    pub tlb_entries_optimal: usize,
}

/// Simulate `num_cores × allocs_per_core` allocations and measure the
/// contention and TLB benefits of per-core hardened allocation with huge pages.
///
/// # Panics
/// Panics (via assertions) if the model invariants are violated — e.g. if the
/// arena is too small to serve all allocations.
pub(crate) fn benchmark_per_core_alloc(num_cores: usize, allocs_per_core: usize) -> PerCoreBenchResult {
    // -----------------------------------------------------------------------
    // Step 1 — Per-core allocation
    // -----------------------------------------------------------------------
    // Each allocation is 16 bytes (class 0 in hardalloc terms).
    // Arena band = arena_size / 8, stride per alloc = 32 (16 user + 8+8 guard).
    // We need: band >= allocs_per_core * 32  =>  arena_size >= allocs_per_core * 256.
    // Add a generous 8× headroom for the band-isolation hash distribution.
    let arena_size = (allocs_per_core * 256).max(65536);
    let mut pool = PerCorePool::new(num_cores, arena_size, 0xDEAD_BEEF_CAFE_BABE);

    // Track handles so we can free them (tests arena integrity).
    let mut handles: Vec<(usize, Handle)> = Vec::new();

    for core in 0..num_cores {
        for _ in 0..allocs_per_core {
            if let Some(h) = pool.alloc_on_core(core, 16) {
                handles.push((core, h));
            }
        }
    }

    let stats = pool.stats();
    let total_allocs = stats.total_allocs;
    let lock_contentions_avoided = stats.lock_contentions_avoided;

    // Allocs on core 0 = allocs_per_core (assuming none failed).
    let allocs_on_core_0 = allocs_per_core as u64;
    let contention_reduction_factor =
        total_allocs as f64 / allocs_on_core_0.max(1) as f64;

    // -----------------------------------------------------------------------
    // Step 2 — Huge-page TLB model
    // -----------------------------------------------------------------------
    let mut huge = HugePageAllocator::new();
    let workload_sizes: &[usize] = &[
        4_096,           // 4 KiB  → Standard
        2_097_152,       // 2 MiB  → Huge2M
        4 * 2_097_152,   // 4 MiB  → Huge2M
        1_073_741_824,   // 1 GiB  → Huge1G
        2 * 1_073_741_824, // 2 GiB → Huge1G
    ];
    for &sz in workload_sizes {
        huge.alloc(sz);
    }

    let tlb_reduction_factor = huge.tlb_reduction_factor();

    // Compute totals for the result struct.
    let mut tlb_entries_standard = 0usize;
    let mut tlb_entries_optimal = 0usize;
    for &sz in workload_sizes {
        let ps = PageSize::best_for(sz);
        tlb_entries_standard += PageSize::Standard.tlb_entries_for(sz);
        tlb_entries_optimal += ps.tlb_entries_for(sz);
    }

    // -----------------------------------------------------------------------
    // Step 3 — Free handles (exercises the hardened free path)
    // -----------------------------------------------------------------------
    for (core, h) in handles {
        pool.free_on_core(core, h);
    }

    // -----------------------------------------------------------------------
    // Assertions
    // -----------------------------------------------------------------------
    let expected_contentions = (allocs_per_core * num_cores.saturating_sub(1)) as u64;
    assert_eq!(
        lock_contentions_avoided, expected_contentions,
        "lock_contentions_avoided should equal allocs_per_core * (num_cores - 1)"
    );
    assert!(
        contention_reduction_factor >= num_cores as f64 * 0.8,
        "contention_reduction_factor {contention_reduction_factor} < {} (num_cores * 0.8)",
        num_cores as f64 * 0.8
    );
    assert!(
        tlb_reduction_factor > 100.0,
        "tlb_reduction_factor {tlb_reduction_factor} should be > 100 (1 GiB alloc: 262144 4KiB pages vs 1 huge page)"
    );

    PerCoreBenchResult {
        num_cores,
        allocs_per_core,
        total_allocs,
        lock_contentions_avoided,
        contention_reduction_factor,
        tlb_reduction_factor,
        tlb_entries_standard,
        tlb_entries_optimal,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_size_best_for_selects_correctly() {
        assert_eq!(PageSize::best_for(4_096), PageSize::Standard);
        assert_eq!(PageSize::best_for(1_000), PageSize::Standard);
        assert_eq!(PageSize::best_for(2_097_152), PageSize::Huge2M);
        assert_eq!(PageSize::best_for(4 * 2_097_152), PageSize::Huge2M);
        assert_eq!(PageSize::best_for(1_073_741_824), PageSize::Huge1G);
        assert_eq!(PageSize::best_for(2 * 1_073_741_824), PageSize::Huge1G);
    }

    #[test]
    fn tlb_entries_huge_page_far_fewer_than_standard() {
        // 1 GiB allocation:
        //   Standard (4 KiB):  1_073_741_824 / 4_096 = 262_144 entries
        //   Huge1G  (1 GiB):   1_073_741_824 / 1_073_741_824 = 1 entry
        let one_gib = 1_073_741_824usize;
        let std_entries = PageSize::Standard.tlb_entries_for(one_gib);
        let huge_entries = PageSize::Huge1G.tlb_entries_for(one_gib);
        assert_eq!(std_entries, 262_144);
        assert_eq!(huge_entries, 1);
        assert!(std_entries > huge_entries * 100_000);
    }

    #[test]
    fn core_arena_alloc_and_free_basic() {
        let mut arena = CoreArena::new(0, 1 << 16, 0xABCD_1234);
        let h = arena.alloc(32).expect("alloc should succeed");
        assert_eq!(arena.alloc_count(), 1);
        assert!(arena.bytes_used() > 0);
        arena.free(h);
    }

    #[test]
    fn per_core_pool_distributes_across_cores() {
        let mut pool = PerCorePool::new(4, 1 << 16, 42);
        let h0 = pool.alloc_on_core(0, 16).expect("core 0 alloc");
        let h1 = pool.alloc_on_core(1, 16).expect("core 1 alloc");
        // Core 0 and core 1 should each have exactly 1 alloc in their arena.
        assert_eq!(pool.arenas[0].alloc_count(), 1);
        assert_eq!(pool.arenas[1].alloc_count(), 1);
        // Handles are logically independent (different arenas).
        assert_ne!(pool.arenas[0].core_id, pool.arenas[1].core_id);
        pool.free_on_core(0, h0);
        pool.free_on_core(1, h1);
    }

    #[test]
    fn contention_savings_equals_non_zero_core_allocs() {
        // 4 cores, 10 allocs each → cores 1,2,3 each do 10 allocs = 30 saved.
        let num_cores = 4;
        let allocs_per_core = 10;
        let arena_size = 1 << 16;
        let mut pool = PerCorePool::new(num_cores, arena_size, 0xBEEF);
        for core in 0..num_cores {
            for _ in 0..allocs_per_core {
                pool.alloc_on_core(core, 16);
            }
        }
        let saved = pool.contention_savings();
        assert_eq!(saved, (allocs_per_core * (num_cores - 1)) as u64);
    }

    #[test]
    fn huge_page_allocator_reduces_tlb_pressure() {
        let mut ha = HugePageAllocator::new();
        ha.alloc(1_073_741_824); // 1 GiB → Huge1G
        ha.alloc(2_097_152);     // 2 MiB → Huge2M
        ha.alloc(4_096);         // 4 KiB → Standard

        let factor = ha.tlb_reduction_factor();
        // The 1 GiB alloc alone saves 262143 entries; factor must be >> 1.
        assert!(factor > 100.0, "factor was {factor}");

        let stats = ha.stats();
        assert_eq!(stats.huge1g_allocs, 1);
        assert_eq!(stats.huge2m_allocs, 1);
        assert_eq!(stats.standard_page_allocs, 1);
    }

    #[test]
    fn benchmark_shows_contention_reduction() {
        let result = benchmark_per_core_alloc(4, 100);
        assert!(
            result.contention_reduction_factor >= 3.0,
            "contention_reduction_factor was {}",
            result.contention_reduction_factor
        );
        assert_eq!(result.total_allocs, 400);
        assert_eq!(result.lock_contentions_avoided, 300);
    }

    #[test]
    fn tlb_reduction_large_alloc() {
        let mut ha = HugePageAllocator::new();
        ha.alloc(1_073_741_824); // 1 GiB
        let factor = ha.tlb_reduction_factor();
        assert!(
            factor > 100.0,
            "tlb_reduction_factor for 1 GiB alloc should be > 100, got {factor}"
        );
    }
}

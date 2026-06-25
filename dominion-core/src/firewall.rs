//! Capability Firewall — **Stage 11.14** (intra-domain authority control).
//!
//! In a Single Address Space OS, memory visibility is not a security boundary;
//! authority *topology* is. The firewall models the system as a directed
//! authority graph (`Identity → Cell → Capability → Object`) and answers one
//! question: can subject A reach object B through a *valid capability path*? If no
//! such path exists, access is impossible regardless of memory visibility.
//!
//! On top of reachability it enforces **dynamic domain segmentation** (cross-domain
//! flow denied unless explicitly authorised), **recursive revocation** (revoking a
//! capability cuts everything derived from it), **rate limiting**, and
//! **quarantine** of suspicious cells. Pure, safe, host-tested.

use crate::hash::Hash256;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;

pub type NodeId = u64;

/// Isolated security domains. Cross-domain authority flow is denied by default.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Domain {
    System,
    Personal,
    Financial,
    Medical,
    Infrastructure,
    Development,
    AiAgent,
    ExternalNetwork,
}

/// Why the firewall refused an operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FwError {
    NoSuchNode,
    CrossDomainDenied,
    RateLimited,
}

/// The global capability authority graph plus its enforcement state.
pub struct CapabilityFirewall {
    domain: BTreeMap<NodeId, Domain>,
    /// Delegation edges: `from` confers reachability to `to`.
    edges: Vec<(NodeId, NodeId)>,
    revoked: BTreeSet<NodeId>,
    quarantined: BTreeSet<NodeId>,
    allow_cross: BTreeSet<(Domain, Domain)>,
    quota: BTreeMap<NodeId, u32>,
}

impl CapabilityFirewall {
    pub fn new() -> CapabilityFirewall {
        CapabilityFirewall {
            domain: BTreeMap::new(),
            edges: Vec::new(),
            revoked: BTreeSet::new(),
            quarantined: BTreeSet::new(),
            allow_cross: BTreeSet::new(),
            quota: BTreeMap::new(),
        }
    }

    /// Register a node (identity/cell/capability/object) in a domain.
    pub fn register(&mut self, node: NodeId, domain: Domain) {
        self.domain.insert(node, domain);
    }

    fn domain_of(&self, node: NodeId) -> Option<Domain> {
        self.domain.get(&node).copied()
    }

    /// Explicitly authorise a one-directional cross-domain flow.
    pub fn authorize_cross(&mut self, from: Domain, to: Domain) {
        self.allow_cross.insert((from, to));
    }

    /// Add a delegation edge. Cross-domain delegation is refused unless authorised.
    pub fn delegate(&mut self, from: NodeId, to: NodeId) -> Result<(), FwError> {
        let df = self.domain_of(from).ok_or(FwError::NoSuchNode)?;
        let dt = self.domain_of(to).ok_or(FwError::NoSuchNode)?;
        if df != dt && !self.allow_cross.contains(&(df, dt)) {
            return Err(FwError::CrossDomainDenied);
        }
        self.edges.push((from, to));
        Ok(())
    }

    fn traversable(&self, node: NodeId) -> bool {
        !self.revoked.contains(&node) && !self.quarantined.contains(&node)
    }

    /// BFS from `from` over valid (non-revoked, non-quarantined, domain-legal) edges.
    /// Returns the set of all reachable nodes (including `from` itself).
    fn bfs_reachable(&self, from: NodeId) -> BTreeSet<NodeId> {
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::new();
        if self.traversable(from) {
            seen.insert(from);
            queue.push_back(from);
        }
        while let Some(n) = queue.pop_front() {
            for &(a, b) in &self.edges {
                if a != n || seen.contains(&b) || !self.traversable(b) {
                    continue;
                }
                // Domain rule re-checked at traversal time (segmentation is live).
                if let (Some(da), Some(db)) = (self.domain_of(a), self.domain_of(b)) {
                    if da != db && !self.allow_cross.contains(&(da, db)) {
                        continue;
                    }
                }
                seen.insert(b);
                queue.push_back(b);
            }
        }
        seen
    }

    /// Can `from` reach `to` along valid (non-revoked, non-quarantined,
    /// domain-legal) capability edges?
    pub fn reachable(&self, from: NodeId, to: NodeId) -> bool {
        if !self.traversable(from) || !self.traversable(to) {
            return false;
        }
        self.bfs_reachable(from).contains(&to)
    }

    /// Recursively revoke a node and everything derived from it (all capabilities
    /// reachable through its edges).
    pub fn revoke(&mut self, node: NodeId) {
        let mut queue = VecDeque::new();
        queue.push_back(node);
        while let Some(n) = queue.pop_front() {
            if !self.revoked.insert(n) {
                continue;
            }
            for &(a, b) in &self.edges {
                if a == n && !self.revoked.contains(&b) {
                    queue.push_back(b);
                }
            }
        }
    }

    pub fn is_revoked(&self, node: NodeId) -> bool {
        self.revoked.contains(&node)
    }

    /// Move a suspicious cell into quarantine (its edges stop conferring authority).
    pub fn quarantine(&mut self, cell: NodeId) {
        self.quarantined.insert(cell);
    }

    /// Set a capability's usage quota (rate limiting).
    pub fn set_quota(&mut self, node: NodeId, ops: u32) {
        self.quota.insert(node, ops);
    }

    /// Consume one unit of a rate-limited capability.
    pub fn consume(&mut self, node: NodeId) -> Result<(), FwError> {
        match self.quota.get_mut(&node) {
            Some(q) if *q > 0 => {
                *q -= 1;
                Ok(())
            }
            Some(_) => Err(FwError::RateLimited),
            None => Ok(()), // unlimited
        }
    }

    /// Authority-diffusion metric: how many distinct nodes this node can reach
    /// (fan-out), ignoring domain segmentation rules. Excessive diffusion signals
    /// privilege creep regardless of whether cross-domain flows are authorised.
    pub fn diffusion(&self, from: NodeId) -> usize {
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::new();
        if self.traversable(from) {
            seen.insert(from);
            queue.push_back(from);
        }
        while let Some(n) = queue.pop_front() {
            for &(a, b) in &self.edges {
                if a == n && !seen.contains(&b) && self.traversable(b) {
                    seen.insert(b);
                    queue.push_back(b);
                }
            }
        }
        seen.len().saturating_sub(1) // exclude self
    }

    /// How many **distinct domains** a node's authority can reach. A capability that can
    /// touch many domains is a classic confused-deputy / escalation pattern.
    pub fn domains_reached(&self, from: NodeId) -> usize {
        self.bfs_reachable(from)
            .into_iter()
            .filter_map(|n| self.domain_of(n))
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// **Escalation-pattern detection**: every registered, live node whose authority
    /// diffusion exceeds `max_diffusion` **or** whose reach spans more than
    /// `max_domains` domains — i.e. privilege creep / cross-domain accumulation. The
    /// verified firewall, not an AI, makes the call; this just surfaces the candidates.
    pub fn detect_escalation(&self, max_diffusion: usize, max_domains: usize) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self
            .domain
            .keys()
            .copied()
            .filter(|&n| self.traversable(n))
            .filter(|&n| self.diffusion(n) > max_diffusion || self.domains_reached(n) > max_domains)
            .collect();
        out.sort_unstable();
        out
    }
}

/// An authority event recorded on the firewall's immutable provenance ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityEvent {
    Delegated(NodeId, NodeId),
    Revoked(NodeId),
    Quarantined(NodeId),
}

/// Encode an [`AuthorityEvent`] into its 17-byte wire tag — shared by [`ProvenanceLedger::record`]
/// and [`ProvenanceLedger::intact`] so the encoding cannot silently diverge.
fn tag_bytes(event: &AuthorityEvent) -> [u8; 17] {
    let mut t = [0u8; 17];
    match event {
        AuthorityEvent::Delegated(a, b) => {
            t[0] = 1;
            t[1..9].copy_from_slice(&a.to_le_bytes());
            t[9..17].copy_from_slice(&b.to_le_bytes());
        }
        AuthorityEvent::Revoked(a) => {
            t[0] = 2;
            t[1..9].copy_from_slice(&a.to_le_bytes());
        }
        AuthorityEvent::Quarantined(a) => {
            t[0] = 3;
            t[1..9].copy_from_slice(&a.to_le_bytes());
        }
    }
    t
}

/// An immutable, hash-chained **provenance ledger** of authority-graph changes — so every
/// delegation / revocation / quarantine is tamper-evidently auditable (Stage 11.14).
#[derive(Default)]
pub struct ProvenanceLedger {
    entries: alloc::vec::Vec<(AuthorityEvent, Hash256)>,
}

impl ProvenanceLedger {
    pub fn new() -> ProvenanceLedger {
        ProvenanceLedger { entries: alloc::vec::Vec::new() }
    }

    /// Append an authority event, chaining it to all prior entries.
    pub fn record(&mut self, event: AuthorityEvent) {
        let prev = self.entries.last().map(|(_, h)| *h).unwrap_or(Hash256::of(b"fw-prov-genesis"));
        let mut input = alloc::vec::Vec::with_capacity(64);
        input.extend_from_slice(&prev.0);
        input.extend_from_slice(&tag_bytes(&event));
        let chain = Hash256::of(&input);
        self.entries.push((event, chain));
    }

    /// The events in order.
    pub fn events(&self) -> alloc::vec::Vec<AuthorityEvent> {
        self.entries.iter().map(|(e, _)| *e).collect()
    }

    /// The number of recorded events.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify the hash chain is internally consistent (tamper-evident).
    pub fn intact(&self) -> bool {
        let mut prev = Hash256::of(b"fw-prov-genesis");
        for (event, chain) in &self.entries {
            let mut input = alloc::vec::Vec::with_capacity(64);
            input.extend_from_slice(&prev.0);
            input.extend_from_slice(&tag_bytes(event));
            if Hash256::of(&input) != *chain {
                return false;
            }
            prev = *chain;
        }
        true
    }
}

impl Default for CapabilityFirewall {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> CapabilityFirewall {
        let mut fw = CapabilityFirewall::new();
        // identity(1) -> cell(2) -> capability(3) -> object(4), all Financial.
        for n in 1..=4 {
            fw.register(n, Domain::Financial);
        }
        fw.delegate(1, 2).unwrap();
        fw.delegate(2, 3).unwrap();
        fw.delegate(3, 4).unwrap();
        fw
    }

    #[test]
    fn reachable_through_a_valid_chain() {
        let fw = graph();
        assert!(fw.reachable(1, 4));
        assert!(!fw.reachable(4, 1)); // edges are directed
    }

    #[test]
    fn cross_domain_delegation_denied_by_default() {
        let mut fw = CapabilityFirewall::new();
        fw.register(1, Domain::Financial);
        fw.register(2, Domain::AiAgent);
        assert_eq!(fw.delegate(1, 2).unwrap_err(), FwError::CrossDomainDenied);
        fw.authorize_cross(Domain::Financial, Domain::AiAgent);
        assert!(fw.delegate(1, 2).is_ok());
        assert!(fw.reachable(1, 2));
    }

    #[test]
    fn revocation_propagates_recursively() {
        let mut fw = graph();
        assert!(fw.reachable(1, 4));
        fw.revoke(2); // revoking the cell should cut 3 and 4 too
        assert!(fw.is_revoked(3));
        assert!(fw.is_revoked(4));
        assert!(!fw.reachable(1, 4));
    }

    #[test]
    fn quarantine_isolates_a_cell() {
        let mut fw = graph();
        fw.quarantine(2);
        assert!(!fw.reachable(1, 4));
    }

    #[test]
    fn rate_limiting_denies_after_quota() {
        let mut fw = graph();
        fw.set_quota(3, 2);
        assert!(fw.consume(3).is_ok());
        assert!(fw.consume(3).is_ok());
        assert_eq!(fw.consume(3).unwrap_err(), FwError::RateLimited);
    }

    #[test]
    fn diffusion_measures_fan_out() {
        let fw = graph();
        assert_eq!(fw.diffusion(1), 3); // reaches 2,3,4
        assert_eq!(fw.diffusion(3), 1); // reaches only 4
    }

    #[test]
    fn escalation_detection_flags_high_diffusion_nodes() {
        let fw = graph(); // node 1 reaches 3 others, all one domain
        // With a low diffusion cap, node 1 is flagged; a high cap flags nobody.
        assert!(fw.detect_escalation(2, 5).contains(&1));
        assert!(fw.detect_escalation(10, 5).is_empty());
        // One domain reached, so the domain-span detector doesn't fire here.
        assert_eq!(fw.domains_reached(1), 1);
    }

    #[test]
    fn escalation_detection_flags_cross_domain_accumulation() {
        let mut fw = CapabilityFirewall::new();
        fw.register(1, Domain::AiAgent);
        fw.register(2, Domain::Personal);
        fw.register(3, Domain::Financial);
        fw.authorize_cross(Domain::AiAgent, Domain::Personal);
        fw.authorize_cross(Domain::Personal, Domain::Financial);
        fw.delegate(1, 2).unwrap();
        fw.delegate(2, 3).unwrap();
        // Node 1's authority spans 3 domains — a confused-deputy pattern.
        assert_eq!(fw.domains_reached(1), 3);
        assert!(fw.detect_escalation(100, 2).contains(&1));
    }

    #[test]
    fn provenance_ledger_is_immutable_and_tamper_evident() {
        let mut led = ProvenanceLedger::new();
        led.record(AuthorityEvent::Delegated(1, 2));
        led.record(AuthorityEvent::Revoked(2));
        led.record(AuthorityEvent::Quarantined(3));
        assert_eq!(led.len(), 3);
        assert!(led.intact());
        assert_eq!(led.events()[1], AuthorityEvent::Revoked(2));
    }
}

//! Processes, isolation, and scheduling — milestone **M2**.
//!
//! DominionOS is a Single Address Space OS (SRS Stage 3): there are no per-process
//! page tables. Isolation is *capability* isolation. A process here is a
//! **domain** — a Software-Isolated Process (SIP) that owns a [`Capability`] over
//! a region of the single address space and may only touch memory that
//! capability authorises. Domains share nothing writable; they communicate only
//! through **explicit channels** carrying references to immutable, content-
//! addressed objects (zero-copy message passing, SRS Stage 3/4).
//!
//! Scheduling is **cooperative round-robin**. This matches the multikernel model
//! where "a minimal CPU driver executes on each core as a non-preemptible event
//! handler" (SRS Stage 4): a domain runs a step, then yields. The scheduler is
//! pure policy/data — it decides *what runs next* and routes messages; the caller
//! (the kernel executor, or a host test) owns the actual task code and reports
//! back whether the step yielded or finished. That separation keeps this module
//! free of `unsafe` and fully deterministic, so it is exhaustively host-testable.

use crate::capability::{CapError, Capability, Rights};
use crate::object::ObjectId;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

/// Stable identifier for a domain.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct DomainId(pub u64);

/// Lifecycle state of a domain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DomainState {
    Ready,
    Running,
    Finished,
}

/// The result the caller reports after running one step of a domain's task.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// The task did some work and wants to be rescheduled.
    Yield,
    /// The task has completed.
    Done,
}

/// A zero-copy message: it carries a *reference* (content hash) to an immutable
/// object in the shared graph, not a byte copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Message {
    pub from: DomainId,
    pub payload: ObjectId,
}

/// Why an inter-domain operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IpcError {
    /// No domain with that id exists.
    NoSuchDomain,
    /// No channel has been opened between these two domains.
    NoChannel,
}

struct Domain {
    id: DomainId,
    name: String,
    capability: Capability,
    state: DomainState,
    inbox: VecDeque<Message>,
    steps_run: u32,
}

/// The cooperative, capability-isolated scheduler.
pub struct Scheduler {
    domains: Vec<Domain>,
    run_queue: VecDeque<DomainId>,
    /// Unordered pairs of domains permitted to message each other.
    channels: Vec<(DomainId, DomainId)>,
    next_id: u64,
    /// The order in which domains were dispatched — used by tests and the `ps`
    /// view to show the round-robin interleaving.
    pub trace: Vec<DomainId>,
}

impl Scheduler {
    pub fn new() -> Scheduler {
        Scheduler {
            domains: Vec::new(),
            run_queue: VecDeque::new(),
            channels: Vec::new(),
            next_id: 1,
            trace: Vec::new(),
        }
    }

    fn index(&self, id: DomainId) -> Option<usize> {
        self.domains.iter().position(|d| d.id == id)
    }

    /// Create a domain owning `capability`, in the Ready state and enqueued.
    pub fn spawn(&mut self, name: impl Into<String>, capability: Capability) -> DomainId {
        let id = DomainId(self.next_id);
        self.next_id += 1;
        self.domains.push(Domain {
            id,
            name: name.into(),
            capability,
            state: DomainState::Ready,
            inbox: VecDeque::new(),
            steps_run: 0,
        });
        self.run_queue.push_back(id);
        id
    }

    pub fn domain_count(&self) -> usize {
        self.domains.len()
    }

    pub fn live_count(&self) -> usize {
        self.domains.iter().filter(|d| d.state != DomainState::Finished).count()
    }

    pub fn state(&self, id: DomainId) -> Option<DomainState> {
        self.index(id).map(|i| self.domains[i].state)
    }

    pub fn name(&self, id: DomainId) -> Option<&str> {
        self.index(id).map(|i| self.domains[i].name.as_str())
    }

    pub fn steps_run(&self, id: DomainId) -> u32 {
        self.index(id).map(|i| self.domains[i].steps_run).unwrap_or(0)
    }

    /// Pick the next Ready domain (round-robin) and mark it Running. The caller
    /// runs its task step, then calls [`yield_back`](Self::yield_back) or
    /// [`finish`](Self::finish).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<DomainId> {
        let id = self.run_queue.pop_front()?;
        if let Some(i) = self.index(id) {
            self.domains[i].state = DomainState::Running;
            self.domains[i].steps_run += 1;
            self.trace.push(id);
        }
        Some(id)
    }

    /// Add `n` dispatch steps to a domain *without* running it. Lets a host bias the
    /// recorded CPU toward the domain that is actually doing work (e.g. the focused
    /// window), so the Task Manager's per-process CPU reflects real activity rather than
    /// an even round-robin share. No-op on a finished/unknown domain.
    pub fn charge(&mut self, id: DomainId, n: u32) {
        if let Some(i) = self.index(id) {
            if self.domains[i].state != DomainState::Finished {
                self.domains[i].steps_run = self.domains[i].steps_run.saturating_add(n);
            }
        }
    }

    /// The domain ran a step and should be rescheduled at the back of the queue.
    pub fn yield_back(&mut self, id: DomainId) {
        if let Some(i) = self.index(id) {
            if self.domains[i].state != DomainState::Finished {
                self.domains[i].state = DomainState::Ready;
                self.run_queue.push_back(id);
            }
        }
    }

    /// The domain completed; it leaves the run queue permanently.
    pub fn finish(&mut self, id: DomainId) {
        if let Some(i) = self.index(id) {
            self.domains[i].state = DomainState::Finished;
        }
    }

    /// True while any domain remains unfinished.
    pub fn has_runnable(&self) -> bool {
        !self.run_queue.is_empty()
    }

    // ---- isolation -------------------------------------------------------

    /// Check that domain `id` may access `[addr, addr+size)` with `rights`,
    /// using *its own* capability. Access outside the domain's region traps with
    /// `OutOfBounds` — this is the SIP isolation boundary, enforced by the same
    /// capability algebra the hardware would use.
    pub fn check_access(&self, id: DomainId, addr: u64, size: u64, rights: Rights) -> Result<(), CapError> {
        let i = self.index(id).ok_or(CapError::TagInvalid)?;
        self.domains[i].capability.check(addr, size, rights)
    }

    // ---- IPC -------------------------------------------------------------

    /// Open a bidirectional channel between two domains (an explicit contractual
    /// link; without it, messaging is refused).
    pub fn open_channel(&mut self, a: DomainId, b: DomainId) -> Result<(), IpcError> {
        if self.index(a).is_none() || self.index(b).is_none() {
            return Err(IpcError::NoSuchDomain);
        }
        if !self.channel_exists(a, b) {
            self.channels.push((a, b));
        }
        Ok(())
    }

    fn channel_exists(&self, a: DomainId, b: DomainId) -> bool {
        self.channels
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
    }

    /// Send a zero-copy message (an object reference) from `from` to `to`.
    /// Requires an open channel — domains cannot inject messages into arbitrary
    /// peers.
    pub fn send(&mut self, from: DomainId, to: DomainId, payload: ObjectId) -> Result<(), IpcError> {
        if self.index(from).is_none() {
            return Err(IpcError::NoSuchDomain);
        }
        let ti = self.index(to).ok_or(IpcError::NoSuchDomain)?;
        if !self.channel_exists(from, to) {
            return Err(IpcError::NoChannel);
        }
        self.domains[ti].inbox.push_back(Message { from, payload });
        Ok(())
    }

    /// Dequeue the next message in a domain's inbox.
    pub fn recv(&mut self, id: DomainId) -> Option<Message> {
        let i = self.index(id)?;
        self.domains[i].inbox.pop_front()
    }

    pub fn inbox_len(&self, id: DomainId) -> usize {
        self.index(id).map(|i| self.domains[i].inbox.len()).unwrap_or(0)
    }

    /// A read-only snapshot of every domain — what the Terminal's `ps` and the Task
    /// Manager render. Order is spawn order (stable), so the view doesn't jump around.
    pub fn snapshot(&self) -> Vec<DomainInfo> {
        self.domains
            .iter()
            .map(|d| DomainInfo {
                id: d.id,
                name: d.name.clone(),
                state: d.state,
                steps: d.steps_run,
                inbox: d.inbox.len(),
                base: d.capability.base(),
                len: d.capability.len(),
            })
            .collect()
    }

    /// Terminate a domain by id (the Task Manager's "End task"): mark it Finished and
    /// drop it from the run queue. Returns false if the id is unknown. A domain that
    /// has already finished stays finished. This is the cooperative analogue of a kill.
    pub fn kill(&mut self, id: DomainId) -> bool {
        match self.index(id) {
            Some(i) => {
                self.domains[i].state = DomainState::Finished;
                self.run_queue.retain(|q| *q != id);
                true
            }
            None => false,
        }
    }
}

/// A read-only snapshot of one domain, for `ps` and the Task Manager.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainInfo {
    pub id: DomainId,
    pub name: String,
    pub state: DomainState,
    pub steps: u32,
    pub inbox: usize,
    /// The base address of the domain's capability region (its memory footprint).
    pub base: u64,
    /// The length (bytes) of the domain's capability region.
    pub len: u64,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Hash256;
    use alloc::collections::BTreeMap;

    fn region(base: u64, len: u64) -> Capability {
        Capability::mint(base, len, Rights::ALL)
    }

    #[test]
    fn round_robin_interleaves_until_done() {
        // Two domains each needing 3 steps; cooperative scheduling must interleave
        // them A,B,A,B,A,B.
        let mut s = Scheduler::new();
        let a = s.spawn("a", region(0, 0x1000));
        let b = s.spawn("b", region(0x1000, 0x1000));
        let mut remaining: BTreeMap<DomainId, u32> = BTreeMap::new();
        remaining.insert(a, 3);
        remaining.insert(b, 3);

        while let Some(id) = s.next() {
            let r = remaining.get_mut(&id).unwrap();
            *r -= 1;
            if *r == 0 {
                s.finish(id);
            } else {
                s.yield_back(id);
            }
        }
        assert_eq!(s.trace, [a, b, a, b, a, b]);
        assert_eq!(s.live_count(), 0);
        assert_eq!(s.steps_run(a), 3);
    }

    #[test]
    fn domain_can_access_its_own_region() {
        let mut s = Scheduler::new();
        let d = s.spawn("d", region(0x2000, 0x1000));
        assert!(s.check_access(d, 0x2000, 16, Rights::READ).is_ok());
        assert!(s.check_access(d, 0x2ff0, 16, Rights::WRITE).is_ok());
    }

    #[test]
    fn domain_cannot_touch_another_domains_region() {
        let mut s = Scheduler::new();
        let a = s.spawn("a", region(0x2000, 0x1000));
        let _b = s.spawn("b", region(0x3000, 0x1000));
        // a reaching into b's region [0x3000,0x4000) must trap.
        assert_eq!(
            s.check_access(a, 0x3000, 16, Rights::READ).unwrap_err(),
            CapError::OutOfBounds
        );
    }

    #[test]
    fn ipc_requires_an_open_channel() {
        let mut s = Scheduler::new();
        let a = s.spawn("a", region(0, 0x1000));
        let b = s.spawn("b", region(0x1000, 0x1000));
        let msg = Hash256::of(b"payload");
        // No channel yet.
        assert_eq!(s.send(a, b, msg).unwrap_err(), IpcError::NoChannel);
        s.open_channel(a, b).unwrap();
        assert!(s.send(a, b, msg).is_ok());
        let got = s.recv(b).unwrap();
        assert_eq!(got.from, a);
        assert_eq!(got.payload, msg);
    }

    #[test]
    fn messages_are_zero_copy_references() {
        // The payload is the object's content hash — sender and receiver share the
        // same immutable object, no bytes are copied.
        let mut s = Scheduler::new();
        let a = s.spawn("a", region(0, 0x1000));
        let b = s.spawn("b", region(0x1000, 0x1000));
        s.open_channel(a, b).unwrap();
        let obj = Hash256::of(b"big shared object");
        s.send(a, b, obj).unwrap();
        assert_eq!(s.recv(b).unwrap().payload, obj);
    }

    #[test]
    fn send_to_unknown_domain_errors() {
        let mut s = Scheduler::new();
        let a = s.spawn("a", region(0, 0x1000));
        assert_eq!(s.send(a, DomainId(999), Hash256::ZERO).unwrap_err(), IpcError::NoSuchDomain);
    }

    #[test]
    fn finished_domains_do_not_reschedule() {
        let mut s = Scheduler::new();
        let a = s.spawn("a", region(0, 0x1000));
        let _ = s.next();
        s.finish(a);
        s.yield_back(a); // must be a no-op
        assert!(s.next().is_none());
        assert_eq!(s.live_count(), 0);
    }
}

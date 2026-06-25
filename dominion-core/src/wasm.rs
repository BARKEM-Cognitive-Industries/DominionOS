//! Polyglot sandbox VM (see `docs/language/polyglot-via-sandbox.md`).
//!
//! Dominion is the only language that runs *intralingually*, with direct access to
//! the object graph and capabilities. Every **other** language — C, Python, Rust,
//! JS — runs as guest bytecode inside this **capability-bounded sandbox VM** (a
//! WASM-style stack machine). The sandbox is the airlock for code: a guest can
//! compute over its own linear memory and call only the **host functions it was
//! explicitly granted**, and *nothing else*. It cannot name, reach, or corrupt the
//! kernel, the object graph, other domains, or the Dominion runtime.
//!
//! Three guarantees make that real and are each tested here:
//! * **Bounded memory** — loads/stores outside the guest's linear memory trap.
//! * **Bounded execution** — a gas meter stops runaway/infinite guests.
//! * **Bounded authority** — a `Call` to a host id the guest was not granted traps;
//!   granted host functions are explicit closures supplied by the host, so the
//!   guest's whole world is exactly what the host chose to hand it.
//!
//! Pure, safe `no_std + alloc`, host-tested.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

/// The guest instruction set — a minimal but Turing-complete stack machine.
#[derive(Clone, Debug)]
pub enum Op {
    /// Push a constant.
    Const(i64),
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// Pop b, a; push (a == b) as 0/1.
    Eq,
    /// Pop b, a; push (a < b) as 0/1.
    Lt,
    /// Read local slot.
    GetLocal(usize),
    /// Pop value into local slot.
    SetLocal(usize),
    /// Pop addr; push memory[addr].
    Load,
    /// Pop value, addr; memory[addr] = value.
    Store,
    /// Unconditional jump to instruction index.
    Jump(usize),
    /// Pop cond; jump if zero.
    JumpIfZero(usize),
    /// Call host import `id` with `argc` args popped from the stack.
    Call { id: u32, argc: usize },
    /// Halt, leaving the top of stack as the result.
    Return,
}

/// Why a guest trapped — every failure is contained, never propagated to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    StackUnderflow,
    /// The operand stack would exceed [`MAX_STACK_DEPTH`]. Bounds the host heap a
    /// guest can drive the VM stack to consume, independent of the gas limit.
    StackOverflow,
    MemoryOutOfBounds,
    LocalOutOfBounds,
    DivideByZero,
    OutOfGas,
    BadJump,
    UngrantedHostCall,
    BadHostArity,
}

/// Hard ceiling on the operand-stack depth. A guest with a generous `gas_limit`
/// could otherwise push (e.g. `Const`/`LocalGet`/`Load`) without bound and drive
/// the host's backing `Vec<i64>` to exhaust memory; capping the depth keeps the
/// worst-case stack allocation at `MAX_STACK_DEPTH * 8` bytes (32 KiB).
pub const MAX_STACK_DEPTH: usize = 4096;

/// The callback behind a host import: pure, takes the popped args, returns a value.
pub type HostCallback = Box<dyn Fn(&[i64]) -> i64>;

/// A host import the sandbox may expose: an arity and a pure callback. The callback
/// is supplied by the host, so it is *by construction* the only authority the guest
/// can exercise.
pub struct HostFn {
    pub id: u32,
    pub arity: usize,
    pub func: HostCallback,
}

/// A sandboxed guest module + the world it is allowed to touch.
pub struct Sandbox {
    code: Vec<Op>,
    memory: Vec<i64>,
    locals: Vec<i64>,
    /// Host imports, keyed by id; only these are callable.
    imports: Vec<HostFn>,
    /// Maximum instructions before [`Trap::OutOfGas`].
    gas_limit: u64,
}

impl Sandbox {
    pub fn new(code: Vec<Op>, mem_cells: usize, locals: usize, gas_limit: u64) -> Sandbox {
        Sandbox {
            code,
            memory: vec![0i64; mem_cells],
            locals: vec![0i64; locals],
            imports: Vec::new(),
            gas_limit,
        }
    }

    /// Grant the guest access to a host function. Without an explicit grant the
    /// corresponding `Call` traps — the guest's authority is allow-list only.
    pub fn grant(&mut self, import: HostFn) {
        self.imports.push(import);
    }

    /// Seed a local before execution (calling convention / arguments).
    pub fn set_local(&mut self, slot: usize, v: i64) -> bool {
        if let Some(s) = self.locals.get_mut(slot) {
            *s = v;
            true
        } else {
            false
        }
    }

    pub fn memory(&self) -> &[i64] {
        &self.memory
    }

    /// Run to a `Return` (or until gas runs out). Returns the top-of-stack result
    /// on success, or the trap that contained the guest.
    pub fn run(&mut self) -> Result<i64, Trap> {
        let mut stack: Vec<i64> = Vec::new();
        let mut pc = 0usize;
        let mut gas = 0u64;

        macro_rules! pop {
            () => {
                stack.pop().ok_or(Trap::StackUnderflow)?
            };
        }

        // Every push goes through this so the operand stack can never grow past
        // MAX_STACK_DEPTH, no matter how much gas the guest is granted.
        macro_rules! push {
            ($v:expr) => {{
                if stack.len() >= MAX_STACK_DEPTH {
                    return Err(Trap::StackOverflow);
                }
                stack.push($v);
            }};
        }

        while pc < self.code.len() {
            gas += 1;
            if gas > self.gas_limit {
                return Err(Trap::OutOfGas);
            }
            match &self.code[pc] {
                Op::Const(v) => push!(*v),
                Op::Add => {
                    let b = pop!();
                    let a = pop!();
                    push!(a.wrapping_add(b));
                }
                Op::Sub => {
                    let b = pop!();
                    let a = pop!();
                    push!(a.wrapping_sub(b));
                }
                Op::Mul => {
                    let b = pop!();
                    let a = pop!();
                    push!(a.wrapping_mul(b));
                }
                Op::Div => {
                    let b = pop!();
                    let a = pop!();
                    if b == 0 {
                        return Err(Trap::DivideByZero);
                    }
                    push!(a.wrapping_div(b));
                }
                Op::Rem => {
                    let b = pop!();
                    let a = pop!();
                    if b == 0 {
                        return Err(Trap::DivideByZero);
                    }
                    push!(a.wrapping_rem(b));
                }
                Op::Eq => {
                    let b = pop!();
                    let a = pop!();
                    push!((a == b) as i64);
                }
                Op::Lt => {
                    let b = pop!();
                    let a = pop!();
                    push!((a < b) as i64);
                }
                Op::GetLocal(i) => {
                    let v = *self.locals.get(*i).ok_or(Trap::LocalOutOfBounds)?;
                    push!(v);
                }
                Op::SetLocal(i) => {
                    let v = pop!();
                    *self.locals.get_mut(*i).ok_or(Trap::LocalOutOfBounds)? = v;
                }
                Op::Load => {
                    let addr = pop!();
                    let cell = self.cell(addr)?;
                    push!(self.memory[cell]);
                }
                Op::Store => {
                    let v = pop!();
                    let addr = pop!();
                    let cell = self.cell(addr)?;
                    self.memory[cell] = v;
                }
                Op::Jump(target) => {
                    if *target > self.code.len() {
                        return Err(Trap::BadJump);
                    }
                    pc = *target;
                    continue;
                }
                Op::JumpIfZero(target) => {
                    let c = pop!();
                    if c == 0 {
                        if *target > self.code.len() {
                            return Err(Trap::BadJump);
                        }
                        pc = *target;
                        continue;
                    }
                }
                Op::Call { id, argc } => {
                    let import = self
                        .imports
                        .iter()
                        .find(|h| h.id == *id)
                        .ok_or(Trap::UngrantedHostCall)?;
                    if import.arity != *argc || stack.len() < *argc {
                        return Err(Trap::BadHostArity);
                    }
                    let args = stack.split_off(stack.len() - *argc);
                    let result = (import.func)(&args);
                    push!(result);
                }
                Op::Return => return Ok(stack.pop().unwrap_or(0)),
            }
            pc += 1;
        }
        Ok(stack.pop().unwrap_or(0))
    }

    /// Bounds-check a guest address into a memory cell index.
    fn cell(&self, addr: i64) -> Result<usize, Trap> {
        if addr < 0 || addr as usize >= self.memory.len() {
            Err(Trap::MemoryOutOfBounds)
        } else {
            Ok(addr as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_program() {
        // (3 + 4) * 5 = 35
        let mut s = Sandbox::new(
            vec![Op::Const(3), Op::Const(4), Op::Add, Op::Const(5), Op::Mul, Op::Return],
            0,
            0,
            1000,
        );
        assert_eq!(s.run(), Ok(35));
    }

    #[test]
    fn loop_sums_one_to_n() {
        // local0 = n (input), local1 = acc; while n>0 { acc += n; n -= 1 }
        let code = vec![
            // 0: if local0 == 0 goto end(13)
            Op::GetLocal(0),
            Op::Const(0),
            Op::Eq,
            Op::JumpIfZero(5), // if (n==0)==0 i.e. n!=0, continue to body
            Op::Jump(13),      // n==0 → end
            // 5: acc += n
            Op::GetLocal(1),
            Op::GetLocal(0),
            Op::Add,
            Op::SetLocal(1),
            // 9: n -= 1
            Op::GetLocal(0),
            Op::Const(1),
            Op::Sub,
            Op::SetLocal(0),
            // 13 was the loop-back target conceptually; jump back to 0
        ];
        // Rebuild with an explicit back-edge and end.
        let mut code = code;
        code.push(Op::Jump(0)); // 13: loop back
        code.push(Op::GetLocal(1)); // 14: push acc
        code.push(Op::Return); // 15
        // Fix the n==0 exit target (index 4) to point at 14.
        let mut code2 = code.clone();
        code2[4] = Op::Jump(14);

        let mut s = Sandbox::new(code2, 0, 2, 100_000);
        s.set_local(0, 10); // n = 10 → 55
        assert_eq!(s.run(), Ok(55));
    }

    #[test]
    fn memory_is_bounded() {
        // Store then load within bounds works.
        let mut ok = Sandbox::new(
            vec![Op::Const(0), Op::Const(99), Op::Store, Op::Const(0), Op::Load, Op::Return],
            4,
            0,
            1000,
        );
        assert_eq!(ok.run(), Ok(99));
        // Out-of-bounds store traps — cannot reach beyond its own memory.
        let mut oob = Sandbox::new(
            vec![Op::Const(9999), Op::Const(1), Op::Store, Op::Return],
            4,
            0,
            1000,
        );
        assert_eq!(oob.run(), Err(Trap::MemoryOutOfBounds));
    }

    #[test]
    fn infinite_loop_runs_out_of_gas() {
        let mut s = Sandbox::new(vec![Op::Jump(0)], 0, 0, 5_000);
        assert_eq!(s.run(), Err(Trap::OutOfGas));
    }

    #[test]
    fn divide_by_zero_traps() {
        let mut s = Sandbox::new(vec![Op::Const(1), Op::Const(0), Op::Div, Op::Return], 0, 0, 100);
        assert_eq!(s.run(), Err(Trap::DivideByZero));
    }

    #[test]
    fn ungranted_host_call_traps() {
        let mut s = Sandbox::new(vec![Op::Const(7), Op::Call { id: 1, argc: 1 }, Op::Return], 0, 0, 100);
        // No grant → the guest cannot call out.
        assert_eq!(s.run(), Err(Trap::UngrantedHostCall));
    }

    #[test]
    fn granted_host_call_succeeds_with_only_that_authority() {
        let mut s = Sandbox::new(
            vec![Op::Const(20), Op::Const(22), Op::Call { id: 7, argc: 2 }, Op::Return],
            0,
            0,
            100,
        );
        // Grant exactly one capability: an "add" host import.
        s.grant(HostFn { id: 7, arity: 2, func: Box::new(|args| args[0] + args[1]) });
        assert_eq!(s.run(), Ok(42));
        // A different id is still ungranted.
        let mut s2 = Sandbox::new(vec![Op::Call { id: 8, argc: 0 }, Op::Return], 0, 0, 100);
        assert_eq!(s2.run(), Err(Trap::UngrantedHostCall));
    }

    #[test]
    fn stack_underflow_traps() {
        let mut s = Sandbox::new(vec![Op::Add, Op::Return], 0, 0, 100);
        assert_eq!(s.run(), Err(Trap::StackUnderflow));
    }

    #[test]
    fn unbounded_pushes_trap_with_stack_overflow_not_oom() {
        // A guest that loops pushing constants forever. With a gas limit far larger
        // than MAX_STACK_DEPTH, the *only* thing standing between the guest and an
        // unbounded host-heap allocation is the depth cap — so it must trap as
        // StackOverflow, never OutOfGas and never by exhausting memory.
        // Program: Const(1); Jump(0)  — push one value per loop, never popping.
        let mut s = Sandbox::new(
            vec![Op::Const(1), Op::Jump(0)],
            0,
            0,
            (MAX_STACK_DEPTH as u64) * 1000,
        );
        assert_eq!(s.run(), Err(Trap::StackOverflow));
    }

    #[test]
    fn pushing_exactly_to_the_cap_is_allowed() {
        // MAX_STACK_DEPTH pushes followed by Return must succeed: the bound rejects
        // the *overflowing* push, not the last legal one.
        let mut code: Vec<Op> = (0..MAX_STACK_DEPTH).map(|_| Op::Const(7)).collect();
        code.push(Op::Return);
        let mut s = Sandbox::new(code, 0, 0, (MAX_STACK_DEPTH as u64) + 8);
        assert_eq!(s.run(), Ok(7));
    }
}

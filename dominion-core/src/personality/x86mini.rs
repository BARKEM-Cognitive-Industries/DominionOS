//! A compact but **broadly-complete** x86-64 interpreter — the execution engine past
//! the "foreign machine code is modeled" boundary (see
//! `docs/architecture/capability-shim-and-foreign-compat.md` §3.2 and
//! [`super::applaunch`]).
//!
//! It decodes and executes real x86-64 instruction bytes against a register file with
//! status flags, a **capability-bounded memory slice** (the sandbox region — the CPU
//! can never address outside it; an out-of-bounds access is a clean [`CpuFault`], not
//! UB), a downward-growing stack inside that slice, and a `syscall` trap that lands on
//! a [`SyscallSink`] (the capability shim). This is what an in-sandbox JIT would
//! ultimately do; an interpreter keeps it small and obviously-safe.
//!
//! ## Coverage
//!
//! Operand sizes 16/32/64-bit (`0x66` prefix, REX.W), full ModRM + SIB + displacement
//! memory addressing, and the instruction families real compiled code is built from:
//!
//! * **data movement** — `mov` (r/m↔r, imm), `movzx`/`movsx`, `lea`, `xchg`;
//! * **arithmetic / logic** — `add adc sub sbb and or xor cmp test` (r/m,r · r,r/m ·
//!   r/m,imm · rAX,imm), `inc dec neg not`, `imul`/`mul`, `div`/`idiv`, `cdq`/`cqo`;
//! * **shifts** — `shl shr sar` (by 1, by imm8, by CL);
//! * **control flow** — `jmp`/`call` (rel8/rel32/indirect), `ret`, the full `jcc`
//!   family (rel8 + rel32), `setcc`, `cmovcc`, `push`/`pop` (reg/imm/r-m), `leave`;
//! * **system** — `syscall` (traps to the shim), `nop`.
//!
//! Every step is metered (budget), every memory/stack access is bounds-checked, and an
//! unknown opcode halts cleanly with [`CpuFault::BadOpcode`]. Pure, safe `no_std`,
//! extensively host-tested. Growing it further is adding match arms — the framework
//! (sizes, ModRM, flags, stack, traps) is complete.

// Register indices in x86-64 encoding order (REX extends to r8..r15 = 8..15).
pub const RAX: usize = 0;
pub const RCX: usize = 1;
pub const RDX: usize = 2;
pub const RBX: usize = 3;
pub const RSP: usize = 4;
pub const RBP: usize = 5;
pub const RSI: usize = 6;
pub const RDI: usize = 7;

/// What a guest `syscall` is dispatched to. `nr` is `rax`; `args` are the System V
/// syscall registers `[rdi, rsi, rdx, r10, r8, r9]`; `mem` is the sandbox memory so
/// pointer args can be read/written in bounds. The return value is placed in `rax`.
pub trait SyscallSink {
    fn syscall(&mut self, nr: u64, args: [u64; 6], mem: &mut [u8]) -> i64;
    /// True once the guest has requested exit, so the CPU halts after the syscall.
    fn exited(&self) -> bool {
        false
    }
}

/// Why execution stopped abnormally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuFault {
    /// An opcode outside the supported subset (halts cleanly, never undefined).
    BadOpcode(u8),
    /// Instruction fetch ran off the end of the code.
    CodeOverrun,
    /// A data/stack access fell outside the sandbox memory slice.
    MemFault(u64),
    /// Integer divide by zero (or quotient overflow).
    DivideByZero,
    /// The step budget was exhausted (runaway guest).
    OutOfGas,
}

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Halt {
    /// A top-level `ret` executed (stack empty) — normal end of a flat code run.
    Ret,
    /// The syscall sink reported the guest exited.
    Exited,
}

/// A decoded operand: a register slot or an effective memory address into the sandbox.
#[derive(Clone, Copy)]
enum Operand {
    Reg(usize),
    Mem(u64),
}

/// x86-64 core: 16 general registers, RIP, and the status flags real code branches on.
pub struct Cpu {
    pub regs: [u64; 16],
    pub rip: usize,
    cf: bool,
    zf: bool,
    sf: bool,
    of: bool,
    pf: bool,
    steps: u64,
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

#[inline]
fn mask(size: u8) -> u64 {
    match size {
        1 => 0xFF,
        2 => 0xFFFF,
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}
#[inline]
fn sign_bit(size: u8) -> u64 {
    1u64 << (size as u64 * 8 - 1)
}
/// Sign-extend the low `size` bytes of `v` to a full u64.
#[inline]
fn sext(v: u64, size: u8) -> u64 {
    let m = mask(size);
    let v = v & m;
    if v & sign_bit(size) != 0 {
        v | !m
    } else {
        v
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu { regs: [0; 16], rip: 0, cf: false, zf: false, sf: false, of: false, pf: false, steps: 0 }
    }

    /// Steps executed in the last `run`.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    // ── register / memory access by operand size ──
    fn reg_read(&self, i: usize, size: u8) -> u64 {
        self.regs[i] & mask(size)
    }
    fn reg_write(&mut self, i: usize, size: u8, val: u64) {
        match size {
            8 => self.regs[i] = val,
            4 => self.regs[i] = val & 0xFFFF_FFFF, // 32-bit writes zero-extend
            2 => self.regs[i] = (self.regs[i] & !0xFFFF) | (val & 0xFFFF),
            _ => self.regs[i] = (self.regs[i] & !0xFF) | (val & 0xFF),
        }
    }
    fn mem_read(&self, addr: u64, size: u8, mem: &[u8]) -> Result<u64, CpuFault> {
        let a = addr as usize;
        let end = a.checked_add(size as usize).ok_or(CpuFault::MemFault(addr))?;
        if end > mem.len() {
            return Err(CpuFault::MemFault(addr));
        }
        let mut v = 0u64;
        for k in 0..size as usize {
            v |= (mem[a + k] as u64) << (8 * k);
        }
        Ok(v)
    }
    fn mem_write(&self, addr: u64, size: u8, val: u64, mem: &mut [u8]) -> Result<(), CpuFault> {
        let a = addr as usize;
        let end = a.checked_add(size as usize).ok_or(CpuFault::MemFault(addr))?;
        if end > mem.len() {
            return Err(CpuFault::MemFault(addr));
        }
        for k in 0..size as usize {
            mem[a + k] = (val >> (8 * k)) as u8;
        }
        Ok(())
    }
    fn op_read(&self, op: Operand, size: u8, mem: &[u8]) -> Result<u64, CpuFault> {
        match op {
            Operand::Reg(i) => Ok(self.reg_read(i, size)),
            Operand::Mem(a) => self.mem_read(a, size, mem),
        }
    }
    fn op_write(&mut self, op: Operand, size: u8, val: u64, mem: &mut [u8]) -> Result<(), CpuFault> {
        match op {
            Operand::Reg(i) => {
                self.reg_write(i, size, val);
                Ok(())
            }
            Operand::Mem(a) => self.mem_write(a, size, val, mem),
        }
    }

    // ── stack (downward-growing inside the sandbox memory) ──
    fn push(&mut self, val: u64, mem: &mut [u8]) -> Result<(), CpuFault> {
        let sp = self.regs[RSP].checked_sub(8).ok_or(CpuFault::MemFault(self.regs[RSP]))?;
        self.mem_write(sp, 8, val, mem)?;
        self.regs[RSP] = sp;
        Ok(())
    }
    fn pop(&mut self, mem: &[u8]) -> Result<u64, CpuFault> {
        let sp = self.regs[RSP];
        let v = self.mem_read(sp, 8, mem)?;
        self.regs[RSP] = sp + 8;
        Ok(v)
    }

    // ── flags ──
    fn set_szp(&mut self, res: u64, size: u8) {
        self.zf = (res & mask(size)) == 0;
        self.sf = (res & sign_bit(size)) != 0;
        self.pf = (res as u8).count_ones() % 2 == 0;
    }

    /// Arithmetic/logic core. Returns the masked result and sets all flags. `Cmp`/
    /// `Test` callers simply discard the result.
    fn alu(&mut self, kind: AluKind, a: u64, b: u64, size: u8) -> u64 {
        let m = mask(size);
        let sb = sign_bit(size);
        let (a, b) = (a & m, b & m);
        let (res, cf, of) = match kind {
            AluKind::Add => {
                let r = a.wrapping_add(b) & m;
                (r, r < a, (!(a ^ b) & (a ^ r) & sb) != 0)
            }
            AluKind::Adc => {
                let c = self.cf as u64;
                let r = a.wrapping_add(b).wrapping_add(c) & m;
                let cf = if c == 1 { r <= a } else { r < a };
                (r, cf, (!(a ^ b) & (a ^ r) & sb) != 0)
            }
            AluKind::Sub | AluKind::Cmp => {
                let r = a.wrapping_sub(b) & m;
                (r, a < b, ((a ^ b) & (a ^ r) & sb) != 0)
            }
            AluKind::Sbb => {
                let c = self.cf as u64;
                let r = a.wrapping_sub(b).wrapping_sub(c) & m;
                let cf = if c == 1 { a <= b } else { a < b };
                (r, cf, ((a ^ b) & (a ^ r) & sb) != 0)
            }
            AluKind::And | AluKind::Test => (a & b & m, false, false),
            AluKind::Or => ((a | b) & m, false, false),
            AluKind::Xor => ((a ^ b) & m, false, false),
        };
        self.cf = cf;
        self.of = of;
        self.set_szp(res, size);
        res
    }

    /// Evaluate a condition-code (the low nibble of a `jcc`/`setcc`/`cmovcc` opcode).
    fn cond(&self, cc: u8) -> bool {
        match cc & 0xF {
            0x0 => self.of,                       // O
            0x1 => !self.of,                      // NO
            0x2 => self.cf,                       // B/C
            0x3 => !self.cf,                      // AE/NC
            0x4 => self.zf,                       // E/Z
            0x5 => !self.zf,                      // NE/NZ
            0x6 => self.cf || self.zf,            // BE
            0x7 => !self.cf && !self.zf,          // A
            0x8 => self.sf,                       // S
            0x9 => !self.sf,                      // NS
            0xA => self.pf,                       // P
            0xB => !self.pf,                      // NP
            0xC => self.sf != self.of,            // L
            0xD => self.sf == self.of,            // GE
            0xE => self.zf || (self.sf != self.of), // LE
            _ => !self.zf && (self.sf == self.of), // G
        }
    }

    fn imm(code: &[u8], at: usize, size: u8) -> Result<u64, CpuFault> {
        let mut v = 0u64;
        for k in 0..size as usize {
            v |= (*code.get(at + k).ok_or(CpuFault::CodeOverrun)? as u64) << (8 * k);
        }
        Ok(v)
    }

    /// Decode a ModRM byte (and any SIB/displacement) at `ip` into the r/m operand and
    /// the reg field, returning the new ip. Computes effective addresses from the live
    /// registers; RIP-relative and SIB forms are handled.
    fn modrm(
        &self,
        code: &[u8],
        ip: usize,
        rex_r: bool,
        rex_x: bool,
        rex_b: bool,
    ) -> Result<(Operand, usize, usize), CpuFault> {
        let m = *code.get(ip).ok_or(CpuFault::CodeOverrun)?;
        let md = m >> 6;
        let reg = ((m >> 3) & 7) as usize + if rex_r { 8 } else { 0 };
        let rm = (m & 7) as usize;
        let mut ip = ip + 1;
        if md == 3 {
            return Ok((Operand::Reg(rm + if rex_b { 8 } else { 0 }), reg, ip));
        }
        // Memory form. Resolve base/index/displacement.
        let mut addr: i64;
        if rm == 4 {
            // SIB byte.
            let sib = *code.get(ip).ok_or(CpuFault::CodeOverrun)?;
            ip += 1;
            let scale = 1i64 << (sib >> 6);
            let index = ((sib >> 3) & 7) as usize + if rex_x { 8 } else { 0 };
            let base = (sib & 7) as usize + if rex_b { 8 } else { 0 };
            let index_val = if ((sib >> 3) & 7) == 4 && !rex_x {
                0
            } else {
                self.regs[index] as i64
            };
            let base_val = if (sib & 7) == 5 && md == 0 {
                let d = sext(Self::imm(code, ip, 4)?, 4) as i64;
                ip += 4;
                d
            } else {
                self.regs[base] as i64
            };
            addr = base_val + index_val * scale;
        } else if rm == 5 && md == 0 {
            // RIP-relative (modeled as an absolute disp32 into the data slice).
            let d = sext(Self::imm(code, ip, 4)?, 4) as i64;
            ip += 4;
            addr = d;
        } else {
            addr = self.regs[rm + if rex_b { 8 } else { 0 }] as i64;
        }
        match md {
            1 => {
                let d = sext(Self::imm(code, ip, 1)?, 1) as i64;
                ip += 1;
                addr = addr.wrapping_add(d);
            }
            2 => {
                let d = sext(Self::imm(code, ip, 4)?, 4) as i64;
                ip += 4;
                addr = addr.wrapping_add(d);
            }
            _ => {}
        }
        Ok((Operand::Mem(addr as u64), reg, ip))
    }

    /// Execute `code` against the sandbox `mem`, dispatching `syscall` to `sink`, until
    /// a top-level `ret`, an exit, a fault, or budget exhaustion. The stack starts at
    /// the top of `mem` (set RSP yourself to override).
    pub fn run(
        &mut self,
        code: &[u8],
        mem: &mut [u8],
        sink: &mut dyn SyscallSink,
        budget: u64,
    ) -> Result<Halt, CpuFault> {
        self.steps = 0;
        if self.regs[RSP] == 0 {
            self.regs[RSP] = mem.len() as u64;
        }
        loop {
            if self.steps >= budget {
                return Err(CpuFault::OutOfGas);
            }
            self.steps += 1;
            match self.step(code, mem, sink)? {
                Some(h) => return Ok(h),
                None => {}
            }
        }
    }

    /// Execute one instruction. Returns `Some(halt)` if the program ended.
    fn step(
        &mut self,
        code: &[u8],
        mem: &mut [u8],
        sink: &mut dyn SyscallSink,
    ) -> Result<Option<Halt>, CpuFault> {
        let mut ip = self.rip;
        // Prefixes: operand-size (0x66) and REX (0x40..0x4F, must be last prefix).
        let mut size: u8 = 4;
        let (mut rex_w, mut rex_r, mut rex_x, mut rex_b) = (false, false, false, false);
        loop {
            let b = *code.get(ip).ok_or(CpuFault::CodeOverrun)?;
            if b == 0x66 {
                size = 2;
                ip += 1;
            } else if (b & 0xF0) == 0x40 {
                rex_w = b & 8 != 0;
                rex_r = b & 4 != 0;
                rex_x = b & 2 != 0;
                rex_b = b & 1 != 0;
                ip += 1;
                break; // REX is the last prefix
            } else {
                break;
            }
        }
        if rex_w {
            size = 8;
        }
        let op = *code.get(ip).ok_or(CpuFault::CodeOverrun)?;
        ip += 1;

        macro_rules! commit {
            () => {{
                self.rip = ip;
                return Ok(None);
            }};
        }

        // ── ALU group: r/m,r (+1) · r,r/m (+3) · rAX,imm (+5) ──
        if op < 0x40 && (op & 7) == 1 {
            let kind = ALU_KINDS[(op >> 3) as usize];
            let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
            ip = nip;
            let a = self.op_read(rm, size, mem)?;
            let b = self.reg_read(reg, size);
            let r = self.alu(kind, a, b, size);
            if kind != AluKind::Cmp {
                self.op_write(rm, size, r, mem)?;
            }
            commit!();
        }
        if op < 0x40 && (op & 7) == 3 {
            let kind = ALU_KINDS[(op >> 3) as usize];
            let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
            ip = nip;
            let a = self.reg_read(reg, size);
            let b = self.op_read(rm, size, mem)?;
            let r = self.alu(kind, a, b, size);
            if kind != AluKind::Cmp {
                self.reg_write(reg, size, r);
            }
            commit!();
        }
        if op < 0x40 && (op & 7) == 5 {
            let kind = ALU_KINDS[(op >> 3) as usize];
            let isz = if size == 8 { 4 } else { size };
            let imm = sext(Self::imm(code, ip, isz)?, isz);
            ip += isz as usize;
            let a = self.reg_read(RAX, size);
            let r = self.alu(kind, a, imm, size);
            if kind != AluKind::Cmp {
                self.reg_write(RAX, size, r);
            }
            commit!();
        }

        match op {
            // mov r/m, r
            0x89 => {
                let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let v = self.reg_read(reg, size);
                self.op_write(rm, size, v, mem)?;
                commit!();
            }
            // mov r, r/m
            0x8B => {
                let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let v = self.op_read(rm, size, mem)?;
                self.reg_write(reg, size, v);
                commit!();
            }
            // lea r, m
            0x8D => {
                let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                if let Operand::Mem(a) = rm {
                    self.reg_write(reg, size, a);
                }
                commit!();
            }
            // xchg r/m, r
            0x87 => {
                let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let a = self.op_read(rm, size, mem)?;
                let b = self.reg_read(reg, size);
                self.op_write(rm, size, b, mem)?;
                self.reg_write(reg, size, a);
                commit!();
            }
            // mov r/m, imm32 (sign-extended)
            0xC7 => {
                let (rm, _reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let isz = if size == 8 { 4 } else { size };
                let imm = sext(Self::imm(code, ip, isz)?, isz);
                ip += isz as usize;
                self.op_write(rm, size, imm, mem)?;
                commit!();
            }
            // mov r, imm (B8+r): full operand-size immediate
            0xB8..=0xBF => {
                let reg = (op - 0xB8) as usize + if rex_b { 8 } else { 0 };
                let imm = Self::imm(code, ip, size)?;
                ip += size as usize;
                self.reg_write(reg, size, imm);
                commit!();
            }
            // group 1: ALU r/m, imm  (0x81 imm32, 0x83 imm8 sign-extended)
            0x81 | 0x83 => {
                let (rm, regfield, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let isz = if op == 0x83 { 1 } else if size == 8 { 4 } else { size };
                let imm = sext(Self::imm(code, ip, isz)?, isz);
                ip += isz as usize;
                let kind = ALU_KINDS[regfield & 7];
                let a = self.op_read(rm, size, mem)?;
                let r = self.alu(kind, a, imm, size);
                if kind != AluKind::Cmp {
                    self.op_write(rm, size, r, mem)?;
                }
                commit!();
            }
            // test r/m, r
            0x85 => {
                let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let a = self.op_read(rm, size, mem)?;
                let b = self.reg_read(reg, size);
                self.alu(AluKind::Test, a, b, size);
                commit!();
            }
            // group 3: F7 /digit (test/not/neg/mul/imul/div/idiv)
            0xF7 => {
                let (rm, regfield, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let a = self.op_read(rm, size, mem)?;
                match regfield & 7 {
                    0 => {
                        let isz = if size == 8 { 4 } else { size };
                        let imm = sext(Self::imm(code, ip, isz)?, isz);
                        ip += isz as usize;
                        self.alu(AluKind::Test, a, imm, size);
                    }
                    2 => {
                        let r = !a & mask(size);
                        self.op_write(rm, size, r, mem)?;
                    }
                    3 => {
                        let r = self.alu(AluKind::Sub, 0, a, size);
                        self.op_write(rm, size, r, mem)?;
                    }
                    4 => self.mul_unsigned(a, size),
                    5 => self.imul_one(a, size),
                    6 => self.div_unsigned(a, size)?,
                    7 => self.idiv_signed(a, size)?,
                    _ => return Err(CpuFault::BadOpcode(op)),
                }
                commit!();
            }
            // group 5: FF /digit (inc/dec/call/jmp/push)
            0xFF => {
                let (rm, regfield, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                match regfield & 7 {
                    0 => {
                        let a = self.op_read(rm, size, mem)?;
                        let cf = self.cf;
                        let r = self.alu(AluKind::Add, a, 1, size);
                        self.cf = cf; // inc preserves CF
                        self.op_write(rm, size, r, mem)?;
                    }
                    1 => {
                        let a = self.op_read(rm, size, mem)?;
                        let cf = self.cf;
                        let r = self.alu(AluKind::Sub, a, 1, size);
                        self.cf = cf; // dec preserves CF
                        self.op_write(rm, size, r, mem)?;
                    }
                    2 => {
                        // call r/m (indirect)
                        let target = self.op_read(rm, 8, mem)?;
                        self.rip = ip;
                        self.push(ip as u64, mem)?;
                        self.rip = target as usize;
                        return Ok(None);
                    }
                    4 => {
                        // jmp r/m (indirect)
                        let target = self.op_read(rm, 8, mem)?;
                        self.rip = target as usize;
                        return Ok(None);
                    }
                    6 => {
                        let v = self.op_read(rm, 8, mem)?;
                        self.push(v, mem)?;
                    }
                    _ => return Err(CpuFault::BadOpcode(op)),
                }
                commit!();
            }
            // group 2 shifts: C1 (imm8), D1 (by 1), D3 (by CL)
            0xC1 | 0xD1 | 0xD3 => {
                let (rm, regfield, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let count = match op {
                    0xC1 => {
                        let c = Self::imm(code, ip, 1)?;
                        ip += 1;
                        c
                    }
                    0xD1 => 1,
                    _ => self.reg_read(RCX, 1),
                } & if size == 8 { 0x3F } else { 0x1F };
                let a = self.op_read(rm, size, mem)?;
                let r = self.shift((regfield & 7) as u8, a, count as u32, size);
                self.op_write(rm, size, r, mem)?;
                commit!();
            }
            // push reg
            0x50..=0x57 => {
                let reg = (op - 0x50) as usize + if rex_b { 8 } else { 0 };
                let v = self.regs[reg];
                self.push(v, mem)?;
                commit!();
            }
            // pop reg
            0x58..=0x5F => {
                let reg = (op - 0x58) as usize + if rex_b { 8 } else { 0 };
                let v = self.pop(mem)?;
                self.regs[reg] = v;
                commit!();
            }
            // push imm32 / imm8
            0x68 => {
                let v = sext(Self::imm(code, ip, 4)?, 4);
                ip += 4;
                self.push(v, mem)?;
                commit!();
            }
            0x6A => {
                let v = sext(Self::imm(code, ip, 1)?, 1);
                ip += 1;
                self.push(v, mem)?;
                commit!();
            }
            // imul r, r/m, imm  (0x69 imm32, 0x6B imm8)
            0x69 | 0x6B => {
                let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                ip = nip;
                let isz = if op == 0x6B { 1 } else if size == 8 { 4 } else { size };
                let imm = sext(Self::imm(code, ip, isz)?, isz);
                ip += isz as usize;
                let a = self.op_read(rm, size, mem)?;
                let r = (sext(a, size) as i64).wrapping_mul(imm as i64) as u64 & mask(size);
                self.reg_write(reg, size, r);
                commit!();
            }
            // cdq / cqo: sign-extend rAX into rDX
            0x99 => {
                let neg = self.reg_read(RAX, size) & sign_bit(size) != 0;
                self.reg_write(RDX, size, if neg { mask(size) } else { 0 });
                commit!();
            }
            // leave: mov rsp,rbp ; pop rbp
            0xC9 => {
                self.regs[RSP] = self.regs[RBP];
                let v = self.pop(mem)?;
                self.regs[RBP] = v;
                commit!();
            }
            // jmp rel8 / rel32
            0xEB => {
                let d = sext(Self::imm(code, ip, 1)?, 1) as i64;
                ip += 1;
                self.rip = (ip as i64 + d) as usize;
                return Ok(None);
            }
            0xE9 => {
                let d = sext(Self::imm(code, ip, 4)?, 4) as i64;
                ip += 4;
                self.rip = (ip as i64 + d) as usize;
                return Ok(None);
            }
            // jcc rel8
            0x70..=0x7F => {
                let d = sext(Self::imm(code, ip, 1)?, 1) as i64;
                ip += 1;
                if self.cond(op - 0x70) {
                    self.rip = (ip as i64 + d) as usize;
                } else {
                    self.rip = ip;
                }
                return Ok(None);
            }
            // call rel32
            0xE8 => {
                let d = sext(Self::imm(code, ip, 4)?, 4) as i64;
                ip += 4;
                self.push(ip as u64, mem)?;
                self.rip = (ip as i64 + d) as usize;
                return Ok(None);
            }
            // ret (top-level ret with empty stack halts)
            0xC3 => {
                if self.regs[RSP] as usize >= mem.len() {
                    self.rip = ip;
                    return Ok(Some(Halt::Ret));
                }
                let r = self.pop(mem)?;
                self.rip = r as usize;
                return Ok(None);
            }
            // ret imm16
            0xC2 => {
                let imm = Self::imm(code, ip, 2)?;
                if self.regs[RSP] as usize >= mem.len() {
                    return Ok(Some(Halt::Ret));
                }
                let r = self.pop(mem)?;
                self.regs[RSP] += imm;
                self.rip = r as usize;
                return Ok(None);
            }
            0x90 => commit!(), // nop
            // two-byte opcodes
            0x0F => {
                let op2 = *code.get(ip).ok_or(CpuFault::CodeOverrun)?;
                ip += 1;
                match op2 {
                    0x05 => {
                        // syscall
                        self.rip = ip;
                        let args = [
                            self.regs[RDI],
                            self.regs[RSI],
                            self.regs[RDX],
                            self.regs[10],
                            self.regs[8],
                            self.regs[9],
                        ];
                        let ret = sink.syscall(self.regs[RAX], args, mem);
                        self.regs[RAX] = ret as u64;
                        if sink.exited() {
                            return Ok(Some(Halt::Exited));
                        }
                        return Ok(None);
                    }
                    // jcc rel32
                    0x80..=0x8F => {
                        let d = sext(Self::imm(code, ip, 4)?, 4) as i64;
                        ip += 4;
                        if self.cond(op2 - 0x80) {
                            self.rip = (ip as i64 + d) as usize;
                        } else {
                            self.rip = ip;
                        }
                        return Ok(None);
                    }
                    // setcc r/m8
                    0x90..=0x9F => {
                        let (rm, _reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                        ip = nip;
                        let v = if self.cond(op2 - 0x90) { 1 } else { 0 };
                        self.op_write(rm, 1, v, mem)?;
                        commit!();
                    }
                    // cmovcc r, r/m
                    0x40..=0x4F => {
                        let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                        ip = nip;
                        let v = self.op_read(rm, size, mem)?;
                        if self.cond(op2 - 0x40) {
                            self.reg_write(reg, size, v);
                        }
                        commit!();
                    }
                    // imul r, r/m
                    0xAF => {
                        let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                        ip = nip;
                        let a = self.reg_read(reg, size);
                        let b = self.op_read(rm, size, mem)?;
                        let r = (sext(a, size) as i64).wrapping_mul(sext(b, size) as i64) as u64
                            & mask(size);
                        self.reg_write(reg, size, r);
                        commit!();
                    }
                    // movzx r, r/m8 (B6) / r/m16 (B7)
                    0xB6 | 0xB7 => {
                        let src_sz = if op2 == 0xB6 { 1 } else { 2 };
                        let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                        ip = nip;
                        let v = self.op_read(rm, src_sz, mem)?;
                        self.reg_write(reg, size, v);
                        commit!();
                    }
                    // movsx r, r/m8 (BE) / r/m16 (BF)
                    0xBE | 0xBF => {
                        let src_sz = if op2 == 0xBE { 1 } else { 2 };
                        let (rm, reg, nip) = self.modrm(code, ip, rex_r, rex_x, rex_b)?;
                        ip = nip;
                        let v = sext(self.op_read(rm, src_sz, mem)?, src_sz);
                        self.reg_write(reg, size, v & mask(size));
                        commit!();
                    }
                    other => return Err(CpuFault::BadOpcode(other)),
                }
            }
            other => Err(CpuFault::BadOpcode(other)),
        }
    }

    fn shift(&mut self, kind: u8, a: u64, count: u32, size: u8) -> u64 {
        if count == 0 {
            return a & mask(size);
        }
        let m = mask(size);
        let a = a & m;
        let r = match kind {
            4 => {
                // shl
                self.cf = if count <= size as u32 * 8 {
                    (a >> (size as u32 * 8 - count)) & 1 != 0
                } else {
                    false
                };
                (a << count) & m
            }
            5 => {
                // shr (logical)
                self.cf = (a >> (count - 1)) & 1 != 0;
                a >> count
            }
            7 => {
                // sar (arithmetic)
                self.cf = (a >> (count - 1)) & 1 != 0;
                // Arithmetic right shift: sign-extend to i64 first so negative
                // values sign-fill for all sizes (incl. size 8, where sext is a no-op).
                ((sext(a, size) as i64) >> count) as u64 & m
            }
            _ => a,
        };
        self.set_szp(r, size);
        r
    }

    fn mul_unsigned(&mut self, src: u64, size: u8) {
        let a = self.reg_read(RAX, size) as u128;
        let p = a * (src & mask(size)) as u128;
        let m = mask(size) as u128;
        self.reg_write(RAX, size, (p & m) as u64);
        let high = (p >> (size as u32 * 8)) & m;
        self.reg_write(RDX, size, high as u64);
        self.cf = high != 0;
        self.of = high != 0;
    }

    fn imul_one(&mut self, src: u64, size: u8) {
        let a = sext(self.reg_read(RAX, size), size) as i128;
        let p = a * sext(src, size) as i128;
        let m = mask(size) as u128;
        self.reg_write(RAX, size, (p as u128 & m) as u64);
        let high = ((p as u128) >> (size as u32 * 8)) & m;
        self.reg_write(RDX, size, high as u64);
        // CF/OF set if the full result doesn't fit in the low half (sign-extended).
        let fits = (p >> (size as i32 * 8 - 1)) == 0 || (p >> (size as i32 * 8 - 1)) == -1;
        self.cf = !fits;
        self.of = !fits;
    }

    fn div_unsigned(&mut self, src: u64, size: u8) -> Result<(), CpuFault> {
        let d = (src & mask(size)) as u128;
        if d == 0 {
            return Err(CpuFault::DivideByZero);
        }
        let dividend = ((self.reg_read(RDX, size) as u128) << (size as u32 * 8))
            | self.reg_read(RAX, size) as u128;
        let q = dividend / d;
        let r = dividend % d;
        if q > mask(size) as u128 {
            return Err(CpuFault::DivideByZero); // quotient overflow → #DE
        }
        self.reg_write(RAX, size, q as u64);
        self.reg_write(RDX, size, r as u64);
        Ok(())
    }

    fn idiv_signed(&mut self, src: u64, size: u8) -> Result<(), CpuFault> {
        let d = sext(src, size) as i128;
        if d == 0 {
            return Err(CpuFault::DivideByZero);
        }
        let dividend = (((self.reg_read(RDX, size) as u128) << (size as u32 * 8))
            | self.reg_read(RAX, size) as u128) as i128;
        // Reconstruct the signed dividend across the RDX:RAX pair.
        let bits = size as u32 * 16;
        let dividend = if bits < 128 && (dividend >> (bits - 1)) & 1 == 1 {
            dividend | (-1i128 << bits)
        } else {
            dividend
        };
        let q = dividend / d;
        let r = dividend % d;
        // Quotient must fit the signed destination width or #DE (e.g. INT_MIN / -1).
        let dbits = size as u32 * 8;
        let qmin = -(1i128 << (dbits - 1));
        let qmax = (1i128 << (dbits - 1)) - 1;
        if q < qmin || q > qmax {
            return Err(CpuFault::DivideByZero); // quotient overflow → #DE
        }
        self.reg_write(RAX, size, q as u64 & mask(size));
        self.reg_write(RDX, size, r as u64 & mask(size));
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AluKind {
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    Cmp,
    Test,
}

/// The 8 primary ALU ops indexed by the opcode's `(op>>3)&7` / group-1 reg field.
const ALU_KINDS: [AluKind; 8] = [
    AluKind::Add,
    AluKind::Or,
    AluKind::Adc,
    AluKind::Sbb,
    AluKind::And,
    AluKind::Sub,
    AluKind::Xor,
    AluKind::Cmp,
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[derive(Default)]
    struct TestSink {
        written: Vec<u8>,
        exited: bool,
        exit_code: i64,
    }
    impl SyscallSink for TestSink {
        fn syscall(&mut self, nr: u64, args: [u64; 6], mem: &mut [u8]) -> i64 {
            match nr {
                1 => {
                    let (buf, len) = (args[1] as usize, args[2] as usize);
                    let end = buf.saturating_add(len).min(mem.len());
                    if buf <= end {
                        self.written.extend_from_slice(&mem[buf..end]);
                    }
                    (end - buf) as i64
                }
                60 => {
                    self.exited = true;
                    self.exit_code = args[0] as i64;
                    0
                }
                _ => -1,
            }
        }
        fn exited(&self) -> bool {
            self.exited
        }
    }

    fn run(code: &[u8], mem_len: usize) -> (Cpu, TestSink) {
        let mut cpu = Cpu::new();
        let mut sink = TestSink::default();
        let mut mem = alloc::vec![0u8; mem_len];
        cpu.run(code, &mut mem, &mut sink, 100_000).unwrap();
        (cpu, sink)
    }
    fn run_mem(code: &[u8], mem: &mut [u8]) -> Cpu {
        let mut cpu = Cpu::new();
        let mut sink = TestSink::default();
        cpu.run(code, mem, &mut sink, 100_000).unwrap();
        cpu
    }

    fn mov_imm(reg: u8, imm: u64) -> Vec<u8> {
        let mut v = alloc::vec![0x48, 0xB8 + reg];
        v.extend_from_slice(&imm.to_le_bytes());
        v
    }

    #[test]
    fn mov_add_sub_and_ret() {
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 10));
        c.extend_from_slice(&mov_imm(RCX as u8, 32));
        c.extend_from_slice(&[0x48, 0x01, 0xC8]); // add rax, rcx
        c.extend_from_slice(&mov_imm(RBX as u8, 2));
        c.extend_from_slice(&[0x48, 0x29, 0xD8]); // sub rax, rbx
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 40);
    }

    #[test]
    fn logic_and_immediates() {
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 0b1100));
        c.extend_from_slice(&[0x48, 0x83, 0xE0, 0b0110]); // and rax, 6
        c.extend_from_slice(&[0x48, 0x83, 0xC8, 0b0001]); // or  rax, 1
        c.extend_from_slice(&[0x48, 0x35]); // xor rax, imm32
        c.extend_from_slice(&5u32.to_le_bytes());
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        // (0b1100 & 0b0110)=0b0100; |1 =0b0101=5; ^5 = 0
        assert_eq!(cpu.regs[RAX], 0);
    }

    #[test]
    fn cmp_sets_flags_and_jcc_branches() {
        // if (5 < 9) rax=111 else rax=222   (signed: jl)
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 5));
        c.extend_from_slice(&mov_imm(RCX as u8, 9));
        c.extend_from_slice(&[0x48, 0x39, 0xC8]); // cmp rax, rcx
        // jl +taken (rel8). Compute distance to the "rax=111" block.
        // layout after this: [0x7C, rel8]
        let else_block = mov_imm(RAX as u8, 222); // 10 bytes
        let then_block = mov_imm(RAX as u8, 111); // 10 bytes
        let jmp_over = 2usize; // EB rel8 skipping then-block
        // jl rel8 → skip the else block + the jmp_over (jump into then-block)
        let rel_to_then = (else_block.len() + jmp_over) as i8;
        c.extend_from_slice(&[0x7C, rel_to_then as u8]);
        c.extend_from_slice(&else_block);
        c.extend_from_slice(&[0xEB, then_block.len() as u8]); // jmp over then
        c.extend_from_slice(&then_block);
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 111);
    }

    #[test]
    fn loop_with_dec_and_jnz_sums() {
        // rax=0; rcx=5; do { rax += rcx; rcx-- } while (rcx != 0)  → 5+4+3+2+1 = 15
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 0)); // 10
        c.extend_from_slice(&mov_imm(RCX as u8, 5)); // 10
        let loop_start = c.len();
        c.extend_from_slice(&[0x48, 0x01, 0xC8]); // add rax, rcx  (3)
        c.extend_from_slice(&[0x48, 0xFF, 0xC9]); // dec rcx       (3)
        // jnz loop_start (rel8, negative)
        let after_jcc = c.len() + 2;
        let rel = (loop_start as i64 - after_jcc as i64) as i8;
        c.extend_from_slice(&[0x75, rel as u8]);
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 15);
        assert_eq!(cpu.regs[RCX], 0);
    }

    #[test]
    fn push_pop_round_trips_through_the_stack() {
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 0xCAFE));
        c.push(0x50); // push rax
        c.extend_from_slice(&mov_imm(RAX as u8, 0)); // clobber
        c.push(0x5B); // pop rbx
        c.push(0xC3);
        let (cpu, _) = run(&c, 256);
        assert_eq!(cpu.regs[RBX], 0xCAFE);
    }

    #[test]
    fn call_ret_invokes_a_subroutine() {
        // main: mov rax,7; call addr; (after) ret.  sub: add rax,rax; ret  → 14
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 7)); // 10
        // call rel32 → target = end (the subroutine) placed after main's ret
        let call_site = c.len();
        c.extend_from_slice(&[0xE8, 0, 0, 0, 0]); // placeholder
        c.push(0xC3); // main ret (top-level → halts)
        let sub_addr = c.len();
        c.extend_from_slice(&[0x48, 0x01, 0xC0]); // add rax, rax
        c.push(0xC3); // sub ret
        // patch call rel32
        let after_call = call_site + 5;
        let rel = (sub_addr as i64 - after_call as i64) as i32;
        c[call_site + 1..call_site + 5].copy_from_slice(&rel.to_le_bytes());
        let (cpu, _) = run(&c, 256);
        assert_eq!(cpu.regs[RAX], 14);
    }

    #[test]
    fn memory_load_store_via_modrm() {
        // mov rbx, 0x1122; mov [0], rbx (store); mov rax,[0] (load) → rax=0x1122
        let mut mem = alloc::vec![0u8; 64];
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RBX as u8, 0x1122));
        // mov [rax+0], rbx — but rax=0 initially; use disp32 form via SIB? Simpler:
        // mov rsi, 0 ; mov [rsi], rbx ; mov rax, [rsi]
        c.extend_from_slice(&mov_imm(RSI as u8, 8));
        c.extend_from_slice(&[0x48, 0x89, 0x1E]); // mov [rsi], rbx  (modrm 00 011 110)
        c.extend_from_slice(&[0x48, 0x8B, 0x06]); // mov rax, [rsi]  (modrm 00 000 110)
        c.push(0xC3);
        // Avoid stack collision: set RSP high already (mem top). 8..16 holds our value.
        let cpu = run_mem(&c, &mut mem);
        assert_eq!(cpu.regs[RAX], 0x1122);
        assert_eq!(&mem[8..10], &[0x22, 0x11]);
    }

    #[test]
    fn imul_and_shifts() {
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 6));
        c.extend_from_slice(&mov_imm(RCX as u8, 7));
        c.extend_from_slice(&[0x48, 0x0F, 0xAF, 0xC1]); // imul rax, rcx → 42
        c.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x01]); // shl rax, 1 → 84
        c.extend_from_slice(&[0x48, 0xC1, 0xE8, 0x02]); // shr rax, 2 → 21
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 21);
    }

    #[test]
    fn unsigned_div_computes_quotient_and_remainder() {
        // rax=100, rdx=0, rcx=7 ; div rcx → rax=14, rdx=2
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 100));
        c.extend_from_slice(&mov_imm(RDX as u8, 0));
        c.extend_from_slice(&mov_imm(RCX as u8, 7));
        c.extend_from_slice(&[0x48, 0xF7, 0xF1]); // div rcx
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 14);
        assert_eq!(cpu.regs[RDX], 2);
    }

    #[test]
    fn setcc_and_movzx() {
        // rax=3; cmp rax,3; sete cl; movzx rax, cl → 1
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 3));
        c.extend_from_slice(&[0x48, 0x83, 0xF8, 0x03]); // cmp rax, 3
        c.extend_from_slice(&[0x0F, 0x94, 0xC1]); // sete cl
        c.extend_from_slice(&[0x48, 0x0F, 0xB6, 0xC1]); // movzx rax, cl
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 1);
    }

    #[test]
    fn real_machine_code_write_then_exit() {
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 1)); // write
        c.extend_from_slice(&mov_imm(RDI as u8, 1));
        c.extend_from_slice(&mov_imm(RSI as u8, 0));
        c.extend_from_slice(&mov_imm(RDX as u8, 2));
        c.extend_from_slice(&[0x0F, 0x05]);
        c.extend_from_slice(&mov_imm(RAX as u8, 60)); // exit
        c.extend_from_slice(&mov_imm(RDI as u8, 0));
        c.extend_from_slice(&[0x0F, 0x05]);
        let mut cpu = Cpu::new();
        let mut sink = TestSink::default();
        let mut mem = [0u8; 64];
        mem[0] = b'h';
        mem[1] = b'i';
        let halt = cpu.run(&c, &mut mem, &mut sink, 1000).unwrap();
        assert_eq!(halt, Halt::Exited);
        assert_eq!(sink.written, b"hi".to_vec());
    }

    #[test]
    fn unknown_opcode_and_budget_and_memfault() {
        let mut cpu = Cpu::new();
        let mut sink = TestSink::default();
        let mut mem = [0u8; 16];
        assert_eq!(cpu.run(&[0xD6], &mut mem, &mut sink, 10), Err(CpuFault::BadOpcode(0xD6)));
        let sled = alloc::vec![0x90u8; 100];
        let mut cpu2 = Cpu::new();
        assert_eq!(cpu2.run(&sled, &mut mem, &mut sink, 5), Err(CpuFault::OutOfGas));
        // divide by zero faults cleanly
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 5));
        c.extend_from_slice(&mov_imm(RDX as u8, 0));
        c.extend_from_slice(&mov_imm(RCX as u8, 0));
        c.extend_from_slice(&[0x48, 0xF7, 0xF1]); // div rcx
        let mut cpu3 = Cpu::new();
        let mut mem3 = [0u8; 64];
        assert_eq!(cpu3.run(&c, &mut mem3, &mut sink, 100), Err(CpuFault::DivideByZero));
    }

    #[test]
    fn fib_via_loop_is_correct() {
        // Iterative fib(10)=55: rax=0(prev), rcx=1(cur), rdx=10(n)
        // loop: rbx=rax+rcx; rax=rcx; rcx=rbx; dec rdx; jnz loop ; result in rax
        let mut c = Vec::new();
        c.extend_from_slice(&mov_imm(RAX as u8, 0));
        c.extend_from_slice(&mov_imm(RCX as u8, 1));
        c.extend_from_slice(&mov_imm(RDX as u8, 10));
        let start = c.len();
        c.extend_from_slice(&[0x48, 0x89, 0xC3]); // mov rbx, rax
        c.extend_from_slice(&[0x48, 0x01, 0xCB]); // add rbx, rcx
        c.extend_from_slice(&[0x48, 0x89, 0xC8]); // mov rax, rcx
        c.extend_from_slice(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
        c.extend_from_slice(&[0x48, 0xFF, 0xCA]); // dec rdx
        let after = c.len() + 2;
        c.extend_from_slice(&[0x75, (start as i64 - after as i64) as i8 as u8]); // jnz start
        c.push(0xC3);
        let (cpu, _) = run(&c, 64);
        assert_eq!(cpu.regs[RAX], 55);
    }
}

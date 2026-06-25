//! Interrupt handling: the IDT plus the 8259 PIC chain.
//!
//! The kernel needs three things working: CPU exceptions must be caught (so a
//! fault produces a readable message instead of a reboot loop), the timer must
//! tick, and keystrokes must reach the terminal. Everything else stays evicted
//! to user space in the spirit of the microkernel design (Stage 1).

use crate::{gdt, println};
use lazy_static::lazy_static;
use pic8259::ChainedPics;
use spin::Mutex;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

pub const PIC_1_OFFSET: u8 = 32;
pub const PIC_2_OFFSET: u8 = PIC_1_OFFSET + 8;

pub static PICS: Mutex<ChainedPics> =
    Mutex::new(unsafe { ChainedPics::new(PIC_1_OFFSET, PIC_2_OFFSET) });

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum InterruptIndex {
    Timer = PIC_1_OFFSET,
    Keyboard,
    /// PS/2 mouse — IRQ12, on the slave PIC (offset + 12).
    Mouse = PIC_2_OFFSET + 4,
}

impl InterruptIndex {
    fn as_u8(self) -> u8 {
        self as u8
    }
    fn as_usize(self) -> usize {
        usize::from(self.as_u8())
    }
}

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        unsafe {
            idt.double_fault
                .set_handler_fn(double_fault_handler)
                .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
        }
        idt.page_fault.set_handler_fn(page_fault_handler);
        idt.general_protection_fault
            .set_handler_fn(general_protection_fault_handler);
        idt.invalid_opcode.set_handler_fn(invalid_opcode_handler);
        idt[InterruptIndex::Timer.as_usize()].set_handler_fn(timer_interrupt_handler);
        idt[InterruptIndex::Keyboard.as_usize()].set_handler_fn(keyboard_interrupt_handler);
        idt[InterruptIndex::Mouse.as_usize()].set_handler_fn(mouse_interrupt_handler);
        idt
    };
}

pub fn init_idt() {
    IDT.load();
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    println!("[cpu] breakpoint trap\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    panic!("DOUBLE FAULT — unrecoverable\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn page_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: x86_64::structures::idt::PageFaultErrorCode,
) {
    use x86_64::registers::control::Cr2;
    // DominionOS is a single-address-space OS: untrusted code never runs against raw
    // page tables — it executes inside the capability-bounded `wasm::Sandbox`, whose
    // memory accesses are bounds-checked in software and trap without ever reaching
    // the CPU fault path. A hardware page fault therefore signals a genuine bug in
    // *kernel* code, not a recoverable user fault to localise to one domain, so we
    // report and halt. Mirror to serial as well as the framebuffer so headless/CI
    // runs (which have no display) still capture the fault.
    crate::serial_println!("[cpu] PAGE FAULT");
    crate::serial_println!("  accessed address: {:?}", Cr2::read());
    crate::serial_println!("  error code: {:?}", error_code);
    crate::serial_println!("{:#?}", stack_frame);
    println!("[cpu] PAGE FAULT");
    println!("  accessed address: {:?}", Cr2::read());
    println!("  error code: {:?}", error_code);
    println!("{:#?}", stack_frame);
    crate::hlt_loop();
}

extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    use core::sync::atomic::{AtomicBool, Ordering};
    static FIRST: AtomicBool = AtomicBool::new(true);
    if FIRST.swap(false, Ordering::Relaxed) {
        crate::serial_println!("[irq] timer handler reached at vector 0x20 (PIC remap OK)");
    }
    crate::keyboard::tick();
    unsafe {
        PICS.lock()
            .notify_end_of_interrupt(InterruptIndex::Timer.as_u8());
    }
}

extern "x86-interrupt" fn general_protection_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    crate::serial_println!("[cpu] GENERAL PROTECTION FAULT (error {:#x})", error_code);
    crate::serial_println!("  RIP = {:?}", stack_frame.instruction_pointer);
    crate::serial_println!("  RSP = {:?}", stack_frame.stack_pointer);
    crate::serial_println!("  CS  = {:#x}  SS = {:#x}", stack_frame.code_segment, stack_frame.stack_segment);
    println!("[cpu] GENERAL PROTECTION FAULT (error {:#x})", error_code);
    println!("{:#?}", stack_frame);
    crate::hlt_loop();
}

extern "x86-interrupt" fn invalid_opcode_handler(stack_frame: InterruptStackFrame) {
    crate::serial_println!("[cpu] INVALID OPCODE\n{:#?}", stack_frame);
    crate::hlt_loop();
}

extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    use x86_64::instructions::port::Port;
    // Only consume a byte if the controller's output buffer is actually full (bit 0)
    // and the byte is from the keyboard, not the aux/mouse port (bit 5 clear). The
    // desktop loop also drains the i8042 via `keyboard::poll()` (inside
    // `without_interrupts`); if the poller wins the race and empties the buffer, this
    // IRQ is still pending and fires with nothing to read. Reading 0x60 unconditionally
    // then returns the *stale* last byte and double-injects the keystroke — the source
    // of the duplicate-character bug. Gating on the status register makes the ISR and
    // the poller mutually safe (mirrors `mouse_interrupt_handler`).
    let status: u8 = unsafe { Port::new(0x64).read() };
    if status & 0x21 == 0x01 {
        let scancode: u8 = unsafe { Port::new(0x60).read() };
        crate::keyboard::add_scancode(scancode);
    }
    unsafe {
        PICS.lock()
            .notify_end_of_interrupt(InterruptIndex::Keyboard.as_u8());
    }
}

extern "x86-interrupt" fn mouse_interrupt_handler(_stack_frame: InterruptStackFrame) {
    use x86_64::instructions::port::Port;
    // Only consume the byte if it actually came from the auxiliary (mouse) port.
    let status: u8 = unsafe { Port::new(0x64).read() };
    if status & 0x21 == 0x21 {
        let byte: u8 = unsafe { Port::new(0x60).read() };
        crate::mouse::add_byte(byte);
    }
    unsafe {
        PICS.lock()
            .notify_end_of_interrupt(InterruptIndex::Mouse.as_u8());
    }
}

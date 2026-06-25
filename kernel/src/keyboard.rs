//! Keyboard input and a millisecond-ish tick counter.
//!
//! The PS/2 keyboard ISR ([`add_scancode`]) decodes scancodes into ASCII and
//! pushes them into a small interrupt-safe ring buffer. The terminal pulls from
//! it via the blocking [`read_char`] / [`read_line`], halting between keys so
//! the VM stays idle. All buffer access from the foreground happens inside
//! `without_interrupts`, so the ISR can never deadlock against the reader.

use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use lazy_static::lazy_static;
use pc_keyboard::{layouts, DecodedKey, HandleControl, KeyCode, KeyState, Keyboard, ScancodeSet1};
use spin::Mutex;
use x86_64::instructions::interrupts;

const QUEUE_CAP: usize = 256;

struct KeyQueue {
    buf: [u8; QUEUE_CAP],
    head: usize,
    tail: usize,
}

impl KeyQueue {
    const fn new() -> KeyQueue {
        KeyQueue {
            buf: [0; QUEUE_CAP],
            head: 0,
            tail: 0,
        }
    }

    fn push(&mut self, byte: u8) {
        let next = (self.tail + 1) % QUEUE_CAP;
        if next != self.head {
            self.buf[self.tail] = byte;
            self.tail = next;
        }
        // queue full → drop the keystroke (acceptable for a terminal)
    }

    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail {
            None
        } else {
            let b = self.buf[self.head];
            self.head = (self.head + 1) % QUEUE_CAP;
            Some(b)
        }
    }
}

static QUEUE: Mutex<KeyQueue> = Mutex::new(KeyQueue::new());
static TICKS: AtomicU64 = AtomicU64::new(0);
/// Live modifier state, tracked from key-down/up events so Shift+motion (selection)
/// and Ctrl+letter (clipboard) chords can be encoded for the text engine.
static SHIFT: AtomicBool = AtomicBool::new(false);
static CTRL: AtomicBool = AtomicBool::new(false);

lazy_static! {
    static ref KEYBOARD: Mutex<Keyboard<layouts::Us104Key, ScancodeSet1>> = Mutex::new(
        Keyboard::new(ScancodeSet1::new(), layouts::Us104Key, HandleControl::Ignore)
    );
}

/// Called from the timer ISR; provides a coarse monotonic clock.
pub fn tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
}

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

/// Called from the keyboard ISR with a raw scancode. Decodes and enqueues any
/// resulting printable character or recognised control byte.
pub fn add_scancode(scancode: u8) {
    use dominion_core::keys;
    let mut kb = KEYBOARD.lock();
    if let Ok(Some(key_event)) = kb.add_byte(scancode) {
        // Track modifier state from the raw event before it is decoded.
        let pressed = !matches!(key_event.state, KeyState::Up);
        match key_event.code {
            KeyCode::LShift | KeyCode::RShift => SHIFT.store(pressed, Ordering::Relaxed),
            KeyCode::LControl | KeyCode::RControl => CTRL.store(pressed, Ordering::Relaxed),
            _ => {}
        }
        let shift = SHIFT.load(Ordering::Relaxed);
        let ctrl = CTRL.load(Ordering::Relaxed);
        if let Some(key) = kb.process_keyevent(key_event) {
            match key {
                DecodedKey::Unicode(c) => {
                    // Ctrl+letter → clipboard / select-all chords (the keyboard ignores
                    // Ctrl for layout, so we encode the chord here from our own flag).
                    if ctrl {
                        let b = match c.to_ascii_lowercase() {
                            'c' => Some(keys::COPY),
                            'x' => Some(keys::CUT),
                            'v' => Some(keys::PASTE),
                            'a' => Some(keys::SELECT_ALL),
                            _ => None,
                        };
                        if let Some(b) = b {
                            QUEUE.lock().push(b);
                        }
                        return; // swallow other Ctrl-combos
                    }
                    let b = c as u32;
                    if b < 0x80 {
                        QUEUE.lock().push(b as u8);
                    }
                }
                // Navigation keys arrive as RawKey events. Map them to the control bytes
                // the global text engine understands (see `dominion_core::keys`); with Shift
                // held they become the selection-extending variants.
                DecodedKey::RawKey(code) => {
                    let b = match code {
                        KeyCode::ArrowLeft => Some(if shift { keys::SEL_LEFT } else { keys::ARROW_LEFT }),
                        KeyCode::ArrowRight => Some(if shift { keys::SEL_RIGHT } else { keys::ARROW_RIGHT }),
                        KeyCode::ArrowUp => Some(if shift { keys::SEL_UP } else { keys::ARROW_UP }),
                        KeyCode::ArrowDown => Some(if shift { keys::SEL_DOWN } else { keys::ARROW_DOWN }),
                        KeyCode::Home => Some(if shift { keys::SEL_HOME } else { keys::HOME }),
                        KeyCode::End => Some(if shift { keys::SEL_END } else { keys::END }),
                        KeyCode::Delete => Some(keys::DELETE),
                        _ => None,
                    };
                    if let Some(b) = b {
                        QUEUE.lock().push(b);
                    }
                }
            }
        }
    }
}

/// Poll the i8042 controller **directly** and dispatch any pending bytes — keyboard
/// bytes to the key decoder, aux (mouse) bytes to the mouse driver. This is what makes
/// input work on hardware where the legacy 8259 PIC never delivers IRQ1/IRQ12 — common
/// under UEFI, where the firmware routes interrupts through the APIC. Bounded so a wedged
/// controller can't spin forever. Harmless when interrupts *do* work: the ISR usually
/// drains port 0x60 first, so this finds the buffer empty and no-ops.
///
/// Requires the firmware's "Legacy USB Support" (i8042 emulation) for a USB keyboard to
/// appear here; with it off and no PS/2 port, there is nothing to poll.
pub fn poll() {
    use x86_64::instructions::port::Port;
    // When the native xHCI USB-HID driver is active the keyboard reaches us through
    // usbhid::poll() below. If the firmware also has "Legacy USB Support" turned on it
    // simultaneously emulates that same keyboard as a PS/2 device on the i8042, so
    // naively reading both paths would double-inject every keystroke. Skip keyboard
    // bytes from i8042 when USB-HID is present; still pass through aux-port bytes
    // (bit 5 of status set) because a PS/2 mouse doesn't have a USB-HID counterpart.
    let usb_kbd = crate::usbhid::present();
    // Disable interrupts for the check-then-read so the keyboard/mouse ISR (on machines
    // where IRQs *do* fire) can't interleave and steal the byte between our status read
    // and data read.
    interrupts::without_interrupts(|| {
        let mut status: Port<u8> = Port::new(0x64);
        let mut data: Port<u8> = Port::new(0x60);
        for _ in 0..32 {
            let st = unsafe { status.read() };
            if st & 0x01 == 0 {
                break; // output buffer empty — nothing pending
            }
            let byte = unsafe { data.read() };
            if st & 0x20 != 0 {
                crate::mouse::add_byte(byte); // bit5 set => byte came from the aux (mouse) port
            } else if !usb_kbd {
                // Only feed i8042 keyboard bytes when there is no USB-HID keyboard;
                // avoids double-injecting every keypress on UEFI with Legacy USB Support.
                add_scancode(byte);
            }
        }
    });
    // Also drain USB-HID input (keyboard + mouse over xHCI). Done outside the i8042
    // critical section: it feeds the same key queue / mouse state. No-op until the HID
    // driver is initialised, and on machines whose input is PS/2 (or firmware-emulated).
    crate::usbhid::poll();
}

/// Inject an already-decoded byte (printable ASCII or an `dominion_core::keys` control
/// byte) into the input queue. Used by the USB-HID keyboard driver, which decodes HID
/// boot-protocol usage codes itself rather than going through the PS/2 scancode decoder.
pub fn inject(byte: u8) {
    QUEUE.lock().push(byte);
}

/// Non-blocking pull of one byte. Polls the hardware first so it works with or without
/// a functioning keyboard IRQ.
pub fn try_read() -> Option<u8> {
    poll();
    interrupts::without_interrupts(|| QUEUE.lock().pop())
}

/// Blocking read of one byte. Busy-polls the i8042 directly rather than `hlt`-ing: on
/// hardware that delivers no interrupts, `hlt` would never wake (a permanent freeze), so
/// we spin instead. A recovery shell is not power-sensitive.
pub fn read_char() -> u8 {
    loop {
        poll();
        if let Some(b) = interrupts::without_interrupts(|| QUEUE.lock().pop()) {
            return b;
        }
        core::hint::spin_loop();
    }
}

/// Read a full line, echoing to the screen and handling Backspace. The trailing
/// newline is consumed but not included in the returned string.
pub fn read_line() -> String {
    use crate::{print, vga_buffer};
    let mut line = String::new();
    loop {
        let b = read_char();
        match b {
            b'\n' | b'\r' => {
                print!("\n");
                return line;
            }
            0x08 | 0x7f => {
                // Backspace / Delete
                if line.pop().is_some() {
                    vga_buffer::backspace();
                }
            }
            0x20..=0x7e => {
                line.push(b as char);
                print!("{}", b as char);
            }
            _ => {}
        }
    }
}

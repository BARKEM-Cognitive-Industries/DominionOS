//! PS/2 mouse driver — the desktop's pointing device.
//!
//! QEMU emulates a standard PS/2 mouse on the second 8042 port. We enable it,
//! turn on IRQ12 reporting, and decode the 3-byte movement packets in the
//! interrupt handler into an absolute cursor position. All controller I/O is
//! bounded by a spin timeout, so a missing or wedged mouse degrades to "no
//! pointer" instead of hanging the boot.

use spin::Mutex;
use x86_64::instructions::port::Port;

const DATA: u16 = 0x60;
const STATUS: u16 = 0x64;
const CMD: u16 = 0x64;

struct MouseState {
    x: i32,
    y: i32,
    max_x: i32,
    max_y: i32,
    left: bool,
    right: bool,
    packet: [u8; 3],
    phase: usize,
    /// Bumps on every processed packet so pollers can detect change.
    seq: u64,
    present: bool,
}

static MOUSE: Mutex<MouseState> = Mutex::new(MouseState {
    x: 100,
    y: 100,
    max_x: 1279,
    max_y: 719,
    left: false,
    right: false,
    packet: [0; 3],
    phase: 0,
    seq: 0,
    present: false,
});

/// A cheap snapshot of the pointer for the desktop event loop.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Pointer {
    pub x: i32,
    pub y: i32,
    pub left: bool,
    pub right: bool,
    pub seq: u64,
}

fn status() -> u8 {
    unsafe { Port::<u8>::new(STATUS).read() }
}

/// Wait until the controller can accept a write (input buffer empty).
fn wait_write() -> bool {
    for _ in 0..200_000 {
        if status() & 0x02 == 0 {
            return true;
        }
    }
    false
}

/// Wait until the controller has a byte to read (output buffer full).
fn wait_read() -> bool {
    for _ in 0..200_000 {
        if status() & 0x01 != 0 {
            return true;
        }
    }
    false
}

fn write_cmd(cmd: u8) {
    wait_write();
    unsafe { Port::<u8>::new(CMD).write(cmd) };
}

fn write_data(data: u8) {
    wait_write();
    unsafe { Port::<u8>::new(DATA).write(data) };
}

fn read_data() -> u8 {
    wait_read();
    unsafe { Port::<u8>::new(DATA).read() }
}

/// Send a command to the mouse itself and await its ACK (0xFA).
fn mouse_cmd(b: u8) -> bool {
    write_cmd(0xD4);
    write_data(b);
    read_data() == 0xFA
}

/// Unmask an IRQ line on the 8259 PIC pair (clearing its mask bit enables it).
fn unmask_irq(irq: u8) {
    unsafe {
        if irq < 8 {
            let mut p: Port<u8> = Port::new(0x21);
            let m = p.read();
            p.write(m & !(1 << irq));
        } else {
            let mut p: Port<u8> = Port::new(0xA1);
            let m = p.read();
            p.write(m & !(1 << (irq - 8)));
            // Ensure the master's cascade line (IRQ2) is also unmasked.
            let mut master: Port<u8> = Port::new(0x21);
            let mm = master.read();
            master.write(mm & !(1 << 2));
        }
    }
}

/// Initialise the PS/2 mouse and enable IRQ12 reporting. Returns `false` if the
/// controller/mouse did not respond (the desktop then runs pointer-less).
///
/// The whole sequence runs with interrupts **masked**. Otherwise the controller's
/// replies (config byte, command ACKs) arrive on port 0x60 and fire the *keyboard*
/// IRQ1, whose handler would steal them — corrupting our polled reads and injecting
/// the response bytes into the key decoder as spurious scancodes.
pub fn init(max_x: usize, max_y: usize) -> bool {
    use x86_64::instructions::interrupts;
    {
        let mut m = MOUSE.lock();
        m.max_x = max_x.saturating_sub(1) as i32;
        m.max_y = max_y.saturating_sub(1) as i32;
        m.x = (max_x / 2) as i32;
        m.y = (max_y / 2) as i32;
    }

    let present = interrupts::without_interrupts(|| {
        // Enable the auxiliary (mouse) PS/2 port.
        write_cmd(0xA8);

        // Read the controller config byte, enable IRQ12 (bit1) + the mouse clock
        // (clear bit5), write it back.
        write_cmd(0x20);
        let mut config = read_data();
        config |= 0x02;
        config &= !0x20;
        write_cmd(0x60);
        write_data(config);

        // Set defaults, then enable data reporting (streaming packets).
        let ok_defaults = mouse_cmd(0xF6);
        let ok_stream = mouse_cmd(0xF4);
        ok_defaults && ok_stream
    });

    MOUSE.lock().present = present;
    if present {
        unmask_irq(12);
    }
    present
}

/// Feed one byte from the IRQ12 handler into the packet assembler.
pub fn add_byte(byte: u8) {
    let mut m = MOUSE.lock();
    // The first byte of a packet always has bit3 set; use it to resync.
    if m.phase == 0 && byte & 0x08 == 0 {
        return;
    }
    let phase = m.phase;
    m.packet[phase] = byte;
    m.phase += 1;
    if m.phase < 3 {
        return;
    }
    m.phase = 0;

    let flags = m.packet[0];
    // Drop packets flagged as overflowed (avoids cursor teleporting).
    if flags & 0xC0 != 0 {
        return;
    }
    let dx = m.packet[1] as i8 as i32;
    let dy = m.packet[2] as i8 as i32;
    let (mx, my) = (m.max_x, m.max_y);
    m.x = (m.x + dx).clamp(0, mx);
    m.y = (m.y - dy).clamp(0, my); // screen-Y grows downward
    m.left = flags & 0x01 != 0;
    m.right = flags & 0x02 != 0;
    m.seq = m.seq.wrapping_add(1);
}

/// Apply one USB-HID boot-protocol mouse report: relative motion + button state. The
/// HID driver decodes the report and calls this; it mirrors what the PS/2 packet
/// assembler does, so the desktop sees the same `Pointer` either way. Marks the mouse
/// present so the desktop knows a pointer exists.
pub fn apply_hid(dx: i32, dy: i32, left: bool, right: bool) {
    let mut m = MOUSE.lock();
    m.present = true;
    let (mx, my) = (m.max_x, m.max_y);
    m.x = (m.x + dx).clamp(0, mx);
    m.y = (m.y + dy).clamp(0, my); // dy already screen-oriented (down = positive) by caller
    m.left = left;
    m.right = right;
    m.seq = m.seq.wrapping_add(1);
}

/// Poll the current pointer state.
pub fn poll() -> Pointer {
    let m = MOUSE.lock();
    Pointer { x: m.x, y: m.y, left: m.left, right: m.right, seq: m.seq }
}

pub fn present() -> bool {
    MOUSE.lock().present
}

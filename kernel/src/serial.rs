//! Serial port (COM1 / 16550 UART) output.
//!
//! Used for two things: mirroring the terminal to the host console when booting
//! with `-serial stdio`, and carrying test results out of the headless QEMU
//! instance during integration testing.

use lazy_static::lazy_static;
use spin::Mutex;
use uart_16550::SerialPort;

lazy_static! {
    pub static ref SERIAL1: Mutex<SerialPort> = {
        // 0x3F8 is the standard COM1 base address.
        let mut serial_port = unsafe { SerialPort::new(0x3F8) };
        serial_port.init();
        Mutex::new(serial_port)
    };
}

/// A fixed-capacity staging buffer for serial output.
///
/// `core::fmt` hands a formatted message to the writer as many small fragments
/// (one per `{}` arg and the literal pieces between them). Writing each fragment
/// straight to the UART means re-entering the byte-at-a-time `out`+status-poll
/// path for every fragment boundary. Instead we coalesce all the fragments of a
/// single message into this buffer and emit them in one contiguous burst, so the
/// per-message dispatch/locking overhead is paid once. The emitted bytes — and
/// their order — are byte-for-byte identical to the old direct path.
///
/// Capacity is sized for typical log/test lines; if a message overflows, the
/// buffer is flushed and refilled transparently, so output is never dropped or
/// reordered.
struct SerialStager<'a> {
    port: &'a mut SerialPort,
    buf: [u8; 256],
    len: usize,
}

impl SerialStager<'_> {
    #[inline]
    fn flush(&mut self) {
        // Tee every emitted chunk into the always-on boot-log ring (allocation-free,
        // captures boot/install/runtime output for post-mortem on bare metal) before
        // it goes out the UART. Identical bytes, same order.
        crate::bootlog::append(&self.buf[..self.len]);
        for &b in &self.buf[..self.len] {
            self.port.send(b);
        }
        self.len = 0;
    }
}

impl core::fmt::Write for SerialStager<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len == self.buf.len() {
                self.flush();
            }
            self.buf[self.len] = b;
            self.len += 1;
        }
        Ok(())
    }
}

#[doc(hidden)]
pub fn _print(args: core::fmt::Arguments) {
    use core::fmt::Write;
    use x86_64::instructions::interrupts;
    // Avoid deadlock if an interrupt handler also wants the port.
    interrupts::without_interrupts(|| {
        let mut port = SERIAL1.lock();
        let mut stager = SerialStager { port: &mut port, buf: [0; 256], len: 0 };
        stager
            .write_fmt(args)
            .expect("printing to serial failed");
        stager.flush();
    });
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        $crate::serial::_print(format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($fmt:expr) => ($crate::serial_print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::serial_print!(concat!($fmt, "\n"), $($arg)*));
}

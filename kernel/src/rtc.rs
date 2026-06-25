//! Real-time clock (CMOS) reader — the wall-clock source the TLS stack needs to
//! check certificate validity windows. The PIT/TSC give *monotonic* time since
//! boot; only the battery-backed CMOS RTC knows the actual date.
//!
//! This reads the standard MC146818-compatible registers over I/O ports
//! 0x70/0x71, handles BCD vs binary encoding and 12/24-hour mode, waits out an
//! in-progress update, and converts to Unix seconds (UTC).

use spin::Mutex;
use x86_64::instructions::port::Port;

/// Cached last result: `(raw_seconds_register, unix_timestamp)`. The CMOS clock
/// only advances once per second, so if the seconds register still reads the
/// same value the full timestamp is unchanged — we then return the cached Unix
/// time instead of re-reading and re-decoding all seven CMOS registers (which is
/// ~14 port operations per `read_raw`, repeated up to nine times for the
/// torn-read guard). A single seconds-register probe (2 port ops) validates the
/// cache. `raw_seconds == 0xFF` marks "no cached value yet".
static RTC_CACHE: Mutex<(u8, u64)> = Mutex::new((0xFF, 0));

fn cmos_read(reg: u8) -> u8 {
    // Port 0x70 selects the register (high bit also gates NMI; keep it clear),
    // port 0x71 returns the value.
    let mut addr = Port::<u8>::new(0x70);
    let mut data = Port::<u8>::new(0x71);
    unsafe {
        addr.write(reg & 0x7f);
        data.read()
    }
}

fn update_in_progress() -> bool {
    cmos_read(0x0a) & 0x80 != 0
}

fn bcd_to_bin(v: u8) -> u8 {
    (v & 0x0f) + ((v >> 4) * 10)
}

/// Days since the Unix epoch for a civil (proleptic Gregorian) date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// The current Unix timestamp (seconds, UTC), or 0 if the RTC looks insane.
pub fn unix_now() -> u64 {
    // Fast path: if the raw seconds register is unchanged since the last full
    // read, the wall clock has not advanced and the cached timestamp is exact.
    // This costs a single register select + read (2 port ops) instead of the
    // full multi-register torn-read loop below.
    let raw_sec_now = cmos_read(0x00);
    {
        let cache = RTC_CACHE.lock();
        if cache.0 == raw_sec_now && cache.0 != 0xFF {
            return cache.1;
        }
    }

    // Read twice with matching values to avoid a torn read across an update.
    let mut last = read_raw();
    for _ in 0..8 {
        let now = read_raw();
        if now == last {
            break;
        }
        last = now;
    }
    let (sec, min, hour, day, mon, year, raw_sec) = last;
    if !(1..=12).contains(&mon) || !(1..=31).contains(&day) || year < 2020 || year > 2200 {
        return 0;
    }
    let days = days_from_civil(year as i64, mon as i64, day as i64);
    let result = (days as u64) * 86400 + hour as u64 * 3600 + min as u64 * 60 + sec as u64;
    // Cache against the raw seconds register from the *same* consistent read, so a
    // later fast-path probe compares like for like (BCD or binary as the hardware
    // reports it).
    *RTC_CACHE.lock() = (raw_sec, result);
    result
}

/// Raw decoded (sec, min, hour, day, month, full-year) honouring BCD/12h flags,
/// plus the raw (un-decoded) seconds register used as the fast-path cache key.
fn read_raw() -> (u8, u8, u8, u8, u8, u16, u8) {
    while update_in_progress() {
        core::hint::spin_loop();
    }
    let sec = cmos_read(0x00);
    let min = cmos_read(0x02);
    let hour_raw = cmos_read(0x04);
    let day = cmos_read(0x07);
    let mon = cmos_read(0x08);
    let yr = cmos_read(0x09);
    let status_b = cmos_read(0x0b);

    let binary = status_b & 0x04 != 0;
    let pm = hour_raw & 0x80 != 0;
    let conv = |v: u8| if binary { v } else { bcd_to_bin(v) };

    let raw_sec = sec; // keep the un-decoded register for the cache key
    let sec = conv(sec);
    let min = conv(min);
    // Hours: mask the AM/PM flag before converting (it is not BCD).
    let mut hour = conv(hour_raw & 0x7f);
    if status_b & 0x02 == 0 {
        // 12-hour mode: 12am→0, 12pm→12, 1–11pm→ +12.
        if hour == 12 {
            hour = 0;
        }
        if pm {
            hour += 12;
        }
    }
    let day = conv(day);
    let mon = conv(mon);
    let year = 2000 + conv(yr) as u16;
    (sec, min, hour, day, mon, year, raw_sec)
}

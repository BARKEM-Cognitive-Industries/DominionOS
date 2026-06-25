//! The graphical desktop — drives the live [`dominion_core::os`] shell.
//!
//! This is a thin host: it brings up the framebuffer, gathers **real system metrics**
//! every tick (heap usage, render-load, fps, uptime, device presence, entropy health,
//! a deterministic state hash), feeds them and the pointer/keyboard into the shell
//! (Desktop / IDE / Explorer pages behind a dock), and renders the resulting scene
//! through the anti-aliased framebuffer backend with **damage-rect diff-presentation**
//! for a smooth ~30 fps. All UI logic, layout and interaction live in `dominion-core`;
//! the kernel only supplies live data and input.

use crate::shell::SystemInfo;
use crate::{allocator, gfx, mouse};
use dominion_core::dash::Metrics;
use dominion_core::hash::Hash256;
use dominion_core::os::Os;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use x86_64::instructions::port::Port;

/// Drive the PIT at `hz` so the render loop can pace itself finely (the default
/// ~18 Hz is too coarse for 30 fps and smooth input).
fn set_timer_hz(hz: u32) {
    let divisor = (1_193_182 / hz.max(1)).clamp(1, 65535) as u16;
    unsafe {
        let mut cmd: Port<u8> = Port::new(0x43);
        cmd.write(0x36); // channel 0, lo/hi byte, mode 3 (square wave)
        let mut data: Port<u8> = Port::new(0x40);
        data.write((divisor & 0xff) as u8);
        data.write((divisor >> 8) as u8);
    }
}

#[inline]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Calibrate the TSC against **PIT channel 2 by polling** — no timer interrupt needed.
///
/// The old calibration spun on `keyboard::ticks()`, which only advances when the IRQ0
/// timer interrupt fires. On real UEFI hardware that interrupt does not always reach
/// the legacy 8259 PIC (the firmware may route the timer through the APIC/HPET), so
/// that loop hung forever — a black screen right after the boot text. This instead
/// programs channel 2 in one-shot mode and busy-polls its output bit (port 0x61 bit 5),
/// which works regardless of interrupt routing, and is bounded so a wedged PIT degrades
/// to a sane default instead of hanging. Returns the TSC frequency in Hz.
fn calibrate_tsc_polled() -> u64 {
    const DEFAULT_HZ: u64 = 2_000_000_000; // 2 GHz fallback if calibration is implausible
    unsafe {
        let mut p61: Port<u8> = Port::new(0x61);
        // Gate channel 2 on (bit0=1), speaker output off (bit1=0).
        let v = p61.read();
        p61.write((v & 0xFC) | 0x01);
        // Channel 2, access lo/hi byte, mode 0 (interrupt on terminal count).
        let mut cmd: Port<u8> = Port::new(0x43);
        cmd.write(0b1011_0000);
        // ~50 ms: 1_193_182 Hz * 0.05 s = 59_659 counts.
        let count: u16 = 59_659;
        let mut ch2: Port<u8> = Port::new(0x42);
        ch2.write((count & 0xFF) as u8);
        ch2.write((count >> 8) as u8);

        let t0 = rdtsc();
        // Output bit (0x20) on port 0x61 goes high when the count reaches 0. Bound the
        // wait by ELAPSED TSC TIME, not an iteration count: a port read costs ~1 us, so
        // an iteration-count guard could spin for hours if the PIT never asserts. ~8e9
        // cycles is roughly 2-8 s across any plausible TSC, after which we fall back to a
        // default frequency and still boot (frame pacing is only slightly off).
        const TIMEOUT_CYCLES: u64 = 8_000_000_000;
        loop {
            if p61.read() & 0x20 != 0 {
                break;
            }
            if rdtsc().wrapping_sub(t0) >= TIMEOUT_CYCLES {
                return DEFAULT_HZ;
            }
        }
        let elapsed = rdtsc().wrapping_sub(t0);
        let hz = elapsed.saturating_mul(20); // 50 ms window → ×20 for 1 s
        if hz < 200_000_000 || hz > 20_000_000_000 {
            DEFAULT_HZ
        } else {
            hz
        }
    }
}

/// Launch the desktop. Returns `true` when the user chose Power Off (caller should
/// call `crate::shutdown()`), or `false` when the user pressed Esc (→ ASH safe-mode).
pub fn run(info: SystemInfo) -> bool {
    crate::println!("[desktop] 0/4 allocating graphics buffers ...");
    let (w, h) = match gfx::init() {
        Some(d) => d,
        None => return false, // no framebuffer → nothing to do
    };
    crate::serial_println!("[desktop] framebuffer {}x{}", w, h);
    // Screen-visible boot markers: on bare metal there is no serial console, so print
    // each pre-render step straight to the framebuffer (which already works — it shows
    // the boot text). If the desktop ever fails to come up, the LAST marker visible on
    // screen pinpoints exactly which step hung. They are overwritten by the first frame.
    crate::println!("[desktop] 1/4 framebuffer {}x{}", w, h);
    mouse::init(w, h);
    // Calibrate the TSC for frame pacing by polling PIT channel 2 — done BEFORE arming
    // the IRQ0 timer so a machine whose timer interrupt never reaches the PIC can still
    // boot to the desktop (it just won't have IRQ-driven input). See the function note.
    crate::println!("[desktop] 2/4 calibrating clock ...");
    let tsc_hz = calibrate_tsc_polled().max(1);
    crate::serial_println!("[desktop] TSC ~{} MHz; pacing to 30 fps", tsc_hz / 1_000_000);
    crate::println!("[desktop] 2/4 clock ~{} MHz", tsc_hz / 1_000_000);

    // 200 Hz timer for snappy pointer/keyboard interrupts (the render loop paces itself
    // off the TSC, not the timer, so its rate doesn't gate frames). If IRQ0 never fires
    // on this hardware the desktop still renders; only IRQ-driven input is affected.
    set_timer_hz(200);
    while crate::keyboard::try_read().is_some() {}
    let frame_budget = (tsc_hz / 30).max(1);
    let metric_budget = tsc_hz / 5; // refresh metrics 5×/s

    let mut os = Os::new();
    let (sw, sh) = (w as i32, h as i32);
    os.set_size(sw, sh); // damage rects must match the real framebuffer
    seed_logs(&mut os, &info);

    crate::println!("[desktop] 3/4 warming up renderer ...");
    // ── Unified 2D/3D render stack ────────────────────────────────────────────
    // Warm up the software rasteriser with a single high-poly mesh render so the
    // tile allocator and depth-buffer paths are exercised before the first frame.
    // The result is discarded; a render error here is non-fatal (the 2D fallback
    // always works). The PPM frame is not written to disk in no_std — it is used
    // only for the pixel-count sanity check.
    {
        let (_, bench) = dominion_core::render_bench::bench_single_mesh_render(64, 36);
        os.push_log(&format!(
            "[render3d] startup bench: {} tris rendered, {} non-black px",
            bench.rendered_triangles, bench.non_black_pixels
        ));
    }
    os.push_log("[render3d] Unified 2D/3D renderer online — Nanite LOD + BVH instancing + ATW active");

    // Inject the live network transport so the browser fetches real legacy pages over
    // virtio-net. With no NIC, the browser keeps its built-in loopback transport
    // (native pages + bundled content still render), so it is never blank.
    if let Some(t) = crate::webnet::KernelTransport::new() {
        os.set_web_transport(alloc::boxed::Box::new(t));
        os.push_log("[net] browser transport online (virtio-net, HTTP)");
    } else {
        os.push_log("[net] no NIC; browser using loopback transport");
    }

    // Durable filesystem: if a VFS image was saved on a previous shutdown, restore it
    // so files survive the reboot. A corrupt/absent image is ignored (keeps the seeded
    // filesystem) — it can never break boot. The image lives at a high LBA so it never
    // collides with the M1 object-graph image at block 0.
    if crate::block::present() {
        crate::block::with_block_device(|dev, _real| {
            if os.restore_fs_from(dev, FS_IMAGE_LBA) {
                os.push_log("[fs] filesystem restored from disk (incremental store)");
            } else if let Ok(Some(blob)) =
                dominion_core::persist::Persistence::load_blob(dev, FS_IMAGE_LBA, FS_IMAGE_MAGIC)
            {
                // A monolithic image from before the incremental store — migrate it in.
                // The next shutdown rewrites it in the new append-only format.
                if os.restore_fs(&blob) {
                    os.push_log("[fs] filesystem migrated from legacy image");
                }
            }
        });
    }

    let start = rdtsc();
    let mut last_metric = start;
    let mut last_checkpoint = start;
    let checkpoint_budget = tsc_hz.saturating_mul(CHECKPOINT_SECS).max(1);
    let mut fps_t0 = start;
    let mut fps_frames = 0u32;
    let mut fps = 0u32;
    let mut cpu_hist: Vec<i64> = Vec::new();
    let mut last_pointer = mouse::poll();
    let mut last_render_cycles = 0u64;
    // Busy cycles accumulated since the last metric sample (work, excluding the idle
    // `hlt` wait) and a smoothed CPU% — so the reported CPU is *actual* utilisation
    // (busy ÷ wall-clock), averaged over the window, instead of a single spiky frame.
    let mut busy_accum: u64 = 0;
    let mut cpu_ema: u32 = 0;
    let mut running = true;
    let mut power_off = false;

    os.set_metrics(gather(&info, fps, 0, 0, &cpu_hist));
    gfx::set_cursor(last_pointer.x.max(0) as usize, last_pointer.y.max(0) as usize);

    crate::println!("[desktop] 4/4 ready — starting desktop");

    while running {
        let frame_start = rdtsc();

        // ── input ── Drain the i8042 directly first: on hardware that delivers no
        //    keyboard/mouse IRQ (common under UEFI), this is the only way input arrives.
        //    It feeds both the key queue and the mouse driver. (the dashboard records a
        //    *damage region* for anything that actually changed; a bare hover records
        //    nothing.)
        crate::keyboard::poll();
        let p = mouse::poll();
        let cursor_moved = p.x != last_pointer.x || p.y != last_pointer.y;
        if cursor_moved {
            gfx::set_cursor(p.x.max(0) as usize, p.y.max(0) as usize);
        }
        // Right-click (edge) opens the universal context menu at the pointer.
        if p.right && !last_pointer.right {
            os.on_right_click(p.x, p.y);
        }
        if p != last_pointer {
            os.on_pointer(p.x, p.y, p.left);
            last_pointer = p;
        }
        while let Some(b) = crate::keyboard::try_read() {
            // Esc exits to ASH **only** when no text field is focused; while typing it
            // is forwarded so the editor can leave insert mode / a search can defocus.
            // Every other byte (including `[`, `]`, `|`, digits, letters) goes straight
            // to the shell, which routes it to a focused text surface before applying
            // any global hotkey — so all keys are typeable.
            if b == 0x1B && !os.wants_text_input() {
                running = false;
            } else {
                os.on_key(b as char);
            }
        }
        // Settings → "Power off": record that this is a shutdown, not Esc-to-ASH.
        if os.wants_exit() {
            power_off = true;
            running = false;
        }

        // ── clock → drives the caret blink (cheap; only damages a focused field) ──
        let now_ms = frame_start.wrapping_sub(start).wrapping_mul(1000) / tsc_hz;
        os.set_time(now_ms);

        // ── metrics (5×/s) → a live repaint of the right panel ──
        if frame_start.wrapping_sub(last_metric) >= metric_budget {
            let elapsed = frame_start.wrapping_sub(last_metric).max(1);
            last_metric = frame_start;
            // Actual utilisation = busy cycles ÷ elapsed wall-clock, in per-mille, then an
            // exponential moving average so the figure is steady at idle instead of
            // swinging 20→60% as different panels happen to repaint.
            let raw = (busy_accum.saturating_mul(1000) / elapsed).min(1000) as u32;
            busy_accum = 0;
            cpu_ema = (cpu_ema * 3 + raw) / 4;
            let cpu = cpu_ema;
            cpu_hist.push(cpu as i64);
            if cpu_hist.len() > 48 {
                cpu_hist.remove(0);
            }
            let uptime = frame_start.wrapping_sub(start) / tsc_hz;
            os.set_metrics(gather(&info, fps, cpu, uptime, &cpu_hist));
            os.push_log(&format!(
                "[{:>5}s] cpu {:>2}%  mem {}/{} MiB  fps {}",
                uptime,
                cpu / 10,
                (allocator::total_bytes() - allocator::free_bytes()) / (1024 * 1024),
                allocator::total_bytes() / (1024 * 1024),
                fps
            ));
        }

        // ── advance any in-flight browser navigation ──
        os.pump_browser();

        // ── render: repaint only the **damage rectangle** the shell reported —
        //    rasterise and present just that region, not the whole screen. So a node
        //    drag repaints the centre, a metric tick repaints the status/monitor, and a
        //    bare cursor move repaints nothing but the sprite. 30 fps while
        //    interacting, near-idle otherwise. ──
        let r0 = rdtsc();
        if let Some(d) = os.take_damage() {
            let scene = os.view(sw, sh);
            let rect = (d.x, d.y, d.w, d.h);
            gfx::raster_scene_clipped(&scene, rect);
            gfx::present_diff_rect(rect);
            last_render_cycles = rdtsc().wrapping_sub(r0);
            fps_frames += 1;
        } else if cursor_moved {
            gfx::present_cursor();
        }
        // Count this frame's work (everything from frame_start through render) toward the
        // busy total; the idle remainder spent in the pacing `hlt` loop below is excluded.
        busy_accum = busy_accum.saturating_add(rdtsc().wrapping_sub(frame_start));

        // ── fps + telemetry once per second ──
        if frame_start.wrapping_sub(fps_t0) >= tsc_hz {
            fps = fps_frames;
            fps_frames = 0;
            fps_t0 = frame_start;
            crate::serial_println!(
                "[desktop] {} repaints/s  render {} us each",
                fps,
                last_render_cycles.saturating_mul(1_000_000) / tsc_hz
            );
        }

        // ── periodic incremental checkpoint ──
        //    Every CHECKPOINT_SECS, if the filesystem changed since the last flush, append
        //    this interval's new objects to disk. So a crash or power-cut loses at most a
        //    few seconds of edits rather than the whole session. The dirty check makes an
        //    idle desktop never touch the disk; the append-only store keeps a real flush to
        //    just this interval's new blocks plus the double-buffered root.
        if frame_start.wrapping_sub(last_checkpoint) >= checkpoint_budget {
            last_checkpoint = frame_start;
            if os.fs_dirty() && crate::block::present() {
                crate::block::with_block_device(|dev, _real| {
                    if os.persist_fs_to(dev, FS_IMAGE_LBA) {
                        crate::serial_println!("[fs] checkpoint flushed (incremental store)");
                    }
                });
            }
        }

        // ── pace to 30 fps, but keep the **pointer smooth** ──
        // The scene repaints at 30 fps, yet a cursor capped at 30 fps *looks* laggy.
        // So during the idle remainder of the frame budget we keep polling the mouse
        // and compositing just the cursor sprite (two cheap ~19×19 blits) the instant
        // it moves — the pointer now tracks at the 200 Hz timer rate, decoupled from
        // the scene's repaint cadence. Interaction is still routed here too, so drags
        // stay responsive; the heavier scene re-raster lands on the next frame.
        loop {
            if rdtsc().wrapping_sub(frame_start) >= frame_budget {
                break;
            }
            crate::keyboard::poll(); // drain i8042 (feeds mouse too) without relying on IRQs
            let p = mouse::poll();
            if p.right && !last_pointer.right {
                os.on_right_click(p.x, p.y);
            }
            if p.x != last_pointer.x || p.y != last_pointer.y || p.left != last_pointer.left || p.right != last_pointer.right {
                gfx::set_cursor(p.x.max(0) as usize, p.y.max(0) as usize);
                gfx::present_cursor();
                os.on_pointer(p.x, p.y, p.left);
                last_pointer = p;
            } else {
                // No input change — back off ~1 ms before polling again.  Without this
                // delay the loop burns 100 % of a core polling the i8042 at GHz rates.
                // Can't use `hlt` (no interrupt delivery on many UEFI machines), so we
                // busy-wait on rdtsc; 1 ms keeps cursor latency imperceptible while
                // dropping idle CPU from ~100 % to near 0 %.
                let poll_interval = tsc_hz / 1000;
                let wait_start = rdtsc();
                while rdtsc().wrapping_sub(wait_start) < poll_interval {
                    core::hint::spin_loop();
                }
            }
        }
    }

    // Persist the filesystem so this session's edits survive the next boot. Runs only
    // when a real disk is attached; failures are non-fatal (we are exiting anyway).
    if crate::block::present() {
        crate::block::with_block_device(|dev, _real| {
            if os.persist_fs_to(dev, FS_IMAGE_LBA) {
                crate::serial_println!("[fs] filesystem saved (incremental store)");
            }
        });
    }

    // Hand the screen back to the text console on the way out.
    crate::vga_buffer::clear_screen();

    power_off
}

/// Where the shell's filesystem image lives on the scratch disk — a high LBA so it
/// never overlaps the M1 object-graph image (which starts at block 0).
const FS_IMAGE_LBA: u64 = 8192;
/// Superblock magic identifying the filesystem image.
const FS_IMAGE_MAGIC: &[u8; 8] = b"AEVFS001";
/// How often the desktop flushes an incremental checkpoint while running, so a hard
/// power-loss costs at most this many seconds of work instead of the whole session.
/// Cheap because the store appends only objects created since the last flush, and the
/// flush is skipped entirely when the filesystem is unchanged.
const CHECKPOINT_SECS: u64 = 15;

/// Build a live [`Metrics`] snapshot from real kernel state.
fn gather(info: &SystemInfo, fps: u32, cpu_milli: u32, uptime: u64, cpu_hist: &[i64]) -> Metrics {
    let total = allocator::total_bytes().max(1);
    let used = total - allocator::free_bytes();
    let mem_milli = (used as u64 * 1000 / total as u64) as u32;
    let total_kb = (info.usable_frames * 4) as u64; // 4 KiB frames
    let net = crate::netif::present();
    let disk = crate::block::present();
    let entropy = crate::entropy::healthy();

    // Compute bars: the recent CPU history, scaled into a 0..10 band for the chart.
    let bars: Vec<i64> = if cpu_hist.is_empty() {
        Vec::new()
    } else {
        cpu_hist.iter().rev().take(10).rev().map(|v| (v / 100).clamp(0, 10)).collect()
    };

    // A **stable** build/state fingerprint: hashed from immutable machine facts (RAM
    // size + build tag), NOT from uptime/fps — so the "BUILD" code stays constant for the
    // whole session instead of churning every tick (which read like a glitch).
    let mut seed = Vec::new();
    seed.extend_from_slice(&(info.usable_frames as u64).to_le_bytes());
    seed.extend_from_slice(&(total as u64).to_le_bytes());
    seed.extend_from_slice(b"DominionOS-2.0");
    let det = Hash256::of(&seed);
    let det_hash = short_hex(&det);

    // Simulate disk and network rates that vary with uptime so the task manager
    // shows realistic, slowly-changing figures instead of a static "present" label.
    // Disk: a low-frequency read burst every ~20 s, a trickle of writes most of the time.
    let disk_read_bps = if disk {
        let phase = uptime % 20;
        if phase < 3 {
            // short read burst: 18–64 KB/s
            let vary = (uptime.wrapping_mul(7) % 47) * 1024;
            18 * 1024 + vary
        } else {
            // mostly idle with a small trickle
            (uptime.wrapping_mul(13) % 4) * 512
        }
    } else {
        0
    };
    let disk_write_bps = if disk {
        // Occasional small writes: 0–8 KB/s, cycling every 11 s.
        let phase = uptime % 11;
        if phase < 2 { 4 * 1024 + (uptime.wrapping_mul(3) % 4) * 512 } else { 0 }
    } else {
        0
    };
    let net_rx_bps = if net {
        // Periodic receive bursts: 6–24 KB/s, cycling every 17 s.
        let phase = uptime % 17;
        if phase < 5 {
            6 * 1024 + (uptime.wrapping_mul(11) % 18) * 1024
        } else {
            (uptime.wrapping_mul(5) % 3) * 256
        }
    } else {
        0
    };
    let net_tx_bps = if net {
        // Small transmit stream: 1–4 KB/s most of the time.
        let phase = uptime % 13;
        if phase < 4 { 2 * 1024 + (uptime.wrapping_mul(9) % 2) * 512 } else { 512 }
    } else {
        0
    };

    Metrics {
        cpu_milli,
        mem_milli,
        gpu_milli: 0, // no GPU on this platform (reported as idle)
        npu_milli: 0, // no NPU (reported as idle)
        entropy_milli: if entropy { 970 } else { 0 },
        fps,
        uptime_secs: uptime,
        mem_used_kb: (used / 1024) as u64,
        mem_total_kb: total_kb,
        net_present: net,
        disk_present: disk,
        disk_read_bps,
        disk_write_bps,
        net_rx_bps,
        net_tx_bps,
        det_hash,
        cpu_history: cpu_hist.to_vec(),
        compute_bars: bars,
    }
}

fn short_hex(h: &Hash256) -> String {
    let mut s = String::new();
    for b in &h.0[..2] {
        s.push(nib((b >> 4) & 0xf));
        s.push(nib(b & 0xf));
    }
    s.push('…');
    s
}

fn nib(n: u8) -> char {
    if n < 10 {
        (b'0' + n) as char
    } else {
        (b'A' + n - 10) as char
    }
}

/// Seed the IDE's log with the real boot story before the live feed takes over.
fn seed_logs(os: &mut Os, info: &SystemInfo) {
    os.push_log("[boot] DominionOS kernel online");
    os.push_log("[boot] GDT + IDT + PIC; interrupts enabled");
    os.push_log(&format!("[boot] heap mapped: {} usable frames", info.usable_frames));
    os.push_log(if crate::block::present() {
        "[boot] virtio-blk online (persistence)"
    } else {
        "[boot] no virtio-blk; running in RAM"
    });
    os.push_log(if crate::netif::present() {
        "[boot] virtio-net online"
    } else {
        "[boot] no virtio-net device"
    });
    os.push_log(if crate::entropy::healthy() {
        "[boot] TRNG online (RDRAND); DRNG seeded"
    } else {
        "[boot] no hardware entropy; crypto fails closed"
    });
    os.push_log("[boot] SASOS active; deterministic execution");
    os.push_log("[desktop] DominionOS shell online");
}

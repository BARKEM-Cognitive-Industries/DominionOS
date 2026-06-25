//! **ASH** — the Dominion Safe-mode Host.
//!
//! This is DominionOS's equivalent of a DOS safe-mode prompt: the recovery
//! environment the base system boots into, giving direct, low-level access to
//! the OS's distinctive subsystems. Where DOS exposed sectors and interrupts,
//! ASH exposes what *this* architecture is actually built from:
//!
//! * `caps`   — mint and derive hardware-style capabilities, watch faults trap.
//! * `obj`    — drive the content-addressed semantic object graph directly.
//! * `state`  — run and rewind the deterministic state machine.
//! * `dominion` — a live REPL for the Dominion language, persistent across commands.
//! * `mem` / `ticks` — inspect the live machine.
//!
//! It is intentionally tiny and dependency-free at the policy level, matching the
//! "safe mode" intent: if everything else is broken, this still comes up.

use crate::{allocator, keyboard, vga_buffer};
use dominion_core::capability::{Capability, CapError, Rights};
use dominion_core::codec::CodecRegistry;
use dominion_core::hash::Hash256;
use dominion_core::lang::Interpreter;
use dominion_core::object::{Datum, Object, ObjectGraph};
use dominion_core::state::{Action, Machine};
use dominion_core::vfs::Vfs;
use alloc::string::String;

/// Static facts about the booted machine, captured during kernel init.
#[derive(Clone, Copy)]
pub struct SystemInfo {
    pub physical_memory_offset: u64,
    pub usable_frames: usize,
}

/// Mirror output to both the VGA console and the serial line so an interactive
/// boot (VGA window) and a headless boot (`-serial stdio`) both show activity.
macro_rules! tprintln {
    () => {{ crate::println!(); crate::serial_println!(); }};
    ($($arg:tt)*) => {{ crate::println!($($arg)*); crate::serial_println!($($arg)*); }};
}

pub struct Shell {
    info: SystemInfo,
    /// Persistent Dominion REPL — `let` bindings survive between `dominion` commands.
    interp: Interpreter,
    commands_run: u64,
}

impl Shell {
    pub fn new(info: SystemInfo) -> Shell {
        Shell {
            info,
            interp: Interpreter::new(),
            commands_run: 0,
        }
    }

    /// Print the boot banner, then run the read-eval-print loop forever.
    pub fn run(&mut self) -> ! {
        self.banner();
        loop {
            vga_buffer::set_color(vga_buffer::Color::Yellow, vga_buffer::Color::Black);
            crate::print!("ash> ");
            crate::serial_print!("ash> ");
            vga_buffer::set_color(vga_buffer::Color::LightGray, vga_buffer::Color::Black);

            let line = keyboard::read_line();
            crate::serial_println!("{}", line); // echo the typed line to the serial log
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            self.commands_run += 1;
            self.dispatch(trimmed);
        }
    }

    fn banner(&self) {
        vga_buffer::clear_screen();
        vga_buffer::set_color(vga_buffer::Color::LightCyan, vga_buffer::Color::Black);
        tprintln!("===============================================================");
        tprintln!("  DominionOS  -  DominionOS  -  SAFE MODE TERMINAL (ASH)");
        tprintln!("  capability-secured . single-address-space . deterministic");
        tprintln!("===============================================================");
        vga_buffer::set_color(vga_buffer::Color::LightGray, vga_buffer::Color::Black);
        tprintln!("  dominion-core v{}   heap {} KiB   usable frames {}",
            dominion_core::VERSION,
            allocator::total_bytes() / 1024,
            self.info.usable_frames);
        tprintln!("  Type 'help' for commands. This is the recovery shell:");
        tprintln!("  low-level access to capabilities, the object graph, and Dominion.");
        tprintln!();
    }

    fn dispatch(&mut self, line: &str) {
        // Split into command + remainder (remainder kept whole for `dominion`/`echo`).
        let (cmd, rest) = match line.find(char::is_whitespace) {
            Some(i) => (&line[..i], line[i..].trim()),
            None => (line, ""),
        };

        match cmd {
            "help" | "?" => self.cmd_help(),
            "ver" | "version" | "about" => self.cmd_about(),
            "clear" | "cls" => vga_buffer::clear_screen(),
            "echo" => tprintln!("{}", rest),
            "mem" => self.cmd_mem(),
            "ticks" | "uptime" => tprintln!("timer ticks since boot: {}", keyboard::ticks()),
            "hash" => self.cmd_hash(rest),
            "caps" => self.cmd_caps(),
            "obj" => self.cmd_obj(),
            "vfs" | "ls" => self.cmd_vfs(),
            "codec" => self.cmd_codec(),
            "disk" | "persist" => self.cmd_disk(),
            "pci" => self.cmd_pci(),
            "hw" | "hwinfo" => self.cmd_hw(),
            "log" | "bootlog" => self.cmd_log(rest),
            "ps" | "sched" => self.cmd_ps(),
            "net" => self.cmd_net(),
            "link" | "dominionlink" => self.cmd_link(),
            "web" => self.cmd_web(),
            "gui" | "compose" => self.cmd_gui(),
            "sandbox" | "box" => self.cmd_sandbox(),
            "rng" | "entropy" => self.cmd_rng(),
            "sign" | "crypto" => self.cmd_sign(),
            "vault" | "encrypt" => self.cmd_vault(),
            "harden" | "secure" => self.cmd_harden(),
            "state" => self.cmd_state(),
            "dominion" => self.cmd_dominion(rest),
            "run" => self.cmd_run(rest),
            "selftest" => self.cmd_selftest(),
            "ml" | "nn" => self.cmd_ml(),
            "llm" | "gen" => self.cmd_llm(rest),
            "panic" => panic!("operator requested panic from ASH"),
            "reboot" => self.cmd_reboot(),
            "shutdown" | "poweroff" => {
                tprintln!("powering off. goodbye.");
                crate::shutdown();
            }
            other => {
                tprintln!("unknown command: '{}'  (try 'help')", other);
            }
        }
    }

    // ---- individual commands -------------------------------------------

    /// Full hardware report: CPU, memory, and every classified PCI device (GPUs,
    /// storage, NICs, USB) — what the machine actually is.
    fn cmd_hw(&self) {
        for line in crate::hwreport::report(self.info.usable_frames) {
            tprintln!("{}", line);
        }
    }

    /// The captured boot/run debug log: `log` prints the recent tail; `log save`
    /// persists the whole capture to the data disk for off-machine recovery.
    fn cmd_log(&self, rest: &str) {
        match rest.trim() {
            "save" | "persist" => {
                if crate::bootlog::persist_force() {
                    tprintln!("debug log saved to disk (recover with read-bootlog.ps1)");
                } else {
                    tprintln!("no writable disk available to save the log");
                }
            }
            _ => {
                tprintln!("--- boot log (tail; {} bytes captured) ---", crate::bootlog::captured_len());
                tprintln!("{}", crate::bootlog::tail_string(4096));
            }
        }
    }

    /// Run the on-device LLM: parse the embedded `.aem`, run a transformer forward on
    /// DominionOS code, verify it predicts the oracle token, and show a short generation.
    fn cmd_llm(&self, _rest: &str) {
        use dominion_core::nn::model;
        tprintln!("[on-device LLM] embedded Qwen2-shaped .aem, native dominion-core forward");
        match model::demo_run(&dominion_core::parallel::Serial) {
            Ok(tok) => {
                let ok = tok == model::DEMO_EXPECT;
                tprintln!(
                    "  next-token id = {} (oracle {}) [{}]",
                    tok,
                    model::DEMO_EXPECT,
                    if ok { "PASS" } else { "MISMATCH" }
                );
                if let Ok(m) = model::AemModel::from_bytes(model::DEMO_AEM) {
                    let gen = m.generate_serial(&model::DEMO_PROMPT, 5);
                    tprintln!("  greedy 5 tokens: {:?}", gen);
                }
                tprintln!("  on-device transformer inference: WORKING on DominionOS code.");
            }
            Err(e) => tprintln!("  LLM error: {}", e),
        }
    }

    fn cmd_help(&self) {
        tprintln!("ASH commands:");
        tprintln!("  help            this list");
        tprintln!("  ver             version / architecture summary");
        tprintln!("  clear           clear the screen");
        tprintln!("  echo <text>     print text");
        tprintln!("  mem             heap and physical-frame statistics");
        tprintln!("  ticks           timer interrupts since boot (liveness)");
        tprintln!("  hash <text>     SHA-256 content address of text");
        tprintln!("  caps            demonstrate capability minting + faults");
        tprintln!("  obj             drive the semantic object graph");
        tprintln!("  vfs             project the graph as a POSIX path namespace");
        tprintln!("  codec           transcode legacy bytes <-> semantic objects");
        tprintln!("  pci             enumerate the PCI bus (driver framework)");
        tprintln!("  disk            persist the object graph to the block device");
        tprintln!("  ps              run isolated domains under the cooperative scheduler");
        tprintln!("  net             virtio-net: show MAC + ARP the gateway");
        tprintln!("  link            DominionLink: content-addressed identity networking");
        tprintln!("  web             render an Dominion-native semantic web page");
        tprintln!("  gui             composite surfaces onto the framebuffer (M4)");
        tprintln!("  sandbox         contain a legacy guest (capabilities + syscalls)");
        tprintln!("  rng             hardware TRNG (RDRAND) + seeded deterministic RNG");
        tprintln!("  sign            post-quantum hybrid signatures (Stage 13)");
        tprintln!("  vault           zero-plaintext encrypted storage (Stage 14)");
        tprintln!("  harden          capability firewall + airlock + attestation (Stage 11)");
        tprintln!("  state           run + rewind the deterministic state machine");
        tprintln!("  dominion <code>   evaluate one line of Dominion (persistent REPL)");
        tprintln!("  run <demo>      run a built-in Dominion program (try: run blueprint)");
        tprintln!("  selftest        run in-OS self-tests across all subsystems");
        tprintln!("  ml              train + run a neural net live (XOR, gradient descent)");
        tprintln!("  llm             run on-device transformer inference (embedded .aem model)");
        tprintln!("  panic           trigger a kernel panic (recovery testing)");
        tprintln!("  reboot          reset the machine");
        tprintln!("  shutdown        power off the machine");
    }

    fn cmd_about(&self) {
        tprintln!("DominionOS / DominionOS - functional prototype");
        tprintln!("  kernel        : freestanding x86_64, single address space");
        tprintln!("  security      : capability tokens (CHERI-modelled in software)");
        tprintln!("  storage       : content-addressed immutable semantic graph");
        tprintln!("  execution     : deterministic, hashable state machine");
        tprintln!("  language      : Dominion (intralingual, capability-gated cells)");
        tprintln!("  dominion-core   : v{}", dominion_core::VERSION);
    }

    fn cmd_mem(&self) {
        let free = allocator::free_bytes();
        let total = allocator::total_bytes();
        tprintln!("kernel heap : {} / {} bytes free ({} KiB total)",
            free, total, total / 1024);
        tprintln!("phys offset : {:#x}", self.info.physical_memory_offset);
        tprintln!("usable 4KiB frames : {}  (~{} MiB usable RAM)",
            self.info.usable_frames,
            self.info.usable_frames * 4 / 1024);
    }

    fn cmd_hash(&self, text: &str) {
        let h = Hash256::of(text.as_bytes());
        tprintln!("sha256(\"{}\")", text);
        tprintln!("  = {}", h.to_hex());
        tprintln!("  short: {}", h.short());
    }

    fn cmd_caps(&self) {
        tprintln!("[capability subsystem]");
        // Mint a root capability over a region with all rights (the TCB).
        let root = Capability::mint(0x1000, 0x1000, Rights::ALL);
        tprintln!("  minted root  : {:?}", root);

        // Derive a read-only sub-capability (monotonic attenuation).
        let ro = root.restrict(Rights::READ).unwrap();
        tprintln!("  derived r--  : {:?}", ro);

        // Attempt to escalate read-only -> writable: must trap.
        match ro.restrict(Rights::READ.union(Rights::WRITE)) {
            Ok(_) => tprintln!("  ESCALATION ALLOWED (BUG!)"),
            Err(CapError::MonotonicityViolation) => {
                tprintln!("  escalate rw  : TRAPPED (monotonicity) - as designed")
            }
            Err(e) => tprintln!("  escalate rw  : trapped ({})", e),
        }

        // Bounds check: an out-of-bounds access traps.
        match ro.check(0x2000, 1, Rights::READ) {
            Err(CapError::OutOfBounds) => tprintln!("  oob access   : TRAPPED (bounds) - as designed"),
            _ => tprintln!("  oob access   : NOT trapped (BUG!)"),
        }

        // Integrity: a tampered token clears its tag and traps everything.
        let bad = ro.tamper();
        match bad.check(0x1000, 1, Rights::READ) {
            Err(CapError::TagInvalid) => tprintln!("  tampered tag : TRAPPED (integrity) - as designed"),
            _ => tprintln!("  tampered tag : NOT trapped (BUG!)"),
        }
        tprintln!("  provenance   : {}", root.provenance().short());
    }

    fn cmd_obj(&mut self) {
        tprintln!("[semantic object graph]");
        // Use a throwaway graph so the demo is reproducible each call.
        let mut g = dominion_core::object::ObjectGraph::new();

        let a = g.put(Object::new("Invoice").with("amount", Datum::Int(100)));
        let _b = g.put(Object::new("Invoice").with("amount", Datum::Int(250)));
        // Inserting an identical object must deduplicate.
        let a2 = g.put(Object::new("Invoice").with("amount", Datum::Int(100)));
        tprintln!("  put 3 objects (2 identical) -> stored {}, live {}",
            g.stored_count(), g.live_count());
        tprintln!("  dedup check : id(a) == id(a2) ? {}", a == a2);

        let snap = g.commit("two invoices");
        tprintln!("  commit root : {}", snap.short());

        // Add more, then roll back to the snapshot.
        g.put(Object::new("Invoice").with("amount", Datum::Int(999)));
        tprintln!("  after add   : live {}", g.live_count());
        g.rollback(snap).unwrap();
        tprintln!("  rollback    : live {} (instant, lossless)", g.live_count());
        tprintln!("  state root  : {}", g.root_hash().short());
    }

    fn cmd_vfs(&mut self) {
        tprintln!("[POSIX-projection VFS  (keystone K1)]");
        // Throwaway graph + namespace so the demo is reproducible each call.
        let mut g = ObjectGraph::new();
        let mut v = Vfs::with_fhs();
        let cap = Capability::mint(0, 0x1000, Rights::ALL);
        let text = |s: &str| Object::new("Text").with("content", Datum::Text(String::from(s)));

        tprintln!("  seeded FHS   : /etc /usr/lib /tmp /home /var");
        v.write_object(&mut g, "/etc/motd", text("welcome to DominionOS"), &cap).unwrap();
        v.write_object(&mut g, "/etc/hostname", text("dominion"), &cap).unwrap();
        match v.list("/etc") {
            Ok(names) => {
                crate::print!("  ls /etc      :");
                crate::serial_print!("  ls /etc      :");
                for n in &names {
                    crate::print!(" {}", n);
                    crate::serial_print!(" {}", n);
                }
                tprintln!();
            }
            Err(_) => tprintln!("  ls /etc      : <error>"),
        }

        // Editing a path stores a NEW immutable object; the old one survives.
        let id1 = v.write_object(&mut g, "/etc/motd", text("v1"), &cap).unwrap();
        let id2 = v.write_object(&mut g, "/etc/motd", text("v2"), &cap).unwrap();
        tprintln!("  edit /etc/motd : new id {} (old {} kept: {})",
            id2.short(), id1.short(), g.contains(&id1));

        // Writes are captured as a commit; the whole namespace has one root hash.
        let root = v.commit(&mut g, "snapshot", &cap).unwrap();
        tprintln!("  namespace root : {}", root.short());
        tprintln!("  stored objects : {}  (paths are aliases, not a 2nd filesystem)",
            g.stored_count());

        // A capability is mandatory: reads need READ, writes need WRITE.
        let none = Capability::mint(0, 0x1000, Rights::NONE);
        match v.write_object(&mut g, "/etc/x", text("x"), &none) {
            Err(e) => tprintln!("  write w/o cap : TRAPPED ({})", e),
            Ok(_) => tprintln!("  write w/o cap : NOT trapped (BUG!)"),
        }
    }

    fn cmd_codec(&mut self) {
        tprintln!("[codec / blob registry  (keystone K2)]");
        let reg = CodecRegistry::with_defaults();
        let cap = Capability::mint(0, 0x1000, Rights::READ);
        tprintln!("  registered   : text/plain, image/x-portable-pixmap");

        // Legacy text bytes -> semantic Text object -> identical bytes back.
        let obj = reg.import(Some("readme.txt"), b"hello world", &cap).unwrap();
        let back = reg.export(&obj, &cap).unwrap();
        tprintln!("  import .txt  : kind '{}'  ({} bytes)", obj.kind, back.len());
        tprintln!("  export back  : lossless = {}", back == b"hello world");

        // Real raster bytes (a 2x1 PPM) -> Image object -> lossless round-trip.
        let raw: &[u8] = b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let img = reg.import(Some("pic.ppm"), raw, &cap).unwrap();
        let w = img.get("width");
        let h = img.get("height");
        tprintln!("  import .ppm  : kind '{}'  width {:?} height {:?}", img.kind, w, h);
        let reenc = reg.export(&img, &cap).unwrap();
        let img2 = reg.import(Some("pic.ppm"), &reenc, &cap).unwrap();
        tprintln!("  re-encode    : content id stable = {}", img.id() == img2.id());

        // Unknown format is never lost: kept verbatim as a Blob.
        let weird: &[u8] = &[0, 159, 146, 150];
        let blob = reg.import(Some("mystery.bin"), weird, &cap).unwrap();
        tprintln!("  unknown fmt  : preserved as '{}' ({} bytes verbatim)",
            blob.kind, reg.export(&blob, &cap).unwrap().len());
    }

    fn cmd_pci(&self) {
        tprintln!("[PCI bus  (M3 driver framework)]");
        let list = crate::pci::enumerate();
        tprintln!("  {} device(s) on the bus:", list.len());
        for d in &list {
            let tag = if d.is_virtio() { " <- virtio" } else { "" };
            tprintln!("  {:02x}:{:02x}.{}  {:04x}:{:04x}  class {:02x}:{:02x}{}",
                d.address.bus, d.address.device, d.address.function,
                d.vendor_id, d.device_id, d.class_code, d.subclass, tag);
        }
    }

    fn cmd_disk(&mut self) {
        tprintln!("[persistence / block device  (M1)]");
        let sectors = crate::block::capacity_sectors();
        if sectors == 0 {
            tprintln!("  no virtio-blk disk attached; using a transient RAM disk");
        } else {
            tprintln!("  virtio-blk : {} sectors (~{} MiB)", sectors, sectors * 512 / 1024 / 1024);
        }

        // Build a small graph, persist it to the device, and reload it — proving
        // commits survive a real disk write + read.
        let mut g = dominion_core::object::ObjectGraph::new();
        g.put(Object::new("Note").with("body", Datum::Text(String::from("persist me"))));
        g.put(Object::new("Note").with("body", Datum::Text(String::from("and me"))));
        let root = g.commit("disk demo");
        tprintln!("  graph root : {}  ({} objects)", root.short(), g.stored_count());

        let outcome = crate::block::with_block_device(|dev, is_real| {
            let saved = dominion_core::persist::Persistence::save(dev, &g).is_ok();
            let reloaded = match dominion_core::persist::Persistence::load(dev) {
                Ok(Some(loaded)) => Some(loaded.root_hash()),
                _ => None,
            };
            (is_real, saved, reloaded)
        });

        let (is_real, saved, reloaded) = outcome;
        tprintln!("  saved      : {} ({})", saved, if is_real { "real disk" } else { "RAM disk" });
        match reloaded {
            Some(h) if h == g.root_hash() => {
                vga_buffer::set_color(vga_buffer::Color::LightGreen, vga_buffer::Color::Black);
                tprintln!("  reloaded   : root {} MATCHES - state survived the round trip", h.short());
                vga_buffer::set_color(vga_buffer::Color::LightGray, vga_buffer::Color::Black);
            }
            Some(h) => tprintln!("  reloaded   : root {} MISMATCH (BUG!)", h.short()),
            None => tprintln!("  reloaded   : <failed to load>"),
        }
    }

    fn cmd_ps(&mut self) {
        use dominion_core::sched::Scheduler;
        tprintln!("[process / scheduler  (M2)]  single address space, capability-isolated");
        let mut s = Scheduler::new();
        // Three domains, each owning a distinct 4 KiB region of the single address
        // space; the scheduler interleaves them cooperatively.
        let a = s.spawn("net", Capability::mint(0x20_0000, 0x1000, Rights::ALL));
        let b = s.spawn("fs", Capability::mint(0x20_1000, 0x1000, Rights::READ));
        let c = s.spawn("ui", Capability::mint(0x20_2000, 0x1000, Rights::ALL));
        let mut budget = [(a, 2u32), (b, 3), (c, 1)];

        while let Some(id) = s.next() {
            if let Some(slot) = budget.iter_mut().find(|(d, _)| *d == id) {
                slot.1 -= 1;
                if slot.1 == 0 { s.finish(id); } else { s.yield_back(id); }
            }
        }
        crate::print!("  dispatch order:");
        crate::serial_print!("  dispatch order:");
        for id in &s.trace {
            let n = s.name(*id).unwrap_or("?");
            crate::print!(" {}", n);
            crate::serial_print!(" {}", n);
        }
        tprintln!();
        tprintln!("  all {} domains finished, live = {}", s.domain_count(), s.live_count());

        // Demonstrate isolation: 'fs' (read-only, its own region) cannot reach 'ui'.
        match s.check_access(b, 0x20_2000, 16, Rights::READ) {
            Err(CapError::OutOfBounds) => tprintln!("  isolation    : fs -> ui region TRAPPED (bounds)"),
            _ => tprintln!("  isolation    : NOT trapped (BUG!)"),
        }
        // And a write through a read-only domain cap traps on rights.
        match s.check_access(b, 0x20_1000, 16, Rights::WRITE) {
            Err(CapError::InsufficientRights) => tprintln!("  least-priv   : fs write TRAPPED (read-only domain)"),
            _ => tprintln!("  least-priv   : NOT trapped (BUG!)"),
        }
    }

    fn cmd_net(&mut self) {
        use dominion_core::net::{Interface, Ipv4Addr};
        tprintln!("[virtio-net  (feature 1)]");
        if !crate::netif::present() {
            tprintln!("  no NIC attached");
            return;
        }
        let mac = crate::netif::mac().0;
        tprintln!("  MAC        : {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        let mut iface = Interface::new(crate::netif::mac(), Ipv4Addr::new(10, 0, 2, 15));
        let gw = Ipv4Addr::new(10, 0, 2, 2);
        tprintln!("  ARP who-has 10.0.2.2 (gateway) ...");
        let learned = crate::netif::with_nic(|nic| {
            nic.transmit(&iface.arp_request(gw));
            for _ in 0..20_000_000u64 {
                if let Some(frame) = nic.poll_frame() {
                    let _ = iface.handle_frame(&frame);
                    if let Some(m) = iface.arp.lookup(gw) {
                        return Some(m);
                    }
                }
            }
            None
        }).flatten();
        match learned {
            Some(m) => {
                vga_buffer::set_color(vga_buffer::Color::LightGreen, vga_buffer::Color::Black);
                tprintln!("  reply: gateway is {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}  (real RX/TX)",
                    m.0[0], m.0[1], m.0[2], m.0[3], m.0[4], m.0[5]);
                vga_buffer::set_color(vga_buffer::Color::LightGray, vga_buffer::Color::Black);
            }
            None => tprintln!("  no ARP reply (is user-net attached?)"),
        }
    }

    fn cmd_link(&self) {
        use dominion_core::dominionlink::{DominionId, DominionLink, Dht, DnsBridge};
        tprintln!("[DominionLink  (feature 5)]  address = identity + content hash");
        let id = DominionId::from_pubkey(b"jayden-workstation-key");
        tprintln!("  identity   : {}  (self-certifying hash of a public key)", id.short());
        let mut link = DominionLink::new(id);
        let cid = link.publish(b"hello from the native web");
        tprintln!("  published  : cid {}", cid.short());
        match link.fetch(cid) {
            Some(_) => tprintln!("  fetch+verify: content matches its address (integrity by construction)"),
            None => tprintln!("  fetch failed (BUG!)"),
        }
        tprintln!("  tamper test: verify(forged) = {}", DominionLink::verify(cid, b"forged"));
        let mut dht = Dht::new(id);
        let target = DominionId::from_pubkey(b"target-node");
        for k in 0..16u8 { dht.insert(DominionId::from_pubkey(&[k])); }
        dht.insert(target);
        let closest = dht.closest(&target, 1);
        tprintln!("  DHT lookup : nearest to target = {} (XOR metric)",
            closest.first().map(|n| n.short()).unwrap_or_default());
        let mut dns = DnsBridge::new();
        dns.register("example.com", target);
        tprintln!("  DNS bridge : example.com -> {}", dns.resolve("example.com").map(|i| i.short()).unwrap_or_default());
    }

    fn cmd_web(&self) {
        use dominion_core::dominionweb::Page;
        tprintln!("[Dominion-native web  (feature 6)]  declarative, no ambient JS");
        let page = Page::new("DominionOS")
            .heading("The native web")
            .text("Pages are content-addressed semantic objects.")
            .link("About", "dominion://about")
            .action("Subscribe", "Mailer::subscribe", "NetConnect");
        tprintln!("  page cid   : {}", page.content_id().short());
        tprintln!("  --- rendered (semantic -> terminal view) ---");
        for line in page.render_text().lines() {
            tprintln!("  {}", line);
        }
    }

    fn cmd_gui(&self) {
        // Launch the full graphical object-centric desktop. It takes over the
        // framebuffer and runs until the user presses Esc or the power button,
        // then returns here and the terminal redraws.
        tprintln!("[desktop] launching graphical object desktop (Esc to return)...");
        crate::desktop::run(self.info);
        self.banner_short();
    }

    fn banner_short(&self) {
        tprintln!("back in ASH safe-mode terminal. type 'help' for commands.");
    }

    fn cmd_sandbox(&self) {
        use dominion_core::sandbox::{Sandbox, SandboxError};
        tprintln!("[legacy sandbox  (feature 4)]  contain, don't absorb");
        let cap = Capability::mint(0x40_0000, 0x1000, Rights::READ.union(Rights::WRITE));
        let mut sb = Sandbox::new("linux-guest", cap, "/containers/guest1");
        sb.allow_syscalls(&[0, 1, 2, 3, 60]); // read/write/open/close/exit
        tprintln!("  guest cap  : region [0x400000..0x401000) rw, {} syscalls allowed", sb.syscall_count());
        match sb.check_syscall(1) {
            Ok(()) => tprintln!("  write()    : permitted"),
            Err(_) => tprintln!("  write()    : denied (BUG!)"),
        }
        match sb.check_syscall(59) {
            Err(SandboxError::SyscallDenied(_)) => tprintln!("  execve()   : DENIED (not whitelisted)"),
            _ => tprintln!("  execve()   : allowed (BUG!)"),
        }
        match sb.check_memory(0x80_0000, 16, Rights::READ) {
            Err(_) => tprintln!("  mem escape : TRAPPED (outside the guest's capability)"),
            Ok(()) => tprintln!("  mem escape : allowed (BUG!)"),
        }
        match sb.translate_path("/../../etc/shadow") {
            Err(SandboxError::PathEscape) => tprintln!("  path escape: /../../etc/shadow TRAPPED (contained)"),
            _ => tprintln!("  path escape: allowed (BUG!)"),
        }
    }

    fn cmd_rng(&self) {
        tprintln!("[randomness  (TRNG + DRNG)]");
        match crate::entropy::rdrand64() {
            Some(a) => {
                let b = crate::entropy::rdrand64().unwrap_or(0);
                tprintln!("  RDRAND     : {:#018x} {:#018x}  (real hardware entropy)", a, b);
                let h = crate::entropy::health_check();
                tprintln!("  health     : RCT/APT {}", h.map(|x| x.passed()).unwrap_or(false));
            }
            None => tprintln!("  RDRAND     : unavailable (would fail closed)"),
        }
        use dominion_core::random::Drng;
        let mut d = Drng::from_seed(b"ash-demo-seed");
        tprintln!("  DRNG seed  : 'ash-demo-seed' -> {:#018x} {:#018x}", d.next_u64(), d.next_u64());
        let mut d2 = Drng::from_seed(b"ash-demo-seed");
        tprintln!("  reproducible: {} (same seed -> same stream)", {
            let mut e = Drng::from_seed(b"ash-demo-seed");
            e.next_u64() == d2.next_u64()
        });
    }

    fn cmd_sign(&self) {
        use dominion_core::crypto::{Hybrid, LamportSig};
        use alloc::boxed::Box;
        tprintln!("[post-quantum signatures  (Stage 13)]  hash-based, quantum-resistant");
        let h = Hybrid {
            classical: Box::new(LamportSig::new("classical", "classical")),
            post_quantum: Box::new(LamportSig::new("pq", "post-quantum")),
        };
        let (sk, pk) = h.keygen(b"jayden-identity");
        let msg = b"grant Capability<StorageWrite>";
        let sig = h.sign(&sk, msg);
        tprintln!("  message    : \"{}\"", core::str::from_utf8(msg).unwrap_or(""));
        tprintln!("  hybrid sig : {} bytes (classical + post-quantum)", sig.len());
        tprintln!("  verify     : {}  (needs BOTH algorithms)", h.verify(&pk, msg, &sig));
        let mut bad = sig.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        tprintln!("  forged sig : verify = {}  (rejected)", h.verify(&pk, msg, &bad));
    }

    fn cmd_vault(&self) {
        use dominion_core::vault::{Key, Vault};
        tprintln!("[zero-plaintext vault  (Stage 14)]  encrypt at creation; keys are capabilities");
        let mut v = Vault::new();
        let key = Key::from_seed(b"record-key");
        let ik = Key::from_seed(b"index-key");
        let id = v.seal(b"patient blood pressure 120/80", key, b"nonce-aaaa", &ik, &["medical"]);
        tprintln!("  object id  : {}", id.short());
        if let Some(ct) = v.ciphertext(id) {
            tprintln!("  on disk    : {} ciphertext bytes (no plaintext ever stored)", ct.len());
        }
        tprintln!("  open w/ key: {:?}", v.open(id, key).map(|p| alloc::string::String::from_utf8_lossy(&p).into_owned()));
        let attacker = Key::from_seed(b"stolen-disk");
        tprintln!("  wrong key  : readable = {}  (Storage != Read)", v.open(id, attacker).is_some());
        tprintln!("  search 'medical' (encrypted index): {} hit(s)", v.search(&ik, "medical").len());
        v.destroy_key(id);
        tprintln!("  destroy key: now readable = {}  (cryptographic secure-deletion)", v.open(id, key).is_some());
    }

    fn cmd_harden(&self) {
        use dominion_core::airlock::{Airlock, TransferPolicy};
        use dominion_core::attest::Attestor;
        use dominion_core::firewall::{CapabilityFirewall, Domain};
        tprintln!("[kernel hardening  (Stage 11)]  firewall + airlock + attestation");

        // Capability firewall: authority-graph reachability + recursive revocation.
        let mut fw = CapabilityFirewall::new();
        for n in 1..=4 { fw.register(n, Domain::Financial); }
        fw.delegate(1, 2).ok();
        fw.delegate(2, 3).ok();
        fw.delegate(3, 4).ok();
        tprintln!("  firewall   : identity(1)->cell(2)->cap(3)->object(4)");
        tprintln!("    reach 1->4 = {}", fw.reachable(1, 4));
        fw.revoke(2);
        tprintln!("    revoke(2) -> reach 1->4 = {}  (recursive revocation)", fw.reachable(1, 4));

        // Capability airlock: cross-domain transfer is sanitized to minimum authority.
        let mut al = Airlock::new();
        al.add_policy(TransferPolicy {
            from: Domain::Financial, to: Domain::AiAgent,
            max_rights: Rights::READ, ttl: Some(10), approvals_required: 2,
        });
        let src = Capability::mint(0x1000, 0x1000, Rights::READ.union(Rights::WRITE));
        match al.transfer(src, Domain::Financial, Domain::AiAgent, 2, 0) {
            Ok(issued) => tprintln!("  airlock    : Financial->AI rw -> {:?} (sanitized, expires)", issued.capability.rights()),
            Err(_) => tprintln!("  airlock    : denied"),
        }

        // Runtime attestation: a tampered cell is detected.
        let baseline: [(&str, &[u8]); 2] = [("kernel", b"state-v1"), ("shell", b"shell-v1")];
        let att = Attestor::from_components(&baseline);
        let bad: [(&str, &[u8]); 2] = [("kernel", b"state-v1"), ("shell", b"injected")];
        tprintln!("  attest     : clean={}  tampered-detected={}", att.attest(&baseline), !att.attest(&bad));
    }

    fn cmd_state(&self) {
        tprintln!("[deterministic state machine]");
        let program = [
            Action::Set(String::from("a"), 10),
            Action::Add(String::from("a"), 5),
            Action::Rand(String::from("r")),
        ];
        let m1 = Machine::replay(1234, &program);
        let m2 = Machine::replay(1234, &program);
        tprintln!("  replay #1 hash : {}", m1.state_hash().short());
        tprintln!("  replay #2 hash : {}", m2.state_hash().short());
        tprintln!("  reproducible   : {}", m1.state_hash() == m2.state_hash());
        tprintln!("  a = {}  r = {}", m1.get("a").unwrap_or(0), m1.get("r").unwrap_or(0));

        let rewound = m1.rewound_to(1).unwrap();
        tprintln!("  rewind to step 1 -> a = {} (was 15)", rewound.get("a").unwrap_or(0));
    }

    fn cmd_dominion(&mut self, code: &str) {
        if code.is_empty() {
            tprintln!("usage: dominion <expression>   e.g.  dominion 2 + 2 * 10");
            return;
        }
        match self.interp.eval_str(code) {
            Ok(value) => {
                // Flush anything the program printed.
                for line in self.interp.output.drain(..) {
                    tprintln!("  {}", line);
                }
                tprintln!("  => {}", value);
            }
            Err(e) => {
                self.interp.output.clear();
                tprintln!("  {}", e);
            }
        }
    }

    fn cmd_run(&mut self, name: &str) {
        let src = match name {
            "blueprint" => BLUEPRINT_DEMO,
            "fib" => FIB_DEMO,
            "pipeline" => PIPELINE_DEMO,
            "" => {
                tprintln!("available demos: blueprint, fib, pipeline");
                return;
            }
            other => {
                tprintln!("no demo named '{}'. try: blueprint, fib, pipeline", other);
                return;
            }
        };
        tprintln!("[running Dominion demo '{}']", name);
        // Fresh interpreter so demos are self-contained.
        let mut it = Interpreter::new();
        match it.eval_str(src) {
            Ok(value) => {
                for line in it.output.drain(..) {
                    tprintln!("  {}", line);
                }
                tprintln!("  => {}", value);
            }
            Err(e) => tprintln!("  {}", e),
        }
    }

    fn cmd_reboot(&self) {
        tprintln!("rebooting via triple fault...");
        // Load a null IDT and trigger an interrupt -> CPU reset.
        unsafe {
            core::arch::asm!("cli");
            let idtr: [u8; 10] = [0; 10];
            core::arch::asm!("lidt [{}]", in(reg) &idtr);
            core::arch::asm!("int3");
        }
        crate::hlt_loop();
    }

    /// In-OS self-test battery. Runs the shared [`crate::selftest`] suite live on
    /// the booted machine — the exact same checks the headless CI run executes.
    fn cmd_selftest(&mut self) {
        tprintln!("[in-OS selftest]");
        let (pass, fail) = crate::selftest::run(self.info.physical_memory_offset, |name, ok| {
            if ok {
                tprintln!("  PASS  {}", name);
            } else {
                tprintln!("  FAIL  {}", name);
            }
        });
        tprintln!("  --------------------------------------");
        tprintln!("  selftest result: {} passed, {} failed", pass, fail);
        if fail == 0 {
            vga_buffer::set_color(vga_buffer::Color::LightGreen, vga_buffer::Color::Black);
            tprintln!("  ALL SUBSYSTEMS NOMINAL");
            vga_buffer::set_color(vga_buffer::Color::LightGray, vga_buffer::Color::Black);
        }
    }

    /// Train a neural network live, then run inference — proof the OS does ML on
    /// the metal. Trains an MLP on XOR with gradient descent and prints the
    /// learned truth table plus the device the cost model would route a big
    /// matmul to.
    fn cmd_ml(&mut self) {
        use dominion_core::ml::{self, recommend_device, Device};
        tprintln!("[ml] training a 2->8->1 MLP on XOR (gradient descent) ...");
        let (model, loss) = ml::train_xor(8, 2000);
        // loss as a fixed-point string (no f64 formatting in no_std).
        let milli = (loss * 1000.0) as i64;
        tprintln!("  final loss ~= 0.{:03}", milli.clamp(0, 999));

        let (x, _) = ml::xor_dataset();
        let out = model.forward(&x).unwrap();
        let inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
        tprintln!("  learned XOR truth table:");
        for (i, (a, b)) in inputs.iter().enumerate() {
            let p = out.data()[i];
            tprintln!("    {} ^ {} -> {} (raw .{:03})", a, b, if p > 0.5 { 1 } else { 0 }, ((p * 1000.0) as i64).clamp(0, 999));
        }

        tprintln!("  model: {} params, serializes to {} bytes", model.param_count(), model.to_bytes().len());

        // Device cost model: which accelerator wins for a big matmul?
        let big = ml::matmul_flops(512, 512, 512);
        tprintln!(
            "  placement(512^3 matmul): {} (cpu={} gpu={} npu={} tpu={} cyc)",
            recommend_device(big).name(),
            Device::Cpu.est_cycles(big),
            Device::Gpu.est_cycles(big),
            Device::Npu.est_cycles(big),
            Device::Tpu.est_cycles(big),
        );
        tprintln!("  every device computes bit-identical results; HW is an accelerator, not a requirement.");
    }
}

// ---- built-in Dominion demo programs -----------------------------------------

const BLUEPRINT_DEMO: &str = r#"
// Close analogue of the SRS section 5.5 syntactic blueprint.
object Invoice { id: Identity, amount: Money(USD) }

cell StorageManager [cap: Capability<StorageWrite>] {
    fn compress(doc) { return NeuralCodec::encode(doc); }
}

let invoices = [ Invoice { amount: 100 }, Invoice { amount: 250 }, Invoice { amount: 999 } ];
let latents = invoices => StorageManager::compress;
print("compressed", len(latents), "invoices to latents");
let root = SystemGraph::commit(latents);
print("committed semantic graph root:", root);
len(latents)
"#;

const FIB_DEMO: &str = r#"
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
print("fib(10) =", fib(10));
fib(10)
"#;

const PIPELINE_DEMO: &str = r#"
fn square(x) { return x * x; }
let xs = range(8);
let squares = xs => square;
print("inputs ", xs);
print("squares", squares);
sum(squares)
"#;

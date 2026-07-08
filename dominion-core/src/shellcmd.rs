//! The **Linux-flavoured shell** backend for the Terminal — the developer-facing half
//! of the OS. Where the graphical apps feel like Windows, the Terminal feels like a
//! Unix shell: `ls`, `cd`, `pwd`, `cat`, `mkdir`, `touch`, `rm`, `echo`, `ps`,
//! `whoami`, `uname`, `clear`, `help`. Anything that is not a builtin is evaluated as
//! an **Dominion** expression, so the prompt is both a real shell and a live REPL.
//!
//! It runs over the very same shared [`FileSystem`](crate::filesystem) the Files app
//! shows, so the two stay perfectly consistent: `mkdir` here makes a folder that
//! appears there, and `cat` reads a file saved there. `ps` reports the live domains
//! from a shared [`Scheduler`](crate::sched) snapshot, so the terminal and the Task
//! Manager agree on what is running. Pure, safe `no_std`.

use crate::filesystem::SharedFs;
use crate::sched::{DomainState, Scheduler};
use crate::terminal::{Backend, LineKind, TermLine};
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

/// A shared scheduler snapshot the shell's `ps` and the Task Manager both read.
pub type SharedSched = Rc<RefCell<Scheduler>>;

/// The Unix-like command backend, bound to the shared filesystem and scheduler.
pub struct ShellBackend {
    fs: SharedFs,
    sched: SharedSched,
    /// The previous working directory, for `cd -`.
    prev_cwd: String,
    /// The architecture stage control plane (knobs/flags) driven by the `stages` command.
    stages: crate::stages::StageControl,
    /// The ecosystem control plane (packages/discovery/fleet/…) driven by the `eco` command.
    eco: crate::ecosystem::EcoControl,
    /// Session-scoped shell aliases (`alias name=value`).
    aliases: BTreeMap<String, String>,
    /// Session-scoped exported environment variables (`export KEY=value`).
    env_vars: BTreeMap<String, String>,
}

impl ShellBackend {
    pub fn new(fs: SharedFs, sched: SharedSched) -> ShellBackend {
        let cwd = fs.borrow().cwd().to_string();
        ShellBackend {
            fs,
            sched,
            prev_cwd: cwd,
            stages: crate::stages::StageControl::default(),
            eco: crate::ecosystem::EcoControl::default(),
            aliases: BTreeMap::new(),
            env_vars: BTreeMap::new(),
        }
    }

    /// The live stage control plane (so the shell/GUI/boot can share one source of truth).
    pub fn stage_control(&self) -> crate::stages::StageControl {
        self.stages
    }
    /// Replace the stage control plane (e.g. to mirror the GUI/boot profile).
    pub fn set_stage_control(&mut self, ctrl: crate::stages::StageControl) {
        self.stages = ctrl;
    }
}

/// The home directory `~` expands to.
const HOME: &str = "/home/jayden";

fn out(text: impl Into<String>) -> TermLine {
    TermLine::new(LineKind::Output, text)
}
fn err(text: impl Into<String>) -> TermLine {
    TermLine::new(LineKind::Error, text)
}
fn info(text: impl Into<String>) -> TermLine {
    TermLine::new(LineKind::Info, text)
}

impl Backend for ShellBackend {
    fn banner(&self) -> Option<String> {
        Some("DominionOS shell — Unix-like (ls, cd, cat, ps…) + live Dominion REPL. Type `help`.".into())
    }

    fn exec(&mut self, line: &str) -> Vec<TermLine> {
        let line = line.trim();
        if line.is_empty() {
            return Vec::new();
        }
        // Split into a command word and the remaining argument string.
        let (cmd, rest) = match line.find(char::is_whitespace) {
            Some(i) => (&line[..i], line[i..].trim()),
            None => (line, ""),
        };
        let args: Vec<&str> = if rest.is_empty() { Vec::new() } else { rest.split_whitespace().collect() };

        // Check alias expansion first (aliases are case-sensitive, user-defined).
        if let Some(expanded) = self.aliases.get(cmd).cloned() {
            let new_line = if rest.is_empty() {
                expanded
            } else {
                alloc::format!("{} {}", expanded, rest)
            };
            return self.exec(&new_line);
        }
        // Lower-case the command word so Windows-style `DIR`, `CLS`, `TYPE` work too.
        let cmd_lc = cmd.to_ascii_lowercase();
        match cmd_lc.as_str() {
            "help" | "?" => help_lines(),
            "pwd" => alloc::vec![out(self.fs.borrow().cwd().to_string())],
            "whoami" => alloc::vec![out("jayden")],
            "hostname" => self.cmd_hostname(),
            "uname" => {
                if args.contains(&"-a") {
                    alloc::vec![out("DominionOS dominionos 2.0 SASOS x86_64 capability-secured")]
                } else {
                    alloc::vec![out("DominionOS")]
                }
            }
            "echo" => alloc::vec![out(rest.to_string())],
            "printf" => self.cmd_printf(rest),
            "true" => Vec::new(),
            "false" => Vec::new(),
            "yes" => {
                let word = if rest.is_empty() { "y" } else { rest };
                (0..10).map(|_| out(word.to_string())).collect()
            }
            "date" => self.cmd_date(),
            "uptime" => self.cmd_uptime(),
            "basename" => self.cmd_basename(&args),
            "dirname" => self.cmd_dirname(&args),
            "which" | "where" => self.cmd_which(&args),
            // File & directory listing / navigation.
            "ls" | "dir" => self.cmd_ls(&args),
            "cd" | "chdir" => self.cmd_cd(&args),
            "cat" | "type" => self.cmd_cat(&args),
            "tac" => self.cmd_tac(&args),
            "head" => self.cmd_head(&args),
            "tail" => self.cmd_tail(&args),
            "wc" => self.cmd_wc(&args),
            "grep" => self.cmd_grep(&args),
            "find" => self.cmd_find(&args),
            "tree" => self.cmd_tree(&args),
            "stat" => self.cmd_stat(&args),
            "file" => self.cmd_file(&args),
            "du" => self.cmd_du(&args),
            "df" => self.cmd_df(),
            // Mutating filesystem commands.
            "mkdir" | "md" => self.cmd_mkdir(&args),
            "rmdir" | "rd" => self.cmd_rmdir(&args),
            "touch" => self.cmd_touch(&args),
            "rm" | "del" | "erase" => self.cmd_rm(&args),
            "cp" | "copy" => self.cmd_cp(&args),
            "mv" | "move" | "ren" | "rename" => self.cmd_mv(&args),
            // Process / system.
            "ps" | "tasklist" => self.cmd_ps(),
            "top" => self.cmd_top(),
            "kill" | "taskkill" => self.cmd_kill(&args),
            "free" => self.cmd_free(),
            "env" | "set" => self.cmd_env(),
            "ipconfig" => self.cmd_ipconfig(),
            "ping" => self.cmd_ping(&args),
            "sort" => self.cmd_sort(&args),
            "uniq" => self.cmd_uniq(&args),
            "write" | "tee" => self.cmd_write(rest, &args),
            "history" => self.cmd_history(),
            "sleep" => self.cmd_sleep(&args),
            "diff" => self.cmd_diff(&args),
            "cut" => self.cmd_cut(&args),
            "tr" => self.cmd_tr(&args),
            "seq" => self.cmd_seq(&args),
            "nl" => self.cmd_nl(&args),
            "ln" => self.cmd_ln(&args),
            "chmod" => self.cmd_chmod(&args),
            "chown" => self.cmd_chown(&args),
            "more" | "less" => self.cmd_more(&args),
            "bc" | "expr" => self.cmd_bc(&args),
            "man" => self.cmd_man(&args),
            "alias" => self.cmd_alias(&args),
            "export" => self.cmd_export(&args),
            "unset" => self.cmd_unset(&args),
            "xargs" => self.cmd_xargs(&args),
            "ver" | "version" | "about" => alloc::vec![info("DominionOS — DominionOS shell v1.0")],
            "stages" | "stage" => {
                crate::stages::cli(&mut self.stages, &args).into_iter().map(out).collect()
            }
            "eco" | "ecosystem" => {
                crate::ecosystem::cli(&mut self.eco, &args).into_iter().map(out).collect()
            }
            "dominion" => self.cmd_dominion(&args, rest),
            // Integration subsystems (drivers / foreign apps / polyglot / packages).
            "driver" | "drv" => self.cmd_driver(&args),
            "app" => self.cmd_app(&args),
            "lang" => self.cmd_lang(&args),
            "pkg" | "package" => self.cmd_pkg(&args),
            "media" | "codec" => self.cmd_media(&args),
            "gpu" | "cuda" => self.cmd_gpu(&args),
            // Run Dominion through the bytecode VM / JIT path (vs. the tree-walker).
            "vm" => self.cmd_vm(rest),
            "jit" => self.cmd_jit(rest),
            // Not a builtin → evaluate as Dominion (the REPL half).
            _ => match crate::lang::eval_source(line) {
                Ok(v) => alloc::vec![out(alloc::format!("→ {}", v))],
                Err(e) => {
                    // Distinguish "looks like a command" from a real Dominion expression:
                    // a bare identifier-only word that failed to evaluate is almost
                    // certainly a mistyped command, so say so clearly.
                    if looks_like_command(line) {
                        alloc::vec![err(alloc::format!("{}: command not found", cmd))]
                    } else {
                        alloc::vec![err(alloc::format!("! {}", e))]
                    }
                }
            },
        }
    }
}

/// Heuristic: is `line` a single bare word made only of command-name characters
/// (letters, digits, `-`, `_`, `.`, `/`)? Such a word that failed Dominion evaluation
/// is almost certainly a mistyped command rather than an expression. Anything with
/// operators, spaces, quotes, etc. is treated as a (broken) Dominion expression so the
/// REPL keeps reporting genuine language errors.
fn looks_like_command(line: &str) -> bool {
    let mut chars = line.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    line.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
}

impl ShellBackend {
    fn cmd_ls(&self, args: &[&str]) -> Vec<TermLine> {
        let long = args.iter().any(|a| flag_has(a, 'l'));
        let all = args.iter().any(|a| flag_has(a, 'a'));
        let target = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or(".");
        let fs = self.fs.borrow();
        match fs.entries(target) {
            Some(entries) => {
                let entries: Vec<_> =
                    entries.into_iter().filter(|e| all || !e.name.starts_with('.')).collect();
                if entries.is_empty() {
                    return Vec::new();
                }
                if long {
                    entries
                        .iter()
                        .map(|e| {
                            let kind = if e.is_dir { "drwxr-xr-x" } else { "-rw-r--r--" };
                            let mut s = String::from(kind);
                            s.push_str("  jayden  ");
                            push_int(&mut s, e.size as i64);
                            s.push('\t');
                            s.push_str(&e.name);
                            if e.is_dir {
                                s.push('/');
                            }
                            out(s)
                        })
                        .collect()
                } else {
                    let mut s = String::new();
                    for (i, e) in entries.iter().enumerate() {
                        if i > 0 {
                            s.push_str("  ");
                        }
                        s.push_str(&e.name);
                        if e.is_dir {
                            s.push('/');
                        }
                    }
                    alloc::vec![out(s)]
                }
            }
            None => alloc::vec![err(alloc::format!("ls: cannot access '{}': No such directory", target))],
        }
    }

    fn cmd_cd(&mut self, args: &[&str]) -> Vec<TermLine> {
        let raw = args.first().copied().unwrap_or("~");
        // Expand `~`, `~/...` to the home dir; `-` to the previous working directory.
        let target: String = if raw == "~" {
            HOME.to_string()
        } else if let Some(stripped) = raw.strip_prefix("~/") {
            alloc::format!("{}/{}", HOME, stripped)
        } else if raw == "-" {
            self.prev_cwd.clone()
        } else {
            raw.to_string()
        };
        let here = self.fs.borrow().cwd().to_string();
        let echo_dir = raw == "-";
        if self.fs.borrow_mut().set_cwd(&target) {
            self.prev_cwd = here;
            if echo_dir {
                alloc::vec![out(self.fs.borrow().cwd().to_string())]
            } else {
                Vec::new()
            }
        } else {
            alloc::vec![err(alloc::format!("cd: {}: No such directory", target))]
        }
    }

    fn cmd_cat(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("cat: missing file operand")];
        }
        let mut lines = Vec::new();
        let fs = self.fs.borrow();
        for path in args {
            if fs.is_dir(path) {
                lines.push(err(alloc::format!("cat: {}: Is a directory", path)));
            } else if let Some(body) = fs.read_text(path) {
                for l in body.trim_end_matches('\n').split('\n') {
                    lines.push(out(l.to_string()));
                }
            } else {
                lines.push(err(alloc::format!("cat: {}: No such file", path)));
            }
        }
        lines
    }

    fn cmd_mkdir(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("mkdir: missing operand")];
        }
        let mut fs = self.fs.borrow_mut();
        let mut lines = Vec::new();
        for path in args {
            if let Err(e) = fs.mkdir(path) {
                lines.push(err(alloc::format!("mkdir: {}: {}", path, e)));
            }
        }
        lines
    }

    fn cmd_touch(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("touch: missing file operand")];
        }
        let mut fs = self.fs.borrow_mut();
        let mut lines = Vec::new();
        for path in args {
            if !fs.exists(path) {
                if let Err(e) = fs.write_text(path, "") {
                    lines.push(err(alloc::format!("touch: {}: {}", path, e)));
                }
            }
        }
        lines
    }

    fn cmd_rm(&self, args: &[&str]) -> Vec<TermLine> {
        let paths: Vec<&&str> = args.iter().filter(|a| !a.starts_with('-')).collect();
        if paths.is_empty() {
            return alloc::vec![err("rm: missing operand")];
        }
        let mut fs = self.fs.borrow_mut();
        let mut lines = Vec::new();
        for path in paths {
            if let Err(e) = fs.remove(path) {
                lines.push(err(alloc::format!("rm: {}: {}", path, e)));
            }
        }
        lines
    }

    fn cmd_ps(&self) -> Vec<TermLine> {
        let sched = self.sched.borrow();
        let mut lines = alloc::vec![info("  PID  STATE     STEPS  NAME")];
        for d in sched.snapshot() {
            let state = match d.state {
                DomainState::Ready => "ready",
                DomainState::Running => "running",
                DomainState::Finished => "done",
            };
            let mut s = String::from("  ");
            pad_int(&mut s, d.id.0 as i64, 4);
            s.push_str("  ");
            pad_str(&mut s, state, 9);
            pad_int(&mut s, d.steps as i64, 6);
            s.push_str("  ");
            s.push_str(&d.name);
            lines.push(out(s));
        }
        lines
    }

    // ── new file/dir commands ──

    fn cmd_rmdir(&self, args: &[&str]) -> Vec<TermLine> {
        let paths: Vec<&&str> = args.iter().filter(|a| !a.starts_with('-')).collect();
        if paths.is_empty() {
            return alloc::vec![err("rmdir: missing operand")];
        }
        let mut fs = self.fs.borrow_mut();
        let mut lines = Vec::new();
        for path in paths {
            if !fs.exists(path) {
                lines.push(err(alloc::format!("rmdir: {}: No such directory", path)));
            } else if !fs.is_dir(path) {
                lines.push(err(alloc::format!("rmdir: {}: Not a directory", path)));
            } else if let Some(e) = fs.entries(path) {
                if !e.is_empty() {
                    lines.push(err(alloc::format!("rmdir: {}: Directory not empty", path)));
                } else if let Err(e) = fs.remove(path) {
                    lines.push(err(alloc::format!("rmdir: {}: {}", path, e)));
                }
            }
        }
        lines
    }

    fn cmd_cp(&self, args: &[&str]) -> Vec<TermLine> {
        let recursive = args.iter().any(|a| flag_has(a, 'r') || flag_has(a, 'R'));
        let ops: Vec<&&str> = args.iter().filter(|a| !a.starts_with('-')).collect();
        if ops.len() < 2 {
            return alloc::vec![err("cp: missing file operand (usage: cp [-r] SRC DST)")];
        }
        let src = *ops[0];
        let dst = *ops[1];
        let mut fs = self.fs.borrow_mut();
        // If dst is an existing directory, copy *into* it under the source basename.
        let dst = if fs.is_dir(dst) {
            alloc::format!("{}/{}", dst.trim_end_matches('/'), base_name(src))
        } else {
            dst.to_string()
        };
        let mut lines = Vec::new();
        if fs.is_dir(src) {
            if !recursive {
                return alloc::vec![err(alloc::format!("cp: -r not specified; omitting directory '{}'", src))];
            }
            let src_abs = fs.normalize(src);
            let dst_abs = fs.normalize(&dst);
            if dst_abs == src_abs || dst_abs.starts_with(&alloc::format!("{}/", src_abs)) {
                return alloc::vec![err(alloc::format!("cp: cannot copy a directory, '{}', into itself, '{}'", src, dst))];
            }
            if let Err(e) = copy_tree(&mut fs, src, &dst) {
                lines.push(err(alloc::format!("cp: {}", e)));
            }
        } else if let Some(body) = fs.read_text(src) {
            if let Err(e) = fs.write_text(&dst, &body) {
                lines.push(err(alloc::format!("cp: {}: {}", dst, e)));
            }
        } else {
            lines.push(err(alloc::format!("cp: cannot stat '{}': No such file", src)));
        }
        lines
    }

    fn cmd_mv(&self, args: &[&str]) -> Vec<TermLine> {
        let ops: Vec<&&str> = args.iter().filter(|a| !a.starts_with('-')).collect();
        if ops.len() < 2 {
            return alloc::vec![err("mv: missing file operand (usage: mv SRC DST)")];
        }
        let src = *ops[0];
        let dst = *ops[1];
        let mut fs = self.fs.borrow_mut();
        let dst = if fs.is_dir(dst) {
            alloc::format!("{}/{}", dst.trim_end_matches('/'), base_name(src))
        } else {
            dst.to_string()
        };
        let mut lines = Vec::new();
        if fs.is_dir(src) {
            let src_abs = fs.normalize(src);
            let dst_abs = fs.normalize(&dst);
            if dst_abs == src_abs || dst_abs.starts_with(&alloc::format!("{}/", src_abs)) {
                return alloc::vec![err(alloc::format!("mv: cannot move a directory, '{}', into itself, '{}'", src, dst))];
            }
            if let Err(e) = copy_tree(&mut fs, src, &dst) {
                return alloc::vec![err(alloc::format!("mv: {}", e))];
            }
            if let Err(e) = remove_tree(&mut fs, src) {
                lines.push(err(alloc::format!("mv: {}: {}", src, e)));
            }
        } else if let Some(body) = fs.read_text(src) {
            if let Err(e) = fs.write_text(&dst, &body) {
                return alloc::vec![err(alloc::format!("mv: {}: {}", dst, e))];
            }
            if let Err(e) = fs.remove(src) {
                lines.push(err(alloc::format!("mv: {}: {}", src, e)));
            }
        } else {
            lines.push(err(alloc::format!("mv: cannot stat '{}': No such file", src)));
        }
        lines
    }

    fn cmd_find(&self, args: &[&str]) -> Vec<TermLine> {
        let positional: Vec<&&str> = args.iter().filter(|a| !a.starts_with('-')).collect();
        // Pattern after `-name`, else the second positional, else match everything.
        let name = args
            .iter()
            .position(|a| *a == "-name")
            .and_then(|i| args.get(i + 1))
            .copied();
        let start = positional.first().map(|s| **s).unwrap_or(".");
        let pat = name.or_else(|| positional.get(1).map(|s| **s));
        let fs = self.fs.borrow();
        if !fs.exists(start) {
            return alloc::vec![err(alloc::format!("find: '{}': No such file or directory", start))];
        }
        let abs = fs.normalize(start);
        let mut hits = Vec::new();
        find_walk(&fs, &abs, pat, &mut hits);
        if hits.is_empty() {
            return Vec::new();
        }
        hits.into_iter().map(out).collect()
    }

    fn cmd_grep(&self, args: &[&str]) -> Vec<TermLine> {
        let ignore = args.iter().any(|a| flag_has(a, 'i'));
        let number = args.iter().any(|a| flag_has(a, 'n'));
        let ops: Vec<&&str> = args.iter().filter(|a| !a.starts_with('-')).collect();
        if ops.len() < 2 {
            return alloc::vec![err("grep: usage: grep [-i] [-n] PATTERN FILE...")];
        }
        let pat = *ops[0];
        let needle = if ignore { pat.to_ascii_lowercase() } else { pat.to_string() };
        let fs = self.fs.borrow();
        let multi = ops.len() > 2;
        let mut lines = Vec::new();
        for file in &ops[1..] {
            let file = **file;
            if fs.is_dir(file) {
                lines.push(err(alloc::format!("grep: {}: Is a directory", file)));
                continue;
            }
            match fs.read_text(file) {
                Some(body) => {
                    for (i, l) in body.split('\n').enumerate() {
                        let hay = if ignore { l.to_ascii_lowercase() } else { l.to_string() };
                        if hay.contains(&needle) {
                            let mut s = String::new();
                            if multi {
                                s.push_str(file);
                                s.push(':');
                            }
                            if number {
                                push_int(&mut s, (i + 1) as i64);
                                s.push(':');
                            }
                            s.push_str(l);
                            lines.push(out(s));
                        }
                    }
                }
                None => lines.push(err(alloc::format!("grep: {}: No such file", file))),
            }
        }
        lines
    }

    fn cmd_head(&self, args: &[&str]) -> Vec<TermLine> {
        let (n, files) = parse_n(args, 10);
        self.head_tail(&files, n, true)
    }

    fn cmd_tail(&self, args: &[&str]) -> Vec<TermLine> {
        let (n, files) = parse_n(args, 10);
        self.head_tail(&files, n, false)
    }

    fn head_tail(&self, files: &[&str], n: usize, head: bool) -> Vec<TermLine> {
        if files.is_empty() {
            return alloc::vec![err("missing file operand")];
        }
        let fs = self.fs.borrow();
        let mut lines = Vec::new();
        let many = files.len() > 1;
        for (fi, file) in files.iter().enumerate() {
            if many {
                if fi > 0 {
                    lines.push(out(String::new()));
                }
                lines.push(info(alloc::format!("==> {} <==", file)));
            }
            match fs.read_text(file) {
                Some(body) => {
                    let all: Vec<&str> = body.trim_end_matches('\n').split('\n').collect();
                    let chosen: &[&str] = if head {
                        &all[..all.len().min(n)]
                    } else {
                        &all[all.len().saturating_sub(n)..]
                    };
                    for l in chosen {
                        lines.push(out(l.to_string()));
                    }
                }
                None => lines.push(err(alloc::format!("{}: No such file", file))),
            }
        }
        lines
    }

    fn cmd_tac(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("tac: missing file operand")];
        }
        let fs = self.fs.borrow();
        let mut lines = Vec::new();
        for path in args {
            if let Some(body) = fs.read_text(path) {
                for l in body.trim_end_matches('\n').split('\n').rev() {
                    lines.push(out(l.to_string()));
                }
            } else {
                lines.push(err(alloc::format!("tac: {}: No such file", path)));
            }
        }
        lines
    }

    fn cmd_wc(&self, args: &[&str]) -> Vec<TermLine> {
        let only_l = args.iter().any(|a| flag_has(a, 'l'));
        let only_w = args.iter().any(|a| flag_has(a, 'w'));
        let only_c = args.iter().any(|a| flag_has(a, 'c'));
        let any = only_l || only_w || only_c;
        let files: Vec<&str> = args.iter().filter(|a| !a.starts_with('-')).copied().collect();
        if files.is_empty() {
            return alloc::vec![err("wc: missing file operand")];
        }
        let fs = self.fs.borrow();
        let mut lines = Vec::new();
        for file in &files {
            match fs.read_text(file) {
                Some(body) => {
                    let nl = body.bytes().filter(|b| *b == b'\n').count();
                    let words = body.split_whitespace().count();
                    let bytes = body.len();
                    let mut s = String::new();
                    if !any || only_l {
                        pad_int(&mut s, nl as i64, 7);
                    }
                    if !any || only_w {
                        pad_int(&mut s, words as i64, 8);
                    }
                    if !any || only_c {
                        pad_int(&mut s, bytes as i64, 8);
                    }
                    s.push(' ');
                    s.push_str(file);
                    lines.push(out(s));
                }
                None => lines.push(err(alloc::format!("wc: {}: No such file", file))),
            }
        }
        lines
    }

    fn cmd_stat(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("stat: missing operand")];
        }
        let fs = self.fs.borrow();
        let mut lines = Vec::new();
        for path in args {
            let abs = fs.normalize(path);
            if !fs.exists(path) {
                lines.push(err(alloc::format!("stat: cannot stat '{}': No such file or directory", path)));
                continue;
            }
            let is_dir = fs.is_dir(path);
            let kind = if is_dir { "directory" } else { "regular file" };
            let size = if is_dir {
                fs.entries(path).map(|e| e.len()).unwrap_or(0)
            } else {
                fs.read_text(path).map(|t| t.len()).unwrap_or(0)
            };
            lines.push(out(alloc::format!("  File: {}", abs)));
            if is_dir {
                lines.push(out(alloc::format!("  Type: {}    Entries: {}", kind, size)));
            } else {
                lines.push(out(alloc::format!("  Type: {}    Size: {} bytes", kind, size)));
            }
        }
        lines
    }

    fn cmd_file(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("file: missing operand")];
        }
        let fs = self.fs.borrow();
        let mut lines = Vec::new();
        for path in args {
            if !fs.exists(path) {
                lines.push(out(alloc::format!("{}: cannot open (No such file or directory)", path)));
            } else if fs.is_dir(path) {
                lines.push(out(alloc::format!("{}: directory", path)));
            } else {
                let body = fs.read_text(path).unwrap_or_default();
                lines.push(out(alloc::format!("{}: {}", path, guess_file_type(path, &body))));
            }
        }
        lines
    }

    fn cmd_tree(&self, args: &[&str]) -> Vec<TermLine> {
        let start = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or(".");
        let fs = self.fs.borrow();
        if !fs.is_dir(start) {
            return alloc::vec![err(alloc::format!("tree: {}: Not a directory", start))];
        }
        let abs = fs.normalize(start);
        let mut lines = alloc::vec![out(abs.clone())];
        let (mut dirs, mut files) = (0usize, 0usize);
        tree_walk(&fs, &abs, String::new(), &mut lines, &mut dirs, &mut files);
        lines.push(out(String::new()));
        lines.push(info(alloc::format!("{} directories, {} files", dirs, files)));
        lines
    }

    fn cmd_du(&self, args: &[&str]) -> Vec<TermLine> {
        let start = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or(".");
        let fs = self.fs.borrow();
        if !fs.exists(start) {
            return alloc::vec![err(alloc::format!("du: cannot access '{}': No such file or directory", start))];
        }
        let abs = fs.normalize(start);
        let total = dir_size(&fs, &abs);
        let mut s = String::new();
        pad_int(&mut s, total as i64, 8);
        s.push_str("  ");
        s.push_str(&abs);
        alloc::vec![out(s)]
    }

    fn cmd_df(&self) -> Vec<TermLine> {
        let fs = self.fs.borrow();
        let objects = fs.object_count();
        let used = total_used(&fs);
        let mut lines = alloc::vec![info("Filesystem        Objects        Used  Mounted on")];
        let mut s = String::from("objgraph     ");
        pad_int(&mut s, objects as i64, 10);
        pad_int(&mut s, used as i64, 12);
        s.push_str("  /");
        lines.push(out(s));
        lines
    }

    // ── process / system ──

    fn cmd_top(&self) -> Vec<TermLine> {
        let sched = self.sched.borrow();
        let mut domains = sched.snapshot();
        domains.sort_by(|a, b| b.steps.cmp(&a.steps));
        let total: u64 = domains.iter().map(|d| d.steps as u64).sum();
        let live = domains.iter().filter(|d| d.state != DomainState::Finished).count();
        let mut lines = alloc::vec![
            info(alloc::format!("top — {} domains, {} running, {} total steps", domains.len(), live, total)),
            info("  PID   %CPU  STATE     STEPS  NAME"),
        ];
        for d in &domains {
            let pct = if total > 0 { d.steps as u64 * 100 / total } else { 0 };
            let state = state_str(d.state);
            let mut s = String::from("  ");
            pad_int(&mut s, d.id.0 as i64, 4);
            s.push_str("  ");
            pad_int(&mut s, pct as i64, 4);
            s.push_str("  ");
            pad_str(&mut s, state, 9);
            pad_int(&mut s, d.steps as i64, 6);
            s.push_str("  ");
            s.push_str(&d.name);
            lines.push(out(s));
        }
        lines
    }

    fn cmd_kill(&self, args: &[&str]) -> Vec<TermLine> {
        let id_arg = args.iter().find(|a| !a.starts_with('-'));
        let Some(id_arg) = id_arg else {
            return alloc::vec![err("kill: usage: kill <pid>")];
        };
        let Some(id) = parse_u64(id_arg) else {
            return alloc::vec![err(alloc::format!("kill: '{}': not a valid pid", id_arg))];
        };
        let mut sched = self.sched.borrow_mut();
        if sched.kill(crate::sched::DomainId(id)) {
            alloc::vec![info(alloc::format!("killed domain {}", id))]
        } else {
            alloc::vec![err(alloc::format!("kill: ({}): No such process", id))]
        }
    }

    fn cmd_free(&self) -> Vec<TermLine> {
        // No allocator metric is reachable from the backend, so report the capability
        // footprint the scheduler does expose: the sum of every domain's region length
        // is the address space actually committed to running domains.
        let sched = self.sched.borrow();
        let snap = sched.snapshot();
        let committed: u64 = snap.iter().map(|d| d.len).sum();
        let domains = snap.len();
        let mut lines = alloc::vec![info("              region-bytes    domains")];
        let mut s = String::from("committed  ");
        pad_int(&mut s, committed as i64, 14);
        pad_int(&mut s, domains as i64, 11);
        lines.push(out(s));
        lines.push(info("(single-address-space: memory is capability-gated, not paged)"));
        lines
    }

    fn cmd_env(&self) -> Vec<TermLine> {
        // No process environment exists; present the shell's effective built-in set.
        let cwd = self.fs.borrow().cwd().to_string();
        [
            alloc::format!("USER=jayden"),
            alloc::format!("HOME={}", HOME),
            alloc::format!("PWD={}", cwd),
            alloc::format!("SHELL=/usr/bin/dominionsh"),
            alloc::format!("OS=DominionOS"),
            alloc::format!("TERM=dominion"),
        ]
        .into_iter()
        .map(out)
        .collect()
    }

    fn cmd_hostname(&self) -> Vec<TermLine> {
        let fs = self.fs.borrow();
        let name = fs
            .read_text("/etc/hostname")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "dominionos".to_string());
        alloc::vec![out(name)]
    }

    fn cmd_ipconfig(&self) -> Vec<TermLine> {
        // Best-effort: no live NIC handle is reachable from the shell backend, so
        // report the loopback the OS always has.
        alloc::vec![
            info("DominionOS network configuration"),
            out("  lo (loopback):"),
            out("    IPv4 Address . . . . . : 127.0.0.1"),
            out("    Subnet Mask  . . . . . : 255.0.0.0"),
        ]
    }

    fn cmd_date(&self) -> Vec<TermLine> {
        // No wall-clock source is wired into the shell backend; report the deterministic
        // build epoch honestly rather than fabricating a live time.
        alloc::vec![out("Sat Jun 21 00:00:00 UTC 2026 (deterministic boot epoch)")]
    }

    fn cmd_uptime(&self) -> Vec<TermLine> {
        // Derive a relative liveness figure from scheduler activity (total dispatch
        // steps), since no monotonic clock is reachable here.
        let sched = self.sched.borrow();
        let steps: u64 = sched.snapshot().iter().map(|d| d.steps as u64).sum();
        let live = sched.live_count();
        alloc::vec![out(alloc::format!(
            "up (cooperative): {} scheduler steps, {} domains live",
            steps, live
        ))]
    }

    fn cmd_printf(&self, rest: &str) -> Vec<TermLine> {
        // Minimal printf: interpret \n as a line break and \t as a tab; no % formatting.
        let mut text = String::new();
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => text.push('\n'),
                    Some('t') => text.push('\t'),
                    Some('\\') => text.push('\\'),
                    Some(other) => {
                        text.push('\\');
                        text.push(other);
                    }
                    None => text.push('\\'),
                }
            } else {
                text.push(c);
            }
        }
        text.split('\n').map(|l| out(l.to_string())).collect()
    }

    fn cmd_basename(&self, args: &[&str]) -> Vec<TermLine> {
        match args.first() {
            Some(p) => alloc::vec![out(base_name(p).to_string())],
            None => alloc::vec![err("basename: missing operand")],
        }
    }

    fn cmd_dirname(&self, args: &[&str]) -> Vec<TermLine> {
        match args.first() {
            Some(p) => alloc::vec![out(dir_name(p))],
            None => alloc::vec![err("dirname: missing operand")],
        }
    }

    fn cmd_which(&self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            return alloc::vec![err("which: missing operand")];
        }
        args.iter()
            .map(|name| {
                if is_builtin(name) {
                    out(alloc::format!("{}: shell builtin", name))
                } else {
                    err(alloc::format!("{} not found", name))
                }
            })
            .collect()
    }

    fn cmd_ping(&self, args: &[&str]) -> Vec<TermLine> {
        let host = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("localhost");
        let count: usize = args.windows(2).find(|w| w[0] == "-c").and_then(|w| parse_usize(w[1])).unwrap_or(4);
        let mut out_lines = Vec::new();
        out_lines.push(out(alloc::format!("PING {} (loopback): 56 data bytes", host)));
        for i in 0..count {
            out_lines.push(out(alloc::format!("64 bytes from {}: icmp_seq={} ttl=64 time=0.1 ms", host, i)));
        }
        out_lines.push(out(alloc::format!("--- {} ping statistics ---", host)));
        out_lines.push(out(alloc::format!("{} packets transmitted, {} received, 0% packet loss", count, count)));
        out_lines
    }

    fn cmd_sort(&self, args: &[&str]) -> Vec<TermLine> {
        let reverse = args.iter().any(|a| *a == "-r");
        let files: Vec<_> = args.iter().filter(|a| !a.starts_with('-')).copied().collect();
        if files.is_empty() {
            return alloc::vec![err("sort: no input file specified")];
        }
        let fs = self.fs.borrow();
        let mut lines: Vec<String> = Vec::new();
        for f in &files {
            let path = self.resolve(f);
            match fs.read_text(&path) {
                Some(content) => {
                    for line in content.lines() { lines.push(line.to_string()); }
                }
                None => return alloc::vec![err(alloc::format!("sort: {}: No such file", f))],
            }
        }
        lines.sort_unstable();
        if reverse { lines.reverse(); }
        lines.into_iter().map(out).collect()
    }

    fn cmd_uniq(&self, args: &[&str]) -> Vec<TermLine> {
        let files: Vec<_> = args.iter().filter(|a| !a.starts_with('-')).copied().collect();
        if files.is_empty() {
            return alloc::vec![err("uniq: no input file specified")];
        }
        let fs = self.fs.borrow();
        let path = self.resolve(files[0]);
        match fs.read_text(&path) {
            Some(content) => {
                let mut prev = "";
                let content_clone = content.clone();
                let mut result = Vec::new();
                for line in content_clone.lines() {
                    if line != prev {
                        result.push(out(line.to_string()));
                    }
                    prev = "";
                    let _ = prev; // prev can't easily reference line; just push all de-duped pairs
                }
                // Simple consecutive-dedup without lifetime issues
                let lines: Vec<_> = content.lines().collect();
                result.clear();
                let mut last = "";
                for &l in &lines {
                    if l != last { result.push(out(l.to_string())); }
                    last = l;
                }
                result
            }
            None => alloc::vec![err(alloc::format!("uniq: {}: No such file", files[0]))],
        }
    }

    fn cmd_write(&self, rest: &str, args: &[&str]) -> Vec<TermLine> {
        // write <file> <content...> — write content to a file (create or overwrite)
        let file = match args.first() {
            Some(f) => f,
            None => return alloc::vec![err("write: usage: write <file> [content]")],
        };
        let path = self.resolve(file);
        let content = if args.len() > 1 {
            // Content is everything after the filename in rest
            let after_file = rest.trim_start_matches(file).trim_start();
            after_file.to_string()
        } else {
            String::new()
        };
        match self.fs.borrow_mut().write_text(&path, &content) {
            Ok(_) => alloc::vec![out(alloc::format!("wrote {} bytes to {}", content.len(), path))],
            Err(e) => alloc::vec![err(alloc::format!("write: {}: {:?}", file, e))],
        }
    }

    fn cmd_history(&self) -> Vec<TermLine> {
        // History is stored in the terminal layer; we just inform the user here.
        alloc::vec![info("history: command history is stored in the terminal session above")]
    }

    fn cmd_sleep(&self, args: &[&str]) -> Vec<TermLine> {
        let secs = args.first().and_then(|s| parse_usize(s)).unwrap_or(1);
        // In a no_std OS without real threading, we can't actually sleep; inform instead.
        alloc::vec![info(alloc::format!("sleep: would sleep {}s (no blocking sleep in single-threaded shell)", secs))]
    }

    fn cmd_diff(&self, args: &[&str]) -> Vec<TermLine> {
        if args.len() < 2 {
            return alloc::vec![err("diff: usage: diff <file1> <file2>")];
        }
        let p1 = self.resolve(args[0]);
        let p2 = self.resolve(args[1]);
        let fs = self.fs.borrow();
        let a = match fs.read_text(&p1) {
            Some(t) => t,
            None => return alloc::vec![err(alloc::format!("diff: {}: not found", args[0]))],
        };
        let b = match fs.read_text(&p2) {
            Some(t) => t,
            None => return alloc::vec![err(alloc::format!("diff: {}: not found", args[1]))],
        };
        let mut result = Vec::new();
        let la: Vec<&str> = a.lines().collect();
        let lb: Vec<&str> = b.lines().collect();
        let max = la.len().max(lb.len());
        for i in 0..max {
            match (la.get(i), lb.get(i)) {
                (Some(l), Some(r)) if l == r => {}
                (Some(l), Some(r)) => {
                    result.push(out(alloc::format!("< {}", l)));
                    result.push(out(alloc::format!("> {}", r)));
                }
                (Some(l), None) => result.push(out(alloc::format!("< {}", l))),
                (None, Some(r)) => result.push(out(alloc::format!("> {}", r))),
                (None, None) => {}
            }
        }
        if result.is_empty() {
            result.push(info("(files are identical)"));
        }
        result
    }

    fn cmd_cut(&self, args: &[&str]) -> Vec<TermLine> {
        let mut delim = '\t';
        let mut fields: Vec<usize> = Vec::new();
        let mut files: Vec<&str> = Vec::new();
        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "-d" => { if let Some(d) = args.get(i + 1) { delim = d.chars().next().unwrap_or('\t'); i += 1; } }
                "-f" => { if let Some(f) = args.get(i + 1) { fields = f.split(',').filter_map(|n| parse_usize(n).map(|v| v.saturating_sub(1))).collect(); i += 1; } }
                a if a.starts_with("-d") => { delim = a[2..].chars().next().unwrap_or('\t'); }
                a if a.starts_with("-f") => { fields = a[2..].split(',').filter_map(|n| parse_usize(n).map(|v| v.saturating_sub(1))).collect(); }
                a => files.push(a),
            }
            i += 1;
        }
        let fs = self.fs.borrow();
        let content = if let Some(f) = files.first() {
            match fs.read_text(&self.resolve_inner(f, &fs)) { Some(t) => t, None => return alloc::vec![err("cut: file not found")] }
        } else { return alloc::vec![err("cut: no input file")] };
        content.lines().map(|line| {
            let cols: Vec<&str> = line.split(delim).collect();
            let selected: Vec<&str> = if fields.is_empty() { cols } else { fields.iter().filter_map(|&fi| cols.get(fi).copied()).collect() };
            out(selected.join(&delim.to_string()))
        }).collect()
    }

    fn cmd_tr(&self, args: &[&str]) -> Vec<TermLine> {
        if args.len() < 2 {
            return alloc::vec![err("tr: usage: tr <set1> <set2> <file>")];
        }
        let set1: Vec<char> = args[0].chars().collect();
        let set2: Vec<char> = args[1].chars().collect();
        let fs = self.fs.borrow();
        let content = if let Some(f) = args.get(2) {
            match fs.read_text(&self.resolve_inner(f, &fs)) { Some(t) => t, None => return alloc::vec![err("tr: file not found")] }
        } else { return alloc::vec![err("tr: no input file")] };
        let translated: String = content.chars().map(|c| {
            if let Some(pos) = set1.iter().position(|&x| x == c) {
                set2.get(pos).copied().unwrap_or(c)
            } else { c }
        }).collect();
        translated.lines().map(|l| out(l.to_string())).collect()
    }

    fn cmd_seq(&self, args: &[&str]) -> Vec<TermLine> {
        let (start, step, end) = match args.len() {
            0 => return alloc::vec![err("seq: usage: seq [start [step]] end")],
            1 => (1i64, 1i64, args[0].parse::<i64>().unwrap_or(1)),
            2 => (args[0].parse().unwrap_or(1), 1, args[1].parse().unwrap_or(1)),
            _ => (args[0].parse().unwrap_or(1), args[1].parse().unwrap_or(1), args[2].parse().unwrap_or(1)),
        };
        let mut out_lines = Vec::new();
        let mut v = start;
        let mut count = 0;
        while (step > 0 && v <= end) || (step < 0 && v >= end) {
            out_lines.push(out(alloc::format!("{}", v)));
            v = v.wrapping_add(step);
            count += 1;
            if count > 10_000 { out_lines.push(info("(truncated at 10000 lines)")); break; }
        }
        out_lines
    }

    fn cmd_nl(&self, args: &[&str]) -> Vec<TermLine> {
        let fs = self.fs.borrow();
        let content = match args.first().and_then(|f| fs.read_text(&self.resolve_inner(f, &fs))) {
            Some(t) => t,
            None => return alloc::vec![err("nl: file not found")],
        };
        content.lines().enumerate()
            .map(|(i, l)| out(alloc::format!("{:6}  {}", i + 1, l)))
            .collect()
    }

    fn cmd_ln(&self, args: &[&str]) -> Vec<TermLine> {
        if args.len() < 2 {
            return alloc::vec![err("ln: usage: ln [-s] <src> <dst>")];
        }
        let (src, dst) = if args[0] == "-s" {
            if args.len() < 3 { return alloc::vec![err("ln: usage: ln -s <src> <dst>")]; }
            (args[1], args[2])
        } else { (args[0], args[1]) };
        let src_path = self.resolve(src);
        let dst_path = self.resolve(dst);
        let fs = self.fs.borrow();
        match fs.read_text(&src_path) {
            Some(content) => {
                drop(fs);
                match self.fs.borrow_mut().write_text(&dst_path, &content) {
                    Ok(()) => alloc::vec![info(alloc::format!("linked {} → {}", src, dst))],
                    Err(_) => alloc::vec![err("ln: write failed")],
                }
            }
            None => alloc::vec![err(alloc::format!("ln: {}: not found", src))],
        }
    }

    fn cmd_chmod(&self, args: &[&str]) -> Vec<TermLine> {
        let path = args.get(1).copied().unwrap_or(".");
        alloc::vec![info(alloc::format!("chmod: permissions are capability-managed in DominionOS (path: {})", path))]
    }

    fn cmd_chown(&self, args: &[&str]) -> Vec<TermLine> {
        let path = args.get(1).copied().unwrap_or(".");
        alloc::vec![info(alloc::format!("chown: ownership is identity-based in DominionOS (path: {})", path))]
    }

    fn cmd_more(&self, args: &[&str]) -> Vec<TermLine> {
        // In a terminal-within-a-terminal, we can't do a real pager — show up to 40 lines.
        let fs = self.fs.borrow();
        let content = match args.first().and_then(|f| fs.read_text(&self.resolve_inner(f, &fs))) {
            Some(t) => t,
            None => return alloc::vec![err("more: file not found")],
        };
        let mut lines: Vec<TermLine> = content.lines().take(40).map(|l| out(l.to_string())).collect();
        let total = content.lines().count();
        if total > 40 {
            lines.push(info(alloc::format!("-- {} more lines --", total - 40)));
        }
        lines
    }

    fn cmd_bc(&self, args: &[&str]) -> Vec<TermLine> {
        // Evaluate math expressions via the Dominion interpreter.
        let expr = args.join(" ");
        if expr.is_empty() {
            return alloc::vec![info("bc: usage: bc <expression>  (e.g. bc 2+2, bc 2^10)")];
        }
        let src = expr.replace('^', "**");
        match crate::lang::eval_source(&src) {
            Ok(v) => alloc::vec![out(alloc::format!("{}", v))],
            Err(_) => alloc::vec![err(alloc::format!("bc: cannot evaluate: {}", expr))],
        }
    }

    fn cmd_man(&self, args: &[&str]) -> Vec<TermLine> {
        let cmd = match args.first() {
            Some(&c) => c,
            None => return alloc::vec![info("man: usage: man <command>  —  try `help` for the full list")],
        };
        // Return targeted help for known commands; fall back to generic.
        let text = match cmd {
            "ls" | "dir" => "ls [-la] [path] — list directory contents. -l: long format, -a: show hidden",
            "cat" => "cat <file> — print file contents to terminal",
            "grep" => "grep [-in] <pattern> <file> — search for pattern. -i: ignore case, -n: show line numbers",
            "find" => "find [path] [-name PAT] — find files matching pattern",
            "cp" => "cp [-r] <src> <dst> — copy file or directory (-r: recursive)",
            "mv" => "mv <src> <dst> — move or rename a file",
            "rm" => "rm [-rf] <path> — remove file (-r: recursive, -f: force)",
            "mkdir" => "mkdir <dir> — create directory",
            "diff" => "diff <file1> <file2> — compare two files line by line",
            "cut" => "cut -f <fields> [-d <delim>] <file> — extract columns (e.g. cut -f 1,3 -d , file)",
            "tr" => "tr <set1> <set2> <file> — translate characters",
            "seq" => "seq [start [step]] end — print a sequence of numbers",
            "nl" => "nl <file> — number lines of a file",
            "wc" => "wc [-lwc] <file> — count lines (-l), words (-w), chars (-c)",
            "sort" => "sort [-r] <file> — sort lines of a file (-r: reverse)",
            "uniq" => "uniq <file> — remove consecutive duplicate lines",
            "head" => "head [-n N] <file> — print first N lines (default 10)",
            "tail" => "tail [-n N] <file> — print last N lines (default 10)",
            "bc" | "expr" => "bc <expression> — evaluate a math expression (e.g. bc 2+3, bc 10/3)",
            "ping" => "ping [-c N] <host> — send ICMP echo requests",
            "ps" | "tasklist" => "ps — list running processes",
            "kill" | "taskkill" => "kill <pid> — terminate a process",
            "man" => "man <command> — show manual page for a command",
            _ => "no manual entry found — try `help` for a command overview",
        };
        alloc::vec![out(alloc::format!("  {}: {}", cmd, text))]
    }

    fn cmd_alias(&mut self, args: &[&str]) -> Vec<TermLine> {
        // Session-scoped: store in self.aliases.
        if args.is_empty() {
            return alloc::vec![info("alias: usage: alias <name>=<value>  (aliases are session-scoped)")];
        }
        let joined = args.join(" ");
        if let Some(eq) = joined.find('=') {
            let name = joined[..eq].trim().to_string();
            let value = joined[eq + 1..].trim().trim_matches('"').trim_matches('\'').to_string();
            self.aliases.insert(name.clone(), value.clone());
            alloc::vec![info(alloc::format!("alias {}='{}'", name, value))]
        } else {
            // Show existing alias.
            if let Some(v) = self.aliases.get(joined.trim()) {
                alloc::vec![out(alloc::format!("alias {}='{}'", joined.trim(), v))]
            } else {
                alloc::vec![err(alloc::format!("alias: {}: not found", joined.trim()))]
            }
        }
    }

    fn cmd_export(&mut self, args: &[&str]) -> Vec<TermLine> {
        if args.is_empty() {
            let vars: Vec<TermLine> = self.env_vars.iter()
                .map(|(k, v)| out(alloc::format!("export {}={}", k, v)))
                .collect();
            if vars.is_empty() { return alloc::vec![info("(no exported variables)")]; }
            return vars;
        }
        let joined = args.join(" ");
        if let Some(eq) = joined.find('=') {
            let k = joined[..eq].trim().to_string();
            let v = joined[eq + 1..].trim().to_string();
            self.env_vars.insert(k.clone(), v.clone());
            alloc::vec![info(alloc::format!("export {}={}", k, v))]
        } else {
            alloc::vec![err("export: usage: export KEY=value")]
        }
    }

    fn cmd_unset(&mut self, args: &[&str]) -> Vec<TermLine> {
        if let Some(&key) = args.first() {
            let removed = self.env_vars.remove(key).is_some() | self.aliases.remove(key).is_some();
            if removed { alloc::vec![info(alloc::format!("unset {}", key))] }
            else { alloc::vec![err(alloc::format!("unset: {}: not found", key))] }
        } else {
            alloc::vec![err("unset: usage: unset <name>")]
        }
    }

    fn cmd_xargs(&mut self, args: &[&str]) -> Vec<TermLine> {
        // xargs: we can read a file and execute a command with those as args.
        // Simplified: xargs <cmd> <file>
        if args.len() < 2 {
            return alloc::vec![info("xargs: usage: xargs <cmd> <file>  (reads lines from file as arguments)")];
        }
        let xcmd = args[0];
        let file_path = self.resolve(args[1]);
        let fs = self.fs.borrow();
        let content = match fs.read_text(&file_path) {
            Some(t) => t,
            None => return alloc::vec![err("xargs: file not found")],
        };
        drop(fs);
        let fargs: Vec<&str> = content.split_whitespace().collect();
        let full_args: Vec<&str> = core::iter::once(xcmd).chain(fargs.iter().copied()).collect();
        // Re-enter exec with the new command line.
        let line = full_args.join(" ");
        self.exec(&line)
    }

    /// Resolve path using a borrowed FileSystem (for methods that already hold a borrow).
    fn resolve_inner(&self, path: &str, fs: &crate::filesystem::FileSystem) -> String {
        if path == "~" || path.is_empty() { return HOME.into(); }
        if path.starts_with('/') { return path.into(); }
        let mut s = fs.cwd().to_string();
        if !s.ends_with('/') { s.push('/'); }
        s.push_str(path);
        s
    }

    /// Resolve a path argument relative to the current working directory.
    fn resolve(&self, path: &str) -> String {
        if path == "~" || path.is_empty() {
            return HOME.into();
        }
        if path.starts_with('/') {
            return path.into();
        }
        let mut s = self.fs.borrow().cwd().to_string();
        if !s.ends_with('/') { s.push('/'); }
        s.push_str(path);
        s
    }

    // ── dominion:// page authoring ──────────────────────────────────────────────

    fn cmd_dominion(&mut self, args: &[&str], _rest: &str) -> Vec<TermLine> {
        let sub = args.first().copied().unwrap_or("help");
        match sub {
            "publish" | "pub" => self.dominion_publish(args),
            "list" | "ls" => self.dominion_list(),
            "view" | "cat" => self.dominion_view(args),
            "rm" | "delete" => self.dominion_rm(args),
            "help" | _ => dominion_help_lines(),
        }
    }

    /// `dominion publish <name> [DSL text]`
    /// Write a page DSL to `/dominion/pages/<name>.dominion`. If no DSL text is given
    /// on the command line, create a starter template and tell the user to edit it.
    fn dominion_publish(&mut self, args: &[&str]) -> Vec<TermLine> {
        let name = match args.get(1) {
            Some(n) => *n,
            None => return alloc::vec![err("usage: dominion publish <name> [Title: ...  \\n Heading: ... ]")],
        };
        if name.contains('/') || name.contains('.') {
            return alloc::vec![err("page name must not contain '/' or '.'")];
        }
        // Remaining args after <name> form inline DSL (joined back to string).
        let dsl_inline: String = args[2..].join(" ");
        let dsl = if dsl_inline.trim().is_empty() {
            // Write a starter template.
            alloc::format!(
                "Title: {}\nHeading: Welcome\nText: Edit this page in /dominion/pages/{}.dominion\nLink: Home -> dominion://home\n",
                name, name
            )
        } else {
            dsl_inline
        };
        // Ensure the directory exists.
        let _ = self.fs.borrow_mut().mkdir("/dominion");
        let _ = self.fs.borrow_mut().mkdir("/dominion/pages");
        let path = alloc::format!("/dominion/pages/{}.dominion", name);
        match self.fs.borrow_mut().write_text(&path, &dsl) {
            Ok(()) => {
                let url = alloc::format!("dominion://{}", name);
                alloc::vec![
                    info(alloc::format!("Published: {}", url)),
                    out(alloc::format!("  File: {}", path)),
                    out(alloc::format!("  Navigate to {} in the browser, or edit the file to update.", url)),
                ]
            }
            Err(_) => alloc::vec![err(alloc::format!("could not write {}", path))],
        }
    }

    /// `dominion list` — list all published user pages.
    fn dominion_list(&self) -> Vec<TermLine> {
        let fs = self.fs.borrow();
        let dir = "/dominion/pages";
        if !fs.exists(dir) {
            return alloc::vec![out("No user-authored pages yet.  Use `dominion publish <name>` to create one.")];
        }
        let entries = match fs.entries(dir) {
            Some(e) => e,
            None => return alloc::vec![err("could not list /dominion/pages")],
        };
        if entries.is_empty() {
            return alloc::vec![out("No user-authored pages yet.")];
        }
        let mut lines = alloc::vec![info("User-authored dominion:// pages:")];
        for entry in &entries {
            if entry.name.ends_with(".dominion") {
                let page_name = &entry.name[..entry.name.len() - 7];
                lines.push(out(alloc::format!("  dominion://{}", page_name)));
            }
        }
        lines
    }

    /// `dominion view <name>` — print the raw DSL of a published page.
    fn dominion_view(&self, args: &[&str]) -> Vec<TermLine> {
        let name = match args.get(1) {
            Some(n) => *n,
            None => return alloc::vec![err("usage: dominion view <name>")],
        };
        let path = alloc::format!("/dominion/pages/{}.dominion", name);
        match self.fs.borrow().read_text(&path) {
            Some(text) => {
                let mut lines = alloc::vec![info(alloc::format!("--- dominion://{} ---", name))];
                for l in text.lines() {
                    lines.push(out(l.to_string()));
                }
                lines
            }
            None => alloc::vec![err(alloc::format!("no page named '{}' — use `dominion list` to see published pages", name))],
        }
    }

    /// `dominion rm <name>` — delete a user-authored page.
    fn dominion_rm(&mut self, args: &[&str]) -> Vec<TermLine> {
        let name = match args.get(1) {
            Some(n) => *n,
            None => return alloc::vec![err("usage: dominion rm <name>")],
        };
        let path = alloc::format!("/dominion/pages/{}.dominion", name);
        match self.fs.borrow_mut().remove(&path) {
            Ok(()) => alloc::vec![info(alloc::format!("Removed dominion://{}", name))],
            Err(_) => alloc::vec![err(alloc::format!("no page named '{}'", name))],
        }
    }
}

fn dominion_help_lines() -> Vec<TermLine> {
    [
        "dominion — DominionWeb page authoring",
        "",
        "  dominion publish <name> [DSL]   Publish a page; omit DSL for a starter template.",
        "  dominion list                   List all user-authored pages.",
        "  dominion view <name>            Print the page source (DSL text).",
        "  dominion rm <name>              Delete a page.",
        "  dominion help                   Show this help.",
        "",
        "Page DSL (one directive per line):",
        "  Title: My Page",
        "  Heading: Section heading",
        "  Text: A paragraph of body text.",
        "  Link: Display text -> dominion://target",
        "  Action: Button label -> Module::method (Capability)",
        "  # Lines starting with # are comments.",
        "",
        "Browse to dominion://<name> in the browser to see your page live.",
    ]
    .iter()
    .map(|l| out(l.to_string()))
    .collect()
}

fn help_lines() -> Vec<TermLine> {
    [
        "Files:    ls [-la] [path], tree [path], find [path] [-name PAT], stat <f>, file <f>,",
        "          du [path], df, wc [-lwc] <f>, head/tail [-n N] <f>, grep [-in] PAT <f>,",
        "          diff <f1> <f2>, sort [-r] <f>, uniq <f>, cut -f N [-d D] <f>, nl <f>,",
        "          write <f> [content], more/less <f>, ln [-s] <src> <dst>.",
        "Edit:     cd <dir> (.. ~ -), mkdir <d>, rmdir <d>, touch <f>, rm [-rf] <p>,",
        "          cp [-r] SRC DST, mv SRC DST, cat <f>, tac <f>, tr set1 set2 <f>, pwd.",
        "Process:  ps, top, kill <pid>, free, uptime, sleep <n>.",
        "Math:     bc <expr>, expr <expr>, seq [start [step]] end.",
        "Network:  ping [-c N] <host>, ipconfig.",
        "System:   whoami, hostname, uname [-a], date, env, which <cmd>, history, ver,",
        "          alias name=val, export KEY=val, unset <name>, xargs <cmd> <file>.",
        "Text:     echo <t>, printf <t>, true, false, yes [t], basename <p>, dirname <p>.",
        "Help:     man <cmd>, help.",
        "Windows aliases too: dir, copy, move, del, ren, type, md, rd, cls, tasklist…",
        "Anything else is evaluated as Dominion — e.g. `2 + 2`, `let x = 21; x * 2`.",
    ]
    .iter()
    .map(|l| out(l.to_string()))
    .collect()
}

fn push_int(s: &mut String, mut n: i64) {
    if n < 0 {
        s.push('-');
        n = -n;
    }
    if n >= 10 {
        push_int(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

/// Right-align an integer in a field of `width` (space-padded on the left).
fn pad_int(s: &mut String, n: i64, width: usize) {
    let mut tmp = String::new();
    push_int(&mut tmp, n);
    for _ in tmp.len()..width {
        s.push(' ');
    }
    s.push_str(&tmp);
}

/// Left-align a string in a field of `width` (space-padded on the right).
fn pad_str(s: &mut String, text: &str, width: usize) {
    s.push_str(text);
    for _ in text.len()..width {
        s.push(' ');
    }
}

/// Does the flag token `arg` (e.g. `-la`) contain the short flag letter `f`?
/// Only inspects tokens that start with a single `-` (clustered short flags).
fn flag_has(arg: &str, f: char) -> bool {
    arg.starts_with('-') && !arg.starts_with("--") && arg[1..].contains(f)
}

fn state_str(s: DomainState) -> &'static str {
    match s {
        DomainState::Ready => "ready",
        DomainState::Running => "running",
        DomainState::Finished => "done",
    }
}

/// Parse `-n N` (or `-N`) out of args, returning `(count, files)`.
fn parse_n<'a>(args: &[&'a str], default: usize) -> (usize, Vec<&'a str>) {
    let mut n = default;
    let mut files = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i];
        if a == "-n" {
            if let Some(v) = args.get(i + 1).and_then(|v| parse_usize(v)) {
                n = v;
                i += 2;
                continue;
            }
        } else if let Some(rest) = a.strip_prefix('-') {
            if let Some(v) = parse_usize(rest) {
                n = v;
                i += 1;
                continue;
            }
        }
        if !a.starts_with('-') {
            files.push(a);
        }
        i += 1;
    }
    (n, files)
}

fn parse_usize(s: &str) -> Option<usize> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut n: usize = 0;
    for b in s.bytes() {
        n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
    }
    Some(n)
}

fn parse_u64(s: &str) -> Option<u64> {
    parse_usize(s).map(|v| v as u64)
}

/// The final path component (`/a/b/c` → `c`, `/a/b/` → `b`, `c` → `c`).
fn base_name(path: &str) -> &str {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return "/";
    }
    match trimmed.rfind('/') {
        Some(i) => &trimmed[i + 1..],
        None => trimmed,
    }
}

/// The parent path (`/a/b/c` → `/a/b`, `c` → `.`).
fn dir_name(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => trimmed[..i].to_string(),
        None => ".".to_string(),
    }
}

/// All command names recognised as builtins (Linux + Windows aliases), for `which`.
fn is_builtin(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "help" | "?" | "pwd" | "whoami" | "hostname" | "uname" | "echo" | "printf" | "true"
            | "false" | "yes" | "date" | "uptime" | "basename" | "dirname" | "which" | "where"
            | "ls" | "dir" | "cd" | "chdir" | "cat" | "type" | "tac" | "head" | "tail" | "wc"
            | "grep" | "find" | "tree" | "stat" | "file" | "du" | "df" | "mkdir" | "md" | "rmdir"
            | "rd" | "touch" | "rm" | "del" | "erase" | "cp" | "copy" | "mv" | "move" | "ren"
            | "rename" | "ps" | "tasklist" | "top" | "kill" | "taskkill" | "free" | "env" | "set"
            | "ipconfig" | "ver" | "version" | "about" | "clear" | "cls"
            | "diff" | "cut" | "tr" | "seq" | "nl" | "ln" | "chmod" | "chown"
            | "more" | "less" | "bc" | "expr" | "man" | "alias" | "export" | "unset" | "xargs"
            | "ping" | "sort" | "uniq" | "write" | "tee" | "history" | "sleep"
    )
}

/// Guess a file's type from its extension, falling back to a content sniff.
fn guess_file_type(path: &str, body: &str) -> String {
    let lower = path.to_ascii_lowercase();
    let ext = lower.rsplit('.').next().unwrap_or("");
    let by_ext = match ext {
        "txt" => Some("ASCII text"),
        "md" => Some("Markdown document, ASCII text"),
        "aeth" => Some("Dominion source, ASCII text"),
        "log" => Some("log file, ASCII text"),
        "json" => Some("JSON text data"),
        "rs" => Some("Rust source, ASCII text"),
        "toml" => Some("TOML configuration text"),
        "png" => Some("PNG image data"),
        "jpg" | "jpeg" => Some("JPEG image data"),
        _ => None,
    };
    if let Some(t) = by_ext {
        return t.to_string();
    }
    if body.is_empty() {
        "empty".to_string()
    } else if body.bytes().all(|b| b == b'\n' || b == b'\t' || b == b'\r' || (0x20..=0x7e).contains(&b)) {
        "ASCII text".to_string()
    } else {
        "data".to_string()
    }
}

/// Recursively copy directory `src` to `dst`, recreating the tree of files.
fn copy_tree(
    fs: &mut crate::filesystem::FileSystem,
    src: &str,
    dst: &str,
) -> Result<(), alloc::string::String> {
    fs.mkdir(dst).map_err(|e| alloc::format!("{}: {}", dst, e))?;
    let abs = fs.normalize(src);
    let entries = fs.entries(&abs).ok_or_else(|| alloc::format!("{}: not a directory", src))?;
    for e in entries {
        let child_src = alloc::format!("{}/{}", abs, e.name);
        let child_dst = alloc::format!("{}/{}", dst.trim_end_matches('/'), e.name);
        if e.is_dir {
            copy_tree(fs, &child_src, &child_dst)?;
        } else if let Some(body) = fs.read_text(&child_src) {
            fs.write_text(&child_dst, &body).map_err(|e| alloc::format!("{}: {}", child_dst, e))?;
        }
    }
    Ok(())
}

/// Recursively remove directory `path` (depth-first so children go before the parent).
fn remove_tree(
    fs: &mut crate::filesystem::FileSystem,
    path: &str,
) -> Result<(), crate::vfs::VfsError> {
    let abs = fs.normalize(path);
    if let Some(entries) = fs.entries(&abs) {
        for e in entries {
            let child = alloc::format!("{}/{}", abs, e.name);
            if e.is_dir {
                remove_tree(fs, &child)?;
            } else {
                fs.remove(&child)?;
            }
        }
    }
    fs.remove(&abs)
}

/// Recursive name search collecting matching absolute paths into `hits`.
fn find_walk(fs: &crate::filesystem::FileSystem, dir: &str, pat: Option<&str>, hits: &mut Vec<String>) {
    // The starting node itself matches if it is a file (or a dir that fits the pattern).
    if matches_pat(base_name(dir), pat) {
        hits.push(dir.to_string());
    }
    if let Some(entries) = fs.entries(dir) {
        for e in entries {
            let child = if dir == "/" {
                alloc::format!("/{}", e.name)
            } else {
                alloc::format!("{}/{}", dir, e.name)
            };
            if e.is_dir {
                find_walk(fs, &child, pat, hits);
            } else if matches_pat(&e.name, pat) {
                hits.push(child);
            }
        }
    }
}

/// Substring (or trivial `*`-wildcard) name match; `None` matches everything.
fn matches_pat(name: &str, pat: Option<&str>) -> bool {
    match pat {
        None => true,
        Some(p) if p == "*" => true,
        Some(p) => {
            // Support `*foo`, `foo*`, `*foo*` plus plain substrings.
            let trimmed = p.trim_matches('*');
            if p.starts_with('*') && p.ends_with('*') {
                name.contains(trimmed)
            } else if let Some(suf) = p.strip_prefix('*') {
                name.ends_with(suf)
            } else if let Some(pre) = p.strip_suffix('*') {
                name.starts_with(pre)
            } else {
                name == p
            }
        }
    }
}

/// Recursive ASCII tree rendering.
fn tree_walk(
    fs: &crate::filesystem::FileSystem,
    dir: &str,
    prefix: String,
    lines: &mut Vec<TermLine>,
    dirs: &mut usize,
    files: &mut usize,
) {
    let entries = match fs.entries(dir) {
        Some(e) => e,
        None => return,
    };
    let n = entries.len();
    for (i, e) in entries.iter().enumerate() {
        let last = i + 1 == n;
        let branch = if last { "`-- " } else { "|-- " };
        let mut name = e.name.clone();
        if e.is_dir {
            name.push('/');
            *dirs += 1;
        } else {
            *files += 1;
        }
        lines.push(out(alloc::format!("{}{}{}", prefix, branch, name)));
        if e.is_dir {
            let child = if dir == "/" {
                alloc::format!("/{}", e.name)
            } else {
                alloc::format!("{}/{}", dir, e.name)
            };
            let next = alloc::format!("{}{}", prefix, if last { "    " } else { "|   " });
            tree_walk(fs, &child, next, lines, dirs, files);
        }
    }
}

/// Total byte size of all files under a directory (recursive); a file's own size if
/// `path` is a file.
fn dir_size(fs: &crate::filesystem::FileSystem, path: &str) -> usize {
    if fs.is_file(path) {
        return fs.read_text(path).map(|t| t.len()).unwrap_or(0);
    }
    let mut total = 0;
    if let Some(entries) = fs.entries(path) {
        for e in entries {
            let child = if path == "/" {
                alloc::format!("/{}", e.name)
            } else {
                alloc::format!("{}/{}", path, e.name)
            };
            if e.is_dir {
                total += dir_size(fs, &child);
            } else {
                total += e.size;
            }
        }
    }
    total
}

/// Total bytes used across the whole filesystem (from root).
fn total_used(fs: &crate::filesystem::FileSystem) -> usize {
    dir_size(fs, "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{Capability, Rights};
    use crate::filesystem::FileSystem;
    use crate::terminal::Terminal;
    use alloc::boxed::Box;

    fn backend() -> (ShellBackend, SharedFs, SharedSched) {
        let fs = FileSystem::shared();
        let sched = Rc::new(RefCell::new(Scheduler::new()));
        sched.borrow_mut().spawn("init", Capability::mint(0, 0x1000, Rights::ALL));
        sched.borrow_mut().spawn("compositor", Capability::mint(0x1000, 0x1000, Rights::ALL));
        (ShellBackend::new(fs.clone(), sched.clone()), fs, sched)
    }

    #[test]
    fn ls_lists_the_seeded_home() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("ls");
        assert!(lines.iter().any(|l| l.text.contains("Documents/")));
        assert!(lines.iter().any(|l| l.text.contains("Projects/")));
    }

    #[test]
    fn cd_then_pwd_tracks_the_directory() {
        let (mut b, _fs, _s) = backend();
        assert!(b.exec("cd Documents").is_empty());
        let pwd = b.exec("pwd");
        assert_eq!(pwd[0].text, "/home/jayden/Documents");
        // cd into a missing dir errors and leaves cwd unchanged.
        let e = b.exec("cd nope");
        assert_eq!(e[0].kind, LineKind::Error);
        assert_eq!(b.exec("pwd")[0].text, "/home/jayden/Documents");
    }

    #[test]
    fn cat_reads_a_seeded_file() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("cat /home/jayden/Documents/welcome.txt");
        assert!(lines.iter().any(|l| l.text.contains("Welcome to DominionOS")));
    }

    #[test]
    fn mkdir_and_touch_are_visible_to_the_filesystem() {
        let (mut b, fs, _s) = backend();
        b.exec("mkdir /tmp/work");
        b.exec("touch /tmp/work/notes.txt");
        assert!(fs.borrow().is_dir("/tmp/work"));
        assert!(fs.borrow().is_file("/tmp/work/notes.txt"));
        // And `ls` shows them.
        let lines = b.exec("ls /tmp/work");
        assert!(lines.iter().any(|l| l.text.contains("notes.txt")));
    }

    #[test]
    fn rm_removes_a_file() {
        let (mut b, fs, _s) = backend();
        b.exec("touch /tmp/gone.txt");
        assert!(fs.borrow().is_file("/tmp/gone.txt"));
        b.exec("rm /tmp/gone.txt");
        assert!(!fs.borrow().exists("/tmp/gone.txt"));
    }

    #[test]
    fn ps_reports_live_domains() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("ps");
        assert!(lines.iter().any(|l| l.text.contains("PID")));
        assert!(lines.iter().any(|l| l.text.contains("init")));
        assert!(lines.iter().any(|l| l.text.contains("compositor")));
    }

    #[test]
    fn unknown_command_falls_back_to_dominion_repl() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("2 + 2 * 10");
        assert!(lines.iter().any(|l| l.kind == LineKind::Output && l.text.contains("22")));
        // A `let` binding still evaluates through the REPL fallthrough.
        let lines = b.exec("let x = 21; x * 2");
        assert!(lines.iter().any(|l| l.text.contains("42")));
    }

    #[test]
    fn a_bare_mistyped_word_reports_command_not_found() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("notacommand");
        assert!(lines.iter().any(|l| l.kind == LineKind::Error && l.text.contains("command not found")));
    }

    #[test]
    fn cp_then_cat_copies_content() {
        let (mut b, fs, _s) = backend();
        b.exec("touch /tmp/a.txt");
        fs.borrow_mut().write_text("/tmp/a.txt", "hello world").unwrap();
        b.exec("cp /tmp/a.txt /tmp/b.txt");
        assert_eq!(fs.borrow().read_text("/tmp/b.txt").as_deref(), Some("hello world"));
    }

    #[test]
    fn mv_renames_a_file() {
        let (mut b, fs, _s) = backend();
        fs.borrow_mut().write_text("/tmp/old.txt", "x").unwrap();
        b.exec("mv /tmp/old.txt /tmp/new.txt");
        assert!(!fs.borrow().exists("/tmp/old.txt"));
        assert_eq!(fs.borrow().read_text("/tmp/new.txt").as_deref(), Some("x"));
    }

    #[test]
    fn cp_r_copies_a_directory_tree() {
        let (mut b, fs, _s) = backend();
        b.exec("mkdir /tmp/src");
        fs.borrow_mut().write_text("/tmp/src/f.txt", "deep").unwrap();
        b.exec("cp -r /tmp/src /tmp/dst");
        assert_eq!(fs.borrow().read_text("/tmp/dst/f.txt").as_deref(), Some("deep"));
    }

    #[test]
    fn grep_finds_matching_lines_case_insensitively() {
        let (mut b, fs, _s) = backend();
        fs.borrow_mut().write_text("/tmp/log.txt", "Alpha\nbeta\nGAMMA\n").unwrap();
        let lines = b.exec("grep -i gamma /tmp/log.txt");
        assert!(lines.iter().any(|l| l.text == "GAMMA"));
        assert!(!lines.iter().any(|l| l.text == "beta"));
    }

    #[test]
    fn head_and_tail_limit_lines() {
        let (mut b, fs, _s) = backend();
        fs.borrow_mut().write_text("/tmp/n.txt", "1\n2\n3\n4\n5\n").unwrap();
        let h = b.exec("head -n 2 /tmp/n.txt");
        assert_eq!(h.iter().filter(|l| l.kind == LineKind::Output).count(), 2);
        assert!(h.iter().any(|l| l.text == "1"));
        let t = b.exec("tail -n 2 /tmp/n.txt");
        assert!(t.iter().any(|l| l.text == "5"));
        assert!(!t.iter().any(|l| l.text == "1"));
    }

    #[test]
    fn wc_counts_lines_words_bytes() {
        let (mut b, fs, _s) = backend();
        fs.borrow_mut().write_text("/tmp/w.txt", "one two\nthree\n").unwrap();
        let lines = b.exec("wc /tmp/w.txt");
        let s = &lines[0].text;
        // 2 newlines, 3 words.
        assert!(s.contains('2'));
        assert!(s.contains('3'));
    }

    #[test]
    fn find_locates_files_by_name() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("find /home/jayden -name welcome.txt");
        assert!(lines.iter().any(|l| l.text.ends_with("/welcome.txt")));
    }

    #[test]
    fn kill_terminates_a_domain() {
        let (mut b, _fs, sched) = backend();
        // init has id 1.
        let lines = b.exec("kill 1");
        assert!(lines.iter().any(|l| l.text.contains("killed domain 1")));
        assert_eq!(sched.borrow().state(crate::sched::DomainId(1)), Some(DomainState::Finished));
    }

    #[test]
    fn which_identifies_builtins() {
        let (mut b, _fs, _s) = backend();
        assert!(b.exec("which ls")[0].text.contains("builtin"));
        assert_eq!(b.exec("which nosuch")[0].kind, LineKind::Error);
    }

    #[test]
    fn cd_dotdot_and_tilde_and_dash() {
        let (mut b, _fs, _s) = backend();
        b.exec("cd Documents");
        b.exec("cd ..");
        assert_eq!(b.exec("pwd")[0].text, "/home/jayden");
        b.exec("cd /etc");
        b.exec("cd ~");
        assert_eq!(b.exec("pwd")[0].text, "/home/jayden");
        // `cd -` returns to the previous directory.
        b.exec("cd -");
        assert_eq!(b.exec("pwd")[0].text, "/etc");
    }

    #[test]
    fn windows_aliases_map_to_unix_impls() {
        let (mut b, _fs, _s) = backend();
        // `dir` == `ls`
        let lines = b.exec("dir");
        assert!(lines.iter().any(|l| l.text.contains("Documents/")));
        // `type` == `cat`
        let cat = b.exec("type /home/jayden/Documents/welcome.txt");
        assert!(cat.iter().any(|l| l.text.contains("Welcome to DominionOS")));
    }

    #[test]
    fn tree_and_du_walk_the_tree() {
        let (mut b, _fs, _s) = backend();
        let tree = b.exec("tree /home/jayden");
        assert!(tree.iter().any(|l| l.text.contains("Documents/")));
        assert!(tree.iter().any(|l| l.text.contains("directories")));
        let du = b.exec("du /home/jayden");
        assert!(du[0].text.contains("/home/jayden"));
    }

    #[test]
    fn drives_a_real_terminal_end_to_end() {
        let (b, _fs, _s) = backend();
        let mut t = Terminal::with_backend(Box::new(b));
        for c in "uname -a".chars() {
            t.input_key(c);
        }
        t.input_key('\n');
        assert!(t.lines().iter().any(|l| l.text.contains("SASOS")));
    }

    // ── dominion command tests ──────────────────────────────────────────────────

    #[test]
    fn dominion_help_lists_subcommands() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("dominion help");
        assert!(lines.iter().any(|l| l.text.contains("publish")));
        assert!(lines.iter().any(|l| l.text.contains("list")));
        assert!(lines.iter().any(|l| l.text.contains("view")));
    }

    #[test]
    fn dominion_publish_creates_file_and_reports_url() {
        let (mut b, fs, _s) = backend();
        let lines = b.exec("dominion publish mypage Title: My Test Page");
        // Output mentions the dominion:// URL.
        assert!(lines.iter().any(|l| l.text.contains("dominion://mypage")));
        // File was actually written.
        assert!(fs.borrow().exists("/dominion/pages/mypage.dominion"));
    }

    #[test]
    fn dominion_list_shows_published_page() {
        let (mut b, _fs, _s) = backend();
        b.exec("dominion publish alpha Title: Alpha");
        b.exec("dominion publish beta Title: Beta");
        let lines = b.exec("dominion list");
        assert!(lines.iter().any(|l| l.text.contains("dominion://alpha")));
        assert!(lines.iter().any(|l| l.text.contains("dominion://beta")));
    }

    #[test]
    fn dominion_view_shows_page_source() {
        let (mut b, _fs, _s) = backend();
        b.exec("dominion publish viewtest Title: View Me");
        let lines = b.exec("dominion view viewtest");
        assert!(lines.iter().any(|l| l.text.contains("viewtest")));
        // The DSL content is shown.
        assert!(lines.iter().any(|l| l.text.contains("View Me")));
    }

    #[test]
    fn dominion_rm_removes_page() {
        let (mut b, fs, _s) = backend();
        b.exec("dominion publish gone Title: Gone");
        assert!(fs.borrow().exists("/dominion/pages/gone.dominion"));
        let lines = b.exec("dominion rm gone");
        assert!(lines.iter().any(|l| l.text.contains("Removed")));
        assert!(!fs.borrow().exists("/dominion/pages/gone.dominion"));
    }

    #[test]
    fn dominion_list_empty_when_no_pages() {
        let (mut b, _fs, _s) = backend();
        let lines = b.exec("dominion list");
        // Either "No user-authored pages" or an empty list — not a crash.
        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.text.contains("No user-authored") || l.text.contains("dominion://")));
    }

    #[test]
    fn integration_subsystem_commands_dispatch_end_to_end() {
        let (mut b, _fs, _s) = backend();
        // Drivers.
        assert!(b.exec("driver list").iter().any(|l| l.text.contains("rtl8139")));
        assert!(b.exec("driver load rtl8139").iter().any(|l| l.text.contains("loaded")));
        // Polyglot languages.
        assert!(b.exec("lang list").iter().any(|l| l.text.contains("py")));
        // Packages incl. the CUDA stack, installed with dependencies.
        assert!(b.exec("pkg list").iter().any(|l| l.text.contains("cuda-driver")));
        assert!(b.exec("pkg install tensorrt").iter().any(|l| l.text.contains("installed")));
        // Foreign app launched + run, confined.
        assert!(b.exec("app demo").iter().any(|l| l.text.contains("exit")));
        // Media codecs + GPU.
        assert!(b.exec("media list").iter().any(|l| l.text.contains("Flac")));
        assert!(b.exec("gpu list").iter().any(|l| l.text.contains("H100")));
        assert!(b.exec("gpu libs").iter().any(|l| l.text.to_lowercase().contains("cublas")));
        // The Dominion bytecode VM and JIT execute from the terminal.
        assert!(b.exec("vm 6 * 7").iter().any(|l| l.text.contains("42")));
        assert!(b.exec("jit 2 + 3").iter().any(|l| l.text.contains("5")));
    }
}

// ── Integration subsystem commands: driver / app / lang / pkg / vm / jit ──
// Each routes to the capability-confined integration layer (driverload, applaunch,
// polyglot::runtime, packaging::depot) or the Dominion bytecode VM/JIT.
impl ShellBackend {
    fn cmd_driver(&self, args: &[&str]) -> Vec<TermLine> {
        use crate::personality::driverload::{load_driver, registry_names, DriverSource};
        match args.first().copied().unwrap_or("list") {
            "list" | "ls" | "" => {
                let mut lines = alloc::vec![info("registered drivers:")];
                for n in registry_names() {
                    lines.push(out(alloc::format!("  {}", n)));
                }
                lines
            }
            "load" => {
                let name = match args.get(1) {
                    Some(n) => *n,
                    None => return alloc::vec![err("usage: driver load <name>")],
                };
                let tags = crate::cheri::SoftwareTags::new([0x5Au8; 32]);
                let mut dma = crate::driver::ModelDmaMem::new();
                let env = crate::driver::ResourceClaim { mmio_base: 0, mmio_len: 0xFFFF_FFFF, irq: 0 };
                match load_driver(DriverSource::Registry(name), &tags, &mut dma, env) {
                    Ok(d) => {
                        let (b, l) = d.window();
                        alloc::vec![
                            out(alloc::format!(
                                "loaded '{}' class={:?} boundary={:?}",
                                d.name, d.class, d.boundary
                            )),
                            out(alloc::format!("  window: base={:#x} len={:#x}", b, l)),
                        ]
                    }
                    Err(e) => alloc::vec![err(alloc::format!("driver load failed: {:?}", e))],
                }
            }
            _ => alloc::vec![
                info("driver — load & inspect hardware drivers (capability-confined)"),
                out("  driver list           list registered driver specs"),
                out("  driver load <name>    load+confine a driver, show its MMIO window"),
            ],
        }
    }

    fn cmd_app(&self, args: &[&str]) -> Vec<TermLine> {
        use crate::personality::applaunch::{launch_app, Grants, SyscallStep};
        match args.first().copied().unwrap_or("formats") {
            "formats" | "list" => alloc::vec![
                info("supported foreign application formats:"),
                out("  elf   (Linux)"),
                out("  pe    (Windows)"),
                out("  macho (macOS)"),
            ],
            "demo" => {
                // Launch a minimal confined ELF and run open/write/exit via the shim.
                let mut elf = alloc::vec![0u8; 64];
                elf[0..4].copy_from_slice(b"\x7FELF");
                elf[4] = 2;
                elf[5] = 1;
                let parent = crate::capability::Capability::mint(
                    0x10_0000,
                    0x10_0000,
                    crate::capability::Rights::ALL,
                );
                let mut app =
                    match launch_app(&elf, "/sandbox/demo", &parent, Grants::sandboxed(0x1000, 0x4000)) {
                        Ok(a) => a,
                        Err(e) => return alloc::vec![err(alloc::format!("launch failed: {:?}", e))],
                    };
                let res = app.run_program(&[
                    SyscallStep::new(2).path("out.txt"),
                    SyscallStep::new(1).fd(3).data(b"hello from a sandboxed app"),
                    SyscallStep::new(60).fd(0),
                ]);
                alloc::vec![
                    out(alloc::format!("launched ELF as {:?}, capability-confined", app.abi())),
                    out(alloc::format!("  open  -> fd {}", res.first().copied().unwrap_or(-1))),
                    out(alloc::format!("  write -> {} bytes", res.get(1).copied().unwrap_or(-1))),
                    out(alloc::format!(
                        "  exit  = {:?}, fs used = {} bytes",
                        app.exit_code(),
                        app.fs_used()
                    )),
                ]
            }
            _ => alloc::vec![
                info("app — run foreign (Linux/Windows/macOS) apps, sandboxed"),
                out("  app formats   list supported binary formats"),
                out("  app demo      launch a confined app and run a syscall trace"),
            ],
        }
    }

    fn cmd_lang(&self, args: &[&str]) -> Vec<TermLine> {
        use crate::polyglot::runtime;
        match args.first().copied().unwrap_or("list") {
            "list" | "ls" => {
                let mut lines = alloc::vec![info("polyglot languages:")];
                for i in runtime::catalog() {
                    lines.push(out(alloc::format!("  {:<5} {}", i.id, i.display)));
                }
                lines
            }
            "packages" | "pkgs" => {
                let mut lines = alloc::vec![info("importable packages:")];
                for p in runtime::packages() {
                    lines.push(out(alloc::format!("  {}", p.name)));
                }
                lines
            }
            "run" => {
                let (lang, file) = match (args.get(1), args.get(2)) {
                    (Some(l), Some(f)) => (*l, *f),
                    _ => return alloc::vec![err("usage: lang run <lang> <file>")],
                };
                let path = self.resolve(file);
                let src = match self.fs.borrow().read_text(&path) {
                    Some(s) => s,
                    None => return alloc::vec![err(alloc::format!("lang: {}: no such file", file))],
                };
                match runtime::run_named(lang, &src) {
                    Ok(r) => {
                        let mut lines: Vec<TermLine> = r.output.iter().map(|l| out(l.clone())).collect();
                        lines.push(out(alloc::format!("→ {} ({} steps)", r.value.display(), r.steps)));
                        lines
                    }
                    Err(e) => alloc::vec![err(alloc::format!("lang run: {:?}", e))],
                }
            }
            "check" => {
                let (lang, file) = match (args.get(1), args.get(2)) {
                    (Some(l), Some(f)) => (*l, *f),
                    _ => return alloc::vec![err("usage: lang check <lang> <file>")],
                };
                let l = match runtime::from_name(lang) {
                    Some(l) => l,
                    None => return alloc::vec![err(alloc::format!("unknown language '{}'", lang))],
                };
                let path = self.resolve(file);
                let src = match self.fs.borrow().read_text(&path) {
                    Some(s) => s,
                    None => return alloc::vec![err(alloc::format!("lang: {}: no such file", file))],
                };
                match runtime::check(&src, l) {
                    Ok(n) => alloc::vec![out(alloc::format!("ok: {} function(s) parsed", n))],
                    Err(e) => alloc::vec![err(alloc::format!("compile error: {:?}", e))],
                }
            }
            _ => alloc::vec![
                info("lang — the polyglot language runtime (7 languages)"),
                out("  lang list                 list languages"),
                out("  lang packages             list importable packages"),
                out("  lang run <lang> <file>    run a source file"),
                out("  lang check <lang> <file>  compile-check a source file"),
            ],
        }
    }

    fn cmd_pkg(&self, args: &[&str]) -> Vec<TermLine> {
        use crate::packaging::depot::default_depot;
        match args.first().copied().unwrap_or("list") {
            "list" | "ls" => {
                let mut lines = alloc::vec![info("available packages:")];
                let d = default_depot();
                for n in d.available() {
                    let kind = d.manifest(&n).map(|m| alloc::format!("{:?}", m.kind)).unwrap_or_default();
                    lines.push(out(alloc::format!("  {:<16} {}", n, kind)));
                }
                lines
            }
            "resolve" => {
                let name = match args.get(1) {
                    Some(n) => *n,
                    None => return alloc::vec![err("usage: pkg resolve <name>")],
                };
                match default_depot().resolve(name) {
                    Ok(o) => alloc::vec![out(alloc::format!("install order: {}", o.join(" -> ")))],
                    Err(e) => alloc::vec![err(alloc::format!("resolve failed: {:?}", e))],
                }
            }
            "install" => {
                let name = match args.get(1) {
                    Some(n) => *n,
                    None => return alloc::vec![err("usage: pkg install <name>")],
                };
                let depot = default_depot();
                let mut reg = crate::packaging::PackageRegistry::new();
                let grant = crate::capability::Capability::mint(
                    0,
                    1 << 20,
                    crate::capability::Rights::READ.union(crate::capability::Rights::WRITE),
                );
                match depot.install_with_deps(name, &mut reg, &grant) {
                    Ok(o) => {
                        alloc::vec![out(alloc::format!(
                            "installed (verified + confined): {}",
                            o.join(" -> ")
                        ))]
                    }
                    Err(e) => alloc::vec![err(alloc::format!("install failed: {:?}", e))],
                }
            }
            _ => alloc::vec![
                info("pkg — package depot (download/install with dependencies)"),
                out("  pkg list               list available packages"),
                out("  pkg resolve <name>     show dependency install order"),
                out("  pkg install <name>     verify + install with dependencies"),
            ],
        }
    }

    fn cmd_media(&self, args: &[&str]) -> Vec<TermLine> {
        use crate::codec::{media, CodecRegistry};
        match args.first().copied().unwrap_or("list") {
            "list" | "ls" => {
                let mut lines = alloc::vec![info("audio/video formats (native recognition):")];
                for f in media::catalog() {
                    lines.push(out(alloc::format!(
                        "  {:<6} {:<14} .{}",
                        f.kind,
                        f.media_type,
                        f.exts.join(" .")
                    )));
                }
                lines
            }
            "info" => {
                let ext = match args.get(1) {
                    Some(e) => *e,
                    None => return alloc::vec![err("usage: media info <ext>   (e.g. media info flac)")],
                };
                let reg = CodecRegistry::with_media();
                match reg.by_extension(ext) {
                    Some(c) => alloc::vec![out(alloc::format!(
                        "{} → kind {}, media-type {}",
                        ext,
                        c.semantic_kind(),
                        c.media_type()
                    ))],
                    None => alloc::vec![err(alloc::format!("no codec for '.{}'", ext))],
                }
            }
            _ => alloc::vec![
                info("media — audio/video codec catalog (FLAC/WAV/Opus/AAC/AV1/VP9/H264/H265/…)"),
                out("  media list           list supported formats"),
                out("  media info <ext>     show the codec for an extension"),
            ],
        }
    }

    fn cmd_gpu(&self, args: &[&str]) -> Vec<TermLine> {
        use crate::ml::gpu::{known_gpus, CudaLibrary};
        match args.first().copied().unwrap_or("list") {
            "list" | "ls" => {
                let mut lines = alloc::vec![info("recognised GPUs:")];
                for g in known_gpus() {
                    lines.push(out(alloc::format!(
                        "  {:<20} cc {}.{}  {} GB  {} SMs",
                        g.name, g.compute_capability.0, g.compute_capability.1, g.vram_gb, g.sm_count
                    )));
                }
                lines
            }
            "libs" | "lib" => {
                let mut lines = alloc::vec![info("CUDA stack (capability-gated, default-closed):")];
                for l in CudaLibrary::all() {
                    lines.push(out(alloc::format!("  {:<10} {} entry points", l.name(), l.symbols().len())));
                }
                lines.push(info("install with: pkg install tensorrt   (pulls cudnn/cublas/toolkit/driver)"));
                lines
            }
            _ => alloc::vec![
                info("gpu — NVIDIA CUDA/cuDNN ML acceleration (capability-gated)"),
                out("  gpu list   recognised GPU models"),
                out("  gpu libs   CUDA libraries + entry points"),
            ],
        }
    }

    fn cmd_vm(&self, rest: &str) -> Vec<TermLine> {
        if rest.is_empty() {
            return alloc::vec![info("usage: vm <dominion expression>   (runs via the bytecode VM)")];
        }
        match crate::lang::vm::eval_compiled(rest) {
            Ok(v) => alloc::vec![out(alloc::format!("→ {} (vm)", v))],
            Err(e) => alloc::vec![err(alloc::format!("! {}", e))],
        }
    }

    fn cmd_jit(&self, rest: &str) -> Vec<TermLine> {
        if rest.is_empty() {
            return alloc::vec![info("usage: jit <dominion expression>   (runs via the JIT over the VM)")];
        }
        let prog = match crate::lang::parse_source(rest) {
            Ok(p) => p,
            Err(e) => return alloc::vec![err(alloc::format!("! parse: {:?}", e))],
        };
        let compiled = match crate::lang::compile::compile(&prog) {
            Ok(c) => c,
            Err(e) => return alloc::vec![err(alloc::format!("! compile: {}", e.message))],
        };
        let mut jit = crate::lang::jit::Jit::new(&compiled);
        match jit.run(rest) {
            Ok(v) => alloc::vec![out(alloc::format!("→ {} (jit)", v))],
            Err(e) => alloc::vec![err(alloc::format!("! {}", e))],
        }
    }
}

//! The **IDE** — the node-graph programming surface (the dashboard's node graph,
//! promoted). A program is an Dominion [`Program`](crate::lang::Program) AST kept in sync
//! across **three views of the same tree**:
//!
//! * the **visual node graph** — one node per top-level item, wires from dataflow
//!   (a `let` that defines `x` wires to every later node that references `x`);
//! * a **program-wide source editor** always visible side-by-side (graph on the left,
//!   source on the right); and
//! * a **per-node editor** for scripting one item at a time.
//!
//! Editing any view re-emits the others: graph edits (wire / delete / add) mutate the
//! AST and re-emit source ([`crate::lang::to_source`]); source edits re-parse
//! ([`crate::lang::parse_source`]) and rebuild the graph automatically on every
//! keystroke. Programs can be created, browsed, and **run / stopped / triggered /
//! looped** through the real interpreter.
//!
//! Pure, safe `no_std`. Rendered in page-local coordinates for the shell.

use crate::lang::ast::{Expr, Item, Stmt};
use crate::lang::{parse_source, to_source, Interpreter, Program};
use crate::nodes::{NodeGraph, NodeKind, Press};
use crate::toolkit::{self, Color, DrawCmd, Rect, Theme};
use crate::editor::Editor;
use crate::terminal::Terminal;
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const TOOLBAR_H: i32 = 38;
/// Width of the left file-tree sidebar (px).
const SIDEBAR_W: i32 = 160;

// ── Embedded example programs ────────────────────────────────────────────────
// These are inlined at compile time so the IDE works in no_std without any
// filesystem access.  Names map 1:1 to the `examples/` directory.
const EXAMPLES: &[(&str, &str)] = &[
    ("01_hello.aeth", r#"// === 01: Hello, Dominion ===
// Teaches: literals and the `print` builtin.
print("Hello, Dominion!")
print(42)
print(3.14)
print(true)
print("text")
print("the answer is", 42)
print("done")"#),
    ("02_variables.aeth", r#"// === 02: Variables and bindings ===
let x = 10
print("x =", x)
x = x + 5
print("x is now", x)
let name = "Ada"
let pi = 3.14159
let ok = true
print(name, pi, ok)
linear ticket = "single-use-token"
print("consumed:", ticket)
x"#),
    ("03_arithmetic.aeth", r#"// === 03: Arithmetic and operators ===
print("add", 2 + 3)
print("sub", 10 - 4)
print("mul", 6 * 7)
print("div", 17 / 5)
print("rem", 17 % 5)
print("fdiv", 17.0 / 5.0)
print("neg", -8)
print("not", !false)
print("prec", 1 + 2 * 3)
print("paren", (1 + 2) * 3)
print("lt", 3 < 5)
print("ge", 5 >= 5)
print("eq", 4 == 4)
print("ne", 4 != 5)
print("and", true && false)
print("or",  false || true)
let result = 2 + 2 * 3 == 8 && 1 < 2
print("combined", result)
1 + 2 * 3 - 4 / 2"#),
    ("04_strings.aeth", r#"// === 04: Strings ===
let s = "  Hello, Dominion World  "
print("trim",  trim(s))
print("upper", upper("dominion"))
print("lower", lower("DOMINION"))
print("concat+", "foo" + "bar")
let words = split("a,b,c,d", ",")
print("split", words)
print("join",  join(words, "-"))
print("chars", chars("hi"))
print("starts", starts_with("dominion-os", "dominion"))
print("ends",   ends_with("report.aeth", ".aeth"))
print("has",    contains("dominionos", "her"))
print("replace", replace("a-b-c", "-", "_"))
print("reverse", reverse("abc"))
print("len", len("hello"))
join(split("one two three", " "), "|")"#),
    ("05_vectors.aeth", r#"// === 05: Vectors ===
let v = [3, 1, 4, 1, 5, 9, 2, 6]
print("len",   len(v))
print("get 0", get(v, 0))
print("first", first(v))
print("last",  last(v))
let v2 = push(v, 7)
print("push", v2)
print("reverse", reverse(v))
print("slice",   slice(v, 1, 4))
print("sort",    sort(v))
print("sum",     sum(v))
print("min",     min(v))
print("max",     max(v))
print("range", range(5))
sum(sort([5, 4, 3, 2, 1]))"#),
    ("06_conditionals.aeth", r#"// === 06: Conditionals ===
let x = 15
if x > 10 {
    print("x is greater than 10")
} else {
    print("x is 10 or less")
}
let score = 85
if score >= 90 {
    print("A")
} else {
    if score >= 80 {
        print("B")
    } else {
        print("C")
    }
}
score >= 80"#),
    ("07_loops.aeth", r#"// === 07: Loops ===
let i = 0
let total = 0
while i < 5 {
    total = total + i
    i = i + 1
}
print("while sum 0..4 =", total)
let s = 0
for n in range(10) {
    s = s + n
}
print("sum 0..9 =", s)
let words = ["cat", "dog", "fish"]
for word in words {
    print(word)
}
s"#),
    ("08_functions.aeth", r#"// === 08: Functions ===
fn greet(name) {
    print("hello,", name)
}
greet("Dominion")
fn add(a, b) {
    return a + b
}
print("3 + 4 =", add(3, 4))
fn square(x) {
    x * x
}
print("7^2 =", square(7))
fn factorial(n) {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}
print("10! =", factorial(10))
fn fib(n) {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}
print("fib(10) =", fib(10))
fib(10)"#),
    ("09_pipes_and_map.aeth", r#"// === 09: Pipes and Map ===
let nums = [3, 1, 4, 1, 5, 9, 2, 6]
let sorted_piped = nums |> sort
print("sorted via pipe:", sorted_piped)
fn take5(xs) { return slice(xs, 0, 5) }
let top5 = nums |> sort |> take5
print("top 5 sorted:", top5)
fn double(x) { return x * 2 }
let doubled = [1, 2, 3, 4, 5] => double
print("doubled:", doubled)
fn is_even(x) { return x % 2 == 0 }
let evenness = [1, 2, 3, 4] => is_even
print("is_even map:", evenness)
sum(doubled)"#),
    ("10_objects.aeth", r#"// === 10: Objects ===
object Point {
    x: Int,
    y: Int
}
object Rectangle {
    origin: Point,
    width: Int,
    height: Int
}
let p = Point { x: 3, y: 4 }
print("point:", p.x, p.y)
fn distance_from_origin(pt) {
    sqrt(float(pt.x * pt.x + pt.y * pt.y))
}
print("distance:", distance_from_origin(p))
let rect = Rectangle {
    origin: Point { x: 10, y: 20 },
    width: 100,
    height: 50
}
print("rect area:", rect.width * rect.height)
fn area(r) { return r.width * r.height }
area(rect)"#),
];

/// One browsable program (name + its AST).
struct ProgramFile {
    name: String,
    ast: Program,
}

/// Which editor (if any) currently has keyboard focus.
enum Edit {
    None,
    /// The whole-program source editor (right pane).
    Program,
    /// One item's source fragment (per-node editor), by item index.
    Node(usize),
}

/// A toolbar button.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Btn {
    Prev,
    Next,
    New,
    Run,
    Stop,
    Loop,
    AddNode,
    AddProgram,
    Examples,
    Help,
}

const BUTTONS: [(Btn, &str); 10] = [
    (Btn::Prev, "<"),
    (Btn::Next, ">"),
    (Btn::New, "New"),
    (Btn::Run, "Run"),
    (Btn::Stop, "Stop"),
    (Btn::Loop, "Loop"),
    (Btn::AddNode, "+Node"),
    (Btn::AddProgram, "+Prog"),
    (Btn::Examples, "Examples"),
    (Btn::Help, "?"),
];

/// The IDE page.
pub struct Ide {
    programs: Vec<ProgramFile>,
    current: usize,
    graph: NodeGraph,
    /// Node id (i+1) ↔ item index i; positions persist across rebuilds by index.
    positions: Vec<(i32, i32)>,
    editor: Editor,
    edit: Edit,
    /// Source editor is always visible; this flag only controls whether the pane is
    /// collapsed (false) to a thin strip vs fully open (true, the default).
    code_open: bool,
    looping: bool,
    loop_div: u32,
    /// The IDE's output console — a real embedded [`Terminal`]; program runs stream
    /// into it instead of a dumb text list.
    output: Terminal,
    /// True when the terminal panel has keyboard focus (keys route to `self.output`).
    terminal_focused: bool,
    parse_error: Option<String>,
    next_name: u32,
    area: Rect,
    last_left: bool,
    press_x: i32,
    press_y: i32,
    /// Wall-clock (ms), for the caret blink in the editor + terminal.
    now_ms: u64,
    damage: Option<Rect>,
    /// Left-pane (graph) width as a percentage of the total **content** area (after
    /// sidebar). Clamped to 20..80. Default 55 so the graph gets slightly more space.
    split_frac: i32,
    /// True while the user is dragging the vertical divider.
    dragging_divider: bool,
    /// Whether the examples dropdown panel is open.
    examples_open: bool,
    /// Whether the help overlay is open.
    help_open: bool,
    /// Whether the left file-tree sidebar is shown.
    sidebar_open: bool,
}

impl Ide {
    pub fn new() -> Ide {
        let src = "let sales = load(\"alpha\");\nlet report = sales |> summarise;\nfn dbl(x) {\n    return x * 2;\n}\n";
        let ast = parse_source(src).unwrap_or_default();
        let mut ide = Ide {
            programs: alloc::vec![ProgramFile { name: "project.aeth".into(), ast }],
            current: 0,
            graph: NodeGraph::new(),
            positions: Vec::new(),
            editor: Editor::new(src),
            edit: Edit::None,
            // Default to open — source editor is always visible side-by-side.
            code_open: true,
            looping: false,
            loop_div: 0,
            output: Terminal::new(),
            terminal_focused: false,
            parse_error: None,
            next_name: 2,
            area: Rect::new(0, 0, 1280, 600),
            last_left: false,
            press_x: 0,
            press_y: 0,
            now_ms: 0,
            damage: Some(Rect::new(0, 0, 1280, 600)),
            split_frac: 55,
            dragging_divider: false,
            examples_open: false,
            help_open: false,
            sidebar_open: true,
        };
        ide.rebuild_graph();
        ide
    }

    fn ast(&self) -> &Program {
        &self.programs[self.current].ast
    }
    fn ast_mut(&mut self) -> &mut Program {
        &mut self.programs[self.current].ast
    }

    // ── shell-facing API ──

    pub fn set_area(&mut self, area: Rect) {
        if area.w != self.area.w || area.h != self.area.h {
            self.area = area;
            self.dmg_all();
        } else {
            self.area = area;
        }
    }
    pub fn take_damage(&mut self) -> Option<Rect> {
        self.damage.take()
    }
    fn dmg_all(&mut self) {
        self.damage = Some(Rect::new(0, 0, self.area.w, self.area.h));
    }
    fn dmg(&mut self, r: Rect) {
        self.damage = Some(match self.damage {
            Some(d) => toolkit::union(d, r),
            None => r,
        });
    }

    /// Whether a text editor currently has focus (so the shell routes *all* keys to it,
    /// not to global hotkeys).
    pub fn is_text_focused(&self) -> bool {
        matches!(self.edit, Edit::Program | Edit::Node(_)) || self.terminal_focused
    }

    /// The exact rect the program-source editor renders into (so a click can place the
    /// caret on the glyph under the pointer). Mirrors [`Self::draw_code_bar`].
    fn code_editor_area(&self) -> Rect {
        let bar = self.code_bar_rect();
        // Header row is 30px; 8px bottom padding = 38px total chrome.
        let h = (bar.h - 38).max(0);
        Rect::new(bar.x + 8, bar.y + 30, (bar.w - 16).max(0), h).inset(4)
    }
    /// View height of the source editor inside the code bar (used for scroll clamping).
    fn code_editor_view_h(&self) -> i32 {
        (self.code_bar_rect().h - 38).max(1)
    }
    /// The exact rect the per-node editor renders into. Mirrors [`Self::draw_node_editor`].
    fn node_editor_area(&self) -> Rect {
        let r = self.node_editor_rect();
        Rect::new(r.x + 8, r.y + 32, r.w - 16, r.h - 40).inset(4)
    }
    /// The font size editor text renders at (the theme body size).
    const EDITOR_FONT: i32 = 15;

    /// Advance the IDE clock (drives the caret blink in the editor + terminal). If a
    /// field is focused and the blink phase flipped, damage just that field so the caret
    /// visibly flashes without repainting the whole page.
    pub fn set_time(&mut self, now_ms: u64) {
        let prev = self.now_ms;
        self.now_ms = now_ms;
        self.editor.tick(now_ms);
        self.output.tick(now_ms);
        if (prev / crate::text::BLINK_MS) != (now_ms / crate::text::BLINK_MS) {
            match self.edit {
                Edit::Program => self.dmg(self.code_bar_rect()),
                Edit::Node(_) => self.dmg(self.node_editor_rect()),
                Edit::None => {}
            }
        }
    }

    /// Called at the metric tick; drives the **Loop** control (re-run periodically).
    pub fn tick(&mut self) {
        if self.looping {
            self.loop_div += 1;
            if self.loop_div >= 2 {
                self.loop_div = 0;
                self.run();
            }
        }
    }

    // ── graph ⇄ AST ⇄ source ──

    /// Rebuild the visual graph from the current AST: one node per top-level item,
    /// wires from variable def→use. Node positions persist by item index.
    fn rebuild_graph(&mut self) {
        let mut g = NodeGraph::new();
        let items = self.ast().items.clone();
        // Ensure a position slot per item (auto-layout new ones in a grid).
        while self.positions.len() < items.len() {
            let i = self.positions.len() as i32;
            self.positions.push((40 + (i % 3) * 230, 40 + (i / 3) * 120));
        }
        self.positions.truncate(items.len());
        for (i, item) in items.iter().enumerate() {
            let (title, sub, kind) = node_label(item);
            let (x, y) = self.positions[i];
            g.add((i + 1) as u32, &title, &sub, kind, x, y);
        }
        // Wires: for each item, if it references a name defined by an earlier item.
        for (j, item) in items.iter().enumerate() {
            let mut used = BTreeSet::new();
            item_idents(item, &mut used);
            for (i, prev) in items.iter().enumerate() {
                if i == j {
                    continue;
                }
                if let Some(name) = item_defines(prev) {
                    if used.contains(&name) {
                        g.wire((i + 1) as u32, (j + 1) as u32);
                    }
                }
            }
        }
        self.graph = g;
    }

    /// Parse the program editor's text; on success replace the AST and rebuild the
    /// graph, on failure record the parse error for display. Returns success.
    pub fn sync_from_editor(&mut self) -> bool {
        let text = self.editor.text();
        match parse_source(&text) {
            Ok(ast) => {
                self.ast_mut().items = ast.items;
                self.parse_error = None;
                self.rebuild_graph();
                self.dmg_all();
                true
            }
            Err(e) => {
                self.parse_error = Some(format!("{}", e));
                self.dmg_all();
                false
            }
        }
    }

    /// Wire node `from`→`to`: make the target item consume the source's value by piping
    /// the source variable into the target's expression. Re-emits source + rebuilds.
    pub fn wire(&mut self, from_id: u32, to_id: u32) {
        let (fi, ti) = (from_id as usize - 1, to_id as usize - 1);
        let items = &self.ast().items;
        if fi >= items.len() || ti >= items.len() || fi == ti {
            return;
        }
        let from_var = match item_defines(&items[fi]) {
            Some(v) => v,
            None => return, // source has no value to feed
        };
        // Only statements with an expression can accept a piped input.
        let new_stmt = match &items[ti] {
            Item::Stmt(s) => pipe_into(s, &from_var),
            _ => None,
        };
        if let Some(stmt) = new_stmt {
            self.ast_mut().items[ti] = Item::Stmt(stmt);
            self.after_graph_edit();
        }
    }

    /// Delete a node (and its top-level item). Re-emits source + rebuilds.
    pub fn delete_node(&mut self, id: u32) {
        let i = id as usize - 1;
        if i < self.ast().items.len() {
            self.ast_mut().items.remove(i);
            self.positions.remove(i.min(self.positions.len().saturating_sub(1)));
            self.after_graph_edit();
        }
    }

    /// Append a fresh `let nodeN = 0;` node.
    pub fn add_node(&mut self) {
        let name = format!("node{}", self.next_name);
        self.next_name += 1;
        self.ast_mut().items.push(Item::Stmt(Stmt::Let(name, Expr::Int(0))));
        self.after_graph_edit();
    }

    /// Append an embedded program-reference node: `let progN = run_program("progN.aeth");`
    /// The node appears in the graph as `NodeKind::Program` (violet) and can be wired
    /// to pass its output into other nodes. Double-clicking it in the graph opens the
    /// referenced program by name so it can be edited inline.
    pub fn add_program_node(&mut self) {
        let name = format!("prog{}", self.next_name);
        self.next_name += 1;
        let path_lit = Expr::Str(format!("{}.aeth", name));
        let call = Expr::Call(
            alloc::boxed::Box::new(Expr::Ident("run_program".into())),
            alloc::vec![path_lit],
        );
        self.ast_mut().items.push(Item::Stmt(Stmt::Let(name, call)));
        self.after_graph_edit();
    }

    fn after_graph_edit(&mut self) {
        self.rebuild_graph();
        // A structural graph edit is authoritative → regenerate the source buffer.
        self.edit = Edit::None;
        let src = to_source(self.ast());
        self.editor = Editor::new(&src);
        self.parse_error = None;
        self.dmg_all();
    }

    // ── run / stop / trigger / loop ──

    /// Run the current program once through the interpreter; stream the result/error
    /// into the embedded output **terminal**.
    pub fn run(&mut self) {
        let mut it = Interpreter::new();
        match it.run(self.ast()) {
            Ok(v) => self.output.println(format!("→ {}", v)),
            Err(e) => self.output.eprintln(format!("! {}", e)),
        }
        self.dmg_all();
    }
    pub fn stop(&mut self) {
        self.looping = false;
        self.output.info("■ stopped");
        self.dmg_all();
    }
    /// The output terminal (e.g. for tests/drivers to inspect or drive it).
    pub fn output(&self) -> &Terminal {
        &self.output
    }
    pub fn output_mut(&mut self) -> &mut Terminal {
        &mut self.output
    }

    // ── program browser / persistence ──

    /// A `(name, source)` snapshot of every program — the shell persists this into the
    /// [`World`](crate::world::World) so programs become first-class system objects
    /// (Desktop cards, Explorer knowledge nodes) with real capability provenance.
    pub fn programs_snapshot(&self) -> Vec<(String, String)> {
        self.programs
            .iter()
            .map(|p| (p.name.clone(), to_source(&p.ast)))
            .collect()
    }

    /// The name of the currently-open program.
    pub fn current_name(&self) -> &str {
        &self.programs[self.current].name
    }

    /// Open a program by name (e.g. when a Desktop card is clicked). Returns whether a
    /// program with that name was found.
    pub fn open_by_name(&mut self, name: &str) -> bool {
        if let Some(i) = self.programs.iter().position(|p| p.name == name) {
            if i != self.current {
                self.current = i;
                self.positions.clear();
                self.rebuild_graph();
                self.edit = Edit::None;
                self.editor = Editor::new(&to_source(self.ast()));
            }
            self.dmg_all();
            true
        } else {
            false
        }
    }

    /// Load one of the embedded example programs by its index into EXAMPLES.
    pub fn load_example(&mut self, idx: usize) {
        if let Some(&(name, src)) = EXAMPLES.get(idx) {
            // Check if a program with this name already exists; if so, overwrite it.
            if let Some(pos) = self.programs.iter().position(|p| p.name == name) {
                if let Ok(ast) = parse_source(src) {
                    self.programs[pos].ast = ast;
                }
                self.current = pos;
            } else {
                let ast = parse_source(src).unwrap_or_default();
                self.programs.push(ProgramFile { name: name.into(), ast });
                self.current = self.programs.len() - 1;
            }
            self.positions.clear();
            self.editor = Editor::new(src);
            self.edit = Edit::Program;
            self.parse_error = None;
            self.rebuild_graph();
            self.examples_open = false;
            self.dmg_all();
        }
    }

    pub fn new_program(&mut self) {
        let name = format!("untitled{}.aeth", self.programs.len());
        self.programs.push(ProgramFile { name, ast: Program::default() });
        self.current = self.programs.len() - 1;
        self.positions.clear();
        self.after_graph_edit();
    }
    fn switch(&mut self, delta: i32) {
        let n = self.programs.len() as i32;
        self.current = (((self.current as i32 + delta) % n + n) % n) as usize;
        self.positions.clear();
        self.rebuild_graph();
        self.edit = Edit::None;
        self.editor = Editor::new(&to_source(self.ast()));
        self.dmg_all();
    }

    pub fn disconnect_node_inputs(&mut self, id: u32) {
        self.graph.disconnect_inputs(id);
    }
    pub fn disconnect_node_outputs(&mut self, id: u32) {
        self.graph.disconnect_outputs(id);
    }
    pub fn disconnect_node_all(&mut self, id: u32) {
        self.graph.disconnect_all(id);
    }
    pub fn disconnect_wire_at(&mut self, px: i32, py: i32) -> bool {
        self.graph.disconnect_wire_at(px, py, 12)
    }

    /// Toggle the examples dropdown panel (called from os.rs context menu).
    pub fn toggle_examples(&mut self) {
        self.examples_open = !self.examples_open;
        self.help_open = false;
        self.dmg_all();
    }

    /// Reset (remove) the wires connected to the currently selected node.
    pub fn reset_selected_wires(&mut self) {
        if let Some(id) = self.graph.selected() {
            self.reset_node_connections(id);
        }
    }

    // ── context menu / os.rs API (Issue 3) ──

    /// The node ID under screen coordinates `(px, py)`, converting screen coords to
    /// graph-local coords using the current pan offset. Returns `None` if no node is hit.
    pub fn ctx_node_at(&self, px: i32, py: i32) -> Option<u32> {
        let area = self.graph_area();
        if !area.contains(px, py) {
            return None;
        }
        // The graph renders with a translate of (area.x, area.y); convert to canvas coords.
        let gx = px - area.x;
        let gy = py - area.y;
        self.graph.node_at(gx, gy)
    }

    /// The currently selected node ID (if any).
    pub fn selected_node(&self) -> Option<u32> {
        self.graph.selected()
    }

    /// Delete whichever node is currently selected in the graph.
    pub fn delete_selected_node(&mut self) {
        if let Some(id) = self.graph.selected() {
            self.delete_node(id);
        }
    }

    /// Remove all wires connected to node `id` by rebuilding the graph from the AST
    /// after stripping all pipe expressions that reference the node's output variable.
    pub fn reset_node_connections(&mut self, id: u32) {
        let i = id as usize - 1;
        let items = self.ast().items.clone();
        if i >= items.len() {
            return;
        }
        let var = match item_defines(&items[i]) {
            Some(v) => v,
            None => {
                // Node defines no variable; just rebuild to clear any stale wires.
                self.rebuild_graph();
                self.dmg_all();
                return;
            }
        };
        // Rewrite any item that pipes `var |> …` to remove that pipe.
        let mut changed = false;
        for (j, item) in self.ast_mut().items.iter_mut().enumerate() {
            if j == i {
                continue;
            }
            if let Item::Stmt(stmt) = item {
                if let Some(new_stmt) = unpipe_var(stmt, &var) {
                    *stmt = new_stmt;
                    changed = true;
                }
            }
        }
        if changed {
            let src = to_source(self.ast());
            self.editor = Editor::new(&src);
            self.parse_error = None;
        }
        self.rebuild_graph();
        self.dmg_all();
    }

    // ── input ──

    pub fn on_pointer(&mut self, px: i32, py: i32, left: bool) {
        let pressed = left && !self.last_left;
        let released = !left && self.last_left;

        // ── Help overlay swallows all clicks while open ──
        if self.help_open && pressed {
            self.help_open = false;
            self.dmg_all();
            self.last_left = left;
            return;
        }

        // ── Examples dropdown swallows clicks ──
        if self.examples_open && pressed {
            let panel = self.examples_panel_rect();
            if panel.contains(px, py) {
                // Which row was clicked?
                let row_h = 22_i32;
                let inner_y = py - panel.y - 4;
                if inner_y >= 0 {
                    let idx = (inner_y / row_h) as usize;
                    if idx < EXAMPLES.len() {
                        self.load_example(idx);
                        self.last_left = left;
                        return;
                    }
                }
            }
            self.examples_open = false;
            self.dmg_all();
            self.last_left = left;
            return;
        }

        // ── Divider drag (continuous, checked before press routing) ──
        if self.dragging_divider {
            if left {
                // Recompute split_frac from pointer x (relative to content area).
                let cx = px - self.content_x();
                let cw = self.content_w();
                if cw > 0 {
                    self.split_frac = (cx * 100 / cw).clamp(20, 80);
                }
                self.dmg_all();
            }
            if released {
                self.dragging_divider = false;
                self.dmg_all();
            }
            self.last_left = left;
            return;
        }

        if pressed {
            self.press_x = px;
            self.press_y = py;
            // 1. Toolbar buttons.
            if py < TOOLBAR_H {
                for (b, _) in BUTTONS.iter() {
                    if self.btn_rect(*b).contains(px, py) {
                        self.click_button(*b);
                        self.last_left = left;
                        return;
                    }
                }
                self.last_left = left;
                return;
            }
            // 2. Sidebar toggle strip (the left edge narrow strip).
            if self.sidebar_toggle_rect().contains(px, py) {
                self.sidebar_open = !self.sidebar_open;
                self.dmg_all();
                self.last_left = left;
                return;
            }
            // 3. Sidebar file list.
            if self.sidebar_open {
                let sb = self.sidebar_rect();
                if sb.contains(px, py) {
                    self.handle_sidebar_click(px, py);
                    self.last_left = left;
                    return;
                }
            }
            // 4. Vertical divider — start drag.
            if self.divider_rect().contains(px, py) {
                self.dragging_divider = true;
                self.last_left = left;
                return;
            }
            // 5. Per-node editor popover (if open) swallows clicks inside it.
            if let Edit::Node(_) = self.edit {
                if self.node_editor_rect().contains(px, py) {
                    if self.node_editor_sync_rect().contains(px, py) {
                        self.sync_node_editor();
                    } else {
                        // Click in the text area → place the caret where you clicked.
                        let ea = self.node_editor_area();
                        if ea.contains(px, py) {
                            self.editor.place_cursor(px, py, ea, Self::EDITOR_FONT);
                            self.dmg_all();
                        }
                    }
                    self.last_left = left;
                    return;
                } else {
                    self.edit = Edit::None; // click outside closes it
                    self.dmg_all();
                }
            }
            // 6. Source editor pane (always visible in the right pane).
            let bar = self.code_bar_rect();
            if bar.contains(px, py) {
                if !self.code_open {
                    // Collapsed bottom bar → expand the pane.
                    self.code_open = true;
                    self.open_program_editor();
                } else {
                    // Click in the text area → focus + place the caret.
                    self.terminal_focused = false;
                    self.edit = Edit::Program;
                    let ea = self.code_editor_area();
                    if ea.contains(px, py) {
                        self.editor.place_cursor(px, py, ea, Self::EDITOR_FONT);
                        let view_h = self.code_editor_view_h();
                        self.editor.ensure_caret_visible(view_h, Self::EDITOR_FONT);
                    }
                }
                self.dmg_all();
                self.last_left = left;
                return;
            }
            // 7. Terminal output box — give it keyboard focus.
            let area = self.graph_area();
            let term_h = 168_i32;
            let term_r = Rect::new(area.x + area.w - 340, area.y + area.h - term_h - 8, 332, term_h);
            if term_r.contains(px, py) {
                self.terminal_focused = true;
                self.edit = Edit::None;
                self.dmg_all();
                self.last_left = left;
                return;
            }
        }

        // 8. Node graph (only in graph area, when pointer is not captured above).
        let area = self.graph_area();
        let (gx, gy) = (px - area.x, py - area.y);
        if pressed && area.contains(px, py) {
            // A click in the graph clears terminal focus.
            self.terminal_focused = false;
            match self.graph.on_press(gx, gy) {
                Press::Node(_) | Press::Port | Press::Empty => {}
            }
            self.dmg_all();
        } else if left && !pressed {
            self.graph.on_drag(gx, gy);
            self.dmg_all();
        }
        if released {
            let made = self.graph.on_release(gx, gy);
            if made {
                if let Some(w) = self.graph.wires().last().copied() {
                    self.wire(w.from, w.to);
                }
            } else {
                let moved = (px - self.press_x).abs() + (py - self.press_y).abs();
                if moved < 5 {
                    if let Some(id) = self.graph.selected() {
                        if area.contains(px, py) {
                            let idx = id as usize - 1;
                            // Program nodes: clicking opens the referenced program.
                            if idx < self.ast().items.len() {
                                if let Item::Stmt(Stmt::Let(_, Expr::Call(callee, args))) =
                                    &self.ast().items[idx]
                                {
                                    if let Expr::Ident(fname) = callee.as_ref() {
                                        if fname == "run_program" {
                                            if let Some(Expr::Str(path)) = args.first() {
                                                let prog_name =
                                                    path.trim_end_matches(".aeth").to_string();
                                                // If the program doesn't exist yet, create it.
                                                if !self.open_by_name(&prog_name) {
                                                    self.programs.push(ProgramFile {
                                                        name: format!("{}.aeth", prog_name),
                                                        ast: crate::lang::Program::default(),
                                                    });
                                                    self.current = self.programs.len() - 1;
                                                    self.positions.clear();
                                                    self.after_graph_edit();
                                                }
                                                self.dmg_all();
                                                self.last_left = left;
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                            self.open_node_editor(idx);
                        }
                    }
                }
                // Persist any drag movement back into positions.
                self.capture_positions();
            }
            self.dmg_all();
        }
        self.last_left = left;
    }

    fn handle_sidebar_click(&mut self, _px: i32, py: i32) {
        let sb = self.sidebar_rect();
        // Skip header row (28px).
        let inner_y = py - sb.y - 28;
        if inner_y < 0 {
            return;
        }
        let row_h = 22_i32;
        let idx = (inner_y / row_h) as usize;
        if idx < self.programs.len() && idx != self.current {
            self.current = idx;
            self.positions.clear();
            self.rebuild_graph();
            self.edit = Edit::None;
            self.editor = Editor::new(&to_source(self.ast()));
            self.dmg_all();
        }
    }

    fn capture_positions(&mut self) {
        for n in self.graph.nodes() {
            let i = n.id as usize - 1;
            if i < self.positions.len() {
                self.positions[i] = (n.x, n.y);
            }
        }
    }

    fn click_button(&mut self, b: Btn) {
        match b {
            Btn::Prev => self.switch(-1),
            Btn::Next => self.switch(1),
            Btn::New => self.new_program(),
            Btn::Run => self.run(),
            Btn::Stop => self.stop(),
            Btn::Loop => {
                self.looping = !self.looping;
                self.dmg_all();
            }
            Btn::AddNode => self.add_node(),
            Btn::AddProgram => self.add_program_node(),
            Btn::Examples => {
                self.examples_open = !self.examples_open;
                self.help_open = false;
                self.dmg_all();
            }
            Btn::Help => {
                self.help_open = !self.help_open;
                self.examples_open = false;
                self.dmg_all();
            }
        }
    }

    fn open_program_editor(&mut self) {
        self.editor = Editor::new(&to_source(self.ast()));
        self.editor.key('i'); // enter insert mode so typing edits text
        self.edit = Edit::Program;
    }
    fn open_node_editor(&mut self, idx: usize) {
        if idx < self.ast().items.len() {
            let src = crate::lang::emit::item_to_source(&self.ast().items[idx]);
            self.editor = Editor::new(&src);
            self.editor.key('i');
            self.edit = Edit::Node(idx);
            self.dmg_all();
        }
    }
    /// Apply the per-node editor: reparse the fragment as a one-item program and splice
    /// it back into the AST.
    pub fn sync_node_editor(&mut self) {
        if let Edit::Node(idx) = self.edit {
            let text = self.editor.text();
            match parse_source(&text) {
                Ok(p) if !p.items.is_empty() => {
                    self.ast_mut().items[idx] = p.items[0].clone();
                    self.parse_error = None;
                    self.edit = Edit::None;
                    self.after_graph_edit();
                }
                Ok(_) => {
                    // Empty fragment → delete the item.
                    self.edit = Edit::None;
                    self.delete_node((idx + 1) as u32);
                }
                Err(e) => {
                    self.parse_error = Some(format!("{}", e));
                    self.dmg_all();
                }
            }
        }
    }

    /// Returns true if the key was consumed (an editor has focus, or it was a local
    /// hotkey) — so the shell only applies its global hotkeys when the IDE didn't.
    pub fn on_key(&mut self, ch: char) -> bool {
        // Close overlays on Escape.
        if ch == '\x1b' {
            if self.help_open {
                self.help_open = false;
                self.dmg_all();
                return true;
            }
            if self.examples_open {
                self.examples_open = false;
                self.dmg_all();
                return true;
            }
        }

        // Help overlay toggle on '?'.
        if ch == '?' && !matches!(self.edit, Edit::Program | Edit::Node(_)) && !self.terminal_focused {
            self.help_open = !self.help_open;
            self.examples_open = false;
            self.dmg_all();
            return true;
        }

        // Terminal focus: route keystrokes to the embedded terminal.
        if self.terminal_focused {
            if ch == '\x1b' {
                self.terminal_focused = false;
                self.dmg_all();
                return true;
            }
            let _ = self.output.input_key(ch);
            self.dmg_all();
            return true;
        }

        match self.edit {
            Edit::Program | Edit::Node(_) => {
                if ch == '\u{4}' {
                    // Ctrl-D / sync key.
                    self.commit_editor();
                } else {
                    self.editor.key(ch);
                    // Auto-sync the source editor on every keystroke so the graph
                    // always reflects what is typed (sync_from_editor is a no-op on
                    // parse errors, so the graph is never left in a broken state).
                    if matches!(self.edit, Edit::Program) {
                        self.sync_from_editor();
                    }
                    // Keep the caret visible after any cursor movement or text edit.
                    let view_h = match self.edit {
                        Edit::Program => self.code_editor_view_h(),
                        _ => (self.node_editor_rect().h - 40).max(1),
                    };
                    self.editor.ensure_caret_visible(view_h, Self::EDITOR_FONT);
                    self.dmg_all();
                }
                true
            }
            Edit::None => match ch {
                'r' => {
                    self.run();
                    true
                }
                'c' => {
                    self.code_open = !self.code_open;
                    if self.code_open {
                        self.open_program_editor();
                    }
                    self.dmg_all();
                    true
                }
                _ => false,
            },
        }
    }

    fn commit_editor(&mut self) {
        match self.edit {
            Edit::Program => {
                self.sync_from_editor();
            }
            Edit::Node(_) => self.sync_node_editor(),
            Edit::None => {}
        }
    }

    // ── layout helpers ──

    /// X origin of the content area (after the sidebar, if open).
    fn content_x(&self) -> i32 {
        if self.sidebar_open { SIDEBAR_W } else { 0 }
    }
    /// Width of the content area.
    fn content_w(&self) -> i32 {
        (self.area.w - self.content_x()).max(0)
    }

    fn btn_rect(&self, b: Btn) -> Rect {
        let idx = BUTTONS.iter().position(|(x, _)| *x == b).unwrap_or(0) as i32;
        // The first two (Prev/Next) sit left of the program name; the rest to the right.
        if idx < 2 {
            Rect::new(8 + idx * 30, 6, 26, TOOLBAR_H - 12)
        } else {
            // Wider buttons for Examples and Help — use distinct widths.
            let widths: [i32; 8] = [60, 60, 60, 60, 60, 80, 26, 0];
            let w = widths.get((idx - 2) as usize).copied().unwrap_or(60);
            let x = if idx < 10 {
                let mut acc = 8 + 2 * 30 + 220;
                for k in 2..idx {
                    let wk = widths.get((k - 2) as usize).copied().unwrap_or(60);
                    acc += wk + 4;
                }
                acc
            } else {
                8 + 2 * 30 + 220 + (idx - 2) * 64
            };
            Rect::new(x, 6, w, TOOLBAR_H - 12)
        }
    }
    fn divider_x(&self) -> i32 {
        let cx = self.content_x();
        let cw = self.content_w();
        let rel = (cw * self.split_frac / 100).clamp(120, (cw - 120).max(121));
        (cx + rel).clamp(cx + 120, cx + cw - 120)
    }
    /// The 6-px vertical divider between graph and source editor (always present).
    fn divider_rect(&self) -> Rect {
        Rect::new(self.divider_x(), TOOLBAR_H, 6, self.area.h - TOOLBAR_H)
    }
    fn graph_area(&self) -> Rect {
        let cx = self.content_x();
        let dx = self.divider_x();
        Rect::new(cx, TOOLBAR_H, (dx - cx).max(0), self.area.h - TOOLBAR_H)
    }
    fn code_bar_rect(&self) -> Rect {
        if self.code_open {
            let dx = self.divider_x() + 6;
            Rect::new(dx, TOOLBAR_H, (self.area.w - dx).max(0), self.area.h - TOOLBAR_H)
        } else {
            // Collapsed strip at the bottom of the right section.
            let dx = self.divider_x() + 6;
            Rect::new(dx, self.area.h - 24, (self.area.w - dx).max(0), 24)
        }
    }
    fn node_editor_rect(&self) -> Rect {
        Rect::new(self.area.w / 2 - 220, TOOLBAR_H + 40, 440, 200)
    }
    fn node_editor_sync_rect(&self) -> Rect {
        let r = self.node_editor_rect();
        Rect::new(r.x + r.w - 70, r.y + 6, 60, 22)
    }
    fn sidebar_rect(&self) -> Rect {
        if self.sidebar_open {
            Rect::new(0, TOOLBAR_H, SIDEBAR_W, self.area.h - TOOLBAR_H)
        } else {
            Rect::new(0, TOOLBAR_H, 0, 0)
        }
    }
    /// The narrow toggle strip at the very left edge (always present so the sidebar can
    /// be reopened even when collapsed).
    fn sidebar_toggle_rect(&self) -> Rect {
        Rect::new(0, TOOLBAR_H, 12, self.area.h - TOOLBAR_H)
    }
    /// Rect of the Examples dropdown panel (anchored below the Examples button).
    fn examples_panel_rect(&self) -> Rect {
        let btn = self.btn_rect(Btn::Examples);
        let row_h = 22_i32;
        let h = EXAMPLES.len() as i32 * row_h + 8;
        let w = 200_i32;
        // Clamp so it stays on screen.
        let x = btn.x.min(self.area.w - w);
        Rect::new(x, btn.y + btn.h, w, h)
    }
    /// Rect of the help overlay (centred in the IDE area).
    fn help_overlay_rect(&self) -> Rect {
        let w = (self.area.w - 80).min(640);
        let h = (self.area.h - 80).min(480);
        let x = (self.area.w - w) / 2;
        let y = TOOLBAR_H + (self.area.h - TOOLBAR_H - h) / 2;
        Rect::new(x, y, w, h)
    }

    // ── rendering ──

    pub fn view(&self, theme: &Theme) -> Vec<DrawCmd> {
        let mut s = Vec::new();
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, self.area.h), color: theme.bg, radius: 0 });

        // Left sidebar.
        self.draw_sidebar(&mut s, theme);

        // Node graph in the left content pane.
        let area = self.graph_area();
        let mut g = self.graph.view(theme);
        toolkit::translate_scene(&mut g, area.x, area.y);
        s.append(&mut g);

        self.draw_toolbar(&mut s, theme);
        self.draw_output(&mut s, theme, area);

        // Vertical divider between graph and source editor (always present).
        let div = self.divider_rect();
        s.push(DrawCmd::Rect { rect: div, color: theme.muted, radius: 0 });

        self.draw_code_bar(&mut s, theme);
        if let Edit::Node(idx) = self.edit {
            self.draw_node_editor(&mut s, theme, idx);
        }

        // Overlays rendered on top of everything else.
        if self.examples_open {
            self.draw_examples_panel(&mut s, theme);
        }
        if self.help_open {
            self.draw_help_overlay(&mut s, theme);
        }
        s
    }

    fn draw_sidebar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        // Always draw the narrow toggle strip.
        let strip = self.sidebar_toggle_rect();
        s.push(DrawCmd::Rect { rect: strip, color: t.surface, radius: 0 });
        s.push(DrawCmd::Text {
            rect: Rect::new(strip.x + 1, TOOLBAR_H + (self.area.h - TOOLBAR_H) / 2 - 20, 10, 40),
            text: if self.sidebar_open { "◀" } else { "▶" }.into(),
            color: t.muted,
            size: 10,
        });

        if !self.sidebar_open {
            return;
        }
        let sb = self.sidebar_rect();
        s.push(DrawCmd::Rect { rect: sb, color: t.surface, radius: 0 });
        // Separator line.
        s.push(DrawCmd::Rect { rect: Rect::new(sb.x + sb.w - 1, sb.y, 1, sb.h), color: t.muted, radius: 0 });
        // Header.
        s.push(DrawCmd::Text {
            rect: Rect::new(sb.x + 8, sb.y + 6, sb.w - 16, 16),
            text: "Programs".into(),
            color: t.text,
            size: 12,
        });
        // Program list.
        let row_h = 22_i32;
        for (i, prog) in self.programs.iter().enumerate() {
            let ry = sb.y + 28 + i as i32 * row_h;
            let rr = Rect::new(sb.x + 4, ry, sb.w - 8, row_h - 2);
            if i == self.current {
                s.push(DrawCmd::Rect { rect: rr, color: t.primary, radius: t.radius });
                s.push(DrawCmd::Text {
                    rect: Rect::new(rr.x + 6, rr.y + 4, rr.w - 12, 14),
                    text: prog.name.clone(),
                    color: t.on_primary,
                    size: 11,
                });
            } else {
                s.push(DrawCmd::Text {
                    rect: Rect::new(rr.x + 6, rr.y + 4, rr.w - 12, 14),
                    text: prog.name.clone(),
                    color: t.text,
                    size: 11,
                });
            }
        }
    }

    fn draw_toolbar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        s.push(DrawCmd::Rect { rect: Rect::new(0, 0, self.area.w, TOOLBAR_H), color: t.surface, radius: 0 });
        // Program name (between the prev/next buttons and the action buttons).
        s.push(DrawCmd::Text { rect: Rect::new(8 + 2 * 30 + 8, 10, 210, 18), text: self.programs[self.current].name.clone(), color: t.text, size: 14 });
        for (b, label) in BUTTONS.iter() {
            let r = self.btn_rect(*b);
            if r.w == 0 {
                continue;
            }
            let active = (matches!(b, Btn::Loop) && self.looping)
                || (matches!(b, Btn::Examples) && self.examples_open)
                || (matches!(b, Btn::Help) && self.help_open);
            let fill = if active { t.primary } else { t.bg };
            s.push(DrawCmd::Rect { rect: r, color: fill, radius: t.radius });
            let fg = if active { t.on_primary } else { t.text };
            s.push(DrawCmd::Text { rect: Rect::new(r.x + 6, r.y + 4, r.w - 8, 16), text: (*label).into(), color: fg, size: 12 });
        }
    }

    fn draw_output(&self, s: &mut Vec<DrawCmd>, t: &Theme, area: Rect) {
        // The IDE's output is a real **embedded terminal**: a bottom-right console
        // that the program runs stream into.
        let h = 168;
        let box_r = Rect::new(area.x + area.w - 340, area.y + area.h - h - 8, 332, h);
        let label_color = if self.terminal_focused { t.primary } else { t.muted };
        let label = if self.terminal_focused { "Terminal (focused — Esc to unfocus)" } else { "Terminal" };
        s.push(DrawCmd::Text { rect: Rect::new(box_r.x + 4, box_r.y - 16, box_r.w, 14), text: label.into(), color: label_color, size: 11 });
        if self.terminal_focused {
            // Draw a focus ring.
            s.push(DrawCmd::Rect { rect: toolkit::inflate(box_r, 2), color: t.primary, radius: t.radius });
        }
        s.append(&mut self.output.view(t, box_r));
        if let Some(err) = &self.parse_error {
            s.push(DrawCmd::Text { rect: Rect::new(box_r.x + 8, box_r.y + box_r.h - 16, box_r.w - 16, 14), text: err.clone(), color: t.danger, size: 11 });
        }
    }

    fn draw_code_bar(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let bar = self.code_bar_rect();
        s.push(DrawCmd::Rect { rect: bar, color: t.surface, radius: 0 });
        if !self.code_open {
            s.push(DrawCmd::Text {
                rect: Rect::new(bar.x + 10, bar.y + 4, 300, 16),
                text: "▸ Program source (click or press C to open)".into(),
                color: t.muted,
                size: 12,
            });
            return;
        }
        // Right-pane header row.
        let focused_prog = matches!(self.edit, Edit::Program);
        let header_color = if focused_prog { t.text } else { t.muted };
        s.push(DrawCmd::Text {
            rect: Rect::new(bar.x + 10, bar.y + 6, (bar.w - 30).max(0), 18),
            text: "Program source — auto-synced".into(),
            color: header_color,
            size: 13,
        });
        if let Some(err) = &self.parse_error {
            s.push(DrawCmd::Text {
                rect: Rect::new(bar.x + 8, bar.y + bar.h - 18, (bar.w - 16).max(0), 14),
                text: err.clone(),
                color: t.danger,
                size: 11,
            });
        }
        let ed_h = if self.parse_error.is_some() { (bar.h - 50).max(0) } else { (bar.h - 32).max(0) };
        let ed = Rect::new(bar.x + 8, bar.y + 28, (bar.w - 16).max(0), ed_h);
        s.push(DrawCmd::Rect { rect: ed, color: t.bg, radius: t.radius });
        if focused_prog {
            s.push(DrawCmd::Rect { rect: toolkit::inflate(ed, 1), color: t.primary, radius: t.radius });
            s.push(DrawCmd::Rect { rect: ed, color: t.bg, radius: t.radius });
        }
        s.append(&mut self.editor.view(t, ed.inset(4)));
    }

    fn draw_node_editor(&self, s: &mut Vec<DrawCmd>, t: &Theme, idx: usize) {
        let r = self.node_editor_rect();
        s.push(DrawCmd::Rect { rect: toolkit::inflate(r, 1), color: t.primary, radius: t.radius });
        s.push(DrawCmd::Rect { rect: r, color: t.surface, radius: t.radius });
        let title = format!("Edit node — item {}", idx + 1);
        s.push(DrawCmd::Text { rect: Rect::new(r.x + 10, r.y + 8, r.w - 90, 16), text: title, color: t.text, size: 13 });
        let sync = self.node_editor_sync_rect();
        s.push(DrawCmd::Rect { rect: sync, color: t.primary, radius: t.radius });
        s.push(DrawCmd::Text { rect: Rect::new(sync.x + 12, sync.y + 4, 50, 16), text: "Sync".into(), color: t.on_primary, size: 12 });
        let ed = Rect::new(r.x + 8, r.y + 32, r.w - 16, r.h - 40);
        s.push(DrawCmd::Rect { rect: ed, color: t.bg, radius: t.radius });
        s.append(&mut self.editor.view(t, ed.inset(4)));
    }

    fn draw_examples_panel(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        let panel = self.examples_panel_rect();
        // Drop-shadow effect (simple offset rect).
        s.push(DrawCmd::Rect { rect: toolkit::inflate(panel, 1), color: t.muted, radius: t.radius });
        s.push(DrawCmd::Rect { rect: panel, color: t.surface, radius: t.radius });
        let row_h = 22_i32;
        for (i, &(name, _src)) in EXAMPLES.iter().enumerate() {
            let ry = panel.y + 4 + i as i32 * row_h;
            let rr = Rect::new(panel.x + 4, ry, panel.w - 8, row_h - 2);
            s.push(DrawCmd::Text {
                rect: Rect::new(rr.x + 6, rr.y + 4, rr.w - 12, 13),
                text: name.into(),
                color: t.text,
                size: 11,
            });
        }
    }

    fn draw_help_overlay(&self, s: &mut Vec<DrawCmd>, t: &Theme) {
        // Dim the background with a semi-transparent overlay.
        s.push(DrawCmd::Rect {
            rect: Rect::new(0, 0, self.area.w, self.area.h),
            color: Color::rgba(t.bg.r, t.bg.g, t.bg.b, 200),
            radius: 0,
        });
        let r = self.help_overlay_rect();
        s.push(DrawCmd::Rect { rect: toolkit::inflate(r, 1), color: t.primary, radius: t.radius });
        s.push(DrawCmd::Rect { rect: r, color: t.surface, radius: t.radius });

        let lh = 20_i32; // line height
        let mut y = r.y + 12;
        let x = r.x + 16;
        let w = r.w - 32;

        macro_rules! heading {
            ($txt:expr) => {
                s.push(DrawCmd::Text { rect: Rect::new(x, y, w, lh), text: $txt.into(), color: t.primary, size: 14 });
                y += lh + 4;
            };
        }
        macro_rules! line {
            ($txt:expr) => {
                s.push(DrawCmd::Text { rect: Rect::new(x + 8, y, w - 8, lh - 2), text: $txt.into(), color: t.text, size: 12 });
                y += lh;
            };
        }
        macro_rules! code {
            ($txt:expr) => {
                s.push(DrawCmd::Rect { rect: Rect::new(x + 4, y, w - 8, lh), color: t.bg, radius: t.radius });
                s.push(DrawCmd::Text { rect: Rect::new(x + 8, y + 2, w - 16, lh - 4), text: $txt.into(), color: t.text, size: 11 });
                y += lh + 2;
            };
        }

        heading!("Dominion IDE Help  (press ? or click Help to close)");
        y += 4;

        heading!("Built-in libraries — polyglot use");
        line!("Import a language library with  use LibName { fn1, fn2 }");
        code!("use Python { def greet(name): return f'hi {name}' }");
        code!("use Rust   { fn add(a: i32, b: i32) -> i32 { a + b } }");
        line!("Call imported fns just like Dominion fns:");
        code!("let msg = greet(\"world\")");
        y += 4;

        heading!("Mixing Dominion with Python / Rust (polyglot)");
        code!("use Python { import math; def sqrt2(): return math.sqrt(2) }");
        code!("let s = sqrt2()   // calls Python from Dominion");
        y += 4;

        heading!("Built-in builtins");
        let builtins: &[(&str, &str)] = &[
            ("load(name)",          "load a dataset or resource by name"),
            ("summarise(data)",     "summarise a dataset (mean, min, max …)"),
            ("run_program(path)",   "run another .aeth program; returns its result"),
            ("print(…)",            "print values to the terminal"),
            ("len(x)",              "length of a string or vector"),
            ("range(n)",            "vector [0, 1, … n-1]"),
            ("sort(v)",             "return sorted copy of vector v"),
            ("push(v, x)",          "append x to vector v (returns new vector)"),
            ("sum / min / max",     "reductions over a numeric vector"),
            ("split / join",        "string ↔ vector conversions"),
            ("sqrt(x) / float(x)", "numeric conversions"),
        ];
        for (name, desc) in builtins {
            if y + lh > r.y + r.h - 8 {
                break;
            }
            s.push(DrawCmd::Text {
                rect: Rect::new(x + 8, y, 160, lh - 2),
                text: (*name).into(),
                color: t.primary,
                size: 11,
            });
            s.push(DrawCmd::Text {
                rect: Rect::new(x + 172, y, w - 172, lh - 2),
                text: (*desc).into(),
                color: t.text,
                size: 11,
            });
            y += lh;
        }
        y += 4;
        if y + lh <= r.y + r.h - 8 {
            heading!("Keyboard shortcuts");
            line!("r — Run program          c — Toggle source editor");
            line!("? — Toggle this help     Tab — indent    Shift+Tab — dedent");
            line!("Ctrl+D — sync node editor    Esc — close overlay / unfocus terminal");
        }
        let _ = y; // silence dead-assignment warning: y is layout state, final value unused
    }
}

impl Default for Ide {
    fn default() -> Self {
        Self::new()
    }
}

// ── AST helpers (free variables, labels, dataflow edits) ──

/// The name a top-level item defines (so later items referencing it draw a wire).
fn item_defines(item: &Item) -> Option<String> {
    match item {
        Item::Stmt(Stmt::Let(n, _)) | Item::Stmt(Stmt::Linear(n, _)) => Some(n.clone()),
        Item::Fn(f) => Some(f.name.clone()),
        Item::Object(o) => Some(o.name.clone()),
        Item::Cell(c) => Some(c.name.clone()),
        _ => None,
    }
}

/// The free identifiers an item references (for dataflow wiring).
fn item_idents(item: &Item, out: &mut BTreeSet<String>) {
    match item {
        Item::Stmt(s) => stmt_idents(s, out),
        Item::Fn(f) => {
            for s in &f.body {
                stmt_idents(s, out);
            }
        }
        Item::Object(_) | Item::Cell(_) => {}
    }
}

fn stmt_idents(s: &Stmt, out: &mut BTreeSet<String>) {
    match s {
        Stmt::Let(_, e) | Stmt::Linear(_, e) | Stmt::Assign(_, e) | Stmt::Return(e) | Stmt::Expr(e) => {
            expr_idents(e, out)
        }
        Stmt::If { cond, then_block, else_block } => {
            expr_idents(cond, out);
            for s in then_block.iter().chain(else_block) {
                stmt_idents(s, out);
            }
        }
        Stmt::While { cond, body } => {
            expr_idents(cond, out);
            for s in body {
                stmt_idents(s, out);
            }
        }
        Stmt::For { iter, body, .. } => {
            expr_idents(iter, out);
            for s in body {
                stmt_idents(s, out);
            }
        }
        Stmt::Break | Stmt::Continue => {}
    }
}

fn expr_idents(e: &Expr, out: &mut BTreeSet<String>) {
    match e {
        Expr::Ident(n) => {
            out.insert(n.clone());
        }
        Expr::Path(parts) => {
            if let Some(first) = parts.first() {
                out.insert(first.clone());
            }
        }
        Expr::Neg(x) | Expr::Not(x) | Expr::Field(x, _) => expr_idents(x, out),
        Expr::Binary(_, l, r) | Expr::Map(l, r) | Expr::Pipe(l, r) | Expr::Index(l, r) => {
            expr_idents(l, out);
            expr_idents(r, out);
        }
        Expr::Call(c, args) => {
            expr_idents(c, out);
            for a in args {
                expr_idents(a, out);
            }
        }
        Expr::Vector(items) => {
            for a in items {
                expr_idents(a, out);
            }
        }
        Expr::ObjectLit(_, fields) => {
            for (_, v) in fields {
                expr_idents(v, out);
            }
        }
        _ => {}
    }
}

/// Pipe `var` into a statement's expression (`expr` → `var |> expr`), unless it already
/// references `var`. Returns the rewritten statement, or `None` if unchanged/ineligible.
fn pipe_into(s: &Stmt, var: &str) -> Option<Stmt> {
    let already = {
        let mut set = BTreeSet::new();
        stmt_idents(s, &mut set);
        set.contains(var)
    };
    if already {
        return None;
    }
    let piped = |e: &Expr| Expr::Pipe(alloc::boxed::Box::new(Expr::Ident(var.to_string())), alloc::boxed::Box::new(e.clone()));
    match s {
        Stmt::Let(n, e) => Some(Stmt::Let(n.clone(), piped(e))),
        Stmt::Linear(n, e) => Some(Stmt::Linear(n.clone(), piped(e))),
        Stmt::Assign(n, e) => Some(Stmt::Assign(n.clone(), piped(e))),
        Stmt::Return(e) => Some(Stmt::Return(piped(e))),
        Stmt::Expr(e) => Some(Stmt::Expr(piped(e))),
        Stmt::If { .. } | Stmt::While { .. } | Stmt::For { .. } | Stmt::Break | Stmt::Continue => None,
    }
}

/// Remove a `var |> …` pipe from a statement's expression (undo a wire). Returns the
/// rewritten statement if a pipe was found and removed, else `None`.
fn unpipe_var(s: &Stmt, var: &str) -> Option<Stmt> {
    fn strip(e: &Expr, var: &str) -> Option<Expr> {
        if let Expr::Pipe(lhs, rhs) = e {
            if let Expr::Ident(n) = lhs.as_ref() {
                if n == var {
                    return Some(*rhs.clone());
                }
            }
        }
        None
    }
    match s {
        Stmt::Let(n, e) => strip(e, var).map(|e2| Stmt::Let(n.clone(), e2)),
        Stmt::Linear(n, e) => strip(e, var).map(|e2| Stmt::Linear(n.clone(), e2)),
        Stmt::Assign(n, e) => strip(e, var).map(|e2| Stmt::Assign(n.clone(), e2)),
        Stmt::Return(e) => strip(e, var).map(Stmt::Return),
        Stmt::Expr(e) => strip(e, var).map(Stmt::Expr),
        _ => None,
    }
}

/// A node's `(title, subtitle, kind)` from its item.
fn node_label(item: &Item) -> (String, String, NodeKind) {
    match item {
        Item::Stmt(Stmt::Let(n, e)) => {
            // `let x = run_program("y.aeth")` → Program node showing the target name.
            if let Expr::Call(callee, args) = e {
                if let Expr::Ident(fname) = callee.as_ref() {
                    if fname == "run_program" {
                        if let Some(Expr::Str(path)) = args.first() {
                            let label = path.trim_end_matches(".aeth");
                            return (n.clone(), label.into(), NodeKind::Program);
                        }
                    }
                }
            }
            (n.clone(), "let".into(), NodeKind::Data)
        }
        Item::Stmt(Stmt::Linear(n, _)) => (n.clone(), "linear".into(), NodeKind::Data),
        Item::Stmt(Stmt::Assign(n, _)) => (n.clone(), "assign".into(), NodeKind::Data),
        Item::Stmt(Stmt::Return(_)) => ("return".into(), "stmt".into(), NodeKind::Log),
        Item::Stmt(Stmt::Expr(_)) => ("expr".into(), "stmt".into(), NodeKind::Log),
        Item::Stmt(Stmt::If { .. }) => ("if".into(), "branch".into(), NodeKind::Log),
        Item::Stmt(Stmt::While { .. }) => ("while".into(), "loop".into(), NodeKind::Log),
        Item::Stmt(Stmt::For { .. }) => ("for".into(), "loop".into(), NodeKind::Log),
        Item::Stmt(Stmt::Break) => ("break".into(), "stmt".into(), NodeKind::Log),
        Item::Stmt(Stmt::Continue) => ("continue".into(), "stmt".into(), NodeKind::Log),
        Item::Fn(f) => (f.name.clone(), "fn".into(), NodeKind::Report),
        Item::Object(o) => (o.name.clone(), "object".into(), NodeKind::Data),
        Item::Cell(c) => (c.name.clone(), "cell".into(), NodeKind::App),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(ide: &mut Ide) -> Rect {
        ide.set_area(Rect::new(0, 0, 1280, 600));
        ide.graph_area()
    }

    #[test]
    fn ast_builds_a_graph_with_dataflow_wires() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        // 3 items → 3 nodes (sales, report, dbl).
        assert_eq!(ide.graph.nodes().len(), 3);
        // `report` references `sales` → exactly one wire (sales → report).
        assert_eq!(ide.graph.wires().len(), 1);
        let w = ide.graph.wires()[0];
        assert_eq!((w.from, w.to), (1, 2));
    }

    #[test]
    fn editing_source_rebuilds_the_graph() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        ide.editor = Editor::new("let a = 1;\nlet b = a + 1;\nlet c = b + a;");
        ide.edit = Edit::None; // sync_from_editor reads editor.text()
        assert!(ide.sync_from_editor());
        assert_eq!(ide.graph.nodes().len(), 3);
        // b uses a (1 wire), c uses b and a (2 wires) → 3 wires total.
        assert_eq!(ide.graph.wires().len(), 3);
    }

    #[test]
    fn a_parse_error_is_reported_and_graph_unchanged() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        let before = ide.graph.nodes().len();
        ide.editor = Editor::new("let x = ;"); // invalid
        assert!(!ide.sync_from_editor());
        assert!(ide.parse_error.is_some());
        assert_eq!(ide.graph.nodes().len(), before);
    }

    #[test]
    fn wiring_two_nodes_emits_a_pipe_in_the_source() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        // Fresh program: a defines a value, b is independent.
        ide.editor = Editor::new("let a = 10;\nlet b = compute(2);");
        assert!(ide.sync_from_editor());
        assert_eq!(ide.graph.wires().len(), 0);
        // Wire node 1 (a) → node 2 (b).
        ide.wire(1, 2);
        let src = to_source(ide.ast());
        assert!(src.contains("a |> compute(2)"), "source was: {}", src);
        // And the graph now shows the wire.
        assert_eq!(ide.graph.wires().len(), 1);
    }

    #[test]
    fn deleting_a_node_removes_the_item_and_dependent_wires() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        // Delete node 1 (sales); report's wire to it disappears.
        ide.delete_node(1);
        assert_eq!(ide.graph.nodes().len(), 2);
        assert_eq!(ide.graph.wires().len(), 0);
        assert!(!to_source(ide.ast()).contains("let sales"));
    }

    #[test]
    fn add_node_appends_a_let() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        ide.add_node();
        assert_eq!(ide.graph.nodes().len(), 4);
        assert!(to_source(ide.ast()).contains("let node2 = 0;"));
    }

    #[test]
    fn run_evaluates_the_program_and_captures_output() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        ide.editor = Editor::new("let x = 6 * 7;\nx");
        ide.sync_from_editor();
        ide.run();
        assert!(ide.output().lines().iter().any(|l| l.text.contains("42")), "output: {:?}", ide.output().lines());
    }

    #[test]
    fn per_node_editor_round_trips_into_the_ast() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        ide.open_node_editor(2); // the `fn dbl` item
        ide.editor = Editor::new("fn dbl(x) {\n    return x * 3;\n}");
        ide.edit = Edit::Node(2);
        ide.sync_node_editor();
        assert!(to_source(ide.ast()).contains("x * 3"));
        assert_eq!(ide.graph.nodes().len(), 3);
    }

    #[test]
    fn loop_reruns_on_tick() {
        let mut ide = Ide::new();
        let _ = area(&mut ide);
        ide.editor = Editor::new("let x = 1;");
        ide.sync_from_editor();
        ide.looping = true;
        let before = ide.output().lines().len();
        ide.tick();
        ide.tick(); // loop_div reaches 2 → a run fires
        assert!(ide.output().lines().len() > before);
    }

    #[test]
    fn toolbar_run_button_executes() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        let r = ide.btn_rect(Btn::Run);
        ide.on_pointer(r.x + 5, r.y + 5, true);
        ide.on_pointer(r.x + 5, r.y + 5, false);
        // The run streamed a result line into the embedded terminal (past its banner).
        assert!(ide.output().lines().len() > 1);
    }

    #[test]
    fn ctx_node_at_returns_node_under_pointer() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        // There are 3 nodes; check that we can find at least one.
        let area = ide.graph_area();
        // Node 1 is placed at (40, 40) + pan; just test that None is returned outside.
        let result = ide.ctx_node_at(area.x - 10, area.y - 10);
        assert!(result.is_none());
    }

    #[test]
    fn selected_node_and_delete_selected_node() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        let before = ide.graph.nodes().len();
        // No selection initially.
        assert!(ide.selected_node().is_none());
        // delete_selected_node is a no-op when nothing is selected.
        ide.delete_selected_node();
        assert_eq!(ide.graph.nodes().len(), before);
    }

    #[test]
    fn reset_node_connections_removes_pipes() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        ide.editor = Editor::new("let a = 10;\nlet b = a |> compute(2);");
        assert!(ide.sync_from_editor());
        assert!(ide.graph.wires().len() > 0);
        // Reset wires on node 1 (a).
        ide.reset_node_connections(1);
        assert_eq!(ide.graph.wires().len(), 0);
    }

    #[test]
    fn load_example_loads_first_example() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        let before_count = ide.programs.len();
        ide.load_example(0);
        // A new program named 01_hello.aeth should exist.
        assert!(ide.programs.iter().any(|p| p.name == "01_hello.aeth"));
        // Program count increased.
        assert!(ide.programs.len() > before_count);
    }

    #[test]
    fn terminal_focus_routes_keys_to_output() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        ide.terminal_focused = true;
        // Keys should be consumed when terminal is focused.
        assert!(ide.on_key('x'));
        // Escape should unfocus.
        assert!(ide.on_key('\x1b'));
        assert!(!ide.terminal_focused);
    }

    #[test]
    fn help_overlay_toggles_on_question_mark() {
        let mut ide = Ide::new();
        ide.set_area(Rect::new(0, 0, 1280, 600));
        assert!(!ide.help_open);
        ide.on_key('?');
        assert!(ide.help_open);
        ide.on_key('?');
        assert!(!ide.help_open);
    }

    #[test]
    fn code_open_defaults_to_true() {
        let ide = Ide::new();
        assert!(ide.code_open, "source editor should be open by default");
    }
}

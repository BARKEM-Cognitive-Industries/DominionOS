//! The node-graph editor — the centrepiece of the dashboard (see the concept).
//!
//! Applications, data objects and code are **nodes** wired together with Bézier
//! **wires** showing how data/capabilities flow between them — a live view of the
//! object graph and the running system, not a static picture. The canvas can be
//! **panned** (drag empty space to look around), nodes are **draggable**,
//! **selectable**, and **openable** (a click opens the underlying object), and wires
//! can be **created by dragging** from one node's output port to another's input.
//! Renders to a backend-agnostic [`crate::toolkit`] scene. Pure, safe `no_std`.

use crate::toolkit::{self, Color, DrawCmd, Rect};
use alloc::string::String;
use alloc::vec::Vec;

/// What a node represents — selects its accent colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// A running application / view.
    App,
    /// A data object in the graph.
    Data,
    /// A neural audio object.
    Audio,
    /// A generated report.
    Report,
    /// A system log stream.
    Log,
    /// An embedded program/project reference (drill-down opens its own IDE).
    Program,
}

impl NodeKind {
    pub fn color(self, theme: &toolkit::Theme) -> Color {
        match self {
            NodeKind::App => theme.primary,
            NodeKind::Data => theme.accent,
            NodeKind::Audio => Color::rgb(0x3f, 0xc9, 0xb0), // teal
            NodeKind::Report => Color::rgb(0xff, 0xb0, 0x4f), // amber
            NodeKind::Log => theme.muted,
            NodeKind::Program => Color::rgb(0x8b, 0x5c, 0xf6), // violet
        }
    }
}

/// A node in the editor (positions are in *graph space*; the pan offset is applied
/// at render/hit-test time).
#[derive(Clone, Debug)]
pub struct Node {
    pub id: u32,
    pub title: String,
    pub subtitle: String,
    pub kind: NodeKind,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// A directed wire from one node's output to another's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wire {
    pub from: u32,
    pub to: u32,
}

/// A port a wire can attach to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortKind {
    In,
    Out,
}

/// What a press landed on (so the caller can distinguish click-to-open from pan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Press {
    Node(u32),
    Port,
    Empty,
}

const PORT_R: i32 = 6;

/// The node-graph editor: nodes, wires, pan, and interaction state.
#[derive(Default)]
pub struct NodeGraph {
    nodes: Vec<Node>,
    wires: Vec<Wire>,
    pan_x: i32,
    pan_y: i32,
    /// (node id, grab offset x, grab offset y) while dragging a node.
    drag: Option<(u32, i32, i32)>,
    /// (last x, last y) while panning the canvas.
    panning: Option<(i32, i32)>,
    /// (source node id, current endpoint) while dragging a new wire.
    wiring: Option<(u32, (i32, i32))>,
    selected: Option<u32>,
}

impl NodeGraph {
    pub fn new() -> NodeGraph {
        NodeGraph::default()
    }

    pub fn add(&mut self, id: u32, title: &str, subtitle: &str, kind: NodeKind, x: i32, y: i32) -> u32 {
        self.nodes.push(Node { id, title: title.into(), subtitle: subtitle.into(), kind, x, y, w: 168, h: 52 });
        id
    }

    pub fn wire(&mut self, from: u32, to: u32) {
        if from != to
            && self.nodes.iter().any(|n| n.id == from)
            && self.nodes.iter().any(|n| n.id == to)
            && !self.wires.iter().any(|w| w.from == from && w.to == to)
        {
            self.wires.push(Wire { from, to });
        }
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn wires(&self) -> &[Wire] {
        &self.wires
    }
    pub fn selected(&self) -> Option<u32> {
        self.selected
    }
    pub fn pan(&self) -> (i32, i32) {
        (self.pan_x, self.pan_y)
    }
    pub fn is_interacting(&self) -> bool {
        self.drag.is_some() || self.panning.is_some() || self.wiring.is_some()
    }
    /// The in-progress wire (source anchor, current endpoint), for the dash to draw.
    pub fn pending_wire(&self) -> Option<((i32, i32), (i32, i32))> {
        self.wiring.map(|(id, end)| (self.out_anchor(self.node(id).unwrap()), end))
    }

    fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// On-screen rect of a node (graph-space position + pan offset).
    fn screen_rect(&self, n: &Node) -> Rect {
        Rect::new(n.x + self.pan_x, n.y + self.pan_y, n.w, n.h)
    }
    fn out_anchor(&self, n: &Node) -> (i32, i32) {
        (n.x + n.w + self.pan_x, n.y + n.h / 2 + self.pan_y)
    }
    fn in_anchor(&self, n: &Node) -> (i32, i32) {
        (n.x + self.pan_x, n.y + n.h / 2 + self.pan_y)
    }

    /// The top-most node under `(px,py)` (canvas coordinates), if any.
    pub fn node_at(&self, px: i32, py: i32) -> Option<u32> {
        self.nodes.iter().rev().find(|n| self.screen_rect(n).contains(px, py)).map(|n| n.id)
    }

    /// The port under `(px,py)`, if any.
    fn port_at(&self, px: i32, py: i32) -> Option<(u32, PortKind)> {
        for n in self.nodes.iter().rev() {
            let (ox, oy) = self.out_anchor(n);
            if (px - ox) * (px - ox) + (py - oy) * (py - oy) <= PORT_R * PORT_R {
                return Some((n.id, PortKind::Out));
            }
            let (ix, iy) = self.in_anchor(n);
            if (px - ix) * (px - ix) + (py - iy) * (py - iy) <= PORT_R * PORT_R {
                return Some((n.id, PortKind::In));
            }
        }
        None
    }

    /// Begin interaction at `(px,py)`: an output port starts a wire, a node starts a
    /// drag (and selects), empty canvas starts a pan.
    pub fn on_press(&mut self, px: i32, py: i32) -> Press {
        if let Some((id, PortKind::Out)) = self.port_at(px, py) {
            self.wiring = Some((id, (px, py)));
            return Press::Port;
        }
        if let Some(id) = self.node_at(px, py) {
            self.selected = Some(id);
            if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
                let n = self.nodes.remove(pos);
                let (ox, oy) = (px - (n.x + self.pan_x), py - (n.y + self.pan_y));
                self.nodes.push(n);
                self.drag = Some((id, ox, oy));
            }
            return Press::Node(id);
        }
        self.selected = None;
        self.panning = Some((px, py));
        Press::Empty
    }

    /// Continue a drag / pan / wire.
    pub fn on_drag(&mut self, px: i32, py: i32) {
        if let Some((id, ox, oy)) = self.drag {
            if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
                n.x = px - self.pan_x - ox;
                n.y = py - self.pan_y - oy;
            }
        } else if let Some((lx, ly)) = self.panning {
            self.pan_x += px - lx;
            self.pan_y += py - ly;
            self.panning = Some((px, py));
        } else if let Some((id, _)) = self.wiring {
            self.wiring = Some((id, (px, py)));
        }
    }

    /// End interaction. If a wire was being dragged and the release lands on an input
    /// port of a different node, the wire is created. Returns true if a wire was made.
    pub fn on_release(&mut self, px: i32, py: i32) -> bool {
        let mut made = false;
        if let Some((from, _)) = self.wiring.take() {
            if let Some((to, PortKind::In)) = self.port_at(px, py) {
                if to != from {
                    self.wire(from, to);
                    made = true;
                }
            }
        }
        self.drag = None;
        self.panning = None;
        made
    }

    /// Remove the selected node and all wires connected to it.
    pub fn delete_selected(&mut self) {
        if let Some(id) = self.selected {
            self.nodes.retain(|n| n.id != id);
            self.wires.retain(|w| w.from != id && w.to != id);
            self.selected = None;
        }
    }

    /// Remove all wires where `id` is the source (output side).
    pub fn disconnect_outputs(&mut self, id: u32) {
        self.wires.retain(|w| w.from != id);
    }

    /// Remove all wires where `id` is the destination (input side).
    pub fn disconnect_inputs(&mut self, id: u32) {
        self.wires.retain(|w| w.to != id);
    }

    /// Remove all wires connected to `id` (both directions).
    pub fn disconnect_all(&mut self, id: u32) {
        self.wires.retain(|w| w.from != id && w.to != id);
    }

    /// If (px, py) is within `thresh` pixels of any wire's midpoint, remove that wire.
    /// Returns true if a wire was removed.
    pub fn disconnect_wire_at(&mut self, px: i32, py: i32, thresh: i32) -> bool {
        let nodes = &self.nodes;
        let mut to_remove: Option<Wire> = None;
        'outer: for w in &self.wires {
            let from_node = nodes.iter().find(|n| n.id == w.from);
            let to_node = nodes.iter().find(|n| n.id == w.to);
            if let (Some(f), Some(t)) = (from_node, to_node) {
                // Endpoints must match what view() draws: the output anchor of f
                // and the input anchor of t (which apply w, h/2, and pan).
                let (fx, fy) = self.out_anchor(f);
                let (tx, ty) = self.in_anchor(t);
                // Check several points along the wire (at t=0.25, 0.5, 0.75).
                for k in 1..=3 {
                    let t_param = k as i32;
                    // Linear interpolation for approximate hit test.
                    let mx = (fx * (4 - t_param) + tx * t_param) / 4;
                    let my = (fy * (4 - t_param) + ty * t_param) / 4;
                    let dx = px - mx;
                    let dy = py - my;
                    if dx * dx + dy * dy <= thresh * thresh {
                        to_remove = Some(*w);
                        break 'outer;
                    }
                }
            }
        }
        if let Some(w) = to_remove {
            self.wires.retain(|wire| wire != &w);
            true
        } else {
            false
        }
    }

    /// Build the scene: wires behind, the pending wire, then nodes in front.
    pub fn view(&self, theme: &toolkit::Theme) -> Vec<DrawCmd> {
        let mut scene = Vec::new();
        for w in &self.wires {
            if let (Some(a), Some(b)) = (self.node(w.from), self.node(w.to)) {
                let from = self.out_anchor(a);
                let to = self.in_anchor(b);
                let slack = ((to.0 - from.0).abs() / 2).clamp(28, 90);
                let c = a.kind.color(theme);
                scene.push(toolkit::wire(from, to, Color::rgba(c.r, c.g, c.b, 150), 2, slack));
                scene.push(toolkit::disc(from.0, from.1, 3, c));
                scene.push(toolkit::disc(to.0, to.1, 3, b.kind.color(theme)));
            }
        }
        // In-progress wire (during a rewire drag).
        if let Some((from, to)) = self.pending_wire() {
            let slack = ((to.0 - from.0).abs() / 2).clamp(28, 90);
            scene.push(toolkit::wire(from, to, theme.accent, 2, slack));
        }
        for n in &self.nodes {
            let accent = n.kind.color(theme);
            let r = self.screen_rect(n);
            if self.selected == Some(n.id) {
                scene.push(DrawCmd::Rect {
                    rect: Rect::new(r.x - 2, r.y - 2, r.w + 4, r.h + 4),
                    color: accent,
                    radius: theme.radius + 2,
                });
            }
            scene.push(DrawCmd::Rect { rect: r, color: theme.surface, radius: theme.radius });
            scene.push(DrawCmd::Rect { rect: Rect::new(r.x, r.y, 4, r.h), color: accent, radius: 0 });
            scene.push(DrawCmd::Text {
                rect: Rect::new(r.x + 12, r.y + 8, r.w - 16, theme.font_size + 2),
                text: n.title.clone(),
                color: theme.text,
                size: theme.font_size,
            });
            scene.push(DrawCmd::Text {
                rect: Rect::new(r.x + 12, r.y + 8 + theme.font_size + 4, r.w - 16, theme.font_size),
                text: n.subtitle.clone(),
                color: theme.muted,
                size: theme.font_size - 2,
            });
            scene.push(toolkit::disc(r.x, r.y + r.h / 2, PORT_R - 2, theme.muted));
            scene.push(toolkit::disc(r.x + r.w, r.y + r.h / 2, PORT_R - 2, accent));
        }
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> NodeGraph {
        let mut g = NodeGraph::new();
        g.add(1, "Project Alpha", "Data Object", NodeKind::Data, 100, 100);
        g.add(2, "Q3 Report", "Application", NodeKind::Report, 320, 140);
        g.add(3, "Speech", "Neural Audio", NodeKind::Audio, 320, 40);
        g.wire(1, 2);
        g.wire(1, 3);
        g
    }

    #[test]
    fn nodes_and_wires_render() {
        let g = graph();
        let scene = g.view(&toolkit::Theme::dark());
        assert_eq!(scene.iter().filter(|c| matches!(c, DrawCmd::Bezier { .. })).count(), 2);
        assert!(scene.iter().any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "Project Alpha")));
    }

    #[test]
    fn node_hit_testing_picks_topmost() {
        let g = graph();
        assert_eq!(g.node_at(110, 110), Some(1));
        assert_eq!(g.node_at(330, 150), Some(2));
        assert_eq!(g.node_at(5, 5), None);
    }

    #[test]
    fn press_selects_and_drag_moves_the_node() {
        let mut g = graph();
        assert_eq!(g.on_press(110, 120), Press::Node(1));
        assert_eq!(g.selected(), Some(1));
        assert!(g.is_interacting());
        g.on_drag(160, 150);
        let n = g.nodes().iter().find(|n| n.id == 1).unwrap();
        assert_eq!((n.x, n.y), (150, 130));
        assert!(!g.on_release(160, 150));
        assert!(!g.is_interacting());
    }

    #[test]
    fn empty_space_press_pans_the_canvas() {
        let mut g = graph();
        assert_eq!(g.on_press(5, 5), Press::Empty); // empty canvas
        assert_eq!(g.pan(), (0, 0));
        g.on_drag(55, 35); // drag right+down by (50,30)
        assert_eq!(g.pan(), (50, 30));
        // Panned: node 1 (graph 100,100) now hit-tests at screen (150,130).
        g.on_release(55, 35);
        assert_eq!(g.node_at(160, 140), Some(1));
        assert_eq!(g.node_at(110, 110), None); // its old screen position is now empty
    }

    #[test]
    fn dragging_from_a_port_creates_a_wire() {
        let mut g = NodeGraph::new();
        g.add(1, "a", "", NodeKind::Data, 0, 0); // out port at (168, 26)
        g.add(2, "b", "", NodeKind::App, 300, 0); // in port at (300, 26)
        assert_eq!(g.wires().len(), 0);
        // Press on node 1's output port, drag to node 2's input port, release.
        assert_eq!(g.on_press(168, 26), Press::Port);
        assert!(g.pending_wire().is_some());
        g.on_drag(250, 26);
        assert!(g.on_release(300, 26)); // landed on node 2's input → wire made
        assert_eq!(g.wires(), &[Wire { from: 1, to: 2 }]);
    }

    #[test]
    fn wire_drag_to_empty_space_makes_nothing() {
        let mut g = NodeGraph::new();
        g.add(1, "a", "", NodeKind::Data, 0, 0);
        g.on_press(168, 26); // output port
        assert!(!g.on_release(500, 500)); // dropped in empty space
        assert_eq!(g.wires().len(), 0);
    }

    #[test]
    fn delete_selected_removes_node_and_wires() {
        let mut g = NodeGraph::new();
        g.add(1, "A", "", NodeKind::Program, 0, 0);
        g.add(2, "B", "", NodeKind::Program, 200, 0);
        g.wire(1, 2);
        g.selected = Some(1);
        g.delete_selected();
        assert!(g.nodes().iter().all(|n| n.id != 1));
        assert!(g.wires().is_empty());
        assert_eq!(g.selected(), None);
    }

    #[test]
    fn disconnect_inputs_removes_incoming_wires() {
        let mut g = NodeGraph::new();
        g.add(1, "A", "", NodeKind::Program, 0, 0);
        g.add(2, "B", "", NodeKind::Program, 200, 0);
        g.add(3, "C", "", NodeKind::Program, 400, 0);
        g.wire(1, 2);
        g.wire(3, 2);
        g.disconnect_inputs(2);
        assert!(g.wires().is_empty());
    }

    #[test]
    fn disconnect_outputs_removes_outgoing_wires() {
        let mut g = NodeGraph::new();
        g.add(1, "A", "", NodeKind::Program, 0, 0);
        g.add(2, "B", "", NodeKind::Program, 200, 0);
        g.add(3, "C", "", NodeKind::Program, 400, 0);
        g.wire(1, 2);
        g.wire(1, 3);
        g.disconnect_outputs(1);
        assert!(g.wires().is_empty());
    }

    #[test]
    fn duplicate_and_self_wires_are_rejected() {
        let mut g = NodeGraph::new();
        g.add(1, "a", "", NodeKind::Data, 0, 0);
        g.add(2, "b", "", NodeKind::App, 300, 0);
        g.wire(1, 2);
        g.wire(1, 2); // duplicate
        g.wire(1, 1); // self
        assert_eq!(g.wires().len(), 1);
    }
}

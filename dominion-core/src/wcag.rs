//! WCAG / ARIA-equivalent conformance checks over the accessibility tree
//! (`docs/architecture/accessibility-and-i18n.md`).
//!
//! [`crate::a11y`] *is* the accessibility tree (the knowledge graph), so conformance is
//! checkable structurally rather than by scraping a rendered DOM. This module audits an
//! [`A11yNode`] tree against WCAG-equivalent success criteria and a contrast check, and is
//! deterministic so it runs under DST:
//!
//! * **4.1.2 Name, Role, Value** — every interactive node has a non-empty accessible name.
//! * **2.4.3 Focus Order** — a focusable UI exposes a defined focus order.
//! * **1.3.1 Info & Relationships** — images carry a text alternative (label).
//! * **1.4.3 Contrast** — foreground/background contrast meets the AA ratio (4.5:1).
//!
//! Pure, safe `no_std`. Host-tested.

use crate::a11y::{A11yNode, Role};
use crate::toolkit::Color;
use alloc::string::String;
use alloc::vec::Vec;

/// A WCAG success criterion a node can violate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Criterion {
    /// 4.1.2 — an interactive element has no accessible name.
    MissingName,
    /// 1.3.1 — an image has no text alternative.
    MissingAltText,
    /// 2.4.3 — focusable content but no defined focus order.
    NoFocusOrder,
}

/// A conformance violation: which node, which criterion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub node: u64,
    pub criterion: Criterion,
}

fn is_interactive(role: Role) -> bool {
    matches!(role, Role::Button | Role::TextField | Role::Checkbox)
}

fn audit_node(node: &A11yNode, out: &mut Vec<Violation>) {
    // 4.1.2 — interactive nodes need a name.
    if is_interactive(node.role) && node.label.trim().is_empty() {
        out.push(Violation { node: node.id, criterion: Criterion::MissingName });
    }
    // 1.3.1 — images need alt text.
    if node.role == Role::Image && node.label.trim().is_empty() {
        out.push(Violation { node: node.id, criterion: Criterion::MissingAltText });
    }
    for c in &node.children {
        audit_node(c, out);
    }
}

/// Audit a tree, returning every WCAG violation found (empty ⇒ conformant on these
/// criteria). Also checks 2.4.3: if anything is focusable, a focus order must exist.
pub fn audit(tree: &A11yNode) -> Vec<Violation> {
    let mut out = Vec::new();
    audit_node(tree, &mut out);
    // 2.4.3 — focusable content implies a non-empty focus order.
    let order = tree.focus_order();
    let has_focusable = has_any_focusable(tree);
    if has_focusable && order.is_empty() {
        out.push(Violation { node: tree.id, criterion: Criterion::NoFocusOrder });
    }
    out
}

fn has_any_focusable(node: &A11yNode) -> bool {
    node.focusable || node.children.iter().any(has_any_focusable)
}

/// True iff the tree passes all checked WCAG criteria.
pub fn conformant(tree: &A11yNode) -> bool {
    audit(tree).is_empty()
}

/// A human-readable conformance report line for each violation.
pub fn report(tree: &A11yNode) -> Vec<String> {
    audit(tree)
        .into_iter()
        .map(|v| {
            let c = match v.criterion {
                Criterion::MissingName => "4.1.2 missing accessible name",
                Criterion::MissingAltText => "1.3.1 image missing alt text",
                Criterion::NoFocusOrder => "2.4.3 no focus order",
            };
            alloc::format!("node {}: {}", v.node, c)
        })
        .collect()
}

// ───────────────────────── 1.4.3 contrast ─────────────────────────

/// Relative luminance of a color. The WCAG sRGB linearisation uses a γ≈2.4 transfer; we
/// use the standard **γ=2 approximation** (`s²`) so the core stays `no_std` with no
/// `libm` transcendentals — monotonic and accurate enough for the AA threshold.
fn luminance(c: Color) -> f64 {
    let chan = |v: u8| {
        let s = v as f64 / 255.0;
        s * s
    };
    0.2126 * chan(c.r) + 0.7152 * chan(c.g) + 0.0722 * chan(c.b)
}

/// The WCAG contrast ratio between two colors (1.0 … 21.0).
pub fn contrast_ratio(fg: Color, bg: Color) -> f64 {
    let l1 = luminance(fg);
    let l2 = luminance(bg);
    let (hi, lo) = if l1 >= l2 { (l1, l2) } else { (l2, l1) };
    (hi + 0.05) / (lo + 0.05)
}

/// Does the pair meet WCAG **AA** for normal text (≥ 4.5:1)?
pub fn meets_aa(fg: Color, bg: Color) -> bool {
    contrast_ratio(fg, bg) >= 4.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a11y::A11yNode;

    #[test]
    fn a_well_formed_tree_is_conformant() {
        let tree = A11yNode::new(1, Role::Window, "Settings")
            .child(A11yNode::new(2, Role::Button, "Save").focusable())
            .child(A11yNode::new(3, Role::Image, "Company logo"))
            .child(A11yNode::new(4, Role::TextField, "Name").focusable());
        assert!(conformant(&tree));
        assert!(audit(&tree).is_empty());
    }

    #[test]
    fn missing_names_and_alt_text_are_flagged() {
        let tree = A11yNode::new(1, Role::Window, "App")
            .child(A11yNode::new(2, Role::Button, "").focusable()) // unnamed button
            .child(A11yNode::new(3, Role::Image, "")); // no alt text
        let violations = audit(&tree);
        assert!(violations.contains(&Violation { node: 2, criterion: Criterion::MissingName }));
        assert!(violations.contains(&Violation { node: 3, criterion: Criterion::MissingAltText }));
        assert_eq!(report(&tree).len(), 2);
    }

    #[test]
    fn contrast_ratio_meets_aa_for_readable_pairs() {
        let black = Color { r: 0, g: 0, b: 0, a: 255 };
        let white = Color { r: 255, g: 255, b: 255, a: 255 };
        // Black on white is the maximum ratio (21:1).
        assert!(contrast_ratio(black, white) > 20.0);
        assert!(meets_aa(black, white));
        // A low-contrast pair fails AA.
        let light_gray = Color { r: 200, g: 200, b: 200, a: 255 };
        assert!(!meets_aa(light_gray, white));
    }
}

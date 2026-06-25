//! Internationalization depth — **an IME framework, CLDR-style locale formatting, and
//! bidi text shaping with font fallback** (`docs/architecture/accessibility-and-i18n.md`).
//!
//! [`crate::a11y`] carries the semantic tree + a locale catalog + RTL direction; this
//! module adds the text-input and text-layout machinery a real multilingual OS needs:
//!
//! * [`InputMethod`] — an **IME** for CJK / complex scripts: keystrokes accumulate in a
//!   composition buffer, a lookup yields candidate characters, and the user commits one.
//! * [`LocaleFormat`] — **CLDR-style locale formatting** of numbers, currency and dates
//!   (grouping + decimal separators, currency symbol/placement, date order per locale).
//! * [`shape`] / [`FontStack`] — **bidirectional reordering** (resolve LTR/RTL runs into
//!   visual order) and **font fallback** (pick the first font that covers each codepoint).
//!
//! Pure, safe `no_std`; deterministic. Host-tested. (Full Unicode UAX#9 / UAX#14 and a
//! real shaping engine remain future work; the algorithms here are faithful, reduced.)

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

// ───────────────────────── IME framework ─────────────────────────

/// An input method: a composition buffer + a phonetic→character candidate table. Models
/// pinyin/romaji-style input where a latin key sequence maps to candidate glyphs.
pub struct InputMethod {
    table: BTreeMap<String, Vec<String>>,
    composition: String,
}

impl InputMethod {
    /// Build an IME from a candidate table (reading → ordered candidates).
    pub fn new(table: BTreeMap<String, Vec<String>>) -> InputMethod {
        InputMethod { table, composition: String::new() }
    }

    /// A small pinyin demo table (illustrative).
    pub fn pinyin_demo() -> InputMethod {
        let mut t = BTreeMap::new();
        t.insert(String::from("ni"), alloc::vec![String::from("你"), String::from("尼")]);
        t.insert(String::from("hao"), alloc::vec![String::from("好"), String::from("号")]);
        t.insert(String::from("ma"), alloc::vec![String::from("吗"), String::from("妈")]);
        InputMethod::new(t)
    }

    /// Feed a key into the composition buffer.
    pub fn feed(&mut self, key: char) {
        self.composition.push(key);
    }

    /// The current (pre-commit) composition string.
    pub fn composition(&self) -> &str {
        &self.composition
    }

    /// Candidate characters for the current composition (empty if none).
    pub fn candidates(&self) -> &[String] {
        self.table.get(&self.composition).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Commit candidate `index`, clearing the composition. Returns the committed text, or
    /// `None` if the index is out of range. With no candidates, commits the raw buffer.
    pub fn commit(&mut self, index: usize) -> Option<String> {
        let chosen = match self.table.get(&self.composition) {
            Some(cands) => cands.get(index)?.clone(),
            None => core::mem::take(&mut self.composition),
        };
        self.composition.clear();
        Some(chosen)
    }

    /// Backspace the composition buffer.
    pub fn backspace(&mut self) {
        self.composition.pop();
    }
}

// ───────────────────────── CLDR-style locale formatting ─────────────────────────

/// The order a locale writes dates in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DateOrder {
    /// Year-Month-Day (e.g. ISO, ja-JP).
    Ymd,
    /// Day-Month-Year (e.g. en-GB, de-DE).
    Dmy,
    /// Month-Day-Year (e.g. en-US).
    Mdy,
}

/// A locale's formatting conventions (a reduced CLDR record).
#[derive(Clone, Copy, Debug)]
pub struct LocaleFormat {
    pub group_sep: char,
    pub decimal_sep: char,
    pub currency_symbol: &'static str,
    /// Currency symbol before the amount (`$1` vs `1 €`).
    pub symbol_first: bool,
    pub date_order: DateOrder,
}

impl LocaleFormat {
    /// `en-US`: `1,234.50`, `$1,234.50`, `M/D/Y`.
    pub fn en_us() -> LocaleFormat {
        LocaleFormat { group_sep: ',', decimal_sep: '.', currency_symbol: "$", symbol_first: true, date_order: DateOrder::Mdy }
    }

    /// `de-DE`: `1.234,50`, `1.234,50 €`, `D.M.Y`.
    pub fn de_de() -> LocaleFormat {
        LocaleFormat { group_sep: '.', decimal_sep: ',', currency_symbol: "€", symbol_first: false, date_order: DateOrder::Dmy }
    }

    /// Format an integer with thousands grouping.
    pub fn format_integer(&self, value: i64) -> String {
        let neg = value < 0;
        let digits = {
            let mut d = String::new();
            let mut n = value.unsigned_abs();
            if n == 0 {
                d.push('0');
            }
            while n > 0 {
                d.push((b'0' + (n % 10) as u8) as char);
                n /= 10;
            }
            d
        };
        // `digits` is reversed; insert a group separator every 3.
        let mut out = String::new();
        for (i, ch) in digits.chars().enumerate() {
            if i > 0 && i % 3 == 0 {
                out.push(self.group_sep);
            }
            out.push(ch);
        }
        let grouped: String = out.chars().rev().collect();
        if neg {
            let mut s = String::from("-");
            s.push_str(&grouped);
            s
        } else {
            grouped
        }
    }

    /// Format a fixed-point value (value is in `10^scale` units, e.g. cents for scale=2).
    pub fn format_decimal(&self, value: i64, scale: u32) -> String {
        if scale == 0 {
            return self.format_integer(value);
        }
        let negative = value < 0;
        let divisor = 10i64.pow(scale);
        let int_part = value / divisor;
        let frac = (value % divisor).unsigned_abs();
        // When the value is negative but the integer part rounds to zero (e.g. -0.50),
        // format_integer(0) produces "0" with no sign; prepend "-" explicitly.
        let mut s = if negative && int_part == 0 {
            String::from("-0")
        } else {
            self.format_integer(int_part)
        };
        s.push(self.decimal_sep);
        // Zero-pad the fraction to `scale` digits.
        let frac_str = {
            let mut f = String::new();
            let raw = {
                let mut d = String::new();
                let mut n = frac;
                if n == 0 { d.push('0'); }
                while n > 0 { d.push((b'0' + (n % 10) as u8) as char); n /= 10; }
                d
            };
            let raw: String = raw.chars().rev().collect();
            for _ in 0..(scale as usize).saturating_sub(raw.len()) {
                f.push('0');
            }
            f.push_str(&raw);
            f
        };
        s.push_str(&frac_str);
        s
    }

    /// Format a currency amount (minor units, e.g. cents). `1234` → `$12.34` / `12,34 €`.
    pub fn format_currency(&self, minor: i64) -> String {
        let amount = self.format_decimal(minor, 2);
        if self.symbol_first {
            let mut s = String::from(self.currency_symbol);
            s.push_str(&amount);
            s
        } else {
            let mut s = amount;
            s.push(' ');
            s.push_str(self.currency_symbol);
            s
        }
    }

    /// Format a calendar date in the locale's field order, zero-padded.
    pub fn format_date(&self, year: u32, month: u32, day: u32) -> String {
        let y = {
            let mut s = String::new();
            let raw = alloc::format!("{year}");
            for _ in 0..4usize.saturating_sub(raw.len()) { s.push('0'); }
            s.push_str(&raw);
            s
        };
        let m = alloc::format!("{month:02}");
        let d = alloc::format!("{day:02}");
        match self.date_order {
            DateOrder::Ymd => alloc::format!("{y}-{m}-{d}"),
            DateOrder::Dmy => alloc::format!("{d}.{m}.{y}"),
            DateOrder::Mdy => alloc::format!("{m}/{d}/{y}"),
        }
    }
}

// ───────────────────────── bidi reordering + font fallback ─────────────────────────

/// The bidi class of a character (reduced UAX#9: just the strong directions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BidiClass {
    /// Left-to-right (latin, CJK, digits-as-LTR here).
    Ltr,
    /// Right-to-left (Hebrew, Arabic).
    Rtl,
    /// Neutral (spaces, punctuation) — takes the surrounding run's direction.
    Neutral,
}

/// Classify a character's strong bidi direction.
pub fn bidi_class(c: char) -> BidiClass {
    let u = c as u32;
    // Hebrew (0590–05FF) and Arabic (0600–06FF, 0750–077F) are RTL.
    if (0x0590..=0x05FF).contains(&u) || (0x0600..=0x06FF).contains(&u) || (0x0750..=0x077F).contains(&u) {
        BidiClass::Rtl
    } else if c.is_whitespace() || c.is_ascii_punctuation() {
        BidiClass::Neutral
    } else {
        BidiClass::Ltr
    }
}

/// Reorder a string from logical to **visual** order given a base paragraph direction.
/// A reduced UAX#9: contiguous RTL runs (RTL chars + the neutrals between them) are
/// reversed in place; an RTL base paragraph reverses the whole line then re-flips LTR
/// runs. Faithful for mixed Hebrew/Arabic + Latin, which is the common case.
pub fn shape(logical: &str, base_rtl: bool) -> String {
    let chars: Vec<char> = logical.chars().collect();
    if chars.is_empty() {
        return String::new();
    }
    // Resolve each char's direction (neutrals follow the previous strong, else base).
    let base = if base_rtl { BidiClass::Rtl } else { BidiClass::Ltr };
    let mut dirs = Vec::with_capacity(chars.len());
    let mut last_strong = base;
    for &c in &chars {
        let d = match bidi_class(c) {
            BidiClass::Neutral => last_strong,
            strong => {
                last_strong = strong;
                strong
            }
        };
        dirs.push(d);
    }
    // Build visual order: walk runs of equal direction; reverse runs opposite to base.
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut j = i;
        while j < chars.len() && dirs[j] == dirs[i] {
            j += 1;
        }
        let run = &chars[i..j];
        if dirs[i] != base {
            out.extend(run.iter().rev());
        } else {
            out.extend(run.iter());
        }
        i = j;
    }
    if base_rtl {
        out.reverse();
    }
    out.into_iter().collect()
}

/// A font with a name and the set of codepoints it covers.
#[derive(Clone, Debug)]
pub struct Font {
    pub name: &'static str,
    coverage: Vec<(u32, u32)>, // inclusive ranges
}

impl Font {
    pub fn new(name: &'static str, ranges: &[(u32, u32)]) -> Font {
        Font { name, coverage: ranges.to_vec() }
    }

    /// Does this font cover `c`?
    pub fn covers(&self, c: char) -> bool {
        let u = c as u32;
        self.coverage.iter().any(|&(lo, hi)| (lo..=hi).contains(&u))
    }
}

/// An ordered font-fallback stack: the first font covering a codepoint wins.
pub struct FontStack {
    fonts: Vec<Font>,
}

impl FontStack {
    pub fn new(fonts: Vec<Font>) -> FontStack {
        FontStack { fonts }
    }

    /// The font that should render `c`, or `None` if no font in the stack covers it
    /// (the caller draws a `.notdef` / tofu box).
    pub fn font_for(&self, c: char) -> Option<&Font> {
        self.fonts.iter().find(|f| f.covers(c))
    }

    /// Split a string into runs of `(font_name, text)` so each run renders with one font.
    pub fn itemize(&self, text: &str) -> Vec<(&'static str, String)> {
        let mut runs: Vec<(&'static str, String)> = Vec::new();
        for c in text.chars() {
            let name = self.font_for(c).map(|f| f.name).unwrap_or(".notdef");
            match runs.last_mut() {
                Some((n, s)) if *n == name => s.push(c),
                _ => runs.push((name, {
                    let mut s = String::new();
                    s.push(c);
                    s
                })),
            }
        }
        runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ime_composes_and_commits_candidates() {
        let mut ime = InputMethod::pinyin_demo();
        for k in "ni".chars() {
            ime.feed(k);
        }
        assert_eq!(ime.composition(), "ni");
        assert_eq!(ime.candidates(), &[String::from("你"), String::from("尼")]);
        assert_eq!(ime.commit(0).as_deref(), Some("你"));
        assert_eq!(ime.composition(), ""); // cleared
        // Unknown reading commits the raw buffer.
        for k in "xyz".chars() {
            ime.feed(k);
        }
        assert!(ime.candidates().is_empty());
        assert_eq!(ime.commit(0).as_deref(), Some("xyz"));
    }

    #[test]
    fn locale_number_currency_and_date_formatting() {
        let us = LocaleFormat::en_us();
        let de = LocaleFormat::de_de();
        assert_eq!(us.format_integer(1234567), "1,234,567");
        assert_eq!(de.format_integer(1234567), "1.234.567");
        assert_eq!(us.format_currency(123456), "$1,234.56");
        assert_eq!(de.format_currency(123456), "1.234,56 €");
        assert_eq!(us.format_date(2026, 6, 20), "06/20/2026");
        assert_eq!(de.format_date(2026, 6, 20), "20.06.2026");
        assert_eq!(us.format_integer(-42), "-42");
        // Negative value where the integer part is zero: sign must not be lost.
        assert_eq!(us.format_decimal(-5, 1), "-0.5");
        assert_eq!(us.format_decimal(-15, 2), "-0.15");
        assert_eq!(de.format_decimal(-15, 2), "-0,15");
    }

    #[test]
    fn bidi_reorders_rtl_runs_into_visual_order() {
        // "abc" + Hebrew "אבג" in an LTR paragraph: the Hebrew run is reversed visually.
        let logical = "abcאבג";
        let visual = shape(logical, false);
        // Latin stays, Hebrew letters appear in reversed (visual) order.
        let v: Vec<char> = visual.chars().collect();
        assert_eq!(&v[..3], &['a', 'b', 'c']);
        assert_eq!(&v[3..], &['ג', 'ב', 'א']);
        // Pure LTR is unchanged.
        assert_eq!(shape("hello", false), "hello");
    }

    #[test]
    fn font_fallback_itemizes_into_covered_runs() {
        let latin = Font::new("Noto Sans", &[(0x0020, 0x024F)]);
        let cjk = Font::new("Noto Sans CJK", &[(0x4E00, 0x9FFF)]);
        let stack = FontStack::new(alloc::vec![latin, cjk]);
        assert_eq!(stack.font_for('A').unwrap().name, "Noto Sans");
        assert_eq!(stack.font_for('好').unwrap().name, "Noto Sans CJK");
        // An uncovered codepoint (emoji) → no font.
        assert!(stack.font_for('😀').is_none());
        let runs = stack.itemize("Aa好b");
        assert_eq!(runs, alloc::vec![
            ("Noto Sans", String::from("Aa")),
            ("Noto Sans CJK", String::from("好")),
            ("Noto Sans", String::from("b")),
        ]);
    }
}

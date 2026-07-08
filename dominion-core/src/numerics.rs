//! High-precision & non-real numeric primitives for the Dominion language.
//!
//! `datatypes.rs` holds the *semantic* primitives (tensors, hypervectors, spike
//! trains, …). This module holds the **numeric** primitives: the shapes a value
//! takes when it must be *exact* or carry *rigorous error bounds*, rather than be
//! a lossy 64-bit IEEE float.
//!
//! The headline type is [`Decimal`] — an **arbitrary-precision base-10 number**.
//! Where an `f64` cannot even represent `0.1` (so `0.1 + 0.2 != 0.3`), a `Decimal`
//! stores the digits themselves and computes on them, so decimal arithmetic is
//! *exact* and division rounds to a caller-chosen precision (tens to hundreds of
//! digits). It is backed by [`BigInt`], a sign-magnitude arbitrary-precision
//! integer.
//!
//! The roster:
//!
//! * [`BigInt`]     — arbitrary-precision integer (no overflow, ever).
//! * [`Decimal`]    — arbitrary-precision decimal; the "insanely accurate float".
//! * [`Rational`]   — an exact fraction `p/q` of two [`BigInt`]s (zero rounding).
//! * [`Complex`]    — a complex number `a + bi`.
//! * [`Dual`]       — a dual number `a + bε` for exact forward-mode autodiff.
//! * [`Interval`]   — a rigorous error-bounded range `[lo, hi]`.
//! * [`Quaternion`] — a quaternion `w + xi + yj + zk` for 3-D rotation algebra.
//!
//! Every type is pure, safe, `no_std + alloc` and host-tested. No special hardware
//! is required; where an accelerator exists the runtime may offload, but the
//! semantics here are the ground truth.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{format, vec};
use core::cmp::Ordering;
use core::fmt;

use crate::datatypes::sqrt as fsqrt;

// ════════════════════════════════ BigInt ════════════════════════════════

/// A sign-magnitude arbitrary-precision integer. The magnitude is stored
/// little-endian in base `2^32` limbs, always normalised (no trailing zero limbs;
/// zero is the empty magnitude with `neg = false`).
///
/// This is the exact-integer substrate beneath [`Decimal`] and [`Rational`]: it
/// never overflows, so a `Decimal` can hold as many significant digits as memory
/// allows.
#[derive(Clone, Debug)]
pub struct BigInt {
    neg: bool,
    mag: Vec<u32>, // little-endian base 2^32
}

impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for BigInt {}

impl Ord for BigInt {
    /// Total order over the integers (sign-aware).
    fn cmp(&self, other: &BigInt) -> Ordering {
        match (self.sign(), other.sign()) {
            (a, b) if a != b => a.cmp(&b),
            (0, 0) => Ordering::Equal,
            _ => {
                let m = cmp_mag(&self.mag, &other.mag);
                if self.neg {
                    m.reverse()
                } else {
                    m
                }
            }
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &BigInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl BigInt {
    /// The integer zero.
    pub fn zero() -> BigInt {
        BigInt { neg: false, mag: Vec::new() }
    }

    /// `1` as a [`BigInt`].
    pub fn one() -> BigInt {
        BigInt { neg: false, mag: vec![1] }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    /// `-1`, `0`, or `+1`.
    pub fn sign(&self) -> i32 {
        if self.mag.is_empty() {
            0
        } else if self.neg {
            -1
        } else {
            1
        }
    }

    pub fn from_i64(v: i64) -> BigInt {
        let neg = v < 0;
        let mut u = (v as i128).unsigned_abs();
        let mut mag = Vec::new();
        while u != 0 {
            mag.push((u & 0xFFFF_FFFF) as u32);
            u >>= 32;
        }
        BigInt { neg: neg && !mag.is_empty(), mag }
    }

    fn from_mag(neg: bool, mut mag: Vec<u32>) -> BigInt {
        while mag.last() == Some(&0) {
            mag.pop();
        }
        BigInt { neg: neg && !mag.is_empty(), mag }
    }

    pub fn abs(&self) -> BigInt {
        BigInt { neg: false, mag: self.mag.clone() }
    }

    pub fn neg(&self) -> BigInt {
        BigInt { neg: !self.neg && !self.mag.is_empty(), mag: self.mag.clone() }
    }

    pub fn add(&self, other: &BigInt) -> BigInt {
        if self.neg == other.neg {
            BigInt::from_mag(self.neg, add_mag(&self.mag, &other.mag))
        } else {
            match cmp_mag(&self.mag, &other.mag) {
                Ordering::Equal => BigInt::zero(),
                Ordering::Greater => BigInt::from_mag(self.neg, sub_mag(&self.mag, &other.mag)),
                Ordering::Less => BigInt::from_mag(other.neg, sub_mag(&other.mag, &self.mag)),
            }
        }
    }

    pub fn sub(&self, other: &BigInt) -> BigInt {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &BigInt) -> BigInt {
        if self.is_zero() || other.is_zero() {
            return BigInt::zero();
        }
        BigInt::from_mag(self.neg != other.neg, mul_mag(&self.mag, &other.mag))
    }

    /// Truncated division: returns `(quotient, remainder)` with the remainder
    /// taking the dividend's sign (`self == q*other + r`). `None` if `other == 0`.
    pub fn divmod(&self, other: &BigInt) -> Option<(BigInt, BigInt)> {
        if other.is_zero() {
            return None;
        }
        let (q, r) = divmod_mag(&self.mag, &other.mag);
        let quo = BigInt::from_mag(self.neg != other.neg, q);
        let rem = BigInt::from_mag(self.neg, r);
        Some((quo, rem))
    }

    /// `self * 10^n` (`n >= 0`).
    pub fn mul_pow10(&self, n: u64) -> BigInt {
        if self.is_zero() || n == 0 {
            return self.clone();
        }
        let mut mag = self.mag.clone();
        // multiply by 10, n times, in chunks of 9 (10^9 fits in u32).
        let mut left = n;
        while left > 0 {
            let step = left.min(9);
            let factor = 10u32.pow(step as u32);
            mul_small_inplace(&mut mag, factor);
            left -= step;
        }
        BigInt::from_mag(self.neg, mag)
    }

    /// Greatest common divisor of the magnitudes (always non-negative).
    pub fn gcd(&self, other: &BigInt) -> BigInt {
        let mut a = self.abs();
        let mut b = other.abs();
        while !b.is_zero() {
            let (_, r) = a.divmod(&b).unwrap();
            a = b;
            b = r.abs();
        }
        a
    }

    /// Base-10 string (with a leading `-` for negatives).
    pub fn to_decimal_string(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut mag = self.mag.clone();
        let mut groups: Vec<u32> = Vec::new();
        while !mag.is_empty() {
            let rem = divmod_small_inplace(&mut mag, 1_000_000_000);
            groups.push(rem);
        }
        let mut s = String::new();
        if self.neg {
            s.push('-');
        }
        // most-significant group has no leading zeros; the rest are zero-padded to 9.
        let last = groups.len() - 1;
        s.push_str(&groups[last].to_string());
        for i in (0..last).rev() {
            s.push_str(&format!("{:09}", groups[i]));
        }
        s
    }

    /// Parse a base-10 integer (optional leading `+`/`-`). `None` on any non-digit.
    pub fn from_decimal_str(s: &str) -> Option<BigInt> {
        let s = s.trim();
        let (neg, digits) = match s.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut mag: Vec<u32> = Vec::new();
        // accumulate 9 digits at a time
        let bytes = digits.as_bytes();
        let mut i = 0;
        // align so the first chunk may be short
        let first = bytes.len() % 9;
        if first != 0 {
            let chunk = core::str::from_utf8(&bytes[..first]).ok()?;
            let v: u32 = chunk.parse().ok()?;
            mag = vec![v];
            i = first;
        }
        while i < bytes.len() {
            let chunk = core::str::from_utf8(&bytes[i..i + 9]).ok()?;
            let v: u32 = chunk.parse().ok()?;
            mul_small_inplace(&mut mag, 1_000_000_000);
            add_small_inplace(&mut mag, v);
            i += 9;
        }
        Some(BigInt::from_mag(neg, mag))
    }
}

// ---- magnitude helpers (operate on normalised little-endian base-2^32 slices) ----

fn cmp_mag(a: &[u32], b: &[u32]) -> Ordering {
    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

fn normalize(mag: &mut Vec<u32>) {
    while mag.last() == Some(&0) {
        mag.pop();
    }
}

fn add_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u64;
    for i in 0..a.len().max(b.len()) {
        let av = *a.get(i).unwrap_or(&0) as u64;
        let bv = *b.get(i).unwrap_or(&0) as u64;
        let s = av + bv + carry;
        out.push((s & 0xFFFF_FFFF) as u32);
        carry = s >> 32;
    }
    if carry != 0 {
        out.push(carry as u32);
    }
    out
}

/// `a - b`, requires `a >= b` (caller guarantees).
fn sub_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for (i, &limb) in a.iter().enumerate() {
        let av = limb as i64;
        let bv = *b.get(i).unwrap_or(&0) as i64;
        let mut d = av - bv - borrow;
        if d < 0 {
            d += 1 << 32;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(d as u32);
    }
    normalize(&mut out);
    out
}

fn mul_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &av) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &bv) in b.iter().enumerate() {
            let idx = i + j;
            let cur = out[idx] as u64 + av as u64 * bv as u64 + carry;
            out[idx] = (cur & 0xFFFF_FFFF) as u32;
            carry = cur >> 32;
        }
        let mut idx = i + b.len();
        while carry != 0 {
            let cur = out[idx] as u64 + carry;
            out[idx] = (cur & 0xFFFF_FFFF) as u32;
            carry = cur >> 32;
            idx += 1;
        }
    }
    normalize(&mut out);
    out
}

fn mul_small_inplace(mag: &mut Vec<u32>, k: u32) {
    let mut carry = 0u64;
    for limb in mag.iter_mut() {
        let cur = *limb as u64 * k as u64 + carry;
        *limb = (cur & 0xFFFF_FFFF) as u32;
        carry = cur >> 32;
    }
    while carry != 0 {
        mag.push((carry & 0xFFFF_FFFF) as u32);
        carry >>= 32;
    }
    normalize(mag);
}

fn add_small_inplace(mag: &mut Vec<u32>, k: u32) {
    let mut carry = k as u64;
    let mut i = 0;
    while carry != 0 {
        if i == mag.len() {
            mag.push(0);
        }
        let cur = mag[i] as u64 + carry;
        mag[i] = (cur & 0xFFFF_FFFF) as u32;
        carry = cur >> 32;
        i += 1;
    }
}

/// Divide `mag` in place by the small divisor `d`, returning the remainder.
fn divmod_small_inplace(mag: &mut Vec<u32>, d: u32) -> u32 {
    let mut rem = 0u64;
    for i in (0..mag.len()).rev() {
        let cur = (rem << 32) | mag[i] as u64;
        mag[i] = (cur / d as u64) as u32;
        rem = cur % d as u64;
    }
    normalize(mag);
    rem as u32
}

fn bit_len(mag: &[u32]) -> usize {
    match mag.last() {
        None => 0,
        Some(&top) => (mag.len() - 1) * 32 + (32 - top.leading_zeros() as usize),
    }
}

fn test_bit(mag: &[u32], i: usize) -> bool {
    let (limb, bit) = (i / 32, i % 32);
    limb < mag.len() && (mag[limb] >> bit) & 1 == 1
}

fn set_bit(mag: &mut Vec<u32>, i: usize) {
    let (limb, bit) = (i / 32, i % 32);
    if limb >= mag.len() {
        mag.resize(limb + 1, 0);
    }
    mag[limb] |= 1 << bit;
}

/// Shift a magnitude left by one bit (multiply by 2).
fn shl1(mag: &mut Vec<u32>) {
    let mut carry = 0u32;
    for limb in mag.iter_mut() {
        let new_carry = *limb >> 31;
        *limb = (*limb << 1) | carry;
        carry = new_carry;
    }
    if carry != 0 {
        mag.push(carry);
    }
}

/// Binary long division of magnitudes → `(quotient, remainder)`.
fn divmod_mag(num: &[u32], den: &[u32]) -> (Vec<u32>, Vec<u32>) {
    debug_assert!(!den.is_empty(), "division by zero magnitude");
    if cmp_mag(num, den) == Ordering::Less {
        return (Vec::new(), num.to_vec());
    }
    let bits = bit_len(num);
    let mut q: Vec<u32> = Vec::new();
    let mut rem: Vec<u32> = Vec::new();
    for i in (0..bits).rev() {
        shl1(&mut rem);
        if test_bit(num, i) {
            if rem.is_empty() {
                rem.push(1);
            } else {
                rem[0] |= 1;
            }
        }
        normalize(&mut rem);
        if cmp_mag(&rem, den) != Ordering::Less {
            rem = sub_mag(&rem, den);
            set_bit(&mut q, i);
        }
    }
    normalize(&mut q);
    (q, rem)
}

// ════════════════════════════════ Decimal ════════════════════════════════

/// Default number of fractional digits produced by [`Decimal`] division and
/// square root when the caller does not specify a precision.
pub const DEFAULT_DIV_PREC: u64 = 50;

/// An **arbitrary-precision decimal**: a value `mant × 10^(-scale)` where `mant`
/// is a [`BigInt`] and `scale >= 0` is the number of fractional digits.
///
/// Because the digits are stored in base 10, every decimal literal is represented
/// *exactly* — there is no `0.1 + 0.2 == 0.30000000000000004` error. Addition,
/// subtraction and multiplication are exact; division and `sqrt` round to a
/// requested precision (default [`DEFAULT_DIV_PREC`] digits). This is the type to
/// reach for when accuracy matters more than speed: money, scientific constants,
/// long error-sensitive reductions.
#[derive(Clone, Debug)]
pub struct Decimal {
    mant: BigInt,
    scale: u32, // value = mant * 10^-scale
}

impl PartialEq for Decimal {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Decimal {
    /// Zero.
    pub fn zero() -> Decimal {
        Decimal { mant: BigInt::zero(), scale: 0 }
    }

    /// An exact integer as a decimal.
    pub fn from_i64(v: i64) -> Decimal {
        Decimal { mant: BigInt::from_i64(v), scale: 0 }
    }

    /// Build directly from a mantissa and scale.
    pub fn from_parts(mant: BigInt, scale: u32) -> Decimal {
        Decimal { mant, scale }
    }

    pub fn scale(&self) -> u32 {
        self.scale
    }

    pub fn is_zero(&self) -> bool {
        self.mant.is_zero()
    }

    pub fn sign(&self) -> i32 {
        self.mant.sign()
    }

    pub fn neg(&self) -> Decimal {
        Decimal { mant: self.mant.neg(), scale: self.scale }
    }

    pub fn abs(&self) -> Decimal {
        Decimal { mant: self.mant.abs(), scale: self.scale }
    }

    /// Parse a decimal: `[-]digits[.digits][(e|E)[±]digits]`. Returns `None` on
    /// malformed input.
    //
    // Deliberately returns `Option` (parse failure carries no error detail) rather
    // than implementing `FromStr`, whose `Result<Self, Err>` contract would force a
    // throwaway error type on every caller.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Decimal> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        // split off exponent
        let (body, exp) = match s.find(['e', 'E']) {
            Some(i) => {
                let e: i64 = s[i + 1..].parse().ok()?;
                (&s[..i], e)
            }
            None => (s, 0),
        };
        let (neg, body) = match body.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, body.strip_prefix('+').unwrap_or(body)),
        };
        let (int_part, frac_part) = match body.find('.') {
            Some(i) => (&body[..i], &body[i + 1..]),
            None => (body, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        if !int_part.bytes().all(|b| b.is_ascii_digit())
            || !frac_part.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let mut digits = String::new();
        digits.push_str(int_part);
        digits.push_str(frac_part);
        if digits.is_empty() {
            digits.push('0');
        }
        let mut mant = BigInt::from_decimal_str(&digits)?;
        if neg {
            mant = mant.neg();
        }
        // scale starts as fractional-digit count, adjusted by the exponent.
        // Reject pathological exponents rather than truncating the i64 scale into
        // a u32 (which would silently corrupt the magnitude) or growing the
        // mantissa without bound in the negative-scale branch.
        const MAX_POW10_SHIFT: i64 = 1_000_000;
        let mut scale = frac_part.len() as i64 - exp;
        if scale < 0 {
            if -scale > MAX_POW10_SHIFT {
                return None;
            }
            mant = mant.mul_pow10((-scale) as u64);
            scale = 0;
        }
        if scale > u32::MAX as i64 {
            return None;
        }
        Some(Decimal { mant, scale: scale as u32 })
    }

    /// Approximate a decimal from an `f64` (via its shortest round-trip string).
    pub fn from_f64(v: f64) -> Decimal {
        Decimal::from_str(&format!("{:?}", v)).unwrap_or_else(Decimal::zero)
    }

    /// Lossy conversion back to `f64` (for routing/printing interop).
    pub fn to_f64(&self) -> f64 {
        self.to_string().parse().unwrap_or(0.0)
    }

    /// Rescale to exactly `new_scale` fractional digits (must not shrink scale —
    /// used internally to align operands).
    fn with_scale(&self, new_scale: u32) -> BigInt {
        if new_scale >= self.scale {
            self.mant.mul_pow10((new_scale - self.scale) as u64)
        } else {
            // round toward nearest when reducing (half away from zero)
            let drop = (self.scale - new_scale) as u64;
            let p = BigInt::one().mul_pow10(drop);
            let (q, r) = self.mant.divmod(&p).unwrap();
            let twice = r.abs().mul(&BigInt::from_i64(2));
            if twice.cmp(&p) != Ordering::Less {
                let one = if self.mant.sign() < 0 { BigInt::from_i64(-1) } else { BigInt::one() };
                q.add(&one)
            } else {
                q
            }
        }
    }

    pub fn add(&self, other: &Decimal) -> Decimal {
        let s = self.scale.max(other.scale);
        Decimal { mant: self.with_scale(s).add(&other.with_scale(s)), scale: s }
    }

    pub fn sub(&self, other: &Decimal) -> Decimal {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Decimal) -> Decimal {
        Decimal { mant: self.mant.mul(&other.mant), scale: self.scale + other.scale }
    }

    /// Divide to `prec` fractional digits (round half away from zero). `None` on
    /// division by zero.
    pub fn div(&self, other: &Decimal, prec: u64) -> Option<Decimal> {
        if other.is_zero() {
            return None;
        }
        // result = self/other ≈ R × 10^-prec
        // R = self.mant·10^(prec + other.scale - self.scale) / other.mant
        let shift = prec as i64 + other.scale as i64 - self.scale as i64;
        let (num, den) = if shift >= 0 {
            (self.mant.mul_pow10(shift as u64), other.mant.clone())
        } else {
            (self.mant.clone(), other.mant.mul_pow10((-shift) as u64))
        };
        let (q, r) = num.divmod(&den)?;
        // round half away from zero
        let twice = r.abs().mul(&BigInt::from_i64(2));
        let mant = if twice.cmp(&den.abs()) != Ordering::Less {
            let one = if num.sign() * den.sign() < 0 { BigInt::from_i64(-1) } else { BigInt::one() };
            q.add(&one)
        } else {
            q
        };
        Some(Decimal { mant, scale: prec as u32 }.trimmed())
    }

    /// Square root to `prec` fractional digits via Newton–Raphson in exact decimal
    /// arithmetic. `None` for negative inputs.
    pub fn sqrt(&self, prec: u64) -> Option<Decimal> {
        if self.sign() < 0 {
            return None;
        }
        if self.is_zero() {
            return Some(Decimal::zero());
        }
        let work = prec + 10; // guard digits
        let two = Decimal::from_i64(2);
        // initial guess from f64
        let mut x = Decimal::from_f64(fsqrt(self.to_f64()));
        if x.is_zero() {
            x = Decimal::from_i64(1);
        }
        // a few extra iterations past convergence is cheap and safe.
        for _ in 0..100 {
            // x' = (x + a/x) / 2
            let ax = self.div(&x, work)?;
            let next = x.add(&ax).div(&two, work)?;
            if next.cmp(&x) == Ordering::Equal {
                x = next;
                break;
            }
            x = next;
        }
        x.div(&Decimal::from_i64(1), prec)
    }

    /// Round to `digits` fractional digits (half away from zero).
    pub fn round(&self, digits: u32) -> Decimal {
        Decimal { mant: self.with_scale(digits), scale: digits }.trimmed()
    }

    /// Drop trailing fractional zeros (canonical form, so `1.50 == 1.5`).
    pub fn trimmed(&self) -> Decimal {
        if self.is_zero() {
            return Decimal::zero();
        }
        let mut mant = self.mant.clone();
        let mut scale = self.scale;
        let ten = BigInt::from_i64(10);
        while scale > 0 {
            let (q, r) = mant.divmod(&ten).unwrap();
            if !r.is_zero() {
                break;
            }
            mant = q;
            scale -= 1;
        }
        Decimal { mant, scale }
    }

}

impl Eq for Decimal {}

impl Ord for Decimal {
    fn cmp(&self, other: &Decimal) -> Ordering {
        let s = self.scale.max(other.scale);
        self.with_scale(s).cmp(&other.with_scale(s))
    }
}

impl PartialOrd for Decimal {
    fn partial_cmp(&self, other: &Decimal) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.trimmed();
        if d.scale == 0 {
            return f.write_str(&d.mant.to_decimal_string());
        }
        let neg = d.mant.sign() < 0;
        let digits = d.mant.abs().to_decimal_string();
        let scale = d.scale as usize;
        if neg {
            f.write_str("-")?;
        }
        if digits.len() <= scale {
            // 0.00…digits
            f.write_str("0.")?;
            for _ in 0..(scale - digits.len()) {
                f.write_str("0")?;
            }
            f.write_str(&digits)
        } else {
            let point = digits.len() - scale;
            f.write_str(&digits[..point])?;
            f.write_str(".")?;
            f.write_str(&digits[point..])
        }
    }
}

// ════════════════════════════════ Rational ════════════════════════════════

/// An **exact fraction** `num/den` of two [`BigInt`]s, kept reduced with a
/// positive denominator. Unlike [`Decimal`], even division is exact: `1/3` is
/// stored as `1/3`, never rounded.
#[derive(Clone, Debug)]
pub struct Rational {
    num: BigInt,
    den: BigInt, // always > 0, gcd(num,den)=1
}

impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Rational {
    /// `num/den`, reduced. `None` if `den == 0`.
    pub fn new(num: BigInt, den: BigInt) -> Option<Rational> {
        if den.is_zero() {
            return None;
        }
        let neg = num.sign() * den.sign() < 0;
        let mut n = num.abs();
        let mut d = den.abs();
        let g = n.gcd(&d);
        if !g.is_zero() && g != BigInt::one() {
            n = n.divmod(&g).unwrap().0;
            d = d.divmod(&g).unwrap().0;
        }
        if neg {
            n = n.neg();
        }
        Some(Rational { num: n, den: d })
    }

    pub fn from_i64(v: i64) -> Rational {
        Rational { num: BigInt::from_i64(v), den: BigInt::one() }
    }

    pub fn numerator(&self) -> &BigInt {
        &self.num
    }
    pub fn denominator(&self) -> &BigInt {
        &self.den
    }
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
    pub fn sign(&self) -> i32 {
        self.num.sign()
    }

    pub fn neg(&self) -> Rational {
        Rational { num: self.num.neg(), den: self.den.clone() }
    }
    pub fn abs(&self) -> Rational {
        Rational { num: self.num.abs(), den: self.den.clone() }
    }

    /// Multiplicative inverse. `None` if `self == 0`.
    pub fn recip(&self) -> Option<Rational> {
        Rational::new(self.den.clone(), self.num.clone())
    }

    pub fn add(&self, other: &Rational) -> Rational {
        // a/b + c/d = (a d + c b)/(b d)
        let num = self.num.mul(&other.den).add(&other.num.mul(&self.den));
        let den = self.den.mul(&other.den);
        Rational::new(num, den).unwrap()
    }
    pub fn sub(&self, other: &Rational) -> Rational {
        self.add(&other.neg())
    }
    pub fn mul(&self, other: &Rational) -> Rational {
        Rational::new(self.num.mul(&other.num), self.den.mul(&other.den)).unwrap()
    }
    /// `None` if dividing by zero.
    pub fn div(&self, other: &Rational) -> Option<Rational> {
        Rational::new(self.num.mul(&other.den), self.den.mul(&other.num))
    }

    /// Render as an exact decimal to `prec` digits (the fraction itself stays exact).
    pub fn to_decimal(&self, prec: u64) -> Decimal {
        Decimal::from_parts(self.num.clone(), 0)
            .div(&Decimal::from_parts(self.den.clone(), 0), prec)
            .unwrap_or_else(Decimal::zero)
    }

}

impl Eq for Rational {}

impl Ord for Rational {
    fn cmp(&self, other: &Rational) -> Ordering {
        // a/b ? c/d  ⇔  a d ? c b  (denominators positive)
        self.num.mul(&other.den).cmp(&other.num.mul(&self.den))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Rational) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == BigInt::one() {
            f.write_str(&self.num.to_decimal_string())
        } else {
            write!(f, "{}/{}", self.num.to_decimal_string(), self.den.to_decimal_string())
        }
    }
}

// ════════════════════════════════ Complex ════════════════════════════════

/// A complex number `re + im·i`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }
    pub fn add(&self, o: &Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
    pub fn sub(&self, o: &Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
    pub fn mul(&self, o: &Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    pub fn conj(&self) -> Complex {
        Complex::new(self.re, -self.im)
    }
    /// Squared modulus `re² + im²` (exact, no sqrt).
    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    /// Modulus `|z|`.
    pub fn modulus(&self) -> f64 {
        fsqrt(self.norm_sqr())
    }
    /// Complex division. `None` if the divisor is exactly zero.
    pub fn div(&self, o: &Complex) -> Option<Complex> {
        let d = o.norm_sqr();
        if d == 0.0 {
            return None;
        }
        Some(Complex::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        ))
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im < 0.0 {
            write!(f, "{}-{}i", self.re, -self.im)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

// ════════════════════════════════ Dual ════════════════════════════════

/// A **dual number** `val + der·ε` (where `ε² = 0`) for exact forward-mode
/// automatic differentiation: carry a value and its derivative through a
/// computation and read off the gradient with zero finite-difference error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub val: f64,
    pub der: f64,
}

impl Dual {
    pub fn new(val: f64, der: f64) -> Dual {
        Dual { val, der }
    }
    /// A constant (derivative 0).
    pub fn constant(v: f64) -> Dual {
        Dual { val: v, der: 0.0 }
    }
    /// The differentiation variable `x = v` (derivative 1).
    pub fn variable(v: f64) -> Dual {
        Dual { val: v, der: 1.0 }
    }
    pub fn add(&self, o: &Dual) -> Dual {
        Dual::new(self.val + o.val, self.der + o.der)
    }
    pub fn sub(&self, o: &Dual) -> Dual {
        Dual::new(self.val - o.val, self.der - o.der)
    }
    pub fn mul(&self, o: &Dual) -> Dual {
        // (a+a'ε)(b+b'ε) = ab + (a'b + ab')ε
        Dual::new(self.val * o.val, self.der * o.val + self.val * o.der)
    }
    /// `None` if dividing by a zero value.
    pub fn div(&self, o: &Dual) -> Option<Dual> {
        if o.val == 0.0 {
            return None;
        }
        Some(Dual::new(
            self.val / o.val,
            (self.der * o.val - self.val * o.der) / (o.val * o.val),
        ))
    }
    /// `sqrt` with the chain rule applied to the derivative.
    pub fn sqrt(&self) -> Option<Dual> {
        if self.val < 0.0 {
            return None;
        }
        let s = fsqrt(self.val);
        let d = if s == 0.0 { 0.0 } else { self.der / (2.0 * s) };
        Some(Dual::new(s, d))
    }
    /// `self` raised to an integer power. The derivative falls out of repeated
    /// multiplication (and reciprocal for negative exponents), so the chain rule
    /// is applied exactly with no special-case formula.
    pub fn powi(&self, n: i64) -> Dual {
        if n == 0 {
            return Dual::constant(1.0);
        }
        let mut acc = Dual::constant(1.0);
        for _ in 0..n.unsigned_abs() {
            acc = acc.mul(self);
        }
        if n < 0 {
            Dual::constant(1.0).div(&acc).unwrap_or_else(|| Dual::constant(0.0))
        } else {
            acc
        }
    }
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}ε", self.val, self.der)
    }
}

// ════════════════════════════════ Interval ════════════════════════════════

/// A rigorous **interval** `[lo, hi]`: a value guaranteed to lie within these
/// bounds. Interval arithmetic propagates worst-case error automatically, so a
/// long reduction yields a proven bracket around the true answer rather than an
/// unquantified `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// `[lo, hi]`, swapping if given out of order.
    pub fn new(lo: f64, hi: f64) -> Interval {
        if lo <= hi {
            Interval { lo, hi }
        } else {
            Interval { lo: hi, hi: lo }
        }
    }
    /// A degenerate (exact) interval `[v, v]`.
    pub fn point(v: f64) -> Interval {
        Interval { lo: v, hi: v }
    }
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
    pub fn mid(&self) -> f64 {
        0.5 * (self.lo + self.hi)
    }
    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }
    pub fn add(&self, o: &Interval) -> Interval {
        Interval::new(self.lo + o.lo, self.hi + o.hi)
    }
    pub fn sub(&self, o: &Interval) -> Interval {
        Interval::new(self.lo - o.hi, self.hi - o.lo)
    }
    pub fn mul(&self, o: &Interval) -> Interval {
        let ps = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi];
        let mut lo = ps[0];
        let mut hi = ps[0];
        for &p in &ps[1..] {
            if p < lo {
                lo = p;
            }
            if p > hi {
                hi = p;
            }
        }
        Interval { lo, hi }
    }
    /// `None` if the divisor interval straddles zero.
    pub fn div(&self, o: &Interval) -> Option<Interval> {
        if o.lo <= 0.0 && o.hi >= 0.0 {
            return None;
        }
        Some(self.mul(&Interval::new(1.0 / o.hi, 1.0 / o.lo)))
    }
    /// The smallest interval containing both (the convex hull).
    pub fn hull(&self, o: &Interval) -> Interval {
        Interval { lo: self.lo.min(o.lo), hi: self.hi.max(o.hi) }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.lo, self.hi)
    }
}

// ════════════════════════════════ Quaternion ════════════════════════════════

/// A **quaternion** `w + xi + yj + zk` — the standard algebra for composing 3-D
/// rotations without gimbal lock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Quaternion {
        Quaternion { w, x, y, z }
    }
    pub fn identity() -> Quaternion {
        Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }
    pub fn add(&self, o: &Quaternion) -> Quaternion {
        Quaternion::new(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(&self, o: &Quaternion) -> Quaternion {
        Quaternion::new(self.w - o.w, self.x - o.x, self.y - o.y, self.z - o.z)
    }
    /// Hamilton product (non-commutative).
    pub fn mul(&self, o: &Quaternion) -> Quaternion {
        Quaternion::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }
    pub fn conj(&self) -> Quaternion {
        Quaternion::new(self.w, -self.x, -self.y, -self.z)
    }
    pub fn norm(&self) -> f64 {
        fsqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
    }
    /// Unit quaternion (`None` if this one is zero).
    pub fn normalized(&self) -> Option<Quaternion> {
        let n = self.norm();
        if n == 0.0 {
            return None;
        }
        Some(Quaternion::new(self.w / n, self.x / n, self.y / n, self.z / n))
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}i+{}j+{}k", self.w, self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fabs(x: f64) -> f64 {
        if x < 0.0 {
            -x
        } else {
            x
        }
    }

    // ---- BigInt ----

    #[test]
    fn bigint_roundtrips_decimal_strings() {
        for s in ["0", "7", "-7", "123456789012345678901234567890", "-1000000000000000000001"] {
            let b = BigInt::from_decimal_str(s).unwrap();
            assert_eq!(b.to_decimal_string(), s);
        }
    }

    #[test]
    fn bigint_add_sub_mul() {
        let a = BigInt::from_decimal_str("99999999999999999999").unwrap();
        let b = BigInt::from_decimal_str("1").unwrap();
        assert_eq!(a.add(&b).to_decimal_string(), "100000000000000000000");
        let big = a.mul(&a);
        // 99999999999999999999^2
        assert_eq!(big.to_decimal_string(), "9999999999999999999800000000000000000001");
        assert_eq!(a.sub(&a).to_decimal_string(), "0");
        assert!(a.sub(&a).is_zero());
    }

    #[test]
    fn bigint_divmod_and_sign() {
        let a = BigInt::from_decimal_str("1000000000000000000007").unwrap();
        let b = BigInt::from_decimal_str("1000000007").unwrap();
        let (q, r) = a.divmod(&b).unwrap();
        // q*b + r == a
        assert_eq!(q.mul(&b).add(&r), a);
        // negative dividend: remainder takes dividend's sign
        let (q2, r2) = BigInt::from_i64(-17).divmod(&BigInt::from_i64(5)).unwrap();
        assert_eq!(q2.to_decimal_string(), "-3");
        assert_eq!(r2.to_decimal_string(), "-2");
        assert!(BigInt::from_i64(1).divmod(&BigInt::zero()).is_none());
    }

    #[test]
    fn bigint_gcd() {
        let a = BigInt::from_i64(48);
        let b = BigInt::from_i64(36);
        assert_eq!(a.gcd(&b).to_decimal_string(), "12");
    }

    // ---- Decimal: the headline accuracy claims ----

    #[test]
    fn decimal_addition_is_exact_where_f64_is_not() {
        // The canonical floating-point failure: 0.1 + 0.2 != 0.3 in f64. Bind the
        // operands so clippy doesn't fold the comparison to a constant assertion.
        let (a01, a02, a03) = (0.1_f64, 0.2_f64, 0.3_f64);
        assert_ne!(a01 + a02, a03, "f64 cannot represent 0.1 + 0.2 exactly");
        // Decimal gets it exactly right.
        let a = Decimal::from_str("0.1").unwrap();
        let b = Decimal::from_str("0.2").unwrap();
        assert_eq!(a.add(&b).to_string(), "0.3");
        assert_eq!(a.add(&b).cmp(&Decimal::from_str("0.3").unwrap()), Ordering::Equal);
    }

    #[test]
    fn decimal_parse_and_print_roundtrip() {
        for s in ["0", "1", "-1", "3.14159", "-0.0007", "1000", "0.5"] {
            assert_eq!(Decimal::from_str(s).unwrap().to_string(), s);
        }
        // exponent form
        assert_eq!(Decimal::from_str("1.5e3").unwrap().to_string(), "1500");
        assert_eq!(Decimal::from_str("25e-2").unwrap().to_string(), "0.25");
    }

    #[test]
    fn decimal_mul_is_exact() {
        let a = Decimal::from_str("1.1").unwrap();
        let b = Decimal::from_str("1.1").unwrap();
        assert_eq!(a.mul(&b).to_string(), "1.21");
        let c = Decimal::from_str("0.0000001").unwrap();
        assert_eq!(c.mul(&c).to_string(), "0.00000000000001");
    }

    #[test]
    fn decimal_div_high_precision() {
        let one = Decimal::from_i64(1);
        let three = Decimal::from_i64(3);
        let third = one.div(&three, 50).unwrap();
        // 50 threes after the point.
        assert_eq!(third.to_string(), "0.33333333333333333333333333333333333333333333333333");
        assert!(one.div(&Decimal::zero(), 10).is_none());
    }

    #[test]
    fn decimal_sqrt_of_two_is_accurate_to_many_digits() {
        let two = Decimal::from_i64(2);
        let r = two.sqrt(40).unwrap();
        // r*r must be ~2 to within 10^-40.
        let sq = r.mul(&r);
        let err = sq.sub(&two).abs();
        let tol = Decimal::from_str("0.0000000000000000000000000000000000001").unwrap(); // 1e-37
        assert_eq!(err.cmp(&tol), Ordering::Less, "sqrt2^2 error too big: {}", err);
        // First digits of sqrt(2).
        assert!(r.to_string().starts_with("1.4142135623730950488"));
    }

    #[test]
    fn decimal_round_and_trim() {
        let d = Decimal::from_str("2.567").unwrap();
        assert_eq!(d.round(2).to_string(), "2.57");
        assert_eq!(d.round(0).to_string(), "3");
        assert_eq!(Decimal::from_str("1.5000").unwrap().to_string(), "1.5");
        assert_eq!(Decimal::from_str("-0.250").unwrap().round(1).to_string(), "-0.3");
    }

    // ---- Rational ----

    #[test]
    fn rational_is_exact_and_reduced() {
        let a = Rational::new(BigInt::from_i64(1), BigInt::from_i64(3)).unwrap();
        let b = Rational::new(BigInt::from_i64(1), BigInt::from_i64(6)).unwrap();
        // 1/3 + 1/6 = 1/2 exactly.
        assert_eq!(a.add(&b).to_string(), "1/2");
        // 2/4 reduces to 1/2.
        assert_eq!(Rational::new(BigInt::from_i64(2), BigInt::from_i64(4)).unwrap().to_string(), "1/2");
        // 1/3 * 3 = 1.
        let three = Rational::from_i64(3);
        assert_eq!(a.mul(&three).to_string(), "1");
        assert!(Rational::new(BigInt::from_i64(1), BigInt::zero()).is_none());
    }

    #[test]
    fn rational_to_decimal() {
        let a = Rational::new(BigInt::from_i64(2), BigInt::from_i64(7)).unwrap();
        assert_eq!(a.to_decimal(12).to_string(), "0.285714285714");
    }

    // ---- Complex ----

    #[test]
    fn complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        assert_eq!(a.add(&b), Complex::new(4.0, 1.0));
        // (1+2i)(3-i) = 3 - i + 6i - 2i² = 5 + 5i
        assert_eq!(a.mul(&b), Complex::new(5.0, 5.0));
        assert!(fabs(Complex::new(3.0, 4.0).modulus() - 5.0) < 1e-9);
        let q = a.div(&a).unwrap();
        assert!(fabs(q.re - 1.0) < 1e-9 && fabs(q.im) < 1e-9);
    }

    // ---- Dual (autodiff) ----

    #[test]
    fn dual_computes_exact_derivative() {
        // f(x) = x³ at x=2 → f=8, f'=3x²=12.
        let x = Dual::variable(2.0);
        let f = x.mul(&x).mul(&x);
        assert!(fabs(f.val - 8.0) < 1e-9);
        assert!(fabs(f.der - 12.0) < 1e-9);
        // g(x) = sqrt(x) at x=4 → g=2, g'=1/(2·2)=0.25.
        let g = Dual::variable(4.0).sqrt().unwrap();
        assert!(fabs(g.val - 2.0) < 1e-9 && fabs(g.der - 0.25) < 1e-9);
        // powi: x^-1 at 2 → 0.5, derivative -1/4.
        let r = Dual::variable(2.0).powi(-1);
        assert!(fabs(r.val - 0.5) < 1e-9 && fabs(r.der + 0.25) < 1e-9);
    }

    // ---- Interval ----

    #[test]
    fn interval_brackets_the_truth() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        assert_eq!(a.add(&b), Interval::new(4.0, 6.0));
        assert_eq!(a.sub(&b), Interval::new(-3.0, -1.0));
        assert_eq!(a.mul(&b), Interval::new(3.0, 8.0));
        assert!(a.contains(1.5));
        assert!(fabs(a.mid() - 1.5) < 1e-9 && fabs(a.width() - 1.0) < 1e-9);
        // division by a zero-straddling interval is rejected.
        assert!(a.div(&Interval::new(-1.0, 1.0)).is_none());
        assert_eq!(a.div(&b).unwrap(), Interval::new(0.25, 2.0 / 3.0));
    }

    // ---- Quaternion ----

    #[test]
    fn quaternion_hamilton_product() {
        // i * j = k  →  (0,1,0,0)*(0,0,1,0) = (0,0,0,1)
        let i = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        let j = Quaternion::new(0.0, 0.0, 1.0, 0.0);
        assert_eq!(i.mul(&j), Quaternion::new(0.0, 0.0, 0.0, 1.0));
        // non-commutative: j*i = -k
        assert_eq!(j.mul(&i), Quaternion::new(0.0, 0.0, 0.0, -1.0));
        assert!(fabs(Quaternion::new(1.0, 1.0, 1.0, 1.0).norm() - 2.0) < 1e-9);
        let u = Quaternion::new(0.0, 3.0, 0.0, 0.0).normalized().unwrap();
        assert!(fabs(u.norm() - 1.0) < 1e-9);
    }
}

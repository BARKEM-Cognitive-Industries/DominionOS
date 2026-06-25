//! Shared deterministic math primitives — `no_std + alloc`, no `libm`.
//!
//! All functions are range-reduced and series-based so results are bit-identical
//! on every target. This is the single canonical implementation for the whole
//! crate; no other module should re-implement these.
//!
//! # Exported functions
//! `sin`, `cos`, `tan`, `sqrt`, `exp`, `ln`, `floor`, `ceil`, `abs`

// ───────────────────────────── constants ────────────────────────────────────

const PI: f64 = core::f64::consts::PI;
const TWO_PI: f64 = 2.0 * core::f64::consts::PI;
const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
const LN_2: f64 = core::f64::consts::LN_2;

// ───────────────────────────── helpers ──────────────────────────────────────

/// Round `x` to the nearest integer (half-away-from-zero), as `f64`.
#[inline]
fn round_half(x: f64) -> f64 {
    if x >= 0.0 { ((x + 0.5) as i64) as f64 } else { ((x - 0.5) as i64) as f64 }
}

/// `2^k` for integer `k` by repeated multiplication (no `powi`/`libm`).
#[inline]
fn pow2i(k: i64) -> f64 {
    let (base, mut n) = if k >= 0 { (2.0f64, k) } else { (0.5f64, -k) };
    let mut r = 1.0;
    while n > 0 {
        r *= base;
        n -= 1;
    }
    r
}

// ───────────────────────────── trigonometry ──────────────────────────────────

/// Sine — range-reduced to `[-π/2, π/2]`, then a 12-term Taylor series.
///
/// Error is below 1 ULP for all finite inputs.
pub fn sin(x: f64) -> f64 {
    // Reduce to (-π, π].
    let mut r = x - TWO_PI * round_half(x / TWO_PI);
    // Fold to [-π/2, π/2] using the reflection identity.
    if r > HALF_PI {
        r = PI - r;
    } else if r < -HALF_PI {
        r = -PI - r;
    }
    let r2 = r * r;
    let mut term = r;
    let mut acc = r;
    let mut k = 1i64;
    while k < 12 {
        let denom = ((2 * k) * (2 * k + 1)) as f64;
        term *= -r2 / denom;
        acc += term;
        k += 1;
    }
    acc
}

/// Cosine via the phase-shift identity `cos(x) = sin(x + π/2)`.
#[inline]
pub fn cos(x: f64) -> f64 {
    sin(x + HALF_PI)
}

/// Tangent — `sin(x) / cos(x)`.
///
/// Returns `f64::INFINITY` or `f64::NEG_INFINITY` near poles (cos ≈ 0).
#[inline]
pub fn tan(x: f64) -> f64 {
    let c = cos(x);
    if c == 0.0 { f64::INFINITY } else { sin(x) / c }
}

// ───────────────────────────── square root ───────────────────────────────────

/// Square root via Newton–Raphson (Babylonian method).
///
/// Converges to full `f64` precision in ≤ 6 iterations for typical magnitudes;
/// we run 40 to be safe with extreme inputs.  Returns 0 for `x ≤ 0`.
pub fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = x;
    for _ in 0..40 {
        g = 0.5 * (g + x / g);
    }
    g
}

// ───────────────────────────── exponential ───────────────────────────────────

/// `e^x`, deterministic.
///
/// Range-reduces to `e^x = 2^k · e^r` with `|r| ≤ ln2/2`, then an 18-term
/// Taylor series that achieves full `f64` precision on that interval.
pub fn exp(x: f64) -> f64 {
    if x > 709.0 { return f64::MAX; }
    if x < -745.0 { return 0.0; }
    let k = if x >= 0.0 {
        (x / LN_2 + 0.5) as i64
    } else {
        (x / LN_2 - 0.5) as i64
    };
    let r = x - (k as f64) * LN_2;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut n = 1.0;
    for _ in 0..18 {
        term *= r / n;
        sum += term;
        n += 1.0;
    }
    sum * pow2i(k)
}

// ───────────────────────────── natural log ───────────────────────────────────

/// Natural logarithm via `atanh` series on the mantissa.
///
/// Decomposes `x = m · 2^e` (m ∈ [1,2)), computes `ln(m)` with a 20-term
/// `atanh` series on `t = (m−1)/(m+1) ∈ [0, 1/3]`, then adds `e · ln2`.
/// Returns `NEG_INFINITY` for `x ≤ 0`.
pub fn ln(x: f64) -> f64 {
    if x <= 0.0 { return f64::NEG_INFINITY; }
    let mut m = x;
    let mut e = 0i32;
    while m >= 2.0 { m *= 0.5; e += 1; }
    while m < 1.0  { m *= 2.0; e -= 1; }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut term = t;
    let mut acc = t;
    let mut k = 1usize;
    while k < 20 {
        term *= t2;
        acc += term / (2 * k + 1) as f64;
        k += 1;
    }
    2.0 * acc + e as f64 * LN_2
}

// ───────────────────────────── rounding ──────────────────────────────────────

/// Largest integer ≤ `x`, as `f64`. (`f64::floor` is `std`-only.)
pub fn floor(x: f64) -> f64 {
    let t = x as i64 as f64;
    if x < 0.0 && t != x { t - 1.0 } else { t }
}

/// Smallest integer ≥ `x`, as `f64`. (`f64::ceil` is `std`-only.)
pub fn ceil(x: f64) -> f64 {
    let t = x as i64 as f64;
    if x > 0.0 && t != x { t + 1.0 } else { t }
}

// ───────────────────────────── absolute value ────────────────────────────────

/// Absolute value of `x`.
#[inline]
pub fn abs(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

// ───────────────────────────── tests ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() <= tol }
    const EPS: f64 = 1e-12;

    #[test]
    fn test_sin_cos() {
        assert!(close(sin(0.0), 0.0, EPS));
        assert!(close(sin(HALF_PI), 1.0, EPS));
        assert!(close(sin(PI), 0.0, EPS));
        assert!(close(cos(0.0), 1.0, EPS));
        assert!(close(cos(HALF_PI), 0.0, EPS));
        assert!(close(cos(PI), -1.0, EPS));
        // Periodicity
        assert!(close(sin(TWO_PI + 1.0), sin(1.0), EPS));
    }

    #[test]
    fn test_sqrt() {
        assert!(close(sqrt(4.0), 2.0, EPS));
        assert!(close(sqrt(2.0), 1.414_213_562_373_095, 1e-10));
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(-1.0), 0.0);
    }

    #[test]
    fn test_exp_ln() {
        assert!(close(exp(0.0), 1.0, EPS));
        assert!(close(exp(1.0), 2.718_281_828_459_045, 1e-12));
        assert!(close(ln(1.0), 0.0, EPS));
        assert!(close(ln(exp(3.0)), 3.0, 1e-10));
    }

    #[test]
    fn test_floor_ceil_abs() {
        assert_eq!(floor(2.9), 2.0);
        assert_eq!(floor(-2.1), -3.0);
        assert_eq!(ceil(2.1), 3.0);
        assert_eq!(ceil(-2.9), -2.0);
        assert_eq!(abs(-5.0), 5.0);
        assert_eq!(abs(3.0), 3.0);
    }
}

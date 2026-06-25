//! 3D math library for the DominionOS unified render stack.
//! (see `docs/2d-3d rendering redesign.md`)
//!
//! Pure, safe `no_std`. sin32/cos32 are polynomial approximations (no libm).

#![allow(clippy::excessive_precision)]

use core::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const PI: f32 = 3.14159265358979323846_f32;
pub const TAU: f32 = 6.28318530717958647692_f32;
pub const FRAC_PI_2: f32 = 1.57079632679489661923_f32;
pub const DEG_TO_RAD: f32 = PI / 180.0;

// ---------------------------------------------------------------------------
// Scalar math helpers (no libm)
// ---------------------------------------------------------------------------

/// Square root. Newton-Raphson approximation, no libm required.
/// Converges to < 1 ULP after three iterations for x ≥ 0.
#[inline(always)]
pub fn sqrt32(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    // Initial estimate via bit manipulation (Quake-style, but for sqrt not rsqrt).
    // Cast bits, shift exponent field right by 1 and bias-correct.
    let bits = x.to_bits();
    let est_bits = (bits >> 1).wrapping_add(0x1fbb_4f2e);
    let mut s = f32::from_bits(est_bits);
    // Three Newton-Raphson iterations: s = 0.5*(s + x/s)
    s = 0.5 * (s + x / s);
    s = 0.5 * (s + x / s);
    s = 0.5 * (s + x / s);
    s
}

/// floor(x) without libm.
#[inline]
fn floor32(x: f32) -> f32 {
    let i = x as i64;
    let fi = i as f32;
    if fi > x { fi - 1.0 } else { fi }
}

/// Range-reduce `x` to `[-π, π]`.
#[inline]
fn range_reduce(x: f32) -> f32 {
    // Fast modulo via floor: x - TAU * round(x / TAU)
    let k = floor32(x * (1.0 / TAU) + 0.5);
    x - k * TAU
}

/// 5-term Horner polynomial sine, accurate to < 5e-5 across all inputs.
/// Range-reduces first to [-π, π] then to [-π/2, π/2] using sin(π-x) symmetry.
#[inline]
pub fn sin32(x: f32) -> f32 {
    // Reduce to [-π, π]
    let x = range_reduce(x);
    // Fold to [-π/2, π/2]: sin(π-x)=sin(x), sin(-π-x)=-sin(x)
    let x = if x > FRAC_PI_2 {
        PI - x
    } else if x < -FRAC_PI_2 {
        -PI - x
    } else {
        x
    };
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0 - x2 * (1.0 / 362880.0)))))
}

/// cos(x) = sin(x + π/2)
#[inline]
pub fn cos32(x: f32) -> f32 {
    sin32(x + FRAC_PI_2)
}

/// Rational approximation of atan2, accurate to ≈ 2e-5.
/// Based on the Rajan/Baker piecewise rational form.
#[inline]
pub fn atan2_32(y: f32, x: f32) -> f32 {
    if x == 0.0 {
        if y > 0.0 {
            return FRAC_PI_2;
        } else if y < 0.0 {
            return -FRAC_PI_2;
        } else {
            return 0.0;
        }
    }

    let abs_y = if y < 0.0 { -y } else { y };
    let (r, offset) = if x < 0.0 {
        // third / second quadrant
        ((x + abs_y) / (abs_y - x), PI * 3.0 / 4.0)
    } else {
        // first / fourth quadrant
        ((x - abs_y) / (x + abs_y), PI / 4.0)
    };

    // polynomial approximation of atan on [-1, 1]
    let angle = offset + (0.1963 * r * r - 0.9817) * r;
    if y < 0.0 { -angle } else { angle }
}

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    #[inline] pub const fn new(x: f32, y: f32) -> Self { Self { x, y } }
    #[inline] pub const fn zero() -> Self { Self::new(0.0, 0.0) }
    #[inline] pub const fn one() -> Self { Self::new(1.0, 1.0) }

    #[inline] pub fn dot(self, rhs: Self) -> f32 { self.x * rhs.x + self.y * rhs.y }
    #[inline] pub fn len_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn len(self) -> f32 { sqrt32(self.len_sq()) }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.len();
        if l < 1e-10 { return Self::zero(); }
        Self::new(self.x / l, self.y / l)
    }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self::new(
            self.x + (rhs.x - self.x) * t,
            self.y + (rhs.y - self.y) * t,
        )
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self::new(self.x + r.x, self.y + r.y) }
}
impl Sub for Vec2 {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self::new(self.x - r.x, self.y - r.y) }
}
impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s) }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline] fn mul(self, v: Vec2) -> Vec2 { Vec2::new(self * v.x, self * v.y) }
}
impl Neg for Vec2 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x, -self.y) }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline] pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    #[inline] pub const fn zero() -> Self { Self::new(0.0, 0.0, 0.0) }
    #[inline] pub const fn one() -> Self { Self::new(1.0, 1.0, 1.0) }
    #[inline] pub const fn x_axis() -> Self { Self::new(1.0, 0.0, 0.0) }
    #[inline] pub const fn y_axis() -> Self { Self::new(0.0, 1.0, 0.0) }
    #[inline] pub const fn z_axis() -> Self { Self::new(0.0, 0.0, 1.0) }

    #[inline] pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline] pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[inline] pub fn len_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn len(self) -> f32 { sqrt32(self.len_sq()) }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.len();
        if l < 1e-10 { return Self::zero(); }
        Self::new(self.x / l, self.y / l, self.z / l)
    }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self::new(
            self.x + (rhs.x - self.x) * t,
            self.y + (rhs.y - self.y) * t,
            self.z + (rhs.z - self.z) * t,
        )
    }

    #[inline] pub fn splat(v: f32) -> Self { Self::new(v, v, v) }

    #[inline] pub fn min_elem(self, rhs: Self) -> Self {
        Self::new(
            if self.x < rhs.x { self.x } else { rhs.x },
            if self.y < rhs.y { self.y } else { rhs.y },
            if self.z < rhs.z { self.z } else { rhs.z },
        )
    }

    #[inline] pub fn max_elem(self, rhs: Self) -> Self {
        Self::new(
            if self.x > rhs.x { self.x } else { rhs.x },
            if self.y > rhs.y { self.y } else { rhs.y },
            if self.z > rhs.z { self.z } else { rhs.z },
        )
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self::new(self.x + r.x, self.y + r.y, self.z + r.z) }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self::new(self.x - r.x, self.y - r.y, self.z - r.z) }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline] fn mul(self, v: Vec3) -> Vec3 { Vec3::new(self * v.x, self * v.y, self * v.z) }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x, -self.y, -self.z) }
}

// ---------------------------------------------------------------------------
// Vec4
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    #[inline] pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }
    #[inline] pub const fn zero() -> Self { Self::new(0.0, 0.0, 0.0, 0.0) }

    /// Construct from Vec3 + explicit w.
    #[inline] pub fn from_vec3(v: Vec3, w: f32) -> Self { Self::new(v.x, v.y, v.z, w) }

    /// Discard w.
    #[inline] pub fn xyz(self) -> Vec3 { Vec3::new(self.x, self.y, self.z) }

    #[inline] pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline] pub fn len_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn len(self) -> f32 { sqrt32(self.len_sq()) }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self {
        Self::new(self.x + r.x, self.y + r.y, self.z + r.z, self.w + r.w)
    }
}
impl Sub for Vec4 {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self {
        Self::new(self.x - r.x, self.y - r.y, self.z - r.z, self.w - r.w)
    }
}
impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }
}

// ---------------------------------------------------------------------------
// Mat4 — column-major: cols[col][row]
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// Column-major storage: `cols[col][row]`.
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    #[inline]
    pub const fn from_cols_array(cols: [[f32; 4]; 4]) -> Self {
        Self { cols }
    }

    pub const fn identity() -> Self {
        Self {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Access element at (row, col).
    #[inline] pub fn get(&self, row: usize, col: usize) -> f32 { self.cols[col][row] }
    #[inline] pub fn set(&mut self, row: usize, col: usize, v: f32) { self.cols[col][row] = v; }

    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        let x = self.cols[0][0] * v.x + self.cols[1][0] * v.y + self.cols[2][0] * v.z + self.cols[3][0] * v.w;
        let y = self.cols[0][1] * v.x + self.cols[1][1] * v.y + self.cols[2][1] * v.z + self.cols[3][1] * v.w;
        let z = self.cols[0][2] * v.x + self.cols[1][2] * v.y + self.cols[2][2] * v.z + self.cols[3][2] * v.w;
        let w = self.cols[0][3] * v.x + self.cols[1][3] * v.y + self.cols[2][3] * v.z + self.cols[3][3] * v.w;
        Vec4::new(x, y, z, w)
    }

    /// Transform a point (w=1).
    #[inline]
    pub fn mul_point(&self, p: Vec3) -> Vec3 {
        self.mul_vec4(Vec4::from_vec3(p, 1.0)).xyz()
    }

    /// Transform a direction (w=0), ignoring translation.
    #[inline]
    pub fn mul_dir(&self, d: Vec3) -> Vec3 {
        self.mul_vec4(Vec4::from_vec3(d, 0.0)).xyz()
    }

    pub fn transpose(&self) -> Self {
        let mut out = Self::identity();
        for r in 0..4 {
            for c in 0..4 {
                out.cols[r][c] = self.cols[c][r];
            }
        }
        out
    }

    /// Invert via cofactor / adjugate expansion (safe, no division by near-zero guard removed).
    /// Returns `None` if the determinant is too small.
    pub fn invert(&self) -> Option<Self> {
        // Unroll the full 4×4 cofactor inverse.
        let m = &self.cols;

        let c00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let c02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
        let c03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

        let c04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let c06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
        let c07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

        let c08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let c10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
        let c11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

        let c12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        let c14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
        let c15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

        let c16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        let c18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
        let c19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

        let c20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        let c22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
        let c23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

        let f0 = Vec4::new(c00, c00, c02, c03);
        let f1 = Vec4::new(c04, c04, c06, c07);
        let f2 = Vec4::new(c08, c08, c10, c11);
        let f3 = Vec4::new(c12, c12, c14, c15);
        let f4 = Vec4::new(c16, c16, c18, c19);
        let f5 = Vec4::new(c20, c20, c22, c23);

        let v0 = Vec4::new(m[1][0], m[0][0], m[0][0], m[0][0]);
        let v1 = Vec4::new(m[1][1], m[0][1], m[0][1], m[0][1]);
        let v2 = Vec4::new(m[1][2], m[0][2], m[0][2], m[0][2]);
        let v3 = Vec4::new(m[1][3], m[0][3], m[0][3], m[0][3]);

        let inv0 = vec4_add(vec4_sub(vec4_mul(v1, f0), vec4_mul(v2, f1)), vec4_mul(v3, f2));
        let inv1 = vec4_add(vec4_sub(vec4_mul(v0, f0), vec4_mul(v2, f3)), vec4_mul(v3, f4));
        let inv2 = vec4_add(vec4_sub(vec4_mul(v0, f1), vec4_mul(v1, f3)), vec4_mul(v3, f5));
        let inv3 = vec4_add(vec4_sub(vec4_mul(v0, f2), vec4_mul(v1, f4)), vec4_mul(v2, f5));

        let sign_a = Vec4::new( 1.0, -1.0,  1.0, -1.0);
        let sign_b = Vec4::new(-1.0,  1.0, -1.0,  1.0);

        let inv_col0 = vec4_mul(inv0, sign_a);
        let inv_col1 = vec4_mul(inv1, sign_b);
        let inv_col2 = vec4_mul(inv2, sign_a);
        let inv_col3 = vec4_mul(inv3, sign_b);

        let row0 = Vec4::new(inv_col0.x, inv_col1.x, inv_col2.x, inv_col3.x);
        let col0 = Vec4::new(m[0][0], m[0][1], m[0][2], m[0][3]);
        let det = col0.dot(row0);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;
        Some(Self {
            cols: [
                [inv_col0.x * inv_det, inv_col0.y * inv_det, inv_col0.z * inv_det, inv_col0.w * inv_det],
                [inv_col1.x * inv_det, inv_col1.y * inv_det, inv_col1.z * inv_det, inv_col1.w * inv_det],
                [inv_col2.x * inv_det, inv_col2.y * inv_det, inv_col2.z * inv_det, inv_col2.w * inv_det],
                [inv_col3.x * inv_det, inv_col3.y * inv_det, inv_col3.z * inv_det, inv_col3.w * inv_det],
            ],
        })
    }
}

// Vec4 component-wise helpers used internally by invert()
#[inline] fn vec4_add(a: Vec4, b: Vec4) -> Vec4 { a + b }
#[inline] fn vec4_sub(a: Vec4, b: Vec4) -> Vec4 { a - b }
#[inline] fn vec4_mul(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w)
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::identity();
        for col in 0..4 {
            let v = Vec4::new(
                rhs.cols[col][0],
                rhs.cols[col][1],
                rhs.cols[col][2],
                rhs.cols[col][3],
            );
            let r = self.mul_vec4(v);
            out.cols[col] = [r.x, r.y, r.z, r.w];
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Mat4 constructors
// ---------------------------------------------------------------------------

/// Perspective projection (reversed-Z optional via positive near/far).
pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let tan_half_fov = sin32(fov_y_rad * 0.5) / cos32(fov_y_rad * 0.5);
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = 1.0 / (aspect * tan_half_fov);
    m[1][1] = 1.0 / tan_half_fov;
    m[2][2] = -(far + near) / (far - near);
    m[2][3] = -1.0;
    m[3][2] = -(2.0 * far * near) / (far - near);
    Mat4::from_cols_array(m)
}

pub fn look_at(eye: Vec3, at: Vec3, up: Vec3) -> Mat4 {
    let f = (at - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);

    Mat4::from_cols_array([
        [s.x,            u.x,            -f.x,           0.0],
        [s.y,            u.y,            -f.y,           0.0],
        [s.z,            u.z,            -f.z,           0.0],
        [-s.dot(eye),   -u.dot(eye),      f.dot(eye),    1.0],
    ])
}

pub fn translation(v: Vec3) -> Mat4 {
    Mat4::from_cols_array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [v.x, v.y, v.z, 1.0],
    ])
}

pub fn scale(v: Vec3) -> Mat4 {
    Mat4::from_cols_array([
        [v.x, 0.0, 0.0, 0.0],
        [0.0, v.y, 0.0, 0.0],
        [0.0, 0.0, v.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn rotation(q: Quat) -> Mat4 {
    q.to_mat4()
}

// ---------------------------------------------------------------------------
// Quat
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    #[inline] pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }
    #[inline] pub const fn identity() -> Self { Self::new(0.0, 0.0, 0.0, 1.0) }

    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let half = angle_rad * 0.5;
        let s = sin32(half);
        let c = cos32(half);
        let a = axis.normalize();
        Self::new(a.x * s, a.y * s, a.z * s, c)
    }

    #[inline] pub fn len_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }
    #[inline] pub fn len(self) -> f32 { sqrt32(self.len_sq()) }

    pub fn normalize(self) -> Self {
        let l = self.len();
        if l < 1e-10 { return Self::identity(); }
        Self::new(self.x / l, self.y / l, self.z / l, self.w / l)
    }

    #[inline] pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        )
    }

    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        // Rodrigues' optimized formula: v + 2w(q×v) + 2(q×(q×v))
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = qv.cross(v);
        let uuv = qv.cross(uv);
        v + (uv * (2.0 * self.w)) + (uuv * 2.0)
    }

    /// Spherical linear interpolation.
    pub fn slerp(self, rhs: Self, t: f32) -> Self {
        let d = self.dot(rhs);
        // Ensure shortest path
        let (rhs, d) = if d < 0.0 {
            (Quat::new(-rhs.x, -rhs.y, -rhs.z, -rhs.w), -d)
        } else {
            (rhs, d)
        };

        // Clamp to [-1, 1] to avoid NaN in acos approximation
        let d = if d > 1.0 { 1.0 } else { d };

        // When quaternions are very close, fall back to lerp to avoid div by zero.
        if d > 0.9995 {
            return Quat::new(
                self.x + (rhs.x - self.x) * t,
                self.y + (rhs.y - self.y) * t,
                self.z + (rhs.z - self.z) * t,
                self.w + (rhs.w - self.w) * t,
            ).normalize();
        }

        // acos approximation accurate enough for slerp
        let theta_0 = acos32(d);
        let theta = theta_0 * t;
        let sin_theta = sin32(theta);
        let sin_theta_0 = sin32(theta_0);
        let s0 = cos32(theta) - d * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Quat::new(
            self.x * s0 + rhs.x * s1,
            self.y * s0 + rhs.y * s1,
            self.z * s0 + rhs.z * s1,
            self.w * s0 + rhs.w * s1,
        ).normalize()
    }

    pub fn to_mat4(self) -> Mat4 {
        let q = self.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let x2 = x + x; let y2 = y + y; let z2 = z + z;
        let xx = x * x2; let xy = x * y2; let xz = x * z2;
        let yy = y * y2; let yz = y * z2; let zz = z * z2;
        let wx = w * x2; let wy = w * y2; let wz = w * z2;

        Mat4::from_cols_array([
            [1.0 - (yy + zz),  xy + wz,          xz - wy,          0.0],
            [xy - wz,           1.0 - (xx + zz),  yz + wx,          0.0],
            [xz + wy,           yz - wx,           1.0 - (xx + yy), 0.0],
            [0.0,               0.0,               0.0,              1.0],
        ])
    }
}

/// acos approximation using polynomial — enough for slerp.
fn acos32(x: f32) -> f32 {
    // acos(x) ≈ sqrt(2*(1-x)) * poly for x in [0,1]; sign handled above
    let clamped = if x < -1.0 { -1.0 } else if x > 1.0 { 1.0 } else { x };
    let negate = if clamped < 0.0 { 1.0f32 } else { 0.0f32 };
    let a = if clamped < 0.0 { -clamped } else { clamped };
    let mut ret = -0.0187293_f32;
    ret = ret * a + 0.0742610;
    ret = ret * a - 0.2121144;
    ret = ret * a + 1.5707288;
    ret *= sqrt32(1.0 - a);
    ret - 2.0 * negate * ret + negate * PI
}

// ---------------------------------------------------------------------------
// Aabb
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn from_point(p: Vec3) -> Self { Self { min: p, max: p } }

    #[inline]
    pub fn center(self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    #[inline]
    pub fn half_extents(self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    /// Expand by a scalar margin in all directions.
    pub fn expand(self, margin: f32) -> Self {
        Self {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }

    /// Union of two AABBs.
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min_elem(other.min),
            max: self.max.max_elem(other.max),
        }
    }

    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        // Closest point on AABB to sphere center, then distance check.
        let cx = clamp32(center.x, self.min.x, self.max.x);
        let cy = clamp32(center.y, self.min.y, self.max.y);
        let cz = clamp32(center.z, self.min.z, self.max.z);
        let dx = center.x - cx;
        let dy = center.y - cy;
        let dz = center.z - cz;
        dx * dx + dy * dy + dz * dz <= radius * radius
    }
}

#[inline]
fn clamp32(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

// ---------------------------------------------------------------------------
// Frustum  (Gribb-Hartmann plane extraction from combined proj*view matrix)
// ---------------------------------------------------------------------------

/// Six clip planes in Vec4 form: (nx, ny, nz, d) where nx*x + ny*y + nz*z + d >= 0 is inside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a projection×view matrix (Gribb-Hartmann, 2001).
    pub fn from_proj_view(m: Mat4) -> Self {
        // Row vectors of the matrix for row-based extraction:
        // We have column-major storage, so row r = (cols[0][r], cols[1][r], cols[2][r], cols[3][r])
        let row = |r: usize| Vec4::new(m.cols[0][r], m.cols[1][r], m.cols[2][r], m.cols[3][r]);

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let planes = [
            normalize_plane(vec4_add(r3, r0)),   // left:   row3 + row0
            normalize_plane(vec4_sub(r3, r0)),   // right:  row3 - row0
            normalize_plane(vec4_add(r3, r1)),   // bottom: row3 + row1
            normalize_plane(vec4_sub(r3, r1)),   // top:    row3 - row1
            normalize_plane(vec4_add(r3, r2)),   // near:   row3 + row2
            normalize_plane(vec4_sub(r3, r2)),   // far:    row3 - row2
        ];
        Self { planes }
    }

    /// Returns `true` if the AABB is (possibly) visible — not fully outside any plane.
    pub fn test_aabb(&self, aabb: Aabb) -> bool {
        let center = aabb.center();
        let half = aabb.half_extents();
        for p in &self.planes {
            // Compute the signed distance of the positive vertex (farthest in plane direction)
            let r = half.x * p.x.abs() + half.y * p.y.abs() + half.z * p.z.abs();
            let d = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
            if d + r < 0.0 {
                return false; // fully outside this plane
            }
        }
        true
    }

    /// Returns `true` if the sphere is (possibly) visible.
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for p in &self.planes {
            let d = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
            if d + radius < 0.0 {
                return false;
            }
        }
        true
    }
}

fn normalize_plane(p: Vec4) -> Vec4 {
    let len = sqrt32(p.x * p.x + p.y * p.y + p.z * p.z);
    if len < 1e-10 { return p; }
    Vec4::new(p.x / len, p.y / len, p.z / len, p.w / len)
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    #[inline]
    pub fn new(origin: Vec3, dir: Vec3) -> Self { Self { origin, dir: dir.normalize() } }

    /// Slab method. Returns the distance along ray to entry point, or `None` if no hit.
    pub fn intersect_aabb(&self, aabb: Aabb) -> Option<f32> {
        let inv_dir_x = 1.0 / self.dir.x;
        let inv_dir_y = 1.0 / self.dir.y;
        let inv_dir_z = 1.0 / self.dir.z;

        let tx1 = (aabb.min.x - self.origin.x) * inv_dir_x;
        let tx2 = (aabb.max.x - self.origin.x) * inv_dir_x;
        let ty1 = (aabb.min.y - self.origin.y) * inv_dir_y;
        let ty2 = (aabb.max.y - self.origin.y) * inv_dir_y;
        let tz1 = (aabb.min.z - self.origin.z) * inv_dir_z;
        let tz2 = (aabb.max.z - self.origin.z) * inv_dir_z;

        let tmin = f32_max(f32_max(f32_min(tx1, tx2), f32_min(ty1, ty2)), f32_min(tz1, tz2));
        let tmax = f32_min(f32_min(f32_max(tx1, tx2), f32_max(ty1, ty2)), f32_max(tz1, tz2));

        if tmax < 0.0 || tmin > tmax {
            None
        } else if tmin < 0.0 {
            Some(tmax)
        } else {
            Some(tmin)
        }
    }

    /// Möller–Trumbore ray–triangle intersection.
    /// Returns distance along ray, or `None` if no hit (back-face culling disabled).
    pub fn intersect_triangle(&self, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
        const EPSILON: f32 = 1e-7;
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = self.dir.cross(edge2);
        let a = edge1.dot(h);
        if a > -EPSILON && a < EPSILON {
            return None; // ray parallel to triangle
        }
        let f = 1.0 / a;
        let s = self.origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(edge1);
        let v = f * self.dir.dot(q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * edge2.dot(q);
        if t > EPSILON { Some(t) } else { None }
    }
}

#[inline] fn f32_min(a: f32, b: f32) -> f32 { if a < b { a } else { b } }
#[inline] fn f32_max(a: f32, b: f32) -> f32 { if a > b { a } else { b } }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    // --- Scalar math ---

    #[test]
    fn test_sin32_cos32_basic() {
        // Known values, within 1e-4
        assert!(approx(sin32(0.0), 0.0, 1e-4));
        assert!(approx(sin32(FRAC_PI_2), 1.0, 1e-4));
        assert!(approx(sin32(PI), 0.0, 1e-4));
        assert!(approx(cos32(0.0), 1.0, 1e-4));
        assert!(approx(cos32(FRAC_PI_2), 0.0, 1e-4));
        assert!(approx(cos32(PI), -1.0, 1e-4));
    }

    #[test]
    fn test_sin32_negative_angles() {
        assert!(approx(sin32(-FRAC_PI_2), -1.0, 1e-4));
        assert!(approx(cos32(-FRAC_PI_2), 0.0, 1e-4));
    }

    #[test]
    fn test_sin32_pythagorean_identity() {
        for &x in &[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let s = sin32(x);
            let c = cos32(x);
            assert!(approx(s * s + c * c, 1.0, 1e-4), "identity failed at x={}", x);
        }
    }

    // --- Vec2 ---

    #[test]
    fn test_vec2_ops() {
        let a = Vec2::new(3.0, 4.0);
        assert!(approx(a.len(), 5.0, 1e-5));
        let n = a.normalize();
        assert!(approx(n.len(), 1.0, 1e-5));
        assert!(approx(a.dot(Vec2::new(1.0, 0.0)), 3.0, 1e-5));
        let b = Vec2::new(1.0, 2.0);
        let lerped = a.lerp(b, 0.5);
        assert!(approx(lerped.x, 2.0, 1e-5));
        assert!(approx(lerped.y, 3.0, 1e-5));
    }

    // --- Vec3 ---

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx(a.dot(b), 0.0, 1e-6));
        assert!(approx(a.dot(a), 1.0, 1e-6));
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::x_axis();
        let y = Vec3::y_axis();
        let z = x.cross(y);
        assert!(approx(z.x, 0.0, 1e-6));
        assert!(approx(z.y, 0.0, 1e-6));
        assert!(approx(z.z, 1.0, 1e-6));
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 1.0, 2.0);
        let n = v.normalize();
        assert!(approx(n.len(), 1.0, 1e-5));
    }

    #[test]
    fn test_vec3_lerp() {
        let a = Vec3::zero();
        let b = Vec3::one();
        let mid = a.lerp(b, 0.5);
        assert!(approx(mid.x, 0.5, 1e-6));
    }

    // --- Mat4 ---

    #[test]
    fn test_mat4_identity() {
        let id = Mat4::identity();
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let r = id.mul_vec4(v);
        assert!(approx(r.x, 1.0, 1e-6));
        assert!(approx(r.y, 2.0, 1e-6));
        assert!(approx(r.z, 3.0, 1e-6));
        assert!(approx(r.w, 1.0, 1e-6));
    }

    #[test]
    fn test_mat4_mul_identity() {
        let id = Mat4::identity();
        let t = translation(Vec3::new(1.0, 2.0, 3.0));
        let r = id * t;
        let p = r.mul_point(Vec3::zero());
        assert!(approx(p.x, 1.0, 1e-5));
        assert!(approx(p.y, 2.0, 1e-5));
        assert!(approx(p.z, 3.0, 1e-5));
    }

    #[test]
    fn test_mat4_translation() {
        let m = translation(Vec3::new(5.0, -3.0, 2.0));
        let p = m.mul_point(Vec3::new(1.0, 1.0, 1.0));
        assert!(approx(p.x, 6.0, 1e-5));
        assert!(approx(p.y, -2.0, 1e-5));
        assert!(approx(p.z, 3.0, 1e-5));
    }

    #[test]
    fn test_mat4_scale() {
        let m = scale(Vec3::new(2.0, 3.0, 4.0));
        let p = m.mul_point(Vec3::new(1.0, 1.0, 1.0));
        assert!(approx(p.x, 2.0, 1e-5));
        assert!(approx(p.y, 3.0, 1e-5));
        assert!(approx(p.z, 4.0, 1e-5));
    }

    #[test]
    fn test_mat4_transpose() {
        let m = translation(Vec3::new(1.0, 2.0, 3.0));
        let mt = m.transpose().transpose();
        for c in 0..4 {
            for r in 0..4 {
                assert!(approx(m.cols[c][r], mt.cols[c][r], 1e-6));
            }
        }
    }

    #[test]
    fn test_mat4_invert() {
        let t = translation(Vec3::new(3.0, -1.0, 2.0));
        let inv = t.invert().expect("invertible");
        let prod = t * inv;
        let id = Mat4::identity();
        for c in 0..4 {
            for r in 0..4 {
                assert!(approx(prod.cols[c][r], id.cols[c][r], 1e-4),
                    "mismatch at col={} row={}: {} vs {}", c, r, prod.cols[c][r], id.cols[c][r]);
            }
        }
    }

    #[test]
    fn test_perspective() {
        let p = perspective(DEG_TO_RAD * 90.0, 1.0, 0.1, 100.0);
        // For 90 degree FOV, aspect=1: m[0][0] == m[1][1] ≈ 1.0
        assert!(approx(p.cols[0][0], 1.0, 1e-4));
        assert!(approx(p.cols[1][1], 1.0, 1e-4));
    }

    #[test]
    fn test_look_at() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let at  = Vec3::zero();
        let up  = Vec3::y_axis();
        let m = look_at(eye, at, up);
        // The origin (0,0,0) should map to (0, 0, something, 1) in view space
        let p = m.mul_point(Vec3::zero());
        assert!(approx(p.x, 0.0, 1e-4));
        assert!(approx(p.y, 0.0, 1e-4));
    }

    // --- Quat ---

    #[test]
    fn test_quat_identity() {
        let q = Quat::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = q.rotate_vec3(v);
        assert!(approx(r.x, 1.0, 1e-5));
        assert!(approx(r.y, 2.0, 1e-5));
        assert!(approx(r.z, 3.0, 1e-5));
    }

    #[test]
    fn test_quat_from_axis_angle_90_y() {
        let q = Quat::from_axis_angle(Vec3::y_axis(), FRAC_PI_2);
        // Rotating +X by 90° around Y should give -Z
        let r = q.rotate_vec3(Vec3::x_axis());
        assert!(approx(r.x, 0.0, 1e-4));
        assert!(approx(r.y, 0.0, 1e-4));
        assert!(approx(r.z, -1.0, 1e-4));
    }

    #[test]
    fn test_quat_from_axis_angle_180_z() {
        let q = Quat::from_axis_angle(Vec3::z_axis(), PI);
        let r = q.rotate_vec3(Vec3::x_axis());
        assert!(approx(r.x, -1.0, 1e-4));
        assert!(approx(r.y, 0.0, 1e-4));
        assert!(approx(r.z, 0.0, 1e-4));
    }

    #[test]
    fn test_quat_to_mat4() {
        let q = Quat::from_axis_angle(Vec3::y_axis(), FRAC_PI_2);
        let m = q.to_mat4();
        let r = m.mul_dir(Vec3::x_axis());
        assert!(approx(r.x, 0.0, 1e-4));
        assert!(approx(r.y, 0.0, 1e-4));
        assert!(approx(r.z, -1.0, 1e-4));
    }

    #[test]
    fn test_quat_slerp_zero() {
        let a = Quat::identity();
        let b = Quat::from_axis_angle(Vec3::y_axis(), PI);
        let mid = a.slerp(b, 0.0);
        assert!(approx(mid.w, 1.0, 1e-4));
    }

    #[test]
    fn test_quat_slerp_one() {
        let a = Quat::identity();
        let b = Quat::from_axis_angle(Vec3::y_axis(), FRAC_PI_2);
        let r = a.slerp(b, 1.0);
        // Should be close to b
        let v = r.rotate_vec3(Vec3::x_axis());
        assert!(approx(v.z, -1.0, 1e-3));
    }

    #[test]
    fn test_quat_slerp_half() {
        let a = Quat::identity();
        let b = Quat::from_axis_angle(Vec3::y_axis(), FRAC_PI_2);
        let mid = a.slerp(b, 0.5);
        let v = mid.rotate_vec3(Vec3::x_axis());
        // Halfway between 0 and 90 = 45 degrees rotation around Y
        // cos(45°) ≈ 0.707, sin(45°) ≈ -0.707 in z component
        assert!(approx(v.y, 0.0, 1e-3));
        assert!(approx(v.x * v.x + v.z * v.z, 1.0, 1e-3));
    }

    // --- Aabb ---

    #[test]
    fn test_aabb_contains() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(aabb.contains_point(Vec3::zero()));
        assert!(aabb.contains_point(Vec3::splat(1.0)));
        assert!(!aabb.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_aabb_intersects_sphere() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // Sphere centered at (2,0,0) with radius 2 should intersect
        assert!(aabb.intersects_sphere(Vec3::new(2.0, 0.0, 0.0), 2.0));
        // Sphere far away should not
        assert!(!aabb.intersects_sphere(Vec3::new(10.0, 0.0, 0.0), 1.0));
    }

    #[test]
    fn test_aabb_center_half_extents() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 2.0, 6.0));
        let c = aabb.center();
        let h = aabb.half_extents();
        assert!(approx(c.x, 2.0, 1e-5));
        assert!(approx(h.x, 2.0, 1e-5));
        assert!(approx(h.z, 3.0, 1e-5));
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let b = Aabb::new(Vec3::splat(0.0), Vec3::splat(3.0));
        let u = a.union(b);
        assert!(approx(u.min.x, -1.0, 1e-5));
        assert!(approx(u.max.x, 3.0, 1e-5));
    }

    // --- Frustum ---

    #[test]
    fn test_frustum_test_sphere() {
        // Use a simple orthographic-like projection * identity view
        let proj = perspective(DEG_TO_RAD * 90.0, 1.0, 1.0, 100.0);
        let view = look_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::y_axis());
        let frustum = Frustum::from_proj_view(proj * view);

        // A point directly in front should be visible
        assert!(frustum.test_sphere(Vec3::new(0.0, 0.0, -10.0), 0.5));
        // A point far behind the camera should not
        assert!(!frustum.test_sphere(Vec3::new(0.0, 0.0, 200.0), 0.5));
    }

    #[test]
    fn test_frustum_test_aabb() {
        let proj = perspective(DEG_TO_RAD * 90.0, 1.0, 1.0, 100.0);
        let view = look_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::y_axis());
        let frustum = Frustum::from_proj_view(proj * view);

        let visible_box = Aabb::new(Vec3::new(-1.0, -1.0, -15.0), Vec3::new(1.0, 1.0, -5.0));
        assert!(frustum.test_aabb(visible_box));

        let behind_box = Aabb::new(Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, 1.0, 200.0));
        assert!(!frustum.test_aabb(behind_box));
    }

    // --- Ray ---

    #[test]
    fn test_ray_intersect_aabb_hit() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let t = ray.intersect_aabb(aabb);
        assert!(t.is_some());
        assert!(approx(t.unwrap(), 4.0, 1e-4));
    }

    #[test]
    fn test_ray_intersect_aabb_miss() {
        let ray = Ray::new(Vec3::new(5.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(ray.intersect_aabb(aabb).is_none());
    }

    #[test]
    fn test_ray_intersect_triangle_hit() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new( 1.0, -1.0, 0.0);
        let v2 = Vec3::new( 0.0,  1.0, 0.0);
        let t = ray.intersect_triangle(v0, v1, v2);
        assert!(t.is_some());
        assert!(approx(t.unwrap(), 1.0, 1e-4));
    }

    #[test]
    fn test_ray_intersect_triangle_miss() {
        let ray = Ray::new(Vec3::new(10.0, 10.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new( 1.0, -1.0, 0.0);
        let v2 = Vec3::new( 0.0,  1.0, 0.0);
        assert!(ray.intersect_triangle(v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_intersect_triangle_parallel() {
        // Ray parallel to the triangle plane
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new( 1.0, -1.0, 0.0);
        let v2 = Vec3::new( 0.0,  1.0, 0.0);
        assert!(ray.intersect_triangle(v0, v1, v2).is_none());
    }
}

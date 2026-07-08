//! Runtime values for the Dominion interpreter.
//!
//! Beyond the lowest-level primitives, Dominion elevates *semantic primitives* to
//! first-class citizens (SRS §5.3): [`Value::Identity`] (a cryptographic
//! principal) and [`Value::Latent`] (a neural-compressed representation for the
//! generative-storage layer). These are real values the language can pass
//! around, not library afterthoughts.

use crate::datatypes::{HyperVector, SpikeTrain, Tensor};
use crate::hash::Hash256;
use crate::ml::Mlp;
use crate::numerics::{BigInt, Complex, Decimal, Dual, Interval, Quaternion, Rational};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Unit,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Vector(Vec<Value>),
    /// A constructed semantic object: a kind plus named fields.
    Object {
        kind: String,
        fields: Vec<(String, Value)>,
    },
    /// A cryptographic identity (user/process/system) — semantic primitive.
    Identity(String),
    /// A natively neural-compressed object: its content hash and the achieved
    /// compression ratio (SRS §7.1 generative compression / `Latent<T>`).
    Latent {
        of: Hash256,
        ratio: f64,
    },
    /// A dense numeric tensor — routed to the GPU node by the type-directed router.
    Tensor(Tensor),
    /// A hyperdimensional vector — routed to the NPU node.
    HyperVector(HyperVector),
    /// A neuromorphic spike train — routed to the NPU/neuromorphic node.
    SpikeTrain(SpikeTrain),
    /// A trainable/inferable neural network (an MLP) — a first-class learned model
    /// the language can build, train and run. Routed to the GPU/TPU node.
    Model(Mlp),
    // ---- high-precision & non-real numeric primitives (numerics.rs) ----
    /// An arbitrary-precision integer (never overflows).
    BigInt(BigInt),
    /// An arbitrary-precision base-10 number — the exact, error-resistant float.
    Decimal(Decimal),
    /// An exact fraction `p/q` (zero rounding).
    Rational(Rational),
    /// A complex number `a + bi`.
    Complex(Complex),
    /// A dual number `a + bε` for exact forward-mode autodiff.
    Dual(Dual),
    /// A rigorous error-bounded range `[lo, hi]`.
    Interval(Interval),
    /// A quaternion `w + xi + yj + zk` for 3-D rotation algebra.
    Quaternion(Quaternion),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "Unit",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "Str",
            Value::Vector(_) => "Vector",
            Value::Object { .. } => "Object",
            Value::Identity(_) => "Identity",
            Value::Latent { .. } => "Latent",
            Value::Tensor(_) => "Tensor",
            Value::HyperVector(_) => "HyperVector",
            Value::SpikeTrain(_) => "SpikeTrain",
            Value::Model(_) => "Model",
            Value::BigInt(_) => "BigInt",
            Value::Decimal(_) => "Decimal",
            Value::Rational(_) => "Rational",
            Value::Complex(_) => "Complex",
            Value::Dual(_) => "Dual",
            Value::Interval(_) => "Interval",
            Value::Quaternion(_) => "Quaternion",
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Unit => false,
            Value::Int(i) => *i != 0,
            Value::BigInt(b) => !b.is_zero(),
            Value::Decimal(d) => !d.is_zero(),
            Value::Rational(r) => !r.is_zero(),
            _ => true,
        }
    }

    /// Content hash of a value — used by the `hash` builtin and by the storage
    /// layer. Encoding mirrors [`crate::object`]'s canonical form.
    pub fn content_hash(&self) -> Hash256 {
        Hash256::of(self.encode().as_bytes())
    }

    /// A stable, collision-free key for use as a memoization cache key, or `None`
    /// for values that must never be memoised on (a `Float` that is `NaN`, since
    /// `NaN != NaN` would make a cache hit observably wrong, and `Model`, which is
    /// large/opaque). Every other value's `encode` is a deterministic, total,
    /// injective-enough string for cache-keying.
    pub fn encode_key(&self) -> Option<String> {
        match self {
            Value::Float(f) if f.is_nan() => None,
            Value::Model(_) => None,
            Value::Vector(items) => {
                let mut key = String::from("v:[");
                for it in items {
                    key.push_str(&it.encode_key()?);
                    key.push(',');
                }
                key.push(']');
                Some(key)
            }
            _ => Some(self.encode()),
        }
    }

    fn encode(&self) -> String {
        match self {
            Value::Unit => "unit".to_string(),
            Value::Int(i) => format!("i:{}", i),
            Value::Float(f) => format!("f:{}", f),
            Value::Bool(b) => format!("b:{}", b),
            Value::Str(s) => format!("s:{}", s),
            Value::Identity(s) => format!("id:{}", s),
            Value::Latent { of, ratio } => format!("lat:{}:{}", of.to_hex(), ratio),
            // Debug formatting of these is fully deterministic over their contents.
            Value::Tensor(t) => format!("tensor:{:?}", t),
            Value::HyperVector(h) => format!("hv:{:?}", h),
            Value::SpikeTrain(s) => format!("spikes:{:?}", s),
            // Hash the canonical model bytes so identical models content-address alike.
            Value::Model(m) => format!("model:{}", Hash256::of(&m.to_bytes()).to_hex()),
            Value::BigInt(b) => format!("bigint:{}", b.to_decimal_string()),
            Value::Decimal(d) => format!("dec:{}", d),
            Value::Rational(r) => format!("rat:{}", r),
            Value::Complex(c) => format!("cplx:{}", c),
            Value::Dual(d) => format!("dual:{}", d),
            Value::Interval(iv) => format!("ivl:{}", iv),
            Value::Quaternion(q) => format!("quat:{}", q),
            Value::Vector(items) => {
                let mut s = String::from("v:[");
                for it in items {
                    s.push_str(&it.encode());
                    s.push(',');
                }
                s.push(']');
                s
            }
            Value::Object { kind, fields } => {
                let mut sorted: Vec<&(String, Value)> = fields.iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                let mut s = format!("o:{}{{", kind);
                for (k, v) in sorted {
                    s.push_str(k);
                    s.push('=');
                    s.push_str(&v.encode());
                    s.push(';');
                }
                s.push('}');
                s
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => f.write_str("()"),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(v) => write!(f, "{}", v),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Str(s) => write!(f, "{}", s),
            Value::Identity(s) => write!(f, "@{}", s),
            Value::Latent { of, ratio } => write!(f, "Latent<{} x{:.2}>", of.short(), ratio),
            Value::Tensor(t) => write!(f, "Tensor{:?}", t.shape()),
            Value::HyperVector(h) => write!(f, "HyperVector<{}>", h.dim()),
            Value::SpikeTrain(s) => write!(f, "SpikeTrain<{} spikes>", s.count()),
            Value::Model(m) => write!(
                f,
                "Model<{}→{}, {} params>",
                m.in_dim(),
                m.out_dim(),
                m.param_count()
            ),
            Value::BigInt(b) => write!(f, "{}", b.to_decimal_string()),
            Value::Decimal(d) => write!(f, "{}", d),
            Value::Rational(r) => write!(f, "{}", r),
            Value::Complex(c) => write!(f, "{}", c),
            Value::Dual(d) => write!(f, "{}", d),
            Value::Interval(iv) => write!(f, "{}", iv),
            Value::Quaternion(q) => write!(f, "{}", q),
            Value::Vector(items) => {
                f.write_str("[")?;
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", it)?;
                }
                f.write_str("]")
            }
            Value::Object { kind, fields } => {
                write!(f, "{} {{ ", kind)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                f.write_str(" }")
            }
        }
    }
}

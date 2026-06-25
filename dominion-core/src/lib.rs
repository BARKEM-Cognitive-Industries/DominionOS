//! # dominion-core
//!
//! The architecture-independent heart of **DominionOS**.
//!
//! This crate is `no_std` when compiled into the kernel, but links `std`
//! during `cargo test` (see the `cfg_attr` below) so that every component
//! ships with fast, native unit tests. None of the logic here depends on a
//! particular machine — it is pure data and transformation, exactly as the
//! SRS demands of an intralingual, deterministic system.
//!
//! Modules map directly onto the System Requirements Specification:
//!
//! * [`hash`]      — content addressing (Stage 5/7: the semantic graph is a system-wide Git).
//! * [`capability`]— Stage 2/3: CHERI-style unforgeable capability tokens.
//! * [`object`]    — Stage 5/7: the content-addressed, immutable semantic object graph.
//! * [`state`]     — Stage 10: the deterministic, hashable machine state machine.
//! * [`lang`]      — the Dominion language (lexer, parser, interpreter).
//! * [`codec`]     — keystone K2: legacy bytes ⇄ semantic objects + verbatim Blobs.
//! * [`vfs`]       — keystone K1: a POSIX path namespace projected over the graph.
#![cfg_attr(not(test), no_std)]
#![deny(unsafe_code)]
// The opt-in `fma` feature (off by default) enables the hardware fused multiply-add in
// the matmul kernel via `core::intrinsics::fmaf64` — a *safe* intrinsic, so the crate
// keeps its `forbid(unsafe_code)` guarantee even with FMA on. It is ≈2× on
// multiply-bound matmul but changes the low bits (single fused rounding), so it is
// off by default to preserve bit-determinism. See `datatypes::madd` / `ml`.
#![cfg_attr(feature = "fma", feature(core_intrinsics))]
#![cfg_attr(feature = "fma", allow(internal_features))]
#![cfg_attr(feature = "simd", feature(portable_simd))]

extern crate alloc;

pub mod math;
pub mod arch;
pub mod enforcement;
pub mod hash;
pub mod capability;
pub mod cheri;
pub mod verify;
pub mod secureboot;
pub mod memenc;
pub mod bytes;
pub mod object;
pub mod content_store;
pub mod datatypes;
pub mod parallel;
pub mod numerics;
pub mod neural;
pub mod ml;
pub mod nn;
pub mod memo;
pub mod defense;
pub mod lang;
pub mod dcg;
pub mod state;
pub mod codec;
pub mod vfs;
pub mod filesystem;
pub mod files;
pub mod persist;
pub mod objstore;
pub mod journal;
pub mod durability;
pub mod sched;
pub mod hlc;
pub mod bft;
pub mod dsasos;
pub mod marketplace;
pub mod settlement;
pub mod hardalloc;
pub mod amnesic;
pub mod deniable;
pub mod privacy;
pub mod lsp;
pub mod credential;
pub mod backup;
pub mod fleetsync;
pub mod pressure;
pub mod memtier;
pub mod ramdedup;
pub mod multikernel;
pub mod zerocopy;
pub mod zcpring;
pub mod governor;
pub mod placement;
pub mod supervisor;
pub mod elf;
pub mod net;
pub mod legacynet;
pub mod dominionlink;
pub mod transport;
pub mod ndn;
pub mod url;
pub mod http;
pub mod tlscrypto;
pub mod x509;
pub mod tls;
pub mod dom;
pub mod css;
pub mod html;
pub mod js;
pub mod webengine;
pub mod pubsub;
pub mod session;
pub mod sandbox;
pub mod wasm;
pub mod polyglot;
pub mod driver;
pub mod drivergen;
pub mod foreign;
pub mod foreignload;
pub mod compat;
pub mod capshim;
pub mod netspec;
pub mod conformance;
pub mod net_bench;
pub mod personality;
pub mod dominionweb;
pub mod surface;
pub mod audio;
pub mod ui;
pub mod toolkit;
pub mod anim;
pub mod widgets;
pub mod nodes;
pub mod dash;
pub mod taskman;
pub mod world;
pub mod desktop_page;
pub mod explorer;
pub mod ide;
pub mod keys;
pub mod window;
pub mod os;
pub mod text;
pub mod editor;
pub mod highlight;
pub mod terminal;
pub mod shellcmd;
pub mod termpage;
pub mod compose;
pub mod workspace;
pub mod browser;
pub mod browserapp;
pub mod editorpage;
pub mod settings;
pub mod shell;
pub mod appkit;
pub mod a11y;
pub mod agent;
pub mod wcag;
pub mod i18n;
pub mod power;
pub mod pmgmt;
pub mod random;
pub mod crypto;
pub mod tokensig;
pub mod lattice;
pub mod memcrypt;
pub mod chacha;
pub mod zk;
pub mod vcompute;
pub mod ctx;
pub mod anon;
pub mod vault;
pub mod firewall;
pub mod airlock;
pub mod consent;
pub mod threat;
pub mod attest;
pub mod secprofile;
pub mod confidential;
pub mod enclave;
pub mod webauth;
pub mod time;
pub mod recovery;
pub mod fleet;
pub mod identity;
pub mod rot;
pub mod update;
pub mod rollout;
pub mod lifecycle;
pub mod compliance;
pub mod dst;
pub mod consistency;
pub mod tensions;
pub mod fuzz;
pub mod props;
pub mod supplychain;
pub mod cryptoct;
pub mod testkit;
pub mod packaging;
pub mod discovery;
pub mod pool;
pub mod prefetch;
pub mod coldcomp;
pub mod percorealloc;
pub mod peerram;
pub mod stages;
pub mod ecosystem;

// ── Unified 2D/3D render stack (2d-3d rendering redesign.md) ─────────────────
// None of these modules are called from os.rs / shell.rs, so they are gated
// behind `render-full` to keep headless / CI builds lean.
#[cfg(feature = "render-full")] pub mod secnode;
#[cfg(feature = "render-full")] pub mod render_provenance;
#[cfg(feature = "render-full")] pub mod render_determinism;
// `math3d` (standalone) and `mesh` (depends only on `math3d`) are used by the
// always-compiled `toolkit` module, so they cannot be gated behind `render-full`.
pub mod math3d;
pub mod mesh;
#[cfg(feature = "render-full")] pub mod scene3d;
#[cfg(feature = "render-full")] pub mod rdg;
#[cfg(feature = "render-full")] pub mod raster3d;
#[cfg(feature = "render-full")] pub mod nanite;
#[cfg(feature = "render-full")] pub mod instances;
#[cfg(feature = "render-full")] pub mod vertanim;
#[cfg(feature = "render-full")] pub mod vectorpath;
#[cfg(feature = "render-full")] pub mod fontgpu;
#[cfg(feature = "render-full")] pub mod render_bench;
#[cfg(feature = "render-full")] pub mod hdr;
#[cfg(feature = "render-full")] pub mod sdf_shadow;
#[cfg(feature = "render-full")] pub mod atw;
#[cfg(feature = "render-full")] pub mod compositor_svc;
#[cfg(feature = "render-full")] pub mod idag;
#[cfg(feature = "render-full")] pub mod media_service;
#[cfg(feature = "render-full")] pub mod input_latch;
#[cfg(feature = "render-full")] pub mod psr2;
#[cfg(feature = "render-full")] pub mod render3d;

/// Re-exports the most common surface of the core library.
pub mod prelude {
    pub use crate::capability::{Capability, Rights};
    pub use crate::codec::{Blob, Codec, CodecRegistry};
    pub use crate::hash::Hash256;
    pub use crate::lang::{eval_source, Interpreter, Value};
    pub use crate::object::{Datum, Object, ObjectGraph, ObjectId};
    pub use crate::persist::{BlockDevice, BlockError, Persistence, RamDisk};
    pub use crate::sched::{DomainId, Scheduler, Step};
    pub use crate::state::Machine;
    pub use crate::vfs::Vfs;
}

/// The semantic version of the core, surfaced to the terminal banner.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

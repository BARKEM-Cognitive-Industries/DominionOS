# DominionOS benchmark suite

Two halves that emit the **same schema** so results sit side by side:

| Half | Where it runs | Script | Status |
|---|---|---|---|
| In-guest perf battery (9 workloads) | inside the booted OS (QEMU) | `../run-bench.ps1` → `kernel/src/bench.rs` | implemented |
| In-guest validation battery | inside the booted OS (QEMU) | `../run-validate.ps1` → `kernel/src/bench.rs` | implemented |
| Linux comparison harness | host / WSL / a Linux box | `linux/run-linux-bench.sh` | skeleton (fill in) |

The **validation battery** (`--features qemu_validate`) answers "is this real on
hardware, and does it survive stress?": a memory-latency mountain (real cache
hierarchy under whpx), the model-vs-hardware boundary, a **real cross-core scaling
curve** (genuine SMP — see below), a soak, and chaos/failure-injection. Run both
accelerators to see the cache mountain is real vs emulated:
`../run-validate.ps1 -Accels whpx,tcg`.

Both print machine-readable lines:

```
BENCH <category> key=value key=value ...
```

`run-bench.ps1` parses them into `../bench-results.json`; the Linux harness writes
`linux-results.json`. Compare matching `category` rows.

## The nine workloads and how each maps to Linux

| # | Category (`BENCH` line) | DominionOS subsystem exercised | Linux comparison |
|---|---|---|---|
| 1 | `task_creation`, `task_completion` | `sched::Scheduler` domains (SIPs) | `fork`/`pthread_create` spawn rate, peak RSS |
| 2 | `message_passing` | zero-copy channel IPC | pipe / UNIX-socket / eventfd ping-pong |
| 3 | `graph_execution`, `graph_scheduler` | DCG linear eval + `multikernel::WorkGraph` | DAG executor (topo eval); `make -j` / Taskflow |
| 4 | `memory_pressure` | fallible heap alloc to OOM + recovery | `malloc` to OOM under a cgroup memory cap |
| 5 | `storage_sequential/random/metadata/persist` | virtio-blk + object graph + persistence | `fio` seq/rand; many-small-files metadata |
| 6 | `distributed_messaging`, `distributed_crdt` | `multikernel` inter-core msgs + CRDT merge | **needs a cluster** — MPI/sockets; K8s pod fan-out |
| 7 | `security_overhead` | Vault SHA-256-CTR & AES-256-GCM vs plaintext | `openssl speed aes-256-gcm` vs plain copy |
| 8 | `dev_build`, `dev_test_suite`, `dev_depresolve` | Dominion parse / interpret / DCG compile | build a real codebase; run its tests; `cargo`/`npm` resolve |

## What is real, modelled, or deferred

- **Real, measured on metal:** 1–5, 7, 8 run actual DominionOS code at scale. Headline
  figures the heap can't hold literally (billions of messages) are measured at a
  feasible size and reported as `projected_*` by linear extrapolation.
- **Real cross-core scaling (single machine):** the validation battery's `scaling`
  rows are now a *genuine* curve — DominionOS brings up the application processors
  (`kernel/src/smp.rs`, the `smp` feature) and runs work on multiple host cores
  (8.3× on 8 cores). This is no longer a model.
- **Modelled (coordination substrate):** workload 6 (`distributed_*`) measures the
  *inter-core messaging + conflict-free merge* primitives, not true multi-machine
  scaling. The guest is one VM (now multi-core).
- **Deferred to a cluster:** true multi-*machine* scaling efficiency and the
  Linux + Kubernetes + containers comparison. The harness below has the hooks and
  TODOs; running them needs ≥2 machines (or a kind/k3d cluster).

## Caveats that change the numbers

- **Accelerator:** under TCG the guest is ~10-100× slower than native. Always
  record the `accel` field (whpx vs tcg) — it is in `bench-results.json`. Only
  compare DominionOS-on-whpx to Linux-on-bare-metal for an apples-ish comparison; a
  TCG run measures relative subsystem cost, not absolute throughput.
- **Single vCPU:** DominionOS is single-core today (no AP bring-up), so it cannot use
  more than one host core regardless of `-smp`. Compare against Linux pinned to one
  core (`taskset -c 0`) for fairness on single-threaded paths.
- **Known scalability findings** (see `../docs/FINDINGS.md`): scheduler dispatch is
  O(n) per step; `WorkGraph::schedule` is O(n²). The bench sizes around these and
  flags them so the comparison is honest.

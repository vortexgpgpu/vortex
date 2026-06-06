# CI Test-Time Optimization Proposal

## Goal

Bring the long-running GitHub CI groups under a reasonable per-job budget
without losing meaningful coverage. Current per-group wall-clock:

| Group | Now | blackbox runs (simx / rtlsim / opae+xrt) | NT=32 (rtlsim) |
|-------|-----|------------------------------------------|----------------|
| `--cache`      | 40m  | 34 (13 / 21 / 0)  | 0 |
| `--config1`    | 56m  | 49 (19 / 30 / 0)  | 0 |
| `--config2`    | 120m | 34 (5 / 7 / 22)   | 0 |
| `--dxa`        | 120m | 57 (30 / 27 / 0)  | 6 (3) |
| `--regression` | 60m  | 11 (5 / 2 / 4)    | 0 |
| `--tensor_sp`  | 120m | 89 (45 / 44 / 0)  | 18 (9) |
| `--tensor_wg`  | 60m  | 24 (14 / 10 / 0)  | 7 (2) |

## The cost model (measured)

A blackbox invocation = **rebuild the simulator for that config, then run**.
The build dominates, and which simulator decides everything:

- **simx**: a C++ recompile; the run is ~100x faster than rtlsim. Effectively cheap.
- **rtlsim**: a full **Verilator recompile (~2-3 min each)** + a cycle-accurate run.
- **opae / xrt**: FPGA *emulation* — the slowest build and multi-minute runs.

Measured on `--tensor_sp` (build64): the 87 kernel runs took **87 seconds total**;
the group's **53m33s** wall-clock was ~98% **per-config rebuilds**. Each distinct
`-D` config forces a recompile (2068 `sim/simx` + 129 `sim/rtlsim` makes).

**Corollary:** coverage on simx is nearly free; cost is the *count of distinct
rtlsim and opae/xrt builds*, and NT=32 rtlsim runs on top.

## Guiding principles

1. **rtlsim is the ground truth — keep its coverage ≥ simx.** rtlsim is the RTL
   that actually ships, so it needs *at least* as much coverage as simx; never
   trim rtlsim below simx to save time. The savings come from **shrinking the
   test matrix itself** — eliminating genuinely redundant configs (principles
   2-5) — applied to **both** backends, plus dropping the most expensive rtlsim
   points (NT=32) where they add no RTL coverage. simx then runs the *same*
   reduced matrix as a fast cross-check / numeric oracle, not a larger one.
2. **Never run both `opae` and `xrt` for the same config.** Both emulate the same
   AFU over the same RTL; one emulation driver per config is enough.
3. **Avoid NT=32 on rtlsim/emulation** except a single max-occupancy smoke.
   Thread/occupancy scaling is a simx concern; on rtlsim NT=32 is the most
   expensive point for no extra RTL coverage.
4. **Datatype axis = operand *size*, not format.** The width-dependent TCU RTL
   (operand packing, the bank-conflict-free RF mapping, lane sizing) is what
   changes with operand width; same-width formats share it, and signedness /
   int-vs-fp is an arithmetic-mux selection, not a separate width path. So
   `int8`≡`uint8`≡`fp8`≡`bf8` (8-bit), `int4`≡`uint4` (4-bit), `fp16`≡`bf16`
   (16-bit). Cover **one representative per size** — e.g. `{int4, int8, fp16,
   tf32}` (4/8/16/32-bit) — instead of all 9 formats. (Numeric correctness of
   each individual encoding is a DPI/softfloat reference-model concern, already
   covered there, not a per-format RTL run.)
5. **Collapse the format axis; keep the NT × size grid for sparse.** For *sparse*
   (2:4) tensor, the metadata storage (kMetaBanks slots, `sp_num_meta_loads`,
   per-element packing) is a function of **both NT and operand size**, so the
   NT × size grid is genuine coverage — keep it. Only the *format* axis collapses
   (principle 4). (Dense/`tensor_wg` already uses a deliberate Latin-square.)
5. **Split a group only if it can't be trimmed under budget.** With the 2-hour
   ceiling the deduped groups all fit, so no group is split.

## Per-group plan

### `--tensor_sp` (120m → ~50m)  — biggest win
Today: 9 formats × {NT=2,4,8,16,32} × {simx, rtlsim} = 90 configs, **45 of them
rtlsim Verilator builds** (the 120m). Only the **format axis** is redundant — the
NT × size grid must stay (sparse metadata storage depends on both, principle 5):
- **Format → size** (principle 4): 9 formats → **4 sizes** `{int4, int8, fp16, tf32}`.
- **Keep the full NT × size grid** {2,4,8,16,32} × 4 sizes on **both** drivers.
- ⇒ 4 × 5 × 2 = **40** runs (20 simx + 20 rtlsim), was 90 (45 rtlsim). rtlsim
  coverage = simx; ~25 fewer Verilator builds.

### `--config2` (120m → ~70m)
The cost is **22 opae/xrt emulation runs**, many duplicated across both drivers
(every `mstress` mem-config runs on opae *and* xrt).
- Collapse opae+xrt duplicates to **one emulation driver per config** (≈ −9 runs).
- Move part of the mem-config coverage (block size, banks, coalescing, ports) to
  **rtlsim/simx**; keep a thin opae + xrt *smoke* so both AFU shims stay exercised.
- The group stays a **single `--config2`** (driver/extension/startup-addr tests
  plus the memory-subsystem matrix); under the 2-hour ceiling it does not need a
  split.

### `--dxa` (120m → ~45m)
Today: `dxa_copy` 1D–5D on **both simx and rtlsim with `--debug=3`** (10 builds,
tracing overhead), plus mcast variants; 27 rtlsim builds, 3 at NT=32.
- `--debug=3` is for trace validation — keep it on **one** dim, drop it from the rest.
- Reduce the 1D–5D sweep to representative dims (e.g. 1D + 3D + 5D) on **both**
  drivers — the redundant intermediate dims add little RTL coverage.
- Drop NT=32 rtlsim to a single point.

### `--config1` (56m → ~30m)
30 rtlsim micro-config builds (warp/thread/core/issue/simd/ALU/FPU/LSU scaling).
- For each scaling axis keep the **endpoints (min + max)** and drop redundant
  intermediate points, on **both** drivers — endpoints exercise the parameterized
  RTL; the middle values rarely add a distinct path.
- The `NT=32` sgemm rtlsim builds (SIMD-width pair) → keep one.

### `--cache` (40m → ~25m)
21 rtlsim builds. Keep the cache-mode **endpoints** (writeback on/off, 1 vs N
banks, L2/L3 present/absent) on both drivers; drop redundant intermediate
parameter values that don't change the cache datapath.

### `--regression` (60m → ~40m)
Only 11 runs but **4 are opae/xrt**. Collapse opae+xrt duplicates to one
emulation smoke; the functional regression already runs on simx + rtlsim.

### `--tensor_wg` (60m → ~30m)
10 rtlsim builds, 2 at NT=32. Same treatment as `tensor_sp`: sweep **sizes**
`{int4, int8, fp16, tf32}` not formats, full coverage on simx, a small rtlsim
parity subset, at most one NT=32 rtlsim point.

## Expected outcome

| Group | Now | Target |
|-------|-----|--------|
| `--cache`      | 40m  | ~25m |
| `--config1`    | 56m  | ~30m |
| `--config2`    | 120m | ~70m (opae/xrt deduped, single group) |
| `--dxa`        | 120m | ~45m |
| `--regression` | 60m  | ~40m |
| `--tensor_sp`  | 120m | ~50m |
| `--tensor_wg`  | 60m  | ~55m (light) |

Every group then lands comfortably **under the 2-hour ceiling**.

## Final step — tighten the CI job timeout (after the cuts land)

Once every group is verified comfortably under budget, hold the per-job ceiling
at **2 hours** so a future regression in test time is caught while leaving margin
for the slowest groups and runner variance:

- `.github/workflows/ci.yml` — `tests` job `timeout-minutes: 120`
- `.github/workflows/apptainer-ci.yml` — `tests` job `timeout-minutes: 150` → **120**

(Do this only *after* the trims are merged and green; flipping the ceiling first
would just fail the still-long jobs.)

## What is explicitly NOT lost

- Every operand **size**, dimension, and config **endpoint** keeps coverage on
  **rtlsim** (the ground truth), with simx running the same matrix.
- The dropped points are redundant *format* variants of an already-covered size,
  the NT × size cross-product, intermediate sweep values, and duplicate
  emulation-driver (`opae`+`xrt`) runs — not unique RTL paths.
- Both `opae` and `xrt` AFU shims keep a smoke run.

## Longer-term lever (separate work)

The root inefficiency is that datatypes (`ITYPE`/`OTYPE`) and many knobs are
**compile-time `-D` macros**, so every point pays a full rebuild. Making the TCU
datatypes **runtime-selectable** would let one simulator build cover all datatypes
across many runs — turning the tensor sweeps from build-bound into run-bound and
collapsing their time by an order of magnitude. Proposed as a follow-up.

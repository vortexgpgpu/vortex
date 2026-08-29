# Vortex Continuous Integration — Catalog-Driven, Driver-Sliceable Test Architecture

Vortex tests are **declarative data** run by **pytest**, replacing the imperative,
driver-pinned bash that used to live in `ci/regression.sh`. `blackbox.sh` stays the
unchanged executor. `ci/regression.sh` is now slim and serves two roles: the **local
entry point** into the catalog (`--all` / `--test <selector>`, thin wrappers over
`pytest [-m …] ci`) and the **host/multi-step backend** for the four categories that
don't fit the common shape (`dtm`, `sst`, `gem5`, `cupbop`), which the catalog's
`via: script` cases reach through the internal `--run <flow>`. This document covers both
halves: the **engine** (test cases + pytest harness) and the **workflow** (GitHub fan-out
+ planner).

---

## 1. Problem

`ci/regression.sh.in` *was* the core CI engine: ~1400 lines of imperative bash, ~30
category functions, **401 driver-pinned invocations** in three execution styles —
326 `./ci/blackbox.sh --driver=<d> …`, 75 `make -C … run-<d>`, and 16 standalone
`make -C sim/<d>` builds. The driver (`simx`/`rtlsim`/`xrtsim`/`opaesim`) is hard-coded
into every line. That single fact is the root of every limitation:

| # | Pathology | Consequence |
|---|-----------|-------------|
| P1 | Driver baked into every line | Can't run "simx only" without editing 401 lines — yet `rtlsim` (~168 runs, the Verilator long pole) dominates cost. |
| P2 | Three execution styles, no single seam | No one place to filter, time, or report a test. |
| P3 | Tests are code, not data | Coverage is unqueryable; path-selection must be a hand-kept regex; no per-test report. |
| P4 | Build/run entangled | A `(driver, CONFIGS)` sim is re-elaborated whenever the config changes between adjacent lines (388 cases span 239 distinct sim builds; e.g. riscv's 11 cases rebuild a sim that 2 builds would cover). |
| P5 | `CONFIGS` as repeated env prefixes | The documented footgun: app and driver must be built with matching `CONFIGS` or results are silently wrong. |
| P6 | Category = coarse unit | One function = one CI job = all its drivers + configs, serially. |
| P7 | No metadata | No `tier`, `needs`, `touches`, `xlen`-applicability per test. |
| P8 | `set -e` fragility | Inline driver-gating must dance around errexit. |

Every attempt to *retrofit* a driver filter onto this (a blackbox gate, a `run_test`
wrapper, `make run-` guards, `set -e` workarounds) fights P1–P3. Making tests **data**
dissolves the problem: the slice becomes a query.

---

## 2. Model: a test case is a point in an N-dimensional space

Axes that today are flattened into one bash line, kept explicit so any one can become a
*filter* or a *matrix dimension*:

```
category   amo, cache, tensor, graphics, …
driver     simx | rtlsim | xrtsim | opaesim          (cost axis)
xlen       32 | 64                                    (build-tree axis)
config     CONFIGS="-DVX_CFG_…"                        (rebuild axis)
shape      cores/warps/threads/l2/l3, args
tier       smoke | full | nightly                     (when-to-run axis)
needs      (none) | mpi | sst | gem5                  (env axis)
touches    source paths this case exercises           (selection axis)
```

---

## 3. Engine

Three things we own — **test-case data + thin pytest glue + the unchanged executor**;
everything else (selection, parallelism, reporting) is pytest.

```
   ci/testcases/*.yaml        markers/-m, -k, --changed     ┌──────────────┐
   (data: cases)  ───────────────────────────────────────▶ │    pytest    │
                     testcase.py + conftest.py + test_runner.py  (the runner)│
                                                            └──────┬───────┘
                                              fixture: build│once per build-key
                                                       ┌────▼────────┐  run many
                                                       │  executor   │  per case
                                                       │ blackbox.sh │  (UNCHANGED)
                                                       └────┬────────┘
                                            --junitxml ┌────▼────────┐
                                                       │  reporter   │ → GitHub test report
                                                       └─────────────┘
```

### 3.1 Test cases (`ci/testcases/<category>.yaml`)

One file per category; fields map 1:1 to existing `blackbox.sh` flags, so it is a
faithful transcription, not a reinterpretation. An entry with `drivers: [...]` expands
to one case per driver. `xlen` is an **outer** dimension — a collection-time filter
against the ambient build tree, never expanded here (build32/ and build64/ are separate
trees).

```yaml
category: amo
defaults:
  configs: "-DVX_CFG_EXT_A_ENABLE"
  xlen: [32, 64]
  tier: smoke
  touches: [hw/rtl/cache, sim/simx/amo, sim/simx/mem]
tests:
  - id: base
    app: amo
    drivers: [simx, rtlsim]                 # -> 2 cases

  - id: wb-dirtybytes1
    app: amo
    drivers: [rtlsim]
    configs+: "-DVX_CFG_DCACHE_WRITEBACK=1 -DVX_CFG_DCACHE_DIRTYBYTES=1 -DVX_CFG_DCACHE_NUM_WAYS=4"

  - id: mc-l3
    app: amo
    drivers: [simx]
    configs+: "-DVX_CFG_L2_WRITEBACK=0"
    shape: {cores: 4, l2cache: true, l3cache: true}
    args: "-n8"
    tier: full
```

The three execution styles collapse into one `via` field:

```yaml
  # make-run (riscv ISA, vulkan, hip, rvc, vm). {driver}/{xlen} are substituted.
  - {id: isa, via: make-run, dir: tests/riscv/isa, target: "run-{driver}-{xlen}a", drivers: [simx, rtlsim]}

  # script — the host/python categories (unittest, synthesis, vector, dtm, sst, gem5,
  # cupbop). Driverless cases self-build. `needs:` records the env a cell must
  # provision (it drives the workflow profile); it does NOT skip — a missing dep fails.
  - {id: legacy, via: script, run: "./ci/regression.sh --sst", needs: [sst]}
```

`configs` overrides the default; `configs+` appends. Metadata absent today —
`tier`/`needs`/`touches`/`xlen` — is first-class.

### 3.2 The runner is pytest, not a hand-rolled engine

A test runner — load, select, run, report — is a solved problem, so we **adopt the
industry standard** (`ctest` is ruled out: it is CMake's, and Vortex is GNU-Make-only).
pytest supplies the machinery; we write three small files of glue and **no config
file** — pytest's own conventions (a `conftest.py`, `test_`-prefixed test module,
markers) carry it.

| Need | pytest mechanism |
|------|------------------|
| case → test matrix | `pytest_generate_tests` parametrizes from the data |
| selection (driver/tier/category) | one **marker per value** + `-m "cache and simx and smoke"` |
| build-once-run-many | a fixture scoped to the `(driver, CONFIGS)` build-key |
| report | `--junitxml` (the universal CI interchange format) |
| parallelism | across GitHub matrix cells (serial within a cell — see §6) |
| dry-run "what would run" | `--collect-only` |

**Three files, all in `ci/`** — the conventional pytest layout (support module +
`conftest.py` + test module):
- `ci/testcase.py` — the `Spec` model + loaders + the planner **CLI** (`lint`/`matrix`/
  `select`). No pytest dependency, so the lightweight plan job imports it freely.
- `ci/conftest.py` — the hooks/fixtures: `pytest_configure` registers **markers derived
  from the data** (so adding a category/driver needs no edit, and `--strict-markers`
  catches `-m` typos), `pytest_generate_tests` parametrizes + applies one marker per
  value + the ambient-XLEN filter, and the `sim_build` fixture builds each
  `(driver,CONFIGS)` once (the P4 fix). Cases run **serially within a cell** — the
  parallelism is across GitHub matrix cells, each its own build tree — so successive
  `CONFIGS` never clobber a `sim/` build that is still in use (see §6).
- `ci/test_runner.py` — the single `test_case` that shells out to `blackbox.sh`/`make`
  and asserts a clean exit. Every failure (and every build warning escalated to an
  error) is a real, red failure — except a case carrying a `known_issue:` reason in
  the catalog, which `conftest.py` turns into a tracked `xfail`: it still builds and
  runs, but its failure is expected and does not fail CI (an unexpected pass surfaces
  as `XPASS`). Reserve it for triaged, documented breakage.

No `pyproject.toml`/`pytest.ini`: markers register dynamically in `conftest.py`,
`test_runner.py` is auto-discovered by the `test_` prefix, and the run passes `ci` as the
path. `blackbox.sh` is untouched; `regression.sh` is reduced to the four host/multi-step
backends (§5).

Selection is idiomatic pytest:

```
VX_XLEN=32 pytest ci -m "cache and simx and smoke" --strict-markers
pytest ci --collect-only -q -m "simx"      # dry-run
```

### 3.3 Cross-driver checks (`check: model_parity`)

SimX is the timing model of the RTL, not just a functional oracle. A case with
`check: model_parity` validates that: it is **not** driver-expanded — the runner
executes the same app/args/configs on **simx and rtlsim** as two legs of one case
(pinned to the rtlsim driver for build/matrix placement, since it elaborates the
RTL) and compares the runtime's final `PERF: instrs=…, cycles=…` summary:

- **instrs must match exactly** — both drivers are deterministic ISA-level
  executions, so any delta is functional divergence, not a timing gap;
- **cycles must agree within `tolerance`** (default 5%, per-case override).

Every case also prints a `PARITY:` line with both counts and the measured gap, so
green runs still leave a trend trail in the logs. The general-pipeline matrix
(vecadd, sgemm) lives in `ci/testcases/core.yaml`; each extension
(tensor*, raytracing, graphics TEX/RASTER/OM, dxa) carries its own
`model_parity-*` case in its category file, with only that extension enabled so a
regression is attributable. Workloads are sized so steady state dominates
(>=~300k cycles for the pipeline cases) — a tiny kernel is all boot/dispatch skew
and makes the gap ratio noisy. A `model_parity` marker selects them all:
`pytest ci -m model_parity`. Use `known_issue:` (not a loosened tolerance) for a
tracked gap under investigation.

**A check is a marker, never a file or a category.** `model_parity` gets a
**dedicated cell** — `-m "model_parity and rtlsim"` sweeps *every* parity case
catalog-wide, one centralized simx↔RTL gate — and a parity case never
double-runs, because each category cell excludes the check markers (`… and not
model_parity and not perf_gate`) and the check cell owns them. It runs at
**`full`** tier (rtlsim-heavy → PR + nightly).

That cell is emitted by the planner **from the check itself** (`cmd_matrix`:
`name = c.check or c.category`), not as a side effect of some category being
*named* after the check. It used to be the latter — so renaming that category
silently deleted the cell, taking every `check:` case in the catalog with it, and
the gate evaporated **green**. Two lint rules now hold the line: a file's name
must equal its `category:`, and a file may not be named after a check. The
workflow reads the check list from `testcase.py checks` rather than hardcoding
it, so the same knowledge does not live in two places.

### 3.4 Perf-regression checks (`check: perf_gate`)

Same gating shape as §3.3 (its own dedicated cell driven by the `perf_gate`
marker, `full` tier, rtlsim-pinned) but a different assertion: instead of
comparing SimX vs RTL, it compares **this commit's rtlsim cycles against a
checked-in golden baseline** within ±2% (`ci/perf_baseline.py`). Because rtlsim
cycle counts are deterministic and host-independent, there is no noise to handle —
the threshold only absorbs benign, intended micro-changes.

- **Baselines** live in the source tree at `ci/baselines/perf/<category>.json`
  (canonical sorted JSON, one file per category). Each entry stores the measured
  `cycles`/`instrs` per xlen, plus a `config_hash` (of app/args/configs/shape)
  and the workload's `instrs` as **staleness guards**: if the run config changes
  (`config_hash` mismatch) or the workload changes (`instrs` mismatch), the check
  errors "regenerate" instead of comparing stale numbers.
- **Direction**: cycles above baseline by >tolerance = **regression** (hard fail);
  cycles below by >tolerance = an unlocked **improvement** — also fails, asking
  you to update the baseline so the gain is ratcheted in and a later silent
  regression back toward the old number is still caught.
- **Updating** is script-generated + human-reviewed, never done by CI:
  `pytest ci -m perf_gate --update-baselines` (a `conftest.py` option that
  flips `_perf_gate` from assert- to record-mode and flushes on session
  finish). A human runs it only for an intended perf change, reviews the JSON
  diff (`cycles: 999027 → 918400` = an explicit, reviewable perf delta), and
  commits. **CI must never pass `--update-baselines`** — an auto-updated baseline
  would silently absorb every regression. Same discipline as a golden image.
- Benchmarks **reuse the steady-state model_parity workloads** (base pipeline in
  `ci/testcases/core.yaml`, alongside their parity twins; extensions as
  `perf_gate-*` cases in their category files) — one run, its own gate. A case
  carries exactly one check, so the parity and perf views of the same workload
  are separate cases: perf ids stay bare (`sgemm` — the golden baseline is keyed
  by it, `core:sgemm:rtlsim`), parity twins are prefixed `parity-`.

### 3.5 Synthesis-regression checks (`fpga_gate`)

The perf_gate catches a change that costs *cycles*. The fpga_gate catches one
that costs *timing closure or area*: it synthesizes a catalog of DUTs with
Vivado and asserts the post-implementation **Fmax** and **LUT** count against a
checked-in golden baseline within ±5% (`ci/fpga_gate.py`). Same discipline as
§3.4 — regression fails, an unlocked improvement also fails and asks you to
record it, and CI never writes a baseline.

It is **not** a pytest cell. A cell is a build tree plus a driver; an fpga_gate
build is an hours-long Vivado run that only exists on a host with the licensed
toolchain, so it is a standalone script driven by its own workflow on the
self-hosted runner (§4.4). `asic_gate` — the same script over
`ci/baselines/synthesis/yosys/` — is the planned second tool.

- **Spec and baseline are split**, exactly as everywhere else in the catalog.
  The spec is `ci/testcases/fpga_gate.yaml` — hand-authored, commented,
  reviewed: per build a DUT target, a target clock, a `CONFIGS` string, an
  optional `known_issue`/`thresholds`, and a `group`. The goldens are
  `ci/baselines/synthesis/xilinx/<group>.json` — machine-written, never
  hand-edited, carrying only measured metrics plus the config fingerprint and
  tool env they were measured under. Groups: `core` (cache+AMO, wide core, full
  4-core AFU), `tensor` (all-datatype TCU), `graphics` (RTU/RASTER/OM/TEX),
  `dxa`. A `config_hash` (dut/clock/configs/device/opt-level/xlen) ties the two
  together — edit the spec and the gate refuses to compare against numbers
  recorded for the old one (`STALE`).
- **Tier `fpga` is opt-in.** An empty `--tier` means "everything", and
  everything is what a *hosted* runner can run — which these cannot: they need a
  licensed Vivado and hours of a whole machine. `OPT_IN_TIERS` in `testcase.py`
  keeps them out of every hosted event (including the nightly) unless asked for
  by name, so they run only on the self-hosted runner's own workflow (§4.4).
- **Metrics** all come from `synth_summary.csv`, which `hw/syn/xilinx/dut/
  project.tcl` already emits post-implementation: Fmax, WNS, LUT, LUTRAM, FF,
  BRAM, URAM, DSP — plus the wall-clock build time the runner measures. Every
  metric is recorded and reported; `--gate` picks which ones are *asserted*
  (Fmax and LUT by default). Build time is bookkeeping, not a gate: it is too
  host-dependent to assert, and it is what the scheduler orders the queue by.
- **Critical paths**: each build also records its **top 10 unique critical
  paths** (slack, logic levels, clock group, startpoint, endpoint) — emitted by
  `project.tcl` whether or not timing closed, because a design that *meets* its
  target still has a worst path, and watching where it sits across commits is
  what turns a Fmax regression from a number into a location. Never gated;
  `-unique_pins` keeps the list 10 distinct paths rather than 10 views of one.
- **Thresholds** resolve most-specific-first: a build's `"thresholds": {"lut":
  0.10}` beats `--metric-threshold lut=0.10` (global, per-metric), which beats
  `--threshold` (global, all metrics, default 5%). Same shape as `model_parity`'s
  `tolerance` (per-case → category `defaults:` → `DEFAULT_PARITY_TOLERANCE`).
  Note `perf_gate` (§3.4) does **not** have this — it reads one hardcoded
  `TOLERANCE` constant, with no per-case override.
- **`known_issue`**: a build carrying a reason string is a tracked expected
  failure — it still builds, still reports, its numbers still land in the table,
  but its verdict does not fail the run. Same contract as a `known_issue:` test
  case (which conftest marks `xfail(strict=False)`), including that a known issue
  which stops reproducing surfaces as **XPASS** — reported loudly, asking you to
  clear the flag, but not converted into a hard failure.
- **Early-failure watch**: a config mistake — a bad define, a missing source, a
  parameter or hierarchy error — kills a build during *RTL elaboration*, seconds
  into an otherwise multi-hour Vivado run. The runner follows each build's log
  live and announces `Finished RTL Elaboration` as it lands, so a typo surfaces
  in minutes instead of at the end of the sweep; a build that dies before that
  point is reported as `FAILED BEFORE SYNTHESIS` with its Vivado `ERROR:` lines
  quoted inline, not as a generic non-zero make.
- **Resumable sessions**: a sweep is hours long, so an interrupted one is picked
  back up rather than restarted. Each build dir carries a stamp
  (`fpga_gate.json`: config hash + status + metrics), so `--resume` reuses a
  build already finished *for this config*, lets an unfinished one pick up from
  its Vivado post-synth/post-impl checkpoint, and runs the rest. A build whose
  config changed since its stamp is rebuilt clean — resuming from those
  checkpoints would silently re-synthesize the old design. State lives next to
  the build tree it describes, not in a central session file, so it survives a
  kill and never desynchronizes from what is on disk.
- **Progress**: each build reports Vivado phase transitions (setup →
  elaboration → synthesis → opt → placement → routing → reporting) as they
  land, with a heartbeat in between; `-v` streams the raw Vivado log instead.
- **Scheduling** is longest-processing-time-first over the recorded build times:
  the longest build is dispatched first so it is in flight from t=0, and the
  remaining slots churn through the short ones behind it. `-j` caps parallel
  builds (2 on the runner) and each build's Vivado job count is derived from it
  so the machine is not oversubscribed. Every build gets a unique `PREFIX`
  (`fpga_gate_<id>`), so it has its own build tree and log and cannot collide
  with a parallel build or with a hand-run synthesis on the same machine.
- **Updating** — `ci/fpga_gate.py --update-baseline`, human-reviewed, committed
  as an explicit `Fmax: 312 → 287` diff. Baselines also record the Vivado
  version they were measured on; a run under a different version warns, because
  Fmax across tool versions is not comparable.

---

## 4. Workflow

### 4.1 `ci.yml` — catalog-driven

`plan` reads the data (via `testcase.py matrix`, no build env) and emits the
`(category × driver × xlen)` cell list for this event; each cell runs
`pytest ci -m "<category> and <driver>"` in its build tree, emits JUnit, and the run is
gated by a single `complete` job.

```
plan:  event × driver-policy × tier × (touches[] ∩ diff)  ->  cells JSON
setup: warm toolchain + third_party caches once (setup-vortex prepare=true)
build: one build tree per xlen, needs setup (restores the warmed caches)
tests: matrix = cells  ->  pytest ci -m "<cat> and <driver>" per cell  ->  JUnit
complete: single green gate (needs plan+setup+build+tests)
```

`setup` exists so a cold cache prepares the toolchain (a prebuilt-tarball download)
and third_party **once**, not once per xlen: the two `build` jobs `needs: setup` and
only restore. On a cache hit it is a fast no-op.

Driver/tier policy by event:

| Trigger | Drivers | Tier |
|---------|---------|------|
| push | simx | smoke |
| pull_request | simx, rtlsim | smoke,full |
| schedule (nightly/weekly) | all | all |
| workflow_dispatch | (inputs) | (inputs) |

This is the whole point: a push runs `simx` (the cheap, high-signal driver) and defers
the ~168 `rtlsim` runs to PR-gate/nightly — `--drivers=simx` is now just `-m "simx"`.

### 4.2 `setup-vortex` composite action

The cache/deps boilerplate (`read-version-pins + cache toolchain + cache third-party +
install deps + pip`) is one local composite action, parameterized by `profile`
(lite/full), used by every job. A `prepare` input (true only in the `setup` job) makes
it additionally **populate** the caches on a miss — building the toolchain + third_party
once — so build/test jobs (`prepare: false`) only ever restore. Prep logic lives in the
action, not duplicated across jobs.

### 4.3 `apptainer-ci.yml` — share setup, not orchestration

The Apptainer flow validates the build/test works **inside the `vortex.sif` container** —
an *environmental* signal, not functional coverage the host run already provides.
It is deliberately **not** folded into `ci.yml` (different intent → the wrong
abstraction). It stays a separate, minimal workflow that:
- **reuses** the `setup-vortex` composite action (genuinely identical host-side work), and
- runs a representative `pytest ci -m "regression and simx"` slice inside the container, not the
  full matrix.

Recommended triggers: weekly **offset** from the host weekly (so a failure is attributable
to the container, not the code) plus `paths:` on the container-definition files.

### 4.4 `fpga_gate.yml` — nightly synthesis on the self-hosted runner

The §3.5 gate needs a licensed Vivado and half a machine for hours, so it cannot
be a cell in `ci.yml`'s hosted matrix. It is its own nightly workflow on the
self-hosted runner, and it **hard-pins `origin/master`** — whatever branch the
schedule fires on, the thing being gated is master's head.

Because the sweep is expensive, it **skips itself when master has not moved**:
the runner keeps the last gated SHA in `~/.cache/vortex/fpga_gate.<repo>.sha`
and the run is a no-op when it matches (`force: true` on `workflow_dispatch`
overrides). The SHA is recorded once the gate reaches a *verdict* — pass or
regression — so a red master is not re-synthesized every night (the failed run
is the record); an infra/build error does not record, so the next nightly
retries it.

---

## 5. Migration — done

The catalog is now the **single source of truth**; `ci/regression.sh.in` no longer
duplicates any cataloged test. Final shape:

- **25 categories native** in the catalog (`via: blackbox`/`make-run`/`script`), including
  the host categories `unittest`, `synthesis`, `vector` (self-contained `via: script` that
  call `make` directly — no `regression.sh`).
- **MX coverage is first-class:** `tensor_mx` (transcribed from the legacy `tensor_mx()`,
  incl. a `-DTCU_MX_TLS` tensor-level-scale variant) and `tensor_sp_mx` are catalog
  categories. `TCU_MX_TLS` is a **sw/test macro** passed per-case in `configs`, **not** a
  hardware `VX_CFG_` knob (it is not in `VX_config.toml`).
- **4 host/multi-step categories** (`dtm`, `sst`, `gem5`, `cupbop`) stay `via: script`
  delegating to `./ci/regression.sh --<cat>`. These are genuinely multi-step host flows
  (special builds: `USE_SST=1`/`USE_GEM5=1`, the gem5 ARM matrix, a cupbop download) that
  don't fit the common shape, so `regression.sh.in` is **kept on purpose** — slimmed from
  ~1400 lines to ~320, holding only those four functions. This is the documented
  steady state, not a pending deletion.
- **440 test cases / 31 categories**; `ci/testcase.py lint` + `pytest --collect-only` clean.
- Local runs go through `ci/regression.sh` against the **same** catalog CI runs, so
  they can never drift from it: `./ci/regression.sh --all` runs every category for the
  build tree's XLEN, and `./ci/regression.sh --test "<selector>"` runs a slice, where
  `<selector>` is a pytest marker expression — a category (`tensor`), a driver
  (`rtlsim`), or a combo (`"tensor and simx"`). Both wrap `pytest [-m …] ci`.

Real per-category sim execution runs on CI, not locally.

### 5.1 Graphics / Vulkan fixed-function coverage

The graphics stack is swept along the **fixed-function (FF) axis** — each of TEX /
RASTER / OM either present in hardware or emulated in SIMT software — because the
FF↔SIMT boundary is the recurring graphics bug surface:

- **Native FF units** (`graphics`): `gfx_tex` / `gfx_raster` / `gfx_om` each drive
  one FF unit; `gfx_draw3d` drives all three (TEX+RASTER+OM).
- **Early-Z** (`graphics`, `gfx_earlyz-*`): `gfx_draw3d` built with
  `-DVX_CFG_RASTER_EARLYZ` — the opt-in occlusion-cull knob (legal only with
  OM+RASTER; off by default → raster byte-identical). Bit-identical to the
  no-early-Z golden on box@128 / evilskull@32 (simx + rtlsim); evilskull@128
  carries a tracked 2-px in-flight-write residual (`known_issue`).
- **FF/SW mix + full software emulation** (`vulkan`, `ff-*`): the vortexpipe 3D
  pipeline (`draw3d`) with per-unit drop knobs — `NO_TEX` / `NO_RASTER` / `NO_OM`
  (`tests/vulkan/common.mk`) — sweeps every FF/SW combination
  (`ff-raster-tex-om` … `ff-om-only`, and `ff-all-sw` = whole pipeline in SIMT
  software). vortexpipe routes each dropped unit to the `gfx_*_sw` ABI, validated
  against the same golden as the all-hardware path. Combos that need SW-TEX or
  SW-RASTER routing are `known_issue` until vortexpipe wires those stages (the SW
  ABI already exists).
- **Ray tracing + hybrid** (`vulkan`): `rt-raytrace` / `rt-rtquery` (VK_KHR_ray_query
  benchmarks on the PRISM RTU), `gfx-rt-rtquery-id` (rasterized fragment shader +
  inline ray query — gfx + RT in one frame), and `gfx-multidraw` (multi-drawcall
  rasterization). The blanket `vulkan` `isa-*` cases still run the whole suite per
  driver; these track the individual scenarios and add the FF/SW-mix builds the
  suite does not produce.

New cases in these yaml files are auto-included in the plan (§4.1) — no workflow
edit needed. `known_issue` cases still build and run (surfacing an `XPASS` when the
underlying support lands), so aspirational coverage is tracked, not silently absent.

---

## 6. Risks & mitigations

- **Build-dedup vs. parallelism (resolved).** Successive `CONFIGS` build into the same
  `sim/` output, so building different keys concurrently in one tree clobbers them — this
  was confirmed on the first CI run (an in-tree `pytest -n auto` raced multiple Verilator
  builds and most cells errored). Resolution: **cases run serially within a cell**, and
  parallelism is taken **across GitHub matrix cells** (each cell is its own runner +
  build tree). A future intra-cell speedup would need per-worker isolated trees
  (`git worktree`); not worth it now.
- **Data drifts from reality.** `testcase.py lint` runs in CI; each migrated category is
  parity-diffed against its legacy function once before the bash is deleted.
- **Script categories.** `via: script` lets the special categories delegate to legacy
  without forcing them into the common shape; their `needs` deps must be provisioned by
  the workflow (a missing dep is a real failure — nothing is skipped).
- **pytest dependency.** Industry standard; Python is already on the CI path; pinned like
  any dev tool. The harness only orchestrates and shells out.
- **Catalog format: YAML** — list-of-records ergonomics, PyYAML/Actions-native; TOML is
  awkward for record arrays, JSON loses comments.

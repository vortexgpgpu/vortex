# Vortex Continuous Integration — Catalog-Driven, Driver-Sliceable Test Architecture

Vortex tests are **declarative data** run by **pytest**, replacing the imperative,
driver-pinned bash that used to live in `ci/regression.sh`. `blackbox.sh` stays the
unchanged executor. `ci/regression.sh` survives only as the slim **host/multi-step
backend** for the four categories that don't fit the common shape (`dtm`, `sst`, `gem5`,
`cupbop`), invoked through the catalog's `via: script`. This document covers both halves:
the **engine** (test cases + pytest harness) and the **workflow** (GitHub fan-out + planner).

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
tier       smoke | fast | slow | nightly              (when-to-run axis)
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
  tier: fast
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
    tier: slow
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
| selection (driver/tier/category) | one **marker per value** + `-m "cache and simx and fast"` |
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
VX_XLEN=32 pytest ci -m "cache and simx and fast" --strict-markers
pytest ci --collect-only -q -m "simx"      # dry-run
```

---

## 4. Workflow

### 4.1 `ci.yml` — catalog-driven

`plan` reads the data (via `testcase.py matrix`, no build env) and emits the
`(category × driver × xlen)` cell list for this event; each cell runs
`pytest ci -m "<category> and <driver>"` in its build tree, emits JUnit, and the run is
gated by a single `complete` job.

```
plan:  event × driver-policy × tier × (touches[] ∩ diff)  ->  cells JSON
build: one build tree per xlen (composite setup-vortex)
tests: matrix = cells  ->  pytest ci -m "<cat> and <driver>" per cell  ->  JUnit
complete: single green gate
```

Driver/tier policy by event:

| Trigger | Drivers | Tier |
|---------|---------|------|
| push | simx | smoke,fast |
| pull_request | simx, rtlsim | smoke,fast,slow |
| schedule (nightly/weekly) | all | all |
| workflow_dispatch | (inputs) | (inputs) |

This is the whole point: a push runs `simx` (the cheap, high-signal driver) and defers
the ~168 `rtlsim` runs to PR-gate/nightly — `--drivers=simx` is now just `-m "simx"`.

### 4.2 `setup-vortex` composite action

The cache/deps boilerplate (`read-version-pins + cache toolchain + cache third-party +
install deps + pip`) lived in three copies across `setup`/`build`/`tests`. It is now one
local composite action, parameterized by `profile` (lite/full), used by every job.

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
- The local full-suite runners `ci_xlen{32,64}.sh` drive the **same** catalog
  (`pytest -m "<category>"` per category, categories discovered from `ci/testcases/`), so
  they can never drift from CI.

Real per-category sim execution runs on CI, not locally.

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

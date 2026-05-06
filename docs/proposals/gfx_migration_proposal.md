**Date:** 2026-05-01
**Status:** Phases 0/1/2 complete; Phase 4 partially complete (smokes); Phase 3/5 deferred
**Author:** Blaise Tine
**Related:**
[simx_v3_proposal.md](simx_v3_proposal.md),
[dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md),
[wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md).

### Update history

- **2026-05-02** — Phases 0–2 complete + Phase 4 smoke milestone reached.
  - All three graphics extensions (TEX, OM, RASTER) elaborate clean and
    pass per-unit verilator smoke unittests (`hw/unittest/{tex,om,raster}_unit/`).
  - Cluster integration via [hw/rtl/VX_graphics.sv](../../hw/rtl/VX_graphics.sv)
    (1 unit + tcache/rcache/ocache cluster + DCR fan-out) ties into
    [VX_cluster.sv](../../hw/rtl/VX_cluster.sv) with new
    `per_socket_*_bus_if [NUM_SOCKETS]` arrays. `L2_NUM_REQS` extended
    by `L2_GFX_REQS = EXT_TEX_ENABLED + EXT_RASTER_ENABLED + EXT_OM_ENABLED`
    for graphics cache mem ports.
  - SFU PE switch extended in [VX_sfu_unit.sv](../../hw/rtl/core/VX_sfu_unit.sv):
    `PE_COUNT = 2 + EXT_DXA_ENABLED + EXT_TEX_ENABLED + EXT_OM_ENABLED + EXT_RASTER_ENABLED`.
    All three agents rewritten against the feature_gfx `sfu_execute_t`/
    `sfu_result_t` `header` convention.
  - Decode entries for `vx_tex` / `vx_om` / `vx_rast` added in
    [VX_decode.sv](../../hw/rtl/core/VX_decode.sv) under `INST_EXT2`
    (CUSTOM1 = 0x2B), funct3 = 1/2/3, with new
    `INST_SFU_TEX/OM/RASTER` op-types.
  - Three end-to-end smoke tests pass on rtlsim:
    [tests/regression/tex_smoke/](../../tests/regression/tex_smoke/) (1×1
    ARGB texel readback), [raster_smoke/](../../tests/regression/raster_smoke/)
    (descriptor pop with `done=1`), [om_smoke/](../../tests/regression/om_smoke/)
    (per-pixel passthrough write through full blend pipeline).
  - Wired into [ci/regression.sh.in](../../ci/regression.sh.in) as the
    new `--gfx` suite; ~1.5 min wall time.
  - SimX (Phase 3) and full PNG-driven regression suites (Phase 4 §4)
    remain deferred per original plan.

# Skybox → feature_gfx Graphics Migration — Proposal

## 1. Summary

Port the skybox-era graphics extensions (TEX, RASTER, OM, plus their
caches, DCR plumbing, runtime hooks, and regression tests) from
`~/dev/skybox` (`develop` @ 4a6636f2) into the current `feature_gfx`
branch (`tinebp-patch-2`).

Skybox was built on a much older Vortex generation. Both the **RTL
socket/cluster topology** and the **SimX simulator** have been
substantially redesigned since (TLM-aligned SimX v3, new SFU PE-switch,
KMU/CTA dispatcher, lane-dispatch/lane-gather rename, DCR arbitration,
TCU + DXA precedents). A naïve copy-paste will not compile and would
violate v3 invariants. This proposal lays out the migration as **six
sequential phases**, each independently buildable and runnable, with
explicit alignment points where v3 rules force a re-design rather than
a port.

The new code lands under `EXT_TEX_ENABLE`, `EXT_RASTER_ENABLE`, and
`EXT_OM_ENABLE` extension flags (default `false`), gated by config and
disabled by default; baseline (no graphics) keeps building and running
unchanged.

All build/test work is done from `build_test32/` (xlen=32). Source-tree
edits land in `feature_gfx/`.

---

## 2. What we are migrating

### 2.1 Source-of-truth inventory (skybox)

| Area | Path | LOC | Notes |
|------|------|----:|-------|
| RTL: TEX | [skybox/hw/rtl/tex/](../../../skybox/hw/rtl/tex/) (18 files) | ~1500 | sampler, address, format, lerp, wrap, sat, stride, mem, csr, dcr, perf_if, agent, arb, bus_if, unit, unit_top |
| RTL: RASTER | [skybox/hw/rtl/raster/](../../../skybox/hw/rtl/raster/) (17 files) | ~2700 | te (tile engine), be (block engine), qe (quad engine), edge, extents, slice, mem, csr, dcr, perf_if, agent, arb, bus_if, unit, unit_top |
| RTL: OM | [skybox/hw/rtl/om/](../../../skybox/hw/rtl/om/) (19 files) | ~2400 | blend (func/minmax/multadd), compare, ds, logic_op, stencil_op, mem, csr, dcr, perf_if, agent, arb, bus_if, unit, unit_top |
| RTL: glue | [skybox/hw/rtl/VX_graphics.sv](../../../skybox/hw/rtl/VX_graphics.sv) | 364 | Cluster-level wrapper that instantiates {RASTER, TEX, OM} units, their caches (rcache/tcache/ocache), and DCR fan-out |
| RTL: header | [skybox/hw/rtl/VX_types.vh](../../../skybox/hw/rtl/VX_types.vh) | (DCR/CSR ranges + ISA_EXT_{TEX,RASTER,OM} bits) | Source for the `VX_DCR_*` and `ISA_EXT_*` definitions we must port |
| SimX | [skybox/sim/simx/{tex_core,raster_core,om_core}.{cpp,h}](../../../skybox/sim/simx/) | ~1370 total | Old procedural model; **see §4.2 — full rewrite required** |
| SimX common | [skybox/sim/common/{gfxutil,graphics}.{cpp,h}](../../../skybox/sim/common/) | (3D math, frag pipeline, CSR conversion helpers) | Used by both simulators and tests |
| Third party | [skybox/third_party/cocogfx/](../../../skybox/third_party/cocogfx/) | (PNG/format/lerp helpers built as `libcocogfx.a`) | Required for image I/O in tests |
| Runtime | references in [skybox/runtime/stub/utils.cpp](../../../skybox/runtime/stub/utils.cpp) | perf scoreboard counters: `scrb_tex`, `scrb_raster`, `scrb_om`, plus `*cache` perf | Need to re-add to feature_gfx perf reporting |
| Kernel intrinsics | [skybox/kernel/include/vx_intrinsics.h](../../../skybox/kernel/include/vx_intrinsics.h) lines 100–115 | `vx_tex(stage, u, v, lod)`, `vx_om(x, y, face, color, depth)`, `vx_rast()` | Three new intrinsics on the kernel side |
| Tests | `skybox/tests/regression/{tex, raster, om, draw3d}` | 4 suites + ~30 `.cgltrace`/PNG assets | `draw3d` is the end-to-end integration test |

**Excluded from this migration** (out of scope):
- skybox FPGA AFU graphics integration (`hw/afu/*` graphics paths) — only software simulators (simx, rtlsim) are required initial targets.
- The `tests/regression/{conv3x, sgemm2x, sgemmx, vecaddx}` "x" variants which relied on a skybox-era kernel build path superseded by `vortex2` libs in feature_gfx.

### 2.2 Target landing zones (feature_gfx)

| Skybox path | feature_gfx target | Status |
|---|---|---|
| `hw/rtl/{tex,raster,om}/` | `hw/rtl/{tex,raster,om}/` (new) | new dirs alongside existing `dxa/`, `tcu/` |
| `hw/rtl/VX_graphics.sv` | inlined into [VX_cluster.sv](../../hw/rtl/VX_cluster.sv) (matches DXA/TCU style) | see §4.1 |
| `VX_types.vh` graphics defines | [VX_config.toml](../../VX_config.toml) (`ISA_EXT_*`) + [VX_types.toml](../../VX_types.toml) (`VX_DCR_*`) | TOML-driven config in v3 |
| `sim/simx/{tex,raster,om}_unit.{cpp,h}` | `sim/simx/{tex,raster,om}/` (new dirs, mirror `dxa/` and `tcu/` layout) | full rewrite — see §4.2 |
| `sim/common/{gfxutil,graphics}.{cpp,h}` | [sim/common/](../../sim/common/) | port largely as-is |
| `third_party/cocogfx/` | [third_party/cocogfx/](../../third_party/cocogfx/) (re-add) | git submodule or vendored — see §6 |
| `runtime/stub/utils.cpp` graphics perf | [sw/runtime/stub/perf.cpp](../../sw/runtime/stub/perf.cpp) (extend `tcu_en`-style block) | already has the pattern for TCU |
| `kernel/include/vx_intrinsics.h` `vx_tex/vx_rast/vx_om` | [sw/kernel/include/vx_intrinsics.h](../../sw/kernel/include/vx_intrinsics.h) | append three new inline functions |
| `tests/regression/{tex,raster,om,draw3d}` | [tests/regression/](../../tests/regression/) | port asset trees + Makefiles |

### 2.3 Already prepared in feature_gfx

The runtime header [sw/runtime/include/vortex.h:65–67](../../sw/runtime/include/vortex.h#L65)
already exposes `VX_ISA_EXT_TEX`, `VX_ISA_EXT_RASTER`, `VX_ISA_EXT_OM`
**but** the underlying `ISA_EXT_TEX/RASTER/OM` macros are **not** yet
defined in [VX_config.toml](../../VX_config.toml) (only `ISA_EXT_TCU=6`,
`ISA_EXT_DXA=7` exist). The runtime macros expand to references that
won't compile cleanly today; the migration's Phase 1 closes this.

---

## 3. Architectural deltas (skybox → feature_gfx)

### 3.1 RTL deltas

| Aspect | Skybox | feature_gfx | Impact on migration |
|---|---|---|---|
| Cluster-level GFX wrapper | `VX_graphics.sv` instantiates RASTER/TEX/OM + caches | No analog; TCU lives inline in core, DXA at SFU/cluster (see [VX_cluster.sv](../../hw/rtl/VX_cluster.sv), [VX_socket.sv](../../hw/rtl/VX_socket.sv)) | Drop the wrapper module; instantiate units + caches directly in `VX_cluster.sv` (RASTER/OM/TEX are still cluster-shared, not per-core, since they consume per-socket bus interfaces) |
| SFU PE attachment | `VX_dispatch_unit` + `VX_gather_unit` per SFU output, hand-wired to TEX/OM/RAST `*_bus_if` | `VX_pe_switch` + `PE_COUNT = 2 + EXT_DXA_ENABLED` ([VX_sfu_unit.sv:51](../../hw/rtl/core/VX_sfu_unit.sv#L51)) | Extend `PE_COUNT` to `2 + EXT_DXA_ENABLED + EXT_TEX_ENABLED + EXT_RASTER_ENABLED + EXT_OM_ENABLED` and add `pe` indices following the DXA template |
| Lane dispatch/gather modules | `VX_dispatch_unit.sv`, `VX_gather_unit.sv` | `VX_lane_dispatch.sv`, `VX_lane_gather.sv` (renamed + signature shifts) | TEX/OM/RASTER units touch these → must be retargeted |
| DCR plumbing | Cluster-level subset filter via `is_*_dcr_addr` + `BUFFER_DCR_BUS_IF` | `VX_dcr_arb` merges from KMU + host; `VX_dcr_data` distributes per-block ([core/VX_dcr_data.sv](../../hw/rtl/core/VX_dcr_data.sv)) | Add new DCR ranges to `VX_types.toml`; route via `VX_dcr_arb` instead of cluster-local subset filter |
| Memory bus | `VX_mem_bus_if` | `VX_mem_bus_if` (kept, but tag-width derivation differs) | Reuse `ASSIGN_VX_MEM_BUS_IF_X`-style adapters; recompute tag widths from new socket topology |
| Cache cluster | `VX_cache_cluster` (skybox version) | `VX_cache_cluster` (revised — different param names, different perf interface) | rcache/tcache/ocache instantiations need new param mappings |
| Issue path | `VX_issue.sv` + `VX_issue_top.sv` (no scheduler/sequencer) | `VX_scheduler` → `VX_uop_sequencer` → `VX_opc_unit` → `VX_dispatcher` → `VX_issue_slice` | TEX/RASTER/OM ops dispatched via SFU still funnel through this chain; uop encoding decisions for graphics ops must be made in `VX_decode.sv` like TCU/DXA |
| Scope/perf macros | `PERF_RASTER_ADD`, `PERF_TEX_ADD`, `PERF_OM_ADD` | `PERF_*` macros still exist but interface-bundle layout has changed | Re-derive macros against current `cache_perf_t` |

### 3.2 SimX deltas (and why a port ≠ rewrite-free)

The SimX in skybox is the pre-v3 simulator: procedural, `Emulator`-god-class
based, with backdoor `core_->mem_read`/`mem_write` everywhere. The
existing skybox graphics units **do** rely on these backdoor paths (e.g.
`om_core.cpp`, `tex_core.cpp`, `raster_core.cpp` synchronously read
texels/framebuffer through `core->mem_read`).

The current SimX v3 design enforces three rules
([simx_v3_proposal.md](simx_v3_proposal.md), see also DXA's §1):

1. **NoC-only memory access.** All TEX/OM/RASTER memory traffic must
   flow through `MemReq`/`MemRsp` channels into the appropriate cache
   cluster (rcache, tcache, ocache), not through `core_->mem_read`.
2. **Functional and timing coupled.** The texel a sampler returns is
   produced from the byte that the `MemRsp` carried, on the cycle the
   response channel accepts it. No "compute the answer at issue, replay
   timing later."
3. **Mirror RTL at module correspondence.** The C++ class graph mirrors
   the RTL module graph one-for-one (TexUnit → tex_unit → tex_arb →
   tex_sampler → tex_addr → tex_format → tex_lerp etc.), so debug and
   perf reasoning carry across.

Direct consequences:

- **Old `tex_core.cpp` (208 LOC) cannot be ported as-is.** The functional
  texel fetch (currently `mem_read` into a precomputed buffer) must be
  re-expressed as a request/response state machine that consumes
  `MemRsp` at the cycle it arrives. We mirror the same pattern that
  [dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md) §3 lays out for DXA.
- **Same for `raster_core.cpp` (547 LOC) and `om_core.cpp` (333 LOC).**
- The old `cache_sim.h`/`cache_sim.cpp` (skybox) is replaced by
  [sim/simx/mem/cache.h](../../sim/simx/mem/cache.h) — the new caches
  carry **data**, not just timing tags. rcache/tcache/ocache instantiate
  this new `Cache` directly.
- The `DCRS` aggregate ([skybox/sim/simx/dcrs.h](../../../skybox/sim/simx/dcrs.h))
  with separate `RasterUnit::DCRS`, `TexUnit::DCRS`, `OMUnit::DCRS`
  members maps cleanly onto feature_gfx's `dcrs.cpp/h` style — extend
  not rewrite.

### 3.3 Runtime / SW deltas

| Item | Notes |
|---|---|
| `vortex.h` ISA bits | already declared (§2.3); we add the `ISA_EXT_*` enum values in `VX_config.toml` to make them resolve |
| Perf reporting | extend [sw/runtime/stub/perf.cpp](../../sw/runtime/stub/perf.cpp)'s `tcu_en`-style block with `tex_en`, `raster_en`, `om_en` checks; reuse the `VX_CSR_MPM_*` framework — new MPM IDs go in `VX_types.toml` |
| Kernel intrinsics | three new inlines (`vx_tex`, `vx_rast`, `vx_om`) in [sw/kernel/include/vx_intrinsics.h](../../sw/kernel/include/vx_intrinsics.h); these emit the corresponding custom RISC-V opcodes that `VX_decode.sv` must recognize under `EXT_*_ENABLE` |
| Test harness | tests need `sim/common/gfxutil.cpp` + `cocogfx` link; integrate into [tests/regression/common.mk](../../tests/regression/common.mk) following the pattern that `sgemm_tcu` uses for `softfloat` |

---

## 4. Phase plan

Each phase is independently shippable: the build passes on a non-graphics
config (`EXT_*_ENABLE=false`) at every step, and the graphics path
(when enabled) reaches a defined milestone.

### Phase 0 — Baseline & scaffolding (no functional change)

- Set up `build_test32/` and confirm baseline build + smoke tests
  (`./ci/blackbox.sh --driver=simx --app=demo`,
  `make -C tests/regression run-simx`).
- Add `EXT_TEX_ENABLE`, `EXT_RASTER_ENABLE`, `EXT_OM_ENABLE` (default
  `false`) and `ISA_EXT_TEX/RASTER/OM` enum values to
  [VX_config.toml](../../VX_config.toml); extend `MISA_EXT` expression
  to OR them in. Re-run configure; confirm baseline still green.
- Vendor `third_party/cocogfx` (or add as submodule, matching skybox
  conventions); wire into [third_party/Makefile](../../third_party/Makefile).
  Confirm baseline still green.

**Exit criteria:** all existing tests pass; `EXT_*_ENABLE=true` is
recognized by configure but no RTL/SimX consumes it yet.

### Phase 1 — DCR + ISA wiring (no behavior)

- Add `VX_DCR_TEX_*`, `VX_DCR_RASTER_*`, `VX_DCR_OM_*` to
  [VX_types.toml](../../VX_types.toml).
- Add `vx_tex`, `vx_rast`, `vx_om` inline functions to
  [sw/kernel/include/vx_intrinsics.h](../../sw/kernel/include/vx_intrinsics.h)
  with the same custom opcodes skybox used (or define new ones if there
  is a collision).
- Add decode entries to [hw/rtl/core/VX_decode.sv](../../hw/rtl/core/VX_decode.sv)
  under `EXT_TEX_ENABLE` / `EXT_RASTER_ENABLE` / `EXT_OM_ENABLE`,
  routing them to a new SFU PE index.
- Add the perf scoreboard counters (`scrb_tex`, `scrb_raster`, `scrb_om`)
  to [sw/runtime/stub/perf.cpp](../../sw/runtime/stub/perf.cpp).

**Exit criteria:** `EXT_*_ENABLE=true` builds; instructions decode and
silently no-op (or raise a defined illegal-inst trap), so a kernel
that uses `vx_tex` etc. compiles. SimX/RTL still have no functional
units.

### Phase 2 — RTL graphics units

- Port `hw/rtl/{tex,raster,om}/` directory by directory, **without**
  cluster-level integration:
  - re-target package imports to current `VX_gpu_pkg`,
  - re-target `VX_*_bus_if` interface declarations to the current
    convention,
  - replace `VX_dispatch_unit`/`VX_gather_unit` references with
    `VX_lane_dispatch`/`VX_lane_gather`,
  - replace skybox `VX_cache_cluster` parameter mapping with the
    feature_gfx version,
  - run [docs/coding_guidelines_verilog.md](../coding_guidelines_verilog.md)
    lint cleanly under verilator.
- Add unit-level standalone tests under [hw/unittest/](../../hw/unittest/)
  mirroring `tcu_unit/` and `dxa_core/`.
- Integrate at cluster scope: instantiate TEX/RASTER/OM units in
  `VX_cluster.sv` directly (no separate `VX_graphics.sv`); add
  rcache/tcache/ocache cache_cluster instances; route DCR via
  `VX_dcr_arb`.
- Wire the per-socket bus_ifs through the new SFU PE_COUNT chain in
  `VX_sfu_unit.sv` following the DXA template.

**Exit criteria:** with `EXT_*_ENABLE=true`, RTL elaborates and a
trivial smoke kernel that issues a single `vx_tex` / `vx_rast` /
`vx_om` op completes (verified under rtlsim). SimX would still error
on these — gated.

### Phase 3 — SimX graphics units (TLM-aligned rewrite)

For each unit (TEX, then OM, then RASTER — increasing complexity):

- Mirror the RTL module split as C++ classes under `sim/simx/{tex,om,raster}/`
  with `*_unit.cpp/h`, and per-block helpers as required by §3.2 rule 3.
- Wire functional behavior on `MemRsp` arrival (rule 2). The unit
  consumes a `MemReq`/`MemRsp` channel against the new
  `Cache`-as-rcache/tcache/ocache instance.
- Hook into `SfuUnit` PE table at the same index used by RTL (rule 3
  consistency).
- Extend `Cluster`/`Socket` to hold the new cache instances (mirroring
  `dxa_core` integration in [sim/simx/cluster.cpp](../../sim/simx/cluster.cpp)).
- Add per-unit unit tests against canonical CSV traces.

**Exit criteria:** SimX matches RTLsim cycle-level on the smoke kernels
from Phase 2. CSV trace diff between simx and rtlsim is empty (or
explained) for `tex`, `om`, `raster` regression tests.

### Phase 4 — Regression tests

- Port `tests/regression/{tex, om, raster}` (asset trees +
  `kernel.cpp`/`main.cpp` + `Makefile`) to the
  current test-build conventions (`vortex2` kernel lib,
  `SW_COMMON_DIR`, etc.), adapting to feature_gfx's
  [tests/regression/common.mk](../../tests/regression/common.mk).
- Port `tests/regression/draw3d` last (it depends on all three units +
  `gfxutil` + `cocogfx`).
- Add to [ci/regression.sh](../../ci/regression.sh) as a new suite,
  gated on `EXT_TEX_ENABLE`/`EXT_RASTER_ENABLE`/`EXT_OM_ENABLE`.

**Exit criteria:** all four suites pass on simx and rtlsim, in 32-bit
mode, with reference PNGs binary-equal to skybox's.

### Phase 5 — Perf + cleanup

- Re-add MPM perf classes (raster scoreboard, tex/om/raster cache
  perf) to [sw/runtime/stub/perf.cpp](../../sw/runtime/stub/perf.cpp),
  mirroring what skybox `runtime/stub/utils.cpp` reports.
- Document in [docs/](../) (architecture overview entry; reference
  from [docs/codebase.md](../codebase.md)).
- Optional: add a perf example invocation to AGENTS.md.

**Exit criteria:** `make -C tests/regression run-simx` and `run-rtlsim`
green with `EXT_TEX_ENABLE` etc.; perf reports include graphics
counters when extensions are on.

---

## 5. Build and test plan (`build_test32/`)

All work happens inside `build_test32/`:

```bash
cd /home/blaisetine/dev/vortex_v3/feature_gfx/build_test32
../configure --xlen=32 --tooldir=$HOME/tools
source ./ci/toolchain_env.sh
make -s
```

Per-phase validation:

- **Baseline (Phase 0):** `make -C tests/regression run-simx`,
  `make -C tests/regression run-rtlsim` (subset).
- **Per RTL unit (Phase 2):** `make -C hw/unittest/<unit>`.
- **Per SimX unit (Phase 3):** unit-level tests + CSV trace diffs vs RTL.
- **Regression (Phase 4):** the four ported suites; sustained green on
  `EXT_*_ENABLE=true` for both drivers.

CSV trace diffs use the same per-cycle CSV logging that
[simx_v3_proposal.md](simx_v3_proposal.md) §6 already describes.

---

## 6. Open questions / risks

1. **`cocogfx` packaging.** Skybox vendored it as a submodule. The
   current branch dropped it. Two options: (a) reinstate as submodule
   (re-introduces a remote dep), (b) vendor a pinned snapshot under
   `third_party/cocogfx/` (one-time copy, no upstream tracking). Lean
   toward (b) unless the user wants upstream tracking.
2. **Custom opcode collision.** Skybox `vx_tex`/`vx_rast`/`vx_om` use
   custom RISC-V opcodes. We need to verify these don't collide with
   TCU/DXA opcodes the current branch added; if they do, re-allocate
   under the existing custom-opcode space and document.
3. **DCR address-space layout.** Skybox `VX_DCR_TEX/RASTER/OM_*` ranges
   were stacked relative to `VX_DCR_BASE_STATE_END=0x006`. Current
   layout reserves `0x010..0x01F` for KMU and `0x100+` for DXA. We need
   to pick fresh ranges; proposal: `VX_DCR_TEX_STATE_BEGIN=0x020`,
   `VX_DCR_RASTER_STATE_BEGIN=0x040`, `VX_DCR_OM_STATE_BEGIN=0x060`
   (subject to confirmation in Phase 1).
4. **Per-socket vs per-cluster placement.** Skybox kept TEX/RASTER/OM
   cluster-shared (not per-core, not per-socket) and arbitrated from
   sockets via `*_arb`. In feature_gfx, DXA is also cluster-shared but
   TCU is per-core. **Proposal:** keep skybox's cluster-shared
   placement for all three, since it matches their typical
   instantiation count (often 1 per cluster) and matches the existing
   per-socket arbitration pattern.
5. **Whether to keep `VX_graphics.sv` as a wrapper.** I lean toward
   inlining into `VX_cluster.sv` (matches DXA/TCU style and reduces
   one level of indirection). Open to keeping the wrapper if the user
   prefers separation of concerns.
6. **simx_v3 phase status for graphics.** The simx_v3 phases 1–4 are
   complete and Phase 5 is in progress. The graphics SimX rewrite
   should land **after** Phase 5 (caches carry data) is complete, since
   tex/om/raster fundamentally depend on the cache hierarchy carrying
   real data. **Proposal:** Phase 0–2 (RTL + DCR + scaffolding) can
   start immediately; Phase 3 (SimX) waits on simx_v3 Phase 5.
7. **Timeline.** Rough estimate: Phase 0 ≈ 0.5d, Phase 1 ≈ 1d, Phase 2
   ≈ 5–7d (RTL is the bulk), Phase 3 ≈ 7–10d (SimX rewrite), Phase 4 ≈
   2–3d, Phase 5 ≈ 1d. Total ~3 weeks of focused work.

---

## 7. Decisions needed before starting

Please confirm or redirect:

- [ ] Migration scope: TEX + RASTER + OM only, plus the four
      regression suites (tex, raster, om, draw3d). Excluding skybox
      AFU graphics paths and the "x" variant tests. **OK?**
- [ ] cocogfx packaging: vendor a snapshot under `third_party/cocogfx/`
      (option b above). **OK?**
- [ ] DCR addresses: `0x020` / `0x040` / `0x060` ranges for TEX /
      RASTER / OM. **OK?**
- [ ] Drop `VX_graphics.sv` wrapper; instantiate units directly in
      `VX_cluster.sv` (matches DXA/TCU style). **OK?**
- [ ] Phase 3 (SimX) gates on simx_v3 Phase 5 completion. **OK to
      proceed Phase 0–2 first?**
- [ ] Cluster-shared placement for all three units (not per-socket /
      per-core). **OK?**

Once aligned, I'll begin Phase 0 inside `build_test32/`.

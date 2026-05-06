# SST Integration for SimX v3 — Proposal

**Date:** 2026-05-03
**Status:** Draft
**Author:** Blaise Tine
**Related:**
[simx_v3_proposal.md](simx_v3_proposal.md) (Phase 5: TLM data path),
[master_merge_v3_proposal.md](master_merge_v3_proposal.md) §10.2 (PR #298 not adopted),
upstream PR #298 (Jagadheesvaran T S, original SST integration on master).

---

## 1. Constraints (load-bearing)

Any design that breaks one of these is wrong.

1. **One source of truth for memory state.** Per
   [simx_v3_proposal.md §3.3](simx_v3_proposal.md), data lives in the
   channel hierarchy: `MemReq`/`MemRsp` packets carry actual bytes
   between `MemCoalescer` → `Cache` → `Memory`. There is no shadow
   `MemBackend` that the LSU reads/writes synchronously while the
   cache plays back timing-only tokens. **PR #298's
   `mem_backend_dram`/`mem_backend_sst` abstraction is the pre-Phase-5
   pattern** that v3 explicitly retired; reintroducing it would undo
   §3.3.
2. **SST plugs in at one boundary, not many.** SimX → SST traffic
   crosses one interface (the v3 `Memory` module's external port).
   The cache hierarchy, scheduler, ALU/FPU, and Emulator (gone) do
   not know SST exists.
3. **Single clock owner per simulation.** Either SST drives the
   simulation (advances simx one cycle at a time and routes
   memory traffic through SST's components) or simx runs free with
   a SST-shaped memory peer. Not both, not "sometimes one and
   sometimes the other." PR #298 picked SST-as-driver; this proposal
   keeps that decision.
4. **No regression for the non-SST build.** `make -C sim/simx` (no
   `USE_SST=1`) continues to produce a self-contained `simx` binary
   with the same behavior as today. SST is opt-in compile-time, not
   a runtime probe.
5. **Authorship preserved.** Jagadheesvaran T S authored upstream
   PR #298; the SST CI test scripts (`ci/sst_test_vortex_*.py`) are
   reused unchanged where possible. Where v3 forces a re-implementation
   of the C++ side, the original commits remain referenced in commit
   bodies (`Original PR #298 / commit <sha>`).

---

## 2. Why upstream PR #298 cannot be cherry-picked as-is

Reading PR #298's actual diff against this branch (per the inventory
in [master_merge_v3_proposal.md](master_merge_v3_proposal.md) §10.2):

### 2.1 The architectural mismatch

PR #298 was built on top of the legacy SimX, where:

- `Emulator` owned the functional state (PC, registers, memory image).
- The cache hierarchy carried *tags only*; data lived in `MemBackend`,
  which the LSU read/wrote synchronously.
- SST integration replaced `MemBackend` with a `mem_backend_sst` adapter
  that forwarded loads/stores to SST's memory model.

v3 explicitly inverts that:

- `Emulator` is decomposed; per-warp PC lives in `Scheduler`,
  registers in `OpcUnit`, no shared backing store.
- The memory hierarchy is the source of truth. `MemReq`/`MemRsp`
  carry data through `Cache` and `Memory`; there is no `MemBackend`
  to swap.

So the **shape of the SST plug-in changes**: not "replace MemBackend
with an SST-backed one" but "replace `Memory` (or its external port)
with an SST-routed peer."

### 2.2 What still ports cleanly

- `sim/simx/VortexGPGPU.{cpp,h}` — the SST Component wrapper class.
  Subclasses `SST::Component`, owns a `Processor` instance, calls
  `processor.cycle()` on the SST clock tick. **Largely v3-compatible.**
- `sim/simx/vortex_simulator.{cpp,h}` — entry-point shim that
  exposes `VortexGPGPU` to the SST element registry. **Largely
  v3-compatible.**
- `Processor::cycle()` — single-cycle entry point that wraps
  `SimPlatform::tick()` + per-cluster running check. **The pattern
  is correct for v3** (it is the same pattern I added for DTM in
  Round 6 — see `sim/simx/main.cpp`'s debug-mode loop). Need to add
  it to the v3 `Processor`/`ProcessorImpl` API.
- `ci/sst_test_vortex_*.py`, `ci/sst_install.sh.in`,
  `ci/regression.sh.in` `sst()` function. **Port unchanged.** The
  Python tests instantiate `vortex.VortexGPGPU` and pass a `program`
  parameter; that's an interface contract, not implementation-tied.

### 2.3 What needs a v3 redesign

- **`mem_backend_dram.{cpp,h}`** — delete. v3's `Memory` already
  models DRAM through ramulator; no separate backend abstraction.
- **`mem_backend_sst.{cpp,h}`** — delete. Replace with a v3-aligned
  `MemorySST` class that mirrors the existing `Memory` class shape
  (same `MemReq` in / `MemRsp` out channels) but instead of feeding
  ramulator, forwards each `MemReq` to an SST `MemHierarchy::Link`
  and emits `MemRsp` when SST's response arrives.
- **`mem_backend.h`** (the abstract base) — delete. The v3 polymorphism
  point is `Memory` vs. `MemorySST` at construction time, not a runtime
  vtable.

---

## 3. Target architecture

```
                ┌─────────────────────────────┐
                │  SST simulator (driver)     │
                │  - sst_test_vortex_*.py     │
                │  - clock = "1GHz"           │
                │                             │
                │  ┌───────────────────────┐  │
                │  │ vortex.VortexGPGPU    │  │
                │  │ (SST::Component)      │  │
                │  │                       │  │
                │  │  Processor processor; │  │
                │  │  on each clock tick:  │  │
                │  │    processor.cycle()  │  │
                │  └────────┬──────────────┘  │
                │           │ owns            │
                │  ┌────────▼─────────────┐   │
                │  │ Processor (v3)       │   │
                │  │  ┌────────────────┐  │   │
                │  │  │ Cluster[]      │  │   │
                │  │  │  Socket[]      │  │   │
                │  │  │   Core[]       │  │   │
                │  │  │    ALU/FPU/SFU │  │   │
                │  │  │    LSU         │  │   │
                │  │  │    Cache (L1)  │  │   │
                │  │  │  Cache (L2)    │  │   │
                │  │  │ Cache (L3)     │  │   │
                │  │  └───────┬────────┘  │   │
                │  │          │ MemReq    │   │
                │  │          │ MemRsp    │   │
                │  │          ▼           │   │
                │  │   ┌──────────────┐   │   │
                │  │   │ MemorySST    │◀──┼───┼──── SST::Link
                │  │   │ (v3 module,  │   │   │     to memHierarchy
                │  │   │  channel-in/ │   │   │     bus + DRAM
                │  │   │  channel-out)│   │   │
                │  │   └──────────────┘   │   │
                │  └──────────────────────┘   │
                └─────────────────────────────┘
```

### 3.1 The plug-in boundary

The replaceable unit is `Memory` (`sim/simx/mem/memory.{cpp,h}`).
Two construction-time variants:

- **`Memory`** — current (default). Owns `ramulator2_handle_t`, drives
  ramulator on `tick()`.
- **`MemorySST`** — new (only compiled with `USE_SST=1`). Same channel
  shape; on each `MemReq` it received, posts to an SST link and
  parks the request in an `outstanding_` map. On the SST event
  callback, completes the matching `MemRsp` into the upstream
  channel.

Selection at `ProcessorImpl` construction:
```cpp
#ifdef USE_SST
  memsim_ = MemorySST::Create("memsim", sst_link_);
#else
  memsim_ = Memory::Create("memsim");
#endif
```

`MemorySST` must implement the same `SimChannel<MemReq>` /
`SimChannel<MemRsp>` interface as `Memory` so the L3 cache binds to
it transparently. **Nothing upstream of `MemorySST` knows SST exists.**
This satisfies §1.2.

### 3.2 The cycle interface

`Processor::cycle()` advances the simulator by one cycle, lazily
initializing on first call. Direct port from upstream PR #298 with
the v3 termination predicate (`!any_running()` instead of the
upstream per-cluster loop):

```cpp
bool ProcessorImpl::cycle() {
  if (!is_cycle_initialized_) {
    SimPlatform::instance().reset();
    this->reset();
    kmu_->start();              // see Round 6 DTM main.cpp
    is_cycle_initialized_ = true;
  }
  SimPlatform::instance().tick();
  return any_running();          // already added in Round 6
}
```

**Reuse from DTM work:** `start_kmu()` and `any_running()` are already
public on `Processor` (added in Round 6 for the debug-mode tick loop).
SST integration uses the same primitives.

### 3.3 Build-time gating

`USE_SST=1` make variable controls compilation of:
- `MemorySST.{cpp,h}` (only built under `USE_SST=1`)
- `sim/simx/sst/vortex_gpgpu.{cpp,h}` (the `SST::Component` wrapper)
- `sim/simx/sst/vortex_simulator.{cpp,h}` (SST element registration)
- Linkage against `sst-core` and `sst-elements` libraries

Default build (`USE_SST=` unset) produces a stand-alone `simx` binary
with no SST dep. Per §1.4.

---

## 4. Phasing

Each phase is independently shippable and validated. The work follows
the same shape as the DTM port in Round 6 (file scaffolding first,
integration second, CI third).

### Phase 1 — `Processor::cycle()` + `MemorySST` skeleton (no SST link yet)

- Add `Processor::cycle()` and `ProcessorImpl::cycle()` mirroring
  Round 6's DTM-loop pattern.
- Create `sim/simx/mem/memory_sst.{cpp,h}` as a *trivial subclass* of
  `Memory` that just calls through to the parent on every `MemReq`.
  No SST yet; this proves the plug-in interface compiles and that
  `MemorySST` can swap in via `USE_SST=1`.
- `make -C sim/simx USE_SST=1` builds; resulting `simx` runs identical
  to the no-SST build (because `MemorySST` is a passthrough).

**Validation:** vecadd, io_addr, arith pass under both `USE_SST=0`
and `USE_SST=1`.

### Phase 2 — SST Component wrapper

- Drop in `sim/simx/sst/vortex_gpgpu.{cpp,h}` and
  `sim/simx/sst/vortex_simulator.{cpp,h}` adapted from upstream
  PR #298 with v3 changes:
  - Replace `processor.cycle()` calls with the new v3 `Processor::cycle()`.
  - Drop any reference to `Emulator`, `mem_backend_*`, or
    `get_first_emulator()` (none survive in v3).
- Wire the SST element-registration boilerplate.
- `apt/sst install` per upstream `ci/sst_install.sh.in`.

**Validation:** `sst ci/sst_test_vortex_hello.py` runs the helloworld
kernel through the SST-driven harness. (The Python script is unchanged
from upstream.) Outputs match a non-SST `./simx hello.vxbin` run.

### Phase 3 — Real `MemorySST` with SST link

- Replace the Phase-1 passthrough with the actual SST integration:
  - Constructor takes an `SST::Link*` (the simx side of the
    SST `MemHierarchy` interface).
  - `MemReq` arriving on `Inputs` channel → translate to SST event
    (address, size, isWrite, payload bytes) → `link->send(event)`.
  - SST event callback (registered in `vortex_gpgpu.cpp`) → match
    against `outstanding_` map by tag → emit `MemRsp` (with response
    bytes) onto `Outputs` channel.
  - `tick()` simply drains/resends pending events; clock advance is
    SST's job.
- The `outstanding_` map tracks `(tag → SimChannel<MemRsp>* sink,
  cycle_issued)` so we can emit the response into the right place
  when SST delivers it.

**Validation:**
- `sst ci/sst_test_vortex_hello.py` end-to-end PASS.
- `sst ci/sst_test_vortex_fibonacci.py` PASS.
- `sst ci/sst_test_vortex_vecadd.py` PASS.
- `sst ci/sst_test_vortex_conform.py` PASS.
- Non-SST regression suite still passes — SST never touches the default
  build.

### Phase 4 — CI

- Add `sst()` function to `ci/regression.sh.in` (port of upstream
  patch — already in the docs of master_merge_v3 §4.3 as deferred).
- Add `sst` matrix entry to `.github/workflows/ci.yml`.
- Drop in `ci/sst_install.sh.in` from upstream.
- Optional: skip `sst` matrix entry on `apptainer-ci.yml` for now (the
  apptainer container would need SST 15.1 + OpenMPI installed; that's
  a separate vortex.def expansion).

**Validation:** `./ci/regression.sh --sst` runs all four
`sst_test_vortex_*.py` scripts cleanly.

### Phase 5 — Documentation

- Add `docs/sst_integration.md` (or fold into
  `docs/simulation.md` as a new section) with:
  - How to install SST 15.1 (point at `ci/sst_install.sh.in`).
  - How to build with `USE_SST=1`.
  - How to write an SST Python script that drives `VortexGPGPU`.
  - The single-source-of-truth invariant (§1.1) for future hackers
    who might be tempted to add an `mem_backend_*` shortcut.

---

## 5. Authorship / history mechanics

- `sim/simx/sst/vortex_gpgpu.{cpp,h}` and `vortex_simulator.{cpp,h}`:
  port from upstream PR #298 with `git format-patch`-style header
  preserving Jagadheesvaran's authorship; commit body cites
  `Original PR #298 (commit <sha>). Re-applied for v3 Processor::cycle
  API and dropped Emulator references.`
- `MemorySST` is a **new file** — no upstream equivalent. Authored
  locally; commit body credits PR #298 for the design intent
  (`Replaces upstream PR #298's mem_backend_sst with a v3-aligned
  Memory subclass.`).
- `ci/sst_test_vortex_*.py` and `ci/sst_install.sh.in`: cherry-pick `-x`
  with Jagadheesvaran's authorship intact.
- `ci/regression.sh.in` `sst()` function and `ci.yml` matrix entry:
  follow the same pattern as the DTM wiring in Round 6.

This matches the rule established in
[`feedback_keep_ours_in_merge.md`](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_keep_ours_in_merge.md):
take theirs unchanged where the v3 side has no opinion, re-implement
where v3 forces a different shape (and credit upstream in the body).

---

## 6. Validation

Each phase ends with the validation listed in §4. Across phases the
acceptance criteria are:

1. **No-SST build identical.** `git diff <pre>..<post>` against a
   `make -C sim/simx` (default flags) build shows no behavioral change
   on the regression suite (io_addr, arith, vecadd, mpi_vecadd,
   tensor*, dxa, dtm). Trace-level diff against pre-merge HEAD on
   vecadd is empty.
2. **SST build runs upstream's tests.** All 4 `sst_test_vortex_*.py`
   scripts pass.
3. **No `core->mem_read` / `core->mem_write` regressions.** Phase 5
   of v3 forbids those (§3.3). The grep gate from
   [master_merge_v3_proposal.md §8 R1](master_merge_v3_proposal.md)
   applies here too — every commit in this work must pass
   `git diff <pre>..<post> -- sim/simx/ | grep -E 'core->mem_(read|write)' | wc -l == 0`.
4. **Single source of truth check.** The new `MemorySST` must hold
   *no* persistent backing-store data — only an in-flight request
   map. Reviewer must verify this in code; mistakes here re-introduce
   the §1.1 violation.

---

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | SST 15.1 API drift between releases. | Pin SST core/elements to a tested version in `ci/sst_install.sh.in` (15.1.0_Final per upstream). Document the pinning in the new `docs/sst_integration.md`. |
| R2 | The `outstanding_` map could leak memory if SST drops a request without responding. | Add a per-request timeout; on tick, drop entries older than `STALL_TIMEOUT` and log. Mirror the pattern already used in `Cache::PerfStats`. |
| R3 | Multi-cluster simulations send `MemReq`s from many sources to one `MemorySST` link → SST sees interleaved unrelated transactions. | Tag every outgoing event with the originating channel; track `(tag, sink_channel*)` in `outstanding_`. The 64-bit MemReq tag width gives plenty of bits. |
| R4 | SST's clock and SimPlatform's tick are now coupled — a slow SST event delivery can starve simx's internal channels. | Keep `MemorySST::tick()` cheap (no SST work happens here — the SST event loop runs on its own clock; we just drain `outstanding_`). |
| R5 | The build-time switch (`USE_SST=1`) creates two variants of `simx` that drift over time. | CI builds both variants every commit. The `sst` matrix entry covers `USE_SST=1`; the rest cover the default. |
| R6 | A future contributor adds `mem_backend_*` "for convenience" and re-introduces the §1.1 violation. | This proposal explicitly deletes that abstraction (§2.3). The follow-up `docs/sst_integration.md` (§4 Phase 5) should call this out. |
| R7 | SST tests in CI need OpenMPI + SST built from source; the `ci/sst_install.sh.in` script takes ~10 min on cache miss. | Cache the SST build directory (same pattern as the existing toolchain cache). Drop the SST install on `apptainer-ci.yml` until the .sif recipe absorbs it. |

---

## 8. Out of scope

- **Apptainer integration.** Adding SST to `miscs/apptainer/vortex.def`
  is a separate concern. Until that's done, `apptainer-ci.yml`'s
  matrix should not include `sst`. See
  [`apptainer-ci.yml` policy notes](../../.github/workflows/apptainer-ci.yml).
- **Multi-rank MPI simulation.** Upstream PR #298 is single-rank; this
  proposal preserves that. Multi-rank is a separate redesign that
  intersects with PR #282's MPI work.
- **SST + DXA / SST + KMU interaction.** DXA's GMEM/LMEM channels
  bind to `Memory` indirectly; if SST replaces `Memory`, DXA traffic
  flows through SST too. That's correct by design but unverified;
  Phase 3 should add a `dxa_copy` test under SST as a smoke check.
  Performance characterization deferred.
- **Runtime SST/non-SST switching.** Keep `USE_SST=1` as a build-time
  switch. A runtime switch would require both `Memory` and `MemorySST`
  in every binary plus a factory; the maintenance cost outweighs the
  benefit.
- **Replacing ramulator with SST's `MemHierarchy::DDR4`.** The default
  (non-SST) build still uses ramulator. SST users get SST's memory
  models via `MemorySST`. Mixing is out of scope.
- **The DTM stack under SST.** Round 6's DTM work uses
  `SimPlatform::tick()` directly; under SST that becomes
  `processor.cycle()`. The two patterns are compatible (DTM's
  halt-gating sits on top of either). Verifying DTM works under SST
  is a Phase 3 stretch goal, not a Phase 3 requirement.

---

## 9. Estimated effort

Based on the DTM port in Round 6 (similar shape: drop standalone
files, add v3 integration glue, wire CI):

- Phase 1 (`cycle()` + passthrough `MemorySST`): **2–4 hours**.
  Mostly mechanical given the DTM patterns we already have.
- Phase 2 (SST wrapper port): **3–5 hours**. SST's
  Component/Element boilerplate is tedious but well-trodden territory
  upstream already has working code.
- Phase 3 (real `MemorySST` with SST link): **8–16 hours**. The
  mapping from `MemReq`/`MemRsp` to SST `MemEventBase` is the
  novel design work. Likely a couple of debug iterations on the
  request-tag plumbing.
- Phase 4 (CI): **1–2 hours**.
- Phase 5 (docs): **1 hour**.

Total: **~15–28 hours** of focused work. Substantial enough to
warrant its own branch (`sst_simx_v3` or similar) and not be folded
into other refactor work.

# Vortex VM Migration Proposal

**Source:** `~/dev/vortex_vm` (branch `bug_fixes`, 1 commit "shell script fail decision fix, basic SATP fix")
**Target:** `~/dev/vortex_v3/feature_vm` (branch `tinebp-patch-2`)
**Build dir (working only):** `~/dev/vortex_v3/feature_vm/build_test32` (configured `--xlen=32`)
**Status:** Phases 0–6 complete. End-user reference doc: [../vm.md](../vm.md).

---

## 0. Final status (per-phase)

| Phase | Outcome |
|---|---|
| 0 — baseline | Clean build, no-VM `vecadd` PASSes on simx + rtlsim |
| 1 — `VM_ENABLE` flag + perf CSR ids | Build still clean, no semantic change |
| 2 — SimX VM end-to-end | `vecadd`/`basic`/`dotproduct`/`sgemm` PASS on simx with VM |
| 3 — RTL MMU (the bulk) | `vecadd`/`basic`/`dotproduct` PASS on rtlsim with VM. dMMU + iMMU per core |
| 4 — randomized VA + perf counters | 24/24 simx VM regression PASS at fixed seed; reproducible. `vm:` perf line surfaced in `vx_dump_perf` |
| 5 — new tests | 9 VM-agnostic + 5 TCU + 5 DXA (incl. TCU+DXA combos) all PASS on simx with VM. §2.2 outcome: **DXA worked without the proposed PA-descriptor refactor** — the runtime-API path naturally translates because DXA reads memory through the same per-core dcache port that already goes through the dMMU |
| 6 — docs cleanup | This file + [../vm.md](../vm.md) |

### Things noted in passing

- `VX_config.h` auto-defines `VM_ENABLE` from the TOML (via `#ifndef VM_DISABLE / #define VM_ENABLE`). Header-level `#ifdef VM_ENABLE` guards must follow the `<VX_config.h>` include, not precede it. CONFIGS=`-DVM_ENABLE` is redundant once the TOML flips.
- `MPM` perf-CSR encoding is 5-bit-per-class (0xB00–0xB1F). Phase 1's TLB CSRs at 0xB20+ overflowed; moved them into a new `VX_DCR_MPM_CLASS_VM` class at 0xB03–0xB08.
- `async_gbarrier` fails on simx with **and** without VM — pre-existing test issue, not a VM regression.

### Two TODO breadcrumbs

- One `-Wstringop-overread` warning on the `(uint64_t*)dest` cast over a 4-byte buffer in [vm.cpp:343](../../sw/runtime/common/vm.cpp#L343) when `XLEN_32`. Functional but noisy. Replace with `memcpy` to a `uint64_t` local.
- Full 24-test rtlsim regression (`./run_vm_regression.sh --driver=rtlsim`) hasn't been run end-to-end. Spot-checked 3 (`vecadd`, `basic`, `dotproduct`) PASS.

---

## 1. Why this migration is non-trivial

Both halves of the codebase have moved on since VM was implemented:

- **Source repo** (`vortex_vm`) is forked from a much older Vortex. RTL still uses `VX_config.vh` / `VX_types.vh`; runtime lives in `runtime/`; kernel in `kernel/`; SimX has only `core/cluster/socket/processor/emulator`; there is no DXA, no TCU pipeline, no mem coalescer, no LSU↔mem adapter, no KMU.
- **Target repo** (`feature_vm`) has reorganized: config is now TOML‑driven (`VX_config.toml` → generated `VX_config.vh`); driver moved `runtime/` → `sw/runtime/`; kernel moved to `sw/kernel/`; new units `dxa/`, `tcu/`, `kmu/` and new memory plumbing (`mem_coalescer`, `lsu_mem_adapter`, `mem_xbar`, `local_mem`, `lmem_switch`) sit between LSU and cache.

**The good news** — the target repo *already* contains a substantial chunk of dormant VM scaffolding from a prior partial merge:

| Where | What's pre‑merged | Status |
|---|---|---|
| `VX_config.toml` | `VM_ENABLE`, `[vm]` block (PT_LEVEL, PTE_SIZE, TLB_SIZE, PAGE_TABLE_BASE_ADDR, MEM_PAGE_SIZE, etc.), `VM_ADDR_MODE` enum (BARE/SV32/SV39/SV48/SV57) | Compiles, gated `VM_ENABLE = false` |
| `VX_types.toml` | `VX_CSR_SATP = 0x180` | Defined |
| [sim/common/mem.h](../../sim/common/mem.h), [sim/common/mem.cpp](../../sim/common/mem.cpp) | `SATP_t`, `PTE_t`, `TLBEntry`, `MemoryUnit::page_table_walk`, `tlbLookup`, `set_satp` (~580 + 790 LOC, larger than source!) | **Orphaned** — class `MemoryUnit` is not referenced from `sim/simx/` or `sim/rtlsim/` |
| [sim/simx/csr_unit.cpp:53](../../sim/simx/csr_unit.cpp#L53), [:257](../../sim/simx/csr_unit.cpp#L257) | `VX_CSR_SATP` cases | **Stub** — read returns 0, write is no‑op |
| [sw/common/mem_alloc.h](../../sw/common/mem_alloc.h) | `#ifdef VM_ENABLE` block tracking | Compiles when flag set |

**The bad news** — the wiring is incomplete and several pieces are missing:

| Missing in target | Present in source |
|---|---|
| `hw/rtl/core/VX_mmu.sv` (401 LOC) | Yes |
| `hw/rtl/core/VX_mmu_tlb.sv` (501 LOC) | Yes |
| `hw/rtl/core/VX_mmu_ptw.sv` (180 LOC) | Yes |
| MMU instantiation in `VX_core.sv` | Yes |
| MMU plumbing in `VX_fetch.sv`, `VX_lsu_unit.sv`, `VX_mem_unit.sv`, `VX_execute.sv` | Yes |
| `mmu_perf_t` struct in `VX_gpu_pkg.sv` + 6 perf CSR ids (`VX_CSR_MPM_TLB_*`, `PTW_*`) | Yes |
| `runtime/common/vm.{h,cpp}` (`VMManager`, page_table_walk, randomized VA) | Yes |
| `vm.cpp` instantiation in simx/rtlsim runtime drivers | Yes |
| SATP setup in `vx_start.S` | Yes — sets PT base + mode, writes `satp` CSR |
| Real SATP read/write in simx `csr_unit.cpp` | Yes (in `core.cpp`/`processor.cpp`) |

---

## 2. Architectural incompatibilities to resolve

### 2.1 Memory path is restructured
Source: `LSU → mem_unit → MMU → cache`. MMU sits inline on the fetch and load/store paths.

Target: `LSU → lsu_slice → lsu_mem_adapter → (mem_xbar / lmem_switch / mem_coalescer) → cache`. There is no single inline insertion point — translation has to be inserted *before* coalescing (so coalescing sees physical addresses) but *after* per‑lane LSU dispatch. Same for instruction fetch.

**Decision required:** Where does the MMU physically live in the new pipeline? Two options:
1. **Per‑LSU‑slice MMU** (closest to source) — instantiate one MMU per `VX_lsu_slice`, before `lsu_mem_adapter`. Adds area but matches source semantics.
2. **Centralized core MMU with N ports** — one MMU shared by fetch + all LSU slices, accessed through arbitration. Smaller area, harder timing.

Recommend **(1)** for the first migration pass — minimizes surgery on the new memory routing — then revisit.

### 2.2 New units that touch memory but didn't exist in source
- **DXA** (Decoupled eXecution Accelerator): issues its own `VX_dxa_gmem_req` to global memory. If `VM_ENABLE`, DXA must either translate addresses too, or be restricted to physical addressing (with the runtime ensuring its descriptors hold PAs).
- **TCU**: tensor unit reads/writes via the LSU path; if MMU is at the LSU‑slice level, TCU inherits translation for free.
- **KMU**: kernel manager loads kernel args / CTA descriptors via DCR; not on the data path, no VM impact.

**Decision required for DXA:** translate inside DXA, or require PA descriptors? Recommend **PA descriptors** initially — runtime walks the page table and writes physical addresses into DXA descriptors, mirroring how host‑side DMA is normally programmed.

### 2.3 Config plumbing (TOML → .vh) — easy
Source defines VM macros directly in `VX_config.vh`. Target defines them in `VX_config.toml` and generates `VX_config.vh`. The `[vm]` block is **already there** — just need to flip `VM_ENABLE` and confirm the generator produces the same macro names the source RTL/runtime expects (`MEM_PAGE_SIZE`, `PT_LEVEL`, `PTE_SIZE`, `NUM_PTE_ENTRY`, `TLB_SIZE`, `PT_SIZE_LIMIT`, `PAGE_TABLE_BASE_ADDR`, `VM_ADDR_MODE`).

Add the missing perf‑counter CSR ids to `VX_types.toml`.

### 2.4 CSR & perf counters
Source adds 6 VM perf counters (`VX_CSR_MPM_TLB_READS/HITS/MISSES/EVICTS`, `VX_CSR_MPM_PTW_WALKS/LATENCY`) and an `mmu_perf_t` struct. Target uses a different perf‑stat plumbing (per‑unit `PerfStats` aggregated in `Core::PerfStats`). MMU perf counters need to be **re‑plumbed**, not copied verbatim.

### 2.5 SimX VM scaffolding is orphaned
`MemoryUnit::page_table_walk` exists in `sim/common/mem.cpp` but no one calls it — simx `Processor` uses `RAM` directly via `attach_ram(RAM*)`. Two paths:
1. Wire LSU's data accesses through `MemoryUnit` (which then talks to RAM) so the existing PTW code becomes live.
2. Add a thin VM helper into `sim/simx/lsu_unit.cpp` and `sim/simx/core.cpp` that does translation on the simx side, leaving the orphaned scaffolding alone (or removing it).

Recommend **(1)** — the scaffolding looks complete enough to use; it would be wasteful to write a parallel implementation. We just need to confirm `MemoryUnit` matches the source's address‑translation semantics and wire the simx LSU through it.

### 2.6 Kernel
Source `vx_start.S` writes SATP early in boot. Target `vx_start.S` has a `KMU_ENABLE` branch and no SATP code. The SATP write should be added inside an `#ifdef VM_ENABLE` block, *before* the first memory access that depends on translation. Compatible with both KMU and non‑KMU paths.

### 2.7 Runtime driver
`runtime/common/vm.{h,cpp}` (514 LOC) needs to be ported to `sw/runtime/common/`. The class API (`VMManager::init`, `phy_to_virt_map`, `page_table_walk`, `map_p2v`, `virtual_mem_reserve`) is small and self‑contained — depends only on `mem_alloc.h` (which is already at `sw/common/mem_alloc.h` with the `VM_ENABLE` block in place). Then instantiate it from `sw/runtime/simx/vortex.cpp` and `sw/runtime/rtlsim/vortex.cpp` next to `global_mem_`. OPAE/XRT/stub backends keep current behavior (no VM).

---

## 3. Test inventory & coverage

Source ships **28 regression tests** (driven by `run_vm_regression.sh`, `run_vm_regression_random.sh`, `run_vm_simx_regression.sh`):

```
basic, bfs, conv3, cta, demo, diverge, dogfood, dotproduct, dotproduct2,
dropout, fence, io_addr, jacobi, madmax, mstress, pathfinder, printf,
priority, raycast, relu, sgemm, sgemm2, sgemv, softmax, sort, stencil3d,
vecadd, sgemm_tcu (special — rebuilds with -DEXT_TCU_ENABLE)
```

Target has **48 regression tests**. Mapping:

- **Common (25)** — already present in target, will be exercised: basic, bfs, conv3, demo, diverge, dotproduct, dotproduct2, dropout, fence, io_addr, jacobi, madmax, mstress, pathfinder, printf, raycast, relu, sgemm, sgemm2, sgemv, softmax, sort, stencil3d, vecadd, sgemm_tcu
- **Removed in target (3)** — cta, dogfood, priority. Decision: **drop from VM regression** (they were retired upstream for non‑VM reasons).
- **New in target (~20)** — async_barrier, async_gbarrier, dxa_copy, dxa_multicast, occupancy, packld, wgather, sgemm_tcu_{mx,sp,wg,wg_dxa,wg_sp,wg_sp_dxa}, sgemm_v1, sgemm2_v1, sgemm2_dxa, sgemmx, vecadd_v1, wsync. Decision: **add to VM regression** in a second pass; flag DXA tests as needing the §2.2 PA‑descriptor work first.

Build configuration the source uses:
```
-DVM_ENABLE -DVM_ADDR_MODE=1 -DPERF_ENABLE   # rtlsim
-DVM_ENABLE -DVM_ADDR_MODE=1                 # simx (no PERF_ENABLE)
VORTEX_RANDOMIZE_VA={0,1}, VORTEX_VA_SEED=N
```

Driver: `rtlsim` and `simx`. Three regression scripts to port: deterministic rtlsim, randomized rtlsim, simx.

---

## 4. Proposed migration phases

Each phase ends with a green build (in `build_test32`) and an explicit pass/fail checkpoint. Stop and reassess if a phase doesn't go green.

### Phase 0 — Baseline (½ day)
- Confirm `build_test32` builds clean with **VM_ENABLE = false** (current state).
- Run a couple of regression tests (`vecadd`, `basic`) on simx and rtlsim to establish a known‑good baseline.
- Capture `make` output and a smoke‑test log for diff later.

### Phase 1 — Config & CSR exposure (½ day)
- Set `VM_ENABLE = true` in `VX_config.toml`, regenerate, confirm build still passes (no semantic change yet — orphaned scaffolding compiles in).
- Add VM perf‑counter CSR ids to `VX_types.toml` (mirror source's 6 ids; re‑plumb them later in §3).
- Verify generated `VX_config.vh` exposes every macro the source expects.
- **Checkpoint:** `make` succeeds with `VM_ENABLE=true`; baseline tests still pass (translation is still bypassed because nothing reads SATP yet).

### Phase 2 — SimX VM (1–2 days)
- Wire SimX LSU/fetch through `MemoryUnit` so the existing `page_table_walk` becomes live.
- Replace stub `VX_CSR_SATP` handler in [sim/simx/csr_unit.cpp:53](../../sim/simx/csr_unit.cpp#L53) and [:257](../../sim/simx/csr_unit.cpp#L257) with real `set_satp` / `get_satp` plumbing.
- Port `runtime/common/vm.{h,cpp}` → `sw/runtime/common/vm.{h,cpp}`. Instantiate from `sw/runtime/simx/vortex.cpp`.
- Add SATP write in `sw/kernel/src/vx_start.S` under `#ifdef VM_ENABLE`.
- **Checkpoint:** `vecadd` passes on simx with `VM_ENABLE=true`. Then run the 25‑test common subset on simx (mirrors `run_vm_simx_regression.sh`).

### Phase 3 — RTL MMU (3–5 days, the bulk of the work)
- Port `VX_mmu.sv`, `VX_mmu_tlb.sv`, `VX_mmu_ptw.sv` into `hw/rtl/core/`. Adapt to current `VX_gpu_pkg.sv` types and current memory interfaces (`VX_lsu_mem_if`, `VX_mem_bus_if`).
- Add `mmu_perf_t` to `VX_gpu_pkg.sv`.
- Decide MMU placement (§2.1) — recommend per‑LSU‑slice — and instantiate.
- Plumb fetch translation (icache request path).
- Wire SATP from `csr_unit` / `sfu_unit` to MMU, mirroring source.
- Instantiate `VMManager` in `sw/runtime/rtlsim/vortex.cpp`.
- **Checkpoint:** `vecadd` passes on rtlsim with `VM_ENABLE=true`. Then port `run_vm_regression.sh` to target and run the 25‑test common subset.

### Phase 4 — Randomized VA + perf counters (1 day)
- Port `VORTEX_RANDOMIZE_VA` / `VORTEX_VA_SEED` env vars.
- Re‑plumb the 6 MMU perf counters into target's `Core::PerfStats` flow.
- Port `run_vm_regression_random.sh`.
- **Checkpoint:** randomized regression at a fixed seed is reproducible and 25/25 pass.

### Phase 5 — New tests (DXA / TCU / async) (timeline TBD)
- TCU tests: should work for free if MMU sits at LSU slice level.
- DXA tests: need §2.2 decision implemented (PA descriptors).
- Async barrier / wsync / occupancy / packld / wgather: should be VM‑agnostic; verify.

### Phase 6 — Documentation & cleanup
- Port `VM_rand_notes.md` content into `docs/`.
- Decide whether to remove or keep DXA bypass for VM (depending on §2.2 outcome).

---

## 5. Risks & open questions

1. **`MemoryUnit` semantic match** — its `page_table_walk` was likely added by whoever did the partial merge. Need to diff it line‑by‑line against `vortex_vm/sim/common/mem.cpp` PTW logic before trusting it. If it's a stale fork, Phase 2 grows.
2. **Per‑slice MMU area cost on FPGA** — fine for simulation, may not synth on smaller targets. Phase 3 should include a quick `make` of a small AFU config to sanity‑check Verilator/Vivado.
3. **DXA + VM interaction** — §2.2 is genuinely a design decision, not a port. Needs your call before Phase 5.
4. **Test diff** — 3 retired tests (cta/dogfood/priority) — confirm OK to drop from VM regression.
5. **`-DPERF_ENABLE` skew** — source enables PERF only on rtlsim, not simx. Target may have a different perf‑build flag; confirm both build configs work before regression scripts are ported.
6. **Branch hygiene** — current target branch is `tinebp-patch-2`, not `feature_vm`. Should this work be done on a new branch (`feature_vm`) cut from `tinebp-patch-2`? Recommend yes.

---

## 6. What I need from you to proceed

- **Approve §2.1** — per‑LSU‑slice MMU vs centralized.
- **Approve §2.2** — DXA gets PA descriptors, or DXA gets its own translator.
- **Confirm §3 test set decisions** — drop cta/dogfood/priority from VM regression; defer DXA tests to Phase 5.
- **Confirm §5.6** — should I cut a new `feature_vm` branch in `~/dev/vortex_v3/feature_vm` for this work, or keep committing to `tinebp-patch-2`?
- Once approved, I'll start Phase 0 and check in after each phase checkpoint.

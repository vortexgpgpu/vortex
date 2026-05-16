**Date:** 2026-05-09
**Status:** Draft — Phase 0 in progress (hip_vortex pruning, scaffolding)
**Author:** Blaise Tine
**Related:**
[wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md),
[master_merge_v3_proposal.md](master_merge_v3_proposal.md).

### Update history

- **2026-05-09** — Initial draft.
  - `git rm -r vortex hipcc` staged in `~/dev/hip_vortex` (Phase 0).
  - Reference tests landed at
    [tests/hip/vecadd/](../../tests/hip/vecadd/) and
    [tests/hip/sgemm/](../../tests/hip/sgemm/) with the Phase-2
    Makefile shape (compiles once `hipcc-vortex` exists).
  - `hip` registered as an opt-in target in
    [tests/Makefile](../../tests/Makefile) (excluded from default
    `all` until the toolchain lands).
- **2026-05-09 (amendment)** — Scope correction: the **full HIP
  s/w stack stays in `~/dev/hip_vortex`**. The Vortex `sw/` tree is
  not touched. Component layout (§4.2), Phase 1, and the
  Makefile defaults under [tests/hip/common.mk](../../tests/hip/common.mk)
  updated to install from hip_vortex into `$(TOOLDIR)/hip-vortex`.
- **2026-05-09 (rename)** — User renamed three sibling repos:
  `amd_hip` → `hip_vortex`, `vx_llvm` → `llvm_vortex`,
  `vx_pocl` → `pocl_vortex`. Install destination uses hyphenated
  form: `$(TOOLDIR)/hip-vortex`. Proposal text and
  [tests/hip/common.mk](../../tests/hip/common.mk) updated.
  Layout convention: `bin/` is a build/install output, not part
  of the source tree — `hipcc-vortex` source lives under
  `hip_vortex/hipcc/` and is installed to `$(TOOLDIR)/hip-vortex/bin/`
  by CMake.

# HIP Support — Proposal

## 1. Summary

Add native HIP API support to Vortex by extending the existing
`llvm_vortex` Clang HIP frontend with a Vortex offload toolchain
(`HIPVortex.cpp`) and shipping a thin runtime
(`libhip_vortex.so`) that wraps `vortex_runtime`. The end goals are
twofold:

1. **HIP-as-programming-surface for Vortex hardware extensions** —
   expose WMMA, WGMMA, TMA, and async barrier through HIP-shaped
   headers, in the same idiom NVIDIA uses for `nvcuda::wmma` /
   `cuda::barrier` and AMD uses for `__builtin_amdgcn_wmma_*` /
   `rocwmma`.

2. **MLIR-based middleware as compiler-research surface** — use
   `llvm_vortex`'s shipped MLIR libraries to host research passes
   (autotuning, fusion, layout selection, async pipelining) between
   HIP source and the Vortex backend, via a new out-of-tree
   `Vortex` MLIR dialect.

This proposal covers what we keep, what we drop, what we add, and
in what order. Compiler infrastructure is plumbing; the new
research surface is the `Vortex` MLIR dialect and its lowerings.

The compilation pipeline adopted is **Shape 1**: stock Clang HIP
frontend → LLVM IR → MLIR middleware (`vortex-opt`) → `llvm-vortex`
backend → fat ELF + `libhip_vortex.so`. The user-facing entry
point is a `hipcc-vortex` Python wrapper script that orchestrates
the pipeline; no clang plugin, no clang fork.

---

## 2. Background — three prior efforts

Before this proposal there were three attempts at HIP-on-Vortex,
none of which fully landed:

| Effort | Approach | Status (2026-05) | Disposition |
|---|---|---|---|
| `~/dev/hip_vortex` | Fork ROCm hipcc + HIP runtime + own Vortex tree | Stub only — `hiprt/src/hip.cpp` is 65 lines, only `hipMalloc` implemented; `hipMemcpy`/`hipModuleLaunchKernel` are TODOs | **Salvage `hiprt/include/` API surface and the `HipDevice` singleton pattern; drop `vortex/` and `hipcc/`.** |
| `~/dev/vortex_hip` | Polygeist (HIP→MLIR) + custom `GPUToVortexLLVM` pass + `llvm-vortex` (LLVM 10) | Phase 1 metadata complete; Phase 2B at 70% | **Architecture is the right shape but built on an obsolete LLVM. Re-host the MLIR work on `llvm_vortex` (LLVM 18.1); salvage Phase 1 metadata and Phase 3 runtime work as references.** |
| chipStar + `pocl_vortex` route | HIP → SPIR-V via `llvm_vortex`'s `HIPSPV.cpp` → `pocl_vortex` Vortex device | Untested end-to-end; `pocl_vortex` has SPIR-V parsing (`ENABLE_SPIRV=ON`) but the Vortex device path is unvalidated | **Secondary path for unmodified-HIP-app portability; not the primary investment. See risk #4.** |

The current proposal supersedes `hip_vortex` and re-hosts the MLIR
work from `vortex_hip` onto the modern `llvm_vortex` tree.

---

## 3. Inheriting from hip_vortex

### 3.1 Kept

The HIP s/w stack stays in `~/dev/hip_vortex`. The Vortex `sw/`
tree is not touched. Only test examples and design docs land in
`feature_hip`.

| Path in `~/dev/hip_vortex` | Disposition | Notes |
|---|---|---|
| `hiprt/include/hip/*.h` | **Stays in place** (possibly upstream-tracked) | HIP API headers — currently a copy of ROCm's headers. May be re-pointed at an upstream HIP-headers reference (chipStar's `chipspv-headers` or ROCm's `hipother` / `hipamd-headers`) to avoid carrying a stale fork; this is a Phase 1 sub-decision, not a relocation. |
| `hiprt/src/hip.cpp` | **Rewritten in place** | The `HipDevice` singleton + `hipMalloc → vx_mem_alloc` mapping pattern is salvaged as the starting shape; the rest of the HIP API surface is unimplemented in `hip_vortex` and is the bulk of Phase 1 below. Output: `libhip_vortex.so` built from `hip_vortex/hiprt/src/`. |
| `bin/hipcc-vortex` | **New** | Python driver wrapper, lives under `hip_vortex/bin/`. Replaces the deleted Perl `hipcc/` (which was the AMD ROCm fork). |
| `dogfood/vadd.cpp` | **Reference-only** in hip_vortex; an evolved version landed in `feature_hip` | The feature_hip copy at [tests/hip/vecadd/main.cpp](../../tests/hip/vecadd/main.cpp) is modernized to match feature_hip's existing test idioms (CLI args, FLOAT_ULP comparison, PASSED/FAILED reporting, `hipDeviceSynchronize`). |
| `dogfood/Makefile` | **Reference-only** in hip_vortex; replaced by [tests/hip/common.mk](../../tests/hip/common.mk) | feature_hip's copy uses the standard out-of-tree-build pattern (mirrors [tests/opencl/common.mk](../../tests/opencl/common.mk)) and points `HIP_INSTALL_PATH` at `$(TOOLDIR)/hip-vortex` (the install destination of the built hip_vortex stack). |

### 3.2 Dropped (deleted from hip_vortex)

| Removed | Why |
|---|---|
| `hip_vortex/vortex/` | Stale Vortex tree from a 2020-era checkpoint; superseded by current `vortex_v3` and `llvm_vortex`. Carrying it adds nothing. |
| `hip_vortex/hipcc/` | AMD's `hipcc` is a Perl wrapper hard-wired to AMDGPU offload-arch values, ROCm paths, AMD-specific clang flags, and the AMD bundler. Forking and stripping AMD-specific bits is more work than writing a small Python wrapper for Vortex; we replace it with `hipcc-vortex` (Phase 2). |

Both removals are staged via `git rm -r vortex hipcc` in `hip_vortex`
as part of Phase 0 (this proposal). The remaining `hiprt/` and
`dogfood/` directories in `hip_vortex` stay; `hiprt/` becomes the
home of `libhip_vortex.so`, `bin/hipcc-vortex` is added in
Phase 2.2.

---

## 4. Target architecture

### 4.1 Compilation pipeline (Shape 1)

```
foo.hip
   │
   ▼  hipcc-vortex  (Python wrapper)
   │
   ├─ clang -x hip --offload-arch=vortex foo.hip
   │       (HIPVortex.cpp toolchain in llvm_vortex)
   │     ↳ host pass     →  host.bc
   │     ↳ device pass   →  device.bc      (RISC-V triple + +vortex attrs)
   │
   ├─ mlir-translate --import-llvm device.bc        →  device.mlir   (LLVM dialect)
   ├─ vortex-opt --convert-llvm-to-gpu              →  device.gpu.mlir
   │             --convert-gpu-to-vortex            →  device.vx.mlir
   │             --convert-vortex-to-llvm
   │             [--research-passes …]
   ├─ mlir-translate --mlir-to-llvmir               →  device.lowered.ll
   ├─ llc -mtriple=riscv32 -mattr=+vortex           →  device.o
   ├─ vxbin.py                                       →  device.vxbin
   ├─ clang-offload-bundler  (host.bc + device.vxbin)  →  fat.o
   └─ clang link fat.o + libhip_vortex.so           →  foo executable
```

The MLIR stage is initially a no-op pass-through (Phase 2);
Phase 3 stands up the `Vortex` dialect and lowerings; Phase 4
adds extension ops. Each phase is independently shippable.

### 4.2 Component layout

Components split across three trees: `llvm_vortex` for compiler bits
that have to live inside Clang, `~/dev/hip_vortex` for the entire
HIP s/w stack (runtime, headers, wrapper script, build infra), and
the `feature_hip` Vortex tree for tests + this proposal only. The
out-of-tree MLIR dialect is its own sibling repo (see §6).

| Component | Path | Source / size |
|---|---|---|
| HIPVortex Clang toolchain | `llvm_vortex/clang/lib/Driver/ToolChains/HIPVortex.{cpp,h}` | New (~500–1000 LOC, modeled on `HIPSPV.{cpp,h}`). Lives in `llvm_vortex` because Clang toolchains cannot be loaded out-of-tree. |
| HIP device-libs | `llvm_vortex/clang/lib/Headers/__clang_hip_vortex_*.h` + bitcode | New (medium, bounded by test surface). Same reason: must be inside Clang. |
| HIP runtime headers | `~/dev/hip_vortex/hiprt/include/hip/` | Already exists (ROCm-derived). Possibly upstream-tracked later. |
| HIP runtime library | `~/dev/hip_vortex/hiprt/src/` → `libhip_vortex.so` | Rewritten on top of stub. |
| HIP runtime build infra | `~/dev/hip_vortex/Makefile` or `CMakeLists.txt` (new) | Builds + installs `libhip_vortex.so`, `hipcc-vortex`, headers into `$(TOOLDIR)/hip-vortex/{lib,bin,include}`. |
| `hipcc-vortex` driver wrapper | `~/dev/hip_vortex/bin/hipcc-vortex` (Python) | New, ~200 LOC. Replaces the deleted AMD Perl `hipcc/`. |
| Toolchain installer hook | [ci/toolchain_install.sh.in](../../ci/toolchain_install.sh.in) | Extend to clone `~/dev/hip_vortex`, build it, and install into `$(TOOLDIR)/hip-vortex`. |
| `Vortex` MLIR dialect | Out-of-tree project (e.g. `~/dev/vortex_mlir`); depends on installed `llvm_vortex` | New (out-of-tree; see §6). Separate from hip_vortex. |
| `vortex-opt`, `vortex-translate` | Same out-of-tree project | New |
| Reference HIP tests | [tests/hip/{vecadd,sgemm}/](../../tests/hip/) | **Already landed** |
| Test aggregator | [tests/hip/{Makefile,common.mk}](../../tests/hip/) | **Already landed**, defaults `HIP_INSTALL_PATH := $(TOOLDIR)/hip-vortex` |
| Test suite registration | [tests/Makefile](../../tests/Makefile) | **Already wired** as opt-in `hip` target |
| CI hook | [ci/regression.sh.in](../../ci/regression.sh.in) | Add `--hip` suite gated on `$(TOOLDIR)/hip-vortex` existing (Phase 2) |

---

## 5. Phases

Each phase is independently buildable and testable. Phases 1–3 are
sequential; Phase 4 (extension exposure) parallelizes once Phase 2
is unblocked.

### Phase 0 — Pruning & scaffolding ✅ in progress

- [x] `git rm -r vortex hipcc` staged in `~/dev/hip_vortex`.
- [x] Reference tests under [tests/hip/{vecadd,sgemm}/](../../tests/hip/).
- [x] Test aggregator [tests/hip/Makefile](../../tests/hip/Makefile) + [common.mk](../../tests/hip/common.mk).
- [x] Opt-in `hip` target in [tests/Makefile](../../tests/Makefile).
- [x] This proposal at `docs/proposals/hip_support_proposal.md`.

### Phase 1 — HIP runtime stub on vortex_runtime

Implement `libhip_vortex.so` in `~/dev/hip_vortex/hiprt/src/` (replacing
the existing 65-line stub). Minimum API surface to run the
reference tests:

- `hipMalloc` / `hipFree` — wrap `vx_mem_alloc` / `vx_mem_free`
  (already sketched in `hip_vortex`'s `HipDevice` singleton).
- `hipMemcpy` / `hipMemcpyAsync` — wrap `vx_copy_to_dev` /
  `vx_copy_from_dev`.
- `hipDeviceSynchronize` — wrap `vx_ready_wait`.
- `hipGetDeviceProperties`, `hipGetDeviceCount` — minimal
  device-info shim.
- `hipStream_t`, `hipEvent_t` — stub against single-queue model
  initially; real streams in Phase 2 follow-up.
- `hipModuleLoad` / `hipModuleGetFunction` /
  `hipModuleLaunchKernel` — load `.vxbin` from a fat-binary blob,
  dispatch via `vx_start`.
- `__hipRegisterFatBinary` / `__hipRegisterFunction` /
  `__hipRegisterVar` — invoked by host code emitted by Clang to
  register device kernels (these are the host-side glue clang
  expects).
- `hipLaunchKernelGGL` — header-side macro/template wrapper over
  `hipModuleLaunchKernel`; the runtime side is just kernel lookup.
- `hipGetErrorString` — required by [tests/hip/*](../../tests/hip/) error path.

**Validation**: a hand-compiled vecadd device kernel + hand-written
host code linking against `libhip_vortex.so` runs end-to-end on
SimX. No HIP frontend / fat-binary involvement yet — the kernel
binary is loaded from a side file.

### Phase 2 — HIPVortex Clang toolchain

Add `llvm_vortex/clang/lib/Driver/ToolChains/HIPVortex.{cpp,h}`,
modeled closely on `HIPSPV.{cpp,h}`:

- Register `--offload-arch=vortex` (and `vortex32`/`vortex64`).
- Set up the device compilation action with
  `riscv32-unknown-elf` / `riscv64-unknown-elf` triple +
  `-mattr=+vortex,+zicond,…` (see
  [tests/opencl/common.mk](../../tests/opencl/common.mk) for the
  current Vortex `-march`/`-mattr` matrix).
- Pull in HIP device-libs (Phase 2.1).
- Hand off to `clang-offload-bundler` for fat-binary assembly.
- Plumb `__hipRegisterFatBinary` etc. into the host pass.
- Add Vortex-aware bundle-target identifier
  (`hip-riscv32-unknown-elf-vortex` / `…-riscv64-…`).

**Phase 2.1 — HIP device-libs for Vortex.** Implement the
`__ockl_*`, math, sync, atomic, and builtin-variable functions
(`threadIdx.x`, `blockIdx.x`, `blockDim.x`, `__syncthreads()`, …)
as a bitcode library that lowers to `vx_*` builtins. This is the
largest mechanical chunk of Phase 2 and the most likely place to
discover gaps in the existing
[VortexIntrinsicFunc.cpp](../../../llvm_vortex/llvm/lib/Target/RISCV/VortexIntrinsicFunc.cpp).

**Phase 2.2 — `hipcc-vortex` wrapper.** Python script in
`~/dev/hip_vortex/bin/`. Drives clang + bundler for now; the MLIR
stage is inserted in Phase 3. Installed to `$(TOOLDIR)/hip-vortex/bin/`
by [ci/toolchain_install.sh.in](../../ci/toolchain_install.sh.in).

**Validation**: [tests/hip/vecadd/main.cpp](../../tests/hip/vecadd/main.cpp)
compiles through `hipcc-vortex` and runs end-to-end on SimX and
RTLsim. A `--hip` suite is added to
[ci/regression.sh.in](../../ci/regression.sh.in).

### Phase 3 — MLIR middleware

Stand up an out-of-tree project `vortex_mlir` (separate repository
under `~/dev/`). Depends on installed `llvm_vortex` via
`find_package(MLIR REQUIRED CONFIG)`. Contents:

- `Dialect/Vortex/` — initial op set: `vortex.thread_id`,
  `vortex.block_id`, `vortex.block_dim`, `vortex.grid_dim`,
  `vortex.barrier_local`, `vortex.atomic_*`,
  `vortex.shared_alloc`. (Hardware-extension ops follow in
  Phase 4.)
- `Conversion/GPUToVortex/` — lowering from `gpu` dialect ops to
  `vortex` dialect.
- `Conversion/VortexToLLVM/` — lowering from `vortex` to LLVM
  dialect, emitting `@llvm.vx.*` intrinsic calls.
- `Tools/vortex-opt/` — driver tool registering the upstream
  dialects + `vortex`.
- `Tools/vortex-translate/` — `mlir-translate` clone with the
  Vortex lowering registered.
- Lit/FileCheck tests under `test/`.

**Wiring into `hipcc-vortex`.** After Phase 2 the wrapper goes
`clang → llc`. Phase 3 inserts
`mlir-translate --import-llvm | vortex-opt | mlir-translate --mlir-to-llvmir`
between `clang` and `llc`, behind `--mlir-pipeline` (default-on
once stable).

**Research surface.** All custom passes (autotuning, layout
selection, async pipelining, fusion) live as additional pipeline
stages inside `vortex-opt` and are gated by command-line flags so
the default pipeline stays predictable.

**Validation**: [tests/hip/sgemm/main.cpp](../../tests/hip/sgemm/main.cpp)
compiled through the MLIR pipeline matches the LLVM-direct path
within `FLOAT_ULP=6` and passes SimX/RTLsim.

### Phase 4 — Hardware extension exposure

For each Vortex extension (WMMA, WGMMA, TMA, async barrier),
land four small artifacts in this order:

1. **LLVM intrinsic.** `@llvm.vx.<ext>.*` declaration + lowering
   in `llvm_vortex/llvm/lib/Target/RISCV/`.
2. **MLIR dialect op.** `vortex.<ext>` in the Phase 3 project.
3. **MLIR lowering.** `gpu.subgroup_mma_*` (or new high-level
   `vortex.<ext>` op) → `vortex.<ext>` → `@llvm.vx.<ext>.*`.
4. **HIP header.** `~/dev/hip_vortex/hiprt/include/hip/vortex_<ext>.h`
   exposing the extension in a `nvcuda::wmma`-style C++ API
   (e.g. `vortex_wmma.h`, `vortex_tma.h`,
   `vortex_async_barrier.h`).

Each extension is one PR; decoupled from Phases 1–3.

---

## 6. Out-of-tree vs. in-tree for the MLIR dialect

**Recommendation: out-of-tree**, in a sibling repo (e.g.
`~/dev/vortex_mlir`). Rationale:

- Vortex-specific dialects will not be upstreamed to MLIR
  (vendor-specific). In-tree means carrying an MLIR fork forever.
- Out-of-tree is the modern default — Triton, IREE, AdaptiveCpp,
  TPP-MLIR, CIRCT (mostly) all live outside the LLVM monorepo.
- Decouples MLIR/LLVM rebases from Vortex research velocity.
- Upstream template at `llvm_vortex/mlir/examples/standalone/` is the
  starting scaffold.

The dialect project does `find_package(MLIR REQUIRED CONFIG)`
against an installed `llvm_vortex` and produces:

- `libVortexDialect.{a,so}` — dialect + passes.
- `vortex-opt`, `vortex-translate` — standalone executables.
- A CMake export package consumable by `hipcc-vortex`.

Linking mode: shared (`LLVM_BUILD_LLVM_DYLIB=ON`,
`LLVM_LINK_LLVM_DYLIB=ON`) for development velocity; static
optional for shipping.

---

## 7. Test plan

| Stage | Test | Driver | Pass criterion |
|---|---|---|---|
| Phase 1 | hand-built vecadd, hand-loaded `.vxbin` | SimX | numerical match against CPU reference |
| Phase 2 | [tests/hip/vecadd](../../tests/hip/vecadd/) via `hipcc-vortex` | SimX, RTLsim | `PASSED!` |
| Phase 2 | [tests/hip/sgemm](../../tests/hip/sgemm/) via `hipcc-vortex` | SimX, RTLsim | `PASSED!` |
| Phase 3 | both tests via `--mlir-pipeline` | SimX, RTLsim | bit-equivalent or within `FLOAT_ULP=6` of Phase-2 path |
| Phase 4 (per ext.) | new `tests/hip/<ext>_*` (e.g. `wmma_smoke`) | SimX, RTLsim | extension intrinsic exercised + numerical check |

CI integration follows the gfx_migration pattern (see
[gfx_migration_proposal.md §4](gfx_migration_proposal.md)): a
`--hip` suite in [ci/regression.sh.in](../../ci/regression.sh.in),
gated on the toolchain being installed, landing in Phase 2.

---

## 8. Risks & open questions

1. **HIP language mode + RISC-V triple.** Clang's HIP mode has
   been used with AMDGPU and SPIR-V triples. Using a RISC-V
   triple as a HIP device target is novel and may surface
   assumptions in `clang/lib/Sema/SemaCUDA.cpp` or the offload
   action graph. `HIPSPV.cpp` is the closest precedent (a
   non-AMDGPU HIP triple). **Risk: medium.** **Mitigation**:
   prototype the toolchain end-to-end on an empty kernel before
   starting Phase 2.1 (device-libs work).

2. **HIP device-libs scope.** The `__ockl_*` surface is large;
   we will only implement what the reference tests touch.
   **Risk: low** — bounded by the test suite, grows incrementally
   per feature.

3. **MLIR API style for extensions (Phase 4).** Three choices:
   mimic `nvcuda::wmma` (best CUTLASS-portability), mimic
   `rocwmma` (most HIP-native), or a Vortex-native `vx::tensor`
   namespace. Decision deferred to start of Phase 4; recommended
   default is NVIDIA-style for research-kernel portability.

4. **chipStar fallback.** `pocl_vortex` already has SPIR-V plumbing
   (`spirv_parser.cc`, `ENABLE_SPIRV=ON`); whether the
   `lib/CL/devices/vortex/` device target ingests SPIR-V
   end-to-end is untested. A one-day spike (compile vecadd via
   `llvm_vortex`'s `HIPSPV.cpp` and load through `pocl_vortex`) is the
   cheapest validation; the result informs whether to invest in
   chipStar+`pocl_vortex` as a parallel secondary path or close that
   door.

5. **LLVM tracking cadence.** `llvm_vortex` currently pins LLVM 18.1
   (last commit 2025-08-08). The MLIR dialect project will pin
   the same llvm_vortex commit. Rebases land as quarterly chunks
   coordinated with llvm_vortex bumps.

6. **`hip_vortex` ownership.** `hip_vortex` becomes the canonical home
   of the HIP s/w stack going forward (runtime, headers,
   wrapper). Repository naming is misleading — it currently
   references AMD HIP rather than Vortex HIP — but renaming is
   deferred to avoid churn during active development. Phase 1
   lands here, not under Vortex `sw/`.

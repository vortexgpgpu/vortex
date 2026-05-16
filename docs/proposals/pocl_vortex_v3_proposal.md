**Date:** 2026-05-09
**Status:** Draft — Phase 0 not yet started
**Author:** Blaise Tine
**Related:**
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md),
[hip_support_proposal.md](hip_support_proposal.md),
[wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md),
[master_merge_v3_proposal.md](master_merge_v3_proposal.md).

### Update history

- **2026-05-09** — Initial draft. Supersedes the (now incorrect)
  Step 0 of [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md):
  forensics established that `vortex_2.x` is already at POCL 6.0,
  not 4.0, and that the POCL-side prerequisites for chipStar are
  already in tree — but the **Vortex device target itself has no
  SPIR-V code path** and still drives kernels via the legacy
  `spawn_thread` model. A redesign on a cleaner upstream base is
  the right unit of work.

# pocl_vortex v3 — Redesign Proposal

## 1. Summary

Redesign `pocl_vortex` from a re-imported, history-disconnected
fork into a small, well-structured layer on top of upstream POCL,
addressing four concrete needs:

1. **Wire the SPIR-V code path** into the Vortex device target —
   the gating prerequisite for [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md).
2. **Migrate the kernel-launch model** from the legacy
   `spawn_thread` runtime to the Vortex v3 KMU dispatcher.
3. **Re-establish git history with upstream POCL** so future
   updates are tractable rebases instead of forensic ports.
4. **Pull in POCL 5.0/6.0 improvements** that are present in the
   shared POCL code today but not consistently exercised by the
   Vortex device target (`cl_ext_buffer_device_address`, the
   shared `spirv_parser`, command-buffer hooks, modern
   device-ops API).

Plus a fifth, recommended:

5. **Establish hygiene** — README, CI, branch policy, and a path
   toward eventually upstreaming the Vortex device target.

The redesign re-bases onto `upstream/release_6_0` (where the fork
is *content-wise* already approximately situated) with a clean,
small, well-named series of Vortex-specific commits on top.

---

## 2. Current state — what the forensics found

Cross-referenced against the [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
recon (steps not yet executed):

| Property | Current `vortex_2.x` |
|---|---|
| Branch | `vortex_2.x` (origin: `vortexgpgpu/pocl`) |
| HEAD | `2ae6d4977` (2026-01-13) |
| Stated POCL version | "4.1 Unreleased" in `CHANGES` (**stale, misleading**) |
| Closest upstream branch | `upstream/release_6_0` — 108 files differ |
| Merge-base with upstream | **empty** (history was re-imported, severed) |
| `cl_ext_buffer_device_address` | ✅ in shared POCL code |
| `lib/CL/devices/spirv_parser.{cc,hh}` | ✅ present |
| Vortex device target SPIR-V code path | ❌ **none** — `pocl-vortex.c` has zero SPIR-V mentions |
| Vortex device kernel launch model | `spawn_thread` (`kernel_main.c` / `kernel_args.h`) — predates Vortex v3 KMU |
| `README.vortex` | ✅ exists, **stale** (refers to spawn-thread model) |
| Last upstream sync | unknown — no shared history |

Vortex-specific files (the layer we'd preserve):

```
README.vortex
lib/CL/devices/vortex/CMakeLists.txt
lib/CL/devices/vortex/kernel_args.h
lib/CL/devices/vortex/kernel_main.c
lib/CL/devices/vortex/pocl-vortex.{c,h}
lib/CL/devices/vortex/vortex_utils.{cc,h}
lib/kernel/vortex/CMakeLists.txt
lib/kernel/vortex/atomics.c
lib/kernel/vortex/barrier.c
lib/kernel/vortex/printf.c
lib/kernel/vortex/workitems.c
```

Plus targeted edits in `lib/CL/devices/CMakeLists.txt`,
`lib/CL/devices/devices.c`, the top-level `CMakeLists.txt`, and a
handful of shared headers — all small, all identifiable by
`grep -i vortex`.

---

## 3. Goals

### 3.1 — chipStar prerequisites: SPIR-V code path in the Vortex device

The shared `lib/CL/devices/spirv_parser.cc` already lowers SPIR-V
into the LLVM IR that the rest of the build pipeline expects. The
gap is that **`pocl-vortex.c`'s `post_build_program` /
`build_program` device hooks don't route SPIR-V input through it**.
chipStar will issue `clCreateProgramWithIL(...)` with a SPIR-V
binary; the current Vortex device returns
`CL_INVALID_PROGRAM_EXECUTABLE` (or similar) because that path is
unwired.

**Concretely**:
- Add a `build_program_with_il` driver-ops entry (or wire into
  `post_build_program` if that's the unified hook in 6.0).
- Call into the shared `spirv_parser` to emit LLVM IR.
- Hand off to `vortex_utils.cc`'s existing LLVM-IR-to-Vortex
  lowering.
- Advertise `cl_khr_il_program` and (where possible)
  `cl_ext_buffer_device_address` in `clGetDeviceInfo` for the
  Vortex device.

Reference precedent: `lib/CL/devices/cuda/pocl-cuda.c` and
`lib/CL/devices/level0/pocl-level0.cc` both do exactly this; the
Vortex implementation can crib structure from them.

### 3.2 — Migrate from `spawn_thread` to Vortex v3 KMU

The current Vortex device's `kernel_main.c` / `kernel_args.h`
implement a software-managed launch model based on
`vx_spawn_threads`-style runtime calls (per-warp, per-core
dispatch managed in software). Vortex v3's **KMU** (Kernel
Management Unit, see [wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md))
moves this responsibility into hardware: the runtime hands the
KMU a kernel descriptor + grid/block dimensions and the KMU
itself handles work distribution.

**Concretely**:
- Replace `kernel_main.c`'s spawn-thread bootstrap with a KMU
  descriptor build + `vx_start_g` (or successor) call.
- Update `kernel_args.h` arg layout to whatever the v3 KMU expects
  (likely flat-buffer-of-kernel-args + per-kernel header — confirm
  with current vortex_v3 docs).
- Audit `lib/kernel/vortex/{atomics,barrier,printf,workitems}.c`
  for any spawn-thread assumptions and update.
- Existing Vortex tests should pass on the new launch path
  before SPIR-V/chipStar work begins.

This is the largest piece of work in the proposal. The KMU API
itself is in flux — coordination with the v3 RTL/SimX teams is
required to pin the dispatch interface before the migration
starts.

### 3.3 — Re-establish history connection with upstream POCL

Because `vortex_2.x` was re-imported at some point, `git rebase`
is not currently a viable update strategy. The redesign fixes
this: the new branch starts as `git checkout upstream/release_6_0`
and accumulates Vortex changes as **discrete commits with real
parent hashes from upstream**. After this, `git rebase
upstream/release_6_1` (or 7.0) becomes a normal, conflict-bounded
operation.

**Concretely**:
- New branch `vortex_3.x` based on `upstream/release_6_0`.
- Each Vortex-specific commit is small, well-named, and bracketed
  by tests.
- The commit graph is preserved through rebases.
- A documented `git remote add upstream` + rebase recipe in
  `README.vortex` so the next maintainer doesn't lose history again.

### 3.4 — Pull in POCL 5.0/6.0 improvements the Vortex device misses

Many features have landed upstream that the Vortex device target
doesn't currently exercise even though the underlying support is
in shared code:

| Feature | Upstream version | Why we want it |
|---|---|---|
| Modern device-ops API (renames, new hooks) | 4.0 → 6.0 | Required for any further upstream merge |
| Generic address spaces in CPU drivers (SPIR-V relevance) | 4.0 | Lets SPIR-V kernels with generic AS pointers compile |
| `cl_ext_buffer_device_address` | 6.0 | chipStar's `hipMalloc` works without SVM |
| `cl_khr_command_buffer` (basic) | 3.1 → 6.0 | HIP graphs / command buffers map to this |
| `cl_intel_unified_shared_memory` | 4.0 | SYCL / DPC++ workloads via chipStar |
| `cl_khr_subgroups` (initial) | 4.0 | Required by HIP `__shfl_*` and many CUTLASS-style kernels |
| `LLVMSPIRVLib` library alternative to `llvm-spirv` binary | 7.0 | Cleaner integration; deferred (we stay on 6.0) |

Most are advertised correctly in shared POCL code; the Vortex
device just needs to opt in via its `clGetDeviceInfo` extension
string and (for some) implement the device-side hook.

### 3.5 — Cleaner future rebase hygiene

Achieved by Goal 3.3 (history reconnection) plus:

- **Minimize shared-file edits.** Push as much Vortex code as
  possible into `lib/CL/devices/vortex/` and `lib/kernel/vortex/`.
  Where shared-file edits are unavoidable (device registration
  in `lib/CL/devices/CMakeLists.txt`, etc.), keep them small,
  guarded by `if(ENABLE_VORTEX)`, and grep-friendly.
- **Pin to a release branch, not main.** Track
  `upstream/release_6_0` (later 6.1 / 7.0). Point releases land
  bugfixes only; no large API churn between point releases.
- **Document the rebase recipe.** Anyone (including future me)
  should be able to do
  `git fetch upstream && git rebase upstream/release_6_0`
  on a routine schedule and produce a buildable result.

### 3.6 — Other recommendations

Beyond the user's stated goals:

- **Update `README.vortex`.** Currently describes the spawn-thread
  model. Replace with: KMU-based launch, SPIR-V code path,
  build/install instructions against `llvm_vortex`, pointer to
  this proposal.
- **Add CI for the Vortex device target.** Even a minimal
  `.github/workflows/build_vortex.yml` that builds against
  `llvm_vortex` and runs a vecadd smoke test would catch
  regressions earlier than the current "build and test only when
  the user remembers" pattern.
- **Vortex-specific test list under `tests/vortex/`.** A small
  set of OpenCL kernels chosen for Vortex coverage (basic, SPIR-V
  variant, KMU edge cases). Mirrors the existing `tests/regression/`
  pattern but Vortex-focused.
- **Plan toward upstream contribution.** The Vortex device target
  has been a private fork for years. Once it's clean (post this
  redesign), pursuing a `pocl/pocl` PR adding it as an experimental
  device target is realistic. Out of scope for this proposal but
  worth keeping the door open — keep code style upstream-friendly.

---

## 4. Target architecture

```
HIP (via chipStar) / SYCL / OpenCL-C source
   │
   ▼  Clang or compile-time tool produces SPIR-V or OpenCL-C
   │
POCL frontend (clBuildProgram, clCreateProgramWithIL)
   │
   ▼  Vortex device hooks (post_build_program / build_program_with_il)
   │     ├── OpenCL-C path: existing flow (untouched)
   │     └── SPIR-V path:   spirv_parser → LLVM IR
   │                              ↓
   │                       vortex_utils.cc lowering (LLVM IR → Vortex .vxbin)
   │
   ▼  Kernel launch via KMU dispatcher (replaces spawn_thread)
   │
Vortex hardware (RTL / SimX)
```

The two flows (OpenCL-C and SPIR-V) converge at LLVM IR; from
there `vortex_utils.cc`'s existing lowering carries the binary
through. KMU dispatch handles all kernel invocations regardless
of source language.

---

## 5. Phases

Each phase is independently buildable and testable. Phase 1 is
the chipStar-unblocking minimum.

### Phase 0 — Branch creation & inventory  (2–3 days)

- New branch `vortex_3.x` from `upstream/release_6_0`.
- Identify the *true* Vortex-specific changes in `vortex_2.x` by
  grepping for "vortex" / `ENABLE_VORTEX` mentions in shared files.
- Cherry-pick Vortex-specific files (`lib/CL/devices/vortex/*`,
  `lib/kernel/vortex/*`, `README.vortex`) onto `vortex_3.x`.
- Re-apply shared-file Vortex registration edits as one or two
  small commits.
- Build green against `llvm_vortex` (LLVM 18.1) with
  `-DENABLE_VORTEX=ON -DENABLE_SPIRV=ON`.
- Run the existing OpenCL vecadd test through `vortex_3.x`'s
  Vortex device on SimX to confirm baseline parity with `vortex_2.x`.

**Exit criterion**: vecadd OpenCL-C test passes on `vortex_3.x`,
matching `vortex_2.x` numerically.

### Phase 1 — Wire SPIR-V code path  (3–5 days)

- Add `build_program_with_il` (or equivalent) hook to
  `pocl-vortex.c`.
- Route SPIR-V → LLVM via shared `spirv_parser`.
- Connect to `vortex_utils.cc` lowering.
- Advertise `cl_khr_il_program` in Vortex device extensions.
- Add a SPIR-V variant of the vecadd OpenCL test (compile
  `kernel.cl` to `kernel.spv` via `clang + llvm-spirv`, load via
  `clCreateProgramWithIL`).

**Exit criterion**: SPIR-V vecadd passes on Vortex device on SimX,
matching the OpenCL-C variant.

### Phase 2 — KMU migration  (1–2 weeks, largest unknown)

- Coordinate with v3 RTL / SimX team to pin the KMU descriptor
  format and dispatch API.
- Replace `kernel_main.c` spawn-thread bootstrap with KMU
  descriptor build + dispatch.
- Update `kernel_args.h` arg layout for KMU.
- Audit `lib/kernel/vortex/{atomics,barrier,printf,workitems}.c`
  for spawn-thread assumptions; update.
- Re-run Phase 0 and Phase 1 tests on KMU launch.

**Exit criterion**: OpenCL-C and SPIR-V vecadd both pass on Vortex
device with KMU dispatch.

### Phase 3 — chipStar enablement  (3–5 days)

- Verify Vortex device advertises `cl_ext_buffer_device_address`
  and `cl_khr_il_program`.
- Implement minimal SVM stubs *or* `vx_mem_reserve`-based fixed-
  address allocation (chipStar's `hipMalloc` only needs one).
- Add basic sub-group reporting (sub-group size = warp size).
- Run [tests/hip/vecadd/main.cpp](../../tests/hip/vecadd/main.cpp)
  via chipStar's `hipcc` against `vortex_3.x` Vortex device on SimX.
- Picks up after this point at
  [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
  Step 4 onward (coverage testing).

**Exit criterion**: vecadd HIP test passes via chipStar
end-to-end.

### Phase 4 — Hygiene  (1–2 days)

- Rewrite `README.vortex`: KMU launch, SPIR-V path, install
  instructions for `llvm_vortex`, rebase recipe.
- Add `.github/workflows/build_vortex.yml` (build + smoke test).
- Add `docs/sphinx/source/vortex.rst` upstream-style docs (groundwork
  for future upstream PR).
- Tag `vortex_3.0` once Phases 0–3 pass.

### Phase 5 — Optional follow-ups  (open-ended)

- POCL `cl_khr_command_buffer` support on Vortex (HIP graphs).
- TCU / DXA exposure as Vortex-specific kernel built-ins.
- Push toward POCL CTS pass on Vortex device.
- Attempt upstreaming `lib/CL/devices/vortex/` to `pocl/pocl`.

---

## 6. Inventory: what stays, what changes, what's new

| Path | Disposition |
|---|---|
| `lib/CL/devices/vortex/CMakeLists.txt` | Update for SPIR-V link, modern POCL device-ops, LLVM 18 |
| `lib/CL/devices/vortex/pocl-vortex.{c,h}` | **Refactor**: add `build_program_with_il` hook; modern device-ops shape |
| `lib/CL/devices/vortex/vortex_utils.{cc,h}` | **Refactor**: accept LLVM IR from either OpenCL-C or SPIR-V path |
| `lib/CL/devices/vortex/kernel_main.c` | **Replace** with KMU dispatch |
| `lib/CL/devices/vortex/kernel_args.h` | **Replace** for KMU arg layout |
| `lib/kernel/vortex/{atomics,barrier,printf,workitems}.c` | Audit + KMU updates |
| `lib/kernel/vortex/CMakeLists.txt` | Audit |
| `lib/CL/devices/CMakeLists.txt` (registration) | Re-apply Vortex device registration onto `release_6_0` base |
| `lib/CL/devices/devices.c` | Re-apply Vortex registration |
| Top-level `CMakeLists.txt` | Re-apply `ENABLE_VORTEX` option onto `release_6_0` base |
| `cmake/LLVM.cmake` | Re-apply Vortex-specific LLVM detection |
| `config.h.in.cmake` | Re-apply Vortex-specific `#define`s |
| `cl_offline_compiler.sh.in.cmake` | Audit; re-apply if Vortex-specific |
| `README.vortex` | **Rewrite** (Phase 4) |
| `.github/workflows/build_vortex.yml` | **New** (Phase 4) |
| `docs/sphinx/source/vortex.rst` | **New** (Phase 4) |

---

## 7. Test plan

| Phase | Test | Driver | Pass criterion |
|---|---|---|---|
| 0 | OpenCL vecadd from [tests/opencl/vecadd](../../tests/opencl/vecadd/) | SimX | numerical match (existing baseline) |
| 1 | OpenCL vecadd compiled to SPIR-V via `clang + llvm-spirv`, loaded via `clCreateProgramWithIL` | SimX | numerical match |
| 1 | OpenCL sgemm SPIR-V variant | SimX | numerical match |
| 2 | All Phase 0 + 1 tests under KMU dispatch | SimX, RTLsim | numerical match |
| 3 | [tests/hip/vecadd](../../tests/hip/vecadd/) via chipStar | SimX | `PASSED!` |
| 3 | [tests/hip/sgemm](../../tests/hip/sgemm/) via chipStar | SimX | `PASSED!` |
| 4 | All of the above on RTLsim | RTLsim | unchanged from SimX |

CI hook in [feature_hip/ci/regression.sh.in](../../ci/regression.sh.in)
gated on `$(TOOLDIR)/pocl_vortex` presence.

---

## 8. Risks

1. **KMU API not yet pinned.** Phase 2 depends on the v3 KMU
   descriptor/dispatch interface being stable enough to code
   against. **Mitigation**: coordinate with the v3 RTL/SimX teams
   before Phase 2 starts; keep Phase 0/1 launch path on
   `vx_spawn_threads` as a fallback if KMU work slips.

2. **Shared-file edits surface for re-applying.** The 108-file
   diff against `release_6_0` is mostly upstream drift, but the
   Vortex-specific shared-file edits (registration, build flags,
   LLVM detection) need careful re-identification on the new
   branch. **Mitigation**: Phase 0 spends time grepping for
   "vortex" mentions before any commit lands; preserve a
   `vortex-port-notes.md` of every re-applied hunk.

3. **POCL device-ops API drift.** Between `vortex_2.x`'s 6.0-era
   snapshot and a fresh `release_6_0` branch tip, hook
   signatures may have changed. **Mitigation**: build green
   first; refactor to current API; don't mix API updates with
   feature work.

4. **chipStar OpenCL feature requirements.** chipStar may need
   features Vortex hardware can't cleanly express (full SVM,
   generic AS, sub-group operations). **Mitigation**: Phase 3
   identifies these specifically; report them as separate
   follow-ups rather than blocking the Phase 1+2 work.

5. **LLVM version drift.** This proposal pins to POCL 6.0 (LLVM
   14–18 supported). If `llvm_vortex` ever moves to LLVM 19+,
   we need to also bump POCL to 7.0. **Mitigation**: explicit
   LLVM-version-gate in the Vortex CMakeLists; fail-fast on
   mismatch.

---

## 9. Timeline

| Phase | Days | Cumulative |
|---|---:|---:|
| 0: branch + inventory | 2–3 | 3 |
| 1: SPIR-V wiring | 3–5 | 8 |
| **2: KMU migration** | **5–10** | **18** |
| 3: chipStar enablement | 3–5 | 23 |
| 4: hygiene | 1–2 | 25 |
| 5: optional follow-ups | open | — |

**Total core (Phases 0–4): ~5 weeks (25 working days).** Phase 2
dominates and has the most uncertainty; if KMU coordination drags,
Phases 0/1/3 can complete on the legacy `spawn_thread` path and
Phase 2 lands later as an isolated change.

---

## 10. Decisions to make before Phase 0

1. **Branch from `upstream/release_6_0` or `upstream/release_6_1`?**
   - 6.0: stable, matches the `cl_ext_buffer_device_address` landing.
   - 6.1: bugfixes only, slightly newer. **Default: 6.0.**
2. **Branch name on the Vortex repo: `vortex_3.x`?** Matches the
   Vortex `v3` series. Alternative: `pocl_vortex_v3` for clarity.
   **Default: `vortex_3.x` (mirrors `vortex_2.x`).**
3. **KMU API freeze.** Confirm v3 KMU descriptor/dispatch
   interface is stable before Phase 2 starts; otherwise Phase 2
   slips behind v3 RTL.
4. **Track `vortex_2.x` in parallel?** Once `vortex_3.x` is
   green through Phase 1, freeze `vortex_2.x` for legacy users
   and direct new work to `vortex_3.x`. Tag `vortex_2.x` final
   commit before deprecation.

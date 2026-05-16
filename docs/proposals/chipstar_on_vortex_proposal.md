**Date:** 2026-05-09
**Status:** Draft — recon plan, not yet executed
**Author:** Blaise Tine
**Related:**
[hip_support_proposal.md](hip_support_proposal.md).

### Update history

- **2026-05-09** — Initial draft. Validation gating
  [hip_support_proposal.md](hip_support_proposal.md) Phase 1+ scope.
- **2026-05-09 (amendment)** — Added **Step 0: rebase `pocl_vortex`
  onto upstream POCL 6.0** as a recon prerequisite. The current
  `pocl_vortex` fork is from POCL 4.0 (Jun 2023) and pre-dates two
  upstream changes that materially affect this experiment:
  `cl_ext_buffer_device_address` (POCL 6.0 — added at chipStar's
  request, makes `hipMalloc` work without SVM) and the SPIR-V
  ingestion work on non-ARM/x86 RISC-V hosts (POCL 5.0). Running
  the recon on the 4.0-era fork would produce misleading negative
  results. Timeline (§11) and version pins (§10) updated.
- **2026-05-15 (post-execution)** — chipStar HIP path is
  **functional but rv64-only** on Vortex. chipStar's hipcc
  invokes `clang++ --offload=spirv64` unconditionally and emits
  SPIR-V with `OpMemoryModel Physical64`. POCL's
  [bitcode_is_spirv_execmodel](../../../pocl_vortex/lib/CL/pocl_util.c)
  checks the SPIR-V address bits against
  `cl_device::address_bits`, so on a Vortex device configured for
  rv32 (`address_bits=32`, kernel BC = `kernel-riscv32`),
  `clCreateProgramWithIL` returns `CL_INVALID_OPERATION` ("No
  device in context supports SPIR"). Verified end-to-end with
  vecadd and sgemm on rv64 (PASS) and rv32 (CL_INVALID_OPERATION).
  This is **not a POCL/Vortex regression** — there is no
  `--offload=spirv32` option in upstream chipStar, so HIP-on-rv32
  Vortex needs either (a) a chipStar patch to support a 32-bit
  SPIR-V emission mode, or (b) the native HIPVortex toolchain
  from [hip_support_proposal.md](hip_support_proposal.md).
  Documenting as a v3.0 known limitation.
- **2026-05-09 (supersession)** — Step 0 forensics were wrong.
  `vortex_2.x` is **already at POCL 6.0** (the local `CHANGES`
  file's "4.1 Unreleased" header was stale and misleading);
  108-file diff vs `upstream/release_6_0` is mostly upstream
  drift, not Vortex-specific work. The shared POCL infrastructure
  for `cl_ext_buffer_device_address` and `spirv_parser` is
  **already present** — the actual gap is that `pocl-vortex.c`
  has no SPIR-V code path and the Vortex device still launches
  via `spawn_thread`. That work is the right unit, but it's a
  redesign, not a rebase, and it has been broken out into its
  own proposal:
  [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md).
  This proposal's **Steps 0–2 are superseded** by Phases 0–1 of
  the redesign; the chipStar recon proper is now just Steps 3–5
  (renumbered 1–3 below). Timeline (§11) and decision gates (§6)
  updated.

# chipStar on Vortex — Validation Proposal

## 1. Summary

Validate whether **chipStar** (HIP/CUDA → SPIR-V → OpenCL) running on
top of `pocl_vortex`'s Vortex OpenCL device target gives a faster,
lower-cost path to HIP-on-Vortex than the bespoke toolchain in
[hip_support_proposal.md](hip_support_proposal.md).

The validation is a three-step recon (build chipStar, run vecadd,
broaden coverage) gated on the **prerequisite** that
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) **Phases
0–1 are complete** — i.e. `vortex_3.x` is built clean against
`llvm_vortex`, accepts SPIR-V kernels via `clCreateProgramWithIL`,
and runs at least an OpenCL-C and a SPIR-V variant of vecadd on
SimX. With that prerequisite in place, this recon is **~1 week**.
No new code in the success path — we are exclusively wiring up
existing components and observing what fails.

If the path works end-to-end, it gives us immediate portability for
unmodified HIP applications and **reduces
[hip_support_proposal.md](hip_support_proposal.md)'s Phase 1+ scope
to what chipStar can't do** (i.e. exposing Vortex hardware
extensions). If it fails, we know precisely where, and the bespoke
toolchain proceeds as-is.

---

## 2. Why now (and why before Phase 1 of the main proposal)

[hip_support_proposal.md](hip_support_proposal.md) §8 risk #4 already
flagged this: *"a one-day spike (compile vecadd via `llvm_vortex`'s
`HIPSPV.cpp` and load through `pocl_vortex`) is the cheapest
validation; the result informs whether to invest in chipStar+pocl_vortex
as a parallel secondary path or close that door."*

The bespoke toolchain (Phase 1: runtime stub; Phase 2: HIPVortex
Clang toolchain; Phase 2.1: device-libs) is a multi-quarter project.
Even a *partial* chipStar success changes the scope:

- **chipStar covers vanilla HIP** → bespoke Phase 1+2 only need to
  exist for hardware-extension exposure (i.e. WMMA/TMA), shrinking
  Phase 1's API surface from ~30 functions to "whatever the
  extension HIP headers want from the runtime."
- **chipStar covers some HIP** → run both, with chipStar as the
  portability default and bespoke for extensions and any tests
  chipStar doesn't handle.
- **chipStar doesn't work** → bespoke toolchain proceeds with
  `hip_support_proposal.md` Phase 1 scope as written.

The cheapest moment to learn this is *now* — before any Phase 1
runtime code beyond the
[skeleton](../../../hip_vortex/) lands. (Phase 0 scaffolding stays
useful regardless of outcome.)

---

## 3. The hypothesis

chipStar takes HIP source through Clang's `HIPSPV.cpp` toolchain to
**SPIR-V**, then dispatches kernels via the **OpenCL runtime** at
load time. `pocl_vortex` is an OpenCL implementation with
`ENABLE_SPIRV=ON` *and* a dedicated Vortex device target.
*In principle*, the entire HIP → Vortex path is:

```
foo.hip
  │
  ▼  chipStar's hipcc        (clang -x hip -> SPIR-V; via llvm_vortex)
  │
HIP host binary  +  device.spv
  │
  ▼  libCHIP.so loads device.spv at runtime
  │
  ▼  clCreateProgramWithIL via pocl_vortex's libOpenCL.so
  │
  ▼  pocl_vortex Vortex device target lowers SPIR-V → Vortex .vxbin
  │
  ▼  vortex_runtime executes
```

**Zero Vortex-specific compiler code in the success path.** Every
component already exists.

### Why it might fail

The gating unknown is whether `pocl_vortex/lib/CL/devices/vortex/`
actually consumes SPIR-V or only OpenCL-C source. POCL's SPIR-V
parsing is in shared infrastructure (`spirv_parser.cc`,
`spirv.hh`), but each device target plugs into the build pipeline
through its own `post_build_program` / `create_kernel` hooks
(observed in [pocl-vortex.c](../../../pocl_vortex/lib/CL/devices/vortex/pocl-vortex.c)).
The Vortex device may be wired only for the LLVM-IR-from-OpenCL-C
path.

Secondary unknowns: chipStar's required OpenCL features (SVM,
generic address space, sub-groups) may exceed what the Vortex
device supports.

---

## 4. Component inventory (already in place)

| Component | Path | What it provides |
|---|---|---|
| Clang HIPSPV toolchain | [llvm_vortex/clang/lib/Driver/ToolChains/HIPSPV.cpp](../../../llvm_vortex/clang/lib/Driver/ToolChains/HIPSPV.cpp) | HIP → SPIR-V compilation |
| Clang HIP headers | [llvm_vortex/clang/lib/Headers/__clang_hip_*.h](../../../llvm_vortex/clang/lib/Headers/) | HIP language mode device-libs |
| chipStar runtime | `~/dev/chipstar` (to clone) | `libCHIP.so` — wraps OpenCL/Level-Zero, exposes HIP API |
| chipStar `hipcc` driver | same | Drives clang HIPSPV + bundles SPIR-V into the host binary |
| POCL with SPIR-V | [pocl_vortex/lib/CL/devices/spirv_parser.{cc,hh}](../../../pocl_vortex/lib/CL/devices/spirv_parser.cc) | SPIR-V → LLVM IR ingestion |
| POCL Vortex device | [pocl_vortex/lib/CL/devices/vortex/](../../../pocl_vortex/lib/CL/devices/vortex/) | LLVM IR → Vortex `.vxbin` lowering |
| Vortex runtime | [feature_hip/sw/runtime/](../../sw/runtime/) | `libvortex.so` — device dispatch |

Versions / pins:
- `llvm_vortex` HEAD `d78d4a25e` (LLVM 18.1, 2025-08-08)
- `pocl_vortex` HEAD `2ae6d49` (2026-01-13)
- chipStar: latest `main` at start of validation; pin commit in §10.

---

## 5. Validation plan

Each step lists: **goal**, **commands**, **success criterion**,
**likely failures**, **fix-or-pivot**.

### Prerequisite: [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) Phases 0–1

What used to be Steps 0–2 (rebase `pocl_vortex`, build with
SPIR-V, run a SPIR-V OpenCL kernel on the Vortex device) has been
absorbed into Phases 0–1 of the `pocl_vortex` v3 redesign:

- Phase 0 of the redesign establishes a clean `vortex_3.x` branch
  on `upstream/release_6_0`, ports the Vortex-specific layer,
  and builds green.
- Phase 1 of the redesign wires the SPIR-V code path into
  `pocl-vortex.c` and validates that a SPIR-V variant of vecadd
  passes on the Vortex device on SimX.

When Phase 1 of that proposal exits green, this recon resumes
below. The original Step 0 (rebase) is **dropped** — it was
based on wrong forensics about the fork's POCL version (see
update history). The original Steps 1–2 (build with SPIR-V; run
SPIR-V kernel on Vortex) are **the exit criteria of Phase 1** of
the redesign, so they don't need to be repeated here.

### Step 0 — Rebase `pocl_vortex` onto upstream POCL 6.0 (SUPERSEDED)

**Goal**: bring the `pocl_vortex` fork up to a POCL revision that
(a) ships **`cl_ext_buffer_device_address`** (the extension chipStar
specifically requested for `hipMalloc()` on non-SVM backends — POCL
6.0), and (b) includes the **5.0/6.0 work on SPIR-V ingestion for
non-ARM/x86** architectures. Without this, a Step 4 failure is
misleading — chipStar may break for reasons upstream POCL has
already fixed.

POCL 6.0 was chosen over 7.0 because 7.0 requires LLVM 19/20 and
`llvm_vortex` is on LLVM 18.1; staying on POCL 6.0 keeps the LLVM
pin stable.

```bash
cd ~/dev/pocl_vortex
git remote add upstream https://github.com/pocl/pocl.git    # if not already
git fetch upstream
git checkout -b vortex-rebase-6.0
git rebase upstream/release_6_0     # or `git merge` if rebase is too painful

# expect conflicts in:
#   lib/CL/devices/CMakeLists.txt          (driver registration changes)
#   lib/CL/devices/vortex/CMakeLists.txt   (LLVM include / linker flags)
#   lib/CL/devices/vortex/pocl-vortex.c    (driver-ops struct shape changed
#                                           between POCL 4.x and 6.x)
#   lib/CL/devices/vortex/vortex_utils.cc  (LLVM API drift, LLVM 16 → 18)
```

After the rebase, the Vortex device hooks must be ported to the
POCL 6.0 driver-ops API (new SVM hooks, command-buffer hooks may
need stub implementations).

**Success**:
- `pocl_vortex` (rebased) builds clean against `llvm_vortex`
  (LLVM 18.1).
- The Vortex device appears in `bin/poclcc -l`.
- `clGetDeviceInfo(..., CL_DEVICE_EXTENSIONS, ...)` reports
  `cl_ext_buffer_device_address` for *some* device (CPU is
  enough — demonstrates the extension code path is in the tree;
  Vortex device support follows in Step 1/2).
- POCL's own non-Vortex tests pass (sanity check on the rebase).

**Likely failures**:
- POCL 4.0 → 6.0 introduced new driver-ops hooks (SVM, command
  buffers, multi-device); the Vortex device target needs stub
  implementations for any hook the runtime now expects.
- LLVM API changes between LLVM 16 (POCL 4.0 era) and LLVM 18
  (current `llvm_vortex`) hit `vortex_utils.cc`.
- Driver renames upstream (`basic` → `cpu-minimal`,
  `pthread` → `cpu`) may surface in build scripts, device-name
  string compares, or env-var defaults.
- Build-system changes — POCL 5.0+ added Level Zero, remote, and
  TBB drivers; CMake option matrix is larger.

**Fix-or-pivot**:
- If the full rebase explodes (> 10 working days of fixes), fall
  back to a **cherry-pick subset**: only the commits introducing
  `cl_ext_buffer_device_address` and the SPIR-V-on-non-ARM/x86
  improvements. Smaller scope, messier diff.
- If LLVM 18 / POCL 6.0 surface a hard incompatibility, the
  decision branches to either bumping `llvm_vortex` to LLVM 19 +
  targeting POCL 7.0, or staying on POCL 4.0 and dropping the
  recon. **That branch decision is out of scope for this
  proposal** — pause and escalate.

**Budget**: 5–10 working days. Hard cap at 10 days before pausing
the recon and declaring chipStar deferred.

> **SUPERSEDED** by [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md)
> Phases 0–1. The forensics that motivated this step were based
> on a stale `CHANGES` header; `vortex_2.x` is already at POCL
> 6.0. Kept inline below as a record of the prior plan; do not
> execute.

### Step 1 — Build `pocl_vortex` with SPIR-V enabled (SUPERSEDED — covered by redesign Phase 0)

**Goal**: produce a `libpocl.so` + `libOpenCL.so` ICD config under
`$(TOOLDIR)/pocl_vortex/` that recognizes both the Vortex device
and SPIR-V kernel input.

```bash
mkdir -p ~/dev/pocl_vortex/build && cd ~/dev/pocl_vortex/build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$TOOLDIR/pocl_vortex \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENABLE_SPIRV=ON \
    -DENABLE_VORTEX=ON \
    -DLLVM_DIR=$TOOLDIR/llvm_vortex/lib/cmake/llvm
cmake --build . -j && cmake --install .
```

**Success**: build prints `ENABLE_SPIRV: 1` *and* the Vortex device
appears in `bin/poclcc -l`.

**Likely failures**: missing `llvm-spirv` translator (POCL needs it
for SPIR-V); LLVM version mismatch in the Vortex device CMake
fragment ([pocl_vortex/lib/CL/devices/vortex/CMakeLists.txt](../../../pocl_vortex/lib/CL/devices/vortex/CMakeLists.txt)).

**Fix-or-pivot**: if `llvm-spirv` is missing, install
`llvm-spirv-translator` from the LLVM-SPIRV-Translator project (one
script). If LLVM version mismatch, rebuild that submodule against
`llvm_vortex`. Both are <1 day fixes; neither closes the door.

### Step 2 — Run a SPIR-V OpenCL kernel on the Vortex device (SUPERSEDED — covered by redesign Phase 1)

**Goal**: prove that the Vortex device target ingests SPIR-V, end of
story. Don't involve chipStar yet.

Take an existing OpenCL test ([tests/opencl/vecadd/](../../tests/opencl/vecadd/))
and compile its kernel to SPIR-V instead of OpenCL-C source:

```bash
cd ~/dev/vortex_v3/feature_hip/tests/opencl/vecadd
$TOOLDIR/llvm_vortex/bin/clang \
    -cc1 -triple spir64-unknown-unknown -emit-llvm-bc \
    -finclude-default-header -O2 \
    kernel.cl -o kernel.bc
$TOOLDIR/llvm_vortex/bin/llvm-spirv kernel.bc -o kernel.spv

# patch main.cc to use clCreateProgramWithIL instead of WithSource;
# OR write a 50-line throwaway host that does the same.
```

Run the host with `VORTEX_DRIVER=simx` and the pocl_vortex ICD:

```bash
OCL_ICD_VENDORS=$TOOLDIR/pocl_vortex/etc/OpenCL/vendors \
LD_LIBRARY_PATH=$TOOLDIR/pocl_vortex/lib:$VORTEX_RT_LIB \
VORTEX_DRIVER=simx \
./vecadd_spirv
```

**Success**: `PASSED!` from the existing vecadd test on numerical
match.

**Likely failures**:
- `CL_INVALID_PROGRAM_EXECUTABLE` — the Vortex device's
  `pocl_vortex_post_build_program` rejects the SPIR-V ingestion
  path (only handles `.cl` source).
- POCL accepts SPIR-V but the Vortex backend's kernel-build hook
  doesn't route through it.

**Fix-or-pivot**: if SPIR-V isn't routed, the missing piece is in
[pocl-vortex.c](../../../pocl_vortex/lib/CL/devices/vortex/pocl-vortex.c)
— a small (~200 LOC) addition to call into POCL's common SPIR-V
parsing before lowering to LLVM IR. **This is still cheaper than
the bespoke HIPVortex toolchain.** If the failure mode is more
fundamental (e.g. Vortex backend can't lower SPIR-V semantics like
generic-AS pointers at all), close the door and revert to
[hip_support_proposal.md](hip_support_proposal.md).

### Step 3 — Build chipStar against `llvm_vortex` (this recon's Step 1)

**Goal**: produce `$(TOOLDIR)/chipstar/bin/hipcc` and
`$(TOOLDIR)/chipstar/lib/libCHIP.so`.

```bash
git clone https://github.com/CHIP-SPV/chipStar.git ~/dev/chipstar
cd ~/dev/chipstar && git submodule update --init --recursive
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$TOOLDIR/chipstar \
    -DLLVM_DIR=$TOOLDIR/llvm_vortex/lib/cmake/llvm \
    -DCHIP_BUILD_TESTS=ON
cmake --build . -j && cmake --install .
```

**Success**: install completes; `hipcc --version` reports a chipStar
build.

**Likely failures**: SPIRV-LLVM-Translator pin in chipStar may not
match `llvm_vortex` exactly. Bump or regenerate.

**Fix-or-pivot**: the SPIRV-LLVM-Translator is small and tracks
LLVM versions cleanly. <1 day if needed.

### Step 4 — Compile and run a HIP test through chipStar+`vortex_3.x` (this recon's Step 2)

**Goal**: the headline result. [tests/hip/vecadd/main.cpp](../../tests/hip/vecadd/main.cpp)
runs end-to-end on Vortex via the chipStar path.

```bash
cd ~/dev/vortex_v3/feature_hip/tests/hip/vecadd
$TOOLDIR/chipstar/bin/hipcc main.cpp -o vecadd_chipstar

OCL_ICD_VENDORS=$TOOLDIR/pocl_vortex/etc/OpenCL/vendors \
CHIP_BE=opencl CHIP_DEVICE_TYPE=pocl \
LD_LIBRARY_PATH=$TOOLDIR/pocl_vortex/lib:$TOOLDIR/chipstar/lib:$VORTEX_RT_LIB \
VORTEX_DRIVER=simx \
./vecadd_chipstar -n 64
```

**Success**: `PASSED!` printed; numerical correctness matches the
CPU reference within `FLOAT_ULP=6`.

**Likely failures**:
- Compile-time: chipStar's clang invocation rejects something in our
  HIP source — most likely if we use a HIP API that chipStar's
  device-libs stub differently than ROCm.
- Link-time: missing OpenCL feature (SVM, generic address space) in
  the Vortex device.
- Runtime: `clEnqueueNDRangeKernel` returns `CL_INVALID_KERNEL_ARGS`
  or similar — argument-passing mismatch between chipStar and POCL.
- Result: completes but produces wrong numbers.

**Fix-or-pivot**: each failure mode is investigated; most are
either pocl_vortex device-side gaps (which we file as small
follow-ups) or chipStar pinning issues. **Only "Vortex hardware
fundamentally can't express the OpenCL feature chipStar requires"
closes the door.**

### Step 5 — Coverage (this recon's Step 3)

**Goal**: scope how much of HIP works.

```bash
# our reference tests
make -C ~/dev/vortex_v3/feature_hip/tests/hip/sgemm \
    HIPCC=$TOOLDIR/chipstar/bin/hipcc \
    HIP_LIB_PATH=$TOOLDIR/chipstar/lib

# chipStar's own test suite, against pocl_vortex
cd ~/dev/chipstar/build
ctest -E "RegressionTest" --output-on-failure
```

**Success metric**: pass/fail counts for chipStar's smoke suite +
our two reference tests. Anything ≥80% of smoke pass is a strong
positive signal.

**Likely failures**: a long tail of OpenCL-feature gaps in
pocl_vortex's Vortex device. Each becomes a small follow-up filed
as a separate work item.

---

## 6. Decision gates

| After step | Outcome | Action |
|---|---|---|
| Prerequisite | [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) Phase 1 green (SPIR-V vecadd passes on `vortex_3.x`) | Continue to Step 3 (chipStar build) |
| Prerequisite | Phase 1 reveals a fundamental Vortex backend gap (can't lower generic-AS / SVM / sub-groups) | **Close the door**; revert to [hip_support_proposal.md](hip_support_proposal.md) |
| 4 | vecadd passes | Strong signal. Run step 5 |
| 4 | vecadd fails, diagnosable cause | Fix and re-run; one fix-loop max |
| 4 | vecadd fails for fundamental reason | Document and pivot to bespoke |
| 5 | Coverage ≥ 80% of chipStar smoke + both our reference tests | **chipStar becomes the primary path**; rewrite [hip_support_proposal.md](hip_support_proposal.md) Phase 1+ scope |
| 5 | Coverage 30–80% | Run **both paths**: chipStar primary for unmodified apps, bespoke for extensions and uncovered tests |
| 5 | Coverage < 30% | chipStar stays as a fallback for the apps it does cover; bespoke toolchain is primary |

---

## 7. Implications for `hip_support_proposal.md`

Three branch points to be inserted after the recon completes:

1. **chipStar primary** (high coverage). Rewrite
   [hip_support_proposal.md](hip_support_proposal.md):
   - **Phase 1** (runtime stub) shrinks to "the few HIP APIs chipStar
     can't or won't expose" (likely just custom intrinsics for
     hardware extensions).
   - **Phase 2** (HIPVortex Clang toolchain) becomes optional /
     dropped.
   - **Phase 3** (MLIR middleware) stays — but its input is now a
     research-flavoured fork of chipStar, not a fresh frontend.
   - **Phase 4** (extension headers) stays as-is. Extensions that
     can't be expressed in SPIR-V get a small bespoke path.

2. **Mixed**. Keep both proposals; chipStar is the default for
   `tests/hip/`, bespoke for tests that need extensions or fail on
   chipStar.

3. **Bespoke only**.
   [hip_support_proposal.md](hip_support_proposal.md) proceeds
   unchanged.

The amendment to `hip_support_proposal.md` is written **after**
this recon completes; the amendment is itself a single-sentence
update-history entry plus the phase-scope changes above.

---

## 8. What this proposal does *not* commit to

- No new compiler code is written until step 4 fails in a
  fixable-but-non-trivial way.
- No edits to `hip_support_proposal.md` until the recon completes.
- No commits to `hip_vortex` (the Phase 0 skeleton stays as-is;
  it is independent of this validation).

---

## 9. Risks

1. **Time investment if it doesn't pan out.** Cap this recon's
   budget at ~1 week (Steps 1–3 below). The much larger
   `pocl_vortex` work is in
   [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) and
   has its own budget; this recon is what runs *after* that work
   completes Phase 1.

2. **`pocl_vortex` SPIR-V is a Trojan horse.** Step 2 succeeds, but
   step 4 reveals a feature gap that requires re-architecting the
   Vortex device's lowering. **Mitigation**: each step has a clean
   exit criterion; we don't commit incremental fixes that exceed
   their step's budget.

3. **chipStar's HIP coverage is narrower than advertised.** The
   2026 paper claims production-ready, but on PowerVR + Level Zero
   targets. **Mitigation**: step 5 is the explicit coverage check;
   we report numbers, not headlines.

4. **Sub-groups / SVM / generic-AS gaps in Vortex.** chipStar
   prefers generic-AS pointers; the Vortex backend may not lower
   them. **Mitigation**: this is exactly what step 2 exposes; if
   it shows up there, we know the door is closed before sinking
   more time.

---

## 10. Versions to pin (filled in at start of execution)

| Component | Pin | Source |
|---|---|---|
| `llvm_vortex` | `d78d4a25e` (current HEAD, LLVM 18.1) | [llvm_vortex git log](../../../llvm_vortex/) |
| `pocl_vortex` | `vortex_3.x` branch produced by [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) Phase 1 — based on `upstream/release_6_0` with Vortex SPIR-V wiring + KMU dispatch | [pocl_vortex git log](../../../pocl_vortex/) |
| chipStar | TBD — latest `main` at execution | [github.com/CHIP-SPV/chipStar](https://github.com/CHIP-SPV/chipStar) |
| SPIRV-LLVM-Translator | TBD — chipStar's submodule pin | (submodule of chipStar) |

---

## 11. Estimated timeline

**Prerequisite:** [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md)
Phases 0–1 (estimate ~8 days in that proposal).

This recon's own steps:

| Step | Days | Cumulative |
|---|---:|---:|
| 1 (was 3): chipStar build against `llvm_vortex` | 1 | 1 |
| 2 (was 4): HIP vecadd via chipStar+`vortex_3.x` | 1–2 | 3 |
| 3 (was 5): coverage (sgemm, chipStar smoke suite) | 2–3 | 6 |

**Total this recon: ~1 week (6 working days), hard cap at 2
weeks.** Once
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) Phase 1
is green, the chipStar recon is fast — most of the formerly
expensive validation steps already passed in that work.

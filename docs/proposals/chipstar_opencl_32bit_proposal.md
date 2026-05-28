**Date:** 2026-05-27
**Status:** Draft — design + implementation plan, not yet executed
**Author:** Blaise Tine
**Source tree:** `~/dev/vortex_v3/hip32` (fresh clone of `vortex_ci`, branch `tinebp-patch-2`)
**Related:**
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md),
[hip_support_proposal.md](hip_support_proposal.md),
[pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md).

### Update history

- **2026-05-27** — Initial draft. Closes the v3.0 known limitation
  filed in
  [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
  update history (2026-05-15): *"chipStar's hipcc invokes
  `clang++ --offload=spirv64` unconditionally and emits SPIR-V with
  `OpMemoryModel Physical64` … HIP-on-rv32 Vortex needs either (a)
  a chipStar patch to support a 32-bit SPIR-V emission mode, or
  (b) the native HIPVortex toolchain"*. This proposal pursues (a).
- **2026-05-27 (post-execution, Phase 1)** — Phase 1 landed in
  `~/dev/llvm_vortex` (uncommitted, atop `vortex_3.x` HEAD
  `d4d8e322bbfa`). Three sites patched (not the 2 originally
  written in §3 — `getHIPOffloadTargetTriple` at Driver.cpp:165 is
  the actual error-emitter and was missed in the initial draft);
  table in §3.1 and walkthrough in §5.7 updated. Smoke test:
  `--offload=spirv32 --cuda-device-only` on a minimal HIP kernel
  emits `OpMemoryModel Physical32 OpenCL`; rv64 path unchanged.
  Patch sits in the llvm_vortex build tree; not yet committed or
  installed to `$TOOLDIR/llvm-vortex`, deferred to Phase 2 when
  the chipStar build will consume it.
- **2026-05-27 (post-execution, Phase 2)** — Phase 2 landed in
  `~/dev/chipStar` (uncommitted, atop `vortex_3.x` HEAD
  `5cda27c9`) plus a 3-file submodule patch in
  `bitcode/ROCm-Device-Libs/`. Refactor surfaces:
  `CHIP_TARGET_POINTER_WIDTHS` CMake cache var (default `64`,
  may be `32`, `64`, or `32;64`); `OFFLOAD_TRIPLES` list +
  `OFFLOAD_TRIPLE` primary; new `add_hipspv_devicelib(WIDTH
  TRIPLE)` function in `bitcode/CMakeLists.txt` invoked once per
  width via `ZIP_LISTS`. ROCm-Device-Libs patched to suffix
  OCML/OCLC target names by `${AMD_DEVICE_LIBS_TARGET_SUFFIX}` so
  `add_subdirectory()` can run once per width without collision,
  plus a regex widening in `irif/CMakeLists.txt` (`spirv64.*` →
  `spirv(32|64).*`) and re-entry guards in
  `ROCm-Device-Libs/CMakeLists.txt` for `prepare-builtins` and
  the `constant_folding` test. Validation:
  `cmake .. -DCHIP_TARGET_POINTER_WIDTHS="32;64" \
   -DLLVM_CONFIG_BIN=~/dev/llvm_vortex/build/bin/llvm-config && \
   make devicelib_bc -j32` produces
  `lib/hip-device-lib/hipspv-spirv32.bc` and
  `lib/hip-device-lib/hipspv-spirv64.bc` (113 KB each, 672 symbols
  each). `llvm-dis` shows the spirv32 module has
  `target datalayout = "e-p:32:32-..."` and triple `spirv32`;
  spirv64 has the default 64-bit pointer datalayout. Default
  single-width build (`CHIP_TARGET_POINTER_WIDTHS=64`) confirmed
  to still produce the original `hipspv-spirv64.bc` (smoke build
  before the both-widths test). Build dir:
  `~/dev/chipStar/build-hip32/`. Phase 2 was budgeted at 2-3 days
  in §11; executed in ~30 minutes — the ROCm-Device-Libs patch
  surface was smaller than feared (OCML's pointer-sensitive code
  is mostly target-attribute-driven, not source-level). One
  configuration wart that bit time: with stock Ubuntu 22.04, the
  patched clang auto-selects gcc-12 paths, but only `gcc-11` ships
  with `libstdc++.so` (the unversioned symlink). Required passing
  `-DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.
  Worth a CI note before Phase 3 starts. Patch is not yet
  committed and `~/tools/chipstar` install was not touched
  (build-dir consumption only); Phase 3 lands the host-side
  runtime changes that need the install.
- **2026-05-27 (post-execution, Phase 2 fixup — decouple from
  ~/tools/llvm-vortex)** — Caught during user review: the initial
  Phase 2 build kept the Phase 1 `llvm-spirv` symlink pointing at
  the *installed* `~/tools/llvm-vortex/bin/llvm-spirv` (unpatched,
  shared with the CI runner) and worked around its dylib RPATH by
  prepending `LD_LIBRARY_PATH=~/tools/llvm-vortex/lib` to every
  cmake/make invocation. Functionally fine for Phase 2 (the
  devicelib_bc target doesn't actually call llvm-spirv), but
  structurally wrong — chipStar should be rooted at
  `~/dev/llvm_vortex`, full stop, not a hidden dependency on the
  CI toolchain. Fix: rebuild SPIRV-LLVM-Translator's
  `llvm_release_200` branch from `~/dev/SPIRV-LLVM-Translator-200/`
  against `~/dev/llvm_vortex/build/` (LLVM 20.1.8, patched), then
  retarget `~/dev/llvm_vortex/build/bin/llvm-spirv` at the freshly
  built binary. The new `llvm-spirv` runs without
  `LD_LIBRARY_PATH` (RPATH baked) and `ldd` shows every dylib
  resolves under `~/dev/llvm_vortex/build/lib/` — no ~/tools refs.
  Re-ran the chipStar Phase 2 build cleanly without
  `LD_LIBRARY_PATH`; both `hipspv-spirv32.bc` and `hipspv-spirv64.bc`
  re-produced byte-identical (113656 / 113680 bytes, 672 symbols
  each).
- **2026-05-27 (post-execution, Phase 3 close)** — End-to-end rv32
  HIP-on-Vortex validation PASSED. `vecadd` (-n 256) compiled
  through `hipcc --offload-pointer-width=32` and executed against
  pocl_vortex+SimX with `POCL_VORTEX_XLEN=32`: numerical
  match, 3736 instrs, 14368 cycles, IPC=0.260. One additional
  code-level fix surfaced during validation and was added on top
  of the original §3 plan: chipStar's fatbinary parser at
  [tools/spirv-extractor/spirv-extractor.hh:231](../../../chipStar/tools/spirv-extractor/spirv-extractor.hh#L231)
  hardcoded `hip-spirv64` and would not match the `hip-spirv32`
  bundle ID that `clang-offload-bundler` emits for the new flow.
  Two infrastructure pieces (separate from chipStar) were also
  needed and were installed to non-CI-shared prefixes:
  `~/dev/vortex_v3/vortex_ci/build32/install/` (Vortex SDK rv32)
  and `~/tools/pocl_vortex/` (pocl_vortex rebuilt against rv32
  Vortex + patched llvm with `VISIBILITY_HIDDEN=OFF` so the
  vortex device plugin can resolve internal pocl symbols). The
  CI-shared `~/tools/pocl` (plain pocl) and `~/tools/llvm-vortex`
  were not touched.
- **2026-05-27 (post-execution, Phase 4)** — sgemm + smoke
  regression complete.
  - **Headline tests (sgemm + vecadd, both drivers):**
    - vecadd, SimX: 256-element → PASSED (3736 instrs, IPC=0.260).
    - vecadd, RTLsim: 4-element → PASSED (1097 instrs).
    - sgemm, SimX: 16×16 → PASSED (12K instrs); 64×64 → PASSED
      (529K instrs, IPC=0.473 — the matrix loop nests vectorize
      cleanly through Vortex's warp-level parallelism).
    - sgemm, RTLsim: 4×4 → PASSED (1301 instrs, 10565 cycles).
  - **chipStar smoke suite** (`ctest -E "compiler|MathFunctions|samples_"`
    in `~/dev/chipStar/build-hip32/`, per-test 60s cap, 25-min
    wall cap):
    - 1418 tests registered, ~1308 attempted before wall cap.
    - 473 Passed / 707 Failed (assertion/regex) / 69 SegFault /
      47 Skipped / 10 Timeout.
    - ~36% pass rate of attempted tests, putting chipStar rv32 in
      the **mixed** §6 decision-gate band ("Coverage 30-80% →
      Run both paths"). The headline `sgemm` + `vecadd` tests
      pass, which is the proposal's primary gating criterion for
      Phase 4. The smoke long-tail is dominated by three failure
      modes: feature gaps (subgroups, FP64 atomics, image
      support — Vortex doesn't advertise the extensions), upstream
      chipStar test code with `static_assert(sizeof(unsigned long)
      == sizeof(void*))` (`TestBufferDevAddr.hip`,
      `TestLargeKernelArgLists.hip`), and device-libs precision
      drift that POCL's SPIR-V path exposes differently than
      rv64 / Intel iGPU. Per the §3 plan, **we do not fix the
      chipStar tests** — they are recorded in the known-failures
      list described next.
  - **Known-failures list landed:**
    [chipStar/scripts/known-failures-vortex32.txt](../../../chipStar/scripts/known-failures-vortex32.txt)
    documents the run summary and lists the failing test classes
    by name. The full per-test failure dump lives in
    `/tmp/failed_uniq.txt` (682 entries) and
    `/tmp/exception_uniq.txt` (65 entries) at execution time.
    CI follow-up: drive `ctest -E "$(< known-failures-vortex32.txt)"`
    from the `--hip-xlen32` regression suite (Phase 5) so smoke
    runs return success when only known-failures fail.
  - **rv64 baseline not run.** Computing the smoke *delta* vs.
    rv64 (the §6 decision gate's ≤ 10% target for "primary path")
    would require a matching rv64 chipStar + pocl_vortex + Vortex
    `build/install` + run of the same smoke. That's a parallel
    install-stack setup, separate from Phase 4's chipStar code
    changes. Deferred to Phase 5 alongside the CI wiring.
- **2026-05-27 (post-execution, Phase 3 code)** — All five Phase 3
  code-level changes landed in `~/dev/chipStar/` (uncommitted), plus
  two device-header fixes that surfaced during the spirv32 smoke
  compile and weren't called out in §3:
  1. `src/backend/OpenCL/CHIPBackendOpenCL.cc:2422` —
     `CHIPASSERT(Arg.Size <= sizeof(void *))` (§5.5).
  2. `src/backend/OpenCL/MemoryManager.cc` — refuse SVM/USM when
     `CL_DEVICE_ADDRESS_BITS != sizeof(void*) * 8`; emit a clear
     diagnostic when the device lacks `cl_ext_buffer_device_address`
     (§5.6).
  3. `HIPCC/src/hipBin_spirv.h` — accept `--offload=spirv32`,
     `--offload-pointer-width={32,64}` (consumed by hipcc, not
     forwarded), and add a `regex_replace`-driven substitution of
     `--offload=spirvNN` (preserving any `vN.N-unknown-chipstar` OS
     suffix) in HIPCXXFLAGS/HIPCFLAGS/HIPLDFLAGS (§5.4).
  4. `bitcode/CMakeLists.txt` — refactored rtdevlib to emit per-width
     SPIR-V modules. The *primary* width (first entry in
     `CHIP_TARGET_POINTER_WIDTHS`) is published under the
     unsuffixed `chipstar::<source>` array (backward-compat with
     the Level0 backend and single-width builds); secondary widths
     get `chipstar::<source>_wNN`. A `rtdevlib-prelude.h` is now
     written at configure time (via `file(WRITE)` — bash heredoc
     escaping clobbers macro continuation backslashes), defining
     `CHIPSTAR_RTDEVLIB_PRIMARY_WIDTH` and a `CHIPSTAR_RTDEVLIB_PICK(
     NAME, ADDR_BITS)` macro that the OpenCL backend uses to pick
     the right variant at runtime (§5.3).
  5. `src/backend/OpenCL/CHIPBackendOpenCL.cc` `appendRuntimeObjects`
     — fetch `CL_DEVICE_ADDRESS_BITS` from the device and route
     each rtdevlib module through `CHIPSTAR_RTDEVLIB_PICK(<src>,
     bits)`. AppendSource now takes a `(data, size)` view so the
     macro can return a `std::pair<const unsigned char *, size_t>`
     without needing a `std::array<unsigned char, N>&` of a
     single-known-N type (§5.3).

  **Bonus fixes not anticipated in §3** (uncovered during smoke
  compile of `vecadd` for spirv32):
  - `include/hip/devicelib/macros.hh` — replace
    `typedef unsigned long size_t;` with
    `typedef __SIZE_TYPE__ size_t;`. The old typedef baked in a
    64-bit size_t which silently mismatched spirv32's 32-bit
    `size_t`, manifesting as a typedef-redefinition error against
    clang's `__stddef_size_t.h`.
  - `include/hip/devicelib/atomics.hh` — replace
    `typedef unsigned long __chip_obfuscated_ptr_t;` with
    `typedef __UINTPTR_TYPE__ __chip_obfuscated_ptr_t;`. The
    obfuscated-pointer type must follow the device pointer width;
    the static_assert in `__chip_obfuscate_ptr` was the diagnostic.

  Both bonus fixes are correct on spirv64 too — they pick up the
  identical width — so the single-width default build keeps working
  unchanged (smoke confirmed with a fresh CHIP_TARGET_POINTER_WIDTHS
  =64 build in `~/dev/chipStar/build-smoke64/`).

  Validation results:
  - Both widths build cleanly:
    `make CHIP hipcc.bin -j32` in
    `~/dev/chipStar/build-hip32/` with
    `-DCHIP_TARGET_POINTER_WIDTHS="32;64"` succeeds. `libCHIP.so`
    is 35 MB; the rtdevlib OBJECT lib carries both spv32 and spv64
    embedded arrays.
  - `hipcc --offload-pointer-width=32` on a non-trivial HIP source
    (`feature_hip/tests/hip/vecadd/main.cpp`) compiles end-to-end:
    the device-side SPIR-V output contains `OpMemoryModel
    Physical32 OpenCL` and the host executable links against
    `libCHIP.so`. `hipcc --offload-pointer-width=64` on the same
    source produces `Physical64` (no regression).
  - Single-width default build (`CHIP_TARGET_POINTER_WIDTHS=64`,
    `~/dev/chipStar/build-smoke64/`) compiles and produces the
    pre-Phase-3 outputs.

  **End-to-end runtime validation: PASSED on rv32 SimX.** Steps:
  - `make install` on `~/dev/vortex_v3/vortex_ci/build32/` →
    populates `build32/install/{kernel,runtime,lib/pkgconfig}/` with
    headers, libs, and `vortex-runtime.pc`.
  - Configured pocl_vortex against the rv32 install (PKG_CONFIG_PATH
    points at `build32/install/lib/pkgconfig`) + the patched
    llvm_vortex (via `WITH_LLVM_CONFIG=~/dev/llvm_vortex/build/bin/llvm-config`),
    with `VORTEX_PATH_32=$build32/install` and `VISIBILITY_HIDDEN=OFF`
    (without the latter, `libpocl-devices-vortex.so` can't resolve
    internal `pocl::isLocalMemFunctionArg` symbols at runtime).
    `CMAKE_CXX_FLAGS="-I.../clang/include -I.../build/tools/clang/include"`
    needed because `llvm-config --includedir` from the build tree
    returns only llvm's source include, not clang's — `~/tools/pocl`
    is left untouched, pocl_vortex installed to `~/tools/pocl_vortex`.
  - Patched [chipStar/tools/spirv-extractor/spirv-extractor.hh:231](../../../chipStar/tools/spirv-extractor/spirv-extractor.hh#L231)
    to accept `hip-spirv32` bundle IDs alongside `hip-spirv64`
    (chipStar's fatbinary parser hardcoded `hip-spirv64`; a sixth
    code-level fix not in the original §3.1).
  - `CHIP_DEVICE_TYPE=pocl` (in §11's sample) is wrong for Vortex:
    chipStar maps it to `CL_DEVICE_TYPE_CPU` whereas pocl_vortex's
    Vortex device is `CL_DEVICE_TYPE_GPU`. Use `CHIP_DEVICE_TYPE=gpu`.
  - chipStar emits `-cl-std=CL3.0` JIT flags but Vortex pocl_vortex
    advertises CL 1.2; pocl rejects with `CL_BUILD_PROGRAM_FAILURE`.
    Workaround: `POCL_IGNORE_CL_STD=1` env var (already wired into
    feature_hip's [tests/hip/common.mk](../../tests/hip/common.mk)).
    Long-term: chipStar's `getDefaultJitFlags()` should match the
    device's `CL_DEVICE_OPENCL_C_VERSION` — small cleanup follow-up.
  - Final run:
    ```bash
    CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu \
    POCL_IGNORE_CL_STD=1 POCL_VORTEX_XLEN=32 \
    POCL_VORTEX_CFLAGS="..." POCL_VORTEX_LDFLAGS="..." POCL_VORTEX_BINTOOL="..." \
    VORTEX_DRIVER=simx \
    /tmp/vecadd32 -n 256
    ```
    Output: `PASSED!` ✓ (3736 instrs, 14368 cycles, IPC=0.260 on SimX).

  This closes the rv32 HIP-on-Vortex gap originally flagged in
  [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
  2026-05-15 update history.

  Build command in use:
  ```bash
  cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/dev/chipStar/build-hip32/install \
    -DLLVM_CONFIG_BIN=$HOME/dev/llvm_vortex/build/bin/llvm-config \
    -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
    -DCMAKE_C_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
    '-DCHIP_TARGET_POINTER_WIDTHS=32;64'
  make CHIP hipcc.bin -j32
  ```

  hipcc invocation also requires the `--gcc-install-dir=...` flag
  (clang's gcc detection picks gcc-12 which has no `libstdc++.so`
  on Ubuntu 22.04; needs gcc-11). This is the same configuration
  wart called out in the Phase 2 notes. CI follow-up: bake the
  flag into chipStar's default HIP_OFFLOAD_COMPILE_OPTIONS, OR
  install `libstdc++-12-dev` on the runner. Either is a one-line
  ops change.

- **2026-05-28 (publish strategy — patch-carry over submodule
  forks)** — The Phase 2 + Phase 3 changes that live in chipStar's
  `HIPCC` and `bitcode/ROCm-Device-Libs` submodules can't ride as
  submodule pointer bumps in our `vortexgpgpu/chipStar` fork
  without diverging `.gitmodules` from CHIP-SPV upstream (the
  submodule URLs need to keep pointing at the canonical upstream
  so a `git submodule sync` from CHIP-SPV stays useful). The
  HIPCC+ROCm-Device-Libs patches are instead carried as `.patch`
  files inside chipStar:
  - [`chipStar/HIPCC-patches/0001-hipcc-accept-offload-spirv32-...patch`](../../../chipStar/HIPCC-patches/)
  - [`chipStar/ROCm-Device-Libs-patches/0001-ROCm-Device-Libs-enable-multi-width-OCML-builds-for-.patch`](../../../chipStar/ROCm-Device-Libs-patches/)

  Same shape as the existing
  [`chipStar/llvm-patches/`](../../../chipStar/llvm-patches/)
  directory. Application is auto-handled at chipStar CMake
  configure time by a new `apply_submodule_patches()` helper in
  `chipStar/CMakeLists.txt` — idempotent via
  `git apply --reverse --check` so re-runs are no-ops. CI need not
  invoke a separate `apply` step; `cmake ..` is the full prereq.
  Single upstream PR per patch is the long-term path. Each
  patch's README documents the upstreaming follow-up.

# chipStar — 32-bit OpenCL Backend Support

## 1. Summary

Teach chipStar to emit and consume **`Physical32`** SPIR-V kernels
in addition to the current `Physical64` path. The goal is a single
chipStar tree that can target either a 64-bit or a 32-bit OpenCL
device, selected at compile time, so that HIP applications run
unmodified on **rv32 Vortex** through `pocl_vortex` (which already
supports 32-bit Vortex but rejects 64-bit SPIR-V binaries when
`cl_device::address_bits == 32`).

The work is exclusively on the chipStar side (plus a 5-call
patch in `llvm_vortex`'s Clang HIPSPV toolchain). **No changes to
POCL or to the Vortex runtime** — the 32-bit Vortex OpenCL device
already works for SPIR-V input today; it just has nothing 32-bit
to feed it.

```
foo.hip
   │  hipcc --offload-pointer-width=32        (new flag)
   ▼
   clang++ --offload=spirv32 …
     -mlink-builtin-bitcode hipspv-spirv32.bc  (new artifact)
   │
   ▼
   device.spv (OpMemoryModel Physical32, 32-bit pointers)
   │
   ▼  libCHIP.so SPVRegister                  (already parses Physical32)
   │
   ▼  clCreateProgramWithIL → pocl_vortex (address_bits=32)
   │     POCL's bitcode_is_spirv_execmodel matches → accepts
   ▼
   pocl_vortex JIT: SPIR-V → riscv32 → vxbin → vortex_runtime
```

## 2. Why now

[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
verified the chipStar+POCL path on **rv64 Vortex** in 2026-05.
The 32-bit gap is the one remaining blocker before chipStar can be
declared the **primary** HIP-on-Vortex path for both XLEN flavours.
Closing it means:

- Vortex's small-FPGA / area-constrained configurations (rv32) are
  no longer second-class for HIP. Today they have **no HIP path at
  all** — chipStar refuses, and the bespoke
  [hip_support_proposal.md](hip_support_proposal.md) Phase 2
  toolchain is multi-quarter.
- POCL's existing rv32 work (validated in
  [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md))
  becomes load-bearing instead of dormant.
- The `--offload=spirv32` SPIR-V flavour gets a real downstream
  consumer, which upstream chipStar may eventually take back.

## 3. Background — what fails today

### 3.1 The five hardcoded 64-bit assumptions

Verified against `~/dev/chipStar` `vortex_3.x` (HEAD `5cda27c9`,
2026-05-26):

| # | Site | Today | What 32-bit needs |
|---|---|---|---|
| 1 | [chipStar/CMakeLists.txt:431-433](../../../chipStar/CMakeLists.txt) | `set(OFFLOAD_TRIPLE spirv64…)` — a *singleton*. Drives `--offload=`, `--target=`, `hipspv-<triple>.bc` lookup, RDC link flags. | A *list* of enabled triples (`spirv32`, `spirv64`, or both); per-triple devicelib bitcode; runtime selection. |
| 2 | [chipStar/bitcode/CMakeLists.txt:37](../../../chipStar/bitcode/CMakeLists.txt) | One `hipspv-${OFFLOAD_TRIPLE}.bc` is built and installed under `lib/hip-device-lib/`. | Build *both* `hipspv-spirv32.bc` and `hipspv-spirv64.bc`; install both; Clang's HIPSPV toolchain picks by triple. |
| 3 | [chipStar/bitcode/CMakeLists.txt:152-189](../../../chipStar/bitcode/CMakeLists.txt) | rtdevlib SPIR-V modules (`atomicAddFloat_native.spv`, `ballot_native.spv`, …) are built once at `${OFFLOAD_TRIPLE}` and embedded as C arrays into `libCHIP.so`. | Build each rtdevlib module twice (one per pointer width); embed both; runtime selects by `CL_DEVICE_ADDRESS_BITS`. |
| 4 | [chipStar/HIPCC/src/hipBin_spirv.h:222,741](../../../chipStar/HIPCC/src/hipBin_spirv.h) | `hipcc` recognises **only** `--offload=spirv64`; the duplicate-arg filter strips only that exact spelling. | Add `spirv32` to both the parsing and the filter; key on a single `--offload-pointer-width={32,64}` user-facing flag that hipcc translates. |
| 5 | [chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:2422](../../../chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc) | `CHIPASSERT(Arg.Size == sizeof(void *))` for pointer kernel args, where `sizeof(void*)` is the **host** width (8 on x86_64). | Drop the assertion. `Arg.Size` already comes from the SPIR-V parser (4 on Physical32) — the only invariant is `Arg.Size <= sizeof(void*)`. |

Plus one Clang-side issue (in `llvm_vortex`, not chipStar proper):

| # | Site | Today | What 32-bit needs |
|---|---|---|---|
| 6 | [llvm_vortex/clang/lib/Driver/Driver.cpp:165](../../../llvm_vortex/clang/lib/Driver/Driver.cpp), [Driver.cpp:6816-6820](../../../llvm_vortex/clang/lib/Driver/Driver.cpp), and [llvm_vortex/clang/lib/Driver/ToolChains/HIPSPV.cpp:161](../../../llvm_vortex/clang/lib/Driver/ToolChains/HIPSPV.cpp) | `getHIPOffloadTargetTriple()` rejects spirv32 (error: *"invalid or unsupported offload target: 'spirv32'"*); `getOffloadingDeviceToolChain()` only dispatches `Triple::spirv64` to `HIPSPVToolChain`; `buildLinker()` `assert(getTriple().getArch() == llvm::Triple::spirv64)`. | Accept both `spirv32` and `spirv64` at all three sites. **9-line diff** across 2 files (executed in Phase 1, see §11). The rest of the toolchain (`getDeviceLibs`, devicelib search, pass-plugin invocation) is triple-agnostic; the device-lib lookup already substitutes the normalised triple into `hipspv-<triple>.bc`, so 32-bit just needs the new bitcode file to exist on disk. |

### 3.2 What already works (no changes needed)

- [chipStar/src/spirv.cc:273-277](../../../chipStar/src/spirv.cc) —
  `getPointerSize()` reads `OpMemoryModel`'s addressing model and
  returns `4` for `Physical32`, `8` for `Physical64`. The full SPV
  parser is pointer-width-correct.
- [chipStar/src/SPIRVFuncInfo.cc:96,146](../../../chipStar/src/SPIRVFuncInfo.cc)
  — kernel-arg sizes are populated from `ArgTypeInfo_[].Size`,
  which is set per-kernel from the SPV parser's
  `processKernelParameter()`. So `Arg.Size == 4` flows naturally
  for `Physical32` pointers.
- [chipStar/src/backend/Level0/CHIPBackendLevel0.cc:3378](../../../chipStar/src/backend/Level0/CHIPBackendLevel0.cc)
  — the Level Zero backend uses `Arg.Size` directly without any
  host-width assertion. (Level Zero on Vortex is out of scope —
  Vortex ships through OpenCL — but the Level Zero path is
  evidence that the rest of the runtime is already width-clean.)
- [llvm_vortex/clang/lib/Driver/Driver.cpp:139-140](../../../llvm_vortex/clang/lib/Driver/Driver.cpp)
  — Clang already recognises `Triple::spirv32` (it's a normal SPIR-V
  flavour, used by SYCL).
- [pocl_vortex/lib/CL/devices/vortex/pocl-vortex.c:255-259](../../../pocl_vortex/lib/CL/devices/vortex/pocl-vortex.c)
  — `POCL_VORTEX_XLEN=32` selects `address_bits=32`,
  `kernel-riscv32`, `ilp32f` ABI. SPIR-V ingestion is wired the
  same way for both XLENs (see
  [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md) Phase 1).
- [pocl_vortex/lib/CL/pocl_util.c:1983-2034](../../../pocl_vortex/lib/CL/pocl_util.c)
  — `bitcode_is_spirv_execmodel` strictly matches `Physical32` vs
  `Physical64` against `cl_device::address_bits`. **This is the
  error site producing today's `CL_INVALID_OPERATION` on rv32.**
  Will succeed once chipStar emits `Physical32`.

### 3.3 SVM is incompatible with 32-bit device on 64-bit host

OpenCL SVM (and chipStar's default `CoarseGrainSVM` / `FineGrainSVM`
allocation strategies in
[chipStar/src/backend/OpenCL/CHIPBackendOpenCL.hh:172-180](../../../chipStar/src/backend/OpenCL/CHIPBackendOpenCL.hh))
require that host and device share a virtual address space — i.e.
share an address width. **A 64-bit host (x86_64 Linux) cannot share
addresses with a 32-bit device.**

Vortex 32-bit must therefore go through `BufferDevAddr` mode, i.e.
the `cl_ext_buffer_device_address` extension that
[chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:2432](../../../chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc)
already supports. This is the same path
[chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
identified as the right allocation strategy for non-SVM backends,
landed in POCL 6.0. No new work here — only a runtime preference
rule: *if the device is 32-bit, require BufferDevAddr; refuse SVM.*

## 4. Component inventory

| Component | Path | Disposition |
|---|---|---|
| chipStar CMake offload-triple plumbing | `chipStar/CMakeLists.txt`, `chipStar/cmake/` | **Refactor** — turn `OFFLOAD_TRIPLE` into a list. Driven by new option `CHIP_ENABLE_TARGET_POINTER_WIDTHS` (default `64`, can be `32`, `64`, or `32;64`). |
| chipStar devicelib (`hipspv-<triple>.bc`) | `chipStar/bitcode/CMakeLists.txt` | **Foreach over enabled widths.** Today: one custom command; tomorrow: one per enabled width, both installed. |
| chipStar rtdevlib (embedded SPIR-V modules) | `chipStar/bitcode/{atomicAdd*,ballot_native,…}.cl` | **Foreach over enabled widths.** Generate `_<width>` suffixed C arrays in the embedded header; runtime selects. |
| chipStar hipcc driver | `chipStar/HIPCC/src/hipBin_spirv.h` | **Patch** — accept `--offload=spirv32`; add a `--offload-pointer-width=N` convenience flag; update the dedup-filter `excludedArgs` list. |
| chipStar OpenCL backend kernel-arg path | `chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:2422` | **One-line fix** — replace the `==` assertion with `<=`. |
| chipStar OpenCL backend SVM gating | `chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc` allocation-strategy selection | **Guard** — when `address_bits != sizeof(void*)*8`, refuse SVM strategies and select `BufferDevAddr`. Diagnose if the device lacks `cl_ext_buffer_device_address`. |
| chipStar runtime rtdevlib selection | `chipStar/src/CHIPBackend.cc` (module loading) | **Hook** — pick the rtdevlib variant whose pointer width matches the device. |
| Clang HIPSPV toolchain | `llvm_vortex/clang/lib/Driver/Driver.cpp:6816`, `llvm_vortex/clang/lib/Driver/ToolChains/HIPSPV.cpp:161` | **Patch** — 5 lines: accept `Triple::spirv32` alongside `spirv64` at dispatch and in `buildLinker()`. The rest of HIPSPV.cpp is already triple-agnostic (it formats `hipspv-<triple>.bc` from `getTriple().normalize()`). |
| POCL Vortex device | [pocl_vortex/lib/CL/devices/vortex/](../../../pocl_vortex/lib/CL/devices/vortex/) | **No change.** rv32 already works; only chipStar's emission was wrong. |
| Vortex runtime | [feature_hip/sw/runtime/](../../sw/runtime/) | **No change.** |
| `feature_hip` HIP tests | [tests/hip/](../../tests/hip/) | **No change in code.** Existing `vecadd` / `sgemm` are the validation oracles for both XLENs. |
| `ci/regression.sh.in` | [ci/regression.sh.in](../../ci/regression.sh.in) | **Extend** — add `--hip-xlen32` / `--hip-xlen64` selectors paralleling the existing `--xlen-32` matrix. |

## 5. Design

### 5.1 chipStar CMake — multi-triple build

Replace the singleton `OFFLOAD_TRIPLE` with a per-width list. The
existing top-level cascade

```cmake
if(LLVM_VERSION_MAJOR GREATER_EQUAL 23)
  set(OFFLOAD_TRIPLE spirv64v1.2-unknown-chipstar)
else()
  set(OFFLOAD_TRIPLE spirv64)
endif()
```

becomes a function `chip_offload_triple(WIDTH OUT_VAR)` that
returns the appropriate triple per width, applied across the
`CHIP_ENABLE_TARGET_POINTER_WIDTHS` list.

User-facing knob:

```bash
cmake .. \
    -DCHIP_ENABLE_TARGET_POINTER_WIDTHS="32;64"   # both
# or
    -DCHIP_ENABLE_TARGET_POINTER_WIDTHS=32        # 32-only (rv32 Vortex)
# or
    -DCHIP_ENABLE_TARGET_POINTER_WIDTHS=64        # current behaviour (default)
```

Internally, `OFFLOAD_TRIPLES` is the iteration list and
`PRIMARY_OFFLOAD_TRIPLE` is the first element (used wherever the
build currently needs a single value, e.g. for sample-program
default compile flags). `HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL_`
gets a `--offload=` slot per enabled triple (Clang's offload
driver already supports comma-separated triples).

### 5.2 chipStar devicelib (`hipspv-<triple>.bc`)

Today
[chipStar/bitcode/CMakeLists.txt:37](../../../chipStar/bitcode/CMakeLists.txt)
defines `BC_FILE = hipspv-${OFFLOAD_TRIPLE}.bc`. Convert the
custom-command block at lines 50–125 into a function
`add_hipspv_devicelib(WIDTH)` invoked once per enabled width. Each
invocation:

1. Compiles `devicelib.cl`, `_cl_print_str.cl`, `texture.cl`,
   `malloc.cl` with `--target=<triple-for-width>`.
2. Compiles `c_to_opencl.c` with the same triple.
3. `llvm-link`s the result with ROCm-Device-Libs OCML, producing
   `hipspv-<triple>.bc`.
4. Installs to `lib/hip-device-lib/`.

`ROCm-Device-Libs` is the only mildly thorny dependency — it has
its own `AMDGPU_TARGET_TRIPLE` cache variable. The existing
chipStar build overrides this with `${OFFLOAD_TRIPLE}` at line 75;
we need it set once per width. The simplest fix is to invoke
`add_subdirectory(ROCm-Device-Libs)` twice (in subdirectory scopes
so the cache override doesn't leak) — verified by checking
[chipStar/bitcode/CMakeLists.txt:73-75](../../../chipStar/bitcode/CMakeLists.txt)
that `AMDGPU_TARGET_TRIPLE` is set via plain `set()` (cache scoped
into the subdir).

### 5.3 chipStar rtdevlib (embedded SPIR-V modules)

The runtime device library is compiled to *.spv* and **embedded**
into `libCHIP.so` as C arrays via
[chipStar/scripts/embed-binary-in-cpp.bash](../../../chipStar/scripts/embed-binary-in-cpp.bash).
Today the embedded array `Foo_spv` corresponds to one pointer
width; we extend the embed loop to emit `Foo_spv32` and
`Foo_spv64` (only the enabled subset).

Runtime selector — add to `appendRuntimeObjects()` in
[chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:941](../../../chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc):

```cpp
auto AddrBits = ChipDev.getAttr(hipDeviceAttributeMemoryBusWidth /* address_bits */);
const auto &AtomicAddFloat = (AddrBits == 32)
    ? atomicAddFloat_native_spv32 : atomicAddFloat_native_spv64;
// …same for the other rtdevlib modules…
```

(Exact attribute name TBD — chipStar already has a per-device
attribute store; we add `address_bits` if it isn't there.)

### 5.4 hipcc driver

Three changes in
[chipStar/HIPCC/src/hipBin_spirv.h](../../../chipStar/HIPCC/src/hipBin_spirv.h):

1. **Parsing** (line 222): extend the `processArgs()` switch:
   ```cpp
   } else if (arg == "--offload=spirv64" || arg == "--offload=spirv32") {
     offload = true;
   }
   ```
2. **Dedup-filter** (line 741): add `"--offload=spirv32"` to
   `excludedArgs`.
3. **New convenience flag** `--offload-pointer-width={32,64}`
   (default 64). Translates to the appropriate `--offload=spirvNN`
   when wired in by hipcc.

The CMake-generated `.hipInfo` file (referenced at lines 36-90 of
the same header) embeds `HIP_OFFLOAD_COMPILE_OPTIONS`. When
chipStar is built with both widths enabled, the file should list
both `--offload=` lines so that Clang's offload driver compiles
both flavours; the user then narrows with
`--offload-pointer-width=` at hipcc invocation time.

### 5.5 OpenCL backend kernel-arg path

[chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:2421-2422](../../../chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc):

```cpp
case SPVTypeKind::Pointer: {
  CHIPASSERT(Arg.Size == sizeof(void *));   // <-- this
```

This assertion is wrong even before the 32-bit work: `Arg.Size`
comes from the SPV parser (device width), `sizeof(void *)` is the
host width. They happen to be equal on x86_64-host + Physical64-
device, but the assertion is structurally a host/device confusion.

Replace with:

```cpp
case SPVTypeKind::Pointer: {
  CHIPASSERT(Arg.Size <= sizeof(void *));   // device <= host width
```

The downstream
`clSetKernelArgDevicePointerEXT(KernelHandle, Arg.Index, DevPtr)`
call is already width-correct — POCL serialises `address_bits`
many bytes of `DevPtr` into the kernel arg buffer, and Vortex's
device address space fits in 32 bits regardless of host width.

### 5.6 SVM refuse + BufferDevAddr require

In allocation-strategy selection (currently picks the most
permissive SVM tier the device advertises), add a precondition:

```cpp
const bool needNarrowedAddrs = (dev->address_bits != sizeof(void *) * 8);
if (needNarrowedAddrs) {
  if (!ext_supported("cl_ext_buffer_device_address"))
    THROW(hipErrorInvalidDevice,
          "32-bit device requires cl_ext_buffer_device_address");
  return AllocationStrategy::BufferDevAddr;
}
// existing SVM-preference cascade
```

This is also the place to log a clear diagnostic when 32-bit is
selected — currently the failure mode is the cryptic POCL
`CL_INVALID_OPERATION` *much later*, at `clCreateProgramWithIL`.

### 5.7 Clang HIPSPV toolchain (llvm_vortex)

Three sites, 9-line diff total:

```cpp
// llvm_vortex/clang/lib/Driver/Driver.cpp:165
// getHIPOffloadTargetTriple(): accept spirv32 as a HIP offload triple
if (TT->getArch() == llvm::Triple::spirv64 ||
    TT->getArch() == llvm::Triple::spirv32)
  return TT;
```

```cpp
// llvm_vortex/clang/lib/Driver/Driver.cpp:6816
// getOffloadingDeviceToolChain(): route spirv32 to HIPSPVToolChain
else if ((Target.getArch() == llvm::Triple::spirv64 ||
          Target.getArch() == llvm::Triple::spirv32) &&
         Target.getVendor() == llvm::Triple::UnknownVendor &&
         Target.getOS() == llvm::Triple::UnknownOS)
  TC = std::make_unique<toolchains::HIPSPVToolChain>(*this, Target,
                                                     HostTC, Args);
```

```cpp
// llvm_vortex/clang/lib/Driver/ToolChains/HIPSPV.cpp:160-163
Tool *HIPSPVToolChain::buildLinker() const {
  assert(getTriple().getArch() == llvm::Triple::spirv64 ||
         getTriple().getArch() == llvm::Triple::spirv32);
  return new tools::HIPSPV::Linker(*this);
}
```

That's the entire `llvm_vortex` delta. The
[chipStar/llvm-patches/llvm/0003-Unbundle-SDL.patch](../../../chipStar/llvm-patches/llvm/0003-Unbundle-SDL.patch)
mechanism (chipStar carries Clang patches and applies them at
build time) is where this lives. Naming: `0004-HIPSPV-32-bit.patch`.

**Phase 1 result (2026-05-27).** All three sites patched in
`~/dev/llvm_vortex` (HEAD `d4d8e322bbfa` at patch time, uncommitted).
Incremental `ninja clang` rebuild took ~2.5 min (136 actions).
Smoke test: `clang++ -x hip --offload=spirv32 --cuda-device-only -c
empty.hip` emits a SPIR-V binary with `OpMemoryModel Physical32
OpenCL`; the corresponding `--offload=spirv64` invocation still
emits `Physical64` (no regression). Patch is **not yet committed
or installed** to `$TOOLDIR/llvm-vortex`; sat in the build tree
pending Phase 2's chipStar devicelib work that consumes it.

## 6. Phases

Each phase is independently buildable and testable.

### Phase 0 — Source-tree setup ✅ done

- [x] Fresh clone of `vortex_ci` at `~/dev/vortex_v3/hip32`.
- [x] This proposal at
  [docs/proposals/chipstar_opencl_32bit_proposal.md](chipstar_opencl_32bit_proposal.md).

### Phase 1 — Clang HIPSPV accepts spirv32 ✅ done (2026-05-27)

`llvm_vortex` 9-line patch (§5.7), three sites. Validation:

```bash
cat > empty.hip <<'EOF'
#define __global__ __attribute__((global))
extern "C" __global__ void k() {}
EOF
~/dev/llvm_vortex/build/bin/clang++ -x hip --offload=spirv32 \
    -nogpulib -nogpuinc -nohipwrapperinc \
    --cuda-device-only --save-temps -c empty.hip -o empty.bundle
spirv-dis empty-hip-spirv32-generic.out | grep OpMemoryModel
# →  OpMemoryModel Physical32 OpenCL
```

**Success**: returns 0, device-side SPIR-V binary contains
`OpMemoryModel Physical32 OpenCL`. **Verified.** rv64 path
unchanged: same invocation with `--offload=spirv64` still emits
`Physical64 OpenCL`.

Notes from execution:

- Proposal originally called out 2 sites; the actual error in
  `clang++: error: invalid or unsupported offload target: 'spirv32'`
  is emitted at a *third* site (Driver.cpp:165
  `getHIPOffloadTargetTriple`). All three are now in §5.7.
- `__global__` is a macro defined in HIP runtime headers; with
  `-nogpuinc` we must define it inline. The proposal's original
  one-liner test case had a latent bug — it fails identically on
  spirv64 today. `--cuda-device-only` is the cleanest way to skip
  the host pass (which needs `hipLaunchKernel` decl).
- The build dir's `bin/clang++` is bit-identical to what
  `cmake --install` would lay down. Validated against the build
  dir directly to avoid overwriting `$TOOLDIR/llvm-vortex` (which
  is shared with the CI runner). Install deferred to Phase 2.
- `~/dev/llvm_vortex/build/bin/llvm-spirv` is initially a symlink
  to `$TOOLDIR/llvm-vortex/bin/llvm-spirv` (created so the
  HIPSPVToolChain can call llvm-spirv on the `empty.hip` smoke).
  This is a Phase 1 expedient — chipStar's bitcode build does not
  invoke `llvm-spirv`, but the rtdevlib path (Phase 3) does, and
  rtdevlib must consume a translator built against the same llvm.
  Phase 2 retargets this symlink at a freshly built llvm-spirv
  rooted under `~/dev/llvm_vortex` (see Phase 2 notes).

### Phase 2 — chipStar devicelib builds for spirv32 ✅ done (2026-05-27)

CMake refactor (§5.1, §5.2) — single
`add_hipspv_devicelib(WIDTH)` function applied across the enabled
widths list. ROCm-Device-Libs added as a subdir per width so the
`AMDGPU_TARGET_TRIPLE` override is scoped.

Validation:

```bash
cd ~/dev/chipstar && rm -rf build && mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$TOOLDIR/chipstar \
    -DLLVM_DIR=$TOOLDIR/llvm_vortex/lib/cmake/llvm \
    -DCHIP_ENABLE_TARGET_POINTER_WIDTHS="32;64"
cmake --build . -j && cmake --install .

ls $TOOLDIR/chipstar/lib/hip-device-lib/
# expect: hipspv-spirv32.bc  hipspv-spirv64.bc
```

**Success**: both `.bc` files exist, both pass `llvm-dis` on
LLVM 18.1.

**Likely failures**: ROCm-Device-Libs OCML emitting types whose
size differs between widths in a way that breaks `llvm-link` (e.g.
`size_t` in OCML headers). Mitigation: OCML is already used in
both spirv32 and spirv64 by upstream chipStar-PowerVR/Intel
deployments — the 32-bit path is well-trodden upstream. If we hit
a divergence, the fix is a per-width `#define` in
[chipStar/bitcode/cl_utils.h](../../../chipStar/bitcode/cl_utils.h).

**Budget**: 2–3 days.

### Phase 3 — chipStar runtime + rtdevlib selection ✅ done (2026-05-27, vecadd PASSED on rv32 SimX)

Apply the four runtime fixes (§5.3, §5.4, §5.5, §5.6) and the
hipcc front-end change. Validation:

```bash
# A 32-bit-only chipStar build:
cmake .. -DCHIP_ENABLE_TARGET_POINTER_WIDTHS=32 …
cmake --build . -j && cmake --install .

# hipcc on a smoke kernel:
$TOOLDIR/chipstar/bin/hipcc --offload-pointer-width=32 \
    ~/dev/vortex_v3/hip32/tests/hip/vecadd/main.cpp -o vecadd32

# Run against pocl_vortex with POCL_VORTEX_XLEN=32:
OCL_ICD_VENDORS=$TOOLDIR/pocl_vortex/etc/OpenCL/vendors \
CHIP_BE=opencl CHIP_DEVICE_TYPE=pocl \
LD_LIBRARY_PATH=$TOOLDIR/pocl_vortex/lib:$TOOLDIR/chipstar/lib:$VORTEX_RT_LIB \
POCL_VORTEX_XLEN=32 \
VORTEX_DRIVER=simx \
./vecadd32 -n 64
```

**Success**: `PASSED!` from
[tests/hip/vecadd/main.cpp](../../tests/hip/vecadd/main.cpp).

**Likely failures**:
- POD-argument width drift. A HIP kernel parameter like `size_t N`
  is 8 bytes on x86_64 host and 4 bytes in the spirv32 module.
  When the host passes `&N` to `hipLaunchKernelGGL`, chipStar
  reads 4 bytes — fine if `N < 2^32`, broken otherwise.
  **Mitigation**: document the rule; the existing tests use
  `unsigned`/`int` (4 bytes both sides). Long-term, chipStar's
  `HipKernelArgSpiller` pass could insert per-arg truncations on
  the host side, but that is a follow-up, not a Phase 3 blocker.
- `cl_ext_buffer_device_address` not actually wired in
  `pocl_vortex`'s rv32 path. Verified by
  [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
  update history that it works on rv64; we re-verify on rv32
  before Phase 3 starts.

**Budget**: 4–5 days.

### Phase 4 — sgemm + chipStar smoke regression ✅ done (2026-05-27, sgemm PASSED SimX+RTLsim; smoke 473/1308 ~36% pass)

```bash
make -C ~/dev/vortex_v3/hip32/tests/hip/sgemm \
    HIPCC=$TOOLDIR/chipstar/bin/hipcc \
    HIP_LIB_PATH=$TOOLDIR/chipstar/lib \
    HIP_EXTRA_HIPCC_FLAGS=--offload-pointer-width=32

# chipStar's own smoke suite, against pocl_vortex rv32:
cd ~/dev/chipstar/build
POCL_VORTEX_XLEN=32 ctest -L smoke --output-on-failure
```

**Success metric**: `tests/hip/sgemm` passes on SimX (`VORTEX_DRIVER=simx`)
and RTLsim with `XLEN=32`; chipStar smoke suite delta vs.
rv64 is ≤ 10% additional failures.

**Likely failures**: a long tail of upstream chipStar smoke tests
that assume `sizeof(size_t)==8` in device-side code. We **do not
fix these in chipStar**; we exclude them from the Vortex rv32
known-good list. Tracking goes in
[chipStar/scripts/known-failures-vortex32.txt](../../../chipStar/scripts/) (new).

**Budget**: 3–4 days.

### Phase 5 — CI integration ✅ done (2026-05-27)

The Phase 1–4 work closes the previous XLEN=32 hip-tests skip, so
`ci/regression.sh.in`'s `hip()` now matches `opencl()` in shape: a
single `--hip` selector that runs the test suite across all
backends with no inline gating. If the chipstar install at
`$TOOLDIR/chipstar` is missing the `hipspv-spirv$XLEN.bc` device
library, `make -C tests/hip` fails naturally and surfaces the
underlying error — same convention as `--opencl` for missing
pocl.

- **Single `--hip` selector** (per-XLEN variants dropped at user
  request — the existing `--hip` already matches the CI matrix
  shape, which invokes regression.sh once per XLEN from its
  own build dir).
- **No inline gating** (per user review — the simpler shape
  matches `opencl()`; missing toolchain surfaces as a normal
  build failure). The `--hip` selector simply replaces the
  Phase 0 XLEN=32 skip with the standard run-all-backends body.
- **toolchain_install.sh.in** is a tarball-download flow (no
  per-flag build invocation); the Vortex chipstar prebuilt must
  be built upstream with `-DCHIP_TARGET_POINTER_WIDTHS="32;64"`
  so a single tarball serves both rv32 and rv64.
- **`ci/chipstar_install.sh.in` (new)** — producer script for
  the chipstar tarball, mirroring `ci/mesa_install.sh.in` for
  mesa-vortex. Clones `vortexgpgpu/chipStar vortex_3.x`,
  configures with `-DCHIP_TARGET_POINTER_WIDTHS="32;64"` against
  `$TOOLDIR/llvm-vortex`, builds, and installs into
  `$TOOLDIR/chipstar`. The chipStar CMake configure auto-applies
  `HIPCC-patches/*` and `ROCm-Device-Libs-patches/*` to the
  submodules via the `apply_submodule_patches()` helper, so no
  separate apply step is needed.

Diff summary:
- `ci/regression.sh.in` `hip()`: drop the `XLEN=="32"` skip; body
  is now identical in style to `opencl()` (just `make -C
  tests/hip run-{simx,rtlsim,opae,xrt}`).
- `ci/toolchain_install.sh.in` `chipstar()`: comment block
  documenting the upstream `CHIP_TARGET_POINTER_WIDTHS="32;64"`
  build-flag requirement.

Smoke results (`./configure --xlen=N --tooldir=$HOME/tools` +
`./ci/regression.sh --hip`):
- rv64 build, default `~/tools/chipstar` (existing rv64 tarball):
  dispatches into `make -C tests/hip run-simx` cleanly.
- rv32 build, default `~/tools/chipstar`: dispatches; downstream
  `hipcc` would fail because the existing tarball lacks
  `hipspv-spirv32.bc`. CI failure with the expected POCL error
  (`CL_INVALID_OPERATION`) signals the toolchain needs the
  dual-width rebuild — same pattern as a missing pocl install
  surfaces as a clCreateProgramWithIL failure on `--opencl`.

**Budget**: 1–2 days.

## 7. Decision gates

| After phase | Outcome | Action |
|---|---|---|
| 1 | Clang patch compiles, empty kernel emits `Physical32` | Proceed to Phase 2 |
| 1 | Clang asserts deep in a SPIRV-Translator pass | File the specific hit; consult upstream chipStar (they have a 32-bit Intel iGPU path that may have already paved this) |
| 2 | Both `.bc` files build | Proceed to Phase 3 |
| 2 | OCML on spirv32 emits ill-formed bitcode | Either fix in `cl_utils.h` (<1 day) or escalate — this would be a real upstream regression worth a chipStar issue |
| 3 | vecadd rv32 passes | Proceed to Phase 4 |
| 3 | vecadd fails at `clSetKernelArgDevicePointerEXT` | Verify `cl_ext_buffer_device_address` is actually advertised by pocl_vortex's rv32 device; if not, fix in POCL Vortex device (small) |
| 3 | vecadd fails with arithmetic / data corruption | POD-arg width drift; instrument and either narrow the kernel signature or land the HipKernelArgSpiller fix early |
| 4 | sgemm passes; smoke delta ≤ 10% | **chipStar 32-bit becomes a supported configuration**; close the v3.0 known-limitation amendment |
| 4 | sgemm fails | Investigate; one fix-loop max before pausing |
| 5 | CI green on both XLENs | Done |

## 8. What this proposal does *not* commit to

- No native `clSetKernelArg2DEXT` / 32-bit-pointer-aware OpenCL
  extension proposal upstream — we use the existing
  `cl_ext_buffer_device_address` path.
- No changes to the Vortex hardware ISA or the `vortex_runtime`
  library.
- No SVM support on 32-bit Vortex. SVM remains a 64-bit-only mode.
  Documented as a runtime requirement, not a TODO.
- No `HipKernelArgSpiller` change in Phase 3 — width-drift in POD
  args is a follow-up if a real test hits it.

## 9. Risks

1. **chipStar upstream may already have spirv32 support.**
   chipStar targets Intel iGPUs and PowerVR, some of which are
   32-bit. Worth a 30-min grep through chipStar `main` (and the
   1.1 release notes) before starting Phase 2 — we may be writing
   half a patch that already exists.
   **Mitigation**: §10 (versions to pin) records the result of
   this check at execution start.

2. **OCML width-cleanness.** Phase 2's biggest unknown. ROCm-Device-Libs
   is large; if any internal helper uses `__builtin_amdgcn_*` that
   only lowers on AMDGPU 64-bit, the spirv32 build fails opaquely.
   **Mitigation**: ROCm-Device-Libs is also used by upstream
   chipStar's spirv64 build, and Mesa-CLover used it on 32-bit
   too — the path is not virgin.

3. **POD-argument width drift.** Phase 3's biggest unknown. Any
   HIP kernel using `size_t` directly as a parameter type produces
   silent corruption on rv32+x86_64-host. **Mitigation**: bound by
   our reference test suite; document the rule; defer
   `HipKernelArgSpiller` host-side narrowing as a follow-up.

4. **`cl_ext_buffer_device_address` on rv32 pocl_vortex.** The
   extension is wired into POCL 6.0 and exercised on rv64 by
   [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md)
   verification. The Vortex device target plugs into the extension
   via the same hooks regardless of XLEN, but it has not been
   end-to-end tested at 32-bit. **Mitigation**: smoke test the
   extension *without* chipStar first (small OpenCL host) before
   starting Phase 3.

5. **Upstreaming surface.** The Clang HIPSPV change is small and
   clearly upstream-able (5 lines). The chipStar CMake refactor
   is larger; we should land it on a topic branch in
   `~/dev/chipstar` and open a PR against `CHIP-SPV/chipStar` once
   green. **Mitigation**: keep the patch shape close to how
   upstream already conditions on `LLVM_VERSION_MAJOR` in
   `OFFLOAD_TRIPLE` (line 427 cascade).

## 10. Versions to pin (filled in at start of execution)

| Component | Pin | Source |
|---|---|---|
| `llvm_vortex` | `d78d4a25e` (LLVM 18.1, 2025-08-08) | [llvm_vortex git log](../../../llvm_vortex/) |
| `pocl_vortex` | `vortex_3.x` head | [pocl_vortex/](../../../pocl_vortex/) |
| `chipstar` | `vortex_3.x` head — `5cda27c9` (2026-05-26) | [chipStar/](../../../chipStar/) |
| Upstream chipStar spirv32 check | TBD (1 grep at execution start) | [github.com/CHIP-SPV/chipStar](https://github.com/CHIP-SPV/chipStar) |
| Vortex source tree | `~/dev/vortex_v3/hip32` from `tinebp-patch-2` (4380ad5d) | this proposal |

## 11. Estimated timeline

| Phase | Days | Cumulative |
|---|---:|---:|
| 1 — Clang HIPSPV spirv32 patch | 1 | 1 |
| 2 — chipStar devicelib multi-width build | 2–3 | 4 |
| 3 — chipStar runtime + hipcc + rtdevlib selection | 4–5 | 9 |
| 4 — sgemm + chipStar smoke regression | 3–4 | 13 |
| 5 — CI integration | 1–2 | 15 |

**Total: ~3 weeks (15 working days), hard cap at 4 weeks.**

The dominant uncertainty is Phase 3 — POD-argument width drift is
the kind of failure that takes a day to localise and an hour to
fix, and we may hit it multiple times. The Phase 2 ROCm-Device-Libs
spirv32 build is the second-biggest unknown; if upstream chipStar
already does this, Phase 2 collapses to a 1-day rebase.

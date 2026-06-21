# Vulkan Conformance Testing on Vortex via mesa_vortex

## 1. Goal

Stand up an end-to-end flow that runs the Khronos **Vulkan Conformance Test
Suite (VK-GL-CTS / `dEQP-VK`)** against the Vortex GPU through the
`mesa_vortex` stack, and report a reproducible, trackable pass/fail/waiver
result set. The deliverable is a CI-runnable harness plus a conformance
status report — not (yet) a submission for an official Khronos conformance
claim, which has additional process requirements (§9).

## 2. Background: what "running Vulkan on Vortex" means

The Vulkan stack used here has four layers:

```
  application / dEQP-VK
        │   Vulkan API (libvulkan loader)
        ▼
  Vulkan loader  ──VK_ICD_FILENAMES──►  lavapipe ICD  (lvp_icd.x86_64.json)
        │                                   │   Mesa Vulkan frontend
        ▼                                   ▼
  lavapipe (LVP)  ──GALLIUM_DRIVER=vortexpipe──►  vortexpipe (Gallium driver)
        │                                   │   NIR → LLVM IR → .vxbin
        ▼                                   ▼
                              libvortex.so (vortex2.h)
                                          │   dlopen libvortex-<backend>.so
                                          ▼
                         simx │ rtlsim │ opae │ xrt  (Vortex device)
```

- **lavapipe (LVP)** is Mesa's software Vulkan driver. It is already a
  Khronos-conformant Vulkan 1.3 implementation (it reports
  `conformanceVersion = 1.3.1.1`, `apiVersion = 1.3/1.4`). It owns all the
  Vulkan API surface: instance/device, descriptor sets, render passes,
  pipelines, synchronization, the SPIR-V → NIR front end, and the host-side
  acceleration-structure builders.
- **vortexpipe** is a Gallium `pipe_screen`/`pipe_context` that sits in front
  of lavapipe's gallium backend. It intercepts compute/vertex/fragment
  dispatches, translates the shader NIR to LLVM IR and then to a Vortex
  `.vxbin` kernel (`vp_nir_to_llvm` + `vp_compile`), and launches it on the
  Vortex device via `vortex2.h`. Fixed-function graphics work (rasterization,
  texturing, blending) is offloaded to the Vortex RASTER/TEX/OM units; ray
  queries are offloaded to the RTU.
- When vortexpipe **cannot** offload a shader (an unsupported NIR op/intrinsic,
  a workgroup larger than the device CTA, a non-opaque ray query, …) it
  **falls back to llvmpipe** — the CPU rasterizer lavapipe inherits — and the
  result is still correct. This fallback is the crux of conformance scoping
  (§4).

**Key consequence for conformance.** Because lavapipe+llvmpipe is already
conformant, a naïve full-suite run with fallback enabled would mostly pass —
but it would be measuring *llvmpipe*, not Vortex. The meaningful conformance
question is: **for the subset of the suite that vortexpipe actually offloads
to Vortex, does Vortex produce conformant results?** The harness must
therefore distinguish "passed on Vortex" from "passed on the CPU fallback."
The existing `tests/vulkan` harness already encodes exactly this distinction
via `MESA_VORTEX_STRICT` (see §3).

## 3. What already exists

The in-tree `tests/vulkan/` harness ([common.mk](../../tests/vulkan/common.mk))
is a working, smaller-scale precedent that the conformance flow extends:

- Selects the driver with `VK_ICD_FILENAMES=<lvp_icd.json>` +
  `GALLIUM_DRIVER=vortexpipe`.
- Picks the Vortex backend with `VORTEX_DRIVER=simx|rtlsim|opae|xrt`
  (`run-simx`, `run-rtlsim`, `run-opae`, `run-xrt` targets). The same `.vxbin`
  runs on every backend.
- Points vortexpipe's runtime `.vxbin` compiler at the tree with
  `VORTEX_HOME` / `VORTEX_TOOLDIR` / `VORTEX_BUILD`, and sets
  `MESA_VORTEX_XLEN=32` (Vortex CI is **32-bit only**).
- **`MESA_VORTEX_STRICT=1`** makes vortexpipe *refuse* to silently fall back to
  llvmpipe: any missing-kernel / runtime / NIR-translation gap becomes a
  `MESA: error` instead of a CPU-computed green light. The harness'
  `check_run` gate then fails the test on (a) non-zero exit, (b) any
  `MESA: error`, or (c) a device string that does not contain `vortex`. This
  is precisely the mechanism that proves a test ran **on Vortex**.
- Current coverage: `compute`, `shmem`, `cflow`, `triangle`, `draw3d`,
  `depth`, `textured`, plus ray-query (`rtquery`, `rtquery_id`, `raytrace`).
  These exercise the compute, graphics, and RTU paths end-to-end on the simx
  backend.

The conformance effort is, in effect, **scaling this harness from a dozen
hand-written tests to the ~hundreds-of-thousands-of-case dEQP-VK suite**, with
result bucketing and a fallback-aware pass criterion.

## 4. Conformance scope: the three buckets

Every dEQP-VK case lands in one of three buckets, determined by running it
once in strict mode and once in fallback mode:

| Bucket | Strict (Vortex-only) | Fallback (llvmpipe allowed) | Meaning |
|---|---|---|---|
| **A — Vortex-conformant** | PASS | PASS | Offloaded to Vortex and correct. The real conformance signal. |
| **B — CPU-only** | `MESA: error` / skip | PASS | Uses a feature vortexpipe does not yet offload; lavapipe/llvmpipe handles it correctly. A gap list, not a failure. |
| **C — Vortex bug** | FAIL (wrong result / crash / hang) | PASS | Offloaded to Vortex but produced a non-conformant result. The bug backlog. |

The conformance report is "**Bucket A pass rate over the offloadable subset, C
== 0**". Bucket B is the driver-feature roadmap (it shrinks as vortexpipe gains
NIR coverage); bucket C must be driven to zero.

Initial scope is bounded by what vortexpipe offloads today:

- **In scope now:** `dEQP-VK.compute.*` (the cleanest match — compute kernels
  are vortexpipe's native path), `dEQP-VK.draw.*` / `dEQP-VK.rasterization.*` /
  `dEQP-VK.texture.*` (graphics offload via RASTER/TEX/OM), and
  `dEQP-VK.ray_query.*` (RTU; the opaque path runs on Vortex today).
- **Out of scope initially:** sparse resources, ray-tracing *pipelines* (SBT;
  vortexpipe routes these to the SW fallback by design — see the RTU
  proposals), tessellation/geometry shaders if unsupported, video, and any
  extension lavapipe itself does not advertise. These are bucket B by
  construction and excluded from the headline number with an explicit waiver
  list.

## 5. End-to-end pipeline

### 5.1 Build the conformant Vulkan stack

1. **mesa_vortex** — build lavapipe + vortexpipe and install to the
   `MESA_PATH` prefix (`$TOOLDIR/mesa-vortex`):
   - meson: `-Dgallium-drivers=llvmpipe,vortexpipe -Dvulkan-drivers=swrast
     -Dllvm=enabled -Dcpp_rtti=false` (RTTI off to match the RTTI-disabled
     `llvm-vortex`), prefix = the mesa-vortex install dir.
   - `ninja -C build-rtu && ninja -C build-rtu install`.
   - This produces `lib/.../libvulkan_lvp.so`, `libgallium-*.so` (contains
     vortexpipe), and `share/vulkan/icd.d/lvp_icd.x86_64.json`.
2. **Vortex runtime** — built out-of-tree under the configure build dir; the
   per-backend `libvortex-<name>.so` is built on demand by the run recipe.
3. **Toolchain** — `llvm-vortex` + the rv32 GNU toolchain / libc32 / libcrt32
   under `$HOME/tools`; vortexpipe shells out to these to compile each shader's
   `.vxbin` at run time.

### 5.2 Build VK-GL-CTS

1. Clone Khronos **VK-GL-CTS** at the tag matching lavapipe's advertised
   conformance version (1.3.x line), and run `external/fetch_sources.py`.
2. Configure with the standard dEQP CMake against the **system Vulkan
   loader** (not a Mesa-internal loader): `cmake -DDEQP_TARGET=default`,
   build `deqp-vk`.
3. The mustpass list `external/vulkancts/mustpass/main/vk-default.txt` is the
   canonical case set; subsets (e.g. `…/compute.txt`) drive the phased plan.

### 5.3 Point the loader at Vortex

dEQP-VK uses the installed Vulkan loader and selects the ICD via environment —
identical to the `tests/vulkan` harness:

```sh
export VK_ICD_FILENAMES=$MESA_PATH/share/vulkan/icd.d/lvp_icd.x86_64.json
export GALLIUM_DRIVER=vortexpipe
export MESA_VORTEX_XLEN=32
export MESA_VORTEX_STRICT=<0|1>          # 0 = bucket survey, 1 = Vortex-only
export VORTEX_HOME=<repo> VORTEX_TOOLDIR=$HOME/tools VORTEX_BUILD=<build>
export VORTEX_DRIVER=simx                # or rtlsim / xrt
export LD_LIBRARY_PATH=$MESA_PATH/lib:$LLVM_PATH/lib:$ZSTD_PATH/lib:$VORTEX_RT_LIB
```

Because the loader enumerates exactly one ICD (lavapipe), `deqp-vk` selects
the Vortex device with no code change. A one-line guard (`vulkaninfo | grep
-i vortex`, or dEQP's `--deqp-log-filename` device banner) confirms the right
device before a run, mirroring `check_run`'s `device:.*vortex` gate.

### 5.4 Execute + bucket

Run dEQP-VK with its batch runner (`deqp-vk --deqp-caselist-file=<subset>
--deqp-log-filename=<qpa>`), twice per subset (strict, then fallback), and
post-process the two `.qpa` logs into the §4 buckets. A small reducer script
maps `(strict_result, fallback_result)` → bucket and emits a CSV/JSON summary
plus the bucket-B waiver list and bucket-C bug list.

### 5.5 Backend selection (functional vs. scale)

| Backend | Use | Speed | Notes |
|---|---|---|---|
| **simx** | Primary functional conformance | ~10⁴–10⁶ instr/s per case | Cycle-approximate C++ model; deterministic; the default for bring-up and CI. |
| **rtlsim** | RTL spot-checks | very slow | Verilator on the real RTL; run a curated bucket-A regression subset, not the full suite. |
| **xrt (FPGA)** | Scale + real-HW sign-off | fastest real path | Alveo U55C bitstream; the only practical way to run the full mustpass in bounded wall-clock. Uses `run-xrt` env (XRT_XCLBIN_PATH, EMCONFIG_PATH, …). |

Strategy: **bring up and debug on simx** (deterministic, traceable), then run
the **full mustpass on FPGA via xrt** for throughput, and use **rtlsim** only
to confirm a representative subset matches RTL (parity with the SimX↔RTL
effort).

## 6. Phased plan

- **Phase 0 — Harness.** Add a `tests/conformance/vulkan/` driver (Makefile +
  reducer script) reusing `tests/vulkan/common.mk`'s env/ICD/strict machinery.
  Wire `run-simx` / `run-xrt`. Validate against a 10-case smoke list. *Exit:*
  one command produces a bucketed summary.
- **Phase 1 — Compute conformance (simx).** Run `dEQP-VK.compute.*` in both
  modes; populate buckets. Drive bucket C to zero (these are Vortex compute
  bugs). *Exit:* compute bucket-A pass rate reported, C == 0.
- **Phase 2 — Graphics conformance (simx).** Add `dEQP-VK.draw.*`,
  `…rasterization.*`, `…texture.*`, `…pipeline.*`. Triage RASTER/TEX/OM
  offload correctness; grow bucket A as graphics NIR coverage improves.
- **Phase 3 — Ray query (simx).** Run `dEQP-VK.ray_query.*`; opaque cases are
  bucket A today, non-opaque/AABB are bucket B (SW fallback) until the RTU
  callback dispatcher lands.
- **Phase 4 — Scale-out (FPGA/xrt).** Re-run the accumulated bucket-A set plus
  the full mustpass on the U55C bitstream; reconcile any simx-vs-FPGA deltas.
- **Phase 5 — CI + tracking.** Nightly bucket-A regression on the self-hosted
  runner; a checked-in conformance report (pass rate, waiver list, bug list)
  updated per run; alert on bucket-A regressions and new bucket-C entries.

## 7. Result tracking & waivers

- **Report artifact:** a versioned `conformance_status.md` (or JSON) per run:
  suite tag, mesa_vortex commit, backend, per-group bucket-A/B/C counts, and
  the full waiver/bug lists.
- **Waivers (bucket B):** each entry names the unsupported NIR
  op/intrinsic/feature and links the driver-roadmap item that would move it to
  bucket A. Waivers are explicit and reviewed, never silent.
- **Bugs (bucket C):** each is a reproducer (caselist of one) plus the strict
  `MESA: error` / wrong-image diff; these feed the Vortex bug backlog and,
  where useful, become hand-written `tests/vulkan/*` regressions.

## 8. Practical considerations & risks

- **Suite size vs. simx speed.** dEQP-VK is enormous (hundreds of thousands of
  cases). The full suite on simx is impractical; mitigate with (a) phased
  subsets, (b) parallel sharded runs across cores via caselist splitting, and
  (c) FPGA for the full pass. Per-case `.vxbin` compilation also dominates
  time — cache compiled kernels keyed by SPIR-V hash to avoid recompiling
  shared shaders.
- **Determinism.** simx is deterministic; FPGA timing is not, but
  conformance is a functional (image/result) check, so HW non-determinism is
  acceptable as long as results are correct. Flaky cases are themselves
  bucket-C signals.
- **32-bit only.** All conformance runs are `XLEN=32`; do not enable build64.
- **Fallback masking.** The single biggest methodology risk is counting
  llvmpipe passes as Vortex passes. The strict/fallback double-run plus the
  `device:.*vortex` and `MESA: error` gates exist specifically to prevent
  this; the reducer must treat a strict-mode fallback as **not** bucket A.
- **Long-running stability.** Large runs surface hangs/timeouts (e.g. the
  known RTU miss-callback trap-resume issue). Wrap each case with a timeout
  and classify timeouts as bucket C.
- **Toolchain coupling.** vortexpipe compiles `.vxbin` with `llvm-vortex` at
  run time; a toolchain/runtime skew reproduces the class of codegen failures
  already seen (e.g. the kernel-entry/bring-up bug). Pin toolchain + mesa
  commits in the report.

## 9. Out of scope (this proposal)

- **Official Khronos conformance submission.** A formal claim requires running
  the exact mustpass for a fixed API version on the target HW, the Adopter
  process, and a submission package. This proposal builds the *technical*
  flow that such a submission would later use; it does not undertake the
  process itself.
- **Features lavapipe/vortexpipe do not advertise** (sparse, RT pipelines,
  video, tessellation/geometry if unsupported) — tracked as bucket B waivers,
  not targeted for bucket A here.

## 10. Deliverables

1. `tests/conformance/vulkan/` harness (Makefile + env wiring + reducer
   script) reusing the existing `tests/vulkan` ICD/strict machinery.
2. Phased bucketed results for compute → graphics → ray_query on simx.
3. Full mustpass run on the U55C FPGA via xrt.
4. A checked-in, versioned conformance status report (bucket-A pass rate,
   waiver list, bug list) and a nightly CI job that regresses bucket A and
   flags new bucket-C entries.

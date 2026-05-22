**Date:** 2026-05-17
**Status:** Draft — Phase 0 ✅, Phase 1 ✅, Phase 2 in progress (#1 of 6 done; Shape C)
**Author:** Blaise Tine
**Related:**
[gfx_migration_proposal.md](gfx_migration_proposal.md),
[hip_support_proposal.md](hip_support_proposal.md),
[simx_v3_proposal.md](simx_v3_proposal.md),
[llvm_vortex_v3_proposal.md](llvm_vortex_v3_proposal.md).

### Update history

- **2026-05-17** — Initial draft.
- **2026-05-21** — Phase 0 baseline reached.
  - [ci/mesa_install.sh.in](../../ci/mesa_install.sh.in) added —
    installs build deps and builds Mesa (lavapipe + llvmpipe) from
    `MESA_REPO`@`MESA_REV` into `$(TOOLDIR)/mesa-vortex`.
  - Vanilla **Mesa 25.1.0** lavapipe builds and runs end-to-end:
    `vulkaninfo` reports `llvmpipe (LLVM 20.1.8, 256 bits)` /
    `DRIVER_ID_MESA_LLVMPIPE`.
  - **Decision (resolves §8 risk 5):** Mesa uses the existing
    `$(TOOLDIR)/llvm-vortex` install as its host LLVM rather than a
    separate distro LLVM. `llvm-vortex` is a full **LLVM 20.1.8**
    build with both X86 (host, for llvmpipe) and RISCV (Vortex
    device) targets and shared libraries — so the host and device
    halves of the toolchain share one LLVM.
  - Two build-environment findings folded into the installer:
    Mesa must be configured `-D cpp_rtti=false` to match the
    RTTI-less `llvm-vortex` (auto-detected from `llvm-config
    --cxxflags`); and `meson >= 1.4` is required, newer than the
    Ubuntu 22.04 distro `meson` — pulled from `pip`.
  - See §4.4 for the install procedure.
- **2026-05-21 (scope)** — Conformance model + RT direction +
  fixed-point invariant agreed.
  - **Conformance** is *inherit-and-accelerate* (new §4.5 +
    acceleration roadmap §4.6): `vortexpipe` inherits lavapipe's
    Vulkan surface (advertised 1.4) and commits/tests at **Vulkan
    1.3 + the RT extension family**. Verified the installed
    lavapipe already exposes the full RT extension set in
    software. Old §8 risk 4 ("Vulkan 1.0 + small KHR set")
    superseded.
  - **Ray tracing** is the end goal but is **not** a HW unit. The
    earlier draft's §5.5 "RT traversal unit" + `vx_trace` intrinsic
    + Phase 8 are **removed**. RT (BVH build, traversal,
    intersection, shading) runs on the **GPU SIMT cores**
    (rewritten Phase 7).
  - New **§4.7 Design invariants**: (1) the graphics HW is exactly
    RASTER/TEX/OM — everything else is SIMT; (2) R/T/O datapaths
    **stay fixed-point** (gfx-v1); FP graphics runs on SIMT;
    native FP in the units is deferred to a future
    **vortex-gfx-v2**. §5 improvements all preserve the
    fixed-point datapath.
- **2026-05-21 (SimX-only scope)** — RTL taken out of scope.
  - New **§4.7 Invariant 3**: the whole proposal targets the
    **SimX** simulator — driver bring-up, the R/T/O improvements
    (§5), RT, and all tests. The graphics units are extended as
    **SimX C++ models**; no RTL is written or modified. RTL
    realization of the gfx-v1 design is a separate later proposal,
    opened once the SimX backend is fully implemented.
  - Phases 4–6 retitled "<unit> integration (SimX model)" and now
    **absorb the gfx_migration Phase-3 SimX-model work** (§8 risk 3
    rewritten) — each phase implements one SimX unit model, then
    integrates it. §7 test plan drops the RTLsim column; §5
    `hw/unittest` verilator tests become SimX unit tests.
  - Mesa fork remote fixed: `github.com/vortexgpgpu/mesa`
    (local clone `~/dev/mesa_vortex`).
- **2026-05-22** — Phase 0 + Phase 1-skeleton landed; Phase 2
  pipeline switched to Shape B.
  - **Phase 0 done:** Mesa fork pushed to
    `github.com/vortexgpgpu/mesa` (branch `vortex_3.x`).
  - **Phase 1 skeleton committed** (`vortexpipe` Gallium driver,
    `vortex_3.x` @ `daf278736e9`): builds, registers, and is
    selectable under lavapipe via `GALLIUM_DRIVER=vortexpipe`.
    Currently an llvmpipe passthrough.
  - **§3 rewritten — Shape B replaces Shape A.** A two-pronged
    survey showed Shape A (fork `lp_bld_nir_soa.c`, ~238 KB of
    host-vectorized codegen) is a multi-month effort that
    re-derives `llvm_vortex`. Shape B reuses the existing Vortex
    SPIR-V→`.vxbin` toolchain (`pocl_vortex` /
    SPIRV-LLVM-Translator) with Mesa's `nir_to_spirv` — glue, not
    a code generator. §3, Phase 2, §8 risk 1, §9 updated.
- **2026-05-22 (Shape C)** — Phase 2 increment #1 landed; Phase 2
  pipeline switched again, Shape B → Shape C, after a spike.
  - **Phase 2 #1 committed** (`vortex_3.x` @ `0e09493314c`,
    `daf278736e9`..): vortexpipe links the Vortex runtime
    (`libvortex.so`), opens a device, and intercepts the compute
    hooks (driver-interception layer).
  - **Shape B disproved by a spike.** Mesa's only `nir_to_spirv`
    is zink-private *and* emits Vulkan-flavored SPIR-V; the spike
    found `llvm-spirv -r` hard-rejects it
    (`InvalidBuiltinSetName: Expects OpenCL.std`) — the
    SPIRV-LLVM-Translator is OpenCL-only. No SPIR-V round-trip
    carries a Vulkan shader to Vortex.
  - **§3 rewritten — Shape C.** A scalar NIR→LLVM-IR translator in
    `vortexpipe` (`vp_nir_to_llvm`). Not Shape A's SoA fork:
    Vortex is SIMT, so each thread runs *scalar* code — the
    translator emits scalar IR, with `lp_bld_nir`'s visitor
    structure only as a reference. §3, Phase 2 (6 increments),
    §4.1, §8 risk 1, §9 updated.

# Vulkan Support — Proposal

## 1. Summary

Add a Vulkan driver for Vortex by forking Mesa and standing up a
Vortex-aware Gallium driver (`vortexpipe`) underneath `lavapipe`.
Shaders go through Mesa's SPIR-V → NIR front-end, then through an
LLVM-IR codegen path that hands the device half to `llvm_vortex` (our
RISC-V + Vortex extension backend) to produce `.vxbin` payloads.
Fixed-function fragment work — rasterization, texture sampling, and
the output merger — is offloaded from `llvmpipe`'s CPU fallback to
the three Vortex hardware units (TEX, RASTER, OM) already landed in
`feature_gfx` via the
[gfx_migration_proposal.md](gfx_migration_proposal.md), keeping the
hybrid s/w-h/w model that draw3d already exercises:

- vertex shading, attribute interpolation, fragment program, and
  primitive setup run as Vortex compute kernels (LLVM-JIT'd from
  SPIR-V);
- per-tile rasterization, per-fragment texture sampling, and
  per-fragment depth/stencil/blend/write run on the TEX / RASTER / OM
  hardware units, invoked from the fragment kernel through the
  existing `vx_tex` / `vx_rast` / `vx_om` intrinsics.

A second, equally important goal is to **improve those three units**
for Vulkan-class workloads while keeping the hybrid model intact —
i.e. do not collapse into a fully-fixed-function pipeline. The
target improvements (§5) are quad-rate sampling and writes, MRT
support in OM, hierarchical-Z and early-Z in RASTER, compressed
texture formats and anisotropic filtering in TEX, and a deeper
raster/OM queue to keep the SFU PE switch fed.

The full Mesa fork and `vortexpipe` driver live in a **fork of
Mesa** — `github.com/vortexgpgpu/mesa` (local clone
`~/dev/mesa_vortex`) — mirroring the hip_vortex layout
([hip_support_proposal.md §4.2](hip_support_proposal.md)). The
Vortex tree (`feature_vulkan`) carries only: this proposal, a
`tests/vulkan/` regression suite, the CI hook, and the SimX
graphics-model changes for the §5 improvements.

The compilation pipeline adopted is **Shape C** (§3): a scalar
NIR→LLVM-IR translator in `vortexpipe`. lavapipe SPIR-V → NIR →
lavapipe lowerings → `vp_nir_to_llvm` (scalar LLVM IR via the LLVM
C API) → `llvm_vortex`'s RISC-V backend → `.vxbin` → loaded into
the Vortex device at pipeline-creation time and cached via
`VkPipelineCache`. (Shape A — forking llvmpipe's SoA codegen — and
Shape B — a SPIR-V round-trip — were both ruled out; see §3.)

The **north-star end goal is ray tracing**. The conformance model
(§4.5) is *inherit-and-accelerate*: `vortexpipe` inherits
lavapipe's full Vulkan surface — including the `VK_KHR_ray_query` /
`VK_KHR_acceleration_structure` / `VK_KHR_ray_tracing_pipeline`
extension family, which lavapipe already implements in software —
and the Vortex backend accelerates a growing subset of it, with
anything not-yet-ported falling back to lavapipe's CPU code. Two
design invariants (§4.7) bound the work: the graphics hardware is
exactly **RASTER, TEX, OM** — no fourth unit; ray tracing and
everything else not-R/T/O runs on the **GPU SIMT cores** as
ordinary kernels. And R/T/O **stay fixed-point** — floating-point
graphics math runs on the SIMT cores; native FP inside the units
is deferred to a future **vortex-gfx-v2**.

**Scope: SimX only.** Driver bring-up, the graphics-unit
improvements (§5), and all testing target the **SimX** simulator —
which keeps the build/iterate/optimize loop fast. The graphics
units are extended and optimized *as SimX models*; **no RTL work**
is in scope here. RTL implementation of the gfx-v1 units is a
separate future proposal, opened once the SimX backend is fully
fleshed out and implemented (see §4.7).

---

## 2. Background

### 2.1 Where the three HW units stand today

Per [gfx_migration_proposal.md](gfx_migration_proposal.md), the
skybox-era TEX / RASTER / OM units are migrated, elaborate clean,
and pass per-unit smoke tests. The hybrid model is preserved:

| Unit | RTL | SimX | Kernel intrinsic | What HW does | What SW does |
|---|---|---|---|---|---|
| RASTER | [hw/rtl/raster/](../../hw/rtl/raster/) | not yet (deferred per gfx Phase 3) | [vx_rast()](../../sw/kernel/include/vx_intrinsics.h#L278) | tile→block→quad engine; emits pos_mask + per-thread bcoords into CSRs | triangle setup, vertex transform, attribute interpolation off bcoords, fragment program |
| TEX | [hw/rtl/tex/](../../hw/rtl/tex/) | not yet | [vx_tex(stage,u,v,lod)](../../sw/kernel/include/vx_intrinsics.h#L262) | address gen, wrap/clamp, format decode, bilinear lerp, LOD selection | descriptor binding, mip computation, derivatives, gather emulation |
| OM | [hw/rtl/om/](../../hw/rtl/om/) | not yet | [vx_om(x,y,face,color,depth)](../../sw/kernel/include/vx_intrinsics.h#L271) | depth compare, stencil op, blend func, logic op, color/depth write | discard, MSAA resolve (currently absent), MRT (currently absent) |

The reference end-to-end test
[tests/regression/draw3d/](../../tests/regression/draw3d/) drives all
three from a single Vortex kernel and already exists in the form
this proposal builds on. The `kernel.cpp` pattern there
([draw3d/kernel.cpp](../../tests/regression/draw3d/kernel.cpp)) is
the same shape the Vulkan fragment kernel will take.

Note the **SimX** column: the TEX/OM/RASTER SimX models do not
exist yet ([gfx_migration_proposal.md](gfx_migration_proposal.md)
scoped them as its Phase 3 and deferred it). Because this proposal
is SimX-only, **bringing those SimX models up is in scope here** —
Phases 4–6 implement them and §5 extends them. The existing RTL
([hw/rtl/{raster,tex,om}/](../../hw/rtl/)) is the behavioral
reference the SimX models mirror; it is **not modified** by this
proposal.

### 2.2 What Mesa gives us for free

Three sub-stacks of Mesa apply here, in decreasing distance from
what we need to write:

| Mesa component | Path in upstream | Why it matters |
|---|---|---|
| `vulkan/runtime` (`vk_*`) | `src/vulkan/runtime/` | Vulkan entry-point dispatch, `vk_device`/`vk_command_buffer`/`vk_pipeline` base classes, descriptor management, WSI plumbing. Used by every modern in-tree Vulkan driver (lavapipe, turnip, radv, v3dv). We inherit, don't reimplement. |
| `lavapipe` (lvp_*) | `src/gallium/frontends/lavapipe/` | Vulkan-on-Gallium frontend that already converts Vulkan command buffers into Gallium calls + SPIR-V into NIR. Sits on top of `llvmpipe`. |
| `llvmpipe` | `src/gallium/drivers/llvmpipe/` | CPU Gallium driver: NIR→LLVM IR JIT, rasterizer (`lp_setup_*`), sampler (`lp_bld_sample`), depth/stencil/blend (`lp_bld_depth`, `lp_bld_blend`). This is what we replace, unit by unit, with the Vortex HW path. |

`lavapipe + llvmpipe` together is what's commonly referred to as
"lavapipe": a complete CPU Vulkan stack. The shortest path to a
Vortex Vulkan driver is therefore *not* a green-field Vulkan driver
— it's a Gallium driver `vortexpipe` that selectively replaces
`llvmpipe`'s fixed-function blocks with Vortex HW calls, with
`lavapipe` mounted unchanged on top.

### 2.3 Why a Gallium driver rather than a native Vulkan driver

Modern Mesa Vulkan drivers (turnip, radv, anv, nvk, v3dv) do not use
Gallium — they sit directly on `vulkan/runtime`. That's the
direction Mesa is heading. But for Vortex specifically:

- llvmpipe already implements the full fixed-function fragment
  pipeline (raster, sampler, OM) in well-factored modules we can
  selectively replace; rewriting these from scratch in a native
  Vulkan driver is months of work that buys nothing relative to
  what llvmpipe already gives us.
- The hybrid s/w-h/w model in §2.1 maps cleanly onto Gallium's
  draw-module / setup / shader split: shaders go through the
  Shape-C scalar NIR→LLVM translator (§3) to produce Vortex device
  binaries; `lp_setup_*` → `VX_raster_*`; `lp_bld_sample` → `vx_tex`;
  `lp_bld_depth/blend` → `vx_om`.
- Lavapipe's Vulkan conformance work (validation layer compliance,
  feature flag plumbing, descriptor set semantics) is inherited
  intact.

**Recommendation: Gallium-based `vortexpipe` as the primary path.**
A native Vulkan driver is a possible v2; see §8 risk #6.

---

## 3. Compilation pipeline (Shape C — scalar NIR→LLVM)

```
foo.comp.spv                              (Vulkan client — VkShaderModule)
   │
   ▼  vkCreateComputePipelines       (lavapipe)
   ├─ spirv_to_nir                          →  foo.nir
   ├─ lavapipe lowering passes
   │    (descriptor-set → flat binding, robustness,
   │     subgroups, shared / explicit I/O …)
   │
   ▼  vortexpipe create_compute_state(nir)  (vortexpipe)
   ├─ vp_nir_to_llvm   (vortexpipe — scalar NIR→LLVM, LLVM C API) →  foo.bc
   │    + the SIMT kernel wrapper (gl_GlobalInvocationID etc.
   │      from the KMU CSRs; the vx_spawn2 launch idiom)
   ├─ llvm_vortex RISC-V backend  (--target=riscv$(XLEN)-unknown-elf
   │                               -mattr=+xvortex,+zicond)       →  foo.elf
   ├─ vxbin.py        (sw/kernel/scripts/)                        →  foo.vxbin
   └─ .vxbin blob stored in the pipe_compute_state object
                                            (cached via VkPipelineCache)

vkCmdDispatch → vortexpipe launch_grid(pipe_grid_info):
   ├─ vx_buffer_create + upload the .vxbin kernel + the argument block
   ├─ vx_enqueue_launch(queue, vx_launch_info_t{ grid_dim, block_dim, … })
   └─ vx_queue_finish
```

**Why Shape C — and why not B or A.** Two earlier shapes were
ruled out:

- **Shape B (SPIR-V round-trip) — disproved by a spike (2026-05-22).**
  Shape B routed NIR → `nir_to_spirv` → SPIRV-LLVM-Translator →
  LLVM. The spike found Mesa's only `nir_to_spirv` is zink-private
  *and* emits Vulkan-flavored SPIR-V (`Shader` / `Logical` /
  `GLSL.std.450`), which SPIRV-LLVM-Translator **hard-rejects**:
  `llvm-spirv -r` → `InvalidBuiltinSetName: Expects OpenCL.std`.
  The translator is OpenCL-only; no SPIR-V round-trip carries a
  Vulkan shader to Vortex.
- **Shape A (fork `lp_bld_nir_soa.c`) — overscoped.** That file is
  ~238 KB because of llvmpipe's **SoA host-CPU vectorization** — it
  vectorizes across invocations with explicit LLVM vector types +
  AVX. Vortex does not need that: Vortex's **SIMT hardware is the
  parallelism**; each Vortex thread runs ordinary *scalar* code.

**Shape C** is the consequence: a **scalar NIR→LLVM-IR translator**
in `vortexpipe` (`vp_nir_to_llvm`). It walks the lavapipe-lowered
NIR and emits straight scalar LLVM IR via the LLVM C API — NIR ALU
op → LLVM instruction, load/store → load/store, control flow →
basic blocks. `lp_bld_nir`'s visitor *structure* is a reference;
the SoA leaf emission is **not** carried over. The LLVM IR then
goes to `llvm_vortex`'s RISC-V backend and `vxbin.py`, exactly as
HIP/OpenCL device code does.

Notes:

1. **It is a real compiler component, but a tractable one.** Shape
   C is not "glue" — it is a NIR→LLVM translator, the genuine core
   of the driver. But emitting *scalar* IR (no SoA vectorizer) is a
   normal-sized component, not the multi-month SoA fork: the
   reusable part is the visitor walk; the replaced part is simple
   per-op scalar emission.

2. **The SIMT kernel wrapper.** A Vulkan compute `main()` is one
   workgroup-invocation. The translator wraps it as a Vortex KMU
   kernel: the entry reads grid/block from the KMU CSRs and binds
   `gl_GlobalInvocationID` / `gl_LocalInvocationID` / `gl_WorkGroupID`
   per thread — the `vx_spawn2` idiom (see
   [sw/kernel/include/vx_spawn2.h](../../sw/kernel/include/vx_spawn2.h)).
   `pocl_vortex` does the equivalent for OpenCL kernels; mirror it.

3. **LLVM C API availability.** vortexpipe already builds inside
   Mesa, which links LLVM (llvmpipe). `vp_nir_to_llvm` emits IR via
   `llvm-c/*` against that LLVM (the same `llvm_vortex` install,
   §8 risk 5) — no extra LLVM dependency.

4. **Graphics shaders (Phase 3+).** The same scalar translator
   serves vertex/fragment shaders; the graphics-specific part is
   lowering the TEX/RASTER/OM ops, deferred to Phase 3. Phase 2
   (compute) commits to Shape C.

---

## 4. Target architecture

### 4.1 Component layout

Mirrors the hip_vortex split: full Mesa fork lives in a sibling
repo; the Vortex tree carries tests + design docs + the SimX
graphics-model changes for the §5 improvements (no RTL — §1 scope).

| Component | Path | Source / size |
|---|---|---|
| Mesa fork | remote `github.com/vortexgpgpu/mesa`; local clone `~/dev/mesa_vortex/` pinned at `mesa-25.1`; `upstream` remote → freedesktop Mesa for rebases | Fork of upstream Mesa; Vortex changes confined to additions under `src/gallium/drivers/vortexpipe/`. |
| `vortexpipe` Gallium driver | `~/dev/mesa_vortex/src/gallium/drivers/vortexpipe/` | New (~5–8 kLOC, mostly forked from `llvmpipe/`). |
| `vortexpipe` shader path (Shape C) | `~/dev/mesa_vortex/src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c` (+ helpers) | New: a scalar NIR→LLVM-IR translator (LLVM C API) — the genuine compiler core — plus the SIMT kernel wrapper, then `llvm_vortex` + `vxbin.py`. |
| Gallium loader entry | `~/dev/mesa_vortex/src/gallium/targets/dri/target.c` + `meson_options.txt` | Register `vortexpipe` so `lavapipe` can pick it via `LAVAPIPE_PIPE_DRIVER=vortexpipe`. |
| `lavapipe` patches | `~/dev/mesa_vortex/src/gallium/frontends/lavapipe/` | Minimal — most diffs are feature-flag plumbing for what `vortexpipe` does or doesn't support. Try to keep zero patches if possible. |
| Vortex runtime host glue | `~/dev/mesa_vortex/src/gallium/drivers/vortexpipe/vx_context.c` | Owns the `vortex_runtime` device handle, kernel cache, descriptor heaps; bridges Gallium calls to `vortex2.h` async API. |
| Mesa build infra | `~/dev/mesa_vortex/meson.build` adjustments | Add `vortexpipe` as a gallium-drivers option; require `llvm_vortex` install via `dependency('LLVMVortex', method: 'cmake')`. |
| Mesa install script | [ci/mesa_install.sh.in](../../ci/mesa_install.sh.in) | **Landed (Phase 0).** Installs build deps, builds Mesa from `MESA_REPO`@`MESA_REV` with `meson`, installs into `$(TOOLDIR)/mesa-vortex/{lib,share,include}`. Uses `$(TOOLDIR)/llvm-vortex` as the host LLVM. Default rev is upstream `mesa-25.1.0` (the Phase 0 baseline); repoint `MESA_REPO` at `github.com/vortexgpgpu/mesa` once the fork is populated. ICD is `lvp_icd.x86_64.json` until `vortexpipe` lands. |
| Vulkan tests | [tests/vulkan/](../../tests/vulkan/) (new) | New: smoke tests + a Vulkan port of [draw3d](../../tests/regression/draw3d/) as the integration end-to-end. |
| Test aggregator | [tests/vulkan/Makefile](../../tests/vulkan/) + `common.mk` (new) | New, defaults `VULKAN_INSTALL_PATH := $(TOOLDIR)/mesa-vortex`, modelled on [tests/hip/common.mk](../../tests/hip/common.mk). |
| Test suite registration | [tests/Makefile](../../tests/Makefile) | Add opt-in `vulkan` target gated on `$(TOOLDIR)/mesa-vortex` existing. |
| CI hook | [ci/regression.sh.in](../../ci/regression.sh.in) | Add `--vulkan` suite gated on the toolchain installation. |
| SimX graphics models (R/T/O) | `sim/simx/{raster,tex,om}/` (new) | Implemented by Phases 4–6 ([gfx_migration_proposal.md](gfx_migration_proposal.md) Phase 3 scope, absorbed here) and extended by the §5 improvements. Mirror the RTL module structure 1:1. |
| RTL (`hw/rtl/{raster,tex,om}/`) | [hw/rtl/](../../hw/rtl/) | **Not modified.** Behavioral reference the SimX models mirror; RTL realization of the gfx-v1 improvements is a separate future proposal (§1 scope). |

### 4.2 Where Vulkan API surface goes vs Vortex runtime API

Per [feedback_runtime_minimalism.md](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_runtime_minimalism.md)
([[feedback_runtime_minimalism]]), `vortex2.h` must stay minimal —
Vulkan-shaped state (renderpasses, descriptor sets, pipeline state
objects, framebuffer caches, command-buffer recording) lives **in
`vortexpipe`**, not in `vortex2.h`. The driver-to-device boundary
remains the same five-or-so calls
(`vx_mem_alloc`/`vx_copy_*`/`vx_upload_kernel`/`vx_start`/
`vx_ready_wait`) that HIP, OpenCL, and the existing OpenCL tests
already use. New device-side capability (e.g. a deeper raster queue
in §5.1) surfaces as a new CSR or DCR, not a new runtime entry
point.

### 4.3 Where Vulkan command buffers land on Vortex

A Vulkan command buffer recorded by the client becomes, at submit
time, a sequence of:

1. **State binds (descriptor sets, push constants, dynamic
   state):** lowered into DCR writes for TEX/RASTER/OM
   configuration (texture descriptors, raster viewport/scissors,
   blend state) and kernel-arg buffer uploads.
2. **`vkCmdDraw*`:** a kernel dispatch (`vx_start`) of the
   vertex/fragment kernel produced at pipeline-create time, with
   the raster front-end CSRs primed to drive `vx_rast()` polls
   from the fragment kernel.
3. **`vkCmdDispatch`:** straight kernel dispatch — same path the
   HIP and OpenCL backends use today; the fragment-pipeline DCRs
   are not touched.
4. **`vkCmdCopyBuffer*` / `vkCmdBlitImage`:** lowered to
   `vx_copy_*` or to a built-in copy kernel (we already have one
   from the DXA path).
5. **`vkCmdPipelineBarrier`:** maps to `vx_ready_wait` for
   coarse-grained host-visible barriers; for fine-grained
   intra-queue barriers we use the async-barrier extension
   tracked in [hip_support_proposal.md §4 Phase 4](hip_support_proposal.md)
   when available — until then, coarse-grained only.

The Vulkan queue model maps to one Vortex device context per
`VkQueue`; multi-queue is supported by multiple contexts on the
same `VkDevice` (the underlying Vortex runtime already serializes
device access under SimX, so multi-queue is correctness-only
initially, not performance).

### 4.4 Building & installing the Mesa toolchain

The driver toolchain is built and installed by
[ci/mesa_install.sh.in](../../ci/mesa_install.sh.in) — a `.in`
template that `configure` substitutes (`@TOOLDIR@`) and copies to
`<build>/ci/mesa_install.sh`, in the same style as `sst_install.sh`
and `gem5_install.sh`. It does four things:

1. **Build deps** — apt packages (`python3-mako`, `flex`/`bison`,
   `libdrm`/`libx*`/`wayland` WSI libs, `libzstd-dev`) plus a
   current `meson` from `pip` (the Ubuntu 22.04 distro `meson` 0.61
   is older than Mesa 25.x's `>= 1.4` requirement).
2. **Host LLVM** — *no separate LLVM is installed.* Mesa's llvmpipe
   is pointed at `$(TOOLDIR)/llvm-vortex` (LLVM 20.1.8, built with
   X86 + RISCV, shared libs). `mesa_install.sh` errors out early if
   `llvm-vortex` is absent — `ci/toolchain_install.sh` must run
   first.
3. **Vulkan runtime stack** — loader, validation layers,
   `vulkaninfo`/`vkcube`, `glslang`/`spirv-tools`.
4. **Mesa** — clone at `MESA_REV`, `meson setup` + `compile` +
   `install` into `$(TOOLDIR)/mesa-vortex`.

Two non-obvious points, both handled by the script:

- **RTTI.** Mesa's C++ RTTI setting must match LLVM's. `llvm-vortex`
  is built without RTTI, so Mesa is configured `-D cpp_rtti=false`
  (auto-detected by grepping `llvm-config --cxxflags` for
  `-fno-rtti`).
- **`gallium-drivers` naming.** The Gallium value for llvmpipe is
  `llvmpipe`; the Vulkan value for lavapipe is `swrast`. Phase 1
  appends `vortexpipe` to `GALLIUM_DRIVERS`.

The Vulkan loader finds the driver via
`VK_ICD_FILENAMES=$(TOOLDIR)/mesa-vortex/share/vulkan/icd.d/lvp_icd.x86_64.json`;
`LD_LIBRARY_PATH` must include both `mesa-vortex/lib` and
`llvm-vortex/lib` (the ICD links `libLLVM` shared). The script
exports these and persists them to `~/.bashrc` / `$GITHUB_ENV`.

### 4.5 Conformance model — inherit-and-accelerate

The conformance target is **not** a version to *reach*; it is a
surface to *inherit*. The Mesa software rasterizer (lavapipe) in
the Phase 0 build already exposes a complete, capable Vulkan
implementation — verified on the installed driver:

```
apiVersion         = 1.4.311          (advertised surface)
conformanceVersion = 1.3.1.1          (CTS-passed level)
deviceName         = llvmpipe (LLVM 20.1.8, 256 bits)
+ VK_KHR_acceleration_structure, VK_KHR_ray_query,
  VK_KHR_ray_tracing_pipeline, VK_KHR_ray_tracing_maintenance1,
  VK_KHR_ray_tracing_position_fetch, VK_KHR_deferred_host_operations,
  VK_KHR_pipeline_library   (RT family — software-implemented)
```

`vortexpipe` therefore **inherits lavapipe's Vulkan surface** and
accelerates a growing subset of it on Vortex; any path the Vortex
backend has not yet ported transparently falls back to lavapipe's
CPU implementation. This is the native Gallium contract (drivers
advertise caps; unsupported paths fall back to Mesa's `draw` /
`util` software helpers) — not a mechanism we invent.

**Conformance statement for the proposal:**

- **Advertised `apiVersion`: Vulkan 1.4** — inherited from
  lavapipe, not capped (capping is pointless extra work).
- **Conformance commitment & CTS target: Vulkan 1.3 + the RT
  extension family** — 1.3 is lavapipe's CTS-passed level and the
  stable `dEQP-VK` suite the §7 test plan runs against. (1.4 is
  only advertised, not conformance-submitted; its extra mandatory
  features are raster/host-side conveniences that inherit-and-
  accelerate supplies for free and that do not affect the Vortex
  HW story.)
- **Not in scope:** an official Khronos conformance submission —
  see §8 risk 4.

Three properties this model buys:

1. **No half-broken driver state.** At every phase, `vortexpipe`
   is a complete, conformant Vulkan driver — whatever Vortex has
   not accelerated yet, lavapipe covers. Every commit is a
   shippable, testable artifact.
2. **A built-in correctness oracle.** Each Vortex-accelerated path
   has the *same lavapipe code* as a CPU reference to diff
   pixel-for-pixel against (§7).
3. **RT for free as a reference.** lavapipe's software BVH build,
   traversal, and RT shaders are the golden reference the Vortex
   SIMT RT path (Phase 7) is validated against.

**Honest caveat:** inheriting the *full* surface means we
advertise extensions Vortex may never accelerate — they run
*correctly* but at CPU speed. The conformance goal ("match the
surface") is therefore paired with the explicit acceleration
roadmap below, which states, per capability, what runs where.
Driver docs must not let "supported" imply "fast."

### 4.6 Acceleration roadmap

Where each capability executes, and the phase that moves it onto
Vortex. "CPU-fallback" = lavapipe's existing software path; it
remains the path until the listed phase lands, and the fallback
for any config in which the Vortex unit is disabled.

| Capability | Target execution | Lands in | Until then |
|---|---|---|---|
| Vertex / fragment / compute shaders | Vortex compute (LLVM-JIT'd kernels) | Phase 2–3 | CPU (lavapipe) |
| Rasterization | Vortex RASTER unit, SimX model (`vx_rast`) | Phase 4 | CPU-fallback |
| Output merger (depth/stencil/blend) | Vortex OM unit, SimX model (`vx_om`) | Phase 5 | CPU-fallback |
| Texture sampling | Vortex TEX unit, SimX model (`vx_tex`) | Phase 6 | CPU-fallback |
| RT acceleration-structure build | Vortex SIMT compute (kernel) | Phase 7 | CPU-fallback |
| RT BVH traversal + intersection | Vortex SIMT compute (kernel) | Phase 7 | CPU-fallback |
| RT raygen / closest-hit / miss shading | Vortex SIMT compute (kernels) | Phase 7 | CPU-fallback |
| §5 unit improvements (MRT, MSAA, Hi-Z, …) | Vortex R/T/O SimX models | post-Phase 6 | base unit behavior |
| Everything else (sparse, query pools, host-image-copy, …) | CPU-fallback (lavapipe) | — not planned | permanent CPU-fallback |

### 4.7 Design invariants

Two invariants constrain everything below. They are not
negotiable within the scope of this proposal — violating either
changes what "the Vortex graphics architecture" means.

**Invariant 1 — three fixed-function units, everything else SIMT.**
The graphics hardware is exactly **RASTER, TEX, OM** — no fourth
unit. Every other part of the Vulkan 1.3 pipeline — vertex
processing, attribute interpolation, primitive assembly, geometry
and tessellation, *ray tracing* (BVH build, traversal,
intersection, and raygen/hit/miss shading), compute — runs on the
**GPU SIMT cores** as ordinary Vortex kernels. There is no RT HW
unit and no `vx_trace` intrinsic; ray tracing is SIMT compute
(Phase 7). This is the inherit-and-accelerate model (§4.5) drawing
its line: R/T/O are the accelerated fixed-function islands, the
SIMT cores are everything else, and lavapipe's CPU code is the
fallback for anything not yet ported to either.

**Invariant 2 — R/T/O datapaths stay fixed-point (gfx-v1).** The
RASTER, TEX, and OM units are fixed-point today (Q-format edge
equations, barycentrics, texel lerp, blend — see the
`fixed16_t`/`fixed24_t` paths in
[draw3d/kernel.cpp](../../tests/regression/draw3d/kernel.cpp)).
They **remain fixed-point** through this proposal. Every §5
improvement extends a fixed-point datapath; none introduces a
floating-point adder/multiplier into R/T/O. Floating-point
graphics math — wherever a Vulkan shader needs it — is done on the
**SIMT cores**, which already have an FPU; values cross the
HW/SIMT boundary as fixed-point CSRs/operands (again, the
`BCOORD_AS_FLOAT` conversion in draw3d is the existing pattern).

Native floating-point *inside* the R/T/O units is a separate,
future effort — **vortex-gfx-v2** — explicitly out of scope here.
This proposal is gfx-v1: improve the fixed-point units, push all
FP work to SIMT.

**Invariant 3 — SimX only; RTL is a separate later proposal.** All
of this proposal — driver bring-up, the R/T/O improvements (§5),
ray tracing, and every test — targets the **SimX** simulator. The
graphics units are extended and validated *as SimX C++ models*;
**no RTL is written or modified.** This keeps the
build/iterate/optimize loop fast (SimX is far faster than RTLsim).
Realizing the gfx-v1 SimX design in RTL is a **separate proposal**,
opened once the SimX backend is fully fleshed out and implemented.
So the roadmap layers: *this* proposal = gfx-v1 on SimX → a future
proposal = gfx-v1 in RTL → vortex-gfx-v2 = floating-point units.

---

## 5. Hardware unit improvements

This section is **not optional polish** — these are the deltas
that get the Vortex graphics path from "skybox-class fixed-pipeline
that runs draw3d" to "Vulkan 1.3-class fragment pipeline." The
existing units pass their smoke tests but were never built for the
following Vulkan-required behaviors. Each improvement is one
self-contained, testable feature
([[feedback_no_prs_direct_commits]]) and lands as its own commit
on `feature_vulkan`.

Per §4.7 Invariant 2, **every improvement below preserves the
fixed-point datapath** — it widens, deepens, or extends
fixed-point logic; it never adds a floating-point unit to R/T/O.
FP-in-the-units is deferred to vortex-gfx-v2.

**All of §5 is realized in the SimX graphics models**
(`sim/simx/{raster,tex,om}/`) — §1 scope; no RTL. The SimX models
mirror the RTL module structure 1:1 (per
[gfx_migration_proposal.md](gfx_migration_proposal.md) §3.2), so
the `VX_*.sv` block name in each *Sketch* cell below identifies the
**architectural block** being changed — the change is implemented
and validated in the corresponding SimX C++ block, and the RTL
file is named only as the structural reference. RTL realization of
these improvements is the separate future proposal.

### 5.1 RASTER

Current raster ([hw/rtl/raster/VX_raster_unit.sv](../../hw/rtl/raster/VX_raster_unit.sv)
+ te/be/qe/edge/extents) emits one quad per warp at a time; threads
poll `vx_rast()` and consume per-thread bcoords from CSRs. Issues
for Vulkan:

| # | Improvement | Why Vulkan needs it | Sketch |
|---|---|---|---|
| 5.1.1 | **Hierarchical-Z + early-Z** | Vulkan early-fragment-test and depth-prepass workloads regress badly without it — every fragment runs the kernel even when occluded. | Add a small Hi-Z buffer in `VX_raster_be.sv` keyed by 8×8 tile; reject occluded tiles before quad emit. Depth read path joins the OM cache (ocache). |
| 5.1.2 | **Deeper quad queue per warp** | The current per-warp single-quad latch stalls the SFU PE switch whenever the fragment kernel takes more than a couple of cycles per quad. | Replace the per-warp single-entry latch in `VX_raster_qe.sv` with a small FIFO (configurable, default 4). `vx_rast()` semantics unchanged. |
| 5.1.3 | **Per-primitive setup HW** | Vulkan requires consistent triangle setup semantics (top-left fill rule, conservative rasterization optional). Today setup is SW; we incur a per-triangle kernel call. | Add a setup unit in front of `VX_raster_te.sv` that consumes a packed-vertex triple and produces edge equations; SW just submits vertex indices. Keep SW-setup path as fallback under a DCR bit. |
| 5.1.4 | **Multisample support (MSAA 2×/4×)** | Vulkan core feature; lavapipe supports 4× MSAA. | Extend `VX_raster_extents.sv` to evaluate edge equations at sample positions; emit per-sample coverage mask alongside `pos_mask`. OM resolve in 5.3.4. |
| 5.1.5 | **Provoking-vertex + primitive-restart semantics** | Strict Vulkan conformance. | Front-of-pipe DCR bits; small front-end queue change. |

### 5.2 TEX

Current TEX
([hw/rtl/tex/VX_tex_unit.sv](../../hw/rtl/tex/VX_tex_unit.sv)) does
single-texel bilinear/trilinear with byte-aligned formats. Issues:

| # | Improvement | Why Vulkan needs it | Sketch |
|---|---|---|---|
| 5.2.1 | **Quad-rate sampling (`vx_tex4`)** | Fragment derivatives (`dFdx`/`dFdy`) for mipmap LOD computation are defined across a 2×2 quad. Today the kernel computes them by issuing 4 serialized `vx_tex` calls and looking at deltas — costs 4× the bandwidth and serializes the warp. | Add a `vx_tex4(stage, u[4], v[4]) -> texel[4]` intrinsic; reuse tex_sampler but issue 4 addresses in lockstep; LOD computed in HW from the quad. New encoding under CUSTOM1 funct3=1, funct2=quad. |
| 5.2.2 | **Compressed formats (BC1–BC7 / ETC2 / ASTC subset)** | Vulkan core / widely-required ext. Apps that ship to mobile or desktop will hit this immediately. | Extend `VX_tex_format.sv` with per-format decoders. BC1/BC3 first (smallest, most common), then BC7, then ETC2. ASTC is a stretch — defer to its own commit. |
| 5.2.3 | **Anisotropic filtering** | Vulkan `samplerAnisotropy` feature; lavapipe supports up to 16×. | New filter stage after tex_lerp; iteratively samples along anisotropy direction in HW. Footprint config from a DCR. |
| 5.2.4 | **3D and cube textures + array layers** | Vulkan core. | Extend `VX_tex_addr.sv` for 3 dimensions + layer; sampler descriptor gains a `dim`/`array_size` field via DCR. |
| 5.2.5 | **Texture gather (`textureGather`)** | Vulkan core (`shaderImageGatherExtended`). | Falls out almost free from 5.2.1 — `vx_tex4` returns the 2×2 footprint of the *single* sample, which is the gather semantic. Add `vx_tex_gather` as an alias with different swizzle. |
| 5.2.6 | **Bindless / large descriptor heap** | Vulkan descriptor sets are large; the current per-stage CSR-driven binding model caps out at low single-digit textures per draw. | Move sampler descriptors from CSR-resident to memory-resident with a base-pointer DCR + a small in-unit cache. Sampler index becomes part of the `vx_tex` operand. |

### 5.3 OM

Current OM
([hw/rtl/om/VX_om_unit.sv](../../hw/rtl/om/VX_om_unit.sv)) does
single-pixel depth/stencil/blend/write to one render target. Issues:

| # | Improvement | Why Vulkan needs it | Sketch |
|---|---|---|---|
| 5.3.1 | **Quad-rate writes (`vx_om4`)** | Same SIMD-bandwidth issue as 5.2.1: 2×2 fragment quads serialize over 4 `vx_om` calls. | `vx_om4(x, y, color[4], depth[4], mask)` — single OM-side cycle for the quad; pos derived from quad origin. Encoding under CUSTOM1 funct3=2, funct2=quad. |
| 5.3.2 | **Multiple render targets (MRT, up to 8)** | Vulkan core; deferred-shading apps require this. | Add per-RT descriptor table (DCR-pointed) in `VX_om_unit.sv`; `vx_om` gains an RT-index operand. Blend/write replicated per RT. |
| 5.3.3 | **Discard / per-fragment mask** | Vulkan `discard` is a fragment-shader builtin. Today expressed by SW-suppressed `vx_om`, but that still routes through the OM pipeline. | Add an explicit mask operand to `vx_om4` so OM filters internally; saves cache-port bandwidth on heavily-discarded scenes. |
| 5.3.4 | **MSAA resolve** | Pairs with 5.1.4. Vulkan `vkCmdResolveImage` and implicit subpass resolves. | New OM mode (DCR-selected) that reads per-sample color, averages, writes resolved. Reuses ocache; no new memory port. |
| 5.3.5 | **Deeper write queue** | Same SFU-PE-switch backpressure issue as 5.1.2 but on the write side. | FIFO in `VX_om_arb.sv` ahead of blend; depth configurable. |
| 5.3.6 | **Hi-Z update path** | Pairs with 5.1.1 — early-Z is meaningless if OM doesn't update Hi-Z on writes. | OM publishes per-tile min/max depth back to a shared Hi-Z buffer (same array as 5.1.1). |

### 5.4 Cross-unit: scheduler-aware backpressure

A subtle issue not localized to any one unit: when raster fills its
quad queue (5.1.2) faster than the fragment kernel consumes, or when
OM stalls on memory, the SFU PE switch (modeled in SimX after
[hw/rtl/core/VX_sfu_unit.sv](../../hw/rtl/core/VX_sfu_unit.sv))
must round-robin over the three PE indices without starving any one.
The existing `VX_pe_switch` is fair-arbitration; for graphics
workloads it may need a priority hint (raster low, OM high, TEX
medium) to avoid livelock on heavy-draw scenes. Characterize it in
the SimX model after 5.1.2 + 5.3.5 land; if a priority hint is
warranted, the SimX change is small and the RTL counterpart is
noted for the future RTL proposal.

---

## 6. Phases

Each phase is independently buildable and testable. Phases 0–3 are
sequential; phases 4–6 (Vortex graphics-model integration) are
sequential relative to each other but **independent of the 5.x
improvements** in §5 — i.e. we can chase Vulkan API coverage in
0–6 while §5 improvements land in parallel commits on a separate
cadence. Everything runs on **SimX** (§1 scope).

Phases 5.x in §5 each map to one substantial commit
([[feedback_no_prs_direct_commits]]); they're not numbered into the
main phase plan because their ordering depends on which Vulkan apps
we target first.

### Phase 0 — Mesa baseline ✅ baseline done

- [x] This proposal at
      `docs/proposals/vulkan_support_proposal.md`.
- [x] [ci/mesa_install.sh.in](../../ci/mesa_install.sh.in) —
      builds + installs Mesa into `$(TOOLDIR)/mesa-vortex`, using
      `$(TOOLDIR)/llvm-vortex` as the host LLVM (see §4.4).
- [x] Build vanilla `lavapipe` (no Vortex code) and confirm
      `vulkaninfo` reports the lavapipe device — the "I haven't
      broken Mesa" baseline. Done with upstream **Mesa 25.1.0**:
      `vulkaninfo` → `llvmpipe (LLVM 20.1.8)` /
      `DRIVER_ID_MESA_LLVMPIPE`.
- [x] Create the Mesa fork — **local** clone `~/dev/mesa_vortex`
      pinned at `mesa-25.1.0` (branch `vortex_3.x`), `origin` →
      `github.com/vortexgpgpu/mesa`, `upstream` → freedesktop Mesa.
- [x] Push the fork — branch `vortex_3.x` is on
      `github.com/vortexgpgpu/mesa` (full Mesa history, at
      `mesa-25.1.0`).
- [ ] Repoint `mesa_install.sh`'s `MESA_REPO` at the fork — once
      it carries the `vortexpipe` driver (Phase 1).
- [ ] Confirm `vkcube` runs on a display (needs a `DISPLAY`; the
      headless CI path stops at `vulkaninfo --summary`).

**Exit criteria:** vanilla lavapipe builds against `llvm-vortex`
and `vulkaninfo` enumerates the device. ✅ **met.** The Mesa fork
is pushed to `github.com/vortexgpgpu/mesa`; only the `MESA_REPO`
repoint (Phase 1) and the optional `vkcube`-on-display remain.
Nothing in the Vortex tree changed except this proposal,
`ci/mesa_install.sh.in`, and the §9 Mesa section of
`docs/building_toolchain.md`.

### Phase 1 — `vortexpipe` skeleton + device context

- [ ] Implement `vx_screen` / `vx_context` / `vx_resource` in
      `vortexpipe/`, modelled on `llvmpipe/lp_screen.c`. The
      context opens a `vortex_runtime` device handle, allocates
      device memory via `vx_mem_alloc`, and uploads/downloads via
      `vx_copy_*`.
- [ ] Implement the no-op shader / no-op pipeline path: a
      `vkCreateGraphicsPipelines` call returns a `VkPipeline`
      whose Vortex binary is an empty kernel; submitting a draw
      runs the empty kernel and writes nothing to the
      framebuffer.

**Exit criteria:** `vkCreateInstance` / `vkCreateDevice` /
`vkCreateGraphicsPipelines` / `vkCmdDraw` / `vkQueueSubmit` all
return `VK_SUCCESS` against a Vortex device; framebuffer is whatever
init color it was cleared to.

### Phase 2 — Compute pipelines (Shape C: scalar NIR→LLVM)

Shape C (§3). The genuine compiler core of the project. Built as
six committable increments — each independently buildable; the
whole of Phase 2 is committed once the exit test passes.

- [x] **#1 Driver interception.** `vortexpipe` patches the
      llvmpipe `context_create` + the compute hooks
      (`create/bind/delete_compute_state`, `launch_grid`); side
      registry holds per-screen/context state. Overrides forward
      to llvmpipe until the increments below fill them in.
- [ ] **#2 `vp_nir_to_llvm` skeleton.** A `nir_shader` → LLVM
      module walk via the LLVM C API: functions, basic blocks, the
      NIR-SSA → `LLVMValueRef` map, control flow. Trivial shaders
      first.
- [ ] **#3 NIR instruction emission.** Per-op scalar emission:
      `nir_alu` (the bulk), `nir_intrinsic` (load/store, the
      `gl_*InvocationID` system values), `nir_load_const`,
      derefs. Scalar — no SoA vectorizer.
- [ ] **#4 SIMT kernel wrapper + `.vxbin`.** Wrap the shader as a
      Vortex KMU kernel (grid/block from CSRs, the `vx_spawn2`
      idiom; the descriptor-set → arg-block ABI). Drive
      `llvm_vortex`'s RISC-V backend + `vxbin.py` →  `.vxbin`,
      stored in the compute-state object.
- [ ] **#5 `launch_grid`.** Upload the `.vxbin` + argument block
      via `vx_buffer_*`; `vx_enqueue_launch` with grid/block from
      `pipe_grid_info`; `vx_queue_finish`. `pipe_resource` ⇄
      `vx_buffer_*` memory bridging. The screen's Vortex device
      (opened in the Phase 2-foundation increment) backs this.
- [ ] **#6 Compute caps + test.** Fill `pipe_screen.compute_caps`
      from `vx_device_query`; wire the vecadd `vkCmdDispatch`
      test.

**Exit criteria:** a SPIR-V compute shader that does vecadd
compiles through `vortexpipe`, runs on SimX, and matches the CPU
reference — the Vulkan analog of
[tests/hip/vecadd](../../tests/hip/vecadd/). Anything `vp_nir_to_llvm`
cannot yet translate falls back to llvmpipe (§4.5), so the driver
stays whole at every increment.

### Phase 3 — Software graphics MVP (all-CPU fallback)

- [ ] Wire `vortexpipe` through `lp_setup_*` / `lp_bld_sample` /
      `lp_bld_depth_blend` — i.e. let llvmpipe's CPU code do all
      the fragment work, on the *host*, while the vertex stage
      runs on Vortex.
- [ ] Get a triangle on screen via the lavapipe → vortexpipe →
      (llvmpipe-CPU-fragment) path. Vertex shading happens on
      Vortex; rasterization, sampling, OM happen on the host CPU.

**Exit criteria:** a hello-triangle Vulkan demo (vkcube minus the
spinning cube, just a triangle) renders correctly via the Vortex
vertex path and CPU fragment path. Pixels match lavapipe-only
reference.

This phase establishes the "skeleton driver works"
checkpoint *before* we get tangled up in HW-unit integration.

### Phase 4 — RASTER integration (SimX model)

- [ ] Implement the **SimX RASTER model** (`sim/simx/raster/`) —
      tile→block→quad engine, edge/extents, bcoord CSRs —
      mirroring [hw/rtl/raster/](../../hw/rtl/raster/) (the
      gfx_migration Phase-3 raster scope, absorbed here; §8 risk 3).
- [ ] Replace `lp_setup_*` in `vortexpipe` with a Vortex path:
      the front-end submits primitives to the RASTER model via
      DCRs, the fragment kernel polls `vx_rast()`.
- [ ] Reuse the [draw3d/kernel.cpp](../../tests/regression/draw3d/kernel.cpp)
      idiom for the fragment kernel template; the NIR-to-LLVM
      path emits something structurally identical.
- [ ] Keep the Phase 3 CPU-fallback path selectable via env var.

**Exit criteria:** hello-triangle renders end-to-end on SimX
through the Vortex RASTER model; pixel-identical against the
Phase 3 reference.

### Phase 5 — OM integration (SimX model)

- [ ] Implement the **SimX OM model** (`sim/simx/om/`) —
      depth/stencil compare, blend, logic op, color/depth write —
      mirroring [hw/rtl/om/](../../hw/rtl/om/).
- [ ] Replace `lp_bld_depth_blend` codegen with `vx_om` emits in
      the fragment kernel.
- [ ] Plumb depth, stencil, blend, color-mask DCR state from
      `lavapipe` through `vortexpipe`.

**Exit criteria:** depth/stencil/blend tests from
[tests/regression/om](../../tests/regression/om/) (or Vulkan-port
equivalents) pass on SimX; pixel-identical against the Phase 4 +
CPU-OM reference.

### Phase 6 — TEX integration (SimX model)

- [ ] Implement the **SimX TEX model** (`sim/simx/tex/`) — address
      gen, wrap/clamp, format decode, bilinear/trilinear lerp, LOD
      selection — mirroring [hw/rtl/tex/](../../hw/rtl/tex/).
- [ ] Replace `lp_bld_sample` codegen with `vx_tex` emits.
- [ ] Plumb texture descriptors through.

**Exit criteria:** [tests/regression/tex](../../tests/regression/tex/)
Vulkan ports pass on SimX; a draw3d-equivalent Vulkan test
(textured triangle with depth) passes end-to-end through all three
Vortex graphics models — the milestone that says "we have a Vulkan
driver on the Vortex graphics path (SimX)."

### Phase 7 — Ray tracing on the SIMT cores

The end-goal RT path, brought up end-to-end. Per §4.7 Invariant 1,
ray tracing is **not** a HW unit — BVH build, traversal,
intersection, and shading all run as ordinary Vortex SIMT compute
kernels. **No new RTL.** This is the complete RT solution for
gfx-v1.

- [ ] Implement BVH **build** as a Vortex SIMT compute kernel,
      invoked when the client calls
      `vkCmdBuildAccelerationStructures`.
- [ ] Implement BVH **traversal** + ray/AABB + ray/triangle
      **intersection** as a device-side library called from the
      shader — plain compute code (FP on the SIMT FPU), not a
      custom intrinsic.
- [ ] Lower `VK_KHR_ray_query` `rayQueryEXT` ops (a NIR pass) to
      calls into that traversal library; raygen/closest-hit/miss
      shaders compile as ordinary Vortex compute kernels via the
      Shape-C path (§3).
- [ ] Keep lavapipe's CPU RT path selectable for diff (§4.5).

**Exit criteria:** a `VK_KHR_ray_query` test (a compute or
fragment shader that traces primary rays against a triangle BVH)
runs end-to-end on Vortex SimX and is pixel-identical to
lavapipe's software RT reference.

`VK_KHR_ray_tracing_pipeline` (raygen/hit/miss *pipeline* stages +
shader binding table, vs. inline `ray_query`) is a later
follow-on — driver-side plumbing over the same SIMT traversal
library; out of scope for this proposal.

### Beyond Phase 6 — graphics-model improvements (§5)

Each §5 sub-item is one commit on `feature_vulkan` with its own
**SimX unit test** under `sim/simx/{raster,tex,om}/tests/` (or the
existing SimX test harness) and (where it extends or adds an
intrinsic) one new runtime test. Prioritization order — proposal:
5.1.2 + 5.3.5
(throughput), then 5.2.1 + 5.3.1 (quad-rate, biggest perf win for
typical fragment kernels), then 5.1.1 + 5.3.6 (Hi-Z, biggest win
for occluded scenes), then 5.2.2 (compressed formats — needed for
real workloads), then 5.3.2 (MRT), then 5.1.4 + 5.3.4 (MSAA), then
the rest.

---

## 7. Test plan

| Stage | Test | Driver | Pass criterion |
|---|---|---|---|
| Phase 0 | vanilla `vkcube` | lavapipe on host | renders |
| Phase 1 | empty `vkCmdDraw` | vortexpipe + Vortex SimX | no crash; framebuffer = clear color |
| Phase 2 | vulkan vecadd compute | vortexpipe + SimX | numerical match against CPU reference |
| Phase 3 | hello-triangle | vortexpipe + SimX (CPU fragment) | pixel-identical against lavapipe reference |
| Phase 4 | hello-triangle | vortexpipe + SimX (RASTER model) | pixel-identical against Phase 3 |
| Phase 5 | depth/blend tests | vortexpipe + SimX (OM model) | pixel-identical against CPU reference |
| Phase 6 | textured-triangle | vortexpipe + SimX (TEX model) | pixel-identical against CPU reference |
| Phase 6 milestone | Vulkan-port of draw3d | vortexpipe + SimX | PNGs binary-equal to [draw3d ref PNGs](../../tests/regression/draw3d/) |
| Phase 7 | `VK_KHR_ray_query` primary-ray test | vortexpipe + SimX (SIMT) | pixel-identical against lavapipe software-RT reference |
| §5 improvements (per item) | new unittest + new regression test | unit-appropriate | as documented per improvement |

CI integration follows the gfx/hip pattern: a `--vulkan` suite in
[ci/regression.sh.in](../../ci/regression.sh.in), gated on
`$(TOOLDIR)/mesa-vortex` existing, landing in Phase 1. Per
[[feedback_test_timeout_120s]] every Vulkan test caps at `timeout 120`.

We also wire the Vulkan CTS subset (`dEQP-VK.*`) at Phase 6 — a small
subset (the `dEQP-VK.api.smoke.*`, `dEQP-VK.draw.basic_draw.*`,
`dEQP-VK.texture.filtering.*` groups) is enough to catch
conformance regressions without exploding CI runtime.

---

## 8. Risks & open questions

1. **`vp_nir_to_llvm` coverage.** Shape C (§3) is a scalar
   NIR→LLVM translator — the genuine compiler core, and the single
   largest piece of work in the project. The risk is breadth: NIR
   has a wide ALU/intrinsic surface and the translator must cover
   what real shaders emit. **Risk: medium (effort, not
   feasibility).** **Mitigation:** (a) it is *scalar* emission —
   no SoA vectorizer, the part that bloats llvmpipe's codegen;
   (b) it is built as bounded increments (Phase 2 #2–#6), trivial
   shaders first; (c) anything not yet translated falls back to
   llvmpipe (§4.5), so the driver stays whole throughout. The
   earlier Shape-A "fork 238 KB" and Shape-B "SPIR-V round-trip"
   risks are retired — Shape A was overscoped, Shape B was
   disproved by the 2026-05-22 spike (`llvm-spirv` rejects
   Vulkan SPIR-V).

2. **Lavapipe assumes host-side framebuffer.** Lavapipe's WSI path
   (`lvp_wsi_*`) assumes the framebuffer is host memory. Under
   SimX the framebuffer is device memory and we need to
   `vx_copy_from_dev` before the WSI present.
   **Risk: low.** Headless tests (the bulk of CI) don't hit WSI at
   all; for interactive use we add a present-time copy.

3. **The SimX graphics models do not exist yet.** This proposal is
   SimX-only, but the TEX/OM/RASTER SimX models were scoped as
   [gfx_migration_proposal.md](gfx_migration_proposal.md) Phase 3
   and deferred. Phases 4–6 therefore **absorb that work** —
   each phase implements one SimX unit model before integrating
   it. **Risk: medium** — it is real, non-trivial implementation
   effort (gfx_migration estimated ~7–10 days for the SimX
   rewrite), and it front-loads Phases 4–6. **Mitigation:** the
   SimX models mirror the existing, working RTL 1:1, so the RTL is
   a precise behavioral spec; build them unit-by-unit (TEX, then
   OM, then RASTER, increasing complexity) per gfx_migration's own
   Phase 3 plan; and the §5 improvements then extend models that
   already exist.

4. **Vulkan conformance scope.** The conformance model is
   inherit-and-accelerate (§4.5): `vortexpipe` exposes lavapipe's
   surface (advertised Vulkan 1.4) and the proposal's commitment /
   `dEQP-VK` target is **Vulkan 1.3 + the RT extension family** —
   lavapipe's CTS-passed level. An *official* Khronos conformance
   submission (the conformant logo) is a multi-engineer-year
   effort and is **not in scope** — anything Vortex has not
   accelerated runs correctly on lavapipe's CPU fallback, so the
   driver is functionally complete but not certified. **Risk:
   acknowledged, not mitigated.** Worth a separate conformance
   push later if we want to publish.

5. **Mesa LLVM version vs `llvm_vortex` LLVM version.**
   **Resolved (2026-05-21).** `llvm-vortex` is an LLVM 20.1.8
   build with X86 + RISCV targets and shared libraries; Mesa
   25.1.0's llvmpipe builds and runs against it directly (see §4.4,
   Phase 0). Host and device now share *one* LLVM install, so the
   version-skew risk is gone by construction. Remaining residual
   risk: a future Mesa-tag bump may outrun what LLVM 20.1.8
   supports — keep Mesa-tag bumps coordinated with `llvm_vortex`
   LLVM bumps. **Risk: low.**

6. **Gallium vs native Vulkan driver, revisited.** Choosing
   Gallium (§2.3) commits us to a path that mainline Mesa is
   gradually moving away from for Vulkan. If we want to follow
   mainline (and avoid the eventual cost of llvmpipe bit-rot),
   we'd need to port `vortexpipe` to a native `vulkan/runtime`
   driver — call it `vortexvk`. Estimate: a few months of focused
   work, after Phase 6. Defer the decision until Phase 6 ships and
   we have real perf/feature data to argue from.

7. **In-tree vs out-of-tree `vortexpipe`.** As with the MLIR
   dialect in [hip_support_proposal.md §6](hip_support_proposal.md#L307),
   out-of-tree is the more flexible choice (independent rebase
   cadence). But unlike MLIR, Gallium drivers are tightly coupled
   to Mesa internals; an out-of-tree Gallium driver is not really
   a supported configuration. **Recommendation:** keep
   `vortexpipe` in-tree in `~/dev/mesa_vortex` (our fork);
   accept the rebase cost. Revisit if/when we go native-Vulkan
   (then out-of-tree against installed Mesa is viable).

8. **Async runtime maturity.** The Vulkan command-buffer →
   `vortex2.h` async path
   ([project_command_processor.md](../../../../.claude/projects/-home-blaisetine-dev/memory/project_command_processor.md))
   wants the async API; until that lands we serialize on
   `vx_ready_wait` (works, just slower). Track but don't gate on
   it.

---

## 9. Decisions needed before starting

Please confirm or redirect:

- [ ] Driver shape: Gallium-based `vortexpipe` under lavapipe, not a
      native Vulkan driver. **OK?**
- [ ] Repo layout: Mesa fork at `github.com/vortexgpgpu/mesa`
      (local clone `~/dev/mesa_vortex`); only tests + this proposal
      + the SimX graphics-model changes for §5 land in
      `feature_vulkan`. **OK?**
- [ ] Mesa pin: `mesa-25.1` tag as the initial baseline. **OK?**
- [x] **Compilation pipeline: Shape C** (decided 2026-05-22, after
      a spike) — a scalar NIR→LLVM-IR translator in `vortexpipe`
      (`vp_nir_to_llvm`) → `llvm_vortex` → `.vxbin`. Shape B (SPIR-V
      round-trip) was disproved — `llvm-spirv` rejects Vulkan
      SPIR-V; Shape A (fork llvmpipe's SoA codegen) was overscoped.
- [ ] Phase ordering: compute pipelines (Phase 2) before any
      graphics; CPU-fragment fallback (Phase 3) before RASTER
      integration (Phase 4); graphics-model integration (4→5→6)
      decoupled from §5 improvements. **OK?**
- [ ] §5 improvement priority order: 5.1.2 + 5.3.5 → 5.2.1 + 5.3.1
      → 5.1.1 + 5.3.6 → 5.2.2 → 5.3.2 → 5.1.4 + 5.3.4 → rest.
      **OK?**
- [ ] Phases 4–6 absorb the gfx_migration Phase-3 SimX-model work
      (§8 risk 3) — each phase implements one SimX unit model
      before integrating it. **OK?**

### Decisions already taken (2026-05-21)

- [x] **Conformance model** — inherit-and-accelerate (§4.5):
      expose lavapipe's surface (advertised Vulkan 1.4),
      commit/test at **Vulkan 1.3 + the RT extension family**; no
      official Khronos submission.
- [x] **Ray tracing is the end goal**, and runs on the **SIMT
      cores**, not a HW unit (§4.7 Invariant 1, Phase 7).
      `VK_KHR_ray_query` first; `ray_tracing_pipeline` deferred.
- [x] **R/T/O stay fixed-point** in this proposal (gfx-v1);
      floating-point graphics runs on SIMT; FP inside the units is
      a future **vortex-gfx-v2** (§4.7 Invariant 2).

Phase 0's baseline (vanilla lavapipe via
[ci/mesa_install.sh.in](../../ci/mesa_install.sh.in)) is already
done; the remaining open items above gate the start of Phase 1.

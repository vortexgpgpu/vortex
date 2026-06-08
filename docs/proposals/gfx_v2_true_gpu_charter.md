# gfx_v2 — The "True GPU" Charter

**Scope:** the gfx_v2 program-level vision and architecture anchor. Defines
*what* Vortex's graphics stack becomes and *why*; the per-subsystem design
docs (tile buffer, on-device binning, CP graphics front-end, software
fallback, compiler stage coverage, FF expansion) hang off this charter and
own the *how*.
**Reference:** Laine & Karras 2011, *High-Performance Software Rasterization
on GPUs* (CUDARaster); NVIDIA pushbuffer / VRAM-resident execution model.
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Charter — vision-level; supersedes the gfx-v2 roadmap notes in
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md) §7.5
and the conformance model in [vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §5.
**Date:** 2026-06-07.
**Related:** [vortexpipe_architecture.md](../designs/vortexpipe_architecture.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md),
[custom_accelerator_isa_extensions.md](../designs/custom_accelerator_isa_extensions.md),
[command_processor_control_plane.md](../designs/command_processor_control_plane.md),
[virtual_memory_subsystem.md](../designs/virtual_memory_subsystem.md).

---

## 1. Thesis

Today **vortexpipe is a polite guest on llvmpipe**: a vtable decorator that
offloads the draws it can and hands everything else back to the CPU
([vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §1.5).
The host runs binning, orchestrates every stage, and round-trips buffers per
draw.

gfx_v2 inverts that posture. **The device becomes the renderer; the host
becomes only the driver.** Vortex executes the *entire* Vulkan pipeline —
every shader stage and the binning/setup front end — over fully
device-resident memory, with the fixed-function units (RASTER, TEX, OM) as
the fast path and an on-device SIMT software pipeline as the always-correct
fallback, orchestrated by the Command Processor, with the host untouched
until the final framebuffer is presented.

The identity shift in one line: **llvmpipe stops being the
renderer-of-last-resort and becomes an offline correctness oracle; Vortex
silicon becomes the renderer.**

---

## 2. The four pillars

1. **Every programmable Vulkan stage runs on the SIMT cores** — vertex,
   tessellation control/eval, geometry, task/mesh, fragment, compute, and
   the ray-tracing stages (raygen / any-hit / closest-hit / miss /
   intersection). All are NIR→Vortex kernels. Today only VS/FS/compute (+
   SIMT-RT, now the PRISM RTU) are covered.

2. **The fixed-function units do their fixed-function jobs, fed entirely
   device-side** — RASTER (rasterization), TEX (sampling), OM
   (depth/stencil/blend/ROP), PRISM RTU (BVH traversal). What changes vs.
   gfx-v1 is that their *input buffers and configuration* are produced and
   programmed **on-device**, never by the host.

3. **The Command Processor is the autonomous front-end** — it consumes a
   device-resident command buffer, sequences VS → setup → bin → raster → FS
   → OM, programs the FF config, and distributes work to the cores. gfx_v2
   sits *on top of* the CP
   ([command_processor_control_plane.md](../designs/command_processor_control_plane.md));
   the CP is precisely the "no host in the loop" mechanism.

4. **Full GPU-memory residency across the whole frame** — every buffer
   (resources, intermediates, attachments) is resident in device memory and
   stays there across the entire sequence of draws. There is **no
   device→host copy between draws**; the *only* egress is the final color
   framebuffer at present/scanout. Depth, tile/prim queues, transformed
   vertices: never surfaced to host.

---

## 3. The keystone: residency forces self-sufficiency

Pillar 4 is what makes "no host assistance" *logically forced* rather than
merely preferred. The current conformance model
([vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §5.2)
can fall back to llvmpipe because the data lives on the host. **Once
intermediate data never touches host memory, there is nothing for the CPU to
compute on** — you cannot drop to llvmpipe mid-frame because llvmpipe
operates on host buffers that no longer exist.

Therefore:

> Full residency ⟹ the on-device SIMT software path (CUDARaster-complete) is
> **mandatory**, not optional. It is the only possible fallback when a
> fixed-function unit cannot represent something, because the device is the
> only place the data lives.

This closes the loop: residency is *why* the FF-gap catcher must be on-chip
software, which is *why* CUDARaster is core to gfx_v2 rather than just the
binning recipe. **FF units = fast path; SIMT sort-middle = complete path;
both on-device; the CP chooses per-draw.**

---

## 4. The conformance-model inversion

| | gfx-v1 (guest on llvmpipe) | gfx_v2 (true GPU) |
|---|---|---|
| Who renders | host CPU fallback + FF acceleration of a subset | Vortex device, always |
| Unsupported feature | "→ llvmpipe (CPU)" (gated fallback) or silent collapse | "→ on-device SIMT software", never host |
| llvmpipe role | runtime fallback **and** oracle | offline golden-image **oracle only** |
| Binning | host `Binning()` (CPU) | SIMT parallel, on-device |
| Orchestration | host enqueues each stage | CP sequences the draw |
| Buffers | per-draw upload/readback | resident; present is the only egress |

Consequence: every **silent-collapse / gated-fallback** hole catalogued in
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §3.6–§3.8
(TEX mip/aniso/format; OM stencil/logic-op/MRT/MSAA/dual-source; rasterizer
overflow/w-clip/non-triangle-list), and the whole gfx-v2 FF roadmap in
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md) §7.5,
must resolve to **either real FF hardware or an on-device SIMT path —
never a host trip.**

---

## 5. Target end-to-end draw flow

```
  host: compile shaders + build command/state block ──submit──► CP command ring
                                                                  │  (host done until present)
   ┌───────────────────────── CP sequences, fully device-side ─────────────────────────┐
   │  ① VS / geometry+tess / task+mesh (SIMT) ─► transformed prims   (resident)         │
   │  ② Full triangle setup (SIMT): cull, clip 0–7 subtris, plane eqs, snap (resident)  │
   │  ③ Parallel binning (SIMT sort-middle) ─► [new tile buffer + tiling]   (resident)  │
   │  ④ CP programs RASTER / OM / TEX config device-side                                │
   │  ⑤ Fine raster (RASTER FF) ─► FS (SIMT) ─► vx_tex (TEX FF) ─► vx_om (OM FF)         │
   │      └─ for any FF-unrepresentable state: on-device SIMT software path instead     │
   └────────────────────────────────────────────────────────────────────────────────────┘
                                                                  │
                                            color attachment resident ──► present (scanout/DMA)
```

Everything between submit and present is device-resident and host-untouched.

---

## 6. Architecture deltas vs gfx-v1 (the build list)

Each item below becomes its own design doc; this charter only fixes scope and
intent.

### 6.1 On-device vertex processing + full triangle setup
Move setup out of host `Binning()`
([sw/runtime/graphics.cpp](../../sw/runtime/graphics.cpp)) onto the SIMT
cores: 1 thread/triangle, frustum + back-face + between-samples cull,
near/guardband **clipping** (variable 0–7 subtris, atomic-reserved
subtriangle array), snap to fixed-point, `(z/w),(u/w),(v/w),(1/w)` plane
equations, min-z for Hi-Z. Ordering implicit via output index. VS output
stays resident (no readback). GS/tessellation/mesh **amplification** also
lands here on SIMT — there is no FF tessellator/GS.

### 6.2 Parallel binning + tile-buffer/tiling redesign
The gfx-v1 `rast_tile_header_t` / `rast_prim_t` layout
([sw/common/vx_gfx_abi.h](../../sw/common/vx_gfx_abi.h)) is a flat,
serially-produced, per-tile prim-ID list — viable only because one host
thread writes it in order. On-device parallel binning forces a new structure
along three axes (each an open fork, §8):
- **Hierarchy** — single-level tiles vs. CUDARaster's two-level bins→tiles
  (needed when per-CTA queue fan-out is too wide for a single CTA).
- **Queue structure** — preallocated worst-case-flat vs. segmented
  linked-list with a global-atomic bump allocator (and device-side overflow
  handling, since host restart is gone).
- **Ordering** — per-CTA-queue + per-segment merge vs. atomic-append with
  order tags; must reconstruct API draw order for OM blend/determinism.

### 6.3 Rasterizer front-end redesign
`VX_raster_mem` / `VX_raster_te` / `VX_raster_arb`
([hw/rtl/raster/](../../hw/rtl/raster/)) currently fetch the flat buffer
stripe-partitioned by `INSTANCE_IDX`. They must walk the new
hierarchical/segmented/ordered structure; work distribution likely changes
with it. RTL: must close **300 MHz on U55C**; modeled in **SimX as oracle
first**, then RTL-parity diffed.

### 6.4 CP graphics front-end (orchestration)
Device-side stage sequencing, FF config (DCR/descriptor) writes, and work
distribution, built on the RTL CP. No host copy between stages or draws; the
renderpass is a self-contained device program over resident memory.

### 6.5 On-device SIMT software fallback (CUDARaster-complete)
A device-side software rasterizer / sampler / ROP that catches everything the
FF units cannot represent (exotic formats, blend/logic-op modes, MSAA
resolves, …), so the device is always self-sufficient. This is the §3
keystone, not an optional nicety.

### 6.6 Full-residency memory model
A persistent device-resident allocator: the frame working set is laid out
once and never spills to host. Leans on the `VX_MEM_PHYS` identity-mapped
pinned region
([virtual_memory_subsystem.md](../designs/virtual_memory_subsystem.md));
makes the pinned budget real; the `VX_CAPS_VM_PINNED_SIZE/_FREE` query the
suballocator plans against already exists in the runtime (§6.6 corrects the
design docs' stale "deferred" note).

### 6.7 Compiler stage coverage
Extend `vp_nir_to_llvm` beyond VS/FS/compute to GS, tessellation
(control/eval), and task/mesh, plus the device-side amplification glue.

### 6.8 FF unit expansion (fixed-point, composable)
Mobile-class **fixed-point** growth of TEX/OM with redesigned ISA/ABI
([gfx_v2_ff_expansion_roadmap.md](gfx_v2_ff_expansion_roadmap.md),
[gfx_v2_tex_v2.md](gfx_v2_tex_v2.md), [gfx_v2_om_v2.md](gfx_v2_om_v2.md)): mip/
trilinear + quad-rate `vx_tex4` (sole TEX op), `vx_om4` (sole OM op) + MRT, more
fixed-point formats, RASTER Hi-Z/early-Z. **No native FP in any FF unit** (area;
FP/float/HDR → §6.5 SW). The units are **composable primitives** — advanced
features (aniso, MSAA, programmable blend) are a thin SW layer over the FF taps,
not new datapaths — so this shrinks *both* dedicated HW and the §6.5 pure-SW
path.

---

## 7. CUDARaster as the device-side reference

CUDARaster did setup + binning **in software on the cores** and wished (its
§6.1) for HW coverage and ROP. Vortex already banked that wish (RASTER + OM,
SIMT-pullable via custom-1). gfx_v2 is the *symmetric* design: keep the FF
coverage/ROP Vortex already has, and add the on-device software front end
(setup + binning) that CUDARaster pioneered — its sort-middle structure,
per-CTA-queue ordering, persistent-thread scheduling, and segment-linked
queues are the blueprint for §6.2/§6.3, and its full software pipeline is the
blueprint for the §6.5 fallback. Hardware unknowns to resolve against the
blueprint: per-core SMEM size (coverage LUT + bin bit-matrix), global AMO +
intra-warp `popc`/ballot availability, and setup precision (Q15.16 vs wider,
per §3.8).

---

## 8. Open forks (to settle in the subsystem docs)

1. **Binning hierarchy** — one level vs two (bins→tiles).
2. **Queue structure** — flat-preallocated vs segmented-dynamic + overflow
   policy.
3. **Ordering mechanism** — per-CTA merge vs atomic-append order tags.
4. **Orchestration** — CP-sequenced (preferred) vs device-side dynamic
   launch vs persistent megakernel.
5. **Tile/bin sizes** — driven by RASTER quad-gen + cache locality (Vortex
   OM writes via ocache, not a shared-mem framebuffer — so the CUDARaster
   8×8-in-SMEM constraint does **not** transfer directly).
6. **FF-vs-software split per feature** — which §3.6–§3.8 gaps become FF
   hardware vs §6.5 software.

---

## 9. Invariants & non-goals

- **Inherit-and-accelerate is retired for the runtime.** llvmpipe remains
  the offline oracle; it is never a runtime path. (Replaces
  [vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §5.2.)
- **Fixed-function surface stays RASTER / TEX / OM (+ RTU), and stays
  fixed-point.** New FF capability extends those units (mobile-class, FP-free,
  composable — §6.8); we do not add a general-purpose graphics co-processor or a
  floating-point datapath inside any FF unit. (Consistent with §5.1.1/5.1.2,
  relaxed for RT by the PRISM RTU.)
- **Synthesizable + SimX-modeled.** Every RTL delta closes 300 MHz on U55C
  and is modeled in SimX first as the correctness oracle.
- **Commitment target: lavapipe's full advertised Vulkan surface** (currently
  **1.4**) **+ the ray-tracing extension family** (gated on PRISM RTU maturity).
  The on-device SW fallback backstops *everything*, so gfx_v2 is **not** capped
  at the gfx-v1 "commit to 1.3" — that cap was a host-fallback notion and is
  retired (supersedes [vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §5.2).
- **Non-goal (this charter):** picking the §8 forks — those are settled in
  the per-subsystem docs.

---

## 10. Phasing (indicative)

1. **Charter** (this doc) + per-subsystem design docs for §6.2/§6.3 (tile
   buffer + RASTER front-end) — the critical path.
2. On-device VS + full triangle setup (§6.1), SimX-first.
3. Parallel binning (§6.2) + RASTER front-end (§6.3), SimX → RTL parity.
4. CP graphics front-end (§6.4) — autonomous draw, zero host.
5. Full-residency memory model (§6.6), planning against the existing
   `VX_CAPS_VM_PINNED_*` query.
6. SIMT software fallback (§6.5); compiler stage coverage (§6.7); FF
   expansion (§6.8) — ongoing, conformance-driven.

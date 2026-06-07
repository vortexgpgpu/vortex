# gfx_v2 — Fixed-Function Unit Expansion Roadmap

**Scope:** the hardware track — extending RASTER / TEX / OM (the only FF
surface, invariant 5.1.1) to cover more of the Vulkan feature set, so the
on-device software fallback ([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md))
is taken less often. A prioritized catalog, not a commit-to-build-all: the SW
path already guarantees correctness, so every item here is a **perf/area
optimization**, never a correctness gate. Consolidates the gfx-v2 FF roadmap
from [graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md) §7.5
and the conformance gaps in
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §3.6–§3.8.
**Reference:** [custom_accelerator_isa_extensions.md](../designs/custom_accelerator_isa_extensions.md)
(ISA doctrine for new ops); U55C @ 300 MHz synthesis target.
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.8.
**Date:** 2026-06-07.
**Related:** [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md),
[gfx_v2_software_fallback.md](gfx_v2_software_fallback.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md),
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md).

---

## 1. Motivation & framing

The SW fallback (§6.5) makes the device **always correct** for any feature; the
FF units are the **fast path**. So FF expansion is a pure optimization curve:
each capability added to RASTER/TEX/OM removes one SW fork
([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md) §2), monotonically
shrinking how often the slow path runs. The two are **co-designed** — the
decision for each feature is *HW now* vs *leave to SW* (charter §8 fork 6),
made by:

> **value = (content frequency × per-fragment SW cost)  vs  HW cost = (U55C area/timing).**

Build the high-value, high-frequency features in hardware; leave the niche long
tail to SW. Correctness never waits on this roadmap.

---

## 2. Tiered roadmap

### Tier 0 — easy wins (cheap HW, or already present)
Closes several §3.7 OM gaps at near-zero datapath cost:
- **Enable stencil** — OM already carries full stencil state/ops in its DCRs
  ([graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md) §2);
  gfx-v1 driver just hard-disables it. Wire it through. (§3.7 stencil gap.)
- **More blend factors / logic-op** — dual-source, `CONSTANT_ALPHA`,
  independent RGB/alpha equation: extra mux cases in `VX_om_blend` /
  `VX_om_logic_op`. (§3.7 dual-source / constant-alpha / logic-op gaps.)
- **Wider RASTER fields** — lift the 16-bit `tile_x/y`/`scissor` ceiling
  (§3.8 overflow); now partly subsumed by the bin-sort key width
  ([gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) §3).

### Tier 1 — high value, common content (the big correctness/quality wins)
- **TEX mip/LOD + derivatives** — the most common missing feature; almost every
  textured draw mips. Requires per-quad derivatives → couples with **`vx_tex4`**
  (§3). Adds the mip-offset table walk (DCRs already reserve 15 mip offsets) and
  LOD selection. (§3.6 mip gap.)
- **sRGB + BC-compressed formats** — sRGB is ubiquitous; BC is the standard
  asset format, and SW decode is costly per-texel. Add format decode in
  `VX_tex_format` + sRGB→linear on sample and linear→sRGB on OM write. (§3.6
  format / §3.7 sRGB gaps.)

### Tier 2 — medium value
- **Anisotropic filtering** — quality feature, expensive in SW. TEX footprint
  sampling along the anisotropy axis.
- **MRT** — `nr_cbufs > 1`; common in deferred renderers. **`vx_om4`**-style
  multi-target write + per-RT OM state. (§3.7 MRT gap.)
- **MSAA** — RASTER per-sample coverage + OM per-sample blend + resolve; the
  highest-value feature the units lack *entirely* today (the SW path’s canonical
  case, §6 of [gfx_v2_software_fallback.md](gfx_v2_software_fallback.md)). Adds
  sample storage pressure on ocache.
- **Hi-Z / early-Z** — perf, not correctness: per-tile `z_max` cull in RASTER,
  early depth test before FS. Pairs with the min-z already produced in setup
  ([gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md) §7).

### Tier 3 — strategic / large
- **Native FP datapaths** (§4) — relaxes the gfx-v1 fixed-point invariant.
- **Bindless textures** — descriptor/addressing model change (TEX reads a
  resident descriptor table rather than per-stage DCRs); needed for modern
  Vulkan, more plumbing than datapath.

---

## 3. Quad-rate intrinsics: `vx_tex4` / `vx_om4`

The FS already shades **2×2 quads** (from `VX_raster_qe`). Quad-rate ops process
all four lanes per instruction:
- **`vx_tex4`** — takes the quad's four `(u,v)`, computes **derivatives** across
  the quad in hardware (the enabler for HW mip/aniso), selects LOD, and returns
  four texels. Both a throughput win (4 texels/op) and the *reason* mip needs
  quad-rate — so Tier-1 mip and `vx_tex4` land together.
- **`vx_om4`** — submits four fragments to OM per op (quad throughput; pairs
  with MRT).

Encode per the ISA doctrine
([custom_accelerator_isa_extensions.md](../designs/custom_accelerator_isa_extensions.md)):
lane-pack the quad operands, address per-thread windows by base register, keep
`funct7` for sub-op/format — i.e. prefer R-type with the quad carried in the
SIMD lanes, not extra source registers.

---

## 4. Native floating-point datapaths

The largest item: gfx-v1 R/T/O are fixed-point (Q15.16 edges, Q7.24 attribs,
8888 color — invariant 5.1.2). Native FP inside the units enables **float color
/ depth formats** (R16F/R32F/D32F), HDR, and removes the precision cliffs §3.8
flags. Scope and phasing:
- **OM blend in FP** first — unlocks float/HDR render targets, the most-wanted
  piece; reuses the existing FP infrastructure (cvfpu / the native FPU work).
- **Attribute interpolation + depth in FP** — removes the Q7.24 attribute
  precision loss and enables float depth.
- **Edge evaluation in FP** — last; the rasterizer's fixed-point edges are the
  most timing-sensitive on U55C.

U55C cost is real: FP datapaths consume more DSP/LUT and are harder to close at
300 MHz than fixed-point — so this is phased and partial, gated on timing. SW
covers float formats until the HW lands.

---

## 5. Memory / cache implications

- **Compressed textures** *reduce* tcache footprint and bandwidth (a win beyond
  correctness) — decode at the sampler.
- **MSAA** multiplies color/depth storage by sample count → ocache pressure and
  resolve bandwidth; the §3.2 sort-middle locality (CUDARaster) argues for
  tile-resident sample storage where it fits.
- **MRT** multiplies OM write bandwidth by target count.
- **Bindless** adds a descriptor-table fetch per sample (cacheable).

These interact with the cluster caches (tcache/rcache/ocache) and the U55C HBM
budget; each Tier-2/3 item needs a bandwidth check, not just a datapath one.

---

## 6. Sequencing & conformance-driven selection

1. **After the core pipeline lands** (binning + CP front-end + SW fallback) —
   the SW path makes the device conformant first, then FF expansion optimizes.
2. **Tier 0 → 1 → 2 → 3**, but **conformance-/profile-driven**: let
   Vulkan-CTS and real-content profiling pick which gaps to close in HW next by
   measured SW-path frequency × cost. A feature never taken by target content
   stays in SW indefinitely.
3. Every RTL delta closes **300 MHz on U55C**, modeled in **SimX first** as the
   oracle, RTL-parity diffed.

---

## 7. Open items

- **HW-vs-SW cut line per feature** — the live decision; revisit as profiling
  data arrives. Document each chosen cut in the per-feature design doc.
- **Native-FP partial adoption** — exactly which datapaths go FP and in what
  order, gated on U55C timing headroom and the FPU-sharing strategy.
- **`vx_tex4`/`vx_om4` ISA encoding** — finalize against the CUSTOM1 opcode
  budget (4 opcodes × 8 funct3) the graphics ISA already consumes.
- **Conservative raster / depth-bounds / alpha-to-coverage** — niche; expected
  to stay SW unless target content demands them.

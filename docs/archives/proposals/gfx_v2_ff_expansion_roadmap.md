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

### 1.1 Invariant: the FF units stay **floating-point free**

All three FF units (RASTER, TEX, OM) remain **fixed-point, by design, to save
area** — this *strengthens* invariant 5.1.2 (it is no longer a gfx-v1-only
constraint). Consequences:

- **Normalized/integer formats are native fixed-point** — UNORM/SNORM color
  (RGBA8, RGB565, …), fixed-point depth (D16/D24), and their blend/test math are
  exactly what fixed-point FF does. The FF units therefore cover the entire
  **mobile-class common case** in hardware with no FP.
- **Floating-point work is the SW path's job, never FF** — D32F depth, float/HDR
  render targets (R16F/RGBA16F/R11G11B10F), float textures, and any HDR blend
  route to the SIMT software fallback
  ([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md)), which is FP-capable
  on the cores. No native-FP datapath is ever added to RASTER/TEX/OM.
- **RASTER coverage must stay fixed-point regardless** — watertight,
  crack-free, deterministic rasterization *requires* fixed-point snapped edges
  (the real-GPU approach); FP edges would introduce cracks. Vulkan precision is a
  `subPixelPrecisionBits` requirement that Q15.16 satisfies; the only fix needed
  is enough sub-pixel precision for huge triangles (§3.8 normalization).

So the expansion target is **mobile-class fixed-point feature growth** for TEX
and OM, with their **ISA and ABI redesigned** (v2). The detailed redesigns live
in dedicated docs: [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) and
[gfx_v2_om_v2.md](gfx_v2_om_v2.md). This roadmap is the high-level catalog and
prioritization; RASTER needs no v2 doc (its ISA is stable; the binning redesign
[gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) covers its
front-end, and it stays fixed-point).

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

### Tier 3 — strategic / large (still fixed-point)
- **Bindless textures** — descriptor/addressing model change (TEX reads a
  resident descriptor table rather than per-stage DCRs); needed for modern
  Vulkan, more plumbing than datapath. (See [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md).)
- **Float/HDR formats are NOT here** — D32F, RGBA16F, etc. are handled by the
  SIMT SW fallback (§1.1), never by an FF FP datapath.

Per-unit detail (mobile-class feature set, redesigned ISA/ABI, cost & SW
cut-line) is in [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) and
[gfx_v2_om_v2.md](gfx_v2_om_v2.md).

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

## 4. No native floating-point in FF (the FP path is SW)

**Decision: the FF units never get a floating-point datapath** (§1.1) — area is
the reason, and for RASTER, fixed-point is also *required* for watertight
rasterization. So:

- The FF units do **fixed-point** color/depth/blend/coverage: all UNORM/SNORM
  formats, D16/D24 depth, normalized blend — the mobile-class common case, in HW.
- **Floating-point work routes to the SIMT SW fallback**, which is FP-capable on
  the cores: D32F depth, float/HDR render targets (R16F/RGBA16F/R11G11B10F),
  float textures, HDR blend. The FF/SW split is per-format, per-unit
  ([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md) §2): a draw to a
  float target uses HW RASTER + HW TEX (for its fixed-point textures) + **SW OM**.
- **Perspective-correct attribute interpolation already runs in FP on the SIMT
  cores** (the FS), using fixed-point barycentrics from RASTER — so no FF FP is
  needed there either.

This permanently removes the largest area item from the FF roadmap. The §3.8
precision concern is addressed by *more fixed-point sub-pixel bits* (wider
intermediate where a huge triangle needs it), not by going FP.

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

- **Three-tier cut line per feature** (native HW / HW-composed / pure-SW) — the
  live decision; revisit as profiling data arrives. Document each chosen cut in
  the per-feature design doc ([gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) §5,
  [gfx_v2_om_v2.md](gfx_v2_om_v2.md) §5).
- **Composition primitives** — `vx_tex4` raw-fetch tap; OM `vx_om_fetch` /
  per-sample / replace modes + the fragment-interlock guarantee — the enablers
  that move features into the composed tier instead of pure-SW.
- **`vx_tex4`/`vx_om4` ISA encoding** — finalize against the CUSTOM1 opcode
  budget (4 opcodes × 8 funct3) the graphics ISA already consumes (both are
  **sole** ops for their unit).
- **Conservative raster / depth-bounds / alpha-to-coverage** — niche; expected
  to stay SW unless target content demands them.

# gfx_v2 — On-Device SIMT Software Fallback (the always-correct path)

**Scope:** the device-side software rasterizer / sampler / output-merger that
runs on the SIMT cores when a fixed-function unit cannot represent a required
Vulkan feature. This is the keystone of the "true GPU" model: full residency
makes a host (llvmpipe) fallback impossible, so the completeness path must live
on the device. Covers the composable per-unit HW/SW fork, the shared front end,
the SW back-end components, and the compile-time selection.
**Reference:** host reference renderer
[sw/common/gfx_render.cpp](../../sw/common/gfx_render.cpp)
(`graphics::Rasterizer` / `DepthStencil` / `Blender`) — both the source and the
oracle; CUDARaster §5.4 (deferred per-sample MSAA).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.5.
**Date:** 2026-06-07.
**Related:** [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md),
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §3.6–§3.8,
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md),
[gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md).

---

## 1. Motivation

The conformance model inverts (charter §4): the §3.6–§3.8 silent-collapse /
gated-fallback holes in
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) — TEX
mip/aniso/formats/compare; OM stencil/logic-op/MRT/MSAA/dual-source/sRGB — used
to route to **llvmpipe on the host**. Full residency forbids that: the data
never leaves the device, so there is nothing for the CPU to compute on. The
completeness path must therefore **run on Vortex itself**.

> Every feature the fixed-function units cannot represent is handled by an
> on-device SIMT software implementation — never refused, never sent to host.
> FF units are the fast path; this is the **always-correct** path; the driver
> picks per-pipeline.

This *replaces* the §3.6–§3.8 "refusal gates" idea entirely: gfx_v2 does not
refuse unsupported state (that was a host-fallback notion) — it **executes it
on-device in software**.

---

## 2. The composable per-unit fork

The expensive front end (VS → setup → **binning**) is **identical** for both
paths — it always produces the bin-sort buffers on the cores
([gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md)). Only the
**per-fragment back end** forks, and it forks **per unit**, independently:

| Stage | FF fast path | SW path | Driven SW by |
|-------|--------------|---------|--------------|
| Fine rasterization | `vx_rast` (RASTER HW) | SW fine rasterizer over the bin buffer | MSAA, conservative raster |
| Texture sampling | `vx_tex` (TEX HW) | SW sampler | mip/LOD, aniso, non-8888 formats, compare, compressed |
| Output merge | `vx_om` (OM HW) | SW ROP | stencil, logic-op, MRT, dual-source, MSAA resolve, sRGB, non-8888 targets |

Because the forks are independent, a draw that only needs (say) stencil runs
**HW RASTER + HW TEX + SW OM** — the FS pulls quads from the RASTER unit and
texture-samples in hardware, then does the depth/stencil/blend RMW in software.
This **maximizes FF usage** (charter intent) — SW is engaged only for the
specific unrepresentable unit, only on the affected draws.

---

## 3. Shared front end

The binning output (`bin_headers[]`, `sorted_pids[]`, `primbuf[]`,
`draw_context`) is consumed by **both** back ends:
- FF: the RASTER unit walks it (it already does, [VX_raster_mem.sv](../../hw/rtl/raster/VX_raster_mem.sv)).
- SW: the SW fine rasterizer reads the *same* buffers — bin header → prim-id
  slice → `primbuf` records → edge-test → coverage.

So the SW path adds **zero** front-end cost; the entire setup/clip/bin
investment is reused. This is why the bin-sort schema (a plain resident buffer,
not HW-private queues) matters for the fallback too.

---

## 4. SW back-end components

All three port the validated host reference renderer
[gfx_render.cpp](../../sw/common/gfx_render.cpp) onto SIMT — same math, so the
SW path matches the FF path *and* the llvmpipe oracle bit-for-bit (§7).
Packaged as a device library (`libgfx_sw`) linked into the FS kernel.

### 4.1 SW fine rasterizer (`graphics::Rasterizer`)
Read the bin's prim slice, evaluate edge functions per sample, emit covered
quads/fragments — the software equivalent of `VX_raster_te`/`be`/`qe`. The
*rarest* fork (HW RASTER covers standard single-sample triangles); needed
mainly for **MSAA** sample coverage and conservative raster (§6).

### 4.2 SW sampler (`graphics::*` texture path)
Address + filter + decode beyond the gfx-v1 TEX block (one stage, mip 0,
A8R8G8B8, point/bilinear, CLAMP/REPEAT/MIRROR): full mip/LOD with derivatives,
anisotropic, the format zoo (R16F, sRGB→linear, BCn compressed, integer),
compare/shadow. Replaces the `vx_tex` call site in the FS.

### 4.3 SW output-merger (`graphics::DepthStencil` + `Blender`)
Depth + **stencil** test/op, **logic-op** (ROP), the full blend factor/equation
set incl. **dual-source** and `CONSTANT_ALPHA`, **MRT** (loop over N targets),
**sRGB** encode, non-8888 target formats, alpha-test/alpha-to-coverage, and the
**MSAA resolve**. Writes color/depth/stencil via the LSU instead of `vx_om`.

---

## 5. Selection mechanism (compile-time, per pipeline)

Selection is **per-pipeline at FS-compile time**, not a per-fragment runtime
branch. vortexpipe's FS wrapper (`emit_fs_wrapper`) already knows the pipeline
state and the §3.6–§3.8 feature gates; it emits, per unit, **either** the HW
intrinsic **or** an inlined `libgfx_sw` call:

```
  sample:  state needs mip/aniso/exotic-format ?  gfx_sw::tex(...)   : vx_tex(...)
  merge:   state needs stencil/logicop/MRT/MSAA ?  gfx_sw::om(...)    : vx_om(...)
  raster:  state needs MSAA/conservative        ?  SW-fine-raster loop: vx_rast loop
```

The raster fork changes the wrapper's loop shape (pull-from-HW vs.
iterate-bin-buffer), so it is two wrapper variants; TEX/OM forks are just call-
site swaps. No per-fragment dispatch overhead; the common all-HW pipeline is
unchanged.

---

## 6. MSAA — the canonical SW case

OM is single-sample; MSAA is the highest-value SW path. Follow CUDARaster §5.4:
**defer per-sample coverage** — conservative triangle-vs-pixel test, early
per-pixel `z_max` cull, then per-sample coverage + per-sample ROP only for
surviving pixels; resolve at the end. Sample storage (color/depth × samples)
lives in resident memory (no host). This combines the SW fine rasterizer (§4.1,
sample coverage) and SW OM (§4.3, per-sample blend + resolve).

---

## 7. The reference renderer is both source and oracle

[gfx_render.cpp](../../sw/common/gfx_render.cpp) already implements
`Rasterizer`/`DepthStencil`/`Blender` as the host reference for the gfx test
suite. `libgfx_sw` is that code compiled for the device. Consequently the SW
path is **correct by construction** against the same oracle the FF path is
validated against — there is no third implementation to keep in sync. This is
the cleanest possible completeness guarantee.

---

## 8. Performance posture

SW is the slow-but-correct path; the design **minimizes its scope**, never its
correctness:
- Per-unit fork (§2): only the unrepresentable unit goes SW.
- Per-draw: only draws whose state trips a gate.
- The FF fast path stays the default for conformant common content.

As FF units gain capability (charter §6.8 roadmap — `vx_tex4`/`vx_om4`, MRT,
MSAA, mip, formats), the SW path's engagement shrinks monotonically. The two
are co-designed: each FF feature added is one fewer SW fork taken.

---

## 9. Memory

The SW path uses the same resident buffers as the FF path plus, for MSAA,
resident per-sample color/depth storage. No new host transfers; everything
stays device-resident (charter pillar 4). The SW fine rasterizer may use a
local-memory tile cache (CUDARaster's shared-memory framebuffer tile) as a
transient optimization — a detail, not a DRAM footprint change.

---

## 10. Validation & phasing

1. **`libgfx_sw`** from `gfx_render.cpp`, built for the device; unit-tested on
   simx against the host reference.
2. **TEX/OM SW forks** wired into the FS wrapper behind the §3.6–§3.8 gates;
   the gfx suite run with FF disabled per-unit, diffed against the FF path and
   the llvmpipe oracle.
3. **MSAA** (§6) — the first feature the FF units don't have at all; validates
   the combined SW raster+OM path.
4. **SW fine rasterizer / conservative raster** — last, as FF RASTER covers the
   common case.

---

## 11. Open items

- **Granularity of the raster fork** — whole-draw SW raster vs. per-tile (HW
  for simple tiles, SW for MSAA/conservative tiles); per-tile is more efficient
  but needs the binning to tag tiles by required path.
- **`libgfx_sw` register/perf budget** — inlining the full sampler+OM into the
  FS kernel grows it; measure I-cache / register pressure.
- **Co-design with FF expansion** (charter §6.8) — which gaps to close in HW
  vs. leave to SW is a perf/area decision per feature.
- **Determinism** — SW path must match FF ordering (draw order from the bin-
  sort key) so mixed HW/SW draws in a pass stay consistent.

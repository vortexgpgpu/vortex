# gfx_v2 — vortexpipe Driver Changes (consolidated)

**Scope:** the host-side **vortexpipe** (Mesa/Gallium) driver changes for gfx_v2,
consolidated in one place — the driver-side counterparts of the on-device
redesign that are otherwise scattered across the per-subsystem proposals. The
gfx-v1 driver is documented in
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md); this doc is
the **delta** to it. Driver source lives in `mesa_vortex`
(`src/gallium/drivers/vortexpipe/`).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — consolidates the driver deltas from §6.1/§6.4/§6.5/§6.6.
**Date:** 2026-06-07.
**Related:** [vortexpipe_architecture.md](../designs/vortexpipe_architecture.md),
[gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md),
[gfx_v2_software_fallback.md](gfx_v2_software_fallback.md),
[gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md),
[gfx_v2_residency_allocator.md](gfx_v2_residency_allocator.md),
[gfx_v2_compiler_stage_coverage.md](gfx_v2_compiler_stage_coverage.md).

---

## 1. What stays from gfx-v1

The driver's skeleton is unchanged (vortexpipe_architecture.md §1–2):
- **Thin decorator on llvmpipe** — owns the `pipe_screen`/`pipe_context`,
  overrides the compute/draw/state hooks, forwards the rest.
- **Shape-C scalar NIR→LLVM translator** (`vp_nir_to_llvm`) shelling out to
  `clang +xvortex +zicond`.
- **The FS wrapper** (`emit_fs_wrapper`) and the per-screen/per-context state
  structs.

What changes is *where the work runs* (host → device) and *how the back end is
selected* (binary → three-tier).

## 2. The draw becomes an on-device CP command sequence

The biggest change. gfx-v1 `vp_draw_vbo` does: VS launch → **read VS output back
to host** → **host `Binning()`** → program DCRs → FS launch → read color back
(vortexpipe §3). gfx_v2 removes every host step in the middle:

- **No `Binning()` on the host, no VS readback.** VS output stays resident
  ([gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md) §4); setup +
  binning run on the cores.
- **`vp_raster_draw` becomes a command-sequence builder.** Instead of host
  binning + per-stage launches, it encodes the whole draw as one batched CP ring
  sequence (VS → setup → binning → RASTER/FS/OM), rings the doorbell once, and
  polls `Q_SEQNUM` only at draw/pass end
  ([gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md) §3/§9). The CP
  sequences the stages; `CMD_LAUNCH` drain is the inter-stage barrier.
- **DCRs are programmed inside the command stream** (CP-side), not by host calls;
  the dynamic `tile_count` is read from resident memory, not host-baked (§5
  there).

## 3. Back-end selection: three-tier, caps-driven, per unit

gfx-v1 selects HW-vs-llvmpipe per draw. gfx_v2's FS wrapper selects, **per unit,
per pipeline, at compile time**, the highest of three tiers
([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md) §2/§5):

1. **Native HW** — emit `vx_rast` / `vx_tex4` / `vx_om4`.
2. **HW-composed** — emit a thin SW layer over FF primitives (multi-tap
   `vx_tex4`; `vx_om_fetch`+`replace`) for aniso/PCF/MSAA/programmable-blend.
3. **Pure SW** — inline `libgfx_sw`.

Driven by **(a)** pipeline state (the §3.6–§3.8 feature gates) **and (b)** device
caps (`has_raster/has_om/has_tex` from `VX_CAPS_ISA_FLAGS`) — a unit physically
absent routes that unit to SIMT, **never llvmpipe**. **Zero-acceleration mode**
(all caps off) runs the whole pipeline in SIMT; it is the first bring-up target.

The translator now emits the **sole** ops `vx_tex4` / `vx_om4` (with `funct7`
modes) instead of gfx-v1's `vx_tex` / `vx_om`
([gfx_v2_custom1_isa_allocation.md](gfx_v2_custom1_isa_allocation.md)).

## 4. Residency & allocation

- **Usage-routed allocator** — FF-bound buffers → pinned-PA heap, shader-only →
  paged-VA ([gfx_v2_residency_allocator.md](gfx_v2_residency_allocator.md) §2/§8);
  derived from usage, not hand-flagged `VX_MEM_PHYS`.
- **Per-pass pool + draw-context** carved once from the pinned region; the tiling
  pool is reset/reused per draw (the static-command-list enabler).
- The driver plans residency against the **existing** `VX_CAPS_VM_PINNED_SIZE/_FREE`
  query (§6 there) — no per-frame host involvement.

## 5. Compiler stage coverage

`vp_nir_to_llvm` extends beyond VS/FS/compute to **GS / tessellation / task+mesh**
([gfx_v2_compiler_stage_coverage.md](gfx_v2_compiler_stage_coverage.md)), each a
count→scan→emit amplification feeding the same setup→binning producer; the
sw-tessellator is a runtime kernel. RT stages already lower to the PRISM RTU.

## 6. Conformance model

- **No llvmpipe runtime fallback.** llvmpipe is the **offline golden oracle**
  only (charter §4); every runtime path is on-device (FF or SIMT).
- **Target = lavapipe's full advertised Vulkan surface** (currently 1.4) + the RT
  family; the SW fallback retires the gfx-v1 "cap at 1.3" (charter §9).
- The §3.6–§3.8 silent-collapse holes are gone: every case is native-HW /
  HW-composed / pure-SW, never wrong-but-accepted.

## 7. Driver-side work list

| Area | Change | Source doc |
|---|---|---|
| `vp_draw_vbo` / `vp_raster_draw` | command-sequence builder; drop host `Binning()` + VS readback | §6.1, §6.4 |
| `emit_fs_wrapper` | three-tier per-unit selection; `vx_tex4`/`vx_om4`/composed/SW emit | §6.5 |
| caps gate | per-unit `has_*` → SIMT; zero-acceleration mode | §6.5 |
| allocator | usage-routed pinned/paged; pool + draw-context; pinned-caps planning | §6.6 |
| `vp_nir_to_llvm` | GS/tess/mesh stages; `vx_tex4`/`vx_om4` emit | §6.7 |
| graphics SDK header | `vx_tex4`/`vx_om4`/`vx_om_fetch` intrinsics (replace `vx_tex`/`vx_om`) | tex_v2/om_v2 §3 |

## 8. Open items

- **`libgfx_sw` packaging** — the device SW raster/sampler/ROP library linked
  into the FS kernel; build integration with the `clang +xvortex` path.
- **Command-sequence cost** — QMD-style atomic launch (CP §10 item 5) to keep the
  per-draw ring traffic small; the driver is the motivator.
- **STRICT-mode equivalent** — gfx-v1's `$MESA_VORTEX_STRICT` becomes "fail
  instead of pure-SW" for CI, since there's no llvmpipe runtime path to mask
  regressions.

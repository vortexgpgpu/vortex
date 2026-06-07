# gfx_v2 — Compiler Stage Coverage (GS / Tessellation / Task+Mesh)

**Scope:** extending the vortexpipe NIR→LLVM translator
(`vp_nir_to_llvm`) and the device runtime to the Vulkan shader stages not yet
covered — geometry, tessellation control/eval, and task/mesh — plus the
software amplification glue that lets every programmable stage run on the SIMT
cores and converge on the setup/binning producer. Ray-tracing stages are
covered separately (PRISM RTU). Implements charter pillar 1.
**Reference:** vortexpipe compiler architecture
([vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §2);
NIR geometry/tess/mesh intrinsics; the tessellation primitive generator.
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`); translator
in `mesa_vortex` (`src/gallium/drivers/vortexpipe/`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.7.
**Date:** 2026-06-07.
**Related:** [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md),
[gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md),
[gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md),
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md).

---

## 1. Motivation

Charter pillar 1: *every* programmable Vulkan stage runs on the SIMT cores.
Today the translator covers only **compute, vertex, fragment** (§2.2 of
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md)), and ray
tracing now lowers to the PRISM RTU. Missing: **geometry (GS)**, **tessellation
(TCS/TES)**, and **task/mesh**. There is no fixed-function tessellator or GS
unit on Vortex (invariant 5.1.1 — FF is RASTER/TEX/OM only), so these stages —
including the work a hardware tessellator would do — must run as **software on
SIMT**.

---

## 2. The unifying pattern: amplification = count→scan→emit

Every geometry stage that changes the primitive count is a **variable-output
expansion**, and they all reduce to the *same* primitive already used for
clipping ([gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md) §6)
and binning ([gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) §4):

> **count → prefix-sum → emit**: a thread computes how many outputs it
> produces, an exclusive scan assigns exact, contiguous, draw-ordered output
> slots, then each thread writes at its offset.

GS amplification (1 prim → 0..N prims), tessellation (1 patch → many tris),
task→mesh dispatch (1 task → N meshlets), and clipping (1 tri → 0..7 subtris)
are all instances of it. So the front end is a **chain of count→scan→emit
expansions** that always converges on **setup → binning** — there is no
bespoke machinery per stage, and ordering/determinism are preserved throughout
(the same property the bin-sort relies on).

---

## 3. Pipeline placement

```
   ┌─ classic path ─────────────────────────────────────────────┐
   │  VS ─► [TCS ─► sw-tessellator ─► TES] ─► [GS] ─► clip+setup  │──► binning ─► RASTER/FS/OM
   └────────────────────────────────────────────────────────────┘
   ┌─ mesh path ────────────────────────────────────────────────┐
   │  [Task ─► ] Mesh ─────────────────────► (prims direct)      │──► binning ─► RASTER/FS/OM
   └────────────────────────────────────────────────────────────┘
```

Each bracketed stage is optional and, when present, a count→scan→emit
expansion. The **mesh path** skips VS/input-assembly — mesh shaders emit
vertices+primitives directly into the stream that feeds setup/binning, which is
why they map naturally onto compute.

---

## 4. Translator stage coverage (`vp_nir_to_llvm`)

Add `nir->info.stage` routing (today: compute/vertex → `kernel_main(ptr)`;
fragment → `fs_main` + raster wrapper, §2.3.1) for the new stages, each a
`kernel_main` with a stage-specific prologue/epilogue and intrinsic set:

### 4.1 Geometry (GS)
- 1 thread per input primitive; fetch the input prim (with adjacency) from the
  assembled stream.
- Lower `emit_vertex` / `end_primitive` (and stream variants) to writes into
  the GS output vertex stream + an emitted-primitive counter.
- Two-phase: a **count** pass (or speculative max-vertices) feeds the scan; an
  **emit** pass writes at the scanned offset. Output stream → setup/binning.

### 4.2 Tessellation
- **TCS** — 1 thread per output control point (+ per-patch work for tess
  levels). Lower per-vertex/per-patch output stores and the outer/inner
  `gl_TessLevel*` writes → a per-patch level + control-point buffer.
- **sw-tessellator** (§5) — consumes tess levels, emits domain points +
  connectivity (the FF-tessellator replacement). count→scan→emit by patch.
- **TES** — 1 thread per generated domain point; lower `load_tess_coord` and
  per-patch/per-vertex input reads; evaluate → output a vertex into the stream.

### 4.3 Task / Mesh
- **Task** — lower `launch_mesh_workgroups` to an emitted mesh-dispatch count
  (amplification → scan → mesh launches).
- **Mesh** — 1 workgroup per meshlet; lower `set_mesh_outputs` + per-vertex /
  per-primitive output stores → vertices + primitives written **directly** into
  the setup/binning input stream (no IA).

### 4.4 Ray tracing (already covered)
`vp_nir_lower_ray_tracing_to_rtu.c` lowers `rq_*` ray-query ops to
`vortex_rt_set/get/trace/wait` CUSTOM1 intrinsics against the PRISM RTU
([vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §6.3). The
RT stages (raygen/any-hit/closest-hit/miss/intersection) thus already run on
SIMT with traversal on the RTU — no new translator work here; cross-referenced
for completeness.

---

## 5. The software tessellator kernel

The one genuine FF-replacement: a runtime kernel (not from NIR) that, given a
patch's outer/inner tessellation levels, domain (`triangles`/`quads`/`isolines`),
and spacing (`equal`/`fractional_odd`/`fractional_even`), generates the domain
points and their triangle connectivity — the standard tessellation primitive
generator. It is parameterized device code in the graphics runtime library,
sequenced between TCS and TES, expanding via count→scan→emit by patch. Its
output (domain points + topology) is exactly TES's input.

---

## 6. Orchestration & dynamic counts

Each amplification stage is a CP-sequenced launch
([gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md) §3), with its
output count written into the resident **draw-context** and consumed by the
next stage via grid-stride (§4.2 there). So a tessellated draw's command
sequence extends to `VS → TCS → sw-tessellator → TES → [GS] → setup → binning →
raster`, every inter-stage size flowing through device memory — no host, fully
composing with the CP front-end (§6.4) and residency model (§6.6). The
expansion buffers are per-pass transients in the tiling pool / paged heap.

---

## 7. Backend & arg block

Unchanged backend: `clang +xvortex +zicond` produces each stage's `.vxbin`
(§2.4). The fixed `i64[VP_ARG_SLOTS]` arg block (§2.5) extends with per-stage
input/output stream + draw-context slots; selection stays **per-stage at
compile time** (the translator emits the right prologue/intrinsics by
`nir->info.stage`), mirroring the FS HW/SW selection
([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md) §5).

---

## 8. Validation & phasing

1. **GS** first (simplest amplification; single count→scan→emit), validated on
   simx against lavapipe golden images for GS test content.
2. **Task/Mesh** next (compute-like, clean mapping; high value for modern
   content).
3. **Tessellation** last (TCS + sw-tessellator + TES is the largest piece);
   validate the sw-tessellator standalone against the FF reference patterns,
   then end-to-end.
4. Each stage diffs against the lavapipe oracle (charter §4) and reuses the
   count→scan→emit infrastructure proven by clipping/binning.

---

## 9. Open items

- **Speculative vs exact GS counts** — `max_vertices` worst-case reserve
  (simpler, wastes transient memory) vs a count pass (tighter). Mirrors the
  binning sizing knob.
- **Per-patch / per-vertex I/O layout** for TCS↔TES — the on-wire intermediate
  format (analogous to the bin-sort schema work).
- **Transform-feedback / stream-out** — another count→scan→emit consumer of the
  geometry stream; deferred.
- **Input-assembly primitive restart / adjacency** — wire through stage B of
  [gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md) §5.
- **Mesh-shader payload (task→mesh)** — the resident payload buffer sizing.

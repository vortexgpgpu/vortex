# gfx_v2 — On-Device Vertex Processing & Triangle Setup

**Scope:** moving the entire vertex front end — vertex shading, primitive
assembly, clipping, and full triangle setup — onto the SIMT cores, with output
staying device-resident. This is the producer of binning **stage 1**
([gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) §4): it
emits the expanded `primbuf` of `rast_prim_t` records plus the per-prim
bin-count. Ports the math currently in host `Binning()`
([sw/runtime/graphics.cpp](../../sw/runtime/graphics.cpp)) into device kernels.
**Reference:** host `Binning()` + `sw/common/gfx_render.cpp` (correctness
oracle); vortexpipe VS model
([vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §3.2).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.1.
**Date:** 2026-06-07.
**Related:** [gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md),
[gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md),
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md).

---

## 1. Motivation

Today the VS runs on Vortex but its output is **read back to host**, the host
runs `Binning()` (which folds triangle setup into the CPU), and only then is
work re-uploaded ([vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §3.2/§3.3).
The "true GPU" charter forbids that round-trip: VS output stays resident,
setup runs on the cores, and the result flows straight into on-device binning.
This doc defines that front end and, in doing so, closes the **w-clip /
guardband** silent-collapse hole flagged in
[vortexpipe_architecture.md](../designs/vortexpipe_architecture.md) §3.8 (host
`Binning()` does only a screen-space bbox clamp — no real near-plane clip).

---

## 2. Pipeline placement

```
  VS (SIMT) ─► [tess / GS amplification (SIMT, deferred §9)] ─► primitive assembly
            ─► clip + setup (SIMT) ─► primbuf + count[]  ══►  binning stage 2 (prefix-sum)
```

Four device stages, all resident, CP-sequenced (no host). Stages A–D below.
Stage D's output **is** binning stage 1.

---

## 3. Data contract

Reuses the existing types ([graphics.h](../../sw/runtime/include/graphics.h),
[vx_gfx_abi.h](../../sw/common/vx_gfx_abi.h)):

```c
struct vertex_t { float pos[4]; float color[4]; float texcoord[2]; }; // VS output record
struct rast_prim_t { vec3e_t edges[3];   rast_attribs_t attribs; };    // setup output (120 B)
```

- **VS output** = resident array of `vertex_t` (clip-space pos + the gfx-v1
  fixed varying set: `color[4]`, `texcoord[2]`; general varyings → §9).
- **Setup output** = expanded `primbuf : rast_prim_t[]` (one record per
  *post-clip* subtriangle) + `count[] : u32` (bins covered per record) — the
  binning stage-1 contract.

---

## 4. Stage A — Vertex shading (SIMT, resident)

Unchanged execution model from vortexpipe §3.2, minus the readback:

- Launch the VS kernel as one CTA of `vertex_count` threads
  (`grid={1,1,1}`, `block={vertex_count,1,1}`); each thread reads
  `gl_VertexIndex` from `VX_CSR_CTA_THREAD_ID_X`, fetches inputs via the
  `{base, stride}[loc]` attribute table (`emit_vs_attr_addr`:
  `table[loc].base + vid*table[loc].stride`), runs the user shader, writes its
  `vertex_t` record to `out_base + vid*stride`.
- **Output stays resident.** The setup kernel reads it directly from device
  memory; nothing returns to host. This is the only change vs. today.

For large vertex counts that exceed one CTA, dispatch a grid of CTAs over
`vertex_count` (block id × block dim + thread id = `vid`) — the per-vertex map
is order-free.

---

## 5. Stage B — Primitive assembly

Turn the index buffer + topology into triangles and assign each a **draw-order
`prim_id`** (the ordering key binning relies on):

- 1 thread per output triangle `t`. Fetch the index triple for `t` from the
  index buffer per topology:
  - `TRIANGLES`: `{3t, 3t+1, 3t+2}`.
  - `TRIANGLE_STRIP` / `FAN`: the standard index expansion (with winding
    parity for strips).
- `prim_id = t` (monotonic in submission order) — this is what makes the
  binning sort restore draw order for free.
- Non-indexed draws: `i = {t·3+0,1,2}` directly.

Baseline = triangle list (matches current `vp_raster.cpp`); strips/fans/lines
are a setup-stage expansion gated until wired
([gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) §11).

---

## 6. Stage C — Clipping (the variable-output stage)

The genuinely hard part: a triangle crossing the near plane (or the
guardband-extended side planes) must be clipped, producing **0–7
subtriangles**, each a new `rast_prim_t` record. This is what host `Binning()`
omits (it only clamps the screen bbox), and the source of the §3.8 shear class.

**Algorithm** — Sutherland–Hodgman in homogeneous clip space:
- Clip the triangle against `w ≥ ε` (near) and, if no guardband margin, the
  frustum side planes. Output polygon has up to `3 + #planes` vertices;
  fan-triangulate into `K` subtriangles (`K ∈ [0,7]`).
- Interpolate the varyings (`color`, `texcoord`) at the introduced vertices in
  clip space (linear in clip coords — perspective-correct after the divide).

**Parallel expansion** — clipping makes the prim stream variable-length, so it
uses the **same count→scan→emit pattern as binning** (one level up):
1. **Clip-count:** 1 thread/assembled-tri computes `K` (subtri count; 1 for
   the common un-clipped case). Write `subtri_count[t]`.
2. **Prefix-sum** `subtri_count[]` → `subtri_offset[]`, total `V_sub` →
   allocate `primbuf` of exactly `V_sub` from the resident pool.
3. **Emit + setup:** 1 thread/assembled-tri writes its `K` subtriangle records
   at `subtri_offset[t]` (Stage D math), assigning each a contiguous `prim_id`.

So `prim_id` space spans subtriangles; draw order is preserved because subtris
of tri `t` occupy a contiguous block ordered by `t`. The common path (no clip)
is `K=1` and degenerates to a direct write — no overhead for interior tris.

---

## 7. Stage D — Triangle setup math

Per (sub)triangle, the exact port of host `Binning()`'s setup
([graphics.cpp:107-151,233-275](../../sw/runtime/graphics.cpp#L107)), now on
the cores:

1. **Edge equations (HDC, homogeneous → perspective-correct):** `ClipToHDC`
   (viewport scale, keep `w`), then `EdgeEquation` → `(a,b,c)` per edge; flip
   winding if `det < 0`; **back-face cull** when `det` sign rejects and
   culling is enabled; **degenerate cull** when `det == 0`.
2. **Half-pixel sample offset:** `edge.c += (edge.a + edge.b)·0.5`.
3. **Fixed-point convert:** `EdgeToFixed` → `vec3e_t` Q15.16 (normalized by
   `1/maxVal`); attribute deltas `(a0−a2, a1−a2, a2)` → Q7.24 for
   `z,r,g,b,a,u,v`.
4. **Screen bbox → bin-AABB:** `ClipToScreen` (perspective divide) per vertex,
   bounding box, clamp to render target; convert to bin range
   `minBin{X,Y}..maxBin{X,Y}` at `BIN_LOGSIZE` (128 px). Skip if empty.
5. **min-z** over the (sub)triangle for Hi-Z (charter FF roadmap).
6. **Write** `primbuf[prim_id] = rast_prim_t{edges, attribs}`;
   `count[prim_id] = (maxBinX−minBinX)·(maxBinY−minBinY)` (AABB; or exact
   per-bin overlap test for a tighter count, §8).

The math is identical to the validated host/`gfx_render.cpp` reference — so the
SimX model diffs **bit-for-bit** against `Binning()` (§10).

---

## 8. Precision & knobs

- **Sub-pixel:** edges in Q15.16 (1/65536) as today. §3.8 notes the
  `EdgeToFixed` `1/maxVal` normalization can shrink small edges below
  advertised sub-pixel bits for very large triangles → optional **64-bit setup
  intermediates** + a reject-below-precision guard (knob).
- **Count tightness:** AABB count (cheap, slight over-count → a few empty-bin
  keys that the RASTER overlap test drops) vs exact per-bin overlap (tighter
  `P`, more setup work). Baseline = AABB.
- **Guardband:** a guardband margin lets side-plane clipping be skipped (only
  near-plane clip needed) — sizing is a knob; without it, full frustum clip.
- **Varying set:** baseline = fixed gfx-v1 `color`/`texcoord`; general
  varyings → §9.

---

## 9. Open items

- **General varyings.** `rast_attribs_t` is the fixed gfx-v1 set
  (`z,r,g,b,a,u,v`). Arbitrary FS varyings need a variable attrib-delta block
  in `rast_prim_t` and matching FS-wrapper interpolation — ties to the
  prim-record-compression lever and FF expansion (charter §6.8). Deferred.
- **Perspective-correct attributes.** gfx-v1 stores affine attrib deltas; true
  perspective correction interpolates `attr/w` and `1/w`. Note where this lands
  (setup deltas vs FS-wrapper divide) when generalizing varyings.
- **Tessellation / geometry / mesh amplification.** These expand the primitive
  stream between VS and assembly; same count→scan→emit expansion pattern as
  clipping (§6). Deferred (charter §6.1/§6.7).
- **Index-buffer cache locality / vertex reuse (post-T&L cache).** Not modeled
  yet; a vertex-reuse cache reduces redundant VS work for shared indices.

---

## 10. Validation & phasing

1. **SimX model first** (oracle): implement stages A–D in the SimX graphics
   path; diff `primbuf` (and resulting render) **bit-for-bit** against host
   `Binning()` / `gfx_render.cpp` on the PNG gfx suite
   (`tests/graphics/gfx_*`). Clipping is validated by near-plane-crossing test
   cases the host path currently gets wrong.
2. **SIMT kernels** for stages A–D; validate against the SimX model.
3. **CP sequencing** (charter §6.4): VS → clip/setup → binning, zero host.

Determinism: every stage is order-free or count→scan→emit (no atomic-ordered
output), so the front end is bit-exact reproducible — required by the SimX↔RTL
parity work and consistent with the binning schema.

---

## 11. Summary

VS stays on SIMT (output now resident); primitive assembly assigns draw-order
`prim_id`s; clipping expands to subtriangles via count→scan→emit; setup ports
the validated `Binning()` math to the cores and emits `primbuf` + `count[]`.
That output is exactly binning stage 1 — closing the host round-trip and the
w-clip correctness hole in one move, with the existing setup math as the oracle.

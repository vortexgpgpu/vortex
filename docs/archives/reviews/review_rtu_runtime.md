# v2 Review: prism_v3 RTU runtime stack

**Date:** 2026-06-17
**Scope:** `sw/runtime/include/raytrace.h` (vortex::raytrace), `sw/common/rtu_cfg.h`,
cross-checked against the SimX walker (`sim/simx/rtu/rtu_bvh.h`, `rtu_memory.cpp`,
`rtu_walker.cpp`) and RTL (`hw/rtl/rtu/VX_rtu_pkg.sv`, `VX_rtu_scheduler.sv`).
Reviewed read-only; no code modified.

---

## 1. Overall assessment (+ maturity grade)

The runtime stack does exactly what ISA-v2 §5.3 specifies and no more: a clean,
header-only `vortex::raytrace` library with two jobs — (1) `build_bvh_scene<W>`
transcodes a host triangle list into the CW-BVH<W> byte layout, and (2)
`config_t`/`program()` packs and writes the per-dispatch `VX_DCR_RTU_*` block.
The code is tidy, well-commented, correctly templated on `W ∈ {4,6}`, and the
per-trace-vs-per-dispatch boundary is drawn exactly where a real RT driver draws
it (TLAS pointer per-trace via the rs1 config register; pipeline-global config in
DCRs). Byte-layout constants in `rtu_cfg.h` agree with the SimX walker and the
RTL package across every field I checked.

The headline limitation is **not** a bug: `build_bvh_scene` emits a *single-leaf*
"BVH" (root *is* a `LEAF_TRI` holding every triangle, `node_count = 0`). The
walker handles a leaf-at-root correctly (`rtu_walker.cpp:248-259`), so the output
is a *valid* CW-BVH that intersects correctly — it just has zero spatial
acceleration. This is a documented stopgap (`raytrace.h:52-55`,
`rtu_cfg.h:60-66`), consistent with the proposal calling partitioning "a future
optimization." It is the right minimal increment to stand up the host surface and
the format contract; it is the wrong thing to ship as the production AS builder.

A second, more substantive observation: the `program()` DCR writes
(`CONFIG`, `CB_ENTRY_LO/HI`, `REFORM_THRESH`) currently have **no consumer** in
either SimX or RTL — the format/width is fixed at compile time
(`VX_CFG_RTU_BVH_WIDTH`) and `rtu_memory.cpp:121-125` explicitly states the
scene_kind word "is no longer read." That is faithful to the proposal ("the
missing host surface, not new state"), but the host writes are presently
dead-ends; this should be called out so nobody assumes `cfg.bvh_width=6` retargets
a width-4 build.

**Maturity grade: B− for the runtime *library as specified*; C for the AS-build
capability as a "true GPU" RT driver.** The library is correct, clean, and matches
its contract; the acceleration structure it produces does not yet accelerate.

---

## 2. Correctness findings (`file:line` — issue — severity)

- **`raytrace.h:73` / `rtu_bvh.h:315-320` — scene-header word2 disagrees host↔SimX/RTL — LOW (benign today).**
  Host writes `hdr[2] = node_count = 0u`. The SimX `VxBvhSceneHeader` and the
  drain path (`rtu_memory.cpp:130`, `memcpy(&scene_bytes, hdr+8, …)`) interpret
  word2 as **`scene_bytes`** (a prefetch-sizing hint). So the host emits 0 where
  the device expects total scene size. Today this is harmless: `rtu_memory.cpp:151`
  falls back to `kRtuMaxBvhSceneBytes` when `scene_bytes == 0`, and the RTL
  scheduler only reads word0 (`VX_rtu_scheduler.sv:621`, `RTU_SCENE_OFF_ROOT`) and
  word1-kind; word2 is unconsumed in RTL. But the host comment ("node_count") and
  the device contract ("scene_bytes") are out of sync, and a real builder *should*
  populate `scene_bytes` so the memory engine prefetches exactly the structure
  instead of the worst-case budget. **Fix:** set `hdr[2] = (uint32_t)out_scene.size()`
  and correct the comment. (`leaf_count` in word3 already matches.)

- **`raytrace.h:64,105` + `program()` — `scene_kind`/`bvh_width`/`cull_defaults` in the CONFIG DCR are written but never read — LOW.**
  No file under `sim/simx/` or `hw/rtl/` references `VX_DCR_RTU_CONFIG`,
  `RTU_CFG_SCENE_KIND_LSB`, `RTU_CFG_BVH_WIDTH_LSB`, or `RTU_CFG_CULL_LSB` as a
  *consumer*. Format and width are compile-time (`VX_CFG_RTU_BVH_WIDTH`), and
  `rtu_memory.cpp:121-125` confirms the runtime no longer reads scene_kind from the
  header either. Correctness is unaffected because the single test
  (`rtu_smoke_host_cfg`) builds width-4 and runs a width-4 build. Risk: a caller
  who sets `cfg.bvh_width=6`/`scene_kind=BVH6` against a width-4 RTL/SimX build gets
  silently wrong traversal, with no runtime check. **Recommend** either wiring the
  DCR (real per-dispatch reconfig) or adding a host-side assert that `cfg.bvh_width`
  matches the built target.

- **`raytrace.h:78-82` — leaf flags/SBT hard-zeroed; per-tri flags only — INFO (correct, but lossy).**
  The leaf header `flags` and `prim_base` are written as 0; opacity/SBT come purely
  from the per-triangle `flags` word (`raytrace.h:95`). The walker honors per-tri
  flags (`rtu_walker.cpp:120-132`), so behavior is correct, but the host API
  (`host_bvh_t`) exposes no way to set leaf-wide `geometry_index`-correlated flags,
  `prim_base` (Vulkan `gl_PrimitiveID` base), or per-leaf SBT — capabilities the
  format and walker already support (`rtu_bvh.h:205-224`). Fine for the stopgap;
  a real builder must thread these through.

- **Per-trace / per-dispatch boundary — CORRECT.**
  `config_t` correctly *omits* the TLAS pointer; `program()` never writes
  `VX_DCR_RTU_TLAS_ROOT_LO/HI` (those DCRs exist but stay unused). The scene/TLAS
  base reaches the walker per-trace via the rs1 config register
  (`rtu_ray_t.scene_base`, `VX_rtu_pkg.sv:211`), exactly as proposal §-table rows
  54-55 prescribe and as Vulkan/DXR require (AS is a per-trace shader binding).
  No finding — this is the right design and it is implemented right.

- **Byte-layout agreement host↔SimX↔RTL — CORRECT for every field checked.**
  Scene hdr 16 B, leaf hdr 16 B, tri stride 40 B, kind low-byte + count<<8,
  child-offset leaf-bit 31 / mask 0x7fffffff, flag bits (OPAQUE=1, PROC=2,
  SBT<<8): `rtu_cfg.h:31-46` == `rtu_bvh.h:47-253` == `VX_rtu_pkg.sv:75-139`.
  The single host-emitted layout (header→leaf→tris) round-trips through the
  walker's leaf-at-root path with no mismatch.

---

## 3. Efficiency findings

- **O(N) leaf scan per ray — the single-leaf build defeats the BVH.**
  With `node_count=0` and one `LEAF_TRI` of `tri_count` triangles, every ray runs
  a full linear scan: `walk_bvh4_subtree` enters the leaf and loops all `count`
  triangles (`rtu_walker.cpp:117-119`), incrementing `perf.bvh_tri_tests` once per
  triangle. There are **zero** `bvh_box_tests` and **zero** internal-node fetches.
  Cost is Θ(rays × tris) Möller-Trumbore tests vs. the Θ(rays × log tris) a real
  CW-BVH delivers. For a 100k-triangle BLAS that is a ~6000× increase in
  intersection work per ray — i.e., no acceleration at all. Acceptable for
  1-triangle smoke tests; catastrophic for any real scene.

- **Prefetch over-fetches because `scene_bytes` is 0.**
  Because word2 is 0 (finding §2.1), `rtu_memory.cpp:151` sizes the prefetch to
  `kRtuMaxBvhSceneBytes` instead of the actual `out_scene.size()`. For a small
  single-leaf scene the memory engine fetches the worst-case line budget rather
  than the few lines the structure occupies — wasted bandwidth that a correct
  `scene_bytes` would eliminate.

- **Allocation is reasonable.** `out_scene.assign(...)` does one sized,
  zero-filled allocation (`raytrace.h:67`) with no reallocation in the tri loop;
  the `float verts[9]` staging copy per triangle is negligible. No efficiency
  concern in the builder itself — the concern is entirely the *structure* it emits.

---

## 4. Performance findings

- **No SAH / binned partitioning — the core gap.** A production BVH builder
  partitions primitives spatially (median, binned-SAH, or HLBVH/LBVH via Morton
  codes) into a tree of `INTERNAL` nodes, so traversal prunes via the quantized
  child AABB tests the walker and RTL box-PE array already implement
  (`rtu_walker.cpp:270-318`, `VX_rtu_pkg.sv:188-199`). The format, the quantized
  internal node (common origin + per-axis exp + 8-bit child AABBs), and the
  nearest-first ordered descent are *all already built and tested* — the hand-packed
  `rtu_smoke_bvh_basic` / `bvh_multilevel` / `bvh_instanced` fixtures prove the
  consumer works on real multi-node trees. The runtime simply never *emits* internal
  nodes. The single largest performance win in this stack is a real
  `build_bvh_scene` that produces the internal-node tree the device is already
  waiting to walk.

- **Width is nominal-only.** `build_bvh_scene<6>` and `<4>` emit *identical*
  bytes (only the scene_kind word differs) because a single leaf has no internal
  nodes — so CW-BVH6's wider fan-out (one node test for 6 children vs. 4)
  contributes nothing until partitioning exists. The width template is plumbing
  with no payload today.

- **Quantization quality untested.** Once partitioning lands, the int8 per-axis
  quantization (origin + 2^exp) is where BVH quality lives; the builder will need a
  tight, conservative exponent/origin fit per node to avoid false-positive box hits.
  None of that machinery exists yet.

---

## 5. "True GPU" alignment vs NVIDIA/AMD/Intel

The runtime/kernel split is well-aligned: kernel sees only the per-ray ISA, the
host owns format + dispatch config, the AS pointer is per-trace. That mirrors
DXR/Vulkan, where the AS is a per-`TraceRay`/`traceRayEXT` shader binding, not
pipeline state. Good.

Where it diverges from shipping RT drivers:

- **The AS build is host-side and trivial; real drivers do real builders, and
  increasingly on-device.** Under DXR (`BuildRaytracingAccelerationStructure`) and
  Vulkan (`vkCmdBuildAccelerationStructures`), the *driver/GPU* builds the BLAS/TLAS
  with a SAH or LBVH builder — and critically, the build itself is a **GPU command
  recorded into a command buffer and executed on the device**, not a CPU memcpy.
  NVIDIA (RTX/TTU), AMD (RDNA2+), and Intel (Xe-HPG, whose `vk_bvh.h` node shape
  this format deliberately mirrors, per `rtu_bvh.h:26-28`) all build BVHs with GPU
  compute kernels (Morton/LBVH + SAH refinement, e.g. PLOC/HLBVH). Mesa's RADV and
  ANV ship exactly these GPU build shaders. The prism_v3 host builder is a CPU
  transcoder that doesn't even partition.

- **Charter implication ("host becomes only the driver").** The gfx_v2 "true GPU"
  charter says the host should be only the driver and the whole pipeline runs
  on-device. A CPU-side BVH build is acceptable as a bring-up driver path (it is
  what early/simple Vulkan drivers do, and a legitimate fallback), but to honor the
  charter the *target* should be an **on-device BVH builder kernel** (an LBVH/PLOC
  compute shader emitting the same CW-BVH bytes), with the host library reduced to
  staging input geometry and recording the build dispatch — exactly the
  `vkCmdBuildAccelerationStructures` shape. The byte format is already shared
  host↔device, so an on-device builder can reuse `rtu_cfg.h` verbatim.

- **TLAS/instancing already designed correctly.** The 64-B instance record,
  per-instance affine + cull mask, and absolute BLAS-root-from-scene-base
  (`rtu_bvh.h:255-286`, `VX_rtu_pkg.sv:165-185`) match the BLAS/TLAS two-level model
  of DXR/Vulkan. The host builder just doesn't emit TLAS yet (`host_bvh_t` is a
  single flat tri list with one `geometry_index`).

---

## 6. v2.1 recommendations (P0/P1/P2)

**P0 — make the build actually accelerate.**
1. Implement a real `build_bvh_scene<W>`: binned-SAH (or LBVH via Morton codes for
   build speed) partitioning that emits the `INTERNAL`-node tree the walker/RTL
   already consume. Validate against the existing hand-packed
   `rtu_smoke_bvh_basic`/`_multilevel` fixtures (same format) and add a
   multi-thousand-triangle test asserting `bvh_box_tests > 0` and
   `bvh_tri_tests ≪ rays × tris`. This is the whole point of the unit.
2. Fix the scene-header word2 contract: write `hdr[2] = (uint32_t)out_scene.size()`
   (`scene_bytes`) and correct the `raytrace.h:73` comment so the device prefetch
   sizes correctly (`rtu_memory.cpp:151`). Pure-win, ~1 line.

**P1 — close the host/device config gap and the API surface.**
3. Either wire the CONFIG DCR (`scene_kind`/`bvh_width`/`cull_defaults`) into a
   SimX/RTL consumer for genuine per-dispatch reconfig, **or** add a host-side
   `static_assert`/runtime check that `cfg.bvh_width` matches the built
   `VX_CFG_RTU_BVH_WIDTH` target — today the DCR write is silently inert
   (`raytrace.h:114-124`). Document the dead-end either way.
4. Extend `host_bvh_t`/`build_bvh_scene` to thread `prim_base` (Vulkan
   `gl_PrimitiveID`), per-leaf `geometry_index`, and per-leaf SBT/flags — all
   already supported by the format/walker (`rtu_bvh.h:200-235`) but currently
   hard-zeroed (`raytrace.h:80`).
5. Add a TLAS build path (multi-`host_bvh_t` BLAS + instance transforms) so the
   instancing format that SimX/RTL already walk has a runtime producer.

**P2 — align with shipping RT drivers / the true-GPU charter.**
6. Plan an **on-device BVH builder** kernel (LBVH/PLOC compute shader emitting the
   same `rtu_cfg.h` CW-BVH bytes), with the host library reduced to staging geometry
   and recording the build dispatch — matching `vkCmdBuildAccelerationStructures`
   and the NVIDIA/AMD/Intel on-GPU build model. The shared byte format makes this a
   straight port of the P0 SAH logic into a kernel; keep the CPU builder as the
   bring-up fallback.
7. Once partitioning exists, validate int8 child-AABB quantization quality
   (conservative origin/exponent fit) to keep false-positive box hits low, and let
   CW-BVH6 earn its wider fan-out.

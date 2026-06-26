# Vortex gfx_v2 + RTU — Master "Bible"

**Status:** authoritative cross-layer reference, **validated against code** (SimX,
RTL, mesa_vortex, runtime) as of **2026-06-25**.
**Trees:** Vortex `~/dev/vortex_v3/prism_v3` (branch `prism`), driver `~/dev/mesa_vortex` (branch `prism`).
**North star (the "true GPU" proposal):** see [proposals/gfx_v2_true_gpu.md](../proposals/gfx_v2_true_gpu.md) §0 —
*pass the Vulkan CTS on the Alveo U55C @ 4 cores, rendering on-device with FF
RASTER+OM+TEX acceleration, host as driver only (compile + submit + present), no
per-draw host round-trips; a mandatory on-device SIMT software fallback for what
the FF units cannot represent; full device residency.*

Alignment legend: ✅ aligned · ⚠️ partial · ❌ missing/divergent.
Every claim below is file:line-backed; "(verified)" means read directly this pass.

---

## Table of contents
- §0 mesa_vortex GFX implementation + gaps
- §1 mesa_vortex RTU implementation + gaps
- §2 Vortex kernel ISA/ABI + gaps
- §3 Vortex runtime SW + gaps
- §4 SimX GFX + gaps
- §5 SimX RTU + gaps
- §6 RTL GFX + gaps
- §7 RTL RTU + gaps
- §8 Consolidated critical + nice-to-have gaps
- §9 Complete gfx + RTU v2 ISA table (operands + descriptions)
- §10 End-to-end: a textured cube from Vulkan API → pixels (**most important**)
- §11 TEX — how the shader calls it, arguments, return path
- §12 OM — how the shader calls it, arguments, read features
- §13 RASTER — how it's called and interacts with the shader
- §14 RTU — how it's called and interacts with the shader
- §15 Validation provenance

---

## §0 — mesa_vortex GFX implementation + alignment

**What it is.** `vortexpipe` is a **Gallium `pipe_screen`/`pipe_context` driver layered
on llvmpipe**, reached by Vulkan apps through the **lavapipe** frontend — *not* a
standalone Vulkan ICD ([vp_screen.c:5-14,79-162](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_screen.c); `llvmpipe_create_screen` is the base, vortexpipe patches `context_create`/`finalize_nir` and caches device ISA caps TEX/RASTER/OM/RTU).

**Draw path — genuinely on-device (no host binning).** `vp_raster_draw`
([vp_raster.cpp:164-464](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_raster.cpp)) embeds the gfx_v2 on-device front end (`expand_k`+`setup_k`+`binning_k`, built into `libvortexpipe`) and runs clip+setup+parallel-bin-sort over device-resident memory — **no host `graphics::Binning()`**. The whole draw is submitted as **one CP command batch** (`expand` → 9 setup/binning stages → RASTER/OM/TEX DCR writes → FS launch) via a single `vx_enqueue_commands` + one doorbell (vp_raster.cpp:385-441). Intermediates (vertex records, primbuf, tilebuf, attachments) are `VX_MEM_PHYS`-pinned and resident; the only host readback is the final color attachment (vp_raster.cpp:443).

**Compilation.** **AOT at pipeline creation** (`vp_create_vs_state`/`vp_create_fs_state`, [vp_context.c:288-372](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_context.c)): NIR → LLVM IR ([vp_nir_to_llvm.c](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_nir_to_llvm.c)) → `clang --target=riscv32/64-unknown-elf` → ELF → `vxbin.py` → `.vxbin` ([vp_compile.c:88-219](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_compile.c), via a `/tmp/vortexpipe.XXXXXX` scratch dir). The blob is cached **in the `vp_cso` for the pipeline's lifetime only** — no on-disk cache, no `VkPipelineCache` integration.

**FS ISA emitted** (vp_nir_to_llvm.c): **legacy `vx_tex`** `.insn r4 43,1,...` (937), **legacy 3-operand `vx_om`** `.insn r4 43,2,0,x0,pos_face,color,depth` (1434-1448, 1660), `vx_frag_fetch` `.insn r 43,3,1` (1375), `vx_rast_begin` (1417). FS runs as a persistent worker: `rast_begin` once, then a `vx_frag_fetch` loop (1565-1670).

| true-GPU criterion | status | evidence |
|---|:--:|---|
| On-device rendering, no host binning | ✅ | vp_raster.cpp:5-18,164-464 |
| Host = compile + submit + present | ✅ | AOT compile; one command batch |
| No per-draw host round-trip | ✅ | single `vx_enqueue_commands` (vp_raster.cpp:440) |
| FF RASTER/OM accelerated | ⚠️ | DCRs programmed; **OM op ABI divergent** (see gaps) |
| TEX accelerated | ⚠️ | emits **legacy** `vx_tex` (gated on `has_tex`) |
| Dual-path SIMT SW fallback | ❌ | fallback is **host llvmpipe**, not on-device SIMT (vp_context.c:243-254,1083-1089) |
| Full residency | ⚠️ | intermediates resident; **texture upload round-trips host** (vp_context.c:1017-1042) |

**CRITICAL gaps:**
1. **OM ABI divergence** — mesa emits the legacy 3-operand `vx_om` (color/depth in regs); the device decodes funct3=2 as the windowed `vx_om4` (color/depth from the gfx window). The driver and device are on **different OM ABIs** (see §8, §12) — a live correctness mismatch and a likely contributor to the known Vulkan-draw failures.
2. **No on-device SIMT software fallback** — unsupported state silently degrades to **host llvmpipe** (CPU). The "mandatory SW path" pillar is unmet on the driver side.
3. **No binary cache** — every `vkCreateGraphicsPipelines` recompiles (clang fork + temp-file round-trip); no disk cache / `VkPipelineCache`.

**Nice-to-have gaps:** mesa still emits legacy `vx_tex` (not `vx_tex4`); single TEX stage (gfx-v1); stencil disabled; no indirect/instanced draws; `.vxbin` goes through `/tmp` files rather than in-memory; texture re-upload per draw.

---

## §1 — mesa_vortex RTU implementation + alignment

**What it is.** A **working Vulkan ray-**query** (`VK_KHR_ray_query`) path** — *not* a full
ray-tracing pipeline. `vp_screen.c:69-75,158` advertises `driver_ray_queries = has_rtu`
and hooks `vp_nir_lower_ray_tracing_to_rtu` in `finalize_nir`
([vp_nir_lower_ray_tracing_to_rtu.c](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_nir_lower_ray_tracing_to_rtu.c), 264 lines).

**Lowering.** `rq_initialize`→stage ray (scene/flags/cull/origin/dir/tmin/tmax);
`rq_proceed`→one synchronous `wtrace`+`wait` (returns false — single-pass opaque path);
`rq_load`→read hit attrs via GETW. LLVM emit (vp_nir_to_llvm.c:949-1049): `wtrace`
binds f0..f7 via inline-asm constraints, packs config via `wgather`, `.insn r 43,7,0`;
`wait` `.insn r 43,7,1`; hit reads `.insn r 43,6,<slot>`.

**Acceleration structures.** Host builds a **CW-BVH4** from lavapipe's BVH and uploads it
device-resident ([vp_launch.c:154-565](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_launch.c): `vp_walk_node`, greedy-median CW-BVH4 builder, transcode+upload gated on `has_rtu`). Traversal is HW (RTU); shading is SIMT, one ray per lane.

| true-GPU RT criterion | status | evidence |
|---|:--:|---|
| On-device AS + HW RTU traversal | ✅ | vp_launch.c:295-565; CUSTOM1 funct3=7 |
| Host = driver only | ✅ | host builds/uploads AS, dispatches |
| Ray-query shader API | ✅ | vp_nir_lower_ray_tracing_to_rtu.c |
| any-hit / candidate / transparency | ❌ | dropped (vp_nir_lower_ray_tracing_to_rtu.c:241-245) |
| vkCmdTraceRays / SBT / raygen-CHS-miss | ❌ | absent |
| Two-level TLAS (instancing) for callbacks | ⚠️ | triangles flattened to world space (vp_launch.c:164) |
| Procedural/AABB geometry | ❌ | AABB nodes dropped (vp_launch.c:288-291) |

**CRITICAL gaps:** no any-hit/candidate handling (opaque-triangle only); no full RT
pipeline (vkCmdTraceRays/SBT). **Nice-to-have:** instancing-aware two-level TLAS,
procedural prims, motion blur, AS refit.

---

## §2 — Vortex kernel ISA/ABI + alignment

**Headers:** [sw/kernel/include/vx_graphics.h](../../sw/kernel/include/vx_graphics.h) (TEX/OM/RASTER + frag-window), [vx_raytrace.h](../../sw/kernel/include/vx_raytrace.h) (RTU + shared window prims), [sw/common/vx_gfx_abi.h](../../sw/common/vx_gfx_abi.h) (on-wire types). All ops ride **CUSTOM1 (`RISCV_CUSTOM1`)**; `funct3` selects the family (full map in §9).

**v2 doctrine (C1–C5).** FF↔SIMT crossings must be scope-partitioned (C1),
single-issue (C2), scoreboard-ordered (C3), free of shared mutable side-band (C4),
and lifecycle-explicit (C5). Enforced by `gfx_doctrine.h::check` at decode (§4).

**Dual ISA reality (verified):**
- **TEX:** *both* legacy `vx_tex` (funct3=1, R4, texel→rd) **and** windowed `vx_tex4`
  single/quad (funct3=5, texel→window + sync handle in rd) exist and are decoded.
  Legacy `vx_tex` is **live** — used by gfx_draw3d/gfx_tex/gfx_pipeline_tex kernels
  *and emitted by mesa's FS*.
- **OM:** **only** windowed `vx_om4` (funct3=2, R-type, `OmType{WRITE}`) — fire-and-forget
  (rd=x0), reads color/depth from the gfx window. The legacy direct `vx_om` intrinsic
  is gone from the Vortex headers (mesa still emits it — see §8).
- **RASTER dispatch:** **single-path** `vx_frag_fetch` (funct3=3) + `vx_rast_begin`
  (funct3=4); the legacy `vx_rast` pull/bcoord-CSR path was deleted (FWD-4d).
- **Shared window prims:** renamed this pass `vx_rt_*` → **`vx_gfx_set`/`vx_gfx_get`/
  `vx_gfx_get_after`** (SETW/GETW) — shared by gfx (OM/TEX staging, frag payload) and
  RTU (callback slots). RTU-domain ops keep `vx_rt_` (`vx_rt_wtrace`/`wait`/`cb_ret`/`get_objray`).

| criterion | status | note |
|---|:--:|---|
| Single dispatch path | ✅ | frag_fetch only; vx_rast deleted |
| Single OM op | ✅ (device) / ❌ (driver) | device=vx_om4; mesa emits legacy vx_om |
| Single TEX op | ❌ | legacy vx_tex + vx_tex4 both live (intentional, §1.4) |
| Doctrine-clean handoff | ⚠️ | TEX/frag/RTU clean; **OM(`vx_om4`)+`SETW` flagged KnownViolation** (C3/C4) |

**CRITICAL gap:** OM/SETW C3/C4 (vx_om4 fire-and-forget + shared cross-unit window) —
needs a scoreboard handle + per-unit windows. **Nice-to-have:** finish TEX migration to
`vx_tex4` and retire legacy `vx_tex` (blocked on mesa, see §8).

---

## §3 — Vortex runtime SW + alignment

**Files:** [sw/runtime/graphics.cpp](../../sw/runtime/graphics.cpp), [sw/runtime/include/graphics.h](../../sw/runtime/include/graphics.h), async API [vortex2.h](../../sw/runtime/include/vortex2.h).

- **DrawCommands / batch:** the draw is one batched CP submission (`vx_enqueue_commands`)
  with one doorbell — 9 front-end launches + FF DCR writes + FS launch, CP-sequenced
  in order with inter-stage barriers (no host round-trip between stages).
- **FF register emitters:** `program_raster/om/tex(DrawCommands&)` drive the DCR
  sequences (one register layout per unit).
- **FrontEndPool residency:** **pooled** this pass — 12 device-scratch regions collapsed
  into one 64B-aligned slab; the 2 FF-pinned-PA outputs (`prim`,`tilebuf`) stay separate.
  **14 → 3 allocations** (graphics.cpp:476-520, commit `9fd13baa`) — the two-heap split
  (shader-scratch vs FF-pinned) is now real.
- **pid width:** `static_assert(PIPE_PRIM_BITS >= VX_RASTER_PID_BITS)` (graphics.cpp:161) +
  runtime bound-check — no silent aliasing.
- **host `Binning()`:** retained only as a coverage **oracle**, not the runtime path.

| criterion | status | note |
|---|:--:|---|
| One doorbell per draw | ✅ | DrawCommands batch |
| Device-resident pool | ✅ | pooled slab (14→3) |
| Two-heap (FF-pinned vs scratch) | ⚠️ | split started; not yet a persistent allocator |
| No host in the loop | ⚠️ | host still *builds* the 9-stage list (no single OP_DRAW) |

**CRITICAL gap:** none blocking. **Nice-to-have:** formalize a persistent two-heap
residency allocator; binning-queue overflow back-pressure (no host restart exists under
full residency).

---

## §4 — SimX GFX + alignment (the functional oracle)

All claims **verified** against `sim/simx/`:

- **RASTER:** producer FSM (`IDLE→LOAD_TILES→LOAD_PIDS→LOAD_PRIMS→RASTERIZE→RASTER_DRAIN→READY`),
  TE/BE walk, 12-byte `rast_bin_header_t`, **cycle model** (`raster_core.cpp` `walk_cycles_`/`RASTER_DRAIN`), perf counters. (raster_core.{h,cpp}, raster_unit.*)
- **OM:** full R-M-W — depth test, stencil test+ops, blend (modes+funcs), logic-op,
  write-masks — + **same-pixel interlock** (`collides_with_inflight`), capture-once
  (`om_captured_`). (om/om_core.cpp; sfu_unit.cpp:326-369)
- **TEX:** trilinear (mip blend), 7 formats, 3 wrap modes, `vx_tex4` single/quad **and**
  legacy `vx_tex`. (tex/tex_core.cpp; decode.cpp:973/984)
- **Dispatch single-path:** `RasterType{BEGIN,FWD_RUN}` only — no POP/vx_rast. (types.h:670; decode.cpp:1009-1025)
- **Doctrine assertion** (`gfx_doctrine.h`): TexType→Scoreboarded, FWD_RUN→Scoreboarded,
  BEGIN/CB_RET→SideEffectFree, **OmType + SETW→KnownViolation**. (gfx_doctrine.h:64-102)
- **FWD-5 window payload:** `FRAG_SLOT_BASE=8`, `FRAG_WORDS=14`; frag_fetch stages into the
  window, not LMEM. (gfx_window.h:49-50; sfu_unit.cpp:234-242)

**Alignment:** ✅ the functional model is complete and is the byte-exact oracle.
**CRITICAL gap:** the OM/SETW KnownViolation is *modeled* (capture-once snapshot) but not
*doctrine-fixed* — see §8. **Nice-to-have:** none material.

Doc note: `gfx_v2_true_gpu.md` §2 rows "OM/TEX RTL not built / dispatch still pull/arb /
setup_k no cull / FrontEndPool 16 allocs" are **stale** (that table is the 2026-06-20
snapshot; this bible + `gfx_v2_status_and_schedule.md` supersede it).

---

## §5 — SimX RTU + alignment

**Files:** `sim/simx/rtu/*`, `sfu_unit.cpp`, `gfx_window.h`. ISA v2: TRACE2/WAIT2 macro-ops
(per-warp sequencer streams the f0..f7 ray window / retires the hit window), callbacks
(CB_RET parked-context release), windowed reads (GETW/GETWF). 23/23 simx RTU tests pass
(per project memory). Doctrine: TRACE2/WAIT2/GETW(F)→Scoreboarded, CB_RET→SideEffectFree,
**SETW→KnownViolation** (shared-window write).

**Alignment:** ✅ functional ray-query + callbacks (CHS/AHS/IS/MISS) + TLAS modeled.
**CRITICAL gap:** SETW shares the C4 cross-unit window (same fix as OM). **Nice-to-have:**
the gfx-window coupling means a per-unit-window redesign also cleans RTU.

---

## §6 — RTL GFX + alignment

All **verified** against `hw/rtl/`:

- **OM datapath — real, not a stub:** `VX_om_{core,ds,blend(+func/minmax/multadd),compare,
  logic_op,stencil_op,mem}.sv` — full depth/stencil/blend/logic-op/writemask R-M-W
  (VX_om_core.sv:132-194; VX_om_logic_op.sv all 16 ops).
- **TEX datapath — real:** `VX_tex_{core,addr,sampler,format,mem,wrap,lerp,sat,stride}.sv`
  — wrap→addr(+mip)→fetch→filter (16 modules).
- **RASTER datapath + FWD dispatch:** `VX_raster_{core,te,be,qe,edge,mem,arb}.sv`;
  `VX_raster_unit.sv` exposes `win_wr_*` (window write port), `is_begin_op`/`is_fetch_op`;
  **no `is_pop_op`/`VX_raster_csr`** (deleted). 12-byte header w/ absolute `pids_offset`
  (VX_raster_mem.sv).
- **Decode** (VX_decode.sv): funct3=1 legacy `vx_tex` (is_tex4=0), 5 `vx_tex4`, 2 `vx_om4`,
  3 `frag_fetch` only, 4 `rast_begin`, 6 window, 7 RTU. Legacy `vx_rast`/`VX_CSR_RASTER_*` gone.
- **gfx window** (VX_gfx_window.sv): `GFXW_FRAG_SLOT_BASE=8`; raster `rast_wr_*` write port
  into a disjoint slot range; shared by TEX/OM/RTU/FWD.

**Alignment:** ✅ FF datapaths **exist in hardware** (not SimX-only).
**CRITICAL gap:** **parity is defined but UNPROVEN** — `ci/testcases/graphics_parity.yaml`
runs simx+rtlsim byte-exact (tol 0) across cores 1/2/4 + multi-cluster + multi-raster +
box 2-draw, but it has **not been run green** ("existence ≠ parity"); the cores>1 cells were
blocked until the CTA-dispatch busy-gap fix (`92d3d28f`) and can now run. **Nice-to-have:**
light-trace/small-frame parity proxies (rtlsim ~60 s/drawcall).

---

## §7 — RTL RTU + alignment

**Files:** `hw/rtl/rtu/*`, standalone synth DUT `hw/syn/xilinx/dut/{rtu,rtu_top}` +
`VX_rtu_core_top` wrapper. Per project memory: 18/18 rtlsim RTU tests pass; flat+BVH
callbacks + TLAS green; v1 ISA removed (v2 window ABI + SETW only); recursion/reform remain.

**Alignment:** ✅ RTU RTL implements the v2 ISA (TRACE2/WAIT2 + window + callbacks) with
SimX↔RTL parity for the covered set.
**CRITICAL gap:** Phase-4 timing/FPGA on U55C @300 MHz not yet closed (synth deferred until
rtlsim-green per project rule). **Nice-to-have:** transform(36)/stack/measurement parity
items; recursion/reform.

---

## §8 — Consolidated gaps

### CRITICAL (block the north star / correctness)
1. **OM driver↔device ABI divergence.** mesa FS emits legacy 3-operand `vx_om`
   (`.insn r4 43,2,...`, color/depth in regs); device decodes funct3=2 as windowed
   `vx_om4` (color/depth from the gfx window). **These are incompatible.** Reconcile:
   migrate mesa FS to `vx_om4` (stage via `vx_gfx_set`) — the correct, doctrine-aligned
   direction. (vp_nir_to_llvm.c:1434-1448 vs decode.cpp case 2 / sfu_unit.cpp:328-340)
2. **OM/SETW doctrine C3/C4.** `vx_om4` fire-and-forget (rd=x0) + reads the cross-unit
   shared gfx window; `SETW` writes it — both `KnownViolation`. Fix = scoreboard handle for
   `vx_om4` + per-unit scoreboard-retired windows (retire the shared window).
3. **RTL FF parity unproven.** Run `graphics_parity.yaml` green on rtlsim (cores 1/2/4,
   multi-cluster, multi-raster, box 2-draw); fix what the migrated OM/TEX/RASTER surface.
4. **No on-device SIMT software fallback.** Driver falls back to **host llvmpipe**; the
   mandatory dual-path completeness pillar is unmet (`gfx_sw.h` is OM-only; no SW raster/sampler).
5. **RT completeness.** No any-hit/candidate (opaque-only); no vkCmdTraceRays/SBT.

### NICE-TO-HAVE
- Shader-binary cache (disk / `VkPipelineCache`); avoid `/tmp` `.vxbin` round-trip.
- Finish TEX migration to `vx_tex4`; retire legacy `vx_tex` (blocked on #1's mesa work).
- CP `OP_DRAW`/`RES_GFX` so a draw is one device-orchestrated command (today: a host-built
  list of 9 launches + DCRs in one batch; `OP_LAUNCH_QMD` already collapses the DCRs).
- Persistent two-heap residency allocator; binning-overflow back-pressure.
- Per-sample sub-triangle clip feedback; guardband; Hi-Z min-z.
- Texture stays device-resident (avoid per-draw re-upload).
- GS/tessellation/mesh stages; multi-sampling; instanced/indirect draws; stencil in driver.
- RTU U55C @300 MHz timing closure; two-level TLAS / procedural prims.

---

## §9 — Complete gfx + RTU v2 ISA (CUSTOM1 / EXT2)

All ops: opcode `CUSTOM1`, decoded under `Opcode::EXT2`. For window ops (funct3 6/7),
`funct2 = funct7[1:0]` (sub-op) and **slot = funct7[6:2]**. `tex4`: `funct7 = {out_slot<<2,
stage<<1, mode}`. "Class" = `gfx_doctrine.h` handoff class.

| funct3 / sel | Op (intrinsic) | rd | rs1 | rs2 | extra / window | Class | Description |
|---|---|---|---|---|---|---|---|
| 0 / funct2=lane | WGATHER (`vx_wgather`) | int | int | int | rs3 int | n/a | Warp lane-pack gather (builds RTU trace config). Not FF. |
| 1 / funct2=stage (R4) | **legacy `vx_tex`** | texel | u | v | rs3=lod | Scoreboarded | Direct sample; texel→rd. *Live; emitted by mesa FS.* |
| 5 / funct7={out_slot,stage,mode} | `vx_tex4_single/quad` | texel+handle | lod / dims | in-slot base | window: u,v in; texel→out_slot | Scoreboarded | Windowed sample; quad mode = HW LOD from 2×2 derivatives. |
| 2 | `vx_om4` | **x0** | quad desc `cov[3:0]\|qx\|qy\|face@31` | window slot base | color[0..3]@base, depth[0..3]@base+4 | **KnownViolation** | OM submit (depth/stencil/blend/logic-op/writemask R-M-W). Fire-and-forget. |
| 3 | `vx_frag_fetch` | drained flag | dest base (legacy operand) | — | stages `frag_payload_t`→window 8..21 | Scoreboarded | FWD self-pull of next covered-quad wave; rd=1 when drained. |
| 4 | `vx_rast_begin` | x0 | — | — | — | SideEffectFree | Per-frame raster trigger (idempotent). |
| 6 / funct2=0 | `vx_rt_cb_ret` | x0 | action | — | — | SideEffectFree | RTU callback: release parked context; then `mret`. |
| 6 / funct2=1 | **`vx_gfx_set`** (SETW) | x0 | value | — | slot=funct7[6:2] | **KnownViolation** | Write one shared-window slot (stage OM/TEX operands, RTU slots). |
| 6 / funct2=2 | `vx_gfx_get_after`(FP), `vx_rt_get_objray` (GETWF) | f-base | sb chain (x0=none) | count | start slot | Scoreboarded | FP windowed read of `count` float slots (count>1 ⇒ macro-op). |
| 6 / funct2=3 | `vx_gfx_get` (GETW), `vx_frag_payload` | int-base | sb chain | count | start slot | Scoreboarded | GP windowed read (RTU hit IDs; FWD-5 frag payload). |
| 7 / funct2=0 | `vx_rt_wtrace` (TRACE2) | handle | lane-packed config | — | implicit f0..f7 ray window | Scoreboarded | Issue one async ray (macro-op streams the ray window). |
| 7 / funct2=1 | `vx_rt_wait` (WAIT2) | status | handle | — | hit window: t/u/v→f0..f2, ids→t3..t5 | Scoreboarded | Block until terminal; single-op park/revive (survives callback trap). |

Naming note (this pass): the **shared** window primitives are `vx_gfx_set`/`vx_gfx_get`/
`vx_gfx_get_after`; RTU-domain ops keep `vx_rt_` (`wtrace`/`wait`/`cb_ret`/`get_objray`).
`vx_rt_wtrace_sync`, `vx_rt_get` etc. are intrinsic *compositions*, not distinct opcodes.

---

## §10 — End-to-end: a textured cube, Vulkan API → pixels (MOST IMPORTANT)

This traces a simple textured-cube `VK_KHR` app from API calls to framebuffer, naming
*when the compiler runs, how binaries are cached/loaded, and who orchestrates each stage.*
Host side = mesa + runtime; device side = CP/KMU/CTA/FF.

### A. Device & pipeline setup (host)
1. **`vkCreateInstance/Device`** → lavapipe → `vortexpipe_create_screen`
   ([vp_screen.c:79](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_screen.c)): builds an llvmpipe base screen, `vx_device_open(0)`, patches Gallium hooks, **caches device ISA caps** (TEX/RASTER/OM/RTU) for capability gating.
2. **`vkCreateGraphicsPipelines`** → lavapipe compiles GLSL/SPIR-V → NIR, then calls
   `create_vs_state`/`create_fs_state` ([vp_context.c:288-372](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_context.c)).
   **➜ COMPILER RUNS HERE (AOT, at pipeline creation — not at draw):**
   NIR → LLVM IR (`vp_nir_to_llvm`) → `clang --target=riscv32-unknown-elf -march=rv32imaf`
   against `libvortex2.a` → `kernel.elf` → `vxbin.py` → **`.vxbin`** ([vp_compile.c:88-219](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_compile.c), via a `/tmp/vortexpipe.XXXXXX` dir).
   **Binary caching:** the `.vxbin` blob is stored **in the `vp_cso` for the pipeline's
   lifetime** (reused for every draw with that pipeline). There is **no on-disk cache and
   no `VkPipelineCache`** — a second identical pipeline recompiles.
3. **Texture/vertex/uniform uploads** (`vkCmd*`/descriptor writes) become device buffers
   (`vx_buffer_create` + `vx_enqueue_write`); descriptor pointers are rewritten to device
   addresses ([vp_launch.c:656-681](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_launch.c)).

### B. The draw (host records, device executes)
4. **`vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDraw`** → lavapipe →
   `pipe_context->draw_vbo` → `vp_draw_vbo` ([vp_context.c:916-1089](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_context.c)): retrieves bound VS/FS `vp_cso`. If the draw is "simple" (direct, has both `.vxbin`, device present) it takes the on-device path; otherwise it **falls back to host llvmpipe** (vp_context.c:1088).
5. **VS launch** (`vp_launch_vs`, [vp_launch.c:737-894](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_launch.c)): loads the VS `.vxbin` module, **one thread per vertex** (grid/block sized to saturate cores), outputs transformed vertex records to a **device-resident** buffer (`vsaddr`) — *not* read back.
6. **Raster draw** (`vp_raster_draw`, [vp_raster.cpp:164-464](../../../mesa_vortex/src/gallium/drivers/vortexpipe/vp_raster.cpp)) builds **one CP command batch** (a `std::vector<vx_command_t>`):
   - `LAUNCH expand_k` (VS records → `setup_vertex_t`, on device),
   - `LAUNCH` the **9 setup/binning stages** (`PIPE_STAGE_SETUP…BSCATTER`) — clip+setup +
     parallel bin-sort → dense primbuf + tilebuf,
   - `DCR_WRITE ×N` programming **RASTER** (TBUF/PBUF/TILE_COUNT/STRIDE/SCISSOR), **OM**
     (CBUF/ZBUF addr+pitch, depth/blend/writemask), **TEX** (stage0 addr/logdim/format/filter/wrap),
   - `LAUNCH` the **FS** persistent worker.
   Submitted with **one `vx_enqueue_commands` + one doorbell** (vp_raster.cpp:440). The
   color/depth attachments and the texture are device buffers; **no host round-trip between stages.**

### C. Device-side execution
7. **CP ingest.** The Command Processor (emulation model [sim/common/cmd_processor.cpp:201-211](../../sim/common/cmd_processor.cpp); RTL `hw/rtl/cp/VX_cp_engine.sv`) walks the host-pinned ring 64 B at a time. Opcodes: `OP_LAUNCH`, `OP_DCR_WRITE`, `OP_MEM_*`, `OP_FENCE`, `OP_EVENT_*`, `OP_LAUNCH_QMD` (in-memory descriptor that collapses many DCRs into one command). **There is no `OP_DRAW`** — the CP replays the batch's launches + DCRs **in order**, draining each launch before the next (the inter-stage device barrier). So the draw is *CP-sequenced from one host batch*, but expressed as explicit commands, not a single device-orchestrated draw.
8. **DCR fan-out.** `OP_DCR_WRITE` → `ProcessorImpl::dcr_write` ([processor.cpp:272-285](../../sim/simx/processor.cpp)): KMU-range DCRs go to the KMU (entry PC 0xce1 seed, arg/mscratch, block/grid dims, lmem size); all others broadcast to every cluster's FF units (RASTER/OM/TEX `VX_DCR_*_STATE`).
9. **Launch → KMU → CTAs.** `OP_LAUNCH` pulses `vortex_start` ([cmd_processor.cpp:312-329](../../sim/common/cmd_processor.cpp)); the **KMU** walks the grid emitting one `kmu_req_t` per CTA ([kmu/kmu.cpp:73-166](../../sim/simx/kmu/kmu.cpp)) — cluster-group → intra-cluster → CTA order.
10. **CTA dispatch → warp activation.** `VX_cta_dispatch` / `CtaDispatcher` ([cta_dispatcher.cpp:72-175](../../sim/simx/cta_dispatcher.cpp)) admits one warp rank per step with **fixed-stride LMEM slots**; `Scheduler::activate_warp` ([scheduler.cpp:114-159](../../sim/simx/scheduler.cpp)) seeds PC/tmask/`mscratch` and the CTA CSRs. The warp enters via `vx_start.S`: reads kernel entry from **CSR 0xce1**, arg from **mscratch**, `jalr` to `kernel_main`. *(The cores>1 launch correctness here depends on the CTA-dispatch busy-gap fix `92d3d28f` — without it the host idle-wait latches a 1-cycle device-busy gap and stops before the kernel runs.)*
11. **Raster production + fragment self-pull.** After the RASTER DCRs and `vx_rast_begin`, the cluster-shared **RasterCore** FSM walks tiles/prims, evaluates edges, enumerates covered 2×2 quads with per-corner barycentrics ([raster/raster_core.cpp:96-176](../../sim/simx/raster/raster_core.cpp)). The FS persistent workers loop on **`vx_frag_fetch`**: the SFU pops the next covered-quad wave and **stages each lane's `frag_payload_t` (pos_mask, pid, bcoord[3][4]) into the gfx window slots 8..21** ([sfu_unit.cpp:234-242](../../sim/simx/sfu_unit.cpp)); rd=drained flag (1 ⇒ exit). The shader reads its payload via `vx_frag_payload` (GETW), interpolates, **samples TEX**, and **submits OM** which R-M-W's the color/depth attachments via the memory hierarchy. This stage is **device-autonomous** (no host).
12. **Completion + present.** As each launch retires the CP publishes a seqnum to the host-pinned `cmpl_addr` ([cmd_processor.cpp:269-273](../../sim/common/cmd_processor.cpp)); device `busy` aggregates up the hierarchy to the host idle-wait. The host `vx_queue_finish` returns, reads back the color attachment (`vx_enqueue_read`), and lavapipe presents it.

**End-to-end summary:** compile **once at pipeline create** (cached in the pipeline object,
not on disk); each draw = **one host-built CP batch** (VS launch, then expand+9 binning
stages + FF DCRs + FS, one doorbell); the device runs binning→raster→FS→OM fully resident;
only the final image is read back. The autonomy gap vs a "true GPU" is that the *host still
builds the 9-stage list* (no `OP_DRAW`), and the OM op the driver emits doesn't match the
device's `vx_om4` (§8 #1).

---

## §11 — TEX: shader call, arguments, return path

**Two forms** (both CUSTOM1):
- **Legacy direct — `vx_tex(stage,u,v,lod)`** (funct3=1, R4): operands in GP regs
  (stage→funct2, u→rs1, v→rs2, lod→rs3); **texel returned directly in rd** (packed RGBA8).
  *This is what mesa's FS emits and what gfx_draw3d/gfx_tex use.*
- **Windowed — `vx_tex4_single/quad`** (funct3=5): `lod`/`dims`→rs1, `in_slot`→rs2,
  `{out_slot,stage,mode}`→funct7. Coordinates **read from the gfx window** (u@in_slot,
  v@in_slot+1; quad: u[0..3], v[0..3] across in_slot..+7, HW derives LOD). **Texel(s) land
  in the window at out_slot.. and rd returns a scoreboard sync handle**; the shader reads
  them back with `vx_gfx_get_after(out_slot, handle)` (C3-clean).

**Per-call (dynamic) arguments:** u, v; explicit lod (single) or HW LOD (quad); stage; quad dims.

**Per-stage config (DCRs, `program_tex`, once per draw):**
`VX_DCR_TEX_STAGE`, `_ADDR` (base), `_LOGDIM` (log2 w/h), `_FORMAT`
(A8R8G8B8/R5G6B5/A1R5G5B5/A4R4G4B4/A8L8/L8/A8), `_FILTER` (POINT/BILINEAR ×
MIP_NONE/MIP_LINEAR), `_WRAP` (CLAMP/REPEAT/MIRROR per axis), `_MIPOFF(lod)`.

**HW pipeline** (`VX_tex_core` / SimX `tex`): wrap → addr-gen (mip select + stride +
baseaddr) → fetch (t-cache) → format decode → bilinear lerp → mip blend (trilinear) →
packed texel. **Return is always the filtered color**; only the transport differs (rd for
legacy, window slot + handle for tex4).

---

## §12 — OM: shader call, arguments, read features

**Call: `vx_om4(desc, base)`** (funct3=2, R-type, **rd=x0 — fire-and-forget**), the *only*
OM op the device decodes.
- `desc`→rs1: `cov_mask[3:0] | qx@[4+:14] | qy@[18+:13] | face@31`.
- `base`→rs2: gfx-window slot base where the shader staged (via `vx_gfx_set`)
  `color[0..3]@base..base+3` and `depth[0..3]@base+4..base+7`.
Per covered fragment F: `pos_x=(qx<<1)|(F&1)`, `pos_y=(qy<<1)|(F>>1)`.

**Per-draw config (DCRs, `program_om`):** CBUF addr/pitch/writemask; ZBUF addr/pitch;
DEPTH_FUNC + DEPTH_WRITEMASK; STENCIL FUNC/ZPASS/ZFAIL/FAIL/REF/MASK/WRITEMASK; BLEND
MODE (ADD/SUB/REV_SUB/MIN/MAX/LOGICOP) + FUNC (15 factors incl ALPHA_SAT) + CONST; LOGIC_OP (16).

**What it does:** per covered sub-pixel, read dst color/depth/stencil → stencil test →
depth test (face-aware) → stencil ops → blend/logic-op → write masks → write back, with a
same-pixel in-flight interlock.

**Does OM return anything / read features?** **No.** `vx_om4` is fire-and-forget (rd=x0);
it returns nothing to the shader and exposes **no readback** — no occlusion query, no
depth/stencil read-to-register, no blend-result return. Its only output is to the
color/depth/stencil attachments in memory; the framebuffer read is internal to the R-M-W.
(This fire-and-forget design is the §8 C3 doctrine item; a fix adds a *completion handle*,
not data return.)

**⚠️ Divergence:** mesa's FS emits a **legacy 3-operand `vx_om`** (color/depth in registers),
which does **not** match this windowed `vx_om4` — see §8 #1.

---

## §13 — RASTER: call + shader interaction

RASTER is a **cluster-shared producer**; the FS is a **persistent worker that self-pulls**.
- **`vx_rast_begin()`** (funct3=4): per-frame trigger, idempotent (HW dedup via
  `fetch_triggered`) — any warp may call without a barrier.
- **`vx_frag_fetch()`** (funct3=3): self-pull; **rd = drained flag (scoreboard handle)**.

**Per-draw config (DCRs, `program_raster`):** TBUF_ADDR + TILE_COUNT (tile/bin headers),
PBUF_ADDR + PBUF_STRIDE (`rast_prim_t`: edges + Q-format attribute deltas), SCISSOR_X/Y.

**Worker loop:**
```c
vx_rast_begin();
for (;;) {
    unsigned drained = vx_frag_fetch();   // pull next covered-quad wave
    if (drained) break;
    frag_payload_t p; vx_frag_load(p, drained);   // GETW from window slots 8..21
    // p.pos_mask, p.pid, p.bcoord[3][4]  → interpolate, sample TEX, submit OM
}
```
**Mechanism:** after trigger+DCRs, the unit walks tiles (TE) + prims (BE), evaluates edges,
enumerates covered 2×2 quads with per-corner barycentrics, packs `NUM_THREADS` quads into a
wave, and **stages each lane's `frag_payload_t` straight into the gfx window** (slots 8..21,
FWD-5 zero-LMEM). rd=0 ⇒ wave staged (shade it); rd=1 ⇒ drained (exit). Synchronous, no
poll/sentinel (C1/C2/C3-clean). Static screen-space tile→core ownership ⇒ each pixel touched
by one core (makes OM R-M-W ordering correct by construction, C4). RASTER is the work source;
its payload → shader interpolation → TEX → OM, all through the (shared) gfx window.

---

## §14 — RTU: call + shader interaction

RTU is an **async ray-traversal coprocessor** driven by two macro-ops + a callback path.
- **`vx_rt_wtrace(scene, payload, flags, cull, ray)`** (funct3=7 funct2=0): issue one async
  ray. rs1 = lane-packed warp-uniform config (scene/payload/flags/cull via `vx_wgather`); the
  per-lane geometry rides the **implicit f0..f7 window** (`vx_ray_t`: origin, dir, tmin, tmax).
  Returns an async **handle** in rd; non-blocking (RTU traverses while the kernel does other work).
- **`vx_rt_wait(handle, &hit)`** (funct3=7 funct2=1): block until terminal; returns a status
  word in rd; single-op park/revive so it **survives the async callback trap**. The hit
  window is delivered by chained GETWF/GETW reads (t/u/v→f0..f2, prim/inst/geom→t3..t5).
- **Callbacks** (CHS/AHS/IS/MISS): on a yield the RTU traps to an `mtvec` dispatcher; the
  dispatcher reads candidate state via `vx_gfx_get`/`vx_rt_get_objray`, decides with
  **`vx_rt_cb_ret(ACCEPT/IGNORE/TERMINATE)`** (funct3=6 funct2=0), then `mret`s to resume the
  post-`wait` PC (tmask restored from the saved CSR).

**Shader interaction:** Vulkan ray-query lowers to `wtrace`+`wait`; the RTU resolves the ray
on FF hardware (BVH4 traversal, box/tri PEs) and writes the hit window back to the shader's
registers under the scoreboard. One ray per lane; the f0..f7 / hit windows are read/written
by HW convention (the encoding names only rd/rs1). Acceleration structures are built host-side
(CW-BVH4) and uploaded device-resident.

---

## §15 — Validation provenance

Built from a direct cross-layer code read on **2026-06-25**: SimX (`sim/simx/`) and RTL
(`hw/rtl/`) validated claim-by-claim; mesa (`~/dev/mesa_vortex` `prism`) driver traced for
the gfx + RTU + end-to-end paths; runtime (`sw/runtime/`) and kernel ABI (`sw/kernel/include/`)
read directly. Supersedes the stale §2 snapshot in `gfx_v2_true_gpu.md` (2026-06-20); pairs
with `gfx_v2_status_and_schedule.md` (status + schedule). Key live findings this pass:
(1) **OM driver↔device ABI divergence** (mesa legacy `vx_om` vs device `vx_om4`);
(2) **legacy `vx_tex` is live** and mesa-emitted (not deletable yet);
(3) RTL FF datapaths **exist but parity is unproven**;
(4) OM/SETW remain the **C3/C4 doctrine** debt.

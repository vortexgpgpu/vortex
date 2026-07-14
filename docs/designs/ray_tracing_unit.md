# PRISM Ray-Tracing Unit (RTU) — Design

**Scope:** the Vortex hardware ray-tracing unit — the async BVH-traversal engine
(RTU), its ISA-v2 register-window ABI, the callback / parked-context dispatch that
runs closest-hit / any-hit / intersection / miss shaders, the CW-BVH acceleration
structure and its host builder, and the SimX model. Covers the RTL
([`hw/rtl/rtu/`](../../hw/rtl/rtu/)), the SimX models
([`sim/simx/rtu/`](../../sim/simx/rtu/)), and the kernel/host ABI
([`sw/kernel/include/vx_raytrace.h`](../../sw/kernel/include/vx_raytrace.h),
[`sw/runtime/include/raytrace.h`](../../sw/runtime/include/raytrace.h),
[`sw/common/rtu_cfg.h`](../../sw/common/rtu_cfg.h)).

This document covers the **RTU hardware microarchitecture, ISA, dispatch, and BVH
format**. The **driver / Vulkan ray-query path** (NIR `rayQueryEXT` → RTU-op
lowering, AS transcode, residency) is in
[`vortexpipe_architecture.md`](vortexpipe_architecture.md) §6.3. The RTU shares the
graphics register window and cluster plumbing documented in
[`graphics_hardware_stack.md`](graphics_hardware_stack.md) §2.1. The program-level
"true GPU" master plan (ray-query conformance is a north-star gate) is
[`../proposals/gfx_v2_true_gpu.md`](../proposals/gfx_v2_true_gpu.md).

The RTU is a RISC-V ISA extension under `custom1` (`INST_EXT2 = 0x2B`), sharing the
opcode with the graphics FF units; it is gated by `VX_CFG_EXT_RTU_ENABLE`.

---

## 1. Architecture overview

The RTU is an **asynchronous, SIMT-dispatched** ray-tracing accelerator: a warp
issues a trace, the RTU traverses the BVH off to the side while the warp
continues or parks, and hits are delivered back through **scoreboarded** register
windows (and, for programmable hit/miss/intersection shaders, through a
callback/parked-context trap). It is a per-core SFU-class unit on the same
cores/caches as the graphics units, reached through the shared graphics register
window and an **RTCache**.

```
   warp: stage ray/config (SETW) ─► vortex_rt_wtrace (TRACE2) ─► [continue / park]
                                                    │
   ┌──────────────────────────── RTU (async) ───────────────────────────────────┐
   │  VX_rtu_bvh_scheduler (context pool, short stack, 2-phase SELECT/EXEC)          │
   │     ├─ TLAS: instance descent + VX_rtu_xform (world→object, FMA-only R^T)   │
   │     ├─ VX_rtu_box_pe   (slab test; quantized child AABBs + raw/proc boxes)  │
   │     ├─ VX_rtu_tri_pe   (Möller–Trumbore; VX_fdivsqrt)                        │
   │     └─ leaf → commit hit  |  proc-AABB / non-opaque → CALLBACK yield         │
   └───────────────────────────────────┬──────────────────────────────────────────┘
                                        │ scoreboarded window (GETW/GETWF)  +  cb bus
                                        ▼
   warp: vortex_rt_wait (WAIT2) ─► read hit attributes (GETW/GETWF)
        └─ callback (CHS/AHS/IS/MISS): parked-context trap → shader → vx_rt_cb_ret
```

The RTU consumes a **CW-BVH** (compressed-wide BVH, width 4 or 6) built host-side
and made resident; traversal, instance transforms, and intersection all run in
**fixed-point / FP on the RTU's own PEs** — no traversal on the SIMT cores.

---

## 2. ISA — RTU v2 register-window ABI

RTU ops are `custom1` (0x2B / EXT2), decoded by funct3/funct2 in
[`VX_decode.sv`](../../hw/rtl/core/VX_decode.sv) and
[`decode.cpp`](../../sim/simx/decode.cpp). **ISA v1 (funct3=5) is retired** —
funct3=5 is now TEX. The v2 surface (intrinsics in
[`vx_raytrace.h`](../../sw/kernel/include/vx_raytrace.h)):

| Op | Encoding | Purpose |
|---|---|---|
| `TRACE2` | funct3=7, funct2=0 | issue one ray; per-trace config lane-packs via `vx_wgather`, ray geometry rides the FP register window; `rd` = scoreboard handle |
| `WAIT2` | funct3=7, funct2=1 | block on a trace handle; `rd` = terminal status (`DONE_HIT`/`DONE_MISS`) |
| `SETW` | funct3=6, funct2=1 | write one window slot (stage ray/config) |
| `GETW` | funct3=6, funct2=3 | read `count` contiguous integer window slots (hit ids/flags) |
| `GETWF` | funct3=6, funct2=2 | read `count` contiguous **FP** window slots (t, barycentrics, object ray) |
| `GETWS` | funct3=4 | slot-indexed window read (warp dim by `block_idx`) — shared with the gfx frag path |
| `CB_RET` | funct3=6, funct2=0 | release this lane's parked callback context (ACCEPT / IGNORE / continue) |

### 2.1 The hit window

The RTU returns results through its own **32-slot hit window**
(`VX_RT_SLOT_COUNT = 32`, one `VX_rtu_unit` per core). The RTU is its only writer
and the shader its only reader: the ray rides the TRACE burst, the payload pointer
rides the arm doorbell, and an intersection shader's hit distance and attribute
ride the operands of the CONTINUE that returns them, so nothing else ever writes a
slot. The record spans **hit attributes 10..16** and the **object ray 17..22**
([`VX_types.toml`](../../VX_types.toml) `[rtu_slots]` is the source of truth).
Every window→rd handoff is **scoreboarded** (C3 of the interface law), so a
trace's results retire in order and survive an async trap.

The window used to be shared with graphics, and the fragment payload overlapped
the RTU's slots: correctness rested on a by-convention mutual exclusion (a warp
never held live fragment *and* ray-query state at once) that was never
HW-enforced, which is what blocked **ray-query-in-a-fragment-shader**. A
fragment's stamp now rides its launch and is read from CSRs, so the window is the
RTU's alone and the overlap is gone.

---

## 3. RTL module inventory ([`hw/rtl/rtu/`](../../hw/rtl/rtu/))

### 3.1 Traversal
- **`VX_rtu_core` / `VX_rtu_bvh_scheduler`** — two compile-time walkers selected by
  `VX_CFG_RTU_BVH_WIDTH`: a **flat** list walker (WIDTH=0) and the **CW-BVH4/6**
  walker (WIDTH=4/6). The scheduler holds a per-lane **context pool**
  (`NUM_CTX = NUM_THREADS`), a **short stack** (`sp`), and a two-phase
  `SELECT`/`EXEC` pipeline that time-multiplexes contexts across the PEs.
- **`VX_rtu_box_pe`** — slab ray/AABB test over the node's quantized child AABBs;
  also handles **raw / procedural boxes** (the proc-AABB leaf path).
- **`VX_rtu_tri_pe`** — Möller–Trumbore triangle intersection (`VX_fdivsqrt_unit`).
- **`VX_rtu_xform`** — TLAS instance transform: world→object via the inverse
  rotation `Rᵀ` + translation, **FMA-only** (reuses `VX_fma_unit`, no new
  datapath); driven by the scheduler's `CS_INST_*/CS_XFORM` states under
  `VX_CFG_RTU_TLAS_ENABLE`.
- **FP helpers** — `VX_rtu_recip` (F32 reciprocal: LUT + Newton-Raphson default,
  or a BRAM-seed DSP-NR variant), `VX_rtu_fmac3` / `VX_rtu_fdot3` /
  `VX_rtu_fcross3`.

### 3.2 Callbacks & parked contexts
- **`VX_rtu_bus_if`** — the callback bus (`CB_RET` / `CBACT` / `CBYIELD`), carrying
  the committed `action_hit_t`.
- **Parked-context dispatch** — when a leaf needs a programmable **any-hit (AHS)**,
  **intersection (IS/proc-AABB)**, **closest-hit (CHS)**, or **miss** shader, the
  scheduler yields: the warp takes an async trap, runs the callback shader, and
  releases the context with `CB_RET` (ACCEPT/IGNORE/continue). An intersection
  shader's hit distance and attribute ride the operands of that `CB_RET`, so it
  hands its verdict back without writing RTU state. **FP is legal inside the
  callback trap** (a scoreboard snapshot/restore around the trap).

### 3.3 Memory / BVH
- **`VX_rtu_core`** reads BVH nodes + triangles through the **RTCache**. A CW-BVH4
  node is exactly one cache line, so a fetch is one aligned line read tagged with
  the requesting context id; the outstanding count is bounded by the cache's own
  MSHRs. Node/tri layouts are in [`VX_rtu_pkg.sv`](../../hw/rtl/rtu/VX_rtu_pkg.sv)
  and must match the host builder byte-for-byte
  ([`sw/common/rtu_cfg.h`](../../sw/common/rtu_cfg.h)).

---

## 4. SimX model ([`sim/simx/rtu/`](../../sim/simx/rtu/))

- **`rtu_unit.cpp`** — the TRACE2/WAIT2 window ABI, macro-op → uop generation, and
  the park/revive across an async trap (`rtu_unit.cpp:65-137,231-313`).
- **`rtu_core.cpp`** — the traversal model: multi-AS / TLAS with instance
  transforms, the object-ray slots (`VX_RT_OBJECT_RAY_*`), and **all four callback
  types** (AHS / IS / CHS / MISS) with per-type counters, a **ReformationEngine**,
  and a warp callback-in-flight ordering gate.
- **`rtu_isect.cpp`** — `ray_aabb_intersect` (proc-AABB) + triangle math.
- SimX is the **fuller oracle**: same-warp reformation is real; multi-warp and
  SBT-divergent reformation are modeled in SimX but not yet in RTL.

---

## 5. Acceleration structure — CW-BVH

- **Format:** a **compressed-wide BVH** of width 4 or 6 — quantized child AABBs
  per node, one triangle per leaf. The byte layout is shared host↔device
  (`raytrace.h` builder ↔ `rtu_cfg.h`/`VX_rtu_pkg.sv` consumer).
- **Host builder:** `vortex::raytrace` in
  [`sw/runtime/include/raytrace.h`](../../sw/runtime/include/raytrace.h)
  (`build_bvh_scene` / `BvhBuilder`): binned-SAH, quantized child AABBs. Two-level
  TLAS→BLAS with instance transforms.
- **Driver bridge:** the Vulkan driver transcodes an app `VkAccelerationStructure`
  to this layout (`vp_transcode_as`) and makes it resident; see
  [`vortexpipe_architecture.md`](vortexpipe_architecture.md) §6.3. (Today it is
  **rebuilt per dispatch** — AS residency is a tracked gap.)

---

## 6. VM / residency tie-in

Like the graphics FF units, the RTU AXI master **bypasses the MMU** and uses the
physical addresses configured for the resident BVH; BVH/scene buffers are carved
from the `VX_MEM_PHYS` pinned slab (VA == PA), validated on the CP submit path.
See [`graphics_hardware_stack.md`](graphics_hardware_stack.md) §8 and
[`virtual_memory_subsystem.md`](virtual_memory_subsystem.md).

---

## 7. State of the implementation

Grades: ✅ done · ⚠️ partial · ❌ pending. All of the below is **committed** on
`prism`.

| Area | State | Note |
|---|:--:|---|
| Flat + CW-BVH4/6 traversal, context pool, short stack | ✅ | RTL + SimX |
| Box PE (quantized + raw/proc), triangle PE | ✅ | |
| TLAS + instance `XformUnit` (FMA-only) | ✅ | `VX_CFG_RTU_TLAS_ENABLE` |
| Callback path (CB_RET) + parked-context async-trap dispatch | ✅ | CHS / MISS / IS(proc) / AHS |
| Procedural-AABB / intersection-shader | ✅ | raw-box leaf → IS yield (RTL + SimX) |
| FP-in-callback-trap | ✅ | scoreboard snapshot/restore |
| ISA v2 window ABI (TRACE2/WAIT2/GETW/GETWF/CB_RET); v1 retired | ✅ | |
| Host CW-BVH builder | ✅ | `vortex::raytrace` |
| Per-triangle AHS in the CW-BVH walker | ✅ | classifier ported from the flat walker into `CS_TRI_WAIT` (face/opacity cull, opaque override, terminate-on-first-hit); non-opaque tri yields an ANYHIT callback, opaque commits |
| **In-trap recursion; multi-warp / SBT-divergent reformation** | ❌ | RTL-deferred (SimX models same-warp reform only) |
| **Sustained multi-warp servicing / async ray pool (§8.6)** | ❌ | `rt_raycast`/`bvh_multinode` wedge the scoreboard under load |
| **Ray-query-in-FS fusion** | ❌ | blocked by the window slot 8..21 ↔ RTU 8..24 overlap (§2.1) |
| **AS residency** | ❌ | BVH re-uploaded per dispatch (driver) |

**Tests:** [`tests/raytracing/`](../../tests/raytracing/) `rt_smoke_*` — **26/26
simx, 20/20 rtlsim** (6 rtlsim-deferred: `recursive`, `reform_mw`, `reform_sbt`,
`async_batch`, `bvh_multinode`, `rt_raycast`). These are the RTU regression gate.
`rt_smoke_ahs_bvh` (a non-opaque triangle in a CW-BVH4 leaf, IGNORE callback →
MISS) is the CW-BVH any-hit gate — it exercises the walker's per-triangle
classify/yield path that the flat-walker `rt_smoke_ahs` covers at WIDTH=0.
The Vulkan ray-query tests (`tests/vulkan/rtquery*`) run the *query* on the RTU but
set `STRICT=0` because the lavapipe **AS-build** shaders fall back to llvmpipe — a
driver gap (`rtquery` fallback, master plan §M7), not an RTU gap.

---

## 8. Relationship to the true-GPU plan

`dEQP-VK.ray_query.*` — ray queries inside graphics/compute shaders — is a
north-star conformance gate. It depends on: (a) closing the FS-fusion slot overlap
(§2.1) and plumbing the resident AS pointer into the draw's FS arg block;
(b) ~~per-triangle AHS in the BVH walker~~ — **done** (§7): the CW-BVH walker now
classifies and yields any-hit like the flat walker; (c) AS + module residency
(stop the per-dispatch rebuild); and (d) fixing the `rtquery` llvmpipe fallback so
RT runs under `STRICT=1`. `dEQP-VK.ray_tracing_pipeline.*` (traceRays + SBT, recursion) is
a larger, separate track gated on the recursion / multi-warp-reformation tails.
These are tracked as **M7** in [`../proposals/gfx_v2_true_gpu.md`](../proposals/gfx_v2_true_gpu.md).

**Superseded / rejected directions** (recorded to avoid revival): RTU ISA v1
(funct3=5, retired for the v2 window ABI); SIMT-software traversal on the cores
(replaced by the FF RTU); a dedicated per-context stack SRAM instead of the
short-stack + context pool.

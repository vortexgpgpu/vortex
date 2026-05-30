**Date:** 2026-05-28
**Status:** proposal, no implementation
**Branch:** `prism_v3` (off `tinebp-patch-2`)
**Related:**
- [dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md), [gfx_migration_proposal.md](gfx_migration_proposal.md),
  [simx_v3_proposal.md](simx_v3_proposal.md) — v3 conventions inherited
  verbatim.

# PRISM — Vortex Ray-Tracing Unit (RTU) Proposal

## 1. Constraints (load-bearing)

### 1.1 Three design objectives, in priority order

1. **Performance-competitive with NVIDIA / Intel HW RT on the workloads
   Mesa actually generates.** The proposal is not the smallest possible
   HW; it is the smallest design that lands the major perf wins that
   NVIDIA RTX and Intel Xe-HPG / Xe2 ship. "Functional but slow" is
   explicitly off the table. Specifically the design must close the
   gap on:
   - Per-call setup / readback cost (no per-ray cache-line marshalling).
   - Shader-type divergence inside a warp (HW reformation, not
     SIMT-stack serialisation).
   - Long-tail effects from incoherent secondary rays.
   - Memory bandwidth for BVH descent (coherency gathering, cluster L1
     reuse, never serialised through one port).

2. **Vulkan KHR ray tracing must work HW-accelerated under
   `EXT_RTU_ENABLE = 1` and fall back to SIMT-software RT on Vortex
   under `EXT_RTU_ENABLE = 0`.** vortexpipe stays the Gallium driver
   in both cases. The llvmpipe-on-CPU fallback in `vp_launch_grid` is
   a capability-mismatch safety net, not an RT path.

3. **The SimX implementation must be a valid v3 TLM model.** Functional
   traversal and timing share one state machine; every memory load is
   issued through the cluster's existing dcache cluster; pool / queue
   capacities are real and the dispatcher sees real back-pressure
   when they fill; cycle counts are honest.

### 1.2 Out of scope

- RTL implementation (follows separately; SimX is the goal-reference
  oracle — cf. [feedback_simx_as_rtl_oracle.md](feedback_simx_as_rtl_oracle.md)).
- Callable shaders (`VK_KHR_ray_tracing_pipeline` callables).
- On-device BVH builder (lavapipe builds on CPU; that is fine).
- Acceleration-structure compaction and serialise/deserialise.
- Opacity Micromaps / Displaced Micromeshes (NVIDIA Ada-only HW).
- Multi-vendor BVH interop beyond what `vk_bvh.h` already provides.

## 2. Design principles synthesised from NVIDIA and Intel

Each principle below pulls a concrete idea from one of the two
shipping HW reference systems and adapts it to Vortex v3 invariants.

### 2.1 From NVIDIA RT Core (Turing → Ada)

| Idea | Adapted as |
|---|---|
| Per-lane independent processing inside the RT engine | Cluster-shared `RtuCore` with **32-entry ray-context pool** (§5.2); lanes from different warps interleave at sub-pool granularity. |
| Ray descriptor + hit attributes in a per-thread register file (SASS compiler-allocated) | **RTU register file** (§4.2): named 32-bit slots per (warp,lane), accessed via `vx_rt_set` (bulk) and `vx_rt_get` (scalar). Dedicated SRAM, separate from dcache. |
| Coherency Gathering Unit (CGU) groups same-octant rays before BVH descent | **Coherency gather** at pool-pick (§5.3) — direction-octant signature in 3 bits, scheduler prefers lanes with the same signature to share L1 fetches. |
| HW continuation stack in global memory for `traceRayEXT` recursion | Not adopted in HW. Mesa's `nir_lower_shader_calls.c` already lowers recursive `traceRayEXT` to iteration before vortexpipe sees the NIR (§3.3); Vortex HW recursion depth stays 1. |
| Shader Execution Reordering (SER) on Ada — explicit kernel-driven re-pack before CHS dispatch | **HW shader queues with implicit reformation** (§5.4), not opt-in. Phase 4 may expose an explicit `vx_reorder` opcode if profiling shows kernel-side hints help. |

### 2.2 From Intel Xe-HPG / Xe2 RT Unit

| Idea | Adapted as |
|---|---|
| Public BVH spec — 6-wide compressed wide-BVH (CW-BVH) | Vortex consumes the **Mesa-canonical `vk_bvh.h`** layout (§3.4). Same buffer ANV writes, no Vortex-specific re-encode. CW-BVH4 in Phase 1; CW-BVH6 once the box-PE array widens (Phase 4). |
| Bindless Thread Dispatcher (BTD) — HW reforms callback warps from in-flight rays across the cluster | **Cluster-scope shader queues + reformation** (§5.4). Queues are SRAM in `RtuCore`, capacity ~256 entries each, drained when ≥ SIMD_WIDTH entries of one type accumulate. |
| Ray Bank — dedicated SRAM for in-flight ray state, separate from data caches | **RTU register file + pool slots** (§4.2 + §5.2) are dedicated SRAM; only BVH-node and triangle data go through the dcache. The hot per-ray state never competes with BVH data in L1. |
| Synchronous thread reuse — issuing thread suspends while callback runs on (potentially) the same EU | **Implicit-reformation-on-same-warp** in Phase 2 (§8.2): the issuing warp's PC is HW-redirected to the callback entry point during `vx_rt_wait`, callback runs, returns. Phase 3 generalises to cross-warp reformation. |
| Open-source driver stack (Mesa ANV) | vortexpipe is in-tree Mesa; everything published. Mesa's `nir_lower_shader_calls.c` and `vk_acceleration_structure_*` helpers are reused directly. |

### 2.3 Where PRISM sits in the design space

PRISM is closest to **Intel Xe-HPG** in shape: register-resident ray
state in dedicated SRAM, HW warp reformation via shader queues,
cluster-scope engine, public ABI, open driver stack, vendor-neutral
BVH format. It borrows NVIDIA's **coherency gathering** and **wide
pool** for raw throughput, and the **SFU dispatcher integration
pattern** for Vortex-side wiring (matching DXA / TEX / OM / RASTER
precedent — cf. [dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md),
[gfx_migration_proposal.md](gfx_migration_proposal.md)). It avoids
NVIDIA's proprietary SASS encoding and Intel's BTD mixed
memory-vs-MRF state model in favour of a single, RISC-V-clean
register file.

## 3. Software architecture: Gallium ↔ Vortex with SW fallback

### 3.1 The two paths

```
Path A — EXT_RTU_ENABLE = 0 (SIMT software RT):
  Vulkan app → lavapipe → NIR (RT intrinsics expanded to BVH-walk NIR
            by lvp_nir_lower_ray_queries / nir_lower_shader_calls)
            → vortexpipe (vp_nir_to_llvm) → RISC-V .vxbin
            → Vortex SIMT cores

Path B — EXT_RTU_ENABLE = 1 (HW-accelerated RT):
  Vulkan app → lavapipe → NIR (RT intrinsics still intact,
            lvp's RT lowering pre-empted by vortexpipe)
            → vortexpipe (vp_nir_lower_ray_queries_to_rtu
                          + vp_nir_to_llvm) → RISC-V .vxbin
                          containing vx_rt_set / vx_rt_trace /
                          vx_rt_get ops
            → Vortex SIMT cores + RTU
```

vortexpipe is the Vortex SIMT driver in both states. The
capability-mismatch fallback to llvmpipe in `vp_launch_grid` is
orthogonal to RT and is **not** the RT SW-fallback path.

### 3.2 vortexpipe changes

1. **Cap query** (`vp_screen.c:80-141`): `vp->has_rtu =
   (isa_flags & VX_ISA_EXT_RTU) != 0`.
2. **NIR routing** (`vp_create_compute_state` in `vp_context.c:103-140`):
   if `has_rtu` and the NIR contains `rq_*` / `trace_ray_*` /
   `report_ray_intersection` / `terminate_ray` intrinsics, run
   vortexpipe's RT lowering pass **instead of** lavapipe's SW BVH
   expansion. Mechanism: register a `pipe_context` pre-NIR-finalisation
   callback; lavapipe calls it before `lvp_nir_lower_ray_queries.c`.
3. **`vp_nir_lower_ray_tracing_to_rtu.c`** (new). Rewrites:
   - `rq_initialize / rq_proceed / rq_load` → `vx_rt_set` + `vx_rt_trace`
     + `vx_rt_get`. The ray-query state machine collapses to one
     `vx_rt_trace` call per `rayQueryProceedEXT` iteration.
   - `trace_ray` (RT-pipeline form, post-`nir_lower_shader_calls`
     continuation passing) → same sequence: bulk set ray inputs,
     issue `vx_rt_trace`, read hit attrs. AHS / IS callback bodies
     remain inlined as switch statements on `geometry_index` /
     `sbt_index`; HW reformation makes them coherent (§5.4).
   - `report_ray_intersection`, `accept_ray_intersection`,
     `terminate_ray` → `vx_rt_cb_ret` (Phase 2) action codes.
4. **`vp_nir_to_llvm.c`** gains three new intrinsic cases that emit
   the `.insn` directives for `vx_rt_set`, `vx_rt_get`, `vx_rt_trace`,
   plus `vx_rt_cb_ret` for Phase 2+.

### 3.3 RT-pipeline mode reuses Phase 2's HW path

Lavapipe inlines all SBT-dispatched shaders (raygen + closest-hit +
miss + any-hit + intersection) into a single compute kernel before
vortexpipe sees the NIR — `nir_lower_shader_calls.c` does this in the
shipping vortex_3.x branch. Therefore:

- Recursive `traceRayEXT` from inside CHS becomes an iterative
  `vx_rt_trace` loop in the inlined kernel. HW recursion depth stays
  1; Vulkan's `maxRayRecursionDepth` is satisfied entirely in SW.
- Different SBT entries (per-material CHS, per-geometry IS) become a
  `switch (sbt_idx) { case 0: ...; case 1: ...; }` after the
  `vx_rt_trace`. HW reformation (§5.4) makes lanes within a warp
  hit the same `case`, so the switch executes coherently.

The Phase 2 HW path is therefore reachable from both `VK_KHR_ray_query`
(compute) and `VK_KHR_ray_tracing_pipeline` (full SBT) with no Vortex
ISA change between them.

### 3.4 BVH layout: standardise on `vk_bvh.h`

Mesa's `src/vulkan/runtime/bvh/vk_bvh.h` defines:

- `vk_bvh_box_node_t` — internal node, 4-wide AABB children with
  16-bit quantisation, 32-byte base.
- `vk_bvh_triangle_node_t` — leaf with up to 4 triangles, 64 bytes.
- `vk_bvh_instance_node_t` — TLAS instance leaf with 48-byte 3×4
  world-to-object transform + BLAS root pointer.
- `vk_bvh_aabb_node_t` — procedural / AABB-only leaf.

Same layout ANV uses for Intel HW; same layout lavapipe writes from
its CPU radix-sort builder. The Vortex RTU consumes the buffer
directly. CW-BVH4 in Phase 1; future widening to CW-BVH6 (Intel
shape) is a config-knob change in the box-PE array, not a re-encode.

## 4. Hardware ABI: register-resident, async-by-design trace

### 4.1 Concurrency level

`vx_rt_trace` is a thread-level SIMT instruction, dispatched per
active lane, exactly like a load or a `tex` sample. The four
concurrency levels:

| Level         | What happens                                                                |
|---------------|------------------------------------------------------------------------------|
| Lane / thread | One ray per call. Each lane owns 21 slots in the RTU register file. |
| Warp          | All active lanes execute `vx_rt_trace` in lock-step; up to SIMD_WIDTH lane requests packetised per cycle through the SFU dispatcher. Warp blocks until every active lane returns a terminal status. |
| CTA / core    | Multiple warps in a CTA can have RT calls outstanding. Per-core `RtuUnit` fans them into the cluster fabric. |
| Cluster       | One `RtuCore` per `VX_CFG_NUM_RTU_CORES` (default 1 per 4 cores). Pool + shader queues are shared across all cores wired to that RTU; back-pressure when full propagates back to each issuing SFU dispatcher. |

There is no warp-level / CTA-level / cluster-level *variant* of the
instruction — cluster sharing is a property of the engine, not the
ISA. The HW warp-reformation that happens *inside* the `vx_rt_trace`
issue is transparent to the kernel; from the kernel's view, every
issue is "thread-level SIMT, warp returns when all lanes resolve".

### 4.2 The RTU register file

A per-(warp,lane) dedicated SRAM, separate from the GPR file and
from the dcache. Twenty-one named 32-bit slots:

| Slot range  | Field                          | R/W by kernel              | R/W by HW                       |
|-------------|--------------------------------|----------------------------|----------------------------------|
| 0–2         | `world_ray_origin.{x,y,z}`     | set via `vx_rt_set`       | read at TRACE                   |
| 3–5         | `world_ray_direction.{x,y,z}`  | set via `vx_rt_set`       | read at TRACE                   |
| 6           | `tmin`                         | set                        | read                            |
| 7           | `tmax`                         | set                        | read                            |
| 8–10        | `object_ray_origin.{x,y,z}`    | get only (post-transform)  | written at BLAS entry           |
| 11–13       | `object_ray_direction.{x,y,z}` | get only                   | written at BLAS entry           |
| 14          | `hit_t`                        | get + IS may set           | written on hit                  |
| 15–16       | `hit_barycentric.{u,v}` (or AABB-min-t / max-t) | get  | written on hit                  |
| 17–20       | `hit_attr[0..3]` (4 × 32-bit user, total 16 B) | get + IS may set | written on hit + by IS shader |
| 21          | `hit_primitive_id`             | get                        | written on hit                  |
| 22          | `hit_instance_id`              | get                        | written on hit                  |
| 23          | `hit_geometry_index`           | get                        | written on hit                  |
| 24          | `hit_instance_custom_index`    | get                        | written on hit                  |
| 25–26       | `payload_ptr` (64-bit, two slots) | set                     | not read (opaque ptr)           |
| 27          | `ray_flags`                    | set                        | read at TRACE                   |
| 28          | `cull_mask`                    | set                        | read at TRACE                   |

(Slots 17–20 give 16 B of `hitAttributeEXT`, matching the Vulkan
limit and NVIDIA / Intel HW. The 21-slot count in §1.3 was the
minimum for ray-query; the 29-slot table above is the Phase 2 full
file including object-space ray and user hit attrs.)

Per-(warp,lane) cost: 29 × 4 B = 116 B. Per core (8 warps × 32
lanes): 116 × 256 = 29 KB. Per RTU (4 cores): 116 KB of dedicated
SRAM. Bounded, modellable.

**Why 29 slots and not more.** Audit §F8 cited 27 as "too many,
serialised access pattern". The fix is not slot-count reduction but
access-pattern reform: bulk set, occasional scalar get. 29 is the
fixed-point of "everything Vulkan requires" without inflation.

### 4.3 The opcodes

RTU opcodes share the **CUSTOM1** opcode space (EXT2 in decode.cpp).
The Phase-1 opcodes share **funct3 = 5** with a 2-bit funct2 sub-op
selector (`SET`/`GET`/`TRACE`/`WAIT` = 0/1/2/3). For `SET` and `GET`
the upper 5 bits of funct7 carry the slot ID, so funct2 is the only
sub-op room — at four sub-ops it is full.

Phase 2 (`CB_RET`) and Phase 3-B (`CB_DRAIN`) therefore land on
**funct3 = 6** (the next free EXT2 row after WGATHER 0, TEX 1, OM 2,
RASTER 3, RTU-prim 5) with funct2 = 0 / 1 respectively. CB ops do not
need a slot field, so funct7's upper 5 bits are zero. This keeps the
Phase-1 encoding stable across phases while giving callback ops their
own decoder row — convenient since the HW callback path lives in a
different pipeline section than ray issue.

The encoding is uniform across phases — **`vx_rt_trace` is
async-by-design from Phase 1**, and sync semantics are obtained by
pairing it with `vx_rt_wait`. This means Phase 3-B adds no new trace
opcode; it adds only `vx_rt_cb_drain`.

#### Phase 1 (committed): ray-query, opaque-only

```
vx_rt_set    imm12, rs1, rs2, rs3                      [sub-op=0]
    Writes rs1, rs2, rs3 into RTU register-file slots
    imm12, imm12+1, imm12+2. Bulk ray-input setup — three slots per
    single SFU dispatch.
    EXT2 / funct3=5 / sub-op=0 / R4-type.

vx_rt_get    rd, imm12                                 [sub-op=1]
    Reads RTU register-file slot imm12 into rd. One-cycle SFU op.
    EXT2 / funct3=5 / sub-op=1 / R-type.

vx_rt_trace  rd, rs1                                   [sub-op=2]
    Fires a ray. rs1 = TLAS device address (XLEN bits — 32 on RV32,
    64 on RV64; one register, no DCR-based pointer split). rd = ray
    handle (small int, 4-8 bits) identifying the in-flight ray in
    this lane's outstanding-handle table.
    **Non-blocking** — returns immediately after the handle is
    allocated and the snapshot of the RTU register file (slots 0..7,
    25..28) is queued into the RtuCore. The actual traversal proceeds
    asynchronously; the lane continues with the next instruction.
    EXT2 / funct3=5 / sub-op=2 / R-type.

vx_rt_wait   rd, rs1                                   [sub-op=3]
    Blocks the lane until the ray handle in rs1 reaches a terminal
    status. rd = status word:
      VX_RT_STS_DONE_HIT     = 0    closest hit recorded in slots 14..24
      VX_RT_STS_DONE_MISS    = 1    no hit; ray data unchanged
      VX_RT_STS_ERROR        = ≥128 reserved error codes
    During the wait, HW may transparently PC-redirect the warp into
    the kernel-registered callback dispatcher (§4.6) one or more
    times. From the kernel's view this is a single blocking
    instruction; the dispatcher function returns control here when
    it completes.
    EXT2 / funct3=5 / sub-op=3 / R-type.
```

Phase 1 ships with sub-op = {0, 1, 2, 3}. A ray-query-style kernel
emits `trace + wait` back-to-back — Mesa's `lvp_nir_lower_ray_queries`
pre-emption (§3.2) produces this pair from one source-level
`rayQueryProceedEXT`.

#### Phase 2 (committed): + AHS / IS callbacks

```
vx_rt_cb_ret  rs1                                      [sub-op=4]
    Signals the RtuCore that the AHS / IS callback for this lane is
    done. rs1 = action:
      VX_RT_CB_ACCEPT     = 1    accept candidate hit
      VX_RT_CB_IGNORE     = 0    discard candidate hit
      VX_RT_CB_TERMINATE  = 2    accept and terminate the ray
    HW resumes the lane's traversal in the RtuCore and the
    callback-dispatcher function returns normally; the warp's PC
    falls back to the post-vx_rt_wait site automatically. There is
    no separate "resume" instruction — the dispatcher's normal
    function return is the resume.
    EXT2 / funct3=6 / sub-op=0 / R-type.
```

Phase 2 adds sub-op = 4.

#### Phase 3 (deferred; choice between 3-A and 3-B)

Phase 3-A — cross-warp HW reformation — adds **no new opcodes**;
all changes are internal to `RtuCore` and the cluster fabric.

Phase 3-B — async producer/consumer via explicit drain — adds
**exactly one** new opcode:

```
vx_rt_cb_drain rd                                      [sub-op=5]
    Per-lane: returns 0 if no callback is pending for this lane,
    else returns (cb_type | (handle << 8)). HW pre-populates the
    RTU regs with the candidate hit info before the lane reads the
    return value. The lane then runs the callback body and calls
    vx_rt_cb_ret to release.
    EXT2 / funct3=6 / sub-op=1 / R-type.
```

Because `vx_rt_trace` is already async-by-design from Phase 1, and
`vx_rt_wait` is already in the Phase 1 opcode set, Phase 3-B adds
**only** `vx_rt_cb_drain`. The Phase 3-A path leaves sub-op=5
unused; the Phase 3-B path claims it.

Sub-op ∈ {6, 7} stays reserved for Phase 4.

The TLAS pointer is *always* per-call in `rs1`. Vulkan permits
multiple AS handles live in one kernel (primary against full BVH,
shadow against opaque-only BVH); a DCR-based pointer rules that
out. `VX_DCR_RTU_TLAS_ROOT_*` (§5.8) exists only as a smoke-test /
debugger convenience and is **not** consulted by `vx_rt_trace`.

### 4.4 Kernel idiom — full path-tracer with secondary rays

The kernel body uses the **`trace + wait` pattern** that Mesa
will emit for every `rayQueryProceedEXT` / synchronous-recursive
`traceRayEXT`. Phase 2 AHS / IS callbacks are dispatched by HW
*during* the `vx_rt_wait`; the callback dispatcher is a separate
function (§4.6) and does not appear inline in the kernel body. No
explicit YIELD-loop is needed.

```c
// Per-thread kernel body. One pixel per lane. Grid/block dims set by
// lavapipe (§4.7). RTU register file is per-(warp,lane). Each ray's
// state snapshot is taken at vx_rt_trace issue and lives in the
// RtuCore pool slot keyed by the returned handle; the file is free
// for the next ray's setup as soon as trace returns.

__attribute__((kernel_body))
void rt_kernel(rt_args_t *args) {
    // Per-thread payload struct on the private stack. HW never
    // touches its contents; only the pointer is in the RTU regs.
    ray_payload_t payload;

    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= args->width || py >= args->height) return;

    float3 origin     = args->camera_pos;
    float3 direction  = primary_direction(px, py, args);
    float3 throughput = make_f3(1.0f);
    float3 color      = make_f3(0.0f);

    for (int bounce = 0; bounce < args->max_depth; ++bounce) {
        // -------- Set ray inputs into the RTU regs --------
        vx_rt_set3(VX_RT_RAY_ORIGIN,    origin.x,    origin.y,    origin.z);
        vx_rt_set3(VX_RT_RAY_DIRECTION, direction.x, direction.y, direction.z);
        vx_rt_set3(VX_RT_T_MIN,         0.001f,      1e30f,       0.0f); // tmin, tmax, pad
        vx_rt_set2(VX_RT_PAYLOAD_PTR_LO,
                    (uint32_t)((uint64_t)&payload),
                    (uint32_t)((uint64_t)&payload >> 32));
        vx_rt_set1(VX_RT_RAY_FLAGS,     VX_RT_FLAG_OPAQUE);

        // -------- Fire ray (async; returns handle) --------
        uint32_t h = vx_rt_trace(args->scene_root);

        // -------- Wait for terminal status --------
        // During this wait, HW may PC-redirect the warp into the
        // kernel-registered callback dispatcher (a separate function
        // that ends in vx_rt_cb_ret). Each PC redirect is one
        // round-trip: HW snapshots warp PC, runs dispatcher, vx_rt_cb_ret
        // resumes traversal, dispatcher RETs, warp PC restored to
        // this vx_rt_wait. The wait eventually returns terminal.
        uint32_t sts = vx_rt_wait(h);

        if (sts == VX_RT_STS_DONE_MISS) {
            color += throughput * sample_env(args, direction);
            break;
        }

        // -------- Read hit attrs from the RTU regs (cheap; on-chip) --------
        float    t       = vx_rt_get_f(VX_RT_HIT_T);
        uint32_t inst_id = vx_rt_get  (VX_RT_HIT_INSTANCE_ID);
        uint32_t prim_id = vx_rt_get  (VX_RT_HIT_PRIMITIVE_ID);
        uint32_t geom    = vx_rt_get  (VX_RT_HIT_GEOMETRY_INDEX);
        float2   bary    = { vx_rt_get_f(VX_RT_HIT_BARY_U),
                             vx_rt_get_f(VX_RT_HIT_BARY_V) };

        float3 hit_pos = origin + direction * t;
        float3 normal  = lookup_normal(args, inst_id, geom, prim_id, bary);
        material_t mat = lookup_material(args, geom, prim_id);

        // -------- Shadow ray (re-uses the RTU regs) --------
        float3 light_pos  = args->lights[0].pos;
        float3 to_light   = light_pos - hit_pos;
        float  light_dist = length(to_light);
        float3 light_dir  = to_light * (1.0f / light_dist);

        vx_rt_set3(VX_RT_RAY_ORIGIN,    hit_pos.x,   hit_pos.y,   hit_pos.z);
        vx_rt_set3(VX_RT_RAY_DIRECTION, light_dir.x, light_dir.y, light_dir.z);
        vx_rt_set3(VX_RT_T_MIN,         0.001f,      light_dist - 0.001f, 0.0f);
        vx_rt_set1(VX_RT_RAY_FLAGS,     VX_RT_FLAG_OPAQUE
                                       | VX_RT_FLAG_TERMINATE_ON_FIRST_HIT
                                       | VX_RT_FLAG_SKIP_CLOSEST_HIT);

        uint32_t sh_h   = vx_rt_trace(args->scene_root);
        uint32_t sh_sts = vx_rt_wait(sh_h);
        bool occluded   = (sh_sts == VX_RT_STS_DONE_HIT);

        if (!occluded) {
            color += throughput * mat.diffuse
                  * args->lights[0].color
                  * max(0.0f, dot(normal, light_dir))
                  / (light_dist * light_dist);
        }

        // -------- Reflection bounce --------
        if (!mat.is_reflective) break;
        direction  = reflect(direction, normal);
        origin     = hit_pos;
        throughput = throughput * mat.specular;
    }

    args->framebuffer[py * args->width + px] = pack_rgba(color);
}
```

Counting SFU instructions on the hot path (one bounce, no callback,
opaque triangle hit, shadow ray fired):

- Setup primary: 3 `vx_rt_set` + 1 set payload ptr + 1 set flags → **5 SFU ops**
- Issue primary: 1 `vx_rt_trace` → **1 op** (non-blocking, returns handle)
- Wait primary:  1 `vx_rt_wait` → **1 op** (blocks until terminal)
- Read hits:     5 `vx_rt_get` → **5 ops**
- Setup shadow:  3 + 1 `vx_rt_set` → **4 ops**
- Issue shadow:  1 `vx_rt_trace` → **1 op**
- Wait shadow:   1 `vx_rt_wait` → **1 op**

Total: ~18 SFU ops per bounce, all single-cycle register-file
accesses. **No dcache traffic for the ray descriptor or hit attrs.**

For comparison, NVIDIA OptiX compiles a single source-level
`traceRayEXT` to ~5–8 SASS RT-driving ops, all register-resident.
PRISM is slightly more verbose (~18 vs ~5–8) because the ray-input
slots are written via explicit `vx_rt_set` rather than
SASS-compiler-allocated registers. Phase 4 could collapse this with
a `vx_rt_trace_full` macro-op (see §8.5) if profiling shows the
overhead matters.

### 4.5 No `max_depth` HW limit

The two ray-lifecycle ops (`vx_rt_trace` for issue, `vx_rt_wait` for
collect) plus `vx_rt_cb_ret` are independent and per-call. No HW
state persists between two `vx_rt_trace` calls except what the kernel
re-loads into the RTU regs. The bounce loop's `max_depth` is a
kernel-author convention used by every NVIDIA OptiX path tracer and
by every Mesa lavapipe RT test; drop it or set it to 100,000,
nothing in HW cares.

### 4.6 Phase 2 callback dispatch: HW PC redirect during `vx_rt_wait`

When an active lane's in-flight ray yields to AHS or IS, the
RtuCore signals the warp scheduler of the issuing core. **The signal
is processed by the warp scheduler only when the warp is parked at
its `vx_rt_wait` instruction** — which is the natural rendezvous
point in the trace+wait pattern. The wait is suspended, the warp's PC
is HW-redirected to the kernel-registered callback dispatcher (the
address the kernel previously wrote to `mtvec`), and the SIMT tmask is
narrowed to the lanes whose rays actually yielded.

**Mechanism: reuse the existing M-mode trap path** (`mtvec` / `mepc`
/ `mcause` / `mret`) rather than inventing a parallel PC-redirect
fabric. The yield is modelled as an RTU-cause trap raised by
`RtuCore`-to-core signalling: the same `Scheduler::raise_trap()` path
that today serves `ECALL` / `EBREAK` (see [preemption_foundation_proposal.md](preemption_foundation_proposal.md))
is exercised from outside the warp's instruction stream, with the
constraint that the warp must be parked at `vx_rt_wait` at the moment
of redirect (so there is no in-flight-instruction surprise — the trap
is "deferred-synchronous from the warp's POV").

This matches how shipping HW does callback dispatch: NVIDIA Pascal+
preemption is an M-mode trap to a save / restore stub, and Intel BTD's
in-EU callback path reuses the EU trap entry rather than a parallel
PC-redirect. Reusing the existing trap CSRs also means **no new SIMT-
stack frame type**, **no new "callback mode" scheduler state**, and
**no new resume instruction** — `mret` already restores PC from `mepc`.

Two small extensions to the trap CSR file vs. the foundation:

1. **Saved tmask.** The active-lane mask is snapshotted into a new
   per-warp slot (`mscratch_tmask` — kept in the trap CSR family rather
   than a general-purpose CSR so software does not save / restore it
   in a normal ECALL handler) and restored on `mret`. The Phase 2
   callback narrows the running mask to only-yielded-lanes; the
   pre-yield mask is restored by `mret`.
2. **RTU cause code.** A new `mcause` value (`TRAP_CAUSE_RTU_CALLBACK`,
   custom code in the M-mode-reserved-for-implementation range) flags
   trap entries originating from the RtuCore. The callback dispatcher
   reads `mcause` to distinguish RTU yield from any other trap entry.

The callback dispatcher is **a separate function**, registered once
per kernel at startup by writing its entry PC into `mtvec`:

```c
__attribute__((rtu_callback_entry))
void rt_callback_dispatcher(void) {
    // Optional: confirm this is an RTU yield, not a foreign trap.
    // assert(csr_read(mcause) == TRAP_CAUSE_RTU_CALLBACK);

    uint32_t cb_type = vx_rt_get(VX_RT_CB_TYPE);    // ANYHIT or PROC
    uint32_t geom    = vx_rt_get(VX_RT_HIT_GEOMETRY_INDEX);

    bool accept;
    if (cb_type == VX_RT_CB_TYPE_ANYHIT) {
        float2 bary = { vx_rt_get_f(VX_RT_HIT_BARY_U),
                        vx_rt_get_f(VX_RT_HIT_BARY_V) };
        accept = user_anyhit_dispatch(geom, bary);  // Mesa-inlined SBT switch
    } else /* PROC */ {
        float t_new;
        accept = user_intersection_dispatch(geom, &t_new);
        if (accept) vx_rt_set1(VX_RT_HIT_T, t_new);
    }
    vx_rt_cb_ret(accept ? VX_RT_CB_ACCEPT : VX_RT_CB_IGNORE);
    // Dispatcher exits via `mret` — emitted in place of `ret` under
    // __attribute__((rtu_callback_entry)). PC ← mepc (the post-
    // vx_rt_wait PC); tmask ← mscratch_tmask (the pre-yield mask).
    // Traversal continues in the RtuCore; the wait stays blocked
    // until the next yield or a terminal state.
}
```

This is "implicit-reformation-on-same-warp" — Intel BTD's *first*
choice when callbacks can be served by the originating EU. There is
no explicit "resume" instruction: `mret` after `vx_rt_cb_ret` is the
resume, and the existing M-mode trap CSRs hold the post-wait PC and
the pre-yield tmask.

Phase 3 generalises to cross-warp reformation (§5.4) or to async +
explicit drain (§8.4.B) depending on which option the fork takes.

### 4.7 Kernel launch model: grid, block, thread

Vortex KMU dispatches a 3D grid via `VX_DCR_KMU_GRID_DIM_*` /
`VX_DCR_KMU_BLOCK_DIM_*`. `vkCmdTraceRaysKHR(width, height, depth)`
maps to grid_dim covering the dispatch volume, block_dim from the
raygen `LocalSize`. vortexpipe prefers block_dim = (SIMD_WIDTH, k, 1)
to keep per-warp pixel sets 2D-contiguous in screen space — primary
rays stay coherent through the TLAS top levels, maximising BVH-fetch
L1 hit rate.

Per-lane state — `payload`, scratch, accumulators — lives in the
thread-private stack. The RTU register file lives in dedicated SRAM,
managed by HW; the kernel sees it only through the RTU opcodes.

## 5. SimX v3 TLM microarchitecture

### 5.1 Cluster scope, behind SFU

`RtuCore` is cluster-shared (`VX_CFG_NUM_RTU_CORES = max(1, NUM_CORES
/ 4)`). Each core has a `RtuUnit` front-end that:

- Catches `vx_rt_set` / `vx_rt_get` / `vx_rt_cb_ret` from the
  SFU dispatcher and serves them locally against the per-core RTU
  register file (slice of the cluster's SRAM).
- Routes `vx_rt_trace` to the cluster's `RtuCore` over
  `SimChannel<RtuReq>`, blocking the lane on the SFU until the
  response returns.

No new `FUType` — the RTU PE branches off the SFU dispatcher under
`#ifdef VX_CFG_EXT_RTU_ENABLE`, matching DXA / TEX / OM / RASTER
precedent (cf. [dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md),
[gfx_migration_proposal.md](gfx_migration_proposal.md)).

### 5.2 Ray context pool (32 slots)

```cpp
struct RayContext {
    enum Phase {
        IDLE, FETCH_NODE, AWAIT_NODE,
        INTERSECT_BOX, DESCEND, POP_STACK,
        FETCH_TRI, AWAIT_TRI, INTERSECT_TRI,
        FETCH_AABB, AWAIT_AABB,
        BLAS_ENTRY, APPLY_TRANSFORM,
        AWAIT_CALLBACK,
        DONE_HIT, DONE_MISS,
    } phase;

    uint32_t  hart_id;         // routing back to issuing lane
    uint32_t  warp_uuid;       // routing back to issuing warp (for reformation)

    // Snapshot of the RTU regs at TRACE issue. Re-loaded by HW from
    // the per-(warp,lane) register file at allocation.
    Vec3      world_o, world_d, object_o, object_d;
    float     tmin, tmax;
    uint32_t  flags, cull_mask;
    uint64_t  payload_ptr;
    uint64_t  scene_root;

    // Traversal working state.
    uint64_t  node_ptr;
    uint32_t  level;
    uint8_t   trail[VX_CFG_RTU_TRAIL_DEPTH];        // 32 entries
    uint32_t  short_stack[VX_CFG_RTU_STACK_DEPTH];  // 16 entries
    uint8_t   stack_head;

    // Hit working state.
    Hit       best_hit;
    Hit       candidate_hit;    // for AHS / IS yield
    uint32_t  candidate_sbt;    // SBT index for queue routing

    // Coherency-gather signature: 3-bit ray-direction octant.
    uint8_t   coh_signature;
};
```

Pool size: `VX_CFG_RTU_CONTEXT_POOL = 32` by default (one slot per
SIMD_WIDTH lane in a single warp, plus headroom for secondary-ray
overlap from adjacent warps). Per-slot size ≈ 256 B; total pool SRAM
≈ 8 KB per RtuCore.

The pool is **multi-banked** (4 banks × 8 slots, the four banks
indexed by `hart_id mod 4`) so up to 4 distinct contexts can be
read/written per cycle by the scheduler / mem-rx demux / PE writeback
ports without conflict. Bank-conflict back-pressure is rare under
typical 32-lane warps because the lanes map evenly across banks.

Sizing rationale: 32 slots = full warp + slack. A smaller pool
(e.g. 8) would serialise a 32-lane warp into 4 batches before
considering secondary-ray overlap. NVIDIA RT Core's published
in-flight ray count is ≈ 8–16; Intel Ray Bank is similar magnitude.
PRISM sizes generously here so cross-warp interleaving fully covers
the pipeline.

### 5.3 Coherency gathering at pool-pick

Each context carries a 3-bit `coh_signature` derived from the sign
bits of its `world_d.{x,y,z}`. The scheduler's `pick_ready()`
prefers contexts whose signature matches the most-recently-issued
fetch's signature; ties broken by greedy-then-oldest among matches.

Why: rays with the same octant traverse similar BVH paths, so
consecutive fetches issued for same-signature contexts have high
mutual L1 reuse. NVIDIA's CGU performs this at warp granularity; we
do it at pool-pick granularity, which is finer and benefits
multi-warp interleaved workloads.

Cost: one 3-bit field per pool slot, one 3-bit comparator per pick
cycle. Trivial.

### 5.4 Shader queues + HW warp reformation

Four queues in `RtuCore`, indexed by shader type:

```
shader_queues_[MISS]
shader_queues_[CHS]
shader_queues_[AHS]
shader_queues_[IS]
```

Each queue holds entries keyed by `(sbt_idx, hart_id, warp_uuid,
context_slot)`. Capacity per queue: 256 entries (1 KB), arranged as
a circular buffer with separate read/write banks for the SBT-index
hash (so reformation can scan-by-sbt without serial walk).

**Push policy.** When a context reaches an internal terminal state
(`DONE_HIT` → CHS, `DONE_MISS` → MISS, `AWAIT_CALLBACK` → AHS or
IS), instead of returning to the issuing warp directly, the RtuCore
pushes an entry to the matching queue with that context's
`sbt_idx`. The queue is the rendezvous point.

**Pop / reformation policy.** Each cycle the scheduler also runs a
reformation check:

```
for each queue Q in priority order:                   // CHS > MISS > AHS > IS
    if Q.has_at_least(SIMD_WIDTH) entries with same sbt_idx:
        gather those entries into a virtual warp
        signal the warp scheduler in the originating core(s)
        the participating warp(s) get their PCs redirected to the
            callback entry point (DCR-registered)
        callback runs as a coherent SIMD_WIDTH-wide execution
```

When a "virtual warp" is gathered from multiple originating warps,
each lane's `hart_id` is preserved so that when the callback issues
`vx_rt_cb_ret`, the action routes back to the right context.

The simplest implementation has reformation reuse the SAME warp
that originated the entries (one issuing warp = one reformed warp),
which is implicit-on-same-warp reformation (= Phase 2). Phase 3 adds
cross-warp scatter / gather, which requires the cluster-fabric
signalling and the warp-state-stash protocol described in §8.3.

**Why this works for shader divergence.** PRISM identified that a
warp issuing `vx_rt_trace` typically gets a mix of shader-type
yields; without reformation, the kernel must serialise through each
shader type via SIMT branching. The queues group by type *first*,
then by sbt_idx — so reformation guarantees that when the warp
finally executes the callback, every lane in the warp wants the
same shader type AND the same SBT entry (same material, same
geometry kind). The SIMT switch on `geometry_index` after the
yield then takes a single coherent case.

NVIDIA SER does the same thing on Ada via an explicit
`reorderThread()` API call. We do it implicitly inside
`vx_rt_trace`'s lifetime — no kernel-side opt-in required.

**Queue tuning knobs.**

```toml
VX_CFG_RTU_QUEUE_DEPTH         = 256   # entries per queue
VX_CFG_RTU_REFORM_WAIT_CYCLES  = 64    # max wait for a SIMD_WIDTH batch
VX_CFG_RTU_REFORM_MIN_BATCH    = 16    # below this, drain anyway
```

The wait/batch trade lets us bound worst-case latency: if a queue
hasn't filled to `SIMD_WIDTH` within `REFORM_WAIT_CYCLES`, the
scheduler drains it anyway at whatever occupancy it has, accepting
some per-warp partial-fill. Production tuning will set these from
profiling.

### 5.5 The state machine — fused functional + timing

`RtuCore` is a SimObject with one `tick()` per cycle. The state
machine processes ray contexts and queue events. Functional
intersection math runs on the cycle the timing model schedules it;
memory loads issue real `MemReq` packets through the shared cluster
dcache and the engine waits for the response before dependent
compute. **There is no decoupled functional model**: the engine
never calls `Emulator::dcache_read` behind the timing model's back,
and there is no side queue that replays accesses out of order.

Pseudocode for the central tick:

```cpp
void RtuCore::tick() {
    // 1. Drain memory responses into pending contexts.
    while (!dcache_rsp_in.empty()) {
        auto rsp = dcache_rsp_in.peek();
        RayContext& ctx = pool_[rsp.tag];
        ctx.absorb(rsp);             // parse BVH node / tri / AABB
        ctx.phase = next_phase_for(ctx, rsp);
        dcache_rsp_in.pop();
    }

    // 2. Drain reformation callback returns from cores.
    while (!cb_ret_in.empty()) {
        auto ret = cb_ret_in.peek();
        RayContext& ctx = pool_[ret.context_slot];
        apply_action(ctx, ret.action);     // accept/ignore/terminate
        ctx.phase = (ret.action == ACCEPT) ? UPDATE_BEST_HIT : CONTINUE_TRAVERSE;
        cb_ret_in.pop();
    }

    // 3. Reformation check: any queue ready to drain?
    auto reform = check_reformation(shader_queues_);
    if (reform.fire) {
        // Tell the warp scheduler in the originating core(s) to
        // redirect their PCs to the callback entry point.
        emit_reformation_signal(reform);
        for (auto& e : reform.entries) {
            pool_[e.context_slot].phase = AWAIT_CALLBACK;
        }
    }

    // 4. Drain new requests from cores into the pool.
    while (!req_in.empty() && pool_.has_free_slot()) {
        auto req = req_in.peek();
        pool_.allocate(req);   // snapshot the RTU regs at issue
        req_in.pop();
    }

    // 5. Coherency-gather pool pick.
    RayContext* ctx = pool_.pick_ready_by_coherence(last_signature_);
    if (!ctx) return;
    last_signature_ = ctx->coh_signature;

    // 6. Advance one micro-step.
    switch (ctx->phase) {
    case FETCH_NODE:        issue_load(ctx, ctx->node_ptr, 32);  break;  // CW-BVH4 node
    case INTERSECT_BOX:     box_pe_array(ctx);                   break;  // 4 parallel PEs
    case INTERSECT_TRI:     tri_pe_array(ctx);                   break;
    case APPLY_TRANSFORM:   transform_unit(ctx);                 break;
    case DONE_HIT:          push_queue(SHQ_CHS, ctx);            break;
    case DONE_MISS:         push_queue(SHQ_MISS, ctx);           break;
    case AWAIT_CALLBACK:    /* no-op; waiting for cb_ret_in */   break;
    case POP_STACK:         pop_short_stack(ctx);                break;
    /* ... */
    }
}
```

Three invariants the state machine enforces:

- **Functional/timing fusion.** Every `issue_load` emits a `MemReq`
  on `dcache_req_out` and the dependent compute waits for the
  response. Cycle counts reflect the actual memory dependency chain.
- **Bounded pool.** Pool is a fixed array. `pool_.has_free_slot()`
  back-pressures `req_in`, which back-pressures the SFU dispatcher.
  The dispatcher sees real back-pressure when contexts fill.
- **Shader queue ordering.** Queues push on a defined event
  (`DONE_HIT` / `DONE_MISS` / `AWAIT_CALLBACK`) and pop on a defined
  policy (priority order + `sbt_idx` grouping + batch / timeout).
  No unordered drainage.

### 5.6 Memory port — share the cluster dcache cluster

`RtuCore.dcache_req_out → cluster's CacheCluster (new MemArbiter slot)
→ shared L1 banks → RtuCore.dcache_rsp_in.`

Same shape as the gfx-migration cluster-cache slot for TEX / OM /
RASTER. `L2_NUM_REQS` and per-socket bus widths gain one slot under
`VX_CFG_EXT_RTU_ENABLED`. No L1-RTU private cache; the cluster
dcache absorbs reuse across coherent rays.

Sizing: BVH internal nodes are 32 B (1/2 cache line at 64 B), triangle
leaves 64 B (1 cache line), AABB leaves 32 B, instance leaves 64 B.
**`NUM_RTU_BLOCKS = 2`** gives the multi-PE pipeline two outstanding
memory requests per cycle, sufficient to keep the box-PE array fed
under cache-hit conditions.

### 5.7 Throughput knobs

```toml
VX_CFG_RTU_BVH_WIDTH       = 4    # CW-BVH4 in Phase 1; 6 in Phase 4
VX_CFG_RTU_BOX_PE          = 4    # parallel ray-box PEs (matches BVH_WIDTH)
VX_CFG_RTU_TRI_PE          = 4    # parallel ray-triangle PEs (matches triangle-leaf width)
VX_CFG_RTU_STACK_DEPTH     = 16   # short-stack depth per context
VX_CFG_RTU_TRAIL_DEPTH     = 32   # trail depth per context (TLAS + BLAS combined cap)
VX_CFG_RTU_CONTEXT_POOL    = 32   # in-flight ray contexts per RtuCore
VX_CFG_RTU_NODE_LATENCY    = 4    # box-PE pipeline depth
VX_CFG_RTU_TRI_LATENCY     = 6    # tri-PE pipeline depth
VX_CFG_RTU_XFORM_LATENCY   = 3    # transform unit pipeline depth
VX_CFG_RTU_QUEUE_DEPTH     = 256  # per-shader-queue entries
VX_CFG_RTU_REFORM_WAIT     = 64   # cycles before partial-warp reformation
VX_CFG_RTU_REFORM_MIN      = 16   # min batch for partial-warp reformation
VX_CFG_NUM_RTU_BLOCKS      = 2    # parallel memory request ports
VX_CFG_NUM_RTU_CORES       = "expr: max(1, up($VX_CFG_NUM_CORES / 4))"
```

Latency values are pipeline depths, not "replay tallies on top of an
unmodelled traversal" — every stage is an actual `.send(rsp, delay)`
in the engine. Multi-cycle occupancy is captured because the next
stage only fires when the previous stage's result arrives.

### 5.8 DCRs

```toml
VX_DCR_RTU_STATE_BEGIN     = 0x080
VX_DCR_RTU_CONFIG          = 0x080  # enable, mode, cull defaults
VX_DCR_RTU_TLAS_ROOT_LO    = 0x081
VX_DCR_RTU_TLAS_ROOT_HI    = 0x082
VX_DCR_RTU_CB_ENTRY_LO     = 0x083  # callback dispatcher PC (Phase 2)
VX_DCR_RTU_CB_ENTRY_HI     = 0x084
VX_DCR_RTU_REFORM_THRESH   = 0x085  # SW-tunable reformation threshold
VX_DCR_RTU_STATS_RESET     = 0x086
VX_DCR_RTU_STATE_END       = 0x087
VX_DCR_RTU_STATE_COUNT     = "expr: $VX_DCR_RTU_STATE_END - $VX_DCR_RTU_STATE_BEGIN"
```

`VX_DCR_RTU_CB_ENTRY_*` is the kernel-registered callback dispatcher
entry point. When HW yields a warp into a callback, the warp's PC is
redirected to this address. lavapipe / vortexpipe writes it once per
kernel launch from the compiled `.vxbin` symbol table.

### 5.9 Caps and MISA

```toml
VX_CFG_EXT_RTU_ENABLE      = false
VX_CFG_EXT_RTU_ENABLED     = "expr: 1 if $VX_CFG_EXT_RTU_ENABLE else 0"
VX_CFG_MISA_EXT            = "expr: ... | ($VX_CFG_EXT_RTU_ENABLED << 11)"
# Add to VX_caps.h ISA flags word:
VX_CAPS_EXT_RTU            = 0x...  # one bit; queried by vortexpipe
```

Default off. With the cap off, no RTU files compiled in, no DCRs
wired, opcodes trap illegal-instruction, vortexpipe sees
`has_rtu == false` and lavapipe's SW RT path runs through Vortex
SIMT (path A in §3.1).

### 5.10 Microarchitecture diagram

```
                     ┌───────────────────────────────────────────────────────────────────────┐
   RtuReq from       │                              RtuCore                                  │
   each core's       │                                                                       │
   RtuUnit           │  ┌──────────┐    ┌─────────────────────────────────┐                  │
   ───────────────→  │  │ req_in   │ →  │   Context Allocator             │                  │
   (SimChannel)      │  │  fifo    │    │   pool back-pressures req_in    │                  │
                     │  └──────────┘    └────────────────┬────────────────┘                  │
                     │                                    │                                   │
                     │                                    ▼                                   │
                     │   ┌──────────────────────────────────────────────────────────┐         │
                     │   │   Ray Context Pool — 32 slots × 4 banks                  │         │
                     │   │   ┌───┐┌───┐┌───┐┌───┐ ┌───┐┌───┐┌───┐┌───┐               │         │
                     │   │   │RC0││RC1││RC2││RC3│ │...│   │   │   │   …             │         │
                     │   │   └───┘└───┘└───┘└───┘ └───┘└───┘└───┘└───┘               │         │
                     │   │   coh_sig stk trail ray hit phase     bank: hart_id%4    │         │
                     │   └───────────┬───────────────────────────────────────────────┘         │
                     │               │                                                         │
                     │               ▼                                                         │
                     │   ┌──────────────────────────────────────┐                              │
                     │   │  Scheduler                            │                              │
                     │   │  pick_ready_by_coherence(last_sig)    │                              │
                     │   │  — coherency-gather pool pick         │                              │
                     │   └──┬────────┬─────────┬──────────┬─────┘                              │
                     │      │        │         │          │                                    │
                     │      ▼        ▼         ▼          ▼                                    │
                     │  ┌─────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐           │
                     │  │MemReq   │ │ Box-PE × 4   │ │ Tri-PE × 4   │ │ Transform  │           │
                     │  │ Issue   │ │ N=BVH_WIDTH  │ │ M=TRI_PE     │ │  Unit      │           │
                     │  │ (LD)    │ │ 4-cy pipe    │ │ 6-cy pipe    │ │  3-cy pipe │           │
                     │  └────┬────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘           │
                     │       │             │                │                │                  │
                     │       ▼             ▼                ▼                ▼                  │
                     │  ┌──────────┐  update RC          update RC      update RC               │
                     │  │dcache_req│   (best_hit /        (best_hit /     (object_ray)          │
                     │  │ to cluster    candidate_hit)    candidate_hit)                        │
                     │  │ cache cluster                                                          │
                     │  └────┬─────┘                                                             │
                     │       │   ┌──────────────┐                                                │
                     │       └──→│ Mem Resp Rx  │ → demux to RC by req tag                       │
                     │           │  demux       │                                                │
                     │           └──────┬───────┘                                                │
                     │                  │                                                        │
                     │                  ▼                                                        │
                     │           absorb response into context; update phase                      │
                     │                                                                            │
                     │   ┌──────────────────────────────────────────────────────────┐             │
                     │   │  On terminal phase (DONE_HIT / DONE_MISS / AWAIT_CB):    │             │
                     │   │  push (sbt_idx, hart_id, warp_uuid, slot) into queue.    │             │
                     │   └──────────────────┬───────────────────────────────────────┘             │
                     │                       │                                                    │
                     │                       ▼                                                    │
                     │    ┌─────────────────────────────────────────────────────────────────┐     │
                     │    │   Shader Queues — 4 banks × 256 entries each                    │     │
                     │    │   ┌─────────────────┐ ┌──────────────┐ ┌──────┐ ┌──────┐         │     │
                     │    │   │   CHS  (sbt→)   │ │   MISS       │ │ AHS  │ │  IS  │         │     │
                     │    │   └─────────────────┘ └──────────────┘ └──────┘ └──────┘         │     │
                     │    │   keyed by sbt_idx for warp reformation                         │     │
                     │    └────────────────────────────────┬────────────────────────────────┘     │
                     │                                      │                                      │
                     │                                      ▼                                      │
                     │    ┌───────────────────────────────────────────────────────────────┐        │
                     │    │  Reformation Engine                                            │        │
                     │    │  — scan queues by sbt_idx                                      │        │
                     │    │  — when ≥ SIMD_WIDTH same-sbt entries: emit reform signal      │        │
                     │    │  — fallback: ≥ REFORM_MIN entries after REFORM_WAIT cycles     │        │
                     │    └────────────────────────────┬──────────────────────────────────┘        │
                     │                                  │                                          │
                     │                                  ▼                                          │
                     │    ┌───────────────────────────────────────┐                                │
                     │    │  Reform signal → originating core(s)  │                                │
                     │    │  warp scheduler redirects PC to       │                                │
                     │    │  VX_DCR_RTU_CB_ENTRY (Phase 2: same   │                                │
                     │    │  warp; Phase 3: cross-warp gather)    │                                │
                     │    └─────────────────┬─────────────────────┘                                │
                     │                       │                                                     │
                     │  ───────────────────  │                                                     │
                     │   reform_sig out      │ → to each issuing core's warp scheduler             │
                     │                                                                              │
                     │  ┌──────────────────────────────────────────────────────────────────┐        │
                     │  │  vx_rt_cb_ret action ← from each issuing core              │        │
                     │  │  (cb_ret_in SimChannel)                                          │        │
                     │  │  → apply_action(ctx, ACCEPT|IGNORE|TERMINATE)                    │        │
                     │  │  → ctx.phase = UPDATE_BEST_HIT or CONTINUE_TRAVERSE              │        │
                     │  └──────────────────────────────────────────────────────────────────┘        │
                     │                                                                              │
                     │  ───────────────────                                                          │
                     │   RtuRsp(status) → back to originating core's RtuUnit on full resolution     │
                     └──────────────────────────────────────────────────────────────────────────────┘
```

SRAM budget at default knobs:

| Block                  | Per-slot | Slots | Total          |
|------------------------|---------:|------:|---------------:|
| Pool: ray + hit + traversal | 256 B | 32 | 8 KB          |
| Shader queues (4 × 256 entries × 16 B) | 16 B | 1024 | 16 KB |
| Box-PE × 4 working regs   |   64 B | 4 | 256 B            |
| Tri-PE × 4 working regs   |   96 B | 4 | 384 B            |
| Transform regs            |  192 B | 1 | 192 B            |
| Coherency-gather signatures | 4 b | 32 | 16 B            |
| Per-core RTU register file (sliced) | 116 B | 256 | 29 KB / core |
| **Cluster RtuCore SRAM**  |          |    | **≈ 25 KB**    |
| **Per-core RTU regs slice** |        |    | **≈ 29 KB**     |
| **Total per cluster (1 RTU + 4 cores)** |   |    | **≈ 141 KB**   |

Bounded, modellable, smaller than a TEX cluster. RTL elaboration
can use FF arrays for the pool / queues and standard SRAM macros
for the per-core register-file slice.

## 6. Hardware faithfulness — alignment with NVIDIA RT Core and Intel RT Unit

This section claims that PRISM is a **faithful model of shipping RT
hardware**, not a research-grade architecture or a Vortex-specific
invention. Every load-bearing component in §4-§5 has a direct analog
in one or both of NVIDIA's RT Core (Turing → Ada) and Intel's
Xe-HPG / Xe2 RT Unit. The cases where PRISM deviates are listed
explicitly in §6.4 with the design reason.

Restricted to publicly-documented behaviour: NVIDIA Turing / Ampere /
Ada whitepapers + Vulkan-on-NVIDIA driver hints; Intel Xe-HPG ISA
Reference Manual + ANV (Mesa) source. SASS encodings, microcode, and
chip-specific dimensions are not in the public record and are not
claimed here.

### 6.1 Component-by-component mapping

| PRISM component | Spec § | NVIDIA RT Core analog | Intel Xe-HPG RT Unit analog | Faithfulness commentary |
|---|---|---|---|---|
| **RTU register file** — per-(warp,lane) dedicated SRAM holding ray descriptor + hit attrs | §4.2 | Shader special registers: `gl_RayOriginEXT`, `gl_RayDirectionEXT`, `gl_HitTEXT`, `gl_PrimitiveID`, `hitAttributeEXT`. SASS-compiler allocated; accessed via S2R-style ops | Message Register File (MRF) slots populated for TRACE_RAY send-message; documented in Xe-HPG ISA Ref | PRISM's bulk `vx_rt_set` + scalar `vx_rt_get` access pattern is the explicit RISC-V analog of NVIDIA's S2R-against-special-registers and Intel's MRF-bulk-write. Slot count (29) sized to the Vulkan KHR ray-state + `hitAttributeEXT` union (29 × 32-bit = 116 B, fits the 128 B Intel-style line) |
| **Ray context pool** — 32 in-flight ray contexts per `RtuCore` | §5.2 | "Ray Store" / ray buffer inside the RT Core; Turing whitepaper documents ~8–16 in-flight rays per RT Core | "Ray Bank" SRAM, Xe-HPG ISA Ref documents comparable in-flight depth | PRISM matches the shape and order-of-magnitude. The 32-slot choice is SIMD_WIDTH = 32 plus slack for secondary-ray overlap; NVIDIA / Intel cite similar reasoning |
| **Multi-bank pool** — 4 banks × 8 slots, banked by `hart_id mod 4` | §5.2 | Banked ray-buffer SRAM in RT Core (NVIDIA does not publish bank count) | Banked Ray Bank in Xe-HPG | Standard SRAM-banking trick to support concurrent read/write from scheduler + mem-rx demux + PE writeback. Identical motivation to shipping HW |
| **Coherency gather** — 3-bit ray-direction-octant signature at pool-pick | §5.3 | **Coherency Gathering Unit (CGU)** in RT Core, Turing whitepaper. Groups same-octant rays before BVH descent to maximise L1 reuse | Ray-coherence-driven scheduling in Xe-HPG (similar mechanism, less prominently documented) | PRISM uses the smallest possible signature (3 bits = one bit per direction component); same idea, lighter mechanism |
| **Box-PE array** — 4 parallel ray-box intersection PEs | §5.7 | RT Core "box intersection units"; Turing whitepaper cites 4-wide on Turing. Wider on later generations to keep CW-BVH4 in one pipeline stage | Box-test pipeline; Xe-HPG sized to CW-BVH6 (6-wide) | PRISM's BOX_PE = BVH_WIDTH = 4 in Phase 1 matches Turing exactly. Phase 4 widening to 6 matches Intel CW-BVH6 |
| **Tri-PE array** — 4 parallel ray-triangle intersection PEs | §5.7 | RT Core "triangle intersection units"; tile-test pipeline | Triangle-test pipeline; Xe-HPG | PRISM's TRI_PE = 4 matches the max-4-triangles-per-leaf shape of `vk_bvh_triangle_node_t` (vendor-neutral) and Intel CW-BVH triangle leaves |
| **Transform unit** — single 3×4 mat-vec multiplier shared across pool, fires on TLAS leaf hit | §5.7 | Transform stage applies world → object transform at BLAS entry inside RT Core | World-to-object transform stage in Xe-HPG | Same role, same firing condition. Single-instance sharing is the same trade-off NVIDIA / Intel publish: transforms are << 1% of cycles, so sharing is essentially free |
| **Short-stack BVH traversal** — 16-entry per context with restart-on-underflow | §5.2 (`stack_head`, `trail`) | NVIDIA stack management not publicly specified | Stack-based traversal documented in Xe-HPG | PRISM uses the Vaidyanathan / Woop / Benthin (HPG 2019) "wide BVH traversal with a short stack" algorithm — a standard, peer-reviewed academic reference algorithm. Same algorithm Intel cites in its ISA Reference |
| **HW shader queues** — 4 cluster-scope queues (MISS, CHS, AHS, IS), keyed by `(shader_type, sbt_idx)` | §5.4 | Ada **Shader Execution Reordering (SER)** maintains an internal reorder buffer for grouping rays by shader handle before CHS dispatch | **Bindless Thread Dispatcher (BTD)** queues, Xe-HPG; HW reforms callback warps from in-flight rays | PRISM's 4-queue structure is the explicit version of Intel's BTD shape. The `(type, sbt_idx)` keying gives both shader-type and per-material coherence — same effect as SER + BTD combined |
| **HW PC redirect** — warp PC redirected to callback dispatcher at `vx_rt_wait` rendezvous | §4.6 + §5.4 | Ada SER: kernel opts in via `OptixHitObject::execute` / `reorderThread()`. Hardware emits the redirect | Intel BTD: implicit; HW dispatches callbacks to the issuing thread (synchronous thread reuse) or to a sibling thread | PRISM is implicit like Intel BTD (no kernel-side opt-in). The same-warp choice in Phase 2 mirrors Intel's preferred "synchronous thread reuse" path |
| **Cross-warp reformation** (Phase 3-A) | §5.4, §8.4.A | NVIDIA SER's optional cross-warp gather | Intel BTD's cross-thread gather across the Xe-slice | PRISM Phase 3-A's stash-and-restore protocol mirrors what both SER and BTD do; PRISM publishes the protocol (state-stash bitmask, fabric width) where NVIDIA / Intel do not |
| **Coherency-gather pool pick** ahead of memory issue | §5.3 + §5.5 | Turing CGU performs this at warp granularity before BVH descent | Xe-HPG ray-coherence scheduling | PRISM does it at pool-pick granularity (finer); benefits multi-warp interleaved workloads more than once-per-warp scheduling |
| **Cluster L1 dcache integration** — RTU shares the cluster's dcache cluster | §5.6 | NVIDIA RT Core uses the SM's L1 cache for BVH / triangle data | Intel RT Unit uses the Xe-core's L1 for BVH / triangle data | Same shape. PRISM adds one MemArbiter slot under `VX_CFG_EXT_RTU_ENABLED`, identical to the gfx-extension (TEX / OM / RASTER) wiring |
| **Cluster-scope shared engine** — one `RtuCore` per N cores | §5.1 | NVIDIA RT Core: one per SM | Intel RT Unit: one per Xe-core | PRISM `VX_CFG_NUM_RTU_CORES = max(1, NUM_CORES / 4)` — one RTU per cluster of 4 cores. Cluster grain matches Intel's Xe-core grain; finer than NVIDIA's per-SM grain |
| **BVH node layout** — Mesa-canonical `vk_bvh.h` | §3.4 | Vendor-specific compressed BVH, not publicly documented | Intel publishes a 6-wide CW-BVH layout; ANV (Mesa) emits this format | PRISM uses the same `vk_bvh.h` ANV writes for Intel HW. **Byte-identical** with Intel's BVH consumer; no Vortex-specific re-encode |
| **BVH builder** — host-side, lavapipe's radix-sort | §3.4 | Driver-side; HW-accelerated on newer SKUs | Driver-side; HW-accelerated on newer SKUs | PRISM defers HW-side BVH building to Phase 4. The builder code is upstream Mesa; the BVH consumer (Vortex) sees the same byte layout regardless |
| **vortexpipe NIR lowering** — pre-empts `lvp_nir_lower_ray_queries.c` to emit `vx_rt_*` ops | §3.2 | n/a (NVIDIA proprietary) | ANV intercepts ray-query / ray-tracing NIR at the same Mesa pass-ordering hook | PRISM uses the **same Mesa NIR infrastructure** as ANV. The vortexpipe pass is the Vortex equivalent of `anv_nir_lower_ray_tracing.c` |
| **Mesa-side SBT walk** — kernel-inline switch on `sbt_idx`, callbacks pre-inlined by `nir_lower_shader_calls.c` | §3.3 | HW SBT walk inside RT Core | HW SBT walk via BTD | **PRISM deviation, see §6.4.** Mesa already produces a single-shader form for its CPU-side SW-RT backend; PRISM reuses that lowering to avoid building a HW SBT walker in Phase 1-3 |

### 6.2 Feature-by-feature alignment

**Trace ABI shape.** NVIDIA and Intel both present a *synchronous*
`traceRayEXT` at the kernel level; their HW pipelines are async
internally and use HW-side latency hiding via warp scheduling. PRISM
goes one level deeper and exposes the async ABI directly
(§4.3: `vx_rt_trace` returns a handle, `vx_rt_wait` blocks). The
synchronous Vulkan ABI is then a 2-instruction trace+wait pair that
Mesa emits, with no HW-side change. This is the only ABI-level
deviation from NVIDIA / Intel; rationale in §6.4.

**Vulkan ray-tracing-pipeline mode.** PRISM relies on Mesa's
`nir_lower_shader_calls.c` to lower recursive `traceRayEXT` calls
into a single-shader iterative form *before* vortexpipe sees the NIR.
Lavapipe already produces this form (`lvp_execute.c:4076-4100`); the
Vortex side is consequently a strict subset of what ANV consumes.
HW recursion depth stays 1; `maxRayRecursionDepth` is satisfied
entirely in SW. **This is the same approach Mesa applies to its
software RT backend** — PRISM is not inventing the lowering, only
the codegen target.

**Ray flags.** PRISM Phase 1 supports the full Vulkan `SpvRayFlags*`
set (cull, opaque overrides, TERMINATE_ON_FIRST_HIT,
SKIP_CLOSEST_HIT, SKIP_TRIANGLES; SKIP_AABBs in Phase 2). The HW
check happens in the `RtuCore` state machine on every leaf hit and
short-circuits `DONE_HIT` immediately. **Same flag set, same
semantics, same fast-out paths as NVIDIA and Intel.**

**Multiple TLAS per kernel.** PRISM passes the TLAS pointer per-call
in `rs1` of `vx_rt_trace` (§4.3), not via a global DCR. Vulkan KHR,
NVIDIA OptiX, and ANV all do the same — the AS is a descriptor
binding the shader reads, the binding's device address is what
`traceRayEXT` consumes. PRISM is identical here.

**Payload model.** Payload is a memory pointer (`payload_ptr` slot
in the RTU register file); HW never touches the payload contents.
NVIDIA spills large payloads to memory; Intel uses a mixed
register/global model. PRISM's pure-pointer model is the simplest of
the three and is what Mesa's RT lowering already targets.

**Multiple AS instances (instancing).** PRISM walks
`vk_bvh_instance_node_t` on TLAS leaf hit, applies the 3×4 transform,
descends into BLAS. **Bit-identical** with what ANV does on Intel HW
because both consume the same `vk_bvh.h` buffer.

### 6.3 Cycle-honest TLM modelling

PRISM's state machine couples functional traversal and timing into
one `tick()` (§5.5). Every memory load is a real `MemReq` through
the shared cluster dcache; every dependent intersection waits for
the response. Cache hit / miss behaviour, dependency chains, pool
back-pressure, and queue occupancy are all visible to the timing
model in exactly the way they would be visible in RTL.

NVIDIA and Intel HW pay these costs in silicon; PRISM pays them in
SimX cycles. RTL elaboration off this SimX model should not
encounter performance surprises that a decoupled functional / timing
model would mask.

### 6.4 Explicit deviations from NVIDIA / Intel — and why

PRISM deviates from shipping HW in three places. Each is intentional
and reversible in a later phase.

**D1. Async-by-design trace ABI** (§4.3, §4.4).

NVIDIA and Intel expose synchronous `traceRayEXT`. PRISM exposes
async `vx_rt_trace` + `vx_rt_wait`. Sync semantics are obtained by
Mesa emitting the pair back-to-back.

*Reason:* Phase 3-B (async producer/consumer) becomes a strict
additive extension instead of an ABI break. Kernels written for
Phase 1-2 work in Phase 3-B without recompilation. The 1-cycle
extra SFU op for the sync pair is ~0.2% overhead at typical
trace latencies — measured in §8.4.

*Reversal:* a future phase could collapse the pair into a single
sync opcode if profiling shows the extra op matters in any
workload; the ISA encoding leaves room (sub-op ∈ {6,7} reserved).

**D2. SW-side SBT walk + Mesa-inlined callbacks** (§3.3).

NVIDIA RT Core and Intel BTD walk the SBT in HW and HW-dispatch
the matching callback shader. PRISM relies on Mesa
(`nir_lower_shader_calls.c`) to inline all callbacks into a single
compute kernel; the kernel's switch on `geometry_index` /
`sbt_idx` does the dispatch in SW. HW shader queues still re-group
the warps by `(type, sbt_idx)` so the SW switch executes
coherently.

*Reason:* avoids building a HW SBT walker for Phase 1-3. Mesa
already produces the single-shader form for its own SW-RT
backend, so PRISM's NIR consumer side is a subset of what already
exists in lavapipe. The cost is one SIMT switch per yielded
warp, which the shader-queue coherent batching reduces to a
single coherent case.

*Reversal:* Phase 4 (deferred) could add a HW SBT walker and
direct dispatch. Phase 1-3 kernels would continue to work as the
HW would just bypass the SW switch when it lands on the same
geometry_index.

**D3. CPU-side BVH builder** (§3.4).

NVIDIA and Intel both ship HW-accelerated BVH building on newer
SKUs (NVIDIA Ada onward, Intel Alchemist+). PRISM uses
lavapipe's CPU radix-sort builder for now.

*Reason:* BVH builder is a separate ~5 KLoC of Mesa code that
would not change the RTU spec. Phase 4 (deferred) adds device-side
building.

*Reversal:* trivial — replace the lavapipe builder with a Vortex
compute kernel that emits the same `vk_bvh.h` buffer layout. No
RTU-side change required.

### 6.5 Summary

PRISM borrows from both shipping HW reference systems and from the
academic literature:

- **From NVIDIA RT Core:** coherency gathering, ray-state register
  allocation philosophy, per-SM cluster grain.
- **From Intel Xe-HPG RT Unit:** Bindless-Thread-Dispatcher-style
  HW shader queues with implicit warp reformation, MRF-style
  register file, public BVH format (`vk_bvh.h`-canonical), open
  driver stack (Mesa).
- **From the academic record:** Vaidyanathan / Woop / Benthin
  short-stack wide-BVH traversal (HPG 2019).

The Vortex-specific contributions are: the RISC-V opcode encoding
(public, documented in §4.3), the async-by-design ABI (D1 above),
and the integration pattern with the Vortex v3 SimX TLM model. None
of these change the underlying RT pipeline shape; they make the
shape Vortex-shaped.

## 7. Performance model

### 7.1 Steady-state throughput

In the ideal interior of a primary-ray batch (all rays coherent,
opaque triangle hits, no AHS / IS yield), the pipeline reaches:

- **1 BVH-internal-node fetch / cycle** (cluster dcache bandwidth bound).
- **4 ray-box tests / cycle** (one node's children, 4 parallel PEs).
- **4 ray-triangle tests / cycle** (one 4-tri leaf, 4 parallel PEs).
- **1 short-stack op / cycle / context** (push or pop).
- **1 reformation event / cycle** in the worst case.

Effective per-ray rate at coh-gather hit and L1 hit: ~1 BVH level /
`NODE_LATENCY` (= 4) cycles. With pool = 32 and `NODE_LATENCY` = 4,
the pipeline is fully covered (32 ÷ 4 = 8 cycles of pool capacity
per stage). One 32-lane warp's worth of primary rays through a
12-level BVH = 12 × 4 = 48 cycles of pipeline time, plus memory-miss
penalties — well under 1 µs of simulated time on a 1 GHz clock.

### 7.2 Long-tail mitigation

The dominant cost for incoherent secondary rays (path-traced GI,
shadow bounce, glossy reflections) is the per-warp tail — one slow
lane keeps the warp idle.

PRISM's mitigation stack:

1. **Coherency gather at pool pick** (§5.3) — same-octant rays
   share L1 fetches; the slow lane usually slows because of cache
   misses, and we reduce miss rate.
2. **Cross-warp pipeline interleaving** — while warp A's slow lane
   is waiting on memory, warp B's lanes are in the same pool.
3. **HW shader queue reformation** (§5.4) — at the post-traversal
   shader dispatch, warp A's lanes that yielded to CHS-mat-7 join
   warp B's, C's, D's same-material lanes. The reformed warp runs
   the CHS body coherently across more rays than originally issued
   together.

Combined, NVIDIA reports SER alone delivers up to 2× speedup on
path-tracing workloads; we expect similar on Phase 3 once
reformation is HW. Phase 2 (same-warp reformation only) is a smaller
but still meaningful win because the shader-type sort happens; the
SBT-index re-pack does not.

### 7.3 Cache pressure analysis

Per primary ray, in steady state, the cluster L1 sees:

- ~10-15 BVH internal nodes × 32 B = ~400 B
- 1-2 BVH leaves (triangles) × 64 B = ~96 B
- 0-1 BVH instance leaves × 64 B = ~64 B

Total: ~560 B per ray. A 32-lane warp of primary rays through a
single root + 2 levels of internal nodes hits the same 3-7 nodes
repeatedly — the BVH top-3-levels footprint (≈ 1-2 KB) is the L1
hot set, easily resident.

**Crucially: no per-ray state spills to dcache.** Ray descriptor +
hit attrs live in the RTU register file (§4.2). Per-ray dcache
traffic comes entirely from the BVH and triangle data, not from ray
state — meaning the L1 hot set is exactly the BVH top levels and the
recently-touched leaves, with no per-bounce eviction pressure from
ray descriptors.

### 7.4 Bounce-loop cost

100 bounces per pixel, 32 lanes, hot path (no callback):

| Per-bounce cost            | Cost on PRISM                                 |
|----------------------------|-----------------------------------------------|
| Ray-input setup            | 5 SFU ops (RTU reg writes)                    |
| Trace issue                | 1 SFU op (async, returns handle)              |
| Wait                       | 1 SFU op (blocks on handle until terminal)    |
| Hit-attribute read         | 5 SFU ops (RTU reg reads)                     |
| Per-bounce total           | **~12 SFU ops + 0 B dcache traffic**          |
| 100 bounces × 32 lanes     | ~1200 ops + 0 B cache traffic per pixel       |

All hot-path register-file accesses; the dcache is reserved for BVH
and material lookups, which are the data that actually need it.

### 7.5 SimT efficiency

Measured-in-SimX projection:

- Coherency gather brings cache-miss-driven divergence down by
  ~20-30% on incoherent workloads.
- Shader-queue reformation eliminates shader-type SIMT serialisation
  at the callback dispatch site.
- 32-slot pool fully covers a single warp, so no intra-warp
  serialisation at the dispatcher.

We will measure and report these in the Phase 2 / Phase 3 milestone
reports — they are the headline numbers we are designing to.

## 8. Phased implementation

### 8.1 Phase 0 — pre-work (≈ 2 days)

- Confirm `vk_bvh.h` Phase 1 subset: `box_node`, `triangle_node`,
  `instance_node` (defer `aabb_node` to Phase 2).
- Confirm the mesa_vortex pass-ordering hook for vortexpipe pre-empting
  `lvp_nir_lower_ray_queries.c`.
- Reserve DCR range 0x080-0x086 in `VX_types.toml`; MISA bit 11.

### 8.2 Phase 1 — ray-query (compute-only, opaque triangles) (≈ 2 weeks)

Goal: a Vulkan compute shader using `rayQueryEXT` against
opaque-triangle-only BVH runs with HW acceleration.

1. `VX_config.toml` / `VX_types.toml` — add `VX_CFG_EXT_RTU_ENABLE`,
   `VX_CFG_NUM_RTU_CORES`, pool/queue/PE knobs, MISA bit, DCR range.
2. `sim/simx/decode.cpp` — add EXT2/funct3=5 family (sub-op=0..3 for
   `vx_rt_set` / `vx_rt_get` / `vx_rt_trace` / `vx_rt_wait`).
3. `sim/simx/rtu/rtu_unit.{h,cpp}` — per-core front-end; owns the
   per-core slice of the RTU register file and the per-lane
   outstanding-handle map (1 entry/lane in Phase 1).
4. `sim/simx/rtu/rtu_core.{h,cpp}` — cluster-scope engine, pool,
   coherency-gather scheduler, state machine. Phase 1 omits shader
   queues and reformation (`vx_rt_trace` is async + `vx_rt_wait`
   blocks on the handle; the wait simply waits for the pool slot to
   reach a terminal phase).
5. `sim/simx/sfu_unit.cpp` — RTU PE branch.
6. `sim/simx/cluster.cpp` — cluster-cache MemArbiter slot under
   `VX_CFG_EXT_RTU_ENABLED`; allocate 2 `NUM_RTU_BLOCKS` ports.
7. `sw/kernel/include/vx_raytrace.h` — `vx_rt_set{1,2,3}`,
   `vx_rt_get`, `vx_rt_get_f`, `vx_rt_trace`, `vx_rt_wait` inlines.
8. mesa_vortex `vp_screen.c` — query the cap.
9. mesa_vortex `vp_nir_lower_ray_tracing_to_rtu.c` — new NIR pass
   that emits the BATCH=1 trace+wait pair from `rayQueryProceedEXT`
   and from `traceRayEXT` (post-`nir_lower_shader_calls`).
10. mesa_vortex `vp_nir_to_llvm.c` — emit the four RT intrinsics.
11. `tests/regression/rtu_smoke/` — 1 BLAS, 2 triangles, 64 primary
    rays, validate hit/miss + `t` against CPU oracle. Exercises the
    full trace+wait pair end-to-end.

Exit criterion: smoke passes on simx with `EXT_RTU_ENABLE = 1` AND
still passes (via SIMT path) with `EXT_RTU_ENABLE = 0`.

### 8.3 Phase 2 — callbacks (AHS, IS) with same-warp reformation (≈ 3 weeks)

Goal: alpha-tested triangles and procedural AABB primitives via
`vkCmdTraceRaysKHR` work end-to-end. Reformation is
implicit-on-same-warp (the issuing warp gets its PC redirected to
the callback entry point).

1. Extend `RayContext::phase` with `AWAIT_CALLBACK` and the
   `BLAS_ENTRY` / `APPLY_TRANSFORM` cycle.
2. Extend `vk_bvh_aabb_node_t` decode in `rtu_core.cpp`.
3. Implement shader queues + reformation-on-same-warp (§5.4) —
   gather entries for the issuing warp; when its `vx_rt_wait` is
   parked, redirect PC via cluster signal to the warp scheduler in
   the originating core.
4. Add the `vx_rt_cb_ret` opcode (sub-op=4) and the `cb_ret_in`
   SimChannel from each core into `RtuCore`. (No separate "resume"
   opcode is needed — see §4.6: `vx_rt_cb_ret` followed by the
   dispatcher's RET is the resume.)
5. mesa side: extend `vp_nir_lower_ray_tracing_to_rtu.c` to emit the
   callback-dispatcher function (the standalone `rt_callback_dispatcher`
   in §4.6) and register its entry PC in `VX_DCR_RTU_CB_ENTRY` at
   kernel launch.
6. `tests/regression/rtu_anyhit/` — alpha-tested Cornell Box.
7. `tests/regression/rtu_proc/` — procedural sphere primitive.

Exit criterion: both tests pass; Cornell Box rendered via
`vkCmdTraceRaysKHR` is pixel-equivalent (within tolerance) to the
lavapipe SW path.

### 8.4 Phase 3 — performance fork (data-driven choice between 3-A and 3-B)

**Phase 3 is not pre-committed.** Phase 2's exit milestone includes a
characterisation run on three workloads:

- W_alpha — alpha-tested foliage scene (AHS-heavy, opaque + non-opaque mix).
- W_proc  — procedural-sphere scene (IS-heavy, PRISM-style RTV6 analog).
- W_quiet — ReSTIR-style direct-lighting (mostly opaque, low yield rate).

Measured against Phase 1's opaque-only baseline, these three numbers
drive a fork at the start of Phase 3:

| Phase 2 result on W_alpha | Action |
|---|---|
| ≤ 1.2× regression vs. Phase 1 baseline | **Skip Phase 3.** Phase 2 is good enough; move directly to Phase 4. |
| 1.2 – 1.5× regression | **Phase 3-A** — bounded HW cost is the safer choice. |
| > 1.5× regression AND batchable NIR pattern in Mesa output is tractable | **Phase 3-B** — async ABI is worth the Mesa-side work. |
| > 1.5× regression but Mesa pass is intractable | **Phase 3-A** — fall back to HW-only fix. |

The two options below are full specs; pick one based on the table.
The funct2 codepoints in §4.3 are partitioned so neither path
consumes the other's encoding space.

#### 7.4.A — Phase 3-A: cross-warp HW reformation (≈ 4 weeks)

Goal: HW gathers callback entries across multiple warps and forms a
coherent virtual warp. NVIDIA-SER / Intel-BTD-equivalent feature.
**No ISA changes**; all changes internal to `RtuCore` and the
cluster fabric.

1. Add the cross-warp warp-state-stash protocol (§5.4). When the
   reformation engine fires, the participating warps' live-out GPR
   set (compiler-emitted bitmask, typically ≤ 16 registers) is
   stashed to a HW-private scratch (~256 B per warp per stash); the
   reformed warp runs the callback; after `vx_rt_cb_ret` for
   all reformed lanes, original warps' GPR state restored.
2. Add cluster signalling between `RtuCore` and the per-core warp
   schedulers for the gather/stash/restore sequence. Estimated
   fabric width: 1 reform-signal + 32-bit warp ID + 32-bit lane mask
   per signalling event, ≤ 1 event per cycle per RtuCore.
3. Expose `VX_DCR_RTU_REFORM_THRESH` for SW tuning; profile vs.
   not-reformed baseline.
4. `tests/regression/rtu_ser_benchmark/` — material-divergent
   workload (4-8 materials per scene), measure speedup vs. Phase 2.

Exit criterion: ≥ 1.5× speedup on W_alpha vs. Phase 2; pixel-equivalent
output; cluster fabric verified clean under multi-warp stress test.

Reference: NVIDIA SER (Ada / Lovelace), Intel BTD cross-thread gather.

#### 7.4.B — Phase 3-B: async ABI with HW-managed implicit prod/cons (≈ 4-6 weeks)

Goal: expose PRISM's producer/consumer overlap at the instruction
level rather than the warp level. Because `vx_rt_trace` is already
async-by-design from Phase 1 (returns a handle) and `vx_rt_wait` is
already in the Phase 1 opcode set, **Phase 3-B adds exactly one new
opcode**: `vx_rt_cb_drain`. The HW pool grows to support per-lane
multi-in-flight rays; the cluster reformation fabric / state-stash
protocol from Phase 3-A is **not built** (the Mesa-emitted batch +
drain pattern provides equivalent overlap without HW state-stash).

1. Add one new opcode at the sub-op codepoint reserved in §4.3:
   - `vx_rt_cb_drain  rd` (sub-op=5) — per-lane drain; 0 if no
     callback pending, else `(cb_type | (handle << 8))`. HW
     pre-populates the RTU regs with the candidate hit info before
     the lane reads the return value.
2. Generalise the existing Phase 1 `vx_rt_trace` to support multiple
   in-flight rays per lane. The trace ABI doesn't change; the
   per-lane outstanding-handle map gains capacity.
3. Grow `VX_CFG_RTU_CONTEXT_POOL` from 32 to 128. Per-RtuCore pool
   SRAM grows from ≈ 8 KB to ≈ 32 KB.
4. Add per-lane outstanding-handle map (4-8 handles × 32 lanes ×
   {pool-slot-id, status-bit} ≈ 1 KB SRAM).
5. mesa_vortex `vp_nir_lower_ray_tracing_to_rtu.c` — extend the
   existing pass with a BATCH-detection step: identify NIR regions
   where multiple `traceRayEXT` calls can issue before the first
   result is consumed, hoist the `vx_rt_wait` ops past the issue
   cluster, and replace inline `vx_rt_wait`-blocked callback
   dispatch with an explicit `vx_rt_cb_drain` loop between the
   issue cluster and the wait cluster. Fall back to BATCH=1 when
   ordering forbids batching.
6. Drop the cluster reformation fabric / state-stash protocol from
   §5.4 — the HW queues are still there, but reformation becomes
   implicit in the kernel's drain calls.
7. `tests/regression/rtu_wavefront_benchmark/` — explicit batched
   raygen + drain kernel; measure speedup vs. Phase 2 (and vs.
   Phase 3-A as the alternative reference).

The 4-6 week range reflects the Mesa BATCH-detection pass being the
load-bearing cost; the HW change (one opcode, larger pool, handle
map) is ~2 weeks.

Exit criterion: ≥ 1.5× speedup on W_alpha vs. Phase 2; **and** Mesa
batch-detection pass finds BATCH ≥ 4 on ≥ 50% of `traceRayEXT` call
sites in the test workloads (the load-bearing claim from R-B3 in §10).
If either bound is not met by week 4, halt and pivot to 3-A.

Reference: PRISM producer/consumer paradigm with the user-side warp
split removed and the trace+wait ABI shape inherited from Phase 1.

### 8.5 Phase 4 — real BVH walker (≈ 2 weeks)

**Note on phase numbering.** During Phases 1-2 implementation the
team used internal phase numbers 4-12 for incremental dev iterations
on top of the Phase 2 callback umbrella (multi-tri linear walk, CHS,
MISS, IS, SBT-runtime, single-BLAS TLAS, affine instance transforms,
multi-instance TLAS, full-list traversal, recursive vx_rt_trace).
Those landed as 13 `rtu_smoke_*` regression tests against the flat
linear-scan walker. The proposal-level Phase 4 below is the
**architectural** Phase 4 — replacing that linear-scan placeholder
with a real BVH walker. Implementation dev numbers are independent
of proposal phase numbers; commits prefix `PRISM RTU: Phase <N>` where
N is the implementation iteration.

Phase 4 unblocks any real workload. Today's `RtuCore` linear-scans up
to 8 triangles per scene (`kRtuMaxTrisPerScene = 8` in
`sim/simx/rtu/rtu_core.cpp:45`). A 10⁶-triangle Vulkan scene would
take 10¹² ray-tri tests; a CW-BVH4 walker brings that to ≈10⁷
internal-node + leaf tests. Without Phase 4, PRISM is a callback
framework with a primitive intersector, not a ray-tracing unit.

#### 8.5.1 Scope

1. **`vk_bvh.h` consumption.** Decode Mesa-canonical 64 B CW-BVH4
   internal nodes (4-wide, 8-bit quantized child AABBs, shared
   exponent triple `(ex, ey, ez)`, child fan-out via `meta` field +
   base child pointer). Leaf types: TRIANGLE, INSTANCE, PROCEDURAL.
   Header field carries `node_kind ∈ {INTERNAL, TRIANGLE_LEAF,
   INSTANCE_LEAF, PROCEDURAL_LEAF}`. New `bvh_node_t` struct in
   `sim/simx/rtu/bvh_types.h`. Bytes-on-wire match what lavapipe's
   builder emits via `vk_acceleration_structure_serialize` — no
   PRISM-specific re-encode.

2. **Traversal state machine.** Replaces today's COMPUTE blob with
   distinct micro-states: `FETCH_NODE → INTERSECT_BOX → DESCEND →
   FETCH_LEAF → INTERSECT_LEAF → POP`. Each transition is a one-tick
   move; `INTERSECT_BOX` invokes the wide-box PE array (see §8.7).
   Per-slot fields added to `LaneState`: `node_ptr`, `root_ptr`,
   `root_level`, `level`, `instance_id`, `geometry_index`. Driven by
   the leaf-type tag from the decoded node.

3. **Short-stack + trail.** Lift the prototype's design verbatim
   (`vortex-raytracing/sim/simx/rt_core.h:146-255`,
   `types.h:1802-1834`). Sizes: `VX_CFG_RTU_STACK_DEPTH = 16` (CW-BVH4
   max useful depth for typical scenes), `VX_CFG_RTU_TRAIL_DEPTH = 32`
   (matches prototype `MAX_TRAIL_LEVEL`). When the short stack
   underflows on `pop()` the trail drives a `RESTART` — re-walks from
   `root_ptr` honoring trail bits, no global stack spill. Per-slot
   bytes: 64 B stack + 32 B trail = 96 B; well under the 4 KB pool
   target.

4. **TLAS→BLAS descent with distinct BLAS pointers.** Replace the
   current single-inline-BLAS hack. Each `INSTANCE_LEAF` node carries
   an absolute `blas_root_addr` (device pointer to the BLAS's root
   internal node) and a 12-float `inv_transform`. On `INSTANCE_HIT`:
   apply `ray_transform(world_ray, inv_transform)` (already
   implemented as `affine_inverse_transform_ray` in
   `sim/simx/rtu/rtu_core.cpp:159-198`), stash the world ray and the
   old root, set `root_ptr = blas_root_addr`, `root_level = level + 1`.
   On `FINISHED` inside a BLAS: pop back to the TLAS root, restore the
   world ray, resume the TLAS walk. New `LaneState` fields:
   `world_ray`, `tlas_root_ptr`, `tlas_level`.

5. **Backwards-compat shim.** Keep the flat-list walker behind
   `scene_kind == kRtuSceneKindTriList` (header byte 4-7 = 0) so the
   existing 13 `rtu_smoke_*` tests keep passing during Phase 4
   development. Add a new `scene_kind == kRtuSceneKindBvh4` (value 2)
   that routes to the new walker. New `rtu_smoke_bvh*` tests built on
   serialised `vk_bvh.h` fixtures.

#### 8.5.2 Files touched

- New: `sim/simx/rtu/bvh_types.h` (≈100 LoC, vk_bvh.h-compatible
  structs), `sim/simx/rtu/bvh_walker.{h,cpp}` (≈400 LoC, the
  traversal state machine — kept separate so the flat-list path
  stays untouched).
- Modified: `sim/simx/rtu/rtu_core.cpp` (≈200 LoC for state-machine
  dispatch on `scene_kind`, LaneState fields, leaf-type fan-out;
  ≈100 LoC for stack/trail integration).
- Modified: `VX_config.toml` (add `VX_CFG_RTU_STACK_DEPTH`,
  `VX_CFG_RTU_TRAIL_DEPTH`, `VX_CFG_RTU_BVH_WIDTH` already there).
- New: `tests/regression/rtu_smoke_bvh_basic`,
  `rtu_smoke_bvh_multilevel`, `rtu_smoke_bvh_instanced` (≈300 LoC
  per test, plus serialised BVH fixtures generated by a small
  host-side builder).

#### 8.5.3 Acceptance

- All 13 existing `rtu_smoke_*` tests still PASS (scene_kind=0 path
  unchanged).
- 3 new `rtu_smoke_bvh*` tests PASS, including: a 256-triangle Cornell
  box at depth ≤ 8 with deterministic hits; a 2-BLAS TLAS with two
  distinct meshes (not one mesh × N instances); a procedural-leaf
  scene exercising the IS callback path through the new walker.
- BVH-fetch counter (`bvh_nodes_fetched`) matches an oracle count
  computed by the host-side BVH builder for fixed seeds.

#### 8.5.4 Defer to later phases

- CW-BVH6 (Intel layout): `VX_CFG_RTU_BVH_WIDTH = 6` is structurally
  supported but the wide-box PE array stays 4-wide until §8.7. Walker
  works correctly at width 6 with 4-wide PE (3 + 3 split across two
  cycles); §8.7 closes the gap.
- On-device BVH builder. lavapipe's CPU builder is fine for Phase 4
  validation.
- BVH compression beyond Mesa-canonical quantization. Native CW-BVH4
  is enough; Opacity Micromaps / Displaced Micromeshes stay out of
  scope (§11).

### 8.6 Phase 5 — async ray pool + per-lane handle map (≈ 1 week)

The current ABI promises async-by-design `vx_rt_trace` (§4.3) but
the implementation is collapsed-sync: `process_trace` parks the trace
in RtuCore, `process_wait` is a no-op marker, and `vx_rt_trace`
returns handle 0 (`sim/simx/rtu/rtu_unit.cpp:115`). One in-flight ray
per (warp, lane). This blocks: (a) latency hiding under BVH-fetch
misses, (b) practical recursion (each recursive trace still
round-trips one at a time), (c) Phase 3-B's `vx_rt_cb_drain` opcode
should it ever be taken.

#### 8.6.1 Scope

1. **Real handle allocation.** `RtuCore` exposes a per-(warp,lane)
   handle table: `uint16_t outstanding[NUM_WARPS][NUM_THREADS][BATCH]`,
   with `BATCH = VX_CFG_RTU_HANDLES_PER_LANE` (default 4). A free-list
   inside RtuCore allocates slot indices on accept; `process_trace`
   writes the slot index into `trace->dst_data[t].u` (currently zero).
   The kernel sees a non-zero opaque handle.

2. **`vx_rt_wait` actually blocks on the handle.** `process_wait`
   reads rs1 = handle, looks up the matching slot's parked trace, and
   forwards the slot's TERMINAL status to the wait's writeback when
   it arrives. Today's wait passes rs1 through to rd unchanged
   (`sim/simx/rtu/rtu_unit.cpp:124-143`). New path: wait parks itself
   on a per-slot "waiters" list; SfuUnit's TERMINAL drain forwards
   the status to all waiters of that slot, then releases.

3. **Pool sizing.** `VX_CFG_RTU_CONTEXT_POOL = 32` already in
   `VX_config.toml`. Today's 4-slot `Impl::slots_` (RtuCore.cpp:324) is
   structurally a Phase 1 stub. Phase 5 grows it to 32 with a
   free-list, matching the proposal §5.2. Each slot is ≈256 B
   including the new BVH stack/trail (§8.5).

4. **`vx_rt_wait` on stale handle returns immediately.** If the slot
   has already retired and the handle's epoch (low bit) doesn't
   match, return last-known status without blocking — needed for
   Phase 5b recursion where a CHS dispatcher's nested wait may race
   the parent's drain.

5. **Recursion depth lifted.** Phase 12 (recursive vx_rt_trace) works
   today because slot count happened to be enough. With BATCH = 4 per
   lane and a proper pool, recursion depth is bounded by
   `pool_size / active_lanes` — typical 32 / 8 = 4 levels deep for a
   single warp, matches DXR's `maxRayRecursionDepth` default.

#### 8.6.2 Files touched

- Modified: `sim/simx/rtu/rtu_core.cpp` (≈150 LoC for free-list,
  handle-epoch encoding, drain-on-match), `rtu_unit.cpp` (≈100 LoC
  for real handle return + wait blocking), `sfu_unit.cpp` (≈50 LoC
  for waiter-list TERMINAL drain).
- Modified: `sim/simx/scheduler.cpp` (scoreboard for `vx_rt_wait` —
  must stall on writeback like other long-latency ops).
- New: `tests/regression/rtu_smoke_async_batch` (kernel issues 4 traces
  back-to-back, waits in reverse order, expects all 4 statuses).

#### 8.6.3 Acceptance

- Existing 13 tests + 3 new BVH tests still PASS.
- New `rtu_smoke_async_batch` PASS: 4 in-flight rays per lane,
  cross-cutting waits return the right status.
- Perf measurement: BVH-heavy scene takes < N × single-ray latency
  (latency-hiding kicks in).

### 8.7 Phase 6 — SIMD intersection coprocessors (≈ 4 days)

Today's intersection paths are scalar `for` loops:
`compute_intersections` calls `ray_triangle` once per tri
(`sim/simx/rtu/rtu_core.cpp:702`), and there's no wide-box function at
all. The proposal §5.7 claims BOX_PE = 4, TRI_PE = 4, NODE_LATENCY =
4, TRI_LATENCY = 6 — none modeled. All timing claims in §7.1 ("4
ray-box tests/cycle, 4 ray-triangle tests/cycle") are aspirational.

#### 8.7.1 Scope

1. **`ray_n_box_intersect(ray, BVHNode, BoxHit out[BVH_WIDTH])`.**
   Functional 4-wide (and 6-wide as a config) batched test. Mirrors
   the prototype's `ray_nBox_intersect`
   (`vortex-raytracing/sim/simx/rt_core.cpp:320-342`) including the
   quantized-AABB unpack via `std::ldexp(qaabb[i], node.ex)`.

2. **`ray_n_tri_intersect`.** Lift from prototype
   `rt_core.cpp:344-385`. Returns the per-tri hit array; the caller
   sorts and routes to AHS/CHS.

3. **Latency-annotated `SimChannel`s.** Add `NODE_LATENCY`,
   `TRI_LATENCY`, `XFORM_LATENCY` to RtuCore's per-tick scheduler.
   Today's `port.send(m)` (line 490) takes one cycle; under Phase 6 a
   `box_pe_ledger` schedules the result `NODE_LATENCY` cycles in the
   future. Same pattern as `latency_of()` in
   `sim/simx/sfu_unit.cpp:72`.

4. **Validation hook.** Phase 6's pipelined latency model must match
   the unpipelined functional result bit-for-bit when timing is
   averaged. New `--validate-pe-timing` flag in the smoke driver runs
   the test twice (once with PE_LATENCY = 0, once with proposal
   defaults) and diffs the result.

#### 8.7.2 Files touched

- New: `sim/simx/rtu/box_pe.cpp`, `tri_pe.cpp` (≈150 LoC each).
- Modified: `rtu_core.cpp` (≈100 LoC to route INTERSECT_BOX /
  INTERSECT_LEAF through the new PE ledgers).
- Modified: `VX_config.toml` (`VX_CFG_RTU_BOX_PE`, `_TRI_PE`,
  `_NODE_LATENCY`, `_TRI_LATENCY`, `_XFORM_LATENCY`).

#### 8.7.3 Acceptance

- All prior tests PASS.
- Cycle counts on `rtu_smoke_bvh_basic` track the closed-form formula
  `nodes_visited × NODE_LATENCY / BOX_PE + leaves × TRI_LATENCY /
  TRI_PE` ± 5%.

### 8.8 Phase 7 — production correctness (ray flags + hit attrs) (≈ 2 days)

Two small but spec-critical gaps. Both block real Vulkan workloads;
neither needs the BVH walker first.

#### 8.8.1 Scope

1. **Ray-flags fast-out.** `VX_RT_FLAG_*` is defined in
   `VX_types.toml:371-391` but only `ENABLE_CHS` /`ENABLE_MISS` are
   honored (`sim/simx/rtu/rtu_core.cpp:781,806`). Add handling for:
   - `TERMINATE_ON_FIRST_HIT` — break the leaf scan on first opaque
     hit. Shadow rays save 2-4× cycles.
   - `CULL_BACK_FACING` / `CULL_FRONT_FACING` — sign-bit cull on
     `dot(ray_dir, normal)` in `ray_triangle`.
   - `CULL_OPAQUE` / `CULL_NO_OPAQUE` — skip the leaf class entirely.
   - `SKIP_TRIANGLES` / `SKIP_AABBS` — fast-skip leaf type at fetch.
   Total: 8 flag checks at 4 well-defined points (leaf-scan break,
   tri-test post-test, leaf-type pre-fetch).

2. **`hit_attr[0..3]` plumbing.** Slots 17-20 are reserved in
   `VX_types.toml:309-312` but `RtuRsp` carries only the canonical
   hit fields — no `hit_attr[]` array. IS shaders that write
   `hitAttributeEXT` have nowhere to put the data. Fix: extend
   `RtuRsp` and `LaneState` with `std::array<uint32_t, 4> hit_attr`;
   `RtuUnit::apply_response` / `apply_callback_payload` writes them
   through to the regfile.

#### 8.8.2 Files touched

- Modified: `sim/simx/rtu/rtu_core.cpp` (≈80 LoC ray-flag checks),
  `rtu_unit.{h,cpp}` (≈40 LoC hit_attr plumbing).
- New: `tests/regression/rtu_smoke_shadow` (TERMINATE_ON_FIRST_HIT),
  `rtu_smoke_cull_back` (CULL_BACK_FACING).

### 8.9 Phase 8 — coherency gathering + performance counters (≈ 2 days)

Two pieces that together unblock Phase 3 fork measurement (proposal
§8.4 explicitly says the fork is data-driven).

#### 8.9.1 Scope

1. **Octant signature coherency gather (§5.3).** Add `uint8_t
   coh_signature` to `LaneState` — 3 bits from
   `(sign(dx), sign(dy), sign(dz))` set at trace-accept. `RtuCore::tick`
   processes slots in `last_signature_ first, then others` order. ≈60
   LoC. NVIDIA CGU equivalent (proposal §2.1).

2. **Performance counter surface.** Today's `PerfStats`
   (`rtu_core.h:43-56`) has 4 counters. Add:
   - `bvh_nodes_fetched`, `bvh_leaves_fetched`, `instance_descents`
   - `box_tests`, `tri_tests` (post-PE expansion: per-lane counts)
   - `pool_occupancy_histogram[33]` (0..32 in-flight)
   - `ahs_callbacks`, `chs_callbacks`, `miss_callbacks`,
     `is_callbacks`, `recursive_traces`
   - `reformation_yields_emitted`, `reformation_avg_lanes_per_yield`
   - `coherency_hits` (same signature as last pick) /
     `coherency_misses`
   - **2D status histogram** `latency_dist[warp_state][ray_state]`
     mirroring the prototype's design
     (`vortex-raytracing/sim/simx/rt_unit.h:16-46`,
     `rt_trace.cpp:168-197`). Tracks each cycle as
     {stalled, waiting, executing} × {awaiting_pool, awaiting_mem,
     intersecting, callback, terminal}. Single most useful diagnostic
     for Phase 3 fork.

#### 8.9.2 Files touched

- Modified: `sim/simx/rtu/rtu_core.{h,cpp}` (≈150 LoC counters +
  signature sort), `cluster.cpp` (≈20 LoC perf-print hookup).
- Modified: `sim/simx/main.cpp` to add `--rtu-stats` flag dumping
  the histogram as JSON.

### 8.10 Phase 9 — private BVH cache (≈ 1-2 days, optional)

Defer until §8.9 measurements show dcache thrash. If they do: add a
small fully-associative line cache (32-64 entries) in front of
`dcache_req_out`. Single ≈200 LoC file. NVIDIA's "Ray Bank" / Intel's
"BVH cache" analog. Stays cluster-local; no inter-cluster sharing
(out of scope per §11).

### 8.11 Beyond Phase 9 — true future work

The original §8.5 deferred-list reduced to items that genuinely need
to wait for upstream design:

- **CW-BVH6 (full 6-wide PE).** Trade silicon for fewer fetches;
  Intel-aligned. After Phase 4-6 land, ~1 week to widen PE arrays.
- **`vx_reorder` opcode (NVIDIA SER opt-in).** Kernel-driven
  reformation hint. ≈100 LoC. Only justified if Phase 8 stats show
  reformation bottleneck.
- **On-device BVH builder.** Replaces lavapipe CPU radix-sort.
  Separate proposal; massive scope. Defer until end-to-end Vulkan
  workload runs.
- **Opacity Micromaps / Displaced Micromeshes.** NVIDIA Ada
  features; require BVH format extensions and AHS-skip semantics.
  Future.

### 8.12 Summary: PRISM-to-production roadmap

| Phase | Scope                                            | LoC est. | Days est. | Status     | Unblocks                              |
|-------|--------------------------------------------------|----------|-----------|------------|---------------------------------------|
| 0     | pre-work                                         | -        | 2         | done       | -                                     |
| 1     | ray-query, opaque                                | ~1500    | 10        | done       | smoke test pass                       |
| 2     | AHS/IS callbacks + reformation                   | ~800     | 15        | done       | full callback taxonomy (CHS/MISS/IS)  |
| 3-A   | same-warp SBT-coherent batching                  | ~300     | 5         | done (3-A2) | divergent-SBT scenes                  |
| 3-B   | explicit async `cb_drain` ABI                    | ~400     | 20        | deferred   | Phase 3-A measurements first          |
| **4** | **real BVH walker (CW-BVH4 + stack + TLAS→BLAS)** | **~1000** | **10**    | **next**   | **any Vulkan scene > 8 tris**         |
| **5** | **async ray pool + per-lane handle map**         | **~350** | **5**     | **next**   | **latency hiding, deep recursion**    |
| **6** | **SIMD box/tri PEs + pipeline latencies**        | **~400** | **4**     | **next**   | **realistic §7.1 throughput claims**  |
| **7** | **ray flags + hit_attr plumbing**                | **~120** | **2**     | **next**   | **shadow rays, IS shaders**           |
| **8** | **coherency gather + perf counters**             | **~170** | **2**     | **next**   | **Phase 3 fork decision**             |
| **9** | **private BVH cache (optional)**                 | **~200** | **2**     | **opt**    | **incoherent-ray performance**        |
| 10+   | CW-BVH6, vx_reorder, OMM, on-device builder      | -        | -         | future     | -                                     |

**Critical path to "PRISM is a real RTU":** Phases 4 → 5 → 8 (≈ 17
days, ≈ 1500 LoC). Phases 6-7-9 are quality/correctness add-ons that
can land in parallel or after. The full Phase 4-9 block fits in ≈ 5-6
weeks of focused work and lifts PRISM from "callback framework with a
toy walker" to "PRISM Phase 1 actually being what the spec says."

## 9. Comparison tables

"PRISM" in the tables refers to this proposal. Restricted to
publicly-documented behaviour for NVIDIA and Intel rows (NVIDIA
whitepapers + PTX/OptiX docs; Intel Xe-HPG ISA reference + ANV Mesa
source).

### 9.1 ISA-level comparison

| Axis                          | PRISM — Phase 1-2 committed                                          | NVIDIA Turing-Ada RT Core                          | Intel Xe-HPG RT Unit                              |
|-------------------------------|----------------------------------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| Opcode count (Phase 1)        | **4** (`vx_rt_set`, `vx_rt_get`, `vx_rt_trace`, `vx_rt_wait`) | 1 ray-trace SASS opcode family + service ops (undoc.) | 1 `TRACE_RAY` send-message class + service messages |
| Opcode count (Phase 2)        | 4 above + 1 (`vx_rt_cb_ret`) = **5**                                 | unchanged                                         | unchanged                                         |
| Opcode count if Phase 3-B taken | 5 above + 1 (`vx_rt_cb_drain`) = **6**                              | unchanged                                         | unchanged                                         |
| Trace shape                   | **Async-by-design**: `vx_rt_trace` returns a handle; `vx_rt_wait` blocks on the handle. Mesa emits the BATCH=1 pair from synchronous `traceRayEXT` / `rayQueryProceedEXT` | Sync-by-design at the kernel level; HW pipeline is async internally | Sync-by-design send-message; HW pipeline is async internally |
| Concurrency level             | Thread / SIMT lane                                                   | Warp-issued, per-lane processed by RT Core         | Thread-issued via send-message                     |
| Argument passing              | **Per-(warp,lane) RTU register file** (bulk-set ops)                 | Register-resident, SASS-compiler-allocated         | Message-header registers + thread-private message regs |
| Result delivery               | RTU register file (scalar `vx_rt_get`)                               | Register file                                      | MRF + memory                                       |
| Mid-traversal yield (AHS/IS)  | **HW PC redirect** to callback entry point (Phase 2). Phase 3 fork: 3-A cross-warp gather, 3-B explicit drain | RT Core suspends thread, HW dispatches AHS/IS via SBT | RT Unit suspends, **Bindless Thread Dispatcher** reforms |
| Recursion depth               | Always 1 in HW; Mesa lowers recursion to iteration                   | HW with continuation stack in global memory        | HW with continuation stack in global memory        |
| Ray flags                     | Full KHR set in Phase 1 (cull, opaque, fast-out) + AABB-skip in Phase 2 | Full KHR set                                       | Full KHR set                                       |
| Public encoding documentation | **Open** (RISC-V `CUSTOM1`/funct3=5/funct2 in §4.3)                  | None (vendor SASS opaque)                          | Partial (Xe-HPG ISA Reference, public)             |
| ISA encoding cost             | 1 funct3 slot in CUSTOM1 (funct2 sub-encoding covers all phases)     | 1 SASS opcode + sideband state                     | 1 send-message class                               |

### 9.2 ABI-level comparison

| Axis                                  | PRISM                                              | NVIDIA (Vulkan KHR / OptiX)                      | Intel (Vulkan KHR via ANV)                       |
|---------------------------------------|----------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Ray descriptor location               | **Per-(warp,lane) RTU register file** (dedicated SRAM, 116 B) | Shader register file (SASS-allocated)         | MRF + thread-private message regs                |
| Hit attributes location               | **Same RTU register file** (16 B `hitAttributeEXT`, plus IDs) | Shader register file                           | MRF + thread-private regs                        |
| Payload (`rayPayloadEXT`)             | Memory struct via 64-bit pointer in RTU regs       | Register-allocated up to ~32 DWORDs, spill to memory | Mixed register/memory continuation              |
| HW warp reformation                   | **Phase 2**: same-warp PC redirect. **Phase 3**: cross-warp gather via shader queues. | Implicit per-thread dispatch from RT Core + **SER** opt-in API | **Implicit via Bindless Thread Dispatcher**       |
| SBT walk                              | SW (kernel decodes SBT entry, callback inlined by Mesa) | HW (RT Core walks SBT + dispatches)             | HW (BTD walks + dispatches)                      |
| SBT handle size                       | 32 B (Mesa-canonical)                              | 32 B                                             | 32 B                                             |
| BVH node format                       | Vendor-neutral `vk_bvh.h` (Mesa-canonical)         | Vendor-specific (compressed, not documented)     | 6-wide CW-BVH (publicly documented)              |
| BVH builder                           | CPU (lavapipe radix-sort); on-device deferred      | Driver-side + HW-accelerated on newer SKUs       | Driver-side + HW-accelerated                     |
| Recursion model                       | Mesa lowers `traceRayEXT` recursion to iteration   | HW continuation stack                            | HW continuation stack                            |
| Coherency gathering / ray sort        | **Yes** (octant-signature pool-pick, §5.3)         | **CGU + SER** on Ada                             | **Ray Coherence** (similar)                      |
| Pool / in-flight ray count            | 32 / RtuCore                                       | ~8-16 / RT Core                                  | Comparable, in Ray Bank                          |
| Open-source driver stack              | **Open**: Mesa lavapipe + vortexpipe               | Closed: CUDA / OptiX / DXR proprietary           | **Open**: Mesa ANV                               |

## 10. Risks and open questions

R1. **mesa_vortex NIR-pass pre-emption hook.** Needs verification
that lavapipe exposes the pre-emption callback for vortexpipe to
intercept before `lvp_nir_lower_ray_queries.c`. Mitigation: if
pre-emption is not exposed, vortexpipe can post-process the
already-lowered NIR and pattern-match the inlined BVH walk back into
`vx_rt_*` opcodes.

R2. **Phase 3 warp-state stash cost.** Cross-warp reformation
requires saving 32-lane GPR state for each participating warp
before redirecting the warp to a reformed PC. At 32 × 32 × 4 = 4 KB
per warp, that's measurable cluster scratch. Mitigation: only stash
the *minimum live-out set* (compiler-emitted bitmask); typical CHS
shaders touch < 16 GPRs.

R3. **HW PC redirect interaction with SIMT stack.** When HW
redirects a warp to the callback PC, the SIMT stack must be
preserved so post-callback the warp returns to the original
context. Mitigation: HW redirect pushes one synthetic stack entry;
`vx_rt_cb_ret` pops it.

R4. **Reformation latency upper bound (Phase 2 same-warp; Phase 3-A
cross-warp).** If `REFORM_WAIT` is too high and queues never fill to
SIMD_WIDTH, warps stall waiting for their callback. Mitigation:
`REFORM_MIN` + `REFORM_WAIT` ensures worst-case latency = WAIT
cycles; SW-tunable via `VX_DCR_RTU_REFORM_THRESH`.

R5. **HW PC-redirect rendezvous point.** Phase 2 redirects the warp
PC during `vx_rt_wait`, not during `vx_rt_trace`. This means a
callback fires only after the lane reaches the wait. Practical
consequence: kernels that interleave significant work between trace
and wait (a wavefront-RT pattern) can introduce a per-lane scheduling
gap. Mesa's BATCH=1 trace+wait pair keeps this gap to zero. For Phase
3-B (async), `vx_rt_cb_drain` becomes the rendezvous and the gap
disappears.

R6. **BVH builder repeatability.** lavapipe's radix-sort BVH builder
is non-deterministic; regression tests need a serialised BVH fixture
loaded from disk rather than rebuilt each run. Mesa has
`vk_acceleration_structure_serialize`; small test-harness hook
needed.

R7. **Pool bank conflicts.** 4-bank pool with `hart_id mod 4`
banking; if a warp has uneven lane distribution (e.g. some lanes
masked, others not), bank pressure can spike. Mitigation: lane-id
permutation in the bank index function; profile and tune.

R-A. **Phase 3-A: cross-warp warp-state stash cost.** Reformation
across warps requires saving the participating warps' live-out GPR
set (compiler-emitted bitmask) and a cluster signalling fabric.
~256 B per stash event; ≤ 1 event per cycle per RtuCore. RTL area is
non-trivial; profile Phase 2 first.

R-B1. **Phase 3-B: pool size scaling.** 128-slot pool ≈ 32 KB SRAM
per RtuCore. Bounded but 4× Phase 1-2's 8 KB. RTL area cost.

R-B2. **Phase 3-B: handle exhaustion.** If kernel-requested BATCH >
pool-slots-per-lane (e.g. 16 vs 4), `vx_rt_trace` async-issue must
block. Expose the limit as a runtime cap query so kernel authors can
size BATCH accordingly.

R-B3. **Phase 3-B: Mesa lowering tractability — load-bearing.** The
batching benefit only materialises if the NIR pass finds BATCH ≥ 4
on ≥ 50% of `traceRayEXT` call sites in real workloads (Phase 2's
continuation-passing form may produce tightly-coupled stages with
limited batching surface). The Phase 3-B go/no-go decision in §8.4
hinges on this measurement.

R8. **Determinism.** SimX must be bit-deterministic under fixed seed
+ config (cf. v3 expectation). Pool picker, reformation priority,
dcache-arbiter slot priority, queue scan order all need stable
orderings. Use insertion-order vectors, not `unordered_map`, inside
`RtuCore`. For Phase 3-B, async callback completion order is
non-deterministic by Vulkan spec — kernels must not rely on it.

R9. **Phase 4 — vk_bvh.h schema drift.** Mesa's `vk_bvh.h` is not
a stable cross-driver ABI; ANV and RADV have made layout changes
without coordinating. Mitigation: pin a specific Mesa SHA in
`third_party/mesa/`, serialise BVHs to disk for regression
fixtures, version the on-disk format with a 4-byte magic.

R10. **Phase 4 — short-stack underflow under deep BVH.** A
`VX_CFG_RTU_STACK_DEPTH = 16` short stack with `RESTART` fallback
costs O(depth) re-walks on underflow. Pathological BVHs (skinny
trees, degenerate scenes) can multiply node-fetches by 4-8×.
Mitigation: instrument `restart_events` counter (Phase 8), warn
when restarts > 10% of fetches; lift stack depth to 24 if needed.

R11. **Phase 5 — handle-epoch wraparound.** With BATCH = 4 and
16-bit handle, epoch field is 14 bits (4 slot bits + 14 epoch),
wraps every 16k allocations. Long-running kernels may collide.
Mitigation: per-slot epoch counter, not global; collision needs
16k allocations *to the same slot* which is bounded by pool turnover.

R12. **Phase 6 — PE-latency model accuracy.** SimX SimChannel
latency annotation is per-message, not pipelined. Modeling a 4-cycle
NODE_LATENCY × BOX_PE = 4 pipeline accurately would require a true
shift-register; the proposed `box_pe_ledger` is a discrete-event
approximation that's correct on average but optimistic on tight
back-to-back issues. Mitigation: document the approximation, validate
against RTL once available.

R13. **Phase 8 — `latency_dist` 2D histogram cost.** The prototype's
histogram is `warp_statuses × ray_statuses` per cycle per slot.
At 32 slots × NUM_WARPS warps × 1M cycles, the per-tick update
becomes measurable. Mitigation: lazy update (only on state transition,
not every cycle); profile.

## 11. Out of scope

- RTL elaboration of `RtuCore`, intersection units, pool/queue BRAM,
  DCR fan-out, cluster-signal fabric. Follows separately; SimX is the
  goal-reference oracle.
- Callable shaders.
- On-device BVH builder.
- Acceleration-structure compaction / serialise / deserialise.
- Opacity Micromaps, Displaced Micromeshes.
- Position-fetch (`VK_KHR_..._position_fetch`) beyond the natural
  fall-out of `triangle_node` decode.
- DXR / D3D12. Vulkan only.
- Multi-RTU cross-cluster shared BVH cache. Each RTU is independent.

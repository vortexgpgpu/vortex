# Minimal Verilog RTU for the raytrace Example

## 1. Goal

Implement a synthesizable **PRISM RTU in Verilog** with the smallest feature
set that runs the `tests/vulkan/raytrace` example end-to-end on hardware
(rtlsim / FPGA) — a cluster-shared fixed-function ray-traversal unit behind the
SFU that walks a **compressed-wide-BVH** scene and returns the closest
opaque-triangle hit. The node fan-out is parametrized by
**`VX_CFG_RTU_BVH_WIDTH`**, **default 4 (CW-BVH4)**. The SimX `sim/simx/rtu/`
model is the golden cycle-reference; this RTL targets the same behaviour and
closes timing on the Alveo **U55C at 300 MHz**.

**Why CW-BVH4 is the default.** A CW-BVH4 internal node is **64 B — exactly one
Vortex cache line** (`L1_LINE_SIZE` = `MEM_BLOCK_SIZE` = 64 B), so every
node fetch is one aligned line, one RTCache lookup, zero waste. CW-BVH6's 96 B
node straddles two 64 B lines (~2× line traffic, ~33 % wasted bandwidth), and
the shallower tree doesn't repay it on a 64 B-line machine — the same reason
**AMD RDNA2/RDNA3 ship BVH4 with 64 B nodes**. BVH6 (`WIDTH=6`) is retained as a
config option for a future wider-cache variant; the RTL parametrizes by
`VX_CFG_RTU_BVH_WIDTH` (and the matching `VX_CFG_RTU_BOX_PE`) so both build.

Scope is deliberately bounded to what the example exercises (§3). Everything the
example does not need — any-hit/intersection callbacks, instancing transforms,
procedural primitives, ray-tracing pipelines — is explicitly out of scope (§4)
and left for follow-on phases.

## 2. What the raytrace example exercises

`tests/vulkan/raytrace` casts one orthographic **inline ray query**
(`VK_KHR_ray_query`) per pixel at three overlapping **opaque** triangles, with
`gl_RayFlagsOpaqueEXT`, and reads back the committed closest hit
(`t`, barycentrics, `gl_PrimitiveID`). vortexpipe lowers the query to the RTU
`vx_rt_*` ops and transcodes the acceleration structure to a **CW-BVH scene**
in device memory (`vp_build_bvh*_scene` in `vp_launch.c`) whose fan-out matches
the RTU's `VX_CFG_RTU_BVH_WIDTH` — **CW-BVH4 `scene_kind=2`** by default,
CW-BVH6 `scene_kind=3` when built `WIDTH=6`. The RTU must therefore provide
exactly:

- the `vx_rt_set / vx_rt_trace / vx_rt_wait / vx_rt_get` ISA surface (CUSTOM1,
  funct3=5), per SIMT lane;
- closest-hit traversal of a single-level CW-BVH (width `VX_CFG_RTU_BVH_WIDTH`)
  of opaque triangles;
- a hit record of `{ hit, t, u, v, geometry_index, primitive_id }` per lane.

That is the entire minimal requirement. The SimX `Bvh4Walker` decodes both
widths from one width-generic `VxBvhNodeView` (`kVxBvhWidth = VX_CFG_RTU_BVH_WIDTH`);
this proposal is the RTL realization of that closest-hit subset.

## 3. Minimal feature set (in scope)

| Capability | Detail |
|---|---|
| ISA ops | `vx_rt_set(slot,val)`, `vx_rt_trace(tlas)→handle`, `vx_rt_wait(handle)→status`, `vx_rt_get(slot,status)→val` — CUSTOM1 opcode 43, funct3=5, decoded SFU-side |
| Ray state | per-context slots: `origin[3]`, `dir[3]`, `t_min`, `t_max`, `flags`, `cull_mask`, scene base ptr |
| Scene | single-level **CW-BVH**, fan-out `VX_CFG_RTU_BVH_WIDTH`: 16 B header; internal node = **64 B (BVH4, default)** or 96 B (BVH6) — common origin + per-axis int8 exponent + 8-bit quantized child AABBs; 56 B leaves (16 B header + 40 B triangle), one triangle per leaf |
| Traversal | closest-hit, short-stack, ray-box (slab) + ray-triangle (Möller–Trumbore) |
| Opaqueness | opaque only — every leaf hit commits immediately, no shader round-trip |
| Result | `{ hit, t, u, v, geometry_index, primitive_id }` per lane, read back via `vx_rt_get` after `vx_rt_wait` |
| SIMT | one independent ray context per active lane of the issuing warp |

The hardware scene format is **exactly** what the host transcode emits and what
the SimX walker decodes (`rtu_bvh.h`: `VxBvhInternalNode` (64 B, BVH4) /
`VxBvh6InternalNode` (96 B, BVH6) / `VxBvhLeafHeader`), so the RTL and the
driver share one format per configured width.

## 4. Explicitly out of scope (deferred)

These are the large simplifications that keep the RTL minimal; each maps to a
known SimX feature that stays disabled/bypassed:

- **Any-hit / intersection callbacks** → no trap-based callback dispatch, no
  `ReformationEngine`, no scoreboard snapshot/restore. Opaque-only means
  traversal never yields back to a shader. This removes the single most complex
  block.
- **TLAS instancing / object-space transforms** → no `XformUnit`. The driver
  flattens the AS to a world-space single-level BVH, so the walker only ever
  sees world-space geometry.
- **Procedural / AABB primitives** (`kVxBvhKindLeafProc`) → unsupported leaf
  kind; the example has only triangle leaves.
- **Ray-tracing pipelines / SBT, miss/closest-hit shaders, recursion** → these
  are driver/SIMT concerns, not RTU hardware.
- **Multi-geometry, non-opaque, alpha test, ray flags beyond Opaque/CullMask.**

A conformant minimal RTU may legally fault or ignore these; the driver's
cap-guard already routes such shaders to the software path.

## 5. Block architecture

Cluster-shared, one `VX_rtu_core` per `VX_CFG_NUM_RTU_CORES`
(= `max(1, NUM_CORES/4)`), reached through the SFU. Mirrors the SimX
`RtuUnit → RtuCore → { walker, MemoryEngine }` decomposition.

```
  SFU (CUSTOM1 funct3=5)
     │  VX_rtu_if  (valid/ready: lane_mask, op, slot, data, tlas)
     ▼
  VX_rtu_unit              — per-core SFU-side shim: latches ray state into
     │                       the context RF, issues trace, returns wait status
     ▼
  VX_rtu_core ───────────────────────────────────────────────┐
     │  VX_rtu_scheduler   — context pool (CONTEXT_POOL slots), per-lane       │
     │                       short-stack (STACK_DEPTH), picks a ready context  │
     │                       each cycle and drives the traversal FSM           │
     │  VX_rtu_box_pe[W]   — ray-AABB slab test (W=BVH_WIDTH, default 4),      │
     │                       dequant qaabb, NODE_LATENCY                       │
     │  VX_rtu_tri_pe[k]   — Möller–Trumbore ray-tri (t,u,v), TRI_LATENCY      │
     │  VX_rtu_mem         — node/leaf fetch, NUM_RTU_BLOCKS outstanding ports │
     └──────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
                              VX_cache (RTCache, read-only BVH)
                                         ▼
                                    L2 / memory
```

Parameters reuse the existing config knobs: `VX_CFG_RTU_BVH_WIDTH` (4 default),
`VX_CFG_RTU_BOX_PE`, `VX_CFG_RTU_TRI_PE`, `VX_CFG_RTU_STACK_DEPTH`,
`VX_CFG_RTU_CONTEXT_POOL`, `VX_CFG_RTU_NODE_LATENCY`, `VX_CFG_RTU_TRI_LATENCY`,
`VX_CFG_NUM_RTU_BLOCKS`, `VX_CFG_NUM_RTCACHES`.

## 6. Traversal datapath (per context)

A short-stack closest-hit walk, one ray per context:

1. **Init.** `vx_rt_trace` seeds the context: push the BVH root (from the
   scene header `root_node_offset`), set `t_hit = t_max`, `hit = 0`.
2. **Node fetch.** Pop the stack; issue a `VX_rtu_mem` read for the node
   (64 B BVH4 / 96 B BVH6 internal, or 56 B leaf — the low byte of
   word0 is the kind tag).
3. **Internal node.** Dequantize the up-to-`VX_CFG_RTU_BVH_WIDTH` child AABBs
   (`min = origin + qaabb_min·2^exp`), run the `VX_CFG_RTU_BVH_WIDTH`
   `VX_rtu_box_pe` slab tests in parallel against the ray, and push the
   children that hit within `[t,t_hit)`
   **far-to-near** (so the nearest is popped first — front-to-back ordering that
   lets `t_hit` prune the rest).
4. **Leaf node.** For each triangle (one per leaf here), run `VX_rtu_tri_pe`
   (Möller–Trumbore). On a hit with `t < t_hit`, update
   `{ t_hit, u, v, geometry_index = leaf.geometry_index,
      primitive_id = leaf.prim_base + i, hit = 1 }`.
5. **Terminate.** Stack empty → write the hit record to the lane result regs
   and retire the context; `vx_rt_wait` unblocks and `vx_rt_get` reads it back.

The box and triangle intersections are the FP-heavy critical paths and are the
pipelined PEs (latencies = the `*_LATENCY` config); the traversal FSM itself is
control-only.

## 7. ISA / SFU integration

The four ops decode in the SFU as CUSTOM1 (opcode 43), funct3=5; the funct7 low
2 bits select set/get/trace/wait and, for set/get, the upper bits carry the RTU
register-file slot (matching `vp_nir_to_llvm`'s `.insn r 43, 5, …` emission and
`sw/kernel/include/vx_raytrace.h`):

- `set` (slot ← rs1): no writeback; latches a ray-state slot for the lane.
- `trace` (rd ← handle ← trace(rs1=scene ptr)): allocates a context, starts the
  walk; `handle` is a scoreboard token.
- `wait` (rd ← status ← wait(rs1=handle)): stalls the lane until the context
  retires (scoreboard-ordered, the SimX `vx_rt_get_after` discipline).
- `get` (rd ← slot, rs1=status): reads a result slot, ordered after `wait`.

SIMT divergence: inactive lanes contribute no context; the scheduler packs
active lanes' rays into the context pool.

## 8. Verilog module plan

New RTL under `hw/rtl/rtu/`, following `docs/coding_guidelines_verilog.md`
(4-space, `VX_` PascalCase modules, mandatory `begin`/`end`, valid/ready
interfaces, `` `UNUSED_* `` tags, no blanket lint pragmas):

| Module | Role |
|---|---|
| `VX_rtu_pkg.sv` | params (BVH width, latencies, pool/stack depth), node/leaf struct types |
| `VX_rtu_if.sv` | SFU↔RTU request/response interfaces (valid/ready) |
| `VX_rtu_unit.sv` | per-core SFU shim: ray-state RF, trace launch, wait/result return |
| `VX_rtu_core.sv` | scheduler + stacks + PE arrays + mem engine integration |
| `VX_rtu_scheduler.sv` | context pool + short-stack + traversal FSM |
| `VX_rtu_box_pe.sv` | pipelined ray-AABB slab test (×`BOX_PE`) |
| `VX_rtu_tri_pe.sv` | pipelined Möller–Trumbore ray-triangle (×`TRI_PE`) |
| `VX_rtu_mem.sv` | node/leaf fetch FSM, RTCache request/response arbitration |

FP arithmetic reuses Vortex's existing FPU primitives where possible; the
box/tri datapaths are register-balanced to the configured latencies so no single
combinational path exceeds the 300 MHz budget (guideline §10).

## 9. SimX as the golden reference

Per the SimX↔RTL parity discipline, the SimX RTU is the oracle: build the
accurate functional model first (already shipped — `FlatWalker`/`Bvh4Walker`
decode both widths today via `kVxBvhWidth = VX_CFG_RTU_BVH_WIDTH`), get the
example passing, then diff SimX↔RTL trace dumps
(rays launched, node/tri tests, hit records) to localize any RTL divergence.
The cycle knobs (`*_LATENCY`, PE counts) are shared, so cycle counts should
track within the project's <5% parity goal.

## 10. Verification & bring-up

- **Functional unit tests:** `tests/raytracing/rtu_smoke_bvh6` (single 6-wide
  node, nearest triangle in the last child — already validates width-6 decode)
  and `rt_raycast`, run through **rtlsim** then signed off on **FPGA via xrt**
  (the canonical RTL coverage path; rtlsim is for unit-level bring-up).
- **End-to-end:** `tests/vulkan/raytrace` through vortexpipe → rtlsim/xrt,
  checking the same pass criterion as SimX (per-primitive colours +
  depth-ordered occlusion, `centre=blue`).
- **Parity:** SimX vs RTL trace diff on ray/box/tri counts and hit records.

## 11. Phased plan

- **Phase 0 — Skeleton + ISA.** `VX_rtu_pkg`, interfaces, `VX_rtu_unit` shim
  decoding the four ops with a stub single-cycle "miss" walker. *Exit:*
  `vx_rt_trace/wait/get` round-trip in rtlsim, no traversal.
- **Phase 1 — Box PE + internal-node walk.** Dequant + `W`-wide slab test +
  short-stack push/pop, all-miss leaves. *Exit:* correct node-visit counts vs
  SimX on `rtu_smoke_bvh6`.
- **Phase 2 — Tri PE + closest hit.** Möller–Trumbore, `t_hit` pruning, hit
  record. *Exit:* `rtu_smoke_bvh6` + `rt_raycast` pass in rtlsim.
- **Phase 3 — SIMT contexts + mem engine.** Context pool, per-lane stacks,
  `NUM_RTU_BLOCKS` outstanding fetches via RTCache. *Exit:* `tests/vulkan/raytrace`
  passes in rtlsim with multiple in-flight rays.
- **Phase 4 — Timing + FPGA.** Pipeline balancing to 300 MHz on U55C; sign off
  the suite via xrt; reconcile SimX↔RTL parity.

## 12. Deliverables

1. `hw/rtl/rtu/` minimal RTL (modules in §8), parametrized by the existing
   `VX_CFG_RTU_*` knobs, CW-BVH closest-hit (default BVH4), opaque-only.
2. Green `rtu_smoke_bvh6`, `rt_raycast`, and `tests/vulkan/raytrace` on rtlsim
   and on the U55C via xrt.
3. A SimX↔RTL parity report (ray/box/tri counts, hit records, cycle delta).
4. Timing closure at 300 MHz on U55C.

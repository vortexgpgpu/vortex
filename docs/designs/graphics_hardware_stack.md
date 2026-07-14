# Graphics Hardware Stack (RASTER / TEX / OM) — Design

**Scope:** the Vortex graphics fixed-function (FF) hardware — the
rasterizer (RASTER), the texture sampler (TEX), and the output-merger /
ROP (OM) — plus the SIMT-side plumbing that binds them (the fragment
dispatch path and the shared graphics register window). Covers the RTL
([`hw/rtl/raster/`](../../hw/rtl/raster/), [`hw/rtl/tex/`](../../hw/rtl/tex/),
[`hw/rtl/om/`](../../hw/rtl/om/), [`hw/rtl/VX_graphics.sv`](../../hw/rtl/VX_graphics.sv)),
the SimX models ([`sim/simx/raster/`](../../sim/simx/raster/),
[`sim/simx/tex/`](../../sim/simx/tex/), [`sim/simx/om/`](../../sim/simx/om/)),
and the kernel ABI.

This document covers the **hardware microarchitecture, ISA, dispatch, and
VM tie-in**. The complementary **software / compiler / rendering pipeline**
(the vortexpipe Gallium driver, the on-device front end, NIR→Vortex
lowering, `vkCmdDraw` flow) is documented in
[`graphics_software_stack.md`](graphics_software_stack.md) and
[`vortexpipe_architecture.md`](vortexpipe_architecture.md). The program-level
"true GPU" master plan is
[`../proposals/gfx_v2_true_gpu.md`](../proposals/gfx_v2_true_gpu.md).

The three units are RISC-V ISA extensions under `custom1` (`INST_EXT2 =
0x2B`), advertised via `MISA` bits TEX=6, RASTER=7, OM=8, each gated by
`VX_CFG_EXT_{TEX,RASTER,OM}_ENABLE`.

---

## 1. Architecture overview

The graphics units are **cluster-shared** fixed-function engines (not
per-core). RASTER **pushes** fragment work onto the SIMT cores; the
fragment shader then invokes TEX and OM as **SFU processing elements**
([`VX_sfu_unit.sv`](../../hw/rtl/core/VX_sfu_unit.sv), `PE_IDX_{TEX,OM}`).
Each unit has a dedicated cluster-level cache — tcache (textures), rcache
(raster tile/prim buffers), ocache (color + depth framebuffers, coherent
with early-Z reads). The cluster wrapper
[`VX_graphics.sv`](../../hw/rtl/VX_graphics.sv) instantiates the per-unit
arbiters, cores, caches, and DCR fan-out.

```
                on-device front end (SIMT): setup + bin-sort  ──► primbuf + bin headers
                                                                       │  (RASTER DCRs)
                                                                       ▼
   ┌──────────────────────── RASTER fixed-function (cluster) ─────────────────────┐
   │  VX_raster_mem → te → be → slice/edge/extents → qe   (tile→block→covered quad)│
   │        └─► VX_raster_earlyz  (optional occlusion cull vs committed depth)     │
   │        └─► VX_raster_packer  (compact sparse quads → dense fragment waves)    │
   │        └─► VX_raster_dispatch (per-core: LAUNCH one 1-warp fragment CTA)      │
   └───────────────────────────────────┬──────────────────────────────────────────┘
                                        │ payload in the gfx register window (at launch)
                                        ▼
                        fragment shader kernel (SIMT)
                          vx_frag_load → persp-correct interp → color/depth
                             │  vx_gfx_set (stage u,v / color,depth)
                             ▼                    ▼
                        vx_tex4 (TEX)   ────►  vx_om4 (OM)
                             │                    │
                     PE_IDX_TEX / tcache    PE_IDX_OM / ocache
```

The pre-doctrine **pull** model (`vx_rast`/`vx_rast_begin` polled by the
shader, per-`(warp,pid,lane)` CSR latch, a sticky cross-core arbiter) has
been **retired** in favor of this push/launch path (§4).

---

## 2. ISA, opcodes, and state

- **Opcodes** (all under `custom1 = 0x2B`), decoded by funct3 in
  [`VX_decode.sv`](../../hw/rtl/core/VX_decode.sv):
  - `vx_om_export` — funct3=3, R4-type: fragment export to the OM aperture.
  - `vx_tex` — funct3=5, R4-type: texture sample; u/v/lod in registers, texel in rd.
  - `SETW`/`GETW` — funct3=6: graphics register-window write/read (funct2
    1=SETW, 3=GETW), the operand-staging primitive shared by TEX/OM/RASTER.
  - RASTER has **no kernel op** in v2 — the raster engine launches the
    fragment shader (push); there is no shader-issued raster instruction.
- **Kernel intrinsics**
  ([`sw/kernel/include/vx_graphics.h`](../../sw/kernel/include/vx_graphics.h),
  `vx_om_export(...)`, `vx_tex(...)`, and the fragment-stamp readers `vx_frag_load`,
  `vx_frag_payload`, `vx_frag_slot` (the FS reads its launched record from
  the window, keyed by the allocated slot = `CTA_BLOCK_ID_X`).
- **DCR state** ([`VX_types.toml`](../../VX_types.toml)):
  - **TEX** `0x040–0x05F`: per-stage addr / logdim / format / filter / wrap
    + mip offsets.
  - **RASTER** `0x060–0x06B`: tbuf/pbuf addrs+strides, tile_count, scissor,
    and the fragment-shader launch descriptor `FRAG_PC_LO/HI` (0x066/0x067),
    `FRAG_ENTRY`, `FRAG_PARAM` — the CP/runtime writes these so the raster
    engine can launch the FS with no host round-trip.
  - **OM** `0x080–0x092`: color/depth buffer addrs + pitches, depth
    func/writemask, full stencil state, blend mode/func/const, logic-op, and
    `EARLYZ_SAFE` (0x092) — the per-draw gate that arms early-Z.
  DCRs are broadcast to all cluster instances; each raster instance
  self-selects its tile stripe by `INSTANCE_IDX`.
- **Perf** MPM classes RASTER=12, TEX=13, OM=14.

### 2.1 The graphics register window (FF↔SIMT handoff)

FF operands and results move in **registers**: TEX takes u/v/lod and returns its
texel in rd, the OM exports through its **aperture** (an ordinary store), and a
fragment's stamp rides its **launch** and is read as CSRs. The only register window
left is the RTU's hit window (RTL
[`VX_rtu_unit.sv`](../../hw/rtl/rtu/VX_rtu_unit.sv)), which the RTU writes and
the shader reads. This satisfies the **interface law** (§1.3 of the master plan): every FF↔SIMT
value is scope-partitioned to the window, single-issue, and — critically —
handed off through **scoreboarded** registers so the op retires in order
(C1–C3), with no shared mutable side-band outliving the op (C4). `vx_om4`
and `vx_tex4` each return a scoreboard handle; RASTER delivers its launch
payload as the FS warp's window contents at launch.

---

## 3. RTL module inventory

### 3.1 RASTER ([`hw/rtl/raster/`](../../hw/rtl/raster/))

The raster **core** walks the coverage pipeline and the **dispatch** path
launches fragment work:

- **Coverage math** (fixed-point walker): `VX_raster_mem`
  (tile/prim-buffer fetch via rcache, stripe-partitioned by `INSTANCE_IDX`)
  → `VX_raster_te` (tile engine) → `VX_raster_be` (block engine) →
  `VX_raster_slice` / `VX_raster_edge` / `VX_raster_extents` (edge-function
  eval) → `VX_raster_qe` (quad engine, emits 2×2 covered-quad stamps). The
  per-sample coverage test in `VX_raster_qe` applies the **Vulkan top-left
  fill rule**: a sample lying exactly on an edge (edge value == 0) is covered
  only if that edge is a top or left edge (gradient `A>0`, or `A==0 && B>0`),
  so a shared edge between two triangles is covered by exactly one of them (no
  cracks, no double-cover). The rule is applied identically in the SimX model
  and the on-device SW-raster fallback; the conservative tile trivial-reject
  stays inclusive (`>=0`).
- **`VX_raster_earlyz`** — optional occlusion cull (P3): evaluates each
  covered pixel's screen-space plane depth (bit-identical to the FS late-Z),
  reads committed depth through the coherent ocache, and clears coverage
  bits that are **strictly behind** the read depth (the reflexive relaxation
  of the depth func — see §5.1). Gated by `VX_CFG_RASTER_EARLYZ` +
  `VX_DCR_OM_EARLYZ_SAFE`.
- **`VX_raster_packer`** — fragment warp aggregator: the walker emits waves
  of `NUM_LANES` quads (one quad/lane) from a single primitive, but a small
  triangle leaves most lanes idle (`mask=0`). The packer compacts sparse
  quads into dense fragment waves to lift shader occupancy.
- **`VX_raster_dispatch`** — per-core fragment work dispatcher: for each
  covered-quad wave it **launches one bare 1-warp fragment CTA** onto the
  core's local KMU bus (merged with the device-KMU stream by
  `VX_kmu_arb`), keyed by an allocated **slot** (not the launched warp-id);
  the per-lane payload (coverage, quad origin, pid) is seeded into the gfx
  window at launch.
- **`VX_raster_arb`** — cluster arbiter (N producers → M consumers,
  fan-in/1:1/fan-out).

`VX_raster_unit.sv` (the old per-core pull consumer with the
`is_begin`/done-sentinel protocol) has been **removed**.

### 3.2 TEX ([`hw/rtl/tex/`](../../hw/rtl/tex/))

`VX_tex_unit` (top) → `VX_tex_arb` → `VX_tex_core` (orchestrator) with the
sampler pipeline: `VX_tex_addr` ((u,v,lod) → mip address, Q-fixed) →
`VX_tex_mem` (4-texel fetch via tcache) → `VX_tex_format` (pixel-format
decode: A8R8G8B8, R5G6B5, A1R5G5B5, A4R4G4B4, A8L8, L8, A8) →
`VX_tex_sampler`/`VX_tex_lerp` (bilinear) → `VX_tex_sat`. Addressing modes
(CLAMP/REPEAT/MIRROR) in `VX_tex_wrap`; per-warp state in `VX_tex_csr`;
DCR slave `VX_tex_dcr`; per-socket interface `VX_tex_bus_if`. The `vx_tex4`
quad form computes one integer mip LOD from the 2×2 quad derivatives.

### 3.3 OM ([`hw/rtl/om/`](../../hw/rtl/om/))

`VX_om_unit` (top) → `VX_om_arb` → `VX_om_core` (orchestrator):
`VX_om_ds` (depth + stencil test/update, via `VX_om_compare` 8 depth funcs
and `VX_om_stencil_op` 8 stencil ops) → `VX_om_blend`
(`VX_om_blend_func`/`_minmax`/`_multadd`) or `VX_om_logic_op` (ROP) →
`VX_om_mem` (read-modify-write color+depth via ocache). A **same-pixel
R-M-W interlock** holds a slot until its writes commit, so a later
same-address fragment's read cannot bypass an in-flight write. `vx_om4`
submits each covered sub-pixel of the quad from the window (color at
`base..base+3`, depth at `base+4..base+7`); the OM is the **authoritative
late-Z** even when early-Z is active.

### 3.4 Cluster glue

[`VX_graphics.sv`](../../hw/rtl/VX_graphics.sv) is a real wrapper module
(kept, not inlined into `VX_cluster.sv`): it instantiates the tex/raster/om
arbiters and cores, the three caches as `VX_cache_cluster` instances, sets
each raster core's `INSTANCE_IDX`, exposes the ocache read port early-Z
uses, and fans DCRs out per unit.
[`VX_cluster.sv`](../../hw/rtl/VX_cluster.sv) carries the per-socket bus
arrays, the `gfx_busy` aggregation (so the device stays busy while raster
dispatch / packer / early-Z have work in flight), and perf aggregation.

---

## 4. Fragment dispatch v2 (RASTER → SIMT, push/launch)

The RASTER control path is a **push** model — the root fix for the
recurring multi-core / multi-drawcall dropped-draw-call class:

- **Push, not pull.** The raster math produces covered quads; the packer
  compacts them into fragment waves; the dispatcher **launches** a 1-warp
  fragment CTA per wave onto the core (via the KMU bus, merged with the
  device-KMU stream by `VX_kmu_arb`). The shader never polls: it runs
  straight-line and reads its payload from the register window with
  `vx_frag_load`/`vx_frag_payload` (C1–C3). `vx_rast` and the bcoord CSRs
  are gone.
- **Slot-keyed delivery.** Each launch is tagged with an allocated slot
  (surfaced to the FS as `CTA_BLOCK_ID_X`); the FS indexes the window's warp
  dimension by that slot, decoupling payload delivery from the physical
  warp-id (C4).
- **DCR-launched.** The FS entry PC/param ride the RASTER DCRs
  (`FRAG_PC_LO/HI`, `FRAG_ENTRY`, `FRAG_PARAM`), written by the CP/runtime,
  so the raster engine self-launches with no host round-trip and no
  device-KMU grid for fragment work.
- **Device-busy.** `gfx_busy` (cluster) + `raster_dispatch_busy` /
  `raster_packer_busy` (core) keep the device from reporting idle while a
  frame is still draining raster → shader → OM.

SimX models the same shape: `RasterCore` produces waves and the fragment
dispatch is modeled 1:1 with the RTL for trace-diffable parity.

---

## 5. Early-Z (occlusion cull)

`VX_raster_earlyz` ([`hw/rtl/raster/VX_raster_earlyz.sv`](../../hw/rtl/raster/VX_raster_earlyz.sv))
is a **read-only** occlusion stage upstream of the shader. Per covered
quad it evaluates the screen-space depth plane (the exact plane MAC + the
`*65336 >>> 24` depth-stage scale the FS uses, so the candidate depth is
bit-identical to the OM late-Z), reads committed depth through the coherent
ocache, and narrows the coverage mask; a fully-culled wave is dropped. The
ROP remains the authoritative late-Z. Gated by `VX_CFG_RASTER_EARLYZ`
(compile) + `VX_DCR_OM_EARLYZ_SAFE` (per-draw; the driver arms it only for
monotone `LESS`/`LEQUAL` with no stencil).

### 5.1 Correctness — strict-behind cull

The committed depth early-Z reads is **not causally pinned** to the
fragment: it may already contain the fragment's own eventual write, a
co-planar (equal-depth) write, or a causally-later nearer write (fragments
do not reach the OM strictly in submission order). So a covered pixel may
be dropped only when it is **strictly behind** the read depth — the
**reflexive relaxation** of the depth func (`LESS`/`LEQUAL` → cull iff
`cand > stored`; `GREATER`/`GEQUAL` → cull iff `cand < stored`; other
funcs never early-cull). A visible fragment has `cand == final-buffer depth
≤ any value early-Z reads`, so strict-behind can never cull it: enabling
early-Z is **image-identical** to the ROP-only path, independent of read
freshness or pipeline ordering. Culling on equality (testing with the exact
func) is the bug that would drop own/co-planar/final writes. The SimX model
(`earlyz_occluded`) mirrors the RTL compare (`earlyz_func` →
`VX_om_compare`) bit-for-bit.

---

## 6. SimX models

SimX ([`sim/simx/{raster,tex,om}/`](../../sim/simx/)) mirrors each unit as a
`*Core` (and, for TEX/OM, a `*Unit` SFU-PE) driving real `MemReq`/`MemRsp`
traffic against the rcache/tcache/ocache, applying the shared host-reference
primitives (`graphics::Rasterizer`, `graphics::DepthStencil`,
`graphics::Blender`) from [`sw/common/`](../../sw/common/). `raster_core.cpp`
holds the producer FSM + TE/BE walker + early-Z (`early_z_cull`); the
per-core raster consumer is header-only (`raster_unit.h`) since the pull
consumer retired. SimX is the **SimX-first** development + evaluation engine
and the correctness oracle; the RTL FF datapaths are built out (§7), with SimX
still ahead only on the few unbuilt RTL features (TEX trilinear, OM MRT).

---

## 7. State of the hardware datapaths

Per the master plan ([`../proposals/gfx_v2_true_gpu.md`](../proposals/gfx_v2_true_gpu.md) §2):

- **RASTER** — coverage math, early-Z, packer, and fragment dispatch are in
  RTL and exercised on rtlsim; the old pull consumer is deleted.
- **OM / TEX** — the **fixed-point datapaths are built out in RTL** and run on
  rtlsim (`VX_om_core`: mem-RMW → depth/stencil → blend + folded logic-op;
  `VX_tex_core`: addr → mem → format-decode (7 formats) → bilinear; `vx_tex4`
  quad = LZC integer-mip LOD). They are **not stubs**. The remaining RTL deficits
  are specific advanced features — **TEX trilinear** (integer-mip + bilinear only
  today) and **OM MRT** (single color/depth target) — plus **proving SimX↔RTL
  byte-exact parity** on the `graphics_parity` matrix. SimX stays the fuller model
  where those features are unbuilt (it does trilinear), so it remains the oracle
  for them.
- **Conformance** — no Vulkan CTS harness on hardware yet.

So the critical path to FF acceleration on the U55C is **parity-proof +
trilinear/MRT**, not building the datapaths.

The FF invariant holds: **no floating-point datapath inside any FF unit**
(fixed-point, mobile-class). Anything the FF units cannot represent (exotic
formats, blend/logic-op modes, MSAA resolve) is served by the on-device
SIMT software fallback (`sw/gfx/libgfx_sw.mk`, `gfx_sw_abi.cpp`), never by
the host.

---

## 8. VM / pinned-buffer tie-in

Under `VX_CFG_VM_ENABLE` the per-core MMU translates VA→PA for kernel LSU
traffic, but the RASTER/TEX/OM AXI masters **bypass** the MMU and use the
physical addresses written into their DCRs. `VX_MEM_PHYS` buffers are
identity-mapped and carved from a dedicated pinned slab so VA == PA. DCR
writes targeting graphics buffer-address registers are validated against
the pinned slab on the CP submit path (returning `VX_ERR_INVALID_VALUE`
for a PA outside the slab). The slab size is `VX_CFG_VM_PINNED_REGION_SIZE`
(overridable via `VORTEX_VM_PINNED_SIZE`). Tests allocate every HW-bound
buffer with `VX_MEM_PHYS` and omit it for write-only LSU buffers. The
VM/MMU subsystem is documented in
[`virtual_memory_subsystem.md`](virtual_memory_subsystem.md).

---

## 9. Relationship to the true-GPU plan

This document describes the **hardware** the master plan
([`../proposals/gfx_v2_true_gpu.md`](../proposals/gfx_v2_true_gpu.md))
schedules against its north star (Vulkan CTS on the U55C at 4 cores,
on-device, FF-accelerated). The dual-path principle (FF fast path +
mandatory on-device SIMT software fallback), the C1–C5 interface law, and
the push/launch dispatch redesign all originate there; the FF unit
microarchitecture, ISA surface, and dispatch/early-Z hardware are here. The
software side — the vortexpipe driver, the on-device front end
(setup + bin-sort), and the CP orchestration — is in
[`graphics_software_stack.md`](graphics_software_stack.md),
[`vortexpipe_architecture.md`](vortexpipe_architecture.md), and
[`command_processor.md`](command_processor.md).

**Superseded / rejected directions** (recorded to avoid revival): the
`vx_rast` pull + `pos_mask==0` sentinel + per-`(warp,pid,lane)` CSR latch
dispatch protocol (replaced by push/launch, §4); the cocogfx dependency
(eliminated in favor of `sw/common/gfx_render.cpp`); inlining
`VX_graphics.sv` into `VX_cluster.sv` (the wrapper was kept); and
reset-clean DCRs (rejected for the BRAM cost).

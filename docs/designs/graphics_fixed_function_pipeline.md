# Graphics Fixed-Function Hardware (TEX / RASTER / OM) — Design

**Scope:** the Vortex fixed-function graphics units — the texture
sampler (TEX), the rasterizer (RASTER), and the output-merger / ROP (OM).
Covers the RTL ([`hw/rtl/tex/`](../../hw/rtl/tex/),
[`hw/rtl/raster/`](../../hw/rtl/raster/), [`hw/rtl/om/`](../../hw/rtl/om/),
[`hw/rtl/VX_graphics.sv`](../../hw/rtl/VX_graphics.sv)), the SimX models
([`sim/simx/tex/`](../../sim/simx/tex/),
[`sim/simx/raster/`](../../sim/simx/raster/),
[`sim/simx/om/`](../../sim/simx/om/)), and the SW surface.

This document covers the **hardware microarchitecture, ISA, scheduling,
and VM tie-in**. The complementary **software / compiler / rendering
pipeline** (the vortexpipe Gallium driver, NIR→Vortex lowering,
`vkCmdDraw` flow) is documented separately in
[`vortexpipe_architecture.md`](vortexpipe_architecture.md).

The three units are RISC-V ISA extensions: `MISA` bits TEX=6, RASTER=7,
OM=8 ([`VX_config.toml:305`](../../VX_config.toml#L305)), each gated by
`VX_CFG_EXT_{TEX,RASTER,OM}_ENABLE`.

---

## 1. Architecture overview

All three units attach to the SIMT cores as **SFU processing elements**
([`VX_sfu_unit.sv:67-81`](../../hw/rtl/core/VX_sfu_unit.sv#L67),
`PE_IDX_{TEX,OM,RASTER}`) and are **cluster-shared** (not per-core),
consuming per-socket bus interfaces. Each has a dedicated cluster-level
cache: tcache (textures), rcache (raster tile/prim buffers), ocache
(color + depth framebuffers). The cluster wrapper
[`VX_graphics.sv`](../../hw/rtl/VX_graphics.sv) instantiates the per-unit
arbiters, cores, caches, and DCR fan-out.

```
   fragment-shader kernel (SIMT)
        │  vx_rast_begin / vx_rast       vx_tex            vx_om
        ▼                                  ▼                 ▼
   VX_sfu_unit  ── PE_IDX_RASTER ──┐  PE_IDX_TEX ──┐  PE_IDX_OM ──┐
                                   ▼               ▼              ▼
   per-socket raster bus      per-socket tex   per-socket om
        │                          │                │
   VX_raster_arb (cluster)    VX_tex_arb       VX_om_arb
        │                          │                │
   VX_raster_core             VX_tex_core      VX_om_core
   (tile→block→quad)          (addr→sample)    (depth/stencil→blend→RMW)
        │                          │                │
      rcache                     tcache           ocache
```

---

## 2. ISA, opcodes, and state

- **Opcodes** (all under `INST_EXT2 = 0x2B` / RISC-V `custom1`,
  [`VX_gpu_pkg.sv:276`](../../hw/rtl/VX_gpu_pkg.sv#L276)):
  `INST_SFU_TEX = 0xB`, `INST_SFU_OM = 0xC`, `INST_SFU_RASTER = 0xD`
  ([`VX_gpu_pkg.sv:508-514`](../../hw/rtl/VX_gpu_pkg.sv#L508)), decoded by
  funct3 (1=tex, 2=om, 3=rast, 4=rast_begin) at
  [`VX_decode.sv:733-775`](../../hw/rtl/core/VX_decode.sv#L733).
- **Kernel intrinsics**
  ([`sw/kernel/include/vx_graphics.h`](../../sw/kernel/include/vx_graphics.h)):
  `vx_tex(stage,u,v,lod)` ([`:49`](../../sw/kernel/include/vx_graphics.h#L49)),
  `vx_om(x,y,face,color,depth)` ([`:58`](../../sw/kernel/include/vx_graphics.h#L58)),
  `vx_rast()` ([`:65`](../../sw/kernel/include/vx_graphics.h#L65)),
  `vx_rast_begin()` ([`:77`](../../sw/kernel/include/vx_graphics.h#L77)).
- **DCR state** ([`VX_types.toml`](../../VX_types.toml)): TEX `0x020–0x03F`
  (stage/addr/logdim/format/filter/wrap + 15 mip offsets,
  [`:126-137`](../../VX_types.toml#L126)); RASTER `0x040–0x045`
  (tbuf/tile_count/pbuf/pbuf_stride/scissor,
  [`:244-253`](../../VX_types.toml#L244)); OM `0x060–0x071` (color/depth
  buffer addrs, pitches, depth-func/writemask, full stencil state,
  blend-mode/func/const, logic-op, [`:255-276`](../../VX_types.toml#L255)).
  DCRs are broadcast to all cluster instances; each raster instance
  self-selects its tile stripe.
- **Perf** MPM classes TEX=3, RASTER=4, OM=5
  ([`VX_types.toml:392-394`](../../VX_types.toml#L392)); reported via
  [`legacy_perf.cpp:229-231`](../../sw/runtime/common/legacy_perf.cpp#L229).
- **Counts** ([`VX_config.toml`](../../VX_config.toml)): `NUM_TEX_CORES`,
  `NUM_RASTER_CORES`, `NUM_OM_CORES`, and `NUM_{TCACHES,RCACHES,OCACHES}`.

---

## 3. RTL module inventory

### 3.1 TEX ([`hw/rtl/tex/`](../../hw/rtl/tex/))

`VX_tex_unit` (top) → `VX_tex_arb` → `VX_tex_core` (orchestrator) with the
sampler pipeline: `VX_tex_addr` ((u,v,lod) → mip address, Q-fixed) →
`VX_tex_mem` (4-texel fetch via tcache) → `VX_tex_format` (pixel-format
decode: A8R8G8B8, R5G6B5, A1R5G5B5, A4R4G4B4, A8L8, L8, A8) →
`VX_tex_sampler`/`VX_tex_lerp` (bilinear) → `VX_tex_sat`. Addressing modes
(CLAMP/REPEAT/MIRROR) in `VX_tex_wrap`; per-warp CSR state in
`VX_tex_csr`; DCR slave `VX_tex_dcr`.

### 3.2 RASTER ([`hw/rtl/raster/`](../../hw/rtl/raster/))

`VX_raster_unit` (per-core consumer; splits `vx_rast`/`vx_rast_begin` on
`op_args.raster.is_begin`,
[`VX_raster_unit.sv:52`](../../hw/rtl/raster/VX_raster_unit.sv#L52); writes
0 to dest on `done`, [`:82`](../../hw/rtl/raster/VX_raster_unit.sv#L82)).
`VX_raster_core` (producer) walks the pipeline `VX_raster_mem`
(tile/prim-buffer fetch via rcache, stripe-partitioned by
`INSTANCE_IDX`/`NUM_INSTANCES`) → `VX_raster_te` (tile engine) →
`VX_raster_be` (block engine) → `VX_raster_slice`/`VX_raster_edge`
(edge-function eval) → `VX_raster_qe` (quad engine, emits 2×2 stamps).
`VX_raster_arb` is the cluster arbiter (see §5).

### 3.3 OM ([`hw/rtl/om/`](../../hw/rtl/om/))

`VX_om_unit` (top) → `VX_om_arb` → `VX_om_core` (orchestrator):
`VX_om_ds` (depth + stencil test/update, via `VX_om_compare` 8 depth funcs
and `VX_om_stencil_op` 8 stencil ops) → `VX_om_blend`
(`VX_om_blend_func`/`_minmax`/`_multadd`) or `VX_om_logic_op` (ROP) →
`VX_om_mem` (read-modify-write color+depth via ocache).

### 3.4 Cluster glue

[`VX_graphics.sv`](../../hw/rtl/VX_graphics.sv) is a real wrapper module
(it was **kept**, not inlined into `VX_cluster.sv`): it instantiates the
tex/raster/om arbiters and cores, the three caches as
`VX_cache_cluster` instances, sets each raster core's
`INSTANCE_IDX = CLUSTER_ID*NUM_RASTER_CORES+i`
([`:258-259`](../../hw/rtl/VX_graphics.sv#L258)), and fans DCRs out per
unit. [`VX_cluster.sv`](../../hw/rtl/VX_cluster.sv) carries the
`per_socket_{tex,raster,om}_bus_if` arrays and perf aggregation.

---

## 4. SimX models and SW

SimX ([`sim/simx/{tex,raster,om}/`](../../sim/simx/)) mirrors each unit as
a `*Unit`/`*Core` pair driving real `MemReq`/`MemRsp` traffic against the
tcache/rcache/ocache, applying the shared host-reference primitives
(`graphics::Rasterizer`, `graphics::DepthTencil`, `graphics::Blender`).
Bus arbiters are `TxRxArbiter<Req,Rsp>` templates
([`sim/simx/types.h:1626`](../../sim/simx/types.h#L1626)).

SW: host-side triangle binning is
[`sw/runtime/graphics.cpp`](../../sw/runtime/graphics.cpp) `Binning()`
([`:160`](../../sw/runtime/graphics.cpp#L160)) — triangle setup, edge
equations, Q-fixed conversion, tile coverage → primbuf + tilebuf, with
**no cocogfx dependency** ([`:14-17`](../../sw/runtime/graphics.cpp#L14)).
The on-wire ABI is single-sourced in
[`sw/common/vx_gfx_abi.h`](../../sw/common/vx_gfx_abi.h) (`fixed_t<F>`,
`rast_prim_t`, `rast_tile_header_t`, 8888 pixel helpers); the host
reference renderer is [`sw/common/gfx_render.cpp`](../../sw/common/gfx_render.cpp).

---

## 5. Raster work scheduling

`VX_raster_arb` ([`hw/rtl/raster/VX_raster_arb.sv`](../../hw/rtl/raster/VX_raster_arb.sv))
handles N producers → M consumers in all three relative sizes (fan-in
N>M, 1:1, **fan-out N<M** via `IS_FANOUT`,
[`:55`](../../hw/rtl/raster/VX_raster_arb.sv#L55)) with per-output sticky
`consumer_served[o]` ([`:76,229-261`](../../hw/rtl/raster/VX_raster_arb.sv#L76)),
a `done_all` gate ([`:123-128`](../../hw/rtl/raster/VX_raster_arb.sv#L123)),
and per-frame flush keyed on the first `begin_pulse` once `frame_drained`
([`:77-106`](../../hw/rtl/raster/VX_raster_arb.sv#L77)). Cross-cluster work
is striped by `INSTANCE_IDX`. The fragment kernel calls `vx_rast_begin()`
once (idempotent per-frame trigger via the producer's `fetch_triggered`
latch), then loops `vx_rast()` until it returns an empty mask (the `done`
drain sentinel).

---

## 6. End-to-end draw flow

1. Host (or vortexpipe/Mesa) runs `Binning()` to produce primbuf (edge
   equations + attribute deltas in Q-fixed) and tilebuf (tile headers +
   per-tile primitive-ID lists), allocates tex/color/depth buffers with
   `VX_MEM_PHYS`, and programs the RASTER/TEX/OM DCRs with their physical
   addresses.
2. The fragment-shader kernel calls `vx_rast_begin()` once, then loops
   `pos_mask = vx_rast()`; the cluster's raster core feeds quad stamps
   back through the arbiter to the per-core `VX_raster_unit`.
3. Per fragment the kernel calls `vx_tex(...)` (sample) and
   `vx_om(...)` (depth/stencil + blend RMW to the framebuffer).

### 6.1 VM / pinned-buffer tie-in

Under `VX_CFG_VM_ENABLE` the per-core MMU translates VA→PA for kernel LSU
traffic, but the TEX/RASTER/OM AXI masters **bypass** the MMU and use the
physical addresses written into their DCRs. `VX_MEM_PHYS` buffers are
identity-mapped ([`vm.cpp:install_identity_map`](../../sw/runtime/common/vm.cpp))
and carved from a dedicated pinned slab
([`device.cpp:61-77`](../../sw/runtime/common/device.cpp#L61)) so VA == PA.
DCR writes targeting graphics buffer-address registers are validated
against the pinned slab on the CP submit path
([`device.cpp:400-428`](../../sw/runtime/common/device.cpp#L400)),
returning `VX_ERR_INVALID_VALUE` for a PA outside the slab. The slab size
is `VX_CFG_VM_PINNED_REGION_SIZE` (256 MB default,
[`VX_config.toml:163`](../../VX_config.toml#L163)), overridable via
`VORTEX_VM_PINNED_SIZE`. Tests allocate every HW-bound buffer with
`VX_MEM_PHYS` and (correctly) omit it for write-only LSU buffers.

---

## 7. Proposed but not yet implemented

1. **`vx_device_query(VX_CAPS_VM_PINNED_SIZE / _FREE)`** — no
   `VX_CAPS_VM_PINNED_*` symbol exists yet; needed for Mesa/HIP
   suballocators to plan around the pinned budget
   (`gfx_vm_pinned_buffers_proposal`).
2. **SimX raster-arbiter parity.** The RTL `VX_raster_arb` has a
   per-output sticky-done / frame-drained fan-out state machine; the SimX
   `RasterBusArbiter` is a generic `TxRxArbiter(N,1)` fan-in, with
   done-drain semantics living in `RasterCore`/`fetch_triggered` instead.
   The gfx suite passes on SimX, but the strict 1:1 RTL↔SimX module mirror
   the proposal demanded is only partially realized
   (`raster_scheduling_completion_proposal` §8.4/§8.6).
3. **`vx_buffer_reserve` explicit pinned-vs-global PA routing** — the
   `mem_alloc` dispatch exists; the caller-chosen-PA reserve branch is
   unconfirmed (`gfx_vm_pinned_buffers_proposal`).
4. **Per-block-header cross-reference docstrings** for `VX_MEM_PHYS`
   (`gfx_vm_pinned_buffers_proposal` R2) — partial.
5. **gfx-v2 fixed-function unit roadmap** (from `vulkan_support_proposal`
   §5 — a large catalog with **zero implementation today**; the units are
   still gfx-v1). For Vulkan-class workloads: TEX mip/LOD beyond LOD-0,
   formats beyond A8R8G8B8, anisotropic filtering, compressed textures,
   bindless; quad-rate `vx_tex4` / `vx_om4` intrinsics; OM multiple render
   targets (MRT) and MSAA; RASTER Hi-Z / early-Z; and native floating-point
   inside the units (relaxing the gfx-v1 fixed-point invariant). The
   conformance-gap audits in
   [`vortexpipe_architecture.md`](vortexpipe_architecture.md) §3.6–§3.8
   enumerate the current per-unit limits these would lift.

**Superseded / rejected directions** (recorded to avoid revival): the
cocogfx dependency (eliminated in favor of `sw/common/gfx_render.cpp`);
inlining `VX_graphics.sv` into `VX_cluster.sv` (the wrapper was kept);
the `tex_smoke`/`raster_smoke`/`om_smoke` standalone suites (replaced by
the full PNG `tests/graphics/gfx_*` suite with real `.cgltrace` assets
running on simx+rtlsim+xrt+opae); and reset-clean DCRs (rejected for the
BRAM cost). Note: the `gfx_migration` phases recorded as "deferred"
(SimX rewrite, full PNG tests) are in fact **done**.

---

## 8. Source proposals

This design consolidates and supersedes the following proposals (now
removed from `docs/proposals/`): `gfx_migration_proposal.md`,
`raster_scheduling_completion_proposal.md`,
`gfx_vm_pinned_buffers_proposal.md`.

The software rendering pipeline is documented in
[`vortexpipe_architecture.md`](vortexpipe_architecture.md); the VM/MMU
subsystem the pinned-buffer model relies on is in
[`virtual_memory_subsystem.md`](virtual_memory_subsystem.md).

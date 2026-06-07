# gfx_v2 — Full-Residency Memory Model & Allocator

**Scope:** the device-resident memory model for a frame — how the entire
working set (resources, intermediates, attachments) is laid out once in device
memory and never spills to host, and the allocator that manages it. Defines the
two-heap split (FF-pinned-PA vs shader-paged-VA), lifetime scopes, the tiling
pool, and the `VX_CAPS_VM_PINNED_*` budget query gfx_v2 requires.
**Reference:** Vulkan memory heaps / VMA suballocation; DX12 residency
(`MakeResident`/`Evict`).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.6.
**Date:** 2026-06-07.
**Related:** [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md),
[virtual_memory_subsystem.md](../designs/virtual_memory_subsystem.md),
[command_processor_control_plane.md](../designs/command_processor_control_plane.md),
[gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md).

---

## 1. Motivation

Charter pillar 4: every buffer stays device-resident across the whole frame;
the only egress is the final framebuffer at present. Two facts make this a
concrete allocator problem rather than a slogan:

1. **The fixed-function units bypass the MMU.** RASTER/TEX/OM AXI masters use
   raw physical addresses from their DCRs
   ([graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md) §6.1),
   validated against the **pinned slab** on the CP submit path
   ([device.cpp:395-428](../../sw/runtime/common/device.cpp#L395) — a non-pinned
   FF address is rejected `VX_ERR_INVALID_VALUE`). So every FF-touched buffer
   **must** live in the identity-mapped `VX_MEM_PHYS` pinned region
   ([device.cpp:55-90](../../sw/runtime/common/device.cpp#L55)).
2. **That region is a fixed budget.** `pinned_mem_` is carved from
   `global_mem_` at `ALLOC_BASE_ADDR`, sized by `resolve_pinned_size()`
   (`VX_CFG_VM_PINNED_REGION_SIZE`, 256 MB default; `VORTEX_VM_PINNED_SIZE`
   override). A real Vulkan frame's FF working set (attachments + textures +
   tiling pool) can exceed 256 MB, so the allocator must **plan residency
   within a known budget** — which requires a way to *query* that budget.

---

## 2. Two resident heaps

Everything is resident; the split is by **who addresses it**, mirroring Vulkan
memory heaps:

| Heap | Backing | Addressing | Holds |
|------|---------|-----------|-------|
| **Pinned-PA** (`pinned_mem_`) | identity-mapped slab (VA == PA) | physical | anything an FF master touches: color/depth/stencil attachments, textures, the bin-sort tiling buffers (`primbuf`, key array, `sorted_pids`, `bin_headers`), draw-context |
| **Paged-VA** (`global_mem_`) | paged pool above the slab | virtual (MMU) | shader-only buffers the cores touch but no FF unit does: SSBO/UBO, descriptor data, scratch |

Both are device-resident and **non-spilling**. Allocation is routed by a
**usage flag** (FF-bound → pinned-PA; shader-only → paged-VA) — the analog of
Vulkan memory-type selection. Because pinned buffers are **identity-mapped
(VA == PA)**, the SIMT cores (VA, through the MMU) and the FF masters (PA,
bypassing it) see the *same address* for a pinned buffer — that identity is the
substrate for the shared producer→FF handoff (binning kernels write `primbuf`
by VA; RASTER reads it by the same PA).

---

## 3. What must be pinned

Every buffer on the FF data path:
- **Attachments** — color, depth, stencil; written by OM, accumulate across
  draws (charter pillar 4), resident for the whole render pass.
- **Textures** — sampled by TEX; resident for their lifetime.
- **Bin-sort tiling buffers** — `primbuf`, key array, `sorted_pids`,
  `bin_headers`; produced by the cores, **read by RASTER** → pinned.
- **draw-context** — the dynamic-count block
  ([gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md) §4.2);
  read by RASTER for `tile_count`.

VS output is read by the setup kernel (cores), not an FF master, so it *may*
live in paged-VA — but is simplest kept pinned alongside the tiling pool.

---

## 4. Lifetime scopes & suballocator

A frame-scoped suballocator over each heap, three scopes:

| Scope | Lifetime | Allocator | Examples |
|-------|----------|-----------|----------|
| **Persistent** | many frames | free-list / buddy | textures, static VBO/IBO, BVH/AS, pipeline constants |
| **Per-pass** | one render pass | linear / ring | attachments, VS output, draw-context |
| **Per-draw transient** | one draw, reset+reused | bump/ring (the tiling pool, §5) | `primbuf`, keys, `sorted_pids`, `bin_headers` |

`pinned_mem_` already provides page/block-granular allocation
([device.cpp:70-74](../../sw/runtime/common/device.cpp#L70)); the gfx residency
allocator layers this scoped discipline on top so transients churn cheaply
(bump-reset) while persistent resources don't fragment.

---

## 5. The tiling pool

The binning intermediates are a **per-draw transient** reserved once per pass
as a bump region in pinned-PA, high-water-sized, and **reset between draws**
(no per-draw alloc/free churn). This is the reserved pool that makes the CP
command list static
([gfx_v2_cp_graphics_frontend.md](gfx_v2_cp_graphics_frontend.md) §4.1): the
binning kernels write into fixed pool addresses; the RASTER DCRs point at the
pool base. Sizing = a configured high-water mark of `(120·V + ~16·P)`; overflow
policy in §7.

---

## 6. `VX_CAPS_VM_PINNED_SIZE` / `_FREE` (already available)

The residency planner needs the pinned budget and current free space. This query
**already exists** in the runtime — `VX_CAPS_VM_PINNED_SIZE`/`_FREE`
(`0x10`/`0x11`, [vortex2.h:75-76](../../sw/runtime/include/vortex2.h#L75)),
served by `vx_device_query`
([device.cpp:736-737](../../sw/runtime/common/device.cpp#L736)) from the
host-tracked `pinned_size_` and the `pinned_mem_` free count. Because
device-memory allocation is host-side bookkeeping (the CP DMAs to whatever
addresses the runtime hands out), these are known on **every backend** including
FPGA — no CP-regfile capability is needed. (The §7.1 "deferred" note in the
design docs is stale; the query landed.)

gfx_v2 therefore simply **uses** the query to decide, *before* a frame, whether
the working set fits on-device and to lay it out without consulting the host
mid-frame; the Mesa/HIP suballocators consume the same query (the original §7.1
motivation).

---

## 7. The no-spill invariant & overflow policy

**Invariant:** a frame's resident set is laid out once and never spills to
host. Overflow is handled **device-side**, in priority order:

1. **Tiling-pool overflow** (transient `P`/`V` exceed the high-water): the
   binning's device-side fallback — route the overflowing draw to the SIMT
   software path ([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md)) or
   flush+re-bin in smaller batches (CUDARaster's restart, now device-driven).
2. **Persistent oversubscription** (textures + attachments exceed the slab):
   this is residency oversubscription. Baseline = **fail the frame / require
   the working set to fit** (the driver sized it via §6); GPU-style
   **residency paging** (evict/restore, DX12 `MakeResident`) is a future item
   (§9). Never a silent host spill.

The pinned budget is therefore a real, surfaced constraint, and the bin-sort
schema's tight footprint ([gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) §5)
directly extends how large a frame fits.

---

## 8. Runtime / driver changes

- **Usage-flag routing** in the buffer allocator: FF-bound → `pinned_mem_`;
  shader-only → `global_mem_`. (Today `VX_MEM_PHYS` selects pinned explicitly;
  gfx_v2 derives it from usage so the upper layers don't hand-flag.)
- **Scoped suballocator** (persistent/per-pass/per-draw) over each heap.
- **`VX_CAPS_VM_PINNED_*`** — already in the runtime (§6); the driver just
  queries it for residency planning (no new caps plumbing).
- **Mesa/HIP** suballocators query the caps to plan around the budget (the
  original §7.1 motivation).

---

## 9. Validation & phasing

1. **Caps query** (`VX_CAPS_VM_PINNED_SIZE/_FREE`) — already present (§6);
   confirm it drives residency planning on every backend.
2. **Usage-routed allocator + scoped suballocator**, validated by the gfx suite
   running fully resident (no per-draw upload/readback).
3. **Tiling-pool reset/reuse** across multi-draw passes; overflow → SW-path
   fallback (§7.1).
4. **Residency planning** in vortexpipe; large-frame fit/fail behavior.

---

## 10. Open items

- **Residency paging** (oversubscription) — evict/restore for working sets
  larger than the slab; the real-GPU answer, deferred past baseline.
- **Heap sizing defaults** — 256 MB is small for real frames; pick a gfx-aware
  default and document the `VORTEX_VM_PINNED_SIZE` knob's frame-size
  implications.
- **Fragmentation / defrag** of the persistent pinned pool across long
  sessions.
- **VM-disabled path** — without `VX_CFG_VM_ENABLE` all of `global_mem_` is
  physical and the two-heap split collapses to one; confirm the allocator
  degrades cleanly.

# Pinned graphics buffers under VM mode

## Problem

When `VX_CFG_VM_ENABLE` is on, the per-core MMU translates VA→PA on every
LSU access from kernel code. The fixed-function graphics blocks
(TEX, RASTER, OM) and the command processor each have their own AXI
master that bypasses the per-core MMU — they read/write device memory
using the **physical** address the runtime writes into their DCR(s).
A graphics buffer therefore has to satisfy two readers at once: the
kernel (which sees a VA) and a HW block (which sees a PA).

The runtime already exposes a flag for exactly this case:

    VX_MEM_PHYS  (sw/runtime/include/vortex2.h:127)

When `VX_MEM_PHYS` is set the runtime allocates a PA, **identity-maps**
it (installs a PTE so VA == PA), and returns the PA in `out_addr`.
The same number is then valid for both the MMU-routed path and the
HW-bypass path. Without the flag the buffer gets a fresh VA, the kernel
reaches it through the MMU, but the HW block — programmed with that VA
via a DCR — sees an unmapped or wrong PA.

Goal: make the pinned-memory contract explicit, audit every gfx test
against it, codify a single pattern, and add a runtime check that catches
violations instead of letting them silently corrupt under VM.

## HW-DCR address registers (single source of truth)

Every DCR below is programmed with a memory address that is consumed by
a device-side master that does **not** go through the per-core MMU.
Any buffer whose address ends up here MUST be allocated with
`VX_MEM_PHYS` under VM.

| DCR                      | Block  | Direction | Notes |
|--------------------------|--------|-----------|-------|
| `VX_DCR_TEX_ADDR`        | TEX    | R         | mip-base for the active texture stage |
| `VX_DCR_RASTER_TBUF_ADDR`| RASTER | R         | tile buffer (binning index) |
| `VX_DCR_RASTER_PBUF_ADDR`| RASTER | R         | primitive buffer (triangle setup) |
| `VX_DCR_OM_CBUF_ADDR`    | OM     | RW        | color buffer (framebuffer) |
| `VX_DCR_OM_ZBUF_ADDR`    | OM     | RW        | depth buffer |
| `VX_DCR_DXA_DESC_BASE_*` | DXA    | RW        | per-slot DMA descriptor base |
| `VX_DCR_KMU_STARTUP_ADDR0/1` | KMU | R         | already PA by construction (resolved from ELF entry) — out of scope |

The CP command ring lives in host memory (`VX_MEM_HOST`) and is
managed separately; the kernel arg blob is staged by the runtime into
a per-launch scratch slot whose address is already a PA — also out of
scope here.

## Audit: current gfx regression tests

Result of `grep vx_buffer_create tests/regression/gfx_*/main.cpp`:

| Test              | Buffer           | HW reader/writer | Flag(s)                              | Verdict |
|-------------------|------------------|------------------|--------------------------------------|---------|
| `gfx_om`          | `depth_buffer`   | OM               | `READ_WRITE | PHYS`                  | ✓       |
| `gfx_om`          | `color_buffer`   | OM               | `READ_WRITE | PHYS`                  | ✓       |
| `gfx_tex`         | `src_buffer`     | TEX              | `READ | PHYS`                        | ✓       |
| `gfx_tex`         | `dst_buffer`     | kernel (LSU)     | `WRITE`                              | ✓ (no HW reader; LSU-only is correct) |
| `gfx_draw3d`      | `tile_buffer`    | RASTER           | `READ | PHYS`                        | ✓       |
| `gfx_draw3d`      | `prim_buffer`    | RASTER           | `READ | PHYS`                        | ✓       |
| `gfx_draw3d`      | `tex_buffer`     | TEX              | `READ | PHYS`                        | ✓       |
| `gfx_draw3d`      | `depth_buffer`   | OM               | `READ_WRITE | PHYS`                  | ✓       |
| `gfx_draw3d`      | `color_buffer`   | OM               | `READ_WRITE | PHYS`                  | ✓       |
| `gfx_raster`      | `tile_buffer`    | RASTER           | `READ | PHYS`                        | ✓       |
| `gfx_raster`      | `prim_buffer`    | RASTER           | `READ | PHYS`                        | ✓       |
| `gfx_raster`      | `color_buffer`   | kernel (LSU)     | `WRITE`                              | ✓ (no HW reader; LSU-only is correct) |

The current tests are correct. The gap is that this correctness is
held together by convention and code review — there is no API affordance,
no compile-time check, and no runtime check that would catch a
regression. The `vm` regression suite hit a separate, unrelated
compile error in `gfx_draw3d` (header collision on `vortex::graphics::fixed_t`
between `sw/kernel/include/vx_graphics.h` and `sw/common/gfx_render.h`)
that prevented the tests from running at all under VM — but that work
is already in flight via the `sw/common/vx_gfx_abi.h` consolidation
visible in the source-tree WIP. This proposal does not duplicate that.

## Proposed changes

### Runtime (sw/runtime)

`vortex2.h` is kept minimal by design (Vulkan/CUDA/Metal-style: flags,
not aliases). The HW-bound idiom stays as the literal
`VX_MEM_READ_WRITE | VX_MEM_PHYS` pair at the call site — no helper
wrapper in the public runtime header, no helper wrapper in the test
harness either.

**R1. Add a debug-build validation in `vx_dcr_write`.**
When the device has VM enabled and the DCR being written is one of
the addresses in the table above, the runtime resolves the value back
to a `vx_buffer_h` (the existing `mem_free` path already maintains a
PA→buffer map indirectly via `global_mem_`) and asserts that the
buffer was created with `VX_MEM_PHYS`. On mismatch, return
`VX_ERR_INVALID_VALUE` (release builds) or abort with a clear message
(debug builds). This catches future tests that forget the flag.

Concrete site: `sw/runtime/common/device.cpp::Device::dcr_write`.
A switch on `addr` against `VX_DCR_{TEX,RASTER,OM,DXA}*_ADDR*`,
guarded by `if (vm_mgr_)`, looks up the buffer via the device's
existing handle table, and checks the cached flag bits stored on the
`Buffer` object. If `(flags & VX_MEM_PHYS) == 0`, fail.

This requires `Buffer` to remember its creation flags. A 32-bit field
is enough.

**R2. Document the contract in the public header.**
Expand the `VX_MEM_PHYS` docstring (currently 9 lines in
`vortex2.h`) to enumerate the DCRs and reference the audit table
above. Cross-reference from each per-block header
(`sw/runtime/include/graphics.h`, `sw/runtime/include/dxa.h`,
`sw/runtime/include/tensor.h`) at the point where the consumer
programs a DCR. Docstring-only change — no new symbols, no new
macros, `vortex2.h` stays minimal.

**R3. Pre-allocate a dedicated pinned region (see §"Pre-allocated
pinned region" below).** The current allocator works correctly but
fragments the VA pool; carving a fixed slab keeps PHYS allocations
predictable and observable.

### Tests (tests/regression/gfx_*)

**T1. Buffers programmed into a HW DCR must be allocated with
`VX_MEM_PHYS`.** Tests call `vx_buffer_create(device, sz,
mode | VX_MEM_PHYS, &buf)` directly. No helper wrapper — the bit
pattern is explicit at the call site and a `grep VX_MEM_PHYS
tests/regression` enumerates the contract surface.

The two `VX_MEM_WRITE`-only buffers (`gfx_tex::dst_buffer`,
`gfx_raster::color_buffer`) deliberately omit `VX_MEM_PHYS`: kernel-
written via LSU only, no HW reader, no pinning needed. A short comment
at each of those call sites explaining the asymmetry prevents future
"make it match" refactors from breaking them.

**T2. Per-test regression notes.**
For each gfx test, add a one-paragraph comment at the top of `main.cpp`
listing which buffers are pinned and why, so the reviewer doesn't have
to grep for `VX_MEM_PHYS` to understand the binding contract.

## Pre-allocated pinned region

### Why pre-allocate at all?

The runtime is correct today — `VMManager::install_identity_map`
([sw/runtime/common/vm.cpp:332](sw/runtime/common/vm.cpp#L332)) reserves
the matching VA range in `virtual_mem_` as a side effect of every
PHYS allocation, so a later non-PHYS allocation's `virtual_mem_->allocate`
cannot mint a VA that collides with a previously pinned PA.

But the policy "all allocations come from one pool, PHYS allocations
incidentally also reserve a VA hole" has real downsides:

1. **VA fragmentation.** Each PHYS buffer punches a hole in the VA
   pool at an arbitrary address (whatever PA the underlying allocator
   handed back). Over the lifetime of a long-running app — e.g.
   mesa-vortex serving a sequence of frames — `virtual_mem_` ends up
   Swiss-cheesed even though the actual memory pressure is moderate.
2. **No PA-range predictability.** A debug tool reading `VX_DCR_OM_CBUF_ADDR`
   from a register dump has to walk the runtime's allocation log to
   know whether the value is plausible. Stamping pinned buffers into
   a known region ("anything below 0x10000000 is a pinned HW buffer")
   makes triage trivial.
3. **No early-OOM signal.** If a workload's pinned footprint exceeds
   what the device can host, today it fails at the Nth `vx_buffer_create`
   call deep inside a render loop. With a pre-allocated region the
   failure surfaces at `vx_device_open` (size check) or at the first
   over-quota allocation — well-defined and actionable.
4. **Performance affinity.** On platforms with non-uniform device
   memory (banked DRAM, HBM stacks, scratchpad-near-HW), a contiguous
   pinned region can be placed on the bank closest to the graphics
   blocks. Today's interleaved layout precludes this.

### Proposed layout

```
device PA space:

  0                                                   GLOBAL_MEM_SIZE
  |                                                                 |
  +------------+--------------+-------------------------------------+
  | reserved   | PINNED       | PAGED                               |
  | (low sys,  | (identity-   | (kernel allocs: PA from this        |
  | page-     |  mapped HW    |  region, VA minted from the same    |
  |  table   ) |  buffers)    |  region — never collides with       |
  +------------+--------------+ PINNED because PINNED is excluded   |
  ^            ^              | from both global_mem_ and           |
  |            |              | virtual_mem_'s free lists)          |
  0          USER_BASE        +-------------------------------------+
                              ^
                              USER_BASE + PINNED_SIZE
```

Concretely:
- At `Device` construction, `global_mem_` is initialized over
  `[USER_BASE + PINNED_SIZE, GLOBAL_MEM_SIZE)`.
- A second allocator `pinned_mem_` is initialized over
  `[USER_BASE, USER_BASE + PINNED_SIZE)`.
- `Device::mem_alloc` dispatches: `(flags & VX_MEM_PHYS) ?
  pinned_mem_.allocate(...) : global_mem_.allocate(...)`.
- `VMManager::install_identity_map` no longer needs to defensively
  reserve the matching VA range — `virtual_mem_` is already
  initialized over the PAGED region only, so it can never mint a VA
  in the PINNED region.
- `vx_buffer_reserve(addr, ...)` (explicit caller-chosen PA) routes
  to `pinned_mem_` if `addr < USER_BASE + PINNED_SIZE`, else
  `global_mem_`. Both paths still identity-map.

### Sizing — how big?

The PINNED region needs to hold every HW-bound buffer live at any
moment. Worst-case offline accounting:

| Workload                    | Pinned buffers                                 | Footprint |
|-----------------------------|------------------------------------------------|-----------|
| `gfx_tex` (128×128 RGBA)    | 1× src texture                                 | < 1 MB    |
| `gfx_om` (128×128)          | color + depth                                  | < 1 MB    |
| `gfx_raster` (128×128)      | tile + prim                                    | few MB    |
| `gfx_draw3d` (128×128)      | tile + prim + tex + color + depth              | ~10 MB    |
| 1080p framebuffer (RGBA8)   | 1 color                                        | 8 MB      |
| 1080p color + depth32       | color + depth                                  | 16 MB     |
| 4K color + depth32          | color + depth                                  | 64 MB     |
| Vulkan compositor, multi-FB | 2–3 color + depth + many textures              | 100s MB   |
| DXA descriptor slots        | per-slot descriptor (~KB each, dozens of slots)| < 1 MB    |

Recommendation:
- **Default**: **256 MB**. Covers the current regression suite with
  ~100× headroom, covers a 4K framebuffer (~64 MB worst case), and
  leaves room for the texture working set of a typical mesa-vortex
  frame.
- **Floor**: **16 MB**. Anything below this breaks the 1080p
  color+depth case.
- **Ceiling**: bounded by `GLOBAL_MEM_SIZE / 2` so the paged pool
  always has room. Today's `GLOBAL_MEM_SIZE` from `common.h` /
  `VX_config.toml` is 2 GiB, so the ceiling is 1 GiB.

### Configurability — three layers

1. **Build-time default** in `VX_config.toml`:
   ```toml
   [memory]
   vm_pinned_region_size = "256M"   # default, emitted as
                                    # VX_CFG_VM_PINNED_REGION_SIZE
   ```
   Tests / regression / FPGA flows that need a different default can
   override at configure time.

2. **Runtime override** via env var, read once at `vx_device_open`:
   ```
   VORTEX_VM_PINNED_SIZE=64M
   ```
   Useful for tuning without rebuilding — e.g. running a stress test
   that wants 1 GiB of pinned, or a tiny smoke test that wants 16 MiB
   to surface OOM behavior quickly.

3. **Per-device query** so callers (mesa, future Vulkan, hipcc) can
   plan around the pinned budget:
   ```c
   vx_device_query(dev, VX_CAPS_VM_PINNED_SIZE, &out);
   vx_device_query(dev, VX_CAPS_VM_PINNED_FREE, &out);
   ```
   Both follow the existing `vx_device_query` pattern. The `_FREE`
   query enables OOM avoidance in higher-level allocators (e.g. mesa
   suballocating multiple textures from one pinned slab).

Out of scope: **dynamic resize** of the pinned region after
`vx_device_open`. Possible later, but the default + env-var + query
trio covers every workload we have on the table today (single-process,
known upfront).

### Out of scope (handled elsewhere)

- The `gfx_render.h` / `vx_graphics.h` `fixed_t` redefinition blocking
  the `vm` suite from compiling `gfx_draw3d` is being addressed
  independently by the `sw/common/vx_gfx_abi.h` consolidation already
  WIP in the source tree (see vortex_ci uncommitted diff: new file
  `sw/common/vx_gfx_abi.h`, slimmed `gfx_render.h` and `vx_graphics.h`).
  This proposal layers cleanly on top of that work — neither blocks
  the other.

- Vulkan / Mesa downstream: mesa-vortex's `vortexpipe` Gallium driver
  allocates GPU buffers through `vx_buffer_create` as well. The same
  contract applies. Mesa-side changes are deferred to a separate
  proposal owned by the Vulkan track (see `project_vulkan_support`).

- DXA descriptor buffers are already pinned in
  `sw/runtime/include/dxa.h`'s helper macros — but worth a follow-up
  audit of `tests/regression/dxa_*` to confirm the test-side allocator
  follows the same convention. Tracked separately.

## Migration plan

1. Land **R2** (doc-only `VX_MEM_PHYS` docstring expansion + per-block
   cross-refs) — purely additive, no behavior change.
2. Land **T1** (rewrite call sites in `gfx_*` regression tests to use
   `vx_buffer_create(…, mode | VX_MEM_PHYS, …)` directly) — no behavior
   change.
3. Land **R3** (pre-allocated pinned region) behind
   `VX_CFG_VM_PINNED_REGION_SIZE`. Default off (size = 0 → fall back
   to today's shared-pool behavior) for one cycle so it can be
   A/B-tested against the regression suite, then flip the default to
   256 MB.
4. Land **R1** (`Device::dcr_write` validation). Safe after R3 because
   any non-PHYS DCR-write under VM now also implies a cross-pool
   address, which is independently detectable.
5. Land **T2** (per-test header notes) as cleanup.

Each step is independently revertable and the test surface stays green
throughout.

## Open questions

1. Should `VX_MEM_PHYS` become the **default** for `vx_buffer_create`
   under VM, with an opt-out flag for kernel-only buffers? Rejected
   for v3: the kernel-only path is the more common case in compute
   workloads (sgemm/diverge/dogfood), and changing the default
   silently moves VAs around. Keep the contract explicit.

2. Should the runtime check (R1) be wired to `vx_buffer_bind_to_dcr()`
   instead of `vx_dcr_write` directly? That would require a new API
   surface and a per-test rewrite — deferred. The DCR-write hook is
   the minimum-disruption point.

3. Should the pinned region be **multiple slabs** (separate budgets for
   GFX vs DXA vs future blocks)? Single slab is simpler and the
   regression workloads don't motivate it. Revisit if/when a
   workload mixes a GB-class texture working set with high-traffic
   DXA descriptor churn.

4. Should `vx_buffer_create(... VX_MEM_PHYS)` fall back to the paged
   pool when `pinned_mem_` is exhausted (silent fallback) or hard-fail
   (`VX_ERR_OUT_OF_DEVICE_MEMORY`)? Hard-fail. Silent fallback would
   resurrect the fragmentation problem the pre-allocation is meant to
   solve, and would mask sizing bugs.

# Vortex Graphics Software Stack — Design

**Scope:** a map of *where every graphics-related source lives* across the two
repositories that make up the Vortex graphics stack, and how they compose from
the Vulkan application down to the hardware. Covers the **mesa_vortex**
vortexpipe driver and the **Vortex platform** (this tree: SDK software, SimX
models, RTL). This is an orientation/index document; the per-layer detail lives
in the companion docs.

**Companion docs:**
[`vortexpipe_architecture.md`](vortexpipe_architecture.md) (the driver / NIR
lowering / draw flow),
[`graphics_fixed_function_pipeline.md`](graphics_fixed_function_pipeline.md)
(the TEX/RASTER/OM hardware microarchitecture + SimX models),
[`command_processor_control_plane.md`](command_processor_control_plane.md) (the
CP that sequences a draw), and the gfx_v2 "true GPU" program in
[`../proposals/gfx_v2_true_gpu_charter.md`](../proposals/gfx_v2_true_gpu_charter.md).

**Two trees:**
- **`mesa_vortex`** (branch `prism`) — the Vulkan/Gallium **driver** (vortexpipe).
- **this tree** (`vortex_v3/prism_v3`, branch `prism`) — the Vortex **platform**:
  the SDK (runtime + device kernels + ABI), the SimX models, and the RTL. It is
  the single unified source-of-truth (graphics + PRISM RTU).

The driver consumes the platform as an SDK (`$VORTEX_PATH` install for
headers/libs, `$VORTEX_HOME` source for the device kernels + toolchain) — a
one-directional `mesa → Vortex` dependency, the same way a userspace driver
consumes a GPU SDK.

---

## 1. `mesa_vortex` — the Vulkan/Gallium driver

All graphics code is the **vortexpipe** Gallium driver, which lavapipe (Mesa's
Vulkan frontend) drives. Path: `src/gallium/drivers/vortexpipe/`.

| File | Contains |
|------|----------|
| `vp_public.h` / `vp_private.h` | Public screen-create entry + internal driver structs (device handle, caps, compiled-kernel cache) |
| `vp_screen.c` | `pipe_screen`: opens the Vortex device, queries caps (`has_rtu`/`tex`/`raster`/`om`), advertises formats |
| `vp_context.c` | `pipe_context`: state tracking + draw/dispatch orchestration — the driver core |
| `vp_nir_to_llvm.c` | NIR shader → LLVM IR codegen for VS / FS / compute |
| `vp_nir_lower_ray_tracing_to_rtu.c` | Lowers Vulkan ray-tracing intrinsics → PRISM RTU ops |
| `vp_compile.c` | LLVM IR → `.vxbin` (drives llvm-vortex clang `+xvortex` + `vxbin.py`) |
| `vp_launch.c` | Loads a `.vxbin` and launches compute / VS kernels on the device |
| `vp_raster.cpp` | Triangle setup + binning (today the **host** `graphics::Binning`) → feeds the RASTER FF + FS + OM |
| `meson.build` | Build; consumes the Vortex SDK via pkg-config (`$VORTEX_PATH`) |

### `kernels/gfx_frontend/` — the on-device front-end build recipe

| File | Contains |
|------|----------|
| `gfx_frontend_kernel.cpp` | Compile unit (one `#include "pipe_frontend.h"`) → `setup_k` + `binning_k` |
| `Makefile` | Builds `gfx_frontend.vxbin`; **consumes the kernel sources from `$VORTEX_HOME/sw/gfx`** (no copies) |
| `README.md` | Provenance + ownership note |

> The driver owns only the **build recipe**; the kernel *sources* live in the
> SDK (`sw/gfx`, below). This is the single-source-of-truth arrangement: the
> SimX tests and the driver compile the *same* files.

---

## 2. The Vortex platform (this tree) — SDK, SimX, RTL

Graphics spans `sw/` (software), `sim/` (SimX models), `hw/` (RTL), and `tests/`.

### 2.1 Software — `sw/`

| Dir | Contains |
|-----|----------|
| [`sw/common/`](../../sw/common/) | **Contracts + oracle.** [`vx_gfx_abi.h`](../../sw/common/vx_gfx_abi.h) (on-wire RASTER buffer ABI — `rast_prim_t`/`rast_tile_header_t`, fixed-point types = the HW contract); [`gfx_frontend_abi.h`](../../sw/common/gfx_frontend_abi.h) (front-end host/device ABI — `pipe_arg_t`, `PIPE_STAGE_*`, `setup_vertex_t`); [`gfx_render.cpp`](../../sw/common/gfx_render.cpp)/[`.h`](../../sw/common/gfx_render.h) (the **reference renderer / golden oracle** — host `Binning`/`Rasterizer`/`Blender`/`DepthStencil`) |
| [`sw/gfx/`](../../sw/gfx/) | **Device front-end kernel sources (single source of truth).** [`pipe_frontend.h`](../../sw/gfx/pipe_frontend.h) (`setup_k`+`binning_k`, the 9-stage front end); [`setup_math.h`](../../sw/gfx/setup_math.h) (near-plane clip + Q15.16 triangle setup); `pipe_abi.h` / `setup_types.h` (bin-grid + render-target macros) |
| [`sw/runtime/`](../../sw/runtime/) | Host driver layer in `libvortex.so`: [`graphics.cpp`](../../sw/runtime/graphics.cpp)/[`graphics.h`](../../sw/runtime/include/graphics.h) — legacy host `graphics::Binning` + `FrontEndPool` (launches the on-device front end) |
| [`sw/kernel/include/`](../../sw/kernel/include/) | [`vx_graphics.h`](../../sw/kernel/include/vx_graphics.h) — device-side graphics intrinsics (`vx_rast`/`vx_tex`/`vx_om` wrappers over the CUSTOM-1 ops) |

### 2.2 SimX models — `sim/simx/` (the SimX-first dev + evaluation engine)

| Dir | Contains |
|-----|----------|
| [`sim/simx/raster/`](../../sim/simx/raster/) | `raster_core.*` (RasterCore: tile/prim walk, TE/BE descent → quads) + `raster_unit.*` (per-core SFU PE) |
| [`sim/simx/om/`](../../sim/simx/om/) | `om_core.*` + `om_unit.*` — depth / stencil / blend / ROP |
| [`sim/simx/tex/`](../../sim/simx/tex/) | `tex_core.*` + `tex_unit.*` — sampler: address / filter / format-decode |

The SimX models `#include` [`gfx_render.h`](../../sw/common/gfx_render.h) +
[`vx_gfx_abi.h`](../../sw/common/vx_gfx_abi.h) as their oracle and on-wire types
— which is why those headers stay owned by the SDK rather than moving to the
driver.

### 2.3 RTL hardware — `hw/rtl/`

| Dir | Contains |
|-----|----------|
| [`hw/rtl/raster/`](../../hw/rtl/raster/) | `VX_raster_*` — rasterizer FF (mem / te / be / qe / edge / slice / arb / csr / dcr) |
| [`hw/rtl/tex/`](../../hw/rtl/tex/) | `VX_tex_*` — sampler FF (addr / format / lerp / wrap / sampler / sat / stride) |
| [`hw/rtl/om/`](../../hw/rtl/om/) | `VX_om_*` — output-merger FF (blend / compare / ds / logic_op / stencil) |
| [`hw/rtl/VX_graphics.sv`](../../hw/rtl/VX_graphics.sv) | Graphics cluster wrapper — instantiates raster/tex/om cores + broadcast |

### 2.4 Tests — `tests/`

| Dir | Contains |
|-----|----------|
| [`tests/graphics/`](../../tests/graphics/) | Image-validated end-to-end: `gfx_draw3d` (trace replay), `gfx_raster`/`tex`/`om` (single FF), `gfx_pipeline_raster`/`om`/`tex` (on-device front end → FF) |
| [`tests/regression/`](../../tests/regression/) | Kernel-level: `gfx_setup_kernel`, `gfx_binsort_kernel`, `gfx_pipeline_kernel` |
| [`tests/unittest/gfx_binsort/`](../../tests/unittest/gfx_binsort/) | Host-reference bin-sort unit test |

---

## 3. The stack

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Vulkan application                                                        │
└──────────────────────────────────────────────────────────────────────────┘
                                   │  vkCmdDraw / vkCmdDispatch / vkCmdTraceRays
                                   ▼
╔════════════════════════════ mesa_vortex (prism) ══════════════════════════╗
║  lavapipe        Mesa Vulkan frontend (src/gallium/frontends/lavapipe)     ║
║      │  Gallium pipe_context / NIR shaders                                 ║
║      ▼                                                                     ║
║  vortexpipe      Gallium driver  (src/gallium/drivers/vortexpipe)          ║
║   ┌──────────────────────────────────────────────────────────────────┐    ║
║   │ vp_screen  vp_context        ── state + draw orchestration        │    ║
║   │ vp_nir_to_llvm → vp_compile  ── VS/FS/compute NIR → .vxbin         │    ║
║   │ vp_nir_lower_ray_tracing_to_rtu ── RT intrinsics → RTU            │    ║
║   │ vp_raster                    ── setup + binning → RASTER/FS/OM    │    ║
║   │ vp_launch                    ── load .vxbin, dispatch on device   │    ║
║   │ kernels/gfx_frontend/        ── builds gfx_frontend.vxbin ········─┼──┐ ║
║   └──────────────────────────────────────────────────────────────────┘  │ ║
╚══════════════════════════════════════════════════════════════════════════╝│
        │  vortex2 API (libvortex.so) + on-wire ABI (vx_gfx_abi.h)           │
   ═════╪═══════════ SDK boundary  ($VORTEX_PATH / $VORTEX_HOME) ════════════╪═
        ▼                                                                    │
╔════════════════════════════ prism_v3 (prism) ═════════════════════════════╗│
║  sw/  VORTEX SDK (software)                                                ║│
║   ├ sw/runtime  libvortex.so : vortex2 API · graphics::Binning · FrontEndPool
║   ├ sw/gfx      device front-end kernels: setup_k + binning_k  ◄───────────╫┘ (built
║   ├ sw/common   ABI contracts (vx_gfx_abi, gfx_frontend_abi) + gfx_render  ║   by both)
║   │             reference oracle                                           ║
║   └ sw/kernel/include  vx_graphics.h  (device FF intrinsics)               ║
║                                   │                                        ║
║         ┌─────────────────────────┴───────────── backend (pick one) ──┐    ║
║         ▼                          ▼                          ▼        │    ║
║  sim/simx (C++ model)        hw/rtl (Verilog)          XRT / FPGA U55C │    ║
║   raster_core  om_core        VX_raster_*  VX_om_*      (synthesized   │    ║
║   tex_core   ◄─ gfx_render    VX_tex_*  VX_graphics.sv   bitstream)    │    ║
║   (SimX-first dev + eval)     (300 MHz signoff)                        │    ║
║         ▲ validated against gfx_render oracle + tests/graphics PNGs    │    ║
║         └──────────────────────────────────────────────────────────────   ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### On-device render flow (the gfx_v2 "true GPU" path, all device-resident)

```
 host submit ─► CP ─► setup_k ───► binning_k ────► RASTER ─► FS ──► TEX ─► OM ─► framebuffer
                     (clip +       (bin-sort:      (FF)    (SIMT)  (FF)   (FF)   └─ present
                      tri setup)    count→scan→                                     (only egress)
                      sw/gfx        emit→sort→
                                    header-scan)
                     └──────────── sw/gfx kernels ───────────┘   └─── sim/simx or hw/rtl FF ───┘
```

---

## 4. The two boundaries

- **SDK boundary** — mesa consumes the Vortex SDK: `$VORTEX_PATH` (install) for
  headers + `libvortex.so`, and `$VORTEX_HOME` (source) for the `sw/gfx` kernel
  sources + the kernel toolchain. One-directional: `mesa → Vortex`. The on-wire
  ABI ([`vx_gfx_abi.h`](../../sw/common/vx_gfx_abi.h)) is the hardware contract
  and stays SDK-owned.
- **Backend boundary** — the same SDK + kernels run on **SimX** (the SimX-first
  development + cycles/perf evaluation target), **RTL** (300 MHz U55C signoff),
  or **FPGA**, selected at runtime (`VORTEX_DRIVER`).

## 5. Single source of truth

The on-device front-end kernels live once, in [`sw/gfx/`](../../sw/gfx/). The
SimX graphics tests compile them to validate the SimX models, and vortexpipe
compiles the *same files* to `gfx_frontend.vxbin` to launch them — no
duplication, no drift. The front-end ABI
([`gfx_frontend_abi.h`](../../sw/common/gfx_frontend_abi.h)) stays in `sw/common`
because the host runtime (`FrontEndPool`) also includes it.

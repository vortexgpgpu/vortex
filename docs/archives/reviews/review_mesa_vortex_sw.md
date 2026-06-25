# v2 Review: mesa_vortex SW stack (vortexpipe driver)
**Date:** 2026-06-17

Scope reviewed: every file in `~/dev/mesa_vortex/src/gallium/drivers/vortexpipe/`
(`vp_context.c`, `vp_nir_to_llvm.c`, `vp_compile.c`, `vp_launch.c`, `vp_raster.cpp`,
`vp_screen.c`, `vp_private.h`, `vp_launch.h`, `vp_raster.h`, `vp_nir_to_llvm.h`,
`meson.build`). Read against the gfx_v2 charter, the vortexpipe-driver delta,
the software-fallback doc, and the compiler-stage-coverage doc. Cross-checked
ABI claims against the device tree (`prism_v3/hw/rtl/core/VX_decode.sv`,
`hw/rtl/om/VX_om_unit.sv`, `sw/kernel/include/vx_graphics.h`,
`sw/runtime/graphics.cpp`).

---

## 1. Overall assessment

The driver is a clean, well-commented **gfx-v1** decorator-on-llvmpipe: it
intercepts the Gallium compute/draw/state hooks, JIT-compiles VS/FS/compute NIR
to `.vxbin`, and drives the FF RASTER/OM/TEX units for the narrow "simple draw"
case. As gfx-v1 engineering it is solid. **Measured against the gfx_v2 "true
GPU" charter it is at most ~20% of the way there**, and it has drifted out of
sync with the device: the FS translator still emits the **legacy `vx_om`/`vx_tex`
direct-register ISA**, but the device decode has reassigned CUSTOM1 funct3=2 to
the **`vx_om4` window ABI** — so the fragment-shader path the driver generates is
ABI-broken against the current silicon/SimX. None of the four charter pillars
(every stage on SIMT; FF fed device-side; CP-orchestrated; full residency) is
realized: there is no CP command ring, no on-device binning, no three-tier
HW/composed/SW selection, no GS/tess/mesh coverage, and the per-draw path still
round-trips color/depth/textures through host memory. The thin-shim rewrite of
`vp_raster.cpp` moved *DCR-register knowledge* into the runtime but left *host
binning, host readback, and per-draw allocation* exactly where gfx-v1 had them.

**Maturity grade: D** (solid gfx-v1 driver; gfx_v2 work essentially not started,
and a live correctness regression against the current device ABI).

---

## 2. Correctness findings

- **`vp_nir_to_llvm.c:1246-1264` (`emit_vx_om`) — High.** The FS wrapper emits the
  legacy OM op `.insn r4 43, 2, 0, x0, $0, $1, $2` with `$0=pos_face`,
  `$1=rgba`, `$2=depth` as **direct register values**. The device decode at
  `VX_decode.sv:773` now maps CUSTOM1 funct3=2 **exclusively to `vx_om4`**, whose
  R-type operands are `rs1 = quad descriptor (pos_mask | face<<31)`,
  `rs2 = payload window-slot base` (color/depth staged into the graphics window
  via `vx_rt_set`/SETW). There is no legacy `vx_om` decode left in the tree
  (grep of `VX_om_unit.sv`/`VX_decode.sv` finds none; `sw/kernel/include/vx_graphics.h:89`
  documents funct3=2 as "the sole OM op"). The driver-generated FS therefore
  passes the **color value where the HW expects a window slot base**, and never
  stages color/depth into the window at all → garbage/no output. This is a hard
  ABI break, not a perf issue.

- **`vp_nir_to_llvm.c:1466-1474` (`pos_face` packing) — High.** Even setting the
  operand-role break aside, the position encoding is wrong for `vx_om4`: the
  wrapper packs `(py<<16)|(px<<1)|face`, but `vx_om4`'s `desc` is the raw `vx_rast`
  `pos_mask` (`cov_mask[3:0]`, `qx@[4 +: 14]`, `qy@[18 +: 13]`, `face<<31`) and
  the unit derives `pos_x=(qx<<1)|(F&1)`, `pos_y=(qy<<1)|(F>>1)` itself
  (`vx_graphics.h:82-88`). The wrapper is doing the sub-pixel expansion the HW
  now does, with an incompatible bit layout.

- **`vp_nir_to_llvm.c:882-898 / 928` (`emit_vx_tex`) — High.** TEX is emitted as
  legacy `vx_tex` (funct3=1, direct `u,v,lod` regs). funct3=1 *does* still decode
  (`VX_decode.sv:747`, "legacy"), so this is less immediately fatal than OM — but
  the charter and the driver-delta doc make `vx_tex4` the **sole** TEX op the
  translator should emit (driver-delta §3, charter §6.8), and the runtime's
  `program_tex`/window staging (`graphics.cpp`) is built for the tex4 window ABI.
  Emitting the legacy op leaves TEX on a deprecated path that the FF-expansion
  work will remove; it is a latent break and a charter-conformance miss.

- **`meson.build:42` vs `vp_compile.c:38-39,94` — High (build/runtime).** meson
  defines `-DVP_VORTEX_PATH=...` but `vp_compile.c` reads `VP_VORTEX_HOME` (never
  defined → defaults to `""`). The baked-in default for the Vortex tree is thus
  empty, so device compilation only works if `$VORTEX_HOME` is exported at
  runtime; an install-time-configured build silently has no usable default.
  Compounding it, `vp_compile.c:116` then derives the build dir as
  `"$VORTEX_HOME/build"` and links `"$VORTEX_HOME/sw/kernel/libvortex2.a"` and
  `"$VORTEX_HOME/sw/kernel/scripts/..."` — i.e. it treats the path as a **source
  tree**, while the meson comment (and the project's pkg-config integration
  policy) says the driver should consume the **install tree** (`$VORTEX_PATH`,
  `runtime/include`, `kernel/include`). The `-DVP_TOOLDIR` define *is* consumed
  (`vp_compile.c:42`), so only the HOME/PATH half is broken, but the net effect
  is a confused and partly dead configuration contract.

- **`vp_raster.cpp:38-39, 205` (`VP_RAST_PRIM_STRIDE 120`) — Med.** The
  primitive-buffer stride is a hard-coded literal `120` ("vec3e_t edges[3] +
  rast_attribs_t") that must match `sizeof(graphics::rast_prim_t)` on the device
  side and the same constant in `vp_nir_to_llvm.c` (`VP_RAST_PRIM_STRIDE` at
  `:1398`). A silent struct-layout change on either side corrupts every
  interpolation. Should be a single shared ABI constant (from `vx_graphics.h` /
  the on-wire ABI header), not duplicated literals in two repos.

- **`vp_raster.cpp:30-36` (`VP_RASTER_BIN_LOGSIZE 7`) — Med.** The 128px coarse-bin
  log-size is hand-copied from the device config with a comment admitting "mesa
  is not compiled with the device config, so it is matched here." If the device's
  `VX_CFG_RASTER_BIN_LOGSIZE` ever changes, host binning and the RASTER front end
  silently disagree. Same root cause as the stride: ABI constants should flow from
  one header, not be re-declared.

- **`vp_raster.cpp:176-179` (depth clear) — Med.** Depth is cleared to
  `0x00`/`0xFF` byte-fill chosen only from `depth_func` (GREATER/GEQUAL→0 else
  max). This ignores the actual Vulkan clear value (`VkClearDepthStencilValue`)
  entirely — it assumes a full-range clear to near/far. Any app that clears depth
  to a non-extremal value renders wrong. gfx-v1-acceptable, but a correctness
  limitation worth flagging.

- **`vp_raster.cpp:89-101` and `vp_nir_to_llvm.c:1337-1339` (varying routing by
  component count) — Med.** Both the host shim and the FS wrapper route a varying
  to "texcoord vs color" purely by its component count (`nc==2`→texcoord,
  `nc>=3`→color). Two vec2 varyings, a vec2 color, or a vec3 texcoord all
  mis-route. It is an undocumented, fragile heuristic standing in for real
  varying-slot linking; it happens to fit `draw3d` but is not general.

- **`vp_context.c:1015` (NPOT texture) — Low.** Non-power-of-two textures are
  rejected (`mesa_logw` + skip HW path). Documented gfx-v1 limitation, correctly
  gated; noted for completeness, not a bug.

- **`vp_launch.c:148-149` / `vp_launch.c` descriptor model — Low.** Only descriptor
  **set 0**, bound as constant-buffer index 1, is relocated; multi-set / dynamic
  offsets are unhandled (header admits "single-SSBO … later generalization").
  Correct for the current tests; a coverage gap, not a defect.

---

## 3. Efficiency findings

- **Per-draw `.vxbin` temp-file marshalling — High.** `vp_raster.cpp:124-139`,
  `vp_launch.c:70-85`, and `vp_launch.c:205-219` each `mkstemp` a file, `write()`
  the already-in-memory `.vxbin`, then `vx_module_load_file()` reads it back —
  **on every draw / every VS launch / every dispatch**. The kernel image is
  already a host blob attached to the cso (`vp_cso.vxbin`); writing it to `/tmp`
  and re-reading it per draw is pure overhead and serializes on the filesystem.
  A `vx_module_load_memory`-style API (or one-time module load cached on the cso)
  is the obvious fix.

- **Per-draw queue/module/buffer create+release churn — High.** `vp_raster.cpp`
  and both `vp_launch*` create a fresh `vx_queue`, load+release the module, and
  create+release every device buffer (tiles, prims, color, depth, tex)
  **per draw** (`vp_raster.cpp:145-167, 287-296`). A real driver creates the queue
  once per context and the module once per pipeline; buffers for persistent
  attachments live for the framebuffer's life. This is the single biggest
  per-draw cost after the host binning.

- **Texture re-conversion + re-upload every draw — Med.** `vp_raster.cpp:231-260`
  reads the bound texture back from the Gallium resource (`vp_context.c:1018`),
  byte-swaps R8G8B8A8→A8R8G8B8 on the host, allocates a `texbuf`, and re-uploads
  it for **every** draw that binds it — there is no caching keyed on the
  texture/sampler cso. Static textures get re-marshalled per frame.

- **`vp_create_compute_state`/`vp_create_*_state` compile via `system()` — Med.**
  `vp_compile.c:145-200` shells out to `clang` and `vxbin.py` through `system()`
  (two process spawns + shell parse) per shader. Acceptable at pipeline-create
  time (cached on the cso), but the file header itself flags moving in-process as
  deferred; combined with the temp-dir churn it makes first-use latency high.

- **Whole-image color upload/clear per draw — Med.** `vp_raster.cpp:179-195`
  builds a full `w*h*4` host clear buffer and uploads color **and** a
  host-generated depth clear every draw, rather than clearing on-device or
  reusing a resident, already-cleared attachment.

---

## 4. Performance findings

- **Host binning is still on the host — High (charter-blocking).**
  `vp_raster.cpp:108-115` calls `graphics::Binning(...)` on the CPU every draw —
  exactly the host `Binning()` the charter (§4, driver-delta §2) says must move to
  on-device SIMT sort-middle. This is the headline gfx_v2 deliverable and it is
  not begun in the driver. VS output is also fully read back to host
  (`vp_context.c:948` → `vp_launch_vs` → `vx_enqueue_read`) before binning, the
  precise "VS readback" the charter forbids.

- **Per-draw color/depth readback round-trip — High (charter-blocking).** The draw
  reads the cleared color attachment to host (`vp_context.c:1040`
  `vp_fb_color_read`), runs the device path, then writes the result back
  (`vp_context.c:1044` `vp_fb_color_write`) and `vp_raster.cpp:281` reads color
  back again. Charter pillar 4 mandates **no device→host copy between draws**;
  present is the only egress. The current path copies color (and depth) host↔device
  twice per draw. Each draw is an independent, fully-drained
  upload→launch→`vx_queue_finish`→readback — no batching across draws.

- **Per-draw heap allocations — Med.** `vp_context.c:938` (`malloc(count*stride)`
  for xverts), `:1016` (`malloc(tw*th*4)` for texels), `:1039`
  (`malloc(w*h*4)` for color), plus `vp_raster.cpp` `std::vector` tilebuf/primbuf/
  texbuf/zclear, all per draw. No pooling; an `std::unordered_map<uint32_t,
  vertex_t>` (`vp_raster.cpp:79`) is allocated and hashed per draw where a flat
  vector indexed by vertex id would do.

- **One CTA-per-core grid, no overlap — Low.** `vp_raster.cpp:267-279` launches
  `grid=num_cores`, `block=threads*warps` and `vx_queue_finish` before returning;
  every draw fully drains the device. Fine for correctness bring-up; leaves all
  cross-draw and cross-stage overlap (the CP's job) on the table.

---

## 5. "True GPU" alignment vs NVIDIA/AMD/Intel

How a real Vulkan UMD (radv/anv) + its compiler (ACO/genxml) is structured, vs.
where vortexpipe stands:

- **Driver thinness / orchestration.** radv/anv build a **command buffer** of GPU
  packets (PM4 / Gen batch) at record time and the **GPU front end / CP** executes
  it; the host does not run pipeline stages or read intermediates back. vortexpipe
  is the opposite: `vp_draw_vbo`/`vp_raster_draw` **is** the orchestrator — it runs
  VS, reads it back, bins on the CPU, programs DCRs (now via the runtime), launches
  the FS, and reads color back, all synchronously per draw. The charter's CP
  command-ring model (driver-delta §2: encode the whole draw as one ring sequence,
  ring the doorbell once, poll `Q_SEQNUM` at pass end) is **entirely absent** —
  grep finds no doorbell/seqnum/DrawCommands/CMD_LAUNCH in the driver. The
  `vp_raster.cpp` header's claim of a "thin driver shim" is true only for *DCR
  register knowledge*; the **pipeline orchestration is still wholly host-side**, so
  the driver is not "thin" in the sense the charter means.

- **Compiler stage coverage.** ACO/anv lower **every** programmable stage (VS, TCS,
  TES, GS, task, mesh, FS, compute, RT) to ISA. `vp_nir_to_llvm.c` routes only
  `is_vs`/`is_fs` (`:1506-1507`) plus the compute entry; **GS / tessellation /
  task+mesh are not handled at all** (grep finds no GEOMETRY/TESS/MESH/emit_vertex
  routing). The charter's count→scan→emit amplification framework and the
  sw-tessellator kernel (compiler-stage-coverage §4-5) are unstarted. This is the
  largest single charter gap after binning. RT lowering is referenced as
  "already covered" by `vp_nir_lower_ray_tracing_to_rtu.c` but that file is not in
  this driver dir and was not exercised here.

- **HW/SW fallback selection.** The software-fallback doc specifies a **three-tier,
  per-unit, compile-time** selection (native HW / HW-composed / pure-SW over the
  shared bin buffers, never llvmpipe). The driver implements **none** of it: it is
  still **binary HW-vs-llvmpipe** at the *whole-draw* granularity
  (`vp_context.c:968` `gfx_hw = has_raster && has_om`; on any miss the entire draw
  falls to `lp_draw_vbo` at `:1075`). There is no `libgfx_sw`, no per-unit fork, no
  composed (multi-tap) path, no zero-acceleration SIMT mode — and crucially the
  fallback is still **to llvmpipe on the host**, which the charter explicitly
  retires for the runtime. `MESA_VORTEX_STRICT` (`vp_context.c:167`) is a good
  bridge (fail instead of silently using llvmpipe) but it is a CI guard, not the
  on-device SW path the charter requires.

- **Residency / allocation.** radv/anv allocate from VRAM heaps and keep the frame
  working set resident. vortexpipe has **no allocator** — every buffer is a
  per-draw `vx_buffer_create`/`release` with an implicit host backing and explicit
  host copies. The usage-routed pinned/paged allocator, per-pass pool, and
  `VX_CAPS_VM_PINNED_*` planning (driver-delta §4) are absent.

- **Where it *is* aligned.** Compiling **every** stage at pipeline-create from NIR
  (no prebuilt blobs), caps-gating FF paths on `VX_CAPS_ISA_FLAGS`
  (`vp_screen.c:95-97`), clamping `pipe_compute_caps` to the HW CTA size
  (`vp_screen.c:128-137`), and an honest device-name string are all genuinely
  driver-shaped and correct. The decorator-patch-in-place approach
  (`vp_private.h` header) is a defensible shortcut over a 140-thunk vtable.

---

## 6. v2.1 recommendations

### P0 — critical (correctness / charter-blocking)

1. **Fix the FS OM ABI: emit `vx_om4`, not legacy `vx_om`.**
   File: `vp_nir_to_llvm.c` `emit_vx_om`/`emit_fs_wrapper` (`:1246-1264`,
   `:1455-1475`). Stage color[0..3]/depth[0..3] into the graphics window
   (SETW / `vx_rt_set` slots), pass `desc = vx_rast pos_mask | face<<31` and
   `base = window slot` to funct3=2, and stop pre-expanding sub-pixel x/y (the
   unit does it). Match `sw/kernel/include/vx_graphics.h:82-90`. Without this the
   driver-generated FS produces no correct output on the current device.

2. **Migrate TEX to `vx_tex4` (window ABI), retire legacy `vx_tex`.**
   File: `vp_nir_to_llvm.c` `emit_vx_tex`/`emit_tex` (`:882-940`). Emit funct3=5
   single/quad mode against the window slots, matching `vx_tex4_single`/`_quad`,
   so it stays valid as the FF-expansion work removes the legacy decode.

3. **Fix the `VORTEX_HOME`/`VORTEX_PATH` define contract.**
   Files: `meson.build:41-44` and `vp_compile.c:38-43,93-117`. Either define
   `-DVP_VORTEX_HOME` (and a real default) or change `vp_compile.c` to read
   `VP_VORTEX_PATH`; then make the toolchain/library paths consume the **install
   tree** (`$VORTEX_PATH/kernel`, `runtime`) per the meson pkg-config policy, not a
   `$VORTEX_HOME/build` source tree. Today the baked default is empty and the
   layout assumptions are inconsistent.

### P1 — important (charter-critical perf + per-draw overhead)

4. **Move binning + VS output on-device; stop the host `Binning()` and VS readback.**
   Files: `vp_raster.cpp:108-115`, `vp_context.c:947-953`. This is *the* gfx_v2
   deliverable (charter §6.1/§6.2, driver-delta §2). Even an interim "VS output
   stays resident; a SIMT binning kernel consumes it" removes the largest host
   round-trip. Keep the bin-sort buffers resident for the FS to read directly.

5. **Eliminate per-draw `.vxbin` temp files and per-draw queue/module/buffer churn.**
   Files: `vp_raster.cpp:124-167,287-296`, `vp_launch.c:70-85,205-219`. Load each
   module once and cache the `vx_module_h`/`vx_kernel_h` on the cso; create the
   queue once per context; keep attachment buffers resident for the framebuffer's
   life. Add/locate a `vx_module_load_memory` path so the blob never touches `/tmp`.

6. **Stop the per-draw color/depth host round-trip.**
   Files: `vp_context.c:1040-1044` (`vp_fb_color_read`/`write`),
   `vp_raster.cpp:179-195,281`. Keep color/depth device-resident across draws;
   clear on-device; only egress at present. Honor the real Vulkan depth clear value
   (`vp_raster.cpp:176-179`) instead of the func-derived byte fill.

7. **Begin the three-tier, per-unit fallback selection (replace binary HW-vs-llvmpipe).**
   Files: `vp_context.c:968-975,1070-1075`, new `libgfx_sw` link. Implement the
   per-unit HW / HW-composed / SIMT-SW fork from the software-fallback doc, route
   absent/unsupported units to **on-device SIMT** (never `lp_draw_vbo`), and stand
   up zero-acceleration mode as the bring-up target. Until then keep
   `MESA_VORTEX_STRICT` mandatory in CI so the llvmpipe fallback can't mask gaps.

### P2 — nice-to-have

8. **Single-source the ABI constants.** Pull `VP_RAST_PRIM_STRIDE`,
   `VP_RASTER_BIN_LOGSIZE`, and the OM/TEX/depth encodings from the shared
   `vx_graphics.h` / on-wire ABI header instead of duplicating literals across
   `vp_raster.cpp:30-39`, `vp_nir_to_llvm.c:1398`, and `vp_context.c:597-655`.

9. **Cache converted textures** keyed on the sampler-view/texture cso
   (`vp_raster.cpp:231-260`) instead of re-reading + re-byte-swapping + re-uploading
   per draw.

10. **Replace the component-count varying heuristic** (`vp_raster.cpp:89-101`,
    `vp_nir_to_llvm.c:1337-1339`) with real varying-slot linking so multi-vec2 /
    vec3-texcoord pipelines route correctly.

11. **Plan GS/tess/mesh translator coverage** (`vp_nir_to_llvm.c` stage routing at
    `:1506`) per compiler-stage-coverage §4 — the count→scan→emit chain + the
    sw-tessellator kernel — so the driver advertises the charter's full Vulkan 1.4
    surface rather than VS/FS/compute only.

12. **Move shader compile in-process** (`vp_compile.c` `system()` calls) to cut
    first-use latency, as the file header already anticipates.

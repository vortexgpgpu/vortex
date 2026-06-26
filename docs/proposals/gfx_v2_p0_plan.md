# gfx_v2 P0 — Legacy purge + true-GPU device-orchestrated draw + mesa code-complete

**Date:** 2026-06-25. **Tree:** `prism_v3` (`prism`) + `~/dev/mesa_vortex` (`prism`).
**Companion:** [../designs/vortex_gfx_v2_bible.md](../designs/vortex_gfx_v2_bible.md) (ground truth), [gfx_v2_status_and_schedule.md](gfx_v2_status_and_schedule.md).

## P0 definition of done (sign-off criteria)
1. **One ISA.** No legacy `vx_tex` (funct3=1), legacy `vx_om` (3-operand R4), or the
   shader-issued `vx_rast_begin` (funct3=4) anywhere — sw kernels, mesa, SimX, RTL; decode +
   datapaths carry no dead legacy branches. The graphics surface is exactly: `vx_tex4`,
   `vx_om4`, `vx_rast_fetch`, `vx_gfx_set/get/get_after`, RTU `vx_rt_*`. (The rasterizer
   arms itself on RASTER config write — no shader/CP "begin" op.)
2. **True-GPU invocation.** A draw is a **single device-orchestrated command** — the host
   submits a draw descriptor and present; the CP sequences VS→setup→bin→FF-config→FS→OM
   on-device with **no intermediate host round-trip** (no host VS-finish-before-raster split,
   no per-stage host involvement, no per-draw re-upload of resident resources).
3. **mesa_vortex up to date / code-complete.** Emits only the v2 ISA, drives the
   device-orchestrated draw, resources resident; the supported graphics path is 100%
   on-device and correct; gfx + vulkan suites pass.
4. **Validated** on **simx AND rtlsim** (the `graphics_parity` matrix, multi-core) byte-exact.

## Explicitly OUT of P0 (tracked separately — do NOT expand scope)
- OM v2 / TBDR + programmable blend (framebuffer-fetch, tile-resident FB, Early-Z).
- OM/SETW **C3/C4 doctrine** handle + per-unit scoreboard-retired windows (the
  KnownViolation stays a warning through P0; `vx_om4` remains fire-and-forget).
- On-device SIMT **software fallback** for *unsupported* state — the host-llvmpipe fallback
  for genuinely-unsupported Vulkan state **remains** in P0; only the *supported* path must be
  fully on-device. (Replacing the fallback = the separate `libgfx_sw` track.)
- GS/tess/mesh stages; U55C synthesis; full Khronos Vulkan CTS.

## Sequencing
ISA unification first (P1–P3) so the device-orchestrated draw (P4–P6) is built against a
clean single ISA. SimX-first then rtlsim per project rule; **commit each phase when
validated**; **defer synth**; **never push without per-push auth**.

---

### Phase 1 — TEX: migrate all callers to `vx_tex4`
**Goal:** nobody emits legacy `vx_tex` (funct3=1) anymore.
**Work:**
- Test kernels (`gfx_draw3d`, `gfx_tex`, `gfx_pipeline_tex`, `gfx_tex4q`): `vx_tex(stage,u,v,lod)`
  → `vx_tex4_single`/`_quad` (stage u,v via `vx_gfx_set`, sample funct3=5, read texel via
  `vx_gfx_get_after(out_slot, handle)`).
- mesa FS: `emit_vx_tex` (R4 funct3=1) → windowed `vx_tex4` emit (window-stage u,v, funct3=5,
  read texel back). (`vp_nir_to_llvm.c`)
**Exit:** legacy `vx_tex` has **zero callers** (grep-clean in kernels + mesa); tex/draw3d/
pipeline_tex pass simx (+ rtlsim spot-check); mesa textured draw renders. *(Watch-item: tex4
is more instrs/sample than legacy; acceptable for one-ISA — a single-sample fast `vx_tex4`
form is a post-P0 option, not a blocker.)*

### Phase 2 — OM: migrate mesa to `vx_om4` (close the ABI divergence)
**Goal:** driver and device on the same OM ABI (the §8 #1 critical gap).
**Work:**
- mesa FS: legacy 3-operand `vx_om` (`.insn r4 43,2,…`, color/depth in regs) → `vx_om4`
  (stage color[0..3]/depth[0..3] via `vx_gfx_set` into the window, emit `vx_om4(desc, base)`).
  (`vp_nir_to_llvm.c:1430-1448,1660`)
- Device OM is already single (`vx_om4` only) — confirm; remove stale legacy-OM comments
  (`vx_graphics.h:27`, `gfx_sw.h:251`).
**Exit:** mesa emits `vx_om4`; gfx_om + vulkan draw OM-correct on simx + rtlsim; OM ABI
unified. *(This is the correctness keystone — vulkan graphics OM was divergent.)*

### Phase 3 — Delete legacy ISA from sw + hw (single-ISA sweep)
**Goal:** the legacy encodings cease to exist in the codebase.
**Work (now that P1–P2 left no callers):**
- Delete legacy `vx_tex`: `vx_graphics.h` intrinsic; SimX `decode.cpp` case 1 + the
  `is_tex4==0` branches in `tex_core`/`tex_unit`; RTL `VX_decode.sv` funct3=1 + the
  `is_tex4` datapath muxes in `VX_tex_*`; the doctrine note.
- Confirm legacy `vx_om` fully absent (device already single); purge dead OM
  branches/enums/comments.
- **Eliminate `vx_rast_begin`:** make the raster producer **auto-arm on RASTER config write**
  (invert the current `has_begun_`-cleared-on-DCR-write to armed-on-config-complete, both SimX
  `raster_core.cpp` and RTL `VX_raster_core`/`VX_raster_unit`); then delete the op — intrinsic
  (`vx_graphics.h`), SimX/RTL decode funct3=4, `RasterType::BEGIN`, doctrine case, mesa
  `emit_vx_rast_begin`. Works for standalone kernels and the Phase-4 `OP_DRAW` path alike.
- Tree-wide sweep of dead raster residue + stale comments; update the bible ISA table
  (drop legacy rows, incl. `vx_rast_begin`).
**Exit:** grep-clean of legacy ISA (sw + hw); single-ISA decode; full suite green
simx + rtlsim; Verilator `-Wall` clean.

### Phase 4 — Device-orchestrated draw: CP draw sequencer (`OP_DRAW`)
**Goal:** the device runs a whole draw from one command (the "true GPU" core).
**Work:**
- Define a resident **draw descriptor** (VS kernel handle, FS kernel handle, FF DCR block,
  grid/block, buffer addresses).
- Implement `OP_DRAW` (or a QMD-style draw built on `OP_LAUNCH_QMD`) in the **Emulation CP**
  (`sim/common/cmd_processor.cpp`) that sequences VS → expand → setup → bin → FF-config →
  FS on-device, draining each stage (inter-stage barrier) with **no host in between**.
- Mirror in the **RTL CP** (`hw/rtl/cp/`).
**Exit:** a draw runs from ONE host-submitted command on simx, then rtlsim; **byte-exact**
vs the current multi-launch batch. *(Largest phase; RTL CP is the least-mature surface.)*

**STATUS (2026-06-26, Batch 2 done):** Emulation CP `OP_DRAW` (the indirect
command-bundle model: descriptor of LAUNCH_QMD/DCR_WRITE/CACHE_FLUSH steps the CP
walks, draining each launch) + runtime `vx_enqueue_draw` shipped and validated on
simx + rtlsim (native gfx suite PASS; draw3d byte-identical to the multi-launch
batch). The **RTL CP** (`hw/rtl/cp/`) `OP_DRAW` mirror is **synth-deferred** (it is
the XRT/FPGA-only path; simx + rtlsim both run the C++ Emulation CP, and the mirror
is unvalidatable until synthesis).

**RESOLVED (2026-06-26): cap-gated ring-batch fallback** — the XRT path no longer
needs the RTL FSM to work. `CP DEV_CAPS` bit 25 (`SUPPORTS_DRAW`): the Emulation CP
sets 1, an RTL CP without the mirror reads 0. `vx_enqueue_draw` submits one
`CMD_DRAW` when supported, else streams the same launches+DCRs as one ring batch
(functionally identical). Both paths validated on simx (cap=1 OP_DRAW; cap=0
fallback). Flipping the RTL cap to 1 once a hardware `OP_DRAW` mirror is
synth-validated upgrades XRT transparently — the FSM itself stays an optional
synth-phase ring-traffic optimisation.

### Phase 5 — mesa thin-shim onto `OP_DRAW`
**Goal:** mesa stops orchestrating; it submits one draw + present.
**Work:**
- `vp_draw_vbo`/`vp_raster_draw`: replace the host-built 27-command list **and the separate
  host-blocking VS launch** with one draw-descriptor submission (`OP_DRAW`). VS folds into the
  device draw program — kills the VS-finish-before-raster host round-trip.
**Exit:** the textured cube renders via a single device command; **no intra-draw host
round-trips** (verified: no host sync between pipeline stages); simx + rtlsim green.

### Phase 6 — Residency + remaining host-call elimination + binary cache
**Goal:** host touches device only at resource-create and present.
**Work:**
- Pooled **two-heap residency** for all per-draw buffers; **textures + attachments stay
  device-resident across draws** (no per-draw re-upload); only egress is present.
- mesa **shader-binary cache** (key by NIR/pipeline hash) — avoid per-pipeline recompile +
  the `/tmp` `.vxbin` round-trip (a mesa code-complete item).
**Exit:** a multi-draw frame = compile-once + upload-resident-once + N device draws + one
present; no per-draw host upload/readback except present; simx + rtlsim green.

### Phase 7 — P0 validation & sign-off
**Goal:** prove and record code-complete.
**Work:**
- Full gfx + `tests/vulkan` suites green on **simx AND rtlsim**, incl. the `graphics_parity`
  matrix (cores 1/2/4, multi-cluster, multi-raster, multi-drawcall) byte-exact.
- **Full cross-layer code review** (the P0 code-complete audit — do all four, top to bottom):
  1. **mesa_vortex implementations** — `vortexpipe` (compile, nir→llvm, context/draw, raster,
     launch, screen) + the lavapipe/Gallium glue: only v2 ISA emitted, device-orchestrated
     draw, residency, no host round-trips on the supported path, fallback clearly fenced.
  2. **Vortex graphics + RTU kernel ISA + ABI** — `sw/kernel/include/vx_graphics.h`,
     `sw/common/vx_gfx_abi.h`, the RTU intrinsics, the window/SETW ABI: single encoding set,
     no dead ops, host/device/SimX/RTL agree byte-for-byte.
  3. **Vortex graphics + RTU runtime stack** — `sw/runtime/**` (device/queue/CP submit,
     `vx_enqueue_draw`, `graphics.cpp`, `raytrace`) + the Emulation CP: minimal core,
     residency, OP_DRAW path, no legacy entry points.
  4. **"True GPU" alignment + zero-legacy sweep** — confirm the whole supported pipeline is
     device-orchestrated like a real GPU driver (submit-and-present, no intermediate host
     interrupts), and grep-prove NO legacy ISA / dead orchestration / stale code remains
     across sw + hw + mesa.
- **mesa code-complete checklist:** only v2 ISA emitted; device-orchestrated draw; resources
  resident; zero legacy; no intra-draw host round-trips.
- Update the bible + status/schedule to the P0-complete state; list what is deferred
  (OM v2, doctrine handle, SW fallback, GS/tess/mesh, CTS, synth, RTL CP OP_DRAW mirror).
**Exit:** **P0 sign-off.**

---

## Dependency graph
P1 → P3, P2 → P3 (delete after migrate). P3 → P4 (build orchestration on clean ISA).
P4 → P5 → P6 → P7. P1/P2 are independent and can run in parallel.

## Risk register
- **Phase 4 (OP_DRAW) is the long pole** — RTL CP maturity; budget the most time here.
- **rtlsim throughput** bottlenecks the parity gate — build light-trace/small-frame proxies.
- **tex4 instr-count** per sample (perf, not correctness) — watch; post-P0 fast-form option.
- **mesa "all issues resolved"** is scoped to the *supported* path; the unsupported-state
  host-llvmpipe fallback is deliberately retained (separate `libgfx_sw` track) — confirm this
  scope boundary is acceptable at sign-off.

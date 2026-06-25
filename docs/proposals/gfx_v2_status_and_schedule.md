# gfx_v2 — State Analysis & Revised Schedule (2026-06-25)

Ground-truth status of the gfx_v2 "true GPU" effort and a revised schedule.
Supersedes the 2026-06-24 revision (which itself superseded the stale §2 snapshot in
`gfx_v2_true_gpu.md`, 2026-06-20). Verdicts are evidence-backed (file:line / commit)
from a fresh code inventory, not plan text.

**Scope of "code-complete":** everything written *and validated in SimX + rtlsim*
(P2 FWD + P3 RTL FF parity + the non-hardware part of P4). The hardware tail — P5
(U55C AFU + 300 MHz closure) and P6 (Vulkan CTS) — is the north star
(`gfx_v2_true_gpu.md` §0) and is explicitly **out of code-complete scope**.

---

## Headline (what changed since 2026-06-24)

The **FWD epic (P2 — RASTER dispatch v2) is essentially done.** Since the last
revision, the four open FWD items all landed:

- **FWD-4c** (mesa FS lowering) — **COMMITTED** and clean (`~/dev/mesa_vortex`
  branch `prism`: `0288583281b` + `403a28aa128`); the RTTI build-block is resolved.
- **FWD-4d** (retire legacy `vx_rast` pull) — **DONE** (`b3673fa0`):
  `VX_raster_csr.sv` deleted; decode is single-path (`VX_decode.sv:787` funct3=3 →
  `frag_fetch` only, funct3=4 → `rast_begin`); SimX `RasterType` has only
  `BEGIN`/`FWD_RUN` (no `POP`); zero `VX_CSR_RASTER_*` references tree-wide.
- **FWD-5** (register-window payload, zero-LMEM C1) — **DONE** (`b01c51fe`):
  payload staged via the gfx window RF (`VX_raster_unit.sv` `win_wr_*` →
  `VX_gfx_window.sv` slots 8..21; SimX `gfx_window.h` `FRAG_SLOT_BASE=8,WORDS=14`;
  kernel `vx_frag_payload`/`vx_frag_load` via GETW). No longer "optional/pending."
- **FWD-6** (cores>1 correctness) — **RESOLVED, and it was never a raster bug**
  (`92d3d28f`, report `cta_dispatch_busy_gap_fix.md`). Root cause was a 1-cycle
  device-`busy` gap in the **CTA dispatcher** during the KMU→core handoff at
  `SOCKET_SIZE>1`, which made the rtlsim host stop clocking before the kernel ran.
  One-line fix in `VX_cta_dispatch.sv`. The whole "raster drain race / owner-routing"
  framing was a red herring. This is a **shared core-path** fix (non-graphics).

**Net:** the *dispatch* path is now **single-path, doctrine-clean, end-to-end**
(host → mesa FS → RTL/SimX → register window). Note the *output* path is NOT yet
doctrine-clean: `vx_om4` is still fire-and-forget (rd=x0) reading the cross-unit
shared graphics window, and `SETW` writes into it — both flagged `KnownViolation`
by `gfx_doctrine.h` (the §3.1 C3/C4 item; `a1e332bb` fixed only a window-cache
*corruption*, not the handoff). The remaining code-complete work is
no longer "build the dispatch redesign" — it is **(a) prove the RTL FF datapaths are
byte-exact vs SimX on the full parity matrix now that cores>1 is unblocked, and
(b) the autonomy/residency/SW-fallback tail of P4.**

---

## Corrected status vs. the 2026-06-24 revision

| Area | 06-24 verdict | Actual (06-25, verified) | Evidence |
|---|:--:|:--:|---|
| FWD-4c mesa FS lowering | uncommitted, build-blocked | **COMMITTED, clean** | mesa `0288583281b`,`403a28aa128`; `vp_nir_to_llvm.c:1373` |
| FWD-4d legacy deletion | PENDING (dual-path decode) | **DONE, single-path** | `b3673fa0`; `VX_decode.sv:787`; `VX_raster_csr.sv` gone |
| FWD-5 register-window payload | optional/deferred | **DONE** | `b01c51fe`; `VX_gfx_window.sv` slots 8..21 |
| FWD-6 cores>1 | PENDING (raster swamp) | **RESOLVED — CTA-dispatch busy gap, not raster** | `92d3d28f`; `cta_dispatch_busy_gap_fix.md` |
| Front-end cull modes | "no cull modes" | **PRESENT** (front/back) | `sw/gfx/setup_math.h:109-137` |
| Front-end near-plane clip | partial | **PRESENT** (Sutherland-Hodgman, 0–2 sub-tris) | `sw/gfx/setup_math.h:48-68`; `pipe_frontend.h:32-48` |
| RTL OM/TEX/RASTER datapaths | exist, parity unproven | **exist, full datapaths, parity still unproven** | `hw/rtl/{om,tex,raster}/*` (see below) |
| graphics_parity matrix | defined, cores>1 red/blocked | **defined; cores>1 now UNBLOCKED, run-green pending** | `ci/testcases/graphics_parity.yaml` |
| CP autonomous draw (CMD_DRAW) | DCR-config only | **still host-orchestrated** (9 launches + DCR writes) | `sw/runtime/graphics.cpp:534-542`; `cmd_processor.h:118-131` (no OP_DRAW) |
| FrontEndPool residency | 16 allocs | **14–16 separate allocs, not pooled** | `sw/runtime/graphics.cpp:480-495` |
| SW fallback | OM-only | **OM-only** (no SW raster/sampler) | `sw/common/gfx_sw.h:36-294` |
| Compiler stages | VS/FS/compute | **VS/FS/compute only** (GS/tess/mesh proposed) | `vx_graphics.h`; archived stage-coverage proposal |

---

## Layer-by-layer ground truth (06-25)

**SimX functional model — DONE.** RASTER (TE/BE walker, 12-byte header, cycle model
`raster_core.cpp walk_cycles_`/`RASTER_DRAIN`), OM (R-M-W + same-pixel interlock),
TEX (trilinear, `vx_tex4` single/quad). P0 doctrine assertion (`gfx_doctrine.h`) +
P1 gfx 7/7 green.

**RTL FF datapaths — EXIST as full datapaths (parity unproven).**
- OM: `VX_om_{core,ds,blend(+func/minmax/multadd),compare,logic_op,stencil_op,mem}.sv`
  — real depth/stencil/blend/logic-op/write-mask R-M-W, not a stub.
- TEX: `VX_tex_{core,addr,mem,sampler,format,lerp,wrap,stride,sat}.sv` — full
  wrap→addr(+mip)→fetch→filter pipeline.
- RASTER: `VX_raster_{core,te,be,qe,edge,slice,extents,mem,arb}.sv` — TE/BE walk +
  FWD `frag_fetch` window write (`VX_raster_unit.sv:47-56`).

**Parity harness — EXISTS and is correctly shaped.** `ci/testcases/graphics_parity.yaml`
runs every cell on **both simx and rtlsim** against the same golden at **tolerance 0**
(golden agreement *is* SimX↔RTL parity). Sweeps cores 1/2/4, 2 clusters × 2 cores,
4 cores + 2 raster cores, on `box.cgltrace` (2-drawcall) + `triangle.cgltrace`.

**P4 autonomy/residency — early.** Draw is host-orchestrated (no CP `OP_DRAW`);
FrontEndPool is 14–16 discrete allocs (no pooled slab, no two-heap PA/VA residency);
SW fallback is OM-only (no SW raster/sampler); per-sample sub-triangle clip feedback
and binning-overflow back-pressure absent. Cull + near-clip are present.

**Dead-code tail (FWD-4d hygiene).** `vx_rast()` inline (`vx_graphics.h:96`), the
stale funct7=0 comment (`vx_graphics.h:28`), and `raster_core.{h,cpp}` /
`VX_raster_arb.sv` / `VX_raster_core.sv` comments still mention the retired pull op.
No decode/SimX support and no caller — pure cleanup.

---

## Revised schedule

| Phase | Status | Contents | Exit gate |
|---|:--:|---|---|
| P0 interface law | **DONE** | doctrine assert + parity matrix | — |
| P1 SimX gfx green | **DONE** | 7/7 + matrix green on SimX | — |
| P2 RASTER dispatch v2 (FWD) | **~95%** | FWD-3/4a/4b/4c/4d/5 done; cores>1 unblocked (FWD-6). Remaining: dead-code cleanup + cores>1 byte-exact confirmation (folds into P3) | box 2-draw × multi-core byte-exact vs SimX gold |
| **P3 RTL FF parity** *(critical path now)* | **~35%** | run `graphics_parity` green on **rtlsim across all cells** (cores 1/2/4, multi-cluster, multi-raster, box 2-draw); fix gaps the migrated OM/TEX/RASTER datapaths surface (existence ≠ parity); light-trace/small-frame proxies for rtlsim runtime | rtlsim parity green on the full matrix |
| P4 autonomy+residency | **~20%** | SW raster+sampler fallback; FrontEndPool→pooled slab + two-heap residency; CP `CMD_DRAW`/`RES_GFX`; per-sample clip feedback + binning overflow back-pressure | frame renders device-resident, host-untouched, on rtlsim |
| P5 U55C bring-up | 0% | graphics AFU + 300 MHz closure + on-card draw3d | draw3d correct on U55C @ 4 cores |
| P6 conformance | 0% | Vulkan CTS harness; SW-fallback completeness; GS/tess/mesh as needed | Vulkan CTS pass on U55C @ 4 cores |

**Code-complete critical path:** **P3 parity-matrix-green → P4 functional subset.**
P2 is done bar cleanup; P5/P6 are the out-of-scope hardware/conformance tail.

---

## Code-complete task plan (ordered)

1. **P3 — run the parity matrix green on rtlsim (THE #1 risk).** The OM/TEX/RASTER
   RTL datapaths *exist* but have never been proven byte-exact vs SimX gold; the
   cores>1 cells were blocked until the FWD-6 busy-gap fix. Now unblocked:
   - Run `ci/testcases/graphics_parity.yaml` rtlsim cells end-to-end (cores 1/2/4,
     2-cluster, 4c+2raster, box 2-draw + triangle).
   - First build **light-trace / small-frame parity proxies** — rtlsim is ~60 s/drawcall
     on heavy traces and will bottleneck the matrix otherwise.
   - Fix whatever the migrated v2-interface datapaths surface (likely OM/TEX window
     ABI or RASTER header edges). *Existence ≠ parity.*
   - Confirms the FWD-6 fix at the matrix level (box 2-draw cores=2/4 byte-exact),
     closing the P2 exit gate too.
2. **FWD-4d cleanup tail** — ✅ **DONE** (`ca85ec17`): dead `vx_rast()` intrinsic +
   stale comments retired; dispatch is single-path.
3. **OM/SETW doctrine C3/C4** *(correctness, surfaced 2026-06-25)* — give `vx_om4` a
   scoreboard handle (parity with `vx_tex4`) and move to per-unit scoreboard-retired
   windows, retiring the cross-unit shared graphics window (and the `SETW` C4 write).
   This is the §3.1 item still flagged `KnownViolation` in `gfx_doctrine.h`; the
   interface law (§1.3) is meant to eliminate it by construction.
4. **P4(a) — SW raster + sampler fallback** *(largest P4 gap, also a P6 dependency).*
   Today `gfx_sw.h` is OM-only. Add on-device SIMT software raster + sampler so the
   three-tier per-unit dispatch (native FF → HW-composed → SIMT software) is real;
   rtlsim-validatable.
5. **P4(b) — residency.** FrontEndPool scratch slab ✅ **DONE** (`9fd13baa`, 14→3
   allocs, scratch pooled + FF-pinned outputs split). Remaining: formalize the
   two-heap split (FF-pinned-PA vs shader-paged-VA) per the residency design; harden
   binning-queue overflow/back-pressure (no host restart exists under full residency).
5. **P4(c) — CP autonomous draw.** Add `OP_DRAW`/`RES_GFX` so the CP sequences
   VS→setup→bin→raster→FS→OM and programs FF config device-side (collapses the host's
   9-launch + DCR-write sequence). Built on the RTL CP / Emulation CP; rtlsim-validatable.
6. **P4(d) — front-end finish.** Per-sample sub-triangle clip feedback in setup_k;
   guardband clip; min-z for Hi-Z. (Cull + near-clip already done.)

P5 (U55C AFU + 300 MHz) and P6 (Vulkan CTS + GS/tess/mesh) are the hardware/
conformance tail — tracked, out of this code-complete scope.

---

## Key risks / corrections to carry

1. **P3 parity is the gate, and it is unproven, not merely unrun.** The migrated
   OM/TEX RTL came from skybox and has never diffed byte-exact against the current
   v2 SimX gold. Budget for surfaced gaps, not a clean pass.
2. **rtlsim throughput (~60 s/drawcall heavy)** bottlenecks the matrix — light-trace/
   small-frame proxies first.
3. **SW fallback is OM-only** — P6 conformance needs SW raster+sampler; larger than
   the original plan implied, and it is on the code-complete list now (P4a) because
   the dual-path doctrine is a correctness property, not just a CTS convenience.
4. **The FWD-6 fix is in the shared CTA dispatcher**, so re-check SimX↔RTL **compute**
   parity at cores>1 too (see `project_simx_rtl_parity`), not just graphics.
5. **Pre-existing `VX_allocator.sv` double-release at cores≥2 on heavy traces** was
   cited as a multi-core blocker; verify whether it still reproduces post-FWD now
   that the raster dispatch path changed, or whether it was also downstream of the
   never-launched kernel.

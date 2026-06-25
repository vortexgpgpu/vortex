# gfx_v2 — P3: vx_om4 on the shared graphics window

**Scope:** make the output-merger (OM) read its per-fragment payload from the
shared per-core graphics window (P1.0), exactly as `vx_tex4` did for TEX (P1/P2).
`vx_om4` is the **sole** OM submit op (om_v2 §3): it subsumes the full 2×2 quad,
partial-coverage edges, and the degenerate single-fragment case via a 4-bit
coverage mask — so the gfx-v1 `vx_om` (CUSTOM1 funct3=2, R4-type) is **removed**
and `vx_om4` reclaims funct3=2 as R-type. There is no free CUSTOM1 funct3 slot
for a coexistence period (unlike tex4's funct3=5), so the legacy op is replaced
outright and validated against the existing `gfx_om` golden reference images.
**Status:** Proposal — implements [gfx_v2_om_v2.md](gfx_v2_om_v2.md) §3 on the
[gfx_v2_window_p1_0.md](gfx_v2_window_p1_0.md) foundation; sibling of
[gfx_v2_tex4_p1.md](gfx_v2_tex4_p1.md)/[gfx_v2_tex4_p2.md](gfx_v2_tex4_p2.md).
**Tree:** `~/dev/vortex_v3/prism_v3`.
**Date:** 2026-06-14.

---

## 1. Encoding

`vx_om4` lands at **CUSTOM1 funct3=2, R-type**, replacing the legacy R4-type
`vx_om` (R-type and R4-type cannot share one funct3 — the 31:25 bits are
`funct7` vs `{rs3,funct2}` with no disambiguating opcode bit, so the reclaim is
a replacement, not an addition). `rd = x0` (OM submit is fire-and-forget, no
register writeback — the analog of a custom store).

```
  vx_om4  rs1 = quad descriptor (per-lane),  rs2 = payload window slot base,
          funct7 = sub-op (reserved = 0 in P3),  rd = x0
```

- **`rs1` — quad descriptor (per-lane).** The `pos_mask` the FS already gets
  back from `vx_rast`, with `face` in the top bit:
  ```
    bits [3:0]                       cov_mask           (covered sub-pixels)
    bits [4 +: VX_RASTER_DIM_BITS-1] qx                 (quad x, 14 bits)
    bits [18 +: 13]                  qy                 (quad y, 13 bits)
    bit  [31]                        face               (front/back)
  ```
  `face=0` ⇒ the descriptor is the raw `pos_mask` (its bit 31, the y MSB, is 0
  for any qy < 8192 — a 16384-px height ceiling, far above the §3.8 limits and
  every test). Per sub-pixel `F`: `pos_x = (qx<<1)|(F&1)`, `pos_y = (qy<<1)|(F>>1)`
  — bit-identical to the gfx-v1 `OUTPUT_i` macro.
- **`rs2` — payload window slot base** (warp-uniform; lane-0 value, as tex4
  reads `in_slot`). Per-lane payload window, single render target:
  ```
    base+0 .. base+3 : color[0..3]   one packed RGBA8 word per sub-pixel
    base+4 .. base+7 : depth[0..3]   VX_OM_DEPTH_BITS fixed-point per sub-pixel
  ```
- **`funct7` — sub-op.** Reserved (0) in P3. The om_v2 `{RT_count, sample_count,
  op}` fields (MRT, per-sample, replace/fetch composition) ride here in later
  steps.

The FS stages the (already-computed) four colors/depths with `SETW` and issues
one `vx_om4` per quad — replacing the per-covered-sub-pixel `vx_om` loop with a
single quad-batched submit (fewer issues; the efficiency win om_v2 §3 cites).

## 2. Dataflow

```
  FS:  SETW base+0..3 <- color[0..3]                 (window writes)
       SETW base+4..7 <- depth[0..3]
       vx_om4 (rs1=desc, rs2=base)                   (submit the quad)
```

`vx_om4` reads `color[0..3]`/`depth[0..3]` for all active lanes from the window
at issue, derives each covered sub-pixel's `(pos_x,pos_y,face)` from the
descriptor, and submits to the **existing** `om_bus_if` → `VX_om_core` →
depth/stencil → blend → ocache RMW path — byte-identical OM math to `vx_om`, so
the framebuffer equals the gfx-v1 result (the validation oracle).

## 3. RTL — `VX_om_unit` (4-sub-pixel sequencer)

One `vx_om4` expands into up to four `om_bus` requests, one per sub-pixel `F`,
each carrying all active lanes' fragment-`F` data — mirroring the tex4 quad
sequencer, but simpler (OM is fire-and-forget: no response round-trip, no window
writeback, no result reassembly):

- **Window read** (8 ports, shared with TEX, §5): `color[F]=cons_rd_data[F]`,
  `depth[F]=cons_rd_data[4+F]`, for all lanes; `cons_rd_slot[p]=base+p`.
- **Sequencer**: a `q_frag` counter issues `F=0..3` on successive cycles, each
  an `om_bus` request whose per-lane `mask = tmask[lane] & cov_mask[lane][F]`,
  `pos_x=(qx<<1)|(F&1)`, `pos_y=(qy<<1)|(F>>1)`, `color`/`depth` from the window,
  `face=desc[31]`. `execute_if.ready` asserts on the 4th issue; the trace then
  retires through `result_if` with no data (`rd=x0`).
- **Empty-`F` skip** (efficiency): if no lane covers sub-pixel `F`
  (`|(tmask & cov_mask[*][F]) == 0`), the sequencer advances without an `om_bus`
  request — interior quads (`cov_mask=0xf`) issue all four; edge quads issue
  only the covered ones.

`VX_om_core` and `om_bus_if` are **unchanged** — they already accept a per-lane
masked `{pos_x,pos_y,color,depth,face}` request.

## 4. ABI — `op_args.om` + decode

`om_args_t` (today all-padding) gains `is_om4` (1 bit). `VX_decode.sv` funct3=2
becomes R-type: `USED_IREG(rs1)`, `USED_IREG(rs2)`, **no** `rs3`, `rd=x0`
(no writeback). SimX `decode.cpp` case 2 likewise drops the rs3 source and sets
`IntrOmArgs.is_om4`.

## 5. Cross-PE window access — sharing the read ports

TEX (P1/P2) already exposes an 8-port window read surface
(`gfxw_cons_rd_*` in `VX_sfu_unit`). OM needs the same 8 reads
(`color[0..3]`,`depth[0..3]`). Because the SFU `VX_pe_switch` demuxes **one** op
to **one** PE per cycle, and a multi-cycle macro-op (tex4 quad / om4) holds
`execute_if.ready` low — stalling the switch input — **TEX and OM never read the
window in the same cycle**. So OM shares the *single* `cons_rd` port set via a
select mux (driven by whichever PE is issuing a window op), rather than
duplicating 8 RF read ports — the area-cheap choice that matters for the 300 MHz
target. OM drives no write port (no window result). Tie-off now keys on
`EXT_TEX || EXT_OM` rather than `EXT_TEX` alone, so an OM-only build still wires
the window.

## 6. SimX

`OmUnit::process` gains a `frag` parameter (like `TexUnit::process`); `sfu_unit`
adds a per-block om quad sequencer (`q_frag`/`q_issued`, one fragment in flight),
reading `color[F]`/`depth[F]` from the shared window
(`rtu_unit_->window_get`) and submitting fragment `F`. The window storage still
lives in `RtuUnit` (the SimX extraction is deferred — P1.0 §5), so the OM tests
co-enable `EXT_RTU` in their SimX path, exactly as `gfx_tex4q` does.

## 7. Caller migration (vx_om removed)

`vx_om` is removed, so all four in-tree callers migrate to `vx_om4`:

- **`gfx_om`** (one thread per pixel): the thread stages its single
  color/depth into the covered sub-pixel slot and issues `vx_om4` with a
  1-bit `cov_mask`. Validated against the existing `whitebox_*.png` references.
- **`gfx_draw3d`, `gfx_pipeline_tex`, `gfx_pipeline_om`** (FS poll-loop): the
  `OUTPUT_QUAD` macro stages `color[0..3]`/`depth[0..3]` with `SETW` and issues
  one `vx_om4(pos_mask | (face<<31), base)` — replacing the four-iteration
  `vx_om` loop. Validated against their existing golden images.

All four co-enable `EXT_RTU` (+`RTU_BVH_WIDTH=0`) so the SimX window is present
(§6); these are unchanged on rtlsim where the window rides `EXT_GFX_ANY`.

The out-of-tree mesa `emit_vx_om` (raw `.insn r4 43,2,…`) is migrated to `vx_om4`
separately on the mesa branch — like P1.0/P1/P2, mesa lags the in-tree ISA.

## 8. Validation

Commit when all are green on **SimX and rtlsim**:
1. `gfx_om` — golden image (`whitebox_*`), the OM oracle.
2. `gfx_pipeline_om`, `gfx_pipeline_tex`, `gfx_draw3d` — golden images
   (the full FS→OM quad path).
3. `gfx_tex4q` + `gfx_tex4` — regression (the shared `cons_rd` mux touches the
   TEX read path).
4. RTU suite (`tests/raytracing`) — stays green (SimX 23/23, rtlsim 18/18).

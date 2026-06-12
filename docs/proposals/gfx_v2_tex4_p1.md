# gfx_v2 — P1: vx_tex4 single mode on the shared window

**Scope:** introduce `vx_tex4` (single mode) — the first FF graphics op that
takes its payload from, and returns its result to, the shared per-core graphics
window (P1.0). Validate it produces texels **identical to `vx_tex`**. Builds the
cross-PE window access mechanism that OM (P3) reuses.
**Status:** Proposal — implements [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) §3 (single mode) on the [gfx_v2_window_p1_0.md](gfx_v2_window_p1_0.md) foundation.
**Tree:** `~/dev/vortex_v3/prism_v3`.
**Date:** 2026-06-12.

---

## 1. Encoding (P1 coexistence)

`vx_tex4` lands at **CUSTOM1 funct3=5** (the free slot; 0=wgather, 1=tex, 2=om,
3/4=raster, 6=gfxw, 7=rtu), **R-type**, so it coexists with `vx_tex` (funct3=1,
R4-type) for the `== vx_tex` validation. P7 (delete legacy v1 ISA) migrates it to
funct3=1 and removes `vx_tex`. (R-type and R4-type cannot share one funct3 — the
31:25 bits are `funct7` vs `{rs3,funct2}` with no disambiguating opcode bit.)

```
  vx_tex4  rd-field = output-slot base,  rs1 = LOD (real reg),
           rs2-field = input-slot base,  funct7 = { mode, stage, <reserved> }
```

- **funct7[0] = mode** — 0 = single (P1), 1 = quad (P2).
- **funct7[1 +: VX_TEX_STAGE_BITS] = stage.**
- **rs1** is a real register operand (explicit LOD), marked `USED_IREG`.
- **rs2-field / rd-field** are reinterpreted as window **slot indices** (0..31,
  fit in the 5-bit register fields), **not** register reads/writes — so they are
  *not* marked used, and `wb=0` (no register writeback). `header.rd` carries the
  output-slot base through to the response.

Single-mode window layout: input `base+0 = u`, `base+1 = v` (S.23 fixed-point);
output `out_base+0 = texel` (packed RGBA8).

## 2. Dataflow (single mode)

```
  FS:  SETW in_base+0 <- u ;  SETW in_base+1 <- v          (window writes)
       vx_tex4 (rs1=lod, rs2=in_base, rd=out_base, stage)  (sample)
       GETW out_base -> reg                                (read texel back)
```

`vx_tex4` reads `u,v` from the window at issue, runs the **existing** async
sample path (`tex_bus_if` → `VX_tex_core` → tcache → texel), and writes the texel
into the window slot on return. The sampling math is byte-identical to `vx_tex`,
so the returned texel equals `vx_tex(stage, u, v, lod)` — the validation oracle.

## 3. Cross-PE window access (the reusable mechanism)

The window RF lives in `VX_gfx_window`; the TEX datapath is the separate
`VX_tex_unit` PE. P1 adds a generic **consumer access** surface to
`VX_gfx_window`, wired PE↔PE inside `VX_sfu_unit`:

- **Combinational read port** — `cons_rd_wid`, `cons_rd_slot[RD_PORTS]` →
  `cons_rd_data[RD_PORTS][NUM_LANES]`. TEX drives `slot[0]=in_base`,
  `slot[1]=in_base+1` to fetch `u,v` for all active lanes at issue. `RD_PORTS=2`
  in P1 (quad mode widens it in P2).
- **Write port** — `cons_wr_en`, `cons_wr_wid`, `cons_wr_mask[NUM_LANES]`,
  `cons_wr_slot`, `cons_wr_data[NUM_LANES]`. TEX drives it when the sampled texel
  returns, writing `out_base` for the active lanes. Merged into the window RF
  write logic alongside `SETW` (distinct warp/slot in practice; defined priority
  otherwise).

`VX_tex_unit` already has a response **tag-store** ([VX_tex_unit.sv](../../hw/rtl/tex/VX_tex_unit.sv))
that echoes `rd`/`wid`/`tmask`/`wb` across the async round-trip — `vx_tex4` reuses
it directly: `header.rd` = output slot, `wb=0`. On response it drives `cons_wr`
instead of a register writeback; the op still retires through `result_if` (so the
scoreboard clears) but commits no register.

When no FF consumer is present (e.g. the RTU-only config), the consumer ports are
tied off (`cons_wr_en=0`) — the window is byte-identical, so the RTU suite
re-validates unchanged. OM (P3) drives the same ports (read its payload window;
no result write).

## 4. SETW → vx_tex4 ordering

`SETW` writes a window slot; `vx_tex4` reads it. The scoreboard tracks *register*
deps, not window-slot deps — same as the RTU's `SETW`→`CB_RET`/`GETW` path. P1
relies on in-order warp program order + the 1-cycle synchronous window write
(the `SETW` commits before the later-issued `vx_tex4` reads). The `== vx_tex`
validation catches any hazard; if one surfaces it is fixed at root (a window-slot
fence / mini-scoreboard), not papered over.

## 5. Validation config: RTU + TEX

P1 validates under a config with **both** `EXT_RTU` and `EXT_TEX` enabled. This
is the smallest config in which *both* models already have everything `vx_tex4`
needs:
- **RTL** — `VX_gfx_window` (P1.0) already serves any graphics ext; the consumer
  ports (§3) wire the TEX PE to it. Fully general (TEX-only also works).
- **SimX** — `SETW`/`GETW` and the window regfile live in `RtuUnit` today, so
  co-enabling RTU makes them available without first porting the SimX decode/
  dispatch off `EXT_RTU`. P1 shares only the *storage*: a small `GfxWindow`
  (the `regfile_` array, Core-owned under `EXT_GFX_ANY`) that both `RtuUnit`
  (its existing `SETW`/`GETW`/trace logic, now on the shared array) and the new
  `vx_tex4` model read/write. The full SimX `SETW`/`GETW`-without-RTU extraction
  (so a TEX-only SimX build has the window) is a later step, not needed for P1.

The SimX `TexUnit` gains the `vx_tex4` path: read `u,v` from the shared
`GfxWindow` slots, call the same `TextureSampler::read` (the oracle), write the
texel to the output slot. `decode.cpp` decodes funct3=5; `sfu_unit.cpp` routes it.

## 6. Validation

- A new `tests/graphics/gfx_tex4` (CONFIGS = `EXT_RTU` + `EXT_TEX`) samples each
  texel **both** ways — `vx_tex(stage,u,v,lod)` and the `SETW`/`vx_tex4`/`GETW`
  sequence — and asserts the two are **bit-identical** per pixel, on SimX and
  rtlsim.
- RTU suite re-validates unchanged (consumer ports tied off in the RTU-only
  build; SimX storage-sharing is behaviourally identical).

Commit when both are green.

# gfx_v2 — CUSTOM1 ISA Encoding Allocation (graphics + RTU)

**Scope:** the shared **CUSTOM1** (`INST_EXT2 = 0x2B`) opcode budget — how the
graphics ops (RASTER/TEX/OM) and the PRISM RTU ops co-exist in the scarce
funct3 space, and how gfx_v2's redesigned/added ops (`vx_tex4`, `vx_om4`,
`vx_om_fetch`, modes) fit **without** a new funct3 row or an RTU collision. A
cross-cutting check before any graphics-ISA RTL.
**Reference:** [custom_accelerator_isa_extensions.md](../designs/custom_accelerator_isa_extensions.md)
§2.5 (R2 vs R4); `hw/rtl/VX_gpu_pkg.sv`, `hw/rtl/core/VX_decode.sv`;
prism_v3 `sw/kernel/include/vx_raytrace.h`.
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — supports [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) /
[gfx_v2_om_v2.md](gfx_v2_om_v2.md) §3.
**Date:** 2026-06-07.
**Related:** [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md),
[gfx_v2_om_v2.md](gfx_v2_om_v2.md),
[gfx_v2_ff_expansion_roadmap.md](gfx_v2_ff_expansion_roadmap.md).

---

## 1. The budget

CUSTOM opcode space is scarce: 4 major opcodes (`INST_EXT1..4` = custom0/1/2/3),
each with 8 funct3 rows. The graphics + RTU extensions live on **CUSTOM1**
(`0x2B`), decoded by funct3 into `INST_SFU_*` ops
([VX_gpu_pkg.sv:276,508-517](../../hw/rtl/VX_gpu_pkg.sv#L508),
[VX_decode.sv:733-775](../../hw/rtl/core/VX_decode.sv#L733)). Current map:

| CUSTOM1 funct3 | Owner | Op(s) | Form |
|---|---|---|---|
| 0 | — | **free** | — |
| 1 | graphics | TEX | R4 today → R-type v2 (`vx_tex4`) |
| 2 | graphics | OM | R4 today → R-type v2 (`vx_om4`) |
| 3 | graphics | RASTER | R (`vx_rast`) |
| 4 | graphics | RASTER | R (`vx_rast_begin`) |
| 5 | **RTU** | set/get/trace/wait (phase-1) | R (slot in funct7, sub-op in funct2 bits) |
| 6 | **RTU** | actions / callback-window (GETWF) | R |
| 7 | **RTU** | ISA v2 (trace2/wait2, multi-AS) | R |

**Only funct3 = 0 is free.** Graphics has {1,2,3,4}; RTU has {5,6,7}. So a new
graphics op **cannot claim its own funct3 row** without taking the last free
slot or colliding with the RTU.

(At the `INST_SFU_*` op-enum level — TEX=`0xB`, OM=`0xC`, RASTER=`0xD`,
RTU=`0xE` — `0xF` is free; but the binding constraint for CUSTOM1 instructions is
the 3-bit **funct3**, not the 4-bit SFU op.)

---

## 2. The gfx_v2 strategy: stay in funct3 {1,2}, sub-encode in `funct7`

gfx_v2 keeps graphics within its existing funct3 rows and expresses every
new/added behavior in the **7-bit `funct7`** of the R-type encoding — which is
exactly why TEX/OM were chosen as **R-type (R2)** over R4 (the doctrine §2.5
recommendation pays off here). No new funct3 row is consumed; the RTU's
{5,6,7} are untouched; funct3 = 0 stays as headroom.

### 2.1 funct3 = 1 — TEX (`vx_tex4`, sole op)

`vx_tex4` (R-type) carries all modes in `funct7`:

```
  funct7 = { stage[…], mode(quad | single | raw-fetch), mip-filter, Dref, format-ovr }
```

This subsumes the gfx-v1 `vx_tex` (single/explicit-LOD = a mode) and the
`texelFetch` raw-fetch composition primitive — all one op, one funct3.

### 2.2 funct3 = 2 — OM (`vx_om4` + `vx_om_fetch`, sole funct3)

Both the submit (`vx_om4`, no `rd`) and the read-half (`vx_om_fetch`, has `rd`)
share funct3 = 2, distinguished by `funct7`:

```
  funct7 = { op(submit | fetch), mode(fused | per-sample | replace), RT_count, sample_count }
```

`vx_om_fetch` is **not** a new funct3 row — it's the `op = fetch` encoding of the
OM row (the decoder keys `rd` use on that bit). So OM's composition primitives
(om_v2 §3.4) cost zero extra funct3.

### 2.3 funct3 = 3, 4 — RASTER

`vx_rast` / `vx_rast_begin` unchanged (ISA stable; the binning redesign is RTL
front-end only, [gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md) §6.3).

---

## 3. Headroom & rules

- **funct3 = 0** is the single free CUSTOM1 row — reserve it; do not spend it on
  a mode that `funct7` can carry.
- **`funct7` is the graphics growth space.** New TEX/OM modes/formats extend the
  `funct7` sub-op fields, never a new funct3.
- **RTU owns {5,6,7}** — graphics must never decode those (and vice-versa). This
  doc is the single place that records the boundary; the RTU side is
  prism_v3 `vx_raytrace.h` + the `rtu_isa_v2_*` proposals.
- If graphics ever genuinely needs a 5th funct3 (it should not, given `funct7`),
  funct3 = 0 is the only option and must be coordinated with the RTU owners.

---

## 4. Open items

- **Exact `funct7` bit assignment** for TEX and OM (stage/mode/format widths) —
  finalize with the RTL decoder.
- **`rd`-use decode on funct3 = 2** — confirm the decoder cleanly distinguishes
  `vx_om4` (no `rd`) from `vx_om_fetch` (`rd`) by the `op` bit.
- **Unified ISA-map ownership** — keep this table and the RTU's in sync; a single
  generated `VX_types.toml` enum block would prevent drift.

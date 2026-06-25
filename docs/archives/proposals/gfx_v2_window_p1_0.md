# gfx_v2 — P1.0: the shared per-core graphics window

**Scope:** the foundation step for the `vx_tex4` / `vx_om4` migration — relocate
the per-(warp, lane) **slot-window register file** and the generic
`SETW` / `GETW` / `GETWF` macro-ops **out of the RTU** into a shared, un-gated
per-core **graphics window** mechanism that TEX, OM (and the RTU) all consume.
Zero behaviour change to the ray-tracing path; the full `tests/raytracing`
suite must remain bit-identical.
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.8 foundation; prerequisite for [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) §3.2/§7 and [gfx_v2_om_v2.md](gfx_v2_om_v2.md) §3.2/§7.
**Tree:** `~/dev/vortex_v3/prism_v3`.
**Date:** 2026-06-12.
**Related:** [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md), [gfx_v2_om_v2.md](gfx_v2_om_v2.md).

---

## 1. Why

Both `vx_tex4` (tex_v2 §3.2/§7) and `vx_om4` (om_v2 §3.2/§7) **resolved their
payload layout to "register window, reusing the RTU's `SET`/`GET` slot-window
macro-op mechanism."** Today that mechanism is **owned by, and compiled only
with, the RTU**:

- The per-`(warp, lane)` × 32-slot × 32-bit register file lives inside
  [hw/rtl/rtu/VX_rtu_unit.sv](../../hw/rtl/rtu/VX_rtu_unit.sv) (`regfile`).
- The op selector (`SETW`/`GETWF`/`GETW`/`TRACE2`/`WAIT2`/`CB_RET`), the RF
  dimensions, and the `op_args` payload all live behind `VX_CFG_EXT_RTU_ENABLE`
  ([VX_rtu_pkg.sv](../../hw/rtl/rtu/VX_rtu_pkg.sv), [VX_gpu_pkg.sv](../../hw/rtl/VX_gpu_pkg.sv) `rtu_args_t`, [VX_decode.sv](../../hw/rtl/core/VX_decode.sv) funct3=6/7).
- The SFU op-type `INST_SFU_RTU`, its PE, the uop expander
  ([VX_rtu_uops.sv](../../hw/rtl/rtu/VX_rtu_uops.sv)), and the SimX `RtuUnit`
  mirror are all RTU-gated.

A TEX-only or OM-only build therefore has **no window at all**. P1.0 makes the
window a first-class, RTU-independent per-core resource so P1+ can attach the
FF units to it.

## 2. The cut: window mechanism vs RT traversal

The RTU is two separable things glued in one module:

| Stays **generic** (the window mechanism) | Stays **RT-specific** (the traversal engine) |
|---|---|
| the `regfile[warp][lane][slot]` storage | `status` / `rt_scene` latches, `terminal_ready` |
| `SETW` (1-slot write), `GETW`/`GETWF` (windowed reads) | the bus FSM (`B_REQ…B_RSP2`), `rtu_bus_if`, `async_trap_if` |
| the op-selector enum + RF dimensions | `TRACE2` / `WAIT2` / `CB_RET` semantics + the `CFG/ORIGIN/DIR/ARM` fills |
| the `GETWF`/`GETW` uop expansion | the `TRACE2` 4-uop expansion; ray-snapshot, hit-window writes |

P1.0 keeps **both halves in one module** (renamed `VX_rtu_unit` →
`VX_gfx_window`) but changes their **gating**: the generic half is present
whenever **any** graphics extension is on (`EXT_GFX_ANY_ENABLE`); the RT half is
`` `ifdef VX_CFG_EXT_RTU_ENABLE `` *inside* the module. Keeping the RF and the
RT FSM physically together is what makes "zero behaviour change" provable — for
an RTU build the synthesised logic is identical, only relabelled. Cross-PE
access ports for the TEX/OM PEs are **not** added here; they arrive in P1/P3
with their first consumer (the window's reusable surface in P1.0 is the
RF + the generic `SETW`/`GETW`/`GETWF` ops on their own un-gated op-type).

## 3. Gating matrix

`EXT_GFX_ANY_ENABLE` already exists in [VX_define.vh](../../hw/rtl/VX_define.vh)
(set when TEX **or** RASTER **or** OM **or** RTU is enabled). P1.0 adds its
numeric twin `EXT_GFX_ANY_ENABLED` (0/1) for PE-count arithmetic, and a SimX
`VX_CFG_EXT_GFX_ANY_ENABLE` equivalent.

| Artifact | gfx-v1 gate | P1.0 gate |
|---|---|---|
| RF + `SETW`/`GETW`/`GETWF` op enum + RF dims | `EXT_RTU` (`VX_rtu_pkg`) | `EXT_GFX_ANY` (new `VX_gfx_window_pkg`) |
| `op_args` payload struct | `EXT_RTU` (`rtu_args_t` / `.rtu`) | `EXT_GFX_ANY` (`gfxw_args_t` / `.gfxw`) |
| SFU op-type | `INST_SFU_RTU` (0xE, `EXT_RTU`) | `INST_SFU_GFXW` (0xE, `EXT_GFX_ANY`) |
| the unit | `VX_rtu_unit` (`EXT_RTU`) | `VX_gfx_window` (`EXT_GFX_ANY`); RT FSM `ifdef EXT_RTU` inside |
| uop expander | `VX_rtu_uops` (`EXT_RTU`) | `VX_gfxw_uops` (`EXT_GFX_ANY`); `TRACE2` arm `ifdef EXT_RTU` inside |
| RT traversal (`VX_rtu_pkg`, walker, box/tri PE, bus, trap) | `EXT_RTU` | **unchanged** — `EXT_RTU` |

The op-selector enum keeps its values (`SETW=0, TRACE2=4, WAIT2=5, GETWF=6,
GETW=7, CB_RET=8`) so `op_args.gfxw.op` is one disjoint namespace shared by the
generic and RT ops; the RT values are simply inert in a non-RTU build.

## 4. Decode split

`VX_decode.sv` CUSTOM1:

- `funct3=6, funct2∈{1,2,3}` (`SETW`/`GETWF`/`GETW`) → `INST_SFU_GFXW`,
  **un-gated** (`EXT_GFX_ANY`).
- `funct3=6, funct2=0` (`CB_RET`) and `funct3=7` (`TRACE2`/`WAIT2`) →
  `INST_SFU_GFXW` but **emitted only under `EXT_RTU`** (RT-specific ops still
  ride the same op-type/PE; the window module dispatches them under its
  internal `EXT_RTU`).

The uop-sequencer match (`VX_uop_sequencer.sv`) routes `INST_SFU_GFXW` +
(`GETWF`/`GETW`) to `VX_gfxw_uops` un-gated, and + `TRACE2` only under `EXT_RTU`.

## 5. SimX — unchanged in P1.0 (extraction deferred to P1)

The SimX functional model is **independent of the RTL relabel**: it has its own
decode (`decode.cpp` → `RtuType` / `IntrRtuArgs`) and dispatch (`sfu_unit` by
`RtuType`), and the window register file is already a *separable member*
(`RtuUnit::regfile_` + `process_set` / `process_getw_uop`), not fused into a
gated op-type/PE the way the RTL was. Nothing in P1.0 references the RTL
`op_args`/`INST_SFU` names from SimX, so the RTL change leaves SimX
**bit-identical** — it simply re-passes the suite.

The asymmetry is deliberate: the RTL *had* to un-gate now because the op-type,
PE, package and `op_args` union were gated as one whole; SimX does not, because
its regfile is already extractable and there is **no SimX window consumer until
the SimX TEX model lands in P1**. Extracting a SimX `GfxWindow` now would be
speculative (its exact surface is set by that first consumer) and would risk a
behaviour-change for zero P1.0 benefit. So the SimX `RtuUnit::regfile_` →
shared `GfxWindow` extraction (mirroring §2, built under `EXT_GFX_ANY`) is done
in **P1**, alongside the SimX `vx_tex4` consumer that defines its access shape.
P1.0's SimX deliverable is therefore: no edits, full `tests/raytracing` re-pass.

## 6. Files

**New:** `hw/rtl/gfx/VX_gfx_window_pkg.sv`, `hw/rtl/gfx/VX_gfx_window.sv`
(from `VX_rtu_unit.sv`), `hw/rtl/gfx/VX_gfxw_uops.sv` (from `VX_rtu_uops.sv`);
`sim/simx/gfx/gfx_window.{h,cpp}` (generic half of `rtu_unit.*`).

**New (SimX):** none in P1.0 (the `GfxWindow` extraction lands in P1, §5).

**Edited:** `VX_define.vh` (numeric `EXT_GFX_ANY_ENABLED`), `VX_gpu_pkg.sv`
(`gfxw_args_t`/`.gfxw`, `INST_SFU_GFXW`, `UOP_GFXW`), `VX_decode.sv`,
`VX_uop_sequencer.sv`, `VX_sfu_unit.sv` (GFXW PE), `VX_rtu_pkg.sv` (drop the
moved enum/dims), and the four **RTL** build makefiles that gate `EXT_RTU`
(`sim/rtlsim`, `hw/syn/xilinx/{xrt,dut/rtu,dut/rtu_top}`) — each gains a
`gfx` package + include dir under `EXT_GFX_ANY`. `VX_execute.sv` needs no change
(it forwards the RTU bus/trap interfaces under `EXT_RTU` unchanged). No SimX
edits.

## 7. Validation

Zero-behaviour-change gate before commit:
1. Build SimX and rtlsim for the RTU config; run the **entire**
   `tests/raytracing` suite on both (the RTL-supported subset for rtlsim per the
   suite's own exclusions) — all must pass, identical results to pre-P1.0.
2. Build a **TEX-only** (no-RTU) config to prove the window now exists without
   the RTU (`SETW`/`GETW` decode + `VX_gfx_window` elaborate; RT FSM compiled
   out). No functional test yet — the first consumer lands in P1.
3. RTL parity (xrt) on the RTU smoke set per the RTL-coverage rule.

Commit P1.0 only when 1–3 are green.

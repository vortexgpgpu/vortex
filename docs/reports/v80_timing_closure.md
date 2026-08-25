# V80: the shipped bitstream does not meet timing

**Status:** root cause identified, fix requires resynthesis
**Applies to:** `build/hw/syn/xilinx/aved/hbm1_aved_hw` (built 2026-08-19 03:36)

---

## Summary

The V80 bitstream in use was implemented with **unmet timing constraints**. Any
result it produces is untrustworthy for workloads that exercise the failing
paths, and that is the root cause of the wrong-answer failures seen in the
hardware regression sweep.

```
WNS(ns)   TNS(ns)   TNS Failing Endpoints   TNS Total Endpoints
-0.241    -3.404    31                      157058

Timing constraints are not met.
```

Source: `bin/vortex_afu.vbin.prj/slash_rm/slash_vortex_afu.runs/impl_1/
route_report_timing_summary_0.rpt` (post-route).

---

## Why this explains the failures

The hardware sweep produced 20 passes and 3 genuine wrong-answer failures
(`demo`, `stencil3d`, `dotproduct`). All three pass in **every** simulator:

| Target | `demo` | What it proves |
|---|---|---|
| SimX (C model) | PASS | algorithm and driver are right |
| avedsim (RTL, Verilator) | PASS | RTL logic is right |
| xrtsim (same AFU RTL) | PASS | AFU integration is right |
| **V80 silicon** | **FAIL** | physical implementation diverges |

The driver was eliminated independently: the CP command stream captured with
`VORTEX_CP_TRACE=1` is byte-identical between avedsim and hardware — same 35
commands, same opcodes, same DCR addresses and values. Device memory was
eliminated too: `VORTEX_MEM_SELFTEST=1` round-trips every page the failing
tests use (`0x10000`–`0x14000`, `0x1000000`, `0x10000000`, `0x80000000`). A
forced clean rebuild of the kernel reproduced the failure byte-for-byte.

Correct logic plus failing timing is exactly this signature: deterministic,
workload-dependent wrong answers on silicon only.

`dotproduct` is a separate, expected matter — it uses atomics, and
`VX_CFG_EXT_A_ENABLED=0` in this build.

---

## The failing paths

Grouped by destination:

| Endpoint | Count |
|---|---:|
| `execute/alu_unit/.../muldiv_unit/.../multiplier/pipe_reg/.../DSP58C<0>_INST` | 10 |
| `cp_core/g_cpe[*].u_engine/seqnum_r_reg[*]/R` | 8 |
| `cp_core/u_completion/cur_ent_reg[addr][*]` | 3 |
| `VX_fpu_unit/VX_fpu_dsp/VX_fdiv_unit/xil_fdiv` | 1 |
| `hbm_sc_01` CDC FIFOs, QDMA (`cpm5`), `clk_wizard`, `smbus` | rest |

Two groups matter for correctness:

**1. The integer multiplier's DSP58 pipeline.** `demo` computes
`gid = gy * dim_x + gx` and `offset = gid * count` — multiplies on the address
path. A wrong product yields wrong addresses, so the kernel reads and writes
the wrong elements. `vecadd` and `sgemm` pass because their inner loops do not
depend on the same multiply paths.

**2. Reset distribution to the CP's seqnum registers.** The violated paths are
reset pins (`/R`) driven from `afu_wrap/vx_reset_shift_r_reg[7]_replica`. A
64-bit seqnum whose reset arrives late can come out of reset with stale upper
bits. This is worth noting alongside the separate finding that the CP's
counters are never cleared by any software mechanism (`q_reset_pulse` is
decoded and discarded in `VX_cp_core`), which the runtime now works around by
adopting `Q_SEQNUM` at open.

The remaining violations are in the static region (QDMA, HBM soft controller,
clock wizard) and are outside the AFU's reconfigurable partition.

---

## Recommended fix

In order of cost:

1. **Lower the kernel clock and rebuild.** `hw/syn/xilinx/aved/platforms.mk`
   sets `KERNEL_FREQ ?= 200`. WNS is −0.241 ns against a 5 ns period, so about
   5% of slack is missing; 180 MHz should close it with margin. This is the
   cheapest path to trustworthy results and should be done before any further
   hardware correctness work.

2. **Fix the reset fanout.** The seqnum reset paths fail from a single
   replicated shift-register stage into wide 64-bit registers. More replication
   on `vx_reset_shift_r`, or a properly pipelined reset tree, removes that
   group independently of clock speed.

3. **Pipeline the multiplier path.** If the DSP58 group persists at a lower
   clock, `VX_CFG_*_LATENCY` knobs or an added pipeline stage in the muldiv
   unit are the levers.

**Do not trust hardware correctness results from the current bitstream.** The
20 passing tests likely are genuinely passing, but "passes" on a design with
31 failing endpoints is not evidence of correctness — only of not having
exercised a broken path.

---

## How to check this on any future bitstream

```bash
grep -A8 "Design Timing Summary" \
  build/hw/syn/xilinx/aved/<build>/bin/vortex_afu.vbin.prj/slash_rm/\
slash_vortex_afu.runs/impl_1/route_report_timing_summary_0.rpt
```

Worth making a build-time gate: a bitstream that does not close timing should
fail the build loudly rather than be packaged and debugged as a software
problem, which is what happened here.

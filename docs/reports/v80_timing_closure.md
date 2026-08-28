# V80: timing closure, and what it does *not* explain

**Status:** timing violations characterised and attributed; they are **not** the
cause of the wrong-answer failures. Root cause of those is still open.
**Applies to:** `hbm1_aved_hw` (200 MHz, 2026-08-19) and `hbm1f180_aved_hw`
(180 MHz, 2026-08-24)

---

## Summary

The V80 design does not meet timing:

```
WNS(ns)   TNS(ns)   TNS Failing Endpoints   TNS Total Endpoints
-0.241    -3.404    31                      157058

Timing constraints are not met.
```

**Every violated path is inside AMD's static shell — none are in the Vortex
AFU.** All ten violated paths in the detailed report are in the HBM soft
controller (`top_i/slash/hbm_sc_01/...`), on the `clk_wizard_0_clk_out1_1`
clock group whose requirement is **2.778 ns (360 MHz)**. That clock has no
relationship to `KERNEL_FREQ`.

This was confirmed empirically. A full rebuild at `KERNEL_FREQ=180` produced
**bit-identical timing numbers** — WNS −0.241, TNS −3.404, 31 endpoints — to
three decimal places. Lowering the kernel clock cannot move paths that are not
on the kernel clock.

## Correction to an earlier claim

An earlier version of this report attributed the violations to the Vortex
integer multiplier's DSP58 pipeline and to reset distribution into the CP's
seqnum registers, and recommended lowering `KERNEL_FREQ`.

**That was wrong.** Those endpoint names were collected by grepping
`Destination:` across the whole report, which lists passing paths as well as
violated ones. Filtering to blocks that actually begin `Slack (VIOLATED)` gives
ten paths, all in `hbm_sc_01`. The 180 MHz rebuild is the independent check: if
the violations had been in AFU logic on the kernel clock, an 11% longer period
would have changed the numbers. It changed nothing.

---

## What this means

* The violations are **pre-existing and vendor-owned**. They are properties of
  the AMD compute shell plus the HBM soft controller configuration, and would
  appear in any design linked against this shell.
* They are **not fixable by changing the AFU or its clock**, and there is no
  evidence they are caused by anything in this repository.
* They are **not a demonstrated explanation** for the wrong-answer failures.
  They sit in the memory path, so data corruption is conceivable, but 20 of 23
  attributable tests pass on the same hardware — including memory-heavy ones
  (`sgemm`, `vecadd` at 1 M elements, `conv3`, `jacobi`). A broken HBM data
  path would not be that selective.

**The root cause of the `demo` and `stencil3d` wrong results is still open.**

---

## What is actually known about those failures

| Target | `demo` | Conclusion |
|---|---|---|
| SimX (C model) | PASS | algorithm and driver correct |
| avedsim (RTL, Verilator) | PASS | RTL logic correct |
| xrtsim (same AFU RTL) | PASS | AFU integration correct |
| **V80 silicon** | **FAIL** | divergence is physical, not logical |

Eliminated by measurement, not assumption:

* **Driver** — the CP command stream captured with `VORTEX_CP_TRACE=1` is
  byte-identical between avedsim and hardware: 35 commands, same opcodes, same
  DCR addresses and values.
* **Device memory** — `VORTEX_MEM_SELFTEST=1` round-trips every page the
  failing tests use (`0x10000`–`0x14000`, `0x1000000`, `0x10000000`,
  `0x80000000`).
* **Stale artifacts** — a forced clean rebuild of `demo` reproduced the failure
  byte-for-byte, same values.
* **Launch geometry** — fails identically at every block shape (`-x16`, `-x8`,
  `-x4 -y4`) and every size, with error count always 16·N.
* **2D indexing** — falsified: `sgemm` and `conv3` are `ndim=2` and use `.y`,
  and both pass.

### The failure, characterised exactly

`demo` seeds with `std::srand(50)`, so its data is reproducible. Regenerating
the exact sequence and comparing against the values the hardware produced:

```
demo actual[0] = 1090072283 = src1[0]
demo actual[1] =  462713693 = src1[1]
demo actual[2] =  410880882 = src1[2]
demo actual[3] = 1176772723 = src1[3]
demo actual[4] =   81804573 = src1[4]
```

Five consecutive exact 32-bit matches. The kernel computes
`dst = src0 + src1` and the hardware produces **`dst == src1`**: the `src0`
term contributes exactly zero. Indices, `src1_ptr` and `dst_ptr` are all
correct — the stores land in the right places with the right `src1` values — so
only the `src0` operand is lost.

Further narrowing:

* **Type-independent.** Rebuilt with `CONFIGS="-DTYPE=float"`: fails
  identically. So this is not the integer ALU; it is the address/load path.
* **Shape- and size-independent.** Same failure at `-x16`, `-x8`, `-x4 -y4`
  and at every `-n`, with the error count always 16·N.
* `src0` is at device address `0x10000`, and **`vecadd` also places `src0` at
  `0x10000` and passes** — so the address itself is not the trigger.

~~The remaining candidates are that `arg->src0_addr` is read as 0 from the
kernel args block, or that loads from that buffer return 0.~~

**RESOLVED 2026-08-27 — the kernel was innocent all along.** A post-mortem
probe re-allocated demo's buffers after a failing run and read the device
memory back directly: `src0` and `src1` uploads intact, and the `dst` region
holds **all 16384 correct sums**. The failure is in the *readback*: the staged
device→host path returned the staging area's previous contents. The old
`dst == src1` signature above is the download reusing `src1`'s freed staging
slot — the "matches" were the stale slot, not the kernel's stores. Two ordering
defects compound: `Q_SEQNUM` is MMIO-visible before the payload writes are
readable in HBM, and the completion line that should fence that was the CP's
only partial-line AXI write, which parks between the AFU port and HBM for
over a second. Fixed in `VX_cp_completion` (full-line write) plus a
completion-fenced `staged_refresh` in the aved runtime; see the commit
`cp: write completions as full cachelines` for the full derivation.

One observed difference not yet ruled in or out: `vecadd`'s two source buffers
land in the *same* 4 KB page (`0x10000` and `0x10100`) while `demo`'s are in
different pages (`0x10000` and `0x11000`). `vecadd` at `-n1048576` uses
multi-page buffers and passes, which argues against it, but its addresses for
that run were not captured.

`dotproduct` is unrelated and expected: it uses atomics, and
`VX_CFG_EXT_A_ENABLED=0` in this build.

---

## The kernel clock: `KERNEL_FREQ` does not change the hardware clock

Measured on silicon, which is the only way this was settled. `sgemm -n1024` on
the `gfx2c` bitstream (built with `KERNEL_FREQ=150`) reported 1,117,115,605
cycles in 5,594 ms:

```
1,117,115,605 / 5.594 s = 199.7 MHz
```

The device runs at **~200 MHz**, not the 150 MHz requested.

What each layer actually does:

* `KERNEL_FREQ` substitutes correctly into `config.cfg` (`freqhz=150000000`).
* `slashkit` records it in `system_map.xml` as `<ClockFrequency>150000000<`,
  and `emit/metadata/timing_freq.py` will cap that value to what WNS allows.
* **Nothing programs the hardware from it.** The compute shell's block design
  hardcodes the kernel clock:
  `set user_clk [create_bd_port -dir I -type clk -freq_hz 200000000 user_clk]`
  (`slashkit/resources/base/compute/scripts/slash_base.tcl:1149`), driven from
  `clk_wizard_0_clk_out1`.

So every V80 bitstream built here runs at 200 MHz regardless of the request,
and `system_map.xml` reports a frequency the device is not using.

Consequence for `gfx2c`: WNS is −0.261 ns against the 5.000 ns synthesis base,
so the design is operating roughly 5% beyond timing closure. It passes
`sgemm -n1024`, but that is not margin to rely on — the violated paths are all
in the TCU (`wgmma/tbuf/bbuf` DSP58 and the TCU core FEDP pipeline).

Two independent things could be fixed: make the shell's `user_clk` frequency
follow the `[clock] freqhz` request (the fork already modifies
`slash_base.tcl`), or pipeline the TCU — `VX_CFG_TCU_LATENCY` is the lever.

## Recommended next steps

1. **`KERNEL_FREQ` cannot change the operating clock** — the shell hardcodes
   it at 200 MHz. Fixing that means changing `slash_base.tcl` in the fork, not
   the Vortex build flow.
2. **Raise the shell timing violations with AMD** or check whether a newer
   compute shell / HBM soft-controller configuration closes. Ours to report,
   not to fix.
3. **Chase `demo` with the SimX-as-oracle method** (`docs/debugging.md`): add
   matching per-instruction trace dumps to SimX and to the hardware run, and
   diff. The first divergence localises it. This is the documented approach for
   exactly this situation and has not been tried yet.
4. **Gate the build on timing closure anyway.** A design that does not close
   should fail loudly rather than be packaged silently — even when the
   violations turn out to be vendor-owned, the build should say so rather than
   leaving it to be discovered during a correctness investigation.

## Checking any future bitstream

```bash
R=build/hw/syn/xilinx/aved/<build>/bin/vortex_afu.vbin.prj/slash_rm/\
slash_vortex_afu.runs/impl_1/route_report_timing_summary_0.rpt
grep -A6 "Design Timing Summary" "$R"
```

To attribute violations rather than merely count them, filter to blocks
beginning `Slack (VIOLATED)` and inspect their `Source:`/`Destination:` — do
not grep `Destination:` across the whole file, which also lists passing paths.
That mistake is what produced the incorrect first version of this report.

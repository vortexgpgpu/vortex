# TCU_OP vs baseline Vortex TCU — runtime and synthesis evaluation

> **STATUS: the runtime half of this report is INVALID. Do not cite sections 2, 4 or 5.**
>
> Cases A and B were measured with `VX_CFG_NUM_{ALU,LSU,FPU}_BLOCKS = 1` while
> `VX_CFG_NUM_TCU_BLOCKS` scaled with `VX_CFG_ISSUE_WIDTH`, so at IW=4 four TCU
> blocks were fed by single-ported ALU/LSU/FPU. Upstream fixed this in
> `96eca9c8f` and measured `sgemm_tcu` at this exact shape going
> **97,509 -> 68,598 cycles (-29.6%)**; the reference run on `wgdxa2k` confirms
> 68,598 cycles / 18,944 instructions against the 99,927 / 21,999 recorded here.
>
> The bias was one-sided: C (TCU_OP) runs one warp with 513 loads and barely
> touches those units, so it was unaffected while both baselines lost ~30%.
> Correcting A alone (~68,600 cycles at 251.1 MHz = ~273 us against C's 78,952 at
> 203.1 MHz = 388.7 us) makes the baseline roughly **42% faster** in wall-clock,
> reversing section 1's conclusion.
>
> Two discrepancies remain unexplained and must be resolved before re-issuing:
> cherry-picking `96eca9c8f` onto this tree changed the elaborated block counts
> from 1 to 4 but left every cycle count bit-identical, and this tree's A retires
> 21,999 instructions against upstream's 18,944, so it is not running quite the
> same kernel. The synthesis half (section 3) is unaffected -- it is IW=1 and
> does not involve these units.

**Config:** MNK = 128×128×128, fp16 in / fp32 out, NT=32, NW=16.
Runtime runs at IW=4 (rtlsim, `--perf=1`). Synthesis at IW=1 — see *Caveats*.
Device xcu55c-fsvh2892-2L-e, 250 MHz target (4.0 ns), OPT_LEVEL=3, Vivado 2025.2.
Date: 2026-09-19. Raw data: `eval/ev1_results.txt`, `build/hw/syn/xilinx/dut/ev_{tcuop,wgmma}_tcu/`.

---

## 1. Headline

TCU_OP executes the GEMM in **31.6% fewer cycles** than TCU+WGMMA on **14% fewer
LUTs** and **an eighth of the DSPs** — but it does not close the 250 MHz gate,
and the baseline does. At each design's own achievable frequency the wall-clock
gap narrows to **15.5%** against WGMMA and to **2.4%** against plain
`sgemm_tcu`. The cycle-count advantage is real — TCU_OP runs at 83% of the
32-MAC/cycle roofline against 66% and 57% for the baselines — but most of it is
currently spent paying for a longer clock period.

One figure that appeared in an earlier draft of this report should be
disregarded: the 24× instruction-count reduction is **not** evidence of
architectural efficiency. It is mostly occupancy (TCU_OP runs 1 warp, the
baselines 10–16) and data movement that bypasses the instruction stream. See
§2.1.

---

## 2. Runtime (rtlsim, 128³, IW=4)

| | app | engine | cycles | instrs | result |
|---|---|---|---:|---:|---|
| **A** | `sgemm_tcu` | baseline TCU | 99,927 | 21,999 | PASSED |
| **B** | `sgemm_tcu_wg_dxa` | TCU + WGMMA + DXA | 115,449 | 20,816 | PASSED |
| **C** | `sgemm_tcu_op` | TCU_OP + DXA | **78,952** | **876** | PASSED |

Relative to C: A is +26.6% cycles, B is +46.2% cycles.

Config was verified per run by reading the generated `*_ENABLED=` values rather
than grepping the define list (the config header names every option, including
the disabled ones, so a plain grep reports false positives):

| | `TCU_OP` | `WGMMA_ENABLED` | `DXA_ENABLED` | IW | NT | NW | `TCU_TYPE` |
|---|---|---|---|---|---|---|---|
| A | no | 0 | 0 | 4 | 32 | 16 | TFR |
| B | no | 1 | 1 | 4 | 32 | 16 | TFR |
| C | yes | 0 | 1 | 4 | 32 | 16 | TFR |

### 2.1 The instruction count: what the 24× actually is

**876 instructions versus ~21,000 looks like a 24× architectural win. It is
mostly not.** Three separate effects are folded into that ratio, and only the
smallest of them is the outer-product architecture:

| | occupancy | loads | stores | tcu share of mix | IPC |
|---|---:|---:|---:|---:|---:|
| A `sgemm_tcu` | 10.2 warps (64%) | 141,377 | 18,692 | 37% | 0.220 |
| B `sgemm_tcu_wg_dxa` | 15.7 warps (98%) | 71,680 | 18,432 | 39% | 0.180 |
| C `sgemm_tcu_op` | **1.0 warp (6%)** | **513** | **68** | 4% | 0.011 |

1. **TCU_OP runs on one warp; the baselines run 10–16.** Each baseline warp
   executes its own copy of the loop, so their instruction totals are multiplied
   by occupancy. This alone accounts for roughly 10–16× of the 24×, and it is an
   artefact of kernel structure, not of engine efficiency.
2. **Data movement is off the instruction stream.** C issues 513 loads where A
   issues 141,377. A 128³ fp16 GEMM must move ≳160 KB; at 128 B per warp-wide
   load that is ≥1,280 loads, so **513 is below the floor for moving the data at
   all**. TCU_OP's operands arrive via DXA and the TCU's own memory port, which
   are not counted as LSU loads. The traffic and its cost are still there — the
   full memory hierarchy is simulated in rtlsim — they are simply not attributed
   to instructions.
3. **One `tcu_op` instruction covers a whole tile.** This is the genuine
   architectural effect, and it is the *smallest* of the three: TCU instructions
   are 4% of C's mix, roughly 35 instructions.

Two corollaries follow, and one earlier claim does not survive:

- The near-zero stall percentages and IPC of 0.011 are what a single warp parked
  on a long-running instruction looks like. They are not evidence of an
  unstalled machine.
- **Running at 1.0 of 16 warp slots is a headroom question, not a win.** Whether
  that is a limit of the engine or a property of this kernel's structure is
  unresolved and worth establishing, because if more warps can be kept in flight
  there may be real performance unclaimed.
- An earlier draft claimed TCU_OP is therefore less sensitive to issue width.
  That was an inference, not a measurement, and at occupancy 1.0 the design is
  barely exercising issue width in either direction. Withdrawn.

None of this affects the cycle counts, which are independently sanity-checked
against the roofline in §4.1.

---

## 3. Synthesis

Both runs are the standalone TCU unit (`hw/unittest/tcu_unit/VX_tcu_unit_top.sv`)
through `hw/syn/xilinx/dut`, at a 4.0 ns target.

| | TCU_OP (`ev_tcuop`) | TCU+WGMMA (`ev_wgmma`) | Δ |
|---|---:|---:|---:|
| **Routed WNS** | **−0.923 ns** | **+0.018 ns** | −0.941 |
| Routed TNS | −2,366.79 | 0.000 | — |
| Failing endpoints | 9,159 / 148,406 | **0** / 83,553 | — |
| **Achievable Fmax** | **203.1 MHz** | **251.1 MHz** | −19.1% |
| Total LUTs | **105,898** (8.12%) | 123,391 (9.46%) | **−14.2%** |
| Logic LUTs | 100,384 | 122,755 | −18.2% |
| LUTRAM | 5,504 | 636 | +8.7× |
| SRL | 10 | 0 | — |
| FFs | 45,084 | 43,458 | +3.7% |
| BRAM / URAM | 0 / 0 | 0 / 0 | — |
| **DSPs** | **32** (0.35%) | 256 (2.84%) | **−87.5%** |

### 3.1 The baseline meets the gate; TCU_OP does not

TCU+WGMMA closes 250 MHz with 0.018 ns to spare and **zero** failing endpoints.
TCU_OP is 0.923 ns short, i.e. 203 MHz. This is the single blocking result of the
evaluation: the requirement is 250 MHz, and only the baseline satisfies it.

TCU_OP's timing has improved substantially over the course of the redesign, and
the trend is good:

| stage | routed WNS | TNS | failing endpoints |
|---|---:|---:|---:|
| legacy TCU_OP datapath | −2.317 | — | 57,455 |
| D3 only (TFR reuse) | −5.346 | — | — |
| D1+D2+D3 | −2.001 | −34,219 | 51,097 |
| + back-pressure/align fixes | −1.228 | −7,276 | 22,636 |
| **+ D4 (decomposed crossbar)** | **−0.923** | **−2,367** | **9,159** |

D4 alone bought 0.305 ns of WNS, cut TNS by 67% and failing endpoints by 60%.
The remaining 0.923 ns is a *plateau*, not a single path: roughly five
independent groups sit between −0.93 and −1.05 ns (the flush stages `pipe_f1..f7`,
`pipe_align1`, `req_buf`, and the `a_bitmap → sorter → operand-word` path).
Each needs its own fix; none is a structural defect of the kind D1–D4 addressed.

### 3.2 Area: what is a real saving and what is scope reduction

The **8× DSP reduction is not entirely a free efficiency win, and should not be
quoted as one.** Two distinct effects are mixed in it:

1. *Genuine architectural saving.* TCU_OP multiplies at native significand width
   (one `VX_tcu_tfr_wmul` per lane, 11-bit padded significands serving
   fp16/bf16/fp8/bf8 from one multiplier) instead of instantiating a full
   X→fp32 multiplier per lane. This is the industry-standard arrangement
   (multiply exactly at native width, align once, sum in fixed point, round once).
2. *Scope reduction.* The D3 redesign dropped fp32×fp32 multiply support, which
   was ~75% of the old FEOP LUTs and two thirds of its DSPs, and which no test in
   the suite exercises. If fp32 input support is required later, some of this
   returns.

The LUTRAM increase (636 → 5,504) is the other side of the same trade and is
expected: TCU_OP holds its 32-bank × 32-slot accumulator and the crossbar input
queues in LUTRAM, where WGMMA keeps its accumulator in registers and logic. On
this device LUTRAM is 0.92% of the budget, so it is a favourable exchange for
18% of the logic LUTs.

---

## 4. Wall-clock: the comparison that matters

Cycles alone flatter TCU_OP, because it runs at a lower clock. At each design's
own measured Fmax:

| | cycles | Fmax | time | vs C |
|---|---:|---:|---:|---:|
| A `sgemm_tcu` | 99,927 | 251.1 MHz\* | 397.9 µs | +2.4% |
| B `sgemm_tcu_wg_dxa` | 115,449 | 251.1 MHz | 459.7 µs | +18.3% |
| **C `sgemm_tcu_op`** | **78,952** | **203.1 MHz** | **388.7 µs** | — |

\* A has no synthesis run of its own (only two were requested). The baseline
TCU+WGMMA frequency is used as a **proxy**, which is conservative in A's favour
— a TCU-only build without the WGMMA path would very likely be smaller and no
slower, so A's true time is probably *at or below* 397.9 µs. Treat A's wall-clock
as an upper bound on its speed advantage, not a measurement.

### 4.1 Roofline sanity check

The cycle counts are credible on their own terms, independent of §2.1. At
32 MACs/cycle, 128³ (2,097,152 MACs) has a floor of 65,536 cycles:

| | cycles | % of 32-MAC/cycle peak |
|---|---:|---:|
| A `sgemm_tcu` | 99,927 | 65.6% |
| B `sgemm_tcu_wg_dxa` | 115,449 | 56.8% |
| **C `sgemm_tcu_op`** | **78,952** | **83.0%** |

All three land in a plausible band, and C sitting closest to the roofline is
consistent with an engine that keeps its MACs fed — which is the claim the
runtime half of this report actually rests on. The instruction-count ratio is
not needed to support it, and should not be quoted in its place.

**The break-even point for TCU_OP against WGMMA is 171.6 MHz**; against plain
`sgemm_tcu` at proxy frequency it is **198.4 MHz**. TCU_OP is at 203.1 MHz, so
it is currently only just ahead of `sgemm_tcu` and would fall behind it if timing
regressed by ~0.1 ns. Closing the plateau to 250 MHz would put TCU_OP at
315.8 µs — a 31% lead over WGMMA and 21% over `sgemm_tcu` — which is where the
architecture's cycle advantage actually pays out.

---

## 5. Analysis

**The outer-product engine is the right architecture on cycles and area.** 26–46%
fewer cycles at 83% of the 32-MAC/cycle roofline, on 14% fewer LUTs and 8×
fewer DSPs, is a coherent result: temporal reduction through a RAM-backed
accumulator keeps the MACs fed, and multiplying at native significand width
removes the DSP bulk. Nothing in this data set contradicts that. (The 24×
instruction-count ratio is *not* part of this case — see §2.1; it is mostly
occupancy and off-stream data movement.)

**Frequency is the entire problem.** TCU_OP is a 203 MHz design competing with a
251 MHz baseline, and that alone converts a 31.6% cycle lead into 15.5% of
wall-clock, and a 26.6% lead over `sgemm_tcu` into 2.4%. Every remaining unit of
engineering value is in the 0.923 ns, not in further cycle reduction.

**Why TCU_OP is harder to time-close than WGMMA.** The baseline distributes its
arithmetic across many small DSP-backed lanes with short local nets — 0 failing
endpoints out of 83,553 says the placer had no difficulty. TCU_OP concentrates
104-bit carry-save state in 32 banks behind a 32-way routing network, so it has
148,406 endpoints (1.8× more) and its critical paths are *route*-dominated: 66–81%
of the delay on the worst paths is net, not logic. That is a floorplanning and
locality problem as much as a logic-depth one, which is precisely why D4's
decomposition helped (it halved the span of the all-to-all region) and why the
next tranche may need pblocking alongside RTL changes.

**Confidence.** The runtime numbers are trustworthy: all three configs were
verified by their generated `_ENABLED=` values, all three PASSED, and the TCU_OP
datapath separately passes 5/5 of the functional suite including both sparse
modes. The synthesis numbers are single runs, so ±0.05 ns of placer noise should
be assumed on WNS; WGMMA's +0.018 ns margin is thin enough that it should be
re-run before being treated as a guaranteed pass.

---

## 6. Caveats

1. **Synthesis is IW=1, runtime is IW=4.** `VX_tcu_unit_top.sv:109` hard-asserts
   `VX_CFG_ISSUE_WIDTH == 1`; the flow synthesises the TCU unit standalone, where
   issue width is out of scope. Runtime used IW=4 as specified. The two halves of
   this report are therefore not the same elaboration.
2. **Baseline uses `TCU_TYPE=TFR`, not `BHF`.** Chosen so the comparison isolates
   engine architecture rather than confounding it with different multiplier and
   rounding IP, since TCU_OP reuses the TFR datapath. The pre-existing
   `tcu250_base` run (BHF, no WGMMA) measured −2.317 ns / 140,976 LUT / 96 DSP and
   is *not* directly comparable.
3. **No standalone synthesis for A (`sgemm_tcu`).** Only two synthesis runs were
   requested. A's wall-clock uses WGMMA's frequency as a proxy — see §4.
4. **A has no DXA; B and C do.** This is inherent to the applications, not a
   configuration choice, but it means A carries its data movement in ordinary
   instructions and is disadvantaged on that axis.
5. **C's simulation wall-time is inflated** by SIMULATION-only debug
   instrumentation (crossbar routing assertions, deferral counters) left in place
   from the D4 investigation. Cycle counts and all synthesis results are
   unaffected — none of it survives `ifdef SIMULATION`.
6. **Single synthesis runs, no seed sweep.** WNS should be treated as ±0.05 ns.

---

## 7. Recommendation

Do not adopt or reject TCU_OP on this data — **close the timing plateau first,
then re-evaluate.** The architecture has demonstrated the cycle and area
advantages it was designed for; it has not yet demonstrated that it can be built
at the required frequency. The specific next steps, in order of measured
leverage:

1. **Flush pipeline `pipe_f1..f7`** — 5,648 failing endpoints across seven stages,
   worst −0.931 ns. This is the largest remaining cluster and it is a rebalancing
   of stages added during D3, so it is the most tractable.
2. **`a_bitmap → 32-way sorter → operand-word mux`** — 18 LUT levels in one cycle.
   A fix is already written (register the sorter output, hold issue for one cycle
   per set transition) but is unvalidated and not applied.
3. **`pipe_align1` and `req_buf`** — −1.044 and −1.027 ns.
4. **Re-census after each step.** Two thirds of the delay on these paths is
   routing, so the ranking shifts materially as placement changes; the current
   ordering was measured before D4 and should be re-derived.

If TCU_OP reaches 250 MHz it is decisively the better engine at this config. If
it stalls near 200 MHz it is roughly at parity with `sgemm_tcu` in wall-clock
while costing a redesign, and the case for it then rests on the DSP and LUT
saving rather than on speed.

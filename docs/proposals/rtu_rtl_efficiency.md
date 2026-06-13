# RTU RTL — FPGA Efficiency & FP IP-Reuse Refactor

**Scope:** a design-efficiency review of the RTU RTL ([hw/rtl/rtu/](../../hw/rtl/rtu/))
targeting the **Alveo U55C @ 300 MHz**, and a plan to (1) close the current
timing miss and (2) cut the LUT/FF footprint by rebalancing onto the idle hard
blocks (DSP → BlockRAM → LUTRAM, in that priority order). The centerpiece is a
**shared, capability-parameterized FP datapath library** in
[hw/rtl/libs/](../../hw/rtl/libs/) so the RTU and the ISA FPU instantiate **one**
FMA/compare/reciprocal core — full IEEE-754 for the FPU, a feature-reduced
configuration for the RTU — with no forked IP.
**Status:** Proposal — review + architecture; no code changed yet.
**Tree:** `~/dev/vortex_v3/prism_v3` (branch `prism`).
**Date:** 2026-06-12.
**Related:** [rtu_implementation.md](rtu_implementation.md),
[rtu_verilog_minimal_proposal.md](rtu_verilog_minimal_proposal.md),
[rtu_isa_v2_proposal.md](rtu_isa_v2_proposal.md).

---

## 1. Baseline — measured, not estimated

Numbers from the existing U55C DUT runs under
`build/hw/syn/xilinx/dut/rtu/` (post-implementation, `core_clock = 3.333 ns`):

| Build | Walker | LUT | FF | SRL | DSP | BRAM | LUTRAM | WNS |
|---|---|---|---|---|---|---|---|---|
| `build_w0` | flat tri-list | 44,854 (3.44%) | 26,955 | 3,588 | 74 (0.82%) | 0 | 0 | **−0.080** |
| `build_w4` | CW-BVH4 | 72,428 (5.56%) | 43,363 | 4,684 | 104 (1.15%) | 0 | 0 | **−0.245** |
| `build_w6` | CW-BVH6 | 72,809 (5.58%) | 43,598 | 4,692 | 104 (1.15%) | 0 | 0 | **−0.135** |
| `rtu_synth_v3` | CW-BVH4 | 72,077 (5.53%) | 42,731 | 4,684 | 104 (1.15%) | 0 | 0 | **−0.237** |

**Every variant fails 300 MHz**, and the hard blocks this project prioritizes are
nearly idle: **DSP 1.15%, BlockRAM 0%, LUTRAM 0%**, while LUT is the binding
resource. The design is *inverted for FPGA* — it spends fabric (LUT/FF) on work
that the abundant hard blocks were built to do.

### 1.1 Where the LUTs go (build_w4)

| Block | LUT | FF | DSP | Note |
|---|---|---|---|---|
| `tri_pe` | 43,275 (60%) | 18,791 | 74 | 37 FMA + 1 div + 9 cmp |
|  — 4× `fdot3` | 12,826 | | | normalizes 3× per dot |
|  — 2× `fcross3` | 12,168 | | | normalizes 2× per cross |
|  — `recip` (`fdivsqrt`) | 2,526 | | **0** | Newton-Raphson in pure LUTs |
| `box_pe` | 17,755 | 7,393 | 30 | 15 FMA |
| scheduler proper | ~3,660 | ~15K | 0 | FSM + per-ctx register files |
|  — 3× setup `recip` | 7,738 | | **0** | one `1/dir` divider **per axis** |
| `xform_pe` | (DCE'd here) | | | dead-code-eliminated when TLAS inactive; ~5 FP units when enabled |

Each [VX_fma_unit](../../hw/rtl/fpu/VX_fma_unit.sv) ≈ **1,000 LUT + 2 DSP**, and
there are **52** of them, all full IEEE-754 (subnormals, 5 rounding modes,
`fflags`, exception logic, per-op normalize+round). The four
[VX_fdivsqrt_unit](../../hw/rtl/fpu/VX_fdivsqrt_unit.sv) dividers total **10,264
LUT at 0 DSP** — 14% of the whole core spent on division, none of it on a DSP.

### 1.2 Critical path — re-baselined (current source)

The `build_w0/w4/w6` reports above are **stale**: they predate commit `2b4bc1e6`
(*"close 300 MHz on the BVH scheduler — precompute struct address in SELECT"*),
which already moved `scene_base + cur_off` into the SELECT phase via
`structaddr_q` ([VX_rtu_scheduler.sv](../../hw/rtl/rtu/VX_rtu_scheduler.sv) L183-187,
493). The earlier `curoff_q → box_pe` scheduler path **no longer exists**.

The current representative build is **`build_w4_pm`** (BVH4, **TLAS/`xform_pe`
active**, post-`structaddr_q`):

| Metric | Value |
|---|---|
| LUT | 98,035 (7.52%) |
| FF / SRL | 57,291 / 6,819 |
| DSP | 146 (1.62%) |
| BRAM / LUTRAM | **0 / 0** |
| WNS (routed / post-physopt) | **−0.135 / −0.096 ns** |

| Block | LUT | FF | DSP |
|---|---|---|---|
| `tri_pe` | 42,320 | 18,222 | 74 |
| **`xform_pe` (TLAS)** | **22,373** | 9,647 | 42 |
| `box_pe` | 17,671 | 7,485 | 30 |

**The bottleneck has moved into the FP core.** The 112+ violating endpoints are
dominated by the **`VX_fma_unit` ALIGN→ACC stage** (`tri_pe/fma_e2/pipe_aln →
pipe_acc`, 14 levels, CARRY8×7 — the ~56-bit FP accumulate carry chain), with a
sibling path in `xform_pe`'s dot-product FMAs. This is *not* a scheduler problem;
it is the IEEE-754 accumulate adder inside the shared FMA. The fix therefore sits
in the §3 library core (split/parameterize the ACC stage, or reduce mantissa
width), **not** in scheduler pipelining — which reorders the rollout (§6).

> The TLAS `xform_pe` is **not** DCE'd in the production config — it is a 22K-LUT
> / 42-DSP consumer in its own right, and on the critical path. §1.1's
> "DCE'd here" note applied only to the older non-TLAS DUT.

---

## 2. Root causes

1. **Full IEEE-754 FMA for geometry math.** Ray–box and ray–triangle
   intersection need none of subnormal gradual-underflow, RNE/RUP/RDN/RMM
   selectability, `fflags`, or NaN-payload propagation. ~50% of each FMA's core
   logic is dead weight for the RTU.
2. **`LATENCY_FMA = 16`** — inherited from the GPU FPU's Vivado/DSP-cascade
   target sized for **F64**. The RTU is **F32-only**. This inflates
   `tri_pe` to a **147-cycle** pipe (`8·F + V + 2`); every side-band signal is
   carried through `F`-scaled [VX_shift_register](../../hw/rtl/libs/VX_shift_register.sv)
   delay lines → the 43K FF / 4.7K SRL.
3. **Per-op normalize+round in `fdot3`/`fcross3`.** A 3-term dot product
   rounds/normalizes 3× instead of once; ~25K LUT of avoidable align+round.
4. **Four Newton-Raphson dividers built in LUTs**, three of which are redundant
   per-axis `1/dir` setups that could share one time-multiplexed unit.
5. **All per-context state in flip-flops + wide muxes.** `stack`
   (`NUM_CTX·16·32b`), `f_buf` node image (`NUM_CTX·1024b`), `inst_xform`, and
   the ray/hit/obj records are 1R1W register files indexed by context id — the
   textbook case for LUTRAM/BRAM, currently 100% fabric.
6. **Dead config knobs.** `RTU_BOX_PE` / `RTU_TRI_PE` (default 4) are declared in
   [VX_rtu_pkg.sv](../../hw/rtl/rtu/VX_rtu_pkg.sv) but **never referenced** — there
   is exactly one streaming `box_pe` and one `tri_pe`; the "parallel lanes"
   knobs do nothing.

---

## 3. Centerpiece — shared FP datapath library (IP reuse)

### 3.1 Where reuse stands today

`VX_fma_unit`, `VX_fncp_unit`, and `VX_fdivsqrt_unit` are **already shared**: the
ISA FPU instantiates them via [VX_fpu_dsp.sv](../../hw/rtl/fpu/VX_fpu_dsp.sv) /
[VX_fpu_std.sv](../../hw/rtl/fpu/VX_fpu_std.sv), and the RTU PEs instantiate them
directly. So the *units* are reused — but they live in `fpu/`, are imported with
`VX_fpu_pkg`, and expose only `LATENCY` / `MAN_BITS` / `EXP_BITS` / `USE_DSP`.
There is **no way to ask for a cheaper core**, so the RTU pays full IEEE-754 cost.

### 3.2 The move — capability-parameterized cores in `libs/`

Relocate the datapath cores into [hw/rtl/libs/](../../hw/rtl/libs/) (the home of
shared, dependency-light IP — `VX_mem_scheduler`, `VX_lzc`, `VX_csa_tree`,
`VX_divider`, …) as **one core per function**, gated by *capability* parameters:

```
libs/VX_fp_mac.sv      // fused a*b±c
    parameter LATENCY, MAN_BITS, EXP_BITS,
    parameter EN_SUBNORMAL = 1,   // 0: flush-to-zero (FTZ) — drops LZC/align corner logic
    parameter RMODE_SET    = ALL, // ALL | RNE_ONLY | RTZ_ONLY — prunes the rounding mux
    parameter EN_FFLAGS    = 1,   // 0: drop NV/OF/UF/NX accumulation + ports
    parameter EN_EXCEPT    = 1,   // 0: assume finite operands — drops NaN/inf classify+mux
    parameter USE_DSP

libs/VX_fp_cmp.sv      // min / max / compare / sign-inject (from VX_fncp_unit)
libs/VX_fp_recip.sv    // 1/x (and rsqrt) — selectable LUT-NR | DSP+seed-BRAM backend
```

- **ISA FPU** keeps a thin `fpu/` wrapper pinning the full-IEEE configuration
  (`EN_SUBNORMAL=1, RMODE_SET=ALL, EN_FFLAGS=1, EN_EXCEPT=1`) — **bit-identical**
  to today, zero behaviour change, RV-compliant.
- **RTU PEs** instantiate the **same module** with
  `EN_SUBNORMAL=0, RMODE_SET=RTZ_ONLY, EN_FFLAGS=0, EN_EXCEPT=0` and a short
  `LATENCY`. The `box_pe`/`tri_pe`/`fdot3`/`fcross3` code is otherwise unchanged
  — they already drive the same ports.

This is the maximal-reuse form of review §Opt 2: **one source of truth**, two
build-time configurations, no RTU-only arithmetic fork to maintain.

> **Dependency note:** `libs/` modules avoid GPU-package coupling. The cores
> import a handful of `VX_fpu_pkg` enums (`INST_FPU_*`, `INST_FRM_*`,
> `fclass_t`). Either (a) keep those as `localparam`/struct inside the lib core
> and pass the op as a small local enum, or (b) split a minimal `VX_fp_pkg` that
> both `libs/` and `fpu/` import. (a) is preferred — it keeps `libs/` standalone.

### 3.3 Reciprocal backend selection (`VX_fp_recip`)

The reciprocal is the cleanest DSP/BRAM rebalancing target — today it is
2,526 LUT × 4 at **zero** DSP. `VX_fp_recip` exposes a `BACKEND` parameter:

- `LUT_NR` — today's Newton-Raphson (portable default).
- `DSP_SEED` — a seed-table reciprocal: **1 BRAM18** (the seed LUT) + ~3 DSP for
  1–2 NR refinement steps. Trades ~2,000 LUT → hard blocks per unit, directly
  consuming the idle BRAM/DSP budget per the project's resource priority.

Combined with **time-multiplexing the three `1/dir` setup dividers onto one**
unit (review §Opt 4), the four dividers collapse toward one shared
`VX_fp_recip`.

---

## 4. `NUM_CTX` scaling — the dominant axis (confirmed)

`NUM_CTX` is a clean parameter, plumbed
`VX_rtu_core.NUM_LANES → VX_rtu_scheduler.NUM_CTX` with no hardcoding, but it is
**bound** to thread count: `NUM_CTX = NUM_LANES = `VX_CFG_NUM_THREADS`` (=4 in
these builds). **The production GPU config raises `NUM_THREADS`/`NUM_CTX` above
4** — which reorders this whole proposal, because the two cost centers scale
differently:

| Cost center | Scaling | At `NUM_CTX=4` | At higher `NUM_CTX` |
|---|---|---|---|
| `box_pe` + `tri_pe` (shared streaming datapath) | **fixed** (one instance, time-shared across contexts) | ~61K LUT (83%) | unchanged |
| Per-context state (`stack`, `f_buf`, `inst_xform`, ray/hit/obj) + select muxes | **O(NUM_CTX)** | ~13K FF + mux | **dominates** |

The FP datapath does **not** replicate with contexts — it streams one
ray-context per cycle. So every doubling of `NUM_CTX` roughly doubles the
flip-flop register files and *widens the context-select muxes* (a super-linear
LUT effect), while the PE cost stays flat. At the production context count the
**per-context state, not the FP units, becomes the binding LUT/FF cost** — which
promotes review §Opt 3 (state → BRAM/LUTRAM) from a modest cleanup to the
**headline win**, and it is behaviour-preserving.

Two coupled actions:

1. **Add `VX_CFG_RTU_NUM_CTX`** (default = `NUM_THREADS`) to **decouple in-flight
   ray contexts from lane count** — fewer contexts → less area; more → deeper
   latency hiding over RTCache miss latency, *independently* of SIMD width.
2. **Back the per-context state with BRAM/LUTRAM** so it stays **flat in fabric**
   as `NUM_CTX` grows: a 1R1W RAM addressed by context id replaces the
   FF-array + `NUM_CTX`:1 mux. `stack` (`NUM_CTX·512b`) and `f_buf`
   (`NUM_CTX·1024b`) map naturally to BRAM18; the narrower records to LUTRAM.
   BRAM is at **0%** today, so the headroom is the entire device.

---

## 5. Optimization summary & estimated gains

Deltas vs the 72,428-LUT BVH4 baseline. Gains are engineering estimates pending
re-synthesis; **direction and hard-block tradeoffs are firm**.

Two columns of impact: **Δ@4** is the saving at today's `NUM_CTX=4`; **scaling**
is how the saving grows with context count — the deciding factor now that
production raises `NUM_CTX` (§4).

| # | Optimization | Δ LUT @4 | Δ FF / SRL @4 | Δ DSP | Δ BRAM/LUTRAM | Scaling w/ `NUM_CTX` | Fmax | Effort |
|---|---|---|---|---|---|---|---|---|
| 5 | **Pipeline scheduler addr-gen** + duplicate `curoff_q` | +0.3K | +~0.3K FF | 0 | 0 | flat | **closes −0.245 ns** | Low |
| 3 | **Per-ctx state → BRAM/LUTRAM** (`stack`,`f_buf`,`inst_xform`,ray/hit) + `VX_CFG_RTU_NUM_CTX` | −2.5K | −7K FF | 0 | +2 BRAM18 / ctx-batch | **grows ∝ `NUM_CTX`** (the headline win at production) | ↑ | Med |
| 1 | **Dedicated RTU FMA latency 16→6** (F32-only; shrinks pipes + delay lines) | −8K (−11%) | −10K FF / −2.5K SRL | 0 | 0 | flat (shared PE) | ↑ | Low |
| 4 | **Reciprocal: share 3 setups → 1** (+ optional `DSP_SEED` backend) | −5K…−7K | −1K FF | 0…+3 | 0…+1 BRAM | flat | = | Low–Med |
| 2 | **Capability-reduced FP core** (FTZ, single RMODE, no fflags/except) + **fuse `fdot3`/`fcross3`** normalize | −13K (−18%) | −4K FF | 0 | 0 | flat (shared PE) | ↑ | Med (precision validation) |
| 6 | **Retire dead `RTU_BOX_PE`/`TRI_PE` knobs** (or wire real parallelism) | — | — | — | — | — | — | Trivial |

**Combined projection at `NUM_CTX=4` (all):** ≈ **72.4K → ~44K LUT (−39%)**,
**43K → ~21K FF (−51%)**, **+2 BRAM18**, DSP roughly flat (~104, ≤1.2%), **300 MHz
closed**.

**At production `NUM_CTX` (>4):** Opts 1/2/4 hold their absolute savings (the PE
is shared), while Opt 3's savings **scale with context count** — without it, the
FF register files + select muxes grow super-linearly and re-break timing; with
it, per-context state stays **flat in fabric** on the idle BRAM. Opt 3 is
therefore the load-bearing change for the production config, not an afterthought.

---

## 6. Phasing

Ordering reflects the confirmed `NUM_CTX` scaling (§4): the timing fix first,
then the context-state move (load-bearing at production), then the shared-PE
wins.

1. **P0 — timing.** ~~Register the BVH offset/address arithmetic and de-fan
   `curoff_q`.~~ **Already done** in commit `2b4bc1e6` (`structaddr_q`). The
   residual **−0.096 ns** miss (post-physopt, `build_w4_pm`) is now the
   **`VX_fma_unit` ALIGN→ACC carry chain** (§1.2), shared by `box/tri/xform` PEs.
   The correct P0 is to **break that stage** in the FP core: either (a) add a
   pipeline register splitting ACC (add vs LZC), exposed as an `ACC_SPLIT`
   capability param on the §3 lib core (1 extra cycle; touches the shared unit,
   so the GPU FPU must stay bit-identical and re-close), or (b) reduce RTU
   mantissa width (shorter carry chain) — which folds into P4. **(a) is the
   minimal behaviour-preserving fix; recommend leading P2 (library extraction)
   with it so P0 and P2 are one step.** Validate `tests/raytracing` bit-identical.
2. **P1 — context state → BRAM/LUTRAM + `VX_CFG_RTU_NUM_CTX` (Opt 3, §4).**
   Convert `stack` / `f_buf` / `inst_xform` / ray-hit records from FF-arrays to
   context-id-addressed RAMs; add the decoupled context knob. **Behaviour-
   preserving** but the highest-leverage change at production `NUM_CTX`, and the
   registered RAM read reinforces the P0 critical path. Validate across the
   target `NUM_CTX` values, not just 4.
3. **P2 — library extraction (§3, Opt 1).** Move cores to `libs/` as
   `VX_fp_mac` / `VX_fp_cmp` / `VX_fp_recip` with capability params **all
   defaulting to the current behaviour**; rewire `fpu/` wrappers and RTU PEs to
   the lib. Set RTU `LATENCY` to 6–8. **Must stay bit-identical** at default
   params (FPU) and re-verified at the RTU latency change.
4. **P3 — reciprocal consolidation (Opt 4).** Share the 3 setup dividers; expose
   `DSP_SEED`.
5. **P4 — capability reduction (Opt 2).** Flip RTU FP params to FTZ / RTZ /
   no-fflags / no-except; fuse dot/cross normalization. **Behavioural** —
   gated on the SimX-oracle validation in §7.

Each phase is independently synthesizable and testable; P0–P3 are
behaviour-preserving, P4 requires the §7 gate. **Synthesize each phase at both
`NUM_CTX=4` and a production-representative count** so the scaling claims in §4/§5
are verified, not assumed.

---

## 7. Validation

- **Functional:** full [tests/raytracing](../../tests/raytracing/) suite via
  **xrt** (the RTL coverage path), plus rtlsim for fast iteration. P0–P2 must be
  **bit-identical**.
- **Precision (P3):** use the SimX model as the goal-reference oracle — build the
  FTZ/RTZ behaviour into SimX first, confirm image parity within agreed ULP/SSIM
  tolerance, then diff SimX↔RTL trace dumps to localize any divergence before
  committing the RTL precision change.
- **Timing/area:** re-run the `build/hw/syn/xilinx/dut/rtu` DUT (w0/w4/w6) after
  each phase; track WNS and the per-block LUT/FF/DSP/BRAM table from §1. Each
  synth/FPGA build uses a unique `PREFIX` (own build tree + log); never two hw
  runs on the U55C at once.

---

## 8. Risks & open questions

- **`libs/` package coupling (§3.2).** Preferred resolution: self-contain the op
  enums in the lib core so `libs/` stays GPU-package-free. Needs a small
  `VX_fpu_pkg` audit.
- **Precision (P3).** FTZ + RTZ shift intersection results in the ULPs;
  watertightness near shared edges is the thing to watch. The SimX gate (§7)
  exists to catch it before RTL.
- **`NUM_CTX` decoupling (§4).** Adds a config dimension to the test matrix; keep
  the default `= NUM_THREADS` so existing configs are unchanged.
- **DSP-priority tension.** `DSP_SEED` reciprocal *adds* DSP (the project's
  scarcest hard block). It is opt-in; at today's 1.15% DSP it is affordable, but
  the LUT_NR default keeps DSP flat for DSP-constrained full-GPU configs.
- **Resolved:** the production GPU config **does** raise `NUM_THREADS`/`NUM_CTX`
  above 4. Opt 3 (per-context state → BRAM/LUTRAM) is therefore promoted to P1
  (§6) and treated as load-bearing, not a cleanup (§4/§5). **Open follow-up:**
  the specific target `NUM_CTX` value(s) to synthesize against in the §6 phase
  gates.

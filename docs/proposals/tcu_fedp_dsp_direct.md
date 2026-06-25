# Direct-DSP FEDP for SPF16→FP32 Mixed-Precision Dot Product

Target device: Xilinx Alveo U55C (`xcu55c-fsvh2892-2L-e`), `core_clock` = 300 MHz (3.333 ns).
Scope: `hw/rtl/tcu/dsp/VX_tcu_fedp_dsp.sv`, with a cross-reference to
`hw/rtl/tcu/tfr/VX_tcu_fedp_tfr.sv` and the `VX_fma_unit::USE_DSP` precedent.

---

## 1. Summary

`VX_tcu_fedp_dsp` today is a thin *wrapper* around the Xilinx Floating-Point
Operator IP (`xil_fmul` / `xil_fadd`). It instantiates one **full FP32 multiplier
IP per lane** and a **full FP32 adder IP per node** of a binary reduction tree.
Each IP carries its own exponent handling, alignment, normalization and rounding,
so the per-lane FP machinery is replicated `TCK` (mul) + `TCK` (add) times. The
result is the worst of all three axes — area, DSP count, and latency — *and it
still misses 300 MHz*, while covering only a subset of the formats TFR supports.

This document proposes replacing that wrapper with a **fused** DSP dot product
that keeps a single shared align/accumulate/normalize/round datapath (the part
TFR already does well) and maps only the dense arithmetic — the small mantissa
multiplies and the wide fixed-point accumulate — onto `DSP48E2` slices.

A central finding falls out of the analysis: **the optimal "direct DSP FEDP" is
architecturally the same thing as "TFR with its multiply stage mapped to DSP."**
That makes the second question in this proposal — adding a
`VX_CFG_TCU_TFR_DSP_ENABLE` knob to TFR — not an independent feature but the
*recommended way to obtain the optimized DSP FEDP itself*, reusing TFR's shared
fused datapath and its full format coverage (fp8/fp4/mx/tf32/int) for free.

---

## 2. Current implementations reviewed

### 2.1 `VX_tcu_fedp_dsp` (the wrapper)

Datapath (per FEDP, `N = TCU_TC_K`, `TCK = 2N`, `LEVELS = clog2(TCK)`):

1. **Convert** (`g_cvt`): every 16-bit lane is widened to FP32 by combinational
   `fp16_to_fp32` / `bf16_to_fp32` blocks (LUT-only), then registered
   (`FCVT_LATENCY = 1`).
2. **Multiply** (`g_prod`): `TCK` instances of `xil_fmul` (FP32, `Full_Usage`,
   `FMUL_LATENCY = 8`).
3. **Reduce** (`g_red_tree`): a binary tree of `xil_fadd` (FP32,
   `FADD_LATENCY = 11`), depth `LEVELS`, i.e. `TCK-1` adders.
4. **Final accumulate**: one more `xil_fadd` adds the delayed `c_val`.
5. **Integer path** (parallel): a *hand-built* INT4/INT8 datapath
   (`g_int_mul`): per-element products → per-word sum → cross-word sum + C, piped
   to match the FP latency, then muxed at the output by `is_int`.

Total FP latency:
`TOTAL_LATENCY = FCVT + FMUL + LEVELS·FADD + FADD = 1 + 8 + 11·LEVELS + 11`.
For `N=4` (`TCK=8`, `LEVELS=3`) that is **53 cycles**; for `N=8` (`TCK=16`)
it is **64 cycles**.

Format coverage: **FP16, BF16, and integer only** — no TF32, FP8, FP4, or any
MX scaled format. `fmt_d` is unused; output is always FP32.

### 2.2 `VX_tcu_fedp_tfr` (the fused reference)

TFR is a true fused dot product — `d = c + Σ aₖ·bₖ` evaluated with a *single*
shared exponent/alignment/accumulator/normalizer, and **one** rounding at the
end. Four pipeline stages (`MUL → ALN → ACC → NRM`), `TOTAL_LATENCY = 4`:

- **Stage 1 — `VX_tcu_tfr_shared_mul`**: per-format mantissa multipliers
  (`mul_f16`/`mul_f8`/`mul_f4`/`mul_i8`/`mul_i4`), exponent computation, the
  max-exponent search and difference matrix, exception reduction, lane mask.
  The FP16/BF16/TF32 mantissa product is a `VX_wallace_mul #(.N(11))`
  (11×11 → 22 bits) **in LUT fabric**, one per lane.
- **Stage 2 — `VX_tcu_tfr_align`**: shift each lane's significand to the common
  (max) exponent — *one* alignment network, shared.
- **Stage 3 — `VX_tcu_tfr_acc`**: sum the aligned fixed-point significands plus C
  into a wide accumulator (`ACC_SIG_W = W + 1 + HR ≈ 30` bits).
- **Stage 4 — `VX_tcu_tfr_norm_round`**: single LZC + normalize + round → FP32.

Format coverage: **everything** — fp16/bf16/tf32/fp8/fp4/mx-scaled/int8/int4.
This is the production TCU FEDP backend (`VX_CFG_TCU_TYPE_TFR`).

---

## 3. Synthesis evidence (U55C @ 300 MHz, per-FEDP, from `hw/syn/xilinx/dut/tcu`)

| DUT | impl LUT | impl FF | DSP | WNS | formats | FP latency |
|---|---:|---:|---:|---:|---|---:|
| `dsp_nt32` (wrapper, N=4) | **5029** | **5401** | **32** | **−0.403** ✗ | fp16/bf16/int | ~53 cyc |
| `tfr_fp16_nt16` | 3187 | 1113 | 0 | (core −0.599) | fp16/bf16/tf32 | 4 cyc |
| `tfr_fp16_nt32` | 3099 | 1125 | 0 | (core −0.699) | fp16/bf16/tf32 | 4 cyc |
| `tfr_sp_nt32` (all fmt) | 3084 | 1091 | 0 | **−2.388** ✗ | all incl fp8/fp4/mx/int | 4 cyc |

Two facts dominate the design space:

1. **The wrapper is strictly worse than TFR on every area axis** — ~60 % more
   LUTs, ~5× the FFs, **32 DSPs vs 0** — while covering *fewer* formats and
   carrying ~13× the latency. It is not a DSP-efficient design; it is a
   DSP-*expensive* one. Each `xil_fmul` consumes 2 DSPs + substantial LUT
   exponent/normalize logic, and each `xil_fadd` consumes 2 DSPs + LUTs;
   `8 fmul + 8 fadd → 32 DSP` for `N=4`. Scaled to a full core (32 FEDPs) this is
   **1024 DSP/core, 2048 DSP for the 2-core unit_top (22.7 % of the 9024 on the
   U55C)** — for the *least* capable backend.

2. **Both failing paths are the multiply.** The wrapper's critical path
   (−0.403 ns) is `pipe_fedp → g_int_mul[3].g_elem_lo[3].pipe_elem` — the
   hand-built **integer** multiply. TFR-SP's critical path (−2.388 ns) is
   `pipe_fedp → pipe_mul` — the **mantissa multiply** stage, 68.8 % routing,
   i.e. congestion from LUT-fabric Wallace multipliers packed densely across all
   lanes and all formats.

The DSP48E2 columns are the natural home for exactly those multiplies, and on
this device they are sitting **completely idle** in the TFR builds.

---

## 4. Why the wrapper is the wrong shape

A dot product is `d = c + Σ aₖ·bₖ`. The *only* parts that must be done per lane
are the significand multiply and the exponent add. Everything else —
finding the max exponent, aligning to it, summing, normalizing, rounding —
is **shared work that should happen once**.

The Xilinx FP IP cannot express that sharing: an `xil_fmul` returns a fully
rounded IEEE FP32 product, so the wrapper is forced to *re-expand* each rounded
product back into an aligned form inside every `xil_fadd`, and re-normalize and
re-round at every tree node. The fused TFR datapath collapses all of that into
one alignment, one accumulator, one normalize/round. The wrapper pays `2·TCK`
copies of FP exponent/normalize/round logic (in both LUTs and DSPs) to compute
something TFR computes once.

Conclusion: a DSP-efficient FEDP must be **fused like TFR**, and use DSPs only
for the dense arithmetic. There is no DSP-efficient version of the per-lane-IP
topology.

---

## 5. Proposal A — Direct-DSP fused FEDP

Rebuild `VX_tcu_fedp_dsp` as a fused datapath mirroring TFR's stage structure,
with the multiplies (and optionally the wide accumulate) on `DSP48E2`.

### 5.1 Datapath

```
            ┌── exponent add  e_k = e_a + e_b + bias_const           (LUT, tiny)
 a_k,b_k ──>│
            └── mantissa mul  p_k = m_a · m_b   ──> DSP48E2           (1 DSP/lane)
                                  │
   [max-exp search + diff matrix]│            (shared, LUT — reuse TFR max_exp)
                                  ▼
   [align: p_k >> (max_exp - e_k)]            (shared, LUT — reuse TFR align)
                                  ▼
   [accumulate: Σ aligned p_k + aligned c]    (LUT carry-chain, or DSP — see 5.3)
                                  ▼
   [normalize (LZC + shift) + round → FP32]   (shared, LUT — reuse TFR norm_round)
```

This is structurally TFR with the `VX_wallace_mul` replaced by a registered DSP
multiply. The shared stages (max-exp, align, acc, norm/round) are imported
verbatim from `hw/rtl/tcu/tfr/`.

### 5.2 Mantissa multiply on DSP48E2

The DSP48E2 multiplier is **27×18 signed**. The mixed-precision operands fit with
enormous headroom:

| format | significand (with hidden 1) | product | DSP48E2 fit |
|---|---:|---:|---|
| FP16 | 11 bits | 22 bits | 1 DSP (11≤18, 11≤27) |
| BF16 | 8 bits | 16 bits | 1 DSP |
| TF32 | 11 bits | 22 bits | 1 DSP |

Map exactly as `VX_fma_unit`'s `g_mul_dsp` branch does:

```systemverilog
(* use_dsp = "yes" *) wire [PW-1:0] prod = PW'(m_a) * PW'(m_b);
VX_pipe_register #(.DATAW(PW), .DEPTH(MUL_LATENCY)) pm (... data_in(prod) ...);
```

The `use_dsp` hint plus a registered output lets the tool pack the operand
register (AREG/BREG), the multiplier register (MREG) and the product register
(PREG) into the slice. A fully-pipelined 11×11 DSP multiply closes well above
700 MHz, so the DSP is *never* the 300 MHz bottleneck — the surrounding fused
logic is, and that is already pipelined in TFR.

**DSP budget:** `TCK` DSPs per FEDP for the FP path (1/lane), vs the wrapper's
`~4·TCK`. For `N=4`: **8 DSP vs 32** — a 4× reduction — while *adding* TF32 and
freeing the LUTs the `xil_fmul`/`xil_fadd` exponent/normalize logic consumed.

### 5.3 Accumulate: LUT carry-chain vs DSP ALU

The accumulate sums `TCK+1` aligned significands of width `ACC_SIG_W ≈ 30` bits.
Two options:

- **LUT carry-chain (default):** `CARRY8`-based adder tree. Cheap, and at ~30
  bits / shallow depth it comfortably meets 300 MHz. Recommended baseline.
- **DSP ALU (optional):** the DSP48E2 48-bit ALU with `PCOUT→PCIN` cascade can
  absorb the reduction (post-add / `OPMODE` accumulate). This frees more LUTs but
  a long combinational PCOUT cascade does **not** meet 300 MHz (this is precisely
  the failure mode documented for wide multiplies in `VX_fma_unit`'s
  `SPLIT_MUL` comment) — it must be a *registered* cascade (one DSP add per
  cycle), which adds latency. Only worth it if LUT/route pressure demands it.

### 5.4 Integer path

The current hand-built integer datapath is the wrapper's *actual* critical path
on hardware. Map its element multiplies to DSP using standard packing:

- **INT8:** DSP48E2 INT8 packing — one DSP computes two independent 8×8 products
  sharing an operand via the 27-bit A port (`A = {b2, 0, b1}`, `B = a`), the
  established "2 INT8 MACs / DSP" technique; or use the 48-bit post-adder to
  accumulate per-word partials in-DSP.
- **INT4:** four 4×4 products per DSP via the same port-packing widened.

This both removes the LUT-fabric integer multiplier (the −0.403 ns path) and
shares the DSP columns already allocated to the FP mantissa multiplies (formats
are mutually exclusive per request, so the DSP can be `fmt`-muxed between FP
mantissa and integer operands — *no extra DSP* if the muxing is done on the DSP
inputs rather than instantiating separate DSPs).

### 5.5 Expected outcome (per FEDP, N=4)

| metric | wrapper (today) | Proposal A (est.) |
|---|---:|---:|
| LUT | 5029 | ~1500–2000 (fused, no per-IP FP logic) |
| FF | 5401 | ~1200 |
| DSP | 32 | **8** (FP) / shared with int |
| WNS @300 MHz | −0.403 ✗ | **≥ 0** (target) |
| FP latency | ~53 cyc | **~6–8 cyc** |
| formats | fp16/bf16/int | fp16/bf16/tf32/int (extensible) |

---

## 6. Proposal B — `VX_CFG_TCU_TFR_DSP_ENABLE` (recommended primary path)

Because Proposal A *is* "TFR with DSP multipliers," the cleanest implementation
is not a separate module at all — it is a compile-time knob on TFR's existing
multiply stage, modeled exactly on `VX_fma_unit`'s `USE_DSP` parameter.

### 6.1 Mechanism

In `VX_tcu_tfr_mul_f16` (and, if desired, `_mul_f8`/`_mul_i8`), gate the mantissa
multiplier:

```systemverilog
`ifdef VX_CFG_TCU_TFR_DSP_ENABLE
    (* use_dsp = "yes" *) wire [21:0] man_prod = 22'(ma_sel) * 22'(mb_sel);
    // input/MREG/PREG registers absorbed into the DSP slice
`else
    VX_wallace_mul #(.N(11), .CPA_KS(...)) wtmul (.a(ma_sel), .b(mb_sel), .p(man_prod));
`endif
```

The DSP form needs registers around the product to pack AREG/MREG/PREG, so
`MUL_LATENCY` grows from 1 to **2–3** (input reg + MREG + PREG). This is the
"+2–3 additional cycles" the proposal anticipates. `TOTAL_LATENCY` and the
`vld_pipe`/`req_pipe` shift registers scale automatically off `MUL_LATENCY`, so
the change is localized to the mul stage and its latency constant.

### 6.2 Why the knob is *more* than a nicety on this device

TFR's large configs are **LUT/route-bound, not logic-bound**, with the DSP
columns idle:

- `tfr_sp_nt32` is at **31 % LUT** and fails at **−2.388 ns**, with the critical
  path *in the multiply stage* (`pipe_mul`) and **68.8 % of the delay in
  routing** — classic fabric congestion from dense LUT multipliers.
- DSP utilization in every TFR build is **0**.

Trading idle DSPs for LUTs and routing is therefore an unusually good deal here:
the U55C has **9024 DSP48E2**. Moving the FP16 mantissa multipliers to DSP at
`N=8` (`TCK=16`) removes ~16 Wallace multipliers/FEDP (~2k LUTs) and the
associated routing, at a cost of 16 DSP/FEDP — i.e. ~512 DSP/core, ~1024 for the
2-core unit (11.3 % of the device). That is the **same DSP cost as today's
wrapper**, but now buys a *fully fused, all-format* FEDP with far fewer LUTs,
instead of a subset-format wrapper with more LUTs.

### 6.3 Effectiveness assessment

**Strongly positive, with caveats:**

- ✓ **Relieves the dominant constraint.** The big SP/MX TFR configs fail in the
  multiply stage on routing; offloading multipliers to DSP is the most direct
  lever on that exact path. This is the realistic route to closing 300 MHz on
  `tfr_sp_nt32` / `tfr_wgmma_sp_nt32`.
- ✓ **DSPs are free capacity** on this device for these configs (currently 0
  used vs 9024 available).
- ✓ **Reuses the entire fused datapath and all formats** — no functional
  regression, no second module to maintain.
- ✓ **Precedent exists and is validated** (`VX_fma_unit::USE_DSP`), so the
  `(* use_dsp *)` + registered-output idiom is known-good in this tree.

- ⚠ **Latency +2–3 cycles.** Irrelevant for throughput (pipeline II stays 1) but
  it lengthens the FEDP result latency; the surrounding `vld_pipe`/scoreboard
  accounting must track the new `LATENCY` (it already derives from
  `TOTAL_LATENCY`, so this is mechanical).
- ⚠ **DSP placement congestion.** DSP48E2 sites are in fixed columns; ~1000 DSPs
  pulled from fabric multipliers create their own DSP-column-to-fabric routing
  demand. The wrapper's residual −0.403 ns was partly DSP-routing. Floorplanning
  / SLR-aware placement (see `slr_partitioning_proposal.md`) may be needed at
  scale.
- ⚠ **Small formats don't pay off 1:1.** FP4 (4×4) and FP8 (8×8) mantissa
  products are too small to justify a whole DSP each; only pack them
  (two/four per DSP) or leave them in LUTs. Gate the knob per-format (start with
  FP16/BF16/TF32, where 1 DSP/lane is clean).
- ⚠ **`use_dsp` is a hint, not a guarantee.** Verify post-synth that the products
  actually land in DSPs (the IP-free path can silently fall back to LUTs if the
  registers aren't positioned for slice packing).

### 6.4 Relationship between A and B

Proposal B subsumes Proposal A. If `VX_CFG_TCU_TFR_DSP_ENABLE` is implemented,
the standalone `VX_tcu_fedp_dsp` wrapper is **redundant** and should be
deprecated: TFR-with-DSP gives a better DSP FEDP than the wrapper across every
metric *and* covers all formats. Recommended plan:

1. Implement Proposal B (the knob) — smallest validatable increment, reuses TFR.
2. Benchmark `tfr_fp16_nt32` and `tfr_sp_nt32` with the knob on/off on the U55C
   DUT flow (`hw/syn/xilinx/dut/tcu`).
3. If timing/area targets are met, deprecate the `xil_fmul`/`xil_fadd` wrapper
   (`VX_CFG_TCU_TYPE_DSP`) rather than maintaining two DSP backends.

---

## 7. Validation plan

- **Functional (rtlsim first):** `hw/unittest/tcu_fedp` (`fedp.py`, `fedp.h`)
  across fp16/bf16/tf32/int with the knob on; bit-exact vs the existing TFR
  golden (the multiply result is mathematically identical — only its mapping
  changes).
- **End-to-end:** `tests/regression/sgemm_tcu*` (sp / mx / wgmma variants) via
  the standard flow, then on the U55C via XRT.
- **Timing/area:** per-block DUTs under `hw/syn/xilinx/dut/tcu` with unique
  `PREFIX` per run — at minimum `tfr_fp16_nt32` and `tfr_sp_nt32` with the knob
  on, compared against the rows in §3. Confirm products land in DSP via
  `post_synth_util.rpt` and that WNS ≥ 0.
- Defer all synthesis until rtlsim is green.

---

## 8. Recommendation

1. Implement **Proposal B** (`VX_CFG_TCU_TFR_DSP_ENABLE`), per-format-gated,
   starting with FP16/BF16/TF32, following the `VX_fma_unit::USE_DSP` idiom.
2. Validate on the LUT/route-bound `tfr_sp_nt32` config — this is where DSP
   offload has the highest expected payoff (failing path is the multiply, 68 %
   routing, 0 DSP used, 31 % LUT).
3. On success, **deprecate the `xil_fmul`/`xil_fadd` wrapper** (`VX_tcu_fedp_dsp`)
   — the fused TFR+DSP path dominates it on area, DSP count, latency, timing,
   and format coverage simultaneously.

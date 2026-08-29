# Vortex FPU Design (RTL)

**Scope:** logical/architectural design of the in-house Vortex floating-point
unit, covering **both single-precision (F32) and double-precision (F64)** across
the two native backends — `VX_fpu_std` (generic RTL / ASIC) and `VX_fpu_dsp`
(FPGA vendor-DSP inference). The arithmetic leaf units
([VX_fma_unit.sv](../../hw/rtl/fpu/VX_fma_unit.sv),
[VX_fdivsqrt_unit.sv](../../hw/rtl/fpu/VX_fdivsqrt_unit.sv),
[VX_fcvt_unit.sv](../../hw/rtl/fpu/VX_fcvt_unit.sv),
[VX_fncp_unit.sv](../../hw/rtl/fpu/VX_fncp_unit.sv)) are format-parameterized and
shared by both backends. The implementation is Verilog, but the design — which
op classes get dedicated vs shared datapaths, how F32 and F64 coexist, how
formats are unpacked/rounded — is implementation-agnostic.

Two other backends exist but are not the subject of this document: `VX_fpu_dpi`
(a C softfloat model for simulation) and `VX_fpu_fpnew` (the third-party CVFPU
library, now **opt-in** rather than mandatory for D — see §7).

---

## 1. Architecture overview

The FPU is a fixed set of per-operation cores fed by a shared dispatch/arbitration
front end. Operands arrive at the full FP register width (`FLEN`); F32 values are
NaN-boxed in the low 32 bits. The active format is selected per instruction from
the decoded `fmt` field, never from the operand width.

```
  FLEN-wide operands (NaN-boxed F32)        per-op cores                writeback
  ──────────────────────────────────►  ┌──────────────────────┐  ───────────────►
        fmt[0] = .D / .S select         │  FMA   (separate     │   fmt-muxed result
                                        │         F32 + F64)   │   + NaN-box F32
                                        │  DIVSQRT (merged)    │   + fflags merge
                                        │  CVT    (merged,     │
                                        │          N-format)   │
                                        │  NCP    (merged)     │
                                        └──────────────────────┘
```

The single most important design decision is **whether F64 is a second datapath
beside the obligatory F32 one (separate), or one F64-wide datapath that runs F32
as a NaN-boxed subset (merged)**. Because the D extension *requires* F, F32
hardware is always present, so the question is per op class — and it is answered
differently for each (§3).

---

## 2. Format model and the `FLEN` knob

- **`FLEN`** is the single static width knob for the FP data and control path,
  defaulted from `VX_CFG_FLEN`. The backends thread it down and derive the
  concrete `(EXP_BITS, MAN_BITS)` per format; the leaf cores stay
  format-agnostic and never see `FLEN`. All F64 logic lives in
  `generate ... if (FLEN >= 64)` (and the leaf-local `HAS_D = (FLEN >= 64)`)
  blocks, so an `FLEN == 32` build elaborates to the F32-only RTL bit-for-bit and
  the synthesizer prunes the entire 64-bit datapath — **zero area/timing cost
  when D is disabled.**
- **Operands flow at full FLEN width.** The FPR file is FLEN-wide with NaN-boxed
  F32, so operand delivery is uniformly wide regardless of the active format.
  The backends do *not* truncate to `[0+:32]`; they select the active format from
  `fmt[0]` and NaN-box F32 results back to FLEN on writeback (upper `FLEN-32`
  bits set to 1s).
- **`FLEN ⊥ XLEN`.** Float width is architecturally independent of integer
  width. The native FPU therefore supports **RV32D** (XLEN 32 / FLEN 64) on the
  32-bit synthesis target, as well as RV64F (XLEN 64 / FLEN 32), not only the
  diagonal RV32F / RV64D cases. Integer operands on the CVT side are XLEN-wide
  (I32/I64/U32/U64).

---

## 3. Separate vs merged — the per-op-class decision

The area saved by merging a unit is bounded by what the F32 version of that unit
*would have* cost. For a multiplier that is ≈ (24/53)² ≈ **20 %** of the F64
cost, so merging FMA saves little while forcing F64 latency and the F64 critical
path onto every F32 op (the hot path). For an iterative divider the iterator
dominates and F32 traffic is rare, so merging saves a lot at no cost that
matters. That asymmetry splits the decision by op class:

| Unit | Choice | Rationale |
|------|--------|-----------|
| **FMA / ADD / MUL** | **Separate** | F32 is the hot path and must not regress; dedicated F32 + F64 cores, result muxed on `fmt[0]`. |
| **DIV / SQRT** | **Merged** | Area-dominant iterative unit, rare traffic; one F64-wide iterator serves both (F32 = fewer iterations). |
| **CVT** | **Merged, N-format** | Conversion *is* cross-format (FCVT.S.D is the point); one shared unpack + align + round tree sized to the widest enabled format. Off the hot path. |
| **NCP** (sgnj/min/max/cmp/class/fmv) | **Merged** | Pure combinational, off the hot path; one FLEN-wide format-aware unit, no meaningful F32 cost. |

Only FMA is duplicated; everything else merges onto a single F64-wide datapath.
This mirrors the trusted FPNEW configuration (`MERGED` DIVSQRT/CONV, dedicated
FMA), with NONCOMP additionally merged because it is trivially cheap.

**Invariant:** F32 timing and area must not regress on the U55C / 300 MHz target.
That hard constraint is what forces FMA to stay separate and is honored
throughout the §4 datapaths.

---

## 4. Subunit designs

### 4.1 FMA — separate F32 / F64 cores

[VX_fma_unit.sv](../../hw/rtl/fpu/VX_fma_unit.sv) is one parametric core
(`MAN_BITS`, `EXP_BITS`, `LATENCY`, `USE_DSP`). The backend instantiates it twice
— `fma32` (`23, 8`) and, under `generate if (FLEN >= 64)`, `fma64` (`52, 11`) —
and muxes the result/flags on the FMA-latency-delayed `fmt[0]` selector. The F32
instance is unchanged from the F32-only design.

Pipeline (latency-budgeted via localparams, summing to `LATENCY`):

```
 INI → MUL(variable) → ALN(1–2) → ACC → NRM → RND
  1        L-5/L-6        1/2       1     1     1
```

- **MUL — DSP-cascade-friendly multiply.** The mantissa multiply is the
  routing/timing-critical path at the F64 width (53×53). It is pipelined behind
  a portable `USE_DSP` parameter:
  - `SPLIT_MUL = (USE_DSP != 0) && (SIG_BITS > 24) && (MUL_LATENCY >= 2)` — i.e.
    **F64 only**. The product is split into partial products
    (`a*b_lo`, `a*b_hi`) each tagged `(* use_dsp = "yes" *)`, registered, then
    summed with the high part shifted — a structure Vivado retimes into a
    pipelined DSP48E2 PCIN cascade. A flat `a*b` does **not** retime into the
    cascade and misses timing.
  - F32 takes the flat `(* use_dsp *)` multiply (single DSP, already meets
    timing). `use_dsp` is an FPGA hint; ASIC synthesis ignores it, so the source
    stays portable.
- **ALN — format-gated barrel shifter.** `ALN_LATENCY = (MAN_BITS+1 > 24) ? 2 : 1`.
  The alignment shift is the other F64-width-critical path, so for F64 it is
  split into a **coarse** shift (high bits) and a **fine** shift (low `FINE_BITS`)
  with a register between, exact because `(x >> coarse) >> fine == x >> amt` and
  the sticky bit is taken from the fully-shifted value. **F32 uses a single-stage
  shifter** — it already meets timing, so it pays neither the extra register nor
  its area. The total `LATENCY` is unchanged for F32 (the MUL stage absorbs the
  freed cycle).
- **ACC/NRM/RND** — single-rounding accumulate, leading-zero normalize, round.
  Because F32 and F64 are physically separate cores, there is **no double-rounding
  hazard** (no F32 → F64-core → F32 path).

**Latencies** (`VX_CFG_LATENCY_FMA`): native STD = **8** (F32) / **12** (F64);
DSP-Vivado = 16; DSP-Quartus = 4. F64 is deeper so the cascade/retimed multiplier
runs at speed; vendor-IP backends keep their fixed IP latency.

### 4.2 DIVSQRT — merged radix-2 carry-save Newton–Raphson

[VX_fdivsqrt_unit.sv](../../hw/rtl/fpu/VX_fdivsqrt_unit.sv) is a single F64-wide
iterator that also serves F32 with fewer iterations. It takes `parameter FLEN`
and sizes everything from `SUPER_MAN = HAS_D ? 52 : 23`:

| Derived param | F32 | F64 |
|---|---|---|
| `SUPER_SIG` (significand) | 24 | 53 |
| `NR_STAGES` (SRT iterations) | 13 | 28 |
| `EXP_W` (working exponent) | 10 | 14 |
| Latency (`FDIV` = `FSQRT`) | 17 | 32 |

- Shared division/sqrt recurrence in **carry-save** form (no carry-propagate in
  the loop); format-aware unpack, normalize, and final conversion/rounding at the
  active significand width.
- Iterative and area-dominant but **off the critical path** — it is LUT-bound,
  not timing-bound; the F64 iteration count rises but per-stage logic is
  unchanged.
- **STD constraint:** `VX_CFG_LATENCY_FDIV` must equal `VX_CFG_LATENCY_FSQRT`
  (shared serializer in `VX_fpu_std`); the config expressions enforce this
  (both 17 for F32, 32 for F64).

### 4.3 CVT — merged N-format converter

[VX_fcvt_unit.sv](../../hw/rtl/fpu/VX_fcvt_unit.sv) is the only inherently
cross-format op class, so it is one datapath sized to the widest enabled format
(`SUPER_EXP`, `SUPER_MAN`) rather than a hardwired F32/F64 pair. Latency
`VX_CFG_LATENCY_FCVT = 5`.

- **Unpack** muxes the source format's exp/man fields, re-biases, and
  left-aligns into a canonical internal form (F64 fields guarded under `HAS_D`).
- **Round/pack** narrows the canonical value to the destination precision/bias
  through one shared rounding tree, reused for every format pair.
- **F2F narrowing (FCVT.S.D)** correctly handles the hard cases: destination
  **subnormal** (denormalize-and-round with a correct guard/round/sticky split)
  and **overflow** to ±inf, plus NaN/Inf/Zero propagation in the float domain.
- **Int side** converts {I32, I64, U32, U64} ↔ any float format; integer width is
  XLEN.
- The format-table generality is free when unused: a `{FP32, FP64}` build is
  area-identical to a hardwired two-format converter, and unreachable encodings
  (e.g. BF16) are pruned when the decoder never emits their selector.

### 4.4 NCP (non-computational) — merged FLEN-wide

[VX_fncp_unit.sv](../../hw/rtl/fpu/VX_fncp_unit.sv) handles FSGNJ[N/X],
FMIN/FMAX, FCMP (EQ/LT/LE), FCLASS, and FMV.X.{W,D}/FMV.{W,D}.X in one
FLEN-wide, format-aware unit selecting on `fmt[0]`. Latency
`VX_CFG_LATENCY_FNCP = 2`. Per-format canonical qNaN and NaN-boxing are
generated under `HAS_D`; combinational and off the hot path, so the merge costs
no meaningful F32 area. `VX_fp_classifier` and `VX_fp_rounding` are already
format-generic and instantiated per format by the units above.

---

## 5. Backend integration

`VX_fpu_std` (generic RTL / ASIC) and `VX_fpu_dsp` (FPGA) are structurally
identical and differ only in how the FMA multiply maps to hardware:

1. Both take `FLEN` and size all FP operand/result/tag widths from it — no
   literal `32`/`64` in the datapath.
2. Each op-class core is fed a full-FLEN operand with an `fmt[0]`-driven format
   select. F64 instances (the `fma64` core, the wider DIVSQRT/CVT/NCP datapaths)
   sit under `generate if (FLEN >= 64)`.
3. F32 results are NaN-boxed to FLEN on writeback (`{{(FLEN-32){1'b1}}, res32}`);
   vendor-IP F32 results in the DSP backend are boxed identically.
4. **`VX_CFG_USE_DSP`** ([VX_fpu_define.vh](../../hw/rtl/fpu/VX_fpu_define.vh)) is
   `1` under `VIVADO`/`QUARTUS`, else `0`. It gates the FMA split-multiply and the
   `use_dsp` attributes, so the same RTL infers DSP cascades on FPGA and lowers to
   a plain retimed multiplier on ASIC.

The result mux and tag/handshake plumbing already existed per op class; F64
support is additive instantiation plus the format select, not a rewrite.

---

## 6. Worked example — `fmt[0]` flow

For `fadd.d` vs `fadd.s` issued to the FMA core (ADD maps onto FMA with an
implicit ×1):

1. Both operands arrive FLEN-wide; for `.s` the low 32 bits are the NaN-boxed
   value.
2. `fmt[0]` (1 for `.d`) routes the operands into `fma64` / `fma32` respectively;
   both cores run, only the selected result is used.
3. The selector is delayed by `VX_CFG_LATENCY_FMA` cycles so it lines up with the
   result emerging from the (separate-latency) cores.
4. The `.s` result is NaN-boxed to FLEN on writeback; the `.d` result fills the
   full width. fflags from the active core are merged into the lane fflags.

---

## 7. Configuration

[VX_config.toml](../../VX_config.toml), `[fpu]`:

- **`VX_CFG_FPU_TYPE`** = `('STD' if $ASIC else 'DSP')` under synthesis, else
  `DPI`/`STD` for simulation. There is **no unconditional FPNEW-on-64-bit rule**:
  the native backends carry FLEN=64, and FPNEW is selected only via an explicit
  `VX_CFG_FPU_TYPE = "FPNEW"`.
- **`VX_CFG_EXT_D_ENABLE`** is independently settable, making RV32D (`XLEN_32` +
  `FLEN_64`) expressible on the 32-bit U55C build; `VX_CFG_FPU_RV64F`
  (`XLEN_64` + `FLEN_32`) is also supported.
- Latencies are EXT_D- and backend-aware: `FMA` 8/12, `FDIV`/`FSQRT` 17/32 for
  native, with the STD `FDIV == FSQRT` invariant preserved across all branches.

---

## 8. Verification & status

Per the SimX-as-oracle methodology: SimX models F32/F64 exactly (rvfloats
softfloat) and is the value/cycle reference; native-RTL bring-up diffs SimX↔RTL
traces to localize datapath bugs, and RTL coverage runs through XRT (the
synthesizable AFU surface) rather than rtlsim alone.

**Status — complete.** The native STD and DSP backends run full F32 and F64. The
RISC-V ISA suites pass on both native backends across both integer widths, with
zero-warning Verilator builds:

| backend | rv32uf | rv64uf | rv64ud |
|---|---|---|---|
| STD | ✅ | ✅ | ✅ |
| DSP | ✅ | ✅ | ✅ |

**Timing (U55C, 300 MHz target).** Both configurations close timing on the FPU
DUT: F32-only (`build32`) ≈ 322 MHz, F32+F64 (`build64`) ≈ 302 MHz. The F64 FMA
multiply DSP cascade and the F64 alignment shifter were the two paths that drove
the split-multiply and coarse/fine-aligner pipelining in §4.1; F32 timing and
area are unchanged from the F32-only baseline by construction (separate FMA core,
format-gated aligner).

**Resource utilization (U55C, `VX_fpu_std` DUT, post-place, OPT_LEVEL=3).**

| Resource | F32 (`build32`) | F32+F64 (`build64`) |
|---|---|---|
| Total LUT | 27,086 | 104,722 |
| — LUT as logic | 27,062 | 104,028 |
| LUTRAM (LUT as memory) | 24 | 694 |
| Flip-flops (FF) | 13,883 | 42,017 |
| Block RAM (RAMB36 / RAMB18) | 0 / 0 | 0 / 0 |
| DSP48E2 | 8 | 48 |

The FPU is pure logic + arithmetic — no Block RAM or URAM, and only a trivial
amount of LUTRAM. DSP scales 8 → 48 with F64 enabled (the F64 FMA cores' split
53×53 multiplies map onto DSP48E2 cascades). The F64 build is ≈ 3.9× the F32
LUT/FF, consistent with adding the wider FMA, divsqrt, CVT and NCP datapaths
alongside the retained F32 cores.

> The LUT/FF figures above are from the two-stage-aligner synthesis; the
> format-gated aligner (single-stage F32, §4.1) trims the F32 path back toward
> its ~26,800-LUT baseline and slightly reduces the `fma32` cores inside the F64
> build. DSP, Block RAM, and LUTRAM are unaffected.

FPNEW remains available as an independent third reference for the same FP64
vectors.

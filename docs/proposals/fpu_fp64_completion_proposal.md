# Completing the Native 64-bit FPU — FP64 across the STD/DSP Backends

Status: draft
Target: Alveo U55C, 300 MHz timing closure
Scope: hw/rtl/fpu, hw/rtl/core (decode already done), VX_config.toml, tests

---

## 1. Motivation

Double-precision floating point (RV64D, and RV32D) is functionally complete in
SimX and in the decoder, but in RTL it executes **only** through the third-party
FPNEW backend. The native Vortex FPU backends are F32-only. `VX_config.toml`
states this explicitly:

```
# 64-bit => FPNEW (STD/DSP are F32-only)
VX_CFG_FPU_TYPE = expr: ('FPNEW' if $VX_CFG_XLEN_64 else ('STD' if $ASIC else 'DSP')) ...
```

This forces an undesirable coupling: any double-precision configuration must
fall back to FPNEW regardless of the synthesis target. FPNEW is a large,
generically-parameterized library whose `MERGED` DIVSQRT and `DISTRIBUTED` pipe
layout are not tuned for our U55C/300 MHz budget, and it sits outside the Vortex
RTL style and verification flow. Concretely:

- **FPGA (DSP backend) loses FP64 entirely.** The DSP backend maps FMA/DIV/SQRT
  onto device DSP primitives and is our path to good area/timing on U55C. With
  FP64 routed to FPNEW instead, the curated DSP datapath is bypassed.
- **No native FP64 means no RV32D on the 32-bit build.** FLEN is architecturally
  independent of XLEN (RV32D = XLEN 32 / FLEN 64), but today `VX_CFG_FLEN_64`
  only ever pairs with `XLEN_64` because the native backends can't widen.
  Completing native FP64 unblocks RV32D on the **32-bit** synthesis target we
  actually build and synthesize — without dragging in FPNEW.
- **Verification asymmetry.** SimX already models FP64 exactly (rvfloats
  softfloat). We can use it as the cycle/value oracle for the native datapath,
  which we cannot meaningfully do against FPNEW.

**Goal:** extend the native STD and DSP backends and their subunits to full
double precision so that any FLEN=64 configuration runs on native Vortex RTL,
and FPNEW becomes optional rather than mandatory for 64-bit.

---

## 2. Current state

| Area | Status | Evidence |
|------|--------|----------|
| Config / `VX_CFG_FLEN`, FLEN⊥XLEN | DONE | `VX_config.toml:27-28,123-124` |
| Decode `.S`/`.D` (`fmt` field), FCVT.D.S/.S.D | DONE | `VX_decode.sv` (fmt[0] dispatch; gated on `VX_CFG_FLEN_64`) |
| RVC C.FLD / C.FSD | DONE | `VX_decompressor.sv` |
| SimX FP64 (oracle) | DONE | `sim/simx/fpu_unit.cpp` (`rv_*_d`) |
| FPNEW backend FP64 | DONE | `VX_fpu_fpnew.sv` (FpFmtMask FP32+FP64) |
| `VX_fma_unit` | READY | parameterized `MAN_BITS`/`EXP_BITS` (F32 default, F64 capable) |
| `VX_fp_classifier`, `VX_fp_rounding` | READY | fully parameterized |
| **`VX_fdivsqrt_unit`** | **F32-only** | `MAN_BITS=23`, `EXP_BITS=8` hardcoded |
| **`VX_fcvt_unit`** | **F32-only** | hardcoded F32 bias/width; no F64 src/dst path |
| **`VX_fncp_unit`** | **F32-only** | 32-bit input ports; no `.D` SGNJ/MINMAX/CLASS/FMV |
| **`VX_fpu_dsp` routing** | **F32-only** | extracts `[0+:32]` from XLEN operands |
| **`VX_fpu_std` routing** | **F32-only** | extracts `[0+:32]` from XLEN operands |
| RV64D / FP64 tests | MISSING | no `rv64ud` / double regression cases |

So the gap is four RTL items (three subunits + backend routing in two files) plus
test coverage. FMA, classify, rounding, decode, config, and SimX are already
there.

---

## 3. Scope

In scope:

- Parameterize `VX_fdivsqrt_unit`, `VX_fcvt_unit`, `VX_fncp_unit` on
  `(EXP_BITS, MAN_BITS)` so each can be instantiated for F32 or F64.
- Format-aware operand routing and result NaN-boxing in `VX_fpu_dsp` and
  `VX_fpu_std`.
- A native-FP64 instantiation strategy (Section 5) for both backends.
- `VX_config.toml` changes to allow native FP64 (STD/DSP) instead of forcing
  FPNEW, and to expose RV32D.
- `rv64ud` + double regression coverage; XRT as the RTL coverage path.

Out of scope:

- Changing FPNEW (it stays as an alternative backend).
- Vector/TCU FP64.
- Any SimX change beyond confirming oracle parity (already complete).

---

## 4. Design principles

1. **Single widest-format unit per op class, format-dispatched by `fmt[0]`.**
   F64 is a superset of F32 in IEEE-754 semantics; a correctly parameterized
   F64 unit can execute F32 by zero/round handling, but doing so on a real GPU
   wastes the F32 fast path. We mirror what shipping GPUs and FPNEW's `PARALLEL`
   mode do for ADD/MUL/NONCOMP and `MERGED` for DIV/SQRT/CONV (see Section 5).
2. **Operands flow at full FLEN width.** The backends already receive
   XLEN-wide lanes. We stop the `[0+:32]` truncation and select the active
   format from `fmt[0]`; F32 results are NaN-boxed to FLEN on writeback (the
   boxing path already exists).
3. **F32 timing/area must not regress.** This is the hard constraint on U55C.
   Adding an F64 datapath cannot lengthen the F32 critical path or blow the DSP
   budget. This drives the parallel-vs-merged choice per op class.
4. **SimX is the oracle.** Build/confirm SimX parity first, then diff SimX↔RTL
   trace dumps to localize any native-FP64 datapath bug.

---

## 5. Subunit work

### 5.1 `VX_fma_unit` — already parameterized
Instantiated today with F32 defaults. ADD/SUB/MUL/MADD are latency-bound and
DSP-heavy, so use **parallel** F32 and F64 instances (matching FPNEW `PARALLEL`
ADDMUL) selected by `fmt[0]`. The F64 instance uses `MAN_BITS=52, EXP_BITS=11`.
No new logic — just a second instantiation + mux.

### 5.2 `VX_fdivsqrt_unit` — parameterize (medium)
Today: single-lane radix-2 non-restoring DIV + SQRT, `MAN_BITS=23`,
`EXP_BITS=8`, 13 SRT iterations, 17-cycle latency.

- Promote `MAN_BITS`/`EXP_BITS` to parameters; the SRT iteration count and the
  carry-save width derive from `MAN_BITS` (F64 needs ~52+ iterations).
- DIV/SQRT are area-expensive and latency-tolerant ⇒ use a **merged** F32/F64
  unit (one F64-width iterator that also serves F32), matching FPNEW `MERGED`
  DIVSQRT. This keeps the DSP/LUT footprint bounded; F64 latency rises but DIV
  is rare.
- New latency is format-dependent; surface it as `VX_CFG_LATENCY_FDIV/FSQRT`
  derived expressions. Note the existing STD constraint that FDIV and FSQRT
  latencies must match (shared serializer in `VX_fpu_std`).

### 5.3 `VX_fcvt_unit` — parameterize (high)
Today: F32-only float side; integer side already handles I32/I64
(`is_src_64`/`is_dst_64`). Needs the full conversion cross-product:

- Float formats: F32 ↔ F64 (`FCVT.S.D`, `FCVT.D.S`).
- Int↔float: {I32,I64,U32,U64} ↔ {F32,F64} (`FCVT.D.W/.WU/.L/.LU` and inverse).
- Parameterize bias/exp/mantissa; the integer width is already XLEN; add the
  float-format select from `fmt`.
- Use a **merged** widest-format datapath (matches FPNEW `MERGED` CONV) — CONV
  is not on the hot path and merging avoids duplicating the alignment/rounding
  tree.

### 5.4 `VX_fncp_unit` — width-parameterize (medium)
Today: 32-bit ports; handles FSGNJ[N/X], FMIN/FMAX, FCLASS, FMV.X.W/FMV.W.X.

- Widen ports to FLEN; parameterize on `(EXP_BITS, MAN_BITS)`.
- Add `.D` variants: FSGNJ.D, FMIN.D/FMAX.D, FCLASS.D, FMV.X.D/FMV.D.X.
- NONCOMP is cheap ⇒ **parallel** F32/F64 (matches FPNEW `PARALLEL` NONCOMP);
  `VX_fp_classifier` is already parameterized and reused per format.

### 5.5 `VX_fp_classifier`, `VX_fp_rounding` — no change
Already format-generic; instantiated per format by the units above.

---

## 6. Backend integration

`VX_fpu_dsp` (FPGA) and `VX_fpu_std` (ASIC) change identically:

1. Replace each `[0+:32]` operand slice with a full-FLEN operand and an
   `fmt[0]`-driven format select into the (now parameterized) subunit.
2. Instantiate subunits per the parallel/merged policy in Section 5
   (parallel: FMA, NCP; merged: DIVSQRT, CVT).
3. Keep the existing NaN-boxing on writeback for F32 results in FLEN=64 configs.
4. Gate the F64 instances behind `VX_CFG_FLEN_64` so FLEN=32 builds are
   bit-identical to today (zero area cost when D is disabled).

The result-mux and tag/handshake plumbing already exist per op class; this is
additive instantiation plus the format select, not a rewrite.

---

## 7. Configuration changes (`VX_config.toml`, `[fpu]`)

- Drop the unconditional `'FPNEW' if $VX_CFG_XLEN_64` rule. New policy:
  native (STD on ASIC / DSP on FPGA) supports FLEN=64; FPNEW stays selectable
  via explicit `VX_CFG_FPU_TYPE = "FPNEW"`.
- Make `VX_CFG_EXT_D_ENABLE` independently settable so RV32D
  (`XLEN_32` + `FLEN_64`) is expressible — this is the configuration that lets
  the **32-bit** U55C build carry double precision.
- Re-derive `VX_CFG_LATENCY_FDIV/FSQRT` for the format-dependent native paths.
- Keep `VX_CFG_FPU_RV64F` (XLEN_64 + FLEN_32) working — verify routing doesn't
  assume FLEN==XLEN.

(Exact expressions to be finalized with the latency numbers from 5.2.)

---

## 8. Verification

Per the SimX-as-oracle approach:

1. **SimX parity first** — confirm `rv64ud`/`rv32ud` pass in SimX (expected
   already green), establishing the value/cycle reference.
2. **Native RTL bring-up** — run the same kernels on STD (ASIC sim) and DSP
   (FPGA sim) native FP64; diff SimX↔RTL trace dumps to localize datapath bugs.
3. **RTL coverage via XRT**, not rtlsim, since the goal is the synthesizable
   AFU surface.

Test additions:

- `rv64ud` / `rv32ud` riscv-tests (fadd.d, fmul.d, fdiv.d, fsqrt.d, fcvt.*,
  fsgnj.d, fmin/fmax.d, fclass.d, fmadd.d, fld/fsd).
- A double-precision regression kernel (e.g. a `dgemm`/`daxpy` analog to the
  existing `sgemm`/`saxpy`).
- Cross-format conversion stress (F32↔F64, I64↔F64).
- Regression matrix: native-STD, native-DSP, and FPNEW must all pass the same
  FP64 vectors (FPNEW as a third independent reference).

---

## 9. Timing / area (U55C, 300 MHz)

The native DSP FP64 datapath must close at 300 MHz and stay within the DSP/LUT
budget. Risk areas and mitigations:

- **FMA F64** (52×52 partial products) — rely on Vivado DSP cascades; if the
  multiplier array doesn't meet 300 MHz, add a pipeline stage to the F64
  instance only (F32 latency unchanged).
- **DIVSQRT F64** — merged iterator is LUT-bound, not timing-bound; iteration
  count rises but per-stage logic is unchanged. Watch the CPA in the recurrence.
- **F32 paths unchanged** — parallel F32 instances guarantee no F32 regression;
  this is the explicit non-negotiable.

A per-format area/Fmax table on U55C is a deliverable of Phase 1.

---

## 10. Phased plan (feature branch)

Each phase is a substantial, end-to-end-testable increment; commit only at full
completion of the feature, tested against the SimX oracle.

- **Phase 0 — parameterize subunits.** `VX_fdivsqrt_unit`, `VX_fcvt_unit`,
  `VX_fncp_unit` gain `(EXP_BITS, MAN_BITS)` parameters; F32 instantiation is
  bit-identical to today (regression: existing F32 suite unchanged).
- **Phase 1 — native FP64 in one backend (DSP).** Format-dispatched routing +
  F64 instances behind `VX_CFG_FLEN_64`; `rv32ud`/`rv64ud` green on DSP via
  SimX↔RTL diff; U55C Fmax/area table.
- **Phase 2 — native FP64 in STD.** Same for the ASIC backend; honor the
  FDIV==FSQRT latency constraint.
- **Phase 3 — config + RV32D.** Flip `VX_config.toml` so native backends carry
  FLEN=64; expose RV32D on the 32-bit build; FPNEW becomes opt-in. Full
  three-backend FP64 regression via XRT.

---

## 11. Open questions

- Parallel-vs-merged is proposed per op class (Section 5); confirm against U55C
  area once Phase 1 numbers exist — merge more aggressively if DSP-bound.
- Should F32-on-F64-hardware ever be allowed (single merged FMA) to save DSPs on
  area-critical builds, accepting an F32 latency hit? Default: no (parallel),
  revisit only if DSP budget forces it.
- RV32D ABI/calling-convention coverage in the toolchain — confirm `llvm-vortex`
  emits `.d` ops for XLEN=32/FLEN=64 targets.

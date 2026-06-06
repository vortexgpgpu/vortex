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

- Parameterize `VX_fdivsqrt_unit` and `VX_fncp_unit` on `(EXP_BITS, MAN_BITS)`
  so each can be instantiated for F32 or F64; rebuild `VX_fcvt_unit` as a
  configurable N-format converter (FP64/FP32/FP16/custom, up to 8 formats via
  SV parameters — see §5.3).
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

## 4. Architecture decision — separate vs merged pipelines

Because **the D extension requires the F extension**, F32 hardware is mandatory
and always present. So the design question is never "F32 or F64" — it is whether
F64 is a *second* datapath beside the obligatory F32 one (**separate**), or a
*single* F64-wide datapath that runs F32 as a NaN-boxed subset (**merged**).

### 4.1 Separate pipelines
Distinct F32 and F64 units; `fmt[0]` muxes the result.

- **+** The existing tuned F32 path is bit-identical and timing-unchanged — zero
  F32 regression on U55C, the one non-negotiable constraint.
- **+** Each unit optimal for its width; latencies tuned independently.
- **+** Clean verification: the F32 regression is unchanged; F64 is purely additive.
- **−** Area ≈ F32_area + F64_area; only one unit is active per instruction, so
  utilization is poor. On FPGA the F64 FMA already consumes many DSP cascades.

### 4.2 Merged pipeline
One F64-wide datapath; F32 unpacked/upconverted in, rounded/repacked out.

- **+** Single arithmetic core — less total area than two full units; one
  verification target for the core.
- **−** F32 pays F64 latency **and** the F64 critical path → F32 timing/area
  regression on the *common* GPU case.
- **−** Correctness hazard: F32 → F64-core → F32 can **double-round** (notably
  FMA) unless explicitly guarded.

### 4.3 The decision is per functional unit, not global
The area saved by merging is bounded by what the F32 unit *would have* cost. For
a multiplier that is ≈(24/53)² ≈ **20%** of the F64 cost — so merging FMA saves
little area while imposing F64 latency on every F32 op (bad trade for the hot
path). For an iterative divider the iterator dominates and F32 traffic is rare,
so merging saves a lot at no cost that matters. That asymmetry splits the
decision by op class:

| Unit | Choice | Rationale |
|------|--------|-----------|
| **FMA / ADD / MUL** | **Separate** | F32 is the hot path and must not regress; dedicated F32+F64 cores. Matches NVIDIA (physically distinct FP32/FP64 cores; F32 never runs on FP64 units). |
| **DIV / SQRT** | **Merged** | Area-dominant iterative unit, rare traffic; one F64-wide iterator serves both (F32 = fewer iterations). |
| **CVT** | **Merged**, generic | Conversion *is* cross-format (FCVT.S.D is the point); built as a configurable N-format engine (FP64/FP32/FP16/BF16/custom, up to 8 via SV parameters) over one shared align+round tree. Off the hot path. See §5.3. |
| **NONCOMP** (sgnj/min/max/class/fmv) | **Merged** | Pure combinational and off the hot path; one FLEN-wide format-parameterized unit handles both formats with no meaningful F32 cost, avoiding duplicate logic. |

Only FMA is separate; everything else merges onto a single F64-wide datapath.
This is corroborated by **FPNEW**, which we already ship and trust: it uses
`MERGED` for DIVSQRT/CONV (we extend the same choice to the trivially-cheap
NONCOMP), keeping a dedicated fast path only where it matters — the FMA core.

### 4.4 Invariant principles (apply to both)
1. **`FLEN` is the single static knob for the FP data + control path.** A module
   `parameter FLEN` (defaulted from `VX_CFG_FLEN`) is threaded through the FPU
   hierarchy and statically sizes *both* the datapath (operand/result port
   widths, mantissa/exponent widths, NaN-box width) *and* the control path
   (the `fmt` selector width, op decode, format-dependent latency). The F64
   logic lives in `generate ... if (FLEN == 64)` blocks, so an `FLEN == 32`
   build elaborates to today's RTL bit-for-bit and the synthesizer prunes the
   entire 64-bit datapath — zero area/timing cost when D is disabled. This
   replaces ad-hoc `ifdef VX_CFG_FLEN_64` scattering with one parameter that is
   per-instance overridable and testable.
2. **Operands flow at full FLEN width.** The FPR file is FLEN-wide with NaN-boxed
   F32, so operand *delivery* is uniformly 64-bit regardless of the core choice.
   We stop the `[0+:32]` truncation and select the active format from `fmt[0]`;
   F32 results are NaN-boxed to FLEN on writeback (the boxing path already exists).
3. **F32 timing/area must not regress.** Hard constraint on U55C; it is what
   drives every separate-vs-merged call in §4.3.
4. **SimX is the oracle.** Confirm SimX parity first, then diff SimX↔RTL trace
   dumps to localize any native-FP64 datapath bug.

---

## 5. Subunit work

Per the §4.3 split: **separate** F32/F64 instances for FMA only; a **merged**
F64-wide unit for DIVSQRT, CVT, and NONCOMP.

**Parameter layering.** `FLEN` is a *backend-level* knob only (§4.4(1)). The
backend (`VX_fpu_dsp`/`VX_fpu_std`) derives the concrete `(EXP_BITS, MAN_BITS)`
per format and passes them down; the arithmetic leaves stay format-agnostic and
never see `FLEN`. This keeps the cores reusable (an F16 or custom instance is
just a different `(EXP_BITS, MAN_BITS)`) and confines the FLEN-gated `generate`
blocks to the backend, not the leaf modules.

### 5.1 `VX_fma_unit` — separate (already parameterized)
Instantiated today with F32 defaults. Add a second F64 instance
(`MAN_BITS=52, EXP_BITS=11`) and mux on `fmt[0]`. No new logic — second
instantiation + mux; the existing F32 instance is untouched.

### 5.2 `VX_fdivsqrt_unit` — merged (medium)
Today: single-lane radix-2 non-restoring DIV + SQRT, `MAN_BITS=23`,
`EXP_BITS=8`, 13 SRT iterations, 17-cycle latency.

- Promote `MAN_BITS`/`EXP_BITS` to parameters; the SRT iteration count and the
  carry-save width derive from `MAN_BITS` (F64 needs ~52+ iterations).
- Instantiate one F64-width iterator that also serves F32 (fewer iterations).
- Latency becomes format-dependent; surface via `VX_CFG_LATENCY_FDIV/FSQRT`
  derived expressions. Honor the existing STD constraint that FDIV and FSQRT
  latencies must match (shared serializer in `VX_fpu_std`).

### 5.3 `VX_fcvt_unit` — merged, generic N-format engine (high)
Conversion is the only op class that is *inherently* cross-format, so this unit
is built as a fully configurable multi-format converter rather than a hardwired
F32/F64 pair. Today it is F32-only on the float side; the integer side already
handles I32/I64 (`is_src_64`/`is_dst_64`).

**Parameterization.** The unit takes a compile-time table of float formats and
sizes its datapath to the widest enabled one:

```systemverilog
// One encoding entry per supported float format (à la fpnew_pkg::fp_encoding_t)
typedef struct packed { int unsigned EXP_BITS; int unsigned MAN_BITS; } fp_encoding_t;

parameter int NUM_FMTS = 4;                       // up to 8
parameter fp_encoding_t FMTS [NUM_FMTS] = '{
    '{11, 52},   // FP64
    '{ 8, 23},   // FP32
    '{ 5, 10},   // FP16  (Zfh)
    '{ 8,  7}    // BF16  (custom)
    // ... up to 8 entries, any custom {exp,man}
};
localparam int FMT_SEL_W = $clog2(NUM_FMTS);
localparam int SUPER_EXP = /* widest EXP_BITS over FMTS */;
localparam int SUPER_MAN = /* widest MAN_BITS over FMTS */;
```

**Datapath (single, format-indexed).**
- `src_fmt` / `dst_fmt` are `FMT_SEL_W`-wide selects from decode.
- **Unpack**: mux the source format's exp/man fields, re-bias and left-align into
  one internal canonical form sized to `(SUPER_EXP, SUPER_MAN)`.
- **Round/pack**: a single shared rounding tree narrows the canonical value to the
  destination format's precision/bias — built once, reused for every format pair
  (this is the whole point of merging: one alignment + rounding tree, not N²).
- **Int side**: {I32,I64,U32,U64} ↔ any float format; integer width is already
  XLEN.

**Relationship to `FLEN`.** The converter is sized by its own `FMTS[]` table,
not by `FLEN` — the two are not competing knobs. `FLEN` is the *float
register/datapath* width (it bounds which formats the rest of the FPU and the
regfile can hold and therefore which table entries the decoder can actually
reach); `FMTS[]` enumerates the encodings this converter is *built* to handle.
A sane build keeps them consistent (every reachable format ⊆ `FMTS[]`, and no
entry wider than `FLEN`), which the config layer (§7) derives rather than
hand-sets.

**ISA vs configurable capability.** The RISC-V `fmt` field is 2 bits (S/D/H/Q),
so the ISA exercises at most four standard formats; the extra entries (BF16,
custom encodings) are a config-time capability for Vortex extensions and are
simply unreachable when the decoder never emits their selector. A build that
enables only `{FP32, FP64}` (`NUM_FMTS = 2`) is area-identical to a hardwired
F32/F64 converter, so the generality costs nothing when unused.

> Note: this format-table style is reusable by the other merged units
> (DIVSQRT, NONCOMP) if FP16/custom *arithmetic* is later wanted; for now only
> CVT needs the full N-format generality, since the others run within a single
> format per op.

### 5.4 `VX_fncp_unit` — merged (medium)
Today: 32-bit ports; handles FSGNJ[N/X], FMIN/FMAX, FCLASS, FMV.X.W/FMV.W.X.

- Widen ports to FLEN; parameterize on `(EXP_BITS, MAN_BITS)` as one F64-wide,
  format-aware unit selecting on `fmt[0]`.
- Add `.D` variants: FSGNJ.D, FMIN.D/FMAX.D, FCLASS.D, FMV.X.D/FMV.D.X.
- Combinational and off the hot path ⇒ a single merged unit avoids duplicate
  logic at no meaningful F32 cost; `VX_fp_classifier` is already parameterized.

### 5.5 `VX_fp_classifier`, `VX_fp_rounding` — no change
Already format-generic; instantiated per format by the units above.

---

## 6. Backend integration

`VX_fpu_dsp` (FPGA) and `VX_fpu_std` (ASIC) change identically:

1. Take `FLEN` as a `parameter` (from `VX_CFG_FLEN`) and size all FP operand,
   result, and tag widths from it — no literal `32`/`64` in the datapath.
2. Replace each `[0+:32]` operand slice with a full-FLEN operand and an
   `fmt[0]`-driven format select into the (now parameterized) subunit.
3. Instantiate subunits per the §4.3 policy
   (separate: FMA only; merged: DIVSQRT, CVT, NONCOMP).
4. Keep the existing NaN-boxing on writeback for F32 results in FLEN=64 configs.
5. Wrap the F64 instances in `generate ... if (FLEN == 64)` so an `FLEN == 32`
   build is bit-identical to today and the 64-bit datapath is pruned (zero area
   cost when D is disabled) — per §4.4(1).

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
- **F32 FMA unchanged** — the dedicated F32 FMA instance (§4.3) guarantees no
  regression on the hot path, the explicit non-negotiable. The merged DIVSQRT/
  CVT/NCP units do run F32 on the wide datapath, but they sit off the critical
  path with large slack, so F32 timing there is not at risk.

A per-format area/Fmax table on U55C is a deliverable of Phase 1.

---

## 10. Phased plan (feature branch)

Each phase is a substantial, end-to-end-testable increment; commit only at full
completion of the feature, tested against the SimX oracle.

- **Phase 0 — parameterize subunits.** `VX_fdivsqrt_unit` and `VX_fncp_unit`
  gain `(EXP_BITS, MAN_BITS)` parameters; `VX_fcvt_unit` becomes the N-format
  `FMTS[]` engine (§5.3). F32 instantiation is bit-identical to today
  (regression: existing F32 suite unchanged).
- **Phase 1 — native FP64 in one backend (DSP).** `FLEN`-parameterized routing +
  F64 instances under `generate if (FLEN == 64)`; `rv32ud`/`rv64ud` green on DSP
  via SimX↔RTL diff; U55C Fmax/area table.
- **Phase 2 — native FP64 in STD.** Same for the ASIC backend; honor the
  FDIV==FSQRT latency constraint.
- **Phase 3 — config + RV32D.** Flip `VX_config.toml` so native backends carry
  FLEN=64; expose RV32D on the 32-bit build; FPNEW becomes opt-in. Full
  three-backend FP64 regression via XRT.

---

## 11. Open questions

- Separate-vs-merged is decided per op class (§4.3); confirm against U55C area
  once Phase 1 numbers exist — merge more aggressively if DSP-bound.
- Should F32-on-F64-hardware ever be allowed (single merged FMA) to save DSPs on
  area-critical builds, accepting an F32 latency hit? Default: no (FMA stays
  separate), revisit only if DSP budget forces it.
- RV32D ABI/calling-convention coverage in the toolchain — confirm `llvm-vortex`
  emits `.d` ops for XLEN=32/FLEN=64 targets.

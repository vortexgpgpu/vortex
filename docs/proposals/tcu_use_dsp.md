# `VX_CFG_TCU_USE_DSP` — DSP-backed TCU FEDP multipliers

Target device: Xilinx Alveo U55C (`xcu55c-fsvh2892-2L-e`), `core_clock` = 300 MHz.
Companion to [`tcu_fedp_dsp_direct.md`](tcu_fedp_dsp_direct.md) (motivation,
synthesis evidence, architectural analysis). This document is the concrete
implementation spec for the single config flag `VX_CFG_TCU_USE_DSP`.

---

## 1. What the flag does

`VX_CFG_TCU_USE_DSP` is a **boolean target-aware switch** that maps the TCU
FEDP's mantissa and integer multipliers onto `DSP48E2` slices instead of
LUT-fabric Wallace trees. It is *orthogonal* to the FEDP backend selector
`VX_CFG_TCU_TYPE`; it modifies the multiply primitive *inside* the pure-RTL
fused backend (`TFR`).

The name deliberately mirrors the FPU's existing `VX_CFG_USE_DSP`
(`hw/rtl/fpu/VX_fpu_define.vh`), which drives `VX_fma_unit`'s `USE_DSP`
parameter — `VX_CFG_TCU_USE_DSP` is its TCU sibling. It is *not* an `_ENABLE`
feature flag (those toggle ISA/format support); it is a backend-implementation
switch, hence the `USE_DSP` naming. The TCU currently has no equivalent — its
only "DSP" path is the vendor-IP wrapper, which is a different (and inferior)
thing entirely.

### 1.1 Naming: `VX_CFG_TCU_USE_DSP` vs `VX_CFG_TCU_TYPE="DSP"`

These are **not** the same and the distinction must be kept sharp:

| symbol | meaning | mechanism |
|---|---|---|
| `VX_CFG_TCU_TYPE = "DSP"` | select the **vendor FP-IP wrapper** backend (`VX_tcu_fedp_dsp.sv`) | `xil_fmul`/`xil_fadd` IP, one full FP32 op per lane/node |
| `VX_CFG_TCU_USE_DSP` | within the **fused TFR** backend, infer mantissa/int multiplies on DSP48E2 | `(* use_dsp="yes" *)` registered multiply |

As established in the companion doc, the wrapper (`TYPE=DSP`) is strictly worse
than TFR on area, DSP count, latency, and format coverage. The end-state this
flag enables is `VX_CFG_TCU_TYPE="TFR"` + `VX_CFG_TCU_USE_DSP=true` becoming
the FPGA default, **replacing** `TYPE="DSP"` (see §7, Deprecation).

---

## 2. Configuration plumbing

### 2.1 `VX_config.toml` — `[tcu]` section

Add in the `[tcu]` block alongside the other `VX_CFG_TCU_*` settings
(after line 240):

```toml
# Map FEDP mantissa/integer multipliers onto FPGA DSP48E2 slices (inferred
# multiply + use_dsp hint). FPGA-only optimization; relieves LUT/routing
# pressure on large configs at the cost of DSP slices and +2 FEDP cycles.
# Ignored by the vendor-IP wrapper backend (VX_CFG_TCU_TYPE="DSP").
VX_CFG_TCU_USE_DSP = false
```

Default `false` is the conservative starting point. After validation (§6), the
recommended production default mirrors `VX_CFG_USE_DSP`: **FPGA-on, ASIC-off**.

### 2.2 Target-aware fallback (mirror `VX_fpu_define.vh`)

Because ASIC flows (yosys/synopsys) must never see a `use_dsp` path, provide the
same auto-default guard the FPU uses. Add to `hw/rtl/tcu/VX_tcu_pkg.sv`'s include
preamble (or a small `VX_tcu_define.vh` if preferred):

```systemverilog
// This is the only target-aware TCU switch; the datapath takes it as a plain
// parameter (USE_DSP) and stays portable. Explicit VX_config.toml value wins.
`ifndef VX_CFG_TCU_USE_DSP
`ifdef VIVADO
`define VX_CFG_TCU_USE_DSP 1
`elsif QUARTUS
`define VX_CFG_TCU_USE_DSP 1
`else
`define VX_CFG_TCU_USE_DSP 0
`endif
`endif
```

(Note: the toml boolean and the macro guard must agree on representation — the
config generator emits `-DVX_CFG_TCU_USE_DSP=1`/`0`, and the `ifndef` guard
only fires when the flag is absent, e.g. ASIC/standalone RTL.)

---

## 3. RTL changes

Thread a single `USE_DSP` parameter from the core to the leaf multipliers,
exactly as `VX_fpu_std`/`VX_fpu_dsp` thread `VX_CFG_USE_DSP` into `VX_fma_unit`.

### 3.1 Parameter threading

```
VX_tcu_core
  └─ VX_tcu_fedp_tfr        (new param: USE_DSP = `VX_CFG_TCU_USE_DSP`)
       └─ VX_tcu_tfr_shared_mul   (USE_DSP)
            ├─ VX_tcu_tfr_mul_f16  (USE_DSP)   ← primary target
            ├─ VX_tcu_tfr_mul_f8   (USE_DSP)   ← optional / packed
            ├─ VX_tcu_tfr_mul_i8   (USE_DSP)   ← packed
            └─ VX_tcu_tfr_mul_i4   (USE_DSP)   ← packed
```

`VX_tcu_core.sv` already computes `FEDP_LATENCY` per backend and passes it as
`.LATENCY(FEDP_LATENCY)`; we make it `USE_DSP`-aware (see §4).

### 3.2 Leaf multiplier gating (`VX_tcu_tfr_mul_f16`)

Replace the unconditional `VX_wallace_mul` (lines 220–228) with a gated form,
following `VX_fma_unit`'s `g_mul_dsp` branch:

```systemverilog
wire [21:0] man_prod;
if (USE_DSP) begin : g_mul_dsp
    // 11x11 fits one DSP48E2 (27x18); registers pack AREG/MREG/PREG.
    (* use_dsp = "yes" *) wire [21:0] dsp_prod = 22'(ma_sel) * 22'(mb_sel);
    VX_pipe_register #(.DATAW(22), .DEPTH(MUL_DSP_REGS)) pm (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in(dsp_prod), .data_out(man_prod));
end else begin : g_mul_wallace
    VX_wallace_mul #(.N(11), .CPA_KS(!`FORCE_BUILTIN_ADDER(11*2))) wtmul (
        .a(ma_sel), .b(mb_sel), .p(man_prod));
end
```

(The mul modules are currently combinational — `clk` is already a port but
mostly `UNUSED`. The DSP branch makes the stage sequential, which is why
`MUL_LATENCY` grows; see §4.)

### 3.3 Per-format policy

| format | significand | DSP fit | policy |
|---|---:|---|---|
| FP16 / TF32 | 11×11 → 22b | 1 DSP/lane, clean | **enable** (phase 1) |
| BF16 | 8×8 → 16b | 1 DSP/lane | **enable** (phase 1) |
| FP8 | 8×8 (4×4 mant) | too small alone | pack 2/DSP or leave LUT (phase 2) |
| FP4 | 4×4 | too small | leave in LUT |
| INT8 | 8×8 | 2 MACs/DSP via 27-bit port packing | enable (phase 2) |
| INT4 | 4×4 | 4/DSP packed | optional |

Phase 1 gates `USE_DSP` to the FP16/BF16/TF32 path only (`mul_f16`), where the
1-DSP-per-lane mapping is unambiguous and the payoff is largest. The other leaf
modules keep their Wallace path until packing is implemented, so `USE_DSP=1`
never wastes a whole DSP on a 4×4 product.

---

## 4. Latency handling (the critical detail)

A registered DSP multiply needs input+MREG+PREG registers to reach Fmax, so the
multiply stage grows from 1 to **2–3 cycles**. This must be threaded
consistently in *two* places that both currently hardcode `1`:

1. **`VX_tcu_fedp_tfr.sv`** localparam (line 56):
   ```systemverilog
   localparam MUL_LATENCY = USE_DSP ? `TCU_MUL_DSP_LATENCY : 1; // 2–3 when DSP
   ```
   `TOTAL_LATENCY`, `S1_IDX..S4_IDX`, and the `vld_pipe_r`/`req_pipe_r` shift
   registers all derive from `MUL_LATENCY`, so they scale automatically. The
   existing `STATIC_ASSERT(LATENCY == TOTAL_LATENCY)` enforces consistency.

2. **`VX_tcu_core.sv`** TFR latency block (lines 68–73):
   ```systemverilog
   localparam FMUL_LATENCY = `VX_CFG_TCU_USE_DSP ? `TCU_MUL_DSP_LATENCY : 1;
   ```
   This feeds `FEDP_LATENCY` → `.LATENCY(FEDP_LATENCY)` and `MDATA_QUEUE_DEPTH`.
   If the two computations disagree, the `STATIC_ASSERT` in the FEDP fires at
   elaboration — a built-in safety net.

Throughput is unchanged (II = 1); only result latency grows by 1–2 cycles. The
scoreboard/`PIPE_LATENCY`/`MDATA_QUEUE_DEPTH` accounting is already derived from
`FEDP_LATENCY`, so no manual fix-ups are needed beyond the two localparams.

Pick `TCU_MUL_DSP_LATENCY = 3` (input reg + MREG + PREG) as the safe default for
300 MHz; `2` may suffice if the operand register is absorbed upstream — decide
from post-route timing, not a priori.

---

## 5. Simulation & verification model

- **SimX (functional + cycle):** the numerical result is **bit-identical** — the
  mantissa product `m_a*m_b` is the same integer whether computed in LUTs or a
  DSP. No change to the SimX FEDP math. The **cycle model** must track the
  latency bump for SimX↔RTL parity (see `project_simx_rtl_parity`): the SimX TCU
  latency constant must read the same `VX_CFG_TCU_USE_DSP` to add the +1–2
  cycles. This is the only SimX-visible effect.
- **rtlsim / Verilator:** the inferred `*` operator simulates natively and the
  `(* use_dsp *)` attribute is ignored by Verilator, so rtlsim validates
  functional bit-equality **without** any DPI shim — unlike the vendor-IP
  wrapper, which needs `DSP_TEST`/`dpi_fmul`. This is a real verification
  advantage of this approach over `TYPE=DSP`.
- **Bit-exact oracle:** the `USE_DSP=0` (Wallace) build is the golden reference;
  `USE_DSP=1` must match it exactly across all enabled formats.

---

## 6. Validation plan

1. **Unit (rtlsim first):** `hw/unittest/tcu_fedp` (`fedp.py`/`fedp.h`) with
   `VX_CFG_TCU_USE_DSP=1`, fp16/bf16/tf32, bit-exact vs the Wallace build.
2. **End-to-end:** `tests/regression/sgemm_tcu*` (incl. `_sp`, `_wg`, `_wgmma`)
   on rtlsim, then on the U55C via XRT.
3. **Timing/area DUTs** (`hw/syn/xilinx/dut/tcu`, unique `PREFIX` per run):
   re-run `tfr_fp16_nt32` and the failing `tfr_sp_nt32` with the flag on.
   Acceptance:
   - WNS ≥ 0 at 300 MHz (today `tfr_sp_nt32` = −2.388, mul stage, 68 % routing);
   - `post_synth_util.rpt` confirms the FP16 products **landed in DSP** (the
     `use_dsp` hint can silently fall back to LUTs if registers aren't packed);
   - net LUT reduction and DSP count within budget (≈ `TCK` DSP/FEDP).
4. Defer all synthesis until rtlsim is green.

Expected per-FEDP movement (from companion-doc data): LUTs down substantially
(remove ~`TCK` Wallace 11×11 trees + routing), DSP up to ~`TCK`/FEDP, latency
+1–2 cycles, and — the goal — `tfr_sp_nt32` closing 300 MHz by relocating its
route-bound multiply stage out of fabric.

---

## 7. Rollout & deprecation

1. **Phase 1:** land `VX_CFG_TCU_USE_DSP` (default `false`), gated to
   FP16/BF16/TF32 in `mul_f16`. Validate per §6.
2. **Phase 2:** extend to INT8/FP8 with DSP packing (2 MACs/DSP); keep FP4 in
   LUT.
3. **Phase 3:** flip the FPGA default on (target-aware, like `VX_CFG_USE_DSP`)
   once `tfr_sp_nt32` and `tfr_fp16_nt32` close timing with it.
4. **Phase 4 (deprecate the wrapper):** with `TFR + VX_CFG_TCU_USE_DSP`
   dominating `TYPE="DSP"` on every axis *and* covering all formats, retire
   `VX_CFG_TCU_TYPE="DSP"` and `hw/rtl/tcu/dsp/VX_tcu_fedp_dsp.sv` rather than
   maintaining two DSP backends. The toml `VX_CFG_TCU_TYPE` FPGA default
   (`'DSP' if not ASIC`) becomes `'TFR'`, with `VX_CFG_TCU_USE_DSP` carrying
   the DSP mapping.

---

## 8. Risks

- **`use_dsp` is a hint.** Must verify post-synth placement; otherwise the flag
  silently does nothing (still functionally correct, just no benefit).
- **DSP-column congestion.** ~`TCK`×(FEDPs) DSPs pulled from fabric create their
  own DSP-to-fabric routing; may need SLR-aware floorplanning at scale
  (`slr_partitioning_proposal.md`).
- **Latency-constant drift.** The two `MUL_LATENCY`/`FMUL_LATENCY` computations
  (§4) must stay in lockstep; the FEDP `STATIC_ASSERT` is the guardrail.
- **Small-format waste.** Never enable the 1-DSP path for FP4 (4×4); gate
  per-format (§3.3).

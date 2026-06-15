# RTU RTL — FPGA Efficiency Refactor

**Scope:** a design-efficiency review of the RTU RTL ([hw/rtl/rtu/](../../hw/rtl/rtu/))
targeting the **Alveo U55C @ 300 MHz**, and a plan to (1) close the current
timing miss and (2) cut the LUT/FF footprint by rebalancing onto the idle hard
blocks (DSP → BlockRAM → LUTRAM, in that priority order). Two levers: a
**feature-reduced FP datapath** for the geometry PEs (the RTU needs none of the
full IEEE-754 machinery the ISA FPU carries), and **moving the per-context state
off flip-flops onto BlockRAM** so it stays flat in fabric as the context count
grows.
**Status:** **Code-complete (P0–P6)** on `prism` — FP-datapath optimizations
(P0–P4, commits `f32f5a92`..`49e575f3`), per-context-state → BRAM (P5,
`49d3abf2`), and the P6 BRAM-read timing restage, all in
[VX_rtu_scheduler.sv](../../hw/rtl/rtu/VX_rtu_scheduler.sv) / `VX_fma_unit.sv`.
Functionally validated (rtlsim raytracing 18/18, bit-identical). **Timing sign-off
is the only open verification** — P5 rebalanced onto BRAM (0 → 20) but moved the
binding path to the BRAM read (WNS −0.388); P6 restages it, post-P6 WNS being
measured (`build_w4_p6`). Deferred (optional / architectural, not core): `DSP_SEED`
reciprocal, `VX_CFG_RTU_NUM_CTX`. Per-phase results in §5; open items in §8.
**Tree:** `~/dev/vortex_v3/prism_v3` (branch `prism`).
**Date:** 2026-06-12 (proposal); as-built update 2026-06-14.
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
in the FP core (split/parameterize the ACC stage, or reduce mantissa width),
**not** in scheduler pipelining — which reorders the rollout (§5).

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

## 3. `NUM_CTX` scaling — the dominant axis (confirmed)

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
   *(Deferred — this needs a ray→context mapping change when `NUM_CTX ≠
   NUM_LANES`, which is architectural; tracked as an open item in §8.1.)*
2. **Back the per-context state with BRAM/LUTRAM** so it stays **flat in fabric**
   as `NUM_CTX` grows: a 1R1W RAM addressed by context id replaces the
   FF-array + `NUM_CTX`:1 mux. `stack` (`NUM_CTX·512b`) and `f_buf`
   (`NUM_CTX·1024b`) map naturally to BRAM18; the narrower records to LUTRAM.
   BRAM is at **0%** today, so the headroom is the entire device. *(Implemented
   for `stack`, `f_buf`, and `inst_xform`; see P5 in §5.)*

---

## 4. Optimization summary & estimated gains

Deltas vs the 72,428-LUT BVH4 baseline. Gains are engineering estimates pending
re-synthesis; **direction and hard-block tradeoffs are firm**.

Two columns of impact: **Δ@4** is the saving at today's `NUM_CTX=4`; **scaling**
is how the saving grows with context count — the deciding factor now that
production raises `NUM_CTX` (§3).

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

## 5. Phasing

Ordering reflects the confirmed `NUM_CTX` scaling (§3): the FP-core timing relief
first, then the shared-PE area wins, then the context-state → BRAM move
(load-bearing at production), with the residual timing close deferred to last.

1. **P0 — FP-core timing relief.** ~~Register the BVH offset/address arithmetic
   and de-fan `curoff_q`.~~ **Already done** in commit `2b4bc1e6` (`structaddr_q`).
   The residual **−0.096 ns** miss (post-physopt, `build_w4_pm`) is now the
   **`VX_fma_unit` ALIGN→ACC carry chain** (§1.2), shared by `box/tri/xform` PEs.
   The correct P0 is to **break that stage** in the FP core: either (a) add a
   pipeline register splitting ACC (add vs LZC), exposed as an `ACC_SPLIT`
   parameter on the shared `VX_fma_unit` (1 extra cycle; touches the shared unit,
   so the GPU FPU must stay bit-identical and re-close), or (b) reduce RTU
   mantissa width (shorter carry chain) — which folds into P4. **(a) is the
   minimal behaviour-preserving fix.** Validate `tests/raytracing` bit-identical.

   **Measured (commit `f32f5a92`, DUT `build_w4_accsplit` vs `build_w4_pm`):**
   approach (a) implemented as `ACC_SPLIT` in [VX_fma_unit.sv](../../hw/rtl/fpu/VX_fma_unit.sv),
   funded by one MUL stage so **total LATENCY is unchanged** (0 extra cycles —
   better than the "1 extra cycle" estimate above; no downstream PE/FPU contract
   change). **WNS −0.096 → +0.024 ns: 300 MHz closes** (at routed, no physopt
   needed). Bit-identical on rtlsim (dogfood full FP matrix + RTU box/tri/xform
   smoke). **DSP 146→146, BRAM/LUTRAM 0→0 (the prioritized resources unchanged).**
   Cost: **LUT +13% (98.0K→110.8K), FF +34% (57.3K→76.9K)** — the FMA datapath
   (`box_pe` FF 7.5K→12.0K confirms it), of which ~72 FF/FMA is the new register
   and the remainder is `-global_retiming on` spending registers to reach the
   tighter path. This area is the *same FMA datapath* phases 1–2 reduce, so it is
   recovered there; the increase is the expected cost of closing timing first
   (§5 sequencing). Caveat: `build_w4_pm` is two commits behind, so the LUT/FF
   delta is not a perfectly clean attribution (though `box_pe` is untouched by
   those commits).
2. **P2 — dedicated F32 FMA latency + retire dead knobs (Opt 1 + Opt 6).**
   Give the geometry PEs a dedicated, F32-sized `LATENCY_FMA` (the ISA FPU keeps
   the wider F64-sized `VX_CFG_LATENCY_FMA`) so the PE side-band delay lines
   shrink; retire the unreferenced `RTU_BOX_PE`/`RTU_TRI_PE`/… pkg knobs. **Must
   stay bit-identical** at the FPU and re-verified at the RTU latency change.

   **Measured (commit `c3da767c`):** a dedicated **`RTU_LATENCY_FMA = 9`**
   ([VX_rtu_pkg.sv](../../hw/rtl/rtu/VX_rtu_pkg.sv) L54 — the F32 floor that keeps
   the mantissa multiply on DSP, i.e. `MUL_LATENCY ≥ LATENCY_IMUL` with ACC_SPLIT
   active) now drives every PE side-band delay line (≈ halves depth vs the
   F64-sized 16); the dead `RTU_BOX_PE` / `RTU_TRI_PE` / `RTU_NODE_LATENCY` /
   `RTU_TRI_LATENCY` pkg localparams were retired (**Opt 6**; the `VX_CFG_*`
   knobs remain for SimX). rtlsim raytracing 20/24 (4 pre-existing
   unimplemented-feature fails, identical on parent).
3. **P3 — reciprocal consolidation (Opt 4).** Share the 3 setup dividers; expose
   the optional `DSP_SEED` (seed-table + DSP-refinement) reciprocal backend.

   **Measured (commit `7d0ed7fc`):** the three parallel `1/dir` setup dividers
   collapsed to **one** `VX_fdivsqrt_unit`, time-multiplexed across axes
   (sequenced by per-context `setup_axis`; a traversal context stays selected for
   its whole setup span, so this is bit-equivalent). Trades two dividers
   (~5K LUT, 0 DSP) for two extra setup passes per ray/instance — negligible vs
   traversal. **`DSP_SEED` was not implemented** — the reciprocal stays LUT-NR
   and DSP stays flat (honoring the hard-block priority; §8.1 open item). rtlsim
   raytracing 20/24, TLAS/instanced object-ray setup paths covered.
4. **P4 — capability reduction (Opt 2).** Flip RTU FP params to FTZ / RTZ /
   no-fflags / no-except; fuse dot/cross normalization. **Behavioural** —
   gated on the SimX-oracle validation in §6.

   **Measured (commits `c91232a0` + `fc8d14f1`, deepened `94aeaa17`→`49e575f3`):**
   `fdot3` / `fcross3` fused onto a **single** normalize+round in new
   [VX_rtu_fmac3.sv](../../hw/rtl/rtu/VX_rtu_fmac3.sv) (form lane products → align
   to the common max exponent → sum in extended precision → normalize+round
   **once**; `fdot3` feeds 3 terms, `fcross3` two-per-axis with the second
   negated). Geometry inputs are finite, so inf/NaN handling is dropped and
   subnormals are FTZ'd — the single rounding is ≥ as accurate as the FMA chain.
   Separately, **`EN_EXCEPT`** was added to `VX_fma_unit` (default 1 → GPU FPU
   bit-identical) and set to **0** on the provably-finite RTU FMAs (box_pe
   origin/dequant, tri_pe edge `v−v`); the box_pe slab FMAs and tri_pe `*invDet`
   FMAs keep `=1` (they see ±inf on axis-aligned / degenerate-det rays). FTZ was
   unnecessary — the FMA already flushes subnormal results via the underflow
   path. The fused MAC initially packed align+add+LZC+normalize+round into two
   cycles (post-impl **−1.88 ns**); split to **7 stages**
   (shift|negate|add|abs|lzc|normalize|round) to give each carry-heavy op its own
   cycle, latency held at 3×/2×`LATENCY_FMA` so the tri/xform PEs need no change.
   rtlsim raytracing 20/24, t/u/v within the tests' 1e-4 tolerance,
   geometry_index exact, full FP matrix bit-identical (FPU unchanged). **The §6
   SimX precision gate has not yet been run** (§8.1 open item).
5. **P5 — per-context state → BRAM (Opt 3, §3).** Convert the per-context
   flip-flop arrays — read through a wide `NUM_CTX`:1 select mux — to
   context-id-addressed RAMs so the working set scales onto BlockRAM as
   `NUM_CTX` grows instead of fabric + mux. **Behaviour-preserving**, and the
   highest-leverage change at production `NUM_CTX`.

   **Implemented (this session, [VX_rtu_scheduler.sv](../../hw/rtl/rtu/VX_rtu_scheduler.sv)):**
   - The **short stacks** moved first (commit `5fd48483`) to a 1R1W `VX_dp_ram`
     keyed by `{context, depth}` (write-first RDW so a same-cycle push and read
     return the freshly pushed top-of-stack).
   - The **node image (`f_buf`)** and the **instance transform (`inst_xform`)**
     now live in [VX_dp_ram](../../hw/rtl/libs/VX_dp_ram.sv) instances
     (`g_fbuf_ram` — one per fetched line slot — and `xform_ram`), `OUT_REG=1`.
     The RAM read is **issued in SELECT** (`raddr = sel`) and its registered
     output is consumed in EXEC, replacing the snapshot registers (`fbuf_q` /
     `xform_q`) at the same one-cycle latency. Because the entries are wide
     (≥16b), `VX_dp_ram`'s `FORCE_BRAM` heuristic fires **even at `NUM_CTX=4`**,
     so they infer BlockRAM (one of the project's idle hard blocks) rather than
     LUTRAM, and stay flat in fabric as the context count grows. The wide
     `NUM_CTX`:1 `f_buf`/`inst_xform` select muxes are removed.

   The `VX_CFG_RTU_NUM_CTX` decoupling knob (§3 action 1) is **not** included —
   it needs a ray→context mapping change and is tracked separately (§8.1). The
   ray/hit/obj records remain FF (narrow; LUTRAM/FF is appropriate there).
   *Validation: rtlsim `tests/raytracing` (BVH4/6 + instanced) must stay
   bit-identical; pending its DUT synthesis pass.*
6. **P6 — residual timing close.** P5's `build_w4_bram` synth showed the BlockRAM
   read had become the binding path: the registered `f_buf` BRAM output fed the
   1024-bit `f_aligned` byte-align shift into `cur_off` (WNS −0.388 ns, every top
   endpoint sourced from `g_fbuf_ram`). Fix: split the micro-step
   **SELECT → ALIGN → EXEC** and register the BRAM node image into a fabric flop
   (`fbuf_q`) in ALIGN, so the shift again starts from a fast FF rather than the
   slower BRAM output. Costs one extra pipeline phase per micro-step (throughput,
   hidden by the context pool); the BRAM win is fully retained.

   **Implemented ([VX_rtu_scheduler.sv](../../hw/rtl/rtu/VX_rtu_scheduler.sv)):**
   rtlsim raytracing **18/18 PASS** (BVH4/6 + instanced), bit-identical. Timing
   re-measured in DUT `build_w4_p6` (synth in flight at time of writing).

Each phase is independently synthesizable and testable; P0–P3 and P5 are
behaviour-preserving, P4 requires the §6 gate. **Synthesize each phase at both
`NUM_CTX=4` and a production-representative count** so the scaling claims in §3/§4
are verified, not assumed.

---

## 6. Validation

- **Functional:** full [tests/raytracing](../../tests/raytracing/) suite via
  **xrt** (the RTL coverage path), plus rtlsim for fast iteration. P0–P2 and P5
  must be **bit-identical**.
- **Precision (P4):** use the SimX model as the goal-reference oracle — build the
  FTZ/RTZ behaviour into SimX first, confirm image parity within agreed ULP/SSIM
  tolerance, then diff SimX↔RTL trace dumps to localize any divergence before
  committing the RTL precision change.
- **Timing/area:** re-run the `build/hw/syn/xilinx/dut/rtu` DUT (w0/w4/w6) after
  each phase; track WNS and the per-block LUT/FF/DSP/BRAM table from §1. Each
  synth/FPGA build uses a unique `PREFIX` (own build tree + log); never two hw
  runs on the U55C at once.

---

## 7. Risks & open questions

- **Precision (P4).** FTZ + RTZ shift intersection results in the ULPs;
  watertightness near shared edges is the thing to watch. The SimX gate (§6)
  exists to catch it before RTL.
- **`NUM_CTX` decoupling (§3).** Adds a config dimension to the test matrix; keep
  the default `= NUM_THREADS` so existing configs are unchanged. The
  ray→context mapping for `NUM_CTX ≠ NUM_LANES` is the unresolved design point.
- **DSP-priority tension.** The optional `DSP_SEED` reciprocal *adds* DSP (the
  project's scarcest hard block). It is opt-in; at today's 1.15% DSP it is
  affordable, but the LUT-NR default keeps DSP flat for DSP-constrained full-GPU
  configs.
- **P5 BRAM width.** At `NUM_CTX=4` the wide (512b/1024b) RAMs map to BlockRAM at
  shallow depth — efficient in *fabric* (the goal) but spends several RAMB tiles
  per memory; the depth fills out and the per-context cost amortizes at
  production `NUM_CTX`. Confirm the RAMB count is acceptable at the target
  context count.
- **Resolved:** the production GPU config **does** raise `NUM_THREADS`/`NUM_CTX`
  above 4. Opt 3 (per-context state → BRAM/LUTRAM) is therefore promoted to P5
  and treated as load-bearing, not a cleanup (§3/§4). **Open follow-up:**
  the specific target `NUM_CTX` value(s) to synthesize against in the §5 phase
  gates.

---

## 8. As-built results (2026-06-14)

Optimizations P0–P4 are implemented on `prism`
(`f32f5a92` → `c3da767c` → `7d0ed7fc` → `5fd48483` → `c91232a0` → `fc8d14f1` →
`94aeaa17` → `49e575f3`) and synthesized; P5 (`f_buf`/`inst_xform` → BRAM) is
coded this session and **pending its synthesis pass**. Per-phase timing/area for
P0–P4 is **not** cleanly separable: P1–P4 were stacked into one DUT
(`build_w4_all`) and then the `VX_rtu_fmac3` depth was iterated (`build_w4_final`
2-stage **−1.88 ns** → `build_w4_final2` 5-stage **−0.375** → `build_w4_signoff`
7-stage **−0.028**). The per-commit rtlsim results in §5 are the clean per-phase
functional evidence; the table below is the combined post-physopt/postRoute
outcome for **P0–P4** (BVH4 + TLAS, `core_clock = 3.333 ns`, `xcu55c-2L`).

| Metric | `build_w4_pm` (baseline, §1.2) | `build_w4_signoff` (P0–P4) | Δ |
|---|---|---|---|
| **WNS** | −0.096 ns | **−0.028 ns** — still **VIOLATED** | +0.068 ns |
| LUT | 98,035 (7.52%) | 80,313 (6.16%) | **−17,722 (−18.1%)** |
| FF | 57,291 (2.20%) | 59,674 (2.29%) | +2,383 (+4.2%) |
| SRL | 6,819 (1.13%) | 2,009 (0.33%) | **−4,810 (−70.5%)** |
| DSP | 146 (1.62%) | 146 (1.62%) | **0 (flat)** |
| BRAM | 0 | 0 | 0 |
| LUTRAM | 0 | 36 | +36 (short stacks) |

Per block (LUT / FF, `pm → signoff`):

| Block | LUT | FF | Note |
|---|---|---|---|
| `tri_pe` | 42,320 → 36,163 (**−14.5%**) | 18,222 → 20,839 | fmac3 fusion + `EN_EXCEPT` |
| `xform_pe` | 22,373 → 16,032 (**−28.3%**) | 9,647 → 7,966 | fmac3 fusion + `EN_EXCEPT` |
| `box_pe` | 17,671 → 19,012 (+7.6%) | 7,485 → 13,775 | no dot/cross to fuse; +FF from ACC_SPLIT + retiming |

**Outcome vs the §4 projection.** Direction holds: **LUT −18%** and **SRL −70%**
land as projected, **DSP held exactly flat (146)** and **BRAM untouched** —
honoring the hard-block priority. Two deltas from the projection: **FF rose
~+4%** instead of the projected −51% (timing was bought with registers —
ACC_SPLIT + `global_retiming` + the 7-stage `VX_rtu_fmac3`), and the projected
state→BRAM move (P5) had not yet landed at signoff (only the 36-LUTRAM short
stacks). P5 is expected to move `f_buf`/`inst_xform` onto BRAM and recover FF.
(Note the §4 table is keyed to the older 72.4K non-TLAS BVH4 baseline; this table
is keyed to the §1.2 TLAS-active `build_w4_pm`, the apples-to-apples
representative build.)

### 8.1 Open items

1. **P5 synthesized** (DUT `build_w4_bram` vs `build_w4_signoff`): **BRAM 0 → 20
   (0.99%)**, **LUT 80.3K → 73.9K (−7.9%)**, FF 59.7K → 58.7K, DSP 146 flat — the
   hard-block rebalancing landed. Timing regressed (WNS −0.028 → −0.388 ns), since
   addressed by P6. Still to synthesize at a production `NUM_CTX`, not just 4.
2. **`VX_CFG_RTU_NUM_CTX` decoupling not done.** Contexts are still bound to lane
   count. Decoupling needs a ray→context mapping (queue rays into a context pool
   when `NUM_CTX ≠ NUM_LANES`) — an architectural change, not a parameter.
3. **`DSP_SEED` reciprocal** not implemented (LUT-NR retained; DSP flat).
4. **§6 precision gate not run.** The P4 FTZ / no-except change passed rtlsim at
   the tests' 1e-4 tolerance, but the SimX-oracle image-parity gate (§6) has not
   been executed.
5. **P6 implemented; timing re-measure in flight.** P5 moved the binding path
   onto the `f_buf` BlockRAM read (WNS −0.388 ns). P6 (the SELECT→ALIGN→EXEC
   restage, §5) registers the BRAM output into a fabric flop ahead of the
   `f_aligned` shift; rtlsim is bit-identical (18/18). The post-P6 WNS is being
   measured in DUT `build_w4_p6`; that result is the timing sign-off for the
   refactor.

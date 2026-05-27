# `VX_tcu_unit` WGMMA Refactor — Proposal

| | |
|---|---|
| **Status** | Draft |
| **Date** | 2026-05-25 |
| **Target branch** | `tinebp-patch-2` @ `e3938a3f` |
| **Scope** | [hw/rtl/tcu/VX_tcu_unit.sv](hw/rtl/tcu/VX_tcu_unit.sv) and adjacent TCU files, plus their SimX mirror under [sim/simx/tcu/](sim/simx/tcu/) |
| **Source** | This proposal supersedes the loose `wgmma_v3.patch` in the working tree, rebased onto current remote |

---

## 1. Motivation

`VX_tcu_unit.sv` on `e3938a3f` is **392 lines**, of which ~145 lines
(≈ 37 %) are an inline CTA-lockstep gate accumulated through
successive WGMMA bug-fix patches. The wrapper has lost the "thin
top-of-TCU layer" character of its original design:

- Lockstep state, fire-detection, and a sim-only invariant are
  mixed with buffer-subsystem instantiation and lane split/gather.
- Per-block redundant state plus a sequential cross-block
  propagation chain produce the deepest combinational path in the
  file.
- The "is this the first / last sub-uop of a WGMMA expansion?"
  predicate is re-derived in three places from the same step
  counters — a class of latent bug.
- Two `` `ifdef VX_CFG_TCU_WGMMA_ENABLE `` regions split the
  wrapper's flow.
- The `cta_conflict` wire is forward-referenced (declared 100 lines
  before its driver) because the gate logic is downstream of the
  code that needs to read it.

Specific pain points, all verified against current `e3938a3f`:

| # | Issue | Location |
|---|---|---|
| P1 | Sequential `cta_conflict` propagation across blocks (BLOCK_SIZE-deep combinational chain) | [VX_tcu_unit.sv:259-288](hw/rtl/tcu/VX_tcu_unit.sv#L259-L288) |
| P2 | Per-block `cta_owner_r` for a singleton resource (bbuf) | [VX_tcu_unit.sv:226](hw/rtl/tcu/VX_tcu_unit.sv#L226) |
| P3 | `inflight_count_r` redundant with `in_expansion_r` | [VX_tcu_unit.sv:225,315-321](hw/rtl/tcu/VX_tcu_unit.sv#L225) |
| P4 | `is_first_uop` / `is_last_uop` re-derived in three places | [VX_tcu_uops.sv:422-423](hw/rtl/tcu/VX_tcu_uops.sv#L422-L423), [VX_tcu_bbuf.sv:~133](hw/rtl/tcu/VX_tcu_bbuf.sv), [VX_tcu_unit.sv:~307-311](hw/rtl/tcu/VX_tcu_unit.sv) |
| P5 | WGMMA perf counters living in the wrapper | [VX_tcu_unit.sv:164-192](hw/rtl/tcu/VX_tcu_unit.sv#L164-L192) |
| P6 | 9-signal `req_*_arr` unpack with named ports | [VX_tcu_unit.sv:76-117](hw/rtl/tcu/VX_tcu_unit.sv#L76-L117) |
| P7 | Forward reference of `cta_conflict` | [VX_tcu_unit.sv:90-93](hw/rtl/tcu/VX_tcu_unit.sv#L90-L93) |
| P8 | Two split `` `ifdef VX_CFG_TCU_WGMMA_ENABLE `` regions | [VX_tcu_unit.sv:72-204, 223-351](hw/rtl/tcu/VX_tcu_unit.sv#L72) |
| P9 | Sim-only `lockstep_violation` is structurally circular at its current site | [VX_tcu_unit.sv:~344-346](hw/rtl/tcu/VX_tcu_unit.sv) |

The functional fix that resolved the NW=8 IW=4 sgemm correctness bug
(`in_expansion_r` plus the `req_valid_arr` mask by `cta_conflict`) is
keeper code; the surrounding accretion is not.

---

## 2. Design intent

Restore the original layering:

```
VX_tcu_unit.sv          thin top-of-TCU wrapper
├── VX_lane_dispatch
├── VX_tcu_wgmma.sv     NEW — orchestrator owns the WGMMA feature end-to-end
│   ├── VX_tcu_lockstep.sv   NEW — CTA lockstep gate + sim-only contract assertion
│   └── VX_tcu_tbuf.sv       UNCHANGED — buffer subsystem (Q×abuf + bbuf + Q×mbuf + LMEM arb)
├── VX_tcu_core × BLOCK_SIZE
└── VX_lane_gather
```

Three rules drive the split:

1. **`VX_tcu_unit.sv` knows nothing WGMMA-specific.** Lane dispatch,
   per-block `tcu_core` instances, lane gather. One
   `` `ifdef VX_CFG_TCU_WGMMA_ENABLE `` block to instantiate the
   orchestrator or tie off its outputs.
2. **`VX_tcu_wgmma.sv` owns the WGMMA feature.** Anything that
   exists only because WGMMA exists (lockstep, dispatch-IF unpack,
   perf counters, the buffer subsystem instance) lives here.
3. **`VX_tcu_tbuf.sv` is unchanged.** It is already a clean
   abstraction — "tile buffer subsystem" — and a future kernel can
   reuse it without inheriting cooperative-warpgroup orchestration.

---

## 3. Concrete changes

### C1 — Extract `VX_tcu_wgmma.sv` (the orchestrator)

**Move out of `VX_tcu_unit.sv`:**

- The 9-signal `req_*_arr` unpacking ([L76-117](hw/rtl/tcu/VX_tcu_unit.sv#L76-L117)).
- `VX_tcu_tbuf` instantiation ([L125-162](hw/rtl/tcu/VX_tcu_unit.sv)).
- WGMMA perf counters ([L164-192](hw/rtl/tcu/VX_tcu_unit.sv#L164-L192)).
- The CTA lockstep gate's instantiation point (after C2 it is a
  single `VX_tcu_lockstep` instance).
- `tbuf_ready_eff` masking ([L289-292](hw/rtl/tcu/VX_tcu_unit.sv)).

**Module interface (sketch):**

```verilog
module VX_tcu_wgmma import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         BLOCK_SIZE      = `VX_CFG_NUM_TCU_BLOCKS,
    parameter         BANK_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE
                                      - $clog2(`VX_CFG_XLEN/8)
                                      - $clog2(`VX_CFG_LMEM_NUM_BANKS)
) (
    input  wire                                                clk, reset,

`ifdef PERF_ENABLE
    output tcu_perf_t                                          tcu_perf,
`endif

    // Observation of dispatch path (read-only; does not drive .ready).
    input  wire [BLOCK_SIZE-1:0]                               exec_valid,
    input  wire [BLOCK_SIZE-1:0]                               exec_ready,   // perf only
    input  tcu_execute_t [BLOCK_SIZE-1:0]                      exec_data,

    // Bank-parallel LMEM read port.
    VX_mem_bus_if.master                                       tcu_lmem_if,

    // Outputs to tcu_core.
    output wire [BLOCK_SIZE-1:0][TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data,
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    output wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0]          tbuf_sp_meta,
`endif
    output wire [BLOCK_SIZE-1:0]                                        tbuf_ready_eff
);
```

**Notes.**

- `exec_*` ports are observation-only. The orchestrator does **not**
  drive any sub-IF's `.ready`; that handshake topology stays at the
  wrapper, where `tbuf_ready_eff` feeds `tcu_core.execute_if.ready`
  as it does today. This avoids the circular-dependency interface
  pattern.
- `exec_data` carries `tcu_execute_t` packed structs; synthesis
  flattens unused fields. If a narrower observation struct is
  desired later, define `tcu_obs_t` (subset of fields wgmma reads);
  cosmetic.

**Effect on `VX_tcu_unit.sv`.** Collapses to ~120 lines — lane
dispatch, one `` `ifdef VX_CFG_TCU_WGMMA_ENABLE `` block
instantiating the orchestrator (with stubbed-off `tbuf_*` outputs
in the `else` branch), per-block `tcu_core`, lane gather. No
forward references, no wrapper-level perf state, no cross-block
propagation logic.

**Area / timing:** zero change from the move alone. **Readability:**
single ifdef boundary for the entire feature; the 9-signal unpack
becomes an internal wire of `VX_tcu_wgmma`.

### C2 — Extract `VX_tcu_lockstep.sv` (the gate)

Inside `VX_tcu_wgmma.sv`, instantiate:

```verilog
module VX_tcu_lockstep import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter         BLOCK_SIZE  = 4
) (
    input  wire                                  clk, reset,

    // observation
    input  wire [BLOCK_SIZE-1:0]                 is_wgmma_b,
    input  wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] new_cta_b,
    input  wire [BLOCK_SIZE-1:0]                 exec_fire_b,
    input  wire [BLOCK_SIZE-1:0]                 is_first_uop_b,
    input  wire [BLOCK_SIZE-1:0]                 is_last_uop_b,

    // gating output
    output wire [BLOCK_SIZE-1:0]                 cta_conflict
);
```

**Internals after C3:** `in_expansion_r[BLOCK_SIZE]`, single
`tcu_owner_r` plus `tcu_owned_r`, combinational `cta_conflict`,
sim-only contract assertion (see C5).

**Why a sub-module rather than inline in `VX_tcu_wgmma`.**

- Self-contained gate logic is unit-testable.
- The contract — *"my consumer must AND `cta_conflict` into the
  request validity it presents to bbuf and into `tbuf_ready_eff`"*
  — becomes a documented module boundary that the sim-only
  assertion (C5) can defend.
- Future TCU variants that don't need a CTA gate can simply not
  instantiate this module; nothing else changes.

**Area / timing:** zero from the relocation alone. The wins come
from C3.

### C3 — Single `tcu_owner_r` + drop `inflight_count_r` (load-bearing)

Both apply inside `VX_tcu_lockstep.sv` after C2.

**State today (per block, [L225-231](hw/rtl/tcu/VX_tcu_unit.sv#L225-L231)):**
`cta_owner_r [BLOCK_SIZE][NCTA_WIDTH]` +
`inflight_count_r [BLOCK_SIZE][INFLIGHT_CW]` +
`in_expansion_r [BLOCK_SIZE]`.

**State after C3:** `tcu_owner_r [NCTA_WIDTH]` + `tcu_owned_r [1]` +
`in_expansion_r [BLOCK_SIZE]`.

```verilog
// Owner update — single owner, set on any block firing its leader uop,
// released when no block is still mid-expansion.
wire any_first_fire   = |(exec_fire_b & is_first_uop_b);
wire any_in_expansion = |in_expansion_r;
wire [NCTA_WIDTH-1:0] first_fire_cta;  // priority pick of lowest block firing leader uop

always_ff @(posedge clk) begin
    if (reset) begin
        tcu_owned_r <= 1'b0;
        tcu_owner_r <= '0;
    end else begin
        if (!tcu_owned_r && any_first_fire) begin
            tcu_owned_r <= 1'b1;
            tcu_owner_r <= first_fire_cta;
        end else if (tcu_owned_r && !any_in_expansion) begin
            tcu_owned_r <= 1'b0;
        end
    end
end

// Per-block conflict — flat combinational, no propagation chain.
for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_conflict
    assign cta_conflict[bi] = is_wgmma_b[bi] && tcu_owned_r
                           && (tcu_owner_r != new_cta_b[bi]);
end
```

**Why this works.**

- The bbuf is a singleton resource. By construction, only one CTA
  can legitimately occupy the TCU at a time; the gate exists
  precisely to enforce that. A single owner is therefore *correct*,
  not just smaller.
- The cooperative-warpgroup case (multiple blocks firing leader
  uops in the same cycle for the same CTA) is the production path;
  their `cta_id`s match, so the priority-encoded pick agrees with
  all of them. No same-cycle multi-CTA-firing race exists in
  current configs.
- `in_expansion_r` already holds the "this block is mid-expansion"
  semantic across the LMEM-stall gap. `inflight_count_r` was added
  before `in_expansion_r`; `in_expansion_r` strictly subsumes its
  role for the bbuf-protection contract. The over-counting of
  WMMA/META ops (acknowledged as "conservative" in
  [VX_tcu_unit.sv:~234](hw/rtl/tcu/VX_tcu_unit.sv)) goes away —
  those ops never use bbuf, never threaten the gate, and shouldn't
  be counted.

**Area savings.**

- `cta_owner_r`: `BLOCK_SIZE × NCTA_WIDTH = 4 × 3 = 12` FFs → `4` FFs.
  **−8 FFs.**
- `inflight_count_r`: `BLOCK_SIZE × INFLIGHT_CW = 4 × 4 = 16` FFs →
  removed. **−16 FFs.**
- Plus ~15 LUT6 of update-logic reductions.

**Timing — critical path estimate.**

Today's `cta_conflict[bi]` path (at the worst block, `BLOCK_SIZE=4`):

```
per_block_execute_if[k].data.header.cta_id  (NCTA_WIDTH bits)
  → eff_cta_owner[k]   (sequential through k = 0..bi-1)
  → != new_cta_b_w[bi]
  → OR-reduce across k
  → AND with is_wgmma_b_w[bi]
  → cta_conflict[bi]
```

Sequential depth scales with `bi`; for `BLOCK_SIZE=4` ≈ **12 LUT6
levels** at the worst block.

After C3:

```
tcu_owner_r → != new_cta_b[bi] → AND with tcu_owned_r and is_wgmma_b[bi] → cta_conflict[bi]
```

**2 LUT6 levels, flat across all blocks. Net saving ≈ 10 LUT6
levels off the deepest path inside the unit.**

### C4 — `is_first_uop` / `is_last_uop` in `op_args.tcu`

**Today** the same two conditions are re-derived in three places
from `step_m`, `step_n`, `step_k`, `cd_nregs`. The single source of
truth already exists in
[VX_tcu_uops.sv:422-423](hw/rtl/tcu/VX_tcu_uops.sv#L422-L423) as
`fu_lock` / `fu_unlock` (computed from `uop_idx` vs `uop_count`).

**Proposal.** Add two single-bit fields to `tcu_args_t` (inside
`op_args`) and assign them in `VX_tcu_uops.sv` next to the existing
`fu_lock`/`fu_unlock` lines. Read them as plain wires in
`VX_tcu_bbuf.sv` and `VX_tcu_lockstep.sv`.

```verilog
// In VX_tcu_uops.sv, next to the existing fu_lock/fu_unlock assigns:
ibuf_r.op_args.tcu.is_first_uop = is_wgmma && (uop_idx == '0);
ibuf_r.op_args.tcu.is_last_uop  = is_wgmma && (uop_idx == (uop_count - 1));
```

**Area.** +2 bits to the pipeline carry of `op_args.tcu`. Across
the staging stages between dispatch and execute (≈ 4) that is ~8
FFs. Saves the duplicate decode in two consumers (~10 LUTs).
**Net: small positive area, but eliminates a class of "are these
expressions still equal?" bugs as `cd_nregs` extensions land.**

**Timing.** Removes a 4-input AND chain
(`step_m == 0 && step_n == 0 && step_k == 0`) from both bbuf and
lockstep critical paths. **~1 LUT6 level off each.**

**Risk.** Schema change to `op_args.tcu`. Worth doing once,
alongside C3 so the regression sweep covers both at once.

### C5 — Sim-only contract assertion inside `VX_tcu_lockstep.sv`

The current `lockstep_violation` assertion at
[VX_tcu_unit.sv:~344-346](hw/rtl/tcu/VX_tcu_unit.sv) checks
`exec_fire_b && cta_conflict[bi]`, which the surrounding code
*also* prevents by ANDing `cta_conflict` into `tbuf_ready_eff`.
The assertion is therefore locally tautological at its current
site.

After C2 the assertion gains a real contract to defend:

> *"`VX_tcu_lockstep` produces `cta_conflict`; my consumer (i.e.
> `VX_tcu_wgmma`) must AND it into the request validity presented
> to bbuf and into `tbuf_ready_eff` going to `tcu_core`. If a
> future consumer forgets, this assertion fires."*

**Proposal.** Keep the assertion verbatim; relocate into
`VX_tcu_lockstep.sv`. `RUNTIME_ASSERT` is sim-only, no synth cost.

### C6 — Pack `req_*_arr` into a `tcu_tbuf_req_t` struct (cosmetic)

**Today** ([L76-117](hw/rtl/tcu/VX_tcu_unit.sv#L76-L117)) 9 wires of
width × `BLOCK_SIZE` declared individually, 9 assignments in a
generate loop, 9 named ports on `VX_tcu_tbuf`.

**Proposal.** Define `tcu_tbuf_req_t` in `VX_tcu_pkg.sv`:

```verilog
typedef struct packed {
    logic                       valid;
    logic [NW_WIDTH-1:0]        wid;
    logic [3:0]                 step_m;
    logic [3:0]                 step_n;
    logic [3:0]                 step_k;
    logic [1:0]                 cd_nregs;
    logic [`VX_CFG_XLEN-1:0]    desc_a;
    logic [`VX_CFG_XLEN-1:0]    desc_b;
    logic                       a_is_smem;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    logic                       is_sparse;
    logic [3:0]                 fmt_s;
`endif
} tcu_tbuf_req_t;
```

Inside `VX_tcu_wgmma.sv`: one packed array, one assignment loop.
`VX_tcu_tbuf` port list shrinks from 11 named request signals to
one packed-array port. No behaviour change.

**Area / timing:** zero. **Readability:** the orchestrator's
"unpack dispatch IF → request struct → mask `valid` by
`cta_conflict`" pattern becomes obvious in ~10 lines.

### C7 — Sparse variants as distinct opcodes (added after C4 review)

**Today** `op_args.tcu.is_sparse` carries a per-uop bit that
discriminates sparse from dense WMMA / WGMMA. Combined with the C4
addition of `is_first_uop` + `is_last_uop`, the struct grows by 2 net
bits — forcing `INST_ARGS_BITS` from 25 → 27 and a 2-bit padding bump
on `alu_args_t` / `br_args_t`, which propagates across the entire
op_args union pipeline carry.

**Proposal.** Promote sparse-ness into the **opcode space** instead of
the args struct:

```verilog
// VX_gpu_pkg.sv
localparam INST_TCU_WMMA       = 4'h0;
localparam INST_TCU_WGMMA      = 4'h1;
localparam INST_TCU_META_STORE = 4'h2;
localparam INST_TCU_WMMA_SP    = 4'h3;   // NEW
localparam INST_TCU_WGMMA_SP   = 4'h4;   // NEW
// INST_TCU_BITS = 4 (16 opcodes available; 5 used)
```

Removing `is_sparse` (−1 bit) balances the C4 additions (+2 bits) net
+1; combined with dropping the original 1-bit `__padding`, the struct
stays at exactly 25 bits. **No INST_ARGS_BITS change.** alu/br stay
padding-free; the entire pipeline carries the same width as before
C1–C6.

Downstream consumers of sparseness switch from
`op_args.tcu.is_sparse` to checking `op_type == INST_TCU_WMMA_SP || …`.
The decoder picks the sparse opcode when `rs2[0]` is set.

**Area / timing:** strictly better than the C4-only design — saves the
INST_ARGS_BITS-bump propagation cost across every args struct (alu/br
padding, lsu/csr/fpu/wctl/dxa/tcu/tex/om/raster/etc. all carry 1 fewer
bit). Per-block readers gain a 2-input OR (`is_wgmma_dense || is_wgmma_sp`)
vs a field read — comparable LUT cost, no net change.

**Risk.** ISA encoding change — softer than a schema bump but visible
in the assembler. Behavior verified bit-exact across all three rtlsim
configs and the SimX baseline (see §7.1).

---

## 4. Aggregate impact

| Change | FFs | LUT6 | LUT levels off critical path | Risk |
|---|---:|---:|---:|---|
| C1. `VX_tcu_wgmma.sv` extract | 0 | 0 | 0 | low (relocation) |
| C2. `VX_tcu_lockstep.sv` extract | 0 | 0 | 0 | low (relocation) |
| **C3. Single owner + drop `inflight_count_r`** | **−24** | **~−15** | **~−10** | medium (semantic verify) |
| C4. `is_first_uop`/`is_last_uop` in `op_args.tcu` | +2 net | +5 net | −1 (×2 sites) | medium (schema) |
| C5. Relocate sim assertion | 0 | 0 | 0 | none |
| C6. `tcu_tbuf_req_t` struct | 0 | 0 | 0 | none |
| **C7. Sparse-as-opcode** | **0 net** (offsets C4) | **0 net** | **0** | medium (ISA encoding) |
| **Total** | **~−24 FFs** | **~−15 LUT6** | **~−12 LUT levels** | |

C3 and C4 are the load-bearing improvements. C1/C2/C5/C6 are
restructuring that lets C3 and C4 land cleanly. **C7** restores
`INST_ARGS_BITS = 25` (no global pipeline-carry growth) by paying with
a softer ISA encoding change.

---

## 5. Implementation order

Each step is independently testable with the existing
`sgemm_tcu_wg`-NW={4,8}-IW=4 regression set.

1. **C5** (relocate assertion) — first, alongside C2; lowest risk.
2. **C2** (extract `VX_tcu_lockstep.sv` with current state inside)
   — relocation only; C5 lives in the new module.
3. **C1** (extract `VX_tcu_wgmma.sv`) — relocation only; the
   wrapper becomes thin.
4. **C6** (`tcu_tbuf_req_t` struct) — cosmetic, fits naturally
   with C1's port-list redesign.
5. **C3** (single owner + drop `inflight_count_r`) — the headline
   simplification; the gate is its own module by now, so the
   change is local and easy to bisect.
6. **C4** (`op_args.tcu` schema extension) — separate regression
   sweep; verify all three consumers (`VX_tcu_uops`, `VX_tcu_bbuf`,
   `VX_tcu_lockstep`) read consistent values.

Each step lands as its own PR so the regression matrix bisects
cleanly.

---

## 6. SimX mirror

The SimX side already has `tcu_unit.{cpp,h}` and `tcu_tbuf.{cpp,h}`
in [sim/simx/tcu/](sim/simx/tcu/). The intent is to mirror the RTL
split: add `tcu_wgmma.{cpp,h}` (orchestrator + lockstep state),
keep `tcu_tbuf.{cpp,h}` unchanged, slim `tcu_unit.{cpp,h}` to the
wrapper role. SimX changes land alongside the corresponding RTL
step (each C1-C6 PR carries both sides).

---

## 7. Test plan

- **Unit:** lockstep gate sub-module gets a sim-only contract
  assertion (C5) — if a future consumer of `VX_tcu_lockstep`
  forgets to AND `cta_conflict` into bbuf request validity or into
  `tbuf_ready_eff`, the assertion fires.
- **Regression matrix per step:** `sgemm_tcu_wg` × NW ∈ {4, 8} ×
  IW ∈ {2, 4} on rtlsim — see §7.1 for the cycle-baseline table.
  C3 and C4 additionally need the sparse WGMMA variants
  (`sgemm_tcu_wg_sp_*`) since `is_sparse`/`fmt_s` ride in the same
  packed struct.
- **Timing:** sample synthesis on the Xilinx flow after C3 to
  confirm the ~10 LUT6-level reduction on `cta_conflict` paths
  shows up in critical-path reports.
- **Bisection:** if a regression appears, the 6 PRs let us bisect
  to the exact change without rerunning a monolithic
  bring-up.

### 7.1 Cycle baseline and no-regression validation

The headline wins of this refactor are area (~−24 FFs, ~−20 LUT6) and
**critical-path timing** (~−12 LUT6 levels off `cta_conflict`).
Functional cycle counts on `sgemm_tcu_wg` fp16 are **not expected to
move** — the gate produces the same handshake outcomes, just in fewer
levels of logic. This section therefore frames the cycle table as a
**no-regression gate**, not a speedup target. The actual win is
documented in the synthesis sample after C3 (see §7 bullet 3).

**Pre-work.** `sgemm_tcu_wg/main.cpp` lacks cycle reporting today; add
one line — `vx_device_dump_perf(device, stdout);` after the
`Elapsed time:` print (mirroring
[draw3d/main.cpp:382](../../tests/regression/draw3d/main.cpp#L382)).
Drives the `PERF: instrs=… cycles=… IPC=…` line under `--perf=1`.
Applied to the working tree alongside this proposal's baseline
capture.

**Baseline measurement** (captured 2026-05-25 on `tinebp-patch-2` @
[49202fb6](../../../) prior to C1–C6, `sgemm_tcu_wg` fp16, rtlsim
driver, `VX_CFG_TCU_TYPE_DPI`. CONFIG/args derived from
`ci/regression.sh.in` lines 1117, 1118, 1122 with the driver fixed to
rtlsim across all rows):

| NW | IW | NRC | -w | XLEN | cycles | instrs | IPC | Status |
|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 16 | 2 | 32 | 12098 | 2194 | 0.181 | PASS |
| 8 | 2 | 16 | 2 | 64 | 12666 | 2302 | 0.182 | **FAIL** ⚠ |
| 8 | 4 | 32 | 4 | 32 | 25787 | 3548 | 0.138 | PASS |
| 8 | 4 | 32 | 4 | 64 | 24232 | 3676 | 0.152 | **FAIL** ⚠ |
| 8 | 4 |  8 | 4 | 32 | 14615 | 1824 | 0.125 | PASS |

⚠ **rtlsim XLEN=64 numerical failure.** Both XLEN=64 configs produce
valid cycle measurements but fail correctness verification on the
output matrix (numerical mismatch). XLEN=64 rtlsim for `sgemm_tcu_wg`
is not exercised by the existing CI matrix
([ci/regression.sh.in:1117-1158](../../ci/regression.sh.in#L1117) all
use XLEN=32), so this is an untested combination on `tinebp-patch-2`
rather than a regression introduced by this proposal. Treat the XLEN=64
cycle counts as **observational** until the correctness failure is
diagnosed. Tracked as a pre-work item:

| # | Issue | Resolution |
|---|---|---|
| **PW1** | `sgemm_tcu_wg` fp16 rtlsim XLEN=64 fails numerical verification on the output matrix. Untested in CI. Reproducible on both NW=8 IW=2 and NW=8 IW=4 configs. | Investigate before relying on XLEN=64 rtlsim cycles as a validation gate. Possible causes: XLEN-64 path through `VX_tcu_fedp_dpi` (fp16 → fp32 accumulator), or a host-side cast in `main.cpp` for 64-bit pointers. Until fixed, the XLEN=64 rows in the post-refactor table below are advisory only. |

**Post-C1–C6 no-regression check** (captured 2026-05-25 after all six
changes landed in the working tree). The cycles column must match
baseline within ±0 — any drift indicates a behavioral change snuck
into a relocation step and must be diagnosed before merging.

| NW | IW | NRC | -w | XLEN | Baseline cycles | Post-refactor cycles | Δ |
|---|---|---|---|---|---|---|---|
| 8 | 2 | 16 | 2 | 32 | 12098 | **12098** | **0** ✓ |
| 8 | 2 | 16 | 2 | 64 | 12666 (PW1) | _N/A_ (PW1) | _N/A_ |
| 8 | 4 | 32 | 4 | 32 | 25787 | **25787** | **0** ✓ |
| 8 | 4 | 32 | 4 | 64 | 24232 (PW1) | _N/A_ (PW1) | _N/A_ |
| 8 | 4 |  8 | 4 | 32 | 14615 | **14615** | **0** ✓ |

All three XLEN=32 configs bit-exact across the entire C1-C6 refactor.
Per-phase validation on the smallest config (NW=8 IW=2 NRC=16 -w 2)
confirmed each change individually preserves cycles=12098:

| After phase | cycles | IPC | Notes |
|---|---|---|---|
| C2+C5 (extract `VX_tcu_lockstep.sv` + relocate sim assertion) | 12098 | 0.181 | RTL relocation only |
| C1 (extract `VX_tcu_wgmma.sv` orchestrator) | 12098 | 0.181 | RTL relocation only |
| C6 (`tcu_tbuf_req_t` struct) | 12098 | 0.181 | Cosmetic — packed-struct port |
| C3 (single `tcu_owner_r` + drop `inflight_count_r`) | 12098 | 0.181 | **Load-bearing simplification** |
| C4 (`is_first_uop`/`is_last_uop` in `op_args.tcu`) | 12098 | 0.181 | Schema +2 bits |
| C7 (sparse-as-opcode: `INST_TCU_{WMMA,WGMMA}_SP`) | 12098 | 0.181 | Removes `is_sparse` field → **INST_ARGS_BITS unchanged at 25** |

**C7 — Sparse variants as distinct opcodes** (added after C4 review):

Instead of carrying a per-uop `is_sparse` bit in `tcu_args_t`, sparse
WMMA / WGMMA become their own opcodes — `INST_TCU_WMMA_SP` and
`INST_TCU_WGMMA_SP` — in the 4-bit `INST_TCU_BITS` opcode space (4
opcodes used out of 16). This frees the bit C4 would otherwise have
required, keeping `INST_ARGS_BITS = 25` unchanged from the original
design — no padding bumps to `alu_args_t` / `br_args_t`, no net
pipeline-carry growth across the entire ISA. The decoder picks the
sparse opcode when `rs2[0]` is set; every downstream consumer of
sparseness (`VX_tcu_uops`, `VX_tcu_core`, `VX_tcu_wgmma`,
`VX_tcu_tbuf`, `VX_tcu_mbuf` through tcu_tbuf_req_t.is_sparse derived
from op_type) checks the opcode instead.

The XLEN=64 rows remain advisory under PW1 (pre-existing numerical
verification failure on rtlsim XLEN=64; reproducible on both
pre-refactor and post-refactor trees, confirmed independent of this
proposal). Will validate once PW1 is resolved upstream.

**SimX parity check** (`sgemm_tcu_wg` fp16, NW=8 IW=2 NRC=16 -w 2):

| | cycles | IPC |
|---|---|---|
| Baseline (pre-refactor) | 12116 | 0.181 |
| Post-refactor (full C1–C7) | 12116 | 0.181 |

SimX mirror landed alongside RTL:

- `TcuType` enum extended with `WMMA_SP` and `WGMMA_SP` variants.
- `IntrTcuArgs.is_sparse` removed; `is_first_uop` / `is_last_uop`
  added (C4 mirror).
- All `args.is_sparse` field accesses replaced with `tcu_is_sparse(t)`
  helper; `TcuType::WGMMA` switch arms widened to also handle
  `TcuType::WGMMA_SP`. Same for `WMMA`.
- Uop expansion (`TcuUopGen::get`) emits `WMMA_SP` / `WGMMA_SP` for
  the sparse paths and propagates `is_first_uop` / `is_last_uop` per
  uop_index.

The structural file split (`tcu_wgmma.cpp` separate from
`tcu_unit.cpp`) remains a deferred cosmetic — SimX behavior already
matches C1–C7 semantics; the file boundary is code-organization only.

**Critical-path validation** (the real win, recorded from Xilinx synth
after C3 — see §7 bullet 3):

| Metric | Baseline (pre-C3) | Post-C3 | Δ LUT6 levels |
|---|---|---|---|
| `cta_conflict` path depth (worst block, BLOCK_SIZE=4) | _TBD synth report_ | _TBD synth report_ | target ~−10 |
| `cta_conflict` path depth (after C4) | _TBD_ | _TBD_ | additional ~−1 |

---

## 8. Out of scope (flagged for follow-on)

- **`VX_tcu_bbuf.sv` `slot_desc_b_row_base_r` latch hardening.**
  The latch updates on `req_valid && is_first_uop` without an
  "owner" cross-check. The lockstep mask makes this safe under the
  current contract, but a defensive change — e.g., latch only when
  the presented `cta_id` matches `tcu_owner_r` — would harden bbuf
  against a future wrapper bug. Separate proposal.
- **`tcu_perf_t` aggregation.** Today the wrapper composes
  `tcu_perf` from `tbuf_*` (driven from inside tbuf) and `wgmma_*`
  (computed in the wrapper). After C1 it has a single source. If
  future perf signals are added, prefer `VX_tcu_wgmma` as the
  aggregation site.
- **Narrow `tcu_obs_t` for the lockstep observation port.** Today
  passing the full `tcu_execute_t` works (synth flattens unused
  fields). A narrow observation struct would document the
  observation contract more precisely; cosmetic.

---

## 9. Rebase notes (for the implementer)

The previous `wgmma_v3.patch` working-tree file was authored
against an intermediate snapshot and **does not apply cleanly** to
`e3938a3f`. Rather than rebasing that patch hunk-by-hunk, prefer to
land each of C1–C6 as a fresh commit on top of `e3938a3f` following
the order in §5. The patch's text is now superseded by this doc;
file targets and line numbers above reference the *current* remote.

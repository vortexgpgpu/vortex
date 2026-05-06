# AMO (RVA) RTL v3 — Proposal

**Date:** 2026-05-02
**Status:** v1.1 — Phases 0/1/3-stub/6 in tree; **Phases 2/4/5 deferred** (see §5).
**Owners:** RTL team
**Related:**
[amo_simx_v3_proposal.md](amo_simx_v3_proposal.md) (companion SimX proposal),
`feedback_vortex_proposals_location`,
`project_amo_simx_landed`.

**Scaffold status (2026-05-02):**
- ✅ Phase 0 — `IS_LLC` parameter wired through `VX_cache.sv` /
  `VX_cache_bank.sv` / `VX_cache_wrap.sv` / `VX_cache_cluster.sv`
  with `IS_LLC=1` set at the right level in `VX_socket.sv` /
  `VX_cluster.sv` / `Vortex.sv`. `amo_req_t` typedef + `INST_AMO_*`
  enum + `HART_ID_BITS` localparam in `VX_gpu_pkg.sv`. Build asserts
  for write-through-strictly-above-LLC in `Vortex.sv`.
- ✅ Phase 1 — `VX_amo_alu.sv` + `VX_amo_unit.sv` standalone modules
  in `hw/rtl/cache/`. Not yet instantiated; lint-checked once
  Phase 2 wires them in.
- ✅ Phase 3 — non-LLC passthrough is documented as v2 work; no
  build-time assertion (would over-trigger for icache).
- ✅ Phase 6 — this proposal status section.
- ⏸️ Phase 2 — bank pipeline integration (S1 commit mux, MSHR rule,
  reservation hooks) requires synchronized changes to the most
  complex module in the codebase plus iterative verilator + rtlsim
  cycles. Deferred to a focused session.
- ⏸️ Phase 4 — decoder + LSU integration. Adding 11 new `INST_LSU_*`
  enum values rippled through every consumer (op_type bits widening,
  scoreboard, dispatch, formatter), plus per-lane sideband packing
  in `VX_lsu_slice.sv`. Deferred together with Phase 2.
- ⏸️ Phase 5 — rtlsim conformance against `rv32ua-p-*` and
  `tests/regression/amo`. Cannot proceed without 2 + 4.

---

## Summary

Add RTL support for the RISC-V `A` extension (LR, SC, AMO\*) to Vortex
under the existing `EXT_A_ENABLE` build flag, mirroring the SimX
implementation that already landed (commit `5077074`). Atomicity for
global addresses is provided by a per-bank `VX_amo_unit` instantiated
**only at the last-level cache (LLC)** of whatever hierarchy Vortex is
configured with. Caches above the LLC operate in *AMO-passthrough*
mode. AMOs targeting Shared (LMEM) or IO regions are out of scope (§6).

---

## 1. Constraints (load-bearing)

The RTL must match the contract the SimX implementation already
exposes — same coverage tests pass on both, same external signals
where applicable. Three specific load-bearing rules:

1. **Single-cycle commit on hit.** AMO commit happens in the same
   bank cycle as a write-hit. No phantom RMW cycle, no separate
   pipeline stage. Latency to `rd` = `tag_lookup + (miss ? miss_penalty : 0)`,
   identical to a load. (Mirrors SimX §3.7.)
2. **Reservation table at LLC bank only.** Caches above the LLC do
   not hold reservations and must not synthesize the table. The build
   guarantees this with parameter gating, not runtime configuration.
3. **Mirror SimX semantics, not its software shape.** SimX runs as
   sequential C++; RTL is fully pipelined with valid/ready handshakes.
   The reservation table that SimX models as a `std::vector<Reservation>`
   becomes a register-file CAM with `AMO_RS_SIZE` entries; SimX's pure
   `amo_compute()` becomes a combinational ALU; SimX's
   `amo_unit_.invalidate()` becomes a one-cycle CAM-walk side-effect on
   every committed write.

---

## 2. Design decisions

The matching SimX proposal §2 enumerates eight design decisions that
shape the AMO engine. All of them carry over to RTL — the
hierarchy-placement, reservation policy, channel extensions, MSHR
integration, address-type scope, no-coalescing rule, reservation-
invalidation policy, and SC return-value encoding are *behaviors*
the RTL must reproduce. The table below highlights the RTL-specific
realizations.

| # | SimX choice | RTL realization |
|---|---|---|
| D1 | `Q = AMO_RS_SIZE` LRU reservation table per LLC bank | `VX_amo_unit` instantiates a register-file CAM, `Q` rows × `{valid, hart_id, line_addr, lru}`. LRU is a small monotonic counter, comparator selects the oldest valid entry. |
| D2 | AMO engine at the LLC only, parameter-gated | One `VX_amo_unit` instance inside `VX_cache_bank.sv` under `if (IS_LLC) generate ... endgenerate`. Non-LLC banks contain no AMO logic — synthesizes to zero gates. |
| D3 | `MemReq` widens with `{op, width, rhs, hart_id}` | `VX_mem_bus_if` (and `VX_lsu_mem_if`) gain the same sideband bits. `req.rw` is unconditionally 0 for AMO requests, including SC. |
| D4 | RMW commits in the same cycle as a write-hit | Bank-pipe commit stage gets one extra mux input: the AMO ALU result feeds `line_data_in` instead of `core_req.data` when `req.amo.valid`. No new stage. |
| D5 | Global only; Shared (LMEM) AMOs hard-asserted | RTL: synth-time `assert` in `VX_local_mem_switch.sv` (analogous to SimX §3.13). IO AMOs likewise. |
| D6 | Same-line AMO lanes must serialize, not coalesce | LSU↔dcache crossbar must not merge AMO lanes. Implementation detail in `VX_lsu_mem_arb.sv` / `VX_mem_coalescer.sv`. |
| D7 | Reservations broken by every committed write reaching the LLC tag array | One-cycle CAM-walk inside `VX_amo_unit` triggered on every bank-tag-array write commit (write-hit and writethrough write-miss); WB evictions do not trigger. |
| D8 | SC returns 0/1 in `rd` via the load formatter | LR/SC payload re-uses the existing read-response data path. SC commit synthesizes a 1-bit value at the AMO byte offset; the LSU read-formatter sign-extends as for a load-word. |

---

## 3. Target architecture

### 3.1 Cache-hierarchy placement

Same rules as SimX §3.1. The Vortex RTL already instantiates L1/L2/L3
as instances of the same `VX_cache` module with different parameters
([VX_cache.sv:16-76](../../hw/rtl/cache/VX_cache.sv#L16),
instantiated at [VX_socket.sv:111-187](../../hw/rtl/core/VX_socket.sv#L111)
for L1, [VX_cluster.sv](../../hw/rtl/core/VX_cluster.sv) for L2,
[VX_processor.sv](../../hw/rtl/VX_processor.sv) for L3), so the LLC
selection is purely a parameter wiring decision at the instantiator.

#### 3.1.1 New parameter: `IS_LLC`

`VX_cache.sv` and `VX_cache_bank.sv` gain one new parameter:

```systemverilog
parameter IS_LLC = 0
```

defaulting to zero. When `IS_LLC == 1`, the bank synthesizes the
`VX_amo_unit` and the AMO commit mux. When `IS_LLC == 0`, the AMO
sideband is wired through but commit logic is generated empty —
non-LLC banks then operate in passthrough mode for AMO requests
(§3.8).

#### 3.1.2 Where `IS_LLC = 1` is wired

Mirror the SimX `Cache::Config::is_llc` table from §3.1.3, set at
exactly one site per build:

| `EXT_A_ENABLE` | `L3_ENABLE` | `L2_ENABLE` | Site that sets `IS_LLC=1` |
|---|---|---|---|
| 1 | 1 | * | `VX_processor.sv` (L3 ctor) |
| 1 | 0 | 1 | `VX_cluster.sv` (L2 ctor) |
| 1 | 0 | 0 | `VX_socket.sv` (dcache ctor) |
| 0 | * | * | nowhere (parameter stays 0) |

A cache instantiated with `PASSTHRU = 1` (existing
[VX_cache_wrap.sv:110-150](../../hw/rtl/cache/VX_cache_wrap.sv#L110)
gating) bypasses banks entirely; AMO requests fall through to the
next level. The build wiring must guarantee exactly one cache on
the AMO path is `IS_LLC = 1`. A synth-time `assert` at the top
processor scope checks this.

#### 3.1.3 Write-through-strictly-above-LLC invariant

SimX enforces this with C++ `static_assert` ([processor.cpp:82-103](../../sim/simx/processor.cpp#L82)).
RTL enforces it with the `$error` ifdef pattern already used in the
codebase:

```systemverilog
`ifdef EXT_A_ENABLE
  `ifdef L3_ENABLE
    `ifndef DCACHE_WRITETHROUGH
      $error("AMO requires write-through L1 (DCACHE_WRITEBACK=0) when L3 is the LLC");
    `endif
    `ifndef L2_WRITETHROUGH
      $error("AMO requires write-through L2 (L2_WRITEBACK=0) when L3 is the LLC");
    `endif
  `elsif L2_ENABLE
    `ifndef DCACHE_WRITETHROUGH
      $error("AMO requires write-through L1 (DCACHE_WRITEBACK=0) when L2 is the LLC");
    `endif
  `endif
`endif
```

placed near the top of `VX_processor.sv`. The LLC itself is
unconstrained (write-back or write-through, both correct).

### 3.2 Module sketch

```
┌──────────────────────────────────────────────────────────────────┐
│ Core (VX_core.sv → VX_lsu_slice.sv)                              │
│                                                                  │
│  Decode (VX_decode.sv)                                           │
│    Opcode 0x2F (AMO) decoded under `EXT_A_ENABLE`. funct5 →      │
│    INST_LSU_{LR,SC,AMOADD,AMOSWAP,...}. funct3 → width.          │
│    aq/rl bits captured but unused (see §6).                      │
│                                                                  │
│  VX_lsu_slice (existing)                                         │
│    AMO ops dispatched on the same lane as loads. MSHR allocated  │
│    (every AMO returns to rd). Per-lane AMO sideband packed:      │
│    {op, width, rhs, hart_id}. mem_req_rw = 0 for all AMO ops.    │
│                                                                  │
│  VX_local_mem_switch                                             │
│    Asserts on a Shared-typed AMO request (§3.13).                │
│                                                                  │
│  VX_lsu_mem_arb / VX_mem_coalescer                               │
│    AMO lanes do NOT coalesce — same-line lanes pass through as   │
│    separate bank requests. (Mirrors SimX `mem_coalescer.cpp`.)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Cache (VX_cache.sv → VX_cache_bank.sv)                           │
│                                                                  │
│  bank_req_t (implicit struct of port wires) gains:               │
│    amo_valid, amo_op, amo_width, amo_rhs, amo_hart_id            │
│                                                                  │
│  Stage S0 (tag lookup): unchanged.                               │
│  Stage S1 (commit):                                              │
│    is_amo = req.amo_valid                                        │
│    sc_fail = (op==SC) && !amo_unit.check(hart_id, line_addr)     │
│    do_store = (op != LR) && !sc_fail                             │
│                                                                  │
│    when is_amo:                                                  │
│      old_word = line_data[byte_off +: width_bits]                │
│      {new_word, ret_word} = VX_amo_alu(op, width, old_word, rhs) │
│      mux line_data_in = patch(line_data, new_word, byte_off)     │
│      core_rsp_data = build_rsp(op, sc_fail, ret_word, byte_off)  │
│      if (op == LR) amo_unit.reserve(hart_id, line_addr)          │
│      if (op == SC) amo_unit.clear  (hart_id, line_addr)          │
│      if (do_store) amo_unit.invalidate(line_addr,                │
│                                        except = hart_id)         │
│                                                                  │
│  ── IS_LLC == 0 (passthrough) ──                                 │
│    `if (IS_LLC) generate ... endgenerate` skipped. AMO request   │
│    forwarded via the existing IO/bypass egress (same path        │
│    SimX uses, see §3.8 of the SimX proposal). The bank tag       │
│    array is probed for hit-dirty (writeback) / hit-clean         │
│    (invalidate) before forwarding (§3.1.2 of the SimX            │
│    proposal). v1 RTL implementation may stub this when only      │
│    L1-only configs are tested.                                   │
│                                                                  │
│  VX_amo_unit (NEW: hw/rtl/cache/VX_amo_unit.sv)                  │
│    Pure compute kernel + reservation CAM. Combinational          │
│    compute, single-cycle reservation lookup/update.              │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 New files

```
hw/rtl/cache/
  VX_amo_unit.sv       // VX_amo_unit module — compute kernel + reservation CAM
  VX_amo_alu.sv        // pure combinational ALU: amo_compute(op, width, lhs, rhs)
                       // → {new_word, ret_word}
```

`VX_amo_unit` is a pure-RTL module instantiated *inside* a generate
block in `VX_cache_bank.sv` when `IS_LLC == 1`. It has no clock
beyond the bank's; reservation reads are combinational (used in the
same cycle as the tag-lookup result), reservation writes are
flopped on the bank clock.

### 3.4 Channel-type extensions

The widening matches SimX §3.4. In RTL, the changes touch the
package and the interfaces:

```systemverilog
// VX_gpu_pkg.sv — new typedef (only present when EXT_A_ENABLED)
`ifdef EXT_A_ENABLE
typedef struct packed {
  logic                       valid;
  logic [INST_AMO_BITS-1:0]   op;       // {LR, SC, ADD, SWAP, XOR, OR, AND, MIN, MAX, MINU, MAXU}
  logic [1:0]                 width;    // 2 = .W, 3 = .D
  logic [`XLEN-1:0]           rhs;      // rs2 for this lane
  logic [`HART_ID_BITS-1:0]   hart_id;  // make_hart_id(cid, wid, tid)
} amo_req_t;
`endif

// VX_lsu_mem_if.sv — per-lane AMO sideband
`ifdef EXT_A_ENABLE
amo_req_t  req_amo  [NUM_LANES];
`endif

// VX_mem_bus_if.sv — per-request AMO sideband
`ifdef EXT_A_ENABLE
amo_req_t  req_amo;
`endif
```

`req_amo.valid` is the predicate; the rest is don't-care when valid is
zero. `req.rw` is unconditionally 0 for AMO requests, including SC.
The decoder/LSU enforce this.

### 3.5 hart_id width and encoding

Match SimX [types.h:1043](../../sim/simx/types.h#L1043):

```systemverilog
`define HART_ID_BITS  ($clog2(`NUM_CORES) + `LOG2_NUM_WARPS + `LOG2_NUM_THREADS)

function automatic logic [`HART_ID_BITS-1:0] make_hart_id(
  input logic [$clog2(`NUM_CORES)-1:0]    cid,
  input logic [`LOG2_NUM_WARPS-1:0]       wid,
  input logic [`LOG2_NUM_THREADS-1:0]     tid
);
  return {cid, wid, tid};  // packed concatenation, low bits = tid
endfunction
```

Per-lane in `VX_lsu_slice.sv`: `req_amo[lane].hart_id = make_hart_id(CORE_ID, wid, lane)`.

### 3.6 `VX_amo_unit` module interface

```systemverilog
module VX_amo_unit #(
  parameter NUM_RES_ENTRIES = `AMO_RS_SIZE,
  parameter HART_ID_BITS    = `HART_ID_BITS,
  parameter LINE_ADDR_BITS  = `CS_LINE_ADDR_WIDTH
) (
  input  logic                             clk,
  input  logic                             reset,

  // Compute kernel (combinational).
  input  logic [INST_AMO_BITS-1:0]         compute_op,
  input  logic [1:0]                       compute_width,
  input  logic [63:0]                      compute_old,
  input  logic [63:0]                      compute_rhs,
  output logic [63:0]                      compute_new_word,
  output logic [63:0]                      compute_ret_word,

  // Reservation table (single-port read+write per cycle, sufficient
  // since the bank issues at most one commit per cycle).
  input  logic                             res_reserve,    // LR fires
  input  logic                             res_clear,      // SC fires
  input  logic                             res_invalidate, // any other-hart write
  input  logic [HART_ID_BITS-1:0]          res_hart_id,
  input  logic [LINE_ADDR_BITS-1:0]        res_line_addr,
  output logic                             res_check       // for SC outcome
);
```

The compute path is purely combinational (`amo_compute()` from SimX's
`amo/amo_ops.h` → `VX_amo_alu` instance). The reservation table is
`NUM_RES_ENTRIES` registers of `{valid, hart_id, line_addr, lru_ctr}`,
walked in parallel by an LR row-evict picker / SC match-checker.

### 3.7 Bank pipeline integration

`VX_cache_bank.sv` already has a four-source priority arbiter
([VX_cache_bank.sv:201-229](../../hw/rtl/cache/VX_cache_bank.sv#L201))
and a tag-lookup→commit pipe. The integration is local:

- **Stage S0 (input arbitration + tag lookup).** No change apart from
  carrying `bank_req.amo_*` along the pipe.
- **Stage S1 (commit).** Mirrors SimX `commitAmo` ([cache.cpp:641](../../sim/simx/mem/cache.cpp#L641)):
  collect all stall signals, then commit. New mux on the
  line-data-in path selects the AMO-patched line bytes; new mux on
  the core-rsp data path selects the SC outcome / AMO ret_word.
- **MSHR allocation** for AMO miss is identical to a load miss: AMO
  always returns to `rd`, so it must reserve an MSHR slot at S0 even
  though `bank_req.rw == 0`. Existing MSHR plumbing
  ([VX_cache_mshr.sv:45-99](../../hw/rtl/cache/VX_cache_mshr.sv#L45))
  is unchanged.
- **Replay (post-fill).** When the fill arrives and the MSHR replay
  fires, the bank routes through the same S1 commit logic. An AMO
  replay always hits (the fill installed it) — the existing `assert`
  on replay-hit covers this without modification.

### 3.8 Non-LLC bank passthrough

When `IS_LLC == 0`, the bank's AMO commit logic generates empty.
Three behaviors depend on whether the cache above the LLC is
configured `PASSTHRU = 1` (already a wire) or `PASSTHRU = 0`:

- **`PASSTHRU = 1`** (current default for L2/L3 in L1-only builds):
  the AMO request flows through the bypass arbiter without ever
  entering bank logic. Already correct in v1.
- **`PASSTHRU = 0`** (multi-level configs L1+L2 / L1+L2+L3): the
  bank must probe-and-invalidate the line before forwarding. v1 RTL
  implementation **may stub this** with a synth-time `$error` until
  multi-level AMO testing is added. v2 implements §3.8 of the SimX
  proposal: writeback-if-dirty, invalidate, forward via bypass arb,
  no install on response.

The Phase 2 plan (§5) breaks this into a separate phase.

### 3.9 Reservation invalidation

Trigger: every committed write reaching the LLC bank's tag array
fires `amo_unit.invalidate(line_addr, except=hart_id)`. This set
includes (per §3.1.5):

- writethroughs from above (L1 always; L2 when L3 is the LLC),
- AMO RMW commits at the LLC,
- write-back write-hit commits if the LLC itself is configured WB.

LLC's own evictions and writethroughs to DRAM do **not** trigger
invalidation (consistent with RVA: eviction is not a write).

The CAM walk is one combinational fan-out in `VX_amo_unit`: each
entry independently compares `(line_addr, hart_id != except_hart_id)`
and clears `valid` on the next clock if both match.

### 3.10 LSU integration

`VX_lsu_slice.sv` ([hw/rtl/core/VX_lsu_slice.sv](../../hw/rtl/core/VX_lsu_slice.sv))
needs three changes:

1. **Recognize AMO ops.** `INST_LSU_*` enum gains `LR, SC, AMOADD, AMOSWAP, AMOXOR, AMOOR, AMOAND, AMOMIN, AMOMAX, AMOMINU, AMOMAXU` under `EXT_A_ENABLE`. AMO traces are dispatched on the LSU's load path.
2. **MSHR allocation rule.** Treat AMO as load-class for MSHR
   allocation (every AMO returns to `rd`), even though it carries a
   side-effect store.
3. **Pack AMO sideband.** Per active lane, set
   `mem_req.amo = {valid=1, op, width, rhs, hart_id=make_hart_id(CORE_ID, wid, lane)}`
   and force `mem_req.rw = 0`. The width comes from `funct3`.

The response path is unchanged: AMO returns flow through the
existing load formatter (sext/zext at the right width). SC returns
0/1 at the byte offset; the formatter naturally produces 0/1 in `rd`
since both fit in the low byte.

### 3.11 Decoder

`VX_decode.sv` ([hw/rtl/core/VX_decode.sv](../../hw/rtl/core/VX_decode.sv))
gains a case for `Opcode::AMO` (0x2F) under `EXT_A_ENABLE`:

```systemverilog
`ifdef EXT_A_ENABLE
  AMO: begin
    fu_type    = FU_LSU;
    lsu_op     = decode_amo_funct5(funct5);  // → INST_LSU_{LR,SC,AMOADD,...}
    lsu_width  = funct3[1:0];                 // 2 = .W, 3 = .D
    rd_valid   = 1'b1;
    rs1_valid  = 1'b1;
    rs2_valid  = (funct5 != FUNCT5_LR);       // LR has no rs2
    // aq, rl decoded but unused — see §6.
  end
`endif
```

`decode_amo_funct5()` is a small combinational lookup; the funct5
encodings match the existing SimX [decode.cpp:599-622](../../sim/simx/decode.cpp#L599).

### 3.12 LMEM / IO guards

Mirror SimX §3.13 in `VX_local_mem_switch.sv`:

```systemverilog
`ifdef EXT_A_ENABLE
  // synth assertion: any Shared-typed AMO request is a build mistake.
  always_comb begin
    if (req_amo.valid && req_addr_type == ADDR_TYPE_SHARED) begin
      $error("AMO on Shared (LMEM) is unsupported");
    end
  end
`endif
```

The IO bypass arbiter likewise asserts on AMO if it ever sees one.

---

## 4. Why this is correct (against the constraints)

- **Constraint 1 (single-cycle commit).** The AMO ALU is purely
  combinational; the reservation table read is one register-file
  port; both fit in the existing S1 stage budget alongside the
  write-hit data-in mux. The proof is structural — the commit logic
  is identical to a write-hit plus one extra mux input.
- **Constraint 2 (LLC-only).** `IS_LLC` gates the entire `VX_amo_unit`
  inside a `generate` block. Non-LLC banks synthesize zero AMO
  hardware. The build assert (§3.1.2) ensures exactly one cache on
  the AMO path is the LLC.
- **Constraint 3 (mirror SimX).** Every functional decision in §2 has
  a SimX line-citation. The RTL must produce the same observable
  results on the existing `tests/regression/amo` and the riscv-tests
  `rv32ua-p-*` suite. Verification methodology is in §5.

---

## 5. Phased implementation

Each phase compiles, lints, simulates under verilator, and is
independently reviewable. Validate via the existing `build_test32/`
flow (`make -C build_test32/sim/rtlsim`) and the SimX-vs-RTLsim CSV
trace diff (per `reference_csv_trace_debugging`).

### Phase 0 — Config + plumbing (no behavior change)

- Wire `IS_LLC` parameter through `VX_cache.sv` / `VX_cache_bank.sv`
  / `VX_cache_wrap.sv` / `VX_cache_cluster.sv`. Default 0 everywhere.
- Wire `IS_LLC = 1` per the §3.1.2 table at `VX_processor.sv`,
  `VX_cluster.sv`, `VX_socket.sv`.
- Add the build-time write-through-strictly-above-LLC `$error` block
  in `VX_processor.sv` (§3.1.3). Does not constrain the LLC's own
  writeback policy.
- Extend `amo_req_t` in `VX_gpu_pkg.sv` and add the per-lane /
  per-request fields to `VX_lsu_mem_if.sv` and `VX_mem_bus_if.sv`.
  Default-zero, no consumer yet. Build remains green with
  `EXT_A_ENABLE` undefined.

### Phase 1 — `VX_amo_unit` standalone

- Create `hw/rtl/cache/VX_amo_unit.sv` and `hw/rtl/cache/VX_amo_alu.sv`.
- Add to `hw/rtl/cache/VX_cache_define.vh` filelist.
- Lint-clean under verilator `-Wall -Wpedantic`.
- Unit-test bench in `hw/unittest/VX_amo_unit_tb.sv`: every AmoType
  across W/D widths cross-checked against the SimX `amo_compute()`
  golden output; reservation alloc/check semantics under capacity
  pressure; invalidation paths. **No cache or LSU touched yet.**

### Phase 2 — Bank wiring (LLC commit path)

- Extend `VX_cache_bank.sv`: add `amo_*` ports, capture into the
  pipeline registers, add the S1 commit branch under
  `if (IS_LLC) generate`. Each LLC bank instantiates one
  `VX_amo_unit`. Wire the reservation invalidation hooks (§3.9).
- LSU still aborts on AMO — Phase 2 is testable via a synthetic
  bank-stim testbench in `hw/unittest/VX_cache_bank_amo_tb.sv`.

### Phase 3 — Non-LLC passthrough

- v1 stub: synth-time `$error` if a non-LLC, non-PASSTHRU cache is
  built with `EXT_A_ENABLE`. Forces L1-only AMO testing initially.
- v2: implement §3.8 — AmoProbe pipe entry, writeback-if-dirty,
  invalidate, forward via bypass arb.

### Phase 4 — LSU + Decoder + Execute

- `VX_decode.sv` grows the AMO opcode case under `EXT_A_ENABLE`
  (§3.11).
- `VX_lsu_slice.sv` recognizes AMO ops, MSHR-allocates them as
  load-class, packs the per-lane AMO sideband (§3.10).
- `VX_local_mem_switch.sv` gains the Shared-AMO guard (§3.12).

### Phase 5 — Conformance & multi-hart contention

- Run `riscv-tests/rv32ua-p-*` and `rv64ua-p-*` under rtlsim.
  Same binaries that pass on SimX must pass on RTL. SimX-vs-RTLsim
  CSV trace diff gives instruction-level equivalence.
- Run `tests/regression/amo` (the multi-hart Vortex-kernel suite
  that landed alongside the SimX implementation) under rtlsim.
  All 12 tests must pass.
- Run on three hierarchy configs: L1-only (default), L1+L2, L1+L2+L3.
  Same binary, same result; only timing differs.

### Phase 6 — Perf counters + docs

- Update `docs/cache_subsystem.md` with the AMO commit path and the
  reservation-table sizing.
- Add RTL perf counters mirroring SimX (`amo_total`, `amo_sc_fail`,
  `amo_reservation_evictions`).
- Flip `EXT_A_ENABLE` default to enabled in `hw/VX_config.toml` once
  rtlsim is green.

---

## 6. Out of scope

- **LMEM (Shared) atomics.** Distinct hardware site (LMEM bank),
  distinct reservation domain. `VX_local_mem_switch.sv` hard-asserts.
- **MMIO atomics.** Architecturally unspecified for Vortex's IO
  region. IO bypass path asserts on AMO if it ever sees one.
- **Write-back intermediate caches.** Build asserts every cache
  *strictly above* the LLC is write-through (§3.1.3). LLC itself is
  unconstrained. Lifting the strictly-above-LLC restriction would
  require a coherence protocol Vortex doesn't implement.
- **Forward-progress guarantees under arbitrary contention.** RVA
  permits livelock; same `AMO_RS_SIZE ≥ 2` floor as SimX gives only
  *eventual* progress.
- **Acquire/Release ordering bits (`aq` / `rl`).** Decoded but
  ignored — Vortex's single-cluster cache hierarchy is trivially
  sequentially consistent. The `tests/regression/amo` suite includes
  `amoadd_aqrl` and `lrsc_counter_aqrl` to verify the encoding paths
  through decode and bank.
- **Cross-cluster reservation invalidation.** Each cluster has its
  own LLC; an LR on cluster 0 followed by a store on cluster 1 to
  the same line cannot be invalidated without a cross-cluster
  coherence path. Default Vortex builds are single-cluster, so this
  is a non-issue today; multi-cluster AMO atomicity is future work
  and would require either a shared LLC tier or a coherence layer.

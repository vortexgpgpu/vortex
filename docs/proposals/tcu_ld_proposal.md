# TCU_LD — Direct Sparsity-Metadata Load + Memory-Subsystem Hoist

**Date:** 2026-05-25
**Status:** Draft
**Owners:** RTL team
**Related:**
[wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md),
[VX_tcu_uops.sv](../../hw/rtl/tcu/VX_tcu_uops.sv),
[VX_tcu_meta.sv](../../hw/rtl/tcu/VX_tcu_meta.sv),
[VX_tcu_mbuf.sv](../../hw/rtl/tcu/VX_tcu_mbuf.sv),
[VX_tcu_tbuf.sv](../../hw/rtl/tcu/VX_tcu_tbuf.sv),
[VX_tcu_unit.sv](../../hw/rtl/tcu/VX_tcu_unit.sv),
[VX_lsu_slice.sv](../../hw/rtl/core/VX_lsu_slice.sv),
[VX_execute.sv](../../hw/rtl/core/VX_execute.sv),
[VX_core.sv](../../hw/rtl/core/VX_core.sv),
[VX_scoreboard.sv](../../hw/rtl/core/VX_scoreboard.sv),
[vx_tensor.h](../../sw/kernel/include/vx_tensor.h).

---

## Summary

Eliminate the register-file detour in `wmma_sp` and `wgmma_sp` metadata
delivery by introducing **TCU_LD** — a TCU-family, warp-level instruction
with its own internal AGU that loads sparsity metadata directly from
memory into `VX_tcu_meta` SRAM, bypassing the architectural register file
entirely.

Because TCU_LD is one of several future warp-level SRAM-preload paths —
RTX, TEX, OM may grow analogous instructions targeting their own
tensor/texture/blend SRAMs — the memory-subsystem boundary
(`VX_mem_scheduler` + dcache port + coalescing/response routing) is
factored out of `VX_lsu_slice` and hoisted to `VX_core.sv` as a shared
block. Multiple FU-local AGUs (regular LSU's per-lane AGU in execute,
TCU FU's warp-level AGU, and future RTX/TEX/OM FU AGUs) all feed into
this single shared boundary.

Hazard handling reuses the existing general-purpose `wr_xregs` /
`rd_xregs` scoreboard machinery — no new mechanism, no new bit (see
§3.5).

Net effect:

| Axis | Before | After |
|---|---|---|
| Metadata register footprint | f26/f27 reserved per warp for sparse uops | None (lives in `VX_tcu_meta`) |
| Sequencer steps that read RF for meta | `MAX_META_STORES` per sparse uop | 0 |
| Memory-subsystem clients | one (per-lane LSU in execute) | many (LSU, TCU, future RTX/TEX/OM) |
| `VX_mem_scheduler` instances | one per LSU block | one shared at `VX_core`, multi-client |
| New scoreboard logic | — | none |
| Sparse-metadata buffers | two (`VX_tcu_meta` for RS, `VX_tcu_mbuf` for SS) | one (`VX_tcu_meta`, filled by TCU_LD for both) |
| Per-block sparse-meta state machines (`VX_tcu_mbuf` × `BLOCK_SIZE`) | 4 (BLOCK_SIZE=4) | 0 (deleted) |

---

## 1. Constraints (load-bearing)

1. **Sync semantics from the warp's POV.** TCU_LD looks like a load.
   A subsequent `wmma_sp` / `wgmma_sp` that consumes the loaded metadata
   stalls in the scoreboard until TCU_LD's writeback releases the slot.
   No async groups, mbarriers, or fences.
2. **No memory-subsystem replication.** Area budget rules out
   duplicating `VX_mem_scheduler` + dcache port for the TCU side. The
   shared boundary at `VX_core` is the single instance that all FU AGUs
   feed into.
3. **No new hazard mechanism.** The scoreboard already tracks a parallel
   bitmask of generic extended-register slots (`wr_xregs` / `rd_xregs`)
   with `inuse_xregs` per warp — see
   [VX_scoreboard.sv:76-119](../../hw/rtl/core/VX_scoreboard.sv#L76).
   TCU_LD claims one existing bit (see §3.5) and rides this machinery
   without modification.
4. **`VX_tcu_meta` SRAM becomes the only metadata home.** Per-warp,
   per-row banked storage already exists at
   [VX_tcu_meta.sv:125-143](../../hw/rtl/tcu/VX_tcu_meta.sv#L125).
   TCU_LD's destination is its write port; no new SRAM is added.
   The today's secondary metadata buffer for SS sparse —
   [VX_tcu_mbuf.sv](../../hw/rtl/tcu/VX_tcu_mbuf.sv) — is **deleted**
   (see §3.8 and P5).
5. **Sparse uop sequencer reads from SRAM, not RF.** After this change
   the `meta_store` phase synthesized by `VX_tcu_uops` for sparse uops
   is removed; the FEDP path reads `vld_block` from `VX_tcu_meta` keyed
   on `{wid, step_m, step_k}` as it already does today. The
   `tbuf_sp_meta` path from `VX_tcu_mbuf` into `VX_tcu_core` is removed
   in P5; FEDP reads only `VX_tcu_meta`.
6. **Generalization to future preload paths.** The hoisted shared
   memory subsystem is the attachment point for analogous future
   instructions in other extensions: RTX (ray-tracing tile SRAM), TEX
   (texture cache fill), OM (output-merger blend-state). Each grows
   its own FU-local warp-level AGU on the same multi-client interface.

---

## 2. Why the current path is suboptimal

Today, `wgmma_sp` and `wmma_sp` deliver metadata via the regular register
file:

- The host runtime loads metadata into `f26`/`f27` (per
  [VX_tcu_uops.sv:156](../../hw/rtl/tcu/VX_tcu_uops.sv#L156)).
- `VX_tcu_uops` synthesizes an RS-mode `meta_store` phase that reads
  f26/f27 and writes them into `VX_tcu_meta` (see
  [VX_tcu_uops.sv:333-335](../../hw/rtl/tcu/VX_tcu_uops.sv#L333)).
- `MAX_META_STORES` extra uops per sparse instruction are accounted for
  in the worst-case sequencer count
  ([VX_tcu_uops.sv:32-34](../../hw/rtl/tcu/VX_tcu_uops.sv#L32),
   [:41-42](../../hw/rtl/tcu/VX_tcu_uops.sv#L41)).

Three costs:

- **RF pressure.** f26/f27 are reserved per warp for the lifetime of the
  sparse instruction even though they hold no architecturally useful data
  past the meta-store phase.
- **Sequencer cycles.** Each sparse uop pays `MAX_META_STORES` extra
  steps that exist only to ferry metadata across the RF.
- **Wasted load bandwidth at the warp level.** The original memory load
  that brought metadata into f26/f27 already went through the LSU; the
  RF round-trip is a pure overhead detour.

A second cost lives on the **SS sparse** path. SS sparse uops fetch
metadata from shared memory via the per-block
[VX_tcu_mbuf.sv](../../hw/rtl/tcu/VX_tcu_mbuf.sv) state machine, which
autonomously refills metadata blocks from LMEM keyed on
`{desc_a, step_k}` ([VX_tcu_mbuf.sv:24-29](../../hw/rtl/tcu/VX_tcu_mbuf.sv#L24)).
mbuf produces `tbuf_sp_meta` as a *second* metadata-storage path,
parallel to `VX_tcu_meta`. The TCU thus carries two SRAMs, two
filler state machines, and two driver paths into `VX_tcu_core`'s FEDP.
TCU_LD makes mbuf redundant: a single TCU_LD instruction in software
fills `VX_tcu_meta` for the next sparse compute region regardless of
whether the metadata source lives in DMEM or LMEM. P5 deletes
`VX_tcu_mbuf` and the `tbuf_sp_meta` path.

TCU_LD targets `VX_tcu_meta` directly from a TCU-FU-local AGU through
the shared memory subsystem, eliminating both the RF reservation and the
META_STORE sequencer phases. The current emulation in
[vx_tensor.h:214-224](../../sw/kernel/include/vx_tensor.h#L214)
(`load_sp_metadata`) — per-thread loads materialized into fragment
registers — is replaced by a single warp-level instruction.

---

## 3. Design

### 3.1 ISA

Add **TCU_LD** — a new opcode in the TCU extension family. Semantics:

- **Operands:** base register (`rs1`, broadcast at warp scope — not
  per-lane), immediate offset (i-type), and a sparse-meta selector
  field that identifies which `VX_tcu_meta` region this fill targets.
- **Effect:** the TCU FU's internal AGU walks the metadata stride
  pattern derived from the current sparse-shape constants
  (`sp_per_warp_depth`, `sp_meta_cols`, `sp_num_meta_loads` —
  [vx_tensor.h:189-198](../../sw/kernel/include/vx_tensor.h#L189)),
  issues the required number of memory transactions through the
  shared memory subsystem, and writes the responses into
  `VX_tcu_meta` for the current warp.
- **Granularity:** one TCU_LD fills the entire metadata working-set
  for the next sparse compute region (one TB-K stripe or a full sparse
  MMA, sized to the same payload the META_STORE sequence delivers
  today).
- **Not per-lane.** The AGU is a single sequential address walker
  internal to the TCU FU; lanes do not contribute addresses.

Software view: emit one TCU_LD per sparse compute region. The
compiler/runtime no longer reserves f26/f27 for metadata, and the
per-thread `load_sp_metadata` materialization in
[vx_tensor.h:214](../../sw/kernel/include/vx_tensor.h#L214) is
replaced by a single intrinsic.

### 3.2 Dispatch routing

TCU_LD is decoded as a **TCU-family** instruction and rides the existing
`dispatch_if[EX_TCU]` stream — no new execute slot, no new dispatch
input. It is differentiated inside the TCU FU by op_type, not by
classification.

Inside `VX_tcu_unit`, TCU_LD is handled by a new sub-block — `VX_tcu_agu`
— alongside the existing tile-buffer and meta-store paths. The AGU:

1. Latches `{rs1_data, imm, wid}` at issue.
2. Walks the metadata stride pattern, issuing requests to the shared
   memory subsystem via a new client port (see §3.3).
3. Routes responses back through a small fill state machine that drives
   `VX_tcu_meta`'s write port.
4. Holds the warp's writeback until all in-flight fills retire, then
   raises the commit with `wr_xregs[0] = 1` to release the scoreboard
   slot.

The `META_STORE` write path into `VX_tcu_meta` (today driven by
`INST_TCU_META_STORE` uops from `VX_tcu_uops`) is preserved through P2
as a fallback, then retired in P3.

### 3.3 Hoist the memory subsystem to `VX_core.sv`

Scope of the hoist is **the memory-subsystem boundary inside
`VX_lsu_slice`**, not the whole LSU:

- `VX_mem_scheduler` ([VX_lsu_slice.sv:356-371](../../hw/rtl/core/VX_lsu_slice.sv#L356))
- coalescing / response routing
- the dcache port (`lsu_mem_if`)

These are factored out of `VX_lsu_slice` into a new shared block
`VX_lsu_scheduler` instantiated at `VX_core.sv`. What stays inside
`VX_lsu_slice` (still in `VX_execute`): the per-lane AGU
(`full_addr[i] = rs1_data[i] + offset`, [VX_lsu_slice.sv:64-66](../../hw/rtl/core/VX_lsu_slice.sv#L64)),
byte-select / alignment, and lane response steering back to the RF
writeback. The LSU slice becomes a client of the shared subsystem.

Topology after the hoist:

```
VX_core.sv
├── VX_execute.sv
│   ├── alu_unit
│   ├── fpu_unit
│   ├── lsu_unit       ── per-lane AGU + RF writeback steering
│   │      │
│   │      ▼ client port 0  (per-lane addresses, NUM_LSU_LANES wide)
│   ├── tcu_unit
│   │   ├── tcu_core / mbuf / tbuf / meta / uops   (existing)
│   │   └── tcu_agu   (new)
│   │          │
│   │          ▼ client port 1  (warp-level addresses, scalar)
│   │          fills VX_tcu_meta directly on response
│   └── sfu_unit
│
└── VX_lsu_scheduler   ◄── client 0 (LSU per-lane)
       │                    client 1 (TCU AGU)
       │                    client 2..N reserved (RTX/TEX/OM)
       │
       ├── arbiter + per-client request queues
       ├── coalescer / mem_scheduler
       └── dcache port (lsu_mem_if)
```

Multi-client interface contract:

- Each client owns its own address-generation. The subsystem takes
  pre-computed addresses + tags + byte-enables and returns responses
  routed back by tag.
- Arbitration is round-robin across clients at the request input.
  Per-client backpressure on `ready`.
- Response routing is tag-based — each client's tag space is
  partitioned at the subsystem boundary.

Future RTX/TEX/OM warp-level preload AGUs attach as additional
clients without re-touching the subsystem internals.

**`lsu_queue_empty` rewiring.** Today the device-idle gating wire is
generated inside `VX_lsu_slice`
([VX_lsu_slice.sv:121-124](../../hw/rtl/core/VX_lsu_slice.sv#L121)) as
`mem_sched_req_queue_empty & ~lsu_mem_if.req_valid & ~execute_if.valid`,
aggregated in `VX_lsu_unit` ([VX_lsu_unit.sv:74](../../hw/rtl/core/VX_lsu_unit.sv#L74)),
exposed as an output of `VX_execute`
([VX_execute.sv:62](../../hw/rtl/core/VX_execute.sv#L62)), and consumed
by `VX_core`'s busy signal
([VX_core.sv:361-364](../../hw/rtl/core/VX_core.sv#L361)). After the
hoist:

- The two scheduler-side terms (`mem_sched_req_queue_empty`,
  `~lsu_mem_if.req_valid`) move to `VX_lsu_scheduler` and are exposed
  as a `mem_subsystem_drained` output at `VX_core` level.
- The input-side term (`~execute_if.valid`) stays per-client inside
  each FU (LSU slice's input idle, TCU AGU's input idle, etc.) and is
  exposed by each FU as a small drain bit.
- The `lsu_queue_empty` output of `VX_execute` is **removed**. The
  busy expression at [VX_core.sv:364](../../hw/rtl/core/VX_core.sv#L364)
  becomes `sched_busy || dcr_busy || ~mem_subsystem_drained ||
  ~(lsu_input_idle & tcu_input_idle & …)`.

### 3.4 `VX_tcu_meta` write-port changes

Today `VX_tcu_meta` is written by `INST_TCU_META_STORE` uops emitted by
the TCU FU
([VX_tcu_meta.sv:24-28](../../hw/rtl/tcu/VX_tcu_meta.sv#L24)). After
TCU_LD lands, the new `VX_tcu_agu` drives the same write port directly
from memory responses. Address-compute on the meta side is unchanged
(`wid` from the in-flight uop, `wr_idx` derived from the AGU's fill
counter). The META_STORE driver is retired in P3.

### 3.5 Scoreboard / hazard

No new mechanism, no new bit. `wr_xregs` / `rd_xregs` is a
general-purpose shadow-register namespace: each bit is a slot that any
instruction class may claim, and the scoreboard self-serializes
producers of the same bit
([VX_scoreboard.sv:94-119](../../hw/rtl/core/VX_scoreboard.sv#L94)):

```
inuse_xregs |= staging.wr_xregs   // producer in flight reserves the slot
mask         = rd_xregs | wr_xregs // self-write included → producers serialize
xregs_busy   = inuse_xregs & mask
stall if xregs_busy != 0
inuse_xregs &= ~writeback.wr_xregs // writeback releases slot
```

**Companion rename (codebase, not behavior):** retire the domain-specific
slot names in [VX_gpu_pkg.sv:73-75](../../hw/rtl/VX_gpu_pkg.sv#L73) in
favor of generic ones:

```
XREG_FFLAGS  →  XREG_0
XREG_FRM     →  XREG_1
NUM_XREGS    =  2   (unchanged)
```

The rename is mechanical (decoder, package, any consumers) and carries
zero behavioral change. It makes the intent explicit: the slots are
generic shadow-register slots, not FP-specific.

**TCU_LD reuses `XREG_0`.** Decoder sets `wr_xregs[0] = 1` on TCU_LD;
decoder sets `rd_xregs[0] = 1` on `wmma_sp` / `wgmma_sp`. Functional
result is correct because the scoreboard already prevents two producers
of bit 0 from being simultaneously in flight — so set-on-issue /
clear-on-writeback always pairs with the correct producer.

The cost is **false stalls** when TCU_LD and FP-fflags-touching ops
share the warp's instruction stream: FP arith holds bit 0 → TCU_LD
waits; TCU_LD holds bit 0 → CSR reads of fflags/fcsr wait. In sparse
GEMM the FP-arith density is dominated by accumulator scaling and
epilogue, so the cross-stall budget is small. In TCU-only builds
(`VX_CFG_EXT_F_ENABLE` off) the cost is zero. If profiling later shows
a non-trivial regression, a follow-up can bump `NUM_XREGS` and migrate
TCU to its own bit — the rename keeps that path open.

The TCU FU raises the TCU_LD writeback only when all in-flight
sub-fills have retired into `VX_tcu_meta`. That writeback carries
`wr_xregs[0] = 1` so the scoreboard slot releases. `writeback_t`
already has the field
([VX_gpu_pkg.sv:786](../../hw/rtl/VX_gpu_pkg.sv#L786)).

### 3.6 Sequencer simplification

With metadata in `VX_tcu_meta` filled by TCU_LD:

- `VX_tcu_uops` no longer emits `INST_TCU_META_STORE` phases for sparse
  uops (drop the synthesis at
  [VX_tcu_uops.sv:333](../../hw/rtl/tcu/VX_tcu_uops.sv#L333) and the
  `MAX_META_STORES` term at
  [VX_tcu_uops.sv:32-34](../../hw/rtl/tcu/VX_tcu_uops.sv#L32)).
- `MAX_UOPS` / `MAX_WG_UOPS_SP` shrink accordingly.
- FEDP-side read remains as is (`vld_block` from `step_m`/`step_k`).

### 3.7 SS sparse path consolidation — delete `VX_tcu_mbuf`

With TCU_LD becoming the universal sparse-metadata fill path,
[VX_tcu_mbuf.sv](../../hw/rtl/tcu/VX_tcu_mbuf.sv) is redundant. The SS
sparse path today autonomously prefetches metadata from LMEM into a
per-block buffer and outputs `tbuf_sp_meta` directly to
[VX_tcu_core](../../hw/rtl/tcu/VX_tcu_core.sv). Once software emits a
TCU_LD before each SS sparse compute region, the same metadata lands
in `VX_tcu_meta` exactly as the RS path does after P3.

**Deletions:**

- `VX_tcu_mbuf.sv` (entire file).
- `tbuf_sp_meta` ports on `VX_tcu_tbuf` and `VX_tcu_core`. After P5,
  `VX_tcu_core` reads sparse metadata only via `VX_tcu_meta.vld_block`
  ([VX_tcu_meta.sv:151](../../hw/rtl/tcu/VX_tcu_meta.sv#L151)).
- The Q-instance `VX_tcu_mbuf` array inside `VX_tcu_tbuf` and the
  associated LMEM-arb client port (the metadata refill port — A, B,
  and now meta sources collapse to A+B).
- The `req_is_sparse` / `req_fmt_s` signals into `VX_tcu_mbuf` are
  removed at the `VX_tcu_tbuf` boundary (still consumed by the FEDP
  for shape decode, but no longer routed to mbuf since mbuf is gone).
- The `INST_TCU_META_STORE` path through `VX_tcu_uops` (already
  removed in P3 for RS; P5 confirms it's gone for SS too).

**Software impact:**

- SS sparse kernels emit TCU_LD with the LMEM base pointer of the
  metadata region. Same opcode and dispatch as DMEM-sourced TCU_LD —
  the `VX_tcu_agu` and the shared memory subsystem don't care whether
  the address resolves to LMEM or DMEM; the dcache port already
  handles both.
- `vx_tensor.h`'s sparse-load intrinsics collapse: one TCU_LD entry
  point regardless of where metadata lives. The current branching
  between RS (`load_sp_metadata` per-thread, removed in P3) and SS
  (autonomous via mbuf) goes away.

**Prefetch concern.** `VX_tcu_mbuf` today overlaps autonomous
metadata refill with current-uop compute (refill keyed on next
`step_k`). With explicit TCU_LD, the runtime is responsible for
issuing TCU_LD far enough ahead of the consuming `wgmma_sp` to hide
the load latency. Two mitigations available, both software-only:

1. **Hoist TCU_LD across K-stripe boundaries.** Compiler/runtime
   schedules TCU_LD ahead of the K-loop body so the metadata is
   already resident in `VX_tcu_meta` by the time the first
   `wgmma_sp` of the stripe issues.
2. **Double-buffer `VX_tcu_meta` slots.** If single-stripe lookahead
   isn't enough, allocate two slots in `VX_tcu_meta` per warp,
   pipeline TCU_LD against the previous stripe's compute. Costs one
   bit of slot select in `VX_tcu_meta`'s addressing; zero RTL beyond
   the meta address widening.

If profiling at P5 shows SS-sparse throughput drops vs. baseline,
(1) lands first; (2) is the fallback. Both are layered above the
single-buffer P5 design — they don't gate P5.

**Area savings beyond §Summary table.**

- `VX_tcu_mbuf` deletion: per-block metadata SRAM (sized by
  `TCU_META_PER_WARP_DEPTH × TCU_META_COLS_PER_LOAD × 32`) ×
  `BLOCK_SIZE` instances, plus refill FSM state. Specific FF/LUT
  count subject to synthesis; non-trivial.
- One client port off the `VX_tcu_tbuf` LMEM arbiter.
- Reduced wiring fan-out from `VX_tcu_core`'s FEDP input mux.

### 3.8 SimX alignment

The SimX model must track the RTL changes so the regression's
SimX-vs-RTL parity check holds. Concretely:

- **Decoder.** Add TCU_LD opcode handling in
  [sim/simx/decode.cpp](../../sim/simx/decode.cpp) mirroring the RTL
  decoder's `wr_xregs[0]` assertion and TCU-family dispatch.
- **Scoreboard.** The XREG rename (P0) propagates to any SimX-side
  symbolic constants. SimX's hazard model already mirrors the RTL
  scoreboard masks; no behavioral change.
- **TCU model.** Add a `tcu_agu` step in
  [sim/simx/tcu/tcu_unit.cpp](../../sim/simx/tcu/tcu_unit.cpp)
  paralleling RTL `VX_tcu_agu`: walks the metadata stride, issues
  memory requests, fills the per-warp `meta_store` state, and signals
  commit only when the fill drains. The existing `meta_store`
  ([tcu_unit.h:99-102](../../sim/simx/tcu/tcu_unit.h#L99)) entry point
  remains for the META_STORE fallback path through P2 and is retired
  in P3 alongside RTL.
- **Memory subsystem.** SimX currently models memory access per-FU;
  the multi-client subsystem at `VX_core` is reflected by routing TCU
  AGU requests through the same memory pathway the LSU model uses (no
  new dispatcher needed — SimX's queueing is functional). The cycle
  model adds round-robin arbitration matching RTL.
- **`load_sp_metadata` runtime.** Once P3 lands and `vx_tensor.h`'s
  `load_sp_metadata` is replaced by a TCU_LD intrinsic, SimX
  automatically exercises the new path through the host program.
- **`tcu_mbuf` model deletion (P5).** The corresponding SimX model
  for the SS sparse mbuf path under
  [sim/simx/tcu/](../../sim/simx/tcu/) is deleted alongside the RTL
  removal. SS sparse uops in SimX read metadata only from the
  `VX_tcu_meta` mirror state, which TCU_LD has already filled.

Validation: `tensor_sp` on XLEN=32/64 must pass under both
`--driver=simx` and `--driver=rtlsim` at every phase, with bit-exact
output. Cycle counts will diverge from baseline at P3 (sequencer step
count drops); document the expected delta in the phase landing notes.
P5 additionally must hold cycle parity on the SS sparse workloads
(see §4.1 SS row group) — any drop in throughput indicates the
prefetch-loss mitigations in §3.7 are needed.

---

## 4. Phasing

Each phase is independently shippable. **Validation for every phase is
the `tensor_sp` regression target
([ci/regression.sh.in:979](../../ci/regression.sh.in#L979)), run from
both the 32-bit and 64-bit build trees** (`XLEN=32` and `XLEN=64`). The
phase is considered complete when both passes are clean.

### 4.0 Pre-work (required before P0)

Two pre-existing issues are uncovered while capturing baselines and
must land first to keep validation honest:

| # | Issue | Resolution |
|---|---|---|
| **PW1** | Verilator `-Wall` build of rtlsim with TCU enabled fails on an `UNUSEDSIGNAL` warning for `delayed_fmt_s[2:0]` in [hw/rtl/tcu/dpi/VX_tcu_fedp_dpi.sv:139](../../hw/rtl/tcu/dpi/VX_tcu_fedp_dpi.sv#L139), introduced by `f9990254` ("strip tensor-core MX support"). | Add ``` `UNUSED_VAR (delayed_fmt_s[2:0])``` next to the declaration. One-line, no behavioral change. **Applied.** |
| **PW2** | `sgemm_tcu_sp` `ITYPE=tf32` on rtlsim returns a poisoned `mma_sync` cycle count (~3.13×10⁹) from the kernel's `rdcycle` CSR while wall-clock elapsed is normal (~5 s). Reproducible. Correctness PASSES. Likely a kernel-side `cycles_buffer` initialization gap or a tf32-specific cycle-CSR sampling bug; happens only on rtlsim, not simx. | Investigate the `PROFILE_ENABLE` cycle-capture path in [tests/regression/sgemm_tcu_sp/kernel.cpp](../../tests/regression/sgemm_tcu_sp/kernel.cpp) for tf32. Fix so rtlsim tf32 baseline is meaningful. Until fixed, the tf32-rtlsim row in §4.1 is bogus and the corresponding P3 speedup cell will be marked _N/A_. |

Validation gate for PW2: re-run `sgemm_tcu_sp ITYPE=tf32 OTYPE=fp32` on
rtlsim XLEN=32; expected `mma_sync max` should fall in the same
order-of-magnitude as the simx counterpart (`6017` baseline) — anything
in the 10³–10⁵ range is plausible; 10⁹+ is wrong.

| Phase | Scope |
|---|---|
| **P0** | XREG rename: `XREG_FFLAGS → XREG_0`, `XREG_FRM → XREG_1` in [VX_gpu_pkg.sv](../../hw/rtl/VX_gpu_pkg.sv) + every consumer ([VX_decode.sv](../../hw/rtl/core/VX_decode.sv) and others). Pure naming; zero behavioral change. |
| **P1** | Factor `VX_mem_scheduler` + dcache port + coalescing out of `VX_lsu_slice` into new `VX_lsu_scheduler` at `VX_core`. Single client (LSU per-lane AGU). Define the multi-client interface but leave other ports tied off. |
| **P2** | Add `VX_tcu_agu` sub-block inside `VX_tcu_unit`. Wire it as client 1 of `VX_lsu_scheduler`. Add TCU_LD opcode + decoder. **Keep** the META_STORE path live as fallback. Provide a runtime intrinsic that emits TCU_LD, but `load_sp_metadata` still uses the per-thread path by default. |
| **P3** | Switch `load_sp_metadata` and the WGMMA sparse runtime to emit TCU_LD. Remove `INST_TCU_META_STORE` synthesis in `VX_tcu_uops`. Shrink `MAX_UOPS` / `MAX_WG_UOPS_SP`. Remove f26/f27 reservation in the runtime. |
| **P4** | SimX parity (see §3.8). |
| **P5** | Delete `VX_tcu_mbuf.sv` and the `tbuf_sp_meta` path (see §3.7). SS sparse software emits TCU_LD; FEDP reads only `VX_tcu_meta`. Removes the second sparse-metadata SRAM and `BLOCK_SIZE` instances of mbuf state machines + LMEM-arb client. |

P0 is a one-shot rename, bit-exact. P1 is a pure RTL refactor — same
behavior with one client. P2 introduces the AGU + opcode but keeps
META_STORE emission as the default path, so `tensor_sp` still exercises
the legacy flow and must remain bit-exact end-to-end. P3 is the
behavior-changing flip for RS sparse: same numerical result, reduced
sequencer step counts (per-uop META_STORE retired). P4 closes SimX/RTL
parity. P5 retires the SS sparse `VX_tcu_mbuf` path, leaving
`VX_tcu_meta` as the only metadata SRAM and a single fill path
(`VX_tcu_agu` → `VX_tcu_meta`) for both RS and SS sparse.

### 4.1 Latency baseline and speedup validation

Beyond bit-exact correctness, P3 must measurably reduce sgemm_tcu_sp
cycle count. Capture three representative configs from the `tensor_sp`
regression — `sgemm_tcu_sp` at `fp16`, `fp8`, and `tf32` (matching
[ci/regression.sh.in:990,993,992](../../ci/regression.sh.in#L990)) —
on both SimX and RTL.

**Baseline measurement** (captured 2026-05-25 on `tinebp-patch-2` @
[49202fb6](../../../) prior to any TCU_LD work, NT=2, with
`-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE -DPROFILE_ENABLE`,
rtlsim adds `-DVX_CFG_TCU_TYPE_DPI`). Metric is the kernel's
`mma_sync` cycle counter (per-block max / sum across blocks):

| Workload | Driver | XLEN | mma_sync max | mma_sync total |
|---|---|---|---|---|
| `sgemm_tcu_sp` `ITYPE=fp16, OTYPE=fp32` | simx   | 32 | 3211 | 662965 |
| `sgemm_tcu_sp` `ITYPE=fp16, OTYPE=fp32` | simx   | 64 | 3745 | 786932 |
| `sgemm_tcu_sp` `ITYPE=fp16, OTYPE=fp32` | rtlsim | 32 | 3649 | 650138 |
| `sgemm_tcu_sp` `ITYPE=fp8,  OTYPE=fp32` | simx   | 32 | 1832 | 339032 |
| `sgemm_tcu_sp` `ITYPE=fp8,  OTYPE=fp32` | simx   | 64 | 1967 | 384320 |
| `sgemm_tcu_sp` `ITYPE=fp8,  OTYPE=fp32` | rtlsim | 32 | 1996 | 333541 |
| `sgemm_tcu_sp` `ITYPE=tf32, OTYPE=fp32` | simx   | 32 | 6017 | 1317639 |
| `sgemm_tcu_sp` `ITYPE=tf32, OTYPE=fp32` | simx   | 64 | 6970 | 1571858 |
| `sgemm_tcu_sp` `ITYPE=tf32, OTYPE=fp32` | rtlsim | 32 | **3,131,961,357** ⚠ | **3,133,439,257** ⚠ |

⚠ **tf32 rtlsim anomaly.** The kernel's `rdcycle`-based counter returns
a poisoned value (~3.13×10⁹) reproducibly, while wall-clock elapsed
(~5 s) is in line with fp16/fp8 rtlsim runs. The kernel itself passes
correctness. Treat the tf32 rtlsim baseline as **bogus** for now —
investigate before relying on it as a regression gate. Likely
candidates: per-warp cycle CSR read path under tf32 in rtlsim, or an
uninitialized read in `cycles_buffer` for tf32 layout. Tracked as an
open question; not blocking the proposal.

**Post-P3 speedup validation** (record after P3 lands, comparing
`mma_sync max` to baseline). Same configs; speedup = baseline / post-P3.

| Workload | Driver | XLEN | Baseline max | Post-P3 max | Speedup |
|---|---|---|---|---|---|
| fp16  | simx   | 32 | 3211 | _TBD_ | _TBD_ |
| fp16  | simx   | 64 | 3745 | _TBD_ | _TBD_ |
| fp16  | rtlsim | 32 | 3649 | _TBD_ | _TBD_ |
| fp8   | simx   | 32 | 1832 | _TBD_ | _TBD_ |
| fp8   | simx   | 64 | 1967 | _TBD_ | _TBD_ |
| fp8   | rtlsim | 32 | 1996 | _TBD_ | _TBD_ |
| tf32  | simx   | 32 | 6017 | _TBD_ | _TBD_ |
| tf32  | simx   | 64 | 6970 | _TBD_ | _TBD_ |
| tf32  | rtlsim | 32 | (anomaly — see above) | _TBD_ | _TBD_ |

**Expected lower bound on speedup.** The sequencer step count drops by
the `MAX_META_STORES` factor in `VX_tcu_uops` per sparse instruction
([VX_tcu_uops.sv:33-34](../../hw/rtl/tcu/VX_tcu_uops.sv#L33)). Net
end-to-end speedup will be less than that ratio because TCU_LD itself
consumes memory cycles. Target: positive speedup on all three types
on both drivers; precise numerical floor set once baselines are
captured.

P3 is rejected if any cell shows a regression. RTL P0/P1/P2 cycle
counts must equal baseline within noise (pure refactors, no
behavioral change).

---

## 5. Open questions

- **`VX_tcu_agu` fill granularity.** A single TCU_LD covers the entire
  next sparse compute region above. The AGU's internal MSHR depth and
  whether it pipelines fills across compute regions (double-buffering)
  are P2-prototyping decisions.
- **Multi-client arbitration policy.** Round-robin is the starting
  point. If profiling shows LSU starvation when TCU_LD is in flight,
  consider priority schemes biased toward whichever client has the
  smaller outstanding request count.
- **XREG slot reuse vs. extension.** Plan of record: reuse `XREG_0`
  (renamed from `XREG_FFLAGS`) for TCU_LD hazard tracking, accepting
  FP-arith ↔ TCU_LD cross-stalls. If profiling later shows the
  cross-stall cost in mixed FP/TCU workloads is material, bump
  `NUM_XREGS` and split TCU off into its own bit. The rename keeps
  this path open with no extra refactor.
- **TCU_LD outside the TCU extension build.** When `VX_CFG_EXT_TCU_ENABLE`
  is off, TCU_LD decodes as illegal. The `VX_lsu_scheduler` hoist (P1)
  is independent of TCU and ships regardless.
- **SS sparse prefetch-loss budget (§3.7).** Whether removing the
  autonomous mbuf refill costs measurable throughput on SS sparse
  workloads depends on how early the compiler/runtime can hoist
  TCU_LD before the consuming WGMMA stripe. Decide at P5 whether
  software-only hoisting is enough or whether double-buffered
  `VX_tcu_meta` slots are required.
- **Future preload clients.** RTX/TEX/OM analogs are sketched here but
  not detailed. Each will be a separate proposal that plugs in a new
  FU-local AGU as another `VX_lsu_scheduler` client + claims its own
  generic XREG bit (or shares, with cross-stall cost).

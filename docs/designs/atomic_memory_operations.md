# Atomic Memory Operations (RVA) — Design

**Scope:** the Vortex implementation of the RISC-V "A" extension
(atomic memory operations: `LR`/`SC` and `AMOSWAP/ADD/AND/OR/XOR/MIN/MAX`
with signed/unsigned min-max, `.W` and `.D` widths). Covers the RTL
([`hw/rtl/cache/VX_amo_unit.sv`](../../hw/rtl/cache/VX_amo_unit.sv),
[`VX_amo_alu.sv`](../../hw/rtl/cache/VX_amo_alu.sv) + the cache-bank /
LSU integration) and the SimX model
([`sim/simx/amo/`](../../sim/simx/amo/)).

AMO is gated by `VX_CFG_EXT_A_ENABLE` (default off,
[`VX_config.toml:34`](../../VX_config.toml#L34)); when enabled it sets
`MISA` bit 0 ([`VX_config.toml:300`](../../VX_config.toml#L300)).

The defining architectural choice: **atomics are resolved at the
last-level cache (LLC) bank**. A build-time assertion enforces that every
cache level strictly above the LLC is write-through
([`Vortex.sv:62-67`](../../hw/rtl/Vortex.sv#L62),
[`processor.cpp:84-94`](../../sim/simx/processor.cpp#L84)), so the LLC
sees every write and can keep LR/SC reservations coherent.

---

## 1. Architecture overview

```
  decode (0x2F)          LSU                  coalescer/switch        LLC cache bank
  ───────────►  pack amo_req sideband  ───►  AMO lanes never    ───►  S1 commit:
  funct5→amo_op {amo_valid,amo_op,            coalesced;              VX_amo_alu RMW
  funct3→width   amo_unsigned,hart_id}        LMEM/IO AMO            + VX_amo_unit
  aq/rl (unused) rw forced to 0 (load)        asserted-out           reservation CAM
                                                                          │
                          rd ◄──────────── ret_word / SC 0|1 ◄───────────┘
```

An AMO travels the **load** path (its `rw` bit is forced to 0, even for
`SC`) so it allocates a load-class MSHR entry and returns a value to `rd`.
The actual read-modify-write happens in one cycle at the LLC bank.

---

## 2. ISA decode

Opcode `0x2F` = `INST_AMO`
([`VX_gpu_pkg.sv:269`](../../hw/rtl/VX_gpu_pkg.sv#L269)). Decode
([`VX_decode.sv:346-365`](../../hw/rtl/core/VX_decode.sv#L346),
[`decode.cpp:628-642`](../../sim/simx/decode.cpp#L628)):

- `funct3` → width: `010` = `.W` (32-bit), `011` = `.D` (64-bit).
- `funct5` → AMO op + `amo_unsigned` (MINU/MAXU collapse to MIN/MAX plus
  the unsigned bit).
- `aq`/`rl` are captured but **unused** (the LLC-resolution model makes
  them no-ops for this microarchitecture).
- `rs2_valid` is suppressed for `LR`.

---

## 3. RTL components

### 3.1 `VX_amo_alu` — the RMW kernel

[`VX_amo_alu.sv`](../../hw/rtl/cache/VX_amo_alu.sv) is pure combinational.
Inputs `{amo_op_e op, amo_unsigned, width[1:0], old_word[63:0],
rhs[63:0]}`; outputs `new_word` (value to store) and `ret_word` (old value
returned to `rd`). The op case is at
[`:45-60`](../../hw/rtl/cache/VX_amo_alu.sv#L45); `.W` vs `.D` selection
and sign-extension for MIN/MAX at [`:36-42`](../../hw/rtl/cache/VX_amo_alu.sv#L36).

The RTL op enum `amo_op_e`
([`VX_gpu_pkg.sv:178-189`](../../hw/rtl/VX_gpu_pkg.sv#L178)) is a private
4-bit space (`LR=0, SC=1, ADD=2, SWAP=3, XOR=4, OR=5, AND=6, MIN=7,
MAX=8`) — 9 ops, with MINU/MAXU folded via `amo_unsigned`.

### 3.2 `VX_amo_unit` — reservations

[`VX_amo_unit.sv`](../../hw/rtl/cache/VX_amo_unit.sv) instantiates the ALU
([`:60`](../../hw/rtl/cache/VX_amo_unit.sv#L60)) plus a small reservation
CAM (`res_entry_t = {valid, hart_id, line_addr, lru}`,
[`:73-78`](../../hw/rtl/cache/VX_amo_unit.sv#L73), size
`VX_CFG_AMO_RS_SIZE = 4`). `SC` performs a combinational `check`
([`:87-93`](../../hw/rtl/cache/VX_amo_unit.sv#L87)); victim selection
prioritizes same-hart > free > LRU
([`:111-143`](../../hw/rtl/cache/VX_amo_unit.sv#L111)). `LR` reserves, `SC`
clears, and any committed store invalidates other harts' reservations to
the line — all firing on the same cycle
([`:159-192`](../../hw/rtl/cache/VX_amo_unit.sv#L159)). The unit exists
only at the LLC bank (`g_amo_unit` when `IS_LLC`,
[`VX_cache_bank.sv:720,789`](../../hw/rtl/cache/VX_cache_bank.sv#L720));
non-LLC banks generate `g_no_amo_unit` (zero AMO gates,
[`:809`](../../hw/rtl/cache/VX_cache_bank.sv#L809)).

### 3.3 Cache-bank integration

The bank decides the AMO outcome at S1
([`VX_cache_bank.sv:835-919`](../../hw/rtl/cache/VX_cache_bank.sv#L835)):
`sc_fail = (op==SC) && !check`, `do_store = (op!=LR) && !sc_fail`. A
dedicated writeback FSM (`amo_wb_pending`/`amo_wb_data_r`,
[`:333-349,850-889`](../../hw/rtl/cache/VX_cache_bank.sv#L333)) defers the
line update to a synthetic write cycle with a 2-cycle forwarding window
for chained same-line AMOs. On a miss the AMO reserves an MSHR like a load
and re-runs the commit on fill (`replay_amo`). Width is derived at the
bank from the `byteen` popcount
([`:762`](../../hw/rtl/cache/VX_cache_bank.sv#L762)).

### 3.4 Interface plumbing

AMOs ride the memory fabric as an opaque per-lane attribute
(`mem_bus_attr_t`) carrying a **slim** `amo_req_t`
(`{amo_valid, amo_op, amo_unsigned, hart_id}`) at `MEM_ATTR_AMO_OFFS`
([`VX_gpu_pkg.sv:194-220`](../../hw/rtl/VX_gpu_pkg.sv#L194)). The LSU packs
this sideband ([`VX_lsu_slice.sv:92-104`](../../hw/rtl/core/VX_lsu_slice.sv#L92)),
forcing `mem_req.rw = 0` for all AMOs including `SC`
([`:179-182`](../../hw/rtl/core/VX_lsu_slice.sv#L179)). AMO lanes are
never coalesced (RVA is non-commutative), and LMEM/shared-memory AMOs are
asserted out ([`VX_lmem_switch.sv:121-124`](../../hw/rtl/mem/VX_lmem_switch.sv#L121)).
Because the attribute is opaque, the shared library IP
(`VX_mem_scheduler`, `VX_mem_coalescer`) is AMO-agnostic — there are no
AMO references in `hw/rtl/libs/`.

---

## 4. SimX model

[`sim/simx/amo/amo_ops.h`](../../sim/simx/amo/amo_ops.h) mirrors the ALU
(`amo_compute(MemOp, width, old, rhs, unsigned_minmax)`,
[`:36-74`](../../sim/simx/amo/amo_ops.h#L36)).
[`amo_unit.{h,cpp}`](../../sim/simx/amo/amo_unit.cpp) is the reservation
`AmoUnit` (a plain class, not a SimObject): `compute`, `reserve` (same-hart
overwrite + LRU evict), `check`, `invalidate(line, except_hart)`, `clear`,
owned by `CacheBank` and active only when `config_.is_llc`
([`cache.cpp:490`](../../sim/simx/mem/cache.cpp#L490)); the RMW commits in
`commitAmo` ([`cache.cpp:705`](../../sim/simx/mem/cache.cpp#L705)).

SimX folds the AMO op into the unified `MemOp` enum
([`types.h:381-399`](../../sim/simx/types.h#L381)) with a `MemFlags`
bitfield ([`:415-452`](../../sim/simx/types.h#L415)) and `hart_id =
make_hart_id(cid,wid,tid)`. Unlike RTL, SimX has a full **non-LLC
passthrough** path (`AmoProbe`, [`cache.cpp:281,809`](../../sim/simx/mem/cache.cpp#L281))
that probe-invalidates and forwards atomics through intermediate cache
levels — the reference for the RTL gap noted in §6.

Conformance: `tests/regression/amo` (~12 cases) and the gated
`rv32ua-p` / `rv64ua-p` riscv-tests
([`tests/riscv/isa/Makefile:33-34`](../../tests/riscv/isa/Makefile#L33)).

---

## 5. End-to-end flow

1. **Decode** — opcode `0x2F` → FU_LSU; funct5/funct3 set op/width/unsigned.
2. **LSU** — pack the `amo_req_t` sideband, force `rw=0`, allocate a
   load-class MSHR (the result returns to `rd`).
3. **Coalescer/switch** — AMO lanes issue per-bank, never merged;
   LMEM/IO AMOs rejected.
4. **LLC bank commit** — single-cycle RMW via `VX_amo_alu`; reservation
   update via `VX_amo_unit`; response = `ret_word` (or 0/1 for `SC`).
5. **Miss/replay** — reserve MSHR, replay commit on fill.
6. **Reservation coherence** — LR reserves `(hart_id, line)`; SC and any
   committed store to the line invalidate other harts. RVA-conformant
   (spurious SC failure via LRU eviction is permitted).

---

## 6. Proposed but not yet implemented

1. **RTL non-LLC AMO passthrough** (`amo_rtl_v3_proposal` §3.8 / Phase 3).
   RTL only stubs the non-LLC case
   ([`VX_cache_bank.sv:809-818`](../../hw/rtl/cache/VX_cache_bank.sv#L809));
   SimX's `AmoProbe` path is the reference implementation. Needed for
   RTL L1+L2 / L1+L2+L3 configurations.
2. **Plain-write reservation invalidation in RTL.** Currently only AMO
   stores invalidate reservations; ordinary write-throughs from other
   harts do not (TODO at
   [`VX_cache_bank.sv:778-782`](../../hw/rtl/cache/VX_cache_bank.sv#L778)).
   Multi-hart correctness against non-AMO stores depends on this.
3. **RTL AMO unit testbench** `hw/unittest/VX_amo_unit_tb.sv` — not built.
4. **RTL AMO perf counters** (`amo_total`, `amo_sc_fail`,
   `amo_reservation_evictions`) — proposed, not present in RTL.
5. **RTL rtlsim/FPGA conformance sign-off** (`amo_rtl_v3` Phase 5):
   `rv32ua`/`rv64ua` + `regression/amo` under rtlsim/xrt — not confirmed.
6. **SimX↔RTL bit-level AMO trace parity.** The RTL `amo_op_e` value space
   (`LR=0…MAX=8`) and the SimX `MemOp` value space (`AMO_LR=3…AMO_MAX=11`,
   with ADD/SWAP and AND/OR/XOR reordered) differ; trace parity needs a
   translation layer and the static-assert parity header proposed in
   `amo_packing_optimization_proposal` §16. Functionally each side is
   self-consistent; this is a tooling gap, not a correctness bug.

**Superseded directions** (recorded to avoid revival): the
`amo_packing_optimization_proposal`'s **Part A** RTL refactor — an
opcode-based `mem_op_e` interface that fully dissolved `amo_req_t` — was
**not** adopted; the codebase instead reached the same "AMO-agnostic
library IP" goal via `libs_feature_agnostic_redesign` (the opaque
`mem_bus_attr_t` attribute carrying a slim `amo_req_t`). `grep mem_op_e
hw/rtl/` is empty. The proposal's **Part B** (the SimX `MemOp`/`MemFlags`
migration, `cid→hart_id` rename, per-lane `tids[]`) *did* ship. The
`amo_rtl_v3_proposal` status header claiming "Phases 2/4/5 deferred" was
stale — Phases 2 and 4 are fully built; only 3 and 5 remain (items 1, 5
above).

---

## 7. Source proposals

This design consolidates and supersedes the following proposals (now
removed from `docs/proposals/`): `amo_rtl_v3_proposal.md`,
`amo_simx_v3_proposal.md`, `amo_packing_optimization_proposal.md`.

The opaque memory-attribute interface that carries the AMO sideband is
part of the broader library-IP refactor (see `vortex_runtime_api.md` /
`libs_feature_agnostic_redesign`).

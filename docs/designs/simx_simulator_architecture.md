# SimX — Cycle-Approximate Simulator — Design

**Scope:** the architecture of SimX, the Vortex cycle-approximate C++
simulator ([`sim/simx/`](../../sim/simx/)), and its SST integration
([`sim/simx/sst/`](../../sim/simx/sst/)).

This document is architectural. The `SimObject`/`SimChannel` framework
mechanics are in [`docs/simobject.md`](../simobject.md) and usage is in
[`docs/simulation.md`](../simulation.md); this doc covers the v3 model —
how functional and timing meet, the module decomposition, and the SST
boundary — without repeating those.

---

## 1. The v3 model: functional + timing in one place

The defining property of SimX v3: there is **no central `Emulator`**. ISA
semantics live in the same module that models the timing of the
corresponding hardware block. ALU and FPU own private `execute()`
methods; the SFU routes to its sub-units; CSR/WCTL semantics live in their
sub-unit classes; warp/CTA/barrier state lives in `Scheduler`; register
files live in `OpcUnit`; decode lives in `Decoder`. Data flows through the
memory hierarchy as real payload — caches and DRAM carry line data, and
there is no `core->mem_read/mem_write` back door (those names survive only
as perf-counter labels).

This makes SimX a faithful, module-by-module twin of the RTL, which is why
it serves as the RTL oracle for cycle-parity debugging.

---

## 2. Framework

[`sim/common/simobject.h`](../../sim/common/simobject.h) provides:

- **`SimObject<Impl>`** (CRTP) — a hardware block with `on_tick`/`on_reset`
  hooks; passive objects auto-skip.
- **`SimChannel<Pkt>`** — a typed, backpressured link: a producer
  `send(pkt, delay)` schedules delivery; `pending_count_` enforces
  capacity-based occupancy; `tx_callback` allows snooping. **The channel
  is the pipeline** — latency is the `send` delay and occupancy is the
  output channel's capacity, so pipelined units need no internal stage
  deque.
- **`SimPlatform`** — the singleton tick engine: a 4096-bucket timing
  wheel plus immediate/delta event queues. Each `tick()` settles delta
  events, ticks every active object in creation order, settles deltas
  again, advances the cycle, then fires wheel events for that cycle.

---

## 3. Component inventory

**Orchestration**
- [`main.cpp`](../../sim/simx/main.cpp) — the standalone `simx` CLI driver.
- [`processor.{h,cpp}`](../../sim/simx/processor.cpp) — `Processor`: owns
  clusters + the single `Memory` (DRAM), aggregates perf, exposes
  `cycle()` (single-cycle step) and `any_running()`.
- `cluster.{cpp,h}` → `socket.{cpp,h}` → `core.{cpp,h}` — the GPU
  hierarchy; `Core` wires the per-core pipeline.

**Per-core pipeline** (mostly `SimObject`s)
- `scheduler.{cpp,h}` — per-warp lifecycle: PC/tmask/fcsr/IPDOM stack,
  barriers, trap state; owns the `CtaDispatcher` and a `BarrierUnit`.
- `cta_dispatcher.{cpp,h}` — CTA → warp (block/grid) mapping.
- `barrier_unit.{cpp,h}` — barrier arrival/release tracking.
- `sequencer.{cpp,h}` — per-warp micro-op expander (multi-uop TCU
  WMMA/WGMMA; simple instructions pass through).
- `decode.{cpp,h}` — `Decoder`: RISC-V decode into `Instr`.
- `decompressor.{cpp,h}` — RVC fetch decompressor.
- `scoreboard.{cpp,h}` — register hazard tracking / in-order completion.
- `operands.{cpp,h}` + `opc_unit.{cpp,h}` — `Operands` routes to per-lane
  `OpcUnit`s, which own the per-warp register files and do writeback
  (mirrors RTL `VX_opc_unit.sv`).
- `dispatcher.{cpp,h}` — aggregates `ISSUE_WIDTH` issue ports onto per-FU
  execution lanes.
- `func_unit.h` — `FuncUnit<NUM_BLOCKS>` CRTP base; each FU owns
  `Inputs[]`/`Outputs[]` channels.

**Functional units**
- `alu_unit`, `fpu_unit` — own private `execute()` (ISA semantics).
- `lsu_unit` — loads/stores/AMO; `LsuUopGen` for packed loads; pulls load
  data from `MemRsp::data`.
- `sfu_unit` — special-function dispatcher routing to sub-units: WCTL,
  CSR, TEX, RASTER, DXA, OM.
- `csr_unit`, `wctl_unit` — SFU sub-units (CSR/FCSR/MPM; warp control →
  `Scheduler`).
- optional `tcu_unit.*` plus graphics `tex/`, `raster/`, `om/`, `dxa/`.

**Memory** ([`sim/simx/mem/`](../../sim/simx/mem/) + `amo/`)
- `mem_coalescer` — per-LSU-block coalescer.
- `cache.{cpp,h}` — `Cache`: tags + MSHRs + replacement **and line data**;
  embeds the AMO unit.
- `cache_cluster` — shared L2/L3 wrapper.
- `memory.{cpp,h}` — `Memory`: bottom-of-hierarchy DRAM (ramulator2);
  `attach_ram(RAM*)` owns the image; carries data in `MemRsp`; exposes
  `set_pre_send_hook` for SST mirroring.
- `amo/amo_unit.{cpp,h}` — AMO RMW in the cache hierarchy (see
  [`atomic_memory_operations.md`](atomic_memory_operations.md)).
- `local_mem*`, `lsu_mem_adapter`, `mmu*` (see [`virtual_memory_subsystem.md`](virtual_memory_subsystem.md)).

`MemReq`/`MemRsp` ([`sim/simx/types.h`](../../sim/simx/types.h), ~L1185)
carry `shared_ptr<mem_block_t> data` + `byteen`; a LOAD response must
carry a line payload.

---

## 4. Pipeline flow

Fetch (via `Scheduler` PC + I-cache) → Decompress (RVC) → Decode →
Sequence (uop expand) → I-buffer → Scoreboard issue gate → Operands /
`OpcUnit` register read → Dispatcher → FuncUnit lanes (ALU / FPU / LSU /
SFU{WCTL,CSR,TEX,RASTER,DXA,OM} / TCU) → commit arbiters → `OpcUnit`
writeback. Memory: LSU → coalescer → L1 `Cache` → L2/L3 `cache_cluster` →
`Memory`. Standalone, `main.cpp` loads an image into `RAM`, programs the
KMU via DCR writes, and ticks `SimPlatform` until `!any_running()`.

---

## 5. SST integration

SST mode is a compile-time opt-in where SST is the clock driver.
[`sst/vortex_gpgpu.{cpp,h}`](../../sim/simx/sst/vortex_gpgpu.cpp) is a
`SST::Component` that owns a `VortexSimulator`
([`sst/vortex_simulator.{cpp,h}`](../../sim/simx/sst/vortex_simulator.cpp),
a v3 `Processor` + `RAM`) and calls `sim_->cycle()` per SST clock tick.

The memory boundary is a **one-way timing mirror**, not a polymorphic
peer: `Memory::set_pre_send_hook` mirrors every accepted `MemReq` to SST's
`memHierarchy` as a block-aligned, byteen-masked `StandardMem::Read/Write`.
SST responses are acknowledged and discarded; the local `RAM` remains the
single source of truth for data. This satisfies the "one boundary, single
source of truth" constraint by a hook rather than the originally proposed
`MemorySST` subclass.

---

## 6. Proposed but not yet implemented

1. **Documentation lock-in** (`simx_v3_proposal` Phase 6): a promoted
   `docs/simx_architecture.md` reference, a `simx_extension_guide.md` with
   a worked "add a FuncUnit" example, and an optional CI lint rejecting
   new cross-module method calls — none exist yet (this design doc is a
   first step).
2. **SST data-routed memory peer** (`sst_simx_v3_proposal`): the full
   `MemorySST` with an `outstanding_` tag→sink map that completes `MemRsp`
   from SST's response is not built — today SST is a one-way mirror. If
   SST-driven memory *timing* should feed back into SimX, this is the
   unbuilt piece (with its leak/timeout, tag-routing, and clock-starvation
   risks).
3. **SST CI/apptainer breadth**: a host-CPU-driven SST integration
   (`ci/sst_run_app.py`), the `sst` apptainer matrix entry, and DXA/KMU-
   under-SST verification are deferred (per-test SST scripts were
   consolidated into one hostless runner).
4. **Memory-model headroom** the design deliberately left open: cache
   coherency (MESI/MOESI) is now an additive change since lines carry
   data + dirty state, and a "sleep-until-input" optimization for the
   per-cycle O(N_objects) tick is noted for future scaling.

**Superseded directions** (recorded to avoid revival): the central
`Emulator` god-object and `MemBackend`/`core->mem_read` data path (deleted
— semantics now live on the units, data lives in the hierarchy); a public
`execute()` on every FuncUnit (reverted to private); an `isa::` namespace
and SystemC (never adopted); and the `MemorySST` polymorphic-subclass SST
boundary (replaced by the `set_pre_send_hook` mirror).

---

> **Upstream divergence note.** The upstream `vortexgpgpu/vortex` PR #298
> SST integration is **intentionally absent** on this `simx_v3` line — the
> SST boundary here is the `set_pre_send_hook` mirror described in §5, not
> the upstream SST path. Future SST users should not expect the upstream
> PR #298 wiring.

---

## 7. Source proposals

This design consolidates and supersedes `simx_v3_proposal.md` and
`sst_simx_v3_proposal.md` (now removed from `docs/proposals/`). The
framework reference is [`docs/simobject.md`](../simobject.md).

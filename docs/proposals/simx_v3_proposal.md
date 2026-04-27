# SimX v3 — TLM-Aligned Refactoring Proposal

**Date:** 2026-04-25
**Status:** In progress (Phases 1–4 complete; Phase 5 redirected)
**Author:** Blaise Tine

### Update history

- **2026-04-26** — Phase 5 ("Functional-only mode") cancelled. Replaced with a
  new Phase 5 that completes the TLM data path: caches and memory carry actual
  data through `MemReq`/`MemRsp` rather than running a shadow backing store
  alongside a tag-only timing model. This also closes the remaining gap with
  §3.3 (functional + timing in one place). Side effects: `execute()` returns
  to private on every FuncUnit (the public promotion in §2.6 was a Phase 5
  preparation that is no longer needed); `CacheSim`/`MemSim` are renamed to
  `Cache`/`Memory` to reflect that they are now first-class TLM modules, not
  instrumentation.

---

## 1. Summary

SimX has organically evolved into a hybrid functional/timing simulator built on
top of `SimObject` + `SimChannel`. The infrastructure already provides ~80% of
what a clean Transaction-Level Model (TLM) simulator needs: typed channels with
backpressure, a timing-wheel scheduler with pool-allocated events, and a uniform
`tick()`/`reset()` lifecycle. What is missing is **layering discipline**:
functional semantics still live inside the `Emulator` god class; execution units
behave as timing-only forwarders rather than owning their behavior; the memory
image is held outside the channel-based memory hierarchy; and there is no
sustainable contract for adding new extensions.

This proposal consolidates SimX onto a TLM-aligned design in which each RTL
module is mirrored by one C++ class, each execution unit owns both its
functional and timing behavior, and the memory hierarchy carries data — not just
timing tokens — so the cache and memory modules behave as silicon does. The
change is delivered as sequential phases, each independently shippable and
validated against CSV trace diffs.

---

## 2. Motivation

### 2.1 What works today

- `SimObject` ([sim/common/simobject.h](../sim/common/simobject.h)) provides a
  lightweight (vs. SystemC) base class with statically-registered ticks, pool
  allocation for events, and a 4096-bucket timing wheel.
- `SimChannel<Pkt>` is typed, supports backpressure (`full()`, `try_send`),
  delayed delivery (`send(pkt, delay)`), bind-with-conversion, and a
  `tx_callback` hook for instrumentation.
- The memory hierarchy (`MemCoalescer` → `CacheSim` → `MemSim`) is already
  channel-based.
- `Core`, `Cluster`, `Socket`, `CacheSim`, `MemSim`, `MemCoalescer`,
  `Dispatcher`, `Operands`, `VecUnit`, `TensorUnit`, all `FuncUnit` subclasses,
  and the various TFifo/Arbiter helpers all derive from `SimObject`.

### 2.2 Where the design is breaking down

1. **Functional execution lives in `Emulator`, not in units.**
   `Emulator::execute()` ([sim/simx/execute.cpp](../sim/simx/execute.cpp)) is a
   ~1000-line switch that owns ALU/FPU/SFU/branch/CSR semantics and synchronously
   calls into `VecUnit::execute()` and `TensorUnit::wmma()`. Adding a new
   instruction requires edits in two unrelated files.
2. **`AluUnit`/`FpuUnit`/`SfuUnit::tick()` are timing-only forwarders.** The
   trace's result is computed at issue time inside `Emulator::execute()`; the
   unit's `tick()` only models pipeline latency. This breaks the conceptual
   parity with `VecUnit`/`TensorUnit`, where the unit owns its semantics.
3. **The memory image is in `Emulator`, not in the memory hierarchy.** LSU
   bypasses the channel-based cache hierarchy for *data values*
   ([sim/simx/func_unit.cpp:188–365](../sim/simx/func_unit.cpp#L188)) by calling
   `core_->mem_read/mem_write` directly. The cache hierarchy carries timing
   only. This produces the "orthogonal cache vs. emulator" problem with no
   single source of truth for memory state.
4. **The cache and memory simulators carry no data.** Even after extracting
   `MemBackend`, the cache hierarchy still only models tags, MSHRs, and
   replacement state; data is read from / written to `MemBackend` synchronously
   by the LSU. Functional and timing therefore meet in two places (LSU's call
   into `MemBackend` and the LSU's req/rsp through the cache), violating §3.3.
   Coherency, sub-line writes, and AMO ordering cannot be modeled where they
   actually happen.
5. **`Emulator` is a god object** holding architectural state, the decoder, the
   ISA execute switch, the warp scheduler, dispatch into vec/tcu/lsu, barrier
   logic, and CTA management — four concerns in one class.

### 2.3 Why now

- The recent block-fetch optimization, ibuffer backpressure fix, and tensor
  core work have all required edits across the
  `Emulator`/`FuncUnit`/`Core`/`Cache` boundary. Each new extension multiplies
  the cross-cutting cost.
- WGMMA v3 deadlock investigations exposed how hard it is to reason about the
  current dispatch path because functional and timing decisions are interleaved
  across files.
- Adding RTL ↔ SimX correspondence for divergence debugging
  ([docs/proposals/simx_rtl_perf_divergence.md](proposals/simx_rtl_perf_divergence.md))
  is hampered by the lack of a 1:1 module mapping.

---

## 3. Design Principles

### 3.1 Mirror the RTL

Every RTL module corresponds to exactly one C++ class. The names match the SV
hierarchy. A reader who knows the RTL can find the corresponding model in one
step, and CSV trace divergence can be traced to a specific module instance.

### 3.2 Four-layer separation

```
┌────────────────────────────────────────────────────────────────┐
│  Orchestration   — Processor::tick(), top-level wiring         │
├────────────────────────────────────────────────────────────────┤
│  Modules         — Fetch, Decode, Issue, AluUnit, FpuUnit,     │
│                    LsuUnit, VecUnit, TcuUnit, Cache, Memory    │
│                    (mirror RTL, derive SimObject)              │
├────────────────────────────────────────────────────────────────┤
│  Execute methods — private to each module, synchronous,        │
│                    called only from the module's own tick()    │
├────────────────────────────────────────────────────────────────┤
│  Module-local    — Per-warp regs in OpcUnit, PC/tmask/fcsr in  │
│  state             Scheduler, CSRs in CsrUnit, line data in    │
│                    Cache, RAM image in Memory                  │
└────────────────────────────────────────────────────────────────┘
```

Layer rules (load-bearing):

- **State lives where it's used.** No god-object data container. Each module
  owns the state it operates on; access from outside is through the channel
  protocol (or, for accessors that have no timing, a direct method call into
  the owning module).
- **Each module owns its own `execute()`** method, **private**, called only
  from that module's own `tick()`. No shared `isa::` namespace.
- **Modules communicate only through channels** for any traffic that has
  timing. No direct cross-module method calls on the data path.

### 3.3 Functional and timing meet in exactly one place

Inside each unit's `tick()`, at the moment a new trace is accepted from the
input channel. `execute()` is a private method called from that one site. No
separate event, no scheduling of execute, no second loop.

For the memory hierarchy specifically, this means **data flows on the same
packets that carry timing**. A load is a `MemReq` that returns a `MemRsp` with
data; a store is a `MemReq` carrying data. There is no shadow backing store
that the LSU consults out-of-band. The cache holds line data; `Memory` holds
the RAM image; and the answer to "what byte is at address X" is whatever the
hierarchy resolves through the channels.

### 3.4 The channel is the pipeline

The channel's `pending_count_` already tracks in-flight packets toward the
`full()` capacity check. Units do not maintain their own `pipeline_` deques or
`cycles_left` counters. Latency is expressed by the `delay` argument to
`send()`; backpressure is expressed by output channel capacity.

### 3.5 No premature abstraction

- No `isa::` namespace. Behavior lives on the unit; nothing else needs ISA
  semantics directly.
- No SystemC. The existing `SimObject` infrastructure is lighter and faster.
- No multi-threaded tick scheduler. Single-threaded with a tick-then-event loop
  is simpler and fast enough for Vortex's object count.
- No fast-functional driver. A separate functional-only mode was considered
  (and originally proposed in §4.5) but cancelled in 2026-04-26: it requires
  promoting `execute()` to public, encourages divergence between two paths,
  and pays for itself only on workloads where pure correctness — not timing —
  is the goal. Once the TLM data path lands, the same code path serves both.

---

## 4. Architecture

### 4.1 Module pattern

Every execution unit follows this skeleton:

```cpp
// alu_unit.h
class AluUnit : public FuncUnit {
public:
  AluUnit(const SimContext& ctx, const std::string& name)
    : FuncUnit(ctx, name)
    , output_(this, /*capacity=*/PIPE_DEPTH) {}

  void tick() override;

private:
  void execute(instr_trace_t* t);     // private; called only from tick()
  uint32_t latency_of(AluOp op) const;
};
```

```cpp
// alu_unit.cpp
void AluUnit::tick() {
  if (input_.empty() || output_.full()) return;
  auto* t = input_.peek();
  input_.pop();
  this->execute(t);
  output_.send(t, latency_of(t->op));
}

void AluUnit::execute(instr_trace_t* t) {
  switch (t->op) {
    case AluOp::ADD: t->result = t->src1 + t->src2; break;
    case AluOp::SUB: t->result = t->src1 - t->src2; break;
    case AluOp::AND: t->result = t->src1 & t->src2; break;
    /* ... */
  }
}
```

Two observations:

- **Pipelined units** (ALU, FPU, SFU) need no internal state. Output channel
  capacity ≥ 1 admits multiple in-flight ops; the channel's `pending_count_`
  enforces the limit.
- **Non-pipelined units** (TCU, divider) get the same skeleton with output
  capacity = 1. The capacity-1 channel is the resource lock; the unit naturally
  stalls until the previous op clears.

`execute()` is **private** on every FuncUnit. The unit's own `tick()` is the
only caller. Cross-module calls go through `SimChannel`.

### 4.2 LSU — the complex case

LSU is the only unit that joins multiple asynchronous responses per
instruction. There is no synchronous backing store — load data arrives in the
`MemRsp`, store data is packed into the `MemReq`:

```cpp
// lsu_unit.h
class LsuUnit : public FuncUnit {
public:
  SimChannel<MemReq>  mem_req_out [NUM_THREADS];
  SimChannel<MemRsp>  mem_rsp_in  [NUM_THREADS];

  void tick() override;

private:
  struct Inflight { instr_trace_t* trace; uint32_t pending; };
  std::unordered_map<uint32_t, Inflight> inflight_;
  uint32_t next_id_ = 0;

  uint64_t addr_of(const instr_trace_t* t, uint32_t tid) const;
  MemOp    memop_of(const instr_trace_t* t) const;
};
```

```cpp
// lsu_unit.cpp
void LsuUnit::tick() {
  // Phase 1: drain timing responses; copy load/AMO-return data out of the
  // response packet into the trace's dst_data.
  for (uint32_t tid = 0; tid < NUM_THREADS; ++tid) {
    while (!mem_rsp_in[tid].empty()) {
      auto rsp = mem_rsp_in[tid].peek();
      mem_rsp_in[tid].pop();
      auto it  = inflight_.find(rsp.tag);
      auto& f  = it->second;
      if (memop_of(f.trace) == MemOp::READ ||
          memop_of(f.trace) >= MemOp::AMO_FIRST) {
        copy_word(&f.trace->dst_data[rsp.thread_id], rsp.data,
                  addr_of(f.trace, rsp.thread_id), f.trace->access_size);
      }
      if (--f.pending == 0 && !output_.full()) {
        output_.send(f.trace, 1);
        inflight_.erase(it);
      }
    }
  }

  // Phase 2: issue new instruction (atomic across all per-thread lanes)
  if (input_.empty() || inflight_.size() >= MAX_INFLIGHT) return;
  auto* t = input_.peek();
  for (uint32_t tid = 0; tid < NUM_THREADS; ++tid) {
    if (t->tmask[tid] && mem_req_out[tid].full()) return;
  }
  input_.pop();

  uint32_t active = popcount(t->tmask);
  if (active == 0) { output_.send(t, 1); return; }

  uint32_t id = next_id_++;
  inflight_.emplace(id, Inflight{t, active});
  for (uint32_t tid = 0; tid < NUM_THREADS; ++tid) {
    if (!t->tmask[tid]) continue;
    MemReq req;
    req.addr      = addr_of(t, tid);
    req.size      = t->access_size;
    req.op        = memop_of(t);
    req.tag       = id;
    req.thread_id = tid;
    if (req.op != MemOp::READ) {
      // Stores and AMO src arrive packaged in the request.
      req.data   = std::make_shared<mem_block_t>();
      req.byteen = byteen_for(req.addr, req.size);
      copy_word_to_block(req.data, &t->src_data[tid], req.addr, req.size);
    }
    mem_req_out[tid].send(req, 1);
  }
}
```

Loads and AMOs return their data via `MemRsp::data`. Stores deposit data into
`MemReq::data` (with byte enables for sub-line writes). The cache hierarchy
performs the actual read/write/RMW; the LSU is unaware of where in the
hierarchy the request is satisfied.

### 4.3 Memory model

**Caches hold data; `Memory` holds the RAM image.** Every load/store/AMO flows
through `MemReq`/`MemRsp` — the same packets carry timing *and* data.
Functional and timing meet at the LSU's req/rsp boundary, not in two places.
There is no shadow backing store.

```cpp
using mem_block_t = std::array<uint8_t, MEM_BLOCK_SIZE>;

struct MemReq {
  uint64_t                     addr;
  uint32_t                     size;
  MemOp                        op;        // READ / WRITE / AMO_*
  uint32_t                     tag;       // instruction id for response join
  uint32_t                     thread_id;
  uint64_t                     byteen;    // sub-line write enables (stores/AMOs)
  std::shared_ptr<mem_block_t> data;      // populated for stores/AMO src
};

struct MemRsp {
  uint32_t                     tag;
  uint32_t                     thread_id;
  std::shared_ptr<mem_block_t> data;      // populated for loads/AMO returns
};
```

`shared_ptr<mem_block_t>` lets MSHR-coalesced replays share a single fill
buffer without copy. The cache lifts a single `mem_block_t` from the response
of the level below and hands it to all coalesced consumers; per-line storage
is one `shared_ptr<mem_block_t>` per (set, way) entry.

```
LOAD:   LSU → MemReq{op=READ}  → Cache → (hit: serve from line) /
                                          (miss: forward to Memory, fill on rsp)
              ← MemRsp{data}    ← Cache (data sourced from cache line)
              LSU copies word from rsp.data into dst_data, retires.

STORE:  LSU → MemReq{op=WRITE, data, byteen} → Cache
              (write-back: byteen-merge into line, mark dirty;
               write-through: also forward to Memory)
              ← MemRsp           ← Cache (write-response, optional)
              LSU retires.

AMO_X:  LSU → MemReq{op=AMO_X, data, byteen} → Cache → Memory (RMW at arrival)
              ← MemRsp{data}                  ← Cache ← Memory
              LSU copies prior value into dst_data, retires.
```

The cache hierarchy updates its tag/dirty/MSHR state *and* the line bytes as
packets transit. AMOs land at `Memory` — RMW order is **arrival order**, not
issue order, which matches RTL.

`Memory` is the single owner of the RAM image at the bottom of the hierarchy.
Last-level cache misses pull data from `Memory`; writebacks deposit data into
`Memory`. Above the cache, no module reads or writes the RAM image directly.

I/O / non-cacheable regions stay on the bypass arbiter path that the cache
already implements; bypass requests skip the data store and forward directly
to `Memory`.

### 4.4 Channel-based timing model

The canonical pattern is:

| Unit kind                   | Output capacity | Send pattern                | State needed |
| --------------------------- | --------------- | --------------------------- | ------------ |
| Pipelined (ALU/FPU/SFU)     | ≥ 1             | `output_.send(t, latency)`  | none |
| Non-pipelined (TCU)         | 1               | `output_.send(t, latency)`  | none |
| Variable response (LSU)     | ≥ 1             | per-lane req out + join     | `inflight_` map |

**Tick-order nuance for non-pipelined units.** Because `pending_count_` is
released when the event fires (end of tick at `cycles_+latency`), and the
consumer cannot pop until *its* tick runs, throughput on a capacity-1 channel
is 1/(N+1), not 1/N. For variable-latency units this off-by-one is in the
noise; for cycle-accurate fixed-latency modeling, send with `delay=N-1` or
register the consumer before the producer in `SimPlatform::objects_`.

---

## 5. Migration Plan

The refactoring is delivered in **six phases**. Each phase is independently
shippable, bisectable, and validated against CSV trace diffs against the
previous tip.

**Status as of 2026-04-26:**

| Phase | Status |
|-------|--------|
| 1 — Per-unit file split | ✅ done |
| 2 — Migrate ISA execution into units | ✅ done |
| 3 — Unified memory path (`MemBackend`) | ✅ done |
| 4 — Demote Emulator → per-module state (no ArchState) | ✅ done |
| 5 — TLM data path (caches/memory carry data) | ⏳ next |
| 6 — Documentation lock-in | ⏳ pending |

Phase 4 was redesigned during execution: rather than introducing an
`ArchState` data container, state was distributed to the modules that own it
(PC/tmask/fcsr/IPDOM in `Scheduler`; per-warp regfiles partitioned across
`OpcUnit` per RTL `VX_opc_unit.sv`; CSRs in `CsrUnit` as a new FuncUnit).
`Decoder`, `Scheduler`, `CtaDispatcher`, `Operands`, `CsrUnit`, and the
per-issue `OpcUnit`s are all SimObjects participating in the standard tick
lifecycle.

### Phase 1 — Per-unit file split (Week 1, very low risk)

**Goal:** Establish per-unit file structure without changing semantics.

| Step | Action | Files |
|------|--------|-------|
| 1.1 | Create `sim/simx/alu_unit.{h,cpp}` and move `AluUnit` from `func_unit.{h,cpp}` | new files |
| 1.2 | Same for `fpu_unit`, `sfu_unit`, `lsu_unit` | new files |
| 1.3 | Update `sim/simx/CMakeLists.txt` and any Makefiles | build files |
| 1.4 | Delete now-empty per-unit code from `func_unit.cpp`; keep `FuncUnit` base in `func_unit.h` | edit |
| 1.5 | Run `../configure` from build dir, full regression | validation |

**Validation:** All CI regressions pass. CSV traces byte-identical to baseline.

**Why first:** Pure code motion. No semantic change. Establishes the file
layout that subsequent phases will populate.

### Phase 2 — Migrate ISA execution into units (Weeks 2–3, medium risk)

**Goal:** Each FuncUnit owns its own `execute()`. `Emulator::execute()` shrinks
to a dispatcher.

| Step | Action | Validation gate |
|------|--------|-----------------|
| 2.1 | Add `void AluUnit::execute(instr_trace_t*)` (initially private). Move ALU cases from `Emulator::execute()` into it. Modify `AluUnit::tick()` to call `execute()` from the ACCEPT step instead of relying on pre-filled `result`. Remove ALU case from `Emulator::execute()`. | CSV diff after each unit |
| 2.2 | Same for `FpuUnit`. Pay attention to FPU rounding-mode CSR access. | CSV diff |
| 2.3 | Same for `SfuUnit`. Branch and CSR ops decision: keep in a `BranchUnit`/`CsrUnit` if they have natural latency, else inline in Issue. | CSV diff |
| 2.4 | Invert `VecUnit`: instead of `Emulator::execute()` calling `VecUnit::execute()` synchronously, have `VecUnit::tick()` consume from input channel and call its own `execute()` in ACCEPT. | CSV diff on vector regressions |
| 2.5 | Same for `TensorUnit`. Watch for the WGMMA CTA-limit interaction documented in `project_wgmma_cta_limit`. | CSV diff on TCU regressions |
| ~~2.6~~ | ~~Promote each unit's `execute()` to public~~ — **reverted in Phase 5.2.** Phase 5 does not need a public `execute()`; the original Phase 5 (functional-only mode) was cancelled. | n/a |

**Risk:** Operand-timing bugs. The trace's source data must be available when
the unit executes. Since execute now runs at ACCEPT (not at issue), the
existing Operands SimObject must still fully populate `src_data` before the
trace enters the unit's input channel. Verify with directed tests.

### Phase 3 — Unified memory path (Weeks 4–5, highest risk)

**Goal:** Eliminate `Emulator::mem_read/mem_write`. All memory access flows
through `MemReq`/`MemRsp` to a `MemBackend` SimObject that owns the image.

| Step | Action | Validation gate |
|------|--------|-----------------|
| 3.1 | Extend `MemReq` with `op` (READ/WRITE/AMO_*) and `thread_id`. Extend `MemRsp` with `thread_id`. **No data field.** Update all packet producers/consumers. | build green, no semantic change yet |
| 3.2 | Create `MemBackend` SimObject. Move the memory image storage from `Emulator` into it. Expose `read/write/atomic` methods. `Emulator::mem_read/mem_write` become thin wrappers calling into `MemBackend` (preserves API for the duration of this phase). | functional regression unchanged |
| 3.3 | Add `LsuUnit::do_backend(trace, tid)` helper. Modify `LsuUnit::tick()`: at issue, for stores/AMOs call `do_backend()` directly; for all ops send `MemReq` for timing; on `MemRsp` for loads, call `do_backend()` to fill `dst_data`. Stop calling `core_->mem_read/mem_write`. | LSU regressions unchanged |
| 3.4 | Validate atomics: confirm `MemBackend::atomic` is RMW-atomic w.r.t. concurrent calls from different warps in the same cycle. Run AMO stress tests. | AMO tests pass |
| 3.5 | Delete `Emulator::mem_read/mem_write` wrappers. | build green |

**Risk landmines:**

- Atomics ordering across warps within a CTA.
- Vector load/store interaction with WGMMA (per `project_wgmma_cta_limit`).
- Sub-line writes (byte enables / write masks).
- Uncached / IO regions if any exist in current Vortex map.

**Mitigation:** Run Phases 3.1–3.3 with the existing `LsuUnit` still using the
`Emulator::mem_read/mem_write` wrappers. Switch the LSU last (3.4) so the
backend infrastructure is well-tested before the consumer migrates.

### Phase 4 — Decompose Emulator (Week 6, low risk) — **redesigned**

**Goal:** Eliminate the `Emulator` god class. State moves to the modules that
own it; no central data container.

| Step | Action |
|------|--------|
| 4.1 | Move warp lifecycle (PC/tmask/fcsr/IPDOM, barriers, CTA dispatch) into `Scheduler` SimObject. |
| 4.2 | Move `decode()` into a `Decoder` SimObject. Stateless w.r.t. simulation; tick is a no-op (mirrors `VX_decode.sv`). |
| 4.3 | Split per-warp register files into `OpcUnit` SimObjects, one per issue lane × NUM_OPCS, per RTL `VX_opc_unit.sv`. `Operands` routes (lane, opc) and owns the OpcUnits. |
| 4.4 | Add `CsrUnit` as a new FuncUnit (`FUType::CSR`). All CSR reads/writes (FCSR, FFLAGS, FRM, MSCRATCH, CTA CSRs, MPM perf counters, SATP) live here. SFU loses CSR. |
| 4.5 | Make `Decoder`, `CtaDispatcher`, `Scheduler`, `Operands`, `OpcUnit`, `CsrUnit` all SimObjects with their own `tick()`. |
| 4.6 | Delete `emulator.{h,cpp}`. |

**Validation:** Pure refactor — CSV traces unchanged across each chunk.

### Phase 5 — TLM data path (replaces "Functional-only mode")

**Goal:** Caches and memory carry actual data. `MemReq`/`MemRsp` become
silicon-faithful TLM packets. The shadow `MemBackend` is dissolved into
`Memory`. After this phase the simulator has a single source of truth for
memory state, residing wherever the hierarchy currently holds it.

| Step | Action | Validation gate |
|------|--------|-----------------|
| 5.1 | Rename `CacheSim` → `Cache` (file `cache.{h,cpp}`); `MemSim` → `Memory` (file `memory.{h,cpp}`). Pure rename. | build green, CSV identical |
| 5.2 | Make `execute()` private on every FuncUnit (revert §2.6 promotion). | build green |
| 5.3 | Extend `MemReq` with `shared_ptr<mem_block_t> data` + `byteen` for stores/AMOs. Extend `MemRsp` with `shared_ptr<mem_block_t> data` for loads/AMO returns. Producers/consumers updated. Cache still doesn't read the data — semantics unchanged. | build green, CSV identical |
| 5.4 | `Cache`: add per-(set, way) line-data storage (`shared_ptr<mem_block_t>`). On read hit: copy line data into the response. On read miss: capture fill data from the `MemRsp` of the level below. On write: byteen-merge data into the line (write-back) or also forward (write-through). LSU continues to consult `MemBackend` synchronously — both produce the same answer; this is the safety net that lets us validate cache data correctness. | LSU regressions still pass; new assertion: cache-returned data matches `MemBackend`-returned data on every load |
| 5.5 | `Memory`: own the RAM image. On read req from above: read from RAM into `MemRsp`. On write req: apply byteen to RAM. (`MemBackend` is now redundant.) | functional regressions identical |
| 5.6 | LSU: stop calling `MemBackend` synchronously. Loads consume from `MemRsp::data`. Stores package `src_data` into `MemReq::data`. AMOs send op+src, receive prior value via response. | full regression suite, CSV diff against pre-5.6 |
| 5.7 | Delete `MemBackend`. CSR/decoder pieces that needed the image directly route through `Memory` or are absorbed by it. | build green |
| 5.8 | Validate atomics. RMW now happens at `Memory` (where requests arrive) — RTL-faithful. | AMO stress tests pass |

**Risk landmines:**

- `mem_block_t` lifetime through `SimChannel`: use `shared_ptr` so MSHR-coalesced
  replays share a single fill buffer. Never copy block bytes when a pointer-bump
  suffices.
- Sub-line writes: byte-enables must be plumbed through every cache hop.
- Concurrent same-cycle AMOs to the same address: serialized by arrival order
  at `Memory`; `Memory::tick()` must process its input channels deterministically
  (existing arbiter ordering is sufficient).
- I/O / non-cacheable bypass path (existing `nc_mem_arbs_`) stays unchanged
  end-to-end.
- Cache initialization cycles, MSHR coalescing, and writeback paths now all
  carry data; each needs its own validation pass.

**Mitigation:** Step 5.4 is the load-bearing one. Run it with a parallel-path
assertion (LSU compares cache-returned data to `MemBackend`-returned data). Only
once 5.4 is green do we cut the synchronous `MemBackend` path in 5.6.

### Phase 6 — Documentation and lock-in

| Step | Action |
|------|--------|
| 6.1 | Promote this proposal to `docs/simx_architecture.md` (active reference, not proposal). |
| 6.2 | Write `docs/simx_extension_guide.md` — "to add a unit, subclass `FuncUnit`, implement `tick()` per the canonical pattern, implement `execute()`, register your op_types." Include `AluUnit` as worked example. |
| 6.3 | Update `docs/codebase.md` to reflect the new module structure. |
| 6.4 | Update `docs/microarchitecture.md` cross-references where applicable. |
| 6.5 | Optional: add a lint rule or CI check that rejects new direct cross-module method calls (only `SimChannel::send/peek/pop` and `execute()` allowed). |

---

## 6. Risks and Tradeoffs

### 6.1 What this gives up

- **No fast-functional driver.** Workloads that only need correctness still
  pay full timing-mode cost. Acceptable: most Vortex regressions are short
  enough that timing cost is dominated by other factors, and a separate
  functional-only path would diverge from the timing path over time.
- **Per-cycle tick over inactive objects.** SimX continues to pay O(N_objects)
  per cycle even when most are idle. For Vortex's object count this is cheap;
  scaling beyond a few thousand objects would require a sleep-until-input
  optimization.
- **Memory cost of cache data.** Each cache line entry now holds a
  `shared_ptr<mem_block_t>` (typically 64 B/line). For a multi-MB simulated
  cache hierarchy this is bounded host RAM, not a concern at current
  configurations. `shared_ptr` lets a single fill buffer be shared across
  MSHR-coalesced replays without copy.

### 6.2 What can go wrong

- **Operand timing in Phase 2.** Moving execute from issue-time to
  unit-accept-time changes when source data is read. The Operands SimObject
  must populate the trace fully before the trace enters the unit channel; any
  late-binding pattern will surface as wrong-result bugs. (Mitigated; Phase 2
  shipped with no observed regressions.)
- **Atomics in Phase 5.** Multi-warp contended atomics are the most subtle
  case. With the TLM data path RMW happens at arrival to `Memory`; verify
  with a dedicated stress test (e.g., concurrent `atomicAdd` on a single
  address) before declaring Phase 5 complete.
- **Sub-line writes / byte enables.** Stores narrower than a cache line must
  apply byteen at every hop (write-back into dirty line, write-through to
  `Memory`). One direct test per store width.
- **Cache fill data correctness.** Step 5.4 introduces a parallel path
  (cache returns data while LSU still hits `MemBackend`); a comparison
  assertion guards correctness during the transition. Do not skip 5.4's
  assertion — it is the entire point of running the parallel paths.
- **Channel ordering for variable-latency ops.** A unit that issues with
  variable `delay` per op will see responses arrive out of order. For LSU this
  is fine (re-joined via `inflight_` tag). For ALU `DIV` this means a later
  short op can finish before an earlier `DIV`. Match RTL semantics by checking
  whether the existing scoreboard assumes in-order completion.
- **Tick-order off-by-one for non-pipelined units.** Per §4.4. Acceptable for
  research-grade modeling; document in the unit pattern guide.

### 6.3 Reversibility

Phases 1–4 are landed and reversible only by reverting their commits. Phase 5
is the largest single commitment of the project: once `MemBackend` is deleted
in 5.7, going back means re-introducing a synchronous data path. Mitigation:
the parallel-path assertion in 5.4 means we have data-path correctness proven
for several commits before the synchronous path is removed.

---

## 7. Out of Scope

Explicitly **not** addressed by this proposal:

- **Multi-threaded simulation.** Single-threaded tick + events remains.
- **SystemC adoption.** `SimObject` is retained as the base infrastructure.
- **Speculative execution / O3 modeling.** Vortex is in-order; the design
  keeps that assumption.
- **Cache coherency protocol modeling.** Caches will hold per-line data and
  dirty state after Phase 5, so adding MESI/MOESI later is now an additive
  change rather than a structural rework. Not in scope for this proposal.
- **Sub-cycle (event-driven) timing for the pipeline.** Pipeline stays
  tick-based; only memory delays remain event-scheduled.
- **Fast-functional / functional-only driver.** Considered and rejected — see
  §3.5. The same hierarchy serves both modes after Phase 5 ships; a separate
  functional driver buys speed at the cost of an alternate code path that
  drifts.

---

## 8. References

- [sim/common/simobject.h](../sim/common/simobject.h) — SimObject, SimChannel,
  SimPlatform implementation.
- [sim/simx/core.cpp](../sim/simx/core.cpp) — current pipeline driver
  (`Core::tick()` and stage methods).
- [sim/simx/emulator.h](../sim/simx/emulator.h),
  [sim/simx/execute.cpp](../sim/simx/execute.cpp) — current `Emulator` class
  and the ~1000-line `execute()` switch to be migrated.
- [sim/simx/func_unit.cpp](../sim/simx/func_unit.cpp) — current FuncUnit
  implementations (Phase 1 source).
- [sim/simx/vec_unit.h](../sim/simx/vec_unit.h),
  [sim/simx/tensor_unit.h](../sim/simx/tensor_unit.h) — units that already
  partially follow the proposed pattern.
- [sim/simx/cache_sim.h](../sim/simx/cache_sim.h) (renamed to `cache.h` in
  Phase 5.1), [sim/simx/mem_sim.h](../sim/simx/mem_sim.h) (renamed to
  `memory.h` in Phase 5.1) — memory subsystem.
- `m151b/project3/gold_v2/src/{cache,memory}.{h,cpp}` — reference TLM-style
  cache+memory implementation. Inspires the data-carrying `MemReq`/`MemRsp`
  shape used in Phase 5.
- [docs/proposals/simx_rtl_perf_divergence.md](proposals/simx_rtl_perf_divergence.md)
  — recent SimX vs. RTLsim divergence analysis; motivates the RTL-mirror goal.
- [docs/coding_guidelines_cpp.md](coding_guidelines_cpp.md) — applies to all
  new code in this refactor.

### Comparison sims

- **gem5** — separate ISA description language; AtomicSimpleCPU /
  TimingSimpleCPU / O3CPU share one ISA. SimX v3 takes the spirit (clean
  layering, dual-mode driver) without the cost (no separate ISA DSL).
- **GPGPU-Sim** — originally split functional (`cuda-sim`) and timing
  simulators; merged over a decade because the dual-state model was too
  expensive to maintain. SimX v3 starts from the merged shape.
- **Accel-Sim** — trace-driven (functional pass dumps a trace, timing model
  consumes). Faster but loses fidelity for timing-dependent control flow. Not
  adopted because Vortex's execution-driven simulation is required for
  validation against RTL.
- **SST** — strict component+link separation, scales to many-node systems.
  Heavier than needed for single-chip GPGPU modeling.

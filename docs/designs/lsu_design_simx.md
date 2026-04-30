# Vortex LSU Design (SimX)

**Scope:** logical/architectural design embodied by the SimX LSU
([sim/simx/lsu_unit.cpp](../../sim/simx/lsu_unit.cpp) +
[sim/simx/lsu_unit.h](../../sim/simx/lsu_unit.h)). The implementation is
C++; the design itself is implementation-agnostic. Read alongside
[lsu_design_rtl.md](lsu_design_rtl.md) — this doc highlights where the
SimX design **deliberately diverges** from RTL and where it merely
**simplifies** the RTL design without changing semantics.

---

## 1. Architecture overview

```
   Inputs[b] (per LSU block)              LsuUnit::on_tick()              core_->lmem_switch(b)
   ─────────────────────────►   ┌──────────────────────────────────┐  ───────────────────────►
                                │  process_response(b)             │
                                │    └─ pending_rd_reqs lookup     │
                                │       └─ load formatter          │
                                │  process_request(b)              │
                                │    ├─ AGU (compute_addrs)        │
                                │    ├─ fence lock check           │
                                │    ├─ pending_rd_reqs full?      │
                                │    └─ batch dispatch (NUM_LSU_LANES) │
                                └──────────────────────────────────┘
                                                                              ▲
   Outputs[b] (writeback)                                                     │
   ◄────────────────────────────────────────────────────────────────  RspOut  │
```

The SimX LSU is one unit instance per LSU block (`NUM_LSU_BLOCKS`).
Per-block it carries:

- `pending_rd_reqs` — `HashTable<pending_req_t>` of size `LSUQ_IN_SIZE`
  (the SimX equivalent of the RTL index buffer).
- `addr_list` + `remain_addrs` — drained `NUM_LSU_LANES` per tick (the
  SimX equivalent of RTL's vector→channel batching).
- `fence_trace` + `fence_lock` — total fence per block.

`on_tick()` runs `process_response(b)` then `process_request(b)` for each
block. Communication with memory is exclusively via `core_->lmem_switch(b)`
(per Rule 1 of [feedback_simx_perf_goal](../../) — NoC-only memory access,
no functional bypass).

---

## 2. Per-tick stages

### 2.1 `process_request(b)` — frontend + dispatch

1. **Fence drain.** If `fence_lock` is set, wait for `pending_rd_reqs` to
   empty, then forward the fence trace and unlock
   ([:192-200](../../sim/simx/lsu_unit.cpp#L192-L200)). Total per-block
   barrier — same semantic as the RTL `fence_lock`.
2. **AGU on first sight.** When `remain_addrs == 0`, `compute_addrs(b,
   trace)` populates `state.addr_list` with one
   `mem_addr_size_t` per active thread:
   `addr = rs1 + stride * rs2 + offset`
   ([:89-120](../../sim/simx/lsu_unit.cpp#L89-L120)). Subsequent dispatches
   of the same trace re-use the cached `addr_list`.
3. **Batch dispatch.** Take the next `NUM_LSU_LANES` entries from
   `addr_list`, package into one `LsuReq`, allocate a `pending_rd_reqs`
   slot for reads, and send to `lmem_switch.ReqIn`. Decrement
   `remain_addrs`; pop the trace from `Inputs[b]` only when fully drained
   ([:250-311](../../sim/simx/lsu_unit.cpp#L250-L311)).

### 2.2 `process_response(b)` — completion + writeback

Pulls one `LsuReq` response per tick from `lmem_switch.RspOut`; looks up
the `pending_rd_reqs` entry by tag; per active lane in the response mask
copies the line bytes at the lane's offset, format-shifts/sign-extends per
RISC-V load semantics, OR-merges into `trace->dst_data[tid]` with NaN-box
for float loads. Decrements `entry.count`; releases the slot and forwards
the trace to `Outputs[b]` when `count == 0 && entry.eop`
([:122-188](../../sim/simx/lsu_unit.cpp#L122-L188)).

### 2.3 PACK macro-uop expansion

`LsuUopGen` in [lsu_unit.h:24-34](../../sim/simx/lsu_unit.h#L24-L34) and
[lsu_unit.cpp:30-72](../../sim/simx/lsu_unit.cpp#L30-L72) expands a
`PACKLB.F` / `PACKLH.F` macro instruction into 4 / 2 sub-load uops, each
with `stride = uop_index` so the AGU formula becomes
`rs1 + uop_index * rs2`. Each uop's `dst_bytesel` selects which bytes of
the destination float register receive the loaded value, and the writeback
path OR-merges. This is one of the few places where SimX models a
**software-visible** expansion that the RTL handles differently (RTL uses
the multi-PID tracker — see RTL doc §2.3).

---

## 3. Memory-level parallelism mechanisms

These are the design-level MLP knobs SimX preserves from the RTL design:

1. **`pending_rd_reqs` hash table.** Same role as the RTL index buffer:
   per-block capacity `LSUQ_IN_SIZE` non-blocking outstanding loads;
   each response carries the slot tag and self-routes back. Out-of-order
   responses are first-class (the table is a hash, not a queue).
2. **Read/write decoupling.** Stores skip `pending_rd_reqs.allocate()`
   ([:285-288](../../sim/simx/lsu_unit.cpp#L285-L288)) — only loads
   consume slots. A store-heavy stream does not back up against a full
   read pending table.
3. **Per-block independence.** `NUM_LSU_BLOCKS` blocks tick concurrently
   in `on_tick()`; each has its own `lsu_state_t` and its own
   `lmem_switch(b)` channel. One block stalled on a slow response does
   not block the others.
4. **Batched dispatch.** A trace with > `NUM_LSU_LANES` active threads
   spans multiple ticks; each tick issues one batch and decrements
   `remain_addrs`. The trace stays at the head of `Inputs[b]` until
   drained, but other blocks proceed.
5. **Lane-mask responses.** `LsuReq::mask` is per-lane; partial responses
   (subset of lanes in one packet) are honoured by `process_response`
   ([:145-178](../../sim/simx/lsu_unit.cpp#L145-L178)). Equivalent to
   RTL's `RSP_PARTIAL = 1` mode.

### Performance ceiling

Per-block peak load issue rate = `1 batch/tick = NUM_LSU_LANES
addrs/tick`. With `LSUQ_IN_SIZE = 8` outstanding and an avg LMEM latency
of `L` ticks: max load throughput = `min(NUM_LSU_LANES, 8 *
NUM_LSU_LANES / L) addrs/tick`. The model lets the user trade `L` against
queue depth to study MLP sensitivity.

---

## 4. Divergences from the RTL design

These are real architectural differences (not just implementation), and
each is intentional for SimX's role as a fast functional + first-order
timing model.

### 4.1 Coalescer placement (LMEM path vs DCache path)

**RTL:** the LSU's own `VX_mem_scheduler` is configured with
`LINE_SIZE = WORD_SIZE` so it doesn't coalesce. The cache subsystem
that sits downstream of the LSU has its own `VX_mem_coalescer` instance
sized by `LSUQ_OUT_SIZE` / `MEM_QUEUE_SIZE` that merges adjacent lanes
hitting the same cache line.

**SimX:** structurally identical — `MemCoalescer`
([sim/simx/mem/mem_coalescer.cpp](../../sim/simx/mem/mem_coalescer.cpp))
is instantiated **per LSU block** between `lmem_switch.ReqOutDC` and
`lsu_dcache_adapter.ReqIn`
([core.cpp:106,162-166](../../sim/simx/core.cpp#L106-L166)), sized by
`LSUQ_OUT_SIZE`, with the same algorithm: group lanes by `addr & ~(LINE-1)`,
emit one merged request per group, replicate response bytes back to all
contributing lanes via a `pending_rd_reqs` HashTable keyed on the merged tag.

Both designs therefore coalesce on the LSU→DCache boundary and **not**
on the LSU→LMEM boundary (LMEM is a banked scratchpad whose bank-row
IS the natural coalesce unit; the LMEM port handles the merge implicitly).

**Implementation differences (not design differences):**

- RTL: pipelined 2-state FSM (`WAIT`/`SEND`) with priority encoder
  over remaining mask each cycle; one outgoing request per cycle.
- SimX: per-tick `on_tick()` with `sent_mask_` tracking; one outgoing
  request per tick. Same batching semantics, just discretized to tick
  granularity.

The earlier draft of this section incorrectly claimed SimX had no
coalescer; that was wrong. The functional design matches RTL on this
boundary.

### 4.2 No multi-PID packet tracker

**RTL:** `VX_lsu_slice` uses a per-slice `VX_allocator` + sop/eop tables
to coalesce multi-PID memory packets back into one logical SOP/EOP for
writeback (RTL doc §2.3).

**SimX:** the macro-uop expander (`LsuUopGen`) splits the macro into
explicit sub-uops in software-instruction form. Each sub-uop has a
distinct UUID and goes through the LSU as an independent load; writeback
OR-merges via `dst_bytesel`. No SOP/EOP tracking is needed at the LSU
level — the sub-uops are independently committed.

**Effect on modeling:** equivalent functional result; different latency
shape. RTL's multi-PID tracker may stall earlier on allocator full; SimX
has no analogous structural limit but issues one uop per tick per block
so it can be slower in the pipelined-throughput sense.

### 4.3 No vector→channel batching infrastructure beyond `NUM_LSU_LANES`

**RTL:** `MEM_BATCHES` parameterized at elab time;
`req_batch_idx_r` cycles through batches with the response carrying the
batch index back ([RTL doc §3.4](lsu_design_rtl.md#34-vector--channel-batching)).
Multiple batches of one request can be in flight.

**SimX:** one batch per tick, sequential drain via `remain_addrs`. The
trace blocks `Inputs[b]` until fully drained. There's no batch-index in
the response tag because each batch has its own `pending_rd_reqs`
allocation.

**Effect on modeling:** SimX's per-trace dispatch is more serialized than
RTL's. For vectors that span N batches, SimX takes N ticks; RTL can
overlap them. This understates MLP for very wide vectors at small
`MEM_CHANNELS` configurations. Mitigation: tune `LSUQ_IN_SIZE` to be
generous so the next trace can start dispatching while the previous one's
late batches are still in flight.

### 4.4 Single-cycle response processing

**RTL:** the response path is fully pipelined; `mem_rsp_valid` can fire
every cycle and the response demux can sustain that rate.

**SimX:** `process_response(b)` consumes at most one `LsuReq` per tick
([:122-124](../../sim/simx/lsu_unit.cpp#L122-L124)). If multiple
responses pile up in `RspOut`, they're drained one per tick.

**Effect on modeling:** under-models burst response throughput. The fix
(loop in `process_response`) is single-line but changes the per-tick cost
profile and would need calibration.

### 4.5 No coalescer-side outstanding count

**RTL:** `MEM_QUEUE_SIZE` is a separate, larger pool (default 8) at the
coalescer; the effective MLP cap = `min(CORE_QUEUE_SIZE,
MEM_QUEUE_SIZE)`.

**SimX:** only `LSUQ_IN_SIZE` matters; there's no second tier of
outstanding tracking.

**Effect on modeling:** SimX caps MLP at `LSUQ_IN_SIZE` even if the
configured RTL would see more outstanding requests via the coalescer
queue. Generally fine because the LSU configuration today doesn't enable
coalescing; matters more for cache-side scheduler comparisons.

### 4.6 Frontend uses an explicit `stride` field

**RTL:** AGU is `rs1 + sext(offset)` only — no per-lane stride. Stride
shows up only via the macro-uop expansion (the dispatcher generates
distinct PIDs with addresses pre-computed by the uop generator).

**SimX:** AGU is `rs1 + stride * rs2 + offset` directly
([:109](../../sim/simx/lsu_unit.cpp#L109)), with `stride` carried in
`IntrLsuArgs`. PACK macro-uop expansion sets `stride = uop_index`
([:69](../../sim/simx/lsu_unit.cpp#L69)).

**Effect on modeling:** purely a representation difference. SimX bakes
the stride into the AGU; RTL bakes it into the address generation upstream
of the LSU. Same arithmetic.

### 4.7 No request-tag width in SimX

**RTL:** `MEM_TAG_WIDTH = UUID_WIDTH + MEM_QUEUE_ADDRW + MEM_BATCH_BITS`
— packed scalar tag flowing through the memory bus.

**SimX:** `LsuReq::tag` is a scalar (`uint32_t`) holding the
`pending_rd_reqs` slot index; the `LsuReq` carries `uuid` separately as
metadata for tracing. No batch-index field because there's no batching
beyond per-tick (§4.3).

**Effect on modeling:** none functionally. Just a different in-memory
representation.

---

## 5. Limitations specific to SimX

In addition to the limitations inherited from the RTL design (small
`LSUQ_IN_SIZE`, no prefetcher, total fence per block, no temporal
coalescer):

- **One response per tick per block** (§4.4) — under-models burst response
  throughput; the LSU can never receive faster than 1 batch's worth of
  data per tick even if the memory subsystem has many ready.
- **Trace blocks `Inputs[b]` for the full batch span** (§4.3) — slower
  than RTL for wide vectors; head-of-line blocking on the input side.
- **`MemCoalescer` is per-tick, not pipelined** (§4.1) — only one outgoing
  coalesced request per tick. RTL pipelines a coalesce + dispatch in the
  same cycle. Visible on consecutive-cycle access bursts.
- **No structural model for the multi-PID allocator** (§4.2) — won't
  reproduce the exact RTL stall conditions when `LSUQ_IN_SIZE` ≪ macro
  uop width.
- **`HashTable` allocation for `pending_rd_reqs`** has unbounded worst
  case in adversarial usage; unlikely in normal traces but worth noting
  as a model-fidelity concern.

---

## 6. Proposed improvements

Ordered by closeness-to-RTL (improving model fidelity) over
performance-of-the-simulator-itself.

1. **Drain multiple responses per tick.** Convert
   `process_response`'s single peek/pop to a loop bounded by
   `NUM_LSU_LANES` (or by `RspOut.size()`). One-line change; bumps
   simulated throughput closer to RTL on high-bandwidth workloads.
2. **Issue multiple batches per tick.** Mirror RTL's `MEM_BATCHES` by
   letting `process_request` issue all pending batches of the current
   trace until either `LSUQ_IN_SIZE` is full or `lmem_switch.ReqIn` is
   full. The trace can then leave `Inputs[b]` in the same tick.
3. **Pipeline the `MemCoalescer`.** Today's `on_tick()` issues one
   coalesced request per tick. Convert to a per-tick loop that drains as
   many groups from `ReqIn.peek()` as `ReqOut.full()` and the
   `pending_rd_reqs` allocator allow. Symmetric with improvement #2 on
   the LSU itself.
4. **Stride/next-line prefetcher.** Same as RTL improvement #2 — and
   easier to prototype in C++. Useful for studying prefetcher policies
   before committing to hardware.
5. **Per-address fence range.** Add `fence_lo` / `fence_hi` to the
   fence-lock condition; only requests inside the range gate the lock.
   Trivial in C++; makes SimX usable for `fence.tso` exploration.
6. **Multi-block fence.** For `fence.i` and similar, model a
   cross-block barrier that waits for all `NUM_LSU_BLOCKS` to drain.
   Currently fence is per-block, which under-models cross-block ordering
   semantics.
7. **Track effective MLP per kernel.** Add a perf counter for "average
   pending_rd_reqs occupancy" — directly visualizes how full the
   index-buffer-equivalent gets. Useful for sizing studies before
   committing to RTL changes.

---

## 7. Notes on design vs implementation

The RTL and SimX implementations share most of the *design*: a frontend
that does AGU and per-lane formatting; a per-block scheduler with a
non-blocking pending-load tracker; vector-to-channel batching when the
issue width exceeds the memory-channel count; partial responses; per-block
fence ordering; read/write decoupling.

The *implementations* differ in obvious ways (Verilog FSMs vs C++ ticks,
elastic buffers vs C++ queues), but those don't change the design.

Where SimX truly **diverges** in design (not just implementation) is in:

- **§4.2** — multi-PID handled by uop expansion (software-visible) instead
  of a hardware tracker
- **§4.3** — one batch per tick instead of pipelined batches
- **§4.4** — one response per tick

§4.1 (`MemCoalescer` placement) is *not* a design divergence — both RTL and
SimX coalesce on the LSU→DCache boundary with the same algorithm. Only
the rate (per-tick vs pipelined) differs, which is an implementation
detail.

These are simplifications that trade modeling fidelity for simulator speed.
The improvements in §6 close those gaps when warranted.

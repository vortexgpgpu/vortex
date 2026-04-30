# Vortex LSU Design (RTL)

**Scope:** logical/architectural design embodied by the RTL LSU
([VX_lsu_slice.sv](../../hw/rtl/core/VX_lsu_slice.sv) +
[VX_mem_scheduler.sv](../../hw/rtl/libs/VX_mem_scheduler.sv)). The
implementation is Verilog, but the design itself — pipeline structure,
queueing discipline, response tracking — is implementation-agnostic and
could be reproduced in C++ or any other substrate.

---

## 1. Architecture overview

```
   execute_if (per-issue-slot)             VX_lsu_slice           VX_mem_scheduler         VX_lsu_mem_if
   ────────────────────────────►   ┌───────────────────┐  ┌─────────────────────────┐  ────────────────►
                                   │ AGU + AddrType    │  │  Request queue (CQS)    │
                                   │ Byte-enable fmt   │  │  Index buffer (CQS)     │
                                   │ Store data shift  │  │  (Optional) coalescer   │
                                   │ Fence lock        │  │  Batched dispatch       │
                                   │ Multi-PID tracker │  │  Response demux         │
                                   └─────────┬─────────┘  └────────┬────────────────┘
                                             │                      │
                                             ▼                      ▼
                                       Tag pack/unpack      MEM_QUEUE_SIZE outstanding
                                       Load formatter       (out-of-order responses)
                                       (sign-extend, NaN-box)
                                             │                      ▲
   result_if (per-issue-slot)               ▲                      │
   ◄────────────────────────────────────────┴───────────── core_rsp ┘
```

The LSU is a two-stage pipeline:

- **Frontend (`VX_lsu_slice`)** — instruction-side adaptation: AGU, address
  classification, byte-enable formation, store-data shifting, fence ordering,
  multi-packet (PID) tracking, response formatting (sign-extension, NaN-boxing).
- **Backend (`VX_mem_scheduler`)** — generic memory-side scheduler shared with
  caches and other clients: queueing, optional coalescing, vector→channel
  batching, out-of-order response demultiplex.

Both stages parameterize over `NUM_LSU_LANES` (= `NUM_THREADS` in canonical
configs); per-issue-slot one slice is instantiated, so the LSU as a whole is
`ISSUE_WIDTH × VX_lsu_slice` and runs slices independently with shared
downstream memory.

---

## 2. The frontend in detail

### 2.1 Address-generation and per-lane formatting

For each of `NUM_LSU_LANES` lanes the slice computes
`full_addr[i] = rs1[i] + sext(offset)` ([VX_lsu_slice.sv:55-58](../../hw/rtl/core/VX_lsu_slice.sv#L55-L58)),
classifies the resulting block address as IO / LMEM / regular (one bit each
in `mem_req_flags` — see [§64-76](../../hw/rtl/core/VX_lsu_slice.sv#L62-L76)),
and packs sub-word stores into the line-width word with the appropriate
shift (`mem_req_data` shifting + `mem_req_byteen` mask). Misalignment is a
runtime assertion, not a hardware-handled fault.

### 2.2 Fence ordering

`fence_lock` (single bit per slice) is set when a fence's last PID packet
fires and cleared when the corresponding response packet completes. While
locked, the slice gates `mem_req_valid` and `execute_if.ready` to zero —
new requests cannot enter, and outstanding responses drain. This is a
**total fence per slice**: any fence on this slice serializes everything on
this slice.

### 2.3 Multi-PID packet tracker (per-slice)

A wide load (e.g., from VPU) may be expanded into multiple PID packets.
Each packet has the same `wid` / `tag` / `rd` but consecutive `pid` values;
SOP marks the first, EOP the last. The slice uses a small `VX_allocator`
+ `pkt_sop`/`pkt_eop`/`pkt_ctr` table sized `LSUQ_IN_SIZE` to
([§212-269](../../hw/rtl/core/VX_lsu_slice.sv#L212-L269)):

- allocate one slot at the SOP request fire,
- count how many sub-packets are outstanding per slot,
- mark the **packet-level** SOP / EOP on the response stream,
- release the slot when the last sub-packet response retires.

This separates the **memory-level** SOP/EOP (per-channel response packets)
from the **logical** SOP/EOP (per macro-load), so the writeback path sees
exactly one SOP and one EOP per macro-load even though responses arrive
out of order across PIDs.

---

## 3. The scheduler in detail

The scheduler is the MLP heart of the LSU.

### 3.1 Request queue

A `CORE_QUEUE_SIZE`-deep elastic buffer holds pending core requests
([:172-185](../../hw/rtl/libs/VX_mem_scheduler.sv#L172-L185)). Each
element holds the full vector `(mask, byteen, addr, flags, data, tag)` for
one logical request. The queue smooths backpressure between the frontend
and the memory bus.

### 3.2 Index buffer (the load tracker)

The crucial MLP structure. For every **read** that enters the request
queue, an `ibuf_waddr` slot is acquired from a `VX_index_buffer`
([:204-218](../../hw/rtl/libs/VX_mem_scheduler.sv#L204-L218)). The slot
index is embedded in the memory request tag as `reqq_tag_u = {uuid,
ibuf_waddr}`. Any response carrying that tag indexes back into the slot
table to recover the original metadata.

Consequences:

- **Up to `CORE_QUEUE_SIZE` reads can be in flight simultaneously per
  slice.** The index buffer is the per-slice MLP cap.
- **Responses can arrive out of order.** The tag self-routes back to the
  right slot; the slot release happens at EOP, not in arrival order.
- **No CAM / no associative search.** The index buffer is a free-list +
  RAM, O(1) acquire/release, no full-tag compare anywhere.
- **Reads and writes share the request queue but not the index buffer.**
  Writes go straight through (no response expected); only reads consume
  ibuf slots. This means a stream of writes does not stall behind a full
  read pending queue.

### 3.3 Optional coalescer

When `COALESCE_ENABLE` is true (which holds when `LINE_SIZE > WORD_SIZE`),
adjacent lane requests targeting the same memory line are merged into one
line request via `VX_mem_coalescer`
([:226-277](../../hw/rtl/libs/VX_mem_scheduler.sv#L226-L277)). The
coalescer carries its own `MEM_QUEUE_SIZE` outstanding state, so the
effective MLP becomes `min(CORE_QUEUE_SIZE, MEM_QUEUE_SIZE)`. The LSU as
configured today disables coalescing (`LINE_SIZE = WORD_SIZE` per
[VX_lsu_slice.sv:307-308](../../hw/rtl/core/VX_lsu_slice.sv#L307-L308));
the cache subsystem uses it.

### 3.4 Vector → channel batching

When `MERGED_REQS > MEM_CHANNELS` (i.e., the vector request can't fit in
one memory cycle), `VX_mem_scheduler` slices it into `MEM_BATCHES`
sub-requests and dispatches them sequentially through a batch counter
`req_batch_idx_r`, embedding the batch index into the memory tag
([:334-387](../../hw/rtl/libs/VX_mem_scheduler.sv#L334-L387)). Degenerate
batches (all-mask-zero) are skipped. Responses carry the batch index back,
so the scheduler reassembles the wide vector without needing in-order
response delivery.

### 3.5 Response handling (full vs partial)

Two modes selected by `RSP_PARTIAL`:

- **`RSP_PARTIAL = 1`** (LSU mode): each per-channel response packet is
  emitted upstream immediately with `crsp_mask = curr_mask` and
  `crsp_sop`/`crsp_eop` derived from the running `rsp_rem_mask` /
  `rsp_sop_r` state. Suits the LSU because writeback can commit lanes as
  they arrive, in any order.
- **`RSP_PARTIAL = 0`** (cache mode): per-lane data is written into a
  small `rsp_store` SRAM as it arrives; the upstream response only fires
  when `rsp_complete = ~|rsp_rem_mask` so the consumer sees one
  full-width response per logical request.

### 3.6 In-flight stall safety

A simulation-time `STALL_TIMEOUT` watchdog
([:559-584](../../hw/rtl/libs/VX_mem_scheduler.sv#L559-L584)) tracks each
allocated ibuf slot's age and asserts if any pending request exceeds
`STALL_TIMEOUT` cycles. Catches deadlocks during testing without affecting
synthesis.

---

## 4. Memory-level parallelism — what gives this design its throughput

In rough order of contribution to MLP:

1. **Index-buffer-decoupled responses.** The LSU need not wait for an
   in-order response stream from the memory subsystem. `CORE_QUEUE_SIZE`
   loads can be outstanding; the cache or LMEM port can return them in any
   order. This is the difference between a blocking LSU (1 outstanding) and
   a non-blocking LSU (N outstanding); the index-buffer pattern realizes
   non-blocking with O(1) hardware per outstanding request.
2. **Coalescing.** When enabled, this collapses redundant traffic before
   the memory bus, freeing channel cycles and effectively widening MLP from
   the memory-bus perspective.
3. **Vector-to-channel batching.** Decouples the issue width
   (`NUM_LSU_LANES`) from the memory channel count (`MEM_CHANNELS`). The
   LSU can issue 32-lane vectors over a 4-channel memory bus across 8
   batches without the consumer caring.
4. **Read/write decoupling.** Writes don't consume ibuf slots and don't
   stall behind read backlog (unless the request queue itself is full).
5. **Per-slice independence.** `ISSUE_WIDTH` slices issue independently
   and share only the downstream memory subsystem. One slice stalled on a
   long-latency miss does not block the other slices.
6. **Partial responses (LSU mode).** The frontend and writeback path commit
   lanes as soon as their bytes return — no all-or-nothing waiting.

### Performance ceiling

Per-slice peak load throughput =
`min(CORE_QUEUE_SIZE / avg_load_latency, 1) requests/cycle`. With the
default `CORE_QUEUE_SIZE = LSUQ_IN_SIZE = 8` (typical gen_config value) and
50-cycle DRAM latency, that's `8/50 ≈ 0.16` requests/cycle/slice — DRAM is
the bottleneck. With cache hits at 5 cycles, `8/5 ≈ 1.6` ⇒ slice-saturated;
the frontend can't issue faster than 1/cycle so the cap binds.

---

## 5. Limitations

- **`CORE_QUEUE_SIZE` is small (default 8).** For long-latency workloads
  (DRAM-bound) this caps MLP well below what the memory system could absorb.
- **No prefetching.** All requests are demand-driven. Strided patterns
  (the common GEMM/SpMV cases) get no head start.
- **No per-address fence granularity.** A fence is a total slice barrier;
  `fence.tso`-style ordering between specific address ranges is not modeled.
- **Coalescer is single-cycle.** Two requests issued on consecutive cycles
  to the same line each pay their own request — no temporal coalescing
  window. Mostly fine for SIMT (one wide issue per cycle) but limits
  scalar-style workloads.
- **No request reordering / no critical-word-first.** Requests issue in
  arrival order. A long-latency cache miss queued first behind which a
  fast LMEM hit got queued blocks the LMEM hit until the miss drains the
  request queue.
- **Misalignment is fatal.** No splitting of unaligned accesses; the
  software contract requires aligned addresses (see
  [VX_lsu_slice.sv:184](../../hw/rtl/core/VX_lsu_slice.sv#L184)).
- **Multi-PID tracker capacity = `LSUQ_IN_SIZE`.** A pathologically
  heavy multi-PID instruction sequence can starve the allocator before the
  request queue fills. Usually balanced but visible at extreme widths.
- **No request priority across lanes.** All-lane vectors are atomic units;
  the scheduler can't issue 4 lanes now and 4 lanes later if one channel
  is congested — it batches deterministically.

---

## 6. Proposed improvements

In rough order of impact-per-effort:

1. **Increase `CORE_QUEUE_SIZE` (deep ibuf).** Single-parameter knob; fold
   the index buffer onto BRAM when the size grows past LUT economics.
   Doubles MLP at high latency, ~free when the request stream isn't
   bursty. Watch for area growth in the response demux logic
   ([:437-525](../../hw/rtl/libs/VX_mem_scheduler.sv#L437-L525)) which
   scales with CORE_QUEUE_SIZE.
2. **Stride prefetcher.** Hook into the address stream pre-AGU; detect
   stride-1 / fixed-stride patterns over a small history window and issue
   speculative loads. Output goes into a small prefetch buffer that the
   AGU consults first. Biggest win on dense GEMM and stencil codes.
3. **Temporal coalescer window.** Extend the existing coalescer to look
   back N cycles for line-matching pending requests. Implementation: a
   small CAM over the last N memory tags. Effective on stripe-fetch
   patterns where consecutive cycles touch the same line.
4. **Critical-word-first / response-priority hints.** Tag requests with a
   "fast-path" hint propagated to the memory subsystem; on the response
   side, allow fast-path responses to overtake slow ones in the response
   buffer. Requires memory-side support (the cache subsystem already
   knows hit-vs-miss).
5. **Per-address-range fences.** Add a `fence_addr_lo` /
   `fence_addr_hi` filter to the fence-lock condition so only requests
   inside the range stall. Enables `fence.tso` and avoids unnecessary
   barriers.
6. **Unaligned access splitting.** Two-cycle access for cross-line
   unaligned loads/stores, transparent to software. Removes a sharp edge
   in the programming model at modest hardware cost (an extra batch
   counter and a merge buffer in the response path).
7. **Decoupled scheduler-per-bank.** Today one scheduler dispatches to all
   `MEM_CHANNELS` simultaneously; one congested channel stalls the whole
   batch. Per-channel sub-schedulers with independent batching can hide
   bank-conflict latency for SIMT codes that already shuffle threads across
   banks.

The first two — deeper queue and a stride prefetcher — together likely
double DRAM-bound MLP. The other items are surgical fixes for specific
patterns and warrant measurement before commitment.

---

## 7. Notes on design vs implementation

This document describes the *design* — pipeline stages, queue discipline,
the index-buffer trick for non-blocking responses, the coalescer/batcher
roles, the read/write decoupling, the partial-response policy. None of
those depend on Verilog. The same design can be (and is, with deliberate
simplifications) realized in C++ in the SimX functional simulator; see
[lsu_design_simx.md](lsu_design_simx.md) for that implementation's
divergence from this design.

The Verilog implementation specifics — `VX_elastic_buffer` sizing,
`VX_index_buffer` structure, `VX_mem_arb` placement — are realization
choices, not design choices. They optimize for FPGA/ASIC timing closure
and don't change the architectural semantics described above.

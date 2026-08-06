# Vortex Cache Flush Architecture

The cache flush machinery implements whole-cache invalidate-and-writeback plus the
reset-time tag initialization. SimX models flush as a one-line `flush_begin()` walk;
this document covers the RTL.

**Files:**
- [hw/rtl/cache/VX_cache_flush.sv](../../hw/rtl/cache/VX_cache_flush.sv) (per-bank state machine)
- [hw/rtl/cache/VX_cache_init.sv](../../hw/rtl/cache/VX_cache_init.sv) (cache-level FSM + input lock)
- [hw/rtl/cache/VX_cache_bank.sv](../../hw/rtl/cache/VX_cache_bank.sv) (sel arbitration + tag/data write paths)
- [hw/rtl/cache/VX_cache_tags.sv](../../hw/rtl/cache/VX_cache_tags.sv) (tag write semantics)
- [hw/rtl/cache/VX_cache_data.sv](../../hw/rtl/cache/VX_cache_data.sv) (data RAM read path on flush, WB only)
- [hw/rtl/core/VX_dcr_flush.sv](../../hw/rtl/core/VX_dcr_flush.sv) (DCR-driven flush trigger)

---

## 1. What "flush" means here

Vortex caches expose **one** flush primitive: an *entire-cache* invalidate-and-writeback,
gated on `MEM_REQ_FLAG_FLUSH` (bit 0 of `flags`, defined in
[VX_gpu_pkg.sv](../../hw/rtl/VX_gpu_pkg.sv)) being set on a `MemReq`. There is no
line-granular invalidate, no way-granular invalidate, no tag-only invalidate, and no
software-addressable per-line writeback. The only thing software can ask for is "drain
this cache." (The AMO subsystem's non-LLC probe path performs its own inline
single-line writeback-invalidate without touching this machinery — see
[§5](#5-relationship-to-line-granular-invalidation).)

A cache flush walks every line in every way and:

- in **write-back** mode: emits a writeback for each dirty line, then clears valid+dirty.
- in **write-through** mode: clears valid only (no writebacks ever — the line is already
  coherent with memory).

The same primitive is reused for the **reset-time tag init** (clears all valid bits) via a
dedicated `STATE_INIT` that runs once before `STATE_IDLE`.

---

## 2. End-to-end flow

```
SW writes DCR command
  └─► VX_dcr_flush  (synthesizes MemReq with FLAG_FLUSH=1, blocks until response)
        └─► dcache (port 0) — arbitrated with the LSU's normal traffic
              └─► VX_cache_init  (input gate at cache top)
                    │   • detects FLAG_FLUSH on any input port
                    │   • blocks ALL input ports (stalls upstream)
                    │   • waits BANK_SEL_LATENCY drains (no_inflight_reqs)
                    │   • broadcasts per_bank_flush_begin pulse to ALL banks
                    └─► VX_cache_flush  (per bank — 6-state FSM)
                          • IDLE → WAIT1 (mshr_empty)
                          • WAIT1 → FLUSH (counter walk)
                          • FLUSH → WAIT2 (bank0 only, drain mreq_queue)
                          • WAIT2 → DONE (1-cycle pulse)
                          • DONE → IDLE
                          └─► drives flush_valid into the bank's sel arbiter
                                └─► pipe_reg0 → S0 (cache_tags.flush=1, cache_data read if WB+dirty)
                                      └─► S1 → mreq_queue.push (writeback if dirty)
              VX_cache_init waits for ALL flush_end ─► acks the original flush MemReq
        └─► dcr_flush_if.done = 1 (one-cycle pulse)
SW polls DCR status, observes flush done
```

---

## 3. Components

### 3.1 `VX_cache_flush` — per-bank state machine

**6 states, ~3 bits of state register:**

| State        | Purpose                                                      |
|--------------|--------------------------------------------------------------|
| `STATE_INIT` | Reset-time tag clear. Walks `[0, 2^LINE_SEL_BITS)` counter, drives `flush_init` ⇒ `cache_tags.init`. Does **not** writeback (lines are X at reset, no dirty bit yet). |
| `STATE_IDLE` | Wait for `flush_begin` pulse from `VX_cache_init`.           |
| `STATE_WAIT1`| Stall until `mshr_empty` so the bank pipeline is quiet.      |
| `STATE_FLUSH`| Walk counter. Width is `LINE_SEL_BITS + (WB ? WAY_SEL_BITS : 0)`. Drives `flush_valid`, `flush_line`, `flush_way`. Increments on `flush_ready`. |
| `STATE_WAIT2`| **Bank 0 only** waits `bank_empty` so the last writeback drains before the cache acks completion. Other banks skip directly to DONE. |
| `STATE_DONE` | 1-cycle `flush_end` pulse to `VX_cache_init`.                |

**Notable behaviors:**
- The reset state is `STATE_INIT`, not `STATE_IDLE`.
- `STATE_INIT` latches an incoming `flush_begin` pulse into `flush_pending_r` and emits
  `STATE_DONE` *immediately* once init completes — init already invalidated everything,
  so an explicit flush would be redundant.
- `flush_pending_n` shadow logic only handles `STATE_INIT` overlap; a flush pulse
  arriving during `STATE_DONE` would be lost, but `VX_cache_init` is the upstream gate
  and never issues one there.
- `STATE_WAIT2` is `BANK_ID == 0` only — a serialization detail so the cache's
  acknowledgement doesn't race ahead of bank 0's pending writeback.

**Counter width:**
- WT: `LINE_SEL_BITS` only. Walks each line once; `cache_tags` clears all ways
  in parallel (`do_flush = flush && (!WRITEBACK || way_en)`).
- WB: `LINE_SEL_BITS + WAY_SEL_BITS`. Walks each `(way, line)` separately because
  writeback emits one `mreq_queue` entry per dirty line.

### 3.2 `VX_cache_init` — cache-level FSM + input lock

5-state FSM:

| State        | Purpose                                                      |
|--------------|--------------------------------------------------------------|
| `STATE_IDLE` | Pass requests through. Detect `flush_req_enable` (any input has `FLAG_FLUSH=1`). |
| `STATE_WAIT1`| Wait for `BANK_SEL_LATENCY * NUM_BANKS` outstanding xbar requests to drain (only when there is xbar latency; bypass otherwise). |
| `STATE_FLUSH`| 1 cycle. Pulses `flush_begin = {NUM_BANKS{1}}` to all banks. |
| `STATE_WAIT2`| Accumulate `flush_done |= flush_end` until all banks pulse done. |
| `STATE_DONE` | Release the lock specifically for the input ports that had `FLAG_FLUSH` set, so the synthetic flush request acks. Other inputs stay locked until their `req_ready` retires the flush ack — then return to IDLE. |

**Input lock mechanism** (the load-bearing piece for correctness):

```systemverilog
wire input_enable = ~flush_req_enable || lock_released[i];
core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && input_enable;
core_bus_in_if[i].req_ready  = core_bus_out_if[i].req_ready && input_enable;
```

While a flush is in flight, every input port presents `valid=0` downstream and `ready=0`
upstream. Upstream stalls. Only the input that originated the flush gets unlocked
(`lock_released_n = flush_req_mask`) so its `MemReq` actually enters the cache and
generates the response acknowledgement. After `STATE_DONE`, normal traffic resumes.

### 3.3 `VX_dcr_flush` — DCR-driven flush trigger

Instantiated in [VX_mem_unit.sv](../../hw/rtl/core/VX_mem_unit.sv), wired between the
LSU port 0 and the dcache. Synthesizes a degenerate `MemReq` (rw=0, addr=0, data=0,
byteen=0, `flags = 1<<MEM_REQ_FLAG_FLUSH`, AMO sideband zero) when `dcr_flush_if.req=1`,
and drives `dcr_flush_if.done = flush_bus_if.rsp_valid`.

A 1-bit `flush_inflight_r` register prevents re-injection while the previous request is
in flight. A 2:1 `VX_mem_arb` (`ARBITER="P"`, priority) merges the synthetic flush into
LSU port 0's stream — flush takes priority.

Notably, `dcr_flush` only routes to **port 0** of the dcache; the input lock in
`VX_cache_init` is what propagates the freeze to the other ports.

### 3.4 `VX_cache_tags` — what `flush=1` does

Per way:

```systemverilog
wire do_flush  = flush && (!WRITEBACK || way_en);
wire line_write = do_init || do_fill || do_flush || do_write;
wire line_valid = fill || write;       // ⇐ both 0 on flush ⇒ valid bit cleared
```

So `flush=1` fires a tag write at `line_idx` (waddr) with `line_valid=0`, clearing the
valid bit. In WT mode, ALL ways flush together (no `way_en` gate). In WB mode, only the
addressed way (`evict_way == i`) flushes — the bank's `flush_way` from
`VX_cache_flush` walks the way axis.

Reset behavior is separately handled by `STATE_INIT` driving `init=1`, which writes the
same `line_valid=0` to all ways at the indexed line. So tag SRAM does **not** need
asynchronous reset — nicer for FPGA / Block RAM mapping.

### 3.5 `VX_cache_data` — what `flush=1` does (WB only)

In write-back mode, `cache_data` reads the line on flush so the writeback path can pick
up the data:

```systemverilog
wire line_read = read || ((fill || flush) && WRITEBACK);
```

The dirty-bytes byteen RAM (when `DIRTY_BYTES=1`) similarly reads on flush so the
writeback's byteen tracks per-byte dirty marks. In WT mode neither happens — the
cache_data module ignores `flush` entirely.

### 3.6 `VX_cache_bank` — sel arbitration

The bank pipeline gives flush its own slot in the priority arbiter, with priority
`init > replay > fill > flush > creq`. The state machine asserts
`flush_valid` only after `mshr_empty` (no fills in flight) and `bank_empty` (after the
walk), so the runtime order is effectively *fills always finish first*. The arbiter
wiring is pessimistic in case of future scheduling changes.

`flush_ready` (= `flush_grant && !mreq_queue_alm_full && !pipe_stall`) gates the FSM's
counter — under WB mode + a near-full mreq queue, the walk pauses until egress drains.

---

## 4. Efficiency analysis

### 4.1 Area

The flush subsystem is **lightweight by design** — it reuses the bank's existing tag/data
write ports.

| Component        | Storage                                      |
|------------------|----------------------------------------------|
| `VX_cache_flush` | 3 bits state, 1 bit pending, ~10 bits counter. ~14 FF / bank. |
| `VX_cache_init`  | 3 bits state, NUM_BANKS-bit `flush_done`, NUM_REQS-bit `lock_released`, optional UUID register. ~15-20 FF for typical configs. |
| `VX_dcr_flush`   | 1 bit `flush_inflight_r` + a 2:1 `VX_mem_arb`. ~5 FF + arbiter. |
| Tag / data RAM   | **No extra storage.** Reuses existing write ports. |
| Mreq queue       | **No extra entries.** Writebacks share the existing fill-request queue. |

Combinational additions: a few muxes in the bank's sel path (`addr_sel`, `tag_sel`),
`do_flush_st0`/`do_flush_st1` decode, and the conditional writeback-emit mux — all small.

Area is essentially "free" relative to the cache tag/data SRAMs.

### 4.2 Speed

**Steady-state** flush time (cycles, ignoring fill/replay drain):

| Mode | Walk count                                     |
|------|------------------------------------------------|
| WT   | `LINES_PER_BANK` = `CACHE_SIZE / (LINE_SIZE * NUM_BANKS * NUM_WAYS)` |
| WB   | `LINES_PER_BANK * NUM_WAYS`                    |

For the default L1 dcache (`CACHE_SIZE=16384, LINE_SIZE=64, NUM_BANKS=2, NUM_WAYS=4`):
`LINES_PER_BANK = 32`. WT flush ≈ 32 cycles + drain, banks in parallel.

For the default L2 (`CACHE_SIZE=1MB, LINE_SIZE=128, NUM_BANKS=4, NUM_WAYS=8`):
`LINES_PER_BANK = 256`. WB flush ≈ `256 × 8 = 2048` walk cycles per bank, all banks in
parallel, plus writeback cycles for the dirty fraction serialized through `mreq_queue`.

**Pre-flush latency** (cycles before the walk starts):

- `STATE_WAIT1` waits `mshr_empty`. Worst case = the longest in-flight memory roundtrip,
  typically dozens of cycles.
- `BANK_SEL_LATENCY` drain in `VX_cache_init` — 1-2 cycles for typical xbar buffer
  sizes.

**Post-flush latency**: bank 0's `STATE_WAIT2` waits `bank_empty` (mreq queue drained),
which is bounded by the mreq queue depth.

**Throughput coupling:** because `flush > creq` priority and the input lock blocks all
new traffic during the walk, the cache is essentially *off* for the duration of the
flush. Other warps' loads/stores stall. Flush is linear in cache size and intended as
an infrequent operation — it is not a fine-grained primitive.

### 4.3 Correctness invariants

1. **Atomicity vs. normal traffic.** Once a flush is in flight, no normal core_req can
   reach the bank pipeline.
   - Enforced by `VX_cache_init`'s input lock (`input_enable=0` while `flush_req_enable`
     until `lock_released[i]`).
   - **Corollary:** `STATE_FLUSH` in the bank assumes the only in-flight requests are its
     own walk entries. `pipe_stall = crsp_queue_stall`, which can't pile up because no
     new reads are entering.

2. **No fills in flight when the walk starts.** Otherwise a fill could install a fresh
   line behind the walk pointer and survive the flush.
   - Enforced by `STATE_WAIT1` waiting for `mshr_empty`. Conservative — an MSHR with
     pending writes (in WT mode) also blocks, even though those don't install lines.

3. **Reset-time tag valid bits are 0.** Tag SRAM is not async-reset; instead `STATE_INIT`
   walks all lines on power-up.
   - **Corollary:** the cache must NOT accept any input while in `STATE_INIT`. The
     bank's `init_valid` gate covers this — every other source is masked off by the
     highest-priority `init_valid`.

4. **All banks finish before the cache acks.** `VX_cache_init.STATE_WAIT2` waits for the
   AND of `flush_end` across all banks before unlocking the originating input.

5. **Bank 0's writeback drains before its `flush_end`.** `STATE_WAIT2` in
   `VX_cache_flush` is bank 0 only — it adds a `bank_empty` (`mreq_queue_empty`)
   precondition so the last writeback hits memory before the cache says "done."
   Bank 0 is the canonical egress for propagating the flush downward to lower cache
   levels; other banks' work is purely local.

6. **Init walk is one-shot per reset.** No way to re-enter `STATE_INIT` mid-operation.
   `flush_pending_r` only handles the case where `flush_begin` arrives during init; it
   does not re-init.

**Edge cases:**

- Flush during init → `flush_pending_r` records it; init ends → `STATE_DONE` pulse
  fires. The init walk stands in for the flush walk (correct because init
  invalidated everything anyway).
- Multiple input ports racing the flush flag → the FOR loop in `STATE_IDLE` picks the
  highest-indexed one's UUID; all are unlocked together at the end.
- DCR-driven flush during a normal load → `VX_mem_arb` priority in `VX_dcr_flush` gives
  the synthetic flush priority, so it injects ahead of the load. The load's MemReq
  stalls in the LSU → `VX_cache_init`'s lock, then proceeds when DONE.
- `STATE_WAIT2`'s `bank_empty` does NOT re-check `mshr_empty` (already enforced by
  WAIT1); the implicit invariant is that nothing fills the MSHR during the walk,
  which the input lock guarantees.

---

## 5. Relationship to line-granular invalidation

The flush machinery is deliberately **not** the substrate for fine-grained
invalidation: it is whole-cache only, counter-driven (no external address input),
and freezes all cache inputs for the duration of the walk.

Where a single line must be written back and invalidated — the non-LLC cache levels
forwarding an atomic operation downstream — the AMO subsystem uses its own inline
probe path in the bank pipeline: the request probes the tag, emits a writeback if
the line is dirty, invalidates the single line, and forwards the operation, all
without stalling unrelated traffic. It reuses the same tag/data SRAM write ports as
flush but bypasses the flush FSM and `VX_cache_init` entirely. See
[multicache_amo_coherence.md](multicache_amo_coherence.md) and
[atomic_memory_operations.md](atomic_memory_operations.md).

# Vortex RTL Cache Flush Architecture — Review

**Date:** 2026-05-06
**Status:** Reference / pre-AMO-passthrough audit
**Scope:** RTL only. SimX flush is a one-line `flush_begin()` walk and is not analyzed
here.
**Files:**
- [hw/rtl/cache/VX_cache_flush.sv](../hw/rtl/cache/VX_cache_flush.sv) (per-bank state machine)
- [hw/rtl/cache/VX_cache_init.sv](../hw/rtl/cache/VX_cache_init.sv) (cache-level FSM + input lock)
- [hw/rtl/cache/VX_cache_bank.sv](../hw/rtl/cache/VX_cache_bank.sv) (sel arbitration + tag/data write paths)
- [hw/rtl/cache/VX_cache_tags.sv](../hw/rtl/cache/VX_cache_tags.sv) (tag write semantics)
- [hw/rtl/cache/VX_cache_data.sv](../hw/rtl/cache/VX_cache_data.sv) (data RAM read path on flush, WB only)
- [hw/rtl/core/VX_dcr_flush.sv](../hw/rtl/core/VX_dcr_flush.sv) (DCR-driven flush trigger)

---

## 1. What "flush" means here

Vortex caches expose **one** flush primitive: an *entire-cache* invalidate-and-writeback,
gated on `MEM_REQ_FLAG_FLUSH` (bit 0 of `flags`,
[VX_gpu_pkg.sv:123](../hw/rtl/VX_gpu_pkg.sv#L123)) being set on a `MemReq`. There is no
line-granular invalidate, no way-granular invalidate, no tag-only invalidate, and no
software-addressable per-line writeback. The only thing software can ask for is "drain
this cache."

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

**Special quirks:**
- The reset state is `STATE_INIT`, not `STATE_IDLE`.
- `STATE_INIT` latches an incoming `flush_begin` pulse into `flush_pending_r` and emits
  `STATE_DONE` *immediately* once init completes — the rationale being "init already
  invalidated everything; an explicit flush would be redundant." Subtle.
- `flush_pending_n` shadow logic only handles `STATE_INIT` overlap; if a flush pulse
  arrives during `STATE_DONE` it's lost (`VX_cache_init` is the upstream gate, so this is
  the gate's responsibility).
- `STATE_WAIT2` is `BANK_ID == 0` only — a serialization detail so the cache's
  acknowledgement doesn't race ahead of bank 0's pending writeback.

**Counter width:**
- WT: `LINE_SEL_BITS` only. Walks each line once; `cache_tags.flush=1` clears all ways
  in parallel (`cache_tags.sv:71`: `do_flush = flush && (!WRITEBACK || way_en)`).
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

Lives in [VX_mem_unit.sv:363](../hw/rtl/core/VX_mem_unit.sv#L363), wired between the
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

The bank pipeline gives flush its own slot in the priority arbiter:

```systemverilog
wire replay_grant = ~init_valid;
wire fill_grant   = ~init_valid && ~replay_enable;
wire flush_grant  = ~init_valid && ~replay_enable && ~fill_enable;
wire creq_grant   = ~init_valid && ~replay_enable && ~fill_enable && ~flush_enable;
```

Priority: `init > replay > fill > flush > creq`. The state machine asserts
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
`do_flush_st0`/`do_flush_st1` decode, and the conditional `is_fill_or_flush_st1` mux on
the writeback emit — all small.

**Verdict:** Area is essentially "free" relative to the cache tag/data SRAMs.

### 4.2 Speed

**Steady-state** flush time (cycles, ignoring fill/replay drain):

| Mode | Walk count                                     |
|------|------------------------------------------------|
| WT   | `LINES_PER_BANK` = `CACHE_SIZE / (LINE_SIZE * NUM_BANKS * NUM_WAYS)` |
| WB   | `LINES_PER_BANK * NUM_WAYS`                    |

For the L1 dcache with `CACHE_SIZE=8192, LINE_SIZE=16, NUM_BANKS=1, NUM_WAYS=1`:
`LINES_PER_BANK = 512`. WT flush ≈ 512 cycles + drain.

For an L2 (`CACHE_SIZE=131072, LINE_SIZE=64, NUM_BANKS=4, NUM_WAYS=4`):
`LINES_PER_BANK = 128`. WB flush ≈ `128 * 4 = 512` walk cycles per bank, all banks in
parallel + drain. Plus `~512 * dirty_fraction` writeback cycles serialized through
`mreq_queue_out`.

**Pre-flush latency** (cycles before the walk starts):

- `STATE_WAIT1` waits `mshr_empty`. Worst case = the longest in-flight memory roundtrip,
  typically dozens of cycles.
- `BANK_SEL_LATENCY` drain in `VX_cache_init` — 1-2 cycles for typical xbar buffer
  sizes.

**Post-flush latency**: bank 0's `STATE_WAIT2` waits `bank_empty` (mreq queue drained),
which is bounded by `MREQ_SIZE` cycles.

**Throughput coupling:** because `flush > creq` priority and the input lock blocks all
new traffic during the walk, the cache is essentially *off* for the duration of the
flush. Other warps' loads/stores stall.

**Verdict:** Linear in cache size, dominated by walk + writeback drain. No clever
optimization (e.g., dirty-only walk via a separate dirty-line list). Acceptable for an
infrequent operation; expensive if used as a fine-grained primitive.

### 4.3 Correctness

**The invariants that have to hold:**

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
     bank's `init_valid` gate (`replay_grant = ~init_valid`, etc.) covers this — every
     other source is masked off by the highest-priority `init_valid`.

4. **All banks finish before the cache acks.** `VX_cache_init.STATE_WAIT2` waits for the
   AND of `flush_end` across all banks before unlocking the originating input.

5. **Bank 0's writeback drains before its `flush_end`.** `STATE_WAIT2` in
   `VX_cache_flush` is bank 0 only — it adds a `bank_empty` (`mreq_queue_empty`)
   precondition so the last writeback hits memory before the cache says "done."
   - **Why bank 0 specifically:** the cache's downstream `mem_bus_if` is a fan-in across
     banks; the comment in `VX_cache_flush.sv:91-93` notes "the flush request to lower
     caches only goes through bank 0" — bank 0 is the canonical egress for the
     "propagate flush downward" message. Other banks don't need this drain because their
     work is purely local.

6. **Init walk is one-shot per reset.** No way to re-enter `STATE_INIT` mid-operation.
   `flush_pending_r` only handles the case where `flush_begin` arrives during init; it
   does not re-init.

**Edge cases handled correctly:**

- Flush during init → `flush_pending_r` records it; init ends → `STATE_DONE` pulse
  fires. The cache effectively double-counts the init walk as the flush walk
  (correct because init invalidated everything anyway).
- Multiple input ports racing the flush flag → the FOR loop in `STATE_IDLE` picks the
  highest-indexed one's UUID; all are unlocked together at the end.
- DCR-driven flush during a normal load → `VX_mem_arb` priority in `VX_dcr_flush` gives
  the synthetic flush priority, so it injects ahead of the load. The load's MemReq
  stalls in the LSU → `VX_cache_init`'s lock, then proceeds when DONE.

**Edge cases that look fragile:**

- The `STATE_INIT → STATE_DONE` early exit assumes "init invalidated everything ⇒ no
  writeback needed." That's true for the WT bring-up but if a future change adds a
  way to dirty lines before init completes, the writeback gets skipped. (Today not
  reachable — input is gated.)
- `flush_pending_r` is a single bit; only one queued flush request is tracked. A second
  flush request arriving during init is silently coalesced into the first. Acceptable
  given `VX_cache_init`'s upstream lock makes this hard to reach.
- `STATE_WAIT2`'s `bank_empty` definition is `~valid_st0 && ~valid_st1 && mreq_queue_empty`
  — it does NOT check `mshr_empty` (already enforced by WAIT1). Implicit invariant: after
  WAIT1 + the walk, MSHR remains empty because nothing fills it during the walk.

---

## 5. Limitations vs. AMO-passthrough requirements

The proposal §3.8 ([amo_simx_v3_proposal.md](proposals/amo_simx_v3_proposal.md#L100))
asks the L1 (non-LLC) bank to: **on each AMO**, probe the local line, writeback if
dirty, invalidate, then forward the AMO downstream. The existing flush machinery cannot
be reused as-is for these reasons:

1. **Whole-cache only.** The flush walks all `LINES_PER_BANK` × `NUM_WAYS`. AMO
   passthrough needs *one specific line* invalidated.
2. **Walks via a counter, not an external address.** `flush_line` and `flush_way` come
   from the FSM's counter; there is no external `addr` input.
3. **Stalls all input traffic for hundreds of cycles.** AMO passthrough must NOT freeze
   the cache — other lanes/warps need to keep flowing. The existing input lock is too
   coarse.
4. **Tag SRAM does not have a single-line invalidate primitive.** `flush=1` is what we
   need at the SRAM level, but the bank pipeline only drives it from `do_flush_st0`,
   which is gated on `is_flush_st0` set by the FSM walk.
5. **WAIT1 / WAIT2 / DONE cycle costs.** Even one "single-line flush" through the
   existing FSM would pay tens of cycles of overhead, dominating the AMO's actual
   downstream roundtrip.
6. **`VX_cache_init`'s input lock is binary.** A flush in flight blocks all inputs,
   not "just the AMO target line."

**Implication for the AMO L1 passthrough RTL design:**

- We need a **new** primitive: single-line probe-invalidate that runs *inline* with the
  bank pipeline (no FSM stall, no input lock).
- The cleanest way is to add a new sel-path source — `is_amo_probe_sel` — driven from
  the AMO request itself at S0, that:
  - reads the tag at S0 (already happens for any creq).
  - at S1, if hit:
    - in WB mode: emit a writeback into `mreq_queue` (reuse the existing `do_writeback_st1`
      path, which currently fires only on `is_fill_or_flush_st1`).
    - drives a 1-cycle tag write with `valid=0` at the AMO line (this needs either a
      new `inv` input on `VX_cache_tags`, or — more pragmatically — a `do_amo_inv_st0`
      pulse on the *next* cycle that reuses `cache_tags.flush` with the AMO line_idx).
  - emit the original AMO MemReq through the bank's existing `mreq_queue` egress, with
    a tag in the *passthru-id namespace* `[MSHR_SIZE, MSHR_SIZE + AMO_PASSTHRU_CAP)`.
  - On the response: route to `crsp_queue_data` via a new mux input, no fill.

Reusing the *flush* SRAM port is correct. Reusing the *flush* FSM is not.

---

## 6. Summary

The Vortex flush architecture is well-engineered for its stated purpose — *infrequent
whole-cache drain* — and pays for it with simplicity, low area, and very few corner
cases. The cost is that it cannot serve as the substrate for fine-grained
invalidation primitives. AMO passthrough at the non-LLC bank needs **new** RTL: a
sel-path source for single-line probe + invalidate + forward, using the cache's
existing tag/data write SRAM ports but bypassing the flush FSM and `VX_cache_init`
entirely.

**Recommendation for the AMO passthrough implementation:**

1. **Don't extend `VX_cache_flush` or `VX_cache_init`.** Their FSMs assume coarse
   coordination that the AMO path cannot pay.
2. **Do extend `VX_cache_tags`** with an explicit single-line `inv` input — small, no
   pipeline impact, no risk of regressing the existing flush walk.
3. **Add a new `is_amo_probe_st1` branch** in `VX_cache_bank.sv`'s S1 logic alongside
   the existing `is_fill_or_flush_st1` branch. Reuses the writeback emit path but with
   a different trigger.
4. **Tag namespace partition** `[0, MSHR_SIZE)` for fills and
   `[MSHR_SIZE, MSHR_SIZE + AMO_PASSTHRU_CAP)` for AMO passthrough — same trick the
   SimX impl uses, survives `TxRxArbiter` shifts because arbiters only inject bits at
   the LSB.
5. **Stall is local.** The AMO probe does not need to stall any other request — its
   tag write happens at S0 of the cycle after S1 detection, exactly like the existing
   AMO writeback FSM (`amo_wb_pending`).

Estimated incremental complexity: comparable to the existing AMO writeback FSM
(~50 lines of bank logic + ~10 lines on `VX_cache_tags`).

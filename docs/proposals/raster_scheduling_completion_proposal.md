# Raster Scheduling & Completion Redesign

**Status:** Draft for review
**Scope:** `hw/rtl/raster/`, `sim/simx/raster/`, `sw/kernel/include/vx_graphics.h` (ISA), `VX_config.toml`
**Date:** 2026-05-26

---

## 0. Terminology — "producer" and "consumer" precisely

Throughout this document:

- **Producer** = the upstream endpoint of an arb's `bus_in` port. For the
  cluster arb, producers are the `raster_core` instances (NUM_RASTER_CORES
  per cluster). For the socket arb, the "producer" is the single
  per-socket bus driven by the cluster arb.
- **Consumer** = the downstream endpoint of an arb's `bus_out` port,
  defined per arb level:

  | Arb                                | `bus_out[o]` consumer = | Count per cluster        |
  | ---------------------------------- | ----------------------- | ------------------------ |
  | Cluster arb (`raster_cluster_arb`) | **a socket's per-socket raster bus** | `NUM_SOCKETS` |
  | Socket arb (inside `VX_socket`)    | **a core's `VX_raster_unit`** | `NUM_SOCKETS * SOCKET_SIZE` |

  These are **physical bus endpoints** — they are NOT warps.

- **Warps are not direct consumers** of the raster bus. The SIMT cores
  can dispatch up to `VX_CFG_NUM_WARPS` (≤64) warps per core, all
  sharing one `VX_raster_unit` (SFU PE). The PE serves vx_rast
  instructions one at a time through its execute_if/result_if pipeline.
  Whichever warp issued the current vx_rast gets the next bus packet.

- **OR-chain semantics for sticky-done.** When the socket arb's
  `consumer_served[core] = 1` (one warp on that core acked done), the
  arb's bus_out for that core becomes a sticky-done stream. Subsequent
  warps' vx_rast reads on the same core will pull additional `{done=1}`
  packets from the same stream — every warp on the core exits. Similarly,
  when the cluster arb's `consumer_served[socket] = 1`, the entire
  downstream socket subtree (all cores in the socket, all warps in those
  cores) eventually drains via the sticky-done propagation.

So the proposal's "per-output sticky-done" applies at **both** arb levels
(cluster and socket), with per-level granularity (per-socket and per-core
respectively). Warps are downstream of both arbs and inherit drain via
the OR-chain.

---

## 1. Background

Vortex's RASTER extension implements a fixed-function tile/triangle rasterizer
that feeds quad descriptors to the SIMT cores via two custom RISC-V
instructions (CUSTOM1 opcode family):

| Instr           | funct3 | Semantics                                             |
| --------------- | :----: | ----------------------------------------------------- |
| `vx_rast`       |   3    | Pop the next quad descriptor (R-type, rd=pos_mask)    |
| `vx_rast_begin` |   4    | Per-frame fetch trigger (R-type, rd=x0, fire-and-forget) |

Today's topology (single instance per cluster, NUM_RASTER_CORES=1 case works):

```
                            ┌─────────────┐
                  ┌────────►│ raster_core │  (producer; reads tilebuf+primbuf
                  │         │   + slices  │   via rcache, walks each tile,
                  │ DCR     │             │   emits quad stamps)
                  │ writes  └──────┬──────┘
                  │                │ raster_bus_if
                  │                │ (req_valid, {stamps, done}, req_ready,
   ┌──────────┐   │                │  begin_pulse [slave→master])
   │ DCR arb  ├───┘                │
   └──────────┘                    ▼
                          ┌────────────────────┐
                          │ raster_cluster_arb │  ← N inputs → M sockets
                          └────────────────────┘
                          ┌────┬────┬────┬────┐
                          ▼    ▼    ▼    ▼
                       socket0 sock1 sock2 sock3
                          │    │    │    │
                          ▼    ▼    ▼    ▼      socket-level arb
                          per-core SFU's `VX_raster_unit` (consumer)
                          ▼
                     vx_rast() ↦ kernel sees pos_mask in result reg
```

The current implementation has correctness bugs that surface when
`NUM_RASTER_CORES > 1` and / or `NUM_SOCKETS > NUM_RASTER_CORES`. This
proposal answers four design questions, analyzes the failure modes in three
canonical topologies (single-cluster 1:M, single-cluster N:M, multi-cluster),
and proposes a clean redesign.

---

## 2. Q1 — How is `done` currently handled?

### Producer side (`VX_raster_core.sv`, slice loop)
```sv
assign slice_raster_bus_if[s].req_data.done = running
                                            && fetch_triggered
                                            && ~has_pending_inputs
                                            && ~(| slice_valid_in)
                                            && ~(| slice_busy_out)
                                            && ~(| slice_valid_out);

assign slice_raster_bus_if[s].req_valid = slice_valid_out[s]
                                       || slice_raster_bus_if[s].req_data.done;
```

A slice asserts `done=1` only when its frame has been triggered
(`fetch_triggered=1`) AND every internal stage is idle. `req_valid` stays
asserted in the done state with `stamps=0` — the producer continuously
publishes a "drain sentinel" until someone consumes it.

### Cluster arb (`VX_raster_arb.sv`)
```sv
wire done_all = (& done_mask);              // AND across N input cores
assign req_data_in[i] = {stamps_i, done_all};   // ← cluster-wide done forwarded
```

The arb forwards **`done_all`** (AND of every producer's done) on the output
bus, **not** the per-producer done. The selected input's stamps still flow
through, but the `done` bit is now the cluster-wide one.

### Consumer side (`VX_raster_unit.sv`)
```sv
assign response_data[i] = (is_begin_op || raster_bus_if.req_data.done)
                          ? 32'd0 : pm;
```

Consumer writes 0 to the destination register only when `done=1`. **But**: if
the selected producer's stamps happen to be 0 (drained), and `done_all=0`
(because another producer is still busy), the consumer reads `response_data
= stamps = 0` and the kernel exits because its check is
`if (pos_mask == 0) return;`.

### Kernel exit condition
```cpp
for (;;) {
    uint32_t pos_mask = vx_rast();
    if (pos_mask == 0) return;
    ...
}
```

The kernel **does not read the `done` bit** — only `pos_mask`. Any zero-byte
response terminates the loop for that warp lane.

### Failure mode (concrete)
In multi-producer configs, an early-finishing producer publishes
`{stamps=0, done=1}` continuously. The cluster arb selects it round-robin
with the still-producing peer. When it's selected, the output is
`{stamps=0, done=done_all=0}` — the consumer kernel sees pos_mask=0 and
exits early, even though there is more work pending on the other producer.

---

## 3. Q2 — Is `done` explicitly routed to each consumer? How does `raster_core` know which consumer?

**It doesn't.** That's the root design issue.

The current data path is a **producer-pushed broadcast over a shared arb
fabric** with no per-consumer addressing. There is no concept of "which
consumer asked for this quad" inside the producer or the arb fabric. The
producer simply emits `{stamps, done}` on its output; the cluster arb picks
one producer per cycle and one consumer per cycle and connects them.

Concretely:

1. `VX_raster_unit` (consumer) sets `req_ready=1` whenever it has a pending
   `vx_rast` instruction with rsp-buf space.
2. The socket-level arb collapses N cores' `req_ready` into a single
   socket-level `req_ready`.
3. The cluster arb sees M socket `req_ready` signals on its outputs and N
   producer `req_valid` signals on its inputs.
4. `VX_stream_arb` (the underlying generic arbiter) **only supports
   `NUM_INPUTS >= NUM_OUTPUTS`** (fan-in / merge direction). When
   `NUM_INPUTS < NUM_OUTPUTS`, the `g_arbiter` path leaves all but
   `output[0]` permanently silent — `valid_in_w[r]` for `r>=1` is hardwired
   to 0 in those output slots:

   ```sv
   // VX_stream_arb.sv:163-171  (NUM_INPUTS=2, NUM_OUTPUTS=4 case)
   for (genvar r = 0; r < NUM_REQS; ++r) begin
       localparam i = r * NUM_OUTPUTS + o;
       if (i < NUM_INPUTS) begin
           assign valid_in_w[r] = valid_in[i];
       end else begin
           assign valid_in_w[r] = 0;   // ← outputs 2,3 only see padding
       end
   end
   ```

This is **not a bug in `VX_stream_arb`** — it's that we're using a fan-in
arb for a fan-out problem. The arb was designed for the inverse direction
(merging consumers' requests into a single producer's queue).

---

## 4. Q3 — On the next frame, is the `done` request fully drained from the NoC?

**No.** This is the dominant correctness hazard in steady-state operation —
reset only happens once at boot, but every multi-draw-call workload crosses
frame boundaries continuously.

### The hazard

There are three pipelined elastic buffers between the slice's done
generator and the per-core `VX_raster_unit`:

1. `slice_req_arb` inside `VX_raster_core` (NUM_SLICES → 1, `OUT_BUF=2`)
2. `raster_cluster_arb` inside `VX_graphics` (NUM_RASTER_CORES → NUM_SOCKETS,
   `OUT_BUF=3` when `NUM_SOCKETS != NUM_RASTER_CORES`)
3. `raster_socket_arb` inside `VX_socket` (1 → cores_per_socket, OUT_BUF
   varies)

### The sequence that breaks

Consider frame N → frame N+1:

| Cycle    | Event                                                                  |
| -------- | ---------------------------------------------------------------------- |
| t₀       | Frame N's last real quads consumed; slices reach idle                  |
| t₁       | Slice asserts `done=1`, emits `{stamps=0, done=1}` sentinels           |
| t₁..tₖ   | Sentinels flow into the 3 elastic buffers above. Some consumers read one and exit (kernel `pos_mask==0` ⇒ `return`); **the buffers between the slice and not-yet-asked consumers still hold sentinel packets** |
| tₖ       | Last warp of frame N exits; host sees `vx_event_wait_value` completion |
| tₖ₊₁     | Host writes new DCRs for frame N+1 (`tile_count`, `tbuf_addr`, ...). My current code: `VX_raster_core` detects `raster_dcr_write` and clears `fetch_triggered` ⇒ slice's `done` deasserts at its output. **But the buffered sentinels downstream are NOT cleared.** |
| tₖ₊₂     | Kernel for frame N+1 launches; first warp executes `vx_rast_begin`     |
| tₖ₊₃     | First warp executes `vx_rast()`. The consumer's `VX_raster_unit` does `req_ready=1`; the upstream elastic buffer **pops the stale `{stamps=0, done=1}` from frame N**. |
| tₖ₊₄     | `response_data=0` (because `done=1` from the stale packet). Kernel sees `pos_mask==0` and returns. **Frame N+1 is never rendered.** |

### Why the existing safeguards don't catch this

- `fetch_triggered` clear-on-DCR-write only deasserts the slice's `done`
  output at the source. The 3 downstream elastic buffers are independent
  storage — they hold whatever was clocked in before the clear.
- The proposed `~fetch_triggered` gate in `has_pending_inputs` (in the
  current branch) prevents the slice from generating NEW drain sentinels
  during the pre-fetch window of frame N+1, but does nothing about the
  stale sentinels that were generated at the END of frame N.
- Per-consumer sticky-done (§7.2) catches single-frame correctness but
  doesn't compose across frames: at frame N's end the consumer's
  `consumer_done_served[i]` latches HIGH; if it's not cleared on
  frame N+1's DCR write, that consumer never sees frame N+1's quads.

### What needs to flush

Three things, atomically on every RASTER DCR write:

1. **Slice-output buffer** (`OUT_BUF=3` in `slice_req_arb`) — drop any
   in-flight `req_valid` packet.
2. **Cluster fanout/arb buffer** — same, between raster_core and socket.
3. **Socket arb buffer** — same, between socket and core.

The OR of (`reset`, `raster_dcr_write`) becomes a cluster-wide
`raster_flush` signal driven by `VX_raster_dcr` and consumed by every
elastic buffer on the raster bus path.

This is the addition motivating proposal §7.3.

---

## 5. Q4 — Two raster_cores with stripe partitioning, one cluster has no work

Consider `NUM_CLUSTERS=2, NUM_RASTER_CORES=1` (per cluster). The cluster-level
DCR programming is broadcast (`VX_dcr_arb` fan-out broadcasts to every
RASTER instance). Each `raster_core` (across both clusters) sees the same
`tile_count`, `tbuf_addr`, etc.

But there is **no per-cluster tile-count partitioning** in the host runtime.
`sw/runtime/graphics.cpp::Binning()` emits a single `tilebuf` for the whole
draw call; `vx_enqueue_dcr_write(VX_DCR_RASTER_TILE_COUNT, num_tiles, ...)`
writes the same `num_tiles` to every cluster's DCR slave.

Both clusters' `raster_core` therefore each process **all `num_tiles`
tiles**, producing duplicate quads. The host runtime never tells cluster 1
"you handle tiles N/2..N-1; cluster 0 handles 0..N/2-1".

Now consider `tile_count=1`:

- Cluster 0, instance 0: `start_tile_count = (1 + 1 - 1 - 0) >> 0 = 1`. One tile. OK.
- Cluster 1, instance 0 (no host-side partition): also processes the same 1 tile. Duplicate output.

If we extend stripe partitioning across clusters too:

- Cluster 0, instance 0: `start_tile_count = (1 + 2 - 1 - 0) / 2 = 1`. One tile.
- Cluster 1, instance 0: `start_tile_count = (1 + 2 - 1 - 1) / 2 = 0`. **No work**.

Cluster 1's `raster_core`:
- `mem_unit_start` pulses on the (eventual) `begin_pulse`
- `VX_raster_mem` sees `start && start_tile_count == 0`, stays in `STATE_IDLE`
- `mem_unit_busy=0`, `mem_unit_valid=0`, all slices idle
- With `fetch_triggered=1` and `has_pending_inputs=0`, slice's
  `done=1` immediately
- `req_valid=1` with `stamps=0, done=1`

Cluster 1's cores (cores 2,3 in this example) pull from cluster 1's
raster_core via cluster 1's arb. They each get one quad: `{stamps=0, done=1}`.
They write `response_data=0` (because `done=1`) and the kernel exits cleanly.

**This case actually works correctly** — provided each cluster's cores
consume from their own cluster's raster_core. The arb topology with
`NUM_INPUTS=1, NUM_OUTPUTS=N` is the broken one (per Q2).

The composite scenario that breaks is **single cluster, multiple
raster_cores, more sockets than raster_cores** — that's where the arb
direction inverts and outputs go dead.

---

## 6. Scenario Analysis

### A. 1 cluster, 1 raster_core, M sockets ← currently works
- arb: 1 → M (fan-out). Per Q2, only socket 0 gets quads; sockets 1..M-1 are dead.
- **Why does the current test pass?** Because the regression's working
  configs all happen to have `NUM_SOCKETS ≤ 1` (default `NUM_CORES=1`,
  `SOCKET_SIZE=1`). The `cores=2, NUM_RASTER_CORES=1` test in regression.sh
  has `NUM_SOCKETS=2 > 1=NUM_RASTER_CORES` and was hidden by the
  build-cache bug that masked the verilator WIDTHTRUNC warning. It is in
  fact incorrect rendering — never noticed because the reference image
  predates the test.

### B. 1 cluster, N raster_cores, M sockets
- N=M: arb is 1:1 passthrough. **Works** — each producer paired with one consumer.
- N>M: arb is fan-in (merge). **Works** — standard `VX_stream_arb` use case.
- N<M: arb is fan-out. **Broken** (per Q2).

### C. 2 clusters, 1 raster_core each, M sockets per cluster
- Each cluster is self-contained: its own DCR slave, raster_core, arb, sockets.
- Same arb topology as scenario A applies **per cluster**.
- Host-side: no inter-cluster tile-count partitioning today. Both clusters
  process the entire tilebuf → duplicate quads → blend-disabled passes are
  ~OK (idempotent writes) but blend-enabled passes accumulate twice.

### 6.1 Comprehensive Scenario Matrix

The table below enumerates every scenario the redesign must handle
correctly, indexed against the design pieces in §7 that cover them.
**Status legend:** ✓ covered by the proposed redesign; ⚠ relies on
documented invariant (e.g., host barrier); ✗ unsupported (kernel /
runtime contract violation).

#### Workload scenarios

| # | Scenario | Status | How it's covered |
|---|----------|:------:|------------------|
| W1 | `tile_count=0` (empty draw, global)                                              | ✓ | §7.3. `start_tile_count=0` in every instance → `mem_unit` stays IDLE → slice `done=1` immediately after `fetch_triggered` latches. Arb sees `done_all=1`, emits `{stamps=0, done=1}` to all outputs. Every warp's first `vx_rast()` returns `pos_mask=0`, kernel exits. |
| W2 | `tile_count > 0`, balanced across instances                                       | ✓ | Each instance gets `tile_count / NUM_INSTANCES` tiles via stripe partitioning (§7.5). Each works in parallel; arb fan-in/fan-out routes their outputs to consumers. |
| W3 | `tile_count > 0`, imbalanced (e.g., `tile_count=3, NUM_INSTANCES=2` → 2 vs 1)     | ✓ | §7.2's `req_valid_in[i] = req_valid && (~done \|\| done_all)` gate suppresses an early-finishing producer's stamps-with-done=0 leakage. Slow producer's real quads continue flowing; when all drain, `done_all=1` opens the gate for the final sticky-done broadcast. |
| W4 | `tile_count > 0`, `NUM_INSTANCES > tile_count` (some instances zero-work)         | ✓ | Same as W1 for the zero-work instances (slice `done=1` immediately). Working instances behave as W2. Arb gating (W3) prevents zero-work instance's `stamps=0` from leaking. |
| W5 | Mid-frame `vx_rast_begin` re-issue by same warp (kernel oddity)                   | ✓ | All begin_pulses during active frame are no-ops (§7.3 — `frame_drained=0`). Producer's `fetch_triggered` already HIGH, no re-trigger. |
| W6 | Kernel forgets to call `vx_rast_begin` before `vx_rast` (buggy kernel)            | ✗ | Producer's `fetch_triggered=0`, slice `done=0`, bus quiet. Consumer's `vx_rast` stalls indefinitely → scoreboard timeout. **Kernel contract violation — must call begin once per frame.** |
| W7 | Host re-launches without writing new DCRs between frames                           | ⚠ | Producer's `fetch_triggered=1` still latched (only cleared by `raster_dcr_write`). New `begin_pulse` doesn't re-trigger mem_unit (per existing `!fetch_triggered` guard). Bus stays quiet → kernel hangs. **Documented invariant: host MUST write at least one RASTER DCR (e.g., re-write `tile_count`) between launches**, even if values are unchanged. |

#### Topology scenarios

| # | Scenario | Status | How it's covered |
|---|----------|:------:|------------------|
| T1 | 1 cluster, 1 raster_core, 1 socket (1:1)                                           | ✓ | Trivial; existing arb is 1:1 passthrough. |
| T2 | 1 cluster, 1 raster_core, M sockets (1:M fan-out)                                   | ✓ | §7.1 new fan-out path: consumer-side round-robin arb per output picks the single producer when valid. |
| T3 | 1 cluster, N raster_cores, M sockets, N=M (1:1 per pair)                            | ✓ | Existing arb (1:1) extended trivially in §7.1. |
| T4 | 1 cluster, N raster_cores, M sockets, N>M (fan-in / merge)                          | ✓ | Existing `VX_stream_arb` fan-in path works today. §7.2/§7.3 sticky-done state still applies to each output. |
| T5 | 1 cluster, N raster_cores, M sockets, N<M (fan-out from multiple producers)         | ✓ | §7.1 new fan-out path with N valid producers. Each output o round-robins across the valid (un-gated) producers. |
| T6 | 2+ clusters, 1 raster_core each, balanced work                                       | ✓ | §7.5 cluster-aware `INSTANCE_IDX`. Each cluster's arb, `consumer_served`, `frame_drained` are independent (begin_pulse is cluster-local). |
| T7 | 2+ clusters, one cluster has zero-work instances (no tiles for cluster 1)            | ✓ | Empty cluster handled as W1 *within* that cluster. Other clusters proceed independently. |
| T8 | 2+ clusters with multi-raster-core per cluster, mixed topology                       | ✓ | Composition of T5+T6. Each cluster's arb is independent; per-cluster stripe (T5) + cross-cluster stripe (T6) compose because `NUM_INSTANCES = NUM_CLUSTERS * NUM_RASTER_CORES_PER_CLUSTER`. |

#### Multi-warp / multi-consumer timing

| # | Scenario | Status | How it's covered |
|---|----------|:------:|------------------|
| C1 | NUM_WARPS=64 per core, all issue `vx_rast_begin` near frame start                   | ✓ | §7.3 gated flush. First begin_pulse of frame N+1 triggers flush (only if `frame_drained=1` from N). Subsequent 63 begin_pulses are no-ops. |
| C2 | NUM_WARPS=64, begin_pulses spread over many cycles (slow dispatch)                  | ✓ | Same gating; fresh in-flight data in OUT_BUFs is preserved while late warps' begins arrive. |
| C3 | One warp drains (sees `done=1`) while other warps in same core still processing     | ✓ | `consumer_served[o]` latches on first done ack — sticky. Other warps' subsequent `vx_rast()` reads return sticky-done from OUT_BUF, also exit. |
| C4 | One socket's warps all drain while other sockets still active                       | ✓ | Per-output `consumer_served[o]` is independent. `frame_drained` waits for `all_active_served`. Active sockets continue receiving real quads (their `consumer_served[o]=0` keeps arb routing producer's stream). |
| C5 | Mid-frame begin_pulse storm — would-be flush while some warps still active          | ✓ | Cannot happen with current ISA (frame N+1 begin_pulse blocked by host barrier until frame N drained). Even if it could: `frame_drained=0` mid-frame → flush gated off → no data loss. |
| C6 | Inactive consumer (e.g., grid_dim < NUM_SOCKETS*NUM_WARPS, some cores get no warps) | ✓ | `consumer_was_active[o]=0` for that socket. Doesn't gate `all_active_served`. Buffer for inactive consumer may accumulate sticky-done but is never read. |
| C7 | Variable grid across launches (consumer active in frame N, inactive in N+1)         | ✓ | Frame N+1's first begin clears all state (per §7.3). Inactive consumer in N+1 doesn't gate `frame_drained`. If consumer returns to active in N+2: flushed before reuse. |

#### Frame boundary / state transitions

| # | Scenario | Status | How it's covered |
|---|----------|:------:|------------------|
| F1 | First frame after POR reset                                                          | ✓ | `consumer_served=0, frame_drained=0` at reset. First begin_pulse → `flush_trigger=0` (no stale state), but `fetch_triggered <= 1` proceeds. OUT_BUFs are reset-clean (from reset, not flush). |
| F2 | Frame N → N+1 with stale `{done=1}` in OUT_BUF from N                               | ✓ | §7.3: at N+1's first begin_pulse, `frame_drained=1` ⇒ flush_trigger fires ⇒ OUT_BUFs reset, `consumer_served`, `consumer_was_active`, `frame_drained` all cleared. N+1's first read of bus gets fresh data only. |
| F3 | Frame N → N+1 with N+1 being an empty job (W1 on new frame)                          | ✓ | F2 flush followed by W1 (zero-work) — both compositions covered. Producer immediately re-asserts `done=1` after `mem_unit_start` pulse, arb forwards fresh `{done=1}`, consumers exit. |
| F4 | Cross-cluster timing skew (cluster 0 finishes much earlier than cluster 1)          | ✓ | Per-cluster state independent. Host barrier waits for ALL warps across clusters. Cluster 0's `frame_drained` stays HIGH from t_drain_0 until N+1's begin_pulse — quietly. |
| F5 | Concurrent first-begin from multiple cores/clusters at frame N+1 start              | ✓ | Each cluster's OR-reduced `begin_pulse_any` is local. Single flush per cluster per frame transition. |

#### Producer-side edge cases

| # | Scenario | Status | How it's covered |
|---|----------|:------:|------------------|
| P1 | Producer's `mem_unit` hangs (memory backpressure, e.g., L2 misbehavior)             | ✗ | `mem_unit_busy=1` stays HIGH, slice `done=0` forever, kernel hangs. Not addressed by this proposal — orthogonal cache/memory issue. |
| P2 | DCR write arrives concurrent with begin_pulse                                        | ✓ | `raster_dcr_write` clears `fetch_triggered` (existing behavior); begin_pulse re-sets it next cycle. Atomic from the producer's POV — but per W7 the host shouldn't program DCRs while kernel is running. |
| P3 | Producer drains, all consumers served, then nothing happens (idle gap before next frame) | ✓ | Producer stays in sticky `{done=1}` state. Arb sticky-dones output. OUT_BUFs may fill up to depth then back-pressure stops accumulation. No data loss. |

**No scenario in this matrix produces incorrect output under the redesign.**
The two ✗ entries (W6, P1) are kernel/runtime contract violations
unrelated to raster scheduling; the one ⚠ (W7) is a documented invariant
that the runtime already follows (DCR re-write per frame is standard).

---

## 7. Proposed Redesign

### 7.1 Topology — fix fan-out direction in `VX_raster_arb`

The cluster arb's job (`raster_cluster_arb` in `VX_graphics.sv`) is to
route quads from `NUM_RASTER_CORES` producers to `NUM_SOCKETS` consumers.
The three relative-size cases must all work:

| Case             | Direction       | Status                              |
| ---------------- | --------------- | ----------------------------------- |
| N > M            | Fan-in / merge  | **Works today** (VX_stream_arb path) |
| N == M           | 1:1 passthrough | **Works today**                     |
| N < M            | Fan-out         | **Broken today** (only output[0] receives) |

Today's `VX_raster_arb` delegates to `VX_stream_arb`, whose `g_arbiter`
path silently leaves outputs `1..M-1` dead when `N < M`. Per §7.2's
"single interface, single module" constraint, the fix is in-place inside
`VX_raster_arb`:

- When `N >= M`: keep the existing `VX_stream_arb` path (no regression).
- When `N < M`: switch to a **consumer-side arb** — for each output `o`,
  pick one input via round-robin on `valid_in[*]`, route its packet to `o`
  when `bus_out[o].req_ready` asserts.

The mapping of which output gets which input each cycle is round-robin
across producers, so over time each consumer sees quads from every
producer (no producer-consumer affinity baked in — the work distribution
happens via the stripe-partitioned tile_count, not the arb).

Single module, no new files. Concrete logic appears in §7.2.

### 7.2 Single-interface constraint — fix inside `VX_raster_arb`, no per-consumer ABI

**Constraint:** `VX_raster_bus_if` stays unchanged: `{req_valid, req_data
{stamps, done}, req_ready, begin_pulse}`. We do not add per-consumer
metadata to the bus. The cluster-level arbiter is a single module
(`VX_raster_arb`, extended in place — not a separate `VX_raster_fanout`)
that handles both fan-in (N>M) and fan-out (N≤M) directions.

The Q1/Q2 bug — producer-state `done` leaking to wrong consumers — is
solved by **per-output sticky-done state internal to the arb**, not by
changing the bus protocol. Each arb output port `o` keeps one register
bit `consumer_served[o]`; this state lives inside `VX_raster_arb` and is
invisible to producers and consumers.

```sv
module VX_raster_arb #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    ...
) (
    input wire clk, reset,
    VX_raster_bus_if.slave  bus_in_if  [NUM_INPUTS],
    VX_raster_bus_if.master bus_out_if [NUM_OUTPUTS]
);
    // Per-output sticky-done flag. Set when this output served a
    // {done=1} packet to its consumer. Cleared by begin_pulse_any
    // (§7.3). Drives a synthesized {stamps=0, done=1} on this output
    // until cleared — no producer interaction needed once latched.
    reg [NUM_OUTPUTS-1:0] consumer_served;

    // Existing N→M direction (when NUM_INPUTS >= NUM_OUTPUTS): use
    // VX_stream_arb (fan-in path — works today).
    //
    // New N→M direction (when NUM_INPUTS < NUM_OUTPUTS): per-output
    // round-robin pull from the available NUM_INPUTS producers. Each
    // output o chooses one input via consumer-side arb (NUM_INPUTS
    // requests, 1 grant per output).

    // Output o's req_valid:
    //   - consumer_served[o] ? 1 (sticky sentinel) :
    //     (valid_in[selected_input[o]] && !bus_in[selected_input[o]].done) || done_all
    //
    // Output o's stamps:
    //   - consumer_served[o] ? 0 :
    //     bus_in[selected_input[o]].stamps
    //
    // Output o's done:
    //   - consumer_served[o] || done_all
    //
    // Set consumer_served[o] when bus_out[o].req_ready && {output produced done}
endmodule
```

Key properties:

1. **No bus interface change.** Producers and consumers see the same
   `VX_raster_bus_if` as today.
2. **No per-consumer ABI exposure.** The sticky-done is a private optimization
   inside the arb — invisible to anything outside.
3. **No premature exit.** Output `o` only emits `{stamps=0, done=1}` to its
   consumer when (a) all producers are done OR (b) it has already served
   done at least once this frame. A still-producing peer's quads will
   reach `o` if `o` hasn't yet been served done.
4. **Sticky after first done.** Subsequent `vx_rast()` from a consumer that
   already saw done returns immediately with pos_mask=0 — no stall on the
   producer side (which may still be serving other consumers' last quads).
5. **Multi-frame.** Cleared on `begin_pulse_any` — see §7.3.

**Applies at both arb levels.** The `consumer_served[o]` mechanism is
instantiated identically in:

1. The **cluster arb** (`raster_cluster_arb` in `VX_graphics.sv`), where
   `o` indexes a socket (`NUM_OUTPUTS = NUM_SOCKETS`). Sticky-done at
   this level guarantees every core under socket `o` can subsequently
   receive done sentinels.
2. The **socket arb** (inside `VX_socket.sv`), where `o` indexes a core
   (`NUM_OUTPUTS = SOCKET_SIZE`). Sticky-done at this level guarantees
   every warp on core `o` can subsequently receive done sentinels.

Per §0's OR-chain semantics, drain at the higher level transitively
implies drain at the lower level. There is no per-warp state anywhere
in the bus fabric — warps inherit drain via the sticky-done stream
visible at their core's `VX_raster_unit`.

### 7.3 NoC drain on frame boundary — gated flush on FIRST `begin_pulse` per frame

**Constraint:** no DCR-write-triggered flush (per §7.6). Bus interface
`VX_raster_bus_if` unchanged (per §7.2). No per-warp state inside
`VX_raster_unit` (would not scale to 64 warps × multiple cores). Mid-frame
`begin_pulse`s from late warps must NOT drop fresh in-flight quads.

**The 64-warp problem.** With up to 64 warps per core, each issuing its own
`vx_rast_begin`, the cluster will see many `begin_pulse`s per frame
spread over a long window (every warp's retire schedule is independent).
A naïve "flush on every begin_pulse" approach would drop fresh quads
already queued in OUT_BUFs when a late warp's begin fires. The flush must
be **gated** to fire only ONCE per frame transition.

**Mechanism: drain-then-begin gating.** The arb maintains a per-frame
sticky bit `frame_drained` that latches HIGH when every active consumer
has been served `done` this frame. The flush is `begin_pulse_any &&
frame_drained` — i.e., a begin_pulse only flushes when the previous
frame's drain is observed. Subsequent begin_pulses within the same frame
are no-ops.

```sv
// VX_raster_arb internal state
reg [NUM_OUTPUTS-1:0] consumer_served;     // §7.2: set when output o acked done
reg [NUM_OUTPUTS-1:0] consumer_was_active; // set when output o acked anything
reg                   frame_drained;

// "Drained" = every active consumer has been served done this frame.
// Inactive consumers (consumer_was_active[o]=0) are excluded — they don't
// gate the next frame's start.
wire all_active_served = &(consumer_served | ~consumer_was_active);

wire begin_pulse_any = (| begin_pulse_in);   // existing OR-reduce
wire flush_trigger   = begin_pulse_any && frame_drained;

always @(posedge clk) begin
    if (reset || flush_trigger) begin
        consumer_served     <= '0;
        consumer_was_active <= '0;
        frame_drained       <= 1'b0;
    end else begin
        for (int o = 0; o < NUM_OUTPUTS; ++o) begin
            if (bus_out_if[o].req_valid && bus_out_if[o].req_ready) begin
                consumer_was_active[o] <= 1'b1;
                if (bus_out_if[o].req_data.done)
                    consumer_served[o] <= 1'b1;
            end
        end
        if (all_active_served && !frame_drained)
            frame_drained <= 1'b1;
    end
end
```

**OUT_BUF flush:** Each elastic buffer on the raster bus path gets its
reset ORed with the same `flush_trigger` from its local arb:

```sv
VX_elastic_buffer #(.DATAW(REQ_DATAW), .SIZE(OUT_BUF)) out_buf (
    .clk    (clk),
    .reset  (reset | flush_trigger),   // ← was just `reset`
    ...
);
```

Same treatment for slice-level arb (inside `VX_raster_core`) and any
socket-level arb. Each arb computes its own `flush_trigger` from its
own `consumer_served`/`consumer_was_active` state machine.

**Full sequence for frame N → N+1 (multi-warp, 64 warps/core):**

| Cycle              | Event                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| Frame N drain      | Producer drains, slice asserts `done=1, valid=1`                                                 |
| ...                | Arb routes done to each active consumer; `consumer_served[o]` latches HIGH per ack               |
| Last consumer done | `all_active_served=1` → `frame_drained <= 1`                                                     |
| Idle gap           | OUT_BUFs may hold queued stale `{done=1}` packets. `frame_drained=1` is latched.                 |
| t_b1               | Frame N+1 starts. **First** `vx_rast_begin` retires (any warp on any core in cluster).           |
| t_b1+1             | `begin_pulse_any=1`, `frame_drained=1` ⇒ `flush_trigger=1`. Three things in parallel:            |
|                    |   1. `consumer_served[*]`, `consumer_was_active[*]`, `frame_drained` all cleared                 |
|                    |   2. Every OUT_BUF on the path resets (drops queued stale `{done=1}`)                            |
|                    |   3. Producer's `fetch_triggered <= 1`, kicks off mem_unit                                       |
| t_b2 (>t_b1)       | **Subsequent** `vx_rast_begin` retires (warp 1, 2, ..., 63 on each core).                        |
| t_b2+1             | `begin_pulse_any=1`, but `frame_drained=0` (just cleared) ⇒ `flush_trigger=0`. **No flush.**     |
|                    |   - Producer's `fetch_triggered` already HIGH ⇒ idempotent re-trigger ignored.                   |
|                    |   - In-flight fresh quads in OUT_BUFs are preserved.                                             |
| t_q                | Producer emits first fresh quad. Arb routes it. Any warp's `vx_rast()` pulls it.                 |
| ...                | Frame N+1 proceeds normally. Many more `begin_pulse`s may still fire as remaining warps retire;  |
|                    | all are no-ops because `frame_drained=0` throughout the active frame.                            |
| Frame N+1 drain    | Producer drains, last active consumer acked, `frame_drained <= 1` again.                         |
| Frame N+2 (...)    | Same pattern as N→N+1.                                                                           |

**Why this handles every problematic case:**

1. **Multi-warp begin_pulse storms.** First begin per frame triggers exactly
   one flush. All subsequent begins are no-ops (`frame_drained=0`).
   In-flight fresh data is never lost.

2. **Zero-work instance (Scenario Q4).** After flush at t_b1+1: producer
   instance with `start_tile_count=0` keeps `mem_unit_busy=0`, slice
   re-asserts `done=1` almost immediately. Arb routes the FRESH `{done=1}`
   to consumers. Consumer kernels exit cleanly — no buffer stale leak
   possible because OUT_BUF was flushed at t_b1+1.

3. **Multi-cluster (Scenario C).** Each cluster has its own arb and its
   own `frame_drained`/`consumer_served` state. Each cluster's
   begin_pulse_any wire is cluster-local (today, and unchanged in the
   redesign). Flush in cluster 0 doesn't disturb cluster 1, and vice
   versa. Stripe partitioning across clusters works because each
   `raster_core` has `INSTANCE_IDX = CLUSTER_ID * N + i` (§7.5).

4. **Inactive consumer (variable grid).** A consumer that never asks has
   `consumer_was_active[o]=0`, so it doesn't gate `all_active_served`.
   `frame_drained` correctly latches when only the active consumers are
   served. Buffer for that consumer may accumulate but is never read.

5. **Warp issues `vx_rast_begin` multiple times within a frame.** Each
   begin fires `begin_pulse`, but `frame_drained=0` throughout the
   active frame, so all are no-ops. (Per §11 Q4: spec allows this; the
   arb just dedupes.)

**Register and area cost:**

| Component                                                       | Cost                                  |
| --------------------------------------------------------------- | ------------------------------------- |
| Per-output `consumer_served[o]` + `consumer_was_active[o]`      | 2 × NUM_OUTPUTS flops per arb (~32/cluster) |
| `frame_drained` flop per arb                                    | 1 flop per arb                        |
| Elastic buffer flush logic                                      | Free — single OR gate on existing reset |
| Consumer-side state                                             | **Zero** — no flops added to `VX_raster_unit` |
| Producer-side state                                             | **Zero** — existing `fetch_triggered` reused |

Per cluster: ~33 flops + a handful of OR gates. Steady-state quad-pop
path is **unchanged** — same combinational depth, same buffer depths.
Frame-boundary cost: 1 cycle of buffer-empty latency, amortized over
hundreds of cycles of producer-side fetch latency.

#### 7.3.1 Draining an N-deep NoC — why one atomic cycle is sufficient

The raster bus is a cluster-local NoC with depth `N = sum of OUT_BUF
depths across all arbs in the path` — typically 6 to 8 stages between
the producer's slice and the consumer's `VX_raster_unit`. Naively
flushing a deep NoC of stale `{done=1}` packets is hard because each
stage has its own latency and reset domain. **The Vortex raster bus
sidesteps this by making `begin_pulse_any` combinational**:

```sv
// In VX_raster_arb (every instance):
wire begin_pulse_any = (| begin_pulse_in);   // OR-reduce, no flop
for (genvar i = 0; i < NUM_INPUTS; ++i) begin
    assign bus_in_if[i].begin_pulse = begin_pulse_any;   // pure combinational
end
```

The chain of OR-reduces from the consumer's `vx_rast_begin` retire all
the way up to every producer's `raster_core` is one combinational path.
Every arb's `begin_pulse_any` (and therefore every elastic buffer's
`flush_trigger`) goes HIGH on the **same clock edge** — there is no
window where some stages have flushed and others still hold stale data.

**Drain sequence:**

| Cycle  | All N stages simultaneously                                                  |
| ------ | ---------------------------------------------------------------------------- |
| t = 0  | Warp retires `vx_rast_begin` → combinational fan-in sets `begin_pulse_any=1` at every arb. Gated by `frame_drained` (only the FIRST begin per frame). |
| t = 1  | Every elastic buffer on the path saw `reset \| flush_trigger = 1` at the t=0→t=1 edge. **All N stages empty atomically.** |
| t = 2+ | Producer's preserved state (sticky-done if drained, fresh fetch if active) drives new packets into the empty slice OUT_BUF. Propagates downstream at 1 cycle per stage. |
| t = N+1 | First post-flush packet reaches the consumer's `raster_bus_if`. Whichever warp's `vx_rast` is pending pulls it. |

**Drain is destructive, not selective.** We don't try to retain "fresh"
packets while dropping "stale" ones. That would require either per-packet
frame_id tags (bus interface change, violates §7.2) or per-arb-output
counters of how many done packets to drain (doesn't generalize across
topologies, complicates timing). Instead, we drop **everything** in one
cycle and re-fill from the preserved producer state. The §7.3 gating
(`flush_trigger = begin_pulse_any && frame_drained`) is what makes this
safe — at flush time the buffers contain ONLY stale done-sentinels
(no fresh data possible), because `frame_drained=1` is the post-condition
"producer has drained AND every active consumer has been told once".

**Cost scaling with N:**

| Quantity                                       | Scaling  | Typical (N≈8)  |
| ---------------------------------------------- | -------- | -------------- |
| Flush cost (cycles to empty all stages)        | `O(1)`   | 1 cycle        |
| Refill latency (post-flush first-packet)       | `O(N)`   | ~8 cycles      |
| Flush hardware (gates added per arb)           | `O(1)`   | 1 OR-gate per buffer + few flops per arb |
| Begin_pulse combinational depth (timing risk)  | `O(N)`   | 6-8 levels of OR-gates — easily one cycle for typical clock targets |

The `O(N)` combinational propagation on `begin_pulse` is the only
timing-relevant cost — and even at N=16+ this stays under typical
clock targets (a 16-fanin OR-tree is ~4 levels of LUTs, single-cycle
on FPGA targets; single-cycle ASIC at standard frequencies). If a
future deeper topology required pipelining `begin_pulse`, the design
would need to add corresponding pipelining to `flush_trigger`
distribution to keep all stages flushing in the same cycle relative to
the registered pulse — but that's a future-proofing exercise, not a
v1 concern.

**What we explicitly do NOT do (and why):**

| Alternative                                          | Why rejected |
| ---------------------------------------------------- | ------------ |
| Per-packet `frame_id` tag on the bus                 | Bus interface change (violates §7.2). Would also require synchronized frame_id counter across producer + consumers. |
| Per-arb drain-counter ("emit K done packets then stop") | Producer can't know fan-out (the cluster's `NUM_SOCKETS × SOCKET_SIZE × NUM_WARPS` isn't visible at the slice). Doesn't compose across multi-cluster. |
| Consumer-side per-warp `seen_begin` filter           | Up to 64 × NUM_CORES flops. Worse: can't terminate in the zero-work-instance case (W1) without an extra signal — see §11 for why. |
| Wait for natural drain (consumer reads until req_valid drops) | With sticky-done, the bus NEVER stops emitting `{done=1}`. Producer is always "drained and emitting." No natural termination. |

### 7.4 begin_pulse — unchanged, but documented

`begin_pulse` already works correctly: every consumer's `VX_raster_unit`
pulses on `vx_rast_begin` retirement; the per-cluster arb chain
OR-reduces all pulses; broadcast back to every `raster_core`. Each
`raster_core` latches independently via `fetch_triggered`. This composes
cleanly across the new fanout module — no changes needed.

The proposal does add one clarification:

- `vx_rast_begin` MUST be issued at least once per frame by **at least one
  warp on at least one core**. The current ISA docs already say this, but
  with N raster_cores per cluster, it's worth re-emphasizing that a SINGLE
  begin_pulse from a single warp is sufficient to start ALL N producers in
  ALL clusters (begin_pulse fans out globally via the DCR-style arb tree).

### 7.5 Multi-cluster tile partitioning — already handled in RTL, needs simx fix

**Status: RTL is correct. SimX is wrong.**

The RTL already stripes tiles across clusters via the existing
`INSTANCE_IDX` / `NUM_INSTANCES` machinery — see
[hw/rtl/VX_graphics.sv:229-230](hw/rtl/VX_graphics.sv#L229-L230):

```sv
INSTANCE_IDX  (CLUSTER_ID * `VX_CFG_NUM_RASTER_CORES + i)
NUM_INSTANCES (`VX_CFG_NUM_CLUSTERS * `VX_CFG_NUM_RASTER_CORES)
```

With NUM_CLUSTERS=2 and NUM_RASTER_CORES=1 per cluster:
- Cluster 0, instance 0: INSTANCE_IDX=0, NUM_INSTANCES=2 → processes tiles 0,2,4,...
- Cluster 1, instance 0: INSTANCE_IDX=1, NUM_INSTANCES=2 → processes tiles 1,3,5,...

DCR broadcast still gives every cluster the same `tile_count` /
`tbuf_addr` — but each instance's `VX_raster_mem` then picks its own
stripe (`start_tile_count = (tile_count + N - 1 - IDX) >> log2(N)`). No
host-side change required.

**SimX bug to fix:** the current simx `cluster.cpp` passes
`instance_idx=r` (per-cluster index) and `num_instances=NUM_RASTER_CORES`
(per-cluster count). This loses the cluster offset, so both clusters
process the same stripe — duplicate rendering on multi-cluster
configurations. Fix:

```cpp
// sim/simx/cluster.cpp — replace
auto raster_core = RasterCore::Create(sname, simobject_, r, kRasterCores);
// with
uint32_t global_idx       = cluster_id * kRasterCores + r;
uint32_t global_instances = VX_CFG_NUM_CLUSTERS * kRasterCores;
auto raster_core = RasterCore::Create(sname, simobject_, global_idx, global_instances);
```

`cluster_id` is already available via `simobject_->id()` (used elsewhere
in the same constructor for socket numbering).

No other changes needed — `RasterCore::Impl::kick_off_load` and
`start_load_pids` already use `instance_idx_` and `num_instances_`
correctly for the stripe arithmetic.

### 7.6 ~~Reset-clean DCR registers~~ — REMOVED

Original proposal text dropped: DCRs live in BRAM (no reset wire), and
software is the sole DCR initializer by design. Adding a reset would
force synthesis to back the DCRs with flip-flops or use a slower
BRAM-with-reset primitive — both unacceptable for area/timing.

The original motivation (avoiding stale `tile_count` between runs) is
covered by the §7.3 begin_pulse-based frame indicator without any
DCR-side change. The producer's `fetch_triggered` only latches HIGH on a
fresh `begin_pulse`, and the arb's per-output sticky-done only sets after
a real done-handshake — neither relies on DCR state surviving reset.

---

## 8. SimX Mirror

`sim/simx/raster/raster_core.cpp` and `sim/simx/cluster.cpp` need parallel
changes:

1. `RasterCore` already accepts `instance_idx`, `num_instances` (added in
   the current branch). Keep.
2. Cluster instantiates `NUM_RASTER_CORES` `RasterCore`s. Done.
3. **Fix cluster_id offset in `RasterCore` instantiation** (§7.5):
   pass `cluster_id * NUM_RASTER_CORES + r` and
   `NUM_CLUSTERS * NUM_RASTER_CORES` as instance_idx / num_instances.
4. **Extend `RasterBusArbiter` to handle fan-out direction** (§7.1).
   `TxRxArbiter` currently uses a `TxArbiter` internal arbiter that
   assumes `num_inputs ≥ num_outputs`. Add a fan-out path for
   `num_inputs < num_outputs`: per-output consumer-side round-robin pull
   from any valid producer.
5. **Add per-output sticky-done state inside `RasterBusArbiter`** (§7.2):
   `consumer_served[NUM_OUTPUTS]` array of bool. Set when output `o`
   acks a `done=true` response. Clear all on begin_pulse arrival from
   any output.
6. **Mirror the `pending_begin` drain in `RasterUnit`** (§7.3 part 2):
   when the SFU dispatches a vx_rast and the bus has `done=true` while
   `pending_begin=true`, discard the response and reissue without
   completing the SFU instruction.
7. `Cluster::dcr_write` already broadcasts to every `raster_core`. Keep.
8. `Cluster::raster_cores()` accessor — keep.

**Validation discipline:** the simx model is the gold reference for the
RTL. Every RTL change in this redesign must have a corresponding simx
change, with side-by-side tests in each `NUM_RASTER_CORES × cores ×
NUM_CLUSTERS` config (per [feedback_simx_as_rtl_oracle]).

---

## 9. Test Plan

`ci/regression.sh.in` adds the following sweeps to the `graphics()` block
(all PASS expected on both simx and rtlsim):

```
# 1:1 (current default)
NUM_RASTER_CORES=1, cores=1, sockets=1, ...

# Fan-in (works in current code, verifies regression)
NUM_RASTER_CORES=4, cores=2, SOCKET_SIZE=2 → sockets=1, 4 → 1

# 1:K fan-out — the broken case the proposal fixes
NUM_RASTER_CORES=1, cores=4, SOCKET_SIZE=1 → sockets=4, 1 → 4
NUM_RASTER_CORES=2, cores=4, SOCKET_SIZE=1 → sockets=4, 2 → 4 (K=2)

# Multi-cluster (Scenario C — needs §7.5 host partitioning)
clusters=2, cores=2, NUM_RASTER_CORES=1, SOCKET_SIZE=1
                                       → 2 raster_cores, 4 total sockets
```

For each config, the test compares `gfx_draw3d box.cgltrace 128x128` output
between simx and rtlsim and against a deterministic reference. **The
reference is regenerated from simx output** (simx is the oracle).

Additional cross-validation per
[feedback_simx_as_rtl_oracle](../../../home/blaisetine/.claude/projects/-home-blaisetine-dev/memory/feedback_simx_as_rtl_oracle.md):
on any divergence, dump `DBG_TRACE_RASTER` from both backends and diff the
per-quad stream.

---

## 10. Implementation Phases

| Phase | Deliverable                                                                              | Backend    |
| ----- | ---------------------------------------------------------------------------------------- | ---------- |
| 1     | Extend `VX_raster_arb` in-place to handle `NUM_INPUTS < NUM_OUTPUTS` (fan-out direction) | RTL        |
| 2     | Add per-output `consumer_served[o]` sticky-done state inside `VX_raster_arb` (§7.2)      | RTL        |
| 3     | begin_pulse → consumer_served clear in `VX_raster_arb` (§7.3 part 1)                     | RTL        |
| 4     | `pending_begin` drain in `VX_raster_unit` (§7.3 part 2)                                  | RTL        |
| 5     | Fix simx cluster_id offset in raster_core instantiation (§7.5)                           | SimX       |
| 6     | Mirror §7.2/§7.3 in simx's `RasterBusArbiter` (per-output sticky-done + begin_pulse clear) | SimX     |
| 7     | New regression configs (Scenario B fan-out + Scenario C multi-cluster with blend ON)     | Tests      |

**Minimum viable set:** Phases 1-6 — fixes the 1789-pixel failure and the
next-frame drain hazard. Phase 7 broadens coverage to expose any remaining
gaps (multi-cluster duplicate-rendering with blend on, multi-cluster +
multi-raster-core, fewer-tiles-than-instances).

**Removed from earlier draft:**
- ~~Separate `VX_raster_fanout` module~~ — folded into `VX_raster_arb` (single interface).
- ~~`flush`-on-DCR-write propagation~~ — replaced by begin_pulse-driven sticky-done clear (no DCR-write listener needed).
- ~~Reset-clean `raster_dcrs`~~ — disallowed (BRAM design constraint).
- ~~Host-side multi-cluster Binning changes~~ — RTL already handles via INSTANCE_IDX.

---

## 11. Open Questions

1. **Should `vx_rast_begin` be required to be issued by every warp, or any
   warp?** Today: any-warp is fine (begin_pulse is OR-reduced). The proposal
   keeps that. But if we ever wanted per-warp resume semantics for
   preemption, this would change.

2. **Register cost in `VX_raster_arb`.** Per-output `consumer_served[o]`:
   one flop per cluster output, so `NUM_SOCKETS_PER_CLUSTER` flops total
   (typically ≤ 16). Plus the `pending_begin` flop in each
   `VX_raster_unit` (one per core). Total cluster-wide: O(NUM_CORES) flops
   — negligible.

3. **What if a consumer never issues `vx_rast`?** (e.g., a core whose CTA
   finished before reaching the raster loop). Without consumer requests,
   the arb's `consumer_served[o]` for that output stays LOW. The producer
   doesn't stall on un-asking consumers (the arb only routes when both
   sides have valid+ready). On the next `begin_pulse`, the flop is
   cleared anyway — no leak.

4. **What if `vx_rast_begin` is issued multiple times within a frame?**
   The producer's `fetch_triggered` dedupes (no re-fetch); the arb's
   `consumer_served` clear is idempotent (already-cleared stays cleared);
   the consumer's `pending_begin` re-arms (harmless — first non-done
   packet still clears it). All paths converge.

5. **Should producer-side `req_valid` stay HIGH when `done=1` and arb's
   `consumer_served[o]` is all-set?** Today the producer's slice keeps
   `req_valid=1, done=1` continuously when drained. With the arb sourcing
   sticky-done from its own state, the producer's stream is no longer
   load-bearing for done — we could allow it to deassert `req_valid` once
   ALL its outputs have been served. Likely not worth it (single bit of
   power savings, complexity not justified) — defer.

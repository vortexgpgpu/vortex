# MMU + PTW Performance Optimization Proposal

**Scope:** [hw/rtl/mem/VX_mmu.sv](../../hw/rtl/mem/VX_mmu.sv), [VX_mmu_tlb.sv](../../hw/rtl/mem/VX_mmu_tlb.sv), [VX_mmu_ptw.sv](../../hw/rtl/mem/VX_mmu_ptw.sv)
**Target:** GPU memory pipeline (high-throughput, SIMD-wide, non-blocking)
**Status:** Proposal — not yet implemented
**Related:** [vm_migration_proposal.md](vm_migration_proposal.md), [../vm.md](../vm.md)

---

## 1. Why this matters

The MMU sits on the critical path of every load, store, and instruction
fetch. On a GPU, a single warp's LSU emits `NUM_LSU_LANES` parallel
requests per cycle (typically 4 or 8). Anything in the MMU that
serializes these lanes, or that blocks the pipeline on a miss, costs
throughput proportional to `(NUM_LSU_LANES × cycles_lost)`.

The current design is **functionally correct but built like a CPU TLB**:
one shared CAM, one in-flight miss, one in-flight PTW walk, fully
sequential page-table fetches. For a GPU dataflow this leaves a lot of
throughput on the table.

This proposal lays out 6 ranked optimizations targeting the
high-throughput, non-blocking, pipelined model the rest of the GPU
already uses.

---

## 2. Bottleneck inventory (current design)

### 2.1 TLB serializes lanes (the big one)

[VX_mmu_tlb.sv:76-92](../../hw/rtl/mem/VX_mmu_tlb.sv#L76)

```sv
VX_stream_arb #(
    .NUM_INPUTS  (NUM_REQS),
    .NUM_OUTPUTS (1),
    ...
) req_serialize_arb (...);
```

The TLB has one CAM and one lookup port. `NUM_REQS` (= `DCACHE_NUM_REQS`)
lanes round-robin through it, **one per cycle**. Followed by a
1-to-N deserialize switch on the way out.

For `NUM_REQS = 4`, peak translation throughput drops from `4 trans/cyc`
(theoretically achievable from N parallel CAMs) to `1 trans/cyc`. **3×
slowdown for SIMD-wide loads/stores when VM is on**, even on TLB hit.

### 2.2 TLB blocks the entire pipeline on a miss

[VX_mmu_tlb.sv:249-303](../../hw/rtl/mem/VX_mmu_tlb.sv#L249)

```sv
TLB_READY:
    if (input_handshake) begin
        if (tlb_hit) ...
        else begin
            ...
            state <= TLB_PTW_WAIT;
        end
    end

assign ser_req_ready = (state == TLB_READY) && deser_req_ready;
```

`ser_req_ready` is gated on `state == TLB_READY`. While a miss is being
walked (`TLB_PTW_WAIT`) and replayed (`TLB_REPLAY`), **no other lane
can translate** — even lanes that would hit. One missed VPN stalls all
subsequent traffic until PTW completes (~10-30 cycles for a 2-level walk).

Combined with §2.1, a single miss serializes the entire dcache port for
the whole walk window.

### 2.3 PTW is fully sequential — one walk at a time

[VX_mmu_ptw.sv:103-139](../../hw/rtl/mem/VX_mmu_ptw.sv#L103) 6-state FSM:
`IDLE → L1_REQ → L1_RESP → L0_REQ → L0_RESP → FILL → IDLE`. Each step
gates on the next. For a single PTW:

- L1 PTE fetch: best case 1 cycle req + N cycles cache latency (often 4-30+ if it misses)
- L0 PTE fetch: same again
- Fill back to TLB: 1 cycle

A best-case walk is ~6-8 cycles; a realistic walk hitting the L1 dcache
is ~15-25 cycles; an L1-miss walk can hit hundreds.

`miss_ready = (state == PTW_IDLE)` — only one walk active. If two
warps miss on different pages back-to-back, the second waits for the first
to complete *plus* the TLB serialization+replay.

### 2.4 No coalescing of duplicate VPN misses

If 4 lanes of the same warp all miss on the same VPN (very common —
that's exactly what GPU SIMD does), each miss triggers a separate
PTW walk after re-acquiring the serialized TLB port.

### 2.5 PTE re-fetching across walks

The L1 page-table directory has 1024 entries each covering 4 MB. A
typical kernel touches a small handful of L1 entries. The PTW currently
re-fetches the L1 PTE on **every** walk — even when 99% of consecutive
walks share the same L1 entry. Wasted memory traffic + ~50% of every
walk's latency.

### 2.6 Linear-scan victim search

[VX_mmu_tlb.sv:189-204](../../hw/rtl/mem/VX_mmu_tlb.sv#L189) — combinational
loop over all 32 entries to find an invalid slot, then over all 32 to
find a non-MRU. Synthesizes to a wide priority encoder; OK at 32, scales
poorly past 64.

---

## 3. Proposed optimizations (ranked by impact)

### Opt 1 — Per-lane parallel TLBs (mandatory for GPU throughput)

**Replace** the 1×[CAM serialized] arrangement with **N small CAMs**, one
per dcache lane. Each lane gets a private 8-entry TLB; lookups are fully
parallel and 1-cycle.

Pseudocode:
```sv
for (genvar i = 0; i < NUM_REQS; i++) begin : g_per_lane_tlb
    VX_mmu_tlb_l1 #(.SIZE(8)) tlb_l1 (
        .req_addr (lsu_mem_if[i].req_data.addr),
        .req_valid(lsu_mem_if[i].req_valid),
        .hit      (l1_hit[i]),
        .pa       (l1_pa[i]),
        .miss     (l1_miss[i]),
        .miss_vpn (l1_miss_vpn[i])
    );
end
```

**Impact:** Removes §2.1 entirely. Per-lane peak throughput becomes
`1 trans/cyc/lane` = `NUM_REQS trans/cyc` core-wide. Eliminates the
serialize/deserialize stream arbiters (and their tag-width inflation —
saves `CLOG2(NUM_REQS) + ARB_BITS` per tag bit).

**Cost:** N × 8 entries vs 1 × 32 entries — same total entries, but each
CAM is smaller (faster). Per-lane CAM is 8 × ~30 bits = trivial LUT/FF on
FPGA. Replication is the cost; throughput is the gain.

**Tradeoff:** Per-lane TLBs don't share entries. To prevent capacity
under-utilization, pair with Opt 4 (shared L2 TLB).

### Opt 2 — Non-blocking TLB miss handling (MSHR-style)

**Replace** the `TLB_READY → TLB_PTW_WAIT → TLB_REPLAY` blocking FSM with
an MSHR-style "miss queue": misses get tagged and parked in a small
in-flight table; the TLB stays open for hits. When the PTW returns a fill,
the matching MSHR entry replays.

```sv
reg [NUM_MSHR-1:0]            mshr_valid;
reg [NUM_MSHR-1:0][VPN_W-1:0] mshr_vpn;
reg [NUM_MSHR-1:0][TAG_W-1:0] mshr_tag;
// miss → allocate MSHR entry, dispatch to PTW
// hit  → bypass MSHR, forward to dcache
// fill → CAM-match in-flight VPN, replay all matching MSHRs
```

**Impact:** Removes §2.2. A missed VPN no longer stalls hitting lanes.
A long PTW walk now overlaps with downstream cache accesses for hitting
traffic — the TLB stops being a head-of-line blocker.

**Cost:** ~16-entry MSHR per TLB. CAM-match VPN on fill (small). Worth it.

### Opt 3 — Multi-walk PTW (multiple concurrent walks)

**Replace** the single-walk FSM with N independent walk slots. Each slot
holds `{state, pending_vaddr, l1_ppn, ...}`. The PTW issues memory
requests round-robin across active slots; responses are matched to slots
via the `tag` field carrying the slot id.

```sv
typedef struct packed {
    ptw_state_t      state;
    logic [31:0]     pending_vaddr;
    logic [PPN_W-1:0] l1_ppn;
    ...
} walker_slot_t;
walker_slot_t walkers [NUM_WALKERS];
```

`miss_ready` becomes `|free_slot_mask` — accept new misses whenever any
slot is free. Use the dcache TAG to multiplex slot ids on memory
responses.

**Impact:** Removes §2.3. With `NUM_WALKERS = 4`, four divergent VPN
misses can walk in parallel. PTW throughput scales linearly with
walker count, bounded by memory bandwidth.

**Cost:** N × walker state (~80 bits each). Plus a small request arbiter
on the memory side.

### Opt 4 — Two-level TLB (L1 per-lane + L2 shared)

Pair Opt 1 (small per-lane L1 TLBs) with a **shared, larger L2 TLB** at
the core level. L1 miss → check L2 → fill L1 from L2 (or trigger PTW
on L2 miss).

```
                       L1 TLB (per-lane, 8 entries, 1-cycle, fully assoc CAM)
LSU lane 0 ─────────►  ┌─────────┐
                       │  hit?   ├─► PA to dcache
                       └────┬────┘
                            │ miss
                            ▼
                       ┌──────────────────────────┐
LSU lane 1 ─────────►  │  L2 TLB (shared, 256     │   2-3 cycles
                       │  entries, 4-way SRAM)    │   BRAM-friendly
                       │      hit? ──► fill L1   │
                       └──────────┬───────────────┘
                                  │ miss
                                  ▼
                       ┌──────────────────────────┐
                       │  PTW (multi-walker, §3.3)│
                       └──────────────────────────┘
```

**Impact:** Restores effective capacity (256+ entries) without paying the
serialization cost of a single big CAM. L2 lookup on L1 miss adds 2-3
cycles; PTW only fires on L2 miss. Industry-standard for GPUs (e.g.
Volta has a per-SM L1 TLB + L2 TLB).

**Cost:** L2 TLB is a small SRAM (256 × ~50 bits = ~13 Kbits). Maps to
1 BRAM block on FPGA. A SRAM-backed 4-way set-assoc lookup is 2-3 cycles
which is acceptable on the L1-miss path.

### Opt 5 — L1-PTE cache inside the PTW

Add a tiny direct-mapped cache of recently-fetched L1 PTEs (the
"directory" level for SV32; "L2 directory" for SV39). On the next PTW,
check it first; on hit, skip the L1 fetch and go straight to L0_REQ.

```sv
// Inside VX_mmu_ptw.sv
reg [PTE_CACHE_SIZE-1:0]                   pte_cache_valid;
reg [PTE_CACHE_SIZE-1:0][VPN1_W-1:0]       pte_cache_vpn1;
reg [PTE_CACHE_SIZE-1:0][PPN_W-1:0]        pte_cache_ppn;

wire l1_skip = pte_cache_valid[idx] && (pte_cache_vpn1[idx] == vpn1);
state_next = l1_skip ? PTW_L0_REQ : PTW_L1_REQ;
```

`PTE_CACHE_SIZE = 8` direct-mapped is enough. Indexed by `vpn1` low bits.

**Impact:** Removes §2.5. Workloads with spatial locality (which is most
of them) skip the L1 fetch on ~80%+ of walks — **walk latency roughly
halves**. Memory bandwidth to PT region drops by similar amount.

**Cost:** ~8 × 32 bits = trivial.

### Opt 6 — Bloom-filter or content-hash victim selection

Replace the wide priority encoder in §2.6 with either:

- **Bloom-filter-aided LRU**: use a small hash to identify likely-dead
  entries. Cheaper combinational tree.
- **Tree-PLRU**: standard binary tree, same as `VX_cache_repl` already
  uses. log2(SIZE) deep, scales much better.

This only matters if TLB grows past 64 entries, which the L2 TLB in Opt 4
will. Trivial implementation; well-understood.

### Bonus — perf counter fix

[VX_mmu_tlb.sv:495](../../hw/rtl/mem/VX_mmu_tlb.sv#L495) sets
`mmu_perf.ptw_walks = perf_tlb_misses` — these are the same number today
(every miss triggers exactly one walk), but with Opt 4 they diverge:
a TLB miss may hit the L2 TLB without triggering a walk. Wire `ptw_walks`
from the PTW side instead.

[VX_mmu_tlb.sv:496](../../hw/rtl/mem/VX_mmu_tlb.sv#L496) sets
`mmu_perf.ptw_latency = '0` — the PTW already produces `perf_ptw_latency`
on its own port; just route it.

---

## 4. Suggested rollout order

Each optimization is independent and produces measurable speedup on its
own. Recommended order:

| Step | Opt | Why first |
|---|---|---|
| 1 | Opt 1 + Opt 4 (per-lane L1 + shared L2) | Biggest single win for GPU SIMD throughput. The two are complementary and naturally land together. |
| 2 | Opt 2 (non-blocking miss) | Removes head-of-line stall. Independent of Opt 3. |
| 3 | Opt 3 (multi-walker PTW) | Pairs with Opt 2 — once misses are non-blocking, having one walk slot becomes the next bottleneck. |
| 4 | Opt 5 (PTE cache in PTW) | Cheap, ~50% walk latency reduction with negligible area. |
| 5 | Opt 6 (replacement) | Only matters once Opt 4's larger L2 TLB lands. |
| 6 | Bonus perf fix | Cleanup. |

---

## 5. Expected wins (back-of-envelope)

Assuming `NUM_LSU_LANES = 4`, baseline workload with ~5% TLB miss rate
and ~20-cycle average walk latency on L1-dcache hit:

| Metric | Today | After Opts 1+2+3+4+5 |
|---|---|---|
| Peak TLB hit throughput | 1 trans/cyc | 4 trans/cyc (4×) |
| Cycles lost per miss | 20-25 (whole pipeline stalled) | 0 (other lanes flow through) |
| Walk latency | ~20 cyc | ~10 cyc (PTE cache halves it) |
| Concurrent walks | 1 | NUM_WALKERS (4×) |
| Effective TLB capacity | 32 | 32 (per-lane sum) + 256 (L2) |
| TLB perf overhead | ~5-10% IPC drop on memory-bound | ~0.5-2% target |

The combination should make the MMU effectively transparent for hit-rate
≥ 95% workloads, and degrade gracefully for randomized-VA stress tests.

---

## 6. Out of scope for this proposal

- **PMP / page protection enforcement**: still TODO; would land on the
  PTW side regardless of these optimizations.
- **ASID support** (multiple address spaces, no flush on context switch):
  needs a separate proposal — touches CSR encoding, kernel/runtime, and
  the TLB tag.
- **Superpage handling beyond what already exists**: current TLB has the
  `page_level` field but PTW always returns 4KB pages. Real superpage
  promotion is a separate feature.
- **SimX-side perf counter modeling for cycle-accurate MMU**: SimX
  currently does functional-only translation. Cycle-accurate MMU
  modeling would mirror these RTL optimizations on the simx side.

---

## 7. Open questions

1. **Per-lane L1 TLB size**: 4? 8? 16? Sensitive to workload — 8 is a
   reasonable starting default; sweep on regression.
2. **L2 TLB associativity**: 4-way is a safe industry default; could go
   8-way at the cost of a wider tag check.
3. **Walker count**: 2 or 4 walkers? 4 doubles area for diminishing
   returns past 2 in low-divergence workloads.
4. **Opt 1 + DXA interaction**: DXA currently goes through the per-core
   dcache MMU (per Phase 5). With per-lane L1 TLBs, DXA gets one of the
   lane-TLBs (or its own). Probably want DXA on its own dedicated L1 TLB
   to avoid evicting LSU entries.

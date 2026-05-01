# DXA SimX v3 — Proposal

**Date:** 2026-04-30
**Status:** Draft
**Related:**
[simx_v3_proposal.md](simx_v3_proposal.md),
[wgmma_simx_v3_proposal.md](wgmma_simx_v3_proposal.md),
`feedback_simx_tlm_design`, `feedback_simx_perf_goal`,
`project_simx_tlm_refactor`.

---

## 1. Constraints (load-bearing)

The same three rules that govern WGMMA SimX v3 govern DXA. Any proposal
that breaks one is wrong.

1. **NoC-only memory access.** Every fetch and store flows through the
   channel hierarchy (`MemReq`/`MemRsp` to L2; `MemReq`/`MemRsp` to
   `LocalMem`). No `core_->mem_read`, no `core_->mem_write`, no
   synchronous backing-store paths. **Note:** unlike WGMMA, DXA *writes*
   to LMEM — the constraint applies symmetrically to write data, byte
   enables, and the completion-flag side-band.
2. **Functional and timing coupled.** The byte that lands in an LMEM
   word is the byte that the GMEM `MemRsp` carried, and it lands the
   cycle the `LocalMem::Inputs` channel accepts the corresponding
   `MemReq`. No "data already in LMEM at issue, timing replay later."
3. **Mirror RTL at module *correspondence*, not at storage.** The RTL
   split is `dxa_unit` (per-SFU dispatch interface, one per core) →
   `dxa_core` (cluster-shared engine: arbiter + queue + dispatch +
   workers + GMEM/LMEM arbs). SimX must mirror that boundary so debug
   and perf reasoning carry over. Internal storage policy is C++'s
   choice — e.g. an in-engine line buffer is fine, but a shadow tile
   image of LMEM is not.

---

## 2. Why the current SimX DXA is broken

[sim/simx/dxa/dxa_core.cpp](../../sim/simx/dxa/dxa_core.cpp),
[sim/simx/sfu_unit.cpp](../../sim/simx/sfu_unit.cpp),
[sim/simx/cluster.cpp](../../sim/simx/cluster.cpp).

### 2.1 Compile-broken wiring

[cluster.cpp:108-109](../../sim/simx/cluster.cpp#L108) binds
`dxa_core_->lmem_req_out` (`SimChannel<DxaCore::SmemReq>`) to
`local_mem()->dxa_req_in` — but
[mem/local_mem.h](../../sim/simx/mem/local_mem.h) declares no
`dxa_req_in`. The DXA → LMEM channel is wired to a member that does not
exist; the SmemReq packet type is private to DxaCore and incompatible
with `LocalMem::Inputs` (`SimChannel<MemReq>`). The branch does not
build with `EXT_DXA_ENABLE`.

### 2.2 Rule 1 violations (NoC bypass)

| # | Location | Defect |
|---|----------|--------|
| D1 | [dxa_core.cpp:247-249](../../sim/simx/dxa/dxa_core.cpp#L247) | `gmem_read` ≡ direct `core->mem_read` — backdoor functional read of global memory. |
| D2 | [dxa_core.cpp:351](../../sim/simx/dxa/dxa_core.cpp#L351), [:360](../../sim/simx/dxa/dxa_core.cpp#L360), [:363](../../sim/simx/dxa/dxa_core.cpp#L363) | `execute_copy` does `core->mem_read`/`core->mem_write` for every element — the entire functional copy is performed via backdoor at *execute* time, before any cycle is simulated. |
| D3 | [dxa_core.cpp:466-469](../../sim/simx/dxa/dxa_core.cpp#L466) | `barrier_event_release` is called directly from `DxaCore::tick`, bypassing `LocalMem`. RTL passes the bar_addr as a flag bit on the last LMEM write packet ([VX_dxa_smem_wr.sv:298-313](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L298)) and `VX_dxa_completion` raises the barrier from the LMEM-side completion. |

### 2.3 Rule 2 violations (functional/timing decoupled)

The SmemReq packet ([dxa_core.h:30-36](../../sim/simx/dxa/dxa_core.h#L30))
is documented as "timing-only request to LocalMem DXA channel (no data
payload)." Data is written by `execute_copy` synchronously at SFU issue
([sfu_unit.cpp:79](../../sim/simx/sfu_unit.cpp#L79)); SmemReq is replayed
later for "timing." This is the textbook split that v3 §3.3 forbids:
the LMEM word a downstream WGMMA reads is correct *before* the GMEM
fetch latency has been simulated. Any consumer (TCU `wgmma`, an LSU load)
sees data ahead of cycle.

### 2.4 Architectural inversion (mirroring RTL)

| | RTL | Current SimX |
|---|---|---|
| Per-SFU interface | `VX_dxa_unit` (one per core, decodes lanes 0–3, emits dxa_req + txbar) — [VX_dxa_unit.sv](../../hw/rtl/dxa/VX_dxa_unit.sv) | inline in `SfuUnit::on_tick` ([sfu_unit.cpp:62-89](../../sim/simx/sfu_unit.cpp#L62)) |
| Cluster-shared engine | `VX_dxa_core` = req_arb (NUM_REQS:1) + queue + desc_table + dispatch (1:N) + N workers + gmem/lmem arbs — [VX_dxa_core.sv](../../hw/rtl/dxa/VX_dxa_core.sv) | `DxaCore::Impl` with hand-rolled queue + slices — has the queue/arb shape but the wrong functional model |
| Worker pipeline | 5-stage: setup → addr_gen → gmem_req → rsp_buf + smem_wr → completion — [VX_dxa_worker.sv](../../hw/rtl/dxa/VX_dxa_worker.sv) | `tick_slice()` with two states `GMEM_FETCH` / `SMEM_WRITE` — collapses the address-gen + rsp-reorder pipeline into a flat replay of an `exe_data` precomputation |
| Completion / barrier | last LMEM write carries `{notify_smem_done, bar_addr}` flags; LMEM-side `VX_dxa_completion` releases barrier — [VX_dxa_pkg.sv:96-101](../../hw/rtl/dxa/VX_dxa_pkg.sv#L96), [VX_dxa_smem_wr.sv:298-313](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L298) | `barrier_event_release` called directly from DxaCore tick |

Beyond the code-shape mismatch, the current SimX inverts *when* work
happens. RTL: SFU dispatches a `dxa_req`, immediately frees the warp
(via `txbar` reservation), the engine runs over many cycles, the LMEM
write's flag releases the barrier. SimX: the entire copy happens
synchronously inside `SfuUnit::on_tick`, then a no-op timing replay runs.
Symptom: any test that depends on DXA latency being visible (a kernel
that races a non-DMA load against a DMA write to the same SMEM word, or
a perf comparison vs RTL) is wrong.

### 2.5 Other smaller defects

- The `slot` returned by `execute_copy` ([dxa_core.cpp:161-244](../../sim/simx/dxa/dxa_core.cpp#L161))
  bundles emulation results (`gmem_lines`, `smem_blocks`) into the same
  TraceData that submit consumes. Two orthogonal sets of fields, mutated
  at different times — fragile.
- `kMaxOutstanding = 8` is hard-coded ([dxa_core.cpp:88](../../sim/simx/dxa/dxa_core.cpp#L88))
  and unrelated to `DXA_MAX_INFLIGHT` (RTL knob).
- Multicast SMEM-write replay assumes `bar_id + cta_warp_idx` ([dxa_core.cpp:497](../../sim/simx/dxa/dxa_core.cpp#L497))
  but the encoding is owned by `bar_decode_id` and `BAR_ID_SHIFT` in
  RTL — duplicated truth.

---

## 3. Target architecture

The design mirrors RTL at the same module boundary; storage is C++'s
choice.

```
┌─────────────────────────────────────────────────────────────────┐
│ Core (one per core)                                             │
│                                                                 │
│  SfuUnit                       DxaUnit (per SFU block)          │
│  ┌────────────┐  DxaType ┌─────────────────────┐                │
│  │ dispatch   ├─────────▶│ decode lanes 0..3 → │                │
│  │ to FU      │          │   DxaReq packet     │                │
│  └────────────┘          │ → submit + barrier  │                │
│                          │   attach            │                │
│                          │ → FU result (latency)│                │
│                          └──────┬──────────────┘                │
│                                 │ DxaReq channel                │
│                                 ▼                               │
│                                bind to cluster.dxa_req_in[cid]  │
│                                                                 │
│  LocalMem ◀────────────── lmem_req_in[cid]   (from DxaCore)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Cluster                                                         │
│                                                                 │
│  DxaCore                                                        │
│   in:  dxa_req_in[NUM_CORES_PER_CLUSTER]   ← per-core DxaUnits  │
│        dcr_bus  (descriptor writes)                             │
│   ├── ReqArb (NC:1) → ReqQueue (depth=DXA_QUEUE_SIZE)           │
│   ├── DescTable (read by queue head)                            │
│   ├── Dispatch (1:N) → Workers[NUM_DXA_UNITS]                   │
│   │     Worker pipeline (per slice):                            │
│   │       setup → addr_gen → gmem_req →                         │
│   │       rsp_buf + smem_wr → completion                        │
│   ├── GmemArb (N:GMEM_OUT_PORTS) → gmem_req_out → L2-arb        │
│   └── LmemArb (N:NC) → lmem_req_out[cid]                        │
│                            └─▶ each core's LocalMem.Inputs[port]│
│                                                                 │
│  out: gmem_req_out[GMEM_OUT_PORTS] / gmem_rsp_in[]              │
│       lmem_req_out[NUM_CORES_PER_CLUSTER]                       │
│        - bound to LocalMem.Inputs.at(port_dxa) on each core     │
│        - response port unused (write-only, no rsp); flags carry │
│          the barrier-release side-band                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Modules (C++)

| RTL module | SimX module | Owns |
|---|---|---|
| `VX_dxa_unit` | new `DxaUnit` (`sim/simx/dxa/dxa_unit.{h,cpp}`) | per-SFU lane-decode + DxaReq submit + result-port write-back. **No** functional reads/writes. |
| `VX_dxa_core` (top) | `DxaCore` (refactored) | request arb, queue, descriptor table, dispatch, GMEM arb, LMEM arb, perf rollup |
| `VX_dxa_desc_table` | `DxaCore::DescTable` (private) | descriptor storage (DCR-written) |
| `VX_dxa_worker` (5 stages) | `DxaCore::Worker` (private; one per slice) | setup → addr_gen → gmem_req → rsp_buf + smem_wr |
| `VX_dxa_completion` | flag handling inside the LMEM bus packet | last-write flag triggers `barrier_event_release` at the LMEM consumer side |

`DxaUnit` is the per-SFU object. `DxaCore` is the cluster-shared
SimObject. Each `Worker` is a `Worker` struct inside `DxaCore`, ticked
from `DxaCore::on_tick` — same pattern as the existing `slices_` vector
but with a real pipeline.

### 3.2 Channel types

```cpp
// New, owned by DxaUnit's per-core DxaReq channel.
struct DxaReq {
  uint32_t core_id;
  uint32_t wid;
  uint64_t uuid;
  uint32_t desc_slot;       // from meta[0:DESC_SLOT_W-1]
  uint32_t bar_id;          // decoded
  uint32_t cta_mask;
  uint64_t smem_addr;
  uint32_t coords[5];
  Core*    core;            // for barrier release routing
};

// LMEM write goes on the existing MemReq channel that LocalMem already
// accepts. The completion flag rides in MemReq::flags.
//   flags[0]                = notify_smem_done
//   flags[1 +: BAR_ADDR_W]  = bar_addr (when notify_smem_done=1)
// (Mirror DXA_LMEM_FLAGS_W in RTL.)
```

DXA reuses `LocalMem::Inputs` exactly like RTL does — DXA's LMEM port
is one more entry in the existing per-core port list (today: `LSU_NUM_REQS`
LSU ports + 1 TCU port; add + 1 DXA port). No new LocalMem channel,
no new packet type for SMEM data.

### 3.3 Worker pipeline (SimX)

Five sub-states ticked in reverse pipeline order each cycle (mirrors
RTL submodules; matches the v3 convention used by TCU/LSU refactors):

1. **Setup** — when `req_in` valid + worker idle, latch `DxaReq` +
   read `DescTable[desc_slot]` into `setup_params`. Emit `pipeline_start`
   for one tick.
2. **Addr-gen** — produce one `(cl_addr, smem_byte_addr, byte_offset,
   valid_length, oob, last)` tuple per cycle into a small FIFO (matches
   `VX_dxa_addr_gen`). Driven by tile/stride iteration over the descriptor.
3. **Gmem-req** — pop addr-gen FIFO; allocate a tag slot (sized by
   `DXA_MAX_INFLIGHT`); send `MemReq` on `DxaCore::gmem_req_out` (the arb
   handles N:GMEM_OUT_PORTS). Forward the per-tag bookkeeping (smem_addr,
   byte_offset, last) into a per-tag side-FIFO consumed by smem_wr.
4. **Rsp-buf** — match incoming `MemRsp` to its tag, attach the
   `mem_block_t` payload to the per-tag entry, mark arrived.
5. **Smem-wr** — at each cycle, take one ready entry in tag order;
   build the LMEM `MemReq`:
     - `addr` = SMEM byte addr aligned to LMEM word
     - `data` = `mem_block_t` carrying the GMEM response bytes (or
       `cfill` pattern when oob)
     - `byteen` = computed from `byte_offset` / `valid_length`
     - `flags` = `{notify_smem_done && is_last, bar_addr}`
     - `tag.value` = `core_id` (so `LocalMem` / `MemCrossBar` carries
       it back to the right barrier consumer)
   Multicast: replay the same payload across each CTA's
   `smem_addr + cta * smem_stride`, with the flag set only on the
   last replay of the last block (per CTA, with `bar_addr + cta_warp_idx`).

### 3.4 SMEM write semantics — flag-carried barrier release

RTL routes the completion flag into `VX_dxa_completion`, which observes
it at the LMEM-side and pulses the barrier event. Mirror in SimX:
extend `LocalMem::Impl::tick` (or a tiny shim ahead of it) to inspect
`bank_req.flags` on the cycle the bank request fires. If the
`notify_smem_done` bit is set, call
`core->barrier_event_release(flags.bar_addr)`. Routing is by
`bank_req.tag.value == core_id`; the multi-cluster wiring already
guarantees one `LocalMem` per core, so the resolution is unambiguous.

This eliminates the synchronous `barrier_event_release` in
`DxaCore::tick` — the release happens *after* the LMEM bank arbiter has
accepted the last write, i.e. coupled to LMEM bank-conflict timing exactly
as in RTL. **Rule 2 satisfied: data + barrier-release flag travel on the
same packet.**

### 3.5 Cluster wiring

```cpp
// in Cluster::Impl ctor (cluster.cpp), under EXT_DXA_ENABLE:

// 1. Per-core DxaUnit submits to per-core DxaReq channel.
//    DxaUnit lives inside SfuUnit (constructed when EXT_DXA_ENABLE).
// 2. DxaCore exposes: dxa_req_in[NUM_CORES_PER_CLUSTER]
//    Bind each core's DxaUnit.req_out → dxa_core->dxa_req_in[cid].
for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
  for (uint32_t c = 0; c < cores_per_socket_; ++c) {
    uint32_t cid = s * cores_per_socket_ + c;
    auto& sfu = sockets_.at(s)->core(c)->sfu_unit();
    sfu->dxa_unit()->req_out.bind(&dxa_core_->dxa_req_in.at(cid));
  }
}

// 3. DxaCore.lmem_req_out[cid] → LocalMem.Inputs[port_dxa] on each core.
//    port_dxa is appended after LSU + TCU ports; LSU ctor must size
//    LocalMem accordingly (see §5).
for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
  for (uint32_t c = 0; c < cores_per_socket_; ++c) {
    uint32_t cid = s * cores_per_socket_ + c;
    uint32_t port = LSU_NUM_REQS + (EXT_TCU_ENABLE ? 1 : 0);
    dxa_core_->lmem_req_out.at(cid).bind(
        &sockets_.at(s)->core(c)->local_mem()->Inputs.at(port));
    // No rsp binding — DXA writes are response-less (matches
    // LocalMem::Config{write_response=false}).
  }
}

// 4. GMEM arb path unchanged (existing 2:1 priority arb wiring stays).
```

### 3.6 SfuUnit changes

`sfu_unit.cpp` no longer owns DXA's functional copy. The DXA branch
becomes:

```cpp
if (std::get_if<DxaType>(&trace->op_type)) {
  // Decode lanes 0..3 into a DxaReq.
  DxaReq req = build_dxa_req_from_lanes(trace, core_);
  if (!core_->dxa_unit()->try_submit(req)) {
    continue; // backpressure on the per-core DxaReq channel
  }
  core_->barrier_event_attach(req.bar_id);
}
// fall through to the latency / output.send / pop path; no execute_copy.
```

`dxa_pending_` and the `execute_copy` bookkeeping go away. The per-core
`DxaUnit` enforces the result-port elastic buffer that the RTL
`VX_dxa_unit` provides (rsp_buf), so the SfuUnit's own latency line
remains correct.

---

## 4. Why this is correct (against the three rules)

- **Rule 1.** Every byte that lands in LMEM came from a `MemRsp` payload,
  carried inside a `mem_block_t`. No `core->mem_read`, no
  `core->mem_write`. `gmem_read` and `execute_copy` are deleted.
- **Rule 2.** The same packet (`MemReq` to `LocalMem.Inputs`) carries the
  data, the byte enables, and the barrier-release flag. The flag fires
  only when LMEM bank arbitration accepts the bank request — which is
  when timing says it should. There is no second source of truth.
- **Rule 3.** Mirrors the RTL module split exactly (`DxaUnit` per SFU,
  `DxaCore` per cluster, 5-stage worker), so debug traces and perf
  comparisons line up. C++-only optimizations (e.g. dedup of repeated
  cache-line reads inside `gmem_req`, sized by an in-engine line buffer
  larger than the RTL response buffer) remain available.

---

## 5. Phased implementation

Each phase compiles, runs, and is independently reviewable. Validate via
`build_test32/` (per `feedback_build_dir`) and the SimX-vs-RTLsim CSV
trace diff (per `reference_csv_trace_debugging`).

### Phase 0 — Repair the build

- Delete the dead `dxa_req_in` reference in
  [cluster.cpp:108-109](../../sim/simx/cluster.cpp#L108).
- Restore a working baseline: either revert to whatever was last green on
  `simx_v3` for DXA, or temporarily wire `DxaCore::lmem_req_out` to a
  null sink. Goal: `EXT_DXA_ENABLE` builds and runs at least one
  smoke kernel under SimX. No behavior change required yet.

### Phase 1 — Introduce DxaUnit (per-SFU)

- Add `sim/simx/dxa/dxa_unit.{h,cpp}`. `DxaUnit` is a SimObject owned by
  each `SfuUnit`. It exposes `req_out` (`SimChannel<DxaReq>`) and a
  small `result_in` for the SFU result-port latency.
- Move lane decode out of `SfuUnit::on_tick` into `DxaUnit::issue()`.
- `DxaCore` exposes `dxa_req_in[NUM_CORES_PER_CLUSTER]`. Cluster wires
  per-core DxaUnit.req_out → DxaCore.dxa_req_in.
- For this phase only, DxaCore can keep the existing functional engine
  (`execute_copy`/SmemReq replay) — the goal is the structural split.
  Tests must still pass.

### Phase 2 — TLM LMEM write path

- Bump `LocalMem` per-core port count by 1 (under `EXT_DXA_ENABLE`).
- Redefine `DxaCore::lmem_req_out` as
  `std::vector<SimChannel<MemReq>>` of length `NUM_CORES_PER_CLUSTER`.
- Replace `SmemReq` with real `MemReq` packets carrying
  `mem_block_t` payloads. Data is supplied from the GMEM `MemRsp` (no
  more `execute_copy` synchronous writes).
- Add the `flags` decode in `LocalMem::Impl::tick`: when a fired bank
  request has `notify_smem_done`, call `core->barrier_event_release`.
  Drop the direct release from `DxaCore::tick`.
- The `cfill` / OOB path: the gmem_req stage marks OOB tags; rsp_buf
  fills them with the `cfill` pattern locally (no GMEM request is
  issued — RTL does the same via `oob_arrived_*`).

### Phase 3 — Refactor Worker into RTL 5-stage shape

- Replace the flat `tick_slice` with `Setup/AddrGen/GmemReq/RspBuf/SmemWr`
  sub-objects ticked in reverse order (matches v3 convention).
- Delete the precomputed `gmem_lines` / `smem_blocks` vectors: the
  AddrGen produces them on demand. The dedup for "consecutive same-CL"
  becomes a single-entry filter inside `gmem_req`.
- Plumb `DXA_MAX_INFLIGHT` from the RTL knob (replace `kMaxOutstanding`).

### Phase 4 — Multicast and bar-stride

- Multicast replay belongs in `smem_wr` (not in `gmem_req` — GMEM is
  read once per CL regardless of CTA count). Each replay uses
  `smem_base + cta * smem_stride`; the completion flag sets
  `bar_addr + cta_warp_idx` per CTA's last write.
- Verify against an RTL trace where `cta_mask` selects a non-trivial
  subset.

### Phase 5 — Validation and perf parity

- Run the WGMMA kernel matrix that depends on DXA loads (see
  `project_wgmma_test_surface`).
- Diff SimX vs RTLsim CSV traces on a kernel that mixes DXA writes and
  WGMMA reads to the same SMEM region. The first-divergence cycle should
  be within RTL's per-bank arbitration tolerance.

---

## 6. Out of scope

- DXA → DXA write-after-write ordering across slices (RTL behavior is
  inherited from the LMEM bank arbiter; no new SimX work needed).
- L2 access pattern changes — the existing 2:1 priority arb between
  socket and DXA stays.
- DCR programming sequence — descriptor writes via DCR remain
  synchronous.
- Performance counters — the existing `PerfStats` shape is preserved;
  only the events that increment them move into the new pipeline.

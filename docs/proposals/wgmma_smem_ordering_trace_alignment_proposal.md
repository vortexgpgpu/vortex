# SimX↔RTL Precision Trace Alignment for the WGMMA SMEM-Ordering Bug

**Date:** 2026-05-30
**Status:** Draft — design + plan only; no code changes yet.
**Owners:** TCU / DXA / sim team
**Related:**
  - [wgmma_kmajor_completion_proposal.md](wgmma_kmajor_completion_proposal.md) (the in-flight K-major work whose timing *exposed* this bug, but did not cause it),
  - [VX_local_mem.sv](../../hw/rtl/mem/VX_local_mem.sv) (shared LMEM; prime suspect),
  - [VX_tcu_abuf.sv](../../hw/rtl/tcu/VX_tcu_abuf.sv) / [VX_tcu_bbuf.sv](../../hw/rtl/tcu/VX_tcu_bbuf.sv) / [VX_dxa_smem_wr.sv](../../hw/rtl/dxa/VX_dxa_smem_wr.sv) (UUID drop points),
  - [sim/simx/barrier_unit.cpp](../../sim/simx/barrier_unit.cpp) / [sim/simx/dxa/dxa_core.cpp](../../sim/simx/dxa/dxa_core.cpp) / [sim/simx/tcu/tcu_unit.cpp](../../sim/simx/tcu/tcu_unit.cpp) (SimX models to instrument).

---

## 1. Summary

A WGMMA matmul produces wrong results (~2% median, ~10% max per-element error) on **rtlsim** while **SimX passes**, at one config corner: `NT=32, fp16, WGMMA_NRC=8` (and only there — NRC=16/32 pass). The failure is **producer-agnostic**: it reproduces with DXA entirely removed (all-cooperative SMEM load) and with either operand loaded cooperatively. It is therefore **not** a DXA-data bug, **not** the DXA completion seam, and **not** the K-major fetch addressing. It is a device-side **SMEM-write → WGMMA-read ordering hazard** that only the cycle-accurate model exhibits.

We do **not** yet understand the exact mechanism with certainty. We have a strong, code-validated leading hypothesis (the shared-LMEM read-during-write interlock does not cover the TCU/DXA read port; §4), but pass/fail bisection cannot *prove* a cycle-level ordering edge. This document proposes the durable way to settle it: a **transaction-accurate SimX TLM** acting as an oracle plus **UUID-keyed precision tracing** in both SimX and RTL, aligned by `(uuid, phase)` so the first divergence names the exact instruction, address, and ordering edge. The instrumentation outlives this bug — it is reusable for every future TCU/DXA/barrier ordering question.

---

## 2. Investigation status (what is proven)

All runs: `NT=32, NUM_WARPS=8, ISSUE_WIDTH=4, XLEN=64`, default `M=N=K=64`, on the self-hosted runner.

| # | Test | A source | B source | Barrier | NRC | SimX | rtlsim |
|---|------|----------|----------|---------|-----|------|--------|
| 1 | `sgemm_tcu_wg_dxa` full | DXA | DXA | txn-bar | 32 | PASS | PASS |
| 2 | `sgemm_tcu_wg_dxa` full | DXA | DXA | txn-bar | 16 | PASS | PASS |
| 3 | `sgemm_tcu_wg_dxa` full | DXA | DXA | txn-bar | **8** | PASS | **FAIL** |
| 4 | `sgemm_tcu_wg_dxa` `SW_LOAD_A` | coop | DXA | txn-bar | 8 | — | FAIL |
| 5 | `sgemm_tcu_wg_dxa` `SW_LOAD_B` | DXA | coop | txn-bar | 8 | — | FAIL |
| 6 | `sgemm_tcu_wg_dxa` all-coop | coop | coop | txn-bar | 8 | **PASS** | **FAIL** |
| 7 | `sgemm_tcu_wg_dxa` all-coop | coop | coop | txn-bar | 16 | — | PASS |
| 8 | `sgemm_tcu_wg` (rmajor-A + kmajor-B) | coop | coop | `__syncthreads` | 8 | — | **PASS** |
| 9 | `sgemm_tcu_wg` block-major | coop | coop | `__syncthreads` | 8 | — | PASS |

Conclusions forced by the table:
- **3 vs 6/7:** removing DXA entirely still fails at NRC=8 and passes at NRC=16 ⇒ not DXA, not K-major data; it is NRC=8-specific and consumer-side.
- **6 (rtlsim FAIL) vs 6 (SimX PASS):** same host reference + data, deterministic functional model passes ⇒ not a host/reference/tolerance bug; it is cycle-timing-dependent device behaviour.
- **6 vs 8:** *identical* device-side layout/ops (row-major A + K-major B, NRC=8) — `sgemm_tcu_wg` passes, `sgemm_tcu_wg_dxa` fails. The two emit the *same* `vx_barrier(...)` call (`bar(0).arrive_and_wait()` ≡ `__syncthreads()`); the only difference is **code layout / instruction timing** ⇒ a latent timing hazard exposed by schedule, not a logic/addressing bug.
- Error distributions across rows 3/4/5/6 are statistically identical (median ~2%, max ~10%, ≥100 elements) ⇒ one common root cause.

**Bounded region:** a deterministic (in rtlsim) write→read ordering hazard between an SMEM producer (LSU cooperative store *or* DXA write) and the WGMMA consumer's SMEM read, exposed when the post-barrier read lands too few cycles after the write commits (least slack at NRC=8).

---

## 3. Why bisection cannot finish the job

Pass/fail bisection got us to a *category*, not a *mechanism*. At least three distinct RTL mechanisms fit every datum above, and each implies a fix in a different unit:

1. **Shared-LMEM RDW interlock gap** — the read-during-write hazard logic protects only the LSU↔LSU path; the TCU/DXA read port bypasses it (§4). *Fix in `VX_local_mem`.*
2. **Barrier releases before write visibility** — `lsu_sched_drained` waits on LSU scheduler drain, which may not guarantee the just-written word is bank-visible to the TCU read port. *Fix in the barrier/drain.*
3. **abuf/bbuf internal pipeline hazard** — residency/`req_inflight` releasing `bbuf_ready` a cycle early so the FEDP reads a not-yet-written storage slot. *Fix in the TCU buffers.*

Only a cycle-resolved, UUID-aligned trace discriminates these. That is the work this document scopes.

---

## 4. Leading hypothesis (to confirm, not assume)

`VX_local_mem` ([VX_local_mem.sv:209-282](../../hw/rtl/mem/VX_local_mem.sv#L209)) instantiates each bank as `VX_sp_ram #(.OUT_REG(1), .RDW_MODE("R"))` — registered output, **read-first** (a same-cycle read-after-write returns the *old* word). A one-cycle read-during-write interlock guards this:

```sv
last_wr_valid <= dma_wr_b || (lsu_active && per_bank_req_rw[i]);   // tracks LSU and DMA writes
wire is_rdw_hazard = last_wr_valid && ~per_bank_req_rw[i]
                  && (per_bank_req_addr[i] == last_wr_addr);
assign bank_rsp_valid       = per_bank_req_valid[i] && ~dma_active && ~per_bank_req_rw[i] && ~is_rdw_hazard;
assign per_bank_req_ready[i] = ~dma_active && (bank_rsp_ready || per_bank_req_rw[i]) && ~is_rdw_hazard;
```

The interlock only ever stalls the **LSU** response path (`~dma_active`, `per_bank_req_addr[i]`). The DMA-class read port — used by **both** the TCU operand fetch (`tcu_lmem_if`) and DXA — **bypasses it** (the source comment at line 259 says so outright: *"DMA reads bypass this check"*). So a TCU read issued the cycle after a producer write to the same bank/word can read pre-write data through the read-first, output-registered SRAM. The `last_wr_valid` window is a single cycle, which is exactly why the hazard only bites at NRC=8 (tightest schedule) and vanishes at NRC≥16, and why SimX — which models the write functionally and immediately — never sees it.

This is consistent with **every** datum in §2. It remains a hypothesis until the aligned trace shows, for a specific wrong-data read: (a) the read hit address X on the DMA port, (b) a producer write committed to X on the prior cycle, (c) same bank, (d) the interlock was bypassed. Items to confirm during the work: that `tcu_lmem_if` is routed to the `dma_bus_if` port of `VX_local_mem` (suspected, not yet traced end-to-end).

---

## 5. Methodology: TLM oracle + UUID alignment

Principle: **SimX is a correct oracle, instruction-for-instruction**, because it passes on the exact stream that rtlsim fails. If both models emit a transaction lifecycle event stream where every event is tagged with the originating instruction **UUID** and a **logical phase**, we align the two streams by `(uuid, phase)` — *not* by cycle (cycles legitimately differ) — and the first payload divergence (or the first reordered `release`/`commit` pair) is the bug, named exactly.

Two requirements make this work, and both are currently unmet:

1. **A transaction-accurate SimX TLM.** SimX must model each relevant transaction as a first-class object that *carries its UUID through its whole life* — not a functional shortcut that computes the right value and discards provenance/ordering. Today the DXA path is modeled as explicit work objects (good), but the **barrier** is counter-only and the **TCU operand read** is a synchronous functional cache lookup (§6) — neither can be aligned against RTL transactions as-is.
2. **Precise UUID threading on both sides.** The moment a UUID is dropped at any hop, alignment breaks. Today every SMEM transaction we care about drops it (§6).

---

## 6. Alignment review: transaction models & UUID-propagation map

UUID is `uint64_t` in SimX (`instr_trace_t::uuid`) and `UUID_WIDTH=44` in RTL (`tag_t.uuid` on `VX_lsu_mem_if` / `VX_mem_bus_if`). Validated drop points and model gaps:

### 6.1 Barrier path
| Hop | SimX | RTL |
|-----|------|-----|
| issue | `instr_trace_t.uuid` present | `execute_if.data.header.uuid` present |
| expect_tx | `event_attach(bar_id, count)` — **no uuid** ([barrier_unit.cpp:130](../../sim/simx/barrier_unit.cpp#L130)) | `is_tx_expect` in `warp_ctl_if` — **no uuid** |
| arrive / drain | `barrier.events`/`count` counters — **no uuid**; LSU drain coarse | `lsu_sched_drained` boolean ([VX_core.sv](../../hw/rtl/core/VX_core.sv)) — **no uuid** |
| release | `global_resume()` by warp-id — **no uuid** | bar unit release — **no uuid** |
> Barrier is a **counter machine, not a TLM** on both sides. Needs an explicit per-arrival/per-release event tagged with the triggering UUID (DXA/LSU uuid on tx, bar phase on release).

### 6.2 DXA producer path
| Hop | SimX | RTL |
|-----|------|-----|
| issue→worker | `DxaReq.uuid` present ([dxa_core.cpp](../../sim/simx/dxa/dxa_core.cpp)) | descriptor/worker context |
| SMEM write | `MemReq.uuid = w.req.uuid` present ([dxa_core.cpp:512](../../sim/simx/dxa/dxa_core.cpp#L512)) | `smem_bus_if.req_data.tag.uuid = '0` **DROPPED** ([VX_dxa_smem_wr.sv:509](../../hw/rtl/dxa/VX_dxa_smem_wr.sv#L509)) |
| write trace | `SIMX_DXA_WR` print — **omits uuid**, logs at write ([dxa_core.cpp:568](../../sim/simx/dxa/dxa_core.cpp#L568)) | `RTL_DXA_WR` print — **no uuid**, logged at **request-fire not commit** |
| bar notify | `barrier_event_release(bar_id)` — **no uuid** ([cluster.cpp](../../sim/simx/cluster.cpp)) | `smem_wr_attr_last`/`attr` carry bar_id only — **no uuid** |

### 6.3 WGMMA consumer path
| Hop | SimX | RTL |
|-----|------|-----|
| issue | `instr_trace_t.uuid` present | WGMMA uop uuid present at dispatch |
| operand read | `plan_wgmma_lines()` / `load_lmem_word()` — **functional cache lookup, no uuid, no explicit MemReq** ([tcu_unit.cpp](../../sim/simx/tcu/tcu_unit.cpp)) | `tcu_lmem_if.req_data.tag = '0` **DROPPED** ([abuf:258](../../hw/rtl/tcu/VX_tcu_abuf.sv#L258), [bbuf:384](../../hw/rtl/tcu/VX_tcu_bbuf.sv#L384)) |
| FEDP / commit | functional | accumulator writeback |
> The TCU SMEM read is **not a modeled transaction** in SimX and is **UUID-zeroed** in RTL — the single most important gap, since the suspected stale read lives here.

### 6.4 LSU producer path (reference)
LSU *does* thread uuid through `lsu_mem_if.req_data.tag.uuid` in RTL and retains `pending_req_t::trace` in SimX — so the cooperative-store side is already partly traceable; we mainly need the matching write-commit event.

---

## 7. Unified trace schema

One line per event, machine-parseable, identical field names in SimX and RTL. Align on `(uuid, phase, addr)`; never on cycle.

```
TRACE,<model>,<uuid>,<phase>,<unit>,cyc=<n>,wid=<n>,bar=<id>,addr=0x..,bank=<n>,
      byteen=0x..,data=0x..,attr=<k=v;...>
```

- `model` ∈ {`SIMX`,`RTL`}; `cyc` is informational only (models differ).
- `phase` is the **logical lifecycle stage**, the alignment key — enumerated identically on both sides:
  `ISSUE, EXPECT_TX, BAR_ARRIVE, BAR_TX_DONE, BAR_RELEASE, SMEM_WR_REQ, SMEM_WR_COMMIT, ABUF_RD, BBUF_RD, FEDP_IN, WGMMA_COMMIT`.
- **`SMEM_WR_COMMIT` must be a true commit event** (data observable to a subsequent reader), distinct from `SMEM_WR_REQ` (request-fire) — the request/commit split is exactly what the bug turns on.

The aligner needs, per candidate failing read: the `*_RD` event (data), the most-recent `SMEM_WR_COMMIT` to the same `(bank,addr)`, and the `BAR_RELEASE` for the consuming warp — to test ordering.

---

## 8. Instrumentation plan — SimX events and their RTL match

Each SimX emit point has a one-to-one RTL counterpart so logs line up. Gated behind a build flag (`DBG_TRACE_TXN`) on both sides.

| phase | SimX emit site | RTL emit site |
|-------|----------------|---------------|
| ISSUE | issue stage (`uuid`) | dispatch (`header.uuid`) |
| EXPECT_TX | `BarrierUnit::event_attach` (add uuid arg) | `VX_wctl_unit` tx-expect (thread uuid) |
| BAR_ARRIVE | `BarrierUnit::arrive` | bar unit arrive |
| BAR_TX_DONE | `barrier_event_release` (pass `req.uuid`) | DXA `smem_wr_attr_last` (carry uuid in attr) |
| BAR_RELEASE | `global_resume` / phase bump | bar unit release |
| SMEM_WR_REQ | LSU store issue + DXA `tick_worker_smem_wr` | `VX_local_mem` write accept (LSU & DMA ports) |
| **SMEM_WR_COMMIT** | model the SRAM commit cycle explicitly (see §9) | `VX_local_mem` post-`OUT_REG` write-visible point |
| ABUF_RD / BBUF_RD | `load_lmem_word` (thread WGMMA uuid in) | `tcu_lmem_if` req/rsp (thread uuid into `tag`) |
| WGMMA_COMMIT | accumulator writeback | FEDP/accumulator writeback |

### UUID plumbing required
- **RTL:** replace the three `tag(.uuid) = '0` drops with the real uuid — `VX_tcu_abuf.sv:258`, `VX_tcu_bbuf.sv:384` (thread the WGMMA uop uuid through `VX_tcu_uops` into the fetch request), `VX_dxa_smem_wr.sv:509` (carry the descriptor's issuing uuid). Add uuid to the DXA bar-notify `attr`.
- **SimX:** add a `uuid` parameter to `event_attach`/`event_release` and the barrier state; pass the WGMMA `uuid` into `plan_wgmma_lines`/`load_lmem_word`; add `uuid` to `SIMX_DXA_WR`.

---

## 9. SimX TLM strengthening (the oracle)

Counter-and-functional models cannot be aligned against RTL transactions. Minimum upgrades so SimX emits a faithful, ordered event stream:

1. **Barrier → event records.** Augment `BarrierUnit` so each `event_attach`/`event_release`/`arrive`/`release` appends a `(uuid, bar_id, phase)` record, without changing the functional counter semantics. (Non-invasive: logging alongside existing counters.)
2. **TCU read → explicit reads.** Make `plan_wgmma_lines`/`load_lmem_word` emit an `ABUF_RD`/`BBUF_RD` event per operand word with the consuming WGMMA uuid and the source LMEM address, even though the data path stays a functional lookup. This gives one SimX read event per RTL `tcu_lmem_if` beat to align against.
3. **SMEM write-commit event.** Emit `SMEM_WR_COMMIT` at the point SimX makes the write visible to subsequent reads (functionally immediate), so the *ordering relation* `WR_COMMIT(X) → RD(X)` is explicit in the oracle and can be compared to RTL's relation for the same `(uuid,addr)`.

> Note: SimX is the oracle for **values and logical ordering**, not cycle counts. We are validating that RTL preserves the same `WR_COMMIT(X) → RD(X)` *order* SimX guarantees; the bug is precisely RTL violating it.

---

## 10. The aligner

A small offline script (`ci/trace_align.py`, no RTL/sim coupling):

1. Parse both logs into event lists; key each by `(uuid, phase, addr)`.
2. **Value diff:** for every `(uuid, ABUF_RD|BBUF_RD, addr)` present in both, compare `data`. First mismatch ⇒ the failing read (instruction + address).
3. **Ordering check:** for that read's `addr`, find the last `SMEM_WR_COMMIT(addr)` and the consuming warp's `BAR_RELEASE`. Report the RTL vs SimX relative order of `{WR_COMMIT(addr), BAR_RELEASE, RD(addr)}`. A read that precedes its producer's commit in RTL but not SimX is the smoking gun and localizes the unit (LMEM bank vs barrier vs abuf/bbuf).
4. Emit a focused report: the one UUID, the address, the bank, the three events with cycles on each side.

Scope the run to one CTA / one K-tile at NRC=8 to keep logs small and the divergence singular.

---

## 11. Implementation phases

| Phase | Scope | Files | Validation |
|------|-------|-------|------------|
| 0 | This draft | `docs/proposals/...` | review |
| 1 | RTL UUID un-drop on the 3 SMEM transactions + DXA attr | `VX_tcu_abuf/bbuf.sv`, `VX_dxa_smem_wr.sv`, `VX_tcu_uops.sv` | uuid non-zero in waveforms |
| 2 | RTL `DBG_TRACE_TXN` emits for all §7 phases incl. true `SMEM_WR_COMMIT` in `VX_local_mem` | `VX_local_mem.sv`, bar unit, tcu, dxa | one RTL log for the failing run |
| 3 | SimX UUID plumbing + TLM strengthening (§9) + matching emits | `barrier_unit.*`, `tcu_unit.cpp`, `dxa_core.cpp`, `cluster.cpp` | one SimX log for the same run |
| 4 | `ci/trace_align.py` + run on NRC=8 one-CTA | `ci/trace_align.py` | divergence report names the unit |
| 5 | Implement the fix the trace points to (likely `VX_local_mem` RDW DMA-port interlock; confirm) + re-run §2 table | per finding | rows 3–6 PASS, 1/2/7–9 still PASS |
| 6 | Keep `DBG_TRACE_TXN` + aligner as standing infra | — | documented |

Phases 1–4 are the diagnosis; 5 is the fix; 6 banks the investment.

---

## 12. Risks & limitations

- **Alignment key drift.** If phase enumeration differs subtly between models, events won't pair. Mitigation: define `phase` enum in one shared header consulted by both descriptions; keep the per-read granularity identical (one event per operand word).
- **Trace volume.** Full-run logs are large. Mitigation: scope to one CTA/one K-tile/NRC=8; filter by uuid range.
- **Heisenbug risk is low.** rtlsim is deterministic; `DBG_TRACE_TXN` is observational ($write / printf), so enabling it does not change timing.
- **TLM strengthening must not change SimX functional results.** All §9 changes are additive logging beside existing counters/lookups; gate behind the flag and re-run the passing SimX baseline to confirm no behavior change.
- **The leading hypothesis may be wrong.** That is the point — the aligner adjudicates among the three §3 mechanisms rather than presuming the §4 one. The fix in Phase 5 follows the trace, not this draft.

---

## 13. Expected outcome

A single aligned report of the form: *"UUID U, BBUF_RD addr X bank B: SIMX data=D0, RTL data=D1; last SMEM_WR_COMMIT(X)=UUID W at RTL cyc C-1; BAR_RELEASE(warp) at RTL cyc C-2; RTL issued RD(X) at cyc C with the LMEM RDW interlock inactive on the DMA port."* — i.e., the exact instruction, address, bank, and ordering edge. From there the fix is unambiguous and verifiable against the §2 table, and the tracing infrastructure remains for the next ordering bug.

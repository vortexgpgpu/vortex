# Tensor Core Unit (TCU / WGMMA) — Design

**Scope:** the Vortex Tensor Core Unit — the matrix-multiply-accumulate
engine implementing WMMA / WGMMA (NVIDIA-style warp / warpgroup MMA),
2:4 structured sparsity, and tensor metadata loads. Covers the RTL
([`hw/rtl/tcu/`](../../hw/rtl/tcu/)), the SimX functional/timing model
([`sim/simx/tcu/`](../../sim/simx/tcu/)), and the SW surface
([`sw/kernel/include/vx_tensor.h`](../../sw/kernel/include/vx_tensor.h),
[`sw/common/tensor_cfg.h`](../../sw/common/tensor_cfg.h)).

The TCU is a RISC-V ISA extension (`MISA` bit 9,
[`VX_config.toml:305`](../../VX_config.toml#L305)). It is gated by
`VX_CFG_EXT_TCU_ENABLE` and configured by a `[tcu]` block in
[`VX_config.toml:229-245`](../../VX_config.toml#L229).

---

## 1. Architecture overview

```
  VX_dispatch_if[ISSUE_WIDTH]                       VX_tcu_unit
  ──────────────────────────►  ┌──────────────────────────────────────────────┐
                               │  VX_lane_dispatch → per_block_execute_if[Q]    │
                               │                                                │
                               │   op_type split:                               │
                               │   ┌── TCU_LD ──► VX_tcu_agu ──► VX_lsu_sched   │
                               │   │              (warp AGU)     (shared client) │
                               │   │                  │                          │
                               │   │                  ▼  meta_wr broadcast       │
                               │   │            VX_tcu_sp_meta (per-warp SRAM)      │
                               │   │                                            │
                               │   └── WMMA/WGMMA ──► VX_tcu_wgmma (orchestrator)│
                               │           │            │                        │
                               │           │       VX_tcu_lockstep (CTA gate)    │
                               │           │            │                        │
                               │           │       VX_tcu_tbuf                   │
                               │           │       (Q×abuf + 1×bbuf + mem_arb)   │
                               │           ▼            │                        │
                               │     Q × VX_tcu_core ◄──┘  operands              │
                               │       (FEDP: dpi|bhf|dsp|tfr) + VX_tcu_sp_mux   │
                               │           │                                     │
                               │     VX_lane_gather ──► commit_if                │
                               └──────────────────────────────────────────────┘
```

A macro MMA op is dispatched across `ISSUE_WIDTH` issue slots; the TCU
treats those `Q = ISSUE_WIDTH` slots as one **warpgroup** of `BLOCK_SIZE`
lock-stepped blocks. [`VX_tcu_uops`](../../hw/rtl/tcu/VX_tcu_uops.sv)
expands the macro into per-block micro-ops; each block executes a FEDP
(fused element dot product) datapath in [`VX_tcu_core`](../../hw/rtl/tcu/VX_tcu_core.sv).

---

## 2. Data types, opcodes, and configuration

**Formats** ([`VX_tcu_pkg.sv:27-39`](../../hw/rtl/tcu/VX_tcu_pkg.sv#L27),
[`sw/common/tensor_cfg.h:25-66`](../../sw/common/tensor_cfg.h#L25)):
fp32, fp16, bf16, fp8 (e4m3), bf8 (e5m2), tf32, and integer i32/i8/u8/i4/u4.
Per-format enables: `VX_CFG_TCU_TF32_ENABLE`, `_BF16_`, `_FP8_`, `_INT_`
([`VX_config.toml:238-241`](../../VX_config.toml#L238)).

**Opcodes** (4-bit `op_type`,
[`VX_gpu_pkg.sv:595-607`](../../hw/rtl/VX_gpu_pkg.sv#L595)):

| Opcode | Value | Meaning |
|---|---|---|
| `INST_TCU_WMMA` | 0 | Warp-level MMA |
| `INST_TCU_WGMMA` | 1 | Warpgroup MMA |
| `INST_TCU_WMMA_SP` | 3 | Sparse warp MMA |
| `INST_TCU_WGMMA_SP` | 4 | Sparse warpgroup MMA |
| `INST_TCU_LD` | 5 | Tensor metadata load (AGU path) |

Sparsity is now a distinct opcode (`*_SP`) rather than a runtime flag.
`INST_TCU_LD` bypasses the register file and hazards through the independent
`XREG_0` (SP) and `XREG_1` (MX) scoreboard bits
([`VX_gpu_pkg.sv:78-80,606`](../../hw/rtl/VX_gpu_pkg.sv#L78)).

**Key config** ([`VX_config.toml:229-245`](../../VX_config.toml#L229)):
`VX_CFG_TCU_TYPE` selects the FEDP backend (`DPI`/`DSP`/`BHF`/`TFR`;
default `TFR` for ASIC, `DSP` for synthesis, `DPI` when DPI is enabled);
`VX_CFG_NUM_TCU_LANES = NUM_THREADS`; `VX_CFG_NUM_TCU_BLOCKS = ISSUE_WIDTH`;
`VX_CFG_TCU_SPARSE_ENABLE`; `VX_CFG_TCU_WGMMA_ENABLE`.

---

## 3. RTL module inventory

`hw/rtl/tcu/`:

| Module | Role |
|---|---|
| [`VX_tcu_pkg.sv`](../../hw/rtl/tcu/VX_tcu_pkg.sv) | Format IDs, tile/block/step geometry, WGMMA tile dims, `tcu_tbuf_req_t` ([`:364-383`](../../hw/rtl/tcu/VX_tcu_pkg.sv#L364)), FP helpers, trace tasks. |
| [`VX_tcu_unit.sv`](../../hw/rtl/tcu/VX_tcu_unit.sv) | Thin top wrapper: lane dispatch/gather, splits TCU_LD (→AGU) from MMA (→core) by `op_type`, OR-muxes results with AGU priority. |
| [`VX_tcu_wgmma.sv`](../../hw/rtl/tcu/VX_tcu_wgmma.sv) | WGMMA orchestrator: owns the lockstep gate, builds `tcu_tbuf_req_t` (masked by `cta_conflict`), instantiates `VX_tcu_tbuf`, WGMMA perf counters. |
| [`VX_tcu_lockstep.sv`](../../hw/rtl/tcu/VX_tcu_lockstep.sv) | Single-owner CTA gate: `tcu_owned_r`/`tcu_owner_r` + per-block `in_expansion_r`; combinational `cta_conflict`. |
| [`VX_tcu_tbuf.sv`](../../hw/rtl/tcu/VX_tcu_tbuf.sv) | Tile-buffer subsystem: `BLOCK_SIZE × VX_tcu_abuf` + 1 shared `VX_tcu_bbuf` + `VX_mem_arb` (Q+1 → 1 LMEM port). |
| [`VX_tcu_abuf.sv`](../../hw/rtl/tcu/VX_tcu_abuf.sv) | Per-block A buffer (k-stripe storage); both block-major and k-major fetch paths selected by descriptor stride; refetches A on every WGMMA first-uop. |
| [`VX_tcu_bbuf.sv`](../../hw/rtl/tcu/VX_tcu_bbuf.sv) | TB-shared B buffer; block-major and k-major dense + sparse paths. |
| [`VX_tcu_core.sv`](../../hw/rtl/tcu/VX_tcu_core.sv) | Per-block FEDP datapath; selects the FEDP backend by `VX_CFG_TCU_TYPE`; sparse gather via `VX_tcu_sp_mux`. |
| [`VX_tcu_uops.sv`](../../hw/rtl/tcu/VX_tcu_uops.sv) | Macro→micro-op sequencer; WGMMA iteration order k-outer/n-middle/m-inner; tags `fu_lock/fu_unlock`, `is_first_uop/is_last_uop`. |
| [`VX_tcu_agu.sv`](../../hw/rtl/tcu/VX_tcu_agu.sv) | Warp-level AGU for `INST_TCU_LD`; FSM IDLE→ISSUE→WAIT→COMMIT; one fetch per TCU_LD into a `VX_tcu_sp_meta` slot via the shared LSU scheduler. |
| [`VX_tcu_sp_meta.sv`](../../hw/rtl/tcu/VX_tcu_sp_meta.sv) | Per-warp sparse-metadata SRAM written by AGU TCU_LD; combinational read. |
| [`VX_tcu_mx_meta.sv`](../../hw/rtl/tcu/VX_tcu_mx_meta.sv) | Independent per-warp MX A/B scale SRAM written by AGU TCU_LD. |
| [`VX_tcu_sp_mux.sv`](../../hw/rtl/tcu/VX_tcu_sp_mux.sv) | 2:4 structured-sparsity B-column gather; I_RATIO ∈ {2,4,8}. |

**FEDP backends** (`hw/rtl/tcu/{dpi,dsp,bhf,tfr}/`) — all four compute
`Σ(a·b)+c → d`, selected by `VX_CFG_TCU_TYPE`:

- **dpi** ([`dpi/VX_tcu_fedp_dpi.sv`](../../hw/rtl/tcu/dpi/VX_tcu_fedp_dpi.sv)) —
  simulation-only; calls SystemVerilog DPI-C softfloat (`dpi_f2f`/
  `dpi_fmul`/`dpi_fadd`). Latency 4.
- **bhf** ([`bhf/`](../../hw/rtl/tcu/bhf/)) — Berkeley HardFloat IEEE FMA
  tree; synthesizable.
- **dsp** ([`dsp/VX_tcu_fedp_dsp.sv`](../../hw/rtl/tcu/dsp/VX_tcu_fedp_dsp.sv)) —
  DSP-mapped path with explicit fp16→fp32 converters; highest latency.
- **tfr** ([`tfr/`](../../hw/rtl/tcu/tfr/)) — default for ASIC/SimX;
  fixed-point reduction tree (align → mul → accumulate → norm/round)
  with max-exp extraction, lane mask, classifier, exception reduce.

---

## 4. Execution model (as-built)

**Issue / dispatch.** A WMMA/WGMMA macro enters via
`VX_dispatch_if[ISSUE_WIDTH]` → `VX_lane_dispatch` →
`per_block_execute_if[BLOCK_SIZE]`. `VX_tcu_uops` expands it into `Q`
lock-stepped micro-ops in **k-outer / n-middle / m-inner** order
(`ctr = k*(n_steps*m_steps) + n*m_steps + m`,
[`VX_tcu_uops.sv:113`](../../hw/rtl/tcu/VX_tcu_uops.sv#L113)), tagging each
with `fu_lock/fu_unlock` and `is_first_uop/is_last_uop`.

**Operand load.** In register mode the A operand comes from registers
(abuf bypassed). In shared-memory mode `VX_tcu_wgmma` builds a
`tcu_tbuf_req_t` (validity masked by `cta_conflict`) and drives
`VX_tcu_tbuf`: each block's `VX_tcu_abuf` fetches the active k-stripe's
A-rows, while the shared `VX_tcu_bbuf` fetches one B bank-row per `(k,n)`
and broadcasts it to all `Q` cores. Both buffers support **block-major**
(descriptor stride 0) and **k-major / row-major** (stride ≠ 0) layouts,
selected at first-uop from `desc[31:16]`
([`VX_tcu_abuf.sv:186-255`](../../hw/rtl/tcu/VX_tcu_abuf.sv#L186),
[`VX_tcu_bbuf.sv:286-601`](../../hw/rtl/tcu/VX_tcu_bbuf.sv#L286)). The
`Q+1` LMEM masters arbitrate to one port through `VX_mem_arb`.

**Lock-step / warpgroup gate.** `VX_tcu_lockstep` enforces single-CTA
occupancy of the shared B buffer: a single owner latch plus per-block
`in_expansion_r` (set on the first sub-uop, cleared on the last). A block
presenting a different CTA's WGMMA is deferred via `cta_conflict[b]` until
the resident warpgroup drains. SimX mirrors this gate exactly
([`tcu_unit.cpp:450-532`](../../sim/simx/tcu/tcu_unit.cpp#L450)).

**Compute / accumulate / writeback.** Each `VX_tcu_core` runs the
configured FEDP cell. Sparse 2:4 routes B through `VX_tcu_sp_mux` using the
`vld_block` metadata read from `VX_tcu_sp_meta` (preloaded by TCU_LD).
Accumulation walks k through the C accumulator register; results return via
`VX_lane_gather` → `commit_if`.

**TCU_LD path.** `INST_TCU_LD` is handled by `VX_tcu_agu`, which walks the
metadata stride and issues to the **shared** `VX_lsu_scheduler`
([`hw/rtl/core/VX_lsu_scheduler.sv`](../../hw/rtl/core/VX_lsu_scheduler.sv),
hoisted to `VX_core` as a multi-client resource: LSU = client 0, TCU =
client 1, with RTX/TEX/OM ports reserved). Responses are written into
`VX_tcu_sp_meta`; the op commits with `wr_xregs[XREG_0]=1` to release the
scoreboard slot, and a following `wgmma_sp` stalls on that bit.

---

## 5. SimX model

[`sim/simx/tcu/tcu_unit.{cpp,h}`](../../sim/simx/tcu/tcu_unit.cpp)
implements `TcuUnit` (FuncUnit) + `TcuUopGen`: wmma/wgmma/meta_store/tcu_ld,
the lock-step probe, `plan_wgmma_lines`, and a `sparse_meta_` SRAM mirror.
Operand load goes through channels (`load_lmem_word`); there is **no**
`core_->mem_read` in the TCU path.
[`tcu_tbuf.{cpp,h}`](../../sim/simx/tcu/tcu_tbuf.cpp) provides `TcuTbuf`
(abuf×Q + bbuf×1 line caches over one `SimChannel` LMEM port) with
`plan_a/plan_b`, `ready_a/ready_b`, `read_a/read_b`. The RTL and SimX share
the same k-outer iteration order, per-block-A + shared-B structure, and
lock-step deadlock contract, enabling SimX↔RTL cycle parity work.

Perf CSRs: `TBUF_STALLS`, `TBUF_CACHE_HITS`, `LMEM_READS`
([`VX_types.toml:570-577`](../../VX_types.toml#L570)).

---

## 6. SW surface

[`sw/kernel/include/vx_tensor.h`](../../sw/kernel/include/vx_tensor.h):
`vx_make_smem_desc(ptr, leading_bytes)`
([`:35`](../../sw/kernel/include/vx_tensor.h#L35)), `wmma_context` /
`wgmma_context`, `load_sp_metadata` (now TCU_LD-based,
[`:213-218`](../../sw/kernel/include/vx_tensor.h#L213)), and block-major
SMEM index helpers `a_blockmajor_idx` / `b_blockmajor_idx`
([`:719-748`](../../sw/kernel/include/vx_tensor.h#L719)).
[`sw/common/tensor_cfg.h`](../../sw/common/tensor_cfg.h) holds the format
structs and tile-geometry templates; `sw/runtime/include/tensor_sp.h` and
`sw/runtime/include/tensor_mx.h` hold host-side helpers.

---

## 7. Proposed but not yet implemented

The following were specified across the source proposals and remain open;
they are recorded so the intent is preserved.

1. **SimX↔RTL precision trace-alignment infrastructure**
   (`wgmma_smem_ordering_trace_alignment_proposal`, Phases 2–6 — the
   least-realized proposal). Only the Phase-1 UUID un-drops landed
   ([`VX_tcu_abuf.sv:259`](../../hw/rtl/tcu/VX_tcu_abuf.sv#L259),
   [`VX_tcu_bbuf.sv:385`](../../hw/rtl/tcu/VX_tcu_bbuf.sv#L385)). Still
   unbuilt: the reusable `DBG_TRACE_TXN` phase emitters, a true
   `SMEM_WR_COMMIT` event in `VX_local_mem`, SimX barrier/TCU-read TLM
   strengthening, and the `ci/trace_align.py` aligner. The proposal frames
   this as standing infrastructure for *all* future TCU/DXA/barrier
   ordering bugs — high preservation value. The underlying `VX_local_mem`
   DMA-port read-during-write interlock is the prime suspect for the
   NRC=8 shared-memory ordering hazard.
2. **WGMMA latency-hiding refinements** (`wgmma_simx_v3_proposal` §4.10):
   k-transition prefetch and B ping-pong buffering — deferred as optional.
3. **DXA / k-major cleanup** (`wgmma_kmajor_completion_proposal` Phases
   6–7): retiring the legacy block-major B-buffer path (both paths
   currently coexist, ≈2× bbuf area) and the U55C PPA quantification on
   Yosys/OpenSTA — deliberately deferred; block-major is retained as the
   transition default.
4. **TCU_LD generalization** (`tcu_ld_proposal` open items): a
   multi-request stride AGU, double-buffered `VX_tcu_sp_meta` slots, and
   RTX/TEX/OM warp-level preload clients on the shared `VX_lsu_scheduler`
   (the multi-client boundary exists; ports 2..N are reserved but unused).
5. **MN-major SS descriptor + SMEM swizzling**
   (`wgmma_kmajor_completion_proposal`, out of scope): NVIDIA's third
   descriptor leg and bank-conflict swizzle — bit space reserved, no
   implementation.

**Open bugs flagged in proposals (status unverified):** XLEN=64
`sgemm_tcu_wg` fp16 rtlsim numerical failure (`wgmma_opt1` PW1) and the
tf32 rtlsim poisoned cycle-counter (`tcu_ld` PW2).

**Superseded directions** (recorded to avoid revival): the
`wgmma_simx_v3_addendum` "drop row-major end-to-end, no fallback" stance
(reversed — k-major is now a first-class retained path); the `simx_v3`
Phase-E "third buffered sparse-streaming path" and the `VX_tcu_mbuf`
metadata buffer (replaced by TCU_LD into `VX_tcu_sp_meta`; `VX_tcu_mbuf.sv`
deleted); and the proposed SimX file/class splits (`TcuTbufA`+`TcuSharedB`,
`tcu_wgmma.cpp`) which were flagged cosmetic and never landed — the
*structure* matches, the *decomposition* differs.

---

## 8. Source proposals

This design consolidates and supersedes the following proposals (now
removed from `docs/proposals/`): `wgmma_simx_v3_proposal.md`,
`wgmma_simx_v3_addendum.md`, `wgmma_opt1_proposal.md`,
`wgmma_kmajor_completion_proposal.md`,
`wgmma_smem_ordering_trace_alignment_proposal.md`, `tcu_ld_proposal.md`.

The k-major DXA writer that pairs with the TCU's k-major buffers is
described in the DXA design (`dxa_async_copy_multicast.md`).

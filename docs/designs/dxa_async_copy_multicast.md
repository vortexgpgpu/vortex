# DXA — Direct eXecution Accelerator (Async Copy + Multicast) — Design

**Scope:** the Vortex DXA engine — an NVIDIA-Hopper-TMA-style
asynchronous bulk-copy and multicast unit. Covers the RTL
([`hw/rtl/dxa/`](../../hw/rtl/dxa/)), the SimX model
([`sim/simx/dxa/`](../../sim/simx/dxa/)), and the SW surface
([`sw/kernel/include/vx_dxa.h`](../../sw/kernel/include/vx_dxa.h),
[`sw/runtime/include/dxa.h`](../../sw/runtime/include/dxa.h)).

DXA is a RISC-V ISA extension (`MISA` bit 10,
[`VX_config.toml:305`](../../VX_config.toml#L305)), gated by
`VX_CFG_EXT_DXA_ENABLE`.

---

## 1. What DXA is

A warp issues a single instruction that launches an asynchronous,
multi-cycle, tiled **global→shared (GMEM→LMEM) copy** described by a
pre-programmed descriptor. The issuing warp is freed immediately; a
barrier "transaction" (tx) is released to the LMEM-side completion logic
when the copy lands. Operations:

- **Async tiled copy** GMEM→LMEM, ranks 1–5, with out-of-bounds clamp and
  constant fill (`cfill`).
- **Multicast** (`cta_mask` with >1 bit): read GMEM once, replay the LMEM
  writes to multiple co-resident CTAs
  (`dest[r] = issuer_smem + r·smem_stride`), releasing each receiver's
  barrier.
- **K-major transpose** (`dest_kmajor`, rank ≤ 2): scatter one element per
  beat, producing the K-major SMEM layout the tensor core (WGMMA)
  consumes directly — the primary DXA↔TCU tie-in (see `tensor_core_wgmma_engine.md`).

There is no software→global path; DXA is GMEM-read / LMEM-write only.

---

## 2. ISA, descriptor, and configuration

- **Opcode** `INST_SFU_DXA = 4'h9`
  ([`VX_gpu_pkg.sv:514`](../../hw/rtl/VX_gpu_pkg.sv#L514)), encoded as
  RISC-V `custom0` / `funct7=0x3`, decoded at
  [`VX_decode.sv:694-698`](../../hw/rtl/core/VX_decode.sv#L694). The issue
  intrinsic packs 4 lanes via `vx_wgather` (lane0 = smem addr, lane1 =
  `meta = (bar<<4)|desc_slot`, coords, lane3 = `cta_mask`).
- **Descriptor** `dxa_desc_t`
  ([`VX_dxa_pkg.sv:54`](../../hw/rtl/dxa/VX_dxa_pkg.sv#L54)): `base_addr` +
  `meta` + packed tile sizes (`tile01/tile23/tile4`) + `cfill` +
  `size0..4` (OOB bounds) + `stride0..3` (byte row strides) +
  `smem_stride` (per-CTA LMEM stride for multicast). Programmed by the
  host over a DCR block `[dcr_dxa]` at 0x100, 16 descriptors
  ([`VX_types.toml:87-123`](../../VX_types.toml#L87)).
- **META subfields** `[dxa_desc_meta]`: `DIM` (3b), `ELEMSZ` (2b),
  `LAYOUT` (1b @5, row-major vs K-major). `SWIZZLE`/`INTERLEAVE`/`L2PROMO`
  fields are allocated but unwired
  ([`VX_types.toml:278-291`](../../VX_types.toml#L278)) — see §7.
- **Config** ([`VX_config.toml:40-58`](../../VX_config.toml#L40)):
  `VX_CFG_NUM_DXA_UNITS = max(1, ceil(NUM_CORES/8))`,
  `VX_CFG_DXA_QUEUE_SIZE = 16`, `VX_CFG_DXA_MAX_INFLIGHT = 8`,
  `VX_DCR_DXA_DESC_COUNT = 16`.
- **Perf CSRs** `[csr_mpm_dxa]` 0xB03–0xB07 (transfers, gmem_reads,
  gmem_dedup, lmem_writes, gmem_latency),
  `VX_DCR_MPM_CLASS_DXA = 6`
  ([`VX_types.toml:557-568`](../../VX_types.toml#L557)).
- **Cluster dispatch** (multicast prerequisite):
  `VX_CSR_CTA_CLUSTER_SIZE = 0xCE0`,
  `VX_DCR_KMU_CLUSTER_DIM_{X,Y,Z}`
  ([`VX_types.toml:80-82,386`](../../VX_types.toml#L80)).

---

## 3. RTL module inventory

DXA is **cluster-shared**: one `VX_dxa_core` per cluster
([`VX_cluster.sv:241`](../../hw/rtl/VX_cluster.sv#L241)) fans a single
queue out to `NUM_DXA_UNITS` workers.

| Module | Role |
|---|---|
| [`VX_dxa_pkg.sv`](../../hw/rtl/dxa/VX_dxa_pkg.sv) | Types/params: `dxa_req_data_t`, `dxa_launch_t`, `dxa_desc_t`, `dxa_setup_params_t` (incl. `dest_kmajor`, `per_lane_stride_bytes`, `elem_bytes`). |
| [`VX_dxa_core.sv`](../../hw/rtl/dxa/VX_dxa_core.sv) | Cluster engine top: `req_arb` (N:1) → `req_queue` (16) → desc-table read → `dispatch` (1:N) → workers → `gmem_arb` + `lmem_arb`. |
| [`VX_dxa_unit.sv`](../../hw/rtl/dxa/VX_dxa_unit.sv) | Per-core SFU front end: decodes the 4-lane wgather, strips `VX_MEM_LMEM_BASE_ADDR`, builds `dxa_req_data_t`, frees the warp via SFU result. `expect_tx` is software (no HW txbar attach). |
| [`VX_dxa_dispatch.sv`](../../hw/rtl/dxa/VX_dxa_dispatch.sv) | `VX_stream_dispatch` routing req+desc to the first idle worker. |
| [`VX_dxa_desc_table.sv`](../../hw/rtl/dxa/VX_dxa_desc_table.sv) | DCR-written descriptor RAM, single read port. |
| [`VX_dxa_worker.sv`](../../hw/rtl/dxa/VX_dxa_worker.sv) | Structural wiring of the active pipeline + watchdog. |
| [`VX_dxa_setup.sv`](../../hw/rtl/dxa/VX_dxa_setup.sv) | Transfer-lifecycle FSM + decoupled setup engine (overlaps setup with prior drain); 3 DSP multipliers compute rolling-cursor deltas, OOB limits, K-major params. |
| [`VX_dxa_addr_gen.sv`](../../hw/rtl/dxa/VX_dxa_addr_gen.sv) | Single rolling-cursor `gmem_cursor_r` + 4-level ripple odometer; one beat/cycle. |
| [`VX_dxa_gmem_req.sv`](../../hw/rtl/dxa/VX_dxa_gmem_req.sv) | Issues GMEM reads; `slot_table_r[MAX_OUTSTANDING]` per-tag credit model (≤8 inflight); direct-drains responses to smem_wr. |
| [`VX_dxa_smem_wr.sv`](../../hw/rtl/dxa/VX_dxa_smem_wr.sv) | Barrel-shift drain → 1 LMEM word/beat; K-major per-element scatter; serial multicast replay; completion flag on last write. |
| [`VX_dxa_completion.sv`](../../hw/rtl/dxa/VX_dxa_completion.sv) | Instantiated **per core** in [`VX_mem_unit.sv:183`](../../hw/rtl/core/VX_mem_unit.sv#L183) (LMEM side): snoops bank writes, reads the completion attr, pushes `bar_addr` into a depth-`NUM_WARPS` FIFO → `txbar_bus_if`. |
| [`VX_dxa_watchdog.sv`](../../hw/rtl/dxa/VX_dxa_watchdog.sv) | `RUNTIME_ASSERT` on `STALL_TIMEOUT` cycles of no progress. |

---

## 4. Architecture as-built (end-to-end)

1. **Issue.** Kernel `vx_dxa_issue_*` → `vx_wgather` packs 4 lanes →
   `.insn r` (custom0/funct7=3) → `VX_decode.sv:698` → `INST_SFU_DXA`.
2. **Per-core front end.** `VX_sfu_unit` → `VX_dxa_unit`: decodes lanes,
   computes LMEM-relative `smem_addr`, emits `dxa_req_data_t`, frees the
   warp. `expect_tx` is software (`vx_barrier.h`).
3. **Cluster engine.** `VX_dxa_core`: `req_arb` (N cores:1) → `req_queue`
   (16) → desc-table read by `meta[desc_slot]` → `dispatch` to the first
   idle worker.
4. **Setup.** `VX_dxa_setup` latches metadata, decodes rank/elem/LAYOUT,
   computes the initial GMEM base, row length, rolling-cursor deltas, OOB
   limits, and (LAYOUT=1, rank ≤ 2) `dest_kmajor` + `per_lane_stride_bytes`.
   Setup overlaps the previous transfer's drain (staged → active).
5. **Address gen.** `VX_dxa_addr_gen` emits one
   `(cl_addr, smem_byte_addr, byte_offset, valid_length, oob, last)` per
   cycle from the rolling cursor + ripple odometer.
6. **GMEM read.** `VX_dxa_gmem_req` allocates a `slot_table_r` tag
   (≤ `DXA_MAX_INFLIGHT=8`) and issues the read (OOB lines synthesize
   immediate arrival, no bus traffic), direct-draining the response.
7. **SMEM write.** `VX_dxa_smem_wr` barrel-shift-aligns and drains one
   LMEM word/beat; the last cache line is deferred so it drains last
   carrying the completion flag. K-major mode scatters one element/beat at
   `+per_lane_stride_bytes`.
8. **Multicast fan-out.** If `popcount(cta_mask) > 1`: a priority-encoder
   walk replays each receiver's writes at `+r·smem_stride` — **serial**,
   `popcount(mask)` beats/word — and the last write per receiver sets
   `attr = {notify, bar_addr + (idx<<NB_BITS)}`.
9. **LMEM + completion.** Writes go through the `VX_mem_unit` LMEM-DMA
   priority arbiter (DXA = idx 0, TCU = idx 1). `VX_dxa_completion` snoops
   the bank fire, reads the attr, and releases barriers via `txbar_bus_if`;
   software `arrive_and_wait` unblocks the consumer.

**K-major / TCU tie-in.** `dest_kmajor` produces the transposed SMEM
layout `smem[i0·tile1 + i1]` that the WGMMA B-tile path consumes directly
from LMEM (the DXA and TCU share the LMEM-DMA port). K-major drains ~8×
slower (1 element/beat) but is amortized over many WGMMA micro-ops.

---

## 5. SimX model

[`sim/simx/dxa/dxa_unit.{h,cpp}`](../../sim/simx/dxa/dxa_unit.cpp) is a
plain `DxaUnit` helper owned by `SfuUnit`; it lane-decodes a `DxaType`
trace into a `DxaReq` channel with no functional read/write.
[`dxa_core.{h,cpp}`](../../sim/simx/dxa/dxa_core.cpp) is the `DxaCore`
SimObject: NoC-only (GMEM via `MemReq`/`MemRsp`, LMEM via real `MemReq`
with `mem_block_t` payload and `flags.dxa_notify_done`/`dxa_notify_bar_id`),
with K-major and serial multicast replay modeled. The SimX worker is a
**flat two-step** engine over a pre-enumerated `work_list`
([`enumerate_work_list`, `dxa_core.cpp:332`](../../sim/simx/dxa/dxa_core.cpp#L332)),
not the RTL's per-stage objects — it matches RTL behavior and the three
SimX design rules, but not module-by-module structure (see §7).

---

## 6. SW surface

[`sw/kernel/include/vx_dxa.h`](../../sw/kernel/include/vx_dxa.h): issue
intrinsics `vx_dxa_issue_{1..5}d_wg` and `..._multicast_wg`, plus the C++
`vortex::dxa_multicast_{1..5}d` helper classes
([`:321-468`](../../sw/kernel/include/vx_dxa.h#L321)) that bundle
`expect_tx` + mask + `group_barrier` rendezvous and issue from
`get_cluster_rank()==0`.
[`sw/runtime/include/dxa.h`](../../sw/runtime/include/dxa.h): host
descriptor programming `vx_dxa_program_desc_{1..5}d`,
`vx_dxa_program_desc_multicast`, and
`vx_dxa_program_desc_set_layout` (ROW_MAJOR / K_MAJOR,
[`:242-261`](../../sw/runtime/include/dxa.h#L242)).

---

## 7. Proposed but not yet implemented

1. **Per-socket DXA relocation** (`dxa_multicast_proposal` §5 / Phase 2 —
   the single biggest unbuilt item). Move `VX_dxa_core` into
   `VX_socket.sv` with a `dxa_dcache_arb` (dcache > DXA priority) and
   delete the cluster-scoped DXA fabric, with a matching `Socket`-owned
   `DxaCore` in SimX. Motivated by measured GMEM latency growth with core
   count. DXA is currently cluster-scoped in both RTL and SimX. The
   planned `VX_cta_table_if.sv` per-slot LMEM-base table was abandoned in
   favor of the shipped "Path A" stride-arithmetic + cluster-contiguous
   LMEM placement ([`VX_mem_unit.sv:144-152`](../../hw/rtl/core/VX_mem_unit.sv#L144)).
2. **Multicast LMEM-arbiter hoist** (`dxa_worker_rtl_redesign` Phase 5):
   collapse M-way multicast from `popcount(mask)` beats/word to one via a
   bank-broadcast / side-band `cta_mask` + `smem_stride` in the LMEM
   arbiter, removing `replay_remaining_r` and the per-beat priority
   encoder. Multicast is serial today in both RTL and SimX.
3. **SimX 5-stage worker objectification** (`dxa_simx_v3` Phase 3):
   replace the flat `work_list` engine with Setup/AddrGen/GmemReq/RspBuf/
   SmemWr sub-objects for true RTL module-correspondence and on-demand
   address generation.
4. **Dead-code cleanup**: remove the uninstantiated
   [`VX_dxa_rsp_buf.sv`](../../hw/rtl/dxa/VX_dxa_rsp_buf.sv), the dead pkg
   types (`dxa_issue_dec_t`, `dxa_worker_cmd_t`, `dxa_completion_info_t`,
   `dxa_smem_done_t`), and correct the stale
   [`VX_dxa_worker.sv:16`](../../hw/rtl/dxa/VX_dxa_worker.sv#L16) header
   that still names 5 submodules including rsp_buf.
5. **Reserved META extension fields** `SWIZZLE`/`INTERLEAVE`/`L2PROMO`
   ([`VX_types.toml:282-291`](../../VX_types.toml#L282)) — allocated but
   wired nowhere; placeholders for SMEM swizzling / L2 prefetch promotion.
6. **Cross-issuer multicast** (`dxa_multicast_from(rank)`) and the
   Hopper-style cross-core DSMEM path (multicast Open Questions), plus a
   zero-arg `dxa_multicast` overload now feasible against the
   `get_cluster_size()` CSR.

**Superseded directions** (recorded to avoid revival): the original
5-stage `setup→addr_gen→gmem_req→rsp_buf→smem_wr` worker with separate
allocator/inflight-FIFO (replaced by the 4-active-stage direct-drain
pipeline with `slot_table` credit model); the `VX_cta_table_if.sv`
per-slot LMEM-base table (replaced by Path-A stride arithmetic); and the
`_mw`/`_mc` test naming (actual tests use `_mcast`, plus `dxa_kmajor_check`).

---

## 8. Source proposals

This design consolidates and supersedes the following proposals (now
removed from `docs/proposals/`): `dxa_multicast_proposal.md`,
`dxa_simx_v3_proposal.md`, `dxa_worker_rtl_redesign_proposal.md`.

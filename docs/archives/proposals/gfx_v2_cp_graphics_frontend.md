# gfx_v2 — CP Graphics Front-End (autonomous draw orchestration)

**Scope:** how a draw becomes a **self-contained device command sequence** the
Command Processor executes end-to-end — VS → setup → binning → RASTER/FS/OM —
with the host idle between submit and present. Covers the command encoding, the
inter-stage barrier mechanism, the static-command-list / dynamic-data
resolution, cache coherence between stages, and the small RASTER config change.
Builds on the CP ([command_processor_control_plane.md](../designs/command_processor_control_plane.md)).
**Reference:** NVIDIA pushbuffer / GPU front-end; DX12 `ExecuteIndirect` /
Vulkan indirect (compute-driven command data).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.4.
**Date:** 2026-06-07.
**Related:** [command_processor_control_plane.md](../designs/command_processor_control_plane.md),
[gfx_v2_vertex_setup_pipeline.md](gfx_v2_vertex_setup_pipeline.md),
[gfx_v2_tile_binning_redesign.md](gfx_v2_tile_binning_redesign.md),
[gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md).

---

## 1. Motivation

Charter pillar 3: the CP is the autonomous front-end. Today the host worker
thread pushes one command (or one launch) into the ring and **busy-polls
`Q_SEQNUM` before the next** ([device.cpp:334-398](../../sw/runtime/common/device.cpp#L334)),
so every draw stage round-trips through the CPU. The "true GPU" model is the
NVIDIA pushbuffer one: the host driver builds the *entire* draw as a command
sequence **once**, rings the doorbell **once**, and the CP runs VS → setup →
binning → RASTER/FS/OM to completion, signaling only at the end. The host CPU
is idle the whole time; the ring lives in host memory but is fetched passively
by the CP (exactly NVIDIA's pushbuffer-in-system-memory model).

---

## 2. The key mechanism: launch-drain serialization = free inter-stage barrier

`CMD_LAUNCH` pulses KMU start and **holds the grant until `busy` deasserts**
([VX_cp_launch.sv](../../hw/rtl/cp/VX_cp_launch.sv) `IDLE→PULSE_START→WAIT_BUSY→WAIT_DRAIN`),
and the per-queue engine retires commands **in order**
([VX_cp_engine.sv](../../hw/rtl/cp/VX_cp_engine.sv)). So a sequence of launches
in one queue executes sequentially, each fully drained before the next begins.

That drain **is** the device-wide barrier the sort-middle pipeline needs
between stages (setup-done before prefix-sum, sort-done before header-scan,
binning-done before RASTER). We get it from the existing CP with **no grid-wide
barrier hardware** — which is exactly why CP-sequencing beats a persistent
megakernel for orchestration (charter §8 fork 4).

---

## 3. A draw as a CP command sequence

One draw = one contiguous run of ring commands (built host-side pre-submit):

| # | Command(s) | Effect |
|---|---|---|
| 0 | `DCR_WRITE`×k + `LAUNCH` | **VS** → resident transformed-vertex buffer |
| 1 | `CACHE_FLUSH` / `FENCE(GPU)` | make VS output visible to setup |
| 2 | `DCR_WRITE`×k + `LAUNCH` | **setup+count** (assembly+clip+setup) → `primbuf`, `count[]` |
| 3 | `LAUNCH` | **prefix-sum** → offsets, totals into the draw-context |
| 4 | `LAUNCH` | **emit** → composite-key array |
| 5 | `LAUNCH`×r | **radix sort** (r digit passes) |
| 6 | `LAUNCH` | **header-scan** → `bin_headers[]`, `sorted_pids[]`, `tile_count` |
| 7 | `CACHE_FLUSH` | make binned buffers visible to RASTER |
| 8 | `DCR_WRITE`×k | program **RASTER/OM/TEX** config (pool addresses + state) |
| 9 | `LAUNCH` | **FS/raster pass**: `vx_rast_begin`→poll-loop→`vx_tex`→`vx_om` |
| 10 | (per render-pass) `EVENT_SIGNAL` | retire → host learns the *whole draw* is done |

Cache-maintenance commands (`CMD_CACHE_FLUSH`, today already appended after
launches, [queue.cpp:382-424](../../sw/runtime/common/queue.cpp#L382)) sit
between producer/consumer stages so each kernel sees the prior's writes; a real
`CMD_FENCE(GPU)` (CP §10 item 3) would be the lighter-weight replacement once
implemented.

---

## 4. Static command list vs. dynamic data — the one real problem

The host builds the command list **before** submit, but binning sizes
(`V_sub` subtris, `P` coverage entries, `tile_count` bins) are computed
**on-device**. The host cannot bake those into commands. Resolution, in three
parts — all keeping the command list fully static:

**4.1 Reserved resident tiling pool ⇒ all addresses are static.** Reserve the
front-end working set (VS output, `primbuf`, key array, `sorted_pids`,
`bin_headers`, draw-context) at **fixed addresses** in the pinned region
([virtual_memory_subsystem.md](../designs/virtual_memory_subsystem.md)), sized
to a configured high-water mark. Every DCR/kernel-arg address the host bakes is
a pool base — never data-dependent. This is the real-GPU tiling-buffer-pool /
TBDR parameter-buffer model; overflow → the on-device software fallback
(charter §6.5) or a pool-flush (§7).

**4.2 Dynamic counts live in a resident draw-context ⇒ kernels read them.** A
small resident `draw_context` block holds `V_sub`, `P`, prefix-sum offsets,
`tile_count`. Each kernel reads the counts it needs from `draw_context` (the
sort reads `P` written by prefix-sum; header-scan reads `P`; etc.). No
dynamic value ever passes through the host-built command.

**4.3 Fixed device-filling launches + grid-stride ⇒ launch dims are static.**
KMU grid/block come from host-baked DCRs, so they cannot depend on `P`. Launch
each data-parallel stage with a **fixed device-filling grid** and have the
kernel **grid-stride** over the dynamic count read from `draw_context`
(`for (i = tid; i < P; i += stride)`). Launch dims static; iteration bounded by
the in-memory count.

Net: **addresses static (pool), counts dynamic (draw-context), launch dims
static (grid-stride)** — the entire draw is a static command sequence the host
emits once, with all data-dependent behavior driven by in-memory counts. Zero
host mid-draw.

---

## 5. RASTER config from resident memory (small front-end change)

The only RASTER input the host can't precompute is `tile_count` (non-empty bin
count). Two options; (a) preferred:

- **(a) Memory-resident value.** RASTER reads `tile_count` from the resident
  `bin_headers`/draw-context on `vx_rast_begin` instead of the
  `VX_DCR_RASTER_TILE_COUNT` DCR — the same "fetch from memory" pattern
  [VX_raster_mem.sv](../../hw/rtl/raster/VX_raster_mem.sv) already uses for tile
  headers. The header-scan kernel writes it; `TBUF_ADDR`/`PBUF_ADDR`/
  `PBUF_STRIDE`/`SCISSOR` stay static DCRs (pool addresses + draw state).
- **(b) CP indirect DCR write.** Add a `CMD_DCR_WRITE` variant that sources its
  value from a device address (header-scan writes a scratch word; CP programs
  `RASTER_TILE_COUNT` from it). Keeps RASTER as-is; more CP work.

Option (a) keeps the dynamic value entirely device-side and is a minimal RASTER
edit (it already reads the tile array from memory).

---

## 6. CP roadmap dependencies (and what this motivates)

- **QMD-style atomic `CMD_LAUNCH`** (CP §10 item 5) is the important one: today
  a launch is ~18 `CMD_DCR_WRITE`s to KMU DCRs + `LAUNCH`. A draw has ~7 launch
  stages ⇒ ~126 DCR writes of ring traffic per draw. The atomic launch
  (grid/block/PC/args inline) collapses each stage to one command and is the
  precondition for compact graphics command sequences. **gfx_v2 is a strong
  motivator for prioritizing it.**
- **Real `CMD_FENCE(GPU)`** (CP §10 item 3) replaces the heavyweight
  `CMD_CACHE_FLUSH` between stages with scoped ordering.
- **Multi-queue** (CP §10 item 6) lets independent draws/passes overlap; the
  baseline uses one queue with in-order draws.

---

## 7. Launch overhead & mitigations

CP-sequencing trades a few launch/drain/flush overheads per draw for simplicity
and determinism. The binning reductions (prefix-sum, header-scan) are short
kernels, so for tiny draws the per-launch overhead can dominate. Mitigations,
in priority order:

1. **QMD atomic launch** (§6) — removes the ~18-DCR-write tax per stage.
2. **Stage fusion** where a barrier isn't required — e.g. emit folded into the
   setup pass's tail, or header-scan into the last sort pass.
3. **Persistent-megakernel binning** as an alternative for the reduction tail
   (one launch, internal device-wide sync via global atomics) — reintroduces
   the cross-CTA sync problem the CP avoids, so only if launch overhead proves
   dominant. Baseline stays CP-sequenced.

---

## 8. Multi-draw / a frame on the device

A render pass = many draws' command sub-sequences concatenated in the ring,
executed in order by the CP (per-queue serialization). Color/depth attachments
**accumulate in resident memory** across draws (draw N+1 depth-tests against
N); the tiling pool (§4.1) is reset between draws. The host builds the whole
pass once and submits; the CP runs every draw autonomously; the only egress is
the final framebuffer at present (charter pillar 4). That is the complete
"frame on the device, host untouched" picture.

---

## 9. Runtime / driver changes

- **Batch the draw.** Replace the per-command submit+poll
  ([queue.cpp](../../sw/runtime/common/queue.cpp)) with a path that encodes the
  full §3 sequence into the ring and rings the doorbell once; poll `Q_SEQNUM`
  only at draw/pass end. The graphics command-sequence builder lives in
  `sw/runtime/graphics.cpp` (host driver helper) and is what vortexpipe's
  `vp_raster_draw` calls instead of the host `Binning()` + per-stage launches.
- **Pool + draw-context allocation** from the pinned region, once per pass.
- **vortexpipe** emits the command sequence; no host `Binning()`, no readback.

---

## 10. Validation & phasing

1. **Emulation CP first** ([sim/common/cmd_processor.cpp](../../sim/common/cmd_processor.cpp)):
   model the batched graphics command sequence; validate the full
   VS→binning→RASTER draw on simx against the host-`Binning()` reference image
   (`tests/graphics/gfx_*`), with zero host intervention between submit and the
   final read.
2. **RASTER config-from-memory** (§5a) in SimX, then RTL (U55C timing).
3. **QMD atomic launch** (CP §10 item 5) to compact the sequence.
4. **FPGA** (XRT) end-to-end once the RTL CP path carries it.

---

## 11. Open items

- **Pool sizing & overflow policy** — high-water reservation vs. a device-side
  count pass that drives a conditional fallback (charter §6.5); ties to the
  `VX_CAPS_VM_PINNED_*` query.
- **Indirect launch grid** — if grid-stride is insufficient for some stage, a
  device-computed grid (CP reads launch dims from `draw_context`) is the
  fallback; not needed for the baseline.
- **Per-draw cache maintenance cost** — measure; motivates real `CMD_FENCE`.
- **Frame-level scheduling / multi-queue overlap** — deferred (CP §10 item 6).

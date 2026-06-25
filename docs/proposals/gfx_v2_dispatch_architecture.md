# gfx_v2 — Graphics dispatch architecture (SW↔HW interface) + FWD-3 v2 plan

Authoritative design for how graphics work reaches the SMs. **Supersedes the
FWD-3 sections of `gfx_v2_fwd_simx_impl.md`** (the direct-`cta_dispatch`-injection
approach is abandoned — see "Why the pivot"). FWD-1/FWD-2 (SimX) in that doc still
stand.

## The SW↔HW interface today (what a draw actually is)

A draw is a **host-recorded command buffer**, not a poll hack at the command level:
- `DrawCommands` (graphics.h) accumulates `launch()` + `dcr_write()` entries and
  submits them as **one CP ring batch** via `vx_enqueue_commands` (one doorbell,
  one completion event).
- `PipelinePool::append()` emits the sort-middle front end's **nine stage launches**
  per draw (setup → binning → raster → …) into that batch, with inter-stage
  barriers. Each stage is a `CMD_LAUNCH`; FF units (raster/tex/om) are configured by
  `CMD_DCR_WRITE` and triggered by their stage's kernel.

The CP command set is compute-shaped: `NOP, MEM_{WRITE,READ,COPY}, DCR_{WRITE,READ},
LAUNCH, FENCE, EVENT_{SIGNAL,WAIT}, CACHE_FLUSH`. **No `DRAW`.** The runtime mirrors
it (`vx_enqueue_{launch,copy,read,write,dcr_*,barrier,signal,wait_value,…}`). The CP
interacts with the rasterizer only via DCR (config) + the launched kernel (trigger);
it has no graphics datapath.

## The gap is two separable layers

- **Layer A — work distribution (HW).** Raster covered-quads → SM fragment waves.
  Today: the `vx_rast` pull/poll/sentinel + bcoord-CSR side-band + cross-core
  `VX_raster_arb` (and the `cores≥2` `raster_mem` crash). **This is the real missing
  HW** and is *interface-agnostic* — needed identically whether the trigger is
  `LAUNCH` or a future `DRAW`. **= FWD-3.**
- **Layer B — command abstraction (SW↔HW).** Who expands draw→pipeline: host records
  nine launches (today) vs. the device expands a `DRAW` packet (true GPU). About
  *recipe ownership*, not how fragments reach SMs. **= a later CP-orchestration step.**

Conflating them is the trap. Layer A is the priority; Layer B builds on it.

## Why the pivot (abandoning direct cta_dispatch injection)

The earlier FWD-3 plan side-injected fragment warps into each core's
`VX_cta_dispatch` (lmem override + a 2nd dispatch source). Rejected: it makes the
**generic compute warp dispatcher graphics-aware**, violating separation of concerns.
`VX_cta_dispatch` must stay a single-source (kmu-bus) consumer.

Options weighed (see also the SW↔HW analysis above):
- **A: extend the device `VX_kmu`** to emit fragment CTAs — *wrong*: the KMU is
  device-level, upstream of the per-cluster raster producers; it would need a
  full-hierarchy round-trip + per-core targeting + per-core LMEM locality it doesn't
  have. Not how real GPUs work (the grid engine doesn't do per-quad distribution).
- **B: a per-core graphics work distributor sharing the kmu bus** — *chosen*. Matches
  real GPU topology (the raster engine's own work distributor feeds SMs through the
  standard warp-launch fabric). `cta_dispatch` untouched.

## Chosen design — Layer A: `VX_raster_kmu` (option B) + warp-self-pull

**Topology.** A per-core `VX_raster_kmu` is the single-owner consumer of the core's
`raster_bus` quads. It launches **bare 1-warp fragment CTAs** onto the core's *local*
`kmu_bus` via a reused 2-input `VX_kmu_arb` (merging the socket-fed device-KMU stream
with the local fragment stream). `VX_cta_dispatch` sees them as ordinary kmu CTAs —
**no `cta_dispatch` change.** Completion is correct for free: fragment warps are
active warps → `sched_busy` → `per_core_busy` → device `busy` stays high until they
retire (verified: VX_core.sv:495 / VX_socket.sv:453).

**Payload = warp-self-pull (dissolves the producer↔warp binding).** Instead of the
producer pre-binding {payload, wid, lmem} (which forced the cta_dispatch override),
each fragment warp pulls its own quad: on entry it issues one scoreboarded
`vx_frag_fetch` that pops the next quad-wave from the per-core quad FIFO and stages
the per-lane `frag_payload_t` into the warp's **own** `__local_mem()` (the LSU base
`cta_dispatch` already assigned, passed as the op operand) via the LMEM DMA write
port (`LMEM_DMA_FWD_IDX`); the FS then LSU-reads it. Binding-free: the warp uses its
own lmem; the FIFO is self-served; same-pixel blend ordering stays correct because
the OM unit's same-pixel interlock serializes regardless of which warp shaded which
quad. No cta_dispatch override, no bcoord CSRs, no sentinel, no cross-core arb.

**Single-owner = the C5/crash fix.** One `VX_raster_kmu` per core consuming a private
quad stream replaces the multi-owner `VX_raster_arb` pull contention that triggers
the `cores≥2` `raster_mem` double-release. Expected to dissolve the crash structurally.

**Epoch.** `VX_raster_kmu` launches enough fragment warps to drain its FIFO, tracks
launched/retired (`warp_done`), and quiesces when the producer is drained and
launched==retired. The host still issues the raster stage as a normal launch (the
driver warp arms the producer via `vx_rast_begin`); the device `busy` covers the
fragment warps, so the existing CP launch-drain completion is unchanged.

## Reuse / rework of in-flight FWD-3a code

| Artifact | Disposition under v2 |
|---|---|
| `VX_gpu_pkg` `is_fwd_run` decode field | **Keep** — becomes the `vx_frag_fetch` discriminator. |
| `VX_decode` funct7=1 | **Keep** — decodes `vx_frag_fetch` (rs1 = dest lmem base, rd = scoreboarded done). |
| `VX_gpu_pkg` + `VX_mem_unit` `LMEM_DMA_FWD_IDX` agent | **Keep** — the fetch op's payload-stage write port. |
| `VX_raster_fwd.sv` (epoch + DMA-writer) | **Rework → `VX_raster_kmu.sv`**: keep producer-pull + quad FIFO + DMA-stage + launched/retired; replace `inject_*`→cta_dispatch with a `kmu_req` onto the local kmu bus; move the DMA stage to be driven by the per-warp `vx_frag_fetch` (operand = warp lmem base) rather than producer-picked bands. |
| `VX_raster_unit.sv` accept-and-defer + cta_dispatch ports | **Simplify**: `vx_frag_fetch` is a normal scoreboarded SFU op (pull + stage + done); no driver-defer, no inject ports. |
| `VX_cta_dispatch.sv` `fwd_inject_*` ports | **Revert** — option B does not touch cta_dispatch. |

## FWD-3 v2 execution plan (staged)

**Staging insight:** the part that fixes the doctrine (C2/C3/C5) *and* the `cores≥2`
crash is the **single-owner producer + warp-self-pull**. That works with
**host-launched** persistent fragment workers — which already enter through the kmu
bus as ordinary CTAs, so `cta_dispatch` is untouched and *no new launch source is
needed*. Device-side fragment launch (raster_kmu emitting kmu CTAs) is a clean,
optional follow-on that only changes *who sizes the worker grid*, not the work
distribution. So Stage 1 lands the fix; Stage 2 is the true-push enhancement.

### Stage 1 — single-owner producer + warp-self-pull (fixes doctrine + crash)

- **3a — op + decode.** `vx_frag_fetch` (funct7=1): rs1 = dest LMEM base, rd =
  scoreboarded done flag (1 = producer drained → worker exits). SFU raster PE routes
  it to the consumer. *(decode hook already landed + validated.)*
- **3b — `VX_raster_kmu.sv`** (per-core, replaces the per-core `VX_raster_unit` pull):
  single-owner consumer of the core's `raster_bus` → quad FIFO; serves `vx_frag_fetch`
  (pop wave → DMA-stage `frag_payload_t` to the rs1 LMEM base via `LMEM_DMA_FWD` →
  return drained flag in rd). No bcoord CSRs, no sentinel, no cross-core arb.
- **3c — collapse `VX_raster_arb`** to the single per-core producer→consumer stream
  (retire the sticky-done/flush/fan-out RR). `cta_dispatch`, kmu bus untouched.
- **3d — kernel.** GFX_FWD `gfx_draw3d` FS fragment role = persistent worker:
  `for(;;){ done=vx_frag_fetch(__local_mem()); if(done)break; p=lmem[lane]; shade;
  vx_om4; }`. Host launches the worker grid exactly as the legacy raster stage does
  (no launch-path change). Image-parity vs the SimX FWD golden (SimX keeps its push
  kernel — parity is byte-exact image, not kernel-identity).
- **3e — validate (cores=1).** rtlsim byte-exact: **box (2-drawcall) byte-exact vs
  golden + vase + evilskull all PASS** ✅. Opt-in via `GFX_FWD`; default builds
  unchanged. **Committed `98eb3d2a` on `prism`.**

> **cores>1 moved to FWD-6.** Multi-core raster correctness is a *pre-existing*
> swamp (legacy `vx_rast` deadlocks identically at cores=2, with and without the
> arb collapse; ≥2 independent bugs: a scheduler deadlock + corrupt multi-core
> quad distribution). It is orthogonal to the FWD-3 dispatch redesign (which is
> cores=1-complete and regression-free), so it is split out as **FWD-6** below.

### Stage 2 — device-launched fragment workers (true push; optional, after Stage 1)

`VX_raster_kmu` emits bare 1-warp fragment CTAs onto the core's *local* kmu bus via a
reused 2-input `VX_kmu_arb` (device stream + local fragment stream), so the device —
not the host — sizes/launches the worker pool. Needs a small arm path (driver warp or
a `RASTER_FRAG_*` DCR descriptor for entry/param/block_dim) + the per-core arb. Still
`cta_dispatch`-untouched (workers are ordinary kmu CTAs). This is the on-device
front-end that a future Layer-B `CMD_DRAW` feeds.

## Layer B roadmap (later, not now) — `CMD_DRAW`

Do **not** repurpose `CMD_LAUNCH` for graphics (it already *is* the per-stage
mechanism; overloading it conflates compute dispatch with the graphics pipeline).
The true-GPU endpoint is a dedicated `CMD_DRAW` + `vx_enqueue_draw`, routed by the CP
to a new graphics **resource** (`RES_GFX` beside `RES_KMU`/`RES_DMA`/`RES_EVT`) that
**decomposes the draw on-device** into the stage dispatches — reusing the launch
fabric and `VX_raster_kmu` internally. This moves the nine-stage front end from host
`PipelinePool` onto the device and enables indirect/multi-draw / GPU-driven rendering.
It earns its keep only with on-device expansion, so it follows FWD-3 (Layer A) and is
forward-compatible: the same `VX_raster_kmu` is the consumer in both cases.

## Register-window payload (FWD-5, perf endpoint, C1)

Warp-self-pull into LMEM is one LSU round-trip per fragment. The zero-load endpoint
delivers the payload straight into the warp's **gfx register window** (the existing
`SETW/GETW` mechanism that TEX/RTU already use) — the `vx_frag_fetch` stages into the
window, the FS reads via `GETW`, no LMEM. Deferred to FWD-5 (perf, not correctness).

## FWD-6 — multi-core (cores>1) raster correctness (pre-existing swamp)

Distinct from the FWD-3 dispatch redesign. The shared cluster raster path
(`VX_raster_arb` fan-out + `VX_raster_mem`/`VX_mem_scheduler`) is buggy at cores>1
**independent of the dispatch mechanism**: legacy `vx_rast` box cores=2 deadlocks
identically (scheduler timeout), with both the original arb and the collapsed arb.
Observed (rtlsim, cores=2):
- **Scheduler deadlock** — legacy box, FWD vase, legacy vase (active_warps stuck on
  core0). The arb begin/flush race was *one* contributor (fixed by the collapse) but
  not the whole cause — legacy still deadlocks with flush removed.
- **Corrupt multi-core quad distribution** — FWD box cores=2 (post-collapse) reaches a
  misaligned access from a garbage `pid` (was masked by the deadlock before).

Approach: a dedicated `verilator --trace` waveform session (not blind RTL mutation —
4 targeted edits did not resolve it). Likely the C4 endpoint: static screen-space
tile→core ownership so each core consumes only its owned quads with an independent
drain (the SimX fix at 302eb580), replacing the cluster arb's work-stealing fan-out.
The FWD-3 v2 single-owner-per-core consumer is already in place to receive it.
Gate: gfx matrix byte-exact on rtlsim at cores=2 (and the multi-cluster matrix).

### FWD-6 progress (2026-06-25) — two bugs, one fixed

Re-measured on the current tree (post FWD-4/FWD-5): the **deadlock is GONE**
(FWD-3/4 eliminated it). `gfx_draw3d` triangle @ cores=2 now *completes*, failing
**32 px** (deterministic) = the last **2 covered-quad waves** dropped; cores=1 is
byte-exact. So FWD-6 is now a bounded correctness bug, not a swamp.

**Bug A — work-stealing distribution (FIXED, 32→16 px).** `VX_raster_arb`'s fanout
round-robined the merged quad stream across cores. Replaced with **static
bin→owner-core routing** (`owner = bin_index % NUM_OUTPUTS`, bin from the wave's
lane-0 screen pos `>> (BIN_LOGSIZE-1)`), so each bin's quads go to exactly one core
(C4) — no work-stealing. Plus a per-output `emit_sticky` gate
(`done_all && ~out_routing_busy[o]`) so the drain never clobbers an in-flight wave.
Verified: cores=1 still byte-exact (no regression); cores=2 32→16 px. The WIP diff
is preserved in `fwd6_owner_routing_wip.patch` (reverted from the shared tree, not
committed — it leaves cores=2 red).

**Bug B — residual core-side one-wave drop (OPEN, 16 px).** With owner-routing a
128×128 frame is a single 128 px bin → all quads route to core0 (core1 idle), yet
core0 still drops its **last wave** (one 4×4 block at px (72–75,76–79)) only at
cores=2, never at cores=1. The per-output sticky gate had **no effect** on it, so
it is NOT the arb — it is core-side: the `VX_raster_unit` quick-pop/window-stage or
the OM drain at end-of-frame behaves differently under the dual-core arb's bus
presentation. Needs a real waveform/`DBG_TRACE_RASTER` trace of core0's last
frag_fetch + OM write at cores=2. NOTE: enabling `DBG_TRACE_RASTER` for rtlsim needs
the define passed to **verilator's SV flags** (the xrt synth Makefile's
`DBG_TRACE_FLAGS`), not via `CONFIGS` (which only reaches the C++ compile).

Next session: re-apply `fwd6_owner_routing_wip.patch`, enable the raster SV trace,
and localize Bug B (core0 last-wave drop) — then a true >1-bin frame (>128 px) to
exercise both cores actively + the multi-cluster matrix.

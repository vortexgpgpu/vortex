# gfx_v2 — Removing `vx_rast_fetch`: the true-GPU fragment-dispatch migration

**Status:** PROPOSAL (for sign-off). **Tree:** `~/dev/vortex_v3/prism_v3` (branch
`prism`), driver `~/dev/mesa_vortex` (branch `prism`). **Date:** 2026-06-29.

**Goal.** Remove the `vx_rast_fetch` self-pull op **100%** from sw + rtl + simx +
mesa + docs, and replace it with the *true-GPU* fragment-dispatch model: a per-core
graphics work distributor that **launches** fragment waves as warps through the
standard warp-launch fabric, with the per-lane payload delivered at launch. This is
the end-state of master-plan §4 ("push, not pull") and the **Stage 2** of the
authoritative dispatch design ([gfx_v2_dispatch_architecture.md](gfx_v2_dispatch_architecture.md));
this doc is its concrete execution + the removal plan.

This proposal does **not** redesign the TE/BE raster math, OM, or TEX. It changes
*how covered quads reach the shader* and nothing else. Per project rule, all sw
changes validate on **SimX only**; RTL parity is a later gate.

---

## 0. TL;DR

`vx_rast_fetch` is a **pull**: the shader runs first as a persistent worker and asks
the rasterizer for work (`for(;;){ if(vx_rast_fetch()) break; shade; }`). A true GPU
is the inverse — the rasterizer's work distributor **pushes** fragment waves onto SMs
as launched warps; the shader is a straight-line program that runs once per wave and
exits. The push model is already partly built (the SimX scheduler `fwd_*` injector,
the `VX_kmu_arb` device-launch fabric, the register-window payload path); this
proposal **wires it up, makes it the only path, and deletes the pull op**.

| | Pull (today: `vx_rast_fetch`) | Push (target: true GPU) |
|---|---|---|
| Who starts | shader worker polls producer | distributor launches shader |
| ISA op | `vx_rast_fetch` (funct3=3 f7=1) | **none** — payload seeded at warp launch |
| Shader shape | persistent `for(;;)` worker loop | straight-line, one wave, exits |
| Work source | warp self-pull via `VX_raster_arb` | per-core `VX_raster_kmu` → local kmu bus |
| Doctrine | C5 single-owner ✗ (cross-core arb), C2 ✗ (loop) | C1–C5 clean |
| Maps to real HW | ✗ (no GPU polls the raster engine) | ✓ (raster engine's WD feeds SMs) |

---

## 1. Current architecture (deep analysis)

### 1.1 The rasterization pull pipeline (`vx_rast_fetch`)

A fragment shader today is a **persistent worker**. Host launches an ordinary
fragment grid (one driver warp per core's worth of work); each warp loops on
`vx_rast_fetch()` until the producer drains:

```c
// tests/graphics/gfx_draw3d/kernel.cpp:168
for (;;) {
    unsigned drained = vx_rast_fetch();   // pop next covered-quad wave
    if (drained) return;                  // producer drained → exit
    frag_payload_t p; vx_frag_load(p, drained);   // GETW window slots 8..21
    // interpolate from p.bcoord, sample TEX, submit vx_om4
}
```

The op is **CUSTOM1 funct3=3, funct7=1** ([vx_graphics.h:96](../../sw/kernel/include/vx_graphics.h#L96)),
decoded to `INST_SFU_RASTER` with `op_args.raster.is_fwd_run`
([VX_decode.sv:775](../../hw/rtl/core/VX_decode.sv#L775)); SimX maps it to
`RasterType::FWD_RUN` ([decode.cpp:1004](../../sim/simx/decode.cpp#L1004),
[types.h:670](../../sim/simx/types.h#L670)). It is a **scoreboarded** SFU op (C3-clean)
returning a *drained flag* in `rd`.

The data path, end to end:

1. **Producer (cluster-shared)** — `VX_raster_core` ([VX_raster_core.sv](../../hw/rtl/raster/VX_raster_core.sv)
   / SimX [raster_core.cpp](../../sim/simx/raster/raster_core.cpp)). Auto-arms on its
   DCR config write; on the first waiting fetch (`raster_bus_if.req_pending`) it kicks
   off the tile/prim load, runs the TE/BE walker, and emits covered-quad waves
   (`raster_stamp_t`: pos_mask + pid + bcoord[3][4]) onto the cluster `raster_bus`.
   Drain is a sticky `done` bit, gated on `fetch_triggered` so the first fetch of a
   frame never sees a stale drain.
2. **Arbiter** — `VX_raster_arb` ([VX_raster_arb.sv](../../hw/rtl/raster/VX_raster_arb.sv)).
   Routes N producers → M consumers with **per-output sticky-done state**
   (`consumer_served`), per-output activity tracking, **frame-rearm flush** on the
   producer's `done 1→0` edge, and a round-robin fan-out. This is the cross-core
   shared mutable side-band the doctrine forbids (C4) and the locus of the cores>1
   bugs (see §1.4).
3. **Consumer (per-core SFU PE)** — `VX_raster_unit` ([VX_raster_unit.sv](../../hw/rtl/raster/VX_raster_unit.sv)
   / SimX [raster_unit.cpp](../../sim/simx/raster/raster_unit.cpp)). "Quick-pops" a
   wave off the bus in one cycle (freeing the arb), then **window-stages** each lane's
   `frag_payload_t` one slot/cycle into the warp's gfx register window
   (`GFXW_FRAG_SLOT_BASE..`, FWD-5 zero-LMEM). Returns the drained flag scoreboarded.
4. **Shader readback** — `vx_frag_load`/`vx_frag_payload` = `GETW` chained on the
   fetch's drained flag ([vx_graphics.h:113-134](../../sw/kernel/include/vx_graphics.h#L113)).

So even in its cleanest current form, `vx_rast_fetch` is **a pull**: the warp must
already be running to ask for work, and the cross-core `VX_raster_arb` is the
single-owner the doctrine says must not exist.

### 1.2 The kernel/CTA launch architecture (the fabric we will reuse)

Compute work reaches warps through a **launch fabric** that the push model rides:

```
host DCR (entry PC 0xce1, grid/block dims, lmem size, warp_step, cluster shape)
  → VX_kmu          : walks the grid, emits one kmu_req_t per CTA on kmu_bus
  → VX_kmu_arb      : merges KMU streams onto a core's local kmu bus  ← reuse point
  → VX_cta_dispatch : admits a CTA into a fixed-stride LMEM slot; per cycle fires one
                      warp: {cta_fire, cta_wid, cta_PC, cta_tmask, cta_param, cta_init}
  → VX_scheduler    : on cta_fire → active_warps[wid]=1, warp_pcs=cta_PC,
                      thread_masks=cta_tmask, mscratch=cta_param; warp runnable now
```

`kmu_req_t` / `cta_csrs_t` carry the full launch context ([VX_gpu_pkg.sv:666-693](../../hw/rtl/VX_gpu_pkg.sv#L666));
SimX mirrors it with `cta_warp_record_t` ([cta_dispatcher.h:26](../../sim/simx/cta_dispatcher.h))
and `Scheduler::activate_warp(wid, rec)` ([scheduler.cpp:114](../../sim/simx/scheduler.cpp#L114)),
which seeds PC/tmask/mscratch + CTA CSRs and makes the warp immediately schedulable.
There are already **two** activation sources merged in the scheduler — `cta_fire`
(KMU) and `wspawn` ([scheduler.cpp:179-196](../../sim/simx/scheduler.cpp#L179)) — plus
the RTU async-trap injection precedent (`raise_async_trap`). A graphics work
distributor is just **a third launch source on the local kmu bus** — exactly how a
real GPU's raster engine feeds its SMs.

### 1.3 The push model is already half-built (and orphaned)

Three pieces of the push end-state already exist in the tree:

- **SimX FWD injector (orphaned).** `Scheduler::fwd_arm / fwd_push_wave /
  fwd_try_inject` + the epoch counters `fwd_launched_`/`fwd_retired_` and
  `fwd_is_fragment_` ([scheduler.cpp:420-514](../../sim/simx/scheduler.cpp#L420),
  [scheduler.h:140-263](../../sim/simx/scheduler.h#L140)). `fwd_try_inject` picks a
  free warp slot, builds a `cta_warp_record_t`, calls `activate_warp`, and seeds the
  per-lane payload — **this is the push model**. It is fully written, guarded by
  `VX_CFG_EXT_RASTER_ENABLE`, and **has no caller**: it is leftover from the
  FWD-1/FWD-2 driver-warp prototype (committed `4ba1e186`) that was later superseded
  by the self-pull op. The retirement hook is even still live
  ([scheduler.cpp:306-312](../../sim/simx/scheduler.cpp#L306)).
- **Device-launch fabric.** `VX_kmu_arb.sv` already merges kmu streams — the 2-input
  arb the chosen design needs ([gfx_v2_dispatch_architecture.md](gfx_v2_dispatch_architecture.md) §Stage 2).
- **Register-window payload (FWD-5).** The zero-load C1 endpoint already ships: the
  consumer write port into the gfx window (`win_wr_*`, [VX_raster_unit.sv:49-54](../../hw/rtl/raster/VX_raster_unit.sv#L49))
  and the `GETW` readback. The push distributor reuses this exact window-seed path —
  it just writes at *launch* instead of on a *pull*.

The migration is therefore **consolidation, not greenfield**: wire the orphaned
injector to a real arm, move the payload seed from the self-pull op to the launch,
delete the op.

### 1.4 Why `vx_rast_fetch` must go (not just be tidied)

- **It is a pull.** No real GPU shader polls the rasterizer; the work distributor
  launches the shader. The persistent-worker loop is a C2 (single-issue-per-logical-op)
  and C5 (single-owner lifecycle) compromise that exists only to bridge to a pull op.
- **It forces the cross-core arb (C4).** The self-pull fans many cores' fetches at one
  cluster producer, requiring `VX_raster_arb`'s sticky-done + frame-rearm-flush shared
  state — the exact C4 "cross-core arbiter sticky state" the doctrine bans, and the
  documented root of the cores>1 swamp (FWD-6 Bug A/B: a 16 px last-wave drop at
  cores=2, [gfx_v2_dispatch_architecture.md](gfx_v2_dispatch_architecture.md) §FWD-6).
- **It blocks `CMD_DRAW`.** Layer B (on-device draw expansion) feeds a *distributor*,
  not a polling shader. Keeping the pull op cements the wrong seam.

Removing it is the structural fix, and it is the gate the user named for "v2 complete."

---

## 2. Target architecture — `VX_raster_kmu` push (option B)

Per the authoritative dispatch doc, the chosen topology is a **per-core graphics work
distributor** on the local kmu bus (not KMU surgery, not cta_dispatch injection):

```
VX_raster_core (cluster producer, unchanged TE/BE math)
  → per-core quad stream (static screen-space tile→core ownership, C4)
  → VX_raster_kmu (NEW, per core): single-owner consumer; packs NUM_THREADS covered
       quads into a fragment wave; emits a bare 1-warp fragment CTA (kmu_req_t) +
       stages the wave's per-lane frag_payload_t into the target warp's register
       window (FWD-5) at launch
  → VX_kmu_arb (2-input: device-KMU stream ∥ local fragment stream)
  → VX_cta_dispatch (UNCHANGED — fragment waves are ordinary kmu CTAs)
  → VX_scheduler.activate_warp → fragment warp runs straight-line FS, exits
```

**Properties (all by construction):**
- **Push.** The distributor launches; the shader never polls. No `vx_rast_fetch`.
- **C1 zero-load payload.** Wave payload seeded into the register window at launch;
  FS reads via `GETW` (reuses the existing FWD-5 port). LMEM-seed is the fallback form
  if a launch-time window write proves awkward in RTL (matches the orphaned SimX
  injector, which seeds LMEM today).
- **C2/C3.** One launch per wave; payload handed off through scoreboarded registers;
  the warp retires under the scoreboard.
- **C4 single-owner.** One `VX_raster_kmu` per core consuming a private,
  statically-owned quad stream **replaces `VX_raster_arb` entirely** — no cross-core
  sticky state, no work-stealing fan-out. This is the FWD-6 C4 endpoint.
- **C5 epoch.** `VX_raster_kmu` is the single owner: `QUIESCED → FILLING → DRAINING →
  QUIESCED` on `producer_drained ∧ launched==retired` (counted, no sentinel). Device
  `busy` already covers active fragment warps ([VX_core.sv:495 / VX_socket.sv:453]),
  so the existing CP launch-drain completion is unchanged — **no KMU/launch-ABI change**.

**Shader becomes straight-line** (the kernel diff):
```c
// fragment role: launched once per wave, payload already in the window
frag_payload_t p; vx_frag_load_at_launch(p);   // GETW; no fetch, no loop
// interpolate from p.bcoord, sample TEX, submit vx_om4
// (no for(;;), no vx_rast_fetch, no drained flag)
```

This is the same FS math as today; only the *entry/exit shape* changes (loop → once).

### 2.1 Fragment-shader dispatch descriptor (DCR ABI)

In v2 **the raster engine launches the fragment shader on-device — the host does not
launch a fragment grid.** For the distributor to launch fragment warps it needs the FS
program address and its argument; these move from a host launch into the raster DCR
block (`0x060`; `0x060–0x065` unchanged, additive before `STATE_END`):

| DCR | addr | meaning |
|---|---|---|
| `VX_DCR_RASTER_FRAG_ENTRY_LO/HI` | `0x066/0x067` | FS entry PC — the distributor sets each fragment warp's PC here |
| `VX_DCR_RASTER_FRAG_PARAM_LO/HI` | `0x068/0x069` | FS kernel-argument pointer — the per-draw constants the shader reads (`mscratch`) |

This is the analog of a real GPU writing the pixel-shader program address into a
register before kicking the raster engine. Runtime carries it as
`raster_state_t::{frag_entry, frag_param}`, emitted by `program_raster`/`emit_raster`.
The raster config write is the fragment-draw trigger: the distributor launches one
fragment warp per covered-quad wave at `frag_entry` with `mscratch = frag_param`, seeds
the per-lane payload into the warp's register window (D2), and the FS runs straight-line
and exits.

No pull op, no arm flag, no driver/fragment role split — the FS is just the FS, and
`vx_rast_fetch` is removed, not toggled.

### 2.2 Trigger & run — how the device sustains a host-less fragment draw (SimX)

The one non-obvious mechanism: with no host fragment grid, something must keep the
device executing until the distributor drains. The seams (verified):

- **Arm (synchronous).** `ProcessorImpl::dcr_write` ([processor.cpp:272](../../sim/simx/processor.cpp#L272))
  broadcasts the raster DCRs to every cluster; `Cluster::dcr_write` routes them to its
  `RasterCore` ([cluster.cpp:542](../../sim/simx/cluster.cpp#L542)). On the descriptor's
  last write, the cluster arms each owned core's scheduler — `Scheduler::fwd_arm(frag_entry,
  frag_param)` — reaching cores via `sockets_.at(s)->core(c)->scheduler()` (the same
  per-core reach the RasterBus wiring uses, [cluster.cpp:356-363](../../sim/simx/cluster.cpp#L356)).
- **Sustain.** `Scheduler::running()` already counts `fwd_armed_`
  ([scheduler.cpp:259](../../sim/simx/scheduler.cpp#L259)), and `ProcessorImpl::run()`
  ([processor.cpp:221](../../sim/simx/processor.cpp#L221)) ticks until **no** cluster is
  `running()` and no channels are in flight. So once armed, the device keeps ticking with
  **zero active warps** — the distributor injects fragment warps each cycle
  (`schedule()` → `fwd_try_inject`, [scheduler.cpp:173](../../sim/simx/scheduler.cpp#L173)),
  they run and retire, and `fwd_done → fwd_disarm` returns the core to idle.
- **Kick.** A draw is one `OP_DRAW` bundle the CP walks step-by-step, draining each
  launch (`exec_inline_cmd_`, [cmd_processor.cpp:326](../../sim/common/cmd_processor.cpp#L326));
  `OP_DCR_WRITE` is inline, `OP_LAUNCH` pulses `vortex_start`). After the binning launches
  drain and the raster DCRs arm `fwd_armed_`, the bundle's final step pulses `vortex_start`
  with **no KMU grid** — a bare doorbell. `WaitBusy` sees busy from `fwd_armed_`,
  `WaitDrain` waits out the fragments. No host warps are created.
- **Layer-B convergence.** `OP_DRAW` already exists in the CP decode
  ([cmd_processor.cpp:243](../../sim/common/cmd_processor.cpp#L243)); folding the
  bare-doorbell kick into `OP_DRAW`'s own completion (so the descriptor write *is* the
  draw, no trailing launch step) is the natural Layer-B step — forward-compatible, same
  distributor underneath.

---

## 3. Migration plan (SimX-first; sw validated on simx only)

Staged so each step is independently green on SimX before the next. RTL parity and the
cores>1 (FWD-6) work are explicit later gates, not preconditions.

### Stage A — SimX device-launched push (replaces the pull path)
1. **Arm (§2.2).** `Cluster::dcr_write` captures `FRAG_ENTRY`/`FRAG_PARAM` into
   `RasterCore` and arms each owned core's `Scheduler::fwd_arm(frag_entry, frag_param)`;
   `RasterCore` starts producing on arm (decoupled from any fetch op). Runtime adds the
   bare-doorbell kick after the raster DCRs.
2. **Inject (descriptor-driven, no driver).** `SfuUnit` feeds `RasterCore` waves into
   `Scheduler::fwd_push_wave`; `fwd_try_inject` launches one fragment warp **into any
   free slot** with `PC/entry = frag_entry`, `mscratch = frag_param` — mechanism reused
   (D5), source changed from the (now-gone) driver warp to the descriptor. Remove the
   `fwd_driver_wid_` skip/park logic.
3. **Payload = register window (D2).** Seed the wave into the injected warp's gfx window
   by reusing the SFU's existing window-stage path (the code `vx_rast_fetch` used). The
   FS reads via `GETW` exactly as today — minus the fetch and its drained-flag chain.
4. **Straight-line FS kernel** for `gfx_draw3d` — run-once, no worker loop. **Delete the
   SimX pull handler** (`FWD_RUN` in `sfu_unit.cpp`) and the pull kernel form in the same
   change — no parallel path.
5. **Gate:** `graphics:simx` + `graphics_parity:simx` byte-exact vs golden (box
   2-drawcall + the scene set), cores=1 **and** multi-core (D4).

### Stage B — RTL `VX_raster_kmu` (parity, deferred per "defer synth until rtlsim-green")
5. **`VX_raster_kmu.sv`** (per core): single-owner `raster_bus` consumer + quad FIFO +
   wave packer + epoch counters; emits `kmu_req_t` (bare 1-warp fragment CTA) and the
   register-window seed. Reuse the reverted `VX_raster_fwd.sv` DMA-writer/epoch logic
   per the disposition table in [gfx_v2_dispatch_architecture.md](gfx_v2_dispatch_architecture.md).
6. **`VX_kmu_arb`** as the 2-input merge (device ∥ local fragment); `VX_cta_dispatch`
   untouched.
7. **Collapse `VX_raster_arb`** → delete it (single producer→distributor stream).
8. **Gate:** rtlsim byte-exact vs the Stage-A SimX golden, cores=1.

### Stage C — delete `vx_rast_fetch` 100% (the removal — see §4 checklist)
9. Remove the op from RTL/decode and mesa FS lowering (the SimX pull handler and the
   pull kernel form are already gone after Stage A); mesa emits the straight-line FS.
10. **Gate:** full `graphics`/`graphics_parity` green on SimX (and rtlsim once Stage B
    lands) with **no `vx_rast_fetch` symbol anywhere in the tree.**

### Stage D — FWD-6 cores>1 (folds in for free)
The single-owner per-core distributor + static tile→core ownership **is** the FWD-6 C4
endpoint; with `VX_raster_arb` gone, Bug A (work-stealing) and Bug B (last-wave drop)
have no host. Re-run the multi-core × multi-cluster matrix.

### Stage E (optional, later) — Layer B `CMD_DRAW`
On-device draw expansion (`RES_GFX`) feeds the same `VX_raster_kmu`. Out of scope here;
forward-compatible by construction.

---

## 4. `vx_rast_fetch` 100%-removal checklist (Stage C)

Every site found by `grep -rn "rast_fetch\|frag_fetch\|FWD_RUN\|is_fwd_run"`:

**SW (kernel/runtime/tests):**
- [vx_graphics.h:96](../../sw/kernel/include/vx_graphics.h#L96) — delete `vx_rast_fetch`;
  delete/retarget `vx_frag_payload`/`vx_frag_load` (become launch-time window reads),
  funct3=3 line in the encoding banner (lines 27).
- [vx_gfx_window.h:86](../../sw/kernel/include/vx_gfx_window.h#L86) — drop the
  "chained on vx_rast_fetch's drained flag" semantics; the window read no longer chains
  on a fetch handle.
- [vx_gfx_abi.h:172](../../sw/common/vx_gfx_abi.h#L172) — `frag_payload_t` comment
  ("On vx_rast_fetch() …") → "seeded at fragment-wave launch".
- All gfx kernels with a worker loop: `gfx_draw3d`, `gfx_raster`, `gfx_pipeline_{raster,
  om,tex}` (`tests/graphics/*/kernel.cpp`, `common.h`, `main.cpp`) → straight-line FS.

**RTL:**
- [VX_decode.sv:774-782](../../hw/rtl/core/VX_decode.sv#L774) — remove the funct3=3 case
  (`is_fwd_run`).
- `VX_gpu_pkg.sv` `is_fwd_run` field ([VX_gpu_pkg.sv:830](../../hw/rtl/VX_gpu_pkg.sv#L830)).
- Delete `VX_raster_unit.sv` (replaced by `VX_raster_kmu.sv`) and `VX_raster_arb.sv`
  (collapsed); prune the `req_pending` auto-arm from `VX_raster_bus_if.sv` /
  `VX_raster_core.sv` (the distributor owns arming).
- `VX_gfx_window.sv` / `VX_gfx_window_pkg.sv` — keep the window write port (now
  launch-driven), drop frag_fetch-specific comments.

**SimX:**
- [sfu_unit.cpp:457-471](../../sim/simx/sfu_unit.cpp#L457) — remove the `FWD_RUN` op
  handler (the SFU no longer services a fetch op; it feeds `fwd_push_wave`).
- [decode.cpp:1004](../../sim/simx/decode.cpp#L1004), [types.h:670](../../sim/simx/types.h#L670)
  — remove `RasterType::FWD_RUN`.
- [gfx_doctrine.h:74](../../sim/simx/gfx_doctrine.h#L74) — remove the FWD_RUN classify.
- `raster_unit.{h,cpp}` — `RasterUnit::process` (the op submit) deleted; `RasterReq`/
  `RasterStamp`/`RasterRsp` retained as the distributor↔producer plumbing.
- Scheduler `fwd_*` — **kept and promoted** from orphaned to the live push path.

**Mesa (`~/dev/mesa_vortex`):**
- `vp_nir_to_llvm.c` — FS lowering: stop emitting the `vx_frag_fetch`/worker-loop form
  (bible §0 notes it still emits `vx_frag_fetch` `.insn r 43,3,1` + a worker loop);
  emit the straight-line FS launched by the distributor.

**Docs:** update the four referencing docs (this one becomes the live design; bible
§13/§10, master-plan §4, the FWD impl doc's superseded sections marked done).

**Definition of done:** `grep -rn "rast_fetch\|frag_fetch\|FWD_RUN\|is_fwd_run"` over
`sw hw/rtl sim/simx ~/dev/mesa_vortex` returns **zero** (docs/history excepted), and
the gfx suites are green.

---

## 5. Decisions (for sign-off)

- **D1 — Arm via DCR descriptor, not a new ISA op.** Introducing `vx_fwd_run` only to
  delete it later (it shares funct3=3 with the op we're removing) is churn. A
  `RASTER_FRAG_*` DCR descriptor (entry/param; fragment waves are always `NUM_THREADS`,
  so no block_dim) armed at the raster stage is the real-GPU shape and survives into
  Layer B `CMD_DRAW`. *(Aligns with
  [gfx_v2_dispatch_architecture.md](gfx_v2_dispatch_architecture.md) §Stage 2.)*
- **D2 — Payload via register window (FWD-5), LMEM seed as fallback.** Window-seed is
  the C1 zero-load endpoint and reuses the shipping `win_wr_*`/`GETW` path; the SimX
  injector's current LMEM seed is the parity fallback if a launch-time window write is
  awkward in RTL.
- **D3 — `VX_cta_dispatch` stays single-source.** Fragment waves enter as ordinary kmu
  CTAs via `VX_kmu_arb`; the generic dispatcher never becomes graphics-aware (the
  rejected direct-injection trap).
- **D4 — SimX-first; cores=1 gates each stage; FWD-6/cores>1 folds into Stage D.**
  Per the standing rule, sw changes validate on SimX only; RTL parity is Stage B/D.
- **D5 — Promote, don't rewrite, the SimX `fwd_*` injector.** It already is the push
  model; wiring it (Stage A) is the lowest-risk path and gives an immediate SimX
  oracle for the RTL `VX_raster_kmu`.

---

## 6. Validation

- **SimX (each stage):** `VX_XLEN=32 pytest ci -m "graphics and simx"` +
  `"graphics_parity and simx"` byte-exact (tol 0); box 2-drawcall must pass.
- **Parity (Stage B/D):** the same matrix on rtlsim, cores 1/2/4 × multi-cluster.
- **Removal:** the §4 grep returns zero.
- **No regression:** `raytracing:simx` unaffected (shares only the gfx window, which is
  retained).

---

## 7. Risk / open items

- **RTL launch-time window seed.** Writing the register window at warp launch (vs. the
  current pull-time write) may need a small `VX_raster_kmu`→window port handshake;
  LMEM-seed (D2 fallback) de-risks it.
- **FWD-6 Bug B** (16 px last-wave drop, cores=2) is currently open; the single-owner
  distributor is its predicted structural fix but must be confirmed on rtlsim (Stage D).
- **Mesa FS rewrite** couples to the driver branch; gate it behind the device path so
  the host-llvmpipe fallback is unaffected during migration.
</content>
</invoke>

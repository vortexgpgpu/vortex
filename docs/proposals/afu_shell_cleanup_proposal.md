**Date:** 2026-05-21
**Status:** Draft — Phase 1 (CP bring-up) in progress
**Author:** Blaise Tine
**Related:**
[command_processor_proposal.md](command_processor_proposal.md),
[cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md),
[cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md),
[cp_pure_v2_callbacks_proposal.md](cp_pure_v2_callbacks_proposal.md),
[cp_xrt_integration_plan.md](cp_xrt_integration_plan.md),
[cp_opae_integration_plan.md](cp_opae_integration_plan.md),
[caps_cp_consolidation_proposal.md](caps_cp_consolidation_proposal.md).

> **Note:** this file's name (`afu_shell_cleanup_proposal.md`) predates
> the proposal's current scope. The AFU-shell cleanup is now only the
> *last phase* of committing Vortex to the Command Processor as its sole
> command path. Recommend renaming to
> `cp_command_path_consolidation_proposal.md`.

# Command Processor as the Sole Command Path — Proposal

## 1. Summary

Vortex should commit to the **Command Processor (CP)** as its single
host→GPU command path and delete the legacy direct-MMIO machinery.

A Phase-1 bring-up pass (see §2) established the key fact that **reframes
this proposal**: the *runtime* is already CP-only. Every kernel launch
and DCR op on all five backends (xrt, opae, simx, rtlsim, gem5) already
flows through `vx::Device::cp_submit_*`. There is **no** `VORTEX_USE_CP`
gate on the live path — that env var, and the `vx_device::start /
cp_post_launch / …` methods in the xrt/opae backends, are **unreachable
dead code**.

So this is not a migration of the dispatch path — that already happened.
What remains is:

1. **Finish bring-up** — prove the CP path green across the full xrt and
   opae CI matrices (Phase 1, nearly complete).
2. **Migrate the last legacy-MMIO surfaces onto the CP** — DCR
   cache-flush, console output (COUT), and the SCOPE debug bus still
   use legacy AFU MMIO registers (P1 / P3 / P6).
3. **Delete the dead runtime code** — `VORTEX_USE_CP` and the unreachable
   `vx_device` command methods (a pure dead-code removal).
4. **Delete the legacy AFU command logic** — the XRT/OPAE shells reduce
   to platform glue.

**Sole exception:** the standalone ISA-test simulators
`sim/simx/main.cpp` and `sim/rtlsim/main.cpp` run bare RISC-V ELFs and
keep driving the GPU's direct `start`/`busy`/`dcr` ports. They are a
bare-metal test harness, not a runtime backend.

This subsumes the earlier AFU-shell-cleanup scope and
`caps_cp_consolidation_proposal.md`'s Phase 3.

## 2. Findings from CP bring-up

### 2.1 The runtime is already CP-only

The dispatcher↔backend contract `callbacks_t`
([callbacks.h](../../sw/runtime/common/callbacks.h)) is **Platform-shaped**:
it exposes only memory primitives + `cp_mmio_read/write`. There is no
`start`, `dcr_write`, or `ap_ctrl` callback. Every backend's
[callbacks.inc](../../sw/runtime/common/callbacks.inc) wires exactly
those primitives.

All launches and DCR ops are built as `CMD_*` descriptors by
`vx::Device::cp_submit_*` ([vx_device.cpp](../../sw/runtime/common/vx_device.cpp))
and posted to the CP ring — see [vx_queue.cpp:327-365](../../sw/runtime/common/vx_queue.cpp).
This is unconditional and identical for all five backends.

Consequently, in [xrt/vortex.cpp](../../sw/runtime/xrt/vortex.cpp) and
[opae/vortex.cpp](../../sw/runtime/opae/vortex.cpp):
`getenv("VORTEX_USE_CP")`, `cp_enabled_`, `vx_device::start`,
`ready_wait`, `cp_init`, `cp_post_launch`, `cp_wait`, `dcr_write` — are
**all unreachable**. `dcr_read` is the one exception: it is still reached
via `download()`'s cache-flush (see §4).

### 2.2 The CP launch path is proven on the hardware AFUs

Because the runtime is CP-only, every `xrtsim`/`opaesim` test already
exercises `VX_cp_core` end-to-end (ring fetch → `CMD_LAUNCH` →
`VX_cp_engine` → `VX_cp_launch` → Vortex `start`). Bring-up drove the
full CI xrt/opae matrices through it:

- **xrt: 13/13 green** — vecadd, sgemm, dogfood (×3, incl. barrier/gbar),
  diverge, mstress (×5 memory configs), demo (×3, incl. L2/sockets/perf,
  and `--scope`).
- **opae: 14/16 green** — `demo_o0` fixed and verified; `printf`
  isolated to the OPAE COUT path (§2.3 bug 5); `dogfood_wb` (a
  2-cluster L2+L3 config) was progressing normally when wall-time-capped
  and is deprioritized in favour of fast single-core iteration.

The CP ring-buffer / `CMD_LAUNCH` / `Q_SEQNUM` machinery is solid.

### 2.3 Bugs found and fixed during bring-up

| # | Site | Nature | Status |
|---|---|---|---|
| 1 | `hw/rtl/fpu/VX_fpu_std.sv` | Bare `XLEN` (×7) instead of `` `VX_CFG_XLEN `` — a `VX_CFG_*`-migration leftover in the *synthesizable* FPU (sim uses the DPI FPU, so it was never compiled) | **Fixed** |
| 2 | `hw/rtl/core/VX_uuid_gen.sv` | Bare `NUM_WARPS` instead of `` `VX_CFG_NUM_WARPS `` — same migration leftover; surfaced via the SCOPE `vortex.json` pass | **Fixed** |
| 3 | `hw/rtl/cp/VX_cp_engine.sv` | False comb loop: one `always_comb` both drove the bid `valid` lines and read the arbiter `grant` lines (`start_evt`). Split `start_evt` into its own `assign` | **Fixed** |
| 4 | `hw/rtl/cp/VX_cp_axi_xbar.sv` | False comb loop: one `always_comb` both drove `axi_m.wvalid` and read `axi_m.wready` (the `s_wready` fan-back). Split into two blocks | **Fixed** |
| 5 | `tests/regression/printf` (opae) | Kernel scoreboard timeout. Root cause: the OPAE AFU's COUT FIFO was drained only by the legacy `ready_wait()` poll, which the CP-only launch path never calls — the FIFO fills, COUT writes back-pressure, the kernel stalls. Not a CP-dispatch bug | **Fixed by P3** — Option C deletes the snoop + FIFO |

Bugs 1–2 are unrelated to the CP. Bugs 3–4 *are* CP RTL — and notably
they are **false** combinational loops that only surface at Verilator
`-O0`: the CP RTL had **never been built at `-O0`, strict-linted, or
synthesized**. The CP launch path runs in every sim test, but its RTL
has not had a synthesis-grade quality pass.

### 2.4 What the findings change

- The "dual command path gated by `VORTEX_USE_CP`" framing in earlier
  drafts was wrong. There is one dispatch path (the CP). The remaining
  duality is leftover *legacy-MMIO surfaces* (DCR cache-flush, COUT,
  SCOPE, the `ap_ctrl` reset) plus dead code.
- The runtime-side "cutover" (§6 Phase 3) is a **dead-code deletion**,
  not a behavioral change.
- The genuinely remaining engineering is: the DCR/COUT/SCOPE migrations
  (P1/P3/P6), a CP-RTL synthesis-grade hardening pass, and the legacy
  AFU RTL deletion.

## 3. Goals and non-goals

### 3.1 Goals
- One command-dispatch path — the CP — for all five runtime backends
  (already true; this proposal makes it *exclusive* and removes the
  alternative dead code).
- All device-side services on the CP — launch, DCR, DMA, COUT, and the
  SCOPE debug bus — so the AFU shells host no Vortex-specific command,
  console, or debug logic.
- Legacy command logic deleted from both AFU shells; the shells reduce
  to platform glue (bus adapters + discovery/`ap_ctrl` contract +
  clock/reset).
- The CP RTL proven end-to-end (full CI green on xrt and opae, COUT and
  SCOPE included) **and** clean through Verilator `-O0`, strict lint,
  and FPGA synthesis.

### 3.2 Non-goals
- **Not** changing the public runtime API (`vortex.h` / `vortex2.h`).
- **Not** redesigning the CP itself.
- **Not** removing the GPU core's direct `start`/`busy`/`dcr` ports —
  the standalone `sim/{simx,rtlsim}/main.cpp` harnesses drive them
  directly, and the CP drives them for the runtime path.
- **Not** a flag-day deletion — legacy RTL is deleted only after the CP
  path is proven green (§6 phasing).

## 4. Current state

| Aspect | Today |
|---|---|
| Dispatch path | **CP only**, all five backends — `vx::Device::cp_submit_*` builds `CMD_*` descriptors into the ring; unconditional |
| `VORTEX_USE_CP` / `cp_enabled_` | Dead. Gates only `vx_device::cp_init` in xrt/opae, which nothing reaches (not in `callbacks_t`) |
| `vx_device::start/ready_wait/cp_post_launch/cp_wait/dcr_write` | Dead — unreachable via `callbacks.inc` |
| Caps | On the CP regfile (`GPU_DEV_CAPS`/`GPU_ISA_CAPS`), per `caps_cp_consolidation_proposal.md` |
| DCR (kernel config) | On the CP — `CMD_DCR_WRITE` descriptors |
| DCR (cache-flush) | Host-side flush **removed** (P1a) — `flush_caches()` + the per-`download()` flush deleted, `MMIO_DCR_ADDR` gone from xrt. Coherence becomes an explicit **`CMD_CACHE_FLUSH`** ring command the CP broadcast-executes (P1b, pending). No-op on write-through caches — the default — so the host removal is already correct there |
| COUT | Interim **Option C** in the tree — kernel appends to a per-hart slot, dispatcher reads it back; OPAE snoop + FIFO and simx/rtlsim taps deleted, XRT gains COUT. **Lossy** (silent truncation) — to be reworked onto the lossless stream ring (§10) |
| SCOPE | **Legacy** — `MMIO_SCP_ADDR` (XRT `VX_afu_ctrl` regs / OPAE `MMIO_SCOPE_*`); not on the CP |
| `ap_ctrl` reset | **Legacy** — `init()` still does one `write_register(MMIO_CTL_ADDR, CTL_AP_RESET)` |
| CP RTL quality | Runs in every sim test, but never `-O0`-built, strict-linted, or synthesized (see §2.3) |
| CP interrupt | `VX_cp_core.interrupt` is `UNUSED_VAR` in both AFUs — dead silicon |

## 5. Target architecture

```
  host runtime (xrt / opae / simx / rtlsim / gem5)
        │  builds CMD_* descriptors into a DRAM ring
        ▼
   Command Processor  (VX_cp_core RTL  |  cmd_processor.cpp model)
        │  fetch → decode → dispatch (launch / DCR / DMA / COUT / SCOPE / events)
        ▼
   Vortex GPU core   ◄── direct start/busy/dcr also driven, unchanged, by
                          sim/simx/main.cpp and sim/rtlsim/main.cpp
```

- The AFU shells (`VX_afu_wrap`, `vortex_afu.sv`) become pure platform
  glue: AXI-Lite/CCI-P plumbing to the CP regfile, AXI/Avalon memory
  adaptation, the discovery/`ap_ctrl` contract, clock/reset.
- The GPU core keeps its direct `start`/`busy`/`dcr` ports.

## 6. Remaining work — completion checklist

Legacy RTL is deleted only after the CP path is proven green — no flag
day without a proven replacement.

### Phase 1 — CP bring-up & validation *(nearly complete)*
- [x] Confirm the runtime dispatches exclusively through the CP (§2.1).
- [x] xrt CI matrix green on the CP path — 13/13.
- [x] Fix `VX_fpu_std` `XLEN` / `VX_uuid_gen` `NUM_WARPS` (bugs 1–2).
- [x] Fix the `VX_cp_engine` / `VX_cp_axi_xbar` false comb loops (bugs 3–4).
- [x] opae CI matrix green on the CP path — 14/16; `demo_o0` xbar fix
      verified. `dogfood_wb` (2-cluster L2+L3) deprioritized — single-core
      xrt/opae is the CP validation target.
- [x] `printf` opae failure characterized — passes simx + rtlsim, fails
      only on opae: the legacy OPAE COUT path. Folded into P3.
- [x] CP RTL clean at Verilator `-O0` (`demo_o0` passes on xrt and opae).
- [ ] Add the `-O0` build + strict lint as a CI gate for the CP RTL.

### Phase 2 — finish the legacy-surface migrations
- [x] **P1a — host-side cache-flush removed.** `Device::flush_caches()`
      and the per-`download()` flush are deleted from all backends;
      `MMIO_DCR_ADDR` + the dead `dcr_read`/`dcr_write` deleted from xrt.
      A host-orchestrated per-core flush loop is not how a GPU works.
- [x] **P1b — `CMD_CACHE_FLUSH`.** Cache coherence is now an explicit
      command in the CP ring (the AMD `ACQUIRE_MEM` model). `CMD_CACHE_FLUSH`
      (opcode `0x0A`, `VX_cp_pkg.sv`) classifies onto the DCR resource;
      `VX_cp_dcr_proxy` executes it by sweeping the proven per-core flush
      DCR-read (`VX_DCR_BASE_CACHE_FLUSH` → `VX_dcr_flush` → `VX_cache_flush`)
      across cores `0..N-1`, retiring only when the last core's flush
      completes. `N` rides in `arg0`, filled by the host from
      `VX_CAPS_NUM_CORES`. `cp_submit_launch()` posts it after every
      `CMD_LAUNCH` so the host then sees coherent results. **Option A** — no
      new cache or DCR-fabric RTL, it reuses the existing flush hardware;
      the CP sweeps cores internally (a true parallel broadcast was deferred
      as a multi-core perf-only optimisation). No-op on write-through caches
      (the Vortex default). Verified: `demo` passes on simx / rtlsim / xrt /
      opae.
  - **COUT-drain hardening (found during P1b bring-up).** The COUT poll-loop
    drain is now gated to the `CMD_LAUNCH` poll loop only — draining during
    the trailing `CMD_CACHE_FLUSH` read the ring while the kernel's tail-end
    writes were still settling. A final `drain_cout()` after the flush picks
    up the residual once memory is coherent. `vx_putchar` also gained a
    producer release fence (`fence ow, ow`) so the character store is
    globally visible before the `wr` pointer publish. Residual: rtlsim
    `printf` still shows rare COUT console-line duplication (≈2/8 runs; no
    data loss — a pre-existing launch-phase drain race, out of P1b scope).
- [x] **P3 — COUT (lossless stream ring).** Reworked onto the lossless
      back-pressured ring of **§10.4** (per-hart sub-rings, the no-atomics
      form of OQ-5: `SLOTS=64`, `RING=512`). `vx_putchar` back-pressures —
      it spins re-reading `rd_ptr` on a full ring, never drops; the host
      drains every iteration of the `cp_submit_cl_` poll loop
      (`drain_cout()`), printing per-hart. The OPAE snoop + FIFO and the
      simx/rtlsim memory taps are deleted — COUT is now plain device RAM.
      Verified: `printf` + `vecadd` pass on simx / rtlsim / xrt / opae
      with correct per-hart console attribution.
- [x] **P6 — SCOPE (lossless, independent sideband).** `VX_scope_tap`'s
      on-chip BRAM is now a back-pressured **ring** (§10.5): capture pauses
      (never overflows) when full; the host drains it over the existing
      bit-serial MMIO sideband every poll-loop iteration (`vx_scope_drain`,
      called from the `cp_submit_cl_` poll loop). The serial sideband is
      kept — independent of the membus/CP by design (§10.6). Confined to
      `VX_scope_tap.sv` + `scope.cpp`/`scope.h`; no new memory master.
      Also fixed: `blackbox.sh` did not propagate `--scope` to the test's
      runtime rebuild (`run_app`), silently relinking without `-DSCOPE`.
      **Verified on xrt *and* opae** (`demo --scope`): both PASS with valid
      VCDs. xrt drained 2454 samples continuously (ring depth 4096, never
      filled). opae's AFU tap is 1691 bits wide (28 serial words/sample) —
      the drain cannot match capture, so the ring fills and capture pauses;
      it drained the full 4096-deep ring losslessly with the pause gaps
      recorded as deltas. Both regimes (drain-keeps-up / drain-falls-
      behind) confirmed; never overflows in either.
- [x] **P5 — CP interrupt.** `VX_cp_core.interrupt` is now a one-cycle
      pulse raised when any queue retires a command (from the
      `VX_cp_completion` retire events), replacing the tied-to-0
      placeholder. On **xrt** it drives the AFU `interrupt` pin
      (`VX_afu_wrap.sv`), ORed with the now-dead legacy `VX_afu_ctrl`
      ap_ctrl interrupt (Phase 4 deletes the latter). **opae**'s CCI-P AFU
      has no dedicated platform interrupt pin — a CCI-P interrupt is an
      af2cp TX-channel request, out of this minimal wire-up's scope; the
      CP interrupt is left unconsumed there. Option (a): no host-visible
      ISR/ack register — the runtime still polls `Q_SEQNUM`; this readies
      the pin for a future interrupt-driven launch-wait. Verified: `demo`
      passes on xrt / opae / rtlsim (simx uses the C++ CP model — the RTL
      interrupt change does not apply).

### Phase 3 — delete dead runtime code *(done)*
- [x] Removed `VORTEX_USE_CP`, `cp_enabled_`, the backend CP-state members,
      and `vx_device::start/cp_init/cp_post_launch/cp_wait` from xrt/opae
      `vortex.cpp` — the dead earlier `VORTEX_USE_CP`-gated CP integration.
      `callbacks.inc` confirms the only live backend surface is `init`,
      `get_caps`, `mem_*`, `upload`/`download`/`copy`, `cp_mmio_*`.
- [x] xrt `ready_wait` was fully dead (xrt DMA is synchronous via XRT) —
      removed. opae `ready_wait` is **live** (the CCI-P DMA-completion poll
      called by `upload`/`download`/`copy`) — kept, with only its dead
      `cp_enabled_` branch stripped.
- [x] Removed the legacy `dcr_read`/`dcr_write` + `MMIO_DCR_RSP` from opae
      (xrt's were already gone with P1a); removed the now-orphaned CP
      regfile `#define`s (`CP_RING_SIZE`, `CP_Q_*`, `CP_OPCODE_LAUNCH`, …),
      keeping `CP_BASE` for the live `cp_mmio_*`.
- [x] Verified: `demo` passes on simx / rtlsim / xrt / opae; both files
      compile `-Werror`-clean.

### Phase 4 — delete legacy AFU RTL *(see §7)*
- [x] **XRT — done.** `VX_afu_ctrl.sv` is slimmed to a minimal AXI-Lite
      slave: an `ap_ctrl` stub at 0x00 (reports the kernel permanently idle,
      writes inert — the CP owns launches) plus the SCOPE bit-serial
      register pair (`ifdef SCOPE`). Deleted: the legacy launch FSM
      (`STATE_*`, `vx_start_legacy`, `vx_pending_writes`), GIE/IER/ISR + the
      legacy `interrupt`, `dev_caps`/`isa_caps`, and the legacy DCR
      registers. In `VX_afu_wrap.sv` the legacy launch FSM, the pending-
      write tracker, the legacy DCR fan-in mux and the `afu_ctrl_irq` wire
      are removed; `vx_start`/DCR now come solely from `cp_gpu_if`. The
      bit-12 MMIO demux is **kept** (it costs nothing and `VX_afu_ctrl`
      stays a slave for `ap_ctrl` + SCOPE — see below). Verified: `demo`
      and `demo --scope` both PASS on xrt; SCOPE drains a 2485-sample trace
      through the slimmed `VX_afu_ctrl`.
- [ ] **OPAE:** delete the `vortex_afu.sv` `STATE_*` FSM, the CCI-P DMA
      engine, `MMIO_STATUS` polling, legacy DCR, and the COUT snoop
      (after P3); keep the CCI-P interface, DFH/AFU-ID header, Avalon
      adapters, `VX_cp_axi_to_membus`, reset sequencing. **Coupled work:**
      opae's `upload`/`download`/`copy` currently use the AFU CCI-P DMA
      engine — deleting it requires migrating opae host↔device DMA onto the
      CP's `CMD_MEM_*` path (a bring-up of its own).
- [x] **MMIO demux:** the earlier "collapse to a single AXI-Lite slave"
      goal is superseded by the §10.6 SCOPE design — SCOPE must stay
      independent of the CP, so its bit-serial registers cannot fold into
      `VX_cp_axil_regfile`. `VX_afu_ctrl` therefore remains a slim second
      slave (`ap_ctrl` stub + SCOPE) and the trivial bit-12 demux stays.

### Phase 5 — regression & sign-off
- [ ] Full xrt + opae CI matrices green (COUT and SCOPE included).
- [ ] simx / rtlsim / gem5 re-verified green.
- [ ] CP RTL through FPGA synthesis — no combinational loops, lint clean.

## 7. Removal inventory (Phase 4)

### 7.1 XRT — `VX_afu_wrap.sv` + `VX_afu_ctrl.sv`
- Legacy launch FSM (`STATE_IDLE/RUN/DONE`, `vx_start_legacy`,
  `saw_busy`) → `vx_start` driven solely by the CP.
- `vx_pending_writes` write-drain folded into `ap_done` → fencing is a
  CP concern (`CMD_FENCE`); keep only a bus-level drain if genuinely
  needed (OQ-1).
- Legacy DCR registers (`ADDR_DCR_0/1`, response capture, stall) and the
  DCR mux → removed once P1 lands.
- `dev_caps`/`isa_caps` AFU copies → already on the CP regfile; delete.
- GIE/IER/ISR interrupt registers → replaced by the CP interrupt (P5).
- `VX_afu_ctrl` SCOPE registers → removed once P6 lands.
- AXI-Lite bit-12 demux → gone; `VX_cp_axil_regfile` is the sole slave.
- `VX_afu_ctrl.sv` → reduced to a minimal `ap_ctrl` stub (P4).

### 7.2 OPAE — `vortex_afu.sv`
- The `STATE_MEM_WRITE/MEM_READ/RUN/DCR_*` command FSM, the
  `cmd_args`/`cmd_type` MMIO decoder, and the entire CCI-P DMA engine →
  the CP (`VX_cp_dma` via `VX_cp_axi_to_membus`) is the device-side DMA.
- `MMIO_STATUS` polling, legacy DCR, `MMIO_SCOPE_READ/WRITE`,
  `dev_caps`/`isa_caps` → removed (SCOPE once P6 lands).
- COUT snoop + FIFOs → **removed (done in P3, Option C).**
- bit-10 MMIO demux + dual `mmio_rsp` mux → gone.
- Kept: CCI-P interface, DFH/AFU-ID header, Avalon adapters +
  `VX_mem_arb`, `VX_cp_axi_to_membus`, reset sequencing.

### 7.3 Runtime — `sw/runtime/{xrt,opae}/vortex.cpp`
- Remove the `VORTEX_USE_CP` env check, `cp_enabled_`, and every dead
  `vx_device` command method (§6 Phase 3). Pure dead-code deletion.

## 8. Risks and open questions

| Id | Item |
|---|---|
| R-1 | **CP RTL synthesis-grade quality.** The CP runs in sim but had never been `-O0`-built/linted/synthesized; bugs 3–4 are the evidence. Phase 1's hardening gate + Phase 5 synthesis must precede the legacy-RTL deletion. |
| R-2 | **`printf` opae failure (bug 5)** — root cause: the OPAE COUT FIFO's only drainer was the legacy `ready_wait()` poll, dead on the CP path. P3 (Option C) deletes the snoop + FIFO entirely, so this is resolved as a side effect. |
| OQ-1 | Does correctness need a bus-level platform-AXI write drain distinct from CP `CMD_FENCE`? If so a minimal drain stays in the shell. |
| OQ-2 | **Superseded by §10.** Option C (the device-memory console buffer) shipped as a lossy interim. The proper COUT design is the lossless back-pressured stream ring — see §10. |
| OQ-3 | **Superseded by §10.** SCOPE is *not* folded into the CP regfile — it stays an independent sideband on the lossless stream ring (§10.5–10.6). |
| OQ-4 | `ap_ctrl` stub semantics — confirm the XRT host runtime is satisfied by an idle/done stub driven from CP status, with no legacy `auto_restart`/interrupt-channel bits. |

## 9. Why commit to this

The CP is already the command path — it just is not yet the *only* one,
and its RTL has not had a synthesis-grade pass. Making it exclusive ends
the dual-surface tax (the legacy DCR/COUT/SCOPE MMIO registers, the
bit-12/bit-10 demux), collapses the AFU shells to thin platform glue,
and forces the CP RTL through the lint/synthesis quality bar it needs to
be trusted on real silicon. The legacy path has no future; §6 sequences
the deletion behind a proven replacement, never ahead of it.

## 10. Lossless observability streams (COUT + SCOPE) — design

COUT and SCOPE are both **device→host observability streams**. A debug
facility must not drop data — losing console output or trace samples
loses exactly the evidence you are debugging with. This section is the
proper design; it **supersedes** the lossy interim P3 (Option C) and the
"SCOPE on the CP" sketch in the earlier P6.

### 10.1 Principle — lossless back-pressure

A bounded ring buffer, drained continuously by the host. When the host
briefly falls behind, the **producer stalls** (back-pressure) — it never
overruns the ring. Never overflow, never drop; an occasional brief stall
is the accepted cost.

This is AMD's hostcall model — the ring HIP device-`printf` rides on,
where the kernel blocks if the host consumer falls behind. It is the
deliberate opposite of NVIDIA's printf FIFO (circular, overwrite-oldest,
lossy). The lossless choice lets COUT and SCOPE share one mechanism.

### 10.2 The shared mechanism — a back-pressured stream ring

The same ring *discipline* backs both COUT and SCOPE — over different
storage, deliberately:

- **COUT** — a bounded ring in **device memory** (large — KBs–MBs; uniform
  with the CP ring).
- **SCOPE** — the tap's existing **on-chip BRAM**, turned into a ring
  (§10.5). It deliberately does *not* use device memory: a debug
  instrument that rode the membus would go blind exactly when the membus
  is wedged (§10.6).

The discipline, identical for both:

- `wr_ptr` — monotonic write offset, advanced by the producer.
- `rd_ptr` — monotonic read offset, advanced by the host consumer.
- Occupancy = `wr_ptr - rd_ptr`; **full** when occupancy == `RING_SIZE`.
- Producer writes only into free space; **stalls** when full.
- Consumer reads `[rd_ptr, wr_ptr)`, then advances `rd_ptr`.

`wr_ptr`/`rd_ptr` live in device memory beside the ring: the host reads
`wr_ptr` and writes `rd_ptr` over its normal memory path; the producer
reads `rd_ptr` and advances `wr_ptr`.

### 10.3 The host drainer — the launch-wait poll loop

The host drains in the **CP launch-wait poll loop** (`cp_submit_cl_`):
every poll iteration, alongside the `Q_SEQNUM` check, it reads `wr_ptr`,
copies out `[rd_ptr, wr_ptr)`, and advances `rd_ptr`.

That is the anti-deadlock guarantee. The producers — the kernel (COUT)
and the scope hardware (SCOPE) — run only *while a kernel is executing*,
and the host sits in the launch-wait poll for that entire time. So the
poll loop, draining every iteration, **is** a drainer running
continuously and concurrently with the producer: a kernel that fills its
ring stalls only until the next poll iteration drains it.

This is exactly the failure mode to design against. The `printf` hang
(bug 5) and the lossy Option C are the two *wrong* answers:
- **Bug 5:** the OPAE COUT FIFO's drainer was the *legacy* `ready_wait()`
  poll — and the CP-only launch path polls `cp_submit_cl_` instead, which
  did **not** drain. Drainer gone → kernel stalls on a full FIFO forever.
  Back-pressure with **no drainer is a deadlock**; the fix is precisely
  to make `cp_submit_cl_` drain.
- **Option C:** drains only *once, after* each launch, and to dodge the
  stall it silently *drops*. Drain-after-the-fact is **lossy**.

(A separate background drainer thread was considered and rejected:
producers are active only during launches — when the poll loop already
spins — so a thread adds nothing, and it would have to serialize against
the non-thread-safe simx/rtlsim backends. The poll loop is simpler and
sufficient.)

### 10.4 COUT mapping

- **Producer:** the kernel — `vx_putchar` writes records into the ring.
- **Record:** `{hartid, char}` (or `{hartid, length, bytes}`), tagged
  with the producing thread so the host attributes output per-thread
  without per-thread buffers.
- **Back-pressure:** `vx_putchar` reads `rd_ptr`; on a full ring it
  spins re-reading `rd_ptr` until the drainer frees space.
- **Multi-producer:** all threads share one ring, so the `wr_ptr`
  advance is a contended reservation:
  - *Atomic reservation* (`amoadd` on `wr_ptr`) — exact, scales to any
    thread count; needs the RISC-V **A extension** (Vortex defaults it
    off). The CUDA/HIP model.
  - *Per-hart sub-rings* — no atomics, but bounded to ≤64 harts by the
    `hartid` fold. Interim Option C is the degenerate per-hart case with
    the drain done wrong.
  Recommendation: the atomic shared ring is the proper form (gate it on
  enabling A); per-hart is the no-atomics fallback. (OQ-5)

### 10.5 SCOPE mapping

- **Producer:** the scope taps. Today `VX_scope_tap` captures into a fixed
  on-chip BRAM of `DEPTH` entries and **stops when full**
  (`TAP_STATE_DONE`) — a one-shot window, read out only *after* the run
  over the bit-serial MMIO bus. It cannot trace longer than `DEPTH`
  events.
- **Lossless form:** the tap's BRAM becomes a **ring**. `waddr` (capture)
  and `raddr` (drain) are free-running pointers; occupancy =
  `waddr - raddr`. Capture no longer stops at `DEPTH` — it runs until an
  explicit `stop`.
- **Back-pressure:** when the ring is full the tap **suppresses the
  write** — capture pauses until the host drains. The delta-counter keeps
  running, so the pause shows up as a longer inter-sample delta: the trace
  stays time-coherent, with an explicit gap rather than overwritten
  (corrupted) data. "Stalls once in a while," never overflows; a deep
  BRAM + continuous drain keeps pauses rare.
- **Drainer:** the host drains in the launch-wait poll loop (§10.3) — each
  iteration `CMD_GET_COUNT` reports occupancy and the host pulls that many
  samples, advancing `raddr`. The scope captures only while a kernel runs
  and the poll loop spins for exactly that window, so the drain is
  concurrent with the producer.
- **The bit-serial MMIO bus is kept** — deliberately. It is the proven
  original SCOPE data path and it is *independent of the membus, device
  memory and the CP*. A SCOPE that streamed to a device-memory ring would
  be coupled to the very fabric it exists to debug — blind whenever that
  fabric is wedged. Keeping the serial sideband is what makes SCOPE a true
  independent debug instrument (§10.6). The change is confined to
  `VX_scope_tap` (BRAM → ring) and `scope.cpp` (drain during the run, not
  after); no new memory master, no AFU-shell or membus surgery.

### 10.6 The CP's role — none in the data path, deliberately

COUT and SCOPE are device→host *observability* streams; the CP ring is
the host→device *command* path. They are orthogonal and stay separate:

- The runtime allocates the stream rings at device-open (like the CP
  ring), but the CP does **not** mediate the data path.
- **SCOPE must not go on the CP** (this revises P6): a debug instrument
  coupled to the command processor goes blind exactly when you need it —
  when the CP is wedged. Real GPUs keep the debug sideband (JTAG/scan)
  independent of the command path for this reason. SCOPE stays an
  independent sideband, just lifted out of the AFU *command* shell into
  its own block.

### 10.7 Open questions

| Id | Item |
|---|---|
| OQ-5 | COUT multi-producer — atomic shared ring (needs ext-A) vs per-hart sub-rings (≤64 harts). *Resolved for P3: per-hart sub-rings (no-atomics form), `SLOTS=64`.* |
| OQ-6 | SCOPE back-pressure granularity. *Resolved: pause the scope (suppress the BRAM write) and record the gap via the delta-counter — no GPU clock-freeze. §10.5.* |
| OQ-7 | COUT `wr_ptr`/`rd_ptr` placement — device memory (one DMA per drain poll) vs a faster-poll location; drainer poll cadence. |
| OQ-8 | SCOPE control/data channel. *Resolved: keep the bit-serial MMIO sideband for both — independent of the membus/CP by design. §10.5/§10.6.* |

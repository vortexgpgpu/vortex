# CP → XRT Integration Plan

**Status:** Draft, May 2026
**Scope:** Closes out the `feature_cp` RTL work and brings up a real
`vx_enqueue_launch` flowing through the Command Processor on an XRT
FPGA bitstream.

This is the *operational* plan for the remaining work. The *design*
of each module lives in [`cp_rtl_impl_proposal.md`](cp_rtl_impl_proposal.md);
this plan sequences the commits, pins down design decisions that were
left open, and lays out the bring-up procedure on hardware.

---

## 1. Current status (as of this writing)

### Done & committed (verilator-tested in `hw/unittest/`)

| Module | Lines | TB scenarios | Status |
|---|---|---|---|
| `VX_cp_pkg` | 184 | n/a (types) | Committed |
| `VX_cp_if`  | 91  | n/a (modports) | Committed |
| `VX_cp_arbiter` | 110 | 5 | Functional, bug fix for power-of-2 N |
| `VX_cp_engine` | 210 | 13 commands | FSM verified end-to-end |
| `VX_cp_launch` | 75  | 3 | KMU start/busy handshake verified |
| `VX_cp_dcr_proxy` | 108 | 4 | Write + read paths verified |
| `VX_cp_unpack` | 119 | 7 | Cache-line walker verified (this commit) |

Six modules functional + tested in isolation. Runtime side
(`vortex2.h` + per-queue worker) is fully landed and exercised by
OpenCL + native tests on simx and rtlsim.

### Untracked skeletons (need AXI infrastructure to be testable)

| Module | Why blocked |
|---|---|
| `VX_cp_fetch` | AXI master read of the cmd ring |
| `VX_cp_dma` | AXI burst engine for `CMD_MEM_*` |
| `VX_cp_completion` | AXI master write of seqnum to `cmpl_addr` |
| `VX_cp_axi_xbar` | Fans N_FETCH + N_HELPERS sources into one master |
| `VX_cp_event_unit` | Wait-op comparator over event-slot reads |
| `VX_cp_profiling` | DMA timestamps into per-event profile slots |
| `VX_cp_core` | Top-level integration of everything above |

### Not started

- AXI-Lite register block (Q_RING_BASE / Q_TAIL / Q_HEAD / Q_CMPL /
  doorbell / CP_CTRL / CP_STATUS / CP_CYCLE / DEV_CAPS).
- AFU shim rework: `VX_afu_wrap.sv` (XRT) instantiating `VX_cp_core`
  alongside Vortex.
- XRT bitstream regen + on-FPGA bring-up.

---

## 2. Sequenced commit plan

Six commits, each a substantial+testable unit per the
[no-skeletons](../../../.claude/projects/-home-blaisetine-dev/memory/feedback_no_prs_direct_commits.md)
rule.

### Commit A — AXI interface definitions + AXI-Lite register block

**Files added:**
- `hw/rtl/cp/VX_cp_axi_m_if.sv` — single AXI4 master interface bundle
  (AR/R/AW/W/B). Mirrors the existing `VX_mem_bus_if` style; the
  bundle is internal to `rtl/cp/` so the XRT AFU's full AXI4 fabric
  doesn't need to change.
- `hw/rtl/cp/VX_cp_axil_s_if.sv` — AXI4-Lite slave bundle.
- `hw/rtl/cp/VX_cp_axil_regfile.sv` — the register block specified in
  `cp_rtl_impl_proposal.md §4` (CP_CTRL / CP_STATUS / DEV_CAPS / per-
  queue Q_RING_BASE / Q_HEAD_ADDR / Q_CMPL_ADDR / Q_RING_SIZE_LOG2 /
  Q_CONTROL / Q_TAIL_LO+HI doorbell / Q_SEQNUM / Q_ERROR). Updates
  the per-queue `cpe_state_t` array on writes; serves reads from
  the same.

**Test:** `hw/unittest/cp_axil_regfile/` — drives synthetic AXI-Lite
W/AW + AR/R transactions, verifies:
- Every register reads back what was written.
- `Q_TAIL_HI` write commits `{tail_hi_staging, tail_lo_staging}` into
  `q_state[qid].tail` atomically; `Q_TAIL_LO` write alone does not.
- `Q_CONTROL.enable` toggles `q_state[qid].enabled`.
- Read-only register writes are dropped silently (no crash).
- Out-of-range addresses return DECERR.

**Why this first:** Every subsequent CP module talks through one of
these two interfaces. Locking the AXI bundles + register layout
prevents a re-plumb after each module commits.

**Open design questions to resolve in this commit:**
1. AXI4 master ID width: parent §6 says 6 bits (`VX_CP_AXI_TID_WIDTH`).
   Confirm against the XRT shell's TID width.
2. Burst size limit for the master: XRT shell typically caps at 256 B
   bursts. Set `VX_CP_AXI_MAX_BURST_BYTES = 256` in `VX_cp_pkg`.
3. Reset semantics: synchronous (matches the rest of Vortex) — confirm.

---

### Commit B — VX_cp_fetch + VX_cp_axi_xbar + VX_cp_completion bundle

These three modules go together because they all share the AXI4
master and only make sense once the AXI fabric exists.

**Files added:**
- `hw/rtl/cp/VX_cp_fetch.sv` (currently skeleton) made functional.
- `hw/rtl/cp/VX_cp_axi_xbar.sv` (currently skeleton) made functional —
  fans `axi_cpe_fetch[NUM_QUEUES]` + `axi_dma` + `axi_event` +
  `axi_cmpl` + `axi_prof` into the single `axi_m`. Round-robin
  arbitration on AR/AW channels; routes R/B back by TID prefix.
- `hw/rtl/cp/VX_cp_completion.sv` (currently skeleton) made functional —
  consumes `retire_evt[NUM_QUEUES]` + `retire_seqnum[NUM_QUEUES]`,
  issues AXI write of the new seqnum to `q_state[qid].cmpl_addr`.

**Test:** `hw/unittest/cp_axi_path/` — instantiates fetch + xbar +
completion against a synthetic AXI4 slave model (simple memory with
configurable latency). Drives:
- Fetch with a programmed ring base + tail; verify it issues AR
  bursts that walk the ring, returns 64 B cache lines on R.
- Completion: pulse `retire_evt`; verify an AW + W + B sequence writes
  the right seqnum to the right address.
- Xbar fairness: two fetches + one completion concurrently; verify
  round-robin grants.

**Open design questions to resolve here:**
1. **Fetch granularity:** does fetch issue one 64 B AR per ring read,
   or batches multiple cache lines? v1 = one CL per AR (simpler).
2. **TID encoding:** parent §15 says high bits select the source
   (fetch[QID] vs DMA vs EVENT vs CMPL vs PROF), low bits carry per-
   source tags. Lock the bit layout in `VX_cp_pkg`.
3. **Completion ordering:** must seqnum writes be strictly in-order
   per queue? Yes (parent §6.8) — the engine pulses retire in order,
   completion just forwards. No reordering inside completion module.
4. **Ring wrap-around:** fetch must handle `tail` wrapping past
   `ring_size_mask`; verify TB covers this case.

---

### Commit C — VX_cp_dma

Standalone enough to commit separately from the fetch bundle: it
shares only the AXI fabric, not any internal state.

**Files added:**
- `hw/rtl/cp/VX_cp_dma.sv` (currently skeleton) made functional.
  Handles `CMD_MEM_WRITE` (host→device), `CMD_MEM_READ` (device→
  host), `CMD_MEM_COPY` (device→device). Encoded:
  - `arg0` = dst address
  - `arg1` = src address (or host pointer for WRITE/READ)
  - `arg2` = size in bytes
  Burst chunker splits into ≤`MAX_BURST_BYTES` AR/AW.

**Test:** `hw/unittest/cp_dma/` — drives `grant` + `cmd` (packed
`cmd_t`), connects DMA's AXI to a synthetic memory model with two
banks, verifies:
- WRITE: bytes appear at the dst address.
- READ: data read back from src matches the seed.
- COPY: dst bank ends up with src bank's contents.
- Size > MAX_BURST splits into multiple bursts; `done` only after
  all bursts complete.

**Open design questions:**
1. Does DMA need a separate AXI master port to Vortex's HBM (vs the
   host-shared AXI)? Parent §17 says CP_DMA_DEV_PORT toggles between
   DEDICATED (separate port to Vortex memory) and SHARED (single port,
   host writes route through xbar). v1 = SHARED (simpler; saves a
   port in the AFU). Document this choice.

---

### Commit D — VX_cp_event_unit + VX_cp_profiling

Both helpers that read/write event/profile slots over AXI but don't
arbitrate for shared resources (no bid lines).

**Files added:**
- `hw/rtl/cp/VX_cp_event_unit.sv` made functional. Handles
  `CMD_EVENT_SIGNAL` (write a seqnum to event slot addr) and
  `CMD_EVENT_WAIT` (poll an event slot until a comparison op holds).
- `hw/rtl/cp/VX_cp_profiling.sv` made functional. On `submit_evt /
  start_evt / end_evt` pulses from CPE, DMAs the (queued_ns,
  submit_ns, start_ns, end_ns) tuple to the per-event `profile_slot`
  address.

**Test:** combined `hw/unittest/cp_event_profile/` — drives
synthetic command + grant, verifies AXI traffic against a memory
model.

**Open design question:**
1. `EVENT_WAIT` polling: every cycle, or rate-limited (e.g. every
   16 cycles)? Rate-limiting reduces AXI bandwidth pressure on the
   xbar but adds latency. Default 16-cycle poll, configurable via
   `VX_CP_EVENT_POLL_INTERVAL` parameter.

---

### Commit E — VX_cp_core integration + AFU shim rework

The big integration commit. Wires every CP module together and
splices the result into `VX_afu_wrap.sv`.

**Files added/modified:**
- `hw/rtl/cp/VX_cp_core.sv` — replace the current skeleton with the
  full instantiation per `cp_rtl_impl_proposal.md §4`. Wires all CPEs,
  arbiters, helpers, xbar, regfile.
- `hw/rtl/afu/xrt/VX_afu_wrap.sv` (modify) — instantiate `VX_cp_core`
  alongside Vortex; route AXI-Lite slave by address range (legacy
  AP_CTRL at `0x000..0x0FF`, CP regs at `0x100..0x3FF`); route AXI4
  master through an AXI-mux that selects between CP and legacy host
  DMA. Keep the legacy AP_CTRL FSM as compat mode (engaged only
  when no CP queue is enabled).

**Test:** verilator lint on the integrated `VX_afu_wrap.sv` must
pass. Add `hw/unittest/cp_core/` — a top wrapper that drives a single
queue end-to-end: program ring base + 1 command in synthetic memory,
ring the doorbell, observe `retire_evt` and the completion write
to the cmpl slot.

**Open design questions to resolve here:**
1. AXI-Lite address map: confirm `0x100..0x3FF` doesn't collide with
   any existing AP_CTRL ranges. Check `hw/rtl/afu/xrt/VX_afu_ctrl.sv`.
2. Whether to keep the legacy compat path or remove it now. **Keep**
   — gives a fallback when bringing up the CP.

---

### Commit F — XRT FPGA bring-up

**Not a code commit until something fails on hardware.** This is the
on-FPGA validation step:

1. Re-run `make -C hw/syn/xilinx/xrt` to regenerate the bitstream
   with the CP-enabled `VX_afu_wrap.sv`.
2. On the target FPGA, run `tests/runtime/test_basic` and
   `tests/runtime/test_async` with `VORTEX_DRIVER=xrt` — these
   should pass via the legacy compat path (no CP queue enabled).
3. Update the xrt runtime backend (`sw/runtime/xrt/vortex.cpp`) to
   open a CP queue at `vx_dev_init` time and route `vx_enqueue_*`
   commands through the CP ring instead of the legacy AP_CTRL path
   (this is the runtime-side of "talking to the CP"). Single-commit
   change of ≈100 LOC. Add a `VORTEX_USE_CP=1` env to opt in;
   default off (legacy compat) until validated.
4. Run `tests/opencl/sgemm` on the FPGA via the CP path. PASS gates
   the milestone.

**Bring-up debug aids to land alongside this work:**
- `VX_CP_TRACE` define enables a per-cycle trace of CPE state, bid
  lines, retire pulses (one line per active CPE per cycle) — too
  expensive to leave on, gated behind the define.
- A `cp_status` print helper in `sw/runtime/xrt/vortex.cpp` that
  reads CP_STATUS + per-queue Q_ERROR via AXI-Lite and dumps to
  stderr on hang.

---

## 3. Estimated effort

| Commit | Rough scope | Risk |
|---|---|---|
| A — AXI bundles + regfile | ~600 LOC RTL + ~300 LOC TB | Low (mechanical) |
| B — fetch + xbar + completion | ~700 LOC RTL + ~400 LOC TB | Medium (TID routing) |
| C — DMA | ~300 LOC RTL + ~200 LOC TB | Low |
| D — event + profiling | ~400 LOC RTL + ~250 LOC TB | Low |
| E — core + AFU shim | ~250 LOC integration + ~300 LOC TB | High (cross-module debugging) |
| F — XRT bring-up | ~100 LOC runtime + bitstream regen | High (hardware) |

Total: ~2.6 kLOC RTL, ~1.5 kLOC test, plus the AFU/runtime wiring.
4-6 weeks of focused work, plus 1-2 weeks of bring-up debug.

---

## 4. What this plan deliberately does NOT cover

- **Phase 4+ features** (real `EVENT_*` / `FENCE` semantics, real
  per-resource `done` aggregation, interrupt path) — these can land
  *after* sgemm runs on XRT.
- **Multi-FPGA / N>1 CPE concurrent kernels** — needs Phase 4
  groundwork; out of scope until single-CPE works.
- **simx / rtlsim re-verification of the new runtime path** —
  postponed to the very last per
  [feature_cp backend priority](../../../.claude/projects/-home-blaisetine-dev/memory/feedback_cp_backend_priority.md).
  These backends build cleanly through the new `callbacks_t` but
  haven't been driven end-to-end on the new runtime; that gap is
  acceptable until CP + XRT is done.
- **opae backend updates** — same reason; deferred.
- **HIP / gem5 / chipStar verification on the new runtime** —
  out of scope of this branch's milestone.
- **Pre-existing simx multi-block `vx_start_g` bug** (vecadd / conv3
  regression tests with -0.001327 garbage on multi-threaded blocks) —
  pre-existing in `c0ba9f41`, not blocking XRT bring-up.

---

## 5. Open architectural questions (must answer before Commit B)

1. **Ring buffer placement:** host-side pinned HBM region (CP reads
   via AXI from the XRT shell's DDR/HBM port), or device-side memory
   (CP reads from Vortex's L2-bypass path)? **Recommendation:**
   host-pinned HBM in v1 — simplest, no contention with Vortex
   memory traffic. Parent §6.2 says this.

2. **Doorbell coalescing:** does the runtime issue one Q_TAIL write
   per command, or batch? Runtime-side decision (in
   [`vx_queue.cpp`](../../sw/runtime/common/vx_queue.cpp) when CP
   submission lands). v1: one write per `vx_queue_flush` call; let
   the host buffer multiple `vx_enqueue_*` between flushes.

3. **Reset propagation:** if the host writes Q_CONTROL.reset, does
   the CPE drain in-flight commands or hard-stop? **v1:** hard-stop
   (drop pending commands, force seqnum write of CP_ERROR_RESET).
   Documented behavior.

4. **Q_RING_SIZE_LOG2 limits:** parent says default 16 (64 KiB ring).
   What's the upper bound the AFU's HBM allocation can sustain? Pin
   in `VX_cp_pkg` as `VX_CP_RING_SIZE_LOG2_MAX`.

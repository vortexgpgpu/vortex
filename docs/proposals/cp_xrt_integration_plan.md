# CP → XRT Integration Plan

**Status:** Updated May 17 2026 (RTL substantially complete).
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
| `VX_cp_pkg` | 184 | n/a (types) | ✅ Committed |
| `VX_cp_if`  | 91  | n/a (modports) | ✅ Committed |
| `VX_cp_arbiter` | 110 | 5 | ✅ Functional + bug fix for power-of-2 N |
| `VX_cp_engine` | 210 | 13 commands | ✅ FSM verified end-to-end |
| `VX_cp_launch` | 75  | 3 | ✅ KMU start/busy handshake verified |
| `VX_cp_dcr_proxy` | 108 | 4 | ✅ Write + read paths verified |
| `VX_cp_unpack` | 119 | 7 | ✅ Cache-line walker verified |
| `VX_cp_axi_m_if` | 110 | n/a (interface) | ✅ AXI4 master bundle |
| `VX_cp_axil_s_if` | 82 | n/a (interface) | ✅ AXI4-Lite slave bundle |
| `VX_cp_axil_regfile` | 366 | 10 | ✅ Host control + atomic Q_TAIL commit |
| `VX_cp_fetch` | 179 | (with axi_path) | ✅ Ring walker + AXI master + embedded unpack |
| `VX_cp_completion` | 177 | (with axi_path) | ✅ Retire → seqnum AXI writeback |
| `VX_cp_axi_xbar` | 316 | (with axi_path) | ✅ N-source round-robin + TID routing |
| `VX_cp_dma` | 165 | 2 | ✅ MEM_READ/WRITE/COPY (single CL) |
| `VX_cp_core` | 408 | end-to-end | ✅ Full integration |

**9 verilator unit tests, all PASS:**
  - `cp_arbiter`, `cp_engine` (13 cmds), `cp_launch`, `cp_dcr_proxy`,
    `cp_unpack` (7 scenarios), `cp_axil_regfile` (10 scenarios),
    `cp_axi_path` (3 scenarios), `cp_dma` (2 scenarios),
    `cp_core` (CP end-to-end NOP retire through full module graph).

### Runtime + multi-backend verification

The async `vortex2.h` runtime + per-queue worker thread + legacy
`vortex.h` wrapper chain is verified on **all four backends**:

| Backend | sgemm (OpenCL) | vecadd (OpenCL) | Mechanism |
|---|---|---|---|
| `simx`     | ✅ PASS | ✅ PASS | functional simulation |
| `rtlsim`   | ✅ PASS | ✅ PASS | full-RTL verilator |
| `xrtsim`   | ✅ PASS | ✅ PASS | XRT-shell verilator (`make run-xrt TARGET=xrtsim`) |
| `opaesim`  | ✅ PASS | ✅ PASS | OPAE-shell simulation (`make run-opae`) |

POCL (the OpenCL implementation) calls into legacy `vortex.h`, which
since `210e1129` is a thin wrapper over `vortex2.h`. Verified that
the **same runtime path** drives every backend without per-backend
specialization.

### Remaining work (not committed)

1. **AFU shim rework**: `hw/rtl/afu/xrt/VX_afu_wrap.sv` to instantiate
   `VX_cp_core` alongside Vortex. Requires AXI-Lite slave address
   widening (kernel.xml change too) + AXI master mux. **Deferred to
   the FPGA bring-up session** — see §6 below — because every
   change here is validation-coupled to a real bitstream.
2. **OPAE AFU rework**: similar to XRT, applied to `vortex_afu.sv`.
3. **`VX_cp_event_unit`** + **`VX_cp_profiling`**: still skeleton.
   Engine retires `CMD_EVENT_*` / profile-flagged commands as NOPs
   today (documented in `VX_cp_engine.sv`), so omitting these is
   correctness-safe. Land as follow-up.
4. **CP-side runtime path** in `sw/runtime/xrt/vortex.cpp` and
   `sw/runtime/opae/vortex.cpp`: opt-in `VORTEX_USE_CP=1` env switch
   that bypasses legacy AP_CTRL and submits via the CP ring. Goes
   together with the AFU rework (no point landing one without the
   other).
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
- **HIP / gem5 / chipStar verification on the new runtime** —
  out of scope of this branch's milestone.
- **Pre-existing simx multi-block `vx_start_g` bug** (vecadd / conv3
  regression tests with -0.001327 garbage on multi-threaded blocks) —
  pre-existing in `c0ba9f41`, not blocking XRT bring-up.

**No longer deferred** (status changed since the original plan was
written): simx / rtlsim / xrt / opae backends are all verified
running OpenCL sgemm + vecadd via the new vortex2.h dispatcher path
(see §1 "Runtime + multi-backend verification" above).

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

---

## 6. FPGA bring-up procedure (next session, FPGA hardware required)

The CP RTL + per-module + integration TBs are all verified in
simulation. The next milestone needs an actual XRT-capable FPGA
(Alveo U50/U200/U280 etc) plus the Xilinx XRT runtime installed on
the host. This procedure is what to do once the hardware is available.

### 6.1 AFU shim rework (RTL side)

Edit `hw/rtl/afu/xrt/VX_afu_wrap.sv`:

1. Widen `C_S_AXI_CTRL_ADDR_WIDTH` from 8 to 12 bits (4 KiB control
   space). Update the matching `kernel.xml` and any synthesis
   metadata in `hw/syn/xilinx/xrt/`.

2. Decode the AXI-Lite slave by address range:
   - `0x000..0x0FF`: route to the existing `VX_afu_ctrl` legacy
     AP_CTRL path (preserves vortex.h drop-in compat).
   - `0x100..0xFFF`: route to a new `VX_cp_axil_s_if` wired to
     `VX_cp_core.axil_s`.

3. Instantiate `VX_cp_core` alongside Vortex:

   ```sv
   VX_cp_axi_m_if cp_axi_m ();
   VX_cp_gpu_if   cp_gpu_if ();

   VX_cp_core u_cp_core (
       .clk        (clk),
       .reset      (reset),
       .axil_s     (cp_axil_s_if),
       .axi_m      (cp_axi_m),
       .gpu_if     (cp_gpu_if),
       .interrupt  (cp_interrupt)
   );
   ```

4. Wire `cp_gpu_if.{dcr_req_*, dcr_rsp_*}` and `cp_gpu_if.{start,busy}`
   to the corresponding Vortex ports, BUT muxed with the legacy
   `VX_afu_ctrl` outputs. Mode select = `cp_enabled` register bit
   exposed by the regfile (mirror of `CP_CTRL.enable_global`); when
   set, CP drives Vortex, AFU_ctrl outputs are ignored. When clear,
   legacy AP_CTRL drives Vortex (current behavior).

5. Add an AXI4 master mux that fans Vortex's memory-bank masters AND
   `cp_axi_m` into the AFU's outputs (or alternatively, dedicate one
   of the memory banks to the CP — simpler but uses a bank).

6. Re-run `verilator --lint-only` on the AFU before any synthesis.

### 6.2 OPAE AFU rework

Same conceptual rework applied to `hw/rtl/afu/opae/vortex_afu.sv`.
The OPAE control plane uses MMIO writes instead of AXI-Lite but the
address-decode + CP instantiation pattern is identical.

### 6.3 Runtime (`sw/runtime/xrt/vortex.cpp`)

Add a `VORTEX_USE_CP` opt-in env var. When set, `vx_dev_init`:

1. Allocates a pinned host buffer for the ring (size = `1 <<
   VX_CP_RING_SIZE_LOG2`, default 64 KiB).
2. Allocates pinned buffers for the per-queue head + cmpl slots.
3. Writes the CP registers via AXI-Lite (mmap'd through XRT's
   `xrt::ip` API): Q_RING_BASE / Q_HEAD_ADDR / Q_CMPL_ADDR /
   Q_RING_SIZE_LOG2 / Q_CONTROL.enable=1, then CP_CTRL.enable_global=1.

Then route every `vx::Platform::*` method through the CP ring:
- `mem_upload` / `mem_download` / `mem_copy` → encode `CMD_MEM_*`
  commands into the ring, doorbell write to `Q_TAIL_HI`.
- `dcr_write` / `dcr_read` → `CMD_DCR_*`.
- `launch_start` / `launch_wait` → `CMD_LAUNCH`, wait on the cmpl
  slot.

When `VORTEX_USE_CP` is unset, the runtime stays on the legacy
AP_CTRL path (no change vs today).

### 6.4 Bring-up sequence on the host

```bash
# 1. Build the CP-enabled bitstream.
cd hw/syn/xilinx/xrt
make TARGET=hw  # or TARGET=hw_emu for SW emulation
# Produces vortex_afu.xclbin with VX_cp_core inside.

# 2. Smoke test on hw_emu (no FPGA needed; XRT-side emulation).
cd build/tests/runtime
make
LD_LIBRARY_PATH=$XILINX_XRT/lib:... VORTEX_DRIVER=xrt XCL_EMULATION_MODE=hw_emu ./test_basic
LD_LIBRARY_PATH=...                  VORTEX_DRIVER=xrt XCL_EMULATION_MODE=hw_emu VORTEX_USE_CP=1 ./test_basic

# 3. On the real FPGA: legacy path first (sanity).
cd build/tests/opencl/sgemm
make run-xrt TARGET=hw   # uses AP_CTRL legacy

# 4. On the real FPGA: CP path.
make run-xrt TARGET=hw OPTS="-n32"
# (env automatically forwards VORTEX_USE_CP=1 if exported)
```

### 6.5 Bring-up debug aids

Two helpers to land alongside the AFU rework so on-hardware hangs
have observability:

- **`VX_CP_TRACE` define** (RTL): enables a per-cycle `$display`
  trace of CPE state, bid lines, retire pulses (one line per active
  CPE per cycle). Too expensive for production but invaluable for
  initial bring-up. Gated behind the define so legacy builds aren't
  affected.
- **`cp_status` dump** (runtime): a function in
  `sw/runtime/xrt/vortex.cpp` that reads `CP_STATUS` + per-queue
  `Q_ERROR` via AXI-Lite and prints to stderr. Called on hang
  detection (e.g. when `launch_wait` times out) or on demand via a
  `VORTEX_USE_CP_DUMP=1` env var.

### 6.6 Known risks for bring-up

1. **AXI-Lite addr widening**: kernel.xml metadata must match the
   widened slave port or XRT bind fails at runtime. Lint the
   regenerated metadata before bitstream cooking.
2. **AXI master mux behavior under contention**: Vortex memory banks
   and CP axi_m sharing one downstream port can starve under heavy
   load. The simpler dedicate-a-bank-to-CP approach trades silicon
   for latency predictability. v1 recommendation: dedicate a bank;
   revisit if HBM bandwidth becomes the bottleneck.
3. **TID prefix collisions**: the xbar packs 2 bits of source ID into
   the high bits of TID. The Vortex memory side also uses TIDs.
   These flow through different AXI masters in the AFU so they don't
   collide directly, but on a shared bank/mux they would — confirm
   the master mux preserves TID independence per source.
4. **Pinned-memory alignment**: XRT's `xrt::bo` returns FPGA-visible
   addresses that are page-aligned (4 KiB). The CP ring + completion
   slot need to live in such pinned regions. The runtime side must
   use `xrt::bo` (not malloc + register).

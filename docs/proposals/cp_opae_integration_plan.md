# CP → OPAE Integration Plan

**Status:** Drafted May 17 2026. XRT integration landed (commit `15440a55`,
sgemm + vecadd PASS via `VORTEX_USE_CP=1` on xrtsim). OPAE is the next
backend to bring up.
**Scope:** Bring `VX_cp_core` into the Intel OPAE/CCIP AFU shell
(`hw/rtl/afu/opae/vortex_afu.sv` + `sim/opaesim/` + `sw/runtime/opae/`)
and verify sgemm + vecadd via the same `VORTEX_USE_CP=1` runtime flag.

This is the *operational* plan. The CP module designs themselves live
in [`cp_rtl_impl_proposal.md`](cp_rtl_impl_proposal.md). The XRT-side
integration that this mirrors is documented in
[`cp_xrt_integration_plan.md`](cp_xrt_integration_plan.md) and in the
commit message of `15440a55`.

---

## 1. Why OPAE is materially different from XRT

The XRT integration was a 5-file, ~550-LOC change. OPAE is structurally
harder because the AFU exposes neither AXI-Lite nor AXI4 at its
boundaries:

| Concern | XRT (done) | OPAE (this plan) |
|---|---|---|
| **Control plane** | `s_axi_ctrl_*` (AXI-Lite slave) — the host writes 32-bit registers at byte addresses 0x00..0xFF | CCIP MMIO packets on `cp2af_sRxPort.c0` — 64-bit writes/reads at 16-bit `mmio_req_hdr.address`. AFU dispatches on a custom command FSM (states `IDLE/MEM_READ/MEM_WRITE/RUN/DCR_WRITE/DCR_READ`) keyed on writes to `MMIO_CMD_TYPE` |
| **Legacy "start"** | Write `CTL_AP_START` bit 0 → `VX_afu_ctrl` pulses `vx_start` | Stage `MMIO_CMD_ARG0..2`, then write `MMIO_CMD_TYPE = CMD_RUN` → state machine pulses `vx_start` |
| **Memory protocol** | AXI4 master to host shell (`m_axi_mem_*`) per bank | Avalon-MM (`avs_address/read/write/waitrequest/burstcount/readdata/readdatavalid`) to local-DRAM banks; cache-coherent host memory goes via separate CCIP TX/RX channels |
| **DCR programming** | Host writes `MMIO_DCR_ADDR` then `MMIO_DCR_ADDR+4` (legacy `VX_afu_ctrl` emits a `dcr_req`) | Host stages `MMIO_CMD_ARG0/1`, writes `MMIO_CMD_TYPE = CMD_DCR_WRITE`, state machine pulses `dcr_req` |
| **AFU file shape** | Two files: thin `VX_afu_wrap.sv` (port + FSM) + reusable `VX_afu_ctrl.sv` (DCR/AP_CTRL register block) — easy to splice a demux at the boundary | One monolithic 1225-LOC `vortex_afu.sv` with inline MMIO/FSM/AVS/CCIP plumbing. Splice point is *inside* the file, not at its edge |
| **Memory arb** | One bank-0 path to arbitrate — fits a simple new 2:1 `VX_axi_arb2` (which we wrote) | Existing 2-input arbiter `cci_vx_mem_arb_in_if[2]` already merges {Vortex memory, CCIP DMA} into local memory; CP becomes input #3. Reuse the existing arb infra; don't roll a new AVS arb |
| **Runtime API** | `xrt::ip::write_register/read_register` (or `xrtKernelWriteRegister`) | `fpgaWriteMMIO64/fpgaReadMMIO64` from `libopae`; in opaesim, the equivalent helpers in `sim/opaesim/fpga.cpp` |

The XRT-style `VX_axi_arb2.sv` library module is **not** reusable on
OPAE — different protocol. The CP regfile and runtime *flag* names
(`VORTEX_USE_CP`) and the `cp_init / cp_post_launch / cp_wait` skeleton
*are* reusable as a runtime template.

---

## 2. Current OPAE architecture (read this first)

A walking tour of the files the next session will be editing.

### 2.1 `hw/rtl/afu/opae/vortex_afu.sv` (1225 LOC, monolithic)

Key landmarks:

| Lines | Block |
|---|---|
| 22–46  | Module port list (CCIP `cp2af_sRxPort`/`af2cp_sTxPort` + AVS local-mem buses per bank + AFU power/error signals) |
| 49–98  | Parameter localparams (CCI/AVS widths, MMIO offsets) |
| 100–106 | `STATE_IDLE/MEM_WRITE/MEM_READ/RUN/DCR_WRITE/DCR_READ` enum |
| 113–131 | `dev_caps` + `isa_caps` constants returned via MMIO reads |
| 137–148 | `vx_mem_req_*` / `vx_mem_rsp_*` wires (Vortex memory port array) |
| 150–161 | Command argument staging (`cmd_args[0..2]`, plus `cmd_dcr_addr`/`cmd_dcr_data` views) |
| 163–171 | MMIO request header decode + response channel binding |
| 277–349 | MMIO **read** handler (returns AFU header, status, dev_caps, isa_caps, DCR response, console output queue heads) |
| 351–392 | MMIO **write** handler (latches `cmd_args[0..2]` on writes to ARG0/1/2) |
| 394–507 | **Command FSM** — observes `is_mmio_wr_cmd` for `MMIO_CMD_TYPE` writes and transitions on `cmd_type` (CMD_RUN, CMD_DCR_WRITE/READ, CMD_MEM_READ/WRITE) |
| 509–680 | AVS/CCIP arbiter chain merging Vortex memory + CCIP DMA into local memory banks |
| 682+   | Vortex instantiation, DCR programming, AVS bank fanout |

The DCR + start signals come out of the command FSM at lines 439–459
(`STATE_DCR_WRITE`, `STATE_DCR_READ`, `STATE_RUN`). These are the
**splice points** for the gpu_if mux.

### 2.2 `sim/opaesim/`

- `vortex_afu_shim.sv` (176 LOC) — Verilator top wrapping `vortex_afu`. Holds parameter defaults.
- `opae_sim.cpp` (610 LOC) — drives the AFU clock, handles `fpgaWriteMMIO64` / `fpgaReadMMIO64` calls by poking `cp2af_sRxPort.c0.mmioWrValid/data/hdr`.
- `fpga.cpp` / `fpga.h` — opaesim shim for `libopae-c` API (matches the OPAE C header).
- `Makefile` — Verilator build with `RTL_PKGS` / `RTL_INCLUDE` (same pattern as xrtsim; needs the same `-I.../rtl/cp` + CP package files added).

### 2.3 `sw/runtime/opae/vortex.cpp` (574 LOC)

- Uses `fpgaWriteMMIO64` / `fpgaReadMMIO64` for control plane.
- `start()` writes `MMIO_CMD_TYPE = CMD_RUN`.
- `ready_wait()` polls `MMIO_STATUS` for the AFU FSM idle bit.
- Memory upload/download uses `fpgaBufAlloc` + CCIP `CMD_MEM_WRITE/READ` commands (the AFU does the actual DMA via CCIP).

Same overall shape as XRT's `vortex.cpp` — port the CP additions
section-for-section.

---

## 3. Design decisions

### 3.1 MMIO → AXI-Lite shim for CP regfile

`VX_cp_axil_regfile` expects an AXI-Lite slave (`VX_cp_axil_s_if`).
CCIP MMIO is a request-response packet protocol with no AXI semantics.
Need a thin SV adapter:

**Proposed module:** `hw/rtl/afu/opae/VX_cp_ccip_mmio_shim.sv` (new, ~150 LOC)

**Inputs:** the relevant subset of `cp2af_sRxPort.c0` (mmioWrValid,
mmioRdValid, hdr, data) and a hook for the MMIO response channel.

**Outputs:** a `VX_cp_axil_s_if.slave` instance.

**Mapping rule:** when host MMIO address bit-12 is set (`mmio_req_hdr.address[12]==1`),
route the access to the CP regfile; otherwise let the existing AFU MMIO
handler see it (same bit-12 split as XRT — keeps `CP_CTRL` at CP-offset
0x000 reachable without colliding with legacy MMIO at 0x000).

**Address translation:** CP regfile sees `axil_s.awaddr = {4'd0, mmio_req_hdr.address[11:2], 2'd0}`
— the CCIP MMIO address is in 64-bit-word units (per CCIP spec, address
units are 4 bytes for 32-bit MMIO and 8 bytes for 64-bit MMIO; verify
in `ccip_if_pkg::t_ccip_c0_ReqMmioHdr`), so a shift may be needed.

**Width translation:** AXI-Lite is 32-bit wide; CCIP MMIO is 64-bit.
The CP regfile only uses 32-bit register values. Two cleanest options:
- Truncate MMIO 64-bit writes to low 32 bits; ignore high half.
- Map host's 64-bit write to a single 32-bit AXI-Lite write; map
  64-bit read to two 32-bit reads concatenated. Adds a small FSM but
  preserves the option of CP regfile expanding to 64-bit later.

Recommend option 1 (truncation) — all CP regs are 32-bit today and the
plan can be re-evaluated when/if any expand.

**MMIO read response:** the existing AFU MMIO read handler already
drives `af2cp_sTxPort.c2`. The shim needs to *steal* the response
channel when the request was a CP read. Pattern: route based on the
same bit-12 split; the legacy handler ignores bit-12 reads, the shim
drives them.

### 3.2 gpu_if mux into Vortex DCR + start

Same pattern as XRT:
- `dcr_req_valid = cp_gpu_if.dcr_req_valid | lg_dcr_req_valid`
- `dcr_req_{rw,addr,data}` = CP wins on simultaneous valid
- `cp_gpu_if.dcr_req_ready = 1'b1` (Vortex DCR always accepts)
- `cp_gpu_if.dcr_rsp_*` = Vortex's `vx_dcr_rsp_*` (fan-out, no mux)
- `cp_gpu_if.busy = vx_busy`
- `vx_start = vx_start_legacy | cp_gpu_if.start`

**Legacy DCR source:** on OPAE that's the `STATE_DCR_WRITE`/`STATE_DCR_READ`
branches of the command FSM (lines 478–492), not a separate `VX_afu_ctrl`
module. Splice the rename: change the inline `vx_dcr_req_*` assignments
to `lg_dcr_req_*` and add the OR mux below.

**Command-FSM auto-advance for CP launches:** identical to the XRT
`saw_busy` guard. The OPAE FSM enters `STATE_RUN` only on `CMD_RUN`
writes today — extend it to also enter on `cp_gpu_if.start` (without
pulsing `vx_start`, since CP already drives `vx_start` via the OR
mux), and gate `STATE_RUN → STATE_IDLE` on `saw_busy && !vx_busy`.

### 3.3 CP `axi_m` → local memory

CP's `axi_m` is AXI4. Local memory is AVS. Two viable paths:

**Path A (recommended): bridge to the existing arb chain.**
The AFU already has `cci_vx_mem_arb_in_if[2]` merging Vortex + CCIP
DMA into local memory. Add a 3rd input:
- Adapt CP `axi_m` → `VX_mem_bus_if` using `VX_mem_data_adapter` (the
  same module the AFU uses for Vortex memory; it handles width/tag
  translation). CP DATA_W is 512, local mem data width depends on
  the platform (usually 512 too on Skylake-FPGA).
- Bump `cci_vx_mem_arb_in_if` to size 3 and feed the adapted CP input
  into slot [2].
- The existing arb already handles AVS conversion downstream.

**Path B: standalone AVS arbiter.**
Write a new `VX_avs_arb2.sv` merging the existing AFU-side AVS output
with CP's converted AVS output. Cleaner separation but doubles the
arbitration logic and burst-tracking work.

Path A is materially less code and uses tested infrastructure.

**Adapter selection:** look at how the AFU adapts `vx_mem_req_*` →
`vx_mem_bus_if[i]` (lines 538–571). Reuse `VX_mem_data_adapter` with
parameters for CP's AXI ID width (6 bits) vs the bus width.

**Alternative consideration:** Should CP's ring/cmpl buffers live in
host memory (CCIP) instead of local memory? Arguments for:
- The host polls `Q_CMPL_ADDR` for seqnum — cache-coherent host
  memory makes the poll trivially correct.
- The XRT integration puts them in local memory only because XRT
  exposes a flat host-mapped BAR.

Arguments against:
- Adds a CCIP master to the picture; CP would need a different
  TX-channel path.
- The runtime poll on xrtsim worked fine because xrtsim's BO sync is
  a no-op (DRAM backdoor). opaesim should be similar.

**Recommendation:** put ring/cmpl in **local memory** for symmetry
with XRT. Revisit only if poll correctness suffers.

### 3.4 Runtime CP path

Port from `sw/runtime/xrt/vortex.cpp`:
- `cp_init()` — `mem_alloc` for ring + head + cmpl; program CP regfile
  via 32-bit MMIO writes (`fpgaWriteMMIO32` or `fpgaWriteMMIO64`
  truncated). Use `CP_BASE = 0x1000`.
- `cp_post_launch()` — upload zeroed CL with `cmd_buf[0] = CMD_LAUNCH`;
  commit `Q_TAIL_LO` then `Q_TAIL_HI`.
- `cp_wait()` — poll `Q_SEQNUM` via MMIO read, then poll AFU `MMIO_STATUS`
  for idle bit (the OPAE equivalent of XRT's `AP_DONE`).
- `start()` and `ready_wait()` dispatch on `cp_enabled_`.

**Open question:** the OPAE MMIO is 64-bit per access. If CP uses
32-bit registers, the host issues a 64-bit write whose low 32 bits is
the value. The MMIO shim (§3.1) needs to drop the high half. Make
sure the runtime always supplies (value << 0) and not (value << 32).

---

## 4. Concrete change list

### 4.1 New files

| File | Purpose | ~LOC |
|---|---|---|
| `hw/rtl/afu/opae/VX_cp_ccip_mmio_shim.sv` | CCIP MMIO → AXI-Lite slave shim for CP regfile | 150 |
| `docs/proposals/cp_opae_integration_plan.md` | This document | (done) |

### 4.2 Modified files

| File | Change |
|---|---|
| `hw/rtl/afu/opae/vortex_afu.sv` | Splice MMIO bit-12 demux to feed `VX_cp_ccip_mmio_shim`; rename inline `vx_dcr_req_*` to `lg_dcr_req_*`; add gpu_if mux; extend `cci_vx_mem_arb_in_if` to 3-way and feed CP `axi_m` through `VX_mem_data_adapter`; instantiate `VX_cp_core`; add `saw_busy` guard to STATE_RUN |
| `sim/opaesim/Makefile` | Add `-I$(RTL_DIR)/cp` + explicit `VX_cp_pkg.sv VX_cp_if.sv VX_cp_axi_m_if.sv VX_cp_axil_s_if.sv` to `RTL_PKGS` |
| `sim/opaesim/vortex_afu_shim.sv` | No changes expected — MMIO addressing is internal to the AFU, not at the shim port boundary |
| `sw/runtime/opae/vortex.cpp` | Add `cp_init`/`cp_post_launch`/`cp_wait` mirroring XRT's; gate on `VORTEX_USE_CP=1`; add CP regfile offset constants (the `CP_BASE = 0x1000` block from `sw/runtime/xrt/vortex.cpp`) |

### 4.3 Estimated effort

| Phase | Effort | Notes |
|---|---|---|
| 4.3.1 CCIP MMIO shim + standalone TB | 1 session | Most novel new RTL; deserves its own unit test |
| 4.3.2 AFU integration + arb extension | 1 session | Splice + 3-way arb + gpu_if mux + saw_busy |
| 4.3.3 opaesim build + legacy regression | 0.5 session | Verifier-pedantic lint will surface issues |
| 4.3.4 OPAE runtime CP path | 0.5 session | Port XRT runtime |
| 4.3.5 sgemm + vecadd via CP | 0.5 session | Debug round-trip (expect a fix or two like XRT had) |
| **Total** | **~3.5 sessions** | Allow for one extra-debug session beyond happy path |

---

## 5. Verification plan

### 5.1 Standalone CCIP MMIO shim TB

New unit test in `hw/unittest/cp_ccip_mmio_shim/`. Scenarios:
1. Host MMIO write below 0x1000 → AFU's existing MMIO handler sees it; shim's `axil_s.awvalid` stays 0.
2. Host MMIO write at 0x1000 → shim drives `axil_s.awvalid` with `axil_s.awaddr=0`; AFU handler ignores.
3. Host MMIO write at 0x1100 → shim drives `axil_s.awaddr=0x100`.
4. Host MMIO read at 0x1004 → shim returns `axil_s.rdata` on the CCIP MMIO response channel.
5. Concurrent CP-range + legacy-range traffic → both sides see correct routing.

### 5.2 Legacy regression (no `VORTEX_USE_CP`)

After all RTL changes land, build opaesim and run:
- `timeout 120 make -C tests/opencl/sgemm run-opae`
- `timeout 120 make -C tests/opencl/vecadd run-opae`

Both must PASS without setting `VORTEX_USE_CP`. This proves the CP
integration is non-invasive when disabled — same property the XRT
integration satisfied (commit `15440a55`).

### 5.3 CP path

- `VORTEX_USE_CP=1 timeout 120 make -C tests/opencl/sgemm run-opae` → PASS
- `VORTEX_USE_CP=1 timeout 120 make -C tests/opencl/vecadd run-opae` → PASS

Expected debug output mirroring XRT:
```
info: CP enabled — ring=0x... head=0x... cmpl=0x...
```

### 5.4 Exit criteria

- All four corners (legacy/CP × sgemm/vecadd) PASS on opaesim
- Single commit mirroring `15440a55`'s structure
- `MEMORY.md` updated to reflect both XRT and OPAE done

---

## 6. Open questions

1. **CCIP MMIO address units.** Verify whether `mmio_req_hdr.address`
   is byte-addressed or word-addressed in the Intel CCIP spec for the
   AFU base address space. The bit-12 split assumes byte-addressed
   (i.e., 0x1000 = byte address 0x1000 = MMIO offset 0x1000).
2. **AVS burst handling for CP.** The CP issues 64-byte single-beat
   bursts (`awsize=6, awlen=0`). The AVS arb chain in the AFU expects
   `VX_mem_bus_if` cache-line writes. Confirm `VX_mem_data_adapter`
   handles this conversion correctly (it does for Vortex; verify the
   CP's TID width and burst shape are compatible).
3. **Real OPAE hardware.** Like XRT, real bitstream bring-up needs
   the AFU manifest (`AFU_image_h2v.json` / `*.json` in `hw/syn/altera/`)
   updated to advertise the new MMIO range. Defer to a hardware
   bring-up phase; not needed for opaesim.
4. **Bank allocation for ring/cmpl.** XRT runtime puts them on bank 0
   because the bank-0 arb is the only one wired to CP. On OPAE, the
   3-way arb is at the AVS level merging all-bank traffic — so CP can
   reach any local memory bank. Still pin ring/cmpl to bank 0 for
   symmetry / debuggability.

---

## 7. Sequencing recommendation

Land changes in this order (one commit per phase, mirroring XRT):

1. **Phase A**: Add CCIP MMIO shim + unit test. Standalone, no AFU
   changes. Verify in `hw/unittest/`.
2. **Phase B**: AFU integration (DCR mux + 3-way arb + VX_cp_core
   instance + saw_busy guard). Verify legacy regression passes on
   opaesim.
3. **Phase C**: Runtime CP path. Verify sgemm + vecadd PASS via CP.
4. **Phase D** (optional): Update `MEMORY.md` and close out the
   `feature_cp` branch's CP integration milestone.

Total: 4 commits, each substantial and testable per the
`feedback_no_prs_direct_commits` rule.

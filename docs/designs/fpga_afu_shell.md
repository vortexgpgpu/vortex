# AFU Shell (XRT / OPAE) — Design

**Scope:** the FPGA acceleration-function-unit (AFU) shells that adapt
Vortex to a host platform — the Xilinx/XRT shell
([`hw/rtl/afu/xrt/`](../../hw/rtl/afu/xrt/)) and the Intel/OPAE shell
([`hw/rtl/afu/opae/`](../../hw/rtl/afu/opae/)). This document covers the
**shell structure** (platform glue, memory adaptation, control/reset/
discovery contracts, and the XRT↔OPAE asymmetries). The Command Processor
integration these shells host is documented in
[`command_processor.md`](command_processor.md) §9.1 and is
not repeated here.

Both shells reduce to **platform glue + the Command Processor**: the CP
(`VX_cp_core`) is the sole launch and DCR source in both, and the legacy
launch FSMs, DCR registers, duplicated capability blocks, and the OPAE
DMA-command engine have been removed.

---

## 1. XRT shell

| File | Role |
|---|---|
| [`vortex_afu.v`](../../hw/rtl/afu/xrt/vortex_afu.v) | Vitis RTL-kernel top — thin wrapper instantiating `VX_afu_wrap`. |
| [`vortex_afu.vh`](../../hw/rtl/afu/xrt/vortex_afu.vh) | Defines/macros (`GEN_AXI_MEM`, `GEN_AXI_HOST`, the bit-12 window). |
| [`VX_afu_wrap.sv`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) | The real shell (~696 LOC): AXI-Lite bit-12 demux, `VX_cp_core`, `m_axi_host`, bank-0 `VX_axi_arb2`, the `Vortex_axi` instance. |
| [`VX_afu_ctrl.sv`](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) | Slimmed AXI-Lite slave (~322 LOC): `ap_ctrl` stub at 0x00 + a SCOPE serial register pair + SCOPE watchdog. |
| [`VX_afu_axil_demux.sv`](../../hw/rtl/afu/common/VX_afu_axil_demux.sv) | AXI-Lite demux splitting the control space on `addr[12]`, one outstanding transaction per direction. |

- **Control.** Host AXI-Lite `addr[12]` splits the slave: `addr[12]=0` →
  `VX_afu_ctrl` (ap_ctrl + SCOPE); `addr[12]=1` → the CP regfile (seeing
  its own 0x000-based space). The split lives in
  [`VX_afu_axil_demux.sv`](../../hw/rtl/afu/common/VX_afu_axil_demux.sv),
  which holds one outstanding write and one outstanding read. **W routing
  falls through from the AW being accepted in the same cycle** rather than
  using the latched route alone — AXI4 permits write data to arrive before or
  with its address, and routing W by a register that only updates at the AW
  handshake sent AW and W to different slaves and deadlocked the interface.
  Covered by `hw/unittest/afu_axil_demux`; see
  [`../proposals/afu_reset_architecture_proposal.md`](../proposals/afu_reset_architecture_proposal.md).
- **Memory.** Vortex banks 1..N pass straight to platform AXI; bank 0
  shares with CP `axi_dev` via `VX_axi_arb2`
  ([`:506-558`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L506)); CP `axi_host`
  drives a **dedicated `m_axi_host` AXI master** for the command ring +
  host DMA ([`:302-326`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L302)).
  `PLATFORM_MEMORY_OFFSET` is applied per bank.
- **Interrupt.** The AFU `interrupt` pin is driven from `cp_interrupt`
  ([`:335`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L335)).

---

## 2. OPAE shell

| File | Role |
|---|---|
| [`vortex_afu.sv`](../../hw/rtl/afu/opae/vortex_afu.sv) | CP-only CCI-P AFU shim (~779 LOC): MMIO demux, `VX_cp_core`, dual `VX_cp_axi_to_membus` bridges (host CCI-P + device Avalon), `VX_mem_arb`, `VX_avs_adapter`, DFH header, SCOPE. |
| `ccip_std_afu.sv`, `ccip_interface_reg.sv`, `ccip/ccip_if_pkg.sv`, `local_mem_cfg_pkg.sv` | CCI-P platform plumbing (unchanged). |

- **Control.** CCI-P MMIO **word-address bit 10** (the 0x1000 byte
  boundary) splits the space
  ([`vortex_afu.sv:116-171`](../../hw/rtl/afu/opae/vortex_afu.sv#L116)):
  low page → DFH / AFU-ID header + SCOPE; high → the CP regfile. A single
  response mux drives the CCI-P TX channel.
- **Memory.** Vortex membus ports + CP `axi_dev` (via
  `VX_cp_axi_to_membus`) merge through `VX_mem_arb` (Vortex-priority) into
  `VX_avs_adapter` → Avalon local memory
  ([`:511-675`](../../hw/rtl/afu/opae/vortex_afu.sv#L511)). CP `axi_host`
  reaches host memory over a hand-rolled CCI-P c0/c1 single-outstanding
  state machine (`HB_IDLE/RD/WR`,
  [`:396-456`](../../hw/rtl/afu/opae/vortex_afu.sv#L396)) — the only CCI-P
  DMA user (the legacy DMA engine is gone).
- **Interrupt.** OPAE has no platform interrupt pin; `cp_interrupt` is
  unused.

---

## 3. Shared components and asymmetries

Reused from the common libraries (not under `afu/`):
[`VX_axi_arb2.sv`](../../hw/rtl/libs/VX_axi_arb2.sv) (XRT bank-0 arbiter),
[`VX_mem_arb.sv`](../../hw/rtl/mem/VX_mem_arb.sv) (OPAE bank-0 arbiter),
[`VX_avs_adapter.sv`](../../hw/rtl/libs/VX_avs_adapter.sv) /
`VX_mem_data_adapter.sv` (OPAE Avalon), and
[`VX_cp_axi_to_membus.sv`](../../hw/rtl/cp/VX_cp_axi_to_membus.sv) (AXI→
membus bridge, both OPAE bridges). The top-level cores are
[`Vortex.sv`](../../hw/rtl/Vortex.sv) (OPAE, membus ports) and
[`Vortex_axi.sv`](../../hw/rtl/Vortex_axi.sv) (XRT, AXI ports), each
keeping direct `start`/`busy`/`dcr_*` ports.

**Key XRT↔OPAE asymmetries:** dedicated host-AXI master (`m_axi_host`) vs.
a CCI-P host-DMA state machine; an interrupt pin vs. none; `VX_axi_arb2`
vs. `VX_mem_arb` for bank-0 sharing; AXI-Lite `addr[12]` vs. CCI-P MMIO
word-address bit 10 for the control demux. Both key the Vortex reset-delay
shift register on `reset` alone (no host ap_reset), and both expose SCOPE
over a serial sideband.

---

## 4. Proposed but not yet implemented

1. **OPAE host↔device DMA still rides the AFU's CCI-P state machine**
   ([`vortex_afu.sv:396-456`](../../hw/rtl/afu/opae/vortex_afu.sv#L396)),
   and OPAE `upload/download/copy` still use it. Fully removing it requires
   migrating OPAE DMA onto the CP `CMD_MEM_*` path (as XRT already does via
   `m_axi_host`) — a real, un-started bring-up.
2. **CP-RTL synthesis-grade hardening gate** — an `-O0` strict-lint CI gate
   and FPGA synthesis sign-off for the CP RTL (it has never been
   synthesized). The most load-bearing open item given the U55C @ 300 MHz
   timing target; some false combinational loops in `VX_cp_engine` /
   `VX_cp_axi_xbar` only surfaced at Verilator `-O0`.
3. **OPAE CP interrupt** — `cp_interrupt` is unconsumed; a CCI-P interrupt
   (af2cp TX request) would enable interrupt-driven launch-wait.
4. **Shell-level platform-AXI write drain** — whether a bus-level write
   drain (distinct from CP `CMD_FENCE`) is needed in the shell is open.
5. **Atomic COUT lossless stream ring** (gated on RVA) — the shipped form
   is per-hart sub-rings without atomics.

**Superseded directions** (recorded to avoid revival): the legacy AP_CTRL
launch FSM, per-AFU DCR registers, duplicated `dev_caps`/`isa_caps`, the
GIE/IER/ISR interrupt block in `VX_afu_ctrl`, and the OPAE `STATE_*`
DMA-command FSM / `MMIO_STATUS` / COUT-snoop — all removed in favor of the
CP being the sole command path. Collapsing everything onto a single
AXI-Lite slave was also considered and rejected: `VX_afu_ctrl` is kept as
a slim second slave.

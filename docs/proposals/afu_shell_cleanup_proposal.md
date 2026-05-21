**Date:** 2026-05-20
**Status:** Draft — not yet approved
**Author:** Blaise Tine
**Related:**
[command_processor_proposal.md](command_processor_proposal.md),
[cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md),
[cp_xrt_integration_plan.md](cp_xrt_integration_plan.md),
[cp_opae_integration_plan.md](cp_opae_integration_plan.md),
[caps_cp_consolidation_proposal.md](caps_cp_consolidation_proposal.md).

# AFU Shell Cleanup — Reducing XRT/OPAE AFUs to Pure Platform Glue — Proposal

## 1. Summary

The Command Processor (`rtl/cp/`) is now integrated into both FPGA AFU
shells, but it runs **alongside** the pre-CP control logic rather than
replacing it. Both shells still carry a complete legacy command path:

- **XRT** — [VX_afu_wrap.sv](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) +
  [VX_afu_ctrl.sv](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) carry a legacy
  launch FSM, a DCR register interface, device caps, an interrupt
  register block, and a write-drain counter.
- **OPAE** — [vortex_afu.sv](../../hw/rtl/afu/opae/vortex_afu.sv)
  carries an *entire hand-rolled command processor*: a
  `STATE_MEM_WRITE/MEM_READ/RUN/DCR_*` FSM, a full CCI-P DMA engine,
  an MMIO command-arg decoder, plus device caps and a console-output
  snoop.

This is migration scaffolding. An AFU shell should be **pure platform
glue** — the bus adapters and discovery contract that bind Vortex+CP
to a specific FPGA platform — and nothing else. Every command-
processing, device-description, or control-state function now has a
home in `rtl/cp/` or the GPU core.

This proposal is the **audit + removal plan**: it inventories the
non-glue logic in both shells, defines the target end-state shell, and
sequences the deletions against legacy-path retirement. It is the
"delete the old surface" companion to the CP integration plans, which
add the new one. It does *not* itself add CP functionality.

The [caps_cp_consolidation_proposal.md](caps_cp_consolidation_proposal.md)
is a strict subset of this work (the DEV/ISA caps line item, §5.1/§6.1);
this proposal subsumes and coordinates it.

---

## 2. Goals and non-goals

### 2.1 Goals

- **Define the end-state AFU shell** — a precise list of what each
  shell keeps (§7) and what it sheds (§5–6).
- **Inventory the non-glue logic** with file:line precision so the
  removal is mechanical, not exploratory.
- **Sequence the deletions** against legacy `vortex.h` retirement so
  no bitstream loses a surface the runtime still uses (§6).
- **Surface backend divergences** the cleanup must resolve — notably
  console output (OPAE-only) and the dropped CP interrupt.

### 2.2 Non-goals

- **Not** adding CP capability — that is the CP integration plans.
- **Not** removing anything while the legacy synchronous `vortex.h`
  runtime still depends on it; §6 gates every deletion.
- **Not** changing CCI-P / XRT platform contracts (DFH header,
  `ap_ctrl`, Avalon/AXI interfaces).
- **Not** redesigning the CP. Where a function must *move* rather than
  delete (console output), this proposal scopes the destination but
  defers the design to the owning component's proposal.

---

## 3. Principle: what an AFU shell is for

An AFU shell exists to bind the platform-agnostic Vortex+CP to one
FPGA platform. Its legitimate responsibilities:

1. **Host control bus** — AXI4-Lite (XRT) / CCI-P MMIO (OPAE)
   plumbing into the CP regfile.
2. **Memory bus adaptation** — width / address / tag conversion
   between Vortex's native memory bus and the platform's
   (AXI4 to platform DRAM; Avalon to local memory), plus the
   arbiter that merges the CP and Vortex memory masters.
3. **Discovery / control contract** — the OPAE DFH header + AFU-ID;
   the XRT `ap_ctrl` handshake.
4. **Clock / reset sequencing.**

Anything else — command decode, DMA, kernel launch, DCR transport,
device-capability packing, console output, interrupt aggregation — is
*not* platform glue and does not belong in the shell.

---

## 4. Current state

### 4.1 XRT

`VX_afu_ctrl.sv` is the legacy pre-CP control block; `VX_afu_wrap.sv`
hosts both it and `VX_cp_core` behind a bit-12 AXI-Lite demux.

| Logic | Location | Belongs to |
|---|---|---|
| Legacy launch FSM (`STATE_IDLE/RUN/DONE`, `ap_start`→`vx_start`) | [VX_afu_wrap.sv:91-95](../../hw/rtl/afu/xrt/VX_afu_wrap.sv), [:233-316](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) | `VX_cp_launch.sv` |
| `vx_pending_writes` write-drain counter, folded into `ap_done` | [VX_afu_wrap.sv:318-352](../../hw/rtl/afu/xrt/VX_afu_wrap.sv), [:239](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) | CP `OP_FENCE` (see §5.4) |
| DCR register interface (0x20/0x24 + response capture + stall) | [VX_afu_ctrl.sv:338-345](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv), [:437-450](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv), [:177](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) | `VX_cp_dcr_proxy.sv` |
| GIE / IER / ISR interrupt registers | [VX_afu_ctrl.sv:326-352](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv), [:454](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) | CP completion / interrupt path |
| DEV_CAPS / ISA_CAPS packing + read | [VX_afu_ctrl.sv:129-147](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv), [:409-420](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) | CP regfile (caps proposal) |
| SCOPE bit-serial bus | [VX_afu_ctrl.sv:181-250](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) | debatable — §5.5 |
| AXI-Lite demux (bit-12 routing) | [VX_afu_wrap.sv:153-231](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) | transitional — dies with legacy |

### 4.2 OPAE

`vortex_afu.sv` runs a complete hand-rolled command processor
alongside `VX_cp_core` ([vortex_afu.sv:1268-1275](../../hw/rtl/afu/opae/vortex_afu.sv)).

| Logic | Location | Belongs to |
|---|---|---|
| Legacy command FSM (`STATE_MEM_WRITE/MEM_READ/RUN/DCR_*`) | [vortex_afu.sv:100-106](../../hw/rtl/afu/opae/vortex_afu.sv), [:473-610](../../hw/rtl/afu/opae/vortex_afu.sv) | `VX_cp_engine.sv` / `VX_cp_launch.sv` |
| MMIO command-arg decode (`cmd_args`/`cmd_type`, `MMIO_CMD_*`) | [vortex_afu.sv:150-161](../../hw/rtl/afu/opae/vortex_afu.sv), [:431-471](../../hw/rtl/afu/opae/vortex_afu.sv) | `VX_cp_unpack.sv` |
| CCI-P DMA engine — read pipeline | [vortex_afu.sv:907-1056](../../hw/rtl/afu/opae/vortex_afu.sv) | `VX_cp_dma.sv` |
| CCI-P DMA engine — write pipeline | [vortex_afu.sv:1076-1173](../../hw/rtl/afu/opae/vortex_afu.sv) | `VX_cp_dma.sv` |
| `MMIO_STATUS` state-polling surface | [vortex_afu.sv:382-389](../../hw/rtl/afu/opae/vortex_afu.sv) | CP completion ring |
| Legacy DCR (`lg_dcr_req_*`, `MMIO_DCR_RSP`) | [vortex_afu.sv:1188-1208](../../hw/rtl/afu/opae/vortex_afu.sv), [:412-417](../../hw/rtl/afu/opae/vortex_afu.sv) | `VX_cp_dcr_proxy.sv` |
| DEV_CAPS / ISA_CAPS packing + read | [vortex_afu.sv:113-131](../../hw/rtl/afu/opae/vortex_afu.sv), [:398-411](../../hw/rtl/afu/opae/vortex_afu.sv) | CP regfile (caps proposal) |
| COUT console-output snoop + FIFOs | [vortex_afu.sv:304-328](../../hw/rtl/afu/opae/vortex_afu.sv), [:1277-1320](../../hw/rtl/afu/opae/vortex_afu.sv) | GPU core / CP — §5.6 |
| SCOPE bit-serial bus | [vortex_afu.sv:247-302](../../hw/rtl/afu/opae/vortex_afu.sv) | debatable — §5.5 |
| Bit-10 MMIO demux + dual `mmio_rsp` mux | [vortex_afu.sv:169-245](../../hw/rtl/afu/opae/vortex_afu.sv) | transitional — dies with legacy |

### 4.3 Scale

The two shells are not symmetric. XRT's legacy surface is a launch FSM
plus a register block; OPAE's is a *full DMA + launch + DCR command
processor* — the majority of `vortex_afu.sv`'s 1416 lines. The OPAE
cleanup is therefore the larger and higher-value half, and is gated on
[cp_opae_integration_plan.md](cp_opae_integration_plan.md) completing.

---

## 5. Removal items

### 5.1 Device / ISA caps

Covered in full by
[caps_cp_consolidation_proposal.md](caps_cp_consolidation_proposal.md).
Once the caps live in the CP regfile, delete the `dev_caps`/`isa_caps`
wires and their MMIO/AXI register cases from both shells. Note the
caps proposal's **OQ-2** is confirmed real here: OPAE's `dev_caps`
memory-bank fields use `LMEM_BYTE_ADDR_WIDTH` / `NUM_LOCAL_MEM_BANKS`
(derived from the Avalon `t_local_mem_*` types) — genuinely
OPAE-platform-specific, the one field that differs by shell.

### 5.2 DCR register interface

Both shells expose DCR over a legacy MMIO register and then mux that
against the CP DCR proxy ("CP wins on simultaneous valid"). Once the
legacy runtime no longer issues register-based DCR, delete the legacy
DCR registers, the response-capture logic, and the mux — `cp_gpu_if`
drives Vortex DCR directly.

### 5.3 Legacy launch / command FSM

- **XRT** — delete the `STATE_IDLE/RUN/DONE` FSM and `vx_start_legacy`;
  `vx_start` is driven solely by `cp_gpu_if.start`.
- **OPAE** — delete the `STATE_*` FSM, the `cmd_args`/`cmd_type`
  decoder, and the entire CCI-P DMA engine (read + write pipelines,
  the `cci_rd_req_queue`, the pending-size counters). `VX_cp_dma.sv`
  via `VX_cp_axi_to_membus` already provides the device-side DMA path.

### 5.4 Write-drain / fencing (XRT `vx_pending_writes`)

XRT couples `ap_done` to a count of outstanding AXI writes
([VX_afu_wrap.sv:318-352](../../hw/rtl/afu/xrt/VX_afu_wrap.sv)) — an
ad-hoc memory fence. Fencing is a CP concern (`OP_FENCE` is already a
CP opcode). **Caveat:** a bus-level drain of the platform AXI master
may still be needed, because `CMD_FENCE` operates on the logical
command stream, not on outstanding platform-AXI transactions. So this
item is *move + redesign*, not pure delete — see OQ-3. The legacy bit
to remove is the coupling into `ap_done`.

### 5.5 SCOPE bit-serial bus

Both shells implement a 64-bit bit-serial shift-register transport for
the GPU's scope debug taps. This is a generic debug transport, not
platform-specific. It is *debatable* whether it belongs in the shell
or as a CP debug register set. Low priority; the recommendation is to
leave it untouched in Phase 1–3 and revisit as a separate, optional
cleanup (OQ-4).

### 5.6 Console output (OPAE-only — backend divergence)

OPAE snoops Vortex memory writes to `VX_CFG_IO_COUT_ADDR`, diverts
them into per-port FIFOs, and lets the host drain them via
`MMIO_STATUS` ([vortex_afu.sv:304-328](../../hw/rtl/afu/opae/vortex_afu.sv),
[:1277-1320](../../hw/rtl/afu/opae/vortex_afu.sv)). The XRT shell has
**no console output at all** — a genuine feature gap, not just a
cleanup item.

Kernel `printf` is a *device feature*, not platform glue. It must
*move*, not delete. Two candidate homes (OQ-5):

- a GPU-core IO peripheral that exposes a console buffer in device
  memory the host polls — uniform across all backends; or
- a CP-mediated stream (a reserved ring or event).

Either way the snoop leaves the AFU shell and console output becomes
backend-uniform. This proposal scopes the destination; the design
belongs to whichever component owns it.

### 5.7 Interrupt aggregation + the dropped CP interrupt

Both shells instantiate `VX_cp_core` but leave its `interrupt` output
as `UNUSED_VAR` ([VX_afu_wrap.sv:418-419](../../hw/rtl/afu/xrt/VX_afu_wrap.sv),
[vortex_afu.sv:1265-1266](../../hw/rtl/afu/opae/vortex_afu.sv)), while
the top-level `interrupt` is driven from the legacy block (XRT's
GIE/IER/ISR; OPAE has none). This is a **live loose end**: the CP's
interrupt is currently dead silicon. The cleanup must wire the CP
interrupt to the platform `interrupt` pin and delete the legacy
GIE/IER/ISR register block.

### 5.8 The MMIO demux (both shells)

The bit-12 (XRT) / bit-10 (OPAE) routing that splits host MMIO between
the legacy block and the CP regfile exists *solely* because the two
control surfaces coexist. When the legacy block is gone, the demux and
the dual response mux go with it: `VX_cp_axil_regfile` becomes the
sole AXI-Lite slave / MMIO target. No separate work item — it falls
out of §5.1–5.3 + 5.7.

---

## 6. Sequencing

Every deletion is gated on the legacy synchronous `vortex.h` runtime
no longer needing the surface it removes. The order:

### Phase A — Prerequisites (other proposals)
- CP integrated in both shells (`cp_xrt_integration_plan.md`,
  `cp_opae_integration_plan.md`) — **done for XRT, in progress for OPAE**.
- Caps moved to the CP regfile (`caps_cp_consolidation_proposal.md`).
- Console output rehomed (§5.6 / OQ-5).

### Phase B — XRT shell reduction
Gated on: the XRT runtime issuing launch + DCR exclusively through the
CP regfile. Then, in one reviewed change:
- delete `VX_afu_ctrl.sv` (caps, DCR regs, GIE/IER/ISR, SCOPE stays or
  moves per OQ-4);
- delete the launch FSM and `vx_pending_writes`→`ap_done` coupling
  from `VX_afu_wrap.sv`;
- collapse the AXI-Lite demux; wire `ap_ctrl` to a minimal stub driven
  by `cp_busy`; wire the CP interrupt out.
- **Testable:** XRT bitstream builds; an end-to-end `vx_enqueue_launch`
  passes through the CP only; `xrt`-path regression green.

### Phase C — OPAE shell reduction
Gated on: `cp_opae_integration_plan.md` complete and the OPAE runtime
issuing all commands through the CP regfile. Then:
- delete the `STATE_*` FSM, `cmd_args`/`cmd_type` decode, and the
  CCI-P DMA engine;
- delete legacy DCR, caps, `MMIO_STATUS` state polling;
- collapse the bit-10 demux + dual `mmio_rsp` mux; wire the CP
  interrupt out.
- **Testable:** OPAE bitstream builds; end-to-end launch through the
  CP only; `opae`-path regression green.

### Phase D — Non-CP build decision
If any shipped configuration still builds an AFU *without* a CP, the
legacy logic cannot be deleted outright and must instead be guarded
(see OQ-1). Resolve before Phase B/C delete vs. guard.

---

## 7. Target end-state shells

After this cleanup:

**XRT** — `VX_afu_wrap.sv` only (`VX_afu_ctrl.sv` deleted):
- AXI4-Lite slave → `VX_cp_axil_regfile` directly (no demux);
- AXI4 master → `Vortex_axi` + memory width/offset/bank glue + the
  bank-0 `VX_axi_arb2` merging CP and Vortex;
- `ap_ctrl` → minimal XRT-required stub tied to `cp_busy`;
- reset sequencing; `interrupt` ← CP.

**OPAE** — `vortex_afu.sv` reduced to:
- CCI-P interface + DFH header / AFU-ID (the discovery contract);
- CCI-P MMIO → `VX_cp_axil_regfile` directly;
- Avalon local-memory interface + `VX_avs_adapter`; the
  `VX_mem_data_adapter` width conversions + the 3-way `VX_mem_arb`
  merging Vortex / CP / (residual CCI-P host access if any);
- `VX_cp_axi_to_membus` bridge; reset sequencing; `interrupt` ← CP.

`ccip_std_afu.sv` and `ccip_interface_reg.sv` are already pure CCI-P
clocking / timing-closure glue and are untouched.

---

## 8. Risks and open questions

| Id | Item |
|---|---|
| **OQ-1** | **Non-CP bitstreams.** Does any shipped config build an AFU without a CP? If yes, Phases B/C *guard* the legacy logic instead of deleting it; if no, delete outright. Blocks the Phase B/C delete-vs-guard decision. |
| **OQ-2** | **`ap_ctrl` stub semantics (XRT).** XRT requires the kernel to expose `ap_ctrl`. Confirm the host XRT runtime is happy with a stub that reports idle/done from `cp_busy` and never needs the legacy `auto_restart`/interrupt-channel bits. |
| **OQ-3** | **Bus-level write drain (XRT).** Does correctness need a platform-AXI outstanding-write drain distinct from CP `OP_FENCE`? If so, keep a minimal drain in the shell (it *is* platform-specific) but decouple it from `ap_done`. |
| **OQ-4** | **SCOPE bus placement.** Leave the bit-serial scope transport in the shell, or relocate to a CP debug register set? Low priority; can be a separate cleanup. |
| **OQ-5** | **Console-output home.** GPU-core IO peripheral vs. CP-mediated stream. Decides §5.6 and closes the XRT/OPAE divergence. Needs an owning proposal before Phase A completes. |
| R-1 | **Legacy runtime coupling.** Deleting a surface the synchronous `vortex.h` path still uses breaks it. Mitigation: §6 gates every deletion on the runtime cutover; Phases B/C are single reviewed changes with regression gates. |
| R-2 | **OPAE DMA path coverage.** The CCI-P DMA engine being deleted is the *host↔device* copy path. Confirm `VX_cp_dma.sv` + `VX_cp_axi_to_membus` cover every transfer shape the legacy `CMD_MEM_READ/WRITE` served before deletion. |

---

## 9. Why this is a separate proposal

The CP integration plans answer "how does the new surface get built."
This proposal answers "what does the old surface's removal look like,
and what is the shell when it's done." Keeping it separate means the
end-state shell is a reviewable artifact, the deletions are sequenced
against a concrete gate (legacy-runtime cutover) rather than done
opportunistically, and the two genuine divergences the audit surfaced
— console output and the dead CP interrupt — get explicit owners
instead of being lost in a large integration diff.

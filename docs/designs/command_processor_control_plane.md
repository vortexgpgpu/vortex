# Command Processor (CP) — Design

**Scope:** the Vortex Command Processor control plane — the hardware
([`hw/rtl/cp/`](../../hw/rtl/cp/)), its functional twin used by the
simulators ([`sim/common/cmd_processor.cpp`](../../sim/common/cmd_processor.cpp)),
and the async runtime that drives it
([`sw/runtime/include/vortex2.h`](../../sw/runtime/include/vortex2.h)).

Terminology used throughout this document:

- **RTL CP** — the SystemVerilog hardware in `hw/rtl/cp/*.sv`.
- **Emulation CP** (a.k.a. **Simulation CP**) — the C++ model
  `sim/common/cmd_processor.{h,cpp}`, instantiated and ticked by the
  simx / rtlsim / gem5 backends.

The CP is the single control plane through which the host submits work
to the GPU: memory transfers, DCR (device control register) programming,
kernel launch, fences, events, and cache maintenance. On FPGA targets
(XRT and OPAE) it is the **sole** launch/DCR path — the legacy AP_CTRL
launch FSM and per-AFU DCR machinery have been removed.

---

## 1. Architecture overview

```
   Host (runtime)                         Device
   ─────────────                          ──────
   vortex2.h enqueue                      ┌───────────────────────────────────────┐
        │                                 │             VX_cp_core                 │
   per-queue worker thread                │                                        │
        │ build 64B command line          │  AXI-Lite ─► VX_cp_axil_regfile        │
        ▼                                 │   (doorbell: Q_TAIL_LO/HI)             │
   host-pinned ring  ◄───── AR ────────── │      │                                 │
   (host_mem_alloc)                       │      ▼  per queue                       │
        │  doorbell (MMIO)                │  VX_cp_fetch ─► VX_cp_unpack ─► VX_cp_engine
        ▼                                 │      │ (ring walker)        (FSM, bids) │
   CP_REG Q_TAIL ──────────────────────► │      ▼                                 │
                                          │  4× VX_cp_arbiter (KMU/DMA/DCR/EVENT)  │
   seqnum poll  ◄──── AW/W (cmpl_addr) ── │      │                                 │
   (Q_SEQNUM / completion writeback)      │  ┌───┴────┬─────────┬──────────┐      │
                                          │  ▼        ▼         ▼          ▼      │
                                          │ launch  dcr_proxy  dma       event_unit│
                                          │ (KMU)   (DCR bus)  (host/dev) (counters)│
                                          │      │        │        │         │     │
                                          │      └────────┴── VX_cp_completion ◄───┘
                                          │           (retire → seqnum writeback)  │
                                          └───────────────────────────────────────┘
```

The CP is **N parallel command engines** (one per queue), feeding **four
round-robin arbiters** that serialize access to four shared resources:
the kernel-management unit (KMU launch), the DMA engine, the DCR bus, and
the event unit. The command ring lives in **host memory**; the CP fetches
from it over a dedicated AXI host port and writes completions back to host
memory over the same port.

Implemented in [`VX_cp_core.sv`](../../hw/rtl/cp/VX_cp_core.sv); the engine
count, ring size, and AXI ID width come from `VX_config.toml`'s `[cp]`
block with safe defaults in
[`VX_cp_pkg.sv:31-50`](../../hw/rtl/cp/VX_cp_pkg.sv#L31) (`NUM_QUEUES=1`,
`RING_SIZE_LOG2=16` → 64 KiB ring, `MAX_CMDS_PER_CL=5`, `AXI_TID_WIDTH=6`).

---

## 2. Command format

A command is a header plus an opcode-specific payload, packed into 64-byte
cache lines (up to 5 commands per line; a zero header terminates the line).
The decoded record is `cmd_t` in
[`VX_cp_pkg.sv:97-103`](../../hw/rtl/cp/VX_cp_pkg.sv#L97).

**Header** (4 B, [`VX_cp_pkg.sv:84-88`](../../hw/rtl/cp/VX_cp_pkg.sv#L84)):

```
{ reserved[15:0], flags[7:0], opcode[7:0] }
```

Flags: `F_PROFILE` (bit 0) appends an 8 B profile-slot trailer;
`F_FENCE_PRE` (bit 1) ([`VX_cp_pkg.sv:81-82`](../../hw/rtl/cp/VX_cp_pkg.sv#L81)).

**Opcodes** ([`VX_cp_pkg.sv:63-75`](../../hw/rtl/cp/VX_cp_pkg.sv#L63)) and
their on-wire sizes ([`cmd_size_bytes`, `VX_cp_pkg.sv:163-181`](../../hw/rtl/cp/VX_cp_pkg.sv#L163)):

| Opcode | Value | Size (B) | Purpose |
|---|---|---|---|
| `CMD_NOP` | 0x00 | 4 | Padding / ring alignment |
| `CMD_MEM_WRITE` | 0x01 | 28 | Host → device copy |
| `CMD_MEM_READ` | 0x02 | 28 | Device → host copy |
| `CMD_MEM_COPY` | 0x03 | 28 | Device → device copy |
| `CMD_DCR_WRITE` | 0x04 | 20 | Write a device control register |
| `CMD_DCR_READ` | 0x05 | 20 | Read a DCR (result in `Q_LAST_DCR_RSP`) |
| `CMD_LAUNCH` | 0x06 | 12 | Pulse KMU start, wait for drain |
| `CMD_FENCE` | 0x07 | 8 | Ordering barrier (see §7) |
| `CMD_EVENT_SIGNAL` | 0x08 | 20 | Write a counter slot |
| `CMD_EVENT_WAIT` | 0x09 | 28 | Spin until counter satisfies compare |
| `CMD_CACHE_FLUSH` | 0x0A | 12 | Per-core cache flush sweep |

`+8 B` to any size when `F_PROFILE` is set. `CMD_CACHE_FLUSH` is an
addition beyond the original proposal command set (AMD `ACQUIRE_MEM`-style
maintenance).

`CMD_EVENT_WAIT` carries a compare op in `arg2[1:0]`
(`WAIT_OP_EQ/GE/GT/NE`, [`VX_cp_pkg.sv:109-114`](../../hw/rtl/cp/VX_cp_pkg.sv#L109)).
`CMD_FENCE` carries scope masks in `arg0[1:0]` (`FENCE_DMA_BIT`,
`FENCE_GPU_BIT`, [`VX_cp_pkg.sv:120-121`](../../hw/rtl/cp/VX_cp_pkg.sv#L120)).

---

## 3. RTL CP module inventory

`hw/rtl/cp/` (~3500 LOC). Interfaces are split into separate `VX_cp_*_if.sv`
bundles per the project rule that `RTL_PKGS` carries only `*_pkg.sv`.

| Module | Role |
|---|---|
| [`VX_cp_pkg.sv`](../../hw/rtl/cp/VX_cp_pkg.sv) | Opcodes, header/`cmd_t`/`cpe_state_t` structs, resource enum, `cmd_size_bytes`. |
| [`VX_cp_core.sv`](../../hw/rtl/cp/VX_cp_core.sv) | Top level: regfile, N×(fetch+engine), 4 arbiters, the 5 resource units, dual AXI xbars, `cp_busy`/`irq` aggregation. |
| [`VX_cp_axil_regfile.sv`](../../hw/rtl/cp/VX_cp_axil_regfile.sv) | The only AXI-Lite slave. Global regs + per-queue blocks; atomic doorbell commit; caps registers. |
| [`VX_cp_fetch.sv`](../../hw/rtl/cp/VX_cp_fetch.sv) | Per-queue ring walker: ARs one 64 B cache line at `ring_base + (head & mask)`, single outstanding (no prefetch). |
| [`VX_cp_unpack.sv`](../../hw/rtl/cp/VX_cp_unpack.sv) | Per-offset single-command decoder (one command/cycle; refactored from a combinational whole-line walk for timing). |
| [`VX_cp_engine.sv`](../../hw/rtl/cp/VX_cp_engine.sv) | Per-queue FSM `IDLE→DECODE→BID→WAIT_DONE→RETIRE`; classifies opcode→resource and retires via valid/ready handshake. |
| [`VX_cp_arbiter.sv`](../../hw/rtl/cp/VX_cp_arbiter.sv) | Generic round-robin, instantiated 4× (one per resource). |
| [`VX_cp_launch.sv`](../../hw/rtl/cp/VX_cp_launch.sv) | KMU start/busy wrapper: pulse start, hold grant until `busy` deasserts (drain). |
| [`VX_cp_dcr_proxy.sv`](../../hw/rtl/cp/VX_cp_dcr_proxy.sv) | DCR req/rsp gateway; also drives the per-core `CMD_CACHE_FLUSH` sweep; publishes last read in `Q_LAST_DCR_RSP`. |
| [`VX_cp_dma.sv`](../../hw/rtl/cp/VX_cp_dma.sv) | Dual-port burst DMA (`axi_host` + `axi_dev`); opcode routes ports; ≤4 KB INCR chunks. |
| [`VX_cp_event_unit.sv`](../../hw/rtl/cp/VX_cp_event_unit.sv) | `EVENT_SIGNAL` write + `EVENT_WAIT` poll-spin against host/device counter slots. |
| [`VX_cp_completion.sv`](../../hw/rtl/cp/VX_cp_completion.sv) | Per-source 1-deep latch + shared drain FIFO with `retire_ready` backpressure; AXI write of 8 B seqnum to `cmpl_addr`. |
| [`VX_cp_axi_xbar.sv`](../../hw/rtl/cp/VX_cp_axi_xbar.sv) | N-source round-robin fan-in to one AXI master, TID-prefix response routing. |
| [`VX_cp_axi_to_membus.sv`](../../hw/rtl/cp/VX_cp_axi_to_membus.sv) | AXI4 → `VX_mem_bus_if` bridge for the OPAE AFU's req/rsp fabric. |
| [`VX_cp_profiling.sv`](../../hw/rtl/cp/VX_cp_profiling.sv) | Free-running cycle counter (skeleton; see §10). |
| [`VX_cp_dcr_proxy.sv`](../../hw/rtl/cp/VX_cp_dcr_proxy.sv), interfaces `VX_cp_axi_m_if`, `VX_cp_axil_s_if`, `VX_cp_gpu_if`, `VX_cp_engine_bid_if` | AXI master / AXI-Lite slave / GPU DCR+start-busy / CPE-bid bundles. |

Per-module Verilator unit tests live under `hw/unittest/cp_*`
(`cp_arbiter, cp_axil_regfile, cp_axi_path, cp_core, cp_dcr_proxy,
cp_dma, cp_engine, cp_launch, cp_unpack`).

### 3.1 Dual AXI topology

`VX_cp_core` builds **two** AXI crossbars
([`VX_cp_core.sv:386-431`](../../hw/rtl/cp/VX_cp_core.sv#L386)):

- **Host xbar** — fan-in of all fetch engines + completion + the DMA
  host port → one `axi_host` master that reaches host (pinned) memory.
- **Device xbar** — DMA device port + event unit → one `axi_dev` master
  that reaches device memory.

`VX_cp_dma` routes its two ports by opcode
([`VX_cp_dma.sv:95-96`](../../hw/rtl/cp/VX_cp_dma.sv#L95)):
`MEM_WRITE` = host→dev, `MEM_READ` = dev→host, `MEM_COPY` = dev→dev.

---

## 4. Per-queue engine and arbitration

Each queue has a `cpe_state_t`
([`VX_cp_pkg.sv:130-141`](../../hw/rtl/cp/VX_cp_pkg.sv#L130)): `ring_base`,
`ring_size_mask`, `head_addr`, `cmpl_addr`, `tail`, `head`, `seqnum`,
`prio`, `enabled`, `profile_en`.

`VX_cp_engine` runs `IDLE→DECODE→BID→WAIT_DONE→RETIRE`
([`VX_cp_engine.sv:126-185`](../../hw/rtl/cp/VX_cp_engine.sv#L126)). It
classifies each opcode to one of `RES_KMU/RES_DMA/RES_DCR/RES_EVT`
([`VX_cp_pkg.sv:149-154`](../../hw/rtl/cp/VX_cp_pkg.sv#L149)), raises a bid
to that resource's arbiter, waits for completion, then retires through
`VX_cp_completion` using a valid/ready handshake
([`VX_cp_engine.sv:173-181`](../../hw/rtl/cp/VX_cp_engine.sv#L173)) so no
seqnum is dropped when multiple engines retire in the same cycle. `NOP`
and `FENCE` retire immediately.

The four arbiters are plain round-robin. `VX_cp_arbiter` exposes a
`bid_priority` input but it is currently **unused**
([`VX_cp_arbiter.sv:89-94`](../../hw/rtl/cp/VX_cp_arbiter.sv#L89)) — see §10.

---

## 5. Register map (AXI-Lite)

`VX_cp_axil_regfile` is the sole AXI-Lite slave
([`VX_cp_axil_regfile.sv:12-44`](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L12)):

- **Globals** `0x000–0x024`: control, status, and read-only capability
  registers `GPU_DEV_CAPS` / `GPU_ISA_CAPS` (`0x018–0x024`), packed from
  config macros ([`VX_cp_axil_regfile.sv:121-139`](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L121)).
- **Per-queue block** at `0x100 + qid*0x40`: ring base/size, head/cmpl
  addresses, doorbell `Q_TAIL_LO/HI`, `Q_SEQNUM`, and `Q_LAST_DCR_RSP`
  (`+0x30`, [`:205`](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L205)).

The doorbell commits atomically on the `Q_TAIL_HI` write
([`:333-336`](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L333)); undecoded
addresses return DECERR.

> **FPGA/sim divergence:** the RTL regfile has **no** `CP_SATP_LO/HI`
> registers. The Emulation CP *does* (`0x028/0x02C`,
> [`cmd_processor.cpp:75-76`](../../sim/common/cmd_processor.cpp#L75)) and
> the runtime always writes them
> ([`device.cpp:303-304`](../../sw/runtime/common/device.cpp#L303)). See §8.

---

## 6. Capability registers

A single source of truth for device/ISA capabilities is exposed through
the CP and consumed identically by every backend (the
`caps_cp_consolidation` proposal, fully realized):

- RTL: `GPU_DEV_CAPS` / `GPU_ISA_CAPS` RO regs
  ([`VX_cp_axil_regfile.sv:121-139`](../../hw/rtl/cp/VX_cp_axil_regfile.sv#L121)).
- Emulation CP: same map
  ([`cmd_processor.cpp:30-49,126-129`](../../sim/common/cmd_processor.cpp#L30)).
- Runtime: one `decode_caps()` in
  [`sw/runtime/common/caps.h`](../../sw/runtime/common/caps.h), read over
  the CP regfile on every backend
  ([`device.cpp:275`](../../sw/runtime/common/device.cpp#L275)).

The duplicated capability blocks were removed from the XRT AFU shell.
(The OPAE AFU still carries some — see §10.)

---

## 7. Resource units

- **KMU launch** ([`VX_cp_launch.sv`](../../hw/rtl/cp/VX_cp_launch.sv)) —
  `IDLE→PULSE_START→WAIT_BUSY→WAIT_DRAIN`; holds the KMU grant until
  `busy` falls, so a queue serializes its own launches.
- **DCR proxy** ([`VX_cp_dcr_proxy.sv`](../../hw/rtl/cp/VX_cp_dcr_proxy.sv)) —
  drives DCR write/read on the GPU's DCR bus and runs the per-core
  `CMD_CACHE_FLUSH` sweep of `VX_DCR_BASE_CACHE_FLUSH`
  ([`:108-135`](../../hw/rtl/cp/VX_cp_dcr_proxy.sv#L108)).
- **DMA** ([`VX_cp_dma.sv`](../../hw/rtl/cp/VX_cp_dma.sv)) — dual-port
  multi-beat bursts in ≤4 KB INCR chunks
  ([`:81-178`](../../hw/rtl/cp/VX_cp_dma.sv#L81)).
- **Event unit** ([`VX_cp_event_unit.sv`](../../hw/rtl/cp/VX_cp_event_unit.sv)) —
  `EVENT_SIGNAL` writes a counter slot; `EVENT_WAIT` AR-polls until the
  compare passes ([`:124-129`](../../hw/rtl/cp/VX_cp_event_unit.sv#L124)).
- **Completion** ([`VX_cp_completion.sv`](../../hw/rtl/cp/VX_cp_completion.sv)) —
  collects retirements (per-source latch + shared FIFO with `retire_ready`
  backpressure) and writes the 8 B seqnum to the queue's `cmpl_addr`
  ([`:60-208`](../../hw/rtl/cp/VX_cp_completion.sv#L60)).

`irq` pulses on retire ([`VX_cp_core.sv:459-471`](../../hw/rtl/cp/VX_cp_core.sv#L459))
but is not yet wired to a host ISR (§10).

---

## 8. Emulation / Simulation CP

[`sim/common/cmd_processor.{h,cpp}`](../../sim/common/cmd_processor.cpp)
is a functional C++ twin (`vortex::CommandProcessor`): single-threaded,
one `tick()` per cycle, modelling a single queue (`q0_`). It is embedded
and ticked by the **simx, rtlsim, and gem5** backends inside their
`cp_reg_*` MMIO handlers (e.g.
[`sw/runtime/rtlsim/vortex.cpp:49,75-82`](../../sw/runtime/rtlsim/vortex.cpp#L49));
`host_mem_alloc` returns a `malloc` buffer whose pointer doubles as the
device-visible address.

The Emulation CP's MMIO map matches the RTL regfile **plus**:

- `CP_SATP_LO/HI` at `0x028/0x02C`
  ([`cmd_processor.cpp:75-76`](../../sim/common/cmd_processor.cpp#L75)),
- `CP_DEV_CAPS.VM_ENABLED` (bit 24,
  [`:117-122`](../../sim/common/cmd_processor.cpp#L117)),
- a software page-table walk `cp_translate` (Sv32/Sv39, megapage-aware,
  [`:157-201`](../../sim/common/cmd_processor.cpp#L157)) honoring the
  `MEM_FLAG_PHYSICAL` flag ([`:138,440-451`](../../sim/common/cmd_processor.cpp#L138)).

This means **CP DMA is MMU-aware in simulation but not on FPGA** — a
deliberate phased rollout (runtime + emulation first, RTL walker later;
see §10).

---

## 9. Runtime architecture

The runtime ([`sw/runtime/`](../../sw/runtime/)) is built around a
**`callbacks_t` dlopen dispatcher** with a minimal 6-function transport
HAL ([`sw/runtime/common/callbacks.h:45-74`](../../sw/runtime/common/callbacks.h#L45)):
`dev_open/close`, `cp_reg_read/write`, `host_mem_alloc/free`. Everything
above that — command encoding, queues, events, buffers, VM — is
backend-agnostic common code. This is the design from the
`cp_pure_v2_callbacks` addendum; it superseded the compile-time
`vx::Platform` model of `cp_runtime_impl_proposal`.

Layout:

- [`include/vortex2.h`](../../sw/runtime/include/vortex2.h) — the async
  API; [`include/vortex.h`](../../sw/runtime/include/vortex.h) — legacy
  API layered over it; helper headers `dxa.h`, `graphics.h`, `tensor.h`.
- [`common/`](../../sw/runtime/common/) — `device.cpp` (CP submit path),
  `queue.cpp` (per-queue worker threads + launch encoding), `event.cpp`,
  `buffer.cpp`, `module.cpp`, `vm.{h,cpp}`, `caps.h`, `callbacks.{h,inc}`,
  legacy wrappers, `vortex2_internal.h`.
- Backends: `simx/`, `rtlsim/`, `xrt/`, `opae/`, `gem5/`, and the `stub/`
  dispatcher.

**Submit path:** `Device::cp_init()`
([`device.cpp:233`](../../sw/runtime/common/device.cpp#L233)) programs each
queue's ring/head/cmpl addresses and sets `CP_REG_CTRL=0x1`. Each
`vx_enqueue_*` pushes a `Command` lambda onto a per-queue worker FIFO
([`queue.cpp:28,69`](../../sw/runtime/common/queue.cpp#L28)); the worker
waits on dependencies, encodes one 64 B cache line, memcpys it into the
host-pinned ring, rings the `Q_TAIL_LO/HI` doorbell, and busy-polls
`Q_SEQNUM` ([`device.cpp:334-398`](../../sw/runtime/common/device.cpp#L334)).
A kernel launch is currently encoded as ~18 `CMD_DCR_WRITE`s to KMU DCRs
followed by `CMD_LAUNCH`, then `CMD_CACHE_FLUSH` and a cout drain
([`queue.cpp:382-424`](../../sw/runtime/common/queue.cpp#L382)).

> **vortex2.h vs the original "minimal" spec:** the shipped header is
> richer than the 34-function minimalist proposal — it keeps first-class
> `vx_module_h` and `vx_kernel_h` handles
> ([`vortex2.h:53-54`](../../sw/runtime/include/vortex2.h#L53)) plus
> rect-copy, fill-buffer, perf-dump, and max-occupancy helpers. This is
> the accepted current shape.

### 9.1 FPGA backends

- **XRT** ([`hw/rtl/afu/xrt/VX_afu_wrap.sv`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv)) —
  AXI-Lite is split on bit 12: `0x000–0x0FFF` → a minimal `VX_afu_ctrl`
  ap_ctrl stub + SCOPE; `0x1000–0x1FFF` → the CP regfile. The legacy
  launch FSM, DCR path, and dev-caps were **removed**; CP is the sole
  launch/DCR path (`vx_start = cp_gpu_if.start`,
  [`:342-353`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L342)). A dedicated
  `m_axi_host` port carries the ring
  ([`:305-316`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L305)); `axi_dev`
  joins memory bank 0 via `VX_axi_arb2`
  ([`:533-546`](../../hw/rtl/afu/xrt/VX_afu_wrap.sv#L533)).
- **OPAE** ([`hw/rtl/afu/opae/vortex_afu.sv`](../../hw/rtl/afu/opae/vortex_afu.sv)) —
  `VX_cp_core` instantiated at [`:328`](../../hw/rtl/afu/opae/vortex_afu.sv#L328);
  `axi_host`→CCI-P and `axi_dev`→local memory both bridged through
  `VX_cp_axi_to_membus` ([`:372,530`](../../hw/rtl/afu/opae/vortex_afu.sv#L372));
  MMIO word-address bit-10 demux (the 0x1000 byte boundary) to the CP
  regfile. The runtime
  ([`sw/runtime/opae/vortex.cpp`](../../sw/runtime/opae/vortex.cpp))
  implements `cp_reg_*` at `CP_BASE=0x1000` and `host_mem_alloc`.

---

## 10. Proposed but not yet implemented

The following were specified across the source proposals (notably
`cp_v3_critical_review`) and remain **open**. They are recorded here so
the intent is not lost.

**Correctness gaps (RTL ↔ emulation divergence):**

1. **Byte-exact DMA in RTL.** `VX_cp_dma` rounds transfer size up to a
   64 B multiple (`rem_beats=(arg2+63)>>6`, `wstrb='1'`,
   [`VX_cp_dma.sv:119`](../../hw/rtl/cp/VX_cp_dma.sv#L119)) while the
   Emulation CP is byte-exact
   ([`cmd_processor.cpp:452-462`](../../sim/common/cmd_processor.cpp#L452)).
   Non-cache-line-aligned transfers can over-write on FPGA. Needs tail
   `wstrb` on the last beat (review item C-2/P-W4).
2. **VM in RTL.** Add `CP_SATP_LO/HI` regfile decode + a hardware
   page-table walker + TLB in `VX_cp_dma`, and route `F_MEM_PHYSICAL`, so
   FPGA matches the simulator's MMU-aware DMA (cp_pure_v2 VM Phase 2;
   review items P-W1/P-W3 and the SATP gap). Today VM works on
   simx/rtlsim/gem5 and silently no-ops on FPGA.
3. **Real `CMD_FENCE` semantics.** The engine retires FENCE as a NOP
   ([`VX_cp_engine.sv:109-112`](../../hw/rtl/cp/VX_cp_engine.sv#L109));
   it should honor `FENCE_DMA_BIT` / `FENCE_GPU_BIT` ordering (C-7).
4. **`dcr_req_ready` backpressure** in `VX_cp_dcr_proxy` — currently
   assumes the DCR bus always accepts
   ([`VX_cp_dcr_proxy.sv:118-125`](../../hw/rtl/cp/VX_cp_dcr_proxy.sv#L118)) (C-3).

**Performance / feature gaps:**

5. **QMD-style atomic `CMD_LAUNCH`.** Replace the ~18-`CMD_DCR_WRITE`
   launch dance with grid/block/PC/args inline in the command
   ([`queue.cpp:382-414`](../../sw/runtime/common/queue.cpp#L382)). This
   is the precondition for true per-queue launch concurrency (A-1/E-3).
6. **Multi-queue everywhere.** RTL is parameterized on `NUM_QUEUES` but
   defaults to 1; the Emulation CP models only `q0_`; the runtime
   serializes launches. Real concurrency depends on item 5.
7. **EVENT_WAIT fairness + backoff.** `EVENT_WAIT` holds its arbiter
   grant for the whole wait
   ([`VX_cp_event_unit.sv:124-129`](../../hw/rtl/cp/VX_cp_event_unit.sv#L124));
   it should release between polls and back off (C-4).
8. **Priority arbitration.** Wire `VX_cp_arbiter.bid_priority` (currently
   unused, [`VX_cp_arbiter.sv:89-94`](../../hw/rtl/cp/VX_cp_arbiter.sv#L89))
   to the per-queue `prio` field.
9. **Profiling writeback.** `VX_cp_profiling` is a bare cycle counter;
   flesh it out to emit the 32 B `{queued,submit,start,end}` record, add a
   `CP_CYCLE_FREQ_HZ` register and the `0x040` MMIO block, so
   `vx_event_get_profiling` (already in the header) has hardware backing.
10. **Interrupt path.** Add IP_ISR/IER/GIER in the AFU and
    `xrt::ip::interrupt` so the runtime sleeps instead of busy-polling
    `Q_SEQNUM` (the `irq` pulse already exists,
    [`VX_cp_core.sv:459`](../../hw/rtl/cp/VX_cp_core.sv#L459)).
11. **Host-coherent completion mailbox / `head_addr` writeback.** The CP
    tracks `head` internally and never DMAs it to `head_addr`; a cacheable
    host mailbox would avoid per-poll MMIO reads (E-10).
12. **`dcr_cp` 0x080–0x0BF DCR reservation** in `VX_types.toml` for
    future CP↔GPU coordination / multi-context KMU (never added).
13. **OPAE caps de-duplication.** The OPAE AFU still carries capability
    blocks that the XRT AFU has already delegated to the CP regfile.

**Abandoned / superseded directions** (recorded so they are not
revived by accident): the device-memory ring + per-command DMA of the
original `command_processor_proposal` (replaced by the host-memory ring);
the `DEDICATED`/`SHARED` DMA build toggle (replaced by always-on dual
`axi_host`/`axi_dev`); the AP_CTRL legacy compat mode and `VORTEX_USE_CP`
opt-in (legacy path removed outright); the compile-time `vx::Platform`
runtime model and file-rename layout of `cp_runtime_impl_proposal`
(replaced by the `callbacks_t` HAL); and the "own runtime"
(LitePCIe/VFIO/Coyote) and extra per-card adapters (U250/U280/U55C/V80/
KV260) of `command_processor_redesign` — XRT and OPAE are the only live
hardware paths.

---

## 11. Source proposals

This design consolidates and supersedes the following proposals
(now removed from `docs/proposals/`): `command_processor_proposal.md`,
`command_processor_redesign.md`, `caps_cp_consolidation_proposal.md`,
`cp_rtl_impl_proposal.md`, `cp_runtime_impl_proposal.md`,
`cp_v3_critical_review.md`, `cp_pure_v2_callbacks_proposal.md`,
`cp_xrt_integration_plan.md`, `cp_opae_integration_plan.md`.

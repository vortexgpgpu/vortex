# Vortex Command Processor — Design

**Status:** as-built (`feature_cp` branch).
**Replaces:** all earlier per-phase CP proposals (`command_processor_proposal.md`,
`cp_rtl_impl_proposal.md`, `cp_runtime_impl_proposal.md`,
`cp_xrt_integration_plan.md`, `cp_opae_integration_plan.md`,
`cp_pure_v2_callbacks_proposal.md`).

---

## 1. Summary

The Vortex runtime used to drive the FPGA in lock-step over MMIO: every
`vx_dcr_write`, `vx_start`, `vx_ready_wait` was a synchronous transaction.
There was no way for the host to queue ahead, overlap DMA with kernel
execution, or express cross-operation dependencies.

The Command Processor (CP) introduces an asynchronous, multi-queue,
event-based submission model that maps cleanly onto OpenCL command queues,
CUDA streams, and SYCL queues. Three layers:

1. A **platform-agnostic CP block** (`hw/rtl/cp/`) that talks to the GPU
   through DCR + KMU and to the host through one canonical AXI4 master +
   AXI4-Lite slave pair.
2. **Thin per-platform AFU shims** (`hw/rtl/afu/xrt/`, `hw/rtl/afu/opae/`)
   that adapt the platform shell to that canonical interface, plus a
   **software CP** (`sim/common/CommandProcessor.{h,cpp}`) that satisfies
   the same interface for simx and rtlsim so all four backends look
   identical from above.
3. A **new runtime layer** (`vortex2.h`) exposing refcounted
   `vx_queue_h` + `vx_event_h` with in-order async semantics, with the
   legacy `vortex.h` becoming a thin wrapper over it. A unified dispatcher
   (`sw/runtime/stub/`) owns all CP protocol; backends expose only
   platform primitives through a 9-field `callbacks_t`.

---

## 2. Goals and non-goals

### Goals

- Make Vortex a conformant OpenCL 1.2 execution backend at the
  hardware/runtime layer: asynchronous enqueue, in-order command queues,
  events with cross-queue dependencies, user events, markers/barriers,
  `CL_QUEUE_PROFILING_ENABLE` timestamps.
- Decouple the CP from the platform shell. CP code lives in `rtl/cp/`
  with one canonical AXI interface; vendor shims are minimal.
- Support multiple general-purpose hardware queues. Each is an in-order
  command stream driven by its own per-queue **Command Processor Engine
  (CPE)**. CPEs converge on shared GPU resources (KMU, DMA, DCR bus)
  through round-robin arbiters.
- Achieve concurrent submission + zero-bubble kernel succession: while
  kernel A is draining through the KMU, queue B's CPE can fetch
  commands, run DMAs, evaluate event-waits, and pre-stage kernel B's
  KMU descriptor so the next launch starts the cycle KMU goes idle.
- Host/device synchronization primitives: host events, intra-queue
  waits, cross-queue semaphores, host-signalled semaphores.
- Per-command profiling timestamps written back to host memory.
- Asynchronous DMA (both directions) and asynchronous kernel launch.
- Unified backend ABI: the runtime dispatcher contains 100% of the CP
  wire protocol; backends expose only platform primitives.

### Non-goals (v1)

- **True per-CTA concurrent kernel execution.** v1 has a single-context
  KMU, so CTAs from two different kernels are never simultaneously in
  flight. v1 ships *concurrent submission + zero-bubble kernel
  succession* instead, which captures the practical CKE win
  (cross-queue DMA/compute overlap, fast kernel-to-kernel switching)
  and is sufficient for conformant OpenCL 1.2. The architecture is
  forward-compatible with a multi-context KMU.
- Hardware out-of-order command queues. The runtime emulates OoO by
  spawning multiple in-order HW queues plus events.
- Preemption, priority inversion, mid-kernel context switch.
- Multi-device. One CP serves one Vortex instance.
- MSI-X / kernel-driver interrupts. Completion is host-polled in v1.

---

## 3. Terminology

| Term | Meaning |
|---|---|
| **Command Processor (CP)** | RTL block under `rtl/cp/` that owns N CPEs plus the shared arbiters, DMA, event unit, and platform interface. |
| **Command Processor Engine (CPE)** | Per-queue engine inside the CP. Fetches the queue's commands, decodes them, drives the per-command FSM, and bids for shared resources. |
| **Queue (`vx_queue_h`)** | An in-order channel from the host to one CPE. Owns a ring buffer and a 64-bit seqnum space. |
| **Event (`vx_event_h`)** | A 64-bit seqnum on some queue (or a host-signalled value) usable in waits. |
| **Completion seqnum** | Per-queue monotonic counter the CP writes to a host-visible memory location after each command retires. |
| **Resource arbiter** | Round-robin arbiter that picks which CPE next gets a shared resource (KMU launch port, DMA, DCR proxy). One per resource. |
| **AFU shim** | Per-platform adapter under `rtl/afu/{xrt,opae}/` that exposes the CP's canonical AXI ports as the platform's native shell. |
| **Software CP** | C++ functional model (`sim/common/CommandProcessor`) used by simx and rtlsim, which have no hardware CP. Mirrors the regfile + engine + launch FSM behavior. |
| **Dispatcher** | The shared library (`libvortex.so`, built from `sw/runtime/stub/`) that implements vortex2.h on top of the backend's platform primitives. Owns 100% of the CP wire protocol. |

---

## 4. High-level architecture

```
   ┌──────────────────── HOST ─────────────────────────────────────┐
   │  application                                                  │
   │     │                                                         │
   │     ▼                                                         │
   │  vortex2.h API   (vx_device / vx_queue / vx_event / vx_buffer)│
   │     │                                                         │
   │     ▼                                                         │
   │  Dispatcher  (libvortex.so — sw/runtime/stub/)                │
   │     │  builds CMD_* descriptors, mem_uploads them into the    │
   │     │  per-queue ring, commits Q_TAIL via cp_mmio_write,      │
   │     │  polls Q_SEQNUM via cp_mmio_read                        │
   │     ▼                                                         │
   │  callbacks_t   (9-field platform primitives ABI)              │
   │     │                                                         │
   │     ▼                                                         │
   │  Backend lib   (libvortex-{simx,rtlsim,xrt,opae}.so)          │
   └─────────────────┬──────────────────────────┬──────────────────┘
                     │ AXI4 master              │ AXI4-Lite slave
                     │ (mem_upload to ring)     │ (cp_mmio_write/read)
                     ▼                          ▼
   ┌─────────────────── Platform shell / AFU ──────────────────────┐
   │  xrt / opae:  hardware CP regfile + ring fetch via VX_cp_core │
   │  simx / rtlsim: software CommandProcessor C++ class           │
   └─────────────────┬──────────────────────────┬──────────────────┘
                     │ DCR req/rsp              │ start / busy
                     ▼                          ▼
                            Vortex.sv (GPU core)
                       (single-context KMU; consumes DCRs,
                        launches one kernel's CTAs at a time)
```

The CP is one block with:

- **N parallel CPEs** (one per HW queue). Each owns its own ring-buffer
  state, FSM, and seqnum counter, independent of the others.
- **Resource arbiters** that round-robin between CPEs for each shared
  resource. A CPE blocked on one resource does not prevent another CPE
  making progress on a different one — this is the source of
  cross-queue overlap.
- One **upstream AXI master** for command fetch, DMA, completion
  writeback, and profile-timestamp writeback, multiplexed via
  `VX_cp_axi_xbar`.
- One **AXI4-Lite slave** for the host to write doorbells and read
  CP status / completion seqnums.
- One **DCR master interface** down into the GPU (request + response).
- One **start/busy** handshake to the single-context KMU.

The single-context KMU is the serialization point for kernel launches:
at any instant only one kernel's CTA grid is being emitted. CPEs not
currently holding the KMU arbiter are free to do everything else
(fetch, decode, DMA, event waits, DCR programming for their *next*
launch). This is what "concurrent submission + zero-bubble kernel
succession" means.

The platform shim's job is only to splice the CP's AXI master/slave
into the shell's AXI infrastructure. The XRT shim is near-trivial
(`Vortex_axi.sv` is already AXI). OPAE needs a small CCIP-MMIO →
AXI-Lite shim and an AXI4 → `VX_mem_bus_if` bridge for local memory.
simx and rtlsim use a software `CommandProcessor` C++ class in lieu of
an RTL CP — same regfile surface, same engine semantics.

### Why AXI as the canonical CP interface

- Vortex's XRT path is already AXI; zero adaptation needed for v1.
- Modern Intel OFS shells expose AXI to the AFU; reviving OPAE means
  writing one PIM-based shim, not a CCI-P bridge plus all the rest.
- Universal vendor and IP support; future-proofs Versal/chiplet/non-FPGA
  retargets.
- Rich verification ecosystem (BFMs, VIP, formal kits).
- Clean separation of control plane (AXI-Lite) from data plane (AXI4).

---

## 5. Hardware design

### 5.1 Source tree

```
hw/rtl/cp/
├── VX_cp_pkg.sv               command opcodes, struct typedefs, parameters
├── VX_cp_if.sv                SV interface bundles (CPE↔arbiters, CP↔Vortex gpu_if)
├── VX_cp_axi_m_if.sv          AXI4 master bundle (CP-internal)
├── VX_cp_axil_s_if.sv         AXI4-Lite slave bundle (CP-internal)
├── VX_cp_core.sv              top-level CP wrapper; instantiates everything below
├── VX_cp_axil_regfile.sv      host-facing AXI-Lite register block (§5.6)
├── VX_cp_engine.sv            one CPE (per HW queue) — decode/bid/retire FSM
├── VX_cp_fetch.sv             AXI master read of next command CL (one per CPE)
├── VX_cp_unpack.sv            cache-line → packed cmd_t stream (≤5 cmds/CL)
├── VX_cp_arbiter.sv           generic round-robin arbiter (3× instances)
├── VX_cp_launch.sv            KMU start/busy handshake wrapper (KMU resource)
├── VX_cp_dcr_proxy.sv         DCR req/rsp into Vortex (DCR resource)
├── VX_cp_dma.sv               AXI ↔ Vortex memory DMA engine (DMA resource)
├── VX_cp_completion.sv        per-queue seqnum + head writeback to host
├── VX_cp_axi_xbar.sv          N→1 AXI master mux for CPEs + DMA + completion
├── VX_cp_event_unit.sv        (skeleton) wait-on-seqnum comparator
└── VX_cp_profiling.sv         (skeleton) per-cmd timestamp writeback

hw/rtl/afu/
├── xrt/   (VX_afu_wrap.sv, VX_afu_ctrl.sv)
└── opae/  (vortex_afu.sv)

hw/rtl/libs/
├── VX_axi_arb2.sv             2:1 AXI4 arbiter used at XRT bank 0
└── VX_cp_axi_to_membus.sv     AXI4 master → VX_mem_bus_if bridge (OPAE)

sim/common/
└── CommandProcessor.{h,cpp}   software CP for simx/rtlsim
```

There is no separate "queue manager." Each CPE manages exactly one
queue; the arbiters live on the *resource* side, not the queue side.

### 5.2 Queue model and CPE state

Each queue is identified by `qid` ∈ `[0, NUM_QUEUES)`. `NUM_QUEUES` is
a compile-time parameter (default 1; the architecture scales). There is
exactly one CPE per queue — an in-order queue has no internal
parallelism, so >1 CPE per queue is pointless; <1 would reintroduce
the head-of-line blocking the design avoids.

Each queue owns:

- A host-allocated, page-aligned ring buffer with power-of-two byte
  capacity (`Q_RING_SIZE_LOG2`, default 16 = 64 KiB).
- A host-published `tail` (producer pointer) and CP-published `head`
  (consumer pointer), both 64-bit byte offsets.
- A completion-seqnum slot in host memory; CP writes the most recent
  retired seqnum after each retirement.
- A 64-bit seqnum counter inside the owning CPE.

Per-CPE programmable state (mirrored into the regfile):

```systemverilog
typedef struct packed {
  logic [63:0] ring_base;        // device address of ring buffer
  logic [VX_CP_RING_SIZE_LOG2_C-1:0] ring_size_mask;
  logic [63:0] head_addr;        // device address where CPE publishes head
  logic [63:0] cmpl_addr;        // device address where CPE publishes seqnum
  logic [63:0] tail;             // host's committed tail
  logic [63:0] head;             // CPE-internal consumer pointer
  logic [63:0] seqnum;           // next-to-retire seqnum
  logic [1:0]  prio;             // 0=lo … 3=hi (priority hint to arbiter)
  logic        enabled;          // = CP_CTRL.enable_global & Q_CONTROL.enable
  logic        profile_en;
} cpe_state_t;
```

### 5.3 Command set

Every command carries a 4-byte header `{opcode[7:0], flags[7:0],
reserved[15:0]}` followed by opcode-specific payload. **Cache-line
framing rule:** a command never crosses a 64 B boundary; the rest of
the line is zero-padded. The unpacker (`VX_cp_unpack`) walks one CL
extracting up to 5 commands, stopping on a zero header (= padding
sentinel).

Header flag bits:

| Bit | Name | Meaning |
|---|---|---|
| `flags[0]` | `F_PROFILE` | Command is profiled. Payload is followed by an 8 B `profile_slot` host address; CP writes 4×8 B timestamps there at retirement. |
| `flags[1]` | `F_FENCE_PRE` | Treat as if `CMD_FENCE(FENCE_ALL)` was inserted immediately before this command. |

Opcodes:

| Opcode | Size | Payload | Purpose |
|---|---|---|---|
| `CMD_NOP` | 4 B | — | padding / pacing |
| `CMD_MEM_WRITE` | 28 B | host_addr, dev_addr, size | host→device DMA |
| `CMD_MEM_READ` | 28 B | host_addr, dev_addr, size | device→host DMA |
| `CMD_MEM_COPY` | 28 B | src_dev, dst_dev, size | device→device DMA |
| `CMD_DCR_WRITE` | 20 B | dcr_addr, dcr_value | program GPU/KMU DCR |
| `CMD_DCR_READ` | 20 B | dcr_addr, tag | read GPU DCR; response in `Q_LAST_DCR_RSP` regfile slot |
| `CMD_LAUNCH` | 12 B | (arg0 reserved) | pulse KMU `start`; assumes KMU is preprogrammed via prior `CMD_DCR_WRITE`s |
| `CMD_FENCE` | 8 B | mask | retirement barrier within this queue |
| `CMD_EVENT_SIGNAL` | 20 B | event_addr, value | write 64 b to a host-visible event slot |
| `CMD_EVENT_WAIT` | 28 B | event_addr, value, op | stall queue until `*event_addr op value` is true |

Notes:

- `CMD_LAUNCH` does **not** reset the GPU. The runtime is responsible
  for emitting `CMD_DCR_WRITE`s into the same queue ahead of
  `CMD_LAUNCH` to configure the KMU (PC, args, grid/block dims, lmem,
  warp step — see `hw/rtl/VX_kmu.sv`).
- `CMD_EVENT_WAIT` is the building block for intra-queue waits and
  cross-queue semaphores: an event slot is just a 64-bit host-memory
  address, and "another queue" means that address is the other queue's
  completion-seqnum slot.

### 5.4 CPE FSM (`VX_cp_engine`)

```
S_IDLE     → fetch CL when head < tail, hand off cmds one at a time
S_DECODE   → classify opcode → KMU / DMA / DCR / skip
S_BID      → assert bid line for the chosen resource arbiter
S_WAIT_DONE → wait for the resource's done pulse
S_RETIRE   → pulse retire_evt + advance seqnum → S_IDLE
```

`S_WAIT_DONE` gates on the resource's **actual** `done` pulse — not on
arbiter grant. This is the v1.1 fix; the original Phase 2b shortcut
that retired on grant raced the resource modules' multi-cycle pipelines
and silently dropped grants on back-to-back commands of the same type.

### 5.5 Resource arbiters

Because each queue has its own CPE, there is no central queue arbiter
choosing "which queue runs next." Instead, each shared resource has
its own round-robin arbiter that decides "which CPE gets me this
cycle":

| Arbiter | Resource gated | When a CPE bids |
|---|---|---|
| **KMU** | `VX_cp_launch` (start pulse + busy observation) | CPE has a `CMD_LAUNCH` decoded |
| **DMA** | `VX_cp_dma` | CPE has a `CMD_MEM_*` decoded |
| **DCR** | `VX_cp_dcr_proxy` | CPE has a `CMD_DCR_*` decoded |

Properties:

- Each arbiter is independent. A CPE blocked on KMU does not prevent
  another CPE from getting DMA or DCR the same cycle.
- Round-robin in v1. Priority is supported via the per-CPE `prio`
  field (configurable; off by default for fairness).
- KMU arbitration **holds** for the entire duration of a launch
  (from `start` pulse until `busy` falls): the single-context KMU
  cannot accept a new descriptor mid-grid. The CPE releases KMU the
  cycle it retires its `CMD_LAUNCH`; the next-winning CPE may
  immediately program its descriptor's DCRs and pulse `start` — zero
  bubble.
- DMA and DCR arbitration are per-transaction (release after each
  command). Long DMAs do not starve DCR programming.

This structure is forward-compatible with a multi-context KMU: the
KMU arbiter would select a *slot* in the KMU rather than a single
shared port; nothing else changes.

### 5.6 AXI-Lite regfile (`VX_cp_axil_regfile`)

CP-internal regfile address map (16-bit). xrt/opae backends add
`0x1000` to translate to host MMIO byte addresses (per the AFU's
bit-12 demux split, §6).

```
─ Globals (0x000..0x0FF) ──────────────────────────────────────────────
0x000  CP_CTRL          RW  bit0=enable_global, bit1=reset_all
0x004  CP_STATUS        RO  bit0=busy, bit1=error
0x008  CP_DEV_CAPS      RO  {AXI_TID_W:8 | RING_SIZE_LOG2:8 | NUM_QUEUES:8}
0x010  CP_CYCLE_LO/HI   RO  free-running 64-bit cycle counter

─ Per-queue (base = 0x100 + qid*0x40) ─────────────────────────────────
+0x00 Q_RING_BASE_LO/HI   RW
+0x08 Q_HEAD_ADDR_LO/HI   RW  device address where CPE publishes head
+0x10 Q_CMPL_ADDR_LO/HI   RW  device address where CPE publishes seqnum
+0x18 Q_RING_SIZE_LOG2    RW  (mask derived: (1<<value) - 1)
+0x1C Q_CONTROL           RW  bit0=enable, bit1=reset, [3:2]=prio, bit4=profile_en
+0x20 Q_TAIL_LO           WO  staging
+0x24 Q_TAIL_HI           WO  staging + atomic commit pulse
+0x28 Q_SEQNUM            RO  latest retired seqnum (mirrors cmpl slot)
+0x2C Q_ERROR             RO  per-queue error word
+0x30 Q_LAST_DCR_RSP      RO  most recent CMD_DCR_READ response
```

**Atomic-tail rule:** the host writes `Q_TAIL_LO` into a staging
register without advancing `tail`, then writes `Q_TAIL_HI` which both
latches the high half AND commits the full 64-bit `{HI, LO}` value into
`q_state.tail` in the same cycle. A host that writes only `Q_TAIL_LO`
does not advance the queue. This removes any dependency on AXI-Lite
ordering across the interconnect.

### 5.7 DCR bus extended to req/rsp

`Vortex.sv` exposes DCR as request + response (formerly write-only at
the top level). Changes:

- `Vortex.sv` and `Vortex_axi.sv` expose `dcr_rsp_valid`, `dcr_rsp_data`.
- `VX_cp_dcr_proxy` issues both reads and writes. For `CMD_DCR_READ` it
  latches the response into `last_rsp_data`, which the regfile exposes
  at `Q_LAST_DCR_RSP` for the host to poll after `Q_SEQNUM` advances.

The proxy latches the full request payload (addr + data + is_read) on
arbiter grant. Driving the DCR bus combinationally from `cmd` would
sample zeros after grant (the upstream `granted_dcr_cmd` mux in
`VX_cp_core` is gated on the grant cycle).

### 5.8 Profiling

A free-running 64-bit cycle counter (`CP_CYCLE_LO/HI`) is exposed via
the AXI-Lite block. The runtime reads `CP_CYCLE_FREQ_HZ` once at
device open and converts cycle timestamps to nanoseconds for OpenCL.

A profiled command (`F_PROFILE` flag set) is followed in the ring by
an 8 B `profile_slot` host address. The CPE samples the cycle counter
at four points: QUEUED (host-side, before doorbell), SUBMIT (CL
fetched into unpacker), START (resource arbiter grants the resource),
END (command retires). `VX_cp_profiling` pushes a 32 B record
`{QUEUED, SUBMIT, START, END}` to `profile_slot` via the AXI master.

`VX_cp_event_unit` and `VX_cp_profiling` are present as RTL skeletons
in v1; the engine retires `CMD_EVENT_*` and profile-flagged commands
as NOPs today. Full wiring is forward work.

### 5.9 DMA engine

`VX_cp_dma` is a generic DMA engine: source/dest address + size, both
endpoints expressible as either the CP's AXI master (host memory) or
the Vortex memory subsystem (device memory). For `CMD_MEM_COPY` both
endpoints are device.

For device-side accesses the CP can either share the Vortex memory
fabric (`SHARED` mode, v1 default — works on every XRT shell) or use
a dedicated Vortex memory port (`DEDICATED` mode, opt-in on multi-bank
shells where contention measurably hurts throughput).

### 5.10 Completion ordering and fences

Within a queue, commands retire in submission order. Across queues,
ordering is the user's job via events. `CMD_FENCE` enforces stronger
guarantees within a queue:

- `FENCE_DMA`: wait until all prior DMAs on this queue have drained.
- `FENCE_GPU`: wait until `vx_busy == 0` (KMU/launch fully drained).
- `FENCE_ALL`: both.

The runtime emits `CMD_FENCE(FENCE_GPU)` automatically before any
`CMD_MEM_READ` that targets memory written by a recent `CMD_LAUNCH`
on the same queue, so `vx_buffer_read` after `vx_enqueue_launch` is
safe by default.

---

## 6. Platform integration

The CP boundary is exposed to the platform shim via four signals:

- One AXI4-Lite slave port for host control (regfile reads/writes).
- One AXI4 master port for command fetch, DMA, completion writeback.
- One `VX_cp_gpu_if` bundle to Vortex (DCR req/rsp, KMU start/busy).
- One interrupt output (tied low in v1).

The shim's job is to splice these into the platform's native shell.

### 6.1 XRT AFU

`hw/rtl/afu/xrt/VX_afu_wrap.sv`:

- **AXI-Lite demux:** host byte addresses `0x0000..0x0FFF` go to legacy
  `VX_afu_ctrl` (8-bit AP_CTRL register block — kept for non-CP debug
  hatches and for SCOPE). Bit 12 of the host address (`0x1000..0x1FFF`)
  selects the CP regfile, mapped to CP's native 0x000-based space. CP
  receives `addr - 0x1000`.
- **`gpu_if` mux:** CP's `dcr_req_*` and the legacy AFU_ctrl's
  `lg_dcr_req_*` are OR-combined into Vortex's DCR input (CP-wins on
  simultaneous valid). Same for `vx_start`. `cp_gpu_if.busy` is wired
  to Vortex's `busy`. CP's `dcr_req_ready` is tied high (Vortex DCR
  always accepts).
- **Bank-0 AXI arbiter:** Vortex's bank-0 AXI master and the CP's
  `axi_m` share output bank 0 via `VX_axi_arb2` (a 2:1 AXI arbiter
  with sticky owner per channel until response completes). Banks
  `1..N-1` are direct passthrough from Vortex.
- **AFU FSM auto-advance:** the legacy outer FSM (`STATE_IDLE` →
  `STATE_RUN` → `STATE_DONE`) now also enters `STATE_RUN` on
  `cp_gpu_if.start`, with a `saw_busy` guard so `STATE_DONE` only
  fires after `vx_busy` has actually risen and fallen.

### 6.2 OPAE AFU

`hw/rtl/afu/opae/vortex_afu.sv`:

- **CCIP MMIO → AXI-Lite shim** (inline): CCIP MMIO addresses are
  4-byte-indexed, so the bit-12 host-byte split surfaces as
  `mmio_req_hdr.address[10]`. Writes/reads in the CP range are
  forwarded to a `VX_cp_axil_s_if` slave. CP reads are latched into
  a separate response register, muxed onto the CCIP c2 channel.
- **`gpu_if` mux + `saw_busy` guard:** same pattern as XRT.
- **3-way memory arbiter:** the existing `cci_vx_mem_arb_in_if[2]`
  merging Vortex memory + CCIP DMA is extended to 3 slots. CP's
  `axi_m` is bridged to `VX_mem_bus_if` (OPAE memory is
  request/response style, not AXI4) via a new
  `VX_cp_axi_to_membus.sv` helper. `AVS_TAG_WIDTH` grows by one bit
  to fit the extra arbiter index.

### 6.3 simx and rtlsim — software CP

simx and rtlsim have no hardware AFU around Vortex. To present the
same `cp_mmio_write/read` ABI as xrt/opae, they instantiate a software
`vortex::CommandProcessor` (`sim/common/CommandProcessor.{h,cpp}`):

```cpp
class CommandProcessor {
public:
    struct Hooks {
        std::function<void(uint64_t, void*,       size_t)> dram_read;
        std::function<void(uint64_t, const void*, size_t)> dram_write;
        std::function<void(uint32_t, uint32_t)>            vortex_dcr_write;
        std::function<uint32_t(uint32_t, uint32_t)>        vortex_dcr_read;
        std::function<void()>                              vortex_start;
        std::function<bool()>                              vortex_busy;
    };
    explicit CommandProcessor(const Hooks&);
    void     mmio_write(uint32_t off, uint32_t value);
    uint32_t mmio_read (uint32_t off) const;
    void     tick();
};
```

**Single-threaded `tick()` model**, not a worker thread. Justification:

| Concern | tick() per host MMIO | Separate CP thread |
|---|---|---|
| Determinism | Reproducible — each MMIO advances the same number of cycles | Race against `Processor::run()` → ordering of memory + DCR accesses depends on scheduler |
| simx fit | simx is *functional* sim built for fast, deterministic test runs | Mutexes on RAM/DCR kill the fast path |
| rtlsim/Verilator | `eval()` is single-threaded by default | Concurrent thread races `eval()` |
| Debugging | Linear execution, `gdb` step works | Race conditions need TSAN |
| Realism | Matches the hardware — CP is a synchronous FSM on the same clock as Vortex | Doesn't model hardware better; adds artificial concurrency |

Each backend wires the hooks to its local `Processor` (which is Verilator
in rtlsim, the SimX C++ functional core in simx) and bounds the
tick budget per `cp_mmio_*` call so polling drives the CP forward
without an explicit drain loop.

The software CP doubles as a **reference implementation**: the
`feature_cp` debug story for the hardware CP was "run vecadd on simx
and xrt with per-command stderr trace, diff outputs, the wrong one is
the bug." That diff localized a one-line combinational vs registered
bug in `VX_cp_dcr_proxy` in a single cycle.

---

## 7. Runtime

### 7.1 The vortex2.h surface

`sw/runtime/include/vortex2.h` is the minimal async runtime surface for
Vortex. Six families:

- **Devices** — `vx_device_open/release/retain`, `vx_device_query`,
  `vx_device_memory_info`.
- **Buffers** — `vx_buffer_create/release/retain`, `vx_buffer_address`,
  `vx_buffer_map/unmap`.
- **Queues** — `vx_queue_create/release/retain`, `vx_queue_flush`,
  `vx_queue_finish`.
- **Events** — `vx_event_release/retain`, `vx_event_wait_all`,
  `vx_event_query`, `vx_event_create_user`, `vx_event_signal_user`.
- **Async enqueue** — `vx_enqueue_write`, `vx_enqueue_read`,
  `vx_enqueue_copy`, `vx_enqueue_launch`, `vx_enqueue_dcr_write`,
  `vx_enqueue_dcr_read`, `vx_enqueue_marker`, `vx_enqueue_barrier`.
- **Profiling** — `vx_event_profile_info`.

Five principles:

1. **Minimal surface.** vortex2.h exposes irreducible primitives.
   Complexity (programming-model abstractions, state-object catalogs,
   command-buffer recording, pipeline caches, descriptor sets,
   contexts) belongs in upper layers (POCL, chipStar, a future Vulkan
   ICD, a CUDA translator, an OpenGL Gallium driver).
2. **Asynchronous by default.** Every device-touching operation takes
   a queue and returns immediately; an optional event captures
   completion. No blocking variants in the core API — blocking is
   built from `vx_event_wait_all` or `vx_queue_finish`.
3. **OpenCL-shaped events.** Events are produced by enqueue calls (not
   recorded by a separate call). Each enqueue takes a wait-list and
   returns an event for the work it just submitted.
4. **Refcounted handles** with explicit `retain`/`release`. Matches
   what OpenCL upper layers already expect.
5. **Versioned create-info structs** (queue, launch). First field is
   `struct_size`; optional `next` extension chain.

The legacy `sw/runtime/include/vortex.h` is preserved as a backwards
compatibility shim — its `vx_dcr_*` / `vx_start` / `vx_ready_wait`
symbols are re-implemented as thin wrappers over `vortex2.h` (and
through it onto the CP).

### 7.2 Dispatcher architecture

```
                  vortex2.h (user-facing API)
                          │
              ┌───────────┴───────────┐
              ▼                       │
       libvortex.so                   │  legacy vortex.h calls
       (sw/runtime/stub/              │  are wrapped onto vortex2.h
        + sw/runtime/common/)         │  by legacy_runtime.cpp
              │                       │
              ▼                       │
       vx::Device / Queue / Buffer / Event  (refcounted C++ classes)
              │
              │ at vx_device_open: dlopen("libvortex-${VORTEX_DRIVER}.so"),
              │ resolve vx_dev_init, populate callbacks_t
              ▼
       callbacks_t  (the backend ABI — see §7.3)
              │
              ▼
       libvortex-{simx,rtlsim,xrt,opae}.so
```

The dispatcher (`libvortex.so`, built from `sw/runtime/stub/`) owns
**100% of the CP wire protocol**. `vx::Device` allocates the per-queue
ring + head + completion buffers via `mem_alloc`, zeros them, programs
the CP regfile via `cp_mmio_write`, and exposes three helpers used by
`vx::Queue`:

```cpp
class Device {
    vx_result_t cp_submit_launch();
    vx_result_t cp_submit_dcr_write(uint32_t addr, uint32_t value);
    vx_result_t cp_submit_dcr_read (uint32_t addr, uint32_t tag,
                                    uint32_t* out_value);
};
```

Each helper builds the on-wire CL (matching `VX_cp_pkg.sv`'s `cmd_t`
layout), uploads it to the ring at the current tail, commits Q_TAIL
with the LO/HI atomic-pair write, and polls Q_SEQNUM until the engine
retires it. `cp_submit_dcr_read` then reads `Q_LAST_DCR_RSP` for the
response. The helpers are synchronous from the worker thread's
perspective; the async semantics are layered above by `vx::Queue`'s
work-lambda model.

### 7.3 `callbacks_t` — the pure-v2 backend ABI

```c
typedef struct {
  int (*dev_open)    (void** out_dev_ctx);
  int (*dev_close)   (void*  dev_ctx);

  int (*query_caps)  (void* dev_ctx, uint32_t caps_id, uint64_t* out);
  int (*memory_info) (void* dev_ctx, uint64_t* free, uint64_t* used);

  int (*mem_alloc)   (void* dev_ctx, uint64_t size, uint32_t flags, uint64_t* out_dev_addr);
  int (*mem_reserve) (void* dev_ctx, uint64_t dev_addr, uint64_t size, uint32_t flags);
  int (*mem_free)    (void* dev_ctx, uint64_t dev_addr);
  int (*mem_access)  (void* dev_ctx, uint64_t dev_addr, uint64_t size, uint32_t flags);

  int (*mem_upload)  (void* dev_ctx, uint64_t dst, const void* src, uint64_t size);
  int (*mem_download)(void* dev_ctx, void* dst, uint64_t src, uint64_t size);
  int (*mem_copy)    (void* dev_ctx, uint64_t dst, uint64_t src, uint64_t size);

  int (*cp_mmio_write)(void* dev_ctx, uint32_t off, uint32_t value);
  int (*cp_mmio_read) (void* dev_ctx, uint32_t off, uint32_t* out_value);
} callbacks_t;
```

The `off` parameter to `cp_mmio_*` is the CP-internal regfile offset
(0x000..0x13F). Hardware backends translate to their own physical MMIO
addresses (xrt/opae add `0x1000` to land on the AFU's bit-12 demux).
Software backends (simx/rtlsim) forward directly to the C++
`CommandProcessor`.

The ABI has no `launch_start`, `launch_wait`, `dcr_write`, or
`dcr_read`. Every kernel launch and DCR op flows through the
dispatcher's `cp_submit_*` helpers → `cp_mmio_*` + `mem_upload`.
Adding a new backend is implementing 9 platform primitives — no
per-command protocol work.

### 7.4 Per-queue ring buffer management

The dispatcher's `vx::Device` allocates one ring (default 64 KiB) +
one head slot + one completion slot per device. The CP regfile is
programmed once at open. Subsequent submissions push CLs into the
ring at the current tail and commit `Q_TAIL` to publish them.

v1 packs one command per CL (CL-aligned tail advance), which is
correct, simple, and uses ≤1 % of the 64 KiB ring per kernel launch
(a typical launch is ~16 commands = 1024 bytes). Packing multiple
commands per CL is a forward optimization the unpack path already
supports.

The runtime's wait-list expansion (events) is built on
`CMD_EVENT_WAIT` plus the per-queue completion-seqnum slot. A
cross-queue wait is just a `CMD_EVENT_WAIT` whose `event_addr` points
at the other queue's completion slot.

---

## 8. Verification

### 8.1 RTL unit tests (`hw/unittest/`)

One Verilator harness per CP module. v1 ships:

- `cp_arbiter` — round-robin fairness, power-of-2 N edge cases.
- `cp_engine` — FSM per opcode, retire ordering, bid behavior.
- `cp_unpack` — cache-line walk with mixed cmd sizes + padding.
- `cp_launch` — start pulse + busy rise/fall handshake.
- `cp_dcr_proxy` — write + read paths with response latching.
- `cp_axil_regfile` — every register slot, atomic Q_TAIL commit.
- `cp_dma` — single-CL read + write paths.
- `cp_axi_path` — fetch + completion through the xbar.
- `cp_core` — end-to-end CMD_NOP retire through the full graph.

### 8.2 Multi-backend end-to-end

The same OpenCL kernels (`tests/opencl/{vecadd,sgemm}`) and v2-native
regression tests (`tests/regression/{vecadd,sgemm}`) run on all four
backends via the dispatcher CP path:

| | simx | rtlsim | xrt | opae |
|---|---|---|---|---|
| vecadd | ✓ | ✓ | ✓ | ✓ |
| sgemm  | ✓ | ✓ | ✓ | ✓ |

simx + rtlsim exercise the software CP; xrt + opae exercise the
hardware CP. Both paths produce bit-identical results.

### 8.3 Diff-debug methodology

The two paths share the same dispatcher code, so any divergence in
behavior between simx (software CP) and xrt (hardware CP) localizes
the bug to one side. Per-command stderr traces from
`Device::cp_submit_cl_` make the comparison cheap. This methodology
caught the `VX_cp_dcr_proxy` combinational-cmd bug — a one-line
"latch on grant" fix — in one cycle, after the same symptom had
silently bitten four prior debug sessions.

---

## 9. Future work

Deliberately out of v1, all forward-compatible with the architecture:

- **True per-CTA concurrent kernel execution** via a multi-context
  KMU. The CPE / arbiter / `ctx_id` plumbing is already in place; the
  KMU arbiter would select a slot rather than a single shared port.
- **Hardware out-of-order command queues.** The runtime already
  emulates OoO via multiple in-order HW queues + events.
- **Preemption, priority inversion, mid-kernel context switch.**
- **MSI-X interrupts** for completion (v1 polls).
- **CMD_EVENT_WAIT / CMD_EVENT_SIGNAL full wiring.** Skeletons exist;
  the engine retires them as NOPs today.
- **CMD_DCR_READ response via host-memory writeback.** Current v1
  exposes the response via the `Q_LAST_DCR_RSP` regfile slot, which
  is sufficient for the per-tag cache-flush case. A ring-driven
  writeback to host memory (using the CP's AXI master) lets multiple
  in-flight reads coexist.
- **CP DMA fully wired.** `CMD_MEM_*` opcodes are implemented in
  hardware but not yet exercised by the runtime, which still uses
  the backend's `mem_upload/download/copy` callbacks directly. The
  DMA path subsumes those once the engine's DMA resource is the
  default for bulk transfers.
- **Per-command profiling writeback.** `VX_cp_profiling` is a
  skeleton; the cycle counter is exposed but no per-command 32 B
  timestamp record is pushed yet.
- **Multi-queue.** `NUM_QUEUES` defaults to 1 in v1; the
  architecture is parameterized for N. Bumping N exercises the
  arbiter cross-queue paths that already exist.
- **Real-bitstream bring-up.** `kernel.xml` for XRT and the OPAE
  AFU manifest need updates to advertise the new MMIO range (8 KiB
  AXI-Lite slave). The simulator paths fully exercise the design;
  real-hardware execution is the remaining "checkpoint."

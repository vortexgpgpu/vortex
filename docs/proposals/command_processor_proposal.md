# Vortex Command Processor and Asynchronous Command Submission

Status: draft proposal
Branch: `feature_cp`
Related review: [docs/designs/command_processor_prototype.md](../designs/command_processor_prototype.md)

## 1. Summary

Today the Vortex runtime drives the FPGA in lock-step over MMIO: every
`vx_copy_to_dev`, `vx_dcr_write`, `vx_start`, etc. is a synchronous
transaction. There is no way for the host to queue ahead, overlap host-to-device
DMA with kernel execution, or express dependencies between operations. This
proposal introduces a proper **Command Processor (CP)** block plus an
**asynchronous, multi-queue, event-based submission model** that maps cleanly to
CUDA streams / OpenCL command queues / SYCL queues.

The design has three pillars:

1. A platform-agnostic `rtl/cp/` block that talks to the GPU through DCR/KMU and
   to the host through a canonical AXI4 + AXI4-Lite interface.
2. Thin per-platform AFU shims (`rtl/afu/xrt/` for v1) that only adapt the
   platform shell to that canonical interface.
3. A new runtime layer that exposes `vx_queue_h` and `vx_event_h` handles with
   in-order asynchronous semantics, host events, intra-queue waits, and
   cross-queue semaphores.

The previous student prototype (`~/dev/vortex_cp`, reviewed separately)
established the value of cache-line-framed commands in pinned host memory and
of an in-AFU dispatch FSM. This proposal keeps those ideas and replaces
everything else: portability layer, queue model, completion model, runtime API,
and KMU integration.

## 2. Goals and non-goals

### Goals (v1)

- **Make Vortex a conformant OpenCL 1.2 execution backend** at the
  hardware/runtime layer. Specifically: asynchronous enqueue, in-order
  command queues, events with cross-queue dependencies, user events,
  markers/barriers, and `CL_QUEUE_PROFILING_ENABLE` timestamps. See §12
  for the full conformance table.
- Decouple the CP from the platform shell. CP code lives in `rtl/cp/` with one
  canonical AXI interface; vendor shims are minimal.
- Support multiple general-purpose hardware queues, each modeled as an
  in-order command stream and each driven by its own per-queue
  **Command Processor Engine (CPE)**. CPEs converge on shared GPU
  resources (KMU, DMA, DCR bus) through round-robin arbiters. Target
  programming models: OpenCL 1.2 in-order command queues, CUDA / HIP
  streams, SYCL in-order queues.
- Achieve **concurrent submission + zero-bubble kernel succession**: while
  kernel A is draining through the KMU, queue B's CPE can independently
  fetch commands, run DMAs, evaluate waits, and pre-stage kernel B's KMU
  descriptor so the next launch starts the cycle KMU goes idle.
- Full host/device synchronization: host events, intra-queue waits,
  cross-queue semaphores, host-signalled semaphores.
- Per-command profiling timestamps written back to host memory, gated by a
  per-queue enable bit (required for `CL_QUEUE_PROFILING_ENABLE`).
- Drop the prototype's full-GPU reset on every kernel launch — launches go
  through the KMU's DCR-configured dispatcher path.
- Asynchronous DMA (both directions) and asynchronous kernel launch.
- XRT-only platform support for v1. OPAE is deprecated; the AXI surface
  leaves the door open to bring it back through an OFS/PIM shell later.

### Non-goals (v1)

- **True per-CTA concurrent kernel execution.** v1 has a single-context KMU,
  so CTAs from two different kernels are never simultaneously in flight in
  the cores. v1 ships with **concurrent submission + zero-bubble kernel
  succession** instead, which captures most of the practical CKE win
  (cross-queue DMA/compute overlap, fast kernel-to-kernel switching) and
  is sufficient for conformant OpenCL 1.2 (the spec permits
  serialization). True CTA-level CKE requires a multi-context KMU and is a
  tracked follow-on proposal — the v1 design is forward-compatible (CPE,
  arbiter, and `ctx_id` plumbing are already there).
- Out-of-order command queues (OpenCL OoO mode) implemented in hardware.
  Runtime emulates OoO by spawning multiple in-order HW queues plus events;
  CP has no native dependency tracker.
- Preemption, priority inversion, mid-kernel context switch.
- Multi-device / multi-GPU. One CP serves one Vortex instance.
- MSI-X / kernel-driver work. Completion is host-polled; interrupt support is
  listed as a v1.1 extension.

## 3. Terminology

| Term                          | Meaning in this proposal                                     |
|-------------------------------|--------------------------------------------------------------|
| **Command Processor (CP)**    | RTL block under `rtl/cp/` that owns all N CPEs plus the shared arbiters, DMA, event unit, and platform interface. |
| **Command Processor Engine (CPE)** | Per-queue engine inside the CP. One CPE per HW queue: fetches the queue's commands, decodes them, drives the per-command FSM, and bids for shared resources (KMU, DMA, DCR bus). |
| **Asynchronous Command Submission** | Runtime mechanism by which host enqueues commands and returns immediately. |
| **Command Stream**            | The ordered byte sequence of commands a queue holds in host memory. |
| **Queue (`vx_queue_h`)**      | An in-order channel from the host to one CPE. Has its own ring buffer and seqnum space. |
| **Event (`vx_event_h`)**      | A 64-bit seqnum on some queue (or a host-signalled value) usable in waits. |
| **Completion seqnum**         | Per-queue monotonic 64-bit counter written by the CP to a host-visible memory location after each command retires. |
| **Resource arbiter**          | Round-robin arbiter that picks which CPE next gets to use a shared resource (KMU launch port, DMA engine, DCR proxy). One arbiter per shared resource. |
| **AFU shim**                  | Per-platform adapter under `rtl/afu/{xrt,opae}/` that exposes the CP's canonical AXI ports as the platform's native shell. |

We deliberately avoid "deferred rendering" — that term refers to a specific
graphics pipeline technique and is unrelated to what the CP does.

## 4. High-level architecture

```
   ┌────────────────────────────── HOST ───────────────────────────────┐
   │  application                                                      │
   │     │                                                             │
   │     ▼                                                             │
   │  runtime  (sw/runtime/include/vortex.h + per-backend impls)       │
   │     │  vx_queue_create / vx_enqueue_* / vx_event_record / wait    │
   │     ▼                                                             │
   │  per-queue ring buffers in pinned host memory                     │
   │  per-queue completion-seqnum slots in pinned host memory          │
   └─────────────────┬─────────────────┬──────────────────────────────-┘
                     │ AXI4 master     │ AXI4-Lite slave (doorbells, status)
                     │ (CP DMA reads/writes)                                 
                     ▼                 ▼                                     
   ┌─────────────────────── rtl/afu/xrt (thin shim) ────────────────────-┐
   │  AXI4 master ↔ Vortex memory subsystem (existing VX_axi_adapter)   │
   │  AXI4-Lite   ↔ doorbell/status register file                       │
   │  Drives the CP's canonical interface                               │
   └─────────────────┬─────────────────────────────────────────────────-─┘
                     │ canonical CP iface (SV interface bundle)
                     ▼
   ┌──────────────────────────── rtl/cp ──────────────────────────────────┐
   │  VX_cp_core                                                           │
   │                                                                      │
   │   ┌─ CPE[0] ─┐  ┌─ CPE[1] ─┐  ┌─ CPE[2] ─┐  ┌─ CPE[N-1] ─┐           │
   │   │ fetch    │  │ fetch    │  │ fetch    │  │ fetch      │           │
   │   │ unpack   │  │ unpack   │  │ unpack   │  │ unpack     │ … one CPE │
   │   │ decode   │  │ decode   │  │ decode   │  │ decode     │   per HW  │
   │   │ ring ptr │  │ ring ptr │  │ ring ptr │  │ ring ptr   │   queue   │
   │   │ seqnum   │  │ seqnum   │  │ seqnum   │  │ seqnum     │           │
   │   │ FSM      │  │ FSM      │  │ FSM      │  │ FSM        │           │
   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬───────┘           │
   │        │             │             │             │                   │
   │        └────────┬────┴─────────────┴─────────────┘                   │
   │                 │  per-CPE bids for shared resources                 │
   │                 ▼                                                    │
   │    ┌─────────────────────────────────────────────────────┐           │
   │    │  Resource arbiters (round-robin, one per resource)  │           │
   │    │   ├── KMU launch arbiter   → VX_cp_launch (start)   │           │
   │    │   ├── DMA arbiter          → VX_cp_dma              │           │
   │    │   └── DCR arbiter          → VX_cp_dcr_proxy        │           │
   │    └─────────────────────────────────────────────────────┘           │
   │                                                                      │
   │   ┌────────────────────────────────────────────────────────────┐     │
   │   │  Shared helpers (used by all CPEs through arbiters):       │     │
   │   │   ├── VX_cp_event_unit       (wait/signal seqnum compare)  │     │
   │   │   ├── VX_cp_completion       (per-queue seqnum writeback)  │     │
   │   │   ├── VX_cp_profiling        (free-running cycle counter   │     │
   │   │   │                           + per-command TS writeback)  │     │
   │   │   └── VX_cp_axi_xbar         (mux of CPE/DMA/event/cmpl    │     │
   │   │                               onto the one AXI master)     │     │
   │   └────────────────────────────────────────────────────────────┘     │
   └─────────┬──────────────────────┬─────────────────────┬───────────────┘
             │ DCR req/rsp           │ start/busy           │ AXI4 master
             ▼                       ▼                      ▼
                            Vortex.sv (GPU core)
                            (single-context KMU; consumes DCRs,
                             launches one kernel's CTAs at a time)
```

The CP is one block with:

- **N parallel CPEs** (one per HW queue, see §6.3). Each CPE owns its own
  ring-buffer state, FSM, and seqnum counter, and runs independently of
  the others.
- **Resource arbiters** that round-robin between CPEs for each shared
  resource (KMU launch port, DMA engine, DCR proxy). A CPE may block on
  one resource while another CPE makes progress on a different one — this
  is where the cross-queue overlap comes from.
- One **upstream AXI master** for command fetch, DMA, completion writeback,
  and profiling-timestamp writeback, multiplexed via `VX_cp_axi_xbar`.
- One **AXI4-Lite slave** for the host to write doorbells and read CP status.
- One **DCR master interface** down into the GPU (request + response).
- One **start/busy** handshake to the single-context KMU.

The single-context KMU is the serialization point for kernel launches: at
any instant only one kernel's CTA grid is being emitted. CPEs not currently
holding the KMU arbiter are free to do everything else (fetch, decode, DMA,
event waits, DCR programming for their *next* launch). This is what we mean
by "concurrent submission + zero-bubble kernel succession."

The platform shim's job is only to splice the CP's AXI master/slave into the
shell's AXI infrastructure. The XRT shim is near-trivial because
`Vortex_axi.sv` is already AXI; the CP and Vortex memory ports just share the
AXI fabric (or live on separate bank groups).

## 5. Why AXI as the canonical CP interface

We pick AXI4 (master) + AXI4-Lite (slave) over CCI-P / Avalon / custom protocols
for the CP's external boundary.

Pros:

- Vortex's XRT path is already AXI; zero adaptation needed in v1.
- Modern Intel OFS shells expose AXI to the AFU; reviving OPAE later means
  writing one PIM-based shim, not a CCI-P bridge plus all the rest.
- Universal vendor and IP support (Xilinx/AMD, Intel/Altera, Microsemi, Lattice,
  ASIC flows, datacenter PCIe→AXI bridges). Future-proofs Versal/Chiplet/non-FPGA
  retargets.
- Rich verification ecosystem (BFMs, VIP, formal kits) — useful because the CP
  is the new fault-prone surface.
- Clean separation of control plane (AXI-Lite) from data plane (AXI4).

Cons / mitigations:

- CCI-P offers cache hints / address-space features AXI lacks. Not used by
  our command-stream workload.
- AXI4 is multi-channel and heavier than a streaming protocol. The cost is in
  the shell, not the CP itself.
- Tag width on the AXI master is shell-dependent, capping outstanding requests.
  We parametrize the CP for `CP_AXI_TID_WIDTH` and degrade gracefully on
  small-tag shells.

## 6. Hardware design

### 6.1 Source tree

```
hw/rtl/cp/
├── VX_cp_pkg.sv               command opcodes, struct typedefs, parameters
├── VX_cp_if.sv                SV interface bundles (CP↔AFU, CP↔Vortex, CPE↔arbiters)
├── VX_cp_core.sv               top-level CP wrapper; instantiates N CPEs + arbiters + helpers
├── VX_cp_engine.sv                  one Command Processor Engine (per HW queue)
│                               — owns ring-buffer state, fetch, unpack, decode, per-cmd FSM
├── VX_cp_fetch.sv             AXI master read of next command cache line (used inside each CPE)
├── VX_cp_unpack.sv            cache-line → packed cmd_t stream (≤5 cmds/CL) (used inside each CPE)
├── VX_cp_arbiter.sv           generic round-robin arbiter; instantiated 3× for KMU/DMA/DCR
├── VX_cp_launch.sv            KMU start/busy port wrapper, owned by KMU arbiter
├── VX_cp_dma.sv               AXI ↔ Vortex memory DMA engine, owned by DMA arbiter
├── VX_cp_dcr_proxy.sv         DCR req/rsp into Vortex/KMU, owned by DCR arbiter
├── VX_cp_event_unit.sv        wait-on-seqnum comparator, signal generator (shared, per-CPE state)
├── VX_cp_completion.sv        writes per-queue completion seqnums + head pointers to host
├── VX_cp_profiling.sv         free-running cycle counter + per-command TS writeback
└── VX_cp_axi_xbar.sv          arbitrates CPEs + DMA + event_unit + completion + profiling onto
                                a single AXI master

hw/rtl/afu/
├── xrt/                       thin AXI-Lite + AXI fabric shim around CP+Vortex
└── opae/                      deprecated for v1; revisited as OFS/PIM shim later
```

There is no separate "queue manager" or "queue arbiter" block. Each CPE is
the manager of exactly one queue; the arbiters live on the *resource* side
(KMU, DMA, DCR), not the queue side.

The current AFU files (`hw/rtl/afu/xrt/VX_afu_wrap.sv`,
`VX_afu_ctrl.sv`) are split so that the AXI fabric, parameterization, and clock
crossing stay in `afu/xrt/` while all command-stream logic moves into `cp/`.

### 6.2 Canonical CP interface (`VX_cp_if`)

The CP is connected to the platform shim via a small set of SV interfaces:

```systemverilog
// to/from host (platform shim translates to/from native shell)
interface VX_cp_axi_if;
  // AXI4 master  (32B/64B data, parameterized addr/tid width)
  axi4_master ar, r, aw, w, b;
  // AXI4-Lite slave for doorbells + CP status
  axi4lite_slave  ctrl;
endinterface

// to/from Vortex GPU
interface VX_cp_gpu_if;
  // DCR req/rsp (both directions; today's Vortex.sv only exposes write-only
  // — this proposal makes DCR a true req/rsp bus, see §6.7)
  dcr_req_t   dcr_req;    logic dcr_req_valid; logic dcr_req_ready;
  dcr_rsp_t   dcr_rsp;    logic dcr_rsp_valid;
  // KMU launch handshake
  logic       start; logic busy;
  // CP DMA borrows a Vortex memory port (or shares the AXI fabric — see §6.6)
endinterface
```

The platform shim only sees `VX_cp_axi_if` and standard memory; it never
parses commands or knows about queues.

### 6.3 Queue model and CPE state

Each queue is identified by a small integer `qid` in `[0, NUM_QUEUES)`.
`NUM_QUEUES` is a compile-time parameter (default 4, configurable). It
also implicitly sets the number of CPEs — **there is exactly one CPE per
queue**; there is no separate `NUM_CPES` knob. The reasoning: an in-order
queue has no internal parallelism, so >1 CPE per queue is pointless; <1
CPE per queue reintroduces the head-of-line blocking the design is built
to avoid; the CPE itself is small (a few hundred FFs + the per-cmd FSM)
so 1-per-queue is cheap.

Each queue has:

- A host-allocated, pinned, page-aligned ring buffer with power-of-two byte
  capacity (`CP_QUEUE_RING_BYTES`, default 64 KiB per queue).
- A device-readable `head` (consumer pointer, written by CP), a host-written
  `tail` (producer pointer), both 64-bit byte offsets, both in pinned host
  memory.
- A completion-seqnum slot in host memory; CP writes the most recent
  retired-command seqnum after each retirement.
- A 64-bit seqnum counter inside the owning CPE, incremented at retirement.

Per-CPE state (one instance of this struct lives inside each `VX_cp_engine`):

```systemverilog
typedef struct packed {
  logic [63:0] ring_base;       // host VA / IO addr of ring buffer
  logic [31:0] ring_size_log2;
  logic [63:0] head_addr;       // host mem address where CPE publishes head
  logic [63:0] cmpl_addr;       // host mem address where CPE publishes seqnum
  logic [63:0] tail;            // last value of tail seen via doorbell
  logic [63:0] head;            // CPE-internal consumer pointer
  logic [63:0] seqnum;          // next retire seqnum
  logic        enabled;
  logic [1:0]  priority;        // 0=lo, 3=hi
  logic        profile_en;      // CL_QUEUE_PROFILING_ENABLE (see §6.11)
} cpe_state_t;
```

The doorbell is one AXI4-Lite write per push (`tail` field), at the
queue's MMIO offset. The CPE can also re-read `tail` from host memory if
a doorbell is coalesced — see §6.10.

### 6.4 Resource arbiters (replaces "queue arbiter")

Because each queue has its own CPE, there is no central queue arbiter to
pick "which queue runs next." Instead, every shared resource has its own
small round-robin arbiter that decides "which CPE gets me this cycle":

| Arbiter             | Resource it gates                              | When a CPE bids                                                |
|---------------------|------------------------------------------------|-----------------------------------------------------------------|
| **KMU arbiter**     | `VX_cp_launch` (start pulse + busy observation) | CPE has a `CMD_LAUNCH` decoded and ready                       |
| **DMA arbiter**     | `VX_cp_dma` (AXI ↔ device-mem engine)          | CPE has a `CMD_MEM_{READ,WRITE,COPY}` decoded and ready        |
| **DCR arbiter**     | `VX_cp_dcr_proxy` (req/rsp into KMU & GPU)     | CPE has a `CMD_DCR_{READ,WRITE}` decoded and ready             |

Properties:

- Each arbiter is independent. A CPE blocked on `KMU` does not prevent
  another CPE from getting `DMA` or `DCR` the same cycle — this is the
  source of cross-queue overlap.
- Round-robin is the v1 policy. Priority is supported through the per-CPE
  `priority` field by skipping low-priority CPEs at the arbiter when a
  high-priority CPE is bidding (configurable; off by default for fairness).
- KMU arbitration holds for the entire duration of a launch (from `start`
  pulse until `busy` falls): the single-context KMU cannot accept a new
  descriptor mid-grid. CPEs holding the KMU release it the cycle they
  retire their `CMD_LAUNCH`; the next-winning CPE may then immediately
  write its descriptor's DCRs (via the DCR arbiter) and pulse `start` —
  zero-bubble succession.
- DMA and DCR arbitration are per-transaction (release after each
  command). This keeps long DMAs from starving DCR programming.

This structure is the entire reason the design is forward-compatible with
a multi-context KMU: the KMU arbiter would simply select a *slot* in the
KMU rather than a single shared port; nothing else changes.

### 6.5 Command set

All commands carry a 4-byte header (`{opcode[7:0], flags[7:0], reserved[15:0]}`)
followed by opcode-specific payload. Cache-line framing rule from the
prototype is kept: a command never crosses a 64 B boundary; the rest of the
line is zero-padded.

Header flag bits used in v1:

| Flag bit | Name              | Meaning                                                                  |
|----------|-------------------|--------------------------------------------------------------------------|
| `flags[0]` | `F_PROFILE`     | Command is profiled. Payload is followed by an 8 B `profile_slot` host address; CP writes 4×8 B timestamps to that slot at retirement (see §6.11). |
| `flags[1]` | `F_FENCE_PRE`   | Treat as if a `CMD_FENCE(FENCE_ALL)` was inserted immediately before this command. Lets the runtime fuse a fence into the next command without spending a CL slot on `CMD_FENCE`. |
| `flags[2-7]` | reserved      | Must be zero in v1.                                                      |

| Opcode             | Payload                                            | Purpose                                            |
|--------------------|----------------------------------------------------|----------------------------------------------------|
| `CMD_NOP`          | —                                                  | padding / pacing                                   |
| `CMD_MEM_WRITE`    | `host_addr, dev_addr, size` (each 8 B)             | host→device DMA                                    |
| `CMD_MEM_READ`     | `host_addr, dev_addr, size`                        | device→host DMA                                    |
| `CMD_MEM_COPY`     | `src_dev, dst_dev, size`                           | device→device DMA                                  |
| `CMD_DCR_WRITE`    | `dcr_addr, dcr_value`                              | program GPU/KMU DCR                                |
| `CMD_DCR_READ`     | `dcr_addr, host_writeback_addr`                    | read GPU DCR, write result to host                 |
| `CMD_LAUNCH`       | `kmu_ctx_id, flags`                                | pulse KMU `start`; assumes KMU is preprogrammed via `CMD_DCR_WRITE`s |
| `CMD_FENCE`        | `mask`                                             | retirement barrier within this queue (caches/DMA flush) |
| `CMD_EVENT_SIGNAL` | `event_addr, value`                                | write a 64-bit value to host-visible event slot    |
| `CMD_EVENT_WAIT`   | `event_addr, value, op`                            | stall queue until `*event_addr op value` is true   |

Notes:

- `CMD_LAUNCH` replaces the prototype's `CMD_RUN`. It does **not** reset the
  GPU. The runtime is responsible for emitting `CMD_DCR_WRITE`s into the
  same queue ahead of `CMD_LAUNCH` to configure KMU (grid/block dims, PC,
  args, lmem, warp step — the full set documented in
  [hw/rtl/VX_kmu.sv](../../hw/rtl/VX_kmu.sv)).
- `CMD_EVENT_WAIT` is the building block for both intra-queue waits and
  cross-queue semaphores: the event slot is just a 64-bit host-memory
  address, and "another queue" simply means that address is the other
  queue's completion-seqnum slot.

Sizes (header + payload): `CMD_NOP` = 4 B, `CMD_LAUNCH` = 8 B,
`CMD_DCR_WRITE` / `CMD_EVENT_SIGNAL` / `CMD_FENCE` = 20 B,
`CMD_MEM_*` / `CMD_EVENT_WAIT` / `CMD_DCR_READ` = 28 B.

### 6.6 DMA engine and memory bus sharing

`VX_cp_dma` is a small generic DMA engine: source/dest address + size, with
both endpoints expressible as either the CP's AXI master (host memory) or
the Vortex memory subsystem (device memory). For `CMD_MEM_COPY` both
endpoints are device.

For device-side accesses the CP can either:

1. **Borrow a dedicated Vortex memory port** — clean isolation, but uses a
   port and may unbalance bank usage. Recommended on configurations with
   `VX_MEM_PORTS > 1`.
2. **Multiplex onto the host AXI fabric** — works when the platform shell
   exposes device memory and host memory on the same AXI fabric (XRT
   typical), but the CP must arbitrate against GPU traffic.

This is a build-time choice (`CP_DMA_DEV_PORT_MODE = DEDICATED|SHARED`).

**v1 default: `SHARED`.** Works on every XRT shell (including single-bank
boards), zero shell-dependence. `DEDICATED` is opt-in via
`--cp-dma-port=dedicated` on multi-bank shells where CP↔GPU memory
contention measurably hurts throughput; phase 5 perf measurements decide
whether to promote `DEDICATED` to the default.

### 6.7 DCR bus becomes request/response

The current `Vortex.sv` exposes a DCR write-only interface. We extend it to
true request/response (the structure is already present internally —
`VX_dcr_bus_if` carries both — only the top-level wires are write-only).

Changes:

- `Vortex.sv` and `Vortex_axi.sv` gain `dcr_rsp_valid, dcr_rsp_data` outputs.
- `VX_cp_dcr_proxy` issues both reads and writes; reads return data the CP
  can either consume directly (for status polling) or writeback to host via
  `CMD_DCR_READ`'s `host_writeback_addr`.

This eliminates the prototype's "software DCR shadow" hack and makes
`vx_dcr_read` observe real GPU state again.

### 6.8 Event unit and completion

`VX_cp_event_unit` evaluates `CMD_EVENT_WAIT`:

- Reads the 8 B at `event_addr` via the AXI master (cached internally with a
  small LRU; entries invalidated when an `EVENT_SIGNAL` writes a matching
  address, or by a watchdog re-read).
- Comparison op is one of `EQ, GE, GT, NE`. `GE` is the common case for
  CUDA-event-style "wait until queue A reaches seqnum N."
- The queue holding the wait is marked `blocked_on_wait` until the
  comparison succeeds; the arbiter skips it.

`VX_cp_completion` retires commands:

- Increments the queue's seqnum on every `CMD_*` retirement except
  `CMD_NOP`.
- Writes the new seqnum to that queue's `cmpl_addr` via the AXI master.
- Updates the queue's `head` and writes it to `head_addr` so the host can
  reclaim ring-buffer space.
- (v1.1) Optionally raises an interrupt to the platform shim.

### 6.9 Completion ordering and fences

Within a queue, commands retire in submission order — that's the entire
point of in-order semantics. Across queues, ordering is the user's job
(events). `CMD_FENCE` forces stronger guarantees within a queue:

- `FENCE_DMA`: wait until all prior DMAs on this queue have drained on the
  host side (CP holds the next command until the AXI write-response budget
  is empty).
- `FENCE_GPU`: wait until `vx_busy == 0` (KMU/launch fully drained).
- `FENCE_ALL`: both.

The runtime emits `CMD_FENCE(FENCE_GPU)` automatically before any
`CMD_MEM_READ` that targets memory written by a recent `CMD_LAUNCH` on the
same queue, so `vx_copy_from_dev` after `vx_launch` is safe by default.

### 6.10 MMIO doorbell layout (AXI4-Lite slave)

```
0x000   CP_CTRL              [0]=enable [1]=soft_reset [2]=irq_enable
0x004   CP_STATUS            [0]=ready  [1..]=per-queue active mask
0x008   CP_DEV_CAPS_LO       num_queues, ring_size_log2, max_cmds_per_cl
0x00C   CP_DEV_CAPS_HI       reserved
0x010   CP_IRQ_STATUS / ACK
...
0x100 + qid*0x40  per-queue block:
    +0x00  Q_RING_BASE_LO/HI    (write at queue-create)
    +0x08  Q_HEAD_ADDR_LO/HI    (write at queue-create)
    +0x10  Q_CMPL_ADDR_LO/HI    (write at queue-create)
    +0x18  Q_RING_SIZE_LOG2
    +0x1C  Q_CONTROL            [0]=enable [1]=reset [2]=priority lo/hi
                                [3]=profile_en (CL_QUEUE_PROFILING_ENABLE)
    +0x20  Q_TAIL_LO            doorbell low-half — latched, not yet committed
    +0x24  Q_TAIL_HI            doorbell high-half + commit pulse — atomically latches
                                {Q_TAIL_HI[31:0], Q_TAIL_LO[31:0]} as the new tail
    +0x28  Q_SEQNUM_LO/HI       (RO) most recent retired seqnum
    +0x30  Q_ERROR              (RO) per-queue error code
    +0x38  reserved
```

The 64-bit `tail` doorbell is committed atomically by the high-half
write: the host writes `Q_TAIL_LO` first (CP latches it but does not
update the queue's tail register), then writes `Q_TAIL_HI`, which both
latches the high half *and* fires a 1-cycle commit pulse that atomically
publishes the 64-bit `{HI, LO}` as the new tail visible to the CPE. This
removes any dependency on AXI-Lite ordering across the interconnect — a
host that writes only `Q_TAIL_LO` cannot accidentally advance the queue.

The AXI-Lite map also exposes a small read-only profiling block at
`0x040..0x05F`:

```
0x040   CP_CYCLE_LO         (RO) low 32 b of free-running cycle counter
0x044   CP_CYCLE_HI         (RO) high 32 b
0x048   CP_CYCLE_FREQ_HZ    (RO) CP clock frequency, for host-side TS conversion
0x04C   reserved
```

The runtime reads `CP_CYCLE_FREQ_HZ` once at device open and uses it to
convert the 64-bit cycle timestamps the CP writes back (§6.11) into the
nanosecond values OpenCL expects.

### 6.11 Profiling timestamps (`VX_cp_profiling`)

To support `CL_QUEUE_PROFILING_ENABLE`, the CP exposes a free-running
64-bit cycle counter (`cp_cycle`) clocked off the CP clock, read-visible
via the AXI-Lite block at `0x040` (§6.10).

A profiled command (any command with `F_PROFILE` set in its header) is
followed in the ring buffer by an 8 B `profile_slot` host address. The
CPE samples the cycle counter at:

| Field   | Sampled at                                              | Notes                                          |
|---------|---------------------------------------------------------|------------------------------------------------|
| QUEUED  | (host-side) before the doorbell is rung                 | Runtime fills this from its own clock          |
| SUBMIT  | CPE fetches the command's cache line into the unpacker  | First time CP "sees" the command               |
| START   | Resource arbiter grants the command its resource        | KMU `start` pulse, DMA `aw`/`ar` fire, etc.    |
| END     | Command retires                                         | Same instant the completion seqnum advances    |

`VX_cp_profiling` performs the writeback by pushing a 32 B record
(`{QUEUED, SUBMIT, START, END}`) to `profile_slot` via the AXI master,
arbitrated through `VX_cp_axi_xbar`. The runtime returns these to OpenCL
via `clGetEventProfilingInfo` after converting cycles → ns using
`CP_CYCLE_FREQ_HZ`.

The per-CPE `profile_en` bit gates the writeback: if zero, the
`F_PROFILE` flag is silently ignored and the `profile_slot` 8 B in the
ring buffer is consumed but not written back. This lets the runtime
build a single command-generation path and only pay the writeback cost
on profiled queues. `profile_en` is set by writing the per-queue
`Q_CONTROL` register at queue create.

### 6.12 DCR address allocations

Per [VX_types.toml](../../VX_types.toml), free ranges are 0x02F–0x0FF
and 0x300–0xFFF. We reserve **`0x080–0x0BF`** (64 entries) for CP-internal
DCRs that the GPU itself needs to be aware of (currently: none; placeholder
for future CP↔GPU coordination such as in-flight kernel barriers).

The host-visible CP control surface is on the AXI4-Lite slave (§6.10), not
the DCR bus, so we do not consume DCR space for doorbells.

## 7. Platform frontends

### 7.1 XRT frontend (v1 target)

`rtl/afu/xrt/VX_afu_wrap.sv` becomes a small wrapper that:

- Instantiates `VX_cp_core` and `Vortex.sv` (or `Vortex_axi.sv`) side by side.
- Splices the CP's AXI master into the existing XRT AXI fabric — either
  sharing the GPU's memory channels (single bank group) or on a dedicated
  bank group (multi-bank kernels).
- Maps the CP's AXI4-Lite slave to the kernel's AXI4-Lite control port. The
  existing AP_CTRL (`ap_start`, `ap_done`) handshake is replaced: the host
  no longer "starts the kernel" once — the CP is the long-running kernel
  that consumes work from its queues.
- Forwards the CP's optional interrupt to the kernel's `interrupt` output
  (v1.1).

### 7.2 OPAE frontend (deprecated for v1)

The OPAE shim is intentionally not built for v1. The CP's AXI surface keeps
the door open: a future OPAE shim, written against an OFS/PIM AXI-native
shell, would be ≈the same size as the XRT shim. Legacy CCI-P-only shells
are out of scope.

## 8. Runtime API

### 8.1 Two headers, one `vx_*` namespace

The CP gets a clean, async-first, OpenCL-shaped API in a **new** header
`sw/runtime/include/vortex2.h`. The existing
[sw/runtime/include/vortex.h](../../sw/runtime/include/vortex.h) is
**kept for backward compatibility** so that POCL, chipStar, SimX/rtlsim
harnesses, and the existing in-tree tests continue to build without
changes.

Both headers share the project-standard `vx_*` symbol prefix. The new
header **`#include`s the legacy `vortex.h`** so that the existing
typedefs (`vx_device_h`, `vx_buffer_h`) and constants are inherited
unchanged, and so that translation units can mix old and new calls
during the migration.

| Header                              | Purpose                                                 | Lifetime                                                   |
|-------------------------------------|---------------------------------------------------------|------------------------------------------------------------|
| `sw/runtime/include/vortex.h`       | Legacy synchronous API as it exists today. Provides `vx_device_h`, `vx_buffer_h`, and the existing `vx_dev_open` / `vx_start` / `vx_ready_wait` / `vx_mpm_query` / etc. family. | Stays for the foreseeable future; no behavioral changes in v1. |
| `sw/runtime/include/vortex2.h`      | New async, refcounted, event-based API. `#include`s `vortex.h`. Adds new handles (`vx_context_h`, `vx_queue_h`, `vx_event_h`, `vx_kernel_h`, plus typed state-object handles per fixed-function block), `vx_enqueue_*`, `vx_event_*`, raw `vx_enqueue_dcr_*`, and the typed state-object constructors. The canonical interface for the CP and the OpenCL 1.2 backend path. | Becomes the only path long-term; legacy is re-implemented as a thin shim over `vortex2` in phase 8. |

Function names in `vortex2.h` are chosen to **not collide** with the
legacy ones (e.g. legacy `vx_dev_open` vs new `vx_device_open`; legacy
`vx_start` vs new `vx_enqueue_launch`). The single existing legacy
function that names a similar concept is `vx_mpm_query`, which the new
header **inherits unchanged** from `vortex.h` — it doesn't redefine it.

This means: **the new CP is wired up through `vortex2.h` from day one**.
Legacy `vortex.h` users keep getting the legacy lock-step path through
the existing AFU control surface (which the CP-aware AFU still exposes
as a compatibility mode), until the legacy shim work in phase 8 lands.

### 8.2 `vortex2.h` design principles

`vortex2.h` is the **minimal async runtime surface** for Vortex.
Complexity — programming-model abstractions, state object catalogs,
command-buffer recording, pipeline caches, descriptor sets, context
grouping, sub-buffers, heaps — belongs in **upper layers** built on
top of vortex2: POCL, chipStar, a future Vulkan-on-Vortex ICD, a CUDA
translator, an OpenGL Gallium driver, etc. The runtime gives those
layers a small, sharp set of primitives and gets out of the way.

Five principles:

1. **Minimal surface.** vortex2.h exposes the irreducible primitives a
   GPU runtime must provide: device lifetime, buffers (including
   zero-copy mapping), queues, asynchronous submission, events, raw
   DCR access. 34 functions total across 6 families (see §8.11 for the
   full surface). Everything else is upper-layer code.
2. **Asynchronous by default.** Every operation that touches the
   device takes a queue and returns immediately; an optional event
   handle captures completion. There is no blocking variant in the
   core API — blocking is built from `vx_event_wait_all` or
   `vx_queue_finish`.
3. **OpenCL-shaped events.** Events are produced by enqueue calls (not
   recorded by a separate call). Each enqueue takes a wait-list and
   returns an event for the work it just submitted.
4. **Refcounted handles with explicit lifecycle.** `retain` / `release`
   on every object class. Closes the prototype's pinned-buffer-leak
   class of bugs and matches what OpenCL upper layers already expect.
5. **Versioned create-info structs** for the two info structs that
   exist (queue, launch). First field is `struct_size`; optional `next`
   extension chain. New fields can be added later without breaking ABI.

What `vortex2.h` deliberately does **not** include (and why):

- **No `vx_context_h`.** A context is a pure software grouping that
  every upper layer (`cl_context`, `VkDevice`, `CUcontext`,
  `hipCtx_t`) keeps in its own bookkeeping anyway. Queues, buffers,
  and events attach to a `vx_device_h` directly.
- **No `vx_kernel_h`.** A kernel is a loaded ELF — pass it as the
  `vx_buffer_h` that holds the ELF. Symbol resolution, kernel argument
  layout, and program management are upper-layer concerns.
- **Buffers use the `vx_buffer_*` namespace in vortex2.h** (§8.5),
  matching the `vx_buffer_h` handle type and the retain/release
  convention used by every other class. `vx_buffer_create`,
  `vx_buffer_release`, `vx_buffer_retain`, `vx_buffer_address`, etc.
  The legacy `vx_mem_*` family stays in `vortex.h` for backward
  compatibility and is internally implemented as wrappers over
  `vx_buffer_*`.
- **No typed state objects (TEX/RASTER/OM/DXA) in vortex2.h.** Per-block
  DCR programming lives in **optional helper headers** owned by the
  block's own proposal (e.g. `vortex_tex.h` under the gfx proposal),
  each built on `vx_enqueue_dcr_write`. Upper layers that don't
  care about a particular block don't include the header.
- **No command buffers, pipeline objects, descriptor sets, heaps,
  sub-buffer views.** All Vulkan/D3D12/CUDA niceties — implemented by
  the API translator that needs them, in its own memory, submitting
  the resulting command sequence via the queue's `vx_enqueue_*`
  primitives.
- **No synchronous shortcuts.** `vortex.h` is the wrapper for callers
  who want simple blocking semantics.
- **No perf-counter / scope wrappers.** Inherited `vx_mpm_query` from
  `vortex.h` covers perf counters; anything else uses raw
  `vx_enqueue_dcr_read`.

DCR programming itself is exposed via `vx_enqueue_dcr_{read,write}`
(§8.6) — first-class in vortex2.h, because raw DCR access is a
legitimate primitive that helper headers and upper layers compose on
top of. See §8.10 for the full layering picture.

### 8.3 Core handle and result types

```c
#include <vortex.h>   // inherits vx_device_h, vx_buffer_h, VX_CAPS_*,
                      // vx_mem_alloc/free/address/info, vx_mpm_query, ...

// new opaque handles introduced by vortex2.h
typedef struct vx_queue*    vx_queue_h;
typedef struct vx_event*    vx_event_h;

// inherited from vortex.h (kept as void* for ABI compatibility):
//   typedef void* vx_device_h;
//   typedef void* vx_buffer_h;

// typed result enum + readable error strings (no more bare ints)
typedef enum {
    VX_SUCCESS = 0,
    VX_ERR_INVALID_HANDLE,
    VX_ERR_INVALID_INFO,
    VX_ERR_OUT_OF_HOST_MEMORY,
    VX_ERR_OUT_OF_DEVICE_MEMORY,
    VX_ERR_DEVICE_LOST,
    VX_ERR_TIMEOUT,
    VX_ERR_EVENT_FAILED,
    VX_ERR_NOT_SUPPORTED,
    /* ... */
} vx_result_t;

const char* vx_result_string(vx_result_t r);

// Profile timestamps returned to host by VX_cp_profiling (§6.11)
typedef struct {
    uint64_t queued_ns;   // host-side, sampled before doorbell
    uint64_t submit_ns;   // CP fetched the command
    uint64_t start_ns;    // CP dispatched the command to its resource
    uint64_t end_ns;      // CP retired the command
} vx_profile_info_t;
```

### 8.4 Devices

vortex2.h exposes the full device API under the `vx_device_*` namespace,
matching the `vx_device_h` handle type. The legacy `vx_dev_open` /
`vx_dev_close` / `vx_dev_caps` functions stay in `vortex.h` as thin
wrappers over these.

```c
/* Enumeration. */
vx_result_t vx_device_count   (uint32_t* out_count);

/* Open a device by index in [0, count). Returns refcount = 1. */
vx_result_t vx_device_open    (uint32_t index, vx_device_h* out);

/* Refcount. */
vx_result_t vx_device_retain  (vx_device_h dev);
vx_result_t vx_device_release (vx_device_h dev);

/* Query a device capability. caps_id uses the VX_CAPS_* constants
 * inherited from vortex.h (VX_CAPS_VERSION, VX_CAPS_NUM_CORES,
 * VX_CAPS_GLOBAL_MEM_SIZE, VX_CAPS_ISA_FLAGS, etc.). */
vx_result_t vx_device_query   (vx_device_h dev, uint32_t caps_id,
                               uint64_t* out_value);

/* Global heap state for the device. */
vx_result_t vx_device_memory_info(vx_device_h dev,
                                  uint64_t* free, uint64_t* used);
```

(For 1.0 → 2.0 mapping of `vx_dev_open` / `vx_dev_close` / `vx_dev_caps`
/ `vx_mem_info`, see §9.)

### 8.4.1 Queues

Each queue is a hardware command stream consumed by one CPE (§6.3).
Refcounted and async-by-default like everything else:

```c
typedef enum {
    VX_QUEUE_PRIORITY_LOW    = 0,
    VX_QUEUE_PRIORITY_NORMAL = 1,
    VX_QUEUE_PRIORITY_HIGH   = 2,
} vx_queue_priority_e;

typedef struct {
    size_t                struct_size;     /* sizeof(vx_queue_info_t) */
    const void*           next;
    vx_queue_priority_e   priority;
    uint32_t              flags;           /* VX_QUEUE_PROFILING_ENABLE, … */
} vx_queue_info_t;

#define VX_QUEUE_PROFILING_ENABLE  (1u << 0)

vx_result_t vx_queue_create  (vx_device_h dev, const vx_queue_info_t* info,
                              vx_queue_h* out);
vx_result_t vx_queue_retain  (vx_queue_h q);
vx_result_t vx_queue_release (vx_queue_h q);
vx_result_t vx_queue_flush   (vx_queue_h q);                       /* doorbell now */
vx_result_t vx_queue_finish  (vx_queue_h q, uint64_t timeout_ns);  /* = clFinish */
```

### 8.5 Buffers

vortex2.h exposes the buffer API under the consistent `vx_buffer_*`
namespace that matches the `vx_buffer_h` handle type. The legacy
`vx_mem_*` family stays in `vortex.h` for backward compatibility; both
families operate on the same underlying handle.

```c
// vortex2.h — canonical buffer API
vx_result_t vx_buffer_create  (vx_device_h dev,
                               uint64_t    size,
                               uint32_t    flags,    // VX_MEM_READ | VX_MEM_WRITE | …
                               vx_buffer_h* out);

vx_result_t vx_buffer_reserve (vx_device_h dev,
                               uint64_t    address,
                               uint64_t    size,
                               uint32_t    flags,
                               vx_buffer_h* out);

vx_result_t vx_buffer_retain  (vx_buffer_h buf);
vx_result_t vx_buffer_release (vx_buffer_h buf);

vx_result_t vx_buffer_address (vx_buffer_h buf, uint64_t* out);
vx_result_t vx_buffer_access  (vx_buffer_h buf,
                               uint64_t    offset,
                               uint64_t    size,
                               uint32_t    flags);

/* Host-side mapping for device-visible buffers (pinned host memory or
 * BAR-mapped device memory). Zero-copy alternative to vx_enqueue_read /
 * vx_enqueue_write. Required by every upper-layer API that exposes
 * mapped memory: clEnqueueMapBuffer, vkMapMemory, cudaHostAlloc +
 * cudaHostGetDevicePointer, Metal newBufferWithBytesNoCopy, glMapBuffer.
 *
 * Returns VX_ERR_NOT_SUPPORTED if the buffer was not created with a
 * host-visible flag (e.g. VX_MEM_PIN_MEMORY). */
vx_result_t vx_buffer_map     (vx_buffer_h buf,
                               uint64_t    offset,
                               uint64_t    size,
                               uint32_t    flags,        /* VX_MEM_READ / WRITE */
                               void**      out_host_ptr);

vx_result_t vx_buffer_unmap   (vx_buffer_h buf, void* host_ptr);
```

(`vx_device_memory_info` is in §8.4 with the rest of the device API,
since it is a property of the device rather than of any single buffer.)

Refcount semantics (same as every other handle class):

- `vx_buffer_create` / `vx_buffer_reserve` return refcount = 1, owned
  by the caller.
- `vx_buffer_retain` increments. Used by the runtime to keep a buffer
  alive across in-flight CP commands, and by upper layers that need
  shared ownership (`cl_mem`, `VkBuffer`).
- `vx_buffer_release` decrements; at 0 the underlying allocation is
  actually freed.

**Why the refcount matters at the runtime layer**: when a CPE has a
`CMD_MEM_{READ,WRITE,COPY}` queued against a buffer, the runtime
internally `vx_buffer_retain`s the buffer at enqueue time and
`vx_buffer_release`s it at command retirement. Without this, an
upper-layer free call could destroy a buffer while the CP still has
DMA in flight against it.

(For 1.0 → 2.0 mapping of the `vx_mem_*` family, see §9.)

### 8.6 Asynchronous enqueue

Every enqueue takes a wait-list and returns an event:

```c
typedef struct {
    size_t       struct_size;       // sizeof(vx_launch_info_t)
    const void*  next;
    vx_buffer_h  kernel;            // loaded ELF; entry PC = buffer base address
    vx_buffer_h  args;              // kernel argument block
    uint32_t     ndim;              // 1, 2, or 3
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     lmem_size;
} vx_launch_info_t;

vx_result_t vx_enqueue_launch (vx_queue_h q,
                                 const vx_launch_info_t* info,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event /* nullable */);

vx_result_t vx_enqueue_copy   (vx_queue_h q,
                                 vx_buffer_h dst, uint64_t dst_off,
                                 vx_buffer_h src, uint64_t src_off,
                                 uint64_t     size,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event);

vx_result_t vx_enqueue_read   (vx_queue_h q,
                                 void* host_dst, vx_buffer_h src,
                                 uint64_t src_off, uint64_t size,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event);

vx_result_t vx_enqueue_write  (vx_queue_h q,
                                 vx_buffer_h dst, uint64_t dst_off,
                                 const void* host_src, uint64_t size,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event);

vx_result_t vx_enqueue_barrier(vx_queue_h q,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event);

/* Raw DCR enqueue — low-level escape hatch (§8.10). Prefer typed
 * state objects from per-block helper headers (vortex_tex.h,
 * vortex_raster.h, …) when one exists for the block you are
 * programming. */
vx_result_t vx_enqueue_dcr_write(vx_queue_h q,
                                 uint32_t addr, uint32_t value,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event);

vx_result_t vx_enqueue_dcr_read (vx_queue_h q,
                                 uint32_t addr, uint32_t* host_dst,
                                 uint32_t          n_wait_events,
                                 const vx_event_h* wait_events,
                                 vx_event_h*       out_event);
```

`vx_enqueue_barrier` with no wait list is OpenCL's `clEnqueueBarrier` —
ordering point in the queue. With a wait list it's
`clEnqueueBarrierWithWaitList` — drain all enqueued work *and* wait on
external events.

`vx_enqueue_dcr_{write,read}` expand to one `CMD_DCR_WRITE` /
`CMD_DCR_READ` in the ring buffer (§6.5). These are the documented
escape hatch for experimental hardware blocks, perf-counter setup, and
backends bringing up new functionality before a typed state object
exists for it. Mainstream user code should reach for the typed
state-object helper headers instead (§8.10).

### 8.7 Events

Events are produced by enqueue calls and consumed by waits. The runtime
also exposes user events for host-driven signalling:

```c
typedef enum {
    VX_EVENT_STATUS_QUEUED      = 0,
    VX_EVENT_STATUS_SUBMITTED   = 1,
    VX_EVENT_STATUS_RUNNING     = 2,
    VX_EVENT_STATUS_COMPLETE    = 3,
    VX_EVENT_STATUS_ERROR       = 4,
} vx_event_status_e;

vx_result_t vx_user_event_create  (vx_device_h dev, vx_event_h* out);
vx_result_t vx_user_event_signal  (vx_event_h ev, vx_result_t status);

vx_result_t vx_event_retain       (vx_event_h ev);
vx_result_t vx_event_release      (vx_event_h ev);

vx_result_t vx_event_status       (vx_event_h ev, vx_event_status_e* out);
vx_result_t vx_event_wait_all     (uint32_t n, const vx_event_h* evs,
                                     uint64_t timeout_ns);
vx_result_t vx_event_get_profiling(vx_event_h ev, vx_profile_info_t* out);
```

Mapping to standard programming models:

- OpenCL `cl_command_queue` (in-order) → `vx_queue_h`
- OpenCL `cl_event`                    → `vx_event_h`
- OpenCL `clCreateUserEvent`           → `vx_user_event_create`
- OpenCL `clSetUserEventStatus`        → `vx_user_event_signal`
- OpenCL `clGetEventProfilingInfo`     → `vx_event_get_profiling`
- CUDA `cudaStream_t`                  → `vx_queue_h`
- CUDA `cudaEvent_t`                   → `vx_event_h` (one-shot per enqueue)
- CUDA `cudaStreamWaitEvent`           → pass event in next enqueue's wait list
- HIP streams                          → same as CUDA

### 8.8 Implementation sketch

- A `vx_queue` owns: pinned ring buffer, head/tail slot, completion slot,
  per-queue 64-bit seqnum counter, a doorbell coalescer.
- A `vx_event` is `{ host_addr, expected_value, refcount, source_queue }`.
  At enqueue, the runtime allocates the next seqnum on the queue, emits
  `CMD_EVENT_SIGNAL(host_addr, seqnum)`, and stamps the event.
- An enqueue with a non-empty wait list emits one `CMD_EVENT_WAIT` per
  external event (events from this same queue are subsumed by in-order
  semantics and skipped). For long wait lists the runtime may insert a
  single `CMD_EVENT_WAIT` against a synthetic merged event to keep the
  ring fan-in bounded — open question for v1.
- `vx_event_wait_all` reads the 8 B host slot for each event with
  acquire semantics. No device round-trip.
- `vx_event_get_profiling` returns the 32 B record `VX_cp_profiling`
  wrote, converting cycles → ns using `CP_CYCLE_FREQ_HZ` (§6.10).

### 8.9 Worked example (vortex2.h)

```c
vx_device_h dev;
vx_device_open(0, &dev);                        /* vortex2.h */

vx_buffer_h kernel, args, dev_in, dev_out;
vx_buffer_create(dev, KERNEL_SIZE, VX_MEM_READ,       &kernel);
vx_buffer_create(dev, ARGS_SIZE,   VX_MEM_READ,       &args);
vx_buffer_create(dev, N,           VX_MEM_READ_WRITE, &dev_in);
vx_buffer_create(dev, N,           VX_MEM_READ_WRITE, &dev_out);
/* … upload kernel ELF into `kernel` and arg block into `args` … */

vx_queue_info_t qi = {
    .struct_size = sizeof(qi),
    .priority    = VX_QUEUE_PRIORITY_NORMAL,
    .flags       = VX_QUEUE_PROFILING_ENABLE,
};
vx_queue_h compute_q, copy_q;
vx_queue_create(dev, &qi, &compute_q);
vx_queue_create(dev, &qi, &copy_q);

vx_event_h h2d_done, kernel_done, d2h_done;

vx_enqueue_write (copy_q, dev_in, 0, host_in, N,
                  0, NULL, &h2d_done);

vx_launch_info_t li = {
    .struct_size = sizeof(li),
    .kernel      = kernel,  .args = args,
    .ndim        = 1,
    .grid_dim    = { grid,  1, 1 },
    .block_dim   = { block, 1, 1 },
    .lmem_size   = 0,
};
vx_enqueue_launch(compute_q, &li,
                  1, &h2d_done, &kernel_done);

vx_enqueue_read  (copy_q, host_out, dev_out, 0, N,
                  1, &kernel_done, &d2h_done);

vx_event_wait_all(1, &d2h_done, /*timeout_ns=*/ UINT64_MAX);

vx_profile_info_t pi;
vx_event_get_profiling(kernel_done, &pi);
/* pi.start_ns, pi.end_ns report device-side kernel timing. */

vx_event_release(h2d_done);
vx_event_release(kernel_done);
vx_event_release(d2h_done);
vx_queue_release(copy_q);
vx_queue_release(compute_q);
vx_buffer_release(dev_in);
vx_buffer_release(dev_out);
vx_buffer_release(args);
vx_buffer_release(kernel);
vx_device_release(dev);
```

The DAG is exactly what the lock-step runtime cannot express. Device
open comes from `vortex.h`; buffers, queues, events, async enqueue,
and profiling all come from `vortex2.h` under a consistent `vx_*`
naming scheme. No context object, no kernel object, no state-object
catalog — the runtime stays minimal.

### 8.10 Layering: where everything else lives

vortex2.h is intentionally tiny. Programming-model conveniences,
fixed-function state catalogs, command-buffer recording, pipeline
caches, descriptor sets, and high-level API surfaces all live above
it. The shape:

```
┌────────────────────────────────────────────────────────────────────┐
│  Application / language runtime                                    │
│  (user C/C++ code, SYCL, Kokkos, OpenMP target, …)                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│  Upper-layer API translators (one library per API surface)         │
│                                                                    │
│   ┌────────────┐  ┌─────────────┐  ┌────────────┐  ┌────────────┐  │
│   │  POCL      │  │ Vulkan-on-  │  │  CUDA-on-  │  │  GL-on-    │  │
│   │ (OpenCL)   │  │   Vortex    │  │   Vortex   │  │  Vortex    │  │
│   └─────┬──────┘  └──────┬──────┘  └─────┬──────┘  └─────┬──────┘  │
│         │                │               │                │        │
│   ┌─────┴─────┐    ┌─────┴─────┐                                   │
│   │ chipStar  │    │ HIP-on-   │                                   │
│   │ (HIP /OCL)│    │  Vortex   │                                   │
│   └─────┬─────┘    └─────┬─────┘                                   │
│         │ Owns: contexts, pipeline objects, command buffers,       │
│         │ descriptor sets, sub-buffers, refcount maps over         │
│         │ inherited handles, OpenCL/Vulkan/CUDA enums, etc.        │
└─────────┴──────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│  Optional per-block helper headers (built on vortex2.h)            │
│                                                                    │
│   vortex_tex.h     — TEX DCR programming + typed state objects     │
│   vortex_raster.h  — RASTER state objects                          │
│   vortex_om.h      — OM blend/depth state objects                  │
│   vortex_dxa.h     — DXA descriptor objects                        │
│                                                                    │
│  Each helper is a thin C library over vx_enqueue_dcr_write that    │
│  encapsulates per-block DCR layout. Upper layers include the       │
│  helpers for the blocks they care about; the runtime does not.     │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│  vortex2.h  — minimal async runtime (this proposal)                │
│   device + queues + events + async enqueue + raw DCR enqueue       │
│  ~22 functions, no programming-model abstractions                  │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│  vortex.h   — legacy synchronous wrapper                           │
│   simple single-queue blocking API for callers who want it         │
│  (re-implemented over vortex2.h in phase 8)                        │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                       CP hardware (RTL)
```

**Per-block helper headers** are the only place fixed-function DCR
layouts are encoded in software. They are designed and owned by the
proposals that own the corresponding RTL:

- [gfx_migration_proposal.md](gfx_migration_proposal.md) owns
  `vortex_tex.h`, `vortex_raster.h`, `vortex_om.h`.
- [dxa_worker_rtl_redesign_proposal.md](dxa_worker_rtl_redesign_proposal.md)
  owns `vortex_dxa.h`.

Each helper exposes typed state-object constructors (e.g.
`vx_tex_state_create`) that compile the user's configuration into a
small DCR-write packet, plus a binding function that emits the packet
via `vx_enqueue_dcr_write` into a queue ahead of a launch. Upper
layers (POCL with the cl_khr_image extension, a future Vulkan ICD,
etc.) include the helper headers they need; the rest of the runtime
is unaware.

**Why this layering is the right shape:**

- vortex2.h compiles in milliseconds, has a tiny API surface to
  audit, and never needs to change when a new HW block is added.
- Per-block knowledge lives with the proposal that owns the HW. No
  cross-coupling, no "one giant runtime knows everything" growth.
- Every upper-layer API surface (OpenCL, Vulkan, CUDA, HIP, OpenGL)
  picks the abstractions its programming model needs and implements
  them in its own code. They share the runtime primitives, not the
  abstractions.
- Raw `vx_enqueue_dcr_{write,read}` in vortex2.h is the universal
  escape hatch — any upper layer or helper can program any DCR
  without depending on per-block helper headers.

### 8.11 Complete `vortex2.h` API surface

For at-a-glance review, every function, type, enum, struct, and macro
introduced by `vortex2.h` in one place. 32 functions total. Inherited
declarations from `vortex.h` (`vx_device_h`, `vx_buffer_h`,
`VX_CAPS_*`, `VX_MEM_*`, `vx_mpm_query`, `vx_upload_kernel_*`, etc.)
are not repeated here.

```c
/* ====================================================================
 * vortex2.h — minimal async runtime for the Vortex Command Processor
 * ==================================================================== */

#include <vortex.h>          /* inherits vx_device_h, vx_buffer_h, VX_CAPS_*,
                                VX_MEM_*, vx_mpm_query, vx_upload_*, ... */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Opaque handles introduced by vortex2.h ----------------------- */
typedef struct vx_queue* vx_queue_h;
typedef struct vx_event* vx_event_h;

/* ----- Result type -------------------------------------------------- */
typedef enum {
    VX_SUCCESS = 0,
    VX_ERR_INVALID_HANDLE,
    VX_ERR_INVALID_INFO,
    VX_ERR_INVALID_VALUE,
    VX_ERR_OUT_OF_HOST_MEMORY,
    VX_ERR_OUT_OF_DEVICE_MEMORY,
    VX_ERR_DEVICE_LOST,
    VX_ERR_TIMEOUT,
    VX_ERR_EVENT_FAILED,
    VX_ERR_NOT_SUPPORTED,
    VX_ERR_INTERNAL,
} vx_result_t;

const char* vx_result_string(vx_result_t r);

/* ----- Enums -------------------------------------------------------- */
typedef enum {
    VX_QUEUE_PRIORITY_LOW    = 0,
    VX_QUEUE_PRIORITY_NORMAL = 1,
    VX_QUEUE_PRIORITY_HIGH   = 2,
} vx_queue_priority_e;

typedef enum {
    VX_EVENT_STATUS_QUEUED    = 0,
    VX_EVENT_STATUS_SUBMITTED = 1,
    VX_EVENT_STATUS_RUNNING   = 2,
    VX_EVENT_STATUS_COMPLETE  = 3,
    VX_EVENT_STATUS_ERROR     = 4,
} vx_event_status_e;

/* ----- Macros ------------------------------------------------------- */
#define VX_QUEUE_PROFILING_ENABLE  (1u << 0)

/* ----- Versioned create-info structs -------------------------------- */
typedef struct {
    size_t                struct_size;
    const void*           next;
    vx_queue_priority_e   priority;
    uint32_t              flags;
} vx_queue_info_t;

typedef struct {
    size_t       struct_size;
    const void*  next;
    vx_buffer_h  kernel;            /* loaded ELF; entry PC = buffer base */
    vx_buffer_h  args;              /* kernel argument block */
    uint32_t     ndim;              /* 1, 2, or 3 */
    uint32_t     grid_dim [3];
    uint32_t     block_dim[3];
    uint32_t     lmem_size;
} vx_launch_info_t;

typedef struct {
    uint64_t queued_ns;
    uint64_t submit_ns;
    uint64_t start_ns;
    uint64_t end_ns;
} vx_profile_info_t;

/* ====================================================================
 * Device  (6 functions)
 * ==================================================================== */
vx_result_t vx_device_count       (uint32_t* out_count);
vx_result_t vx_device_open        (uint32_t index, vx_device_h* out);
vx_result_t vx_device_retain      (vx_device_h dev);
vx_result_t vx_device_release     (vx_device_h dev);
vx_result_t vx_device_query       (vx_device_h dev, uint32_t caps_id,
                                   uint64_t* out_value);
vx_result_t vx_device_memory_info (vx_device_h dev,
                                   uint64_t* free, uint64_t* used);

/* ====================================================================
 * Buffer  (8 functions)
 * ==================================================================== */
vx_result_t vx_buffer_create  (vx_device_h dev, uint64_t size, uint32_t flags,
                               vx_buffer_h* out);
vx_result_t vx_buffer_reserve (vx_device_h dev, uint64_t address,
                               uint64_t size, uint32_t flags,
                               vx_buffer_h* out);
vx_result_t vx_buffer_retain  (vx_buffer_h buf);
vx_result_t vx_buffer_release (vx_buffer_h buf);
vx_result_t vx_buffer_address (vx_buffer_h buf, uint64_t* out_addr);
vx_result_t vx_buffer_access  (vx_buffer_h buf, uint64_t offset,
                               uint64_t size, uint32_t flags);
vx_result_t vx_buffer_map     (vx_buffer_h buf, uint64_t offset, uint64_t size,
                               uint32_t flags, void** out_host_ptr);
vx_result_t vx_buffer_unmap   (vx_buffer_h buf, void* host_ptr);

/* ====================================================================
 * Queue  (5 functions)
 * ==================================================================== */
vx_result_t vx_queue_create   (vx_device_h dev, const vx_queue_info_t* info,
                               vx_queue_h* out);
vx_result_t vx_queue_retain   (vx_queue_h q);
vx_result_t vx_queue_release  (vx_queue_h q);
vx_result_t vx_queue_flush    (vx_queue_h q);                       /* ring doorbell */
vx_result_t vx_queue_finish   (vx_queue_h q, uint64_t timeout_ns);  /* = clFinish */

/* ====================================================================
 * Async enqueue  (7 functions)
 *
 * Every enqueue takes a wait-list and returns an event for the work
 * just submitted. out_event may be NULL if the caller does not need
 * to observe completion of this particular command.
 * ==================================================================== */
vx_result_t vx_enqueue_launch    (vx_queue_h q,
                                  const vx_launch_info_t* info,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_copy      (vx_queue_h q,
                                  vx_buffer_h dst, uint64_t dst_off,
                                  vx_buffer_h src, uint64_t src_off,
                                  uint64_t    size,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_read      (vx_queue_h q,
                                  void* host_dst,
                                  vx_buffer_h src, uint64_t src_off,
                                  uint64_t    size,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_write     (vx_queue_h q,
                                  vx_buffer_h dst, uint64_t dst_off,
                                  const void* host_src,
                                  uint64_t    size,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_barrier   (vx_queue_h q,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_dcr_write (vx_queue_h q,
                                  uint32_t addr, uint32_t value,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

vx_result_t vx_enqueue_dcr_read  (vx_queue_h q,
                                  uint32_t addr, uint32_t* host_dst,
                                  uint32_t          n_wait_events,
                                  const vx_event_h* wait_events,
                                  vx_event_h*       out_event);

/* ====================================================================
 * Events  (7 functions)
 * ==================================================================== */
vx_result_t vx_user_event_create   (vx_device_h dev, vx_event_h* out);
vx_result_t vx_user_event_signal   (vx_event_h ev, vx_result_t status);

vx_result_t vx_event_retain        (vx_event_h ev);
vx_result_t vx_event_release       (vx_event_h ev);

vx_result_t vx_event_status        (vx_event_h ev, vx_event_status_e* out);
vx_result_t vx_event_wait_all      (uint32_t n, const vx_event_h* evs,
                                    uint64_t timeout_ns);
vx_result_t vx_event_get_profiling (vx_event_h ev, vx_profile_info_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif
```

**Function count, by family:**

| Family   | Count | Functions                                                                 |
|----------|-------|---------------------------------------------------------------------------|
| Device   | 6     | count, open, retain, release, query, memory_info                          |
| Buffer   | 8     | create, reserve, retain, release, address, access, map, unmap             |
| Queue    | 5     | create, retain, release, flush, finish                                    |
| Enqueue  | 7     | launch, copy, read, write, barrier, dcr_write, dcr_read                   |
| Events   | 7     | user_create, user_signal, retain, release, status, wait_all, get_profiling |
| Misc     | 1     | result_string                                                              |
| **Total**| **34**|                                                                           |

Plus 2 new opaque handle types (`vx_queue_h`, `vx_event_h`), 3 enums
(`vx_result_t`, `vx_queue_priority_e`, `vx_event_status_e`), 3 structs
(`vx_queue_info_t`, `vx_launch_info_t`, `vx_profile_info_t`), and 1
macro (`VX_QUEUE_PROFILING_ENABLE`).

Everything else — contexts, kernel objects, pipelines, command
buffers, descriptor sets, sub-buffers, image objects, sampler state,
rasterizer state, output-merger state, DXA descriptors, CL-event
profiling helpers, etc. — lives in upper-layer translators or
per-block helper headers (§8.10).

## 9. Legacy `vortex.h` compatibility and 1.0 → 2.0 mapping

`vortex.h` continues to expose the existing synchronous calls
(`vx_dev_open`, `vx_mem_alloc`, `vx_copy_to_dev`, `vx_start`,
`vx_ready_wait`, etc.) with unchanged signatures and unchanged
semantics. In v1 these continue to drive the legacy MMIO command path
that the CP-aware AFU keeps available as a compatibility mode — the
existing AP_CTRL / single-command MMIO interface is *not* removed from
the AFU; the CP simply sits in parallel and is engaged only when the
new `vortex2` runtime opens a queue.

Phase 8 of the migration plan (§13) re-implements `vortex.h` as a thin
shim over `vortex2.h`, at which point the legacy MMIO path can be
retired from the AFU.

### 9.1 1.0 → 2.0 function mapping

The complete legacy `vortex.h` surface translated to its `vortex2.h`
equivalent. Where a legacy call has no direct 2.0 equivalent (because
the new model is fundamentally different), the "2.0 equivalent" column
gives the canonical replacement sequence.

| `vortex.h` (1.0)            | `vortex2.h` (2.0) equivalent                                      | Notes                                                       |
|-----------------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| `vx_dev_open`               | `vx_device_open(0, &dev)`                                         | 1.0 always opens device 0; 2.0 takes an explicit index.     |
| `vx_dev_close`              | `vx_device_release(dev)`                                          | Release the caller's primary reference; closes at refcount 0. |
| `vx_dev_caps`               | `vx_device_query`                                                 | Same `VX_CAPS_*` constants; new returns `vx_result_t`.      |
| `vx_mem_alloc`              | `vx_buffer_create`                                                | Same parameters, just consistent `vx_buffer_*` naming.      |
| `vx_mem_reserve`            | `vx_buffer_reserve`                                               | Same parameters.                                            |
| `vx_mem_free`               | `vx_buffer_release(buf)`                                          | Releases caller's primary reference.                        |
| `vx_mem_access`             | `vx_buffer_access`                                                | Same parameters.                                            |
| `vx_mem_address`            | `vx_buffer_address`                                               | Same parameters.                                            |
| `vx_mem_info`               | `vx_device_memory_info`                                           | Device-level heap query; relocated under device family.     |
| (no 1.0 equivalent)         | `vx_buffer_map` / `vx_buffer_unmap`                               | Zero-copy host mapping of device-visible buffers. New in 2.0; required by `clEnqueueMapBuffer` / `vkMapMemory` / `cudaHostGetDevicePointer` / `glMapBuffer`. |
| `vx_copy_to_dev`            | `vx_enqueue_write(default_queue, …)` + `vx_event_wait_all`        | Blocking 1.0 call = enqueue + wait on returned event.       |
| `vx_copy_from_dev`          | `vx_enqueue_read (default_queue, …)` + `vx_event_wait_all`        | Same shape.                                                 |
| `vx_start`                  | `vx_enqueue_launch(default_queue, &li, 0, NULL, &ev)`             | Caller fills `vx_launch_info_t` from previously-set DCRs.   |
| `vx_start_g`                | `vx_enqueue_launch(default_queue, &li, 0, NULL, &ev)`             | `vx_launch_info_t` carries ndim / grid / block / lmem natively. |
| `vx_ready_wait`             | `vx_queue_finish(default_queue, timeout)`                         | Per-queue wait, not device-wide.                            |
| `vx_dcr_write`              | `vx_enqueue_dcr_write(default_queue, addr, value, 0, NULL, NULL)` | DCR programming is enqueued; the legacy synchronous call is a wrapper that flushes. |
| `vx_dcr_read`               | `vx_enqueue_dcr_read (default_queue, addr, &val, 0, NULL, &ev)` + `vx_event_wait_all` | Real device read instead of the prototype's software shadow. |
| `vx_mpm_query`              | `vx_mpm_query`                                                    | Inherited unchanged; no `vortex2.h` rewrap.                 |
| `vx_flush_commands` (prototype only) | `vx_queue_flush(q)`                                      | Per-queue doorbell; legacy global flush is gone.            |
| `vx_upload_kernel_bytes`    | utility: stays in `vortex.h`                                      | Convenience over `vx_buffer_create` + `vx_enqueue_write`.   |
| `vx_upload_kernel_file`     | utility: stays in `vortex.h`                                      | Same.                                                       |
| `vx_upload_bytes`           | utility: stays in `vortex.h`                                      | Same.                                                       |
| `vx_upload_file`            | utility: stays in `vortex.h`                                      | Same.                                                       |
| `vx_check_occupancy`        | utility: stays in `vortex.h`                                      | Pure software helper.                                       |
| `vx_dump_perf`              | utility: stays in `vortex.h`                                      | Pure software helper over `vx_mpm_query`.                   |

"default_queue" above refers to a per-device implicit queue that the
`vortex.h` shim opens at `vx_dev_open` time and finishes/releases at
`vx_dev_close` time. Legacy callers never see the queue handle.

### 9.2 Constant / handle / type mapping

| `vortex.h` (1.0)            | `vortex2.h` (2.0) equivalent | Notes                                            |
|-----------------------------|------------------------------|--------------------------------------------------|
| `vx_device_h`               | same handle, inherited        | Type definition stays in `vortex.h`.            |
| `vx_buffer_h`               | same handle, inherited        | Type definition stays in `vortex.h`.            |
| `VX_CAPS_*`                 | inherited unchanged           | Used by `vx_device_query`.                      |
| `VX_ISA_*`                  | inherited unchanged           |                                                  |
| `VX_MEM_READ` / `_WRITE` / `_READ_WRITE` / `_PIN_MEMORY` | inherited unchanged | Used as `flags` in `vx_buffer_create`. |
| `VX_MAX_TIMEOUT`            | inherited unchanged           | Suitable for `vx_queue_finish` / `vx_event_wait_all` `timeout_ns` argument. |
| (no equivalent)             | `vx_queue_h`                  | New in 2.0.                                     |
| (no equivalent)             | `vx_event_h`                  | New in 2.0.                                     |
| `int` (return code)         | `vx_result_t` enum + `vx_result_string` | 2.0 uses a typed enum; 1.0 still returns `int`. |

### 9.3 Coexistence during transition

Both headers coexist in the same shared library and may be included in
the same translation unit (`vortex2.h` `#include`s `vortex.h`). During
the transition the two paths target the same hardware but through
different AFU surfaces:

| Caller                              | Header used  | Path through AFU                 |
|-------------------------------------|--------------|----------------------------------|
| POCL / chipStar (today)             | `vortex.h`   | Legacy MMIO command FSM          |
| New CP-aware POCL / chipStar backend| `vortex2.h`  | CP queues                        |
| SimX / rtlsim harnesses             | `vortex.h`   | Legacy MMIO command FSM          |
| In-tree tests (today)               | `vortex.h`   | Legacy MMIO command FSM          |
| New tests + perf demos              | `vortex2.h`  | CP queues                        |

At phase 8 (§13), `vortex.h` is re-implemented as a thin shim over
`vortex2.h`'s default queue, and the AFU's MMIO compatibility mode is
retired.

## 10. Reset, KMU, and the launch path

The prototype reset the entire GPU around every `CMD_RUN`. We drop that:

- KMU is configured by a sequence of `CMD_DCR_WRITE`s (PC, grid_dim,
  block_dim, lmem, warp_step, block_size, args).
- `CMD_LAUNCH` pulses a `start_evt` into the KMU's start input. KMU drains
  its grid, the GPU runs CTAs, KMU drops `busy` when done.
- The CP detects `busy` falling and retires `CMD_LAUNCH`. Subsequent
  commands on the same queue may include the next `CMD_DCR_WRITE` block
  for a fresh launch — no reset required.

This unblocks the multi-context KMU work tracked as phase 7 (§13): the
CP's launch path is already context-aware via `kmu_ctx_id` in
`CMD_LAUNCH`'s payload, even though v1 only ever uses ctx 0. When the
multi-context KMU lands, the same `CMD_LAUNCH` opcode will populate one
of N KMU descriptor slots rather than the single shared one — no change
to the command format or the CPE FSMs.

## 11. Build and configuration

New entries in `VX_config.toml`:

```
[cp]
VX_CP_ENABLE          = true        # build CP into the AFU
VX_CP_NUM_QUEUES      = 4           # also sets the number of CPEs (1 CPE per queue)
VX_CP_RING_SIZE_LOG2  = 16          # 64 KiB per queue
VX_CP_MAX_CMDS_PER_CL = 5
VX_CP_DMA_DEV_PORT    = "dedicated" # or "shared"
VX_CP_AXI_TID_WIDTH   = 6
VX_CP_PROFILE_DEFAULT = false       # default per-queue profile_en at queue create
```

There is intentionally **no separate `VX_CP_NUM_CPES` knob**: the CPE count
is locked to `VX_CP_NUM_QUEUES`. See §6.3 for the rationale.

Configure-script flags: `--enable-cp`, `--cp-num-queues=N`,
`--cp-ring-size=BYTES`, `--cp-profile-default`. The runtime backend is
selected exactly as today (`fpga_xrt`).

## 12. OpenCL 1.2 backend conformance

A primary objective of this proposal is to bring Vortex up to a level
where the **POCL backend** (and chipStar for HIP) can implement a
conformant OpenCL 1.2 surface on top of it. vortex2.h does not implement
OpenCL itself — POCL does, on top of vortex2.h's primitives. The table
below identifies which OpenCL 1.2 features need what from vortex2.h.

| OpenCL 1.2 requirement                          | v1 status   | vortex2.h primitive POCL uses to implement it                |
|-------------------------------------------------|-------------|--------------------------------------------------------------|
| `cl_context` (logical grouping)                 | upper-layer | POCL keeps `cl_context` in its own bookkeeping; vortex2.h has no context object. |
| `cl_command_queue` (in-order)                   | covered     | `vx_queue_h`; one CPE per queue; in-order is native.         |
| `cl_command_queue` (out-of-order)               | upper-layer*| POCL maps each OoO command to its own in-order `vx_queue_h`, expressing dependencies through events. No native OoO in the CP. |
| `clEnqueue*` asynchronous semantics             | covered     | Every `vx_enqueue_*` returns after recording into the ring buffer. |
| `cl_event` + `clWaitForEvents` + `clFinish`     | covered     | `vx_event_h` returned from each enqueue; `vx_event_wait_all`; `vx_queue_finish`. |
| Inter-command event dependencies (event lists)  | covered     | `wait_events` list on every `vx_enqueue_*` → `CMD_EVENT_WAIT` (§6.5). |
| User events (`clCreateUserEvent` / `clSetUserEventStatus`) | covered | `vx_user_event_create` / `vx_user_event_signal` (§8.7).   |
| Markers / barriers                              | covered     | `vx_enqueue_barrier`; `CMD_FENCE` (§6.5, §6.9).              |
| `CL_QUEUE_PROFILING_ENABLE`                     | covered     | `VX_QUEUE_PROFILING_ENABLE` queue flag → per-CPE `profile_en`; `F_PROFILE` flag; `VX_cp_profiling` writeback (§6.11). |
| `clGetEventProfilingInfo` (QUEUED/SUBMIT/START/END) | covered | `vx_event_get_profiling` (§8.7); 4 timestamps written per command (§6.11), converted ns ← cycles via `CP_CYCLE_FREQ_HZ` (§6.10). |
| Concurrent enqueue from multiple host threads   | covered     | Per-queue tail pointer is locked by POCL; HW is per-queue isolated. |
| Buffer / sub-buffer objects                     | covered     | `vx_buffer_*` family (§8.5); sub-buffers are POCL views over a `vx_buffer_h`. |
| Image objects                                   | upper-layer + helper | Built by POCL on top of `vortex_tex.h` (gfx proposal). |
| `clEnqueueMigrateMemObjects` (explicit migration) | covered    | Maps to `vx_enqueue_copy` / `read` / `write`.                |
| Native kernels                                  | n/a         | Vortex is not a CPU device.                                  |
| Built-in kernels                                | upper-layer | POCL concept.                                                |
| Sub-devices (`clCreateSubDevices`)              | out of scope| Requires GPU-side partitioning; v2.                          |
| Concurrent kernel execution on the device       | spec-permitted to serialize | Single-context KMU; v1 serializes. No conformance impact. |
| Multiple devices (`clCreateContextFromType`)    | out of scope  | One CP per Vortex instance.                                 |

(*) Out-of-order command queues are not natively supported by the CP. The
runtime exposes them by allocating multiple in-order HW queues on demand
and inserting `CMD_EVENT_WAIT`s for each event in the wait list. This is
spec-conformant — OpenCL does not require the implementation to *actually*
execute commands out of order, only to honor the explicit dependencies.

**Bottom line**: vortex2.h provides every primitive POCL needs to
implement a conformant minimal OpenCL 1.2 backend. Anything labeled
"upper-layer" is implemented by POCL in its own code over vortex2.h's
primitives — that is the intended division of responsibility, not a
gap. Features marked "out of scope" (sub-devices, multi-device) are
extensions or optional features a conformant minimal implementation
may omit. Profiling — which the prototype completely lacked — is a v1
must-have, not a follow-on.

## 13. Migration plan

The migration is staged so the tree stays buildable at every step.

| Phase | Scope                                                                                        | Branch              |
|-------|----------------------------------------------------------------------------------------------|---------------------|
| 0     | Land this proposal; lock terminology, DCR allocations, AXI interface contract, CPE-per-queue rule, two-header runtime plan (`vortex.h` legacy, `vortex2.h` new). | `feature_cp` (now)  |
| 1     | Make Vortex DCR bus req/rsp at the top level. Update XRT AFU to forward `dcr_rsp_*`. Land `sw/runtime/include/vortex2.h` skeleton (handles + result enum + empty impl). No CP yet. | `feature_cp`        |
| 2     | Land `rtl/cp/` skeleton: `VX_cp_core` with **one CPE** (NUM_QUEUES=1), `CMD_LAUNCH` + `CMD_DCR_WRITE` + `CMD_MEM_*` only. XRT shim wires it up. `vortex2.h`: device retain/release + `vx_buffer_*` family + queue create/finish + `vx_enqueue_write/read/launch` (no events yet). Legacy `vortex.h` `vx_mem_*` functions are reimplemented as thin wrappers over `vx_buffer_*`; AFU keeps its MMIO compatibility mode for legacy `vx_start` / `vx_ready_wait` callers. | `feature_cp`        |
| 3     | Scale to N CPEs + resource arbiters (KMU/DMA/DCR) + completion writeback. `vortex2.h`: events from enqueues, `vx_event_wait_all`, `vx_user_event_*`. | `feature_cp`        |
| 4     | Cross-queue waits (`CMD_EVENT_WAIT`), barriers, `CMD_DCR_READ`, `CMD_MEM_COPY`. Profiling unit + `F_PROFILE` flag + per-queue `profile_en`. `vortex2.h`: `vx_event_get_profiling`, `vx_enqueue_barrier`, `vx_enqueue_dcr_{read,write}`. **vortex2.h is feature-complete and minimal.** Per-block helper headers (`vortex_tex.h`, `vortex_raster.h`, `vortex_om.h`, `vortex_dxa.h`) land in their own proposals (see §15). POCL backend on top of vortex2.h reaches OpenCL 1.2 conformance (§12). | `feature_cp`        |
| 5     | Performance pass: doorbell coalescing, intra-CPE pipelining (DMA-behind-launch), head-writeback batching, AXI tag tuning. | `feature_cp`        |
| 6     | (Optional v1.1) Interrupt path through XRT `interrupt` port; runtime sleeps on interrupt instead of polling. | `feature_cp_irq`    |
| 7     | (Follow-on proposal) Multi-context KMU for true per-CTA concurrent kernel execution. `kmu_ctx_id` in `CMD_LAUNCH` becomes meaningful; KMU arbiter selects a slot rather than a single port. | TBD                |
| 8     | (Follow-on cleanup) Re-implement `vortex.h` as a thin shim over `vortex2.h`. Retire the AFU's MMIO compatibility mode once POCL/chipStar/tests/SimX/rtlsim have migrated. | TBD                |

Each phase is independently testable. SimX and rtlsim back-ends need no
changes for phases 0–4 since they don't go through the AFU; the runtime
keeps the old synchronous shims for them.

## 14. Open questions

1. **Interrupt vs. polling for v1.** Polling is simpler and works on any XRT
   shell. Interrupt support is significantly nicer for long-running kernels.
   Proposal defers interrupts to v1.1 — confirm.
2. ~~**DMA dedicated port vs. shared fabric default.**~~ **Resolved**:
   v1 default = `SHARED` (works on every shell, no shell-dependent
   surprises). `DEDICATED` opt-in via `--cp-dma-port=dedicated`; phase 5
   measurements decide whether to promote it to the default on
   multi-bank shells. See §6.6.
3. **Per-CPE intra-queue pipelining.** Each CPE today retires one command
   at a time and stalls its FSM while waiting on `vx_busy` for `CMD_LAUNCH`.
   Letting a single CPE issue a `CMD_MEM_*` while its own `CMD_LAUNCH` is
   still in flight (DMA-while-own-kernel-runs) is a free win — propose to
   land in phase 5 once basic correctness is in.
4. **Host-memory model for completion / event / profile slots.** We assume
   the host can pin 8 B / 32 B slots and the CP writes them via the AXI
   master with a write-response. On systems with weak ordering, the
   runtime's poll loop needs `std::atomic` / acquire-load semantics — to be
   documented in the runtime guide.
5. **Profiling cycle-counter source.** v1 uses the CP clock. If CP and
   GPU clocks differ (likely on FPGA), the conversion between
   `CMD_LAUNCH` START/END timestamps and any in-kernel `vx_get_clock()`
   value the user observes will diverge — runtime should document the
   policy. A future option: derive the profiling counter from the same
   clock the GPU uses, at the cost of a CDC.
6. **AXI tag-width sensitivity.** `VX_CP_AXI_TID_WIDTH` caps outstanding
   AXI requests across all CPEs + DMA + event_unit + completion +
   profiling. Need to characterize where it bottlenecks on each target
   shell.

## 15. References

- [docs/designs/command_processor_prototype.md](../designs/command_processor_prototype.md) — review of the OPAE prototype this proposal supersedes.
- [hw/rtl/VX_kmu.sv](../../hw/rtl/VX_kmu.sv) — KMU module the CP launches via.
- [hw/rtl/Vortex.sv](../../hw/rtl/Vortex.sv) — GPU top, currently DCR-write-only at top level (§6.7 extends to req/rsp).
- [hw/rtl/afu/xrt/VX_afu_wrap.sv](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) — current XRT AFU wrapper, target of the §7.1 rework.
- [VX_types.toml](../../VX_types.toml) — DCR address map; CP block reserves 0x080–0x0BF.
- [sw/runtime/include/vortex.h](../../sw/runtime/include/vortex.h) — legacy synchronous wrapper; preserved unchanged in v1, full 1.0 → 2.0 mapping in §9. Still the home of `vx_dev_open` / `vx_dev_close`, the `vx_mem_*` family (now thin wrappers over the `vx_buffer_*` family in vortex2.h), and `vx_mpm_query`.
- `sw/runtime/include/vortex2.h` (new) — minimal async runtime introduced by this proposal (§8). 34 functions across 6 families (full surface in §8.11). `#include`s `vortex.h` to share the `vx_*` namespace. Owns: device enumerate/open/refcount/query, the `vx_buffer_*` family (incl. zero-copy map/unmap), queues, events, async enqueue, raw DCR enqueue.
- **Per-block optional helper headers** (built on `vx_enqueue_dcr_write`, owned by the block's own proposal — §8.10):
  - `sw/runtime/include/vortex_tex.h`, `vortex_raster.h`, `vortex_om.h` — owned by [gfx_migration_proposal.md](gfx_migration_proposal.md).
  - `sw/runtime/include/vortex_dxa.h` — owned by [dxa_worker_rtl_redesign_proposal.md](dxa_worker_rtl_redesign_proposal.md).
- **Upper-layer API translators** (each is a separate library on top of vortex2.h; not in this proposal):
  - POCL OpenCL backend — owned by [pocl_vortex_v3_proposal.md](pocl_vortex_v3_proposal.md).
  - chipStar HIP/OpenCL backend — owned by [chipstar_on_vortex_proposal.md](chipstar_on_vortex_proposal.md).
  - HIP-on-Vortex direct backend — owned by [hip_support_proposal.md](hip_support_proposal.md).
  - Future Vulkan-on-Vortex, CUDA-on-Vortex, OpenGL-on-Vortex translators — separate proposals when they land.
- OpenCL 1.2 Specification (Khronos) — runtime semantics POCL implements on top of vortex2.h, scored in §12.
- CUDA Streams and Events; Vulkan timeline semaphores; HIP Streams — additional programming models that map cleanly onto vortex2.h primitives.

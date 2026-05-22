# Command Processor — full redesign

Status: **draft for review**
Supersedes: the unfinished tail of `afu_shell_cleanup_proposal.md` (its Phases 1–4-xrt
remain valid foundation work; Phase 4-opae and Phase 5 are replaced by this document).

---

## 1. Motivation — why a redesign

The Command Processor (CP) brought up in `afu_shell_cleanup` Phases 1–4 is
**functionally correct but architecturally wrong**. It was a bring-up — "make
the launch path work" — and it uses the command ring in a way that defeats the
entire purpose of having a command processor.

Concretely, today `Device::cp_submit_cl_()` does, **per command**:

1. `mem_upload(ring_addr + offset, cl, 64)` — a full **platform DMA transaction
   to move one 64-byte cache line** into a ring that lives in **device memory**.
2. Bump `Q_TAIL` (doorbell).
3. **Block** — the host CPU spins on `Q_SEQNUM` until *that one command* retires.

A single kernel launch is *N* `CMD_DCR_WRITE`s + `CMD_LAUNCH` + `CMD_CACHE_FLUSH`
— each one separately DMA'd, doorbelled, and **waited on**. The defects:

- **A DMA per 64-byte command.** DMA setup/teardown dwarfs the payload.
- **Synchronous, one command at a time.** No batching, no overlap.
- **The host CPU is *inside* the consumer loop** — it babysits processing and
  completion per command.
- **Net:** all of the CP's machinery (ring, fetch, completion, seqnums) for
  *none* of its benefit. Plain MMIO commands would be no worse.

It is also **XRT-bound**, and XRT cannot be the common layer: the target
**AMD V80** does not support XRT at all.

### Design principles (the fix)

1. **The command queue is a GPU-owned consumer structure.** The host CPU's only
   role is to **append** commands and ring a doorbell — *write-only*. The host
   never reads the queue, never processes it, never polls it.
2. **The CP is the sole reader and processor.** It fetches, decodes, executes,
   and retires commands autonomously and asynchronously.
3. **The queue lives in host memory.** Appending a command is a plain `memcpy`
   into host RAM — free and batchable. The CP fetches it directly.
4. **Submit-and-walk-away.** The host appends a *batch* (`[copy-in, launch,
   copy-out, …]`), rings *one* doorbell, and returns immediately.
5. **The CP owns DMA.** Host↔device data movement is a CP command, executed by
   the CP — so data copies sit *in* the command stream and the host need not
   orchestrate them.
6. **Completion is a separate, lightweight channel** — the CP writes a
   seqnum/fence to a host-memory slot (writeback) and/or raises an interrupt.

## 2. Functional description of the CP

### 2.1 Role

The CP is an autonomous front-end engine. It is the **single consumer** of one
or more host-resident command queues. It fetches command batches, decodes them,
dispatches each to an execution resource, retires them in program order, and
reports progress to the host through a writeback/interrupt channel — with **zero
host-CPU involvement** between doorbell and completion.

### 2.2 Interfaces (`cp_core`)

| Interface | Type | Purpose |
|---|---|---|
| `axil_ctrl` | AXI-Lite **slave** | Host configures the CP: per-queue ring base/size, doorbell (tail), completion-slot address, enable, interrupt control. Small register file. |
| `axi_host`  | AXI **master** | CP ↔ **host memory**: command-ring fetch, the host side of `CMD_MEM_*` DMA, completion writeback. |
| `axi_dev`   | AXI **master** | CP ↔ **device memory**: the device side of `CMD_MEM_*` DMA, and Vortex's GPU memory. |
| `gpu_if`    | bus | CP ↔ Vortex: DCR + `start`/`busy`. |
| `interrupt` | wire | completion interrupt to the platform. |

`cp_core` is **identical RTL on every platform**. The platform adapter only
wires these interfaces to the platform's physical fabrics (§6).

### 2.3 Command queue semantics

- A queue is a **ring in host memory**, sized a power of two. Programmable
  per-queue: `ring_base`, `ring_size`, `head_addr` (CP-published), `cmpl_addr`
  (CP-published seqnum), priority, enable.
- **Producer (host):** writes one or more command cache-lines at the tail, then
  writes `Q_TAIL` once (the doorbell). Purely append; never reads the ring.
- **Consumer (CP):** fetches `[head, tail)` over `axi_host`, in CL bursts,
  decodes and executes, advances `head`, and publishes `head` + retired
  `seqnum`.
- Multiple queues with priorities; one CP arbitrates across them.

### 2.4 Command set

| Opcode | Meaning |
|---|---|
| `CMD_NOP` | padding / ring sentinel |
| `CMD_LAUNCH` | launch a kernel |
| `CMD_DCR_WRITE` / `CMD_DCR_READ` | device-control-register config / read-back |
| `CMD_MEM_WRITE` / `CMD_MEM_READ` / `CMD_MEM_COPY` | **DMA** — host↔device / device↔device, **bursted** |
| `CMD_CACHE_FLUSH` | broadcast cache coherence flush |
| `CMD_EVENT_SIGNAL` / `CMD_EVENT_WAIT` | cross-queue synchronization |

`CMD_MEM_*` is now a **first-class, used** path (today it exists but is dead).

### 2.5 Execution model

`fetch (axi_host) → decode → dispatch to a resource → retire`. Resources:
KMU/launch, DMA (`VX_cp_dma`), DCR proxy, event unit. Commands within a queue
retire in program order; cross-queue ordering is expressed with events. A batch
is executed without re-involving the host.

### 2.6 Completion

The CP publishes the retired `seqnum` to the queue's host-memory `cmpl_addr`
(writeback) and pulses `interrupt`. The host learns "my work is done" by reading
*that one word* (cheap) or waiting on the IRQ — **never** by inspecting the
command queue.

### 2.7 DMA (the CP handles it)

`VX_cp_dma` executes `CMD_MEM_*`: it reads `src` and writes `dst`, where each
address is routed — by the platform adapter — to host memory (`axi_host`) or
device memory (`axi_dev`). It must be extended from the current single-cache-line
limit to **multi-beat bursts / arbitrary length** so a buffer transfer is one
command, not one-per-64-bytes.

## 3. Architecture — host-memory queue + CP-owned DMA

- **Ring in host memory.** The host appends with a `memcpy`; the CP fetches over
  `axi_host`. No per-command DMA, no device-memory bootstrap problem (the ring
  is never "loaded" — it is simply produced in host RAM).
- **Submission is O(1):** append the batch, write `Q_TAIL`, return. Total host
  cost per launch ≈ a few `memcpy`s + one MMIO write.
- **The CP does all data movement.** `upload`/`download` become `CMD_MEM_*`
  posted into the ring — so the host posts `[copy-in, launch, flush, copy-out]`
  as one batch and walks away; the CP enforces ordering.
- **Completion writeback** (a pattern proven by Coyote, §5): the CP updates a
  host-memory counter; the host polls a cacheable word or takes the IRQ.

## 4. `cp_core` interface design

Today `VX_cp_core` has `axil_s` + **one** `axi_m` (device-memory only). The
redesign splits the master into the **two explicit masters** of §2.2:

- `axi_host` — the CP's window onto host memory. Carries ring fetch, completion
  writeback, and the host side of DMA.
- `axi_dev` — the CP's window onto device memory.

`VX_cp_dma` bridges the two for `CMD_MEM_*`; `VX_cp_fetch` and
`VX_cp_completion` use `axi_host`. The CP is then **uniform** — every
platform-specific decision (what host memory *is*; what device memory *is* —
DDR4 vs HBM, channel count, see §6.2) lives entirely in the thin adapter.

## 5. The host runtime — "our own XRT", and what to reuse first

We will **not** use Xilinx XRT (cannot support V80; heavy; framework lock-in).
We need a thin userspace runtime that: allocates the host-memory ring + the
completion slot, appends commands, rings the doorbell, and checks completion.
The vortex2 runtime's `callbacks_t` (`mem_*` + `cp_mmio_*`) is already the
thin-platform seam — each platform implements those few callbacks.

### 5.1 Open-source alternatives to evaluate before building our own

| Project | What it is | Fit |
|---|---|---|
| **Coyote** (`fpgasystems/Coyote`) | Open-source "OS/shell for FPGAs": PCIe + XDMA driver, **shared virtual memory host↔FPGA**, **completion writeback to host counters**, reconfiguration. | **Best architectural reference.** It already solves host-memory access and completion writeback — exactly our model. Heavy as a whole (RDMA/TCP/multi-tenancy we don't need), but the shell + driver are an adopt-or-model candidate. |
| **LitePCIe** (`enjoy-digital/litepcie`) | BSD-licensed, small-footprint PCIe core + **DMA (scatter-gather)** + **Linux driver (mmap + DMA)** + MSI-X. UltraScale(+)/7-series/Cyclone5. | **Best transport candidate** for the UltraScale+ Alveo cards — open RTL *and* driver, permissive license, no Xilinx-XDMA entanglement. Versal support must be checked. |
| **OPAE** | Intel's open FPGA framework (`libopae` + the upstreamed DFL driver). | The clean **structural model** for our runtime (the user's reference point). |
| **Xilinx `dma_ip_drivers`** | XDMA/QDMA IP drivers (source-available). | The conventional path; QDMA spans UltraScale+ **and** Versal. |
| **DPDK / VFIO** | Userspace PCIe driver framework (`vfio-pci`, in-tree). | The model for a **kernel-module-free** userspace driver. |

### 5.2 Recommendation

- **Transport:** evaluate **LitePCIe** (UltraScale+ Alveo — open core + driver,
  BSD) and **QDMA/CPM** (Versal V80). Treat **Coyote** as the reference design
  for host-memory + writeback; adopt pieces rather than the whole framework.
- **Runtime:** a minimal userspace library structured like OPAE — build only the
  CP-specific glue (ring management, append, doorbell, completion). Prefer a
  **VFIO** userspace driver (no custom kernel module; see the host-memory /
  XRT-alternative analysis in the conversation record).
- Build our own only where nothing fits; do not reinvent the PCIe/DMA core.

## 6. Thin platform adapters

`cp_core` + Vortex are the **shared block**. Each platform is a **thin adapter**
that wires the three AXI interfaces to its physical fabric, and binds Vortex's
device-memory ports to the card's memory controllers.

### 6.1 The adapter's job

| Family | Cards | `axi_host` → | `axi_dev` / Vortex mem → |
|---|---|---|---|
| UltraScale+ Alveo | U50, U250, U280, U55C | LitePCIe / XDMA host path | HBM or DDR4 controllers |
| Versal | V80 | CPM / QDMA host path | HBM controllers |
| Embedded | KV260 | shared PS-DDR | shared PS-DDR (PL ports) |

`axi_host` is **uniform** — always a flat window onto host memory; the adapter
only changes *how* that window is realized (slave bridge / CCI-P / mmap). The
**device side** is where the real per-card variation lives (§6.2).

### 6.2 Device memory — DDR4 and HBM

Device memory is **heterogeneous** across the matrix; the design must treat it as
a first-class, build-time-parameterized concern (host memory does not vary —
this is device-side only):

| Card | Device memory | Channels / banks |
|---|---|---|
| U250 | DDR4, 64 GB | 4 channels |
| U280 | HBM2 8 GB (+ DDR4) | 32 HBM channels |
| U50  | HBM2 8 GB | 32 HBM channels |
| U55C | HBM2 16 GB | 32 HBM channels |
| V80  | HBM (Versal) | many channels |
| KV260 | DDR4 (shared PS-DDR) | 1 |

DDR4 = a few wide channels; HBM = many (16–32) narrow pseudo-channels. This
already flows through Vortex via `VX_CFG_PLATFORM_MEMORY_*` (`NUM_BANKS`,
`ADDR_WIDTH`, `DATA_SIZE`, `INTERLEAVE`, `PEAK_BW`) — the cache `*_MEM_PORTS`
derive from `min(cache_banks, PLATFORM_MEMORY_NUM_BANKS)`, so the channel count
parameterizes Vortex's whole memory subsystem. The redesign keeps this: the
adapter binds Vortex's `m_axi_mem_*` ports + the CP's `axi_dev` to the card's
controllers (DDR4 MIG vs HBM controller), in either per-bank or a merged-
interface mode (`PLATFORM_MERGED_MEMORY_INTERFACE`), as the card warrants.

### 6.3 Per-card configuration

`hw/syn/xilinx/xrt/platforms.mk` already encodes this per card — keyed on the
XRT `XSA` it sets `PLATFORM_MEMORY_NUM_BANKS` / `ADDR_WIDTH`, the merged-
interface flag, an address offset, and the v++ `--connectivity.sp` HBM-channel
mapping. The redesign **generalizes** that into a per-card config (no longer
`XSA`-keyed, since XRT is dropped) — one entry per card capturing:

- device memory type + channel/bank count + address width + interleave;
- the transport (LitePCIe / CPM-QDMA / mmap) and host-memory path;
- the channel/bank → controller connectivity mapping;
- peak bandwidth and any address offset.

The build selects a card → pulls its config → parameterizes both Vortex's memory
subsystem *and* the adapter. A new card = one new config entry — nothing in
`cp_core` or Vortex changes.

### 6.4 Finishing the cleanup

With the CP owning commands *and* DMA, the legacy AFU machinery is fully
removable — the opae `STATE_*` FSM and CCI-P command engine, all legacy
DCR/caps, the XRT/OPAE-specific shells collapse to adapters. This subsumes
`afu_shell_cleanup` Phase 4-opae and Phase 5.

## 7. Phased plan

- **Phase A — `cp_core` redesign (RTL).** Three-AXI `cp_core`; host-memory ring
  fetch; completion writeback; `VX_cp_dma` extended to bursts. Verify on rtlsim.
- **Phase B — host runtime.** Pick the transport (LitePCIe / VFIO); build the
  minimal append/doorbell/completion runtime; the host-memory ring.
- **Phase C — platform adapters.** UltraScale+ first — U250 (DDR4) and U55C
  (HBM) together, to exercise both device-memory paths early — then Versal V80,
  then KV260. Each card lands as one per-card config entry (§6.3).
- **Phase D — finish the cleanup.** Delete all legacy AFU + runtime command
  code.
- **Phase E — regression & sign-off** across the full card matrix.

## 8. Scope statement

Once Phases A–E are executed, **all CP-related work is complete**: an autonomous,
host-memory-queue command processor that owns DMA; a thin uniform runtime; thin
per-platform adapters covering U50/U250/U280/U55C/V80/KV260 (and extensible to
future cards); and no legacy AFU command machinery anywhere.

## 9. Open questions

| Id | Item |
|---|---|
| OQ-1 | Transport: adopt LitePCIe wholesale, adopt Coyote's shell, or a minimal VFIO driver of our own? Decide per card-family. |
| OQ-2 | Versal V80 host-memory path — CPM5 integrated DMA vs QDMA vs LitePCIe-on-Versal. |
| OQ-3 | Completion: writeback-only, interrupt-only, or both (host picks)? |
| OQ-4 | Multi-queue: how many, and the priority scheme. |
| OQ-5 | KV260 coherency — HPC/ACP coherent ports vs HP + explicit cache management. |
| OQ-6 | `axi_host`/`axi_dev` — one unified address space (adapter address-routes) vs two structural masters. §4 assumes two; confirm. |
| OQ-7 | Simulation: a per-platform sim, or rtlsim of `cp_core`+Vortex + a light host-memory model. |
| OQ-8 | HBM cards — per-channel `axi_dev` fan-out vs a merged interface (`PLATFORM_MERGED_MEMORY_INTERFACE`); the channel-interleave policy. |

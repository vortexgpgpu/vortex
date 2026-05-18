# gem5 v2 Backend Redesign — CP-First, Event-Driven Architecture

**Date:** 2026-05-18
**Status:** Draft for review (supersedes the v1 draft of this file)
**Author:** Blaise Tine

**Related:**
- [gem5_simx_v3_proposal.md](gem5_simx_v3_proposal.md) — the prior gem5 integration design (OPAE-style MMIO command FSM). This proposal supersedes its §3 (host/device protocol) and §4 (gem5 SimObject design).
- Upstream proposals on `origin/tinebp-patch-2`:
  - [command_processor_proposal.md](command_processor_proposal.md) — CP RTL architecture, vortex2.h API, OpenCL 1.2 mapping.
  - [cp_pure_v2_callbacks_proposal.md](cp_pure_v2_callbacks_proposal.md) — pure-v2 `callbacks_t` + `vortex::CommandProcessor` C++ class for simx/rtlsim.
- Upstream commits this proposal targets:
  - `086d26b` runtime: strip legacy launch_*/dcr_* from callbacks_t (Phase E — pure v2)
  - `8bc2564` runtime: add cp_mmio_write/read callbacks; wire all 4 backends
  - `16aa1ca` sim/common: software CommandProcessor C++ class + unit test
  - `04971a2` tests/regression: rewrite vecadd + sgemm from scratch on vortex2.h
  - `00aa42f` docs: pure-v2 callbacks_t + software CP for simx/rtlsim

**Decisions ratified before this draft (recorded for traceability):**
- D1 — **Data plane unified through CP**. All ordered host↔device transfers go through `CMD_MEM_*` in a CP queue. `callbacks_t::mem_upload/download/copy` are reserved for the dispatcher's cold-start writes (ring buffer seeding, kernel ELF preload). No second data path.
- D2 — **Single clock domain for CP + Vortex in v1**. CP and Vortex tick at the same rate; separate `ClockDomain`s are a follow-on.
- D3 — **In-process VRAM with DMA-port seam designed in**. CP and Vortex memory accesses go through a single accessor interface backed by the in-process `RAM` in v1; v2 swaps in a gem5 `SimpleMemory` via the SimObject's DMA port behind the same interface.
- D4 — **MAX_QUEUES = 4** in the gem5 PIO map (matches upstream `VX_CP_NUM_QUEUES` default). v1 host runtime exercises Q0 only; Q1–Q3 hardware is ready for future v2.h multi-queue work.

---

## 0. Purpose

The original gem5 backend ([gem5_simx_v3_proposal.md](gem5_simx_v3_proposal.md))
shipped an OPAE-style MMIO command FSM on the device and a synchronous
`vx_start`/`vx_ready_wait` runtime on the host. That was a deliberate
bring-up choice — it reused the OPAE protocol so we could validate the
gem5 SE-mode integration (PIO, DMA, cross-arch, ELF interp redirection)
in isolation from the broader v2 runtime work.

That bring-up is done. With upstream's pure-v2 `callbacks_t` landed,
keeping the OPAE FSM means:
- Two control planes coexist on the device (legacy CMD_* state machine
  AND the CP regfile), doubling the device-side surface.
- The host runtime carries dead code: `start()`, `ready_wait()`,
  `dcr_write/read`, and their MMIO poll loops, none of which the
  dispatcher will call again.
- The SimObject's polled-tick model misuses gem5's event scheduler:
  it ticks every clock period even when there's no work, and the host
  has to spin-wait on `Q_SEQNUM` between unsynchronized tick events.

This proposal is a **redesign**, not a port. It deletes the OPAE
control plane entirely, makes the CP a first-class event-driven gem5
device block, runs the Vortex `Processor` as a parallel gem5 event
chain, and rebuilds the host runtime as a thin shim over the CP
regfile. The end-state is structurally identical to how a real PCIe
GPU is modeled in gem5: a SimObject hosting an FSM that fetches
commands, dispatches DMAs, and kicks an asynchronous compute engine.

---

## 1. What changed upstream (verbatim summary)

The new pure-v2 `callbacks_t` ([sw/runtime/common/callbacks.h](../../sw/runtime/common/callbacks.h)
on `origin/tinebp-patch-2`) contains:

```
dev_open, dev_close
query_caps, memory_info
mem_alloc, mem_reserve, mem_free, mem_access
mem_upload, mem_download, mem_copy
cp_mmio_write, cp_mmio_read       // NEW — sole control plane
```

It no longer contains `start`, `ready_wait`, `dcr_write`, `dcr_read`.

The dispatcher in [sw/runtime/common/vx_device.cpp](../../sw/runtime/common/vx_device.cpp)
is now the single source of truth for CP command building. Every
kernel launch, every DCR program, every fence, every event becomes a
`CMD_*` descriptor written into a ring buffer in device memory, with
`cp_mmio_write(Q_TAIL_HI)` as the publish doorbell.

The CP itself ([sim/common/CommandProcessor.h](../../sim/common/CommandProcessor.h))
is a clock-ticked FSM with 5 hooks (note: not 6 — `vortex_dcr_read` is
handled by the CP's `dram_write` path back to the requesting `CMD_DCR_READ`'s
writeback address, not a dedicated hook):

```cpp
struct Hooks {
    std::function<void(uint64_t addr, void* dst, size_t bytes)> dram_read;
    std::function<void(uint64_t addr, const void* src, size_t bytes)> dram_write;
    std::function<void(uint32_t addr, uint32_t value)> vortex_dcr_write;
    std::function<void()> vortex_start;
    std::function<bool()> vortex_busy;
};
```

The CP regfile lives at offset `0x1000` on opae/xrt (the AFU shim adds
the base); the simulator-internal contract per `cp_pure_v2 §6.3` is
that `cp_mmio_write(off, val)` takes a **CP-internal** offset and each
backend wrapper adds its own base.

---

## 2. Design pillars

Six pillars define the redesign. Each is a deliberate departure from
the v1 OPAE-style design.

### 2.1 Single control plane: CP regfile MMIO only

The PIO range is sized for the CP regfile, period. No more legacy
OPAE CMD_TYPE / CMD_ARG / STATUS registers, no reserved 4 KiB hole,
no CMD_* state machine on the SimObject.

PIO layout:

```
PIO_BASE + 0x0000 .. 0x003F   CP global header (CTRL, STATUS, CAPS, IRQ)
PIO_BASE + 0x0040 .. 0x004F   CP profiling block (CYCLE_LO/HI, FREQ_HZ)
PIO_BASE + 0x0100 .. 0x01FF   CP per-queue regfile (4 × 0x40)
                              Q0: 0x0100..0x013F
                              Q1: 0x0140..0x017F
                              Q2: 0x0180..0x01BF
                              Q3: 0x01C0..0x01FF
PIO_BASE + 0x0200 .. end      reserved (future profiling per-queue, IRQ, …)
```

Total PIO size: **`0x0200`** (was `0x1000`).

The host wrapper `cp_mmio_write(off, val)` is:

```cpp
// sw/runtime/gem5/vortex.cpp
int cp_mmio_write(uint32_t off, uint32_t value) {
    driver_.mmio_write32(PIO_BASE_ADDR + off, value);
    return 0;
}
```

No `+0x1000` adjustment — gem5 doesn't need to match the AFU's `bit[12]`
control/data split because there is no AFU. CP regfile starts at
`PIO_BASE + 0x0`.

### 2.2 Single data plane: CP commands via `CMD_MEM_*`

`vx_enqueue_write/read/copy` (vortex2.h) emit `CMD_MEM_*` descriptors
into a queue's ring buffer. The CP executes them via its DMA hooks
against device VRAM. The same path serves user buffer transfers as
serves CP descriptor fetches — one accessor interface.

`callbacks_t::mem_upload/download/copy` are reserved for the
dispatcher's **cold-start** writes only: seeding ring buffers at queue
create, preloading kernel ELFs into device VRAM before they are
referenced by a `vx_launch_info_t`. The dispatcher does not use them
on the user-facing data plane.

In our gem5 setup this is essentially free: PIN_BASE_ADDR is
identity-mapped into the host process VA via `Process::map`, so
`mem_upload(dev_va, host_src, size)` is `memcpy(host_va_of_PIN_BASE +
dev_va, host_src, size)` — a regular store sequence that gem5
translates through the page table to the same physical bytes the
SimObject's `ram_` sees. No PIO trigger, no command descriptor, no
state machine.

### 2.3 Event-driven CP, not polled tick

The CP is a self-scheduling gem5 event:

```cpp
// sim/simx/gem5/vortex_gpgpu_dev.hh — sketch
class VortexGPGPU : public DmaDevice {
    EventFunctionWrapper cpTickEvent_;
    EventFunctionWrapper vortexTickEvent_;

    void cpTick();      // calls cp_.tick(); reschedules if cp_ has work
    void vortexTick();  // calls processor_.cycle(); reschedules if !is_done()
};
```

`cpTick()` calls `cp_.tick()` once and reschedules itself at
`clockEdge(Cycles(1))` **only if** the CP reports it still has work
(queue non-empty, command in flight, completion writeback pending).
Otherwise it returns and the CP is dormant.

CP wake-up paths:
- Host writes `Q_TAIL_HI` (queue doorbell) → `cp_mmio_write` schedules
  `cpTickEvent_` at the next clock edge if not already scheduled.
- Host writes `CP_CTRL.enable` → same.
- Vortex `tickEvent` observes `is_done()` and signals CP → CP wakes
  to retire `CMD_LAUNCH`.
- CP DMA completion → CP self-reschedules until the DMA retires.

**No bounded-tick-burst around `cp_mmio_*`.** No `VORTEX_USE_CP=0`
transparent-mode escape hatch. The CP is always real, always
event-driven, and the gem5 event queue arbitrates between CP, Vortex,
host CPU, and any other SimObjects in the system the way gem5
expects.

### 2.4 Vortex `Processor` as a parallel event chain

The Vortex GPU runs in its own gem5 event chain, scheduled by the
CP's `vortex_start` hook and torn down when `processor_.is_done()`:

```cpp
auto vortex_start = [this]() {
    if (!vortexTickEvent_.scheduled())
        schedule(vortexTickEvent_, clockEdge(Cycles(1)));
};

void VortexGPGPU::vortexTick() {
    processor_->cycle();
    if (processor_->any_running())
        schedule(vortexTickEvent_, clockEdge(Cycles(1)));
    // CP polls processor_->any_running() via the vortex_busy hook
    // from its own tick; no notification needed.
}
```

Both `cpTickEvent_` and `vortexTickEvent_` use the same `ClockDomain`
(D2). The gem5 event queue interleaves them with whatever simulated
host CPU work is happening at the same simulated time — that's where
the concurrency-realism win comes from. It is also what makes the
simulation faster overall: idle blocks (CP between commands, Vortex
between launches) do not consume tick events.

### 2.5 Single VRAM accessor — in-process for v1, DMA-port seam for v2

All device memory access — CP ring fetches, completion writebacks,
DMA payload reads/writes, Vortex's `MemSim` accesses — goes through
one accessor interface:

```cpp
// sim/simx/gem5/dev_mem.h
class DevMemAccessor {
public:
    virtual void read (uint64_t addr, void* dst, size_t bytes) = 0;
    virtual void write(uint64_t addr, const void* src, size_t bytes) = 0;
};

class InProcessDevMem : public DevMemAccessor { /* wraps simx::RAM */ };
class DmaPortDevMem   : public DevMemAccessor { /* wraps DmaDevice port */ };
```

v1: `Gem5Device` constructs an `InProcessDevMem` wrapping the existing
`simx::RAM`. CP `dram_read/write` hooks call through it. Vortex's
`MemSim::read/write` calls through it. PIN_BASE_ADDR's identity
mapping makes the host process see the same bytes.

v2 seam: replace `InProcessDevMem` with `DmaPortDevMem` (and back VRAM
with a gem5 `SimpleMemory` connected to the SimObject's DMA port).
**Zero changes to CP hooks, zero changes to Vortex memory code, zero
changes to host runtime.** That's the entire point of the abstraction
— the v2 path is a localized swap, not a rewrite.

This pillar is the reason "in-process for v1" is the right answer
(per D3): the accessor seam captures the design intent of v2 without
paying its cost upfront.

### 2.6 Multi-queue PIO map from day one

PIO map reserves 4 queue regfile slots (D4). v1 host runtime enables
Q0 only and the CP runs Q0 only. The 3 unused queue slots cost ~96
bytes of PIO range and let the hardware grow into v2.h multi-queue
without re-versioning the PIO layout (and without bumping the host
process's mmap size).

Picking 4 now means the gem5 device's regfile shape **matches
upstream `VX_CP_NUM_QUEUES = 4`**. The OPAE/XRT AFUs will instantiate
the same 4-queue CP; gem5 should not be the odd one out.

---

## 3. Address space layout

The full memory map after the redesign:

```
Host process VA (simulated, gem5 SE-mode)
  0x0000_0000_0000 .. 0x0000_0FFF_FFFF   normal heap / stack / mmap
  0x0000_1000_0000 .. 0x0000_1FFF_FFFF   PIN_BASE_ADDR (device VRAM,
                                          identity-mapped via Process::map)
  0x0000_2000_0000 .. 0x0000_2000_01FF   PIO_BASE_ADDR (CP regfile)
  0x0000_2000_0200 .. 0x0000_2FFF_FFFF   reserved (future PIO blocks)

gem5 SimObject PA
  PIN_BASE_ADDR .. PIN_BASE_ADDR + ram_size   device VRAM backing store
  PIO_BASE_ADDR .. PIO_BASE_ADDR + 0x0200      CP regfile (PIO range)
```

`PIN_BASE_ADDR` is the same VA on both sides because `Process::map`
identity-maps it into the simulated host process. The CP and Vortex
see the same physical bytes; the host process writes to them as
ordinary memory.

`PIO_BASE_ADDR` is **only** the CP regfile after this redesign. The
4 KiB OPAE legacy reserved block is gone.

---

## 4. Data flow walkthroughs

### 4.1 Cold start (queue create)

```
host runtime                                 gem5 SimObject + CP
─────────────────────────────────────────    ────────────────────────────
vx_device_open                                — (handle alloc; no IO)
  └─ callbacks->dev_open()
       └─ open libvortex-gem5-x86_64.so
       └─ vortex_gem5_dev_open(...)          construct Gem5Device:
                                               - new simx::RAM
                                               - new InProcessDevMem
                                               - new simx::Processor (wired to InProcessDevMem)
                                               - new vortex::CommandProcessor
                                                     (hooks: dram_read/write
                                                      → InProcessDevMem,
                                                      vortex_dcr_write → proc_.dcr_write,
                                                      vortex_start → schedule(vortexTickEvent_),
                                                      vortex_busy  → proc_.any_running())
                                               cpTickEvent_ deschduled (no work yet)
                                               vortexTickEvent_ deschduled

dispatcher: vx_queue_create
  └─ mem_alloc(ring_size, &ring_va)          allocate from device VRAM bump allocator
  └─ mem_alloc(8, &head_va)
  └─ mem_alloc(8, &cmpl_va)
  └─ mem_upload(ring_va, zeros, ring_size)   memcpy through PIN_BASE mapping
  └─ cp_mmio_write(Q0_RING_BASE_LO, ring_va lo)
  └─ cp_mmio_write(Q0_RING_BASE_HI, ring_va hi)
  └─ cp_mmio_write(Q0_HEAD_ADDR_LO/HI, head_va)
  └─ cp_mmio_write(Q0_CMPL_ADDR_LO/HI, cmpl_va)
  └─ cp_mmio_write(Q0_RING_SIZE_LOG2, log2)
  └─ cp_mmio_write(Q0_CONTROL, enable=1)     → SimObject PIO write handler:
                                                   cp_.mmio_write(off, val);
                                                   if cp_.has_work() and
                                                      !cpTickEvent_.scheduled():
                                                     schedule(cpTickEvent_,
                                                              clockEdge(Cycles(1)))
                                                   (CP has work because Q0 is now enabled
                                                    and may have a non-empty ring)
  └─ cp_mmio_write(CP_CTRL, enable=1)        — already enabled (idempotent)
```

### 4.2 Kernel launch (`vx_enqueue_launch`)

```
dispatcher                                   CP (in SimObject)
─────────────────────────────────────────    ────────────────────────────
mem_upload(ring_va + tail, CMD_DCR_WRITE     (rings now non-empty;
            for KMU PC, grid, block, args)   CP will fetch when scheduled)
mem_upload(ring_va + tail, CMD_LAUNCH)
cp_mmio_write(Q0_TAIL_LO, tail_lo)
cp_mmio_write(Q0_TAIL_HI, tail_hi)           → cpTickEvent_ schedule check fires;
                                              schedule for next clock edge

(host returns immediately — async by design;
 dispatcher does not block here. Polling
 happens later via cp_mmio_read(Q0_SEQNUM)
 from vx_event_wait_all.)
```

At next clock edge:

```
cpTick():
  cp_.tick()
    [CPE0 FSM: fetch ring head cache line
     via dram_read(ring_va, &cl, 64) → InProcessDevMem.read → ram_.read]
    [decode CMD_DCR_WRITE; route through vortex_dcr_write hook
     → proc_.dcr_write(addr, value); retire; bump seqnum]
    [dram_write(cmpl_va, &seqnum, 8); dram_write(head_va, &head, 8)]
  reschedule cpTickEvent_ (still has CMD_LAUNCH pending)

cpTick() (next):
  cp_.tick()
    [CPE0 FSM: fetch next CL, decode CMD_LAUNCH]
    [vortex_start() → schedule(vortexTickEvent_, clockEdge(Cycles(1)))]
    [CPE0 enters WAIT_FOR_BUSY state — polls vortex_busy() each tick]
  reschedule cpTickEvent_ (CMD_LAUNCH in flight)

… concurrent vortexTick() advances processor_.cycle() …
… until processor_.is_done(); on next CP tick:

cpTick():
  cp_.tick()
    [CPE0 sees !vortex_busy(); retire CMD_LAUNCH; bump seqnum]
    [dram_write(cmpl_va, &seqnum, 8)]
  CP has no more work; do NOT reschedule cpTickEvent_.
  vortexTickEvent_ stopped scheduling itself when is_done() became true.
  System is dormant.

Host poll:
  cp_mmio_read(Q0_SEQNUM_LO)                 → SimObject PIO read handler:
                                                   return cp_.mmio_read(off);
                                              (returns the retired seqnum;
                                               no tick burst needed because the
                                               cmpl_va writeback already happened
                                               in earlier cpTick())
```

The host never spins. The CP never idle-ticks. Vortex never runs past
`is_done()`. This is the win.

### 4.3 `vx_enqueue_write` (data plane through CP)

```
dispatcher                                   CP
─────────────────────────────────────────    ────────────────────────────
(host_src is in regular heap, not PIN_BASE   — note: dispatcher copies
 — so the dispatcher copies it into a       payloads into PIN_BASE first
 pinned device buffer it owns OR the         on backends that require it.
 caller used vx_buffer_map to write          For gem5 + Process::map this is
 directly into a host-mapped device          a memcpy through the mapped
 buffer)                                     pages.)

mem_upload(ring_va + tail, CMD_MEM_WRITE
            { src=pinned_host_va,            (pinned_host_va is in PIN_BASE
              dst=dev_va,                     so it's also a device PA)
              size=N })
cp_mmio_write(Q0_TAIL_HI, ...)               → CP schedules

cpTick():
  cp_.tick()
    [decode CMD_MEM_WRITE]
    [CP DMA FSM: dram_read(src, &buf, chunk)
                  dram_write(dst, &buf, chunk)
     looping over the transfer in 64 B steps]
    [retire; bump seqnum; cmpl writeback]
```

Both endpoints (`src`, `dst`) are in the same flat physical space
(PIN_BASE region). The CP's DMA FSM doesn't distinguish host vs.
device addresses — they're the same accessor.

---

## 5. Component design

### 5.1 `sim/simx/gem5/vortex_gpgpu.{cpp,h}` — device library

**Responsibilities:**
- Construct `RAM`, `Processor`, `CommandProcessor`, `InProcessDevMem`.
- Provide C ABI: `vortex_gem5_dev_open/close`, `vortex_gem5_cp_mmio_{read,write}`,
  `vortex_gem5_dram_access` (for SimObject DMA path → backing store),
  `vortex_gem5_cp_tick`, `vortex_gem5_vortex_tick`,
  `vortex_gem5_cp_has_work`, `vortex_gem5_vortex_busy`.
- Provide kernel preload for the Phase 3 standalone test (unchanged).

**Removed (all OPAE state machine carry-over):**
- `pending_cmd_`, `cmd_args_`, `dcr_rsp_`, `busy_` fields
- `mmio_write64`/`mmio_read64` and the CMD_TYPE dispatch
- `pop_pending_cmd`, `get_cmd_arg`, `set_busy`, `load_args`
- `process_cmd` and the `CMD_RUN`/`CMD_DCR_*`/`CMD_MEM_*` handlers
  (last one re-emerges inside the CP, not here)

**Added:**
- `cp_` member (`vortex::CommandProcessor`) with hooks bound in ctor.
- `dev_mem_` member (`std::unique_ptr<DevMemAccessor>`) — `InProcessDevMem`
  for v1.
- C-ABI surface for the SimObject (below).

### 5.2 `sim/simx/gem5/vortex_gpgpu_dev.{cc,hh}` — gem5 SimObject

**Class:** `VortexGPGPU : public DmaDevice` (unchanged from current).

**Members:**
- `pioAddr_, pioSize_ = 0x0200` (was `0x1000`).
- `EventFunctionWrapper cpTickEvent_;`
- `EventFunctionWrapper vortexTickEvent_;`
- `deviceHandle_` — opaque from device library.

**PIO `read(PacketPtr)`:**
```cpp
const Addr off = pkt->getAddr() - pioAddr_;
uint32_t value = 0;
abi_.cp_mmio_read(deviceHandle_, uint32_t(off), &value);
pkt->setLE<uint32_t>(value);
pkt->makeAtomicResponse();
return pioLatency_;
```

**PIO `write(PacketPtr)`:**
```cpp
const Addr off = pkt->getAddr() - pioAddr_;
const uint32_t value = pkt->getLE<uint32_t>();
abi_.cp_mmio_write(deviceHandle_, uint32_t(off), value);
maybeWakeCp();
pkt->makeAtomicResponse();
return pioLatency_;
```

**`maybeWakeCp()`:**
```cpp
if (abi_.cp_has_work(deviceHandle_) && !cpTickEvent_.scheduled())
    schedule(cpTickEvent_, clockEdge(Cycles(1)));
```

**`cpTick()`:**
```cpp
abi_.cp_tick(deviceHandle_);
if (abi_.cp_has_work(deviceHandle_))
    schedule(cpTickEvent_, clockEdge(Cycles(1)));
```

**`vortexTick()`:**
```cpp
abi_.vortex_tick(deviceHandle_);
if (abi_.vortex_busy(deviceHandle_))
    schedule(vortexTickEvent_, clockEdge(Cycles(1)));
```

**`vortex_start` hook callback (from device library into the SimObject):**
schedules `vortexTickEvent_` at next clock edge if not scheduled.
Implemented as a small C ABI: `vortex_gem5_set_start_handler(handle,
fn, ctx)` registered in `VortexGPGPU::init()`; the device library
calls it from the CP's `vortex_start` lambda.

### 5.3 `sw/runtime/gem5/vortex.cpp` — host runtime

**Responsibilities (shrunken):**
- `init` / `get_caps` / `mem_info` (unchanged)
- `mem_alloc` / `mem_reserve` / `mem_free` / `mem_access` (unchanged
  bump allocator + `PIN_BASE_ADDR` math)
- `mem_upload` / `mem_download` / `mem_copy` → `memcpy` through the
  PIN_BASE identity mapping (renamed from `upload`/`download`/`copy`
  in the v1 backend)
- `cp_mmio_write` → `driver_.mmio_write32(PIO_BASE_ADDR + off, val)`
- `cp_mmio_read` → `driver_.mmio_read32(PIO_BASE_ADDR + off, &val)`

**Removed:**
- `start()`, `ready_wait()`, `dcr_write()`, `dcr_read()` methods
- `MMIO_CMD_TYPE` / `MMIO_STATUS` constants and their poll loop
- `<sched.h>` and the `sched_yield()` back-off (no host poll loop —
  the dispatcher's `vx_event_wait_all` does its own polling against
  `Q_SEQNUM`)

**Kept:**
- The pinned-region setup, `PIN_BASE_ADDR`, `PIO_BASE_ADDR`,
  `mmio_fence()`, the bump allocator state.

### 5.4 `sw/runtime/gem5/driver.{cpp,h}` — pinned region + MMIO helpers

**Added:**
- `mmio_write32(uint64_t pa, uint32_t value)` — 4-byte store with
  fence. Implemented as `*reinterpret_cast<volatile uint32_t*>(pa) =
  value; mmio_fence();`.
- `mmio_read32(uint64_t pa, uint32_t* value)` — symmetric.

**Removed:**
- `mmio_write64` / `mmio_read64` — no caller after the redesign.
- The 64-bit MMIO path was a v1 choice for OPAE-style 8-byte argument
  registers. The CP regfile is 32-bit.

### 5.5 `sim/simx/gem5/VortexGPGPU.py` — SimObject Python binding

**Params:**
- `library = Param.String(...)` (unchanged)
- `kernel = Param.String("")` (Phase 3 standalone preload — unchanged)
- `pio_addr = Param.Addr(0x20000000)` (unchanged)
- `pio_size = Param.Addr(0x0200)` — **changed from 0x1000** to match
  the redesigned PIO map
- `pio_latency = Param.Latency("100ns")` (unchanged)
- `dma_latency = Param.Latency("100ns")` (unchanged)
- (new) `max_queues = Param.Unsigned(4)` — for forward compatibility;
  v1 enforces == 4

### 5.6 `sim/simx/Makefile` — build wiring

- Add `$(SIM_COMMON_DIR)/CommandProcessor.cpp` to the `USE_GEM5=1`
  source list (the device library links it; the SimObject indirects
  via the C ABI).

### 5.7 `sw/runtime/gem5/Makefile` — build wiring

- No source-list changes (the CommandProcessor lives in the device
  library, not the host runtime).
- `<sched.h>` include and any sched-related CFLAGS go away with the
  `sched_yield` poll loop.

---

## 6. Migration phasing

The whole redesign lands as **one commit** per the "substantial,
testable feature" rule. The internal phasing below is for validation
checkpoints during implementation, not for separate commits.

### Phase M1 — Merge upstream

- `git merge --no-commit --no-ff origin/tinebp-patch-2`
- Conflicts (all expected):
  - `sw/runtime/stub/Makefile` — keep HOST_ARCH; take new v2 dispatcher SRCS
  - Possibly `sw/runtime/common/callbacks.{h,inc}` — defer to upstream version
- Build will not compile until M2 + M3 complete. That is acceptable
  inside one commit; the commit is only created when M3 builds and
  passes regression.

### Phase M2 — Device-side redesign

- Add `sim/simx/gem5/dev_mem.{h,cpp}` (`DevMemAccessor` + `InProcessDevMem`).
- Rewrite `sim/simx/gem5/vortex_gpgpu.{cpp,h}`:
  - Delete OPAE state machine (per §5.1).
  - Embed `cp_` with hooks bound to `InProcessDevMem` + `proc_`.
  - Export the new C ABI.
- Rewrite `sim/simx/gem5/vortex_gpgpu_dev.{cc,hh}`:
  - PIO range shrinks to 0x0200.
  - `read`/`write` route 32-bit packets to `cp_mmio_{read,write}`.
  - `cpTickEvent_` + `vortexTickEvent_` self-scheduling per §2.3, §2.4.
  - `vortex_start` callback registration.
- Update `VortexGPGPU.py` (`pio_size = 0x0200`, `max_queues = 4`).
- `sim/simx/Makefile`: add `CommandProcessor.cpp`.

**Validation:** `make -C build/sim/simx USE_GEM5=1` builds.
`./hw/unittest/cp_sim/` unit test passes (smoke-tests the
CommandProcessor wiring; runnable without gem5 itself).

### Phase M3 — Host runtime redesign

- Rewrite `sw/runtime/gem5/vortex.cpp`:
  - Drop `start`/`ready_wait`/`dcr_*`.
  - Rename `upload`/`download`/`copy` → `mem_upload`/`mem_download`/`mem_copy`.
  - Add `cp_mmio_{read,write}` (3-line MMIO wrappers).
  - Drop `<sched.h>` and the poll loop.
- Add `mmio_{read,write}32` to `driver.{cpp,h}`; drop the 64-bit helpers.
- Build for x86_64 (default) and aarch64 (cross-compile via existing
  HOST_ARCH switch).

**Validation:**
- Hostless test (`ci/gem5_run_hostless_app.py`): PASSES.
  (No host runtime involvement.)
- `./ci/regression.sh --gem5`: PASSES — hello + vecadd + sgemm e2e on x86.
- `VORTEX_GEM5_ARM=1 ./ci/regression.sh --gem5`: PASSES — same 3 tests
  on aarch64. Total 6/6 PASS matches pre-redesign baseline.

### Phase M4 — Documentation

Update [docs/gem5_integration.md](../gem5_integration.md):
- Replace the OPAE protocol description with the CP regfile + ring
  buffer architecture.
- Update the 6 load-bearing invariants list:
  - Drop OPAE CMD_* invariants.
  - Add: "CP regfile is at `PIO_BASE + 0x0`, 0x200 bytes, 32-bit
    register stride."
  - Add: "PIN_BASE is identity-mapped via Process::map; host
    runtime's `mem_upload` is a direct memcpy."
  - Add: "CP and Vortex tick events self-schedule only while work is
    pending; idle is observable as cpTickEvent_ unscheduled."

Update [docs/proposals/gem5_simx_v3_proposal.md](gem5_simx_v3_proposal.md):
- Add a "Status: Superseded by gem5_v2_cp_migration_proposal" header
  on §3 (host/device protocol) and §4 (SimObject design).
- Keep §0–§2 (motivation, source-tree layout) and §5+ (testing,
  install, cross-arch) — those parts remain accurate.

---

## 7. Validation criteria

The redesign is complete when all of the following hold:

1. **`./ci/regression.sh --gem5`** PASSES on x86 (hello + vecadd +
   sgemm e2e). Total wall time ≤ 30 s (was 16 s pre-redesign; the
   event-driven design should be at least as fast because idle blocks
   no longer tick).
2. **`VORTEX_GEM5_ARM=1 ./ci/regression.sh --gem5`** PASSES on
   aarch64 (same 3 tests).
3. **No regression on non-gem5 builds.** `make -C build/sim/simx`
   (default), `USE_SST=1` still build and pass.
4. **No OPAE leftovers grep-detectable.** `grep -r CMD_TYPE\|CMD_RUN\|
   pending_cmd_\|get_cmd_arg sim/simx/gem5/ sw/runtime/gem5/` returns
   zero hits.
5. **Event-driven invariants hold.** Run a sim with a 100 ms idle gap
   between two enqueues; verify (via debug log) that `cpTickEvent_`
   is unscheduled during the gap and that the host CPU advances
   unhindered.
6. **PIO map size matches design.** `pio_size = 0x0200` exposed in
   `VortexGPGPU.py`; host runtime never writes outside that range.

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | CP `vortex_start` hook needs to schedule a gem5 event from inside a hook called during PIO write handling. gem5 SimObjects can `schedule()` from anywhere, but only from the gem5 thread. Verify the C ABI doesn't route the hook from a different thread. | Hooks are bound at construction; called from `cp_.tick()` which is called from `cpTick()` which is itself a gem5 event handler — same thread. No issue. |
| R2 | `vortexTick()` advancing `processor_.cycle()` per Vortex clock period is slow if the cycle()-per-tick ratio is high (a Vortex clock period is shorter than a CPU-host instruction time). | Match Vortex's `ClockDomain` to a realistic Vortex frequency (1 GHz). gem5 only schedules events at actual clock edges; the per-tick cost is one C++ function call. Acceptable. |
| R3 | New vecadd/sgemm tests (rewritten on vortex2.h upstream) may use features (events, queue priority) we don't validate end-to-end on gem5. | M3 validation surfaces this. If a test uses an unsupported vortex2.h primitive, file a follow-up; M3 acceptance is contingent on the existing 3-test matrix passing. |
| R4 | `DevMemAccessor` interface change forces a Vortex `MemSim` rewrite. | `MemSim` already takes a memory backend. v1 wires it to `InProcessDevMem` which delegates to `simx::RAM` — same backing buffer as today. Zero code change in Vortex itself. |
| R5 | The Phase 3 standalone test loads a kernel via `kernel=` SimObject param, then primes KMU DCRs directly. After the redesign, KMU DCR programming must route through the CP, which means the standalone test needs a tiny one-shot ring submission instead of direct `proc_.dcr_write` calls. | Add a `vortex_gem5_run_standalone_kernel(handle, kernel_path)` C ABI in the device library that builds a synthetic CMD_DCR_WRITE+CMD_LAUNCH ring and runs the CP to completion. ~30 LoC. Keeps the standalone test path real instead of a back-door. |
| R6 | Vortex's `cycle()` does not handle being called only when scheduled; e.g. internal counters reset assuming consecutive ticks. | Audit during M2. Vortex's existing implementation already supports being suspended (simx uses it both ways). |

---

## 9. Out of scope

- **XLEN=64 device library.** Current setup is XLEN=32 only.
  Orthogonal.
- **Separate `ClockDomain` for Vortex vs. CP.** D2 ratifies single
  domain for v1.
- **Gem5 `SimpleMemory` backing VRAM via DMA port.** D3 ratifies
  in-process for v1 with the accessor seam in place. v2 is a
  follow-on commit that swaps the accessor.
- **PCIe BAR mapping** instead of raw PIO range. Original gem5_simx_v3
  §3.6 commits to this; orthogonal to the CP redesign.
- **Multi-queue host runtime.** Q1–Q3 hardware is there but the host
  runtime exercises Q0 only. Multi-queue runtime work follows
  upstream vortex2.h.
- **Profiling timestamp writeback path.** The upstream CP supports
  `F_PROFILE` flag + `VX_cp_profiling`; gem5 backend will get it for
  free once the CP implementation lands. No gem5-specific work.

---

## 10. Estimated effort

Calibrated against the v1 OPAE backend (~3 days from scratch) and the
recent rejected inline-adaptation attempt (Option A reached ~50%
completion in ~30 min before being stopped):

- **Phase M1 (merge):** 10 min. Three known conflicts, all mechanical.
- **Phase M2 (device redesign):** 8–10 h. Bulk of the work:
  - `dev_mem.{h,cpp}` — 1 h
  - `vortex_gpgpu.{cpp,h}` rewrite — 4 h (mostly subtraction)
  - `vortex_gpgpu_dev.{cc,hh}` rewrite — 3 h (event-driven scheduling)
  - `VortexGPGPU.py` + Makefile + standalone test ABI — 1 h
- **Phase M3 (host runtime):** 2–3 h. Mostly subtraction; new code is
  small.
- **Phase M4 (docs):** 1 h.

**Total: 11–14 h focused work, single commit on `feature_gem5`.**

Calibration vs. the v1 draft of this proposal (which claimed 7–11 h):
the redesign is longer because (a) event-driven scheduling needs more
care than a polled tick, (b) the OPAE deletion is comprehensive
(M4 was "optional" in v1), and (c) we now have to wire the standalone
test path through a real CP ring submission instead of direct DCR
writes.

---

## 11. Why not a smaller change?

For the record — the alternatives that were considered and rejected:

- **Adapt-only (the rejected Option A from v1 of this doc).** Embed
  `vortex::CommandProcessor` in the host runtime; translate each CP
  hook back into the existing OPAE MMIO protocol. **Rejected:**
  CP runs on the wrong side of the host/device boundary, every ring
  fetch costs an MMIO+DMA round trip across the simulated bus,
  device-side OPAE state machine stays as permanent dead code, no
  alignment with how opae/xrt do it on real silicon.
- **Device-side CP, keep OPAE for `mem_upload` data plane.** The v1
  draft of this proposal. **Rejected:** Two control planes coexist,
  two protocols to keep in sync, no clean line between "what goes
  through CP" and "what doesn't."
- **`VORTEX_USE_CP=0` transparent mode** as a permanent bring-up
  escape hatch. **Rejected:** defeats the purpose of a cycle-accurate
  simulator; the gem5 backend's job is to model the hardware, not to
  emulate around it.

The redesign in this proposal is the minimum that does not leave dead
code, dead protocols, or bring-up hacks in the final state.

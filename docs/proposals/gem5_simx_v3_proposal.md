# gem5 Integration for SimX v3 — Proposal

**Date:** 2026-05-16
**Status:** ✅ ALL PHASES (0–7) COMPLETE on BOTH x86_64 AND aarch64 (hello + vecadd + sgemm × 2 ISAs all PASS end-to-end in 16 s wall via `VORTEX_GEM5_ARM=1 ./ci/regression.sh --gem5`)
**Author:** Blaise Tine
**Related:**
[simx_v3_proposal.md](simx_v3_proposal.md) (Phase 5: TLM data path),
[sst_simx_v3_proposal.md](sst_simx_v3_proposal.md) (the sister integration whose patterns this proposal follows),
[master_merge_v3_proposal.md](master_merge_v3_proposal.md) §10.2 (the precedent for cross-simulator integrations on this line),
[`~/dev/vortex_gem5`](https://github.com/sij814/vortex_gem5) on branch `gem5`, commit `91dcf17` ("working Vortex with gem5", 2025-05-22 — Injae Shin, UCLA capstone),
[Injae Shin, "gem5-Vortex: Heterogeneous Cross-ISA Integration of Vortex GPGPU in gem5"](#) (capstone report, 2025).

---

## 1. Constraints (load-bearing)

Any design that breaks one of these is wrong.

1. **One source of truth for memory state.** Per
   [simx_v3_proposal.md §3.3](simx_v3_proposal.md), data lives in the
   channel hierarchy: `MemReq`/`MemRsp` packets carry actual bytes
   between `MemCoalescer` → `Cache` → `Memory`, and the `RAM` image
   attached to `Memory` is authoritative. There is no shadow backing
   store and no parallel `MemBackend`. The gem5 integration plugs in at
   exactly one boundary (the device's DMA port maps to `RAM`
   read/write); it does **not** introduce a second data path.
2. **Single clock owner per simulation.** Under gem5, gem5 drives the
   clock: `VortexGPGPU::tick()` (a gem5 `EventFunctionWrapper` that
   reschedules itself every cycle at the device clock) calls
   `Processor::cycle()`. SimX does not advance on its own and there is
   no worker thread doing async `Processor::run()` in the background.
   (This is a deliberate departure from the legacy `vortex_gem5` design
   — see §2.2 — which is the source of most of that branch's bugs.)
3. **gem5 plugs in at one boundary, not many.** Vortex → gem5 traffic
   crosses two well-defined interfaces:
   - **PIO** for MMIO command/status registers (the OPAE AFU image
     layout, unchanged from `sw/runtime/opae`).
   - **DMA** for staging-buffer host↔device transfers, and for any
     future host-visible memory window.
   The cache hierarchy, scheduler, ALU/FPU, KMU, and the new
   `Processor::cycle()` entry point do not know gem5 exists.
4. **No regression for non-gem5 builds.** `make -C sim/simx` (no
   `USE_GEM5=1`) continues to produce a self-contained `simx` binary
   identical to today's. gem5 is opt-in compile-time, not a runtime
   probe, and ships as a separate shared library (`libvortex-gem5.so`)
   that the gem5 SimObject loads. Per §1.4 of
   [sst_simx_v3_proposal.md](sst_simx_v3_proposal.md).
5. **The Vortex tree owns the integration code.** All gem5-facing C++
   (the `DmaDevice` SimObject) and Python (SimObject config + test
   scripts) live under `sim/simx/gem5/` and `ci/gem5_test_vortex_*.py`
   in this repo. `ci/gem5_install.sh` fetches a pinned upstream gem5
   release and copies/symlinks our SimObject into its source tree
   before building. Versioning the integration alongside Vortex is what
   makes it possible to review API-breaking changes in a single PR;
   the legacy split across two repos is what froze `vortex_gem5` at a
   two-year-old SimX.
6. **Author attribution.** The legacy `vortex_gem5` design (DMA-bouncing
   through a pinned staging buffer, OPAE-shaped MMIO command set, ARM
   SE-mode runtime) is Injae Shin's capstone work. The
   re-implementation is a rewrite, not a port (§2), but each new file's
   commit body cites the capstone report and the legacy commit
   (`vortex_gem5@91dcf17`).

---

## 2. Why the legacy `vortex_gem5` cannot be ported as-is

### 2.1 The architectural mismatch

`vortex_gem5` was built on pre-v3 SimX (`Arch`, `Processor*`,
single-step `run()`, `set_running(true)`, `VX_DCR_BASE_*` startup DCRs
broadcast to all cores). v3 explicitly retired all of those:

| Concern | Legacy SimX (vortex_gem5) | SimX v3 (this branch) |
|---|---|---|
| Sizing | `Arch arch(NUM_THREADS, NUM_WARPS, NUM_CORES)` object | Macros (`NUM_THREADS`, etc.) — no `Arch` class |
| Top-level | `Processor(arch)` ctor with arg | `Processor()` no-arg ctor |
| Run model | `processor->run()` is one cycle | `processor.run()` blocks to completion |
| Single-cycle step | `processor->run()` per cycle from `proc_tick()` | does not exist — must be added (`Processor::cycle()`) |
| Kernel dispatch | `set_running(true)` + `VX_DCR_BASE_STARTUP_*` | `KMU::start()` + `VX_DCR_KMU_*` (startup + grid/block dims) |
| Cache flush | implicit in `run()` finish | explicit: `dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy)` per core before host read-back |
| Memory hierarchy | `MemSim` + `CacheSim` are timing-only, data sits in `MemBackend` (`Emulator`-side) | `Memory` + `Cache` carry data through `MemReq`/`MemRsp`; backing image is in `RAM` attached to `Memory` |
| Runtime layout | top-level `runtime/{stubarm,opaesimx}/` | reorganized under `sw/runtime/` per [master_merge §3](master_merge_v3_proposal.md) |

So the **shape of the gem5 plug-in changes**: not "tick the legacy
single-cycle Processor" but "add a `cycle()` entry point to the v3
Processor and call it from the gem5 SimObject," with KMU-style dispatch
and an explicit cache-flush before host read-back.

### 2.2 Specific bugs in the legacy code

A walk-through of `vortex_gem5/sim/{simx,opaesimx}/` and
`vortex_gem5/runtime/{stubarm,opaesimx}/` found the following defects.
Each is called out so the redesign does not re-introduce it.

| # | File | Defect | Why it matters |
|---|---|---|---|
| B1 | `sim/simx/simx_device.cpp:122` (`proc_tick`) | Calls `processor_->run()` directly. On legacy SimX this was a single step; on v3 it would block until program completion. | The "tick per gem5 cycle" pattern simply won't work. We must add a real single-cycle `Processor::cycle()` (already required for SST). |
| B2 | `sim/simx/simx_device.cpp:111` (`start`) | `processor_->set_running(true)` — that API does not exist in v3. The KMU now drives execution and requires `VX_DCR_KMU_GRID_DIM_*` / `VX_DCR_KMU_BLOCK_DIM_*` to be written before the first cycle. | Even after re-pluming, kernels won't launch without the KMU DCR setup (see `sim/simx/main.cpp:101–116`). |
| B3 | `sim/opaesimx/opae_simx.cpp:185, 199` (`read_mmio64`/`write_mmio64`) | Implementation is `*(uint64_t*)(GEM5_BASE_ADDR + offset)` — a raw host-pointer dereference into a fixed virtual address. | Only works when the host runtime and the gem5 device share an address space (i.e., when the host runtime is *not* actually inside gem5). It is a stand-in for the real path, not the real path. Cross-ISA simulation defeats the assumption: an ARM userspace process inside gem5 cannot dereference `0x20000000` and reach the device. The legacy code papers over this with a co-resident driver hack; v3 needs a real PIO/DMA path. |
| B4 | `sim/opaesimx/opae_simx.cpp:204–399` | Several hundred lines of commented-out CCI/AVS bus + Verilator (`device_->…`) plumbing left in place, referencing fields and types that do not exist in this file. | Dead code that obscures what the module actually does. Drop it; the new gem5 wrapper has no CCI bus to model. |
| B5 | `sim/opaesimx/opae_simx.cpp:71` (`dram_sim_` field) | DRAM model is constructed but never ticked or consulted after the gem5 hack landed. | Dead state. |
| B6 | `sim/opaesimx/opae_simx.cpp:103` (`pinned_alloc_`) | Uses `PIN_BASE_ADDR = 0x10000000` with `PINNED_MEM_SIZE = 0xFFFFFF` (16 MB), hardcoded. No bounds check beyond `MemoryAllocator::allocate` failure. | Tiny by design — large kernel inputs would silently fail. The v3 design should size from `GLOBAL_MEM_SIZE`/`ALLOC_BASE_ADDR` and surface OOM errors. |
| B7 | `runtime/opaesimx/vortex.cpp:324, 367` | `auto ls_shift = (int)std::log2(CACHE_BLOCK_SIZE);` — uses float `log2` for an integer constant, then discards the result. | Cosmetic / dead, but a smell. Use `log2ceil(CACHE_BLOCK_SIZE)` from `sw/common/util.h`. |
| B8 | `runtime/opaesimx/vortex.cpp:418–474` (`ready_wait`) | `nanosleep` call is **commented out**; the busy loop only decrements `timeout_ms` and never sleeps. On a long-running kernel inside gem5 SE-mode this saturates the simulated ARM core. | Either use the gem5 device's interrupt path (preferred — implementable as an MMIO doorbell) or restore the `nanosleep` so the ARM CPU is idle while the GPU runs. |
| B9 | `runtime/opaesimx/vortex.cpp:349–390` (`download`) | No cache-flush step before reading back results from device memory. | On v3, dirty lines must be drained via `dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy)` per core (see `sim/simx/main.cpp:194–197`, `sw/runtime/simx/vortex.cpp:191–197`) or the host sees stale data. |
| B10 | `runtime/opaesimx/vortex.cpp:478–489` (`dcr_write`) | OPAE protocol has `CMD_DCR_WRITE` but no `CMD_DCR_READ`. | The cache-flush fix above requires a `dcr_read` path. Current `sw/runtime/opae` already adds `CMD_DCR_READ` + `MMIO_DCR_RSP` — adopt the same shape on the gem5 device. |
| B11 | `runtime/stubarm/vortex.cpp:54` | `static callbacks_t g_callbacks;` global with `vx_dev_init(&g_callbacks)` resolved at link time. | Works for a single-device test but breaks `vx_dev_open` from being called concurrently from two host processes. Less critical for the gem5 use case (single device per simulation) but worth flagging. |
| B12 | `sim/simx/simx_device.cpp` (`Impl`) | Uses `std::future<void> future_` for shutdown synchronization but `proc_tick()` calls `processor_->run()` directly on the caller thread. The mutex / future plumbing implies an async model that isn't actually used. | Confused concurrency contract. The v3 design must pick one: synchronous tick from the gem5 event loop (this proposal) **or** async run with a doorbell — not both. |
| B13 | `runtime/stubarm/Makefile:7` + `runtime/opaesimx/Makefile:9` | Cross-compiler hardcoded to `arm-linux-gnueabihf-g++` (32-bit ARM hard-float). | gem5 also models AArch64 ARMv8 and x86_64, and most contemporary ARM ports are 64-bit. The v3 build selects compiler from a `HOST_ARCH` make variable (`x86_64`, `aarch64`, `armhf`); see Phase 4. |
| B14 | `runtime/opaesimx/vortex.cpp:489` (`dcr_write`) and `stubarm/vortex.cpp:139` | Both runtimes write to DCR via the OPAE protocol but no MMIO ordering / fence is established between DCR writes and the `CMD_RUN` MMIO. | Inside gem5 the host CPU model may reorder MMIO. Need an explicit barrier before `CMD_RUN` (per `HOST_ARCH`: `mfence` for x86, `dmb sy` for ARM). Phase 4 provides a `vortex_gem5_mmio_fence()` inline helper. |
| B15 | `sim/opaesimx/opae_simx.cpp:138–157` (`prepare_buffer`) | Returns `*buf_addr = (void*)buffer.ioaddr;` — casts an integer device IO address back to a `void*`. | The runtime then dereferences this pointer to do `memcpy(staging_ptr_, host_ptr, size)` (line 322 of `runtime/opaesimx/vortex.cpp`). Same root cause as B3 — only works when host runtime and device share an address space. Under real gem5 the runtime must `mmap` the pinned region via a syscall the gem5 device intercepts, or the gem5 device must expose the pinned region as a PIO/DMA window. |

Together B1, B2, B3, B6, B9, B14 and B15 mean the legacy integration as
literally written does not run a kernel correctly under v3 even after
the path renames are applied; it requires architectural rework, not
porting.

### 2.3 What still ports as design intent

The legacy paper's design intent — and these are what we keep:

- **OPAE-shaped MMIO command set.** `CMD_RUN`, `CMD_MEM_READ`,
  `CMD_MEM_WRITE`, `CMD_DCR_WRITE`, `MMIO_CMD_TYPE`, `MMIO_CMD_ARG0..2`,
  `MMIO_STATUS`. Add `CMD_DCR_READ` + `MMIO_DCR_RSP` per the v3 OPAE
  runtime (B10). The kernel runtime under `sw/runtime/gem5/` reuses
  this layout so the same `vortex.h` shim layer that drives `opae`
  also drives `gem5`.
- **Pinned staging buffer pattern** for host↔device transfers. A
  fixed device-visible region of host address space; runtime
  `memcpy`'s into it, device DMAs out of it. Sizing is dynamic
  (allocate-on-demand) rather than the legacy fixed-16-MB chunk (B6).
- **Single-PIO-range device** registered to gem5 with the OPAE MMIO
  offsets. The runtime issues 64-bit MMIO writes; the SimObject
  decodes them in `write()` / `read()`.
- **The host SE-mode runtime** (`sw/runtime/gem5/`, native x86 or cross-compiled ARM)
  shipped into gem5's SE-mode app, **NOT** a full-system Linux on the
  guest. The paper makes this point explicitly and it is the
  differentiator vs. NoMali (FS-only) and AMD GPU (FS-only). See
  `capstone §IIC`.

### 2.4 What needs a v3 redesign

- **`sim/simx/simx_device.{cpp,h}`** — replace with
  `sim/simx/gem5/vortex_gpgpu.{cpp,h}` (the SimObject wrapper)
  plus reuse of the new `Processor::cycle()` API. The legacy file's
  `Impl` class is the wrong shape (B1, B2, B12).
- **`sim/opaesimx/opae_simx.{cpp,h}`** — delete entirely. The legacy
  module is a host-side OPAE stub whose `read_mmio64`/`write_mmio64`
  do raw pointer arithmetic (B3, B15). The v3 design routes MMIO
  through gem5's PIO port; there is no host-side stub.
- **`runtime/opaesimx/`** — delete. The OPAE-stub path was a
  pre-gem5 debugging convenience; under v3 we test the gem5 device
  end-to-end via a gem5 Python script (§4, Phase 5), not via a
  co-resident driver.
- **`runtime/stubarm/`** — replace with `sw/runtime/gem5/`,
  re-implemented against the same `callbacks.h` ABI as
  `sw/runtime/simx`/`opae`/`rtlsim`, with cache-flush plumbed in
  (B9), MMIO fences before `CMD_RUN` (B14), and a configurable ARM
  cross-compiler target (B13).

---

## 3. Target architecture

```
                ┌───────────────────────────────────────────────┐
                │  gem5 simulation                              │
                │  ─────────────────                            │
                │  ./ci/gem5_test_vortex_hello.py               │
                │  (gem5.opt is build/X86/gem5.opt or           │
                │   build/ARM/gem5.opt; both supported)         │
                │                                               │
                │  ┌─────────────┐         ┌─────────────────┐  │
                │  │ Host CPU    │ ──PIO─▶ │ VortexGPGPU     │  │
                │  │ (X86 or ARM,│ ◀─PIO── │ (DmaDevice ↓    │  │
                │  │  SE mode)   │         │  PioDevice)     │  │
                │  │ user        │         │  ┌───────────┐  │  │
                │  │ binary:     │         │  │ MMIO regs │  │  │
                │  │  hello +    │         │  └───────────┘  │  │
                │  │  libvortex- │         │  ┌───────────┐  │  │
                │  │  gem5.so    │ ──DMA─▶ │  │ Pinned    │  │  │
                │  │  (native    │ ◀─DMA── │  │ staging   │  │  │
                │  │   for X86,  │         │  │ buffer    │  │  │
                │  │   cross-    │         │  │ window    │  │  │
                │  │   compiled  │         │  └───────────┘  │  │
                │  │   for ARM)  │         │       │         │  │
                │  └─────────────┘         │       ▼         │  │
                │         │                │  ┌───────────┐  │  │
                │         │ MemPort        │  │ vortex::  │  │  │
                │         ▼                │  │ Processor │  │  │
                │  ┌─────────────┐         │  │ (SimX v3) │  │  │
                │  └─────────────┘         │  │           │  │  │
                │                          │  │  Cluster[]│  │  │
                │                          │  │   Cache   │  │  │
                │                          │  │   Memory ─┼──┼──┼─▶ RAM (Vortex VRAM,
                │                          │  └───────────┘  │  │      held inside the
                │                          │   ▲             │  │      device — separate
                │                          │   │ cycle()     │  │      address space from
                │                          │  ┌┴──────────┐  │  │      gem5 DRAM)
                │                          │  │ tick      │  │  │
                │                          │  │ (gem5     │  │  │
                │                          │  │  event)   │  │  │
                │                          │  └───────────┘  │  │
                │                          └─────────────────┘  │
                └───────────────────────────────────────────────┘
```

### 3.1 The plug-in boundary

The Vortex side exposes **one** plug-in unit: `libvortex-gem5.so`. It
is built from the same `sim/simx/*.{cpp,h}` sources as the default
`simx` binary, plus a single new wrapper file
(`sim/simx/gem5/vortex_gpgpu.{cpp,h}`) that holds:

- A `vortex::Gem5Wrapper` C++ class that owns a `vortex::Processor`,
  a `vortex::RAM` (the device VRAM), and a thin `cycle()` entry
  point — exactly mirroring `vortex::VortexSimulator` in
  `sim/simx/sst/`.
- A C-ABI shim (`vortex_gem5_create()`, `vortex_gem5_tick()`,
  `vortex_gem5_mmio_write64()`, `vortex_gem5_mmio_read64()`,
  `vortex_gem5_dma_read()`, `vortex_gem5_dma_write()`, …) so the
  gem5-side SimObject is decoupled from C++ ABI changes in
  `vortex::Processor`. **The C ABI is the contract;** changing it
  requires a coordinated update of the gem5-side SimObject.

The gem5 side is **one** SimObject + **one** Python file, both shipped
in this repo at `sim/simx/gem5/`:

- `vortex_gpgpu_dev.{cc,hh}` — subclasses `gem5::DmaDevice` (which
  itself subclasses `PioDevice`). Holds an opaque
  `vortex_gem5_handle_t`; on `tick()`, calls `vortex_gem5_tick()`. PIO
  reads/writes decode the OPAE MMIO offsets and forward to
  `vortex_gem5_mmio_*`. DMA reads/writes triggered by
  `CMD_MEM_{READ,WRITE}` use gem5's `DmaPort` and copy bytes into the
  device VRAM via `vortex_gem5_dma_*`.
- `VortexGPGPU.py` — `gem5.SimObject` definition with `pio_addr`,
  `pio_size`, `pio_latency`, `dma_latency`, `clock`, `library`
  (path to `libvortex-gem5.so`), and `kernel` (path to `*.vxbin` —
  loaded into VRAM at boot, in lieu of the runtime upload path, for
  smoke tests).

`ci/gem5_install.sh.in` fetches a pinned gem5 release
(see §3.4 for version), copies the two files into
`<gem5>/src/dev/vortex/`, drops a one-line `SConscript`, and runs
`scons build/ARM/gem5.opt`.

**Nothing upstream of `vortex_gem5_create()` knows gem5 exists.** This
satisfies §1.3.

### 3.2 The cycle interface

`Processor::cycle()` does **not exist** in v3 today. It is a direct
prerequisite of both the SST integration (per
[sst_simx_v3_proposal.md §3.2](sst_simx_v3_proposal.md)) and this
proposal. The signature and shape are identical to what SST needs:

```cpp
// processor.h — public additions
bool cycle();        // advance one cycle; returns false when nothing is running
Memory* memsim();    // for optional gem5/SST memory-mirroring hooks
```

```cpp
// processor.cpp — implementation
bool ProcessorImpl::cycle() {
  if (!is_cycle_initialized_) {
    SimPlatform::instance().reset();
    this->reset();
    kmu_->start();                  // dispatch CTAs into the cluster
    is_cycle_initialized_ = true;
  }
  SimPlatform::instance().tick();
  return this->any_running();
}

Memory* ProcessorImpl::memsim() { return memsim_.get(); }
```

The two pieces (`SimPlatform::reset()` → `start_kmu()` →
`SimPlatform::tick()` and `any_running()`) are already factored on
`Processor` from Round 6 DTM work. `cycle()` just packages them into a
single-cycle step.

**Reuse from DTM work:** `start_kmu()` and `any_running()` are already
public on `Processor`. We add `cycle()` and `memsim()` and that is the
entire SimX-side API surface required by both SST and gem5.

### 3.3 The MMIO command protocol

Identical to `sw/runtime/opae` v3 (the OPAE driver), reusing
`hw/syn/altera/opae/vortex_afu.h`:

| Offset | Name | Direction | Purpose |
|---|---|---|---|
| `MMIO_CMD_TYPE` | `CMD_*` | W64 | Dispatch one of: `MEM_READ`, `MEM_WRITE`, `RUN`, `DCR_WRITE`, `DCR_READ` |
| `MMIO_CMD_ARG0..2` | command-specific | W64 | DCR addr / device addr / size / value |
| `MMIO_STATUS` | bit0=busy | R64 | Polled by runtime's `ready_wait` |
| `MMIO_DCR_RSP` | response | R64 | Result of `CMD_DCR_READ` (used for cache-flush) |
| `MMIO_DEV_CAPS` / `MMIO_ISA_CAPS` | caps bitfield | R64 | Encoded device capabilities |

The runtime issues commands by writing args first, then `CMD_TYPE`
(B14 fix: emit a `DMB SY` before the type write). The device latches
on `CMD_TYPE`, performs the action synchronously (PIO write returns
when the operation is enqueued, or completes synchronously for
fast ones like `DCR_WRITE`), and clears the status busy bit when done.

`CMD_MEM_{READ,WRITE}` use the staging-buffer protocol from the
capstone paper Fig. 5 (§3.4 below).

### 3.4 The staging-buffer protocol

The gem5 device exposes a PIO-addressable register `MMIO_PINNED_BASE`
that returns the base address of a pinned region inside gem5's host
address space. The runtime, on `vx_mem_alloc`, lazily picks a slice of
that region as a staging buffer.

For a `vx_copy_to_dev(host_ptr, dev_addr, size)`:
1. Runtime `memcpy(staging_buf, host_ptr, size)`.
2. Runtime writes `staging_buf_addr`, `dev_addr`, `size` to
   `MMIO_CMD_ARG{0,1,2}`.
3. Runtime writes `CMD_MEM_WRITE` to `MMIO_CMD_TYPE`.
4. Device's PIO handler enqueues a `gem5::DmaPort::dmaAction()` read
   from `staging_buf_addr` into a local scratch.
5. On DMA completion, the device copies the scratch bytes into Vortex's
   `RAM` at `dev_addr` (via `RAM::write`).
6. Device clears the status busy bit.
7. Runtime polls `MMIO_STATUS` until busy=0.

`vx_copy_from_dev` is the reverse, with **cache flush first** (B9):
the runtime issues `CMD_DCR_READ(VX_DCR_BASE_CACHE_FLUSH, cid)` for
every core before the `CMD_MEM_READ`. The device's DCR-read handler
plumbs through to `Processor::dcr_read`, which already invokes
`flush_caches()` for the cache-flush DCR
([processor.cpp:251–258](../../sim/simx/processor.cpp#L251)).

This is the same protocol the v3 OPAE runtime already uses, so the
runtime under `sw/runtime/gem5/` differs from `sw/runtime/opae/` only
in:
- The `driver.{cpp,h}` backend (gem5 mmaps a `/dev/vortex_gem5`
  character device path **OR**, in SE-mode, gem5 sets up the device's
  PIO/DMA windows directly in the simulated process's address space —
  see §3.6).
- The lack of an `fpgaPrepareBuffer` API (the device exposes the
  pinned region itself; no per-call buffer allocation by an OPAE
  layer).

### 3.5 Build-time gating

`USE_GEM5=1` make variable controls compilation of:
- `sim/simx/gem5/vortex_gpgpu.{cpp,h}` (the C ABI wrapper).
- Link target `libvortex-gem5.so` produced alongside `libsimx.so`
  (mirrors the SST `libvortex.so` pattern in `sim/simx/Makefile`).

`USE_GEM5=1` does **not** affect the default build:
`make -C sim/simx` (no flag) still produces a stand-alone `simx`
binary with no gem5 dep. Per §1.4.

The host-side runtime supports both x86 (native) and ARM (cross-
compiled) targets via a `HOST_ARCH` switch:
```
make -C sw/runtime/gem5                                     # x86 default
make -C sw/runtime/gem5 HOST_ARCH=x86_64                    # explicit x86
make -C sw/runtime/gem5 HOST_ARCH=aarch64                   # AArch64 cross
make -C sw/runtime/gem5 HOST_ARCH=armhf                     # ARMv7 cross
```
producing `libvortex-gem5-{x86_64,aarch64,armhf}.so`. Test scripts
select the matching `(gem5.opt, libvortex-gem5-*.so)` pair via the
`HOST_ARCH` make variable. Native x86 needs no toolchain install; ARM
requires `gcc/g++-aarch64-linux-gnu` (or `-arm-linux-gnueabihf` for
ARMv7), which `ci/gem5_install.sh` installs as part of Phase 0.

### 3.6 gem5 SE-mode wiring + ISA selection

**Host ISA: both x86 and ARM, equally first-class** (decision recorded
2026-05-16 after Phase 0 prototyping). Phase 0's `ci/gem5_install.sh`
builds `build/X86/gem5.opt` *and* `build/ARM/gem5.opt`; phases 4–6
test both. Rationale:

- **x86** is the path of least resistance for users — no
  cross-toolchain, native `g++` builds `sw/runtime/gem5/`, faster
  gem5 CPU model, and PCIe is canonical on x86 (relevant to the
  Phase 5+ upgrade path below).
- **ARM** is the research-narrative path matching the capstone paper
  (Injae Shin 2025) and actually-deployed ARM+accelerator HPC
  platforms (Grace Hopper, Fugaku, Graviton, Apple Silicon). Kept
  as a first-class matrix variant; not a stretch goal.

Three MMIO/DMA paths exist; this proposal picks one for the initial
work and notes the others as future upgrades:

| Path | Description | Status in this proposal |
|---|---|---|
| **1. SE-mode + custom PIO+DMA wiring** | The device is a `DmaDevice` subclass attached to `system.membus` at a configurable `pio_addr` (default `0x20000000`, matching the legacy paper). Host binary touches the address via `mmap`/inline asm. Works in both x86 SE-mode and ARM SE-mode. | **Phase 2–6: this is the design.** Matches legacy paper, lightweight, fast iteration. |
| **2. FS-mode + PCIe device** | Subclass `PciDevice` (which already inherits `DmaDevice`); BARs expose MMIO, DMA for staging. Full Linux boot inside gem5 with a tiny PCI kernel module to bind the device. | **Phase 5+ upgrade.** Realistic accelerator-modeling story expected by x86 users. The C ABI committed in Phase 2 is shape-compatible — `PciDevice` and the custom `DmaDevice` both use the same `vortex_gem5_dma_*` callbacks; only the gem5-side wrapper class differs. |
| **3. `/dev/vortex_gem5` pseudo-file** | The gem5 device implements `SyscallReturn open(...)` + `mmap` for a synthetic device path. Runtime `open("/dev/vortex_gem5", O_RDWR)` + `mmap`. | Out of scope. Closest to how real OPAE drivers work but requires a custom syscall handler in gem5; cost outweighs the benefit when Path 1 already works. |

**Doorbell queues** are a Phase 7+ realism upgrade orthogonal to the
transport choice above. AMD GPU (gem5 `src/dev/amdgpu/`, derived
from `PciEndpoint`) and NVIDIA-style modern accelerators use a ring
buffer in host DRAM plus a single MMIO "doorbell" write per dispatch:
the host appends commands to the ring, then writes the new tail
offset to the doorbell register; the device asynchronously walks the
ring and processes commands. The Phase 2-6 design instead uses
**status polling** — the host writes args + `CMD_TYPE`, then polls
`MMIO_STATUS` until done — which matches the legacy OPAE FPGA driver.
Polling is fine for the capstone-paper scope (small kernels, one at
a time) but burns simulated cycles on the spin. If later research
wants batched-dispatch realism comparable to AMD GPU, the upgrade
swaps the OPAE MMIO command set for a ring + doorbell protocol; the
C ABI in Phase 2 stays compatible (a new `vortex_gem5_doorbell_ring(handle, tail)`
entry point alongside the existing `vortex_gem5_mmio_*`).

### 3.7 gem5 version pinning

`ci/gem5_install.sh.in` pins gem5 to v25.0.0 (the most recent stable
release as of 2026-05). The pinned tag goes in `VERSION` alongside
`TOOLCHAIN_REV` and `SST_VER` — bumps require a CI re-run on the
self-hosted runner first (small risk of API drift on gem5's
`DmaDevice`/`PioDevice` between major releases). **Picking and
validating this pin is the first deliverable of Phase 0** — every
other phase is a no-op if Phase 0 reveals that v25.0.0 no longer
supports SE-mode PIO mapping or the SimObject install path we depend
on.

### 3.8 Why this is not just a copy of the SST pattern

SST and gem5 are similar in shape (external simulator drives the
Vortex clock through a C++ wrapper around `Processor::cycle()`) but
differ in three load-bearing ways:

1. **The host process is simulated under gem5.** Under SST the host
   "process" is the SST Python script itself, running natively on the
   developer's machine. Under gem5 the host is a userspace process
   (x86 or ARM, per §3.6) running inside the gem5 model. So the gem5
   integration also needs a host-side runtime under `sw/runtime/gem5/`
   (native compile for x86, cross-compile for ARM); SST does not.
   (This is the bulk of the work that makes gem5 the bigger project —
   see §9 effort estimate.)
2. **Memory is in two address spaces.** Under SST, the SimX `Processor`
   and any optional SST memHierarchy share the same simulator. Under
   gem5, the host CPU's DRAM is a gem5 `AddrRange`, the Vortex VRAM is
   a `RAM` inside the device, and the only way bytes cross between
   them is via DMA through the device. The staging-buffer protocol
   (§3.4) implements this; SST has no equivalent.
3. **PIO bus integration.** SST's `StandardMem` interface is the
   only one we plug into; gem5 has separate `PioPort` and `DmaPort`
   with different timing models. The wrapper must manage both.

---

## 4. Phasing

Each phase is independently shippable and validated. The work follows
the same shape as the SST integration in
[sst_simx_v3_proposal.md §4](sst_simx_v3_proposal.md): **environment
first**, API + library second, gem5-side wiring third, ARM runtime
fourth, CI last.

### Phase 0 — gem5 environment + API survey *(derisking; nothing else can start until this is done)*

The legacy `vortex_gem5` was built against a forked gem5 that no
longer exists publicly. Before we design the C ABI in Phase 2 or
write a single line of `DmaDevice` glue in Phase 3, we need a
known-good gem5 build on the bench so the API surface we are about
to commit to is **real**, not assumed-from-headers-we-haven't-read.
This is the "solve gem5 setup first" phase.

Concretely:

- **Pick and pin the gem5 version.** Default target: v25.0.0.1
  (patch release on top of v25.0.0, most recent stable as of 2026-05).
  Pin the tag in `VERSION` alongside `TOOLCHAIN_REV` and `SST_VER`:
  ```
  GEM5_REV=v25.0.0.1
  ```
- **Write `ci/gem5_install.sh.in`** (no Vortex integration yet — just
  the install). Mirrors the structure of `ci/sst_install.sh.in`:
  - `apt install scons python3-dev python3-pip libprotobuf-dev
    protobuf-compiler libprotoc-dev libgoogle-perftools-dev m4
    libboost-all-dev gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`
    (gem5's documented build deps + ARM cross-toolchain for the ARM
    matrix variant).
  - Fetch gem5 working tree at `$GEM5_REV` into `$TOOLDIR/gem5`.
  - `scons build/X86/gem5.opt -j$(nproc)` and
    `scons build/ARM/gem5.opt -j$(nproc)` — **both ISAs by default**
    per the dual-ISA decision in §3.6. Targets selectable via
    `GEM5_TARGETS="X86"` / `"ARM"` / `"X86 ARM"`.
  - Export `GEM5_HOME=$TOOLDIR/gem5` to `~/.bashrc`.
- **Validate the X86 native compiler produces SE-mode binaries.**
  Trivial — `gcc -static -o /tmp/hello-x86 sim/simx/gem5/hello.c`
  then run under `gem5.opt configs/example/gem5_library/arm-hello.py`
  -shape config (substituting `ISA.X86`). Confirm exit code 0 and
  the expected stdout.
- **Validate the ARM cross-toolchain produces SE-mode binaries.**
  Cross-compile `hello.c` with `aarch64-linux-gnu-gcc -static -o
  /tmp/hello-arm`, run under
  `build/ARM/gem5.opt configs/example/gem5_library/arm-hello.py`
  (or the deprecated SE script). Confirms the cross-toolchain
  produces something gem5 ARM-mode can load.
- **Read the gem5 source for the API surface we are about to use**
  and record findings in a short scratch file
  `sim/simx/gem5/gem5_api_notes.md` (not committed to docs/, just a
  Phase 0 deliverable):
  - `src/dev/io_device.hh` — `PioDevice::read`/`write` signatures
    in v25.0.0. Compare to what the legacy paper assumed.
  - `src/dev/dma_device.hh` — `DmaDevice::dmaAction`, `DmaPort`
    timing model. Confirm 64-bit address support, async completion
    callback shape.
  - `src/python/m5/objects/Device.py` — SimObject Python bindings.
    Confirm that out-of-tree `src/dev/<our-dir>/SConscript` is
    picked up by `scons build/ARM/gem5.opt` (this is the install
    mechanism we rely on in Phase 3).
  - `configs/example/se.py` — how SE-mode wires a CPU to a
    `Workload`. Confirm that we can attach a `PioDevice` and have
    the SE-mode loader map its PIO range into the workload's address
    space (the legacy paper's `0x20000000` magic). If this is no
    longer supported, the design changes — better to know now than
    in Phase 3.
- **Smoke-build a trivial out-of-tree SimObject** to prove the
  install mechanism end-to-end. Three files
  (`Dummy.{cc,hh,py}` + `SConscript`) under `sim/simx/gem5/dummy/`,
  installed by `sim/simx/gem5/install.sh` (Phase 0 only ships the
  installer; the real SimObject lands in Phase 3). After
  `ci/gem5_install.sh` re-runs, `gem5.opt --list-sim-objects` shows
  `Dummy`. Delete `dummy/` once verified — it was scaffolding.

**Validation:**
- `ci/gem5_install.sh` finishes successfully on the self-hosted
  runner. Wall time recorded in `gem5_api_notes.md` (drives CI
  caching strategy in Phase 6).
- `$GEM5_HOME/build/ARM/gem5.opt configs/example/se.py
  --cmd ./hello-arm` exits 0.
- `gem5.opt --list-sim-objects` lists the dummy SimObject installed
  via `sim/simx/gem5/install.sh`.
- `gem5_api_notes.md` documents the `DmaDevice` / `PioDevice` /
  `EventFunctionWrapper` signatures we will commit to in Phase 2's
  C ABI design.

**Why this is its own phase:** if any of those validations fails
(e.g. gem5 v25 has dropped SE-mode PIO mapping, or the SimObject
install mechanism has changed), the rest of the proposal needs
redesign before code lands. Phase 0 is a ~1-day gate, not a tracked
deliverable; everything downstream depends on its outputs.

### Phase 1 — `Processor::cycle()` + `Memory*` accessor

Prerequisite shared with SST. Can run in parallel with Phase 0
(no gem5 dependency) and lands first into the SimX-side codebase.

- Add `Processor::cycle()` and `Memory* Processor::memsim()` as in
  §3.2. This is a ~50-line patch to `processor.{cpp,h}` and
  `processor_impl.h` plus an `is_cycle_initialized_` bool.
- Add `Memory::set_pre_send_hook()` (already in v3 per
  `sim/simx/mem/memory.h:42` — verify still there; if so, this part
  of Phase 1 is a no-op).
- Update SST's `vortex_simulator.cpp` to use the new public
  `Processor::cycle()` API (currently calls `proc_->cycle()` which
  does not compile against `processor.h` HEAD — see
  `sim/simx/sst/vortex_simulator.cpp:64`). **This is a pre-existing
  bug that Phase 1 fixes for both integrations.**

**Validation:** `make -C sim/simx` (default), `make -C sim/simx
USE_SST=1`, and `make -C sim/simx USE_GEM5=1` all build. SST tests
that previously failed to link now link and run (`sst
ci/sst_test_vortex_hello.py` passes).

### Phase 2 — `libvortex-gem5.so` + C ABI

**Prerequisite: Phase 0 complete.** The C ABI is designed *against*
the `DmaDevice`/`PioDevice` shapes recorded in
`gem5_api_notes.md`, not from headers we haven't read.

- Create `sim/simx/gem5/vortex_gpgpu.{cpp,h}` mirroring
  `sim/simx/sst/vortex_simulator.{cpp,h}` shape:
  - Owns a `Processor`, a `RAM` (device VRAM at `MEM_PAGE_SIZE`).
  - Exposes a C ABI (`vortex_gem5_*`) sufficient for the gem5 device
    to MMIO/DMA/tick it. ABI signatures match what gem5's
    `DmaDevice::dmaAction` and `PioDevice::read`/`write` need to
    call into (per Phase 0 survey).
- Add `USE_GEM5=1` build target to `sim/simx/Makefile` producing
  `libvortex-gem5.so` (no SST symbols; no `sst-core` link). Pattern:
  duplicate the `ifeq ($(USE_SST),1)` block.
- Add a tiny in-process smoke driver
  `sim/simx/gem5/gem5_smoke_main.cpp` (built with the lib) that:
  1. Loads a `.vxbin` via the C ABI.
  2. Ticks until `cycle()` returns false.
  3. Reads the MPM exit code via DCR_READ.

  This is the "library compiles and a kernel runs through it without
  gem5 installed" smoke test (§6.2).

**Validation:**
- `make -C sim/simx USE_GEM5=1` builds.
- `LD_LIBRARY_PATH=. ./gem5_smoke hello.vxbin` returns 0.
- `make -C sim/simx` (no flag) still builds and `./simx hello.vxbin`
  returns 0 (no regression on default).

### Phase 3 — gem5 SimObject + Python config

**Prerequisite: Phases 0 + 2 complete.** The install mechanism is
already proven by Phase 0's dummy SimObject; this phase replaces
the dummy with the real device.

- `sim/simx/gem5/vortex_gpgpu_dev.{cc,hh}` — the gem5 `DmaDevice`
  subclass. PIO `read`/`write` decode MMIO offsets and call
  `vortex_gem5_mmio_*`. DMA actions triggered by `CMD_MEM_*`. A
  registered `EventFunctionWrapper` re-schedules itself every
  `clock_period_ticks()` and calls `vortex_gem5_tick()`.
- `sim/simx/gem5/VortexGPGPU.py` — Python SimObject definition.
- `sim/simx/gem5/SConscript` — for gem5's scons build.
- `sim/simx/gem5/install.sh` — copies the four files above into
  `<gem5>/src/dev/vortex/`. (Phase 0 already wrote this for the
  dummy SimObject; just extend it.)
- Update `ci/gem5_install.sh.in` to re-run `install.sh` and rebuild
  `build/ARM/gem5.opt` after the Vortex SimObject lands.

**Validation:** `ci/gem5_install.sh` succeeds with the real
SimObject installed. `gem5.opt --list-sim-objects` shows
`VortexGPGPU`. `gem5.opt configs/example/se.py --help` accepts
`VortexGPGPU` parameters.

### Phase 4 — Host runtime (`sw/runtime/gem5/`, x86 + ARM)

- New backend mirroring `sw/runtime/opae/` shape:
  - `vortex.cpp` — implements the `vx_*` callbacks against the OPAE
    MMIO protocol (§3.3), but the `driver.{cpp,h}` underneath does
    raw `mmap`/MMIO writes to the PIO address rather than calling
    `libopae`.
  - `Makefile` — selects compiler from `HOST_ARCH`:
    - `x86_64` (default): native `g++`
    - `aarch64`: `aarch64-linux-gnu-g++`
    - `armhf`: `arm-linux-gnueabihf-g++`
- Cache-flush integration (B9): the v3 `download` path issues
  `CMD_DCR_READ(VX_DCR_BASE_CACHE_FLUSH, cid)` per core before
  `CMD_MEM_READ`.
- MMIO ordering fence (B14): emit the right barrier for `HOST_ARCH`:
  - `x86_64`: `__asm__ volatile ("mfence" ::: "memory")`
  - `aarch64`: `__asm__ volatile ("dmb sy" ::: "memory")`
  - `armhf`: `__asm__ volatile ("dmb sy" ::: "memory")`
  Provide a `vortex_gem5_mmio_fence()` inline helper that compiles
  to the right barrier per `HOST_ARCH`.
- Multi-target build (B13 obsolete; replaced by clean multi-target
  support): `HOST_ARCH` make variable.

**Validation:**
- `make -C sw/runtime/gem5` (default `HOST_ARCH=x86_64`) builds.
  `file build/sw/runtime/libvortex-gem5-x86_64.so` confirms x86-64
  ELF.
- `make -C sw/runtime/gem5 HOST_ARCH=aarch64` builds (requires
  cross-toolchain, installed by Phase 0's `ci/gem5_install.sh`).
  `file build/sw/runtime/libvortex-gem5-aarch64.so` confirms
  AArch64 ELF.

### Phase 5 — End-to-end gem5 test

- `ci/gem5_test_vortex_hello.py` — gem5 Python config that wires:
  - A `System` with one `TimingSimpleCPU` core in SE mode (host ISA
    selected at runtime via `--host-arch=x86|arm`).
  - A `VortexGPGPU` device on `system.membus` at
    `pio_addr=0x20000000`, mapped into the process's address space.
  - The native-or-cross-compiled test binary
    (`tests/kernel/hello/hello` re-linked against the matching
    `libvortex-gem5-{x86_64,aarch64}.so`) as the SE-mode workload.
- `ci/gem5_test_vortex_vecadd.py` — same with a vecadd kernel that
  actually exercises DMA in both directions and the cache-flush path.
- Add a top-level wrapper test in `tests/regression/gem5/` (mirrors
  `tests/regression/dxa/`) that builds the kernels and invokes the
  Python scripts for both `HOST_ARCH=x86_64` and `HOST_ARCH=aarch64`.

**Validation:**
- `build/X86/gem5.opt ci/gem5_test_vortex_hello.py --host-arch=x86`
  exits with code 0 and the expected `Hello World` on stdout.
- `build/ARM/gem5.opt ci/gem5_test_vortex_hello.py --host-arch=arm`
  exits with code 0 and the expected `Hello World` on stdout.
- Both `ci/gem5_test_vortex_vecadd.py` variants exit 0 with the
  vecadd result buffer matching the CPU-computed reference (checked
  by the test binary itself).

### Phase 6 — CI integration

- Add `gem5()` function to `ci/regression.sh.in` (mirroring `sst()`
  on line ~80):
  ```bash
  gem5()
  {
      echo "begin gem5 tests..."

      make -C sim/simx USE_GEM5=1
      make -C tests/kernel

      # X86 default: native compile, no cross-toolchain needed.
      make -C sw/runtime/gem5 HOST_ARCH=x86_64
      cp sim/simx/libvortex-gem5.so $GEM5_HOME/build/X86/

      timeout 120 $GEM5_HOME/build/X86/gem5.opt \
          ci/gem5_test_vortex_hello.py  --host-arch=x86
      timeout 120 $GEM5_HOME/build/X86/gem5.opt \
          ci/gem5_test_vortex_vecadd.py --host-arch=x86

      # ARM matrix entry — requires gcc-aarch64-linux-gnu (installed
      # by ci/gem5_install.sh in Phase 0).
      if [ -n "$VORTEX_GEM5_ARM" ]; then
          make -C sw/runtime/gem5 HOST_ARCH=aarch64
          cp sim/simx/libvortex-gem5.so $GEM5_HOME/build/ARM/

          timeout 120 $GEM5_HOME/build/ARM/gem5.opt \
              ci/gem5_test_vortex_hello.py  --host-arch=arm
          timeout 120 $GEM5_HOME/build/ARM/gem5.opt \
              ci/gem5_test_vortex_vecadd.py --host-arch=arm
      fi

      echo "gem5 tests done!"
  }
  ```
  Per `feedback_test_timeout_120s.md`, every test invocation is
  `timeout 120`-capped. ARM is opt-in via `VORTEX_GEM5_ARM=1` so
  hosted CI without the ARM toolchain still passes; the self-hosted
  runner sets the env var.
- Add `gem5-x86` and `gem5-arm` matrix entries to
  `.github/workflows/ci.yml` (both run on the self-hosted runner
  only, per
  [`project_ci_machine.md`](../../../../.claude/projects/-home-blaisetine-dev/memory/project_ci_machine.md);
  the hosted runners do not have enough resources for a full
  gem5 build).
- Add `ci/gem5_install.sh` to the Apptainer recipe
  ([`miscs/apptainer/vortex.def`](../../miscs/apptainer/vortex.def))
  so the .sif has gem5 pre-installed. **Out of scope for Phase 6;
  see §8.**

**Validation:** `./ci/regression.sh --gem5` runs both
`gem5_test_vortex_*.py` cleanly on the self-hosted runner.

### Phase 7 — Documentation

- `docs/gem5_integration.md`:
  - How to install gem5 v25.0.0 (point at `ci/gem5_install.sh`).
  - How to build with `USE_GEM5=1`.
  - How to cross-compile the ARM runtime + kernels.
  - How to write a gem5 Python script that drives `VortexGPGPU`.
  - The single-source-of-truth invariant (§1.1) and the cache-flush
    contract (§3.4) for future hackers who might be tempted to skip
    the flush "because it's fast".

---

## 5. Authorship / history mechanics

- `sim/simx/gem5/vortex_gpgpu.{cpp,h}` and the gem5-side
  `vortex_gpgpu_dev.{cc,hh}` + `VortexGPGPU.py`: **new files**, no
  upstream equivalent. Commit body cites:
  > Replaces legacy `vortex_gem5/sim/simx/simx_device.{cpp,h}`
  > (Injae Shin, UCLA 2025-05-22 commit 91dcf17) and the gem5-side
  > SimObject described in his capstone report.
  > Re-implemented for SimX v3 Processor::cycle() API. Original
  > design intent (OPAE MMIO + pinned staging buffer + ARM SE-mode
  > runtime) preserved.

- `sw/runtime/gem5/`: **new files** mirroring `sw/runtime/opae/`'s
  shape. Same authorship attribution as above; the file-level
  similarity is to `sw/runtime/opae`, not to `runtime/opaesimx` from
  the legacy tree (which has the bugs catalogued in §2.2).

- `ci/gem5_install.sh.in` and `ci/gem5_test_vortex_*.py`: new files;
  follow the structure of `ci/sst_install.sh.in` and
  `ci/sst_test_vortex_*.py`. `ci/gem5_install.sh.in` lands in
  Phase 0 (initially installing the dummy SimObject); the test
  scripts land in Phase 5.

- `Processor::cycle()` / `Processor::memsim()`: new public API on
  `Processor`, lands in Phase 1. Single commit on the simx_v3 line;
  mentioned as a prerequisite of both SST and gem5 integrations in
  the commit body.

- `sim/simx/gem5/gem5_api_notes.md`: Phase 0 deliverable, scratch
  notes only — **not** committed to `docs/`. Captures the gem5
  v25.0.0 API surface our C ABI design depends on; deleted once
  Phase 2 commits the C ABI itself.

This is consistent with the rule established in
[`feedback_keep_ours_in_merge.md`](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_keep_ours_in_merge.md):
the legacy code is not a "theirs" we apply; it is a prior design that
informs our redesign. Credit the designer in the body; do not pretend
the bits are a port.

---

## 6. Validation

Each phase ends with the validation listed in §4. Across phases the
acceptance criteria are:

1. **No-gem5 build identical.** `make -C sim/simx` (default flags)
   produces a binary identical in behavior to today's on the
   regression suite (io_addr, arith, vecadd, mpi_vecadd, tensor*,
   dxa, dtm). The Phase 0 `Processor::cycle()` addition must not
   change `Processor::run()` semantics — verify by trace-diffing
   `vecadd` before and after Phase 0.

2. **In-process smoke (no gem5 needed).** `gem5_smoke hello.vxbin`,
   the Phase 2 driver, runs the same kernels the `simx` binary runs
   and produces matching output. This is the unit-test layer that
   shakes out C-ABI breakage without requiring gem5 to be installed
   beyond what Phase 0 already set up.

3. **End-to-end gem5 PASS.** Both `gem5_test_vortex_hello.py` and
   `gem5_test_vortex_vecadd.py` exit 0 under the pinned gem5 v25.0.0.1,
   on *both* `build/X86/gem5.opt` and `build/ARM/gem5.opt`, timed out
   at 120 s (each). The pin and the install path are both already
   validated by Phase 0; this validation just exercises the real
   `VortexGPGPU` SimObject end-to-end.

4. **No `core->mem_read` / `core->mem_write` regressions.** Phase 5
   of v3 forbids those
   ([simx_v3_proposal.md §3.3](simx_v3_proposal.md)). The grep gate
   from
   [master_merge_v3_proposal.md §8 R1](master_merge_v3_proposal.md)
   applies here: every commit must pass
   `git diff <pre>..<post> -- sim/simx/ | grep -E 'core->mem_(read|write)' | wc -l == 0`.

5. **Single source of truth check.** The gem5 device's pinned region
   is `RAM`-backed (i.e., a slice of host memory exposed to gem5's
   DRAM AddrRange via `mmap`); Vortex's VRAM is the `RAM` attached to
   `Memory` inside `vortex::Processor`. **There is no shadow image.**
   `vortex_gem5_dma_{read,write}` copies bytes between the two via
   `RAM::read`/`RAM::write` — no additional buffer level. Mistakes
   here re-introduce the §1.1 violation.

---

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | gem5 v25.0.0 `DmaDevice` API drifts in v26+. | Pin in `ci/gem5_install.sh.in` (Phase 0). Document the pin in `docs/gem5_integration.md`. CI catches regressions on bump. |
| R2 | ARM cross-compiler not available in the Apptainer recipe. | Phase 6 says gem5 CI is on the self-hosted runner only, which already has the ARM toolchain per [`project_ci_machine.md`](../../../../.claude/projects/-home-blaisetine-dev/memory/project_ci_machine.md). Apptainer absorption is out of scope (§8). |
| R3 | `MMIO_PINNED_BASE` PIO range collides with another gem5 device's PIO range. | Pick a default (`0x20000000`, matching the legacy paper) but make it a Python-configurable parameter (`pio_addr`). Phase 0 confirms the default is reachable from SE-mode in v25.0.0; document collisions in the integration guide. |
| R4 | The gem5 ARM CPU model reorders MMIO writes, breaking the args-then-CMD_TYPE protocol (B14). | `DMB SY` (AArch64) or `dmb sy` (ARMv7) before `CMD_TYPE` write in the runtime. Add a regression test that issues a back-to-back `CMD_MEM_WRITE` + `CMD_RUN` and verifies the kernel observed the correct args. |
| R5 | Future contributor re-introduces the host-pointer-MMIO hack (B3) "for convenience". | This proposal explicitly deletes that abstraction (§2.4). The follow-up `docs/gem5_integration.md` (Phase 7) should call this out. |
| R6 | `Processor::cycle()` for a never-launched kernel hangs (no `kmu_->start()` because `is_cycle_initialized_` was never reset). | Reset is implicit on first `cycle()`. If a second kernel is launched in the same device lifetime (rare; supported by gem5 only for back-to-back tests), the gem5 device's `CMD_RUN` handler must call a new `Processor::reset_cycle()` that clears `is_cycle_initialized_`. Add this in Phase 2. |
| R7 | The cross-compiled ARM `libvortex-gem5.so` and the gem5-loaded `libvortex-gem5.so` (x86) have the same SONAME and get confused at install time. | Suffix the ARM build (`libvortex-gem5-aarch64.so`) and the gem5 build (`libvortex-gem5.so`). Document in Phase 2+4. |
| R8 | gem5's `DmaPort` request size is unbounded; a 1 GB `CMD_MEM_WRITE` would burn simulated time. | Cap per-transaction size at 1 MB in the device's `CMD_MEM_*` handler; chunk larger requests into multiple DMA actions. Mirrors how the OPAE `fpgaPrepareBuffer` page-aligns transfers. |
| R9 | Cache flush via `CMD_DCR_READ` returns synchronously per core; for `NUM_CORES * NUM_CLUSTERS = 16` that is 16 PIO round-trips per download. | Acceptable for Phase 5; can be batched into a single `CMD_FLUSH_ALL` MMIO later if measured to hurt. |
| R10 | The gem5 SimObject install (`sim/simx/gem5/install.sh`) modifies the gem5 source tree in place; rebuilds can leave stale artifacts. | `install.sh` is idempotent (copies, doesn't patch); `ci/gem5_install.sh` does a clean `scons -c` before re-build on toolchain version mismatch. Phase 0 proves the install path end-to-end with a dummy SimObject before any real code depends on it. |
| R11 | Phase 0 reveals gem5 v25.0.0 has dropped SE-mode PIO mapping (the legacy `0x20000000` magic). | Switch design to the `/dev/vortex_gem5` pseudo-file path (§3.6 option 2) before Phase 2 commits the C ABI. Cost: ~1 week added to Phase 0 redesign window. Acceptable because Phase 0 is explicitly a gate — no downstream phase has shipped code yet. |
| R12 | Phase 0 install takes hours on first run; blocks parallel work. | Cache the `$TOOLDIR/gem5-src/build` directory in CI the same way SST and toolchain caches work. Self-hosted runner's local toolchain dir survives across runs. |

---

## 8. Out of scope

- **Apptainer integration.** Adding gem5 + the ARM cross-toolchain
  to `miscs/apptainer/vortex.def` is a separate concern. Until that
  is done, `apptainer-ci.yml`'s matrix should not include `gem5`. The
  self-hosted runner runs the gem5 matrix entry on hosted ci.yml; the
  Apptainer pipeline skips it. See
  [`apptainer-ci.yml` policy notes](../../.github/workflows/apptainer-ci.yml).

- **Full-system Linux on gem5.** The capstone paper restricts itself
  to SE-mode (per the paper's §IIC: "gem5-Vortex's implementation
  allows users to use gem5's system call emulation (SE) mode"). This
  proposal does the same. FS-mode requires booting a Linux kernel
  inside gem5 with a Vortex device driver — possible, but a separate
  redesign that intersects with kernel-mode driver work the project
  has not started.

- **Multi-device simulation.** One `VortexGPGPU` per gem5 system.
  Multi-device support requires per-instance PIO ranges and a runtime
  side that supports `vx_dev_open` returning >1 handle — the legacy
  `g_callbacks` global (B11) blocks this on the runtime side, and
  the device side needs per-instance state isolation. Defer.

- **AMD GPU / NoMali comparison.** The capstone paper compares
  gem5-Vortex to NoMali (stub GPU) and AMD GPU (full-system). Those
  comparisons live in the paper; reproducing them as benchmarks is
  out of scope. Comparing performance to SimX standalone or to the
  SST integration is also out of scope — separate analysis work.

- **DMA performance modeling.** The capstone paper §V measures DMA
  delay variation per kernel size. Replicating that as a CI
  performance gate is out of scope; could be a follow-up perf
  proposal once the integration is stable.

- **SST + gem5 simultaneous.** Both integrations replace different
  parts of the harness; running them together is not a use case
  anyone has asked for. Build flags are mutually exclusive:
  `USE_SST=1` and `USE_GEM5=1` together is rejected by `sim/simx/Makefile`.

- **gem5 fork branch.** We do not maintain a long-lived fork of gem5.
  `ci/gem5_install.sh` fetches a clean release tarball and applies
  our SimObject; if the user wants a persistent gem5 working tree,
  that is their setup. Avoids the "fork rot" that froze
  `vortex_gem5`.

- **Runtime gem5/non-gem5 switching.** Keep `USE_GEM5=1` as a
  build-time switch. A runtime switch would require both `Processor`
  and a gem5 wrapper in every binary plus a factory; not worth the
  maintenance cost for a single-device research integration.

---

## 9. Estimated effort

Based on the SST integration in
[sst_simx_v3_proposal.md §9](sst_simx_v3_proposal.md) (~15–28 h):

- **Phase 0** (gem5 env + API survey + dummy SimObject install):
  **6–10 h estimated; ✅ COMPLETE 2026-05-16** in ~3 h of
  attended + ~25 min unattended scons build. The wall time to
  install gem5 was 13 min (ARM) + 11 min (X86) parallel on the
  self-hosted 64-core runner. All six validations
  (see `sim/simx/gem5/gem5_api_notes.md`) pass on both ISAs.
  Key discoveries committed: (1) SE-mode PIO attachment is
  possible but requires bypassing the `SimpleBoard` high-level
  API; (2) out-of-tree SimObject install needs **no** top-level
  SConstruct patch — pure `cp -r`; (3) PCIe (Path 2 in §3.6) is
  a clean Phase 5+ upgrade because `PciDevice` inherits
  `DmaDevice` and shares the same C ABI surface.
- **Phase 1** (`Processor::cycle()` + `memsim()`): **1–2 h estimated;
  ✅ COMPLETE 2026-05-16** in ~1 h. ~50-line patch to
  `processor.{cpp,h}` + `processor_impl.h`. Default `make -C
  sim/simx` and `USE_SST=1` both build clean; `simx hello.vxbin`
  prints `#0: Hello World!`. **Bonus:** the SST integration was
  previously broken at the `proc_->cycle()` call site
  (`sim/simx/sst/vortex_simulator.cpp:64`) and would not link; with
  Phase 1 in place, `sst ci/sst_test_vortex_hello.py` runs
  end-to-end and exits cleanly at 4.643 µs simulated time.
- **Phase 2** (`libvortex-gem5.so` + C ABI + in-process smoke):
  **4–6 h estimated; ✅ COMPLETE 2026-05-16** in ~1.5 h. Files added:
  `sim/simx/gem5/vortex_gpgpu.{h,cpp}` (the C ABI library) and
  `sim/simx/gem5/gem5_smoke_main.cpp` (the in-process smoke driver).
  `sim/simx/Makefile` extended with a `USE_GEM5=1` gate that
  produces `libvortex-gem5.so` (1.5 MB) + `gem5_smoke` (16 KB
  driver linking against the lib). `gem5_smoke hello.vxbin` →
  `#0: Hello World!`, 4642 cycles, exit_code=0 (correctly read back
  via `vortex_gem5_vram_read` after the cache-flush DCR path —
  validating B9 from §2.2 is fixed). Default `make -C sim/simx`
  unchanged (only `simx` produced; gem5 sources fully gated).
  `USE_SST=1 USE_GEM5=1` correctly rejected by the Makefile per
  §8 (mutual exclusion). Side fix: `sw/common/bitmanip.h` was
  missing `<type_traits>` and `<algorithm>` includes — header
  hygiene fix benefits any caller (per
  [feedback_always_correct_fix_not_patch](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_always_correct_fix_not_patch.md)).
- **Phase 3** (gem5 SimObject + Python + install.sh): **6–10 h
  estimated; ✅ COMPLETE 2026-05-16** in ~1.5 h. Files added:
  `sim/simx/gem5/vortex_gpgpu_dev.{cc,hh}` (gem5 `DmaDevice` subclass
  with `dlopen` + `EventFunctionWrapper` tick scheduling),
  `sim/simx/gem5/VortexGPGPU.py` (Python binding with `library=` +
  `kernel=` parameters), `sim/simx/gem5/SConscript`. Updated
  `install.sh` to install the real device and remove the Phase 0
  dummy scaffolding from `$GEM5_HOME` cleanly. New test:
  `ci/gem5_test_vortex_hello.py` (standalone-device variant, no
  host CPU needed). Validation: both `build/X86/gem5.opt` and
  `build/ARM/gem5.opt` import `VortexGPGPU` and run hello.vxbin to
  completion at tick 4,643,000 (1 GHz clock → 4643 cycles, matching
  Phase 1 SST + Phase 2 in-process within 1 cycle). **Three
  harnesses now validated through the same `Processor::cycle()` API:
  SST, in-process C ABI, and gem5 SimObject.**
- **Phase 4** (host runtime, x86 + ARM): **6–10 h estimated; ✅ x86
  PATH COMPLETE 2026-05-16** in ~1 h; aarch64 cross-build gated on
  the user's `sudo apt install gcc-aarch64-linux-gnu`. Files added:
  `sw/runtime/gem5/driver.{cpp,h}` (direct MMIO + mmio_fence helper
  with per-arch barrier; bump-allocator for the pinned region),
  `sw/runtime/gem5/vortex.cpp` (OPAE-shaped `vx_device` with the
  full callback table — compile-time caps from VX_config.h since
  the host runtime and the device library are built from the same
  source tree), `sw/runtime/gem5/Makefile` (HOST_ARCH ∈
  {x86_64,aarch64,armhf} → matching cross-compiler; produces
  `libvortex-gem5-$ARCH.so`). All three B-bugs addressed: B9 (cache
  flush before download via per-core `dcr_read(VX_DCR_BASE_CACHE_FLUSH,
  cid)`), B13 (per-arch compiler via `HOST_ARCH`), B14 (mmio_fence()
  centralised in `issue_cmd()` so every CMD_TYPE write is fenced
  by construction). Validation: `make -C sw/runtime/gem5 HOST_ARCH=x86_64`
  → `libvortex-gem5-x86_64.so` (43 KB, ELF 64-bit x86-64, SONAME
  correct, exports `vx_dev_init` matching the OPAE/SimX backend
  pattern).
- **Phase 5** (end-to-end gem5 tests): **4–6 h estimated; ✅ x86
  PATH COMPLETE 2026-05-17** in ~3 h. The bulk of the work turned
  out to be the OPAE state machine on the device side (cmd_args
  latching, busy bit, dcr_rsp register) plus the dmaAction
  dispatch in the SimObject — the test scripts themselves were
  small. Files added:
  `ci/gem5_test_vortex_vecadd.py` (full e2e: x86 CPU + identity-mapped
  PIO+PIN regions + Process.map() + Vortex device). The Phase 3
  standalone `ci/gem5_test_vortex_hello.py` continues to pass as a
  fast smoke test. Phase 5 also extended Phase 2's
  `sim/simx/gem5/vortex_gpgpu.{cpp,h}` with the full OPAE protocol
  state machine and Phase 3's `sim/simx/gem5/vortex_gpgpu_dev.cc`
  with `pop_pending_cmd` → `dmaRead`/`dmaWrite` dispatch.
  Validation: `vecadd -n16` PASSED!, kernel ran 454 cycles at
  IPC 0.247 on 4×4 threads/warps. Side fix: glibc's `nanosleep()`
  routes through `clock_nanosleep` (#230) which gem5 SE-mode
  doesn't implement — switched the host runtime's poll-loop back-off
  to `sched_yield()` (in gem5's syscall table). ARM e2e gated on
  user `sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`
  (same gate as Phase 4's aarch64 build).
- **Phase 6** (CI): **2–3 h estimated; ✅ COMPLETE 2026-05-17** in
  ~30 min. Added `gem5()` function to `ci/regression.sh.in`
  (mirrors `sst()` shape; builds prerequisites + runs both Phase 3
  standalone and Phase 5 e2e tests via `timeout 120` per
  [feedback_test_timeout_120s](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_test_timeout_120s.md);
  ARM matrix opt-in via `VORTEX_GEM5_ARM=1`). Added `--gem5` case
  dispatch + `--gem5` to the show_usage line. Updated
  `.github/workflows/ci.yml`: appended `ci/gem5_install.sh` to the
  `Setup Toolchain` step (gated on `cache-toolchain.outputs.cache-hit`
  like SST), added `Export gem5 paths` step (GEM5_HOME + PATH for
  `build/X86`), added `gem5` to the `tests.matrix.name` list with
  `exclude: name=gem5 xlen=64` (the device library is XLEN-locked
  by the gem5 install; one entry is sufficient). Validation:
  `./ci/regression.sh --gem5` PASSED end-to-end in **5 seconds**
  (Phase 3 hello standalone + Phase 5 vecadd e2e, both clean).
- **Phase 7** (docs): **1–2 h estimated; ✅ COMPLETE 2026-05-17** in
  ~45 min. Added `docs/gem5_integration.md` covering: install
  (`ci/gem5_install.sh`), Vortex+gem5 build (`USE_GEM5=1`), host
  runtime cross-compile (`HOST_ARCH`), running tests
  (`./ci/regression.sh --gem5` and standalone hand commands),
  a complete minimal Python recipe for hosting Vortex in a custom
  gem5 system, **six load-bearing invariants** (Process.map order,
  identity-mapped PIO+PIN, cache flush before download, MMIO
  fence, single source of truth for memory, USE_SST/GEM5 mutex),
  architectural choices worth revisiting (doorbells vs. polling,
  PCIe upgrade path, C ABI rationale), CI integration, and a
  troubleshooting table covering the 6 most common error modes
  (wrong library path, missing LD_LIBRARY_PATH, clock_nanosleep
  syscall, orphan Process, wrong `library=` param, busy-bit hang,
  ccache stale objects). Added to `docs/index.md`.

Total: **~30–49 hours** of focused work (was ~26–41 h before Phase 0
was added as a separate phase; the actual work has not grown — the
gem5 install was implicit in the old Phase 2 estimate and is now
explicit in Phase 0). Substantial enough to warrant its own branch
(`gem5_simx_v3` or similar).

**Sequencing with SST:** Phase 1 (`Processor::cycle()`) is shared;
do it once and both integrations benefit. If SST lands first, gem5
reuses `Processor::cycle()` unchanged. If gem5 lands first, the SST
integration's broken `proc_->cycle()` reference
(`sim/simx/sst/vortex_simulator.cpp:64`) gets fixed as a side effect
of Phase 1 — net win for both. Phase 0 is gem5-only; SST integration
does not benefit from it.

# gem5 Integration — Design

**Scope:** the architecture of running Vortex inside the gem5
full-system/syscall-emulation simulator — the gem5 device model that wraps
the SimX core ([`sim/simx/gem5/`](../../sim/simx/gem5/)) and the runtime
backend the simulated host uses to drive it
([`sw/runtime/gem5/`](../../sw/runtime/gem5/)).

This document is **architectural**; the build/install/run mechanics and
the load-bearing gem5 Python invariants are in the how-to
[`docs/gem5_integration.md`](../gem5_integration.md) and are not repeated
here.

---

## 1. The two-domain model

Under gem5 there are two domains:

- **Simulated host** — the runtime runs as ordinary code on a gem5 CPU
  model (`AtomicSimpleCPU`, x86_64 or aarch64).
- **Device** — the CP and the Vortex `Processor` run **natively** inside
  the `VortexGPGPU` gem5 SimObject (a dlopened device library).

The only shared state is device VRAM, reached by the host through an
identity-mapped BAR window and by the CP/Vortex through in-process
`simx::RAM`.

```
   simulated host (gem5 CPU, x86/arm)            VortexGPGPU SimObject (native)
   ───────────────────────────────              ──────────────────────────────
   libvortex.so (dispatcher)                     libvortex-gem5.so:
     │  callbacks_t HAL                            Gem5Device {
     ├─ cp_reg_write ── PIO 32b ──► [pio_addr] ──►   vortex::CommandProcessor cp_
     ├─ cp_reg_read  ◄─ PIO 32b ──  [pio_addr] ◄─    Processor proc_
     └─ host_mem_alloc ─► VRAM ◄── [pin_addr] ──►    RAM ram_ / InProcessDevMem
                                                   }
                                            cpTickEvent_ / vortexTickEvent_ (gem5 events)
```

This is the only backend where the CP lives **across the bus** from the
runtime (see §4).

---

## 2. Device library (`sim/simx/gem5/`)

Builds `libvortex-gem5.so` (`USE_GEM5=1`), dlopened by the gem5 SimObject.

- [`vortex_gpgpu.{h,cpp}`](../../sim/simx/gem5/vortex_gpgpu.cpp) — the C
  ABI contract (`vortex_gem5_create/destroy`, `load_kernel`,
  `cp_mmio_{write,read}`, `cp_tick/cp_has_work`, `vortex_tick/vortex_busy`,
  `vram_{read,write}`). `Gem5Device` owns `RAM ram_`, `Processor proc_`,
  `InProcessDevMem dev_mem_`, and an embedded
  `vortex::CommandProcessor cp_`
  ([`vortex_gpgpu.cpp:46-54`](../../sim/simx/gem5/vortex_gpgpu.cpp#L46)).
  CP hooks (`dram_read/write`, `vortex_dcr_write/read`, `vortex_start`,
  `vortex_busy`) are bound in `make_cp_hooks()`
  ([`:165-189`](../../sim/simx/gem5/vortex_gpgpu.cpp#L165)).
- [`dev_mem.{h,cpp}`](../../sim/simx/gem5/dev_mem.cpp) — the
  `DevMemAccessor` seam; `InProcessDevMem` wraps `simx::RAM` (the only
  implementation today; `DmaPortDevMem` is the unbuilt v2 seam, §6).
- [`vortex_gpgpu_dev.{cc,hh}`](../../sim/simx/gem5/vortex_gpgpu_dev.cc) —
  the gem5 `VortexGPGPU : public DmaDevice`. dlopens the library and
  resolves a 13-symbol ABI struct up front; routes PIO packets in
  `[pio_addr, +pio_size)` to `cp_mmio_{read,write}` (32-bit) and packets
  in the PIN range to `vram_{read,write}`. Two self-scheduling events
  `cpTickEvent_`/`vortexTickEvent_` (§3).
- [`VortexGPGPU.py`](../../sim/simx/gem5/VortexGPGPU.py) — SimObject
  params (`pio_addr=0x20000000`, `pio_size=0x0200`,
  `pin_addr=0x100000000`, `max_queues=4`).
- `SConscript`, `install.sh` — gem5 source registration + install.

The device exposes Vortex as: **one 32-bit PIO range that *is* the CP
regfile** (no OPAE window, no AFU `+0x1000` split) and **one BAR-mapped
VRAM range**. Despite the `DmaDevice` base class, host↔VRAM today is plain
in-process `simx::RAM` access, not gem5 `DmaPort` traffic — the
`DmaDevice` base is kept only as the v2 seam.

---

## 3. Event-driven control

The SimObject drives two independent gem5 event chains:

- `cpTickEvent_` self-schedules only while `cp_has_work()` (CP enabled and
  busy). A host doorbell PIO write (`Q_TAIL_HI`) triggers `maybeWakeCp()`
  ([`vortex_gpgpu_dev.cc:192-198`](../../sim/simx/gem5/vortex_gpgpu_dev.cc#L192)).
- `vortexTickEvent_` self-schedules only while `vortex_busy()`. A
  `CMD_LAUNCH` retirement fires the `vortex_start` hook → trampoline →
  schedules the Vortex chain
  ([`:200-212`](../../sim/simx/gem5/vortex_gpgpu_dev.cc#L200)).

Idle is observable as both events unscheduled — no polled-every-cycle, no
bounded tick burst. Standalone vs. hosted mode is chosen in `startup()` by
whether a `kernel=` param is set.

---

## 4. Runtime backend (`sw/runtime/gem5/`)

Builds `libvortex-gem5-<arch>.so`, loaded by the simulated host process.

- [`vortex.cpp`](../../sw/runtime/gem5/vortex.cpp) implements the
  **pure-v2 `callbacks_t` transport HAL** via `#include <callbacks.inc>`:
  `dev_open/close`, `cp_reg_read/write` (32-bit PIO + fence),
  `host_mem_alloc/free` (carves a 64 MB aperture at the top of the PIN
  window onto VRAM). No CP, no kernel logic — pure transport.
- [`driver.{h,cpp}`](../../sw/runtime/gem5/driver.cpp) — fixed VAs
  (`PIN_BASE_ADDR=0x100000000`, `PIO_BASE_ADDR=0x20000000`); `mmio_*32`
  are raw volatile derefs; `mmio_fence()` emits `mfence`/`dmb sy` per arch.
- `Makefile` — `HOST_ARCH ∈ {x86_64, aarch64, armhf}` selects the
  compiler and emits `libvortex-gem5-<arch>.so`.

**CP side-of-boundary asymmetry (key architectural point).** In
simx/rtlsim the CP model lives in the **host runtime** (`cp_reg_write`
does `cp_.mmio_write` plus a bounded tick burst). In gem5 the CP lives in
the **device library** and the host runtime does **real MMIO** to it
(`cp_reg_write` → `mmio_write32` at `PIO_BASE`). Same `callbacks_t`
surface, same `vortex::CommandProcessor` class
([`sim/common/cmd_processor.{h,cpp}`](../../sim/common/cmd_processor.cpp)),
opposite side of the host/device boundary. The CP command-building
(rings, `CMD_LAUNCH`, doorbell, `Q_SEQNUM` poll) lives in the shared
common-core dispatcher
([`sw/runtime/common/device.cpp`](../../sw/runtime/common/device.cpp)),
identical to every other backend. See
[`command_processor_control_plane.md`](command_processor_control_plane.md).

**ISA portability.** The device side is always an x86 gem5 binary
regardless of simulated ISA. Only the host runtime changes ISA: x86_64
native, aarch64 cross-compiled (opt-in via `VORTEX_GEM5_ARM=1`), armhf
supported but 32-bit-limited (BAR above 4 GiB unreachable). CI runs gem5
via `ci/regression.sh --gem5` (hostless `hello`, e2e `vecadd`/`sgemm`),
not via `ci_xlen{32,64}.sh`.

---

## 5. Kernel-run path (hosted)

`vx_device_open` → dispatcher dlopens the host backend → `dev_open`. Queue
create → `host_mem_alloc` rings/head/cmpl in VRAM, `cp_reg_write` programs
Q0 and enables. Launch → the dispatcher writes `CMD_DCR_WRITE` +
`CMD_LAUNCH` descriptors into the ring (host stores through the PIN
window), then rings the `Q_TAIL_HI` doorbell. The SimObject's
`maybeWakeCp()` → `cp_.tick()` fetches the ring cache line, routes DCRs via
`vortex_dcr_write`, and `vortex_start` schedules the Vortex chain; the CP
launch FSM waits on `vortex_busy()`, then retires and writes the seqnum.
The host polls `cp_reg_read(Q_SEQNUM)`. Ring, staging, and VRAM are all the
same in-process `simx::RAM` bytes, so memory is single-source-of-truth.

---

## 6. Proposed but not yet implemented

1. **v2 DMA-port memory seam** (`gem5_v2_cp_migration` §2.5). The
   `DevMemAccessor` interface exists, but `DmaPortDevMem` backing VRAM
   through gem5's `SimpleMemory` over the SimObject DMA port is not built —
   VRAM is in-process `simx::RAM` despite the `DmaDevice` base class. The
   seam is the design's whole point; preserve the intent.
2. **Multi-queue host runtime** (`gem5_v2_cp_migration` §2.6). The PIO map
   reserves 4 queues (`max_queues=4`) but the CP model is single-queue
   (`q0_`) and the host exercises Q0 only — the growth path for vortex2.h
   multi-queue.
3. **PCIe `PciDevice` / BAR upgrade** (`gem5_simx_v3` §3.6): the C ABI is
   shape-compatible; only the gem5 wrapper class changes. Doorbell-ring
   realism (vs. today's `Q_SEQNUM` polling) is the matching upgrade.
4. **FS-mode Linux + a kernel driver** (both proposals, out of scope) —
   SE-mode only today.
5. **Multi-device** (one `VortexGPGPU` per system today) and a **separate
   ClockDomain** for CP vs. Vortex (single-domain in v1).
6. **Profiling timestamp writeback** — arrives "for free" once the CP
   `F_PROFILE` path lands (see `command_processor_control_plane.md` §10).

**Known discrepancies to fix** (not future work): the gem5 entry in
`.github/workflows/ci.yml` lists both `xlen: [32, 64]` with **no
`exclude:`** for xlen=64, contradicting the project's 32-bit-only gem5
policy; and `ci/gem5_run_app.py` carries **stale comments** describing the
superseded `cp_mmio_write`/`mem_upload` HAL rather than the landed
`cp_reg_*` + `host_mem_alloc` transport.

**Superseded directions** (recorded to avoid revival): the OPAE-style MMIO
command FSM (`CMD_TYPE`/`CMD_RUN`/`MMIO_STATUS`) and staging-buffer DMA
protocol of `gem5_simx_v3` (torn out — grep-clean of OPAE leftovers); and
the `cp_mmio_{read,write}` + `mem_upload/download/copy` HAL signature of
`gem5_v2_cp_migration` (replaced by the 5-callback `cp_pure_v2` transport
HAL the whole runtime tree now shares). The `gem5_v2_cp_migration` claim
that `vortex_dcr_read` is not a hook was also discarded — the shipped
`CommandProcessor::Hooks` keeps it as a real 6th hook.

---

## 7. Source proposals

This design consolidates and supersedes the following proposals (now
removed from `docs/proposals/`): `gem5_simx_v3_proposal.md`,
`gem5_v2_cp_migration_proposal.md`. The build/run how-to remains in
[`docs/gem5_integration.md`](../gem5_integration.md); the CP architecture
is in [`command_processor_control_plane.md`](command_processor_control_plane.md).

# gem5 Integration

Vortex runs inside the [gem5](https://www.gem5.org/) simulator as a
`DmaDevice` SimObject, exposing the Vortex GPGPU to a simulated host
CPU (x86 or ARM) through a Command Processor regfile + BAR-mapped
VRAM. Use this when you want to model heterogeneous host-CPU +
accelerator workloads with realistic cross-ISA cache and DMA timing,
or to validate the v2 Command Processor architecture against a real
host/device split.

This document covers both the **architecture** of the integration and
the **build / install / run** mechanics for the current (v2 CP-first)
design. The CP architecture itself is in
[`command_processor_control_plane.md`](command_processor_control_plane.md).
The earlier OPAE-protocol and v2 CP-migration proposals have been
consolidated into this document (see *Source proposals* at the end).

## At a glance

Three parts live in this repo:

| Part | Source | Built artifact | Loaded by |
|---|---|---|---|
| Device library | `sim/simx/gem5/vortex_gpgpu.{cpp,h}` + `dev_mem.{cpp,h}` | `build/sim/simx/libvortex-gem5.so` | gem5 SimObject via `dlopen` |
| gem5 SimObject | `sim/simx/gem5/vortex_gpgpu_dev.{cc,hh}` + `VortexGPGPU.py` + `SConscript` | Linked into `gem5.opt` after install | gem5 itself |
| Host runtime | `sw/runtime/gem5/{vortex.cpp,driver.{cpp,h},Makefile}` | `build/sw/runtime/libvortex-gem5-{x86_64,aarch64}.so` | The simulated process inside gem5 |

Plus `ci/gem5_install.sh` which fetches gem5 v25.0.0.1, drops the
SimObject sources into `$GEM5_HOME/src/dev/vortex/`, and builds
`build/{X86,ARM}/gem5.opt`.

## Architecture in one paragraph

The simulated host process loads the upstream dispatcher
(`libvortex.so`) which dlopens the gem5 backend
(`libvortex-gem5-x86_64.so`). The backend's only platform primitives
are `mem_upload/download/copy` (regular memcpy through a host-visible
BAR mapped to device VRAM) and `cp_mmio_{read,write}` (32-bit PIO to
the device's CP regfile). All kernel launches, DCR programming, and
fences flow through the dispatcher's Command Processor submission
path: it writes `CMD_*` descriptors into a ring buffer in device VRAM
(via mem_upload), commits via `cp_mmio_write(Q_TAIL_HI, ...)`, and
polls completion via `cp_mmio_read(Q_SEQNUM, ...)`. The CP itself is
the upstream `vortex::CommandProcessor` C++ class embedded in the
device library; the SimObject ticks it on its own gem5 event chain
and ticks the Vortex Processor on a parallel chain. Both event chains
self-schedule only while they have work — the device is genuinely
idle between commands.

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
through the catalog's `gem5` category (hostless `hello`, e2e
`vecadd`/`sgemm`), whose `via: script` calls `ci/regression.sh --run gem5`.

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

## One-time setup

Vortex install / build as usual ([docs/install_vortex.md](../install_vortex.md)),
then add gem5:

```bash
cd build/   # standard Vortex out-of-tree build directory
./ci/gem5_install.sh
```

This runs `sudo apt install` for gem5's build dependencies (scons,
libprotobuf, m4, libboost, **gcc-aarch64-linux-gnu**, …), clones gem5
v25.0.0.1 into `$TOOLDIR/gem5`, copies the Vortex SimObject sources
into `$GEM5_HOME/src/dev/vortex/`, and builds `gem5.opt` for both X86
and ARM (~15 min on a 64-core machine, ~30-45 min on a typical CI
runner). The script is idempotent — re-running with the same
`GEM5_REV` is a no-op.

To install only one ISA:

```bash
GEM5_TARGETS="X86" ./ci/gem5_install.sh   # default
GEM5_TARGETS="ARM" ./ci/gem5_install.sh
GEM5_TARGETS="X86 ARM" ./ci/gem5_install.sh   # both (default)
```

The pinned gem5 revision lives in `VERSION` (`GEM5_REV=v25.0.0.1`);
bumping it requires re-running `ci/gem5_install.sh` and verifying
both `gem5.opt` builds still load `VortexGPGPU` cleanly.

## Building Vortex with gem5 support

The device library is gated behind `USE_GEM5=1`. The default
`make -C sim/simx` is **unchanged** — no gem5 dep, no `libvortex-gem5.so`
produced.

```bash
make -C sim/simx                     # default; no gem5 artifacts
make -C sim/simx USE_GEM5=1          # produces libvortex-gem5.so + gem5_smoke
```

`USE_SST=1` and `USE_GEM5=1` are mutually exclusive (the Makefile
errors out if both are set).

### Host runtime + tests (cross-compile)

The simulated process inside gem5 loads the **host runtime**
`libvortex-gem5-$HOST_ARCH.so`, which exposes the pure-v2 `callbacks_t`
to the dispatcher. The `HOST_ARCH` knob is consistent across three
Makefiles — runtime backend, stub, and regression tests:

```bash
# Native x86 (default)
make -C sw/runtime/stub                          # → build/sw/runtime/libvortex.so
make -C sw/runtime/gem5                          # → build/sw/runtime/libvortex-gem5-x86_64.so
make -C tests/regression/vecadd                  # → build/tests/regression/vecadd/vecadd

# Cross-compiled aarch64 — outputs land in $arch/ subdirs so x86
# and ARM artifacts coexist:
make -C sw/runtime/stub HOST_ARCH=aarch64        # → build/sw/runtime/aarch64/libvortex.so
make -C sw/runtime/gem5 HOST_ARCH=aarch64        # → build/sw/runtime/aarch64/libvortex-gem5-aarch64.so
make -C tests/regression/vecadd HOST_ARCH=aarch64 # → build/tests/regression/vecadd/vecadd-aarch64

# armhf works the same way (note: armhf is 32-bit so the BAR
# mapping above 4 GiB is out of reach — only standalone tests work):
make -C sw/runtime/stub HOST_ARCH=armhf
make -C sw/runtime/gem5 HOST_ARCH=armhf
```

The ARM targets require `gcc-aarch64-linux-gnu` /
`gcc-arm-linux-gnueabihf` respectively — `ci/gem5_install.sh`
installs these.

## Running tests

### From the regression harness

```bash
cd build/
./ci/regression.sh --gem5
```

Runs both the standalone Phase-3 smoke test (kernel preloaded on the
SimObject, no host CPU) and the Phase-5 end-to-end test (real SE-mode
host program drives the device through CP submissions).

To also run the ARM matrix entry (needs `gcc-aarch64-linux-gnu`):

```bash
VORTEX_GEM5_ARM=1 ./ci/regression.sh --gem5
```

Runs 6 tests:
- X86 standalone hello (no host CPU; SimObject preloads kernel)
- X86 e2e vecadd `-n16` (host CPU drives device via CP regfile)
- X86 e2e sgemm `-n4`
- ARM standalone hello
- ARM e2e vecadd `-n16`
- ARM e2e sgemm `-n4`

Cross-arch e2e relies on two gem5 mechanisms working together:

1. **`setInterpDir(prefix)`** prepends a sysroot to the dynamic
   linker path embedded in the cross-compiled ELF
   (`/lib/ld-linux-aarch64.so.1` → `/usr/aarch64-linux-gnu/lib/...`).
   The Python config calls this when `VORTEX_DRIVER=gem5-aarch64`.
2. **`system.redirect_paths`** redirects the *guest process's*
   open()/stat() syscalls for `/lib/aarch64-linux-gnu/*` →
   `/usr/aarch64-linux-gnu/lib/*` so the dynamic linker can resolve
   libc, libstdc++, etc.

Both paths point at the Ubuntu `gcc-aarch64-linux-gnu` package's
install location — no extra setup needed.

### By hand

**Hostless** (no host CPU; kernel preloaded via SimObject parameter):

```bash
VORTEX_GEM5_DEV_LIB=$(pwd)/sim/simx/libvortex-gem5.so \
VORTEX_TEST_DIR=$(pwd)/tests/kernel/hello \
VORTEX_TEST_KERNEL=hello.vxbin \
    $GEM5_HOME/build/X86/gem5.opt ci/gem5_run_hostless_app.py
```

`VORTEX_TEST_KERNEL` defaults to `kernel.vxbin`, so any standard
regression test's kernel can be driven hostless without the host
binary — e.g. `VORTEX_TEST_DIR=$(pwd)/tests/regression/vecadd
ci/gem5_run_hostless_app.py`.

**End-to-end** — any standard Vortex regression test (host binary +
kernel.vxbin) runs through the generic
[`ci/gem5_run_app.py`](../../ci/gem5_run_app.py) runner.

```bash
# vecadd
VORTEX_GEM5_DEV_LIB=$(pwd)/sim/simx/libvortex-gem5.so \
VORTEX_GEM5_HOST_RT_DIR=$(pwd)/sw/runtime \
VORTEX_TEST_DIR=$(pwd)/tests/regression/vecadd \
VORTEX_TEST_BIN=vecadd \
VORTEX_TEST_ARGS="-n16" \
    $GEM5_HOME/build/X86/gem5.opt ci/gem5_run_app.py

# sgemm
VORTEX_GEM5_DEV_LIB=$(pwd)/sim/simx/libvortex-gem5.so \
VORTEX_GEM5_HOST_RT_DIR=$(pwd)/sw/runtime \
VORTEX_TEST_DIR=$(pwd)/tests/regression/sgemm \
VORTEX_TEST_BIN=sgemm \
VORTEX_TEST_ARGS="-n4" \
    $GEM5_HOME/build/X86/gem5.opt ci/gem5_run_app.py
```

Expected output ends with:
```
PASSED!
```

### Sizing tests for the 120 s budget

Tests are bounded by the project's 120 s per-test budget. gem5 SE-mode
runs the host CPU's CP poll loop in simulated time too, so **kernel
runtime + dispatcher poll budget translate directly into gem5 wall
time**. The regression script's default sizes fit; larger sizes are
fine when run by hand outside the budget cap.

## Address space layout

```
Host process VA (simulated, gem5 SE-mode) | Simulated PA | Backed by
------------------------------------------+--------------+----------------------
[0x0000_0000_0000, 0x0000_1000_0000)      | same         | gem5 DDR3 (process
                                          |              |   heap/stack/code)
[0x0000_2000_0000, 0x0000_2000_0200)      | same         | VortexGPGPU CP regfile
                                          |              |   (32-bit PIO)
[0x0001_0000_0000, 0x0002_0000_0000)      | same         | VortexGPGPU VRAM
                                          |              |   (BAR-mapped to
                                          |              |    in-process simx::RAM)
```

PIN_BASE_ADDR = `0x100000000` is identity-mapped via `Process.map()`
so host stores at PIN_BASE+dev_addr land in the same in-process
simx::RAM bytes the CP and Vortex read. PIO_BASE_ADDR = `0x20000000`
is identity-mapped (cacheable=False) so the dispatcher's PIO MMIO
reaches the SimObject's regfile decoder.

These constants are duplicated in two places — `sw/runtime/gem5/driver.h`
and `ci/gem5_run_app.py`. If you change one, change the other.

## Writing your own gem5 Python script

The minimal recipe for hosting Vortex inside a custom gem5 system:

```python
from m5.objects import (
    AddrRange, AtomicSimpleCPU, DDR3_1600_8x8, MemCtrl, Process,
    Root, SEWorkload, SrcClockDomain, System, SystemXBar,
    VoltageDomain, VortexGPGPU,
)

# Mappings expected by sw/runtime/gem5/driver.h.
PIO_BASE, PIO_SIZE = 0x20000000, 0x0200          # CP regfile (32-bit)
PIN_BASE, PIN_SIZE = 0x100000000, 0x100000000    # BAR-mapped VRAM
NUM_CPUS = 4   # >=2 required for the dispatcher's per-Queue worker thread

system = System()
system.clk_domain = SrcClockDomain(clock="3GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("1GiB")]
system.membus = SystemXBar()
system.system_port = system.membus.cpu_side_ports

# Multiple CPU contexts — the upstream dispatcher spawns a per-Queue
# worker thread; clone() in SE-mode needs a free HW context to land on.
system.cpu = [AtomicSimpleCPU(cpu_id=i) for i in range(NUM_CPUS)]
system.multi_thread = True
for cpu in system.cpu:
    cpu.createInterruptController()
    cpu.icache_port = system.membus.cpu_side_ports
    cpu.dcache_port = system.membus.cpu_side_ports
    # X86 needs explicit interrupt port wiring; ARM does not.
    cpu.interrupts[0].pio           = system.membus.mem_side_ports
    cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
    cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# DRAM serves the process's address space below PIO_BASE.
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = AddrRange(0, PIO_BASE)
system.mem_ctrl.port = system.membus.mem_side_ports

# The Vortex device — claims both the CP regfile PIO range and the
# BAR-mapped VRAM range.
system.vortex = VortexGPGPU(
    library = "/path/to/build/sim/simx/libvortex-gem5.so",
    kernel  = "",   # NO preload — the host binary uploads via CP
)
system.vortex.pio_addr = PIO_BASE
system.vortex.pio_size = PIO_SIZE
system.vortex.pin_addr = PIN_BASE
system.vortex.pin_size = PIN_SIZE
system.vortex.pio = system.membus.mem_side_ports
system.vortex.dma = system.membus.cpu_side_ports

# Workload — the host binary loads libvortex.so + libvortex-gem5-x86_64.so.
process = Process(
    pid=100,
    cwd="/path/to/your/test",
    cmd=["/path/to/your/test/binary"],
    executable="/path/to/your/test/binary",
    env=[
        "VORTEX_DRIVER=gem5-x86_64",
        "LD_LIBRARY_PATH=/path/to/build/sw/runtime",
    ],
)

system.workload = SEWorkload.init_compatible(process.executable)
for cpu in system.cpu:
    cpu.workload = process       # required: workload size must equal numThreads
    cpu.createThreads()

import m5
root = Root(full_system=False, system=system)
m5.instantiate()

# CRITICAL: Process.map() must come AFTER m5.instantiate().
# Identity-mapping PIO + PIN gives the runtime direct CPU access to
# the device's CP regfile and to BAR-mapped VRAM.
system.cpu[0].workload[0].map(PIO_BASE, PIO_BASE, PIO_SIZE, cacheable=False)
system.cpu[0].workload[0].map(PIN_BASE, PIN_BASE, PIN_SIZE, cacheable=False)

m5.simulate()
```

Reference implementations:
- [ci/gem5_run_hostless_app.py](../../ci/gem5_run_hostless_app.py) — hostless variant (preload via `kernel=` param; no host CPU)
- [ci/gem5_run_app.py](../../ci/gem5_run_app.py) — e2e variant (any regression test via `VORTEX_TEST_BIN`)

## Load-bearing invariants — do not violate

### 1. Process.map() goes AFTER m5.instantiate()

`Process.map(vaddr, paddr, size)` is a C++ method on the underlying
`gem5::Process` object; that object only exists after
`m5.instantiate()` builds the SimObject tree. Calling `.map()`
before instantiate raises `RuntimeError: Attempt to instantiate
orphan node <orphan Process>`. Confirmed by gem5's own AMD GPU
integration at `$GEM5_HOME/configs/example/apu_se.py:1055`.

### 2. PIO and PIN regions must be identity-mapped — and PIN must be cacheable=False

`sw/runtime/gem5/driver.h` hard-codes:
- `PIO_BASE_ADDR = 0x20000000` (CP regfile; 0x200 bytes)
- `PIN_BASE_ADDR = 0x100000000` (BAR-mapped VRAM; 4 GB)

The Python config must `process.map()` both at the same physical
addresses, with `cacheable=False` on PIN. With caching enabled the
host CPU's L1 could hold the new ring entry while `Q_TAIL_HI` is
observed by the CP — the CP fetches a stale CL and the dispatcher
hangs polling `Q_SEQNUM`.

Changing either constant requires updating both the Python config
**and** `sw/runtime/gem5/driver.h` (they are not auto-synced).

### 3. CPU thread context count must be >= 2

The upstream dispatcher (commit `157e7a1`) spawns a per-Queue worker
thread at `vx_queue_create`. SE-mode `clone()` returns EAGAIN if
there is no free HW context, which surfaces as
`std::system_error: Resource temporarily unavailable` at the
dispatcher constructor.

Use multiple CPU instances (one per thread) and
`system.multi_thread = True`. Assigning the same Process to every
CPU is required because gem5 fatals if
`workload.size() != numThreads`.

### 4. PIO accesses to the CP regfile are 32-bit

The CP regfile is 32-bit-wide; `cp_mmio_write/read` in the host
runtime are explicitly 32-bit (`mmio_write32` / `mmio_read32` in
`driver.cpp`). Don't issue 64-bit accesses — gem5 will deliver a
single packet of the wrong width and the SimObject will route the
extra bytes into the next regfile slot.

### 5. The Vortex `Processor` and `CommandProcessor` are independent gem5 event chains

`cpTickEvent_` advances the CP one functional cycle; `vortexTickEvent_`
advances the Vortex `Processor::cycle()`. Both self-schedule only
while their respective busy flag is true. When the CP fires
`CMD_LAUNCH`, the `vortex_start` hook schedules `vortexTickEvent_`
via the registered start handler (set at `VortexGPGPU` construction).
Don't try to combine them into a single tick — that breaks
"concurrent host + CP + GPU progress" which is the whole point of
the simulation model.

### 6. USE_SST=1 and USE_GEM5=1 are mutually exclusive

The Makefile rejects both at once. Different external simulators,
different LDFLAGS, different `libvortex.so` shapes. Pick one per
build.

## CI

`./ci/regression.sh --gem5` (built into `--all` is intentionally
**out**: gem5 install is heavy and gated like SST). The
`.github/workflows/ci.yml` matrix includes a `gem5` entry that runs
on hosted runners; ARM matrix gated on `VORTEX_GEM5_ARM=1`.

Apptainer integration (the `apptainer-ci.yml` pipeline) does **not**
include gem5 — adding it to `miscs/apptainer/vortex.def` is out of
scope. Use the hosted CI for gem5.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `dlopen('libvortex-gem5.so') failed: cannot open shared object file` | gem5 SimObject can't find the device library | Set `VortexGPGPU(library="/abs/path/to/libvortex-gem5.so", ...)` to absolute path |
| `Cannot open library: libvortex-gem5-x86_64.so: cannot open shared object file` | Stub can't find the host runtime backend | Set `LD_LIBRARY_PATH=/path/to/sw/runtime` in the `env=[...]` list passed to `Process()` |
| `terminate called after throwing an instance of 'std::system_error': Resource temporarily unavailable` | Dispatcher's per-Queue worker `std::thread` can't `clone()` into a free HW context | Use multiple CPU instances + `system.multi_thread = True`; assign the same Process to every CPU (invariant §3) |
| `system.membus has two ports responding within range [...]` | DRAM `mem_ctrl.dram.range` overlaps with VortexGPGPU's PIO or PIN range | Shrink `dram.range = AddrRange(0, PIO_BASE)` so the device-owned ranges have exclusive routing |
| `Tried to write unmapped address 0xXXX` | Host runtime is using stale PIN_BASE_ADDR (mismatch with Python config), or `Process.map()` was skipped | Confirm both `sw/runtime/gem5/driver.h` and the Python config use the same `PIN_BASE_ADDR`; ensure `Process.map(PIN_BASE, PIN_BASE, PIN_SIZE)` runs after `m5.instantiate()` |
| `Attempt to instantiate orphan node <orphan Process>` | `Process.map()` called before `m5.instantiate()` | Move all `.map()` calls AFTER `m5.instantiate()` — see invariant §1 above |
| `fatal: VortexGPGPU: dlsym(vortex_gem5_cp_mmio_write) failed` | Device library is missing the C ABI symbol — usually means the `library=` parameter points at the wrong .so | `library=` is the **device** library `build/sim/simx/libvortex-gem5.so` (no arch suffix), NOT the host runtime `libvortex-gem5-x86_64.so` |
| `fatal: system.membus has two ports responding within range [0x10000000:0x20000000]` (standalone hello) | `pin_size` defaulted to non-zero in an old gem5.opt; standalone test doesn't need the BAR | Re-install + rebuild gem5.opt OR explicitly set `pin_size = 0` on the VortexGPGPU instance |
| Test hangs polling `Q_SEQNUM` after first launch | Cacheable PIN region — host's L1 holds the ring entry; CP sees stale bytes | Set `cacheable=False` on the PIN `Process.map()` call (invariant §2) |
| `ccache g++ ... undefined reference to fmt::v8::detail::error_handler::on_error` | ccache served a stale object compiled against a different `fmt` version | `CCACHE_DISABLE=1 make -C sim/simx clean && CCACHE_DISABLE=1 make ...` |

---

## 6. Proposed but not yet implemented

1. **v2 DMA-port memory seam.** The `DevMemAccessor` interface exists, but
   `DmaPortDevMem` backing VRAM through gem5's `SimpleMemory` over the
   SimObject DMA port is not built — VRAM is in-process `simx::RAM` despite
   the `DmaDevice` base class. The seam is the design's whole point;
   preserve the intent.
2. **Multi-queue host runtime.** The PIO map reserves 4 queues
   (`max_queues=4`) but the CP model is single-queue (`q0_`) and the host
   exercises Q0 only — the growth path for vortex2.h multi-queue.
3. **PCIe `PciDevice` / BAR upgrade.** The C ABI is shape-compatible; only
   the gem5 wrapper class changes. Doorbell-ring realism (vs. today's
   `Q_SEQNUM` polling) is the matching upgrade — let the CP raise an
   interrupt and let the dispatcher sleep until it fires, instead of
   spinning on `Q_SEQNUM` PIO reads.
4. **FS-mode Linux + a kernel driver** (out of scope) — SE-mode only today.
5. **Multi-device** (one `VortexGPGPU` per system today) and a **separate
   ClockDomain** for CP vs. Vortex (single-domain today; real silicon has
   separate clocks, so v2 would add a second `ClockDomain` and rate-match
   the tick events).
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
protocol (torn out — grep-clean of OPAE leftovers); and the
`cp_mmio_{read,write}` + `mem_upload/download/copy` HAL signature (replaced
by the 5-callback `cp_pure_v2` transport HAL the whole runtime tree now
shares). The claim that `vortex_dcr_read` is not a hook was also discarded —
the shipped `CommandProcessor::Hooks` keeps it as a real 6th hook.

---

## 7. Source proposals

This design consolidates and supersedes the following proposals (now
removed from `docs/proposals/`): `gem5_simx_v3_proposal.md`,
`gem5_v2_cp_migration_proposal.md`. The CP architecture is in
[`command_processor_control_plane.md`](command_processor_control_plane.md).

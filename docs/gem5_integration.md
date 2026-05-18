# gem5 Integration

Vortex can run inside the [gem5](https://www.gem5.org/) full-system
simulator as a `DmaDevice` SimObject, exposing a Vortex GPGPU to a
simulated host CPU (x86 or ARM) over the standard OPAE MMIO+DMA
command protocol. Use this when you want to model heterogeneous
host-CPU+accelerator workloads with realistic cross-ISA cache and
DMA timing.

For the design rationale see
[docs/proposals/gem5_simx_v3_proposal.md](proposals/gem5_simx_v3_proposal.md).
This document is the operator manual.

## At a glance

The integration has three moving parts that live in this repo:

| Part | Source | Built artifact | Loaded by |
|---|---|---|---|
| Device library | `sim/simx/gem5/vortex_gpgpu.{cpp,h}` | `build/sim/simx/libvortex-gem5.so` | gem5 SimObject via `dlopen` |
| gem5 SimObject | `sim/simx/gem5/vortex_gpgpu_dev.{cc,hh}` + `VortexGPGPU.py` + `SConscript` | Linked into `gem5.opt` after install | gem5 itself |
| Host runtime | `sw/runtime/gem5/{vortex.cpp,driver.{cpp,h},Makefile}` | `build/sw/runtime/libvortex-gem5-{x86_64,aarch64}.so` | The simulated process inside gem5 |

Plus one external piece: `ci/gem5_install.sh` fetches gem5
v25.0.0.1, drops our SimObject sources into `$GEM5_HOME/src/dev/vortex/`,
and builds `build/{X86,ARM}/gem5.opt` (both ISAs by default).

## One-time setup

Vortex install / build as usual ([docs/install_vortex.md](install_vortex.md)),
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
errors out if both are set) — they're different external simulators
with different LDFLAGS; building both into one binary makes no sense.

### Host runtime + tests (cross-compile)

The simulated process inside gem5 loads the **host runtime**
`libvortex-gem5-$HOST_ARCH.so`, which speaks the OPAE MMIO/DMA
protocol to the device. The `HOST_ARCH` knob is consistent across
three Makefiles — runtime backend, stub, and regression tests:

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

# armhf works the same way:
make -C sw/runtime/stub HOST_ARCH=armhf
make -C sw/runtime/gem5 HOST_ARCH=armhf
make -C tests/regression/vecadd HOST_ARCH=armhf
```

The ARM targets require `gcc-aarch64-linux-gnu` / `gcc-arm-linux-gnueabihf`
respectively — `ci/gem5_install.sh` installs these.

## Running tests

### From the regression harness

```bash
cd build/
./ci/regression.sh --gem5
```

Runs both the standalone Phase-3 smoke test (kernel preloaded on the
SimObject, no host CPU) and the Phase-5 end-to-end test (real
SE-mode host program drives the device through MMIO+DMA). Total
wall time ~5 s on a fast box.

To also run the ARM matrix entry (needs `gcc-aarch64-linux-gnu`):

```bash
VORTEX_GEM5_ARM=1 ./ci/regression.sh --gem5
```

Runs 6 tests in ~16 s wall:
- X86 standalone hello (no host CPU; SimObject preloads kernel)
- X86 e2e vecadd `-n16` (host CPU drives device via OPAE MMIO+DMA)
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
   `/usr/aarch64-linux-gnu/lib/*` so the dynamic linker can
   resolve libc, libstdc++, etc.

Both paths point at the Ubuntu `gcc-aarch64-linux-gnu` package's
install location — no extra setup needed.

### By hand

**Standalone** (no host CPU; kernel preloaded via SimObject parameter):

```bash
VORTEX_GEM5_LIB=$(pwd)/sim/simx/libvortex-gem5.so \
VORTEX_GEM5_KERNEL=$(pwd)/tests/kernel/hello/hello.vxbin \
    $GEM5_HOME/build/X86/gem5.opt ci/gem5_test_vortex_hello.py
```

**End-to-end** — any standard Vortex regression test (host binary
+ kernel.vxbin) runs through the generic
[`ci/gem5_test_vortex_app.py`](../ci/gem5_test_vortex_app.py)
runner. Set `VORTEX_TEST_BIN` to the test name:

```bash
# vecadd
VORTEX_GEM5_DEV_LIB=$(pwd)/sim/simx/libvortex-gem5.so \
VORTEX_GEM5_HOST_RT_DIR=$(pwd)/sw/runtime \
VORTEX_TEST_DIR=$(pwd)/tests/regression/vecadd \
VORTEX_TEST_BIN=vecadd \
VORTEX_TEST_ARGS="-n16" \
    $GEM5_HOME/build/X86/gem5.opt ci/gem5_test_vortex_app.py

# sgemm
VORTEX_GEM5_DEV_LIB=$(pwd)/sim/simx/libvortex-gem5.so \
VORTEX_GEM5_HOST_RT_DIR=$(pwd)/sw/runtime \
VORTEX_TEST_DIR=$(pwd)/tests/regression/sgemm \
VORTEX_TEST_BIN=sgemm \
VORTEX_TEST_ARGS="-n4" \
    $GEM5_HOME/build/X86/gem5.opt ci/gem5_test_vortex_app.py
```

Expected vecadd output (truncated):
```
allocate device memory
upload source buffer0
upload source buffer1
Upload kernel binary
start device
wait for completion
download destination buffer
verify result
PASSED!
```

### Sizing tests for the 120 s budget

Each `timeout 120` per test bound comes from
[feedback_test_timeout_120s](../../../../.claude/projects/-home-blaisetine-dev/memory/feedback_test_timeout_120s.md).
gem5 SE-mode runs the host CPU's `ready_wait` poll loop in
simulated time too, so **kernel runtime translates directly into
gem5 wall time**. The regression script's default sizes fit:

| Test | Args | Device cycles | Wall (atomic CPU) |
|---|---|---|---|
| vecadd | `-n16` | ~450 | ~3 s |
| sgemm  | `-n4`  | ~780 | ~3 s |
| sgemm  | `-n16` | ~10k+ | **> 120 s** (overruns) |

Larger sizes are fine when run by hand outside the budget cap.

## Writing your own gem5 Python script

The minimal recipe for hosting Vortex inside a custom gem5 system:

```python
from m5.objects import (
    AddrRange, AtomicSimpleCPU, DDR3_1600_8x8, MemCtrl, Process,
    Root, SEWorkload, SrcClockDomain, System, SystemXBar,
    VoltageDomain, VortexGPGPU,
)

# Mappings expected by sw/runtime/gem5/driver.h.
PIO_BASE, PIO_SIZE = 0x20000000, 0x1000
PIN_BASE, PIN_SIZE = 0x10000000, 0x10000000   # 256 MB pinned region

system = System()
system.clk_domain = SrcClockDomain(clock="3GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("1GiB")]
system.membus = SystemXBar()
system.system_port = system.membus.cpu_side_ports

# CPU (x86 example). For ARM, swap to ArmAtomicSimpleCPU + adjust
# interrupt wiring.
system.cpu = AtomicSimpleCPU()
system.cpu.createInterruptController()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.interrupts[0].pio           = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# DRAM serves [0, 512MB). PIO at 0x20000000 above goes to the
# Vortex device (membus routes by address).
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = AddrRange(0, size="512MiB")
system.mem_ctrl.port = system.membus.mem_side_ports

# The Vortex device.
system.vortex = VortexGPGPU(
    library = "/path/to/build/sim/simx/libvortex-gem5.so",
)
system.vortex.pio_addr = PIO_BASE
system.vortex.pio_size = PIO_SIZE
system.vortex.pio = system.membus.mem_side_ports
system.vortex.dma = system.membus.cpu_side_ports

# Workload — the host binary uses the OPAE protocol via libvortex.so
# + libvortex-gem5-x86_64.so (selected by VORTEX_DRIVER).
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

system.workload = SEWorkload.init_compatible("/path/to/your/test/binary")
system.cpu.workload = process
system.cpu.createThreads()

import m5
root = Root(full_system=False, system=system)
m5.instantiate()

# CRITICAL: Process.map() must come AFTER m5.instantiate().
# Identity-mapping PIO + PIN makes the runtime's volatile-pointer
# MMIO and DMA staging buffer "just work" from the simulated process.
system.cpu.workload[0].map(PIO_BASE, PIO_BASE, PIO_SIZE, cacheable=False)
system.cpu.workload[0].map(PIN_BASE, PIN_BASE, PIN_SIZE, cacheable=True)

m5.simulate()
```

Reference implementations:
- [ci/gem5_test_vortex_hello.py](../ci/gem5_test_vortex_hello.py) — standalone Phase-3 variant (preload via `kernel=` param; no host CPU)
- [ci/gem5_test_vortex_app.py](../ci/gem5_test_vortex_app.py) — Phase-5 e2e variant (any regression test via `VORTEX_TEST_BIN`)

## Load-bearing invariants — do not violate

These are the rules that, if broken, will silently produce wrong
answers or hangs. Each is repeated from the proposal but is
load-bearing enough to call out here:

### 1. Process.map() goes AFTER m5.instantiate()

`Process.map(vaddr, paddr, size)` is a C++ method on the underlying
`gem5::Process` object; that object only exists after
`m5.instantiate()` builds the SimObject tree. Calling `.map()`
before instantiate raises `RuntimeError: Attempt to instantiate
orphan node <orphan Process>`.

Confirmed by gem5's own AMD GPU integration at
`$GEM5_HOME/configs/example/apu_se.py:1055`.

### 2. PIO and PIN regions must be identity-mapped

`sw/runtime/gem5/driver.h` hard-codes:
- `PIO_BASE_ADDR = 0x20000000` (device MMIO; 4 KB)
- `PIN_BASE_ADDR = 0x10000000` (DMA staging; 256 MB)

The Python config must `process.map()` both at the same physical
addresses so:
- CPU's `*(volatile uint64_t*)0x20000000` → membus routes to the device
- Device's DmaPort read at phys `0x10000000+N` → membus routes to DRAM
- Both sides agree on the same bytes without any virtual-to-physical
  translation surprise.

Changing either constant requires updating both the Python config
**and** `sw/runtime/gem5/driver.h` (they are not auto-synced).

### 3. The CPU runtime MUST issue a cache flush before reading back results

The host runtime's `download()` path issues a per-core
`dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy)` BEFORE the
`CMD_MEM_READ` DMA. Skipping it returns stale data — the L1/L2/L3
caches may still hold writes that haven't reached VRAM.

This is bug **B9** in the legacy `vortex_gem5` code; the v3 host
runtime fixes it. If you write your own runtime, do the same.

### 4. MMIO writes need an explicit memory barrier before CMD_TYPE

The host CPU model in gem5 (especially out-of-order variants) can
reorder MMIO writes. `sw/runtime/gem5/driver.cpp` centralises the
fence in `issue_cmd()` so it's impossible to forget:
- x86: `__asm__ volatile("mfence" ::: "memory")`
- AArch64/ARMv7: `__asm__ volatile("dmb sy" ::: "memory")`

If your custom runtime bypasses `issue_cmd()`, replicate this. This
is bug **B14** in the legacy code.

### 5. One source of truth for memory state

Vortex's VRAM is owned by `vortex::RAM` inside the device library.
The pinned region is owned by gem5's DRAM. **The device library
does not maintain a shadow copy of host pinned memory; the host
runtime does not maintain a shadow copy of device VRAM.** Bytes
cross between the two only via the explicit DMA staging path
(steps 1-6 in §5 of `gem5_simx_v3_proposal.md`).

Don't add a "fast path" that reads/writes the other side's memory
directly. That breaks the timing model and reintroduces bug **B3**
from the legacy code.

### 6. USE_SST=1 and USE_GEM5=1 are mutually exclusive

The Makefile rejects both at once. Different external simulators,
different LDFLAGS, different `libvortex.so` shapes. Pick one per
build.

## Architectural choices you may want to revisit

These are documented in [the proposal](proposals/gem5_simx_v3_proposal.md)
but worth surfacing:

- **Status polling, not doorbell queues** (proposal §3.6 "Doorbell
  queues" note). The host runtime polls `MMIO_STATUS` between
  commands; modern GPUs (AMD, NVIDIA) use ring-buffer + doorbell.
  Phase 7+ upgrade if your research needs batched-dispatch realism.
- **SE-mode + custom PIO+DMA wiring**, not FS-mode + PCIe (proposal
  §3.6). Matches the legacy capstone paper; faster iteration. PCIe
  upgrade is a Phase 5+ enhancement that swaps the SimObject base
  class from `DmaDevice` to `PciDevice` (both inherit `DmaDevice`
  so the C ABI stays compatible).
- **C ABI between the device library and gem5 SimObject** instead
  of C++ linkage (proposal §3.1). Lets you rebuild
  `libvortex-gem5.so` without rebuilding `gem5.opt` — Vortex
  internals can churn freely.

## CI

`./ci/regression.sh --gem5` (built into `--all` is intentionally
**out**: gem5 install is heavy and gated like SST). The
`.github/workflows/ci.yml` matrix includes a `gem5` entry that runs
on hosted runners; ARM matrix gated on
`VORTEX_GEM5_ARM=1`.

Apptainer integration (the `apptainer-ci.yml` pipeline) does **not**
include gem5 — adding it to `miscs/apptainer/vortex.def` is out of
scope for this integration (proposal §8). Use the hosted CI for
gem5.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `dlopen('libvortex-gem5.so') failed: cannot open shared object file` | gem5 SimObject can't find the device library | Set `VortexGPGPU(library="/abs/path/to/libvortex-gem5.so", ...)` to absolute path |
| `Cannot open library: libvortex-gem5-x86_64.so: cannot open shared object file` | Stub can't find the host runtime backend | Set `LD_LIBRARY_PATH=/path/to/sw/runtime` in the `env=[...]` list passed to `Process()` |
| `fatal: syscall clock_nanosleep (#230) unimplemented` | gem5 SE-mode doesn't implement clock_nanosleep; glibc's `nanosleep()` routes through it | Already fixed in `sw/runtime/gem5/vortex.cpp` (uses `sched_yield()` instead). If you wrote your own runtime, do the same. |
| `Attempt to instantiate orphan node <orphan Process>` | `Process.map()` called before `m5.instantiate()` | Move all `.map()` calls AFTER `m5.instantiate()` — see invariant §1 above |
| `fatal: VortexGPGPU: dlsym(vortex_gem5_build_info) failed` | Device library is missing the C ABI symbol — usually means the `library=` parameter points at the wrong .so | `library=` is the **device** library `build/sim/simx/libvortex-gem5.so` (no arch suffix), NOT the host runtime `libvortex-gem5-x86_64.so` |
| Test hangs forever in `vx_ready_wait` | Device's busy bit never clears, usually because the SimObject didn't schedule the tick event | Confirm you set `system.vortex.dma = system.membus.cpu_side_ports` and the device's `tick()` is reachable. Check gem5 with `--debug-flags=VortexGPGPU` |
| `ccache g++ ... undefined reference to fmt::v8::detail::error_handler::on_error` | ccache served a stale object compiled against a different `fmt` version | `CCACHE_DISABLE=1 make -C sim/simx clean && CCACHE_DISABLE=1 make ...` |

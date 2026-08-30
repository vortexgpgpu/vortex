# SLASH — Software and Hardware Architecture Knowledge Base

**Scope.** The complete SLASH platform for the AMD Alveo V80: what each layer
is, the interfaces between them, the kernel ABI, the daemon's wire protocol, the
VRT API, the vrtbin container format, the linker and its connectivity language,
the memory model, and the three execution platforms. Includes the empirical
findings from running Vortex on this stack — marked **[field]** — which are not
in the upstream documentation.

Source of truth: `~/dev/SLASH-compute` (`docs/`, `driver/`, `vrt/`, `linker/`,
`smi/`).

---

## 1. What SLASH is, and what it replaces

SLASH is AMD's open-source platform for the Alveo V80. It is a *replacement* for
XRT on this board, not a port of it — the API surface is smaller, more
opinionated, and targets exactly one card.

| Concern | XRT (U-series) | SLASH (V80) |
|---|---|---|
| Runtime library | `libxrt_core` | `libvrt` (C++17) |
| Application object | `xrt::device` / `kernel` / `bo` / `run` | `vrt::Device` / `Kernel` / `Buffer<T>` |
| Deployment artefact | `.xclbin` | `.vbin` (*vrtbin*) |
| CLI | `xbutil` | `v80-smi` |
| Emulation selection | `XCL_EMULATION_MODE` env var | encoded in the vbin |
| Kernel drivers | `xocl` + `xclmgmt` (both privileged, direct ioctl) | `slash.ko` (PF1+PF2) + `ami` (PF0) |
| Privilege mediation | none — library issues ioctls directly | **`vrtd` daemon** between library and driver |
| Memory bank selection | `xrt::kernel::group_id` | `vrt::Kernel::argMemoryConfig` |
| Clock control | platform-fixed | `vrt::Device::setFrequency` |

The daemon is the structural difference. XRT's library talks to the driver
directly; SLASH interposes `vrtd`, which owns the privileged operations
(programming, reset, hotplug) and enforces per-role permissions for
multi-tenancy. Every unprivileged application therefore gets its device access
through a socket, not through `/dev`.

---

## 2. The layer stack

```
┌─────────────────────────────────────────────┐
│              User Application               │  C++17
├─────────────────────────────────────────────┤
│            VRT  (libvrt)                    │  C++17  ─ MIT
├─────────────────────────────────────────────┤
│          libvrtd++  (C++ RAII wrapper)      │  C++20  ─ MIT
├─────────────────────────────────────────────┤
│          libvrtd    (C wire-protocol)       │  C11    ─ MIT
├──────────────── AF_UNIX ────────────────────┤
│          vrtd       (daemon)                │  C11    ─ MIT
├─────────────────────────────────────────────┤
│          libslash   (driver wrapper)        │  C      ─ MIT
├─────────────────────────────────────────────┤
│       Linux kernel module  (slash)          │  C      ─ GPLv2
├─────────────────────────────────────────────┤
│          AMD Alveo V80 Hardware             │
└─────────────────────────────────────────────┘
```

Two components sit *alongside* rather than inside the stack:

- **`v80-smi`** — CLI for listing, inspecting, programming, resetting,
  querying, and validating boards.
- **`slashkit`** — Python linker that turns compiled HLS kernel IP plus a
  connectivity config into a vrtbin.

The licence boundary is deliberate: only the kernel module is GPLv2; everything
above it is MIT, so downstream userspace can be relicensed or vendored.

### Repository layout

| Directory | Component |
|---|---|
| `vrt/` | VRT C++17 runtime (and the bundled `vrtd`) |
| `driver/` | `slash.ko` kernel module + `libslash` C wrapper |
| `smi/` | `v80-smi` |
| `linker/` | `slashkit` Python linker |
| `cmake/` | `add_vbin()`, `buildhls`, `findtools` modules |
| `examples/` | 8 worked example designs |
| `packaging/` | Debian and RPM packaging, DKMS |

---

## 3. PCIe topology

Each V80 exposes **three** physical functions under SLASH's compute shell.

```
┌─────────────── V80 Board ───────────────┐
│   PF0 (.0)      PF1 (.1)      PF2 (.2)  │
│   ami           slash_qdma    slash_ctl │
│   0x50B4        0x50C1        0x50C2    │
│   Management    DMA           BAR MMIO  │
└─────────────────────────────────────────┘
```

| PF | Device ID | Driver | Device node | Role |
|---|---|---|---|---|
| PF0 | `0x50B4` | `ami` | (AMI subsystem) | AVED management — sensors, identity, firmware version, PDI flashing |
| PF1 | `0x50C1` | `slash_qdma` | `/dev/slash_qdma_ctl<N>` | QDMA — H2C/C2H for buffers and streaming |
| PF2 | `0x50C2` | `slash_ctl` | `/dev/slash_ctl<N>` | BAR MMIO — kernel register read/write |

All share vendor ID `0x10EE`. For backward compatibility the driver also binds
the legacy IDs `0x50B5` (PF1) and `0x50B6` (PF2), plus the AVED/V80P QDMA ID
`0x50BD`, so cards carrying an older bitstream still work.

Module initialisation order is **QDMA → Hotplug → PCIe**; teardown is reversed.

### Discovery

`v80-smi list` scans sysfs:

1. Enumerate `/sys/bus/pci/devices/` for vendor `0x10EE`, device `0x50B4`.
2. Extract the board BDF.
3. Verify PF1 (`0x50C1`) and PF2 (`0x50C2`) exist at the same bus:device.
4. Check the correct driver is bound to each, via the `driver` symlink.
5. Ask `vrtd` whether the board is registered
   (`vrtd::Session::getDeviceByBdf()`).

All four checks — PF0, PF1, PF2, VRTD — must pass before VRT can use the board.

### The node-number caveat

Device nodes are `miscdevice`s (major 10, dynamic minors). Two documented
hazards that userspace must handle:

- Node numbers are **not stable across remove+rescan**, so a path does not
  identify a board. Always verify with `GET_DEVICE_INFO`.
- `slash_ctl` and `slash_qdma_ctl` numbers come from **separate counters**, so
  `/dev/slash_ctl0` and `/dev/slash_qdma_ctl1` may be the same physical card.

---

## 4. The kernel module

`slash.ko` is ~6900 lines across a dozen files. The shape:

| File | Lines | Responsibility |
|---|---|---|
| `slash_qdma.c` | 3751 | PF1 driver, queue-pair lifecycle, transfers, io_uring cmd, host-profile programming |
| `slash_ctldev.c` | 745 | PF2 control character device — BAR info, BAR fd, device info |
| `slash_hotplug.c` | 495 | Global `/dev/slash_hotplug` — REMOVE / TOGGLE_SBR / RESCAN / HOTPLUG |
| `slash_dmabuf.c` | 309 | dma-buf exporter for device BAR regions |
| `slash_hostbuf.c` | 171 | DMA-coherent **host** memory addressable by a kernel's AXI master |
| `slash_pcie.c` | 164 | PF2 PCI driver — probe/remove |
| `slash_main.c` | 134 | module init/exit, parameter plumbing |
| `slash_config.h` | 108 | PCI IDs, PF assignments, node naming, `pr_fmt` |

Plus `kcompat/` — compatibility shims for `io_uring_cmd`, `vm_flags_set`, modern
timer API, and namespace import tokens, so one source tree builds across a wide
kernel range.

### 4.1 Module parameters

Exposed under `/sys/module/slash/parameters/`, all runtime-writable:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `qdma_num_threads` | uint | 8 | libqdma worker threads |
| `qdma_debugfs_path` | charp | disabled | debugfs mount path for libqdma |

### 4.2 `slash_hostbuf.c` — the host-mastering path

This file is small but architecturally important, because it is the mechanism by
which an FPGA kernel reads host DRAM:

> *"Kernels reach host DRAM through the QDMA slave bridge, which addresses it by
> bus address rather than CPU virtual address. Userspace needs both views of the
> same bytes: a mapping to write through, and the bus address to hand the
> device. The allocation is coherent, so a design can keep a command ring here
> and have the device pull from it without an explicit sync on either side."*

```c
struct slash_hostbuf {
    struct pci_dev *pdev;      /* referenced for the buffer's lifetime */
    void           *cpu_addr;  /* kernel VA from dma_alloc_coherent()   */
    dma_addr_t      dma_addr;  /* bus address the DEVICE masters to     */
    size_t          len;       /* page-rounded                          */
};
```

It refuses `dma_buf` device attachments deliberately — importing into a second
device would need a second mapping, and the slave-bridge address handed out here
would not describe it.

This is distinct from `slash_dmabuf.c`, which exports *device BAR* regions. One
exports card memory to the host; the other exports host memory to the card.

> **[field] The driver support exists and works; the fabric path does not, on
> this shell.** Routing an HLS `m_axi` port to the `HOST` target produced an AXI
> master whose reads never completed. See §9.4.

### 4.3 `slash_hotplug.c` — reset primitives

Four ioctls on the single global `/dev/slash_hotplug`:

| Op | Effect |
|---|---|
| `REMOVE` | Remove a device by BDF from the PCI bus |
| `TOGGLE_SBR` | Assert Secondary Bus Reset on the upstream root port, deassert, wait for link retrain |
| `RESCAN` | Rescan the whole PCI bus |
| `HOTPLUG` | Atomic REMOVE + RESCAN for one device |

The canonical reprogramming sequence:

```
1. REMOVE  PF0, PF1, PF2       ← tear down all three functions
2. TOGGLE_SBR on root port     ← reset the FPGA, reload the bitstream
3. RESCAN                      ← re-enumerate
4. HOTPLUG each function       ← rebind drivers
```

Two implementation details worth copying into any driver that does this:

- `REMOVE` / `RESCAN` / `HOTPLUG` hold `pci_lock_rescan_remove()` for their
  whole duration. `TOGGLE_SBR` holds it only across `pci_find_bus()` +
  `pci_dev_get()` and **drops it before calling
  `pci_bridge_secondary_bus_reset()`**, to avoid deadlocking against the PCI
  slot lock.
- The post-SBR settle is **1000 ms**, not the PCIe spec minimum of 100 ms —
  real FPGA endpoints need longer, and the margin also covers the
  kernel-internal window between reset completion and the link being usable.

> **[field] `TOGGLE_SBR` twice hard-reset this host.** The journal ends at
> `slash_hotplug: ioctl: TOGGLE_SBR succeeded` with no shutdown record and no
> MCE. The same operation succeeded three times in an earlier boot, so it is a
> race. Root cause is a platform firmware policy escalating a fatal PCIe error
> to a system reset; see KB-2 §9.3 for the BIOS mitigation.

---

## 5. Kernel ABI

Full formal spec: `docs/reference/kernel-abi/index.rst`. The essentials.

### 5.1 Versioning convention

Every ioctl struct leads with `__u32 size`, set by the caller to
`sizeof(struct)`. The kernel reads `size` first, copies in
`min(user_size, kernel_size)` bytes, zero-fills fields the caller's older struct
lacks, writes back `min(user_size, kernel_size)`, and `clear_user()`s the tail
if the caller's struct is newer. Driver and library version independently in
both directions. Unknown command numbers return `-ENOTTY`.

### 5.2 Control device — `/dev/slash_ctl<N>`

Magic byte `'v'` (0x76), sequences `0x30`–`0x32`. ioctl-only — no `read`,
`write`, or `mmap` on the fd itself.

| ioctl | Purpose |
|---|---|
| `SLASH_CTLDEV_IOCTL_GET_BAR_INFO` (`0x30`) | per-BAR: `usable`, `start_address`, `length` |
| `SLASH_CTLDEV_IOCTL_GET_BAR_FD` (`0x31`) | returns a **dma-buf fd as the ioctl return value**; fills `length` |
| `SLASH_CTLDEV_IOCTL_GET_DEVICE_INFO` (`0x32`) | BDF string + vendor/device/subsystem IDs |

MMIO is done by `mmap()`ing the dma-buf fd, with every access bracketed by
`DMA_BUF_IOCTL_SYNC`:

```c
struct dma_buf_sync s = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE };
ioctl(bar_fd, DMA_BUF_IOCTL_SYNC, &s);
/* ... writes through the mapped pointer ... */
s.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
ioctl(bar_fd, DMA_BUF_IOCTL_SYNC, &s);
```

Documented caveats: BAR mappings are **not inherited across `fork()`** — each
child needs its own fd. After a device is removed from the PCI hierarchy the
mapping stays valid in virtual memory but reads return `0xFFFFFFFF` and writes
are discarded; treat the mapping as invalid.

### 5.3 QDMA device — `/dev/slash_qdma_ctl<N>`

Queue-pair lifecycle:

```c
/* 1. Add a queue pair — MM mode, bidirectional */
struct slash_qdma_qpair_add add = {
    .size = sizeof(add), .mode = 0 /* MM */, .dir_mask = 0x3 /* H2C|C2H */,
};
ioctl(qdma_fd, SLASH_QDMA_IOCTL_QPAIR_ADD, &add);

/* 2. Start it */
struct slash_qdma_qpair_op op = { .size = sizeof(op), .qid = add.qid, .op = 0 };
ioctl(qdma_fd, SLASH_QDMA_IOCTL_Q_OP, &op);

/* 3. Obtain the per-qpair I/O fd (anon inode) */
int io_fd = ioctl(qdma_fd, SLASH_QDMA_IOCTL_QPAIR_GET_FD, &fd_req);

/* 4. Kernel allocates the pages, builds the SGL, DMA-maps once */
int buf_fd = ioctl(io_fd, SLASH_QDMA_IOCTL_BUF_CREATE, &bc);
void *host_buf = mmap(NULL, nbytes, PROT_READ|PROT_WRITE, MAP_SHARED, buf_fd, 0);

/* 5. Transfer: registered buffer + offset, device address, length, direction */
struct slash_qdma_transfer xfer = { .size = sizeof(xfer), .count = 1,
    .xfers[0] = { .qpair_index = 0, /* ... */ } };
```

The design decision worth noting: **the kernel owns the DMA buffer.** Rather
than pinning user pages per transfer, `BUF_CREATE` allocates, builds the
scatter-gather list, and DMA-maps once. Per-transfer cost then drops to writing
descriptors. `io_uring_cmd` support (`slash_qdma_qpair_uring_cmd`) layers async
submission on the same fd.

### 5.4 Concurrency model

Stated intent: all ioctls and `read`/`write` are safe to call concurrently from
multiple threads or processes, on the same fd or different fds; the kernel
serialises internally.

Stated honestly in the same document:

> *"The current kernel driver is not exhaustively tested for concurrent access
> and bugs in this area may exist. Treat the safety property as an intent rather
> than a verified guarantee."*

A queue pair is conceptually **sequential** — hardware processes one `read()` or
`write()` at a time on a given qpair, and the kernel serialises. Multi-threading
one qpair is safe but pointless; for parallel I/O, allocate multiple qpairs.

### 5.5 NoC channel selection

The AXI-MM/NoC channel is chosen **per queue pair** at add time
(`mm_channel`, `enum slash_qdma_mm_channel`): `auto` stripes queues across both
channels by `qid & 1`; `0`/`1` pin to a single channel. This exists so a design
can A/B-test whether both PCIe NMUs actually contribute bandwidth — see KB-2
§6.2. Debug builds with `SLASH_QDMA_OP_DEBUG=1` log each queue's channel.

---

## 6. libslash and vrtd

### 6.1 libslash

A thin C wrapper over the ioctl interface, three modules:

- `slash/ctldev.h` — BAR MMIO via PF2
- `slash/qdma.h` — queue-based DMA via PF1
- `slash/hotplug.h` — SBR and rescan

It also ships **mock implementations** (`ctldev_mock.c`, `qdma_mock.c`) so the
layers above can be tested with no hardware. That is the mechanism behind VRT's
mock mode (`docs/howto/use-mock-mode.rst`).

### 6.2 vrtd — the daemon

`vrtd` multiplexes device access and enforces permission rules for
multi-tenancy. It runs as a systemd service, listens on a Unix domain socket
(`VRTD_STANDARD_PATH`, typically `/run/vrtd.sock`), and translates client
requests into libslash calls. Configuration is `vrtd.conf`.

**Wire protocol**

- Transport: `AF_UNIX` + `SOCK_SEQPACKET` (message-preserving, unlike
  `SOCK_STREAM`).
- Messages: request/response headers (size, opcode, seqno) + body.
- **FD passing via `SCM_RIGHTS`** — this is how a BAR fd crosses the privilege
  boundary. The daemon holds the privileged `/dev/slash_ctl<N>` fd; the client
  receives only a dma-buf fd for the BAR it is allowed to touch.
- Body size capped at `VRTD_MSG_MAX_SIZE` minus headers.
- `vrtd_raw_request` is a generic escape hatch for arbitrary opcodes; typed
  helpers are preferred.

**Client libraries**

`libvrtd` (C11) exposes typed request/response helpers. `libvrtd++` (C++20)
wraps it in RAII: `vrtd::Session`, `vrtd::Device`, `vrtd::Bar`,
`vrtd::BarFile`, `vrtd::BarFilePtr`.

```cpp
vrtd::Session s;                       // connects, RAII-closed
vrtd::Device  d  = s.getDevice(0);
vrtd::Bar     b  = d.getBar(0);
vrtd::BarFile bf = b.openBarFile();    // owns fd + mapping
auto p = bf.getPtr<uint32_t>(vrtd::BarFile::Direction::Read, /*address=*/0);
uint32_t value = *p;                   // brackets START/END automatically
```

**Lifetime rules that bite:** moving or closing a `Session` invalidates every
`Device` and `Bar` obtained from it. `BarFile` is move-only, is *not*
thread-safe, permits only one active read or write at a time, and throws on a
re-entrant `getPtr()`. All `BarFilePtr`s must be destroyed before `close()`.

**Error model:** C returns `enum vrtd_ret`; C++ throws `vrtd::Error`. Codes:
`OK`, `BAD_LIB_CALL`, `BAD_CONN`, `BAD_REQUEST`, `INVALID_ARGUMENT`, `NOEXIST`,
`INTERNAL_ERROR`, `AUTH_ERROR` (permission denied by role config).

### 6.3 The flash worker

vrtd owns PDI programming. The critical behaviour, in
`vrt/vrtd/src/flash_worker.c` with the reset implementation in
`vrt/vrtd/src/reset.c`:

> `reset_with_ami` runs **only when the requested shell differs from the current
> shell.**

This is the single most consequential line in the daemon. See §9.1.

`reset.c:334` also documents the hazard explicitly:

> *"If any function remains bound while the bus is reset, the kernel may attempt
> MMIO or config-space accesses to a device whose link is down, which can cause
> machine checks or system hangs."*

---

## 7. VRT — the application API

| Class | Header | Purpose |
|---|---|---|
| `vrt::Device` | `vrt/device.hpp` | Open a board by BDF, load a vrtbin, expose kernels and memory config |
| `vrt::Kernel` | `vrt/kernel.hpp` | Argument setting, start, wait, register read/write |
| `vrt::Buffer<T>` | `vrt/buffer.hpp` | Typed device memory with `sync()` |
| `vrt::StreamingBuffer<T>` | `vrt/streaming_buffer.hpp` | QDMA streaming I/O for kernel ports |
| `vrt::Vrtbin` | `vrt/vrtbin.hpp` | Archive extraction and metadata lookup |

Dependencies: libxml2 (`system_map.xml`), ZeroMQ (emu/sim IPC), JsonCpp
(emulation manifest), zlib (archive), vrtd.

### 7.1 The canonical flow

```cpp
vrt::Device device(bdf, vrtbinPath);
vrt::Kernel kernel(device, "my_kernel_0");

// Allocate on the exact bank this argument is wired to — no manual bank choice
vrt::Buffer<float> buf(device, 1024, kernel.argMemoryConfig("in"));

for (uint32_t i = 0; i < 1024; i++) buf[i] = float(i);
buf.sync(vrt::SyncType::HOST_TO_DEVICE);

kernel.setArg(0, 1024);
kernel.setArg(1, buf);
kernel.start();
kernel.wait();

uint32_t result = kernel.read(0x18);
```

What each step does underneath:

1. `Device` — extract the gzipped tar, parse `system_map.xml`, determine the
   platform, connect to vrtd, open the device, program the FPGA, discover
   kernels and memory config.
2. `Kernel` — look up the kernel in the loaded design.
3. `Buffer` — allocate device memory through vrtd → libslash → QDMA.
4. `sync` — QDMA H2C transfer.
5. `setArg` / `start` — AXI-Lite register writes, then set `ap_start`.
6. `wait` — poll `ap_done` / `ap_idle`.
7. `read` — BAR MMIO through PF2.

### 7.2 The self-configuring pattern

`portMemoryConfig(port)` and `argMemoryConfig(arg)` read the connection map out
of the vbin's `system_map.xml` and return a `MemoryConfig`:

```cpp
struct MemoryConfig {
    MemoryRangeType        type;      // DDR | HBM | HBM_VNOC
    std::optional<uint8_t> hbmPort;
};
```

Passing it straight into the `Buffer` constructor guarantees the allocation
matches the linker configuration. **This is the pattern to use** — hardcoding
`MemoryRangeType::HBM, 1` in host code silently diverges the moment someone
edits `config.cfg`.

`portMemoryConfig` throws if the port has no memory target, which makes it a
usable *probe*: catching the exception tells you the port is wired to `HOST`
rather than to device memory. **[field]** — this is exactly how the Vortex
runtime's `staged_probe()` self-configures between the two build variants.

### 7.3 Known API sharp edges **[field]**

- **`vrt::Kernel::wait()` has no timeout and blocks forever.** Poll `AP_CTRL`
  (offset `0x00`: bit0 `ap_start`, bit1 `ap_done`, bit2 `ap_idle`, bit3
  `ap_ready`; `ap_return` at `0x10`) with your own bound instead.
- **`vrt::Device`'s third constructor parameter is `program`.**
  `vrt::Device(bdf, vbin, /*program=*/false)` skips the PDI load entirely. This
  is what makes hardware iteration affordable (§9.1).
- **`std::cout` is fully buffered under `runuser`/`tee`.** Progress output must
  go to `std::cerr` or it is lost when a run is killed.

---

## 8. The vrtbin container

A `.vbin` is a **gzip-compressed tar archive**.

| File | Present | Purpose |
|---|---|---|
| `system_map.xml` | always | design metadata — the file that drives everything |
| `*.pdi` | hardware | FPGA image(s); VRT prefers `design.pdi` when several exist |
| `vpp_emu` | emulation | compiled C-model executable |
| `emu_manifest.json` | emulation | argument→call-type routing, register read-back |
| `vpp_sim` | simulation | Verilog simulator wrapper |
| `report_utilization.xml` | optional | LUT/FF/BRAM/URAM/DSP usage |

### 8.1 `system_map.xml`

```xml
<SystemMap>
  <Platform>Hardware</Platform>
  <ShellType>compute</ShellType>
  <ClockFrequency>250000000</ClockFrequency>

  <ServiceLayer>
    <Ethernet enabled="true"><eth index="0"/></Ethernet>
    <VIRT><interface index="0" connection="eth0"/></VIRT>
  </ServiceLayer>

  <Kernel>
    <Name>increment_0</Name>
    <BaseAddress>0x20100000000</BaseAddress>
    <Range>0x1000</Range>
    <register offset="0x00" name="CTRL" access="RW" range="32"/>
    <register offset="0x10" name="size" access="W"  range="32"/>
    <functional_args>
      <arg idx="0" name="size" type="scalar" offset="0x10" range="32" r="0" w="1"/>
      <arg idx="1" name="in"   type="buffer" offset="0x18" range="64" r="1" w="1"
           port="m_axi_gmem0"/>
    </functional_args>
    <connection port="m_axi_gmem0" target="HBM1"/>
  </Kernel>
</SystemMap>
```

Key elements:

- **`<Platform>`** — `Hardware` | `Emulation` | `Simulation`. Selects the VRT
  back-end. The *application binary is identical across all three*; only the
  vbin differs.
- **`<ShellType>`** — `service` or `compute`. VRT passes this to vrtd *before*
  programming, so the daemon can reset the board to the matching boot partition
  when needed. Legacy vbins without it are treated as `service`. **This element
  is the trigger for the shell-switch reset in §9.1.**
- **`<connection port=… target=…>`** — the port→bank map that
  `portMemoryConfig()` reads.
- **`<BaseAddress>`** — `0x201_0000_0000` is the PF0 BAR0 window into PL memory
  space (KB-2 §5.2), so kernel AXI-Lite maps land inside the 256 MB aperture.

Inspect without programming:

```bash
v80-smi inspect my_design.vbin
tar tzf my_design.vbin
```

---

## 9. Platform modes

| Platform | Transport | Target | Use |
|---|---|---|---|
| Hardware | PCIe BAR (PF2) + QDMA (PF1) | `hw` | production |
| Emulation | ZeroMQ IPC to a C-model | `emu` | functional verification, no board |
| Simulation | Verilog register map over ZeroMQ | `sim` | cycle-accurate RTL |

```cmake
add_vbin(TARGET "axilite_hw"  PLATFORM "hw"  CFG "${CFG_FILE}" KERNELS ${_KERNELS})
add_vbin(TARGET "axilite_emu" PLATFORM "emu" CFG "${CFG_FILE}" KERNELS ${_KERNELS})
add_vbin(TARGET "axilite_sim" PLATFORM "sim" CFG "${CFG_FILE}" KERNELS ${_KERNELS})
```

Documented limitations:

- **Emulation** — timing is not modelled; HLS kernels must have at least one
  AXI4-Lite interface; freerunning streaming chains (example `02_chain`) are
  unsupported.
- **Simulation** — much slower; needs Vivado and a simulator licence; memory
  round-trip fidelity can differ (floating-point representation in the simulator
  can introduce NaN artefacts the application must tolerate).

> **[field] Neither non-hardware platform can validate a host-mastering path.**
> Emulation shares process memory with the C-model, and simulation copies host
> memory into the model. A design whose kernel masters into host DRAM will pass
> both and fail on hardware. See §9.4.

---

## 10. Memory model

### 10.1 Targets

| VRT type | Linker target | Notes |
|---|---|---|
| `MemoryRangeType::DDR` | `DDR0`–`DDR3` | large capacity, lower bandwidth |
| `MemoryRangeType::HBM` + port | `HBM0`–`HBM63` | 64 pseudo-channels; **port is mandatory**, omitting it throws `std::invalid_argument` |
| `MemoryRangeType::HBM_VNOC` | `MEM` | aggregate across channels via the virtual NoC; no explicit channel |
| — | `VIRT` | network/virtual interface endpoint |
| — | `HOST` | QDMA slave bridge into host DRAM |

The kernel port and the buffer allocation must agree. `sp=increment_0.m_axi_gmem0:HBM1`
in the config must match `Buffer(device, n, MemoryRangeType::HBM, 1)` — or,
better, use `argMemoryConfig()` and let it match automatically.

### 10.2 The buddy allocator

Three tiers on hardware:

| Tier | Range | Implementation |
|---|---|---|
| SmallBlock | 4 KB – 2 MB | `BuddySuperblockBase<12, 21>`, carved from a 2 MB superblock |
| MediumBlock | 2 MB – 64 MB | `BuddySuperblockBase<21, 26>`, carved from a 64 MB superblock |
| LargeBlock | > 64 MB | allocated directly from vrtd as a standalone DMA buffer |

Allocation rounds up to a power of two, finds the smallest fitting block, and
splits repeatedly, returning unused halves to the free list. Free coalesces with
the buddy up the hierarchy.

> **[field] The practical consequence: the minimum allocation is 4096 bytes**
> (2^12). Requesting less throws `Size too small for MediumBlockSuperblock`. Any
> runtime that allocates small control structures — a head pointer, a completion
> word — must round up. This bit twice in this project.

### 10.3 Platform address schemes

| Platform | HBM base | DDR base | Mechanism |
|---|---|---|---|
| Hardware | real physical/bus addresses | " | buddy allocator over vrtd/libslash/QDMA |
| Emulation | `0x40_0000_0000` | `0x600_0000_0000` | fake addresses; data via ZeroMQ, no DMA |
| Simulation | same fake scheme | same | matches the memory map in the linker's `run_pre.tcl` |

The `Buffer<T>` API — construction, `sync()`, `operator[]` — is identical
across all three.

---

## 11. The hardware side: static shell + slashkit

### 11.1 The static shell

The *static shell* is the pre-built FPGA platform base shipped inside the
`slashkit` package. It holds the fixed infrastructure — including the SMBus
controller IP used for board management — that every hardware vrtbin links
against.

Build requirements that trip people up:

- **Vivado 2025.1 and Vitis 2025.1**, both sourced.
- **A Vivado Enterprise licence** — the SMBus IP is not available at the
  standard tier.
- The SMBus IP (`xilinx.com:ip:smbus:1.1`) is **not in the repository and not
  bundled with Vivado**. It must be downloaded from the AMD member portal and
  dropped into `linker/slashkit/resources/base/iprepo/`.

`linker/slashkit/resources/` contains the shell variants:
`static_shell_compute/`, `aved/`, `base/`, `dcmac/`, `sim/`, `templates/`.

### 11.2 slashkit — the linker

Python. Reads compiled HLS kernel IP (`component.xml`) plus a connectivity
config, emits a vrtbin. Structure: `parser/` (config), `core/` (block-design
port model, command config), `emit/{hw,emu,sim,metadata}/` (per-platform
generation).

Driven from CMake:

```cmake
add_vbin(TARGET "my_design_hw" PLATFORM "hw" CFG "config.cfg" KERNELS ${_KERNELS})
```

### 11.3 The connectivity language — `config.cfg`

INI-style, three sections that matter.

```ini
[connectivity]
nk=dma:1:dma_0                                   # instantiate kernel: <ip>:<count>:<names>
nk=offset:1:offset_0
stream_connect=offset_0.axis_out:dma_0.axis_in   # AXI4-Stream wiring
sp=dma_0.m_axi_gmem0:HBM0                        # AXI4-MM port → memory bank
sp=offset_0.m_axi_gmem0:DDR0

[network]
eth_0=1
eth_2=1

[source_scripts]
pre_synth=network_layer.tcl
```

- **`nk=`** — number of kernels: IP name, instance count, dot-separated instance
  names.
- **`sp=`** — the memory-bank map. Grammar: `<instance>.<port>:<TARGET>` where
  `TARGET` ∈ {`HBM0`…`HBM63`, `DDR0`…`DDR3`, `MEM`, `VIRT<n>`, `HOST`}. `MEM`
  and `HOST` take no index. The parser rejects anything else with
  *"Invalid memory target … Expected e.g. HBM0, DDR3, MEM, HOST."*
- **`stream_connect=`** — point-to-point AXI4-Stream between kernel ports, or
  between a kernel and an Ethernet port (`eth_0.tx0`, `eth_2.rx0`).

Example `05_perf` instantiates 76 kernels and maps them across all 64 HBM
channels, 8 `MEM` (VNOC) ports, and 4 DDR banks — a useful reference for what
the shell can actually route.

### 11.4 `HOST` vs device memory **[field]**

`HOST` resolves to the QDMA slave bridge, sinking at
`/qdma_slave_bridge_noc/S00_AXI` (`slashkit/emit/hw/tcl_gen.py:86` — *"HOST
(QDMA bridge) uses NoC pin"*). It is a designed, supported feature with driver
backing (`slash_hostbuf.c`).

**On the V80 compute shell, it does not work.** Controlled experiment, two
builds whose HLS source is byte-identical (`diff -q` verified), differing only
in one line of `config.cfg`:

| build | `sp=` target | AP_CTRL trace | outcome |
|---|---|---|---|
| `hostprobe` | `HOST` | stuck at `0x1` (`ap_start`) for ~15 min | never completes |
| `hostprobe_hbm` | `HBM0` | `0x4` → `0x1` → `0xe` | `sum=524800` correct, < 0.1 s |

The fix used in this project: route the CP's command ring to **HBM1** instead
and stage host-visible structures in device memory, with the runtime detecting
which variant it was handed via `portMemoryConfig("m_axi_host")`. With that,
`TARGET=hw` passes:

```
[VXDRV] m_axi_host targets device memory; staging CP memory there (HBM port 1)
[minimal] queue_create → buffer_create → write → readback → wait → verify
PASSED!
```

---

## 12. v80-smi

BDF forms accepted everywhere: `BB:DD`, `BB:DD.F`, `DDDD:BB:DD`, `DDDD:BB:DD.F`.

| Command | Purpose |
|---|---|
| `version [-p]` | version, `-p` for bare `x.y.z` |
| `list [-l] [-s] [-j\|-J]` | enumerate boards, readiness (PF0/PF1/PF2/VRTD), long, sensors, JSON |
| `inspect <vbin>` | metadata from a file on disk, no device needed |
| `program <vbin> -d <BDF>` | extract the PDI and program the FPGA |
| `query -d <BDF>` | what *you* last wrote to that BDF — **not** what is physically loaded |
| `reset -d <BDF>` | SBR + rescan; board returns unprogrammed |
| `validate -d <BDF> [-j N]` | HBM+DDR integrity (`i ^ seed`) and H2C/C2H bandwidth; N threads, default 8, max 64 |
| `write-static-shell --flash` | write the SLASH static shell to OSPI over PCIe |

The `query` caveat is worth repeating because it misleads: *querying the actual
on-board design is not currently possible.* Treat the output as a guide.

---

## 13. Testing and coverage

The kernel module ships a **kselftest** suite requiring a physical V80:

```sh
make                                   # build slash.ko
make -C tests/ all
sudo insmod ./slash.ko
echo 1 | sudo tee /sys/bus/pci/rescan
sudo make -C tests/ run                # TAP output
sudo SLASH_TEST_DMA_ADDR=0x100000000 make -C tests/ run   # override DMA target
```

Each fixture tears down its queue pairs on failure, so a failing test does not
leave the device broken.

Coverage, on a `CONFIG_GCOV_KERNEL=y` kernel with `lcov`/`genhtml`:

```sh
./test_module.sh 0000:03:00     # build with GCOV=1 → load → test → coverage/index.html
```

`libslash`'s mock backends (`ctldev_mock.c`, `qdma_mock.c`) let everything above
the driver be tested with no board at all.

---

## 14. Practical notes for this project **[field]**

Consolidated rules learned running Vortex on this stack:

1. **Program once per session.** A design write succeeds only on a freshly reset
   device; vrtd resets only on a shell *change*. Use
   `vrt::Device(bdf, vbin, false)` for every run after the first. A failed load
   drives the AMC to `NO_AMC` and costs a JTAG recovery.
2. **A JTAG fabric load sets `Shell: unknown`**, which forces a shell switch —
   and therefore an SBR — on the next program. Prefer not to JTAG-load unless
   the card is genuinely off the bus.
3. **`HOST` mastering is unusable on this shell.** Route kernel `m_axi` ports to
   HBM/DDR and stage anything the device must pull from in device memory.
4. **Detect, don't hardcode.** `portMemoryConfig(port)` throwing means the port
   targets `HOST`; returning a config means device memory. One runtime binary
   can then serve both build variants.
5. **Round small allocations up to 4096 bytes** before handing them to the
   allocator.
6. **Never sync a command ring device→host.** A refresh loop that syncs *all*
   staged regions will clobber descriptors that have been appended but not yet
   published. Exclude the ring explicitly — VRT's own `sim_refresh()` does, and
   that is where the bug was caught before it cost a reset.
7. **Bound every wait yourself.** `Kernel::wait()` never returns on a wedged
   kernel; poll `AP_CTRL` with a deadline.
8. **Print to `stderr`.** Anything on `stdout` is lost when a run is killed.
9. **Unload `slash` and `ami` before any link transition** — AMD's own comment
   in `reset.c:334` says a bound driver touching a down link can machine-check
   the host.

---

## Sources

**Local (authoritative for SLASH):** `~/dev/SLASH-compute/`
- `README.md`, `driver/README.md`, `vrt/README.md`
- `docs/explanation/{architecture,pcie-topology,memory-model,platform-modes,vrtbin-format}.rst`
- `docs/reference/kernel-abi/index.rst`
- `docs/reference/vrtd/{client-flow,configuration}.rst`
- `docs/reference/libslash-api/{ctldev,qdma,hotplug}.rst`
- `docs/reference/smi/commands.rst`
- `docs/tutorials/admin/{bootstrap-aved,platform-setup,device-management}.rst`
- `docs/howto/migrate-from-xrt.rst`
- `driver/{slash_config.h,slash_pcie.c,slash_hotplug.c,slash_hostbuf.c,slash_qdma.c}`
- `linker/slashkit/{parser/config_parser.py,core/bd_ports.py,emit/hw/tcl_gen.py}`
- `examples/*/config.cfg`

**Upstream**
- [AVED Overview](https://xilinx.github.io/AVED/latest/AVED+Overview.html)
- [AVED — Host to Card Communication](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+-+Host+to+Card+Communication.html)
- [AVED V80 — CIPS Configuration](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+V80+-+CIPS+Configuration.html)
- [XRT and Vitis Platform Overview](https://xilinx.github.io/XRT/master/html/platforms.html)
- [QDMA Linux Driver Architecture (DeepWiki)](https://deepwiki.com/Xilinx/dma_ip_drivers/2.1.1-qdma-linux-driver-architecture)
- [Controlling Hardware — UG1399 (Vitis HLS AP_CTRL)](https://docs.amd.com/r/2020.2-English/ug1399-vitis-hls/Controlling-Hardware)

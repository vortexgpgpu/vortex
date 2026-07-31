# AVED/SLASH AFU — Alveo V80 Backend for Vortex

**Scope:** `hw/rtl/afu/aved/`, `hw/syn/xilinx/aved/`, `sw/runtime/aved/`, `sim/avedsim/`, [VX_config.toml](../../VX_config.toml), [ci/blackbox.sh](../../ci/blackbox.sh)
**Reference:** existing XRT backend — [hw/rtl/afu/xrt/](../../hw/rtl/afu/xrt/), [hw/syn/xilinx/xrt/](../../hw/syn/xilinx/xrt/), [sw/runtime/xrt/](../../sw/runtime/xrt/), [sim/xrtsim/](../../sim/xrtsim/)
**Target platform:** AMD Alveo V80 (Versal `xcv80-lsva4737-2MHP-e-S`) via the SLASH platform (VRT runtime, `slashkit` linker, `vrtd` daemon, AVED shell)
**Status:** Phases 3-6 implemented; phase 7 gated on §10 Q7

---

## 1. Motivation

Vortex's only Xilinx FPGA path today is XRT: an `.xclbin` built by `v++`,
loaded through `libxrt_coreutil`, targeting Alveo XDMA shells (U50, U55C,
U200, U250, U280). The Alveo V80 is not an XDMA-shell board. It ships with
AVED (the Alveo Versal Example Design) and is driven by the SLASH stack —
a different runtime (VRT), a different device binary (`.vbin`), a different
linker (`slashkit`), and a different kernel-driver topology (three PFs:
`ami` / `slash_qdma` / `slash_ctl`). XRT does not apply.

Adding an AVED backend gives Vortex:

- A Versal-class prototyping target with HBM, complementing the U55C
  baseline that [docs/coding_guidelines_verilog.md](../coding_guidelines_verilog.md#10-combinational-logic-depth--timing-closure)
  names as the 300 MHz timing-closure reference.
- A second, independent AFU surface. Per
  [AGENTS.md §4](../../AGENTS.md), `xrt` is the canonical RTL-integration
  coverage path; a second platform backend keeps the AFU boundary honest by
  proving it is not silently XRT-shaped.
- A migration path off Vitis packaging. SLASH consumes raw IP-XACT
  directories, so the `.xo` container step disappears.

---

## 2. The existing XRT backend (the shape we are copying)

The XRT path is four cooperating pieces. The AVED backend mirrors all four.

| Layer | XRT artifact | Responsibility |
|---|---|---|
| AFU RTL | [hw/rtl/afu/xrt/VX_afu_wrap.sv](../../hw/rtl/afu/xrt/VX_afu_wrap.sv) | AXI-Lite demux, CP instantiation, bank-0 arbitration, `m_axi_host` wiring |
| Synthesis | [hw/syn/xilinx/xrt/Makefile](../../hw/syn/xilinx/xrt/Makefile) + [platforms.mk](../../hw/syn/xilinx/xrt/platforms.mk) | `gen_xo.tcl` → `.xo` → `v++ --link` → `.xclbin` |
| Runtime | [sw/runtime/xrt/vortex.cpp](../../sw/runtime/xrt/vortex.cpp) | implements the 6-callback transport HAL over XRT |
| Simulation | [sim/xrtsim/](../../sim/xrtsim/) | Verilator model of the AFU behind an XRT-shaped C shim |

### 2.1 The backend contract is small

The entire backend surface is [sw/runtime/common/callbacks.h](../../sw/runtime/common/callbacks.h)
— six function pointers:

```c
int (*dev_open)  (void** out_dev_ctx);
int (*dev_close) (void*  dev_ctx);
int (*cp_reg_write)(void* dev_ctx, uint32_t off, uint32_t value);
int (*cp_reg_read) (void* dev_ctx, uint32_t off, uint32_t* out_value);
int (*host_mem_alloc)(void* dev_ctx, uint64_t size,
                      void** out_host_ptr, uint64_t* out_cp_addr);
int (*host_mem_free) (void* dev_ctx, uint64_t cp_addr);
```

Device-memory allocation, DMA, and capability decoding all live in the
common core; the Command Processor is the sole memory engine. That means a
new backend is **a register channel plus a CP-visible memory region** — and
nothing else. Five of the six callbacks map onto VRT trivially. The sixth,
`host_mem_alloc`, is the entire technical risk of this proposal and is
treated separately in §4.

### 2.2 The AFU address map

`VX_afu_wrap` splits the AXI-Lite slave on address bit 12:

- `0x0000..0x0FFF` → `VX_afu_ctrl`: an `ap_ctrl` stub at `0x00` plus the
  SCOPE bit-serial register pair at `0x28`/`0x2C`.
- `0x1000..0x1FFF` → the Command Processor regfile
  ([VX_cp_axil_regfile.sv](../../hw/rtl/cp/VX_cp_axil_regfile.sv)), mapped
  to the CP's native `0x000`-based 12-bit space.

The data plane is `VX_CFG_PLATFORM_MEMORY_NUM_BANKS` AXI4 masters
(`m_axi_mem_<i>`) plus one host-memory master (`m_axi_host`). The CP shares
bank 0 with Vortex through `VX_mm_axi_arb`.

This map is platform-independent and carries over to AVED unchanged.

---

## 3. Target platform: SLASH / VRT

### 3.1 Stack

```
App  →  libvrt  →  vrtd (daemon)  →  slash (kernel module)  →  V80
```

Three PCIe physical functions, each with its own driver and role:

| PF | Device ID | Driver | Role |
|---|---|---|---|
| PF0 | `0x50B4` | `ami` | AVED management (sensors, identity, firmware) |
| PF1 | `0x50B5` | `slash_qdma` | H2C/C2H DMA for buffers and streaming |
| PF2 | `0x50B6` | `slash_ctl` | BAR MMIO — kernel register reads/writes |

VRT addresses boards by **board BDF** (`BB:DD` or `DDDD:BB:DD`, no function
suffix). All four readiness checks (PF0/PF1/PF2/VRTD) must pass.

### 3.2 API mapping

| XRT | VRT | Note |
|---|---|---|
| `xrt::device(index)` + `load_xclbin` | `vrt::Device(bdf, vbin)` | opening and programming are one step |
| `xrt::ip` / `xrt::kernel` | `vrt::Kernel(device, "name_0")` | instance suffix required |
| `xrtKernelWriteRegister` | `kernel.write(offset, value)` | |
| `xrtKernelReadRegister` | `kernel.read(offset)` | |
| `xrt::bo(dev, bytes, group)` | `vrt::Buffer<T>(dev, elems, cfg)` | typed, element-counted |
| `bo.sync(XCL_BO_SYNC_*)` | `buf.sync(vrt::SyncType::*)` | |
| `kernel.group_id(n)` | `kernel.argMemoryConfig("name")` | |
| `.xclbin` | `.vbin` (tar: PDI + `system_map.xml`) | platform embedded in metadata |
| `XCL_EMULATION_MODE` | (none — platform is a vbin property) | |
| `xbutil` | `v80-smi` | |

### 3.3 Terminology inversion — a real footgun

SLASH renames both emulation tiers, and the names collide with XRT's:

| Vitis/XRT | SLASH | Meaning |
|---|---|---|
| software emulation | **emulation** (`emu`) | behavioural C-model |
| hardware emulation | **simulation** (`sim`) | RTL Verilog simulation |

SLASH's `emu` target is **not supported for RTL kernels** — only `hw` and
`sim`. Vortex is an RTL kernel, so only two SLASH targets are reachable.
Vortex's own `sim/avedsim` Verilator model (§6.3) is a third, distinct
thing and must not be conflated with either.

---

## 4. Gap analysis

### 4.1 Critical: there is no host-memory aperture in VRT

The XRT AFU reaches host DRAM through the platform slave bridge, requested
in [platforms.mk](../../hw/syn/xilinx/xrt/platforms.mk) as:

```make
VPP_FLAGS += --connectivity.sp vortex_afu_1.m_axi_host:HOST[0]
```

and allocated host-side as an `XRT_BO_FLAGS_HOST_ONLY` BO. The runtime
places the CP command ring and DMA staging there, and the callback contract
states the region **must be coherent with the CP's `m_axi_host` view, with
no explicit sync callback**.

VRT has no equivalent. `MemoryRangeType` is exactly `{DDR, HBM, HBM_VNOC}`
(`vrt/include/vrt/allocator/allocator.hpp`). There is no host-only buffer
flag, no `HOST[n]` connectivity tag in the `slashkit` config grammar, and
no documented slave-bridge aperture. A sweep of `vrt/`, `docs/`, and the
linker for `host_only` / slave-bridge / `HOST[` turned up nothing.

**This is the one design decision that must be settled before implementation
starts.** Three candidate resolutions, in order of preference:

**(a) Device-resident command ring + explicit sync (recommended).**
Place the ring in HBM. `host_mem_alloc` returns a shadow host buffer plus
the HBM physical address; the runtime memcpy's into the shadow and a new
`host_mem_sync` callback issues `Buffer::sync(HOST_TO_DEVICE)` before the
doorbell write. `m_axi_host` is then just a second device-memory master.

Cost: extends `callbacks.h` with a seventh callback. Every existing
backend gains a no-op implementation. This is a change to a shared
contract, so it needs sign-off — but it is a *small*, explicit change, and
it makes the host/device coherence assumption visible instead of implied.
Latency cost is one QDMA H2C descriptor per submission batch, amortized
across the ring entries in that batch.

**(b) BAR-mapped HBM aperture.** `vrtd` exposes `vrtd::Bar` /
`vrtd::BarFile` with `mmap()` and a raw `volatile void*`. If the AVED shell
maps an HBM window into PF2's BAR, `host_mem_alloc` can return that mapped
pointer directly and coherence is PCIe-ordered — no contract change, no
sync. This is the cleanest outcome *if the aperture exists and is large
enough for the ring*. **Requires hardware verification** (§10, Q1).

**(c) Versal CPM/slave-bridge host access.** If AVED exposes a host-memory
AXI aperture the way XDMA shells do, this is a direct port. Nothing in the
SLASH sources or docs suggests it does. **Requires verification** (§10, Q1).

Recommendation: implement against (a), and collapse to (b) if hardware
inspection shows a usable BAR-mapped aperture. Do **not** silently break
the coherence contract by returning an unsynced device pointer from
`host_mem_alloc` — that would produce intermittent, near-undebuggable
command-ring corruption.

### 4.2 Interface naming

SLASH requires the AXI-Lite control interface to be named **exactly**
`s_axi_control` (case-sensitive). Vortex's AFU uses `s_axi_ctrl`. Resolve
in the AVED shim's port list, not by renaming `VX_afu_wrap` — the XRT shim
depends on the current name.

The `interrupt` port name is already correct (`interrupt`, active-high).

### 4.3 Packaging

| XRT | AVED |
|---|---|
| `gen_xo.tcl` + `package_kernel.tcl` → `.xo` | Vivado `ipx::package_project` → IP-XACT `component.xml` |
| `v++ --link --platform <xpfm>` | `slashkit link -c cfg -p hw -k component.xml -o out.vbin` |
| `--connectivity.sp k.port:HBM[0:31]` | `sp=vortex_afu_0.m_axi_mem_0:HBM1` in a `[connectivity]` cfg |
| `--kernel_frequency` | `vrt::Device::setFrequency()` at runtime, plus linker target |

SLASH additionally recommends `HAS_BURST=0` and `SUPPORTS_NARROW_BURST=0`
in the IP `bd.tcl`, and requires 64-bit `m_axi` addresses with no WRAP or
FIXED bursts. Vortex's AXI masters already satisfy the burst-type
constraints; `AxSIZE` must be checked to match the 512-bit data width.

### 4.4 Toolchain version

SLASH documents Vivado/Vitis **2025.1** and warns other versions may break.
This host has **2025.2** installed at `/opt/xilinx/2025.2`. Either
validate 2025.2 against the SLASH flow or install 2025.1 alongside. Flag
before committing to a synthesis run — a mid-build toolchain mismatch is
expensive.

---

## 5. Proposed structure

```
hw/rtl/afu/aved/
    vortex_afu.vh          # port macros (AVED naming: s_axi_control)
    VX_afu_wrap_aved.sv    # AVED AFU top; reuses VX_afu_ctrl + VX_cp_core
    VX_afu_ctrl.sv         # shared with xrt, or symlinked/included

hw/syn/xilinx/aved/
    Makefile               # package IP → slashkit link → .vbin
    package_ip.tcl         # ipx::package_project → component.xml
    platforms.mk           # V80 HBM bank/addr-width config
    config.cfg.in          # slashkit [connectivity] template

sw/runtime/aved/
    Makefile
    vortex.cpp             # 6-callback HAL over vrt::Device / vrt::Kernel

sim/avedsim/
    Makefile
    vortex_afu_shim.sv     # AVED-named shim over VX_afu_wrap_aved
    vrt_c.h / vrt_c.cpp    # VRT-shaped C shim (mirrors xrt_c.{h,cpp})
    vrt_sim.h / vrt_sim.cpp
    verilator.vlt.in
```

Naming note: `aved` is the user-requested directory name and matches the
shell. `slash` would name the software stack instead. Keeping `aved`
throughout is consistent with `xrt` naming the runtime rather than the
shell — a minor inconsistency, accepted for continuity with the request.

---

## 6. Component design

### 6.1 `hw/rtl/afu/aved/`

`VX_afu_wrap_aved.sv` is a near-copy of the XRT wrapper. Deltas:

- AXI-Lite slave ports renamed to the `s_axi_control_*` prefix.
- `m_axi_host` becomes a second device-memory master under resolution (a),
  addressed into a reserved HBM range rather than a host aperture. Under
  (b)/(c) it stays a host master and the file is closer to identical.
- `PLATFORM_MEMORY_OFFSET` is applied per bank exactly as in XRT; the V80
  HBM base offset comes from `system_map.xml` at link time.

The AXI-Lite demux, `VX_afu_ctrl` instance, `VX_cp_core` instance, bank-0
`VX_mm_axi_arb`, and reset-shift-register logic are unchanged. Per
[coding_guidelines_verilog.md §11](../coding_guidelines_verilog.md#11-reuse-the-hardware-ip-library),
no new arbitration or buffering logic should be written — reuse
`VX_mm_axi_arb` and the existing elastic buffers.

**Duplication concern.** `VX_afu_wrap.sv` is ~760 lines and the AVED copy
would differ in perhaps 40. A shared core module parameterized on port
naming is the better factoring, but refactoring the XRT wrapper is
out of scope here and touches the canonical RTL coverage path. Proposal:
land the copy first, prove the flow, then factor the common body into
`hw/rtl/afu/common/` as a follow-up with both backends green.

### 6.2 `sw/runtime/aved/vortex.cpp`

Direct translation of the XRT backend. The class shape (`vx_device` with
`init`, `cp_reg_read/write`, `host_mem_alloc/free`, then
`#include <callbacks.inc>`) carries over verbatim.

```cpp
int init() {
  const char* bdf  = getenv("VRT_DEVICE_BDF");   // e.g. "11:00"
  const char* vbin = getenv("VRT_VBIN_PATH");    // default "vortex_afu.vbin"
  device_ = vrt::Device(bdf, vbin);
  kernel_ = vrt::Kernel(device_, "vortex_afu_0");
  // ap_ctrl reset handshake — identical to the XRT path
}

int cp_reg_write(uint32_t off, uint32_t value) {
  kernel_.write(CP_BASE + off, value);           // CP_BASE = 0x1000
  return 0;
}
```

VRT throws on error, so every VRT-touching member needs the same
try/catch discipline as `XRT_TRY`/`XRT_CATCH` — propagating an exception
across the `extern "C"` `callbacks_t` boundary is UB.

Device discovery via `VRT_DEVICE_BDF` mirrors XRT's `XRT_DEVICE_INDEX`.
Defaulting to the first board found by scanning sysfs for `10ee:50b4` is a
reasonable convenience, but the env var must win.

Per [coding_guidelines_cpp.md §8](../coding_guidelines_cpp.md#8-source-tree-layering--sw--hwsim-bidirectional-isolation),
this file must not reach into `hw/*` or `sim/*`. The simulation build
includes `sim/avedsim/vrt_c.h` the same way the XRT backend includes
`xrt_c.h` under `-DXRTSIM` — that inclusion is from `sw/runtime/`, so it
is a boundary violation in the same shape as the existing one. **This
mirrors an existing exemption rather than introducing a new class of
violation**, but `ci/check_sw_sim_boundary.sh` must be checked to confirm
the AVED path is covered by whatever mechanism already permits the XRT
one; if it is not, the shim header belongs in `sw/common/`.

### 6.3 `sim/avedsim/`

Mirrors `sim/xrtsim/` exactly: a Verilator build of `vortex_afu_shim`
producing `libavedsim.so`, with `vrt_c.{h,cpp}` exposing a VRT-shaped C
surface and `vrt_sim.{h,cpp}` implementing the AXI bus models
(`axi_ctrl_bus_eval`, `axi_mem_bus_eval`, `axi_host_bus_eval`) against
Ramulator-backed memory.

Under resolution (a) the host-memory model changes meaningfully: the
`axi_host` master now targets simulated device memory, so `host_mem_alloc`
allocates from the Ramulator bank rather than returning process memory,
and `host_mem_write` becomes a real memory write instead of a `memcpy` to
a host pointer. This is a *simplification* of `xrt_sim.cpp`'s host path,
not an addition.

This is the fast iteration loop — Verilator, no Vivado, no hardware. It is
what `blackbox.sh --driver=aved` should use.

SLASH's own `sim` platform (Vivado `xsim` behind a vbin) is a separate,
much slower loop that validates the *packaging and linker* output rather
than the RTL. Worth running once per integration milestone, not per commit.

### 6.4 `hw/syn/xilinx/aved/`

```
package_ip.tcl:   read_verilog … ; ipx::package_project → component.xml
Makefile:         slashkit link -c config.cfg -p hw \
                    -k <ip>/component.xml -o vortex_afu.vbin
```

`platforms.mk` for V80 sets the HBM bank count, address width, and
`PLATFORM_MEMORY_OFFSET`. Concrete values must be read from the platform
metadata rather than assumed — see §10, Q2. The `PLATFORM_MERGED_MEMORY_INTERFACE`
path used for U55C/U50 (one wide master fanned across all HBM channels) is
likely the right starting point for V80 HBM as well, since it avoids the
per-bank virtual-base problem that forced U250 down to a single bank.

`config.cfg` is generated from a template so bank count stays consistent
with `VX_config.toml`:

```
[connectivity]
nk=vortex_afu:1:vortex_afu_0
sp=vortex_afu_0.m_axi_mem_0:HBM0
```

---

## 7. Build targets — `TARGET=hw|emu|sim|avedsim`

### 7.1 The four targets

`TARGET` selects *what executes the RTL*. It is the single knob that spans
the runtime link, the device binary, and the test harness.

| `TARGET` | Executes the RTL | Device binary | Runtime links | Speed |
|---|---|---|---|---|
| `hw` | V80 silicon | `.vbin` (`slashkit -p hw`) | `libvrt` | real time |
| `sim` | Vivado `xsim` | `.vbin` (`slashkit -p sim`) | `libvrt` | very slow (~1 s per trivial kernel) |
| `emu` | SLASH C-model | `.vbin` (`slashkit -p emu`) | `libvrt` | **unavailable — see §7.2** |
| `avedsim` | Verilator, in-tree | none | `libavedsim.so` | fast — the iteration loop |

The important structural property: **for `hw`, `sim`, and `emu` the runtime
binary is byte-identical.** VRT auto-detects the platform from the vbin
metadata — there is no `XCL_EMULATION_MODE` equivalent — so only the vbin
changes. Just `avedsim` alters the link, swapping `libvrt` for the
in-tree shim. This is simpler than the XRT path, where `hw_emu` still
linked real XRT and the distinction lived in the Vitis build.

### 7.2 `TARGET=emu` is accepted but not supported

SLASH's `emu` platform is a *behavioural C-model* target and, per
`docs/howto/use-rtl-kernels.rst`, **RTL kernels support only `hw` and
`sim`**. Vortex is an RTL kernel, so `emu` can never produce a working
vbin.

It is nonetheless named in the value space, for two reasons: the value
exists in SLASH and silently omitting it invites someone to assume it
works, and the failure must be a clear diagnostic rather than an obscure
`slashkit` traceback. `hw/syn/xilinx/aved/Makefile` should reject it
explicitly:

```make
ifeq ($(TARGET), emu)
$(error TARGET=emu is unavailable: SLASH emulation is a C-model platform \
        that does not accept RTL kernels. Use TARGET=avedsim for fast \
        functional iteration, or --driver=simx for the C-model)
endif
```

Note the terminology trap from §3.3: a reader coming from Vitis will read
"emulation" as *hardware* emulation, which in SLASH is called `sim` and
*is* supported. The error message names the alternative to short-circuit
that confusion.

### 7.3 Resolving `TARGET` in each Makefile

Two Makefiles consume it, and Vortex already overloads the name
inconsistently — `sw/runtime/xrt/Makefile` uses `TARGET ?= xrtsim`
(a backend selector) while `hw/syn/xilinx/xrt/Makefile` uses `TARGET ?= hw`
(a Vitis build target). The AVED backend should **not** inherit that split.
One name, one value space, both files:

`sw/runtime/aved/Makefile` — only `avedsim` differs:

```make
TARGET ?= avedsim

ifeq ($(TARGET), avedsim)
	AVEDSIM = $(DESTDIR)/libavedsim.so
	CXXFLAGS += -DAVEDSIM -I$(SIM_DIR)/avedsim
	LDFLAGS += -L$(DESTDIR) -lavedsim
else
	# hw / sim / emu all link real VRT; the vbin selects the platform
	LDFLAGS += -lvrt
endif
```

`hw/syn/xilinx/aved/Makefile` — `avedsim` needs no vbin at all:

```make
TARGET ?= hw

ifeq ($(TARGET), avedsim)
$(error TARGET=avedsim builds no device binary; build sim/avedsim instead)
endif

VBIN_PLATFORM := $(TARGET)          # hw | sim  (emu rejected above)
```

### 7.4 Test-harness plumbing

`ci/blackbox.sh` currently selects only a driver. AVED needs the target
too, since `--driver=aved` is ambiguous across four execution modes:

```bash
./ci/blackbox.sh --driver=aved --target=avedsim --app=demo   # default
./ci/blackbox.sh --driver=aved --target=sim     --app=demo
./ci/blackbox.sh --driver=aved --target=hw      --app=demo
```

`--target` defaults to `avedsim` so the common case stays a one-flag
invocation and CI never accidentally reaches for Vivado or a board. It is
passed through to `build_driver` as `TARGET=$TARGET`, exactly as `DEBUG`
and `SCOPE` already are, and is inert for every other driver.

`hw` and `sim` additionally need the device binary located at run time via
`VRT_VBIN_PATH` (and `VRT_DEVICE_BDF` for `hw`); `avedsim` needs neither,
since the shim links the model directly.

### 7.5 Which target to use when

| Situation | Target |
|---|---|
| Day-to-day RTL/runtime iteration, CI smoke, regression | `avedsim` |
| Validating IP-XACT packaging, linker metadata, `system_map.xml` | `sim` |
| Performance, timing closure, final acceptance | `hw` |
| Fast functional check with no AFU involvement | `--driver=simx` (not an AVED target) |

Only `avedsim` belongs in per-commit CI. `sim` is a per-milestone gate —
measured at ~0.9 s of host time for a *two-HLS-kernel* design
([slash_v80_bringup_report.md §3](slash_v80_bringup_report.md)), so a
Vortex-sized design is impractical to run per commit.

**`TARGET=sim` needs neither a board nor `vrtd`.** This was confirmed
empirically during bring-up: all three SLASH examples ran to completion on
the `sim` platform while `v80-smi` reported `VRTD: NOT READY` and the V80
itself was unusable. `vrt::Device` still takes a BDF argument, but on the
simulation platform it is used only to key a metadata cache directory —
the device is never opened. That makes phases 5–6 reachable on any machine
with Vivado, with no dependency on Q7 or on working hardware, and it means
`sim` is a viable CI gate on a build machine that has no FPGA at all.

---

## 8. Build and test integration

- `sw/runtime/Makefile`: add `aved` to the `all` target and the clean list,
  alongside `stub rtlsim simx opae xrt`.
- `ci/blackbox.sh`: add `aved` to the `--driver` list and to
  `set_driver_path`. The usage string at line 29 already has a typo
  (`oape`) — fix while touching it.
- `configure` / `config.mk.in`: expose `VRT_HOME` (or reuse `SLASH_HOME`)
  the way `XILINX_XRT` is exposed today.
- Smoke: `./ci/blackbox.sh --driver=aved --app=demo`
- Regression: `make -C tests/regression run-aved` once the driver is green.

Per [AGENTS.md §2](../../AGENTS.md), the AVED work needs its own build
directory (`build_aved64/`) — never share one with the XRT or simx configs.

---

## 9. Phased plan

| Phase | Deliverable | `TARGET` | Gate |
|---|---|---|---|
| 1 | Start static-shell procurement (§10 Q7) | — | Runs in parallel with 2–6; must land before 7 |
| 2 | Resolve §10 Q1 (host-memory aperture) | — | Design decision recorded; §4.1 resolution chosen |
| 3 | ✅ `sim/avedsim` + `sw/runtime/aved` against the **XRT** AFU RTL | `avedsim` | exact parity with `--driver=xrt` on demo, sgemm, 2c/2w/2t |
| 4 | ✅ `hw/rtl/afu/aved` shim with AVED port naming | `avedsim` | green; shared body factored to `hw/rtl/afu/common/`, xrt parity unchanged |
| 5 | ✅ `hw/syn/xilinx/aved` packaging → `component.xml` | `sim` | IP-XACT infers `s_axi_control`, `m_axi_mem_0`, `m_axi_host`; `HAS_BURST=0` |
| 6 | ✅ `slashkit link -p sim` produces a vbin | `sim` | 60 MB vbin; `v80-smi inspect` resolves `vortex_afu_0` |
| 7 | `hw` vbin, on-hardware bring-up | `hw` | `demo` + `sgemm` pass on the V80 |
| 8 | Regression suite, CI wiring, `docs/designs/` writeup | `avedsim` | `run-aved` green |

Phase 3 deliberately runs the new runtime against the *old* RTL. That
isolates runtime bugs from RTL bugs — the single highest-value sequencing
decision in this plan.

**Phases 3–6 need no hardware and no static shell.** Only phase 7 does.
Phase 2 is listed early because it decides §4.1, but if Q1 cannot be
answered without a board, implement against resolution (a) and revisit —
(a) is correct-but-suboptimal if the aperture turns out to exist, whereas
(b) is *incorrect* if it does not.

---

## 10. Open questions

**Q1 (blocking, phase 2).** Does the V80/AVED shell expose any host-memory
aperture reachable from an AXI master, or a BAR-mapped HBM window large
enough for the CP command ring? This determines §4.1's resolution and
whether `callbacks.h` needs a seventh callback. Answer by inspecting a
programmed board's `system_map.xml` and PF2 BAR length via
`vrtd::Bar::getLength()`.

**Q2.** V80 HBM channel count, per-channel capacity, and the address offset
the linker assigns. Read from `system_map.xml` / `v80-smi inspect`, not
from datasheet recall.

**Q3.** Is `slashkit`'s `[connectivity]` grammar able to express a single
master fanned across all HBM channels, the equivalent of
`--connectivity.sp k.m_axi_mem_0:HBM[0:31]`? If not,
`PLATFORM_MERGED_MEMORY_INTERFACE` is unavailable and the AFU needs one
master per channel.

**Q4.** Does `vrt::Kernel::write/read` have acceptable latency for the CP
doorbell path? It routes through `vrtd` over `AF_UNIX` rather than a
direct `mmap`, unlike XRT's ioctl path. If per-register latency is high,
the `vrtd::BarFile` direct mapping is the escape hatch — `Kernel` already
caches one internally.

**Q5.** Vivado 2025.1 vs the installed 2025.2 (§4.4). *Partly answered by
[slash_v80_bringup_report.md](slash_v80_bringup_report.md): 2025.2 works for
HLS synthesis, IP-XACT packaging, `slashkit link -p sim`, and `xsim`
execution. The hardware link path is still untested against it.*

**Q6 — answered, with a caveat worth raising separately.**
`ci/check_sw_sim_boundary.sh` passes for the AVED backend. But it passes
for the *wrong reason*, and the same is true of the existing XRT backend:

- The `#include <vrt_c.h>` in `sw/runtime/aved/vortex.cpp` is genuinely
  clean — a bare header name, no `sim/` path component — so the
  include scan has nothing to flag.
- The `-I$(SIM_DIR)/avedsim` build flag **should** trip the Makefile scan
  (`-I[^[:space:]]*(hw|sim)/`), but does not: the pattern matches
  lowercase `sim/` against unexpanded Makefile text, and `$(SIM_DIR)` is
  uppercase. `sw/runtime/xrt/Makefile` evades it identically with
  `-I$(SIM_DIR)/xrtsim`.

So this is a **blind spot in the checker, not a sanctioned exemption**.
The AVED backend is exactly as compliant as the XRT backend — no better,
no worse — which is the right outcome for consistency, but neither is
truly clean. Tightening the pattern to catch `$(SIM_DIR)` / `$(HW_DIR)`
would flag both, and `sw/runtime/opae` already shows the intended way to
record a deliberate exception (explicit path exclusion). Fixing the
checker is out of scope here; it should be raised as its own issue so the
exemption becomes explicit rather than accidental.

**Q7 (blocking, procurement — not engineering).** A hardware vbin links
against a prebuilt *static shell* that is not in the SLASH repository, and
**PF1/PF2 do not exist until that shell is flashed to OSPI** via
`ami_tool cfgmem_program -t primary`. Obtaining it requires the SMBus IP
(`xilinx.com:ip:smbus:1.1`) from the AMD member portal **and** a Vivado
Enterprise license, plus Vivado 2025.1. Neither is present on the current
host, and there is no published SLASH package feed supplying a prebuilt
shell. The same member-portal archive also carries the Tandem PCIe image
that fixes the enumeration race documented in
[slash_v80_bringup_report.md §6.6](slash_v80_bringup_report.md).
**This gates the whole hardware phase and has a lead time no engineering
work removes — start it before anything else.**

---

## 11. Validation criteria

Criteria 1–3 are reachable with no hardware and no static shell, and are
the acceptance bar for phases 3–6. Criteria 4–5 are gated on §10 Q7.

1. `./ci/blackbox.sh --driver=aved --target=avedsim --app=demo` passes in
   Verilator.
2. `--driver=aved` and `--driver=xrt` produce identical retired-instruction
   counts for the same app and config — the AFU surface is platform-neutral,
   so any divergence is a backend bug.
3. `make -C tests/regression run-aved` matches `run-rtlsim` results.
4. `--target=sim` runs `demo` from a linked vbin — proves IP-XACT packaging
   and `system_map.xml` metadata, independent of hardware.
5. *(gated on Q7)* On hardware: `demo` and `sgemm` pass on the V80 with
   results matching simx.
6. *(gated on Q7)* Timing closure at the `KERNEL_FREQ` target, reported in
   the phase-7 writeup.

Note that criterion 2 is a *parity* check across backends, distinct from
the `model_parity` CI gate ([AGENTS.md §4](../../AGENTS.md)), which
compares simx against rtlsim. This proposal does not change SimX timing
behavior and so should not move any `model_parity` case.

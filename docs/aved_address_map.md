# Proposal: fix the AVED device-memory address map

Status: reviewed, approved for implementation

Review notes (adversarial pass over §3, the one change with cross-platform
reach): `hw/rtl/cp/` contains no base/offset register — `grep -n 'OFFSET|MEM_BASE|dev_base' hw/rtl/cp/*.sv`
is empty — so the CP has no way to receive an absolute device base and must be
driven with the raw addresses from the command stream. Removing the subtraction
therefore cannot regress XRT: on `xilinx_u250` and `xilinx_vck5000` the CP is
today driving `A - OFF` into the arbiter, which re-offsets to `A` and misses
the aperture by exactly the same amount. The change fixes those platforms in
the same stroke. Arbiter wiring re-verified: `.m_awaddr (m_axi_mem_awaddr_u[0])`
at `VX_afu_wrap.sv:586`, re-offset at `:375`.

Scope: `hw/syn/xilinx/aved/platforms.mk`, `hw/rtl/afu/common/VX_afu_wrap.sv`,
`sw/runtime/aved/vortex.cpp`
Affects: `TARGET=sim` **and** `TARGET=hw`

---

## 1. Summary

The AVED backend leaves `PLATFORM_MEMORY_OFFSET` at its default of `0`, but on
the V80 the AFU's device-memory port is **not** mapped at address 0. Both the
simulation and the hardware builds place it at `0x40_0000_0000`. Vortex's
device allocator hands out addresses based at `VX_MEM_USER_BASE_ADDR = 0x10000`,
so every device access — from the Vortex cores *and* from the Command
Processor's DMA engine — targets an address that decodes to nothing.

This single defect explains both open failures:

- **`TARGET=sim` hangs at launch.** Writes to an unmapped address still get a
  `BRESP` from the SmartConnect, so all uploads retire; the kernel binary is
  discarded. The launch then fetches instructions from `0x10000`, gets nothing,
  and never completes.
- **`TARGET=hw` wedges the board.** Same transactions, same unmapped address,
  but on real silicon an unrouteable AXI transaction stalls the master. The CP
  never retires, the AFU never goes idle, and the shell's AXI path eventually
  locks up — which is exactly the state JTAG recovery has been clearing.

The two are the same bug. The sim work was on the critical path after all.

---

## 2. Evidence

### 2.1 Simulation address map

`build/hw/syn/xilinx/aved/build32_aved_sim/bin/vortex_afu.vbin.prj/run_pre.tcl:117-118`

```tcl
assign_bd_address -offset 0x4000000000  -range 128M [get_bd_addr_segs /bram_ctrl/S_AXI/Mem0]     -force
assign_bd_address -offset 0x60000000000 -range 128M [get_bd_addr_segs /bram_ctrl_ddr/S_AXI/Mem0] -force
```

and the Vivado log for that project confirms the segments land in the AFU's
master address spaces:

```
Slave segment '/bram_ctrl/S_AXI/Mem0'     ... address space '/vortex_afu_0/m_axi_mem_0' at <0x40_0000_0000  [ 128M ]>
Slave segment '/bram_ctrl_ddr/S_AXI/Mem0' ... address space '/vortex_afu_0/m_axi_mem_0' at <0x600_0000_0000 [ 128M ]>
```

There is **no segment at 0x10000.**

### 2.2 Hardware address map

`build/hw/syn/xilinx/aved/build32_aved_hw/bin/vortex_afu.vbin.prj/logs/slash_project_build.log:973-974`

```
Slave segment '/QDMA_SLAVE_BRIDGE_0/Reg' ... address space '/vortex_afu_0/m_axi_host'  at <0x208_0000_0000 [ 32G ]>
Slave segment '/HBM_AXI_00/Reg'          ... address space '/vortex_afu_0/m_axi_mem_0' at <0x40_0000_0000  [  1G ]>
```

Hardware puts HBM0 at the **same** `0x40_0000_0000` as simulation. Again no
segment at `0x10000`.

### 2.3 The offset is zero

`hw/rtl/afu/common/vortex_afu.vh:21-22`

```systemverilog
`ifndef PLATFORM_MEMORY_OFFSET
`define PLATFORM_MEMORY_OFFSET 0
```

`hw/syn/xilinx/aved/platforms.mk` never overrides it. By contrast the XRT
platform file sets it for every shell whose memory is not based at 0:

```make
CONFIGS += -DPLATFORM_MEMORY_OFFSET=40'hC000000000    # vck5000
CONFIGS += -DPLATFORM_MEMORY_OFFSET=40\'h4000000000   # u250
```

### 2.4 Why raising the allocator base instead is not an option

`VX_types.toml:20` — for a 32-bit build (`build32_aved_*` is XLEN=32) the whole
device memory map is 32-bit:

```toml
VX_MEM_USER_BASE_ADDR  = 0x00010000
VX_MEM_STACK_BASE_ADDR = 0xFFFF0000
```

A 32-bit core cannot emit `0x40_0000_0000`. Relocating `ALLOC_BASE_ADDR` — the
fix considered before this investigation — is therefore impossible without
moving to XLEN=64. `PLATFORM_MEMORY_OFFSET` exists precisely to bridge this
gap, and the AFU port is wide enough to carry it: `C_M_AXI_MEM_ADDR_WIDTH = 64`
(`VX_afu_wrap.sv:44`), with the narrow 34-bit internal address zero-extended
before the offset is added (`VX_afu_wrap.sv:375-376`).

### 2.5 A prior attempt failed for an unrelated reason

An earlier attempt passed the offset as a **decimal** constant. Vivado silently
discarded it:

```
WARNING: [VRFC 10-8884] decimal constant 274877906944 should be smaller than
2147483648; using 0 instead
```

The offset must be written as a sized Verilog literal — `40'h4000000000` — as
the XRT platform file already does. This is why the earlier experiment appeared
to disprove the theory: the define never reached the RTL.

---

## 3. The second defect: the CP path cancels the offset

`VX_afu_wrap.sv:520-524` subtracts the offset from the Command Processor's
device-memory address before handing it to the bank-0 arbiter:

```systemverilog
// Drop the platform offset from the CP address so the arbiter's slave
// port sees an offset-relative bank-0 address (matches vx_awaddr_a[0]).
wire [M_AXI_MEM_ADDR_WIDTH-1:0] cp_awaddr_offset =
    M_AXI_MEM_ADDR_WIDTH'(cp_axi_dev.awaddr - `PLATFORM_MEMORY_OFFSET);
```

The arbiter's output is then re-offset at the port (`:375`). Net effect for a
nonzero offset:

```
Vortex core:  vx_awaddr = A          --> arbiter --> A + OFF     (mapped)
CP DMA:       cp_awaddr = A, -OFF    --> arbiter --> A           (unmapped)
```

The subtraction presumes the CP is handed *absolute* device addresses while the
cores use *relative* ones. That is not the case. The CP's device addresses come
from `vx_buffer_address()`, i.e. from `Device::global_mem_`
(`sw/runtime/common/device.cpp:54`) — the very same allocator whose addresses
are written into kernel arguments and dereferenced by the cores as 32-bit
pointers. One namespace, two consumers.

The subtraction is a latent bug. It is invisible today only because
`PLATFORM_MEMORY_OFFSET` is 0 on every platform that has run the CP.

Note the CP's *host* master (`m_axi_host`) is correctly left un-offset
(`VX_afu_wrap.sv:312`): host bus addresses are a different namespace and must
pass through untouched.

---

## 4. The third defect: sim host buffers collide with device memory

`sw/runtime/aved/vortex.cpp:303-304` draws CP-visible host memory from VRT's
simulated **HBM** window:

```cpp
uint64_t addr = vrt::detail::reserveFakePhysAddr(asize, vrt::MemoryRangeType::HBM);
```

`SLASH/vrt/include/vrt/buffer.hpp:42-52` bases that window at `0x4000000000`
and bump-allocates upward — the same 128 MB `bram_ctrl` region that
`m_axi_mem_0` will target once the offset is applied. Device buffers would
start at `0x40_0000_0000 + 0x10000` while the CP ring, completion record and
DMA staging buffers grow up from `0x40_0000_0000`. A kernel binary staged
through a buffer larger than 64 KB overruns straight into device memory.

Today this is harmless because device memory is unmapped. Applying the offset
turns it into silent corruption, so it must be fixed in the same change.

The simulated design provides a second, independent 128 MB region at
`0x600_0000_0000` (`bram_ctrl_ddr`), which is assigned into `m_axi_host`'s
address space and is *not* the target of `m_axi_mem_0`. Drawing host buffers
from `MemoryRangeType::DDR` puts them there. This also mirrors the hardware
topology, where `m_axi_host` and `m_axi_mem_0` genuinely target different
slaves (QDMA slave bridge vs HBM0).

---

## 5. Proposed changes

### 5.1 `hw/syn/xilinx/aved/platforms.mk`

```make
# The V80 maps the AFU's device-memory port (m_axi_mem_0 -> HBM0) at
# 0x40_0000_0000 in both the simulation and hardware builds. Vortex's device
# allocator is based at VX_MEM_USER_BASE_ADDR (0x10000) and a 32-bit core
# cannot emit the aperture base itself, so the AFU wrapper rebases every
# device access by this synthesis-time offset.
#
# Sized Verilog literal, not decimal: a decimal constant this large is
# silently truncated to 0 (VRFC 10-8884).
override CONFIGS += -DPLATFORM_MEMORY_OFFSET=40\'h4000000000
```

### 5.2 Quoting — investigated, no change needed

The sized literal contains a `'`, and both Makefiles wrap `$(CONFIGS)` in
single quotes when invoking `gen_config.py`:

```make
XCONFIGS := $(shell python3 .../gen_config.py --config=... --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
```

Fed the define, that line dies with `/bin/sh: Syntax error: Unterminated
quoted string` — reproduced in an isolated Makefile. It does **not** fire in
the real build: `XCONFIGS` is a simply-expanded assignment at
`aved/Makefile:104` (`xrt/Makefile:99`) while `platforms.mk` is included at
`:126` (`:134`), so `CONFIGS` does not yet hold the offset when that shell runs.
Every consumer that *does* see the define — `gen_sources.sh -P $(CFLAGS)` and
the `verilog_define` path — expands it unquoted, where the escaped quote
survives intact (verified: `printf '%s\n' $(CONFIGS)` emits
`-DPLATFORM_MEMORY_OFFSET=40'h4000000000`).

No change is made here. The ordering dependency is worth recording, though:
moving `include platforms.mk` above the `XCONFIGS` line would break the build
with an opaque shell error rather than anything pointing at the offset.

### 5.3 `hw/rtl/afu/common/VX_afu_wrap.sv`

Pass the CP's device address through unmodified so it shares the single
offset applied at the bank port:

```systemverilog
// The CP's device addresses come from the same host-side allocator as the
// pointers handed to the cores, so they are offset-relative already. Feed
// them to the arbiter unchanged; PLATFORM_MEMORY_OFFSET is applied once, at
// the bank port, for both masters.
wire [M_AXI_MEM_ADDR_WIDTH-1:0] cp_awaddr_offset =
    M_AXI_MEM_ADDR_WIDTH'(cp_axi_dev.awaddr);
```

(and likewise for `cp_araddr_offset`).

### 5.4 `sw/runtime/aved/vortex.cpp`

Draw simulated host buffers from the DDR window so they cannot collide with
device memory:

```cpp
uint64_t addr = vrt::detail::reserveFakePhysAddr(asize, vrt::MemoryRangeType::DDR);
```

with the surrounding comment updated — the current one claims sharing the
window is what prevents overlap, which is now the opposite of the truth.

---

## 6. Risks and limits

**Requires a bitstream rebuild.** 5.1 and 5.2 are synthesis-time. The sim vbin
rebuilds in minutes (xsim elaboration only); the hardware vbin is a full
place-and-route. Sequence sim first so the long build is only spent on a
validated fix.

**Only the first 128 MB / 1 GB of device memory is backed.** The AFU's internal
address width is 34 bits (16 GB) and Vortex's `GLOBAL_MEM_SIZE` is 4 GB, but the
mapped aperture is 128 MB in sim and 1 GB on hardware. Allocations past the
aperture will fail the same silent way this bug did. Out of scope here, but it
should become a runtime bound on `global_mem_` — filed as a follow-up rather
than fixed in this change, so that the fix under test stays minimal.

**Unverified: the hardware host-address translation.** On hardware the CP's
host master targets the QDMA slave bridge at `0x208_0000_0000`, while
`allocHostBuffer` returns a bus address from `dma_map_*`. Whether the bridge's
translation registers make those coincide has not been confirmed. This is a
distinct question on the `m_axi_host` path and does not affect the device-memory
fix, but it may be a second hardware blocker; it can only be settled on the
board.

**No shared runtime code is touched.** All three changes are confined to the
AVED platform file, the AFU wrapper's CP path, and the AVED backend.
`sw/runtime/common/` and `callbacks.h` are untouched.

---

## 7. Test plan

1. Rebuild the sim vbin with 5.1 + 5.2.
2. `TARGET=sim` `minimal` — expect `PASSED`, and with `VORTEX_AVED_TRACE=1` a
   readback of the magic pattern rather than `0xDEADBEEF`.
3. `TARGET=sim` `vecadd`, then `sgemm`.
4. Code-review the three files.
5. Commit.
6. Rebuild the hardware vbin, reload the board, run `sgemm` on the V80.

The pass/fail signal at step 2 is unambiguous: `0xDEADBEEF` in the output buffer
means the kernel never ran, anything else means the launch reached memory.

---

## 8. Results

**The device-memory fix is verified on real RTL.** `minimal -n4 -l` (loopback:
write a pattern to device memory, read it straight back, no launch) **PASSES**
on `TARGET=sim`. The readback returns `0x5A5A0000..0x5A5A0003` rather than the
`0xDEADBEEF` the destination was primed with, so the bytes genuinely made the
round trip through the simulated fabric at `0x40_0001_0000`. Before this change
the same path wrote into an unmapped hole.

Supporting evidence from the same run:

- Host buffers now sit at `0x600_0000_0000` (the DDR model), device memory at
  `0x40_0001_0000` — no overlap. The pre-fix trace shows exactly the collision
  §4 predicted: a 64 KB ring at `0x40_0000_0000` with the next host allocation
  at `0x40_0001_0000`, precisely where device buffers would have landed.
- The define reaches the RTL intact: `sources.txt` carries
  `PLATFORM_MEMORY_OFFSET=40'h4000000000` and no `VRFC 10-8884` truncation
  warning appears anywhere in the build.

**Hardware bitstream rebuilt** with the same two synthesis-time changes:
0 errors, timing met (`Post Routing Timing Summary | WNS=0.016 | TNS=0.000`).

**Still open: the full launch does not complete under `TARGET=sim`.** With the
kernel enabled the run reaches `launch` → `readback` → `wait` and then sits
there; two runs were cut off at 25 and 40 minutes. The CP is demonstrably
healthy throughout — 26 commands retired, the console-drain loop polling and
completing normally — so this is downstream of the command path that this
change fixes. Whether it is simply xsim being slow or a second defect is not
yet established, and the loopback result deliberately does not speak to it.
That is the next investigation, tracked separately from this fix.

---

# Follow-up: the aperture is too small for Vortex's stack

Found after the offset fix landed; a distinct defect with the same symptom.

## Evidence

With the offset applied, `minimal -l` (loopback) passes but a real launch never
retires. A cycle-count heartbeat added to the sim harness
(`VRT_SIM_HEARTBEAT=30`) measured the simulator at a steady **~2,070 cycles/s**,
so the 40-minute run covered **~5.0M cycles**. avedsim runs the far heavier
*sgemm* in 795,458 cycles. This is a hang, not a slow simulator — and the clock
keeps advancing throughout, which is the signature of a memory access that never
receives a response rather than a stuck core.

The address that cannot be answered is the **stack**:

- `sw/kernel/src/vx_start.S:95` sets `sp = VX_MEM_STACK_BASE_ADDR - (hartid << VX_MEM_STACK_LOG2_SIZE)`,
  i.e. `0xFFFF0000`, and the stack grows **down**.
- `hw/rtl/core/VX_lsu_slice.sv:108-110` decodes local memory **upward** from the
  same base: `[0xFFFF0000, 0xFFFF0000 + (1 << LMEM_LOG_SIZE))` = `[0xFFFF0000, 0xFFFF4000)`.

So the very first stack push lands at `0xFFFEFxxx` — just *below* the LMEM
window, hence global memory, hence `0x40_FFFE_Fxxx` once rebased.

Neither aperture reaches it:

| Target | Mapped aperture | Stack address |
|---|---|---|
| sim BRAM   | `0x40_0000_0000` + 128 MB | `0x40_FFFE_F000` |
| hw HBM_AXI_00 | `0x40_0000_0000` + 1 GB | `0x40_FFFE_F000` |

Vortex's 32-bit map places the stack at the top of a 4 GB space, so a platform
must map the full 4 GB. `xilinx_u250` does — 16 GB of DDR at `0x40_0000_0000`
— which is why this never surfaced there. The V80 build asks for `HBM0`, a
single 1 GB channel.

**This is almost certainly what wedges the board.** An unanswered AXI read
stalls the LSU indefinitely; the AFU never goes idle and the shell's AXI path
locks up behind it.

## Fix

`hw/syn/xilinx/aved/Makefile` defaults `MEM_TAG ?= HBM0`. The linker also
offers a `MEM` tag routed through the HBM VNOC
(`slashkit/core/bd_ports.py:260-263`), and the compute shell maps it at:

```tcl
assign_bd_address -offset 0x004000000000 -range 0x000800000000 ... HBM_VNOC_INI_00/Reg
```

— the **same base**, with **32 GB** of range instead of 1 GB. So `MEM_TAG=MEM`
covers the whole 32-bit space and `PLATFORM_MEMORY_OFFSET=40'h4000000000` stays
exactly as committed. One variable, no RTL change.

## Limits

**Simulation cannot be fixed the same way.** The sim BRAM model is 128 MB and
Vivado's AXI BRAM controller caps at 2 GB, still short of the 4 GB needed. So
`TARGET=sim` will keep hanging on any kernel that touches its stack, which is
all of them. Loopback (`minimal -l`) remains the useful sim-side check, since it
exercises the CP DMA path without running code. Making sim run real kernels
would need Vortex's stack relocated into a smaller window — a change to the
HW/SW memory contract in `VX_types.toml`, out of scope here and not required
for hardware.

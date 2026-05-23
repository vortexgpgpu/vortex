# Virtual Memory — Software-Stack Redesign

**Status:** Proposal (May 2026). Supersedes the host-side `page_table_walk`
model and the per-backend VM stubs. Companion to
[`cp_pure_v2_callbacks_proposal.md`](cp_pure_v2_callbacks_proposal.md)
(the 6-function transport HAL) and
[`command_processor_proposal.md`](command_processor_proposal.md).

**Scope:** redesign the VM software stack around the Command Processor so
virtual memory is correct, efficient, and aligned with how modern NVIDIA /
AMD GPUs work — and so it survives the `callbacks_t` collapse that moved
`mem_alloc` + DMA into the common runtime core.

---

## 1. Motivation — why the old VM model is wrong

Vortex VM today: the host runtime builds page tables in device memory, and
`upload` / `download` did a **software `page_table_walk` on the host** to
turn a VA into a PA before each DMA.

That model is wrong three ways:

1. **It is not how real GPUs work.** On AMD (GPUVM + SDMA) and NVIDIA
   (GMMU + Copy Engines), the copy/DMA engines are MMU-aware; the driver
   builds page tables and *never* translates addresses on the engines'
   behalf.
2. **It does not fit the CP-sole-DMA architecture.** After the
   `callbacks_t` collapse the Command Processor owns every transfer; the
   host has no DMA path of its own and cannot call back into a software
   walker mid-transfer.
3. **It put VM in the backends.** VM translation lived inside each
   backend's `mem_alloc` / `upload` / `download`. Those moved into the
   common core, so VM came along only as dead `#ifdef` stubs.

The `callbacks_t` collapse already removed the host-side walks; VM must be
rebuilt on the correct model rather than patched back.

## 2. Principle — the copy engine is MMU-aware

On a modern GPU:

- the **driver builds page tables** and does not walk them at run time;
- **every hardware engine — compute units *and* copy/DMA engines —
  translates in hardware** through those same page tables;
- software works purely in **virtual addresses**; there is one address
  space and the hardware translates.

Vortex's CP already has a DMA engine (`VX_cp_dma`) — it *is* Vortex's copy
engine. The redesign makes it **MMU-aware**, exactly like AMD SDMA or an
NVIDIA Copy Engine. The host runtime becomes translation-free.

## 3. Architecture

```
  host driver (vx::Device + VMManager)      the CP / its DMA engine
  ──────────────────────────────────       ───────────────────────────
  mem_alloc:  PA <- global_mem_             CMD_MEM_* device operand
              VA <- phy_to_virt_map(PA)       = a VIRTUAL address
  builds PTEs in a host shadow              VX_cp_dma walks the page table
  flushes dirty PT pages via                  (HW walker + small TLB) -> PA,
  CMD_MEM_WRITE(physical)                      then bursts AXI
  runtime API is VA-only — no walks         shares the kernel's PTs + SATP
```

### 3.1 Host driver = the page-table builder

`vx::Device` owns the `VMManager` (the page-table builder + VA allocator).
`mem_alloc` allocates a PA from `global_mem_`, installs PTEs, and returns a
**VA**; `mem_free` unmaps. The runtime API is VA-only — this is the
driver's job and *only* the driver's job, like `amdgpu`'s GPUVM manager.

Page tables are host-shadowed (already the `VMManager` design): all
`read_pte` / `write_pte` hit the host shadow; modified PT pages are
flushed to device memory in **one bulk transfer per dirty PT page** —
`CMD_MEM_WRITE` with the `physical` flag. This matches the batched
PT-update pattern of CUDA / ROCm / Level Zero drivers.

### 3.2 CP DMA = MMU-aware copy engine

The device-side operand of every `CMD_MEM_*` is a **virtual address**.
`VX_cp_dma` translates it via a hardware page-table walker with a small
TLB, using the kernel's page tables. Translation is per-command, at the
transfer base (a buffer is one contiguous PA allocation, so one base
translate covers the whole transfer); the TLB amortizes walks across the
4 KB burst chunks `VX_cp_dma` already issues.

In the simulators the software `CommandProcessor` model performs the same
walk in its device-memory path — this models *the CP* walking, mirroring
hardware, not a host-side shortcut.

### 3.3 Runtime VM discovery — query the device, never `#ifdef`

`libvortex.so` is the generic dispatcher: one binary, any backend,
`dlopen`'d per `$VORTEX_DRIVER`. It must **not** be tied to a device
config by `#ifdef VX_CFG_VM_ENABLE`, and it must not depend on the
per-config generated `VX_config.h`.

VM-enabled is a **device property discovered at `vx_device_open`**, exactly
as a real GPU driver queries "does this device have an MMU?":

- a `VM_ENABLED` bit in the CP `DEV_CAPS` register (`0x008`, currently
  `{AXI_TID_W | RING_LOG2 | NUM_QUEUES}` — free bits remain), published by
  the RTL `VX_cp_axil_regfile` and the `cmd_processor` model from their
  build config;
- the runtime reads it once via `cp_reg_read` and stores `vm_enabled_`.

Consequence: `vm.h` / `vm_types.h` / `vm.cpp` **drop the
`#ifdef VX_CFG_VM_ENABLE` guard** — `VMManager` is always compiled (inert
in BARE mode). `vx::Device` uses a runtime `if (vm_enabled_)`, never a
preprocessor `#ifdef`. The common runtime stops including `VX_config.h`
entirely.

### 3.4 The physical / identity (VA == PA) allocation class

Every real GPU driver has a **physically-contiguous, pinned, non-paged**
allocation class beside ordinary virtual allocations — for hardware that
has no MMU or paths that must not fault. Canonical cases: the display
scanout engine; the command-ring fetch; MMU-less fixed-function engines.

Vortex uses it for:

- the **command ring** — a trusted, can't-fault fetch path (in Track 1 the
  ring is host memory and outside device VM entirely; a device-resident
  ring would use this class);
- **graphics fixed-function buffers** — `RASTER` / `OM` / `TEX`. These
  fixed-point gfx-v1 units carry no MMU/TLB, so their buffers must be
  identity-mapped (the address the unit issues *is* the physical address)
  and physically contiguous (the unit cannot gather scattered pages).

`VX_MEM_PHYS` is that class: `mem_alloc(VX_MEM_PHYS)` →
`global_mem_.allocate()` (contiguous PA) + `install_identity_map`
(VA == PA). The kernel (which has an MMU) and the MMU-less fixed-function
unit then reach the same bytes at the same address.

Correctness rule: an identity (VA == PA) allocation is reserved in **both**
the PA allocator and the VA allocator (`virtual_mem_`), so a later
`phy_to_virt_map`-minted VA can never collide with an identity range.

> Note: a top-tier modern GPU instead gives texture units and ROPs their
> own MMUs (textures / render targets are virtual there). Vortex's MMU-less
> fixed-function units are a legitimate simplification; the identity
> allocation class is the correct way to serve that choice.

### 3.5 SATP plumbing

The host programs the page-table root + mode into a CP regfile register
pair, `CP_SATP_LO/HI` (CP-internal offsets `0x028` / `0x02C`), at VM init,
so the CP DMA's walker can find the page table. The compute cores continue
to receive SATP via the kernel's boot `csrw` — unchanged.

### 3.6 Address-space map

| Region | In device VA space? | CP DMA access |
|---|---|---|
| Command ring / DMA staging | no — host memory (`m_axi_host`) | untranslated |
| Page-table region | physical | `CMD_MEM_*(physical)` |
| IO / COUT `[0, USER_BASE)` | identity-mapped (by `VMManager::init`) | translate → self |
| Kernel image / reserved regions | identity-mapped (`mem_reserve`) | translate → self |
| Graphics buffers (`VX_MEM_PHYS`) | identity-mapped | translate → self |
| User buffers (`mem_alloc`) | virtual | translate |

## 4. Wire-protocol additions

| Addition | RTL (`VX_cp_*`) | `cmd_processor` model | runtime |
|---|---|---|---|
| `CP_SATP_LO/HI` regs `0x028/0x02C` | `VX_cp_axil_regfile` | `mmio_write/read` | `cp_reg_write` at init |
| `CMD_MEM_*` `physical` flag (`flags` bit2, `F_MEM_PHYSICAL` — a dedicated bit, distinct from `F_PROFILE`) | `VX_cp_dma` decoder | `cp_translate` skip | `cp_submit_mem_(…, physical)` |
| `DEV_CAPS.VM_ENABLED` bit | `VX_cp_axil_regfile` | `mmio_read 0x008` | `cp_reg_read` at `vx_device_open` |

The protocol is identical for the simulator model and the RTL — Phase 2's
`VX_cp_dma` hardware walker consumes exactly what Phase 1's model does.

## 5. Phasing

- **Phase 1 — simulators, no RTL.** The `CommandProcessor` model becomes
  MMU-aware (a software walker in its device-memory path); add the
  `physical` flag, `CP_SATP`, and `DEV_CAPS.VM_ENABLED` to the model.
  `vx::Device` owns the `VMManager`. → **done and validated on simx**
  (Sv39 and Sv32) **and gem5** (§7). Validates the whole runtime +
  protocol end to end.
- **Phase 2 — RTL.** *(deferred past the v3 release.)* The v3 ship target
  is SimX-only VM. The Phase 2 design has also shifted away from the
  per-engine walker sketched in §3.2: a real GPU exposes one shared
  device-side MMU (NVIDIA's GMMU/UTCL2 hub, AMD's UTCL2) that every DMA /
  compute / display engine routes through, not a private walker per
  copy engine. When Phase 2 lands we expect a single CP-side MMU shared
  by `VX_cp_dma` (and any future engines that walk device addresses), a
  small TLB in front of it, and the `CMD_MEM_*` decoder honoring the
  `physical` flag — and **rtlsim belongs here**, not Phase 1, because
  rtlsim runs the RTL core MMU. The CI `vm()` job is SimX-only until
  Phase 2 is brought up; the RTL paths (xrtsim / xrt / opae) continue to
  run their non-VM regressions unchanged.

## 6. Implementation plan (file-by-file)

**`sim/common/cmd_processor.{h,cpp}` — MMU-aware CP model**
- `#include <vm_types.h>`; `cp_translate(vaddr, physical)` — Sv32/39 walk;
  `CP_SATP_LO/HI` regs; `MEM_FLAG_PHYSICAL`; `DEV_CAPS.VM_ENABLED` bit.
- `CMD_MEM_*` handler translates the device operand (write→`arg0`,
  read→`arg1`, copy→both) unless `physical`.
- *(implemented; see §7.)*

**`sw/common/vm_types.h`, `sw/runtime/common/vm.{h,cpp}` — un-gate VM**
- Remove the `#ifdef VX_CFG_VM_ENABLE` guard — `VMManager` is always
  compiled (inert when the device has no MMU). These files no longer
  include HW-private `VX_config.h`; the Sv32/Sv39 split comes from
  `VX_VM_ADDR_MODE` in the SW-facing `VX_types.h`.
- Add `uint64_t VMManager::satp() const`.

**`sw/runtime/common/{vortex2_internal.h,device.cpp}` — host driver**
- `vx::Device` owns `VMManager` + a `CpMemIO` (`DeviceMemIO` over
  `cp_submit_mem_*` with `physical=true`) + a `bool vm_enabled_`.
- At `vx_device_open`: read `DEV_CAPS.VM_ENABLED` → `vm_enabled_`.
- `cp_init`: `if (vm_enabled_)` → `VMManager::init()`, program `CP_SATP`.
- `mem_alloc`: `vm_enabled_` → `phy_to_virt_map`→VA (or
  `install_identity_map` for `VX_MEM_PHYS`). `mem_free`/`mem_reserve`
  VM-aware. `cp_submit_mem_` carries the `physical` flag.
- All VM branches are runtime `if (vm_enabled_)`, **never `#ifdef`**.

**`sw/runtime/{simx,rtlsim}/vortex.cpp` — already done:** dead VM stubs
removed; VM is wholly common-core.

**Build / validate:** the simulators are built with `VX_CFG_VM_ENABLE`
(device-side); the runtime stays config-agnostic and discovers VM at
open. Validate with sgemm + tex on simx / rtlsim, then xrt / opae.

## 7. Implementation status

Phase 1 is **complete and validated** — on simx for both Sv39 and Sv32
(vecadd, sgemm, tex all pass: XLEN=64 `build_vm`/Sv39 and XLEN=32
`build_vm32`/Sv32), and on **gem5** (vecadd and sgemm pass end-to-end
through the gem5 co-simulation; device lib built with
`make -C sim/simx USE_GEM5=1`). The stack runs end to end: `VMManager`
init, host-shadow page-table build + batched flush, the MMU-aware
CP-DMA, and a `VX_VM_PT_LEVEL`-deep hardware-style walk in the core MMU
(3 levels for Sv39, 2 for Sv32). The runtime is fully `#ifdef`-free for
VM — the legacy MPM perf dump now gates its TLB/PTW counters on a
runtime `VX_CAPS_VM_SUPPORT` query.

Bring-up corrected several defects and finished the structural work:

- **Megapage/gigapage offset reconstruction.** `cp_translate` and
  `VMManager::page_table_walk` returned `(leaf_ppn<<12) + pgoff`, dropping
  VA bits `12..(12+L*VPN_BITS-1)` for an L1/L2 superpage leaf. Fixed in
  both: `pa = ((leaf_ppn<<12) & ~mask) | (va & mask)`,
  `mask = (1 << (12 + L*VPN_BITS)) - 1`.
- **`update_page_table` followed a superpage leaf as an interior node.**
  Re-identity-mapping a sub-range already covered by a superpage walked
  into mapped data. Fixed: an existing leaf makes the re-map idempotent
  when it already yields the requested translation, a conflict otherwise.
- **`phy_to_virt_map` page-count truncation.** `num_pages = size >> 12`
  truncated to 0 for any sub-4 KB buffer, leaving it unmapped. Fixed with
  a ceil-divide.
- **Core MMU — Sv39.** The simx core MMU (`sim/simx/mem/mmu.cpp`) was a
  two-level (Sv32) walker; generalized to a `VX_VM_PT_LEVEL`-deep
  level-counter FSM (`PTW_REQ` / `PTW_WAIT` / `PTW_FILL`), and an 8-byte
  Sv39 PTE read that used a 4-byte load was fixed.
- **PA-allocator coordination.** `cp_init` reserves
  `[VX_MEM_PAGE_TABLE_BASE_ADDR, +VX_VM_PT_SIZE_LIMIT)` out of
  `global_mem_`, so `mem_alloc` can never hand back a PA overlapping the
  page tables (the two allocators previously overlapped unguarded).
- **Config-agnostic dispatcher (§3.3) — implemented.** VM is discovered
  at runtime from `CP DEV_CAPS.VM_ENABLED` (bit 24), read once at
  `vx_device_open`. The `#ifdef VX_CFG_VM_ENABLE` guards are gone from the
  runtime — `VMManager` is always compiled into `libvortex.so` and
  `vx::Device` branches on a runtime `vm_enabled_`. `vm.h` / `vm.cpp` /
  `vm_types.h` no longer include HW-private `VX_config.h`; the Sv32/Sv39
  split comes from `VX_VM_ADDR_MODE` in the SW-facing `VX_types.h`.
  Verified by preprocessor dependency trace: no file under `sw/` includes
  `VX_config.h`. This retires the build skew that had left `libvortex.so`
  unlinkable and `libsimx.so` unloadable; the per-build
  `CONFIGS += -DVX_CFG_VM_ENABLE` now configures only the sim, which is
  config-specific by design.

## 8. Why this is correct and efficient

- **Correct:** one page table, hardware-walked by every engine; the host
  is translation-free — the AMD SDMA / NVIDIA CE model verbatim.
- **Config-agnostic dispatcher:** `libvortex.so` discovers VM from the
  device; no `#ifdef`, no `VX_config.h` dependency in the runtime.
- **No `callbacks_t` impact:** VM lives entirely in the common core (PT
  builder) and the CP (translation); the 6-function transport HAL is
  untouched — a backend never sees VM.
- **Efficient:** host-shadow page tables + batched flush (one DMA per
  dirty PT page, not per PTE); the CP-DMA TLB amortizes walks across 4 KB
  bursts; megapage PTEs cut walk depth and TLB pressure; VA-only host API
  → zero host-side translation cost and no host/HW page-table race.

## 9. Remaining work

- **`configure --vm` flag.** Sv32 and Sv39 VM builds (`build_vm32` /
  `build_vm`) currently force VM on per-build via `CONFIGS +=
  -DVX_CFG_VM_ENABLE` in `config.mk` plus a VM-on `VX_config.h`. A
  first-class `configure --vm` option (a small `gen_config.py` +
  `configure` change) would make that automatic and durable. Validation
  itself is done — both ISAs pass (§7).
- **Phase 2 — RTL.** *(deferred past v3 release.)* The v3 release ships
  VM on SimX only — `ci/regression.sh` `vm()` runs `simx` for both
  default-mode and BARE-mode passes, no `xrtsim` / `xrt` / `rtlsim`.
  Phase 2 is also re-scoped from the per-engine walker in §3.2 to a
  single shared CP-side MMU (NVIDIA GMMU / AMD UTCL2 model) that every
  device-address engine routes through, with a small TLB in front. When
  Phase 2 lands, `VX_cp_axil_regfile` gets `CP_SATP` and
  `DEV_CAPS.VM_ENABLED`, the `CMD_MEM_*` decoder honors the `physical`
  flag, and validation extends to rtlsim and xrt / opae.
- **Residual compile-time config in the runtime (non-VM).**
  `device.cpp::query_caps` still reads `VX_CFG_PLATFORM_CLOCK_RATE` /
  `VX_CFG_PLATFORM_MEMORY_PEAK_BW` from the `XCONFIGS` `-D` projection (it
  does *not* include `VX_config.h`). Per the §5 layering these are runtime
  device properties and belong behind `vx_device_query()` — caps work,
  independent of VM.

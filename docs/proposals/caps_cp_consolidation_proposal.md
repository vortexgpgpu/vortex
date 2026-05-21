**Date:** 2026-05-20
**Status:** Draft — not yet approved
**Author:** Blaise Tine
**Related:**
[command_processor_proposal.md](command_processor_proposal.md),
[cp_rtl_impl_proposal.md](cp_rtl_impl_proposal.md),
[cp_runtime_impl_proposal.md](cp_runtime_impl_proposal.md),
[cp_xrt_integration_plan.md](cp_xrt_integration_plan.md),
[cp_opae_integration_plan.md](cp_opae_integration_plan.md),
[gem5_v2_cp_migration_proposal.md](gem5_v2_cp_migration_proposal.md),
[config_macro_namespace_proposal.md](config_macro_namespace_proposal.md).

# Device / ISA Caps Query Consolidation into the Command Processor — Proposal

## 1. Summary

The device-capability query (`vx_device_query` / legacy `vx_dev_caps`)
is backed today by **five independent implementations** of the same
two values:

| Backend | `dev_caps` / `isa_caps` source | `get_caps` decoder |
|---|---|---|
| XRT | RTL — [VX_afu_ctrl.sv](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) `dev_caps`/`isa_caps` | [sw/runtime/xrt/vortex.cpp](../../sw/runtime/xrt/vortex.cpp) |
| OPAE | RTL — [vortex_afu.sv](../../hw/rtl/afu/opae/vortex_afu.sv) `dev_caps`/`isa_caps` | [sw/runtime/opae/vortex.cpp](../../sw/runtime/opae/vortex.cpp) |
| simx | C++ macros, inline | [sw/runtime/simx/vortex.cpp](../../sw/runtime/simx/vortex.cpp) |
| rtlsim | C++ macros, inline | [sw/runtime/rtlsim/vortex.cpp](../../sw/runtime/rtlsim/vortex.cpp) |
| gem5 | C++ macros, inline | [sw/runtime/gem5/vortex.cpp](../../sw/runtime/gem5/vortex.cpp) |

Both the **producer** (the `{VX_CFG_MISA_EXT, VX_CFG_XLEN,
VX_CFG_MISA_STD}` / device-config bit-packing) and the **consumer**
(`get_caps`, the bit-field decoder) are duplicated. The `isa_caps`
packing is byte-identical in all five copies; the `dev_caps` packing
has *already drifted* between the XRT and OPAE RTL (§3.3). This is the
classic five-copies-of-one-fact maintenance hazard.

The Command Processor gives us the right home for this. The CP is the
single host-control surface every backend now talks to:

- on hardware (XRT, OPAE), the CP regfile
  [VX_cp_axil_regfile.sv](../../hw/rtl/cp/VX_cp_axil_regfile.sv) is the
  only AXI-Lite slave on the CP;
- in simulation (simx, rtlsim, gem5), the C++
  [CommandProcessor](../../sim/common/cmd_processor.h) is its declared
  functional twin ("Address map (matches VX_cp_axil_regfile)").

This proposal moves the device/ISA caps into the CP register map — as
**static, read-only `GPU_DEV_CAPS` / `GPU_ISA_CAPS` registers** in both
the RTL regfile and the C++ model — and collapses the five `get_caps`
implementations into **one shared decoder** in
[sw/runtime/common/](../../sw/runtime/common/). After the change there
is exactly one caps producer per representation (one RTL, one C++) and
one caps consumer.

This is deliberately a small, mechanical change: two new RO register
pairs in two files that are already maintained as twins, plus a
decoder lift into common code. No new mechanism, no DCR round-trip, no
runtime protocol.

---

## 2. Goals and non-goals

### 2.1 Goals

- **One caps producer per representation.** The `dev_caps`/`isa_caps`
  bit-packing exists once in RTL (`VX_cp_axil_regfile.sv`) and once in
  C++ (`CommandProcessor`). The two are already contractually kept in
  sync as functional twins.
- **One caps consumer.** A single `decode_caps()` in
  `sw/runtime/common/` replaces the five per-backend `get_caps`
  bit-field decoders.
- **Uniform runtime path.** `query_caps` reads the CP regfile on every
  backend — `cp_mmio_read` of a fixed offset — whether that regfile is
  RTL (XRT/OPAE) or the C++ model (simx/rtlsim/gem5).
- **Backends become pure glue.** The XRT/OPAE AFU shells stop carrying
  caps-packing logic; they are AXI-Lite ↔ CCI-P plumbing.

### 2.2 Non-goals

- **Not** changing the public API. `vx_device_query`, the `VX_CAPS_*`
  ids in [vortex.h](../../sw/runtime/include/vortex.h), and the legacy
  `vx_dev_caps` shim are untouched. The returned values are
  bit-for-bit identical.
- **Not** making caps dynamic. `dev_caps`/`isa_caps` remain
  synthesis-time / compile-time constants exposed through a static RO
  register. No DCR/CP-command round-trip is introduced.
- **Not** touching the legacy non-CP datapath beyond the scoped
  deletion in Phase 3 (§6.3).
- **Not** addressing `VX_CAPS_CLOCK_RATE` / `VX_CAPS_PEAK_MEM_BW`,
  which are genuinely backend/platform-specific and stay backend-local
  (§5.3).

---

## 3. Current state

### 3.1 The producer — RTL

[VX_afu_ctrl.sv:129-147](../../hw/rtl/afu/xrt/VX_afu_ctrl.sv) and
[vortex_afu.sv:113-131](../../hw/rtl/afu/opae/vortex_afu.sv) each build
two 64-bit words:

```systemverilog
wire [63:0] isa_caps = {
    32'(`VX_CFG_MISA_EXT),
    2'(`CLOG2(`VX_CFG_XLEN)-4),
    30'(`VX_CFG_MISA_STD)
};
```

`isa_caps` is a pure function of build-time defines — identical in
both files. `dev_caps` packs the core/memory configuration; its
layout (MSB-first):

```
[63:42] reserved          [22:20] clog2(NUM_CLUSTERS)
[41:37] BANK_ADDR_W - 20  [19:17] clog2(CLUSTER_SIZE)
[36:34] clog2(NUM_BANKS)  [16:14] clog2(SOCKET_SIZE)
[33:26] LMEM_LOG_SIZE     [13:11] clog2(NUM_WARPS)
[25:23] clog2(ISSUE_W)    [10:8]  clog2(NUM_THREADS)
                          [7:0]   VX_ISA_IMPL_ID
```

### 3.2 The producer — C++

simx, rtlsim and gem5 do not pack a word; they inline the same
constants in their `get_caps` switch. The `isa_caps` line is
byte-identical across all three —
[simx/vortex.cpp:123](../../sw/runtime/simx/vortex.cpp),
[rtlsim/vortex.cpp:119](../../sw/runtime/rtlsim/vortex.cpp),
[gem5/vortex.cpp:82](../../sw/runtime/gem5/vortex.cpp):

```cpp
_value = ((uint64_t(VX_CFG_MISA_EXT))<<32)
       | ((log2floor(VX_CFG_XLEN)-4)<<30)
       |   VX_CFG_MISA_STD;
```

### 3.3 Evidence of drift

The XRT and OPAE `dev_caps` already disagree on the memory-bank
fields:

| Field | XRT (`VX_afu_ctrl.sv`) | OPAE (`vortex_afu.sv`) |
|---|---|---|
| `[41:37]` | `5'(MEMORY_BANK_ADDR_WIDTH-20)` | `5'(LMEM_BYTE_ADDR_WIDTH-20)` |
| `[36:34]` | `3'($clog2(\`VX_CFG_PLATFORM_MEMORY_NUM_BANKS))` | `3'($clog2(NUM_LOCAL_MEM_BANKS))` |

where the XRT shell derives `MEMORY_BANK_ADDR_WIDTH` from
`VX_CFG_PLATFORM_MEMORY_ADDR_WIDTH` / `VX_CFG_PLATFORM_MEMORY_NUM_BANKS`,
while the OPAE shell uses `LMEM_BYTE_ADDR_WIDTH` /
`NUM_LOCAL_MEM_BANKS` (from its `local_mem_cfg_pkg`). These *should*
describe the same platform DRAM, but they reference different
parameter sources per shell. Whether they currently evaluate equal
must be verified; either way it is exactly the divergence a single
source eliminates. **This is the bug class the proposal prevents, not
a bug to fix in passing** — the consolidation must pick one
platform-neutral parameterization (§7, open question OQ-2).

### 3.4 The consumer

`get_caps` (the `dev_caps`-bit-field decoder + `isa_caps` passthrough)
is implemented five times. XRT/OPAE cache the two 64-bit words at
device-open then slice them; simx/rtlsim/gem5 slice compile-time
macros. All five `switch` blocks are structurally identical.

### 3.5 Where the CP already sits

The CP regfile is the established host-control surface:

- [VX_cp_axil_regfile.sv](../../hw/rtl/cp/VX_cp_axil_regfile.sv) — the
  globals region `0x000..0x0FF` currently uses `0x000` `CP_CTRL`,
  `0x004` `CP_STATUS`, `0x008` `CP_DEV_CAPS`, `0x010/0x014`
  `CP_CYCLE_LO/HI`. **Free in-region offsets include
  `0x00C`, `0x018`, `0x01C`, `0x020`, `0x024`, …**
- [cmd_processor.{h,cpp}](../../sim/common/cmd_processor.h) — the C++
  functional twin (class `vortex::CommandProcessor`), with the
  matching `mmio_read`/`mmio_write` decode. simx and rtlsim
  `#include <cmd_processor.h>` and own a `CommandProcessor cp_`
  directly; the gem5 device model in `sim/simx/gem5/` instantiates the
  same class behind its PIO range.
- The XRT runtime already routes host addresses `0x1000+` to the CP
  regfile, with `CP_REG_DEV_CAPS (CP_BASE + 0x008)` already defined in
  [sw/runtime/xrt/vortex.cpp](../../sw/runtime/xrt/vortex.cpp).

> **Naming caution.** The existing `CP_DEV_CAPS` at `0x008` describes
> the **CP fabric** (`{AXI_TID_W, RING_LOG2, NUM_QUEUES}`) — a
> different thing from the **GPU** device caps. The new registers must
> be named `GPU_DEV_CAPS` / `GPU_ISA_CAPS` to avoid conflation.

---

## 4. Design

### 4.1 New CP register map entries

Add two read-only 64-bit values to the CP globals region, each as a
LO/HI 32-bit pair (AXI-Lite is 32-bit, matching the existing
`CP_CYCLE_LO/HI` convention):

| CP offset | Host offset (XRT) | Name | Contents |
|---|---|---|---|
| `0x018` | `0x1018` | `GPU_DEV_CAPS_LO` | `dev_caps[31:0]` |
| `0x01C` | `0x101C` | `GPU_DEV_CAPS_HI` | `dev_caps[63:32]` |
| `0x020` | `0x1020` | `GPU_ISA_CAPS_LO` | `isa_caps[31:0]` |
| `0x024` | `0x1024` | `GPU_ISA_CAPS_HI` | `isa_caps[63:32]` |

(gem5's PIO range *is* the CP regfile, so it sees the bare CP offsets
`0x018..0x024` with no `0x1000` base.)

All four are RO; host writes are ignored (DECERR in RTL, silently
dropped in the C++ model, exactly as `CP_DEV_CAPS`/`CP_STATUS` behave
today). The values are synthesis-time / compile-time constants — no
state, no clocking, no FSM interaction.

### 4.2 RTL — `VX_cp_axil_regfile.sv`

The regfile builds the two words from the GPU config macros and adds
four `read_reg`/`is_decoded` cases. The packing expression is **lifted
verbatim** from `VX_afu_ctrl.sv` (the canonical copy), with the
memory-bank fields resolved per OQ-2:

```systemverilog
// Static GPU caps — synthesis-time constants, exposed RO so the
// host runtime has a single, platform-neutral caps source.
wire [63:0] gpu_dev_caps = { /* §3.1 layout */ };
wire [63:0] gpu_isa_caps = {
    32'(`VX_CFG_MISA_EXT), 2'(`CLOG2(`VX_CFG_XLEN)-4), 30'(`VX_CFG_MISA_STD)
};
```

```systemverilog
// in read_reg():
if (is_global(addr, 8'h18)) return gpu_dev_caps[31:0];
if (is_global(addr, 8'h1C)) return gpu_dev_caps[63:32];
if (is_global(addr, 8'h20)) return gpu_isa_caps[31:0];
if (is_global(addr, 8'h24)) return gpu_isa_caps[63:32];
```

with the four offsets also added to `is_decoded()`. The module-header
register-map comment block is updated.

**Define visibility.** `VX_afu_ctrl.sv` reaches `` `VX_CFG_MISA_EXT ``,
`` `VX_CFG_MISA_STD ``, `` `VX_CFG_XLEN ``, `` `VX_ISA_IMPL_ID `` and
the core/memory-config macros, and imports `VX_gpu_pkg::*`. The CP
regfile currently `` `include``s `VX_define.vh` and imports
`VX_cp_pkg::*`. The `VX_CFG_*` / `VX_ISA_*` identifiers are generated
preprocessor macros (post the `config_macro_namespace_proposal.md`
rename) and should resolve once `VX_define.vh` is in scope; **this
must be confirmed in the CP compilation unit before the layout is
frozen** (it may require adding `import VX_gpu_pkg::*`). See OQ-1.

### 4.3 C++ model — `CommandProcessor` (`sim/common/cmd_processor.*`)

The C++ twin gains the same four offsets in `mmio_read`, computed from
the C++ config macros (`VX_CFG_MISA_EXT`, `VX_CFG_MISA_STD`,
`VX_CFG_XLEN`, `VX_ISA_IMPL_ID`, the core/memory defines) — the same
build-time constants the RTL reads:

```cpp
// cmd_processor.cpp, mmio_read() globals switch:
case 0x018: return uint32_t(gpu_dev_caps() & 0xFFFFFFFF);
case 0x01C: return uint32_t(gpu_dev_caps() >> 32);
case 0x020: return uint32_t(gpu_isa_caps() & 0xFFFFFFFF);
case 0x024: return uint32_t(gpu_isa_caps() >> 32);
```

`gpu_dev_caps()` / `gpu_isa_caps()` are two small `constexpr`/inline
helpers — ideally in a header shared with the runtime decoder (§5.2)
so the packing constants live in exactly one C++ location. `mmio_write`
adds `0x018/0x01C/0x020/0x024` to its RO-ignore list. The header's
address-map comment block is updated to match.

### 4.4 Diagram

```
                  ┌───────────────── one decoder ─────────────────┐
                  │   sw/runtime/common/  decode_caps(dev,isa,id)  │
                  └───────────────────────┬───────────────────────┘
                                          │ cp_mmio_read(GPU_*_CAPS_LO/HI)
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
      XRT / OPAE                  simx / rtlsim / gem5              (uniform)
   VX_cp_axil_regfile.sv          CommandProcessor (C++)
   GPU_DEV/ISA_CAPS @0x018+       GPU_DEV/ISA_CAPS @0x018+
   (one RTL producer)            (one C++ producer)
```

---

## 5. Runtime consolidation

### 5.1 Single decoder in `common/`

Add to `sw/runtime/common/`:

```cpp
// Decode a VX_CAPS_* id from the two raw caps words. Pure function;
// no device handle, no platform knowledge.
vx_result_t decode_caps(uint64_t dev_caps, uint64_t isa_caps,
                        uint32_t caps_id, uint64_t* out);
```

This is the current `get_caps` `switch` body, lifted once. Every
backend's `query_caps` becomes:

```cpp
vx_result_t query_caps(uint32_t caps_id, uint64_t* out) override {
    return decode_caps(dev_caps_, isa_caps_, caps_id, out);
}
```

### 5.2 Acquiring the raw words

Each backend obtains `dev_caps_`/`isa_caps_` once, at device-open, via
the CP regfile:

- **XRT / OPAE** — four `cp_mmio_read` (host offsets `0x1018..0x1024`),
  replacing today's reads of the legacy AFU `MMIO_DEV_ADDR (0x10)` /
  `MMIO_ISA_ADDR (0x18)` registers. New runtime constants:
  `CP_REG_GPU_DEV_CAPS (CP_BASE + 0x018)`,
  `CP_REG_GPU_ISA_CAPS (CP_BASE + 0x020)`.
- **simx / rtlsim** — the same four reads through the existing
  `cp_mmio_read` path into the `CommandProcessor cp_` member.
- **gem5** — the same four reads through `cp_mmio_read`, which the
  gem5 runtime issues as 32-bit PIO accesses to the gem5 device's CP
  regfile range (bare CP offsets, no `CP_BASE`).

Because the packing constants would then exist in two C++ spots (the
`CommandProcessor` producer and, implicitly, nowhere in the consumer —
the consumer only decodes), keep the two `gpu_*_caps()` helpers in a
shared header so the **only** C++ copy of the bit-packing is the one
the `CommandProcessor` includes.

### 5.3 What stays backend-local

`VX_CAPS_CLOCK_RATE` and `VX_CAPS_PEAK_MEM_BW` are genuinely
platform-specific (XRT/OPAE board constants; simx/rtlsim/gem5 return
`VX_CFG_PLATFORM_MEMORY_PEAK_BW` and `0` for clock). `decode_caps`
returns a "not handled here" sentinel for these two ids and the
backend supplies them — a tiny `switch` of two cases — or `decode_caps`
takes them as two extra scalar arguments. Either way they do **not**
go into the caps words.

---

## 6. Implementation plan

Per the no-skeletons rule, this lands as a small number of complete,
testable commits.

### 6.1 Phase 1 — Producers (RTL + C++ twin)

- `VX_cp_axil_regfile.sv`: add `gpu_dev_caps`/`gpu_isa_caps` and the
  four RO register cases; update header comment.
- `cmd_processor.{h,cpp}`: add the four `mmio_read` cases, the
  `gpu_*_caps()` helpers, the RO-ignore entries; update header
  comment.
- Extend the `cp_axil_regfile` verilator unit test (`hw/unittest/`)
  with reads of the four new offsets, asserting they equal the values
  `VX_afu_ctrl.sv` produces for the same config.
- **Testable:** unit test green; values cross-checked against the
  legacy AFU packing.

### 6.2 Phase 2 — Consumer consolidation

- Add `decode_caps()` to `sw/runtime/common/`.
- Repoint all five backends' `query_caps` at `decode_caps`, sourcing
  the words via `cp_mmio_read` of the new offsets.
- Delete the five per-backend `get_caps` switch bodies.
- **Testable:** any `vx_device_query` consumer returns bit-identical
  values on simx, rtlsim and gem5 before/after; XRT on hardware
  verified during the next CP bring-up.

### 6.3 Phase 3 — Remove the duplicated RTL (scoped)

- Delete `dev_caps`/`isa_caps` and the `ADDR_DEV*/ADDR_ISA*` register
  cases from `VX_afu_ctrl.sv`, and `MMIO_DEV_CAPS`/`MMIO_ISA_CAPS`
  from `vortex_afu.sv` — **for CP-enabled builds**.
- If a non-CP legacy bitstream is still produced, the AFU copies stay
  behind a build guard until that configuration is retired; see OQ-3.
- **Testable:** CP-enabled XRT/OPAE bitstreams build and the runtime
  caps query passes against the CP regfile only.

---

## 7. Risks and open questions

| Id | Item |
|---|---|
| **OQ-1** | **Macro visibility in the CP compilation unit.** Confirm `` `VX_CFG_MISA_EXT ``, `` `VX_CFG_MISA_STD ``, `` `VX_CFG_XLEN ``, `` `VX_ISA_IMPL_ID `` and the core/memory-config macros resolve inside `VX_cp_axil_regfile.sv`. May need `import VX_gpu_pkg::*`. Blocks Phase 1. |
| **OQ-2** | **`dev_caps` memory-bank fields.** XRT and OPAE use different parameter sources (§3.3). Pick the platform-neutral parameterization for the single CP copy and verify both shells currently evaluate to the same value; if they do not, that divergence is a separate bug to triage before consolidation. |
| **OQ-3** | **Legacy non-CP bitstreams.** Does any shipped configuration build an XRT/OPAE AFU *without* a CP? If yes, Phase 3 deletion is build-guarded; if no, the AFU caps logic is deleted outright. |
| **OQ-4** | **Twin-sync discipline.** The RTL regfile and the C++ `CommandProcessor` must agree — but this is already the standing contract for every queue register. The Phase 1 unit test makes a mismatch fail loudly. |
| R-1 | **Bit-exactness regression.** Mitigation: Phase 2 asserts before/after equality of every `VX_CAPS_*` id on simx/rtlsim/gem5, where a regression is cheap to catch. |
| R-2 | **Address-map churn.** The four new offsets are appended in free globals slots; no existing offset moves. Runtime `CP_REG_*` constants are additive. |

---

## 8. Why the CP, restated

`dev_caps`/`isa_caps` describe the **GPU core**, computed purely from
build configuration — nothing about them is XRT- or OPAE- or
CCI-P-specific. The platform shells have no reason to know the
bit-packing. The CP is the one host-control surface common to all five
backends, and it already has a faithfully-mirrored RTL/C++ register
pair. Putting the caps there yields:

- one RTL producer, one C++ producer (declared twins);
- one runtime consumer;
- AFU shells reduced to pure platform glue;
- the §3.3 drift class structurally eliminated.

The cost is two RO register pairs in two files whose job is already to
stay in lockstep — the smallest change that removes the duplication.

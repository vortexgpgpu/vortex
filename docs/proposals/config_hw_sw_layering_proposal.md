**Date:** 2026-05-21
**Status:** Draft — not yet approved
**Author:** Blaise Tine
**Related:** [config_macro_namespace_proposal.md](config_macro_namespace_proposal.md), [caps_cp_consolidation_proposal.md](caps_cp_consolidation_proposal.md)

# Vortex Config Layering: HW/SW Split — Proposal

## 1. Summary

`VX_config.toml` is the **hardware build configuration**. Its C/C++
projection, `VX_config.h`, is today `#include`d by the host runtime, the
kernel-side headers, and ~25 regression tests. This is a layering
violation: the software and test stacks inherit the *entire*
microarchitecture surface — cache MSHR depths, buffer sizes, arbiter
knobs, pipeline latencies — when they have no business seeing it.

The fix is **not** to relocate config wholesale into `VX_types.h`. Every
build/config value already has a natural home determined by its *kind*,
and — importantly — **every mechanism it needs already exists**:
`gen_config.py --cflags`, `config.mk`, `vx_device_query()`, and
`VX_types.h`. The defect is purely that host software reaches into the
wrong file. This proposal re-points each consumer at the right
mechanism, relocates one coherent ISA/ABI subset (re-prefixed to its new
home), and adds a single CI guard. No new generator *formats* — the
existing `sv_pkg` generator is reused to emit `VX_types_pkg.sv`.

## 2. Goals and non-goals

### 2.1 Goals

- `VX_config.h` is no longer reachable from host software. The
  microarchitecture surface stops leaking.
- Each build/config value is sourced from its natural home.
- The HW→SW dependency direction is one-way and enforced in CI.

### 2.2 Non-goals

- **Not** relocating config wholesale into `VX_types.h`. An earlier draft
  did that; it was wrong and is withdrawn — see §9.
- **Not** adding a new generator *format* or a `VX_config.mk`.
  `gen_config.py --cflags` already projects the TOML to compiler flags;
  the existing `sv_pkg` format is reused to emit `VX_types_pkg.sv`.
- **Not** re-opening the `VX_CFG_*` macro namespace. (Symbols relocated
  *out* of `VX_config.toml` are necessarily re-prefixed to their new
  domain — `VX_MEM_*` / `VX_VM_*` — because `VX_CFG_*` denotes the build
  config and must not appear in `VX_types.toml`; see §5.3.)
- **Minimal** RTL impact — but not *zero*. Relocating an RTL-consumed
  symbol requires `VX_gpu_pkg` to also `import`/`export` a new
  `VX_types_pkg`; no other RTL file changes, because modules reference
  the unchanged short names (`IO_BASE_ADDR`, `LMEM_BASE_ADDR`).
- **Not** fixing the simx/rtlsim/gem5 caps backends (§7) — that is the
  caps→CP consolidation proposal's scope.

## 3. Current state

### 3.1 The pollution

`VX_config.h` is `#include`d from, among others:

- the runtime: `sw/runtime/common/{common.h,vm.h,vm.cpp,vx_queue.cpp,scope.cpp}`, `sw/runtime/simx/vortex.cpp`
- shared/kernel headers: `sw/kernel/include/vx_spawn.h`, `sw/common/vm_types.h`
- **~25 regression tests** — every TCU/WGMMA test, every DXA test, `draw3d`, `amo`, `io_addr`, …

`ci/check_public_headers.sh` (from the namespace migration) guards only
`sw/runtime/include/` — the *public* runtime headers. Everything else
above is unguarded.

### 3.2 The leaked subset

Across the whole software and test tree only **28 distinct** `VX_CFG_*`
symbols are referenced — out of ~174 emitted into `VX_config.h`. Software
uses **16%** of the header and inherits the other 84% as dead weight.
§5 classifies all 28.

### 3.3 Consequences of the status quo

- **False rebuild dependencies** — re-tuning a cache MSHR depth
  recompiles 25 tests that never read it.
- **No enforced boundary** — nothing stops a test from `#if`-ing on a
  microarchitecture knob and silently coupling to an implementation
  detail.
- **Confused intent** — a reader cannot tell whether a `VX_config.h`
  include is a real dependency or accidental convenience.

## 4. The model: each value has a home by its kind

| Value kind | Home / mechanism for software |
|---|---|
| **Build / link parameter** | `config.mk` (← `config.mk.in`) and the linker scripts |
| **Runtime device property / capability** | `vx_device_query()` (`VX_CAPS_*`) |
| **Fixed ISA/ABI encoding** | `VX_types.h` |
| **Microarchitecture implementation** | `VX_config.h` — HW/sim-private; not exposed to software at all |
| **Software-only constant / policy** | a software-audience header |

`VX_types.h` is **not** a catch-all. For a value that should *not* be
queried, the destination is a conjunction:

> → `VX_types.h` **iff** software needs it **∧** it is a genuine HW↔SW
> contract (an encoding/format both sides must agree on bit-for-bit)
> **∧** it is not a build/link parameter **∧** it is not pure
> microarchitecture.

Decision procedure for any symbol:

1. Is it a runtime-discoverable device property? → `vx_device_query()`.
2. Else, does host/kernel software need it at all? **No** → it stays
   pure HW in `VX_config.h`; do not move it.
3. Else, is it a build-target / link-time parameter? → `config.mk` /
   linker scripts.
4. Else, is it a fixed HW↔SW contract? **Yes** → `VX_types.h`. **No** →
   a software-owned header.

This mirrors industry practice: capabilities are *queried* (`CPUID`,
`cudaGetDeviceProperties`), the ISA/ABI contract is *specified* (never
queried), the build target is *selected* at compile time (`-march`, CUDA
compute capability), and platform description comes from firmware tables
(ACPI / Device Tree).

## 5. Classification of the 28 leaked symbols

### 5.1 Runtime query — `vx_device_query()`

Device properties and capabilities. Host code obtains them at run time;
never as compile-time constants in host code.

| Symbols | `VX_CAPS_*` |
|---|---|
| `NUM_THREADS`, `NUM_CORES`, `NUM_WARPS`, `ISSUE_WIDTH` | already exist |
| `VM_ENABLE`, `EXT_DXA_ENABLE`, `EXT_TEX_ENABLE`, `EXT_RASTER_ENABLE`, `EXT_OM_ENABLE`, `TCU_SPARSE_ENABLE` | capability flags |
| `MISA_STD` / `MISA_EXT` (assembled ISA word) | `VX_CAPS_ISA_FLAGS` (exists) |

(`RASTER_TILE_LOGSIZE` / `RASTER_BLOCK_LOGSIZE` are graphics device
parameters — query likewise.)

### 5.2 Build parameter

| Symbol | Home |
|---|---|
| `XLEN` | `config.mk` — its sole job there is selecting the software toolchain (`libc$(XLEN)`, `riscv$(XLEN)-…`). Already a `-D`, not in any generated header. |

### 5.3 ISA/ABI contract → `VX_types.toml` (relocated + re-prefixed)

The device **memory map** and the VM **page-table format** are genuine
HW↔SW contracts. They relocate from `VX_config.toml` to `VX_types.toml`
**as two coherent units** — `[memmap]` and `[vm]` — because their members
cross-reference each other (`LMEM_BASE_ADDR = STACK_BASE_ADDR`,
`IO_EXIT_CODE` chains off `IO_COUT_*`); splitting them (an earlier draft
sent some to the linker scripts) would break the dependency DAG.

Relocated symbols are **re-prefixed** — `VX_CFG_*` denotes the build
config, so a symbol living in `VX_types.toml` must not carry it:

| Section | New prefix | Symbols |
|---|---|---|
| `[memmap]` | `VX_MEM_*` | `USER_BASE_ADDR`, `STACK_BASE_ADDR`, `STACK_LOG2_SIZE`, `LMEM_BASE_ADDR`, `IO_BASE_ADDR`, `PAGE_TABLE_BASE_ADDR`, `IO_COUT_ADDR`, `IO_COUT_SIZE`, `IO_EXIT_CODE`, `IO_END_ADDR` |
| `[vm]` | `VX_VM_*` | `PAGE_LOG2_SIZE`, `PAGE_SIZE`, `ADDR_MODE`, `PT_LEVEL`, `PTE_SIZE`, `PT_SIZE`, `PT_SIZE_LIMIT` |

`VX_types.toml` is generated `--resolved` and these `expr:` forms
reference `XLEN`. It declares an explicit `[[builtin]] XLEN` — and
`gen_config.py` sources `[[builtin]]` variables from the environment
(`configure` `export`s `XLEN`). No hand-built predicate flags, no
generator-side XLEN special-casing.

### 5.4 Pure microarchitecture → stays in `VX_config.h`

Software has no business seeing these; they are **not** moved anywhere.

| Symbols | Why |
|---|---|
| `TLB_SIZE` | TLB depth |
| `MEM_BLOCK_SIZE`, `MEM_ADDR_WIDTH`, `L1/L2/L3_LINE_SIZE`, `PLATFORM_MEMORY_*` | cache / memory / platform microarchitecture |
| `MISA_STD`, `MISA_EXT` | bit-packed expressions over `EXT_*_ENABLED` that *define* the hardware's `misa` CSR / AFU register (`VX_csr_data.sv`, `VX_afu_ctrl.sv`). Host obtains the assembled ISA word via `VX_CAPS_ISA_FLAGS` (§5.1) — it never needs the config macros. |

### 5.5 Software-only constants

`PT_SIZE_LIMIT` (a runtime sizing bound) and `PAGE_TABLE_BASE_ADDR`
(runtime placement policy — the HW reads `SATP`, never a fixed address)
are not strictly HW contracts. They are nonetheless relocated *with*
their coherent units (`[vm]` / `[memmap]`, §5.3) rather than into a
separate software header: keeping each unit whole outweighs the finer
distinction, and `VX_types.h` is already software-accessible.

### 5.6 Removed entirely

| Symbol | Why |
|---|---|
| `STARTUP_ADDR` | 100% software — the runtime programs the entry point via `VX_DCR_KMU_STARTUP_ADDR0/1`; the KMU launches cores at whatever it is given, so the HW has no hardwired entry point. The correct source is the ELF `e_entry`. |
| `NUM_PTE_ENTRY` | dead code — defined in the TOML, referenced nowhere |

### 5.7 Device-kernel compile-time constants

`kernel.cpp` in the TCU tests needs `NUM_THREADS` as a C++ **template
argument** (`wgmma_context<…>`); a runtime query cannot feed a template
parameter, and the kernel is device code compiled for a specific target.

**Resolution — use the mechanism that already exists.** `gen_config.py`
already projects the TOML to compiler flags via `--cflags`; the simx
Makefile uses it ([sim/simx/Makefile:33](../../sim/simx/Makefile#L33)):

```make
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py \
              --config=$(VORTEX_HOME)/VX_config.toml --cflags='…')
```

The test/kernel build does the same: it injects the config via
`gen_config.py --cflags` and adds the alias

```make
CXXFLAGS += -DNUM_THREADS=VX_CFG_NUM_THREADS
```

in the test `common.mk` (not each per-test Makefile). `kernel.cpp` then
gets `NUM_THREADS` via `-D`, with **no `#include <VX_config.h>`**. The
same value legitimately arrives two ways — `vx_device_query()` for the
host, a `--cflags` `-D` for the kernel — exactly as CUDA compute
capability is both queried and compiled-for; the existing
`if (NT != NUM_THREADS)` host check still cross-validates them.

## 6. Mechanisms

Most machinery already exists; the changes are small and listed below.

| Mechanism | Status | Role |
|---|---|---|
| `gen_config.py --cflags` | **exists** (simx Makefile) | projects `VX_config.toml` → `-DVX_CFG_*` for builds needing the HW config at compile time |
| `config.mk` ← `config.mk.in` | **exists** | carries `XLEN` + toolchain/paths for the software build |
| `vx_device_query()` / `VX_CAPS_*` | **exists** | host capability & dimension discovery |
| `VX_types.h` ← `VX_types.toml` | **exists** | the ABI header software already includes; receives the §5.3 relocation |
| `VX_types_pkg.sv` ← `VX_types.toml` | **new file, existing `sv_pkg` format** | typed `localparam` view of the relocated symbols, `import`/`export`-ed by `VX_gpu_pkg` so RTL keeps the unchanged short names |
| `gen_config.py` `[[builtin]]` from environment | **changed** | `[[builtin]]` variables are sourced from the environment — typed and declared in the TOML. `VX_types.toml` declares `[[builtin]] XLEN`, so its relocated `expr:` forms resolve from the environment with no XLEN special-casing in the generator |
| `gen_config.py` `sv_pkg` prefixes | **changed** | the `sv_pkg` emitter is generalized from the hardcoded `VX_CFG_` to a small prefix set (`VX_CFG_` / `VX_MEM_` / `VX_VM_`) so `VX_types_pkg` packagizes the relocated symbols |
| `ci/check_config_boundary.sh` | **new** | enforces the boundary |

`check_config_boundary.sh` generalizes `check_public_headers.sh`: CI
fails if any file outside `hw/`, `sim/simx/`, `sim/rtlsim/` includes
`VX_config.h` (directly or transitively). Wired into `regression.sh`.

## 7. Adjacent issues found (out of scope, recorded)

- **`NUM_PTE_ENTRY` is dead code** — removed as part of §5.6.
- **RTL VM is SV32-only.** `VX_mmu_ptw.sv` is hardcoded SV32 (20-bit VPN,
  4-byte PTE, 2-level); SV39 is unimplemented. `XLEN=64` virtual memory
  is effectively unsupported in RTL today. A standalone RTL completeness
  gap.
- **MISA caps-backend leak.** `sw/runtime/{rtlsim,simx,gem5}/vortex.cpp`
  fabricate `VX_CAPS_ISA_FLAGS` from `VX_CFG_MISA_*` instead of reading
  the device model. Fix belongs to
  [caps_cp_consolidation_proposal.md](caps_cp_consolidation_proposal.md).
- **`vortex2.h` drift hazard** — carries hand-written
  `// must match VX_CFG_MISA_*` comments; a constant duplicated by
  eyeball.

## 8. Implementation plan

Phased; reviewed per phase. RTL stays buildable throughout (short names
unchanged); C/C++ consumers break after Phase 1 and are restored by
Phases 2–3 — expected for a phased landing.

- **Phase 1 — Relocate.** Move the §5.3 contract subset
  `VX_config.toml` → `VX_types.toml` `[memmap]`/`[vm]`, re-prefixed
  `VX_MEM_*` / `VX_VM_*`; remove `STARTUP_ADDR` and `NUM_PTE_ENTRY`.
  `configure` passes `XLEN` predicates to the `VX_types` generation;
  `gen_config.py`'s `sv_pkg` emitter is generalized to the new prefixes;
  `VX_gpu_pkg` `import`/`export`s the generated `VX_types_pkg`. Verify
  the generated headers and that RTL still elaborates.
- **Phase 2 — Re-point host code.** Drop `#include <VX_config.h>` from
  `main.cpp` and the runtime; use `vx_device_query()` for
  dimensions/capabilities, `config.mk` for `XLEN`, `VX_types.h` for the
  contract subset. Move the §5.5 SW-only constants into a runtime header.
- **Phase 3 — Re-point kernel/test build.** Drop `#include <VX_config.h>`
  from `kernel.cpp`; the test `common.mk` injects config via
  `gen_config.py --cflags` and adds `-DNUM_THREADS=VX_CFG_NUM_THREADS`;
  kernels include `VX_types.h` for the contract subset.
- **Phase 4 — Enforce.** Land `check_config_boundary.sh`; wire into
  `regression.sh`; retire `check_public_headers.sh`.
- **Phase 5 — Verify.** `simx` + `rtlsim` regression on both ISAs; spot
  re-build the TCU/DXA/graphics tests.

## 9. Alternatives considered

- **Relocate everything into `VX_types.toml`** (this proposal's earlier
  draft). Withdrawn: it treated device *properties* and *build*
  parameters as ISA/ABI *contract*. `NUM_THREADS` is a device property;
  `XLEN` is a build parameter — folding them into `VX_types.h` only
  replaces one mislayered header with another.
- **A new `gen_config.py` make-dialect / `VX_config.mk`.** Rejected:
  `gen_config.py --cflags` already projects the TOML to compiler flags,
  as the simx Makefile demonstrates. Inventing a parallel output is
  redundant.
- **Tag-and-filter in `VX_config.toml`.** Tag each key with an
  `audience`. Rejected: it keeps the *contract defined in the
  build-config file*, and §5 shows most leaked symbols are not contract.
- **Do nothing.** The boundary erodes monotonically — every new test
  copied from an existing one inherits the `VX_config.h` include. The fix
  cost only grows. Rejected.

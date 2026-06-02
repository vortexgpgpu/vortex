# Build Configuration System — Design

**Scope:** the TOML-driven build-configuration system — how a single
source of config values flows to RTL, simulators, runtime, and kernels;
the `VX_CFG_*` macro namespace; and the hardware/software layering between
[`VX_config.toml`](../../VX_config.toml) and
[`VX_types.toml`](../../VX_types.toml).

---

## 1. Sources and generator

Two TOML files are the single source of truth:

- [`VX_config.toml`](../../VX_config.toml) — **hardware build config**
  (~245 keys: platform, ISA, pipeline, memory, caches, FPU, TCU, graphics,
  toolchain, debug). Keys carry their namespace prefix in the TOML itself
  (`VX_CFG_*`, `VX_DBG_*`; `[toolchain]` predicates stay bare).
- [`VX_types.toml`](../../VX_types.toml) — the **software-facing ISA/ABI
  contract** (`VX_CSR_*`, `VX_DCR_*`, `VX_ISA_*`), plus the relocated
  `[memmap]` (`VX_MEM_*`) and `[vm]` (`VX_VM_*`) sections and the
  `[[builtin]] XLEN`.

The generator is [`ci/gen_config.py`](../../ci/gen_config.py): given one
`--config <toml>` and a `--format {cflags,cpp,verilog}`, it resolves the
symbol table from that one file (`expr:` keys are evaluated; `[[builtin]]`
vars come from the environment) and emits the corresponding header.
[`configure`](../../configure) drives this: for each `*.toml` it emits
`<build>/hw/<name>.vh` (Verilog `` `define ``) and `<build>/sw/<name>.h`
(C `#define`); `VX_types` is generated `--resolved`, and `export XLEN`
feeds the `[[builtin]]`.

---

## 2. Value flow and layering

```
  VX_config.toml ──gen_config.py──► build/hw/VX_config.vh  (`define)  ──► RTL (`VX_CFG_*)
                                 └─► build/sw/VX_config.h   (#define)  ──► sim/runtime
                   gen_config.py --cflags ──► -DVX_CFG_*  ──► sim & kernel/test builds

  VX_types.toml  ──gen_config.py──► build/hw/VX_types.vh   ──► RTL (`VX_MEM_*, `VX_CSR_*)
                                 └─► build/sw/VX_types.h    ──► SW/runtime (ABI contract)
```

RTL includes both generated headers via
[`hw/rtl/VX_define.vh`](../../hw/rtl/VX_define.vh) and references
`` `VX_CFG_* `` / `` `VX_MEM_* `` macros directly — there is **no**
SystemVerilog config package.

**Hardware/software layering.** `VX_config.toml` → `VX_config.{h,vh}` is
HW/sim-private. `VX_types.toml` → `VX_types.{h,vh}` is the shared ABI
(`VX_types.vh` consumed by RTL, `VX_types.h` by SW). The genuine HW↔SW
contracts — the device memory map and the VM page-table format — were
relocated out of `VX_config.toml` into `VX_types.toml`'s `[memmap]`/`[vm]`
sections (re-prefixed `VX_MEM_*` / `VX_VM_*`), so the public runtime
headers never include `VX_config.h`. Two CI guards enforce the boundary:
[`ci/check_config_boundary.sh`](../../ci/check_config_boundary.sh) and
`ci/check_sw_sim_boundary.sh`.

---

## 3. The `VX_CFG_*` namespace

The macro namespace is authored directly in the TOML key spelling (the
generator has no naming logic): `VX_CFG_*` for hardware config, `VX_DBG_*`
for debug knobs, `VX_MEM_*`/`VX_VM_*` for the relocated contract,
`VX_CSR_*`/`VX_DCR_*`/`VX_ISA_*` for types, and bare `[toolchain]`
predicates (`ASIC`, `SYNTHESIS`, `VIVADO`, …). The public runtime header
[`vortex2.h`](../../sw/runtime/include/vortex2.h) is fully decoupled — it
uses inlined literal bit positions (e.g. `VX_ISA_EXT_*` as
`(1ull << (32 + N))`) rather than including any generated config. `XLEN`
reaches kernels/tests via `-DVX_CFG_XLEN=$(XLEN)`.

---

## 4. Proposed but not yet implemented

1. **SystemVerilog typed-config package** (`config_macro_namespace` §9,
   `config_hw_sw_layering` §2.2) — **deliberately not done and not to be
   re-attempted without first solving sv2v re-export.** A typed-localparam
   `VX_config_pkg`/`VX_types_pkg` was built and reverted: sv2v drops
   wildcard `export pkg::*` symbols (the yosys ASIC flow failed on
   `L3_CACHE_SIZE` etc.), and package compile-ordering is tool-divergent
   (Verilator strict, Vivado auto-sorts, VCS order-sensitive). Backtick
   macros are the one representation every tool handles identically. The
   post-mortem rationale is the load-bearing artifact to preserve.
2. **RTL VM Sv39** — `VX_VM_ADDR_MODE` resolves to `SV39` for XLEN=64 in
   `VX_types.toml`, but `VX_mmu_ptw.sv` is hardcoded Sv32 (see
   [`virtual_memory_subsystem.md`](virtual_memory_subsystem.md) §8).
3. **MISA caps-backend leak** — `sw/runtime/{rtlsim,simx,gem5}/vortex.cpp`
   fabricate `VX_CAPS_ISA_FLAGS` from `VX_CFG_MISA_*` rather than reading
   the device model; deferred to the capability-register consolidation
   (now landed; see [`command_processor_control_plane.md`](command_processor_control_plane.md)
   §6). `vortex2.h` still carries "match VX_CFG_MISA_EXT" drift-hazard
   comments.

**Superseded directions** (recorded to avoid revival): `ci/check_public_headers.sh`
(replaced by `check_config_boundary.sh`); the
`VX_gpu_pkg import VX_types_pkg` package-import model (RTL reads
`VX_types.vh` backtick macros directly); and the `[isa_signatures]`
expansion (dissolved to `VX_CFG_MISA_STD/EXT` with literal bit shifts).
Note: both source proposals still carry "Draft"/"not approved" status
headers despite their cores being landed.

---

## 5. Source proposals

This design consolidates and supersedes `config_hw_sw_layering_proposal.md`
and `config_macro_namespace_proposal.md` (now removed from
`docs/proposals/`).

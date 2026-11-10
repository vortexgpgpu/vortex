---
name: rtl-review
description: Review checklist for Vortex RTL changes under hw/rtl, hw/dpi, hw/unittest and hw/scripts. Use together with code-review whenever a diff touches SystemVerilog, an interface, VX_gpu_pkg, a library module or the synthesis flow.
---

# Reviewing Vortex RTL

Load `docs/coding_guidelines_verilog.md`, `docs/hardware_library.md` and
AGENTS.md §4–§6 first. Each item below is a rule from those documents or a
failure the project has had; cite the rule in the finding.

## Contracts

- **Interfaces** (`hw/rtl/interfaces/`): a new or changed signal or modport
  changes every master and slave. Find every instantiation (`grep -rn
  "<name>_if"`) and check each one drives or consumes it.
- **`VX_gpu_pkg` structs**: a field change moves every packed slice that indexes
  it, the trace macros that print it and the SimX mirror. Check widths that
  depend on `VX_CFG_*` at the extremes the toml allows (1 lane, max lanes,
  XLEN 64).
- **`VX_types.toml`**: a CSR, DCR, opcode or memory-map value is shared with
  SimX, the runtime, the kernel library and LLVM. All change together.
- **`VX_CFG_*`** come from `VX_config.toml` only; no `ifndef/define` fallback in
  RTL.

## Handshakes and ownership

- valid/ready: `valid` must not depend combinationally on `ready` of the same
  interface; data holds while `valid && !ready`; nothing is dropped or
  duplicated on a stall. Look for a fire condition that ignores `ready`.
- The module that drives an interface registers it at its own output (`*_OUT_BUF`,
  `VX_elastic_buffer`, a bus slice); no consumer or routing parent re-registers
  an interface it does not own (guidelines §4).
- Reset: every state register that gates `valid` resets; a partial reset is
  stated and safe.
- Arbiters and round-robin: check fairness under a held request, and that a
  grant cannot change while the winner is stalled unless the protocol allows it.
- Tags and IDs: a response tag is wide enough for every outstanding request
  (MSHR, bank, port index), for read and write paths alike.

## Library and memories

- `hw/rtl/libs/` is shared IP. A change there needs an equivalence argument for
  every existing user, or a new parameter that keeps the old behavior.
- Memories go through `VX_dp_ram`/`VX_sp_ram`; their address must stay
  computable by the RAM fold (guidelines §11). `VX_dp_ram` and `VX_sp_ram`
  evolve together.
- Assertions (`RUNTIME_ASSERT`) qualify on `valid` so X or idle data cannot fire
  them.

## Style rules that hide bugs

- `begin`/`end` on every branch and loop; generate blocks labelled `g_*`.
- No blanket `verilator lint_off`; `UNUSED_*` per signal.
- One port or parameter per line in declarations and instantiations.
- `TRACE` arguments comma-separated.
- No comment that names SimX, the runtime, a task, a PR or a fix.

## Timing and area

- A change to a pipeline, arbiter, queue, interface or datapath can add logic
  levels: the PR shows Beacon (`ci/logic_gate.py -b <dut> --against <parent>`)
  clean, or the reason a unit got deeper. Baselines under
  `ci/baselines/synthesis/` are re-recorded by a human running the gate's
  `--update-baseline`, never by an agent or CI, and never edited.
- FPU latency changes move `C_Latency`, `VX_CFG_*_LATENCY` and SimX together.
- A new DUT or a renamed one is updated in both `hw/syn/{xilinx,yosys}/dut/catalog.mk`.

## Lockstep with SimX

- A change that moves cycles (pipeline depth, arbitration, queue depth, cache
  or memory behavior) lands with the SimX timing-model change, and a
  `model_parity` case exercises it. Never widen a parity tolerance to absorb a
  gap.
- A change that should not move cycles says which parity or perf case shows
  that it did not.

## Coverage

- RTL coverage is `xrt`, not only `rtlsim`, for anything at the AFU or host
  surface (`hw/rtl/afu/`).
- A bug fix adds the case that failed before it (`docs/bug_fixes.md` §4).

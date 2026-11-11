# Dynamic Warp Allocation

## Goal

Size each warp's register allocation from the kernel instead of reserving all
63 architectural registers per warp. A core keeps `VX_CFG_NUM_WARPS` warp
contexts, but how many are resident depends on how many register sets of the
kernel's size fit in the physical register file. This is the NVIDIA model
(`-maxrregcount`, occupancy from registers per thread).

Register demand varies widely between kernels (below). Non-tensor kernels use
a fraction of the 63 registers they reserve. TMEM kernels use nearly all of
them, but their integer registers hold mostly warp-uniform values (TMEM
addresses, descriptors, barrier ids, loop bounds: 74-83% of the operands the
TMEM GEMM kernels read are uniform, against 18-20% in the chained GEMM and
FA4), so with uniform registers their `Rx` drops and they gain the most warps.

## Motivation

Registers used, of 63, in the current regression kernels (static, whole binary):

| Kernels | Used |
|---|---|
| vecadd, relu, dropout, sort | 10-13 |
| bfs, gfx_tex, pathfinder | 12-15 |
| sgemm, conv3, gfx_draw3d | 25-29 |
| jacobi, stencil3d, softmax, gfx_tex4 | 32-35 |
| WMMA, WGMMA | 38-54 |
| TMEM, FA4 | 57-61 |

Cycles against warp count (SimX, 1 core, 8 threads, issue width 1, default
memory latency): 8 to 16 warps gives -7% to -25% on vecadd, pathfinder,
conv3, sgemm and gfx_tex; 16 to 32 warps gives at most -9%, and sgemv is +18%
slower at 32. At default latency the benefit is fitting 16 warps in the register
file of 8; the benefit at realistic DRAM latency is measured in phase 0.

## Design

### Register count

The kernel's register count is `R = Rx + Rf`: `Rx` integer registers
(x0..x(Rx-1)) and `Rf` FP registers (f0..f(Rf-1)), each rounded up to a
multiple of 4, the default `VX_CFG_NUM_GPR_BANKS`, so the bank select stays a
bit slice of the physical index (a configuration with more banks rounds to the
bank count). A kernel without a count runs with `R = 64`, which is today's
layout.

### Register packing

A post-link pass renames registers consistently across the whole statically
linked kernel image so the integer registers in use are x1..x(Rx-1) and the FP
registers in use are f0..f(Rf-1). A consistent renaming of a closed binary
preserves its meaning, including the calling convention, so no ABI change or
library rebuild is needed.

FP registers are packed only in images with no tensor instructions. An image
containing WMMA, WGMMA or TMEM_LD/TMEM_ST keeps its FP numbering and takes
`Rf = 32`: those instructions name FP registers implicitly (WMMA's `TCU_RC`,
`TCU_RA`, `TCU_RB` in `VX_tcu_pkg.sv`; WGMMA's accumulators and register-sourced
A; TMEM's ranges that follow an encoded first register), and a TMEM kernel's
fragments use the whole FP file anyway.

Integer registers are always packed; only `x0` stays fixed. TMEM tensor-job
instructions encode their integer operands, and registers pinned only to pick
an encoded operand (DXA's `x30`/`x31`) are renamed like any other.

Images with warp-issued WGMMA (descriptors read implicitly from `x10`/`x11`,
`VX_tcu_uops.sv`) or RTU instructions are left unpacked (`R = 64`).

The pass decodes every Vortex custom encoding, so `.insn` and `.word` forms are
rewritten too; an instruction it cannot decode fails the build rather than
passing through.

`R` is recorded in the kernel image; the runtime reads it at launch.

### Physical register file

Integer and FP registers share one register file, as they do today: each
operand collector has one set of banked RAMs, and a warp's region holds x0-x31
then f0-f31, the register id's type bit being an address bit
(`VX_opc_unit.sv`). With dynamic allocation a warp's region is `Rx` integer rows
followed by `Rf` FP rows, so a kernel without FP (`Rf = 0`) turns the FP space
into more warps. SimX keeps its separate integer and FP arrays as storage; only
the row arithmetic and the capacity are modeled.

`VX_CFG_GPR_ROWS` sets the physical rows per operand collector (default
`NUM_REGS * PER_OPC_WARPS`, today's size, so existing configurations are
unchanged). The physical row is

```
phys = base[wid] + (is_fp ? Rx : 0) + reg
```

on both the read path and the writeback path, with the thread-group select
(`sid`) as today when `NUM_THREADS > SIMD_WIDTH`. `x0` keeps its slot and is
never written, as today.

A register at or above its class count is outside the warp's set and aliases
another warp's registers of the same kernel, like an out-of-bounds shared
memory access; the packing pass derives `R` from the image, so a built kernel
never names one.

### CTA dispatch

The KMU runs one kernel at a time, so every resident warp has the same `R` and
the register file divides into equal slots, as LMEM does (`VX_cta_dispatch.sv`
fixed-stride allocator):

- `usable_warps_r`: the largest `m` in `[1, NUM_WARPS]` with
  `ceil(m / NC) * R <= GPR_ROWS`, where `NC = ISSUE_WIDTH * NUM_OPCS` is the
  number of operand collectors and warps interleave across them; computed by a
  comparator tree and registered at launch.
- Free-warp selection masks warp ids at or above `usable_warps_r`; warp ids stay
  dense, so slots never fragment.
- Every warp of a kernel has the same `R`. `base[wid] = (wid / NC) * R` within
  its collector (the slot `wid_to_opc_slot` computes today), filled by a
  running sum when the kernel starts, before the first CTA is admitted. The
  table has one entry per warp, not per slot, so per-rank counts can be added
  without changing the register file.
- A CTA is admitted only when it fits the LMEM bound; its warps launch into
  free warp slots below the register bound, one CTA at a time, as today. A CTA
  with more warps than the bound could never finish launching: the dispatcher
  grants it its warps with a warning (as it does an LMEM overrun), and the
  runtime is to reject such a launch once the register file size is a device
  capability.
- Fragment waves take free warp slots below the same bound.

`VX_CFG_NUM_WARPS` becomes the maximum number of resident warps.

### Launch interface

- `VX_DCR_KMU_NUM_REGS`: `{Rf, Rx}`, set by the runtime with the other KMU DCRs;
  reset value gives `R = 64`.
- `kmu_req_t` carries the counts to the dispatcher with `aligned_lmem_size`.
- Runtime: reads `R` from the kernel image. Additive; no API change.

## Implementation

| Area | Files | Change |
|---|---|---|
| Config | `VX_config.toml`, `VX_types.toml` | `VX_CFG_GPR_ROWS`; `VX_DCR_KMU_NUM_REGS` |
| Packing pass | new script in `sw/kernel/scripts/`, kernel link rule in the test makefiles | rename, record `R`, self-test |
| Runtime | `sw/runtime/common/` | read `R`, write the DCR |
| KMU | `VX_kmu.sv`, `sim/simx/kmu/` | DCR, `kmu_req_t` field |
| Dispatch | `VX_cta_dispatch.sv`, `sim/simx/cta_dispatcher.cpp` | register bound, warp mask, base table |
| Register file | `VX_opc_unit.sv` | physical size, `base + offset + reg` on read and write (SimX keeps its per-warp arrays: the bank of a register is unchanged) |
| Perf counters | `VX_types.toml`, SimX and RTL perf | resident-warp limit per launch |

## Uniform registers

Designed in [uniform_registers_proposal.md](uniform_registers_proposal.md) and
built with this proposal: warp-invariant integer values (kernel arguments,
bases, strides, bounds, descriptors) move to a per-warp `u` file, which lowers
`Rx` and so raises the warps that fit. The `u` file is one fixed 32 x 64-bit
set per warp context, not dynamically allocated: at 8 threads it is an eighth
of a full `x`+`f` set, and the KMU presets `u1`/`u2` at warp launch.

The packing pass decodes the `[1:0]` mode of each instruction to tell `x`
fields from `u` fields; `u` registers are not renamed.

### Where the packing runs

Packing stays a post-link pass, also after LLVM allocates `u` registers:

- `R` is the maximum over everything `kernel_main` reaches, library code
  included, so it is a whole-image property in either place.
- Packing in LLVM needs an ascending allocation order **and** a calling
  convention that passes arguments in low registers (the standard one puts
  them at x10-x17), with compiler-rt, libc and libm rebuilt for it and shipped
  through `vortex-toolchain-prebuilt`. A consistent renaming of the linked image
  gets the same result with neither.
- A per-function renaming inside LLVM would break calls between functions; a
  whole-image one would need LTO over the libraries, which is the post-link pass
  again.

LLVM's part is the `u` register class and uniformity analysis; the pass only
renumbers what LLVM allocated.

## Phases

The two features share the register-file, dispatch and launch work, and the
LLVM work for `u` registers is the long pole, so the toolchain work starts
first and the hardware work runs alongside it.

0. **Measure.**
   - The warp sweep with `VX_CFG_PLATFORM_MEMORY_LATENCY` at V80-like and
     H100-like values.
   - Per kernel, the integer registers that only ever hold warp-uniform values
     (SimX trace), to estimate how far `u` registers lower `Rx`.
   - Pick the `GPR_ROWS` sizes worth evaluating.
1. **Toolchain MC** (Vortex LLVM, then a `vortex-toolchain-prebuilt` refresh,
   landed before any Vortex change that uses it): `+xvortexu`, the `[1:0]`
   length decoding, `u` registers in the assembler and disassembler.
2. **Packing pass** (in parallel with 1): every regression kernel retires the
   same instructions with the same results packed and unpacked, on SimX at the
   default configuration.
3. **SimX**: the physical register file, dispatch bound and base table,
   `VX_DCR_KMU_NUM_REGS`; the `u` file, both modes, KMU preload,
   `vx_uniform.h` intrinsics. Regression at the default size (unchanged
   cycles), at a reduced `GPR_ROWS`, and a test per uniform form (uniform ALU,
   branch, move, per-warp load/store, mixed operand).
4. **RTL**: the same, with `model_parity` cases:
   - `NUM_WARPS=16` with `GPR_ROWS` for 8 full warps: a low-`R` kernel, a
     high-`R` kernel and a TMEM kernel;
   - one case per uniform form.
   `logic_gate` impact reviewed before any baseline is re-recorded.
5. **LLVM codegen**: `u` register class, uniformity analysis, the `u` ABI;
   prebuilt refresh. Regression kernels rebuilt with it, then packed.
6. **Evaluation**: `Rx` with and without `u` registers, resident warps, cycles
   against `GPR_ROWS` at both latencies, area and Fmax. A V80 build with
   approval.
7. **Later**: `u` operands on Vortex custom instructions (TMEM, DXA, barriers);
   64-bit global addressing through a `u` base; x0 storage removed by zeroing
   the read register; WGMMA and RTU images packed once their implicit operands
   are encoded; per-rank register counts fixed at launch for warp-specialized
   kernels (FA4's control warps against its softmax warps), with each rank's
   base a prefix sum within the CTA, ranks pinned to collectors, and
   per-function register limits in LLVM.

## Open questions

- Whether a kernel may request fewer resident warps than fit (`sgemv` and
  `stencil3d` are slower with more warps), as a launch hint.

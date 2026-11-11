# Uniform Registers and 64-bit Addressing at XLEN=32

## Goals

1. A per-warp uniform register file with ALU, load, store and branch
   instructions, so warp-invariant values (kernel arguments, bases, strides,
   bounds, loop counters) are held and computed once per warp.
2. Uniform operands in per-thread instructions, as RVV's `.vx` forms take a
   scalar operand.
3. Device memory above 4 GB at XLEN=32 without virtual memory: a 64-bit
   uniform base plus a 32-bit per-thread offset (NVIDIA `[R.U32 + UR.64]`, AMD
   `global_load v, v_off, s[base]`, RVV indexed loads). V80 has 32 GB of HBM;
   Vortex reaches 4 GB of it today.

Non-goals: per-thread 64-bit pointers, virtual memory.

## Encoding

`VX_CFG_EXT_U_ENABLE` requires C disabled. Without C every instruction is
32 bits and bits `[1:0]` are always `11`; the extension assigns two of the
three unused values:

| `[1:0]` | Mode | Analogue |
|---|---|---|
| `11` | standard | |
| `10` | **uniform**: the instruction runs once per warp on `u` registers | RVV's scalar instructions |
| `01` | **uniform operand**: a per-thread instruction with one `u` source | RVV `.vx` / indexed loads |
| `00` | reserved | |

Bits `[31:2]` keep their standard fields, funct values and immediates; the
mode only changes which register file an operand names. A combination not
listed below is reserved and decodes to no unit.

This is a Vortex convention, not standard RISC-V: standard tools read
`[1:0] != 11` as a 16-bit instruction and lose sync. Vortex's LLVM decodes it,
and binaries carry an ELF attribute so other tools refuse them.

### Mode `10`: uniform

| Opcode | Meaning |
|---|---|
| OP, OP-IMM, OP-32, OP-IMM-32, LUI (incl. M, Zicond, Zba, Zbb) | every register field names a `u` register; RV64 semantics at any XLEN (`add` 64-bit, `addw` 32-bit sign-extended) |
| BRANCH | `us1 cmp us2`: a uniform branch, no divergence |
| JALR | `jalr ud, imm(us1)` |
| LOAD / STORE | `ud = mem[us1 + imm]`, `mem[us1 + imm] = us2`: one access per warp (kernel arguments, spills) |
| SYSTEM (Zicsr) | `csrr* ud, csr, us1` |
| OP-FP `fmv.d.x` / `fmv.x.d` encodings | `u.mvx ud, xs1` (first active lane) / `x.mvu xd, us1` (broadcast): the `f` field names a `u` register, the `x` field stays `x` |

### Mode `01`: uniform operand

The second source is uniform, as RVV's `vop.vx vd, vs2, rs1` computes
`vs2 op rs1`.

| Opcode | Meaning |
|---|---|
| OP, OP-32 (incl. M, Zicond, Zba, Zbb) | `xd = xs1 op us2`: `i * stride`, `x << shift`, `x - base`, `i < n` |
| OP-IMM, OP-IMM-32 | `xd = us1 op imm` (the only source is uniform); `addi xd, us1, 0` is a broadcast |
| BRANCH | `xs1 cmp us2`: `blt i, n` |
| STORE | `mem[xs1 + imm] = us2` |
| AMO | **global indexed access**, address `us1 + zext(xs2)` (64-bit), width in funct3 as LOAD/STORE: funct5 `11110` load to `xd`; funct5 `11111` store from the `rd` field; standard AMO funct5 values are the atomics, `rd` both the operand and the old value |
| Vortex custom (TCU, TMEM, DXA, warp control) | per instruction, the operand that is warp-invariant (descriptor, TMEM address, barrier id) from `u`; defined in phase 5 |

A uniform first operand of a non-commutative op (`u - x`, `u << x`, `u < x`)
takes a broadcast (`addi xt, us, 0` in mode `01`) first. If that proves
common, mode `00` is the reserved place for reversed forms.

## Semantics

- `u0`–`u31` per warp, 64 bits at either XLEN; `u0` reads zero.
- A uniform operand of a per-thread instruction is broadcast to every lane;
  at XLEN=32 its low 32 bits are used, except as a global base.
- `u.mvx` takes the first active lane, the rule Vortex already uses for CSR
  writes and branch direction.
- Global indexed addresses are 64-bit, truncated to the physical address
  width, and are never decoded as LMEM/stack/IO/OM apertures. Ordinary
  loads and stores keep 32-bit addresses (low 4 GB, apertures unchanged).

## ABI

- At warp spawn the KMU sets `u1` to the kernel argument block and `u2` to a
  per-warp scratch area for `u` spills; the others are zero.
- `u3`–`u15` caller-saved, `u16`–`u31` callee-saved.
- With the extension, device pointers in kernel arguments are 64-bit at
  XLEN=32; a kernel loads them with `ld ud, off(u1)` (mode `10`).

## Implementation

### Hardware

| Area | Files | Change |
|---|---|---|
| Config | `VX_config.toml` | `VX_CFG_EXT_U_ENABLE` (exclusive with C); `VX_CFG_PHYS_ADDR_WIDTH` split from `VX_CFG_MEM_ADDR_WIDTH` (35 for V80) |
| Register classes | `VX_gpu_pkg.sv` | `REG_TYPE_U`; `REG_TYPES` = 3, `NUM_REGS` = 96 |
| Decode | `VX_decode.sv` | mode from `[1:0]`; operand classes per the tables; `USED_UREG` |
| Uniform register file | new `VX_ureg_file.sv`; `VX_operands.sv`, `VX_opc_unit.sv` | per-warp 32 x 64 bits, read once and broadcast |
| Scoreboard | `VX_scoreboard.sv` | wider register id |
| Uniform ALU | new `VX_alu_uniform.sv` in `VX_alu_unit.sv` | one 64-bit lane: ALU, M, branches |
| LSU | `VX_lsu_agu.sv`, `VX_lsu_block.sv`, `VX_lsu_mem_path.sv`, `VX_uop_amo.sv` | single-lane mode-`10` access with `u` writeback; 64-bit `us1 + zext(xs2)` global indexed loads, stores and atomics |
| Writeback | `VX_commit*.sv` | `u` destination, first-active-lane select |
| CSR | `VX_csr_unit.sv` | `u` source and destination |
| KMU | `VX_kmu.sv` | `u1`/`u2` preload |
| Memory path | `VX_gpu_pkg.sv`, `hw/rtl/cache`, `hw/rtl/mem`, `hw/rtl/afu`, `hw/rtl/dxa` | physical width below the LSU; DXA descriptor base at full width |

### SimX

`decode.cpp` (modes), `types.h` (`RegType::U`), the warp's `u` file,
`alu_unit.cpp` (64-bit uniform ops), `lsu_unit.cpp` and `mem/` (64-bit
global addresses), `operands.cpp`, `scoreboard.cpp`, the KMU preload, and
`decompressor.cpp` (disabled with the extension). The uniform ALU latency and
the single-lane access are modeled to match the RTL.

### Toolchain and runtime

- Vortex LLVM, through `vortex-toolchain-prebuilt`: a `+xvortexu` feature
  that changes instruction-length decoding; `u` registers in the assembler and
  disassembler; both modes generated from the standard instruction
  definitions. Code generation follows in phase 6.
- `sw/kernel/include/vx_uniform.h`: inline-assembly intrinsics.
- Runtime: `GLOBAL_MEM_SIZE` from the platform, 64-bit `MemoryAllocator`
  accessors, allocations above 4 GB at XLEN=32, 64-bit pointer arguments.

## Phases

Built with dynamic warp allocation; see the combined phases in
[dynamic_warp_allocation_proposal.md](dynamic_warp_allocation_proposal.md#phases).
Wide memory (64-bit global indexed access, the physical address width below the
LSU, the runtime allocator above 4 GB) follows as its own phase.

## Open questions

- 32 or 16 `u` registers (32 x 64 bits is 32 Kbit per core at 16 warps).
- Whether the `u` file is also allocated per kernel, as the `x`/`f` file is.
- Global indexed stores and atomics use `rd` as a source; LLVM models that as
  a tied operand.
- V80 host staging: the CP staging aperture sits in HBM8 and up, inside a
  32 GB device space; it needs another home.

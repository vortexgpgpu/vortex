# Uniform Registers

## Goals

1. A per-warp uniform register file with ALU, load, store and branch
   instructions, so warp-invariant values (kernel arguments, bases, strides,
   bounds, loop counters) are held and computed once per warp.
2. Uniform operands in per-thread instructions, as RVV's `.vx` forms take a
   scalar operand.

Uniform registers are XLEN bits wide and keep the base ISA's semantics: a
uniform instruction is the standard instruction applied to values that are the
same in every lane. Memory above 4 GB at XLEN=32 is a separate question.

## Encoding

`VX_CFG_EXT_U_ENABLE` requires C disabled. Without C every instruction is
32 bits and bits `[1:0]` are always `11`; the extension assigns two of the
three unused values:

| `[1:0]` | Mode | Analogue |
|---|---|---|
| `11` | standard | |
| `10` | **uniform**: the instruction runs once per warp on `u` registers | RVV's scalar instructions |
| `01` | **uniform operand**: a per-thread instruction whose second source (or only source) is a `u` register | RVV `.vx` |
| `00` | **uniform first operand**: a per-thread instruction whose first source is a `u` register, the other per-thread | |

Bits `[31:2]` keep their standard fields, funct values and immediates; the
mode only changes which register file an operand names. A combination not
listed below is reserved and decodes to no unit.

This is a Vortex convention, not standard RISC-V: standard tools read
`[1:0] != 11` as a 16-bit instruction and lose sync. Vortex's LLVM decodes it,
and binaries carry an ELF attribute so other tools refuse them.

### Mode `10`: uniform

| Opcode | Meaning |
|---|---|
| OP, OP-IMM, LUI, AUIPC (incl. M, Zicond); OP-32, OP-IMM-32 at XLEN=64 | every register field names a `u` register |
| BRANCH | `us1 cmp us2`: a uniform branch, no divergence |
| JALR | `jalr ud, imm(us1)` |
| LOAD / STORE | `ud = mem[us1 + imm]`, `mem[us1 + imm] = us2`: one access per warp (kernel arguments, spills); `ld`, `sd`, `lwu` at XLEN=64 |
| SYSTEM (Zicsr) | `csrr* ud, csr, us1` |
| OP-FP `fmv.d.x` / `fmv.x.d` encodings | `u.mvx ud, xs1` (first active lane) / `x.mvu xd, us1` (broadcast): the `f` field names a `u` register, the `x` field stays `x` |
| custom-0: `tmc`, `wspawn`, `bar`, `bar.arrive`/`bar.wait`; `tcu.ld`; TMEM `ld`/`st`/`alloc`/`dealloc`; `tc.idesc`/`tc.commit`/`tc.cp`/`tc.mma` | every integer operand (and an integer destination) is a `u` register; FP operands stay `f`. These read one value per warp already, so the `u` form only changes where it comes from. The others (`split`, `pred`, `vote`, DXA issue, which reads each lane's operands) stay per-thread |

### Mode `01`: uniform operand

The second source is uniform, as RVV's `vop.vx vd, vs2, rs1` computes
`vs2 op rs1`.

| Opcode | Meaning |
|---|---|
| OP (incl. M, Zicond); OP-32 at XLEN=64 | `xd = xs1 op us2`: `i * stride`, `x << shift`, `x - base`, `i < n` |
| OP-IMM; OP-IMM-32 at XLEN=64 | `xd = us1 op imm` (the only source is uniform); `addi xd, us1, 0` is a broadcast |
| BRANCH | `xs1 cmp us2`: `blt i, n` |
| LOAD, LOAD-FP (`flw`, `fld`) | `xd = mem[us1 + imm]`, `fd = mem[us1 + imm]`: every lane reads the same address |
| STORE | `mem[xs1 + imm] = us2` |

### Mode `00`: uniform first operand

The first source is uniform, for the operations that do not commute
(`u - x`, `u << x`, `u < x`) and for a uniform base address.

| Opcode | Meaning |
|---|---|
| OP (incl. M, Zicond); OP-32 at XLEN=64 | `xd = us1 op xs2` |
| BRANCH | `us1 cmp xs2` |
| STORE, STORE-FP (`fsw`, `fsd`) | `mem[us1 + imm] = xs2`, `mem[us1 + imm] = fs2` |

LOAD stays reserved in mode `00`, so the all-zero word is still illegal.

## Semantics

- `u0`–`u31` per warp, XLEN bits; `u0` reads zero.
- A uniform operand of a per-thread instruction is broadcast to every lane.
- A uniform destination takes the first active lane's result, the rule
  Vortex already uses for CSR writes and branch direction; `u.mvx` is the
  move that relies on it.

## ABI

- Nothing is preset at warp launch, so the hardware has no launch-time write
  port into the file. The CTA trampoline reads the argument block with a
  uniform CSR read (`csrrs u10, mscratch, u0`), and a `__kernel` entry takes it
  in `u10`.
- `u1`–`u15` caller-saved, `u16`–`u31` callee-saved. A `u` register spills to
  the thread's stack with a mode-`01` store and reloads with a load and
  `u.mvx`.

## Implementation

A uniform instruction computes on the per-thread datapath with its uniform
sources broadcast, so it needs no datapath of its own: the uniform file, the
decode of the two modes, the single-lane access of a uniform load or store and
the first-lane writeback are the new parts.

### Hardware

| Area | Files | Change |
|---|---|---|
| Config | `VX_config.toml` | `VX_CFG_EXT_U_ENABLE` (exclusive with C) |
| Register classes | `VX_gpu_pkg.sv` | `REG_TYPE_U`, a third register type in the register id |
| Decode | `VX_decode.sv` | mode from `[1:0]`; operand classes per the tables |
| Fetch | `VX_warp_sched.sv` | 4-byte PC steps whatever `[1:0]` is, without C |
| Uniform register file | `VX_opc_unit.sv` | per-warp 32 x XLEN bits, its own read port, broadcast to the lanes |
| Scoreboard | `VX_scoreboard.sv` | the uniform registers' busy bits |
| LSU | decode | a mode-`10` access narrowed to the first active lane |
| Writeback | `VX_opc_unit.sv` | `u` destination from the first active lane |

### SimX

`decode.cpp` (modes), `types.h` (`RegType::Uniform`), the `u` file in
`opc_unit.cpp`, the single-lane access in `lsu_unit.cpp`, and the
4-byte PC step in `core.cpp`.

### Toolchain

- Vortex LLVM, through `vortex-toolchain-prebuilt`: a `+xvortexu` feature
  that changes instruction-length decoding; `u` registers in the assembler and
  disassembler. Code generation follows later.
- `sw/kernel/include/vx_uniform.h`: inline-assembly intrinsics on `.insn`, so
  they build with the current toolchain.

## Later: a 64-bit base at XLEN=32

An even-aligned pair of uniform registers holds a 64-bit base, and a
per-thread register adds its offset: `addr = {u[n+1], u[n]} + zext(xs2)` on a
reserved mode-`01` encoding (the AMO opcode). The per-thread datapath stays
XLEN wide; the address path below the add, the caches and the memory bus carry
the wider physical address. It lands with that wider address path.

## Phases

Built with dynamic warp allocation; see the combined phases in
[dynamic_warp_allocation_proposal.md](dynamic_warp_allocation_proposal.md#phases).

## Open questions

- 32 or 16 `u` registers.
- Whether the `u` file is also allocated per kernel, as the `x`/`f` file is.

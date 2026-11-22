# Per-Operand Register Windows

Status: SimX, LLVM and RTL prototype with measurements (branch
`feature/register-windows`; LLVM branch `feature/register-windows` on the fork).

## Goal

Give each thread up to 256 registers without widening the 32-bit RISC-V
instruction, and keep the instruction stream as dense as today's. Register-heavy
kernels (FlashAttention softmax, GEMM epilogues, FFT) spill or run small tiles
today; more registers per thread trade occupancy for fewer spills and larger
tiles.

## Design

### One register file (Zfinx)

FP instructions read and write the x registers (Zfinx; no Zdinx, since RV32
Vortex has no D). There are no f registers, FP loads and stores, or FP/integer
moves. `VX_CFG_EXT_ZFINX_ENABLE` selects it; F stays the default. One file means
one register budget per kernel instead of two padded halves, and one class for
the compiler.

### Windows

Each warp holds `W = {W.rd, W.rs1, W.rs2, W.rs3}`, 3 bits per instruction field
(12 bits per warp). A register operand in field `p` with 5-bit value `f` names

```
reg = (f == 0) ? x0 : (W[p] << 5) | f
```

The field is the bit position the operand is encoded in (`rd` [11:7], `rs1`
[19:15], `rs2` [24:20], `rs3` [31:27]), not its role: `amocas` reads its `rd`
field through `W.rd`. Field value 0 is x0 in every window, so x32, x64, ... are
never reachable. `VX_CFG_NUM_RV_REGS` (32, 64, 128 or 256) sets the registers
per thread; a window beyond it is illegal.

### `vx_rwin`

| Instruction | opcode | funct7 | rd | rs1 | rs2 | funct3 |
|---|---|---|---|---|---|---|
| `vx_rwin w.rd, w.rs1, w.rs2, w.rs3` | custom-2 `0x5B` | `0x01` | `W.rd` | `W.rs1` | `W.rs2` | `W.rs3` |

The windows are immediates in the low 3 bits of each field (upper 2 bits zero);
nothing is read from a register, so decode applies `vx_rwin` without a hazard.
`funct7 0x00` stays the TFU.

### State lifetime

- **Launch.** Zero when a warp is activated (CTA dispatch, `wspawn`).
- **Calls, traps, divergence.** Zero at function entry and return, at `ecall`
  and `ebreak` (Vortex traps are synchronous only), and at `vx_split`,
  `vx_split_n`, `vx_join`, `vx_pred`, `vx_pred_n` and `vx_tmc`. The compiler
  guarantees it; hardware saves nothing. Vortex runs the two sides of a split in
  turn, so with zero state at split and join one copy per warp suffices.
  Asynchronous preemption would save the 12 bits with the register file.
- **Implicit FP operands.** TCU fragments and the RTU ray window move to the
  fixed block x32 up under Zfinx (they used f0 up).

### Pipeline: fused fetch

Decode applies the windows, so everything after decode sees 8-bit register
numbers. `vx_rwin` takes no issue slot. Fetch reads 8-byte words: when a
`vx_rwin` is the first word, decode applies it and decodes the second word in
the same step, so the pair costs one fetch, like one 64-bit instruction, while
the code keeps 32-bit density. A `vx_rwin` in the second word costs its warp a
fetch round trip, which fewer warps (larger register budgets) hide less well.
Decode applies each field's window as it extracts the field, beside the opcode
decode, so the windows add no logic depth after it.

Per-warp state grows by 12 bits. Register numbers widen from 6 to 8 bits through
the pipeline; a tag-based scoreboard keeps its cost independent of the register
count.

## Compiler (LLVM)

- **Registers.** x32 to x255 are added to the GPR class (and to the Zfinx f16
  and f32 sub-register classes) and reserved unless `-riscv-vortex-max-regs`
  allows them; x32, x64, ... are always reserved. Inline asm and the operands of
  the divergence instructions stay in x1 to x31. With the default of 32,
  generated code matches the shipped compiler apart from register-choice noise
  in two of six kernels checked (same instruction and spill counts).
- **Window placement** (post-RA, after pseudo expansion; branch relaxation runs
  again after it). Each instruction's requirement per field is found by encoding
  the instruction with probe registers. Placement is exact within a block
  (dynamic programming over the window states); across blocks a forward dataflow
  fixes entry states, and a block with one successor prefers to end in the state
  its successor wants, which moves loop-entry switches into the preheader. A
  final check replays every block.
- **Reordering** (post-RA, in the same pass). Within a region between calls,
  divergence instructions and other zero-window points, a list scheduler takes
  the earliest ready instruction the current windows already satisfy. A phase
  whose operands span several blocks then switches once per block, not once per
  element.
- **Level-order scheduling** (pre-RA). Unrolled code holds many independent
  copies of one computation; scheduling one step of every copy before the next
  keeps each field in one block across a phase. It yields when the register file
  is three quarters full. It also cuts spills in register-rich code under any
  encoding (on the epilogue, 65% fewer instructions than LLVM's default
  schedule), so the 64-bit comparison uses it too.
- **Allocation.** Plain low-first. Block-choosing allocation orders and hints
  were tried (`-riscv-vortex-window-hints`) and never helped: on the epilogue
  they doubled the window switches.

## Evaluation

SimX, one core, 4 threads per warp, issue width 1, 16 KB I-cache. Equal
register-file bytes: F (32 x + 32 f) and 64 registers run 16 warps, 128
registers 8 warps, 256 registers 4 warps. Each kernel exposes the tile it holds
in registers (`rp_softmax` CHUNK, `rp_epilogue` CHUNK, `rp_fft` RADIX); every
model is reported at its best tile.

| Kernel | What lives in registers |
|---|---|
| `rp_softmax` | FlashAttention-style online softmax: a chunk of a score row, the running max and sum, unnormalized P; the output row (as tensor memory holds it) is rescaled per chunk, so larger chunks rescale less |
| `rp_epilogue` | GEMM epilogue on a 128-column accumulator row: bias, GELU, row absmax, int8 quantization; a row held in registers takes one pass instead of two |
| `rp_fft` | batched 4096-point FFT, one transform per thread, interleaved batch; a larger radix means fewer passes over memory |

The 64-bit encodings are modelled with a one-instruction prefix (`vx_rpfx`,
study only) fetched together with its instruction: "mixed 32/64-bit" prefixes
only instructions that name a register above x31, "all 64-bit" prefixes every
instruction (`-riscv-vortex-rwin-mode=1/2`). Code layout and I-cache behaviour
are therefore real.

### Speedup over F's best

Windows use the fused fetch the RTL implements (an 8-byte fetch word: a
`vx_rwin` in the first word fuses with the second). "Unfused" and "line fused"
(a `vx_rwin` fuses with the next word anywhere in the 64-byte line) bound it.

| Registers (warps) | Model | Softmax | Epilogue | FFT | Raycast |
|---|---|---|---|---|---|
| 64 (16) | windows | 1.082 | **1.228** | 1.008 | 0.999 |
| | windows, unfused | 1.083 | 1.227 | 0.941 | 0.835 |
| | windows, line fused | 1.082 | 1.227 | 1.007 | 1.036 |
| | mixed 32/64-bit | 1.075 | 1.226 | 1.007 | 1.036 |
| 128 (8) | windows | **1.087** | 1.203 | 1.010 | 0.658 |
| | mixed 32/64-bit | 1.086 | 1.198 | 1.009 | 0.703 |
| | all 64-bit | 1.085 | 1.195 | 1.010 | 0.679 |
| 256 (4) | every model | 0.91 to 0.93 | 0.90 to 0.96 | 0.72 to 0.81 | not run |
| 32 (32) | Zfinx without windows | 0.994 | 1.026 | 0.957 | not run |

### Instruction bytes relative to windows (same budget)

| Registers | Model | Fetched bytes | Code (text) |
|---|---|---|---|
| 64 | mixed 32/64-bit | +3% to +11% | +5% to +11% |
| 128 | mixed 32/64-bit | +8% to +35% | +8% to +36% |
| 128 | all 64-bit | +33% to +74% | +37% to +74% |

With an 8 KB I-cache, all-64-bit softmax loses 4.9% of its cycles to misses;
windows and the mixed encoding lose under 0.5%. The epilogue streams more code
than either cache holds and FFT fits in both, so neither changes.

### Findings

1. More registers pay up to 128 per thread: up to 1.23 times F at equal
   register-file bytes. At 256, four warps cannot hide latency under any
   encoding; the latency-bound raycast already loses at 128.
2. With fused fetch, windows match the 64-bit encodings within 1% on softmax,
   epilogue and FFT at 64 and 128 registers, from 8% to 74% fewer fetched bytes
   and 5% to 74% less code.
3. Irregular code (raycast) switches windows often: with the 8-byte fetch word
   about half of its `vx_rwin` fuse and windows trail the 64-bit encodings by
   3.6% at 64 registers; line-granular fusion (a two-bank fetch) closes it.
   Moving `vx_rwin` to even words in the compiler was tried and does not help:
   the instruction before a `vx_rwin` almost always needs the old window.
4. Fusing about half the `vx_rwin` (8-byte word) gives the same cycles as fusing
   almost all of them on the three phase-structured kernels.
5. The compiler matters more than the encoding: level-order scheduling is worth
   up to 30% on the epilogue under every encoding; block-choosing allocation
   hints never helped and are off.

### RTL

Implemented in `hw/rtl` (decode, fetch, warp scheduler): Zfinx as one register
type, windows applied as each field is extracted (in parallel with opcode
decode), `vx_rwin` consumed in decode with its instruction-buffer slot
released, and the 8-byte fetch word with fusion. The scoreboard, register file
and operand collector take the wider register numbers unchanged; the register
file grows with the register count (dynamic warp allocation is not merged here).

- **Correctness and parity.** The `rwin` test and the Zfinx `rp_softmax`,
  `rp_epilogue` and `rp_fft` pass in rtlsim, with the same instruction counts as
  SimX and cycles within 0.0002% to 2.3%.
- **Cost** (Beacon, ASAP7, core DUT at 8 warps, against master, 64 registers so
  the register file matches F's 32 x + 32 f): Fmax 1,052 MHz and the deepest path
  (30 stages) unchanged, no register deeper, cells +0.01%, flip-flops +98 (12
  window bits per warp). Applying the windows after the opcode decode instead
  made the instruction-buffer inputs 6 to 10 stages deeper.
- **At 256 registers** the depth is still unchanged; cells grow 6.2% and
  flip-flops 2.2%, mostly the per-warp scoreboard bit vector (64 to 256 bits per
  warp). A tag-based scoreboard removes that growth.

### Limits

- Four kernels, one core configuration, SimX timing (RTL parity on small runs).
- The fetch-energy advantage depends on the I-cache: from 0% to 10% if each
  access reads a whole line, to 8% to 74% if it reads only the instruction's
  bytes. The characterized 28 nm macros do not cover a 16 KB I-cache.
- The 64-bit models give the wider encoding only its register reach, not
  predication or 32-bit immediates.
- The TMEM FlashAttention kernel is not included: its register-group TMEM loads
  are inline asm pinned to f registers and need compiler intrinsics before
  windows can place them.
- 256-register runs of the 64-bit models on the epilogue and FFT used LLVM's
  default scheduler.

## Plan

1. Rebase on dynamic warp allocation, so a kernel's register count sets its
   warps; a tag-based scoreboard for 256 registers.
2. TMEM intrinsics with register-group operands, then the TMEM FlashAttention
   kernel.
3. A `model_parity` case for windows and fused fetch.
4. Write-up.

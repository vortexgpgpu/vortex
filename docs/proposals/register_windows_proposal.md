# Per-Operand Register Windows

Status: draft, SimX prototype in progress (branch `feature/register-windows`).

## Goal

Give each thread more than 32 integer and 32 FP registers, up to 256 per
class, without widening the 32-bit instruction. Register-heavy kernels (tensor
tiles, FlashAttention) spill today; more registers trade occupancy for fewer
spills.

## Why not the alternatives

- **Wider register fields.** With a 7-bit opcode, 25 bits are left; four 8-bit
  register fields do not fit, and three leave one bit of function code.
- **Prefix only.** A 32-bit prefix carrying 3 high bits per operand works and is
  stateless, but a loop that uses high registers on every instruction doubles
  in size and in fetch slots.
- **One window for all operands (SPARC-style).** One base shared by every
  operand cannot combine registers from two blocks in one instruction. In
  `acc += a * b`, with accumulators and inputs in different blocks, every FMA
  needs a move or a window switch.

## Design

Each warp holds a window state `W = {W.rd, W.rs1, W.rs2, W.rs3}`, 3 bits per
operand position (12 bits per warp). For a register operand encoded in field
position `p` with 5-bit field value `f`:

```
reg = (p is an integer operand && f == 0) ? x0 : (W[p] << 5) | f
```

The position is the bit field the operand is encoded in (`rd` [11:7],
`rs1` [19:15], `rs2` [24:20], `rs3` [31:27]), not its role. An instruction that
reads its `rd` field as a source (`amocas`) uses `W.rd` for it.

`x0` stays the zero register in every window, so `x32`, `x64`, ... are not
addressable as integer registers. FP registers have no such exception.

`VX_CFG_NUM_RV_REGS` (32, 64, 128 or 256) sets the registers per class. A window
value with `(W[p] << 5) >= VX_CFG_NUM_RV_REGS` is illegal.

### Instructions

`vx_rwin` uses custom-2 (`0x5B`) with `funct7 0x01`; `funct7 0x00` stays TFU.
The window for each field is written into that same field:

| Instruction | funct7 | rd | rs1 | rs2 | funct3 | Effect |
|---|---|---|---|---|---|---|
| `vx_rwin`  | `0x01` | `W.rd` | `W.rs1` | `W.rs2` | `W.rs3` | set the window state |

The upper two bits of the `rd`, `rs1` and `rs2` fields are reserved and must be
zero. In assembly the window values are written as literal register names:
`.insn r 0x5B, w3, 1, x<w0>, x<w1>, x<w2>`.

There is no one-instruction prefix: it would double the per-warp state, and
a switch and switch-back costs only two fetch slots (see Pipeline).

### State lifetime

- **Launch.** The window state is zero when a warp is activated (CTA dispatch,
  `wspawn`), so code that never uses the extension sees no change.
- **Traps.** Vortex traps are synchronous only (`ecall`, `ebreak`), so the
  window state must be zero at any instruction that can trap or enter a
  callback; the compiler guarantees it as at calls. Hardware saves nothing.
  Asynchronous preemption would save the window state with the register file,
  through a CSR.
- **Divergence.** The state is not saved on the IPDOM stack. Software rule: at a
  `vx_join`, the window state must equal the state at the matching `vx_split`.
  Vortex runs the two sides of a split one after the other, so with that rule a
  single copy per warp is enough. Independent thread scheduling, or running
  both sides concurrently, would need a copy per path, like the PC.
- **Calls.** ABI rule: the window state is zero at function entry and return.

### Macro-ops with register groups

Instructions that expand to consecutive registers (`vx_flw.d/.q`,
`vx_fsw.d/.q`, matrix fragments) add the offset inside the 32-register window
of the base register: `(base & ~31) | ((base + k) & 31)`.

### Pipeline

The window state is applied in decode, which sees each warp's instructions in
program order, so everything after decode works on 8-bit register numbers and
`vx_rwin` needs no serialization. Decode consumes `vx_rwin`: it takes a fetch
slot but no instruction-buffer entry or issue slot.

Per-warp state grows by 12 bits.
The scoreboard and register-file addresses widen to 8-bit register numbers; a
tag-based scoreboard keeps scoreboard cost independent of the register count.

## Compiler

LLVM sees `NUM_REGS` registers per class. A post-RA pass, structured like
`RISCVInsertVSETVLI`, computes the required window state per instruction with a
forward dataflow over the CFG, inserts `vx_rwin` where the state changes
(hoisted out of loops), and enforces the split/join, call and trap rules. Correctness depends only on this pass; allocation
hints that keep each loop's operands in consistent blocks reduce the number of
inserted instructions.

## Plan

1. SimX: config, decode, window state, tests with hand-encoded instructions.
2. LLVM: register classes, MC encoding, insertion pass; measure inserted
   instructions and code size on register-heavy kernels.
3. Allocation hints.
4. RTL and SimX/RTL parity, if step 2 shows the overhead is small.

## Open questions

- Whether `rs1` and `rs2` need separate windows (two windows, 6 bits, would
  halve the state if they rarely differ).
- CSR access to the window state for debuggers and context switch.

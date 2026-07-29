# Barrier Extension Isolation

## Scope

Vortex keeps the existing hard barrier as the default implementation.  The
software barrier and memory barrier are experimental alternatives selected at
compile time:

- `VX_CFG_EXT_SBAR_ENABLE`: a shared-memory software barrier using LMEM AMOs.
- `VX_CFG_EXT_MBAR_ENABLE`: a custom-instruction memory barrier unit.
- `VX_CFG_DXA_SBAR_ENABLE`: DXA completes a software barrier transaction.
- `VX_CFG_DXA_MBAR_ENABLE`: DXA completes a memory barrier transaction.

The software and memory barrier extensions are mutually exclusive.  A DXA
completion mode requires its corresponding barrier extension and the DXA
extension.  Invalid combinations fail during configuration/compilation.

## Hard Barrier Boundary

The optional implementations do not change `VX_bar_unit.sv`, `txbar_t`, or the
hard-barrier DXA completion path.  With both optional extensions disabled, the
generated RTL and SimX module graph are the existing hard-barrier design.

DXA completion is selected at compile time:

- default: the existing completion FIFO emits `txbar_t` events;
- software barrier: a separate completion path performs one AMO decrement on
  the barrier object's transaction counter;
- memory barrier: a separate completion port updates the mbarrier object.

There is no run-time "combined" barrier kind.

## Software Barrier

The object is ordinary, readable shared-memory state:

```c++
struct soft_barrier_state {
  volatile uint32_t phase;
  volatile uint32_t arrivals;
  volatile uint32_t pending_transactions;
};
```

One lane per participating warp atomically increments `arrivals`.  The CTA
leader polls `arrivals` and `pending_transactions`; after both reach their
completion values, it resets `arrivals` and release-stores the next phase.
Every waiter polls `phase`.  The C++ implementation uses ordinary lane
conditionals and compiler atomics, with no TMC, split/join, inline assembly, or
software fence stack.

The caller publishes initialization with an existing CTA hard barrier before
first use.  `expect_tx` must precede the final arrival for a phase.

## Memory Barrier

The mbarrier object is one aligned 32-bit backing word containing phase,
pending arrivals, expected arrivals, and pending transactions.  The public C++
API validates arguments and delegates only the four hardware operations to
intrinsics: init, arrive, expect transaction, and wait.

The RTL is compiled only with `VX_CFG_EXT_MBAR_ENABLE`.  It uses a small
write-through state cache and a per-warp waiter CAM.  A phase transition
matches and releases waiters without a serial all-warp scan.  Instruction and
DXA completion inputs are separate ports, so an accepted instruction cannot be
mutated or revalidated because of a completion arriving later.

SimX communicates completion and warp release only through channels.

## Verification

The review gate checks:

1. `VX_bar_unit.sv`, `VX_dxa_completion.sv`, and `txbar_t` are byte-identical
   to the shared-atomic baseline.
2. Hard-only builds pass their existing barrier and DXA tests.
3. Software and memory barrier builds pass lifecycle, visibility, repeated
   phase, delayed completion, and DXA completion tests.
4. SimX and RTL execute matching barrier semantics before performance sweeps.

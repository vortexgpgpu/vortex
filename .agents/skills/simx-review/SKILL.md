---
name: simx-review
description: Review checklist for Vortex SimX changes under sim/simx and sim/common. Use together with code-review whenever a diff touches the cycle model, the functional emulator, the SimObject framework or a model of an RTL unit.
---

# Reviewing SimX

Load `docs/coding_guidelines_cpp.md`, `docs/designs/simobject.md`,
`docs/designs/simx_simulator_architecture.md` and AGENTS.md §4–§6 first.

## The model's role

- SimX is the RTL's timing model and the functional reference. A change to
  either side that moves cycles needs the other side, or a `model_parity` case
  showing the gap stays within tolerance (AGENTS.md §4).
- A SimX-vs-RTL divergence is not fixed by editing SimX until it is shown which
  side is wrong (`docs/bug_fixes.md` §5). A SimX timing change must describe
  hardware the RTL builds or will build, not a convenience.
- Instruction counts must match rtlsim exactly; any change to what retires
  (decode, traps, warp control, barriers) is functional, not timing.

## Structure

- **Cardinal rule**: modules talk only through their bound `SimChannel`/
  `SimEventLink` ports. No reaching across the hierarchy
  (`core_->processor()->...`, a leaf unit grabbing global memory).
- No threading vocabulary in model code (atomics, mutexes, threads, deferred
  calls); `ci/check_simx_mt_boundary.sh` enforces it, and the model must stay
  correct under serial and parallel execution.
- Determinism: no reseeding of a global RNG, no iteration over unordered
  containers that decides timing, same-cycle events delivered in send order.
- Configuration comes from the generated `VX_config.h`; never inject
  `-DVX_CFG_*` in a Makefile to work around a stale header (AGENTS.md §2).

## Correctness

- Queues: a bounded queue models the RTL depth and applies backpressure; an
  unbounded `std::queue` where the RTL has a FIFO hides a stall.
- Lifetimes: a packet or request referenced after `send` or after its owner
  retires; iterator invalidation in a container that a callback can modify.
- Width and sign: XLEN 64 addresses in 32-bit fields, `int` loop counters over
  `uint64_t` sizes, shifts by `>= width`.
- Perf counters: a counter SimX reports must count the same event as the RTL's
  counter of the same name, or `model_parity` counter checks compare different
  things.

## Style

- Braces on every branch and loop; two-space indent.
- `DT`/`DP` trace arguments comma-separated.
- No comment that names an RTL signal or parameter, a task, a PR or a fix.

## Evidence

- A timing change names the parity case before and after, and the perf baseline
  diff if `perf_gate` moves (re-recorded by a human with `--update-baselines`,
  never by an agent or CI).
- A functional change names the regression or ISA test that covers it.

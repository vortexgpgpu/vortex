# TSPLIT (SCS) RTL Design Review

**Module:** [hw/rtl/core/VX_scheduler.sv](../../hw/rtl/core/VX_scheduler.sv) under
`VX_CFG_DIVERGE_TYPE_SPLIT` · **Config:** NT=8/NW=8, U55C, 300 MHz target
**Status:** validated (rtlsim 10/10; routed scheduler DUT closes 300 MHz)

## 1. What TSPLIT is

TSPLIT (schedulable convergence stack, "SCS") keeps the baseline IPDOM split/join model
but makes each masked-off subgroup *independently schedulable* instead of stack-suspended.
A divergent `vx_pred` no longer drops the not-taken lanes on the floor: it **parks** them
as a runnable split, and `vx_yield` rotates a spinning subgroup out so a lock holder in a
sibling subgroup can run. This is what buys forward progress on the lock benchmarks, which
the strict-LIFO baseline deadlocks on.

## 2. Storage — already memory-first

The design was built to the BRAM-first rule from the start; the audit confirms it needs no
change:

| Structure | Where | Size (NT=8/NW=8) | Rationale |
|---|---|---|---|
| Parked-split **pool** `{tmask, pc}` | **BRAM** (`VX_dp_ram`, `cs_pool_ram`) | `NW × 2NT` entries | The one large structure; a per-warp round-robin FIFO. 1W1R, addressed `{wid, slot}`, registered read → the 1-cycle pop. |
| Pending-split scratch (`cs_ptmask/cs_ppc/cs_done/cs_inpool/cs_head/cs_cnt`) | flops | ~520 bits | Per-warp, mask-scanned every cycle — same storage class as the baseline's own `warp_pcs`/`thread_masks`. |

The pool is deliberately off the warp-PC critical path: a push writes the tail
combinationally, a pop registers the head read and installs `{tmask, pc}` one cycle later
(the warp is parked meanwhile). An earlier revision that read the pool combinationally out
of a large FF array is what failed 300 MHz; moving it to BRAM with the pipelined pop closed
timing.

## 3. Critical-path shape

The schedule path is baseline-shaped: `ready_warps = active & ~stalled`, priority-encode,
read `{thread_masks, warp_pcs}` for the selected warp. The SCS additions
(pend/pool bookkeeping) sit in the warp-state `always @(*)` next to the existing
TMC/branch/split writers and do not deepen the schedule cone. `VX_split_join` (the IPDOM
stack) is retained — TSPLIT is a *superset* of the baseline stack, not a replacement.

## 4. Measured cost (routed scheduler DUT, U55C, 300 MHz)

| scheduler DUT | LUT (logic+mem) | FF | BRAM | WNS | Fmax |
|---|---|---|---|---|---|
| DEFAULT (baseline IPDOM) | 4,019 | 4,009 | 0 | +0.350 | 335 MHz |
| **TSPLIT** | **6,340** (1.58×) | 4,525 (+516) | 1 | +0.323 | 332 MHz |

TSPLIT's forward-progress capability costs ~1.6× scheduler LUTs and one BRAM over the
baseline, closes 300 MHz with margin, and adds no divergence-state flop array (the pool is
in BRAM). At the full-core level the delta shrinks to +2.6% LUT / +276 FF.

## 5. Correctness envelope

- Bit-inert fences: a SPLIT build is code-identical to the pre-switch HEAD (mechanical
  ifdef resolution); the SCS logic is exactly this branch's validated behavior.
- rtlsim: all ten evaluation kernels pass, including the three lock kernels the baseline
  and barriers-only ITS deadlock on.
- The two subtle SCS invariants that took bring-up to find (documented in the module):
  pending-mask **accumulation** across staggered loop exits, and reconvergence reabsorbing
  a parked split **only when its resume PC matches** (`cs_ppc == warp_pcs`), so an inner
  loop never steals an outer loop's parked lanes.

## 6. Verdict

TSPLIT is production-shaped: memory-first storage, baseline-shaped schedule path, 300 MHz
closure, and the only arm of the three that passes every benchmark while adding modest
area. No redesign warranted.

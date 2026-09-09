# NVIDIA-style ITS RTL Design Review

**Module:** [hw/rtl/core/VX_scheduler.sv](../../hw/rtl/core/VX_scheduler.sv) under
`VX_CFG_DIVERGE_TYPE_NV_ITS` (+ `VX_CFG_SCS_YIELD_ENABLE`) · **Config:** NT=8/NW=8, U55C
**Status:** validated (rtlsim 10/10 under ITS+Y; ablation reproduces the barriers-only
deadlocks). Storage redesigned memory-first (v2 group-cache engine).

## 1. What ITS is

Independent Thread Scheduling (US 10,067,768 B2): threads carry **per-thread PCs** and
reconverge through **convergence barriers** (`vx_bar_add` / `vx_bar_wait`) instead of a
divergence stack. The scheduler issues the group of runnable threads sharing the **lowest
pending PC**, which drains loops and reconverges naturally. Forward progress on blocking
loops comes from the **Yielded** thread state (`vx_yield`): a yielded thread leaves the
schedulable set and is excluded from barrier release, so a spinner cannot starve a lock
holder. This is the single feature added beyond the reference barriers-only arm; it is the
minimal A100-equivalent (released Volta/Ampere expose YIELD; mainline GPGPU-Sim/Accel-Sim
model no ITS microarchitecture at all).

## 2. The storage problem, and the v2 fix

**v1 (reference-shaped) was non-conformant.** All 2,944 bits of divergence state
(`thread_pcs` 1,920 + barrier masks 1,024) were flops with full next-state mux fabric, and
the min-PC tournament tree was replicated per warp and sat combinationally in the schedule
path. Routed cost: **17,092 logic LUTs, WNS −0.725** — 4.3× the baseline scheduler and off
300 MHz.

**v2 is a group-cache engine.** Key observation: per-thread PCs are only *needed* when a
warp's runnable group changes, and at most one group-changing event per warp is in flight
(the warp is schedule-stalled from the causing instruction until its resolution). So:

| Structure | v1 | v2 | Primitive |
|---|---|---|---|
| Per-thread PCs | `NW×NT×PC` flops, per-warp tree in schedule path | **LUTRAM row store** `NW × NT·PC`, 1W1R | `VX_dp_ram(LUTRAM)` |
| Barrier `{participate, arrived}` | `2·NW·NBAR·NT` flops + global runnable scan | **LUTRAM row store** `NW × 2·NBAR·NT` | `VX_dp_ram(LUTRAM)` |
| Group cache `{grp_pc, grp_mask}` | — (recomputed every cycle) | **registered**, read by the schedule path exactly like baseline `warp_pcs`/`thread_masks` | flops |
| min-PC tree | replicated ×NW, in schedule path | **one shared instance** in the event-service datapath | — |
| Event serialization | — | 4-way RR arbiter + per-class LUTRAM FIFOs | `VX_fifo_queue(LUTRAM)`, `VX_rr_arbiter` |

The schedule path becomes baseline-shaped: `ready = active & ~stalled & ~grp_stale &
(grp_mask != 0)`, priority-encode, read the registered `{grp_pc, grp_mask}`. The min-PC
tree leaves the schedule path entirely — it runs once, in the service datapath, off
registered event payloads into registers.

## 3. The regroup engine

One event is serviced per cycle. Sources, arbitrated round-robin: (a) branch resolution
(`NUM_ALU_BLOCKS==1`), (b) wctl (`bar_add`/`bar_wait`/`yield`/`tmc`), (c) warp init
(CTA/wspawn), (d) yield wake. The event warp stays schedule-stalled until served, so each
source has at most `NUM_WARPS` events outstanding — per-source 1W1R LUTRAM FIFOs of that
depth absorb collisions, and a live event is served directly when its class is granted and
its FIFO is empty.

**The service is a 2-stage pipeline** (`st1_*` → registered `s1_*` → install), because a
single-cycle service — LUTRAM row read → event apply → barrier release → min-PC tree →
group install — routed at WNS −6.474 ns (a 36-level cone from `amask` through the RAM read
and comparator tree into `thread_masks`). Stage 1 reads the rows, applies the event, and
re-evaluates release, producing the post-event per-thread PCs and runnable set (and
committing the row writes). Stage 2 runs the min-PC tree over the registered stage-1 result
and installs the group cache. The serviced warp is schedule-stalled across both stages, so
the extra cycle is hidden behind the other warps — measured cost is nil: lockht is 3,375
cycles pipelined vs 3,380 single-cycle. A collision costs the losing warp +1 cycle, within
SimX parity tolerance.

**Two hazards found by SimX-as-oracle trace diff and fixed:**

1. **bar_add must not regroup.** `bar_add` changes no thread's PC; the running group's row
   entries are stale (their truth is `grp_pc`). A tree pass on `bar_add` mis-steers the
   warp — it kept `srv_keep_grp` and preserves the current group instead.
2. **yield→wake row read-after-write.** A full-warp yield writes the resume PC into the row
   RAM and parks the group; the separate wake event fired the next cycle and read the row
   RAM before that write committed, waking threads at a stale PC (lockht misaligned load).
   The wake is folded into the yield service so the just-yielded group's fresh resume PC
   comes from the row **bypass** (`srv_wpcs`), never a racing next-cycle read.

## 4. Constraints (hard elaboration guards)

- `NUM_ALU_BLOCKS == 1` (single branch event source).
- `NUM_ALU_LANES == NUM_THREADS` (per-thread branch masks are meaningless under lane
  blocking).
- Incompatible with `EXT_C` (the RVC decode-advance path bypasses the group-PC advance)
  and with graphics extensions (SFU opcode overlap).

## 5. Cost — v1 vs v2 (routed scheduler DUT, U55C)

| scheduler DUT | LUT (logic+mem) | FF | WNS | note |
|---|---|---|---|---|
| DEFAULT | 4,019 | 4,009 | +0.350 | |
| SCS | 6,340 | 4,525 | +0.323 | |
| NV_ITS v1 (flop state, per-warp tree) | 17,408 | 7,451 | −0.725 | reference-shaped |
| NV_ITS+Y v2 (group-cache, 1-cycle service) | 7,686 | 4,249 | −6.474 | area met, timing broke |
| **NV_ITS+Y v3 (group-cache, pipelined)** | **7,393** | 4,565 | **−1.579** | shipping |

The area target was met decisively: **17,408 → 7,393 LUTs (2.35× smaller)** and **7,451 → 4,565
FFs**, with all the divergence state now in distributed LUTRAM (0 BRAM, 815 LUT-as-memory). The
single-cycle service datapath, however, routed at −6.474 ns; the 2-stage pipeline recovered it to
−1.579 ns. The residual is 64% routing and standalone-DUT-placement-bound — the authoritative
figure is the full core, where the scheduler places densely. Whatever residual delta remains over
SCS is the genuine, irreducible ITS cost — per-thread PC storage and the barrier machinery —
now paid in the right primitive rather than in LUT fabric.

## 6. Verdict

ITS is functionally complete for the benchmark suite (10/10 under ITS+Y) and its storage is
now memory-first, matching the baseline's use of RAM for large state. It remains the most
expensive of the three arms in both area and its extra ISA/compiler surface (per-thread
PCs, convergence barriers, the dropped `setRequiresStructuredCFG`), and its lowest-PC
scheduling shows real runtime pathologies on some kernels (softmax group splintering). The
paper's headline — SCS matches ITS's forward-progress capability at a fraction of the
area and ISA cost — holds even after completing the ITS arm.

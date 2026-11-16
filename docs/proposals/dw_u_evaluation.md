# Dynamic warps and uniform registers: evaluation log

Working log for the `dw-eval` branch: is dynamic warp allocation (DW), with or
without uniform registers (U), worth its cost? Design background is in
`dynamic_warp_allocation_proposal.md` and `uniform_registers_proposal.md`.

## Setup

Unless noted: one core, 8 threads per warp, issue width 1, one operand
collector, FDIV/FSQRT PE ratio 8, 300-cycle memory latency (SimX), XLEN 32.

- **S32**: static allocation, 32 warps, full register file (2048 rows).
- **S16**: static allocation, 16 warps, half register file (1024 rows).
- **D32**: dynamic allocation, 32 warps, half register file (1024 rows).
- **D32U**: D32 plus uniform registers (512 uniform rows).

Static runs strip register counts from the kernel (`regs.py zero`); dynamic
runs build with `VX_REG_COMPACT=1`.

Area is ASAP7 logic from Yosys (`stat_lib.rpt`); the register file and other
RAMs are black boxes and are not in these numbers.

## Results before the warp-scaling merge (dw-dev `f6e2e2b76`)

### Same register file: D32 vs S16 (SimX cycles)

| Kernel | S32 | S16 | D32 | D32 speedup over S16 |
|---|---|---|---|---|
| sgemm -n128 | 2,164,969 | 3,390,769 | 2,150,959 | 1.58x |
| relu -n65536 | 336,299 | 461,635 | 338,800 | 1.36x |
| vecadd -n65536 | 334,130 | 451,107 | 335,867 | 1.34x |
| pathfinder -n256 | 766,396 | 953,278 | 763,261 | 1.25x |
| conv3 -n128 | 213,678 | 226,087 | 214,800 | 1.05x |
| stencil3d -n32 | 1,535,512 | 1,571,215 | 1,520,704 | 1.03x |
| sort (1,024 points) | 2,881,181 | 2,884,385 | 2,796,242 | 1.03x |
| bfs -n4096 | 50,317 | 49,506 | 49,995 | 0.99x |
| sgemv -m512 -n512 | 616,714 | 497,120 | 623,118 | 0.80x |

- D32 matches S32 within about 1% everywhere: 32-warp performance from a
  16-warp register file.
- sort sizes its input per hardware thread, so S16 ran `-n8` for the same
  1,024 points.
- sgemv runs faster with fewer warps (memory contention); DW needs a way to
  run fewer warps than fit.

### Cost of 16 more warps (logic area, um2)

| Structure | S16 | D32 | Added |
|---|---|---|---|
| Instruction buffer (queues + uop sequencers) | 7,773 | 16,298 | +8,500 |
| Warp scheduler (trap CSRs, PCs, masks, CTA table) | 5,556 | 11,423 | +5,900 |
| Barrier table (per CTA x barrier, warp masks) | 2,107 | 6,746 | +4,600 |
| CTA context table (SFU) | 3,425 | 6,779 | +3,400 |
| Scoreboard | 3,135 | 6,299 | +3,200 |
| Fetch, dispatch, CTA dispatcher | | | +1,200 |
| **Core logic** | **84,933** | **112,325** | **+27,400 (+32%)** |

About 1,700 um2 per extra warp. The 32-warp static core (113,011) and D32
(112,325) have the same logic: DW itself adds only about 155 um2 in the
operand collector.

### Register-file energy

- DW does not change register-file access counts (sgemm reads: 2,052,192 in
  both S32 and D32), so it barely changes register-file dynamic energy. It
  saves register-file area and leakage.
- An earlier "-28% register-file power / -12% core energy" figure came from
  pricing a 512-word macro beyond the characterized 28 nm library (largest is
  256 words). It is withdrawn.
- U does cut register-file traffic: -11% to -30% register-file energy (28 nm
  macro estimate), but adds about 1,400 um2 of logic (uniform scoreboard tags)
  plus the uniform storage, and slows some kernels (pathfinder +11%, sgemv
  +5% cycles at issue width 1).

### Instruction buffer depth (D32, change in cycles vs depth 4)

| Kernel | Depth 1 | Depth 2 |
|---|---|---|
| sgemm | +15.6% | +1.2% |
| pathfinder | +12.3% | +0.6% |
| vecadd | +6.5% | -2.5% |
| relu | +5.9% | -1.6% |
| stencil3d | +2.8% | +1.9% |
| bfs | +1.7% | +0.5% |
| conv3 | +0.9% | +0.3% |
| sort | +0.3% | -1.0% |
| sgemv | -21.8% | -16.6% |

Depth 2 costs at most 2.5%; depth 1 up to 16%.

## Warp-scaling merge (2026-11-16)

`origin2/feature/warp-scaling` (`267533fed`, based on master) merged into
`dw-eval` as `d78870a0d`. It brings trap CSR sharing, the instruction buffer
in one RAM per issue slice with depth 2, operand collector caps, scoreboard
lookahead and arbiter changes for many warps, per-slice PC targets, and
master up to #477.

Resolution notes:
- **Scoreboard:** the branch's head/next-entry interface is kept; dw-dev's
  uniform tags, the x/f in-use width and the ASIC in-use clock gate are
  carried over. The head's uniform operands are looked up in the tags like the
  next entry's. dw-dev's "look up a staged instruction's operands once, on
  entry" register is dropped: the head now comes from the buffer RAM, and the
  direct lookup gives the same answer.
- **Warp scheduler:** the ASIC state clock gate now also enables on the
  per-slice sequential PC write (`seq_valid`), which replaced the decode-time
  PC advance; without it the gate's assertion fired on the first warp.
- **SimX core:** the branch's version, plus dw-dev's uniform writeback/x0
  rule, the single-lane uniform load/store and the C-disabled instruction
  size. dw-dev's pending micro-op fix is the same as master #476 and takes
  master's form.
- **TCU A/B buffers:** dw-dev (TMEM) already applied the in-row descriptor
  offset that master #475 adds; the B offset now applies without TMEM too, as
  on master.
- **Launch requests:** `kmu_req_t` carries both dw-dev's `num_regs` and
  master's `lmem_slots`.
- **Perf and synthesis baselines:** master's side taken for the conflicted
  JSON files. They are stale for this branch and are not regenerated here.

Smoke check after the merge: sgemm `-n16`, default config, SimX 18,369 vs
rtlsim 18,319 cycles, both pass.

Known issue in the merged `267533fed` (reported by the warp-scaling work):
macro-ops (TRACE, WMMA, packld, AMO fences) expand at the instruction-buffer
input and block decode for the whole issue slice; with IBUF_SIZE=2 and 16 or
more warps per slice, ray tracing deadlocks on SimX and RTL. The fix (buffer
macro-ops whole, expand at issue) is not pushed yet.

### 64 warps after the merge (logic area, um2)

Dynamic warps, 1024-row register file (`ws/run_d64.sh`):

| Structure | S16 (pre-merge) | D32 (pre-merge) | D64 (merged) |
|---|---|---|---|
| Instruction buffer | 7,773 | 16,298 | 12,681 |
| Warp scheduler | 5,556 | 11,423 | 9,160 |
| Scoreboard | 3,135 | 6,299 | 5,768 |
| CTA context table (SFU) | 3,425 | 6,779 | 0 (RAM macro) |
| Barrier table | 2,231 | 6,979 | 23,879 |
| **Core logic** | **84,933** | **112,325** | **118,177** |
| Estimated SRAM | 41,121 | 41,620 | 46,511 |

The barrier table grows with warps squared (an entry per CTA x barrier, each
with a warp mask, and resident CTAs = warps). Without it, 64 warps cost about
+14% over 16; with it, +39%. A count-based barrier table is in progress on the
warp-scaling work. The S16/D32 baselines predate the merge, which also changes
non-warp logic (execute units -2,600).

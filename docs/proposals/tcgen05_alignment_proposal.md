# Aligning the TMEM tensor path with tcgen05

## Why

The TMEM path ([tmem_dataflow_proposal.md](tmem_dataflow_proposal.md)) drives
the tensor core through jobs: a 16-word descriptor posted with two
instructions and run by a per-core sequencer (`VX_tcu_tjob`) that can also
issue the stage DXA loads itself (CLOSED). Measured:

- CLOSED beats warp-issued per-stage jobs (PERSTAGE) by only 1.06-1.10x.
- The sequencer is ~22k LUTs standalone, but CLOSED is only 1.1k LUTs and
  1.6k FFs of it; a tc-only engine is still 20.5k LUTs, mostly the TMEM
  datapath (V80 OOC synthesis). CLOSED's fill counting is the sequencer's
  worst path (25 levels, 5.97 ns; 4.2-4.9 ns without it).
- The tensor core's losses are elsewhere: operand fills (18-43% of cycles) and,
  for attention, softmax latency (FA-4 idle 41%). FA-4's CLOSED jobs wait on
  stages 31% of the time; on `tc.mma` that time is not recovered (below).
- TMEM columns are owned in proportion to the CTA's shared memory, so FA-4
  reserves 16 KB of shared memory it never touches, and one extra column of
  that rule keeps head dimension 64 out of a 32 KB core.

Blackwell has no hardware K loop. One thread issues one asynchronous
`tcgen05.mma` per K step and `tcgen05.commit` arrives on an mbarrier once the
thread's earlier tcgen05 operations complete. AGENTS.md asks for the
mainstream model, so the TMEM path adopts it.

## ISA

`tmem.*` are CUSTOM0, funct7 = 5; `tc.*` are funct7 = 6. `tc.*` operands are
scalars from the lowest active thread; `tc.mma` and `tc.cp` read a third
source register named by the rd field.

| Vortex | tcgen05 | Operands |
|---|---|---|
| `tc.idesc` | (the `idesc` and smem descriptor templates) | slot, idesc (formats, sparse, A in TMEM, `n_frag`, `k_tiles`), leading bytes |
| `tc.mma` | `tcgen05.mma.cta_group::1` | slot, d_taddr, A (smem offset or TMEM column), B smem offset, sparse metadata column, enable_input_d |
| `tc.commit` | `tcgen05.commit.mbarrier::arrive::one` | barrier id |
| `tc.cp` | `tcgen05.cp` | taddr, smem source, stride, columns |
| `tmem.alloc` / `tmem.dealloc` | `tcgen05.alloc` / `dealloc` | columns (power of two, >= 32); first column in rd, ~0 if none fits |
| `tmem.ld` / `tmem.st` | `tcgen05.ld` / `st` | unchanged |
| `tm::wait_st()` | `tcgen05.wait::st` | `vx_wsync` |

`tc.mma` computes `k_tiles` consecutive K-tiles (`tileK` each) of a stage for
all `n_frag` N fragments of the core's M lanes. The fields fixed across a K
loop (formats, sparse, A in TMEM, `n_frag`, `k_tiles`, leading bytes) live in
one of four per-CTA descriptor slots loaded by `tc.idesc`; a B leading-bytes
field of 0 selects the block-major layout the DXA `BlockMajor` mode writes.

## Semantics

- `tc.mma`, `tc.cp` and `tc.commit` enter one in-order queue per core.
  `tc.commit` arrives (one transaction) on its barrier when every earlier
  queue entry has completed. Blackwell tracks the issuing thread's operations;
  waiting for all earlier entries is a superset and keeps the queue a FIFO.
- Software owns the pipeline: a producer warp issues DXA loads that complete
  on full barriers; the MMA warp waits full[s], issues `tc.mma`, then
  `tc.commit(empty[s])` to release the stage, and `tc.commit(done)` after the
  last K step.
- The column-overlap interlock and `tm::wait_st()` rules of the TMEM proposal
  are unchanged.
- `tmem.alloc` allocates from a per-core column map in 32-column units and
  returns ~0 when the request does not fit; `tm::alloc()` retries, because a
  stalled allocation would block the TMEM op port that the freeing
  `tmem.dealloc` needs. Shared memory no longer reserves TMEM.

## What went away

Job descriptors, `tj.desc`/`tj.post`, the CONSUME/CLOSED/COPY modes,
sequencer-issued DXA loads (DXA slots, coordinates, K advance) and the TCU's
DXA request port, stage barriers inside the TCU (transaction attach and the
barrier unit's phase-completion feedback), and the stage-wait perf counter.
The sequencer is now the per-CTA queue, the step engine, the copy engine and
the interlock.

## Plan

Each phase kept SimX and RTL in lockstep.

1. Done: `tmem.alloc` / `tmem.dealloc` (`tm::alloc`, `tm::dealloc`) in SimX
   and RTL; kernels no longer pad shared memory for TMEM.
2. Done: `tc.mma` / `tc.commit` in SimX and RTL on the existing step engine.
   On rtlsim the GEMM ran 0.1-7.6% behind CLOSED, the gap shrinking with K.
   CONSUME jobs (and the per-stage `sw_full` form) are removed.
3. Done: FA-2, FA-4 and the chained GEMM run on `tc.mma`. FA-4 has a producer
   and an MMA warp; FA-2 and the chained GEMM have none to spare, so warp 0
   stages and issues both. rtlsim, against CLOSED:

   | Case | CLOSED | `tc.mma` |
   |---|---|---|
   | FA-4 `-b4 -c1` | 58,565 | 63,131 (+7.8%) |
   | FA-2 `-l64 -n32 -d64 -b4 -c1` | 141,832 | 163,356 (+15.2%) |
   | FA-2, 3 CTAs, 256 columns | 155,666 | 174,855 (+12.3%) |
   | Chained GEMM | 23,805 | 29,026 (+21.9%) |

   The cost per K-step is the issuing warp's instruction stream: building and
   posting a descriptor is ~25 dependent instructions at ~9 cycles each, more
   than one K-tile's MMA. A third stage does not help.
   - A data-triggered `tc.mma` (the op names its full barrier and phase, and
     the TCU starts it from a mirror of the barrier phase store) was built in
     SimX and RTL and measured: chained GEMM +1.5% (SimX), FA-4 -0.05%
     (rtlsim). The warp still posts every op, so it was removed.
   - Staging `KB` K-tiles per stage, consumed by one `tc.mma` with
     `k_tiles = KB`, amortizes the issue. rtlsim, CI configurations:

     | Case | 1 K-tile per stage | `KB` K-tiles | |
     |---|---|---|---|
     | FA-4 `-b4 -c1`, `KB` = 4 | 69,354 | 62,673 | -9.6% |
     | FA-2 `-b2 -c2`, `KB` = 2 | 99,416 | 91,447 | -8.0% |
     | Chained GEMM, `KB` = 4 | 29,026 | 24,248 | -16.5% |

     FA-2 keeps `KB` = 2: larger stages leave room for one of its two CTAs.
   - SimX now models three RTL port rules that longer ops expose: a `tmem.ld`
     takes the running op's TMEM read for a cycle, a step presents the cycle
     after its operands land, and a step's D write-back blocks a `tmem.st`.
     FA-4 `-b4` parity moved from 7.5% to 4.9%; the rest is DXA writes and
     tile-buffer reads interleaving on the shared LMEM DMA port.
4. Done: CLOSED and COPY jobs, `tj.desc`/`tj.post` and the TCU's DXA port are
   removed; `tc.cp` replaces COPY. The sparse GEMM with `tc.cp` takes 7,746
   rtlsim cycles at 64^3 against 9,821 with warp-copied metadata, and 29,233
   at 128^3 against CLOSED + COPY's 28,607. Next: resynthesize (LUTs, Fmax) on
   the V80 flow.
5. Done: block-major B (`ldb` = 0, the bbuf's native layout, as written by
   the DXA `BlockMajor` mode) in `tc.mma`, so row-major B runs on TMEM. It is
   not faster: at one K-tile per stage K-major B already fills a bank row per
   block, and `BlockMajor` scatters cost 2.5x the DXA write beats (64x64x256
   GEMM 43,473 vs 41,789 rtlsim cycles). A K-major source can reach the layout
   through a rank-3 box (K-tile, rows, K-tiles), but the DXA then fetches each
   line once per K-tile (3.3x the global reads), which costs more than the 37%
   fewer tile-buffer reads save on FA-4. Swizzled layouts pay on GPUs because
   each bank is addressed separately; the LMEM DMA port reads one bank row per
   access, and a separate DXA-write path is worth only 0-0.3% (SimX).
6. Next: the tile buffers, which fetch one operand block at a time with one
   read in flight and none ahead (K-major B at `KB` = 4: 7 cycles per two
   steps). Prefetching the next B block and the next K-tile's A stripes is
   the remaining lever on MMA time.
7. Done: scalar `tc.mma` operands. The warp used to gather all eight
   descriptor words into lanes 0..3 of x31/x30 for every op, which needed
   lanes 0..3 active, pinned registers and NT >= 4. Now `tc.idesc` loads one of
   four per-CTA slots with the fields fixed across a K loop, and `tc.mma`
   carries `taddr`, `acc_init`, the metadata column, A and B as scalars from
   the lowest active thread; `tc.cp` takes scalars too. rtlsim: FA-4 `-b4`
   62,673 -> 62,016, the GEMM 41,789 -> 41,523, block-major B 43,473 ->
   42,788; the chained GEMM 24,248 -> 25,169, all of it a P-conversion loop
   the compiler now allocates three instructions longer. FA-4 parity 3.9%.

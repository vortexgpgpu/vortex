# Tensor Memory and Barrier-Triggered Tensor Jobs

Status: draft. The job ISA this proposal introduced (`tj.desc`/`tj.post`
with CONSUME, CLOSED and COPY jobs) has been replaced by `tc.mma`, `tc.commit`
and `tc.cp` ([tcgen05_alignment_proposal.md](tcgen05_alignment_proposal.md)),
so the warp is back in the K loop as on Blackwell. TMEM, its allocator, the
step engine and the copy engine below are current; Results and Findings were
measured with jobs.

## Motivation

WGMMA keeps the accumulator in the FP register file. Every WGMMA uop reads a
1024-bit C fragment and writes a 1024-bit D fragment through issue, the
operand collector, `VX_lane_dispatch` and writeback. This has three costs:

- Tile size is capped by the register file: `NRC=32` uses all of `f0..f31`,
  so a warp's tile is 16x64.
- The issuing warps occupy their issue slices for every uop.
- The K loop runs in software: per stage, a producer warp arms DXA, all warps
  wait on a fill barrier, issue the MMAs, and wait on a gate barrier.

Blackwell (`tcgen05`) moves the accumulator into Tensor Memory (TMEM) and makes
the MMA asynchronous, but a warp still waits on every stage and issues every
MMA. This proposal keeps the accumulator in TMEM and also removes the warp from
the K loop: the TCU fires a K-tile when its DXA stage lands, releases the stage
when it has fetched it, and can re-arm DXA itself. DXA and the TCU form a
hardware producer-consumer pipeline whose tokens are the existing transaction
barriers.

Related work: decoupled access/execute (Smith, 1982), triggered instructions
(Parashar et al., ISCA 2013), descriptor-driven DMA/compute pipelines in
accelerators (TPU, Gaudi). The contribution is applying barrier-token
triggering between an async copy engine and the tensor core of a SIMT GPU.

## Compute structure

| Level | Computes |
|---|---|
| FEDP | one output, `FEDP_K` of K (16 fp16 / 32 fp8 / 64 fp4) |
| TCU block, one uop | 8x4 tile (`TC_M x TC_N`), `FEDP_K` of K |
| `tc.mma`, one K-tile, one block | 16 x `4*n_frag` (2 M steps x `n_frag` N steps) |
| `tc.mma` | `ISSUE_WIDTH*16` x `4*n_frag`, `k_tiles` K-tiles |

- The 2D FEDP array is kept: it reuses each A element `TC_N` times and each B
  element `TC_M` times per pass, which the LMEM bandwidth requires.
- Narrower types pack 2 (8-bit) or 4 (4-bit) products per FEDP lane, so K per
  pass grows and the number of K-tiles shrinks; operand bits per uop are fixed.
- FEDP2K is required: one uop covers a whole K-tile, so there is no
  intra-tile K step.

## Dataflow

```
for kt in 0..k_tiles-1:              # GEMM K, 16 fp16 at a time; arrives by DXA stage
    abuf <- A[16 rows x FEDP_K]      # stationary for the K-tile
    for n in 0..n_frag-1:
        bbuf <- B[FEDP_K x TC_N]     # streamed, broadcast to all blocks
        for m in 0..1:
            D[m,n] = A[m] . B[n] + (kt == 0 && acc_init ? 0 : D[m,n])   # TMEM RMW
```

- The K-tile loop is outermost because DXA delivers K over time. Each stage is
  used for every output, then freed, so every operand element is read from
  LMEM once. K-innermost (output-stationary) would avoid TMEM traffic but
  needs ~12x the LMEM bandwidth and the whole K strip resident.
- M is fixed per block (16 rows); N is the expandable dimension, bounded by
  TMEM columns, as on Blackwell (M = TMEM lanes, N = columns).
- An MMA spans all blocks so B can be broadcast.
- A D fragment is reused every `2*n_frag` uops, which exceeds the FEDP latency
  (5 on Vivado) plus the TMEM read latency for `n_frag >= 4`.

### LMEM bandwidth

The TCU reads one 1024-bit LMEM bank row per cycle. At one uop per cycle per
block (bits per cycle):

| n_frag | B | A | Total |
|---|---|---|---|
| 16 | 512 | 512 | 1024 |
| 32 | 512 | 256 | 768 |
| 64 | 512 | 128 | 640 |

## TMEM

- One TMEM per core: `NUM_TCU_BLOCKS * M` lanes by `TCU_TMEM_COLS` 32-bit
  columns, M being the WGMMA tile M (`2 * TC_M`). A lane is an M row of an
  MMA and a column is an N element, as on Blackwell (128 lanes = 4 warps x 32
  threads x one M row each).
- Banked by lane: block `b` computes and owns lanes `[b*M, (b+1)*M)`, one
  `M x TCU_TMEM_COLS` bank of BRAM/URAM next to its FEDP array. A D fragment
  (`TC_M x TC_N`, one FP register across a warp) is `TC_M` lanes by `TC_N`
  columns of one bank; the bank's ports carry one fragment per cycle. The
  sequencer has priority over `tmem.ld/st`.
- At NT=8 there are 32 lanes (M = 8 = NT); `TCU_TMEM_COLS = 512` is 64 KB per
  core, one Blackwell sub-partition.
- Address `{lane [31:16], column [15:0]}`. An MMA's D address is a column
  (lane 0); it writes the same columns on every lane.
- `tmem.ld/st` move one bank's `M` lanes; the lane must be a multiple of M.
  Two register layouts, like Blackwell's `tcgen05.ld` shapes:
  - lane per thread (`tmem.ld`): thread `t` holds lane `lane + t % M`,
    register `r` column `col + r * (NT / M) + t / M`. At NT = M this is
    `.32x32b`: thread `t` gets M row `t`, consecutive registers consecutive
    columns.
  - lane per register (`tmem.ld.r`): register `r` holds lane `lane + r % M`,
    thread `t` column `col + (r / M) * NT + t`. Each register is NT contiguous
    columns of one row, so the epilogue's global stores are coalesced; with
    lane per thread each store touches NT rows (SimX, 128^3 with jobs:
    81,943 vs 65,697 cycles).
  Any warp may name
  any bank: the CTA dispatcher assigns a CTA the lowest free warps, so its
  warps are not guaranteed to sit one per TCU block, at the cost of an
  `ISSUE_WIDTH`:1 bank mux (Blackwell instead restricts warp `w` to the lanes
  of sub-partition `w % 4`).
- Physical column addresses from an allocator, as `tcgen05.alloc/dealloc`:
  `tmem.alloc rd, rs1` returns the first of rs1 columns (a power of two from
  32 to `TCU_TMEM_COLS`), the lowest free run aligned to its size in a
  per-core map of 32-column chunks, or ~0 when none fits; `tm::alloc()`
  retries. A stalled allocation would hold the one TMEM op port that the
  freeing CTA's `tmem.dealloc` needs. `tmem.dealloc rs1, rs2` waits for the
  CTA's ops on the columns, then frees them. One warp allocates and shares
  the column through shared memory; the CTA frees it before exiting.
  Contents are undefined after allocation.

## ISA

CUSTOM0 (`0x0B`), funct7 `0x05`:

| funct3 | Instruction | rd field | rs1 | rs2 field |
|---|---|---|---|---|
| 0 | `tmem.ld fd, (taddr), n` | first destination FP reg | `taddr` | immediate `n-1` |
| 1 | `tmem.st fs, (taddr), n` | first source FP reg | `taddr` | immediate `n-1` |
| 2-3 | reserved | | | |
| 4 | `tmem.ld.r fd, (taddr), n` | first destination FP reg | `taddr` | immediate `n-1` |
| 5 | `tmem.st.r fs, (taddr), n` | first source FP reg | `taddr` | immediate `n-1` |
| 6 | `tmem.alloc rd, rs1` | first column | columns | - |
| 7 | `tmem.dealloc rs1, rs2` | - | first column | columns |

funct7 `0x06`, the asynchronous ops of one in-order queue per core. The rd
field of `tc.mma` and `tc.cp` names a third source register (rs3):

| funct3 | Instruction | rs1 | rs2 | rd field |
|---|---|---|---|---|
| 0 | `tc.idesc s, rs1, rs2` | `idesc` | leading bytes | slot `s` (0..3) |
| 1 | `tc.commit` | barrier id | arrival count | - |
| 2 | `tc.cp` | {columns [31:16], `taddr` [15:0]} | source | rs3: stride |
| 3 | reserved | | | |
| 4..7 | `tc.mma` with slot funct3 - 4 | {`acc_init` [31], metadata column [30:16], `taddr` [15:0]} | A | rs3: B |

- `tmem.ld/st` expand into `n` uops in the front-end uop sequencer (register
  `f+i` to/from its columns) and use the normal scoreboard and writeback.
- `tc.*` operands are scalars read from the warp's lowest active thread, so
  any one thread may issue them, as `tcgen05.mma` is issued by one elected
  thread. They are single uops with no writeback that reach one in-order TCU
  input, so their operands need no memory fence.
- `tc.idesc` loads one of the CTA's four descriptor slots with the fields
  fixed across a K loop; a `tc.mma` carries only what changes per stage. A
  slot written by another warp needs a barrier before the `tc.mma` that uses
  it.
- Each CTA has two queue entries; `tc.mma` and `tc.cp` stall at issue while
  they are full. A CTA's ops run in order; `tc.commit` arrives on its barrier
  once the CTA's earlier ops have completed.
- `tc.cp` is `tcgen05.cp`: its column count of every lane from shared memory,
  lane `l` from `source + l * stride` bytes, written from column `taddr`.
  Source and stride are NT-word aligned and the column count a multiple of NT.
  The TCU reads one NT-word chunk per LMEM read on its lowest-priority port,
  one read in flight; a response is written to TMEM in its cycle unless an MMA
  result or `tmem.st` holds the bank.

Compared with `tcgen05`: `alloc`, `dealloc`, `mma`, `commit` and `cp` map
one to one; `wait::ld` is unnecessary because `tmem.ld` goes through the
in-order scoreboard, and `wait::st` is `tm::wait_st()`; `relinquish`,
`shift`, `.ws` and `cta_group::2` are not provided.

### Descriptor slot (`tc.idesc`)

| Operand | Bits | Field |
|---|---|---|
| `idesc` | [4:0] | `fmt_s` |
| | [9:5] | `fmt_d` |
| | [10] | sparse (2:4 compressed A, metadata in TMEM) |
| | [11] | reserved |
| | [12] | A source: 0 = LMEM, 1 = TMEM |
| | [20:13] | `n_frag` |
| | [23:21] | reserved |
| | [31:24] | `k_tiles` |
| leading bytes | [15:0] / [31:16] | A / B; B = 0 selects block-major B (dense only) |

A `tc.mma`'s `taddr` is its first D column (lane 0), A a shared-memory offset
or, when A is in TMEM, its first column, B a shared-memory offset, and
`acc_init` makes the first K-tile overwrite D.

### Ordering

- Software issues a `tc.mma` only after its operands' stage has landed.
- An MMA writes only TMEM. After waiting on a later `tc.commit`'s barrier,
  `tmem.ld` observes all its D writes.
- A `tmem.ld/st` uop waits while an op of its CTA is pending whose columns
  (D, plus A and metadata when in TMEM) overlap the uop's columns. Transfers
  on other columns run alongside the CTA's ops, so one Q tile's softmax can
  overlap the next tile's MMA within a CTA.
- Cross-warp reuse of TMEM columns is ordered by software barriers.
- `tmem.st` is posted: the warp moves on once it issues, and a barrier does
  not wait for it (a barrier drains only the LSU). A warp whose stores feed an
  MMA (P for P.V, rescaled O, A in TMEM) calls `tm::wait_st()` before the
  barrier that hands the columns over, the counterpart of
  `tcgen05.wait::st`. It is `vx_wsync`, which waits for all of the warp's
  issued uops to commit; a store commits once written.

### Scheduling

CTAs share the tensor core at op granularity: the running `tc.mma` keeps it
until its last K-tile, then the oldest CTA head `tc.mma` takes it. `tc.cp`
runs on its own engine. One commit arrival is issued per cycle.

A step's A stripes (first step of a K-tile) and B block are fetched through
the WGMMA `abuf`/`bbuf` when the step is presented; there is no B prefetch
yet.

## RTL

- `VX_tcu_tmem` (in the core-level `VX_tcu_shared`, beside the WGMMA tile
  buffers): TMEM banks and allocator, eight op slots (two per CTA), the copy
  engine, commit arrivals, and a one-step issue register that presents each
  (m, n) step to all four `VX_tcu_core`s with C read from TMEM. Each
  `VX_tcu_block` reaches it through a `VX_tcu_job_if`: the block sends its
  TMEM and tc ops there, MMA steps take its core ahead of warp MMAs, and a
  per-block order queue routes the core's MMA results to TMEM writes and the
  rest to commit.
- TMEM bank per block: `NT` 32-bit sub-banks of `(TMEM_COLS / TC_N) * M_STEPS`
  rows, 1R1W, one-cycle read. Cell (lane `m*TC_M+i`, column `c`) is in
  sub-bank `(TC_N*i + c + (NT/M)*m) mod NT`, row `(c/TC_N)*M_STEPS + m`, so a
  C/D fragment and either `tmem.ld/st` layout touch each sub-bank once.
  `tmem.ld/st` and MMA steps share one port (one op per cycle); a load has
  priority over the MMA's C read, MMA D writes over `tmem.st`.
- Barriers: `tc.commit` arrivals enter the barrier unit in
  `VX_warp_shared_unit` after warp requests and DXA completions.
- Requires WGMMA (tile buffers); a TMEM build does not accept warp WGMMA.
  XLEN=32, `VX_CFG_TCU_PE_RATIO=1` (an MMA step issues to every core in one
  cycle). FEDP2K is optional: a `tc.mma` K-tile is one FEDP width of A, so
  without it a tile spans two K-tiles, and a sparse K-tile carries one
  metadata row instead of two. Block-major B needs one B block per LMEM bank
  row (FEDP2K at NT=8); K-major B has no such limit.

## Results (RTL vs SimX)

rtlsim and SimX, NT=8, 16 warps, ISSUE_WIDTH=4, LSU_BLOCKS=1, FEDP2K, DPI
FEDP, fp16 -> fp32, B K-major, PIPE_N=2. Instruction counts match exactly.

| Case | rtlsim | SimX |
|---|---|---|
| 64^3 CONSUME | 11,909 | 11,417 |
| 64x64x256 CONSUME | 41,276 | 40,884 |
| 128^3 CONSUME / CLOSED / sw_full | 69,625 / 67,647 / 69,925 | 65,763 / 64,777 / 66,255 |
| 128x128x512 CONSUME | 308,056 | 308,550 |
| 128^3 CLOSED, 512 columns, N = 32 / 64 / 128 per CTA | 66,582 / 48,839 / 42,344 | 63,704 / 46,404 / 40,149 |
| 128^3 WGMMA deepbuf (register accumulator) | 78,673 | 94,921 |

The remaining TMEM gap (4-6% at 128^3) is in the stage round trip (producer
wake-up, DXA, barrier), not the job datapath: steady-state K-tiles match. The
WGMMA deepbuf baseline is 17% pessimistic in SimX at 128^3 (1.3% at 64^3).

128x128x512, rtlsim:

| Case | Cycles | vs WGMMA |
|---|---|---|
| WGMMA deepbuf | 312,224 | 1.00x |
| CONSUME, 128 columns, N = 32 | 308,056 | 1.01x |
| CLOSED, 512 columns, N = 128 (TFR FEDP) | 179,049 (SimX 178,253) | 1.74x |

All 128^3 and 64^3 cases also pass with the TFR FEDP at the same cycle counts
as DPI. Long CLOSED jobs leave every warp waiting on a barrier for over
`STALL_TIMEOUT` cycles, so the scheduler watchdog treats barrier traffic as
progress.

### Findings

- Tile size is the whole gain. At 128 columns a CTA's tile (16 warps x 32
  registers x NT = 4096 accumulators) is exactly what the register file
  already holds, so per-stage work and reuse equal WGMMA's: 1.01-1.16x, from
  issue/RF relief only. Wider N per CTA (512 columns) gives 1.6-1.9x.
- K-loop placement is small on GEMM. At 128^3: Blackwell-style (sw_full,
  69,925), CONSUME (69,625, -0.4%), CLOSED (67,647, -3.3%). CLOSED cuts
  instructions 39% vs CONSUME (9,392 vs 15,504), which GEMM cannot use
  because nothing else competes for issue.
- CLOSED costs ~1.3k flops (91 static + 70 counter bits per slot); fill state
  is per CTA head, so it could be kept per CTA instead of per slot.
- B is not the bottleneck: a one-block `bbuf` prefetch (next block read while
  the last M step fires) cut the B cadence from 5 to 3-4 cycles per block but
  64^3 only from 11,909 to 11,851 cycles; dropped. The per-stage cost is the
  serial A stripe load (~39 cycles at 64^3: four `abuf`s in turn on the
  shared LMEM port, each bank row read twice) plus the stage round trip.

## Front-end decoupling

MMAs never touch the register file, so their uop expansion moves from
`VX_tcu_uops` (issue) into a step engine in the TCU. The remaining front-end
coupling is: the per-CTA in-flight count back to issue (for `tmem.ld/st`),
`tc.commit` arrivals into the barrier unit, and queue back-pressure on
`tc.mma`/`tc.cp`.

In a TMEM-only build (WMMA and register-accumulator WGMMA disabled), the TCU
path in `VX_lane_dispatch` needs one 1024-bit operand (`tmem.st`) instead of
three (`rs1`/`rs2`/`rs3`), plus the `tmem.ld` result path.

## Operand buffers and layout

`tbuf` area is layout steering, not storage (V80, 1 core, NT=32: `bbuf` 55k
LUTs for 2048 FFs; each `abuf` 14.7k LUTs for 4096 FFs), and K-major B costs
`TC_N` LMEM reads per B block. TMEM MMAs accept one TCU-native LMEM layout,
written by a new DXA layout mode:

- One B block (`FEDP_K x TC_N`) is one LMEM bank row, in FEDP lane order.
- One block's A slice (16 x `FEDP_K`) is four consecutive bank rows, two per
  M step.

`bbuf` becomes a double-buffered 1024-bit pipeline register with prefetch;
`abuf` becomes a 4096-bit stationary register with a 2:1 M-step select.

## A from TMEM and sparse MMAs

Warps fill TMEM before the `tc.mma` and synchronize on a barrier: an MMA's
reads are ordered only behind the issuing warp's own transfers. Both regions must be
disjoint from D (descriptor check). A K-tile prologue reads them over the TMEM
read port, on all blocks in parallel, ahead of the K-tile's first step and once
the previous K-tile's last step has fired (`tmem.ld` > prologue/C read). A read
is one `tmem.ld` lane-per-thread access: `NT / M` columns of every lane.

- A: `FEDP_K` columns per K-tile of K-packed words; a lane is an M row.
  `FEDP_K * M / NT` reads (4 at NT=8) fill `abuf`'s storage, against ~39 cycles
  of serial LMEM stripe load. The fetch FSM stays idle, and producers stage B
  only.
- Sparse: compressed A (from a stage or TMEM) with the FULLK uop shape, so a
  K-tile covers twice the dense K; requires `VX_CFG_TCU_SPARSE_FULLK`. A lane's
  K-tile metadata is its two `VX_tcu_sp_mux` rows, low bits first, over
  `ceil(2 * row_bits / 32)` columns (fp16 and fp8: one, fp4: two at NT=8): one read
  per K-tile, into a per-block register, against a warp TCU_LD per K-tile in
  WGMMA. The kernel loads a CTA's metadata as NVIDIA does (TMA -> smem ->
  `tcgen05.cp`): one DXA copy into the not-yet-used stages, then `tc.cp`
  (`TMEM_CP`), or unrolled smem loads and `tmem.st` by the warps.

Warp WGMMA and TMEM are mutually exclusive: warps never issue WGMMA or WGMMA_SP
in a TMEM build, so VX_tcu_core takes those ops' metadata from `VX_tcu_tmem`.
`VX_CFG_TCU_WGMMA_ENABLE` still selects the tile buffers these builds reuse.

## Unchanged

WMMA, WGMMA (register accumulator), `TCU_LD`, DXA issue, and barriers.

## Extensions

- Operand prefetch in the tile buffers: the B buffer fetches a block only
  after the previous block's steps, one read in flight, so K-major B at four
  K-tiles per stage takes 7 cycles per two steps; the A stripes of a K-tile
  are read row by row before its first step. Fetching the next block (and the
  next K-tile's stripes) during the current steps overlaps the two.
- Fine-grained firing: start a K-tile as its bank rows land rather than on
  the stage barrier, shortening fill and inter-stage gaps.
- Softmax jobs for attention, reusing the existing units: `exp(S - m)` through
  the MUFU `EX2` behind an arbiter (job vs warp), row max by comparators as S
  streams from TMEM, row sum from the tensor core (a ones column appended to
  V), packed P written back to TMEM; O rescale left to warps under FA4's
  threshold rule, `1/l` by warps at the end. Worth building only if software
  attention on TMEM is bound by issue and TMEM<->RF traffic rather than MUFU
  throughput; measure that first.
- One configurable array per core (`TCU_ARRAY_M x TCU_ARRAY_N`), independent
  of `NUM_THREADS` and `ISSUE_WIDTH`. A 16x8 array of the same 128 FEDPs needs
  ~1.5x less LMEM bandwidth per MAC than 4 blocks of 8x4. Requires dropping
  or separating the register-path MMAs.
- Dynamic TMEM allocation; non-FEDP2K configurations.

TMEM variants drain C with `tmem.ld.r` (one row per register, coalesced
stores; one lane per thread costs ~25% at 128^3). At N=32 per CTA they are
bound by stage loads; TMEM's main lever is the larger N per CTA that the
register file cannot hold: more reuse, less DXA traffic per FLOP.

## Plan

Steps 1-3 were carried out with jobs; the job ISA's removal is tracked in
[tcgen05_alignment_proposal.md](tcgen05_alignment_proposal.md).

1. SimX
   - Per-core TMEM (lanes x columns); `tmem.ld/st` decode, uops, scoreboard
     interlock.
   - `tj.post` and the per-block sequencer, CONSUME mode, `done_bar`
     completion; sequencer subscribes to barrier phase completion.
   - CLOSED mode: sequencer-issued DXA requests.
   - Kernels: `sgemm_tcu_tmem` (dataflow) and a Blackwell-style variant
     (one-K-tile job per stage), sharing `common.h` with
     `sgemm_tcu_wg_dxa_deepbuf`.
2. Measure (SimX, paper flags): cycles, TCU busy fraction, inter-stage gaps,
   issue-slot occupancy and barrier stall cycles for Hopper-style /
   Blackwell-style / CONSUME / CLOSED at `n_frag` = 16/32/64. Proceed to RTL
   only if the dataflow variants show a clear gain over Blackwell-style.
3. RTL (done; see above)
   - Close the WGMMA deepbuf SimX gap before comparing against it.
   - B prefetch and multi-outstanding `bbuf` reads, in RTL and SimX together.
4. Area
   - TCU-native DXA layout; simplified `abuf`/`bbuf`; out-of-context
     synthesis of `tbuf`.
   - TMEM-only build without register-path MMAs; measure `lane_dispatch`.
5. Extensions: readiness scheduling, fine-grained firing, then the
   single-array TCU.

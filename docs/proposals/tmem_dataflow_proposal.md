# Tensor Memory and Barrier-Triggered Tensor Jobs

Status: draft

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
| Job, one K-tile, one block | 16 x `4*n_frag` (2 M steps x `n_frag` N steps) |
| Job | `ISSUE_WIDTH*16` x `4*n_frag`, full K |

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
- A job spans all blocks so B can be broadcast.
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
  columns, M being the WGMMA tile M (`2 * TC_M`). A lane is an M row of a job
  and a column is an N element, as on Blackwell (128 lanes = 4 warps x 32
  threads x one M row each).
- Banked by lane: block `b` computes and owns lanes `[b*M, (b+1)*M)`, one
  `M x TCU_TMEM_COLS` bank of BRAM/URAM next to its FEDP array. A D fragment
  (`TC_M x TC_N`, one FP register across a warp) is `TC_M` lanes by `TC_N`
  columns of one bank; the bank's ports carry one fragment per cycle. The
  sequencer has priority over `tmem.ld/st`.
- At NT=8 there are 32 lanes (M = 8 = NT); `TCU_TMEM_COLS = 512` is 64 KB per
  core, one Blackwell sub-partition.
- Address `{lane [31:16], column [15:0]}`. A job's D address is a column
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
    lane per thread each store touches NT rows (SimX, 128^3 CONSUME: 81,943
    vs 65,697 cycles).
  Any warp may name
  any bank: the CTA dispatcher assigns a CTA the lowest free warps, so its
  warps are not guaranteed to sit one per TCU block, at the cost of an
  `ISSUE_WIDTH`:1 bank mux (Blackwell instead restricts warp `w` to the lanes
  of sub-partition `w % 4`).
- Physical column addresses, no allocator. A CTA owns the columns
  proportional to its shared-memory window, which the CTA dispatcher already
  keeps disjoint across co-resident CTAs:

  ```
  bytes_per_col = LMEM_SIZE / TCU_TMEM_COLS
  tmem_base     = ceil(window_offset / bytes_per_col)
  cols          >= window_bytes / bytes_per_col - 1
  ```

  A kernel that needs `n` columns sizes its window to at least
  `(n + 1) * bytes_per_col` (`vx_tmem.h`: `smem_bytes_for_cols`), so TMEM
  occupancy is admitted together with shared memory. The CTA slot id cannot
  be used: slots are allocated round-robin over all LMEM-fitting slots and
  exceed the number of co-resident CTAs. Contents are undefined at CTA start.

## ISA

CUSTOM0 (`0x0B`), funct7 `0x05`:

| funct3 | Instruction | rd field | rs1 | rs2 field |
|---|---|---|---|---|
| 0 | `tmem.ld fd, (taddr), n` | first destination FP reg | `taddr` | immediate `n-1` |
| 1 | `tmem.st fs, (taddr), n` | first source FP reg | `taddr` | immediate `n-1` |
| 2 | `tj.desc rs1, rs2` | - | descriptor words 0..3 (lanes 0..3) | x-reg: words 4..7 (lanes 0..3) |
| 3 | `tj.post rs1, rs2` | - | descriptor words 8..11 | x-reg: words 12..15 |
| 4 | `tmem.ld.r fd, (taddr), n` | first destination FP reg | `taddr` | immediate `n-1` |
| 5 | `tmem.st.r fs, (taddr), n` | first source FP reg | `taddr` | immediate `n-1` |
| 6-7 | reserved | | | |

- `tmem.ld/st` expand into `n` uops in the front-end uop sequencer (register
  `f+i` to/from its columns) and use the normal scoreboard and writeback. They stall while their CTA has a job in flight.
- `tj.desc` + `tj.post` carry the 16-word descriptor in registers, gathered
  across lanes 0..3 (`vx_wgather`, as DXA issue does). Both are single uops
  with no writeback that reach the same in-order TCU input queue, so the
  descriptor needs no memory fence. (A shared-memory descriptor read through
  `tcu_lmem_if` races the warp's own stores: LSU stores are fire-and-forget and
  `fence` does not wait for them.)
- `tj.post` enqueues the job on the TCU; it runs on every block. Each CTA has a
  2-entry job queue and `tj.post` stalls at issue while it is full. Jobs of one
  CTA run in order.
- A CONSUME job with `sw_full` and `k_tiles = kb`, posted by a warp after it
  waits on each stage, reproduces Blackwell's `tcgen05.mma` + `commit` loop.

Compared with `tcgen05`: `alloc`/`dealloc`/`relinquish` are replaced by the
slot-derived base; `mma` and `commit` by `tj.post`; `wait::ld/st` and the
`fence` variants are unnecessary because `tmem.ld/st` go through the in-order
scoreboard and `done_bar` is signalled after the last TMEM write; `cp`,
`shift`, `.ws` and `cta_group::2` are not provided.

### Instruction descriptor (`idesc`)

| Bits | Field |
|---|---|
| [4:0] | `fmt_s` |
| [9:5] | `fmt_d` |
| [10] | sparse |
| [11] | `acc_init` (first K-tile overwrites D) |
| [12] | A source: 0 = LMEM, 1 = TMEM (reserved) |
| [20:13] | `n_frag` |
| [31:21] | reserved |

### Job descriptor (16 words, in registers)

| Word | Field |
|---|---|
| 0 | `idesc` |
| 1 | `mode` [1:0] (0 = CONSUME, 1 = CLOSED), `sw_full` [2] (software already waited on the full barriers), `nstages` [7:4], `kb` [15:8] (K-tiles per stage), `k_tiles` [31:16] |
| 2 | `taddr`: first D column (lane 0) |
| 3 / 4 / 5 | stage base: A / B / sparse metadata |
| 6 | stage stride (bytes) |
| 7 | leading bytes: A [15:0], B [31:16] (0 = block-major B, `kb` must be 1) |
| 8 / 9 / 10 | `full_bar` base / `empty_bar` base / `done_bar`; stage `s` uses `base \| (s << 8)` |
| 11 | DXA descriptor slots: A [3:0], B [7:4] (CLOSED) |
| 12 / 13 | A / B start coordinates {c0 [15:0], c1 [31:16]} (CLOSED) |
| 14 | K advance per stage [15:0]; K coordinate select for A [16], B [17] (CLOSED) |
| 15 | first stage [3:0] |

### Modes

- CONSUME: a producer warp issues DXA. The producer and the sequencer are the
  two participants (count 2) of every `full_bar[s]` and `empty_bar[s]`:
  - Producer, per fill: on refills, `p = arrive(empty[s], 2); wait(empty[s], p)`;
    then `expect_tx(full[s], 2)`, the two DXA issues, `arrive(full[s], 2)`.
    After the last fill it closes the final generation of each used empty
    barrier the same way.
  - Sequencer, per stage use: sample the phase, arrive on `full[s]`, and wait
    for the phase to change. Its arrival is required, so the phase cannot
    advance before it looks and no phase state is carried in the descriptor.
    After fetching the stage it arrives on `empty[s]`.
  Multicast is unchanged because the producer warp owns it.
- CLOSED: the sequencer also issues DXA and is the full barriers' only
  participant: per fill it attaches two transactions, issues the A and B loads
  and arrives with count 1. It keeps `nstages` fills ahead of the K-tile being
  computed; a stage is refilled once the job has moved past it. No warp runs
  in the K loop. No multicast in v1.
- After the last D write on every block, the sequencer counts one transaction
  on `done_bar` (armed with `vx_barrier_expect_tx(done_bar, 1)`).

### Ordering

- A stage is read only after its `full_bar` phase completes.
- A job writes only TMEM. After waiting on `done_bar`, `tmem.ld` observes all
  its D writes.
- Cross-warp reuse of TMEM columns is ordered by software barriers.

### Scheduling

Jobs of different CTAs share the tensor core at stage granularity. The
running job keeps it until its stage ends; then the oldest CTA head whose
next stage is full takes it. Every waiting head keeps its own barrier
arrivals and (CLOSED) stage loads moving meanwhile: the sequencer issues one
barrier operation (done release, empty arrival, fill attach, full arrival, in
that priority) and one DXA load per cycle. Strict FIFO order stalls the
tensor core on one CTA's next stage while others are ready.

A step's A stripes (first step of a K-tile) and B block are fetched through
the WGMMA `abuf`/`bbuf` when the step is presented; there is no B prefetch
yet.

## RTL

- `VX_tcu_tjob` (in `VX_tcu_unit`): TMEM banks, eight job slots (two per
  CTA), the barrier/DXA request generation, and a one-step issue register
  that presents each (m, n) step to all four `VX_tcu_core`s with C read from
  TMEM. Job steps take a core ahead of warp MMAs; a per-block order queue
  routes the core's job results to TMEM writes and the rest to commit.
- TMEM bank per block: `NT` 32-bit sub-banks of `(TMEM_COLS / TC_N) * M_STEPS`
  rows, 1R1W, one-cycle read. Cell (lane `m*TC_M+i`, column `c`) is in
  sub-bank `(TC_N*i + c + (NT/M)*m) mod NT`, row `(c/TC_N)*M_STEPS + m`, so a
  C/D fragment and either `tmem.ld/st` layout touch each sub-bank once.
  `tmem.ld/st` and job ops share one port (one op per cycle); a load has
  priority over the job's C read, job D writes over `tmem.st`.
- Barriers: the sequencer's operations enter `VX_wctl_unit` after warp
  instructions and DXA completions; `VX_bar_unit` reports every local phase
  completion back (`warp_ctl_if.bar_flip_*`).
- DXA: a second request port per core (`tcu_dxa_req_bus_if`), socket index
  `SOCKET_SIZE + core`.
- Requires WGMMA (tile buffers) and FEDP2K; a TMEM build does not accept
  warp WGMMA. XLEN=32, K-major B.

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

## Front-end decoupling

Jobs never touch the register file, so their uop expansion moves from
`VX_tcu_uops` (issue) into a per-block sequencer in the TCU. The remaining
front-end coupling is: the per-CTA in-flight count back to issue (for
`tmem.ld/st`), `VX_bar_unit` phase-complete events into the TCU, completion
on the existing txbar bus, and job-queue back-pressure on `tj.post`.

In a TMEM-only build (WMMA and register-accumulator WGMMA disabled), the TCU
path in `VX_lane_dispatch` needs one 1024-bit operand (`tmem.st`) instead of
three (`rs1`/`rs2`/`rs3`), plus the `tmem.ld` result path.

## Operand buffers and layout

`tbuf` area is layout steering, not storage (V80, 1 core, NT=32: `bbuf` 55k
LUTs for 2048 FFs; each `abuf` 14.7k LUTs for 4096 FFs), and K-major B costs
`TC_N` LMEM reads per B block. TMEM jobs accept one TCU-native LMEM layout,
written by a new DXA layout mode:

- One B block (`FEDP_K x TC_N`) is one LMEM bank row, in FEDP lane order.
- One block's A slice (16 x `FEDP_K`) is four consecutive bank rows, two per
  M step.

`bbuf` becomes a double-buffered 1024-bit pipeline register with prefetch;
`abuf` becomes a 4096-bit stationary register with a 2:1 M-step select.

## Unchanged

WMMA, WGMMA (register accumulator), `TCU_LD`, DXA issue, and barriers.

## Extensions

- Fine-grained firing: start a K-tile as its bank rows land rather than on
  the stage barrier, shortening fill and inter-stage gaps.
- Readiness scheduling: run whichever queued job (across CTAs) has its next
  stage ready, hiding one job's DXA latency behind another's compute.
- A from TMEM (`idesc[12]`): replaces `abuf`, frees A's LMEM bandwidth, needs
  a wider TMEM read port.
- Multicast in CLOSED mode.
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

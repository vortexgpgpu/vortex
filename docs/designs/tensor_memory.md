# Tensor Memory (TMEM) and the `tc.*` Engine

**Scope:** the per-core tensor memory, the single-thread asynchronous MMA
engine that writes it (`tc.mma`, `tc.cp`, `tc.commit`), the warp transfers
(`tmem.ld` / `tmem.st`), allocation, and the launch support (TMEM budget,
block-aligned warp placement). RTL:
[`VX_tcu_tmem.sv`](../../hw/rtl/tcu/VX_tcu_tmem.sv),
[`VX_cta_dispatch.sv`](../../hw/rtl/core/VX_cta_dispatch.sv),
[`VX_kmu.sv`](../../hw/rtl/VX_kmu.sv). SimX:
[`tcu_unit.cpp`](../../sim/simx/tcu/tcu_unit.cpp),
[`cta_dispatcher.cpp`](../../sim/simx/cta_dispatcher.cpp). Software:
[`vx_tmem.h`](../../sw/kernel/include/vx_tmem.h).

The capability set follows Blackwell's `tcgen05`: accumulators in TMEM
instead of registers, a single thread issuing asynchronous MMAs, completion
signalled to a barrier, sparse metadata in TMEM, A from TMEM, and
shared-memory-to-TMEM copies. The hardware is built from the TCU's existing
pieces (the WGMMA tile buffers, the blocks' FEDP arrays, the barrier unit);
what is TMEM-specific is the storage, the shape logic on the transfer path,
a small op queue and the allocator.

---

## 1. Geometry and storage

TMEM is `NUM_TCU_BLOCKS * xtileM` lanes by `VX_CFG_TCU_TMEM_COLS` 32-bit
columns. A lane is an M row of an MMA and a column an N element. TCU block
`b` computes, and stores, lanes `[b*xtileM, (b+1)*xtileM)`.

Storage is **fragment-major**. A storage row holds one C/D fragment: `tcM`
lanes by `tcN` columns, word `i*tcN + j` = lane `m*tcM + i`, column
`g*tcN + j`. Column group `g`'s two M steps are row `g` of two RAMs per
block (`VX_dp_ram`, `NT*32` bits wide, `COLS/tcN` deep). An MMA step reads
C from, and writes D to, one row of one RAM with identity wiring; the
transfer shapes (§3) are the only place data is rearranged. D, metadata and
A-from-TMEM columns, and transfer column addresses, are multiples of `tcN`.

At XLEN=64, `tmem.ld` NaN-boxes the 32-bit cells into the FP registers.

## 2. ISA

| Instruction | Operands | Semantics |
|---|---|---|
| `tc.mma` (custom-0, funct7 6, funct3 4) | rs1 idesc, rs2 A, rs3 (rd field) B | One K step: D (+)= A·B, asynchronous, from thread 0. |
| `tc.cp` (funct3 2) | rs1 `{cols, col}`, rs2 source | Copies pre-packed fragment rows from shared memory, in order with `tc.mma`. |
| `tc.commit` (funct3 1) | rs1 barrier, rs2 arrival count | Arrives once the core's earlier `tc.mma` / `tc.cp` have completed. |
| `tmem.ld` / `tmem.st` (funct7 5, funct3 0/1, +4 = fragment shape) | rs1 taddr, rd FP base, rs2 count-1 | Warp transfer to/from the issuing block's lanes. |
| `tmem.alloc` (funct3 6) | rs1 columns | First column of a power-of-two run, owned by the CTA, in thread 0 only; waits until free. |
| `tmem.dealloc` (funct3 7) | rs1 column, rs2 columns | Frees the CTA's run. |

**Instruction descriptor** (`tc.mma` rs1, `tmem::make_idesc<IType>`): `fmt_s` [4:0],
sparse [5], A in TMEM [6], `acc_init` [7] (D is overwritten, Blackwell's
`enable_input_d` = 0), `n_frag - 1` [14:8], D's first fragment group [22:15],
metadata column [31:23]. The output format is fp32 (int32 for integer
inputs). A and B are the WGMMA shared-memory descriptors
(`tensor::smem_matrix_desc`, offset and leading bytes, never 0: A is
row-major and B K-major, as DXA loads them); with A in TMEM (a `tmem::tmem_col` operand, which sets
bit 6), rs2 is A's TMEM column. There is no descriptor state: every `tc.mma`
carries its whole description, and software loops over K, one instruction
per `fedpK` words of K, as on Blackwell.

**`tc.cp` source** is TMEM's storage order: column group by group, block by
block, M step by M step, NT words each. The copy moves whole LMEM bank rows
(the source and size are bank-row multiples), each response writing whole
fragment rows with fixed wiring.

**Sparse metadata** is stored pre-packed: for each K step a lane holds, in
one fragment group from the metadata column, the `VX_tcu_sp_mux` rows of the
step's lower then upper `TC_K` words. It is read once per instruction.

**MX scale factors** (`fmt_s` an MX format) use the same side column. A step
takes `SF = TCU_MX_MAX_SF` scales per A row and per B column
(`tensor::mx_max_fedp_sf(fedpK)`); scale `s` covers the step's K elements
from `s * step_elems / SF`. From the side column:

- A: lane `l` (an M row), columns `side + s/4`, byte `s % 4`: one fragment
  group, read by the prologue with the metadata's read and latched.
- B: a dense byte stream, byte `q = col * SF + s`, in rows of `NT*4` bytes;
  row `r` is M step 1 of block `r % NUM_TCU_BLOCKS`, at group
  `side/TC_N + 1 + r / NUM_TCU_BLOCKS`. A step `(m = 0, n)` whose `n` starts a
  row reads it on the RAM its C read leaves free; every block takes it, as
  they all take B.

So the scales cost no MMA cycles and no per-step latch beyond one row. A
kernel copies them in with `tc.cp` like the metadata
(`tests/regression/sgemm_tcu_tmem_mx`).

## 3. Transfers (`tmem.ld` / `tmem.st`)

A warp's transfers reach only its own block's lanes: warp `w` issues to
block `w % ISSUE_WIDTH`, and `taddr`'s lane is that block's first. Each
block has its own transfer port; nothing crosses blocks. One uop moves one
register:

- **Lane per thread** (`LANE_PER_THREAD`): thread `t` holds lane `t % xtileM`;
  register `r`, column `r*TPL + t/xtileM`. One read or write of both RAMs at
  one row; a per-thread `tcN/TPL`:1 select. FlashAttention's softmax uses it
  (a thread owns a row, so max and sum need no shuffles).
- **Fragment** (`FRAGMENT`): register `r` is the fragment of M step `r % 2`,
  columns `(r/2)*tcN`, in the WGMMA accumulator register layout. One read or
  write of one RAM, identity data. Epilogues use it.

A step's D write-back, then a `tc.cp` row, have a block's write port before
a store; a transfer read has the read port before the engine (which loses
the cycle).

## 4. The engine

`tc.mma`, `tc.cp`, `tc.commit` and the allocation ops reach a scalar-only
port (thread 0's operands; a block arbiter). `tc.mma` and `tc.cp` enter a
4-deep in-order queue. A `tc.commit` rides on the youngest op that has not
left the engine (up to two per op); with none it queues as an entry of its
own.

The engine runs the queue head:

- **`tc.mma`:** a cycle to set the descriptors, a prologue that reads the
  metadata row and/or A's `fedpK/tcN` fragment groups (all blocks in
  parallel), then the (m, n) walk, one step per cycle on all blocks in
  lockstep over the tile buffers (A stripes per block, B shared). A step waits while its C fragment's write-back
  is in flight (the consecutive-MMA row hazard, the only hazard checked in
  hardware).
- **`tc.cp`:** waits for the steps in flight to be written back, then reads
  one bank row at a time through the tile buffers' LMEM port.

An op leaving the engine posts its commits with the count of steps issued
so far; a commit arrives on its barrier once that many steps have been
written back, one arrival per cycle through `VX_tcu_bar_if`.

## 5. Ordering contract

As on Blackwell, ordering between transfers and MMAs is the software's:
`tmem.ld` of columns an MMA writes needs that MMA's `tc.commit` and a
barrier wait; an MMA reading columns a warp stored needs `tmem::wait_st`
(`vx_wsync`) and a barrier first. The hardware has no range interlock. SimX
checks the contract: a transfer or dealloc touching columns of an
unsettled op of its CTA aborts with the op it raced.

**Undefined behavior.** As in PTX, misuse of tensor memory is undefined:
the hardware does not check it, and the result is unspecified (it may
overwrite other columns, another CTA's included, or wait forever). RTL
simulation asserts on each case below and SimX checks the ordering
contract; an FPGA reports nothing, as for the rest of the core (a
misaligned access or an illegal instruction is also checked only in
simulation). Undefined:

- a `tc.mma` whose D, A or metadata columns are unaligned or out of range,
  or whose shared-memory descriptors are invalid;
- a `tc.cp` whose columns, size or source are unaligned or out of range;
- a `tmem.alloc` or `tmem.dealloc` size that is not a power-of-two number
  of 32-column chunks within TMEM, a dealloc of columns its CTA does not
  own, and allocations beyond the launch's TMEM budget (§6);
- a transfer of another block's lanes or at a column that is not a
  multiple of `tcN`, and one issued with any thread inactive (a transfer is
  warp-wide, as `tcgen05.ld/st` are `.sync.aligned`);
- a `tc.*` or allocation op issued without thread 0 active;
- breaking the ordering contract above;
- a WGMMA instruction in a TMEM build (§7).

## 6. Allocation and launch

`tmem.alloc` waits off the scalar port until a run of its size is free, so
a waiting allocation never blocks other ops. Runs are tagged with the owner
CTA, and a CTA's runs are freed when its last warp retires; a dealloc of
another CTA's run is undefined (§5).

A kernel declares its CTAs' TMEM in `vx_launch_info_t::tmem_cols` (the KMU
DCR `VX_DCR_KMU_TMEM_COLS`). The budget, rounded up to a power-of-two run,
counts toward occupancy like LMEM, so co-resident CTAs' allocations always
fit and an allocation within budget never waits on a CTA that is not
running. A launch with a budget also places its warps block-aligned: rank
`r` runs on launch lane, and TCU block, `r % ISSUE_WIDTH` (TMEM builds have
one launch lane per block). The runtime writes the DCR only when a budget
is set or was set, so other launches are unchanged.

## 7. TMEM builds and WGMMA

A TMEM build drives the tile buffers from the engine, so warps issue no
WGMMA: the per-warp WGMMA expansion and its front-end logic are left out
(`TCU_WARP_WGMMA`), and a WGMMA instruction is undefined (§5): simulation
rejects it at decode. The tile buffers keep only the layouts a `tc.mma`
takes, row-major A and K-major B, which DXA loads.

WMMA stays unless `VX_CFG_TCU_WMMA_ENABLE` is off (`-DVX_CFG_TCU_WMMA_DISABLE`,
TMEM builds only). Without it, `tcu.wmma` and `tcu.ld` are undefined and
rejected at decode, and every core op is a tensor op's step: the core reads
its operands from the tile buffers only, its results go straight to TMEM
with no landing queue or ordering queue, and sparse and MX builds lose the
per-block metadata store (tensor ops take metadata and scales from TMEM).
Beacon on `tcuw` (§8 configuration): 106,551 → 101,447 LUT and 62,412 →
61,220 FF dense; sparse and every-format builds save 10.9k and 20.4k LUT
and 1,592 flip-flops. No real Blackwell kernel mixes `mma.sync` with
`tcgen05` in one launch, so a TMEM kernel loses nothing.

## 8. Results

4 TCU blocks, 8 warps, issue width 4, FEDP2K, fp16 → fp32, TMEM 512
columns (NT=8: 32 lanes, 64 KB; NT=32: 128 lanes, 256 KB). Area: Beacon on
the `tcuw` DUT (`VX_tcu_unit` + `VX_tcu_shared` at full issue width, TFR
FEDP; `make tcuw-beacon` in `hw/syn/xilinx/dut`). Cycles: rtlsim, SimX in
parentheses, NT=8 with the configs of `ci/testcases/tensor_wg.yaml` (16
warps).

**Area against the same build without TMEM** (LUT / FF; TMEM adds 32 block
RAMs at NT=8 and 116 at NT=32):

| Build | WGMMA only | TMEM | TMEM, no WMMA (§7) |
|---|---|---|---|
| NT=8 FP16 | 100,302 / 60,960 | +6.2% / +2.4% | +1.1% / +0.4% |
| NT=8 sparse | 109,580 / 61,360 | +12.6% / +5.7% | +2.7% / +3.1% |
| NT=8 every format | 1,121,926 / 263,728 | +1.4% / +1.8% | −0.1% / +1.2% |
| NT=32 FP16 | 680,638 / 378,042 | +3.0% / +0.4% | −1.6% / −0.7% |
| NT=32 sparse | 744,037 / 378,538 | +12.5% / +2.6% | +0.5% / +1.3% |

Most of the sparse cost is the full-K step's second row of 2:4 selects and
its upper-half B gather; a WGMMA-only build shares one select row per FEDP
row.

| Where they go, FP16 (TMEM − WGMMA only) | NT=8 LUT | NT=8 FF | NT=32 LUT | NT=32 FF |
|---|---|---|---|---|
| `VX_tcu_tmem` | 4,790 | 1,846 | 12,225 | 2,061 |
| A buffers: the A-from-TMEM fill (4 blocks) | 936 | 0 | 4,052 | 0 |
| B buffer: K-major row merging, wide N step | −26 | 41 | 1,242 | 43 |
| Tile-buffer LMEM port: the `tc.cp` master | 105 | 1 | 397 | 1 |
| Blocks: engine steps beside warp ops | 912 | 0 | 3,044 | 0 |
| Blocks: completion order queue | 76 | 76 | 76 | 76 |
| Cores | −256 | −240 | −460 | −336 |
| Warp WGMMA logic left out (§7) | −297 | −272 | −297 | −272 |

**Inside `VX_tcu_tmem`.** Flip-flops are exact (register widths in the
mapped netlist). LUTs are the netlist's LUTs by source line, scaled to
Beacon's module total, so they are approximate.

| Part | NT=8 LUT | NT=8 FF | NT=32 LUT | NT=32 FF |
|---|---|---|---|---|
| RAM write data (D, `tc.cp` row, store in either shape) and port control | ~1,730 | 0 | ~7,290 | 0 |
| Op slots: one registered op per block, without the store data | ~870 | 744 | ~1,040 | 840 |
| `tmem.ld` / `tmem.st` transfer units | ~440 | — | ~1,250 | — |
| Response headers and the load/alloc result | — | 391 | — | 511 |
| Engine: decode, walk, step register, prologue, write-back queues | ~640 | 348 | ~1,400 | 347 |
| Shared next-state logic | ~460 | — | ~470 | — |
| Op queue and commits (attach, post, fire) | ~290 | 258 | ~400 | 258 |
| Allocator | ~270 | 64 | ~300 | 64 |
| Row-hazard check | ~50 | 24 | ~55 | 24 |
| `tc.cp` sequencing | ~30 | 17 | ~15 | 17 |

The RAM write data is one 6-input LUT per bit: a 4-way select whose
encoding is computed once per RAM. A `tmem.st` writes from the block's
dispatch output, which holds its data until accepted, so the op slot keeps
only thread 0's operands. `tmem.alloc` returns its column in thread 0 only,
which thread 0 stores to shared memory for the CTA, as `tcgen05.alloc` does.

**Builds without TMEM** keep their cycle counts: every non-TMEM perf gate is
unchanged. The tile buffers' K-major write path, a fixed shifter per storage
word, makes WGMMA builds smaller (logic gate `tcu` DUT 117,205 → 96,340 LUT,
same flip-flops and depth).

**Cycles:**

| Kernel | TMEM | WGMMA only | Speedup |
|---|---|---|---|
| GEMM 128×128×128 | 47,036 (48,328) | 113,243 (113,900) | 2.41× |
| GEMM 128×128×512 | 184,044 (185,630) | 454,704 (457,421) | 2.47× |
| Sparse GEMM 64×64×64, metadata by `tc.cp` | 8,191 (8,509) | 35,475 (34,047) | 4.33× |
| FlashAttention-4, 4 tiles (`-l64 -n32 -d64 -b4`) | 61,935 (61,588) | — | — |

The WGMMA builds drop TMEM and run `sgemm_tcu_wg_dxa` (`WGMMA_NRC=32`), the
kernel `sgemm_tcu_tmem` derives from, and `sgemm_tcu_wg_sp`. No WGMMA kernel
matches FlashAttention-4: `flash_attn_tcu_wg` computes softmax with
software `expf` and does not split softmax and MMA across warps.

## 9. Not covered

- WGMMA and TMEM in one build.
- `.ws`, `cta_group::2`, `shift`, multicast commit.

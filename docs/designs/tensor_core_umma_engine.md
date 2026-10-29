# Tensor Core Unit (TCU / UMMA + TMEM) — Design

**Scope:** UMMA (Unified/warpgroup Matrix-Multiply-Accumulate with a
Tensor Memory accumulator, modeled on NVIDIA Blackwell's `tcgen05`) and
TMEM (Tensor Memory) — the accumulator-in-scratchpad extension to
Vortex's TCU. UMMA rides the TCU's existing WGMMA pipeline end-to-end
(operand fetch, lockstep gating, FEDP compute); this document assumes
that machinery and covers only what's new for UMMA/TMEM. See
[`tensor_core_wgmma_engine.md`](tensor_core_wgmma_engine.md) for the
underlying WGMMA/tile-buffer design. Covers the RTL
([`hw/rtl/tcu/`](../../hw/rtl/tcu/)), the SimX functional/timing model
([`sim/simx/tcu/`](../../sim/simx/tcu/)), and the SW surface
([`sw/kernel/include/vx_tensor.h`](../../sw/kernel/include/vx_tensor.h)).

UMMA/TMEM is gated by `VX_CFG_TCU_TMEM_ENABLE` and requires
`VX_CFG_TCU_WGMMA_ENABLE` (it reuses WGMMA's dense-SS compute path).
Configured by the `[tcu]` block in
[`VX_config.toml:236-258`](../../VX_config.toml#L236).

---

## 1. Architecture overview

UMMA is a variant of WGMMA that reads its C operand from, and writes
its result to, **TMEM** (a dedicated scratchpad private to the TCU)
instead of the register file. Everything upstream of the FEDP compute
stage — operand dispatch, tile-buffer fetch, lockstep gating — is
unmodified WGMMA machinery:

```
  VX_dispatch_if[ISSUE_WIDTH]                       VX_tcu_unit
  ──────────────────────────►  ┌──────────────────────────────────────────────┐
                               │  VX_lane_dispatch → per_block_execute_if[Q]    │
                               │                                                │
                               │   WGMMA/UMMA ──► VX_tcu_wgmma (orchestrator)  │
                               │           │            │                       │
                               │           │       VX_tcu_lockstep (CTA gate)   │
                               │           │            │                       │
                               │           │       VX_tcu_tbuf                  │
                               │           │       (Q×abuf + 1×bbuf + mem_arb)  │
                               │           ▼            │                       │
                               │     Q × VX_tcu_core ◄──┘  operands             │
                               │       (FEDP compute)                          │
                               │           │  ▲                                 │
                               │           ▼  │  TMEM read (C) / write (result) │
                               │       VX_tcu_tmem (storage + ALLOC/DEALLOC/    │
                               │                     ST/LD, backed by           │
                               │                     VX_tcu_tmem_alloc)         │
                               │           │                                    │
                               │     VX_lane_gather ──► commit_if (non-UMMA)    │
                               └──────────────────────────────────────────────┘
```

A macro UMMA op is dispatched across `ISSUE_WIDTH` issue slots, exactly
like WGMMA; the `Q = ISSUE_WIDTH` slots form one **warpgroup** of
`BLOCK_SIZE` lock-stepped blocks. UMMA's operand C and result never
reach `commit_if` — they stay entirely inside the TCU's TMEM
scratchpad, read and written by explicit `TMEM_LD`/`TMEM_ST`
instructions or implicitly by the UMMA compute op itself.

---

## 2. ISA, data types, and configuration

**Formats:** as A/B inputs (`It`), UMMA supports dense WGMMA's plain
operand formats — fp32, fp16, bf16, fp8, bf8, tf32, integer — see the
WGMMA doc §2 for the full list. As the C/D accumulate type (`Ot`), only
`fp32`/`int32`/`tf32` are correct — the three 32-bit TCU formats,
matching TMEM's fixed-32-bit-per-cell storage; narrower `Ot` is
unsupported (a project-wide gap affecting WGMMA/WMMA too, not
UMMA-specific). 
**Block-scaled MX formats are not yet functionally supported for UMMA**,
even though nothing in decode stops a kernel from encoding one:
`fmt_s`/`fmt_d` accept any 5-bit format ID including the MX ones, but
`VX_tcu_core.sv`'s MX scale-factor indexing branches only on
`is_wgmma`. Wiring `is_umma` into that
path is a future work.

**Opcodes** (custom-0, `funct7 = 2`, TCU family; `funct3` sub-selects,
[`VX_gpu_pkg.sv:609-613`](../../hw/rtl/VX_gpu_pkg.sv#L609), gated by
`VX_CFG_TCU_TMEM_ENABLE`; SW-side mirror in
[`vx_tensor.h:1190-1194`](../../sw/kernel/include/vx_tensor.h#L1190)):

| funct3 | Mnemonic | Opcode (`INST_TCU_*`) | Meaning |
|---|---|---|---|
| 3 | TMEM_ALLOC | `4'h6` | `rd = handle`, `rs1 = ncols` |
| 4 | TMEM_DEALLOC | `4'h7` | `rs1 = handle` |
| 5 | TMEM_ST | `4'h8` | `rs1 = addr`, `rs2(f) = value` — direct raw 32-bit word store into TMEM |
| 6 | TMEM_LD | `4'h9` | `rd(f) = value`, `rs1 = addr` — direct raw 32-bit word load from TMEM |
| 7 | UMMA | `4'hA` | Warpgroup MMA with TMEM-resident C/result |

Decode is in [`VX_decode.sv:652`](../../hw/rtl/core/VX_decode.sv#L652).

**TMEM address encoding.** A single 32-bit word packs `bits[31:16] =
lane_base`, `bits[15:0] = col`. `TMEM_ST`/`TMEM_LD` use this directly
(`rs1 = addr`); UMMA's own compute path computes the equivalent
lane/col pair internally per FEDP grid cell (§4.1) — kernel code never
constructs a UMMA compute address by hand.

**UMMA fixed-register convention** — read only on the macro-op's first
micro-op, all `REG_TYPE_I`:

| Register | Meaning |
|---|---|
| `x10` (a0) | A shared-memory descriptor (`desc_a`) |
| `x11` (a1) | B shared-memory descriptor (`desc_b`) |
| `x12` (a2) | TMEM handle, in `rs3` |

The register file only populates `rs3` on the first micro-op of the
macro-op — this is a hardware constraint of the TCU pipeline, not a
software convention.

**NRC (per-warpgroup N-tile width)** is a 3-bit code carried directly
in `cd_nregs` (widened from WGMMA's 2 bits):

| code | NRC |
|---|---|
| 0 | 8 |
| 1 | 16 |
| 2 | 32 |
| 3 | 64 |
| 4 | 128 |

`-DUMMA_NRC=<N>` selects NRC at kernel build time
(`tests/regression/sgemm_tcu_tmem`, §6).

**A source.** The `flags` immediate (the instruction's `rs2` field) is:

| bits | meaning |
|---|---|
| 0 | reserved for `is_sparse` (0 for now) |
| 3:1 | NRC code (0..4) |
| 4 | A from shared memory (1 = shared memory, 0 = TMEM) |

WGMMA's width field (`cd_nregs`) is 2 bits, so its A-source bit is at 3,
while UMMA widens that field to 3 bits, so its A-source bit lands at 4.
The largest UMMA encoding is 24, inside the 5-bit `rs2` field.

B is always shared-memory-sourced.

When A comes from TMEM, `a0` carries an absolute TMEM **column** rather
than a shared-memory descriptor. There is no lane field: A's rows are
the same TMEM lanes the accumulator uses, so element `(row, k)` is at

```
TMEM[cta_rank*xtileM + row][a_col + k / elems_per_word]
```

with the same `elems_per_word` packing the shared-memory path uses, so a
sub-32-bit `It` packs several k elements into one TMEM word.

Status: the ISA and decode are in place in both simx and RTL. The
operand datapath is not implemented in either: A still comes from shared
memory. Issuing the mode is rejected.

**Key config** ([`VX_config.toml:236-258`](../../VX_config.toml#L236)):

| Parameter | Meaning |
|---|---|
| `VX_CFG_TCU_TMEM_ENABLE` | Enables UMMA/TMEM (requires `VX_CFG_TCU_WGMMA_ENABLE`) |
| `VX_CFG_TCU_TMEM_COLS` | Total TMEM columns per bank (default 256, must be a power of 2); the shared capacity all live allocations divide up |

---

## 3. RTL module inventory

UMMA reuses every WGMMA module in
[`tensor_core_wgmma_engine.md`](tensor_core_wgmma_engine.md)'s module
table (`VX_tcu_wgmma`, `VX_tcu_lockstep`, `VX_tcu_tbuf`,
`VX_tcu_abuf`/`VX_tcu_bbuf`, `VX_tcu_uops`) unchanged. UMMA adds two
modules, both under `hw/rtl/tcu/`:

| Module | Role |
|---|---|
| [`VX_tcu_core.sv`](../../hw/rtl/tcu/VX_tcu_core.sv) | Shared with WGMMA (FEDP datapath); adds the TMEM read/write ports and address generation UMMA needs. |
| [`VX_tcu_tmem.sv`](../../hw/rtl/tcu/VX_tcu_tmem.sv) | TMEM storage (one bank per `cta_rank`) plus `TMEM_ALLOC`/`TMEM_DEALLOC`/`TMEM_ST`/`TMEM_LD` request handling and bank arbitration. |
| [`VX_tcu_tmem_alloc.sv`](../../hw/rtl/tcu/VX_tcu_tmem_alloc.sv) | The column allocator: free-list + per-CTA live-allocation tracking, instantiated by `VX_tcu_tmem.sv`. |

`VX_tcu_unit.sv` recognizes `INST_TCU_UMMA` alongside WMMA/WGMMA and
routes it through the same `VX_tcu_wgmma` orchestrator; the only thing
that differs at that level is where operand C and the result live.

---

## 4. Execution model

**Issue / dispatch / operand load.** Identical to WGMMA (see that
doc's §4) — UMMA is treated as WGMMA's dense-SS case throughout, with
the same `k`-outer/`n`-middle/`m`-inner micro-op ordering, the same
tile-buffer fetch machinery, and the same lockstep CTA gate. The only
structural differences from WGMMA: UMMA never has a "setup" micro-op,
its operand C comes from TMEM instead of the register file, and its
result writes to TMEM instead of to `commit_if`.

### 4.1 TMEM geometry & addressing

TMEM lanes are sized to **one warpgroup's width**
(`TCU_TMEM_LANES = NUM_TCU_BLOCKS × TCU_WG_TILE_M`) and shared/reused
across concurrent warpgroups, addressed by each warp's **CTA-local
rank** (`cta_rank`, 0-indexed within its own CTA) rather than its
physical warp ID — this lets multiple CTAs share the same
physical TMEM hardware, differentiated only by which columns their
allocation owns (mirroring `tcgen05` semantics). Address for FEDP
grid cell `(i,j)` on a warp with rank `cta_rank`, at step indices
`(step_m, step_n)`:

```systemverilog
umma_lane_base = cta_rank * TCU_WG_TILE_M + step_m * TCU_TC_M
tmem_lane      = umma_lane_base + i
tmem_col       = umma_handle + step_n * TCU_TC_N + j
```

`umma_handle` is a warp-uniform scalar, read from `rs3_data[0]` on the
macro-op's first micro-op and latched for the rest of the macro-op
(§2).

This addressing scheme fixes **CTA size == warpgroup size
(`NUM_TCU_BLOCKS`)** as an architectural assumption — matching real
Blackwell hardware (`tcgen05` warpgroups are a fixed 4 warps). On this
RTL, `NUM_TCU_BLOCKS` (= `VX_CFG_ISSUE_WIDTH`) is a build-time
parameter, not a hardware constant: `VX_config.toml`'s own global
default resolves to 1 (`VX_CFG_ISSUE_WIDTH = up(VX_CFG_NUM_WARPS /
16)`, `VX_CFG_NUM_WARPS` itself defaulting to 4), but every WGMMA/UMMA
regression test Makefile (`sgemm_tcu_wg`, `sgemm_tcu_tmem`, etc.)
overrides this itself — `CONFIGS += -DVX_CFG_ISSUE_WIDTH=4` — so in practice every such test defaults to
a 4-wide warpgroup. Checked at runtime by a
`RUNTIME_ASSERT` in `VX_tcu_core.sv` (`cta_rank >= NUM_TCU_BLOCKS`
fails loudly in simulation).

**Storage.** TMEM is `BLOCK_SIZE` (= `NUM_TCU_BLOCKS`) independent
banks, one per `cta_rank`, each a `VX_dp_ram` SRAM macro. Bank `r`
holds the entire `TCU_WG_TILE_M`-row lane range belonging to
`cta_rank r`; each bank word spans that whole row range
(`TCU_WG_TILE_M` rows × `TCU_TC_N` cols) and is addressed by
column-group.

**Arbitration.** Because a CTA's compute traffic, `TMEM_LD`, and
`TMEM_ST` can all target the same bank in the same cycle, bank access
is arbitrated per bank, independently for reads and writes. Losing
requesters stall and retry the next cycle; the `tmem_bank_stalls` perf counter counts these
events.

### 4.2 TMEM allocator (`VX_tcu_tmem_alloc.sv`)

A free-list + CAM allocator over the column dimension:

- **CTA-scoped, idempotent `TMEM_ALLOC`**: keyed by the requesting
  instruction's CTA ID. Every warp of a CTA calling `alloc()`
  independently gets the same handle back — no elected-thread
  broadcast is required in the kernel.
- **CTA-scoped barrier `TMEM_DEALLOC`**: a handle's column range only
  actually returns to the free list once every warp of the warpgroup
  (`WARPGROUP_SIZE = NUM_TCU_BLOCKS` calls) has deallocated it.
- **First-fit allocation** over the column dimension, with adjacent-
  range coalescing on free.
- A kernel requesting more concurrent TMEM columns than
  `VX_CFG_TCU_TMEM_COLS` provides is rejected before the kernel ever
  launches (§6).

`alloc()` only ever reserves a column range — it does not initialize
TMEM contents; kernels that need a defined starting value call
`vx_tmem_fill`-style helpers explicitly (§6).

### 4.3 RAW-hazard interlock

Because TMEM's accumulator is read-modify-write across the k-loop, an
explicit address-compare interlock in `VX_tcu_core.sv` stalls
admission of any new UMMA compute op whose TMEM tile address collides
with one still in flight in the FEDP pipeline. This makes correctness independent of micro-op
ordering timing margins.

---

## 5. SimX model

[`sim/simx/tcu/tcu_unit.cpp`](../../sim/simx/tcu/tcu_unit.cpp)'s
`umma()` mirrors the RTL design directly: lane/handle addressing uses
`core_->scheduler().warp(wid).cta_csrs.cta_rank` (a persistent per-warp
member). `kTmemLanes = wg_cfg::xtileM * VX_CFG_NUM_TCU_BLOCKS` — the
same per-warp-tile-height basis as RTL's `TCU_TMEM_LANES`. The
free-list allocator (`tmem_alloc`/`tmem_dealloc`/`cta_uid`) implements
the same idempotent-alloc, barrier-dealloc, first-fit-coalescing
semantics as the RTL allocator (§4.2), and additionally performs a
`validate_tmem_lane_col`-style bounds check on every TMEM access that
RTL does not.

`wid` is used for genuinely per-physical-warp
bookkeeping elsewhere in `umma()` (descriptor caching, handle caching)
— only the TMEM-lane addressing itself uses `cta_rank`.

---

## 6. SW surface

[`sw/kernel/include/vx_tensor.h`](../../sw/kernel/include/vx_tensor.h),
under `#ifdef VX_CFG_TCU_TMEM_ENABLE`:

- `vx_tmem_alloc(ncols) -> handle`, `vx_tmem_dealloc(handle)` —
  CTA-scoped and idempotent; every warp of a CTA calls it directly and
  all get the same handle back.
- `vx_tmem_st(addr, uint32_t value)` / `vx_tmem_ld(addr) -> uint32_t` —
  direct raw 32-bit word store/load into TMEM by address (§2). Moves the
  value's bits as-is through the float regfile at the ISA level; `Ot`'s
  accumulate domain (fp32 vs. int32) is determined by the TCU's input
  format, not by this function.
- `umma_context<NT, It, Ot, NRC_>` — reuses `wgmma_context`'s tile
  geometry (`tcM`, `tcN`, `xtileM`, `xtileN`, `tileK`, `n_steps`, etc.).
  - `umma_sync(op_a, desc_b, handle)` issues the macro-op.
  - `fill_tmem(handle, value)`, `store_output(handle, pC, ...)` —
    helpers; both address TMEM via `vx_cta_rank()` (the
    warp's CTA-local rank intrinsic).

**Device capability query.** A kernel should check the device's TMEM
column budget before choosing NRC and CTA occupancy:

```cpp
uint64_t tmem_cols;
vx_device_query(device, VX_CAPS_TCU_TMEM_COLS, &tmem_cols);
```

`tmem_cols` is 0 if `VX_CFG_TCU_TMEM_ENABLE` is off. A kernel launch
whose `UMMA_NRC × max-concurrently-resident-CTAs` exceeds `tmem_cols`
structurally cannot fit — `tests/regression/sgemm_tcu_tmem/main.cpp`
performs this check and refuses to launch.

**Test kernel:**
[`tests/regression/sgemm_tcu_tmem/`](../../tests/regression/sgemm_tcu_tmem/)
— an SGEMM kernel built on `umma_context`, comparable in structure to
the WGMMA doc's `sgemm_tcu_wg` example.

```bash
cd /vortex/build_test
CONFIGS="-DUMMA_NRC=<NRC>" ./ci/blackbox.sh --driver=rtlsim --warps=<W> --threads=<T> --app=sgemm_tcu_tmem --args="-m <M> -n <N> -k <K>"
```
or
```bash
cd /vortex/build_test
MAKEFLAGS=-s make -C tests/regression/sgemm_tcu_tmem run-rtlsim \
  CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_NUM_WARPS=<W> -DVX_CFG_ISSUE_WIDTH=<W> -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_TMEM_ENABLE -DUMMA_NRC=<NRC>" \
  OPTS="-m <M> -n <N> -k <K>"
```

`blackbox.sh` sets `CONFIGS` as a
shell environment variable ahead of `make`, so the Makefile's
default flags (`VX_CFG_EXT_TCU_ENABLE`, `VX_CFG_TCU_WGMMA_ENABLE`,
`VX_CFG_TCU_TMEM_ENABLE`, `VX_CFG_ISSUE_WIDTH=4`, ...) still apply.

Swap `rtlsim` for `simx` to use the SimX driver. `<NRC>` ∈
{8, 16, 32, 64, 128}.

---

## 7. Known limitations

- **No sparse UMMA** — there is no `*_SP` UMMA opcode; 2:4 structured
  sparsity is WGMMA-only.
- **MX UMMA is not functionally correct, despite being encodable** — see
  §2's Formats note: decode doesn't reject an MX format ID on a UMMA
  op, but the RTL's scale-factor indexing isn't wired for UMMA yet, so
  the result would be wrong.
- **`TMEM_ST`/`TMEM_LD` bounds checking (RTL)**: `VX_tcu_tmem.sv`
  checks, per access, that the lane is physically valid and the column
  falls within *some* live allocation — a `RUNTIME_ASSERT`, so it's
  simulation-only, mirroring SimX's `validate_tmem_lane_col`. Neither
  confirms the column belongs to the *requesting* CTA's own allocation,
  just that it isn't dead space.
- **CTA size must equal warpgroup size (`NUM_TCU_BLOCKS`)** — enforced
  by a runtime assertion in simulation.
- **Accumulate type (`Ot`) restricted to `fp32`/`int32`/`tf32`** — the
  three 32-bit TCU formats, matching TMEM's fixed-32-bit cell storage.
  Narrower `Ot` (`bf16`, `bf8`, etc.) remains unsupported — not a
  TMEM-specific gap: WGMMA/WMMA's own narrow-`Ot` path is unverified
  against RTL.

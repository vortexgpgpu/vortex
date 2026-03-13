# Int8 Kernel Dump Comparison: Dense vs Sparse TCU (NT=32)

**Config**: NT=32, ITYPE=int8, OTYPE=int32
**Date**: 2026-03-13
**Build**: branch `260313_after_int4_kernel_opt`, commit `f357a5f9b`

## Configuration Parameters (int8/int32, NT=32)

From `tensor_cfg.h` (`wmma_config_t<32, int8, int32>`):

| Parameter | Dense | Sparse | Notes |
|-----------|-------|--------|-------|
| tileM, tileN | 16, 16 | 16, 16 | Same |
| tileK (register-element units) | 16 | 16 | `xtileK` |
| tileK (actual int8 elements) | 64 | 64 | `tileK * i_ratio`, i_ratio = 4/sizeof(int8) = 4 |
| tcM, tcN, tcK | 8, 4, 4 | 8, 4, 4 | Micro-tile for single MMA uop |
| m_steps, n_steps, k_steps | 2, 4, 4 | 2, 4, 4 | |
| NRA | 8 | 8 (only 4 used + 1 meta) | Sparse uses sparse_regs=4 for A data |
| NRB | 8 | 8 | Same count, different data layout |
| NRC | 8 | 8 | Same |
| i_ratio | 4 | 4 | `sizeof(vreg_element)/sizeof(int8) = 4/1` |
| b_block_size | 16 | 32 (b_block_size_sp) | Dense: tcK*tcN=4*4=16, Sparse: tcK*tcN*2=32=NT |
| b_sub_blocks / b_sub_blocks_sp | 2 / — | — / 1 | Dense: NT/b_block=32/16=2, Sparse: NT/32=1 |
| b_sub_steps / b_sub_steps_sp | 2 / — | — / 4 | Dense: n_steps/2=2, Sparse: n_steps/1=4 |
| sparse_regs | — | 4 | m_steps * (k_steps/2) = 2*2 |
| sp_meta_cols | — | 8 | (NT*2*i_ratio_rtl)/32 = (32*2*4)/32 |
| sp_num_meta_loads | — | 1 | ceil(8/(32/4)) = ceil(8/8) = 1 |
| meta_stride | — | 32 | num_meta_loads * NT = 1*32 |
| a_k_stride_sp | — | 32 | tileK_actual/2 = 64/2 |

Key int8-specific detail: `sizeof(int8) = 1 < sizeof(vreg_t) = 4`, so B-load uses byte-level `pack_row` (4× `lbu` + 3× `slli` + 3× `or` + `fmv.w.x` per register). A-load uses aligned `flw` (int8 A is stored row-major contiguous, 4 bytes per word).

## Register Allocation

| Register | Dense | Sparse |
|----------|-------|--------|
| ft0-ft7 (f0-f7) | fragC accumulator (NRC=8) | fragC accumulator (NRC=8) |
| fa0-fa3 (f10-f13) | fragA data [0:3] | fragA compressed data [0:3] |
| fa4-fa7 (f14-f17) | fragA data [4:7] | fa4: metadata (1 load), fa5-fa7: unused |
| fs8-fs11 (f24-f27) | fragB data [0:3] | fragB data [0:3] |
| ft8-ft11 (f28-f31) | fragB data [4:7] | fragB data [4:7] |

Key difference: Dense loads 8 A registers (full K), sparse loads only 4 (compressed K/2) + 1 metadata register.

## Instruction Count Summary

| Region | Dense | Sparse | Delta |
|--------|-------|--------|-------|
| **Prologue** (stack frame) | 13 | 11 | -2 |
| **Setup** (GOT + args + fragC + mcycle) | 22 | 24 | +2 |
| **Pre-loop setup** | 8 | 13 | +5 |
| **Loop body** (per iteration) | 180 | 151 | -29 |
| **Epilogue** (store_D + cycles + restore) | 48 | 45 | -3 |
| **Total static** | **271** | **244** | **-27** |

Sparse has **27 fewer static instructions**. The dominant savings is the **loop body** — 29 fewer instructions per K-tile iteration. This is driven almost entirely by the B-load difference (29 fewer insns from stride-chaining vs independent `mul` per column).

## Section-by-Section Analysis

### 1. Prologue — Stack Frame & Callee-Saved Registers

**Dense** (13 insns, lines 50-62):
```asm
addi  sp, sp, -0x30
sw    s0, 0x2c(sp)     ; \
sw    s1, 0x28(sp)     ;  |
sw    s2, 0x24(sp)     ;  | 7 integer callee-saved
sw    s3, 0x20(sp)     ;  |
sw    s4, 0x1c(sp)     ;  |
sw    s5, 0x18(sp)     ;  |
sw    s6, 0x14(sp)     ; /
fsw   fs8, 0x10(sp)    ; \
fsw   fs9, 0xc(sp)     ;  | 4 float callee-saved
fsw   fs10, 0x8(sp)    ;  |
fsw   fs11, 0x4(sp)    ; /
sw    a0, 0x0(sp)      ; save kernel_arg_t pointer
```

**Sparse** (11 insns, lines 50-60):
```asm
addi  sp, sp, -0x30
sw    s0, 0x2c(sp)     ; \
sw    s1, 0x28(sp)     ;  |
sw    s2, 0x24(sp)     ;  | 5 integer callee-saved
sw    s3, 0x20(sp)     ;  |
sw    s4, 0x1c(sp)     ; /
fsw   fs8, 0x18(sp)    ; \
fsw   fs9, 0x14(sp)    ;  | 4 float callee-saved
fsw   fs10, 0x10(sp)   ;  |
fsw   fs11, 0xc(sp)    ; /
sw    a0, 0x8(sp)      ; save kernel_arg_t pointer
```

**Diff (-2 insns)**: Dense saves s0-s6 (7 registers), sparse saves only s0-s4 (5 registers). The dense B-load's per-column address computation requires more temporaries (s5, s6) as callee-saved scratch.

---

### 2. Setup — GOT, Argument Loading, fragC Zeroing, Cycle Measurement

**Dense** (22 insns, lines 63-84):
```asm
lw    t1, 0x20(a0)       ; A_addr
lw    t2, 0x28(a0)       ; B_addr
auipc a1, 0x7             ; GOT base
lw    a1, 0x508(a1)       ; blockIdx pointer
lw    a4, 0x30(a0)        ; C_addr
lw    a2, 0x14(a0)        ; N
lw    a7, 0x18(a0)        ; K
add   a1, a1, tp          ; resolve thread-local blockIdx
lw    a5, 0x4(a1)         ; blockIdx.y
lw    a6, 0x0(a1)         ; blockIdx.x
fmv.w.x fa5, zero         ; \
fmv.s ft0, fa5            ;  |
...                        ;  | fragC = 0 (9 insns)
fmv.s ft7, fa5            ; /
csrr  a3, mcycle          ; start_cycles
slli  a5, a5, 0x4         ; blockIdx.y * 16 (tileM=16)
slli  a6, a6, 0x4         ; blockIdx.x * 16 (tileN=16)
```

**Sparse** (24 insns, lines 61-84):
```asm
lw    t6, 0x20(a0)        ; A_addr
lw    t4, 0x28(a0)        ; B_addr
lw    a4, 0x30(a0)        ; C_addr
auipc a1, 0x7              ; GOT base
lw    a1, 0x4a0(a1)       ; blockIdx pointer
lw    t5, 0x38(a0)        ; meta_addr        ← sparse only
lw    a3, 0x14(a0)        ; N
lw    t0, 0x18(a0)        ; K
add   a1, a1, tp          ; resolve thread-local blockIdx
lw    a6, 0x4(a1)         ; blockIdx.y
lw    a7, 0x0(a1)         ; blockIdx.x
fmv.w.x fa5, zero         ; \
fmv.s ft0, fa5            ;  |
...                        ;  | fragC = 0 (9 insns)
fmv.s ft7, fa5            ; /
csrr  a2, mcycle          ; start_cycles
slli  a6, a6, 0x4         ; blockIdx.y * 16
slli  a7, a7, 0x4         ; blockIdx.x * 16
slli  a5, a3, 0x5         ; a5 = N * 32      ← pre-compute store_D stride
```

**Diff (+2 insns)**:
- Sparse loads `meta_addr` (+1 `lw`)
- Sparse pre-computes `N * 32` for store_D stride (+1 `slli`), saves 1 insn in epilogue
- Both have `csrr mcycle` for cycle measurement (unlike int4 where sparse omitted it)

Register assignment after setup:

| Register | Dense | Sparse |
|----------|-------|--------|
| a1 | blockIdx base (kept for epilogue) | blockIdx base (kept for epilogue) |
| a2 | N | start_cycles |
| a3 | start_cycles | N |
| a4 | C_addr | C_addr |
| a5 | blockIdx.y * 16 | N * 32 (store_D stride) |
| a6 | blockIdx.x * 16 | blockIdx.y * 16 |
| a7 | K | blockIdx.x * 16 |
| t0 | — | K |
| t1 | A_addr | — |
| t2 | B_addr | — |
| t4 | — | B_addr |
| t5 | — | meta_addr |
| t6 | — | A_addr |

---

### 3. Pre-Loop Setup — Pointer Computation

**Dense** (8 insns, lines 85-92):
```asm
beqz  a7, exit             ; skip loop if K == 0
li    t0, 0x0               ; loop index i = 0
mul   t3, a5, a7            ; tile_row * K (for A pointer)
add   t1, t3, t1            ; A_base = A_addr + tile_row * K
addi  t1, t1, 0x30          ; + 48 for negative-offset addressing
add   t2, a6, t2            ; B_base = B_addr + tile_col
addi  t2, t2, 0x8           ; + 8 for negative-offset addressing
slli  t3, a2, 0x6           ; t3 = N * 64 (B stride per K-tile)
```

Dense computes: `pTileA = A + tile_row*K + loop_i`, `pTileB = B + tile_col + loop_i * N`. A uses counter-based offset (`t1 + t0`), B advances by `t3 = N*64` each iteration.

**Sparse** (13 insns, lines 85-97):
```asm
blez  t0, exit              ; skip loop if K <= 0
li    t1, 0x0                ; loop counter = 0
srli  t2, t0, 0x1            ; t2 = K/2 (compressed A stride)
slli  t3, a3, 0x6            ; t3 = N * 64 (B stride per K-tile)
lw    s0, 0x4(a1)            ; blockIdx.y (re-read for meta base)
add   t4, t4, a7             ; B_base = B_addr + tile_col
mul   s1, a6, t2             ; tile_row * (K/2) (A offset)
andi  s2, t2, -0x20          ; K/2 rounded down to multiple of 32
mul   s0, s2, s0             ; (K/2_rounded) * blockIdx.y
slli  s0, s0, 0x2            ; * sizeof(float) for meta byte offset
add   t5, t5, s0             ; meta_base for this tile row
add   t6, s1, t6             ; A_base = A_addr + tile_row * (K/2)
addi  t6, t6, 0x10           ; + 16 for negative-offset addressing
```

**Diff (+5 insns)**: Sparse must:
1. Compute `K/2` for compressed A stride (+1 `srli`)
2. Compute metadata base pointer: `blockIdx.y * (K/2_rounded) * 4` (+4 insns: `lw`, `andi`, `mul`, `slli`)

Register assignment after pre-loop:

| Register | Dense | Sparse |
|----------|-------|--------|
| t0 | loop counter (increments by 0x40) | K (loop bound) |
| t1 | A_base (fixed) | loop counter (increments by 0x40) |
| t2 | B_base (advances by t3 per iter) | K/2 (compressed A stride, constant) |
| t3 | N * 64 (B stride, constant) | N * 64 (B stride, constant) |
| t4 | — | B_base (advances by t3) |
| t5 | — | meta_base (advances by 0x80) |
| t6 | — | A_base (advances by 0x20) |

---

### 4. Loop Body — The Core K-Tile Iteration

Both iterate over K in steps of tileK (= 64 int8 elements = 16 register-elements × 4 bytes). The loop phases are: A-load, [metadata], B-load (pack_row), MMA, loop control.

#### 4a. A-Load

Both use aligned `flw` for int8 A (4 bytes per word, row-major contiguous).

**Dense** (21 insns, lines 93-113) — loads 8 registers (fa0-fa7), full K stride:
```asm
csrr  t4, tid              ; 1
srli  t5, t4, 0x2           ; 2  block_row = tid >> 2
slli  t4, t4, 0x2           ; 3
andi  t4, t4, 0xc           ; 4  block_col = (tid & 3) * 4

; A register group 0 (fa0-fa3): m_step=0, k_step=0..3
mul   t6, t5, a7            ; 5  row * K (a7 = K)
add   t6, t6, t4            ; 6  + col
add   s0, t1, t0            ; 7  A_base + loop_i
add   t6, s0, t6            ; 8  ptr0
flw   fa0, -0x30(t6)        ; 9  A[0] (k_step=0)
flw   fa1, -0x20(t6)        ; 10 A[1] (k_step=1)
flw   fa2, -0x10(t6)        ; 11 A[2] (k_step=2)
flw   fa3, 0x0(t6)          ; 12 A[3] (k_step=3)

; A register group 1 (fa4-fa7): m_step=1, k_step=0..3
addi  t5, t5, 0x8           ; 13 row += tcM (=8)
mul   t5, a7, t5            ; 14 (row+8) * K
add   t4, t0, t4            ; 15 loop_i + col
add   t4, t4, t5            ; 16 ptr1
add   t4, t1, t4            ; 17 + A_base
flw   fa4, -0x30(t4)        ; 18 A[4]
flw   fa5, -0x20(t4)        ; 19 A[5]
flw   fa6, -0x10(t4)        ; 20 A[6]
flw   fa7, 0x0(t4)          ; 21 A[7]
```

**Sparse** (15 insns, lines 98-112) — loads 4 registers (fa0-fa3), compressed K/2 stride:
```asm
csrr  s0, tid               ; 1
srli  s1, s0, 0x2            ; 2  block_row = tid >> 2
slli  s0, s0, 0x2            ; 3
andi  s0, s0, 0xc            ; 4  block_col = (tid & 3) * 4

; A register group 0 (fa0-fa1): m_step=0, sparse_k_step=0..1
mul   s2, s1, t2             ; 5  row * (K/2) (t2 = K/2)
add   s2, s2, s0             ; 6  + col
add   s2, t6, s2             ; 7  + A_base (t6 = A base)
flw   fa0, -0x10(s2)         ; 8  A_compressed[0]
flw   fa1, 0x0(s2)           ; 9  A_compressed[1]

; A register group 1 (fa2-fa3): m_step=1, sparse_k_step=0..1
addi  s1, s1, 0x8            ; 10 row += tcM
mul   s1, t2, s1             ; 11 (row+8) * (K/2)
add   s0, t6, s0             ; 12 A_base + col
add   s0, s0, s1             ; 13 ptr1
flw   fa2, -0x10(s0)         ; 14 A_compressed[2]
flw   fa3, 0x0(s0)           ; 15 A_compressed[3]
```

**Diff (-6 insns)**: Sparse A is compressed to half — `sparse_regs = m_steps × (k_steps/2) = 2×2 = 4` vs dense `NRA = m_steps × k_steps = 2×4 = 8`. Per m_step: 2 `flw` instead of 4, and 1 fewer address computation.

#### 4b. Metadata Load (sparse only, 4 insns, lines 113-116)

```asm
csrr  s0, tid               ; 1  lane_id
slli  s0, s0, 0x2            ; 2  * sizeof(float)
add   s0, t5, s0             ; 3  meta_base + lane_id * 4
flw   fa4, 0x0(s0)           ; 4  metadata word 0
```

Only 1 `flw` because `num_meta_loads = 1` for int8 NT=32 (all 8 metadata columns fit in one load of 32 threads × 4 PD entries). This is much cheaper than int4 which needs 2 metadata loads.

Dense has no metadata — 0 instructions.

**Diff (+4 insns)**: `csrr tid` + address calc + 1 `flw`.

#### 4c. B-Load (row-major B, int8 pack_row)

This is the **dominant source of savings**. Int8 B-load packs 4 bytes per register using `lbu` + shift/or + `fmv.w.x`. The addressing pattern differs fundamentally between dense and sparse.

**Dense** (155 insns, lines 114-268) — 4 column groups × 2 registers, independent `mul` per column:

The dense B-load with `b_block_size=16` has `b_sub_blocks=2`, meaning each register pair covers a different "column group" of B. For each column group, the compiler generates independent per-column address computation:

```
Pattern per column group (first register, ~26 insns):
  addi  sX, t5, offset     ; column index (e.g., 0x10, 0x20, 0x30)
  mul   sX, a2, sX         ; column * N (row stride)
  add   sY, t2, t4         ; B_base + loop_offset
  add   sX, sY, sX         ; full column pointer
  lbu   sZ, -0x8(sX)       ; load byte
  ... (repeat for 3 more columns with addi+mul+add+add+lbu each)
  ... (3× slli + 3× or + fmv.w.x = 7 pack insns)

Pattern per column group (second register, ~11 insns):
  4× lbu from same pointers with offset 0x0
  3× slli + 3× or + fmv.w.x
```

The 155 instruction breakdown:
- tid decomposition: 4 insns (csrr + srli + slli + andi)
- Column group 0 (fs8 + fs9): 26 + 11 = 37 insns
- Column group 1 (fs10 + fs11): 27 + 11 = 38 insns
- Column group 2 (ft8 + ft9): 27 + 11 = 38 insns
- Column group 3 (ft10 + ft11): 27 + 11 = 38 insns

Each group's first register costs ~26-27 insns because it computes 4 independent column addresses with `addi + mul + add × 2-3` per column. The second register reuses pointers (offset 0x0 vs -0x8), costing only 11 insns for 4 `lbu` + pack.

**Sparse** (126 insns, lines 117-242) — single block, stride chaining with `add ptr, ptr, N`:

With `b_block_size_sp=32=NT`, all threads belong to one block. The tid decomposition uses wider fields:
```asm
csrr  s0, tid                ; row = tid >> 3 (8 rows in block)
srli  s1, s0, 0x3            ; col = (tid & 7) * 4 (8 columns in block)
slli  s0, s0, 0x2
andi  s0, s0, 0x1c
```

The key optimization: after computing the base pointer once, each subsequent column is reached by `add ptr, ptr, N` (stride chaining), replacing the `addi + mul + add + add` sequence:

```
Pattern (first register, fs8, ~20 insns):
  mul   s0, s0, a3           ; col_offset = col * N (one mul)
  add   s1, t4, s1           ; B_base + row
  add   s1, s1, s0           ; ptr_col0 = base + col*N
  add   s0, s1, a3           ; ptr_col1 = ptr_col0 + N  ← stride chain!
  lbu   s2, 0x0(s0)          ; byte from col1
  lbu   s3, 0x0(s1)          ; byte from col0
  ...
  add   s0, s0, a3           ; ptr_col2 = ptr_col1 + N  ← stride chain!
  lbu   s4, 0x0(s0)          ; byte from col2
  add   s0, s0, a3           ; ptr_col3 = ptr_col2 + N  ← stride chain!
  lbu   s0, 0x0(s0)          ; byte from col3
  ... (pack: slli + or + fmv.w.x)
  addi  s3, s1, 0x4          ; prep for next k_group
  add   s3, s3, a3           ; chain continues...

Pattern (subsequent registers, ~12-16 insns):
  lbu   s0, offset(s1)       ; byte from base + k_offset
  ... chain add + lbu for remaining columns
  ... pack + fmv.w.x
```

The 126 instruction breakdown:
- tid decomposition: 4 insns
- Address setup + fs8: 20 insns
- fs9: 15 insns
- fs10: 16 insns
- fs11: 14 insns
- ft8: 15 insns
- ft9: 15 insns
- ft10: 15 insns
- ft11: 12 insns

**Diff (-29 insns)**: The savings come from replacing `addi + mul` per-column address computation with single `add ptr, ptr, N` stride chaining:
- Dense: 4 columns × `addi + mul + add + add` = 16 insns per column group for addressing
- Sparse: 1 `mul` for base + 3 `add` for chaining = 4 insns per column group for addressing
- Net savings ≈ 12 insns per column group × 4 groups = ~48 potential, reduced by structural overhead to 29 actual.

#### 4d. MMA + Loop Control

**Dense** (4 insns, lines 269-272):
```asm
.insn r 0x0b, 0, 2, x8, x9, x0   ; mma_sync (dense, rd=int32, rs1=int8, rs2=0)
addi  t0, t0, 0x40                 ; i += 0x40 (tileK = 64 int8 bytes)
add   t2, t2, t3                   ; B_base += N * 64 (t3 = N*tileK)
bltu  t0, a7, loop_top             ; if i < K, continue
```

**Sparse** (6 insns, lines 243-248):
```asm
.insn r 0x0b, 0, 2, x8, x9, x1   ; mma_sync (sparse, rd=int32, rs1=int8, rs2=1)
addi  t5, t5, 0x80                 ; meta_ptr += 0x80 (128 = meta_stride*4 bytes)
add   t4, t4, t3                   ; B_base += N * 64
addi  t1, t1, 0x40                 ; loop_counter += 0x40
addi  t6, t6, 0x20                 ; A_ptr += 0x20 (32 = tileK/2 = a_k_stride_sp)
blt   t1, t0, loop_top             ; if counter < K, continue
```

**Diff (+2 insns)**: Sparse advances 4 pointers (meta, B, counter, A) vs dense advancing 1 counter + 1 pointer. The sparse MMA uses fused metadata — `rs2=1` triggers a 2-phase micro-op (metadata write + computation) with no separate `meta_store` instruction.

#### Loop Body Summary

| Sub-section | Dense | Sparse | Delta | Cause |
|-------------|-------|--------|-------|-------|
| A-load | 21 | 15 | -6 | 8→4 regs (compressed A, K/2 stride) |
| Metadata load | 0 | 4 | +4 | csrr + addr + 1 flw (1 meta_load only) |
| B-load | 155 | 126 | -29 | Stride chaining vs independent mul per column |
| MMA + loop control | 4 | 6 | +2 | 4 pointer increments vs 2 |
| **Total** | **180** | **151** | **-29** | |

---

### 5. Epilogue — Store D + Cycle Measurement + Restore

**Dense** (48 insns, lines 273-320):
```asm
; --- Store D (23 insns) ---
slli  a6, a6, 0x2           ; tile_col * sizeof(int32)
add   a4, a4, a6            ; C_addr + tile_col * 4
csrr  a6, tid
mul   a5, a5, a2            ; tile_row * N
slli  a5, a5, 0x2           ; * sizeof(int32)
add   a4, a4, a5            ; base + tile_row offset
srli  a5, a6, 0x2           ; lane_row = tid >> 2
andi  a6, a6, 0x3           ; lane_col = tid & 3
mul   a5, a5, a2            ; lane_row * N
slli  a5, a5, 0x2
slli  a6, a6, 0x2
add   a5, a5, a6
add   a4, a4, a5            ; final output pointer
fsw   ft0-ft3, 0x0/0x10/0x20/0x30(a4)    ; 4 stores (m_step=0)
slli  a2, a2, 0x5           ; N * 32 = tcM * N * 4 (stride to m_step=1)
add   a2, a4, a2
fsw   ft4-ft7, 0x0/0x10/0x20/0x30(a2)    ; 4 stores (m_step=1)

; --- Cycle measurement (12 insns) ---
csrr  a2, mcycle            ; end_cycles
lw    a4, 0x4(a1)           ; blockIdx.y
lw    a5, 0x0(a0)           ; blockIdx.x
lw    a0, 0x38(a0)          ; cycles_addr
lw    a1, 0x0(a1)           ; grid_dim[0]
mul   a4, a5, a4            ; block_id = x * y
sub   a2, a2, a3            ; end - start cycles
slli  a4, a4, 0x2           ; * 4
slli  a1, a1, 0x2
add   a0, a0, a1
add   a0, a0, a4
sw    a2, 0x0(a0)           ; pCycles[block_id] = delta

; --- Restore (13 insns) ---
lw    s0-s6                  ; 7 integer restores
flw   fs8-fs11               ; 4 float restores
addi  sp, sp, 0x30
ret
```

**Sparse** (45 insns, lines 249-293):
```asm
; --- Store D (22 insns) ---
slli  a7, a7, 0x2           ; tile_col * sizeof(int32)
add   a4, a4, a7
csrr  a7, tid
mul   a6, a6, a3            ; tile_row * N
slli  a6, a6, 0x2
add   a4, a4, a6
srli  a6, a7, 0x2           ; lane_row
andi  a7, a7, 0x3           ; lane_col
mul   a3, a6, a3
slli  a3, a3, 0x2
slli  a7, a7, 0x2
add   a3, a3, a7
add   a3, a4, a3
fsw   ft0-ft3, 0x0/0x10/0x20/0x30(a3)    ; 4 stores
add   a3, a3, a5            ; a5 = N*32 (pre-computed in setup)
fsw   ft4-ft7, 0x0/0x10/0x20/0x30(a3)    ; 4 stores

; --- Cycle measurement (12 insns) ---
csrr  a3, mcycle
lw    a4, 0x4(a1)
lw    a5, 0x0(a0)
lw    a0, 0x40(a0)          ; cycles_addr at offset 0x40 (sparse has extra meta_addr field)
lw    a1, 0x0(a1)
mul   a4, a5, a4
sub   a3, a3, a2            ; end - start
slli  a4, a4, 0x2
slli  a1, a1, 0x2
add   a0, a0, a1
add   a0, a0, a4
sw    a3, 0x0(a0)

; --- Restore (11 insns) ---
lw    s0-s4                  ; 5 integer restores
flw   fs8-fs11               ; 4 float restores
addi  sp, sp, 0x30
ret
```

**Diff (-3 insns)**:
- Store D: 22 vs 23 (-1). Sparse uses pre-computed `a5 = N*32` via `add a3, a3, a5` instead of inline `slli a2, a2, 0x5; add a2, a4, a2`.
- Cycle measurement: identical (12 vs 12). Both kernels measure cycles in this version (unlike int4 where sparse omitted cycle measurement).
- Restore: 11 vs 13 (-2). Sparse restores s0-s4 (5 regs), dense restores s0-s6 (7 regs).

---

## Full Loop Body: Side-by-Side

### Dense Loop (180 instructions per K-tile)

```
Line  Addr       Instruction              Section
───── ────────── ──────────────────────── ──────────────
  93  80000140   csrr  t4, tid            A-load: tid
  94  80000144   srli  t5, t4, 0x2        A-load: row = tid>>2
  95  80000148   slli  t4, t4, 0x2        A-load: col offset
  96  8000014c   andi  t4, t4, 0xc        A-load: col = (tid&3)*4
  97  80000150   mul   t6, t5, a7         A-load: row * K
  98  80000154   add   t6, t6, t4         A-load: + col
  99  80000158   add   s0, t1, t0         A-load: A_base + loop_i
 100  8000015c   add   t6, s0, t6         A-load: ptr0
 101  80000160   flw   fa0, -0x30(t6)     A[0]
 102  80000164   flw   fa1, -0x20(t6)     A[1]
 103  80000168   flw   fa2, -0x10(t6)     A[2]
 104  8000016c   flw   fa3, 0x0(t6)       A[3]
 105  80000170   addi  t5, t5, 0x8        A-load: row += tcM
 106  80000174   mul   t5, a7, t5         A-load: (row+8)*K
 107  80000178   add   t4, t0, t4         A-load: loop_i + col
 108  8000017c   add   t4, t4, t5         A-load: ptr1
 109  80000180   add   t4, t1, t4         A-load: + A_base
 110  80000184   flw   fa4, -0x30(t4)     A[4]
 111  80000188   flw   fa5, -0x20(t4)     A[5]
 112  8000018c   flw   fa6, -0x10(t4)     A[6]
 113  80000190   flw   fa7, 0x0(t4)       A[7]
───── ────────── ──────────────────────── ──────────────
                                           B-load: tid decomp
 114  80000194   csrr  t5, tid            B-load: tid
 115  80000198   srli  t4, t5, 0x2        B-load: row = tid>>2
 116  8000019c   slli  t5, t5, 0x2        B-load: col offset
 117  800001a0   andi  t5, t5, 0xc        B-load: col = (tid&3)*4
                                           B-load: col group 0 (fs8)
 118  800001a4   mul   t6, t5, a2         col * N
 119  800001a8   add   s0, t2, t4         B_base + loop_offset
 120  800001ac   add   t6, s0, t6         ptr_col0
 121  800001b0   lbu   s0, -0x8(t6)       byte col0
 122  800001b4   addi  s1, t5, 0x1        col+1
 123  800001b8   mul   s1, a2, s1         (col+1)*N
 124  800001bc   add   s2, t2, t4
 125  800001c0   add   s1, s2, s1         ptr_col1
 126  800001c4   lbu   s2, -0x8(s1)       byte col1
 127  800001c8   addi  s3, t5, 0x2        col+2
 128  800001cc   mul   s3, a2, s3
 129  800001d0   add   s4, t2, t4
 130  800001d4   add   s3, s4, s3         ptr_col2
 131  800001d8   lbu   s4, -0x8(s3)       byte col2
 132  800001dc   addi  s5, t5, 0x3        col+3
 133  800001e0   mul   s5, a2, s5
 134  800001e4   add   s6, t2, t4
 135  800001e8   add   s5, s6, s5         ptr_col3
 136  800001ec   lbu   s6, -0x8(s5)       byte col3
 137  800001f0   slli  s2, s2, 0x8        pack
 138  800001f4   or    s0, s2, s0
 139  800001f8   slli  s4, s4, 0x10
 140  800001fc   slli  s6, s6, 0x18
 141  80000200   or    s2, s4, s6
 142  80000204   or    s0, s0, s2
 143  80000208   fmv.w.x fs8, s0          → B[0]
                                           B-load: reuse ptrs (fs9)
 144  8000020c   lbu   s0, 0x0(s1)        col1 at k+1
 145  80000210   lbu   t6, 0x0(t6)        col0 at k+1
 146  80000214   lbu   s1, 0x0(s3)        col2 at k+1
 147  80000218   lbu   s2, 0x0(s5)        col3 at k+1
 148  8000021c   slli  s0, s0, 0x8
 149  80000220   or    t6, s0, t6
 150  80000224   slli  s1, s1, 0x10
 151  80000228   slli  s2, s2, 0x18
 152  8000022c   or    s0, s1, s2
 153  80000230   or    t6, t6, s0
 154  80000234   fmv.w.x fs9, t6          → B[1]
                                           ... (col groups 1-3: same pattern ×3)
 155-192                                   col group 1 → fs10, fs11 (38 insns)
 193-230                                   col group 2 → ft8, ft9 (38 insns)
 231-268                                   col group 3 → ft10, ft11 (38 insns)
───── ────────── ──────────────────────── ──────────────
 269  80000400   .insn (MMA dense)        mma_sync (rs2=0)
 270  80000404   addi  t0, t0, 0x40       i += tileK
 271  80000408   add   t2, t2, t3         B_base += N*tileK
 272  8000040c   bltu  t0, a7, loop       branch if i < K
```

### Sparse Loop (151 instructions per K-tile)

```
Line  Addr       Instruction              Section
───── ────────── ──────────────────────── ──────────────
  98  80000154   csrr  s0, tid            A-load: tid
  99  80000158   srli  s1, s0, 0x2        A-load: row = tid>>2
 100  8000015c   slli  s0, s0, 0x2        A-load: col offset
 101  80000160   andi  s0, s0, 0xc        A-load: col = (tid&3)*4
 102  80000164   mul   s2, s1, t2         A-load: row * (K/2)
 103  80000168   add   s2, s2, s0         A-load: + col
 104  8000016c   add   s2, t6, s2         A-load: + A_base
 105  80000170   flw   fa0, -0x10(s2)     A_compressed[0]
 106  80000174   flw   fa1, 0x0(s2)       A_compressed[1]
 107  80000178   addi  s1, s1, 0x8        A-load: row += tcM
 108  8000017c   mul   s1, t2, s1         A-load: (row+8) * (K/2)
 109  80000180   add   s0, t6, s0         A-load: A_base + col
 110  80000184   add   s0, s0, s1         A-load: ptr1
 111  80000188   flw   fa2, -0x10(s0)     A_compressed[2]
 112  8000018c   flw   fa3, 0x0(s0)       A_compressed[3]
───── ────────── ──────────────────────── ──────────────
 113  80000190   csrr  s0, tid            Meta-load: tid
 114  80000194   slli  s0, s0, 0x2        Meta-load: * 4
 115  80000198   add   s0, t5, s0         Meta-load: meta_base + tid*4
 116  8000019c   flw   fa4, 0x0(s0)       metadata word 0
───── ────────── ──────────────────────── ──────────────
 117  800001a0   csrr  s0, tid            B-load: tid
 118  800001a4   srli  s1, s0, 0x3        B-load: row = tid>>3 (b_block=32)
 119  800001a8   slli  s0, s0, 0x2        B-load: col offset
 120  800001ac   andi  s0, s0, 0x1c       B-load: col = (tid&7)*4
                                           B-load: base + stride chain
 121  800001b0   mul   s0, s0, a3         col_offset = col * N
 122  800001b4   add   s1, t4, s1         B_base + row
 123  800001b8   add   s1, s1, s0         ptr_col0
 124  800001bc   add   s0, s1, a3         ptr_col1 = ptr_col0 + N
 125  800001c0   lbu   s2, 0x0(s0)        byte col1
 126  800001c4   lbu   s3, 0x0(s1)        byte col0
 127  800001c8   slli  s2, s2, 0x8        pack
 128  800001cc   add   s0, s0, a3         ptr_col2 = ptr_col1 + N
 129  800001d0   lbu   s4, 0x0(s0)        byte col2
 130  800001d4   add   s0, s0, a3         ptr_col3 = ptr_col2 + N
 131  800001d8   lbu   s0, 0x0(s0)        byte col3
 132  800001dc   or    s2, s2, s3
 133  800001e0   slli  s4, s4, 0x10
 134  800001e4   or    s2, s2, s4
 135  800001e8   slli  s0, s0, 0x18
 136  800001ec   addi  s3, s1, 0x4        prep next k-group
 137  800001f0   add   s3, s3, a3         chain to col1
 138  800001f4   lbu   s4, 0x0(s3)        pre-load
 139  800001f8   or    s0, s2, s0
 140  800001fc   fmv.w.x fs8, s0          → B[0]
                                           B-load: chain continues (fs9)
 141  80000200   lbu   s0, 0x4(s1)        col0 at k+4
 142  80000204   slli  s4, s4, 0x8
 143  80000208   add   s3, s3, a3         chain col2
 144  8000020c   lbu   s2, 0x0(s3)
 145  80000210   add   s3, s3, a3         chain col3
 146  80000214   lbu   s3, 0x0(s3)
 147  80000218   or    s0, s4, s0
 148  8000021c   slli  s2, s2, 0x10
 149  80000220   or    s0, s0, s2
 150  80000224   slli  s3, s3, 0x18
 151  80000228   addi  s2, s1, 0x8        prep next
 152  8000022c   add   s2, s2, a3
 153  80000230   lbu   s4, 0x0(s2)
 154  80000234   or    s0, s0, s3
 155  80000238   fmv.w.x fs9, s0          → B[1]
                                           ... (remaining 6 registers: same chain pattern)
 156-171                                   fs10 (16 insns)
 172-185                                   fs11 (14 insns)
 186-200                                   ft8 (15 insns)
 201-215                                   ft9 (15 insns)
 216-230                                   ft10 (15 insns)
 231-242                                   ft11 (12 insns)
───── ────────── ──────────────────────── ──────────────
 243  80000398   .insn (MMA sparse)       mma_sync (rs2=1, fused meta+compute)
 244  8000039c   addi  t5, t5, 0x80       meta_ptr += 128
 245  800003a0   add   t4, t4, t3         B_base += N*64
 246  800003a4   addi  t1, t1, 0x40       counter += 64
 247  800003a8   addi  t6, t6, 0x20       A_ptr += 32 (tileK/2)
 248  800003ac   blt   t1, t0, loop       branch if counter < K
```

---

## Key Structural Differences

### 1. B-Load: Independent mul vs Stride Chaining (-29 insns)

This is by far the largest source of instruction count difference. The root cause is the different `b_block_size`:

| | Dense | Sparse |
|--|-------|--------|
| b_block_size | 16 (tcK×tcN = 4×4) | 32 (tcK×tcN×2 = NT) |
| b_sub_blocks | 2 | 1 |
| tid row shift | `tid >> 2` (4 threads per row) | `tid >> 3` (8 threads per row) |
| tid col mask | `(tid & 3) * 4` = andi 0xc | `(tid & 7) * 4` = andi 0x1c |
| Columns per register | 4 (stride N apart) | 4 (stride N apart) |
| Column addressing | `addi col_idx; mul col_idx, N` (2 insns per col) | `add ptr, ptr, N` (1 insn per col) |
| Column groups | 4 (independently computed) | 2 n_groups with continuous chaining |

The dense pattern needs `addi + mul` (2 insns) per column to compute `(col+offset) * N`. With 4 columns per register and 4 column groups, that's ~32 multiplications. The sparse pattern computes one base via `mul`, then chains `add ptr, ptr, N` for all subsequent columns — only 1 `mul` total plus ~7 `add` instructions for stride chaining.

Additionally, the dense code uses more callee-saved registers (s5, s6) for temporary column pointers, adding 2 insns each to prologue and epilogue.

### 2. A-Load: 8 regs → 4 regs (-6 insns)

Dense loads all `k_steps=4` worth of A data into 8 registers (fa0-fa7).
Sparse loads only `k_steps/2=2` into 4 registers (fa0-fa3) because compressed A has half the elements. Per m_step: 4 `flw` → 2 `flw`, with correspondingly fewer address computations.

### 3. Metadata Load: 0 → 4 insns

Int8 NT=32 needs only 1 metadata `flw` (vs int4 which needs 2). The flat load covers all 8 metadata columns in a single load because `cols_per_load = NT/PD = 32/4 = 8 = meta_cols`. This makes the metadata overhead quite small.

### 4. Loop Control: 2 → 5 pointer advances (+2 insns)

Dense advances 1 counter (`t0 += 0x40`) + 1 pointer (`t2 += N*64`) + branch = 3 insns.
Sparse advances 4 values (meta, B, counter, A) + branch = 6 insns.

### 5. Fused Sparse MMA: No Separate meta_store

Unlike earlier designs, the current sparse kernel uses a fused MMA instruction (`rs2=1` flag on `INST_TCU_WMMA_SP`). The metadata loaded into `fa4` is consumed by the TCU's 2-phase micro-op sequence — no separate `.insn` for `meta_store`. This saves 1 instruction vs the legacy approach.

### 6. Cycle Measurement: Both Present

Both int8 NT=32 kernels have cycle measurement (unlike int4 where sparse omitted it). This means the cycle measurement overhead (1 `csrr` in setup + 12 insns in epilogue = 13 insns total) is equal in both kernels.

---

## Dynamic Instruction Count Comparison

For a given problem size M×N×K with NT=32, tileM=tileN=16, tileK=64 (int8 elements):

- Number of K-tile iterations: `T = K / 64`
- Number of tile blocks: `(M/16) × (N/16)`

**Per block (single kernel_body call)**:

| Component | Dense | Sparse | Formula |
|-----------|-------|--------|---------|
| Prologue | 13 | 11 | constant |
| Setup | 22 | 24 | constant |
| Pre-loop | 8 | 13 | constant |
| Loop body | 180 × T | 151 × T | T = K / 64 |
| Epilogue | 48 | 45 | constant |
| **Total** | **91 + 180T** | **93 + 151T** | |

**Break-even**: `91 + 180T = 93 + 151T` → `29T = 2` → `T ≈ 0.07`.

Since T ≥ 1 always, **sparse always has fewer dynamic instructions** for int8 NT=32.

| K (int8 elements) | T | Dense | Sparse | Savings | % fewer |
|----|---|-------|--------|---------|---------|
| 64 | 1 | 271 | 244 | 27 | 10.0% |
| 128 | 2 | 451 | 395 | 56 | 12.4% |
| 256 | 4 | 811 | 697 | 114 | 14.1% |
| 512 | 8 | 1531 | 1301 | 230 | 15.0% |
| 1024 | 16 | 2971 | 2509 | 462 | 15.6% |
| 2048 | 32 | 5851 | 4925 | 926 | 15.8% |
| 4096 | 64 | 11611 | 9757 | 1854 | 16.0% |

At large K, sparse approaches 16.1% fewer instructions (= 29/180 = loop body savings ratio).

## Comparison with Int4 NT=8

| Metric | Int4 NT=8 | Int8 NT=32 |
|--------|-----------|------------|
| B-load method | `flw` (aligned, packed at host) | `lbu` + pack_row (byte-level packing) |
| Loop body delta | -1 insn/iter | **-29 insns/iter** |
| Break-even (no cycles) | T=9 (K=288) | T < 1 (always sparse wins) |
| Dominant savings | A-load (-6) | B-load (-29) |
| B-load savings cause | Stride chaining vs 2nd mul group | Stride chaining vs 4× addi+mul per col |

Int8 benefits far more from sparsity's B-load optimization because:
1. **Int8 pack_row is expensive**: Each register needs 4 `lbu` + `slli`/`or`/`fmv.w.x` = 11 insns for packing, PLUS per-column address computation. The column addressing dominates.
2. **Stride chaining eliminates multiplications**: Dense needs `addi + mul` per column (because each column is at `col * N` offset). Sparse chains `add ptr, ptr, N`. With 4 columns × 8 registers, this saves ~24 multiplications.
3. **Int4 uses `flw`**: Int4 B is pre-packed by the host, so B-load is just `flw` (1 insn per register). No per-column addressing overhead to optimize away.

## Implication for TCU_CYCLES

### HW Micro-Op Counts

Sparse MMA **reduces** the number of hardware micro-ops compared to dense:

| | Dense | Sparse | Notes |
|--|-------|--------|-------|
| k_steps in MMA phase | 4 | 2 | Halved by 2:4 sparsity compression |
| MMA uops | 32 | 16 | m_steps × n_steps × k_steps = 2×4×{4,2} |
| Metadata uops | 0 | 8 | meta_cols = (32×2×4)/32 = 8 |
| **Total uops per mma_sync** | **32** | **24** | Sparse = TCU_UOPS/2 + meta_cols |
| **Uops per full K** | **32 × (K/64)** | **24 × (K/64)** | Same loop iterations, fewer uops each |

Sparse has **25% fewer HW uops** per mma_sync (24 vs 32). Over the full K dimension, this is a 25% reduction in MMA pipeline work.

### Why Is Sparse Still Slower?

Despite 16% fewer dynamic instructions AND 25% fewer HW uops, the measured TCU_CYCLES show sparse (10781) is 55% slower than dense (6955). The performance gap must come from **memory system effects**, not compute:

1. **B-load data is the same size**: Both dense and sparse load NRB=8 registers per thread (8 × 32 × 4 = 1024 bytes per warp). The `b_block_size_sp=32` vs `b_block_size=16` only changes the thread-to-element mapping, not the total data volume. B is not doubled.

2. **Metadata load traffic**: Each K-tile iteration adds 1 `flw` per thread for metadata (32 threads × 4 bytes = 128 bytes). This is pure overhead with no dense equivalent.

3. **Metadata SRAM write latency**: The 8 metadata uops at the start of each sparse MMA must complete before the 16 compute uops can begin. This serialization adds pipeline bubbles that don't exist in the dense path.

4. **Different B access pattern**: Sparse B-load uses `tid>>3` (4 rows × 8 cols per block) vs dense `tid>>2` (8 rows × 4 cols). This changes cache line access locality — needs further investigation whether this worsens or improves cache behavior.

The software instruction count and HW uop count are both **already favorable** for sparse int8 — the bottleneck requires deeper investigation, likely involving memory access pattern profiling and pipeline stall analysis.

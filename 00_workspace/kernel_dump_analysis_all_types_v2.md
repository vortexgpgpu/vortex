# Kernel Dump Analysis: Dense vs Sparse TCU, All Types (NT=8)

**Config**: NT=8, col-major B layout, cycle measurement in both dense and sparse kernels
**Date**: 2026-03-16
**Branch**: `260313_after_int4_kernel_opt`

## 1. Configuration Parameters

### 1a. Shared parameters (NT=8, type-independent)

From `tensor_cfg.h` (`wmma_config_t<8>`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| xtileM, xtileN | 8, 8 | `1 << tile_em`, `1 << tile_en` |
| xtileK | 8 | `tile_cap / max(xtileM, xtileN) = 64/8` |
| tcM, tcN, tcK | 4, 2, 2 | `1 << block_em`, `1 << block_en`, `block_cap/max(tcM,tcN)` |
| m_steps, n_steps, k_steps | 2, 4, 4 | `xtile / tc` per dimension |
| a_block_size | 8 (= NT) | `tcM * tcK = 4*2` |
| NRA, NRB, NRC | 8, 8, 8 | `xtileX * xtileY / NT` |
| sparse_regs | 4 | `m_steps * (k_steps/2) = 2*2` |

### 1b. Type-specific parameters

| Parameter | FP16/FP32 | INT8/INT32 | INT4/INT32 |
|-----------|-----------|------------|------------|
| It::dtype | uint16_t | int8_t | uint8_t |
| It::bits | 16 | 8 | 4 |
| It::id | 1 | 9 | 11 |
| Ot::id | 0 (fp32) | 8 (int32) | 8 (int32) |
| i_ratio (XB/sizeof) | 2 | 4 | 4 |
| tileK | 16 | 32 | 32 |
| tileK (actual elements) | 16 | 32 | 32 (†) |
| rtl_i_ratio (32/bits) | 2 | 4 | 8 |

(†) INT4 tileK = xtileK * i_ratio = 8*4 = 32 register-elements. Each register-element holds 32/4=8 int4 values, so 32*8=256 int4 elements per tile — but the K-loop stride and addressing use register-element units.

### 1c. Sparse metadata parameters

| Parameter | FP16 | INT8 | INT4 |
|-----------|------|------|------|
| meta_cols | 1 | 2 | 4 |
| per_warp_depth (PD) | 4 | 4 | 4 |
| meta_cols_per_load | 2 | 2 | 2 |
| num_meta_loads | 1 | 1 | 2 |
| meta_stride (words) | 8 | 8 | 16 |
| a_k_stride_sp | 8 | 16 | 16 |

### 1d. Register allocation

| Register | Dense FP16 | Dense INT8/INT4 | Sparse FP16 | Sparse INT8/INT4 |
|----------|-----------|-----------------|-------------|------------------|
| ft0-ft7 (f0-f7) | fragC (8) | fragC (8) | fragC (8) | fragC (8) |
| fa0-fa7 (f10-f17) | fragA (8) | fragA (8) | fa0-fa3: fragA (4), fa4(-fa5): meta | fa0-fa3: fragA (4), fa4(-fa5): meta |
| fs8-fs11 (f24-f27) | fragB [0:3] | fragB [0:3] | fragB [0:3] | fragB [0:3] |
| ft8-ft11 (f28-f31) | fragB [4:7] | fragB [4:7] | fragB [4:7] | fragB [4:7] |
| Callee-saved int | s0 | — | s0-s4 | s0-s2 |
| Callee-saved float | fs8-fs11 | fs8-fs11 | fs8-fs11 | fs8-fs11 |

---

## 2. Instruction Count Summary

### 2a. Full 6-column table (sections × 6 configs)

| Section | D-FP16 | D-INT8 | D-INT4 | S-FP16 | S-INT8 | S-INT4 |
|---------|--------|--------|--------|--------|--------|--------|
| **Prologue** | 6 | 5 | 5 | 10 | 8 | 8 |
| **Setup** | 20 | 20 | 20 | 21 | 21 | 21 |
| **Pre-loop** | 13 | 10 | 10 | 20 | 20 | 18 |
| **Loop: A-load** | 22 | 21 | 21 | 18 | 15 | 15 |
| **Loop: Meta-load** | — | — | — | 4 | 5 | 5 |
| **Loop: B-load** | 22 | 21 | 21 | 18 | 19 | 19 |
| **Loop: MMA+ctrl** | 4 | 3 | 3 | 5 | 4 | 5 |
| **Epilogue** | 42 | 41 | 41 | 46 | 44 | 44 |
| **Loop body** | **48** | **45** | **45** | **45** | **43** | **44** |
| **Total** | **129** | **121** | **121** | **142** | **136** | **135** |

Cross-reference: matches `kernel_dump_comparison_all_types.md` NT=8 table exactly.

### 2b. Dense vs sparse delta (NT=8)

| Section | FP16 (S−D) | INT8 (S−D) | INT4 (S−D) |
|---------|------------|------------|------------|
| Prologue | +4 | +3 | +3 |
| Setup | +1 | +1 | +1 |
| Pre-loop | +7 | +10 | +8 |
| A-load | −4 | −6 | −6 |
| Meta-load | +4 | +5 | +5 |
| B-load | −4 | −2 | −2 |
| MMA+ctrl | +1 | +1 | +2 |
| Epilogue | +4 | +3 | +3 |
| **Loop body** | **−3** | **−2** | **−1** |
| **Total** | **+13** | **+15** | **+14** |

---

## 3. FP16/FP32 Analysis

### 3a. Section-by-section commentary

**Prologue (D=6, S=10)**: Dense saves s0 + fs8-fs11 (1+4 = 5 regs, 6 insns with `addi sp`). Sparse saves s0-s4 + fs8-fs11 (5+4 = 9 regs, 10 insns). Extra s1-s4 are needed for the B-load stride chaining and meta pointer.

**Setup (D=20, S=21)**: Both load kernel_arg_t fields (A/B/C addr, N, K), resolve blockIdx via GOT+tp, zero fragC (8 × fmv.s). Sparse loads `meta_addr` (+1 `lw`), accounting for the difference.

**Pre-loop (D=13, S=20)**: Dense uses 2 loop counters (t0=byte offset, t1=element count) because FP16 byte stride (0x20) ≠ element stride (0x10). Sparse must additionally compute compressed A base (`mul a5*t1`, where t1=K/2), metadata base pointer (`lw blockIdx.y`, `mul * num_k_tiles * per_k_tile_words`), and pre-compute K*2 stride for B-load chaining.

**A-load (D=22, S=18)**: Dense loads 8 regs (NRA=8, full tileK=16 elements). Sparse loads 4 regs (sparse_regs=4, compressed K/2=8 elements). The 4 fewer flw plus saved address computations yield −4 instructions.

**Meta-load (D=0, S=4)**: Sparse loads 1 metadata word (fa4) via `csrr tid + slli + add + flw`. FP16 has num_meta_loads=1 (meta_cols=1 ≤ cols_per_load=2), so only 1 flw.

**B-load (D=22, S=18)**: Dense uses 2 independent mul-based row groups (tid>>1 decomposition, b_block_size=4). Sparse uses stride chaining (tid>>2 decomposition, b_block_size_sp=8=NT). The 3 chain `add` instructions replace the second `addi+mul+add+add+add` sequence, saving 4 instructions. FP16 sparse saves more than INT types because the stride register (`s0 = K*2`) generates more efficient chain addressing.

**MMA+ctrl (D=4, S=5)**: Dense advances 2 counters (`addi t1 +0x10`, `addi t0 +0x20`). Sparse advances 3 pointers (`addi t4` meta, `addi t0` B-offset, `addi t2` B-base). The extra pointer costs +1.

**Epilogue (D=42, S=46)**: Both have identical store_D (23 insns) and cycle measurement (12 insns). Sparse restore is longer: 5 `lw` (s0-s4) + 4 `flw` + `addi sp` + `ret` = 11, vs dense 1 `lw` (s0) + 4 `flw` + `addi sp` + `ret` = 7. Delta = +4.

### 3b. Full annotated loop body — Dense FP16 (48 instructions)

```
Line  Addr       Hex              Instruction              Section
───── ────────── ──────────────── ──────────────────────── ──────────────
  1   80000130   73 2f 00 cc     csrr  t5, tid            A-load: tid
  2   80000134   93 5f 1f 00     srli  t6, t5, 0x1        A-load: row = tid>>1
  3   80000138   b3 8f fe 03     mul   t6, t4, t6         A-load: row * (K*2)
  4   8000013c   13 14 2f 00     slli  s0, t5, 0x2        A-load: tid<<2
  5   80000140   13 74 44 00     andi  s0, s0, 0x4        A-load: col = (tid&1)*4
  6   80000144   b3 8f 8f 00     add   t6, t6, s0         A-load: row*(K*2) + col
  7   80000148   b3 8f f2 01     add   t6, t0, t6         A-load: + byte_offset
  8   8000014c   b3 8f f3 01     add   t6, t2, t6         A-load: + A_base → ptr0
  9   80000150   07 a5 0f 00     flw   fa0, 0x0(t6)       A[0]
 10   80000154   87 a5 8f 00     flw   fa1, 0x8(t6)       A[1]
 11   80000158   07 a6 0f 01     flw   fa2, 0x10(t6)      A[2]
 12   8000015c   87 a6 8f 01     flw   fa3, 0x18(t6)      A[3]
 13   80000160   13 7f ef ff     andi  t5, t5, -0x2       A-load: tid & ~1
 14   80000164   13 0f 8f 00     addi  t5, t5, 0x8        A-load: (tid|~1)+8 → row+tcM
 15   80000168   33 8f e8 03     mul   t5, a7, t5         A-load: K * (row+tcM)  [uses K not K*2]
 16   8000016c   33 84 82 00     add   s0, t0, s0         A-load: byte_offset + col
 17   80000170   33 0f e4 01     add   t5, s0, t5         A-load: + K*(row+tcM)
 18   80000174   33 8f e3 01     add   t5, t2, t5         A-load: + A_base → ptr1
 19   80000178   07 27 0f 00     flw   fa4, 0x0(t5)       A[4]
 20   8000017c   87 27 8f 00     flw   fa5, 0x8(t5)       A[5]
 21   80000180   07 28 0f 01     flw   fa6, 0x10(t5)      A[6]
 22   80000184   87 28 8f 01     flw   fa7, 0x18(t5)      A[7]
───── ────────── ──────────────── ──────────────────────── ──────────────
 23   80000188   73 2f 00 cc     csrr  t5, tid            B-load: tid
 24   8000018c   93 5f 1f 00     srli  t6, t5, 0x1        B-load: row = tid>>1
 25   80000190   b3 8f fe 03     mul   t6, t4, t6         B-load: row * (K*2)
 26   80000194   13 14 2f 00     slli  s0, t5, 0x2        B-load: tid<<2
 27   80000198   13 74 44 00     andi  s0, s0, 0x4        B-load: col = (tid&1)*4
 28   8000019c   b3 8f 8f 00     add   t6, t6, s0         B-load: + col
 29   800001a0   b3 8f f2 01     add   t6, t0, t6         B-load: + byte_offset
 30   800001a4   b3 0f fe 01     add   t6, t3, t6         B-load: + B_base → ptr0
 31   800001a8   07 ac 0f 00     flw   fs8, 0x0(t6)       B[0]
 32   800001ac   13 7f ef ff     andi  t5, t5, -0x2       B-load: tid & ~1
 33   800001b0   13 0f 8f 00     addi  t5, t5, 0x8        B-load: row+tcM
 34   800001b4   33 8f e8 03     mul   t5, a7, t5         B-load: K * (row+tcM)
 35   800001b8   33 84 82 00     add   s0, t0, s0         B-load: byte_offset + col
 36   800001bc   33 0f e4 01     add   t5, s0, t5         B-load: ptr1
 37   800001c0   33 0f ee 01     add   t5, t3, t5         B-load: + B_base
 38   800001c4   87 2c 0f 00     flw   fs9, 0x0(t5)       B[1]
 39   800001c8   07 ad 8f 00     flw   fs10, 0x8(t6)      B[2] (via offset from ptr0)
 40   800001cc   87 2d 8f 00     flw   fs11, 0x8(t5)      B[3]
 41   800001d0   07 ae 0f 01     flw   ft8, 0x10(t6)      B[4]
 42   800001d4   87 2e 0f 01     flw   ft9, 0x10(t5)      B[5]
 43   800001d8   07 af 8f 01     flw   ft10, 0x18(t6)     B[6]
 44   800001dc   87 2f 8f 01     flw   ft11, 0x18(t5)     B[7]
───── ────────── ──────────────── ──────────────────────── ──────────────
 45   800001e0   0b 80 00 04     .insn r (MMA)            mma_sync [rd=x0(fp32), rs1=x1(fp16), rs2=x0(dense)]
 46   800001e4   13 03 03 01     addi  t1, t1, 0x10       elem_count += 0x10
 47   800001e8   93 82 02 02     addi  t0, t0, 0x20       byte_offset += 0x20
 48   800001ec   e3 62 13 f5     bltu  t1, a7, loop       if elem_count < K
```

### 3c. Full annotated loop body — Sparse FP16 (45 instructions)

```
Line  Addr       Hex              Instruction              Section
───── ────────── ──────────────── ──────────────────────── ──────────────
  1   80000160   f3 24 00 cc     csrr  s1, tid            A-load: tid
  2   80000164   13 d9 14 00     srli  s2, s1, 0x1        A-load: row = tid>>1
  3   80000168   33 89 2f 03     mul   s2, t6, s2         A-load: row * (K/2)*2  [t6=slli(K/2,1)]
  4   8000016c   93 99 24 00     slli  s3, s1, 0x2        A-load: tid<<2
  5   80000170   93 f9 49 00     andi  s3, s3, 0x4        A-load: col = (tid&1)*4
  6   80000174   33 09 39 01     add   s2, s2, s3         A-load: row*(K/2)*2 + col
  7   80000178   33 89 22 01     add   s2, t0, s2         A-load: + byte_offset
  8   8000017c   33 09 2e 01     add   s2, t3, s2         A-load: + A_base → ptr0
  9   80000180   07 25 09 00     flw   fa0, 0x0(s2)       A_compressed[0]
 10   80000184   87 25 89 00     flw   fa1, 0x8(s2)       A_compressed[1]
 11   80000188   93 f4 e4 ff     andi  s1, s1, -0x2       A-load: tid & ~1
 12   8000018c   93 84 84 00     addi  s1, s1, 0x8        A-load: row + tcM
 13   80000190   b3 04 93 02     mul   s1, t1, s1         A-load: (K/2) * (row+tcM)
 14   80000194   b3 89 32 01     add   s3, t0, s3         A-load: byte_offset + col
 15   80000198   b3 84 99 00     add   s1, s3, s1         A-load: ptr1
 16   8000019c   b3 04 9e 00     add   s1, t3, s1         A-load: + A_base
 17   800001a0   07 a6 04 00     flw   fa2, 0x0(s1)       A_compressed[2]
 18   800001a4   87 a6 84 00     flw   fa3, 0x8(s1)       A_compressed[3]
───── ────────── ──────────────── ──────────────────────── ──────────────
 19   800001a8   f3 24 00 cc     csrr  s1, tid            Meta-load: tid
 20   800001ac   93 94 24 00     slli  s1, s1, 0x2        Meta-load: tid * 4
 21   800001b0   b3 84 9e 00     add   s1, t4, s1         Meta-load: meta_ptr + tid*4
 22   800001b4   07 a7 04 00     flw   fa4, 0x0(s1)       metadata word 0
───── ────────── ──────────────── ──────────────────────── ──────────────
 23   800001b8   f3 24 00 cc     csrr  s1, tid            B-load: tid
 24   800001bc   13 d9 24 00     srli  s2, s1, 0x2        B-load: row = tid>>2
 25   800001c0   33 09 2f 03     mul   s2, t5, s2         B-load: row * (K*2)  [t5=slli(K,1)]
 26   800001c4   93 f4 34 00     andi  s1, s1, 0x3        B-load: tid & 3
 27   800001c8   93 94 24 00     slli  s1, s1, 0x2        B-load: col = (tid&3)*4
 28   800001cc   b3 04 99 00     add   s1, s2, s1         B-load: row*(K*2) + col
 29   800001d0   b3 84 93 00     add   s1, t2, s1         B-load: + B_base [t2 includes byte_offset]
 30   800001d4   33 89 84 00     add   s2, s1, s0         B-load: ptr1 = ptr0 + K*2  [s0=K*4 stride]
 31   800001d8   b3 09 89 00     add   s3, s2, s0         B-load: ptr2 = ptr1 + K*2
 32   800001dc   33 8a 89 00     add   s4, s3, s0         B-load: ptr3 = ptr2 + K*2
 33   800001e0   07 ac 04 00     flw   fs8, 0x0(s1)       B[0] (row0, n_group0)
 34   800001e4   87 2c 09 00     flw   fs9, 0x0(s2)       B[1] (row1, n_group0)
 35   800001e8   07 ad 09 00     flw   fs10, 0x0(s3)      B[2] (row2, n_group0)
 36   800001ec   87 2d 0a 00     flw   fs11, 0x0(s4)      B[3] (row3, n_group0)
 37   800001f0   07 ae 04 01     flw   ft8, 0x10(s1)      B[4] (row0, n_group1)
 38   800001f4   87 2e 09 01     flw   ft9, 0x10(s2)      B[5] (row1, n_group1)
 39   800001f8   07 af 09 01     flw   ft10, 0x10(s3)     B[6] (row2, n_group1)
 40   800001fc   87 2f 0a 01     flw   ft11, 0x10(s4)     B[7] (row3, n_group1)
───── ────────── ──────────────── ──────────────────────── ──────────────
 41   80000200   0b 80 10 04     .insn r (MMA)            mma_sync [rd=x0(fp32), rs1=x1(fp16), rs2=x1(sparse)]
 42   80000204   93 8e 0e 02     addi  t4, t4, 0x20       meta_ptr += 0x20 (meta_stride*4 = 8*4)
 43   80000208   93 82 02 01     addi  t0, t0, 0x10       byte_offset += 0x10 (A advance)
 44   8000020c   93 83 03 02     addi  t2, t2, 0x20       B_base += 0x20 (tileK*2 bytes)
 45   80000210   e3 c8 12 f5     blt   t0, a7, loop       if byte_offset < K
```

### 3d. Key structural differences (FP16)

1. **A-load: 8 regs → 4 regs** (22→18 = −4 insns). Sparse loads 4 registers (compressed K/2) vs 8 (full K). Each m_step group loads 2 flw instead of 4.

2. **Metadata: 0 → 4 insns**. FP16 has meta_cols=1, num_meta_loads=1, so a single `csrr+slli+add+flw` sequence.

3. **B-load: −4 insns** (22→18). Sparse B uses stride chaining (3 `add` for ptr1/ptr2/ptr3) instead of a second independent `andi+addi+mul+add+add+add` group. The tid decomposition changes from `tid>>1, mask 0x4` (2 rows × 4 k-steps) to `tid>>2, mask 0x3` (4 rows × 2 n-groups).

4. **MMA+ctrl: +1 insn** (4→5). Dense advances 2 counters (byte offset + element count). Sparse advances 3 pointers (meta, A, B).

5. **Two loop counters in dense FP16**: The FP16 byte stride per tileK is 0x20 (=16 halfwords × 2 bytes), but the element count compares against K in elements. Dense uses `t0` for byte offset and `t1` for element count. INT8/INT4 don't need this because their byte stride equals their element stride.

---

## 4. INT8/INT32 Analysis

### 4a. Section-by-section commentary

**Prologue (D=5, S=8)**: Dense saves only fs8-fs11 (no integer callee-saved regs). Sparse saves s0-s2 + fs8-fs11.

**Setup (D=20, S=21)**: Both identical structure. Sparse loads `meta_addr` (+1).

**Pre-loop (D=10, S=20)**: Dense uses a single loop counter `t0` (byte stride 0x20 = tileK=32 bytes, which equals K in element units). Only needs 1 `mul+add` each for A and B base pointers, plus `addi t2` for negative-offset addressing. Sparse has +10 more instructions for: `srli K/2`, metadata base computation (`lui+addi+and` for mask that doesn't fit 12-bit imm, `mul`, `slli`), and K*2 stride pre-computation.

**A-load (D=21, S=15)**: Dense loads 8 regs via 2 groups of `mul+add+add+add + 4×flw`. Sparse loads 4 regs via 2 groups of `mul+add+add + 2×flw`. The −6 comes from 4 fewer flw + 2 fewer address computations.

**Meta-load (D=0, S=5)**: INT8 has num_meta_loads=1 (meta_cols=2, cols_per_load=2), but needs 5 insns instead of FP16's 4: the extra `add` is because the meta pointer is shared with the B loop offset register `t0`, requiring `csrr+slli+add(t0)+add(t3)+flw`.

**B-load (D=21, S=19)**: Dense uses 2 independent mul-based row groups (same pattern as FP16 but with K not K*2). Sparse uses stride chaining. The −2 savings is smaller than FP16's −4 because INT8 dense B-load is already more efficient (single-counter addressing, no `andi -0x2` trick needed).

**MMA+ctrl (D=3, S=4)**: Dense has 1 counter advance + MMA + branch. Sparse shares the B+meta counter (both stride 0x20), so only 2 `addi` (shared counter, A pointer) + MMA + branch.

**Epilogue (D=41, S=44)**: Same store_D (23) and cycles (12). Dense restore = 4 flw + addi sp + ret = 6. Sparse restore = 3 lw (s0-s2) + 4 flw + addi sp + ret = 9. Delta = +3.

### 4b. Full annotated loop body — Dense INT8 (45 instructions)

```
Line  Addr       Hex              Instruction              Section
───── ────────── ──────────────── ──────────────────────── ──────────────
  1   80000120   73 2e 00 cc     csrr  t3, tid            A-load: tid
  2   80000124   93 5e 1e 00     srli  t4, t3, 0x1        A-load: row = tid>>1
  3   80000128   13 1e 2e 00     slli  t3, t3, 0x2        A-load: tid<<2
  4   8000012c   13 7e 4e 00     andi  t3, t3, 0x4        A-load: col = (tid&1)*4
  5   80000130   33 8f 1e 03     mul   t5, t4, a7         A-load: row * K
  6   80000134   33 0f cf 01     add   t5, t5, t3         A-load: + col
  7   80000138   b3 8f 53 00     add   t6, t2, t0         A-load: A_base + loop_i
  8   8000013c   33 8f ef 01     add   t5, t6, t5         A-load: → ptr0
  9   80000140   07 25 8f fe     flw   fa0, -0x18(t5)     A[0] (m=0, k=0)
 10   80000144   87 25 0f ff     flw   fa1, -0x10(t5)     A[1] (m=0, k=1)
 11   80000148   07 26 8f ff     flw   fa2, -0x8(t5)      A[2] (m=0, k=2)
 12   8000014c   87 26 0f 00     flw   fa3, 0x0(t5)       A[3] (m=0, k=3)
 13   80000150   93 8e 4e 00     addi  t4, t4, 0x4        A-load: row += tcM
 14   80000154   b3 8e d8 03     mul   t4, a7, t4         A-load: K * (row+4)
 15   80000158   33 8e c2 01     add   t3, t0, t3         A-load: loop_i + col
 16   8000015c   33 0e de 01     add   t3, t3, t4         A-load: + K*(row+4)
 17   80000160   33 8e c3 01     add   t3, t2, t3         A-load: + A_base → ptr1
 18   80000164   07 27 8e fe     flw   fa4, -0x18(t3)     A[4] (m=1, k=0)
 19   80000168   87 27 0e ff     flw   fa5, -0x10(t3)     A[5] (m=1, k=1)
 20   8000016c   07 28 8e ff     flw   fa6, -0x8(t3)      A[6] (m=1, k=2)
 21   80000170   87 28 0e 00     flw   fa7, 0x0(t3)       A[7] (m=1, k=3)
───── ────────── ──────────────── ──────────────────────── ──────────────
 22   80000174   73 2e 00 cc     csrr  t3, tid            B-load: tid
 23   80000178   93 5e 1e 00     srli  t4, t3, 0x1        B-load: row = tid>>1
 24   8000017c   13 1e 2e 00     slli  t3, t3, 0x2        B-load: tid<<2
 25   80000180   13 7e 4e 00     andi  t3, t3, 0x4        B-load: col = (tid&1)*4
 26   80000184   33 8f 1e 03     mul   t5, t4, a7         B-load: row * K
 27   80000188   33 0f cf 01     add   t5, t5, t3         B-load: + col
 28   8000018c   33 8f e2 01     add   t5, t0, t5         B-load: + loop_i
 29   80000190   33 0f e3 01     add   t5, t1, t5         B-load: + B_base → ptr0
 30   80000194   07 2c 0f 00     flw   fs8, 0x0(t5)       B[0]
 31   80000198   93 8e 4e 00     addi  t4, t4, 0x4        B-load: row += 4
 32   8000019c   b3 8e d8 03     mul   t4, a7, t4         B-load: K * (row+4)
 33   800001a0   33 8e c2 01     add   t3, t0, t3         B-load: loop_i + col
 34   800001a4   33 0e de 01     add   t3, t3, t4         B-load: ptr1
 35   800001a8   33 0e c3 01     add   t3, t1, t3         B-load: + B_base
 36   800001ac   87 2c 0e 00     flw   fs9, 0x0(t3)       B[1]
 37   800001b0   07 2d 8f 00     flw   fs10, 0x8(t5)      B[2] (offset from ptr0)
 38   800001b4   87 2d 8e 00     flw   fs11, 0x8(t3)      B[3]
 39   800001b8   07 2e 0f 01     flw   ft8, 0x10(t5)      B[4]
 40   800001bc   87 2e 0e 01     flw   ft9, 0x10(t3)      B[5]
 41   800001c0   07 2f 8f 01     flw   ft10, 0x18(t5)     B[6]
 42   800001c4   87 2f 8e 01     flw   ft11, 0x18(t3)     B[7]
───── ────────── ──────────────── ──────────────────────── ──────────────
 43   800001c8   93 82 02 02     addi  t0, t0, 0x20       loop_i += 0x20 (tileK bytes)
 44   800001cc   0b 84 04 04     .insn r (MMA)            mma_sync [rd=x8(int32), rs1=x9(int8), rs2=x0(dense)]
 45   800001d0   e3 e8 12 f5     bltu  t0, a7, loop       if loop_i < K
```

### 4c. Full annotated loop body — Sparse INT8 (43 instructions)

```
Line  Addr       Hex              Instruction              Section
───── ────────── ──────────────── ──────────────────────── ──────────────
  1   80000158   f3 2f 00 cc     csrr  t6, tid            A-load: tid
  2   8000015c   13 d4 1f 00     srli  s0, t6, 0x1        A-load: row = tid>>1
  3   80000160   93 9f 2f 00     slli  t6, t6, 0x2        A-load: tid<<2
  4   80000164   93 ff 4f 00     andi  t6, t6, 0x4        A-load: col = (tid&1)*4
  5   80000168   b3 04 74 02     mul   s1, s0, t2         A-load: row * (K/2)  [t2=srli(K,1)]
  6   8000016c   b3 84 f4 01     add   s1, s1, t6         A-load: + col
  7   80000170   b3 04 9f 00     add   s1, t5, s1         A-load: + A_base → ptr0  [t5=A base]
  8   80000174   07 a5 84 ff     flw   fa0, -0x8(s1)      A_compressed[0]
  9   80000178   87 a5 04 00     flw   fa1, 0x0(s1)       A_compressed[1]
 10   8000017c   13 04 44 00     addi  s0, s0, 0x4        A-load: row += tcM
 11   80000180   33 84 83 02     mul   s0, t2, s0         A-load: (K/2) * (row+4)
 12   80000184   b3 0f ff 01     add   t6, t5, t6         A-load: A_base + col
 13   80000188   b3 8f 8f 00     add   t6, t6, s0         A-load: + (K/2)*(row+4)
 14   8000018c   07 a6 8f ff     flw   fa2, -0x8(t6)      A_compressed[2]
 15   80000190   87 a6 0f 00     flw   fa3, 0x0(t6)       A_compressed[3]
───── ────────── ──────────────── ──────────────────────── ──────────────
 16   80000194   f3 2f 00 cc     csrr  t6, tid            Meta-load: tid
 17   80000198   93 9f 2f 00     slli  t6, t6, 0x2        Meta-load: tid * 4
 18   8000019c   b3 8f f2 01     add   t6, t0, t6         Meta-load: loop_offset + tid*4
 19   800001a0   b3 0f fe 01     add   t6, t3, t6         Meta-load: + meta_base → ptr
 20   800001a4   07 a7 0f 00     flw   fa4, 0x0(t6)       metadata word 0
───── ────────── ──────────────── ──────────────────────── ──────────────
 21   800001a8   f3 2f 00 cc     csrr  t6, tid            B-load: tid
 22   800001ac   13 d4 2f 00     srli  s0, t6, 0x2        B-load: row = tid>>2
 23   800001b0   93 9f 2f 00     slli  t6, t6, 0x2        B-load: tid<<2
 24   800001b4   93 ff cf 00     andi  t6, t6, 0xc        B-load: col = (tid&3)*4
 25   800001b8   33 04 14 03     mul   s0, s0, a7         B-load: row * K
 26   800001bc   b3 0f f4 01     add   t6, s0, t6         B-load: + col
 27   800001c0   b3 8f f2 01     add   t6, t0, t6         B-load: + loop_offset
 28   800001c4   b3 0f f3 01     add   t6, t1, t6         B-load: + B_base → ptr0
 29   800001c8   33 84 df 01     add   s0, t6, t4         B-load: ptr1 = ptr0 + K*2  [t4=K*2]
 30   800001cc   b3 04 d4 01     add   s1, s0, t4         B-load: ptr2 = ptr1 + K*2
 31   800001d0   33 89 d4 01     add   s2, s1, t4         B-load: ptr3 = ptr2 + K*2
 32   800001d4   07 ac 0f 00     flw   fs8, 0x0(t6)       B[0] (row0, n_group0)
 33   800001d8   87 2c 04 00     flw   fs9, 0x0(s0)       B[1] (row1, n_group0)
 34   800001dc   07 ad 04 00     flw   fs10, 0x0(s1)      B[2] (row2, n_group0)
 35   800001e0   87 2d 09 00     flw   fs11, 0x0(s2)      B[3] (row3, n_group0)
 36   800001e4   07 ae 0f 01     flw   ft8, 0x10(t6)      B[4] (row0, n_group1)
 37   800001e8   87 2e 04 01     flw   ft9, 0x10(s0)      B[5] (row1, n_group1)
 38   800001ec   07 af 04 01     flw   ft10, 0x10(s1)     B[6] (row2, n_group1)
 39   800001f0   87 2f 09 01     flw   ft11, 0x10(s2)     B[7] (row3, n_group1)
───── ────────── ──────────────── ──────────────────────── ──────────────
 40   800001f4   0b 84 14 04     .insn r (MMA)            mma_sync [rd=x8(int32), rs1=x9(int8), rs2=x1(sparse)]
 41   800001f8   93 82 02 02     addi  t0, t0, 0x20       loop_offset += 0x20 (shared B+meta stride)
 42   800001fc   13 0f 0f 01     addi  t5, t5, 0x10       A_ptr += 0x10 (a_k_stride_sp = tileK/2 = 16)
 43   80000200   e3 cc 12 f5     blt   t0, a7, loop       if loop_offset < K
```

### 4d. Key structural differences (INT8)

1. **A-load: −6 insns** (21→15). Largest per-section saving. 4 fewer flw + 2 fewer addr computations.

2. **Meta-load: +5 insns** (0→5). The extra `add` (vs FP16's 4) comes from sharing `t0` as both loop counter and meta offset — requires an extra addition to combine base and offset.

3. **B-load: −2 insns** (21→19). Stride chaining saves 2 instructions over independent mul groups.

4. **MMA+ctrl: +1 insn** (3→4). Dense has 1 addi + MMA + branch. Sparse shares B/meta stride (both 0x20), so 2 addi (t0 for B+meta, t5 for A) + MMA + branch.

5. **Net loop: −2 insns/iter** (45→43). Second-best loop savings after FP16.

---

## 5. INT4/INT32 Analysis

### 5a. Section-by-section commentary

**Prologue (D=5, S=8)**: Identical to INT8 in structure and count. Dense: 4 fsw + addi sp = 5. Sparse: 3 sw + 4 fsw + addi sp = 8.

**Setup (D=20, S=21)**: Identical to INT8. Sparse loads `meta_addr` (+1).

**Pre-loop (D=10, S=18)**: Dense identical to INT8 (10 insns). Sparse has 18 (vs INT8's 20) because INT4's `num_k_tiles = K/32` computation uses `srli K, 5` (1 insn) instead of INT8's mask construction (`lui+addi+and` = 3 insns for the `andi t4, K/4, -0x8` mask that doesn't fit in 12 bits). This saves 2 instructions.

**A-load (D=21, S=15)**: Identical to INT8 (sizeof(dtype)=1 for both, same i_ratio=4).

**Meta-load (D=0, S=5)**: INT4 loads 2 metadata words (fa4, fa5) because meta_cols=4 > cols_per_load=2, requiring num_meta_loads=2. The 5 insns are: `csrr+slli+add+flw+flw`. Same total as INT8 despite an extra flw, because INT4's meta pointer is direct (no shared offset), saving one `add`.

**B-load (D=21, S=19)**: Identical to INT8 in both dense and sparse.

**MMA+ctrl (D=3, S=5)**: Dense: 1 addi + MMA + branch = 3. Sparse: 3 addi (t0 for B, t4 for meta, t5 for A — all three strides differ) + MMA + branch = 5. This is +2 over dense (vs INT8's +1) because INT4's meta stride (0x40) differs from B stride (0x20), preventing the shared-counter optimization that INT8 enjoys.

**Epilogue (D=41, S=44)**: Identical to INT8.

### 5b. Full annotated loop body — Dense INT4 (45 instructions)

The dense INT4 loop is **bit-identical** to the dense INT8 loop (section 4b above) except for the MMA encoding byte:

```
 43   800001c8   93 82 02 02     addi  t0, t0, 0x20       loop_i += 0x20
 44   800001cc   0b 84 05 04     .insn r (MMA)            mma_sync [rd=x8(int32), rs1=x11(int4), rs2=x0(dense)]
 45   800001d0   e3 e8 12 f5     bltu  t0, a7, loop
```

MMA encoding differs only in rs1: `x11` (int4::id=11) vs `x9` (int8::id=9). All address computations are identical because both types use `sizeof(dtype)=1` (`uint8_t` for int4, `int8_t` for int8), giving `i_ratio=4` for both.

### 5c. Full annotated loop body — Sparse INT4 (44 instructions)

```
Line  Addr       Hex              Instruction              Section
───── ────────── ──────────────── ──────────────────────── ──────────────
  1   80000150   f3 2f 00 cc     csrr  t6, tid            A-load: tid
  2   80000154   13 d4 1f 00     srli  s0, t6, 0x1        A-load: row = tid>>1
  3   80000158   93 9f 2f 00     slli  t6, t6, 0x2        A-load: tid<<2
  4   8000015c   93 ff 4f 00     andi  t6, t6, 0x4        A-load: col = (tid&1)*4
  5   80000160   b3 04 64 02     mul   s1, s0, t1         A-load: row * (K/2)  [t1=srli(K,1)]
  6   80000164   b3 84 f4 01     add   s1, s1, t6         A-load: + col
  7   80000168   b3 04 9f 00     add   s1, t5, s1         A-load: + A_base → ptr0
  8   8000016c   07 a5 84 ff     flw   fa0, -0x8(s1)      A_compressed[0]
  9   80000170   87 a5 04 00     flw   fa1, 0x0(s1)       A_compressed[1]
 10   80000174   13 04 44 00     addi  s0, s0, 0x4        A-load: row += tcM
 11   80000178   33 04 83 02     mul   s0, t1, s0         A-load: (K/2) * (row+4)
 12   8000017c   b3 0f ff 01     add   t6, t5, t6         A-load: A_base + col
 13   80000180   b3 8f 8f 00     add   t6, t6, s0         A-load: + (K/2)*(row+4)
 14   80000184   07 a6 8f ff     flw   fa2, -0x8(t6)      A_compressed[2]
 15   80000188   87 a6 0f 00     flw   fa3, 0x0(t6)       A_compressed[3]
───── ────────── ──────────────── ──────────────────────── ──────────────
 16   8000018c   f3 2f 00 cc     csrr  t6, tid            Meta-load: tid
 17   80000190   93 9f 2f 00     slli  t6, t6, 0x2        Meta-load: tid * 4
 18   80000194   b3 8f fe 01     add   t6, t4, t6         Meta-load: meta_ptr + tid*4
 19   80000198   07 a7 0f fe     flw   fa4, -0x20(t6)     metadata word 0
 20   8000019c   87 a7 0f 00     flw   fa5, 0x0(t6)       metadata word 1
───── ────────── ──────────────── ──────────────────────── ──────────────
 21   800001a0   f3 2f 00 cc     csrr  t6, tid            B-load: tid
 22   800001a4   13 d4 2f 00     srli  s0, t6, 0x2        B-load: row = tid>>2
 23   800001a8   93 9f 2f 00     slli  t6, t6, 0x2        B-load: tid<<2
 24   800001ac   93 ff cf 00     andi  t6, t6, 0xc        B-load: col = (tid&3)*4
 25   800001b0   33 04 14 03     mul   s0, s0, a7         B-load: row * K
 26   800001b4   b3 0f f4 01     add   t6, s0, t6         B-load: + col
 27   800001b8   b3 8f f2 01     add   t6, t0, t6         B-load: + loop_offset
 28   800001bc   b3 8f f3 01     add   t6, t2, t6         B-load: + B_base → ptr0
 29   800001c0   33 84 cf 01     add   s0, t6, t3         B-load: ptr1 = ptr0 + K*2  [t3=K*2]
 30   800001c4   b3 04 c4 01     add   s1, s0, t3         B-load: ptr2 = ptr1 + K*2
 31   800001c8   33 89 c4 01     add   s2, s1, t3         B-load: ptr3 = ptr2 + K*2
 32   800001cc   07 ac 0f 00     flw   fs8, 0x0(t6)       B[0]
 33   800001d0   87 2c 04 00     flw   fs9, 0x0(s0)       B[1]
 34   800001d4   07 ad 04 00     flw   fs10, 0x0(s1)      B[2]
 35   800001d8   87 2d 09 00     flw   fs11, 0x0(s2)      B[3]
 36   800001dc   07 ae 0f 01     flw   ft8, 0x10(t6)      B[4]
 37   800001e0   87 2e 04 01     flw   ft9, 0x10(s0)      B[5]
 38   800001e4   07 af 04 01     flw   ft10, 0x10(s1)     B[6]
 39   800001e8   87 2f 09 01     flw   ft11, 0x10(s2)     B[7]
───── ────────── ──────────────── ──────────────────────── ──────────────
 40   800001ec   0b 84 15 04     .insn r (MMA)            mma_sync [rd=x8(int32), rs1=x11(int4), rs2=x1(sparse)]
 41   800001f0   93 82 02 02     addi  t0, t0, 0x20       loop_offset += 0x20 (B stride)
 42   800001f4   93 8e 0e 04     addi  t4, t4, 0x40       meta_ptr += 0x40 (meta_stride*4 = 16*4)
 43   800001f8   13 0f 0f 01     addi  t5, t5, 0x10       A_ptr += 0x10 (a_k_stride_sp = 16)
 44   800001fc   e3 ca 12 f5     blt   t0, a7, loop       if loop_offset < K
```

### 5d. Key structural differences (INT4)

1. **A-load identical to INT8** (both dense and sparse). `sizeof(uint8_t) = sizeof(int8_t) = 1`, so all pointer arithmetic is the same.

2. **Meta-load: 2 flw instead of 1** (5 insns total, same as INT8). INT4 has meta_cols=4 but cols_per_load=2, requiring num_meta_loads=2. The second flw uses offset `-0x20` from the same base, so no extra address computation — just one more flw. The `add` count happens to be 1 less than INT8 (direct meta_ptr instead of shared counter), making the total the same at 5.

3. **MMA+ctrl: 3 separate pointer advances** (5 insns). INT4's meta stride (0x40 = meta_stride*4 = 16*4) differs from B stride (0x20), so they can't share a counter like INT8. This costs +2 over dense (vs INT8's +1).

4. **Net loop: only −1 insn/iter** (45→44). The worst savings because meta-load has 2 flw (+1 over FP16/INT8) and MMA+ctrl has 3 separate advances (+2 over dense).

---

## 6. Cross-Type Comparison

### 6a. Why dense INT8 and INT4 are bit-identical

Both INT8 (`int8_t`, sizeof=1) and INT4 (`uint8_t`, sizeof=1) produce identical `i_ratio = 4`, `tileK = 32`, and all addressing code. The compiler generates the same instructions because:
- Same `a_block_size`, `b_block_size`, stride computations
- Same number of flw/fsw in every section
- Only difference: MMA encoding rs1 field (x9 vs x11)

Hex comparison: `0b 84 04 04` (INT8) vs `0b 84 05 04` (INT4) — differ in byte 3 only.

### 6b. Why FP16 has more instructions

FP16 (`uint16_t`, sizeof=2) has `i_ratio = 2`, causing:
1. **Two loop counters (dense)**: Byte stride per tileK = 0x20, element stride = 0x10. Dense INT8/INT4 have byte stride = element stride = 0x20, needing only 1 counter.
2. **Extra slli in A-load addressing**: A-pointer arithmetic uses `K*2` (halfword-to-byte conversion) instead of just `K`.
3. **+1 instruction in prologue/epilogue**: FP16 uses callee-saved `s0` for the A-load addressing scratch register, while INT8/INT4 can use caller-saved temporaries only.

### 6c. Meta-load scaling

| Type | meta_cols | num_meta_loads | flw in meta-load | Total meta insns |
|------|-----------|----------------|-------------------|-----------------|
| FP16 | 1 | 1 | 1 (fa4) | 4 |
| INT8 | 2 | 1 | 1 (fa4) | 5 |
| INT4 | 4 | 2 | 2 (fa4, fa5) | 5 |

INT8 has the same num_meta_loads as FP16 but needs +1 instruction for the shared counter address computation. INT4 has 2 loads but direct meta_ptr, cancelling out to the same 5 total as INT8.

### 6d. Loop body savings by type

| Type | Dense loop | Sparse loop | Savings/iter | Savings breakdown |
|------|-----------|-------------|--------------|-------------------|
| FP16 | 48 | 45 | **3** | A(−4) + meta(+4) + B(−4) + ctrl(+1) |
| INT8 | 45 | 43 | **2** | A(−6) + meta(+5) + B(−2) + ctrl(+1) |
| INT4 | 45 | 44 | **1** | A(−6) + meta(+5) + B(−2) + ctrl(+2) |

FP16 saves the most per iteration because its A-load and B-load savings are proportionally larger relative to its meta-load cost. INT4 saves the least because meta_load costs the same 5 instructions while MMA+ctrl costs +2 (vs INT8's +1).

### 6e. MMA encoding summary

| Variant | Hex bytes | rd (Ot::id) | rs1 (It::id) | rs2 (flags) |
|---------|-----------|-------------|---------------|-------------|
| Dense FP16 | `0b 80 00 04` | x0 (fp32) | x1 (fp16) | x0 (dense) |
| Dense INT8 | `0b 84 04 04` | x8 (int32) | x9 (int8) | x0 (dense) |
| Dense INT4 | `0b 84 05 04` | x8 (int32) | x11 (int4) | x0 (dense) |
| Sparse FP16 | `0b 80 10 04` | x0 (fp32) | x1 (fp16) | x1 (sparse) |
| Sparse INT8 | `0b 84 14 04` | x8 (int32) | x9 (int8) | x1 (sparse) |
| Sparse INT4 | `0b 84 15 04` | x8 (int32) | x11 (int4) | x1 (sparse) |

All use opcode=0x0b (CUSTOM0), funct3=0, funct7=2.

---

## 7. Dynamic Instruction Count

### 7a. Formulas

Per block: `Total = Overhead + LoopBody × T`, where `T = K / tileK`.

| Variant | Overhead | LoopBody | tileK |
|---------|----------|----------|-------|
| Dense FP16 | 81 | 48 | 16 |
| Dense INT8 | 76 | 45 | 32 |
| Dense INT4 | 76 | 45 | 32 |
| Sparse FP16 | 97 | 45 | 16 |
| Sparse INT8 | 93 | 43 | 32 |
| Sparse INT4 | 91 | 44 | 32 |

Overhead = Total − LoopBody = Prologue + Setup + Pre-loop + Epilogue.

### 7b. Break-even K (sparse has fewer dynamic instructions than dense)

| Type | Equation | T_break | K_break |
|------|----------|---------|---------|
| FP16 | 81 + 48T = 97 + 45T → 3T = 16 | 5.3 | 96 |
| INT8 | 76 + 45T = 93 + 43T → 2T = 17 | 8.5 | 288 |
| INT4 | 76 + 45T = 91 + 44T → T = 15 | 15.0 | 480 |

FP16 breaks even earliest (K=96) because it saves 3 instructions/iter — the most of any type.
INT4 breaks even latest (K=480) because it saves only 1 instruction/iter.

### 7c. Sample dynamic counts (NT=8)

**FP16** (tileK=16):

| K | T | Dense | Sparse | Δ (D−S) |
|---|---|-------|--------|---------|
| 16 | 1 | 129 | 142 | −13 |
| 32 | 2 | 177 | 187 | −10 |
| 64 | 4 | 273 | 277 | −4 |
| 128 | 8 | 465 | 457 | **+8** |
| 256 | 16 | 849 | 817 | **+32** |
| 512 | 32 | 1617 | 1537 | **+80** |
| 1024 | 64 | 3153 | 2977 | **+176** |

**INT8** (tileK=32):

| K | T | Dense | Sparse | Δ (D−S) |
|---|---|-------|--------|---------|
| 32 | 1 | 121 | 136 | −15 |
| 64 | 2 | 166 | 179 | −13 |
| 128 | 4 | 256 | 265 | −9 |
| 256 | 8 | 436 | 437 | **−1** |
| 512 | 16 | 796 | 781 | **+15** |
| 1024 | 32 | 1516 | 1469 | **+47** |
| 2048 | 64 | 2956 | 2845 | **+111** |

**INT4** (tileK=32):

| K | T | Dense | Sparse | Δ (D−S) |
|---|---|-------|--------|---------|
| 32 | 1 | 121 | 135 | −14 |
| 64 | 2 | 166 | 179 | −13 |
| 128 | 4 | 256 | 267 | −11 |
| 256 | 8 | 436 | 443 | −7 |
| 512 | 16 | 796 | 795 | **+1** |
| 1024 | 32 | 1516 | 1499 | **+17** |
| 2048 | 64 | 2956 | 2907 | **+49** |

(Positive Δ = sparse has fewer instructions)

### 7d. Key takeaway

**Software instruction count is NOT the primary performance driver.** The actual speedup from sparse TCU comes from hardware: each MMA executes `k_steps/2` micro-ops instead of `k_steps`, halving the TCU execution time. At typical matrix sizes (K≥256), the HW benefit dominates. The per-iteration instruction savings (1–3 insns) provide a small additional benefit by reducing instruction fetch/decode overhead.

---

## Source Files Referenced

- `tests/regression/sgemm_tcu/kernel.cpp` — Dense kernel (K-loop at line 31)
- `tests/regression/sgemm_tcu_sp/kernel.cpp` — Sparse kernel (K-loop at line 48)
- `kernel/include/vx_tensor.h` — Fragment load/store, meta load, mma_sync templates
- `sim/common/tensor_cfg.h` — `wmma_config_t` configuration parameters
- `00_workspace/kernel_dump_comparison_all_types.md` — Existing instruction count tables (cross-referenced)
- `00_workspace/int4_kernel_dump_analysis_v2.md` — Reference format for annotated assembly

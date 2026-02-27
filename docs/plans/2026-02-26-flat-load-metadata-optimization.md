# Flat Load Metadata Optimization

**Date**: 2026-02-26
**Status**: Implemented and verified

## Problem

`load_metadata_sync` issued `num_cols` `flw` instructions, one per metadata column. Each `flw` executes on ALL NT threads (SIMT), but the `meta_store<col>` HW only reads from threads 0..PER_WARP_DEPTH-1 (PD=4). Threads PD..NT-1 waste load bandwidth.

| Type | NT | `flw` instructions | Useful loads | Wasted loads | Efficiency |
|------|-----|---------------------|-------------|-------------|------------|
| fp16 | 8   | 1                   | 4            | 4           | 50%        |
| int8 | 8   | 2                   | 8            | 8           | 50%        |
| int4 | 8   | 4                   | 16           | 16          | 50%        |
| fp16 | 32  | 4                   | 16           | 112         | 12.5%      |
| int8 | 32  | 8                   | 32           | 224         | 12.5%      |
| int4 | 32  | 16                  | 64           | 448         | 12.5%      |

The int4/NT=32 case was particularly bad: 16 `flw` instructions with 448 wasted loads, contributing to -160% sparse slowdown.

## Solution: Flat Load with Thread-Offset Indexing

### Key Idea

1. **Flat load**: `base[l * NT + lane_id]` — one `flw` loads NT contiguous words
2. **Thread-offset HW**: `meta_store<col>` reads from thread group `(col % cols_per_load) * PD + r` for r=0..PD-1
3. **`cols_per_load`** = NT / PD (columns that fit in one flat load)
4. **`num_flat_loads`** = ceil(num_cols / cols_per_load)

### Memory Layout Change

Old: `per_k_tile_words = NT * num_cols` (thread-strided, padded)
New: `per_k_tile_words = PD * num_cols` (flat, bank-interleaved)

Flat layout: `word[sram_row + col * PD]` where sram_row=0..PD-1, col=0..num_cols-1

### Load Instruction Savings

| Type | NT | Before | After (ceil(PD*num_cols/NT)) | Saved |
|------|-----|--------|------------------------------|-------|
| fp16 | 8   | 1      | 1                            | 0     |
| int8 | 8   | 2      | 1                            | **1** |
| int4 | 8   | 4      | 2                            | **2** |
| fp16 | 32  | 4      | 1                            | **3** |
| int8 | 32  | 8      | 1                            | **7** |
| int4 | 32  | 16     | 2                            | **14**|

### Example: NT=8, int8 (PD=4, num_cols=2, cols_per_load=2, 1 flat load)

```
addr 0: bank 0 col 0    <- thread 0 loads
addr 1: bank 1 col 0    <- thread 1 loads
addr 2: bank 2 col 0    <- thread 2 loads
addr 3: bank 3 col 0    <- thread 3 loads
addr 4: bank 0 col 1    <- thread 4 loads
addr 5: bank 1 col 1    <- thread 5 loads
addr 6: bank 2 col 1    <- thread 6 loads
addr 7: bank 3 col 1    <- thread 7 loads
```

meta_store<0>: HW reads threads 0-3 (bank 0-3, col 0)
meta_store<1>: HW reads threads 4-7 (bank 0-3, col 1)

### Example: NT=8, int4 (PD=4, num_cols=4, cols_per_load=2, 2 flat loads)

```
l=0: load addrs 0-7   -> meta_store<0> from threads 0-3, meta_store<1> from threads 4-7
l=1: load addrs 8-15  -> meta_store<2> from threads 0-3, meta_store<3> from threads 4-7
```

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `hw/rtl/tcu/VX_tcu_core.sv` | Thread-offset meta_wr_data selection |
| `kernel/include/vx_tensor.h` | Flat load + nested unroll in load_metadata_sync |
| `tests/regression/sgemm_tcu_struct_sparse/main.cpp` | Transpose pack_metadata layout + reduce buffer |
| `tests/regression/sgemm_tcu_struct_sparse/kernel.cpp` | per_k_tile_words = PD * meta_cols |

No changes to: `VX_tcu_meta.sv`, `VX_tcu_pkg.sv`, `tensor_cfg.h`, SimX files.

### 1. HW: VX_tcu_core.sv — Thread-offset meta_wr_data

Added localparams and offset wire:

```verilog
localparam COLS_PER_LOAD = TCU_BLOCK_CAP / PER_WARP_DEPTH;
localparam LG_CPL = $clog2(COLS_PER_LOAD);
localparam LG_PD  = $clog2(PER_WARP_DEPTH);
wire [$clog2(TCU_BLOCK_CAP)-1:0] meta_thread_offset;
assign meta_thread_offset = {fmt_d[LG_CPL-1:0], {LG_PD{1'b0}}};
for (genvar r = 0; r < PER_WARP_DEPTH; ++r) begin : g_meta_wr
    assign meta_wr_data[r] = 32'(execute_if.data.rs1_data[meta_thread_offset + r]);
end
```

`fmt_d[LG_CPL-1:0]` extracts low bits of col_idx, left-shift by log2(PD) gives the thread group start. For NT=8,PD=4: LG_CPL=1, uses `fmt_d[0]`. For NT=32,PD=4: LG_CPL=3, uses `fmt_d[2:0]`.

### 2. SW: vx_tensor.h — Flat load + nested unroll

```cpp
static void load_metadata_sync(const void* meta_ptr) {
    constexpr uint32_t rtl_i_ratio = 32 / It::bits;
    constexpr uint32_t num_cols = (NT * 2 * rtl_i_ratio) / 32;
    constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
    constexpr uint32_t cols_per_load = NT / PD;
    constexpr uint32_t num_loads = (num_cols + cols_per_load - 1) / cols_per_load;
    uint32_t lane_id = vx_thread_id();
    auto base = reinterpret_cast<const float*>(meta_ptr);
    detail::unroll_for<num_loads>([&](auto l) {
        float data = base[l * NT + lane_id];  // flat load
        detail::unroll_for<cols_per_load>([&](auto c) {
            constexpr uint32_t col = l * cols_per_load + c;
            if constexpr (col < num_cols) {
                meta_store<col>(data);
            }
        });
    });
}
```

### 3. Host: main.cpp — Transposed metadata layout

pack_metadata() changes:
- `per_k_tile_words` from `NT * mcols` to `PD * mcols`
- Storage index from `sram_row * mcols + word_idx` to `sram_row + word_idx * PD`
- Buffer allocation from `NT * meta_cols` to `PD * meta_cols` per k-tile

### 4. Kernel: kernel.cpp — Reduced stride

```cpp
using kcfg = vt::wmma_config_t<NUM_THREADS>;
constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
constexpr uint32_t per_k_tile_words = PD * meta_cols;
```

Note: Uses `vt::wmma_config_t<NUM_THREADS>` (aliased `kcfg`) because `cfg` is private inside `wmma_context`. m_steps and k_steps are type-independent (only depend on NT), so PD is correct for all input types.

## Verification

All 6 sparse configs pass (3 types x 2 NTs):

| Type | NT=8 | NT=32 |
|------|------|-------|
| fp16/fp32 | PASSED | PASSED |
| int8/int32 | PASSED | PASSED |
| int4/int32 | PASSED | PASSED |

## Performance Results (TCU_CYCLES)

Test sizes: NT=8 uses `-m8 -n8 -k32`, NT=32 uses `-m16 -n16 -k64`.

| Config | NT | Before | After | Cycles Saved | Speedup |
|--------|-----|--------|-------|-------------|---------|
| fp16/fp32 | 8 | 1492 | 1544 | -52 | -3.5% |
| int8/int32 | 8 | 1594 | 1554 | 40 | 2.5% |
| int4/int32 | 8 | 853 | 719 | 134 | **15.7%** |
| fp16/fp32 | 32 | 2778 | 2409 | 369 | **13.3%** |
| int8/int32 | 32 | 2971 | 2487 | 484 | **16.3%** |
| int4/int32 | 32 | 2490 | 1162 | 1328 | **53.3%** |

### Observations

- Savings scale directly with `num_cols` (number of `flw` instructions saved)
- **int4/int32 NT=32**: Biggest win (53.3%) — 14 of 16 `flw` eliminated
- **int8/int32 NT=32**: 16.3% — 7 of 8 `flw` eliminated
- **fp16/fp32 NT=32**: 13.3% — 3 of 4 `flw` eliminated
- **fp16/fp32 NT=8**: Slight regression (-3.5%) — 0 `flw` saved (num_cols=1, already optimal); different instruction scheduling in the flat-load path causes minor overhead
- The optimization is most impactful at NT=32 where the SIMT width mismatch was worst

## Dense vs Sparse (Flat Load) — TCU K-loop Cycles (csr_read 0xB00)

```
┌─────┬──────┬────────────┬─────────────┬──────────────┬─────────┐
│ NT  │ Type │    Size    │ Dense (cyc) │ Sparse (cyc) │ Speedup │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 8   │ fp16 │ m8n8k32    │ 1879        │ 1544         │ 17.8%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 8   │ fp16 │ m16n16k64  │ 5413        │ 4555         │ 15.8%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 32  │ fp16 │ m16n16k64  │ 2719        │ 2409         │ 11.4%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 32  │ fp16 │ m32n32k128 │ 9244*       │ 10015        │ -8.3%*  │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 8   │ int8 │ m8n8k32    │ 2694        │ 1554         │ 42.3%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 8   │ int8 │ m16n16k64  │ 10609       │ 4168         │ 60.7%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 32  │ int8 │ m16n16k64  │ 6504        │ 2487         │ 61.8%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 32  │ int8 │ m32n32k128 │ 36544       │ 10186        │ 72.1%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 8   │ int4 │ m8n8k32    │ 541         │ 719          │ -32.9%  │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 8   │ int4 │ m16n16k64  │ 2082        │ 1805         │ 13.3%   │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 32  │ int4 │ m16n16k64  │ 878         │ 1162         │ -32.3%  │
├─────┼──────┼────────────┼─────────────┼──────────────┼─────────┤
│ 32  │ int4 │ m32n32k128 │ 3606        │ 3137         │ 13.0%   │
└─────┴──────┴────────────┴─────────────┴──────────────┴─────────┘
* fp16 NT=32 m32n32k128 dense FAILED verification (pre-existing upstream bug)
```

### Comparison with previous results (before flat load optimization)

| Config | Before flat load | After flat load | Change |
|--------|-----------------|-----------------|--------|
| int4 NT=32 m32n32k128 | -145.1% | **+13.0%** | +158pp |
| int4 NT=32 m16n16k64 | -160.0% | -32.3% | +128pp |
| int4 NT=8 m16n16k64 | -14.7% | **+13.3%** | +28pp |
| fp16 NT=32 m16n16k64 | -3.7% | **+11.4%** | +15pp |
| int8 NT=32 m16n16k64 | 45.3% | **61.8%** | +17pp |
| int8 NT=32 m32n32k128 | 65.8% | **72.1%** | +6pp |
| int8 NT=8 m16n16k64 | 57.1% | **60.7%** | +4pp |
| int8 NT=8 m8n8k32 | 36.0% | **42.3%** | +6pp |

### Observations

- **int8** is the best sparse use case: 42-72% speedup across all sizes
- **fp16** benefits at small-medium sizes (11-18% speedup), but overhead dominates at larger NT=32 sizes
- **int4 at larger sizes** (m16n16k64+ for NT=8, m32n32k128 for NT=32) now shows positive speedup (+13%) — previously always negative
- **int4 at minimum sizes** still negative — metadata overhead not amortized at tiny matrix sizes
- The flat load optimization was transformative for int4 NT=32: from -145%/-160% to -32%/+13%

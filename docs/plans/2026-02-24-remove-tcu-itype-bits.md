# Remove TCU_ITYPE_BITS: Dynamic Multi-Type Sparse TCU

**Date**: 2026-02-24
**Status**: Implemented

## Problem

Switching the sparse TCU between input types (int8, fp16, int4) required rebuilding the RTL with `-DTCU_ITYPE_BITS=N`. This was because VX_tcu_meta (metadata SRAM) and VX_tcu_sel (B-column gather mux) were sized/structured at compile time based on `I_RATIO = 32 / TCU_ITYPE_BITS`.

## Solution

Eliminate `TCU_ITYPE_BITS` entirely. A single RTL build supports all sparse input types, selected at runtime via `fmt_s` — matching the upstream FEDP pattern (VX_tcu_fedp_dpi.sv already handles all types dynamically).

## Design

### VX_tcu_meta
Sized SRAM to worst-case width (`TCU_MAX_META_BLOCK_WIDTH = TCU_NT * 2 * TCU_MAX_ELT_RATIO`). Software writes only the columns it needs; unused columns retain init data (harmless).

### VX_tcu_sel
Instantiates all three gather variants (I_RATIO=2,4,8) in parallel. Each reads its own metadata row slice via compile-time `ROW_IDX` parameter. A final mux selects output based on `fmt_s`:
- `TCU_FP16_ID, TCU_BF16_ID` → r2 path (ELT_W=16)
- `TCU_NVFP4_ID, TCU_I4_ID, TCU_U4_ID` → r8 path (ELT_W=4)
- default (int8, fp8, etc.) → r4 path (ELT_W=8)

### VX_tcu_core
Removed `TCU_ITYPE_BITS` ifdef block. Passes `fmt_s` and full `vld_meta_block` to VX_tcu_sel.

## Files Changed

| File | Change |
|------|--------|
| `hw/rtl/tcu/VX_tcu_pkg.sv` | Added `TCU_MAX_META_ROW_WIDTH`, `TCU_MAX_META_BLOCK_WIDTH` |
| `hw/rtl/tcu/VX_tcu_core.sv` | Removed `TCU_ITYPE_BITS` block, use MAX constants |
| `hw/rtl/tcu/VX_tcu_sel.sv` | Full redesign: three parallel gather paths + output mux |
| `hw/rtl/tcu/VX_tcu_meta.sv` | No changes (parameterized by `META_BLOCK_WIDTH`) |
| Build scripts (4 files) | Removed `-DTCU_ITYPE_BITS=$BITS`, single RTL build |

## Area Impact

| Component | Before | After | Overhead |
|-----------|--------|-------|----------|
| VX_tcu_meta SRAM | Sized for 1 type | Sized for max (int4) | Width grows for fp16/int8 |
| VX_tcu_sel muxes | 1 variant | 3 variants + output mux | ~3x mux logic (combinational) |

For NT=8: 4 VX_tcu_sel instances x 3 variants = 12 gather paths (was 4). All purely combinational priority muxes on 32-bit words.

## Build

```bash
# One RTL build, all types:
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI" make -C runtime/rtlsim

# Switch test type without RTL rebuild:
CONFIGS="-DNUM_THREADS=8 -DITYPE=int8 -DOTYPE=int32" make -C tests/regression/sgemm_tcu_struct_sparse
CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32" make -C tests/regression/sgemm_tcu_struct_sparse
CONFIGS="-DNUM_THREADS=8 -DITYPE=int4 -DOTYPE=int32" make -C tests/regression/sgemm_tcu_struct_sparse
```

# TLB Integration - Step 1: Replace demo_module with VX_tlb_wrapper

## Summary
Successfully replaced `demo_module` with `VX_tlb_wrapper` in the cache pipeline. This is a drop-in replacement that maintains the same pass-through behavior while preparing for future TLB functionality.

## Changes Made

### 1. Created `VX_tlb_wrapper.sv`
- **Location**: `/nethome/pgoel61/vortex/hw/rtl/cache/VX_tlb_wrapper.sv`
- **Purpose**: Per-bank TLB wrapper stage that sits between request crossbar and cache banks
- **Current Behavior**: Simple pass-through with 1-entry elastic buffer (identical to demo_module)
- **Future**: Will integrate TLB lookup and page table walk logic

**Key Features:**
- Maintains all signal paths from demo_module
- Preserves memory request injection capability (for future PTW)
- Includes INSTANCE_ID and BANK_ID parameters for debug tracing
- Has hooks for memory arbiter monitoring (arb_mem_req_* ports)

### 2. Updated `VX_cache.sv`
Modified to instantiate `VX_tlb_wrapper` instead of `demo_module`:

#### Changed Sections:
1. **Module instantiation (lines 383-440)**:
   - Renamed loop from `g_demo_stage` → `g_tlb_wrapper`
   - Changed module from `demo_module` → `VX_tlb_wrapper`
   - Updated instance name from `demo_stage` → `tlb_wrapper`
   - Added INSTANCE_ID and BANK_ID parameters

2. **Signal renaming throughout**:
   - `demo_mem_req_*` → `tlb_mem_req_*` (all 8 signals × NUM_BANKS)
   - Updated comments to reflect TLB purpose

3. **Memory arbiter connections (lines 602-631)**:
   - Changed signal names in declarations
   - Updated arbiter input concatenations
   - Updated comments: "demo requests" → "TLB requests (for page table walks)"

## Architecture Position

```
Core Requests 
    ↓
Request Crossbar (VX_stream_xbar)
    ↓
[VX_tlb_wrapper] ← NEW (per-bank, only when VM_ENABLE)
    ↓
Cache Banks (VX_cache_bank)
```

When `VM_ENABLE` is defined, requests flow through TLB wrapper before reaching banks.

## Signal Flow

### Request Path:
- **Input**: `per_bank_core_req_*[bank_id]` (from crossbar)
- **Output**: `per_bank_core_req_*_d[bank_id]` (to bank)
- **Delayed by**: 1 cycle (elastic buffer)

### Memory Request Injection:
- **Source**: `tlb_mem_req_*[bank_id]` (from TLB wrapper)
- **Destination**: Memory arbiter (alongside bank fill/writeback requests)
- **Current Status**: Tied to `1'b0` (no requests generated)
- **Future Use**: Page table walk requests on TLB miss

## Compilation Status
✅ **No linter errors** - Both files compile cleanly

## Functional Equivalence
The current implementation is **functionally identical** to demo_module:
- Same 1-cycle latency
- Same elastic buffer behavior
- Same backpressure propagation
- No memory requests generated yet

## Next Steps

### Step 2: Integrate Actual TLB Logic (Future)
Inside `VX_tlb_wrapper.sv`, replace pass-through logic with:
1. **TLB lookup** on incoming address
2. **Address translation** on TLB hit
3. **PTW request generation** on TLB miss
4. **Stall handling** while waiting for PTW response

### Remaining TODOs:
- [ ] Fix VX_tlb_wrap.sv module name/signature (cluster-level wrapper)
- [ ] Stub VX_tlb.sv into compiling minimal pass-through TLB
- [ ] Stub VX_ptw.sv into compiling minimal module

## Testing Recommendations
1. **Regression test**: Ensure cache still works with `VM_ENABLE` defined
2. **Compare traces**: Verify identical behavior to demo_module version
3. **Timing**: Check that 1-cycle delay doesn't break timing constraints

## Files Modified
- ✅ `VX_tlb_wrapper.sv` (created, 153 lines)
- ✅ `VX_cache.sv` (modified, ~30 lines changed)

## Files Unchanged (for now)
- `demo_module.sv` (kept for reference)
- `VX_tlb_wrap.sv` (cluster-level, will fix in next step)
- `VX_tlb.sv` (TLB core, will stub in next step)
- `VX_ptw.sv` (PTW, will stub in next step)
- `VX_cache_cluster.sv` (has TLB integration issues, not touched yet)


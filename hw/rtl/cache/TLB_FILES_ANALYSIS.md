# TLB Files Analysis and Integration Plan

## Files Overview

### 1. VX_tlb_define.vh ✅ GOOD
**Status**: Ready to use as-is

**Contents**:
- Sv32/Sv39 page table entry definitions
- Permission flags (`V`, `R`, `W`, `X`, `U`, `G`, `A`, `D`)
- TLB entry typedef (`tlb_entry_t`)
- Helper functions for permission checking

**No changes needed** - This provides solid foundation for TLB entries.

---

### 2. VX_tlb_bank.sv ✅ FIXED
**Status**: Fixed major issues, now compilable

**What Was Fixed**:
1. ❌ **Was**: Combinational `always @(*)` blocks updating arrays
   ✅ **Now**: Sequential `always @(posedge clk)` with proper reset

2. ❌ **Was**: Incomplete reset logic
   ✅ **Now**: Full array initialization on reset

**Current Structure**:
- **Input**: Single request per cycle
  - `core_req_valid`, `core_req_addr` (virtual), `core_req_rw`, `core_req_tag`
- **Output**: Single response per cycle
  - `core_rsp_valid`, `core_rsp_addr` (physical), `core_rsp_rw`, `core_rsp_tag`
- **TLB Miss Interface**:
  - `tlb_miss_valid`, `tlb_miss_vpn` → to PTW
  - `tlb_update_valid`, `tlb_update_vpn`, `tlb_update_ppn` ← from PTW

**Internal Components**:
- Tag array: VPN storage (associative lookup)
- Data array: PPN + flags storage
- PLRU replacement policy (`VX_cache_repl`)
- One-hot encoder for hit way selection
- Elastic buffer for responses

**Key Behavior**:
- **TLB Hit**: Translates immediately, pushes to response queue
- **TLB Miss**: Stalls the request, asserts `tlb_miss_valid`
- **PTW Update**: Accepts translation and updates arrays

**Latency**: 1-2 cycles (lookup + response queue)

---

### 3. VX_tlb.sv ⚠️ NOT SUITABLE FOR WRAPPER
**Status**: Has issues, designed for multi-request crossbar scenario

**Major Issues**:
1. ❌ `localparam` declarations **inside** port list (lines 48-61) - **SYNTAX ERROR!**
   ```systemverilog
   ) (
       localparam PAGE_OFFSET_WIDTH = 12;  // ❌ WRONG PLACE!
   ```

2. ❌ Missing signal declarations (`CORE_REQ_DATAW`, `CORE_RSP_DATAW`, `per_bank_req_idx`)

3. ⚠️ **Architectural Mismatch**: Designed for:
   - Multiple requests (`NUM_REQS`) → Request Crossbar → Multiple TLB banks
   - Per-bank routing and response gathering
   - This is **top-level multi-bank TLB**, not per-bank unit

**Why Not Use in Wrapper?**:
- VX_tlb_wrapper is **already per-bank** (instantiated NUM_BANKS times in VX_cache.sv)
- Using VX_tlb.sv would create nested banking (banks within banks)
- Wrapper needs simple single-request-in/single-response-out TLB

**Recommendation**: 
- ❌ **Do NOT use VX_tlb.sv in VX_tlb_wrapper**
- ✅ **Use VX_tlb_bank.sv directly in VX_tlb_wrapper**
- Keep VX_tlb.sv for reference/future cluster-level use

---

## Integration Plan for VX_tlb_wrapper

### Phase 1: Simple TLB with Pass-Through on Miss (CURRENT GOAL)

**Goal**: TLB hit → translate; TLB miss → pass-through (no stall, go to next cache level)

**Approach**:
```
VX_tlb_wrapper (per-bank):
├── Input: per_bank_core_req_* (line address from crossbar)
├── VX_tlb_bank (instantiate 1 per wrapper)
│   ├── On Hit: Translate VPN → PPN
│   └── On Miss: Set miss flag
├── On Miss: Pass original address through (no translation)
└── Output: per_bank_core_req_*_d (to cache bank)
```

**Key Changes to VX_tlb_wrapper.sv**:
1. Instantiate `VX_tlb_bank`
2. Convert line address ↔ full address (add/strip bank+word bits)
3. On TLB hit: Use translated address
4. On TLB miss: Pass original address (no stall!)
5. Forward all other signals (rw, byteen, data, tag, etc.)

**Behavior**:
- **Hit**: 1-2 cycle translation latency
- **Miss**: 1 cycle pass-through (elastic buffer)
- **No PTW yet**: Miss flag ignored, no page table walks

---

### Phase 2: TLB with PTW (FUTURE)

**Goal**: TLB miss → trigger PTW → update TLB → retry request

**Additional Changes**:
1. Add state machine to handle miss/PTW/retry
2. Use `mem_req_*` outputs to generate PTW requests
3. Wait for PTW response (`tlb_update_valid`)
4. Replay stalled request after TLB update

---

## Interface Mapping

### VX_tlb_wrapper → VX_tlb_bank

| Wrapper Signal | TLB Bank Signal | Conversion Needed? |
|----------------|------------------|--------------------|
| `in_addr` (line addr) | `core_req_addr` (full VA) | ✅ Add bank+word bits back |
| `in_valid` | `core_req_valid` | ✅ Direct |
| `in_rw` | `core_req_rw` | ✅ Direct |
| `in_tag` | `core_req_tag` | ✅ Direct |
| `out_addr` | `core_rsp_addr` (full PA) | ✅ Strip to line addr |
| `out_valid` | `core_rsp_valid` | ✅ Direct |
| `mem_req_*` | `tlb_miss_*` | ⚠️ Need FSM for PTW |

### Address Conversion

**Wrapper receives** (from crossbar):
```
in_addr [ADDR_WIDTH-1:0]  // CS_LINE_ADDR_WIDTH
        ├─ Tag bits
        └─ Line index bits
```

**TLB bank expects** (full virtual address):
```
core_req_addr [`XLEN-1:0]
        ├─ [XLEN-1:12]  VPN (Virtual Page Number)
        └─ [11:0]        Page Offset
```

**Reconstruction**:
```systemverilog
// Add back bank ID and word select to form full virtual address
wire [`XLEN-1:0] full_vaddr = {in_addr, bank_id_bits, word_sel_bits};
```

**TLB bank returns** (full physical address):
```
core_rsp_addr [`XLEN-1:0]
        ├─ [XLEN-1:12]  PPN (Physical Page Number) ← TRANSLATED
        └─ [11:0]        Page Offset ← UNCHANGED
```

**Extraction**:
```systemverilog
// Extract line address from full physical address
wire [ADDR_WIDTH-1:0] line_paddr = full_paddr[ADDR_WIDTH+WORD_SEL_BITS+BANK_SEL_BITS-1 : WORD_SEL_BITS+BANK_SEL_BITS];
```

---

## Compilation Status

| File | Status | Issues |
|------|--------|--------|
| VX_tlb_define.vh | ✅ Clean | None |
| VX_tlb_bank.sv | ✅ Fixed | Sequential logic corrected |
| VX_tlb.sv | ❌ Broken | Localparam in port list, missing signals |
| VX_tlb_wrapper.sv | ⚠️ Pass-through | Need to add TLB integration |

---

## Next Steps

1. ✅ **DONE**: Fix VX_tlb_bank.sv sequential logic
2. **TODO**: Update VX_tlb_wrapper.sv to instantiate VX_tlb_bank
3. **TODO**: Add address conversion logic
4. **TODO**: Handle TLB miss pass-through
5. **FUTURE**: Add PTW state machine

---

## Testing Strategy

### Phase 1 Testing (TLB + Pass-through on Miss):
1. **Empty TLB**: All requests miss → pass-through → should work like before
2. **Manual TLB Fill**: Use debug interface to populate TLB entries
3. **Hit Test**: Verify translation happens correctly
4. **Miss Test**: Verify pass-through doesn't stall

### Phase 2 Testing (TLB + PTW):
1. **PTW Generation**: Verify correct page table walk requests
2. **TLB Update**: Verify TLB fills from PTW responses
3. **Replay**: Verify stalled requests retry after TLB fill


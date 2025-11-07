# Demo Module Integration in VX_cache_cluster

## Overview
This document describes the integration of `demo_module` instances between the core arbiter and cache instances in `VX_cache_cluster.sv`. This is a preparatory step for TLB wrapper integration.

---

## Changes Made

### 1. **Integration Location**
The `demo_module` instances are inserted **between the arbiter output and cache input** in the request path:

```
Flow without VM_ENABLE:
  Core Arbiter → arb_core_bus_if → VX_cache_wrap

Flow with VM_ENABLE:
  Core Arbiter → arb_core_bus_if → demo_module → demo_core_bus_if → VX_cache_wrap
```

### 2. **Conditional Compilation**
All demo module code is wrapped in `ifdef VM_ENABLE` / `endif` blocks:
- When `VM_ENABLE` is **defined**: Demo modules are instantiated, data flows through them
- When `VM_ENABLE` is **not defined**: Direct connection from arbiter to cache (original behavior)

---

## Code Structure

### Intermediate Bus Interface (lines 152-157)
```systemverilog
`ifdef VM_ENABLE
    // Create intermediate bus interfaces between arbiter and demo modules
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (ARB_TAG_WIDTH)
    ) demo_core_bus_if[NUM_CACHES * NUM_REQS]();
```

**Purpose:** Creates bus interfaces to carry data from demo_module outputs to cache inputs.

**Array Size:** `NUM_CACHES × NUM_REQS` (e.g., 2 cache units × 4 requests = 8 bus interfaces)

---

### Demo Module Instantiation (lines 159-226)

```systemverilog
// Instantiate demo_module for each cache unit (one per request channel)
for (genvar i = 0; i < NUM_CACHES; ++i) begin : g_demo_modules
    for (genvar j = 0; j < NUM_REQS; ++j) begin : g_demo_per_req
        demo_module #(...) demo_inst (...);
    end
end
```

**Instantiation Count:** `NUM_CACHES × NUM_REQS` demo modules
- Example: 2 cache units × 4 request channels = **8 demo_module instances**

**Why per-request-channel?**
- Each request channel operates independently
- Allows per-channel address translation (future TLB integration)
- Maintains parallelism at cluster level

---

### Parameter Configuration

```systemverilog
demo_module #(
    .ADDR_WIDTH      (`CS_WORD_ADDR_WIDTH),     // Word address width
    .WSEL_WIDTH      (1),                       // Not used (bank-level feature)
    .BYTEEN_WIDTH    (WORD_SIZE),               // Byte enable width
    .DATA_WIDTH      (WORD_SIZE * 8),           // Data width in bits
    .TAG_WIDTH       (ARB_TAG_WIDTH),           // Arbiter tag width
    .IDX_WIDTH       (1),                       // Not used (bank-level feature)
    .FLAGS_WIDTH     (`UP(FLAGS_WIDTH)),        // Request flags width
    .MEM_PORTS       (1),                       // Placeholder
    .MEM_ARB_SEL_WIDTH (1),                     // Placeholder
    .PROCESS_ADDR    (0)                        // Pass-through mode
) demo_inst (...)
```

**Key Settings:**
- `PROCESS_ADDR = 0`: Configures demo_module in **pass-through mode**
- Unused parameters (WSEL_WIDTH, IDX_WIDTH): Set to 1 for synthesis compatibility
- Arbiter monitor ports: Tied off (not used in cluster level)

---

### Request Path Connections (lines 180-202)

```systemverilog
// Input from arbiter
.in_valid        (arb_core_bus_if[i * NUM_REQS + j].req_valid),
.in_addr         (arb_core_bus_if[i * NUM_REQS + j].req_data.addr),
.in_rw           (arb_core_bus_if[i * NUM_REQS + j].req_data.rw),
.in_byteen       (arb_core_bus_if[i * NUM_REQS + j].req_data.byteen),
.in_data         (arb_core_bus_if[i * NUM_REQS + j].req_data.data),
.in_tag          (arb_core_bus_if[i * NUM_REQS + j].req_data.tag),
.in_flags        (`UP(FLAGS_WIDTH)'(arb_core_bus_if[i * NUM_REQS + j].req_data.flags)),
.in_ready        (arb_core_bus_if[i * NUM_REQS + j].req_ready),

// Output to cache
.out_valid       (demo_core_bus_if[i * NUM_REQS + j].req_valid),
.out_addr        (demo_core_bus_if[i * NUM_REQS + j].req_data.addr),
.out_rw          (demo_core_bus_if[i * NUM_REQS + j].req_data.rw),
.out_byteen      (demo_core_bus_if[i * NUM_REQS + j].req_data.byteen),
.out_data        (demo_core_bus_if[i * NUM_REQS + j].req_data.data),
.out_tag         (demo_core_bus_if[i * NUM_REQS + j].req_data.tag),
.out_flags       (demo_core_bus_if[i * NUM_REQS + j].req_data.flags),
.out_ready       (demo_core_bus_if[i * NUM_REQS + j].req_ready),
```

**Data Flow:**
1. Arbiter output (`arb_core_bus_if`) → demo_module input
2. Demo_module applies 1-cycle buffering (elastic buffer)
3. Demo_module output → cache input (`demo_core_bus_if`)

---

### Response Path (lines 220-223)

```systemverilog
// Pass-through response path (demo_module doesn't modify responses)
assign demo_core_bus_if[i * NUM_REQS + j].rsp_valid = arb_core_bus_if[i * NUM_REQS + j].rsp_valid;
assign demo_core_bus_if[i * NUM_REQS + j].rsp_data  = arb_core_bus_if[i * NUM_REQS + j].rsp_data;
assign arb_core_bus_if[i * NUM_REQS + j].rsp_ready  = demo_core_bus_if[i * NUM_REQS + j].rsp_ready;
```

**Why direct assignment?**
- Demo_module only processes **request path** (address translation)
- **Response path** bypasses demo_module (no modification needed)
- Reduces latency for data returning to cores

---

### Cache Wrapper Connection Update (lines 260-266)

```systemverilog
`ifdef VM_ENABLE
    // When VM_ENABLE is defined, connect to demo_module output
    .core_bus_if (demo_core_bus_if[i * NUM_REQS +: NUM_REQS]),
`else 
    // When VM_ENABLE is not defined, connect directly to arbiter output
    .core_bus_if (arb_core_bus_if[i * NUM_REQS +: NUM_REQS]),
`endif
```

**Behavior:**
- `VM_ENABLE` defined: Cache sees demo_module-processed requests
- `VM_ENABLE` undefined: Cache sees arbiter output directly (original behavior)

---

## Signal Indexing Explanation

### Why `[i * NUM_REQS + j]`?

The bus interface arrays are **flattened 2D arrays**:

```
For NUM_CACHES=2, NUM_REQS=4:

Cache 0, Req 0 → index 0
Cache 0, Req 1 → index 1
Cache 0, Req 2 → index 2
Cache 0, Req 3 → index 3
Cache 1, Req 0 → index 4  (1*4 + 0)
Cache 1, Req 1 → index 5  (1*4 + 1)
Cache 1, Req 2 → index 6  (1*4 + 2)
Cache 1, Req 3 → index 7  (1*4 + 3)
```

**Formula:** `index = cache_id * NUM_REQS + request_channel`

---

## Latency Impact

### Request Path
- **Without VM_ENABLE:** Arbiter → Cache (arbiter latency only)
- **With VM_ENABLE:** Arbiter → Demo_module (1 cycle buffer) → Cache
  - **Added latency:** +1 cycle per request

### Response Path
- **No change:** Responses bypass demo_module via direct assignment

---

## Future TLB Integration Plan

This demo_module integration prepares for TLB wrapper:

### Phase 1 (Current)
```
Arbiter → demo_module (pass-through) → Cache
```

### Phase 2 (Next Step)
```
Arbiter → demo_module → TLB_wrapper → Cache
                ↓
            Address translation
```

### Phase 3 (Final)
```
Arbiter → TLB_wrapper (replaces demo_module) → Cache
```

**Migration Path:**
1. Test with demo_module in pass-through mode (verify no functionality change)
2. Replace demo_module instantiation with TLB_wrapper
3. Update port connections for TLB-specific signals (VPN, PPN, PTW interface)

---

## Testing Considerations

### Functional Testing
1. **Compile with VM_ENABLE undefined:** Verify original behavior unchanged
2. **Compile with VM_ENABLE defined:** Verify demo_module pass-through works
3. **Performance:** Measure +1 cycle latency impact on cache hit rate

### Synthesis Checks
- Verify no combinational loops introduced
- Check timing on demo_module elastic buffer
- Ensure unused ports are properly tied off (avoid floating signals)

---

## Files Modified

### VX_cache_cluster.sv
- **Lines 152-226:** Added VM_ENABLE block with demo_module instantiation
- **Lines 260-266:** Updated cache_wrap connection to use demo_core_bus_if when VM_ENABLE defined

### Files Referenced
- `demo_module.sv`: Pass-through module with address transform hook
- `VX_cache_define.vh`: Address width macros (`CS_WORD_ADDR_WIDTH`)

---

## Configuration Example

### Typical D-cache Cluster with VM_ENABLE
```systemverilog
VX_cache_cluster #(
    .NUM_UNITS   (2),          // 2 cache units
    .NUM_INPUTS  (4),          // 4 cores
    .NUM_REQS    (4),          // 4 LSU lanes per core
    .NUM_BANKS   (4),
    .WORD_SIZE   (16),
    .FLAGS_WIDTH (4)
) dcache_cluster (...)
```

**With VM_ENABLE defined:**
- **Demo modules instantiated:** 2 units × 4 reqs = **8 instances**
- **Intermediate buses:** 8 × `VX_mem_bus_if` (WORD_SIZE=16B, ARB_TAG_WIDTH)
- **Request latency:** +1 cycle from arbiter to cache

---

## Debug/Trace Hooks

### Future TLB Debug Points
When replacing demo_module with TLB:

```systemverilog
`ifdef DBG_TRACE_TLB
    always @(posedge clk) begin
        if (in_valid && in_ready) begin
            $display("TLB[%0d][%0d]: VA=0x%h → PA=0x%h", i, j, in_addr, out_addr);
        end
    end
`endif
```

### Performance Counters
Add to each demo_module (future TLB):
- TLB hits
- TLB misses
- PTW requests
- Stall cycles due to translation

---

## Known Limitations

1. **Bank-level features disabled at cluster level:**
   - Word select (WSEL): Not needed; cache handles internally
   - Request index (IDX): Not needed; handled by cache banks

2. **Memory arbiter monitoring disabled:**
   - `arb_mem_req_*` ports tied to 0
   - These are bank-level features (not visible at cluster level)

3. **Flags width handling:**
   - Uses `UP(FLAGS_WIDTH)` to handle FLAGS_WIDTH=0 case
   - Ensures synthesis doesn't create 0-width signals

---

## Next Steps

1. **Verify compilation:**
   ```bash
   make compile CONFIGS="-DVM_ENABLE"
   ```

2. **Run regression tests:**
   - Compare results with/without VM_ENABLE
   - Ensure identical functionality

3. **Prepare TLB_wrapper:**
   - Create TLB_wrapper module matching demo_module interface
   - Add VPN→PPN translation logic
   - Integrate PTW (Page Table Walker)

4. **Replace demo_module with TLB_wrapper:**
   - Update instantiation in VX_cache_cluster.sv
   - Add TLB-specific configuration parameters
   - Connect PTW memory interface

---

## Revision History
- **v1.0** (2025-01-XX): Initial integration of demo_module in VX_cache_cluster
- Focus: Preparatory step for TLB wrapper integration
- Status: Ready for testing and TLB development


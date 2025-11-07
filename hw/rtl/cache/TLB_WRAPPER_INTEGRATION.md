# VX_tlb_wrapper Integration in VX_cache_cluster

## Overview
This document describes the integration of `VX_tlb_wrapper` instances into `VX_cache_cluster.sv` to provide a virtual memory translation layer between the core request arbiter and cache instances.

## Integration Point
The TLB wrapper is inserted **between the core arbiter and cache instances** when `VM_ENABLE` is defined:

```
Core Requests → Core Arbiter → [TLB Wrapper] → Cache Instance → Memory
                                 ↑
                           (VM_ENABLE only)
```

## Architecture

### Without VM_ENABLE
```
arb_core_bus_if[NUM_CACHES * NUM_REQS]
         ↓
    VX_cache_wrap instances
```

### With VM_ENABLE
```
arb_core_bus_if[NUM_CACHES * NUM_REQS]
         ↓
    VX_tlb_wrapper instances (NUM_CACHES * NUM_REQS)
         ↓
tlb_core_bus_if[NUM_CACHES * NUM_REQS]
         ↓
    VX_cache_wrap instances
```

## Implementation Details

### 1. Intermediate Bus Interface
When `VM_ENABLE` is defined, an intermediate `VX_mem_bus_if` array is created:

```systemverilog
`ifdef VM_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (ARB_TAG_WIDTH)
    ) tlb_core_bus_if[NUM_CACHES * NUM_REQS]();
`endif
```

- **Array Size**: `NUM_CACHES * NUM_REQS` (one per cache unit, per request channel)
- **DATA_SIZE**: `WORD_SIZE` (typically 16 bytes)
- **TAG_WIDTH**: `ARB_TAG_WIDTH` (includes arbiter selection bits)

### 2. TLB Wrapper Instantiation
One `VX_tlb_wrapper` instance is created for each cache unit and each request channel:

```systemverilog
for (genvar i = 0; i < NUM_CACHES; ++i) begin : g_tlb_wrappers
    for (genvar j = 0; j < NUM_REQS; ++j) begin : g_tlb_per_req
        VX_tlb_wrapper #(
            .INSTANCE_ID     (`SFORMATF(("%s-tlb%0d", INSTANCE_ID, i))),
            .BANK_ID         (j),
            .ADDR_WIDTH      (TLB_ADDR_WIDTH),
            .WSEL_WIDTH      (1),
            .BYTEEN_WIDTH    (WORD_SIZE),
            .DATA_WIDTH      (TLB_DATA_WIDTH),
            .TAG_WIDTH       (ARB_TAG_WIDTH),
            .IDX_WIDTH       (1),
            .FLAGS_WIDTH     (`UP(FLAGS_WIDTH)),
            .MEM_PORTS       (1),
            .MEM_ARB_SEL_WIDTH (1)
        ) tlb_wrapper_inst (
            // Connections...
        );
    end
end
```

### 3. Parameter Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `INSTANCE_ID` | `"{INSTANCE_ID}-tlb{i}"` | Unique identifier for debugging |
| `BANK_ID` | `j` (0 to NUM_REQS-1) | Request channel identifier |
| `ADDR_WIDTH` | `` `CS_WORD_ADDR_WIDTH`` | Word address width from cache defines |
| `WSEL_WIDTH` | `1` | Word select (not used at cluster level) |
| `BYTEEN_WIDTH` | `WORD_SIZE` | Byte enable width (16 bytes typical) |
| `DATA_WIDTH` | `WORD_SIZE * 8` | Data width in bits (128 bits typical) |
| `TAG_WIDTH` | `ARB_TAG_WIDTH` | Request tag including arbiter bits |
| `IDX_WIDTH` | `1` | Index width (not used at cluster level) |
| `FLAGS_WIDTH` | `` `UP(FLAGS_WIDTH)`` | Request flags width (rounded up) |
| `MEM_PORTS` | `1` | Placeholder for future PTW |
| `MEM_ARB_SEL_WIDTH` | `1` | Placeholder for future PTW |

### 4. Request Path Connections

#### Input (from arbiter)
```systemverilog
// Input from arbiter
.in_valid        (arb_core_bus_if[i * NUM_REQS + j].req_valid),
.in_addr         (arb_core_bus_if[i * NUM_REQS + j].req_data.addr),
.in_rw           (arb_core_bus_if[i * NUM_REQS + j].req_data.rw),
.in_wsel         (1'b0),        // Not used at cluster level
.in_byteen       (arb_core_bus_if[i * NUM_REQS + j].req_data.byteen),
.in_data         (arb_core_bus_if[i * NUM_REQS + j].req_data.data),
.in_tag          (arb_core_bus_if[i * NUM_REQS + j].req_data.tag),
.in_idx          (1'b0),        // Not used at cluster level
.in_flags        (`UP(FLAGS_WIDTH)'(arb_core_bus_if[i * NUM_REQS + j].req_data.flags)),
.in_ready        (arb_core_bus_if[i * NUM_REQS + j].req_ready),
```

#### Output (to cache)
```systemverilog
// Output to cache
.out_valid       (tlb_core_bus_if[i * NUM_REQS + j].req_valid),
.out_addr        (tlb_core_bus_if[i * NUM_REQS + j].req_data.addr),
.out_rw          (tlb_core_bus_if[i * NUM_REQS + j].req_data.rw),
.out_wsel        (),            // Unused
.out_byteen      (tlb_core_bus_if[i * NUM_REQS + j].req_data.byteen),
.out_data        (tlb_core_bus_if[i * NUM_REQS + j].req_data.data),
.out_tag         (tlb_core_bus_if[i * NUM_REQS + j].req_data.tag),
.out_idx         (),            // Unused
.out_flags       (tlb_core_bus_if[i * NUM_REQS + j].req_data.flags),
.out_ready       (tlb_core_bus_if[i * NUM_REQS + j].req_ready),
```

### 5. Response Path (Pass-through)
Since `VX_tlb_wrapper` currently acts as a pass-through for responses, the response path is directly assigned:

```systemverilog
// Pass-through response path (VX_tlb_wrapper doesn't modify responses)
assign tlb_core_bus_if[i * NUM_REQS + j].rsp_valid = arb_core_bus_if[i * NUM_REQS + j].rsp_valid;
assign tlb_core_bus_if[i * NUM_REQS + j].rsp_data  = arb_core_bus_if[i * NUM_REQS + j].rsp_data;
assign arb_core_bus_if[i * NUM_REQS + j].rsp_ready  = tlb_core_bus_if[i * NUM_REQS + j].rsp_ready;
```

### 6. Future PTW Support (Placeholder)
The memory request and arbiter monitor ports are currently tied off as they will be used for future Page Table Walk (PTW) functionality:

```systemverilog
// Memory request (unused currently - for future PTW)
.mem_req_valid   (),
.mem_req_addr    (),
.mem_req_rw      (),
.mem_req_byteen  (),
.mem_req_data    (),
.mem_req_tag     (),
.mem_req_flags   (),
.mem_req_ready   (1'b0),

// Arbiter monitor (for future PTW coordination)
.arb_mem_req_valid   (1'b0),
.arb_mem_req_ready   (1'b0),
.arb_mem_req_sel_out (1'b0)
```

### 7. Conditional Connection to Cache
The `VX_cache_wrap` instances conditionally connect to either the TLB wrapper output or directly to the arbiter output:

```systemverilog
for (genvar i = 0; i < NUM_CACHES; ++i) begin : g_cache_wrap
    VX_cache_wrap #(
        // Parameters...
    ) cache_wrap (
        // Other connections...
    `ifdef VM_ENABLE
        // When VM_ENABLE is defined, connect to TLB wrapper output
        .core_bus_if (tlb_core_bus_if[i * NUM_REQS +: NUM_REQS]),
    `else 
        // When VM_ENABLE is not defined, connect directly to arbiter output
        .core_bus_if (arb_core_bus_if[i * NUM_REQS +: NUM_REQS]),
    `endif
        // Other connections...
    );
end
```

## Data Flow

### Request Flow (VM_ENABLE defined)
1. **Core Arbiter** (`VX_mem_arb`) arbitrates requests from multiple cores/inputs
2. **Arbiter Output** → `arb_core_bus_if[NUM_CACHES * NUM_REQS]`
3. **TLB Wrapper** receives request from `arb_core_bus_if`
   - Currently performs 1-cycle elastic buffering
   - Future: TLB lookup and address translation
4. **TLB Output** → `tlb_core_bus_if[NUM_CACHES * NUM_REQS]`
5. **Cache Wrapper** receives translated request from `tlb_core_bus_if`

### Response Flow (VM_ENABLE defined)
1. **Cache Wrapper** generates response → `tlb_core_bus_if.rsp_*`
2. **Direct Assignment** to `arb_core_bus_if.rsp_*` (no TLB involvement)
3. **Core Arbiter** routes response back to original requester

## Current Behavior (Pass-through Mode)

The `VX_tlb_wrapper` currently operates in **pass-through mode**:
- **Address Translation**: None (addresses passed through unchanged)
- **Latency**: 1 cycle (elastic buffer)
- **Throughput**: 1 request per cycle when flowing
- **Backpressure**: Properly propagated from cache to arbiter
- **PTW**: Not implemented (mem_req_* outputs tied to 0)

## Future Enhancements

### Phase 1: Basic TLB Functionality
- Add TLB lookup logic (tag array, valid bits)
- Implement hit/miss detection
- Perform address translation on TLB hit

### Phase 2: Page Table Walk (PTW)
- Generate memory requests via `mem_req_*` ports on TLB miss
- Integrate with memory arbiter via `arb_mem_req_*` monitor ports
- Implement multi-level page table traversal
- Update TLB on PTW completion

### Phase 3: Advanced Features
- Multi-cycle TLB lookup support
- Outstanding request tracking
- TLB flush support
- ASID (Address Space ID) support
- Different page sizes (4KB, 2MB, 1GB)

## Instance Count

For a typical configuration:
- `NUM_CACHES = 2` (e.g., L1D cache cluster)
- `NUM_REQS = 4` (4 request channels per cache)
- **Total TLB Wrappers**: 2 × 4 = **8 instances**

Each instance is independent and operates on its assigned request channel.

## Compile Flags

- **Enable**: Define `VM_ENABLE` during compilation
- **Disable**: Do not define `VM_ENABLE` (TLB wrapper bypassed entirely)

## Testing Considerations

1. **Functional Equivalence**: With `VM_ENABLE` defined, cache behavior should be identical to without it (since TLB is pass-through)
2. **Latency**: Expect 1 additional cycle latency due to elastic buffer
3. **Backpressure**: Verify ready/valid handshaking works correctly
4. **Resource Usage**: Each TLB wrapper adds minimal logic (just registers for buffering)

## Related Files

- **Module Definition**: `VX_tlb_wrapper.sv`
- **Integration Point**: `VX_cache_cluster.sv`
- **Module Documentation**: `tlb_wrapper_structure.md`
- **TLB Analysis**: `TLB_FILES_ANALYSIS.md`
- **Integration Steps**: `TLB_INTEGRATION_STEP1.md`

## Changelog

### 2025-10-22: Initial Integration
- Replaced `demo_module` with `VX_tlb_wrapper` in `VX_cache_cluster.sv`
- Created intermediate bus interface `tlb_core_bus_if`
- Instantiated one TLB wrapper per cache unit, per request channel
- Configured TLB wrappers in pass-through mode
- All unused PTW ports tied off
- Verified syntax and linter checks pass

---

**Status**: ✅ Integration Complete  
**Mode**: Pass-through (no translation)  
**Next Steps**: Implement actual TLB lookup and page table walk logic


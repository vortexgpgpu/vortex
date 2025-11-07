# VX_tlb_wrapper Structure and Interface

## Module Position in Cache Hierarchy

```
VX_cache.sv:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Core Requests [NUM_REQS]                                   │
│          ↓                                                   │
│  ┌──────────────────┐                                       │
│  │ Request Crossbar │ (VX_stream_xbar)                      │
│  │   (NUM_REQS →    │                                       │
│  │    NUM_BANKS)    │                                       │
│  └──────────────────┘                                       │
│          ↓                                                   │
│  per_bank_core_req_*[NUM_BANKS]                             │
│          ↓                                                   │
│  ┌─────────────────────────────────────┐  ← VM_ENABLE only │
│  │  VX_tlb_wrapper [NUM_BANKS]         │                    │
│  │  ┌─────────────────────────────┐    │                    │
│  │  │ Bank 0 TLB Wrapper          │    │                    │
│  │  │  • 1-entry elastic buffer   │    │                    │
│  │  │  • Pass-through (for now)   │────┼──→ tlb_mem_req_*  │
│  │  │  • Future: TLB lookup       │    │    (to mem arb)   │
│  │  └─────────────────────────────┘    │                    │
│  │  │ Bank 1 TLB Wrapper │ ...         │                    │
│  └─────────────────────────────────────┘                    │
│          ↓                                                   │
│  per_bank_core_req_*_d[NUM_BANKS]                           │
│          ↓                                                   │
│  ┌──────────────────┐                                       │
│  │  Cache Banks     │ (VX_cache_bank × NUM_BANKS)           │
│  │  [NUM_BANKS]     │                                       │
│  └──────────────────┘                                       │
│          ↓                                                   │
│  per_bank_mem_req_*[NUM_BANKS]                              │
│          ↓                                                   │
│  ┌──────────────────────────────────┐                       │
│  │ Memory Request Arbiter           │                       │
│  │ Inputs:                          │                       │
│  │  • tlb_mem_req_*[NUM_BANKS]      │← VM_ENABLE only      │
│  │  • per_bank_mem_req_*[NUM_BANKS] │                       │
│  │ Total: NUM_BANKS×2 when VM_ENABLE│                       │
│  └──────────────────────────────────┘                       │
│          ↓                                                   │
│  Memory Ports [MEM_PORTS]                                   │
└─────────────────────────────────────────────────────────────┘
```

## VX_tlb_wrapper Interface

### Input Ports (from Request Crossbar)
```systemverilog
input  wire                         in_valid      // Request valid
input  wire [ADDR_WIDTH-1:0]        in_addr       // Virtual/physical address (line addr)
input  wire                         in_rw         // Read=0, Write=1
input  wire [WSEL_WIDTH-1:0]        in_wsel       // Word select within line
input  wire [BYTEEN_WIDTH-1:0]      in_byteen     // Byte enables
input  wire [DATA_WIDTH-1:0]        in_data       // Write data
input  wire [TAG_WIDTH-1:0]         in_tag        // Request tag (ID)
input  wire [IDX_WIDTH-1:0]         in_idx        // Request index (which of NUM_REQS)
input  wire [FLAGS_WIDTH-1:0]       in_flags      // Request flags
output wire                         in_ready      // Ready to accept
```
- **Width Notes**:
  - `ADDR_WIDTH` = `CS_LINE_ADDR_WIDTH` (word addr - bank - word select bits)
  - `DATA_WIDTH` = `CS_WORD_WIDTH` (typically 128 bits for WORD_SIZE=16)
  - `TAG_WIDTH` = UUID + other tag bits (for matching responses)

### Output Ports (to Cache Bank)
```systemverilog
output wire                         out_valid     // Request valid (after TLB)
output wire [ADDR_WIDTH-1:0]        out_addr      // Translated address (future)
output wire                         out_rw        // Read/write (pass-through)
output wire [WSEL_WIDTH-1:0]        out_wsel      // Word select (pass-through)
output wire [BYTEEN_WIDTH-1:0]      out_byteen    // Byte enables (pass-through)
output wire [DATA_WIDTH-1:0]        out_data      // Write data (pass-through)
output wire [TAG_WIDTH-1:0]         out_tag       // Request tag (pass-through)
output wire [IDX_WIDTH-1:0]         out_idx       // Request index (pass-through)
output wire [FLAGS_WIDTH-1:0]       out_flags     // Request flags (pass-through)
input  wire                         out_ready     // Bank ready
```

### Memory Request Output (for Page Table Walks)
```systemverilog
output wire                         mem_req_valid   // PTW request valid
output wire [ADDR_WIDTH-1:0]        mem_req_addr    // PT entry address
output wire                         mem_req_rw      // Always read for PTW
output wire [BYTEEN_WIDTH-1:0]      mem_req_byteen  // Byte enables
output wire [DATA_WIDTH-1:0]        mem_req_data    // Data (unused for PTW)
output wire [TAG_WIDTH-1:0]         mem_req_tag     // PTW request tag
output wire [FLAGS_WIDTH-1:0]       mem_req_flags   // Request flags
input  wire                         mem_req_ready   // Mem arbiter ready
```
- **Current**: All outputs tied to `1'b0` (no PTW yet)
- **Future**: Generate read requests to fetch page table entries

### Monitor/Tap Inputs (from Memory Arbiter)
```systemverilog
input  wire [MEM_PORTS-1:0]                         arb_mem_req_valid
input  wire [MEM_PORTS-1:0]                         arb_mem_req_ready
input  wire [MEM_PORTS-1:0][MEM_ARB_SEL_WIDTH-1:0]  arb_mem_req_sel_out
```
- **Purpose**: Visibility into memory arbiter for future PTW coordination
- **Current**: Unused (marked with `UNUSED_VAR`)

## Current Internal Logic (Pass-through)

```systemverilog
// 1-entry elastic buffer
reg r_valid;
reg [ADDR_WIDTH-1:0] r_addr;
reg r_rw;
// ... (all other fields)

// Flow control
wire accept_in = in_valid && in_ready;
wire send_out  = out_valid && out_ready;

assign in_ready  = ~r_valid || send_out;  // Ready when empty or draining
assign out_valid = r_valid;               // Valid when buffer full

// Pass-through assignments
assign out_addr   = r_addr;   // NO translation yet
assign out_rw     = r_rw;     // Pass through
// ... (all fields pass through)

// State update
always @(posedge clk) begin
    if (reset) 
        r_valid <= 1'b0;
    else if (accept_in)
        {r_valid, r_addr, r_rw, ...} <= {1'b1, in_addr, in_rw, ...};
    else if (send_out)
        r_valid <= 1'b0;
end
```

### Timing Behavior
- **Latency**: 1 cycle minimum (elastic buffer)
- **Throughput**: 1 request/cycle when flowing
- **Backpressure**: Propagates from bank to crossbar

## Parameters

| Parameter          | Typical Value         | Purpose                          |
|--------------------|-----------------------|----------------------------------|
| INSTANCE_ID        | "cache"               | Debug string                     |
| BANK_ID            | 0..NUM_BANKS-1        | Bank identifier                  |
| ADDR_WIDTH         | ~28 bits              | Line address width               |
| WSEL_WIDTH         | 2-3 bits              | Word select (4-8 words/line)     |
| BYTEEN_WIDTH       | 16 bytes (WORD_SIZE)  | Byte enable width                |
| DATA_WIDTH         | 128 bits              | Word data width                  |
| TAG_WIDTH          | 32+ bits              | Request tag width                |
| IDX_WIDTH          | 2-3 bits              | Request index (NUM_REQS)         |
| FLAGS_WIDTH        | 4-8 bits              | Request flags                    |
| MEM_PORTS          | 1-2                   | Number of memory ports           |
| MEM_ARB_SEL_WIDTH  | depends on MEM_PORTS  | Arbiter select width             |

## Future TLB Integration Points

### 1. Address Translation (on TLB hit)
```systemverilog
// Replace:
assign out_addr = r_addr;

// With:
wire tlb_hit;
wire [ADDR_WIDTH-1:0] translated_addr;
// TLB lookup logic here
assign out_addr = tlb_hit ? translated_addr : r_addr;
```

### 2. TLB Miss Handling
```systemverilog
// On TLB miss:
// 1. Stall the output (out_valid = 0)
// 2. Generate PTW request:
assign mem_req_valid = tlb_miss;
assign mem_req_addr  = page_table_base + vpn_index;
assign mem_req_rw    = 1'b0; // Always read

// 3. Wait for PTW response
// 4. Update TLB
// 5. Retry request
```

### 3. Multi-cycle TLB Lookup
May need to extend elastic buffer or add state machine for:
- Multi-level page tables (L1, L2, L3)
- TLB refill from memory
- Outstanding request tracking

## Debug/Tracing
```systemverilog
`ifdef DBG_TRACE_CACHE
    // Traces each accepted request with:
    // - Instance ID, Bank ID
    // - Address, RW flag, Tag
    `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: addr=0x%0h, rw=%b, tag=0x%0h\n",
        $time, INSTANCE_ID, BANK_ID, in_addr, in_rw, in_tag))
`endif
```

## Compatibility
- ✅ Drop-in replacement for `demo_module`
- ✅ Same interface, same latency
- ✅ Same memory arbiter integration
- ✅ No functional changes to cache operation


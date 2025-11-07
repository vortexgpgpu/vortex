// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Functional TLB wrapper with address translation capability
// - TLB hit: Translates virtual to physical address
// - TLB miss: Forwards request to memory (treated as next cache level miss)
// - No PTW integration (misses handled like cache misses)

`include "VX_cache_define.vh"
`include "VX_define.vh"

module VX_tlb_wrapper import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BANK_ID             = 0,
    
    parameter ADDR_WIDTH          = 1,
    parameter WSEL_WIDTH          = 1,
    parameter BYTEEN_WIDTH        = 1,
    parameter DATA_WIDTH          = 1,
    parameter TAG_WIDTH           = 1,
    parameter IDX_WIDTH           = 1,
    parameter FLAGS_WIDTH         = 1,
    parameter MEM_PORTS           = 1,
    parameter MEM_ARB_SEL_WIDTH   = 1,
    
    // TLB-specific parameters
    parameter TLB_ENTRIES         = 32,     // Total TLB entries
    parameter TLB_WAYS            = 4,      // Associativity
    parameter TLB_REPL_POLICY     = 0       // 0=FIFO, 1=LRU
) (
    input  wire                         clk,
    input  wire                         reset,

    // Input from request crossbar
    input  wire                         in_valid,
    input  wire [ADDR_WIDTH-1:0]        in_addr,
    input  wire                         in_rw,
    input  wire [WSEL_WIDTH-1:0]        in_wsel,
    input  wire [BYTEEN_WIDTH-1:0]      in_byteen,
    input  wire [DATA_WIDTH-1:0]        in_data,
    input  wire [TAG_WIDTH-1:0]         in_tag,
    input  wire [IDX_WIDTH-1:0]         in_idx,
    input  wire [FLAGS_WIDTH-1:0]       in_flags,
    output wire                         in_ready,

    // Output to bank/cache
    output wire                         out_valid,
    output wire [ADDR_WIDTH-1:0]        out_addr,
    output wire                         out_rw,
    output wire [WSEL_WIDTH-1:0]        out_wsel,
    output wire [BYTEEN_WIDTH-1:0]      out_byteen,
    output wire [DATA_WIDTH-1:0]        out_data,
    output wire [TAG_WIDTH-1:0]         out_tag,
    output wire [IDX_WIDTH-1:0]         out_idx,
    output wire [FLAGS_WIDTH-1:0]       out_flags,
    input  wire                         out_ready,

    // Memory arbiter outputs (monitor/tap - for future PTW use)
    input  wire [MEM_PORTS-1:0]                         arb_mem_req_valid,
    input  wire [MEM_PORTS-1:0]                         arb_mem_req_ready,
    input  wire [MEM_PORTS-1:0][MEM_ARB_SEL_WIDTH-1:0]  arb_mem_req_sel_out,

    // Memory request output (for future page table walks)
    output wire                         mem_req_valid,
    output wire [ADDR_WIDTH-1:0]        mem_req_addr,
    output wire                         mem_req_rw,
    output wire [BYTEEN_WIDTH-1:0]      mem_req_byteen,
    output wire [DATA_WIDTH-1:0]        mem_req_data,
    output wire [TAG_WIDTH-1:0]         mem_req_tag,
    output wire [FLAGS_WIDTH-1:0]       mem_req_flags,
    input  wire                         mem_req_ready
);

    `UNUSED_PARAM (arb_mem_req_valid)
    `UNUSED_PARAM (arb_mem_req_ready)
    `UNUSED_PARAM (arb_mem_req_sel_out)
    `UNUSED_PARAM (mem_req_ready)

    ///////////////////////////////////////////////////////////////////////////
    // TLB Configuration
    ///////////////////////////////////////////////////////////////////////////
    
    localparam PAGE_OFFSET_BITS = 12;  // 4KB pages
    localparam VPN_WIDTH = ADDR_WIDTH - PAGE_OFFSET_BITS;  // Virtual Page Number width
    localparam PPN_WIDTH = ADDR_WIDTH - PAGE_OFFSET_BITS;  // Physical Page Number width
    
    localparam TLB_SETS = TLB_ENTRIES / TLB_WAYS;
    localparam TLB_INDEX_BITS = `CLOG2(TLB_SETS);
    localparam TLB_INDEX_WIDTH = `UP(TLB_INDEX_BITS);
    localparam TLB_WAY_BITS = `CLOG2(TLB_WAYS);
    localparam TLB_WAY_WIDTH = `UP(TLB_WAY_BITS);
    
    ///////////////////////////////////////////////////////////////////////////
    // TLB Storage Arrays
    ///////////////////////////////////////////////////////////////////////////
    
    // Tag array: stores VPN tags
    reg [TLB_WAYS-1:0][VPN_WIDTH-1:0] tlb_tag_array [TLB_SETS];
    
    // Data array: stores PPN (physical page numbers)
    reg [TLB_WAYS-1:0][PPN_WIDTH-1:0] tlb_data_array [TLB_SETS];
    
    // Valid bits
    reg [TLB_WAYS-1:0] tlb_valid_array [TLB_SETS];
    
    // Replacement policy state (FIFO counter or LRU bits)
    reg [TLB_WAY_WIDTH-1:0] tlb_repl_state [TLB_SETS];
    
    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Stage 1: TLB Lookup
    ///////////////////////////////////////////////////////////////////////////
    
    // Extract VPN and offset from input address
    wire [VPN_WIDTH-1:0] in_vpn = in_addr[ADDR_WIDTH-1:PAGE_OFFSET_BITS];
    wire [PAGE_OFFSET_BITS-1:0] in_offset = in_addr[PAGE_OFFSET_BITS-1:0];
    
    // TLB index from VPN
    wire [TLB_INDEX_WIDTH-1:0] tlb_index = in_vpn[TLB_INDEX_BITS-1:0];
    
    // Tag matching logic
    wire [TLB_WAYS-1:0] way_matches;
    wire [PPN_WIDTH-1:0] way_ppns [TLB_WAYS];
    
    genvar i;
    generate
        for (i = 0; i < TLB_WAYS; i = i + 1) begin : g_tlb_match
            assign way_matches[i] = tlb_valid_array[tlb_index][i] && 
                                   (tlb_tag_array[tlb_index][i] == in_vpn);
            assign way_ppns[i] = tlb_data_array[tlb_index][i];
        end
    endgenerate
    
    // Hit detection
    wire tlb_hit = |way_matches;
    wire tlb_miss = in_valid && ~tlb_hit;
    
    // Select PPN from hitting way
    wire [TLB_WAY_WIDTH-1:0] hit_way;
    wire [PPN_WIDTH-1:0] hit_ppn;
    
    VX_find_first #(
        .N       (TLB_WAYS),
        .REVERSE (0)
    ) hit_way_selector (
        .valid_in   (way_matches),
        .index_out  (hit_way),
        `UNUSED_PIN (found)
    );
    
    assign hit_ppn = way_ppns[hit_way];
    
    // Translated physical address (on hit) or original address (on miss)
    wire [ADDR_WIDTH-1:0] translated_addr = tlb_hit ? {hit_ppn, in_offset} : in_addr;
    
    ///////////////////////////////////////////////////////////////////////////
    // Pipeline Stage 2: Output Buffer with Translation
    ///////////////////////////////////////////////////////////////////////////
    
    reg                                r_valid;
    reg  [ADDR_WIDTH-1:0]              r_addr;
    reg                                r_rw;
    reg  [WSEL_WIDTH-1:0]              r_wsel;
    reg  [BYTEEN_WIDTH-1:0]            r_byteen;
    reg  [DATA_WIDTH-1:0]              r_data;
    reg  [TAG_WIDTH-1:0]               r_tag;
    reg  [IDX_WIDTH-1:0]               r_idx;
    reg  [FLAGS_WIDTH-1:0]             r_flags;
    reg                                r_was_miss;  // Track if this was a TLB miss
    
    wire accept_in  = in_valid && in_ready;
    wire send_out   = out_valid && out_ready;
    
    assign in_ready  = ~r_valid || send_out;
    assign out_valid = r_valid;
    
    // Output assignments (translated address on hit, original on miss)
    assign out_addr   = r_addr;
    assign out_rw     = r_rw;
    assign out_wsel   = r_wsel;
    assign out_byteen = r_byteen;
    assign out_data   = r_data;
    assign out_tag    = r_tag;
    assign out_idx    = r_idx;
    assign out_flags  = r_flags;
    
    // No memory requests generated (PTW not implemented)
    assign mem_req_valid  = 1'b0;
    assign mem_req_addr   = {ADDR_WIDTH{1'b0}};
    assign mem_req_rw     = 1'b0;
    assign mem_req_byteen = {BYTEEN_WIDTH{1'b0}};
    assign mem_req_data   = {DATA_WIDTH{1'b0}};
    assign mem_req_tag    = {TAG_WIDTH{1'b0}};
    assign mem_req_flags  = {FLAGS_WIDTH{1'b0}};
    
    // Pipeline register update
    always @(posedge clk) begin
        if (reset) begin
            r_valid <= 1'b0;
            r_was_miss <= 1'b0;
        end else begin
            if (accept_in) begin
                r_valid    <= 1'b1;
                r_addr     <= translated_addr;  // Use translated address
                r_rw       <= in_rw;
                r_wsel     <= in_wsel;
                r_byteen   <= in_byteen;
                r_data     <= in_data;
                r_tag      <= in_tag;
                r_idx      <= in_idx;
                r_flags    <= in_flags;
                r_was_miss <= tlb_miss;
            end else if (send_out) begin
                r_valid <= 1'b0;
            end
        end
    end
    
    ///////////////////////////////////////////////////////////////////////////
    // TLB Update Logic (Simplified - No PTW)
    ///////////////////////////////////////////////////////////////////////////
    
    // For now, TLB is populated statically or through external initialization
    // In a full implementation, this would be driven by PTW responses
    
    // Replacement way selection (FIFO)
    wire [TLB_WAY_WIDTH-1:0] repl_way = tlb_repl_state[tlb_index];
    
    // Example: Initialize TLB with identity mapping (VA == PA) on reset
    // This is just for testing - real TLB would be populated by PTW
    integer j, k;
    initial begin
        for (j = 0; j < TLB_SETS; j = j + 1) begin
            for (k = 0; k < TLB_WAYS; k = k + 1) begin
                tlb_valid_array[j][k] = 1'b0;  // All entries invalid initially
                tlb_tag_array[j][k] = {VPN_WIDTH{1'b0}};
                tlb_data_array[j][k] = {PPN_WIDTH{1'b0}};
            end
            tlb_repl_state[j] = {TLB_WAY_WIDTH{1'b0}};
        end
    end
    
    // Reset logic
    always @(posedge clk) begin
        if (reset) begin
            for (j = 0; j < TLB_SETS; j = j + 1) begin
                for (k = 0; k < TLB_WAYS; k = k + 1) begin
                    tlb_valid_array[j][k] <= 1'b0;
                end
                tlb_repl_state[j] <= {TLB_WAY_WIDTH{1'b0}};
            end
        end
    end
    
    // Replacement policy update on access (simplified FIFO)
    always @(posedge clk) begin
        if (accept_in && !reset) begin
            if (tlb_miss) begin
                // On miss, update replacement state (FIFO increment)
                if (TLB_WAYS > 1) begin
                    tlb_repl_state[tlb_index] <= (tlb_repl_state[tlb_index] + 1'b1) % TLB_WAYS;
                end
            end
        end
    end
    
`ifdef DBG_TRACE_CACHE
    always @(posedge clk) begin
        if (accept_in) begin
            if (tlb_hit) begin
                `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: HIT - VA=0x%0h -> PA=0x%0h, rw=%b, tag=0x%0h\n",
                    $time, INSTANCE_ID, BANK_ID, in_addr, translated_addr, in_rw, in_tag))
            end else begin
                `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: MISS - VA=0x%0h (forwarded), rw=%b, tag=0x%0h\n",
                    $time, INSTANCE_ID, BANK_ID, in_addr, in_rw, in_tag))
            end
        end
    end
`endif

`ifdef PERF_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_tlb_hits;
    reg [`PERF_CTR_BITS-1:0] perf_tlb_misses;
    
    always @(posedge clk) begin
        if (reset) begin
            perf_tlb_hits   <= '0;
            perf_tlb_misses <= '0;
        end else if (accept_in) begin
            if (tlb_hit) begin
                perf_tlb_hits <= perf_tlb_hits + 1'b1;
            end else begin
                perf_tlb_misses <= perf_tlb_misses + 1'b1;
            end
        end
    end
`endif

endmodule


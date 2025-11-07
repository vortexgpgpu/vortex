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

// Functional TLB wrapper that instantiates VX_tlb_bank
// Provides address translation with TLB lookup
// TLB misses are forwarded to cache (treated as cache misses)

`include "VX_cache_define.vh"
`include "VX_tlb_define.vh"

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
    parameter TLB_ENTRIES         = 32,
    parameter TLB_WAYS            = 4,
    parameter TLB_REPL_POLICY     = `CS_REPL_FIFO
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

    `UNUSED_VAR (arb_mem_req_valid)
    `UNUSED_VAR (arb_mem_req_ready)
    `UNUSED_VAR (arb_mem_req_sel_out)
    `UNUSED_VAR (mem_req_ready)

    ///////////////////////////////////////////////////////////////////////////
    // TLB Bank Instantiation
    ///////////////////////////////////////////////////////////////////////////
    
    // TLB bank interface signals
    wire                    tlb_req_valid;
    wire [`XLEN-1:0]        tlb_req_addr;
    wire                    tlb_req_rw;
    wire [TAG_WIDTH-1:0]    tlb_req_tag;
    wire                    tlb_req_ready;
    
    wire                    tlb_rsp_valid;
    wire [`XLEN-1:0]        tlb_rsp_addr;    // Translated address
    wire                    tlb_rsp_rw;
    wire [TAG_WIDTH-1:0]    tlb_rsp_tag;
    wire                    tlb_rsp_ready;
    
    wire                    tlb_miss_valid;
    wire [`XLEN-13:0]       tlb_miss_vpn;
    wire                    tlb_miss_ready;
    
    wire                    tlb_update_valid;
    wire [`XLEN-13:0]       tlb_update_vpn;
    wire [`XLEN-13:0]       tlb_update_ppn;
    wire                    tlb_update_ready;
    
    // Input buffering - convert to TLB bank interface
    // For ADDR_WIDTH < XLEN, zero-extend
    localparam ADDR_PAD_BITS = `XLEN - ADDR_WIDTH;
    
    assign tlb_req_valid = in_valid;
    assign tlb_req_addr  = (ADDR_PAD_BITS > 0) ? {{ADDR_PAD_BITS{1'b0}}, in_addr} : in_addr[`XLEN-1:0];
    assign tlb_req_rw    = in_rw;
    assign tlb_req_tag   = in_tag;
    assign in_ready      = tlb_req_ready;
    
    // Instantiate TLB bank
    VX_tlb_bank #(
        .INSTANCE_ID      (INSTANCE_ID),
        .BANK_ID          (BANK_ID),
        .TLB_ENTRIES      (TLB_ENTRIES),
        .TLB_WAYS         (TLB_WAYS),
        .NUM_BANKS        (1),              // Single bank in wrapper
        .WORD_SIZE        (BYTEEN_WIDTH),
        .TLB_REPL_POLICY  (TLB_REPL_POLICY),
        .UUID_WIDTH       (0),
        .TAG_WIDTH        (TAG_WIDTH)
    ) tlb_bank_inst (
        .clk              (clk),
        .reset            (reset),
        
        // Core Request
        .core_req_valid   (tlb_req_valid),
        .core_req_addr    (tlb_req_addr),
        .core_req_rw      (tlb_req_rw),
        .core_req_tag     (tlb_req_tag),
        .core_req_ready   (tlb_req_ready),
        
        // Core Response (with translated address)
        .core_rsp_valid   (tlb_rsp_valid),
        .core_rsp_addr    (tlb_rsp_addr),
        .core_rsp_rw      (tlb_rsp_rw),
        .core_rsp_tag     (tlb_rsp_tag),
        .core_rsp_ready   (tlb_rsp_ready),
        
        // TLB miss interface (tie off - no PTW)
        .tlb_miss_valid   (tlb_miss_valid),
        .tlb_miss_vpn     (tlb_miss_vpn),
        .tlb_miss_ready   (1'b0),           // No PTW support
        
        // TLB update interface (tie off - no PTW)
        .tlb_update_valid (1'b0),           // No PTW support
        .tlb_update_vpn   ('0),
        .tlb_update_ppn   ('0),
        .tlb_update_ready (tlb_update_ready)
    );
    
    ///////////////////////////////////////////////////////////////////////////
    // Output Buffering and Field Pass-through
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
    
    wire accept_rsp  = tlb_rsp_valid && tlb_rsp_ready;
    wire send_out    = out_valid && out_ready;
    
    assign tlb_rsp_ready = ~r_valid || send_out;
    assign out_valid     = r_valid;
    
    // Output assignments
    assign out_addr   = r_addr;
    assign out_rw     = r_rw;
    assign out_wsel   = r_wsel;
    assign out_byteen = r_byteen;
    assign out_data   = r_data;
    assign out_tag    = r_tag;
    assign out_idx    = r_idx;
    assign out_flags  = r_flags;
    
    // Pipeline register - buffer TLB response and pass through other fields
    always @(posedge clk) begin
        if (reset) begin
            r_valid <= 1'b0;
        end else begin
            if (accept_rsp) begin
                r_valid    <= 1'b1;
                // Extract translated address (truncate if needed)
                r_addr     <= tlb_rsp_addr[ADDR_WIDTH-1:0];
                r_rw       <= tlb_rsp_rw;
                r_wsel     <= in_wsel;          // Pass through
                r_byteen   <= in_byteen;        // Pass through
                r_data     <= in_data;          // Pass through
                r_tag      <= tlb_rsp_tag;
                r_idx      <= in_idx;           // Pass through
                r_flags    <= in_flags;         // Pass through
            end else if (send_out) begin
                r_valid <= 1'b0;
            end
        end
    end
    
    // No memory requests generated (PTW not implemented)
    assign mem_req_valid  = 1'b0;
    assign mem_req_addr   = {ADDR_WIDTH{1'b0}};
    assign mem_req_rw     = 1'b0;
    assign mem_req_byteen = {BYTEEN_WIDTH{1'b0}};
    assign mem_req_data   = {DATA_WIDTH{1'b0}};
    assign mem_req_tag    = {TAG_WIDTH{1'b0}};
    assign mem_req_flags  = {FLAGS_WIDTH{1'b0}};
    
`ifdef DBG_TRACE_CACHE
    always @(posedge clk) begin
        if (tlb_req_valid && tlb_req_ready) begin
            `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: REQ addr=0x%0h, rw=%b, tag=0x%0h\n",
                $time, INSTANCE_ID, BANK_ID, tlb_req_addr, tlb_req_rw, tlb_req_tag))
        end
        if (tlb_rsp_valid && tlb_rsp_ready) begin
            `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: RSP addr=0x%0h (translated), rw=%b, tag=0x%0h\n",
                $time, INSTANCE_ID, BANK_ID, tlb_rsp_addr, tlb_rsp_rw, tlb_rsp_tag))
        end
        if (tlb_miss_valid) begin
            `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: MISS vpn=0x%0h\n",
                $time, INSTANCE_ID, BANK_ID, tlb_miss_vpn))
        end
    end
`endif

endmodule


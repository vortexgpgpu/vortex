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

// TLB wrapper stage between cache request crossbar and banks
// Currently acts as pass-through with single-entry elastic buffering.
// Future: Will integrate TLB lookup and page table walk on misses.

`include "VX_cache_define.vh"

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
    parameter MEM_ARB_SEL_WIDTH   = 1
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

    // Output to bank
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

    `UNUSED_PARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (arb_mem_req_valid)
    `UNUSED_VAR (arb_mem_req_ready)
    `UNUSED_VAR (arb_mem_req_sel_out)
    `UNUSED_VAR (mem_req_ready)

    ///////////////////////////////////////////////////////////////////////////
    // Pass-through stage with single-entry elastic buffer
    // TODO: Replace with actual TLB lookup logic
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

    wire accept_in  = in_valid && in_ready;
    wire send_out   = out_valid && out_ready;

    assign in_ready  = ~r_valid || send_out;
    assign out_valid = r_valid;

    // Pass-through: no address translation yet
    assign out_addr   = r_addr;
    assign out_rw     = r_rw;
    assign out_wsel   = r_wsel;
    assign out_byteen = r_byteen;
    assign out_data   = r_data;
    assign out_tag    = r_tag;
    assign out_idx    = r_idx;
    assign out_flags  = r_flags;

    // No memory requests generated yet (will be used for page table walks)
    assign mem_req_valid  = 1'b0;
    assign mem_req_addr   = {ADDR_WIDTH{1'b0}};
    assign mem_req_rw     = 1'b0;
    assign mem_req_byteen = {BYTEEN_WIDTH{1'b0}};
    assign mem_req_data   = {DATA_WIDTH{1'b0}};
    assign mem_req_tag    = {TAG_WIDTH{1'b0}};
    assign mem_req_flags  = {FLAGS_WIDTH{1'b0}};

    always @(posedge clk) begin
        if (reset) begin
            r_valid <= 1'b0;
        end else begin
            if (accept_in) begin
                r_valid  <= 1'b1;
                r_addr   <= in_addr;
                r_rw     <= in_rw;
                r_wsel   <= in_wsel;
                r_byteen <= in_byteen;
                r_data   <= in_data;
                r_tag    <= in_tag;
                r_idx    <= in_idx;
                r_flags  <= in_flags;
            end else if (send_out) begin
                r_valid <= 1'b0;
            end
        end
    end

`ifdef DBG_TRACE_CACHE
    always @(posedge clk) begin
        if (accept_in) begin
            `TRACE(3, ("%t: %s-bank%0d tlb-wrapper: addr=0x%0h, rw=%b, tag=0x%0h\n",
                $time, INSTANCE_ID, BANK_ID, in_addr, in_rw, in_tag))
        end
    end
`endif

endmodule
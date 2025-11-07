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

`include "VX_cache_define.vh"
`include "VX_tlb_define.vh"

module VX_tlb_bank #(
    parameter `STRING INSTANCE_ID = "",
    parameter BANK_ID            = 0,

    // TLB entries per bank
    parameter TLB_ENTRIES        = 32,
    // TLB ways (associativity)
    parameter TLB_WAYS           = 4,
    // Number of banks
    parameter NUM_BANKS          = 4,

    // Size of a word in bytes
    parameter WORD_SIZE          = 16,

    // Replacement policy
    parameter TLB_REPL_POLICY    = `CS_REPL_PLRU,

    // Request debug identifier
    parameter UUID_WIDTH         = 0,

    // core request tag size
    parameter TAG_WIDTH          = UUID_WIDTH + 1
) (
    input wire clk,
    input wire reset,

    // Core Request
    input wire                          core_req_valid,
    input wire [`XLEN-1:0]              core_req_addr,    // Virtual address
    input wire                          core_req_rw,      // Read/Write signal
    input wire [TAG_WIDTH-1:0]          core_req_tag,     // Request tag for response matching
    output wire                         core_req_ready,

    // Core Response
    output wire                         core_rsp_valid,
    output wire [`XLEN-1:0]             core_rsp_addr,    // Translated physical address
    output wire                         core_rsp_rw,      // Read/Write signal (passed through)
    output wire [TAG_WIDTH-1:0]         core_rsp_tag,     // Response tag (matches request)
    input wire                          core_rsp_ready,

    // TLB miss interface (to PTW)
    output wire                         tlb_miss_valid,   // TLB miss signal
    output wire [`XLEN-13:0]            tlb_miss_vpn,    // Virtual Page Number that missed
    input wire                          tlb_miss_ready,   // PTW ready to accept miss

    // TLB update interface (from PTW)
    input wire                          tlb_update_valid, // New translation available
    input wire [`XLEN-13:0]             tlb_update_vpn,  // Virtual Page Number to update
    input wire [`XLEN-13:0]             tlb_update_ppn,  // Physical Page Number translation
    output wire                         tlb_update_ready  // TLB ready to accept update
);

    // Page size is 4KB (2^12 bytes)
    localparam PAGE_OFFSET_WIDTH = 12;
    localparam VPN_WIDTH = `XLEN - PAGE_OFFSET_WIDTH;  // Virtual Page Number width
    localparam PPN_WIDTH = `XLEN - PAGE_OFFSET_WIDTH;  // Physical Page Number width

    // TLB tag is the VPN, data is the PPN
    localparam TLB_TAG_WIDTH = VPN_WIDTH;
    localparam TLB_DATA_WIDTH = PPN_WIDTH;

    // Bank selection and request handling
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);

    // TLB bank parameters
    localparam ENTRIES_PER_BANK = TLB_ENTRIES / NUM_BANKS;
    localparam TLB_INDEX_BITS = `CLOG2(ENTRIES_PER_BANK / TLB_WAYS);
    localparam TLB_INDEX_WIDTH = `UP(TLB_INDEX_BITS);
    localparam TLB_WAY_SEL_BITS = `CLOG2(TLB_WAYS);
    localparam TLB_WAY_SEL_WIDTH = `UP(TLB_WAY_SEL_BITS);

    // Extract VPN and page offset from virtual address
    wire [VPN_WIDTH-1:0] req_vpn = core_req_addr[`XLEN-1:PAGE_OFFSET_WIDTH];
    wire [PAGE_OFFSET_WIDTH-1:0] req_offset = core_req_addr[PAGE_OFFSET_WIDTH-1:0];

    // TLB tag array
    wire [TLB_WAYS-1:0] tag_matches;
    wire [TLB_WAY_SEL_WIDTH-1:0] hit_way;
    wire [TLB_WAY_SEL_WIDTH-1:0] repl_way;
    wire [TLB_DATA_WIDTH-1:0] hit_ppn;

    // TLB lookup and hit detection
    wire tlb_hit = |tag_matches;
    wire tlb_miss = core_req_valid && ~tlb_hit;

    // TLB update handling
    wire tlb_update_fire = tlb_update_valid && tlb_update_ready;

    // Core request handling
    wire core_req_fire = core_req_valid && core_req_ready;
    wire core_rsp_fire = core_rsp_valid && core_rsp_ready;

    // Extract TLB index from VPN
    wire [TLB_INDEX_BITS-1:0] req_index = req_vpn[TLB_INDEX_BITS-1:0];

    // TLB tag array (VPN storage)
    reg [TLB_WAYS-1:0] tag_valid [ENTRIES_PER_BANK / TLB_WAYS];
    reg [TLB_WAYS-1:0][TLB_TAG_WIDTH-1:0] tag_data [ENTRIES_PER_BANK / TLB_WAYS];

    // TLB data array (PPN + flags storage)
    reg [TLB_WAYS-1:0][TLB_DATA_WIDTH-1:0] data_ppn [ENTRIES_PER_BANK / TLB_WAYS];  // Physical Page Number
    reg [TLB_WAYS-1:0][7:0] data_flags [ENTRIES_PER_BANK / TLB_WAYS];  // Access control flags
    reg [TLB_WAYS-1:0] data_mru [ENTRIES_PER_BANK / TLB_WAYS];  // Most Recently Used bit
    reg [TLB_WAYS-1:0][5:0] data_size [ENTRIES_PER_BANK / TLB_WAYS];  // Page size in bits (log2)

    // Tag matching - combinational logic
    for (genvar i = 0; i < TLB_WAYS; ++i) begin : g_tlb_ways
        // Tag matching happens in the same cycle as the request
        assign tag_matches[i] = tag_valid[req_index][i] && (tag_data[req_index][i] == req_vpn);
    end

    // TLB update handling using pipeline register
    wire update_valid;
    wire [TLB_INDEX_BITS-1:0] update_index;
    wire [TLB_WAY_SEL_WIDTH-1:0] update_way;
    wire [TLB_TAG_WIDTH-1:0] update_vpn;
    wire [TLB_DATA_WIDTH-1:0] update_ppn;
    wire [7:0] update_flags;
    wire update_mru;
    wire [5:0] update_size;

    VX_pipe_register #(
        .DATAW  (1 + TLB_INDEX_BITS + TLB_WAY_SEL_WIDTH + TLB_TAG_WIDTH + TLB_DATA_WIDTH + 8 + 1 + 6),
        .RESETW (1)
    ) update_pipe (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({tlb_update_fire, req_index, repl_way, tlb_update_vpn, 
                   tlb_update_ppn, 8'hFF, 1'b1, 6'd12}), // Default flags: all permissions, MRU=1, 4KB page
        .data_out ({update_valid, update_index, update_way, update_vpn, 
                   update_ppn, update_flags, update_mru, update_size})
    );

    // Update tag and data arrays (sequential logic!)
    always @(posedge clk) begin
        if (reset) begin
            // Arrays will be cleared by reset_pipe mechanism
        end else if (update_valid) begin
            tag_valid[update_index][update_way] <= 1'b1;
            tag_data[update_index][update_way] <= update_vpn;
            data_ppn[update_index][update_way] <= update_ppn;
            data_flags[update_index][update_way] <= update_flags;
            data_mru[update_index][update_way] <= update_mru;
            data_size[update_index][update_way] <= update_size;
        end
    end

    // Reset handling using pipeline register
    wire reset_valid;
    wire [TLB_INDEX_BITS-1:0] reset_index;

    VX_pipe_register #(
        .DATAW  (1 + TLB_INDEX_BITS),
        .RESETW (1)
    ) reset_pipe (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({reset, req_index}),
        .data_out ({reset_valid, reset_index})
    );

    // Clear valid bits on reset (sequential logic!)
    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < (ENTRIES_PER_BANK / TLB_WAYS); i = i + 1) begin
                tag_valid[i] <= '0;
            end
        end else if (reset_valid) begin
            tag_valid[reset_index] <= '0;
        end
    end

    // PPN Mux - selects the correct PPN based on which way had a hit
    VX_mux #(
        .DATAW (TLB_DATA_WIDTH),
        .N     (TLB_WAYS)
    ) ppn_mux (
        .data_in  (data_ppn[req_index]),  // Array of PPNs from all ways at this index
        .sel_in   (hit_way),              // Which way had a hit (from one-hot encoder)
        .data_out (hit_ppn)               // Selected PPN for translation
    );

    // PLRU replacement policy
    VX_cache_repl #(
        .CACHE_SIZE (ENTRIES_PER_BANK),
        .LINE_SIZE  (1),  // Not used for TLB
        .NUM_BANKS  (1),  // Not used for TLB
        .NUM_WAYS   (TLB_WAYS),
        .REPL_POLICY(TLB_REPL_POLICY)
    ) tlb_repl (
        .clk          (clk),
        .reset        (reset),
        .stall        (1'b0),
        .init         (reset),
        .lookup_valid (core_req_fire),
        .lookup_hit   (tlb_hit),
        .lookup_line  (req_index),
        .lookup_way   (hit_way),
        .repl_valid   (tlb_update_fire),
        .repl_line    (req_index),
        .repl_way     (repl_way)
    );

    // One-hot to binary encoder for hit way
    VX_onehot_encoder #(
        .N (TLB_WAYS)
    ) way_idx_enc (
        .data_in  (tag_matches),
        .data_out (hit_way),
        `UNUSED_PIN (valid_out)
    );

    // Response queue
    wire rsp_queue_valid;
    wire rsp_queue_ready;
    wire [`XLEN-1:0] rsp_queue_addr;
    wire rsp_queue_rw;
    wire [TAG_WIDTH-1:0] rsp_queue_tag;

    VX_elastic_buffer #(
        .DATAW   (TAG_WIDTH + 1 + `XLEN),
        .SIZE    (4),  // Small FIFO for responses
        .OUT_REG (1)
    ) rsp_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (core_req_fire && tlb_hit),
        .ready_in  (rsp_queue_ready),
        .data_in   ({core_req_tag, core_req_rw, {hit_ppn, req_offset}}),
        .data_out  ({core_rsp_tag, core_rsp_rw, core_rsp_addr}),
        .valid_out (core_rsp_valid),
        .ready_out (core_rsp_ready)
    );

    // TLB miss handling
    assign tlb_miss_valid = tlb_miss;
    assign tlb_miss_vpn = req_vpn;
    assign tlb_update_ready = ~core_req_valid;  // Ready when no lookup in progress

    // Core request ready when:
    // 1. Response queue has space
    // 2. No TLB update in progress
    assign core_req_ready = rsp_queue_ready && ~tlb_update_valid;

    // Performance counters
`ifdef PERF_ENABLE
    wire perf_tlb_hit;
    wire perf_tlb_miss;
    wire perf_tlb_stall;
    assign perf_tlb_hit = core_req_fire && tlb_hit;
    assign perf_tlb_miss = core_req_fire && ~tlb_hit;
    assign perf_tlb_stall = core_req_valid && ~core_req_ready;
`endif

endmodule

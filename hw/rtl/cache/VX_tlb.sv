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

module VX_tlb import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID   = "",

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // TLB entries per bank
    parameter TLB_ENTRIES           = 32,
    // TLB ways (associativity)
    parameter TLB_WAYS              = 4,
    // Number of banks (must match cache)
    parameter NUM_BANKS             = 4,

    // Size of a word in bytes
    parameter WORD_SIZE             = 16,

    // Size of line in bytes
    parameter LINE_SIZE             = 64,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 4,

    // Replacement policy
    parameter TLB_REPL_POLICY       = `CS_REPL_FIFO,

    // core request tag size
    parameter TAG_WIDTH             = 1,

    // core request flags
    parameter FLAGS_WIDTH           = 0
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    output tlb_perf_t     tlb_perf,
`endif

    // Dhruv Q: what is this master slave, and do we need it if not accessing memory?
    /* Core request interface */
    input wire [NUM_REQS-1:0] core_req_valid,
    input wire [NUM_REQS-1:0][`XLEN-1:0] core_req_addr,
    input wire [NUM_REQS-1:0] core_req_rw,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0] core_req_byteen,
    input wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_req_data,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0] core_req_tag,
    input wire [NUM_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] core_req_flags,
    output wire [NUM_REQS-1:0] core_req_ready,

    /* TLB miss interface (for future PTW integration)  */
    // Dhruv Q: multiple TLB misses will be handled sequentially by the PTW, how to do so?
    // Another arbiter?
    // 1. output to PTW, indicates hit or miss
    output wire [NUM_BANKS-1:0] tlb_miss_o,
    // 2. input from PTW, indicates ready to accept miss request
    input wire [NUM_BANKS-1:0] tlb_miss_ready_i,
    // 3. output to PTW, the Virtual Page Number to be translated
    output wire [NUM_BANKS-1:0][VPN_WIDTH-1:0] tlb_miss_vpn_o,
    // 4. output to PTW, indicates TLB is ready to accept miss request
    output wire [NUM_BANKS-1:0] tlb_miss_valid_o,

    /* TLB update interface (for future PTW integration) */
    // 1. input from PTW, indicates PTW is ready with translation
    input wire [NUM_BANKS-1:0] tlb_update_valid_i,
    // 2. input from PTW, virtual page number to be translated
    input wire [NUM_BANKS-1:0][VPN_WIDTH-1:0] tlb_update_vpn_i,
    // 3. input from PTW, physical page number translation
    input wire [NUM_BANKS-1:0][PPN_WIDTH-1:0] tlb_update_ppn_i,
    // 4. output to PTW, indicates TLB is ready to accept update
    output wire [NUM_BANKS-1:0] tlb_update_ready_o,

    /* Translated address output to cache */
    output wire [NUM_REQS-1:0] core_rsp_valid,
    output wire [NUM_REQS-1:0][`XLEN-1:0] core_rsp_addr,
    output wire [NUM_REQS-1:0] core_rsp_rw,
    output wire [NUM_REQS-1:0][WORD_SIZE-1:0] core_rsp_byteen,
    output wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0] core_rsp_tag,
    input wire [NUM_REQS-1:0] core_rsp_ready
);

    // Page size is 4KB (2^12 bytes)
    localparam PAGE_OFFSET_WIDTH = 12;
    localparam VPN_WIDTH = `XLEN - PAGE_OFFSET_WIDTH;  // Virtual Page Number width
    localparam PPN_WIDTH = `XLEN - PAGE_OFFSET_WIDTH;  // Physical Page Number width

    // TLB tag is the VPN, data is the PPN
    localparam TLB_TAG_WIDTH = VPN_WIDTH;
    localparam TLB_DATA_WIDTH = PPN_WIDTH;

    // Bank selection and request handling
    localparam REQ_SEL_WIDTH   = `UP(`CS_REQ_SEL_BITS);
    localparam WORD_SEL_WIDTH  = `UP(`CS_WORD_SEL_BITS);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);
    
    // Data widths for crossbar
    localparam CORE_REQ_DATAW = VPN_WIDTH + PAGE_OFFSET_WIDTH + 1 + TAG_WIDTH;
    localparam CORE_RSP_DATAW = PPN_WIDTH + PAGE_OFFSET_WIDTH + 1 + TAG_WIDTH;

    `STATIC_ASSERT(NUM_BANKS == (1 << `CLOG2(NUM_BANKS)), ("invalid parameter: number of banks must be power of 2"))
    `STATIC_ASSERT(TLB_ENTRIES == (1 << `CLOG2(TLB_ENTRIES)), ("invalid parameter: number of TLB entries must be power of 2"))


    /* Per-bank request signals */
    wire [NUM_BANKS-1:0] per_bank_req_valid;
    wire [NUM_BANKS-1:0][VPN_WIDTH-1:0] per_bank_req_vpn;      // Virtual Page Number for lookup
    wire [NUM_BANKS-1:0][PAGE_OFFSET_WIDTH-1:0] per_bank_req_offset;  // Page offset (passed through)
    wire [NUM_BANKS-1:0] per_bank_req_rw;                      // Read/Write signal (passed through)
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0] per_bank_req_tag;     // Request tag for response matching
    wire [NUM_BANKS-1:0] per_bank_req_ready;                  // Ready signal from bank

    /* Per-bank response signals */
    wire [NUM_BANKS-1:0] per_bank_rsp_valid;
    wire [NUM_BANKS-1:0][PPN_WIDTH-1:0] per_bank_rsp_ppn;      // Translated Physical Page Number
    wire [NUM_BANKS-1:0][PAGE_OFFSET_WIDTH-1:0] per_bank_rsp_offset;  // Page offset (passed through)
    wire [NUM_BANKS-1:0] per_bank_rsp_rw;                      // Read/Write signal (passed through)
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0] per_bank_rsp_tag;     // Response tag (matches request)
    wire [NUM_BANKS-1:0] per_bank_rsp_ready;                  // Ready signal for response
    
    // Request index from crossbar
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_req_idx;

    // Performance counters
`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_tlb_hit_per_bank;
    wire [NUM_BANKS-1:0] perf_tlb_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_tlb_stall_per_bank;
`endif

// TLB bank instances
    for (genvar bank_id = 0; bank_id < NUM_BANKS; ++bank_id) begin : g_tlb_banks
        VX_tlb_bank #(
            .INSTANCE_ID  (`SFORMATF(("%s-bank%0d", INSTANCE_ID, bank_id))),
            .BANK_ID      (bank_id),
            .TLB_ENTRIES  (TLB_ENTRIES),
            .TLB_WAYS     (TLB_WAYS),
            .NUM_BANKS    (NUM_BANKS),
            .WORD_SIZE    (WORD_SIZE),
            .TLB_REPL_POLICY (TLB_REPL_POLICY),
            .UUID_WIDTH   (UUID_WIDTH),
            .TAG_WIDTH    (TAG_WIDTH)
        ) tlb_bank (
            .clk            (clk),
            .reset          (reset),

            // Core Request
            .core_req_valid (per_bank_req_valid[bank_id]),
            .core_req_addr  (per_bank_req_vpn[bank_id]),
            .core_req_rw    (per_bank_req_rw[bank_id]),
            .core_req_tag   (per_bank_req_tag[bank_id]),
            .core_req_ready (per_bank_req_ready[bank_id]),

            // Core Response
            .core_rsp_valid (per_bank_rsp_valid[bank_id]),
            .core_rsp_addr  (per_bank_rsp_ppn[bank_id]),
            .core_rsp_rw    (per_bank_rsp_rw[bank_id]),
            .core_rsp_tag   (per_bank_rsp_tag[bank_id]),
            .core_rsp_ready (per_bank_rsp_ready[bank_id]),

            // TLB miss interface
            .tlb_miss_valid (tlb_miss_o[bank_id]),
            .tlb_miss_vpn   (tlb_miss_vpn_o[bank_id]),
            .tlb_miss_ready (tlb_miss_ready_i[bank_id]),

            // TLB update interface
            .tlb_update_valid (tlb_update_valid_i[bank_id]),
            .tlb_update_vpn   (tlb_update_vpn_i[bank_id]),
            .tlb_update_ppn   (tlb_update_ppn_i[bank_id]),
            .tlb_update_ready (tlb_update_ready_o[bank_id])
        );
    end

    // Request distribution to banks
    wire [NUM_REQS-1:0][CORE_REQ_DATAW-1:0] core_req_data_in;
    wire [NUM_BANKS-1:0][CORE_REQ_DATAW-1:0] core_req_data_out;

    // Bank selection from addresses
    wire [NUM_REQS-1:0][BANK_SEL_WIDTH-1:0] req_bank_sel;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_bank_sel
        if (BANK_SEL_BITS > 0) begin : g_bank_sel_bits
            assign req_bank_sel[i] = core_req_addr[i][BANK_SEL_BITS-1:0];
        end else begin : g_no_bank_sel
            assign req_bank_sel[i] = '0;
        end
    end

    // Core request data packing
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_data_in
        assign core_req_data_in[i] = {
            core_req_addr[i][`XLEN-1:PAGE_OFFSET_WIDTH],  // VPN
            core_req_addr[i][PAGE_OFFSET_WIDTH-1:0],      // Page offset
            core_req_rw[i],
            core_req_tag[i]
        };
    end

    // Request crossbar
    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (CORE_REQ_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .valid_in  (core_req_valid),
        .data_in   (core_req_data_in),
        .sel_in    (req_bank_sel),  // Bank selection based on address
        .ready_in  (core_req_ready),
        .valid_out (per_bank_req_valid),
        .data_out  (core_req_data_out),
        .sel_out   (per_bank_req_idx),
        .ready_out (per_bank_req_ready)
    );

    // Unpack request data for each bank
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_req_data_out
        assign {
            per_bank_req_vpn[i],
            per_bank_req_offset[i],
            per_bank_req_rw[i],
            per_bank_req_tag[i]
        } = core_req_data_out[i];
    end

    // PTW interface (placeholder until PTW is implemented)
    // tlb_miss_valid_o indicates we have a miss and want to start PTW
    assign tlb_miss_valid_o = tlb_miss_o & tlb_miss_ready_i;

    // Response gathering
    wire [NUM_BANKS-1:0][CORE_RSP_DATAW-1:0] core_rsp_data_in;
    wire [NUM_REQS-1:0][CORE_RSP_DATAW-1:0] core_rsp_data_out;

    // Pack response data from banks
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_rsp_data_in
        // TODO: Implement proper offset passthrough via buffer/queue
        // For now, pass through request offset (works for single cycle TLB hits)
        assign per_bank_rsp_offset[i] = per_bank_req_offset[i];
        
        assign core_rsp_data_in[i] = {
            per_bank_rsp_ppn[i],
            per_bank_rsp_offset[i],
            per_bank_rsp_rw[i],
            per_bank_rsp_tag[i]
        };
    end

    // Response crossbar
    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (CORE_RSP_DATAW),
        .ARBITER     ("R")
    ) rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .valid_in  (per_bank_rsp_valid),
        .data_in   (core_rsp_data_in),
        .sel_in    (per_bank_req_idx),
        .ready_in  (per_bank_rsp_ready),
        .valid_out (core_rsp_valid),
        .data_out  (core_rsp_data_out),
        .ready_out (core_rsp_ready),
        `UNUSED_PIN (sel_out)
    );

    // Unpack response data for core
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_rsp_data_out
        wire [PPN_WIDTH-1:0] rsp_ppn;
        wire [PAGE_OFFSET_WIDTH-1:0] rsp_offset;
        wire rsp_rw;
        wire [TAG_WIDTH-1:0] rsp_tag;

        assign {
            rsp_ppn,
            rsp_offset,
            rsp_rw,
            rsp_tag
        } = core_rsp_data_out[i];

        assign core_rsp_addr[i] = {rsp_ppn, rsp_offset};
        assign core_rsp_rw[i] = rsp_rw;
        assign core_rsp_tag[i] = rsp_tag;
    end
    
    // Pass through byteen and data (TLB doesn't modify these)
    assign core_rsp_byteen = core_req_byteen;
    assign core_rsp_data = core_req_data;

    // Performance counters
`ifdef PERF_ENABLE
    wire [`PERF_CTR_BITS-1:0] perf_tlb_hits;
    wire [`PERF_CTR_BITS-1:0] perf_tlb_misses;
    wire [`PERF_CTR_BITS-1:0] perf_tlb_stalls;

    `POP_COUNT(perf_tlb_hits, perf_tlb_hit_per_bank);
    `POP_COUNT(perf_tlb_misses, perf_tlb_miss_per_bank);
    `POP_COUNT(perf_tlb_stalls, perf_tlb_stall_per_bank);

    reg [`PERF_CTR_BITS-1:0] perf_tlb_hits_r;
    reg [`PERF_CTR_BITS-1:0] perf_tlb_misses_r;
    reg [`PERF_CTR_BITS-1:0] perf_tlb_stalls_r;

    always @(posedge clk) begin
        if (reset) begin
            perf_tlb_hits_r <= '0;
            perf_tlb_misses_r <= '0;
            perf_tlb_stalls_r <= '0;
        end else begin
            perf_tlb_hits_r <= perf_tlb_hits_r + perf_tlb_hits;
            perf_tlb_misses_r <= perf_tlb_misses_r + perf_tlb_misses;
            perf_tlb_stalls_r <= perf_tlb_stalls_r + perf_tlb_stalls;
        end
    end

    assign tlb_perf.hits = perf_tlb_hits_r;
    assign tlb_perf.misses = perf_tlb_misses_r;
    assign tlb_perf.stalls = perf_tlb_stalls_r;
`endif

endmodule

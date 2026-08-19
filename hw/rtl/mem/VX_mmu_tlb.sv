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

`include "VX_define.vh"

// Banked TLB: NUM_REQS lanes are spread over NUM_BANKS banks by the low VPN
// bits, so up to NUM_BANKS translations proceed per cycle and a miss parked
// in one bank does not stall hits in the others. Every bank owns one
// outstanding walk on the shared PTW bus, identified by its bank index.
module VX_mmu_tlb import VX_gpu_pkg::*; #(
    parameter NUM_REQS      = DCACHE_NUM_REQS,
    parameter NUM_BANKS     = 1,
    parameter NUM_ENTRIES   = `VX_CFG_TLB_SIZE,
    parameter DATA_SIZE     = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH_IN  = DCACHE_TAG_WIDTH_BASE,
    parameter TAG_WIDTH_OUT = TAG_WIDTH_IN + `UP(`CLOG2(NUM_REQS)),
    parameter ADDR_WIDTH    = DCACHE_ADDR_WIDTH,
    parameter ATTR_WIDTH    = MEM_ATTR_WIDTH,
    parameter PTW_TAG_WIDTH = PTW_TLB_TAG_WIDTH
) (
    input wire clk,
    input wire reset,
    input wire flush,

    input wire [VM_PPN_WIDTH-1:0] root_ppn,

    VX_mem_bus_if.slave  tlb_in_if [NUM_REQS],
    VX_mem_bus_if.master tlb_out_if [NUM_REQS],

    VX_ptw_bus_if.master ptw_bus_if,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf
`else
    output wire          mmu_perf_placeholder
`endif
);
    `STATIC_ASSERT(`IS_POW2(NUM_BANKS), ("NUM_BANKS must be a power of 2"))
    `STATIC_ASSERT((NUM_ENTRIES % NUM_BANKS) == 0, ("NUM_ENTRIES must be a multiple of NUM_BANKS"))
    `STATIC_ASSERT(`CLOG2(NUM_BANKS) <= PTW_TAG_WIDTH, ("bank index does not fit the PTW tag"))

    localparam DATA_WIDTH    = DATA_SIZE * 8;
    localparam SOURCE_BITS   = `UP(`CLOG2(NUM_REQS));
    localparam REQ_DATAW_IN  = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + ATTR_WIDTH + TAG_WIDTH_IN;
    localparam REQ_DATAW_OUT = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + ATTR_WIDTH + TAG_WIDTH_OUT;
    localparam BANK_SIZE     = NUM_ENTRIES / NUM_BANKS;
    localparam BANK_SEL_BITS = `CLOG2(NUM_BANKS);
    localparam BANK_BITS     = `UP(BANK_SEL_BITS);
    localparam PAGE_OFFSET_BITS = VM_PAGE_OFFSET_BITS - `CLOG2(DATA_SIZE);
    localparam MISS_DATAW    = VM_VPN_WIDTH;
    localparam FILL_DATAW    = VM_PPN_WIDTH + VM_LEVEL_BITS + VM_PTE_FLAGS_WIDTH + 1;

    // -------------------------------------------------------------------------
    // Request distribution: lane -> bank by low VPN bits
    // -------------------------------------------------------------------------

    wire [NUM_REQS-1:0]                   req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW_IN-1:0] req_data_in;
    wire [NUM_REQS-1:0]                   req_ready_in;
    wire [NUM_REQS-1:0][BANK_BITS-1:0]    req_bank_sel;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_in
        assign req_valid_in[i] = tlb_in_if[i].req_valid;
        assign req_data_in[i]  = {
            tlb_in_if[i].req_data.rw,
            tlb_in_if[i].req_data.addr,
            tlb_in_if[i].req_data.data,
            tlb_in_if[i].req_data.byteen,
            tlb_in_if[i].req_data.attr[ATTR_WIDTH-1:0],
            tlb_in_if[i].req_data.tag[TAG_WIDTH_IN-1:0]
        };
        assign tlb_in_if[i].req_ready = req_ready_in[i];
        if (NUM_BANKS > 1) begin : g_bank_sel
            assign req_bank_sel[i] = tlb_in_if[i].req_data.addr[PAGE_OFFSET_BITS +: BANK_SEL_BITS];
        end else begin : g_single_bank
            assign req_bank_sel[i] = '0;
        end
    end

    wire [NUM_BANKS-1:0]                   bank_req_valid;
    wire [NUM_BANKS-1:0][REQ_DATAW_IN-1:0] bank_req_data;
    wire [NUM_BANKS-1:0][SOURCE_BITS-1:0]  bank_req_sel;
    wire [NUM_BANKS-1:0]                   bank_req_ready;

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (REQ_DATAW_IN),
        .ARBITER     ("R"),
        .OUT_BUF     (0)
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .sel_in    (req_bank_sel),
        .ready_in  (req_ready_in),
        .valid_out (bank_req_valid),
        .data_out  (bank_req_data),
        .sel_out   (bank_req_sel),
        .ready_out (bank_req_ready)
    );

    // -------------------------------------------------------------------------
    // Banks
    // -------------------------------------------------------------------------

    wire [NUM_BANKS-1:0]                    bank_out_valid;
    wire [NUM_BANKS-1:0][REQ_DATAW_OUT-1:0] bank_out_data;
    wire [NUM_BANKS-1:0]                    bank_out_ready;

    wire [NUM_BANKS-1:0]                  bank_miss_valid;
    wire [NUM_BANKS-1:0][MISS_DATAW-1:0]  bank_miss_data;
    wire [NUM_BANKS-1:0]                  bank_miss_ready;
    wire [NUM_BANKS-1:0]                  bank_fill_valid;
    wire [NUM_BANKS-1:0][FILL_DATAW-1:0]  bank_fill_data;
    wire [NUM_BANKS-1:0]                  bank_fill_ready;

`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0][PERF_CTR_BITS-1:0] bank_perf_reads, bank_perf_hits, bank_perf_misses, bank_perf_evictions;
`endif

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_banks
        VX_mmu_tlb_bank #(
            .NUM_ENTRIES   (BANK_SIZE),
            .DATA_SIZE     (DATA_SIZE),
            .ADDR_WIDTH    (ADDR_WIDTH),
            .TAG_WIDTH_IN  (TAG_WIDTH_IN),
            .SOURCE_BITS   (SOURCE_BITS),
            .ATTR_WIDTH    (ATTR_WIDTH),
            .TAG_WIDTH_OUT (TAG_WIDTH_OUT)
        ) bank (
            .clk        (clk),
            .reset      (reset),
            .flush      (flush),
            .req_valid  (bank_req_valid[b]),
            .req_data   (bank_req_data[b]),
            .req_sel    (bank_req_sel[b]),
            .req_ready  (bank_req_ready[b]),
            .out_valid  (bank_out_valid[b]),
            .out_data   (bank_out_data[b]),
            .out_ready  (bank_out_ready[b]),
            .miss_valid (bank_miss_valid[b]),
            .miss_ready (bank_miss_ready[b]),
            .miss_vpn   (bank_miss_data[b]),
            .fill_valid (bank_fill_valid[b]),
            .fill_ready (bank_fill_ready[b]),
            .fill_ppn   (bank_fill_data[b][FILL_DATAW-1 -: VM_PPN_WIDTH]),
            .fill_level (bank_fill_data[b][1 + VM_PTE_FLAGS_WIDTH +: VM_LEVEL_BITS]),
            .fill_flags (bank_fill_data[b][1 +: VM_PTE_FLAGS_WIDTH]),
            .fill_fault (bank_fill_data[b][0])
        `ifdef PERF_ENABLE
            ,.perf_tlb_reads     (bank_perf_reads[b])
            ,.perf_tlb_hits      (bank_perf_hits[b])
            ,.perf_tlb_misses    (bank_perf_misses[b])
            ,.perf_tlb_evictions (bank_perf_evictions[b])
        `else
            ,`UNUSED_PIN (perf_placeholder)
        `endif
        );
    end

    // -------------------------------------------------------------------------
    // Output gather: bank -> originating lane (skid buffers decouple the
    // banks' same-cycle hit path from the gather arbiter)
    // -------------------------------------------------------------------------

    wire [NUM_BANKS-1:0]                    bank_buf_valid;
    wire [NUM_BANKS-1:0][REQ_DATAW_OUT-1:0] bank_buf_data;
    wire [NUM_BANKS-1:0]                    bank_buf_ready;
    wire [NUM_BANKS-1:0][SOURCE_BITS-1:0]   bank_buf_sel;

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_bank_out_buf
        VX_elastic_buffer #(
            .DATAW   (REQ_DATAW_OUT),
            .SIZE    (2),
            .OUT_REG (0)
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (bank_out_valid[b]),
            .data_in   (bank_out_data[b]),
            .ready_in  (bank_out_ready[b]),
            .valid_out (bank_buf_valid[b]),
            .data_out  (bank_buf_data[b]),
            .ready_out (bank_buf_ready[b])
        );
        assign bank_buf_sel[b] = bank_buf_data[b][SOURCE_BITS-1:0];
    end

    wire [NUM_REQS-1:0]                    out_valid;
    wire [NUM_REQS-1:0][REQ_DATAW_OUT-1:0] out_data;
    wire [NUM_REQS-1:0]                    out_ready;

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (REQ_DATAW_OUT),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) out_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .valid_in  (bank_buf_valid),
        .data_in   (bank_buf_data),
        .sel_in    (bank_buf_sel),
        .ready_in  (bank_buf_ready),
        .valid_out (out_valid),
        .data_out  (out_data),
        `UNUSED_PIN (sel_out),
        .ready_out (out_ready)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_out
        assign tlb_out_if[i].req_valid = out_valid[i];
        assign {
            tlb_out_if[i].req_data.rw,
            tlb_out_if[i].req_data.addr,
            tlb_out_if[i].req_data.data,
            tlb_out_if[i].req_data.byteen,
            tlb_out_if[i].req_data.attr,
            tlb_out_if[i].req_data.tag
        } = out_data[i];
        assign out_ready[i] = tlb_out_if[i].req_ready;
    end

    // Responses return on the lane that issued the request; only the lane
    // bits folded into the tag need stripping.
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_rsp
        assign tlb_in_if[i].rsp_valid     = tlb_out_if[i].rsp_valid;
        assign tlb_in_if[i].rsp_data.data = tlb_out_if[i].rsp_data.data;
        assign tlb_in_if[i].rsp_data.tag  = tlb_out_if[i].rsp_data.tag[TAG_WIDTH_OUT-1:SOURCE_BITS];
        assign tlb_out_if[i].rsp_ready    = tlb_in_if[i].rsp_ready;
    end

    // -------------------------------------------------------------------------
    // Walker bus: bank misses arbitrated onto one request stream, fills
    // routed back by the bank index carried in the tag
    // -------------------------------------------------------------------------

    wire                  miss_valid;
    wire [MISS_DATAW-1:0] miss_data;
    wire [BANK_BITS-1:0]  miss_bank;
    wire                  miss_ready;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (1),
        .DATAW       (MISS_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (0)
    ) miss_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bank_miss_valid),
        .ready_in  (bank_miss_ready),
        .data_in   (bank_miss_data),
        .data_out  (miss_data),
        .sel_out   (miss_bank),
        .valid_out (miss_valid),
        .ready_out (miss_ready)
    );

    assign ptw_bus_if.req_valid         = miss_valid;
    assign ptw_bus_if.req_data.vpn      = miss_data;
    assign ptw_bus_if.req_data.root_ppn = root_ppn;
    assign ptw_bus_if.req_data.tag      = PTW_TAG_WIDTH'(miss_bank);
    assign miss_ready = ptw_bus_if.req_ready;

    wire [BANK_BITS-1:0]  fill_bank = BANK_BITS'(ptw_bus_if.rsp_data.tag);
    wire [FILL_DATAW-1:0] fill_data = {
        ptw_bus_if.rsp_data.ppn,
        ptw_bus_if.rsp_data.level,
        ptw_bus_if.rsp_data.flags,
        ptw_bus_if.rsp_data.fault
    };

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (FILL_DATAW),
        .OUT_BUF     (0)
    ) fill_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (fill_bank),
        .valid_in  (ptw_bus_if.rsp_valid),
        .ready_in  (ptw_bus_if.rsp_ready),
        .data_in   (fill_data),
        .data_out  (bank_fill_data),
        .valid_out (bank_fill_valid),
        .ready_out (bank_fill_ready)
    );

    // -------------------------------------------------------------------------
    // Performance counters
    // -------------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] sum_reads, sum_hits, sum_misses, sum_evictions;
    always @(*) begin
        sum_reads     = '0;
        sum_hits      = '0;
        sum_misses    = '0;
        sum_evictions = '0;
        for (integer b = 0; b < NUM_BANKS; ++b) begin
            sum_reads     = sum_reads     + bank_perf_reads[b];
            sum_hits      = sum_hits      + bank_perf_hits[b];
            sum_misses    = sum_misses    + bank_perf_misses[b];
            sum_evictions = sum_evictions + bank_perf_evictions[b];
        end
    end
    assign mmu_perf.tlb_reads     = sum_reads;
    assign mmu_perf.tlb_hits      = sum_hits;
    assign mmu_perf.tlb_misses    = sum_misses;
    assign mmu_perf.tlb_evictions = sum_evictions;
`else
    assign mmu_perf_placeholder = 1'b0;
`endif

endmodule

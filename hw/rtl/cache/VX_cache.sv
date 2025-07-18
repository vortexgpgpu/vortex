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

module VX_cache import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID   = "",

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // Number of memory ports
    parameter MEM_PORTS             = 1,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 32768,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64,
    // Number of banks
    parameter NUM_BANKS             = 4,
    // Number of associative ways
    parameter NUM_WAYS              = 4,
    // Size of a word in bytes
    parameter WORD_SIZE             = 16,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 4,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 16,
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 4,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Enable cache writeback
    parameter WRITEBACK             = 0,

    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES           = 0,

    // Replacement policy
    parameter REPL_POLICY           = `CS_REPL_FIFO,

    // core request tag size
    parameter TAG_WIDTH             = UUID_WIDTH + 1,

    // Core response output register
    parameter CORE_OUT_BUF          = 3,

    // Memory request output register
    parameter MEM_OUT_BUF           = 3
 ) (
    // PERF
`ifdef PERF_ENABLE
    output cache_perf_t     cache_perf,
`endif

    input wire clk,
    input wire reset,

    VX_mem_bus_if.slave     core_bus_if [NUM_REQS],
    VX_mem_bus_if.master    mem_bus_if [MEM_PORTS]
);

    `STATIC_ASSERT(NUM_BANKS == (1 << `CLOG2(NUM_BANKS)), ("invalid parameter: number of banks must be power of 2"))
    `STATIC_ASSERT(WRITE_ENABLE || !WRITEBACK, ("invalid parameter: writeback requires write enable"))
    `STATIC_ASSERT(WRITEBACK || !DIRTY_BYTES, ("invalid parameter: dirty bytes require writeback"))
    `STATIC_ASSERT(NUM_BANKS >= MEM_PORTS, ("invalid parameter: number of banks must be greater or equal to number of memory ports"))

    localparam REQ_SEL_WIDTH   = `UP(`CS_REQ_SEL_BITS);
    localparam WORD_SEL_WIDTH  = `UP(`CS_WORD_SEL_BITS);
    localparam MSHR_ADDR_WIDTH = `LOG2UP(MSHR_SIZE);
    localparam MEM_TAG_WIDTH   = `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, MEM_PORTS, UUID_WIDTH);
    localparam WORDS_PER_LINE  = LINE_SIZE / WORD_SIZE;
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam WORD_SEL_BITS   = `CLOG2(WORDS_PER_LINE);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);
    localparam LINE_ADDR_WIDTH = (`CS_WORD_ADDR_WIDTH - BANK_SEL_BITS - WORD_SEL_BITS);
    localparam CORE_REQ_DATAW  = LINE_ADDR_WIDTH + 1 + WORD_SEL_WIDTH + WORD_SIZE + WORD_WIDTH + TAG_WIDTH + `UP(MEM_FLAGS_WIDTH);
    localparam CORE_RSP_DATAW  = WORD_WIDTH + TAG_WIDTH;
    localparam BANK_MEM_TAG_WIDTH = UUID_WIDTH + MSHR_ADDR_WIDTH;
    localparam MEM_REQ_DATAW   = (`CS_LINE_ADDR_WIDTH + 1 + LINE_SIZE + `CS_LINE_WIDTH + BANK_MEM_TAG_WIDTH + `UP(MEM_FLAGS_WIDTH));
    localparam MEM_RSP_DATAW   = `CS_LINE_WIDTH + MEM_TAG_WIDTH;
    localparam MEM_PORTS_SEL_BITS = `CLOG2(MEM_PORTS);
    localparam MEM_PORTS_SEL_WIDTH = `UP(MEM_PORTS_SEL_BITS);
    localparam MEM_ARB_SEL_BITS = `CLOG2(`CDIV(NUM_BANKS, MEM_PORTS));
    localparam MEM_ARB_SEL_WIDTH = `UP(MEM_ARB_SEL_BITS);

    localparam REQ_XBAR_BUF    = (NUM_REQS > 2) ? 2 : 0;
    localparam CORE_RSP_BUF_ENABLE = (NUM_BANKS != 1) || (NUM_REQS != 1);
    localparam MEM_REQ_BUF_ENABLE = (NUM_BANKS != 1);

`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_read_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_write_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_mshr_stall_per_bank;
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) core_bus2_if[NUM_REQS]();

    wire [NUM_BANKS-1:0] per_bank_flush_begin;
    wire [`UP(UUID_WIDTH)-1:0] flush_uuid;
    wire [NUM_BANKS-1:0] per_bank_flush_end;

    wire [NUM_BANKS-1:0] per_bank_core_req_fire;

    VX_cache_init #(
        .NUM_REQS  (NUM_REQS),
        .NUM_BANKS (NUM_BANKS),
        .TAG_WIDTH (TAG_WIDTH),
        .BANK_SEL_LATENCY (`TO_OUT_BUF_REG(REQ_XBAR_BUF)) // request xbar latency
    ) cache_init (
        .clk             (clk),
        .reset           (reset),
        .core_bus_in_if  (core_bus_if),
        .core_bus_out_if (core_bus2_if),
        .bank_req_fire   (per_bank_core_req_fire),
        .flush_begin     (per_bank_flush_begin),
        .flush_uuid      (flush_uuid),
        .flush_end       (per_bank_flush_end)
    );

    // Memory response gather /////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) mem_bus_tmp_if[MEM_PORTS]();

    wire [MEM_PORTS-1:0]                    mem_rsp_queue_valid;
    wire [MEM_PORTS-1:0][MEM_RSP_DATAW-1:0] mem_rsp_queue_data;
    wire [MEM_PORTS-1:0]                    mem_rsp_queue_ready;

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_rsp_queue
        VX_elastic_buffer #(
            .DATAW   (MEM_RSP_DATAW),
            .SIZE    (MRSQ_SIZE),
            .OUT_REG (MRSQ_SIZE > 2)
        ) mem_rsp_queue (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (mem_bus_tmp_if[i].rsp_valid),
            .data_in    (mem_bus_tmp_if[i].rsp_data),
            .ready_in   (mem_bus_tmp_if[i].rsp_ready),
            .valid_out  (mem_rsp_queue_valid[i]),
            .data_out   (mem_rsp_queue_data[i]),
            .ready_out  (mem_rsp_queue_ready[i])
        );
    end

    wire [MEM_PORTS-1:0][MEM_RSP_DATAW-MEM_ARB_SEL_BITS-1:0] mem_rsp_queue_data_s;
    wire [MEM_PORTS-1:0][BANK_SEL_WIDTH-1:0] mem_rsp_queue_sel;

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_rsp_queue_data_s
        wire [BANK_MEM_TAG_WIDTH-1:0] mem_rsp_tag_s = mem_rsp_queue_data[i][MEM_TAG_WIDTH-1:MEM_ARB_SEL_BITS];
        wire [`CS_LINE_WIDTH-1:0] mem_rsp_data_s = mem_rsp_queue_data[i][MEM_RSP_DATAW-1:MEM_TAG_WIDTH];
        assign mem_rsp_queue_data_s[i] = {mem_rsp_data_s, mem_rsp_tag_s};
    end

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_rsp_queue_sel
        if (NUM_BANKS > 1) begin : g_multibanks
            if (NUM_BANKS != MEM_PORTS) begin : g_arb_sel
                VX_bits_concat #(
                    .L (MEM_ARB_SEL_BITS),
                    .R (MEM_PORTS_SEL_BITS)
                ) mem_rsp_sel_concat (
                    .left_in  (mem_rsp_queue_data[i][MEM_ARB_SEL_BITS-1:0]),
                    .right_in (MEM_PORTS_SEL_WIDTH'(i)),
                    .data_out (mem_rsp_queue_sel[i])
                );
            end else begin : g_no_arb_sel
                assign mem_rsp_queue_sel[i] = MEM_PORTS_SEL_WIDTH'(i);
            end
        end else begin : g_singlebank
            assign mem_rsp_queue_sel[i] = 0;
        end
    end

    wire [NUM_BANKS-1:0] per_bank_mem_rsp_valid;
    wire [NUM_BANKS-1:0][MEM_RSP_DATAW-MEM_ARB_SEL_BITS-1:0] per_bank_mem_rsp_pdata;
    wire [NUM_BANKS-1:0] per_bank_mem_rsp_ready;

    VX_stream_omega #(
        .NUM_INPUTS  (MEM_PORTS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (MEM_RSP_DATAW-MEM_ARB_SEL_BITS),
        .ARBITER     ("R"),
        .OUT_BUF     (3)
    ) mem_rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_queue_valid),
        .data_in   (mem_rsp_queue_data_s),
        .sel_in    (mem_rsp_queue_sel),
        .ready_in  (mem_rsp_queue_ready),
        .valid_out (per_bank_mem_rsp_valid),
        .data_out  (per_bank_mem_rsp_pdata),
        `UNUSED_PIN (sel_out),
        .ready_out (per_bank_mem_rsp_ready),
        `UNUSED_PIN (collisions)
    );

    wire [NUM_BANKS-1:0][`CS_LINE_WIDTH-1:0] per_bank_mem_rsp_data;
    wire [NUM_BANKS-1:0][BANK_MEM_TAG_WIDTH-1:0] per_bank_mem_rsp_tag;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_mem_rsp_data
        assign {
            per_bank_mem_rsp_data[i],
            per_bank_mem_rsp_tag[i]
        } = per_bank_mem_rsp_pdata[i];
    end

    // Core requests dispatch /////////////////////////////////////////////////

    wire [NUM_BANKS-1:0]                        per_bank_core_req_valid;
    wire [NUM_BANKS-1:0][`CS_LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_rw;
    wire [NUM_BANKS-1:0][WORD_SEL_WIDTH-1:0]    per_bank_core_req_wsel;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]         per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_core_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]         per_bank_core_req_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     per_bank_core_req_idx;
    wire [NUM_BANKS-1:0][`UP(MEM_FLAGS_WIDTH)-1:0]  per_bank_core_req_flags;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_ready;

    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]         per_bank_core_rsp_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     per_bank_core_rsp_idx;
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_mem_req_valid;
    wire [NUM_BANKS-1:0][`CS_LINE_ADDR_WIDTH-1:0] per_bank_mem_req_addr;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_rw;
    wire [NUM_BANKS-1:0][LINE_SIZE-1:0]         per_bank_mem_req_byteen;
    wire [NUM_BANKS-1:0][`CS_LINE_WIDTH-1:0]    per_bank_mem_req_data;
    wire [NUM_BANKS-1:0][BANK_MEM_TAG_WIDTH-1:0] per_bank_mem_req_tag;
    wire [NUM_BANKS-1:0][`UP(MEM_FLAGS_WIDTH)-1:0]  per_bank_mem_req_flags;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_ready;

    wire [NUM_REQS-1:0]                      core_req_valid;
    wire [NUM_REQS-1:0][`CS_WORD_ADDR_WIDTH-1:0] core_req_addr;
    wire [NUM_REQS-1:0]                      core_req_rw;
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]       core_req_byteen;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0]  core_req_data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]       core_req_tag;
    wire [NUM_REQS-1:0][`UP(MEM_FLAGS_WIDTH)-1:0] core_req_flags;
    wire [NUM_REQS-1:0]                      core_req_ready;

    wire [NUM_REQS-1:0][LINE_ADDR_WIDTH-1:0] core_req_line_addr;
    wire [NUM_REQS-1:0][BANK_SEL_WIDTH-1:0]  core_req_bid;
    wire [NUM_REQS-1:0][WORD_SEL_WIDTH-1:0]  core_req_wsel;

    wire [NUM_REQS-1:0][CORE_REQ_DATAW-1:0]  core_req_data_in;
    wire [NUM_BANKS-1:0][CORE_REQ_DATAW-1:0] core_req_data_out;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req
        assign core_req_valid[i]  = core_bus2_if[i].req_valid;
        assign core_req_rw[i]     = core_bus2_if[i].req_data.rw;
        assign core_req_byteen[i] = core_bus2_if[i].req_data.byteen;
        assign core_req_addr[i]   = core_bus2_if[i].req_data.addr;
        assign core_req_data[i]   = core_bus2_if[i].req_data.data;
        assign core_req_tag[i]    = core_bus2_if[i].req_data.tag;
        assign core_req_flags[i]  = `UP(MEM_FLAGS_WIDTH)'(core_bus2_if[i].req_data.flags);
        assign core_bus2_if[i].req_ready = core_req_ready[i];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_wsel
        if (WORDS_PER_LINE > 1) begin : g_wsel
            assign core_req_wsel[i] = core_req_addr[i][0 +: WORD_SEL_BITS];
        end else begin : g_no_wsel
            assign core_req_wsel[i] = '0;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_line_addr
        assign core_req_line_addr[i] = core_req_addr[i][(BANK_SEL_BITS + WORD_SEL_BITS) +: LINE_ADDR_WIDTH];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_bid
        if (NUM_BANKS > 1) begin : g_multibanks
            assign core_req_bid[i] = core_req_addr[i][WORD_SEL_BITS +: BANK_SEL_BITS];
        end else begin : g_singlebank
            assign core_req_bid[i] = '0;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_data_in
        assign core_req_data_in[i] = {
            core_req_line_addr[i],
            core_req_rw[i],
            core_req_wsel[i],
            core_req_byteen[i],
            core_req_data[i],
            core_req_tag[i],
            core_req_flags[i]
        };
    end

    assign per_bank_core_req_fire = per_bank_core_req_valid & per_bank_mem_req_ready;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] perf_collisions;
`endif

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (CORE_REQ_DATAW),
        .PERF_CTR_BITS (PERF_CTR_BITS),
        .ARBITER     ("R"),
        .OUT_BUF     (REQ_XBAR_BUF)
    ) core_req_xbar (
        .clk       (clk),
        .reset     (reset),
    `ifdef PERF_ENABLE
        .collisions(perf_collisions),
    `else
        `UNUSED_PIN(collisions),
    `endif
        .valid_in  (core_req_valid),
        .data_in   (core_req_data_in),
        .sel_in    (core_req_bid),
        .ready_in  (core_req_ready),
        .valid_out (per_bank_core_req_valid),
        .data_out  (core_req_data_out),
        .sel_out   (per_bank_core_req_idx),
        .ready_out (per_bank_core_req_ready)
    );

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_req_data_out
        assign {
            per_bank_core_req_addr[i],
            per_bank_core_req_rw[i],
            per_bank_core_req_wsel[i],
            per_bank_core_req_byteen[i],
            per_bank_core_req_data[i],
            per_bank_core_req_tag[i],
            per_bank_core_req_flags[i]
        } = core_req_data_out[i];
    end

    // Banks access ///////////////////////////////////////////////////////////

    for (genvar bank_id = 0; bank_id < NUM_BANKS; ++bank_id) begin : g_banks
        VX_cache_bank #(
            .BANK_ID      (bank_id),
            .INSTANCE_ID  (`SFORMATF(("%s-bank%0d", INSTANCE_ID, bank_id))),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .WORD_SIZE    (WORD_SIZE),
            .NUM_REQS     (NUM_REQS),
            .WRITE_ENABLE (WRITE_ENABLE),
            .WRITEBACK    (WRITEBACK),
            .DIRTY_BYTES  (DIRTY_BYTES),
            .REPL_POLICY  (REPL_POLICY),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .TAG_WIDTH    (TAG_WIDTH),
            .CORE_OUT_REG (CORE_RSP_BUF_ENABLE ? 0 : `TO_OUT_BUF_REG(CORE_OUT_BUF)),
            .MEM_OUT_REG  (MEM_REQ_BUF_ENABLE ? 0 : `TO_OUT_BUF_REG(MEM_OUT_BUF))
        ) bank (
            .clk                (clk),
            .reset              (reset),

        `ifdef PERF_ENABLE
            .perf_read_miss    (perf_read_miss_per_bank[bank_id]),
            .perf_write_miss   (perf_write_miss_per_bank[bank_id]),
            .perf_mshr_stall   (perf_mshr_stall_per_bank[bank_id]),
        `endif

            // Core request
            .core_req_valid     (per_bank_core_req_valid[bank_id]),
            .core_req_addr      (per_bank_core_req_addr[bank_id]),
            .core_req_rw        (per_bank_core_req_rw[bank_id]),
            .core_req_wsel      (per_bank_core_req_wsel[bank_id]),
            .core_req_byteen    (per_bank_core_req_byteen[bank_id]),
            .core_req_data      (per_bank_core_req_data[bank_id]),
            .core_req_tag       (per_bank_core_req_tag[bank_id]),
            .core_req_idx       (per_bank_core_req_idx[bank_id]),
            .core_req_flags     (per_bank_core_req_flags[bank_id]),
            .core_req_ready     (per_bank_core_req_ready[bank_id]),

            // Core response
            .core_rsp_valid     (per_bank_core_rsp_valid[bank_id]),
            .core_rsp_data      (per_bank_core_rsp_data[bank_id]),
            .core_rsp_tag       (per_bank_core_rsp_tag[bank_id]),
            .core_rsp_idx       (per_bank_core_rsp_idx[bank_id]),
            .core_rsp_ready     (per_bank_core_rsp_ready[bank_id]),

            // Memory request
            .mem_req_valid      (per_bank_mem_req_valid[bank_id]),
            .mem_req_addr       (per_bank_mem_req_addr[bank_id]),
            .mem_req_rw         (per_bank_mem_req_rw[bank_id]),
            .mem_req_byteen     (per_bank_mem_req_byteen[bank_id]),
            .mem_req_data       (per_bank_mem_req_data[bank_id]),
            .mem_req_tag        (per_bank_mem_req_tag[bank_id]),
            .mem_req_flags      (per_bank_mem_req_flags[bank_id]),
            .mem_req_ready      (per_bank_mem_req_ready[bank_id]),

            // Memory response
            .mem_rsp_valid      (per_bank_mem_rsp_valid[bank_id]),
            .mem_rsp_data       (per_bank_mem_rsp_data[bank_id]),
            .mem_rsp_tag        (per_bank_mem_rsp_tag[bank_id]),
            .mem_rsp_ready      (per_bank_mem_rsp_ready[bank_id]),

            // Flush request
            .flush_begin        (per_bank_flush_begin[bank_id]),
            .flush_uuid         (flush_uuid),
            .flush_end          (per_bank_flush_end[bank_id])
        );
    end

    // Core responses gather //////////////////////////////////////////////////

    wire [NUM_BANKS-1:0][CORE_RSP_DATAW-1:0] core_rsp_data_in;
    wire [NUM_REQS-1:0][CORE_RSP_DATAW-1:0]  core_rsp_data_out;

    wire [NUM_REQS-1:0]                  core_rsp_valid_s;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_rsp_data_s;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   core_rsp_tag_s;
    wire [NUM_REQS-1:0]                  core_rsp_ready_s;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_rsp_data_in
        assign core_rsp_data_in[i] = {per_bank_core_rsp_data[i], per_bank_core_rsp_tag[i]};
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (CORE_RSP_DATAW),
        .ARBITER     ("R")
    ) core_rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .valid_in  (per_bank_core_rsp_valid),
        .data_in   (core_rsp_data_in),
        .sel_in    (per_bank_core_rsp_idx),
        .ready_in  (per_bank_core_rsp_ready),
        .valid_out (core_rsp_valid_s),
        .data_out  (core_rsp_data_out),
        .ready_out (core_rsp_ready_s),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_rsp_data_s
        assign {core_rsp_data_s[i], core_rsp_tag_s[i]} = core_rsp_data_out[i];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_rsp_buf
        VX_elastic_buffer #(
            .DATAW   (`CS_WORD_WIDTH + TAG_WIDTH),
            .SIZE    (CORE_RSP_BUF_ENABLE ? `TO_OUT_BUF_SIZE(CORE_OUT_BUF) : 0),
            .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
        ) core_rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_rsp_valid_s[i]),
            .ready_in  (core_rsp_ready_s[i]),
            .data_in   ({core_rsp_data_s[i], core_rsp_tag_s[i]}),
            .data_out  ({core_bus2_if[i].rsp_data.data, core_bus2_if[i].rsp_data.tag}),
            .valid_out (core_bus2_if[i].rsp_valid),
            .ready_out (core_bus2_if[i].rsp_ready)
        );
    end

    // Memory request arbitration /////////////////////////////////////////////

    wire [NUM_BANKS-1:0][MEM_REQ_DATAW-1:0] per_bank_mem_req_pdata;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_mem_req_pdata
        assign per_bank_mem_req_pdata[i] = {
            per_bank_mem_req_rw[i],
            per_bank_mem_req_addr[i],
            per_bank_mem_req_data[i],
            per_bank_mem_req_byteen[i],
            per_bank_mem_req_flags[i],
            per_bank_mem_req_tag[i]
        };
    end

    wire [MEM_PORTS-1:0] mem_req_valid;
    wire [MEM_PORTS-1:0][MEM_REQ_DATAW-1:0] mem_req_pdata;
    wire [MEM_PORTS-1:0] mem_req_ready;
    wire [MEM_PORTS-1:0][MEM_ARB_SEL_WIDTH-1:0] mem_req_sel_out;

    VX_stream_arb #(
        .NUM_INPUTS (NUM_BANKS),
        .NUM_OUTPUTS(MEM_PORTS),
        .DATAW      (MEM_REQ_DATAW),
        .ARBITER    ("R")
    ) mem_req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_bank_mem_req_valid),
        .data_in   (per_bank_mem_req_pdata),
        .ready_in  (per_bank_mem_req_ready),
        .valid_out (mem_req_valid),
        .data_out  (mem_req_pdata),
        .ready_out (mem_req_ready),
        .sel_out   (mem_req_sel_out)
    );

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_req_buf
        wire                          mem_req_rw;
        wire [`CS_LINE_ADDR_WIDTH-1:0] mem_req_addr;
        wire [`CS_LINE_WIDTH-1:0]     mem_req_data;
        wire [LINE_SIZE-1:0]          mem_req_byteen;
        wire [`UP(MEM_FLAGS_WIDTH)-1:0]   mem_req_flags;
        wire [BANK_MEM_TAG_WIDTH-1:0] mem_req_tag;

        assign {
            mem_req_rw,
            mem_req_addr,
            mem_req_data,
            mem_req_byteen,
            mem_req_flags,
            mem_req_tag
        } = mem_req_pdata[i];

        wire [`CS_MEM_ADDR_WIDTH-1:0] mem_req_addr_w;
        wire [MEM_TAG_WIDTH-1:0] mem_req_tag_w;
        wire [`UP(MEM_FLAGS_WIDTH)-1:0] mem_req_flags_w;

        if (NUM_BANKS > 1) begin : g_mem_req_tag_multibanks
            if (NUM_BANKS != MEM_PORTS) begin : g_arb_sel
                wire [`CS_BANK_SEL_BITS-1:0] mem_req_bank_id;
                VX_bits_concat #(
                    .L (MEM_ARB_SEL_BITS),
                    .R (MEM_PORTS_SEL_BITS)
                ) bank_id_concat (
                    .left_in  (mem_req_sel_out[i]),
                    .right_in (MEM_PORTS_SEL_WIDTH'(i)),
                    .data_out (mem_req_bank_id)
                );
                assign mem_req_addr_w = `CS_MEM_ADDR_WIDTH'({mem_req_addr, mem_req_bank_id});
                assign mem_req_tag_w = {mem_req_tag, mem_req_sel_out[i]};
            end else begin : g_no_arb_sel
                `UNUSED_VAR (mem_req_sel_out)
                assign mem_req_addr_w = `CS_MEM_ADDR_WIDTH'({mem_req_addr, MEM_PORTS_SEL_WIDTH'(i)});
                assign mem_req_tag_w = MEM_TAG_WIDTH'(mem_req_tag);
            end
        end else begin : g_mem_req_tag
            `UNUSED_VAR (mem_req_sel_out)
            assign mem_req_addr_w = `CS_MEM_ADDR_WIDTH'(mem_req_addr);
            assign mem_req_tag_w = MEM_TAG_WIDTH'(mem_req_tag);
        end

        VX_elastic_buffer #(
            .DATAW   (1 + LINE_SIZE + `CS_MEM_ADDR_WIDTH + `CS_LINE_WIDTH + MEM_TAG_WIDTH + `UP(MEM_FLAGS_WIDTH)),
            .SIZE    (MEM_REQ_BUF_ENABLE ? `TO_OUT_BUF_SIZE(MEM_OUT_BUF) : 0),
            .OUT_REG (`TO_OUT_BUF_REG(MEM_OUT_BUF))
        ) mem_req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (mem_req_valid[i]),
            .ready_in  (mem_req_ready[i]),
            .data_in   ({mem_req_rw,                    mem_req_byteen,                    mem_req_addr_w,                  mem_req_data,                    mem_req_tag_w,                  mem_req_flags}),
            .data_out  ({mem_bus_tmp_if[i].req_data.rw, mem_bus_tmp_if[i].req_data.byteen, mem_bus_tmp_if[i].req_data.addr, mem_bus_tmp_if[i].req_data.data, mem_bus_tmp_if[i].req_data.tag, mem_req_flags_w}),
            .valid_out (mem_bus_tmp_if[i].req_valid),
            .ready_out (mem_bus_tmp_if[i].req_ready)
        );

        if (MEM_FLAGS_WIDTH != 0) begin : g_mem_req_flags
            assign mem_bus_tmp_if[i].req_data.flags = mem_req_flags_w;
        end else begin : g_no_mem_req_flags
            assign mem_bus_tmp_if[i].req_data.flags = '0;
            `UNUSED_VAR (mem_req_flags_w)
        end

        if (WRITE_ENABLE) begin : g_mem_bus_if
            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], mem_bus_tmp_if[i]);
        end else begin : g_mem_bus_if_ro
            `ASSIGN_VX_MEM_BUS_RO_IF (mem_bus_if[i], mem_bus_tmp_if[i]);
        end
    end

`ifdef PERF_ENABLE
    wire [NUM_REQS-1:0]  perf_core_reads_per_req;
    wire [NUM_REQS-1:0]  perf_core_writes_per_req;
    wire [NUM_REQS-1:0]  perf_crsp_stall_per_req;
    wire [MEM_PORTS-1:0] perf_mem_stall_per_port;

    `BUFFER(perf_core_reads_per_req, core_req_valid & core_req_ready & ~core_req_rw);
    `BUFFER(perf_core_writes_per_req, core_req_valid & core_req_ready & core_req_rw);

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_perf_crsp_stall_per_req
        assign perf_crsp_stall_per_req[i] = core_bus_if[i].rsp_valid && ~core_bus_if[i].rsp_ready;
    end

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_perf_mem_stall_per_port
        assign perf_mem_stall_per_port[i] = mem_bus_if[i].req_valid && ~mem_bus_if[i].req_ready;
    end

    // per cycle: read misses, write misses, msrq stalls, pipeline stalls
    wire [`CLOG2(NUM_REQS+1)-1:0]  perf_core_reads_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0]  perf_core_writes_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0]  perf_crsp_stall_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_read_miss_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_write_miss_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_mshr_stall_per_cycle;
    wire [`CLOG2(MEM_PORTS+1)-1:0] perf_mem_stall_per_cycle;

    `POP_COUNT(perf_core_reads_per_cycle, perf_core_reads_per_req);
    `POP_COUNT(perf_core_writes_per_cycle, perf_core_writes_per_req);
    `POP_COUNT(perf_read_miss_per_cycle, perf_read_miss_per_bank);
    `POP_COUNT(perf_write_miss_per_cycle, perf_write_miss_per_bank);
    `POP_COUNT(perf_mshr_stall_per_cycle, perf_mshr_stall_per_bank);
    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);
    `POP_COUNT(perf_mem_stall_per_cycle, perf_mem_stall_per_port);

    reg [PERF_CTR_BITS-1:0] perf_core_reads;
    reg [PERF_CTR_BITS-1:0] perf_core_writes;
    reg [PERF_CTR_BITS-1:0] perf_read_misses;
    reg [PERF_CTR_BITS-1:0] perf_write_misses;
    reg [PERF_CTR_BITS-1:0] perf_mshr_stalls;
    reg [PERF_CTR_BITS-1:0] perf_mem_stalls;
    reg [PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_core_reads   <= '0;
            perf_core_writes  <= '0;
            perf_read_misses  <= '0;
            perf_write_misses <= '0;
            perf_mshr_stalls  <= '0;
            perf_mem_stalls   <= '0;
            perf_crsp_stalls  <= '0;
        end else begin
            perf_core_reads   <= perf_core_reads   + PERF_CTR_BITS'(perf_core_reads_per_cycle);
            perf_core_writes  <= perf_core_writes  + PERF_CTR_BITS'(perf_core_writes_per_cycle);
            perf_read_misses  <= perf_read_misses  + PERF_CTR_BITS'(perf_read_miss_per_cycle);
            perf_write_misses <= perf_write_misses + PERF_CTR_BITS'(perf_write_miss_per_cycle);
            perf_mshr_stalls  <= perf_mshr_stalls  + PERF_CTR_BITS'(perf_mshr_stall_per_cycle);
            perf_mem_stalls   <= perf_mem_stalls   + PERF_CTR_BITS'(perf_mem_stall_per_cycle);
            perf_crsp_stalls  <= perf_crsp_stalls  + PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign cache_perf.reads        = perf_core_reads;
    assign cache_perf.writes       = perf_core_writes;
    assign cache_perf.read_misses  = perf_read_misses;
    assign cache_perf.write_misses = perf_write_misses;
    assign cache_perf.bank_stalls  = perf_collisions;
    assign cache_perf.mshr_stalls  = perf_mshr_stalls;
    assign cache_perf.mem_stalls   = perf_mem_stalls;
    assign cache_perf.crsp_stalls  = perf_crsp_stalls;
`endif

endmodule

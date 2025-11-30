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

`include "VX_platform.vh"

`TRACING_OFF
module VX_mem_scheduler #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_REQS     = 1,
    parameter MEM_CHANNELS  = 1,
    parameter WORD_SIZE     = 4,
    parameter LINE_SIZE     = WORD_SIZE,
    parameter ADDR_WIDTH    = 32 - `CLOG2(WORD_SIZE),
    parameter FLAGS_WIDTH   = 0,
    parameter TAG_WIDTH     = 8,
    parameter UUID_WIDTH    = 0, // upper section of the request tag contains the UUID
    parameter CORE_QUEUE_SIZE= 8,
    parameter MEM_QUEUE_SIZE= CORE_QUEUE_SIZE,
    parameter RSP_PARTIAL   = 0,
    parameter CORE_OUT_BUF  = 0,
    parameter MEM_OUT_BUF   = 0,

    parameter WORD_WIDTH    = WORD_SIZE * 8,
    parameter LINE_WIDTH    = LINE_SIZE * 8,
    parameter COALESCE_ENABLE = (CORE_REQS > 1) && (LINE_SIZE != WORD_SIZE),
    parameter PER_LINE_REQS = LINE_SIZE / WORD_SIZE,
    parameter MERGED_REQS   = CORE_REQS / PER_LINE_REQS,
    parameter MEM_BATCHES   = `CDIV(MERGED_REQS, MEM_CHANNELS),
    parameter MEM_BATCH_BITS= `CLOG2(MEM_BATCHES),
    parameter MEM_QUEUE_ADDRW= `CLOG2(COALESCE_ENABLE ? MEM_QUEUE_SIZE : CORE_QUEUE_SIZE),
    parameter MEM_ADDR_WIDTH= ADDR_WIDTH - `CLOG2(PER_LINE_REQS),
    parameter MEM_TAG_WIDTH = UUID_WIDTH + MEM_QUEUE_ADDRW + MEM_BATCH_BITS
) (
    input wire clk,
    input wire reset,

    // Core request
    input wire                              core_req_valid,
    input wire                              core_req_rw,
    input wire [CORE_REQS-1:0]              core_req_mask,
    input wire [CORE_REQS-1:0][WORD_SIZE-1:0] core_req_byteen,
    input wire [CORE_REQS-1:0][ADDR_WIDTH-1:0] core_req_addr,
    input wire [CORE_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] core_req_flags,
    input wire [CORE_REQS-1:0][WORD_WIDTH-1:0] core_req_data,
    input wire [TAG_WIDTH-1:0]              core_req_tag,
    output wire                             core_req_ready,
    output wire                             core_req_empty,
    output wire                             core_req_wr_notify,

    // Core response
    output wire                             core_rsp_valid,
    output wire [CORE_REQS-1:0]             core_rsp_mask,
    output wire [CORE_REQS-1:0][WORD_WIDTH-1:0] core_rsp_data,
    output wire [TAG_WIDTH-1:0]             core_rsp_tag,
    output wire                             core_rsp_sop,
    output wire                             core_rsp_eop,
    input wire                              core_rsp_ready,

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,
    output wire [MEM_CHANNELS-1:0]          mem_req_mask,
    output wire [MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen,
    output wire [MEM_CHANNELS-1:0][MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [MEM_CHANNELS-1:0][`UP(FLAGS_WIDTH)-1:0] mem_req_flags,
    output wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data,
    output wire [MEM_TAG_WIDTH-1:0]         mem_req_tag,
    input wire                              mem_req_ready,

    // Memory response
    input wire                              mem_rsp_valid,
    input wire [MEM_CHANNELS-1:0]           mem_rsp_mask,
    input wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_rsp_data,
    input wire [MEM_TAG_WIDTH-1:0]          mem_rsp_tag,
    output wire                             mem_rsp_ready
);
    localparam BATCH_SEL_WIDTH = `UP(MEM_BATCH_BITS);
    localparam STALL_TIMEOUT   = 10000000;
    localparam CORE_QUEUE_ADDRW= `CLOG2(CORE_QUEUE_SIZE);
    localparam TAG_ID_WIDTH    = TAG_WIDTH - UUID_WIDTH;
    localparam REQQ_TAG_WIDTH  = UUID_WIDTH + CORE_QUEUE_ADDRW;
    localparam MERGED_TAG_WIDTH= UUID_WIDTH + MEM_QUEUE_ADDRW;
    localparam CORE_CHANNELS   = COALESCE_ENABLE ? CORE_REQS : MEM_CHANNELS;
    localparam CORE_BATCHES    = COALESCE_ENABLE ? 1 : MEM_BATCHES;
    localparam CORE_BATCH_BITS = `CLOG2(CORE_BATCHES);

    `STATIC_ASSERT ((MEM_CHANNELS <= CORE_REQS), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(CORE_REQS * WORD_SIZE, LINE_SIZE), ("invalid parameter"))
    `STATIC_ASSERT ((TAG_WIDTH >= UUID_WIDTH), ("invalid parameter"))
    `RUNTIME_ASSERT((~core_req_valid || core_req_mask != 0), ("%t: invalid request mask", $time))

    wire                            ibuf_push;
    wire                            ibuf_pop;
    wire [CORE_QUEUE_ADDRW-1:0]     ibuf_waddr;
    wire [CORE_QUEUE_ADDRW-1:0]     ibuf_raddr;
    wire                            ibuf_full;
    wire                            ibuf_empty;
    wire [TAG_ID_WIDTH-1:0]         ibuf_din;
    wire [TAG_ID_WIDTH-1:0]         ibuf_dout;

    wire                            reqq_valid;
    wire [CORE_REQS-1:0]            reqq_mask;
    wire                            reqq_rw;
    wire [CORE_REQS-1:0][WORD_SIZE-1:0] reqq_byteen;
    wire [CORE_REQS-1:0][ADDR_WIDTH-1:0] reqq_addr;
    wire [CORE_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] reqq_flags;
    wire [CORE_REQS-1:0][WORD_WIDTH-1:0] reqq_data;
    wire [REQQ_TAG_WIDTH-1:0]       reqq_tag;
    wire                            reqq_ready;

    wire                            reqq_valid_s;
    wire [MERGED_REQS-1:0]          reqq_mask_s;
    wire                            reqq_rw_s;
    wire [MERGED_REQS-1:0][LINE_SIZE-1:0] reqq_byteen_s;
    wire [MERGED_REQS-1:0][MEM_ADDR_WIDTH-1:0] reqq_addr_s;
    wire [MERGED_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] reqq_flags_s;
    wire [MERGED_REQS-1:0][LINE_WIDTH-1:0] reqq_data_s;
    wire [MERGED_TAG_WIDTH-1:0]     reqq_tag_s;
    wire                            reqq_ready_s;

    wire                            mem_req_valid_s;
    wire [MEM_CHANNELS-1:0]         mem_req_mask_s;
    wire                            mem_req_rw_s;
    wire [MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen_s;
    wire [MEM_CHANNELS-1:0][MEM_ADDR_WIDTH-1:0] mem_req_addr_s;
    wire [MEM_CHANNELS-1:0][`UP(FLAGS_WIDTH)-1:0] mem_req_flags_s;
    wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data_s;
    wire [MEM_TAG_WIDTH-1:0]        mem_req_tag_s;
    wire                            mem_req_ready_s;

    wire                            mem_rsp_valid_s;
    wire [CORE_CHANNELS-1:0]        mem_rsp_mask_s;
    wire [CORE_CHANNELS-1:0][WORD_WIDTH-1:0] mem_rsp_data_s;
    wire [MEM_TAG_WIDTH-1:0]        mem_rsp_tag_s;
    wire                            mem_rsp_ready_s;

    wire                            crsp_valid;
    wire [CORE_REQS-1:0]            crsp_mask;
    wire [CORE_REQS-1:0][WORD_WIDTH-1:0] crsp_data;
    wire [TAG_WIDTH-1:0]            crsp_tag;
    wire                            crsp_sop;
    wire                            crsp_eop;
    wire                            crsp_ready;

    // Request queue //////////////////////////////////////////////////////////

    wire req_sent_all;

    wire ibuf_ready = (core_req_rw || ~ibuf_full);
    wire reqq_valid_in = core_req_valid && ibuf_ready;
    wire reqq_ready_in;

    wire [REQQ_TAG_WIDTH-1:0] reqq_tag_u;
    if (UUID_WIDTH != 0) begin : g_reqq_tag_u_uuid
        assign reqq_tag_u = {core_req_tag[TAG_WIDTH-1 -: UUID_WIDTH], ibuf_waddr};
    end else begin : g_reqq_tag_u
        assign reqq_tag_u = ibuf_waddr;
    end

    VX_elastic_buffer #(
        .DATAW   (1 + CORE_REQS * (1 + WORD_SIZE + ADDR_WIDTH + `UP(FLAGS_WIDTH) + WORD_WIDTH) + REQQ_TAG_WIDTH),
        .SIZE    (CORE_QUEUE_SIZE),
        .OUT_REG (1)
    ) req_queue (
        .clk      (clk),
        .reset    (reset),
        .valid_in (reqq_valid_in),
        .ready_in (reqq_ready_in),
        .data_in  ({core_req_rw, core_req_mask, core_req_byteen, core_req_addr, core_req_flags, core_req_data, reqq_tag_u}),
        .data_out ({reqq_rw,     reqq_mask,     reqq_byteen,     reqq_addr,     reqq_flags,     reqq_data,     reqq_tag}),
        .valid_out(reqq_valid),
        .ready_out(reqq_ready)
    );

    // can accept another request?
    assign core_req_ready = reqq_ready_in && ibuf_ready;

    // no pending requests
    assign core_req_empty = !reqq_valid && ibuf_empty;

    // notify write request submisison
    assign core_req_wr_notify = reqq_valid && reqq_ready && reqq_rw;

    // Index buffer ///////////////////////////////////////////////////////////

    wire core_req_fire = core_req_valid && core_req_ready;
    wire crsp_fire = crsp_valid && crsp_ready;

    assign ibuf_push  = core_req_fire && ~core_req_rw;
    assign ibuf_pop   = crsp_fire && crsp_eop;
    assign ibuf_raddr = mem_rsp_tag_s[CORE_BATCH_BITS +: CORE_QUEUE_ADDRW];
    assign ibuf_din   = core_req_tag[TAG_ID_WIDTH-1:0];

    VX_index_buffer #(
        .DATAW (TAG_ID_WIDTH),
        .SIZE  (CORE_QUEUE_SIZE)
    ) req_ibuf (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (ibuf_push),
        .write_addr   (ibuf_waddr),
        .write_data   (ibuf_din),
        .read_data    (ibuf_dout),
        .read_addr    (ibuf_raddr),
        .release_en   (ibuf_pop),
        .full         (ibuf_full),
        .empty        (ibuf_empty)
    );

    `UNUSED_VAR (ibuf_empty)

    // Handle memory coalescing ///////////////////////////////////////////////

    if (COALESCE_ENABLE) begin : g_coalescer

        VX_mem_coalescer #(
            .INSTANCE_ID    (`SFORMATF(("%s-coalescer", INSTANCE_ID))),
            .NUM_REQS       (CORE_REQS),
            .DATA_IN_SIZE   (WORD_SIZE),
            .DATA_OUT_SIZE  (LINE_SIZE),
            .ADDR_WIDTH     (ADDR_WIDTH),
            .FLAGS_WIDTH    (FLAGS_WIDTH),
            .TAG_WIDTH      (REQQ_TAG_WIDTH),
            .UUID_WIDTH     (UUID_WIDTH),
            .QUEUE_SIZE     (MEM_QUEUE_SIZE)
        ) coalescer (
            .clk            (clk),
            .reset          (reset),

            `UNUSED_PIN (misses),

            // Input request
            .in_req_valid   (reqq_valid),
            .in_req_mask    (reqq_mask),
            .in_req_rw      (reqq_rw),
            .in_req_byteen  (reqq_byteen),
            .in_req_addr    (reqq_addr),
            .in_req_flags   (reqq_flags),
            .in_req_data    (reqq_data),
            .in_req_tag     (reqq_tag),
            .in_req_ready   (reqq_ready),

            // Input response
            .in_rsp_valid   (mem_rsp_valid_s),
            .in_rsp_mask    (mem_rsp_mask_s),
            .in_rsp_data    (mem_rsp_data_s),
            .in_rsp_tag     (mem_rsp_tag_s),
            .in_rsp_ready   (mem_rsp_ready_s),

            // Output request
            .out_req_valid  (reqq_valid_s),
            .out_req_mask   (reqq_mask_s),
            .out_req_rw     (reqq_rw_s),
            .out_req_byteen (reqq_byteen_s),
            .out_req_addr   (reqq_addr_s),
            .out_req_flags  (reqq_flags_s),
            .out_req_data   (reqq_data_s),
            .out_req_tag    (reqq_tag_s),
            .out_req_ready  (reqq_ready_s),

            // Output response
            .out_rsp_valid  (mem_rsp_valid),
            .out_rsp_mask   (mem_rsp_mask),
            .out_rsp_data   (mem_rsp_data),
            .out_rsp_tag    (mem_rsp_tag),
            .out_rsp_ready  (mem_rsp_ready)
        );

    end else begin : g_no_coalescer
        assign reqq_valid_s = reqq_valid;
        assign reqq_mask_s  = reqq_mask;
        assign reqq_rw_s    = reqq_rw;
        assign reqq_byteen_s= reqq_byteen;
        assign reqq_addr_s  = reqq_addr;
        assign reqq_flags_s = reqq_flags;
        assign reqq_data_s  = reqq_data;
        assign reqq_tag_s   = reqq_tag;
        assign reqq_ready   = reqq_ready_s;

        assign mem_rsp_valid_s = mem_rsp_valid;
        assign mem_rsp_mask_s  = mem_rsp_mask;
        assign mem_rsp_data_s  = mem_rsp_data;
        assign mem_rsp_tag_s   = mem_rsp_tag;
        assign mem_rsp_ready   = mem_rsp_ready_s;

    end

    // Handle memory requests /////////////////////////////////////////////////

    wire [MEM_BATCHES-1:0][MEM_CHANNELS-1:0] mem_req_mask_b;
    wire [MEM_BATCHES-1:0][MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen_b;
    wire [MEM_BATCHES-1:0][MEM_CHANNELS-1:0][MEM_ADDR_WIDTH-1:0] mem_req_addr_b;
    wire [MEM_BATCHES-1:0][MEM_CHANNELS-1:0][`UP(FLAGS_WIDTH)-1:0] mem_req_flags_b;
    wire [MEM_BATCHES-1:0][MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data_b;

    wire [BATCH_SEL_WIDTH-1:0] req_batch_idx;

    for (genvar i = 0; i < MEM_BATCHES; ++i) begin : g_mem_req_data_b
        for (genvar j = 0; j < MEM_CHANNELS; ++j) begin : g_j
            localparam r = i * MEM_CHANNELS + j;
            if (r < MERGED_REQS) begin : g_valid
                assign mem_req_mask_b[i][j]   = reqq_mask_s[r];
                assign mem_req_byteen_b[i][j] = reqq_byteen_s[r];
                assign mem_req_addr_b[i][j]   = reqq_addr_s[r];
                assign mem_req_flags_b[i][j]  = reqq_flags_s[r];
                assign mem_req_data_b[i][j]   = reqq_data_s[r];
            end else begin : g_padding
                assign mem_req_mask_b[i][j]   = 0;
                assign mem_req_byteen_b[i][j] = '0;
                assign mem_req_addr_b[i][j]   = '0;
                assign mem_req_flags_b[i][j]  = '0;
                assign mem_req_data_b[i][j]   = '0;
            end
        end
    end

    assign mem_req_mask_s   = mem_req_mask_b[req_batch_idx];
    assign mem_req_rw_s     = reqq_rw_s;
    assign mem_req_byteen_s = mem_req_byteen_b[req_batch_idx];
    assign mem_req_addr_s   = mem_req_addr_b[req_batch_idx];
    assign mem_req_flags_s  = mem_req_flags_b[req_batch_idx];
    assign mem_req_data_s   = mem_req_data_b[req_batch_idx];

    if (MEM_BATCHES != 1) begin : g_batch
        reg [MEM_BATCH_BITS-1:0] req_batch_idx_r;

        wire is_degenerate_batch = ~(| mem_req_mask_s);
        wire mem_req_valid_b = reqq_valid_s && ~is_degenerate_batch;
        wire mem_req_ready_b = mem_req_ready_s || is_degenerate_batch;

        always @(posedge clk) begin
            if (reset) begin
                req_batch_idx_r <= '0;
            end else begin
                if (reqq_valid_s && mem_req_ready_b) begin
                    if (req_sent_all) begin
                        req_batch_idx_r <= '0;
                    end else begin
                        req_batch_idx_r <= req_batch_idx_r + MEM_BATCH_BITS'(1);
                    end
                end
            end
        end

        wire [MEM_BATCHES-1:0] req_batch_valids;
        wire [MEM_BATCHES-1:0][MEM_BATCH_BITS-1:0] req_batch_idxs;
        wire [MEM_BATCH_BITS-1:0] req_batch_idx_last;

        for (genvar i = 0; i < MEM_BATCHES; ++i) begin : g_req_batch
            assign req_batch_valids[i] = (| mem_req_mask_b[i]);
            assign req_batch_idxs[i] = MEM_BATCH_BITS'(i);
        end

        VX_find_first #(
            .N       (MEM_BATCHES),
            .DATAW   (MEM_BATCH_BITS),
            .REVERSE (1)
        ) find_last (
            .valid_in  (req_batch_valids),
            .data_in   (req_batch_idxs),
            .data_out  (req_batch_idx_last),
            `UNUSED_PIN (valid_out)
        );

        assign mem_req_valid_s = mem_req_valid_b;
        assign req_batch_idx = req_batch_idx_r;
        assign req_sent_all  = mem_req_ready_b && (req_batch_idx_r == req_batch_idx_last);
        assign mem_req_tag_s = {reqq_tag_s, req_batch_idx};

    end else begin : g_no_batch

        assign mem_req_valid_s = reqq_valid_s;
        assign req_batch_idx = '0;
        assign req_sent_all  = mem_req_ready_s;
        assign mem_req_tag_s = reqq_tag_s;

    end

    assign reqq_ready_s = req_sent_all;

    wire [MEM_CHANNELS-1:0][`UP(FLAGS_WIDTH)-1:0] mem_req_flags_u;

    VX_elastic_buffer #(
        .DATAW   (MEM_CHANNELS + 1 + MEM_CHANNELS * (LINE_SIZE + MEM_ADDR_WIDTH + `UP(FLAGS_WIDTH) + LINE_WIDTH) + MEM_TAG_WIDTH),
        .SIZE    (`TO_OUT_BUF_SIZE(MEM_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(MEM_OUT_BUF))
    ) mem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_valid_s),
        .ready_in  (mem_req_ready_s),
        .data_in   ({mem_req_mask_s, mem_req_rw_s, mem_req_byteen_s, mem_req_addr_s, mem_req_flags_s, mem_req_data_s, mem_req_tag_s}),
        .data_out  ({mem_req_mask,   mem_req_rw,   mem_req_byteen,   mem_req_addr,   mem_req_flags_u, mem_req_data,   mem_req_tag}),
        .valid_out (mem_req_valid),
        .ready_out (mem_req_ready)
    );

    if (FLAGS_WIDTH != 0) begin : g_mem_req_flags
        assign mem_req_flags = mem_req_flags_u;
    end else begin : g_mem_req_flags_0
        `UNUSED_VAR (mem_req_flags_u)
        assign mem_req_flags = '0;
    end

    // Handle memory responses ////////////////////////////////////////////////

    wire [BATCH_SEL_WIDTH-1:0] rsp_batch_idx;
    if (CORE_BATCHES > 1) begin : g_rsp_batch_idx
        assign rsp_batch_idx = mem_rsp_tag_s[CORE_BATCH_BITS-1:0];
    end else begin : g_rsp_batch_idx_0
        assign rsp_batch_idx = '0;
    end

    if (CORE_REQS == 1) begin : g_rsp_1
        `UNUSED_VAR (rsp_batch_idx)

        assign crsp_valid = mem_rsp_valid_s;
        assign crsp_mask  = mem_rsp_mask_s;
        assign crsp_sop   = 1'b1;
        assign crsp_eop   = 1'b1;
        assign crsp_data  = mem_rsp_data_s;

        assign mem_rsp_ready_s = crsp_ready;

    end else begin : g_rsp_N

        reg [CORE_QUEUE_SIZE-1:0][CORE_REQS-1:0] rsp_rem_mask;
        wire [CORE_REQS-1:0] rsp_rem_mask_n, curr_mask;

        for (genvar r = 0; r < CORE_REQS; ++r) begin : g_curr_mask
            localparam i = r / CORE_CHANNELS;
            localparam j = r % CORE_CHANNELS;
            assign curr_mask[r] = (BATCH_SEL_WIDTH'(i) == rsp_batch_idx) && mem_rsp_mask_s[j];
        end

        assign rsp_rem_mask_n = rsp_rem_mask[ibuf_raddr] & ~curr_mask;

        wire mem_rsp_fire_s = mem_rsp_valid_s && mem_rsp_ready_s;

        always @(posedge clk) begin
            if (ibuf_push) begin
                rsp_rem_mask[ibuf_waddr] <= core_req_mask;
            end
            if (mem_rsp_fire_s) begin
                rsp_rem_mask[ibuf_raddr] <= rsp_rem_mask_n;
            end
        end

        wire rsp_complete = ~(| rsp_rem_mask_n) || (CORE_REQS == 1);

        if (RSP_PARTIAL != 0) begin : g_rsp_partial

            reg [CORE_QUEUE_SIZE-1:0] rsp_sop_r;

            always @(posedge clk) begin
                if (ibuf_push) begin
                    rsp_sop_r[ibuf_waddr] <= 1;
                end
                if (mem_rsp_fire_s) begin
                    rsp_sop_r[ibuf_raddr] <= 0;
                end
            end

            assign crsp_valid = mem_rsp_valid_s;
            assign crsp_mask  = curr_mask;
            assign crsp_sop   = rsp_sop_r[ibuf_raddr];

            for (genvar r = 0; r < CORE_REQS; ++r) begin : g_crsp_data
                localparam j = r % CORE_CHANNELS;
                assign crsp_data[r] = mem_rsp_data_s[j];
            end

            assign mem_rsp_ready_s = crsp_ready;

        end else begin : g_rsp_full

            wire [CORE_CHANNELS-1:0][CORE_BATCHES-1:0][WORD_WIDTH-1:0] rsp_store_n;
            reg [CORE_REQS-1:0] rsp_orig_mask [CORE_QUEUE_SIZE-1:0];

            for (genvar i = 0; i < CORE_CHANNELS; ++i) begin : g_rsp_store
                for (genvar j = 0; j < CORE_BATCHES; ++j) begin : g_j
                    reg [WORD_WIDTH-1:0] rsp_store [0:CORE_QUEUE_SIZE-1];
                    wire rsp_wren = mem_rsp_fire_s
                                && (BATCH_SEL_WIDTH'(j) == rsp_batch_idx)
                                && ((CORE_CHANNELS == 1) || mem_rsp_mask_s[i]);
                    always @(posedge clk) begin
                        if (rsp_wren) begin
                            rsp_store[ibuf_raddr] <= mem_rsp_data_s[i];
                        end
                    end
                    assign rsp_store_n[i][j] = rsp_wren ? mem_rsp_data_s[i] : rsp_store[ibuf_raddr];
                end
            end

            always @(posedge clk) begin
                if (ibuf_push) begin
                    rsp_orig_mask[ibuf_waddr] <= core_req_mask;
                end
            end

            assign crsp_valid = mem_rsp_valid_s && rsp_complete;
            assign crsp_mask  = rsp_orig_mask[ibuf_raddr];
            assign crsp_sop   = 1'b1;

            for (genvar r = 0; r < CORE_REQS; ++r) begin : g_crsp_data
                localparam i = r / CORE_CHANNELS;
                localparam j = r % CORE_CHANNELS;
                assign crsp_data[r] = rsp_store_n[j][i];
            end

            assign mem_rsp_ready_s = crsp_ready || ~rsp_complete;
        end

        assign crsp_eop = rsp_complete;
    end

    if (UUID_WIDTH != 0) begin : g_crsp_tag
        assign crsp_tag = {mem_rsp_tag_s[MEM_TAG_WIDTH-1 -: UUID_WIDTH], ibuf_dout};
    end else begin : g_crsp_tag_0
        assign crsp_tag = ibuf_dout;
    end

    // Send response to caller

    VX_elastic_buffer #(
        .DATAW   (CORE_REQS + 1 + 1 + (CORE_REQS * WORD_WIDTH) + TAG_WIDTH),
        .SIZE    (`TO_OUT_BUF_SIZE(CORE_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsp_valid),
        .ready_in  (crsp_ready),
        .data_in   ({crsp_mask,     crsp_sop,     crsp_eop,     crsp_data,     crsp_tag}),
        .data_out  ({core_rsp_mask, core_rsp_sop, core_rsp_eop, core_rsp_data, core_rsp_tag}),
        .valid_out (core_rsp_valid),
        .ready_out (core_rsp_ready)
    );

`ifdef SIMULATION
    wire [`UP(UUID_WIDTH)-1:0] req_dbg_uuid;

    if (UUID_WIDTH != 0) begin : g_req_dbg_uuid
        assign req_dbg_uuid = core_req_tag[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_req_dbg_uuid_0
        assign req_dbg_uuid = '0;
    end

    reg [(`UP(UUID_WIDTH) + TAG_ID_WIDTH + 64)-1:0] pending_reqs_time [CORE_QUEUE_SIZE-1:0];
    reg [CORE_QUEUE_SIZE-1:0] pending_reqs_valid;

    always @(posedge clk) begin
        if (reset) begin
            pending_reqs_valid <= '0;
        end else begin
            if (ibuf_push) begin
                pending_reqs_valid[ibuf_waddr] <= 1'b1;
            end
            if (ibuf_pop) begin
                pending_reqs_valid[ibuf_raddr] <= 1'b0;
            end
        end

        if (ibuf_push) begin
            pending_reqs_time[ibuf_waddr] <= {req_dbg_uuid, ibuf_din, $time};
        end

        for (integer i = 0; i < CORE_QUEUE_SIZE; ++i) begin
            if (pending_reqs_valid[i]) begin
                `ASSERT(($time - pending_reqs_time[i][63:0]) < STALL_TIMEOUT,
                    ("%t: *** %s response timeout: tag=0x%0h (#%0d)",
                        $time, INSTANCE_ID, pending_reqs_time[i][64 +: TAG_ID_WIDTH], pending_reqs_time[i][64+TAG_ID_WIDTH +: `UP(UUID_WIDTH)]));
            end
        end
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

`ifdef DBG_TRACE_MEM
    wire [`UP(UUID_WIDTH)-1:0] mem_req_dbg_uuid;
    wire [`UP(UUID_WIDTH)-1:0] mem_rsp_dbg_uuid;
    wire [`UP(UUID_WIDTH)-1:0] rsp_dbg_uuid;

    if (UUID_WIDTH != 0) begin : g_dbg_uuid
        assign mem_req_dbg_uuid = mem_req_tag_s[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
        assign mem_rsp_dbg_uuid = mem_rsp_tag_s[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
        assign rsp_dbg_uuid     = core_rsp_tag[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_dbg_uuid_0
        assign mem_req_dbg_uuid = '0;
        assign mem_rsp_dbg_uuid = '0;
        assign rsp_dbg_uuid     = '0;
    end

    wire [CORE_QUEUE_ADDRW-1:0] ibuf_waddr_s = mem_req_tag_s[MEM_BATCH_BITS +: CORE_QUEUE_ADDRW];

    wire mem_req_fire_s = mem_req_valid_s && mem_req_ready_s;

    always @(posedge clk) begin
        if (core_req_fire) begin
            if (core_req_rw) begin
                `TRACE(2, ("%t: %s core-req-wr: valid=%b, addr=", $time, INSTANCE_ID, core_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", core_req_addr, CORE_REQS)
                `TRACE(2, (", byteen="))
                `TRACE_ARRAY1D(2, "0x%h", core_req_byteen, CORE_REQS)
                `TRACE(2, (", data="))
                `TRACE_ARRAY1D(2, "0x%0h", core_req_data, CORE_REQS)
            end else begin
                `TRACE(2, ("%t: %s core-req-rd: valid=%b, addr=", $time, INSTANCE_ID, core_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", core_req_addr, CORE_REQS)
            end
            `TRACE(2, (", tag=0x%0h (#%0d)\n", core_req_tag, req_dbg_uuid))
        end
        if (core_rsp_valid && core_rsp_ready) begin
            `TRACE(2, ("%t: %s core-rsp: valid=%b, sop=%b, eop=%b, data=", $time, INSTANCE_ID, core_rsp_mask, core_rsp_sop, core_rsp_eop))
            `TRACE_ARRAY1D(2, "0x%0h", core_rsp_data, CORE_REQS)
            `TRACE(2, (", tag=0x%0h (#%0d)\n", core_rsp_tag, rsp_dbg_uuid))
        end
        if (| mem_req_fire_s) begin
            if (| mem_req_rw_s) begin
                `TRACE(2, ("%t: %s mem-req-wr: valid=%b, addr=", $time, INSTANCE_ID, mem_req_mask_s))
                `TRACE_ARRAY1D(2, "0x%h", mem_req_addr_s, CORE_CHANNELS)
                `TRACE(2, (", byteen="))
                `TRACE_ARRAY1D(2, "0x%h", mem_req_byteen_s, CORE_CHANNELS)
                `TRACE(2, (", data="))
                `TRACE_ARRAY1D(2, "0x%0h", mem_req_data_s, CORE_CHANNELS)
            end else begin
                `TRACE(2, ("%t: %s mem-req-rd: valid=%b, addr=", $time, INSTANCE_ID, mem_req_mask_s))
                `TRACE_ARRAY1D(2, "0x%h", mem_req_addr_s, CORE_CHANNELS)
            end
            `TRACE(2, (", ibuf_idx=%0d, batch_idx=%0d (#%0d)\n", ibuf_waddr_s, req_batch_idx, mem_req_dbg_uuid))
        end
        if (mem_rsp_valid_s && mem_rsp_ready_s) begin
            `TRACE(2, ("%t: %s mem-rsp: valid=%b, data=", $time, INSTANCE_ID, mem_rsp_mask_s))
            `TRACE_ARRAY1D(2, "0x%0h", mem_rsp_data_s, CORE_CHANNELS)
            `TRACE(2, (", ibuf_idx=%0d, batch_idx=%0d (#%0d)\n", ibuf_raddr, rsp_batch_idx, mem_rsp_dbg_uuid))
        end
    end
`endif

endmodule
`TRACING_ON

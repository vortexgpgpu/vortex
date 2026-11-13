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

// Warp-wide local memory. It takes whole warp requests from the LSU blocks and
// serves each in bank-conflict-free passes (VX_lmem_engine): same-word reads
// are broadcast to every lane reading the word, conflicting lanes replay, and
// requests retire in order, so every lane's accesses reach the banks in program
// order.
//
// With several LSU blocks a second engine, for plain reads only, works the
// banks' second port, so two warps' reads go through each cycle. Stores and
// atomics use the first engine alone: one enters it only once the read engine
// has drained, and the read engine takes no read while a store or atomic sits in
// the first engine, while one waited at an input the cycle before, or in a cycle
// the first engine may be taking one. A write therefore never meets a read of
// the other port in a bank, and keeps its order against every read.
//
// The DMA port (DXA writes, TCU reads) takes whole bank rows. A DMA request seen
// in one cycle gets the banks the next: no pass is issued in a cycle a DMA
// request is seen, and it is accepted in the cycle that follows. An atomic owns
// its bank for three cycles (read, capture, write back), so no pass or DMA access
// is issued in the two cycles after an atomic pass.
module VX_lmem_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",

    // Size in bytes
    parameter SIZE          = (1024*16),
    // Number of LSU blocks
    parameter NUM_BLOCKS    = 1,
    // Lanes of a warp request
    parameter NUM_LANES     = 4,
    // Number of banks
    parameter NUM_BANKS     = 4,
    // Size of a word in bytes
    parameter WORD_SIZE     = `VX_CFG_XLEN/8,
    // Word address width
    parameter ADDR_WIDTH    = `CLOG2(SIZE / WORD_SIZE),
    // Request tag size
    parameter TAG_WIDTH     = 16,

    // Enable DMA port
    parameter DMA_ENABLE    = 0,
    parameter DMA_TAG_WIDTH = 1,

    // Synthesize atomic-op + LR/SC logic in the banks
    parameter AMO_ENABLE    = 0
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output lmem_perf_t lmem_perf,
`endif

    // LSU blocks' warp requests
    VX_lsu_mem_if.slave lsu_mem_if [NUM_BLOCKS],

    // DMA read/write port
    VX_mem_bus_if.slave dma_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam NUM_WORDS       = SIZE / WORD_SIZE;
    localparam WORDS_PER_BANK  = NUM_WORDS / NUM_BANKS;
    localparam BANK_ADDR_WIDTH = `CLOG2(WORDS_PER_BANK);
    localparam BLOCK_BITS      = `CLOG2(NUM_BLOCKS);
    localparam BLOCK_WIDTH     = `UP(BLOCK_BITS);
    localparam DUAL            = (NUM_BLOCKS > 1);
    // A pass's response reaches its block's response register four cycles
    // after the pass at the earliest, and its slot is free for a pass the cycle
    // after the block takes it: eight slots keep one pass per cycle going.
    localparam RSP_QUEUE_SIZE  = 8;
    localparam DONE_WIDTH      = `CLOG2(NUM_BLOCKS + 1);
    localparam AMO_OP_WIDTH    = $bits(amo_op_e);
    // Compare-and-swap adds a comparand: the request's data word already
    // carries the swap value, so the third operand needs its own room.
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    localparam AMO_CMP_WIDTH   = `VX_CFG_XLEN;
`else
    localparam AMO_CMP_WIDTH   = 0;
`endif
    // per lane: the hart id that holds a reservation, and the comparand
    localparam LANE_SB_WIDTH   = AMO_ENABLE ? (HART_ID_WIDTH + AMO_CMP_WIDTH) : 0;
    localparam PAYLOAD_WIDTH   = WORD_WIDTH + WORD_SIZE + LANE_SB_WIDTH;
    // common to all lanes: the atomic op and its signedness
    localparam ATTR_WIDTH      = AMO_OP_WIDTH + 1;

    `STATIC_ASSERT(ADDR_WIDTH == (BANK_ADDR_WIDTH + `CLOG2(NUM_BANKS)), ("invalid parameter"))
    `STATIC_ASSERT(`IS_POW2(NUM_BANKS), ("invalid parameter"))

    // the blocks' requests

    wire [NUM_BLOCKS-1:0]                                   in_valid;
    wire [NUM_BLOCKS-1:0]                                   in_rw;
    wire [NUM_BLOCKS-1:0]                                   in_amo;
    wire [NUM_BLOCKS-1:0]                                   in_read;
    wire [NUM_BLOCKS-1:0][ATTR_WIDTH-1:0]                   in_attr;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]                    in_mask;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0]    in_addr;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][PAYLOAD_WIDTH-1:0] in_payload;
    wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]                    in_tag;
    wire [NUM_BLOCKS-1:0]                                   in_ready;

    for (genvar k = 0; k < NUM_BLOCKS; ++k) begin : g_in
        // the atomic op is the same on every lane of a request
        mem_bus_attr_t attr0;
        assign attr0 = mem_bus_attr_t'(lsu_mem_if[k].req_data.user[0]);
        assign in_valid[k] = lsu_mem_if[k].req_valid;
        assign in_rw[k]    = lsu_mem_if[k].req_data.rw;
        assign in_amo[k]   = AMO_ENABLE ? attr0.amo.amo_valid : 1'b0;
        assign in_read[k]  = ~in_rw[k] && ~in_amo[k];
        assign in_attr[k]  = AMO_ENABLE ? {attr0.amo.amo_op, attr0.amo.amo_unsigned} : '0;
        assign in_mask[k]  = lsu_mem_if[k].req_data.mask;
        assign in_tag[k]   = lsu_mem_if[k].req_data.tag;
        `UNUSED_VAR (attr0)
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
            assign in_addr[k][i] = lsu_mem_if[k].req_data.addr[i][ADDR_WIDTH-1:0];
            if (AMO_ENABLE != 0) begin : g_amo
                mem_bus_attr_t lane_attr;
                assign lane_attr = mem_bus_attr_t'(lsu_mem_if[k].req_data.user[i]);
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                assign in_payload[k][i] = {lane_attr.amo.hart_id, lane_attr.amo.amo_cmp,
                                           lsu_mem_if[k].req_data.byteen[i], lsu_mem_if[k].req_data.data[i]};
            `else
                assign in_payload[k][i] = {lane_attr.amo.hart_id,
                                           lsu_mem_if[k].req_data.byteen[i], lsu_mem_if[k].req_data.data[i]};
            `endif
                `UNUSED_VAR (lane_attr)
            end else begin : g_no_amo
                assign in_payload[k][i] = {lsu_mem_if[k].req_data.byteen[i], lsu_mem_if[k].req_data.data[i]};
                `UNUSED_VAR (lsu_mem_if[k].req_data.user[i])
            end
            `UNUSED_VAR (lsu_mem_if[k].req_data.addr[i])
        end
        assign lsu_mem_if[k].req_ready = in_ready[k];
    end

    // the engines

    wire  pass_en;
    wire  a_pass_amo;

    wire                   a_req_valid;
    wire                   a_req_ready;
    wire                   a_busy;
    wire                   a_holds_write;
    wire [BLOCK_WIDTH-1:0] a_req_block;
    wire                   b_req_valid;
    wire                   b_req_ready;
    wire [BLOCK_WIDTH-1:0] b_req_block;

    wire [NUM_BANKS-1:0]                      a_bank_valid;
    wire                                      a_bank_rw;
    wire                                      a_bank_amo;
    wire [ATTR_WIDTH-1:0]                     a_bank_attr;
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] a_bank_addr;
    wire [NUM_BANKS-1:0][PAYLOAD_WIDTH-1:0]   a_bank_payload;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]      a_bank_rdata;

    wire                                 a_rsp_valid;
    wire                                 a_rsp_take;
    wire [DONE_WIDTH-1:0]                a_rsp_done;
    wire [BLOCK_WIDTH-1:0]               a_rsp_block;
    wire [NUM_LANES-1:0]                 a_rsp_mask;
    wire [NUM_LANES-1:0][WORD_WIDTH-1:0] a_rsp_data;
    wire [TAG_WIDTH-1:0]                 a_rsp_tag;

    wire [NUM_LANES-1:0] a_perf_served;
    wire                 a_perf_store;
    wire                 a_perf_replay;

    wire                                      b_busy;
    wire [NUM_BANKS-1:0]                      b_bank_valid;
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] b_bank_addr;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]      b_bank_rdata;

    wire                                 b_rsp_valid;
    wire                                 b_rsp_take;
    wire [DONE_WIDTH-1:0]                b_rsp_done;
    wire [BLOCK_WIDTH-1:0]               b_rsp_block;
    wire [NUM_LANES-1:0]                 b_rsp_mask;
    wire [NUM_LANES-1:0][WORD_WIDTH-1:0] b_rsp_data;
    wire [TAG_WIDTH-1:0]                 b_rsp_tag;

    wire [NUM_LANES-1:0] b_perf_served;
    wire                 b_perf_replay;

    // Block arbitration. The ready each block sees comes from registers only:
    // each engine's block is chosen a cycle ahead, from the requests waiting
    // then, and the engine takes it if it is still there. A block that was
    // just served may present a new request of either kind, so a choice of it
    // counts as a store: the first engine takes a store only while the read
    // engine is idle, and the read engine takes nothing in that cycle.

    wire [NUM_BLOCKS-1:0] in_fire = in_valid & in_ready;

    if (DUAL) begin : g_dual_arb

        reg                   a_sel_valid_r;
        reg [BLOCK_WIDTH-1:0] a_sel_r;
        reg                   a_sel_read_r;
        reg                   b_sel_valid_r;
        reg [BLOCK_WIDTH-1:0] b_sel_r;
        reg                   write_waiting_r;

        wire a_take = a_sel_valid_r && a_req_ready && (a_sel_read_r || ~b_busy);
        wire b_take = b_sel_valid_r && b_req_ready && ~a_holds_write && ~write_waiting_r
                   && ~(a_take && ~a_sel_read_r);

        for (genvar k = 0; k < NUM_BLOCKS; ++k) begin : g_ready
            assign in_ready[k] = (a_take && (a_sel_r == BLOCK_WIDTH'(k)))
                              || (b_take && (b_sel_r == BLOCK_WIDTH'(k)));
        end

        assign a_req_valid = a_take && in_valid[a_sel_r];
        assign a_req_block = a_sel_r;
        assign b_req_valid = b_take && in_valid[b_sel_r];
        assign b_req_block = b_sel_r;

        // the next cycle's choices, from the requests waiting now
        wire [NUM_BLOCKS-1:0] known_read = in_valid & in_read & ~in_fire;
        wire b_busy_n = b_busy || (b_req_valid && b_req_ready);
        wire [NUM_BLOCKS-1:0] a_eligible = known_read | (in_valid & {NUM_BLOCKS{~b_busy_n}});

        wire                   a_next_valid;
        wire [BLOCK_WIDTH-1:0] a_next;
        wire [NUM_BLOCKS-1:0]  a_next_onehot;

        VX_generic_arbiter #(
            .NUM_REQS (NUM_BLOCKS),
            .TYPE     ("R")
        ) a_arbiter (
            .clk          (clk),
            .reset        (reset),
            .requests     (a_eligible),
            .grant_index  (a_next),
            .grant_onehot (a_next_onehot),
            .grant_valid  (a_next_valid),
            .grant_ready  (1'b1)
        );

        wire [NUM_BLOCKS-1:0] b_eligible = known_read & ~a_next_onehot;

        wire                   b_next_valid;
        wire [BLOCK_WIDTH-1:0] b_next;
        wire [NUM_BLOCKS-1:0]  b_next_onehot;

        VX_generic_arbiter #(
            .NUM_REQS (NUM_BLOCKS),
            .TYPE     ("R")
        ) b_arbiter (
            .clk          (clk),
            .reset        (reset),
            .requests     (b_eligible),
            .grant_index  (b_next),
            .grant_onehot (b_next_onehot),
            .grant_valid  (b_next_valid),
            .grant_ready  (1'b1)
        );
        `UNUSED_VAR (b_next_onehot)

        always @(posedge clk) begin
            if (reset) begin
                a_sel_valid_r   <= 1'b0;
                b_sel_valid_r   <= 1'b0;
                write_waiting_r <= 1'b0;
            end else begin
                a_sel_valid_r   <= a_next_valid;
                b_sel_valid_r   <= b_next_valid;
                write_waiting_r <= (| (in_valid & ~in_read & ~in_fire));
            end
            a_sel_r      <= a_next;
            a_sel_read_r <= known_read[a_next];
            b_sel_r      <= b_next;
        end

    end else begin : g_single_arb

        // one block: its request goes straight to the first engine
        assign a_req_valid = in_valid[0];
        assign a_req_block = '0;
        assign in_ready[0] = a_req_ready;
        assign b_req_valid = 1'b0;
        assign b_req_block = '0;
        `UNUSED_VAR (in_fire)
        `UNUSED_VAR (b_req_ready)
        `UNUSED_VAR (b_busy)

    end

    VX_lmem_engine #(
        .INSTANCE_ID    (`SFORMATF(("%s-engine0", INSTANCE_ID))),
        .NUM_LANES      (NUM_LANES),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .TAG_WIDTH      (TAG_WIDTH),
        .PAYLOAD_WIDTH  (PAYLOAD_WIDTH),
        .ATTR_WIDTH     (ATTR_WIDTH),
        .NUM_BLOCKS     (NUM_BLOCKS),
        .RSP_QUEUE_SIZE (RSP_QUEUE_SIZE)
    ) engine_a (
        .clk          (clk),
        .reset        (reset),
        .req_valid    (a_req_valid),
        .req_block    (a_req_block),
        .req_rw       (in_rw[a_req_block]),
        .req_read     (in_read[a_req_block]),
        .req_amo      (in_amo[a_req_block]),
        .req_attr     (in_attr[a_req_block]),
        .req_mask     (in_mask[a_req_block]),
        .req_addr     (in_addr[a_req_block]),
        .req_payload  (in_payload[a_req_block]),
        .req_tag      (in_tag[a_req_block]),
        .req_ready    (a_req_ready),
        .busy         (a_busy),
        .holds_write  (a_holds_write),
        .pass_en      (pass_en),
        .pass_amo     (a_pass_amo),
        .bank_valid   (a_bank_valid),
        .bank_rw      (a_bank_rw),
        .bank_amo     (a_bank_amo),
        .bank_attr    (a_bank_attr),
        .bank_addr    (a_bank_addr),
        .bank_payload (a_bank_payload),
        .bank_rdata   (a_bank_rdata),
        .rsp_valid    (a_rsp_valid),
        .rsp_block    (a_rsp_block),
        .rsp_mask     (a_rsp_mask),
        .rsp_data     (a_rsp_data),
        .rsp_tag      (a_rsp_tag),
        .rsp_take     (a_rsp_take),
        .rsp_done     (a_rsp_done),
        .perf_served  (a_perf_served),
        .perf_store   (a_perf_store),
        .perf_replay  (a_perf_replay)
    );
    `UNUSED_VAR (a_busy)

    if (DUAL) begin : g_read_engine

        wire [NUM_LANES-1:0][0:0] b_req_payload = '0;
        wire                      b_bank_rw;
        wire                      b_bank_amo;
        wire                      b_holds_write;
        wire                      b_pass_amo;
        wire                      b_perf_store;
        wire [ATTR_WIDTH-1:0]     b_bank_attr;
        wire [NUM_BANKS-1:0][0:0] b_bank_payload;

        VX_lmem_engine #(
            .INSTANCE_ID    (`SFORMATF(("%s-engine1", INSTANCE_ID))),
            .NUM_LANES      (NUM_LANES),
            .NUM_BANKS      (NUM_BANKS),
            .WORD_SIZE      (WORD_SIZE),
            .ADDR_WIDTH     (ADDR_WIDTH),
            .TAG_WIDTH      (TAG_WIDTH),
            .PAYLOAD_WIDTH  (1),
            .ATTR_WIDTH     (ATTR_WIDTH),
            .NUM_BLOCKS     (NUM_BLOCKS),
            .RSP_QUEUE_SIZE (RSP_QUEUE_SIZE)
        ) engine_b (
            .clk          (clk),
            .reset        (reset),
            .req_valid    (b_req_valid),
            .req_block    (b_req_block),
            .req_rw       (1'b0),
            .req_read     (1'b1),
            .req_amo      (1'b0),
            .req_attr     ('0),
            .req_mask     (in_mask[b_req_block]),
            .req_addr     (in_addr[b_req_block]),
            .req_payload  (b_req_payload),
            .req_tag      (in_tag[b_req_block]),
            .req_ready    (b_req_ready),
            .busy         (b_busy),
            .holds_write  (b_holds_write),
            .pass_en      (pass_en),
            .pass_amo     (b_pass_amo),
            .bank_valid   (b_bank_valid),
            .bank_rw      (b_bank_rw),
            .bank_amo     (b_bank_amo),
            .bank_attr    (b_bank_attr),
            .bank_addr    (b_bank_addr),
            .bank_payload (b_bank_payload),
            .bank_rdata   (b_bank_rdata),
            .rsp_valid    (b_rsp_valid),
            .rsp_block    (b_rsp_block),
            .rsp_mask     (b_rsp_mask),
            .rsp_data     (b_rsp_data),
            .rsp_tag      (b_rsp_tag),
            .rsp_take     (b_rsp_take),
            .rsp_done     (b_rsp_done),
            .perf_served  (b_perf_served),
            .perf_store   (b_perf_store),
            .perf_replay  (b_perf_replay)
        );
        `UNUSED_VAR ({b_bank_rw, b_bank_amo, b_holds_write, b_pass_amo, b_bank_attr, b_bank_payload, b_perf_store})

    end else begin : g_no_read_engine

        assign b_busy        = 1'b0;
        assign b_bank_valid  = '0;
        assign b_bank_addr   = '0;
        assign b_rsp_valid   = 1'b0;
        assign b_rsp_block   = '0;
        assign b_rsp_mask    = '0;
        assign b_rsp_data    = '0;
        assign b_rsp_tag     = '0;
        assign b_perf_served = '0;
        assign b_perf_replay = 1'b0;
        `UNUSED_VAR (b_bank_rdata)
        `UNUSED_VAR (b_rsp_take)
        `UNUSED_VAR (b_rsp_done)
        `UNUSED_VAR (a_holds_write)
        `UNUSED_VAR (b_req_valid)
        `UNUSED_VAR (b_req_block)
        assign b_req_ready = 1'b0;

    end

    // Responses. Each block has a response register, which takes the first
    // engine's queue head ahead of the read engine's when it is empty or is
    // being emptied. A response's queue slot is free again once its block has
    // taken it on.

    wire [NUM_BLOCKS-1:0] a_load;
    wire [NUM_BLOCKS-1:0] b_load;
    wire [NUM_BLOCKS-1:0] a_done;
    wire [NUM_BLOCKS-1:0] b_done;

    for (genvar k = 0; k < NUM_BLOCKS; ++k) begin : g_rsp
        reg                                 rsp_valid_r;
        reg                                 rsp_from_b_r;
        reg [NUM_LANES-1:0]                 rsp_mask_r;
        reg [NUM_LANES-1:0][WORD_WIDTH-1:0] rsp_data_r;
        reg [TAG_WIDTH-1:0]                 rsp_tag_r;

        wire rsp_fire = rsp_valid_r && lsu_mem_if[k].rsp_ready;
        wire rsp_free = ~rsp_valid_r || lsu_mem_if[k].rsp_ready;
        wire a_hit    = a_rsp_valid && (a_rsp_block == BLOCK_WIDTH'(k));
        wire b_hit    = b_rsp_valid && (b_rsp_block == BLOCK_WIDTH'(k));

        assign a_load[k] = rsp_free && a_hit;
        assign b_load[k] = rsp_free && ~a_hit && b_hit;
        assign a_done[k] = rsp_fire && ~rsp_from_b_r;
        assign b_done[k] = rsp_fire && rsp_from_b_r;

        always @(posedge clk) begin
            if (reset) begin
                rsp_valid_r <= 1'b0;
            end else if (rsp_free) begin
                rsp_valid_r <= a_hit || b_hit;
            end
        end

        // The response is read only with rsp_valid_r, so on ASIC it loads only
        // when one is captured.
        wire rsp_load = rsp_free && (a_hit || b_hit);
    `ifdef ASIC
        wire rsp_clk;
        VX_clockgate rsp_cg (.clk_in (clk), .en (rsp_load), .clk_out (rsp_clk));
        always @(posedge rsp_clk) begin
            if (rsp_load) begin
                rsp_from_b_r <= ~a_hit;
                rsp_mask_r   <= a_hit ? a_rsp_mask : b_rsp_mask;
                rsp_data_r   <= a_hit ? a_rsp_data : b_rsp_data;
                rsp_tag_r    <= a_hit ? a_rsp_tag  : b_rsp_tag;
            end
        end
    `else
        `UNUSED_VAR (rsp_load)
        always @(posedge clk) begin
            if (rsp_free) begin
                rsp_from_b_r <= ~a_hit;
                rsp_mask_r   <= a_hit ? a_rsp_mask : b_rsp_mask;
                rsp_data_r   <= a_hit ? a_rsp_data : b_rsp_data;
                rsp_tag_r    <= a_hit ? a_rsp_tag  : b_rsp_tag;
            end
        end
    `endif

        assign lsu_mem_if[k].rsp_valid     = rsp_valid_r;
        assign lsu_mem_if[k].rsp_data.mask = rsp_mask_r;
        assign lsu_mem_if[k].rsp_data.data = rsp_data_r;
        assign lsu_mem_if[k].rsp_data.tag  = rsp_tag_r;
    end

    `POP_COUNT(a_rsp_done, a_done);
    `POP_COUNT(b_rsp_done, b_done);
    assign a_rsp_take = (| a_load);
    assign b_rsp_take = (| b_load);

    // DMA and atomic slots

    reg [1:0] amo_stall_r;
    wire amo_stall = (amo_stall_r != 2'd0);
    always @(posedge clk) begin
        if (reset) begin
            amo_stall_r <= 2'd0;
        end else if (a_pass_amo) begin
            amo_stall_r <= 2'd2;
        end else if (amo_stall) begin
            amo_stall_r <= amo_stall_r - 2'd1;
        end
    end

    wire dma_valid = DMA_ENABLE && dma_bus_if.req_valid;
    reg  dma_slot_r;
    always @(posedge clk) begin
        if (reset) begin
            dma_slot_r <= 1'b0;
        end else begin
            dma_slot_r <= dma_valid && ~amo_stall;
        end
    end
    assign pass_en = ~dma_valid && ~amo_stall;

    // DMA port handshake
    //   rw=0 reads  : accepted when the response pipe-buffer has space.
    //   rw=1 writes : no response issued.

    wire dma_rsp_buf_ready;
    wire dma_fire = dma_slot_r && dma_valid && (dma_bus_if.req_data.rw || dma_rsp_buf_ready);
    wire dma_rd_fire = dma_fire && ~dma_bus_if.req_data.rw;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] per_bank_rdata_a;

    if (DMA_ENABLE) begin : g_dma_enable
        `UNUSED_VAR (dma_bus_if.req_data.attr)

        assign dma_bus_if.req_ready = dma_slot_r && (dma_bus_if.req_data.rw || dma_rsp_buf_ready);

        // the tag follows the SRAM read to its output register
        VX_pipe_buffer #(
            .DATAW (DMA_TAG_WIDTH)
        ) dma_rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (dma_rd_fire),
            .ready_in  (dma_rsp_buf_ready),
            .data_in   (dma_bus_if.req_data.tag),
            .valid_out (dma_bus_if.rsp_valid),
            .data_out  (dma_bus_if.rsp_data.tag),
            .ready_out (dma_bus_if.rsp_ready)
        );

        // The response buffer holds only the tag; the data is the live SRAM
        // output, which the next pass's reads overwrite. While a response is
        // back-pressured, serve a copy of the row latched on its first cycle.
        reg [NUM_BANKS-1:0][WORD_WIDTH-1:0] dma_rsp_hold_data_r;
        reg                                 dma_rsp_hold_valid_r;
        wire dma_rsp_consumed = dma_bus_if.rsp_valid && dma_bus_if.rsp_ready;
        always @(posedge clk) begin
            if (reset) begin
                dma_rsp_hold_valid_r <= 1'b0;
            end else if (dma_rsp_consumed) begin
                dma_rsp_hold_valid_r <= 1'b0;
            end else if (dma_bus_if.rsp_valid && ~dma_rsp_hold_valid_r) begin
                dma_rsp_hold_data_r  <= per_bank_rdata_a;
                dma_rsp_hold_valid_r <= 1'b1;
            end
        end
        for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_dma_rsp_data
            assign dma_bus_if.rsp_data.data[i*WORD_WIDTH +: WORD_WIDTH] =
                dma_rsp_hold_valid_r ? dma_rsp_hold_data_r[i] : per_bank_rdata_a[i];
        end

    end else begin : g_no_dma
        assign dma_rsp_buf_ready    = 1'b0;
        assign dma_bus_if.req_ready = 1'b0;
        assign dma_bus_if.rsp_valid = 1'b0;
        assign dma_bus_if.rsp_data  = '0;
        `UNUSED_VAR (dma_bus_if.req_valid)
        `UNUSED_VAR (dma_bus_if.req_data)
        `UNUSED_VAR (dma_bus_if.rsp_ready)
    end

    // banks

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_data_store

        wire dma_wr_b = dma_fire && dma_bus_if.req_data.rw
                     && (| dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]);

        wire lsu_active = a_bank_valid[i];

        wire [WORD_WIDTH-1:0]    req_data;
        wire [WORD_SIZE-1:0]     req_byteen;
        wire [HART_ID_WIDTH-1:0] req_hart_id;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        wire [AMO_CMP_WIDTH-1:0] req_amo_cmp;
    `endif
        if (AMO_ENABLE != 0) begin : g_payload_amo
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            assign {req_hart_id, req_amo_cmp, req_byteen, req_data} = a_bank_payload[i];
        `else
            assign {req_hart_id, req_byteen, req_data} = a_bank_payload[i];
        `endif
        end else begin : g_payload
            assign {req_byteen, req_data} = a_bank_payload[i];
            assign req_hart_id = '0;
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            assign req_amo_cmp = '0;
        `endif
        end

        // Bank-facing AMO interface, tied off when the A extension is absent so
        // none of the atomic hardware is synthesized.
        wire                       amo_wb_store;  // owed atomic write-back
        wire [BANK_ADDR_WIDTH-1:0] amo_wb_addr;
        wire [WORD_SIZE-1:0]       amo_wb_byteen;
        wire [WORD_WIDTH-1:0]      amo_wb_data;
        wire                       sc_resolving;  // SC outcome replaces the response
        wire                       sc_fail;
        wire                       sc_lane_hi;    // outcome lane of a resolving .W SC

        if (AMO_ENABLE != 0) begin : g_amo

            // Atomics. An atomic carries rw=0, so the read returning the old
            // value is the ordinary read path. What the bank owes is the write
            // of the computed value.
            //
            // The old word is registered before the ALU sees it. Feeding the ALU
            // straight from the SRAM output puts the BRAM clock-to-out, the adder
            // and the BRAM setup in a single period; the register splits that
            // into BRAM-out-to-flop and flop-through-adder-to-BRAM-in.
            //
            // An atomic owns its bank for three cycles: read, capture, write back.
            reg                       amo_rd_valid_r;
            reg                       amo_wb_valid_r;
            reg [WORD_WIDTH-1:0]      amo_old_r;
            reg [BANK_ADDR_WIDTH-1:0] amo_wb_addr_r;
            reg [WORD_SIZE-1:0]       amo_wb_byteen_r;
            reg [AMO_OP_WIDTH-1:0]    amo_wb_op_r;
            reg                       amo_wb_unsigned_r;
            reg [WORD_WIDTH-1:0]      amo_wb_rhs_r;
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            reg [AMO_CMP_WIDTH-1:0]   amo_wb_cmp_r;
        `endif

            wire [AMO_OP_WIDTH-1:0] req_amo_op = a_bank_attr[1 +: AMO_OP_WIDTH];
            wire req_amo_unsigned = a_bank_attr[0];

            wire amo_accept = lsu_active && a_bank_amo;

            // Reservation commit stage. Every accepted request that can touch a
            // reservation -- an atomic, or a plain store -- reaches it one cycle
            // after acceptance, so a single address port serves both. The
            // look-ahead read is driven from the acceptance cycle, which is what
            // makes the registered reservation entry land here.
            wire is_lr_in = a_bank_amo && (amo_op_e'(req_amo_op) == AMO_OP_LR);
            wire is_sc_in = a_bank_amo && (amo_op_e'(req_amo_op) == AMO_OP_SC);

            reg                       res_valid_r;
            reg                       res_is_lr_r;
            reg                       res_is_sc_r;
            reg                       res_is_amo_r;
            reg                       res_is_store_r;
            reg [BANK_ADDR_WIDTH-1:0] res_addr_r;
            reg [HART_ID_WIDTH-1:0]   res_hart_id_r;

            always @(posedge clk) begin
                if (reset) begin
                    res_valid_r <= 1'b0;
                end else begin
                    res_valid_r <= lsu_active;
                end
                if (lsu_active) begin
                    res_is_lr_r    <= is_lr_in;
                    res_is_sc_r    <= is_sc_in;
                    res_is_amo_r   <= a_bank_amo;
                    res_is_store_r <= a_bank_rw;
                    res_addr_r     <= a_bank_addr[i];
                    res_hart_id_r  <= req_hart_id;
                end
            end

            // A store-conditional fails when its reservation is gone. The
            // outcome is known at the commit stage, one cycle before the
            // write-back, so it can both suppress the write and be returned as
            // this request's response.
            wire res_check;
            assign sc_fail = res_is_sc_r && ~res_check;
            assign sc_resolving = res_valid_r && res_is_sc_r;

            // The SC outcome must land in the addressed lane: the LSU extracts
            // the subword by address, so a .W outcome left at bit 0 would read
            // as 0 ("success") for every upper-lane SC, including a failed one
            // whose store was suppressed.
            if (WORD_WIDTH == 64) begin : g_sc_lane
                reg res_lane_hi_r;
                always @(posedge clk) begin
                    if (lsu_active) begin
                        res_lane_hi_r <= ~req_byteen[0];
                    end
                end
                assign sc_lane_hi = res_lane_hi_r;
            end else begin : g_sc_lane0
                assign sc_lane_hi = 1'b0;
            end

            // Everything that ends up writing the word: a plain store, a
            // read-modify-write atomic (which carries rw=0 and so is not a store
            // by the request flag), and a store-conditional that found its
            // reservation.
            wire res_commits_store = res_is_store_r
                                  || (res_is_amo_r && ~res_is_lr_r && ~res_is_sc_r)
                                  || (res_is_sc_r && res_check);

            reg  amo_wb_is_lr_r;
            reg  amo_wb_sc_fail_r;
            always @(posedge clk) begin
                if (amo_rd_valid_r) begin
                    amo_wb_is_lr_r   <= res_is_lr_r;
                    amo_wb_sc_fail_r <= sc_fail;
                end
            end

            // A load-reserved leaves the word alone, and a failed
            // store-conditional must not store; every other atomic writes its
            // computed value back.
            assign amo_wb_store = amo_wb_valid_r && ~amo_wb_is_lr_r && ~amo_wb_sc_fail_r;

            always @(posedge clk) begin
                if (reset) begin
                    amo_rd_valid_r <= 1'b0;
                    amo_wb_valid_r <= 1'b0;
                end else begin
                    amo_rd_valid_r <= amo_accept;
                    amo_wb_valid_r <= amo_rd_valid_r;
                end
                if (amo_rd_valid_r) begin
                    amo_old_r <= per_bank_rdata_a[i];
                end
                if (amo_accept) begin
                    amo_wb_addr_r     <= a_bank_addr[i];
                    amo_wb_byteen_r   <= req_byteen;
                    amo_wb_op_r       <= req_amo_op;
                    amo_wb_unsigned_r <= req_amo_unsigned;
                    amo_wb_rhs_r      <= req_data;
                `ifdef VX_CFG_EXT_ZACAS_ENABLE
                    amo_wb_cmp_r      <= req_amo_cmp;
                `endif
                end
            end

            wire [63:0] amo_new_word;
            wire [63:0] amo_ret_word;

            // Lane-align the RMW operands. The bank word is XLEN wide, so on a
            // 64-bit hart a .W atomic addresses one of two 32-bit lanes; the
            // ALU computes right-aligned. Store data arrives lane-positioned
            // (same packing as a plain store), so old and rhs shift down by the
            // lane offset and the result shifts back up under the request
            // byteen. The comparand arrives right-aligned from rs3 and needs no
            // shift. The addressed lane is recovered from the byteen: an atomic
            // is naturally aligned, so a .W to the upper lane never covers byte
            // 0 and a .D covers the full word.
            wire [63:0] amo_old_lane;
            wire [63:0] amo_rhs_lane;
            wire [1:0]  amo_width;
            if (WORD_WIDTH == 64) begin : g_amo_lane
                wire lane_hi = ~amo_wb_byteen_r[0];
                assign amo_width    = (&amo_wb_byteen_r) ? 2'd3 : 2'd2;
                assign amo_old_lane = lane_hi ? {32'h0, amo_old_r[63:32]} : amo_old_r;
                assign amo_rhs_lane = lane_hi ? {32'h0, amo_wb_rhs_r[63:32]} : amo_wb_rhs_r;
                assign amo_wb_data  = lane_hi ? {amo_new_word[31:0], 32'h0} : amo_new_word;
            end else begin : g_amo_word
                assign amo_width    = 2'd2;
                assign amo_old_lane = 64'(amo_old_r);
                assign amo_rhs_lane = 64'(amo_wb_rhs_r);
                assign amo_wb_data  = amo_new_word[WORD_WIDTH-1:0];
            end
            // VX_CFG_AMO_RS_SIZE sizes ONE shared LLC bank's contention; deployed
            // per lmem bank it multiplies by NUM_LMEM_BANKS x NUM_CORES. After
            // the holder-credit protocol a hot word maps to exactly one station,
            // so depth only reduces false eviction between distinct reserved
            // words aliasing in one bank -- and a spurious SC failure is
            // architecturally legal. Clamp the depth.
            // A credit is spent per refused LR, not per cycle: a warp's LR
            // reaches a bank one lane per pass, so between a holder's LR and its
            // SC every other hart of the core can be refused, and a refused hart
            // can retry once more. The budget covers two refusals per hart; a
            // smaller one lets one warp's own LR burst displace its holder and a
            // contended retry loop stops making progress.
            VX_amo_unit #(
                .NUM_RES_ENTRIES  (`MIN(`VX_CFG_AMO_RS_SIZE, 4)),
                .HOLD_CREDIT_BITS (`CLOG2(`VX_CFG_NUM_WARPS * `VX_CFG_NUM_THREADS) + 1),
                .LINE_ADDR_BITS   (BANK_ADDR_WIDTH),
                .DATA_WIDTH       (WORD_WIDTH)
            ) amo_unit (
                .clk              (clk),
                .reset            (reset),
                .pipe_stall       (1'b0),
                .compute_op       (amo_op_e'(amo_wb_op_r)),
                .compute_unsigned (amo_wb_unsigned_r),
                .compute_width    (amo_width),
                .compute_old      (amo_old_lane),
                .compute_rhs      (amo_rhs_lane),
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                .compute_cmp      (64'(amo_wb_cmp_r)),
            `else
                .compute_cmp      (64'b0),
            `endif
                .compute_new_word (amo_new_word),
                .compute_ret_word (amo_ret_word),
                // A load-reserved claims the word; a store-conditional gives up
                // its own reservation whether it succeeds or fails; any committing
                // store breaks whoever else holds one.
                .res_reserve      (res_valid_r && res_is_lr_r),
                .res_clear        (res_valid_r && res_is_sc_r),
                .res_invalidate   (res_valid_r && res_commits_store),
                .res_hart_id      (res_hart_id_r),
                .res_line_addr    (res_addr_r),
                .res_line_addr_n  (a_bank_addr[i]),
                .res_check        (res_check)
            );
            `UNUSED_VAR (amo_ret_word)
            if (WORD_WIDTH < 64) begin : g_amo_hi_unused
                `UNUSED_VAR (amo_new_word[63:WORD_WIDTH])
            end

            assign amo_wb_addr   = amo_wb_addr_r;
            assign amo_wb_byteen = amo_wb_byteen_r;

        end else begin : g_no_amo
            assign amo_wb_store  = 1'b0;
            assign amo_wb_addr   = '0;
            assign amo_wb_byteen = '0;
            assign amo_wb_data   = '0;
            assign sc_resolving  = 1'b0;
            assign sc_fail       = 1'b0;
            assign sc_lane_hi    = 1'b0;
            `UNUSED_VAR (req_hart_id)
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            `UNUSED_VAR (req_amo_cmp)
        `endif
        end

        // Port A address / write-data / write-enable mux: DMA, an owed atomic
        // write-back, or the first engine's pass. They never coincide.
        wire [BANK_ADDR_WIDTH-1:0] bank_sram_addr;
        wire [WORD_WIDTH-1:0]      bank_sram_wdata;
        wire [WORD_SIZE-1:0]       bank_sram_wren;

        assign bank_sram_addr  = dma_fire     ? BANK_ADDR_WIDTH'(dma_bus_if.req_data.addr)
                               : amo_wb_store ? amo_wb_addr
                                              : a_bank_addr[i];
        assign bank_sram_wdata = dma_fire     ? dma_bus_if.req_data.data[i*WORD_WIDTH +: WORD_WIDTH]
                               : amo_wb_store ? amo_wb_data
                                              : req_data;
        assign bank_sram_wren  = dma_fire     ? dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]
                               : amo_wb_store ? amo_wb_byteen
                                              : req_byteen;

        wire sram_read  = dma_rd_fire || (lsu_active && ~a_bank_rw);
        wire sram_write = dma_wr_b || amo_wb_store || (lsu_active && a_bank_rw);

        if (DUAL) begin : g_dual_port
            VX_1rw1r_ram #(
                .DATAW (WORD_WIDTH),
                .SIZE  (WORDS_PER_BANK),
                .WRENW (WORD_SIZE)
            ) lmem_store (
                .clk     (clk),
                .reset   (reset),
                .read_a  (sram_read),
                .write_a (sram_write),
                .wren_a  (bank_sram_wren),
                .addr_a  (bank_sram_addr),
                .wdata_a (bank_sram_wdata),
                .rdata_a (per_bank_rdata_a[i]),
                .read_b  (b_bank_valid[i]),
                .addr_b  (b_bank_addr[i]),
                .rdata_b (b_bank_rdata[i])
            );
        end else begin : g_single_port
            VX_sp_ram #(
                .DATAW    (WORD_WIDTH),
                .SIZE     (WORDS_PER_BANK),
                .WRENW    (WORD_SIZE),
                .OUT_REG  (1),
                .RDW_MODE ("R")
            ) lmem_store (
                .clk   (clk),
                .reset (reset),
                .read  (sram_read),
                .write (sram_write),
                .wren  (bank_sram_wren),
                .addr  (bank_sram_addr),
                .wdata (bank_sram_wdata),
                .rdata (per_bank_rdata_a[i])
            );
            assign b_bank_rdata[i] = '0;
            `UNUSED_VAR (b_bank_valid[i])
            `UNUSED_VAR (b_bank_addr[i])
        end

        // A store-conditional returns its outcome, not the word it read: zero
        // for success, one for failure, positioned in the addressed lane for
        // the LSU's subword extraction. The outcome is resolved at the commit
        // stage, the cycle the response takes the SRAM output.
        assign a_bank_rdata[i] = sc_resolving
                               ? (sc_lane_hi ? (WORD_WIDTH'(sc_fail) << (WORD_WIDTH/2))
                                             : WORD_WIDTH'(sc_fail))
                               : per_bank_rdata_a[i];
    end

    if (AMO_ENABLE == 0) begin : g_no_amo_attr
        `UNUSED_VAR ({a_bank_amo, a_bank_attr})
    end

`ifdef PERF_ENABLE
    wire [`CLOG2(NUM_LANES+1)-1:0] perf_a_served;
    wire [`CLOG2(NUM_LANES+1)-1:0] perf_b_served;
    `POP_COUNT(perf_a_served, a_perf_served);
    `POP_COUNT(perf_b_served, b_perf_served);

    wire [NUM_BLOCKS-1:0] perf_crsp_stall_per_blk;
    for (genvar k = 0; k < NUM_BLOCKS; ++k) begin : g_perf_crsp
        assign perf_crsp_stall_per_blk[k] = lsu_mem_if[k].rsp_valid && ~lsu_mem_if[k].rsp_ready;
    end
    wire [`CLOG2(NUM_BLOCKS+1)-1:0] perf_crsp_stalls_per_cycle;
    `POP_COUNT(perf_crsp_stalls_per_cycle, perf_crsp_stall_per_blk);

    reg [PERF_CTR_BITS-1:0] perf_reads;
    reg [PERF_CTR_BITS-1:0] perf_writes;
    reg [PERF_CTR_BITS-1:0] perf_bank_stalls;
    reg [PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_reads       <= '0;
            perf_writes      <= '0;
            perf_bank_stalls <= '0;
            perf_crsp_stalls <= '0;
        end else begin
            perf_reads       <= perf_reads + PERF_CTR_BITS'(a_perf_store ? 0 : perf_a_served) + PERF_CTR_BITS'(perf_b_served);
            perf_writes      <= perf_writes + PERF_CTR_BITS'(a_perf_store ? perf_a_served : 0);
            perf_bank_stalls <= perf_bank_stalls + PERF_CTR_BITS'(a_perf_replay) + PERF_CTR_BITS'(b_perf_replay);
            perf_crsp_stalls <= perf_crsp_stalls + PERF_CTR_BITS'(perf_crsp_stalls_per_cycle);
        end
    end

    assign lmem_perf.reads       = perf_reads;
    assign lmem_perf.writes      = perf_writes;
    assign lmem_perf.bank_stalls = perf_bank_stalls;
    assign lmem_perf.crsp_stalls = perf_crsp_stalls;
`else
    `UNUSED_VAR ({a_perf_served, a_perf_store, a_perf_replay, b_perf_served, b_perf_replay})
`endif

`ifdef DBG_TRACE_MEM
    for (genvar k = 0; k < NUM_BLOCKS; ++k) begin : g_trace
        always @(posedge clk) begin
            if (lsu_mem_if[k].req_valid && lsu_mem_if[k].req_ready) begin
                `TRACE(2, ("%t: %s req[%0d]: rw=%b, mask=%b, tag=0x%0h\n",
                    $time, INSTANCE_ID, k, lsu_mem_if[k].req_data.rw, lsu_mem_if[k].req_data.mask, lsu_mem_if[k].req_data.tag))
            end
            if (lsu_mem_if[k].rsp_valid && lsu_mem_if[k].rsp_ready) begin
                `TRACE(2, ("%t: %s rsp[%0d]: mask=%b, tag=0x%0h\n",
                    $time, INSTANCE_ID, k, lsu_mem_if[k].rsp_data.mask, lsu_mem_if[k].rsp_data.tag))
            end
        end
    end
`endif

endmodule

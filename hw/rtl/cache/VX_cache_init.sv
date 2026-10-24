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

// Flush sequencer. Flush tokens (requests with the flush attribute) at any
// input start one walk: every input is blocked, in-flight requests drain and
// every bank writes back and invalidates its lines. With FLUSH_FORWARD, the
// cache then sends one token down each memory port, behind that port's
// write-backs, and waits for every port to answer, so the next level has
// each port's data before it releases that port's token. Finally each token
// visible at the end of the walk is answered locally, one at a time through
// the cache's own response path, and the inputs reopen.
// A token that arrives during the walk is visible only after all of its
// input's earlier requests were accepted, so the walk covered them.
// A request crossbar that can hold requests apart from their input order
// (XBAR_SIZE != 0) drains into the banks first, while the inputs are held,
// so no earlier request of a token's input is passed by a later one.
// An L1-only token (any byteen bit set: a next level with wider words sees
// the bit at the token's line offset) is forwarded with its kind. A
// last-level cache that alone serves every memory client is the point of
// coherence for it: once the token is visible, its input's earlier requests
// are inside the cache, ahead of anything the next kernel sends, so it answers
// the token without a walk. Any other last-level cache still walks, since a
// replicated instance or a client reaching memory beside it may hold or need
// lines it has not written back.
// An IO token (the io attribute: a fence ordering only device accesses) walks
// nowhere. Once nothing is held between the cache's inputs and its memory
// ports, on the cacheable and the non-cacheable path alike, it goes down every
// memory port, so its answer proves the earlier IO writes reached memory.
// Kinds visible together combine to the weakest one that serves them all.
module VX_cache_init import VX_gpu_pkg::*; #(
    parameter NUM_REQS         = 4,              // Number of Word requests per cycle
    parameter NUM_BANKS        = 1,              // Number of banks
    parameter MEM_PORTS        = 1,              // Number of memory ports
    parameter FLUSH_FORWARD    = 1,              // Forward the flush to the next level
    parameter SHARED_LLC       = 0,              // The LLC serving every memory client alone: answers L1-only tokens without a walk
    parameter TAG_WIDTH        = UUID_WIDTH + 1, // core request tag size
    parameter XBAR_SIZE        = 0,              // Requests the request crossbar can hold apart from input order
    parameter REQ_SEL_WIDTH    = `UP(`CLOG2(NUM_REQS))
) (
    input wire                        clk,
    input wire                        reset,
    VX_mem_bus_if.slave               core_bus_in_if [NUM_REQS],
    VX_mem_bus_if.master              core_bus_out_if [NUM_REQS],
    input wire [NUM_REQS-1:0]         xbar_req_fire,
    input wire [NUM_BANKS-1:0]        bank_req_fire,
    output wire [NUM_BANKS-1:0]       flush_begin,
    output wire                       flush_skip,    // the banks only quiesce: no walk is owed
    output wire [`UP(UUID_WIDTH)-1:0] flush_uuid,
    input wire [NUM_BANKS-1:0]        flush_end,
    // flush tokens to the next level
    input wire                        mem_req_empty, // no write-back left between the banks and the ports
    input wire                        nc_req_empty,  // no request held between the inputs and the memory side
    output wire [MEM_PORTS-1:0]       mem_flush_valid,
    output wire                       mem_flush_l1_only,
    output wire                       mem_flush_io,
    input wire [MEM_PORTS-1:0]        mem_flush_ready,
    output wire                       mem_flush_rsp_phase,
    input wire [MEM_PORTS-1:0]        mem_flush_rsp_valid,
    // answers to the requesters' tokens
    output wire                       flush_rsp_valid,
    output wire [REQ_SEL_WIDTH-1:0]   flush_rsp_idx,
    output wire [TAG_WIDTH-1:0]       flush_rsp_tag,
    input wire                        flush_rsp_ready
);
    localparam STATE_IDLE    = 0;
    localparam STATE_WAIT1   = 1;
    localparam STATE_FLUSH   = 2;
    localparam STATE_WAIT2   = 3;
    localparam STATE_FORWARD = 4;
    localparam STATE_RESPOND = 5;

    reg [2:0] state, state_n;

    // track requests inside the request crossbar

    wire no_inflight_reqs;

    if (XBAR_SIZE != 0) begin : g_xbar_inflight

        localparam REQS_W  = `CLOG2(NUM_REQS+1);
        localparam INCRW   = `CLOG2(`MIN(NUM_REQS, XBAR_SIZE)+1);
        localparam BANKS_W = `CLOG2(NUM_BANKS+1);

        wire [REQS_W-1:0]  xbar_req_cnt;
        wire [BANKS_W-1:0] bank_req_cnt;

        `POP_COUNT(xbar_req_cnt, xbar_req_fire);
        `POP_COUNT(bank_req_cnt, bank_req_fire);

        VX_pending_size #(
            .SIZE  (XBAR_SIZE),
            .INCRW (INCRW),
            .DECRW (BANKS_W)
        ) pending_size (
            .clk   (clk),
            .reset (reset),
            .incr  (INCRW'(xbar_req_cnt)),
            .decr  (bank_req_cnt),
            .empty (no_inflight_reqs),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

    end else begin : g_no_xbar_inflight
        assign no_inflight_reqs = 0;
        `UNUSED_VAR ({xbar_req_fire, bank_req_fire})
    end

    wire [NUM_REQS-1:0] flush_req_mask;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_flush_req_mask
        assign flush_req_mask[i] = core_bus_in_if[i].req_valid && core_bus_in_if[i].req_data.attr[MEM_ATTR_FLUSH_OFFS];
    end
    wire flush_req_enable = (| flush_req_mask);

    wire [NUM_REQS-1:0] flush_l1_mask, flush_io_mask;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_flush_kind_mask
        assign flush_io_mask[i] = flush_req_mask[i] && core_bus_in_if[i].req_data.attr[MEM_ATTR_IO_OFFS];
        assign flush_l1_mask[i] = flush_req_mask[i] && ~flush_io_mask[i] && (| core_bus_in_if[i].req_data.byteen);
    end
    wire any_full = (| (flush_req_mask & ~flush_l1_mask & ~flush_io_mask));
    wire any_l1   = (| flush_l1_mask);
    wire any_io   = (| flush_io_mask);
    // every visible token is L1-only
    wire flush_l1_only = any_l1 && ~any_full && ~any_io;
    // a walk is owed to a full token, and to an L1-only one above the point of coherence
    wire flush_walk = any_full || (any_l1 && (SHARED_LLC == 0));

    reg [NUM_BANKS-1:0] flush_done, flush_done_n;
    reg                 fwd_go, fwd_go_n;   // the crossbar drained; only tokens enter it now
    reg [MEM_PORTS-1:0] fwd_sent, fwd_sent_n;
    reg [MEM_PORTS-1:0] fwd_done, fwd_done_n;
    reg [NUM_REQS-1:0]  rsp_pending, rsp_pending_n; // token still to be answered
    reg [`UP(UUID_WIDTH)-1:0] flush_uuid_r, flush_uuid_n;
    reg                 l1_only_r, l1_only_n;
    reg                 io_r, io_n;
    reg                 skip_r, skip_n;

    // answer the lowest pending input; its token stays at the input until then
    wire [NUM_REQS-1:0]      rsp_sel_onehot;
    wire [REQ_SEL_WIDTH-1:0] rsp_sel_idx;
    wire                     rsp_sel_valid;
    VX_priority_encoder #(
        .N (NUM_REQS)
    ) rsp_sel (
        .data_in    (rsp_pending),
        .onehot_out (rsp_sel_onehot),
        .index_out  (rsp_sel_idx),
        .valid_out  (rsp_sel_valid)
    );
    assign flush_rsp_valid = (state == STATE_RESPOND) && rsp_sel_valid;
    assign flush_rsp_idx   = rsp_sel_idx;
    wire flush_rsp_fire = flush_rsp_valid && flush_rsp_ready;

    wire [NUM_REQS-1:0][TAG_WIDTH-1:0] core_bus_in_tag;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_in_tag
        assign core_bus_in_tag[i] = core_bus_in_if[i].req_data.tag;
    end
    VX_onehot_mux #(
        .DATAW (TAG_WIDTH),
        .N     (NUM_REQS)
    ) rsp_tag_mux (
        .data_in  (core_bus_in_tag),
        .sel_in   (rsp_sel_onehot),
        .data_out (flush_rsp_tag)
    );

    wire [NUM_REQS-1:0] dbg_stall;
`ifdef DBG_FLUSH_STALL
    // Random input backpressure, a new mask every 128 cycles (each input
    // stalled half the time), to widen ordering windows under test.
    reg [31:0] dbg_lfsr_r;
    reg [6:0]  dbg_cnt_r;
    reg [NUM_REQS-1:0] dbg_stall_r;
    always @(posedge clk) begin
        if (reset) begin
            dbg_lfsr_r  <= 32'hACE1_2468;
            dbg_cnt_r   <= '0;
            dbg_stall_r <= '0;
        end else begin
            dbg_lfsr_r <= {dbg_lfsr_r[30:0], dbg_lfsr_r[31] ^ dbg_lfsr_r[21] ^ dbg_lfsr_r[1] ^ dbg_lfsr_r[0]};
            dbg_cnt_r  <= dbg_cnt_r + 7'd1;
            if (dbg_cnt_r == '0) begin
                dbg_stall_r <= (NUM_REQS > 1) ? NUM_REQS'(dbg_lfsr_r) : '0;
            end
        end
    end
    assign dbg_stall = dbg_stall_r;
`else
    assign dbg_stall = '0;
`endif

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_req
        // Normal requests stay blocked while any token is visible. A token is
        // consumed here when its answer is sent and never reaches the banks.
        wire take_token = flush_rsp_fire && rsp_sel_onehot[i];
        wire input_enable = ~flush_req_enable && ~dbg_stall[i];
        assign core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && input_enable;
        assign core_bus_out_if[i].req_data  = core_bus_in_if[i].req_data;
        assign core_bus_in_if[i].req_ready  = take_token || (core_bus_out_if[i].req_ready && input_enable);
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_in_rsp
        assign core_bus_in_if[i].rsp_valid = core_bus_out_if[i].rsp_valid;
        assign core_bus_in_if[i].rsp_data  = core_bus_out_if[i].rsp_data;
        assign core_bus_out_if[i].rsp_ready = core_bus_in_if[i].rsp_ready;
    end

    reg [NUM_REQS-1:0][`UP(UUID_WIDTH)-1:0] core_bus_in_uuid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_in_uuid
        if (UUID_WIDTH != 0) begin : g_uuid
            assign core_bus_in_uuid[i] = core_bus_in_if[i].req_data.tag.uuid;
        end else begin : g_no_uuid
            assign core_bus_in_uuid[i] = 0;
        end
    end

    // an IO token also waits for the non-cacheable path, which carries the IO
    // writes it orders
    wire fwd_ready = mem_req_empty && (~io_r || nc_req_empty);

    always @(*) begin
        state_n       = state;
        flush_done_n  = flush_done;
        fwd_go_n      = fwd_go;
        fwd_sent_n    = fwd_sent;
        fwd_done_n    = fwd_done;
        rsp_pending_n = rsp_pending & ~(rsp_sel_onehot & {NUM_REQS{flush_rsp_fire}});
        flush_uuid_n  = flush_uuid_r;
        l1_only_n     = l1_only_r;
        io_n          = io_r;
        skip_n        = skip_r;
        case (state)
            // STATE_IDLE:
            default: begin
                if (flush_req_enable) begin
                    if (XBAR_SIZE != 0) begin
                        state_n = STATE_WAIT1;
                    end else if (~flush_walk && ~any_io) begin
                        rsp_pending_n = flush_req_mask;
                        state_n = STATE_RESPOND;
                    end else begin
                        skip_n = ~flush_walk;
                        state_n = STATE_FLUSH;
                    end
                    for (integer i = NUM_REQS-1; i >= 0; --i) begin
                        if (flush_req_mask[i]) begin
                            flush_uuid_n = core_bus_in_uuid[i];
                        end
                    end
                end
            end
            STATE_WAIT1: begin
                if (no_inflight_reqs) begin
                    if (~flush_walk && ~any_io) begin
                        rsp_pending_n = flush_req_mask;
                        state_n = STATE_RESPOND;
                    end else begin
                        skip_n = ~flush_walk;
                        state_n = STATE_FLUSH;
                    end
                end
            end
            STATE_FLUSH: begin
                // generate a flush request pulse
                state_n = STATE_WAIT2;
            end
            STATE_WAIT2: begin
                // wait for all banks to finish flushing
                flush_done_n = flush_done | flush_end;
                if (flush_done_n == {NUM_BANKS{1'b1}}) begin
                    flush_done_n = '0;
                    // answer the tokens visible now; later ones start a new walk.
                    // A quiesce without a walk serves only the tokens owing none.
                    rsp_pending_n = skip_r ? (flush_io_mask | (SHARED_LLC ? flush_l1_mask : '0))
                                           : flush_req_mask;
                    l1_only_n = flush_l1_only && ~skip_r;
                    io_n = skip_r;
                    state_n = FLUSH_FORWARD ? STATE_FORWARD : STATE_RESPOND;
                end
            end
            STATE_FORWARD: begin
                fwd_go_n   = fwd_go | fwd_ready;
                fwd_sent_n = fwd_sent | (mem_flush_valid & mem_flush_ready);
                fwd_done_n = fwd_done | mem_flush_rsp_valid;
                if (fwd_done_n == {MEM_PORTS{1'b1}}) begin
                    fwd_go_n   = 1'b0;
                    fwd_sent_n = '0;
                    fwd_done_n = '0;
                    state_n = STATE_RESPOND;
                end
            end
            STATE_RESPOND: begin
                if (rsp_pending_n == '0) begin
                    state_n = STATE_IDLE;
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state       <= STATE_IDLE;
            flush_done  <= '0;
            fwd_go      <= 1'b0;
            fwd_sent    <= '0;
            fwd_done    <= '0;
            rsp_pending <= '0;
        end else begin
            state       <= state_n;
            flush_done  <= flush_done_n;
            fwd_go      <= fwd_go_n;
            fwd_sent    <= fwd_sent_n;
            fwd_done    <= fwd_done_n;
            rsp_pending <= rsp_pending_n;
        end
        flush_uuid_r <= flush_uuid_n;
        l1_only_r    <= l1_only_n;
        io_r         <= io_n;
        skip_r       <= skip_n;
    end

    assign flush_begin = {NUM_BANKS{state == STATE_FLUSH}};
    assign flush_skip  = skip_r;
    assign flush_uuid = flush_uuid_r;

    // The drain counter counts the tokens too, so the first one sent must not
    // take the others' valid away.
    assign mem_flush_valid     = {MEM_PORTS{(state == STATE_FORWARD) && (fwd_go || fwd_ready)}} & ~fwd_sent;
    assign mem_flush_rsp_phase = (state == STATE_FORWARD);
    assign mem_flush_l1_only   = l1_only_r;
    assign mem_flush_io        = io_r;

endmodule

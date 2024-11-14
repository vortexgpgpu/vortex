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

module VX_cache_flush #(
    // Number of Word requests per cycle
    parameter NUM_REQS  = 4,
    // Number of banks
    parameter NUM_BANKS = 1,
    // Request debug identifier
    parameter UUID_WIDTH = 0,
    // core request tag size
    parameter TAG_WIDTH = UUID_WIDTH + 1,
    // Bank select latency
    parameter BANK_SEL_LATENCY = 1
) (
    input wire              clk,
    input wire              reset,
    VX_mem_bus_if.slave     core_bus_in_if [NUM_REQS],
    VX_mem_bus_if.master    core_bus_out_if [NUM_REQS],
    input wire [NUM_BANKS-1:0] bank_req_fire,
    output wire [NUM_BANKS-1:0] flush_begin,
    output wire [`UP(UUID_WIDTH)-1:0] flush_uuid,
    input wire [NUM_BANKS-1:0] flush_end
);
    localparam STATE_IDLE  = 0;
    localparam STATE_WAIT1 = 1;
    localparam STATE_FLUSH = 2;
    localparam STATE_WAIT2 = 3;
    localparam STATE_DONE  = 4;

    reg [2:0] state, state_n;

    // track in-flight core requests

    wire no_inflight_reqs;

    if (BANK_SEL_LATENCY != 0) begin : g_bank_sel_latency

        localparam NUM_REQS_W  = `CLOG2(NUM_REQS+1);
        localparam NUM_BANKS_W = `CLOG2(NUM_BANKS+1);

        wire [NUM_REQS-1:0] core_bus_out_fire;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_fire
            assign core_bus_out_fire[i] = core_bus_out_if[i].req_valid && core_bus_out_if[i].req_ready;
        end

        wire [NUM_REQS_W-1:0] core_bus_out_cnt;
        wire [NUM_BANKS_W-1:0] bank_req_cnt;

        `POP_COUNT(core_bus_out_cnt, core_bus_out_fire);
        `POP_COUNT(bank_req_cnt, bank_req_fire);
        `UNUSED_VAR (core_bus_out_cnt)

        VX_pending_size #(
            .SIZE  (BANK_SEL_LATENCY * NUM_BANKS),
            .INCRW (NUM_BANKS_W),
            .DECRW (NUM_BANKS_W)
        ) pending_size (
            .clk   (clk),
            .reset (reset),
            .incr  (NUM_BANKS_W'(core_bus_out_cnt)),
            .decr  (bank_req_cnt),
            .empty (no_inflight_reqs),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

    end else begin : g_no_bank_sel_latency
        assign no_inflight_reqs = 0;
        `UNUSED_VAR (bank_req_fire)
    end

    reg [NUM_BANKS-1:0] flush_done, flush_done_n;

    wire [NUM_REQS-1:0] flush_req_mask;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_flush_req_mask
        assign flush_req_mask[i] = core_bus_in_if[i].req_valid && core_bus_in_if[i].req_data.flags[`MEM_REQ_FLAG_FLUSH];
    end
    wire flush_req_enable = (| flush_req_mask);

    reg [NUM_REQS-1:0] lock_released, lock_released_n;
    reg [`UP(UUID_WIDTH)-1:0] flush_uuid_r, flush_uuid_n;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_req
        wire input_enable = ~flush_req_enable || lock_released[i];
        assign core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && input_enable;
        assign core_bus_out_if[i].req_data  = core_bus_in_if[i].req_data;
        assign core_bus_in_if[i].req_ready  = core_bus_out_if[i].req_ready && input_enable;
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_in_rsp
        assign core_bus_in_if[i].rsp_valid  = core_bus_out_if[i].rsp_valid;
        assign core_bus_in_if[i].rsp_data   = core_bus_out_if[i].rsp_data;
        assign core_bus_out_if[i].rsp_ready = core_bus_in_if[i].rsp_ready;
    end

    reg [NUM_REQS-1:0][`UP(UUID_WIDTH)-1:0] core_bus_out_uuid;
    wire [NUM_REQS-1:0] core_bus_out_ready;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_uuid
        if (UUID_WIDTH != 0) begin : g_uuid
            assign core_bus_out_uuid[i] = core_bus_in_if[i].req_data.tag[TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin : g_no_uuid
            assign core_bus_out_uuid[i] = 0;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_ready
        assign core_bus_out_ready[i] = core_bus_out_if[i].req_ready;
    end

    always @(*) begin
        state_n = state;
        flush_done_n = flush_done;
        lock_released_n = lock_released;
        flush_uuid_n = flush_uuid_r;
        case (state)
            //STATE_IDLE:
            default: begin
                if (flush_req_enable) begin
                    state_n = (BANK_SEL_LATENCY != 0) ? STATE_WAIT1 : STATE_FLUSH;
                    for (integer i = NUM_REQS-1; i >= 0; --i) begin
                        if (flush_req_mask[i]) begin
                            flush_uuid_n = core_bus_out_uuid[i];
                        end
                    end
                end
            end
            STATE_WAIT1: begin
                if (no_inflight_reqs) begin
                    state_n = STATE_FLUSH;
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
                    state_n = STATE_DONE;
                    flush_done_n = '0;
                    // only release current flush requests
                    // and keep normal requests locked
                    lock_released_n = flush_req_mask;
                end
            end
            STATE_DONE: begin
                // wait until released flush requests are issued
                // when returning to IDLE state other requests will unlock
                lock_released_n = lock_released & ~core_bus_out_ready;
                if (lock_released_n == 0) begin
                    state_n = STATE_IDLE;
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
            flush_done <= '0;
            lock_released <= '0;
        end else begin
            state <= state_n;
            flush_done <= flush_done_n;
            lock_released <= lock_released_n;
        end
        flush_uuid_r <= flush_uuid_n;
    end

    assign flush_begin = {NUM_BANKS{state == STATE_FLUSH}};
    assign flush_uuid = flush_uuid_r;

endmodule

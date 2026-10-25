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

module VX_cache_init import VX_gpu_pkg::*; #(
    parameter NUM_REQS  = 4,                  // Number of Word requests per cycle
    parameter NUM_BANKS = 1,                  // Number of banks
    parameter TAG_WIDTH = UUID_WIDTH + 1,     // core request tag size
    parameter BANK_SEL_LATENCY = 1,           // Bank select latency
    parameter IS_LLC    = 0,                  // last-level cache
    parameter PORT_FENCE = 0                  // a flush ends with a fence on every memory port
) (
    input wire              clk,
    input wire              reset,
    VX_mem_bus_if.slave     core_bus_in_if [NUM_REQS],
    VX_mem_bus_if.master    core_bus_out_if [NUM_REQS],
    input wire [NUM_BANKS-1:0] bank_req_fire,
    output wire [NUM_BANKS-1:0] flush_begin,
    output wire [`UP(UUID_WIDTH)-1:0] flush_uuid,
    input wire [NUM_BANKS-1:0] flush_end,
    output wire             port_fence_begin,
    output wire             port_fence_is_fence,
    input wire              port_fence_end
);
    `UNUSED_PARAM (TAG_WIDTH)

    localparam STATE_IDLE  = 0;
    localparam STATE_WAIT1 = 1;
    localparam STATE_FLUSH = 2;
    localparam STATE_WAIT2 = 3;
    localparam STATE_DONE  = 4;
    localparam STATE_PORT1 = 5;
    localparam STATE_PORT2 = 6;

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
        assign flush_req_mask[i] = core_bus_in_if[i].req_valid && core_bus_in_if[i].req_data.attr[MEM_ATTR_FLUSH_OFFS];
    end
    wire flush_req_enable = (| flush_req_mask);

    // A fence orders memory at the device scope. The last-level cache, when
    // it is the one cache every core's requests pass, is the device's point of
    // coherence: there a fence only waits for the requests already admitted
    // to reach their banks, and passes without walking them.
    wire [NUM_REQS-1:0] fence_req_mask;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_fence_req_mask
        assign fence_req_mask[i] = core_bus_in_if[i].req_data.attr[MEM_ATTR_FENCE_OFFS];
    end
    wire flush_is_fence = ((flush_req_mask & ~fence_req_mask) == '0);
    wire fence_pass = (IS_LLC != 0) && LLC_IS_COHERENT && flush_is_fence;

    reg [NUM_REQS-1:0] lock_released, lock_released_n;
    reg [`UP(UUID_WIDTH)-1:0] flush_uuid_r, flush_uuid_n;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_req
        // block core requests if a flush is active,
        // unless the lock for this specific request has been released
        wire input_enable = ~flush_req_enable || lock_released[i];

        assign core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && input_enable;
        assign core_bus_in_if[i].req_ready  = core_bus_out_if[i].req_ready && input_enable;
        // Once the ports' fences have carried a flush down, the released
        // request only waits for its response: it goes on as a plain read.
        always @(*) begin
            core_bus_out_if[i].req_data = core_bus_in_if[i].req_data;
            if (PORT_FENCE != 0) begin
                core_bus_out_if[i].req_data.attr[MEM_ATTR_FLUSH_OFFS] = 1'b0;
                core_bus_out_if[i].req_data.attr[MEM_ATTR_FENCE_OFFS] = 1'b0;
            end
        end
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
            assign core_bus_out_uuid[i] = core_bus_in_if[i].req_data.tag.uuid;
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
            // STATE_IDLE:
            default: begin
                if (flush_req_enable) begin
                    if (BANK_SEL_LATENCY != 0) begin
                        state_n = STATE_WAIT1;
                    end else if (fence_pass) begin
                        state_n = STATE_DONE;
                        lock_released_n = flush_req_mask;
                    end else begin
                        state_n = STATE_FLUSH;
                    end
                    for (integer i = NUM_REQS-1; i >= 0; --i) begin
                        if (flush_req_mask[i]) begin
                            flush_uuid_n = core_bus_out_uuid[i];
                        end
                    end
                end
            end
            STATE_WAIT1: begin
                if (no_inflight_reqs) begin
                    if (fence_pass) begin
                        state_n = STATE_DONE;
                        lock_released_n = flush_req_mask;
                    end else begin
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
                    if (PORT_FENCE != 0) begin
                        state_n = STATE_PORT1;
                    end else begin
                        state_n = STATE_DONE;
                        // only release current flush requests
                        // and keep normal requests locked
                        lock_released_n = flush_req_mask;
                    end
                end
            end
            STATE_PORT1: begin
                // send a fence on every memory port
                state_n = STATE_PORT2;
            end
            STATE_PORT2: begin
                // wait for every port's fence to come back
                if (port_fence_end) begin
                    state_n = STATE_DONE;
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

    // The next level treats the port fences as the requests at the inputs'
    // heads: as fences only if every one of those is a fence.
    assign port_fence_begin    = (state == STATE_PORT1);
    assign port_fence_is_fence = flush_is_fence;

endmodule

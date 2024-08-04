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

module VX_bank_flush #(
    parameter BANK_ID    = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE  = 64,
    // Number of banks
    parameter NUM_BANKS  = 1,
    // Number of associative ways
    parameter NUM_WAYS   = 1,
    // Enable cache writeback
    parameter WRITEBACK  = 0
) (
    input  wire clk,
    input  wire reset,
    input  wire flush_begin,
    output wire flush_end,
    output wire flush_init,
    output wire flush_valid,
    output wire [`CS_LINE_SEL_BITS-1:0] flush_line,
    output wire [NUM_WAYS-1:0] flush_way,
    input  wire flush_ready,
    input  wire mshr_empty,
    input  wire bank_empty
);
    // ways interation is only needed when eviction is enabled
    localparam CTR_WIDTH = `CS_LINE_SEL_BITS + (WRITEBACK ? `CS_WAY_SEL_BITS : 0);

    localparam STATE_IDLE  = 0;
    localparam STATE_INIT  = 1;
    localparam STATE_WAIT1 = 2;
    localparam STATE_FLUSH = 3;
    localparam STATE_WAIT2 = 4;
    localparam STATE_DONE  = 5;

    reg [2:0] state_r, state_n;

    reg [CTR_WIDTH-1:0] counter_r;

    always @(*) begin
        state_n = state_r;
        case (state_r)
            STATE_IDLE: begin
                if (flush_begin) begin
                    state_n = STATE_WAIT1;
                end
            end
            STATE_INIT: begin
                if (counter_r == ((2 ** `CS_LINE_SEL_BITS)-1)) begin
                    state_n = STATE_IDLE;
                end
            end
            STATE_WAIT1: begin
                // wait for pending requests to complete
                if (mshr_empty) begin
                    state_n = STATE_FLUSH;
                end
            end
            STATE_FLUSH: begin
                if (counter_r == ((2 ** CTR_WIDTH)-1) && flush_ready) begin
                    state_n = (BANK_ID == 0) ? STATE_DONE : STATE_WAIT2;
                end
            end
            STATE_WAIT2: begin
                // ensure the bank is empty before notifying the cache flush unit,
                // because the flush request to lower caches only goes through bank0
                // and it is important that request gets send out last.
                if (bank_empty) begin
                    state_n = STATE_DONE;
                end
            end
            STATE_DONE: begin
                // generate a completion pulse
                state_n = STATE_IDLE;
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state_r   <= STATE_INIT;
            counter_r <= '0;
        end else begin
            state_r <= state_n;
            if (state_r != STATE_IDLE) begin
                if ((state_r == STATE_INIT)
                || ((state_r == STATE_FLUSH) && flush_ready)) begin
                    counter_r <= counter_r + CTR_WIDTH'(1);
                end
            end else begin
                counter_r <= '0;
            end
        end
    end

    assign flush_end   = (state_r == STATE_DONE);
    assign flush_init  = (state_r == STATE_INIT);
    assign flush_valid = (state_r == STATE_FLUSH);
    assign flush_line  = counter_r[`CS_LINE_SEL_BITS-1:0];

    if (WRITEBACK && `CS_WAY_SEL_BITS > 0) begin
        reg [NUM_WAYS-1:0] flush_way_r;
        always @(*) begin
            flush_way_r = '0;
            flush_way_r[counter_r[`CS_LINE_SEL_BITS +: `CS_WAY_SEL_BITS]] = 1;
        end
        assign flush_way = flush_way_r;
    end else begin
        assign flush_way = {NUM_WAYS{1'b1}};
    end

endmodule

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
    input  wire flush_in_valid,
    output wire flush_in_ready,
    output wire flush_out_init,
    output wire flush_out_valid,
    output wire [`CS_LINE_SEL_BITS-1:0] flush_out_line,
    output wire [NUM_WAYS-1:0] flush_out_way,
    input  wire flush_out_ready,
    input  wire mshr_empty
);
    // ways interation is only needed when eviction is enabled
    parameter CTR_WIDTH = `CS_LINE_SEL_BITS + (WRITEBACK ? `CS_WAY_SEL_BITS : 0);

    parameter STATE_IDLE  = 2'd0;
    parameter STATE_INIT  = 2'd1;
    parameter STATE_FLUSH = 2'd2;

    reg [CTR_WIDTH-1:0] counter_r;
    reg [1:0] state_r, state_n;
    reg flush_in_ready_r, flush_in_ready_n;

    always @(*) begin
        state_n = state_r;
        flush_in_ready_n = 0;
        case (state_r)
            // STATE_IDLE
            default: begin
                if (flush_in_valid && mshr_empty) begin
                    state_n = STATE_FLUSH;
                end
            end
            STATE_INIT: begin
                if (counter_r == ((2 ** `CS_LINE_SEL_BITS)-1)) begin
                    state_n = STATE_IDLE;
                end
            end
            STATE_FLUSH: begin
                if (counter_r == ((2 ** CTR_WIDTH)-1)) begin
                    state_n = STATE_IDLE;
                    flush_in_ready_n = 1;
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state_r <= STATE_INIT;
            counter_r <= '0;
            flush_in_ready_r <= '0;
        end else begin
            state_r <= state_n;
            flush_in_ready_r <= flush_in_ready_n;
            if (state_r != STATE_IDLE) begin
                if ((state_r == STATE_INIT) || flush_out_ready) begin
                    counter_r <= counter_r + CTR_WIDTH'(1);
                end
            end else begin
                counter_r <= '0;
            end
        end
    end

    assign flush_in_ready  = flush_in_ready_r;
    assign flush_out_init  = (state_r == STATE_INIT);
    assign flush_out_valid = (state_r == STATE_FLUSH);
    assign flush_out_line  = counter_r[`CS_LINE_SEL_BITS-1:0];

    if (WRITEBACK && `CS_WAY_SEL_BITS > 0) begin
        reg [NUM_WAYS-1:0] flush_out_way_r;
        always @(*) begin
            flush_out_way_r = '0;
            flush_out_way_r[counter_r[`CS_LINE_SEL_BITS +: `CS_WAY_SEL_BITS]] = 1;
        end
        assign flush_out_way = flush_out_way_r;
    end else begin
        assign flush_out_way = {NUM_WAYS{1'b1}};
    end

endmodule

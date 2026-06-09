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

// Iterative integer multiplier
// An adaptation of ZipCPU algorithm for a multi-lane elastic architecture.
// https://zipcpu.com/zipcpu/2021/07/03/slowmpy.html

`TRACING_OFF
module VX_fold_mul #(
    parameter NUM_INPUTS = 2,
    parameter IN_WIDTH   = 32,
    parameter OUT_WIDTH  = 32,
    parameter SIGNED     = 0,
    parameter LATENCY    = 3
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_INPUTS-1:0][IN_WIDTH-1:0] data_in,
    output wire [OUT_WIDTH-1:0] result,

    input  wire strobe,
    output wire busy

);
    // State definitions
    localparam STATE_IDLE = 1'b0;
    localparam STATE_BUSY = 1'b1;

    reg state;
    reg [$clog2(LATENCY+1)-1:0] cycles;
    reg [$clog2(NUM_INPUTS)-1:0] count;

    // Multiplier interface
    logic [OUT_WIDTH-1:0] mul_in_a;
    logic [NUM_INPUTS-2:0][IN_WIDTH-1:0] mul_in_b;
    wire [OUT_WIDTH-1:0] mul_out;
    wire mul_enable = (state == STATE_BUSY);

    VX_multiplier #(
        .A_WIDTH (OUT_WIDTH),
        .B_WIDTH (IN_WIDTH),
        .R_WIDTH (OUT_WIDTH),
        .SIGNED  (SIGNED),
        .LATENCY (LATENCY)
    ) shared_mul (
        .clk    (clk),
        .enable (mul_enable),
        .dataa  (mul_in_a),
        .datab  (mul_in_b[0]),
        .result (mul_out)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            state  <= STATE_IDLE;
            cycles <= 0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    if (strobe) begin
                        mul_in_b <= data_in[NUM_INPUTS-1:1];
                        if (SIGNED) begin
                            mul_in_a <= OUT_WIDTH'($signed(data_in[0]));
                        end else begin
                            mul_in_a <= OUT_WIDTH'(data_in[0]);
                        end
                        count  <= (NUM_INPUTS-2);
                        cycles <= LATENCY;
                        state  <= STATE_BUSY;
                    end
                end
                STATE_BUSY: begin
                    if (cycles == 0) begin
                        if (count == 0) begin
                            state <= STATE_IDLE;
                        end else begin
                            mul_in_a <= mul_out;
                            for (integer i = 0; i < NUM_INPUTS-2; i++) begin
                                mul_in_b[i] <= mul_in_b[i+1];
                            end
                            count  <= count - 1;
                            cycles <= LATENCY;
                        end
                    end else begin
                        cycles <= cycles - 1;
                    end
                end
            endcase
        end
    end

    assign busy = (state == STATE_BUSY);
    assign result = mul_out;

endmodule
`TRACING_ON

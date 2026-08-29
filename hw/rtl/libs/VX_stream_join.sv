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
module VX_stream_join #(
    parameter NUM_INPUTS = 1,
    parameter DATAW      = 1,
    parameter OUT_BUF    = 0,
    parameter EAGER      = 0
) (
    input  wire                              clk,
    input  wire                              reset,

    input  wire [NUM_INPUTS-1:0]             valid_in,
    input  wire [NUM_INPUTS-1:0][DATAW-1:0]  data_in,
    output wire [NUM_INPUTS-1:0]             ready_in,

    output wire                              valid_out,
    output wire [NUM_INPUTS-1:0][DATAW-1:0]  data_out,
    input  wire                              ready_out
);
    if (NUM_INPUTS > 1) begin : g_multi_inputs

        wire valid_in_w;
        wire [NUM_INPUTS-1:0][DATAW-1:0] data_in_w;
        wire ready_in_w;

        if (EAGER != 0) begin : g_eager

            reg [NUM_INPUTS-1:0] received_r;
            reg [NUM_INPUTS-1:0][DATAW-1:0] data_in_r;

            wire flush = (& received_r) && ready_in_w;
            wire fire_in_w = valid_in_w && ready_in_w;

            wire [NUM_INPUTS-1:0] fire_in = valid_in & ready_in;

            always @(posedge clk) begin
                if (reset) begin
                    received_r <= '0;
                end else begin
                    if (fire_in_w) begin
                        received_r <= flush ? valid_in : '0;
                    end else begin
                        received_r <= received_r | valid_in;
                    end
                end
            end

            always @(posedge clk) begin
                for (integer i = 0; i < NUM_INPUTS; ++i) begin
                    if (fire_in[i]) begin
                        data_in_r[i] <= data_in[i];
                    end
                end
            end

            assign valid_in_w = & (valid_in | received_r);

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_data_in
                assign data_in_w[i] = received_r[i] ? data_in_r[i] : data_in[i];
            end

            assign ready_in = ~received_r | {NUM_INPUTS{flush}};

        end else begin : g_lockstep

            wire valid_all = (& valid_in);

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_ready_in
                assign ready_in[i] = ready_in_w && valid_all;
            end

            assign valid_in_w = valid_all;
            assign data_in_w  = data_in;

        end

        VX_elastic_buffer #(
            .DATAW   (NUM_INPUTS * DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in_w),
            .data_in   (data_in_w),
            .ready_in  (ready_in_w),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end else begin : g_passthru

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .data_in   (data_in),
            .ready_in  (ready_in),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end

endmodule
`TRACING_ON

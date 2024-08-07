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
module VX_elastic_buffer #(
    parameter DATAW   = 1,
    parameter SIZE    = 1,
    parameter OUT_REG = 0,
    parameter LUTRAM  = 0
) (
    input  wire             clk,
    input  wire             reset,

    input  wire             valid_in,
    output wire             ready_in,
    input  wire [DATAW-1:0] data_in,

    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
);
    if (SIZE == 0) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end else if (SIZE == 1) begin

        VX_pipe_buffer #(
            .DATAW (DATAW)
        ) pipe_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .data_in   (data_in),
            .ready_in  (ready_in),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end else if (SIZE == 2 && LUTRAM == 0) begin

        VX_skid_buffer #(
            .DATAW   (DATAW),
            .HALF_BW (OUT_REG == 2),
            .OUT_REG (OUT_REG)
        ) skid_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .data_in   (data_in),
            .ready_in  (ready_in),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end else begin

        wire empty, full;

        wire [DATAW-1:0] data_out_t;
        wire ready_out_t;

        wire push = valid_in && ready_in;
        wire pop = ~empty && ready_out_t;

        VX_fifo_queue #(
            .DATAW   (DATAW),
            .DEPTH   (SIZE),
            .OUT_REG (OUT_REG == 1),
            .LUTRAM  (LUTRAM)
        ) fifo_queue (
            .clk    (clk),
            .reset  (reset),
            .push   (push),
            .pop    (pop),
            .data_in(data_in),
            .data_out(data_out_t),
            .empty  (empty),
            .full   (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign ready_in = ~full;

        VX_pipe_buffer #(
            .DATAW (DATAW),
            .DEPTH ((OUT_REG > 0) ? (OUT_REG-1) : 0)
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (~empty),
            .data_in   (data_out_t),
            .ready_in  (ready_out_t),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end

endmodule
`TRACING_ON

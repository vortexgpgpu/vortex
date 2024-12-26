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

`TRACING_OFF
module VX_stream_xbar #(
    parameter NUM_INPUTS    = 4,
    parameter NUM_OUTPUTS   = 4,
    parameter DATAW         = 4,
    parameter ARBITER       = "R",
    parameter OUT_BUF       = 0,
    parameter MAX_FANOUT    = `MAX_FANOUT,
    parameter PERF_CTR_BITS = `CLOG2(NUM_INPUTS+1),
    parameter IN_WIDTH      = `LOG2UP(NUM_INPUTS),
    parameter OUT_WIDTH     = `LOG2UP(NUM_OUTPUTS)
) (
    input wire                              clk,
    input wire                              reset,

    input wire [NUM_INPUTS-1:0]             valid_in,
    input wire [NUM_INPUTS-1:0][DATAW-1:0]  data_in,
    input wire [NUM_INPUTS-1:0][OUT_WIDTH-1:0] sel_in,
    output wire [NUM_INPUTS-1:0]            ready_in,

    output wire [NUM_OUTPUTS-1:0]           valid_out,
    output wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out,
    output wire [NUM_OUTPUTS-1:0][IN_WIDTH-1:0] sel_out,
    input  wire [NUM_OUTPUTS-1:0]           ready_out,

    output wire [PERF_CTR_BITS-1:0]         collisions
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    if (NUM_INPUTS != 1) begin : g_multi_inputs

        if (NUM_OUTPUTS != 1) begin : g_multiple_outputs

            // (#inputs > 1) and (#outputs > 1)

            wire [NUM_INPUTS-1:0][NUM_OUTPUTS-1:0] per_output_valid_in;
            wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0] per_output_valid_in_w;

            wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0] per_output_ready_in;
            wire [NUM_INPUTS-1:0][NUM_OUTPUTS-1:0] per_output_ready_in_w;

            VX_transpose #(
                .N (NUM_OUTPUTS),
                .M (NUM_INPUTS)
            ) rdy_in_transpose (
                .data_in (per_output_ready_in),
                .data_out (per_output_ready_in_w)
            );

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_ready_in
                assign ready_in[i] = | per_output_ready_in_w[i];
            end

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_sel_in_demux
                VX_demux #(
                    .DATAW (1),
                    .N (NUM_OUTPUTS)
                ) sel_in_demux (
                    .sel_in   (sel_in[i]),
                    .data_in  (valid_in[i]),
                    .data_out (per_output_valid_in[i])
                );
            end

            VX_transpose #(
                .N (NUM_INPUTS),
                .M (NUM_OUTPUTS)
            ) val_in_transpose (
                .data_in (per_output_valid_in),
                .data_out (per_output_valid_in_w)
            );

            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_xbar_arbs
                VX_stream_arb #(
                    .NUM_INPUTS  (NUM_INPUTS),
                    .NUM_OUTPUTS (1),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF)
                ) xbar_arb (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (per_output_valid_in_w[i]),
                    .data_in   (data_in),
                    .ready_in  (per_output_ready_in[i]),
                    .valid_out (valid_out[i]),
                    .data_out  (data_out[i]),
                    .sel_out   (sel_out[i]),
                    .ready_out (ready_out[i])
                );
            end

        end else begin : g_one_output

            // (#inputs >= 1) and (#outputs == 1)

            VX_stream_arb #(
                .NUM_INPUTS  (NUM_INPUTS),
                .NUM_OUTPUTS (1),
                .DATAW       (DATAW),
                .ARBITER     (ARBITER),
                .MAX_FANOUT  (MAX_FANOUT),
                .OUT_BUF     (OUT_BUF)
            ) xbar_arb (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in),
                .data_in   (data_in),
                .ready_in  (ready_in),
                .valid_out (valid_out),
                .data_out  (data_out),
                .sel_out   (sel_out),
                .ready_out (ready_out)
            );

            `UNUSED_VAR (sel_in)
        end

    end else if (NUM_OUTPUTS != 1) begin : g_single_input

        // (#inputs == 1) and (#outputs > 1)

        wire [NUM_OUTPUTS-1:0] valid_out_w, ready_out_w;
        wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out_w;

        VX_demux #(
            .DATAW (1),
            .N (NUM_OUTPUTS)
        ) sel_in_demux (
            .sel_in   (sel_in[0]),
            .data_in  (valid_in[0]),
            .data_out (valid_out_w)
        );

        assign ready_in[0] = ready_out_w[sel_in[0]];
        assign data_out_w = {NUM_OUTPUTS{data_in[0]}};

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out_buf
            VX_elastic_buffer #(
                .DATAW   (DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_out_w[i]),
                .ready_in  (ready_out_w[i]),
                .data_in   (data_out_w[i]),
                .data_out  (data_out[i]),
                .valid_out (valid_out[i]),
                .ready_out (ready_out[i])
            );
        end

        assign sel_out = 0;

    end else begin : g_passthru

        // (#inputs == 1) and (#outputs == 1)

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .ready_in  (ready_in),
            .data_in   (data_in),
            .data_out  (data_out),
            .valid_out (valid_out),
            .ready_out (ready_out)
        );

        `UNUSED_VAR (sel_in)
        assign sel_out = 0;

    end

    // compute inputs collision
    // we have a collision when there exists a valid transfer with multiple input candicates
    // we count the unique duplicates each cycle.

    reg [NUM_INPUTS-1:0] per_cycle_collision, per_cycle_collision_r;
    wire [`CLOG2(NUM_INPUTS+1)-1:0] collision_count;
    reg [PERF_CTR_BITS-1:0] collisions_r;

    always @(*) begin
        per_cycle_collision = '0;
        for (integer i = 0; i < NUM_INPUTS; ++i) begin
            for (integer j = i + 1; j < NUM_INPUTS; ++j) begin
                per_cycle_collision[i] |= valid_in[i]
                                       && valid_in[j]
                                       && (sel_in[i] == sel_in[j])
                                       && (ready_in[i] | ready_in[j]);
            end
        end
    end

    `BUFFER(per_cycle_collision_r, per_cycle_collision);
    `POP_COUNT(collision_count, per_cycle_collision_r);

    always @(posedge clk) begin
        if (reset) begin
            collisions_r <= '0;
        end else begin
            collisions_r <= collisions_r + PERF_CTR_BITS'(collision_count);
        end
    end

    assign collisions = collisions_r;

endmodule
`TRACING_ON

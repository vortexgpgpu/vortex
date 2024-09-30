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
module VX_stream_arb #(
    parameter NUM_INPUTS    = 1,
    parameter NUM_OUTPUTS   = 1,
    parameter DATAW         = 1,
    parameter `STRING ARBITER = "R",
    parameter MAX_FANOUT    = `MAX_FANOUT,
    parameter OUT_BUF       = 0,
    parameter NUM_REQS      = `CDIV(NUM_INPUTS, NUM_OUTPUTS),
    parameter LOG_NUM_REQS  = `CLOG2(NUM_REQS),
    parameter NUM_REQS_W    = `UP(LOG_NUM_REQS)
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_INPUTS-1:0]             valid_in,
    input  wire [NUM_INPUTS-1:0][DATAW-1:0]  data_in,
    output wire [NUM_INPUTS-1:0]             ready_in,

    output wire [NUM_OUTPUTS-1:0]            valid_out,
    output wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out,
    output wire [NUM_OUTPUTS-1:0][NUM_REQS_W-1:0] sel_out,
    input  wire [NUM_OUTPUTS-1:0]            ready_out
);
    if (NUM_INPUTS > NUM_OUTPUTS) begin : g_more_inputs

        if (NUM_OUTPUTS > 1) begin : g_multiple_outputs

            // (#inputs > #outputs) and (#outputs > 1)

            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_arb_slices

                localparam SLICE_BEGIN = i * NUM_REQS;
                localparam SLICE_END   = `MIN(SLICE_BEGIN + NUM_REQS, NUM_INPUTS);
                localparam SLICE_SIZE  = SLICE_END - SLICE_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (SLICE_SIZE),
                    .NUM_OUTPUTS (1),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[SLICE_END-1: SLICE_BEGIN]),
                    .ready_in  (ready_in[SLICE_END-1: SLICE_BEGIN]),
                    .data_in   (data_in[SLICE_END-1: SLICE_BEGIN]),
                    .data_out  (data_out[i]),
                    .sel_out   (sel_out[i]),
                    .valid_out (valid_out[i]),
                    .ready_out (ready_out[i])
                );
            end

        end else if (MAX_FANOUT != 0 && (NUM_INPUTS > (MAX_FANOUT + MAX_FANOUT /2))) begin : g_fanout

            // (#inputs > max_fanout) and (#outputs == 1)

            localparam NUM_SLICES    = `CDIV(NUM_INPUTS, MAX_FANOUT);
            localparam LOG_NUM_REQS2 = `CLOG2(MAX_FANOUT);
            localparam LOG_NUM_REQS3 = `CLOG2(NUM_SLICES);

            wire [NUM_SLICES-1:0]   valid_tmp;
            wire [NUM_SLICES-1:0][DATAW+LOG_NUM_REQS2-1:0] data_tmp;
            wire [NUM_SLICES-1:0]   ready_tmp;

            for (genvar i = 0; i < NUM_SLICES; ++i) begin : g_fanout_slice_arbs

                localparam SLICE_BEGIN = i * MAX_FANOUT;
                localparam SLICE_END   = `MIN(SLICE_BEGIN + MAX_FANOUT, NUM_INPUTS);
                localparam SLICE_SIZE  = SLICE_END - SLICE_BEGIN;

                wire [DATAW-1:0] data_tmp_u;
                wire [`LOG2UP(SLICE_SIZE)-1:0] sel_tmp_u;

                VX_stream_arb #(
                    .NUM_INPUTS  (SLICE_SIZE),
                    .NUM_OUTPUTS (1),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (3)
                ) fanout_slice_arb (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[SLICE_END-1: SLICE_BEGIN]),
                    .data_in   (data_in[SLICE_END-1: SLICE_BEGIN]),
                    .ready_in  (ready_in[SLICE_END-1: SLICE_BEGIN]),
                    .valid_out (valid_tmp[i]),
                    .data_out  (data_tmp_u),
                    .sel_out   (sel_tmp_u),
                    .ready_out (ready_tmp[i])
                );

                assign data_tmp[i] = {data_tmp_u, LOG_NUM_REQS2'(sel_tmp_u)};
            end

            wire [DATAW+LOG_NUM_REQS2-1:0] data_out_u;
            wire [LOG_NUM_REQS3-1:0] sel_out_u;

            VX_stream_arb #(
                .NUM_INPUTS  (NUM_SLICES),
                .NUM_OUTPUTS (1),
                .DATAW       (DATAW + LOG_NUM_REQS2),
                .ARBITER     (ARBITER),
                .MAX_FANOUT  (MAX_FANOUT),
                .OUT_BUF     (OUT_BUF)
            ) fanout_join_arb (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_tmp),
                .ready_in  (ready_tmp),
                .data_in   (data_tmp),
                .data_out  (data_out_u),
                .sel_out   (sel_out_u),
                .valid_out (valid_out),
                .ready_out (ready_out)
            );

            assign data_out = data_out_u[LOG_NUM_REQS2 +: DATAW];
            assign sel_out = {sel_out_u, data_out_u[0 +: LOG_NUM_REQS2]};

        end else begin : g_one_output

            // (#inputs <= max_fanout) and (#outputs == 1)

            wire                    valid_in_w;
            wire [DATAW-1:0]        data_in_w;
            wire                    ready_in_w;

            wire                    arb_valid;
            wire [NUM_REQS_W-1:0]   arb_index;
            wire [NUM_REQS-1:0]     arb_onehot;
            wire                    arb_ready;

            VX_generic_arbiter #(
                .NUM_REQS (NUM_REQS),
                .TYPE     (ARBITER)
            ) arbiter (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in),
                .grant_valid  (arb_valid),
                .grant_index  (arb_index),
                .grant_onehot (arb_onehot),
                .grant_ready  (arb_ready)
            );

            assign valid_in_w = arb_valid;
            assign data_in_w  = data_in[arb_index];
            assign arb_ready  = ready_in_w;

            for (genvar i = 0; i < NUM_REQS; ++i) begin : g_ready_in
                assign ready_in[i] = ready_in_w && arb_onehot[i];
            end

            VX_elastic_buffer #(
                .DATAW   (LOG_NUM_REQS + DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in_w),
                .ready_in  (ready_in_w),
                .data_in   ({arb_index, data_in_w}),
                .data_out  ({sel_out, data_out}),
                .valid_out (valid_out),
                .ready_out (ready_out)
            );
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin : g_more_outputs

        if (NUM_INPUTS > 1) begin : g_multiple_inputs

            // (#inputs > 1) and (#outputs > #inputs)

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_arb_slices

                localparam SLICE_BEGIN = i * NUM_REQS;
                localparam SLICE_END   = `MIN(SLICE_BEGIN + NUM_REQS, NUM_OUTPUTS);
                localparam SLICE_SIZE  = SLICE_END - SLICE_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (1),
                    .NUM_OUTPUTS (SLICE_SIZE),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[i]),
                    .ready_in  (ready_in[i]),
                    .data_in   (data_in[i]),
                    .data_out  (data_out[SLICE_END-1: SLICE_BEGIN]),
                    .valid_out (valid_out[SLICE_END-1: SLICE_BEGIN]),
                    .ready_out (ready_out[SLICE_END-1: SLICE_BEGIN]),
                    `UNUSED_PIN (sel_out)
                );

                for (genvar j = SLICE_BEGIN; j < SLICE_END; ++j) begin : g_sel_out
                    assign sel_out[j] = i;
                end
            end

        end else if (MAX_FANOUT != 0 && (NUM_OUTPUTS > (MAX_FANOUT + MAX_FANOUT /2))) begin : g_fanout

            // (#inputs == 1) and (#outputs > max_fanout)

            localparam NUM_SLICES = `CDIV(NUM_OUTPUTS, MAX_FANOUT);

            wire [NUM_SLICES-1:0]            valid_tmp;
            wire [NUM_SLICES-1:0][DATAW-1:0] data_tmp;
            wire [NUM_SLICES-1:0]            ready_tmp;

            VX_stream_arb #(
                .NUM_INPUTS  (1),
                .NUM_OUTPUTS (NUM_SLICES),
                .DATAW       (DATAW),
                .ARBITER     (ARBITER),
                .MAX_FANOUT  (MAX_FANOUT),
                .OUT_BUF     (3)
            ) fanout_fork_arb (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in),
                .ready_in  (ready_in),
                .data_in   (data_in),
                .data_out  (data_tmp),
                .valid_out (valid_tmp),
                .ready_out (ready_tmp),
                `UNUSED_PIN (sel_out)
            );

            for (genvar i = 0; i < NUM_SLICES; ++i) begin : g_fanout_slice_arbs

                localparam SLICE_BEGIN = i * MAX_FANOUT;
                localparam SLICE_END   = `MIN(SLICE_BEGIN + MAX_FANOUT, NUM_OUTPUTS);
                localparam SLICE_SIZE  = SLICE_END - SLICE_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (1),
                    .NUM_OUTPUTS (SLICE_SIZE),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF)
                ) fanout_slice_arb (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_tmp[i]),
                    .ready_in  (ready_tmp[i]),
                    .data_in   (data_tmp[i]),
                    .data_out  (data_out[SLICE_END-1: SLICE_BEGIN]),
                    .valid_out (valid_out[SLICE_END-1: SLICE_BEGIN]),
                    .ready_out (ready_out[SLICE_END-1: SLICE_BEGIN]),
                    `UNUSED_PIN (sel_out)
                );
            end

        end else begin : g_one_input

            // (#inputs == 1) and (#outputs <= max_fanout)

            wire [NUM_OUTPUTS-1:0]  ready_in_w;

            wire [NUM_OUTPUTS-1:0]  arb_requests;
            wire                    arb_valid;
            wire [NUM_OUTPUTS-1:0]  arb_onehot;
            wire                    arb_ready;

            VX_generic_arbiter #(
                .NUM_REQS (NUM_OUTPUTS),
                .TYPE     (ARBITER)
            ) arbiter (
                .clk          (clk),
                .reset        (reset),
                .requests     (arb_requests),
                .grant_valid  (arb_valid),
                `UNUSED_PIN   (grant_index),
                .grant_onehot (arb_onehot),
                .grant_ready  (arb_ready)
            );

            assign arb_requests = ready_in_w;
            assign arb_ready    = valid_in[0];
            assign ready_in     = arb_valid;

            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out_buf
                VX_elastic_buffer #(
                    .DATAW   (DATAW),
                    .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                    .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                    .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
                ) out_buf (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in && arb_onehot[i]),
                    .ready_in  (ready_in_w[i]),
                    .data_in   (data_in),
                    .data_out  (data_out[i]),
                    .valid_out (valid_out[i]),
                    .ready_out (ready_out[i])
                );
            end
        end

        assign sel_out = 0;

    end else begin : g_passthru

        // #Inputs == #Outputs

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out_buf
            VX_elastic_buffer #(
                .DATAW   (DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in[i]),
                .ready_in  (ready_in[i]),
                .data_in   (data_in[i]),
                .data_out  (data_out[i]),
                .valid_out (valid_out[i]),
                .ready_out (ready_out[i])
            );
            assign sel_out[i] = NUM_REQS_W'(i);
        end
    end

endmodule
`TRACING_ON

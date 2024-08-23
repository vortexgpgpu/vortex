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
    parameter `STRING ARBITER = "P",
    parameter MAX_FANOUT    = `MAX_FANOUT,
    parameter OUT_BUF       = 0,
    parameter LUTRAM        = 0,
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
    if (NUM_INPUTS > NUM_OUTPUTS) begin

        if (NUM_OUTPUTS > 1) begin

            // (#inputs > #outputs) and (#outputs > 1)

            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin

                localparam BATCH_BEGIN = i * NUM_REQS;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + NUM_REQS, NUM_INPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                `RESET_RELAY (slice_reset, reset);

                VX_stream_arb #(
                    .NUM_INPUTS  (BATCH_SIZE),
                    .NUM_OUTPUTS (1),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF),
                    .LUTRAM      (LUTRAM)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (slice_reset),
                    .valid_in  (valid_in[BATCH_END-1: BATCH_BEGIN]),
                    .ready_in  (ready_in[BATCH_END-1: BATCH_BEGIN]),
                    .data_in   (data_in[BATCH_END-1: BATCH_BEGIN]),
                    .data_out  (data_out[i]),
                    .sel_out   (sel_out[i]),
                    .valid_out (valid_out[i]),
                    .ready_out (ready_out[i])
                );
            end

        end else if (MAX_FANOUT != 0 && (NUM_INPUTS > MAX_FANOUT)) begin

            // (#inputs > max_fanout) and (#outputs == 1)

            localparam NUM_BATCHES = `CDIV(NUM_INPUTS, MAX_FANOUT);
            localparam LOG_NUM_REQS2 = `CLOG2(MAX_FANOUT);
            localparam LOG_NUM_REQS3 = `CLOG2(NUM_BATCHES);

            wire [NUM_BATCHES-1:0]                  valid_tmp;
            wire [NUM_BATCHES-1:0][DATAW+LOG_NUM_REQS2-1:0] data_tmp;
            wire [NUM_BATCHES-1:0]                  ready_tmp;

            for (genvar i = 0; i < NUM_BATCHES; ++i) begin

                localparam BATCH_BEGIN = i * MAX_FANOUT;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + MAX_FANOUT, NUM_INPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                wire [DATAW-1:0] data_tmp_u;
                wire [`LOG2UP(BATCH_SIZE)-1:0] sel_tmp_u;

                `RESET_RELAY (slice_reset, reset);

                if (MAX_FANOUT != 1) begin
                    VX_stream_arb #(
                        .NUM_INPUTS  (BATCH_SIZE),
                        .NUM_OUTPUTS (1),
                        .DATAW       (DATAW),
                        .ARBITER     (ARBITER),
                        .MAX_FANOUT  (MAX_FANOUT),
                        .OUT_BUF     (3), // registered output
                        .LUTRAM      (LUTRAM)
                    ) fanout_slice_arb (
                        .clk       (clk),
                        .reset     (slice_reset),
                        .valid_in  (valid_in[BATCH_END-1: BATCH_BEGIN]),
                        .data_in   (data_in[BATCH_END-1: BATCH_BEGIN]),
                        .ready_in  (ready_in[BATCH_END-1: BATCH_BEGIN]),
                        .valid_out (valid_tmp[i]),
                        .data_out  (data_tmp_u),
                        .sel_out   (sel_tmp_u),
                        .ready_out (ready_tmp[i])
                    );
                end

                assign data_tmp[i] = {data_tmp_u, LOG_NUM_REQS2'(sel_tmp_u)};
            end

            wire [DATAW+LOG_NUM_REQS2-1:0] data_out_u;
            wire [LOG_NUM_REQS3-1:0] sel_out_u;

            VX_stream_arb #(
                .NUM_INPUTS  (NUM_BATCHES),
                .NUM_OUTPUTS (1),
                .DATAW       (DATAW + LOG_NUM_REQS2),
                .ARBITER     (ARBITER),
                .MAX_FANOUT  (MAX_FANOUT),
                .OUT_BUF     (OUT_BUF),
                .LUTRAM      (LUTRAM)
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

        end else begin

            // (#inputs <= max_fanout) and (#outputs == 1)

            wire                    valid_in_r;
            wire [DATAW-1:0]        data_in_r;
            wire                    ready_in_r;

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

            assign valid_in_r = arb_valid;
            assign arb_ready  = ready_in_r;

            VX_onehot_mux #(
                .DATAW (DATAW),
                .N     (NUM_REQS)
            ) onehot_mux (
                .data_in  (data_in),
                .sel_in   (arb_onehot),
                .data_out (data_in_r)
            );

            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign ready_in[i] = ready_in_r && arb_onehot[i];
            end

            VX_elastic_buffer #(
                .DATAW   (LOG_NUM_REQS + DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                .LUTRAM  (LUTRAM)
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in_r),
                .ready_in  (ready_in_r),
                .data_in   ({arb_index, data_in_r}),
                .data_out  ({sel_out, data_out}),
                .valid_out (valid_out),
                .ready_out (ready_out)
            );
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin

        if (NUM_INPUTS > 1) begin

            // (#inputs > 1) and (#outputs > #inputs)

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin

                localparam BATCH_BEGIN = i * NUM_REQS;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + NUM_REQS, NUM_OUTPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                `RESET_RELAY (slice_reset, reset);

                VX_stream_arb #(
                    .NUM_INPUTS  (1),
                    .NUM_OUTPUTS (BATCH_SIZE),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF),
                    .LUTRAM      (LUTRAM)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (slice_reset),
                    .valid_in  (valid_in[i]),
                    .ready_in  (ready_in[i]),
                    .data_in   (data_in[i]),
                    .data_out  (data_out[BATCH_END-1: BATCH_BEGIN]),
                    .valid_out (valid_out[BATCH_END-1: BATCH_BEGIN]),
                    .ready_out (ready_out[BATCH_END-1: BATCH_BEGIN]),
                    `UNUSED_PIN (sel_out)
                );

                for (genvar j = BATCH_BEGIN; j < BATCH_END; ++j) begin
                    assign sel_out[j] = i;
                end
            end

        end else if (MAX_FANOUT != 0 && (NUM_OUTPUTS > MAX_FANOUT)) begin

            // (#inputs == 1) and (#outputs > max_fanout)

            localparam NUM_BATCHES = `CDIV(NUM_OUTPUTS, MAX_FANOUT);

            wire [NUM_BATCHES-1:0]            valid_tmp;
            wire [NUM_BATCHES-1:0][DATAW-1:0] data_tmp;
            wire [NUM_BATCHES-1:0]            ready_tmp;

            VX_stream_arb #(
                .NUM_INPUTS  (1),
                .NUM_OUTPUTS (NUM_BATCHES),
                .DATAW       (DATAW),
                .ARBITER     (ARBITER),
                .MAX_FANOUT  (MAX_FANOUT),
                .OUT_BUF     (3), // registered output
                .LUTRAM      (LUTRAM)
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

            for (genvar i = 0; i < NUM_BATCHES; ++i) begin

                localparam BATCH_BEGIN = i * MAX_FANOUT;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + MAX_FANOUT, NUM_OUTPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                `RESET_RELAY (slice_reset, reset);

                VX_stream_arb #(
                    .NUM_INPUTS  (1),
                    .NUM_OUTPUTS (BATCH_SIZE),
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .OUT_BUF     (OUT_BUF),
                    .LUTRAM      (LUTRAM)
                ) fanout_slice_arb (
                    .clk       (clk),
                    .reset     (slice_reset),
                    .valid_in  (valid_tmp[i]),
                    .ready_in  (ready_tmp[i]),
                    .data_in   (data_tmp[i]),
                    .data_out  (data_out[BATCH_END-1: BATCH_BEGIN]),
                    .valid_out (valid_out[BATCH_END-1: BATCH_BEGIN]),
                    .ready_out (ready_out[BATCH_END-1: BATCH_BEGIN]),
                    `UNUSED_PIN (sel_out)
                );
            end

        end else begin

            // (#inputs == 1) and (#outputs <= max_fanout)

            wire [NUM_OUTPUTS-1:0]  ready_in_r;

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

            assign arb_requests = ready_in_r;
            assign arb_ready    = valid_in[0];
            assign ready_in     = arb_valid;

            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
                VX_elastic_buffer #(
                    .DATAW   (DATAW),
                    .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                    .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                    .LUTRAM  (LUTRAM)
                ) out_buf (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in && arb_onehot[i]),
                    .ready_in  (ready_in_r[i]),
                    .data_in   (data_in),
                    .data_out  (data_out[i]),
                    .valid_out (valid_out[i]),
                    .ready_out (ready_out[i])
                );
            end
        end

        assign sel_out = 0;

    end else begin

        // #Inputs == #Outputs

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin

            `RESET_RELAY_EN (out_buf_reset, reset, (NUM_OUTPUTS > 1));

            VX_elastic_buffer #(
                .DATAW   (DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                .LUTRAM  (LUTRAM)
            ) out_buf (
                .clk       (clk),
                .reset     (out_buf_reset),
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

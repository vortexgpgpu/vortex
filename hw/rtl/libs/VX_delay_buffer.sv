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

// Elastic buffer with a fixed latency: an entry accepted on cycle t is offered
// on cycle t + LATENCY, one entry per cycle at full throughput. The entries
// wait in a FIFO; only their valid bits travel a delay line, which releases
// each one in order into a stream buffer, the registered output. ready_in is
// the FIFO's registered full flag, so ready_out reaches only the stream buffer.
`TRACING_OFF
module VX_delay_buffer #(
    parameter DATAW   = 1,
    parameter LATENCY = 1,
    // FIFO entries: those in flight at full throughput, plus the one accepted
    parameter SIZE    = (1 << `CLOG2(LATENCY)),
    parameter LUTRAM  = 0
) (
    input  wire             clk,
    input  wire             reset,

    input  wire             valid_in,
    output wire             ready_in,
    input  wire [DATAW-1:0] data_in,

    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out,

    output wire             empty
);
    if (LATENCY == 0) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_PARAM (SIZE)
        `UNUSED_PARAM (LUTRAM)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;
        assign empty     = 1'b1;

    end else begin : g_delay

        wire             rel_valid;
        wire [DATAW-1:0] rel_data;
        wire             rel_ready;
        wire             fifo_empty;

        if (LATENCY > 1) begin : g_fifo

            `STATIC_ASSERT(SIZE >= LATENCY, ("invalid parameter: SIZE=%0d must cover LATENCY=%0d", SIZE, LATENCY))

            wire full;
            wire push = valid_in && ready_in;
            wire pop  = rel_valid && rel_ready;

            VX_fifo_queue #(
                .DATAW  (DATAW),
                .DEPTH  (SIZE),
                .LUTRAM (LUTRAM)
            ) fifo (
                .clk       (clk),
                .reset     (reset),
                .push      (push),
                .pop       (pop),
                .data_in   (data_in),
                .data_out  (rel_data),
                .empty     (fifo_empty),
                .full      (full),
                `UNUSED_PIN (alm_empty),
                `UNUSED_PIN (alm_full),
                `UNUSED_PIN (size)
            );

            assign ready_in = ~full;

            // The release count and the stream buffer each register a cycle.
            wire release_in;

            VX_shift_register #(
                .DATAW  (1),
                .RESETW (1),
                .DEPTH  (LATENCY - 2)
            ) release_line (
                .clk      (clk),
                .reset    (reset),
                .enable   (1'b1),
                .data_in  (push),
                .data_out (release_in)
            );

            wire none_released;

            VX_pending_size #(
                .SIZE (SIZE)
            ) release_count (
                .clk       (clk),
                .reset     (reset),
                .incr      (release_in),
                .decr      (pop),
                .empty     (none_released),
                `UNUSED_PIN (alm_empty),
                `UNUSED_PIN (full),
                `UNUSED_PIN (alm_full),
                `UNUSED_PIN (size)
            );

            assign rel_valid = ~none_released;

        end else begin : g_no_fifo

            `UNUSED_PARAM (SIZE)
            `UNUSED_PARAM (LUTRAM)

            assign rel_valid  = valid_in;
            assign rel_data   = data_in;
            assign ready_in   = rel_ready;
            assign fifo_empty = 1'b1;

        end

        VX_stream_buffer #(
            .DATAW   (DATAW),
            .OUT_REG (1)
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rel_valid),
            .ready_in  (rel_ready),
            .data_in   (rel_data),
            .data_out  (data_out),
            .ready_out (ready_out),
            .valid_out (valid_out)
        );

        assign empty = fifo_empty && ~valid_out;

    end

endmodule
`TRACING_ON

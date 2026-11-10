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
module VX_pe_serializer #(
    parameter NUM_LANES      = 1,
    parameter NUM_PES        = 1,
    parameter LATENCY        = 1,
    parameter DATA_IN_WIDTH  = 1,
    parameter DATA_OUT_WIDTH = DATA_IN_WIDTH,
    parameter SHARED_WIDTH   = 0,
    parameter TAG_WIDTH      = 0,
    parameter PE_REG         = 0,
    parameter OUT_BUF        = 0,
    // a zero-latency skid register on the output in place of OUT_BUF: the
    // input ready is registered, the output is not
    parameter OUT_SKID       = 0,
    // copies of the PE clock enable, each its own net
    parameter PE_ENABLES     = 1
) (
    input wire                          clk,
    input wire                          reset,

    // input
    input wire                          valid_in,
    input wire [NUM_LANES-1:0]          mask_in,
    input wire [NUM_LANES-1:0][DATA_IN_WIDTH-1:0] data_in,
    input wire [`UP(SHARED_WIDTH)-1:0]  shared_in,
    input wire [`UP(TAG_WIDTH)-1:0]     tag_in,
    output wire                         ready_in,

    // PE
    output wire [PE_ENABLES-1:0]        pe_enable,
    output wire [NUM_PES-1:0]           pe_mask_out,
    output wire [NUM_PES-1:0][DATA_IN_WIDTH-1:0] pe_data_out,
    output wire [`UP(SHARED_WIDTH)-1:0] pe_shared_out,
    input wire [NUM_PES-1:0][DATA_OUT_WIDTH-1:0] pe_data_in,

    // output
    output wire                         valid_out,
    output wire [NUM_LANES-1:0]         mask_out,
    output wire [NUM_LANES-1:0][DATA_OUT_WIDTH-1:0] data_out,
    output wire [`UP(TAG_WIDTH)-1:0]    tag_out,
    input wire                          ready_out
);
    localparam SHARED_WIDTH_S = `UP(SHARED_WIDTH);
    localparam TAG_WIDTH_S = `UP(TAG_WIDTH);

    wire                    valid_out_u;
    wire [NUM_LANES-1:0]    mask_out_u;
    wire [NUM_LANES-1:0][DATA_OUT_WIDTH-1:0] data_out_u;
    wire [TAG_WIDTH_S-1:0]  tag_out_u;
    wire                    ready_out_u;

    wire [NUM_PES-1:0][DATA_IN_WIDTH-1:0] pe_data_out_w;
    wire [NUM_PES-1:0]      pe_mask_out_w;
    wire                    pe_valid_in;
    wire [NUM_PES-1:0]      pe_mask_in_w;
    wire [TAG_WIDTH_S-1:0]  pe_tag_in;
    wire enable;
    // the last stage's valid as it would load when enabled
    wire                    valid_out_n;

    localparam DEPTH = PE_REG + LATENCY;
    // The skid's registered enable also reads the stage before the last.
    localparam NUM_TAPS = (OUT_SKID != 0 && DEPTH > 1 && NUM_LANES == NUM_PES) ? 2 : 1;

    `STATIC_ASSERT (OUT_SKID == 0 || DEPTH != 0, ("invalid parameter"))

    wire [NUM_TAPS-1:0][1 + NUM_PES + TAG_WIDTH_S-1:0] shift_out;

    VX_shift_register #(
        .DATAW     (1 + NUM_PES + TAG_WIDTH_S),
        .DEPTH     (DEPTH),
        .RESETW    (1),
        .NUM_TAPS  (NUM_TAPS),
        .TAP_START (DEPTH - NUM_TAPS)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in, pe_mask_out_w, tag_in}),
        .data_out (shift_out)
    );

    assign {pe_valid_in, pe_mask_in_w, pe_tag_in} = shift_out[NUM_TAPS-1];

    VX_pipe_register #(
        .DATAW  (NUM_PES * DATA_IN_WIDTH + SHARED_WIDTH_S),
        .DEPTH  (PE_REG)
    ) pe_data_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({pe_data_out_w, shared_in}),
        .data_out ({pe_data_out,   pe_shared_out})
    );

    assign pe_mask_out = pe_mask_out_w;

    if (NUM_LANES != NUM_PES) begin : g_serialize

        localparam BATCH_SIZE = NUM_LANES / NUM_PES;
        localparam BATCH_SIZEW = `LOG2UP(BATCH_SIZE);

        reg [BATCH_SIZEW-1:0] batch_in_idx, batch_out_idx;
        reg batch_in_done, batch_out_done;

        // Each PE selects among its BATCH_SIZE lanes only, a BATCH_SIZE:1 mux
        // rather than one over every lane.
        for (genvar i = 0; i < NUM_PES; ++i) begin : g_pe_data_out_w
            wire [BATCH_SIZE-1:0][DATA_IN_WIDTH-1:0] batch_data;
            wire [BATCH_SIZE-1:0] batch_mask;
            for (genvar b = 0; b < BATCH_SIZE; ++b) begin : g_batch
                assign batch_data[b] = data_in[b * NUM_PES + i];
                assign batch_mask[b] = mask_in[b * NUM_PES + i];
            end
            assign pe_data_out_w[i] = batch_data[batch_in_idx];
            assign pe_mask_out_w[i] = batch_mask[batch_in_idx];
        end

        always @(posedge clk) begin
            if (reset) begin
                batch_in_idx   <= '0;
                batch_out_idx  <= '0;
                batch_in_done  <= 0;
                batch_out_done <= 0;
            end else if (enable) begin
                batch_in_idx   <= batch_in_idx + BATCH_SIZEW'(valid_in);
                batch_out_idx  <= batch_out_idx + BATCH_SIZEW'(pe_valid_in);
                batch_in_done  <= valid_in && (batch_in_idx == BATCH_SIZEW'(BATCH_SIZE-2));
                batch_out_done <= valid_out_n;
            end
        end

        assign valid_out_n = pe_valid_in && (batch_out_idx == BATCH_SIZEW'(BATCH_SIZE-2));

        reg [BATCH_SIZE-1:0][(NUM_PES * DATA_OUT_WIDTH)-1:0] data_out_r, data_out_n;
        reg [BATCH_SIZE-1:0][NUM_PES-1:0] mask_out_r, mask_out_n;

        always @(*) begin
            data_out_n = data_out_r;
            mask_out_n = mask_out_r;
            if (pe_valid_in) begin
                data_out_n[batch_out_idx] = pe_data_in;
                mask_out_n[batch_out_idx] = pe_mask_in_w;
            end
        end

        always @(posedge clk) begin
            data_out_r <= data_out_n;
            mask_out_r <= mask_out_n;
        end

        assign ready_in    = enable && batch_in_done;

        assign valid_out_u = batch_out_done;
        assign data_out_u  = data_out_n;
        assign mask_out_u  = mask_out_n;
        assign tag_out_u   = pe_tag_in;

    end else begin : g_passthru

        assign pe_data_out_w = data_in;
        assign pe_mask_out_w = mask_in;
        assign ready_in    = enable;
        assign valid_out_u = pe_valid_in;
        assign data_out_u  = pe_data_in;
        assign mask_out_u  = pe_mask_in_w;
        assign tag_out_u   = pe_tag_in;

        if (NUM_TAPS > 1) begin : g_valid_tap
            assign valid_out_n = shift_out[0][NUM_PES + TAG_WIDTH_S];
        end else begin : g_valid_in
            assign valid_out_n = valid_in;
        end

    end

    if (OUT_SKID != 0) begin : g_enable_r

        // The core's clock enable, ready_out_u || ~valid_out_u, computed a
        // cycle ahead from the skid's and the last stage's next state, so the
        // net that reaches every unit of the core starts at a flop.
        wire has_data   = ~ready_out_u;
        wire has_data_n = ~ready_out && (has_data || valid_out_u);
        wire valid_n    = enable ? valid_out_n : valid_out_u;

        wire enable_n = ~has_data_n || ~valid_n;

        reg enable_r;
        always @(posedge clk) begin
            if (reset) begin
                enable_r <= 1'b1;
            end else begin
                enable_r <= enable_n;
            end
        end

        // one flop per copy, so no single net carries every unit's enable
        `PRESERVE_NET reg [PE_ENABLES-1:0] pe_enable_r;
        always @(posedge clk) begin
            if (reset) begin
                pe_enable_r <= '1;
            end else begin
                pe_enable_r <= {PE_ENABLES{enable_n}};
            end
        end

        assign enable    = enable_r;
        assign pe_enable = pe_enable_r;

        `RUNTIME_ASSERT (enable_r == (ready_out_u || ~valid_out_u), ("%t: pe_serializer enable diverged", $time))

    end else begin : g_enable

        `UNUSED_VAR (valid_out_n)
        assign enable    = ready_out_u || ~valid_out_u;
        assign pe_enable = {PE_ENABLES{enable}};

    end

    if (OUT_SKID != 0) begin : g_out_skid

        `UNUSED_PARAM (OUT_BUF)

        VX_bypass_buffer #(
            .DATAW ((NUM_LANES * DATA_OUT_WIDTH) + NUM_LANES + TAG_WIDTH_S)
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_out_u),
            .ready_in  (ready_out_u),
            .data_in   ({data_out_u, mask_out_u, tag_out_u}),
            .data_out  ({data_out, mask_out, tag_out}),
            .valid_out (valid_out),
            .ready_out (ready_out)
        );

    end else begin : g_out_buf

        VX_elastic_buffer #(
            .DATAW   ((NUM_LANES * DATA_OUT_WIDTH) + NUM_LANES + TAG_WIDTH_S),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_out_u),
            .ready_in  (ready_out_u),

            .data_in   ({data_out_u, mask_out_u, tag_out_u}),
            .data_out  ({data_out, mask_out, tag_out}),
            .valid_out (valid_out),
            .ready_out (ready_out)
        );

    end

endmodule
`TRACING_ON

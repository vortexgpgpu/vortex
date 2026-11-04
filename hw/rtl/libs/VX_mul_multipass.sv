// Copyright © 2019-2026
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

// Multi-pass 64-bit integer multiplier.
//
// A GPU multiplies 64-bit integers by running its pipelined 32-bit multiplier
// (IMAD) several times: the low 64 bits of a 64 x 64 product take three
// 32 x 32 partial products, the high 64 bits all four. This unit does the
// same in hardware: one pipelined 33 x 33 signed multiplier per lane (the
// multiplier the 32-bit configuration uses, so the same DSP mapping), fed
// one partial product a cycle by a pass counter, and a per-lane accumulator
// that sums the partial products as they leave the multiplier.
//
//   pass 0: a.lo x b.lo  (weight 2^0)   1 pass : W  (low 32 bits, sign-extended)
//   pass 1: a.lo x b.hi  (weight 2^32)  3 passes: low 64 bits
//   pass 2: a.hi x b.lo  (weight 2^32)  4 passes: high 64 bits
//   pass 3: a.hi x b.hi  (weight 2^64)
//
// The low halves enter the multiplier zero-extended, the high halves with
// their operand's sign (signed_a / signed_b), so every partial product is a
// signed 66-bit value and MULH, MULHSU and MULHU need no correction step.
// After passes 0 and 2 the accumulator retires its low 32 bits into the
// result's low word, so its adder stays 68 bits wide, and the adder is split
// in carry-select halves.
//
// An operation holds its input (ready_in low) until its last pass issues; a
// new operation can issue the cycle after. The result leaves LATENCY + 1
// cycles after the last pass: 1 pass for W, 3 for the low and 4 for the high
// product. The whole pipeline stalls while the output is held.

`TRACING_OFF
module VX_mul_multipass #(
    parameter LANES     = 1,
    parameter TAG_WIDTH = 1,
    parameter LATENCY   = 3  // the multiplier's pipeline depth
) (
    input wire                          clk,
    input wire                          reset,

    input wire                          valid_in,
    output wire                         ready_in,
    input wire                          is_w,     // 32 x 32, low 32 bits sign-extended
    input wire                          is_high,  // high 64 bits of the 128-bit product
    input wire                          signed_a,
    input wire                          signed_b,
    input wire [LANES-1:0][63:0]        dataa,
    input wire [LANES-1:0][63:0]        datab,
    input wire [TAG_WIDTH-1:0]          tag_in,

    output wire                         valid_out,
    input wire                          ready_out,
    output wire [LANES-1:0][63:0]       result,
    output wire [TAG_WIDTH-1:0]         tag_out
);
    localparam ACC_WIDTH = 68;
    localparam LO_WIDTH  = ACC_WIDTH / 2;
    localparam HI_WIDTH  = ACC_WIDTH - LO_WIDTH;

    // the pipeline advances unless a result waits at the output
    wire enable = ready_out || ~valid_out;

    // pass sequencer
    reg [1:0] pass;
    wire [1:0] last_pass = is_w ? 2'd0 : (is_high ? 2'd3 : 2'd2);
    wire pass_last = (pass == last_pass);

    always @(posedge clk) begin
        if (reset) begin
            pass <= '0;
        end else if (valid_in && enable) begin
            pass <= pass_last ? 2'd0 : (pass + 2'd1);
        end
    end

    assign ready_in = enable && pass_last;

    // partial-product operands: pass[1] picks a's high half, pass[0] b's
    wire [LANES-1:0][65:0] prod;

    for (genvar i = 0; i < LANES; ++i) begin : g_mul
        wire [32:0] mul_a = pass[1] ? {signed_a && dataa[i][63], dataa[i][63:32]} : {1'b0, dataa[i][31:0]};
        wire [32:0] mul_b = pass[0] ? {signed_b && datab[i][63], datab[i][63:32]} : {1'b0, datab[i][31:0]};

        VX_multiplier #(
            .A_WIDTH (33),
            .B_WIDTH (33),
            .R_WIDTH (66),
            .SIGNED  (1),
            .LATENCY (LATENCY)
        ) multiplier (
            .clk    (clk),
            .enable (enable),
            .dataa  (mul_a),
            .datab  (mul_b),
            .result (prod[i])
        );
    end

    // each pass's control beside its partial product
    wire       m_valid, m_last, m_w, m_high;
    wire [1:0] m_pass;
    wire [TAG_WIDTH-1:0] m_tag;

    VX_shift_register #(
        .DATAW  (1 + 2 + 1 + 1 + 1 + TAG_WIDTH),
        .DEPTH  (LATENCY),
        .RESETW (1)
    ) ctl_pipe (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in, pass,   pass_last, is_w, is_high, tag_in}),
        .data_out ({m_valid,  m_pass, m_last,    m_w,  m_high,  m_tag})
    );

    // accumulate stage, which is also the output register
    reg valid_r, w_r, high_r;
    reg [TAG_WIDTH-1:0] tag_r;

    always @(posedge clk) begin
        if (reset) begin
            valid_r <= 0;
        end else if (enable) begin
            valid_r <= m_valid && m_last;
        end
        if (enable) begin
            tag_r  <= m_tag;
            w_r    <= m_w;
            high_r <= m_high;
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_acc
        reg [ACC_WIDTH-1:0] acc;  // the running sum above the retired words
        reg [63:0]          lo;   // the retired words: the product's low 64 bits

        // carry-select at LO_WIDTH: the upper part both ways beside the lower
        // part's carry, each a single adder (a carry-in rides in a forced low
        // bit), so the accumulate loop is a half-width carry chain
        wire [ACC_WIDTH-1:0] acc_in = (m_pass == 2'd0) ? '0 : acc;
        wire [ACC_WIDTH-1:0] addend = ACC_WIDTH'($signed(prod[i]));
        wire [LO_WIDTH:0] sum_lo = {1'b0, acc_in[LO_WIDTH-1:0]} + {1'b0, addend[LO_WIDTH-1:0]};
        wire [HI_WIDTH-1:0] sum_hi0 = acc_in[ACC_WIDTH-1:LO_WIDTH] + addend[ACC_WIDTH-1:LO_WIDTH];
        wire [HI_WIDTH:0] sum_hi1 = {acc_in[ACC_WIDTH-1:LO_WIDTH], 1'b1} + {addend[ACC_WIDTH-1:LO_WIDTH], 1'b1};
        wire [ACC_WIDTH-1:0] sum = {(sum_lo[LO_WIDTH] ? sum_hi1[HI_WIDTH:1] : sum_hi0), sum_lo[LO_WIDTH-1:0]};
        `UNUSED_VAR (sum_hi1[0])

        always @(posedge clk) begin
            if (enable && m_valid) begin
                // passes 0 and 2 complete a 32-bit word of the product
                acc <= m_pass[0] ? sum : ACC_WIDTH'($signed(sum[ACC_WIDTH-1:32]));
                if (m_pass == 2'd0) begin
                    lo[31:0] <= sum[31:0];
                end
                if (m_pass == 2'd2) begin
                    lo[63:32] <= sum[31:0];
                end
            end
        end

        assign result[i] = w_r    ? 64'($signed(lo[31:0])) :
                           high_r ? acc[63:0] :
                                    lo;
        `UNUSED_VAR (acc[ACC_WIDTH-1:64])
    end

    assign valid_out = valid_r;
    assign tag_out   = tag_r;

endmodule
`TRACING_ON

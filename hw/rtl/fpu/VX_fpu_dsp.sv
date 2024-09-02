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

`include "VX_fpu_define.vh"

`ifdef FPU_DSP

module VX_fpu_dsp import VX_fpu_pkg::*; #(
    parameter NUM_LANES = 4,
    parameter TAG_WIDTH = 4,
    parameter OUT_BUF   = 0
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [NUM_LANES-1:0] mask_in,

    input wire [TAG_WIDTH-1:0] tag_in,

    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_FMT_BITS-1:0] fmt,
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`XLEN-1:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    localparam FPU_FMA     = 0;
    localparam FPU_DIVSQRT = 1;
    localparam FPU_CVT     = 2;
    localparam FPU_NCP     = 3;
    localparam NUM_FPCORES = 4;
    localparam FPCORES_BITS = `LOG2UP(NUM_FPCORES);

    localparam REQ_DATAW = NUM_LANES + TAG_WIDTH + `INST_FPU_BITS + `INST_FMT_BITS + `INST_FRM_BITS + 3 * (NUM_LANES * 32);
    localparam RSP_DATAW = (NUM_LANES * 32) + 1 + $bits(fflags_t) + TAG_WIDTH;

    `UNUSED_VAR (fmt)

    wire [NUM_FPCORES-1:0] per_core_valid_in;
    wire [NUM_FPCORES-1:0][REQ_DATAW-1:0] per_core_data_in;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0] per_core_mask_in;
    wire [NUM_FPCORES-1:0][TAG_WIDTH-1:0] per_core_tag_in;
    wire [NUM_FPCORES-1:0][`INST_FPU_BITS-1:0] per_core_op_type;
    wire [NUM_FPCORES-1:0][`INST_FMT_BITS-1:0] per_core_fmt;
    wire [NUM_FPCORES-1:0][`INST_FRM_BITS-1:0] per_core_frm;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][31:0] per_core_dataa;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][31:0] per_core_datab;
    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][31:0] per_core_datac;
    wire [NUM_FPCORES-1:0] per_core_ready_in;

    wire [NUM_FPCORES-1:0][NUM_LANES-1:0][31:0] per_core_result;
    wire [NUM_FPCORES-1:0][TAG_WIDTH-1:0] per_core_tag_out;
    wire [NUM_FPCORES-1:0] per_core_ready_out;
    wire [NUM_FPCORES-1:0] per_core_valid_out;
    wire [NUM_FPCORES-1:0] per_core_has_fflags;
    fflags_t [NUM_FPCORES-1:0] per_core_fflags;

    wire [1:0] div_sqrt_ready_in;
    wire [1:0][NUM_LANES-1:0][31:0] div_sqrt_result;
    wire [1:0][TAG_WIDTH-1:0] div_sqrt_tag_out;
    wire [1:0] div_sqrt_ready_out;
    wire [1:0] div_sqrt_valid_out;
    wire [1:0] div_sqrt_has_fflags;
    fflags_t [1:0] div_sqrt_fflags;

    `RESET_RELAY (fma_reset, reset);
    `RESET_RELAY (div_reset, reset);
    `RESET_RELAY (sqrt_reset, reset);
    `RESET_RELAY (cvt_reset, reset);
    `RESET_RELAY (ncp_reset, reset);

    wire [NUM_LANES-1:0][31:0] dataa_s;
    wire [NUM_LANES-1:0][31:0] datab_s;
    wire [NUM_LANES-1:0][31:0] datac_s;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign dataa_s[i] = dataa[i][31:0];
        assign datab_s[i] = datab[i][31:0];
        assign datac_s[i] = datac[i][31:0];
    end

    `UNUSED_VAR (dataa)
    `UNUSED_VAR (datab)
    `UNUSED_VAR (datac)

    // Decode fpu core type
    wire [FPCORES_BITS-1:0] core_select = op_type[3:2];

    VX_stream_switch #(
        .DATAW       (REQ_DATAW),
        .NUM_OUTPUTS (NUM_FPCORES)
    ) req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (core_select),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({mask_in, tag_in, op_type, fmt, frm, dataa_s, datab_s, datac_s}),
        .data_out  (per_core_data_in),
        .valid_out (per_core_valid_in),
        .ready_out (per_core_ready_in)
    );

    for (genvar i = 0; i < NUM_FPCORES; ++i) begin
        assign {
            per_core_mask_in[i],
            per_core_tag_in[i],
            per_core_op_type[i],
            per_core_fmt[i],
            per_core_frm[i],
            per_core_dataa[i],
            per_core_datab[i],
            per_core_datac[i]
        } = per_core_data_in[i];
    end

    // FMA core

    wire is_madd = per_core_op_type[FPU_FMA][1];
    wire is_neg  = per_core_op_type[FPU_FMA][0];
    wire is_sub  = per_core_fmt[FPU_FMA][1];

    VX_fpu_fma #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) fpu_fma (
        .clk        (clk),
        .reset      (fma_reset),
        .valid_in   (per_core_valid_in[FPU_FMA]),
        .ready_in   (per_core_ready_in[FPU_FMA]),
        .mask_in    (per_core_mask_in[FPU_FMA]),
        .tag_in     (per_core_tag_in[FPU_FMA]),
        .frm        (per_core_frm[FPU_FMA]),
        .is_madd    (is_madd),
        .is_sub     (is_sub),
        .is_neg     (is_neg),
        .dataa      (per_core_dataa[FPU_FMA]),
        .datab      (per_core_datab[FPU_FMA]),
        .datac      (per_core_datac[FPU_FMA]),
        .has_fflags (per_core_has_fflags[FPU_FMA]),
        .fflags     (per_core_fflags[FPU_FMA]),
        .result     (per_core_result[FPU_FMA]),
        .tag_out    (per_core_tag_out[FPU_FMA]),
        .ready_out  (per_core_ready_out[FPU_FMA]),
        .valid_out  (per_core_valid_out[FPU_FMA])
    );

    // Div/Sqrt cores

    wire is_sqrt = per_core_op_type[FPU_DIVSQRT][0];
    assign per_core_ready_in[FPU_DIVSQRT] = div_sqrt_ready_in[is_sqrt];

    VX_fpu_div #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) fpu_div (
        .clk        (clk),
        .reset      (div_reset),
        .valid_in   (per_core_valid_in[FPU_DIVSQRT] && ~is_sqrt),
        .ready_in   (div_sqrt_ready_in[0]),
        .mask_in    (per_core_mask_in[FPU_DIVSQRT]),
        .tag_in     (per_core_tag_in[FPU_DIVSQRT]),
        .frm        (per_core_frm[FPU_DIVSQRT]),
        .dataa      (per_core_dataa[FPU_DIVSQRT]),
        .datab      (per_core_datab[FPU_DIVSQRT]),
        .has_fflags (div_sqrt_has_fflags[0]),
        .fflags     (div_sqrt_fflags[0]),
        .result     (div_sqrt_result[0]),
        .tag_out    (div_sqrt_tag_out[0]),
        .valid_out  (div_sqrt_valid_out[0]),
        .ready_out  (div_sqrt_ready_out[0])
    );

    VX_fpu_sqrt #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) fpu_sqrt (
        .clk        (clk),
        .reset      (sqrt_reset),
        .valid_in   (per_core_valid_in[FPU_DIVSQRT] && is_sqrt),
        .ready_in   (div_sqrt_ready_in[1]),
        .mask_in    (per_core_mask_in[FPU_DIVSQRT]),
        .tag_in     (per_core_tag_in[FPU_DIVSQRT]),
        .frm        (per_core_frm[FPU_DIVSQRT]),
        .dataa      (per_core_dataa[FPU_DIVSQRT]),
        .has_fflags (div_sqrt_has_fflags[1]),
        .fflags     (div_sqrt_fflags[1]),
        .result     (div_sqrt_result[1]),
        .tag_out    (div_sqrt_tag_out[1]),
        .valid_out  (div_sqrt_valid_out[1]),
        .ready_out  (div_sqrt_ready_out[1])
    );

    // CVT core

    wire is_itof = per_core_op_type[FPU_CVT][1];
    wire is_signed = ~per_core_op_type[FPU_CVT][0];
    wire cvt_ret_int_in = ~is_itof;
    wire cvt_ret_int_out;

    VX_fpu_cvt #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (1+TAG_WIDTH)
    ) fpu_cvt (
        .clk        (clk),
        .reset      (cvt_reset),
        .valid_in   (per_core_valid_in[FPU_CVT]),
        .ready_in   (per_core_ready_in[FPU_CVT]),
        .mask_in    (per_core_mask_in[FPU_CVT]),
        .tag_in     ({cvt_ret_int_in, per_core_tag_in[FPU_CVT]}),
        .frm        (per_core_frm[FPU_CVT]),
        .is_itof    (is_itof),
        .is_signed  (is_signed),
        .dataa      (per_core_dataa[FPU_CVT]),
        .has_fflags (per_core_has_fflags[FPU_CVT]),
        .fflags     (per_core_fflags[FPU_CVT]),
        .result     (per_core_result[FPU_CVT]),
        .tag_out    ({cvt_ret_int_out, per_core_tag_out[FPU_CVT]}),
        .valid_out  (per_core_valid_out[FPU_CVT]),
        .ready_out  (per_core_ready_out[FPU_CVT])
    );

    // NCP core

    wire ncp_ret_int_in = (per_core_op_type[FPU_NCP] == `INST_FPU_CMP)
                      || `INST_FPU_IS_CLASS(per_core_op_type[FPU_NCP], per_core_frm[FPU_NCP])
                      || `INST_FPU_IS_MVXW(per_core_op_type[FPU_NCP], per_core_frm[FPU_NCP]);
    wire ncp_ret_int_out;

    wire ncp_ret_sext_in = `INST_FPU_IS_MVXW(per_core_op_type[FPU_NCP], per_core_frm[FPU_NCP]);
    wire ncp_ret_sext_out;

    VX_fpu_ncp #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH+2)
    ) fpu_ncp (
        .clk        (clk),
        .reset      (ncp_reset),
        .valid_in   (per_core_valid_in[FPU_NCP]),
        .ready_in   (per_core_ready_in[FPU_NCP]),
        .mask_in    (per_core_mask_in[FPU_NCP]),
        .tag_in     ({ncp_ret_sext_in, ncp_ret_int_in, per_core_tag_in[FPU_NCP]}),
        .op_type    (per_core_op_type[FPU_NCP]),
        .frm        (per_core_frm[FPU_NCP]),
        .dataa      (per_core_dataa[FPU_NCP]),
        .datab      (per_core_datab[FPU_NCP]),
        .result     (per_core_result[FPU_NCP]),
        .has_fflags (per_core_has_fflags[FPU_NCP]),
        .fflags     (per_core_fflags[FPU_NCP]),
        .tag_out    ({ncp_ret_sext_out, ncp_ret_int_out, per_core_tag_out[FPU_NCP]}),
        .valid_out  (per_core_valid_out[FPU_NCP]),
        .ready_out  (per_core_ready_out[FPU_NCP])
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [1:0][RSP_DATAW-1:0] div_sqrt_arb_data_in;
    for (genvar i = 0; i < 2; ++i) begin
        assign div_sqrt_arb_data_in[i] = {
            div_sqrt_result[i],
            div_sqrt_has_fflags[i],
            div_sqrt_fflags[i],
            div_sqrt_tag_out[i]
        };
    end

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW),
        .ARBITER    ("P"),
        .OUT_BUF    (0)
    ) div_sqrt_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (div_sqrt_valid_out),
        .ready_in  (div_sqrt_ready_out),
        .data_in   (div_sqrt_arb_data_in),
        .data_out  ({
            per_core_result[FPU_DIVSQRT],
            per_core_has_fflags[FPU_DIVSQRT],
            per_core_fflags[FPU_DIVSQRT],
            per_core_tag_out[FPU_DIVSQRT]
        }),
        .valid_out (per_core_valid_out[FPU_DIVSQRT]),
        .ready_out (per_core_ready_out[FPU_DIVSQRT]),
        `UNUSED_PIN (sel_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    reg [NUM_FPCORES-1:0][RSP_DATAW+2-1:0] per_core_data_out;

    always @(*) begin
        for (integer i = 0; i < NUM_FPCORES; ++i) begin
            per_core_data_out[i][RSP_DATAW+1:2] = {
                per_core_result[i],
                per_core_has_fflags[i],
                per_core_fflags[i],
                per_core_tag_out[i]
            };
            per_core_data_out[i][1:0] = '0;
        end
        per_core_data_out[FPU_CVT][1:0] = {1'b1, cvt_ret_int_out};
        per_core_data_out[FPU_NCP][1:0] = {ncp_ret_sext_out, ncp_ret_int_out};
    end

    wire [NUM_LANES-1:0][31:0] result_s;

    wire [1:0] op_ret_int_out;
    `UNUSED_VAR (op_ret_int_out)

    VX_stream_arb #(
        .NUM_INPUTS (NUM_FPCORES),
        .DATAW      (RSP_DATAW + 2),
        .ARBITER    ("R"),
        .OUT_BUF    (OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_core_valid_out),
        .ready_in  (per_core_ready_out),
        .data_in   (per_core_data_out),
        .data_out  ({result_s, has_fflags, fflags, tag_out, op_ret_int_out}),
        .valid_out (valid_out),
        .ready_out (ready_out),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
    `ifdef FPU_RV64F
        reg [`XLEN-1:0] result_w;
        always @(*) begin
            case (op_ret_int_out)
            2'b11:   result_w = `XLEN'($signed(result_s[i]));
            2'b01:   result_w = {32'h00000000, result_s[i]};
            default: result_w = {32'hffffffff, result_s[i]};
            endcase
        end
        assign result[i] = result_w;
    `else
        assign result[i] = result_s[i];
    `endif
    end

endmodule

`endif

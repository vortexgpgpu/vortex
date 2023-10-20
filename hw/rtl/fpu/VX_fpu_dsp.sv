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
    parameter TAGW      = 4,
    parameter OUT_REG   = 0
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [NUM_LANES-1:0] lane_mask,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_FMT_BITS-1:0] fmt,
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`XLEN-1:0] result, 

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    localparam FPU_FMA     = 0;
    localparam FPU_DIVSQRT = 1;
    localparam FPU_CVT     = 2;
    localparam FPU_NCP     = 3;
    localparam NUM_FPC     = 4;
    localparam FPC_BITS    = `LOG2UP(NUM_FPC);

    localparam RSP_DATAW = (NUM_LANES * 32) + 1 + $bits(fflags_t) + TAGW;

    `UNUSED_VAR (fmt)    

    wire [NUM_FPC-1:0] per_core_ready_in;
    wire [NUM_FPC-1:0][NUM_LANES-1:0][31:0] per_core_result;
    wire [NUM_FPC-1:0][TAGW-1:0] per_core_tag_out;
    wire [NUM_FPC-1:0] per_core_ready_out;
    wire [NUM_FPC-1:0] per_core_valid_out;    
    wire [NUM_FPC-1:0] per_core_has_fflags;  
    fflags_t [NUM_FPC-1:0] per_core_fflags;

    wire div_ready_in, sqrt_ready_in;
    wire [NUM_LANES-1:0][31:0] div_result, sqrt_result;
    wire [TAGW-1:0] div_tag_out, sqrt_tag_out;
    wire div_ready_out, sqrt_ready_out;
    wire div_valid_out, sqrt_valid_out;    
    wire div_has_fflags, sqrt_has_fflags;  
    fflags_t div_fflags, sqrt_fflags;

    reg [FPC_BITS-1:0] core_select;
    reg is_madd, is_sub, is_neg, is_div, is_itof, is_signed;

    always @(*) begin
        is_madd   = 0;
        is_sub    = 0;        
        is_neg    = 0;
        is_div    = 0;
        is_itof   = 0;
        is_signed = 0;
        case (op_type)
            `INST_FPU_ADD:    begin core_select = FPU_FMA; end
            `INST_FPU_SUB:    begin core_select = FPU_FMA; is_sub = 1; end
            `INST_FPU_MUL:    begin core_select = FPU_FMA; is_neg = 1; end
            `INST_FPU_MADD:   begin core_select = FPU_FMA; is_madd = 1; end
            `INST_FPU_MSUB:   begin core_select = FPU_FMA; is_madd = 1; is_sub = 1; end
            `INST_FPU_NMADD:  begin core_select = FPU_FMA; is_madd = 1; is_neg = 1; end
            `INST_FPU_NMSUB:  begin core_select = FPU_FMA; is_madd = 1; is_sub = 1; is_neg = 1; end
            `INST_FPU_DIV:    begin core_select = FPU_DIVSQRT; is_div = 1; end
            `INST_FPU_SQRT:   begin core_select = FPU_DIVSQRT; end
            `INST_FPU_F2I:    begin core_select = FPU_CVT; is_signed = 1; end
            `INST_FPU_F2U:    begin core_select = FPU_CVT; end
            `INST_FPU_I2F:    begin core_select = FPU_CVT; is_itof = 1; is_signed = 1; end
            `INST_FPU_U2F:    begin core_select = FPU_CVT; is_itof = 1; end
            default:          begin core_select = FPU_NCP; end
        endcase
    end

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

    VX_fpu_fma #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fpu_fma (
        .clk        (clk), 
        .reset      (fma_reset), 
        .valid_in   (valid_in && (core_select == FPU_FMA)),
        .ready_in   (per_core_ready_in[FPU_FMA]),
        .lane_mask  (lane_mask),
        .tag_in     (tag_in), 
        .frm        (frm),
        .is_madd    (is_madd),
        .is_sub     (is_sub),
        .is_neg     (is_neg),
        .dataa      (dataa_s), 
        .datab      (datab_s), 
        .datac      (datac_s), 
        .has_fflags (per_core_has_fflags[FPU_FMA]),
        .fflags     (per_core_fflags[FPU_FMA]),
        .result     (per_core_result[FPU_FMA]),
        .tag_out    (per_core_tag_out[FPU_FMA]),
        .ready_out  (per_core_ready_out[FPU_FMA]),
        .valid_out  (per_core_valid_out[FPU_FMA])
    );

    VX_fpu_div #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fpu_div (
        .clk        (clk), 
        .reset      (div_reset), 
        .valid_in   (valid_in && (core_select == FPU_DIVSQRT) && is_div),
        .ready_in   (div_ready_in),
        .lane_mask  (lane_mask),
        .tag_in     (tag_in),
        .frm        (frm), 
        .dataa      (dataa_s), 
        .datab      (datab_s), 
        .has_fflags (div_has_fflags),
        .fflags     (div_fflags), 
        .result     (div_result),
        .tag_out    (div_tag_out),
        .valid_out  (div_valid_out),
        .ready_out  (div_ready_out)
    );

    VX_fpu_sqrt #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fpu_sqrt (
        .clk        (clk), 
        .reset      (sqrt_reset), 
        .valid_in   (valid_in && (core_select == FPU_DIVSQRT) && ~is_div),
        .ready_in   (sqrt_ready_in),
        .lane_mask  (lane_mask),
        .tag_in     (tag_in),
        .frm        (frm), 
        .dataa      (dataa_s), 
        .has_fflags (sqrt_has_fflags),
        .fflags     (sqrt_fflags),
        .result     (sqrt_result),
        .tag_out    (sqrt_tag_out),
        .valid_out  (sqrt_valid_out),
        .ready_out  (sqrt_ready_out)
    );

    wire cvt_rt_int_in = ~is_itof;
    wire cvt_rt_int_out;

    VX_fpu_cvt #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW+1)
    ) fpu_cvt (
        .clk        (clk), 
        .reset      (cvt_reset), 
        .valid_in   (valid_in && (core_select == FPU_CVT)),
        .ready_in   (per_core_ready_in[FPU_CVT]),
        .lane_mask  (lane_mask),
        .tag_in     ({cvt_rt_int_in, tag_in}), 
        .frm        (frm),
        .is_itof    (is_itof), 
        .is_signed  (is_signed), 
        .dataa      (dataa_s), 
        .has_fflags (per_core_has_fflags[FPU_CVT]),
        .fflags     (per_core_fflags[FPU_CVT]),
        .result     (per_core_result[FPU_CVT]),
        .tag_out    ({cvt_rt_int_out, per_core_tag_out[FPU_CVT]}),
        .valid_out  (per_core_valid_out[FPU_CVT]),
        .ready_out  (per_core_ready_out[FPU_CVT])
    );

    wire ncp_rt_int_in = (op_type == `INST_FPU_CMP)
                      || `INST_FPU_IS_CLASS(op_type, frm) 
                      || `INST_FPU_IS_MVXW(op_type, frm);
    wire ncp_rt_int_out;

    wire ncp_rt_sext_in = `INST_FPU_IS_MVXW(op_type, frm);
    wire ncp_rt_sext_out;
    
    VX_fpu_ncomp #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW+2)
    ) fpu_ncomp (
        .clk        (clk),
        .reset      (ncp_reset), 
        .valid_in   (valid_in && (core_select == FPU_NCP)),
        .ready_in   (per_core_ready_in[FPU_NCP]),
        .lane_mask  (lane_mask),
        .tag_in     ({ncp_rt_sext_in, ncp_rt_int_in, tag_in}),
        .op_type    (op_type),
        .frm        (frm),
        .dataa      (dataa_s),
        .datab      (datab_s), 
        .result     (per_core_result[FPU_NCP]), 
        .has_fflags (per_core_has_fflags[FPU_NCP]),
        .fflags     (per_core_fflags[FPU_NCP]),
        .tag_out    ({ncp_rt_sext_out, ncp_rt_int_out, per_core_tag_out[FPU_NCP]}),
        .valid_out  (per_core_valid_out[FPU_NCP]),
        .ready_out  (per_core_ready_out[FPU_NCP])
    );

    ///////////////////////////////////////////////////////////////////////////

    assign per_core_ready_in[FPU_DIVSQRT] = is_div ? div_ready_in : sqrt_ready_in;

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW), 
        .ARBITER    ("R"),
        .OUT_REG    (0)
    ) div_sqrt_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({sqrt_valid_out, div_valid_out}), 
        .ready_in  ({sqrt_ready_out, div_ready_out}),
        .data_in   ({{sqrt_result, sqrt_has_fflags, sqrt_fflags, sqrt_tag_out}, 
                     {div_result, div_has_fflags, div_fflags, div_tag_out}}),
        .data_out  ({per_core_result[FPU_DIVSQRT], per_core_has_fflags[FPU_DIVSQRT], per_core_fflags[FPU_DIVSQRT], per_core_tag_out[FPU_DIVSQRT]}),
        .valid_out (per_core_valid_out[FPU_DIVSQRT]),
        .ready_out (per_core_ready_out[FPU_DIVSQRT]),
        `UNUSED_PIN (sel_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    reg [NUM_FPC-1:0][RSP_DATAW+2-1:0] per_core_data_out;
    
    always @(*) begin
        for (integer i = 0; i < NUM_FPC; ++i) begin
            per_core_data_out[i][RSP_DATAW+1:2] = {per_core_result[i], per_core_has_fflags[i], per_core_fflags[i], per_core_tag_out[i]};
            per_core_data_out[i][1:0] = '0;
        end        
        per_core_data_out[FPU_CVT][1:0] = {1'b1, cvt_rt_int_out};
        per_core_data_out[FPU_NCP][1:0] = {ncp_rt_sext_out, ncp_rt_int_out};
    end

    wire [NUM_LANES-1:0][31:0] result_s;
    wire [1:0] op_rt_int_out;

    VX_stream_arb #(
        .NUM_INPUTS (NUM_FPC),
        .DATAW      (RSP_DATAW + 2), 
        .ARBITER    ("R"),
        .OUT_REG    (OUT_REG)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_core_valid_out), 
        .ready_in  (per_core_ready_out),
        .data_in   (per_core_data_out),
        .data_out  ({result_s, has_fflags, fflags, tag_out, op_rt_int_out}),
        .valid_out (valid_out),
        .ready_out (ready_out),
        `UNUSED_PIN (sel_out)
    );

`ifndef FPU_RV64F
    `UNUSED_VAR (op_rt_int_out)
`endif

    for (genvar i = 0; i < NUM_LANES; ++i) begin        
    `ifdef FPU_RV64F
        reg [`XLEN-1:0] result_r;
        always @(*) begin
            case (op_rt_int_out)
            2'b11:   result_r = `XLEN'($signed(result_s[i]));
            2'b01:   result_r = {32'h00000000, result_s[i]};
            default: result_r = {32'hffffffff, result_s[i]};
            endcase
        end
        assign result[i] = result_r;
    `else
        assign result[i] = result_s[i];
    `endif
    end

    // can accept new request?
    assign ready_in = per_core_ready_in[core_select];

endmodule
`endif 

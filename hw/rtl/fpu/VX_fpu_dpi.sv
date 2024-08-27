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

`ifdef FPU_DPI

module VX_fpu_dpi import VX_fpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1,
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
    localparam NUM_FPC     = 4;
    localparam FPC_BITS    = `LOG2UP(NUM_FPC);

    localparam RSP_DATAW = (NUM_LANES * `XLEN) + 1 + $bits(fflags_t) + TAG_WIDTH;

    wire [NUM_FPC-1:0] per_core_ready_in;
    wire [NUM_FPC-1:0][NUM_LANES-1:0][`XLEN-1:0] per_core_result;
    wire [NUM_FPC-1:0][TAG_WIDTH-1:0] per_core_tag_out;
    reg  [NUM_FPC-1:0] per_core_ready_out;
    wire [NUM_FPC-1:0] per_core_valid_out;
    wire [NUM_FPC-1:0] per_core_has_fflags;
    fflags_t [NUM_FPC-1:0] per_core_fflags;

    wire div_ready_in, sqrt_ready_in;
    wire [NUM_LANES-1:0][`XLEN-1:0] div_result, sqrt_result;
    wire [TAG_WIDTH-1:0] div_tag_out, sqrt_tag_out;
    wire div_ready_out, sqrt_ready_out;
    wire div_valid_out, sqrt_valid_out;
    wire div_has_fflags, sqrt_has_fflags;
    fflags_t div_fflags, sqrt_fflags;

    reg [FPC_BITS-1:0] core_select;

    reg is_fadd, is_fsub, is_fmul, is_fmadd, is_fmsub, is_fnmadd, is_fnmsub;
    reg is_div, is_fcmp, is_itof, is_utof, is_ftoi, is_ftou, is_f2f;

    reg [NUM_LANES-1:0][63:0] operands [3];

    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            operands[0][i] = 64'(dataa[i]);
            operands[1][i] = 64'(datab[i]);
            operands[2][i] = 64'(datac[i]);
        end
    end

    wire f_fmt = fmt[0];
    wire i_fmt = fmt[1];

    always @(*) begin
        is_fadd   = 0;
        is_fsub   = 0;
        is_fmul   = 0;
        is_fmadd  = 0;
        is_fmsub  = 0;
        is_fnmadd = 0;
        is_fnmsub = 0;
        is_div    = 0;
        is_fcmp   = 0;
        is_itof   = 0;
        is_utof   = 0;
        is_ftoi   = 0;
        is_ftou   = 0;
        is_f2f    = 0;

        case (op_type)
            `INST_FPU_ADD:   begin core_select = FPU_FMA; is_fadd = ~i_fmt; is_fsub = i_fmt; end
            `INST_FPU_MADD:  begin core_select = FPU_FMA; is_fmadd = ~i_fmt; is_fmsub = i_fmt; end
            `INST_FPU_NMADD: begin core_select = FPU_FMA; is_fnmadd = ~i_fmt; is_fnmsub = i_fmt; end
            `INST_FPU_MUL:   begin core_select = FPU_FMA; is_fmul = 1; end
            `INST_FPU_DIV:   begin core_select = FPU_DIVSQRT; is_div = 1; end
            `INST_FPU_SQRT:  begin core_select = FPU_DIVSQRT; end
            `INST_FPU_CMP:   begin core_select = FPU_NCP; is_fcmp = 1; end
            `INST_FPU_F2I:   begin core_select = FPU_CVT; is_ftoi = 1; end
            `INST_FPU_F2U:   begin core_select = FPU_CVT; is_ftou = 1; end
            `INST_FPU_I2F:   begin core_select = FPU_CVT; is_itof = 1; end
            `INST_FPU_U2F:   begin core_select = FPU_CVT; is_utof = 1; end
            `INST_FPU_F2F:   begin core_select = FPU_CVT; is_f2f  = 1; end
            default:         begin core_select = FPU_NCP; end
        endcase
    end

    generate
    begin : fma

        reg [NUM_LANES-1:0][`XLEN-1:0] result_fma;
        reg [NUM_LANES-1:0][63:0] result_fadd;
        reg [NUM_LANES-1:0][63:0] result_fsub;
        reg [NUM_LANES-1:0][63:0] result_fmul;
        reg [NUM_LANES-1:0][63:0] result_fmadd;
        reg [NUM_LANES-1:0][63:0] result_fmsub;
        reg [NUM_LANES-1:0][63:0] result_fnmadd;
        reg [NUM_LANES-1:0][63:0] result_fnmsub;

        fflags_t [NUM_LANES-1:0] fflags_fma;
        fflags_t [NUM_LANES-1:0] fflags_fadd;
        fflags_t [NUM_LANES-1:0] fflags_fsub;
        fflags_t [NUM_LANES-1:0] fflags_fmul;
        fflags_t [NUM_LANES-1:0] fflags_fmadd;
        fflags_t [NUM_LANES-1:0] fflags_fmsub;
        fflags_t [NUM_LANES-1:0] fflags_fnmadd;
        fflags_t [NUM_LANES-1:0] fflags_fnmsub;

        wire fma_valid = (valid_in && core_select == FPU_FMA);
        wire fma_ready = per_core_ready_out[FPU_FMA] || ~per_core_valid_out[FPU_FMA];
        wire fma_fire  = fma_valid && fma_ready;

        always @(*) begin
            for (integer i = 0; i < NUM_LANES; ++i) begin
                dpi_fadd   (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], frm, result_fadd[i], fflags_fadd[i]);
                dpi_fsub   (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], frm, result_fsub[i], fflags_fsub[i]);
                dpi_fmul   (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], frm, result_fmul[i], fflags_fmul[i]);
                dpi_fmadd  (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], operands[2][i], frm, result_fmadd[i], fflags_fmadd[i]);
                dpi_fmsub  (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], operands[2][i], frm, result_fmsub[i], fflags_fmsub[i]);
                dpi_fnmadd (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], operands[2][i], frm, result_fnmadd[i], fflags_fnmadd[i]);
                dpi_fnmsub (fma_fire, int'(f_fmt), operands[0][i], operands[1][i], operands[2][i], frm, result_fnmsub[i], fflags_fnmsub[i]);

                result_fma[i] = is_fadd   ? result_fadd[i][`XLEN-1:0] :
                                is_fsub   ? result_fsub[i][`XLEN-1:0] :
                                is_fmul   ? result_fmul[i][`XLEN-1:0] :
                                is_fmadd  ? result_fmadd[i][`XLEN-1:0] :
                                is_fmsub  ? result_fmsub[i][`XLEN-1:0] :
                                is_fnmadd ? result_fnmadd[i][`XLEN-1:0] :
                                is_fnmsub ? result_fnmsub[i][`XLEN-1:0] :
                                            '0;

                fflags_fma[i] = is_fadd   ? fflags_fadd[i] :
                                is_fsub   ? fflags_fsub[i] :
                                is_fmul   ? fflags_fmul[i] :
                                is_fmadd  ? fflags_fmadd[i] :
                                is_fmsub  ? fflags_fmsub[i] :
                                is_fnmadd ? fflags_fnmadd[i] :
                                is_fnmsub ? fflags_fnmsub[i] :
                                            '0;
            end
        end

        fflags_t fflags_merged;
        `FPU_MERGE_FFLAGS(fflags_merged, fflags_fma, mask_in, NUM_LANES);

        VX_shift_register #(
            .DATAW  (1 + TAG_WIDTH + NUM_LANES * `XLEN + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FMA),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fma_ready),
            .data_in  ({fma_valid, tag_in, result_fma, fflags_merged}),
            .data_out ({per_core_valid_out[FPU_FMA], per_core_tag_out[FPU_FMA], per_core_result[FPU_FMA], per_core_fflags[FPU_FMA]})
        );

        assign per_core_has_fflags[FPU_FMA] = 1;
        assign per_core_ready_in[FPU_FMA] = fma_ready;

    end
    endgenerate

    generate
    begin : fdiv

        reg [NUM_LANES-1:0][`XLEN-1:0] result_fdiv_r;
        reg [NUM_LANES-1:0][63:0] result_fdiv;
        fflags_t [NUM_LANES-1:0] fflags_fdiv;

        wire fdiv_valid = (valid_in && core_select == FPU_DIVSQRT) && is_div;
        wire fdiv_ready = div_ready_out || ~div_valid_out;
        wire fdiv_fire  = fdiv_valid && fdiv_ready;

        always @(*) begin
            for (integer i = 0; i < NUM_LANES; ++i) begin
                dpi_fdiv (fdiv_fire, int'(f_fmt), operands[0][i], operands[1][i], frm, result_fdiv[i], fflags_fdiv[i]);
                result_fdiv_r[i] = result_fdiv[i][`XLEN-1:0];
            end
        end

        fflags_t fflags_merged;
        `FPU_MERGE_FFLAGS(fflags_merged, fflags_fdiv, mask_in, NUM_LANES);

        VX_shift_register #(
            .DATAW  (1 + TAG_WIDTH + NUM_LANES * `XLEN + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FDIV),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fdiv_ready),
            .data_in  ({fdiv_valid, tag_in, result_fdiv_r, fflags_merged}),
            .data_out ({div_valid_out, div_tag_out, div_result, div_fflags})
        );

        assign div_has_fflags = 1;
        assign div_ready_in = fdiv_ready;

    end
    endgenerate

    generate
    begin : fsqrt

        reg [NUM_LANES-1:0][`XLEN-1:0] result_fsqrt_r;
        reg [NUM_LANES-1:0][63:0] result_fsqrt;
        fflags_t [NUM_LANES-1:0] fflags_fsqrt;

        wire fsqrt_valid = (valid_in && core_select == FPU_DIVSQRT) && ~is_div;
        wire fsqrt_ready = sqrt_ready_out || ~sqrt_valid_out;
        wire fsqrt_fire  = fsqrt_valid && fsqrt_ready;

        always @(*) begin
            for (integer i = 0; i < NUM_LANES; ++i) begin
                dpi_fsqrt (fsqrt_fire, int'(f_fmt), operands[0][i], frm, result_fsqrt[i], fflags_fsqrt[i]);
                result_fsqrt_r[i] = result_fsqrt[i][`XLEN-1:0];
            end
        end

        fflags_t fflags_merged;
        `FPU_MERGE_FFLAGS(fflags_merged, fflags_fsqrt, mask_in, NUM_LANES);

        VX_shift_register #(
            .DATAW  (1 + TAG_WIDTH + NUM_LANES * `XLEN + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FSQRT),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fsqrt_ready),
            .data_in  ({fsqrt_valid, tag_in, result_fsqrt_r, fflags_merged}),
            .data_out ({sqrt_valid_out, sqrt_tag_out, sqrt_result, sqrt_fflags})
        );

        assign sqrt_has_fflags = 1;
        assign sqrt_ready_in = fsqrt_ready;

    end
    endgenerate

    generate
    begin : fcvt

        reg [NUM_LANES-1:0][`XLEN-1:0] result_fcvt;
        reg [NUM_LANES-1:0][63:0] result_itof;
        reg [NUM_LANES-1:0][63:0] result_utof;
        reg [NUM_LANES-1:0][63:0] result_ftoi;
        reg [NUM_LANES-1:0][63:0] result_ftou;
        reg [NUM_LANES-1:0][63:0] result_f2f;

        fflags_t [NUM_LANES-1:0] fflags_fcvt;
        fflags_t [NUM_LANES-1:0] fflags_itof;
        fflags_t [NUM_LANES-1:0] fflags_utof;
        fflags_t [NUM_LANES-1:0] fflags_ftoi;
        fflags_t [NUM_LANES-1:0] fflags_ftou;

        wire fcvt_valid = (valid_in && core_select == FPU_CVT);
        wire fcvt_ready = per_core_ready_out[FPU_CVT] || ~per_core_valid_out[FPU_CVT];
        wire fcvt_fire  = fcvt_valid && fcvt_ready;

        always @(*) begin
            for (integer i = 0; i < NUM_LANES; ++i) begin
                dpi_itof (fcvt_fire, int'(f_fmt), int'(i_fmt), operands[0][i], frm, result_itof[i], fflags_itof[i]);
                dpi_utof (fcvt_fire, int'(f_fmt), int'(i_fmt), operands[0][i], frm, result_utof[i], fflags_utof[i]);
                dpi_ftoi (fcvt_fire, int'(i_fmt), int'(f_fmt), operands[0][i], frm, result_ftoi[i], fflags_ftoi[i]);
                dpi_ftou (fcvt_fire, int'(i_fmt), int'(f_fmt), operands[0][i], frm, result_ftou[i], fflags_ftou[i]);
                dpi_f2f  (fcvt_fire, int'(f_fmt), operands[0][i], result_f2f[i]);

                result_fcvt[i] = is_itof ? result_itof[i][`XLEN-1:0] :
                                is_utof ? result_utof[i][`XLEN-1:0] :
                                is_ftoi ? result_ftoi[i][`XLEN-1:0] :
                                is_ftou ? result_ftou[i][`XLEN-1:0] :
                                is_f2f  ? result_f2f[i][`XLEN-1:0] :
                                        '0;

                fflags_fcvt[i] = is_itof ? fflags_itof[i] :
                                is_utof ? fflags_utof[i] :
                                is_ftoi ? fflags_ftoi[i] :
                                is_ftou ? fflags_ftou[i] :
                                       '0;
            end
        end

        fflags_t fflags_merged;
        `FPU_MERGE_FFLAGS(fflags_merged, fflags_fcvt, mask_in, NUM_LANES);

        VX_shift_register #(
            .DATAW  (1 + TAG_WIDTH + NUM_LANES * `XLEN + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FCVT),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fcvt_ready),
            .data_in  ({fcvt_valid, tag_in, result_fcvt, fflags_merged}),
            .data_out ({per_core_valid_out[FPU_CVT], per_core_tag_out[FPU_CVT], per_core_result[FPU_CVT], per_core_fflags[FPU_CVT]})
        );

        assign per_core_has_fflags[FPU_CVT] = 1;
        assign per_core_ready_in[FPU_CVT] = fcvt_ready;

    end
    endgenerate

    generate
    begin : fncp

        reg [NUM_LANES-1:0][`XLEN-1:0]  result_fncp;
        reg [NUM_LANES-1:0][63:0] result_fclss;
        reg [NUM_LANES-1:0][63:0] result_flt;
        reg [NUM_LANES-1:0][63:0] result_fle;
        reg [NUM_LANES-1:0][63:0] result_feq;
        reg [NUM_LANES-1:0][63:0] result_fmin;
        reg [NUM_LANES-1:0][63:0] result_fmax;
        reg [NUM_LANES-1:0][63:0] result_fsgnj;
        reg [NUM_LANES-1:0][63:0] result_fsgnjn;
        reg [NUM_LANES-1:0][63:0] result_fsgnjx;
        reg [NUM_LANES-1:0][63:0] result_fmvx;
        reg [NUM_LANES-1:0][63:0] result_fmvf;

        fflags_t [NUM_LANES-1:0] fflags_fncp;
        fflags_t [NUM_LANES-1:0] fflags_flt;
        fflags_t [NUM_LANES-1:0] fflags_fle;
        fflags_t [NUM_LANES-1:0] fflags_feq;
        fflags_t [NUM_LANES-1:0] fflags_fmin;
        fflags_t [NUM_LANES-1:0] fflags_fmax;

        wire fncp_valid = (valid_in && core_select == FPU_NCP);
        wire fncp_ready = per_core_ready_out[FPU_NCP] || ~per_core_valid_out[FPU_NCP];
        wire fncp_fire  = fncp_valid && fncp_ready;

        always @(*) begin
            for (integer i = 0; i < NUM_LANES; ++i) begin
                dpi_fclss  (fncp_fire, int'(f_fmt), operands[0][i], result_fclss[i]);
                dpi_fle    (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_fle[i], fflags_fle[i]);
                dpi_flt    (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_flt[i], fflags_flt[i]);
                dpi_feq    (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_feq[i], fflags_feq[i]);
                dpi_fmin   (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_fmin[i], fflags_fmin[i]);
                dpi_fmax   (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_fmax[i], fflags_fmax[i]);
                dpi_fsgnj  (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_fsgnj[i]);
                dpi_fsgnjn (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_fsgnjn[i]);
                dpi_fsgnjx (fncp_fire, int'(f_fmt), operands[0][i], operands[1][i], result_fsgnjx[i]);
                result_fmvx[i] = f_fmt ? operands[0][i] : 64'($signed(operands[0][i][31:0]));      // sign-extension
                result_fmvf[i] = f_fmt ? operands[0][i] : (operands[0][i] | 64'hffffffff00000000); // nan-boxing
            end
        end

        always @(*) begin
            result_fncp = 'x;
            fflags_fncp = 'x;
            for (integer i = 0; i < NUM_LANES; ++i) begin
                case (frm)
                0:  begin result_fncp[i] = is_fcmp ? result_fle[i][`XLEN-1:0] : result_fsgnj[i][`XLEN-1:0];  fflags_fncp[i] = fflags_fle[i]; end
                1:  begin result_fncp[i] = is_fcmp ? result_flt[i][`XLEN-1:0] : result_fsgnjn[i][`XLEN-1:0]; fflags_fncp[i] = fflags_flt[i]; end
                2:  begin result_fncp[i] = is_fcmp ? result_feq[i][`XLEN-1:0] : result_fsgnjx[i][`XLEN-1:0]; fflags_fncp[i] = fflags_feq[i]; end
                3:  begin result_fncp[i] = result_fclss[i][`XLEN-1:0]; end
                4:  begin result_fncp[i] = result_fmvx[i][`XLEN-1:0]; end
                5:  begin result_fncp[i] = result_fmvf[i][`XLEN-1:0]; end
                6:  begin result_fncp[i] = result_fmin[i][`XLEN-1:0]; fflags_fncp[i] = fflags_fmin[i]; end
                7:  begin result_fncp[i] = result_fmax[i][`XLEN-1:0]; fflags_fncp[i] = fflags_fmax[i]; end
                endcase
            end
        end

        fflags_t fflags_merged;
        `FPU_MERGE_FFLAGS(fflags_merged, fflags_fncp, mask_in, NUM_LANES);

        wire has_fflags_fncp = (frm >= 6) || is_fcmp;

        VX_shift_register #(
            .DATAW  (1 + TAG_WIDTH + 1 + NUM_LANES * `XLEN + $bits(fflags_t)),
            .DEPTH  (`LATENCY_FNCP),
            .RESETW (1)
        ) shift_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (fncp_ready),
            .data_in  ({fncp_valid, tag_in, has_fflags_fncp, result_fncp, fflags_merged}),
            .data_out ({per_core_valid_out[FPU_NCP], per_core_tag_out[FPU_NCP], per_core_has_fflags[FPU_NCP], per_core_result[FPU_NCP], per_core_fflags[FPU_NCP]})
        );

        assign per_core_ready_in[FPU_NCP] = fncp_ready;

    end
    endgenerate

    ///////////////////////////////////////////////////////////////////////////

    assign per_core_ready_in[FPU_DIVSQRT] = is_div ? div_ready_in : sqrt_ready_in;

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW),
        .ARBITER    ("P"),
        .OUT_BUF    (0)
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

    wire [NUM_FPC-1:0][RSP_DATAW-1:0] per_core_data_out;

    for (genvar i = 0; i < NUM_FPC; ++i) begin
        assign per_core_data_out[i] = {per_core_result[i], per_core_has_fflags[i], per_core_fflags[i], per_core_tag_out[i]};
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_FPC),
        .DATAW      (RSP_DATAW),
        .ARBITER    ("R"),
        .OUT_BUF    (OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_core_valid_out),
        .ready_in  (per_core_ready_out),
        .data_in   (per_core_data_out),
        .data_out  ({result, has_fflags, fflags, tag_out}),
        .valid_out (valid_out),
        .ready_out (ready_out),
        `UNUSED_PIN (sel_out)
    );

    assign ready_in = per_core_ready_in[core_select];

endmodule
`endif

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

`ifdef FPU_TYPE_DSP

`ifdef SIMULATION
`include "dpi_float.vh"
`endif

module VX_fpu_fma import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter NUM_PES   = `UP(NUM_LANES / `FMA_PE_RATIO),
    parameter TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    output wire ready_in,
    input wire  valid_in,

    input wire [NUM_LANES-1:0] mask_in,

    input wire [TAG_WIDTH-1:0] tag_in,

    input wire [INST_FPU_BITS-1:0] op_type,
    input wire [INST_FMT_BITS-1:0] fmt,
    input wire [INST_FRM_BITS-1:0] frm,

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
    // Pack dataa, datab, datac, op_type, fmt, frm into each lane's data
    localparam CTRL_OFF = 3 * `XLEN;
    localparam DATAW    = CTRL_OFF + INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS;

    wire [NUM_LANES-1:0][DATAW-1:0] data_in;

    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
    wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0] pe_mask_out;
    `UNUSED_VAR (pe_mask_out)
    wire [NUM_PES-1:0][DATAW-1:0] pe_data_in;
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_data_in
        assign data_in[i][0              +: `XLEN]         = dataa[i];
        assign data_in[i][`XLEN          +: `XLEN]         = datab[i];
        assign data_in[i][2*`XLEN        +: `XLEN]         = datac[i];
        assign data_in[i][CTRL_OFF       +: INST_FPU_BITS]  = op_type;
        assign data_in[i][CTRL_OFF + INST_FPU_BITS
                                    +: INST_FMT_BITS]        = fmt;
        assign data_in[i][CTRL_OFF + INST_FPU_BITS + INST_FMT_BITS
                                    +: INST_FRM_BITS]        = frm;
    end

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (`LATENCY_FMA),
        .DATA_IN_WIDTH (DATAW),
        .DATA_OUT_WIDTH (`FP_FLAGS_BITS + `XLEN),
        .TAG_WIDTH  (TAG_WIDTH),
        .PE_REG     (0),
        .OUT_BUF    (2)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .data_in    (data_in),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .pe_enable  (pe_enable),
        .pe_mask_out(pe_mask_out),
        .pe_data_out(pe_data_in),
        .pe_data_in (pe_data_out),
        .valid_out  (valid_out),
        .mask_out   (mask_out),
        .data_out   (data_out),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

    `UNUSED_VAR (pe_data_in)

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
        assign result[i]     = data_out[i][0 +: `XLEN];
        assign fflags_out[i] = data_out[i][`XLEN +: `FP_FLAGS_BITS];
    end

    fflags_t [NUM_LANES-1:0] per_lane_fflags;

`ifdef QUARTUS

    // F32-only hardware path: derive sign-manipulation from op_type/fmt
    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fmas
        wire [INST_FPU_BITS-1:0] op_pe  = pe_data_in[0][CTRL_OFF +: INST_FPU_BITS];
        wire [INST_FMT_BITS-1:0] fmt_pe = pe_data_in[0][CTRL_OFF + INST_FPU_BITS +: INST_FMT_BITS];
        wire is_madd_pe = op_pe[1];
        wire is_neg_pe  = op_pe[0];
        wire is_sub_pe  = fmt_pe[1];

        reg [31:0] a32, b32, c32;
        always @(*) begin
            if (is_madd_pe) begin
                a32 = {is_neg_pe ^ pe_data_in[i][31],          pe_data_in[i][0 +: 31]};
                b32 = pe_data_in[i][`XLEN +: 32];
                c32 = {(is_neg_pe ^ is_sub_pe) ^ pe_data_in[i][2*`XLEN + 31],
                       pe_data_in[i][2*`XLEN +: 31]};
            end else begin
                if (is_neg_pe) begin // MUL
                    a32 = pe_data_in[i][0 +: 32];
                    b32 = pe_data_in[i][`XLEN +: 32];
                    c32 = '0;
                end else begin // ADD/SUB
                    a32 = pe_data_in[i][0 +: 32];
                    b32 = 32'h3f800000; // 1.0f
                    c32 = {is_sub_pe ^ pe_data_in[i][`XLEN + 31], pe_data_in[i][`XLEN +: 31]};
                end
            end
        end

        acl_fmadd fmadd (
            .clk (clk),
            .areset (1'b0),
            .en (pe_enable),
            .a  (a32),
            .b  (b32),
            .c  (c32),
            .q  (pe_data_out[i][0 +: 32])
        );
        assign pe_data_out[i][`XLEN +: `FP_FLAGS_BITS] = 'x;
    end

    assign has_fflags = 0;
    assign per_lane_fflags = 'x;

`elsif VIVADO

    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fmas
        wire [INST_FPU_BITS-1:0] op_pe  = pe_data_in[0][CTRL_OFF +: INST_FPU_BITS];
        wire [INST_FMT_BITS-1:0] fmt_pe = pe_data_in[0][CTRL_OFF + INST_FPU_BITS +: INST_FMT_BITS];
        wire is_madd_pe = op_pe[1];
        wire is_neg_pe  = op_pe[0];
        wire is_sub_pe  = fmt_pe[1];

        reg [31:0] a32, b32, c32;
        always @(*) begin
            if (is_madd_pe) begin
                a32 = {is_neg_pe ^ pe_data_in[i][31],          pe_data_in[i][0 +: 31]};
                b32 = pe_data_in[i][`XLEN +: 32];
                c32 = {(is_neg_pe ^ is_sub_pe) ^ pe_data_in[i][2*`XLEN + 31],
                       pe_data_in[i][2*`XLEN +: 31]};
            end else begin
                if (is_neg_pe) begin // MUL
                    a32 = pe_data_in[i][0 +: 32];
                    b32 = pe_data_in[i][`XLEN +: 32];
                    c32 = '0;
                end else begin // ADD/SUB
                    a32 = pe_data_in[i][0 +: 32];
                    b32 = 32'h3f800000; // 1.0f
                    c32 = {is_sub_pe ^ pe_data_in[i][`XLEN + 31], pe_data_in[i][`XLEN +: 31]};
                end
            end
        end

        wire [2:0] tuser;
        xil_fma fma (
            .aclk                (clk),
            .aclken              (pe_enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (a32),
            .s_axis_b_tvalid     (1'b1),
            .s_axis_b_tdata      (b32),
            .s_axis_c_tvalid     (1'b1),
            .s_axis_c_tdata      (c32),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (pe_data_out[i][0 +: 32]),
            .m_axis_result_tuser (tuser)
        );
                                                      // NV, DZ, OF, UF, NX
        assign pe_data_out[i][`XLEN +: `FP_FLAGS_BITS] = {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0};
    end

    assign has_fflags = 1;
    assign per_lane_fflags = fflags_out;

`else

    // Simulation path: call specific DPI functions based on op_type (avoids sign-bit manipulation)
    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fmas
        wire [INST_FPU_BITS-1:0] op_pe  = pe_data_in[0][CTRL_OFF +: INST_FPU_BITS];
        wire [INST_FMT_BITS-1:0] fmt_pe = pe_data_in[0][CTRL_OFF + INST_FPU_BITS +: INST_FMT_BITS];
        wire [INST_FRM_BITS-1:0] frm_pe = pe_data_in[0][CTRL_OFF + INST_FPU_BITS + INST_FMT_BITS +: INST_FRM_BITS];
        wire f_fmt_pe = fmt_pe[0];  // float format: 0=F32, 1=F64
        wire i_fmt_pe = fmt_pe[1];  // integer format (used as SUB flag for FMA ops)

        reg [63:0] r;
        `UNUSED_VAR (r)
        fflags_t f;

        always @(*) begin
            r = '0;
            f = '0;
            case (op_pe)
                INST_FPU_ADD: begin
                    if (i_fmt_pe) // FSUB
                        dpi_fsub (pe_enable, int'(f_fmt_pe),
                                  64'(pe_data_in[i][0       +: `XLEN]),
                                  64'(pe_data_in[i][`XLEN   +: `XLEN]),
                                  frm_pe, r, f);
                    else // FADD
                        dpi_fadd (pe_enable, int'(f_fmt_pe),
                                  64'(pe_data_in[i][0       +: `XLEN]),
                                  64'(pe_data_in[i][`XLEN   +: `XLEN]),
                                  frm_pe, r, f);
                end
                INST_FPU_MUL:
                    dpi_fmul (pe_enable, int'(f_fmt_pe),
                              64'(pe_data_in[i][0       +: `XLEN]),
                              64'(pe_data_in[i][`XLEN   +: `XLEN]),
                              frm_pe, r, f);
                INST_FPU_MADD: begin
                    if (i_fmt_pe) // FMSUB
                        dpi_fmsub (pe_enable, int'(f_fmt_pe),
                                   64'(pe_data_in[i][0       +: `XLEN]),
                                   64'(pe_data_in[i][`XLEN   +: `XLEN]),
                                   64'(pe_data_in[i][2*`XLEN +: `XLEN]),
                                   frm_pe, r, f);
                    else // FMADD
                        dpi_fmadd (pe_enable, int'(f_fmt_pe),
                                   64'(pe_data_in[i][0       +: `XLEN]),
                                   64'(pe_data_in[i][`XLEN   +: `XLEN]),
                                   64'(pe_data_in[i][2*`XLEN +: `XLEN]),
                                   frm_pe, r, f);
                end
                INST_FPU_NMADD: begin
                    if (i_fmt_pe) // FNMSUB
                        dpi_fnmsub (pe_enable, int'(f_fmt_pe),
                                    64'(pe_data_in[i][0       +: `XLEN]),
                                    64'(pe_data_in[i][`XLEN   +: `XLEN]),
                                    64'(pe_data_in[i][2*`XLEN +: `XLEN]),
                                    frm_pe, r, f);
                    else // FNMADD
                        dpi_fnmadd (pe_enable, int'(f_fmt_pe),
                                    64'(pe_data_in[i][0       +: `XLEN]),
                                    64'(pe_data_in[i][`XLEN   +: `XLEN]),
                                    64'(pe_data_in[i][2*`XLEN +: `XLEN]),
                                    frm_pe, r, f);
                end
                default:;
            endcase
        end

        VX_shift_register #(
            .DATAW  (`FP_FLAGS_BITS + `XLEN),
            .DEPTH  (`LATENCY_FMA)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (pe_enable),
            .data_in  ({f, r[`XLEN-1:0]}),
            .data_out (pe_data_out[i])
        );
    end

    assign has_fflags = 1;
    assign per_lane_fflags = fflags_out;

`endif

`FPU_MERGE_FFLAGS(fflags, per_lane_fflags, mask_out, NUM_LANES);

endmodule

`endif
